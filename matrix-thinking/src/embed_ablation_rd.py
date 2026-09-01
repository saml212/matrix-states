#!/usr/bin/env python3
"""embed_ablation_rd.py -- Parameter-matched matrix-vs-flat embedding ablation.

Pre-registration: matrix-thinking/EMBEDDING_ABLATION_DESIGN.md (read that
file first -- this script implements exactly the two arms and two matching
conditions it specifies, nothing more).

WHAT THIS TESTS (one sentence each, copied from the design doc so the two
files can never drift silently):
  Hypothesis: at equal total parameter count and equal depth (n_layers,
  n_iterations T), a matrix-native model (outer-product embedding + matrix
  ops) reaches lower T=1 token-BPB on a GPT-2-tokenized corpus than a
  flat-vector model (direct embedding table + standard transformer ops).
  Falsifier: the matrix arm does NOT beat the param-matched flat arm's T=1
  token-BPB by a margin exceeding the seed spread, on >=2/3 seeds, at BOTH
  model sizes.

TWO ARMS:
  matrix -- outer-product embedding (u,v -> u (x) v, rank-1 d x d matrix)
            + RowThenColProjection-based attention/multiplicative layers
            (silu(A @ M) @ B, A,B in R^{dxd} -- a Kronecker-RESTRICTED
            linear map on vec(M), never a full d^2 x d^2 map) + a matrix-
            native MultiProbeHead output.
  flat   -- direct (V, d_model) embedding table (no outer product, no rank
            restriction) + standard nn.MultiheadAttention/FFN blocks (FULL,
            unrestricted d_model x d_model linear maps) + a standard
            nn.Linear(d_model, V) output head.
  The flat arm is not a reshape of the matrix arm at any point: its
  embedding is not required to factor as an outer product, and its per-
  layer linear maps are drawn from the full linear group, not the
  Kronecker-restricted RowThenCol subspace the matrix arm uses. This is
  what CLAUDE.md's "structure only matters if OPERATIONS preserve it" rule
  requires closed off before any params-matched result can be trusted.

TWO MATCHING CONDITIONS (--match):
  P (params-matched, PRIMARY / VERDICT CARRIER) -- d_model is solved
     numerically (solve_matched_d_model) so that flat total params fall
     within +/-1% of the matrix arm's total params, AT THE SAME n_layers
     and the SAME n_iterations T as the matrix arm. Because GPT-2 vocab
     (50,257) dominates both arms' parameter budgets, this search is
     well-conditioned and always converges to <0.3% in practice (verified
     for both registered sizes in the design doc's table) -- unlike the
     historical d_model = mat_dim^2 "reshape parity" choice, which is what
     made Run 22 / Run 18 param-asymmetric by 2.2x-10x.
  D (depth-matched / params-UNMATCHED, secondary/disclosed control) --
     d_model is fixed at the "natural" value 2*mat_dim (so the flat
     embedding alone has the same free-parameter count per token as the
     matrix embedding's two (u,v) tables), at the SAME n_layers. This
     reproduces the historical Round-1/Run-22 shape of comparison (flat
     ends up with MORE total params, ratio disclosed, never gated) so the
     new result stays comparable to the old one.

Usage:
  python3 embed_ablation_rd.py --selftest
  python3 embed_ablation_rd.py --run-cell --arm matrix --size S --match P \\
      --seed 0 --steps 2000 --batch-size 64 --seq-len 512 \\
      --data-dir /data/deltanet_rd_data --corpus wikitext-mix-ext \\
      --ckpt-dir /data/embed_ablation_ckpts/matrix_S_s0 \\
      --out /home/nvidia/embed_ablation/results/embed_ablation_matrix_S_s0.json \\
      --ceiling-gpuh 2.0
  python3 embed_ablation_rd.py --harvest --results-dir <dir> --out <summary.json>
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ═══════════════════════════════════════════════════════════════════════
# Corpus constants (mirrors matrix-thinking/deltanet_rd/lm_pretrain_rd.py's
# CORPUS_DIRS / load_corpus contract exactly, reimplemented standalone here
# so this script has zero import dependency on lm_pretrain_rd.py -- that
# file is under concurrent edit elsewhere and pulls in frozen-bias/DeltaNet
# machinery this ablation does not need).
# ═══════════════════════════════════════════════════════════════════════

EOT_TOKEN_ID = 50256  # GPT-2 <|endoftext|>
GPT2_VOCAB_SIZE = 50257

CORPUS_DIRS = {
    "openr1": "reasoning_eot",
    "wikitext": "wikitext103_eot",
    "openr1-mix": "reasoning_mix_eot",
    "wikitext-mix": "wikitext103_mix_eot",
    "openr1-mix-ext": "reasoning_mix_eot_extended",
    "wikitext-mix-ext": "wikitext103_mix_eot_extended",
}


def load_corpus_tokens(data_dir: str, name: str, split: str) -> torch.Tensor:
    """Loads one {split}.pt (flat int64 GPT-2 token id tensor), asserting
    the same vocab/tokenizer/eot_separated contract lm_pretrain_rd.py's
    load_corpus asserts. split in {"train", "val"}."""
    assert name in CORPUS_DIRS, f"unknown corpus {name!r}, expected one of {sorted(CORPUS_DIRS)}"
    assert split in ("train", "val")
    d = os.path.join(data_dir, CORPUS_DIRS[name])
    assert os.path.isdir(d), f"{name}: corpus dir {d} not found"
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    assert meta.get("vocab_size") == GPT2_VOCAB_SIZE and meta.get("tokenizer") == "gpt2", (
        f"{name}: meta.json fields {meta} do not match the expected GPT-2 (vocab_size=50257) "
        f"tokenization this ablation assumes."
    )
    assert meta.get("eot_separated") is True, f"{name}: meta.json lacks eot_separated=true"
    toks = torch.load(os.path.join(d, f"{split}.pt"), map_location="cpu")
    assert toks.dtype == torch.int64
    return toks, meta


def get_batch(tokens: torch.Tensor, batch_size: int, seq_len: int, generator: torch.Generator):
    """Same random-contiguous-window sampling as lm_pretrain_rd.get_batch."""
    n = tokens.numel()
    assert n > seq_len + 1, f"corpus too small ({n} tokens) for seq_len={seq_len}"
    ix = torch.randint(0, n - seq_len - 1, (batch_size,), generator=generator)
    offs = torch.arange(seq_len + 1)
    idx = ix.unsqueeze(1) + offs.unsqueeze(0)
    window = tokens[idx]
    x, y = window[:, :-1].contiguous(), window[:, 1:].contiguous()
    return x, y


# ═══════════════════════════════════════════════════════════════════════
# MATRIX ARM -- outer-product embedding + RowThenCol matrix ops.
# Copied verbatim (module-for-module) from experiment-runs/8xh100-session1/
# round2_matrix_script.py's MatrixThinker so the architecture is IDENTICAL
# to the one the finding-01/finding-04 claims were made about -- this
# ablation changes the comparison's rigor (param matching, seeds, metric),
# not the model being defended.
# ═══════════════════════════════════════════════════════════════════════

class MatrixRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, d))
        self.eps = eps

    def forward(self, M):
        rms = torch.sqrt(M.pow(2).mean(dim=(-2, -1), keepdim=True) + self.eps)
        return M / rms * self.weight


class RowThenColProjection(nn.Module):
    """silu(A @ M) @ B, A,B in R^{dxd} -- 2*d^2 params.

    The induced linear map on vec(M) is (B^T (x) A), a Kronecker-restricted
    subspace of the full d^2 x d^2 linear group -- NOT every d x d -> d x d
    linear map is expressible this way. This restriction is exactly what
    finding 04 (parameter-efficiency.html) calls "RowThenCol bilinear":
    2*d^2 params vs d^4 for the unrestricted flattened Linear.
    """
    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
        self.B = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))

    def forward(self, M):
        return torch.einsum('...ij,jk->...ik',
                             F.silu(torch.einsum('ij,...jk->...ik', self.A, M)), self.B)


class MatrixFrobeniusAttention(nn.Module):
    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        assert d % n_heads == 0, f"mat_dim={d} must be divisible by n_heads={n_heads}"
        self.d, self.n_heads, self.head_dim = d, n_heads, d // n_heads
        self.norm = MatrixRMSNorm(d)
        self.q_proj = RowThenColProjection(d)
        self.k_proj = RowThenColProjection(d)
        self.v_proj = RowThenColProjection(d)
        self.o_proj = RowThenColProjection(d)
        self.dropout_p = dropout

    def forward(self, M):
        B, L, d, _ = M.shape
        H, hd = self.n_heads, self.head_dim
        M_n = self.norm(M)
        Q = self.q_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4).reshape(B, H, L, hd * d)
        K = self.k_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4).reshape(B, H, L, hd * d)
        V = self.v_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4).reshape(B, H, L, hd * d)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True,
                                              dropout_p=self.dropout_p if self.training else 0.0)
        out = out.reshape(B, H, L, hd, d).permute(0, 2, 1, 3, 4).reshape(B, L, d, d)
        return M + self.o_proj(out)


class MatrixMultiplicativeLayer(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.norm = MatrixRMSNorm(d)
        self.delta_gate = RowThenColProjection(d)
        self.delta_value = RowThenColProjection(d)
        self.delta_up = RowThenColProjection(d)
        self.gamma_gate = RowThenColProjection(d)
        self.gamma_value = RowThenColProjection(d)
        self.gamma_up = RowThenColProjection(d)
        self.key_col = nn.Parameter(torch.randn(d, 1) * 0.02)
        self.val_col = nn.Parameter(torch.randn(d, 1) * 0.02)
        self.gate_mult_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_mult_bias = nn.Parameter(torch.tensor(-2.0))
        self.gate_write_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_write_bias = nn.Parameter(torch.tensor(-2.0))
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('I', torch.eye(d))

    def forward(self, M):
        M_n = self.norm(M)
        s = self.scale.clamp(0.01, 0.5)
        delta = self.delta_up(F.silu(self.delta_gate(M_n)) * self.delta_value(M_n)) * s
        gamma = self.gamma_up(F.silu(self.gamma_gate(M_n)) * self.gamma_value(M_n)) * s
        M_mult = torch.matmul(torch.matmul(self.I + delta, M_n), self.I + gamma)
        k = torch.matmul(M_n, self.key_col).squeeze(-1)
        v = torch.matmul(M_n, self.val_col).squeeze(-1)
        M_write = torch.einsum('...i,...j->...ij', v, k)
        g_m = torch.sigmoid((self.gate_mult_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_mult_bias)
        g_w = torch.sigmoid((self.gate_write_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_write_bias)
        return M + self.dropout(g_m * (M_mult - M_n) + g_w * M_write)


class MatrixThinkingBlock(nn.Module):
    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = MatrixFrobeniusAttention(d, n_heads, dropout)
        self.think = MatrixMultiplicativeLayer(d, dropout)

    def forward(self, M):
        return self.think(self.attn(M))


class MultiProbeHead(nn.Module):
    """K bilinear probes -> Linear -> vocab logits. Never flattens M.
    Params: 2*K*d + K*V. Source: matrix-thinking/src/matrix_output_heads.py.
    """
    def __init__(self, d, vocab_size, n_probes=None):
        super().__init__()
        K = n_probes or d
        self.U = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))
        self.V = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))
        self.out = nn.Linear(K, vocab_size, bias=False)

    def forward(self, M):
        MV = torch.einsum('blij, kj -> blik', M, self.V)
        probes = torch.einsum('ki, blik -> blk', self.U, MV)
        return self.out(probes)


class MatrixThinker(nn.Module):
    """Outer-product embedding + shared ThinkingBlock stack applied
    n_iterations times (weight-shared iterative refinement -- T does NOT
    change parameter count, only n_layers does)."""
    def __init__(self, mat_dim=16, n_layers=6, n_heads=4, max_len=512,
                 vocab_size=GPT2_VOCAB_SIZE, dropout=0.1):
        super().__init__()
        self.mat_dim = mat_dim
        d = mat_dim
        self.embed_u = nn.Embedding(vocab_size, d)
        self.embed_v = nn.Embedding(vocab_size, d)
        self.pos_u = nn.Embedding(max_len, d)
        self.pos_v = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([MatrixThinkingBlock(d, n_heads, dropout) for _ in range(n_layers)])
        self.final_norm = MatrixRMSNorm(d)
        self.output_head = MultiProbeHead(d, vocab_size, n_probes=d)

    def _one_iteration(self, M):
        for layer in self.layers:
            if self.training:
                M = torch.utils.checkpoint.checkpoint(layer, M, use_reentrant=False)
            else:
                M = layer(M)
        return M

    def forward(self, token_ids, n_iterations=1):
        B, L = token_ids.shape
        u, v = self.embed_u(token_ids), self.embed_v(token_ids)
        M = torch.einsum('...i,...j->...ij', u, v)
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        pu, pv = self.pos_u(pos), self.pos_v(pos)
        M = M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1
        for _ in range(n_iterations):
            M = self._one_iteration(M)
        M_n = self.final_norm(M)
        return self.output_head(M_n)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self):
        embed = sum(p.numel() for n, p in self.named_parameters() if n.startswith(('embed_', 'pos_')))
        backbone = sum(p.numel() for n, p in self.named_parameters() if n.startswith('layers'))
        head = sum(p.numel() for n, p in self.named_parameters() if n.startswith(('output_head', 'final_norm')))
        return {"embed": embed, "backbone": backbone, "head": head, "total": self.count_params()}


# ═══════════════════════════════════════════════════════════════════════
# FLAT ARM -- direct (V, d_model) embedding table + standard transformer.
# Copied verbatim from round1_vector_script.py's VectorThinker. This is
# NOT a reshape of the matrix arm: the embedding is a plain lookup table
# (no outer product, no rank restriction) and the attention/FFN layers are
# full unrestricted nn.Linear/nn.MultiheadAttention maps, never RowThenCol.
# ═══════════════════════════════════════════════════════════════════════

class VectorThinkingBlock(nn.Module):
    """Standard pre-norm transformer block: FULL d_model x d_model linear
    maps throughout (not Kronecker-restricted)."""
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, h):
        h_n = self.norm1(h)
        L = h.shape[1]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(L, device=h.device)
        attn_out, _ = self.attn(h_n, h_n, h_n, attn_mask=mask)
        h = h + attn_out
        h = h + self.ffn(self.norm2(h))
        return h


class VectorThinker(nn.Module):
    """Direct flat-vector embedding + shared VectorThinkingBlock stack
    applied n_iterations times (same weight-sharing convention as
    MatrixThinker, so T never changes either arm's parameter count)."""
    def __init__(self, d_model=24, n_layers=6, n_heads=4, max_len=512,
                 vocab_size=GPT2_VOCAB_SIZE, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([VectorThinkingBlock(d_model, n_heads, dropout)
                                      for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def _one_iteration(self, h):
        for layer in self.layers:
            if self.training:
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        return h

    def forward(self, token_ids, n_iterations=1):
        B, L = token_ids.shape
        h = self.embed(token_ids) + self.pos(torch.arange(L, device=token_ids.device)).unsqueeze(0)
        for _ in range(n_iterations):
            h = self._one_iteration(h)
        h = self.norm(h)
        return self.output(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self):
        embed = sum(p.numel() for n, p in self.named_parameters() if n.startswith(('embed', 'pos')))
        backbone = sum(p.numel() for n, p in self.named_parameters() if n.startswith('layers'))
        head = sum(p.numel() for n, p in self.named_parameters() if n.startswith(('output', 'norm')))
        return {"embed": embed, "backbone": backbone, "head": head, "total": self.count_params()}


# ═══════════════════════════════════════════════════════════════════════
# Param-count solving and the params-matched gate.
# ═══════════════════════════════════════════════════════════════════════

# Registered sizes (mirrors EMBEDDING_ABLATION_DESIGN.md's table exactly).
SIZE_CONFIGS = {
    "S": dict(mat_dim=16, n_layers=6, n_heads=4),
    "M": dict(mat_dim=24, n_layers=8, n_heads=4),
}
N_ITERATIONS = 8          # T for the "matrix model's iterative T" eval leg
EVAL_ITERATIONS = (1, 8)  # T=1 (headline) and T=N_ITERATIONS, BOTH reported
MAX_LEN = 512
PARAM_MATCH_TOL = 0.01    # +/-1%, per the design doc's requirement


def flat_params_analytic(d_model, n_layers, vocab_size=GPT2_VOCAB_SIZE, max_len=MAX_LEN):
    """Closed-form flat-arm param count -- used ONLY to drive the search
    below. The authoritative count always comes from instantiating the
    real nn.Module (see verify step in solve_matched_d_model)."""
    embed = vocab_size * d_model
    pos = max_len * d_model
    mha = 4 * d_model * d_model + 4 * d_model
    ln = 2 * (2 * d_model)
    ffn = (d_model * 4 * d_model + 4 * d_model) + (4 * d_model * d_model + d_model)
    per_layer = mha + ln + ffn
    final_ln = 2 * d_model
    head = d_model * vocab_size
    return embed + pos + n_layers * per_layer + final_ln + head


def solve_matched_d_model(target_params, n_layers, n_heads, vocab_size=GPT2_VOCAB_SIZE,
                           max_len=MAX_LEN, lo=4, hi=8192):
    """Finds the d_model (multiple of n_heads) whose FLAT arm total params
    is closest to target_params, then verifies by REAL instantiation
    (never trusts the analytic formula alone for the final answer)."""
    candidates = [c for c in range(lo, hi) if c % n_heads == 0]
    best = None
    for c in candidates:
        t = flat_params_analytic(c, n_layers, vocab_size, max_len)
        if best is None or abs(t - target_params) < abs(best[1] - target_params):
            best = (c, t)
        if t > target_params * 2.0 and c > lo + n_heads:
            break
    d_model = best[0]
    real = VectorThinker(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                          max_len=max_len, vocab_size=vocab_size).count_params()
    return d_model, real


def check_param_match(matrix_total, flat_total, tol=PARAM_MATCH_TOL, label=""):
    """The negative-test gate: raises loudly if the two arms are not
    within `tol` of each other. Called for --match P cells; --match D
    cells call this in report-only mode (raise=False) since they are
    PRE-REGISTERED as unmatched."""
    ratio = flat_total / matrix_total
    diff = abs(ratio - 1.0)
    ok = diff <= tol
    msg = (f"{label} param match: matrix={matrix_total:,} flat={flat_total:,} "
           f"ratio={ratio:.4f} diff={diff*100:.3f}% tol={tol*100:.1f}% -> "
           f"{'PASS' if ok else 'FAIL'}")
    return ok, ratio, msg


def build_arm(arm, size, match, dropout=0.1, vocab_size=GPT2_VOCAB_SIZE, max_len=MAX_LEN):
    """Builds one model + returns (model, meta_dict). meta_dict always
    carries params_total and, for the flat arm, the params_ratio vs the
    matrix arm at the same size (computed by building the matrix twin,
    which is cheap -- a few million params, CPU-instantiable in <1s)."""
    cfg = SIZE_CONFIGS[size]
    mat_dim, n_layers, n_heads = cfg["mat_dim"], cfg["n_layers"], cfg["n_heads"]

    matrix_ref = MatrixThinker(mat_dim=mat_dim, n_layers=n_layers, n_heads=n_heads,
                                max_len=max_len, vocab_size=vocab_size, dropout=dropout)
    matrix_total = matrix_ref.count_params()

    if arm == "matrix":
        model = matrix_ref
        meta = {"arm": "matrix", "size": size, "match": match, "mat_dim": mat_dim,
                "n_layers": n_layers, "n_heads": n_heads, "params_total": matrix_total,
                "params_breakdown": matrix_ref.param_breakdown()}
        return model, meta

    assert arm == "flat"
    del matrix_ref  # only needed for its param count above
    if match == "P":
        d_model, flat_total = solve_matched_d_model(matrix_total, n_layers, n_heads,
                                                      vocab_size, max_len)
        ok, ratio, msg = check_param_match(matrix_total, flat_total, PARAM_MATCH_TOL,
                                            label=f"flat-P {size}")
        if not ok:
            raise RuntimeError(
                f"PARAMS-MATCHED (--match P) GATE FAILED for size={size}: {msg}. "
                f"This cell must NOT be trusted as the params-matched arm -- either widen "
                f"the d_model search range in solve_matched_d_model or re-register this "
                f"size's config in EMBEDDING_ABLATION_DESIGN.md before running it."
            )
    elif match == "D":
        d_model = 2 * mat_dim  # natural: matches the (u,v) free-param count per token
        flat_total = flat_params_analytic(d_model, n_layers, vocab_size, max_len)
        ok, ratio, msg = check_param_match(matrix_total, flat_total, PARAM_MATCH_TOL,
                                            label=f"flat-D {size}")
        # D is PRE-REGISTERED as params-UNMATCHED -- do not gate, only disclose.
    else:
        raise ValueError(f"unknown match condition {match!r}, expected 'P' or 'D'")

    model = VectorThinker(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           max_len=max_len, vocab_size=vocab_size, dropout=dropout)
    real_total = model.count_params()
    meta = {"arm": "flat", "size": size, "match": match, "d_model": d_model,
            "n_layers": n_layers, "n_heads": n_heads, "params_total": real_total,
            "params_breakdown": model.param_breakdown(),
            "matrix_twin_params": matrix_total,
            "params_ratio_vs_matrix": real_total / matrix_total}
    return model, meta


# ═══════════════════════════════════════════════════════════════════════
# Eval / training
# ═══════════════════════════════════════════════════════════════════════

def evaluate(model, val_tokens, vocab_size, device, n_iterations, seq_len,
             eval_batch_size, max_eval_batches, generator):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for _ in range(max_eval_batches):
            x, y = get_batch(val_tokens, eval_batch_size, seq_len, generator)
            x, y = x.to(device), y.to(device)
            logits = model(x, n_iterations=n_iterations)
            n_tok = y.numel()
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item() * n_tok
            total_tokens += n_tok
    model.train()
    avg_loss_nats = total_loss / max(total_tokens, 1)
    return {
        "val_loss_nats": avg_loss_nats,
        "token_bpb": avg_loss_nats / math.log(2),   # bits per TOKEN, not per byte -- see design doc S3
        "ppl": math.exp(min(avg_loss_nats, 20)),
    }


def run_cell(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model, meta = build_arm(args.arm, args.size, args.match, vocab_size=GPT2_VOCAB_SIZE,
                             max_len=args.seq_len)
    model = model.to(device)
    meta.update({"seed": args.seed, "steps_target": args.steps, "batch_size": args.batch_size,
                 "seq_len": args.seq_len, "corpus": args.corpus, "n_iterations_train": N_ITERATIONS,
                 "eval_iterations": list(EVAL_ITERATIONS)})

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

    train_tokens, train_meta = load_corpus_tokens(args.data_dir, args.corpus, "train")
    val_tokens, _ = load_corpus_tokens(args.data_dir, args.corpus, "val")
    gen = torch.Generator().manual_seed(args.seed)
    val_gen = torch.Generator().manual_seed(args.seed + 10_000)  # independent val stream

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warmup = max(1, args.steps // 10)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(args.steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start = time.time()
    ceiling_s = args.ceiling_gpuh * 3600.0
    training_curve = []
    step = 0
    status = "COMPLETED"
    model.train()
    while step < args.steps:
        if time.time() - start > ceiling_s:
            status = "CEILING_STOP"
            break
        x, y = get_batch(train_tokens, args.batch_size, args.seq_len, gen)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        autocast_ctx = torch.autocast(device.type, dtype=torch.bfloat16) if device.type == "cuda" \
            else torch.autocast("cpu", enabled=False)
        with autocast_ctx:
            logits = model(x, n_iterations=N_ITERATIONS)
            loss = F.cross_entropy(logits.reshape(-1, GPT2_VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % args.eval_interval == 0 or step == args.steps:
            evals = {}
            for T in EVAL_ITERATIONS:
                evals[f"T{T}"] = evaluate(model, val_tokens, GPT2_VOCAB_SIZE, device, T,
                                           args.seq_len, args.eval_batch_size,
                                           args.eval_batches, val_gen)
            training_curve.append({"step": step, "train_loss_nats": loss.item(), "evals": evals})
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "last.pt"))

    elapsed_min = (time.time() - start) / 60.0
    final_evals = training_curve[-1]["evals"] if training_curve else {}

    result = {
        "experiment": "embed_ablation_rd",
        **meta,
        "status": status,
        "complete": status == "COMPLETED",
        "steps_completed": step,
        "time_min": elapsed_min,
        "gpu_h_actual_approx": elapsed_min / 60.0,
        "final_evals": final_evals,
        "training_curve": training_curve,
        "corpus_train_meta": {k: v for k, v in train_meta.items() if k != "original_recipe_verification"},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in result.items() if k != "training_curve"}, indent=2, default=float))
    return result


# ═══════════════════════════════════════════════════════════════════════
# Harvest: aggregate cells -> pre-registered decision rule
# ═══════════════════════════════════════════════════════════════════════

def harvest(results_dir, out_path):
    results = []
    for fn in sorted(os.listdir(results_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fn)) as f:
            results.append(json.load(f))

    by_key = {}
    for r in results:
        if r.get("arm") != "matrix" and r.get("match") != "P":
            continue  # decision rule only uses matrix vs flat-P
        key = (r["size"], r["arm"])
        by_key.setdefault(key, []).append(r)

    verdicts = {}
    for size in SIZE_CONFIGS:
        matrix_runs = by_key.get((size, "matrix"), [])
        flat_runs = by_key.get((size, "flat"), [])
        if not matrix_runs or not flat_runs:
            verdicts[size] = {"status": "INCOMPLETE", "n_matrix": len(matrix_runs), "n_flat": len(flat_runs)}
            continue

        def t1_bpb(r):
            return r.get("final_evals", {}).get("T1", {}).get("token_bpb")

        m_vals = sorted(v for v in (t1_bpb(r) for r in matrix_runs) if v is not None)
        f_vals = sorted(v for v in (t1_bpb(r) for r in flat_runs) if v is not None)
        n_seed_pairs = min(len(m_vals), len(f_vals))
        m_spread = (max(m_vals) - min(m_vals)) if len(m_vals) > 1 else 0.0
        f_spread = (max(f_vals) - min(f_vals)) if len(f_vals) > 1 else 0.0
        seed_spread = max(m_spread, f_spread)

        wins = sum(1 for mv, fv in zip(m_vals, f_vals) if (fv - mv) > seed_spread)
        verdicts[size] = {
            "status": "SCORED",
            "matrix_t1_bpb": m_vals,
            "flat_t1_bpb": f_vals,
            "seed_spread": seed_spread,
            "wins": wins,
            "n_pairs": n_seed_pairs,
            "size_pass": wins >= 2 and n_seed_pairs >= 3,
        }

    scored = [v for v in verdicts.values() if v.get("status") == "SCORED"]
    all_sizes_scored = len(scored) == len(SIZE_CONFIGS)
    overall = "PENDING"
    if all_sizes_scored:
        overall = "STRENGTHEN" if all(v["size_pass"] for v in scored) else "DROP"

    summary = {"per_size": verdicts, "decision": overall,
               "rule": "STRENGTHEN iff matrix beats flat-P T=1 token_bpb on >=2/3 seeds "
                       "by more than the seed spread, at BOTH sizes. Else DROP. No third outcome.",
               "n_results_loaded": len(results)}
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    return summary


# ═══════════════════════════════════════════════════════════════════════
# Self-test (CPU-only, tiny configs): forward/backward/grad check for BOTH
# arms + the param-count negative test. Run before ANY box deployment.
# ═══════════════════════════════════════════════════════════════════════

def _grad_check(model, x, y, vocab_size, label):
    model.train()
    logits = model(x, n_iterations=2)
    assert logits.shape == (x.shape[0], x.shape[1], vocab_size), \
        f"{label}: bad output shape {logits.shape}"
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    assert torch.isfinite(loss), f"{label}: non-finite loss {loss.item()}"
    model.zero_grad()
    loss.backward()
    n_params, n_with_grad, n_nonzero_grad = 0, 0, 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n_params += 1
        if p.grad is not None:
            n_with_grad += 1
            if p.grad.abs().sum().item() > 0:
                n_nonzero_grad += 1
            assert torch.isfinite(p.grad).all(), f"{label}: non-finite grad in {name}"
    assert n_with_grad == n_params, f"{label}: {n_params - n_with_grad} params got NO gradient at all"
    frac_nonzero = n_nonzero_grad / max(n_params, 1)
    print(f"  [{label}] forward OK shape={tuple(logits.shape)} loss={loss.item():.4f} "
          f"grads: {n_with_grad}/{n_params} present, {n_nonzero_grad}/{n_params} nonzero "
          f"({frac_nonzero*100:.0f}%)")
    assert frac_nonzero > 0.5, f"{label}: suspiciously few params received a nonzero gradient"
    return loss.item()


def selftest():
    torch.manual_seed(0)
    print("=" * 74)
    print("EMBED ABLATION SELFTEST (CPU, tiny configs)")
    print("=" * 74)
    failures = []

    # ---- 1. Tiny forward/backward/grad check, both arms, both configs ----
    tiny_vocab = 97
    B, L = 3, 11
    for size, cfg in [("tiny-S", dict(mat_dim=8, n_layers=2, n_heads=2)),
                       ("tiny-M", dict(mat_dim=12, n_layers=2, n_heads=4))]:
        print(f"\n--- size {size}: {cfg} ---")
        x = torch.randint(0, tiny_vocab, (B, L))
        y = torch.randint(0, tiny_vocab, (B, L))
        try:
            mm = MatrixThinker(mat_dim=cfg["mat_dim"], n_layers=cfg["n_layers"],
                                n_heads=cfg["n_heads"], max_len=L, vocab_size=tiny_vocab)
            _grad_check(mm, x, y, tiny_vocab, f"matrix/{size}")
        except Exception as e:
            failures.append(f"matrix/{size}: {e}")
            print(f"  FAIL matrix/{size}: {e}")

        try:
            # Solve for a matched d_model at this TOY scale too, purely to
            # exercise the flat arm's forward/backward/grad path -- NOT a
            # test of matching precision. At tiny_vocab=97 the integer
            # d_model search has very coarse granularity (a handful of
            # candidates total), so the +/-1% gate is not expected to hold
            # here; the real precision claim is checked below, at the
            # actual registered GPT-2-vocab (50,257) sizes, where the
            # search space is large enough to converge tightly. So this
            # sub-check is INFORMATIONAL only and never added to `failures`.
            matrix_total = mm.count_params()
            d_model, flat_total = solve_matched_d_model(matrix_total, cfg["n_layers"],
                                                          cfg["n_heads"], vocab_size=tiny_vocab,
                                                          max_len=L, lo=cfg["n_heads"], hi=512)
            ok, ratio, msg = check_param_match(matrix_total, flat_total, PARAM_MATCH_TOL,
                                                label=f"flat-P/{size}")
            print(f"  {msg}  [informational at toy vocab={tiny_vocab}; real gate is the "
                  f"'registered sizes at real GPT-2 vocab' section below]")
            vm = VectorThinker(d_model=d_model, n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                                max_len=L, vocab_size=tiny_vocab)
            _grad_check(vm, x, y, tiny_vocab, f"flat-P/{size} (d_model={d_model})")
        except Exception as e:
            failures.append(f"flat-P/{size}: {e}")
            print(f"  FAIL flat-P/{size}: {e}")

    # ---- 2. Negative test: a deliberately mismatched flat model must FAIL
    #         the param-match gate loudly (not silently pass). ----
    print("\n--- negative test: forced param mismatch must raise ---")
    mm = MatrixThinker(mat_dim=16, n_layers=6, n_heads=4, max_len=64, vocab_size=1000)
    matrix_total = mm.count_params()
    bogus_flat_total = flat_params_analytic(d_model=4, n_layers=6, vocab_size=1000, max_len=64)
    ok, ratio, msg = check_param_match(matrix_total, bogus_flat_total, PARAM_MATCH_TOL, label="NEGATIVE")
    print(f"  {msg}")
    if ok:
        failures.append("NEGATIVE TEST FAILED TO FAIL: a >1% mismatch was reported as PASS")
    else:
        print("  negative test correctly reports FAIL for the mismatched pair (as expected)")

    raised = False
    try:
        d_model_bad = 4  # far too small; ratio will be << 1 - tol
        vm_bad = VectorThinker(d_model=d_model_bad, n_layers=6, n_heads=4, max_len=64, vocab_size=1000)
        ok2, ratio2, msg2 = check_param_match(matrix_total, vm_bad.count_params(), PARAM_MATCH_TOL,
                                               label="NEGATIVE-build_arm-path")
        if not ok2:
            raise RuntimeError(f"simulated build_arm(match='P') gate: {msg2}")
    except RuntimeError as e:
        raised = True
        print(f"  build_arm-style gate correctly RAISED: {e}")
    if not raised:
        failures.append("build_arm-style param-match gate did not raise on a forced mismatch")

    # ---- 3. Registered sizes (S, M) actually solve within tolerance at
    #         GPT-2 vocab scale -- the real config the box will run. ----
    print("\n--- registered sizes at real GPT-2 vocab (50257), match=P ---")
    for size, cfg in SIZE_CONFIGS.items():
        mm = MatrixThinker(mat_dim=cfg["mat_dim"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                            max_len=MAX_LEN, vocab_size=GPT2_VOCAB_SIZE)
        matrix_total = mm.count_params()
        d_model, flat_total = solve_matched_d_model(matrix_total, cfg["n_layers"], cfg["n_heads"])
        ok, ratio, msg = check_param_match(matrix_total, flat_total, PARAM_MATCH_TOL, label=f"size {size}")
        print(f"  {msg}  (d_model={d_model})")
        if not ok:
            failures.append(f"registered size {size} failed params-matched gate: {msg}")

        d_model_d = 2 * cfg["mat_dim"]
        flat_d_total = flat_params_analytic(d_model_d, cfg["n_layers"])
        okd, ratiod, msgd = check_param_match(matrix_total, flat_d_total, PARAM_MATCH_TOL,
                                               label=f"size {size} (match=D, expected UNMATCHED)")
        print(f"  {msgd}  (d_model={d_model_d}) -- D is pre-registered as unmatched, this is informational")

    print("\n" + "=" * 74)
    if failures:
        print(f"SELFTEST: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        print("=" * 74)
        return 1
    print("SELFTEST: ALL CHECKS PASSED")
    print("=" * 74)
    return 0


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--run-cell", action="store_true")
    p.add_argument("--harvest", action="store_true")

    p.add_argument("--arm", choices=["matrix", "flat"])
    p.add_argument("--size", choices=list(SIZE_CONFIGS))
    p.add_argument("--match", choices=["P", "D"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--data-dir", type=str, default="/data/deltanet_rd_data")
    p.add_argument("--corpus", type=str, default="wikitext-mix-ext")
    p.add_argument("--ckpt-dir", type=str)
    p.add_argument("--out", type=str)
    p.add_argument("--ceiling-gpuh", type=float, default=2.0,
                   help="hard wall-clock cutoff in GPU-hours; writes a CEILING_STOP result "
                        "rather than running over budget")
    p.add_argument("--cpu", action="store_true", help="force CPU (selftest/debug only)")

    p.add_argument("--results-dir", type=str, help="--harvest: dir of per-cell result JSONs")

    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())
    elif args.run_cell:
        for req in ("arm", "size", "match", "ckpt_dir", "out"):
            if getattr(args, req.replace("-", "_")) is None:
                p.error(f"--run-cell requires --{req.replace('_', '-')}")
        run_cell(args)
    elif args.harvest:
        if not args.results_dir or not args.out:
            p.error("--harvest requires --results-dir and --out")
        harvest(args.results_dir, args.out)
    else:
        p.error("specify one of --selftest, --run-cell, --harvest")


if __name__ == "__main__":
    main()
