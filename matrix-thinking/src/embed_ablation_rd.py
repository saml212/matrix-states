#!/usr/bin/env python3
"""embed_ablation_rd.py -- Parameter-matched matrix-vs-flat-vs-flatten
embedding ablation. REV-narrow (post-audit).

Pre-registration: matrix-thinking/EMBEDDING_ABLATION_DESIGN.md (read that
file first -- this script implements exactly the three arms and two
matching conditions it specifies, nothing more).

WHAT THIS TESTS -- TWO SEPARATE, INDEPENDENTLY-DECIDED CLAIMS (audit M3):
  STRENGTHEN-01 (embedding mechanism, matrix vs flatten): at equal total
    params and equal depth, does keeping the outer-product embedding
    AND matrix-native downstream ops (the full matrix arm) beat keeping
    the SAME outer-product embedding but flattening it into a standard
    dense backbone (the flatten arm)? This isolates the operation family
    while holding the embedding mechanism fixed -- Run 18's own historical
    recipe, now genuinely params-matched.
  STRENGTHEN-04 (architecture at equal params, matrix vs flat-P): at equal
    total params and equal depth, does the full matrix arm (outer-product
    embedding + matrix ops) beat a fully flat arm (direct embedding table
    + standard ops) -- varying BOTH embedding and operations at once?
  Each claim: STRENGTHEN iff the matrix arm beats its comparison arm's T=1
  token-BPB by a margin exceeding the seed spread, on >=2/3 seeds, at BOTH
  registered sizes. DROP otherwise. No third outcome per claim.

THREE ARMS:
  matrix  -- outer-product embedding (u,v -> u⊗v, rank-1 d×d matrix) +
             RowThenColProjection-based ops (silu(A@M)@B, Kronecker-
             RESTRICTED) + matrix-native MultiProbeHead output.
  flat    -- direct (V, d_model) embedding table (no outer product, no
             rank restriction) + standard nn.MultiheadAttention/FFN
             (FULL, unrestricted d_model×d_model linear maps) + standard
             nn.Linear output head. `--match P` (params-matched, PRIMARY)
             solves d_model to +/-1% of the matrix arm's total params at
             equal n_layers; `--match D` (disclosed, UNMATCHED control)
             fixes d_model=2*mat_dim (the historical Round-1/Run-22 shape
             of comparison), never gates a decision.
  flatten -- the SAME outer-product embedding as the matrix arm, flattened
             to a d^2 vector, resized via one Linear(d^2, d_model) into
             the SAME dense VectorThinkingBlock backbone the flat arm
             uses. d_model (and its own n_heads, searched independently --
             see solve_matched_d_model_flatten) is solved to +/-1% of the
             matrix arm's total params. Only `--match P` is registered.

Neither `flat` nor `flatten` is a reshape of the `matrix` arm: `flat`'s
embedding is never required to factor as an outer product and its linear
maps are drawn from the full linear group, never the Kronecker-restricted
RowThenCol subspace; `flatten` shares the matrix arm's embedding exactly
but its POST-embedding operations are the same full/unrestricted family
`flat` uses, never RowThenCol. CLAUDE.md: "structure only matters if
OPERATIONS preserve it."

Usage:
  python3 embed_ablation_rd.py --selftest
  python3 embed_ablation_rd.py --harvest-selftest
  python3 embed_ablation_rd.py --check-admission --probe-results-dir <dir> \\
      --intended-steps 2000 --intended-batch 64
  python3 embed_ablation_rd.py --run-cell --arm matrix --size S --match P \\
      --seed 0 --steps 2000 --batch-size 64 --seq-len 512 \\
      --data-dir /data/deltanet_rd_data --corpus wikitext-mix-ext \\
      --ckpt-dir /data/embed_ablation_ckpts/matrix_S_s0 \\
      --out /home/nvidia/embed_ablation/results/matrix_S_s0.json \\
      --ceiling-gpuh 2.0 --role cell
  python3 embed_ablation_rd.py --harvest --results-dir <dir> --out <summary.json>
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
import zlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ═══════════════════════════════════════════════════════════════════════
# Corpus constants (mirrors matrix-thinking/deltanet_rd/lm_pretrain_rd.py's
# CORPUS_DIRS / load_corpus / corpus_fixed_seed contract exactly,
# reimplemented standalone here so this script has zero import dependency
# on lm_pretrain_rd.py -- that file is under concurrent edit elsewhere and
# pulls in frozen-bias/DeltaNet machinery this ablation does not need).
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


def corpus_fixed_seed(corpus_name: str) -> int:
    """AUDIT FIX-1 pattern, ported from lm_pretrain_rd.py's
    `corpus_fixed_seed` (that file's own comment: "NEVER from the training
    seed"). Used to seed the EVAL generator so every seed/arm/T-leg in
    this ablation scores identical validation windows in identical order
    -- see `evaluate()`'s M1 fix below."""
    return zlib.crc32(corpus_name.encode("utf-8"))


def load_corpus_tokens(data_dir: str, name: str, split: str) -> torch.Tensor:
    """Loads one {split}.pt (flat int64 GPT-2 token id tensor), asserting
    the same vocab/tokenizer/eot_separated contract lm_pretrain_rd.py's
    load_corpus asserts. split in {"train", "val"}.

    Host-RAM disclosure (audit): this loads the ENTIRE {split}.pt tensor
    into host RAM via torch.load(map_location="cpu") before any .to(device)
    transfer. For the wikitext-mix-ext corpus this is an ESTIMATED ~3.3 GB
    host RAM per process (train+val combined, int64 tokens at 8 bytes each
    -- consistent with an extended-mix corpus on the order of ~400M
    combined tokens, the scale implied by SCALE_TRANSFER_DESIGN.md's own
    "rung 2's 1.5B-token/run target" sizing note for these corpora). This
    was not measured directly (no box access from this sandbox -- see
    EMBEDDING_ABLATION_DESIGN.md S10) and should be confirmed by the first
    real probe cell's own process RSS.
    """
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
    """Same random-contiguous-window sampling as lm_pretrain_rd.get_batch.
    NOTE (audit-flagged, EMBEDDING_ABLATION_DESIGN.md S10): this is the
    plain uniform-window sampler from round1/round2_*_script.py, NOT
    lm_pretrain_rd.py's document-boundary-aware sampler (boundary_stats,
    doc_offsets). Both arms see the identical procedure and the corpus is
    EOT-separated, so this does not asymmetrize the comparison, but it
    does mean this design does not inherit that file's boundary-crossing
    diagnostics."""
    n = tokens.numel()
    assert n > seq_len + 1, f"corpus too small ({n} tokens) for seq_len={seq_len}"
    ix = torch.randint(0, n - seq_len - 1, (batch_size,), generator=generator)
    offs = torch.arange(seq_len + 1)
    idx = ix.unsqueeze(1) + offs.unsqueeze(0)
    window = tokens[idx]
    x, y = window[:, :-1].contiguous(), window[:, 1:].contiguous()
    return x, y


# ═══════════════════════════════════════════════════════════════════════
# Global constants (defined before the model classes / solvers that use
# them -- init std, matching tolerance, registered sizes).
# ═══════════════════════════════════════════════════════════════════════

# CLAUDE.md hard rule (audit M2 RULING, applied explicitly here -- neither
# matrix_thinker.py nor round1/round2_*_script.py ever set this): "Outer-
# product embedding init: u,v std must be sqrt(target_std), not
# target_std. Products have std=sigma^2." TARGET_STD is the desired std of
# the CONSTRUCTED representation each arm's embedding produces (the matrix
# arm's M=u⊗v entries; the flat/flatten arms' embedding-table entries
# directly) -- matched across all three arms so the only asymmetry left is
# the operation, not the init scale.
TARGET_STD = 0.02

SIZE_CONFIGS = {
    "S": dict(mat_dim=16, n_layers=6, n_heads=4),
    "M": dict(mat_dim=24, n_layers=8, n_heads=4),
}
N_ITERATIONS = 8          # T for the "matrix model's iterative T" eval leg
EVAL_ITERATIONS = (1, 8)  # T=1 (headline) and T=N_ITERATIONS, BOTH reported
MAX_LEN = 512
PARAM_MATCH_TOL = 0.01    # +/-1%, per the design doc's requirement


# ═══════════════════════════════════════════════════════════════════════
# MATRIX ARM -- outer-product embedding + RowThenCol matrix ops.
# Copied module-for-module from experiment-runs/8xh100-session1/
# round2_matrix_script.py's MatrixThinker (this ablation changes the
# comparison's rigor, not the architecture the claims were made about),
# with ONE addition: explicit outer-product init (audit M2).
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


def _init_outer_product_embedding(embed_u, embed_v, target_std=TARGET_STD):
    """CLAUDE.md rule (audit M2): u,v std must be sqrt(target_std), not
    target_std -- Var(u_i * v_j) = Var(u)*Var(v) for independent zero-mean
    u,v, so std(u)=std(v)=s must satisfy s^4=target_std^2, i.e.
    s=sqrt(target_std), for the CONSTRUCTED matrix M=u⊗v to have entry std
    == target_std."""
    s = math.sqrt(target_std)
    nn.init.normal_(embed_u.weight, std=s)
    nn.init.normal_(embed_v.weight, std=s)


class MatrixThinker(nn.Module):
    """Outer-product embedding + shared ThinkingBlock stack applied
    n_iterations times (weight-shared iterative refinement -- T does NOT
    change parameter count, only n_layers does)."""
    def __init__(self, mat_dim=16, n_layers=6, n_heads=4, max_len=512,
                 vocab_size=GPT2_VOCAB_SIZE, dropout=0.1, target_std=TARGET_STD):
        super().__init__()
        self.mat_dim = mat_dim
        d = mat_dim
        self.embed_u = nn.Embedding(vocab_size, d)
        self.embed_v = nn.Embedding(vocab_size, d)
        self.pos_u = nn.Embedding(max_len, d)
        self.pos_v = nn.Embedding(max_len, d)
        _init_outer_product_embedding(self.embed_u, self.embed_v, target_std)
        _init_outer_product_embedding(self.pos_u, self.pos_v, target_std)
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
# Copied module-for-module from round1_vector_script.py's VectorThinker,
# PLUS (audit M2): explicit init at the SAME target_std as the matrix
# arm's constructed entries (direct std=target_std here, not sqrt --
# there is no outer-product factoring to compensate for), and the SAME
# 0.1 residual scaling on the positional term the matrix arm uses (the
# original round1_vector_script.py added position with NO scaling --
# fixed here so the only asymmetry between arms is the operation, not an
# accidental init/scale mismatch).
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
                 vocab_size=GPT2_VOCAB_SIZE, dropout=0.1, target_std=TARGET_STD):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embed.weight, std=target_std)
        nn.init.normal_(self.pos.weight, std=target_std)
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
        pos_ids = torch.arange(L, device=token_ids.device).unsqueeze(0)
        h = self.embed(token_ids) + self.pos(pos_ids) * 0.1  # matches matrix arm's pos-term scaling
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
# FLATTEN ARM (audit M3, NEW) -- the SAME outer-product embedding as the
# matrix arm, flattened to a d^2 vector, resized into the SAME dense
# VectorThinkingBlock backbone the flat arm uses. This is Run 18's own
# historical recipe ("same outer-product embedding, then FLATTEN to a
# 256-dim vector, standard transformer") made genuinely params-matched:
# Run 18 ran the backbone at the FULL d^2 width with no resize and ended
# up 10x asymmetric (flat favored); here d_model (the backbone's own
# operating width, distinct from the d^2 embedding width) is solved for
# total-param equality.
# ═══════════════════════════════════════════════════════════════════════

class FlattenThinker(nn.Module):
    def __init__(self, mat_dim=16, d_model=16, n_layers=6, n_heads=4, max_len=512,
                 vocab_size=GPT2_VOCAB_SIZE, dropout=0.1, target_std=TARGET_STD):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.mat_dim = mat_dim
        d = mat_dim
        self.embed_u = nn.Embedding(vocab_size, d)
        self.embed_v = nn.Embedding(vocab_size, d)
        self.pos_u = nn.Embedding(max_len, d)
        self.pos_v = nn.Embedding(max_len, d)
        _init_outer_product_embedding(self.embed_u, self.embed_v, target_std)
        _init_outer_product_embedding(self.pos_u, self.pos_v, target_std)
        self.resize_in = nn.Linear(d * d, d_model)
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
        d = self.mat_dim
        u, v = self.embed_u(token_ids), self.embed_v(token_ids)
        M = torch.einsum('...i,...j->...ij', u, v)
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        pu, pv = self.pos_u(pos), self.pos_v(pos)
        M = M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1
        h = self.resize_in(M.reshape(B, L, d * d))  # FLATTEN, then dense resize -- structure gone here
        for _ in range(n_iterations):
            h = self._one_iteration(h)
        h = self.norm(h)
        return self.output(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self):
        embed = sum(p.numel() for n, p in self.named_parameters() if n.startswith(('embed_', 'pos_')))
        resize = sum(p.numel() for n, p in self.named_parameters() if n.startswith('resize_in'))
        backbone = sum(p.numel() for n, p in self.named_parameters() if n.startswith('layers'))
        head = sum(p.numel() for n, p in self.named_parameters() if n.startswith(('output', 'norm')))
        return {"embed": embed, "resize_in": resize, "backbone": backbone, "head": head,
                "total": self.count_params()}


# ═══════════════════════════════════════════════════════════════════════
# Param-count solving and the params-matched gate.
# ═══════════════════════════════════════════════════════════════════════

def flat_params_analytic(d_model, n_layers, vocab_size=GPT2_VOCAB_SIZE, max_len=MAX_LEN):
    """Closed-form flat-arm param count -- used ONLY to drive the search
    below. The authoritative count always comes from instantiating the
    real nn.Module (see the `real = ...count_params()` verification in
    every solve_* function)."""
    embed = vocab_size * d_model
    pos = max_len * d_model
    mha = 4 * d_model * d_model + 4 * d_model
    ln = 2 * (2 * d_model)
    ffn = (d_model * 4 * d_model + 4 * d_model) + (4 * d_model * d_model + d_model)
    per_layer = mha + ln + ffn
    final_ln = 2 * d_model
    head = d_model * vocab_size
    return embed + pos + n_layers * per_layer + final_ln + head


def flatten_params_analytic(mat_dim, d_model, n_layers, vocab_size=GPT2_VOCAB_SIZE, max_len=MAX_LEN):
    """Closed-form flatten-arm param count (embed/pos are shared with the
    matrix arm's own outer-product tables at the SAME mat_dim -- only the
    resize+backbone+head differ)."""
    d = mat_dim
    embed = 2 * vocab_size * d
    pos = 2 * max_len * d
    resize_in = d * d * d_model + d_model
    mha = 4 * d_model * d_model + 4 * d_model
    ln = 2 * (2 * d_model)
    ffn = (d_model * 4 * d_model + 4 * d_model) + (4 * d_model * d_model + d_model)
    per_layer = mha + ln + ffn
    final_ln = 2 * d_model
    head = d_model * vocab_size
    return embed + pos + resize_in + n_layers * per_layer + final_ln + head


def solve_matched_width(target_params, params_fn, n_heads_candidates, hi=8192, tol=PARAM_MATCH_TOL):
    """Searches integer widths for the closest total-param match, trying
    each n_heads candidate in order and stopping at the FIRST one that
    hits `tol` (preferring more heads / coarser-but-sufficient granularity
    over fewer heads / finer granularity when either would do). Falling
    through the whole candidate list returns the best achievable match
    even if it misses tol -- the caller's check_param_match() is what
    actually gates on tol, this function only searches.

    Why multiple candidates are needed (audit-discovered): fixing a single
    n_heads a priori (e.g. 4) can leave gaps of >1% between adjacent valid
    widths for some (mat_dim, n_layers) combinations -- this is exactly
    what happened when the flatten arm's size-M search was first tried at
    a fixed n_heads=4 (best achievable was 1.09% off, outside tol) and
    fixed by allowing n_heads in {8,4,2,1} instead (found 0.39% off at
    n_heads=1, d_model=25). Registered flat-P arms did not need this
    (both hit <=0.25% at a fixed n_heads=4) and keep their original
    single-candidate call for reproducibility of already-verified numbers.
    """
    last_best = None
    for nh in n_heads_candidates:
        best = None
        for c in range(nh, hi, nh):
            t = params_fn(c)
            if best is None or abs(t - target_params) < abs(best[1] - target_params):
                best = (c, t)
            if t > target_params * 1.6 and c > nh * 2:
                break
        last_best = (best[0], best[1], nh)
        if abs(best[1] / target_params - 1.0) <= tol:
            return last_best
    return last_best


def solve_matched_d_model(target_params, n_layers, n_heads, vocab_size=GPT2_VOCAB_SIZE,
                           max_len=MAX_LEN, hi=8192):
    """flat-P arm: single n_heads candidate (the size's registered
    n_heads), then verified by REAL instantiation."""
    fn = lambda c: flat_params_analytic(c, n_layers, vocab_size, max_len)
    d_model, _, _ = solve_matched_width(target_params, fn, (n_heads,), hi=hi)
    real = VectorThinker(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                          max_len=max_len, vocab_size=vocab_size).count_params()
    return d_model, real


def solve_matched_d_model_flatten(target_params, mat_dim, n_layers, vocab_size=GPT2_VOCAB_SIZE,
                                   max_len=MAX_LEN, hi=8192):
    """flatten-P arm: searches n_heads in {8,4,2,1} (its own, independent
    of the matrix/flat-P arms' n_heads) for the first that hits tol, then
    verified by REAL instantiation."""
    fn = lambda c: flatten_params_analytic(mat_dim, c, n_layers, vocab_size, max_len)
    d_model, _, n_heads = solve_matched_width(target_params, fn, (8, 4, 2, 1), hi=hi)
    real = FlattenThinker(mat_dim=mat_dim, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           max_len=max_len, vocab_size=vocab_size).count_params()
    return d_model, n_heads, real


def check_param_match(matrix_total, other_total, tol=PARAM_MATCH_TOL, label=""):
    """The negative-test gate: raises loudly if the two arms are not
    within `tol` of each other. Called for --match P cells; --match D
    cells call this in report-only mode (raise=False) since they are
    PRE-REGISTERED as unmatched."""
    ratio = other_total / matrix_total
    diff = abs(ratio - 1.0)
    ok = diff <= tol
    msg = (f"{label} param match: matrix={matrix_total:,} other={other_total:,} "
           f"ratio={ratio:.4f} diff={diff*100:.3f}% tol={tol*100:.1f}% -> "
           f"{'PASS' if ok else 'FAIL'}")
    return ok, ratio, msg


def build_arm(arm, size, match, dropout=0.1, vocab_size=GPT2_VOCAB_SIZE, max_len=MAX_LEN):
    """Builds one model + returns (model, meta_dict). meta_dict always
    carries params_total and, for flat/flatten, the params_ratio vs the
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

    del matrix_ref  # only needed for its param count above

    if arm == "flat":
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
            raise ValueError(f"unknown match condition {match!r} for arm=flat, expected 'P' or 'D'")

        model = VectorThinker(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                               max_len=max_len, vocab_size=vocab_size, dropout=dropout)
        real_total = model.count_params()
        meta = {"arm": "flat", "size": size, "match": match, "d_model": d_model,
                "n_layers": n_layers, "n_heads": n_heads, "params_total": real_total,
                "params_breakdown": model.param_breakdown(),
                "matrix_twin_params": matrix_total,
                "params_ratio_vs_matrix": real_total / matrix_total}
        return model, meta

    if arm == "flatten":
        if match != "P":
            raise ValueError("arm='flatten' is only registered for --match P (params-matched by "
                              "construction) -- no unmatched control is defined for this arm.")
        d_model, n_heads_flatten, flatten_total = solve_matched_d_model_flatten(
            matrix_total, mat_dim, n_layers, vocab_size, max_len)
        ok, ratio, msg = check_param_match(matrix_total, flatten_total, PARAM_MATCH_TOL,
                                            label=f"flatten-P {size}")
        if not ok:
            raise RuntimeError(
                f"PARAMS-MATCHED (--match P) GATE FAILED for flatten arm, size={size}: {msg}. "
                f"Widen solve_matched_d_model_flatten's n_heads candidates or hi bound before "
                f"trusting this cell."
            )
        model = FlattenThinker(mat_dim=mat_dim, d_model=d_model, n_layers=n_layers,
                                n_heads=n_heads_flatten, max_len=max_len, vocab_size=vocab_size,
                                dropout=dropout)
        real_total = model.count_params()
        meta = {"arm": "flatten", "size": size, "match": match, "mat_dim": mat_dim,
                "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads_flatten,
                "params_total": real_total, "params_breakdown": model.param_breakdown(),
                "matrix_twin_params": matrix_total,
                "params_ratio_vs_matrix": real_total / matrix_total}
        return model, meta

    raise ValueError(f"unknown arm {arm!r}, expected 'matrix', 'flat', or 'flatten'")


# ═══════════════════════════════════════════════════════════════════════
# Eval / training
# ═══════════════════════════════════════════════════════════════════════

def evaluate(model, val_tokens, vocab_size, device, n_iterations, seq_len,
             eval_batch_size, max_eval_batches, corpus_name):
    """M1 fix (audit): reseeds a FRESH generator from
    corpus_fixed_seed(corpus_name) at the START of every call -- never
    from the training seed, never carried over between calls. This
    guarantees every seed, every arm, and both T-legs (T=1, T=8, called
    back-to-back within the same eval round) score the IDENTICAL
    validation windows in the IDENTICAL order, so any BPB difference
    reflects the model, not which random windows it happened to see."""
    model.eval()
    gen = torch.Generator().manual_seed(corpus_fixed_seed(corpus_name))
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for _ in range(max_eval_batches):
            x, y = get_batch(val_tokens, eval_batch_size, seq_len, gen)
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
                 "eval_iterations": list(EVAL_ITERATIONS), "role": args.role,
                 "corpus_seed_used_for_eval": corpus_fixed_seed(args.corpus)})

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)

    train_tokens, train_meta = load_corpus_tokens(args.data_dir, args.corpus, "train")
    val_tokens, _ = load_corpus_tokens(args.data_dir, args.corpus, "val")
    gen = torch.Generator().manual_seed(args.seed)  # TRAINING order intentionally varies by seed

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warmup = max(1, args.steps // 10)

    def lr_lambda(step):
        # minor fix (audit): step=0 used to give (0/warmup)=0, wasting the
        # first step at LR=0. warmup now starts at 1/warmup, not 0/warmup.
        if step < warmup:
            return (step + 1) / warmup
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
                                           args.eval_batches, args.corpus)
            training_curve.append({"step": step, "train_loss_nats": loss.item(), "evals": evals})
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "last.pt"))

    if status == "CEILING_STOP":
        # minor fix (audit): a ceiling stop must never leave final_evals
        # stale or empty just because the stop happened between eval
        # boundaries -- force one more eval pass before saving.
        forced_evals = {}
        for T in EVAL_ITERATIONS:
            forced_evals[f"T{T}"] = evaluate(model, val_tokens, GPT2_VOCAB_SIZE, device, T,
                                              args.seq_len, args.eval_batch_size,
                                              args.eval_batches, args.corpus)
        training_curve.append({"step": step, "train_loss_nats": None, "evals": forced_evals,
                                "forced_final_eval_on_ceiling_stop": True})
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
# Harvest (audit F1 REV): strict filtering, per-group exact-3 invariant,
# two independent decisions (01: matrix vs flatten; 04: matrix vs flat-P).
# ═══════════════════════════════════════════════════════════════════════

def _load_results_dir(results_dir):
    """Non-recursive top-level scan only -- a `probes/` subdirectory
    (where probe results are written, per audit F1) is a directory entry
    that does not end in `.json` and is skipped automatically, in
    addition to the explicit `_is_probe` filter below (belt-and-suspenders
    in case a probe result is ever misplaced into the main dir)."""
    records = []
    for fn in sorted(os.listdir(results_dir)):
        full = os.path.join(results_dir, fn)
        if not os.path.isfile(full) or not fn.endswith(".json"):
            continue
        with open(full) as f:
            r = json.load(f)
        r["_source_filename"] = fn
        records.append(r)
    return records


def _is_probe(r):
    return (r.get("role") == "probe"
            or "probe" in str(r.get("_source_filename", "")).lower()
            or "probe" in str(r.get("id", "")).lower())


def _harvest_records(records, out_path=None):
    """F1: filters to complete==True AND steps_target>=2000 AND not a
    probe (role field OR filename), then enforces the exact-3 invariant
    per (arm, match, size) group, then scores the two independent
    decisions. A group with 1-2 or 4+ valid records raises loudly --
    that is a bug (lost/duplicate seed or a leaked partial/probe record),
    never silently averaged over or dropped."""
    valid = [r for r in records
             if r.get("complete") is True
             and r.get("steps_target", 0) >= 2000
             and not _is_probe(r)]

    groups = {}
    for r in valid:
        key = (r.get("arm"), r.get("match"), r.get("size"))
        groups.setdefault(key, []).append(r)
    for key, recs in groups.items():
        assert len(recs) == 3, (
            f"F1 INVARIANT VIOLATED: group (arm,match,size)={key} has {len(recs)} valid seed "
            f"records after filtering (complete=True, steps_target>=2000, non-probe); expected "
            f"exactly 3 (the pre-registered seed count) or 0 (not yet run). 1-2 means a missing "
            f"or lost seed; 4+ means a duplicate or a partial/probe record leaked through the "
            f"filter -- do not trust ANY harvest output until this is resolved. Offending files: "
            f"{[r.get('_source_filename') for r in recs]}"
        )

    def t1_bpb(r):
        return r.get("final_evals", {}).get("T1", {}).get("token_bpb")

    def score(ref_arm, other_arm, size):
        ref = groups.get((ref_arm, "P", size), [])
        other = groups.get((other_arm, "P", size), [])
        if not ref or not other:
            return {"status": "PENDING", "n_ref": len(ref), "n_other": len(other)}
        ref_vals = sorted(t1_bpb(r) for r in ref)
        other_vals = sorted(t1_bpb(r) for r in other)
        assert len(ref_vals) == 3 and len(other_vals) == 3 and None not in ref_vals \
            and None not in other_vals, (
            f"missing T1 token_bpb in a supposedly-complete {ref_arm}/{other_arm} {size} record"
        )
        seed_spread = max(max(ref_vals) - min(ref_vals), max(other_vals) - min(other_vals))
        wins = sum(1 for rv, ov in zip(ref_vals, other_vals) if (ov - rv) > seed_spread)
        return {"status": "SCORED", f"{ref_arm}_t1_bpb": ref_vals, f"{other_arm}_t1_bpb": other_vals,
                "seed_spread": seed_spread, "wins": wins, "n_pairs": 3, "size_pass": wins >= 2}

    decisions = {}
    for label, (ref_arm, other_arm) in (
        ("STRENGTHEN-01 (matrix vs flatten)", ("matrix", "flatten")),
        ("STRENGTHEN-04 (matrix vs flat-P)", ("matrix", "flat")),
    ):
        per_size = {size: score(ref_arm, other_arm, size) for size in SIZE_CONFIGS}
        scored = [v for v in per_size.values() if v["status"] == "SCORED"]
        decision = "PENDING"
        if len(scored) == len(SIZE_CONFIGS):
            decision = "STRENGTHEN" if all(v["size_pass"] for v in scored) else "DROP"
        decisions[label] = {"per_size": per_size, "decision": decision}

    flatd = {size: {"n": len(groups.get(("flat", "D", size), [])),
                     "t1_bpb": sorted(t1_bpb(r) for r in groups.get(("flat", "D", size), []))}
             for size in SIZE_CONFIGS}

    summary = {
        **decisions,
        "flat-D_disclosed_not_gating": flatd,
        "n_records_loaded": len(records),
        "n_records_valid_after_filter": len(valid),
        "rule": ("STRENGTHEN-XX iff the named arm beats its comparison arm's T=1 token_bpb on "
                 ">=2/3 seed pairs by more than the seed spread, at BOTH sizes S and M. Else "
                 "DROP. PENDING if not all sizes scored yet. No third outcome per size; 01 and "
                 "04 are decided completely independently."),
    }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    return summary


def harvest(results_dir, out_path=None):
    return _harvest_records(_load_results_dir(results_dir), out_path)


def _make_synth_record(arm, match, size, seed, t1_bpb, role="cell", complete=True):
    rec = {"experiment": "embed_ablation_rd", "arm": arm, "match": match, "size": size,
           "seed": seed, "steps_target": 2000, "steps_completed": 2000 if complete else 480,
           "complete": complete, "status": "COMPLETED" if complete else "CEILING_STOP"}
    if role is not None:
        rec["role"] = role
    if t1_bpb is not None:
        rec["final_evals"] = {"T1": {"token_bpb": t1_bpb}, "T8": {"token_bpb": t1_bpb * 0.9}}
    else:
        rec["final_evals"] = {}
    return rec


def harvest_selftest():
    """F1: builds the exact synthetic scenario the audit specified --
    15 clean complete cells (5 groups x 3 seeds: matrix/S, matrix/M,
    flat-P/S, flat-P/M, flatten-P/S -- flatten-P/M deliberately MISSING)
    + 2 probe-like records (one with role='probe', one with no role field
    but 'probe' in its filename, exercising BOTH detection paths F1
    names) + 1 CEILING_STOP record (for the missing flatten-P/M group,
    complete=False) -- then proves harvest()'s verdict is byte-identical
    whether the 3 extras are present or not."""
    print("\n--- harvest-selftest: synthetic 15-clean + 2-probe + 1-CEILING_STOP dir ---")
    bpb_map = {
        ("matrix", "P", "S"): [2.10, 2.12, 2.11],
        ("matrix", "P", "M"): [1.95, 1.97, 1.96],
        ("flat", "P", "S"): [2.40, 2.42, 2.41],
        ("flat", "P", "M"): [2.20, 2.22, 2.21],
        ("flatten", "P", "S"): [2.30, 2.32, 2.31],
    }
    with tempfile.TemporaryDirectory() as dA, tempfile.TemporaryDirectory() as dB:
        for d in (dA, dB):
            for (arm, match, size), vals in bpb_map.items():
                for seed, v in enumerate(vals):
                    fn = os.path.join(d, f"embed_ablation_{arm}_{size}_s{seed}.json")
                    with open(fn, "w") as f:
                        json.dump(_make_synth_record(arm, match, size, seed, v), f)
        # extras: dB only
        with open(os.path.join(dB, "embed_ablation_probe_matrix_S.json"), "w") as f:
            json.dump(_make_synth_record("matrix", "P", "S", 99, 9.99, role="probe"), f)
        with open(os.path.join(dB, "probe_flat_S_extra.json"), "w") as f:
            json.dump(_make_synth_record("flat", "P", "S", 98, 9.98, role=None), f)
        with open(os.path.join(dB, "embed_ablation_flatten_M_s0.json"), "w") as f:
            json.dump(_make_synth_record("flatten", "P", "M", 0, None, complete=False), f)

        print(f"  dir A: {len(os.listdir(dA))} files (expect 15)")
        print(f"  dir B: {len(os.listdir(dB))} files (expect 18 = 15 + 2 probes + 1 ceiling-stop)")
        summary_a = _harvest_records(_load_results_dir(dA))
        summary_b = _harvest_records(_load_results_dir(dB))

    ok = True
    decision_keys = ["STRENGTHEN-01 (matrix vs flatten)", "STRENGTHEN-04 (matrix vs flat-P)",
                      "flat-D_disclosed_not_gating"]
    for k in decision_keys:
        if summary_a[k] != summary_b[k]:
            ok = False
            print(f"  MISMATCH in {k!r}:\n    A={summary_a[k]}\n    B={summary_b[k]}")
    if summary_a["n_records_valid_after_filter"] != summary_b["n_records_valid_after_filter"]:
        ok = False
        print(f"  MISMATCH in n_records_valid_after_filter: A={summary_a['n_records_valid_after_filter']} "
              f"B={summary_b['n_records_valid_after_filter']}")
    if summary_a["n_records_valid_after_filter"] != 15:
        ok = False
        print(f"  expected exactly 15 valid records in the clean baseline dir, got "
              f"{summary_a['n_records_valid_after_filter']}")
    if summary_b["n_records_loaded"] != 18:
        ok = False
        print(f"  expected 18 total loaded records in dir B, got {summary_b['n_records_loaded']}")

    print(f"  STRENGTHEN-04 decision: A={summary_a['STRENGTHEN-04 (matrix vs flat-P)']['decision']} "
          f"B={summary_b['STRENGTHEN-04 (matrix vs flat-P)']['decision']}")
    print(f"  STRENGTHEN-01 decision: A={summary_a['STRENGTHEN-01 (matrix vs flatten)']['decision']} "
          f"B={summary_b['STRENGTHEN-01 (matrix vs flatten)']['decision']} (expected PENDING: "
          f"flatten/M is missing from both dirs by construction)")

    print(f"  harvest-selftest: {'PASS' if ok else 'FAIL'} -- probes + the CEILING_STOP cell "
          f"{'provably do not change' if ok else 'CHANGED'} the verdict")
    return ok


# ═══════════════════════════════════════════════════════════════════════
# Admission check (audit M4): reads phase-A probe results, pre-registers
# whether phase-B may be staged.
# ═══════════════════════════════════════════════════════════════════════

def check_admission(probe_results_dir, intended_steps, intended_batch, ceiling_gpuh=2.0):
    """M4: run AFTER the phase-A probes land and BEFORE phase-B is
    staged. For every probe result JSON in probe_results_dir, checks:
      (a) the last three T=1 evals in its training_curve are monotone
          NON-INCREASING (loss going down or flat -- a coarse 'is this
          actually learning, not diverging' check). FAILS LOUDLY, never
          silently passes, if not.
      (b) the probe's measured wall time, extrapolated linearly (per
          step x per batch-unit) to intended_steps x intended_batch, does
          not exceed ceiling_gpuh. If it does, this is reported and the
          caller MUST NOT stage phase B without first re-deriving --steps
          (or investigating the non-monotone arm).
    Returns True only if EVERY probe passes BOTH checks."""
    results = _load_results_dir(probe_results_dir)
    if not results:
        print(f"ADMISSION CHECK: no probe results found in {probe_results_dir} -- cannot admit phase B.")
        return False
    all_ok = True
    for r in results:
        label = r.get("id") or r.get("_source_filename") or "?"
        curve = r.get("training_curve", [])
        t1_seq = [c["evals"]["T1"]["token_bpb"] for c in curve
                  if "T1" in c.get("evals", {}) and c["evals"]["T1"].get("token_bpb") is not None]
        last3 = t1_seq[-3:]
        monotone = all(a >= b for a, b in zip(last3, last3[1:])) if len(last3) >= 2 else True
        if not monotone:
            all_ok = False
            print(f"  [{label}] FAIL (a): last-3 T1 token_bpb not monotone non-increasing: {last3}")
        else:
            print(f"  [{label}] OK (a): last-3 T1 token_bpb monotone non-increasing: {last3}")

        probe_steps = r.get("steps_completed", 0)
        probe_batch = r.get("batch_size", 0)
        elapsed_min = r.get("time_min", 0.0)
        if probe_steps <= 0 or probe_batch <= 0 or elapsed_min <= 0:
            all_ok = False
            print(f"  [{label}] FAIL (b): missing steps_completed/batch_size/time_min, cannot extrapolate rate")
            continue
        gpu_h_per_step_batch_unit = (elapsed_min / 60.0) / (probe_steps * probe_batch)
        extrapolated_gpu_h = gpu_h_per_step_batch_unit * intended_steps * intended_batch
        within = extrapolated_gpu_h <= ceiling_gpuh
        print(f"  [{label}] (b) extrapolated to steps={intended_steps} batch={intended_batch}: "
              f"{extrapolated_gpu_h:.3f} GPU-h ({'within' if within else 'EXCEEDS'} the "
              f"{ceiling_gpuh} GPU-h/cell ceiling)")
        if not within:
            all_ok = False

    print(f"ADMISSION CHECK: {'PASS -- phase B may be staged' if all_ok else 'FAIL -- STOP, do NOT stage phase B; re-derive --steps or investigate the failing arm first'}")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════
# Self-test (CPU-only, tiny configs): forward/backward/grad check for ALL
# THREE arms + the param-count negative test + harvest-selftest.
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

    # ---- 0. Init rule check (audit M2): embed_u/v std ~ sqrt(TARGET_STD),
    #         flat/flatten embed std ~ TARGET_STD directly. ----
    print("\n--- init std check (M2 ruling) ---")
    mm0 = MatrixThinker(mat_dim=16, n_layers=1, n_heads=4, max_len=64, vocab_size=1000)
    got_u = mm0.embed_u.weight.std().item()
    want_u = math.sqrt(TARGET_STD)
    print(f"  matrix embed_u std={got_u:.4f} target=sqrt({TARGET_STD})={want_u:.4f}")
    if abs(got_u - want_u) / want_u > 0.15:
        failures.append(f"matrix embed_u init std off: got {got_u:.4f} want ~{want_u:.4f}")
    vm0 = VectorThinker(d_model=32, n_layers=1, n_heads=4, max_len=64, vocab_size=1000)
    got_flat = vm0.embed.weight.std().item()
    print(f"  flat embed std={got_flat:.4f} target=TARGET_STD={TARGET_STD:.4f}")
    if abs(got_flat - TARGET_STD) / TARGET_STD > 0.15:
        failures.append(f"flat embed init std off: got {got_flat:.4f} want ~{TARGET_STD:.4f}")

    # ---- 1. Tiny forward/backward/grad check, all three arms, both toy configs ----
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
            # test of matching precision (see the real-scale check below).
            matrix_total = mm.count_params()
            d_model, flat_total = solve_matched_d_model(matrix_total, cfg["n_layers"],
                                                          cfg["n_heads"], vocab_size=tiny_vocab,
                                                          max_len=L, hi=512)
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

        try:
            d_model_f, n_heads_f, flatten_total = solve_matched_d_model_flatten(
                matrix_total, cfg["mat_dim"], cfg["n_layers"], vocab_size=tiny_vocab,
                max_len=L, hi=512)
            ok, ratio, msg = check_param_match(matrix_total, flatten_total, PARAM_MATCH_TOL,
                                                label=f"flatten-P/{size}")
            print(f"  {msg}  [informational at toy vocab; real gate below]  n_heads={n_heads_f}")
            fm = FlattenThinker(mat_dim=cfg["mat_dim"], d_model=d_model_f, n_layers=cfg["n_layers"],
                                 n_heads=n_heads_f, max_len=L, vocab_size=tiny_vocab)
            _grad_check(fm, x, y, tiny_vocab, f"flatten-P/{size} (d_model={d_model_f})")
        except Exception as e:
            failures.append(f"flatten-P/{size}: {e}")
            print(f"  FAIL flatten-P/{size}: {e}")

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
    #         GPT-2 vocab scale -- the real config the box will run,
    #         for ALL THREE arms. ----
    print("\n--- registered sizes at real GPT-2 vocab (50257), match=P ---")
    for size, cfg in SIZE_CONFIGS.items():
        mm = MatrixThinker(mat_dim=cfg["mat_dim"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                            max_len=MAX_LEN, vocab_size=GPT2_VOCAB_SIZE)
        matrix_total = mm.count_params()

        d_model, flat_total = solve_matched_d_model(matrix_total, cfg["n_layers"], cfg["n_heads"])
        ok, ratio, msg = check_param_match(matrix_total, flat_total, PARAM_MATCH_TOL, label=f"size {size} flat-P")
        print(f"  {msg}  (d_model={d_model}, n_heads={cfg['n_heads']})")
        if not ok:
            failures.append(f"registered size {size} flat-P failed params-matched gate: {msg}")

        d_model_f, n_heads_f, flatten_total = solve_matched_d_model_flatten(
            matrix_total, cfg["mat_dim"], cfg["n_layers"])
        ok_f, ratio_f, msg_f = check_param_match(matrix_total, flatten_total, PARAM_MATCH_TOL,
                                                  label=f"size {size} flatten-P")
        print(f"  {msg_f}  (d_model={d_model_f}, n_heads={n_heads_f})")
        if not ok_f:
            failures.append(f"registered size {size} flatten-P failed params-matched gate: {msg_f}")

        d_model_d = 2 * cfg["mat_dim"]
        flat_d_total = flat_params_analytic(d_model_d, cfg["n_layers"])
        okd, ratiod, msgd = check_param_match(matrix_total, flat_d_total, PARAM_MATCH_TOL,
                                               label=f"size {size} (match=D, expected UNMATCHED)")
        print(f"  {msgd}  (d_model={d_model_d}) -- D is pre-registered as unmatched, this is informational")

    # ---- 4. Harvest selftest (audit F1) ----
    harvest_ok = harvest_selftest()
    if not harvest_ok:
        failures.append("harvest_selftest FAILED -- probes/CEILING_STOP changed the harvested verdict")

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
    p.add_argument("--harvest-selftest", action="store_true")
    p.add_argument("--check-admission", action="store_true")
    p.add_argument("--run-cell", action="store_true")
    p.add_argument("--harvest", action="store_true")

    p.add_argument("--arm", choices=["matrix", "flat", "flatten"])
    p.add_argument("--size", choices=list(SIZE_CONFIGS))
    p.add_argument("--match", choices=["P", "D"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=50)  # audit M1: raised from 20
    p.add_argument("--data-dir", type=str, default="/data/deltanet_rd_data")
    p.add_argument("--corpus", type=str, default="wikitext-mix-ext")
    p.add_argument("--ckpt-dir", type=str)
    p.add_argument("--out", type=str)
    p.add_argument("--role", type=str, default="cell", choices=["cell", "probe"])
    p.add_argument("--ceiling-gpuh", type=float, default=2.0,
                   help="hard wall-clock cutoff in GPU-hours; writes a CEILING_STOP result "
                        "(with a forced final eval) rather than running over budget")
    p.add_argument("--cpu", action="store_true", help="force CPU (selftest/debug only)")

    p.add_argument("--results-dir", type=str, help="--harvest: dir of per-cell result JSONs")
    p.add_argument("--probe-results-dir", type=str, help="--check-admission: dir of probe result JSONs")
    p.add_argument("--intended-steps", type=int, default=2000, help="--check-admission")
    p.add_argument("--intended-batch", type=int, default=64, help="--check-admission")

    args = p.parse_args()

    if args.selftest:
        sys.exit(selftest())
    elif args.harvest_selftest:
        sys.exit(0 if harvest_selftest() else 1)
    elif args.check_admission:
        if not args.probe_results_dir:
            p.error("--check-admission requires --probe-results-dir")
        sys.exit(0 if check_admission(args.probe_results_dir, args.intended_steps,
                                       args.intended_batch, args.ceiling_gpuh) else 1)
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
        p.error("specify one of --selftest, --harvest-selftest, --check-admission, --run-cell, --harvest")


if __name__ == "__main__":
    main()
