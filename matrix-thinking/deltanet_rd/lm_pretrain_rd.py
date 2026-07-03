"""lm_pretrain_rd.py -- Wave 2 (RD-2) instrumented-LM pretraining entry
point. See DELTANET_REALDATA_DESIGN.md section 6 (Wave 2 design: the
corpus-level reasoning-vs-narrative contrast, MAJOR-5's frequency-normalized
fallback for the deferred third corpus), section 4.2 (corrected cost model:
softmax-head-bottlenecked, ~0.6-2.2h/run at 400M tokens -- SCALED DOWN here
for the probe tier: ~100M tokens targeted per run), and section 14.7's
validity framing.

**Claim tier, stated once here and carried into every result JSON this
script writes (never silently upgraded at write-up time, section 6.3):
this arm is DESCRIPTIVE + INTERVENTIONAL. It is NOT the premise-conditional
causal tier (that is Wave 1 / RD-1 only, section 5's C16 premise machinery
-- Gram deviation, salvage ratio, bind->query alignment). None of that
machinery exists in this file. The strongest claim this arm can ever earn
is "inference-time causal + descriptive" (section 6.3), never "SGD was
pushed toward a rank-disciplined solution by training on real text."**

Architecture (probe-tier scale-down of DELTANET_REALDATA_DESIGN.md section
4.1's Wave-2 table -- d_model=256, d_state=64, 1-2 layers, per this build's
task brief, not the design doc's own full-scale 384/2-3-layer/20-28M
recommendation):

  x_0          = embed(token_ids)                          # (B,T,d_model), GPT-2 vocab (50,257)
  per layer i:
    a = RMSNorm(x_{i-1})
    q,k,v       = W_q(a), W_k(a), W_v(a)                    # (B,T,d_state) each, LEARNED, no bias
    q_eff,k_eff,v_eff = SiLU-conv1d(q,k,v)                  # causal short conv, matches fla's own
                                                              # DeltaNet layer convention exactly
    beta        = sigmoid(W_beta(a))                        # (B,T,H), PLAIN LEARNED gate -- NO C9
                                                              # hard mask, NO reserved buffer token
                                                              # (this is the one deliberate
                                                              # subtraction from model_rd.py's
                                                              # audited block per this build's brief:
                                                              # "WITHOUT the beta-mask/buffer
                                                              # machinery -- plain learned beta via
                                                              # a b_proj for LM mode")
    o, S        = chunk_delta_rule(q_eff,k_eff,v_eff,beta)  # PRODUCTION kernel, called DIRECTLY
                                                              # (R2-3's discipline, reused verbatim:
                                                              # bf16 ONLY at the kernel boundary)
    x_i         = x_{i-1} + W_o(RMSNorm_gate(o))
    x_i         = x_i + FFN(RMSNorm(x_i))
  logits        = RMSNorm_f(x_N) @ embed.weight^T            # TIED head (section 4.2's own cost
                                                               # model assumes this)

Unlike model_rd.py's DeltaNetRDBlock (Wave 1: external pinned readout only,
NO q_proj at all -- the kernel's own per-token output is architecturally
irrelevant there), this LM block DOES use a real, learned q_proj/q_conv1d:
an LM needs chunk_delta_rule's own per-token output o_t at EVERY position
for next-token prediction, not just the final recurrent state.

Reuses the audited block/kernel path where it transfers: F15-LM's measured
_MIN_KERNEL_T / _SAFE_D_STATE facts (imported directly from model_rd.py,
same-directory, pod-safe -- NOT re-measured here), and rank_utils.py's
effective_rank/stable_rank for the checkpoint rank instrumentation.

AUDIT FIXES (independent audit round 1, 2026-07-03 -- all three MAJORs):
  FIX-1  Eval/rank/intervention window samplers are seeded from
         corpus_fixed_seed(corpus) -- NEVER from the training seed -- so
         cross-seed aggregates compare identical held-out windows; window
         digests are logged per checkpoint for byte-identity verification.
  FIX-2  Tied embedding init std=0.02 (GPT-2 convention; the PyTorch
         N(0,1) default put initial loss at ~202 nats vs ~10.8 expected).
         Smoke item [1c] asserts initial loss < 12 at the real vocab.
  FIX-3  Trains ONLY on the rebuilt EOT-separated corpora
         (rebuild_lm_corpora_rd.py; load_corpus hard-refuses the original
         boundary-free tensors), rank sampling is document-aligned, and
         train/eval window boundary-contamination stats are logged into
         every result JSON so Wave C/D analysis can condition on them.

Run the smoke gate FIRST: python lm_pretrain_rd.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports

from fla.modules import RMSNorm, ShortConvolution
from fla.ops.delta_rule import chunk_delta_rule

from model_rd import _MIN_KERNEL_T, _SAFE_D_STATE
from rank_utils import effective_rank, stable_rank

HERE = os.path.dirname(os.path.abspath(__file__))

# Corpus registry. AUDIT FIX-3 (2026-07-03): points at the REBUILT
# EOT-separated corpora (rebuild_lm_corpora_rd.py), NOT the original flat
# concatenations under reasoning/ and wikitext103_tokenized/ -- the
# originals contain ZERO <|endoftext|> tokens and (OpenR1 avg doc length
# 466 < seq_len 512) a majority of training windows span unrelated
# concatenated problems with no boundary signal, which would undermine the
# reasoning-vs-narrative contrast itself. The rebuilt dirs carry an
# `eot_separated: true` meta field (asserted in load_corpus) and a
# `{split}_doc_offsets.pt` int64 tensor of document START positions
# alongside each token tensor. The original dirs are left untouched.
CORPUS_DIRS = {"openr1": "reasoning_eot", "wikitext": "wikitext103_eot"}
OTHER_CORPUS = {"openr1": "wikitext", "wikitext": "openr1"}
DEFAULT_DATA_DIR = "/data/deltanet_rd_data"
EOT_TOKEN_ID = 50256   # GPT-2 <|endoftext|>

# Rank-stat sampling: fractional positions within a sampled document
# (section 6, this build's brief: "whole-state effective/stable rank per
# head over sampled positions, split by document"). See
# sample_state_rank_stats's docstring for the document-alignment behavior
# (AUDIT FIX-3: windows start at document starts, truncation logged).
RANK_SAMPLE_FRACS = (0.25, 0.5, 0.75, 1.0)


def corpus_fixed_seed(corpus_name: str) -> int:
    """AUDIT FIX-1 (2026-07-03): the seed for EVERY held-out-window sampler
    (checkpoint eval loss, rank-stat sampling, Wave D intervention windows)
    -- a pure function of the CORPUS NAME, fully independent of the
    training seed (args.seed). Pre-fix behavior seeded these from
    args.seed(+offset+step), so each training seed evaluated on DIFFERENT
    held-out windows, confounding cross-seed aggregates with eval-set
    sampling noise. zlib.crc32 (not Python's hash()) because hash() is
    salted per-process -- crc32 is stable across processes/runs/machines."""
    return zlib.crc32(corpus_name.encode("utf-8"))


def window_digest(starts: torch.Tensor) -> int:
    """crc32 digest of a window-start index tensor -- logged per eval so
    FIX-1's cross-seed byte-identity requirement is VERIFIABLE from two
    runs' result JSONs (identical digests <=> identical window index sets),
    not merely asserted from code reading."""
    return zlib.crc32(starts.detach().cpu().to(torch.int64).numpy().tobytes())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FFN(nn.Module):
    """Plain 2-matrix GELU MLP (no gate) -- keeps the per-layer param count
    close to DELTANET_REALDATA_DESIGN.md section 4.1's own ~590K/layer
    estimate (a SwiGLU-style 3-matrix FFN would overshoot that by ~50%)."""

    def __init__(self, d_model: int, mult: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * mult, bias=False)
        self.fc2 = nn.Linear(d_model * mult, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class DeltaNetLMMixer(nn.Module):
    """LM-mode DeltaNet mixing sublayer. Reuses the R2-3 audited block's
    kernel-calling discipline VERBATIM (call fla.ops.delta_rule.chunk_delta_rule
    DIRECTLY, bf16 ONLY at the kernel boundary, the measured _SAFE_D_STATE
    head-dim floor) -- but this build's brief explicitly drops Wave 1's C9
    hard beta-mask and R2-3 buffer-pinning machinery: LM mode has no
    reserved BUFFER token and no hard-masked write positions. beta is a
    PLAIN LEARNED sigmoid gate at every position, exactly the stock
    fla.layers.delta_net.DeltaNet convention (verified against the
    installed fla 0.5.1 source, 2026-07-03).

    Also unlike model_rd.py's block: a REAL, learned q_proj/q_conv1d exists
    here (an LM needs the kernel's own per-token output o_t at every
    position; Wave 1's external, q-free, pinned readout does not transfer).
    use_qk_l2norm_in_kernel=True (kernel-internal q/k L2-normalization) is
    used instead of model_rd.py's external zero-safe F.normalize -- LM mode
    has no buffer/pad rows that could be exact-zero (every position is a
    real token), so the kernel's own normalization is safe here (matches
    the stock layer's own default, qk_norm='l2')."""

    def __init__(self, d_model: int, d_state: int, conv_size: int = 4, num_heads: int = 1):
        super().__init__()
        assert d_state % num_heads == 0, \
            f"d_state={d_state} must be divisible by num_heads={num_heads}"
        head_dim = d_state // num_heads
        assert head_dim in _SAFE_D_STATE, (
            f"head_dim={head_dim} (d_state={d_state} / num_heads={num_heads}) is NOT in the "
            f"measured-safe head-dim set {_SAFE_D_STATE} for chunk_delta_rule's backward on this "
            f"box's build (model_rd.py's _SAFE_D_STATE, F15-LM 2026-07-02: D<64 crashes "
            f"prepare_wy_repr_bwd's Triton autotuner). Re-measure before widening this set."
        )
        self.d_model = d_model
        self.d_state = d_state
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(d_model, d_state, bias=False)
        self.k_proj = nn.Linear(d_model, d_state, bias=False)
        self.v_proj = nn.Linear(d_model, d_state, bias=False)
        self.b_proj = nn.Linear(d_model, num_heads, bias=False)   # plain learned beta, NO mask (LM mode)
        self.q_conv1d = ShortConvolution(hidden_size=d_state, kernel_size=conv_size, bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(hidden_size=d_state, kernel_size=conv_size, bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(hidden_size=d_state, kernel_size=conv_size, bias=False, activation="silu")
        self.o_norm = RMSNorm(head_dim, eps=1e-5)
        self.o_proj = nn.Linear(d_state, d_model, bias=False)

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor | None = None):
        """x: (B,T,d_model). initial_state: None, or (B,H,head_dim,head_dim) in fla's OWN
        native [K,V] layout -- NOT reconciled to model_rd.py's [V,K] design layout, and
        deliberately so: LM mode never calls apply_state_power / any external readout that
        depends on that convention. The state only ever flows back into THIS SAME kernel (a
        self-consistent round trip -- verified by f15_lm_checkpoint.py's own round-trip item,
        section 4.4), and DeltaNet-RD/lm_intervene_rd.py's rank truncation is the best rank-k
        Frobenius approximation (Eckart-Young) of whatever matrix it is handed -- a
        layout-agnostic operation, unlike model_rd.py's FATAL-0 readout bug (which was about
        getting apply_state_power's RETRIEVAL formula's axis convention wrong, not about
        truncation). This is a deliberate build decision, not an oversight -- flagged here
        explicitly for audit review.

        T (or, for a continuation call, the length of x) MUST be >= _MIN_KERNEL_T. LM mode has
        no state-neutral pad token to extend a too-short call with (unlike Wave 1's buffer-token
        trick) -- callers must choose seq_len/ctx_len/continuation_len >= _MIN_KERNEL_T directly
        (validated at CLI-parse time in this file's/lm_intervene_rd.py's main()); this assert is
        a defensive backstop, not the primary enforcement point."""
        B, T, _ = x.shape
        assert T >= _MIN_KERNEL_T, (
            f"sequence length {T} < _MIN_KERNEL_T={_MIN_KERNEL_T} -- chunk_delta_rule's backward "
            f"crashes below this floor (F15-LM, measured 2026-07-02). Choose "
            f"seq_len/ctx_len/continuation_len >= {_MIN_KERNEL_T}."
        )
        q, _ = self.q_conv1d(self.q_proj(x))
        k, _ = self.k_conv1d(self.k_proj(x))
        v, _ = self.v_conv1d(self.v_proj(x))
        beta = torch.sigmoid(self.b_proj(x))                       # (B,T,H), plain learned gate

        q = q.reshape(B, T, self.num_heads, self.head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)

        # Kernel boundary: bf16 ONLY here (chunk_delta_rule categorically
        # rejects float32 -- section 4.3), mirroring model_rd.py's
        # kernel_state_design_layout discipline exactly.
        q_bf, k_bf, v_bf, beta_bf = (t.to(torch.bfloat16) for t in (q, k, v, beta))
        init_bf = initial_state.to(torch.bfloat16) if initial_state is not None else None
        o, final_state = chunk_delta_rule(q=q_bf, k=k_bf, v=v_bf, beta=beta_bf, initial_state=init_bf,
                                           output_final_state=True, use_qk_l2norm_in_kernel=True)
        o = o.float()
        final_state = final_state.float()

        o = self.o_norm(o)
        o = o.reshape(B, T, self.d_state)
        o = self.o_proj(o)
        return o, final_state


class DeltaNetLMBlock(nn.Module):
    """Pre-norm mixing sublayer + pre-norm FFN sublayer, standard residual
    stacking."""

    def __init__(self, d_model: int, d_state: int, conv_size: int = 4, num_heads: int = 1, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=1e-5)
        self.mixer = DeltaNetLMMixer(d_model, d_state, conv_size=conv_size, num_heads=num_heads)
        self.norm2 = RMSNorm(d_model, eps=1e-5)
        self.ffn = FFN(d_model, mult=ffn_mult)

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor | None = None):
        o, final_state = self.mixer(self.norm1(x), initial_state=initial_state)
        x = x + o
        x = x + self.ffn(self.norm2(x))
        return x, final_state


class DeltaNetLM(nn.Module):
    """The full LM: embed -> N DeltaNetLMBlocks -> final norm -> TIED head
    (section 4.2's own cost-model assumption: "tied embedding is the
    head"). d_model=256/d_state=64/n_layers in {1,2} is this build's
    probe-tier scale-down of section 4.1's Wave-2 table (~14M params at
    n_layers=2, ~13.5M at n_layers=1 -- both clear CLAUDE.md's 10M hard
    floor)."""

    def __init__(self, vocab_size: int, d_model: int = 256, d_state: int = 64, n_layers: int = 2,
                 conv_size: int = 4, num_heads: int = 1, ffn_mult: int = 4):
        super().__init__()
        assert n_layers in (1, 2), \
            "probe-tier build supports 1-2 layers only (task brief) -- widen deliberately, not by accident"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.conv_size = conv_size
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult

        self.embed = nn.Embedding(vocab_size, d_model)
        # AUDIT FIX-2 (2026-07-03): nn.Embedding's PyTorch default init is
        # N(0,1); with the TIED head that puts initial logit std ~= 16 and
        # initial loss ~= 202 nats instead of the ~ln(V) ~= 10.8 a
        # near-uniform softmax should give (auditor reproduced: rescaling
        # to std=0.02 gives 10.90). GPT-2's own convention: std=0.02.
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([
            DeltaNetLMBlock(d_model, d_state, conv_size=conv_size, num_heads=num_heads, ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model, eps=1e-5)

    def config(self) -> dict:
        return {"vocab_size": self.vocab_size, "d_model": self.d_model, "d_state": self.d_state,
                "n_layers": self.n_layers, "conv_size": self.conv_size, "num_heads": self.num_heads,
                "ffn_mult": self.ffn_mult}

    def forward(self, token_ids: torch.Tensor, initial_states: list | None = None,
                return_states: bool = False):
        """token_ids: (B,T) int64. initial_states: None (every layer starts fresh -- the ONLY
        mode training ever uses) or a list of length n_layers, each entry None or a
        (B,H,head_dim,head_dim) state (the intervention script's two-phase context/continuation
        use). Returns logits (B,T,vocab_size), and if return_states: also the list of each
        layer's FINAL state for this call."""
        B, T = token_ids.shape
        x = self.embed(token_ids)
        if initial_states is None:
            initial_states = [None] * self.n_layers
        assert len(initial_states) == self.n_layers, \
            f"initial_states must have one entry per layer ({self.n_layers}), got {len(initial_states)}"
        final_states = []
        for blk, s0 in zip(self.blocks, initial_states):
            x, s_final = blk(x, initial_state=s0)
            final_states.append(s_final)
        x = self.norm_f(x)
        logits = F.linear(x, self.embed.weight)                    # tied head, no bias
        return (logits, final_states) if return_states else logits


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_corpus(data_dir: str, name: str, device: str):
    """Loads one corpus's {train,val}.pt + {train,val}_doc_offsets.pt +
    meta.json, asserting (a) the shared-tokenizer fact this design's whole
    reasoning-vs-narrative contrast depends on, and (b) AUDIT FIX-3's
    eot_separated rebuild marker -- this harness REFUSES to run on the
    original boundary-free flat concatenations (run rebuild_lm_corpora_rd.py
    first; see CORPUS_DIRS' comment for why).

    Returns (train, val, meta, train_doc_offsets, val_doc_offsets) --
    doc_offsets are int64 tensors of document START positions in the
    corresponding token tensor (every document ends with EOT_TOKEN_ID)."""
    assert name in CORPUS_DIRS, f"unknown corpus {name!r}, expected one of {sorted(CORPUS_DIRS)}"
    d = os.path.join(data_dir, CORPUS_DIRS[name])
    assert os.path.isdir(d), (
        f"{name}: rebuilt corpus dir {d} not found -- AUDIT FIX-3 requires the EOT-separated "
        f"rebuild (rebuild_lm_corpora_rd.py). The original flat corpora are deliberately NOT "
        f"accepted (zero <|endoftext|> tokens; majority of windows span unrelated documents)."
    )
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    assert meta.get("vocab_size") == 50257 and meta.get("tokenizer") == "gpt2", (
        f"{name}: unexpected meta.json fields {meta} -- the gated data-transfer verification "
        f"(section 8: 'checks meta.json field values ... before Wave -1 can launch') failed. "
        f"Both corpora must share the GPT-2 tokenizer for the cross-corpus rank/loss comparison "
        f"to mean anything."
    )
    assert meta.get("eot_separated") is True, (
        f"{name}: meta.json lacks eot_separated=true -- this dir predates the AUDIT FIX-3 "
        f"rebuild; run rebuild_lm_corpora_rd.py."
    )
    train = torch.load(os.path.join(d, "train.pt"), map_location=device)
    val = torch.load(os.path.join(d, "val.pt"), map_location=device)
    train_offs = torch.load(os.path.join(d, "train_doc_offsets.pt"), map_location=device)
    val_offs = torch.load(os.path.join(d, "val_doc_offsets.pt"), map_location=device)
    assert train.dtype == torch.int64 and val.dtype == torch.int64
    assert train_offs.dtype == torch.int64 and val_offs.dtype == torch.int64
    # spot-check the rebuild's own invariant: every doc ends with EOT, so the
    # token right BEFORE each non-first doc start must be EOT
    for toks, offs, split in ((train, train_offs, "train"), (val, val_offs, "val")):
        probe = offs[1:min(65, offs.numel())]
        assert (toks[probe - 1] == EOT_TOKEN_ID).all(), (
            f"{name}/{split}: token before a doc start is not <|endoftext|> -- doc_offsets and "
            f"the token tensor disagree; the rebuild is corrupt.")
    return train, val, meta, train_offs, val_offs


def boundary_stats(starts: torch.Tensor, seq_len: int, doc_offsets: torch.Tensor) -> dict:
    """AUDIT FIX-3's contamination quantification for RANDOM (non-doc-
    aligned) windows: given window start indices and the corpus's document
    start offsets, computes
      - frac_windows_crossing: fraction of windows containing >= 1 document
        boundary (i.e. NOT fully inside one document), and
      - frac_tokens_cross_doc: fraction of window TOKENS belonging to a
        different document than the window's FIRST token ("the fraction of
        window tokens crossing a boundary" -- the audit's literal metric).
    Pure integer arithmetic on the offsets (searchsorted), no decode."""
    starts = starts.to(doc_offsets.device)
    ends = starts + seq_len                              # exclusive window end
    # doc index containing each window's first token / each window's span
    doc_of_start = torch.searchsorted(doc_offsets, starts, right=True) - 1
    # first doc boundary strictly after the window start
    next_boundary = torch.full_like(starts, torch.iinfo(torch.int64).max)
    nb_idx = doc_of_start + 1
    has_next = nb_idx < doc_offsets.numel()
    next_boundary[has_next] = doc_offsets[nb_idx[has_next]]
    crossing = next_boundary < ends
    tokens_in_first_doc = torch.minimum(next_boundary, ends) - starts
    frac_tokens_cross = 1.0 - (tokens_in_first_doc.float() / seq_len).mean().item()
    return {
        "n_windows": int(starts.numel()),
        "frac_windows_crossing": crossing.float().mean().item(),
        "frac_tokens_cross_doc": frac_tokens_cross,
    }


def get_batch(tokens: torch.Tensor, batch_size: int, seq_len: int, generator: torch.Generator,
               return_starts: bool = False):
    """Samples `batch_size` independent contiguous windows of length
    seq_len+1 from `tokens` (a flat 1D int64 tensor), zero initial state per
    window -- the standard small-LM training regime (random-window
    sampling, no cross-batch state carry-over; documents ARE EOT-separated
    in the rebuilt corpora, so a window that crosses a boundary carries an
    in-band <|endoftext|> signal -- and the caller can quantify the crossing
    rate via boundary_stats, AUDIT FIX-3). Returns (x, y), both (B,seq_len),
    y being x shifted by one position; with return_starts=True also returns
    the window start indices (for boundary_stats / window_digest)."""
    n = tokens.numel()
    assert n > seq_len + 1, f"corpus too small ({n} tokens) for seq_len={seq_len}"
    ix = torch.randint(0, n - seq_len - 1, (batch_size,), generator=generator, device=tokens.device)
    offs = torch.arange(seq_len + 1, device=tokens.device)
    idx = ix.unsqueeze(1) + offs.unsqueeze(0)          # (B, seq_len+1)
    window = tokens[idx]
    x, y = window[:, :-1].contiguous(), window[:, 1:].contiguous()
    return (x, y, ix) if return_starts else (x, y)


def get_lr(step: int, max_lr: float, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1) -> float:
    """Standard linear-warmup + cosine-decay schedule."""
    if step < warmup_steps:
        return max_lr * step / max(1, warmup_steps)
    if step >= total_steps:
        return max_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = max_lr * min_lr_ratio
    return floor + (max_lr - floor) * coeff


# ---------------------------------------------------------------------------
# Checkpoint instrumentation: whole-state rank stats + cross-corpus val loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_state_rank_stats(model: DeltaNetLM, tokens: torch.Tensor, doc_offsets: torch.Tensor,
                             n_docs: int, seq_len: int, generator: torch.Generator,
                             fracs=RANK_SAMPLE_FRACS) -> tuple[list, dict]:
    """Descriptive whole-state rank instrumentation (this build's brief:
    "entity-subspace doesn't apply -- whole-state effective/stable rank per
    head over sampled positions, split by document").

    AUDIT FIX-3 (2026-07-03, replacing the earlier document-PROXY build
    decision): windows are now DOCUMENT-ALIGNED -- each sampled window
    starts at a real document start (from the rebuilt corpus's
    doc_offsets), so "split by document" means genuine documents, not
    random-window proxies. Two disclosed imperfections, both LOGGED per
    record and summarized (never silent):
      - doc_len > L ("truncated"): the window covers only the document's
        first L tokens;
      - doc_len < L ("contaminated"): the window runs past the document's
        EOT into following documents (in-band EOT boundary signal present).

    "sampled positions" = the `fracs` fractional lengths of the seq_len
    window (0.25/0.5/0.75/1.0 by default); each is a FRESH forward pass
    from the document start (numerically identical to a continuation-
    chained call at that prefix length).

    AUDIT FIX-1: `generator` must be seeded from corpus_fixed_seed(...) so
    every training seed samples the SAME documents; the digest of the
    sampled doc-start indices is returned in the summary for cross-seed
    byte-identity verification.

    Returns (records, summary): records = raw per-(doc,frac,layer,head)
    list (per-document identity preserved, never collapsed here); summary
    = sampling digest + truncation/contamination fractions."""
    model.eval()
    n = tokens.numel()
    assert int(seq_len * min(fracs)) >= _MIN_KERNEL_T, (
        f"seq_len={seq_len} * min(fracs)={min(fracs)} = {seq_len * min(fracs)} < _MIN_KERNEL_T="
        f"{_MIN_KERNEL_T} -- the shortest rank-sample sub-window would crash the kernel's "
        f"backward (moot here since this is no_grad, but chunk_delta_rule's forward-only safety "
        f"at T<64 is also not guaranteed; keep every sampled length >= _MIN_KERNEL_T)."
    )
    # eligible docs: those whose START leaves room for a full seq_len window
    # inside the corpus (the window may still cross INTO following docs --
    # that is the logged "contaminated" case, not an exclusion criterion)
    eligible = doc_offsets[doc_offsets + seq_len + 1 <= n]
    assert eligible.numel() >= n_docs, \
        f"only {eligible.numel()} eligible docs for n_docs={n_docs} at seq_len={seq_len}"
    pick = torch.randint(0, eligible.numel(), (n_docs,), generator=generator,
                          device=doc_offsets.device)
    starts = eligible[pick]
    # per-document length (distance to the next doc start; last doc runs to corpus end)
    doc_idx_of = torch.searchsorted(doc_offsets, starts, right=True) - 1
    next_start = torch.full_like(starts, n)
    has_next = doc_idx_of + 1 < doc_offsets.numel()
    next_start[has_next] = doc_offsets[doc_idx_of[has_next] + 1]
    doc_lens = next_start - starts

    records = []
    for frac in fracs:
        L = max(_MIN_KERNEL_T, int(round(seq_len * frac)))
        offs = torch.arange(L, device=tokens.device)
        idx = starts.unsqueeze(1) + offs.unsqueeze(0)          # (n_docs, L)
        batch = tokens[idx]
        _, states = model(batch, return_states=True)
        for layer_idx, S in enumerate(states):                 # S: (n_docs, H, d, d)
            er = effective_rank(S)
            sr = stable_rank(S)
            H = S.shape[1]
            for doc_idx in range(n_docs):
                for head_idx in range(H):
                    records.append({
                        "doc": doc_idx, "frac": frac, "n_tokens": L,
                        "layer": layer_idx, "head": head_idx,
                        "doc_len": int(doc_lens[doc_idx].item()),
                        "window_within_doc": bool(doc_lens[doc_idx].item() >= L),
                        "effective_rank": er[doc_idx, head_idx].item(),
                        "stable_rank": sr[doc_idx, head_idx].item(),
                    })
    L_full = max(_MIN_KERNEL_T, int(round(seq_len * max(fracs))))
    summary = {
        "doc_start_digest": window_digest(starts),             # FIX-1 verification hook
        "n_docs_sampled": n_docs,
        "frac_docs_truncated": (doc_lens > L_full).float().mean().item(),
        "frac_docs_contaminated": (doc_lens < L_full).float().mean().item(),
        "mean_doc_len": doc_lens.float().mean().item(),
    }
    model.train()
    return records, summary


def summarize_rank_stats(records: list) -> dict:
    """Mean effective_rank per (layer, head, frac), keyed as a flat
    string -- convenience aggregate ON TOP OF the raw per-document records
    (never a replacement for them; see sample_state_rank_stats's "split by
    document" note)."""
    agg = {}
    for r in records:
        key = f"L{r['layer']}_H{r['head']}_f{r['frac']}"
        agg.setdefault(key, []).append(r["effective_rank"])
    return {k: round(sum(v) / len(v), 4) for k, v in agg.items()}


@torch.no_grad()
def eval_loss(model: DeltaNetLM, tokens: torch.Tensor, doc_offsets: torch.Tensor, n_batches: int,
              batch_size: int, seq_len: int, generator: torch.Generator) -> tuple[float, dict]:
    """Cross-entropy val loss, capped batch size / batch count (house VRAM
    rule: eval batches capped independently of the train batch -- the 50K-
    vocab logits tensor is the VRAM bottleneck, not model activations).

    AUDIT FIX-1: `generator` must be seeded from corpus_fixed_seed(...) so
    every training seed evaluates on the SAME windows. Returns
    (mean_loss, info) where info carries the FIX-1 verification digest of
    the drawn window starts and the FIX-3 boundary-contamination stats."""
    model.eval()
    total, count = 0.0, 0
    all_starts = []
    for _ in range(n_batches):
        x, y, starts = get_batch(tokens, batch_size, seq_len, generator, return_starts=True)
        all_starts.append(starts)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        total += loss.item()
        count += 1
    model.train()
    starts_cat = torch.cat(all_starts)
    info = {"eval_window_digest": window_digest(starts_cat),
            "boundary": boundary_stats(starts_cat, seq_len, doc_offsets)}
    return total / max(1, count), info


# ---------------------------------------------------------------------------
# Result assembly + incremental dump (house convention)
# ---------------------------------------------------------------------------

CLAIM_TIER = ("descriptive+interventional (Wave 2 / RD-2, DELTANET_REALDATA_DESIGN.md section 6.3) "
              "-- NOT premise-conditional causal (that tier is Wave 1 / RD-1 only, section 5's C16 "
              "premise machinery). No Gram-deviation / salvage-ratio / alignment-cosine premise gating "
              "applies to this arm's metrics (section 14.7).")


def _dump(path, obj):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"  (incremental write failed, non-fatal: {e!r})", flush=True)


def set_and_log_tf32() -> dict:
    """Audit non-blocking item (2026-07-03): explicitly SET (never inherit)
    the TF32 flags, and return them for the result JSON. Values chosen to
    MATCH the state the on-box cost calibration was measured under
    (torch 2.12.1 defaults on this box: matmul.allow_tf32=False,
    cudnn.allow_tf32=True -- read directly, 2026-07-03), so the measured
    ~438K tok/s number stays valid; an explicit set also means a future
    torch upgrade changing the default cannot silently change numerics
    between runs of one wave."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    return {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def _assemble_result(args, run_name, other_corpus, trajectory, checkpoints, n_params, t0,
                      timed_out, steps_completed, complete, n_skipped: int = 0,
                      train_contamination: dict | None = None) -> dict:
    result = {
        "run_name": run_name, "claim_tier": CLAIM_TIER,
        "corpus": args.corpus, "other_corpus": other_corpus, "seed": args.seed,
        "eval_window_seed_policy": "corpus_fixed (AUDIT FIX-1: crc32(corpus)+step, independent of seed)",
        "d_model": args.d_model, "d_state": args.d_state, "n_layers": args.n_layers,
        "conv_size": args.conv_size, "num_heads": args.num_heads, "ffn_mult": args.ffn_mult,
        "seq_len": args.seq_len, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "warmup_steps": args.warmup_steps,
        "steps": args.steps, "steps_completed": steps_completed,
        "complete": complete,                                    # crash-safety sentinel
        "n_skipped_steps": n_skipped,                            # Wave 1 parity (audit item)
        "skip_rate": (n_skipped / steps_completed) if steps_completed > 0 else 0.0,
        "train_window_contamination": train_contamination or {},  # AUDIT FIX-3
        "tf32": set_and_log_tf32(),
        "n_params": n_params,
        "trajectory": trajectory, "checkpoints": checkpoints,
        "wall_s": time.time() - t0, "timed_out": timed_out,
    }
    if checkpoints:
        result["final_step"] = checkpoints[-1]["step"]
        result["final_checkpoint_path"] = checkpoints[-1].get("checkpoint_path")
    return result


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(model: DeltaNetLM, args, train_tokens: torch.Tensor, train_doc_offsets: torch.Tensor,
          val_same: tuple, val_other: tuple, other_corpus: str, device: str, run_name: str,
          out_path: str | None = None, ckpt_dir: str | None = None, timeout_s: float | None = None) -> dict:
    """val_same / val_other: (val_tokens, val_doc_offsets) per corpus.

    Seeding discipline (AUDIT FIX-1): the TRAINING batch sampler is seeded
    from args.seed (each seed must see different training data -- that IS
    the seed axis); every EVAL-SIDE sampler (val loss windows, rank-stat
    documents) is seeded from corpus_fixed_seed(corpus)+step, fully
    independent of args.seed, so cross-seed aggregates compare numbers
    measured on IDENTICAL held-out windows. Digests of every drawn window
    set are logged per checkpoint for byte-identity verification."""
    t0 = time.time()
    tf32_state = set_and_log_tf32()            # SET before any training math, logged in the result
    print(f"  tf32: {tf32_state}", flush=True)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    val_tokens_same, val_offs_same = val_same
    val_tokens_other, val_offs_other = val_other

    trajectory, checkpoints = [], []
    timed_out = False
    steps_completed = 0
    n_skipped = 0                              # non-finite-grad skipped steps (Wave 1 parity)
    # FIX-3: training-window contamination accounting (cheap searchsorted
    # per step) -- accumulated over ALL training steps, reported in the
    # result JSON so the Wave C/D analysis can condition on it
    train_windows_total = 0
    train_windows_crossing = 0.0
    train_tokens_cross_frac_sum = 0.0

    for step in range(1, args.steps + 1):
        steps_completed = step
        lr = get_lr(step, args.lr, args.warmup_steps, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y, starts = get_batch(train_tokens, args.batch_size, args.seq_len, gen, return_starts=True)
        bs_stats = boundary_stats(starts, args.seq_len, train_doc_offsets)
        train_windows_total += bs_stats["n_windows"]
        train_windows_crossing += bs_stats["frac_windows_crossing"] * bs_stats["n_windows"]
        train_tokens_cross_frac_sum += bs_stats["frac_tokens_cross_doc"] * bs_stats["n_windows"]

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1

        if step % args.log_every == 0 or step == 1:
            trajectory.append({"step": step, "loss": loss.item(), "lr": lr, "grad_finite": finite,
                                "skip_rate_so_far": n_skipped / step})
            print(f"  step {step:6d}  loss {loss.item():.4f}  lr {lr:.2e}{'  [NON-FINITE GRAD, SKIPPED]' if not finite else ''}",
                  flush=True)

        if step % args.ckpt_every == 0 or step == args.steps:
            # FIX-1: eval/rank samplers seeded from the CORPUS, never args.seed
            eval_gen_same = torch.Generator(device=device).manual_seed(
                corpus_fixed_seed(args.corpus) + 10_000 + step)
            eval_gen_other = torch.Generator(device=device).manual_seed(
                corpus_fixed_seed(other_corpus) + 10_000 + step)
            val_loss_same, eval_info_same = eval_loss(model, val_tokens_same, val_offs_same,
                                                       args.eval_batches, args.eval_batch_size,
                                                       args.seq_len, eval_gen_same)
            val_loss_other, eval_info_other = eval_loss(model, val_tokens_other, val_offs_other,
                                                         args.eval_batches, args.eval_batch_size,
                                                         args.seq_len, eval_gen_other)

            rank_gen_same = torch.Generator(device=device).manual_seed(
                corpus_fixed_seed(args.corpus) + 20_000 + step)
            rank_gen_other = torch.Generator(device=device).manual_seed(
                corpus_fixed_seed(other_corpus) + 20_000 + step)
            rank_same, rank_sum_same = sample_state_rank_stats(
                model, val_tokens_same, val_offs_same, args.rank_sample_docs, args.seq_len, rank_gen_same)
            rank_other, rank_sum_other = sample_state_rank_stats(
                model, val_tokens_other, val_offs_other, args.rank_sample_docs, args.seq_len, rank_gen_other)

            ckpt_path = None
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"{run_name}_step{step}.pt")
                torch.save({"step": step, "model_state_dict": model.state_dict(),
                            "config": model.config(), "corpus": args.corpus, "seed": args.seed,
                            "run_name": run_name}, ckpt_path)

            res = {
                "step": step,
                "val_loss": {args.corpus: val_loss_same, other_corpus: val_loss_other},
                "eval_windows": {args.corpus: eval_info_same, other_corpus: eval_info_other},
                "rank_stats_summary": {args.corpus: summarize_rank_stats(rank_same),
                                        other_corpus: summarize_rank_stats(rank_other)},
                "rank_sampling": {args.corpus: rank_sum_same, other_corpus: rank_sum_other},
                "rank_stats_raw": {args.corpus: rank_same, other_corpus: rank_other},
                "checkpoint_path": ckpt_path,
            }
            checkpoints.append(res)
            partial = _assemble_result(args, run_name, other_corpus, trajectory, checkpoints, n_params,
                                        t0, timed_out=False, steps_completed=steps_completed,
                                        complete=False, n_skipped=n_skipped,
                                        train_contamination=_train_contamination(
                                            train_windows_total, train_windows_crossing,
                                            train_tokens_cross_frac_sum))
            _dump(out_path, partial)
            print(f"  [checkpoint step {step}] val_loss[{args.corpus}]={val_loss_same:.4f}  "
                  f"val_loss[{other_corpus}]={val_loss_other:.4f}  "
                  f"rank_L0_H0_f1.0[{args.corpus}]="
                  f"{res['rank_stats_summary'][args.corpus].get('L0_H0_f1.0', float('nan')):.3f}  "
                  f"eval_digest[{args.corpus}]={eval_info_same['eval_window_digest']}", flush=True)

        if timeout_s is not None and time.time() - t0 > timeout_s:
            print(f"  internal timeout ({timeout_s}s) reached at step {step}; stopping early", flush=True)
            timed_out = True
            break

    return _assemble_result(args, run_name, other_corpus, trajectory, checkpoints, n_params, t0,
                             timed_out, steps_completed,
                             complete=(not timed_out and steps_completed >= args.steps),
                             n_skipped=n_skipped,
                             train_contamination=_train_contamination(
                                 train_windows_total, train_windows_crossing,
                                 train_tokens_cross_frac_sum))


def _train_contamination(total: int, crossing_weighted: float, tokens_cross_weighted: float) -> dict:
    """FIX-3: aggregate training-window contamination over the whole run."""
    if total == 0:
        return {"n_train_windows": 0}
    return {"n_train_windows": total,
            "frac_windows_crossing": crossing_weighted / total,
            "frac_tokens_cross_doc": tokens_cross_weighted / total}


# ---------------------------------------------------------------------------
# Smoke gate. REQUIRES CUDA (chunk_delta_rule has no CPU path). Uses
# SYNTHETIC random token data (NOT the real corpus files) so this suite is
# fast, portable, and independent of /data being mounted -- data-integrity
# is a separate, already-gated concern (section 8's meta.json field check,
# exercised by load_corpus's own assert, not re-tested here).
# ---------------------------------------------------------------------------

def smoke(device: str):
    print("=" * 60 + "\n  LM_PRETRAIN_RD SMOKE GATE\n" + "=" * 60)
    assert device == "cuda", "chunk_delta_rule has no CPU path"
    torch.manual_seed(0)
    V = 500   # tiny synthetic vocab -- shape/gradient checks don't need the real 50,257

    print("\n[1] forward / backward / grad-finite at LM shapes (T=_MIN_KERNEL_T=128, d_state=64, "
          "both n_layers in {1,2})")
    for n_layers in (1, 2):
        model = DeltaNetLM(V, d_model=64, d_state=64, n_layers=n_layers, conv_size=4).to(device)
        x = torch.randint(0, V, (4, _MIN_KERNEL_T), device=device)
        y = torch.randint(0, V, (4, _MIN_KERNEL_T), device=device)
        logits = model(x)
        assert logits.shape == (4, _MIN_KERNEL_T, V)
        assert torch.isfinite(logits).all(), "non-finite logits"
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        loss.backward()
        n_none = [n for n, p in model.named_parameters() if p.grad is None]
        assert not n_none, f"no grad for: {n_none}"
        for n, p in model.named_parameters():
            assert torch.isfinite(p.grad).all(), f"non-finite grad at {n}"
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  n_layers={n_layers}: forward {tuple(logits.shape)}, loss {loss.item():.4f}, "
              f"{n_params} params, ALL grads finite")

    print("\n[1b] param count at the REAL probe-tier config (d_model=256, d_state=64) matches "
          "section 4.1's ~14M estimate (n_layers=2) / ~13.5M (n_layers=1), both clear the 10M "
          "CLAUDE.md floor")
    for n_layers, lo, hi in ((1, 10_000_000, 15_000_000), (2, 10_000_000, 16_000_000)):
        m = DeltaNetLM(50257, d_model=256, d_state=64, n_layers=n_layers, conv_size=4)
        n_params = sum(p.numel() for p in m.parameters())
        assert lo <= n_params <= hi, f"n_layers={n_layers}: {n_params} params outside [{lo},{hi}]"
        print(f"  n_layers={n_layers}: {n_params:,} params (in [{lo:,},{hi:,}])")

    print("\n[1c] AUDIT FIX-2 regression: INITIAL loss at the REAL vocab must be ~ln(50257)~=10.8, "
          "NOT the ~202 nats the PyTorch-default N(0,1) tied-embedding init produced -- assert < 12 "
          "so this class of init bug is caught here in the future")
    with torch.no_grad():
        m_init = DeltaNetLM(50257, d_model=256, d_state=64, n_layers=2, conv_size=4).to(device)
        x_i = torch.randint(0, 50257, (4, _MIN_KERNEL_T), device=device)
        y_i = torch.randint(0, 50257, (4, _MIN_KERNEL_T), device=device)
        logits_i = m_init(x_i)
        init_loss = F.cross_entropy(logits_i.reshape(-1, 50257), y_i.reshape(-1)).item()
    assert init_loss < 12.0, (
        f"FIX-2 REGRESSION: initial loss {init_loss:.2f} >= 12 (expected ~ln(50257)~=10.8) -- "
        f"the tied-embedding init is mis-scaled again")
    del m_init, logits_i
    print(f"  initial loss at d_model=256/n_layers=2/V=50257: {init_loss:.3f} (< 12, ~ln(V)=10.82)")

    print("\n[2] _SAFE_D_STATE guard has teeth (num_heads=2 at d_state=64 -> head_dim=32, unsafe)")
    guard_raised = False
    try:
        DeltaNetLMMixer(d_model=64, d_state=64, num_heads=2)
    except AssertionError:
        guard_raised = True
    assert guard_raised, "_SAFE_D_STATE guard FAILED to reject head_dim=32"
    print("  head_dim=32 correctly REJECTED")

    print("\n[3] mini end-to-end training loop (synthetic EOT-separated corpus, a handful of REAL "
          "SGD steps) stays finite; checkpoint carries val_loss for BOTH corpora + doc-aligned "
          "rank stats + eval-window digests + contamination stats")
    torch.manual_seed(1)
    n_tok = 200_000
    seq_len_smoke = _MIN_KERNEL_T * 4

    def synth_eot_corpus(n_tok, mean_doc_len, gen_seed):
        """Synthetic corpus with REAL EOT structure: random tokens with EOT
        every ~mean_doc_len, plus the matching doc_offsets tensor -- so the
        smoke exercises the same (tokens, doc_offsets) contract load_corpus
        provides."""
        g = torch.Generator(device=device).manual_seed(gen_seed)
        toks = torch.randint(0, V, (n_tok,), generator=g, device=device)
        # place EOTs at ~Poisson-ish spacing (uniform in [0.5, 1.5]*mean)
        pos, offsets = 0, [0]
        while True:
            step_len = int(mean_doc_len * (0.5 + torch.rand(1, generator=g, device=device).item()))
            pos += max(2, step_len)
            if pos >= n_tok - 1:
                break
            toks[pos] = EOT_TOKEN_ID % V if V <= EOT_TOKEN_ID else EOT_TOKEN_ID
            offsets.append(pos + 1)
        return toks, torch.tensor(offsets, dtype=torch.int64, device=device)

    corpus_a, offs_a = synth_eot_corpus(n_tok, 300, 11)
    corpus_b, offs_b = synth_eot_corpus(n_tok, 800, 22)
    model = DeltaNetLM(V, d_model=64, d_state=64, n_layers=1, conv_size=4).to(device)

    class Args:
        pass
    args = Args()
    args.corpus, args.seed = "openr1", 0
    args.d_model, args.d_state, args.n_layers, args.conv_size, args.num_heads, args.ffn_mult = 64, 64, 1, 4, 1, 4
    # seq_len = 4x _MIN_KERNEL_T so the smallest RANK_SAMPLE_FRACS entry (0.25) lands EXACTLY at
    # _MIN_KERNEL_T -- sample_state_rank_stats's own assert requires seq_len*min(fracs) >=
    # _MIN_KERNEL_T (a labeling-accuracy guard: at a smaller seq_len, the max(_MIN_KERNEL_T, ...)
    # clamp inside that function would silently make the "frac=0.25" position label wrong).
    args.seq_len, args.batch_size = seq_len_smoke, 4
    args.lr, args.weight_decay, args.warmup_steps = 3e-4, 0.0, 2
    args.eval_batches, args.eval_batch_size = 2, 4
    args.rank_sample_docs = 3
    args.steps, args.log_every, args.ckpt_every = 8, 4, 8

    result = train(model, args, corpus_a, offs_a, (corpus_a, offs_a), (corpus_b, offs_b),
                    "wikitext", device, "smoke_run", out_path=None, ckpt_dir=None)
    assert result["steps_completed"] == 8 and result["complete"] is True
    assert result["checkpoints"], "no checkpoint written"
    ck = result["checkpoints"][-1]
    assert "openr1" in ck["val_loss"] and "wikitext" in ck["val_loss"]
    assert ck["val_loss"]["openr1"] == ck["val_loss"]["openr1"], "NaN val_loss"
    assert ck["val_loss"]["wikitext"] == ck["val_loss"]["wikitext"], "NaN val_loss"
    assert len(ck["rank_stats_raw"]["openr1"]) == 3 * len(RANK_SAMPLE_FRACS) * 1 * 1  # docs*fracs*layers*heads
    assert result["claim_tier"] == CLAIM_TIER
    assert "n_skipped_steps" in result and "skip_rate" in result             # Wave 1 parity item
    assert result["train_window_contamination"]["n_train_windows"] == 8 * 4  # FIX-3 accounting
    assert 0.0 <= result["train_window_contamination"]["frac_tokens_cross_doc"] <= 1.0
    assert "eval_window_digest" in ck["eval_windows"]["openr1"]              # FIX-1 hook present
    assert "boundary" in ck["eval_windows"]["openr1"]
    assert "doc_start_digest" in ck["rank_sampling"]["openr1"]
    # doc-alignment sanity: corpus_a's mean doc len ~300 < the full 512 window, so its rank
    # sampling should show contamination; every record carries doc_len/window_within_doc
    assert all("doc_len" in r and "window_within_doc" in r for r in ck["rank_stats_raw"]["openr1"])
    print(f"  8-step mini run: complete={result['complete']}, skip_rate={result['skip_rate']:.4%}, "
          f"val_loss[openr1]={ck['val_loss']['openr1']:.4f}, val_loss[wikitext]={ck['val_loss']['wikitext']:.4f}, "
          f"train frac_tokens_cross_doc={result['train_window_contamination']['frac_tokens_cross_doc']:.3f}, "
          f"rank sampling frac_docs_contaminated[openr1]={ck['rank_sampling']['openr1']['frac_docs_contaminated']:.2f}")

    print("\n[3b] AUDIT FIX-1 regression: eval/rank window sets are IDENTICAL across two different "
          "training seeds (digest equality on every checkpoint's every corpus), and the TRAINING "
          "batch stream is NOT (sensitivity control -- the training seed axis must still exist)")
    digests = {}
    train_losses = {}
    for seed_probe in (0, 1):
        torch.manual_seed(seed_probe)
        args.seed = seed_probe
        m_p = DeltaNetLM(V, d_model=64, d_state=64, n_layers=1, conv_size=4).to(device)
        r_p = train(m_p, args, corpus_a, offs_a, (corpus_a, offs_a), (corpus_b, offs_b),
                     "wikitext", device, f"smoke_seedprobe{seed_probe}", out_path=None, ckpt_dir=None)
        ck_p = r_p["checkpoints"][-1]
        digests[seed_probe] = (ck_p["eval_windows"]["openr1"]["eval_window_digest"],
                                ck_p["eval_windows"]["wikitext"]["eval_window_digest"],
                                ck_p["rank_sampling"]["openr1"]["doc_start_digest"],
                                ck_p["rank_sampling"]["wikitext"]["doc_start_digest"])
        train_losses[seed_probe] = [t["loss"] for t in r_p["trajectory"]]
    args.seed = 0
    assert digests[0] == digests[1], (
        f"FIX-1 REGRESSION: eval/rank window digests differ across training seeds: "
        f"{digests[0]} vs {digests[1]}")
    assert train_losses[0] != train_losses[1], (
        "FIX-1 sensitivity control FAILED: two different training seeds produced identical "
        "training-loss trajectories -- the training stream is not actually seed-dependent, "
        "so the digest-equality check above proves nothing")
    print(f"  digests (eval x2 corpora, rank x2 corpora) IDENTICAL across seeds 0/1: {digests[0]}")
    print(f"  training trajectories DIFFER across seeds (sensitivity control holds)")

    print("\n[4] checkpoint save/load round-trip (torch.save -> fresh model -> load_state_dict -> "
          "identical forward pass)")
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="lm_rd_smoke_")
    model2 = DeltaNetLM(V, d_model=64, d_state=64, n_layers=1, conv_size=4).to(device)
    # train() needs gradients (it calls loss.backward() internally) -- ONLY the final
    # round-trip COMPARISON below is a no_grad forward-only check, not this training call.
    result2 = train(model2, args, corpus_a, offs_a, (corpus_a, offs_a), (corpus_b, offs_b),
                     "wikitext", device, "smoke_ckpt", out_path=None, ckpt_dir=tmpdir)
    ckpt_path = result2["checkpoints"][-1]["checkpoint_path"]
    assert ckpt_path and os.path.exists(ckpt_path), "checkpoint file was not written"
    loaded = torch.load(ckpt_path, map_location=device)
    assert loaded["config"] == model2.config()
    model3 = DeltaNetLM(**loaded["config"]).to(device)
    model3.load_state_dict(loaded["model_state_dict"])
    model2.eval()
    model3.eval()
    with torch.no_grad():
        x_probe = torch.randint(0, V, (2, _MIN_KERNEL_T), device=device)
        logits2 = model2(x_probe)
        logits3 = model3(x_probe)
    assert torch.equal(logits2, logits3), "checkpoint round-trip mismatch: reloaded model differs"
    print(f"  round-trip: {ckpt_path} loaded into a fresh model, logits BIT-IDENTICAL "
          f"(max abs diff {(logits2 - logits3).abs().max().item():.2e})")

    print("\n[5] boundary_stats sanity on a HAND-CONSTRUCTED case (pure arithmetic): known doc "
          "layout, known window placement, exact expected fractions")
    doc_offs = torch.tensor([0, 100, 250, 400], dtype=torch.int64, device=device)
    # window A: [10, 60)  -- fully inside doc0 (no crossing)
    # window B: [80, 130) -- 20 tokens in doc0, 30 in doc1 (crossing; 30/50 = 0.6 cross-doc)
    starts_h = torch.tensor([10, 80], dtype=torch.int64, device=device)
    bs = boundary_stats(starts_h, 50, doc_offs)
    assert bs["n_windows"] == 2
    assert abs(bs["frac_windows_crossing"] - 0.5) < 1e-6, bs
    assert abs(bs["frac_tokens_cross_doc"] - (0.0 + 0.6) / 2) < 1e-6, bs
    print(f"  hand case: frac_windows_crossing={bs['frac_windows_crossing']} (=0.5), "
          f"frac_tokens_cross_doc={bs['frac_tokens_cross_doc']:.3f} (=0.3) -- exact match")

    print("\n" + "=" * 60 + "\n  ALL LM_PRETRAIN_RD SMOKE CHECKS PASSED\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--corpus", choices=sorted(CORPUS_DIRS), default=None,
                     help="training corpus. REQUIRED for a real run (no default -- section 6.1's "
                          "whole point is the two-corpus contrast, an accidental default risks a "
                          "silent same-corpus rerun).")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-state", type=int, default=64, choices=[64, 128])
    ap.add_argument("--n-layers", type=int, default=2, choices=[1, 2])
    ap.add_argument("--conv-size", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=1)
    ap.add_argument("--ffn-mult", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=16,
                     help="CAPPED independently of --batch-size (house VRAM rule: eval can OOM "
                          "even if training fits -- the logits tensor is the bottleneck).")
    ap.add_argument("--eval-batches", type=int, default=8, help="capped eval batch COUNT per corpus per checkpoint")
    ap.add_argument("--rank-sample-docs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=1000,
                     help="section 8's build requirement: mid-training eval checkpoints every <=2000 steps")
    ap.add_argument("--ckpt-dir", type=str, default=None,
                     help="directory to torch.save model checkpoints -- REQUIRED for a real run "
                          "(the intervention script needs them); omit only for a throwaway/dry run.")
    ap.add_argument("--internal-timeout", type=float, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.smoke:
        smoke(device)
        return

    assert device == "cuda", "lm_pretrain_rd requires CUDA for real training (chunk_delta_rule has no CPU path)"
    assert args.corpus is not None, "--corpus is required for a real (non-smoke) run"
    if args.ckpt_every > 2000:
        print(f"WARNING: --ckpt-every={args.ckpt_every} > 2000: violates section 8's build requirement.", flush=True)
    if args.ckpt_dir is None:
        print("WARNING: --ckpt-dir not given -- no model checkpoints will be saved; "
              "lm_intervene_rd.py will have nothing to load.", flush=True)
    assert args.seq_len >= _MIN_KERNEL_T, \
        f"--seq-len={args.seq_len} < _MIN_KERNEL_T={_MIN_KERNEL_T} (F15-LM measured floor)"

    other_corpus = OTHER_CORPUS[args.corpus]
    train_tokens, val_same, meta_same, train_offs, val_offs_same = load_corpus(args.data_dir, args.corpus, device)
    _, val_other, meta_other, _, val_offs_other = load_corpus(args.data_dir, other_corpus, device)
    print(f"corpus={args.corpus} (train {train_tokens.numel():,} tok / {train_offs.numel():,} docs, "
          f"val {val_same.numel():,} tok / {val_offs_same.numel():,} docs)  "
          f"other_corpus={other_corpus} (val {val_other.numel():,} tok / {val_offs_other.numel():,} docs)",
          flush=True)

    model = DeltaNetLM(meta_same["vocab_size"], d_model=args.d_model, d_state=args.d_state,
                        n_layers=args.n_layers, conv_size=args.conv_size, num_heads=args.num_heads,
                        ffn_mult=args.ffn_mult).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    run_name = f"lmC_{args.corpus}_dm{args.d_model}_ds{args.d_state}_L{args.n_layers}_s{args.seed}"
    print(f"run_name={run_name}  d_model={args.d_model} d_state={args.d_state} n_layers={args.n_layers} "
          f"seq_len={args.seq_len} batch_size={args.batch_size} steps={args.steps} params={n_params} "
          f"device={device}", flush=True)

    result = train(model, args, train_tokens, train_offs, (val_same, val_offs_same),
                    (val_other, val_offs_other), other_corpus, device, run_name,
                    out_path=args.out, ckpt_dir=args.ckpt_dir, timeout_s=args.internal_timeout)

    summary = {k: v for k, v in result.items() if k not in ("trajectory", "checkpoints")}
    print("\nRESULT SUMMARY:", json.dumps(summary, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
