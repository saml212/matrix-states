"""Stage 0 model variants — see STAGE0_DESIGN.md section 4 (candidate
interventions) and section 6.3 (build requirements).

Does NOT modify or import-and-mutate model_v4.py. model_v4.BindingEncoder is
the audited object under diagnosis (Task E's model_e.py reuses it verbatim);
CLAUDE.md's hard constraint is that its default construction stay bit-
identical. Rather than subclassing it (which would require overriding
private init/forward internals in ways that risk drifting the parent's
behavior under future model_v4.py edits), this file REIMPLEMENTS the same
forward pass in a new class, `BindingEncoderS0`, whose `variant="baseline"`
preset reproduces model_v4.BindingEncoder's architecture and init
DISTRIBUTIONS exactly (verified by run_stage0.py's smoke gate, section
[3]: parameter names/shapes/counts are asserted identical). This is the
"new file" reading of the task's constraint 5, chosen because architecture-
touching candidates (2's row_out spectral init, 4's self-attention, 7's
per-row row_out) are easiest to reason about and audit as one small,
self-contained, from-scratch module rather than as monkey-patched overrides
of an existing nn.Module's __init__.

Self-contained apart from rank_utils and task_d (both same-directory, reused
per the design doc's explicit instruction — candidate 2 reuses
task_d._random_directions' QR-orthogonal machinery verbatim). No src/
imports.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import task_d as td
from model_v4 import MatrixMemoryModel        # reuse the pure, static unbind() only
from rank_utils import truncate_to_rank


def _orthogonal_rows(d: int, h: int, target_norm: float) -> torch.Tensor:
    """QR-orthonormal (d, h) row_queries init (candidate 2, STAGE0_DESIGN.md
    section 4 candidate 2), reusing task_d.py::_random_directions' EXACT
    machinery per the design doc's explicit instruction (cheap code reuse, no
    new numerical-stability surface). For d <= h this gives exactly
    orthonormal rows (a complete orthonormal basis of R^h when d == h, per
    the design's d=64/h=64 zero-slack discussion, MINOR-9); for d > h it
    falls back to task_d's own documented Gaussian fallback (not exercised by
    Stage 0's core d in {16,32,64} <= h=64 grid, kept safe for completeness).

    Rows are rescaled from unit norm to `target_norm` so the intervention
    isolates DIRECTION (collision structure) without also changing the
    row_queries' overall SCALE relative to model_v4's default
    N(0, 0.02^2) init — a confound the design doc doesn't explicitly flag but
    is the minimal-risk reading, since collision (not magnitude) is the
    targeted mechanism (STAGE0_DESIGN.md section 2.2).
    """
    gen = torch.Generator().manual_seed(torch.initial_seed() % (2 ** 31))
    rows = td._random_directions(1, d, h, True, gen, "cpu", torch.float32)[0]  # (d, h) unit rows
    return rows * target_norm


class PerRowLinear(nn.Module):
    """Candidate 7 (STAGE0_DESIGN.md section 4, MAJOR-7 / the H_cap isolator):
    `d` INDEPENDENT (h -> d) projections, replacing model_v4's single shared
    `nn.Linear(h, d)`. Removes the `rank(Z) <= h+1` architectural ceiling
    (Z can reach rank d) while leaving row_queries and the reader —the entire
    H_collision surface— untouched: `Z[b,i,:] = W_i @ q[b,i,:] + b_i`, each
    row with its OWN `W_i in R^{d x h}`, `b_i in R^d` (matches the design's
    math exactly — the shared layer's bias is one vector broadcast to every
    row; here each row gets its own).

    Params: `d*(d*h) + d*d = d^2*(h+1)` vs the shared layer's `d*(h+1)` —
    ~32x at d=32/h=64 (2,080 -> 66,560, matches the design doc's derivation,
    section 5). Held as a fallback/Wave-B-only arm (NOT screened in Wave A)
    since H_cap provably does not bind at d=32/64 (section 2.1) — available
    here as a non-default `row_out_mode="per_row"` variant per the task
    brief's explicit instruction.
    """

    def __init__(self, d: int, h: int, orthogonal_init: bool = False):
        super().__init__()
        rows_w, rows_b = [], []
        for _ in range(d):
            w = torch.empty(d, h)
            nn.init.kaiming_uniform_(w, a=5 ** 0.5)      # matches nn.Linear(h, d)'s default, per row
            if orthogonal_init:
                nn.init.orthogonal_(w)                    # Saxe et al. 2014 (candidate 2's "spectral" half, if combined)
            b = torch.empty(d)
            bound = 1.0 / (h ** 0.5)
            nn.init.uniform_(b, -bound, bound)             # matches nn.Linear(h, d)'s default bias init, per row
            rows_w.append(w)
            rows_b.append(b)
        self.weight = nn.Parameter(torch.stack(rows_w))    # (d_row, d_out, h)
        self.bias = nn.Parameter(torch.stack(rows_b))       # (d_row, d_out)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: (B, d, h) -> Z: (B, d, d);  Z[b,i,:] = weight[i] @ q[b,i,:] + bias[i]
        return torch.einsum("bih,ioh->bio", q, self.weight) + self.bias


class BindingEncoderS0(nn.Module):
    """Reimplements model_v4.BindingEncoder's forward pass with additive
    variant flags (STAGE0_DESIGN.md section 4). All flags default to the
    model_v4-identical behavior — see the module docstring above and
    `VARIANTS["baseline"]` below.

    NOTE on reproducibility (corrected per audit MINOR-3, verified by
    execution across 3 seeds): under a given global seed, baseline-preset
    construction IS bit-identical to model_v4.BindingEncoder — the default
    path creates the same modules in the same order (in_proj, encoder stack,
    row_queries via torch.randn, reader, row_norm, row_out) and the disabled
    variant branches consume no RNG draws, so the global RNG stream is
    consumed in exactly the same sequence. Same names, shapes, counts, AND
    values: Arm 0 (Wave 0) is a true bit-level rerun of the unmodified
    baseline at the same seed, not merely a distribution-level one.
    """

    def __init__(self, d: int, h: int = 64, n_layers: int = 3, n_heads: int = 4,
                 n_refine: int = 1, row_query_init: str = "normal",
                 row_out_init: str = "default", row_out_mode: str = "shared",
                 use_row_self_attn: bool = False):
        super().__init__()
        assert row_query_init in ("normal", "orthogonal"), row_query_init
        assert row_out_init in ("default", "orthogonal"), row_out_init
        assert row_out_mode in ("shared", "per_row"), row_out_mode
        self.d, self.h, self.n_refine = d, h, n_refine
        self.use_row_self_attn = use_row_self_attn

        self.in_proj = nn.Linear(2 * d, h)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=n_heads, dim_feedforward=4 * h,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                             enable_nested_tensor=False)

        if row_query_init == "normal":
            rq = torch.randn(d, h) * 0.02                          # model_v4-identical
        else:                                                       # candidate 2
            rq = _orthogonal_rows(d, h, target_norm=0.02 * (h ** 0.5))
        self.row_queries = nn.Parameter(rq)

        self.reader = nn.MultiheadAttention(h, n_heads, batch_first=True, dropout=0.0)
        self.row_norm = nn.LayerNorm(h)

        if use_row_self_attn:                                       # candidate 4
            self.row_self_attn = nn.MultiheadAttention(h, n_heads, batch_first=True, dropout=0.0)
            self.row_self_norm = nn.LayerNorm(h)

        if row_out_mode == "shared":
            self.row_out = nn.Linear(h, d)                          # model_v4-identical default
            if row_out_init == "orthogonal":                         # candidate 2's "spectral" half
                nn.init.orthogonal_(self.row_out.weight)             # Saxe et al. 2014 (arXiv:1312.6120)
        else:                                                        # candidate 7 (fallback / non-default)
            self.row_out = PerRowLinear(d, h, orthogonal_init=(row_out_init == "orthogonal"))

    def forward(self, keys: torch.Tensor, values: torch.Tensor,
                return_intermediate: bool = False):
        # keys, values: (B, K, d)
        B = keys.shape[0]
        tok = self.in_proj(torch.cat([keys, values], dim=-1))       # (B, K, h)
        mem = self.encoder(tok)                                     # (B, K, h)
        q = self.row_queries.unsqueeze(0).expand(B, self.d, self.h)  # (B, d, h)
        for _ in range(self.n_refine):
            read, _ = self.reader(q, mem, mem, need_weights=False)   # (B, d, h)
            q = self.row_norm(q + read)                              # residual refine
        if self.use_row_self_attn:
            # Candidate 4 (STAGE0_DESIGN.md section 4): one Slot-Attention-
            # style (Locatello et al. 2020) self-attention pass among the d
            # row outputs, AFTER the cross-attend refine loop, so near-
            # duplicate row identities can differentiate via competition
            # rather than independently reading the same shared memory.
            self_read, _ = self.row_self_attn(q, q, q, need_weights=False)
            q = self.row_self_norm(q + self_read)
        Z = self.row_out(q)                                         # (B, d, d): rows of Z
        return (Z, q) if return_intermediate else Z


def mup_lr_mult(h: int, mup_h_base: int) -> float:
    """Candidate 3's LR half of "muP-style init/LR scaling" (STAGE0_DESIGN.md
    section 4 candidate 3): the row_out (output-layer) LR is scaled by
    `mup_h_base / h` relative to a base width, adapted from Yang & Hu 2021
    (arXiv:2011.14522) / Yang et al. 2022 (arXiv:2203.03466, Tensor Programs
    V — the output-layer LR-scaling rule under Adam). muP is formally about
    HIDDEN width; here it is applied to this model's h (the row-reader's
    identity/attention width, which IS a hidden width) as the fan-in of the
    row_out layer specifically — flagged, per the design doc's own caveat, as
    an adapted (not literal textbook) application.

    Deliberately LR-ONLY: row_out's INIT distribution is left at nn.Linear's
    PyTorch default (no muP init-time change) rather than also changing init
    variance, to avoid stacking a second, less-precisely-cited modification
    on top of the LR change. This is the minimal-risk reading of "muP-style
    init/LR scaling" for this build, recorded here per the task's explicit
    instruction to record ambiguity resolutions in code comments.
    """
    return mup_h_base / h


# ---------------------------------------------------------------------------
# Top-level model, mirroring model_v4.MatrixMemoryModel's interface exactly
# (encode / unbind / forward) so run_stage0.py's train/eval loop can reuse
# Task D's audited call pattern almost verbatim.
# ---------------------------------------------------------------------------

class MatrixMemoryModelS0(nn.Module):
    """Encoder (variant-selectable) -> Z (optional rank forcing) -> the SAME
    pinned linear unbind readout as model_v4.MatrixMemoryModel (reused via
    static-method reference, not reimplemented — zero risk of the two
    decoders silently drifting apart)."""

    def __init__(self, d: int, h: int = 64, n_layers: int = 3, n_heads: int = 4,
                 n_refine: int = 1, **variant_kwargs):
        super().__init__()
        self.d = d
        self.encoder = BindingEncoderS0(d, h, n_layers, n_heads, n_refine, **variant_kwargs)

    def encode(self, keys, values, force_rank_k: int | None = None,
               return_intermediate: bool = False):
        out = self.encoder(keys, values, return_intermediate=return_intermediate)
        Z, q = out if return_intermediate else (out, None)
        if force_rank_k is not None and force_rank_k > 0:
            Z = truncate_to_rank(Z, force_rank_k)         # C1: train-time rank constraint
        return (Z, q) if return_intermediate else Z

    unbind = staticmethod(MatrixMemoryModel.unbind)         # pure fn of (Z, query_keys); reused verbatim

    def forward(self, batch: dict, force_rank_k: int | None = None):
        Z = self.encode(batch["keys"], batch["values"], force_rank_k=force_rank_k)
        pred = self.unbind(Z, batch["query_keys"])
        return pred, Z


# ---------------------------------------------------------------------------
# Variant presets — the ONLY interface run_stage0_sweep.py's orchestrator
# uses (it passes `--variant NAME`, never the underlying flags directly,
# matching run_task_e_sweep.py's convention of not overriding audited
# per-run recipes from the orchestrator).
#
# Candidates 1-4 are the Wave-A screen (STAGE0_DESIGN.md section 4); the task
# brief's own enumeration of "the intervention variants from the design's
# candidate list" names 1 (warmup), 3 (muP h-scaling), 2 (orthogonal/spectral
# write-path init), and 7 (per-row row_out, explicitly "available but not
# default") but does not name candidate 4 (self-attention). Candidate 4 is
# implemented anyway: STAGE0_DESIGN.md section 6's Wave-A budget table
# requires "candidates 1-4 x {d=32 K=8, d=32 K=16}" (8 of Wave A's 11 runs)
# — omitting candidate 4 would make the orchestrator's dry-run manifest
# counts (which the task brief separately requires to match the design's
# 3/12/11/12 budget table) impossible to satisfy. Minimal-risk reading of
# the ambiguity, recorded per the task's instruction.
# ---------------------------------------------------------------------------

VARIANTS = {
    "baseline": dict(
        desc="Arm 0 / Wave 0 diagnostic: unmodified, model_v4-identical architecture.",
        model=dict(row_query_init="normal", row_out_init="default",
                   row_out_mode="shared", use_row_self_attn=False),
        warmup_steps=0, mup_h_base=None,
    ),
    "c1_warmup": dict(
        desc="Candidate 1: LR warmup + cosine decay (Goyal et al. 2017; Vaswani et al. 2017).",
        model=dict(row_query_init="normal", row_out_init="default",
                   row_out_mode="shared", use_row_self_attn=False),
        warmup_steps="10pct", mup_h_base=None,
    ),
    "c2_orthogonal": dict(
        desc="Candidate 2: QR-orthogonal row_queries + orthogonal (spectral) row_out init.",
        model=dict(row_query_init="orthogonal", row_out_init="orthogonal",
                   row_out_mode="shared", use_row_self_attn=False),
        warmup_steps=0, mup_h_base=None,
    ),
    "c3_mup": dict(
        desc="Candidate 3: width scaling (h, passed via --h) + muP-style row_out LR compensation.",
        model=dict(row_query_init="normal", row_out_init="default",
                   row_out_mode="shared", use_row_self_attn=False),
        warmup_steps=0, mup_h_base=64,
    ),
    "c4_selfattn": dict(
        desc="Candidate 4: self-attention among row-query outputs (competitive differentiation).",
        model=dict(row_query_init="normal", row_out_init="default",
                   row_out_mode="shared", use_row_self_attn=True),
        warmup_steps=0, mup_h_base=None,
    ),
    "c7_rowout": dict(
        desc="Candidate 7 (fallback, non-default): per-row unshared row_out, the H_cap isolator.",
        model=dict(row_query_init="normal", row_out_init="default",
                   row_out_mode="per_row", use_row_self_attn=False),
        warmup_steps=0, mup_h_base=None,
    ),
}
