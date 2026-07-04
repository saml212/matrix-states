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

from model_rd import (_MIN_KERNEL_T, _SAFE_D_STATE, _polar_via_eigh,
                       newton_schulz_orthogonalize)
from rank_utils import effective_rank, stable_rank
from hard_selectivity_rd import (
    hard_topk_beta_mask, apply_hard_select_ste, chunk_sparsemax_beta,
    soft_topk_comparator_beta, tau_schedule, random_topk_mask,
    renormalize_to_b_pinned, classify_budget_partial,
    churn_rate, tv_distance_from_uniform, support_size,
    trackb_override_stamp_payload, trackb_assemble_gate_override_fields,
)

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
CORPUS_DIRS = {
    "openr1": "reasoning_eot", "wikitext": "wikitext103_eot",
    # SCALE_TRANSFER_DESIGN.md sec 5.4/5.6 (Track C, MAJOR-5's required
    # same-mix control cell + the rung-1 augmented-mix training corpora):
    # domain-blended corpora built by build_mix_corpora_rd.py, NOT built by
    # rebuild_lm_corpora_rd.py (that script owns the two ORIGINAL
    # single-source corpora only). Dirs written under the same data-dir
    # convention (meta.json + {split}.pt + {split}_doc_offsets.pt), so
    # load_corpus's existing field/eot_separated/bit-layout checks apply
    # unmodified to these too.
    "openr1-mix": "reasoning_mix_eot", "wikitext-mix": "wikitext103_mix_eot",
    # SCALE_TRANSFER_DESIGN.md sec 5.6 AMENDMENT (Rev 2.1, 2026-07-04) item 2:
    # waves >=2 train on these EXTENDED mixes (the epoch-cap remedy -- rung
    # 2's 1.5B-token/run target exceeds the original mixes' <=5-epoch
    # ceilings, sec 5.4). Built side-by-side with the originals by
    # build_mix_corpora_rd.py --out-suffix _extended; val/test are
    # BYTE-IDENTICAL to the originals (md5-verified), only train grew (more
    # augmentation pulled in). Same meta.json/eot_separated contract --
    # load_corpus applies unmodified, no format change.
    "openr1-mix-ext": "reasoning_mix_eot_extended", "wikitext-mix-ext": "wikitext103_mix_eot_extended",
    # TRACKB_REDESIGN.md sec 5.1 (Rev 3 NEW-7): the duplicate-key stress slice for the geo3-LM
    # NaN-stability smoke -- built by wave_neg1_trackb.py --build-stress-slice (window-per-document,
    # one EOT appended per window, same meta.json/eot_separated contract so load_corpus applies
    # unmodified). Only the stability-smoke cell ever trains on it.
    "openr1-stress": "reasoning_stress_eot",
}
OTHER_CORPUS = {
    "openr1": "wikitext", "wikitext": "openr1",
    # mix corpora pair with EACH OTHER (mirrors the original pairing exactly
    # -- the cross-corpus eval axis stays reasoning-vs-narrative, now on the
    # augmented-mix side of that same contrast).
    "openr1-mix": "wikitext-mix", "wikitext-mix": "openr1-mix",
    # extended mixes pair with EACH OTHER too (sec 5.6 amendment item 2) --
    # same reasoning-vs-narrative contrast, now on the extended-mix side.
    "openr1-mix-ext": "wikitext-mix-ext", "wikitext-mix-ext": "openr1-mix-ext",
    # stress slice pairs with the plain wikitext val (one-directional: nothing trains on
    # "wikitext" expecting the stress slice back -- the smoke only needs SOME cross-corpus val).
    "openr1-stress": "wikitext",
}
DEFAULT_DATA_DIR = "/data/deltanet_rd_data"
EOT_TOKEN_ID = 50256   # GPT-2 <|endoftext|>
# SCALE_TRANSFER_DESIGN.md sec 5.3: the ladder's deepest registered rung
# (rung 3) is n_layers=22; this ceiling carries a small deliberate margin
# above that, not a config-specific bound (sec 5.2's harness-change scope).
_MAX_N_LAYERS = 24

# Rank-stat sampling: fractional positions within a sampled document
# (section 6, this build's brief: "whole-state effective/stable rank per
# head over sampled positions, split by document"). See
# sample_state_rank_stats's docstring for the document-alignment behavior
# (AUDIT FIX-3: windows start at document starts, truncation logged).
RANK_SAMPLE_FRACS = (0.25, 0.5, 0.75, 1.0)

# ---------------------------------------------------------------------------
# Track B / geo3-in-LM (SCALE_TRANSFER_DESIGN.md sec 4): per-chunk, beta-gated
# (or naive-window) key orthogonalization. Off by default everywhere
# (geo3_active=False) -- the default path is byte-identical to the
# pre-Track-B code (regression-checked in smoke() item [6]).
# ---------------------------------------------------------------------------

GEO3_LM_SELECTION_MODES = ("beta_topk", "naive_window")
GEO3_LM_CHUNK_SIZE_DEFAULT = 64   # sec 4.2 item 1: chunk_delta_rule's own sequence-tiling constant
                                    # (d_state-INDEPENDENT -- MAJOR-3's correction; the binding rank
                                    # constraint is k_sel <= head_dim, asserted separately below)

# ---------------------------------------------------------------------------
# Track B hard-selectivity wave (TRACKB_REDESIGN.md Rev 3): candidates 1
# (hard_ste), 2 (entmax/sparsemax), the M7 comparator (soft_topk_comparator),
# and the Cell 2R/4R control (random_topk). OFF by default everywhere
# (hard_select_active=False) -- the default path is byte-identical to the
# pre-Track-B code (regression-checked alongside geo3's own smoke item [6]).
# ---------------------------------------------------------------------------
HARD_SELECT_MECHANISMS = ("hard_ste", "entmax", "soft_topk_comparator", "random_topk")


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


def gini_coefficient(x: torch.Tensor) -> torch.Tensor:
    """Gini coefficient of a non-negative distribution along the LAST dim
    (batched). Standard rank-based closed form on x sorted ASCENDING:
    G = 2*sum(i*x_sorted_i) / (n*sum(x)) - (n+1)/n, i in [1,n]. G=0 for
    perfect equality, G->(n-1)/n for one point holding all the mass (both
    verified by hand in this function's own smoke case). x: (...,n), values
    assumed >= 0 (beta -- a sigmoid output -- always satisfies this).
    Degenerate (n<=1, or all-zero) inputs return 0 by convention (matches
    the standard "no inequality to measure" edge-case handling), not NaN."""
    n = x.shape[-1]
    if n <= 1:
        return torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
    x_sorted, _ = torch.sort(x, dim=-1)
    total = x_sorted.sum(dim=-1)
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    weighted = (idx * x_sorted).sum(dim=-1)
    safe_total = total.clamp(min=1e-12)
    gini = (2.0 * weighted) / (n * safe_total) - (n + 1.0) / n
    return torch.where(total > 1e-12, gini, torch.zeros_like(gini))


def _geo3_lm_select_and_orthogonalize(k_raw: torch.Tensor, beta: torch.Tensor,
                                       content_mask: torch.Tensor, chunk_size: int, k_sel: int,
                                       n_iter: int, resid_tol: float, selection_mode: str,
                                       forced_topk_idx: torch.Tensor | None = None):
    """The Track B / geo3-in-LM chunk-local, (beta-gated OR naive-window)
    top-K_sel construction (SCALE_TRANSFER_DESIGN.md sec 4.2/4.3). Reuses
    model_rd.py's audited gather -> orthogonalize -> scatter pattern
    (DeltaNetRDBlock.bind()'s geo3_active branch, model_rd.py:871-888) and
    its orthogonalization machinery (newton_schulz_orthogonalize +
    _polar_via_eigh, imported unmodified) -- what is NEW here is (a) the
    selection-SET construction (per-chunk top-K_sel by beta magnitude, or a
    positionally-fixed naive-window selection, in place of model_rd.py's
    hand-specified item_pos, which has no free-text analog, sec 4.2), and
    (b) the degenerate-episode handling around the fallback (masked-identity
    residual + eigh-fallback denial for invalid-slot-bearing episodes --
    AUDIT ROUND-2 MAJOR-1, see the inline comment at the call site below;
    the earlier draft's direct geo3_orthogonalize_logged reuse was wrong for
    exactly the degenerate case the synthetic harness never produces).

    k_raw: (B,T,H,head_dim) -- PRE-kernel-normalization conv output (the
      kernel applies its OWN internal L2-norm via use_qk_l2norm_in_kernel=
      True; the F.normalize call inside this function is ONLY the
      pre-scaling Newton-Schulz's own convergence proof requires --
      model_rd.py's newton_schulz_orthogonalize docstring: "rows ALREADY
      L2-normalized" -- NOT a substitute for the kernel's final internal
      norm. This is this build's resolution of sec 4.3's flagged
      plumbing-mismatch risk: selected AND non-selected positions both
      still pass through the SAME final kernel-internal normalization
      afterward -- there is no asymmetric double- or skipped-normalization
      between them. Flagged for audit scrutiny: the kernel's own final
      renorm CAN slightly perturb the joint Q@Q^T~=I_K guarantee Newton-
      Schulz just built (a real, measured, sub-1e-2-scale effect per this
      build's own smoke item [7], not assumed away).
    beta: (B,T,H) the plain learned gate (post-sigmoid; monotonic, so
      ranking is unaffected by using post- vs pre-sigmoid, but the
      post-sigmoid value is also the interpretable "write mass" this
      function's own diagnostics report).
    content_mask: (B,T) bool, True = eligible for selection (sec 4.2 item 3:
      "EOT and any padding positions are hard-excluded from selection").
    chunk_size: window (sec 4.2 item 1: kernel-aligned tiling constant,
      d_state-INDEPENDENT -- MAJOR-3's correction).
    k_sel: positions orthogonalized per (chunk, head). MUST be
      <= min(chunk_size, head_dim) -- head_dim (not d_state) is this
      build's generalization of sec 4.2 item 1's "K_sel <= min(window,
      d_state)" rule to the per-HEAD orthogonalization this function
      performs (the two bounds coincide exactly whenever num_heads==1,
      i.e. head_dim==d_state -- the ONLY configuration any cell in this
      design's own manifest, sec 4.5, registers; num_heads>1 is supported
      here for generality but carries NO Wave 1 cell coverage -- flagged
      for audit scrutiny).
    selection_mode: "beta_topk" (sec 4.2's primary construction: rank by
      beta magnitude) or "naive_window" (sec 4.2/4.5's rejected-but-kept
      comparison arm -- a beta-BLIND, purely positional selection. This
      build's literal reading of "unconditional fixed-window
      orthogonalization" at the SAME K_sel granularity sec 4.5's manifest
      table runs it at: "the first K_sel content positions of each chunk,
      by sequence index." OPEN INTERPRETIVE CALL, flagged for audit
      scrutiny -- sec 4.2's own prose ("orthogonalize every token in a
      window") is not perfectly literal at k_sel<chunk_size; sec 4.5's own
      table is the more load-bearing source since it pins the K_sel grid).

    Returns (k_out (B,T,H,head_dim), diag: dict). diag carries per-call
    scalars (resid mean/max/min, fallback rate, valid-selection fraction)
    PLUS the raw per-selected-slot tensors (_topk_idx, _valid_sel, _Q) a
    caller needs to build the sec 4.4 drift diagnostic (token identity is
    looked up by the CALLER, which owns token_ids; this function only ever
    sees k_raw/beta/content_mask, never token_ids directly, keeping its
    contract layout-agnostic).

    forced_topk_idx (TRACKB_REDESIGN.md Rev 2 -- M6, Cell 4's composition
    rule): (B,n_chunks,k_sel,H) int64 in-chunk offsets, OPTIONAL. When
    given, this function's OWN internal selection (both the beta_topk and
    naive_window branches, and `selection_mode` itself) is SKIPPED entirely
    -- the caller's own hard-selectivity mask supplies the selected set
    (hard_selectivity_rd.py's hard_topk_beta_mask/random_topk_mask), and
    this function performs ONLY its gather -> orthogonalize -> scatter role
    on that forced set. Callers MUST ensure hard_select_k_sel == k_sel
    themselves (M6's single-selection-source rule; this function only
    shape-asserts the consequence, it does not know the caller's own
    hard_select_k_sel). Validity (which forced slots are genuine content,
    not EOT/padding) is NOT derived from an internal topk/priority
    comparison -- there is none in this path -- it is derived by GATHERING
    content_mask AT the forced indices (Rev 3 -- MINOR-2's registered fix),
    feeding the SAME downstream degenerate-episode machinery (zero-row
    handling / masked-identity residual / fallback denial) unchanged."""
    B, T, H, d = k_raw.shape
    assert T % chunk_size == 0, (
        f"T={T} is not a multiple of chunk_size={chunk_size} -- geo3-in-LM's chunking requires an "
        f"exact tiling (sec 4.2 item 1's kernel-aligned window); pad or choose a compatible "
        f"seq_len/ctx_len/cont_len upstream.")
    assert k_sel <= min(chunk_size, d), (
        f"k_sel={k_sel} must be <= min(chunk_size={chunk_size}, head_dim={d}) -- head_dim is the "
        f"actual Newton-Schulz orthogonalization dimension per head (sec 4.2 item 1's d_state bound "
        f"generalizes to head_dim under per-head orthogonalization; the two coincide when "
        f"num_heads==1).")
    assert selection_mode in GEO3_LM_SELECTION_MODES, \
        f"selection_mode={selection_mode!r} not in {GEO3_LM_SELECTION_MODES}"
    n_chunks = T // chunk_size

    k_c = k_raw.view(B, n_chunks, chunk_size, H, d)
    beta_c = beta.view(B, n_chunks, chunk_size, H)
    content_c = content_mask.view(B, n_chunks, chunk_size, 1).expand(B, n_chunks, chunk_size, H)

    if forced_topk_idx is not None:
        # TRACKB_REDESIGN.md Rev 2 -- M6: geo3's OWN selection is REPLACED by the caller's
        # hard-selectivity mask (single selection source). Validity is derived by GATHERING
        # content_mask at the forced indices (Rev 3 -- MINOR-2), NOT from a priority/topk
        # comparison (there is none here).
        assert forced_topk_idx.shape == (B, n_chunks, k_sel, H), (
            f"forced_topk_idx shape {tuple(forced_topk_idx.shape)} != expected "
            f"(B,n_chunks,k_sel,H)=({B},{n_chunks},{k_sel},{H}) -- M6 requires "
            f"hard_select_k_sel == geo3_k_sel; the caller must guarantee this BEFORE calling "
            f"(this assert only catches the shape consequence, not the caller's own config).")
        topk_idx = forced_topk_idx.to(torch.int64)
        valid_sel = torch.gather(content_c, 2, topk_idx)                   # exact, EOT/padding excluded
    else:
        neg_inf = torch.finfo(beta_c.dtype).min
        if selection_mode == "beta_topk":
            priority = torch.where(content_c, beta_c, torch.full_like(beta_c, neg_inf))
        else:  # naive_window: purely positional, beta-BLIND (this build's sensitivity-control target,
               # sec 4.8 item 1 -- "tests whether the selection refinement matters at all")
            pos_priority = torch.arange(chunk_size, 0, -1, device=k_raw.device, dtype=beta_c.dtype)
            pos_priority = pos_priority.view(1, 1, chunk_size, 1).expand(B, n_chunks, chunk_size, H)
            priority = torch.where(content_c, pos_priority, torch.full_like(pos_priority, neg_inf))

        topk_val, topk_idx = torch.topk(priority, k_sel, dim=2)                # (B,n_chunks,k_sel,H)
        valid_sel = topk_val > (neg_inf / 2)                                    # strictly-content slots only
                                                                                  # (excludes EOT/padding
                                                                                  # positions selected only
                                                                                  # because a chunk ran out
                                                                                  # of real content, sec 4.2
                                                                                  # item 3 -- degenerate but
                                                                                  # handled EXACTLY, not by
                                                                                  # a tolerance-slack shortcut)

    idx_expand = topk_idx.unsqueeze(-1).expand(B, n_chunks, k_sel, H, d)
    k_gathered_raw = torch.gather(k_c, 2, idx_expand)                      # (B,n_chunks,k_sel,H,d)

    # -> (B,n_chunks,H,k_sel,d): fold H into the orthogonalization batch axis (per-head Newton-Schulz)
    k_gathered_raw_p = k_gathered_raw.permute(0, 1, 3, 2, 4).contiguous()
    valid_p = valid_sel.permute(0, 1, 3, 2).contiguous()                   # (B,n_chunks,H,k_sel)

    flat_n = B * n_chunks * H
    k_flat_raw = k_gathered_raw_p.reshape(flat_n, k_sel, d)
    valid_flat = valid_p.reshape(flat_n, k_sel)

    # Zero-safe: invalid (degenerate, insufficient-content-chunk) slots become an all-zero row
    # BEFORE normalizing -- Newton-Schulz's pre-scaling proof (sigma_max(X_0)<=1, model_rd.py's own
    # docstring) holds UNCHANGED with zero rows present (the Frobenius norm can only shrink), and
    # F.normalize is zero-safe (0-row -> 0-row, model_rd.py's own convention).
    k_flat_for_ns = torch.where(valid_flat.unsqueeze(-1), k_flat_raw, torch.zeros_like(k_flat_raw))
    k_flat_norm = F.normalize(k_flat_for_ns, dim=-1)

    # AUDIT ROUND-2 MAJOR-1 restructure (2026-07-04, independent audit): the earlier draft called
    # model_rd.geo3_orthogonalize_logged here directly. Two problems, both specific to degenerate
    # (invalid-slot-bearing) episodes that have NO analog in the synthetic harness:
    #   (1) SPURIOUS FALLBACK: newton_schulz_orthogonalize's resid measures ||QQ^T - I_K||_F, and
    #       every invalid ZERO row contributes exactly +1.0 to its square (its (i,i) diagonal entry:
    #       0 vs 1; all its cross terms are 0 vs 0) -- so ANY degenerate episode in the batch had
    #       resid >= 1 >> resid_tol and dragged the WHOLE batch into the eigh fallback every call.
    #       Fixed by correcting the residual to the masked-identity target: resid_corrected^2 =
    #       resid^2 - n_invalid (exact algebra, clamped at 0 for fp noise). fp32 PRECISION FLOOR,
    #       measured via smoke [8b]'s own first (failed) draft: for a degenerate episode the
    #       subtraction cannot resolve a valid-block residual below ~sqrt(n_invalid * fp32_eps)
    #       (~1e-3 at n_invalid=8) -- it clamps to exactly 0. Conservative in the safe direction
    #       (an under-measured residual can only SKIP a fallback the default resid_tol=1e-2 would
    #       not have demanded anyway, since the floor sits below the tol).
    #   (2) NaN GRADIENTS in the fallback: _polar_via_eigh on a Gram matrix with >=~6 coincident
    #       (jittered-to-eps) zero eigenvalues from invalid rows produces NaN in eigh's backward
    #       (1/(lambda_i - lambda_j) at repeated eigenvalues -- EMPIRICALLY reproduced by the audit,
    #       onset at n_invalid=6, persists in fp64). Fixed by DENYING the eigh fallback to any
    #       episode containing invalid rows: those episodes keep their Newton-Schulz output (whose
    #       gradient is polynomial and always finite; its zero rows are exactly preserved). Denials
    #       are counted in diag["n_fallback_denied_degenerate"] -- never silent.
    # KNOWN RESIDUAL RISK (documented, not closed here): a FULLY-VALID episode whose selected keys
    # contain >=~6 exactly-duplicated rows (identical conv-context 4-grams, e.g. heavily tabular/
    # repetitive text) also yields coincident Gram eigenvalues in the fallback and can NaN the same
    # way; train()'s isfinite-grad skip-step guard catches it (step skipped, counted in skip_rate),
    # so it is observable in every result JSON rather than corrupting -- flagged for Wave 1
    # monitoring, not assumed away.
    Q, resid_raw = newton_schulz_orthogonalize(k_flat_norm, n_iter=n_iter)
    with torch.no_grad():
        n_invalid_per_ep = (~valid_flat).sum(dim=-1).to(resid_raw.dtype)          # (flat_n,)
        resid = (resid_raw.pow(2) - n_invalid_per_ep).clamp(min=0.0).sqrt()        # masked-identity target
    fully_valid = valid_flat.all(dim=-1)                                           # (flat_n,)
    exceeds = resid > resid_tol
    fallback_triggered = bool(exceeds.any())
    n_fallback_denied = 0
    if fallback_triggered:
        n_fallback_denied = int((exceeds & ~fully_valid).sum().item())
        if bool(fully_valid.all()):
            # no degenerate episodes anywhere: EXACT model_rd whole-batch-retry semantics
            Q = _polar_via_eigh(k_flat_norm)
        else:
            idx_fv = fully_valid.nonzero(as_tuple=True)[0]
            if idx_fv.numel() > 0:
                # whole-batch-among-eligible: every FULLY-VALID episode goes through eigh
                # (matching model_rd's batch-granularity convention as closely as the
                # degenerate-exclusion rule allows); index_copy is out-of-place + autograd-clean
                Q = Q.index_copy(0, idx_fv, _polar_via_eigh(k_flat_norm.index_select(0, idx_fv)))

    # Invalid slots: EXACT no-op -- scatter back the ORIGINAL raw (pre-normalize, pre-zero) value,
    # NEVER Q (whose invalid-slot rows are provably exactly zero by Newton-Schulz's own recursion --
    # scattering them would ZERO OUT a real EOT-token key that must stay untouched, sec 4.2 item 3).
    scatter_src_flat = torch.where(valid_flat.unsqueeze(-1), Q, k_flat_raw)
    scatter_src = scatter_src_flat.reshape(B, n_chunks, H, k_sel, d).permute(0, 1, 3, 2, 4).contiguous()

    k_c_out = k_c.clone()
    k_c_out.scatter_(2, idx_expand, scatter_src)
    k_out = k_c_out.reshape(B, T, H, d)

    diag = {
        # resid_* report the CORRECTED (masked-identity-target) residual -- the decision quantity
        "resid_mean": resid.mean().item(), "resid_max": resid.max().item(),
        "resid_min": resid.min().item(),
        "fallback_triggered": bool(fallback_triggered),
        "n_fallback_denied_degenerate": n_fallback_denied,     # AUDIT ROUND-2 MAJOR-1: never silent
        "frac_valid_selections": valid_flat.float().mean().item(),
        "n_chunks": n_chunks, "k_sel": k_sel, "chunk_size": chunk_size,
        "selection_mode": "forced" if forced_topk_idx is not None else selection_mode,
        # raw tensors for the OPTIONAL caller-side drift diagnostic (sec 4.4) -- NOT reduced here,
        # the caller (which owns token_ids) maps these back to token identity.
        "_topk_idx": topk_idx, "_valid_sel": valid_sel, "_Q": Q.reshape(B, n_chunks, H, k_sel, d),
        # TRACKB_REDESIGN.md sec 5.1 (Rev 3 NEW-7)'s positive-control input: the RAW gathered
        # selected rows, exposed so train()'s --nan-probe-counter path can count exact duplicates
        # among the SELECTED set per forward call (wave_neg1_trackb.NanStabilityProbeCounter) --
        # the caller-side check the smoke's >=25-calls/>=6-dup-rows floor is measured against.
        "_k_selected_raw": k_flat_raw.detach(),
    }
    return k_out, diag


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

    def __init__(self, d_model: int, d_state: int, conv_size: int = 4, num_heads: int = 1,
                 geo3_active: bool = False, geo3_k_sel: int = 16,
                 geo3_chunk_size: int = GEO3_LM_CHUNK_SIZE_DEFAULT, geo3_n_iter: int = 12,
                 geo3_resid_tol: float = 1e-2, geo3_selection_mode: str = "beta_topk",
                 geo3_excluded_token_ids: tuple = (EOT_TOKEN_ID,),
                 hard_select_active: bool = False, hard_select_mechanism: str = "hard_ste",
                 hard_select_k_sel: int = 32, hard_select_chunk_size: int = GEO3_LM_CHUNK_SIZE_DEFAULT,
                 hard_select_b_pinned: float | None = None, hard_select_seed: int = 0,
                 hard_select_tau_total_steps: int | None = None,
                 hard_select_tau_anneal_frac: float = 0.10,
                 hard_select_excluded_token_ids: tuple = (EOT_TOKEN_ID,)):
        """geo3_* (SCALE_TRANSFER_DESIGN.md sec 4, Track B): OFF by default
        (geo3_active=False) -- when off, none of this __init__'s new
        parameters are read anywhere in forward(), and forward()'s default
        code path is UNCHANGED line-for-line (smoke item [6] regression-
        checks this). See _geo3_lm_select_and_orthogonalize's docstring for
        the full parameter contract.

        hard_select_* (TRACKB_REDESIGN.md Rev 3 -- Track B's hard-
        selectivity wave). OFF by default (hard_select_active=False) --
        mirrors geo3_active's own additive-off-by-default pattern exactly:
        when off, none of these parameters are read in forward() and the
        default path (both flags False) is byte-identical to the
        pre-Track-B code. When active, REPLACES the plain independent
        sigmoid at the mixer's beta site with one of HARD_SELECT_MECHANISMS
        (hard_selectivity_rd.py):
          "hard_ste"              Candidate 1 (PRIMARY, sec 3.1): hard top-K
                                   beta mask + straight-through gradient.
          "entmax"                Candidate 2 (sec 3.2): chunk-normalized
                                   sparsemax (only sparsemax is built; the
                                   entmax bisection variant is a named,
                                   unbuilt stretch per sec 3.2's own ranking).
          "soft_topk_comparator"  M7 comparator (sec 3.1, Rev 3 NEW-6):
                                   temperature-annealed soft top-K.
          "random_topk"           Cell 2R/4R control (sec 5.1): beta-blind
                                   random top-K, resampled per
                                   (chunk instance, training step).
        hard_select_active and geo3_active are INDEPENDENTLY toggleable
        (sec 5's 2x2 factorial axis). Cell 4's composition rule (Rev 2 --
        M6) fires automatically when BOTH are True and the mechanism
        produces a fixed top-K SET ("hard_ste"/"random_topk"/
        "soft_topk_comparator" all do, via hard_topk_beta_mask's own
        topk_idx; "entmax" does not -- its support size is content/score-
        dependent, so there is no single fixed-shape forced_topk_idx to
        thread, and the combination is HARD-REFUSED below, mirroring
        geo3_active's own num_heads==1 untested-at-scope discipline):
        hard_select_k_sel MUST equal geo3_k_sel, and geo3's OWN
        beta_topk/naive_window selection is REPLACED by the mask's own
        selected set (a single selection source, never a double-selection)."""
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
        if geo3_active:
            assert geo3_selection_mode in GEO3_LM_SELECTION_MODES, \
                f"geo3_selection_mode={geo3_selection_mode!r} not in {GEO3_LM_SELECTION_MODES}"
            assert geo3_k_sel <= min(geo3_chunk_size, head_dim), (
                f"geo3_k_sel={geo3_k_sel} must be <= min(geo3_chunk_size={geo3_chunk_size}, "
                f"head_dim={head_dim}) -- sec 4.2 item 1's corrected binding constraint "
                f"(head_dim generalizes the design's d_state bound to per-head orthogonalization)."
            )
            # AUDIT ROUND-2 recommendation: _geo3_lm_select_and_orthogonalize's H-folding math is
            # H-general (audit-verified), but NO registered Track B cell (sec 4.5) covers
            # num_heads>1 with geo3 active, and no automated test combines them -- hard-refuse the
            # untested-at-scope combination rather than let it launch by accident at d_state=128
            # (where head_dim=64 would pass _SAFE_D_STATE). Widen deliberately, with a real test,
            # not by deleting this line.
            assert num_heads == 1, (
                f"geo3_active with num_heads={num_heads} is UNTESTED AT SCOPE (every registered "
                f"Track B cell is num_heads=1) -- see this assert's comment before widening."
            )
        if hard_select_active:
            assert hard_select_mechanism in HARD_SELECT_MECHANISMS, (
                f"hard_select_mechanism={hard_select_mechanism!r} not in {HARD_SELECT_MECHANISMS}")
            assert 1 <= hard_select_k_sel <= hard_select_chunk_size, (
                f"hard_select_k_sel={hard_select_k_sel} must be in [1,{hard_select_chunk_size}]")
            if geo3_active:
                # TRACKB_REDESIGN.md Rev 2 -- M6: Cell 4's composition rule. "entmax" has no fixed
                # top-K set to force (variable support size) -- hard-refused, not silently mishandled.
                assert hard_select_mechanism != "entmax", (
                    "hard_select_mechanism='entmax' + geo3_active=True is NOT SUPPORTED: "
                    "entmax/sparsemax's support size is content/score-dependent, so there is no "
                    "single forced_topk_idx of fixed shape (k_sel,H) to thread into M6's composition "
                    "rule. TRACKB_REDESIGN.md sec 5.1's Cell 4 is defined over 'the surviving "
                    "candidate' -- if candidate 2 is ever the surviving mechanism, Cell 4's "
                    "composition needs its own (unbuilt) variable-support forcing scheme. "
                    "Untested-at-scope, hard-refused (mirrors geo3_active's own num_heads==1 "
                    "discipline above)."
                )
                assert hard_select_k_sel == geo3_k_sel, (
                    f"Cell 4 composition rule (M6): hard_select_k_sel={hard_select_k_sel} must "
                    f"equal geo3_k_sel={geo3_k_sel} when both hard_select_active and geo3_active "
                    f"are True (single selection source, TRACKB_REDESIGN.md sec 5.1)."
                )
                assert hard_select_chunk_size == geo3_chunk_size, (
                    f"hard_select_chunk_size={hard_select_chunk_size} must equal "
                    f"geo3_chunk_size={geo3_chunk_size} when composing with geo3 (M6) -- both "
                    f"selections must tile the sequence identically."
                )
        self.hard_select_active = hard_select_active
        self.hard_select_mechanism = hard_select_mechanism
        self.hard_select_k_sel = hard_select_k_sel
        self.hard_select_chunk_size = hard_select_chunk_size
        self.hard_select_b_pinned = hard_select_b_pinned
        self.hard_select_seed = hard_select_seed
        self.hard_select_tau_total_steps = hard_select_tau_total_steps
        self.hard_select_tau_anneal_frac = hard_select_tau_anneal_frac
        self.hard_select_excluded_token_ids = tuple(hard_select_excluded_token_ids)
        self.hard_select_last_diag: dict | None = None   # mirrors geo3_last_diag's side-channel convention
        # AUDIT FIX (independent audit 2026-07-04, F2): read-only implicit-selection probe for the
        # UNMASKED reference pilot (sec 4.3's churn Null A / TV-ceiling data source). When set (an
        # int K_sel, via main()'s --trackb-selection-probe -- post-construction, never a
        # constructor arg), forward()'s plain-sigmoid branch ADDITIONALLY computes the implicit
        # top-K-by-beta selection on the DETACHED beta and stores it in hard_select_last_diag;
        # beta itself and every downstream tensor are untouched (read-only by construction, the
        # same discipline as sec 5.2's Cell-1 instrument).
        self.selection_probe_k_sel: int | None = None
        self.d_model = d_model
        self.d_state = d_state
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.geo3_active = geo3_active
        self.geo3_k_sel = geo3_k_sel
        self.geo3_chunk_size = geo3_chunk_size
        self.geo3_n_iter = geo3_n_iter
        self.geo3_resid_tol = geo3_resid_tol
        self.geo3_selection_mode = geo3_selection_mode
        self.geo3_excluded_token_ids = tuple(geo3_excluded_token_ids)
        self.geo3_last_diag: dict | None = None   # sec 14.4-style per-forward-call side channel

        self.q_proj = nn.Linear(d_model, d_state, bias=False)
        self.k_proj = nn.Linear(d_model, d_state, bias=False)
        self.v_proj = nn.Linear(d_model, d_state, bias=False)
        self.b_proj = nn.Linear(d_model, num_heads, bias=False)   # plain learned beta, NO mask (LM mode)
        self.q_conv1d = ShortConvolution(hidden_size=d_state, kernel_size=conv_size, bias=False, activation="silu")
        self.k_conv1d = ShortConvolution(hidden_size=d_state, kernel_size=conv_size, bias=False, activation="silu")
        self.v_conv1d = ShortConvolution(hidden_size=d_state, kernel_size=conv_size, bias=False, activation="silu")
        self.o_norm = RMSNorm(head_dim, eps=1e-5)
        self.o_proj = nn.Linear(d_state, d_model, bias=False)

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor | None = None,
                token_ids: torch.Tensor | None = None, step: int | None = None):
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
        a defensive backstop, not the primary enforcement point.

        step (TRACKB_REDESIGN.md Rev 3, hard-selectivity wave): the current
        TRAINING step (1-indexed, matches train()'s own loop counter,
        lm_pretrain_rd.py's `for step in range(1, args.steps+1)`). REQUIRED
        when hard_select_active with a step-dependent mechanism
        ("random_topk"'s per-(chunk,step) RNG stream, "soft_topk_comparator"'s
        tau(t) anneal). Build-phase note (1), load-bearing: an eval-time
        forward-hook probe MUST pass the checkpoint's OWN recorded step here
        -- never a separate probe-specific value, and never omitted in favor
        of a default -- so the RNG stream CONTINUES exactly what a
        training-time forward call at that step would have drawn, rather
        than freezing a fresh draw per checkpoint (hard_selectivity_rd.py's
        derive_step_rng docstring states the same contract from the callee
        side; test_trackb_smokes.py's continuation-equivalence smoke
        verifies it)."""
        B, T, _ = x.shape
        assert T >= _MIN_KERNEL_T, (
            f"sequence length {T} < _MIN_KERNEL_T={_MIN_KERNEL_T} -- chunk_delta_rule's backward "
            f"crashes below this floor (F15-LM, measured 2026-07-02). Choose "
            f"seq_len/ctx_len/continuation_len >= {_MIN_KERNEL_T}."
        )
        q, _ = self.q_conv1d(self.q_proj(x))
        k, _ = self.k_conv1d(self.k_proj(x))
        v, _ = self.v_conv1d(self.v_proj(x))

        # Track B / geo3-in-LM (SCALE_TRANSFER_DESIGN.md sec 4.3's insertion point) AND the
        # hard-selectivity wave (TRACKB_REDESIGN.md Rev 3) share ONE content_mask -- computed
        # whenever EITHER flag (or the F2 read-only selection probe) is active. When
        # hard_select_active=False and no probe, excluded_ids reduces to EXACTLY
        # geo3_excluded_token_ids (byte-identical to the pre-Track-B-hard-selectivity code, smoke
        # item [6]'s own regression-check convention extended here).
        content_mask = None
        if self.geo3_active or self.hard_select_active or self.selection_probe_k_sel is not None:
            assert token_ids is not None, (
                "geo3_active / hard_select_active / selection_probe require token_ids "
                "(EOT-exclusion mask, sec 4.2 item 3) -- the caller (DeltaNetLMBlock.forward / "
                "DeltaNetLM.forward) must thread token_ids down to every mixer.forward() call."
            )
            assert token_ids.shape == (B, T), f"token_ids shape {tuple(token_ids.shape)} != (B,T)=({B},{T})"
            excluded_ids = set()
            if self.geo3_active:
                excluded_ids |= set(self.geo3_excluded_token_ids)
            if self.hard_select_active or self.selection_probe_k_sel is not None:
                excluded_ids |= set(self.hard_select_excluded_token_ids)
            content_mask = torch.ones(B, T, dtype=torch.bool, device=k.device)
            for excluded_id in excluded_ids:
                content_mask = content_mask & (token_ids != excluded_id)

        # Hard-selectivity mechanism (TRACKB_REDESIGN.md Rev 3). OFF by default
        # (hard_select_active=False): this entire block does not execute, and `beta` falls through
        # to the plain sigmoid `else` branch below, UNCHANGED (smoke item [6]'s regression-check
        # convention extended to this new axis).
        forced_topk_idx = None
        if self.hard_select_active:
            mech = self.hard_select_mechanism
            if mech == "hard_ste":
                beta_soft = torch.sigmoid(self.b_proj(x))
                mask, topk_idx, valid_sel = hard_topk_beta_mask(
                    beta_soft, content_mask, self.hard_select_chunk_size, self.hard_select_k_sel)
                beta = apply_hard_select_ste(beta_soft, mask)
                sel_diag = {"mechanism": mech, "topk_idx": topk_idx, "valid_sel": valid_sel}
            elif mech == "entmax":
                scores = self.b_proj(x)
                beta = chunk_sparsemax_beta(scores, content_mask, self.hard_select_chunk_size)
                sel_diag = {"mechanism": mech}
            elif mech == "soft_topk_comparator":
                assert self.hard_select_tau_total_steps is not None and step is not None, (
                    "soft_topk_comparator requires BOTH hard_select_tau_total_steps (constructor) "
                    "AND step (this forward() call's own arg, the tau(t) anneal's input) -- never "
                    "silently defaulted (Rev 3 NEW-6's registered 10%-of-steps anneal pin)."
                )
                tau = tau_schedule(step, self.hard_select_tau_total_steps, self.hard_select_tau_anneal_frac)
                beta_soft = torch.sigmoid(self.b_proj(x))
                beta, mask, topk_idx, valid_sel = soft_topk_comparator_beta(
                    beta_soft, content_mask, self.hard_select_chunk_size, self.hard_select_k_sel, tau)
                sel_diag = {"mechanism": mech, "tau": tau, "topk_idx": topk_idx, "valid_sel": valid_sel}
            else:
                assert mech == "random_topk"
                assert step is not None, (
                    "random_topk (Cell 2R/4R) requires `step` (this forward() call's own arg) -- "
                    "the per-(chunk,step) RNG stream's input (hard_selectivity_rd.derive_step_rng, "
                    "Rev 3 NEW-2/NEW-3's registered per-instance-per-step cadence)."
                )
                beta_soft = torch.sigmoid(self.b_proj(x))
                mask, topk_idx, valid_sel = random_topk_mask(
                    (B, T, self.num_heads), content_mask, self.hard_select_chunk_size,
                    self.hard_select_k_sel, self.hard_select_seed, step, device=x.device)
                beta = beta_soft * mask
                sel_diag = {"mechanism": mech, "topk_idx": topk_idx, "valid_sel": valid_sel}

            if self.hard_select_b_pinned is not None:
                # sec 2 principle 4 (Rev 3 NEW-1, symmetric): mandatory for EVERY masking mechanism,
                # including candidate 2's structurally-partial case.
                beta, shortfall = renormalize_to_b_pinned(
                    beta, self.hard_select_chunk_size, self.hard_select_b_pinned)
                sel_diag["shortfall"] = shortfall
                sel_diag["budget_partial"] = classify_budget_partial(shortfall)
            # AUDIT FIX (2026-07-04, F2): the post-mechanism beta itself is part of the diag
            # side-channel -- entmax's support-size/TV diagnostics have no topk_idx to read and
            # must derive selection from beta's own exact zeros (sample_hard_select_diagnostics).
            sel_diag["_beta"] = beta.detach()
            self.hard_select_last_diag = sel_diag

            if self.geo3_active and mech != "entmax":
                forced_topk_idx = topk_idx        # M6: single selection source
        else:
            beta = torch.sigmoid(self.b_proj(x))                       # (B,T,H), plain learned gate
            if self.selection_probe_k_sel is not None:
                # AUDIT FIX (2026-07-04, F2): the unmasked pilot's IMPLICIT top-K-by-beta
                # selection, read-only on detached beta (sec 4.3 Null A / TV-ceiling source) --
                # beta and every downstream tensor are untouched.
                _, probe_idx, probe_valid = hard_topk_beta_mask(
                    beta.detach(), content_mask, self.hard_select_chunk_size,
                    self.selection_probe_k_sel)
                self.hard_select_last_diag = {
                    "mechanism": "implicit_probe", "topk_idx": probe_idx, "valid_sel": probe_valid,
                    "_beta": beta.detach(),
                }

        q = q.reshape(B, T, self.num_heads, self.head_dim)
        k = k.reshape(B, T, self.num_heads, self.head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim)

        # Track B / geo3-in-LM (SCALE_TRANSFER_DESIGN.md sec 4.3's insertion point: "after
        # k, _ = self.k_conv1d(...) and beta = torch.sigmoid(self.b_proj(x)), before the
        # chunk_delta_rule(...) call"). OFF by default (geo3_active=False): this entire block
        # does not execute, so the default path below is UNCHANGED (smoke item [6]).
        if self.geo3_active:
            k, geo3_diag = _geo3_lm_select_and_orthogonalize(
                k, beta, content_mask, self.geo3_chunk_size, self.geo3_k_sel,
                self.geo3_n_iter, self.geo3_resid_tol, self.geo3_selection_mode,
                forced_topk_idx=forced_topk_idx)
            self.geo3_last_diag = geo3_diag

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

    def __init__(self, d_model: int, d_state: int, conv_size: int = 4, num_heads: int = 1, ffn_mult: int = 4,
                 geo3_active: bool = False, geo3_k_sel: int = 16,
                 geo3_chunk_size: int = GEO3_LM_CHUNK_SIZE_DEFAULT, geo3_n_iter: int = 12,
                 geo3_resid_tol: float = 1e-2, geo3_selection_mode: str = "beta_topk",
                 geo3_excluded_token_ids: tuple = (EOT_TOKEN_ID,),
                 hard_select_active: bool = False, hard_select_mechanism: str = "hard_ste",
                 hard_select_k_sel: int = 32, hard_select_chunk_size: int = GEO3_LM_CHUNK_SIZE_DEFAULT,
                 hard_select_b_pinned: float | None = None, hard_select_seed: int = 0,
                 hard_select_tau_total_steps: int | None = None,
                 hard_select_tau_anneal_frac: float = 0.10,
                 hard_select_excluded_token_ids: tuple = (EOT_TOKEN_ID,)):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=1e-5)
        self.mixer = DeltaNetLMMixer(
            d_model, d_state, conv_size=conv_size, num_heads=num_heads,
            geo3_active=geo3_active, geo3_k_sel=geo3_k_sel,
            geo3_chunk_size=geo3_chunk_size, geo3_n_iter=geo3_n_iter,
            geo3_resid_tol=geo3_resid_tol, geo3_selection_mode=geo3_selection_mode,
            geo3_excluded_token_ids=geo3_excluded_token_ids,
            hard_select_active=hard_select_active, hard_select_mechanism=hard_select_mechanism,
            hard_select_k_sel=hard_select_k_sel, hard_select_chunk_size=hard_select_chunk_size,
            hard_select_b_pinned=hard_select_b_pinned, hard_select_seed=hard_select_seed,
            hard_select_tau_total_steps=hard_select_tau_total_steps,
            hard_select_tau_anneal_frac=hard_select_tau_anneal_frac,
            hard_select_excluded_token_ids=hard_select_excluded_token_ids)
        self.norm2 = RMSNorm(d_model, eps=1e-5)
        self.ffn = FFN(d_model, mult=ffn_mult)

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor | None = None,
                token_ids: torch.Tensor | None = None, step: int | None = None):
        o, final_state = self.mixer(self.norm1(x), initial_state=initial_state, token_ids=token_ids, step=step)
        x = x + o
        x = x + self.ffn(self.norm2(x))
        return x, final_state


class DeltaNetLM(nn.Module):
    """The full LM: embed -> N DeltaNetLMBlocks -> final norm -> TIED head
    (section 4.2's own cost-model assumption: "tied embedding is the
    head"). d_model=256/d_state=64/n_layers in {1,2} is this build's
    probe-tier scale-down of section 4.1's Wave-2 table (~14M params at
    n_layers=2, ~13.5M at n_layers=1 -- both clear CLAUDE.md's 10M hard
    floor).

    SCALE_TRANSFER_DESIGN.md sec 5.2 (Track C build-time harness change):
    n_layers is widened from the original {1,2} probe-tier ceiling to
    1..._MAX_N_LAYERS so the scaling-ladder's rung configs (sec 5.3: rung 1
    n_layers=12, rung 2 n_layers=16, rung 3 n_layers=22) can be built on
    this SAME architecture-generic harness -- no positional embedding and
    no other n_layers-dependent scaling exists anywhere in this class or
    DeltaNetLMBlock/DeltaNetLMMixer (verified by direct code read at build
    time: the stack is a plain pre-norm residual loop, checkpoint save/load
    round-trips via model.config()'s own n_layers field, generalizing
    automatically); widening the assert is therefore the FULL scope of
    sec 5.2's required change, exercised by smoke() item [11] at rung 1's
    real shapes (d_model=768/d_state=64/n_layers=12), item [12] at rung 2's
    (d_model=1536/d_state=128/n_layers=16), and item [13] at rung 3's
    (d_model=2560/d_state=128/n_layers=22) -- each run BEFORE any GPU time is
    spent on that rung's real wave, per sec 5.2's own blocking-smoke-test
    requirement (items [12]/[13] added by trackC-rung23-build)."""

    def __init__(self, vocab_size: int, d_model: int = 256, d_state: int = 64, n_layers: int = 2,
                 conv_size: int = 4, num_heads: int = 1, ffn_mult: int = 4,
                 geo3_active: bool = False, geo3_k_sel: int = 16,
                 geo3_chunk_size: int = GEO3_LM_CHUNK_SIZE_DEFAULT, geo3_n_iter: int = 12,
                 geo3_resid_tol: float = 1e-2, geo3_selection_mode: str = "beta_topk",
                 geo3_excluded_token_ids: tuple = (EOT_TOKEN_ID,),
                 hard_select_active: bool = False, hard_select_mechanism: str = "hard_ste",
                 hard_select_k_sel: int = 32, hard_select_chunk_size: int = GEO3_LM_CHUNK_SIZE_DEFAULT,
                 hard_select_b_pinned: float | None = None, hard_select_seed: int = 0,
                 hard_select_tau_total_steps: int | None = None,
                 hard_select_tau_anneal_frac: float = 0.10,
                 hard_select_excluded_token_ids: tuple = (EOT_TOKEN_ID,)):
        """geo3_* (SCALE_TRANSFER_DESIGN.md sec 4, Track B): threaded to EVERY
        block/mixer uniformly (this build does not support a per-layer geo3
        toggle -- out of sec 4's scope). OFF by default everywhere -- the
        default (geo3_active=False) path is byte-identical to the
        pre-Track-B code (smoke item [6]). hard_select_* (TRACKB_REDESIGN.md
        Rev 3): same uniform-per-layer threading, same off-by-default
        discipline -- see DeltaNetLMMixer.__init__'s own docstring for the
        full mechanism contract."""
        super().__init__()
        assert 1 <= n_layers <= _MAX_N_LAYERS, (
            f"n_layers={n_layers} outside the registered range [1,{_MAX_N_LAYERS}] -- "
            f"SCALE_TRANSFER_DESIGN.md sec 5.3's ladder tops out at rung 3's n_layers=22; "
            f"_MAX_N_LAYERS carries a small deliberate headroom margin above that. Widen "
            f"deliberately (and re-verify no n_layers-dependent assumption exists elsewhere, "
            f"sec 5.2), not by accident.")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.conv_size = conv_size
        self.num_heads = num_heads
        self.ffn_mult = ffn_mult
        self.geo3_active = geo3_active
        self.geo3_k_sel = geo3_k_sel
        self.geo3_chunk_size = geo3_chunk_size
        self.geo3_n_iter = geo3_n_iter
        self.geo3_resid_tol = geo3_resid_tol
        self.geo3_selection_mode = geo3_selection_mode
        self.geo3_excluded_token_ids = tuple(geo3_excluded_token_ids)
        self.hard_select_active = hard_select_active
        self.hard_select_mechanism = hard_select_mechanism
        self.hard_select_k_sel = hard_select_k_sel
        self.hard_select_chunk_size = hard_select_chunk_size
        self.hard_select_b_pinned = hard_select_b_pinned
        self.hard_select_seed = hard_select_seed
        self.hard_select_tau_total_steps = hard_select_tau_total_steps
        self.hard_select_tau_anneal_frac = hard_select_tau_anneal_frac
        self.hard_select_excluded_token_ids = tuple(hard_select_excluded_token_ids)

        self.embed = nn.Embedding(vocab_size, d_model)
        # AUDIT FIX-2 (2026-07-03): nn.Embedding's PyTorch default init is
        # N(0,1); with the TIED head that puts initial logit std ~= 16 and
        # initial loss ~= 202 nats instead of the ~ln(V) ~= 10.8 a
        # near-uniform softmax should give (auditor reproduced: rescaling
        # to std=0.02 gives 10.90). GPT-2's own convention: std=0.02.
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([
            DeltaNetLMBlock(
                d_model, d_state, conv_size=conv_size, num_heads=num_heads, ffn_mult=ffn_mult,
                geo3_active=geo3_active, geo3_k_sel=geo3_k_sel, geo3_chunk_size=geo3_chunk_size,
                geo3_n_iter=geo3_n_iter, geo3_resid_tol=geo3_resid_tol,
                geo3_selection_mode=geo3_selection_mode,
                geo3_excluded_token_ids=geo3_excluded_token_ids,
                hard_select_active=hard_select_active, hard_select_mechanism=hard_select_mechanism,
                hard_select_k_sel=hard_select_k_sel, hard_select_chunk_size=hard_select_chunk_size,
                hard_select_b_pinned=hard_select_b_pinned, hard_select_seed=hard_select_seed,
                hard_select_tau_total_steps=hard_select_tau_total_steps,
                hard_select_tau_anneal_frac=hard_select_tau_anneal_frac,
                hard_select_excluded_token_ids=hard_select_excluded_token_ids)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model, eps=1e-5)

    def config(self) -> dict:
        return {"vocab_size": self.vocab_size, "d_model": self.d_model, "d_state": self.d_state,
                "n_layers": self.n_layers, "conv_size": self.conv_size, "num_heads": self.num_heads,
                "ffn_mult": self.ffn_mult,
                "geo3_active": self.geo3_active, "geo3_k_sel": self.geo3_k_sel,
                "geo3_chunk_size": self.geo3_chunk_size, "geo3_n_iter": self.geo3_n_iter,
                "geo3_resid_tol": self.geo3_resid_tol, "geo3_selection_mode": self.geo3_selection_mode,
                "geo3_excluded_token_ids": list(self.geo3_excluded_token_ids),
                "hard_select_active": self.hard_select_active,
                "hard_select_mechanism": self.hard_select_mechanism,
                "hard_select_k_sel": self.hard_select_k_sel,
                "hard_select_chunk_size": self.hard_select_chunk_size,
                "hard_select_b_pinned": self.hard_select_b_pinned,
                "hard_select_seed": self.hard_select_seed,
                "hard_select_tau_total_steps": self.hard_select_tau_total_steps,
                "hard_select_tau_anneal_frac": self.hard_select_tau_anneal_frac,
                "hard_select_excluded_token_ids": list(self.hard_select_excluded_token_ids)}

    def forward(self, token_ids: torch.Tensor, initial_states: list | None = None,
                return_states: bool = False, step: int | None = None):
        """token_ids: (B,T) int64. initial_states: None (every layer starts fresh -- the ONLY
        mode training ever uses) or a list of length n_layers, each entry None or a
        (B,H,head_dim,head_dim) state (the intervention script's two-phase context/continuation
        use). Returns logits (B,T,vocab_size), and if return_states: also the list of each
        layer's FINAL state for this call.

        token_ids is now ALSO threaded down to every block/mixer (sec 4.2 item 3's EOT-exclusion
        mask) -- a pure additive plumbing change: non-geo3/non-hard-select mixers never read it.

        step (TRACKB_REDESIGN.md Rev 3): threaded to every block/mixer uniformly -- see
        DeltaNetLMMixer.forward's own docstring for the continuation contract eval-time probes
        must honor. None is a valid value ONLY for mechanisms that do not read it (hard_select
        inactive, or mechanism in {"hard_ste","entmax"})."""
        B, T = token_ids.shape
        x = self.embed(token_ids)
        if initial_states is None:
            initial_states = [None] * self.n_layers
        assert len(initial_states) == self.n_layers, \
            f"initial_states must have one entry per layer ({self.n_layers}), got {len(initial_states)}"
        final_states = []
        for blk, s0 in zip(self.blocks, initial_states):
            x, s_final = blk(x, initial_state=s0, token_ids=token_ids, step=step)
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
                             fracs=RANK_SAMPLE_FRACS, step: int | None = None) -> tuple[list, dict]:
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
        # TRACKB_REDESIGN.md Rev 3 build-phase note (1): `step` is threaded through so an
        # eval-time forward-hook probe like this one CONTINUES the same (seed,step)-keyed RNG
        # stream a hard_select_active "random_topk" mechanism uses at training time (see
        # DeltaNetLMMixer.forward's own docstring) -- None is safe/inert whenever hard_select is
        # inactive or uses a step-independent mechanism.
        _, states = model(batch, return_states=True, step=step)
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
def sample_geo3_diagnostics(model: DeltaNetLM, tokens: torch.Tensor, doc_offsets: torch.Tensor,
                             n_docs: int, seq_len: int, generator: torch.Generator,
                             step: int | None = None) -> dict:
    """SCALE_TRANSFER_DESIGN.md sec 4.4's two REQUIRED instrumentation items
    for a geo3-active model, both computed from ONE shared forward pass over
    n_docs document-aligned windows:

    (a) key_gram_deviation: the Newton-Schulz post-orthogonalization
        residual (model_rd.py's own Gram-deviation statistic, corrected to
        the masked-identity target ``||Q Q^T - diag(valid)||_F`` -- see
        _geo3_lm_select_and_orthogonalize's AUDIT ROUND-2 MAJOR-1 comment),
        summarized per layer.
    (b) a free-text drift diagnostic, ADAPTED from geo3_drift_diagnostic.py
        (sec 14.6's pinned cross-episode-drift statistic). DOCUMENTED SCOPING
        DEVIATION (sec 4.4's own words: "free text's chunk membership for
        any given token is far less stable than a synthetic episode's fixed
        K-cycle"): free text has no artificial K-cycle to resample
        identically, so this harvests NATURALLY-RECURRING token ids across
        the n_docs sampled documents' own selected (top-K_sel) positions and
        reports pooled pairwise cosine similarity of each recurring token's
        post-orthogonalization key direction across its distinct
        occurrences -- mean + p10 of the pooled distribution, mirroring
        geo3_simulator.pairwise_drift_stats' aggregation convention exactly,
        but over OPPORTUNISTIC natural recurrence rather than a controlled
        resample. Tokens that occur only once in the sample contribute
        nothing (need >=2 occurrences for a pairwise comparison) -- reported
        as n_recurring_tokens so a checkpoint with too little recurrence to
        be informative is visible in the JSON, not silently averaged over
        nothing.

    Requires at least one geo3_active mixer; asserts this (calling it on a
    non-geo3 model is a caller bug, not a legitimate empty-result case).

    step (TRACKB_REDESIGN.md Rev 3 -- this is the designated sec 4.4
    selected-key logger, the ONE instrument every Track B hard-selectivity
    cell's Gram-deviation bar reads (sec 5.2's F1 fix); build-phase note
    (1) requires this eval-time forward-hook probe to CONTINUE the same
    (seed,step)-keyed RNG stream a hard_select_active "random_topk"
    mechanism used at training time, never freeze a fresh draw per
    checkpoint -- callers MUST pass the checkpoint's own recorded training
    step here when probing a hard_select_active model with a step-dependent
    mechanism)."""
    assert any(blk.mixer.geo3_active for blk in model.blocks), \
        "sample_geo3_diagnostics requires at least one geo3_active mixer"
    model.eval()
    n = tokens.numel()
    eligible = doc_offsets[doc_offsets + seq_len + 1 <= n]
    assert eligible.numel() >= n_docs, \
        f"only {eligible.numel()} eligible docs for n_docs={n_docs} at seq_len={seq_len}"
    pick = torch.randint(0, eligible.numel(), (n_docs,), generator=generator, device=doc_offsets.device)
    starts = eligible[pick]
    offs = torch.arange(seq_len, device=tokens.device)
    idx = starts.unsqueeze(1) + offs.unsqueeze(0)
    batch = tokens[idx]                                              # (n_docs, seq_len)

    _ = model(batch, return_states=False, step=step)

    per_layer_gram = {}
    token_key_pairs: list[tuple[int, torch.Tensor]] = []
    for layer_idx, blk in enumerate(model.blocks):
        diag = blk.mixer.geo3_last_diag
        if diag is None:
            continue
        per_layer_gram[layer_idx] = {
            "resid_mean": diag["resid_mean"], "resid_max": diag["resid_max"],
            "resid_min": diag["resid_min"], "fallback_triggered": diag["fallback_triggered"],
            "n_fallback_denied_degenerate": diag["n_fallback_denied_degenerate"],
            "frac_valid_selections": diag["frac_valid_selections"],
        }
        topk_idx, valid_sel, Q = diag["_topk_idx"], diag["_valid_sel"], diag["_Q"]
        # topk_idx/valid_sel: (n_docs,n_chunks,k_sel,H) in-CHUNK indices; Q: (n_docs,n_chunks,H,k_sel,d)
        n_chunks, k_sel = diag["n_chunks"], diag["k_sel"]
        H = Q.shape[2]
        chunk_size = diag["chunk_size"]
        batch_view = batch.view(n_docs, n_chunks, chunk_size)
        for b in range(n_docs):
            for c in range(n_chunks):
                for h in range(H):
                    for s in range(k_sel):
                        if not bool(valid_sel[b, c, s, h]):
                            continue
                        tok = int(batch_view[b, c, topk_idx[b, c, s, h]].item())
                        token_key_pairs.append((tok, Q[b, c, h, s, :].detach()))

    by_token: dict[int, list[torch.Tensor]] = {}
    for tok, vec in token_key_pairs:
        by_token.setdefault(tok, []).append(vec)
    pooled = []
    n_recurring_tokens = 0
    for tok, vecs in by_token.items():
        if len(vecs) < 2:
            continue
        n_recurring_tokens += 1
        rows = torch.stack(vecs, dim=0)
        m = rows.shape[0]
        pw = F.cosine_similarity(rows.unsqueeze(0), rows.unsqueeze(1), dim=-1)
        pooled.append(pw[~torch.eye(m, dtype=torch.bool, device=rows.device)])
    if pooled:
        pooled_t = torch.cat(pooled)
        drift = {"mean": pooled_t.mean().item(), "p10": torch.quantile(pooled_t, 0.10).item(),
                 "n_recurring_tokens": n_recurring_tokens, "n_pooled_pairs": int(pooled_t.numel())}
    else:
        drift = {"n_recurring_tokens": 0, "n_pooled_pairs": 0,
                 "note": "no token recurred >=2x in this sample -- drift statistic undefined"}
    model.train()
    return {"per_layer_gram_deviation": per_layer_gram, "drift": drift,
            "n_docs": n_docs, "doc_start_digest": window_digest(starts)}


@torch.no_grad()
def sample_hard_select_diagnostics(model: DeltaNetLM, tokens: torch.Tensor, doc_offsets: torch.Tensor,
                                    n_docs: int, seq_len: int, generator: torch.Generator,
                                    step: int, prev_selection: dict | None) -> tuple[dict, dict]:
    """AUDIT FIX (independent audit 2026-07-04 -- F2): TRACKB_REDESIGN.md
    sec 4.3's checkpoint-time selection diagnostics -- churn rate,
    positional TV-from-uniform, support size -- computed on a FIXED probe
    batch (the caller seeds `generator` from corpus_fixed_seed WITHOUT a
    step offset, so the SAME document windows are probed at every log
    point: sec 4.3's own "same batch every log point, so churn measures the
    mechanism, not data variation"). Mirrors sample_geo3_diagnostics'
    document-aligned sampling idiom exactly.

    Requires at least one mixer with hard_select_active OR
    selection_probe_k_sel set (the unmasked pilot's implicit-probe path);
    asserts this. `step` is the CURRENT training step (threaded to
    forward() -- the build-phase-note-(1) continuation contract for
    step-dependent mechanisms). `prev_selection`: {layer_idx: cpu tensor
    (B,n_chunks,H,k_sel)} from the PREVIOUS checkpoint's call (None at the
    first checkpoint -- churn_vs_prev is then None, never fabricated).

    Per-layer outputs: churn_vs_prev (None for entmax -- variable support
    has no fixed-K set difference; sec 4.3 assigns the churn bar to
    candidate 1 + candidate 4's hard-snap only), tv_from_uniform,
    support_median, support_p10, mechanism. Invalid (content-starved)
    slots are included in the churn set comparison as-is: with a FIXED
    probe batch the content mask is identical at every log point, so their
    contribution is stable across checkpoints -- a documented
    approximation, not a silent one.

    Returns (diag_dict_for_result_json, new_prev_selection)."""
    probed = [blk.mixer for blk in model.blocks
              if blk.mixer.hard_select_active or blk.mixer.selection_probe_k_sel is not None]
    assert probed, ("sample_hard_select_diagnostics requires hard_select_active or "
                    "selection_probe_k_sel on at least one mixer (caller bug)")
    model.eval()
    n = tokens.numel()
    eligible = doc_offsets[doc_offsets + seq_len + 1 <= n]
    assert eligible.numel() >= n_docs, \
        f"only {eligible.numel()} eligible docs for n_docs={n_docs} at seq_len={seq_len}"
    pick = torch.randint(0, eligible.numel(), (n_docs,), generator=generator, device=doc_offsets.device)
    starts = eligible[pick]
    offs = torch.arange(seq_len, device=tokens.device)
    batch = tokens[starts.unsqueeze(1) + offs.unsqueeze(0)]          # (n_docs, seq_len)

    _ = model(batch, return_states=False, step=step)

    per_layer = {}
    new_prev: dict = {}
    for layer_idx, blk in enumerate(model.blocks):
        diag = blk.mixer.hard_select_last_diag
        if diag is None:
            continue
        chunk_size = blk.mixer.hard_select_chunk_size
        if "topk_idx" in diag:
            topk_idx, valid_sel = diag["topk_idx"], diag["valid_sel"]   # (B,n_chunks,k_sel,H)
            k_sel = topk_idx.shape[2]
            sel = topk_idx.permute(0, 1, 3, 2).contiguous()             # (B,n_chunks,H,k_sel)
            churn = None
            if prev_selection is not None and layer_idx in prev_selection:
                churn = churn_rate(prev_selection[layer_idx].to(sel.device), sel, k_sel).mean().item()
            new_prev[layer_idx] = sel.detach().cpu()
            counts = torch.bincount(topk_idx[valid_sel].reshape(-1),
                                     minlength=chunk_size).float()      # offset marginal, valid only
            tv = tv_distance_from_uniform(counts)
            support = valid_sel.sum(dim=2).float()                      # (B,n_chunks,H)
        else:
            # entmax: no fixed top-K set -- selection IS the support (beta's exact zeros)
            beta_used = diag["_beta"]                                    # (B,T,H)
            support = support_size(beta_used, chunk_size).float()
            Bb, Tb, Hb = beta_used.shape
            supp_c = (beta_used > 0).view(Bb, Tb // chunk_size, chunk_size, Hb)
            counts = supp_c.sum(dim=(0, 1, 3)).float()
            tv = tv_distance_from_uniform(counts)
            churn = None
        support_flat = support.reshape(-1)
        per_layer[str(layer_idx)] = {
            "churn_vs_prev": churn,
            "tv_from_uniform": tv,
            "support_median": support_flat.median().item(),
            "support_p10": torch.quantile(support_flat, 0.10).item(),
            "mechanism": diag.get("mechanism"),
        }
    model.train()
    return ({"per_layer": per_layer, "n_docs": n_docs, "doc_start_digest": window_digest(starts)},
            new_prev)


@torch.no_grad()
def eval_loss(model: DeltaNetLM, tokens: torch.Tensor, doc_offsets: torch.Tensor, n_batches: int,
              batch_size: int, seq_len: int, generator: torch.Generator,
              step: int | None = None) -> tuple[float, dict]:
    """Cross-entropy val loss, capped batch size / batch count (house VRAM
    rule: eval batches capped independently of the train batch -- the 50K-
    vocab logits tensor is the VRAM bottleneck, not model activations).

    AUDIT FIX-1: `generator` must be seeded from corpus_fixed_seed(...) so
    every training seed evaluates on the SAME windows. Returns
    (mean_loss, info) where info carries the FIX-1 verification digest of
    the drawn window starts and the FIX-3 boundary-contamination stats.

    step (TRACKB_REDESIGN.md Rev 3, build-phase note (1)): threaded through
    to model() so a hard_select_active model's step-dependent mechanism
    continues its training-time RNG stream during eval, rather than
    freezing a fresh draw -- inert when hard_select is inactive."""
    model.eval()
    total, count = 0.0, 0
    all_starts = []
    for _ in range(n_batches):
        x, y, starts = get_batch(tokens, batch_size, seq_len, generator, return_starts=True)
        all_starts.append(starts)
        logits = model(x, step=step)
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
                      train_contamination: dict | None = None,
                      nan_probe: dict | None = None) -> dict:
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
        "geo3_lm": {
            "active": getattr(args, "use_geo3_lm", False),
            "k_sel": getattr(args, "geo3_k_sel", None),
            "chunk_size": getattr(args, "geo3_chunk_size", None),
            "n_iter": getattr(args, "geo3_n_iter", None),
            "resid_tol": getattr(args, "geo3_resid_tol", None),
            "selection_mode": getattr(args, "geo3_selection", None),
        },
        # TRACKB_REDESIGN.md Rev 3: hard-selectivity mechanism config, mirroring geo3_lm's own
        # always-present (even when inactive) block convention.
        "hard_select": {
            "active": getattr(args, "hard_select_active", False),
            "mechanism": getattr(args, "hard_select_mechanism", None),
            "k_sel": getattr(args, "hard_select_k_sel", None),
            "chunk_size": getattr(args, "hard_select_chunk_size", None),
            "b_pinned": getattr(args, "hard_select_b_pinned", None),
            "seed": getattr(args, "hard_select_seed", None),
            "tau_anneal_frac": getattr(args, "hard_select_tau_anneal_frac", None),
        },
        # TRACKB sec 5.1 (Rev 3 NEW-7): the stability smoke's positive-control summary --
        # non-None ONLY when --nan-probe-counter ran (a probative/NON-PROBATIVE verdict the
        # smoke's pass bars are conditioned on).
        "nan_probe_positive_control": nan_probe,
        "trajectory": trajectory, "checkpoints": checkpoints,
        "wall_s": time.time() - t0, "timed_out": timed_out,
        # SCALE_TRANSFER_DESIGN.md sec 5.6 Wave -1 (Track C calibration): peak CUDA memory over
        # this run (reset at train()'s own start, see the reset_peak_memory_stats() call above) --
        # None on CPU (smoke's own CPU-less path never reaches here; defensive only).
        "peak_memory_allocated_bytes": (torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None),
        "peak_memory_reserved_bytes": (torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None),
    }
    # TRACKB_REDESIGN.md Rev 2 -- M5 / sec 5.1: the Cell-3 override stamp, written AT ASSEMBLY
    # TIME (never post-hoc) via a CLI-threaded --gate-override-reason. gate_override is ALWAYS
    # present (True or False) so its absence can never be misread as "a clean, non-override run"
    # (mirrors key_anchoring.py's assemble_claim_tier_fields / unblind_override precedent, R4-1).
    gate_override_reason = getattr(args, "gate_override_reason", None)
    stamp = trackb_override_stamp_payload(reason=gate_override_reason) if gate_override_reason else None
    result.update(trackb_assemble_gate_override_fields(stamp))
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
    # SCALE_TRANSFER_DESIGN.md sec 5.6 Wave -1 (Track C): "measure real tok/s + MEMORY before
    # committing full budget" -- reset peak stats now (before the first training step) so
    # peak_memory_*_bytes in the result JSON reflects THIS run only, not whatever a prior run in
    # the same process left behind.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
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
    hs_prev_selection: dict | None = None      # TRACKB sec 4.3 (F2): previous checkpoint's
                                               # selected sets, per layer -- churn's own state
    nan_probe_counter = None                   # TRACKB sec 5.1 (Rev 3 NEW-7) positive control
    if getattr(args, "nan_probe_counter", False):
        assert any(blk.mixer.geo3_active for blk in model.blocks), (
            "--nan-probe-counter only means anything on a geo3-active run (it counts duplicates "
            "among geo3's SELECTED rows) -- refusing a silently-inert flag")
        from wave_neg1_trackb import NanStabilityProbeCounter   # lazy: avoids a circular import
                                                                 # (wave_neg1_trackb imports THIS module)
        nan_probe_counter = NanStabilityProbeCounter()
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

        logits = model(x, step=step)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1

        # TRACKB_REDESIGN.md sec 5.1 (Rev 3 NEW-7) positive control: per TRAINING forward call,
        # count exact duplicates among the geo3-SELECTED raw key rows (the >=25-calls-with->=6-
        # dup-rows floor the stability smoke's probativeness is measured against). Only when
        # --nan-probe-counter is passed (the stability-smoke cell's own flag; per-call cost is a
        # CPU byte-compare over the gathered rows, acceptable for a 2,000-step smoke, not free
        # for a full run).
        if nan_probe_counter is not None:
            for blk in model.blocks:
                gdiag = blk.mixer.geo3_last_diag
                if gdiag is not None and "_k_selected_raw" in gdiag:
                    nan_probe_counter.observe(gdiag["_k_selected_raw"])

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
            # TRACKB_REDESIGN.md Rev 3 build-phase note (1): every eval-time forward-hook probe
            # in this checkpoint block passes THIS SAME `step` (the training loop's own counter,
            # not a separately-derived probe step) -- the mechanism that makes the continuation
            # contract hold trivially: a hard_select_active model's random_topk/soft_topk_
            # comparator mechanism draws IDENTICALLY here as it did in this iteration's own
            # training forward call above.
            val_loss_same, eval_info_same = eval_loss(model, val_tokens_same, val_offs_same,
                                                       args.eval_batches, args.eval_batch_size,
                                                       args.seq_len, eval_gen_same, step=step)
            val_loss_other, eval_info_other = eval_loss(model, val_tokens_other, val_offs_other,
                                                         args.eval_batches, args.eval_batch_size,
                                                         args.seq_len, eval_gen_other, step=step)

            rank_gen_same = torch.Generator(device=device).manual_seed(
                corpus_fixed_seed(args.corpus) + 20_000 + step)
            rank_gen_other = torch.Generator(device=device).manual_seed(
                corpus_fixed_seed(other_corpus) + 20_000 + step)
            rank_same, rank_sum_same = sample_state_rank_stats(
                model, val_tokens_same, val_offs_same, args.rank_sample_docs, args.seq_len, rank_gen_same,
                step=step)
            rank_other, rank_sum_other = sample_state_rank_stats(
                model, val_tokens_other, val_offs_other, args.rank_sample_docs, args.seq_len, rank_gen_other,
                step=step)

            # Track B / geo3-in-LM (sec 4.4's required instrumentation): only computed when
            # geo3 is active on this model -- a separate seeded generator (offset 40_000, never
            # colliding with eval's 10_000 or rank's 20_000 offsets) so this diagnostic is
            # deterministic/reproducible independent of what else was sampled this checkpoint.
            geo3_diag = None
            if any(blk.mixer.geo3_active for blk in model.blocks):
                geo3_gen_same = torch.Generator(device=device).manual_seed(
                    corpus_fixed_seed(args.corpus) + 40_000 + step)
                geo3_gen_other = torch.Generator(device=device).manual_seed(
                    corpus_fixed_seed(other_corpus) + 40_000 + step)
                geo3_diag = {
                    args.corpus: sample_geo3_diagnostics(
                        model, val_tokens_same, val_offs_same, args.rank_sample_docs,
                        args.seq_len, geo3_gen_same, step=step),
                    other_corpus: sample_geo3_diagnostics(
                        model, val_tokens_other, val_offs_other, args.rank_sample_docs,
                        args.seq_len, geo3_gen_other, step=step),
                }

            # AUDIT FIX (independent audit 2026-07-04, F2): TRACKB_REDESIGN.md sec 4.3's
            # checkpoint-time selection diagnostics (churn/TV/support), computed on a FIXED probe
            # batch: seeded from corpus_fixed_seed + 50_000 with NO step offset (unlike every
            # other sampler in this block) -- sec 4.3's own registered requirement ("same batch
            # every log point, so churn measures the mechanism, not data variation"). Offset
            # 50_000 never collides with eval's 10_000, rank's 20_000, or geo3's 40_000 families.
            # Runs when hard_select is active OR the --trackb-selection-probe implicit-probe is
            # set (the unmasked reference pilot -- churn Null A's data source).
            hard_select_diag = None
            if any(blk.mixer.hard_select_active or blk.mixer.selection_probe_k_sel is not None
                   for blk in model.blocks):
                hs_gen = torch.Generator(device=device).manual_seed(
                    corpus_fixed_seed(args.corpus) + 50_000)
                hard_select_diag, hs_prev_selection = sample_hard_select_diagnostics(
                    model, val_tokens_same, val_offs_same, args.rank_sample_docs, args.seq_len,
                    hs_gen, step, hs_prev_selection)

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
                "geo3_diagnostics": geo3_diag,             # sec 4.4, only non-None when geo3-active
                "hard_select_diagnostics": hard_select_diag,   # TRACKB sec 4.3 (F2 audit fix), only
                                                                # non-None when hard_select/probe active
                "checkpoint_path": ckpt_path,
            }
            checkpoints.append(res)
            partial = _assemble_result(args, run_name, other_corpus, trajectory, checkpoints, n_params,
                                        t0, timed_out=False, steps_completed=steps_completed,
                                        complete=False, n_skipped=n_skipped,
                                        train_contamination=_train_contamination(
                                            train_windows_total, train_windows_crossing,
                                            train_tokens_cross_frac_sum),
                                        nan_probe=(nan_probe_counter.summary()
                                                    if nan_probe_counter is not None else None))
            _dump(out_path, partial)
            geo3_log = ""
            if geo3_diag is not None:
                gl0 = geo3_diag[args.corpus]["per_layer_gram_deviation"].get(0, {})
                gdrift = geo3_diag[args.corpus]["drift"]
                geo3_log = (f"  geo3_resid_L0[{args.corpus}]={gl0.get('resid_mean', float('nan')):.4f}  "
                            f"geo3_drift_mean[{args.corpus}]={gdrift.get('mean', float('nan'))}")
            print(f"  [checkpoint step {step}] val_loss[{args.corpus}]={val_loss_same:.4f}  "
                  f"val_loss[{other_corpus}]={val_loss_other:.4f}  "
                  f"rank_L0_H0_f1.0[{args.corpus}]="
                  f"{res['rank_stats_summary'][args.corpus].get('L0_H0_f1.0', float('nan')):.3f}  "
                  f"eval_digest[{args.corpus}]={eval_info_same['eval_window_digest']}{geo3_log}", flush=True)

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
                                 train_tokens_cross_frac_sum),
                             nan_probe=(nan_probe_counter.summary()
                                         if nan_probe_counter is not None else None))


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
    # sec 5.6 Wave -1 (Track C calibration): peak-memory fields present, finite, positive on CUDA
    assert result["peak_memory_allocated_bytes"] is not None and result["peak_memory_allocated_bytes"] > 0
    assert result["peak_memory_reserved_bytes"] >= result["peak_memory_allocated_bytes"]
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

    print("\n[3c] geo3-ACTIVE mini end-to-end training loop (REUSES item [3]'s synthetic EOT corpus) "
          "-- checkpoint-time instrumentation (sample_geo3_diagnostics, sec 4.4) must run inside the "
          "REAL train() checkpoint block without crashing and produce a well-formed geo3_diagnostics "
          "entry in the result JSON. Items [6]-[10] only exercise forward/backward in isolation; this "
          "is the only smoke item that exercises the train()-loop wiring itself (the "
          "`if any(blk.mixer.geo3_active ...): sample_geo3_diagnostics(...)` block).")
    torch.manual_seed(1)
    model3c = DeltaNetLM(V, d_model=64, d_state=64, n_layers=1, conv_size=4, geo3_active=True,
                          geo3_k_sel=16, geo3_chunk_size=64, geo3_n_iter=12, geo3_resid_tol=1e-2,
                          geo3_selection_mode="beta_topk").to(device)
    args3c = Args()
    args3c.corpus, args3c.seed = "openr1", 0
    args3c.use_geo3_lm, args3c.geo3_k_sel = True, 16
    args3c.geo3_chunk_size, args3c.geo3_n_iter = 64, 12
    args3c.geo3_resid_tol, args3c.geo3_selection = 1e-2, "beta_topk"
    args3c.d_model, args3c.d_state, args3c.n_layers = 64, 64, 1
    args3c.conv_size, args3c.num_heads, args3c.ffn_mult = 4, 1, 4
    args3c.seq_len, args3c.batch_size = seq_len_smoke, 4
    args3c.lr, args3c.weight_decay, args3c.warmup_steps = 3e-4, 0.0, 2
    args3c.eval_batches, args3c.eval_batch_size = 2, 4
    args3c.rank_sample_docs = 3
    args3c.steps, args3c.log_every, args3c.ckpt_every = 8, 4, 8

    result3c = train(model3c, args3c, corpus_a, offs_a, (corpus_a, offs_a), (corpus_b, offs_b),
                      "wikitext", device, "smoke_run_geo3", out_path=None, ckpt_dir=None)
    assert result3c["steps_completed"] == 8 and result3c["complete"] is True
    assert result3c["geo3_lm"]["active"] is True and result3c["geo3_lm"]["k_sel"] == 16
    ck3c = result3c["checkpoints"][-1]
    assert ck3c["geo3_diagnostics"] is not None, "geo3_diagnostics missing from checkpoint despite geo3_active=True"
    for corpus_name in ("openr1", "wikitext"):
        gd = ck3c["geo3_diagnostics"][corpus_name]
        assert "per_layer_gram_deviation" in gd and "drift" in gd
        assert 0 in gd["per_layer_gram_deviation"], "layer 0's gram deviation missing from the diagnostic"
        resid_mean = gd["per_layer_gram_deviation"][0]["resid_mean"]
        assert resid_mean == resid_mean and resid_mean >= 0.0, f"non-finite/negative resid_mean: {resid_mean}"
        assert "n_recurring_tokens" in gd["drift"]
    # also confirm the NON-geo3 corpus_b-derived checkpoint from item [3] above did NOT carry
    # geo3_diagnostics (sensitivity control: the field is genuinely conditional, not always-on)
    assert ck["geo3_diagnostics"] is None, \
        "item [3]'s geo3_active=False run unexpectedly carries geo3_diagnostics -- the conditional is broken"
    print(f"  8-step geo3-active mini run: complete={result3c['complete']}, "
          f"geo3_lm.active={result3c['geo3_lm']['active']}, "
          f"resid_mean(L0,openr1)={ck3c['geo3_diagnostics']['openr1']['per_layer_gram_deviation'][0]['resid_mean']:.4f}, "
          f"drift[openr1].n_recurring_tokens={ck3c['geo3_diagnostics']['openr1']['drift']['n_recurring_tokens']}; "
          f"item [3]'s geo3_active=False checkpoint correctly carries geo3_diagnostics=None")

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

    print("\n[4b] GEO3-ACTIVE checkpoint round-trip (the exact path lm_intervene_rd.py's Wave 3 "
          "truncation grid depends on: DeltaNetLM(**ckpt['config']) must reconstruct a model with "
          "geo3_active/k_sel/chunk_size/selection_mode etc. all correctly restored, not just the "
          "geo3_active=False case item [4] above covers)")
    model2b = DeltaNetLM(V, d_model=64, d_state=64, n_layers=1, conv_size=4, geo3_active=True,
                          geo3_k_sel=16, geo3_chunk_size=64, geo3_n_iter=12, geo3_resid_tol=1e-2,
                          geo3_selection_mode="naive_window").to(device)
    result2b = train(model2b, args, corpus_a, offs_a, (corpus_a, offs_a), (corpus_b, offs_b),
                      "wikitext", device, "smoke_ckpt_geo3", out_path=None, ckpt_dir=tmpdir)
    ckpt_path_b = result2b["checkpoints"][-1]["checkpoint_path"]
    loaded_b = torch.load(ckpt_path_b, map_location=device)
    assert loaded_b["config"] == model2b.config()
    assert loaded_b["config"]["geo3_active"] is True
    assert loaded_b["config"]["geo3_k_sel"] == 16
    assert loaded_b["config"]["geo3_selection_mode"] == "naive_window"
    model3b = DeltaNetLM(**loaded_b["config"]).to(device)
    assert model3b.blocks[0].mixer.geo3_active is True, \
        "reconstructed model's mixer did not inherit geo3_active from the checkpoint's saved config"
    model3b.load_state_dict(loaded_b["model_state_dict"])
    model2b.eval()
    model3b.eval()
    tok_b = torch.randint(1, V, (2, 128), device=device)              # geo3 needs token_ids in forward
    with torch.no_grad():
        logits2b = model2b(tok_b)
        logits3b = model3b(tok_b)
    assert torch.equal(logits2b, logits3b), \
        "GEO3-ACTIVE checkpoint round-trip mismatch: reloaded model differs -- Wave 3's truncation " \
        "grid would silently score the WRONG (non-reconstructed) construction"
    print(f"  geo3-active round-trip: {ckpt_path_b} loaded into a fresh model with geo3_active/k_sel/"
          f"selection_mode restored from the saved config, logits BIT-IDENTICAL "
          f"(max abs diff {(logits2b - logits3b).abs().max().item():.2e})")

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

    # -------------------------------------------------------------------
    # Track B / geo3-in-LM (SCALE_TRANSFER_DESIGN.md sec 4) smoke items,
    # required by the build brief: [6] default-path bit-identity regression,
    # [7] Newton-Schulz convergence at LM shapes, [8] EOT-exclusion
    # correctness (including the negative/degenerate case, run to
    # completion), [9] naive_window vs beta_topk selection-mode distinction,
    # [10] geo3-active forward/backward/grad-finite + an on/off sensitivity
    # control.
    # -------------------------------------------------------------------

    print("\n[6] DEFAULT-PATH REGRESSION (geo3_active=False must be BIT-IDENTICAL to a hand-rolled "
          "INDEPENDENT reimplementation of the pre-Track-B forward computation -- proves inserting "
          "the geo3 branch did not perturb the default path, not merely that geo3_active=False "
          "skips executing it)")
    torch.manual_seed(42)
    V6 = 500
    with torch.no_grad():
        model6 = DeltaNetLM(V6, d_model=64, d_state=64, n_layers=1, conv_size=4, geo3_active=False).to(device)
        assert model6.geo3_active is False and model6.blocks[0].mixer.geo3_active is False
        x6 = torch.randint(0, V6, (3, _MIN_KERNEL_T), device=device)
        logits6 = model6(x6)

        mixer6 = model6.blocks[0].mixer
        x_embed6 = model6.embed(x6)
        a6 = model6.blocks[0].norm1(x_embed6)
        B6, T6, _ = a6.shape
        q_ref, _ = mixer6.q_conv1d(mixer6.q_proj(a6))
        k_ref, _ = mixer6.k_conv1d(mixer6.k_proj(a6))
        v_ref, _ = mixer6.v_conv1d(mixer6.v_proj(a6))
        beta_ref = torch.sigmoid(mixer6.b_proj(a6))
        q_ref = q_ref.reshape(B6, T6, mixer6.num_heads, mixer6.head_dim)
        k_ref = k_ref.reshape(B6, T6, mixer6.num_heads, mixer6.head_dim)
        v_ref = v_ref.reshape(B6, T6, mixer6.num_heads, mixer6.head_dim)
        q_bf, k_bf, v_bf, beta_bf = (t.to(torch.bfloat16) for t in (q_ref, k_ref, v_ref, beta_ref))
        o_ref, _ = chunk_delta_rule(q=q_bf, k=k_bf, v=v_bf, beta=beta_bf, initial_state=None,
                                     output_final_state=True, use_qk_l2norm_in_kernel=True)
        o_ref = o_ref.float()
        o_ref = mixer6.o_norm(o_ref).reshape(B6, T6, mixer6.d_state)
        o_ref = mixer6.o_proj(o_ref)
        x_ref = x_embed6 + o_ref
        x_ref = x_ref + model6.blocks[0].ffn(model6.blocks[0].norm2(x_ref))
        x_ref = model6.norm_f(x_ref)
        logits_ref = F.linear(x_ref, model6.embed.weight)
    assert torch.equal(logits6, logits_ref), (
        "DEFAULT-PATH REGRESSION FAILED: geo3_active=False forward diverges from the hand-rolled "
        "pre-Track-B reference computation")
    print(f"  geo3_active=False forward is BIT-IDENTICAL to the independent hand-rolled reference "
          f"(max abs diff {(logits6 - logits_ref).abs().max().item():.2e})")

    print("\n[7] Newton-Schulz convergence AT LM SHAPES (head_dim=64, K_sel in {16,32} -- Track B's "
          "own registered d_state=64 config, sec 4.5) -- random unit-normalized keys, n_iter=12 "
          "(K=16's registered value) and n_iter=20 (K=32's registered escalation, sec 1.1)")
    torch.manual_seed(7)
    for k_sel_t, n_iter_t in ((16, 12), (32, 12), (32, 20)):
        A7 = F.normalize(torch.randn(5, k_sel_t, 64, device=device), dim=-1)
        Q7, resid7 = newton_schulz_orthogonalize(A7, n_iter=n_iter_t)
        assert torch.isfinite(Q7).all() and torch.isfinite(resid7).all()
        print(f"  K_sel={k_sel_t} n_iter={n_iter_t}: resid mean={resid7.mean().item():.4f} "
              f"max={resid7.max().item():.4f} (finite; sec 1.1's K=32/n_iter=12 fallback trigger "
              f"rate is small -- 56-374/20000 steps -- so a single random draw usually still "
              f"converges, informational only, not asserted)")

    print("\n[8] EOT-EXCLUSION CORRECTNESS on a HAND-CONSTRUCTED chunk (sec 4.2 item 3): a "
          "HIGH-beta EOT position must NEVER be selected, and (the degenerate case) a chunk with "
          "FEWER real content positions than k_sel must leave every shortfall position's raw key "
          "EXACTLY untouched -- not corrupted by a spurious Newton-Schulz zero-row output. A "
          "negative-control case run to completion, per house rule.")
    torch.manual_seed(8)
    B8, chunk8, H8, d8 = 1, 8, 1, 8
    k_raw8 = torch.randn(B8, chunk8, H8, d8, device=device)
    beta8 = torch.rand(B8, chunk8, H8, device=device)
    token_ids8 = torch.randint(1, 1000, (B8, chunk8), device=device)
    EOT8 = 999
    token_ids8[0, 0] = EOT8
    beta8[0, 0, 0] = 10.0                                    # far above every other position's beta
    content_mask8 = (token_ids8 != EOT8)
    assert content_mask8[0, 0].item() is False and int(content_mask8.sum().item()) == chunk8 - 1

    k_out8, diag8 = _geo3_lm_select_and_orthogonalize(k_raw8, beta8, content_mask8, chunk8, 6, 12,
                                                        1e-2, "beta_topk")
    assert torch.equal(k_out8[0, 0, 0], k_raw8[0, 0, 0]), \
        "EOT position's key was MODIFIED despite the sec 4.2 item 3 hard-exclusion rule"
    print(f"  positive control: EOT position (beta=10.0, would rank #1) key UNTOUCHED "
          f"(max diff {(k_out8[0, 0, 0] - k_raw8[0, 0, 0]).abs().max().item():.2e})")

    selected_content_changed = any(
        bool(content_mask8[0, t]) and not torch.equal(k_out8[0, t, 0], k_raw8[0, t, 0])
        for t in range(chunk8))
    assert selected_content_changed, \
        "sensitivity control FAILED: no content position's key changed at all -- item [8]'s no-op checks would be vacuous"
    print("  sensitivity control: at least one SELECTED content position's key DID change (the "
          "no-op checks above have teeth)")

    token_ids8b = token_ids8.clone()
    token_ids8b[0, 1] = EOT8
    token_ids8b[0, 2] = EOT8                                  # now only 5 real content positions
    content_mask8b = (token_ids8b != EOT8)
    assert int(content_mask8b.sum().item()) == chunk8 - 3
    k_out8b, diag8b = _geo3_lm_select_and_orthogonalize(k_raw8, beta8, content_mask8b, chunk8, 6, 12,
                                                          1e-2, "beta_topk")               # k_sel=6 > 5 available
    for eot_pos in (0, 1, 2):
        assert torch.equal(k_out8b[0, eot_pos, 0], k_raw8[0, eot_pos, 0]), \
            f"degenerate case: EOT position {eot_pos}'s key was MODIFIED (insufficient-content shortfall leaked)"
    assert diag8b["frac_valid_selections"] < 1.0, \
        "test construction bug: the degenerate (insufficient-content) case produced no invalid selection slot"
    print(f"  degenerate case (5 content positions < k_sel=6): ALL 3 EOT positions' keys UNTOUCHED "
          f"(frac_valid_selections={diag8b['frac_valid_selections']:.3f} < 1.0, confirming the "
          f"shortfall was real, not vacuously avoided)")

    print("\n[8b] AUDIT ROUND-2 MAJOR-1 REGRESSION (independent audit, 2026-07-04): a degenerate "
          "episode with MANY (8 >= the audit's measured NaN onset of 6) invalid zero-row slots, "
          "plus a FORCED eigh fallback (resid_tol=0.0 -- every episode's corrected residual "
          "exceeds it), must produce (i) FINITE gradients everywhere (the pre-fix code NaN'd "
          "through _polar_via_eigh's backward at >=6 coincident eps-eigenvalues -- empirically "
          "reproduced by the audit, persists in fp64), (ii) an exact no-op on every invalid slot, "
          "and (iii) a nonzero n_fallback_denied_degenerate count (the degenerate episode is "
          "DENIED eigh and keeps its Newton-Schulz output) while the fully-valid episode DOES go "
          "through eigh (fallback_triggered=True).")
    torch.manual_seed(81)
    B8c, chunk8c, H8c, d8c, k_sel8c = 2, 16, 1, 16, 10
    k_raw8c_base = torch.randn(B8c, chunk8c, H8c, d8c, device=device)
    # Episode 1's TWO valid keys are EXACT DUPLICATES: two identical rows can NEVER be
    # orthogonalized (QQ^T's off-diagonal equals its diagonal for identical rows; the corrected
    # residual is provably >= 1), so the degenerate episode GENUINELY fails NS and would demand
    # the fallback -- exercising the denial branch for real. (A first draft of this test used
    # random valid keys; the negative run caught that their block CONVERGES, corrected resid
    # clamps to exactly 0.0, the episode never wants the fallback, and n_fallback_denied stays 0 --
    # the denial branch was silently unexercised. Run the negative test to completion, per house
    # rule; this is that lesson applied to the test itself.)
    k_raw8c_base[1, 1] = k_raw8c_base[1, 0]
    k_raw8c = k_raw8c_base.clone().requires_grad_(True)
    beta8c = torch.rand(B8c, chunk8c, H8c, device=device)
    content8c = torch.ones(B8c, chunk8c, dtype=torch.bool, device=device)
    content8c[1, 2:] = False                    # episode 1: only 2 content positions -> 8 invalid slots
    k_out8c, diag8c = _geo3_lm_select_and_orthogonalize(k_raw8c, beta8c, content8c, chunk8c,
                                                          k_sel8c, 12, 0.0, "beta_topk")
    assert diag8c["fallback_triggered"] is True, \
        "test construction bug: resid_tol=0.0 failed to force the fallback -- [8b] would be vacuous"
    assert diag8c["n_fallback_denied_degenerate"] >= 1, (
        "the degenerate episode was NOT denied the eigh fallback -- the MAJOR-1 fix is not active "
        f"(diag: {['%s=%s' % (k, v) for k, v in diag8c.items() if not k.startswith('_')]})")
    loss8c = k_out8c.square().sum()
    loss8c.backward()
    assert k_raw8c.grad is not None and torch.isfinite(k_raw8c.grad).all(), \
        "MAJOR-1 REGRESSION: NaN/Inf gradient through the forced-fallback degenerate-episode path"
    with torch.no_grad():
        for t in range(2, chunk8c):             # every excluded position in episode 1: exact no-op
            assert torch.equal(k_out8c[1, t, 0], k_raw8c[1, t, 0]), \
                f"degenerate episode's excluded position {t} was modified under forced fallback"
        ep1_changed = any(not torch.equal(k_out8c[1, t, 0], k_raw8c[1, t, 0]) for t in range(2))
    assert ep1_changed, \
        "sensitivity control FAILED: the degenerate episode's VALID positions were untouched -- " \
        "the denial branch would be vacuously passing"
    print(f"  forced fallback (resid_tol=0.0): fallback_triggered={diag8c['fallback_triggered']}, "
          f"n_fallback_denied_degenerate={diag8c['n_fallback_denied_degenerate']} (episode with 8 "
          f"invalid slots kept its NS output), ALL grads FINITE "
          f"(max |grad| {k_raw8c.grad.abs().max().item():.3e}), every excluded position an exact "
          f"no-op, valid positions genuinely changed")

    print("\n[8c] num_heads>1 + geo3_active is REFUSED at construction (untested-at-scope guard, "
          "audit round-2 recommendation): d_state=128/num_heads=2 gives head_dim=64 which PASSES "
          "_SAFE_D_STATE -- only the new scope guard stands between this and an accidental launch")
    guard8c_raised = False
    try:
        DeltaNetLMMixer(d_model=64, d_state=128, num_heads=2, geo3_active=True)
    except AssertionError as e:
        guard8c_raised = True
        assert "UNTESTED AT SCOPE" in str(e), f"wrong assert fired: {e}"
    assert guard8c_raised, "num_heads=2 + geo3_active was NOT rejected"
    guard8c_ok = DeltaNetLMMixer(d_model=64, d_state=128, num_heads=2, geo3_active=False)
    assert guard8c_ok.geo3_active is False       # sensitivity: same config WITHOUT geo3 constructs fine
    print("  num_heads=2 + geo3_active=True correctly REJECTED; same config with geo3_active=False "
          "constructs fine (the guard is geo3-scoped, not a blanket num_heads restriction)")

    print("\n[9] NAIVE-WINDOW selection mode (sec 4.5's ablation arm): must be BETA-BLIND -- "
          "selection is IDENTICAL regardless of beta values (positive control) -- and DIFFERENT "
          "from beta_topk when beta strongly favors the chunk's opposite end (sensitivity control "
          "proving the two modes are not accidentally identical)")
    torch.manual_seed(9)
    B9, chunk9, H9, d9, k_sel9 = 2, 8, 1, 8, 4
    k_raw9 = torch.randn(B9, chunk9, H9, d9, device=device)
    content_mask9 = torch.ones(B9, chunk9, dtype=torch.bool, device=device)
    beta9a = torch.rand(B9, chunk9, H9, device=device)
    beta9b = torch.rand(B9, chunk9, H9, device=device)          # DIFFERENT beta draw
    _, diag9a = _geo3_lm_select_and_orthogonalize(k_raw9, beta9a, content_mask9, chunk9, k_sel9, 12,
                                                    1e-2, "naive_window")
    _, diag9b = _geo3_lm_select_and_orthogonalize(k_raw9, beta9b, content_mask9, chunk9, k_sel9, 12,
                                                    1e-2, "naive_window")
    assert torch.equal(diag9a["_topk_idx"], diag9b["_topk_idx"]), \
        "naive_window selection changed when ONLY beta changed -- it must be beta-BLIND by construction"
    print("  positive control: naive_window's selected indices are IDENTICAL across two different "
          "beta draws (beta-blind, as required)")

    beta9_last = torch.zeros(B9, chunk9, H9, device=device)
    beta9_last[:, -k_sel9:, :] = 10.0                            # last k_sel9 positions strongly favored
    _, diag9c = _geo3_lm_select_and_orthogonalize(k_raw9, beta9_last, content_mask9, chunk9, k_sel9, 12,
                                                    1e-2, "beta_topk")
    naive_idx_set = set(diag9a["_topk_idx"][0, 0, :, 0].tolist())
    betatopk_idx_set = set(diag9c["_topk_idx"][0, 0, :, 0].tolist())
    assert naive_idx_set != betatopk_idx_set, (
        "sensitivity control FAILED: naive_window and beta_topk selected the SAME positions even "
        "though beta strongly favors the opposite end of the chunk -- item [9]'s positive control "
        "would be vacuous")
    print(f"  sensitivity control: naive_window selects {sorted(naive_idx_set)}, beta_topk (beta "
          f"favoring the LAST {k_sel9} positions) selects {sorted(betatopk_idx_set)} -- DIFFERENT "
          f"sets, as required")

    print("\n[10] geo3-ACTIVE forward/backward/grad-finite at REAL LM shapes (T=128=2x64 chunks, "
          "d_state=64/head_dim=64), BOTH selection modes x BOTH k_sel values -- plus an on/off "
          "sensitivity control at IDENTICAL weights")
    torch.manual_seed(10)
    V10 = 500
    for selection_mode10 in GEO3_LM_SELECTION_MODES:
        for k_sel10 in (16, 32):
            model10 = DeltaNetLM(V10, d_model=64, d_state=64, n_layers=2, conv_size=4,
                                  geo3_active=True, geo3_k_sel=k_sel10, geo3_chunk_size=64,
                                  geo3_n_iter=12, geo3_resid_tol=1e-2,
                                  geo3_selection_mode=selection_mode10).to(device)
            x10 = torch.randint(0, V10, (3, 128), device=device)
            y10 = torch.randint(0, V10, (3, 128), device=device)
            logits10 = model10(x10)
            assert logits10.shape == (3, 128, V10) and torch.isfinite(logits10).all()
            loss10 = F.cross_entropy(logits10.reshape(-1, V10), y10.reshape(-1))
            loss10.backward()
            n_none10 = [n for n, p in model10.named_parameters() if p.grad is None]
            assert not n_none10, f"no grad for: {n_none10}"
            for n, p in model10.named_parameters():
                assert torch.isfinite(p.grad).all(), \
                    f"non-finite grad at {n} (selection_mode={selection_mode10}, k_sel={k_sel10})"
            assert model10.blocks[0].mixer.geo3_last_diag is not None
            assert model10.blocks[1].mixer.geo3_last_diag is not None
            print(f"  selection_mode={selection_mode10} k_sel={k_sel10}: forward {tuple(logits10.shape)}, "
                  f"loss {loss10.item():.4f}, ALL grads finite, "
                  f"resid_mean(L0)={model10.blocks[0].mixer.geo3_last_diag['resid_mean']:.4f}")

    torch.manual_seed(11)
    V10b = 300
    model_off = DeltaNetLM(V10b, d_model=64, d_state=64, n_layers=1, conv_size=4, geo3_active=False).to(device)
    model_on = DeltaNetLM(V10b, d_model=64, d_state=64, n_layers=1, conv_size=4, geo3_active=True,
                           geo3_k_sel=16, geo3_chunk_size=64).to(device)
    model_on.load_state_dict(model_off.state_dict())            # IDENTICAL weights
    x10b = torch.randint(0, V10b, (2, 128), device=device)
    with torch.no_grad():
        logits_off = model_off(x10b)
        logits_on = model_on(x10b)
    assert not torch.equal(logits_off, logits_on), (
        "sensitivity control FAILED: geo3_active=True produced IDENTICAL output to geo3_active=False "
        "at identical weights/input -- the geo3 construction is not doing anything")
    print(f"  sensitivity control: identical weights, geo3_active=True vs False -> DIFFERENT logits "
          f"(mean abs diff {(logits_off - logits_on).abs().mean().item():.4f}) -- geo3 construction has teeth")

    print("\n[11] SCALE_TRANSFER_DESIGN.md sec 5.2's REQUIRED harness-change smoke test: "
          "n_layers widened past the old {1,2} ceiling -- forward/backward/grad-finite at TRACK C "
          "RUNG 1's real shapes (d_model=768, d_state=64, n_layers=12), plus rung 1's own "
          "measured-param-count-vs-target check (sec 5.3/5.6: within 15% or the config needs "
          "adjusting BEFORE any GPU time is spent on a real rung-1 run)")
    torch.manual_seed(12)
    # Rung-1 dims duplicated from lm_rd_rung_configs.RUNGS[1] (this codebase's pod-safe
    # no-cross-script-import convention, restated in run_lm_rd_geo3_sweep.py's own docstring) --
    # keep in sync deliberately if sec 5.3's table ever changes.
    RUNG1_D_MODEL, RUNG1_N_LAYERS, RUNG1_D_STATE = 768, 12, 64
    RUNG1_APPROX_PARAMS, RUNG1_TOLERANCE = 98_000_000, 0.15
    V11 = 500   # tiny synthetic vocab for the forward/backward/grad check -- real vocab is a
                # SEPARATE, forward-pass-free param-count-only check right below (mirrors item [1b])
    model11 = DeltaNetLM(V11, d_model=RUNG1_D_MODEL, d_state=RUNG1_D_STATE, n_layers=RUNG1_N_LAYERS,
                          conv_size=4).to(device)
    x11 = torch.randint(0, V11, (2, _MIN_KERNEL_T), device=device)
    y11 = torch.randint(0, V11, (2, _MIN_KERNEL_T), device=device)
    logits11 = model11(x11)
    assert logits11.shape == (2, _MIN_KERNEL_T, V11)
    assert torch.isfinite(logits11).all(), "rung-1 shapes: non-finite logits"
    loss11 = F.cross_entropy(logits11.reshape(-1, V11), y11.reshape(-1))
    loss11.backward()
    n_none11 = [n for n, p in model11.named_parameters() if p.grad is None]
    assert not n_none11, f"rung-1 shapes: no grad for: {n_none11}"
    for n, p in model11.named_parameters():
        assert torch.isfinite(p.grad).all(), f"rung-1 shapes: non-finite grad at {n}"
    print(f"  rung 1 shapes (d_model={RUNG1_D_MODEL}, d_state={RUNG1_D_STATE}, "
          f"n_layers={RUNG1_N_LAYERS}): forward {tuple(logits11.shape)}, loss {loss11.item():.4f}, "
          f"ALL grads finite")
    del model11, x11, y11, logits11, loss11

    with torch.no_grad():
        model11b = DeltaNetLM(50257, d_model=RUNG1_D_MODEL, d_state=RUNG1_D_STATE,
                               n_layers=RUNG1_N_LAYERS, conv_size=4)
        n_params11b = sum(p.numel() for p in model11b.parameters())
    rel_err11b = abs(n_params11b - RUNG1_APPROX_PARAMS) / RUNG1_APPROX_PARAMS
    assert rel_err11b <= RUNG1_TOLERANCE, (
        f"rung 1: measured {n_params11b:,} params is {rel_err11b*100:.1f}% off target "
        f"{RUNG1_APPROX_PARAMS:,} (tolerance {RUNG1_TOLERANCE*100:.0f}%) -- sec 5.3's config table "
        f"needs adjustment before any GPU time is spent on rung 1.")
    print(f"  rung 1 real-vocab param count: {n_params11b:,} (target {RUNG1_APPROX_PARAMS:,}, "
          f"{rel_err11b*100:.1f}% off, within {RUNG1_TOLERANCE*100:.0f}% tolerance)")
    del model11b

    print("\n[12] SCALE_TRANSFER_DESIGN.md sec 5.3's TRACK C RUNG 2 real shapes (d_model=1536, "
          "d_state=128, n_layers=16) -- forward/backward/grad-finite + measured-param-count-vs-"
          "target check, SAME discipline as item [11]'s rung-1 check, run BEFORE any GPU time is "
          "spent on a real rung-2 run (trackC-rung23-build)")
    torch.manual_seed(13)
    # Rung-2 dims duplicated from lm_rd_rung_configs.RUNGS[2] (this codebase's pod-safe
    # no-cross-script-import convention) -- keep in sync deliberately if sec 5.3's table ever changes.
    RUNG2_D_MODEL, RUNG2_N_LAYERS, RUNG2_D_STATE = 1536, 16, 128
    RUNG2_APPROX_PARAMS, RUNG2_TOLERANCE = 392_000_000, 0.15
    V12 = 500   # tiny synthetic vocab for the forward/backward/grad check -- real vocab is a
                # SEPARATE, forward-pass-free param-count-only check right below (mirrors item [11])
    model12 = DeltaNetLM(V12, d_model=RUNG2_D_MODEL, d_state=RUNG2_D_STATE, n_layers=RUNG2_N_LAYERS,
                          conv_size=4).to(device)
    x12 = torch.randint(0, V12, (2, _MIN_KERNEL_T), device=device)
    y12 = torch.randint(0, V12, (2, _MIN_KERNEL_T), device=device)
    logits12 = model12(x12)
    assert logits12.shape == (2, _MIN_KERNEL_T, V12)
    assert torch.isfinite(logits12).all(), "rung-2 shapes: non-finite logits"
    loss12 = F.cross_entropy(logits12.reshape(-1, V12), y12.reshape(-1))
    loss12.backward()
    n_none12 = [n for n, p in model12.named_parameters() if p.grad is None]
    assert not n_none12, f"rung-2 shapes: no grad for: {n_none12}"
    for n, p in model12.named_parameters():
        assert torch.isfinite(p.grad).all(), f"rung-2 shapes: non-finite grad at {n}"
    print(f"  rung 2 shapes (d_model={RUNG2_D_MODEL}, d_state={RUNG2_D_STATE}, "
          f"n_layers={RUNG2_N_LAYERS}): forward {tuple(logits12.shape)}, loss {loss12.item():.4f}, "
          f"ALL grads finite")
    del model12, x12, y12, logits12, loss12

    with torch.no_grad():
        model12b = DeltaNetLM(50257, d_model=RUNG2_D_MODEL, d_state=RUNG2_D_STATE,
                               n_layers=RUNG2_N_LAYERS, conv_size=4)
        n_params12b = sum(p.numel() for p in model12b.parameters())
    rel_err12b = abs(n_params12b - RUNG2_APPROX_PARAMS) / RUNG2_APPROX_PARAMS
    assert rel_err12b <= RUNG2_TOLERANCE, (
        f"rung 2: measured {n_params12b:,} params is {rel_err12b*100:.1f}% off target "
        f"{RUNG2_APPROX_PARAMS:,} (tolerance {RUNG2_TOLERANCE*100:.0f}%) -- sec 5.3's config table "
        f"needs adjustment before any GPU time is spent on rung 2.")
    print(f"  rung 2 real-vocab param count: {n_params12b:,} (target {RUNG2_APPROX_PARAMS:,}, "
          f"{rel_err12b*100:.1f}% off, within {RUNG2_TOLERANCE*100:.0f}% tolerance)")
    del model12b

    print("\n[13] SCALE_TRANSFER_DESIGN.md sec 5.3's TRACK C RUNG 3 real shapes (d_model=2560, "
          "d_state=128, n_layers=22) -- forward/backward/grad-finite + measured-param-count-vs-"
          "target check, SAME discipline as item [11]'s rung-1 check, run BEFORE any GPU time is "
          "spent on a real rung-3 run (trackC-rung23-build)")
    torch.manual_seed(14)
    # Rung-3 dims duplicated from lm_rd_rung_configs.RUNGS[3] (pod-safe convention, see item [12]).
    RUNG3_D_MODEL, RUNG3_N_LAYERS, RUNG3_D_STATE = 2560, 22, 128
    RUNG3_APPROX_PARAMS, RUNG3_TOLERANCE = 1_310_000_000, 0.15
    V13 = 500
    model13 = DeltaNetLM(V13, d_model=RUNG3_D_MODEL, d_state=RUNG3_D_STATE, n_layers=RUNG3_N_LAYERS,
                          conv_size=4).to(device)
    x13 = torch.randint(0, V13, (2, _MIN_KERNEL_T), device=device)
    y13 = torch.randint(0, V13, (2, _MIN_KERNEL_T), device=device)
    logits13 = model13(x13)
    assert logits13.shape == (2, _MIN_KERNEL_T, V13)
    assert torch.isfinite(logits13).all(), "rung-3 shapes: non-finite logits"
    loss13 = F.cross_entropy(logits13.reshape(-1, V13), y13.reshape(-1))
    loss13.backward()
    n_none13 = [n for n, p in model13.named_parameters() if p.grad is None]
    assert not n_none13, f"rung-3 shapes: no grad for: {n_none13}"
    for n, p in model13.named_parameters():
        assert torch.isfinite(p.grad).all(), f"rung-3 shapes: non-finite grad at {n}"
    print(f"  rung 3 shapes (d_model={RUNG3_D_MODEL}, d_state={RUNG3_D_STATE}, "
          f"n_layers={RUNG3_N_LAYERS}): forward {tuple(logits13.shape)}, loss {loss13.item():.4f}, "
          f"ALL grads finite")
    del model13, x13, y13, logits13, loss13

    with torch.no_grad():
        model13b = DeltaNetLM(50257, d_model=RUNG3_D_MODEL, d_state=RUNG3_D_STATE,
                               n_layers=RUNG3_N_LAYERS, conv_size=4)
        n_params13b = sum(p.numel() for p in model13b.parameters())
    rel_err13b = abs(n_params13b - RUNG3_APPROX_PARAMS) / RUNG3_APPROX_PARAMS
    assert rel_err13b <= RUNG3_TOLERANCE, (
        f"rung 3: measured {n_params13b:,} params is {rel_err13b*100:.1f}% off target "
        f"{RUNG3_APPROX_PARAMS:,} (tolerance {RUNG3_TOLERANCE*100:.0f}%) -- sec 5.3's config table "
        f"needs adjustment before any GPU time is spent on rung 3.")
    print(f"  rung 3 real-vocab param count: {n_params13b:,} (target {RUNG3_APPROX_PARAMS:,}, "
          f"{rel_err13b*100:.1f}% off, within {RUNG3_TOLERANCE*100:.0f}% tolerance)")
    del model13b

    print("\n[14] TRACKB_REDESIGN.md Rev 3 hard-selectivity: forward/backward/grad-finite for EVERY "
          "HARD_SELECT_MECHANISM, non-geo3 (mirrors item [1]'s shape/grad-finite discipline, one "
          "mechanism at a time; NOTE this smoke item requires CUDA like the rest of this gate -- "
          "the mechanism logic itself is ALSO covered CPU-side by test_trackb_smokes.py, which "
          "exercises hard_selectivity_rd.py's functions directly without the kernel)")
    torch.manual_seed(21)
    V14 = 500
    for mech in HARD_SELECT_MECHANISMS:
        kwargs = dict(hard_select_active=True, hard_select_mechanism=mech, hard_select_k_sel=16,
                      hard_select_chunk_size=64)
        if mech == "soft_topk_comparator":
            kwargs["hard_select_tau_total_steps"] = 100
        model14 = DeltaNetLM(V14, d_model=64, d_state=64, n_layers=1, conv_size=4, **kwargs).to(device)
        x14 = torch.randint(0, V14, (2, 128), device=device)
        y14 = torch.randint(0, V14, (2, 128), device=device)
        step14 = 50 if mech in ("soft_topk_comparator", "random_topk") else None
        logits14 = model14(x14, step=step14)
        assert logits14.shape == (2, 128, V14)
        assert torch.isfinite(logits14).all(), f"mechanism={mech}: non-finite logits"
        loss14 = F.cross_entropy(logits14.reshape(-1, V14), y14.reshape(-1))
        loss14.backward()
        for n, p in model14.named_parameters():
            assert torch.isfinite(p.grad).all(), f"mechanism={mech}: non-finite grad at {n}"
        print(f"  mechanism={mech}: forward {tuple(logits14.shape)}, loss {loss14.item():.4f}, ALL grads finite")
        del model14, x14, y14, logits14, loss14

    print("\n[15] TRACKB_REDESIGN.md Rev 2 M6: Cell 4 composition (hard_select_active=True + "
          "geo3_active=True, hard_select_k_sel==geo3_k_sel) forward/backward/grad-finite, PLUS the "
          "entmax+geo3 combination is HARD-REFUSED at construction (untested-at-scope, mirrors item "
          "[8c]'s num_heads>1 guard)")
    torch.manual_seed(22)
    V15 = 500
    model15 = DeltaNetLM(V15, d_model=64, d_state=64, n_layers=1, conv_size=4,
                          geo3_active=True, geo3_k_sel=16, geo3_n_iter=12,
                          hard_select_active=True, hard_select_mechanism="hard_ste",
                          hard_select_k_sel=16, hard_select_chunk_size=64).to(device)
    x15 = torch.randint(0, V15, (2, 128), device=device)
    y15 = torch.randint(0, V15, (2, 128), device=device)
    logits15 = model15(x15)
    assert logits15.shape == (2, 128, V15)
    assert torch.isfinite(logits15).all(), "Cell 4 composition: non-finite logits"
    loss15 = F.cross_entropy(logits15.reshape(-1, V15), y15.reshape(-1))
    loss15.backward()
    for n, p in model15.named_parameters():
        assert torch.isfinite(p.grad).all(), f"Cell 4 composition: non-finite grad at {n}"
    diag15 = model15.blocks[0].mixer.geo3_last_diag
    assert diag15["selection_mode"] == "forced", (
        f"Cell 4 composition: geo3_last_diag['selection_mode']={diag15['selection_mode']!r}, "
        f"expected 'forced' (M6: geo3's own selection must be REPLACED by the mask's)")
    print(f"  Cell 4 composition: forward {tuple(logits15.shape)}, loss {loss15.item():.4f}, "
          f"ALL grads finite, geo3 selection_mode={diag15['selection_mode']!r} (forced, as required)")
    del model15, x15, y15, logits15, loss15

    guard15_raised = False
    try:
        DeltaNetLM(V15, d_model=64, d_state=64, n_layers=1, conv_size=4,
                   geo3_active=True, geo3_k_sel=16,
                   hard_select_active=True, hard_select_mechanism="entmax",
                   hard_select_k_sel=16, hard_select_chunk_size=64)
    except AssertionError:
        guard15_raised = True
    assert guard15_raised, "entmax+geo3_active guard FAILED to reject the untested-at-scope combination"
    print("  entmax + geo3_active=True correctly REFUSED at construction")

    print("\n[16] TRACKB re-probe path (auditor-prescribed, 2026-07-04): wave_neg1_trackb's "
          "capture hooks on a REAL forward pass -- captured beta must equal a DIRECT "
          "sigmoid(b_proj(norm1(embed(x)))) recomputation at layer 0, and "
          "cell1_readout_from_captured must return finite, sane per-layer statistics")
    import wave_neg1_trackb as _wn1   # lazy: wave_neg1_trackb imports THIS module at its top --
                                       # a module-scope import back would be a real cycle
    torch.manual_seed(23)
    V16 = 500
    model16 = DeltaNetLM(V16, d_model=64, d_state=64, n_layers=2, conv_size=4).to(device)
    x16 = torch.randint(0, V16, (2, 128), device=device)
    handles16, captured16 = _wn1.register_kbeta_capture_hooks(model16)
    try:
        with torch.no_grad():
            _ = model16(x16)
            # direct recomputation at layer 0 (deeper layers need the residual stream -- layer 0's
            # pre-mixer input is exactly norm1(embed(x)), recomputable independently)
            a16 = model16.blocks[0].norm1(model16.embed(x16))
            beta_direct = torch.sigmoid(model16.blocks[0].mixer.b_proj(a16))
        assert set(captured16.keys()) == {0, 1}, f"expected captures for layers 0/1, got {sorted(captured16)}"
        for li in (0, 1):
            assert "k_raw" in captured16[li] and "beta" in captured16[li], f"layer {li} capture incomplete"
        assert torch.equal(captured16[0]["beta"], beta_direct), (
            "hook-captured beta diverges from the direct sigmoid(b_proj(norm1(embed(x)))) "
            "recomputation at layer 0 -- the capture path is NOT reading the real forward's beta")
        readout16 = _wn1.cell1_readout_from_captured(
            captured16, x16, chunk_size=64, k_sel=16, n_iter=12, resid_tol=1e-2,
            excluded_token_ids=(EOT_TOKEN_ID,))
        assert set(readout16.keys()) == {0, 1}
        for li, r in readout16.items():
            assert 0.0 <= r["resid_mean"] < 16.0, f"layer {li}: resid_mean {r['resid_mean']} out of range"
            assert torch.isfinite(r["per_chunk_total_mass"]).all()
            assert (r["support_size"] >= 0).all()
        print(f"  captured beta == direct recomputation (bit-identical at layer 0); "
              f"cell1 readout: resid_mean L0={readout16[0]['resid_mean']:.4f} "
              f"L1={readout16[1]['resid_mean']:.4f}, all stats finite")
    finally:
        for h in handles16:
            h.remove()
    del model16, x16, captured16

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
    ap.add_argument("--n-layers", type=int, default=2,
                     help=f"1..{_MAX_N_LAYERS} (SCALE_TRANSFER_DESIGN.md sec 5.2: widened from the "
                          f"original probe-tier {{1,2}} ceiling for Track C's scaling ladder, sec "
                          f"5.3). Range validated at parse time below, not via argparse choices= "
                          f"(the ladder's rungs are not a small enumerable set).")
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
    ap.add_argument("--use-geo3-lm", action="store_true",
                     help="SCALE_TRANSFER_DESIGN.md sec 4 (Track B): enable the geo3-in-LM "
                          "beta-gated (or naive-window) per-chunk key orthogonalization. Default "
                          "OFF -- the default path is byte-identical to the pre-Track-B code "
                          "(smoke item [6]).")
    ap.add_argument("--geo3-k-sel", type=int, default=16, choices=[16, 32],
                     help="sec 4.2: positions orthogonalized per (chunk,head). The two values "
                          "with F-geo-3 synthetic reference results (sec 1.1).")
    ap.add_argument("--geo3-chunk-size", type=int, default=GEO3_LM_CHUNK_SIZE_DEFAULT,
                     help="sec 4.2 item 1: kernel-aligned tiling window (chunk_delta_rule's own "
                          "chunk_size, NOT d_state -- MAJOR-3's correction).")
    ap.add_argument("--geo3-n-iter", type=int, default=12,
                     help="Newton-Schulz iteration count (model_rd.py's own default; sec 1.1's "
                          "K=32 escalation used n_iter=20 -- pass --geo3-n-iter 20 to replicate).")
    ap.add_argument("--geo3-resid-tol", type=float, default=1e-2)
    ap.add_argument("--geo3-selection", choices=list(GEO3_LM_SELECTION_MODES), default="beta_topk",
                     help="sec 4.2/4.5: 'beta_topk' is the primary construction; 'naive_window' is "
                          "the cheap comparison arm (Wave 2, sec 4.5) -- a beta-BLIND positional "
                          "selection, same K_sel grid.")
    ap.add_argument("--hard-select-active", action="store_true",
                     help="TRACKB_REDESIGN.md Rev 3: enable a hard-selectivity beta mechanism "
                          "(replaces the plain sigmoid). Default OFF -- byte-identical to the "
                          "pre-hard-selectivity code (smoke items [6]/[14]).")
    ap.add_argument("--hard-select-mechanism", choices=list(HARD_SELECT_MECHANISMS), default="hard_ste",
                     help="'hard_ste'=candidate 1 (PRIMARY), 'entmax'=candidate 2 (sparsemax only), "
                          "'soft_topk_comparator'=M7 comparator, 'random_topk'=Cell 2R/4R control.")
    ap.add_argument("--hard-select-k-sel", type=int, default=32,
                     help="sec 5.3: K_sel is PINNED at 32 for the Wave 1/2x2 manifest (K=16 is the "
                          "first registered follow-on axis, not this default).")
    ap.add_argument("--hard-select-chunk-size", type=int, default=GEO3_LM_CHUNK_SIZE_DEFAULT)
    ap.add_argument("--hard-select-b-pinned", type=float, default=None,
                     help="sec 2 principle 4: the Wave -1-pinned per-chunk total write mass "
                          "(BANDS_PINNED-TrackB.json's own b_pinned field). Omit to skip renorm "
                          "(Wave -1 probes only, BEFORE b_pinned is pinned) -- a real Wave 1 cell "
                          "MUST pass this.")
    ap.add_argument("--hard-select-seed", type=int, default=0,
                     help="Cell 2R/4R random control's RNG seed (combined with the training step "
                          "via hard_selectivity_rd.derive_step_rng) -- independent of --seed so the "
                          "control's own randomness axis is explicit, not silently tied to the "
                          "training-data seed.")
    ap.add_argument("--hard-select-tau-anneal-frac", type=float, default=0.10,
                     help="M7 comparator: tau anneals 1->0 over this fraction of --steps, Rev 3 "
                          "NEW-6's registered 10%% pin.")
    ap.add_argument("--gate-override-reason", type=str, default=None,
                     help="TRACKB_REDESIGN.md sec 5.1 (Rev 2 M5): non-empty ONLY for the Cell-3 "
                          "override-stamped reference-arm manifest, threaded by "
                          "run_trackb_wave.py's own --accept-no-launch-reference-arm flag. Stamps "
                          "gate_override=True/gate_override_reason/gate_override_at/"
                          "claim_tier='descriptive' into this run's result JSON AT ASSEMBLY TIME "
                          "(_assemble_result). Omit for every non-override run -- gate_override "
                          "then writes False, never absent.")
    ap.add_argument("--trackb-selection-probe", type=int, default=None, metavar="K_SEL",
                     help="TRACKB_REDESIGN.md sec 4.3 (F2 audit fix): enable the READ-ONLY "
                          "implicit top-K-by-beta selection probe on an otherwise-unmasked run "
                          "(the reference pilot -- churn Null A / TV-ceiling / support-floor data "
                          "source). Adds per-checkpoint hard_select_diagnostics without touching "
                          "beta or the forward computation. Mutually exclusive with "
                          "--hard-select-active (an active mechanism IS its own diagnostics "
                          "source).")
    ap.add_argument("--nan-probe-counter", action="store_true",
                     help="TRACKB_REDESIGN.md sec 5.1 (Rev 3 NEW-7): count exact duplicates among "
                          "geo3's SELECTED raw key rows per training forward call (the stability "
                          "smoke's >=25-calls/>=6-dup-rows positive-control floor). geo3-active "
                          "runs only; writes nan_probe_positive_control into the result JSON.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.smoke:
        smoke(device)
        return

    assert device == "cuda", "lm_pretrain_rd requires CUDA for real training (chunk_delta_rule has no CPU path)"
    assert args.corpus is not None, "--corpus is required for a real (non-smoke) run"
    assert 1 <= args.n_layers <= _MAX_N_LAYERS, (
        f"--n-layers={args.n_layers} outside the registered range [1,{_MAX_N_LAYERS}] "
        f"(SCALE_TRANSFER_DESIGN.md sec 5.3's ladder tops out at 22) -- validated at CLI-parse "
        f"time, matching DeltaNetLM.__init__'s own assert (never rely on that one alone).")
    if args.ckpt_every > 2000:
        print(f"WARNING: --ckpt-every={args.ckpt_every} > 2000: violates section 8's build requirement.", flush=True)
    if args.ckpt_dir is None:
        print("WARNING: --ckpt-dir not given -- no model checkpoints will be saved; "
              "lm_intervene_rd.py will have nothing to load.", flush=True)
    assert args.seq_len >= _MIN_KERNEL_T, \
        f"--seq-len={args.seq_len} < _MIN_KERNEL_T={_MIN_KERNEL_T} (F15-LM measured floor)"
    if args.use_geo3_lm:
        assert args.seq_len % args.geo3_chunk_size == 0, (
            f"--use-geo3-lm requires --seq-len ({args.seq_len}) to be a multiple of "
            f"--geo3-chunk-size ({args.geo3_chunk_size}) -- sec 4.2 item 1's exact chunk tiling "
            f"(validated at CLI-parse time, not left to fail deep inside the first forward pass)."
        )
        head_dim = args.d_state // args.num_heads
        assert args.geo3_k_sel <= min(args.geo3_chunk_size, head_dim), (
            f"--geo3-k-sel={args.geo3_k_sel} must be <= min(geo3_chunk_size={args.geo3_chunk_size}, "
            f"head_dim={head_dim}={args.d_state}/{args.num_heads})."
        )
    if args.trackb_selection_probe is not None:
        assert not args.hard_select_active, (
            "--trackb-selection-probe is for UNMASKED runs only (the reference pilot); a "
            "hard_select_active run's own mechanism already produces hard_select_diagnostics -- "
            "passing both would silently overwrite the mechanism's diag with the probe's."
        )
        assert 1 <= args.trackb_selection_probe <= args.hard_select_chunk_size, (
            f"--trackb-selection-probe={args.trackb_selection_probe} must be in "
            f"[1,{args.hard_select_chunk_size}]"
        )
        assert args.seq_len % args.hard_select_chunk_size == 0, (
            f"--trackb-selection-probe requires --seq-len ({args.seq_len}) to be a multiple of "
            f"--hard-select-chunk-size ({args.hard_select_chunk_size}) -- the probe tiles chunks "
            f"exactly like a real mechanism would."
        )
    if args.hard_select_active:
        assert args.seq_len % args.hard_select_chunk_size == 0, (
            f"--hard-select-active requires --seq-len ({args.seq_len}) to be a multiple of "
            f"--hard-select-chunk-size ({args.hard_select_chunk_size})."
        )
        assert args.hard_select_k_sel <= args.hard_select_chunk_size, (
            f"--hard-select-k-sel={args.hard_select_k_sel} must be <= "
            f"--hard-select-chunk-size={args.hard_select_chunk_size}."
        )
        if args.use_geo3_lm:
            assert args.hard_select_mechanism != "entmax", (
                "--hard-select-mechanism=entmax + --use-geo3-lm is NOT SUPPORTED (M6's forced-"
                "selection composition has no fixed-shape set to force under variable sparsemax "
                "support) -- see DeltaNetLMMixer.__init__'s own assert for the full reasoning."
            )
            assert args.hard_select_k_sel == args.geo3_k_sel, (
                f"Cell 4 composition rule (M6): --hard-select-k-sel={args.hard_select_k_sel} must "
                f"equal --geo3-k-sel={args.geo3_k_sel}."
            )
            assert args.hard_select_chunk_size == args.geo3_chunk_size, (
                f"--hard-select-chunk-size={args.hard_select_chunk_size} must equal "
                f"--geo3-chunk-size={args.geo3_chunk_size} when composing with geo3 (M6)."
            )

    other_corpus = OTHER_CORPUS[args.corpus]
    train_tokens, val_same, meta_same, train_offs, val_offs_same = load_corpus(args.data_dir, args.corpus, device)
    _, val_other, meta_other, _, val_offs_other = load_corpus(args.data_dir, other_corpus, device)
    print(f"corpus={args.corpus} (train {train_tokens.numel():,} tok / {train_offs.numel():,} docs, "
          f"val {val_same.numel():,} tok / {val_offs_same.numel():,} docs)  "
          f"other_corpus={other_corpus} (val {val_other.numel():,} tok / {val_offs_other.numel():,} docs)",
          flush=True)

    hard_select_tau_total_steps = args.steps if args.hard_select_mechanism == "soft_topk_comparator" else None
    model = DeltaNetLM(meta_same["vocab_size"], d_model=args.d_model, d_state=args.d_state,
                        n_layers=args.n_layers, conv_size=args.conv_size, num_heads=args.num_heads,
                        ffn_mult=args.ffn_mult, geo3_active=args.use_geo3_lm,
                        geo3_k_sel=args.geo3_k_sel, geo3_chunk_size=args.geo3_chunk_size,
                        geo3_n_iter=args.geo3_n_iter, geo3_resid_tol=args.geo3_resid_tol,
                        geo3_selection_mode=args.geo3_selection,
                        hard_select_active=args.hard_select_active,
                        hard_select_mechanism=args.hard_select_mechanism,
                        hard_select_k_sel=args.hard_select_k_sel,
                        hard_select_chunk_size=args.hard_select_chunk_size,
                        hard_select_b_pinned=args.hard_select_b_pinned,
                        hard_select_seed=args.hard_select_seed,
                        hard_select_tau_total_steps=hard_select_tau_total_steps,
                        hard_select_tau_anneal_frac=args.hard_select_tau_anneal_frac).to(device)
    if args.trackb_selection_probe is not None:
        # F2 audit fix: the read-only implicit-selection probe, set post-construction on every
        # mixer (see DeltaNetLMMixer's selection_probe_k_sel comment).
        for blk in model.blocks:
            blk.mixer.selection_probe_k_sel = args.trackb_selection_probe
    n_params = sum(p.numel() for p in model.parameters())
    geo3_tag = f"_geo3{args.geo3_selection.replace('_', '')[:8]}k{args.geo3_k_sel}" if args.use_geo3_lm else ""
    hs_tag = f"_hs{args.hard_select_mechanism.replace('_', '')[:8]}k{args.hard_select_k_sel}" \
        if args.hard_select_active else ""
    run_name = (f"lmC_{args.corpus}_dm{args.d_model}_ds{args.d_state}_L{args.n_layers}_s{args.seed}"
                f"{geo3_tag}{hs_tag}")
    print(f"run_name={run_name}  d_model={args.d_model} d_state={args.d_state} n_layers={args.n_layers} "
          f"seq_len={args.seq_len} batch_size={args.batch_size} steps={args.steps} params={n_params} "
          f"use_geo3_lm={args.use_geo3_lm} geo3_k_sel={args.geo3_k_sel if args.use_geo3_lm else '-'} "
          f"geo3_selection={args.geo3_selection if args.use_geo3_lm else '-'} "
          f"hard_select_active={args.hard_select_active} "
          f"hard_select_mechanism={args.hard_select_mechanism if args.hard_select_active else '-'} "
          f"hard_select_k_sel={args.hard_select_k_sel if args.hard_select_active else '-'} "
          f"hard_select_b_pinned={args.hard_select_b_pinned} "
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
