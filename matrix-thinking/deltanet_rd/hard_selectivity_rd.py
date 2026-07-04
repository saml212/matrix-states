"""hard_selectivity_rd.py -- TRACKB_REDESIGN.md Rev 3 (commit 8ab089d): the
hard-selectivity mechanisms this redesign registers to force the Wave -1
gate's missing precondition (SCALE_TRANSFER_DESIGN.md sec 4's geo3-in-LM,
Track B). This module holds every piece of mechanism that is CPU-testable
in isolation (no chunk_delta_rule call anywhere in this file -- the Triton
kernel has no CPU path, lm_pretrain_rd.py's own smoke() docstring/assert);
the GPU-only wiring (threading these into DeltaNetLMMixer.forward) lives in
lm_pretrain_rd.py itself, as small additive edits next to the existing
geo3-in-LM machinery this file extends (Rev 3's own build brief: "extending
the existing lm_pretrain_rd.py geo3-LM machinery ... following its idiom").

Contents, one section per TRACKB_REDESIGN.md reference:
  - derive_step_rng                    sec 5.1 (Rev 3 NEW-2/NEW-3) + this
                                        build's incorporated note (1): the
                                        per-(chunk,step) RNG stream that
                                        CONTINUES across training and
                                        eval-time forward-hook probes.
  - hard_topk_beta_mask / apply_hard_select_ste   Candidate 1 (sec 3.1, PRIMARY)
  - ChunkSparsemax / chunk_sparsemax_beta          Candidate 2 (sec 3.2)
  - tau_schedule / soft_topk_comparator_beta       M7 comparator (sec 3.1, Rev 3 NEW-6)
  - random_topk_mask                               Cell 2R / 4R control (sec 5.1)
  - renormalize_to_b_pinned / classify_budget_partial   sec 2 principle 4
                                                    (Rev 3 NEW-1, symmetric)
  - churn_rate / tv_distance_from_uniform / support_size / per_chunk_total_mass
                                                    sec 4.3 BANDS_PINNED-TrackB metrics
  - mc_recompute_anchors                           sec 5.3 (K_sel=32,d=64 MC anchors)
  - candidate4_hard_snap_active                     Candidate 4 (sec 3.4, "only if cheap")
  - trackb_override_stamp_payload / trackb_assemble_gate_override_fields
                                                    sec 5.1 (Rev 2 M5) Cell-3 override stamping,
                                                    mirroring key_anchoring.py's
                                                    override_stamp_payload/assemble_claim_tier_fields
                                                    precedent exactly.

All shape conventions below match lm_pretrain_rd.py's own:
  beta / content_mask are (B,T,H) / (B,T) at the mixer's beta site
  (:529-531, upstream of the q/k/v reshape -- N2's corrected insertion),
  reshaped internally to (B,n_chunks,chunk_size,H) exactly like
  _geo3_lm_select_and_orthogonalize's own beta_c/content_c (:287-289).
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# sec 5.1 (Rev 3 NEW-2/NEW-3) + build-phase note (1): the per-(chunk,step) RNG
# stream. A PURE function of (seed, step) -- the single choke point every
# step-resampled mechanism in this file goes through, so that an eval-time
# forward-hook probe run AT a checkpoint's own recorded training step draws
# EXACTLY what a training-time forward call at that same step would have
# drawn (CONTINUING the stream, never freezing a fresh draw per checkpoint --
# the build-phase note this module is required to implement and document).
# ---------------------------------------------------------------------------

def derive_step_rng(seed: int, step: int, device: str = "cpu") -> torch.Generator:
    """Deterministic generator keyed to (seed, step) ONLY -- no other input
    (not batch index, not "is this a probe or training step") may perturb
    it, or the continuation property below breaks. Combined via a wide odd
    multiplier (avoids adjacent-step seed collisions a small multiplier
    could produce) then reduced into torch.Generator.manual_seed's accepted
    range.

    Continuation contract (build-phase note (1), verified by
    test_trackb_smokes.py's continuation-equivalence smoke): calling this
    with the SAME (seed, step) from two different call sites (a training
    loop's `step` counter vs. an eval-time forward-hook probe reading a
    checkpoint's saved `step` field) returns a generator in an IDENTICAL
    state -- the mechanism that consumes it (random_topk_mask below) then
    produces byte-identical output at that step, whether called from
    training or from a later probe."""
    combined = (int(seed) * 1_000_003 + int(step)) % (2 ** 31 - 1)
    gen = torch.Generator(device=device)
    gen.manual_seed(combined)
    return gen


# ---------------------------------------------------------------------------
# Candidate 1 (sec 3.1, PRIMARY) -- hard top-K beta masking with a
# straight-through estimator.
# ---------------------------------------------------------------------------

def hard_topk_beta_mask(beta: torch.Tensor, content_mask: torch.Tensor, chunk_size: int, k_sel: int):
    """Candidate 1's selection-SET construction: top-k_sel content positions
    by beta MAGNITUDE, per (chunk, head) -- literally
    _geo3_lm_select_and_orthogonalize's own beta_topk `priority`/`topk`
    idiom (lm_pretrain_rd.py:292-301), reused one call EARLIER: applied to
    `beta` itself (shape (B,T,H), reshape-unaffected), at the corrected
    insertion site (Rev 2 -- N2: lm_pretrain_rd.py:529-531, upstream of both
    the q/k/v reshape and the existing geo3 call at :539-551).

    beta: (B,T,H) post-sigmoid gate. content_mask: (B,T) bool, True =
    eligible (EOT/padding hard-excluded, sec 4.2 item 3). chunk_size: T must
    be an exact multiple. k_sel: <= chunk_size.

    Returns:
      mask: (B,T,H) float {0,1} -- exactly k_sel ones per (chunk,head) among
        content positions (fewer only if a chunk has < k_sel content
        positions, reflected in valid_sel below -- EXACT, not a
        tolerance-slack check, mirroring the house discipline on
        integer/structural correctness [LEARN] rule).
      topk_idx: (B,n_chunks,k_sel,H) int64, in-chunk offsets [0,chunk_size).
      valid_sel: (B,n_chunks,k_sel,H) bool, True iff that slot is a genuine
        content position.
    """
    B, T, H = beta.shape
    assert T % chunk_size == 0, f"T={T} not a multiple of chunk_size={chunk_size}"
    assert 1 <= k_sel <= chunk_size, f"k_sel={k_sel} must be in [1,{chunk_size}]"
    assert content_mask.shape == (B, T), f"content_mask shape {tuple(content_mask.shape)} != (B,T)=({B},{T})"
    n_chunks = T // chunk_size
    beta_c = beta.view(B, n_chunks, chunk_size, H)
    content_c = content_mask.view(B, n_chunks, chunk_size, 1).expand(B, n_chunks, chunk_size, H)
    neg_inf = torch.finfo(beta_c.dtype).min
    priority = torch.where(content_c, beta_c, torch.full_like(beta_c, neg_inf))
    topk_val, topk_idx = torch.topk(priority, k_sel, dim=2)          # (B,n_chunks,k_sel,H)
    valid_sel = topk_val > (neg_inf / 2)                              # exact, mirrors :301
    mask_c = torch.zeros_like(beta_c)
    mask_c.scatter_(2, topk_idx, valid_sel.to(beta_c.dtype))
    return mask_c.view(B, T, H), topk_idx, valid_sel


class _HardSelectSTE(torch.autograd.Function):
    """sec 3.1: dL/d(beta_soft) := dL/d(beta_hard) -- Bengio, Leonard &
    Courville 2013 (the trick underlying hard-concrete/L0 gating, Louizos,
    Welling & Kingma 2018). Forward masks EXACTLY (literal zero write mass
    at non-selected positions); backward ignores the mask entirely (as if
    it were all-ones), so b_proj keeps a training signal even for positions
    the current forward pass zeroed."""

    @staticmethod
    def forward(ctx, beta_soft: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return beta_soft * mask

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None                 # identity pass-through; mask gets no gradient


def apply_hard_select_ste(beta_soft: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Candidate 1's forward value + STE backward, in one call."""
    return _HardSelectSTE.apply(beta_soft, mask)


# ---------------------------------------------------------------------------
# Candidate 2 (sec 3.2) -- chunk-normalized sparsemax beta (Martins &
# Astudillo, ICML 2016): exact zeros, adaptively sparse, differentiable via
# its own closed-form analytic backward (no STE, no external dependency --
# this codebase's own "hand-roll simple closed forms" discipline, sec 3.2's
# own text).
# ---------------------------------------------------------------------------

class _Sparsemax(torch.autograd.Function):
    """Euclidean projection onto the simplex, along the LAST dim. Forward:
    sort descending, find support size k(z) via the standard
    "1 + j*z_(j) > cumsum_j" monotone-prefix test (Martins & Astudillo 2016
    Algorithm 1), tau(z) = (cumsum_{k(z)} - 1)/k(z), p_i = max(z_i-tau,0).
    Backward: the exact analytic Jacobian J = diag(support) -
    (1/|support|) support support^T (their sec 3, eq. for the Jacobian) --
    NOT the naive "gradient=1 on support, 0 off" approximation, which drops
    the simplex-sum-to-1 coupling term."""

    @staticmethod
    def forward(ctx, z: torch.Tensor) -> torch.Tensor:
        orig_shape = z.shape
        z2 = z.reshape(-1, z.shape[-1])
        n = z2.shape[-1]
        z_sorted, _ = torch.sort(z2, dim=-1, descending=True)
        rng = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
        cumsum = z_sorted.cumsum(dim=-1)
        support_bool = (1 + rng * z_sorted) > cumsum                 # (N,n), True entries form a prefix
        k = support_bool.sum(dim=-1, keepdim=True).clamp(min=1)      # (N,1) int64
        tau = (torch.gather(cumsum, -1, k - 1) - 1) / k.to(z2.dtype)
        p = torch.clamp(z2 - tau, min=0).reshape(orig_shape)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        p, = ctx.saved_tensors
        support = (p > 0).to(p.dtype)
        s = support.sum(dim=-1, keepdim=True).clamp(min=1)
        grad_support = grad_output * support
        avg = grad_support.sum(dim=-1, keepdim=True) / s
        return support * (grad_output - avg)


def sparsemax(z: torch.Tensor) -> torch.Tensor:
    """Sparsemax along the last dim (batched over every leading dim)."""
    return _Sparsemax.apply(z)


_SPARSEMAX_EXCLUDE_SENTINEL = -1e6   # sec 3.2 build note: sparsemax's cumsum-based algorithm cannot
                                     # take a literal -inf (NaN from -inf arithmetic in the cumsum/
                                     # threshold steps) the way _geo3_lm_select_and_orthogonalize's
                                     # pure-ranking topk path can (:291's neg_inf = finfo.min is safe
                                     # there ONLY because topk never adds/subtracts it). A large-but-
                                     # finite sentinel is used instead, deliberately far below any
                                     # realistic b_proj raw-score range so it never enters the
                                     # support (verified by test_trackb_smokes.py's exact-zero-at-
                                     # excluded-positions smoke) -- a documented implementation
                                     # difference from the topk path's exact-tolerance convention,
                                     # not an oversight.


def chunk_sparsemax_beta(scores: torch.Tensor, content_mask: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Candidate 2's beta: sparsemax over the chunk_size axis, per (chunk,
    head) -- replaces the independent per-position sigmoid with a
    chunk-normalized, adaptively-sparse, sum-to-<=1 distribution (sec 3.2).
    scores: (B,T,H) raw b_proj(x) output (pre-sigmoid; sparsemax is applied
    directly to these, no elementwise nonlinearity first). content_mask:
    (B,T) bool, EOT/padding excluded from the simplex entirely (excluded
    positions get beta=0 exactly, verified never to enter the support)."""
    B, T, H = scores.shape
    assert T % chunk_size == 0
    n_chunks = T // chunk_size
    s_c = scores.view(B, n_chunks, chunk_size, H).permute(0, 1, 3, 2)   # (B,n_chunks,H,chunk_size)
    content_c = content_mask.view(B, n_chunks, chunk_size, 1).expand(B, n_chunks, chunk_size, H) \
        .permute(0, 1, 3, 2)
    s_masked = torch.where(content_c, s_c, torch.full_like(s_c, _SPARSEMAX_EXCLUDE_SENTINEL))
    p = sparsemax(s_masked)                                             # (B,n_chunks,H,chunk_size)
    return p.permute(0, 1, 3, 2).reshape(B, T, H)


# ---------------------------------------------------------------------------
# M7 comparator (sec 3.1, Rev 3 NEW-6) -- temperature-annealed soft-top-K.
# Same top-K SELECTION as candidate 1; non-selected beta multiplied by a
# decaying (never masked+STE) factor tau(t). No gradient estimator anywhere:
# gradient flows through beta_soft naturally via the (1-mask)*tau term.
# ---------------------------------------------------------------------------

def tau_schedule(step: int, total_steps: int, anneal_frac: float = 0.10) -> float:
    """Rev 3 NEW-6: tau anneals 1->0 LINEARLY over the first
    anneal_frac*total_steps steps, then EXACTLY 0.0 for the remaining 90%
    (re-pinned from Rev 2's 80%-of-steps schedule, which was itself
    candidate 4's soft-warmup curriculum in disguise -- see this module's
    docstring / TRACKB_REDESIGN.md sec 3.1). `step` is 1-indexed, matching
    lm_pretrain_rd.py's train() loop convention (`for step in range(1,
    args.steps+1)`, :1169)."""
    anneal_steps = max(1, int(round(anneal_frac * total_steps)))
    if step >= anneal_steps:
        return 0.0
    return max(0.0, 1.0 - step / anneal_steps)


def soft_topk_comparator_beta(beta_soft: torch.Tensor, content_mask: torch.Tensor, chunk_size: int,
                               k_sel: int, tau: float):
    """M7 comparator's beta: same selection-set as candidate 1
    (hard_topk_beta_mask), but beta_out = beta_soft * (mask + (1-mask)*tau)
    -- selected positions keep beta_soft exactly, non-selected are decayed
    by tau (1.0 at init -> plain baseline behavior at the selected/
    non-selected split; 0.0 at/after the anneal endpoint -> numerically
    bit-equivalent to candidate 1's forward VALUE, a non-load-bearing
    observation per Rev 3, see tau_schedule's docstring). No STE: this is a
    plain differentiable multiply.

    Returns (beta_out, mask, topk_idx, valid_sel)."""
    mask, topk_idx, valid_sel = hard_topk_beta_mask(beta_soft, content_mask, chunk_size, k_sel)
    beta_out = beta_soft * (mask + (1.0 - mask) * tau)
    return beta_out, mask, topk_idx, valid_sel


def comparator_endpoint_matches_hard_mask(beta_soft: torch.Tensor, content_mask: torch.Tensor,
                                           chunk_size: int, k_sel: int, atol: float = 1e-6) -> float:
    """Registered numeric equivalence check (sec 3.1: 'max-abs forward diff
    <= 1e-6 on a fixed probe batch') at the comparator's tau=0 endpoint vs.
    candidate 1's own hard-STE forward VALUE (not the gradient/backward path
    -- only the forward numbers are claimed equivalent). Returns the
    max-abs difference; caller asserts it against atol."""
    mask, _, _ = hard_topk_beta_mask(beta_soft, content_mask, chunk_size, k_sel)
    hard_val = apply_hard_select_ste(beta_soft, mask)
    comparator_val, _, _, _ = soft_topk_comparator_beta(beta_soft, content_mask, chunk_size, k_sel, tau=0.0)
    return (hard_val - comparator_val).abs().max().item()


# ---------------------------------------------------------------------------
# Cell 2R / 4R (sec 5.1, Rev 3 NEW-2/NEW-3) -- budget-matched random control.
# Beta-BLIND top-k_sel selection, RESAMPLED INDEPENDENTLY per (chunk
# instance, training step): one random-priority tensor drawn over the WHOLE
# (B,n_chunks,chunk_size,H) shape from a single (seed,step)-keyed generator
# gives every chunk instance its own independently-random row (varies by
# chunk instance) from a draw that itself changes with step (varies by
# step) -- the cadence Rev 3 NEW-2 registers, and the one that does NOT
# degenerate into a fixed positional schedule (candidate 3's own mechanism).
# ---------------------------------------------------------------------------

def random_topk_mask(shape: tuple, content_mask: torch.Tensor, chunk_size: int, k_sel: int,
                      seed: int, step: int, device: str = "cpu"):
    """shape: (B,T,H). Returns (mask, topk_idx, valid_sel) -- same contract
    as hard_topk_beta_mask, but priority is i.i.d. uniform noise (beta-
    blind), keyed to derive_step_rng(seed, step) -- see that function's
    continuation contract, which this control depends on for its eval-probe
    behavior exactly as candidate 1 does."""
    B, T, H = shape
    assert T % chunk_size == 0
    n_chunks = T // chunk_size
    gen = derive_step_rng(seed, step, device=device)
    rand_priority = torch.rand(B, n_chunks, chunk_size, H, generator=gen, device=device)
    content_c = content_mask.view(B, n_chunks, chunk_size, 1).expand(B, n_chunks, chunk_size, H)
    neg_inf = torch.finfo(rand_priority.dtype).min
    priority = torch.where(content_c, rand_priority, torch.full_like(rand_priority, neg_inf))
    topk_val, topk_idx = torch.topk(priority, k_sel, dim=2)
    valid_sel = topk_val > (neg_inf / 2)
    mask_c = torch.zeros(B, n_chunks, chunk_size, H, device=device)
    mask_c.scatter_(2, topk_idx, valid_sel.to(mask_c.dtype))
    return mask_c.view(B, T, H), topk_idx, valid_sel


# ---------------------------------------------------------------------------
# sec 2 principle 4 (Rev 3 NEW-1, orchestrator-pinned, SYMMETRIC) -- B_pinned
# renormalization + the ONE numeric BUDGET-PARTIAL rule for every masking
# cell (candidates 1/2/3, candidate 4's hard-snap phase, the comparator,
# Cells 2R/4R).
# ---------------------------------------------------------------------------

def renormalize_to_b_pinned(beta_selected: torch.Tensor, chunk_size: int, b_pinned: float,
                             clamp_max: float = 1.0, eps: float = 1e-12):
    """beta_selected: (B,T,H), ALREADY masked/selected -- exact zero at
    every non-selected position (candidate 1/3/4's binary mask output and
    candidate 2's sparsemax support both satisfy this by construction, so
    this function takes no separate mask argument). Per (chunk,head): scale
    = b_pinned / sum(selected beta); rescale; clamp at clamp_max (the
    sigmoid range chunk_delta_rule's stock convention assumes);
    shortfall_c = max(0, b_pinned - sum(clamped beta)) / b_pinned -- NEVER
    silent (returned for every chunk, not just flagged ones).

    A chunk with zero selected mass (e.g. candidate 2's sparsemax collapsed
    to nothing, or a fully-non-content chunk) gets scale=b_pinned/eps
    (large but finite) times an exact-zero row -> exactly zero (no NaN/inf),
    and shortfall=1.0 (maximal, correctly flagging it).

    Returns (beta_out (B,T,H), shortfall (B,n_chunks,H))."""
    B, T, H = beta_selected.shape
    assert T % chunk_size == 0
    assert b_pinned > 0, f"b_pinned={b_pinned} must be > 0"
    n_chunks = T // chunk_size
    b_c = beta_selected.view(B, n_chunks, chunk_size, H)
    chunk_sum = b_c.sum(dim=2, keepdim=True)                       # (B,n_chunks,1,H)
    scale = b_pinned / chunk_sum.clamp(min=eps)
    scaled = b_c * scale
    clamped = scaled.clamp(max=clamp_max)
    shortfall = (b_pinned - clamped.sum(dim=2)).clamp(min=0.0) / b_pinned   # (B,n_chunks,H)
    return clamped.view(B, T, H), shortfall


def classify_budget_partial(shortfall: torch.Tensor, median_thresh: float = 0.10,
                             frac_thresh: float = 0.25) -> dict:
    """Rev 3 NEW-1's SYMMETRIC rule -- ONE numeric rule, every masking cell:
    BUDGET-PARTIAL iff median(shortfall_c) > median_thresh OR
    frac(shortfall_c > median_thresh) > frac_thresh. shortfall: any shape
    (pooled over chunks/heads/batch by the caller before calling this, or
    passed with extra leading dims -- flattened here)."""
    flat = shortfall.reshape(-1)
    median_shortfall = flat.median().item()
    frac_exceeding = (flat > median_thresh).float().mean().item()
    is_partial = (median_shortfall > median_thresh) or (frac_exceeding > frac_thresh)
    return {
        "median_shortfall": median_shortfall,
        "frac_chunks_shortfall_gt_thresh": frac_exceeding,
        "median_thresh": median_thresh, "frac_thresh": frac_thresh,
        "budget_partial": bool(is_partial),
        "rule": "median(shortfall)>0.10 OR frac(shortfall>0.10)>0.25 (Rev 3 NEW-1, symmetric)",
    }


# ---------------------------------------------------------------------------
# sec 4.3 BANDS_PINNED-TrackB metrics -- churn rate, positional TV-distance,
# support size, per-chunk total mass (B_pinned's own raw ingredient).
# ---------------------------------------------------------------------------

def per_chunk_total_mass(beta: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Baseline (unconstrained) per-chunk total write mass -- sec 2
    principle 4's B_pinned is the MEAN of this quantity over Cell 1's own
    archived-checkpoint re-measurement (sec 5.3). beta: (B,T,H). Returns
    (B,n_chunks,H)."""
    B, T, H = beta.shape
    assert T % chunk_size == 0
    n_chunks = T // chunk_size
    return beta.view(B, n_chunks, chunk_size, H).sum(dim=2)


def support_size(beta: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """sec 4.3 candidate-2 support-size metric: count of strictly-positive
    beta entries per (chunk,head). beta: (B,T,H). Returns (B,n_chunks,H)
    int64."""
    B, T, H = beta.shape
    assert T % chunk_size == 0
    n_chunks = T // chunk_size
    return (beta.view(B, n_chunks, chunk_size, H) > 0).sum(dim=2)


def churn_rate(sel_prev: torch.Tensor, sel_curr: torch.Tensor, k_sel: int) -> torch.Tensor:
    """sec 4.3: per-episode |S_t \\ S_{t-1}| / K_sel, mean taken by the
    caller over whatever (chunk,head) axes it pools. sel_prev/sel_curr:
    (...,k_sel) int64 in-chunk selected offsets from the SAME fixed probe
    batch at two log points. |S_t\\S_{t-1}| = |S_{t-1}\\S_t| since
    |S_t|=|S_{t-1}|=k_sel (set-difference sizes coincide for equal-size
    sets), computed here as k_sel - |intersection|."""
    assert sel_prev.shape == sel_curr.shape and sel_prev.shape[-1] == k_sel
    prev = sel_prev.unsqueeze(-1)                # (...,k_sel,1)
    curr = sel_curr.unsqueeze(-2)                # (...,1,k_sel)
    retained = (prev == curr).any(dim=-1).sum(dim=-1).float()   # (...,) |intersection|
    return (k_sel - retained) / k_sel


def tv_distance_from_uniform(offset_counts: torch.Tensor) -> float:
    """sec 4.3 positional-concentration metric (Rev 3 NEW-5): offset_counts
    (n_offsets,) selection counts per intra-chunk offset over a fixed probe
    batch. p_o = f_o/sum(f); TV = 0.5*sum|p_o - 1/n_offsets|. TV in [0,1];
    0 = exactly uniform, larger = more concentrated."""
    n = offset_counts.shape[-1]
    total = offset_counts.sum().clamp(min=1)
    p = offset_counts.float() / total
    uniform = 1.0 / n
    return 0.5 * (p - uniform).abs().sum().item()


# ---------------------------------------------------------------------------
# sec 5.3 -- MC anchor recomputation at (K,d), cross-checked against the
# closed forms (SCALE_TRANSFER_DESIGN.md sec 6.8): E||G-I||_F ~=
# sqrt(K(K-1)/d) (random unit vectors), sqrt(K(K-1)) (full collapse). CPU,
# free, deterministic given seed -- this is the Wave -1 computation the
# task brief directs be done NOW (planning values ~=3.94/~=31.50 at
# K=32,d=64, superseded here by the real MC).
# ---------------------------------------------------------------------------

def mc_recompute_anchors(K: int, d: int, n_samples: int = 200_000, seed: int = 0,
                          batch: int = 4096) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    I_K = torch.eye(K)
    devs = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch, remaining)
        remaining -= b
        X = torch.randn(b, K, d, generator=gen)
        X = F.normalize(X, dim=-1)
        G = X @ X.transpose(-1, -2)
        devs.append((G - I_K).norm(dim=(-2, -1)))
    all_dev = torch.cat(devs)
    mc_random = all_dev.mean().item()
    mc_random_std = all_dev.std(unbiased=True).item()
    closed_random = (K * (K - 1) / d) ** 0.5

    # Collapse anchor: computed in float64 (unlike the random-anchor MC loop above, which stays
    # float32 for MC-scale speed) -- this is a SINGLE deterministic vector, not a Monte-Carlo
    # average, so there is no excuse for float32 rounding to show up as a spurious "MC-vs-closed-
    # form" discrepancy (measured: ~7e-7 absolute at float32, i.e. exactly float32 eps-scale, not
    # a real algebraic mismatch) -- float64 pushes the residual to <1e-12, a genuine numerical
    # verification rather than a precision artifact.
    v = F.normalize(torch.randn(1, d, generator=gen, dtype=torch.float64), dim=-1)
    Xc = v.expand(K, d)
    Gc = Xc @ Xc.transpose(-1, -2)
    dev_collapse = (Gc - I_K.double()).norm().item()
    closed_collapse = (K * (K - 1)) ** 0.5

    return {
        "K": K, "d": d, "n_samples": n_samples, "seed": seed,
        "anchor_random": mc_random, "anchor_random_std": mc_random_std,
        "anchor_random_closed_form": closed_random,
        "anchor_random_mc_minus_closed": mc_random - closed_random,
        "anchor_collapse": dev_collapse, "anchor_collapse_closed_form": closed_collapse,
        "anchor_collapse_mc_minus_closed": dev_collapse - closed_collapse,
    }


# ---------------------------------------------------------------------------
# Candidate 4 (sec 3.4) -- "only if cheap" scope decision (task brief item
# 1): the hard-snap FINAL phase reuses candidate 1's own machinery verbatim
# (sec 3.4's own text: "mechanically identical to candidate 1"). The soft
# auxiliary-loss phase (L_sel = -Gini(beta) or a direct non-selected-mass
# penalty, lambda-scheduled) is NOT built here -- sec 3.4 itself predicts
# it underperforms (F-geo-1 precedent) and sec 10 item 4 ranks it the FIRST
# candidate-mechanism cut; building its calibration-heavy lambda-schedule
# machinery before Wave -1 even runs candidate 1 would not be a cheap
# addition. This function is the full "hard-snap" scope: a schedule gate
# that, once tripped, hands off to hard_topk_beta_mask/apply_hard_select_ste
# unchanged.
# ---------------------------------------------------------------------------

def candidate4_hard_snap_active(step: int, total_steps: int, snap_frac: float = 0.5) -> bool:
    """snap_frac is a PLACEHOLDER default, not a registered spec value (sec
    3.4 leaves the exact snap point unspecified beyond "a final phase") --
    a real Wave 1 candidate-4 cell must register its own snap_frac before
    launch, not silently inherit this default."""
    snap_step = max(1, int(round(snap_frac * total_steps)))
    return step >= snap_step


# ---------------------------------------------------------------------------
# sec 5.1 (Rev 2 M5) -- Cell-3 override stamping, mirroring
# key_anchoring.py's override_stamp_payload / assemble_claim_tier_fields
# precedent exactly (itself adopting lm_pretrain_rd.py::_assemble_result's
# per-run-JSON convention). gate_override is ALWAYS present (True or
# False) at assembly time so its absence can never be misread as "a clean,
# non-override run."
# ---------------------------------------------------------------------------

DEFAULT_GATE_OVERRIDE_REASON = "reference arm, SCALE_TRANSFER_DESIGN.md sec 4.2 outcome (iii)"


def trackb_override_stamp_payload(reason: str = DEFAULT_GATE_OVERRIDE_REASON,
                                   timestamp: float | None = None) -> dict:
    ts = timestamp if timestamp is not None else time.time()
    return {"gate_override": True, "gate_override_reason": reason, "gate_override_at": ts}


def trackb_assemble_gate_override_fields(stamp: dict | None) -> dict:
    if stamp is not None and stamp.get("gate_override"):
        return {"gate_override": True, "gate_override_reason": stamp["gate_override_reason"],
                "gate_override_at": stamp["gate_override_at"], "claim_tier": "descriptive"}
    return {"gate_override": False}
