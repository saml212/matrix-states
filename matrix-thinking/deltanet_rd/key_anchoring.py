"""key_anchoring.py -- shared, FLA-FREE (pure torch) mechanism library for
KEY_ANCHORING_DESIGN.md (Rev 5, commit ca308d6). Every function here is
CPU-testable WITHOUT importing fla/chunk_delta_rule -- deliberately split out
of model_rd.py (which imports fla at module scope for the Triton kernel) so
every piece of THIS design's own new machinery is independently verifiable in
a plain CPU-torch environment: a throwaway venv with no fla at all (the exact
methodology KEY_ANCHORING_ATTACK_R1/R2/R3.md used for every round), AND the
box's own /home/nvidia/tdenv (which does have fla, but whose GPUs are all
busy on other waves -- CPU-only verification is the point, not a fallback).

model_rd.py's DeltaNetRDBlock imports anchor_blend_gather_scatter /
frame_potential_init / raw_table_conditioning / the LAMBDA_* constants /
classify_lambda_band FROM this module rather than reimplementing them, so the
function under test here IS the function bind() actually calls at its geo3
insertion site -- never a hand-copied twin with independent drift risk (the
project's own standing "verify before claiming" / no-silent-duplication
discipline).

Newton-Schulz reuse (flagged for audit scrutiny, a deliberate build
decision): Gate 2's G2-c leg and every CPU smoke that needs an
orthogonalization pass import `newton_schulz` from geo3_simulator.py rather
than `newton_schulz_orthogonalize` from model_rd.py. geo3_simulator.py is
ALSO fla-free by its own docstring ("No fla/chunk_delta_rule dependency --
pure torch, runs anywhere"); the two implementations are mathematically
IDENTICAL (same pre-scaled cubic Newton-Schulz iteration,
X_{t+1} = 1.5 X_t - 0.5 (X_t X_t^T) X_t, X_0 = A/sqrt(K)) and have been
cross-verified to convergence-precision agreement across every attack round
on this design (KEY_ANCHORING_ATTACK_R1/R2/R3.md). Using geo3_simulator's
copy here means Gate 2 (a committed CPU test) and the smoke suite never pull
fla into their import graph just to reach a pure-torch function; the
PRODUCTION bind() path (model_rd.py) still calls its own
geo3_orthogonalize_logged (the fallback-aware, side-channel-logging version)
unchanged.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from geo3_simulator import newton_schulz as _geo3sim_newton_schulz
import rev7_threshold_derive as r7d   # KEY_ANCHORING_DESIGN.md sec 10.3.3 -- pure Python, fla-free,
                                       # no scipy/torch (own docstring); safe at module scope here.

# ---------------------------------------------------------------------------
# sec 2.2 / sec 3.2's registered constants (Rev 4 -- R3 finding 1; Rev 5
# header). Also imported by run_deltanet_rd.py (lambda-trajectory logging +
# the startup cadence assertion) and run_deltanet_rd_exactness_sweep.py (the
# band-derivation window / manifest naming) -- ONE definition, never
# re-typed at each call site.
# ---------------------------------------------------------------------------
LAMBDA_LOG_CADENCE_STEPS = 200
LAMBDA_WINDOW_LOG_POINTS = 5

# sec 4's Gate 2 legs (the amended, three-leg CPU construction gate).
GATE2_SIGMA_RATIO_MIN = 0.1        # G2-a: sigma_64/sigma_1 >= 0.1 (raw table, train-row block)
GATE2_MAX_COS_MAX = 0.5            # G2-b: max_{i!=j} |cos(A_i,A_j)| <= 0.5
GATE2_RESID_TOL = 1e-2             # G2-c: NS admission tolerance (sec 14.10 item 2 semantics)
GATE2_N_ITER_BY_K = {16: 12, 32: 20, 48: 20,
                     34: 20, 38: 20, 42: 20, 46: 20,
                     68: 20, 76: 20, 84: 20, 92: 20,
                     # KEY_ANCHORING_SCALING_DRAFT.md sec 15.6's own fix, the OTHER half of it:
                     # sec 15.6 flags K=48 as a COLLISION risk (an EXISTING key, verified only at
                     # d=64, that this wave's own d=80 grid coincidentally reuses) and requires the
                     # wave's own per-cell n_iter selection to consult a NEW d_state-namespaced
                     # dict instead (run_deltanet_rd_exactness_sweep.py's KEYANCHOR_SCALING_
                     # GATE2_N_ITER_BY_D_K, never this flat dict, for that purpose) -- see
                     # _keyanchor_scaling_spec there. But gate2_construction_check (this module,
                     # directly below) is a SHARED, wave-agnostic utility with no d_state parameter
                     # at all -- it has ALWAYS read this flat dict directly (exactly how 34-46 and
                     # 68-92 above were each added by their own wave's build), and every K below
                     # that is genuinely NEW (never appeared in this dict before, at ANY d_state) is
                     # a plain, safe extension -- there is nothing existing to silently collide
                     # with. 20/24 (the mandatory low-K anchor points, sec 15.3/15.8) and 43/51/53/
                     # 57/58/63/69 (sec 15.3's translated K-grids) are ALL genuinely new; 48 is
                     # deliberately NOT re-added here (already present above, untouched, still only
                     # verified at d=64) -- gate2_construction_test.py's own mandatory pre-launch
                     # gate would otherwise KeyError on this wave's full K-grid (found by this
                     # build's own smoke suite, smoke_keyanchor_scaling.py smoke 13, which executes
                     # gate2_construction_check at d=80/96 with the real registered K-grids). All at
                     # n_iter=20 "by analogy only" (sec 15.6 item 2) pending the mandatory Wave -1
                     # n_iter-sufficiency check -- NOT yet run (see run_deltanet_rd_exactness_
                     # sweep.py's own KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K module-note).
                     20: 20, 24: 20, 43: 20, 51: 20, 53: 20, 57: 20, 58: 20, 63: 20, 69: 20,
                     }   # production tier per K (sec 16.3's own
                                                # n_iter escalation; 48 added per KEY_ANCHORING_
                                                # DESIGN.md sec 11.3 -- "NS at n_iter=12 lands at
                                                # resid 0.104 > tol on realistic near-collinear
                                                # probes; 20 converges to ~1e-6" -- verified this
                                                # build, see gate2_construction_test.py's
                                                # ks=(16,32,48). 34/38/42/46 added per sec 12.2.1
                                                # (Rev 12.1): n_iter=20 was originally chosen BY
                                                # ANALOGY to K=32/48 (both already 20) -- sec
                                                # 12.2.2's own registered Wave -1 sufficiency check
                                                # (attack finding 7), keyanchor_cliff_niter_check.py,
                                                # has since RUN and CONFIRMED convergence at all 4
                                                # new K's (rel_change 20->24 == 0.000000 at every
                                                # K, well under the 0.5% tolerance) -- n_iter=20 is
                                                # verified sufficient, not merely assumed.
                                                # 68/76/84/92 (d_state=128) added per sec 13.2/13.3
                                                # item 5 (Rev 13.2): d=128 is a GENUINELY UNTESTED
                                                # d, not a by-analogy extension -- keyanchor_dstate_
                                                # niter_check.py measured rel_change 20->24 ==
                                                # 0.000000 at ALL FOUR new K's (2026-07 build
                                                # session), same as the d=64 K's above. n_iter=20 is
                                                # therefore verified sufficient at d=128 too, though
                                                # the underlying geometry differs sharply from d=64:
                                                # at d=128 the pooled post-NS pairwise cosine is
                                                # EXACTLY 1.000000 (mean AND p10) at every n_iter in
                                                # {12,16,20,24} tested, not merely n_iter-invariant
                                                # like the d=64 entries (which converge to a
                                                # near-but-not-exactly-1 constant, e.g. 0.9344 at
                                                # K=34) -- because n_entities=107 < d_state=128, the
                                                # Welch coherence floor sqrt((n-d)/(d(n-1))) is
                                                # NEGATIVE (degenerate/zero), i.e. a 107-row table CAN
                                                # be made EXACTLY mutually orthogonal in a 128-dim
                                                # space, so any K<=107 subset is already exactly
                                                # orthonormal pre-NS and NS converges trivially at
                                                # any n_iter -- a genuine geometric regime change
                                                # from d=64 (where n_entities=107 > d_state=64 forces
                                                # a nonzero coherence floor), not a bug or a
                                                # coincidence.
GATE2_N_SUBSETS = 512

# sec 2.2's anchor-init recipe (the frozen frame-potential construction).
ANCHOR_FRAME_POTENTIAL_N_STEPS = 4000
ANCHOR_FRAME_POTENTIAL_LR = 0.05
# Registered construction seed for the anchor table's frame-potential init
# (a build-time choice this wave registers -- the design text pins the
# RECIPE exactly but not a specific numeric seed value, see KEY_ANCHORING_
# ATTACK_R3.md Target 1's own "no seed pinned for [non-regression] cases"
# note; this constant is what makes THIS build's own draw reproducible).
# Swept seeds {1..5, 42, 20260703-5} at construction time: sigma_ratio is
# 1.0000 (the Welch-floor tight-frame property) at EVERY seed (a fixed
# invariant of frame-potential minimization, sec 2.2's own disclosure);
# max|cos| varies 0.278-0.326 across seeds (no exact (107,64) ETF exists,
# so max|cos| is draw-dependent even at the frame-potential optimum) --
# 20260705 was selected because it lands closest to the design's own
# cited session number (max|cos|=0.2832; this build measures 0.28415 at
# 20260705, see the build report / gate2_construction_test.py's printed
# diagnostic for the exact achieved value).
ANCHOR_INIT_SEED = 20260705


# ---------------------------------------------------------------------------
# sec 2.2: frame-potential-minimized anchor-table construction.
# ---------------------------------------------------------------------------

def frame_potential_init(n: int, d: int, seed: int,
                          n_steps: int = ANCHOR_FRAME_POTENTIAL_N_STEPS,
                          lr: float = ANCHOR_FRAME_POTENTIAL_LR) -> torch.Tensor:
    """sec 2.2's pinned recipe: start from seeded random unit rows (n,d);
    minimize the frame potential F(X) = sum_{i!=j} (x_i.x_j)^2 by projected
    gradient descent (grad_i = 4 * sum_{j!=i} (x_i.x_j) x_j, renormalize
    every step); n_steps=4000, lr=0.05 -- deterministic, seeded, CPU,
    one-time. Frame-potential minimizers are TIGHT FRAMES (the rms
    coherence this converges to is a fixed function of (n,d) alone -- the
    Welch floor sqrt((n-d)/(d(n-1))) -- regardless of which specific tight
    frame the descent lands in; max|cos| varies run to run within a small
    band since an exact equiangular tight frame need not exist at every
    (n,d), sec 2.2's own disclosure). fp64 throughout for the iterative
    descent's numerical stability; cast to fp32 on return (the table's
    actual storage dtype, matching every other frozen-table convention in
    this codebase, e.g. embed_arms.py's own QR constructions)."""
    g = torch.Generator().manual_seed(seed)
    X = F.normalize(torch.randn(n, d, generator=g, dtype=torch.float64), dim=-1)
    eye_mask = torch.eye(n, dtype=torch.bool)
    for _ in range(n_steps):
        gram = X @ X.transpose(0, 1)                    # (n,n)
        gram = gram.masked_fill(eye_mask, 0.0)           # exclude i==j from the i!=j sum
        grad = 4.0 * (gram @ X)                          # sum_{j!=i} (x_i.x_j) x_j per row i
        X = F.normalize(X - lr * grad, dim=-1)
    return X.float()


def random_unit_rows_init(n: int, d: int, seed: int) -> torch.Tensor:
    """KEY_ANCHORING_DESIGN.md sec 10.13's registered candidate (e) stub
    ("frozen-random-table ablation") calls for a table that is BOTH never
    trained AND not frame-potential-structured -- the stub's own name and
    sec 10.13.4's motivating text ("a random, FROZEN anchor table... does
    not require the anchor table to carry any entity-specific information
    at all") describe an UNOPTIMIZED random draw, not frame_potential_init's
    own tight-frame-minimized construction (a specific, non-random
    geometric object). This is exactly frame_potential_init's own
    pre-optimization starting point (n_steps=0): seeded random unit rows,
    F.normalize(randn), same RNG convention (fp64 draw, fp32 return) so the
    only difference between candidate (d)'s init and candidate (e)'s init
    is whether the frame-potential descent loop ran at all -- never a
    different distribution or dtype path. requires_grad handling (the
    'frozen' half) is the CALLER's job (model_rd.py's
    anchor_table_frozen flag), matching this module's own init-vs-training
    separation of concerns (frame_potential_init also does not decide
    trainability)."""
    g = torch.Generator().manual_seed(seed)
    X = F.normalize(torch.randn(n, d, generator=g, dtype=torch.float64), dim=-1)
    return X.float()


def raw_table_conditioning(table: torch.Tensor) -> dict:
    """items 6a/6b (sec 3.1, Rev 2 attack-F1 rewrite -- REWRITTEN to the RAW
    table, never anything downstream of Newton-Schulz, which forces
    sigma_K/sigma_1 ~= 1 on ANY input and is therefore tautologically
    blind). table: (n,d), the anchor table's TRAIN-ROW block (e.g. 107x64).
    6a: sigma_min/sigma_max via direct SVD (n>d here, so this is
    sigma_d/sigma_1 -- the house salvage-ratio convention, generalized off
    -square exactly the way salvage_ratio()/model_rd.py already does for
    (K,d) key matrices). 6b: max_{i!=j} |cos(row_i, row_j)|."""
    s = torch.linalg.svdvals(table.float())
    sigma_ratio = (s[-1] / s[0].clamp(min=1e-12)).item()
    row_norm = F.normalize(table.float(), dim=-1)
    cos = row_norm @ row_norm.transpose(0, 1)
    n = table.shape[0]
    off_diag = cos[~torch.eye(n, dtype=torch.bool, device=table.device)]
    max_abs_cos = off_diag.abs().max().item()
    return {
        "sigma_ratio": sigma_ratio, "max_abs_cos": max_abs_cos,
        "sigma_ratio_pass": bool(sigma_ratio >= GATE2_SIGMA_RATIO_MIN),
        "max_abs_cos_pass": bool(max_abs_cos <= GATE2_MAX_COS_MAX),
    }


# ---------------------------------------------------------------------------
# sec 4's Gate 2 -- three-leg CPU construction gate (G2-a/6a, G2-b/6b,
# G2-c NS-conditioning). Runs BOTH at wave-launch time (init) and again at
# every admission checkpoint on the then-current trained table (sec 4's
# "re-run at every admission checkpoint" build requirement, R2-M1).
# ---------------------------------------------------------------------------

def gate2_ns_leg(table: torch.Tensor, K: int, n_iter: int,
                  n_subsets: int = GATE2_N_SUBSETS, resid_tol: float = GATE2_RESID_TOL,
                  seed: int = 0) -> dict:
    """G2-c: >= n_subsets random K-subsets (without replacement) of the
    table's rows, through the SAME (mathematically) Newton-Schulz iteration
    at the production tier for this K (sec 16.3's n_iter=12 @ K=16 /
    n_iter=20 @ K=32), 100% converging to residual <= resid_tol with ZERO
    fallbacks (sec 14.10 item 2's admission semantics, unchanged)."""
    n = table.shape[0]
    assert K <= n, f"cannot draw K={K} rows without replacement from a {n}-row table"
    g = torch.Generator().manual_seed(seed)
    max_resid = 0.0
    n_fallback = 0
    for _ in range(n_subsets):
        idx = torch.randperm(n, generator=g)[:K]
        sub = F.normalize(table[idx], dim=-1).unsqueeze(0)     # (1,K,d)
        _, resid_hist = _geo3sim_newton_schulz(sub, n_iter)
        final_resid = resid_hist[-1]
        max_resid = max(max_resid, final_resid)
        if final_resid > resid_tol:
            n_fallback += 1
    return {"K": K, "n_iter": n_iter, "n_subsets": n_subsets,
            "n_fallback": n_fallback, "max_resid": max_resid,
            "pass": n_fallback == 0}


def gate2_construction_check(table: torch.Tensor, ks=(16, 32), seed: int = 0) -> dict:
    """The full amended Gate 2 (sec 4): all three legs mandatory, ANY leg
    failing blocks the launch. table: the anchor's TRAIN-ROW block only
    (n_train=107 x d_state=64 for the registered wave)."""
    cond = raw_table_conditioning(table)
    ns_legs = {K: gate2_ns_leg(table, K, GATE2_N_ITER_BY_K[K], seed=seed + K) for K in ks}
    overall_pass = (cond["sigma_ratio_pass"] and cond["max_abs_cos_pass"]
                     and all(v["pass"] for v in ns_legs.values()))
    return {
        "g2a_sigma_ratio": cond["sigma_ratio"], "g2a_pass": cond["sigma_ratio_pass"],
        "g2b_max_abs_cos": cond["max_abs_cos"], "g2b_pass": cond["max_abs_cos_pass"],
        "g2c_ns_legs": ns_legs,
        "pass": bool(overall_pass),
    }


# ---------------------------------------------------------------------------
# sec 4's pinned REGRESSION-CASE table constructions. Reconstructed from the
# design's own prose per KEY_ANCHORING_ATTACK_R3.md's Target-1 methodology
# (its own from-scratch reproduction independently matched the design's
# pinned 6a/6b numbers to 4 decimal places using this exact recipe -- "the
# most natural reading of the design's own prose"), so these functions are
# not a fresh guess; they are a verified-reproducible transcription.
# ---------------------------------------------------------------------------

def build_collapsed_table(n: int = 107, d: int = 64, noise_sigma: float = 0.30,
                            seed: int = 42) -> torch.Tensor:
    """sec 4's moderate (noise_sigma=0.30) / severe (noise_sigma=0.05)
    pinned collapse constructions: two shared unit directions + a random
    2-way row assignment + additive Gaussian noise, renormalized, ALL drawn
    from a SINGLE torch.Generator().manual_seed(seed) -- reproduces the
    design's pinned moderate-case numbers (G2-a=0.0762, G2-b=0.5371) and
    severe-case numbers (G2-a=0.0141, G2-b=0.9333) to 4 decimal places
    (KEY_ANCHORING_ATTACK_R3.md Target 1)."""
    g = torch.Generator().manual_seed(seed)
    directions = F.normalize(torch.randn(2, d, generator=g), dim=-1)
    assign = torch.randint(0, 2, (n,), generator=g)
    base = directions[assign]
    noise = torch.randn(n, d, generator=g) * noise_sigma
    return F.normalize(base + noise, dim=-1)


def build_localized_collapse_table(healthy_table: torch.Tensor, n_collapsed: int = 10,
                                     sigma: float = 0.02, seed: int = 0) -> torch.Tensor:
    """sec 4's localized-collapse context case: n_collapsed of the healthy
    (frame-potential) init's own rows replanted onto ONE shared direction
    with small noise sigma, renormalized -- the max-statistic (6b)
    dilution-immunity demonstration: 6a passes (aggregate dilution over the
    other 97 healthy rows), 6b still fails (a max statistic cannot be
    diluted). No single seed is pinned for this case in the design text
    (KEY_ANCHORING_ATTACK_R3.md Target 1's own provenance-gap note); this
    build registers `seed` as its own reproducible choice."""
    n, d = healthy_table.shape
    assert n_collapsed <= n
    g = torch.Generator().manual_seed(seed)
    direction = F.normalize(torch.randn(d, generator=g), dim=-1)
    idx = torch.randperm(n, generator=g)[:n_collapsed]
    table = healthy_table.clone()
    noise = torch.randn(n_collapsed, d, generator=g) * sigma
    table[idx] = F.normalize(direction.unsqueeze(0) + noise, dim=-1)
    return table


# ---------------------------------------------------------------------------
# KEY_ANCHORING_DESIGN.md sec 14.1/14.1b item 1 (Rev 14.3) -- the coherence
# dose dial. PORTED VERBATIM from `dose_dial_verify.py` (the committed,
# CPU-verified Rev-14.3 reference implementation that found and fixed the
# subspace_rank=d_state degeneracy, sec 14.1's own boxed correction) --
# same function bodies, same bisection contract, same monotonicity
# assumption. `smoke_keyanchor_dose.py`'s dose-dial-matches-reference-JSON
# smoke re-derives every t/achieved pair in dose_dial_verify_results.json's
# own `calibration` block from THESE functions (not a hand-copied twin) and
# asserts agreement to 1e-4, per sec 14.1b item 1's acceptance criterion.
#
# DIFFUSE_SUBSPACE_RANK=48 (sec 14.1's Rev-14.3 fix): the MAXIMUM scanned
# rank whose achieved-dose ceiling (at t=1) still clears the 0.42 target --
# subspace_rank=d_state (128) is a mathematically degenerate no-op dial
# (QR of a square Gaussian gives a full orthogonal Q, so the projector is
# the identity and the blend collapses to a no-op at every t, sec 14.1's
# `degeneracy_proof`); this is NOT used anywhere in this design.
# ---------------------------------------------------------------------------

DIFFUSE_SUBSPACE_RANK = 48   # sec 14.1's Rev-14.3 chosen diffuse rank (ceiling 0.422673 >= 0.42 target)


def build_dose_table(healthy_table: torch.Tensor, t: float, subspace_rank: int,
                      subspace_seed: int) -> torch.Tensor:
    """sec 14.1's dose dial: blend EVERY row of healthy_table toward a
    shared low-rank random subspace at strength t in [0,1], renormalize.
    t=0 -> healthy_table unchanged (rows already unit-norm, renormalize is
    a no-op). t=1 -> every row replaced by its unit-normalized projection
    onto the subspace (near-maximal coherence, bounded by subspace_rank).

    subspace_rank=4 -> CONCENTRATED (rank-4) injection (this wave's primary
    structure). subspace_rank=DIFFUSE_SUBSPACE_RANK (48) -> the DIFFUSE
    injection (Rev 14.3's co-primary arm): every row still blends toward a
    fixed random rotation-subspace of itself, spreading the blend across
    more directions than rank-4 concentrates it into. subspace_rank=d_state
    is DEGENERATE (see module note above) -- never pass it here.

    fp64 throughout (matching frame_potential_init's own numerical-
    stability convention); cast to fp32 on return (the table's actual
    storage dtype)."""
    n, d = healthy_table.shape
    g = torch.Generator().manual_seed(subspace_seed)
    Q, _ = torch.linalg.qr(torch.randn(d, subspace_rank, generator=g, dtype=torch.float64))
    proj = healthy_table.double() @ Q @ Q.transpose(0, 1)
    proj_unit = F.normalize(proj, dim=-1)
    blended = (1.0 - t) * healthy_table.double() + t * proj_unit
    return F.normalize(blended, dim=-1).float()


def achieved_dose(healthy_table: torch.Tensor, t: float, subspace_rank: int,
                   subspace_seed: int) -> float:
    """The achieved max|cos| (raw_table_conditioning's own statistic) of
    build_dose_table's output at a given (t, subspace_rank, subspace_seed)
    -- the quantity calibrate_dose bisects against."""
    dosed = build_dose_table(healthy_table, t, subspace_rank, subspace_seed)
    return raw_table_conditioning(dosed)["max_abs_cos"]


def calibrate_dose(healthy_table: torch.Tensor, target_dose: float, subspace_rank: int,
                    subspace_seed: int, tol: float = 1e-4, max_iter: int = 80) -> dict:
    """Bisection on t in [0,1] against achieved max|cos| (sec 14.1b item 1).
    Returns {"t", "achieved", "n_iters", "converged", "ceiling_at_t1"}. If
    the achieved dose at t=1 (the dial's own maximum reach) is still below
    target_dose, calibration cannot converge -- reported explicitly
    (converged=False, ceiling_at_t1=dose at t=1), never silently clamped or
    fudged. Relies on achieved_dose(t) being monotone non-decreasing in t
    for a fixed subspace (more blend toward a fixed direction set cannot
    reduce coherence) -- standard bisection applies."""
    lo, hi = 0.0, 1.0
    dose_lo = achieved_dose(healthy_table, lo, subspace_rank, subspace_seed)
    dose_hi = achieved_dose(healthy_table, hi, subspace_rank, subspace_seed)
    if dose_hi < target_dose - tol:
        return {"t": 1.0, "achieved": dose_hi, "n_iters": 0, "converged": False,
                "ceiling_at_t1": dose_hi}
    n_iters = 0
    t = hi
    achieved = dose_hi
    for i in range(max_iter):
        n_iters = i + 1
        t = 0.5 * (lo + hi)
        achieved = achieved_dose(healthy_table, t, subspace_rank, subspace_seed)
        if abs(achieved - target_dose) <= tol:
            break
        if achieved < target_dose:
            lo = t
        else:
            hi = t
    converged = abs(achieved - target_dose) <= tol
    return {"t": t, "achieved": achieved, "n_iters": n_iters, "converged": converged,
            "ceiling_at_t1": dose_hi}


# ---------------------------------------------------------------------------
# sec 2.2's Rev-4 masked gather/scatter anchor blend -- THE function bind()
# calls at its geo3 insertion site (model_rd.py imports this directly).
# ---------------------------------------------------------------------------

def anchor_blend_gather_scatter(k_eff_raw: torch.Tensor, anchor_weight: torch.Tensor,
                                  anchor_trained_mask: torch.Tensor, key_ids: torch.Tensor,
                                  lam) -> torch.Tensor:
    """sec 2.2's Rev-4 masked gather/scatter blend -- closes R3 finding 4's
    `torch.where` 0xNaN backward-poisoning path (verified real, then
    verified fixed, by an executable contrast run in
    KEY_ANCHORING_ATTACK_R3.md's round-4 verify; this is the SAME
    construction, reproduced again by this build's own smoke 4). Arithmetic
    touches TRAINED-entity rows ONLY; held-out rows are a bit-exact `clone`
    with NO graph edge into them in either direction (forward OR backward)
    -- `k_blend_raw`'s held-out rows are never multiplied, added, or
    re-normalized, so a strict `torch.equal` bypass check (sec 3.3) is
    exactly correct, not a floating-point-lucky approximation.

    k_eff_raw: (B,K,d) this episode's raw gathered (pre-orthogonalization)
      keys -- `_gather_at(k_norm_raw, item_pos)` in model_rd.py's bind().
    anchor_weight: (vocab,d) the anchor nn.Embedding's `.weight`.
    anchor_trained_mask: (vocab,) bool buffer, True at trained-entity rows
      (`pools.train_name_ids`) -- the M1/C17 bypass (sec 3.3).
    key_ids: (B,K) int64 real entity token ids (`batch["key_ids"]`).
    lam: scalar tensor or python float in [0,1], the CURRENT lambda value
      (sigmoid(raw_param) for learned mode, or the fixed-grid value).
    Returns k_blend_raw (B,K,d) -- feed DIRECTLY into
    geo3_orthogonalize_logged, an UNCHANGED call (sec 2.2)."""
    trained_here = anchor_trained_mask[key_ids]                # (B,K) bool
    t_idx = trained_here.nonzero(as_tuple=True)                 # (row_idx, col_idx)
    sub_blend = F.normalize(
        (1.0 - lam) * k_eff_raw[t_idx] + lam * anchor_weight[key_ids[t_idx]], dim=-1)
    k_blend_raw = k_eff_raw.clone()
    k_blend_raw[t_idx] = sub_blend
    return k_blend_raw


# ---------------------------------------------------------------------------
# sec 3.2's lambda-trajectory band classification + sec 3.7's engaged_frac
# band classification. Pure arithmetic on already-computed scalars -- no
# torch dependency needed, kept here so run_deltanet_rd.py / the sweep
# harness / the smoke suite share ONE classifier.
# ---------------------------------------------------------------------------

def assert_lambda_log_cadence(log_every: int) -> None:
    """sec 3.2/R3 finding 1's startup assertion: a run with learned-lambda
    trajectory logging active must use EXACTLY the registered cadence
    (LAMBDA_LOG_CADENCE_STEPS=200) -- a cadence change fires this assertion
    LOUDLY instead of silently resizing the trailing-window statistic.
    Called once at train() startup whenever anchor_active and
    anchor_lambda_mode=='learned' (run_deltanet_rd.py)."""
    assert log_every == LAMBDA_LOG_CADENCE_STEPS, (
        f"lambda-trajectory logging requires log_every == LAMBDA_LOG_CADENCE_STEPS "
        f"({LAMBDA_LOG_CADENCE_STEPS}), got log_every={log_every} -- sec 3.2/R3 finding 1: a "
        f"cadence change would silently resize the 5-logged-point trailing window this design's "
        f"lambda-band classification depends on. Fix --log-every, or (if the cadence is "
        f"deliberately changing) re-register LAMBDA_LOG_CADENCE_STEPS/LAMBDA_WINDOW_LOG_POINTS "
        f"as a documented deviation, not a silent default change.")


def _lambda_point_band(x: float) -> str:
    if 0.2 <= x <= 0.8:
        return "interior"
    if x > 0.95:
        return "pin_rediscovery"
    if x < 0.05:
        return "not_recruited"
    return "ambiguous"


def classify_lambda_band(final_value: float, trailing_mean: float, trailing_range: float) -> str:
    """sec 3.2's three-part per-seed band rule (Rev 3 -- R2 MAJOR 5;
    window pinned in LOG-POINT space at Rev 4 -- R3 finding 1): ALL THREE
    of (i) the final logged value, (ii) the trailing mean (over the last
    LAMBDA_WINDOW_LOG_POINTS logged points), in the SAME band, AND (iii)
    the trailing range (max-min over that same window) < 0.1 -- any
    failure lands in 'ambiguous' regardless of the mean (a pre-registered
    exclusion, not a post-hoc judgment call)."""
    if trailing_range >= 0.1:
        return "ambiguous"
    b_final, b_mean = _lambda_point_band(final_value), _lambda_point_band(trailing_mean)
    if b_final != b_mean:
        return "ambiguous"
    return b_final


def arm_lambda_label(per_seed_bands: list[str]) -> str:
    """Arm-level label (sec 3.2): >=2/3 seeds in a band assigns that band;
    seeds spread across non-adjacent bands (no band reaching >=2/3) ->
    'ambiguous', no headline."""
    from collections import Counter
    counts = Counter(per_seed_bands)
    n = len(per_seed_bands)
    for band, ct in counts.items():
        if band != "ambiguous" and ct / n >= 2.0 / 3.0 - 1e-9:
            return band
    return "ambiguous"


def engaged_frac_band(engaged_frac: float) -> str:
    """sec 3.7's per-entity anchor-input-alignment engagement bands:
    >=90% headline-eligible; [50%,90%) partial anchoring (Outcome A'');
    <50% not recruited regardless of aggregate drift (routes to Outcome C)."""
    if engaged_frac >= 0.9:
        return "headline_eligible"
    if engaged_frac >= 0.5:
        return "partial_anchoring"
    return "not_recruited"


def lambda_window_summary(lambda_trajectory: list[dict], window_points: int = LAMBDA_WINDOW_LOG_POINTS) -> dict:
    """Computes the sec 3.2 per-seed summary fields from a run's own logged
    lambda trajectory (a list of {"step":.., "lambda":..} dicts, one per
    logged point at LAMBDA_LOG_CADENCE_STEPS cadence). Returns final value,
    trailing-window mean/range, and the classified band -- the exact
    machine-readable fields sec 2.2's logging bullet registers."""
    assert len(lambda_trajectory) >= 1, "empty lambda trajectory"
    vals = [pt["lambda"] for pt in lambda_trajectory]
    window = vals[-window_points:]
    final_value = vals[-1]
    trailing_mean = sum(window) / len(window)
    trailing_range = max(window) - min(window)
    band = classify_lambda_band(final_value, trailing_mean, trailing_range)
    return {"final_value": final_value, "trailing_mean": trailing_mean,
            "trailing_range": trailing_range, "n_window_points": len(window), "band": band}


# ---------------------------------------------------------------------------
# sec 3.6's mechanical BANDS_PINNED blinding gate -- writer, launcher-gate
# validator, and readout-timestamp assertion. Shared by
# run_deltanet_rd_exactness_sweep.py (writer + launcher gate) and any
# readout/analysis script (timestamp assertion).
# ---------------------------------------------------------------------------

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def reference_arm_result_valid(result: dict, expected_steps: int) -> bool:
    """A reference-arm result JSON 'validates as complete' (sec 3.6 writer
    requirement 1) iff: complete==true, steps_completed==expected_steps,
    AND its final checkpoint carries the drift fields (both post-NS and
    pre-NS, sec 3.6's 'instrument symmetry' requirement)."""
    if result.get("complete") is not True:
        return False
    if result.get("steps_completed", 0) < expected_steps:
        return False
    checkpoints = result.get("checkpoints") or []
    if not checkpoints:
        return False
    final = checkpoints[-1]
    drift = final.get("drift_probe")
    if not isinstance(drift, dict):
        return False
    return ("post_ns" in drift and "mean" in drift["post_ns"]
            and "pre_ns" in drift and "mean" in drift["pre_ns"])


def reference_final_drift(result: dict) -> float:
    """The ONE extraction both the BANDS_PINNED writer (run_deltanet_rd_
    exactness_sweep.write_bands_pinned_if_ready) and the validate-time
    re-derivation below use: a reference arm's final-checkpoint post-NS
    pooled drift mean (sec 3.6's band-derivation input). One definition --
    a writer/reader extraction seam here would let validate re-derive a
    DIFFERENT statistic than the one that was pinned (the e633862-audit F1
    failure mode, applied to field extraction instead of file paths)."""
    return result["checkpoints"][-1]["drift_probe"]["post_ns"]["mean"]


def derive_engaged_bands(per_k_final_drift: dict, ceiling_by_k: dict) -> dict:
    """sec 3.6's band derivation (n=3 per K, Rev 4 -- R3 finding 3):
    engaged_K = mean_ref + 2*s_ref (sample std, n=3, df=2) over the 3
    reference-arm FINAL-CHECKPOINT post-NS drift values at that K.
    Degenerate-case guard: engaged_K >= ceiling_K - 0.005 -> UNRESOLVABLE
    at that K, with a mandatory leave-one-out sensitivity report
    (disclosure-only -- no re-decision rule hangs off it, per Rev 4's own
    scoping).

    per_k_final_drift: {K: [v0, v1, v2]} (3 reference seeds' own final
    post-NS drift means). ceiling_by_k: {K: computed_ceiling} (sec 2.2:
    0.9423 @ K=32, 0.9745 @ K=16)."""
    out = {}
    for K, vals in per_k_final_drift.items():
        assert len(vals) == 3, f"K={K}: band derivation requires exactly 3 reference seeds, got {len(vals)}"
        t = torch.tensor(vals, dtype=torch.float64)
        mean_ref = t.mean().item()
        s_ref = t.std(unbiased=True).item()          # sample std, n=3, df=2
        engaged_k = mean_ref + 2.0 * s_ref
        ceiling = ceiling_by_k[K]
        unresolvable = engaged_k >= ceiling - 0.005
        loo = []
        for drop in range(3):
            keep = [vals[i] for i in range(3) if i != drop]
            kt = torch.tensor(keep, dtype=torch.float64)
            loo.append(kt.mean().item() + 2.0 * kt.std(unbiased=True).item())
        out[K] = {
            "mean_ref": mean_ref, "s_ref": s_ref, "engaged_k": engaged_k,
            "ceiling": ceiling, "unresolvable": bool(unresolvable),
            "per_seed": vals, "leave_one_out_engaged_k": loo,
        }
    return out


def write_bands_pinned(path: str, per_k_final_drift: dict, ceiling_by_k: dict,
                        reference_result_paths: dict) -> dict:
    """sec 3.6 writer requirement (a): writes BANDS_PINNED.json ONLY after
    every reference arm's result JSON validates complete (the CALLER is
    responsible for having already checked reference_arm_result_valid on
    every one of the 6 files -- this function's OWN job is to compute the
    bands and stamp the sha256 hashes + timestamp, matching the design's
    'derived bands + per-seed inputs + formula version + sha256 hashes of
    all 6 reference-arm result JSONs + timestamp' content spec).

    reference_result_paths: {K: [path0, path1, path2]} -- the 6 reference
    result JSON paths, hashed here (never re-derived silently later)."""
    bands = derive_engaged_bands(per_k_final_drift, ceiling_by_k)
    hashes = {str(K): [sha256_of_file(p) for p in paths] for K, paths in reference_result_paths.items()}
    doc = {
        "formula_version": "sec 3.6 Rev4 (n=3, engaged_K = mean_ref + 2*s_ref, df=2)",
        "bands": {str(K): v for K, v in bands.items()},
        "reference_result_paths": {str(K): paths for K, paths in reference_result_paths.items()},
        "reference_result_sha256": hashes,
        "pinned_at": time.time(),
        "pinned_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return doc


def validate_bands_pinned(path: str, ceiling_by_k: dict | None = None) -> dict | None:
    """sec 3.6 launcher-gate requirement (b): anchor cells REFUSE to launch
    unless this file exists AND re-hashing every referenced reference-arm
    JSON matches the recorded hash (a changed reference JSON post-pin is a
    pin-integrity error, never silently re-derived) AND -- e633862 audit
    F2 fix, the REV7-pin M1 live-re-derivation pattern applied here -- the
    STORED bands dict is value-identical to a fresh derive_engaged_bands()
    re-run over those same (hash-validated) reference JSONs. Before this
    fix, only the reference files were hash-checked: a tampered engaged_k
    /per_seed/unresolvable INSIDE the pin itself passed silently, making
    the 'tamper-evident' claim false for the pin's own load-bearing
    contents.

    ceiling_by_k (optional): the registered {K: ceiling} constants (e.g.
    run_deltanet_rd_exactness_sweep.keyanchor_ceiling_by_k()). When given,
    the stored per-K ceilings must equal them exactly -- closing the one
    field the content re-derivation alone cannot pin (the re-derivation
    reads the stored ceiling as its input, so a tampered ceiling would
    otherwise re-derive self-consistently). Optional so readout-only
    callers without the sweep module still get the full content check.

    Returns the parsed doc on success, None on any validation failure
    (missing file, missing field, hash mismatch, bands-content mismatch,
    or ceiling mismatch) -- callers are responsible for the loud refusal
    message; this function never raises for an ordinary "not ready yet"
    state, only for a genuinely malformed file (a bug, not a sequencing
    issue)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    for K, paths in doc.get("reference_result_paths", {}).items():
        recorded_hashes = doc["reference_result_sha256"][K]
        for p, expected in zip(paths, recorded_hashes):
            if not os.path.exists(p):
                return None
            if sha256_of_file(p) != expected:
                return None
    # -- F2: content re-derivation. Rebuild per-K final drifts from the
    # hash-validated reference JSONs (same reference_final_drift extraction
    # the writer used), re-run derive_engaged_bands with the STORED
    # ceilings, and require value-identity with the stored bands dict
    # (json round-trip normalizes container types; floats serialize
    # losslessly, and the recomputation is the same deterministic float64
    # arithmetic on the same inputs, so equality is exact, not approximate).
    try:
        per_k_final_drift = {}
        stored_ceilings = {}
        for K, paths in doc["reference_result_paths"].items():
            vals = []
            for p in paths:
                with open(p) as f:
                    vals.append(reference_final_drift(json.load(f)))
            per_k_final_drift[K] = vals
            stored_ceilings[K] = doc["bands"][K]["ceiling"]
        rederived = derive_engaged_bands(per_k_final_drift, stored_ceilings)
        rederived_norm = json.loads(json.dumps({str(K): v for K, v in rederived.items()}))
        if rederived_norm != doc["bands"]:
            return None
    except (KeyError, IndexError, TypeError, AssertionError):
        return None
    if ceiling_by_k is not None:
        for K, expected in ceiling_by_k.items():
            entry = doc["bands"].get(str(K))
            if entry is None or entry.get("ceiling") != expected:
                return None
    return doc


def assert_blind_not_broken(bands_doc: dict, anchor_started_ats: list) -> None:
    """sec 3.6 readout-assertion requirement (c): the pin timestamp must
    STRICTLY PRECEDE the earliest anchor-arm start time. Raises
    AssertionError (the readout aborts, the wave summary marks the blind
    broken, every affected anchor readout demotes to descriptive tier --
    sec 3.6) if violated."""
    assert anchor_started_ats, "no anchor-arm start times given -- nothing to check the blind against"
    earliest_anchor_start = min(anchor_started_ats)
    pinned_at = bands_doc["pinned_at"]
    assert pinned_at < earliest_anchor_start, (
        f"BLIND BROKEN: BANDS_PINNED.json pinned_at={pinned_at} does NOT strictly precede the "
        f"earliest anchor-arm start_at={earliest_anchor_start} -- sec 3.6's mechanical readout "
        f"assertion. Every affected anchor readout must report at descriptive tier only.")


# ---------------------------------------------------------------------------
# sec 3.6 Rev 5's --unblind-override demotion stamping -- writes into EVERY
# affected anchor-arm's OWN result JSON at assembly time (the R4-1 fix:
# never the wave summary alone), mirroring lm_pretrain_rd.py's own
# claim_tier / _assemble_result precedent (L1090, smoke-asserted L1420).
# ---------------------------------------------------------------------------

def override_stamp_payload(timestamp: float | None = None) -> dict:
    """What the launcher threads down to a spawned anchor run when
    --unblind-override is passed (sec 3.6 Rev 5). `timestamp` defaults to
    now() -- passed explicitly in tests for determinism."""
    ts = timestamp if timestamp is not None else time.time()
    return {"unblind_override": True, "unblind_override_at": ts}


def assemble_claim_tier_fields(stamp: dict | None) -> dict:
    """sec 3.6 Rev 5: fields written into EVERY anchor-arm result JSON at
    assembly time (run_deltanet_rd.py's _assemble_result), regardless of
    the override path. `claim_tier` is written ONLY on the override path
    (the demotion is the one tier verdict knowable at launch time,
    independent of anything the run later earns); `unblind_override` is
    ALWAYS present (True or False) so its absence can never be read as
    evidence of a clean blind -- presence is mandatory either way."""
    if stamp is not None and stamp.get("unblind_override"):
        return {"claim_tier": "descriptive", "unblind_override": True,
                "unblind_override_at": stamp["unblind_override_at"]}
    return {"unblind_override": False}


# ---------------------------------------------------------------------------
# sec 14.6's pinned per-entity drift measurement -- moved here (out of
# geo3_drift_diagnostic.py) so BOTH geo3_drift_diagnostic.py (reference
# arms / bare geo3) and run_deltanet_rd.py's own optional per-checkpoint
# reference-arm drift probe (sec 3.6: "PLUS the per-checkpoint drift
# diagnostic active") can import the SAME functions without creating a
# circular import (geo3_drift_diagnostic.py already does
# `from run_deltanet_rd import train as train_geo3` at module scope; having
# run_deltanet_rd.py import THAT module back would be a real cycle -- this
# fla-free shared module is the fix, not a workaround).
# ---------------------------------------------------------------------------

def sample_batch_fixed_entity(cfg, batch_size, fixed_entity_id, gen, pools, device):
    """Verbatim (moved, not rewritten) from geo3_drift_diagnostic.py's own
    function of the same name -- sec 14.6's 'fixing one entity's raw key
    and resampling its K-1 episode-mates' construction. Entity slot 0 is
    PINNED to fixed_entity_id in every row; the other K-1 slots are
    freshly, independently drawn without replacement from the train pool
    minus the fixed entity."""
    import grammar_rd as grd
    B, K = batch_size, cfg.K
    buf_len, clause_len, T_bind = cfg.buf_len, cfg.clause_len, cfg.T_bind
    other_pool = pools.train_name_ids[pools.train_name_ids != fixed_entity_id].to(device)
    assert other_pool.numel() >= K - 1, \
        f"train pool minus the fixed entity ({other_pool.numel()}) < K-1={K - 1}"
    scores = torch.rand(B, other_pool.numel(), generator=gen, device=device)
    pool_idx = scores.argsort(dim=-1)[:, :K - 1]
    other_entities = other_pool[pool_idx]
    fixed_col = torch.full((B, 1), int(fixed_entity_id), dtype=torch.int64, device=device)
    entity_ids = torch.cat([fixed_col, other_entities], dim=1)

    succ = grd._permutation_graph(B, K, gen, device, dtype=torch.float32)
    key_ids = entity_ids
    value_ids = torch.gather(entity_ids, 1, succ)

    rel_pool = pools.rel_a_ids.to(device)
    rel_idx = torch.randint(0, rel_pool.numel(), (B,), generator=gen, device=device)
    rel_id = rel_pool[rel_idx]

    token_ids = torch.full((B, T_bind), int(pools.buffer_id), dtype=torch.int64, device=device)
    item_pos = (torch.arange(K, device=device) * clause_len + buf_len + 2).unsqueeze(0).expand(B, K).contiguous()
    key_pos = item_pos - 2
    rel_pos = item_pos - 1
    period_pos = item_pos + 1
    token_ids.scatter_(1, key_pos, key_ids)
    token_ids.scatter_(1, rel_pos, rel_id.unsqueeze(1).expand(B, K))
    token_ids.scatter_(1, item_pos, value_ids)
    token_ids.scatter_(1, period_pos,
                        torch.full((B, K), int(pools.period_id), dtype=torch.int64, device=device))
    beta_mask = torch.zeros(B, T_bind, device=device, dtype=torch.float32)
    beta_mask.scatter_(1, item_pos, torch.ones(B, K, device=device))
    return {"token_ids": token_ids, "beta_mask": beta_mask, "item_pos": item_pos, "key_ids": key_ids}


def pairwise_drift_stats(out_rows: torch.Tensor) -> tuple[float, float]:
    """(mean, p10) of all pairwise cosines among one entity's rows across
    context resamples. out_rows: (n_draws, d)."""
    n = out_rows.shape[0]
    pw = F.cosine_similarity(out_rows.unsqueeze(0), out_rows.unsqueeze(1), dim=-1)
    off = pw[~torch.eye(n, dtype=torch.bool, device=out_rows.device)]
    return off.mean().item(), torch.quantile(off, 0.10).item()


@torch.no_grad()
def measure_entity_rows(model, cfg, pools, fixed_entity_id, n_resamples, gen, device,
                          pre_ns_attr: str | None = None):
    """One entity's n_resamples (post-NS) orthogonalized-key rows PLUS
    (sec 3.1's item-5 instrument symmetry) its pre-NS rows read from the
    model's own side channel, if `pre_ns_attr` names one present on the
    model (`geo3_last_k_eff_raw` for bare geo3 / candidate (c);
    `anchor_last_k_blend_raw` for candidate (d)). Returns
    (post_ns_rows, pre_ns_rows_or_None), each (n_resamples, d_state)."""
    model.eval()
    b = sample_batch_fixed_entity(cfg, n_resamples, fixed_entity_id, gen, pools, device)
    _, k_eff_items, _ = model.bind(b)
    post_ns = k_eff_items[:, 0, :].detach()
    pre_ns = None
    if pre_ns_attr is not None:
        raw = getattr(model, pre_ns_attr, None)
        if raw is not None:
            pre_ns = raw[:, 0, :].detach()
    return post_ns, pre_ns


def measure_drift(model, cfg, pools, n_entities, n_resamples, gen, device,
                    pre_ns_attr: str | None = None):
    """sec 14.6's full pinned statistic (post-NS), PLUS (when pre_ns_attr
    is given) the same pooled mean/p10 statistic on the pre-NS side
    channel -- sec 3.1's item-5 evidentiary quantity for anchor arms."""
    entity_pool = pools.train_name_ids.to(device)
    assert entity_pool.numel() >= n_entities
    perm = torch.randperm(entity_pool.numel(), generator=gen, device=device)[:n_entities]
    entities = entity_pool[perm]
    per_entity_stats = []
    all_off_diag_post, all_off_diag_pre = [], []
    for eid in entities.tolist():
        post_rows, pre_rows = measure_entity_rows(model, cfg, pools, eid, n_resamples, gen, device,
                                                    pre_ns_attr=pre_ns_attr)
        mean_c, p10_c = pairwise_drift_stats(post_rows)
        entry = {"entity_id": eid, "post_ns_mean": mean_c, "post_ns_p10": p10_c}
        n = post_rows.shape[0]
        pw = F.cosine_similarity(post_rows.unsqueeze(0), post_rows.unsqueeze(1), dim=-1)
        all_off_diag_post.append(pw[~torch.eye(n, dtype=torch.bool, device=post_rows.device)])
        if pre_rows is not None:
            mean_pre, p10_pre = pairwise_drift_stats(pre_rows)
            entry["pre_ns_mean"] = mean_pre
            entry["pre_ns_p10"] = p10_pre
            pw_pre = F.cosine_similarity(pre_rows.unsqueeze(0), pre_rows.unsqueeze(1), dim=-1)
            all_off_diag_pre.append(pw_pre[~torch.eye(n, dtype=torch.bool, device=pre_rows.device)])
        per_entity_stats.append(entry)
    pooled_post = torch.cat(all_off_diag_post)
    out = {
        "post_ns": {"mean": pooled_post.mean().item(), "p10": torch.quantile(pooled_post, 0.10).item()},
        "n_entities": n_entities, "n_resamples": n_resamples,
        "n_pooled_pairs": pooled_post.numel(), "per_entity": per_entity_stats,
    }
    if all_off_diag_pre:
        pooled_pre = torch.cat(all_off_diag_pre)
        out["pre_ns"] = {"mean": pooled_pre.mean().item(), "p10": torch.quantile(pooled_pre, 0.10).item()}
    return out


# ---------------------------------------------------------------------------
# sec 3.7's full-pool per-entity anchor-INPUT-alignment sweep + h=1
# behavioral companion. MOVED HERE (2026-07-06, keyanchor-confirm build) from
# keyanchor_drift_diagnostic.py, where they were previously the ONLY place
# either computation was wired into ANY running code (KEY_ANCHORING_DESIGN.md
# sec 9.3's documented gap: the mandatory Wave-1 training cells never called
# either function, since keyanchor_wave1_manifest() never threaded
# drift_probe=True into their specs). Moving them here lets BOTH
# keyanchor_drift_diagnostic.py's standalone probe driver AND
# run_deltanet_rd.py's own --drift-probe checkpoint block (train(), the
# actual 20,000-step admitted arms) call the SAME function -- one
# definition, never a hand-copied twin (this project's standing
# no-silent-duplication discipline, the same reasoning key_anchoring.py's own
# docstring gives for sample_batch_fixed_entity above).
#
# Both functions need model_rd.DeltaNetRDBlock / grammar_rd's batch sampler --
# imported LOCALLY inside each function body (never at this module's top
# level), so key_anchoring.py's own module-level import stays fla-free
# (model_rd.py imports fla at module scope; importing it only INSIDE these
# two functions means merely importing key_anchoring.py never requires fla --
# only CALLING these two functions does, exactly the same constraint
# sample_batch_fixed_entity's own local `import grammar_rd as grd` respects).
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_full_pool_alignment(model, cfg, pools, n_resamples, gen, device):
    """sec 3.7's per-entity anchor-INPUT-alignment sweep, full pool (ALL
    train entities, not a random n_entities sample): a_e = mean over
    >= n_resamples independent episode resamples of cos(pre-NS blended key
    of entity e, A[e]). Uses the SAME sample_batch_fixed_entity construction
    as measure_drift (sec 14.6's sampling machinery) -- eval-only, no
    training-path change. `engaged_frac` = fraction of entities with
    a_e >= 0.9 (sec 3.7's registered band driver)."""
    assert model.anchor_active, "measure_full_pool_alignment requires anchor_active=True"
    model.eval()
    entity_ids = pools.train_name_ids.tolist()
    a_e = {}
    for eid in entity_ids:
        b = sample_batch_fixed_entity(cfg, n_resamples, eid, gen, pools, device)
        _, k_eff_items, _ = model.bind(b)
        blended = model.anchor_last_k_blend_raw[:, 0, :]              # (n_resamples, d_state)
        anchor_row = model.anchor_table.weight[eid].unsqueeze(0).expand_as(blended)
        cos = F.cosine_similarity(blended, anchor_row, dim=-1)
        a_e[eid] = cos.mean().item()
    engaged_frac = sum(1 for v in a_e.values() if v >= 0.9) / len(a_e)
    return {"a_e": a_e, "engaged_frac": engaged_frac, "n_resamples": n_resamples}


@torch.no_grad()
def measure_h1_behavioral_companion(model, cfg, pools, gen, device, n_batches=4, batch_size=64):
    """sec 3.7 Rev4's non-load-bearing behavioral companion: per-entity h=1
    recovery, restricted to eval episodes CONTAINING that entity --
    bookkeeping on the existing eval (tagging each per-item cosine with its
    row's own key_ids before pooling), NO new forward passes beyond a
    standard h=1 eval sweep this diagnostic already needs to run once."""
    import grammar_rd as grd_local

    model.eval()
    per_entity_cos: dict[int, list[float]] = {}
    for _ in range(n_batches):
        b = grd_local.sample_batch_rd(cfg, batch_size, gen, hop_set=(1,), pools=pools, device=device)
        pred, targets, S_T, k_eff_items, v_eff_items = model(b, force_rank_k=None)
        cos = F.cosine_similarity(pred, targets, dim=-1)         # (B,Q)
        key_ids = b["key_ids"]                                    # (B,K) -- entities present THIS episode
        for row in range(cos.shape[0]):
            row_entities = set(key_ids[row].tolist())
            row_cos = cos[row].mean().item()                      # this row's own h=1 recovery
            for eid in row_entities:
                per_entity_cos.setdefault(eid, []).append(row_cos)
    return {eid: sum(vals) / len(vals) for eid, vals in per_entity_cos.items()}


# =============================================================================
# KEY_ANCHORING_DESIGN.md sec 10 (Rev 7.1) -- mechanism-tier confirmation wave
# machinery. Everything below is NEW (2026-07-06 keyanchor-mech build):
#   - sec 10.3.3's pin triple (writer already exists as rev7_threshold_
#     derive.py/REV7_THRESHOLD_PINNED.json, committed at e740a12; this adds
#     the LAUNCHER-GATE and READOUT legs).
#   - sec 10.2/10.3's r_e + null-pool (C matrix) measurement, computed on the
#     PRE-BLEND raw key (model.geo3_last_k_eff_raw), never backed out
#     through the lambda-dependent blend -- and the SAME mean-of-cosines
#     code path for both r_e and the null pool (M2's fix), so C[e,e]==r_e[e]
#     is an in-memory identity, not an assertion.
#   - sec 10.3.2's pooled null check, per-entity empirical percentile, and
#     the hub-detection diagnostic -- AMENDED to median+2*MAD (see
#     HUB_MAD_K's own docstring; KEYANCHOR_REV7_ATTACK.md sec 16.5).
#   - sec 10.3.1's BH-FDR (primary)/Bonferroni/BY engagement test, ALL
#     constants loaded from REV7_THRESHOLD_PINNED.json, never recomputed
#     inline (sec 10.3.3 leg (iii)'s own discipline).
#   - sec 10.3.4's joint (detection-rate AND effect-size) band assignment.
#   - sec 10.5.1's candidate (d') per-entity lambda blend, interior-band-
#     fraction, Spearman rank check, and the formalized Hartigan's dip test.
# =============================================================================

# ---------------------------------------------------------------------------
# sec 10.3.3: the pin triple's launcher-gate (ii) and readout (iii) legs.
# The writer (i) already exists (rev7_threshold_derive.py, committed at
# e740a12) -- these two functions are the mechanical enforcement Rev 7.1's
# text specifies but the Rev-7.1-bounded-verify round (KEYANCHOR_REV7_ATTACK
# .md sec 16.3) correctly flagged as "not yet written -- Wave -1/build-phase
# work". Mirrors validate_bands_pinned/assert_blind_not_broken's own
# contract exactly (sec 3.6), applied to the SEPARATE, zero-data-dependency
# REV7_THRESHOLD_PINNED.json artifact.
# ---------------------------------------------------------------------------

def validate_rev7_pin(pin_path: str | None = None, script_path: str | None = None) -> dict | None:
    """sec 10.3.3 leg (ii), the launcher gate: anchor-arm cells (candidate
    (d) AND (d')) REFUSE to launch unless (i) REV7_THRESHOLD_PINNED.json
    exists and parses, (ii) sha256(rev7_threshold_derive.py) in the working
    tree matches the hash recorded INSIDE the pin, AND (iii) a LIVE re-run
    of derive() (using the pin's own recorded inputs) reproduces the pin's
    `derived` block byte-identically -- a hash match alone would not catch
    an environment-dependent numeric drift in a pure-Python implementation;
    this live re-run is the entire point of leg (ii)'s third clause (sec
    10.3.3, matching KEYANCHOR_REV7_ATTACK.md sec 16.3's own "gate must
    include a live re-run, not just a hash check" requirement).

    Returns the parsed pin dict on success, None on ANY validation failure
    (missing file, script-hash mismatch, or a derive()-mismatch) -- callers
    are responsible for the loud refusal message, exactly
    validate_bands_pinned's own contract."""
    here = os.path.dirname(os.path.abspath(__file__))
    pin_path = pin_path or os.path.join(here, "REV7_THRESHOLD_PINNED.json")
    script_path = script_path or os.path.join(here, "rev7_threshold_derive.py")
    if not os.path.exists(pin_path) or not os.path.exists(script_path):
        return None
    try:
        with open(pin_path) as f:
            pin = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    recorded_hash = pin.get("provenance", {}).get("script_sha256")
    if recorded_hash is None or sha256_of_file(script_path) != recorded_hash:
        return None
    try:
        inputs = pin["derived"]["inputs"]
        live_derived = r7d.derive(n_entities=inputs["n_entities"], d_state=inputs["d_state"],
                                    alpha=inputs["alpha"])
    except (KeyError, TypeError):
        return None
    if live_derived != pin["derived"]:
        return None
    return pin


def assert_rev7_pin_not_broken(pin_doc: dict, anchor_started_ats: list) -> None:
    """sec 10.3.3 leg (iii), the readout assertion: the pin's own
    provenance timestamp must STRICTLY PRECEDE the earliest anchor-arm
    start. REV7_THRESHOLD_PINNED.json's provenance block (already
    committed at e740a12, attack-round-verified byte-identical) carries
    only an ISO-8601 string (`generated_at`), no raw epoch field --
    parsed here rather than adding a new field to an already-committed,
    hash-chained artifact (which would require regenerating and
    re-committing it, re-opening a question the bounded verify already
    closed). Raises AssertionError (the readout aborts, every affected
    anchor readout demotes to descriptive tier) on violation -- same
    failure-mode class as assert_blind_not_broken (sec 3.6)."""
    assert anchor_started_ats, "no anchor-arm start times given -- nothing to check the pin against"
    ts_str = pin_doc["provenance"]["generated_at"]
    import datetime
    pinned_at = datetime.datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.timezone.utc).timestamp()
    earliest_anchor_start = min(anchor_started_ats)
    assert pinned_at < earliest_anchor_start, (
        f"REV7 PIN BROKEN: REV7_THRESHOLD_PINNED.json provenance.generated_at={ts_str} "
        f"({pinned_at:.0f}) does NOT strictly precede the earliest anchor-arm "
        f"start_at={earliest_anchor_start:.0f} -- sec 10.3.3 leg (iii)'s mechanical readout "
        f"assertion. Every affected anchor readout must report at DESCRIPTIVE TIER ONLY.")


# ---------------------------------------------------------------------------
# sec 10.2/10.3.2: r_e (measured directly, pre-blend) + the null-pool matrix
# C[i,j], computed via the IDENTICAL mean-of-cosines code path (M2's fix) --
# C[e,e] is bit-identical to r_e[e] by CONSTRUCTION (same tensor, same
# formula), not merely asserted equal.
# ---------------------------------------------------------------------------

def compute_C_matrix(raw_keys: torch.Tensor, anchor_rows: torch.Tensor) -> torch.Tensor:
    """raw_keys: (n, R, d) -- n entities' own R per-resample PRE-BLEND raw
    keys (model.geo3_last_k_eff_raw[:,0,:], sec 10.1.1's non-degenerate
    construction). anchor_rows: (n, d) -- the SAME n entities' anchor rows,
    in the SAME order. Returns C (n,n): C[i,j] = mean over the R resamples
    of cos(raw_keys[i,r,:], anchor_rows[j,:]) -- MEAN-OF-COSINES (per-
    resample cosine, then averaged), never cosine-of-mean (attack-R7
    finding 4 / the sec 9.7.10 item-5 Jensen's-gap recurrence). C's
    diagonal (i==j, same entity ordering on both axes) is therefore
    EXACTLY r_e as sec 10.2 defines it -- an identity, not an assertion."""
    raw_norm = F.normalize(raw_keys, dim=-1)                 # (n,R,d)
    anchor_norm = F.normalize(anchor_rows, dim=-1)           # (n,d)
    per_resample = torch.einsum('nrd,md->nmr', raw_norm, anchor_norm)   # (n,m,R) -- [i,j,r]
    return per_resample.mean(dim=-1)                          # (n,n) mean-of-cosines over resamples


def pooled_null_check(off_diag: torch.Tensor, mean_tol: float | None = None,
                        sd_range: tuple[float, float] | None = None,
                        d_state: int = 64) -> dict:
    """sec 10.3.2's pooled decision rule: compare the null-pool's empirical
    (mean, SD) against the analytic null's (0, sigma_chance=1/sqrt(d_state)).
    Both tolerances derive from the SAME pre-committed, round, symmetric
    formulas (0.25*sigma and +/-~25% of sigma) -- NOT tuned after seeing a
    number. sec 13 build-audit Finding 4 (2026-07-06): the tolerances were
    previously HARDCODED at d=64's sigma=0.125 (mean_tol=0.03125,
    sd_range=(0.100, 0.156)); at d=128 sigma=0.0884 sits BELOW the stale
    0.100 floor, so every d=128 null pool would spuriously fail sd_ok and
    silently switch engagement classification to the empirical fallback.
    Now derived from d_state with the d=64 defaults reproduced EXACTLY at
    d_state=64 (0.25*0.125=0.03125; the sd band keeps the original
    registered endpoints' ratios 0.8*sigma and 1.248*sigma, giving the
    identical (0.100, 0.156) at d=64). Explicit mean_tol/sd_range args
    still override (smoke fixtures use this). `pass=True` confirms the
    exact-Beta test (sec 10.3.1) as primary; `pass=False` switches the
    primary result to the empirical permutation p-value (sec 10.3.2), a
    decision this function reports but never makes for the caller."""
    sigma = 1.0 / math.sqrt(d_state)
    if mean_tol is None:
        mean_tol = 0.25 * sigma
    if sd_range is None:
        sd_range = (0.8 * sigma, 1.248 * sigma)
    mean_v = off_diag.mean().item()
    sd_v = off_diag.std(unbiased=True).item()
    mean_ok = abs(mean_v) <= mean_tol
    sd_ok = sd_range[0] <= sd_v <= sd_range[1]
    return {"mean": mean_v, "sd": sd_v, "mean_tol": mean_tol, "sd_range": list(sd_range),
            "mean_ok": mean_ok, "sd_ok": sd_ok, "pass": bool(mean_ok and sd_ok),
            "n_pairs": off_diag.numel()}


def per_entity_empirical_percentile(C: torch.Tensor) -> torch.Tensor:
    """sec 10.3.2's per-entity null-validation layer (attack-R7 finding 5):
    each entity's own r_e=C[e,e] ranked within ITS OWN row of n-1
    mismatched-pair cosines: p_emp_e = (1 + #{j!=e: C[e,j] >= r_e}) / n.
    Resolution floor 1/n (n=107 -> ~0.0093) -- a validation layer, never a
    replacement primary (BH on the exact-Beta p-values, sec 10.3.1, stays
    primary regardless of this layer's readings)."""
    n = C.shape[0]
    r_e = C.diag()
    ge = (C >= r_e.unsqueeze(1)).sum(dim=1).to(torch.float64) - 1.0   # exclude the i==j entry itself
    return ((1.0 + ge) / n).to(torch.float64)


# sec 10.3.2 item 2, AMENDED at the Rev 7.1 bounded verify
# (KEYANCHOR_REV7_ATTACK.md sec 16.5, MINOR, folded in before build per the
# coordinator's clearance): mean+2*SD has NO formal breakdown-point
# guarantee and is provably masked by a contaminated fraction around
# 20-25/107 (the attack round's own hub-COUNT sweep, holding effect size
# fixed at 0.35: flagged at n_hub=20 [threshold 0.339<0.35], no longer
# flagged at n_hub=25 [threshold 0.379>0.35]). median+2*MAD (breakdown
# point 50%) does not share this failure mode while still correctly
# flagging the ORIGINAL constructed 5/107-hub scenario the rule was built
# to catch (re-verified sec 16.5). r_min_partial=2*sigma_chance (sec
# 10.3.4) is UNCHANGED -- a fixed theoretical constant on the null scale,
# never an empirical dispersion estimate, and not subject to this masking
# mechanism at all.
HUB_MAD_K = 2.0


def median_mad(x: torch.Tensor) -> tuple[float, float]:
    """Median and UNSCALED median absolute deviation: MAD = median(|x -
    median(x)|) -- the textbook definition, no 1.4826 normal-consistency
    rescaling (this is used as a robust DISPERSION ESTIMATE for a
    threshold comparison, not as a sigma estimator, so no rescaling
    convention is imposed)."""
    med = x.median().item()
    mad = (x - med).abs().median().item()
    return med, mad


def hub_detection_median_mad(row_means: torch.Tensor, k: float = HUB_MAD_K) -> dict:
    """sec 10.3.2 item 2 (AMENDED, see HUB_MAD_K docstring): any entity with
    m_e > median(row_means) + k*MAD(row_means) is flagged as a hub.
    row_means: (n,) -- each entity's own mismatched-row mean m_e =
    mean_{j!=e}(C[e,j]). Returns the threshold + a (n,) bool mask; the
    caller maps the mask back onto entity ids (this function is pure
    tensor math, entity-id-agnostic, mirroring compute_C_matrix)."""
    med, mad = median_mad(row_means)
    threshold = med + k * mad
    flagged_mask = row_means > threshold
    return {"median": med, "mad": mad, "k": k, "threshold": threshold, "flagged_mask": flagged_mask}


def bh_step_up_decision(p_values: dict, thresholds: list) -> dict:
    """Generic Benjamini-Hochberg (or Benjamini-Yekutieli, given pre-scaled
    thresholds) step-up decision: sort p ascending, find the LARGEST k
    with p_(k) <= thresholds[k-1] (1-indexed rank k, thresholds a
    length-n ascending list); declare ranks 1..k 'engaged'. `thresholds`
    is loaded FROM the pin (sec 10.3.1's BH `step_up_thresholds`, or the
    caller's own BY-rescaled list) -- never recomputed with a different
    formula at this call site."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    n = len(items)
    assert len(thresholds) == n, f"thresholds length {len(thresholds)} != n_tests {n}"
    k_max = 0
    for k in range(1, n + 1):
        if items[k - 1][1] <= thresholds[k - 1]:
            k_max = k
    engaged_ids = {eid for eid, _ in items[:k_max]}
    engaged = {eid: (eid in engaged_ids) for eid, _ in items}
    return {"k_discoveries": k_max, "n_tests": n, "engaged": engaged,
            "sorted_p": [[eid, p] for eid, p in items]}


def band_v3_assign(discovery_rate: float, median_r_e: float, pin_derived: dict) -> str:
    """sec 10.3.4's joint (detection-rate AND effect-size) band assignment,
    both floors read FROM the pin (never a literal re-typed here). Proved
    (KEYANCHOR_REV7_ATTACK.md sec 16.6) a total, mutually exclusive
    partition: C is the exact negation of A_partial's own criterion, and A
    is a strict subset of A_partial -- every (rate, median) combination
    routes somewhere, none is unrouted."""
    floors = pin_derived["effect_size_floors"]
    r_partial, r_headline = floors["r_min_partial_band"], floors["r_min_headline_band"]
    if discovery_rate >= 0.90 and median_r_e >= r_headline:
        return "A"
    if discovery_rate >= 0.50 and median_r_e >= r_partial:
        return "A_partial"
    return "C"


@torch.no_grad()
def measure_r_e_and_null_pool(model, cfg, pools, n_resamples, gen, device, pin_derived: dict) -> dict:
    """sec 10.2/10.3's full per-entity engagement measurement -- the
    primary registered driver for `engaged_frac_v3`. Uses the SAME
    sample_batch_fixed_entity + model.bind() construction as
    measure_full_pool_alignment, but reads `model.geo3_last_k_eff_raw`
    (the PRE-BLEND raw key, sec 10.1.1's non-degenerate object) instead of
    `model.anchor_last_k_blend_raw` -- r_e is measured upstream of the
    lambda-dependent blend entirely, per sec 10.2's own redesign.

    pin_derived: the `derived` block of a `validate_rev7_pin(...)` result
    (loaded ONCE by the caller, never re-validated per call)."""
    assert model.anchor_active and model.geo3_active, \
        "measure_r_e_and_null_pool requires anchor_active=True and geo3_active=True"
    model.eval()
    entity_ids = pools.train_name_ids.tolist()
    n = len(entity_ids)
    d = model.d_state
    raw_keys = torch.empty(n, n_resamples, d, device=device)
    for i, eid in enumerate(entity_ids):
        b = sample_batch_fixed_entity(cfg, n_resamples, eid, gen, pools, device)
        model.bind(b)
        raw_keys[i] = model.geo3_last_k_eff_raw[:, 0, :].detach()
    idx_t = torch.tensor(entity_ids, device=device, dtype=torch.int64)
    anchor_rows = model.anchor_table.weight[idx_t].detach()          # (n,d)
    anchor_row_norms = anchor_rows.norm(dim=-1)                       # (n,) sec 10.2.1

    C = compute_C_matrix(raw_keys, anchor_rows)                       # (n,n)
    r_e = C.diag().clone()
    eye = torch.eye(n, dtype=torch.bool, device=C.device)
    off_diag_all = C[~eye]
    # sec 13 build-audit Finding 4: derive the tolerance band from the pin's
    # own d_state (falls back to the anchor table's width -- both resolve to
    # 64 for every pre-sec-13 caller, preserving prior behavior exactly).
    _d_state = int((pin_derived.get("inputs") or {}).get("d_state") or anchor_rows.shape[-1])
    pooled = pooled_null_check(off_diag_all, d_state=_d_state)

    row_means = (C.sum(dim=1) - C.diag()) / (n - 1)                   # m_e, excluding the diagonal
    hub = hub_detection_median_mad(row_means)
    hub_flagged = {eid: bool(hub["flagged_mask"][i].item()) for i, eid in enumerate(entity_ids)}

    p_emp_vec = per_entity_empirical_percentile(C)
    p_emp = {eid: p_emp_vec[i].item() for i, eid in enumerate(entity_ids)}

    d_state = pin_derived["inputs"]["d_state"]
    use_empirical = not pooled["pass"]
    p_values = {}
    if use_empirical:
        n_pool = off_diag_all.numel()
        for i, eid in enumerate(entity_ids):
            cnt = (off_diag_all >= r_e[i]).sum().item()
            p_values[eid] = (1.0 + cnt) / (n_pool + 1.0)
    else:
        for i, eid in enumerate(entity_ids):
            p_values[eid] = r7d.p_one_sided_exact(r_e[i].item(), d_state)

    bh_thresholds = pin_derived["primary_bh"]["step_up_thresholds"]
    bh = bh_step_up_decision(p_values, bh_thresholds)
    by_q = pin_derived["by_crosscheck"]["by_effective_q"]
    by_thresholds = [(k / n) * by_q for k in range(1, n + 1)]
    by = bh_step_up_decision(p_values, by_thresholds)
    r_crit = pin_derived["bonferroni_crosscheck"]["r_crit_exact_beta"]
    bonferroni = {eid: bool(r_e[i].item() >= r_crit) for i, eid in enumerate(entity_ids)}

    engaged = bh["engaged"]
    n_engaged_all = sum(1 for v in engaged.values() if v)
    rate_with_hubs = n_engaged_all / n
    non_hub_ids = [eid for eid in entity_ids if not hub_flagged[eid]]
    n_engaged_ex = sum(1 for eid in non_hub_ids if engaged[eid])
    rate_without_hubs = (n_engaged_ex / len(non_hub_ids)) if non_hub_ids else float("nan")

    median_r_e = r_e.median().item()
    band_with = band_v3_assign(rate_with_hubs, median_r_e, pin_derived)
    band_without = band_v3_assign(rate_without_hubs, median_r_e, pin_derived)

    return {
        "r_e": {eid: r_e[i].item() for i, eid in enumerate(entity_ids)},
        "anchor_row_norms": {
            "per_entity": {eid: anchor_row_norms[i].item() for i, eid in enumerate(entity_ids)},
            "mean": anchor_row_norms.mean().item(), "min": anchor_row_norms.min().item(),
            "max": anchor_row_norms.max().item()},
        "pooled_null_check": pooled,
        "hub_detection": {"median_rowmeans": hub["median"], "mad_rowmeans": hub["mad"],
                            "k": hub["k"], "threshold": hub["threshold"],
                            "flagged": hub_flagged, "n_flagged": sum(hub_flagged.values())},
        "per_entity_empirical_percentile": p_emp,
        "primary_branch": "empirical_permutation" if use_empirical else "exact_beta",
        "p_values": p_values,
        "bh": {"k_discoveries": bh["k_discoveries"], "engaged": bh["engaged"]},
        "by": {"k_discoveries": by["k_discoveries"], "engaged": by["engaged"]},
        "bonferroni": bonferroni,
        "engaged_frac_v3_with_hubs": rate_with_hubs,
        "engaged_frac_v3_without_hubs": rate_without_hubs,
        "median_r_e": median_r_e,
        "band_with_hubs": band_with, "band_without_hubs": band_without,
        "band": band_without,   # sec 10.3.2: assigned from the EXCLUDING-hubs figure on divergence
        "n_entities": n, "n_resamples": n_resamples,
    }


# ---------------------------------------------------------------------------
# sec 10.5.1: candidate (d') -- per-entity learned lambda. Blend function
# (the ONE architectural extension model_rd.py's bind() needs), interior-
# band-fraction, Spearman rank check, and the formalized Hartigan's dip
# test.
# ---------------------------------------------------------------------------

def anchor_blend_gather_scatter_per_entity(k_eff_raw: torch.Tensor, anchor_weight: torch.Tensor,
                                              anchor_trained_mask: torch.Tensor, key_ids: torch.Tensor,
                                              anchor_lambda_weight: torch.Tensor) -> torch.Tensor:
    """sec 10.5.1, candidate (d'): the IDENTICAL masked gather/scatter/
    held-out-bypass shape as anchor_blend_gather_scatter, with the single
    scalar lambda replaced by a PER-ENTITY value gathered from
    anchor_lambda_weight (the anchor_lambda_table nn.Embedding's own
    (vocab,1) .weight) instead. sigmoid is applied HERE (init raw=0 ->
    lambda_e=0.5 for every entity, matching candidate (d)'s own starting
    point exactly, sec 10.5.1's own registered init discipline)."""
    trained_here = anchor_trained_mask[key_ids]
    t_idx = trained_here.nonzero(as_tuple=True)
    lam_e = torch.sigmoid(anchor_lambda_weight[key_ids[t_idx]].squeeze(-1))   # (len(t_idx),), NOT a scalar
    sub_blend = F.normalize(
        (1.0 - lam_e[:, None]) * k_eff_raw[t_idx] + lam_e[:, None] * anchor_weight[key_ids[t_idx]], dim=-1)
    k_blend_raw = k_eff_raw.clone()
    k_blend_raw[t_idx] = sub_blend
    return k_blend_raw


def interior_band_fraction_per_entity(per_entity_trajectory: dict,
                                        window_points: int = LAMBDA_WINDOW_LOG_POINTS) -> dict:
    """sec 10.5.1's per-entity extension of classify_lambda_band /
    lambda_window_summary (sec 3.2): per_entity_trajectory is
    {entity_id: [v0, v1, ...]} (one value per LOGGED cadence point, in
    step order, one list per trained entity). Applies the SAME
    three-part (final + trailing-mean + trailing-range<0.1) band rule
    PER ENTITY and reports the fraction landing 'interior' -- mirroring
    engaged_frac's own fraction framing."""
    bands = {}
    for eid, vals in per_entity_trajectory.items():
        window = vals[-window_points:]
        final_value = vals[-1]
        trailing_mean = sum(window) / len(window)
        trailing_range = max(window) - min(window)
        bands[eid] = classify_lambda_band(final_value, trailing_mean, trailing_range)
    n = len(bands)
    interior_frac = (sum(1 for b in bands.values() if b == "interior") / n) if n else float("nan")
    return {"per_entity_band": bands, "interior_frac": interior_frac, "n_entities": n}


def spearman_rank_corr(x: dict, y: dict) -> dict:
    """Two-sided Spearman rank correlation over the shared entity-id keys
    of x and y (e.g. sec 10.5.1's final lambda_e vs final r_e check) --
    pure torch/Python, no scipy.

    Tie handling (e633862 audit fold-in 3 -- the earlier docstring claimed
    ties are "measure-zero", which is FALSE for the data this actually
    runs on: fp32 quantization plus training dynamics can produce exact
    duplicates, e.g. multiple entities pinned at an identical lambda_e or
    saturated at sigmoid's numeric limits): ties are broken by argsort
    order (each of a tied group gets a distinct consecutive integer rank
    in input order), NOT the textbook average-rank convention. Direction
    of the deviation: input-order tie-breaking injects rank noise that is
    uncorrelated with the other variable, which attenuates |rho| toward 0
    relative to average-rank Spearman -- i.e. it biases AGAINST declaring
    significance, the conservative direction for this test's role in the
    sec 10.6 routing table (a significant Spearman upgrades candidate
    (d')'s finding; ties can only make that harder to earn, never easier).

    Significance via the standard Fisher z-transform large-n normal
    approximation (z = atanh(rho)*sqrt(n-3), two-sided p from the normal
    CDF via math.erf) -- standard practice for Spearman at n>10, no
    scipy/table lookup needed. A SINGLE test per seed (sec 10.5.1: no BH
    correction -- one test, not 107)."""
    keys = sorted(set(x.keys()) & set(y.keys()))
    n = len(keys)
    assert n >= 4, f"Spearman requires >=4 paired observations, got {n}"
    xs = torch.tensor([x[k] for k in keys], dtype=torch.float64)
    ys = torch.tensor([y[k] for k in keys], dtype=torch.float64)
    rx = xs.argsort().argsort().to(torch.float64)
    ry = ys.argsort().argsort().to(torch.float64)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.norm() * ry.norm()).item()
    rho = (rx @ ry).item() / denom if denom > 0 else 0.0
    rho = max(-1.0, min(1.0, rho))
    if n > 3 and abs(rho) < 1.0:
        z = math.atanh(rho) * math.sqrt(n - 3)
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    else:
        z, p_value = float("nan"), (0.0 if abs(rho) >= 1.0 else float("nan"))
    return {"rho": rho, "n": n, "z": z, "p_value": p_value, "significant_at_0.05": bool(p_value < 0.05)}


# -- Hartigan's dip statistic (Hartigan & Hartigan, 1985), formalizing the
# Rev-6 informal max-gap check (KEYANCHOR_REV7_ATTACK.md sec 9.7.10 item 5's
# own suggestion). Implementation note (a disclosed simplification, NOT a
# bit-exact port of the reference Fortran/C `dip.f`): for every candidate
# modal-split index, the greatest convex minorant (GCM) of the ECDF
# restricted to the left segment and the least concave majorant (LCM) of
# the right segment are computed via the standard monotone-chain convex-
# hull algorithm (GCM = lower hull, LCM = upper hull of the (x_i, F_i)
# point set) -- mathematically the same GCM/LCM objects the reference
# algorithm tracks incrementally, reformulated as an O(n^2 log n) "try
# every split" search rather than an O(n) incremental scan. At n<=107
# (this design's own entity-pool size) this is trivially fast (well under
# a second per call) -- efficiency was never the constraint; the
# correctness of a from-scratch reimplementation was, and a convex-hull
# construction is far less bug-prone than a hand-ported incremental scan.
# Validated (sec 10.9 item 11) against a known-unimodal and a known-
# bimodal synthetic sample BEFORE it ever touches real r_e/lambda_e data.
def _cross(o: tuple, a: tuple, b: tuple) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _lower_convex_hull(pts: list) -> list:
    """Monotone-chain lower convex hull (points pre-sorted by x)."""
    hull: list = []
    for p in pts:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return hull


def _upper_convex_hull(pts: list) -> list:
    """Monotone-chain upper convex hull (points pre-sorted by x)."""
    hull: list = []
    for p in pts:
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    return hull


def _hull_interp(hull: list, x: float) -> float:
    """Piecewise-linear interpolation of a (sorted-by-x) convex hull at x,
    via bisect (O(log hull size))."""
    hull_xs = [p[0] for p in hull]
    if x <= hull_xs[0]:
        return hull[0][1]
    if x >= hull_xs[-1]:
        return hull[-1][1]
    j = bisect.bisect_right(hull_xs, x) - 1
    j = max(0, min(j, len(hull) - 2))
    x0, y0 = hull[j]
    x1, y1 = hull[j + 1]
    if x1 == x0:
        return y1
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def hartigan_dip_statistic(x) -> float:
    """The sup-norm distance from the empirical CDF to the CLOSEST
    unimodal distribution function (one whose CDF is convex up to a mode,
    concave after), minimized over the choice of modal split point --
    Hartigan & Hartigan's (1985) dip statistic. See the module-level
    comment above for the implementation note."""
    xs = sorted(float(v) for v in x)
    n = len(xs)
    if n < 4:
        return 0.0
    Fv = [(i + 1) / n for i in range(n)]
    pts = list(zip(xs, Fv))
    best_dip = float("inf")
    for j in range(1, n):
        left = pts[:j + 1]
        right = pts[j:]
        if len(left) < 2 or len(right) < 2:
            continue
        gcm = _lower_convex_hull(left)
        lcm = _upper_convex_hull(right)
        dev_left = max(abs(Fv[i] - _hull_interp(gcm, xs[i])) for i in range(0, j + 1))
        dev_right = max(abs(Fv[i] - _hull_interp(lcm, xs[i])) for i in range(j, n))
        best_dip = min(best_dip, max(dev_left, dev_right))
    return 0.0 if best_dip == float("inf") else best_dip


def gate_checkpoint_round_trip(ckpt_paths: list, expected_d_state: int = 64) -> dict:
    """sec 10.10 item 2, the readout-time GATE: the wave cannot be marked
    KEYANCHOR_MECH_CHAIN_DONE until every cell's checkpoint file (a) exists,
    (b) round-trip-loads (torch.load succeeds, the expected keys are
    present, every tensor is finite), and (c) -- e633862 audit fold-in 2 --
    every tensor has the EXPECTED SHAPE: anchor_table_trained_rows must be
    (n, expected_d_state) with n == anchor_train_ids.numel() (a 107x32
    impostor is a structurally wrong artifact, not a tolerable variant);
    anchor_lambda_table_trained_rows, when present, must be (n, 1);
    anchor_lambda_raw, when present, must be a single scalar. Refusal, not
    a silent skip, matching sec 3.6's own pattern for BANDS_PINNED.
    Extends smoke item 12's synthetic round-trip check to REAL run output."""
    REQUIRED_KEYS = ("step", "anchor_train_ids", "anchor_table_trained_rows")
    bad = []
    for p in ckpt_paths:
        if not os.path.exists(p):
            bad.append((p, "missing"))
            continue
        try:
            payload = torch.load(p, weights_only=True)
        except Exception as e:
            bad.append((p, f"load_failed: {e!r}"))
            continue
        missing_keys = [k for k in REQUIRED_KEYS if k not in payload]
        if missing_keys:
            bad.append((p, f"missing_keys: {missing_keys}"))
            continue
        non_finite = [k for k, v in payload.items()
                      if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]
        if non_finite:
            bad.append((p, f"non_finite_tensors: {non_finite}"))
            continue
        n = payload["anchor_train_ids"].numel()
        rows = payload["anchor_table_trained_rows"]
        if tuple(rows.shape) != (n, expected_d_state):
            bad.append((p, f"shape_mismatch: anchor_table_trained_rows {tuple(rows.shape)} != "
                            f"({n}, {expected_d_state})"))
            continue
        lam_rows = payload.get("anchor_lambda_table_trained_rows")
        if lam_rows is not None and tuple(lam_rows.shape) != (n, 1):
            bad.append((p, f"shape_mismatch: anchor_lambda_table_trained_rows "
                            f"{tuple(lam_rows.shape)} != ({n}, 1)"))
            continue
        lam_raw = payload.get("anchor_lambda_raw")
        if lam_raw is not None and lam_raw.numel() != 1:
            bad.append((p, f"shape_mismatch: anchor_lambda_raw numel {lam_raw.numel()} != 1"))
    return {"n_checked": len(ckpt_paths), "n_bad": len(bad), "bad": bad, "pass": len(bad) == 0}


def hartigan_dip_test(x, n_boot: int = 500, seed: int = 0, alpha: float = 0.05) -> dict:
    """sec 10.5.1's formalized Hartigan's dip test (replacing Rev 6's
    informal max-gap check). p-value via Monte Carlo bootstrap against the
    Uniform(0,1) null at the SAME sample size (the standard dip-test
    calibration distribution, Hartigan & Hartigan 1985 -- uniform is the
    'least-favorable' unimodal case for this statistic). Run on BOTH the
    lambda_e distribution (candidate (d') only) and the r_e distribution
    (both (d) and (d')), alpha=0.05 pre-registered."""
    n = len(x)
    observed = hartigan_dip_statistic(x)
    g = torch.Generator().manual_seed(seed)
    boot_dips = [hartigan_dip_statistic(torch.rand(n, generator=g).tolist()) for _ in range(n_boot)]
    boot_t = torch.tensor(boot_dips)
    p_value = ((boot_t >= observed).sum().item() + 1.0) / (n_boot + 1.0)
    return {"dip_statistic": observed, "n": n, "n_boot": n_boot, "p_value": p_value,
            "significant_at_alpha": bool(p_value < alpha), "alpha": alpha}
