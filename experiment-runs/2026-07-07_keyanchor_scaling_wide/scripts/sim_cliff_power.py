"""sim_cliff_power.py -- KEY_ANCHORING_DESIGN.md sec 12, Rev 12.1.

CPU-only power/precision simulation for the capacity-cliff localization
wave, run BEFORE any GPU cell of that wave launches (attack finding 1,
sec 12.7 attack-round-1 item 3: "no power analysis is shown"). This
script IS that power analysis, and its own output numbers are the ones
sec 12.4's re-registered PARAMETER-ESTIMATION deliverable (x0 +/- CI,
w <= C) is pre-registered against -- not a decoration computed after
the fact.

What changed and why (sec 12's own revision log, sec 12.8, has the full
finding -> fix map): the original design (Rev 12.0) posed a BINARY
model-comparison test (sigmoid vs. linear, RSS-ratio >=3x / <=1.5x /
"ambiguous" in between) on only 6 points total (2 archived + K in
{36,40,44}), fit least-squares with NO account for the archived
per-seed noise (~7-11% relative, measured directly off the K=32/K=48
h4 arrays this session -- see NOISE_REL_SD below). An attack round
simulated that binary test under the archived noise level and found
P(ambiguous) = 86.8% at a true cliff width w=0.03 and 95.8% at w=0.08 --
the test was underpowered by construction, most runs would report
"ambiguous" regardless of the true shape, and the 3 archived (K/d, h4)
points do not sit cleanly on ANY noiseless sigmoid (best-fit noiseless
residual RSS ~= 0.0076, i.e. the "clean" curve this design imagined
already doesn't exist in the calibration data it's built on).

Reframing (finding 1a): stop asking "sigmoid or linear, yes/no" and
instead FIT the sigmoid and report the cliff LOCATION x0 and WIDTH w
with a seed-level parametric bootstrap CI. There is no "ambiguous"
outcome under this framing by construction -- a wide CI is itself the
answer ("we cannot localize the cliff tighter than +/- B"), reported
honestly rather than forced into a binary label.

This script answers, PURELY IN SIMULATION (no GPU, no archived-cell
re-use beyond reading noise levels and the K=16/32/48 anchor points
already in the design document):
  1. Given the archived per-seed relative noise, how wide is the
     bootstrap CI on x0 for several CANDIDATE new-point sets (varying
     which K's get sampled and how many seeds each)?
  2. Which point set + seed count minimizes expected CI(x0) width
     under a fixed GPU budget?
  3. Is the best affordable configuration's expected CI narrower than
     the (0.5, 0.75) interval it's trying to subdivide? (sec 12.7
     finding 1c's WITHDRAWN criterion.)

Usage:
    python sim_cliff_power.py --out sim_cliff_power_results.json
    python sim_cliff_power.py --n-trials 2000 --out /tmp/quick.json   # faster, coarser
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Ground truth calibration -- taken DIRECTLY from archived, already-published
# design-document numbers (sec 11.12.2's curve table; sec 12 task brief),
# not invented for this simulation:
#   K/d=0.25 (K=16): h4 ~= 1.00 (saturated, treated as a fixed anchor, not
#     simulated with per-seed noise -- no archived seed-level h4 array
#     exists for K=16 candidate (d) at this design's own citation depth;
#     using the point as a fixed boundary condition is conservative --
#     it does not affect the x0/w estimate in the (0.5,0.75) interior,
#     which is what this wave actually needs to resolve).
#   K/d=0.50 (K=32): h4 seeds [0.674072265625, 0.7125244140625,
#     0.61407470703125] -- verified this session directly from
#     experiment-runs/2026-07-06_keyanchor_mech/wavekeyanchor-mech/
#     wkeyanchor-mech_rdx_K32_armd_s1{0,1,2}_*.json,
#     checkpoints[-1]['M3_held_out']['4']['recovered_frac@0.9'].
#   K/d=0.75 (K=48): h4 seeds [0.02294921875, 0.0228678397834301,
#     0.0187174491584301] -- verified this session directly from
#     experiment-runs/2026-07-07_keyanchor_k48/wavekeyanchor-k48/
#     wkeyanchor-k48_rdx_K48_armd_s3{0,1,2}_*.json, same JSON path.
# ---------------------------------------------------------------------------

D_STATE = 64
K32_H4_SEEDS = np.array([0.674072265625, 0.7125244140625, 0.61407470703125])
K48_H4_SEEDS = np.array([0.02294921875, 0.0228678397834301, 0.0187174491584301])

K32_MEAN, K32_SD = float(K32_H4_SEEDS.mean()), float(K32_H4_SEEDS.std(ddof=1))
K48_MEAN, K48_SD = float(K48_H4_SEEDS.mean()), float(K48_H4_SEEDS.std(ddof=1))
# Relative (coefficient-of-variation) noise -- the quantity that transfers to
# a NEW K, since absolute sd scales with the local mean (a mean near 0 or 1
# necessarily has small absolute sd; relative noise is the portable number).
NOISE_REL_SD = {32: K32_SD / K32_MEAN, 48: K48_SD / K48_MEAN}
# Single pooled relative-noise figure used for K values we have no direct
# calibration at (36/38/40/42/44/46) -- the mean of the two measured points,
# ~9.3%, matches the task brief's own "~8% relative" characterization within
# rounding. Disclosed as an interpolation, not a measurement, exactly like
# sec 12.2's own K-cost interpolation is disclosed as approximate.
POOLED_REL_SD = float(np.mean(list(NOISE_REL_SD.values())))  # ~0.0934

ANCHOR_X = {16: 16.0 / D_STATE, 32: 32.0 / D_STATE, 48: 48.0 / D_STATE}
ANCHOR_H4 = {16: 1.0, 32: K32_MEAN, 48: K48_MEAN}


def sigmoid(x, L, x0, w):
    """Logistic collapse, sec 12.4's own registered fit form."""
    return L / (1.0 + np.exp((x - x0) / w))


# ---------------------------------------------------------------------------
# Ground-truth curves to simulate FROM. We do not know the true (x0, w) --
# that is the whole question -- so we sweep a small grid of plausible truths
# anchored to the only real information available: the archived K=32/K=48
# means straddle the interval, and sec 12.7 attack-round-1's own reproduction
# already characterized w=0.03 and w=0.08 as the two informative extremes
# (sharp cliff vs. cliff that's actually fairly wide relative to (0.5,0.75)).
# We also fit L to pass through (0.25, ~1.0) and add a THIRD truth at the
# noiseless best-fit of the 3 archived anchor points themselves (the
# attacker's own "residual RSS ~= 0.0076" fit), so the sim is not restricted
# to hand-picked round numbers.
# ---------------------------------------------------------------------------


def noiseless_anchor_fit():
    """Reproduce the attacker's 3-point noiseless sigmoid fit (16/32/48 means
    only) -- this is what "residual RSS ~= 0.0076" refers to; recomputed here
    independently rather than taken on faith."""
    xs = np.array([ANCHOR_X[16], ANCHOR_X[32], ANCHOR_X[48]])
    ys = np.array([ANCHOR_H4[16], ANCHOR_H4[32], ANCHOR_H4[48]])
    p0 = [1.0, 0.6, 0.05]
    try:
        popt, _ = curve_fit(sigmoid, xs, ys, p0=p0, maxfev=20000,
                             bounds=([0.5, 0.3, 0.005], [1.2, 0.9, 0.5]))
        resid = ys - sigmoid(xs, *popt)
        rss = float(np.sum(resid ** 2))
        return {"L": float(popt[0]), "x0": float(popt[1]), "w": float(popt[2]), "rss": rss}
    except Exception as e:  # noqa: BLE001 -- diagnostic path only, sim continues with grid truths
        return {"error": str(e)}


TRUTH_GRID = [
    {"name": "sharp_w03", "L": 1.0, "x0": 0.625, "w": 0.03},
    {"name": "sharp_w05", "L": 1.0, "x0": 0.625, "w": 0.05},
    {"name": "moderate_w08", "L": 1.0, "x0": 0.625, "w": 0.08},
    {"name": "gradual_w15", "L": 1.0, "x0": 0.625, "w": 0.15},
    # Off-center truths, added at Rev 12.2 fix 2 (round-2 attack finding
    # R2-5): all four truths above are centered at the bracket midpoint
    # x0=0.625, so the 0%-degenerate-fit result reported in Rev 12.1 was
    # only ever observed there -- a truth whose x0 sits near one of the
    # new point set's own extremes (0.55, close to K=34's 0.53125; 0.70,
    # close to K=46's 0.71875) was never swept. Both widths (w=0.03 sharp,
    # w=0.08 moderate) are checked at each off-center location, cheap
    # (CPU-only, same n_trials) and directly answers whether the fit
    # degenerate-frac/CI-width numbers hold up away from the midpoint.
    {"name": "offcenter_x055_w03", "L": 1.0, "x0": 0.55, "w": 0.03},
    {"name": "offcenter_x055_w08", "L": 1.0, "x0": 0.55, "w": 0.08},
    {"name": "offcenter_x070_w03", "L": 1.0, "x0": 0.70, "w": 0.03},
    {"name": "offcenter_x070_w08", "L": 1.0, "x0": 0.70, "w": 0.08},
]

# Noise-level sweep, added at Rev 12.2 fix 2 (round-2 attack finding R2-2):
# the pooled 9.35% relative-noise figure (POOLED_REL_SD below) is the mean
# of exactly two measured points (K=32, K=48) applied uniformly to four
# UN-measured K's -- a point estimate doing silent work. Multiplying it by
# a small bracket checks whether the CI-width/degenerate-frac conclusions
# are sensitive to the true noise at the new K's being higher (K=46 sits
# closer to the noisier K=48 anchor) or lower than the pooled estimate.
NOISE_SD_MULTIPLIERS = [0.85, 1.0, 1.15]


# ---------------------------------------------------------------------------
# Candidate new-point sets. The task brief's own suggestion is re-picked
# for identifiability: K in {34,38,42,46} (K/d = 0.53125/0.59375/0.65625/
# 0.71875) instead of the Rev-12.0 design's {36,40,44}, because 4 points
# spread WIDER (closer to both the 0.5 and 0.75 archived anchors) constrain
# the logistic's x0 and w with one more degree of freedom than 3 evenly-
# spaced interior points. Compared against the original {36,40,44} set and
# a 5-point superset, all under the SAME per-K seed budget, so the marginal
# value of the extra point (and the extra GPU-h it costs) is visible
# directly in the CI-width numbers below, not asserted.
# ---------------------------------------------------------------------------

POINT_SETS = {
    "orig_3pt_36_40_44": [36, 40, 44],
    "repick_4pt_34_38_42_46": [34, 38, 42, 46],
    "repick_5pt_34_37_40_43_46": [34, 37, 40, 43, 46],
}

SEED_COUNTS = [3, 4, 5]


def simulate_once(rng, new_ks, n_seeds_new, truth, include_anchors=True, noise_sd_multiplier=1.0,
                   d_state=D_STATE):
    """One synthetic replicate: draw noisy per-seed h4 at each K (archived
    K=32/K=48 anchors at their OWN archived seed count of 3, using their
    OWN measured relative noise; new K's at n_seeds_new seeds using the
    pooled relative-noise estimate), fit the sigmoid + linear forms to the
    per-K MEANS (sec 12.4's own "mean, always" convention, inherited), and
    return the fitted x0 (or None if the fit fails to converge).

    noise_sd_multiplier (Rev 12.2 fix 2, R2-2): scales BOTH the archived
    anchor relative noise and the pooled new-K relative noise uniformly,
    so the whole noise MODEL shifts together rather than only the
    un-measured K's -- a check of whether the pooled 9.35% point estimate
    is doing silent work, not a claim that the archived anchors themselves
    are mis-measured.

    d_state (sec 13.3 item 8, Rev 13.2, cliff-universality-across-d_state
    build): the d used to convert new_ks into K/d ratios. Default 64
    (byte-identical to the pre-generalization behavior). include_anchors is
    ALREADY the mechanism for an anchor-free simulation (this function's
    pre-existing parameter, unused by any caller before this build) -- the
    d=128 registered power sim (sec 13.3 item 8's own required pre-launch
    task) calls this with include_anchors=False, d_state=128, since NO
    archived K=16/32/48-equivalent flanking point exists at d=128 (sec
    13.2's own disclosure: 'the d=128 fit is a pure 4-point curve... with
    NO flanking anchors on either side of the transition')."""
    xs, ys = [], []

    if include_anchors:
        # K=16 point: treated as noiseless (saturated, no per-seed archive
        # exists at this design's citation depth) -- conservative, does not
        # inflate CI width artificially since it sits far from the interior.
        xs.append(ANCHOR_X[16]); ys.append(ANCHOR_H4[16])
        for k, rel_sd in ((32, NOISE_REL_SD[32]), (48, NOISE_REL_SD[48])):
            true_mean = sigmoid(ANCHOR_X[k], truth["L"], truth["x0"], truth["w"])
            seeds = rng.normal(true_mean, rel_sd * noise_sd_multiplier * max(true_mean, 1e-6), size=3)
            seeds = np.clip(seeds, 0.0, 1.0)
            xs.append(ANCHOR_X[k]); ys.append(float(seeds.mean()))

    for k in new_ks:
        x = k / d_state
        true_mean = sigmoid(x, truth["L"], truth["x0"], truth["w"])
        seeds = rng.normal(true_mean, POOLED_REL_SD * noise_sd_multiplier * max(true_mean, 1e-6), size=n_seeds_new)
        seeds = np.clip(seeds, 0.0, 1.0)
        xs.append(x); ys.append(float(seeds.mean()))

    xs = np.array(xs); ys = np.array(ys)
    try:
        popt, pcov = curve_fit(
            sigmoid, xs, ys,
            p0=[1.0, 0.625, 0.08],
            bounds=([0.5, 0.3, 0.005], [1.2, 0.9, 0.5]),
            maxfev=20000,
        )
        if not np.all(np.isfinite(pcov)):
            return None
        return {"L": float(popt[0]), "x0": float(popt[1]), "w": float(popt[2])}
    except Exception:  # noqa: BLE001 -- degenerate fit, counted as a miss by the caller
        return None


def bootstrap_ci_width(rng, new_ks, n_seeds_new, truth, n_trials, ci=0.95, noise_sd_multiplier=1.0,
                        include_anchors=True, d_state=D_STATE):
    """Seed-level parametric bootstrap (finding 1a's registered method):
    repeat the noisy-draw-and-fit procedure n_trials times, report the
    (ci*100)% interval width of the resulting x0 distribution, plus the
    fraction of trials where the fit degenerated (did not converge /
    hit a bound) -- itself a required disclosure (finding 1a: "what
    counts as a degenerate fit" must be pre-registered).

    include_anchors/d_state (sec 13.3 item 8, Rev 13.2): threaded straight
    through to simulate_once -- default include_anchors=True, d_state=64
    reproduces the pre-generalization behavior byte-identically."""
    x0s = []
    n_degenerate = 0
    for _ in range(n_trials):
        fit = simulate_once(rng, new_ks, n_seeds_new, truth, include_anchors=include_anchors,
                             noise_sd_multiplier=noise_sd_multiplier, d_state=d_state)
        if fit is None:
            n_degenerate += 1
            continue
        # Degenerate-fit guard, pre-registered: a fit that pins to a bound
        # (w within 1% of the [0.005, 0.5] box, or x0 within 1% of [0.3,0.9])
        # is counted as degenerate even though curve_fit "succeeded" --
        # otherwise boundary-pinned fits would silently narrow the reported
        # CI by excluding exactly the runs where the data couldn't
        # constrain the parameter.
        if fit["w"] <= 0.005 * 1.01 or fit["w"] >= 0.5 * 0.99:
            n_degenerate += 1
            continue
        if fit["x0"] <= 0.3 * 1.01 or fit["x0"] >= 0.9 * 0.99:
            n_degenerate += 1
            continue
        x0s.append(fit["x0"])
    if len(x0s) < max(20, 0.05 * n_trials):
        return {"ci_width": None, "n_degenerate": n_degenerate, "n_trials": n_trials,
                "degenerate_frac": n_degenerate / n_trials, "note": "too few valid fits to report a CI"}
    x0s = np.array(x0s)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    lo, hi = np.quantile(x0s, [lo_q, hi_q])
    return {
        "ci_width": float(hi - lo),
        "ci_lo": float(lo), "ci_hi": float(hi),
        "median_x0": float(np.median(x0s)),
        "n_degenerate": n_degenerate, "n_trials": n_trials,
        "degenerate_frac": n_degenerate / n_trials,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=4000,
                     help="bootstrap replicates per (point-set, seed-count, truth) cell")
    ap.add_argument("--seed", type=int, default=20260712)
    ap.add_argument("--out", default="sim_cliff_power_results.json")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    anchor_fit = noiseless_anchor_fit()

    # Primary grid: all point sets x all seed counts x the full TRUTH_GRID
    # (now including the off-center truths, Rev 12.2 fix 2), at the
    # registered noise level (multiplier=1.0). This is the grid the
    # point-set/seed-count decision (sec 12.4a) is made from -- unchanged
    # in spirit from Rev 12.1, extended with the off-center truths.
    results = []
    for set_name, ks in POINT_SETS.items():
        for n_seeds in SEED_COUNTS:
            for truth in TRUTH_GRID:
                r = bootstrap_ci_width(rng, ks, n_seeds, truth, args.n_trials, noise_sd_multiplier=1.0)
                cells = len(ks) * n_seeds  # candidate (d) only -- reference arm is CUT from this
                # wave's mandatory scope (sec 12.2 item 2, Rev 12.1); previously mirrored
                # "candidate (d) + reference arm" (x2) but that arm was cut in the same
                # revision that introduced this script, leaving the x2 stale (Rev 12.2 fix 1).
                results.append({
                    "point_set": set_name, "ks": ks, "n_seeds_new": n_seeds,
                    "truth": truth["name"], "true_x0": truth["x0"], "true_w": truth["w"],
                    "noise_sd_multiplier": 1.0,
                    "mandatory_cells": cells,
                    **r,
                })

    # Noise-sensitivity sub-sweep (Rev 12.2 fix 2, round-2 attack finding
    # R2-2): restricted to the REGISTERED configuration only (re-picked
    # 4pt set, 3 seeds/K -- sec 12.2's actual choice), swept across
    # NOISE_SD_MULTIPLIER in {0.85, 1.0, 1.15} and every truth in the full
    # grid (including the off-center ones). Kept as a separate, smaller
    # sweep rather than folded into the primary grid above because the
    # question here is "how sensitive is the REGISTERED config to the
    # noise assumption", not "which config to pick" (already answered by
    # the primary grid) -- full cross product of all 3 point sets x all 3
    # seed counts x all 8 truths x all 3 multipliers would be 216 cells at
    # n_trials=4000 each, needlessly slow for a question that is only
    # about the one config this wave will actually run.
    REGISTERED_SET = "repick_4pt_34_38_42_46"
    REGISTERED_SEEDS = 3
    noise_sensitivity = []
    for mult in NOISE_SD_MULTIPLIERS:
        for truth in TRUTH_GRID:
            r = bootstrap_ci_width(
                rng, POINT_SETS[REGISTERED_SET], REGISTERED_SEEDS, truth,
                args.n_trials, noise_sd_multiplier=mult,
            )
            noise_sensitivity.append({
                "point_set": REGISTERED_SET, "n_seeds_new": REGISTERED_SEEDS,
                "truth": truth["name"], "true_x0": truth["x0"], "true_w": truth["w"],
                "noise_sd_multiplier": mult,
                "mandatory_cells": len(POINT_SETS[REGISTERED_SET]) * REGISTERED_SEEDS,
                **r,
            })

    # Best configuration per point set (averaging CI width across the
    # full truth grid, since the true w/x0 is exactly what's unknown
    # pre-registration -- a config should be chosen for robustness across
    # plausible truths, not tuned to whichever truth happens to look
    # best). Uses the primary grid (multiplier=1.0) only.
    by_set = {}
    for set_name, ks in POINT_SETS.items():
        for n_seeds in SEED_COUNTS:
            widths = [r["ci_width"] for r in results
                      if r["point_set"] == set_name and r["n_seeds_new"] == n_seeds
                      and r["ci_width"] is not None]
            if not widths:
                continue
            by_set.setdefault(set_name, []).append({
                "n_seeds_new": n_seeds,
                "mean_ci_width_across_truths": float(np.mean(widths)),
                "max_ci_width_across_truths": float(np.max(widths)),
                "mandatory_cells": len(ks) * n_seeds,  # candidate (d) only, see fix at cell computation above
            })

    # Degenerate-fit fractions and CI widths per cell, isolated at the two
    # boundary truths (x0=0.55, x0=0.70) and the pessimistic noise corner
    # (multiplier=1.15, the gradual_w15 truth -- the widest-CI, highest-
    # noise combination in the grid), for the registered config only --
    # this is the table sec 12.4a/12.4b's WITHDRAWN check reads directly.
    boundary_truth_names = {"offcenter_x055_w03", "offcenter_x055_w08",
                             "offcenter_x070_w03", "offcenter_x070_w08"}
    boundary_report = [r for r in results
                        if r["point_set"] == REGISTERED_SET and r["n_seeds_new"] == REGISTERED_SEEDS
                        and r["truth"] in boundary_truth_names]
    pessimistic_corner = [r for r in noise_sensitivity
                           if r["truth"] == "gradual_w15" and r["noise_sd_multiplier"] == 1.15]

    # ---------------------------------------------------------------------
    # KEY_ANCHORING_DESIGN.md sec 13.3 item 8 (Rev 13.2, cliff-universality-
    # across-d_state build) -- the REGISTERED, MANDATORY pre-launch d=128
    # anchor-free power sim. Unlike every sweep above (which is anchored --
    # K=16 fixed + K=32/K=48 archived, at d=64), the d=128 fit has NO
    # flanking anchor on either side of the transition (sec 13.2's own
    # disclosure) -- so both the FULL 4-point grid (K=68,76,84,92) and the
    # sec 13.6 Option-C 2-point grid (K=76,84) are simulated with
    # include_anchors=False, d_state=128, using the SAME pooled 9.35%
    # relative-noise estimate this file already measured off the d=64
    # archives (sec 13.3 item 8's own registered instruction: "with the
    # d=64-measured noise (pooled 9.35%, multiplier sweep)" -- no d=128
    # noise measurement exists yet, so the d=64 estimate is reused AS THE
    # BEST AVAILABLE PROXY, disclosed as such, not silently assumed
    # identical). Swept across the SAME noise-multiplier bracket
    # (NOISE_SD_MULTIPLIERS) and the SAME TRUTH_GRID as the d=64 sweeps
    # above, so the two are directly comparable.
    # ---------------------------------------------------------------------
    D_STATE_128 = 128
    DSTATE_FULL_4PT_KS = [68, 76, 84, 92]     # sec 13.2's full K-grid
    DSTATE_OPTION_C_2PT_KS = [76, 84]         # sec 13.6 Option C's own registered 2-point choice
    DSTATE_SEED_COUNTS = [3]                  # sec 13.2's registered seed count (3/K), not swept here
    dstate_results = []
    for point_set_name, ks in (("dstate_full_4pt_68_76_84_92", DSTATE_FULL_4PT_KS),
                                ("dstate_optionC_2pt_76_84", DSTATE_OPTION_C_2PT_KS)):
        for n_seeds in DSTATE_SEED_COUNTS:
            for mult in NOISE_SD_MULTIPLIERS:
                for truth in TRUTH_GRID:
                    r = bootstrap_ci_width(rng, ks, n_seeds, truth, args.n_trials,
                                            noise_sd_multiplier=mult, include_anchors=False,
                                            d_state=D_STATE_128)
                    dstate_results.append({
                        "point_set": point_set_name, "ks": ks, "n_seeds_new": n_seeds,
                        "d_state": D_STATE_128, "anchored": False,
                        "truth": truth["name"], "true_x0": truth["x0"], "true_w": truth["w"],
                        "noise_sd_multiplier": mult,
                        "mandatory_cells": len(ks) * n_seeds,
                        **r,
                    })
    # Summary at the REGISTERED noise multiplier (1.0) only, for the direct
    # "is Option C viable" read sec 13.6 needs -- averaged/maxed across the
    # full truth grid, same discipline as summary_by_point_set above.
    dstate_summary = {}
    for point_set_name, ks in (("dstate_full_4pt_68_76_84_92", DSTATE_FULL_4PT_KS),
                                ("dstate_optionC_2pt_76_84", DSTATE_OPTION_C_2PT_KS)):
        widths = [r["ci_width"] for r in dstate_results
                  if r["point_set"] == point_set_name and r["noise_sd_multiplier"] == 1.0
                  and r["ci_width"] is not None]
        degenerate_fracs = [r["degenerate_frac"] for r in dstate_results
                             if r["point_set"] == point_set_name and r["noise_sd_multiplier"] == 1.0]
        dstate_summary[point_set_name] = {
            "n_ks": len(ks), "ks": ks,
            "mean_ci_width_across_truths": float(np.mean(widths)) if widths else None,
            "max_ci_width_across_truths": float(np.max(widths)) if widths else None,
            "max_degenerate_frac_across_truths": float(np.max(degenerate_fracs)) if degenerate_fracs else None,
            "any_degenerate_frac_exceeds_10pct": bool(any(d > 0.10 for d in degenerate_fracs)),
        }

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "sim_cliff_power.py",
        "purpose": "KEY_ANCHORING_DESIGN.md sec 12.4/12.7 finding-1 power simulation, CPU-only "
                   "(Rev 12.2: extended with off-center truths + noise-multiplier sweep, "
                   "round-2 attack findings R2-2/R2-5)",
        "d_state": D_STATE,
        "archived_anchor_seeds": {
            "K32_h4": K32_H4_SEEDS.tolist(), "K48_h4": K48_H4_SEEDS.tolist(),
        },
        "measured_relative_noise": NOISE_REL_SD,
        "pooled_relative_noise_used_for_new_Ks": POOLED_REL_SD,
        "noise_sd_multipliers_swept": NOISE_SD_MULTIPLIERS,
        "noiseless_3point_anchor_fit": anchor_fit,
        "truth_grid": TRUTH_GRID,
        "point_sets": POINT_SETS,
        "seed_counts_swept": SEED_COUNTS,
        "n_trials_per_cell": args.n_trials,
        "results_full_grid": results,
        "summary_by_point_set": by_set,
        "noise_sensitivity_registered_config": noise_sensitivity,
        "boundary_truth_report_registered_config": boundary_report,
        "pessimistic_corner_registered_config": pessimistic_corner,
        # sec 13.3 item 8 (Rev 13.2): the d=128 anchor-free power sim --
        # full 4-point grid (K=68,76,84,92) AND the sec 13.6 Option-C
        # 2-point grid (K=76,84), both include_anchors=False, d_state=128,
        # reusing this file's own d=64-measured pooled relative noise
        # (POOLED_REL_SD) as the best available proxy (no d=128 noise
        # measurement exists).
        "dstate_d128_anchor_free": {
            "d_state": D_STATE_128,
            "point_sets": {"dstate_full_4pt_68_76_84_92": DSTATE_FULL_4PT_KS,
                            "dstate_optionC_2pt_76_84": DSTATE_OPTION_C_2PT_KS},
            "seed_counts_swept": DSTATE_SEED_COUNTS,
            "noise_proxy_disclosure": "reuses this file's own d=64-measured POOLED_REL_SD "
                                       f"({POOLED_REL_SD:.4f}), swept by NOISE_SD_MULTIPLIERS -- no "
                                       "d=128 noise measurement exists yet; this is the best "
                                       "available proxy, disclosed, not assumed identical.",
            "results_full_grid": dstate_results,
            "summary_by_point_set": dstate_summary,
        },
        "wall_s": time.time() - t0,
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.out} ({time.time()-t0:.1f}s)")
    print(f"Noiseless 3-anchor-point fit: {anchor_fit}")
    print(f"Pooled relative noise used for new Ks: {POOLED_REL_SD:.4f}")
    print()
    print(f"{'point_set':30s} {'n_seeds':>7s} {'cells':>6s} {'mean CI(x0)':>12s} {'max CI(x0)':>11s}")
    for set_name, rows in by_set.items():
        for row in rows:
            print(f"{set_name:30s} {row['n_seeds_new']:7d} {row['mandatory_cells']:6d} "
                  f"{row['mean_ci_width_across_truths']:12.4f} {row['max_ci_width_across_truths']:11.4f}")

    print()
    print("Boundary-truth report (registered config, repick_4pt, 3 seeds/K, multiplier=1.0):")
    for r in boundary_report:
        print(f"  {r['truth']:22s} ci_width={r['ci_width']} degenerate_frac={r['degenerate_frac']:.4f}")

    print()
    print("Pessimistic noise corner (gradual_w15 truth, multiplier=1.15):")
    for r in pessimistic_corner:
        print(f"  ci_width={r['ci_width']} degenerate_frac={r['degenerate_frac']:.4f}")

    print()
    print("sec 13.3 item 8 (Rev 13.2): d=128 ANCHOR-FREE power sim (K/d matched to sec 12.2, "
          "d=64-measured noise reused as proxy) -- registered noise multiplier=1.0 summary:")
    for point_set_name, s in dstate_summary.items():
        print(f"  {point_set_name:32s} n_ks={s['n_ks']} ks={s['ks']} "
              f"mean_ci_width={s['mean_ci_width_across_truths']} "
              f"max_ci_width={s['max_ci_width_across_truths']} "
              f"max_degenerate_frac={s['max_degenerate_frac_across_truths']} "
              f"({'EXCEEDS' if s['any_degenerate_frac_exceeds_10pct'] else 'within'} the 10% bar "
              f"at some truth)")


if __name__ == "__main__":
    main()
