"""fit_cliff_curve.py -- KEY_ANCHORING_DESIGN.md sec 12.4 (Rev 12.1/12.2)'s
pre-registered real-data readout for the capacity-cliff localization wave.

Written and committed BEFORE any K=34/38/42/46 GPU cell exists (sec 12.4
item 5: "the fitting script... is written and committed BEFORE any
K=34/38/42/46 GPU cell launches, exactly as rev7_threshold_derive.py and
bands_pinned_trackb.py were written before their own respective waves' data
existed"). REUSES sim_cliff_power.py's own sigmoid()/bootstrap_ci_width()
directly (imported, not re-implemented) so the pre-registered simulation
(sec 12.4a) and this real-data fit can never silently diverge in fit form,
bounds, or bootstrap procedure (sec 12.4 item 5's own registered
requirement).

Deliverable (sec 12.4): fit `h4(x) = L / (1 + exp((x-x0)/w))` to the full
6-point curve (K=16 fixed anchor + K=32/K=48 archived 3-seed means + this
wave's own K=34/38/42/46 3-seed means), report `x0 +/- CI` and `w <= C`
(95% seed-level parametric bootstrap CI), the degenerate-fit fraction (sec
12.4 item 3's pre-registered definition), and the linear-fit RSS as a
disclosed, non-load-bearing secondary comparison (sec 12.4 item 1).

ZERO-REFERENCE-ARM-PATHS assertion (sec 12.4 item 5 / sec 12.2.1's second
Wave -1 smoke): this script's own input file list is asserted to contain no
reference-arm result path before any JSON is opened -- trivially satisfied
since the reference arm is cut from this wave's scope (sec 12.2 item 2), but
checked mechanically anyway as a guard against a future accidental
re-introduction (smoke_keyanchor_cliff.py's own negative unit test feeds
this exact assertion a reference-arm path and confirms it raises).

Usage:
    python fit_cliff_curve.py --cliff-out-dir results/deltanet_rd_exactness/wavekeyanchor-cliff \\
        --k32-dir experiment-runs/2026-07-06_keyanchor_mech/wavekeyanchor-mech \\
        --k48-dir experiment-runs/2026-07-07_keyanchor_k48/wavekeyanchor-k48 \\
        --out fit_cliff_curve_results.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports

# REUSED directly (sec 12.4 item 5) -- never re-implemented, so the
# pre-registered power simulation (sec 12.4a) and this real-data readout
# share the identical fit form/bounds/bootstrap procedure. `sigmoid`/
# `bootstrap_ci_width` are d-agnostic pure functions (no D_STATE baked into
# their own bodies); D_STATE itself is imported ONLY as the historical d=64
# default below (sec 13.3 items 4/8, cliff-universality-across-d_state
# build) -- callers that need a DIFFERENT d (e.g. 128) pass --d-state,
# which this script now threads through everywhere D_STATE was previously
# read as a bare module constant.
from sim_cliff_power import sigmoid, bootstrap_ci_width, D_STATE as _SIM_D_STATE_DEFAULT   # noqa: E402

D_STATE = _SIM_D_STATE_DEFAULT   # kept as a module-level name for backward compat (any external
                                  # reader of fit_cliff_curve.D_STATE still sees 64 unless main()
                                  # itself is invoked with --d-state, which operates on LOCAL args)

ANCHOR_X16 = 16.0 / D_STATE
ANCHOR_H4_16 = 1.0   # sec 12.4 item 1: K=16, fixed anchor, no per-seed archive at this citation depth

# The four candidate-(d)-only cells the d=64 cliff wave's own manifest
# registers (sec 12.2 item 2: reference arms are CUT, so 'ref'/'georef'-
# tagged wave-1-style paths must NEVER appear in this script's own input
# list). Sec 13.3 item 8 (Rev 13.2): this is the D=64-SPECIFIC K-grid --
# other d values (e.g. d=128's K in {68,76,84,92}) pass their own --k-grid,
# never silently reusing this tuple.
CLIFF_KS = (34, 38, 42, 46)

# sec 13.3 item 8's own registered per-d anchor availability: ONLY d=64 has
# archived K=16 (fixed)/K=32/K=48 flanking points (sec 12.9's own 6-point
# curve). At any OTHER d (currently only d=128, sec 13.2), NO archived
# flanking anchor exists on either side of the transition -- the fit must
# degrade gracefully to a bare N-point (or, under sec 13.6 Option C, 2-point)
# fit with ZERO anchors, never silently reuse the d=64 anchor VALUES (which
# are themselves d=64-specific h4 numbers, meaningless at a different d) or
# crash on a missing k32_seeds/k48_seeds lookup for a d that has none.
ANCHORED_D_STATES = (64,)

# Linear-fit form (sec 12.4 item 1: "retained as a DISCLOSED SECONDARY
# comparison... never load-bearing").
def _linear(x, a, b):
    return a * x + b


class ReferenceArmPathError(ValueError):
    """Raised by assert_no_reference_arm_paths when a path in the input
    list looks like a reference-arm result (sec 12.4 item 5's negative
    unit test target)."""


def assert_no_reference_arm_paths(paths: list) -> None:
    """sec 12.2.1's second Wave -1 smoke / sec 12.4 item 5's registered
    negative unit test: the fitting script's own input file list must
    contain ZERO reference-arm result paths. Reference-arm cells are built
    exclusively by reference_arms_manifest()/keyanchor_k48_reference_
    manifest() (run_deltanet_rd_exactness_sweep.py), whose wave tag is
    always 'ref' and whose arm tag is always 'georef' -- _spec()'s own
    naming convention gives every such filename the literal substring
    'wref_rdx_' at its start (out_path()'s own f"{spec['name']}.json",
    spec['name'] built from bits=[f"w{wave}_rdx", ...] with wave=='ref').
    Checked on the PATH's basename (never a substring match against the
    full path, which could false-positive on a directory literally named
    'ref' for an unrelated reason) -- exact, not fuzzy."""
    for p in paths:
        base = os.path.basename(p)
        if base.startswith("wref_rdx_") or "_armgeoref_" in base:
            raise ReferenceArmPathError(
                f"reference-arm result path found in fit input list: {p!r} -- sec 12.2 item 2 cut "
                f"the reference arm from this wave's scope; sec 12.4's fit reads candidate-(d) "
                f"values ONLY. This assertion exists to catch a future accidental re-introduction, "
                f"not because one is expected now.")


def load_k_mean_h4(cell_dir: str, K: int, arm: str = "d") -> tuple[float, list]:
    """Read every completed candidate-(d) result JSON for this K in
    cell_dir, extract h4 rec@0.9 from each, return (mean, per_seed_list).
    Glob pattern matches _spec()'s own naming convention: any file whose
    basename contains f"_K{K}_arm{arm}_" (armd, never armdprime/armd-fixed-
    lam1/arme/arme-fp -- the trailing underscore after arm's own bit
    excludes those, since _spec()'s bits join with '_' and 'd' is a
    strict-prefix substring of 'dprime'/'d-fixed-lam1')."""
    pattern = os.path.join(cell_dir, f"*_K{K}_arm{arm}_*.json")
    paths = sorted(glob.glob(pattern))
    assert_no_reference_arm_paths(paths)
    per_seed = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if not d.get("complete"):
            continue
        try:
            h4 = d["checkpoints"][-1]["M3_held_out"]["4"]["recovered_frac@0.9"]
        except (KeyError, IndexError, TypeError):
            continue
        per_seed.append(h4)
    if not per_seed:
        return None, []
    return float(np.mean(per_seed)), per_seed


def fit_sigmoid(xs: np.ndarray, ys: np.ndarray):
    """sec 12.4 item 1's registered form/bounds -- IDENTICAL to
    sim_cliff_power.simulate_once's own curve_fit call (same p0/bounds/
    maxfev), so the real fit and the power sim can never silently diverge."""
    popt, pcov = curve_fit(
        sigmoid, xs, ys, p0=[1.0, 0.625, 0.08],
        bounds=([0.5, 0.3, 0.005], [1.2, 0.9, 0.5]), maxfev=20000,
    )
    resid = ys - sigmoid(xs, *popt)
    rss = float(np.sum(resid ** 2))
    return {"L": float(popt[0]), "x0": float(popt[1]), "w": float(popt[2]), "rss": rss}


def fit_linear(xs: np.ndarray, ys: np.ndarray):
    """sec 12.4 item 1's disclosed, non-load-bearing secondary comparison."""
    popt, _ = curve_fit(_linear, xs, ys, maxfev=20000)
    resid = ys - _linear(xs, *popt)
    rss = float(np.sum(resid ** 2))
    return {"a": float(popt[0]), "b": float(popt[1]), "rss": rss}


def real_bootstrap_ci(per_seed_by_k: dict, k32_seeds: np.ndarray | None, k48_seeds: np.ndarray | None,
                       n_trials: int, d_state: int = D_STATE, ci: float = 0.95, seed: int = 0):
    """sec 12.4 item 2's pre-registered CI method, on REAL bootstrap
    resamples (not the simulation's synthetic draws): for each of n_trials
    replicates, resample 3 seeds WITH replacement from each K's own
    ARCHIVED/REAL array (K=32, K=48, and each of the wave's own
    K=34/38/42/46), take the mean, hold K=16 fixed (no per-seed archive),
    re-fit the sigmoid, record (x0, w). Same degenerate-fit definition as
    sim_cliff_power.bootstrap_ci_width (fit fails to converge, OR w/x0 lands
    within 1% of a bound) -- NOT a re-implementation of that function
    (which only knows how to draw from a synthetic ground-truth curve); this
    is the real-data analogue of the identical procedure.

    sec 13.3 item 8 (Rev 13.2, cliff-universality-across-d_state build):
    generalized to a `d_state` parameter and OPTIONAL k32_seeds/k48_seeds
    (None at any d NOT in ANCHORED_D_STATES, e.g. d=128 -- sec 13.2's own
    disclosure that the d=128 fit is a PURE new-K-only curve, no flanking
    anchors on either side of the transition). Default d_state=64 with
    k32_seeds/k48_seeds both given reproduces the pre-generalization
    anchored behavior byte-identically (sec 13.3 item 6's own d=64
    regression-smoke discipline, applied here too)."""
    anchored = k32_seeds is not None and k48_seeds is not None
    rng = np.random.default_rng(seed)
    ks_new = sorted(per_seed_by_k.keys())
    x0s, ws = [], []
    n_degenerate = 0
    for _ in range(n_trials):
        if anchored:
            xs, ys = [16.0 / d_state], [ANCHOR_H4_16]
            k32_draw = rng.choice(k32_seeds, size=len(k32_seeds), replace=True)
            xs.append(32.0 / d_state); ys.append(float(k32_draw.mean()))
            k48_draw = rng.choice(k48_seeds, size=len(k48_seeds), replace=True)
            xs.append(48.0 / d_state); ys.append(float(k48_draw.mean()))
        else:
            xs, ys = [], []
        for K in ks_new:
            arr = np.array(per_seed_by_k[K])
            draw = rng.choice(arr, size=len(arr), replace=True)
            xs.append(K / d_state); ys.append(float(draw.mean()))
        xs_a, ys_a = np.array(xs), np.array(ys)
        try:
            popt, pcov = curve_fit(
                sigmoid, xs_a, ys_a, p0=[1.0, 0.625, 0.08],
                bounds=([0.5, 0.3, 0.005], [1.2, 0.9, 0.5]), maxfev=20000,
            )
            if not np.all(np.isfinite(pcov)):
                n_degenerate += 1
                continue
        except Exception:
            n_degenerate += 1
            continue
        L, x0, w = float(popt[0]), float(popt[1]), float(popt[2])
        # sec 12.4 item 3's pre-registered degenerate-fit definition: fitted
        # w or x0 within 1% of either bound.
        if w <= 0.005 * 1.01 or w >= 0.5 * 0.99:
            n_degenerate += 1
            continue
        if x0 <= 0.3 * 1.01 or x0 >= 0.9 * 0.99:
            n_degenerate += 1
            continue
        x0s.append(x0); ws.append(w)
    n_valid = len(x0s)
    degenerate_frac = n_degenerate / n_trials
    out = {"n_trials": n_trials, "n_degenerate": n_degenerate, "degenerate_frac": degenerate_frac}
    if n_valid < max(20, 0.05 * n_trials):
        out.update({"ci_x0": None, "ci_w": None, "note": "too few valid fits to report a CI"})
        return out
    x0s_a, ws_a = np.array(x0s), np.array(ws)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    x0_lo, x0_hi = np.quantile(x0s_a, [lo_q, hi_q])
    w_lo, w_hi = np.quantile(ws_a, [lo_q, hi_q])
    out.update({
        "ci_x0": {"lo": float(x0_lo), "hi": float(x0_hi), "width": float(x0_hi - x0_lo),
                  "median": float(np.median(x0s_a))},
        "ci_w": {"lo": float(w_lo), "hi": float(w_hi), "width": float(w_hi - w_lo),
                 "median": float(np.median(ws_a))},
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cliff-out-dir", required=True,
                     help="wavekeyanchor-cliff/ (or wavekeyanchor-dstate/, sec 13) directory holding "
                          "the wave's own candidate-(d) result JSONs")
    ap.add_argument("--k32-dir", default=None,
                     help="directory holding the archived K=32 candidate-(d) 3-seed result JSONs "
                          "(Wave 1, sec 12's own citation: experiment-runs/2026-07-06_keyanchor_mech/"
                          "wavekeyanchor-mech). REQUIRED at --d-state 64 (the only ANCHORED_D_STATES "
                          "entry); omit entirely at any other d (sec 13.3 item 8: no archived K=32-"
                          "equivalent flanking point exists at d=128).")
    ap.add_argument("--k48-dir", default=None,
                     help="directory holding the archived K=48 candidate-(d) 3-seed result JSONs "
                          "(experiment-runs/2026-07-07_keyanchor_k48/wavekeyanchor-k48). Same "
                          "REQUIRED-at-d=64-only rule as --k32-dir.")
    ap.add_argument("--d-state", type=int, default=D_STATE,
                     help="sec 13.3 items 4/8: the d_state this fit's K/d ratio and anchor "
                          "availability are computed at. Default 64 (byte-identical to the "
                          "pre-generalization script when combined with --k32-dir/--k48-dir).")
    ap.add_argument("--k-grid", type=int, nargs="+", default=None,
                     help="sec 13.3 item 8: the wave's own new-K list (e.g. 68 76 84 92 at d=128). "
                          "Defaults to CLIFF_KS (34,38,42,46) ONLY at --d-state 64 -- REQUIRED "
                          "(no silent default) at any other d, since reusing the d=64 K-grid at a "
                          "different d would silently fit the wrong K/d ratios.")
    ap.add_argument("--n-trials", type=int, default=2000, help="bootstrap replicates, sec 12.4 item 2: >=2000")
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--out", default=os.path.join(HERE, "fit_cliff_curve_results.json"))
    args = ap.parse_args()
    assert args.n_trials >= 2000, f"sec 12.4 item 2 requires N_BOOT >= 2000, got {args.n_trials}"

    d_state = args.d_state
    anchored = d_state in ANCHORED_D_STATES
    if args.k_grid is not None:
        k_grid = tuple(args.k_grid)
    else:
        assert d_state == D_STATE, (
            f"--k-grid is REQUIRED at --d-state {d_state} (only --d-state {D_STATE} has a registered "
            f"default K-grid, CLIFF_KS={CLIFF_KS}) -- sec 13.3 item 8: reusing the d=64 K-grid at a "
            f"different d would silently fit the wrong K/d ratios.")
        k_grid = CLIFF_KS
    if anchored:
        assert args.k32_dir is not None and args.k48_dir is not None, (
            f"--d-state {d_state} is in ANCHORED_D_STATES={ANCHORED_D_STATES} -- --k32-dir/--k48-dir "
            f"are REQUIRED (the archived K=32/K=48 flanking anchors this d has).")
    else:
        assert args.k32_dir is None and args.k48_dir is None, (
            f"--d-state {d_state} is NOT in ANCHORED_D_STATES={ANCHORED_D_STATES} -- --k32-dir/"
            f"--k48-dir must be OMITTED (sec 13.3 item 8: no archived flanking anchor exists at this "
            f"d; passing them would silently fit d=64-specific h4 values as if they belonged to this "
            f"d's own curve).")

    t0 = time.time()

    if anchored:
        k32_mean, k32_seeds = load_k_mean_h4(args.k32_dir, 32, arm="d")
        k48_mean, k48_seeds = load_k_mean_h4(args.k48_dir, 48, arm="d")
    else:
        k32_mean, k32_seeds, k48_mean, k48_seeds = None, None, None, None
    per_k_mean, per_k_seeds = {}, {}
    for K in k_grid:
        mean, seeds = load_k_mean_h4(args.cliff_out_dir, K, arm="d")
        per_k_mean[K] = mean
        per_k_seeds[K] = seeds

    missing = [K for K, m in per_k_mean.items() if m is None]
    if (anchored and (k32_mean is None or k48_mean is None)) or missing:
        print(f"NOT READY: k32_mean={k32_mean} k48_mean={k48_mean} missing new-K means={missing} -- "
              f"cannot fit until every K has >=1 completed candidate-(d) cell.", file=sys.stderr)
        return 1

    if anchored:
        xs = [16.0 / d_state, 32.0 / d_state, 48.0 / d_state] + [K / d_state for K in k_grid]
        ys = [ANCHOR_H4_16, k32_mean, k48_mean] + [per_k_mean[K] for K in k_grid]
        labels = ["K16(anchor)", "K32(archived)", "K48(archived)"] + [f"K{K}(this wave)" for K in k_grid]
    else:
        xs = [K / d_state for K in k_grid]
        ys = [per_k_mean[K] for K in k_grid]
        labels = [f"K{K}(this wave)" for K in k_grid]
    xs_a, ys_a = np.array(xs), np.array(ys)

    sigmoid_fit = fit_sigmoid(xs_a, ys_a)
    linear_fit = fit_linear(xs_a, ys_a)
    ci_report = real_bootstrap_ci(
        per_k_seeds, np.array(k32_seeds) if anchored else None, np.array(k48_seeds) if anchored else None,
        n_trials=args.n_trials, d_state=d_state, seed=args.seed)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "fit_cliff_curve.py",
        "design_ref": "KEY_ANCHORING_DESIGN.md sec 12.4 (Rev 12.2) / sec 13.3 item 8 (Rev 13.2)",
        "d_state": d_state, "k_grid": list(k_grid), "anchored": anchored,
        "curve_points": {"x": xs, "h4": ys, "labels": labels},
        "sigmoid_fit": sigmoid_fit,
        "linear_fit_disclosed_secondary": linear_fit,
        "bootstrap_ci": ci_report,
        "degenerate_fraction_disclosure": {
            "value": ci_report["degenerate_frac"],
            "exceeds_10pct": bool(ci_report["degenerate_frac"] > 0.10),
            "rule": "sec 12.4 item 3: if the REAL degenerate fraction exceeds 10%, report explicitly "
                    "as a reliability caveat on the CI, not a silently-excluded footnote",
        },
        "wall_s": time.time() - t0,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out} ({out['wall_s']:.1f}s)")
    print(f"sigmoid fit: x0={sigmoid_fit['x0']:.4f} w={sigmoid_fit['w']:.4f} L={sigmoid_fit['L']:.4f} "
          f"rss={sigmoid_fit['rss']:.4g}")
    print(f"linear fit (disclosed, non-load-bearing): rss={linear_fit['rss']:.4g}")
    if ci_report["ci_x0"] is not None:
        print(f"bootstrap CI(x0): [{ci_report['ci_x0']['lo']:.4f}, {ci_report['ci_x0']['hi']:.4f}] "
              f"width={ci_report['ci_x0']['width']:.4f}")
        print(f"bootstrap CI(w): [{ci_report['ci_w']['lo']:.4f}, {ci_report['ci_w']['hi']:.4f}] "
              f"width={ci_report['ci_w']['width']:.4f}")
    print(f"degenerate fraction: {ci_report['degenerate_frac']:.4f} "
          f"({'EXCEEDS' if out['degenerate_fraction_disclosure']['exceeds_10pct'] else 'within'} the 10% bar)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
