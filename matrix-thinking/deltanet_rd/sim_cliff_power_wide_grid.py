"""sim_cliff_power_wide_grid.py -- KEY_ANCHORING_SCALING_DRAFT.md sec 15.20,
attack-round-1 MAJOR-2 fix (2026-07-07): the promoted, BLOCKING Stage -1 CPU
power check for the d=96 wide-grid wave's own ACTUAL, heterogeneous-seed
grid.

This is a purpose-built driver importing sim_cliff_power.py's own
`sigmoid`/`POOLED_REL_SD`/`NOISE_SD_MULTIPLIERS` UNMODIFIED (sec 15.8's own
"zero code changes to the reused script, only a new driver call" convention,
restated at sec 15.20.1's own citation of the same discipline) -- the
original script's `bootstrap_ci_width`/`simulate_once` take a single
`n_seeds_new` applied uniformly to every new K, which cannot represent this
wave's own registered grid: K=69 REUSED at n=2 (one seed, 1730, excluded on
admissibility -- sec 15.20.1/15.20.3) while K in {72,78,84,90} are fresh at
n=3 each (sec 15.20.1's seed table). This file adds a heterogeneous-n
sibling of `simulate_once`/`bootstrap_ci_width`, structurally identical
(same sigmoid form, same pooled relative-noise model, same degenerate-fit
guard thresholds) except for accepting a `{K: n_seeds}` dict instead of one
scalar.

Truths simulated (attack-round-1 MAJOR-2's own instruction: measure power
against the wave's own pre-registered rival predictions, not the original
sec 12 design's (0.5,0.75)-bracket-centered TRUTH_GRID, which sits entirely
below this wave's own K/d window of [0.71875, 0.9375] and would silently
simulate an already-fully-collapsed curve across every sampled K):
  - abs_slack_center: x0=0.729667 (sec 15.20.4's own d=80-recalibrated
    absolute-slack prediction, UNROUNDED per attack-round-1 MINOR-3 -- see
    sec 15.20.4's revised text)
  - power_law_center: x0=0.804619 (sec 15.20.4's own 2-point power-law
    prediction)
  - grid_interior: x0=0.75, a control point strictly between the abs-slack
    and power-law bands, inside the sampled K-window itself (between K=72
    and K=78)
  - upper_edge: x0=0.90, a control point near the grid's own rightmost
    sampled K (K=90, ratio 0.9375) -- NOT a CLIFF-BEYOND-WINDOW simulation
    (that scenario is structurally unrepresentable under this fit's own
    x0<=0.9 bound, sec 15.20.4's revised outcome-table adjudication;
    x0=0.90 is the closest legal boundary-adjacent truth)
each at w in {0.03 (sharp), 0.0597 (the ONLY real measured cliff width in
this program, d=64's own fitted w, cited sec 15.8), 0.08 (moderate)} --
mirrors the existing TRUTH_GRID's own sharp/measured/moderate width
convention (sec 12's TRUTH_GRID uses 0.03/0.05/0.08/0.15; 0.0597 replaces
the arbitrary 0.05 with the one width this program has actually measured).

Usage:
    ../../.venv/bin/python sim_cliff_power_wide_grid.py \
        --out sim_cliff_power_wide_grid_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import curve_fit

from sim_cliff_power import sigmoid, POOLED_REL_SD, NOISE_SD_MULTIPLIERS

# ---------------------------------------------------------------------------
# The wave's own REAL, registered, heterogeneous grid (sec 15.20.1's seed
# table): K=69 reused at n=2 (seed 1730 excluded on admissibility, seeds
# 1731/1732 survive), K in {72,78,84,90} fresh at n=3 each (primary seeds
# only -- the +2 contingency/Gate-1-probe slots are NOT counted here, this
# sim answers "what CI does the MANDATORY grid alone deliver").
# ---------------------------------------------------------------------------
D_STATE = 96
K_SEEDS = {69: 2, 72: 3, 78: 3, 84: 3, 90: 3}

# Rival predictions, sec 15.20.4 (Rev 1, unrounded per MINOR-3):
ABS_SLACK_X0 = 0.729667
POWER_LAW_X0 = 0.804619
# Inter-band gap, sec 15.20.4: [0.71833, 0.739] vs [0.768, 0.837], gap=0.029
# (0.768 - 0.739), half-gap = 0.0145 -- the derived MAJOR-2 trigger threshold.
INTER_BAND_GAP = 0.029
HALF_GAP_THRESHOLD = INTER_BAND_GAP / 2.0

WIDTHS = {"sharp_w03": 0.03, "measured_w0597": 0.0597, "moderate_w08": 0.08}

TRUTH_GRID_WIDE = []
for center_name, x0 in (
    ("abs_slack_center", ABS_SLACK_X0),
    ("power_law_center", POWER_LAW_X0),
    ("grid_interior_x075", 0.75),
    ("upper_edge_x090", 0.90),
):
    for w_name, w in WIDTHS.items():
        TRUTH_GRID_WIDE.append({"name": f"{center_name}_{w_name}", "L": 1.0, "x0": x0, "w": w})


def simulate_once_hetero(rng, k_seeds: dict, truth: dict, noise_sd_multiplier: float, d_state: int):
    """Heterogeneous-n sibling of sim_cliff_power.simulate_once (include_
    anchors=False always -- d=96 is not in fit_cliff_curve.py's
    ANCHORED_D_STATES, sec 15.20.0's own scoping note). Same sigmoid form,
    same pooled-relative-noise draw, same curve_fit bounds -- the ONLY
    difference from the original is n_seeds varying per K via the dict."""
    xs, ys = [], []
    for k, n_seeds in k_seeds.items():
        x = k / d_state
        true_mean = sigmoid(x, truth["L"], truth["x0"], truth["w"])
        seeds = rng.normal(true_mean, POOLED_REL_SD * noise_sd_multiplier * max(true_mean, 1e-6),
                            size=n_seeds)
        seeds = np.clip(seeds, 0.0, 1.0)
        xs.append(x); ys.append(float(seeds.mean()))
    xs = np.array(xs); ys = np.array(ys)
    try:
        popt, pcov = curve_fit(
            sigmoid, xs, ys,
            p0=[1.0, 0.75, 0.08],
            bounds=([0.5, 0.3, 0.005], [1.2, 0.9, 0.5]),
            maxfev=20000,
        )
        if not np.all(np.isfinite(pcov)):
            return None
        return {"L": float(popt[0]), "x0": float(popt[1]), "w": float(popt[2])}
    except Exception:  # noqa: BLE001 -- degenerate fit, counted as a miss by the caller
        return None


def bootstrap_ci_width_hetero(rng, k_seeds: dict, truth: dict, n_trials: int, ci: float = 0.95,
                               noise_sd_multiplier: float = 1.0, d_state: int = D_STATE):
    """Byte-identical degenerate-fit guard to sim_cliff_power.bootstrap_ci_width
    (same 1%-of-bound pin thresholds), only the inner simulate call differs."""
    x0s = []
    n_degenerate = 0
    for _ in range(n_trials):
        fit = simulate_once_hetero(rng, k_seeds, truth, noise_sd_multiplier, d_state)
        if fit is None:
            n_degenerate += 1
            continue
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
        "ci_width": float(hi - lo), "ci_lo": float(lo), "ci_hi": float(hi),
        "median_x0": float(np.median(x0s)),
        "n_degenerate": n_degenerate, "n_trials": n_trials,
        "degenerate_frac": n_degenerate / n_trials,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260706,
                     help="reuses sec 15.8's own registered power-check seed for continuity")
    ap.add_argument("--out", default="sim_cliff_power_wide_grid_results.json")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    primary = []
    for truth in TRUTH_GRID_WIDE:
        r = bootstrap_ci_width_hetero(rng, K_SEEDS, truth, args.n_trials, noise_sd_multiplier=1.0)
        primary.append({
            "truth": truth["name"], "true_x0": truth["x0"], "true_w": truth["w"],
            "noise_sd_multiplier": 1.0, "mandatory_cells": sum(K_SEEDS.values()),
            **r,
        })

    # Noise-sensitivity sub-sweep at the two rival-center truths only (house
    # convention, sec 12 Rev 12.2 fix 2 / sec 15.8), measured width w=0.0597.
    sensitivity = []
    for center_name, x0 in (("abs_slack_center", ABS_SLACK_X0), ("power_law_center", POWER_LAW_X0)):
        truth = {"name": f"{center_name}_measured_w0597", "L": 1.0, "x0": x0, "w": 0.0597}
        for mult in NOISE_SD_MULTIPLIERS:
            r = bootstrap_ci_width_hetero(rng, K_SEEDS, truth, args.n_trials, noise_sd_multiplier=mult)
            sensitivity.append({
                "truth": truth["name"], "true_x0": x0, "true_w": 0.0597,
                "noise_sd_multiplier": mult, "mandatory_cells": sum(K_SEEDS.values()),
                **r,
            })

    # MAJOR-2's own pre-registered response-rule check: projected CI
    # half-width vs. HALF_GAP_THRESHOLD (0.0145 = half of the 0.029
    # inter-band gap, sec 15.20.4), evaluated at the two rival-center,
    # measured-width (w=0.0597) truths -- the single most policy-relevant
    # read this sim produces.
    trigger_check = []
    for row in primary:
        if row["truth"] in ("abs_slack_center_measured_w0597", "power_law_center_measured_w0597"):
            half_width = row["ci_width"] / 2.0 if row["ci_width"] is not None else None
            trigger_check.append({
                "truth": row["truth"],
                "ci_width": row["ci_width"],
                "projected_half_width": half_width,
                "half_gap_threshold": HALF_GAP_THRESHOLD,
                "exceeds_threshold": (half_width is not None and half_width > HALF_GAP_THRESHOLD),
            })

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "sim_cliff_power_wide_grid.py",
        "purpose": "KEY_ANCHORING_SCALING_DRAFT.md sec 15.20 attack-round-1 MAJOR-2: power check "
                   "at the wave's REAL heterogeneous grid (K=69 n=2 + K in {72,78,84,90} n=3), "
                   "d_state=96, include_anchors=False.",
        "d_state": D_STATE,
        "k_seeds": K_SEEDS,
        "pooled_relative_noise_reused_from_sim_cliff_power": POOLED_REL_SD,
        "abs_slack_x0_unrounded": ABS_SLACK_X0,
        "power_law_x0": POWER_LAW_X0,
        "inter_band_gap": INTER_BAND_GAP,
        "half_gap_threshold": HALF_GAP_THRESHOLD,
        "n_trials_per_cell": args.n_trials,
        "seed": args.seed,
        "results_primary": primary,
        "results_noise_sensitivity": sensitivity,
        "major2_trigger_check": trigger_check,
        "wall_s": time.time() - t0,
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.out} ({time.time()-t0:.1f}s)")
    print(f"K_SEEDS grid: {K_SEEDS}  d_state={D_STATE}")
    print(f"half-gap threshold (MAJOR-2 trigger): {HALF_GAP_THRESHOLD:.4f}")
    print()
    print(f"{'truth':30s} {'ci_width':>10s} {'half_width':>11s} {'degen_frac':>11s}")
    for r in primary:
        cw = r["ci_width"]
        hw = cw / 2.0 if cw is not None else None
        print(f"{r['truth']:30s} {cw if cw is not None else float('nan'):10.4f} "
              f"{hw if hw is not None else float('nan'):11.4f} {r['degenerate_frac']:11.4f}")
    print()
    print("MAJOR-2 trigger check (rival-center, measured w=0.0597 width):")
    for t in trigger_check:
        print(f"  {t['truth']:30s} half_width={t['projected_half_width']:.4f} "
              f"threshold={t['half_gap_threshold']:.4f} "
              f"{'EXCEEDS -> fire response rule' if t['exceeds_threshold'] else 'within threshold'}")
    print()
    print("Noise sensitivity (rival centers, w=0.0597):")
    for r in sensitivity:
        print(f"  {r['truth']:30s} mult={r['noise_sd_multiplier']:.2f} "
              f"ci_width={r['ci_width']} degen_frac={r['degenerate_frac']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
