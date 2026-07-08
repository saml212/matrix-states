"""sim_d96_scatter_resolution_power_extended.py -- REVISION-agent session,
KEY_ANCHORING_SCALING_DRAFT.md sec 15.26 Rev 1 fix for MAJOR-1/finding
registration. RUN THIS SESSION, CPU-only.

Three independent pieces of additional confirmation for the sec 15.26.1
power-check finding (0/40,000 combined trials non-degenerate+monotonic
at n=20,000/null, seed=20260708, already archived), none of them
re-running the ALREADY-archived 40,000-trial result -- all NEW:

  (A) EXTENDED MULTI-SEED CONFIRMATION: the existing, audited
      sim_d96_scatter_resolution_power.py driver (imported UNMODIFIED,
      same discipline as that script's own import of sim_cliff_power.py),
      re-run at 7 NEW seeds (distinct from the archived 20260708),
      20,000 trials/null/seed -> 7*20,000*2 = 280,000 additional trials.

  (B) FROM-SCRATCH REIMPLEMENTATION: an INDEPENDENTLY re-typed sigmoid
      form + curve_fit call + degeneracy check (never imports sigmoid()
      from sim_cliff_power.py or fit_sigmoid_once() from the original
      driver) -- rules out a shared bug across all three scripts that
      import the same sigmoid(). Run once, one seed, 20,000 trials/null.

  (C) POSITIVE CONTROL: a synthetic truth deliberately constructed to be
      NON-degenerate (sigmoid centered at the abs-slack rival x0=0.729667,
      w=0.0597, sampled at LOW noise sd=0.01, not this program's own much
      noisier real per-K sds) fed through the SAME degeneracy check used
      in (A) -- proves the check is not vacuously always-degenerate (has
      teeth), the CLAUDE.md "assert has teeth" discipline applied to this
      diagnostic instrument itself, not just to the geo3_admission checks
      elsewhere in this program.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sim_d96_scatter_resolution_power as base  # noqa: E402  -- REUSED unmodified for (A)

D_STATE = 96
RIVAL_BANDS = {"abs_slack": (0.718, 0.739), "power_law": (0.768, 0.837)}

# ---------------------------------------------------------------------
# (A) Extended multi-seed confirmation -- reuses base.run_null unmodified
# ---------------------------------------------------------------------
NEW_SEEDS = [20260801, 20260802, 20260803, 20260804, 20260805, 20260806, 20260807]
N_TRIALS_PER_SEED = 20000

def run_extended_multiseed():
    per_seed = []
    for s in NEW_SEEDS:
        rng = np.random.default_rng(s)
        h0 = base.run_null(rng, "H0", N_TRIALS_PER_SEED)
        h1 = base.run_null(rng, "H1", N_TRIALS_PER_SEED)
        per_seed.append({"seed": s, "H0_degenerate_frac": h0["degenerate_frac"],
                          "H0_monotonic_frac": h0["monotonic_frac"],
                          "H1_degenerate_frac": h1["degenerate_frac"],
                          "H1_monotonic_frac": h1["monotonic_frac"]})
    total_trials = len(NEW_SEEDS) * N_TRIALS_PER_SEED * 2
    total_degenerate = sum(round(r["H0_degenerate_frac"] * N_TRIALS_PER_SEED) +
                            round(r["H1_degenerate_frac"] * N_TRIALS_PER_SEED) for r in per_seed)
    total_nondeg_and_monotonic = sum(
        round((1 - r["H0_degenerate_frac"]) * r["H0_monotonic_frac"] * N_TRIALS_PER_SEED) +
        round((1 - r["H1_degenerate_frac"]) * r["H1_monotonic_frac"] * N_TRIALS_PER_SEED)
        for r in per_seed)
    return {"n_seeds": len(NEW_SEEDS), "n_trials_per_seed_per_null": N_TRIALS_PER_SEED,
            "total_trials": total_trials, "per_seed": per_seed,
            "combined_degenerate_frac": total_degenerate / total_trials,
            "all_seeds_100pct_degenerate_both_nulls": all(
                r["H0_degenerate_frac"] == 1.0 and r["H1_degenerate_frac"] == 1.0 for r in per_seed)}


# ---------------------------------------------------------------------
# (B) From-scratch reimplementation -- independently re-typed, no shared
# sigmoid()/fit_sigmoid_once() import from base or sim_cliff_power.
# ---------------------------------------------------------------------
def sigmoid_reimpl(x, L, x0, w):
    return L / (1.0 + np.exp((x - x0) / w))


def fit_and_check_reimpl(xs, ys):
    try:
        popt, pcov = curve_fit(sigmoid_reimpl, xs, ys, p0=[1.0, 0.625, 0.08],
                                bounds=([0.5, 0.3, 0.005], [1.2, 0.9, 0.5]), maxfev=20000)
        if not np.all(np.isfinite(pcov)):
            return {"degenerate": True, "reason": "non-finite covariance"}
    except Exception as e:
        return {"degenerate": True, "reason": f"fit raised: {e}"}
    L, x0, w = float(popt[0]), float(popt[1]), float(popt[2])
    is_degenerate = (w <= 0.005 * 1.01 or w >= 0.5 * 0.99 or
                      x0 <= 0.3 * 1.01 or x0 >= 0.9 * 0.99)
    return {"degenerate": is_degenerate, "L": L, "x0": x0, "w": w}


def run_reimpl_check(seed=20260810, n_trials=20000):
    rng = np.random.default_rng(seed)
    KS = base.KS
    for null_name in ("H0", "H1"):
        true_mean_fn = (lambda k: base.OBS[k][0]) if null_name == "H0" else (lambda k: base.H1_TRUE_MEAN[k])
        n_deg = 0
        n_mono = 0
        for _ in range(n_trials):
            xs, ys = [], []
            for k in KS:
                mu = true_mean_fn(k)
                sd = base.OBS[k][1]
                new_draws = rng.normal(mu, sd, size=2) if sd > 0 else np.full(2, mu)
                new_draws = np.clip(new_draws, 0.0, 1.0)
                all5 = np.concatenate([base.REAL_SEEDS_H4[k], new_draws])
                xs.append(k / D_STATE)
                ys.append(float(all5.mean()))
            xs_a, ys_a = np.array(xs), np.array(ys)
            if np.all(np.diff(ys_a) <= 1e-12):
                n_mono += 1
            fit = fit_and_check_reimpl(xs_a, ys_a)
            if fit["degenerate"]:
                n_deg += 1
        yield {"null": null_name, "n_trials": n_trials, "degenerate_frac": n_deg / n_trials,
               "monotonic_frac": n_mono / n_trials}


# ---------------------------------------------------------------------
# (C) Positive control -- synthetic non-degenerate truth, SAME degeneracy
# check (reuses base.fit_sigmoid_once, the EXACT function used in (A)/the
# original archived 40,000-trial run) -- proves the check has teeth.
# ---------------------------------------------------------------------
def run_positive_control(seed=20260811, n_trials=2000, x0_true=0.729667, w_true=0.0597, noise_sd=0.01):
    rng = np.random.default_rng(seed)
    KS = base.KS
    n_deg = 0
    n_mono = 0
    n_band_hit_abs_slack = 0
    for _ in range(n_trials):
        xs, ys = [], []
        for k in KS:
            x = k / D_STATE
            true_y = base.sigmoid(x, 1.0, x0_true, w_true)
            draw = np.clip(rng.normal(true_y, noise_sd), 0.0, 1.0)
            xs.append(x)
            ys.append(float(draw))
        xs_a, ys_a = np.array(xs), np.array(ys)
        if np.all(np.diff(ys_a) <= 1e-12):
            n_mono += 1
        fit = base.fit_sigmoid_once(xs_a, ys_a)
        if fit is None or fit["degenerate"]:
            n_deg += 1
            continue
        lo, hi = RIVAL_BANDS["abs_slack"]
        if lo <= fit["x0"] <= hi:
            n_band_hit_abs_slack += 1
    return {"n_trials": n_trials, "x0_true": x0_true, "w_true": w_true, "noise_sd": noise_sd,
            "degenerate_frac": n_deg / n_trials, "monotonic_frac": n_mono / n_trials,
            "abs_slack_band_hit_frac": n_band_hit_abs_slack / n_trials}


def main():
    t0 = time.time()
    ext = run_extended_multiseed()
    reimpl = list(run_reimpl_check())
    pos_ctrl = run_positive_control()

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "sim_d96_scatter_resolution_power_extended.py",
        "design_ref": "KEY_ANCHORING_SCALING_DRAFT.md sec 15.26 (Rev 1), MAJOR-1/finding-registration",
        "part_A_extended_multiseed": ext,
        "part_B_from_scratch_reimplementation": reimpl,
        "part_C_positive_control": pos_ctrl,
        "wall_s": time.time() - t0,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "sim_d96_scatter_resolution_power_extended_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_path} ({out['wall_s']:.1f}s)")
    print(f"(A) extended multiseed: {ext['n_seeds']} seeds x {ext['n_trials_per_seed_per_null']} "
          f"trials/null = {ext['total_trials']} total trials; "
          f"all_seeds_100pct_degenerate_both_nulls={ext['all_seeds_100pct_degenerate_both_nulls']}; "
          f"combined_degenerate_frac={ext['combined_degenerate_frac']:.6f}")
    for r in reimpl:
        print(f"(B) reimpl {r['null']}: degenerate_frac={r['degenerate_frac']:.4f} "
              f"monotonic_frac={r['monotonic_frac']:.4f} (n={r['n_trials']})")
    print(f"(C) positive control: degenerate_frac={pos_ctrl['degenerate_frac']:.4f} "
          f"monotonic_frac={pos_ctrl['monotonic_frac']:.4f} "
          f"abs_slack_band_hit_frac={pos_ctrl['abs_slack_band_hit_frac']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
