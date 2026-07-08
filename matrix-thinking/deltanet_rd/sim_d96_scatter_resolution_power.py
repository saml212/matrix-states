"""sim_d96_scatter_resolution_power.py -- KEY_ANCHORING_SCALING_DRAFT.md
sec 15.26 (Rev 0)'s pre-registered, RUN-THIS-SESSION power sketch for the
d=96 SCATTER-RESOLUTION wave (+2 seeds/K at K in {69,72,78,84,90}, n=3->5).

Mirrors sim_cliff_power_wide_grid.py's own precedent (sec 15.20.4 MAJOR-2):
a purpose-built, CPU-only driver that imports sigmoid()/curve_fit bounds
from sim_cliff_power.py UNMODIFIED (never re-implemented), promoted to a
Stage -1 BLOCKING check and RUN THIS SESSION, before any GPU cell of this
wave launches -- the same discipline, applied to a different question.

QUESTION THIS SCRIPT ANSWERS (sec 15.26's own hypothesis section): given
the REAL, already-archived n=3 per-K h4 values (post sec 15.25.4's
tolerance-miscalibration unlock -- verified directly against the raw
JSONs by this design session, not re-typed from the draft prose), what is
the probability that adding 2 MORE real seeds per K (n=3->5) produces a
non-degenerate sigmoid fit, under two disclosed, pre-registered nulls:

  H0 "SCATTER-IS-REAL" -- the 2 new draws at each K are generated from
     THAT SAME K's own observed (mean, sd). If the current n=3 sample
     already sits near the true population values (including K90's exact
     3/3 ceiling, sd=0), this is what "nothing changes" looks like.

  H1 "DIP-IS-NOISE" -- a minimal, targeted alternative: K69/K84/K90's
     true means are left AT their own observed sample means (unchanged --
     those three are not in question), but K72/K78 (the two K's driving
     the observed dip) have their TRUE mean replaced by the K69-to-K84
     linear interpolation at that K's own K/d ratio (i.e. "what if the
     dip is pure between-seed noise and the true curve is actually flat-
     ish near ceiling across this whole window, as K69/K84/K90 already
     suggest"). Noise SCALE (sd) at every K, under EITHER null, reuses
     that K's own observed sd -- disclosed, not a free parameter.

DELIBERATELY NOT SIMULATED: a literal external "true curve = one of the
two rival bands' sigmoid" null. Checked and rejected as incoherent given
the FIXED, already-real K90 anchor: sigmoid(x=90/96, x0=0.729667 or
0.80, w=0.05, L=1.0) evaluates to <0.02 at either rival center -- flatly
contradicted by K90's own real, already-collected 3/3-exact-ceiling
data (sd=0.0000), which this script holds FIXED (never re-drawn) at
every trial. Simulating draws from that truth at K90 would silently mix
one incoherent synthetic point with two already-decided-and-fixed real
identical-1.0 points, produproducing an internally inconsistent
5-point curve that tests nothing real. Disclosed here rather than run
anyway and buried in an appendix.

Per-trial procedure (single-level simulation, mirrors sim_cliff_power.
bootstrap_ci_width's own precedent exactly: repeat "draw synthetic data,
fit ONCE" n_trials times, and let the ACROSS-TRIAL distribution of
outcomes answer the power question -- no nested inner bootstrap, since
what this sketch needs is "how often would a hypothetical n=5 wave's own
MAIN fit degenerate", the same quantity sec 12.4/15.20.4's own house
convention already reports this way):
  1. For each K in {69,72,78,84,90}: draw 2 new synthetic h4 values from
     N(true_mean_K, obs_sd_K) [H0 or H1, see above], clip to [0,1].
  2. Combine with the 3 REAL, FIXED archived seeds at that K (never
     re-drawn) -> 5 values; take the mean.
  3. Fit the sigmoid to the resulting 5-point (K/d, mean_h4) curve, same
     form/p0/bounds/maxfev as fit_cliff_curve.fit_sigmoid/sim_cliff_
     power.simulate_once (imported, not re-implemented).
  4. Record: degenerate (w or x0 within 1% of a bound, or fit raises),
     whether the 5-point mean curve is itself monotonically
     non-increasing (a assumption-free structural precondition, cheaper
     than curve_fit and reported alongside it), and, for non-degenerate
     fits, whether x0 lands inside EITHER registered rival band
     (abs-slack [0.718,0.739], power-law [0.768,0.837]).

Usage:
    python sim_d96_scatter_resolution_power.py --out sim_d96_scatter_resolution_power_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# REUSED directly (mirrors fit_cliff_curve.py's own sec 12.4 item 5 discipline)
from sim_cliff_power import sigmoid  # noqa: E402

D_STATE = 96

# Real, archived, post-sec-15.25.4-unlock per-K per-seed h4 values --
# independently re-verified this design session directly against
# experiment-runs/2026-07-07_keyanchor_scaling_wide/results/
# deltanet_rd_exactness/wavekeyanchor-scaling-wide/*.json
# (checkpoints[-1]['M3_held_out']['4']['recovered_frac@0.9']), matching
# sec 15.25.5's own table to 4 decimal places. Seeds 1731/1732/1733 for
# K=69 (seed 1730 excluded, disclosed sec 15.25.4 scope decision, out of
# this wave's own n=3 baseline); 1740/1741/1742 for K=72;
# 1840/1841/1842 for K=78; 1940/1941/1942 for K=84; 2040/2041/2042 for
# K=90.
REAL_SEEDS_H4 = {
    69: [0.9985846877098083, 0.9614187479019165, 0.91745924949646],
    72: [0.931884765625, 0.8425564169883728, 0.9903971552848816],
    78: [0.8666616678237915, 0.9822967052459717, 0.9487680196762085],
    84: [0.9806315302848816, 0.9573102593421936, 0.9363374710083008],
    90: [1.0, 1.0, 1.0],
}
KS = sorted(REAL_SEEDS_H4.keys())

RIVAL_BANDS = {
    "abs_slack": (0.718, 0.739),
    "power_law": (0.768, 0.837),
}


def obs_mean_sd(k):
    arr = np.array(REAL_SEEDS_H4[k])
    return float(arr.mean()), float(arr.std(ddof=1))


OBS = {k: obs_mean_sd(k) for k in KS}


def h1_true_mean(k):
    """K69/K84/K90 unchanged (own observed mean); K72/K78 replaced by the
    K69->K84 linear interpolation at that K's own K/d (the 'dip is noise,
    true curve is flat-ish near ceiling across this window' null)."""
    if k in (69, 84, 90):
        return OBS[k][0]
    x69, x84 = 69.0 / D_STATE, 84.0 / D_STATE
    m69, m84 = OBS[69][0], OBS[84][0]
    x = k / D_STATE
    frac = (x - x69) / (x84 - x69)
    return m69 + frac * (m84 - m69)


H1_TRUE_MEAN = {k: h1_true_mean(k) for k in KS}


def fit_sigmoid_once(xs, ys):
    """Byte-identical form/bounds/maxfev to fit_cliff_curve.fit_sigmoid /
    sim_cliff_power.simulate_once's own curve_fit call."""
    try:
        popt, pcov = curve_fit(
            sigmoid, xs, ys, p0=[1.0, 0.625, 0.08],
            bounds=([0.5, 0.3, 0.005], [1.2, 0.9, 0.5]), maxfev=20000,
        )
        if not np.all(np.isfinite(pcov)):
            return None
    except Exception:
        return None
    L, x0, w = float(popt[0]), float(popt[1]), float(popt[2])
    degenerate = (w <= 0.005 * 1.01 or w >= 0.5 * 0.99 or
                  x0 <= 0.3 * 1.01 or x0 >= 0.9 * 0.99)
    return {"L": L, "x0": x0, "w": w, "degenerate": degenerate}


def run_null(rng, null_name, n_trials):
    true_mean_fn = (lambda k: OBS[k][0]) if null_name == "H0" else (lambda k: H1_TRUE_MEAN[k])
    n_degenerate = 0
    n_monotonic = 0
    x0s_nondeg = []
    band_hits = {name: 0 for name in RIVAL_BANDS}
    flat_ge_098 = 0  # sec 15.20.4 row-1a-style flatness signature on the n=5 means
    for _ in range(n_trials):
        xs, ys = [], []
        for k in KS:
            mu = true_mean_fn(k)
            sd = OBS[k][1]
            new_draws = rng.normal(mu, sd, size=2) if sd > 0 else np.full(2, mu)
            new_draws = np.clip(new_draws, 0.0, 1.0)
            all5 = np.concatenate([REAL_SEEDS_H4[k], new_draws])
            xs.append(k / D_STATE)
            ys.append(float(all5.mean()))
        xs_a, ys_a = np.array(xs), np.array(ys)
        if np.all(np.diff(ys_a) <= 1e-12):
            n_monotonic += 1
        if np.all(ys_a >= 0.98):
            flat_ge_098 += 1
        fit = fit_sigmoid_once(xs_a, ys_a)
        if fit is None or fit["degenerate"]:
            n_degenerate += 1
            continue
        x0s_nondeg.append(fit["x0"])
        for name, (lo, hi) in RIVAL_BANDS.items():
            if lo <= fit["x0"] <= hi:
                band_hits[name] += 1
    out = {
        "null": null_name,
        "n_trials": n_trials,
        "degenerate_frac": n_degenerate / n_trials,
        "monotonic_frac": n_monotonic / n_trials,
        "flat_ge_0.98_frac": flat_ge_098 / n_trials,
        "n_nondegenerate": len(x0s_nondeg),
    }
    if x0s_nondeg:
        arr = np.array(x0s_nondeg)
        out["x0_nondegenerate"] = {
            "median": float(np.median(arr)),
            "q025": float(np.quantile(arr, 0.025)),
            "q975": float(np.quantile(arr, 0.975)),
        }
        out["band_hit_frac_of_all_trials"] = {name: v / n_trials for name, v in band_hits.items()}
    else:
        out["x0_nondegenerate"] = None
        out["band_hit_frac_of_all_trials"] = {name: 0.0 for name in RIVAL_BANDS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--out", default=os.path.join(HERE, "sim_d96_scatter_resolution_power_results.json"))
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    obs_report = {k: {"mean": OBS[k][0], "sd": OBS[k][1], "se_n3": OBS[k][1] / (3 ** 0.5),
                       "se_n5_projected": OBS[k][1] / (5 ** 0.5)} for k in KS}
    h0 = run_null(rng, "H0", args.n_trials)
    h1 = run_null(rng, "H1", args.n_trials)

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": "sim_d96_scatter_resolution_power.py",
        "design_ref": "KEY_ANCHORING_SCALING_DRAFT.md sec 15.26 (Rev 0)",
        "d_state": D_STATE,
        "real_seeds_h4_n3_post_unlock": REAL_SEEDS_H4,
        "obs_mean_sd_se": obs_report,
        "h1_true_mean_used": H1_TRUE_MEAN,
        "rival_bands": RIVAL_BANDS,
        "se_shrink_n3_to_n5": (5 / 3) ** 0.5,
        "results": {"H0_scatter_is_real": h0, "H1_dip_is_noise": h1},
        "wall_s": time.time() - t0,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.out} ({out['wall_s']:.1f}s)")
    print(f"SE shrink factor n=3->n=5: {out['se_shrink_n3_to_n5']:.4f}")
    print()
    for name, r in out["results"].items():
        print(f"--- {name} ---")
        print(f"  degenerate_frac={r['degenerate_frac']:.4f}  monotonic_frac={r['monotonic_frac']:.4f}  "
              f"flat>=0.98_frac={r['flat_ge_0.98_frac']:.4f}")
        if r["x0_nondegenerate"] is not None:
            print(f"  x0 (non-degenerate trials only): median={r['x0_nondegenerate']['median']:.4f} "
                  f"[{r['x0_nondegenerate']['q025']:.4f},{r['x0_nondegenerate']['q975']:.4f}]")
            print(f"  band hit frac (of ALL trials): {r['band_hit_frac_of_all_trials']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
