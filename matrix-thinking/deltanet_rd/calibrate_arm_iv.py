#!/usr/bin/env python3
"""calibrate_arm_iv.py -- DELTANET_RD_EXACTNESS_DESIGN.md sec 4.5 (Rev 3,
BLOCK-1): the closed-loop (alpha,rho) calibration for arm (iv)
frozen_gram_matched, per K. NOT embarrassingly parallel (each iteration's
next probe depends on the previous iteration's measured result), so this
is a SEPARATE sequential driver script, not a run_deltanet_rd_exactness_
sweep.py manifest entry -- that sweep's Wave -1 launches every OTHER short
probe and points here for arm (iv)'s calibration specifically.

Protocol (sec 4.5, exact; search mechanics revised per the 2026-07-03
independent audit's FIX-1 -- see calibrate_one_K's docstring for the
before/after):
  1. initial (alpha,rho) from the raw-row reachability arithmetic
     (embed_arms.reachability_iid_closed_form / estimate_gram_deviation),
     via a TWO-stage CPU-only scan (coarse 0.05-step, then fine 0.01-step
     around the coarse winner) -- the measured per-K target bands are only
     ~0.01-0.02 WIDE in rho, so seeding must land within ~0.005 of the
     proxy optimum for the <=3-iteration cap to be reachable at all.
     This build fixes alpha=1.0 (the family's pure-Gaussian/max-disorder
     endpoint) and searches rho ALONE -- a documented simplification (see
     the note below), since the target is a single scalar (Gram
     deviation) and rho alone has a strong, monotone lever over the full
     target range (design's own measured rho=0.3 -> ~1.72/~3.44 numbers).
  2. one 5,000-step calibration probe per K (k_eff geometry is meaningless
     without a trained W_k/conv -- the probe trains one).
  3. if the probe's final-checkpoint k_eff Gram deviation misses the
     per-K target band, adjust rho by DETERMINISTIC BISECTION inside a
     bracket ANCHORED at the seeded winner (rho0 +- 0.05, clipped to
     [0, 0.95]) and re-probe; <=3 iterations per K, HARD CAP,
     pre-registered. On non-convergence, final_rho is the BEST-FOUND
     iterate (min |deviation - band midpoint|), never the last-tried one
     (FIX-1) -- sec 4.5's >25%-miss demotion gate then decides what the
     miss costs at the full runs.
  4. freeze the table (write the final (alpha,rho) + the constructed
     frozen table to --out-dir for reuse by the actual Wave 1 runs).

Target bands come from analyze_exactness_w0.py's own per-K Gram-deviation
extraction (sec 3.1) -- pass --w0-report (the JSON analyze_exactness_w0.py
--out-json wrote) or --target-lo/--target-hi directly.

DOCUMENTED SIMPLIFICATION (not in the design's literal text, flagged for
audit scrutiny): sec 4.5 says "adjust (alpha,rho) by deterministic
bisection" without specifying how the 2D adjustment splits across two
parameters for a 1D (scalar Gram-deviation) target. This driver fixes
alpha=1.0 for every K (matching the reachability calc's own alpha=1
characterization) and searches rho only. If a future revision needs the
anisotropy-vs-noise split alpha itself calibrates, extend the bisection
to a 2D search; not needed to hit the measured target bands at rho in
[0,1] (embed_arms.estimate_gram_deviation confirms the family spans both
K=16 and K=32's bands with margin at alpha=1).

THIS SCRIPT LAUNCHES REAL GPU TRAINING (5,000-step probes, up to 3 per K
-- Wave -1 GPU spend, NOT a smoke). Check nvidia-smi and GPU availability
before running for real; --dry-run prints the planned probe sequence
without launching anything; --simulate replaces every GPU probe with the
raw-row Monte-Carlo proxy (embed_arms.estimate_gram_deviation, fresh RNG
per iteration, CPU-only) -- the audit-prescribed convergence-verification
harness, writing CALIBRATION_SUMMARY_SIMULATED.json (a DIFFERENT filename
so a simulation can never be mistaken for a real calibration by
run_deltanet_rd_exactness_sweep.py's FIX-2 gate).

Usage:
  python calibrate_arm_iv.py --k 16 32 --w0-report w0_report.json \\
      --gpu 6 --out-dir results/deltanet_rd_exactness/arm_iv_calib
  python calibrate_arm_iv.py --k 16 32 --w0-report w0_report.json \\
      --simulate  # CPU-only convergence check against the raw-row oracle
  python calibrate_arm_iv.py --k 16 32 --target-lo 1.257 --target-hi 1.338 \\
      --dry-run   # prints the plan only, launches nothing
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embed_arms

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "run_deltanet_rd.py")
CALIB_STEPS = 5000
MAX_ITERS = 3
ALPHA_FIXED = 1.0


def target_band_for_K(K: int, w0_report_path: str | None, target_lo: float | None,
                        target_hi: float | None) -> tuple[float, float]:
    if target_lo is not None and target_hi is not None:
        return target_lo, target_hi
    assert w0_report_path, "need either --w0-report or explicit --target-lo/--target-hi"
    with open(w0_report_path) as f:
        report = json.load(f)
    band = report.get("gram_bands_per_K", {}).get(str(K)) or report.get("gram_bands_per_K", {}).get(K)
    assert band and "key_gram_deviation" in band, \
        f"no key_gram_deviation band for K={K} in {w0_report_path} -- run analyze_exactness_w0.py first"
    return band["key_gram_deviation"]["min"], band["key_gram_deviation"]["max"]


def _seed_rho(K: int, d_model: int, target_mid: float) -> float:
    """sec 4.5 step 1's raw-row starting point, TWO-stage (FIX-1,
    2026-07-03 audit): coarse 0.05-step scan over [0.05, 0.70], then a
    fine 0.01-step scan over coarse-winner +- 0.10, both against the
    cheap CPU Monte-Carlo proxy (embed_arms.estimate_gram_deviation, no
    GPU). WHY fine seeding is load-bearing, not polish: the measured
    per-K target bands are only ~0.01-0.02 wide in rho (K=16
    [1.257,1.338] maps to rho ~[0.229,0.246]; K=32 [2.663,2.780] to rho
    ~[0.240,0.251], closed-form check this session) -- a coarse-only seed
    can start up to 0.05 away, and the audit's simulation showed that
    combined with the old full-domain bracket this made the <=3-iteration
    cap unreachable at BOTH K."""
    best_rho, best_gap = 0.3, float("inf")
    for i in range(1, 15):                                        # 0.05 .. 0.70, step 0.05
        c = i * 0.05
        est = embed_arms.estimate_gram_deviation(alpha=ALPHA_FIXED, rho=c, K=K,
                                                    d_model=d_model, n_trials=200, seed=K)
        if abs(est - target_mid) < best_gap:
            best_gap, best_rho = abs(est - target_mid), c
    coarse = best_rho
    best_gap = float("inf")                                        # reset: fine stage uses its own RNG seed
    for j in range(-10, 11):                                       # coarse-winner +- 0.10, step 0.01
        c = min(0.95, max(0.01, coarse + j * 0.01))
        est = embed_arms.estimate_gram_deviation(alpha=ALPHA_FIXED, rho=c, K=K,
                                                    d_model=d_model, n_trials=400, seed=K + 1)
        if abs(est - target_mid) < best_gap:
            best_gap, best_rho = abs(est - target_mid), c
    return best_rho


def calibrate_one_K(K: int, target_lo: float, target_hi: float, gpu: int | None, out_dir: str,
                      d_model: int = 256, dry_run: bool = False, simulate: bool = False) -> dict:
    """FIX-1 (2026-07-03 audit, MAJOR): the pre-fix version initialized
    the bisection bracket to the full [0, 0.95] domain and never narrowed
    it around the grid-search winner -- the auditor simulated that exact
    algorithm against estimate_gram_deviation as oracle and BOTH K=16 and
    K=32 failed to converge within the 3-iteration cap (first bisection
    step jumped to rho=0.575 -> deviation ~5.15, wildly overshooting a
    band only ~0.01-0.05 wide in rho). Fixed: (a) bracket anchored to the
    seeded winner (rho0 +- 0.05, clipped -- +-0.05 rather than the
    audit's suggested +-0.1 floor because the fine 0.01-step seeding
    stage above already localizes rho0 to ~0.005 of the proxy optimum,
    and a tighter bracket makes each bisection step ~2x finer within the
    same hard cap); (b) on non-convergence, final_rho = the BEST-FOUND
    iterate (min |measured - band midpoint|), never the last-tried; (c)
    the dead lo/hi parameters of the old _bisect_rho are gone (bisection
    is inlined). Monotonicity assumption unchanged: deviation is monotone
    non-decreasing in rho at fixed alpha (verified on the realized table,
    embed_arms._self_test [embed_arms 4]); the trained-k_eff mapping is
    assumed to inherit it -- a violation surfaces as visible
    non-convergence in the recorded trace, never silently."""
    target_mid = (target_lo + target_hi) / 2.0
    rho0 = _seed_rho(K, d_model, target_mid)
    rho_lo, rho_hi = max(0.0, rho0 - 0.05), min(0.95, rho0 + 0.05)
    rho = rho0
    trace = []
    best = None                       # (gap_to_mid, rho, measured)
    for it in range(MAX_ITERS):
        record = {"iter": it, "K": K, "alpha": ALPHA_FIXED, "rho": rho,
                  "bracket": [rho_lo, rho_hi],
                  "target_lo": target_lo, "target_hi": target_hi, "target_mid": target_mid}
        if dry_run:
            record["measured_key_gram_deviation"] = None
            record["note"] = "DRY RUN -- no probe launched"
            trace.append(record)
            print(f"  [K={K} iter={it}] alpha={ALPHA_FIXED} rho={rho:.4f} "
                  f"bracket=[{rho_lo:.4f},{rho_hi:.4f}] -> would launch a "
                  f"{CALIB_STEPS}-step probe (DRY RUN, not launched)")
            continue

        if simulate:
            # The audit-prescribed verification harness: the raw-row
            # Monte-Carlo proxy stands in for the trained probe, with a
            # per-iteration RNG seed INDEPENDENT of the seeding scans
            # (seeds K / K+1 above) so convergence is a real test of the
            # search, not a replay of the seeding draw.
            measured = embed_arms.estimate_gram_deviation(
                alpha=ALPHA_FIXED, rho=rho, K=K, d_model=d_model,
                n_trials=400, seed=90_000 + K * 17 + it)
            record["oracle"] = "estimate_gram_deviation (SIMULATED -- no GPU probe)"
        else:
            os.makedirs(out_dir, exist_ok=True)
            probe_name = f"calib_iv_K{K}_it{it}"
            out_path = os.path.join(out_dir, f"{probe_name}.json")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
            cmd = [sys.executable, RUN, "--K", str(K), "--steps", str(CALIB_STEPS),
                   "--seed", str(1000 + it), "--embed-source", "frozen_gram_matched",
                   "--gram-alpha", str(ALPHA_FIXED), "--gram-rho", str(rho),
                   "--ckpt-every", str(CALIB_STEPS), "--out", out_path]
            print(f"  [K={K} iter={it}] alpha={ALPHA_FIXED} rho={rho:.4f} -> launching: "
                  f"{' '.join(cmd)}", flush=True)
            log_path = os.path.join(out_dir, f"{probe_name}.log")
            with open(log_path, "w") as lf:
                rc = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=HERE)
            if rc != 0 or not os.path.exists(out_path):
                record["error"] = f"probe FAILED (rc={rc}), see {log_path}"
                trace.append(record)
                print(f"  [K={K} iter={it}] PROBE FAILED (rc={rc}) -- see {log_path}", flush=True)
                break
            with open(out_path) as f:
                result = json.load(f)
            ck = result.get("checkpoints") or []
            h1 = str(min(result.get("H_train", [1])))
            m2 = (ck[-1].get("M2_in_distribution") if ck else {}) or {}
            entry = m2.get(h1, {})
            measured = entry.get("key_gram_deviation_mean")
            record["result_path"] = out_path

        record["measured_key_gram_deviation"] = measured
        trace.append(record)
        if measured is None:
            print(f"  [K={K} iter={it}] measured deviation UNAVAILABLE (probe did not complete "
                  f"a checkpoint) -- stopping this K's calibration", flush=True)
            break
        gap = abs(measured - target_mid)
        if best is None or gap < best[0]:
            best = (gap, rho, measured)
        in_band = target_lo <= measured <= target_hi
        print(f"  [K={K} iter={it}] measured key_gram_deviation={measured:.4f}  "
              f"target=[{target_lo:.4f},{target_hi:.4f}]  in_band={in_band}", flush=True)
        if in_band:
            break
        # inlined deterministic bisection (monotone non-decreasing in rho)
        if measured < target_lo:
            rho_lo = rho            # too orthonormal -- need MORE shared-direction correlation
        else:
            rho_hi = rho             # too correlated -- need LESS
        rho = (rho_lo + rho_hi) / 2.0

    converged = best is not None and target_lo <= best[2] <= target_hi
    # FIX-1(b): BEST-FOUND on non-convergence, never last-tried; falls
    # back to the seeded rho0 only if no probe produced a measurement.
    final_rho = best[1] if best is not None else rho0
    return {"K": K, "alpha": ALPHA_FIXED, "final_rho": final_rho, "converged": converged,
            "simulated": bool(simulate),
            "best_measured": best[2] if best is not None else None,
            "seed_rho0": rho0, "n_iterations": len(trace), "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", required=True, help="K values to calibrate, e.g. 16 32")
    ap.add_argument("--w0-report", default=None, help="analyze_exactness_w0.py --out-json output")
    ap.add_argument("--target-lo", type=float, default=None)
    ap.add_argument("--target-hi", type=float, default=None)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=None, help="physical GPU index (check nvidia-smi first)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "results/deltanet_rd_exactness/arm_iv_calib"))
    ap.add_argument("--dry-run", action="store_true", help="print the planned probe sequence, launch nothing")
    ap.add_argument("--simulate", action="store_true",
                     help="FIX-1's convergence-verification harness: replace every GPU probe with "
                          "the raw-row Monte-Carlo proxy (CPU-only, fresh RNG per iteration). "
                          "Writes CALIBRATION_SUMMARY_SIMULATED.json -- a DIFFERENT filename, so "
                          "a simulation can never satisfy the sweep's FIX-2 converged-calibration "
                          "gate.")
    args = ap.parse_args()

    if not args.dry_run and not args.simulate and args.gpu is None:
        print("ERROR: --gpu is required for a real launch (no default on purpose -- check "
              "nvidia-smi first). Use --dry-run to preview the plan without a GPU, or "
              "--simulate for the CPU-only oracle harness.", file=sys.stderr)
        sys.exit(1)

    results = {}
    for K in args.k:
        lo, hi = target_band_for_K(K, args.w0_report, args.target_lo, args.target_hi)
        mode = " [SIMULATED oracle]" if args.simulate else ""
        print(f"\n=== calibrating arm (iv) at K={K}, target key_gram_deviation band "
              f"[{lo:.4f}, {hi:.4f}]{mode} ===")
        results[K] = calibrate_one_K(K, lo, hi, args.gpu, args.out_dir, d_model=args.d_model,
                                       dry_run=args.dry_run, simulate=args.simulate)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_name = "CALIBRATION_SUMMARY_SIMULATED.json" if args.simulate else "CALIBRATION_SUMMARY.json"
    summary_path = os.path.join(args.out_dir, summary_name)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {summary_path}")
    for K, r in results.items():
        print(f"  K={K}: alpha={r['alpha']} rho={r['final_rho']:.4f} converged={r['converged']} "
              f"n_iterations={r['n_iterations']} simulated={r['simulated']}")


if __name__ == "__main__":
    main()
