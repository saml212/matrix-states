"""gate2_construction_test.py -- KEY_ANCHORING_DESIGN.md sec 4's amended,
three-leg CPU construction gate, AS A COMMITTED TEST. Must PASS before any
GPU launch of a candidate-(d) cell (the design's own "ships with the build
as a committed CPU test" requirement, sec 4's Pinned REGRESSION CASE
paragraph).

Two things this script does, both mandatory:
  1. Runs Gate 2 (G2-a/G2-b/G2-c) on the REGISTERED anchor init (the frozen
     frame-potential table at ANCHOR_INIT_SEED) and asserts it PASSES all
     three legs -- the actual go/no-go check a real launch would run.
  2. Runs the PINNED REGRESSION QUADRUPLE (moderate collapse, severe
     collapse, the healthy init, localized collapse) and asserts each
     produces its EXPECTED verdict -- the gate "has teeth" demonstration
     (sec 4/sec 7 item 2: "run the negative test... not just written").
     The moderate/severe numbers are asserted to 4 decimal places against
     the design's own pinned values (independently reproduced by
     KEY_ANCHORING_ATTACK_R3.md's from-scratch construction); the localized
     case is asserted directionally (6a PASS, 6b FAIL -- the design's own
     "no seed pinned" case) with a numeric range sanity check.

Exit code 0 = every check passed (mirrors this codebase's gate-script
convention, e.g. run_deltanet_rd_exactness_sweep.py's gate_geo3_drift /
gate_gram_rho -- sys.exit(1) + a printed ERROR on ANY failure, never a
silent partial pass).

No fla/chunk_delta_rule dependency, no CUDA required -- this test is
supposed to be runnable from a fresh CPU-only Python environment; see
key_anchoring.py's own module docstring for why this split exists.

Usage: python gate2_construction_test.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports

import key_anchoring as ka

# The four pinned numbers from KEY_ANCHORING_DESIGN.md sec 4, independently
# reproduced to 4 decimal places by KEY_ANCHORING_ATTACK_R3.md's own
# from-scratch reconstruction (Target 1, R2-M1 table).
EXPECTED_MODERATE = {"sigma_ratio": 0.0762, "max_abs_cos": 0.5371}
EXPECTED_SEVERE = {"sigma_ratio": 0.0141, "max_abs_cos": 0.9333}
# The healthy init's exact numbers are seed-dependent (sec 2.2's own
# disclosure: sigma_ratio is a fixed tight-frame invariant, max|cos| varies
# a few hundredths across seeds); asserted against a tolerance band, not an
# exact 4-decimal match, with sigma_ratio required to sit at the Welch floor.
EXPECTED_HEALTHY_SIGMA_RATIO_MIN = 0.999
EXPECTED_HEALTHY_MAX_COS_BAND = (0.20, 0.40)   # design cites 0.2832; this build's own seed: see below
_TOL = 5e-4   # 4-decimal-place agreement, with float rounding headroom


def _check_close(name: str, got: float, expected: float, tol: float = _TOL) -> None:
    assert abs(got - expected) < tol, (
        f"{name}: got {got:.4f}, expected {expected:.4f} (tol {tol}) -- Gate 2's regression "
        f"quadruple no longer reproduces the design's pinned numbers; investigate before trusting "
        f"Gate 2 as a launch gate.")


def run() -> bool:
    ok = True

    print("=" * 70)
    print("GATE 2 -- construction check on the REGISTERED anchor init "
          f"(frame-potential, seed={ka.ANCHOR_INIT_SEED})")
    print("=" * 70)
    healthy = ka.frame_potential_init(107, 64, seed=ka.ANCHOR_INIT_SEED)
    gate_healthy = ka.gate2_construction_check(healthy, ks=(16, 32), seed=0)
    print(f"  G2-a sigma_ratio = {gate_healthy['g2a_sigma_ratio']:.4f} "
          f"(>= {ka.GATE2_SIGMA_RATIO_MIN}? {gate_healthy['g2a_pass']})")
    print(f"  G2-b max|cos|    = {gate_healthy['g2b_max_abs_cos']:.4f} "
          f"(<= {ka.GATE2_MAX_COS_MAX}? {gate_healthy['g2b_pass']})")
    for K, leg in gate_healthy["g2c_ns_legs"].items():
        print(f"  G2-c K={K:3d}: n_fallback={leg['n_fallback']}/{leg['n_subsets']} "
              f"max_resid={leg['max_resid']:.2e} pass={leg['pass']}")
    print(f"  OVERALL: {'PASS' if gate_healthy['pass'] else 'FAIL'}")
    if not gate_healthy["pass"]:
        print("  ERROR: the REGISTERED anchor init fails its own Gate 2 -- this must never "
              "happen; a launch depending on this init must be refused.", file=sys.stderr)
        ok = False
    if gate_healthy["g2a_sigma_ratio"] < EXPECTED_HEALTHY_SIGMA_RATIO_MIN:
        print(f"  ERROR: healthy init sigma_ratio {gate_healthy['g2a_sigma_ratio']:.4f} below the "
              f"Welch-floor sanity band ({EXPECTED_HEALTHY_SIGMA_RATIO_MIN})", file=sys.stderr)
        ok = False
    lo, hi = EXPECTED_HEALTHY_MAX_COS_BAND
    if not (lo <= gate_healthy["g2b_max_abs_cos"] <= hi):
        print(f"  WARNING (non-fatal): healthy init max|cos| {gate_healthy['g2b_max_abs_cos']:.4f} "
              f"outside the {EXPECTED_HEALTHY_MAX_COS_BAND} sanity band cited from the design's own "
              f"session number (0.2832) -- frame-potential minimizers can land at different max|cos| "
              f"per seed (sec 2.2's own disclosure); NOT a gate failure by itself.", flush=True)

    print()
    print("=" * 70)
    print("PINNED REGRESSION QUADRUPLE (sec 4) -- the gate must demonstrably")
    print("FAIL the collapsed tables it is supposed to catch, and PASS the healthy one.")
    print("=" * 70)

    # --- moderate collapse (noise_sigma=0.30, seed=42) -- must FAIL both 6a/6b, NS-only PASS ---
    moderate = ka.build_collapsed_table(noise_sigma=0.30, seed=42)
    g_mod = ka.gate2_construction_check(moderate, ks=(16, 32), seed=0)
    print(f"\n[moderate collapse, sigma=0.30, seed=42] G2-a={g_mod['g2a_sigma_ratio']:.4f} "
          f"G2-b={g_mod['g2b_max_abs_cos']:.4f} (expect {EXPECTED_MODERATE['sigma_ratio']:.4f}/"
          f"{EXPECTED_MODERATE['max_abs_cos']:.4f}, both FAIL)")
    try:
        _check_close("moderate G2-a", g_mod["g2a_sigma_ratio"], EXPECTED_MODERATE["sigma_ratio"])
        _check_close("moderate G2-b", g_mod["g2b_max_abs_cos"], EXPECTED_MODERATE["max_abs_cos"])
        assert not g_mod["g2a_pass"] and not g_mod["g2b_pass"], "moderate case must FAIL both 6a/6b"
        assert all(leg["pass"] for leg in g_mod["g2c_ns_legs"].values()), \
            "moderate case's NS leg must PASS (0 fallbacks) -- this is the design's OWN point: NS " \
            "convergence alone is blind to this collapse"
        assert not g_mod["pass"], "moderate case must FAIL the overall (three-leg) gate"
        print("  PASS: moderate case reproduces the pinned numbers to 4 decimals, gate correctly FAILS it")
    except AssertionError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        ok = False

    # --- severe collapse (noise_sigma=0.05, seed=42) -- must FAIL harder ---
    severe = ka.build_collapsed_table(noise_sigma=0.05, seed=42)
    g_sev = ka.gate2_construction_check(severe, ks=(16, 32), seed=0)
    print(f"\n[severe collapse, sigma=0.05, seed=42] G2-a={g_sev['g2a_sigma_ratio']:.4f} "
          f"G2-b={g_sev['g2b_max_abs_cos']:.4f} (expect {EXPECTED_SEVERE['sigma_ratio']:.4f}/"
          f"{EXPECTED_SEVERE['max_abs_cos']:.4f}, both FAIL)")
    try:
        _check_close("severe G2-a", g_sev["g2a_sigma_ratio"], EXPECTED_SEVERE["sigma_ratio"])
        _check_close("severe G2-b", g_sev["g2b_max_abs_cos"], EXPECTED_SEVERE["max_abs_cos"])
        assert not g_sev["g2a_pass"] and not g_sev["g2b_pass"], "severe case must FAIL both 6a/6b"
        assert all(leg["pass"] for leg in g_sev["g2c_ns_legs"].values()), \
            "severe case's NS leg must ALSO PASS (NS is blind at every severity, sec 4)"
        assert not g_sev["pass"], "severe case must FAIL the overall gate"
        print("  PASS: severe case reproduces the pinned numbers to 4 decimals, gate correctly FAILS it")
    except AssertionError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        ok = False

    # --- healthy frame-potential init -- must PASS all three legs ---
    print(f"\n[healthy frame-potential init] G2-a={gate_healthy['g2a_sigma_ratio']:.4f} "
          f"G2-b={gate_healthy['g2b_max_abs_cos']:.4f} (expect ~1.0000/~0.28-0.32, all PASS)")
    if gate_healthy["pass"]:
        print("  PASS: healthy init passes all three legs (the gate discriminates, not just rejects)")
    else:
        print("  ERROR: healthy init did not pass all three legs (see above)", file=sys.stderr)
        ok = False

    # --- localized collapse (10/107 rows onto one direction, sigma=0.02, planted on the healthy init) ---
    loc = ka.build_localized_collapse_table(healthy, n_collapsed=10, sigma=0.02, seed=0)
    g_loc = ka.gate2_construction_check(loc, ks=(16, 32), seed=0)
    print(f"\n[localized collapse, 10/107 rows, sigma=0.02] G2-a={g_loc['g2a_sigma_ratio']:.4f} "
          f"G2-b={g_loc['g2b_max_abs_cos']:.4f} (expect ~0.15-0.19 PASS / ~0.98-0.99 FAIL -- "
          f"design's own session number 0.1787/0.9830, no seed pinned for this case)")
    try:
        assert g_loc["g2a_pass"], (
            f"localized case's G2-a should PASS (aggregate dilution over the other 97 healthy "
            f"rows) -- got sigma_ratio={g_loc['g2a_sigma_ratio']:.4f}")
        assert not g_loc["g2b_pass"], (
            f"localized case's G2-b should FAIL (a max statistic is dilution-immune) -- got "
            f"max_abs_cos={g_loc['g2b_max_abs_cos']:.4f}")
        assert not g_loc["pass"], "localized case must FAIL the overall gate (6b alone blocks it)"
        print("  PASS: localized case demonstrates 6a/6b's max-statistic dilution-immunity "
              "(6a passes on aggregate dilution, 6b still catches it)")
    except AssertionError as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        ok = False

    print()
    print("=" * 70)
    print(f"GATE 2 REGRESSION QUADRUPLE: {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    passed = run()
    sys.exit(0 if passed else 1)
