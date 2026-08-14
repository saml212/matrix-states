"""Tests for band_partition.py -- re-verifies D5.2's partition stays
hole/double-fire-free AFTER M3's C-margin predicate is added (the charter's
own explicit requirement: "re-verify the partition stays hole/double-fire-
free after the addition -- re-run the outcome construction"). Extends
attack R4's own r4_8_bands.py methodology (18,018 constructed outcomes,
boundary-focused) with a new axis: CONTROL_C readings, swept including the
exact 0.15-margin boundary the same way R4 swept the 0.15 GAP boundary.

Run to completion, no torch needed.
"""
from __future__ import annotations

from band_partition import C_MARGIN_WIN, CHANCE_WIN_FLOOR, TAU, classify

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Test 1: M3's own counter-example, before and after the repair.
# ---------------------------------------------------------------------------
def test_M3_counter_example_closed():
    """The report's own worked counter-example: PRIMARY reading x=0.7100
    (compB anchors: fraction_closed = (0.71-0.0664)/0.9102 = 0.7071 >=
    0.70) scored WIN identically whether CONTROL_C read 0.0664 (supervision
    did everything) or 0.7000 (ortho removal alone did everything,
    supervision-specific increment only +0.0100). After M3's repair the
    two cases must diverge."""
    x = 0.7100
    gap = 0.20   # > GAP_WIN, assumed satisfied for this illustration

    r_good = classify(x, "compB", gap, control_c_reading=0.0664)
    check("M3 repair: supervision-did-everything case (C=0.0664) still scores WIN",
          r_good["label"] == "WIN", r_good)

    r_bad = classify(x, "compB", gap, control_c_reading=0.7000)
    check("M3 repair: ortho-removal-did-everything case (C=0.7000, margin=0.01<0.15) "
          "NO LONGER scores WIN -- the confound is closed",
          r_bad["label"] != "WIN", r_bad)
    check("M3 repair: the C=0.7000 case correctly falls to PARTIAL (not silently NULL either)",
          r_bad["label"] == "PARTIAL", r_bad)

    check("M3 repair: win_base_before_c_margin is IDENTICAL in both cases "
          "(confirms the OLD predicate really did fail to distinguish them)",
          r_good["win_base_before_c_margin"] == r_bad["win_base_before_c_margin"] is True)


# ---------------------------------------------------------------------------
# Test 2: exhaustive-ish sweep, compB (the recipe M3's predicate touches),
# including exact boundary values on every one of the four clauses.
# ---------------------------------------------------------------------------
def test_partition_no_hole_no_double_fire_compB():
    xs = [round(0.0 + 0.001 * i, 6) for i in range(0, 1201)]        # 0.000 .. 1.200, fine grid
    xs += [TAU, TAU + 1e-6, TAU - 1e-6, CHANCE_WIN_FLOOR, CHANCE_WIN_FLOOR + 1e-6, CHANCE_WIN_FLOOR - 1e-6]
    gaps = [0.10, 0.1499999, 0.15, 0.1500001, 0.20, 0.30]
    c_margins_offsets = [0.0, C_MARGIN_WIN - 1e-6, C_MARGIN_WIN, C_MARGIN_WIN + 1e-6, 0.30, -0.10]

    n_checked = 0
    n_hole = 0
    n_double = 0
    for x in xs:
        for gap in gaps:
            for c_off in c_margins_offsets:
                control_c = x - c_off     # so (x - control_c) == c_off exactly, sweeping the margin boundary directly
                r = classify(x, "compB", gap, control_c_reading=control_c)
                flags = int(r["null"]) + int(r["win"]) + int(r["partial"])
                n_checked += 1
                if flags == 0:
                    n_hole += 1
                elif flags > 1:
                    n_double += 1
    check(f"compB sweep: {n_checked} constructed outcomes, zero holes", n_hole == 0, f"n_hole={n_hole}")
    check(f"compB sweep: {n_checked} constructed outcomes, zero double-fires", n_double == 0, f"n_double={n_double}")


def test_partition_no_hole_no_double_fire_compA_primary():
    xs = [round(0.0 + 0.002 * i, 6) for i in range(0, 601)]
    xs += [TAU, TAU + 1e-6, TAU - 1e-6, CHANCE_WIN_FLOOR, CHANCE_WIN_FLOOR + 1e-6, CHANCE_WIN_FLOOR - 1e-6]
    gaps = [0.10, 0.1499999, 0.15, 0.1500001, 0.20, 0.30]
    n_checked = 0
    n_hole = 0
    n_double = 0
    for recipe in ("compA", "primary"):
        for x in xs:
            for gap in gaps:
                r = classify(x, recipe, gap, control_c_reading=None)
                flags = int(r["null"]) + int(r["win"]) + int(r["partial"])
                n_checked += 1
                if flags == 0:
                    n_hole += 1
                elif flags > 1:
                    n_double += 1
    check(f"compA/primary sweep: {n_checked} constructed outcomes, zero holes", n_hole == 0, f"n_hole={n_hole}")
    check(f"compA/primary sweep: {n_checked} constructed outcomes, zero double-fires", n_double == 0, f"n_double={n_double}")


def test_compA_primary_disclosure_flag_fires_exactly_on_win():
    """compA/primary carry no matched CONTROL_C -- every WIN there must be
    labeled ortho_confounded_disclosed=True, and NEVER on a non-WIN."""
    cases = [
        (0.80, "compA", 0.20),   # should WIN (fraction_closed way above 0.70)
        (0.05, "compA", 0.20),   # should NULL
        (0.40, "compA", 0.20),   # should PARTIAL
    ]
    for x, recipe, gap in cases:
        r = classify(x, recipe, gap)
        check(f"{recipe} x={x}: ortho_confounded_disclosed matches win exactly",
              r["ortho_confounded_disclosed"] == r["win"], r)


def test_compB_requires_control_c():
    try:
        classify(0.71, "compB", 0.20, control_c_reading=None)
        ok = False
    except AssertionError:
        ok = True
    check("compB: classify() REFUSES to run without control_c_reading (M3's repair has teeth)", ok)


if __name__ == "__main__":
    test_M3_counter_example_closed()
    test_partition_no_hole_no_double_fire_compB()
    test_partition_no_hole_no_double_fire_compA_primary()
    test_compA_primary_disclosure_flag_fires_exactly_on_win()
    test_compB_requires_control_c()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        raise SystemExit(1)
    print("ALL TESTS PASSED")
