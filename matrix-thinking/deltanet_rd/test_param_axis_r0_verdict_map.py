"""test_param_axis_r0_verdict_map.py -- forced-fail regression fixtures for
param_axis_r0_betafit.py's map_verdict() / _conservative_combine().

Standalone, assert-based (no pytest dependency -- matches this tree's
existing self-contained-script convention, e.g. test_premise_valid_fixtures.py).
Run directly:

    python3 test_param_axis_r0_verdict_map.py

Context (audit finding, adjudicated PARAM_AXIS_SCALING_DESIGN.md §34, record-
first, commit preceding the fix this file pins): when the clean sub-ladder
A64 beta-hat read RISES but the confounded 3-rung beta-hat read INDETERMINATE
(or DECLINES), map_verdict() set FINAL_VERDICT solely from the clean fit +
Delta64's sign and only APPENDED an advisory line -- FINAL_VERDICT itself
still read "RISES / ATTRIBUTED..." instead of the pre-registered WITHHELD
headline (DSTATE_CONFOUND_PREREG.md §5 / PARAM_AXIS_SCALING_DESIGN.md §33.1:
the confounded fit is verdict-WITHHOLDING ONLY -- it may never grant what the
clean fit withholds; on disagreement, take the more conservative reading).
TEST 6 below is that forced-fail case: it goes RED without the
_conservative_combine fix and GREEN with it. TEST 6b pins the symmetric
agreement case (must NOT regress into always-withholding).
"""
from __future__ import annotations

from param_axis_r0_betafit import map_verdict, _conservative_combine

FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got!r} want={want!r}")
    if not ok:
        FAILURES.append(name)


def check_true(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


GATE_A_PASS = {"passes": True}
DELTA64_POS = {"ci": [0.001, 0.010]}   # Delta64 CI excludes 0, positive -> d_sig_pos
DELTA64_NEG = {"ci": [-0.010, -0.001]}  # Delta64 CI excludes 0, negative -> d_sig_neg
DELTA64_INDET = {"ci": [-0.003, 0.004]}  # Delta64 CI includes 0


# ---------------------------------------------------------------------------
# Part 1: _conservative_combine unit fixtures (the mechanical rank-based veto)
# ---------------------------------------------------------------------------

def part1_conservative_combine():
    print("\n--- Part 1: _conservative_combine unit fixtures ---")

    # TEST 6 (the audit's original forced-fail): clean RISES, confounded
    # INDETERMINATE -> confounded is strictly MORE conservative -> downgrade.
    check("clean=RISES, confounded=INDETERMINATE -> (INDETERMINATE, downgraded)",
          _conservative_combine("RISES", "INDETERMINATE"), ("INDETERMINATE", True))

    # Symmetric: clean DECLINES, confounded INDETERMINATE -> also downgrades
    # (the rule is direction-agnostic, not a RISES-only special case).
    check("clean=DECLINES, confounded=INDETERMINATE -> (INDETERMINATE, downgraded)",
          _conservative_combine("DECLINES", "INDETERMINATE"), ("INDETERMINATE", True))

    # Agreement: both RISES -> no downgrade, clean grade stands.
    check("clean=RISES, confounded=RISES -> (RISES, not downgraded)",
          _conservative_combine("RISES", "RISES"), ("RISES", False))

    # Agreement: both DECLINES -> no downgrade.
    check("clean=DECLINES, confounded=DECLINES -> (DECLINES, not downgraded)",
          _conservative_combine("DECLINES", "DECLINES"), ("DECLINES", False))

    # Sign disagreement at equal rank (RISES vs DECLINES): a substantive
    # contradiction is not an agreement -> escalate to INDETERMINATE.
    check("clean=RISES, confounded=DECLINES -> (INDETERMINATE, downgraded)",
          _conservative_combine("RISES", "DECLINES"), ("INDETERMINATE", True))
    check("clean=DECLINES, confounded=RISES -> (INDETERMINATE, downgraded)",
          _conservative_combine("DECLINES", "RISES"), ("INDETERMINATE", True))

    # Never GRANT: clean already INDETERMINATE, confounded RISES -> confounded
    # cannot upgrade a withheld clean verdict.
    check("clean=INDETERMINATE, confounded=RISES -> (INDETERMINATE, not downgraded)",
          _conservative_combine("INDETERMINATE", "RISES"), ("INDETERMINATE", False))

    # No confounded fit available -> clean stands untouched (n_days-missing
    # ladder / cells not run -- regression guard for the pre-existing
    # no-confounded-data path).
    check("clean=RISES, confounded=n/a -> (RISES, not downgraded)",
          _conservative_combine("RISES", "n/a"), ("RISES", False))
    check("clean=RISES, confounded=None -> (RISES, not downgraded)",
          _conservative_combine("RISES", None), ("RISES", False))

    # Defensive future-proofing: VOID/FLOOR are not reachable via factor1()'s
    # current return set, but the rank table must still combine correctly if
    # that ever changes (docstring/table completeness -- §34's ordering is
    # pinned to include them).
    check("clean=RISES, confounded=VOID -> (VOID, downgraded)",
          _conservative_combine("RISES", "VOID"), ("VOID", True))
    check("clean=INDETERMINATE, confounded=VOID -> (VOID, downgraded)",
          _conservative_combine("INDETERMINATE", "VOID"), ("VOID", True))
    check("clean=RISES, confounded=FLOOR -> (FLOOR, downgraded)",
          _conservative_combine("RISES", "FLOOR"), ("FLOOR", True))
    check("clean=VOID, confounded=RISES -> (VOID, not downgraded)",
          _conservative_combine("VOID", "RISES"), ("VOID", False))


# ---------------------------------------------------------------------------
# Part 2: map_verdict end-to-end fixtures (FINAL_VERDICT string + lines)
# ---------------------------------------------------------------------------

def part2_map_verdict():
    print("\n--- Part 2: map_verdict end-to-end fixtures ---")

    # TEST 6 forced-fail, end-to-end: FINAL_VERDICT must be the WITHHELD /
    # conservative string, NOT "RISES / ATTRIBUTED...". This is the exact
    # case the audit flagged; it goes RED without the fix.
    v, lines = map_verdict("RISES", DELTA64_POS, "INDETERMINATE", GATE_A_PASS)
    check_true("TEST6: FINAL_VERDICT starts with INDETERMINATE (withheld, not RISES)",
               v.startswith("INDETERMINATE"), detail=v)
    check_true("TEST6: FINAL_VERDICT does NOT read a bare RISES headline",
               not v.startswith("RISES"), detail=v)
    check_true("TEST6: FINAL_VERDICT cites §34 / DSTATE §5 / §33.1",
               ("§34" in v) and ("§5" in v or "§33.1" in v), detail=v)
    check_true("TEST6: clean-RISES + confounded-INDETERMINATE disclosed in verdict_lines",
               any("clean" in ln.lower() and "RISES" in ln and "INDETERMINATE" in ln for ln in lines),
               detail=repr(lines))

    # TEST 6b: symmetric agreement case MUST still yield the granted RISES
    # headline -- pins that the fix doesn't over-correct into always-withhold.
    v2, lines2 = map_verdict("RISES", DELTA64_POS, "RISES", GATE_A_PASS)
    check_true("TEST6b: agreement (clean=RISES, confounded=RISES) -> RISES headline stands",
               v2.startswith("RISES / ATTRIBUTED"), detail=v2)

    # DECLINES downgrade (symmetric direction, not just RISES):
    v3, lines3 = map_verdict("DECLINES", DELTA64_INDET, "INDETERMINATE", GATE_A_PASS)
    check_true("DECLINES+INDETERMINATE-confound -> withheld, not 'DECLINES / ATTRIBUTED'",
               v3.startswith("INDETERMINATE") and not v3.startswith("DECLINES"), detail=v3)

    # DECLINES agreement -> DECLINES a-fortiori headline stands.
    v4, lines4 = map_verdict("DECLINES", DELTA64_INDET, "DECLINES", GATE_A_PASS)
    check_true("DECLINES+DECLINES-confound (agree) -> DECLINES/ATTRIBUTED a-fortiori stands",
               v4.startswith("DECLINES / ATTRIBUTED a-fortiori"), detail=v4)

    # Sign-conflict downgrade: clean RISES, confounded DECLINES.
    v5, lines5 = map_verdict("RISES", DELTA64_POS, "DECLINES", GATE_A_PASS)
    check_true("RISES+DECLINES-confound (sign conflict) -> withheld, not RISES",
               v5.startswith("INDETERMINATE") and not v5.startswith("RISES"), detail=v5)

    # No confounded fit ("n/a") -> original clean-only behavior preserved
    # (regression guard: rung 392M cells not yet run / not admissible).
    v6, lines6 = map_verdict("RISES", DELTA64_POS, "n/a", GATE_A_PASS)
    check_true("clean=RISES, confounded=n/a -> RISES/ATTRIBUTED headline (unaffected by veto)",
               v6.startswith("RISES / ATTRIBUTED"), detail=v6)

    # Never-grant regression: clean already INDETERMINATE, confounded RISES
    # -> must stay INDETERMINATE (no upgrade), and NOT go through the
    # downgrade-disclosure branch (nothing was vetoed -- there was nothing
    # to grant in the first place).
    v7, lines7 = map_verdict("INDETERMINATE", DELTA64_INDET, "RISES", GATE_A_PASS)
    check("clean=INDETERMINATE, confounded=RISES -> unchanged INDETERMINATE text (never GRANT)",
          v7, "INDETERMINATE (clean A₆₄ slope CI includes 0, or FLOOR/VOID; FLAT unavailable at n_seeds=1)")

    # CONTRADICTED path (clean RISES, Delta64 negative) must still fire when
    # the confounded fit agrees/absent -- untouched by the veto logic.
    v8, lines8 = map_verdict("RISES", DELTA64_NEG, "RISES", GATE_A_PASS)
    check_true("clean=RISES + Delta64<0 + confounded agrees -> CONTRADICTED (Delta64 math untouched)",
               v8.startswith("CONTRADICTED"), detail=v8)

    # GATE-A-fails passthrough (unrelated branch) must be untouched.
    v9, lines9 = map_verdict("RISES", DELTA64_POS, "RISES", {"passes": False})
    check_true("GATE-A fails -> unchanged passthrough text",
               v9.startswith("INDETERMINATE (A₆₄ GATE-A fails"), detail=v9)


if __name__ == "__main__":
    part1_conservative_combine()
    part2_map_verdict()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        raise SystemExit(1)
    print("ALL FIXTURES PASS")
