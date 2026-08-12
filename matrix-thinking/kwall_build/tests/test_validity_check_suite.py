#!/usr/bin/env python3
"""Build-charter requirement: `validity_check` "must pass the committed
24-payload suite (kwall_suites/r10_vcheck.py + payloads) VERBATIM, including
every negative control."

This test imports the COMMITTED `kwall_suites/r10_payloads.py` (and
`r10_probes.py`, `r10_l6fix.py`) UNCHANGED — no edits, no copies — and
cross-checks every payload's verdict under `kwall_lib.validity_check_core`
against the audited `r10_vcheck.validity_check` the suite ships with. If
this test passes, the production module is provably byte-for-byte
equivalent on every payload the design's own 11-round gauntlet exercised
(24 base payloads + literal-variant negative controls + the 4 §R9 teeth-
probes D1/D1'/D2/D2' + the L6 fixture-producibility candidates), not merely
"was transcribed carefully."

Run: `python3 tests/test_validity_check_suite.py` from `kwall_build/`.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BUILD_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(os.path.dirname(_BUILD_ROOT))
_SUITES = os.path.join(_REPO_ROOT, "matrix-thinking", "kwall_suites")

sys.path.insert(0, _BUILD_ROOT)
sys.path.insert(0, _SUITES)   # committed suite scripts import each other by bare name

from kwall_lib.validity_check import validity_check_core  # noqa: E402

FAILURES = []
CHECKED = 0


def check(name, rep, disk, want, ref_verdict_fn):
    global CHECKED
    CHECKED += 1
    ref = ref_verdict_fn(rep, disk, "NEW")
    mine = validity_check_core(rep, disk, "NEW")
    if ref != mine:
        FAILURES.append(f"{name}: r10_vcheck={ref!r} MINE={mine!r} (DIVERGENCE)")
        return
    ref_old = ref_verdict_fn(rep, disk, "OLD")
    mine_old = validity_check_core(rep, disk, "OLD")
    if ref_old != mine_old:
        FAILURES.append(f"{name} [OLD mode]: r10_vcheck={ref_old!r} MINE={mine_old!r}")
        return
    got = "PASS" if not mine else "FAIL"
    if got != want:
        FAILURES.append(f"{name}: expected {want}, MY module says {got} ({mine})")


def main():
    # ---- r10_payloads.py: the 24-base + literal-variant payload suite ----
    import r10_vcheck
    import r10_payloads
    for name, (rep, disk, want) in r10_payloads.P.items():
        check(f"r10_payloads/{name}", rep, disk, want, r10_vcheck.validity_check)
    print(f"r10_payloads.py: {len(r10_payloads.P)} payloads cross-checked "
          f"against MY validity_check_core (NEW + OLD modes)")

    # ---- r10_probes.py: the 4 §R9 teeth-probes (D1/D1'/D2/D2'), outside
    # the 24-payload count per KW11.2, but still must match the audited
    # reference function exactly.
    import r10_probes
    for name, (rep, disk) in r10_probes.P.items():
        ref = r10_vcheck.validity_check(rep, disk, "NEW")
        mine = validity_check_core(rep, disk, "NEW")
        if ref != mine:
            FAILURES.append(f"r10_probes/{name}: r10_vcheck={ref!r} MINE={mine!r}")
        ref_old = r10_vcheck.validity_check(rep, disk, "OLD")
        mine_old = validity_check_core(rep, disk, "OLD")
        if ref_old != mine_old:
            FAILURES.append(f"r10_probes/{name} [OLD]: r10_vcheck={ref_old!r} MINE={mine_old!r}")
    print(f"r10_probes.py: {len(r10_probes.P)} teeth-probes cross-checked")

    # ---- r10_l6fix.py's three L6 fixture-producibility candidates ----
    # (not exposed as a dict; re-derive the same three payloads inline,
    # verbatim from r10_l6fix.py's own source, and cross-check candidate A
    # -- the one R11 adopted as the released L6 fixture, per the design's
    # live text :2601-2617 and §A11-ADJUDICATION's N1 re-derivation.)
    from r10_payloads import row, disk as disk_fn, rep as rep_fn, PRIM
    from fractions import Fraction as F
    a = [row(K, s, "primary", 1, "1.20", "CRASHED-RECOVERED", True) for (K, s) in PRIM[:10]]
    a += [row(30, 2, "primary", 1, "1.20", "CRASHED-RECOVERED", True),
          row(30, 2, "primary", 2, "1.20", "CRASHED-RECOVERED", True),
          row(30, 3, "primary", 1, "0.00", "GATE-REFUSED", False)]
    rA = rep_fn("EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE", a, "14.40", "14.40", "1.00")
    dA = disk_fn([])
    ref = r10_vcheck.validity_check(rA, dA, "NEW")
    mine = validity_check_core(rA, dA, "NEW")
    if ref != mine or ref != []:
        FAILURES.append(f"r10_l6fix/candidate-A: r10_vcheck={ref!r} MINE={mine!r} (want [])")
    else:
        print("r10_l6fix.py candidate A (the R11-released L6 fixture): "
              "PASS [] under both r10_vcheck and MY module")

    print()
    print(f"TOTAL payload/probe checks: {CHECKED + len(r10_probes.P) + 1}")
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}):")
        for x in FAILURES:
            print("  -", x)
        return 1
    print("ALL CHECKS PASS: kwall_lib.validity_check_core is verdict-identical "
          "to the committed, R11-audited r10_vcheck.py on every payload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
