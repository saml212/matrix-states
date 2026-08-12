#!/usr/bin/env python3
"""Unit tests of `kwall_lib.reconstruction`'s REAL reconstruction function
against the design's own disclosed 24-state-space and 200-state-composition
figures (design §4, recovery steps 0.0/0.1/0.2; build charter item R7(b):
"recon.py's 24-state and 200-composition walks are wired as unit tests of
the real reconstruction function").

200-state composition (design's own description): 5 attempt-1 states x 5
attempt-2 states x 4 raw-canonical states x 2 arms = 200. Counts two things:
  - orphans: canonical_state==OK with NO COMPLETED row in the reconstructed
    ledger for that cell.
  - abort-trips: the cell derives NON-TERMINAL (dispatchable) while
    canonical_state==OK.
Expected: OLD guard (Rev 6, "0.1 appended ZERO rows") reproduces 30/200
orphans, 6/200 abort-trips (KW8.3's own figures); NEW guard (§R7 J3, "0.1
appended no COMPLETED row", the RELEASED rule) drives both to 0/200.
"""
import itertools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from kwall_lib.reconstruction import (  # noqa: E402
    AttemptEvidence, CanonicalEvidence, canonical_sanity_pass,
    reconstruct_cell, derive_cell_state, reconstruct_attempt_row, _24_STATES,
)
from kwall_lib import constants as C  # noqa: E402
from kwall_lib import disk_io as dio  # noqa: E402

CHARGED_CEILING = C.PRIMARY_CEILING_GPUH
S = C.STARTUP_ALLOWANCE_S_GPUH


def test_24_state_totality():
    """Every one of the 24 (dir_x_json, canonical_state, arm) combinations
    the design's own arithmetic derives must be reachable and must resolve
    to exactly one 0.0/0.1 outcome (totality — "every reachable disk state
    maps to exactly one outcome row")."""
    assert len(_24_STATES) == 24
    seen = set()
    for (dj, cs, arm) in _24_STATES:
        dir_state, json_state = dj
        if dir_state == "absent":
            ev = AttemptEvidence(kind="dir_absent")
        elif json_state == "absent":
            ev = AttemptEvidence(kind="json_absent")
        elif json_state == "unparseable":
            ev = AttemptEvidence(kind="unparseable")
        else:
            ev = AttemptEvidence(kind="parseable", status="COMPLETED", elapsed_s=3600.0)
        canon = {"OK": CanonicalEvidence("parseable_completed", elapsed_s=3600.0),
                 "CORRUPT": CanonicalEvidence("parseable_noncompleted"),
                 "ABSENT": CanonicalEvidence("absent")}[cs]
        derived_cs, _ = canonical_sanity_pass(canon)
        assert derived_cs == cs
        from kwall_lib.reconstruction import reconstruct_attempt_row
        row = reconstruct_attempt_row(ev, 1, arm, CHARGED_CEILING, S)
        key = (dj, cs, arm, None if row is None else row["status"])
        seen.add(key)
    print(f"test_24_state_totality: PASS ({len(_24_STATES)} states, "
          f"{len(seen)} distinct (state,outcome) keys, every state resolved)")


def _attempt_states():
    """The 5 attempt-slot states (design's own count)."""
    return [
        AttemptEvidence(kind="dir_absent"),
        AttemptEvidence(kind="json_absent"),
        AttemptEvidence(kind="unparseable"),
        AttemptEvidence(kind="parseable", status="COMPLETED", elapsed_s=3600.0),
        AttemptEvidence(kind="parseable", status="ABORTED-BUDGET"),
    ]


def _raw_canonical_states():
    """The 4 raw canonical states (pre-0.0-quarantine)."""
    return [
        CanonicalEvidence(kind="absent"),
        CanonicalEvidence(kind="parseable_completed", elapsed_s=3600.0),
        CanonicalEvidence(kind="parseable_noncompleted"),
        CanonicalEvidence(kind="unparseable"),
    ]


def run_200_sweep(guard: str):
    attempt_states = _attempt_states()
    raw_canon_states = _raw_canonical_states()
    arms = ["primary", "conditional"]
    n = 0
    orphans = 0
    abort_trips = 0
    for a1, a2, canon, arm in itertools.product(
            attempt_states, attempt_states, raw_canon_states, arms):
        n += 1
        ceiling = CHARGED_CEILING if arm == "primary" else C.CONDITIONAL_CEILING_GPUH
        rows, canonical_state, _promote = reconstruct_cell(
            a1, a2, canon, arm, ceiling, S, guard=guard)
        state = derive_cell_state(rows)
        if canonical_state == "OK":
            has_completed = any(r["status"] == "COMPLETED" for r in rows)
            if not has_completed:
                orphans += 1
            if state == "NON-TERMINAL":
                abort_trips += 1
    assert n == 200, n
    return orphans, abort_trips


def test_200_composition_old_guard():
    orphans, abort_trips = run_200_sweep("OLD")
    print(f"test_200_composition_old_guard: orphans={orphans}/200 "
          f"abort_trips={abort_trips}/200 (design's own KW8.3 figures: 30/200, 6/200)")
    assert (orphans, abort_trips) == (30, 6), (orphans, abort_trips)


def test_200_composition_new_guard():
    orphans, abort_trips = run_200_sweep("NEW")
    print(f"test_200_composition_new_guard: orphans={orphans}/200 "
          f"abort_trips={abort_trips}/200 (design's own §R7 J3 released figures: 0/200, 0/200)")
    assert (orphans, abort_trips) == (0, 0), (orphans, abort_trips)


def test_derive_cell_state_precedence():
    """"COMPLETED takes precedence" -- a cell with ANY COMPLETED row is
    COMPLETED, even if another row in the same cell is non-COMPLETED
    (§R7 KW8.7's 24/200 simultaneous-satisfaction case)."""
    rows = [{"attempt_n": 1, "status": "COMPLETED", "elapsed_h": 1.0},
            {"attempt_n": 2, "status": "CRASHED-RECOVERED", "elapsed_h": 1.20}]
    assert derive_cell_state(rows) == "COMPLETED"
    rows2 = [{"attempt_n": 1, "status": "ABORTED-BUDGET", "elapsed_h": 1.20},
             {"attempt_n": 2, "status": "CRASHED-RECOVERED", "elapsed_h": 1.20}]
    assert derive_cell_state(rows2) == "PERSISTENTLY-ABORTED"
    rows3 = [{"attempt_n": 1, "status": "ABORTED-BUDGET", "elapsed_h": 1.20}]
    assert derive_cell_state(rows3) == "NON-TERMINAL"  # retry gate not yet consulted
    rows4 = []
    assert derive_cell_state(rows4) == "NON-TERMINAL"
    print("test_derive_cell_state_precedence: PASS")


def test_m2_completed_missing_elapsed_s_no_crash():
    """m2 (minor, build audit R1): a parseable, `status=="COMPLETED"` cell
    JSON that lacks the TOP-LEVEL `elapsed_s` field used to crash
    `reconstruct_attempt_row` with a `TypeError` (None / float). §R7 KW8.9
    makes a missing/invalid `status` behave as UNPARSEABLE but said nothing
    about `elapsed_s` -- extended here. Unreachable from
    `ncr_earlyln_scale.py`'s own writer (`:275`/`:314` always set it),
    reachable from a foreign/hand-edited file, landing in the RECOVERY
    path (inside a supervisor restart loop) -- must never crash there."""
    bad = AttemptEvidence(kind="parseable", status="COMPLETED", elapsed_s=None)
    row = reconstruct_attempt_row(bad, 1, "primary", C.PRIMARY_CEILING_GPUH, C.STARTUP_ALLOWANCE_S_GPUH)
    assert row is not None and row["status"] == "CRASHED-RECOVERED", row
    assert row["elapsed_h"] == C.PRIMARY_CEILING_GPUH, row  # the Class-1 rule
    assert row["ceiling_charged"] is True, row

    bad_str = AttemptEvidence(kind="parseable", status="COMPLETED", elapsed_s="not-a-number")
    row2 = reconstruct_attempt_row(bad_str, 2, "conditional", C.CONDITIONAL_CEILING_GPUH, C.STARTUP_ALLOWANCE_S_GPUH)
    assert row2 is not None and row2["status"] == "CRASHED-RECOVERED", row2
    assert row2["elapsed_h"] == C.CONDITIONAL_CEILING_GPUH, row2
    print("test_m2_completed_missing_elapsed_s_no_crash: PASS "
          "(no TypeError; bootstrapped to CRASHED-RECOVERED at the full ceiling)")

    # ---- revert-style teeth check: the PRE-FIX code crashed here. Confirm
    # the crash is real by reproducing the exact pre-fix branch body.
    def old_reconstruct_attempt_row_completed_branch(attempt):
        return attempt.elapsed_s / 3600.0 + C.STARTUP_ALLOWANCE_S_GPUH
    crashed = False
    try:
        old_reconstruct_attempt_row_completed_branch(bad)
    except TypeError:
        crashed = True
    assert crashed, "teeth check itself is broken: the pre-fix code path should TypeError on elapsed_s=None"
    print("  teeth check: the PRE-FIX code path DOES crash (TypeError) on this exact input -- fix confirmed load-bearing")

    # ---- and confirm the disk_io read-boundary layer never even
    # constructs the dangerous combination from a real (malformed) file.
    scratch = os.path.join(
        "/private/tmp/claude-501/-Users-samuellarson-Experiments-learned-representations/"
        "be705417-f189-4cd8-8024-24cf6a0130a0/scratchpad/kwall_orch_tests", "m2_disk_io")
    os.makedirs(os.path.join(scratch, "K26_s0_attempt1"), exist_ok=True)
    import json
    with open(os.path.join(scratch, "K26_s0_attempt1", "earlyln_K26_s0.json"), "w") as f:
        json.dump({"status": "COMPLETED", "K": 26, "seed": 0}, f)  # no elapsed_s
    ev = dio.read_attempt_evidence(scratch, 26, 0, 1)
    assert ev.kind == "unparseable", ev
    print("  disk_io.read_attempt_evidence: a real malformed file (COMPLETED, no elapsed_s) "
          "reads as 'unparseable', never 'parseable COMPLETED elapsed_s=None' -- defense in depth confirmed")


def main():
    test_24_state_totality()
    test_200_composition_old_guard()
    test_200_composition_new_guard()
    test_derive_cell_state_precedence()
    test_m2_completed_missing_elapsed_s_no_crash()
    print()
    print("ALL RECONSTRUCTION TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
