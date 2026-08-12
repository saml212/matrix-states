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
    reconstruct_cell, derive_cell_state, _24_STATES,
)
from kwall_lib import constants as C  # noqa: E402

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


def main():
    test_24_state_totality()
    test_200_composition_old_guard()
    test_200_composition_new_guard()
    test_derive_cell_state_precedence()
    print()
    print("ALL RECONSTRUCTION TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
