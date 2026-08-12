"""Ledger reconstruction from disk (design §4, recovery procedure steps
0.0/0.1/0.2) and the derived-CELL-state rule ("COMPLETED takes precedence...
PERSISTENTLY-ABORTED iff...").

This module is written as PURE functions over abstract on-disk "evidence"
descriptors (`AttemptEvidence`/`CanonicalEvidence`) so the same code can be
(a) exhaustively unit-tested against the design's own disclosed 24-state /
200-state-composition figures (see `tests/test_reconstruction.py`) and
(b) driven by `disk_io.py`'s real-filesystem readers in production, without
duplicating the reconstruction RULES in two places.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AttemptEvidence:
    """One attempt-slot's on-disk evidence. `kind` in:
      - "dir_absent"      : the attempt{n}/ directory does not exist at all
                             (this design's dispatch path never ran it).
      - "json_absent"     : directory exists, JSON absent (mid-attempt crash
                             before any status write).
      - "unparseable"     : directory exists, JSON present but unparseable,
                             OR parseable with a missing/invalid `status`
                             key (§R7 KW8.9 — treated identically).
      - "parseable"       : directory exists, JSON parses; `status` is a
                             valid string. `elapsed_s` is required iff
                             `status == "COMPLETED"` (the subprocess's own
                             TOP-LEVEL `rec["elapsed_s"]`, never the nested
                             `rec["train"]["elapsed_s"]`, §R8 K7/KW9.11)."""
    kind: str
    status: Optional[str] = None
    elapsed_s: Optional[float] = None


@dataclass(frozen=True)
class CanonicalEvidence:
    """The RAW canonical-path file state, before the 0.0 quarantine pass.
    `kind` in {"absent", "parseable_completed", "parseable_noncompleted",
    "unparseable"} — 4 raw states, matching the design's own "4 raw-
    canonical states" in the 200-state composition sweep."""
    kind: str
    elapsed_s: Optional[float] = None


def canonical_sanity_pass(canon: CanonicalEvidence) -> tuple[str, bool]:
    """§4 recovery step 0.0. Returns (canonical_state, quarantined)."""
    if canon.kind == "absent":
        return "ABSENT", False
    if canon.kind == "parseable_completed":
        return "OK", False
    # parseable_noncompleted OR unparseable -> QUARANTINE-RENAME
    return "CORRUPT", True


def reconstruct_attempt_row(attempt: AttemptEvidence, n: int, arm: str,
                             charged_ceiling: float, s: float) -> Optional[dict]:
    """§4 recovery step 0.1, ONE row of the 11-row table (a single attempt
    slot, independent of the sibling slot and of `canonical_state` — the
    table's OWN row selection depends on `canonical_state` only through
    the PROMOTE decision, which `reconstruct_cell` handles separately, per
    the design's own "0.1... totality" framing: this attempt's own row
    shape is a function of `(attempt, arm)` alone; only whether a PROMOTE
    copy also happens depends on `canonical_state`)."""
    if attempt.kind == "dir_absent":
        return None  # this attempt slot never ran
    if attempt.kind in ("json_absent", "unparseable"):
        return {"attempt_n": n, "elapsed_h": charged_ceiling,
                "status": "CRASHED-RECOVERED", "outdir": None,
                "ceiling_charged": True}
    if attempt.kind == "parseable":
        if attempt.status == "COMPLETED":
            if not isinstance(attempt.elapsed_s, (int, float)):
                # m2 (minor, build audit R1): a parseable, status=="COMPLETED"
                # record whose TOP-LEVEL `elapsed_s` is missing or invalid
                # cannot produce a measured elapsed_h -- §R7 KW8.9's own
                # precedent (a missing/invalid `status` -> UNPARSEABLE)
                # extended here rather than crashing with a TypeError.
                # Unreachable from `ncr_earlyln_scale.py`'s own writer
                # (`:275`/`:314` always set it); reachable only from a
                # foreign/hand-edited file, which is exactly what
                # UNPARSEABLE evidence already models.
                return {"attempt_n": n, "elapsed_h": charged_ceiling,
                        "status": "CRASHED-RECOVERED", "outdir": None,
                        "ceiling_charged": True}
            elapsed_h = attempt.elapsed_s / 3600.0 + s
            return {"attempt_n": n, "elapsed_h": elapsed_h, "status": "COMPLETED",
                    "outdir": "<canonical>", "ceiling_charged": False}
        return {"attempt_n": n, "elapsed_h": charged_ceiling,
                "status": attempt.status, "outdir": None, "ceiling_charged": True}
    raise ValueError(f"unknown attempt evidence kind: {attempt.kind!r}")


def reconstruct_cell(attempt1: AttemptEvidence, attempt2: AttemptEvidence,
                      canon: CanonicalEvidence, arm: str,
                      charged_ceiling: float, s: float,
                      guard: str = "NEW") -> tuple[list[dict], str, bool]:
    """§4 recovery steps 0.0 + 0.1 + 0.2 for one (K, seed) cell.

    `guard` selects the 0.2 bootstrap trigger condition:
      - "NEW" (§R7 J3, the RELEASED rule): fires iff 0.1 appended NO row
        with `status=="COMPLETED"`.
      - "OLD" (Rev 6, KW8.3's own defect, kept ONLY for the regression
        comparison the design's own R7 report ran): fires iff 0.1 appended
        ZERO rows at all.

    Returns (rows, canonical_state, promote_needed) — `promote_needed` is
    True iff a COMPLETED attempt row's PROMOTE copy is required because
    `canonical_state != "OK"` (the caller/disk_io driver performs the
    actual copy+atomic-rename; this module only decides whether it must
    happen)."""
    canonical_state, _quarantined = canonical_sanity_pass(canon)
    rows = []
    promote_needed = False
    for n, ev in ((1, attempt1), (2, attempt2)):
        row = reconstruct_attempt_row(ev, n, arm, charged_ceiling, s)
        if row is not None:
            rows.append(row)
            if row["status"] == "COMPLETED" and canonical_state != "OK":
                promote_needed = True

    has_completed_row = any(r["status"] == "COMPLETED" for r in rows)
    fires = (not has_completed_row) if guard == "NEW" else (len(rows) == 0)
    if fires and canonical_state != "ABSENT":
        bootstrap_n = max((r["attempt_n"] for r in rows), default=0) + 1
        if canonical_state == "OK":
            elapsed_h = canon.elapsed_s / 3600.0 + s
            rows.append({"attempt_n": bootstrap_n, "elapsed_h": elapsed_h,
                         "status": "COMPLETED", "outdir": "<canonical>",
                         "ceiling_charged": False})
            # canonical already OK -- no promote needed for the bootstrap row
        else:  # CORRUPT (already quarantined at 0.0)
            rows.append({"attempt_n": bootstrap_n, "elapsed_h": charged_ceiling,
                         "status": "CRASHED-RECOVERED", "outdir": None,
                         "ceiling_charged": True})
    return rows, canonical_state, promote_needed


def derive_cell_state(rows: list[dict]) -> str:
    """"COMPLETED takes precedence... PERSISTENTLY-ABORTED iff (no row is
    COMPLETED) and [attempt-2 row exists and non-COMPLETED, OR attempt-1
    row is non-COMPLETED, no attempt-2 row exists, and the retry gate
    closed it]."

    The "no attempt-2 row exists, and the retry gate closed it" disjunct
    is EXACTLY the attempt-1-`GATE-REFUSED` case: the dispatch loop's own
    text ("Refused -> append a GATE-REFUSED row... move to the next cell")
    means a refused FIRST attempt never gets a retry dispatched at all --
    no attempt-2 row of ANY kind (not even a refusal) is ever appended for
    it, so this is the ONLY way "attempt-1 non-COMPLETED, no attempt-2
    row" is ever reached as a TERMINAL state in normal operation (an
    attempt-1 ABORTED-BUDGET/CRASHED with no attempt-2 row YET is instead
    NON-TERMINAL -- a retry is still owed, subject to a live gate check
    the orchestrator's own dispatch loop performs, not this function).

    Returns one of "COMPLETED", "PERSISTENTLY-ABORTED" (attempt-2 row
    exists and non-COMPLETED, OR attempt-1 was itself GATE-REFUSED), or
    "NON-TERMINAL" (attempt-1 genuinely ran and failed with no attempt-2
    row yet, or no rows at all — a retry/fresh dispatch is still owed)."""
    if any(r["status"] == "COMPLETED" for r in rows):
        return "COMPLETED"
    if any(r["attempt_n"] == 2 for r in rows):
        return "PERSISTENTLY-ABORTED"
    a1 = next((r for r in rows if r["attempt_n"] == 1), None)
    if a1 is not None and a1["status"] == "GATE-REFUSED":
        return "PERSISTENTLY-ABORTED"
    return "NON-TERMINAL"


# ---------------------------------------------------------------------------
# The 24-state-space derivation (§4: "2 (attempt dir present/absent) x 3
# (attempt JSON parseable/unparseable/absent) x 3 (canonical_state OK/
# CORRUPT/ABSENT) x 2 (arm) = 36 nominal, minus the 12 impossible dir=absent
# x JSON-present combinations = 24 valid"). Verified here by direct
# enumeration, not merely cited.

def _enumerate_24_states():
    """Every reachable (attempt_dir, attempt_json, canonical_state, arm)
    combination, using canonical_state directly (post-0.0) rather than the
    4 raw pre-quarantine states (this is the STATE-SPACE count, distinct
    from the 200-state COMPOSITION sweep below which uses raw canonical
    states as its own axis)."""
    combos = []
    dir_json_states = [
        ("dir_absent", None),
        ("json_absent", None),
        ("unparseable", None),
        ("parseable", "COMPLETED"),
        ("parseable", "ABORTED-BUDGET"),
    ]
    # Collapse to the design's own 2x3 space for the state-COUNT check:
    # dir present/absent x JSON parseable/unparseable/absent, dropping the
    # "which non-COMPLETED status" distinction (irrelevant to row SHAPE).
    dir_x_json = [
        ("absent", "absent"),          # dir absent (JSON necessarily absent)
        ("present", "absent"),          # dir present, JSON absent
        ("present", "unparseable"),
        ("present", "parseable"),
    ]
    # dir=absent x JSON=parseable/unparseable are the 2 impossible combos
    # per dir-state (already excluded above by construction: "absent" dir
    # only pairs with "absent" JSON in this list) -- 4 valid dir x json
    # combos, not 6 (2x3), because "dir=absent x JSON=unparseable" and
    # "dir=absent x JSON=parseable" are structurally impossible and are
    # simply never listed, matching the design's "minus the 12 impossible
    # combinations" arithmetic (4 valid dir x json states, not 6).
    canonical_states = ["OK", "CORRUPT", "ABSENT"]
    arms = ["primary", "conditional"]
    for dj in dir_x_json:
        for cs in canonical_states:
            for arm in arms:
                combos.append((dj, cs, arm))
    return combos


_24_STATES = _enumerate_24_states()
assert len(_24_STATES) == 4 * 3 * 2 == 24, len(_24_STATES)
