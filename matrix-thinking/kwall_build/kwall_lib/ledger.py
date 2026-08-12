"""The orchestrator ledger: schema, atomic persistence, and the recovery
(read-or-reconstruct) entry point (design §4, "Ledger persistence +
write-ahead recovery" — G1).
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

from . import constants as C
from . import disk_io as dio
from .reconstruction import (
    AttemptEvidence, CanonicalEvidence, reconstruct_cell, derive_cell_state,
    canonical_sanity_pass,
)


def empty_ledger() -> dict:
    return {"realized_gpu_h": 0.0, "attempts": [], "open_attempt": None}


def load_ledger(path: str) -> Optional[dict]:
    """Returns the parsed ledger, or None if missing/unparseable (the
    caller must then run `reconstruct_ledger_from_disk`, design §4 step 0)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def save_ledger(path: str, ledger: dict) -> None:
    dio.atomic_write_json(path, ledger)


def all_cell_specs():
    """Every (K, seed, arm) this design can ever dispatch -- 12 primary +
    4 conditional (per whichever K_trig it eventually resolves to; the
    reconstruction walk iterates the primary 12 UNCONDITIONALLY plus a
    conditional 4 UNCONDITIONALLY, never gated on the conditional
    canonical directory's existence, §R6 I1 closing KW7.1(b))."""
    specs = [(K, s, "primary") for K in C.K_VALUES_PRIMARY for s in C.SEEDS]
    return specs


def reconstruct_ledger_from_disk(primary_outdir: str, conditional_outdir: str,
                                  conditional_K: Optional[int] = None) -> dict:
    """§4 recovery step 0: CONSERVATIVE RECONSTRUCTION FROM DISK,
    PER-ATTEMPT-DIRECTORY, run when the ledger is missing or unparseable.
    Iterates the primary 12 unconditionally; the conditional 4 only if
    `conditional_K` is given (the orchestrator itself always knows this
    from its own prior trigger evaluation once one has happened — on a
    FIRST-EVER start with no ledger and no trigger yet evaluated, there is
    no conditional tree to scan at all, so `conditional_K=None` is correct
    there too)."""
    ledger = empty_ledger()
    rows_all = []

    def _do_cell(outdir, K, seed, arm, ceiling):
        canon_ev = dio.read_canonical_evidence(outdir, K, seed)
        canonical_state, quarantined = canonical_sanity_pass(canon_ev)
        if quarantined:
            dio.quarantine_canonical(outdir, K, seed)
        a1 = dio.read_attempt_evidence(outdir, K, seed, 1)
        a2 = dio.read_attempt_evidence(outdir, K, seed, 2)
        rows, canonical_state2, promote_needed = reconstruct_cell(
            a1, a2, canon_ev, arm, ceiling, C.STARTUP_ALLOWANCE_S_GPUH, guard="NEW")
        # PROMOTE any row that needs it (COMPLETED attempt row whose
        # canonical isn't already OK) -- §R6 I2, run for EVERY such row,
        # not just the first, since 0.1 can promote from either slot.
        for row in rows:
            if row["status"] == "COMPLETED" and row.get("outdir") == "<canonical>" \
                    and canonical_state2 != "OK":
                # find which attempt_n produced this row and promote it
                dio.promote_to_canonical(outdir, K, seed, row["attempt_n"])
                canonical_state2 = "OK"
        for row in rows:
            row["K"], row["seed"], row["arm"] = K, seed, arm
            row["d_override"] = K + 1
            if row.get("outdir") == "<canonical>":
                row["outdir"] = dio.canonical_path(outdir, K, seed)
        return rows

    for K, seed, arm in all_cell_specs():
        rows_all.extend(_do_cell(primary_outdir, K, seed, arm, C.PRIMARY_CEILING_GPUH))
    if conditional_K is not None:
        for seed in C.SEEDS:
            rows_all.extend(_do_cell(conditional_outdir, conditional_K, seed,
                                     "conditional", C.CONDITIONAL_CEILING_GPUH))

    ledger["attempts"] = rows_all
    ledger["realized_gpu_h"] = sum(r["elapsed_h"] for r in rows_all)
    ledger["open_attempt"] = None
    return ledger


def cell_rows(ledger: dict, K: int, seed: int, arm: str) -> list[dict]:
    return [a for a in ledger["attempts"]
            if a["K"] == K and a["seed"] == seed and a["arm"] == arm]


def cell_state(ledger: dict, K: int, seed: int, arm: str) -> str:
    return derive_cell_state(cell_rows(ledger, K, seed, arm))


def append_row(ledger: dict, row: dict) -> None:
    ledger["attempts"].append(row)
    ledger["realized_gpu_h"] += row["elapsed_h"]
