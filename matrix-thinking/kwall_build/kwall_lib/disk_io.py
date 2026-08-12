"""Real-filesystem I/O primitives the orchestrator and its reconstruction
driver need: atomic JSON writes (`rn.atomic_write_json`'s own pattern,
`matrix-thinking/ncr/run_ncr.py:105-109` — tmp file + `os.replace`), the
canonical-path contract (G2), and the attempt-directory readers 0.0/0.1
consult.

No design RULES live here — only I/O. The rules live in
`reconstruction.py` (pure functions over `AttemptEvidence`/`CanonicalEvidence`)
and `classify.py`; this module's job is turning real files into those
evidence objects and back.
"""
from __future__ import annotations

import json
import os
import time

from . import constants as C
from .reconstruction import AttemptEvidence, CanonicalEvidence


def atomic_write_json(path: str, obj) -> None:
    """Byte-identical pattern to `run_ncr.py:105-109` — the SAME primitive
    the design names explicitly as what the orchestrator must reuse for
    `ORCHESTRATOR_LEDGER.json` (design §4, "Every ledger write is ATOMIC")."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def cell_id(K: int, seed: int) -> str:
    return f"earlyln_K{K}_s{seed}"


def attempt_dir(outdir: str, K: int, seed: int, attempt_n: int) -> str:
    return os.path.join(outdir, f"K{K}_s{seed}_attempt{attempt_n}")


def attempt_json_path(outdir: str, K: int, seed: int, attempt_n: int) -> str:
    return os.path.join(attempt_dir(outdir, K, seed, attempt_n), f"{cell_id(K, seed)}.json")


def canonical_path(outdir: str, K: int, seed: int) -> str:
    return os.path.join(outdir, f"{cell_id(K, seed)}.json")


def _read_json_or(path: str):
    """Returns (kind, rec_or_None) where kind in {"absent", "unparseable",
    "parseable"}."""
    if not os.path.exists(path):
        return "absent", None
    try:
        with open(path) as f:
            return "parseable", json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return "unparseable", None


def read_attempt_evidence(outdir: str, K: int, seed: int, attempt_n: int) -> AttemptEvidence:
    """§4 recovery step 0.1's own row-selection inputs, read from real
    disk. `os.makedirs(outdir, exist_ok=True)` (ncr_earlyln_scale.py:237)
    means a dispatched attempt always leaves its directory, so
    "directory absent" is sound evidence this design's own dispatch path
    never ran it (design's own caveat, KW7.15 — never "provably never
    ran" against every possible cause, only against this writer)."""
    adir = attempt_dir(outdir, K, seed, attempt_n)
    if not os.path.isdir(adir):
        return AttemptEvidence(kind="dir_absent")
    jpath = attempt_json_path(outdir, K, seed, attempt_n)
    kind, rec = _read_json_or(jpath)
    if kind == "absent":
        return AttemptEvidence(kind="json_absent")
    if kind == "unparseable":
        return AttemptEvidence(kind="unparseable")
    status = rec.get("status") if isinstance(rec, dict) else None
    # The cell script itself (`ncr_earlyln_scale.py`, :199/:305) can only
    # ever write "COMPLETED" or "ABORTED-BUDGET" to a cell JSON -- the
    # orchestrator-only statuses (GATE-REFUSED/CRASHED/CRASHED-RECOVERED/
    # STOPPED-BY-OPERATOR) never appear there. §R7 KW8.9: a parseable JSON
    # whose `status` is missing or anything else is treated as UNPARSEABLE.
    if status not in ("COMPLETED", "ABORTED-BUDGET"):
        return AttemptEvidence(kind="unparseable")
    if status == "COMPLETED":
        elapsed_s = rec.get("elapsed_s")  # TOP-LEVEL field, :302 (§R8 K7/KW9.11)
        # m2 (minor, build audit R1): §R7 KW8.9's own precedent (a
        # missing/invalid `status` -> UNPARSEABLE) extended to a missing or
        # non-numeric TOP-LEVEL `elapsed_s` on an otherwise-COMPLETED
        # record -- reconstruction cannot compute a measured elapsed_h
        # without it, so this is UNPARSEABLE evidence at the READ boundary
        # (never even constructing the invalid "parseable COMPLETED,
        # elapsed_s=None" combination `reconstruct_attempt_row` also now
        # guards defensively).
        if not isinstance(elapsed_s, (int, float)):
            return AttemptEvidence(kind="unparseable")
        return AttemptEvidence(kind="parseable", status="COMPLETED", elapsed_s=elapsed_s)
    return AttemptEvidence(kind="parseable", status=status)


def read_canonical_evidence(outdir: str, K: int, seed: int) -> CanonicalEvidence:
    cpath = canonical_path(outdir, K, seed)
    kind, rec = _read_json_or(cpath)
    if kind == "absent":
        return CanonicalEvidence(kind="absent")
    if kind == "unparseable":
        return CanonicalEvidence(kind="unparseable")
    status = rec.get("status") if isinstance(rec, dict) else None
    if status == "COMPLETED":
        return CanonicalEvidence(kind="parseable_completed", elapsed_s=rec.get("elapsed_s"))
    return CanonicalEvidence(kind="parseable_noncompleted")


def quarantine_canonical(outdir: str, K: int, seed: int) -> None:
    """§4 recovery step 0.0: `os.rename(canonical_path, canonical_path +
    f".CORRUPT-{int(time.time())}")` — preserves the evidence on disk,
    renamed, no longer at `canonical_path` (so G2's exists-check can never
    trip against it on a later re-run)."""
    cpath = canonical_path(outdir, K, seed)
    if os.path.exists(cpath):
        os.rename(cpath, cpath + f".CORRUPT-{int(time.time())}")


def promote_to_canonical(outdir: str, K: int, seed: int, attempt_n: int) -> None:
    """§4 G2 / §R6 I2 PROMOTE: copy the attempt-dir JSON to
    `<canonical_path>.tmp`, then `os.replace` — the SAME atomic pattern
    live G2 acceptance uses (crash-mid-copy leaves only a harmless
    orphaned `.tmp`, never a truncated canonical file)."""
    src = attempt_json_path(outdir, K, seed, attempt_n)
    with open(src) as f:
        rec = json.load(f)
    dst = canonical_path(outdir, K, seed)
    atomic_write_json(dst, rec)


def canonical_exists(outdir: str, K: int, seed: int) -> bool:
    return os.path.exists(canonical_path(outdir, K, seed))


def canonical_status(outdir: str, K: int, seed: int):
    kind, rec = _read_json_or(canonical_path(outdir, K, seed))
    if kind == "parseable":
        return rec.get("status")
    return None


def list_canonical(outdir: str) -> list[dict]:
    """G2's own flat, non-recursive glob (`harvest()`'s `discover_seeds_by_K`
    pattern, `ncr_earlyln_scale.py:358-380`), reused here for
    `validity_check`'s disk view and for the primary/conditional canonical
    counts the trigger/band procedures need."""
    out = []
    if not os.path.isdir(outdir):
        return out
    for name in sorted(os.listdir(outdir)):
        if not name.startswith("earlyln_K") or not name.endswith(".json"):
            continue
        if name.endswith(".axis_c_lock.json"):
            continue
        kind, rec = _read_json_or(os.path.join(outdir, name))
        if kind == "parseable" and isinstance(rec, dict):
            out.append({"K": rec.get("K"), "seed": rec.get("seed"), "status": rec.get("status")})
        else:
            out.append({"K": None, "seed": None, "status": None})
    return out
