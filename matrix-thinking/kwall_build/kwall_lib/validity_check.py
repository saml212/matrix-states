"""`validity_check` — the job-spec assertion the box's `queue_worker.sh`
runs after `cmd` exits (`bash -c "$vcheck"`; exit code decides
`completed/` vs `failed/`, never the raw exit code of `cmd`).

The CORE assertion logic below (`validity_check_core`) is a byte-for-byte
port of `matrix-thinking/kwall_suites/r10_vcheck.py`'s `validity_check`
function — audited CLEAR through round 11 (§A11-ADJUDICATION) and proven,
by that audit's own execution, to reproduce every one of the committed
`kwall_suites/` payload verdicts exactly (`tests/test_validity_check_suite.py`
runs the unmodified suite scripts against this port and asserts identical
results — the build charter's "must pass the committed 24-payload suite
VERBATIM" requirement). `mode` is always `"NEW"` in production — the `"OLD"`
branch exists only so the suite's own OLD-vs-NEW regression/flip check
still runs unmodified against this module.

This module ADDS the disk-reading glue the design's in-text `validity_check`
assumes but does not itself specify a Python implementation for: building
the `disk` dict (`primary_canonical`, `cond_canonical`) from the two real
canonical directories, and a CLI entry point (`validity_check_cli`) that
`orchestrator_report.json` + the two canonical dirs feed, exiting 0/1 for
`queue_worker.sh`."""
from __future__ import annotations

import glob
import json
import os
import sys
from fractions import Fraction as F

from . import constants as C

PRIMARY_CEIL = F(str(C.PRIMARY_CEILING_GPUH))
COND_CEIL = F(str(C.CONDITIONAL_CEILING_GPUH))
BAND_LABELS = C.BAND_LABELS
RESOLUTIONS = C.TRIGGER_RESOLUTIONS
ACCEPT = C.RUN_STATUS_ACCEPT_SET
EPS = F(1, 1000000)


def _prim_canon_count(disk):
    return len([f for f in disk["primary_canonical"] if f["status"] == "COMPLETED"])


def _prim_canon_count_K(disk, K):
    return len([f for f in disk["primary_canonical"]
                if f["K"] == K and f["status"] == "COMPLETED"])


def _cond_canon_completed(disk):
    return len([f for f in disk["cond_canonical"] if f["status"] == "COMPLETED"])


def _pairs(att, arm, status=None):
    return {(a["K"], a["seed"]) for a in att
            if a["arm"] == arm and (status is None or a["status"] == status)}


def validity_check_core(rep: dict, disk: dict, mode: str = "NEW") -> list[str]:
    """mode in {'OLD','NEW'} -> list of failure reasons ([] == PASS).
    Byte-for-byte port of `kwall_suites/r10_vcheck.py:validity_check`
    (identifiers renamed `f` kept identical for diffability against the
    audited original)."""
    f = []
    led = rep["ledger"]
    att = led["attempts"]
    realized = F(str(led["realized_gpu_h_final"]))

    # ---- universal 1
    if rep["run_status"] not in ACCEPT:
        f.append("U1: run_status %s not in accept-set" % rep["run_status"])
        return f                      # excluded outright; no branch fires
    # ---- universal 2
    if not realized <= F("15.50"):
        f.append("U2: realized %s > 15.50" % realized)
    # ---- universal 3
    tot = sum((F(str(a["elapsed_h"])) for a in att), F(0))
    if abs(realized - tot) > EPS:
        f.append("U3: realized %s != sum(elapsed_h) %s" % (realized, tot))
    # ---- universal 4
    if not all(a["K"] + 1 == a["d_override"] for a in att):
        f.append("U4: d_override != K+1 somewhere")
    # ---- universal 5
    if not all(v == "PASS" for v in rep["smoke"].values()):
        f.append("U5: smoke not all PASS")
    # ---- universal 6
    if rep["band"]["label"] not in BAND_LABELS:
        f.append("U6: band label %r" % rep["band"]["label"])
    if rep["trigger"]["resolution"] not in RESOLUTIONS:
        f.append("U6: trigger.resolution %r" % rep["trigger"]["resolution"])
    # ---- universal 7
    cond = rep["conditional"]
    n_cond_canon = _cond_canon_completed(disk)
    n_cond_led = len(_pairs(att, "conditional", "COMPLETED"))
    if cond is not None and cond["qualifier_band"] is not None:
        a_ok = (cond["launched"] is True and n_cond_canon == 4 and n_cond_led == 4)
        b_ok = (cond["launched"] is False and rep["trigger"]["K_trig"] == 32)
        if not (a_ok or b_ok):
            f.append("U7: qualifier_band with neither clause (a) nor (b)")
    elif (cond is not None and cond["qualifier_band"] is None
          and cond.get("launched") is True):
        if mode == "NEW":                       # R8 K1 mirror clause
            if n_cond_canon != n_cond_led:
                f.append("U7-mirror: cond_canon %d != ledger cond COMPLETED %d"
                         % (n_cond_canon, n_cond_led))
            if not n_cond_canon < 4:
                f.append("U7-mirror: cond_canon %d not < 4" % n_cond_canon)
    else:
        # conditional is None, or band None and launched False/absent
        if mode == "NEW":                       # R9 m3
            if n_cond_canon != 0:
                f.append("U7-otherwise: cond_canon %d != 0" % n_cond_canon)
    # ---- universal 8 (NEW only, R8 K3)
    if mode == "NEW":
        cvm = rep["charged_vs_measured"]
        ccgh_rec = sum((F(str(a["elapsed_h"])) for a in att if a["ceiling_charged"]), F(0))
        if abs(F(str(cvm["ceiling_charged_gpu_h"])) - ccgh_rec) > EPS:
            f.append("U8-ccgh: declared %s != recomputed %s"
                     % (cvm["ceiling_charged_gpu_h"], ccgh_rec))
        if realized > 0:
            frac_rec = ccgh_rec / realized
            if abs(F(str(cvm["ceiling_charged_fraction"])) - frac_rec) > EPS:
                f.append("U8-frac: declared %s != recomputed %s"
                         % (float(F(str(cvm['ceiling_charged_fraction']))), float(frac_rec)))

    # ---- per-run_status (exactly one branch)
    rs = rep["run_status"]
    band = rep["band"]
    if rs == "COMPLETE":
        if band["interval_resolved_Ks"] == [] and band["incomplete_at_K"] is None:
            # STRICT
            if _prim_canon_count(disk) != 12:
                f.append("COMPLETE/strict: primary canonical %d != 12"
                         % _prim_canon_count(disk))
            if mode == "NEW":                   # R9 m7
                missing = [(K, s) for K in (26, 28, 30) for s in range(4)
                           if (K, s) not in _pairs(att, "primary")]
                if missing:
                    f.append("COMPLETE/strict-m7-J1a: %d primary pairs with no row"
                             % len(missing))
                if _prim_canon_count(disk) != len(_pairs(att, "primary", "COMPLETED")):
                    f.append("COMPLETE/strict-m7-J1b: canonical %d != ledger COMPLETED pairs %d"
                             % (_prim_canon_count(disk),
                                len(_pairs(att, "primary", "COMPLETED"))))
        else:
            named = set(band["interval_resolved_Ks"] or []) | set(band["incomplete_at_K"] or [])
            for K in (26, 28, 30):
                c = _prim_canon_count_K(disk, K)
                if K in named:
                    if not c < 4:
                        f.append("COMPLETE/otherwise: K=%d count %d not <4" % (K, c))
                else:
                    if c != 4:
                        f.append("COMPLETE/otherwise: K=%d count %d != 4" % (K, c))
            missing = [(K, s) for K in (26, 28, 30) for s in range(4)
                       if (K, s) not in _pairs(att, "primary")]
            if missing:
                f.append("COMPLETE/otherwise-J1a: %d primary pairs with no row" % len(missing))
            if _prim_canon_count(disk) != len(_pairs(att, "primary", "COMPLETED")):
                f.append("COMPLETE/otherwise-J1b: canonical %d != ledger COMPLETED pairs %d"
                         % (_prim_canon_count(disk), len(_pairs(att, "primary", "COMPLETED"))))
            if mode == "NEW":                   # R8 K2
                if not len(_pairs(att, "primary", "COMPLETED")) >= 1:
                    f.append("COMPLETE/otherwise-K2: 0 COMPLETED primary pairs")
    elif rs == "COMPLETE-DEGRADED":
        missing = [(K, s) for K in (26, 28, 30) for s in range(4)
                   if (K, s) not in _pairs(att, "primary")]
        if missing:
            f.append("CD-J1a: %d primary pairs with no row" % len(missing))
        if _prim_canon_count(disk) != len(_pairs(att, "primary", "COMPLETED")):
            f.append("CD-J1b: canonical %d != ledger COMPLETED pairs %d"
                     % (_prim_canon_count(disk), len(_pairs(att, "primary", "COMPLETED"))))
        if not any(a["status"] in ("GATE-REFUSED", "PERSISTENTLY-ABORTED") for a in att):
            f.append("CD: no throttle evidence row")
        if mode == "NEW":                       # R8 K2
            if not len(_pairs(att, "primary", "COMPLETED")) >= 1:
                f.append("CD-K2: 0 COMPLETED primary pairs")
    elif rs in ("EXHAUSTED-BUDGET", "EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE"):
        if not realized > F("13.80"):
            f.append("EB base1: realized %s not >13.80" % realized)
        if not _prim_canon_count(disk) < 12:
            f.append("EB base2: primary canonical %d not <12" % _prim_canon_count(disk))
        if not any(a["arm"] == "primary" and a["attempt_n"] == 1
                   and a["status"] == "GATE-REFUSED" for a in att):
            f.append("EB base3: no primary attempt_n==1 GATE-REFUSED row")
        frac = F(str(rep["charged_vs_measured"]["ceiling_charged_fraction"]))
        if rs == "EXHAUSTED-BUDGET":
            if not frac <= F("0.50"):
                f.append("EB J4: ceiling_charged_fraction not <=0.50")
        else:
            if not frac > F("0.50"):
                f.append("EBSO J4-mirror: ceiling_charged_fraction not >0.50")
    return f


# ---------------------------------------------------------------------------
# Production disk-reading glue (not part of the audited suite; this is the
# NEW code the build stage adds).

def _glob_canonical(outdir: str, pattern: str) -> list[dict]:
    """Reads every JSON file matching `pattern` in `outdir` and returns
    `[{"K":..,"seed":..,"status":..}, ...]` — the shape `validity_check_core`
    expects as `disk["primary_canonical"]`/`disk["cond_canonical"]`. A file
    that fails to parse is recorded with `status=None` (never "COMPLETED"),
    so it can never inflate a COMPLETED count — matching G2's own
    canonical-path contract (a canonical file is written ONLY on COMPLETED
    acceptance; a malformed file at that path is an invariant violation,
    not silently treated as either state)."""
    out = []
    for path in sorted(glob.glob(os.path.join(outdir, pattern))):
        if path.endswith(".axis_c_lock.json"):
            continue
        try:
            with open(path) as fh:
                rec = json.load(fh)
            out.append({"K": rec.get("K"), "seed": rec.get("seed"),
                       "status": rec.get("status")})
        except (json.JSONDecodeError, OSError):
            out.append({"K": None, "seed": None, "status": None})
    return out


def build_disk_view(primary_outdir: str, conditional_outdir: str) -> dict:
    return {
        "primary_canonical": _glob_canonical(primary_outdir, "earlyln_K*_s*.json"),
        "cond_canonical": _glob_canonical(conditional_outdir, "earlyln_K*_s*.json"),
    }


def validity_check_report(report_path: str, primary_outdir: str,
                           conditional_outdir: str) -> list[str]:
    """Reads `orchestrator_report.json` from `report_path` and the two
    canonical directories, returns the failure-reason list (`[]` == PASS)."""
    with open(report_path) as fh:
        rep = json.load(fh)
    disk = build_disk_view(primary_outdir, conditional_outdir)
    return validity_check_core(rep, disk, mode="NEW")


def main(argv=None):
    """CLI entry point: `python3 -m kwall_lib.validity_check <report_path>
    <primary_outdir> <conditional_outdir>`. Exit 0 on PASS, 1 on FAIL
    (failure reasons printed to stderr) — the shape `queue_worker.sh`'s
    `bash -c "$vcheck"` expects."""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print("usage: validity_check.py <report_path> <primary_outdir> "
              "<conditional_outdir>", file=sys.stderr)
        return 2
    report_path, primary_outdir, conditional_outdir = argv
    # m7 (minor, build audit R1): when `run()` refuses at the startup smoke
    # gate (exit 3) or aborts loudly on a G2 invariant violation, NO report
    # is ever written -- routing to failed/ is still correct either way
    # (non-zero exit), but reading a nonexistent report used to raise an
    # uncaught FileNotFoundError, so the job log got a Python traceback
    # instead of a diagnosis. Diagnose explicitly instead.
    if not os.path.exists(report_path):
        print(f"VALIDITY_CHECK FAIL: no orchestrator_report.json at "
              f"{report_path!r} -- the orchestrator never wrote one (startup "
              f"smoke gate refusal, or a loud G2-invariant abort) -- see the "
              f"job's own stdout/stderr log for the real cause.", file=sys.stderr)
        return 1
    reasons = validity_check_report(report_path, primary_outdir, conditional_outdir)
    if reasons:
        print("VALIDITY_CHECK FAIL:", file=sys.stderr)
        for r in reasons:
            print("  -", r, file=sys.stderr)
        return 1
    print("VALIDITY_CHECK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
