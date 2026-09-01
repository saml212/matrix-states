#!/usr/bin/env python3
"""Re-adjudicate attribution cells with the CORRECTED validity_check.

Procedure of record (EXPERIMENT_LOG 2026-08-23 #1 and #2): the queue worker
snapshots a job spec at CLAIM time, so a cell in flight when a spec is fixed
carries the STALE checker and can land in ~/queue/failed on a checker bug
rather than a real defect. The remedy is NOT to trust either verdict but to
re-run the corrected check FROM ~/ncr_scaleaxis/job_specs_attribution/ against
the artifact, and promote failed->completed ONLY on a clean pass.

Clause-level reporting: the check is one python -c of chained asserts, so a
single failure hides the rest. This runs the whole check (the verdict of
record) AND then each clause independently, so a failure is attributed to a
named clause instead of "the check failed".

NOTHING is promoted on a real-clause failure. In particular the GATE-0 clause
(`h[-1][1] < h[0][1]`) was ruled MIS-SCOPED for resumed segments by #2 (CE is
at plateau over the marginal 20k steps); per the coordinator's standing
instruction a cell failing on it is REPORTED, not promoted.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

SPEC_DIR = "/home/nvidia/ncr_scaleaxis/job_specs_attribution"
QUEUE = "/home/nvidia/queue"
PY = "/home/nvidia/tdenv/bin/python3"
GATE0_MARKER = "GATE-0 NOT CONVERGED"


def spec_for(cell):
    for p in glob.glob(os.path.join(SPEC_DIR, "*.json")):
        if cell in os.path.basename(p):
            return p
    return None


def run_check(code):
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    return r.returncode, (r.stderr or "").strip()


def clauses(code):
    """Split the inner python source into its leading setup + each assert."""
    m = re.match(r'^\s*"?(.*)"?\s*$', code, re.S)
    body = code
    parts, buf, depth = [], "", 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == ";" and depth == 0:
            parts.append(buf); buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf)
    setup = [p for p in parts if not p.strip().startswith("assert")]
    asserts = [p for p in parts if p.strip().startswith("assert")]
    return setup, asserts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--promote", action="store_true",
                    help="move failed/<id> -> completed/<id> for cells that PASS the whole check")
    args = ap.parse_args()

    specs = sorted(glob.glob(os.path.join(SPEC_DIR, "*.json")))
    rows = []
    for sp in specs:
        spec = json.load(open(sp))
        sid = spec["id"]
        cell = os.path.basename(sp).replace(".json", "").split("_", 1)[1]
        if args.cells and not any(c in cell for c in args.cells):
            continue
        raw = spec["validity_check"]
        # the stored value is a shell command: <py> -c "<source>"
        m = re.search(r'-c\s+"(.*)"\s*$', raw, re.S)
        src = m.group(1).replace('\\"', '"').replace("\\'", "'") if m else None
        if src is None:
            print(f"!!! {sid} {cell}: could not extract check source -- STOP")
            rows.append((sid, cell, "UNEXTRACTABLE", []))
            continue
        rc, err = run_check(src)
        failed = []
        if rc != 0:
            setup, asrt = clauses(src)
            pre = ";".join(setup)
            for a in asrt:
                c2 = pre + ";" + a
                rc2, err2 = run_check(c2)
                if rc2 != 0:
                    lab = a.strip()[:90]
                    failed.append((lab, err2.splitlines()[-1] if err2 else ""))
        rows.append((sid, cell, "PASS" if rc == 0 else "FAIL", failed))

    print(f"{'id':>5s} {'cell':38s} {'verdict':8s}  failing clauses")
    promoted, held = [], []
    for sid, cell, v, failed in rows:
        print(f"{sid:>5s} {cell:38s} {v:8s}  "
              + ("-" if not failed else "; ".join(f"[{a}] -> {e}" for a, e in failed)))
        # spec["id"] is the FULL stem (e.g. 0238_ncr_scaleaxis_attrib40k_K24_compB_s0)
        # and the queue file is "<stem>.json" or "<stem>.g<N>.json" -- an earlier
        # "{sid}_*" glob matched NEITHER and silently promoted nothing while
        # printing PASS. Match the stem with any suffix.
        fpath = glob.glob(os.path.join(QUEUE, "failed", f"{sid}.*"))
        if v == "PASS" and fpath:
            if args.promote:
                dst = os.path.join(QUEUE, "completed", os.path.basename(fpath[0]))
                os.replace(fpath[0], dst)
                promoted.append(sid)
            else:
                promoted.append(sid + " (dry-run)")
        elif v != "PASS" and fpath:
            held.append((sid, cell, [a for a, _ in failed]))

    print(f"\nPROMOTED (failed->completed): {promoted if promoted else 'none'}")
    if held:
        print("HELD -- real-clause failures, NOT promoted, for coordinator adjudication:")
        for sid, cell, cl in held:
            g0 = any(GATE0_MARKER in c for c in cl)
            print(f"   {sid} {cell}: {cl}"
                  + ("   [GATE-0 clause -- #2 ruled this MIS-SCOPED for resumed segments]" if g0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
