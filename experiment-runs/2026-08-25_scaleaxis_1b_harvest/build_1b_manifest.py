#!/usr/bin/env python3
"""Enumerate the 24 1.31B cells (22 sweep + the 2 calibration K=24 s0 cells).
Emits TSV: K, tag, ckpt, cellcfg. Nonzero exit on any missing checkpoint or
missing/incomplete training record -- a silently-dropped cell is the failure
class this program has hit repeatedly.

Scope of record: EXPERIMENT_LOG 2026-08-24 #3 -- specs 0382/0385 alias the
calibration pair's paths and are NOT QUEUED, so the SWEEP is 22 cells and
K=24 s0 (both recipes) comes from the CALIBRATION pair.
"""
import json
import os
import sys

KS = (16, 24, 32, 40)
REC = ("primary", "compB")
CK = "/ephemeral/scaleaxis1b/ckpts"
RES = "/ephemeral/scaleaxis1b/results"


def main():
    rows, bad = [], []
    for k in KS:
        for r in REC:
            for s in (0, 1, 2):
                name = f"scaleaxis1310m_K{k}_{r}_s{s}"
                ck = os.path.join(CK, name, f"{name}.ckpt.pt")
                cfg = os.path.join(RES, f"{name}.json")
                if not os.path.exists(ck):
                    bad.append(f"MISSING-CKPT {name}: {ck}")
                    continue
                if not os.path.exists(cfg):
                    bad.append(f"MISSING-RECORD {name}: {cfg}")
                    continue
                try:
                    d = json.load(open(cfg))
                except Exception as e:
                    bad.append(f"UNPARSEABLE-RECORD {name}: {e!r}")
                    continue
                if d.get("status") != "COMPLETED":
                    bad.append(f"NOT-COMPLETED {name}: status={d.get('status')} step={d.get('step')}")
                    continue
                rows.append((k, f"depthext6_1310m_{name}", ck, cfg))
    for m in bad:
        print(m, file=sys.stderr)
    for k, tag, ck, cfg in rows:
        print(f"{k}\t{tag}\t{ck}\t{cfg}")
    print(f"MANIFEST: {len(rows)} cells (expected 24), {len(bad)} PROBLEMS", file=sys.stderr)
    return 1 if (bad or len(rows) != 24) else 0


if __name__ == "__main__":
    sys.exit(main())
