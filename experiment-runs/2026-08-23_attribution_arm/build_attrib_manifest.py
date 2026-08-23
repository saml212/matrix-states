#!/usr/bin/env python3
"""Enumerate the 12 attribution cells (Option A, ids 0230-0241) and resolve
checkpoint + training record. Emits TSV: K, recipe, seed, name, ckpt, cellcfg,
status, step. Nonzero exit if any expected cell has no checkpoint.

Cell set of record (ATTRIBUTION_ARM.md §2 union x seeds 0-1, #22 Option A):
  (16,compB) (24,compB) (32,compB) (40,compB) (32,primary) (40,primary)
"""
import json
import os
import sys

CELLS = [(16, "compB"), (24, "compB"), (32, "compB"),
         (40, "compB"), (32, "primary"), (40, "primary")]
SEEDS = (0, 1)
CK = "/ephemeral/scaleaxis/attribution/ckpts"
RES = "/ephemeral/scaleaxis/attribution/results"


def main():
    only_done = "--completed-only" in sys.argv
    rows, missing = [], []
    for k, rec in CELLS:
        for s in SEEDS:
            name = f"attrib40k_K{k}_{rec}_s{s}"
            ck = os.path.join(CK, name, f"{name}.ckpt.pt")
            cfg = os.path.join(RES, f"{name}.json")
            if not os.path.exists(ck):
                missing.append(f"MISSING-CKPT {name}: {ck}")
                continue
            st, step = "UNKNOWN", None
            if os.path.exists(cfg):
                try:
                    d = json.load(open(cfg))
                    st, step = d.get("status", "UNKNOWN"), d.get("step")
                except Exception as e:
                    missing.append(f"UNPARSEABLE-RECORD {name}: {e!r}")
                    continue
            if only_done and not (st == "COMPLETED" and step == 40000):
                continue
            rows.append((k, rec, s, name, ck, cfg if os.path.exists(cfg) else "-", st, step))
    for m in missing:
        print(m, file=sys.stderr)
    for r in rows:
        print("\t".join(str(x) for x in r))
    exp = len(CELLS) * len(SEEDS)
    print(f"MANIFEST: {len(rows)} rows emitted ({'COMPLETED-only' if only_done else 'all'}), "
          f"expected total {exp}, {len(missing)} MISSING", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
