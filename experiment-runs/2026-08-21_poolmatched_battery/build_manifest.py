#!/usr/bin/env python3
"""Enumerate every COMPLETED cell and resolve its checkpoint across ALL roots.

Emits a TSV manifest (tag, ckpt, cellcfg) on stdout and a MISSING-CKPT report
on stderr. Exits nonzero if any COMPLETED cell has no checkpoint anywhere --
a completed cell silently dropping out of the battery is exactly the
silent-zero-scoring failure this program has hit six times.
"""
import glob
import json
import os
import sys

CKPT_ROOTS = ["/ephemeral/embed_path_ckpts",
              "/ephemeral/reseed_ckpts",
              "/home/nvidia/ncr_g3b31_contrastive/results",
              "/home/nvidia/ncr_g3b24_rebalance/results"]
CFG_ROOTS = ["/home/nvidia/ncr_g3b31_contrastive/results",
             "/home/nvidia/ncr_g3b24_rebalance/results"]


def main():
    cells = {}
    for r in CFG_ROOTS:
        for p in sorted(glob.glob(os.path.join(r, "mob_*.json"))):
            try:
                d = json.load(open(p))
            except Exception as e:
                print(f"UNPARSEABLE-CONFIG {p}: {e!r}", file=sys.stderr)
                continue
            cid = d.get("cell_id")
            if not cid or d.get("status") != "COMPLETED":
                continue
            cells[cid] = p

    rows, missing = [], []
    for cid, cfgp in sorted(cells.items()):
        ck = None
        for r in CKPT_ROOTS:
            cand = os.path.join(r, f"{cid}_ckpts", f"{cid}.ckpt.pt")
            if os.path.exists(cand):
                ck = cand
                break
        if ck is None:
            missing.append(cid)
            print(f"MISSING-CKPT {cid} (COMPLETED training record at {cfgp}, but no checkpoint "
                  f"in any of {CKPT_ROOTS})", file=sys.stderr)
            continue
        rows.append((cid, ck, cfgp))

    for cid, ck, cfgp in rows:
        print(f"{cid}\t{ck}\t{cfgp}")
    print(f"MANIFEST: {len(rows)} cells resolved, {len(missing)} MISSING", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
