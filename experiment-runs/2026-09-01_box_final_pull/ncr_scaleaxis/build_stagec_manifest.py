#!/usr/bin/env python3
"""Enumerate the 24 392M Stage-C cells and resolve each checkpoint.
Emits TSV: K, tag, ckpt, cellcfg. Nonzero exit if any expected cell is missing.
Inventory of record: EXPERIMENT_LOG 2026-08-22 #20 -- 4 ported K x 2 recipes x 3 seeds.
"""
import os
import sys

KS = (16, 24, 32, 40)
RECIPES = ("primary", "compB")
SEEDS = (0, 1, 2)
CKPT_ROOT = "/ephemeral/scaleaxis/ckpts"
CFG_ROOT = "/ephemeral/scaleaxis/results"


def main():
    rows, missing = [], []
    for k in KS:
        for rec in RECIPES:
            for s in SEEDS:
                name = f"scaleaxis392m_K{k}_{rec}_s{s}"
                ck = os.path.join(CKPT_ROOT, name, f"{name}.ckpt.pt")
                cfg = os.path.join(CFG_ROOT, f"{name}.json")
                if not os.path.exists(ck):
                    missing.append(f"MISSING-CKPT {name}: {ck}")
                    continue
                rows.append((k, f"depthext6_392m_{name}", ck, cfg if os.path.exists(cfg) else "-"))
    for m in missing:
        print(m, file=sys.stderr)
    for k, tag, ck, cfg in rows:
        print(f"{k}\t{tag}\t{ck}\t{cfg}")
    exp = len(KS) * len(RECIPES) * len(SEEDS)
    print(f"MANIFEST: {len(rows)} cells resolved (expected {exp}), {len(missing)} MISSING",
          file=sys.stderr)
    return 1 if (missing or len(rows) != exp) else 0


if __name__ == "__main__":
    sys.exit(main())
