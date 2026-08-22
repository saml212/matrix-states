#!/usr/bin/env python3
"""Enumerate the 36 distinct K-scaling cells and resolve each checkpoint.

Emits TSV: K, tag, ckpt, cellcfg, anchor_runner_tag ("-" if none).
Fails loudly (nonzero exit) if any expected cell has no checkpoint.

Inventory of record (EXPERIMENT_LOG 2026-08-22 #2 Curve 1): 6 K values x
2 recipes x 3 seeds = 36 cells, of which K=24 IS the anchor stratum
(mob_g3b31_{primary,compB}_s{0,1,2}) and the other 30 live under
/ephemeral/kscaling/ckpts/. The anchors carry the pinned gate3 runner tag
and need the audited --anchor-runner-tag allowlist to load read-only.
"""
import os
import sys

DEFAULT_SWEEP_K = (12, 16, 20, 28, 32)  # K=24 is the anchor stratum, below
FRONTIER_K = (36, 40)                   # design sec 14 frontier extension
ANCHOR_K = 24
RECIPES = ("primary", "compB")
SEEDS = (0, 1, 2)
SWEEP_ROOT = "/ephemeral/kscaling/ckpts"
SWEEP_CFG = "/ephemeral/kscaling/results"
ANCHOR_ROOTS = ["/ephemeral/reseed_ckpts", "/home/nvidia/ncr_g3b31_contrastive/results"]
ANCHOR_CFG_ROOT = "/home/nvidia/ncr_g3b31_contrastive/results"
ANCHOR_TAG = "ncr_gate3_wave1_runner_v1"


def main():
    # --ks 36,40 / --no-anchors select the frontier extension; defaults preserve
    # the original six-stratum manifest byte-for-byte.
    ks, anchors = DEFAULT_SWEEP_K, True
    argv = sys.argv[1:]
    if "--frontier" in argv:
        ks, anchors = FRONTIER_K, False
    for i, a in enumerate(argv):
        if a == "--ks":
            ks = tuple(int(x) for x in argv[i + 1].split(","))
        if a == "--no-anchors":
            anchors = False

    rows, missing = [], []

    for k in ks:
        for rec in RECIPES:
            for s in SEEDS:
                name = f"kscaling_K{k}_{rec}_s{s}"
                ck = os.path.join(SWEEP_ROOT, name, f"{name}.ckpt.pt")
                cfg = os.path.join(SWEEP_CFG, f"{name}.json")
                if not os.path.exists(ck):
                    missing.append(f"MISSING-CKPT {name}: {ck}")
                    continue
                rows.append((k, f"depthext_{name}", ck,
                             cfg if os.path.exists(cfg) else "-", "-"))

    for rec in (RECIPES if anchors else ()):
        for s in SEEDS:
            name = f"mob_g3b31_{rec}_s{s}"
            ck = None
            for root in ANCHOR_ROOTS:
                cand = os.path.join(root, f"{name}_ckpts", f"{name}.ckpt.pt")
                if os.path.exists(cand):
                    ck = cand
                    break
            cfg = os.path.join(ANCHOR_CFG_ROOT, f"{name}.json")
            if ck is None:
                missing.append(f"MISSING-CKPT {name}: not in any of {ANCHOR_ROOTS}")
                continue
            rows.append((ANCHOR_K, f"depthext_anchor_{name}", ck,
                         cfg if os.path.exists(cfg) else "-", ANCHOR_TAG))

    for m in missing:
        print(m, file=sys.stderr)
    for k, tag, ck, cfg, at in rows:
        print(f"{k}\t{tag}\t{ck}\t{cfg}\t{at}")
    expected = len(ks) * len(RECIPES) * len(SEEDS) + (len(RECIPES) * len(SEEDS) if anchors else 0)
    print(f"MANIFEST: {len(rows)} cells resolved (expected {expected}), "
          f"{len(missing)} MISSING", file=sys.stderr)
    return 1 if (missing or len(rows) != expected) else 0


if __name__ == "__main__":
    sys.exit(main())
