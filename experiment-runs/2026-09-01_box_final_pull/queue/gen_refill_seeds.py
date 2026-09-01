#!/usr/bin/env python3
"""Durable-queue refill generator (GPU-hot doctrine, 2026-07-21).

Emits additional laneB 392M per_token replicate-seed cells into
/home/nvidia/queue/pending/ following the EXACT schema of the 718-series.
Purpose: keep the on-box queue >=2 days deep so the 8xH100 never idle even
if the coordinator compacts before the PI steers NCR. Config is fixed/matched
to 033/034 by construction -> outcome-safe, reopens no verdict.

Idempotent: skips any id whose pending/claimed/completed file already exists.
"""
import json, os, glob

QDIR = "/home/nvidia/queue"
PEND = os.path.join(QDIR, "pending")
CORPORA = ["openr1-mix-ext", "wikitext-mix-ext"]
SEEDS = list(range(30, 52))  # 30..43 inclusive -> 14 seeds x 2 corpora = 28 cells
START_ID = 1101              # 1001-1016 already used by the 22-29 batch; 1101+ is clear

CKPT_ROOT = "/data/fixscale_ckpts/train"
RES_DIR = "/home/nvidia/chapter2/deltanet_rd/results/fixscale/train"
PY = "/home/nvidia/tdenv/bin/python3"

# ids already present in ANY state (avoid collisions / double-run)
existing_ids = set()
for sub in ["pending", "claimed", "completed", "failed", "cancelled", "parked_k24plus"]:
    for f in glob.glob(os.path.join(QDIR, sub, "*.json")):
        existing_ids.add(os.path.basename(f)[:-5])

def make_cell(cid, seed, corpus):
    tag = f"fixscale_fulltoken_arm_per_token_392mY_ds64_{corpus}_s{seed}"
    ckpt = f"{CKPT_ROOT}/{tag}"
    out = f"{RES_DIR}/{tag}.json"
    cmd = (
        f"mkdir -p {ckpt} {RES_DIR} && cd /home/nvidia/chapter2/deltanet_rd && "
        f"{PY} lm_pretrain_rd.py --corpus {corpus} --data-dir /data/deltanet_rd_data "
        f"--d-model 1536 --d-state 64 --n-layers 16 --seq-len 512 --batch-size 32 "
        f"--steps 67547 --ckpt-every 3377 --seed {seed} --internal-timeout 86400 "
        f"--frozen-bias-arm per_token --frozen-bias-lambda 0.58 "
        f"--ckpt-dir {ckpt} --out {out}"
    )
    vcheck = (
        f"{PY} -c \"import json; d=json.load(open('{out}')); "
        f"assert d.get('complete') is True; assert d.get('steps_completed', 0) >= 67547 - 1\""
    )
    return {
        "id": cid,
        "lane": "B",
        "hypothesis": (
            "Durable-queue refill 2026-07-21 (GPU-hot doctrine): additional replicate "
            f"seed {seed} ({corpus}) for rung Y d_state=64 per_token at the SAME config "
            "as 033/034 and the 706-723 laneB series (dm=1536, ds=64, L=16, per_token, "
            "lambda 0.58, 67,547 steps). PRIMARY purpose: maintain >=2-day on-box queue "
            "depth so the 8xH100 never idle if the coordinator compacts before the NCR "
            "steer lands. SECONDARY: marginal CI-tightening on the d_state=64 rung. "
            "Outcome-safe: config fixed/matched by construction; does NOT reopen the "
            "attribution verdict and does not depend on the T2a param-axis instrument."
        ),
        "cmd": cmd,
        "gpu_h_estimate": 15.76,
        "output_dir": RES_DIR,
        "validity_check": vcheck,
        "notes": (
            "COST rail identical to 033/034/718-series: 0.840 s/step upper bound x 67,547 "
            "steps = 15.76 GPU-h/cell; --internal-timeout 86400s = ~1.5x that bound. "
            "Insurance refill only; if the PI steers NCR/ship these can be cancelled "
            "(mv to cancelled/) without loss -- config redundant-but-valid."
        ),
    }

written, skipped = [], []
cid_n = START_ID
for seed in SEEDS:
    for corpus in CORPORA:
        cid = f"{cid_n}_laneB_392mY_ds64_per_token_{corpus}_s{seed}"
        cid_n += 1
        # skip if a same-tag run already exists in any state (dedupe by tag, not just id)
        tag = f"laneB_392mY_ds64_per_token_{corpus}_s{seed}"
        if any(tag in e for e in existing_ids) or cid in existing_ids:
            skipped.append(cid); continue
        cell = make_cell(cid, seed, corpus)
        path = os.path.join(PEND, cid + ".json")
        with open(path, "w") as fh:
            json.dump(cell, fh, indent=1)
        written.append(cid)

print(f"WROTE {len(written)} cells, SKIPPED {len(skipped)}")
for c in written:
    print("  +", c)
if skipped:
    print("skipped (already exist):")
    for c in skipped:
        print("  =", c)
gpu_h = len(written) * 15.76
print(f"ADDED ~{gpu_h:.0f} GPU-h ; pending now = {len(glob.glob(os.path.join(PEND,'*.json')))}")
