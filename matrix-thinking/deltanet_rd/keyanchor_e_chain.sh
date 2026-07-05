#!/bin/bash
# KEY_ANCHORING_DESIGN.md sec 10.13's registered candidate (e)
# ("frozen-random-table ablation") chain (2026-07 K48+e build). NOT RUN by
# the build session -- build + CPU-verify only, no GPU work performed here
# (see the build report). The orchestrator runs this once a free GPU set is
# confirmed (check nvidia-smi FIRST, per house rule).
#
# INDEPENDENT of keyanchor_k48_chain.sh -- candidate (e) is K=32, reuses the
# EXISTING K=16/32 BANDS_PINNED.json (sec 3.6, already pinned by an earlier
# wave; this script does NOT re-run --wave ref/keyanchor-bands), so it has
# NO dependency on the K=48 wave's own reference arms/bands file. The two
# chains may run CONCURRENTLY on disjoint free GPUs (sec 11.0's own
# "candidate-(e) cells independent -- can interleave on free GPUs" build
# instruction) -- this script does not itself coordinate GPU assignment
# with keyanchor_k48_chain.sh; the orchestrator is responsible for passing
# disjoint --gpu-offset/--gpus (or KEYANCHOR_E_GPU_OFFSET/KEYANCHOR_E_GPUS
# env vars) to the two scripts if run at the same time.
#
# Cost: ~1 GPU-h (3 cells, sec 10.13's own estimate; this build's own
# registered ceiling is 1.5 GPU-h, keyanchor_e_budget_guard).
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs

# GPUs 0-1 run rung-3 (untouchable at build time). GPUs 2-7 free but
# CPU-verify only during THIS build session -- override via env vars once a
# real GPU set is confirmed free (check nvidia-smi FIRST, per house rule;
# if running CONCURRENTLY with keyanchor_k48_chain.sh, pick a DISJOINT
# offset from that script's own KEYANCHOR_K48_GPU_OFFSET/KEYANCHOR_K48_GPUS).
KEYANCHOR_E_GPUS=${KEYANCHOR_E_GPUS:-3}
KEYANCHOR_E_GPU_OFFSET=${KEYANCHOR_E_GPU_OFFSET:-5}

# ckpt-dir pre-created (task's own explicit ask).
mkdir -p /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-e

# Pre-launch CPU smoke gates (also re-run automatically by the sweep
# script's own main() -- run here too so a failure is visible before ANY
# subprocess dispatch attempt). smoke_key_anchoring.py's own smoke 15/16/17
# (frozen-table-receives-no-grad, fixed-lambda-never-moves,
# random_unit_rows_init genuinely distinct from frame_potential_init) are
# THE load-bearing pre-launch checks for this specific wave's own new
# mechanism -- never skip this smoke for --wave keyanchor-e.
$PY smoke_key_anchoring.py 2>&1 | tee logs/30_smoke_key_anchoring_e.log
$PY smoke_keyanchor_k48_e.py 2>&1 | tee logs/31_smoke_keyanchor_k48_e_for_e.log
$PY gate2_construction_test.py 2>&1 | tee logs/32_gate2_construction_test_e.log

# candidate (e): K=32, seeds {60,61,62}, frozen random-unit-rows table,
# fixed lambda=0.58. REUSES the existing K=16/32 BANDS_PINNED.json gate
# (--wave ref must have already completed and been pinned by an earlier
# wave -- this script does not launch --wave ref/keyanchor-bands itself).
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-e \
    --gpus "$KEYANCHOR_E_GPUS" --gpu-offset "$KEYANCHOR_E_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-e \
    2>&1 | tee logs/33_wave_keyanchor_e.log

$PY readout_rev7.py --out-dir results/deltanet_rd_exactness --manifest keyanchor-e \
    2>&1 | tee logs/34_readout_rev7_e.log

touch results/deltanet_rd_exactness/KEYANCHOR_E_CHAIN_DONE
