#!/bin/bash
# KEY_ANCHORING_DESIGN.md sec 11 (Rev K48.1, CLEARED-FOR-BUILD per sec
# 11.11's external verify) -- keyanchor-k48 chain (2026-07 K48+e build).
# NOT RUN by the build session -- build + CPU-verify only, no GPU work
# performed here (see the build report). The orchestrator runs this once a
# free GPU set is confirmed (check nvidia-smi FIRST, per house rule).
#
# STAGE ORDER (mandatory, sec 11.1's own dependency structure): K=48
# reference arms -> K48 bands pinned -> K=48 candidate cells. The Gate-1-
# style probe (DISCLOSED, NON-GATING, sec 11.1 item 4) is launched before
# the reference arms since it has no dependency on them and is cheap
# (5,000 steps) -- reordering it does not change the mandatory dependency
# chain above, only when a non-gating, informational cell happens to run.
#
# Gate chain enforced by run_deltanet_rd_exactness_sweep.py's own
# --wave keyanchor-k48 dispatch (not re-duplicated here): (1)
# BANDS_PINNED_K48.json (sec 11.1.1, a SEPARATE file/gate from K=16/32's
# own BANDS_PINNED.json); (2) the K=48 Gate-1-style probe read (DISCLOSED,
# NON-GATING -- reported, never blocks); (3) the budget guard, using the
# RECONCILED program-spent constant (sec 11.5's own required pre-launch
# reconciliation, see run_deltanet_rd_exactness_sweep.py's
# KEYANCHOR_PROGRAM_SPENT_GPUH_RECONCILED comment for the itemized ledger);
# (4) the disk gate (sec 10.10's checkpoint writer, reused).
#
# CONDITIONAL cells (NOT launched by this script by default -- both require
# an explicit flag, sec 11.1 items 3/5):
#   --include-k48-fixed-lambda1  (OPTIONAL, lowest cut priority, sec 11.4.3)
#   --include-k48-dprime --accept-k48-dprime-orchestrator-signoff
#     (CONDITIONAL on Rev 7.1's own K=32 verdict reaching A(d')/D',
#     sec 11.1 item 3 -- an ORCHESTRATOR sign-off, not a mechanical gate;
#     do NOT pass this flag pair until a human has read Rev 7.1's own
#     readout_rev7.py output and confirmed that verdict)
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs

# GPUs 0-1 run rung-3 (untouchable at build time). GPUs 2-7 free but
# CPU-verify only during THIS build session -- override via env vars once a
# real GPU set is confirmed free (check nvidia-smi FIRST, per house rule).
KEYANCHOR_K48_GPUS=${KEYANCHOR_K48_GPUS:-4}
KEYANCHOR_K48_GPU_OFFSET=${KEYANCHOR_K48_GPU_OFFSET:-2}

# ckpt-dir symlinks/directories pre-created (task's own explicit ask) --
# run_deltanet_rd.py's own os.makedirs(ckpt_dir, exist_ok=True) would also
# create these lazily, but pre-creating here makes the disk gate's
# resolved-path check meaningful even before the first cell writes to it.
mkdir -p /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-k48
mkdir -p /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-k48-gate1

# Pre-launch CPU smoke gates (also re-run automatically by the sweep
# script's own main() -- run here too so a failure is visible before ANY
# subprocess dispatch attempt).
$PY smoke_key_anchoring.py 2>&1 | tee logs/20_smoke_key_anchoring_k48.log
$PY smoke_keyanchor_k48_e.py 2>&1 | tee logs/21_smoke_keyanchor_k48_e.log
$PY gate2_construction_test.py 2>&1 | tee logs/22_gate2_construction_test_k48.log

# Gate-1-style probe (candidate (d) arch, K=48, 5,000 steps, 1 run) --
# DISCLOSED, NON-GATING (sec 11.1 item 4); run first since it has no
# dependency on the reference arms and is cheap. --wave keyanchor-k48 below
# reads its result (KEYANCHOR_K48_GATE1_JSON_DEFAULT, e633862-audit-F1-style
# writer==reader equality, smoke_keyanchor_k48_e.py does not re-check this
# specific equality since it mirrors an already-verified pattern) but never
# blocks on it.
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-k48-gate1 \
    --gpus 1 --gpu-offset "$KEYANCHOR_K48_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-k48-gate1 \
    2>&1 | tee logs/23_wave_keyanchor_k48_gate1.log

# STAGE 1 (mandatory dependency): K=48 reference arms (bare geo3, seeds
# {1,2,3}, drift_probe=True) -- MUST complete before bands can be pinned.
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-k48-ref \
    --gpus "$KEYANCHOR_K48_GPUS" --gpu-offset "$KEYANCHOR_K48_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    2>&1 | tee logs/24_wave_keyanchor_k48_ref.log

# STAGE 2 (mandatory dependency): pin BANDS_PINNED_K48.json (sec 11.1.1 --
# a SEPARATE file from K=16/32's own BANDS_PINNED.json). No-GPU action;
# exits nonzero if any of the 3 K=48 reference arms is not yet valid --
# `set -e` stops the chain here rather than proceeding to launch candidate
# (d) against a missing/invalid bands file.
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-k48-bands \
    --out-dir results/deltanet_rd_exactness \
    2>&1 | tee logs/25_wave_keyanchor_k48_bands.log

# STAGE 3: K=48 candidate (d), the wave's MANDATORY/PRIMARY cells (seeds
# {30,31,32}). Add --include-k48-fixed-lambda1 and/or --include-k48-dprime
# (+ --accept-k48-dprime-orchestrator-signoff) here once those conditional
# cells are actually authorized (see this script's own header comment) --
# NEITHER is appended by default.
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-k48 \
    --gpus "$KEYANCHOR_K48_GPUS" --gpu-offset "$KEYANCHOR_K48_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-k48 \
    2>&1 | tee logs/26_wave_keyanchor_k48.log

$PY readout_rev7.py --out-dir results/deltanet_rd_exactness --manifest keyanchor-k48 \
    2>&1 | tee logs/27_readout_rev7_k48.log

touch results/deltanet_rd_exactness/KEYANCHOR_K48_CHAIN_DONE
