#!/bin/bash
# KEY_ANCHORING_DESIGN.md sec 12 (Rev 12.2, CLEARED-FOR-BUILD, human sign-off
# recorded 2026-07-06) -- keyanchor-cliff chain (capacity-cliff localization
# wave). NOT RUN by the build session -- build + CPU-verify only, no GPU
# work performed here. The orchestrator runs this once a free GPU set is
# confirmed (check nvidia-smi FIRST, per house rule) AND the sec 12 header's
# PI-DECISION sign-off is on record (already true as of this build).
#
# STAGE ORDER (sec 12.2.3's staged launch, mandatory): Stage 1 (K=38+K=42,
# the two interior/most-informative-for-x0 points) launches FIRST AND ALONE
# on its own GPUs -- no other multi-cell program (incl. the concurrent
# frozen-bias LM program on the SAME shared GPUs 2-7) starts until Stage 1's
# realized rate is confirmed within the K48-calibrated bracket. Stage 2
# (K=34+K=46) is MECHANICALLY GATED on Stage 1 clearing (run_deltanet_rd_
# exactness_sweep.py's own keyanchor_cliff_stage_gate, --wave keyanchor-cliff
# --stage 2 -- NOT an honor-system flag).
#
# set -euo pipefail is REQUIRED here (project hard rule: no `| tee`
# swallowing exits) -- under pipefail, `cmd | tee log` still propagates
# cmd's own exit code, so a failed smoke/gate/launch step stops this chain
# rather than silently continuing past a `tee`d failure.
#
# NO BANDS_PINNED_CLIFF-style new gate: this wave REUSES the EXISTING
# K=16/32 BANDS_PINNED.json blinding gate (candidate (d)'s architecture and
# frame-potential init are UNCHANGED from keyanchor-k48, sec 12.1 -- and the
# reference-arm cut, sec 12.2 item 2, means there is no new per-K reference
# measurement at K=34/38/42/46 to derive a fresh bands file from anyway).
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs

# GPUs 0-1 run rung-3 (untouchable). GPUs 2-7 free but shared CONCURRENTLY
# with the frozen-bias LM program (sec 12.5's own GPU plan, sec 12.2.3's
# compounding-contention mitigation: THIS wave's Stage 1 must clear the
# bracket before the LM program's own calibration cell starts) -- override
# via env vars once a real GPU set is confirmed free (check nvidia-smi
# FIRST, per house rule).
KEYANCHOR_CLIFF_GPUS=${KEYANCHOR_CLIFF_GPUS:-2}
KEYANCHOR_CLIFF_GPU_OFFSET=${KEYANCHOR_CLIFF_GPU_OFFSET:-2}

mkdir -p /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-cliff

# Pre-launch CPU smoke gates (also re-run automatically by the sweep
# script's own main() -- run here too so a failure is visible before ANY
# subprocess dispatch attempt). Zero-GPU items registered by sec 12.2.1/
# 12.2.2/12.3 that are NOT re-run automatically by main() (they are
# Wave -1 pre-registration artifacts, not per-launch gates) are run here
# explicitly, once, ahead of any training cell.
$PY sim_cliff_power.py --n-trials 4000 --out sim_cliff_power_results.json \
    2>&1 | tee logs/30_sim_cliff_power.log
$PY keyanchor_cliff_niter_check.py --out keyanchor_cliff_niter_check_results.json \
    2>&1 | tee logs/31_keyanchor_cliff_niter_check.log
$PY rev7_threshold_derive.py --out REV7_THRESHOLD_PINNED_K34_K38_K42_K46.json \
    2>&1 | tee logs/32_rev7_threshold_derive_cliff.log
$PY smoke_key_anchoring.py 2>&1 | tee logs/33_smoke_key_anchoring_cliff.log
$PY smoke_keyanchor_cliff.py 2>&1 | tee logs/34_smoke_keyanchor_cliff.log
$PY gate2_construction_test.py 2>&1 | tee logs/35_gate2_construction_test_cliff.log

# STAGE 1: K=38 + K=42 (6 cells, 3 seeds each), launched first and alone on
# this wave's own 2 GPUs -- no --include-cliff-gate1 by default (sec 12.2.3:
# first cut under budget pressure; add the flag once the mandatory cells'
# own realized rate confirms headroom for the optional probes).
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-cliff --stage 1 \
    --gpus "$KEYANCHOR_CLIFF_GPUS" --gpu-offset "$KEYANCHOR_CLIFF_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-cliff \
    2>&1 | tee logs/36_wave_keyanchor_cliff_stage1.log

# CHECKPOINT (sec 12.2.3): Stage 1's realized wall_s/GPU-h must clear the
# K48-calibrated bracket before Stage 2 launches. --wave keyanchor-cliff
# --stage 2 below RE-CHECKS this mechanically (keyanchor_cliff_stage_gate)
# and REFUSES (exit 1) if it does not -- this script does not need its own
# duplicate check; `set -e` stops the chain here if Stage 2's own launch
# command exits non-zero for that reason.
#
# STAGE 2 (conditional on Stage 1 clearing): K=34 + K=46 (6 cells, 3 seeds
# each). If Stage 1's realized rate was elevated but under the 1.5x hard-halt
# threshold, re-price per sec 12.2.3 item 3 BEFORE running this command (a
# human decision, not automated here) -- do not blindly re-run this line
# after a Stage-1 near-miss without that re-pricing step.
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-cliff --stage 2 \
    --gpus "$KEYANCHOR_CLIFF_GPUS" --gpu-offset "$KEYANCHOR_CLIFF_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-cliff \
    2>&1 | tee logs/37_wave_keyanchor_cliff_stage2.log

# Real-data fit (sec 12.4) -- reads the now-complete 12-cell manifest plus
# the archived K=32/K=48 candidate-(d) 3-seed arrays. Paths below match this
# wave's own citations (sec 12.1's archive-path correction); update if the
# archive location differs on the actual training box.
$PY fit_cliff_curve.py \
    --cliff-out-dir results/deltanet_rd_exactness/wavekeyanchor-cliff \
    --k32-dir experiment-runs/2026-07-06_keyanchor_mech/wavekeyanchor-mech \
    --k48-dir experiment-runs/2026-07-07_keyanchor_k48/wavekeyanchor-k48 \
    --n-trials 4000 --out fit_cliff_curve_results.json \
    2>&1 | tee logs/38_fit_cliff_curve.log

touch results/deltanet_rd_exactness/KEYANCHOR_CLIFF_CHAIN_DONE
