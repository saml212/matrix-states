#!/bin/bash
# KEY_ANCHORING_DESIGN.md sec 10 (Rev 7.1) -- keyanchor-mech chain
# (2026-07-06 build). NOT RUN by the build session -- build + CPU-verify
# only, no GPU work performed here (see the build report). The orchestrator
# runs this once GPUs 2-7's other trajectory-probe wave frees up.
#
# Gate chain enforced by run_deltanet_rd_exactness_sweep.py's own
# --wave keyanchor-mech dispatch (not re-duplicated here): (1) the Rev-7.1
# pin triple (REV7_THRESHOLD_PINNED.json exists + script-hash match + live
# derive() reproduction); (2) candidate (d')'s own Gate-1 probe
# (keyanchor-mech-gate1, launched first, below); (3) sec 3.6's
# BANDS_PINNED reference-arm reuse-or-refuse gate; (4) the budget guard
# (12 GPU-h registered ceiling vs. the 80 GPU-h exactness-program cap);
# (5) the disk gate (sec 10.10's checkpoint writer).
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs

# GPUs 0-1 run rung-3 (untouchable). GPUs 2-7 run another agent's
# trajectory-probe wave at build time -- DO NOT launch until they free up;
# override via env vars once a real GPU set is confirmed free (check
# nvidia-smi FIRST, per house rule).
KEYANCHOR_MECH_GPUS=${KEYANCHOR_MECH_GPUS:-4}
KEYANCHOR_MECH_GPU_OFFSET=${KEYANCHOR_MECH_GPU_OFFSET:-2}

# Pre-launch CPU smoke gates (also re-run automatically by the sweep
# script's own main() -- run here too so a failure is visible before ANY
# subprocess dispatch attempt).
$PY smoke_key_anchoring.py 2>&1 | tee logs/10_smoke_key_anchoring_mech.log
$PY smoke_keyanchor_mech.py 2>&1 | tee logs/11_smoke_keyanchor_mech.log
$PY gate2_construction_test.py 2>&1 | tee logs/12_gate2_construction_test_mech.log

# Candidate (d') Gate-1 probe (K=32, 5,000 steps, 1 run) -- MUST complete
# before --wave keyanchor-mech below. gate_keyanchor_mech_probe reads the
# probe's result at KEYANCHOR_MECH_GATE1_JSON_DEFAULT, which is DERIVED
# from the writer's own out_path()/manifest functions (e633862 audit F1),
# so no --keyanchor-mech-gate1-json flag is needed here: writer and reader
# resolve to the same file by construction (the relative --out-dir below
# equals the argparse default under this script's own `cd` above;
# smoke_keyanchor_mech.py asserts the executed writer==reader equality).
# The probe runs with rev7_engagement=True (set in its manifest spec, not
# just claimed -- audit fold-in 1) and a scratch --ckpt-base-dir, so the
# never-yet-executed train() checkpoint-block wiring (r_e engagement,
# checkpoint writer, per-entity lambda_e trajectory) gets its first real
# GPU exercise on this cheap probe before any 20,000-step cell.
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-mech-gate1 \
    --gpus 1 --gpu-offset "$KEYANCHOR_MECH_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-mech-gate1 \
    2>&1 | tee logs/13_wave_keyanchor_mech_gate1.log

$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-mech \
    --gpus "$KEYANCHOR_MECH_GPUS" --gpu-offset "$KEYANCHOR_MECH_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    2>&1 | tee logs/14_wave_keyanchor_mech.log

$PY readout_rev7.py --out-dir results/deltanet_rd_exactness \
    2>&1 | tee logs/15_readout_rev7.log

touch results/deltanet_rd_exactness/KEYANCHOR_MECH_CHAIN_DONE
