#!/bin/bash
# KEY_ANCHORING_DESIGN.md sec 13 (Rev 13.2, DESIGN-CLEARED-FOR-BUILD) --
# keyanchor-dstate chain (cliff universality across d_state). NOT RUN by the
# build session -- build + CPU-verify only, no GPU work performed here. The
# orchestrator runs this once a free GPU set is confirmed (check nvidia-smi
# FIRST, per house rule) AND sec 13.6's PI-DECISION path (if ESCALATE fires)
# has a documented resolution.
#
# CALIBRATION-FIRST (sec 13.5, mandatory house rule -- this wave's OWN risk
# profile, restated in full, not merely "mirrors keyanchor-cliff"): unlike
# keyanchor-cliff (which reused an ALREADY-MEASURED d=64 K48 bracket), this
# wave has NO prior same-d measurement at all. The K=68/seed=530 calibration
# cell MUST run to completion, ALONE, before ANY other d=128 cell launches --
# `--wave keyanchor-dstate --dstate-stage full` mechanically REFUSES
# (run_deltanet_rd_exactness_sweep.py::keyanchor_dstate_stage_gate) until
# that cell exists and validates complete, then reads ITS OWN realized
# wall_s (via read_wall_s_only, sec 13.5's h4-blinding rule -- the scope
# decision below is made WITHOUT ever reading that cell's own h4) and routes
# through sec 13.6's mechanical decision table (PROCEED_FULL/OPTION_B/
# OPTION_C/ESCALATE) to pick the K-grid for the remaining manifest.
#
# set -euo pipefail is REQUIRED here (project hard rule: no `| tee`
# swallowing exits) -- under pipefail, `cmd | tee log` still propagates
# cmd's own exit code, so a failed smoke/gate/launch/decision-table-ESCALATE
# step stops this chain rather than silently continuing past a `tee`d
# failure.
#
# NO BANDS_PINNED_DSTATE-style new gate: this wave REUSES the EXISTING
# K=16/32 BANDS_PINNED.json blinding gate (candidate (d)'s own architecture/
# frame-potential-init RECIPE is unchanged by d_state alone -- only the
# table's own shape (n,d) changes; d_state is threaded through the model's
# own d_state constructor arg, not a different anchor mechanism) -- same
# reasoning keyanchor_cliff_chain.sh's own header already gives.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs

# GPUs 0-1 run rung-3 (untouchable). GPUs 2-7 free but shared CONCURRENTLY
# with other programs on this box (sec 13.6.1's own contention-diagnosis
# item 1(b)) -- override via env vars once a real GPU set is confirmed free
# (check nvidia-smi FIRST, per house rule). CONTENTION NOTE (registered
# explicitly, per this wave's own task brief): this wave's calibration cell
# is the FIRST-EVER d_state=128 measurement on this box -- if it runs
# concurrently with ANY other GPU-bound program (this design's own
# frozen-bias LM program, or a future one), the realized rate cannot
# cleanly separate a genuine d-scaling effect from ordinary shared-GPU
# contention (sec 13.4's own "2x contention contingency" already prices
# SOME of this in, but the calibration cell's OWN re-pricing, sec 13.5 item
# 3, assumes the realized rate reflects d-scaling primarily) -- run the
# calibration cell on as-quiet-as-possible GPUs if a choice exists.
KEYANCHOR_DSTATE_GPUS=${KEYANCHOR_DSTATE_GPUS:-2}
KEYANCHOR_DSTATE_GPU_OFFSET=${KEYANCHOR_DSTATE_GPU_OFFSET:-2}

mkdir -p /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-dstate

# Pre-launch CPU smoke gates (also re-run automatically by the sweep
# script's own main() -- run here too so a failure is visible before ANY
# subprocess dispatch attempt). Zero-GPU items registered by sec 13.2/13.3
# that are NOT re-run automatically by main() (Wave -1 pre-registration
# artifacts, not per-launch gates) are run here explicitly, once, ahead of
# any training cell.
$PY sim_cliff_power.py --n-trials 4000 --out sim_cliff_power_results.json \
    2>&1 | tee logs/40_sim_cliff_power_dstate.log
$PY keyanchor_dstate_niter_check.py --d-state 128 --out keyanchor_dstate_niter_check_results.json \
    2>&1 | tee logs/41_keyanchor_dstate_niter_check.log
$PY rev7_threshold_derive.py --d-state 128 --out REV7_THRESHOLD_PINNED_D128.json \
    2>&1 | tee logs/42_rev7_threshold_derive_d128.log
$PY smoke_key_anchoring.py 2>&1 | tee logs/43_smoke_key_anchoring_dstate.log
$PY smoke_keyanchor_dstate.py 2>&1 | tee logs/44_smoke_keyanchor_dstate.log
$PY gate2_construction_test.py 2>&1 | tee logs/45_gate2_construction_test_dstate.log

# STAGE: CALIBRATION -- the SINGLE K=68/seed=530 cell, alone, on this wave's
# own GPUs. This cell's realized wall_s is the ONLY input sec 13.6's
# mechanical decision table reads (sec 13.5's F9 blinding rule -- its h4 is
# quarantined until the full manifest is generated and launched).
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-dstate --dstate-stage calibration \
    --gpus "$KEYANCHOR_DSTATE_GPUS" --gpu-offset "$KEYANCHOR_DSTATE_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-dstate \
    2>&1 | tee logs/46_wave_keyanchor_dstate_calibration.log

# CHECKPOINT (sec 13.5/13.6): the calibration cell's realized wall_s is
# read (blinded to wall_s only) and routed through the mechanical decision
# table by --dstate-stage full below. `set -e` stops this chain here if the
# decision table returns ESCALATE (sys.exit(1), per sec 13.6's own
# PI-DECISION framing: this wave does NOT unilaterally descope below Option
# C) -- do NOT re-run this line with --accept-dstate-stage-override without
# a documented PI decision first (sec 13.6 items 1/2/3).
#
# STAGE: FULL -- launches whichever K-grid the decision table selects
# (PROCEED_FULL=12 cells all 4 K's, OPTION_B=9 cells drop K=84, OPTION_C=6
# cells K=76/84 only). --include-dstate-gate1 is OFF by default (sec 13.6:
# first cut under budget pressure, same discipline as keyanchor-cliff's own
# --include-cliff-gate1).
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-dstate --dstate-stage full \
    --gpus "$KEYANCHOR_DSTATE_GPUS" --gpu-offset "$KEYANCHOR_DSTATE_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-dstate \
    2>&1 | tee logs/47_wave_keyanchor_dstate_full.log

# Real-data fit (sec 13.3 item 8) -- ANCHOR-FREE at d=128 (no archived
# K=16/32/48-equivalent flanking point exists at this d, sec 13.2's own
# disclosure) -- --k32-dir/--k48-dir are OMITTED entirely (never passed at
# any d != 64, fit_cliff_curve.py's own argparse asserts this). --k-grid is
# REQUIRED (no silent d=64 default) -- pass the SAME K-grid the 'full'
# stage above actually launched (read from CALIBRATION_DONE's own decision
# block if a subset branch fired; the full {68,76,84,92} shown below as the
# PROCEED_FULL example).
$PY fit_cliff_curve.py \
    --cliff-out-dir results/deltanet_rd_exactness/wavekeyanchor-dstate \
    --d-state 128 --k-grid 68 76 84 92 \
    --n-trials 4000 --out fit_cliff_curve_d128_results.json \
    2>&1 | tee logs/48_fit_cliff_curve_d128.log

touch results/deltanet_rd_exactness/KEYANCHOR_DSTATE_CHAIN_DONE
