#!/bin/bash
# KEY_ANCHORING_DESIGN.md sec 14 (Rev 14.3, DESIGN-CLEARED-FOR-BUILD) --
# keyanchor-dose chain (coherence dose-response wave). NOT RUN by the build
# session -- build + CPU-verify only, no GPU work performed here. The
# orchestrator runs this once a free GPU set is confirmed (check nvidia-smi
# FIRST, per house rule).
#
# CALIBRATION-FIRST (sec 14.4b, mandatory house rule): the single most
# informative cell -- K=68, dose=0.40 (super-d64), rank4 -- runs FIRST AND
# ALONE. `--wave keyanchor-dose --dose-stage rank4` mechanically REFUSES
# (run_deltanet_rd_exactness_sweep.py::main()'s own keyanchor-dose dispatch,
# via keyanchor_dose_is_calibration_dose_verified) until that cell exists,
# validates complete, AND its own dose re-verifies within +/-10% of target
# (a SECOND, orchestrator-level check on top of run_deltanet_rd.py's own
# construction-time assert, sec 14.3) -- the scope decision is never made
# off a peek at h4 (sec 14.4b's blinding rule, inherited from sec 13.5's F9
# fix): only wall_s/dose fields are read before Stage 1's own 9 cells launch.
#
# NOTE on the calibration cell's own dose (flagged, not silently resolved):
# sec 14.4's own budget TABLE lists dose=0.284 for this row, while sec
# 14.4b's own prose states "the HIGHEST dose (~0.40, super-d64)" THREE
# times as the calibration cell's identity ("the single most informative
# cell... the ALREADY-VALIDATED construction from Rev 14.1... this exact
# cell type, at this exact dose, was the Rev 14.1 calibration cell too").
# This build follows the prose (dose=0.40) -- see
# run_deltanet_rd_exactness_sweep.py's own module-level comment immediately
# above KEYANCHOR_DOSE_D_STATE for the full disclosure. Flagged for the
# design doc's own next revision, not resolved either way by this chain.
#
# STAGING (sec 14.4 Option 1, the registered mechanical default): Stage 1 =
# calibration (1 cell, alone) -> the 9 remaining rank-4 grid cells (3 doses
# x 3 seeds) -- fits the FULL 13.68 GPU-h wave ceiling at BOTH 1x and 2x
# contingency with ZERO PI decision required (6.410/12.820 GPU-h). Stage 2
# (diffuse, 9 co-primary cells) is NOT auto-launched here -- it requires an
# EXPLICIT PI decision (sec 14.4 Option 1's own abort/decision table: "does
# the co-primary result matter enough... to justify a +10.678 GPU-h ceiling
# amendment") recorded via KEYANCHOR_DOSE_STAGE2_PI_SIGNOFF=1 (this chain's
# own env-override convention, documented below as the PI-ask path) --
# commented OUT by default so a bare run of this script never launches
# Stage 2 without that decision being made explicitly, ahead of time, by a
# human reading this exact comment block.
#
# set -euo pipefail is REQUIRED here (project hard rule: no `| tee`
# swallowing exits) -- under pipefail, `cmd | tee log` still propagates
# cmd's own exit code, so a failed smoke/gate/launch/dose-reverification
# step stops this chain rather than silently continuing past a `tee`d
# failure.
#
# NO NEW BANDS_PINNED-style gate: this wave REUSES the EXISTING K=16/32
# BANDS_PINNED.json blinding gate (candidate (d)'s own architecture is
# UNCHANGED by the dosed-table override alone -- only the anchor table's
# own row CONTENT differs, sec 14.2) -- same reasoning keyanchor_dstate_
# chain.sh's own header already gives.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs

# GPUs 0-1 run rung-3 (untouchable). GPUs 2-7 free but shared CONCURRENTLY
# with other programs on this box -- override via env vars once a real GPU
# set is confirmed free (check nvidia-smi FIRST, per house rule).
KEYANCHOR_DOSE_GPUS=${KEYANCHOR_DOSE_GPUS:-2}
KEYANCHOR_DOSE_GPU_OFFSET=${KEYANCHOR_DOSE_GPU_OFFSET:-2}

mkdir -p /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-dose

# Pre-launch CPU smoke gates (also re-run automatically by the sweep
# script's own main() -- run here too so a failure is visible before ANY
# subprocess dispatch attempt).
$PY smoke_key_anchoring.py 2>&1 | tee logs/50_smoke_key_anchoring_dose.log
$PY smoke_keyanchor_dose.py 2>&1 | tee logs/51_smoke_keyanchor_dose.log
$PY gate2_construction_test.py 2>&1 | tee logs/52_gate2_construction_test_dose.log

# STAGE: CALIBRATION -- the SINGLE K=68/dose=0.40/rank4/seed=939 cell,
# alone, on this wave's own GPUs. This cell's realized wall_s/achieved dose
# are the ONLY inputs read before Stage 1's own 9 grid cells launch (sec
# 14.4b's blinding rule -- h4 is quarantined until the full manifest exists).
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-dose --dose-stage calibration \
    --gpus "$KEYANCHOR_DOSE_GPUS" --gpu-offset "$KEYANCHOR_DOSE_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-dose \
    2>&1 | tee logs/53_wave_keyanchor_dose_calibration.log

# CHECKPOINT (sec 14.4b): the calibration cell's own completeness AND dose
# re-verification (+/-10%, a second orchestrator-level check on top of
# run_deltanet_rd.py's own construction-time assert) are re-checked by
# `--dose-stage rank4` below. `set -e` stops this chain here if that check
# fails (sys.exit(1)) -- do NOT re-run with --accept-dose-stage-override
# without first understanding WHY the calibration cell's dose failed to
# re-verify (this should never happen given the construction-time assert
# already ran; a mismatch here means the result JSON was corrupted or
# tampered with post-hoc, not an expected outcome).
#
# STAGE: RANK4 -- the 9 mandatory rank-4 grid cells (3 doses x 3 seeds).
# Fits the full wave ceiling alone (Stage 1 = calibration + rank4 = 10
# cells, 6.410/12.820 GPU-h at 1x/2x) -- launches with ZERO PI decision.
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-dose --dose-stage rank4 \
    --gpus "$KEYANCHOR_DOSE_GPUS" --gpu-offset "$KEYANCHOR_DOSE_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-dose \
    2>&1 | tee logs/54_wave_keyanchor_dose_rank4.log

touch results/deltanet_rd_exactness/KEYANCHOR_DOSE_STAGE1_DONE

# =============================================================================
# STAGE: DIFFUSE (Stage 2, the co-primary structure) -- NOT AUTO-LAUNCHED.
#
# sec 14.4 Option 1's own registered PI-decision point: Stage 2's cumulative
# cost (both stages, 19 cells) exceeds the 13.68 GPU-h wave sub-ledger
# ceiling at 2x contingency by 10.678 GPU-h (small in REAL hardware terms
# against the Brev grant's own ~192 GPU-h/day uptime-metered window, sec
# 14.4's own F6 disclosure -- but per this project's "no ask is trivially
# waved through" standing rule, not launched here without that decision on
# record). This is the PI-ASK PATH sec 14.4 Option 1 requires: a human
# reviewer reads Stage 1's own rank-4 dose-response result (CONFIRM /
# EXONERATE / STRUCTURE-DEPENDENT / degenerate, sec 14.0) and DECIDES
# whether the co-primary diffuse result is worth the +10.678 GPU-h ask,
# per sec 14.4 Option 1's own abort/decision table ("if stage 1 alone reads
# CONFIRM or a clean EXONERATE at the top dose... stage 2 MAY be deferred
# rather than mandatory").
#
# Stage 2 runs ONLY when KEYANCHOR_DOSE_STAGE2_PI_SIGNOFF=1 is set in the
# environment (the PI-signoff record; a bare `bash keyanchor_dose_chain.sh`
# run still never launches Stage 2 by accident). PI SIGNOFF RECORDED
# 2026-07-07 ~02:15 UTC: user present at check-in, directed full GPU
# capacity with the queued gauntleted work; launch authorized at 1x
# (measured-rate) budget within the EXISTING 80 GPU-h ceiling -- the
# in-code budget guard + per-cell abort enforce the ceiling mechanically;
# no amendment made.
if [ "${KEYANCHOR_DOSE_STAGE2_PI_SIGNOFF:-0}" = "1" ]; then
    KEYANCHOR_DOSE_STAGE2_PI_SIGNOFF=1 $PY run_deltanet_rd_exactness_sweep.py \
        --wave keyanchor-dose --dose-stage diffuse \
        --gpus "$KEYANCHOR_DOSE_GPUS" --gpu-offset "$KEYANCHOR_DOSE_GPU_OFFSET" --per-gpu 1 \
        --out-dir results/deltanet_rd_exactness \
        --ckpt-base-dir /data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-dose \
        2>&1 | tee logs/55_wave_keyanchor_dose_diffuse.log

    touch results/deltanet_rd_exactness/KEYANCHOR_DOSE_CHAIN_DONE
fi
# =============================================================================
