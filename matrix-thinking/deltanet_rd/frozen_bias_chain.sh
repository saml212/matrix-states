#!/bin/bash
# FROZEN_BIAS_LM_DESIGN.md (ROUND-5, RUNG-1-ONLY, DESIGN-CLEARED-FOR-BUILD) -- frozen-bias LM
# program chain. NOT RUN by this build session -- build + CPU-verify only, no GPU work performed
# here. The orchestrator runs this once a free GPU set is confirmed (check nvidia-smi FIRST, per
# house rule) AND sec 8.2a's own contention gate clears (see below -- mechanically re-checked by
# frozen_bias_lm_sweep.py's own --calibration-only launch, NOT an honor-system flag here).
#
# STAGE ORDER (sec 6.3's mandatory calibration-before-sweep discipline, mirroring
# keyanchor_cliff_chain.sh's own staged-launch structure): the calibration cell (Arm 2,
# openr1-mix-ext, seed 0, lambda=0.58) launches FIRST AND ALONE -- inspect its own span_frac
# trajectory and val-loss curve (sec 6.3) BEFORE the remaining 19 cells launch. This chain does
# NOT auto-chain past the calibration cell (unlike keyanchor_cliff_chain.sh's fully-automated
# stage-1-then-stage-2 sequence) -- sec 6.3's own "inspect... only then launch" text is a human
# judgment call this script deliberately stops for, not automates past. A supervisor wrapper (the
# `while [ ! -f STOP ]; do ... done` pattern) is appropriate around EACH of the two `python
# frozen_bias_lm_sweep.py` invocations below individually, not around this whole script, so a
# crash mid-calibration does not silently skip the human-inspection checkpoint.
#
# set -euo pipefail is REQUIRED here (project hard rule: no `| tee` swallowing exits) -- under
# pipefail, `cmd | tee log` still propagates cmd's own exit code, so a failed smoke/gate/launch
# step stops this chain rather than silently continuing past a `tee`d failure.
#
# sec 8.2a's contention gate (mirrored from KEY_ANCHORING_DESIGN.md sec 12.2.3): the calibration
# cell below is gated on the K-anchoring wave's own Stage-1 clearance sentinel
# (STAGE1_SENTINEL below, a parameter with the design's own registered default path) -- checked
# MECHANICALLY by frozen_bias_lm_sweep.py's own contention_gate() (loud refusal, sys.exit(4), NOT
# an honor-system flag in this shell script). --accept-contention-override is available as an
# explicit escape hatch on the calibration-cell invocation ONLY if a human has independently
# confirmed the compounded-contention risk is acceptable.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs

# GPUs 0-1 run rung-3 (untouchable, per STATE.md). GPUs 2-7 free but shared CONCURRENTLY with the
# K-anchoring wave (sec 8.2's own GPU plan) -- override via env vars once a real GPU set is
# confirmed free (check nvidia-smi FIRST, per house rule).
FROZENBIAS_GPUS=${FROZENBIAS_GPUS:-2}
FROZENBIAS_GPU_OFFSET=${FROZENBIAS_GPU_OFFSET:-2}
FROZENBIAS_OUT_DIR=${FROZENBIAS_OUT_DIR:-results/frozen_bias_lm}
FROZENBIAS_DATA_DIR=${FROZENBIAS_DATA_DIR:-/data/deltanet_rd_data}
FROZENBIAS_CKPT_BASE=${FROZENBIAS_CKPT_BASE:-/data/deltanet_rd_frozenbias_ckpts}
# sec 8.2a's own registered sentinel path (as seen from THIS box's deltanet_rd dir) -- the
# K-anchoring wave's own Stage-1 clearance file. Override only if that wave's own out-dir differs
# from its own registered default.
STAGE1_SENTINEL=${STAGE1_SENTINEL:-results/deltanet_rd_exactness/wavekeyanchor-cliff/STAGE1_RATES_OK}
# --rung1-steps is NOT hardcoded here -- sec 6.3's calibration cell itself is what DERIVES a real
# step count; this chain requires it as an explicit env var so a human has actually looked at a
# calibration run's own measured tok/s before choosing it (the SAME "no silent placeholder"
# discipline run_lm_rd_trackc_sweep.py's own --rungN-steps enforces). NOT set here: this script
# will refuse (frozen_bias_lm_sweep.py's own --rung1-steps required-arg check) until a human sets
# it, e.g. `FROZENBIAS_RUNG1_STEPS=20000 ./frozen_bias_chain.sh`.
: "${FROZENBIAS_RUNG1_STEPS:?FROZENBIAS_RUNG1_STEPS must be set (derive from a real calibration runs own measured tok/s -- sec 6.3, no silent placeholder)}"

mkdir -p "$FROZENBIAS_OUT_DIR"
mkdir -p "$FROZENBIAS_CKPT_BASE"

# Pre-launch CPU smoke gates (also re-run automatically by frozen_bias_lm_sweep.py's own
# _run_manifest -- run here too so a failure is visible before ANY subprocess dispatch attempt,
# mirroring keyanchor_cliff_chain.sh's own precedent).
$PY smoke_frozen_bias_wave_neg1.py 2>&1 | tee logs/50_smoke_frozen_bias_wave_neg1.log
$PY smoke_frozen_bias_lm.py 2>&1 | tee logs/51_smoke_frozen_bias_lm.log
$PY fit_frozenbias_estimation.py --self-test 2>&1 | tee logs/52_fit_frozenbias_estimation_selftest.log

# CALIBRATION CELL (sec 6.3, sec 8.2a's own gated item): launches FIRST AND ALONE. The contention
# gate (sec 8.2a) is checked HERE, mechanically, by frozen_bias_lm_sweep.py's own
# --calibration-only path -- refuses (sys.exit(4)) unless STAGE1_SENTINEL exists, or
# --accept-contention-override is passed explicitly (NOT default in this chain -- add the flag
# below only after a human has confirmed the compounded-contention risk is acceptable).
$PY frozen_bias_lm_sweep.py --wave rung1 --calibration-only \
    --rung1-steps "$FROZENBIAS_RUNG1_STEPS" \
    --out-dir "$FROZENBIAS_OUT_DIR" --data-dir "$FROZENBIAS_DATA_DIR" \
    --ckpt-base-dir "$FROZENBIAS_CKPT_BASE" \
    --gpus 1 --gpu-offset "$FROZENBIAS_GPU_OFFSET" \
    --stage1-sentinel "$STAGE1_SENTINEL" \
    2>&1 | tee logs/53_frozenbias_calibration.log

# ------------------------------------------------------------------------------------------------
# HUMAN CHECKPOINT (sec 6.3, non-negotiable, NOT automated past): inspect the calibration cell's
# own span_frac trajectory and val-loss curve (its result JSON under $FROZENBIAS_OUT_DIR) BEFORE
# proceeding. This script deliberately STOPS here -- uncomment the block below (or re-invoke this
# script with an env var gate of your own) only after that inspection is done.
# ------------------------------------------------------------------------------------------------
echo "=========================================================================="
echo "CALIBRATION CELL COMPLETE. Inspect its span_frac trajectory + val-loss curve"
echo "under $FROZENBIAS_OUT_DIR (sec 6.3) BEFORE launching the remaining 19 cells."
echo "Re-run this script with FROZENBIAS_LAUNCH_REMAINING=1 once inspected."
echo "=========================================================================="

if [ "${FROZENBIAS_LAUNCH_REMAINING:-0}" != "1" ]; then
    echo "FROZENBIAS_LAUNCH_REMAINING not set to 1 -- stopping after the calibration cell (sec 6.3's"
    echo "own mandatory human-inspection checkpoint). Nothing further launched."
    exit 0
fi

# REMAINING 19 CELLS (sec 6.1's full manifest minus the calibration cell) -- gated on the
# calibration cell's own completion (frozen_bias_lm_sweep.py's own is_done_cell check, NOT an
# honor-system flag) via the SAME --rung1-steps (must match the calibration cell's own step count
# exactly -- a different count would silently reintroduce a comparability confound, mirroring
# run_lm_rd_trackc_sweep.py's own --wave 1ext discipline for its analogous re-run-at-fixed-steps
# requirement).
$PY frozen_bias_lm_sweep.py --wave rung1 \
    --rung1-steps "$FROZENBIAS_RUNG1_STEPS" \
    --out-dir "$FROZENBIAS_OUT_DIR" --data-dir "$FROZENBIAS_DATA_DIR" \
    --ckpt-base-dir "$FROZENBIAS_CKPT_BASE" \
    --gpus "$FROZENBIAS_GPUS" --gpu-offset "$FROZENBIAS_GPU_OFFSET" \
    2>&1 | tee logs/54_frozenbias_remaining19.log

touch "$FROZENBIAS_OUT_DIR/FROZENBIAS_RUNG1_CHAIN_DONE"
