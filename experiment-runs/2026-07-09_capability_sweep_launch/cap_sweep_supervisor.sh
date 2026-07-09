#!/usr/bin/env bash
# Self-healing supervisor: Stage 1 = capability-sep 58-cell --sweep,
# Stage 2 (chained, automatic) = attractor-robustness 2x2 n=3 escalation.
# GPU 0 ONLY -- never touches GPUs 1-7 (fixscale_pilots / h2h_calib3 / reserved).
#
# Authorization: matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md S1.32
# (commit 0179b73) -- "SWEEP-READY -> --sweep AUTHORIZED"; CAPABILITY_SEP_
# PI_SIGNOFF=1 below cites that section as the recorded decision point.
#
# Pattern: CLAUDE.md's `while [ ! -f STOP ]; do <cmd>; sleep 15; done`
# self-healing loop. Both run_capability_sep.py and
# run_attractor_robustness_2x2.py are resume-safe (validity-checked output,
# not existence), so re-invoking after a crash costs nothing beyond the
# crashed cell's partial work.
set -uo pipefail

CAP_DIR=/home/nvidia/chapter2/capability_separation
DN_DIR=/home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python

# ---------------------------------------------------------------------
# STAGE 1: 58-cell sweep (58 cells, ~2.5-4.6 GPU-h projected, 30 GPU-h cap)
# ---------------------------------------------------------------------
cd "$CAP_DIR"
rm -f STOP
export CUDA_VISIBLE_DEVICES=0
export CAPABILITY_SEP_PI_SIGNOFF=1   # S1.32 authorization (see header)

echo "[supervisor] === STAGE 1: 58-cell sweep START $(date -u) ===" | tee -a cap_sweep.log
while [ ! -f STOP ]; do
  "$PY" run_capability_sep.py --sweep --device cuda 2>&1 | tee -a cap_sweep.log
  RC=${PIPESTATUS[0]}
  if [ "$RC" -eq 0 ]; then
    echo "[supervisor] sweep exited 0 -- STAGE 1 done $(date -u)" | tee -a cap_sweep.log
    touch STOP
  else
    echo "[supervisor] sweep exited $RC -- retrying in 15s ($(date -u))" | tee -a cap_sweep.log
    sleep 15
  fi
done
touch "$CAP_DIR/results/SWEEP_STAGE_DONE"
echo "[supervisor] === STAGE 1 COMPLETE $(date -u) === chaining to STAGE 2 ===" | tee -a "$CAP_DIR/cap_sweep.log"

# ---------------------------------------------------------------------
# STAGE 2 (CHAINED, automatic): attractor-robustness 2x2 n=3 escalation.
# 12-cell pre-registered wave (4 already done at s0 from screening -> only
# 8 new s1/s2 cells actually train), ceiling 3.03 GPU-h, ~2.02 GPU-h
# incremental. Fires because escalation.fire=true in the existing
# AGGREGATE.json (qkTrue_gateTrue delta 6.16 > 2x noise-floor threshold
# 4.489, pre-registered rule).
# ---------------------------------------------------------------------
cd "$DN_DIR"
rm -f STOP_2x2_esc
echo "[supervisor] === STAGE 2: 2x2 n=3 escalation START $(date -u) ===" | tee -a cap_sweep_2x2_esc.log
while [ ! -f STOP_2x2_esc ]; do
  "$PY" run_attractor_robustness_2x2.py --run --n-seeds 3 \
    --out-dir results/attractor_robustness_2x2 \
    --data-dir /data/deltanet_rd_data --device cuda \
    2>&1 | tee -a cap_sweep_2x2_esc.log
  RC=${PIPESTATUS[0]}
  if [ "$RC" -eq 0 ]; then
    echo "[supervisor] 2x2 escalation exited 0 -- STAGE 2 done $(date -u)" | tee -a cap_sweep_2x2_esc.log
    touch STOP_2x2_esc
  else
    echo "[supervisor] 2x2 escalation exited $RC -- retrying in 15s ($(date -u))" | tee -a cap_sweep_2x2_esc.log
    sleep 15
  fi
done
touch "$DN_DIR/results/attractor_robustness_2x2/ESCALATION_STAGE_DONE"
echo "[supervisor] === STAGE 2 COMPLETE -- ALL DONE $(date -u) ===" | tee -a "$DN_DIR/cap_sweep_2x2_esc.log"
