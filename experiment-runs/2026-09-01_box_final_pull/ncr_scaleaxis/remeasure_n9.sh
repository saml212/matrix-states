#!/bin/bash
# Pre-registered single-seed wall-excursion re-measure (KSCALING 7.2 / 6.1 Curve 2): base seed 31337.
# Cells: the four NEW single-seed P0 excursions in the n=9 harvest (2026-09-01 recompute from raw JSONs).
# Byte-pattern identical to run_1b_reserved.sh payload 2/3 (the K16 s2 re-measure), cell substituted.
set -u
cd /home/nvidia/ncr_scaleaxis
PY=/home/nvidia/tdenv/bin/python3
export NCR_SCALE=1310m
BATT=/home/nvidia/ncr_scaleaxis/results
LOG=/home/nvidia/ncr_scaleaxis/remeasure_n9.log
i=0
for spec in 24:scaleaxis1310m_n9_K24_primary_s7 32:scaleaxis1310m_n9_K32_compB_s6 32:scaleaxis1310m_n9_K32_compB_s7 40:scaleaxis1310m_thicken_K40_compB_s5; do
  k=${spec%%:*}; cell=${spec#*:}
  if [ -f "$BATT/remeasure1b_${cell}_kscaling.json" ]; then echo "already re-measured: $cell" | tee -a "$LOG"; continue; fi
  echo "[$(date -u +%FT%TZ)] START remeasure $cell (K=$k, GPU $i)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$i NCR_K=$k "$PY" kscaling_battery.py --k "$k" \
      --ckpt "/ephemeral/scaleaxis1b/ckpts/${cell}/${cell}.ckpt.pt" \
      --cellcfg "/ephemeral/scaleaxis1b/results/${cell}.json" \
      --base-seed 31337 --tag "remeasure1b_${cell}" >> "$LOG" 2>&1 &
  i=$((i+1))
done
wait
echo "[$(date -u +%FT%TZ)] REMEASURE_N9_DONE" | tee -a "$LOG"
