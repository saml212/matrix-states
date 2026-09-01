#!/bin/bash
# Score the 1.31B K=16 grace wave (seeds 9-16, both recipes) with the standard battery.
# Same invocation as score_sweep1b.sh (n=9 branch), seed range extended; run AFTER the compB block completes (~14:15 UTC Sep 2).
# Usage: bash score_grace.sh [GPU]   -- one GPU, ~75-90 s/cell, 16 cells.
set -u
cd ~/ncr_scaleaxis
export CUDA_VISIBLE_DEVICES=${1:-0}
export NCR_SCALE=1310m
scored=0
for arm in primary compB; do
  for s in 9 10 11 12 13 14 15 16; do
    cell="scaleaxis1310m_grace_K16_${arm}_s${s}"
    OUT=~/ncr_scaleaxis/results/sweep1b_${cell}_kscaling.json
    [ -f "$OUT" ] && { echo "already scored: $cell"; continue; }
    RES=/ephemeral/scaleaxis1b/results/${cell}.json
    [ -f "$RES" ] || { echo "NO-RESULT $cell"; continue; }
    grep -q '"status": "COMPLETED"' "$RES" || { echo "NOT-COMPLETED $cell"; continue; }
    CK=/ephemeral/scaleaxis1b/ckpts/${cell}/${cell}.ckpt.pt
    [ -f "$CK" ] || { echo "MISSING-CKPT $cell"; continue; }
    NCR_K=16 ~/tdenv/bin/python3 kscaling_battery.py --k 16 --ckpt "$CK" --cellcfg "$RES" --tag sweep1b_${cell} 2>&1 | grep -E "^\[sweep1b|LOUD|Error|Traceback"
    scored=$((scored+1))
  done
done
echo "SCORED_NEW=$scored"; echo DONE
