#!/usr/bin/env bash
set -u
cd ~/ncr_scaleaxis
export CUDA_VISIBLE_DEVICES=0
for arm in primary compB; do
  cell="scaleaxis1310m_K24_${arm}_s0"
  CK=$(ls /ephemeral/scaleaxis1b/ckpts/*${cell}*/*.ckpt.pt 2>/dev/null | tail -1)
  [ -f "${CK:-/nonexistent}" ] || { echo "MISSING-CKPT $cell"; continue; }
  NCR_SCALE=1310m NCR_K=24 ~/tdenv/bin/python3 kscaling_battery.py --k 24 \
    --ckpt "$CK" --cellcfg /ephemeral/scaleaxis1b/results/${cell}.json \
    --tag calib1b_${cell} 2>&1 | grep -E "^\[calib1b|LOUD"
done
echo CALIB1B_SCORING_DONE
