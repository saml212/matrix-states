#!/usr/bin/env bash
set -u
cd ~/ncr_scaleaxis
export CUDA_VISIBLE_DEVICES=0
for arm in primary compB; do
  for s in 0 1 2; do
    cell="scaleaxis392m_K24_${arm}_s${s}"
    CK=$(ls /ephemeral/scaleaxis/ckpts/*${cell}*/*.ckpt.pt 2>/dev/null | grep -v step | head -1)
    [ -f "${CK:-/nonexistent}" ] || CK=$(ls /ephemeral/scaleaxis/ckpts/*${cell}*/*.ckpt.pt 2>/dev/null | tail -1)
    [ -f "${CK:-/nonexistent}" ] || { echo "MISSING-CKPT $cell"; continue; }
    NCR_SCALE=392m NCR_K=24 ~/tdenv/bin/python3 kscaling_battery.py --k 24 \
      --ckpt "$CK" --cellcfg /ephemeral/scaleaxis/results/${cell}.json \
      --tag calib_${cell} 2>&1 | tail -3
  done
done
echo CALIB_SCORING_DONE
