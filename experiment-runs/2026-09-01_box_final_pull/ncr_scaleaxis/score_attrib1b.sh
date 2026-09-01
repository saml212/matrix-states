#!/usr/bin/env bash
set -u
cd ~/ncr_scaleaxis
export CUDA_VISIBLE_DEVICES=0
for K in 32 40; do
  for s in 0 1 2; do
    cell="attrib1b_K${K}_compB_s${s}"
    RES=$(ls /ephemeral/scaleaxis1b/attrib/results/*K${K}*compB_s${s}*.json 2>/dev/null | head -1)
    [ -f "${RES:-/nonexistent}" ] || RES=$(ls /ephemeral/scaleaxis1b/*/results/*attrib*K${K}*s${s}*.json 2>/dev/null | head -1)
    [ -f "${RES:-/nonexistent}" ] || { echo "MISSING-RES $cell"; continue; }
    CK=$(ls $(dirname $(dirname $RES))/ckpts/*K${K}*compB_s${s}*/*.ckpt.pt 2>/dev/null | tail -1)
    [ -f "${CK:-/nonexistent}" ] || { echo "MISSING-CKPT $cell"; continue; }
    NCR_SCALE=1310m NCR_K=$K ~/tdenv/bin/python3 kscaling_battery.py --k $K \
      --ckpt "$CK" --cellcfg "$RES" --tag ${cell} --required-step 40000 2>&1 | tail -2
  done
done
echo ATTRIB1B_SCORING_DONE
