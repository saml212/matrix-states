#!/usr/bin/env bash
set -u
cd ~/ncr_scaleaxis
export CUDA_VISIBLE_DEVICES=0
for s in 0 1; do
  cell="v2prime_K40_compB_s${s}"
  CK=$(ls /ephemeral/scaleaxis/v2prime/ckpts/*${cell}*/*.ckpt.pt 2>/dev/null | tail -1)
  [ -f "${CK:-/nonexistent}" ] || { echo "MISSING-CKPT $cell"; continue; }
  NCR_SCALE=392m NCR_K=40 ~/tdenv/bin/python3 kscaling_battery.py --k 40 \
    --ckpt "$CK" --cellcfg /ephemeral/scaleaxis/v2prime/results/${cell}.json \
    --tag v2p_${cell} --required-step 40000 2>&1 | grep -E "^\[v2p|LOUD"
done
echo V2P_SCORING_DONE
