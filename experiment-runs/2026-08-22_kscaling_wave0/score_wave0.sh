#!/usr/bin/env bash
set -u
cd ~/ncr_kscaling
export CUDA_VISIBLE_DEVICES=2
for cell in primary_s0 primary_s1 primary_s2 compB_s0 compB_s1 compB_s2; do
  CK=/ephemeral/kscaling/ckpts/kscaling_K32_${cell}_ckpts/kscaling_K32_${cell}.ckpt.pt
  [ -f "$CK" ] || CK=$(ls /ephemeral/kscaling/ckpts/*${cell}*/*.ckpt.pt 2>/dev/null | head -1)
  [ -f "$CK" ] || { echo "MISSING-CKPT $cell"; continue; }
  NCR_K=32 ~/tdenv/bin/python3 kscaling_battery.py --k 32 \
    --ckpt "$CK" --cellcfg /ephemeral/kscaling/results/kscaling_K32_${cell}.json \
    --tag k32_wave0_${cell} 2>&1 | grep -E "^\[k32|LOUD|FAIL"
done
echo WAVE0_SCORING_DONE
