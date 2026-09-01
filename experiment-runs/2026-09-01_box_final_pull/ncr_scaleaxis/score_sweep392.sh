#!/usr/bin/env bash
set -u
cd ~/ncr_scaleaxis
export CUDA_VISIBLE_DEVICES=${1:-0}
scored=0
for K in 16 32 40; do
  for arm in primary compB; do
    for s in 0 1 2; do
      cell="scaleaxis392m_K${K}_${arm}_s${s}"
      OUT=~/ncr_scaleaxis/results/sweep_${cell}_kscaling.json
      [ -f "$OUT" ] && continue
      RES=/ephemeral/scaleaxis/results/${cell}.json
      [ -f "$RES" ] || continue
      grep -q "\"status\": \"COMPLETED\"" "$RES" || continue
      CK=$(ls /ephemeral/scaleaxis/ckpts/*${cell}*/*.ckpt.pt 2>/dev/null | tail -1)
      [ -f "${CK:-/nonexistent}" ] || { echo "MISSING-CKPT $cell"; continue; }
      NCR_SCALE=392m NCR_K=$K ~/tdenv/bin/python3 kscaling_battery.py --k $K \
        --ckpt "$CK" --cellcfg "$RES" --tag sweep_${cell} 2>&1 | grep -E "^\[sweep|LOUD"
      scored=$((scored+1))
    done
  done
done
echo "SCORED=$scored"
