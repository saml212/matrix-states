#!/usr/bin/env bash
set -u
cd ~/ncr_scaleaxis
export CUDA_VISIBLE_DEVICES=${1:-0}
scored=0
for K in 16 24 32 40; do
  for arm in primary compB; do
    for s in 0 1 2 3 4 5 6 7 8; do
      cell="scaleaxis1310m_K${K}_${arm}_s${s}"; [ $s -ge 3 ] && [ $s -le 5 ] && cell="scaleaxis1310m_thicken_K${K}_${arm}_s${s}"; [ $s -ge 6 ] && cell="scaleaxis1310m_n9_K${K}_${arm}_s${s}"; [ "$K" = "24" ] && [ "$s" = "0" ] && continue
      OUT=~/ncr_scaleaxis/results/sweep1b_${cell}_kscaling.json
      [ -f "$OUT" ] && continue
      RES=/ephemeral/scaleaxis1b/results/${cell}.json
      [ -f "$RES" ] || continue
      grep -q "\"status\": \"COMPLETED\"" "$RES" || continue
      CK=$(ls /ephemeral/scaleaxis1b/ckpts/*${cell}*/*.ckpt.pt 2>/dev/null | tail -1)
      [ -f "${CK:-/nonexistent}" ] || { echo "MISSING-CKPT $cell"; continue; }
      NCR_SCALE=1310m NCR_K=$K ~/tdenv/bin/python3 kscaling_battery.py --k $K \
        --ckpt "$CK" --cellcfg "$RES" --tag sweep1b_${cell} 2>&1 | grep -E "^\[sweep1b|LOUD"
      scored=$((scored+1))
    done
  done
done
echo "SCORED_NEW=$scored"
