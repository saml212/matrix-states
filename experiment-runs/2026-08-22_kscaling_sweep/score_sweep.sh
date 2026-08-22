#!/usr/bin/env bash
set -u
cd ~/ncr_kscaling
export CUDA_VISIBLE_DEVICES=1
scored=0; missing=0
for K in 12 16 20 28; do
  for arm in primary compB; do
    for s in 0 1 2; do
      cell="kscaling_K${K}_${arm}_s${s}"
      CK=$(ls /ephemeral/kscaling/ckpts/*${cell}*/*.ckpt.pt 2>/dev/null | head -1)
      [ -f "${CK:-/nonexistent}" ] || { echo "MISSING-CKPT $cell"; missing=$((missing+1)); continue; }
      NCR_K=$K ~/tdenv/bin/python3 kscaling_battery.py --k $K \
        --ckpt "$CK" --cellcfg /ephemeral/kscaling/results/${cell}.json \
        --tag sweep_${cell} 2>&1 | grep -E "^\[sweep|LOUD|Traceback" && scored=$((scored+1))
    done
  done
done
for arm in primary compB; do
  for s in 0 1 2; do
    cell="mob_g3b31_${arm}_s${s}"
    CK=""
    for cand in "/ephemeral/reseed_ckpts/${cell}_ckpts/${cell}.ckpt.pt" "/home/nvidia/ncr_g3b31_contrastive/results/${cell}_ckpts/${cell}.ckpt.pt"; do
      [ -f "$cand" ] && { CK="$cand"; break; }
    done
    [ -n "$CK" ] || { echo "MISSING-CKPT $cell"; missing=$((missing+1)); continue; }
    NCR_K=24 ~/tdenv/bin/python3 kscaling_battery.py --k 24 \
      --ckpt "$CK" --cellcfg /home/nvidia/ncr_g3b31_contrastive/results/${cell}.json \
      --tag anchor_${cell} --anchor-runner-tag ncr_gate3_wave1_runner_v1 2>&1 | grep -E "^\[anchor|LOUD|Traceback" && scored=$((scored+1))
  done
done
echo "SCORED=$scored MISSING=$missing"
echo SWEEP_SCORING_DONE
