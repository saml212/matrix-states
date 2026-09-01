#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
cd ~/ncr_scaleaxis
cell=$(ls /ephemeral/scaleaxis1b/results/ | grep -v calib | head -1 | sed "s/.json//")
grep -o "\"status\": \"[A-Z]*\"" /ephemeral/scaleaxis1b/results/${cell}.json | head -1
CK=$(ls /ephemeral/scaleaxis1b/ckpts/*${cell}*/*.ckpt.pt 2>/dev/null | tail -1)
echo "cell=$cell ck=$CK"
K=$(echo $cell | grep -o "K[0-9]*" | tr -d K)
NCR_SCALE=1310m NCR_K=$K ~/tdenv/bin/python3 kscaling_battery.py --k $K \
  --ckpt "$CK" --cellcfg /ephemeral/scaleaxis1b/results/${cell}.json \
  --tag diag_${cell} 2>&1 | tail -5
