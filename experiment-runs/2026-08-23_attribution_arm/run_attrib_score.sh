#!/usr/bin/env bash
# Score attribution cells at their EXTENDED (40,000-step) checkpoints.
#   Usage: run_attrib_score.sh <shard> <n_shards> <gpu>
# Two instruments per cell, both at --required-step 40000:
#   (1) kscaling_battery.py  -> Curve 1 (h_top) / Curve 4 (h_fix) / wall  [V1,V2]
#   (2) depthext6_392m_driver.py -> six-rung {5,7,9,11,13,15} ladder      [V3]
# NCR_SCALE=392m by env (the battery takes no --scale flag).
set -u
cd /home/nvidia/ncr_scaleaxis
SHARD="$1"; NSHARD="$2"; GPU="$3"
export CUDA_VISIBLE_DEVICES="$GPU"
export NCR_SCALE=392m
PY=/home/nvidia/tdenv/bin/python3
MANIFEST="${ATTRIB_MANIFEST:-/home/nvidia/ncr_scaleaxis/attrib_manifest.tsv}"
OUT=/home/nvidia/ncr_scaleaxis/results_attrib
mkdir -p "$OUT"
i=0; ok=0; bad=0
while IFS=$'\t' read -r k rec s name ck cfg status step; do
  [ -z "${k:-}" ] && continue
  if [ $(( i % NSHARD )) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  BA=(--k "$k" --ckpt "$ck" --tag "attrib_$name" --required-step 40000 --outdir "$OUT")
  DA=(--k "$k" --ckpt "$ck" --tag "depthext6_attrib_$name" --required-step 40000 --outdir "$OUT")
  [ "$cfg" != "-" ] && { BA+=(--cellcfg "$cfg"); DA+=(--cellcfg "$cfg"); }
  NCR_K="$k" "$PY" kscaling_battery.py "${BA[@]}"      || { echo "!!! FAILED battery $name ($?)"; bad=$((bad+1)); }
  NCR_K="$k" "$PY" depthext6_392m_driver.py "${DA[@]}" || { echo "!!! FAILED depthext $name ($?)"; bad=$((bad+1)); }
  ok=$((ok+1))
done < "$MANIFEST"
echo "SHARD $SHARD DONE cells=$ok errs=$bad"
