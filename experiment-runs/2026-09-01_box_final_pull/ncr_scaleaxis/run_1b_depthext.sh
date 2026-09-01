#!/usr/bin/env bash
# Six-rung depth extension at 1.31B. ONE cell at a time on ONE GPU: the
# checkpoints are ~30 GB and load to CUDA, and the box is running the seed
# thickening on the other slots -- never disturb training.
set -u
cd /home/nvidia/ncr_scaleaxis
GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export NCR_SCALE=1310m
PY=/home/nvidia/tdenv/bin/python3
MANIFEST="${DEPTHEXT_MANIFEST:-/home/nvidia/ncr_scaleaxis/manifest_1b.tsv}"
OUT=/home/nvidia/ncr_scaleaxis/results_1b_depthext
mkdir -p "$OUT"
ok=0; bad=0
while IFS=$'\t' read -r k tag ck cfg; do
  [ -z "${k:-}" ] && continue
  [ -f "$OUT/${tag}_depthext.json" ] && { echo "skip (already scored) $tag"; continue; }
  NCR_K="$k" "$PY" depthext6_1310m_driver.py --k "$k" --ckpt "$ck" --tag "$tag" \
      --cellcfg "$cfg" --outdir "$OUT"
  rc=$?
  if [ "$rc" -eq 0 ]; then ok=$((ok+1)); else echo "!!! FAILED $tag (exit $rc)"; bad=$((bad+1)); fi
done < "$MANIFEST"
echo "DEPTHEXT1B DONE ok=$ok bad=$bad"
