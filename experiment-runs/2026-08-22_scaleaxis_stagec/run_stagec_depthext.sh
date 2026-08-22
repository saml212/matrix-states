#!/usr/bin/env bash
# Sharded driver for the 392M six-rung depth extension (design sec 4.6, Stage C).
#   Usage: run_stagec_depthext.sh <shard_index> <n_shards> <gpu_id>
# NCR_SCALE=392m is exported here (the battery/wrapper take env, not a flag);
# NCR_K comes per cell from the manifest's K column. depthext_eval asserts --k
# against NCR_K, against the checkpoint's d_ncr, AND (B5) against its backbone.
set -u
cd /home/nvidia/ncr_scaleaxis

SHARD="$1"; NSHARD="$2"; GPU="$3"
export CUDA_VISIBLE_DEVICES="$GPU"
export NCR_SCALE=392m
PY=/home/nvidia/tdenv/bin/python3
MANIFEST="${STAGEC_MANIFEST:-/home/nvidia/ncr_scaleaxis/stagec_manifest.tsv}"
OUTDIR=/home/nvidia/ncr_scaleaxis/results

i=0; ok=0; bad=0
while IFS=$'\t' read -r k tag ck cfg; do
  [ -z "${k:-}" ] && continue
  if [ $(( i % NSHARD )) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  ARGS=(--k "$k" --ckpt "$ck" --tag "$tag" --outdir "$OUTDIR")
  [ "$cfg" != "-" ] && ARGS+=(--cellcfg "$cfg")
  NCR_K="$k" "$PY" depthext6_392m_driver.py "${ARGS[@]}"
  rc=$?
  if [ "$rc" -eq 0 ]; then ok=$((ok+1)); else echo "!!! FAILED $tag (exit $rc)"; bad=$((bad+1)); fi
done < "$MANIFEST"
echo "SHARD $SHARD DONE ok=$ok bad=$bad"
