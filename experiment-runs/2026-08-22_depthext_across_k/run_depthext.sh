#!/usr/bin/env bash
# Sharded driver for the depth-extension-across-K eval (EXPERIMENT_LOG 2026-08-22 #2).
#   Usage: run_depthext.sh <shard_index> <n_shards> <gpu_id>
# NCR_K is exported PER CELL from the manifest's K column -- kscaling_config
# resolves K from that env var, and depthext_eval.py asserts --k against it and
# against the checkpoint's own recorded d_ncr, so a K mislabel cannot survive.
set -u
cd /home/nvidia/ncr_kscaling

SHARD="$1"; NSHARD="$2"; GPU="$3"
export CUDA_VISIBLE_DEVICES="$GPU"
PY=/home/nvidia/tdenv/bin/python3
MANIFEST="${DEPTHEXT_MANIFEST:-/home/nvidia/ncr_kscaling/depthext_manifest.tsv}"
OUTDIR=/home/nvidia/ncr_kscaling/results

i=0; ok=0; bad=0
while IFS=$'\t' read -r k tag ck cfg anchortag; do
  [ -z "${k:-}" ] && continue
  if [ $(( i % NSHARD )) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  ARGS=(--k "$k" --ckpt "$ck" --tag "$tag" --outdir "$OUTDIR")
  [ "$cfg" != "-" ] && ARGS+=(--cellcfg "$cfg")
  [ "$anchortag" != "-" ] && ARGS+=(--anchor-runner-tag "$anchortag")
  NCR_K="$k" "$PY" depthext_eval.py "${ARGS[@]}"
  rc=$?
  if [ "$rc" -eq 0 ]; then ok=$((ok+1)); else echo "!!! FAILED $tag (exit $rc)"; bad=$((bad+1)); fi
done < "$MANIFEST"

echo "SHARD $SHARD DONE ok=$ok bad=$bad"
