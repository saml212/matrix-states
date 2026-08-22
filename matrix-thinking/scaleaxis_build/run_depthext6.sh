#!/usr/bin/env bash
# Sharded driver for the SIX-RUNG 98M depth-extension re-score
# (NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 4.6.1 step 2).
#   Usage: run_depthext6.sh <shard_index> <n_shards> <gpu_id>
#
# Identical shape to the four-rung wave's run_depthext.sh, with two changes:
#   (1) it calls depthext6_driver.py (which imports the audited depthext_eval.py
#       unmodified and sets its SQUARING_PROFILE to {5,7,9,11,13,15});
#   (2) tags are prefixed `depthext6_` so no archived four-rung JSON is
#       overwritten -- the archive is the reproduction cross-check.
# NCR_K is exported PER CELL from the manifest's K column.
set -u
cd /home/nvidia/ncr_kscaling

SHARD="$1"; NSHARD="$2"; GPU="$3"
export CUDA_VISIBLE_DEVICES="$GPU"
PY=/home/nvidia/tdenv/bin/python3
MANIFEST="${DEPTHEXT_MANIFEST:-/home/nvidia/ncr_kscaling/depthext48_manifest.tsv}"
OUTDIR=/home/nvidia/ncr_kscaling/results_depthext6

mkdir -p "$OUTDIR"
i=0; ok=0; bad=0
while IFS=$'\t' read -r k tag ck cfg anchortag; do
  [ -z "${k:-}" ] && continue
  if [ $(( i % NSHARD )) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  tag6="depthext6_${tag#depthext_}"
  ARGS=(--k "$k" --ckpt "$ck" --tag "$tag6" --outdir "$OUTDIR")
  [ "$cfg" != "-" ] && ARGS+=(--cellcfg "$cfg")
  [ "$anchortag" != "-" ] && ARGS+=(--anchor-runner-tag "$anchortag")
  NCR_K="$k" "$PY" depthext6_driver.py "${ARGS[@]}"
  rc=$?
  if [ "$rc" -eq 0 ]; then ok=$((ok+1)); else echo "!!! FAILED $tag6 (exit $rc)"; bad=$((bad+1)); fi
done < "$MANIFEST"

echo "SHARD $SHARD DONE ok=$ok bad=$bad"
