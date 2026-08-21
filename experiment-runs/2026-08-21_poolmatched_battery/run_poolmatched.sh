#!/usr/bin/env bash
# Sharded driver for the pool-matched premise battery.
#   NGPU shards, one tmux-free worker per GPU (the caller runs this inside tmux).
#   Usage: run_poolmatched.sh <shard_index> <n_shards> <gpu_id>
# Every cell writes results/<cell_id>_poolmatched.json. Nothing of record is
# overwritten. Per-cell nonzero exits are reported and counted, never swallowed.
set -u
cd /home/nvidia/ncr_writecond

SHARD="$1"; NSHARD="$2"; GPU="$3"
export CUDA_VISIBLE_DEVICES="$GPU"
PY=/home/nvidia/tdenv/bin/python3
MANIFEST=/home/nvidia/ncr_writecond/poolmatched_manifest.tsv
OUTDIR=/home/nvidia/ncr_writecond/results

i=0; ok=0; bad=0
while IFS=$'\t' read -r tag ck cfg; do
  [ -z "${tag:-}" ] && continue
  if [ $(( i % NSHARD )) -ne "$SHARD" ]; then i=$((i+1)); continue; fi
  i=$((i+1))
  "$PY" poolmatched_battery.py --ckpt "$ck" --tag "$tag" --cellcfg "$cfg" --outdir "$OUTDIR"
  rc=$?
  if [ "$rc" -eq 0 ]; then ok=$((ok+1)); else echo "!!! FAILED $tag (exit $rc)"; bad=$((bad+1)); fi
done < "$MANIFEST"

echo "SHARD $SHARD DONE ok=$ok bad=$bad"
