#!/usr/bin/env bash
# Opportunistic, VRAM-gated 1.31B scorer.
#
# The 1B checkpoints are ~29 GiB and load onto the GPU; a training cell holds
# ~70 GiB of an 81.5 GiB part, so there is NO room while a slot is occupied.
# The first wave died on exactly that (CUDA OOM, "Process ... has 56.37 GiB in
# use") when a thickening worker claimed the slot mid-run.
#
# This runner therefore NEVER competes: before each cell it polls for a GPU
# with at least MIN_FREE_MIB genuinely free and only then starts. It is
# resume-safe (skips cells whose output already exists), so it can be killed
# and relaunched without losing work, and it processes exactly one cell at a
# time. Two payloads, battery gaps first (they gate Curve 1 / T_W@h_top).
set -u
cd /home/nvidia/ncr_scaleaxis
export NCR_SCALE=1310m
PY=/home/nvidia/tdenv/bin/python3
MIN_FREE_MIB="${MIN_FREE_MIB:-44000}"
OUT=/home/nvidia/ncr_scaleaxis/results_1b_depthext
BATT=/home/nvidia/ncr_scaleaxis/results
mkdir -p "$OUT"

pick_gpu() {   # echoes a gpu index with enough free VRAM, or nothing
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
  | awk -v need="$MIN_FREE_MIB" -F', *' '{ if (($3-$2) >= need) { print $1; exit } }'
}

wait_for_gpu() {
  local g=""
  while :; do
    g=$(pick_gpu)
    [ -n "$g" ] && { echo "$g"; return 0; }
    sleep 10
  done
}

echo "=== battery gap-fill ==="
for cell in scaleaxis1310m_K16_primary_s0 scaleaxis1310m_K16_primary_s1; do
  O="$BATT/sweep1b_${cell}_kscaling.json"
  [ -f "$O" ] && { echo "already scored: $cell"; continue; }
  CK="/ephemeral/scaleaxis1b/ckpts/${cell}/${cell}.ckpt.pt"
  RES="/ephemeral/scaleaxis1b/results/${cell}.json"
  [ -f "$CK" ] || { echo "!!! MISSING-CKPT $cell"; continue; }
  G=$(wait_for_gpu); echo "--- battery $cell on GPU $G ($(date -u +%H:%M:%S)) ---"
  CUDA_VISIBLE_DEVICES="$G" NCR_K=16 "$PY" kscaling_battery.py --k 16 \
      --ckpt "$CK" --cellcfg "$RES" --tag "sweep1b_${cell}" 2>&1 | grep -vi "hf hub"
done

echo "=== depth-ext wave (24 cells, six rungs) ==="
ok=0; bad=0
while IFS=$'\t' read -r k tag ck cfg; do
  [ -z "${k:-}" ] && continue
  [ -f "$OUT/${tag}_depthext.json" ] && { echo "skip (scored): $tag"; continue; }
  G=$(wait_for_gpu); echo "--- $tag on GPU $G ($(date -u +%H:%M:%S)) ---"
  CUDA_VISIBLE_DEVICES="$G" NCR_K="$k" "$PY" depthext6_1310m_driver.py \
      --k "$k" --ckpt "$ck" --tag "$tag" --cellcfg "$cfg" --outdir "$OUT" 2>&1 | grep -vi "hf hub"
  if [ -f "$OUT/${tag}_depthext.json" ]; then ok=$((ok+1)); else echo "!!! NO OUTPUT $tag"; bad=$((bad+1)); fi
done < /home/nvidia/ncr_scaleaxis/manifest_1b.tsv

echo "DEPTHEXT1B ok=$ok bad=$bad  outputs=$(ls $OUT/*.json 2>/dev/null | wc -l)/24"
echo OPPORTUNISTIC_DONE
