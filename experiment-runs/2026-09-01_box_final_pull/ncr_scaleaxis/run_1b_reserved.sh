#!/usr/bin/env bash
# RESERVED-GPU 1.31B harvest. Coordinator election (b), 2026-08-25.
#
# Grabs the first GPU that goes genuinely free (by the queue worker's OWN B6
# predicate: zero compute-apps AND < 2048 MiB used), immediately pins it with a
# ~3 GiB resident allocation so the worker's next 60 s poll skips it, then runs
# all remaining harvest payloads one at a time on that GPU and RELEASES it.
#
# Never pauses, kills or preempts a training cell: it only takes a slot that is
# already empty, and it takes it in the way the worker is built to respect.
set -u
cd /home/nvidia/ncr_scaleaxis
PY=/home/nvidia/tdenv/bin/python3
export NCR_SCALE=1310m
OUT=/home/nvidia/ncr_scaleaxis/results_1b_depthext
BATT=/home/nvidia/ncr_scaleaxis/results
STOP=/home/nvidia/ncr_scaleaxis/RELEASE_RESERVATION
RESLOG=/home/nvidia/ncr_scaleaxis/reserve.log
mkdir -p "$OUT"; rm -f "$STOP"

free_gpu() {   # worker's own predicate, inverted: echo the first genuinely-free index
  for g in 0 1 2 3 4 5 6 7; do
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null | grep -c '[0-9]')
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ "${napps:-1}" -eq 0 ] && [ "${mem:-9999}" -lt 2048 ] && { echo "$g"; return 0; }
  done
  return 1
}

losses=0
GPU=""
hb=0
echo "$(date -u +%FT%TZ) waiting for a genuinely-free GPU (worker predicate: 0 compute-apps AND <2048 MiB)."
echo "  NOTE: all 8 slots hold 1.31B thickening cells claimed 00:40Z at ~15.2 GPU-h each"
echo "  (observed 2000/20000 steps at +85 min), so the first release is ~13 h out."
while [ -z "$GPU" ]; do
  G=$(free_gpu) || {
    hb=$((hb+1))
    # heartbeat every ~30 min (180 x 10s) so a multi-hour wait is verifiable
    [ $((hb % 180)) -eq 0 ] && echo "$(date -u +%FT%TZ) still waiting; free VRAM: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 10; continue; }
  echo "$(date -u +%H:%M:%S) GPU $G is free -- claiming"
  CUDA_VISIBLE_DEVICES="$G" nohup "$PY" reserve_gpu.py "$STOP" 3.0 >> "$RESLOG" 2>&1 &
  RPID=$!
  for _ in $(seq 1 30); do
    grep -q "RESERVED pid=" "$RESLOG" 2>/dev/null && break
    sleep 1
  done
  napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$G" 2>/dev/null | grep -c '[0-9]')
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$G" 2>/dev/null | tr -d ' ')
  if kill -0 "$RPID" 2>/dev/null && [ "$napps" -ge 1 ] && [ "$mem" -ge 2048 ]; then
    GPU="$G"; echo "$(date -u +%H:%M:%S) RESERVATION HELD on GPU $G (napps=$napps mem=${mem}MiB)"
  else
    losses=$((losses+1)); kill "$RPID" 2>/dev/null
    echo "$(date -u +%H:%M:%S) !!! LOST THE RACE on GPU $G (loss $losses) -- napps=$napps mem=${mem}MiB"
    [ "$losses" -ge 2 ] && { echo "!!! LOST TWICE -- ESCALATE: ask for a pending spec to be held back"; exit 9; }
    sleep 5
  fi
done
export CUDA_VISIBLE_DEVICES="$GPU"

run() { echo "--- $1 ($(date -u +%H:%M:%S)) ---"; shift; "$@" 2>&1 | grep -vi "hf hub"; }

echo "=== payload 1/3: battery gaps (unblock Curve 1 / TEST-W) ==="
for cell in scaleaxis1310m_K16_primary_s0 scaleaxis1310m_K16_primary_s1; do
  [ -f "$BATT/sweep1b_${cell}_kscaling.json" ] && { echo "already scored: $cell"; continue; }
  run "battery $cell" env NCR_K=16 "$PY" kscaling_battery.py --k 16 \
      --ckpt "/ephemeral/scaleaxis1b/ckpts/${cell}/${cell}.ckpt.pt" \
      --cellcfg "/ephemeral/scaleaxis1b/results/${cell}.json" --tag "sweep1b_${cell}"
done

echo "=== payload 2/3: K=16 wall-excursion re-measure at base seed 31337 ==="
cell=scaleaxis1310m_K16_primary_s2
if [ ! -f "$BATT/remeasure1b_${cell}_kscaling.json" ]; then
  run "remeasure $cell" env NCR_K=16 "$PY" kscaling_battery.py --k 16 \
      --ckpt "/ephemeral/scaleaxis1b/ckpts/${cell}/${cell}.ckpt.pt" \
      --cellcfg "/ephemeral/scaleaxis1b/results/${cell}.json" \
      --base-seed 31337 --tag "remeasure1b_${cell}"
else echo "already re-measured"; fi

echo "=== payload 3/3: depth-ext, 24 cells, six rungs ==="
ok=0; bad=0
while IFS=$'\t' read -r k tag ck cfg; do
  [ -z "${k:-}" ] && continue
  [ -f "$OUT/${tag}_depthext.json" ] && { echo "skip (scored): $tag"; continue; }
  run "$tag" env NCR_K="$k" "$PY" depthext6_1310m_driver.py --k "$k" --ckpt "$ck" \
      --tag "$tag" --cellcfg "$cfg" --outdir "$OUT"
  if [ -f "$OUT/${tag}_depthext.json" ]; then ok=$((ok+1)); else echo "!!! NO OUTPUT $tag"; bad=$((bad+1)); fi
done < /home/nvidia/ncr_scaleaxis/manifest_1b.tsv

echo "DEPTHEXT1B ok=$ok bad=$bad outputs=$(ls $OUT/*.json 2>/dev/null | wc -l)/24"
echo "=== releasing reservation on GPU $GPU ==="
touch "$STOP"; sleep 8
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPU"
echo RESERVED_RUN_DONE
