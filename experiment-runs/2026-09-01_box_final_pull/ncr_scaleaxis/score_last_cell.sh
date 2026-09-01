#!/usr/bin/env bash
# Score the ONE remaining battery cell (K16_primary_s0) and re-run the finaliser.
#
# ROOT-CAUSE FIX. The previous reservation used nvidia-smi alone to decide a GPU
# was free. A queue worker CLAIMS a GPU (writing ~/queue/claimed/<id>.g<N>.json)
# and only allocates VRAM seconds-to-minutes later, so during that window
# nvidia-smi reports the GPU idle and my runner reserved a slot that was already
# spoken for. Both then ran on GPU 0 and contended -- that is what OOM'd the
# thickening cells AND this very cell.
#
# The claim directory is the authoritative statement of intent and it encodes
# the GPU index in the filename, so it is consulted BEFORE nvidia-smi here.
# After reserving we re-check that no claim appeared during our own startup and
# back off if one did, closing the race in both directions.
set -u
cd /home/nvidia/ncr_scaleaxis
PY=/home/nvidia/tdenv/bin/python3
export NCR_SCALE=1310m
CELL=scaleaxis1310m_K16_primary_s0
BATT=/home/nvidia/ncr_scaleaxis/results
STOP=/home/nvidia/ncr_scaleaxis/RELEASE_RESERVATION2
RESLOG=/home/nvidia/ncr_scaleaxis/reserve2.log
rm -f "$STOP" "$RESLOG"

gpu_claimed() { ls /home/nvidia/queue/claimed/ 2>/dev/null | grep -q "\.g$1\.json$"; }

safe_gpu() {
  for g in 0 1 2 3 4 5 6 7; do
    gpu_claimed "$g" && continue                       # claimed -- worker owns it
    napps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$g" 2>/dev/null | grep -c '[0-9]')
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    [ "${napps:-1}" -eq 0 ] && [ "${mem:-9999}" -lt 2048 ] && { echo "$g"; return 0; }
  done
  return 1
}

if [ -f "$BATT/sweep1b_${CELL}_kscaling.json" ]; then echo "already scored"; else
GPU=""; hb=0
echo "$(date -u +%FT%TZ) waiting for a slot that is BOTH unclaimed and idle"
while [ -z "$GPU" ]; do
  G=$(safe_gpu) || { hb=$((hb+1)); [ $((hb % 120)) -eq 0 ] && echo "$(date -u +%FT%TZ) waiting; claimed=$(ls /home/nvidia/queue/claimed/ 2>/dev/null | wc -l)/8"; sleep 15; continue; }
  echo "$(date -u +%FT%TZ) GPU $G unclaimed+idle -- reserving"
  CUDA_VISIBLE_DEVICES="$G" nohup "$PY" reserve_gpu.py "$STOP" 3.0 >> "$RESLOG" 2>&1 &
  RPID=$!
  for _ in $(seq 1 30); do grep -q "RESERVED pid=" "$RESLOG" 2>/dev/null && break; sleep 1; done
  sleep 20                                              # settle, then re-check for a late claim
  if gpu_claimed "$G"; then
    echo "$(date -u +%FT%TZ) BACKING OFF: a worker claimed GPU $G during our startup"
    touch "$STOP"; wait "$RPID" 2>/dev/null; rm -f "$STOP" "$RESLOG"; sleep 10; continue
  fi
  kill -0 "$RPID" 2>/dev/null && GPU="$G" && echo "$(date -u +%FT%TZ) RESERVATION HELD on GPU $G (unclaimed, verified twice)"
done
export CUDA_VISIBLE_DEVICES="$GPU"
echo "=== scoring $CELL ==="
NCR_K=16 "$PY" kscaling_battery.py --k 16 \
    --ckpt "/ephemeral/scaleaxis1b/ckpts/${CELL}/${CELL}.ckpt.pt" \
    --cellcfg "/ephemeral/scaleaxis1b/results/${CELL}.json" \
    --tag "sweep1b_${CELL}" 2>&1 | grep -vi "hf hub"
echo "=== releasing GPU $GPU ==="; touch "$STOP"; sleep 8
fi

n=$(ls "$BATT"/sweep1b_*.json 2>/dev/null | wc -l)
echo "sweep1b coverage: $n/22"
[ "$n" -ne 22 ] && { echo "!!! STILL INCOMPLETE -- not finalising"; echo LASTCELL_FAILED; exit 1; }

echo "=== re-running the finaliser for the COMPLETE table set ==="
STAGE=/home/nvidia/ncr_scaleaxis/harvest1b
cp -f /home/nvidia/ncr_scaleaxis/results_1b_depthext/depthext6_1310m_*_depthext.json "$STAGE"/ 2>/dev/null
cp -f "$BATT"/sweep1b_*_kscaling.json "$BATT"/calib1b_*_kscaling.json "$STAGE"/ 2>/dev/null
echo "  1310M depth  : $(ls $STAGE/depthext6_1310m_*_depthext.json | wc -l)/24"
echo "  1310M battery: $(ls $STAGE/sweep1b_*.json $STAGE/calib1b_*.json | wc -l)/24"
"$PY" aggregate_1b.py "$STAGE" > /home/nvidia/ncr_scaleaxis/FINAL_TABLES.txt 2>&1
echo "aggregator exit=$?  lines=$(wc -l < /home/nvidia/ncr_scaleaxis/FINAL_TABLES.txt)"
echo LASTCELL_DONE
