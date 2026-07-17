#!/usr/bin/env bash
# NCR REAL-LM WAVE-1 CALIBRATION driver -- sec G3-B6, ONE cell (the two-arm
# full_graft/backbone_only calibration, NCR_REAL_LM_DESIGN.md sec G3-B5's
# ratified design). Same terminal-status-gated self-healing supervisor
# pattern as experiment-runs/2026-07-17_ncr_ortho_fallback_stage0/
# orchestration/run_stage0_v3.sh: retries ONLY on a genuine crash (no valid
# terminal-status JSON), stops on COMPLETED or ABORTED-BUDGET -- never
# re-spends the frozen ceiling after a graceful budget abort. Resume is
# handled INSIDE ncr_lm_wave1_runner.py itself (true step-level checkpoint/
# resume, sec G3-B6) -- a supervisor restart re-invokes the SAME command,
# and the runner picks up from its own last checkpoint, not from step 0.
#
# NOT LAUNCHED BY THIS BUILD (coordinator gates GPU spend, per the build
# brief). --ceiling-gpuh below MUST be set from a --mode phase0-timing
# measurement (contended-priced) run FIRST, on the SAME box/operating
# point -- see the sec G3-B6 record for the measured value at build time
# (4.865 GPU-h for steps=20000 at batch=32); RE-MEASURE if steps/batch/box
# load has changed materially since then, do not reuse the number blind.
set -u
GPU="${1:?usage: run_wave1_calibration.sh GPU_ID CEILING_GPUH [STEPS] [SEED]}"
CEIL="${2:?usage: run_wave1_calibration.sh GPU_ID CEILING_GPUH [STEPS] [SEED]}"
STEPS="${3:-20000}"
SEED="${4:-0}"

cd /home/nvidia
OUT=/home/nvidia/results_gate3_wave1
LOG="$OUT/run_wave1_calibration.log"
PY=/home/nvidia/tdenv/bin/python3
SCRIPT=ncr_lm_wave1_runner.py
CID="wave1_calib_K24_s${SEED}"
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"
STOP_FILE="$OUT/STOP"
# External hang backstop: ~15% above the internal --ceiling-gpuh (matches
# stage0_v3's own internal-vs-external ratio) -- fires ONLY on a wedge the
# internal eval-interval polling loop can't reach (e.g. a hung CUDA kernel).
CELL_TO=$(python3 -c "print(int(float(\"$CEIL\") * 3600 * 1.15))")

status_of() {
  "$PY" -c "
import json
try:
    d = json.load(open('$OUT/${CID}.json'))
    print(d.get('status', ''))
except Exception:
    print('')
" 2>/dev/null
}

echo "=== wave1 calibration driver START g$GPU $(date -u +%FT%TZ) ceiling=${CEIL}GPUh steps=$STEPS seed=$SEED cell_to=${CELL_TO}s ===" >> "$LOG"

while [ ! -f "$STOP_FILE" ]; do
  st=$(status_of)
  if [ "$st" = "COMPLETED" ] || [ "$st" = "ABORTED-BUDGET" ]; then
    echo "=== wave1 calibration [$CID] TERMINAL status=$st -- driver exiting $(date -u +%FT%TZ) ===" >> "$LOG"
    break
  fi
  echo "--- wave1 calibration [$CID] attempt START $(date -u +%FT%TZ) g$GPU (prior status='$st') ---" >> "$LOG"
  timeout "$CELL_TO" "$PY" "$SCRIPT" --mode calibration --device cuda \
    --steps "$STEPS" --seed "$SEED" --ceiling-gpuh "$CEIL" \
    --batch-size 32 --eval-batch-size 64 --warmup-steps 200 \
    --ckpt-every 500 --eval-every 500 \
    --cell-id "$CID" --ckpt-dir "$OUT" --stop-file "$STOP_FILE" \
    --out "$OUT/${CID}.json" >> "$LOG" 2>&1
  rc=$?
  echo "--- wave1 calibration [$CID] attempt END rc=$rc $(date -u +%FT%TZ) g$GPU ---" >> "$LOG"
  sleep 15
done

echo "=== wave1 calibration driver DONE g$GPU $(date -u +%FT%TZ) ===" >> "$LOG"

# Launch convention (matches run_stage0_v3.sh's own precedent):
#   tmux new-session -d -s wave1_calib \
#     "bash orchestration/run_wave1_calibration.sh 5 4.865 20000 0"
