#!/usr/bin/env bash
# NCR ortho-fallback STAGE-0 v3 RELAUNCH driver -- ONE cell
# (stage0_damped_K24_s0), NCR_ORTHO_FALLBACK_DESIGN.md §2 as amended by
# §B6 (composed guard, v3 md5 c7c2f7c36e0ab33c1efaeed04fbcf1bb), launched
# under §B6's PRE-AUTHORIZED AUTO-CLEAR w/ §B3(2) re-priced params:
# --ceiling-gpuh 1.75 (contended-regime pricing), external timeout 7200s.
#
# Same terminal-status-gated supervisor as the v1 driver (see that file's
# header for the rationale): retries ONLY on a genuine crash (no valid
# terminal-status JSON), stops on COMPLETED or ABORTED-BUDGET -- never
# re-spends the frozen ceiling after a graceful budget abort.
set -u
GPU="${1:?usage: run_stage0_v3.sh GPU_ID}"
cd /home/nvidia/ncr
OUT=/home/nvidia/ncr/results_ortho_fallback
LOG="$OUT/run_stage0.log"
PY=/home/nvidia/tdenv/bin/python3
SCRIPT=ncr_ortho_fallback_stage0.py
CID=stage0_damped_K24_s0
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"
STOP_FILE="$OUT/STOP"
CEIL=1.75          # SSB3(2) re-priced internal ceiling (6300s), graceful abort
CELL_TO=7200       # SSB3(2)/SSB6-cond5 external hang backstop, fires only on a wedge

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

echo "=== stage0 v3 driver START g$GPU $(date -u +%FT%TZ) design=244aec0 script_md5=c7c2f7c36e0ab33c1efaeed04fbcf1bb ===" >> "$LOG"

while [ ! -f "$STOP_FILE" ]; do
  st=$(status_of)
  if [ "$st" = "COMPLETED" ] || [ "$st" = "ABORTED-BUDGET" ]; then
    echo "=== stage0 v3 [$CID] TERMINAL status=$st -- driver exiting $(date -u +%FT%TZ) ===" >> "$LOG"
    break
  fi
  echo "--- stage0 v3 [$CID] attempt START $(date -u +%FT%TZ) g$GPU (prior status='$st') ---" >> "$LOG"
  timeout "$CELL_TO" "$PY" "$SCRIPT" --primary-cell --arm damped_polar --K 24 --seed 0 \
    --steps 42000 --eps-rel 1e-3 --ns-iter 40 --ns-power 12 --anneal-frac 0.5 \
    --ceiling-gpuh "$CEIL" --outdir "$OUT" >> "$LOG" 2>&1
  rc=$?
  echo "--- stage0 v3 [$CID] attempt END rc=$rc $(date -u +%FT%TZ) g$GPU ---" >> "$LOG"
  sleep 15
done

echo "=== stage0 v3 driver DONE g$GPU $(date -u +%FT%TZ) ===" >> "$LOG"
