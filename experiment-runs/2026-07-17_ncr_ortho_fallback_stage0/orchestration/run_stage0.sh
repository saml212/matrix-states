#!/usr/bin/env bash
# NCR ortho-fallback STAGE-0 driver -- ONE cell (stage0_damped_K24_s0),
# NCR_ORTHO_FALLBACK_DESIGN.md §2, frozen commit 28e69cb.
#
# Self-healing supervisor (H100_SETUP.md pattern, scaled to 1 cell): retries
# on a genuine CRASH (process died without writing a valid terminal-status
# JSON -- SSH hiccup, OOM kill, segfault), but STOPS once the cell reaches a
# TERMINAL status (COMPLETED or ABORTED-BUDGET). This is deliberately NOT a
# naive infinite "always sleep-then-rerun" loop: ABORTED-BUDGET is a
# legitimate outcome under the frozen §2 ceiling (<=0.5 GPU-h this attempt,
# <=1.0 GPU-h Stage-0 total incl. ONE pre-authorized eps_rel retry) -- an
# infinite retry-on-any-exit loop would silently re-spend the ceiling budget
# every restart and violate that frozen rule. Validity is checked via the
# cell's own JSON `status` field (parses + terminal), not process exit code
# or file existence alone (H100_SETUP.md hard rule).
#
# This launches ONLY the eps_rel=1e-3 primary attempt. The §2 branch logic's
# pre-authorized retry (eps_rel=1e-2 same-signature FAIL / 1e-4
# different-signature FAIL) requires reading this attempt's FAIL signature
# first -- an ASSESS-time judgment call, not launched here.
set -u
GPU="${1:?usage: run_stage0.sh GPU_ID}"
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
# External per-attempt backstop ABOVE the 0.5h (1800s) internal ceiling --
# mirrors the parent run's own timeout-above-ceiling discipline (the internal
# ceiling should fire first with a graceful status write; this timeout only
# catches a hang/wedge). 2400s = 1800s + 600s (the 500-step graceful-check-lag
# margin, generously rounded).
CELL_TO=2400

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

echo "=== stage0 driver START g$GPU $(date -u +%FT%TZ) design=28e69cb ===" >> "$LOG"

while [ ! -f "$STOP_FILE" ]; do
  st=$(status_of)
  if [ "$st" = "COMPLETED" ] || [ "$st" = "ABORTED-BUDGET" ]; then
    echo "=== stage0 [$CID] TERMINAL status=$st -- driver exiting $(date -u +%FT%TZ) ===" >> "$LOG"
    break
  fi
  echo "--- stage0 [$CID] attempt START $(date -u +%FT%TZ) g$GPU (prior status='$st') ---" >> "$LOG"
  timeout "$CELL_TO" "$PY" "$SCRIPT" --primary-cell --arm damped_polar --K 24 --seed 0 \
    --steps 42000 --eps-rel 1e-3 --ns-iter 40 --ns-power 12 --anneal-frac 0.5 \
    --ceiling-gpuh 0.5 --outdir "$OUT" >> "$LOG" 2>&1
  rc=$?
  echo "--- stage0 [$CID] attempt END rc=$rc $(date -u +%FT%TZ) g$GPU ---" >> "$LOG"
  sleep 15
done

echo "=== stage0 driver DONE g$GPU $(date -u +%FT%TZ) ===" >> "$LOG"
