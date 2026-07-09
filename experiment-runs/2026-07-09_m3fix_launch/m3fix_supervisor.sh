#!/usr/bin/env bash
# Self-healing supervisor: Stage-1 M3 FIX WAVE (S1.33/S1.34/S1.35).
# GPU 0 ONLY -- never touches GPUs 3/4 (live 392m wave) or GPU 6 (98m resume).
#
# Authorization: matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md S1.35
# (commit b6f0641) -- "LAUNCH DISPATCHED per B1-B4" citing S1.32+S1.35;
# CAPABILITY_SEP_PI_SIGNOFF=1 below cites that record. C1 metric pin
# (crosscheck_recovered_frac_90 / crosscheck_mean_cos, full-Q Procrustes,
# not the scale-only primary) is discharged by S1.35, applied at harvest time.
#
# Pattern: CLAUDE.md's `while [ ! -f STOP ]; do <cmd>; sleep 15; done`
# self-healing loop, mirroring cap_sweep_supervisor.sh's house convention.
# run_capability_sep.py is resume-safe (validity-checked output, not just
# existence), so re-invoking after a crash costs nothing beyond the crashed
# cell's partial work. NO --steps override (would clobber the Rev-7
# per-group STEP_BUDGET pins the m3fix manifest prices against).
set -uo pipefail

CAP_DIR=/home/nvidia/chapter2/capability_separation
PY=/home/nvidia/tdenv/bin/python

cd "$CAP_DIR"
rm -f STOP_m3fix
mkdir -p results_m3fix
export CUDA_VISIBLE_DEVICES=0
export CAPABILITY_SEP_PI_SIGNOFF=1   # S1.32+S1.35 authorization (see header)

echo "[supervisor] === M3 FIX WAVE START $(date -u) ===" | tee -a m3fix_wave.log
while [ ! -f STOP_m3fix ]; do
  "$PY" run_capability_sep.py --m3fix --device cuda --results-dir results_m3fix/ 2>&1 | tee -a m3fix_wave.log
  RC=${PIPESTATUS[0]}
  if [ "$RC" -eq 0 ]; then
    echo "[supervisor] m3fix exited 0 -- WAVE DONE $(date -u)" | tee -a m3fix_wave.log
    touch STOP_m3fix
  else
    echo "[supervisor] m3fix exited $RC -- retrying in 15s ($(date -u))" | tee -a m3fix_wave.log
    sleep 15
  fi
done
touch "$CAP_DIR/results_m3fix/M3FIX_STAGE_DONE"
echo "[supervisor] === M3 FIX WAVE COMPLETE $(date -u) ===" | tee -a "$CAP_DIR/m3fix_wave.log"
