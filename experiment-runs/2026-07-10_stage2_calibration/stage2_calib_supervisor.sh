#!/bin/bash
# stage2_calib_supervisor.sh -- CAPABILITY_SEPARATION_DESIGN.md S2.26 chain
# resume: Arm-1 retrain (0.2148 GPU-h priced) then the 11-cell calibration
# gate (production run_calibration_wave_real, 7 depths gated per cell).
# CLAUDE.md self-healing pattern (tmux session: stage2_calib, GPU 0).
# Resume-safe on both legs:
#   - the Arm-1 retrain leg is skip-guarded on all 5 arm1__*__seed0.pt
#     checkpoints existing (the utility itself is deterministic but not
#     skip-aware, so the guard lives here);
#   - the calibration wave is resume-safe by construction
#     (is_valid_output(strict_real=True); tiny/corrupted outputs re-run).
# PI signoff: CAPABILITY_SEP_STAGE2_PI_SIGNOFF=1 citing S2.24 (CLEARED-FOR-
# DEPLOY) + S2.26 (fla cross-check gate PASSED 3/3, box smoke 6/6).
cd /home/nvidia/chapter2/capability_separation || exit 1
export CAPABILITY_SEP_STAGE2_PI_SIGNOFF=1
export CUDA_VISIBLE_DEVICES=0
PY=/home/nvidia/tdenv/bin/python
RD=stage2_results
FAILS=0
rm -f STOP_stage2_calib
while [ ! -f STOP_stage2_calib ]; do
  n_arm1=$(ls $RD/arm1__*__seed0.pt 2>/dev/null | wc -l)
  if [ "$n_arm1" -lt 5 ]; then
    echo "[supervisor $(date -u +%FT%TZ)] Arm-1 retrain leg starting ($n_arm1/5 ckpts present)" >> stage2_calib_supervisor.log
    $PY stage2_run.py --retrain-arm1 --results-dir $RD >> stage2_calib_arm1.log 2>&1
  fi
  n_arm1=$(ls $RD/arm1__*__seed0.pt 2>/dev/null | wc -l)
  if [ "$n_arm1" -eq 5 ]; then
    echo "[supervisor $(date -u +%FT%TZ)] calibration wave leg starting" >> stage2_calib_supervisor.log
    if $PY stage2_run.py --calibration-only --results-dir $RD >> stage2_calib_wave.log 2>&1; then
      echo "[supervisor $(date -u +%FT%TZ)] calibration wave COMPLETED cleanly" >> stage2_calib_supervisor.log
      touch $RD/CALIB_WAVE_DONE
      break
    fi
  fi
  FAILS=$((FAILS + 1))
  echo "[supervisor $(date -u +%FT%TZ)] leg exited nonzero (consecutive fails: $FAILS); retrying in 15s" >> stage2_calib_supervisor.log
  if [ "$FAILS" -ge 20 ]; then
    # deterministic aborts (e.g. a budget-guard trip) would hot-loop forever;
    # bail loudly instead so the failure is diagnosed, not retried blindly.
    echo "[supervisor $(date -u +%FT%TZ)] 20 consecutive failures -- STOPPING for diagnosis" >> stage2_calib_supervisor.log
    touch $RD/CALIB_WAVE_FAILED
    break
  fi
  sleep 15
done
