#!/bin/bash
# stage2_calib2_supervisor.sh -- CAPABILITY_SEPARATION_DESIGN.md S2.27 chain
# resume: relaunch of the S2.26-authorized 11-cell calibration gate after the
# S2.27 instrument device-boundary fix (stage2_instrument.py anchor_raw
# .to(real_raw.device)) was independently audited and deployed.
# CLAUDE.md self-healing pattern (tmux session: stage2_calib2, GPU 0).
# Differences from stage2_calib_supervisor.sh (v1, S2.26):
#   - clears the v1 supervisor's CALIB_WAVE_FAILED sentinel at start (it is
#     written only by the supervisor, read by nothing in stage2_run.py --
#     recorded in S2.27);
#   - fresh log files (stage2_calib2_*.log) so the v1 failure record stays
#     intact for the archive.
# Resume-safe on both legs:
#   - Arm-1 retrain leg skip-guarded on all 5 arm1__*__seed0.pt existing
#     (they exist -- the leg is expected to skip);
#   - calibration wave resume-safe by construction
#     (is_valid_output(strict_real=True); no valid outputs exist yet, so all
#     11 cells re-run, incl. retraining the partial S3 arm2 cell -- the
#     code's own protocol, not overridden here).
# PI signoff: CAPABILITY_SEP_STAGE2_PI_SIGNOFF=1 citing S2.24 (CLEARED-FOR-
# DEPLOY) + S2.26 (fla cross-check 3/3, box smoke 6/6) + S2.27 (device fix
# audited: CPU bit-identity + box CUDA teeth proof).
cd /home/nvidia/chapter2/capability_separation || exit 1
export CAPABILITY_SEP_STAGE2_PI_SIGNOFF=1
export CUDA_VISIBLE_DEVICES=0
PY=/home/nvidia/tdenv/bin/python
RD=stage2_results
FAILS=0
rm -f STOP_stage2_calib2
rm -f $RD/CALIB_WAVE_FAILED
while [ ! -f STOP_stage2_calib2 ]; do
  n_arm1=$(ls $RD/arm1__*__seed0.pt 2>/dev/null | wc -l)
  if [ "$n_arm1" -lt 5 ]; then
    echo "[supervisor $(date -u +%FT%TZ)] Arm-1 retrain leg starting ($n_arm1/5 ckpts present)" >> stage2_calib2_supervisor.log
    $PY stage2_run.py --retrain-arm1 --results-dir $RD >> stage2_calib2_arm1.log 2>&1
  fi
  n_arm1=$(ls $RD/arm1__*__seed0.pt 2>/dev/null | wc -l)
  if [ "$n_arm1" -eq 5 ]; then
    echo "[supervisor $(date -u +%FT%TZ)] calibration wave leg starting" >> stage2_calib2_supervisor.log
    if $PY stage2_run.py --calibration-only --results-dir $RD >> stage2_calib2_wave.log 2>&1; then
      echo "[supervisor $(date -u +%FT%TZ)] calibration wave COMPLETED cleanly" >> stage2_calib2_supervisor.log
      touch $RD/CALIB_WAVE_DONE
      break
    fi
  fi
  FAILS=$((FAILS + 1))
  echo "[supervisor $(date -u +%FT%TZ)] leg exited nonzero (consecutive fails: $FAILS); retrying in 15s" >> stage2_calib2_supervisor.log
  if [ "$FAILS" -ge 20 ]; then
    # deterministic aborts (e.g. a budget-guard trip) would hot-loop forever;
    # bail loudly instead so the failure is diagnosed, not retried blindly.
    echo "[supervisor $(date -u +%FT%TZ)] 20 consecutive failures -- STOPPING for diagnosis" >> stage2_calib2_supervisor.log
    touch $RD/CALIB_WAVE_FAILED
    break
  fi
  sleep 15
done
