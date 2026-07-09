#!/bin/bash
# fixscale_slot0_waiter.sh -- operational convenience wrapper (NOT part of the audited bd40ebb
# codebase; touches no committed file). Purpose: the armoff slot0 partition (arm_off seed=0, both
# corpora) for a given scale is EXCLUSIVELY satisfied by the gate-tier's own already-running
# calibration cells via fixscale_wave.py's out_path() reuse convention -- launching slot0 while
# the gate-tier's OWN seed=0 cells are still training risks a genuine duplicate-compute race
# (out_path only defers to the gate-tier JSON once it reports complete=true; a slot0 session
# started too early would see 'absent' and launch a REDUNDANT independent training run on a
# DIFFERENT free GPU than the one gate-tier is using -- the GPU-occupancy guard alone cannot catch
# this because gate-tier's two cells for one scale run on TWO DIFFERENT GPUs, so pinning slot0 to
# just one of them only protects one of its two cells).
#
# This script removes that race by WAITING for both of the gate-tier's own result JSONs (for the
# given scale) to report complete=true (polled every 60s, matching the wave's own poll cadence)
# before invoking armoff-loop for slot0 at all. Functionally, slot0 is NOT required for the wave's
# own pin barrier to succeed (await_armoff_and_pin polls cell_state() directly, which resolves via
# out_path()'s gate-tier-reuse transparently regardless of whether slot0 was ever run) -- this is
# run purely so the wave's own bookkeeping (manifest --with-state, per-cell logs) reflects an
# explicit resume-skip for slot0's two cells, matching launch_scale's normal 3-slot convention.
#
# Usage: fixscale_slot0_waiter.sh <98m|392m> <gpu> <gpu_offset>
set -u
SCALE="$1"; GPU="$2"; GPU_OFFSET="$3"
cd /home/nvidia/chapter2/deltanet_rd
RESULT_DIR=results/fixscale/calib
if [ "$SCALE" = "98m" ]; then
  J1="$RESULT_DIR/fixscale_calib_off_98m_openr1-mix-ext_s0.json"
  J2="$RESULT_DIR/fixscale_calib_off_98m_wikitext-mix-ext_s0.json"
elif [ "$SCALE" = "392m" ]; then
  J1="$RESULT_DIR/fixscale_calib_off_392m_openr1-mix-ext_s0.json"
  J2="$RESULT_DIR/fixscale_calib_off_392m_wikitext-mix-ext_s0.json"
else
  echo "bad scale: $SCALE" >&2; exit 1
fi

is_complete() {
  [ -f "$1" ] || return 1
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('complete') is True else 1)" "$1" 2>/dev/null
}

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] slot0-waiter($SCALE): $*"; }

log "waiting for gate-tier's own $J1 and $J2 to report complete=true"
while true; do
  [ -f STOP_fixscale_wave ] && { log "STOP file present -- exiting without launching slot0"; exit 3; }
  if is_complete "$J1" && is_complete "$J2"; then
    log "both gate-tier seed=0 cells complete -- safe to launch slot0 (resume-skip expected, zero new training)"
    break
  fi
  sleep 60
done

log "invoking armoff-loop $SCALE 3 $GPU_OFFSET 0 (target gpu=$GPU)"
exec ./fixscale_supervisor.sh armoff-loop "$SCALE" 3 "$GPU_OFFSET" 0
