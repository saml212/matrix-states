#!/usr/bin/env bash
# ONE-SHOT DEFERRED LAUNCHER (PI instruction 2026-08-06; record in
# EXPERIMENT_LOG.md 2026-08-06 entry). The box is handed to another
# workstream (vLLM RL sampling). If the box then sits COMPLETELY GPU-idle
# (zero compute apps on ALL GPUs) for 3 continuous hours, promote the
# audit-released Jacobian training batch 1200/1202/1203 into the standing
# queue; once all three reach completed/, promote 1204 (the audit's
# held-spec dependency convention, EXPERIMENT_LOG 2026-07-30 sequencing
# ruling). Self-terminating; leaves idle_launcher.DONE when finished.
#
# Controls:
#   ~/queue/STOP               -> global queue shutdown: stand down (no DONE,
#                                 resumes if STOP is removed)
#   ~/queue/idle_launcher.HOLD -> freeze + reset the idle counter while
#                                 present (reserve the box indefinitely
#                                 without killing anything)
#
# Supervised by tmux session `idle_launcher`:
#   while [ ! -f STOP ] && [ ! -f idle_launcher.DONE ]; do bash <this>; sleep 60; done
# A crash/restart resets the idle counter — conservative: only ever DELAYS
# launch, never fires early. nvidia-smi failure also counts as NOT idle
# (never launch on blind data). If trainings were already promoted before a
# restart, the idle phase is skipped so the 1204 dependency watch resumes
# directly.
set -u
QROOT="$HOME/queue"
SRC="$HOME/jacobian_erank_exp"
LOG="$QROOT/idle_launcher.log"
DONE="$QROOT/idle_launcher.DONE"
HOLD="$QROOT/idle_launcher.HOLD"
TRAIN_JOBS="1200_jacobian_train_flatten 1202_jacobian_train_svd_aug 1203_jacobian_train_quadratic"
MEASURE_JOB="1204_jacobian_erank_measure"
NEED=37   # 37 consecutive 5-min all-idle samples => >=3h0m confirmed idle span (36 fires as early as 2h55m)
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

[ -f "$DONE" ] && exit 0

# Arming sanity: every job must be somewhere legitimate in its lifecycle.
for j in $TRAIN_JOBS $MEASURE_JOB; do
  if [ ! -f "$SRC/$j.json.CANDIDATE" ] && [ ! -f "$QROOT/pending/$j.json" ] \
     && [ ! -f "$QROOT/completed/$j.json" ] && [ ! -f "$QROOT/failed/$j.json" ] \
     && ! ls "$QROOT/claimed/$j".g*.json >/dev/null 2>&1; then
    log "SPEC MISSING for $j (not candidate/pending/claimed/completed/failed) -- refusing to arm"
    touch "$DONE"; exit 1
  fi
done

promoted=1
for j in $TRAIN_JOBS; do [ -f "$SRC/$j.json.CANDIDATE" ] && promoted=0; done

if [ "$promoted" -eq 0 ]; then
  log "armed: need $NEED consecutive 5-min zero-compute-app samples (3h); HOLD honored at $HOLD"
  idle=0
  while :; do
    [ -f "$QROOT/STOP" ] && { log "STOP present -- standing down"; exit 0; }
    if [ -f "$HOLD" ]; then
      [ "$idle" -gt 0 ] && log "HOLD present -- idle counter reset from $idle"
      idle=0; sleep 300; continue
    fi
    if out=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
      napps=$(printf '%s' "$out" | grep -c '[0-9]')
    else
      napps=999   # nvidia-smi failed: treat as busy, never launch blind
    fi
    if [ "$napps" -eq 0 ]; then
      idle=$((idle+1))
    else
      [ "$idle" -gt 0 ] && log "activity detected ($napps compute apps) -- idle counter reset from $idle"
      idle=0
    fi
    [ "$idle" -ge "$NEED" ] && break
    sleep 300
  done
  log "3h continuous idle confirmed -- promoting training batch"
  for j in $TRAIN_JOBS; do
    [ -f "$SRC/$j.json.CANDIDATE" ] && mv "$SRC/$j.json.CANDIDATE" "$QROOT/pending/$j.json" && log "promoted $j"
  done
else
  log "trainings already promoted -- skipping idle phase, resuming 1204 dependency watch"
fi

# Phase 2: promote 1204 only when ALL three trainings validity-passed.
while :; do
  [ -f "$QROOT/STOP" ] && { log "STOP present -- standing down"; exit 0; }
  n=0
  for j in $TRAIN_JOBS; do
    if [ -f "$QROOT/failed/$j.json" ]; then
      log "$j FAILED validity -- 1204 needs all four checkpoints; NOT promoting. Launcher done."
      touch "$DONE"; exit 0
    fi
    [ -f "$QROOT/completed/$j.json" ] && n=$((n+1))
  done
  [ "$n" -eq 3 ] && break
  sleep 300
done
if [ -f "$SRC/$MEASURE_JOB.json.CANDIDATE" ]; then
  mv "$SRC/$MEASURE_JOB.json.CANDIDATE" "$QROOT/pending/$MEASURE_JOB.json" && log "promoted $MEASURE_JOB (all trainings complete)"
fi
touch "$DONE"
log "launcher done -- self-terminated"
