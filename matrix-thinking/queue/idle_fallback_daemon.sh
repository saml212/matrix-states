#!/usr/bin/env bash
# STANDING IDLE-FALLBACK DAEMON (PI 2026-08-06: "GPUs always have something
# to do; falling back on the scaling ladder is vitally important" — with RL
# sampling having ABSOLUTE priority). Chained behind the one-shot Jacobian
# launcher: inert until ~/queue/idle_launcher.DONE exists. Then, forever:
# if ALL GPUs show zero compute apps for 3+ continuous hours AND pending/
# and claimed/ are both empty, promote the next wave (up to WAVE specs,
# filename order) from ~/queue/fallback_pool/. ONLY audited, queue-eligible
# specs may ever be placed in the pool — the pool is the runway, the
# ceremony gate stays upstream of it. If the box is starving (3h idle, no
# queue work, pool DRY) raise the alarm ONCE per dry spell:
# ~/queue/FALLBACK_POOL_DRY + a log ALERT — the coordinator's signal to
# refill via the design/audit gauntlet.
#
# RL-sampling priority: any GPU compute app resets the countdown, so this
# can never contend with live sampling. ~/queue/idle_launcher.HOLD (shared
# with the one-shot) freezes the countdown to reserve the box indefinitely.
# STOP = stand down. nvidia-smi failure counts as BUSY (never launch blind).
# Supervised by tmux session `idle_fallback` + the idle-daemon cron watchdog.
set -u
QROOT="$HOME/queue"
POOL="$QROOT/fallback_pool"
LOG="$QROOT/idle_fallback.log"
HOLD="$QROOT/idle_launcher.HOLD"
DRYFLAG="$QROOT/FALLBACK_POOL_DRY"
NEED=37   # 37 consecutive 5-min all-idle samples => >=3h confirmed idle
WAVE=8
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

mkdir -p "$POOL"
log "fallback daemon up (NEED=$NEED, WAVE=$WAVE); pool has $(ls "$POOL" 2>/dev/null | wc -l | tr -d ' ') specs"

idle=0
while :; do
  [ -f "$QROOT/STOP" ] && { log "STOP present -- standing down"; exit 0; }
  if [ ! -f "$QROOT/idle_launcher.DONE" ]; then
    sleep 300; continue   # the Jacobian one-shot owns the idle trigger until it finishes
  fi
  if [ -f "$HOLD" ]; then
    [ "$idle" -gt 0 ] && log "HOLD present -- idle counter reset from $idle"
    idle=0; sleep 300; continue
  fi
  if out=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); then
    napps=$(printf '%s' "$out" | grep -c '[0-9]')
  else
    napps=999   # nvidia-smi failed: treat as busy
  fi
  if [ "$napps" -eq 0 ]; then
    idle=$((idle+1))
  else
    [ "$idle" -gt 0 ] && log "activity detected ($napps compute apps) -- idle counter reset from $idle"
    idle=0
    rm -f "$DRYFLAG"   # box is in use again; a future dry spell re-alarms
  fi
  if [ "$idle" -ge "$NEED" ]; then
    npend=$(ls "$QROOT/pending" 2>/dev/null | wc -l | tr -d ' ')
    nclaim=$(ls "$QROOT/claimed" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$npend" -gt 0 ] || [ "$nclaim" -gt 0 ]; then
      # Queue has work yet the box idles 3h: PAUSE is set or workers are
      # down. Not a promotion problem — do not stack more; the worker
      # watchdog / a human owns this. Log and re-arm.
      log "idle >=3h but pending=$npend claimed=$nclaim -- queue already has work (PAUSE or workers down?); not promoting"
      idle=0
    else
      wave=$(ls "$POOL" 2>/dev/null | sort | head -n "$WAVE")
      if [ -z "$wave" ]; then
        if [ ! -f "$DRYFLAG" ]; then
          touch "$DRYFLAG"
          log "ALERT: box idle >=3h, queue empty, fallback pool DRY -- runway starvation; refill requires the design/audit ceremony"
        fi
        idle=0
      else
        for f in $wave; do
          mv "$POOL/$f" "$QROOT/pending/$f" && log "promoted $f from fallback pool"
        done
        idle=0
      fi
    fi
  fi
  sleep 300
done
