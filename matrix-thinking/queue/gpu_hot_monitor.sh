#!/usr/bin/env bash
# GPU-HOT MONITOR (coordinator, 2026-08-17; PI directive "these GPUs are never
# ever not at 100%"). Runs every minute from cron under flock. Three jobs:
#
#   1. SAMPLE   — log per-GPU utilization + queue depth to gpu_hot.log (the
#                 utilization-not-occupancy record the doctrine asks for).
#   2. REFILL   — if the live queue is empty (pending+claimed == 0), promote up
#                 to REFILL_N specs from fallback_pool/ into pending/ IMMEDIATELY.
#                 This is the "never idle" path: it does not wait for the 3h
#                 idle_fallback gate, because an empty queue with free GPUs is
#                 already the failure the gate exists to prevent.
#   3. ALARM    — raise flags a human/agent can find later:
#                 GPU_UNDERUTILIZED  : jobs are claimed but sustained util <50%
#                                      (doctrine: sustained <50% is a bug)
#                 FALLBACK_POOL_DRY  : queue empty AND pool empty -> no runway
#
# Safety: never kills anything, never touches a GPU holding foreign processes,
# only ever MOVES audited spec files that a human/agent already placed in
# fallback_pool/. Honors PAUSE/STOP like every other queue actor.
set -u
Q="$HOME/queue"
POOL="$Q/fallback_pool"
LOG="$Q/gpu_hot.log"
STATE="$Q/.gpu_hot_lowutil_streak"
REFILL_N=${REFILL_N:-8}
LOWUTIL_NEED=${LOWUTIL_NEED:-10}   # consecutive minutes of <50% with work claimed

log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

[ -f "$Q/STOP" ] && exit 0

# ---- 1. sample -------------------------------------------------------------
if ! utils=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null); then
  log "nvidia-smi FAILED -- no action taken (never act on blind data)"
  exit 0
fi
n_gpu=$(printf '%s\n' "$utils" | grep -c .)
sum=0; hot=0
while read -r u; do
  [ -z "$u" ] && continue
  sum=$((sum + u))
  [ "$u" -ge 50 ] && hot=$((hot + 1))
done <<< "$utils"
mean=$(( n_gpu > 0 ? sum / n_gpu : 0 ))
n_pending=$(ls "$Q/pending" 2>/dev/null | wc -l | tr -d ' ')
n_claimed=$(ls "$Q/claimed" 2>/dev/null | wc -l | tr -d ' ')
n_pool=$(ls "$POOL" 2>/dev/null | wc -l | tr -d ' ')
log "util_mean=${mean}% hot_gpus=${hot}/${n_gpu} pending=${n_pending} claimed=${n_claimed} pool=${n_pool}"

# ---- 2. refill -------------------------------------------------------------
if [ "$n_pending" -eq 0 ] && [ "$n_claimed" -eq 0 ] && [ ! -f "$Q/PAUSE" ]; then
  if [ "$n_pool" -gt 0 ]; then
    moved=0
    for f in $(ls "$POOL"/*.json 2>/dev/null | head -n "$REFILL_N"); do
      mv "$f" "$Q/pending/" && moved=$((moved + 1))
    done
    [ "$moved" -gt 0 ] && log "REFILL: promoted $moved spec(s) from fallback_pool -> pending (queue was empty)"
    rm -f "$Q/FALLBACK_POOL_DRY"
  else
    if [ ! -f "$Q/FALLBACK_POOL_DRY" ]; then
      touch "$Q/FALLBACK_POOL_DRY"
      log "ALARM FALLBACK_POOL_DRY: queue empty AND pool empty -- GPUs will go idle, refill needed"
    fi
  fi
fi

# ---- 3. low-utilization alarm ---------------------------------------------
if [ "$n_claimed" -gt 0 ] && [ "$mean" -lt 50 ]; then
  streak=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
  echo "$streak" > "$STATE"
  if [ "$streak" -ge "$LOWUTIL_NEED" ] && [ ! -f "$Q/GPU_UNDERUTILIZED" ]; then
    touch "$Q/GPU_UNDERUTILIZED"
    log "ALARM GPU_UNDERUTILIZED: ${streak} consecutive minutes at ${mean}% mean util with ${n_claimed} job(s) claimed"
  fi
else
  rm -f "$STATE" "$Q/GPU_UNDERUTILIZED"
fi
