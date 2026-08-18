#!/usr/bin/env bash
# GPU-HOT MONITOR (coordinator, 2026-08-17; PI directive "these GPUs are never
# ever not at 100%"). Runs every minute from cron under flock. Three jobs:
#
#   1. SAMPLE   — log per-GPU utilization + queue depth to gpu_hot.log (the
#                 utilization-not-occupancy record the doctrine asks for).
#   2. REFILL   — if there are IDLE GPUs and nothing pending, promote up to one
#                 spec per idle GPU from fallback_pool/ into pending/ IMMEDIATELY.
#                 It does not wait for the 3h idle_fallback gate, and (fixed
#                 2026-08-18, first live tick after install) it does NOT require
#                 the queue to be fully drained: a draining wave leaves free GPUs
#                 while jobs are still claimed, which was exactly the hole the
#                 original `pending+claimed == 0` predicate left open.
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
# An IDLE GPU is one with no compute process on it (occupancy, not utilization --
# a GPU running a real job at a low instantaneous sample is NOT idle).
if busy_uuids=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null); then
  n_busy=$(printf '%s\n' "$busy_uuids" | grep -c . || true)
else
  n_busy=$n_gpu   # probe failed: assume fully busy, never refill on blind data
fi
n_idle=$(( n_gpu - n_busy ))
[ "$n_idle" -lt 0 ] && n_idle=0

if [ "$n_idle" -gt 0 ] && [ "$n_pending" -eq 0 ] && [ ! -f "$Q/PAUSE" ]; then
  if [ "$n_pool" -gt 0 ]; then
    want=$(( n_idle < REFILL_N ? n_idle : REFILL_N ))
    moved=0
    for f in $(ls "$POOL"/*.json 2>/dev/null | head -n "$want"); do
      mv "$f" "$Q/pending/" && moved=$((moved + 1))
    done
    [ "$moved" -gt 0 ] && log "REFILL: promoted $moved spec(s) from fallback_pool -> pending (${n_idle} idle GPU(s), 0 pending)"
    rm -f "$Q/FALLBACK_POOL_DRY"
  else
    if [ ! -f "$Q/FALLBACK_POOL_DRY" ]; then
      touch "$Q/FALLBACK_POOL_DRY"
      log "ALARM FALLBACK_POOL_DRY: ${n_idle} idle GPU(s), 0 pending, pool EMPTY -- refill needed"
    fi
  fi
fi

# ---- 2b. disk guard --------------------------------------------------------
# Added 2026-08-18 after the root filesystem hit 100% mid-wave and killed 12
# training cells at their step-10000 checkpoint (torch.save iostream error).
# Training writes ~2.2GB per cell; a queue that refills itself can fill a disk
# faster than a human notices, so utilization monitoring without a disk guard
# is incomplete. PAUSEs the queue rather than letting jobs die at their save.
root_pct=$(df -P / | awk 'NR==2 {gsub(/%/,"",$5); print $5}')
log "disk_root=${root_pct}%"
if [ -n "$root_pct" ] && [ "$root_pct" -ge 92 ]; then
  if [ ! -f "$Q/DISK_CRITICAL" ]; then
    touch "$Q/DISK_CRITICAL"
    log "ALARM DISK_CRITICAL: root filesystem at ${root_pct}% -- pausing new claims (jobs die at checkpoint save when full)"
  fi
  # PAUSE is honored by every worker; a human/agent clears it after freeing space.
  [ -f "$Q/PAUSE" ] || echo "Auto-paused by gpu_hot_monitor: root fs at ${root_pct}%. Free space (checkpoints belong on /ephemeral), then rm PAUSE and DISK_CRITICAL." > "$Q/PAUSE"
elif [ -n "$root_pct" ] && [ "$root_pct" -lt 85 ]; then
  if [ -f "$Q/DISK_CRITICAL" ]; then
    rm -f "$Q/DISK_CRITICAL"
    grep -q "gpu_hot_monitor" "$Q/PAUSE" 2>/dev/null && rm -f "$Q/PAUSE"
    log "disk recovered to ${root_pct}% -- DISK_CRITICAL cleared, auto-PAUSE lifted"
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
