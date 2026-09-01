#!/usr/bin/env bash
# AUTO-UNPAUSE (coordinator, 2026-08-12, PI "get GPUs hot" directive 2026-08-11).
# The 2026-08-06 PAUSE was set by the gh_* (vLLM model-screen) workstream to
# stop queue claims while its servers hold the GPUs. This removes THAT pause
# (marker-checked) automatically once the gh_* workstream is fully gone, so
# staged research jobs start without a coordinator session alive.
# Conditions (ALL required; cron runs this every minute under flock):
#   1. ~/queue/PAUSE exists and its text mentions 'gh_'  (never removes a
#      pause someone else set for a different reason)
#   2. zero tmux sessions named gh_*  (sessions outlive engine restarts, so
#      this protects the between-evals engine-restart race that PAUSE was
#      created to guard against)
#   3. zero vLLM compute apps on any GPU (belt and suspenders; nvidia-smi
#      failure counts as busy — never act on blind data)
#   4. conditions 2+3 held on the previous minute too (2-consecutive checks)
# Workers additionally self-gate per-GPU (claim only on zero compute apps),
# so unpausing never places a job onto an occupied GPU.
# Test hooks: TEST_TMUX_GH / TEST_APPS override the probes; DRYRUN=1 logs the
# decision instead of removing.
set -u
Q="$HOME/queue"
P="$Q/PAUSE"
LOG="$Q/auto_unpause.log"
STATE="$Q/.auto_unpause_streak"
log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

[ -f "$P" ] || { rm -f "$STATE"; exit 0; }
grep -q 'gh_' "$P" || exit 0   # not the gh_ pause; never touch

tmux_gh=$(tmux ls 2>/dev/null | grep -c '^gh_')
if nvidia-smi >/dev/null 2>&1; then
  apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -ci 'vllm')
else
  apps=999   # nvidia-smi failed: treat as busy
fi
[ -n "${TEST_TMUX_GH:-}" ] && tmux_gh=$TEST_TMUX_GH
[ -n "${TEST_APPS:-}" ] && apps=$TEST_APPS

if [ "$tmux_gh" -eq 0 ] && [ "$apps" -eq 0 ]; then
  if [ -f "$STATE" ]; then
    if [ "${DRYRUN:-0}" = "1" ]; then
      log "DRYRUN: would remove PAUSE (gh_ sessions=0, vllm apps=0, streak confirmed)"
      exit 0
    fi
    rm -f "$P" "$STATE"
    log "REMOVED PAUSE: gh_* workstream gone (0 sessions, 0 vllm apps, 2-check streak); queue claims re-enabled"
  else
    touch "$STATE"
    log "conditions met once (gh_ sessions=0, vllm apps=0); confirming next check"
  fi
else
  rm -f "$STATE"
fi
