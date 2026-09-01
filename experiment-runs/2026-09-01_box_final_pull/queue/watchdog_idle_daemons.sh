#!/usr/bin/env bash
# CRON WATCHDOG FOR THE IDLE DAEMONS (2026-08-06, adversarial-analysis fix
# F1: tmux-server death / box reboot revives queue workers via the existing
# watchdog but NOT idle_launcher/idle_fallback — the deferred launch would
# silently never fire and the GPUs would idle forever).
#
# Same safety construction as watchdog_workers.sh: only ever (re)launches a
# MISSING tmux session (has-session guard — a healthy session is never
# touched), honors the STOP sentinel, and skips a one-shot whose .DONE
# sentinel exists. Cron line (installed alongside the worker watchdog's):
#   * * * * * /usr/bin/flock -n $HOME/queue/.watchdog_idle.lock \
#       /usr/bin/bash $HOME/queue/watchdog_idle_daemons.sh >> $HOME/queue/watchdog_idle.log 2>&1
set -u
QROOT="${QROOT:-$HOME/queue}"
TS="$(date -u +%FT%TZ)"

if [ -f "$QROOT/STOP" ]; then
  echo "$TS STOP present -- idle-daemon watchdog standing down"
  exit 0
fi

ensure() {
  name="$1"; done_sentinel="$2"; script="$3"
  if [ -n "$done_sentinel" ] && [ -f "$done_sentinel" ]; then
    return 0
  fi
  if ! tmux has-session -t "$name" 2>/dev/null; then
    if [ -n "$done_sentinel" ]; then
      cond="while [ ! -f $QROOT/STOP ] && [ ! -f $done_sentinel ]"
    else
      cond="while [ ! -f $QROOT/STOP ]"
    fi
    tmux new-session -d -s "$name" "$cond; do bash $script; sleep 60; done"
    echo "$TS relaunched missing session: $name"
  fi
}

ensure idle_launcher "$QROOT/idle_launcher.DONE" "$QROOT/idle_launch_jacobian.sh"
ensure idle_fallback ""                          "$QROOT/idle_fallback_daemon.sh"
