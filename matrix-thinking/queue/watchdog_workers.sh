#!/usr/bin/env bash
# META-SUPERVISOR WATCHDOG (added 2026-07-15, red-team idle-GPU hardening).
#
# CLOSES THE ONE REMAINING SINGLE POINT OF FAILURE in the queue system:
# each per-GPU worker is kept alive by a self-healing supervisor loop
#   tmux new-session -d -s queue_worker_g<N> \
#     "while [ ! -f $QROOT/STOP ]; do bash queue_worker.sh <N>; sleep 15; done"
# That loop respawns queue_worker.sh if the *inner script* crashes -- BUT the
# loop itself runs INSIDE the tmux session, so if the whole SESSION dies (tmux
# server killed, node hiccup/reboot, the session OOM-killed), the loop dies
# with it and NOTHING respawns that worker. Its GPU then sits idle -- and this
# box is uptime-metered, so an idle GPU bills for nothing until a human notices
# and re-runs launch_workers.sh by hand. This already happened once: worker
# queue_worker_g1 was manually relaunched 2026-07-13, a full day after the
# other 7 (Jul 12) -- i.e. g1's session had died and its GPU was unsupervised
# until someone caught it.
#
# cron is the root-of-tree supervisor that survives tmux/SSH/Claude death.
# This script is what cron runs (default: every 1 min):
#   * * * * * /usr/bin/flock -n $HOME/queue/.watchdog.lock \
#       /usr/bin/bash $HOME/queue/watchdog_workers.sh >> $HOME/queue/watchdog.log 2>&1
#
# SAFE BY CONSTRUCTION -- this can NEVER disturb a healthy live worker or the
# 8 live training jobs / the falcon eval:
#   1. It only ever calls launch_workers.sh, which is IDEMPOTENT: its
#      `tmux has-session` guard means a session that already exists is SKIPPED,
#      never killed, restarted, or duplicated. Only a genuinely-MISSING session
#      is (re)started. So the steady-state action is 8x "skip: already running".
#   2. Verified 2026-07-15: a minimal cron-like environment
#      (`env -i PATH=/usr/bin:/bin HOME=/home/nvidia tmux ls`) sees the SAME
#      tmux server and all 8 live sessions -- so cron's tmux will correctly
#      skip them rather than spawn a duplicate set on a hidden socket.
#   3. Honors the STOP sentinel (the global kill switch): if STOP is present,
#      the watchdog stands down and does NOT relaunch, so it never fights an
#      intentional shutdown. (PAUSE is deliberately NOT honored: PAUSE keeps
#      workers alive but idle-claiming; we still want them supervised so they
#      resume instantly when PAUSE clears.)
#   4. flock -n (in the cron line) makes an overlapping tick a no-op rather
#      than stacking a second launch pass.
#
# To DISABLE: `crontab -e` and remove the line (or `crontab -r` if this is the
# only entry). Fully reversible; leaves no residue beyond watchdog.log.
set -u
QROOT="${QROOT:-$HOME/queue}"
TS="$(date -u +%FT%TZ)"

if [ -f "$QROOT/STOP" ]; then
  echo "$TS STOP present -- watchdog standing down (intentional shutdown, not relaunching)"
  exit 0
fi

# Idempotent: relaunches ONLY missing sessions, skips healthy ones. Its own
# stdout (the "skip: ... already running" / "launched: ..." lines) is captured
# by the cron redirect, so watchdog.log doubles as a session-health heartbeat.
echo "$TS watchdog tick -- ensuring all 8 queue_worker sessions are up"
bash "$QROOT/launch_workers.sh"
