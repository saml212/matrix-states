#!/usr/bin/env bash
# Starts (or, run again later, IDEMPOTENTLY restarts only the missing) the
# 8 per-GPU queue workers, each in its own tmux session wrapped in the
# house self-healing supervisor loop. Safe to run at any time regardless
# of current GPU occupancy -- workers themselves poll-and-wait for a free
# GPU (queue_worker.sh's own free-GPU gate), so launching does NOT require
# the GPUs to be idle right now.
#
# Usage (box-side): bash ~/queue/launch_workers.sh ["0 1 2 3 4 5 6 7"]
set -u

QROOT="${QROOT:-$HOME/queue}"
GPUS="${1:-0 1 2 3 4 5 6 7}"
mkdir -p "$QROOT/pending" "$QROOT/claimed" "$QROOT/completed" "$QROOT/failed" "$QROOT/logs"

for g in $GPUS; do
  SESSION="queue_worker_g${g}"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "skip: $SESSION already running"
    continue
  fi
  tmux new-session -d -s "$SESSION" \
    "cd $QROOT && while [ ! -f $QROOT/STOP ]; do bash $QROOT/queue_worker.sh $g; sleep 15; done"
  echo "launched: $SESSION"
done

echo "---"
tmux ls 2>/dev/null | grep '^queue_worker_g' || echo "(no queue_worker_g* sessions found -- launch failed?)"
