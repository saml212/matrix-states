#!/usr/bin/env bash
# Per-GPU NCR finisher. Waits for THIS GPU's driver tmux (ncr_ortho_g<GPU>) to
# exit (all its assigned cells attempted, OR a crash), then RELEASES this GPU's
# holder so queue_worker_g<GPU> reclaims the GPU and the box returns to full
# training. Survives the launching agent's session ending. The holder's 54h
# self-timeout is the ultimate backstop if this finisher itself dies.
set -u
GPU="${1:?usage: finish_ncr_ortho_v2.sh GPU_ID}"
SENT="/home/nvidia/queue/NCR_HOLD_RELEASE_g${GPU}"
DRV="ncr_ortho_g${GPU}"
HOLD="ncr_holder_g${GPU}"
LOG="/home/nvidia/ncr/results_ortho_write/finisher_g${GPU}.log"

echo "finisher_g$GPU up $(date -u); waiting for $DRV to finish" >> "$LOG"
# Wait for the driver loop to end (exits after all its cells are attempted).
while tmux has-session -t "$DRV" 2>/dev/null; do sleep 120; done
echo "$DRV ended $(date -u); releasing holder $HOLD" >> "$LOG"

# Release the holder (it polls this sentinel every 5s and exits).
touch "$SENT"
for i in $(seq 1 120); do
  tmux has-session -t "$HOLD" 2>/dev/null || break
  sleep 5
done
echo "holder $HOLD gone $(date -u)" >> "$LOG"

# Remove the sentinel so no future holder is auto-released by a stale file.
sleep 15
rm -f "$SENT"
echo "sentinel removed; GPU$GPU now reclaimable by worker g$GPU $(date -u)" >> "$LOG"
