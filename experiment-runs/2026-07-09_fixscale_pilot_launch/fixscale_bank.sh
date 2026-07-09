#!/bin/bash
# fixscale_bank.sh -- self-healing supervisor for ONE gate-tier GPU bank of the
# FIX-AT-SCALE wave (FROZEN_BIAS_LM_DESIGN.md sec 13.10 items 3+4; launched via
# tmux session fixscale_pilots, one window per bank -- the standing
# tmux+supervisor unattended-run discipline). Stop cleanly by touching
# /home/nvidia/chapter2/deltanet_rd/STOP_fixscale (NEVER pkill).
#
# Banks (GPU pinned inside the driver call; GPUs 1-4 -- GPU 0 belongs to
# attrrob_2x2, GPUs 5-7 left free for the concurrent capability wave):
#   bank98    GPU 1: 98M timing pilot (two-point, off+per_token) -> 98M openr1 calib
#   calib98w  GPU 2: waits for the 98M pilot verdict -> 98M wikitext calib
#   bank392   GPU 3: 392M timing pilot -> 392M openr1 calib
#   calib392w GPU 4: waits for the 392M pilot verdict -> 392M wikitext calib
#
# rc convention (fixscale_gate_tier.py): 0 = done/terminal-skip (incl. a
# REFUSED calibration after a FAIL pilot verdict -- deliberately exit 0 so this
# loop does not spin); 2 = terminal timed_out pilot (do NOT auto-retry --
# inspect first); anything else = transient, retry in 60s (the driver is
# resume-safe: completed sub-runs are skipped by output validity, not existence).
set -u
cd /home/nvidia/chapter2/deltanet_rd || exit 65
PYBIN=/home/nvidia/tdenv/bin/python
BANK="${1:?usage: fixscale_bank.sh <bank98|calib98w|bank392|calib392w>}"
mkdir -p results/fixscale
LOG="results/fixscale/${BANK}.log"

run_until_terminal () {
  while true; do
    "$@" >> "$LOG" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then return 0; fi
    if [ "$rc" -eq 2 ]; then echo "TERMINAL rc=2 from: $*" >> "$LOG"; return 2; fi
    if [ -f STOP_fixscale ]; then echo "STOPPED via STOP_fixscale" >> "$LOG"; return 3; fi
    echo "transient rc=$rc from: $* -- retry in 60s ($(date -u))" >> "$LOG"
    sleep 60
  done
}

wait_for_file () {
  while [ ! -f "$1" ]; do
    [ -f STOP_fixscale ] && { echo "STOPPED while waiting for $1" >> "$LOG"; return 3; }
    sleep 60
  done
  return 0
}

case "$BANK" in
  bank98)
    run_until_terminal "$PYBIN" fixscale_gate_tier.py pilot --scale 98m --gpu 1 && \
    run_until_terminal "$PYBIN" fixscale_gate_tier.py calib --scale 98m --corpus openr1-mix-ext --gpu 1
    ;;
  calib98w)
    wait_for_file results/fixscale/pilots/PILOT_98M_VERDICT.json && \
    run_until_terminal "$PYBIN" fixscale_gate_tier.py calib --scale 98m --corpus wikitext-mix-ext --gpu 2
    ;;
  bank392)
    run_until_terminal "$PYBIN" fixscale_gate_tier.py pilot --scale 392m --gpu 3 && \
    run_until_terminal "$PYBIN" fixscale_gate_tier.py calib --scale 392m --corpus openr1-mix-ext --gpu 3
    ;;
  calib392w)
    wait_for_file results/fixscale/pilots/PILOT_392M_VERDICT.json && \
    run_until_terminal "$PYBIN" fixscale_gate_tier.py calib --scale 392m --corpus wikitext-mix-ext --gpu 4
    ;;
  *) echo "unknown bank: $BANK" >> "$LOG"; exit 64 ;;
esac
echo "BANK $BANK finished rc-chain=$? $(date -u)" >> "$LOG"
# keep the pane alive for post-hoc inspection; session teardown is manual
sleep infinity
