#!/bin/bash
# fixscale_supervisor.sh -- self-healing supervisor for the FIX-AT-SCALE wave's FULL SWEEP
# (FROZEN_BIAS_LM_DESIGN.md sec 13.9/13.10, fixscale_wave.py's own manifest/sweep/pin driver).
# Distinct from fixscale_bank.sh (already deployed+running, gate-tier ONLY: timing pilots +
# arm_off seed-0 calibration, commit ea5c3d7) -- this script drives the REMAINING arm_off seeds,
# all arm_per_token/arm_global_probe cells, and the blind-pin barrier. It does NOT relaunch or
# collide with the gate tier's own tmux session/results layout (fixscale_wave.py's TRAIN_ROOT/
# PIN_ROOT are disjoint from the gate tier's pilots/calib dirs; arm_off seed=0 is recognized via
# the gate tier's own output, never re-run -- see fixscale_wave.py's `out_path`).
#
# Sequencing per scale (sec 13.10 gate 5's own ordering requirement, mechanically enforced by
# fixscale_wave.py itself, not just this script): armoff slots run first -> ONE pin-leader session
# barriers on all 6 arm_off cells then writes BANDS_PINNED-FrozenBias-<SCALE>.json -> post_pin
# slots (arm_per_token + arm_global_probe) each call check_blind() immediately before their own
# subprocess launch and REFUSE (retryable, sec 13.12's own call-site requirement) until the pin
# exists.
#
# Stop cleanly by touching STOP_fixscale_wave in this directory (NEVER pkill -- CLAUDE.md hard
# rule: a pattern-matched pkill on a remote box can match the SSH command string itself).
#
# Usage:
#   fixscale_supervisor.sh launch-scale <98m|392m> <n_gpus> <gpu_offset>
#       Fans out into 2*n_gpus + 1 tmux sessions (n_gpus armoff slots, 1 pin leader, n_gpus
#       post_pin slots), each self-healing. Run once per scale; call with DIFFERENT gpu_offset
#       values for 98m vs 392m so their GPU ranges don't overlap (sec 13.9's own partition, GPU
#       occupancy re-verified live before calling this -- `nvidia-smi`/`tmux ls` first, per house
#       rule, NOT hardcoded here).
#   fixscale_supervisor.sh armoff-loop  <scale> <n_gpus> <gpu_offset> <slot>
#   fixscale_supervisor.sh pin-loop     <scale>
#   fixscale_supervisor.sh post-loop    <scale> <n_gpus> <gpu_offset> <slot>
#       The actual per-tmux-session work loops (launch-scale calls these; can also be invoked
#       directly for a single slot, e.g. to add a slot after occupancy changes).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 65
PYBIN="${FIXSCALE_PYBIN:-python3}"
DRIVER="./fixscale_wave.py"
STOP_FILE="STOP_fixscale_wave"
mkdir -p results/fixscale/train results/fixscale/pins

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# run_until_terminal CMD... -- retries CMD every 60s on a nonzero exit (transient: OOM, disk
# hiccup, or -- for post-loop -- the blind pin not written yet), stops cleanly on STOP_FILE, never
# retries a CMD that itself already returned 0 (terminal success FOR THIS INVOCATION; the caller's
# own loop decides whether to advance to the next slot's cells or exit).
run_until_terminal() {
  while true; do
    "$@"
    rc=$?
    if [ "$rc" -eq 0 ]; then return 0; fi
    if [ -f "$STOP_FILE" ]; then log "STOPPED via $STOP_FILE (last rc=$rc from: $*)"; return 3; fi
    log "transient rc=$rc from: $* -- retry in 60s"
    sleep 60
  done
}

armoff_loop() {
  local scale="$1" n_gpus="$2" gpu_offset="$3" slot="$4"
  log "armoff-loop scale=$scale slot=$slot/$n_gpus gpu=$((gpu_offset+slot)) starting"
  run_until_terminal "$PYBIN" "$DRIVER" sweep --scale "$scale" --phase arm_off \
      --gpus "$n_gpus" --gpu-offset "$gpu_offset" --slot "$slot"
  log "armoff-loop scale=$scale slot=$slot done rc=$?"
}

pin_loop() {
  local scale="$1"
  log "pin-loop scale=$scale starting (barrier on all 6 arm_off cells)"
  run_until_terminal "$PYBIN" "$DRIVER" await-armoff-and-pin --scale "$scale" \
      --poll-s 60 --stop-path "$STOP_FILE"
  log "pin-loop scale=$scale done rc=$?"
}

post_loop() {
  local scale="$1" n_gpus="$2" gpu_offset="$3" slot="$4"
  log "post-loop scale=$scale slot=$slot/$n_gpus gpu=$((gpu_offset+slot)) starting (waits on the pin)"
  run_until_terminal "$PYBIN" "$DRIVER" sweep --scale "$scale" --phase post_pin \
      --gpus "$n_gpus" --gpu-offset "$gpu_offset" --slot "$slot"
  log "post-loop scale=$scale slot=$slot done rc=$?"
  echo "keeping pane alive for inspection"; sleep infinity
}

launch_scale() {
  local scale="$1" n_gpus="$2" gpu_offset="$3"
  log "launch-scale $scale: $n_gpus GPUs at offset $gpu_offset (GPUs $gpu_offset..$((gpu_offset+n_gpus-1)))"
  for slot in $(seq 0 $((n_gpus - 1))); do
    tmux new-session -d -s "fixscale_${scale}_armoff_${slot}" \
      "$0 armoff-loop $scale $n_gpus $gpu_offset $slot"
  done
  tmux new-session -d -s "fixscale_${scale}_pin" "$0 pin-loop $scale"
  for slot in $(seq 0 $((n_gpus - 1))); do
    tmux new-session -d -s "fixscale_${scale}_post_${slot}" \
      "$0 post-loop $scale $n_gpus $gpu_offset $slot"
  done
  log "launch-scale $scale: $((2 * n_gpus + 1)) tmux sessions started " \
      "(armoff x$n_gpus, pin x1, post_pin x$n_gpus -- post_pin sessions will retry/wait until the pin lands)"
}

case "${1:-}" in
  launch-scale) launch_scale "$2" "$3" "$4" ;;
  armoff-loop)  armoff_loop "$2" "$3" "$4" "$5" ;;
  pin-loop)     pin_loop "$2" ;;
  post-loop)    post_loop "$2" "$3" "$4" "$5" ;;
  *)
    echo "usage: $0 {launch-scale <98m|392m> <n_gpus> <gpu_offset> |" >&2
    echo "            armoff-loop <scale> <n_gpus> <gpu_offset> <slot> |" >&2
    echo "            pin-loop <scale> |" >&2
    echo "            post-loop <scale> <n_gpus> <gpu_offset> <slot>}" >&2
    exit 64
    ;;
esac
