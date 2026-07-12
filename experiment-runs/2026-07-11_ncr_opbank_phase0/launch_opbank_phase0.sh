#!/usr/bin/env bash
# NCR operator-bank Phase-0 launcher (box-side). NOVEL_ARCH_WATERFALL.md
# S8.1.7 Phase 0: the three trained-arm calibration cells (ncr-bank,
# loopedvec-bank, fwm-bank; K=8, R=3, seed 0) + gate table.
#
# Conventions (CLAUDE.md hard rules): tmux new-session (never backgrounded
# ssh shells), self-healing supervisor loop with a STOP file, resume-safe
# orchestrator (run_ncr_opbank.py's run_cell_bank skips COMPLETED cells),
# exact tmux kills only (tmux kill-session -t ncr_opbank_p0 -- NEVER pkill).
#
# Unlike the single-relation Phase-0 (3 cells co-located on ONE GPU), this
# launcher assigns ONE GPU PER CELL -- 3 distinct idle GPUs run in parallel
# (genuine device parallelism, not just co-location contention), since the
# box has 8 idle GPUs and only 3 pre-registered cells this wave; this
# shortens wall-clock without inventing any extra cells beyond what S8.1.7
# pre-registered.
#
# GPU selection: REFUSES to launch on any of the 3 GPUs unless genuinely
# idle (memory < 2 GiB used AND util < 5%) -- never preempts running work.
#
# Usage (on the box): bash launch_opbank_phase0.sh <GPU_NCR> <GPU_LOOPEDVEC> <GPU_FWM> [STEPS]
# Stop:               touch <NCR_DIR>/results_opbank_phase0/STOP
# Kill (exact):        tmux kill-session -t ncr_opbank_p0
set -euo pipefail

GPU_NCR="${1:?usage: launch_opbank_phase0.sh <GPU_NCR> <GPU_LOOPEDVEC> <GPU_FWM> [STEPS]}"
GPU_LOOPEDVEC="${2:?missing GPU_LOOPEDVEC}"
GPU_FWM="${3:?missing GPU_FWM}"
STEPS="${4:-20000}"
NCR_DIR="${NCR_DIR:-$HOME/ncr}"
RESULTS="$NCR_DIR/results_opbank_phase0"
SESSION="ncr_opbank_p0"
PY="${NCR_PYTHON:-$HOME/tdenv/bin/python}"
[ -x "$PY" ] || PY=python3

mkdir -p "$RESULTS"

check_idle() {
    local gpu="$1"
    read -r MEM UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits -i "$gpu" | tr -d ',')
    if [ "$MEM" -ge 2048 ] || [ "$UTIL" -ge 5 ]; then
        echo "REFUSING launch: GPU $gpu busy (mem=${MEM}MiB util=${UTIL}%)." >&2
        echo "Pick a genuinely idle GPU (nvidia-smi) -- never preempt running work." >&2
        exit 2
    fi
}
check_idle "$GPU_NCR"
check_idle "$GPU_LOOPEDVEC"
check_idle "$GPU_FWM"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "REFUSING launch: tmux session '$SESSION' already exists." >&2
    exit 2
fi

rm -f "$RESULTS/STOP" "$RESULTS/DONE"

WORKER="$RESULTS/opbank_p0_worker.sh"
cat > "$WORKER" <<EOF
#!/usr/bin/env bash
cd "$NCR_DIR"
while [ ! -f "$RESULTS/STOP" ]; do
    CUDA_VISIBLE_DEVICES=$GPU_NCR $PY run_ncr_opbank.py --cell ncr-bank --seed 0 \\
        --steps $STEPS --outdir "$RESULTS" --stop-file "$RESULTS/STOP" \\
        --ceiling-gpuh 2.0 --device cuda \\
        >> "$RESULTS/opbank_p0_cell_ncr-bank.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU_LOOPEDVEC $PY run_ncr_opbank.py --cell loopedvec-bank --seed 0 \\
        --steps $STEPS --outdir "$RESULTS" --stop-file "$RESULTS/STOP" \\
        --ceiling-gpuh 2.0 --device cuda \\
        >> "$RESULTS/opbank_p0_cell_loopedvec-bank.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=$GPU_FWM $PY run_ncr_opbank.py --cell fwm-bank --seed 0 \\
        --steps $STEPS --outdir "$RESULTS" --stop-file "$RESULTS/STOP" \\
        --ceiling-gpuh 2.0 --device cuda \\
        >> "$RESULTS/opbank_p0_cell_fwm-bank.log" 2>&1 &
    wait
    CUDA_VISIBLE_DEVICES=$GPU_NCR $PY run_ncr_opbank.py --phase0 --steps $STEPS \\
        --outdir "$RESULTS" --stop-file "$RESULTS/STOP" --device cuda \\
        >> "$RESULTS/opbank_p0_supervisor.log" 2>&1
    rc=\$?
    echo "[supervisor] gate pass exited rc=\$rc at \$(date -u +%FT%TZ)" \\
        >> "$RESULTS/opbank_p0_supervisor.log"
    if [ \$rc -eq 0 ] || [ \$rc -eq 3 ]; then
        touch "$RESULTS/DONE"
        break
    fi
    sleep 15
done
echo '[supervisor] done' >> "$RESULTS/opbank_p0_supervisor.log"
EOF
chmod +x "$WORKER"

tmux new-session -d -s "$SESSION" "bash $WORKER"
echo "Launched tmux session '$SESSION': ncr-bank on GPU $GPU_NCR, loopedvec-bank on GPU $GPU_LOOPEDVEC, fwm-bank on GPU $GPU_FWM (steps=$STEPS, one GPU per cell)."
echo "  log:   tail -f $RESULTS/opbank_p0_supervisor.log"
echo "  stop:  touch $RESULTS/STOP"
echo "  kill:  tmux kill-session -t $SESSION   (exact, never pkill)"
