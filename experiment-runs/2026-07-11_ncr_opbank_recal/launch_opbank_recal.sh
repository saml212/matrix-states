#!/usr/bin/env bash
# NCR operator-bank Phase-0 RE-CALIBRATION launcher (box-side).
# NOVEL_ARCH_WATERFALL.md §8.5a: a single ncr-bank cell at the PROVEN
# single-relation recipe (batch 256, 80K steps) to test whether the
# contender converges at precedent budget after the 20K/48 Phase-0 FAIL
# (which was decisively a step-budget miss, NOT a bug). This is a
# Phase-0-mandate calibration cell -- NOT wave-1.
#
# Conventions (CLAUDE.md): tmux new-session, self-healing supervisor loop
# with STOP file, resume-safe (run_cell_bank skips COMPLETED), exact tmux
# kill only. REFUSES a busy GPU.
#
# Usage (box): bash launch_opbank_recal.sh <GPU_ID> [STEPS] [TRAIN_BATCH]
# Stop:        touch <NCR_DIR>/results_opbank_recal/STOP
# Kill:        tmux kill-session -t ncr_opbank_recal
set -euo pipefail

GPU_ID="${1:?usage: launch_opbank_recal.sh <GPU_ID> [STEPS] [TRAIN_BATCH]}"
STEPS="${2:-80000}"
TRAIN_BATCH="${3:-256}"
NCR_DIR="${NCR_DIR:-$HOME/ncr}"
RESULTS="$NCR_DIR/results_opbank_recal"
SESSION="ncr_opbank_recal"
PY="${NCR_PYTHON:-$HOME/tdenv/bin/python}"
[ -x "$PY" ] || PY=python3

mkdir -p "$RESULTS"

read -r MEM UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ',')
if [ "$MEM" -ge 2048 ] || [ "$UTIL" -ge 5 ]; then
    echo "REFUSING launch: GPU $GPU_ID busy (mem=${MEM}MiB util=${UTIL}%)." >&2
    exit 2
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "REFUSING launch: tmux session '$SESSION' already exists." >&2
    exit 2
fi

rm -f "$RESULTS/STOP" "$RESULTS/DONE"

WORKER="$RESULTS/recal_worker.sh"
cat > "$WORKER" <<EOF
#!/usr/bin/env bash
cd "$NCR_DIR"
while [ ! -f "$RESULTS/STOP" ]; do
    CUDA_VISIBLE_DEVICES=$GPU_ID $PY run_ncr_opbank.py --cell ncr-bank --seed 0 \\
        --steps $STEPS --train-batch $TRAIN_BATCH --ceiling-gpuh 3.0 \\
        --outdir "$RESULTS" --stop-file "$RESULTS/STOP" --device cuda \\
        >> "$RESULTS/recal_cell_ncr-bank.log" 2>&1
    rc=\$?
    echo "[recal] cell exited rc=\$rc at \$(date -u +%FT%TZ)" >> "$RESULTS/recal_supervisor.log"
    if [ \$rc -eq 0 ] || [ \$rc -eq 3 ]; then touch "$RESULTS/DONE"; break; fi
    sleep 15
done
echo '[recal] done' >> "$RESULTS/recal_supervisor.log"
EOF
chmod +x "$WORKER"

tmux new-session -d -s "$SESSION" "bash $WORKER"
echo "Launched tmux '$SESSION' on GPU $GPU_ID: ncr-bank, steps=$STEPS, batch=$TRAIN_BATCH."
echo "  log:  tail -f $RESULTS/recal_cell_ncr-bank.log"
echo "  stop: touch $RESULTS/STOP    kill: tmux kill-session -t $SESSION"
