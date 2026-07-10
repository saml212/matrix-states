#!/usr/bin/env bash
# NCR Phase-0 launcher (box-side). NOVEL_ARCH_WATERFALL.md S3.6 Phase 0:
# the three trained-arm calibration cells (K=8, seed 0) + gate table.
#
# Conventions (CLAUDE.md hard rules): tmux new-session (never backgrounded
# ssh shells), self-healing supervisor loop with a STOP file, resume-safe
# orchestrator (run_ncr.py skips COMPLETED cells and resumes from valid
# checkpoints), exact tmux kills only (tmux kill-session -t ncr_phase0 --
# NEVER pkill).
#
# GPU selection: REFUSES to launch unless the requested GPU is genuinely
# idle (memory < 2 GiB used AND util < 5%) -- never preempts running work
# (a Stage-2 sweep and a task2 round own the box until they finish).
#
# Usage (on the box):   bash launch_phase0.sh <GPU_ID> [STEPS]
# Stop:                 touch <REPO>/matrix-thinking/ncr/results/STOP
# Kill (exact):         tmux kill-session -t ncr_phase0
set -euo pipefail

GPU_ID="${1:?usage: launch_phase0.sh <GPU_ID> [STEPS]}"
STEPS="${2:-40000}"
# Box layout (H100_SETUP.md convention): code is scp'd per-directory, not a
# git clone -- the Task-E lineage lives at ~/chapter2/ and this directory
# deploys as its SIBLING ~/ncr/ (so ncr_task.py's ../chapter2 relative
# import resolves identically on the box and in the repo).
NCR_DIR="${NCR_DIR:-$HOME/ncr}"
RESULTS="$NCR_DIR/results"
SESSION="ncr_phase0"
PY="${NCR_PYTHON:-$HOME/tdenv/bin/python}"
[ -x "$PY" ] || PY=python3

mkdir -p "$RESULTS"

# --- idle check: never touch a busy GPU -------------------------------------
read -r MEM UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ',' )
if [ "$MEM" -ge 2048 ] || [ "$UTIL" -ge 5 ]; then
    echo "REFUSING launch: GPU $GPU_ID busy (mem=${MEM}MiB util=${UTIL}%)." >&2
    echo "Pick a genuinely idle GPU (nvidia-smi) -- never preempt running work." >&2
    exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "REFUSING launch: tmux session '$SESSION' already exists." >&2
    exit 2
fi

rm -f "$RESULTS/STOP"

# --- worker script (written to disk so the tmux command stays auditable) ----
# Co-location: the three Phase-0 cells are tiny (<1 GiB, d=16, Python-bound)
# and run in PARALLEL on the ONE granted GPU, so the device draw stays near
# the longest single cell (~1.2 GPU-h at the 2.4/80K anchor) instead of the
# 3.6 GPU-h serial-sum -- the coordinator's <=2 GPU-h Phase-0 cap is the
# operative constraint. The per-cell breaker rate is widened 2x (4.8) for
# the co-located run ONLY, per the S2.21/S2.29 contention-false-abort
# lesson (a breaker must not abort a healthy cell for sharing a device);
# TRUE realized rates are measured from each cell's own elapsed time and
# recorded in phase0_rate.json -- wave 1 plans against those, not the
# widened allowance. After all three cells complete, --phase0 re-runs as a
# cheap gate-table pass (COMPLETED cells skip; resume-safe by validity).
WORKER="$RESULTS/phase0_worker.sh"
cat > "$WORKER" <<EOF
#!/usr/bin/env bash
cd "$NCR_DIR"
while [ ! -f "$RESULTS/STOP" ]; do
    for arm in ncr loopedvec fwm; do
        CUDA_VISIBLE_DEVICES=$GPU_ID $PY run_ncr.py --cell \$arm --K 8 --seed 0 \
            --steps $STEPS --outdir "$RESULTS" --stop-file "$RESULTS/STOP" \
            --rate-gpuh-80k 4.8 \
            >> "$RESULTS/phase0_cell_\${arm}.log" 2>&1 &
    done
    wait
    CUDA_VISIBLE_DEVICES=$GPU_ID $PY run_ncr.py --phase0 --steps $STEPS \
        --outdir "$RESULTS" --stop-file "$RESULTS/STOP" \
        >> "$RESULTS/phase0_supervisor.log" 2>&1
    rc=\$?
    echo "[supervisor] gate pass exited rc=\$rc at \$(date -u +%FT%TZ)" \
        >> "$RESULTS/phase0_supervisor.log"
    if [ \$rc -eq 0 ] || [ \$rc -eq 3 ]; then break; fi
    sleep 15
done
echo '[supervisor] done' >> "$RESULTS/phase0_supervisor.log"
EOF
chmod +x "$WORKER"

tmux new-session -d -s "$SESSION" "bash $WORKER"
echo "Launched tmux session '$SESSION' on GPU $GPU_ID (steps=$STEPS, 3 cells co-located)."
echo "  log:   tail -f $RESULTS/phase0_supervisor.log"
echo "  stop:  touch $RESULTS/STOP"
echo "  kill:  tmux kill-session -t $SESSION   (exact, never pkill)"
