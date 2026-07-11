#!/usr/bin/env bash
# NCR K=12 SEED-EXTENSION launcher (box-side). NOVEL_ARCH_WATERFALL.md §7h:
# 5 fresh ncr-arm seeds (5,6,7,8,9) at K=12, 80K steps, IDENTICAL frozen
# config to §7f/§7g (ncr_task.py/run_ncr.py/ncr_models.py/ncr_spectral.py
# unmodified). Baselines (loopedvec/fwm) are NOT re-run -- ncr arm only.
# One cell per GPU (5 cells, 5 of the 6 idle GPUs) -- no co-location
# needed since cell count < GPU count; the 6th GPU stays idle/reserve.
# Same worker+supervisor+tmux+STOP-file+DONE-sentinel architecture as
# launch_phase2.sh (§7f), fresh results dir so this wave's cells never
# collide with or mutate the archived results_phase2/ directory.
#
# Usage (box):  bash launch_k12ext.sh "2 3 4 5 6" [STEPS]
# Stop:         touch ~/ncr/results_k12ext/STOP
# Kill (exact): tmux kill-session -t ncr_k12ext
set -euo pipefail

GPUS_STR="${1:-2 3 4 5 6}"
STEPS="${2:-80000}"
RATE="1.678"          # identical to §7f's K-cost-scaled breaker rate
K="12"
NCR_DIR="${NCR_DIR:-$HOME/ncr}"
RESULTS="$NCR_DIR/results_k12ext"
SESSION="ncr_k12ext"
PY="${NCR_PYTHON:-$HOME/tdenv/bin/python}"
[ -x "$PY" ] || PY=python3

read -r -a GPUS <<< "$GPUS_STR"
[ "${#GPUS[@]}" -eq 5 ] || { echo "need exactly 5 GPUs (1 cell each x 5 fresh ncr seeds)" >&2; exit 2; }

mkdir -p "$RESULTS"

for g in "${GPUS[@]}"; do
    if [ "$g" -eq 0 ] || [ "$g" -eq 1 ]; then
        echo "REFUSING launch: GPU $g is reserved for the FIX-5 grid + reasoning-link validation, never touch." >&2
        exit 2
    fi
    read -r MEM UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits -i "$g" | tr -d ',')
    if [ "$MEM" -ge 2048 ] || [ "$UTIL" -ge 5 ]; then
        echo "REFUSING launch: GPU $g busy (mem=${MEM}MiB util=${UTIL}%)." >&2
        exit 2
    fi
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "REFUSING launch: tmux session '$SESSION' already exists." >&2
    exit 2
fi
rm -f "$RESULTS/STOP" "$RESULTS/K12EXT_DONE"

# Fresh seeds only -- never overlaps the archived s0-s4 pool.
CELLS=(ncr:5 ncr:6 ncr:7 ncr:8 ncr:9)

# Worker: one GPU, its single cell.
WORKER="$RESULTS/k12ext_worker.sh"
cat > "$WORKER" <<EOF
#!/usr/bin/env bash
GPU="\$1"; shift
cd "$NCR_DIR"
[ -f "$RESULTS/STOP" ] && exit 3
for cell in "\$@"; do
    arm="\${cell%%:*}"; seed="\${cell##*:}"
    CUDA_VISIBLE_DEVICES=\$GPU $PY run_ncr.py --cell "\$arm" --K $K \\
        --seed "\$seed" --steps $STEPS --outdir "$RESULTS" \\
        --stop-file "$RESULTS/STOP" --rate-gpuh-80k $RATE \\
        >> "$RESULTS/k12ext_cell_\${arm}_s\${seed}.log" 2>&1 &
done
wait
EOF
chmod +x "$WORKER"

SUPER="$RESULTS/k12ext_supervisor.sh"
cat > "$SUPER" <<EOF
#!/usr/bin/env bash
log() { echo "[supervisor] \$1 \$(date -u +%FT%TZ)" >> "$RESULTS/k12ext_supervisor.log"; }
log "launch GPUs=($GPUS_STR) K=$K steps=$STEPS rate=$RATE seeds=5,6,7,8,9 (ncr arm only)"
while [ ! -f "$RESULTS/STOP" ]; do
    PIDS=()
    bash "$WORKER" ${GPUS[0]} ${CELLS[0]} & PIDS+=(\$!)
    bash "$WORKER" ${GPUS[1]} ${CELLS[1]} & PIDS+=(\$!)
    bash "$WORKER" ${GPUS[2]} ${CELLS[2]} & PIDS+=(\$!)
    bash "$WORKER" ${GPUS[3]} ${CELLS[3]} & PIDS+=(\$!)
    bash "$WORKER" ${GPUS[4]} ${CELLS[4]} & PIDS+=(\$!)
    wait "\${PIDS[@]}"
    N=\$($PY - <<'PYEOF'
import glob, json
n = 0
for p in glob.glob("$RESULTS/ncr_ncr_K${K}_s*.json"):
    if ".axis_c_lock." in p: continue
    try:
        if json.load(open(p)).get("status") == "COMPLETED": n += 1
    except Exception: pass
print(n)
PYEOF
)
    log "pass complete: \$N/5 cells COMPLETED"
    if [ "\$N" -eq 5 ]; then
        date -u +%FT%TZ > "$RESULTS/K12EXT_DONE"
        log "K12EXT_DONE written"
        break
    fi
    sleep 15
done
log "supervisor exiting"
EOF
chmod +x "$SUPER"

tmux new-session -d -s "$SESSION" "bash $SUPER"
echo "Launched tmux '$SESSION': 5 K=12 ncr-only cells (seeds 5-9) across GPUs ($GPUS_STR), steps=$STEPS."
echo "  log:   tail -f $RESULTS/k12ext_supervisor.log"
echo "  stop:  touch $RESULTS/STOP"
echo "  kill:  tmux kill-session -t $SESSION   (exact, never pkill)"
