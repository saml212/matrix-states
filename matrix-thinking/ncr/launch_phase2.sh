#!/usr/bin/env bash
# NCR Phase-2 launcher (box-side). NOVEL_ARCH_WATERFALL.md §7f LAUNCH TERMS:
# the K=12 replication grid (ncr/loopedvec/fwm x seeds 0-4 + cmlp x 0-2 =
# 18 cells), 80K steps, MA3 pre-registration untouched, six GPUs x 3
# co-located cells = ONE chunk round, tmux `ncr_phase2`, supervisor,
# STOP file, PHASE2_DONE sentinel at 18/18.
#
# Breaker rate 1.678 = 1.1185 (K=8-measured) x 1.5 (K cost axis: 12/8
# encoder tokens + queries) -- the §2.29 anchor-scaled-ceiling lesson
# applied a priori; realized K=12 rates are measured into the §7g record.
#
# Usage (box):  bash launch_phase2.sh "2 3 4 5 6 7" [STEPS]
# Stop:         touch ~/ncr/results_phase2/STOP
# Kill (exact): tmux kill-session -t ncr_phase2
set -euo pipefail

GPUS_STR="${1:-2 3 4 5 6 7}"
STEPS="${2:-80000}"
RATE="1.678"
K="12"
NCR_DIR="${NCR_DIR:-$HOME/ncr}"
RESULTS="$NCR_DIR/results_phase2"
SESSION="ncr_phase2"
PY="${NCR_PYTHON:-$HOME/tdenv/bin/python}"
[ -x "$PY" ] || PY=python3

read -r -a GPUS <<< "$GPUS_STR"
[ "${#GPUS[@]}" -eq 6 ] || { echo "need exactly 6 GPUs (3 cells each x 18)" >&2; exit 2; }

mkdir -p "$RESULTS"

for g in "${GPUS[@]}"; do
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
rm -f "$RESULTS/STOP" "$RESULTS/PHASE2_DONE"

ALL_CELLS=(ncr:0 ncr:1 ncr:2 ncr:3 ncr:4 \
           loopedvec:0 loopedvec:1 loopedvec:2 loopedvec:3 loopedvec:4 \
           fwm:0 fwm:1 fwm:2 fwm:3 fwm:4 \
           cmlp:0 cmlp:1 cmlp:2)

# Worker: one GPU, its 3 cells co-located in a single parallel chunk.
WORKER="$RESULTS/phase2_worker.sh"
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
        >> "$RESULTS/phase2_cell_\${arm}_s\${seed}.log" 2>&1 &
done
wait
EOF
chmod +x "$WORKER"

SUPER="$RESULTS/phase2_supervisor.sh"
cat > "$SUPER" <<EOF
#!/usr/bin/env bash
log() { echo "[supervisor] \$1 \$(date -u +%FT%TZ)" >> "$RESULTS/phase2_supervisor.log"; }
log "launch GPUs=($GPUS_STR) K=$K steps=$STEPS rate=$RATE"
while [ ! -f "$RESULTS/STOP" ]; do
    PIDS=()
    bash "$WORKER" ${GPUS[0]} ${ALL_CELLS[0]} ${ALL_CELLS[1]} ${ALL_CELLS[2]} & PIDS+=(\$!)
    bash "$WORKER" ${GPUS[1]} ${ALL_CELLS[3]} ${ALL_CELLS[4]} ${ALL_CELLS[5]} & PIDS+=(\$!)
    bash "$WORKER" ${GPUS[2]} ${ALL_CELLS[6]} ${ALL_CELLS[7]} ${ALL_CELLS[8]} & PIDS+=(\$!)
    bash "$WORKER" ${GPUS[3]} ${ALL_CELLS[9]} ${ALL_CELLS[10]} ${ALL_CELLS[11]} & PIDS+=(\$!)
    bash "$WORKER" ${GPUS[4]} ${ALL_CELLS[12]} ${ALL_CELLS[13]} ${ALL_CELLS[14]} & PIDS+=(\$!)
    bash "$WORKER" ${GPUS[5]} ${ALL_CELLS[15]} ${ALL_CELLS[16]} ${ALL_CELLS[17]} & PIDS+=(\$!)
    wait "\${PIDS[@]}"
    N=\$($PY - <<'PYEOF'
import glob, json
n = 0
for p in glob.glob("$RESULTS/ncr_*_K${K}_s*.json"):
    if ".axis_c_lock." in p: continue
    try:
        if json.load(open(p)).get("status") == "COMPLETED": n += 1
    except Exception: pass
print(n)
PYEOF
)
    log "pass complete: \$N/18 cells COMPLETED"
    if [ "\$N" -eq 18 ]; then
        date -u +%FT%TZ > "$RESULTS/PHASE2_DONE"
        log "PHASE2_DONE written"
        break
    fi
    sleep 15
done
log "supervisor exiting"
EOF
chmod +x "$SUPER"

tmux new-session -d -s "$SESSION" "bash $SUPER"
echo "Launched tmux '$SESSION': 18 K=12 cells across GPUs ($GPUS_STR), steps=$STEPS."
echo "  log:   tail -f $RESULTS/phase2_supervisor.log"
echo "  stop:  touch $RESULTS/STOP"
echo "  kill:  tmux kill-session -t $SESSION   (exact, never pkill)"
