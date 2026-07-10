#!/usr/bin/env bash
# NCR wave-1 launcher (box-side). NOVEL_ARCH_WATERFALL.md §7d LAUNCH TERMS:
# Phase-1 grid (ncr/loopedvec/fwm x seeds 0-4 + cmlp x seeds 0-2 = 18 cells,
# K=8), 80K steps, hard sub-cap 50 GPU-h, two idle GPUs, tmux `ncr_wave1`,
# self-healing supervisor, resume-by-validity, STOP file, WAVE1_DONE
# sentinel at 18/18 COMPLETED.
#
# Results dir is results_wave1/ (FRESH -- §7d: wave-1 seed-0 cells at 80K
# are distinct experiments from Phase-0's 40K seed-0 cells; sharing the dir
# would false-skip them via the COMPLETED check).
#
# Breaker rate: the Phase-0 MEASURED 1.1185 GPU-h/80K-equiv (itself measured
# under 3-way co-location -- no extra contention allowance taken).
#
# Usage (box):  bash launch_wave1.sh [GPU_A] [GPU_B] [STEPS]
# Stop:         touch ~/ncr/results_wave1/STOP
# Kill (exact): tmux kill-session -t ncr_wave1
set -euo pipefail

GPU_A="${1:-6}"
GPU_B="${2:-7}"
STEPS="${3:-80000}"
RATE="1.1185"
NCR_DIR="${NCR_DIR:-$HOME/ncr}"
RESULTS="$NCR_DIR/results_wave1"
SESSION="ncr_wave1"
PY="${NCR_PYTHON:-$HOME/tdenv/bin/python}"
[ -x "$PY" ] || PY=python3

mkdir -p "$RESULTS"

for g in "$GPU_A" "$GPU_B"; do
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
rm -f "$RESULTS/STOP" "$RESULTS/WAVE1_DONE"

# 18 cells, round-robin sharded across the two GPUs (balanced arms).
ALL_CELLS=(ncr:0 ncr:1 ncr:2 ncr:3 ncr:4 \
           loopedvec:0 loopedvec:1 loopedvec:2 loopedvec:3 loopedvec:4 \
           fwm:0 fwm:1 fwm:2 fwm:3 fwm:4 \
           cmlp:0 cmlp:1 cmlp:2)
SHARD_A=""; SHARD_B=""
for i in "${!ALL_CELLS[@]}"; do
    if [ $((i % 2)) -eq 0 ]; then SHARD_A+="${ALL_CELLS[$i]} "; else SHARD_B+="${ALL_CELLS[$i]} "; fi
done

# Per-GPU worker: runs its shard in chunks of 3 co-located cells (the
# Phase-0-proven concurrency). run_ncr.py --cell is resume-safe (COMPLETED
# cells skip; partial cells resume from their last valid checkpoint), so a
# supervisor restart of the whole worker converges without losing work.
WORKER="$RESULTS/wave1_worker.sh"
cat > "$WORKER" <<EOF
#!/usr/bin/env bash
GPU="\$1"; shift
cd "$NCR_DIR"
CELLS=("\$@")
for ((i=0; i<\${#CELLS[@]}; i+=3)); do
    [ -f "$RESULTS/STOP" ] && exit 3
    for cell in "\${CELLS[@]:i:3}"; do
        arm="\${cell%%:*}"; seed="\${cell##*:}"
        CUDA_VISIBLE_DEVICES=\$GPU $PY run_ncr.py --cell "\$arm" --K 8 \\
            --seed "\$seed" --steps $STEPS --outdir "$RESULTS" \\
            --stop-file "$RESULTS/STOP" --rate-gpuh-80k $RATE \\
            >> "$RESULTS/wave1_cell_\${arm}_s\${seed}.log" 2>&1 &
    done
    wait
done
EOF
chmod +x "$WORKER"

SUPER="$RESULTS/wave1_supervisor.sh"
cat > "$SUPER" <<EOF
#!/usr/bin/env bash
log() { echo "[supervisor] \$1 \$(date -u +%FT%TZ)" >> "$RESULTS/wave1_supervisor.log"; }
log "launch GPUs=$GPU_A/$GPU_B steps=$STEPS rate=$RATE shards: A=[$SHARD_A] B=[$SHARD_B]"
while [ ! -f "$RESULTS/STOP" ]; do
    bash "$WORKER" $GPU_A $SHARD_A &
    PA=\$!
    bash "$WORKER" $GPU_B $SHARD_B &
    PB=\$!
    wait \$PA \$PB
    N=\$($PY - <<'PYEOF'
import glob, json
n = 0
for p in glob.glob("$RESULTS/ncr_*_K8_s*.json"):
    try:
        if json.load(open(p)).get("status") == "COMPLETED": n += 1
    except Exception: pass
print(n)
PYEOF
)
    log "pass complete: \$N/18 cells COMPLETED"
    if [ "\$N" -eq 18 ]; then
        date -u +%FT%TZ > "$RESULTS/WAVE1_DONE"
        log "WAVE1_DONE written"
        break
    fi
    sleep 15
done
log "supervisor exiting"
EOF
chmod +x "$SUPER"

tmux new-session -d -s "$SESSION" "bash $SUPER"
echo "Launched tmux '$SESSION': 18 cells, GPUs $GPU_A+$GPU_B, steps=$STEPS, chunks of 3/GPU."
echo "  log:   tail -f $RESULTS/wave1_supervisor.log"
echo "  stop:  touch $RESULTS/STOP"
echo "  kill:  tmux kill-session -t $SESSION   (exact, never pkill)"
