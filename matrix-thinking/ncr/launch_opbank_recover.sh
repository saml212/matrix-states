#!/usr/bin/env bash
# NCR operator-bank CONVERGENCE-RECOVERY diagnosis launcher (box-side).
# NOVEL_ARCH_WATERFALL.md §8.7/§8.8: 4 recipe arms (baseline/warmup/earlyln/
# curriculum), one GPU per arm, on GPUs 0/1/6/7 ONLY (2/3/4/5 belong to the
# §9 write-capacity diagnostic -- NEVER touched).
#
# Conventions (CLAUDE.md): tmux new-session, self-healing supervisor loop
# with STOP file, resume-safe (run_recover_cell skips COMPLETED), exact
# tmux kill only. REFUSES any busy GPU.
#
# Usage (box): bash launch_opbank_recover.sh [STEPS] [TRAIN_BATCH]
# Stop:        touch <NCR_DIR>/results_opbank_recover/STOP
# Kill:        tmux kill-session -t ncr_opbank_recover
set -euo pipefail

STEPS="${1:-80000}"
TRAIN_BATCH="${2:-256}"
NCR_DIR="${NCR_DIR:-$HOME/ncr}"
RESULTS="$NCR_DIR/results_opbank_recover"
SESSION="ncr_opbank_recover"
PY="${NCR_PYTHON:-$HOME/tdenv/bin/python}"
[ -x "$PY" ] || PY=python3

# Fixed arm->GPU map, GPUs 0/1/6/7 ONLY (2/3/4/5 = §9 diagnostic, off-limits).
declare -A ARM_GPU=( [baseline]=0 [warmup]=1 [earlyln]=6 [curriculum]=7 )

mkdir -p "$RESULTS"

for arm in "${!ARM_GPU[@]}"; do
    gpu="${ARM_GPU[$arm]}"
    read -r MEM UTIL < <(nvidia-smi --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits -i "$gpu" | tr -d ',')
    if [ "$MEM" -ge 2048 ] || [ "$UTIL" -ge 5 ]; then
        echo "REFUSING launch: GPU $gpu (arm $arm) busy (mem=${MEM}MiB util=${UTIL}%)." >&2
        exit 2
    fi
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "REFUSING launch: tmux session '$SESSION' already exists." >&2
    exit 2
fi

rm -f "$RESULTS/STOP" "$RESULTS/DONE"

WORKER="$RESULTS/recover_worker.sh"
cat > "$WORKER" <<EOF
#!/usr/bin/env bash
cd "$NCR_DIR"
while [ ! -f "$RESULTS/STOP" ]; do
    for arm in baseline warmup earlyln curriculum; do
        case \$arm in
            baseline) gpu=0;; warmup) gpu=1;; earlyln) gpu=6;; curriculum) gpu=7;;
        esac
        CUDA_VISIBLE_DEVICES=\$gpu $PY ncr_opbank_recover.py --cell \$arm --seed 0 \\
            --steps $STEPS --train-batch $TRAIN_BATCH --ceiling-gpuh 2.0 \\
            --outdir "$RESULTS" --stop-file "$RESULTS/STOP" --device cuda \\
            >> "$RESULTS/recover_cell_\${arm}.log" 2>&1 &
    done
    wait
    # DONE iff all 4 cells wrote a COMPLETED json
    n_done=\$($PY -c "import json,glob; print(sum(1 for f in glob.glob('$RESULTS/ncropbank_recover_*.json') if json.load(open(f)).get('status')=='COMPLETED'))")
    echo "[recover] round complete, \$n_done/4 COMPLETED at \$(date -u +%FT%TZ)" >> "$RESULTS/recover_supervisor.log"
    if [ "\$n_done" -ge 4 ]; then touch "$RESULTS/DONE"; break; fi
    sleep 15
done
echo '[recover] done' >> "$RESULTS/recover_supervisor.log"
EOF
chmod +x "$WORKER"

tmux new-session -d -s "$SESSION" "bash $WORKER"
echo "Launched tmux '$SESSION': baseline=GPU0 warmup=GPU1 earlyln=GPU6 curriculum=GPU7, steps=$STEPS, batch=$TRAIN_BATCH."
echo "  log:  tail -f $RESULTS/recover_supervisor.log"
echo "  stop: touch $RESULTS/STOP    kill: tmux kill-session -t $SESSION"
