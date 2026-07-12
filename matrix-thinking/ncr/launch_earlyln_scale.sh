#!/usr/bin/env bash
# NCR EARLY-LN K-SCALING launcher (box-side). NOVEL_ARCH_WATERFALL.md §11:
# does the earlyln recipe (§8.9's parameter-free inter-hop LayerNorm,
# annealed 1->0 over the first training half, eval always pure-matmul
# exact) extend K=14's convergence up the K axis, and does exact
# composition survive to far depth for cells that converge? Grid: K in
# {14,15,16,24} x seeds {0,1,2,3} = 16 cells, one script (ncr_earlyln_scale.py)
# ADDITIVE to the existing ncr/ tree (imports ncr_task/ncr_models/ncr_spectral/
# run_ncr verbatim; the only shared-file edit anywhere is GRIDS[24], already
# additive and landed in ncr_task.py).
#
# Reused TWICE by design (the §9.4 Phase-0a discipline, mandatory per §11's
# own pre-registration): once at STEPS=500 (the rate probe, outdir
# results_earlyln_probe) to MEASURE the real per-K rate BEFORE committing
# the full budget, then again at STEPS=80000 (outdir results_earlyln_scale)
# once the §11 budget-fit rule has resolved the final seed list.
#
# 16 cells across 8 GPUs -- 2 cells/GPU, run SEQUENTIALLY per GPU (never
# concurrent on one device -- avoids VRAM/compute contention for no
# benefit, these cells are tiny and each GPU is 100% free otherwise).
# Same worker+supervisor+tmux+STOP-file+DONE-sentinel architecture as
# launch_k12ext.sh/launch_wcap_diag.sh.
#
# Usage (box):  bash launch_earlyln_scale.sh "G0 G1 G2 G3 G4 G5 G6 G7" STEPS RESULTS_SUBDIR [SEEDS_CSV] [K24_SEEDS_CSV]
#   e.g. probe:  bash launch_earlyln_scale.sh "0 1 2 3 4 5 6 7" 500 results_earlyln_probe 0,1,2,3
#   e.g. main:   bash launch_earlyln_scale.sh "0 1 2 3 4 5 6 7" 80000 results_earlyln_scale 0,1,2,3
#   e.g. trimmed (§11 budget-fit rule, K=24 only): ... results_earlyln_scale 0,1,2,3 0,1,2
# Stop:         touch ~/ncr/<RESULTS_SUBDIR>/STOP
# Kill (exact): tmux kill-session -t ncr_earlyln_scale_<RESULTS_SUBDIR>
set -euo pipefail

GPUS_STR="${1:?usage: launch_earlyln_scale.sh \"G0..G7\" STEPS RESULTS_SUBDIR [SEEDS_CSV] [K24_SEEDS_CSV]}"
STEPS="${2:?usage: ... STEPS RESULTS_SUBDIR [SEEDS_CSV] [K24_SEEDS_CSV]}"
SUBDIR="${3:?usage: ... STEPS RESULTS_SUBDIR [SEEDS_CSV] [K24_SEEDS_CSV]}"
SEEDS_CSV="${4:-0,1,2,3}"
K24_SEEDS_CSV="${5:-$SEEDS_CSV}"    # §11 budget-fit rule: trim K=24 ONLY, never K=15/16
# §11 pins the per-cell abort breaker at 1.5x the rate-probe-MEASURED rate,
# not a blind default (audit MINOR-1). Compute from the probe results
# before invoking the main run and export CEILING_GPUH; falls back to a
# conservative 2.0h if unset (e.g. during the probe run itself, which has
# no prior measurement to breaker against).
CEILING_GPUH="${CEILING_GPUH:-2.0}"
NCR_DIR="${NCR_DIR:-$HOME/ncr}"
RESULTS="$NCR_DIR/$SUBDIR"
SESSION="ncr_earlyln_scale_${SUBDIR}"
PY="${NCR_PYTHON:-$HOME/tdenv/bin/python}"
[ -x "$PY" ] || PY=python3

read -r -a GPUS <<< "$GPUS_STR"
[ "${#GPUS[@]}" -eq 8 ] || { echo "need exactly 8 GPUs (2 cells each x 16 cells)" >&2; exit 2; }
read -r -a SEEDS <<< "$(echo "$SEEDS_CSV" | tr ',' ' ')"
[ "${#SEEDS[@]}" -ge 1 ] || { echo "need at least 1 seed" >&2; exit 2; }
read -r -a K24_SEEDS <<< "$(echo "$K24_SEEDS_CSV" | tr ',' ' ')"
[ "${#K24_SEEDS[@]}" -ge 1 ] || { echo "need at least 1 K=24 seed" >&2; exit 2; }

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

mkdir -p "$RESULTS"
rm -f "$RESULTS/STOP" "$RESULTS/EARLYLN_SCALE_DONE"

# Build the (up to 16, fewer if K24_SEEDS_CSV is trimmed per the §11
# budget-fit rule -- K=24 trims first, never K=15/16) K:seed cells, K in
# {14,15,16,24} outer, seed inner -- contiguous blocks assigned round-robin
# across the 8 GPUs (ceil-division, so a trimmed grid still spreads evenly
# rather than piling onto the first few GPUs).
CELLS=()
for s in "${SEEDS[@]}"; do CELLS+=("14:$s"); done
for s in "${SEEDS[@]}"; do CELLS+=("15:$s"); done
for s in "${SEEDS[@]}"; do CELLS+=("16:$s"); done
for s in "${K24_SEEDS[@]}"; do CELLS+=("24:$s"); done
NCELLS="${#CELLS[@]}"
PER_GPU=$(( (NCELLS + 7) / 8 ))
[ "$PER_GPU" -ge 1 ] || { echo "internal: no cells to run" >&2; exit 2; }
echo "Grid: ${NCELLS} cells (K=14/15/16 seeds ${SEEDS[*]}; K=24 seeds ${K24_SEEDS[*]}), ~${PER_GPU}/GPU across 8 GPUs."

WORKER="$RESULTS/earlyln_worker.sh"
cat > "$WORKER" <<EOF
#!/usr/bin/env bash
GPU="\$1"; shift
cd "$NCR_DIR"
for cell in "\$@"; do
    [ -f "$RESULTS/STOP" ] && exit 3
    K="\${cell%%:*}"; seed="\${cell##*:}"
    CUDA_VISIBLE_DEVICES=\$GPU $PY ncr_earlyln_scale.py --cell --K "\$K" \\
        --seed "\$seed" --steps $STEPS --outdir "$RESULTS" \\
        --stop-file "$RESULTS/STOP" --ceiling-gpuh $CEILING_GPUH \\
        >> "$RESULTS/earlyln_cell_K\${K}_s\${seed}.log" 2>&1
done
EOF
chmod +x "$WORKER"

# Emit one per-GPU cell-list literal (bash arrays can't be exported cleanly
# into the supervisor heredoc, so write them as space-joined strings).
GPU_CELL_LISTS=()
for ((i = 0; i < 8; i++)); do
    lo=$(( i * PER_GPU ))
    hi=$(( lo + PER_GPU ))
    [ "$hi" -gt "$NCELLS" ] && hi="$NCELLS"
    if [ "$lo" -lt "$hi" ]; then
        GPU_CELL_LISTS+=("${CELLS[*]:$lo:$((hi-lo))}")
    else
        GPU_CELL_LISTS+=("")
    fi
done

SUPER="$RESULTS/earlyln_supervisor.sh"
{
    echo '#!/usr/bin/env bash'
    echo "log() { echo \"[supervisor] \$1 \$(date -u +%FT%TZ)\" >> \"$RESULTS/earlyln_supervisor.log\"; }"
    echo "log \"launch GPUs=($GPUS_STR) steps=$STEPS subdir=$SUBDIR cells=${CELLS[*]}\""
    echo "while [ ! -f \"$RESULTS/STOP\" ]; do"
    echo "    PIDS=()"
    for ((i = 0; i < 8; i++)); do
        if [ -n "${GPU_CELL_LISTS[$i]}" ]; then
            echo "    bash \"$WORKER\" ${GPUS[$i]} ${GPU_CELL_LISTS[$i]} & PIDS+=(\$!)"
        fi
    done
    echo "    wait \"\${PIDS[@]}\""
    echo "    N=\$($PY - <<PYEOF"
    echo "import glob, json"
    echo "n = 0"
    echo "for p in glob.glob(\"$RESULTS/earlyln_K*_s*.json\"):"
    echo "    if \".axis_c_lock.\" in p: continue"
    echo "    try:"
    echo "        if json.load(open(p)).get(\"status\") == \"COMPLETED\": n += 1"
    echo "    except Exception: pass"
    echo "print(n)"
    echo "PYEOF"
    echo "    )"
    echo "    log \"pass complete: \$N/$NCELLS cells COMPLETED\""
    echo "    if [ \"\$N\" -eq $NCELLS ]; then"
    echo "        date -u +%FT%TZ > \"$RESULTS/EARLYLN_SCALE_DONE\""
    echo "        log \"EARLYLN_SCALE_DONE written\""
    echo "        break"
    echo "    fi"
    echo "    sleep 15"
    echo "done"
    echo "log \"supervisor exiting\""
} > "$SUPER"
chmod +x "$SUPER"

tmux new-session -d -s "$SESSION" "bash $SUPER"
echo "Launched tmux '$SESSION': $NCELLS early-LN K-scaling cells across GPUs ($GPUS_STR), steps=$STEPS, outdir=$SUBDIR."
echo "  log:   tail -f $RESULTS/earlyln_supervisor.log"
echo "  stop:  touch $RESULTS/STOP"
echo "  kill:  tmux kill-session -t $SESSION   (exact, never pkill)"
