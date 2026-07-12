#!/usr/bin/env bash
# NCR WRITE-CAPACITY DIAGNOSTIC launcher (box-side).
# NOVEL_ARCH_WATERFALL.md §9.7: the cheap (<=10 GPU-h hard) leg of §9's
# scaling ladder -- does growing encoder capacity proportionally with K
# (Condition B) rescue the write, at the minimal 4-cell set: spare-probe
# K=14 (d=16,h=64), spare-probe K=15 (d=16,h=64), Condition A K=16
# (d=32,h=64), Condition B K=16 (d=32,h=128). NCR arm only, seed 0, one
# cell per GPU (4 cells, 4 GPUs -- no co-location needed).
#
# Reused TWICE by design (§9.7's two-stage plan): once at STEPS=2000 (the
# Phase-0a rate probe, outdir results_wcap_probe) to MEASURE the real
# rate before any budget is committed, then again at the computed main
# step count (outdir results_wcap_diag) once §9.8 records that number.
# Same worker+supervisor+tmux+STOP-file+DONE-sentinel architecture as
# launch_k12ext.sh/launch_phase2.sh.
#
# Usage (box):  bash launch_wcap_diag.sh "GPU_A GPU_B GPU_C GPU_D" STEPS RESULTS_SUBDIR
#   e.g. probe: bash launch_wcap_diag.sh "2 3 4 5" 2000 results_wcap_probe
#   e.g. main:  bash launch_wcap_diag.sh "2 3 4 5" 40000 results_wcap_diag
# Stop:         touch ~/ncr/<RESULTS_SUBDIR>/STOP
# Kill (exact): tmux kill-session -t ncr_wcap_diag_<RESULTS_SUBDIR>
set -euo pipefail

GPUS_STR="${1:?usage: launch_wcap_diag.sh \"GPU_A GPU_B GPU_C GPU_D\" STEPS RESULTS_SUBDIR}"
STEPS="${2:?usage: ... STEPS RESULTS_SUBDIR}"
SUBDIR="${3:?usage: ... STEPS RESULTS_SUBDIR}"
NCR_DIR="${NCR_DIR:-$HOME/ncr}"
RESULTS="$NCR_DIR/$SUBDIR"
SESSION="ncr_wcap_diag_${SUBDIR}"
PY="${NCR_PYTHON:-$HOME/tdenv/bin/python}"
[ -x "$PY" ] || PY=python3

read -r -a GPUS <<< "$GPUS_STR"
[ "${#GPUS[@]}" -eq 4 ] || { echo "need exactly 4 GPUs (1 cell each x 4 diagnostic cells)" >&2; exit 2; }

for g in "${GPUS[@]}"; do
    if [ "$g" -eq 0 ] || [ "$g" -eq 1 ]; then
        echo "REFUSING launch: GPU $g is reserved, never touch." >&2
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

mkdir -p "$RESULTS"
rm -f "$RESULTS/STOP" "$RESULTS/WCAP_DIAG_DONE"

# label:K:d:h:rate80k -- rate80k is the CONDITION-SPECIFIC 1.678-GPU-h
# (conservative toy anchor) x §9.3-corrected F-ratio-scaled 80K-equivalent
# rate (§9.7's paper math), fed to run_ncr.py's own 1.5x-breaker so a
# runaway cell aborts at 1.5x ITS OWN projected cost, not the anchor's.
CELLS=(
  "spareK14:14:16:64:2.832"
  "spareK15:15:16:64:3.027"
  "condA_K16:16:32:64:3.527"
  "condB_K16:16:32:128:13.424"
)
[ "${#CELLS[@]}" -eq 4 ] || { echo "internal: CELLS must have 4 entries" >&2; exit 2; }

WORKER="$RESULTS/wcap_worker.sh"
cat > "$WORKER" <<EOF
#!/usr/bin/env bash
GPU="\$1"; LABEL="\$2"; K="\$3"; D="\$4"; H="\$5"; RATE="\$6"
cd "$NCR_DIR"
[ -f "$RESULTS/STOP" ] && exit 3
CUDA_VISIBLE_DEVICES=\$GPU $PY run_ncr.py --cell ncr --K "\$K" --d "\$D" --h "\$H" \\
    --seed 0 --steps $STEPS --outdir "$RESULTS" \\
    --stop-file "$RESULTS/STOP" --rate-gpuh-80k "\$RATE" \\
    >> "$RESULTS/wcap_cell_\${LABEL}.log" 2>&1
EOF
chmod +x "$WORKER"

SUPER="$RESULTS/wcap_supervisor.sh"
cat > "$SUPER" <<EOF
#!/usr/bin/env bash
log() { echo "[supervisor] \$1 \$(date -u +%FT%TZ)" >> "$RESULTS/wcap_supervisor.log"; }
log "launch GPUs=($GPUS_STR) steps=$STEPS subdir=$SUBDIR cells=${CELLS[*]}"
EXPECT=(ncr_ncr_K14_s0 ncr_ncr_K15_s0 ncr_ncr_K16_d32_h64_s0 ncr_ncr_K16_d32_h128_s0)
while [ ! -f "$RESULTS/STOP" ]; do
    PIDS=()
    bash "$WORKER" ${GPUS[0]} ${CELLS[0]//:/ } & PIDS+=(\$!)
    bash "$WORKER" ${GPUS[1]} ${CELLS[1]//:/ } & PIDS+=(\$!)
    bash "$WORKER" ${GPUS[2]} ${CELLS[2]//:/ } & PIDS+=(\$!)
    bash "$WORKER" ${GPUS[3]} ${CELLS[3]//:/ } & PIDS+=(\$!)
    wait "\${PIDS[@]}"
    N=0
    for e in "\${EXPECT[@]}"; do
        f="$RESULTS/\$e.json"
        if [ -f "\$f" ] && $PY -c "import json,sys; sys.exit(0 if json.load(open('\$f')).get('status')=='COMPLETED' else 1)" 2>/dev/null; then
            N=\$((N+1))
        fi
    done
    log "pass complete: \$N/4 cells COMPLETED"
    if [ "\$N" -eq 4 ]; then
        date -u +%FT%TZ > "$RESULTS/WCAP_DIAG_DONE"
        log "WCAP_DIAG_DONE written"
        break
    fi
    sleep 15
done
log "supervisor exiting"
EOF
chmod +x "$SUPER"

tmux new-session -d -s "$SESSION" "bash $SUPER"
echo "Launched tmux '$SESSION': 4 write-capacity diagnostic cells (${CELLS[*]}) across GPUs ($GPUS_STR), steps=$STEPS, outdir=$SUBDIR."
echo "  log:   tail -f $RESULTS/wcap_supervisor.log"
echo "  stop:  touch $RESULTS/STOP"
echo "  kill:  tmux kill-session -t $SESSION   (exact, never pkill)"
