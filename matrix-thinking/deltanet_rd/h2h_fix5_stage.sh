#!/bin/bash
# h2h_fix5_stage.sh -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.44: the flagship
# rebuttal's FIX-5 transformer-task1 LR-grid round (9 fresh cells: 3 LRs x
# 3 seeds; the 3e-4 column is reused from the round-4 sweep, never
# relaunched -- sec 1.44's own provenance record). Mirrors
# h2h_task2diag_stage.sh's discipline verbatim (gate-5 tokens, code-level
# margin-freeze release, validity-checked resume, 3-strike FATAL,
# mechanical budget ceiling) with two deliberate differences:
#   - GPUs 0-1 (this round's dispatch assignment; GPUs 2-7 are NCR
#     Phase-2's, never touched).
#   - 2-way PARALLEL cells (mirrors h2h_rung1_chain.sh's run_cells_par),
#     not single-GPU sequential -- 9 cells across 2 free GPUs.
#
# Launch (exactly once):
#   tmux new-session -d -s h2h_fix5_grid \
#     "while [ ! -f results/h2h_rung1/fix5_lrgrid/FIX5_STOP ] && \
#            [ ! -f results/h2h_rung1/fix5_lrgrid/FATAL ]; do \
#        bash h2h_fix5_stage.sh >> logs/h2h_fix5_supervisor.log 2>&1; sleep 15; done"
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
RES=results/h2h_rung1
FIX5=$RES/fix5_lrgrid
GATES=$RES/gates
CKPT_DIR=/data/h2h_rung1_ckpts
mkdir -p "$FIX5/remetric" "$FIX5/failcount" logs "$CKPT_DIR"

FIX5_GPUS=${FIX5_GPUS:-"0,1"}
if echo ",$FIX5_GPUS," | grep -qE ',[2-7],'; then
  echo "REFUSE: sec 1.44 pins this round to GPUs 0-1 (GPUs 2-7 are NCR Phase-2's)." >&2
  exit 1
fi
IFS=',' read -ra GPU_LIST <<< "$FIX5_GPUS"
N_GPUS=${#GPU_LIST[@]}
BUDGET_CEILING_GPU_H=6.0                     # sec 1.44 cost record: ~2.44 expected, 6.0 ceiling
MAX_CELL_STRIKES=3
DIAL_ROUND=4

[ -f "$FIX5/FIX5_STOP" ] && { echo "FIX5_STOP present -- round complete."; exit 0; }
[ -f "$FIX5/FATAL" ] && { echo "FATAL present -- refusing to run (coordinator review required)." >&2; exit 1; }

# --- sec 1.7 gate 5 tokens (files + env), verbatim from stage-D / task2diag ---
for tok in GATE6_MATCH_GATE_PASSED.token GATE7_PROBE_CAPACITY_NULL_PASSED.token; do
  if [ ! -s "$GATES/$tok" ]; then
    echo "REFUSE: $GATES/$tok missing." >&2
    touch "$FIX5/FATAL"
    exit 1
  fi
done
export HEADTOHEAD_PI_SIGNOFF=1
export HEADTOHEAD_MATCH_GATE_SIGNOFF=1
export H2H_DIAL_ROUND=$DIAL_ROUND            # filename versioning only; role="sweep" on every
                                              # cell here means the dial is never evaluated
$PY h2h_cell_train_rd.py --token-probe --gates-dir "$GATES" 2>&1 | tee logs/h2h_fix5_00_token_probe.log

if [ ! -s "$RES/MARGINS_FROZEN.token" ]; then
  echo "REFUSE: $RES/MARGINS_FROZEN.token missing." >&2
  exit 1
fi

budget_check() {
  local projected_gpu_h
  projected_gpu_h=$($PY -c "print(($SECONDS * $N_GPUS) / 3600.0)")
  echo "[budget] elapsed=${SECONDS}s n_gpus=$N_GPUS projected_gpu_h=${projected_gpu_h} ceiling=${BUDGET_CEILING_GPU_H}"
  if $PY -c "import sys; sys.exit(0 if $projected_gpu_h > $BUDGET_CEILING_GPU_H else 1)"; then
    echo "ABORT: projected GPU-h exceeds the sec 1.44 ceiling -- halting mechanically." >&2
    touch "$FIX5/FATAL"
    exit 1
  fi
}

run_step() {  # run_step <strike-key> <cmd...>: 3-strike FATAL wrapper (stage-D convention)
  local key=$1; shift
  if ! "$@"; then
    local fc="$FIX5/failcount/$key"
    echo x >> "$fc"
    if [ "$(wc -l < "$fc")" -ge $MAX_CELL_STRIKES ]; then
      echo "FATAL: step $key failed $MAX_CELL_STRIKES times -- deterministic failure, halting." >&2
      touch "$FIX5/FATAL"
    fi
    echo "ERROR: step $key FAILED -- exiting (supervisor retries unless FATAL)." >&2
    exit 1
  fi
}

# --- 0) ONE tiny real-CUDA smoke cell (proves the production path end-to-end before the
#     full launch); its checkpoint is harmlessly overwritten by that cell's real run below. ---
if [ ! -f "$FIX5/SMOKE_PASSED.token" ]; then
  echo "SMOKE: h2h_fix5_transformer_task1_lr1e-04_s0 (steps_override=3, gpu=${GPU_LIST[0]})"
  run_step smoke env CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PY" h2h_fix5_lrgrid_rd.py \
      --run-cell h2h_fix5_transformer_task1_lr1e-04_s0 \
      --out "$FIX5/SMOKE_h2h_fix5_transformer_task1_lr1e-04_s0.json" \
      --steps-override 3 --ckpt-dir "$CKPT_DIR" --gates-dir "$GATES" \
      --margins-token "$RES/MARGINS_FROZEN.token" --device cuda 2>&1 | tee logs/h2h_fix5_smoke.log
  touch "$FIX5/SMOKE_PASSED.token"
fi

# --- 1) the 9 fresh training cells, 2-way parallel on GPUs 0-1, resume-safe ---
mapfile -t FIX5_CELLS < <($PY h2h_fix5_lrgrid_rd.py --list-cells)
if [ "${#FIX5_CELLS[@]}" -ne 9 ]; then
  echo "REFUSE: expected 9 fresh FIX-5 cells, got ${#FIX5_CELLS[@]}" >&2
  exit 1
fi

pending=()
for name in "${FIX5_CELLS[@]}"; do
  out="$FIX5/${name}.json"
  if $PY -c "import sys; from h2h_sweep_runner_rd import is_valid_result; sys.exit(0 if is_valid_result('$out') else 1)" 2>/dev/null; then
    echo "SKIP (already valid): $out"
    continue
  fi
  pending+=("$name")
done

i=0
while [ $i -lt ${#pending[@]} ]; do
  pids=(); names=()
  for slot in $(seq 0 $((N_GPUS - 1))); do
    idx=$((i + slot))
    [ $idx -ge ${#pending[@]} ] && break
    name="${pending[$idx]}"
    out="$FIX5/${name}.json"
    log="logs/h2h_fix5_${name}.log"
    gpu="${GPU_LIST[$slot]}"
    echo "LAUNCH (gpu=$gpu): $name -> $out"
    ( CUDA_VISIBLE_DEVICES="$gpu" $PY h2h_fix5_lrgrid_rd.py --run-cell "$name" \
          --out "$out" --ckpt-dir "$CKPT_DIR" --gates-dir "$GATES" \
          --margins-token "$RES/MARGINS_FROZEN.token" --device cuda 2>&1 | tee "$log" ) &
    pids+=($!); names+=("$name")
  done
  fail=0; k=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
      fc="$FIX5/failcount/${names[$k]}"
      echo x >> "$fc"
      if [ "$(wc -l < "$fc")" -ge $MAX_CELL_STRIKES ]; then
        echo "FATAL: cell ${names[$k]} failed $MAX_CELL_STRIKES times -- deterministic failure, halting." >&2
        touch "$FIX5/FATAL"
      fi
    fi
    k=$((k + 1))
  done
  if [ "$fail" -ne 0 ]; then
    echo "ERROR: >=1 cell in this GPU wave FAILED -- exiting (supervisor retries unless FATAL)." >&2
    exit 1
  fi
  budget_check
  i=$((i + N_GPUS))
done

# --- 2) audited re-metric pass (acc_A via the SAME run_cell_round4 every other h2h number uses) ---
run_step remetric env CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "$PY" h2h_fix5_lrgrid_rd.py \
    --remetric --ckpt-dir "$CKPT_DIR" --remetric-dir "$FIX5/remetric" --dial-round "$DIAL_ROUND" \
    --gates-dir "$GATES" --margins-token "$RES/MARGINS_FROZEN.token" \
    --device cuda 2>&1 | tee logs/h2h_fix5_remetric.log
budget_check

touch "$FIX5/FIX5_STOP"
echo "FIX5 LR-GRID ROUND COMPLETE (FIX5_STOP written). Harvest next: h2h_fix5_lrgrid_rd.py --harvest"
echo "(torch-free; run locally off the downloaded $FIX5/ + $FIX5/remetric/ artifacts, or on-box)."
