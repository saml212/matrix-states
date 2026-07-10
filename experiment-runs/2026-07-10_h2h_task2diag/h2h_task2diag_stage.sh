#!/bin/bash
# h2h_task2diag_stage.sh -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.42: the TASK2
# DIAGNOSIS ROUND (12-cell sec 1.8 seed extension) + the transformer_K48 stress
# cell (sec 1.37's sweep-stage path, sec 1.42 A6). Mirrors h2h_sweep_stage_d.sh's
# discipline verbatim (gate-5 tokens, code-level margin-freeze release + blind
# negative test, validity-checked resume, 3-strike FATAL, mechanical budget
# ceiling) with TWO deliberate differences, both pinned in sec 1.42:
#   - GPU 7 ONLY (this round's dispatch assignment; GPUs 0-6 belong to the
#     Stage-2 sweep). The stage-D script's own GPU-7 refusal was that script's
#     directive, superseded for THIS script by the sec 1.42 dispatch record.
#   - single-GPU sequential cells (no run_cells_par).
#
# Launch (exactly once):
#   tmux new-session -d -s h2h_task2diag \
#     "while [ ! -f results/h2h_rung1/task2diag/TASK2DIAG_STOP ] && \
#            [ ! -f results/h2h_rung1/task2diag/FATAL ]; do \
#        bash h2h_task2diag_stage.sh >> logs/h2h_task2diag_supervisor.log 2>&1; sleep 15; done"
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
RES=results/h2h_rung1
DIAG=$RES/task2diag
GATES=$RES/gates
CKPT_DIR=/data/h2h_rung1_ckpts
mkdir -p "$DIAG/remetric" "$DIAG/failcount" logs "$CKPT_DIR"

TASK2DIAG_GPU=${TASK2DIAG_GPU:-7}
if [ "$TASK2DIAG_GPU" != "7" ]; then
  echo "REFUSE: sec 1.42 pins this round to GPU 7 (GPUs 0-6 are the Stage-2 sweep's)." >&2
  exit 1
fi
BUDGET_CEILING_GPU_H=8.0                    # sec 1.42 cost record: ~4.1 expected, 8.0 ceiling
MAX_CELL_STRIKES=3

[ -f "$DIAG/TASK2DIAG_STOP" ] && { echo "TASK2DIAG_STOP present -- round complete."; exit 0; }
[ -f "$DIAG/FATAL" ] && { echo "FATAL present -- refusing to run (coordinator review required)." >&2; exit 1; }

# --- sec 1.7 gate 5 tokens (files + env), verbatim from stage-D ---
for tok in GATE6_MATCH_GATE_PASSED.token GATE7_PROBE_CAPACITY_NULL_PASSED.token BOX_SMOKE_ITEMS_1_4_PASSED.token; do
  if [ ! -s "$GATES/$tok" ]; then
    echo "REFUSE: $GATES/$tok missing." >&2
    touch "$DIAG/FATAL"
    exit 1
  fi
done
export HEADTOHEAD_PI_SIGNOFF=1
export HEADTOHEAD_MATCH_GATE_SIGNOFF=1
# Filename versioning only (_r4): every training cell here carries role="sweep",
# so the dial is never evaluated (AUD2-F4 structural guard; sec 1.42 A1/A6).
export H2H_DIAL_ROUND=4
$PY h2h_cell_train_rd.py --token-probe --gates-dir "$GATES" 2>&1 | tee logs/h2h_task2diag_00_token_probe.log

# --- MARGIN FREEZE blind check (sec 13.13 pattern), verbatim from stage-D ---
if [ ! -s "$RES/MARGINS_FROZEN.token" ]; then
  echo "REFUSE: $RES/MARGINS_FROZEN.token missing." >&2
  exit 1
fi
$PY - "$RES/MARGINS_FROZEN.token" <<'PYEOF' 2>&1 | tee logs/h2h_task2diag_01_blind_check.log
import json, sys, time
sys.path.insert(0, ".")
from key_anchoring import assert_blind_not_broken
tok = json.load(open(sys.argv[1]))
now = time.time()
try:
    assert_blind_not_broken({"pinned_at": now + 3600.0}, [now])
except AssertionError:
    pass
else:
    raise SystemExit("NEGATIVE TEST FAILED TO FAIL: a future pinned_at did not raise")
assert_blind_not_broken(tok, [now])
print(f"blind check PASS: pinned_at={tok['pinned_at']} strictly precedes launch={now}; negative test exercised")
PYEOF

budget_check() {
  local projected_gpu_h
  projected_gpu_h=$($PY -c "print($SECONDS / 3600.0)")     # 1 GPU
  echo "[budget] elapsed=${SECONDS}s projected_gpu_h=${projected_gpu_h} ceiling=${BUDGET_CEILING_GPU_H}"
  if $PY -c "import sys; sys.exit(0 if $projected_gpu_h > $BUDGET_CEILING_GPU_H else 1)"; then
    echo "ABORT: projected GPU-h exceeds the sec 1.42 ceiling -- halting mechanically." >&2
    touch "$DIAG/FATAL"
    exit 1
  fi
}

run_step() {  # run_step <strike-key> <cmd...>: 3-strike FATAL wrapper (stage-D convention)
  local key=$1; shift
  if ! "$@"; then
    local fc="$DIAG/failcount/$key"
    echo x >> "$fc"
    if [ "$(wc -l < "$fc")" -ge $MAX_CELL_STRIKES ]; then
      echo "FATAL: step $key failed $MAX_CELL_STRIKES times -- deterministic failure, halting." >&2
      touch "$DIAG/FATAL"
    fi
    echo "ERROR: step $key FAILED -- exiting (supervisor retries unless FATAL)." >&2
    exit 1
  fi
}

# --- 1) the 12 seed-extension training cells, sequential on GPU 7, resume-safe ---
mapfile -t DIAG_CELLS < <($PY h2h_task2diag_rd.py --list-cells)
if [ "${#DIAG_CELLS[@]}" -ne 12 ]; then
  echo "REFUSE: expected 12 diag cells, got ${#DIAG_CELLS[@]}" >&2
  exit 1
fi
for name in "${DIAG_CELLS[@]}"; do
  out="$DIAG/${name}.json"
  if $PY -c "import sys; from h2h_sweep_runner_rd import is_valid_result; sys.exit(0 if is_valid_result('$out') else 1)" 2>/dev/null; then
    echo "SKIP (already valid): $out"
    continue
  fi
  echo "LAUNCH (gpu=$TASK2DIAG_GPU): $name -> $out"
  run_step "$name" env CUDA_VISIBLE_DEVICES="$TASK2DIAG_GPU" "$PY" h2h_task2diag_rd.py \
      --run-cell "$name" --out "$out" --ckpt-dir "$CKPT_DIR" --gates-dir "$GATES" \
      --margins-token "$RES/MARGINS_FROZEN.token" --device cuda 2>&1 | tee "logs/h2h_task2diag_${name}.log"
  budget_check
done

# --- 2) the K48 stress cell (fresh train + full re-metric in one audited call) ---
run_step k48_cell env CUDA_VISIBLE_DEVICES="$TASK2DIAG_GPU" "$PY" h2h_task2diag_rd.py \
    --run-k48 --out-dir "$DIAG" --ckpt-dir "$CKPT_DIR" --gates-dir "$GATES" \
    --margins-token "$RES/MARGINS_FROZEN.token" --device cuda 2>&1 | tee logs/h2h_task2diag_k48.log
budget_check

# --- 3) verdict-grade re-metric of the 12 cells (sec 1.40 harvest protocol) ---
run_step remetric env CUDA_VISIBLE_DEVICES="$TASK2DIAG_GPU" "$PY" h2h_task2diag_rd.py \
    --remetric --ckpt-dir "$CKPT_DIR" --remetric-dir "$DIAG/remetric" \
    --gates-dir "$GATES" --margins-token "$RES/MARGINS_FROZEN.token" \
    --device cuda 2>&1 | tee logs/h2h_task2diag_remetric.log
budget_check

# --- 4) pooled analysis (pure JSON math; writes TASK2DIAG_VERDICT.json) ---
run_step analyze "$PY" h2h_task2diag_rd.py --analyze --out-dir "$DIAG" \
    --remetric-dir "$DIAG/remetric" --sweep-remetric-dir "$RES/sweep_remetric" \
    --round4-dir "$RES/round4" 2>&1 | tee logs/h2h_task2diag_analyze.log

touch "$DIAG/TASK2DIAG_STOP"
echo "TASK2DIAG ROUND COMPLETE (TASK2DIAG_STOP written). Harvest: $DIAG/ (12 training JSONs,"
echo "remetric/, transformer_task1_stress_K48_round4.json, TASK2DIAG_VERDICT.json)."
echo "Conditional sec 1.42 A5 horizon reads (if trigger seeds nonempty) run at harvest dispatch."
