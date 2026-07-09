#!/bin/bash
# h2h_sweep_stage_d.sh -- HEAD_TO_HEAD_DEMO_DESIGN.md Stage D, 27-cell sweep ONLY
# (sec 1.37 disposition -> sec 1.38 MARGIN FREEZE record). Extracted from
# h2h_rung1_chain.sh's own Stage D because the full chain cannot be resumed as-is:
# its Stage B would RETRAIN the two calibration cells whose round-3 results never
# landed in calib/ (transformer_task2 was retrained in ROUND 4 instead, sec 1.36-1.37,
# and its dial check DialExhausted'ed -- re-running it would re-fire the exhausted
# dial), and its Stage-B band checker still enforces the DEAD rf>0 conjunct
# (sec 1.31 dead-clause item 4). The ladder/bands review that replaces Stage B's
# check ran coordinator-side on the round-4 raws (sec 1.38; archived at
# experiment-runs/2026-07-09_h2h_sweep_launch/ladder_bands_review.json): ALL
# LAUNCH-BLOCKING BANDS PASS.
#
# WHAT THIS RUNS: the 27 training cells (3 archs x 3 tasks x 3 seeds) ONLY.
# The 90-pass M-sweep fan-out + contender horizon references (the M* protocol,
# axis 2) run AFTER the sweep per sec 1.31.1 -- next dispatch, NOT here (the per-M
# gap statistic was re-registered to acc_A and the fan-out module must be
# re-verified against that before it runs; also the chain's Stage-D ckpt_map block
# expects UNSUFFIXED checkpoint names, stale vs sec 1.31.4 item 4's _r{round}
# versioning -- fix at the M* dispatch).
#
# GPUs: 0,1 ONLY by default (fixscale wave post_pin cells own 2-7 -- never touch;
# GPU 7 pool-reserved, standing directive). To expand into freed GPUs later:
# tmux kill-session -t h2h_sweep, then relaunch with H2H_GPUS="0,1,2,..." --
# resume-safety (is_valid_result) skips completed cells, nothing is lost.
#
# Budget: 13.25 GPU-h ceiling (sec 1.26a re-priced sweep upper), enforced
# mechanically per supervisor pass (same $SECONDS convention as the chain).
# Failure discipline: identical to the chain (transient -> exit 1, supervisor
# retries; 3 strikes on one cell -> FATAL; the supervisor refuses past FATAL).
#
# Launch (deploy agent, exactly once):
#   tmux new-session -d -s h2h_sweep \
#     "while [ ! -f results/h2h_rung1/SWEEP_STOP ] && [ ! -f results/h2h_rung1/FATAL ]; do \
#        bash h2h_sweep_stage_d.sh >> logs/h2h_sweep_supervisor.log 2>&1; sleep 15; done"
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
RES=results/h2h_rung1
GATES=$RES/gates
CKPT_DIR=/data/h2h_rung1_ckpts
mkdir -p "$RES/sweep" "$RES/failcount" logs "$CKPT_DIR"

H2H_GPUS=${H2H_GPUS:-"0,1"}
if echo ",$H2H_GPUS," | grep -q ",7,"; then
  echo "REFUSE: H2H_GPUS contains GPU 7 (pool-reserved by the deploy directive)." >&2
  exit 1
fi
IFS=',' read -ra GPU_LIST <<< "$H2H_GPUS"
N_GPUS=${#GPU_LIST[@]}
BUDGET_CEILING_GPU_H=13.25                  # sec 1.26a re-priced sweep upper (27 training cells)
MAX_CELL_STRIKES=3

[ -f "$RES/SWEEP_STOP" ] && { echo "SWEEP_STOP present -- sweep complete."; exit 0; }
[ -f "$RES/FATAL" ] && { echo "FATAL present -- refusing to run (coordinator review required)." >&2; exit 1; }

# --- sec 1.7 gate 5 tokens (files + env) ---
for tok in GATE6_MATCH_GATE_PASSED.token GATE7_PROBE_CAPACITY_NULL_PASSED.token BOX_SMOKE_ITEMS_1_4_PASSED.token; do
  if [ ! -s "$GATES/$tok" ]; then
    echo "REFUSE: $GATES/$tok missing." >&2
    touch "$RES/FATAL"
    exit 1
  fi
done
export HEADTOHEAD_PI_SIGNOFF=1          # sec 1.20 addendum DEPLOY AUTHORIZED (unchanged)
export HEADTOHEAD_MATCH_GATE_SIGNOFF=1  # attested by the gate-6/7 token files verified above
# Round label for checkpoint filename versioning (_r4, sec 1.31.4 item 4), pinned explicitly
# per the round-4 precedent (sec 1.34 note v). Sweep cells NEVER evaluate the dial (the
# role=='sweep' structural guard, AUD2-F4 fix) -- this cannot re-fire the task2 lane's
# DIAL_EXHAUSTED, it only versions filenames.
export H2H_DIAL_ROUND=4
$PY h2h_cell_train_rd.py --token-probe --gates-dir "$GATES" 2>&1 | tee logs/h2h_sweep_00_token_probe.log

# --- MARGIN FREEZE blind check (sec 13.13-verified pattern): the token must exist
# (require_margins_frozen also enforces per-cell, code-level) AND its pinned_at must
# STRICTLY PRECEDE this launch; the negative test is EXECUTED every launch (has-teeth rule). ---
if [ ! -s "$RES/MARGINS_FROZEN.token" ]; then
  echo "REFUSE: $RES/MARGINS_FROZEN.token missing -- the sweep is NOT released." >&2
  exit 1
fi
$PY - "$RES/MARGINS_FROZEN.token" <<'PYEOF' 2>&1 | tee logs/h2h_sweep_01_blind_check.log
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
print(f"blind check PASS: pinned_at={tok['pinned_at']} ({tok.get('pinned_at_utc','?')}) "
      f"strictly precedes launch={now}; negative test exercised")
PYEOF

budget_check() {
  local projected_gpu_h
  projected_gpu_h=$($PY -c "print(($SECONDS * $N_GPUS) / 3600.0)")
  echo "[budget] elapsed=${SECONDS}s n_gpus=$N_GPUS projected_gpu_h=${projected_gpu_h} ceiling=${BUDGET_CEILING_GPU_H}"
  if $PY -c "import sys; sys.exit(0 if $projected_gpu_h > $BUDGET_CEILING_GPU_H else 1)"; then
    echo "ABORT: projected GPU-h exceeds the sec 1.26a sweep ceiling -- halting mechanically." >&2
    touch "$RES/FATAL"
    exit 1
  fi
}

# N-way parallel cell launcher over GPU_LIST -- verbatim from h2h_rung1_chain.sh's
# run_cells_par (same resume-safety, per-cell GPU pinning, fail-fast per wave,
# 3-strike FATAL counter). NEVER pkill by pattern (house hard rule).
run_cells_par() {
  local cellset=$1; shift
  local cells=("$@")
  local pending=()
  local name out
  for name in "${cells[@]}"; do
    out="$RES/${cellset}/${name}.json"
    if $PY -c "import sys; from h2h_sweep_runner_rd import is_valid_result; sys.exit(0 if is_valid_result('$out') else 1)" 2>/dev/null; then
      echo "SKIP (already valid): $out"
      continue
    fi
    pending+=("$name")
  done

  local i=0
  while [ $i -lt ${#pending[@]} ]; do
    local pids=() names=()
    local slot idx gpu log
    for slot in $(seq 0 $((N_GPUS - 1))); do
      idx=$((i + slot))
      [ $idx -ge ${#pending[@]} ] && break
      name="${pending[$idx]}"
      out="$RES/${cellset}/${name}.json"
      log="logs/h2h_${cellset}_${name}.log"
      gpu="${GPU_LIST[$slot]}"
      echo "LAUNCH (gpu=$gpu): $name -> $out"
      ( CUDA_VISIBLE_DEVICES="$gpu" $PY h2h_cell_train_rd.py --run-cell "$name" --set sweep \
            --out "$out" --ckpt-dir "$CKPT_DIR" --gates-dir "$GATES" \
            --margins-token "$RES/MARGINS_FROZEN.token" --device cuda 2>&1 | tee "$log" ) &
      pids+=($!); names+=("$name")
    done
    local fail=0 k=0 pid
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        fail=1
        local fc="$RES/failcount/${names[$k]}"
        echo x >> "$fc"
        if [ "$(wc -l < "$fc")" -ge $MAX_CELL_STRIKES ]; then
          echo "FATAL: cell ${names[$k]} failed $MAX_CELL_STRIKES times -- deterministic failure, halting." >&2
          touch "$RES/FATAL"
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
}

mapfile -t SWEEP_CELLS < <($PY h2h_cell_train_rd.py --list-cells sweep)
if [ "${#SWEEP_CELLS[@]}" -ne 27 ]; then
  echo "REFUSE: expected 27 sweep cells, got ${#SWEEP_CELLS[@]} (list-cells failed?)" >&2
  exit 1
fi
run_cells_par sweep "${SWEEP_CELLS[@]}"
budget_check

touch "$RES/SWEEP_STOP"
echo "27-CELL SWEEP COMPLETE (SWEEP_STOP written). Harvest: $RES/sweep/ (27 JSONs, _r4 ckpts in $CKPT_DIR)."
echo "NEXT (sec 1.31.1): the M* protocol (90-pass fan-out + contender horizon refs) -- own dispatch."
