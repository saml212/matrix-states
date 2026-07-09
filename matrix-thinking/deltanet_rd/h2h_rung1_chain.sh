#!/bin/bash
# h2h_rung1_chain.sh -- HEAD_TO_HEAD_DEMO_DESIGN.md rung-1 wave chain (deploy
# stage, sec 1.19 items (3)-(6); sec 1.20 addendum "DEPLOY STAGE AUTHORIZED").
#
# Stages (each gated on the previous stage's outputs' VALIDITY, resume-safe):
#   -1  CPU self-tests (forced stub) + grammar_rd injectivity-guard teeth re-run
#    0  gates-6/7 + box-smoke token files VERIFIED PRESENT (written by
#       h2h_box_smoke_driver.py, run by the deploy agent BEFORE launch), then
#       sec-1.7-gate-5 env tokens exported + the negative-test triple run
#    A  sec 1.7 gate 2: per arch x task timing pilots (9, parallel on GPUs 0-6)
#       + mechanical share-of-ceiling gate (h2h_cell_train_rd --gate-pilots)
#    B  sec 1.7 gate 1: the 14-cell 3-arm calibration manifest (13 launchable;
#       contender/task3 is reused from FROZEN_BIAS rung-1, sec 1.6 item E),
#       parallel on GPUs 0-6, then the band check (h2h_calibration_bands_rd --
#       ANY failed band hard-aborts, sanity bands included)
#   B2  R3-F4 M-sweep timing pilot (checkpoints-resident; needs a trained
#       transformer checkpoint, hence ordered AFTER calibration)
#    C  CALIBRATION_COMPLETE.json written, then IDLE-WAIT (60 s poll, never
#       exit) on MARGINS_FROZEN.token -- written ONLY by the coordinator
#    D  (post-token) 27-cell sweep (GPUs 0-6) -> 90-pass M-sweep fan-out ->
#       contender horizon references -> STOP
#
# GPU 7 IS NEVER USED (pool-reserved, deploy directive).
#
# Failure discipline: transient failures exit 1 WITHOUT the FATAL marker (the
# tmux supervisor retries; resume-safety makes retries cheap); gate-class
# failures (missing gate tokens, pilot-gate abort, band failure, budget
# ceiling, 3 strikes on the same cell) touch results/h2h_rung1/FATAL first --
# the supervisor loop refuses to restart past it. NEVER pkill by pattern
# anywhere in this file (house hard rule) -- per-cell PIDs are wait'ed only.
#
# Launch (deploy agent, exactly once):
#   tmux new-session -d -s h2h_rung1 \
#     "while [ ! -f results/h2h_rung1/STOP ] && [ ! -f results/h2h_rung1/FATAL ]; do \
#        bash h2h_rung1_chain.sh >> logs/h2h_rung1_supervisor.log 2>&1; sleep 15; done"
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
RES=results/h2h_rung1
GATES=$RES/gates
CKPT_DIR=/data/h2h_rung1_ckpts
mkdir -p "$RES" "$GATES" "$RES/calib" "$RES/pilots" "$RES/sweep" "$RES/fanout" "$RES/failcount" logs "$CKPT_DIR"

H2H_GPUS=${H2H_GPUS:-"0,1,2,3,4,5,6"}      # GPU 7 pool-reserved -- never in this list
if echo ",$H2H_GPUS," | grep -q ",7,"; then
  echo "REFUSE: H2H_GPUS contains GPU 7 (pool-reserved by the deploy directive)." >&2
  exit 1
fi
IFS=',' read -ra GPU_LIST <<< "$H2H_GPUS"
N_GPUS=${#GPU_LIST[@]}
BUDGET_CEILING_GPU_H=125.672                # sec 1.6 enforced circuit-breaker bracket (R3-F6)
MAX_CELL_STRIKES=3

[ -f "$RES/STOP" ] && { echo "STOP present -- chain complete/halted."; exit 0; }
[ -f "$RES/FATAL" ] && { echo "FATAL present -- refusing to run (coordinator review required)." >&2; exit 1; }

budget_check() {
  local projected_gpu_h
  projected_gpu_h=$($PY -c "print(($SECONDS * $N_GPUS) / 3600.0)")
  echo "[budget] elapsed=${SECONDS}s n_gpus=$N_GPUS projected_gpu_h=${projected_gpu_h} ceiling=${BUDGET_CEILING_GPU_H}"
  if $PY -c "import sys; sys.exit(0 if $projected_gpu_h > $BUDGET_CEILING_GPU_H else 1)"; then
    echo "ABORT: projected GPU-h exceeds the registered sec 1.6 ceiling -- halting mechanically." >&2
    touch "$RES/FATAL"
    exit 1
  fi
}

# N-way parallel cell launcher over GPU_LIST (generalizes phase2b_chain.sh's
# run_cells_2way: same resume-safety via output-JSON validity, same per-cell GPU
# pinning, fail-fast per wave, plus a 3-strike FATAL counter per cell).
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

  local setname
  setname=$([ "$cellset" = "calib" ] && echo calibration || echo sweep)
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
      ( CUDA_VISIBLE_DEVICES="$gpu" $PY h2h_cell_train_rd.py --run-cell "$name" --set "$setname" \
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
    i=$((i + N_GPUS))
  done
}

# ---------------------------------------------------------------------------
# Stage -1 -- CPU self-tests, forced stub (gate 3's negative tests included:
# injectivity-guard teeth, band-checker teeth, token-refusal triple inside the
# cell trainer's own selftest).
# ---------------------------------------------------------------------------
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY h2h_box_smoke_checklist.py --list \
    2>&1 | tee logs/h2h_00_checklist.log
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY h2h_calibration_wrappers_rd.py --smoke \
    2>&1 | tee logs/h2h_01_wrappers.log
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY h2h_sweep_runner_rd.py --smoke \
    2>&1 | tee logs/h2h_02_sweep_runner.log
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY h2h_msweep_fanout_rd.py --smoke \
    2>&1 | tee logs/h2h_03_fanout.log
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY h2h_calibration_bands_rd.py --selftest \
    2>&1 | tee logs/h2h_04_bands_selftest.log
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY h2h_cell_train_rd.py --selftest \
    2>&1 | tee logs/h2h_05_cell_train_selftest.log
# sec 1.7 gate 3: the injectivity guard's own negative test, re-run for THIS wave.
CUDA_VISIBLE_DEVICES= $PY -c "import grammar_rd; grammar_rd._test_injectivity_guard_detects_merge(); print('injectivity guard teeth: PASS')" \
    2>&1 | tee logs/h2h_06_injectivity_teeth.log

# ---------------------------------------------------------------------------
# Stage 0 -- gates 6+7 / box-smoke tokens VERIFIED, then gate-5 env tokens +
# negative-test triple (the trainer refuses when ANY of the three legs is out).
# ---------------------------------------------------------------------------
for tok in GATE6_MATCH_GATE_PASSED.token GATE7_PROBE_CAPACITY_NULL_PASSED.token BOX_SMOKE_ITEMS_1_4_PASSED.token; do
  if [ ! -s "$GATES/$tok" ]; then
    echo "REFUSE: $GATES/$tok missing -- run h2h_box_smoke_driver.py first (sec 1.20 checklist)." >&2
    touch "$RES/FATAL"
    exit 1
  fi
done
# Negative-test triple (each requirement independently removed must REFUSE, exit 3):
env -u HEADTOHEAD_PI_SIGNOFF HEADTOHEAD_MATCH_GATE_SIGNOFF=1 \
    $PY h2h_cell_train_rd.py --token-probe --gates-dir "$GATES" && { echo "NEG-TEST 1 FAILED TO FAIL" >&2; touch "$RES/FATAL"; exit 1; } || true
env HEADTOHEAD_PI_SIGNOFF=1 -u HEADTOHEAD_MATCH_GATE_SIGNOFF \
    $PY h2h_cell_train_rd.py --token-probe --gates-dir "$GATES" && { echo "NEG-TEST 2 FAILED TO FAIL" >&2; touch "$RES/FATAL"; exit 1; } || true
env HEADTOHEAD_PI_SIGNOFF=1 HEADTOHEAD_MATCH_GATE_SIGNOFF=1 \
    $PY h2h_cell_train_rd.py --token-probe --gates-dir /nonexistent_gates_dir && { echo "NEG-TEST 3 FAILED TO FAIL" >&2; touch "$RES/FATAL"; exit 1; } || true
export HEADTOHEAD_PI_SIGNOFF=1          # launch authorization: sec 1.20 addendum DEPLOY AUTHORIZED
export HEADTOHEAD_MATCH_GATE_SIGNOFF=1  # attested by the gate-6/7 token files verified above
$PY h2h_cell_train_rd.py --token-probe --gates-dir "$GATES" 2>&1 | tee logs/h2h_07_token_probe.log
budget_check

# ---------------------------------------------------------------------------
# Stage A -- gate-2 timing pilots (9 arch x task pairs, parallel over GPUs),
# then the mechanical share-of-ceiling gate.
# ---------------------------------------------------------------------------
pilot_jobs=()
for arch in contender ablation transformer; do
  for task in task1 task2 task3; do
    out="$RES/pilots/pilot_${arch}_${task}.json"
    if $PY -c "
import json, sys
try:
    d = json.load(open('$out'))
    sys.exit(0 if 'projected_gpu_h_per_full_cell' in d else 1)
except Exception:
    sys.exit(1)" 2>/dev/null; then
      echo "SKIP pilot (already valid): $out"
    else
      pilot_jobs+=("${arch}:${task}")
    fi
  done
done
i=0
while [ $i -lt ${#pilot_jobs[@]} ]; do
  pids=()
  for slot in $(seq 0 $((N_GPUS - 1))); do
    idx=$((i + slot)); [ $idx -ge ${#pilot_jobs[@]} ] && break
    IFS=':' read -r arch task <<< "${pilot_jobs[$idx]}"
    echo "PILOT LAUNCH (gpu=${GPU_LIST[$slot]}): $arch x $task"
    ( CUDA_VISIBLE_DEVICES="${GPU_LIST[$slot]}" $PY h2h_cell_train_rd.py --pilot-pair "$arch" "$task" \
        --out "$RES/pilots/pilot_${arch}_${task}.json" --gates-dir "$GATES" --device cuda \
        2>&1 | tee "logs/h2h_pilot_${arch}_${task}.log" ) &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || { echo "pilot failed -- exiting (supervisor retries)" >&2; exit 1; }; done
  i=$((i + N_GPUS))
done
if ! $PY h2h_cell_train_rd.py --gate-pilots --pilots-dir "$RES/pilots" --out "$RES/pilots/PILOTS_GATE.json" \
    2>&1 | tee logs/h2h_10_pilot_gate.log; then
  echo "PILOT GATE ABORT (sec 1.7 gate 2) -- halting mechanically." >&2
  touch "$RES/FATAL"
  exit 1
fi
budget_check

# ---------------------------------------------------------------------------
# Stage B -- gate-1 calibration: 13 launchable cells, then the band check.
# ---------------------------------------------------------------------------
mapfile -t CALIB_CELLS < <($PY h2h_cell_train_rd.py --list-cells calibration)
if [ "${#CALIB_CELLS[@]}" -ne 13 ]; then
  echo "REFUSE: expected 13 launchable calibration cells, got ${#CALIB_CELLS[@]} (list-cells failed?)" >&2
  exit 1
fi
run_cells_par calib "${CALIB_CELLS[@]}"
budget_check
if ! $PY h2h_calibration_bands_rd.py --check-dir "$RES/calib" 2>&1 | tee logs/h2h_20_band_check.log; then
  echo "BAND CHECK FAILED -- hard abort of the 27-cell sweep (sec 1.7 gate 1)." >&2
  touch "$RES/FATAL"
  exit 1
fi

# ---------------------------------------------------------------------------
# Stage B2 -- R3-F4 M-sweep timing pilot (transformer task1 PRIMARY calibration
# checkpoint, held resident across both pilot passes).
# ---------------------------------------------------------------------------
# _auxrev2: the aux_weight=2.0 re-run's checkpoint (sec 1.3.1.3 calibration revision;
# the aux_weight=0.1 run's checkpoint keeps its unsuffixed name, archived).
MSWEEP_CKPT="$CKPT_DIR/h2h_calib_transformer_task1_calib_primary_K32_auxrev2.pt"
if [ ! -s "$MSWEEP_CKPT" ]; then
  echo "REFUSE: expected calibration checkpoint $MSWEEP_CKPT missing." >&2
  exit 1
fi
if ! $PY -c "
import json, sys
try:
    d = json.load(open('$RES/pilots/msweep_pilot.json'))
    sys.exit(0 if d.get('fanout_gate', {}).get('ok') else 1)
except Exception:
    sys.exit(1)" 2>/dev/null; then
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" $PY h2h_cell_train_rd.py --msweep-pilot \
      --ckpt "$MSWEEP_CKPT" --out "$RES/pilots/msweep_pilot.json" --gates-dir "$GATES" \
      --headroom-gpu-h 100.0 --device cuda 2>&1 | tee logs/h2h_21_msweep_pilot.log \
    || { echo "M-SWEEP PILOT GATE ABORT (R3-F4)." >&2; touch "$RES/FATAL"; exit 1; }
else
  echo "SKIP msweep pilot (already valid)"
fi
budget_check

# ---------------------------------------------------------------------------
# Stage C -- CALIBRATION_COMPLETE report, then BLOCK on the coordinator's
# margin freeze. This loop never exits on its own (deploy directive: the
# 27-cell sweep is NOT released until the coordinator records the freeze).
# ---------------------------------------------------------------------------
$PY h2h_cell_train_rd.py --write-calibration-report --res-dir "$RES" \
    --out "$RES/CALIBRATION_COMPLETE.json" 2>&1 | tee logs/h2h_30_calibration_report.log
echo "STAGE C COMPLETE: $RES/CALIBRATION_COMPLETE.json written."
echo "BLOCKING on $RES/MARGINS_FROZEN.token (coordinator-only write; polling every 60 s)..."
n_polls=0
while [ ! -f "$RES/MARGINS_FROZEN.token" ]; do
  [ -f "$RES/STOP" ] && { echo "STOP while blocked -- exiting cleanly."; exit 0; }
  sleep 60
  n_polls=$((n_polls + 1))
  if [ $((n_polls % 30)) -eq 0 ]; then
    echo "[margins-freeze wait] still blocked after $((n_polls / 60))h$((n_polls % 60))m ($(date -u +%FT%TZ))"
  fi
done
echo "MARGINS_FROZEN.token found -- sweep stage released."

# ---------------------------------------------------------------------------
# Stage D -- 27-cell sweep, then the 90-pass fan-out + contender references.
# ---------------------------------------------------------------------------
mapfile -t SWEEP_CELLS < <($PY h2h_cell_train_rd.py --list-cells sweep)
if [ "${#SWEEP_CELLS[@]}" -ne 27 ]; then
  echo "REFUSE: expected 27 sweep cells, got ${#SWEEP_CELLS[@]} (list-cells failed?)" >&2
  exit 1
fi
run_cells_par sweep "${SWEEP_CELLS[@]}"
budget_check

# checkpoint maps for the fan-out (b-primary transformer) + references (contender)
$PY - <<'PYEOF'
import json, os
ckpt_dir = "/data/h2h_rung1_ckpts"
tr, ref = {}, {}
for task in ("task1_sweep", "task2_sweep"):
    for s in range(3):
        tr[f"{task}|{s}"] = os.path.join(ckpt_dir, f"h2h_transformer_{task}_s{s}.pt")
        ref[f"{task}|{s}"] = os.path.join(ckpt_dir, f"h2h_contender_{task}_s{s}.pt")
for m, path in (list(tr.items()) + list(ref.items())):
    assert os.path.exists(path), f"missing sweep checkpoint {path}"
json.dump(tr, open("results/h2h_rung1/fanout/ckpt_map_transformer.json", "w"), indent=2)
json.dump(ref, open("results/h2h_rung1/fanout/ckpt_map_contender.json", "w"), indent=2)
print("checkpoint maps written")
PYEOF

CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" $PY h2h_cell_train_rd.py --fanout-all \
    --ckpt-map "$RES/fanout/ckpt_map_transformer.json" --out-dir "$RES/fanout" \
    --gates-dir "$GATES" --device cuda 2>&1 | tee logs/h2h_40_fanout.log
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" $PY h2h_cell_train_rd.py --horizon-ref \
    --ckpt-map "$RES/fanout/ckpt_map_contender.json" --out "$RES/fanout/contender_horizon_refs.json" \
    --gates-dir "$GATES" --device cuda 2>&1 | tee logs/h2h_41_horizon_refs.log
budget_check

touch "$RES/STOP"
echo "H2H RUNG-1 CHAIN COMPLETE (STOP written). Harvest: $RES/CALIBRATION_COMPLETE.json, $RES/sweep/, $RES/fanout/"
