#!/bin/bash
# fixscale harvest measurement driver (2026-07-10): the sec 13.7-budgeted eval-only passes.
# 16 shared-forward-pass comparator runs (12 arm_per_token + 4 arm_global_probe final
# checkpoints, each on its own corpus) -> results/fixscale/measure/, then verify-pin x2
# (tamper-evidence re-check, sec 13.10 gate 7), then probe-report x4 (CPU-only assembly).
# Sequential on ONE GPU (CUDA_VISIBLE_DEVICES pinned by caller). set -u only: a single
# failed pass must not kill the remaining passes (standing sweep try/except rule).
set -u
PY=/home/nvidia/tdenv/bin/python3
RD=/home/nvidia/chapter2/deltanet_rd
TRAIN=$RD/results/fixscale/train
MEAS=$RD/results/fixscale/measure
mkdir -p "$MEAS"
cd "$RD"

run_comparator () {
  local cell_id=$1 corpus=$2
  local out="$MEAS/comparator_${cell_id}.json"
  if [ -s "$out" ] && $PY -c "import json,sys; json.load(open('$out'))" 2>/dev/null; then
    echo "[skip] $out exists and parses"
    return 0
  fi
  local ckpt
  ckpt=$($PY -c "import json; print(json.load(open('$TRAIN/${cell_id}.json'))['final_checkpoint_path'])")
  if [ ! -f "$ckpt" ]; then
    echo "[FAIL] checkpoint missing for $cell_id: $ckpt"
    return 1
  fi
  echo "[run ] comparator $cell_id ($corpus) ckpt=$ckpt"
  $PY fixscale_wave.py comparator --checkpoint "$ckpt" --corpus "$corpus" --out "$out" \
    || echo "[FAIL] comparator rc=$? for $cell_id"
}

for scale in 98m 392m; do
  for corpus in openr1-mix-ext wikitext-mix-ext; do
    for seed in 0 1 2; do
      run_comparator "fixscale_train_arm_per_token_${scale}_${corpus}_s${seed}" "$corpus"
    done
    run_comparator "fixscale_train_arm_global_probe_${scale}_${corpus}_s0" "$corpus"
  done
done

echo "=== verify-pin (tamper-evidence, both scales) ==="
for scale in 98m 392m; do
  $PY fixscale_wave.py verify-pin --scale "$scale" \
    && echo "[ok  ] verify-pin $scale" || echo "[FAIL] verify-pin $scale rc=$?"
done

echo "=== probe-report x4 ==="
for scale in 98m 392m; do
  for corpus in openr1-mix-ext wikitext-mix-ext; do
    $PY fixscale_wave.py probe-report --scale "$scale" --corpus "$corpus" \
      --out "$MEAS/probe_report_${scale}_${corpus}.json" \
      || echo "[FAIL] probe-report $scale $corpus rc=$?"
  done
done

echo "ALL_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
