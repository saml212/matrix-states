#!/bin/bash
# h2h_mstar_stage.sh -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.4.2 / sec 1.31.1 / sec 1.40-NEXT(a):
# the AXIS-2 M* PROTOCOL stage. Eval-only (no training): the 90-pass capped-transformer
# fan-out (5 eligible M x 3 horizons x 2 tasks x 3 seeds, acc_A decision metric per the
# sec 1.31.1 re-registration) + the contender horizon references (18 passes) + two disclosed
# extra reference sets (uncapped-transformer horizon refs for the forced-locality question;
# the floor-excluded M=1 descriptive row, R3-F6's eval-overhead line) -- all on the sweep's
# _r4.pt task1/task2 checkpoints (sec 1.38 pre-flight item 2's suffix fix applied).
#
# GPU: single GPU, default 1 (GPU 0 = stage2_calib live; 2-7 = fixscale wave; 7 additionally
# pool-reserved -- never touch any of them).
#
# Budget: 3.0 GPU-h hard ceiling (expected well under 1 -- forward-only passes), enforced
# mechanically per supervisor pass (stage-D's own $SECONDS convention).
#
# Launch (deploy agent, exactly once):
#   tmux new-session -d -s h2h_mstar \
#     "while [ ! -f results/h2h_rung1/MSTAR_STOP ] && [ ! -f results/h2h_rung1/MSTAR_FATAL ]; do \
#        bash h2h_mstar_stage.sh >> logs/h2h_mstar_supervisor.log 2>&1; sleep 15; done"
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
RES=results/h2h_rung1
GATES=$RES/gates
mkdir -p "$RES/fanout" logs

MSTAR_GPU=${MSTAR_GPU:-1}
if [ "$MSTAR_GPU" != "1" ]; then
  echo "NOTE: MSTAR_GPU=$MSTAR_GPU (non-default); GPUs 0/2-7 are owned by other campaigns." >&2
fi
if [ "$MSTAR_GPU" = "7" ]; then
  echo "REFUSE: GPU 7 is pool-reserved (standing deploy directive)." >&2
  exit 1
fi
BUDGET_CEILING_GPU_H=3.0

[ -f "$RES/MSTAR_STOP" ] && { echo "MSTAR_STOP present -- M* stage complete."; exit 0; }
[ -f "$RES/MSTAR_FATAL" ] && { echo "MSTAR_FATAL present -- refusing (coordinator review)." >&2; exit 1; }

# --- sec 1.7 gate 5 tokens (files + env), verbatim from stage D ---
for tok in GATE6_MATCH_GATE_PASSED.token GATE7_PROBE_CAPACITY_NULL_PASSED.token BOX_SMOKE_ITEMS_1_4_PASSED.token; do
  if [ ! -s "$GATES/$tok" ]; then
    echo "REFUSE: $GATES/$tok missing." >&2
    touch "$RES/MSTAR_FATAL"
    exit 1
  fi
done
export HEADTOHEAD_PI_SIGNOFF=1          # sec 1.20 addendum DEPLOY AUTHORIZED (unchanged)
export HEADTOHEAD_MATCH_GATE_SIGNOFF=1  # attested by the gate-6/7 token files verified above
export H2H_DIAL_ROUND=4                 # filename versioning ONLY (the sweep saved _r4.pt)
$PY h2h_cell_train_rd.py --token-probe --gates-dir "$GATES" 2>&1 | tee logs/h2h_mstar_00_token_probe.log

# --- MARGIN FREEZE blind check (sec 13.13 pattern; negative test EXECUTED every launch) ---
if [ ! -s "$RES/MARGINS_FROZEN.token" ]; then
  echo "REFUSE: $RES/MARGINS_FROZEN.token missing -- axis 2 is NOT released." >&2
  exit 1
fi
$PY - "$RES/MARGINS_FROZEN.token" <<'PYEOF' 2>&1 | tee logs/h2h_mstar_01_blind_check.log
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
print(f"blind check PASS: pinned_at={tok['pinned_at']} strictly precedes launch={now}; "
      f"negative test exercised")
PYEOF

budget_check() {
  local projected_gpu_h
  projected_gpu_h=$($PY -c "print($SECONDS / 3600.0)")   # single GPU
  echo "[budget] elapsed=${SECONDS}s projected_gpu_h=${projected_gpu_h} ceiling=${BUDGET_CEILING_GPU_H}"
  if $PY -c "import sys; sys.exit(0 if $projected_gpu_h > $BUDGET_CEILING_GPU_H else 1)"; then
    echo "ABORT: projected GPU-h exceeds the M* stage ceiling -- halting mechanically." >&2
    touch "$RES/MSTAR_FATAL"
    exit 1
  fi
}

# --- CPU-stub smoke gates for the two fixed modules (sec 1.38 pre-flight items 1/3),
# run ON THE BOX at every launch -- cheap, and proves the deployed copies are the fixed ones ---
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY h2h_msweep_fanout_rd.py --smoke \
    2>&1 | tee logs/h2h_mstar_02_fanout_smoke.log
$PY h2h_mstar_analysis_rd.py --smoke 2>&1 | tee logs/h2h_mstar_03_analysis_smoke.log

# --- checkpoint maps: the sweep's _r4-suffixed task1/task2 cells (pre-flight item 2) ---
$PY - <<'PYEOF' 2>&1 | tee logs/h2h_mstar_04_ckpt_maps.log
import json, os
ckpt_dir = "/data/h2h_rung1_ckpts"
suffix = f"_r{os.environ.get('H2H_DIAL_ROUND', '4')}"
tr, ref = {}, {}
for task in ("task1_sweep", "task2_sweep"):
    for s in range(3):
        tr[f"{task}|{s}"] = os.path.join(ckpt_dir, f"h2h_transformer_{task}_s{s}{suffix}.pt")
        ref[f"{task}|{s}"] = os.path.join(ckpt_dir, f"h2h_contender_{task}_s{s}{suffix}.pt")
for m, path in (list(tr.items()) + list(ref.items())):
    assert os.path.exists(path), f"missing sweep checkpoint {path}"
json.dump(tr, open("results/h2h_rung1/fanout/ckpt_map_transformer.json", "w"), indent=2)
json.dump(ref, open("results/h2h_rung1/fanout/ckpt_map_contender.json", "w"), indent=2)
print(f"checkpoint maps written (suffix {suffix})")
PYEOF

# --- the 90-pass fan-out (resume-safe: is_valid_result requires acc_A, so any rf-era
# leftover re-runs rather than being skipped) ---
CUDA_VISIBLE_DEVICES="$MSTAR_GPU" $PY h2h_cell_train_rd.py --fanout-all \
    --ckpt-map "$RES/fanout/ckpt_map_transformer.json" --out-dir "$RES/fanout" \
    --gates-dir "$GATES" --device cuda 2>&1 | tee logs/h2h_mstar_40_fanout.log
budget_check

# --- contender horizon references (axis 2's comparison side, 18 passes) ---
if [ ! -s "$RES/fanout/contender_horizon_refs.json" ]; then
  CUDA_VISIBLE_DEVICES="$MSTAR_GPU" $PY h2h_cell_train_rd.py --horizon-ref \
      --ckpt-map "$RES/fanout/ckpt_map_contender.json" --out "$RES/fanout/contender_horizon_refs.json" \
      --gates-dir "$GATES" --device cuda 2>&1 | tee logs/h2h_mstar_41_contender_refs.log
fi
budget_check

# --- DISCLOSED EXTRAS (cheap, non-decision): uncapped transformer at the same horizons
# (the forced-locality question under the degenerate-baseline clause) + the M=1 descriptive
# row (R3-F6's eval-overhead line; floor-excluded, never M*-eligible) ---
if [ ! -s "$RES/fanout/transformer_uncapped_horizon_refs.json" ]; then
  CUDA_VISIBLE_DEVICES="$MSTAR_GPU" $PY h2h_cell_train_rd.py --horizon-ref \
      --ckpt-map "$RES/fanout/ckpt_map_transformer.json" \
      --out "$RES/fanout/transformer_uncapped_horizon_refs.json" \
      --gates-dir "$GATES" --device cuda 2>&1 | tee logs/h2h_mstar_42_tfm_uncapped_refs.log
fi
if [ ! -s "$RES/fanout/transformer_m1_descriptive.json" ]; then
  CUDA_VISIBLE_DEVICES="$MSTAR_GPU" $PY h2h_cell_train_rd.py --horizon-ref \
      --ckpt-map "$RES/fanout/ckpt_map_transformer.json" --capped-m 1 \
      --out "$RES/fanout/transformer_m1_descriptive.json" \
      --gates-dir "$GATES" --device cuda 2>&1 | tee logs/h2h_mstar_43_m1_descriptive.log
fi
budget_check

touch "$RES/MSTAR_STOP"
echo "M* STAGE COMPLETE (MSTAR_STOP written). Harvest: $RES/fanout/ -> h2h_mstar_harvest_rd.py"
