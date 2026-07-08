#!/bin/bash
# phase2_chain.sh -- REASONING_LINK_DESIGN.md sec 16.2 (Rev 5, CLEARED-FOR-BUILD) Phase-2 task
# familiarization: orchestrates Stage -1 self-tests -> OFF-arm-first sequencing -> BANDS_PINNED ->
# per-checkpoint Stage-0.5 launch gate -> the 12 per_token/global cells -> trajectory analysis ->
# hexachotomy classification -> a final summary JSON. 18 core cells total (3 arms x 2 corpora x
# 3 seeds), steps=5000, trajectory checkpoints {250,500,1000,2500,5000}.
#
# BUDGET (sec 16.2.3, Rev 3): raw combined total (training + Stage-0.5 gate null-bands)
# ~=0.30-1.21 GPU-h; debug-tax-inclusive registered bracket ~=1.48-12.06 GPU-h. This script tracks
# wall-clock via bash's own SECONDS builtin and ABORTS (real process-boundary halt, not a narrated
# warning) if the projected GPU-h -- wall-clock seconds x N_GPUS / 3600 -- exceeds the registered
# UPPER bound (12.06 GPU-h) at any checkpoint between phases. GPUs 0-1 (2 GPUs).
#
# GATE ENFORCEMENT (sec 16.5 Constraint 1's own "every gate needs a chain enforcement branch"
# requirement, mirroring reasoning_link_phase1b_stage0_chain.sh's own PROVEN pattern -- an `if`
# guard reading a REAL subprocess exit code, never `set -e`'s narrated-only default):
#   1. BANDS_PINNED blind-sequencing: pin BEFORE per_token/global launch (mechanical, by
#      construction -- this script's own step ordering IS the enforcement; phase2_bands_pinned.
#      assert_blind_not_broken_phase2 is the mechanical proof, exercised at Stage -1).
#   2. Per-checkpoint Stage-0.5 launch gate: after the OFF arm's 6 cells complete, the TERMINAL
#      (5,000-step) checkpoint's own gate (phase2_gate_enforce.py) is checked for AT LEAST ONE
#      (corpus,seed) OFF-arm cell -- if NONE clear it, this script HALTS before launching ANY
#      per_token/global cell (mirrors reasoning_link_phase1b_stage0_chain.sh's own "at least one
#      candidate clears" convention). The FULL per-checkpoint gate (all 5 checkpoints, every
#      corpus/seed) is separately fed into phase2_trajectory_analysis.py's own hexachotomy
#      classification later -- a checkpoint failing there routes that trajectory toward
#      UNRESOLVED-GATE rather than halting the whole chain (sec 16.2.1's own per-checkpoint,
#      not per-run, exclusion granularity).
#
# RESUME-SAFETY (house convention, mech_stage2_chain.sh's own "[LEARN] from Stage 1: run 1 died on
# an unshipped dependency" + this design's own optimizer_state_dict requirement): every cell launch
# below is skipped if its own --out JSON already exists AND validates as complete (steps_completed
# == steps); phase2_familiarization_train.py's own --resume (default ON) additionally resumes a
# PARTIAL cell from its own last trajectory checkpoint (model+optimizer+step) rather than
# restarting at step 1.
#
# Same `set -euo pipefail` discipline every sibling chain in this dir uses.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs results/phase2/ckpts

PHASE2_GPUS=${PHASE2_GPUS:-"0,1"}
N_GPUS=$(echo "$PHASE2_GPUS" | tr ',' '\n' | wc -l | tr -d ' ')
IFS=',' read -ra GPU_LIST <<< "$PHASE2_GPUS"   # MINOR-2 fix: per-cell GPU assignment for the 2-way
                                                 # parallel launcher below (run_cells_2way).
FROZEN_BIAS_CKPT_ROOT=${FROZEN_BIAS_CKPT_ROOT:-/data/deltanet_rd_frozenbias_ckpts}
CKPT_DIR=results/phase2/ckpts
STEPS=5000
CKPT_STEPS="250 500 1000 2500 5000"
CORPORA=(openr1-mix-ext wikitext-mix-ext)
SEEDS=(0 1 2)
# K in {20,32} (Leg A's own committed pair) is NOT a training-cell axis -- it is applied at readout
# time only, inside step 6's phase2_trajectory_analysis.py (its own K_pair=(32,20) default), on
# these SAME 18 training cells' saved checkpoints. No bash-level K loop here (see step 2's own
# comment for the full disclosure).
FROZEN_BIAS_LAMBDA=0.58   # sec 5's registered fixed blend weight (used below, step 5's lam_tag)

# Registered bracket ceiling (sec 16.2.3, Rev 3): ~=1.48-12.06 GPU-h, debug-tax-inclusive. This
# script aborts if PROJECTED GPU-h exceeds the UPPER bound -- never silently runs past it.
BUDGET_CEILING_GPU_H=12.06

budget_check() {
  # Real process-boundary check (not narrated): wall-clock seconds x N_GPUS / 3600, compared
  # against the registered ceiling. Called between major phases below.
  local elapsed_s=$SECONDS
  local projected_gpu_h
  projected_gpu_h=$($PY -c "print(($elapsed_s * $N_GPUS) / 3600.0)")
  echo "[budget] elapsed=${elapsed_s}s  n_gpus=$N_GPUS  projected_gpu_h=${projected_gpu_h}  ceiling=${BUDGET_CEILING_GPU_H}"
  if $PY -c "import sys; sys.exit(0 if $projected_gpu_h > $BUDGET_CEILING_GPU_H else 1)"; then
    echo "ABORT: projected GPU-h (${projected_gpu_h}) exceeds the registered ceiling (${BUDGET_CEILING_GPU_H}, sec 16.2.3) -- halting mechanically." >&2
    touch results/phase2/BUDGET_ABORTED
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# MINOR-2 fix (Phase-2 build-audit round): 2-way parallel cell launcher. As-built (pre-fix), every
# cell ran strictly serially even though the chain reserves BOTH `CUDA_VISIBLE_DEVICES=0,1` and
# prices `N_GPUS=2` in budget_check -- GPU 1 sat idle the entire chain. This function launches cells
# in pairs, one per physical GPU (CUDA_VISIBLE_DEVICES pinned per-subprocess to a SINGLE device, not
# the full $PHASE2_GPUS list), waits on BOTH before moving to the next pair, and aborts the whole
# chain (mirrors every other gate in this script: a real subprocess exit code, never narrated-only)
# if either half of a pair fails. Resume-safety is preserved exactly as before: a cell whose own
# --out JSON already validates as complete (steps_completed==STEPS) is skipped before it ever
# enters the pending queue, so a chain restart never re-launches finished work.
#
# Budget arithmetic stays honest under this schedule (MINOR-2's own "keep the budget arithmetic
# honest" requirement): budget_check's own `elapsed_s * N_GPUS / 3600` formula prices the RESERVED
# capacity (both GPUs held for the script's own $SECONDS wall-clock span) rather than per-cell
# runtime, so it needs no change here -- pre-fix, that formula OVER-priced relative to REALIZED
# utilization (2 GPUs reserved, 1 ever busy); post-fix, realized utilization actually matches what
# was always being priced, and the same wall-clock $SECONDS counter (script-scoped, unaffected by
# how many cells run concurrently inside it) still feeds it unchanged.
# ---------------------------------------------------------------------------
run_cells_2way() {
  # $@: cell descriptors "arm:corpus:seed". Resolves each into its own --init-checkpoint/--out/--log
  # exactly as the pre-fix serial loops did, then launches PENDING (not-already-complete) cells two
  # at a time, one per entry in $GPU_LIST.
  local cells=("$@")
  local pending=()
  local arm corpus seed out
  for c in "${cells[@]}"; do
    IFS=':' read -r arm corpus seed <<< "$c"
    out="results/phase2/${arm}_${corpus}_s${seed}.json"
    if [ -s "$out" ] && $PY -c "import json,sys; d=json.load(open('$out')); sys.exit(0 if d.get('steps_completed')==$STEPS else 1)" 2>/dev/null; then
      echo "SKIP (already complete): $out"
      continue
    fi
    pending+=("$c")
  done

  local i=0
  while [ $i -lt ${#pending[@]} ]; do
    local pids=()
    local slot idx gpu log lam_tag init_ckpt
    for slot in 0 1; do
      idx=$((i + slot))
      [ $idx -ge ${#pending[@]} ] && break
      IFS=':' read -r arm corpus seed <<< "${pending[$idx]}"
      out="results/phase2/${arm}_${corpus}_s${seed}.json"
      log="logs/99_phase2_${arm}_${corpus}_s${seed}.log"
      gpu="${GPU_LIST[$((slot % ${#GPU_LIST[@]}))]}"
      if [ "$arm" = "off" ]; then
        init_ckpt="$FROZEN_BIAS_CKPT_ROOT/frozenbias_lm_off_lam0p00_${corpus}_dm256_ds64_L2_s${seed}/lmC_${corpus}_dm256_ds64_L2_s${seed}_step20000.pt"
      else
        lam_tag="lam$($PY -c "print(f'{$FROZEN_BIAS_LAMBDA:.2f}'.replace('.', 'p'))")"
        init_ckpt="$FROZEN_BIAS_CKPT_ROOT/frozenbias_lm_${arm}_${lam_tag}_${corpus}_dm256_ds64_L2_s${seed}/lmC_${corpus}_dm256_ds64_L2_s${seed}_step20000.pt"
      fi
      echo "LAUNCH (gpu=$gpu): arm=$arm corpus=$corpus seed=$seed -> $out"
      ( CUDA_VISIBLE_DEVICES="$gpu" $PY phase2_familiarization_train.py \
            --init-checkpoint "$init_ckpt" --arm "$arm" --corpus "$corpus" --seed "$seed" \
            --steps $STEPS --ckpt-steps $CKPT_STEPS --ckpt-dir "$CKPT_DIR" --out "$out" \
            --device cuda 2>&1 | tee "$log" ) &
      pids+=($!)
    done
    local fail=0 pid
    for pid in "${pids[@]}"; do
      wait "$pid" || fail=1
    done
    if [ "$fail" -ne 0 ]; then
      echo "ERROR: at least one cell in this GPU-pair FAILED -- halting mechanically (see per-cell logs above)." >&2
      exit 1
    fi
    i=$((i + 2))
  done
}

# ---------------------------------------------------------------------------
# Step 1 -- Stage -1 self-tests (CPU, gate). Both this design's OWN new items AND Phase-1's own
# suite (a transitive dependency-closure check -- phase2_familiarization_train.py imports
# reasoning_link_probe.py directly; a stale/broken copy of THAT file must be caught here, before
# any GPU time is spent, exactly like reasoning_link_phase1b_stage0_chain.sh's own step 1).
# ---------------------------------------------------------------------------
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY reasoning_link_probe.py --mode selftest \
    2>&1 | tee logs/97_phase2_reasoning_link_stage_minus1.log
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY phase2_stage_minus1.py \
    2>&1 | tee logs/98_phase2_stage_minus1.log

# ---------------------------------------------------------------------------
# Step 1.5 -- REAL-KERNEL smoke gate (deploy decision, 2026-07-07): step 1 above runs under the CPU
# fla stub BY DESIGN (fla 0.5.1's RMSNorm has no CPU fallback -- forcing those suites onto real
# kernels crashes in Triton, verified on-box 2026-07-07, logs/predeploy_*_real.log), so NO Stage -1
# item has ever exercised the REAL fla/Triton kernel path the 18 cells train on. This gate closes
# that gap on the PRODUCTION path itself (phase2_smoke_gpu.py's own docstring: strict init-ckpt
# load -> measure_cell_all_h -> per-step finiteness loop -> run_familiarization_cell end-to-end
# incl. optimizer_state_dict checkpoint + resume + Q=K Stage-0.5 gate), on GPU_LIST[0], with NO
# stub env vars. Same bare-command + `set -euo pipefail` abort discipline as step 1; its abort
# branch is proven by the PHASE2_SMOKE_FORCE_FAIL=1 negative test (run at deploy time, re-runnable
# any time). The CPU-stub Stage -1 above stays as-is: it tests LOGIC, this gate tests KERNELS.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" $PY phase2_smoke_gpu.py \
    --init-checkpoint "$FROZEN_BIAS_CKPT_ROOT/frozenbias_lm_off_lam0p00_openr1-mix-ext_dm256_ds64_L2_s0/lmC_openr1-mix-ext_dm256_ds64_L2_s0_step20000.pt" \
    2>&1 | tee logs/98b_phase2_smoke_gpu.log
budget_check

# ---------------------------------------------------------------------------
# Step 2 -- OFF arm, ALL 6 cells (2 corpora x 3 seeds), run FIRST and ALONE (sec 16.2.1's own
# Gate/blind-pin granularity requirement) -- per_token/global do NOT launch until BANDS_PINNED
# is written (step 4) AND the per-checkpoint Stage-0.5 launch gate clears (step 5). MINOR-2 fix:
# launched 2-way parallel (one cell per physical GPU) via run_cells_2way -- this call BLOCKS until
# every one of the 6 OFF cells has completed (or the chain has aborted on a failure), preserving the
# exact same "all 6 OFF cells complete -> bands pinned -> gate -> then per_token/global" barrier the
# pre-fix serial loop also enforced; only the WITHIN-this-step scheduling changed (pairs instead of
# one-at-a-time), never the barrier itself.
# ---------------------------------------------------------------------------
# K is NOT a training-cell axis here (build-time correction: sec 16.2.3's own cost arithmetic prices
# 18 TRAINING cells total -- 3 arms x 2 corpora x 3 seeds, NO x2-for-K anywhere in the training-cost
# lines; K in {20,32} enters ONLY at readout time, step 6 below, via phase2_trajectory_analysis.py's
# own killer-prediction re-application on these SAME 18 cells' saved checkpoints -- see
# phase2_familiarization_train.py's K_TRAIN_DEFAULT for the full disclosure). Each cell below trains
# at the SINGLE fixed K_TRAIN_DEFAULT (32); --k is left at its own default, never swept here.
off_cells=()
for corpus in "${CORPORA[@]}"; do
  for seed in "${SEEDS[@]}"; do
    off_cells+=("off:${corpus}:${seed}")
  done
done
run_cells_2way "${off_cells[@]}"
budget_check

# ---------------------------------------------------------------------------
# Step 3 -- pin BANDS_PINNED-Phase2Familiarization.json from the OFF arm's own 6 cells' val_loss,
# ONE band per trajectory checkpoint step, BEFORE any per_token/global cell launches (sec 16.2.1
# MAJOR-6/MINOR-NEW-1's registered sequencing -- mirrors the exact FROZEN_BIAS_LM_DESIGN.md rung-1
# process failure this design names explicitly, closed here by construction: this Python step runs
# to completion and WRITES the pin file BEFORE bash proceeds to step 5's per_token/global launch).
# ---------------------------------------------------------------------------
$PY - <<PYEOF
import json, os, sys
sys.path.insert(0, ".")
import phase2_bands_pinned as pbp

corpora = ["${CORPORA[0]}", "${CORPORA[1]}"]
seeds = [0, 1, 2]

per_checkpoint_values, result_paths = {}, {}
for step in pbp.CHECKPOINTS:
    per_checkpoint_values[step], result_paths[step] = {}, {}
    for corpus in corpora:
        vals, paths = [], []
        for seed in seeds:
            out = f"results/phase2/off_{corpus}_s{seed}.json"
            with open(out) as f:
                d = json.load(f)
            ck = next(c for c in d["checkpoints"] if c["step"] == step)
            vals.append(ck["val_loss"][corpus])
            paths.append(out)
        per_checkpoint_values[step][corpus] = vals
        result_paths[step][corpus] = paths

doc = pbp.write_bands_pinned_phase2("results/phase2/BANDS_PINNED-Phase2Familiarization.json",
                                     per_checkpoint_values, result_paths)
print(f"BANDS_PINNED written: pinned_at_iso={doc['pinned_at_iso']}")
print("DISCLOSURE (sec 16.2.1 MINOR-R3-4): each pinned band is computed FROM the OFF arm's own "
      "per-seed val-loss -- the OFF arm's own 6 cells are therefore near-tautologically inside their "
      "own band. 'OFF passed its own val-loss gate' is NEVER evidence of anything and must never be "
      "cited as if it corroborated this gate's real target, which is whether per_token's or global's "
      "own val-loss (an INDEPENDENT quantity, never used to construct the band) falls inside a band "
      "built without reference to either of them.")
PYEOF

# ---------------------------------------------------------------------------
# Step 4 -- per-checkpoint Stage-0.5 launch gate (sec 16.5 Constraint 1's own "must not launch on
# failure" requirement): the TERMINAL (5,000-step) checkpoint's own OFF-arm gate must clear for AT
# LEAST ONE (corpus,seed) cell before ANY per_token/global cell launches -- an `if` guard reading a
# REAL subprocess exit code (phase2_gate_enforce.py), never `set -e`'s narrated-only default.
# ---------------------------------------------------------------------------
ANY_GATE_CLEARED=0
for corpus in "${CORPORA[@]}"; do
  for seed in "${SEEDS[@]}"; do
    gate_json="$CKPT_DIR/stage05_gate_phase2fam_off_${corpus}_s${seed}_step5000.json"
    if [ -f "$gate_json" ] && $PY phase2_gate_enforce.py "$gate_json"; then
      echo "Stage-0.5 launch gate CLEARED for corpus=$corpus seed=$seed at the terminal checkpoint."
      ANY_GATE_CLEARED=1
    else
      echo "Stage-0.5 launch gate did NOT clear for corpus=$corpus seed=$seed at the terminal checkpoint."
    fi
  done
done

if [ "$ANY_GATE_CLEARED" != "1" ]; then
  touch results/phase2/STAGE05_LAUNCH_GATE_REFUSED
  echo "REFUSE: NO (corpus,seed) OFF-arm cell cleared the terminal-checkpoint Stage-0.5 gate -- per" >&2
  echo "sec 16.5 Constraint 1, per_token/global familiarization is NOT a licensed launch on this" >&2
  echo "instrument. Halting mechanically (this is NOT a stopping rule for the whole program -- it" >&2
  echo "indicts the readout construct on FAMILIARIZED checkpoints, sec 16.2.1's own disclosed" >&2
  echo "double-duty instrument-validation reading)." >&2
  exit 1
fi
touch results/phase2/STAGE05_LAUNCH_GATE_CLEARED
budget_check

# ---------------------------------------------------------------------------
# Step 5 -- per_token/global, ALL 12 cells (2 arms x 2 corpora x 3 seeds), resume-safe. MINOR-2 fix:
# launched 2-way parallel via the SAME run_cells_2way helper step 2 uses (lam_tag resolution for the
# non-off init-checkpoint path lives inside that function now).
# ---------------------------------------------------------------------------
nonoff_cells=()
for arm in per_token global; do
  for corpus in "${CORPORA[@]}"; do
    for seed in "${SEEDS[@]}"; do
      nonoff_cells+=("${arm}:${corpus}:${seed}")
    done
  done
done
run_cells_2way "${nonoff_cells[@]}"
budget_check

# ---------------------------------------------------------------------------
# Step 6 -- trajectory analysis + hexachotomy classification, ONE PER CORPUS (sec 16.2.1's own
# worked-totality-table structure and the "3 seeds classify... individually" disclosure both point
# to a single, 3-seed-CI-pooled classification per corpus -- phase2_trajectory_analysis.py's own
# module docstring records this scoping decision explicitly, flagged for the independent audit).
# ---------------------------------------------------------------------------
for corpus in "${CORPORA[@]}"; do
  out="results/phase2/trajectory_${corpus}.json"
  CUDA_VISIBLE_DEVICES=$PHASE2_GPUS $PY phase2_trajectory_analysis.py --corpus "$corpus" \
      --ckpt-dir "$CKPT_DIR" --out "$out" 2>&1 | tee "logs/99_phase2_trajectory_${corpus}.log"
done

# ---------------------------------------------------------------------------
# Step 7 -- final summary JSON.
# ---------------------------------------------------------------------------
$PY - <<PYEOF
import json, time
summary = {"design_ref": "REASONING_LINK_DESIGN.md sec 16.2 (Rev 5, CLEARED-FOR-BUILD)",
           "program": "REASONING-LINK Phase 2 (task familiarization)",
           "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "elapsed_s": $SECONDS, "n_gpus": $N_GPUS,
           "projected_gpu_h": ($SECONDS * $N_GPUS) / 3600.0,
           "trajectories": {}}
for corpus in ["${CORPORA[0]}", "${CORPORA[1]}"]:
    with open(f"results/phase2/trajectory_{corpus}.json") as f:
        d = json.load(f)
    summary["trajectories"][corpus] = d["classification"]
with open("results/phase2/PHASE2_SUMMARY.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PYEOF

echo "PHASE-2 CHAIN COMPLETE. See results/phase2/PHASE2_SUMMARY.json"
