#!/bin/bash
# phase2b_chain.sh -- REASONING_LINK_DESIGN.md sec 16.16 (Rev 2.2, DESIGN-CLEARED-FOR-BUILD)
# Phase-2b vocab-space behavioral-contrast instrument: Stage -1 (CPU) -> real-kernel smoke x3 arms
# -> sha256 reuse gate (30 reused OFF checkpoints) -> timing pilot -> OFF-eval cache +
# FLOOR_PINNED write -> OFF-floor gate -> 12 new cells (per_token/global x 2 corpora x 3 seeds)
# 2-way parallel on GPUs 0-1 -> per-checkpoint trajectory analysis (hexachotomy + OOD secondary
# readout) -> summary JSON.
#
# FORKED from phase2_chain.sh (sec 16.16.8's own MAJOR-2 fix), not reused naively: phase2_chain.sh's
# own Step 4 terminal Stage-0.5 REFUSE gate is PERMANENTLY FAILED (sec 16.15.1's 30/30 REFUSED
# table) -- a naive re-run of that chain on the reused OFF checkpoints would self-refuse via
# STAGE05_LAUNCH_GATE_REFUSED before a single Phase-2b cell ever launches, even though sec 16.16.3's
# own MAJOR-1 fix already RETIRES that gate for Phase-2b. phase2_chain.sh stays untouched as the
# historical record of the Phase-2 OFF-arm run that already happened.
#
# BUDGET (sec 16.16.8, Rev 2): raw combined total ~=2.06 GPU-h (cached); debug-tax-inclusive
# registered bracket ~=10.3-20.6 GPU-h. ABORTS (real process-boundary halt) if projected GPU-h
# exceeds the registered ceiling (20.6 GPU-h) at any checkpoint between phases, mirroring
# phase2_chain.sh's own budget_check() exactly, PLUS a dedicated pre-launch TIMING PILOT (sec
# 16.16.11 item 1) that times ONE real eval pass and projects the full 360-pass cost before the
# OFF-eval cache is even built.
#
# GATE ENFORCEMENT (sec 16.5 Constraint 1's own "every gate needs a chain enforcement branch"):
#   1. sha256 reuse gate: the 30 reused OFF .pt files must match the pinned manifest BEFORE any of
#      them is read by the new eval pass (phase2b_ckpt_reuse_gate.py, real subprocess exit code).
#   2. Timing pilot: projects the full 360-pass eval-readout cost from ONE real pass; aborts BEFORE
#      the (expensive) cache build if the projection would blow the registered ceiling.
#   3. OFF-floor gate (sec 16.16.6): REPLACES the old per-checkpoint Stage-0.5 launch gate entirely
#      -- per corpus, phase2b_floor_gate_enforce.py reads the BLIND-pinned FLOOR_PIN
#      (FLOOR_PINNED-Phase2b.json, written BEFORE this gate runs, sec 16.16.6's own sequencing) and
#      classifies FLOOR-PASS / PARTIAL-BELOW-FLOOR / FAMILIARIZATION-NULL. Wave-level: if BOTH
#      corpora land FAMILIARIZATION-NULL, the chain REFUSES the 12-cell launch (mirrors the retired
#      Stage-0.5 gate's own "at least one clears" convention).
#
# RESUME-SAFETY: every cell launch is skipped if its own --out JSON already exists AND validates as
# complete; phase2_familiarization_train.py's own --resume (default ON) additionally resumes a
# PARTIAL cell from its own last trajectory checkpoint. The OFF-eval cache build is idempotent at
# the FILE level (re-running overwrites the same deterministic values) but is NOT itself
# incrementally resumable mid-build in this Rev -- a crash mid-cache-build re-runs the full 120-pass
# build (cheap, ~0.26 GPU-h at the registered reference rate).
#
# Same `set -euo pipefail` discipline every sibling chain in this dir uses.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs results/phase2/ckpts

PHASE2_GPUS=${PHASE2_GPUS:-"0,1"}
N_GPUS=$(echo "$PHASE2_GPUS" | tr ',' '\n' | wc -l | tr -d ' ')
IFS=',' read -ra GPU_LIST <<< "$PHASE2_GPUS"
FROZEN_BIAS_CKPT_ROOT=${FROZEN_BIAS_CKPT_ROOT:-/data/deltanet_rd_frozenbias_ckpts}
CKPT_DIR=results/phase2/ckpts
STEPS=5000
CKPT_STEPS="250 500 1000 2500 5000"
CORPORA=(openr1-mix-ext wikitext-mix-ext)
SEEDS=(0 1 2)
FROZEN_BIAS_LAMBDA=0.58
# Box-local path (mirrors phase2b_ckpt_reuse_gate.py's own DEFAULT_MANIFEST convention, the SAME
# K=69-precedent `results/`-relative pattern -- NOT `experiment-runs/`, a repo-root path never
# scp'd to the box). The repo's own committed copy (source of truth, git-tracked) lives at
# experiment-runs/2026-07-08_phase2_familiarization/gates/phase2b_off_ckpts_reuse_manifest.sha256
# -- a deploy step (out of scope for this LOCAL BUILD) must copy that SAME file content here first.
SHA256_MANIFEST=results/phase2/gates/phase2b_off_ckpts_reuse_manifest.sha256
OFF_CACHE=results/phase2/off_lquery_cache-Phase2b.json
FLOOR_PINNED=results/phase2/FLOOR_PINNED-Phase2b.json

# Registered bracket ceiling (sec 16.16.8, Rev 2): ~=10.3-20.6 GPU-h, debug-tax-inclusive, CACHED
# primary figure. This script aborts if PROJECTED GPU-h exceeds the UPPER bound -- never silently
# runs past it.
BUDGET_CEILING_GPU_H=20.6

budget_check() {
  local elapsed_s=$SECONDS
  local projected_gpu_h
  projected_gpu_h=$($PY -c "print(($elapsed_s * $N_GPUS) / 3600.0)")
  echo "[budget] elapsed=${elapsed_s}s  n_gpus=$N_GPUS  projected_gpu_h=${projected_gpu_h}  ceiling=${BUDGET_CEILING_GPU_H}"
  if $PY -c "import sys; sys.exit(0 if $projected_gpu_h > $BUDGET_CEILING_GPU_H else 1)"; then
    echo "ABORT: projected GPU-h (${projected_gpu_h}) exceeds the registered ceiling (${BUDGET_CEILING_GPU_H}, sec 16.16.8) -- halting mechanically." >&2
    touch results/phase2/PHASE2B_BUDGET_ABORTED
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# 2-way parallel cell launcher (forked verbatim from phase2_chain.sh's own MINOR-2-fixed
# run_cells_2way -- same resume-safety, same per-cell GPU pinning, same fail-fast-on-either-half
# discipline). Reused byte-for-byte rather than sourced (phase2_chain.sh is not written to be
# sourced -- it executes its own chain top-level immediately).
# ---------------------------------------------------------------------------
run_cells_2way() {
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
      log="logs/99_phase2b_${arm}_${corpus}_s${seed}.log"
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
# Step 1 -- Stage -1 self-tests (CPU, gate). Reused verbatim from phase2_chain.sh's own step 1 --
# Phase-2b's own new Stage -1 items (the _references teeth-run, the FTTTT behavioral test, the
# surgery-absence assertions, the --arm-vs-config assertion, the arm-independence pairing positive
# test, the 2 new seed kinds' collision-freedom, AND the sha256-reuse-gate's / floor-gate's own
# negative-test fixture suites, items 20-21) all live INSIDE phase2_stage_minus1.py (mirrors items
# 6/7's own pre-existing convention of wiring a gate module's `_run_selftest()` in here rather than
# requiring a separate chain step) -- pure logic/fixtures, CPU-only, no real checkpoint or GPU needed.
# ---------------------------------------------------------------------------
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY reasoning_link_probe.py --mode selftest \
    2>&1 | tee logs/97_phase2b_reasoning_link_stage_minus1.log
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY phase2_stage_minus1.py \
    2>&1 | tee logs/98_phase2b_stage_minus1.log

# ---------------------------------------------------------------------------
# Step 1.5 -- REAL-KERNEL smoke gate, THREE arms (sec 16.16.9 MAJOR-5, EXPANDED from phase2_chain.
# sh's own single off-arm-only smoke step). per_token and global exercise DIFFERENT tensor
# operations inside apply_frozen_bias_blend (per-token gather-lookup vs. global broadcast add) --
# ALL THREE required to pass before the 12-cell launch, not merely one non-off arm as a stand-in.
# ---------------------------------------------------------------------------
for smoke_arm in off per_token global; do
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" $PY phase2_smoke_gpu.py --arm "$smoke_arm" \
      --corpus openr1-mix-ext \
      2>&1 | tee "logs/98c_phase2b_smoke_gpu_${smoke_arm}.log"
done
budget_check

# ---------------------------------------------------------------------------
# Step 2 -- sha256 runtime verification of the 30 reused OFF checkpoints (sec 16.16.8's own
# belt-and-suspenders gate, run BEFORE any of them is read by the new eval pass): bash-level
# `sha256sum -c` (belt) + Python-level phase2b_ckpt_reuse_gate.py (suspenders, independent
# recomputation, not merely trusting the belt's own exit code alone).
# ---------------------------------------------------------------------------
SHA256_MANIFEST_ABS="$(pwd)/$SHA256_MANIFEST"
( cd "$CKPT_DIR" && sha256sum -c "$SHA256_MANIFEST_ABS" ) 2>&1 | tee logs/99_phase2b_sha256_belt.log \
  || { echo "REFUSE: bash-level sha256sum -c FAILED on the 30 reused OFF checkpoints -- halting mechanically." >&2; touch results/phase2/PHASE2B_SHA256_REUSE_GATE_REFUSED; exit 1; }
if ! $PY phase2b_ckpt_reuse_gate.py --ckpt-dir "$CKPT_DIR" --manifest "$SHA256_MANIFEST" \
    2>&1 | tee logs/99_phase2b_sha256_suspenders.log; then
  echo "REFUSE: Python-level sha256 reuse gate FAILED -- halting mechanically." >&2
  touch results/phase2/PHASE2B_SHA256_REUSE_GATE_REFUSED
  exit 1
fi
touch results/phase2/PHASE2B_SHA256_REUSE_GATE_CLEARED

# ---------------------------------------------------------------------------
# Step 3 -- pre-launch timing pilot (sec 16.16.11 item 1's own mandatory obligation): times ONE
# real eval pass on a reused OFF checkpoint, projects the full 360-cached-pass cost, aborts BEFORE
# the (expensive) cache build if the projection would exceed the registered ceiling.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" $PY phase2b_off_cache.py --ckpt-dir "$CKPT_DIR" --time-pilot \
    --device cuda 2>&1 | tee logs/99a_phase2b_timing_pilot.log
budget_check

# ---------------------------------------------------------------------------
# Step 4 -- OFF-eval cache + FLOOR_PINNED writes (sec 16.16.8's "New first Python step" / sec
# 16.16.6's blind-pin sequencing) -- BEFORE any new cell launches. Writes off_lquery_cache-
# Phase2b.json (120 passes: 6 reused OFF cells x 5 checkpoints x 2 K's x 2 hop-sets) and
# FLOOR_PINNED-Phase2b.json (blind, data-derived FLOOR_PIN per corpus).
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" $PY phase2b_off_cache.py --ckpt-dir "$CKPT_DIR" \
    --out-cache "$OFF_CACHE" --out-floor "$FLOOR_PINNED" --device cuda \
    2>&1 | tee logs/99b_phase2b_off_cache.log
budget_check

# ---------------------------------------------------------------------------
# Step 5 -- OFF-floor gate (sec 16.16.6, REPLACES the old per-checkpoint Stage-0.5 launch gate
# entirely -- sec 16.16.9's own gate list drops the Stage-0.5 launch gate). Per corpus: read the
# ALREADY-PINNED FLOOR_PIN (never recomputed here) and this corpus's own pooled ratio, classify via
# phase2b_floor_gate_enforce.py. Wave-level: if BOTH corpora land FAMILIARIZATION-NULL, REFUSE.
# ---------------------------------------------------------------------------
ANY_NOT_NULL=0
for corpus in "${CORPORA[@]}"; do
  ratio=$($PY -c "
import json
with open('$FLOOR_PINNED') as f:
    doc = json.load(f)
print(doc['floor_by_corpus']['$corpus']['pooled_ratio'])
")
  floor_pin=$($PY -c "
import json
with open('$FLOOR_PINNED') as f:
    doc = json.load(f)
print(doc['floor_by_corpus']['$corpus']['floor_pin'])
")
  if $PY phase2b_floor_gate_enforce.py --ratio "$ratio" --floor-pin "$floor_pin" --corpus "$corpus"; then
    echo "OFF-floor gate: corpus=$corpus NOT FAMILIARIZATION-NULL -- proceeds to hexachotomy classification."
    ANY_NOT_NULL=1
  else
    echo "OFF-floor gate: corpus=$corpus FAMILIARIZATION-NULL -- excluded from any reading."
  fi
done

if [ "$ANY_NOT_NULL" != "1" ]; then
  touch results/phase2/PHASE2B_FLOOR_GATE_REFUSED
  echo "REFUSE: BOTH corpora landed FAMILIARIZATION-NULL at the OFF-floor gate -- per sec 16.5" >&2
  echo "Constraint 1, the 12-cell per_token/global launch is NOT a licensed launch on this" >&2
  echo "instrument. Halting mechanically (mirrors the retired Stage-0.5 gate's own 'at least one" >&2
  echo "clears' convention, sec 16.16.6)." >&2
  exit 1
fi
budget_check

# ---------------------------------------------------------------------------
# Step 6 -- per_token/global, ALL 12 cells (2 arms x 2 corpora x 3 seeds), resume-safe, 2-way
# parallel on GPUs 0-1 (reuses run_cells_2way above verbatim, the SAME helper phase2_chain.sh's own
# Step 5 uses).
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
# Step 7 -- per-checkpoint trajectory analysis + hexachotomy classification + OOD secondary
# readout, ONE PER CORPUS (phase2_trajectory_analysis.py's own MAJOR-1-rewritten analyze_corpus,
# now reading off_vals from the cache built at step 4, sha256-verified against FLOOR_PINNED-
# Phase2b.json's own pinned hash before any verdict is computed -- build-audit MAJOR fix).
# ---------------------------------------------------------------------------
for corpus in "${CORPORA[@]}"; do
  out="results/phase2/trajectory_${corpus}_phase2b.json"
  CUDA_VISIBLE_DEVICES=$PHASE2_GPUS $PY phase2_trajectory_analysis.py --corpus "$corpus" \
      --ckpt-dir "$CKPT_DIR" --off-cache "$OFF_CACHE" --floor-pinned "$FLOOR_PINNED" --out "$out" \
      2>&1 | tee "logs/99c_phase2b_trajectory_${corpus}.log"
done

# ---------------------------------------------------------------------------
# Step 8 -- final summary JSON.
# ---------------------------------------------------------------------------
$PY - <<PYEOF
import json, time
summary = {"design_ref": "REASONING_LINK_DESIGN.md sec 16.16 (Rev 2.2, DESIGN-CLEARED-FOR-BUILD)",
           "program": "REASONING-LINK Phase 2b (vocab-space behavioral contrast)",
           "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "elapsed_s": $SECONDS, "n_gpus": $N_GPUS,
           "projected_gpu_h": ($SECONDS * $N_GPUS) / 3600.0,
           "trajectories": {}, "secondary_ood": {}}
for corpus in ["${CORPORA[0]}", "${CORPORA[1]}"]:
    with open(f"results/phase2/trajectory_{corpus}_phase2b.json") as f:
        d = json.load(f)
    summary["trajectories"][corpus] = d["classification"]
    summary["secondary_ood"][corpus] = {
        arm: {str(c): {"det32": v["det32"], "det20": v["det20"]} for c, v in table.items()}
        for arm, table in d["secondary_ood"].items()
    }
with open("results/phase2/PHASE2B_SUMMARY.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
PYEOF

echo "PHASE-2B CHAIN COMPLETE. See results/phase2/PHASE2B_SUMMARY.json"
