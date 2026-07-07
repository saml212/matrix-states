#!/bin/bash
# REASONING_LINK_DESIGN.md (Rev 6, CLEARED-FOR-BUILD) -- Phase 1 REAL run.
# Zero new training: two legs of eval-only, hooked forward passes over
# already-archived checkpoints (Leg A: /data/deltanet_rd_frozenbias_ckpts/,
# Leg B: /data/lm_rd_trackc_ckpts/). Committed budget ~=24.20 GPU-h,
# margin ~=0.80 GPU-h, ceiling 25 GPU-h (sec 10) -- the abort rules below
# (Stage 0's 3x trigger, Stage 0.5's 2x/4x two-tier trigger), not the
# arithmetic margin, are the real safety valve (sec 10's own framing).
#
# Six steps:
#   1. Stage -1 self-tests (sec 9, ALL 14 items + the sec-12 outcome-
#      routing gate checks) -- pure CPU, no GPU, no checkpoints. A failure
#      here means the scp'd/rsync'd copy of this deploy is stale/corrupt
#      relative to what was built+audited -- stop before any real
#      checkpoint pass (mirrors mech_stage1_chain.sh/mech_stage2_chain.sh's
#      own "self-test gate" step 1 convention).
#   2. Stage 0 calibration (sec 9): the PINNED Leg-A control-arm checkpoint
#      (frozenbias_lm_off_lam0p00_openr1-mix-ext_dm256_ds64_L2_s0, final
#      step 20000 -- Rev 6's own fix: this is NOT "14M mixcontrol", it is
#      the Leg-A comparison family's own control arm, which happens to
#      share the mixcontrol architecture), K=32, h in {1,2,3,4}, BOTH
#      query markers (sec 7.6's robustness replicate) -- BLINDED to any
#      headline decision: this script's own stdout/log NEVER shows a
#      recovered_frac/cosine/premise-median number (see
#      reasoning_link_probe.py::blinded_console_summary), only structural
#      pass/fail gate outcomes and timing; the real numbers live ONLY in
#      the JSON this step writes, read at a deliberate later unblinding
#      step, never at chain-launch time.
#   3. Stage 0.5 (sec 9): rung-3 (1.31B) pass-cost calibration at K=64
#      (rung-3's sole committed K), BLINDED to any h4-style recovery
#      readout (timing only, scoring is never computed in this step at
#      all). Two-tier abort (>2x/>4x the 392M two-forward baseline,
#      0.09 GPU-h/pass) + a bounded, one-shot OOM fallback (retry once at
#      batch_size=8, then halt-with-disclosure) -- both implemented inside
#      reasoning_link_probe.py::run_stage05, not re-implemented here.
#   4-9. The committed Phase-1 grid, in the design's own order (sec 9
#      Stage 1, sec 10): Leg A native + surgery at K in {20,32}, then Leg
#      B's 4 rungs at their own single committed near-cliff K (K=32 for
#      the two d=64 rungs, K=64 for the two d=128 rungs). Budget-abort
#      checks (registered, not merely narrated) run BETWEEN stages 2/3 and
#      the grid launch -- see the CHECK_* functions below. The chain runs
#      every committed cell regardless of premise-gating outcomes (sec 9's
#      own note: "the chain can run all cells -- the GATING affects
#      interpretation/unblinding, which happens at harvest"); only the
#      registered BUDGET aborts are chain-level gates.
#   10. touch results/reasoning_link/REASONING_LINK_PHASE1_DONE.
#
# ---------------------------------------------------------------------------
# CRITICAL dependency-closure list (mech_stage2_chain.sh's own "[LEARN] from
# Stage 1: run 1 died on an unshipped dependency" -- restated here so THIS
# chain's deploy does not repeat it). Every file reasoning_link_probe.py /
# reasoning_link_stage_minus1.py import, TRANSITIVELY, at MODULE-IMPORT
# time -- verified by direct read of every file's own import lines this
# build session, not assumed:
#   reasoning_link_probe.py        (this program's own probe script)
#   reasoning_link_stage_minus1.py (this program's own Stage -1 script,
#                                    imports reasoning_link_probe)
#   grammar_rd.py                  (reasoning_link_probe.py's own import:
#                                    build_entity_pools, DeltaNetRDTaskConfig,
#                                    sample_batch_rd, _iterate_permutation,
#                                    _permutation_graph, self_query_tokens,
#                                    EntityPools, load_gpt2_tokenizer)
#   lm_pretrain_rd.py               (reasoning_link_probe.py's own import:
#                                    DeltaNetLM, FROZEN_BIAS_ARM_MODES,
#                                    apply_frozen_bias_blend,
#                                    build_frozen_bias_table,
#                                    frozen_bias_global_vector,
#                                    _MIN_KERNEL_T, _SAFE_D_STATE)
#   model_rd.py                     (lm_pretrain_rd.py's own import:
#                                    _MIN_KERNEL_T, _SAFE_D_STATE,
#                                    _polar_via_eigh, newton_schulz_orthogonalize;
#                                    ALSO reasoning_link_probe.py's own
#                                    DIRECT import: gram_deviation)
#   rank_utils.py                   (lm_pretrain_rd.py's own import:
#                                    effective_rank, stable_rank; ALSO
#                                    model_rd.py's own import)
#   key_anchoring.py                (lm_pretrain_rd.py's own import:
#                                    random_unit_rows_init, ANCHOR_INIT_SEED;
#                                    ALSO model_rd.py's own import)
#   hard_selectivity_rd.py          (lm_pretrain_rd.py's own import --
#                                    several names, unused by this probe's
#                                    own cells but executed at
#                                    lm_pretrain_rd.py's own module-import
#                                    time regardless)
#   deltanet_core.py                (model_rd.py's own import: apply_state_power)
#   geo3_simulator.py                (key_anchoring.py's own import: newton_schulz)
#   rev7_threshold_derive.py        (key_anchoring.py's own import, "pure
#                                    Python, fla-free")
# PLUS: the real `fla` package (pip-installed on this box; this program's
# own CPU fla-stub is Stage -1-only and never installed when the real
# package import succeeds -- reasoning_link_probe.py's own module
# docstring), the `transformers` package + a cached/reachable 'gpt2'
# tokenizer (grammar_rd.py's own `load_gpt2_tokenizer`, already a
# pre-existing dependency of every sibling script in this dir that builds
# BIND/QUERY episodes), and the two checkpoint roots
# (/data/deltanet_rd_frozenbias_ckpts/, /data/lm_rd_trackc_ckpts/) already
# present on this box per REASONING_LINK_DESIGN.md sec 0. If ANY of the
# repo-file list above is missing from the scp'd/rsync'd copy on this box,
# step 1 below fails at Python import time, before any GPU time is spent --
# verify the deploy's file list against this comment BEFORE launching, not
# after a crash.
#
# Same `set -euo pipefail` discipline every sibling chain in this dir uses
# (mech_stage1_chain.sh, mech_stage2_chain.sh, keyanchor_confirm_chain.sh)
# -- does NOT rely on `cmd 2>&1 | tee log && next_cmd` masking a crashed
# command's real exit status behind `tee`'s own (successful) one.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs results/reasoning_link

# GPU assignment -- override via env var if the box's reservation map has
# changed before this actually launches (this build pass performs NO
# SSH/GPU execution; see the build report). Default GPU 3 per this
# program's own registered default -- RE-VERIFY with nvidia-smi before this
# chain actually runs, per house rule and every sibling chain's own
# "re-verify immediately before any BUILD agent executes anything" standing
# constraint.
REASONING_LINK_GPU=${REASONING_LINK_GPU:-3}

FROZEN_BIAS_CKPT_ROOT=${FROZEN_BIAS_CKPT_ROOT:-/data/deltanet_rd_frozenbias_ckpts}
TRACKC_CKPT_ROOT=${TRACKC_CKPT_ROOT:-/data/lm_rd_trackc_ckpts}

# sec 9 Stage 0's pinned checkpoint (Rev 6 fix -- the ACTUAL Leg-A control
# arm, not "14M mixcontrol", which is a disjoint Leg-B/Track-C checkpoint
# with no arms at all).
STAGE0_CKPT="$FROZEN_BIAS_CKPT_ROOT/frozenbias_lm_off_lam0p00_openr1-mix-ext_dm256_ds64_L2_s0/lmC_openr1-mix-ext_dm256_ds64_L2_s0_step20000.pt"

# ---------------------------------------------------------------------------
# Step 1 -- Stage -1 self-tests (gate). Pure CPU BY REGISTRATION (design
# sec 9's tolerances are fp32/CPU; the suite's toy models live on CPU).
# LAUNCH FIX (2026-07-07, after the first box launch crashed here): the
# suite must run through the probe's own CPU fla-stub even though the box
# has real fla installed -- CPU toy tensors into the real Triton kernel
# crash with "Pointer argument cannot be accessed from Triton". A first
# attempted fix (pinning a GPU) was WRONG: the tensors are CPU-side by
# design, so GPU visibility is irrelevant. REASONING_LINK_FORCE_CPU_STUB=1
# makes _ensure_fla_stub() install the stub unconditionally -- the EXACT
# audited local semantics. The real kernel is still exercised for real at
# Stage 0 (sec 9 purpose (a) re-runs the item-11 causality assertion on a
# real checkpoint).
# ---------------------------------------------------------------------------
REASONING_LINK_FORCE_CPU_STUB=1 CUDA_VISIBLE_DEVICES= $PY reasoning_link_probe.py --mode selftest \
    2>&1 | tee logs/90_reasoning_link_stage_minus1.log

# ---------------------------------------------------------------------------
# Step 2 -- Stage 0 calibration (sec 9). BLINDED console output (see
# reasoning_link_probe.py::blinded_console_summary) -- the tee'd log below
# will show pass/fail gate outcomes and timing ONLY, never a
# recovered_frac/cosine/premise-median number. The full numbers land ONLY
# in the --out JSON, read at a deliberate unblinding step, not at launch.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$REASONING_LINK_GPU $PY reasoning_link_probe.py \
    --mode stage0 --ckpt "$STAGE0_CKPT" --k 32 --batch-size 16 --device cuda \
    --out results/reasoning_link/stage0_calibration.json \
    2>&1 | tee logs/91_reasoning_link_stage0.log

# Registered abort check (sec 10, abort rule 1): if Stage 0's real
# measured wall-clock exceeds 3x the two-forward d=64 baseline (3 x
# 0.0696 GPU-h), halt before Stage 1 launches and re-price the full grid
# from the real number -- this is a REAL check on the JSON's own `wall_s`
# field, not a narrated rule.
$PY - << 'PYEOF'
import json, sys
d = json.load(open("results/reasoning_link/stage0_calibration.json"))
wall_s = d["wall_s"]
baseline_s = 0.0696 * 3600.0
ratio = wall_s / baseline_s
print(f"Stage 0 abort check: wall_s={wall_s:.1f}s baseline={baseline_s:.1f}s ratio={ratio:.2f}x "
      f"(abort threshold 3x)")
if ratio > 3.0:
    print("ABORT: Stage 0 exceeded 3x baseline -- halt before Stage 1, re-price the grid (sec 10).",
          file=sys.stderr)
    sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------
# Step 3 -- Stage 0.5 rung-3 cost calibration (sec 9 MAJOR-6). Timing-only,
# nothing to blind. Two-tier abort + OOM fallback both implemented inside
# run_stage05 itself; this step only reads the returned `action` field.
# ---------------------------------------------------------------------------
RUNG3_CKPT="$TRACKC_CKPT_ROOT/wave3/lmC_openr1-mix-ext_dm2560_ds128_L22_s0_step20000.pt"  # LAUNCH FIX 3: flat layout, no checkpoints/ subdir (verified on box)
# BEST-EFFORT path (see reasoning_link_probe.py::leg_b_ckpt_path_bestguess's
# own disclosure -- Track C naming is "per-wave, not uniform", sec 0). If
# this path does not exist, find the real rung-3 checkpoint on the box
# (`find $TRACKC_CKPT_ROOT -iname '*ds128_L22*step20000.pt'`) and set
# RUNG3_CKPT explicitly before re-running this step.
CUDA_VISIBLE_DEVICES=$REASONING_LINK_GPU $PY reasoning_link_probe.py \
    --mode stage05 --ckpt "$RUNG3_CKPT" --k 64 --batch-size 4 --device cuda \
    --out results/reasoning_link/stage05_rung3_cost_calibration.json \
    2>&1 | tee logs/92_reasoning_link_stage05.log

RUNG3_ACTION=$($PY -c "import json; print(json.load(open('results/reasoning_link/stage05_rung3_cost_calibration.json'))['action'])")
echo "Stage 0.5 action: $RUNG3_ACTION"
case "$RUNG3_ACTION" in
  HALT*) echo "ABORT: Stage 0.5 says HALT Leg-B rung-3 rows -- skipping rung-3 cells below (sec 9)."
         RUNG3_HALTED=1 ;;
  *)     RUNG3_HALTED=0 ;;
esac

# ---------------------------------------------------------------------------
# Steps 4-7 -- Leg A committed grid (sec 5, sec 9 Stage 1): 18 core cells
# (3 arms x 2 corpora x 3 seeds) x K in {20,32} x native, PLUS surgery for
# the two arm-only cells (per_token, global) -- 12 arm-only cells x 2 K,
# sec 5.2a/sec 10's own "12 arm-only cells" surgery-grid subtotal (MAJOR-3
# fix, this audit round: the 'off' arm's surgery pass is skipped below, see
# the condition inside the loop). Looped here rather than one invocation
# per cell -- run to completion, per-cell budget abort is NOT re-checked
# per cell (sec 10's abort rules are Stage-0/0.5-level, not
# per-Stage-1-cell); a mid-grid crash/OOM on a real box is a separate,
# unregistered failure mode this chain does not paper over (it will simply
# stop, per set -euo pipefail).
# ---------------------------------------------------------------------------
LEG_A_ARMS=(off per_token global)
LEG_A_CORPORA=(openr1-mix-ext wikitext-mix-ext)
LEG_A_SEEDS=(0 1 2)
LEG_A_KS=(20 32)

for arm in "${LEG_A_ARMS[@]}"; do
  for corpus in "${LEG_A_CORPORA[@]}"; do
    for seed in "${LEG_A_SEEDS[@]}"; do
      for k in "${LEG_A_KS[@]}"; do
        for surgery in native off; do
          # MAJOR-3 fix (this audit round): sec 5.2a registers the 'off' arm's own blend-OFF cell
          # as mechanically IDENTICAL to its native pass ("the off arm never trains a
          # frozen_bias_table -- there is nothing to blend ... the one cell, reported once") --
          # the surgery grid is priced (sec 10) as 12 ARM-ONLY cells x 2 K (per_token + global
          # only), never including 'off'. Running surgery='off' for arm='off' would waste an
          # entire (checkpoint,K) pass recomputing a cell already covered by that arm's own
          # native pass above.
          if [ "$arm" = "off" ] && [ "$surgery" = "off" ]; then
            continue
          fi
          out="results/reasoning_link/leg_a_${arm}_${corpus}_s${seed}_k${k}_${surgery}.json"
          log="logs/93_leg_a_${arm}_${corpus}_s${seed}_k${k}_${surgery}.log"
          # LAUNCH FIX 6 resume-safety (house [LEARN]): skip completed cells.
          if [ -s "$out" ] && $PY -c "import json,sys; d=json.load(open('$out')); sys.exit(0 if d.get('forward_counts') else 1)" 2>/dev/null; then
            echo "SKIP (already complete): $out"
            continue
          fi
          CUDA_VISIBLE_DEVICES=$REASONING_LINK_GPU $PY reasoning_link_probe.py \
              --mode cell --family leg_a --arm "$arm" --corpus "$corpus" --ckpt-seed "$seed" \
              --ckpt-base-dir "$FROZEN_BIAS_CKPT_ROOT" --k "$k" --hops 1,2,3,4 \
              --surgery "$surgery" --batch-size 16 --device cuda --out "$out" \
              2>&1 | tee "$log"
        done
      done
    done
  done
done

# ---------------------------------------------------------------------------
# Steps 8-9 -- Leg B committed grid (sec 6, sec 9 Stage 1): 4 rungs x each
# rung's own single committed near-cliff K (K=32 for d=64 rungs 0/1, K=64
# for d=128 rungs 2/3). Rung 3 skipped if Stage 0.5 said HALT.
# ---------------------------------------------------------------------------
LEG_B_CORPORA=(openr1-mix-ext wikitext-mix-ext)

RUNG3_DEFERRED=0
for rung in 0 1 2 3; do
  if [ "$rung" = "3" ] && [ "$RUNG3_HALTED" = "1" ]; then
    echo "Skipping rung 3 (Stage 0.5 HALT)."
    continue
  fi
  # LAUNCH FIX 6: rung-3 cells evaluate the FINAL 1.31B checkpoint, which
  # exists only after the trackc wave-3 training completes (ALL_DONE
  # sentinel). Defer -- do NOT let the final-step glob grab a mid-training
  # intermediate. Re-run this chain after ALL_DONE; resume-safety below
  # skips every already-completed cell, so the re-run costs only the
  # deferred rung-3 rows.
  if [ "$rung" = "3" ] && [ ! -f results/lm_rd_trackc/wave3/ALL_DONE ]; then
    echo "DEFERRED: rung 3 cells (trackc wave3 ALL_DONE not present -- final 1.31B ckpt pending)."
    RUNG3_DEFERRED=1
    continue
  fi
  if [ "$rung" -lt 2 ]; then k=32; else k=64; fi
  # rung 3 is 2 corpora x 1 seed (sec 6.1's PINNED configuration); rungs 0-2 are 2 corpora x 3 seeds.
  if [ "$rung" = "3" ]; then seeds=(0); else seeds=(0 1 2); fi
  # LAUNCH FIX 5 (2026-07-07, debug-agent root cause): fla 0.5.1's
  # layer_norm_fwd_kernel1 (the D>512 one-program-per-row branch every
  # d_model=2560 norm call routes through) computes int32 pointer offsets --
  # at batch 16 the combined main+self-query forward-B rows x T x d_model
  # exceeds 2^31 and the offset wraps (illegal memory access; surfaced
  # asynchronously as CUBLAS_INTERNAL_ERROR in run 5). batch 8 clears the
  # overflow but OOMs on the 20GiB fp32 FFN intermediate; batch 4 PASSES
  # end-to-end (diag_stage05_b4.json: ratio_to_baseline=0.085, "OK: within
  # budget, proceed"). Scoped to rung 3 ONLY: smaller rungs' d_model<=1536
  # never reaches the overflow region at batch 16, and Stage 0 already
  # passed at batch 16. Mirrors the training sweep's own per-rung batch
  # precedent (BATCH_SIZE_BY_RUNG in run_lm_rd_trackc_sweep.py).
  # LAUNCH FIX 7 (2026-07-07, run 7): rung 2 (392M, dm1536/L16) OOMs at
  # batch 16 -- the combined forward-B FFN intermediate is 2*16*64 rows x
  # T512 x 4*dm1536 x 4B = 24.0 GiB in one op. batch 8 halves it to 12 GiB
  # (fits alongside the ~1.5GB weights) and its flat-rows x dm = 805M stays
  # far under the int32 offset line. Same per-rung principle as fix 5.
  if [ "$rung" = "3" ]; then cell_batch=4; elif [ "$rung" = "2" ]; then cell_batch=8; else cell_batch=16; fi
  for corpus in "${LEG_B_CORPORA[@]}"; do
    for seed in "${seeds[@]}"; do
      out="results/reasoning_link/leg_b_rung${rung}_${corpus}_s${seed}_k${k}.json"
      log="logs/94_leg_b_rung${rung}_${corpus}_s${seed}_k${k}.log"
      # LAUNCH FIX 6 resume-safety (house [LEARN]: skip already-completed
      # work by checking output validity, not just existence).
      if [ -s "$out" ] && $PY -c "import json,sys; d=json.load(open('$out')); sys.exit(0 if d.get('forward_counts') else 1)" 2>/dev/null; then
        echo "SKIP (already complete): $out"
        continue
      fi
      CUDA_VISIBLE_DEVICES=$REASONING_LINK_GPU $PY reasoning_link_probe.py \
          --mode cell --family leg_b --rung "$rung" --corpus "$corpus" --ckpt-seed "$seed" \
          --ckpt-base-dir "$TRACKC_CKPT_ROOT" --k "$k" --hops 1,2,3,4 \
          --surgery native --batch-size "$cell_batch" --device cuda --out "$out" \
          2>&1 | tee "$log"
    done
  done
done

if [ "$RUNG3_DEFERRED" = "1" ]; then
  touch results/reasoning_link/REASONING_LINK_PHASE1_PARTIAL
  echo "REASONING-LINK Phase 1 grid complete EXCEPT deferred rung-3 (re-run after trackc ALL_DONE)."
else
  touch results/reasoning_link/REASONING_LINK_PHASE1_DONE
  echo "REASONING-LINK Phase 1 committed grid complete."
fi
