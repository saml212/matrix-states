#!/bin/bash
# FROZEN_BIAS_LM_DESIGN.md sec 12.5 -- Stage 2 REAL run: H3's gradient-flow
# instrumented training cells. Gated on sec 12.11.4's frozen-rule result
# (already evaluated, not re-decided here): "Stage-2 gate (sec 12.5's frozen
# rule): FULL 20,000-step branch. Arm2: sign(delta@5000)=sign(delta@20000)
# [pass] and |-0.0721| >= 0.5*|-0.0903|=0.0451 [pass] -> full. Arm2': sign
# [pass] but |-0.0570| < 0.5*|-0.1203|=0.0601 -> truncated. Per the rule's
# own clause ('if the two comparisons select different branches, the full
# 20,000-step branch governs for both'): Stage 2 is authorized at full
# 20,000 steps." This chain therefore runs the FULL branch, NOT the cheaper
# 3,000-step branch sec 12.5 also specifies (that branch was NOT selected).
#
# Two steps:
#   1. frozen_bias_gradflow_probe.py --smoke (pure CPU, no GPU, no
#      checkpoints/data -- pushes sec 12.5's own three-part smoke spec
#      through the REAL DeltaNetLM/DeltaNetLMMixer classes and the REAL
#      register_kraw_gradnorm_hooks() function this chain's real cells also
#      call, closing sec 12.9 item 4's formula-vs-plumbing gap. Uses a
#      CPU-safe fla stub ONLY because this box's own `fla` package IS
#      installed for the real cells below -- see the script's own module
#      docstring for why the smoke forces a k-dependent stub kernel where
#      the sibling smoke files' stub does not need to).
#   2. ONE script invocation training all 3 arms (off, per_token, global)
#      SEQUENTIALLY in a single process (sec 12.5: "Output: one JSON per
#      cell (or one combined)" -- this build picked "one combined",
#      amortizing Python/CUDA/model-import overhead across all 3 cells
#      instead of paying it 3x, per the script's own module docstring) --
#      1 seed (0), 1 corpus (openr1-mix-ext), FULL 20,000 steps/cell,
#      backward-hook grad-norm telemetry on k_raw every 100 steps/layer.
#
# Cost: sec 12.5 states ~912s/cell x 3 ~= 0.76 GPU-h at the full 20,000-step
# branch (~0.0456 s/step). Re-derivation attempted this build session (no
# SSH to the box from here -- could not read the real rung-1 frozen-bias
# calibration.json, which is not committed to this repo, only small
# archived JSONs are under the repo's size-capped archive policy). As a
# CPU-only corroboration, this repo DOES carry a two-point calibration at
# the IDENTICAL architecture (dm256/ds64/L2, openr1-mix corpus):
# results/lm_rd_trackc/calibration/calib_control_ptA_lm_openr1-mix_dm256_
# ds64_L2_s0.json (100 steps, 6.0068s) and .../calib_control_ptB_..._s0.json
# (300 steps, 14.7468s) -> (14.7468-6.0068)/(300-100) = 0.0437 s/step --
# within 5% of sec 12.5's own 0.0456 s/step figure, corroborating (not
# replacing) the design doc's own pinned 0.76 GPU-h estimate, reproduced
# here verbatim. Frozen-bias LM program ledger: 6.93/135 GPU-h committed so
# far (EXPERIMENT_LOG.md 2026-07-07 entry, post-Stage-1 harvest) -- this
# wave's full worst-case (sec 12.6: every sec 12 item, both branches) costs
# under 3.82 GPU-h at 2x contingency, under 3% of the 135 GPU-h program
# ceiling. No budget concern; see the script's own module docstring for the
# full 1x/2x accounting this chain does not repeat.
#
# CRITICAL (sec 8.0's own "[LEARN] from Stage 1: run 1 died on an unshipped
# dependency" -- restated here so THIS chain's deploy does not repeat it).
# Full runtime-dependency CLOSURE (every file frozen_bias_gradflow_probe.py
# imports, transitively, at MODULE-IMPORT time -- verified by direct read of
# every file's own `import`/`from` lines this build session, not assumed):
#   frozen_bias_gradflow_probe.py   (this chain's own script)
#   lm_pretrain_rd.py               (DeltaNetLM/DeltaNetLMBlock/DeltaNetLMMixer,
#                                     apply_frozen_bias_blend, load_corpus,
#                                     get_batch, get_lr, set_and_log_tf32,
#                                     CORPUS_DIRS, DEFAULT_DATA_DIR,
#                                     FROZEN_BIAS_ARM_MODES,
#                                     FROZEN_BIAS_LAMBDA_PRIMARY, _MIN_KERNEL_T)
#   model_rd.py                     (lm_pretrain_rd.py's own import: _MIN_KERNEL_T,
#                                     _SAFE_D_STATE, _polar_via_eigh,
#                                     newton_schulz_orthogonalize)
#   rank_utils.py                   (lm_pretrain_rd.py's own import: effective_rank,
#                                     stable_rank; ALSO model_rd.py's own import)
#   key_anchoring.py                (lm_pretrain_rd.py's own import:
#                                     random_unit_rows_init, ANCHOR_INIT_SEED;
#                                     ALSO model_rd.py's own import)
#   hard_selectivity_rd.py          (lm_pretrain_rd.py's own import -- several names,
#                                     unused by this script's own cells but executed
#                                     at lm_pretrain_rd.py's own module-import time
#                                     regardless)
#   deltanet_core.py                (model_rd.py's own import: apply_state_power)
#   geo3_simulator.py                (key_anchoring.py's own import: newton_schulz)
#   rev7_threshold_derive.py        (key_anchoring.py's own import, "pure Python,
#                                     fla-free")
#   mech_schema.py                  (this chain's own script's import: wrap_exploratory)
# PLUS: the real `fla` package (pip-installed on this box; this script's own
# CPU fla-stub is smoke-only and never installed when the real package
# import succeeds) and the rebuilt corpus dir
# /data/deltanet_rd_data/reasoning_mix_eot_extended/{train.pt,val.pt,
# train_doc_offsets.pt,val_doc_offsets.pt,meta.json} (openr1-mix-ext,
# CORPUS_DIRS' own mapping) -- already present on this box (Stage 1 already
# trained/read from it). If ANY of the repo-file list above is missing from
# the scp'd/rsync'd copy on this box, step 2 below fails at Python import
# time, before any GPU time is spent -- verify the deploy's file list
# against this comment BEFORE launching, not after a crash.
#
# Same `set -euo pipefail` discipline every sibling chain in this dir uses
# (mech_stage1_chain.sh, mech_h4_chain.sh, keyanchor_confirm_chain.sh) --
# does NOT rely on `cmd 2>&1 | tee log && next_cmd` masking a crashed
# command's real exit status behind `tee`'s own (successful) one.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs results/mech_wave

# GPU assignment -- override via env var if the box's reservation map has
# changed before this actually launches (this build pass performs NO SSH/GPU
# execution; see the build report). Default GPU 2 mirrors mech_stage1_chain.sh's
# own MECH_STAGE1_GPU default (GPUs 0-1 rung-3, GPU 6 wave-1ext -- DO NOT
# TOUCH; GPUs 2-5,7 idle/reserved at Stage-1 launch time -- RE-VERIFY with
# nvidia-smi before this chain actually runs, per house rule and sec 12.7's
# own "re-verify the 12.0 pre-check ... immediately before any BUILD agent
# executes anything" standing constraint). This chain uses ONE GPU
# throughout (all 3 arms run sequentially, one process) -- cheap enough
# (~0.76 GPU-h total) that parallelizing across multiple GPUs isn't worth
# the added bookkeeping.
MECH_STAGE2_GPU=${MECH_STAGE2_GPU:-2}

# ---------------------------------------------------------------------------
# Step 1 -- mandatory Stage-2 smoke (sec 12.5's three-part spec). A failure
# here means the scp'd/rsync'd copy of this file (or one of its runtime
# dependencies, listed above) is stale/corrupt relative to what was audited
# -- stop before any real training cell.
# AUDIT MINOR-1 FIX (2026-07-07): pin the smoke to this wave's own GPU. On
# the box (real fla installed, CUDA available) the probe's auto-device picks
# cuda, so an unpinned smoke would land on GPU 0 -- rung-3's, DO NOT TOUCH.
# Pinning to $MECH_STAGE2_GPU also exercises the REAL kernel (strictly
# stronger than the local CPU-stub pass, per the audit's own note).
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$MECH_STAGE2_GPU $PY frozen_bias_gradflow_probe.py --smoke \
    2>&1 | tee logs/80_mech_stage2_smoke.log

# ---------------------------------------------------------------------------
# Step 2 -- the 3 real cells (Arm 1 off, Arm 2 per_token lambda=0.58, Arm 2'
# global lambda=0.58), 1 seed (0), 1 corpus (openr1-mix-ext), FULL 20,000
# steps each, sequential in ONE process on ONE GPU (sec 12.5's own "Output:
# one JSON per cell (or one combined)" -- this build picked "one combined").
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$MECH_STAGE2_GPU $PY frozen_bias_gradflow_probe.py \
    --corpus openr1-mix-ext --seed 0 --steps 20000 --arms off,per_token,global \
    --gradflow-cadence 100 \
    --out results/mech_wave/mech_stage2_gradflow_openr1-mix-ext_s0.json \
    2>&1 | tee logs/81_mech_stage2_gradflow_real.log

touch results/mech_wave/MECH_STAGE2_DONE
