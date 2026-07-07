#!/bin/bash
# FROZEN_BIAS_LM_DESIGN.md sec 12.4/12.4.1 -- Stage 1 REAL run: the primary
# H1/H5 mechanism-wave statistic, `repeat_excess`, on already-existing
# frozen-bias LM checkpoints (Arm 1 "off", Arm 2 "per_token", Arm 2' "global").
# Gated on sec 12.3.4 Stage 0.5 (already RUN and PASSED, sec 12.10.1 --
# step 1 below re-runs it as a same-file freshness gate before any real
# checkpoint pass, same discipline mech_h4_chain.sh already uses for its own
# --self-test-first step).
#
# Three steps:
#   1. frozen_bias_token_identity_probe.py --self-test (pure CPU, no GPU,
#      no checkpoints -- pushes sec 12.3.4's pinned constructions through
#      THIS repo's own episode-grained plumbing, sec 12.9 item 4).
#   2. Descriptive pass (sec 12.4): seed 0, BOTH corpora trained on
#      (openr1-mix-ext, wikitext-mix-ext), 3 arms, step 20000 only -- 6
#      checkpoint-passes total, 2 script invocations (one per corpus, this
#      script's own "one JSON per (corpus, seed) invocation" contract).
#   3. Trajectory sub-study (sec 12.4.1): openr1-mix-ext ONLY, seed 0, 3
#      arms, steps {1000,5000,10000,15000,20000} -- 15 checkpoint-passes,
#      reusing already-existing checkpoints (zero new training).
#
# Cost: step 2 ~0.21 GPU-h (6 passes x ~0.0348 GPU-h/pass) + step 3 ~0.52
# GPU-h (15 passes) ~= 0.73 GPU-h total. Frozen-bias LM program ledger:
# 6.9/135 GPU-h committed so far (well under the mechanism-wave's own
# sec 12.6 budget-summary ceiling, which puts this ENTIRE wave's most
# expensive plausible path at 1.91/3.82 GPU-h, 1x/2x contingency).
#
# Every output JSON is stamped via mech_schema.wrap_exploratory
# ("exploratory-mechanism-wave -- NOT a confirmatory bar", sec 12.3.1) --
# nothing here is a headline claim, sec 12.9 items 1-2's correlational
# ceiling applies throughout.
#
# Same `set -euo pipefail` discipline keyanchor_confirm_chain.sh's own
# header explains: does NOT rely on `cmd 2>&1 | tee log && next_cmd`
# masking a crashed command's real exit status behind `tee`'s own
# (successful) one -- `set -e` stops this script the instant ANY line
# fails, no `&&`-chaining bookkeeping required.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs results/mech_wave

# GPU assignment -- override via env var if the box's reservation map has
# changed before this actually launches (this build pass performs NO SSH/GPU
# execution; see the build report). Default GPU 2 mirrors keyanchor_confirm_
# chain.sh's own idle-GPU assumption at build time (GPUs 0-1 rung-3, GPU 6
# wave-1ext -- DO NOT TOUCH; GPUs 2-5,7 idle/reserved). This chain uses ONE
# GPU throughout (both corpora/steps run sequentially on it) -- cheap enough
# (~0.73 GPU-h total) that parallelizing across multiple GPUs isn't worth the
# added bookkeeping.
MECH_STAGE1_GPU=${MECH_STAGE1_GPU:-2}

# ---------------------------------------------------------------------------
# Step 1 -- self-test gate (pure CPU, no GPU touched). A failure here means
# the scp'd copy of this file is stale/corrupt relative to what was audited
# -- stop before any real checkpoint pass.
# ---------------------------------------------------------------------------
$PY frozen_bias_token_identity_probe.py --self-test \
    2>&1 | tee logs/70_mech_stage1_selftest.log

# Re-confirm the Stage 0.5 gate this same session (sec 12.3.4's own "must be
# RUN and PASSED" requirement, sec 12.10.1 recorded PASSED already -- this is
# a freshness re-check, not a first run) -- also pure CPU.
$PY mech_stage05_selftests.py \
    2>&1 | tee logs/71_mech_stage05_selftests_recheck.log

# ---------------------------------------------------------------------------
# Step 2 -- descriptive pass (sec 12.4): seed 0, step 20000 only, 3 arms,
# BOTH corpora -- one script invocation per corpus (this script's own "one
# JSON per (corpus, seed) invocation" contract), 6 checkpoint-passes total.
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$MECH_STAGE1_GPU $PY frozen_bias_token_identity_probe.py \
    --ckpt-base-dir /data/deltanet_rd_frozenbias_ckpts \
    --corpus openr1-mix-ext --seed 0 --steps 20000 --arms off,per_token,global \
    --out results/mech_wave/mech_stage1_descriptive_openr1-mix-ext_s0.json \
    2>&1 | tee logs/72_mech_stage1_descriptive_openr1.log

CUDA_VISIBLE_DEVICES=$MECH_STAGE1_GPU $PY frozen_bias_token_identity_probe.py \
    --ckpt-base-dir /data/deltanet_rd_frozenbias_ckpts \
    --corpus wikitext-mix-ext --seed 0 --steps 20000 --arms off,per_token,global \
    --out results/mech_wave/mech_stage1_descriptive_wikitext-mix-ext_s0.json \
    2>&1 | tee logs/73_mech_stage1_descriptive_wikitext.log

# ---------------------------------------------------------------------------
# Step 3 -- trajectory sub-study (sec 12.4.1): openr1-mix-ext ONLY, seed 0, 3
# arms, steps {1000,5000,10000,15000,20000} -- 15 checkpoint-passes, all 400
# checkpoints already exist at 1,000-step spacing (zero new training). The
# script's own trajectory_ambiguity_check_sec_12_4_1_step2 block (computed
# mechanically whenever >=2 steps are present) reads directly off this run's
# output -- if it reports ambiguous=true for either comparison, step 3's own
# densification fallback (steps 2000/3000/4000, sec 12.4.1 step 3) is a
# follow-up invocation with --steps extended accordingly, NOT automated by
# this chain (a human/orchestrator decision gate, per sec 12.4.1 step 3's own
# "densify for free... before concluding anything is genuinely ambiguous").
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=$MECH_STAGE1_GPU $PY frozen_bias_token_identity_probe.py \
    --ckpt-base-dir /data/deltanet_rd_frozenbias_ckpts \
    --corpus openr1-mix-ext --seed 0 --steps 1000,5000,10000,15000,20000 \
    --arms off,per_token,global \
    --out results/mech_wave/mech_stage1_trajectory_openr1-mix-ext_s0.json \
    2>&1 | tee logs/74_mech_stage1_trajectory_openr1.log

touch results/mech_wave/MECH_STAGE1_DONE
