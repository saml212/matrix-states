#!/bin/bash
# KEY_ANCHORING confirmatory-wave chain (KEY_ANCHORING_DESIGN.md sec 9.5,
# 2026-07-06 keyanchor-confirm build). Confirmatory RE-run: candidate (d),
# K=32 seeds {0,1,2} (+ an optional K=16 seed-0 spot check) WITH
# drift_probe=True, closing sec 9.3's UNASSIGNABLE gap (item 5's pre-NS
# blend drift, item 6's table conditioning, sec 3.7's engaged_frac + h=1
# behavioral companion were never measured on the admitted Wave-1 cells)
# for the ALREADY-PUBLISHED h4 result (mean 0.613, 3/3 seeds clearing the
# >=0.5 bar) -- WITHOUT a new 20,000-step training decision. Same config as
# the original candidate-(d) K=32 cells otherwise (20,000 steps, n_iter=20,
# same sec 3.6 blinding gate -- these ARE anchor-arm cells).
#
# Cost: ~3 x 0.28 = ~0.9 GPU-h for the mandatory K=32 cells (+ a cheap K=16
# spot check) -- see run_deltanet_rd_exactness_sweep.py's
# keyanchor_confirm_manifest() docstring for the realized-cost caveat
# (drift_probe's per-checkpoint overhead is not yet folded into that
# anchor).
#
# UNLIKE experiment-runs/2026-07-05_keyanchor_wave/scripts/keyanchor_chain.sh
# (the historical archive of the ORIGINAL wave, left untouched as a
# faithful record of what actually ran), THIS script sets `pipefail` from
# the start and does NOT rely on `&&`-chaining commands piped through
# `tee` to propagate failure: that pattern is exactly how the original
# chain's own log_every crash (this build's fix, see
# keyanchor_drift_diagnostic.py's module docstring) went unnoticed --
# `cmd 2>&1 | tee logs/X.log` reports `tee`'s own (successful) exit status
# to a bare `&&`, silently discarding a crashed command's real one. `set -e`
# below stops this script the moment ANY line fails, with no `&&`
# bookkeeping required.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs

# GPUs 0-1 run rung-3, GPU 6 runs wave-1ext -- DO NOT TOUCH (standing
# reservation at build time). GPUs 2-5 and 7 are idle and reserved for
# THIS wave; 4 cells x 1 GPU each (GPUs 2-5) leaves GPU 7 as spare
# capacity, never claimed by default. Override via env vars if the
# reservation changes before this actually launches (it does NOT launch
# from this build session -- see the build report: build + CPU-verify
# only, no GPU work performed here).
KEYANCHOR_CONFIRM_GPUS=${KEYANCHOR_CONFIRM_GPUS:-4}
KEYANCHOR_CONFIRM_GPU_OFFSET=${KEYANCHOR_CONFIRM_GPU_OFFSET:-2}
KEYANCHOR_CONFIRM_DIAG_GPU=${KEYANCHOR_CONFIRM_DIAG_GPU:-$KEYANCHOR_CONFIRM_GPU_OFFSET}

# Pre-launch CPU gate (Gate 1's read, task-directed): the FIXED
# keyanchor_drift_diagnostic.py must run clean BEFORE any confirm cell
# dispatches. This is the ONE step in this chain that itself needs a GPU
# (bind()/chunk_delta_rule has no CPU path); the gate-CHECK that consumes
# its output (gate_keyanchor_drift_diag, inside the sweep script below) is
# pure CPU file-read, same as gate_geo3_drift's existing pattern.
CUDA_VISIBLE_DEVICES=$KEYANCHOR_CONFIRM_DIAG_GPU $PY keyanchor_drift_diagnostic.py --k 16 32 \
    --out results/deltanet_rd_exactness/keyanchor_drift/wave_neg1_drift.json \
    2>&1 | tee logs/08_drift_diag_fixed.log

$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-confirm \
    --gpus "$KEYANCHOR_CONFIRM_GPUS" --gpu-offset "$KEYANCHOR_CONFIRM_GPU_OFFSET" --per-gpu 1 \
    --out-dir results/deltanet_rd_exactness \
    2>&1 | tee logs/09_wave_keyanchor_confirm.log

touch results/deltanet_rd_exactness/KEYANCHOR_CONFIRM_CHAIN_DONE
