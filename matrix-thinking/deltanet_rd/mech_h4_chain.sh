#!/bin/bash
# FROZEN_BIAS_LM_DESIGN.md sec 12.3.3 -- H4 checkpoint parameter-diff, REAL
# run against the frozen-bias LM checkpoints on the box (CPU-only: six
# torch.load map_location="cpu" calls + norm/cosine arithmetic; touches NO
# GPU -- safe to run while rung-3 owns GPUs 0-1).
#
# The registered cell (sec 12.3.3's own recommendation, matching the script
# defaults): corpus=openr1-mix-ext, seed=0, step_early=1000, step_late=20000.
# Exploratory tier -- the script's own wrap_exploratory() stamps the output;
# no headline verdict is derivable from this run alone.
#
# mech_h4_paramdiff.py exits 2 (registered-not-run) if any of the 6
# checkpoints is missing -- under set -e that stops this chain loudly.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs results/mech_wave

# Self-test first (same-file synthetic gate, exact key-naming convention) --
# a failure here means the scp'd copy is stale/corrupt, stop before real run.
$PY mech_h4_paramdiff.py --self-test 2>&1 | tee logs/60_mech_h4_selftest.log

$PY mech_h4_paramdiff.py \
    --ckpt-base-dir /data/deltanet_rd_frozenbias_ckpts \
    --corpus openr1-mix-ext --seed 0 \
    --step-early 1000 --step-late 20000 \
    --out results/mech_wave/mech_h4_paramdiff_results.json \
    2>&1 | tee logs/61_mech_h4_paramdiff_real.log

touch results/mech_wave/MECH_H4_DONE
