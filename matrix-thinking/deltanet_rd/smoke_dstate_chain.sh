#!/bin/bash
# KEY_ANCHORING_SCALING_DRAFT.md sec 15 blocking gate -- runs the d_state
# kernel-safety smoke (smoke_dstate_kernel.py, shipped alongside) on ONE
# idle GPU. ~1 min. Exit codes: 0 = 80+96 kernel-safe; 3 = wave dead;
# 2 = positive-control failure (environment, no verdict).
set -euo pipefail
cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs
$PY smoke_dstate_kernel.py 2>&1 | tee logs/smoke_dstate_kernel.log
