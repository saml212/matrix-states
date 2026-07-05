#!/bin/bash
# KEY-ANCHORING wave chain (design Rev 5 ca308d6, build e99a8f7+46ced70, audited).
# Sequential prefix on GPU 7 (free since H_e finished); final 12-cell keyanchor
# stage waits for wave-2 to release GPUs 0-5. One-shot &&-gated stages (audited
# convention): any failure stops the chain and surfaces. Stages are is_done
# resume-safe; rerun this script to resume. Kill: tmux kill-session -t keyanchor_wave.
cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
mkdir -p logs
CUDA_VISIBLE_DEVICES=7 $PY run_deltanet_rd.py --smoke 2>&1 | tee logs/00_model17_smoke.log && \
$PY smoke_key_anchoring.py 2>&1 | tee logs/01_smoke_key_anchoring.log && \
$PY gate2_construction_test.py 2>&1 | tee logs/02_gate2.log && \
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-neg1 --gpus 1 --gpu-offset 7 \
    --out-dir results/deltanet_rd_exactness 2>&1 | tee logs/03_wave_neg1.log && \
$PY keyanchor_drift_diagnostic.py --k 16 32 \
    --out results/deltanet_rd_exactness/keyanchor_drift/wave_neg1_drift.json 2>&1 | tee logs/04_drift_diag.log && \
$PY run_deltanet_rd_exactness_sweep.py --wave ref --gpus 1 --gpu-offset 7 \
    --out-dir results/deltanet_rd_exactness 2>&1 | tee logs/05_wave_ref.log && \
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor-bands \
    --out-dir results/deltanet_rd_exactness 2>&1 | tee logs/06_bands_pinned.log && \
{ echo "waiting for wave-2 to release GPUs 0-5..."; \
  until [ -f results/lm_rd_trackc/wave2/ALL_DONE ]; do sleep 60; done; } && \
$PY run_deltanet_rd_exactness_sweep.py --wave keyanchor --gpus 6 --gpu-offset 0 \
    --out-dir results/deltanet_rd_exactness 2>&1 | tee logs/07_wave_keyanchor.log && \
touch results/deltanet_rd_exactness/KEYANCHOR_CHAIN_DONE
