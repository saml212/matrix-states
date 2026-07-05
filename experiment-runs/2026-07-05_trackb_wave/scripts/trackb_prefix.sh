#!/bin/bash
# TRACK B prefix (design Rev 3 8ab089d, build 1a1ea82+c8e4b4c, cleared round 3):
# GPU smokes [14]/[15]/[16] then the single-GPU waves on GPU 7 while wave-2
# still holds 0-5. Factorial waves (1/2/3/4r) launch separately after rung-3.
# &&-gated one-shot; is_done resume-safe; kill: tmux kill-session -t trackb_wave.
cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
CKPTS=$(ls results/lm_rd/waveC/checkpoints/*_step6103.pt)
CUDA_VISIBLE_DEVICES=7 $PY lm_pretrain_rd.py --smoke 2>&1 | tee logs/tb_00_gpu_smoke.log && \
$PY run_trackb_wave.py --wave neg1 --gpus 1 --gpu-offset 7 \
    --out-dir results/trackb 2>&1 | tee logs/tb_01_neg1.log && \
$PY run_trackb_wave.py --wave cell1probe --gpus 1 --gpu-offset 7 \
    --cell1-checkpoints $CKPTS \
    --out-dir results/trackb 2>&1 | tee logs/tb_02_cell1probe.log && \
$PY run_trackb_wave.py --wave bands-pinned \
    --out-dir results/trackb 2>&1 | tee logs/tb_03_bands.log && \
touch results/trackb/TRACKB_PREFIX_DONE
