#!/usr/bin/env bash
GPU="$1"; shift
cd "/home/nvidia/ncr"
for cell in "$@"; do
    [ -f "/home/nvidia/ncr/results_earlyln_scale/STOP" ] && exit 3
    K="${cell%%:*}"; seed="${cell##*:}"
    CUDA_VISIBLE_DEVICES=$GPU /home/nvidia/tdenv/bin/python ncr_earlyln_scale.py --cell --K "$K" \
        --seed "$seed" --steps 80000 --outdir "/home/nvidia/ncr/results_earlyln_scale" \
        --stop-file "/home/nvidia/ncr/results_earlyln_scale/STOP" --ceiling-gpuh 0.75 \
        >> "/home/nvidia/ncr/results_earlyln_scale/earlyln_cell_K${K}_s${seed}.log" 2>&1
done
