#!/usr/bin/env bash
GPU="$1"; shift
cd "/home/nvidia/ncr"
[ -f "/home/nvidia/ncr/results_phase2/STOP" ] && exit 3
for cell in "$@"; do
    arm="${cell%%:*}"; seed="${cell##*:}"
    CUDA_VISIBLE_DEVICES=$GPU /home/nvidia/tdenv/bin/python run_ncr.py --cell "$arm" --K 12 \
        --seed "$seed" --steps 80000 --outdir "/home/nvidia/ncr/results_phase2" \
        --stop-file "/home/nvidia/ncr/results_phase2/STOP" --rate-gpuh-80k 1.678 \
        >> "/home/nvidia/ncr/results_phase2/phase2_cell_${arm}_s${seed}.log" 2>&1 &
done
wait
