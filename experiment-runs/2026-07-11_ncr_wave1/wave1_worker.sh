#!/usr/bin/env bash
GPU="$1"; shift
cd "/home/nvidia/ncr"
CELLS=("$@")
for ((i=0; i<${#CELLS[@]}; i+=3)); do
    [ -f "/home/nvidia/ncr/results_wave1/STOP" ] && exit 3
    for cell in "${CELLS[@]:i:3}"; do
        arm="${cell%%:*}"; seed="${cell##*:}"
        CUDA_VISIBLE_DEVICES=$GPU /home/nvidia/tdenv/bin/python run_ncr.py --cell "$arm" --K 8 \
            --seed "$seed" --steps 80000 --outdir "/home/nvidia/ncr/results_wave1" \
            --stop-file "/home/nvidia/ncr/results_wave1/STOP" --rate-gpuh-80k 1.1185 \
            >> "/home/nvidia/ncr/results_wave1/wave1_cell_${arm}_s${seed}.log" 2>&1 &
    done
    wait
done
