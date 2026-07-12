#!/usr/bin/env bash
GPU="$1"; LABEL="$2"; K="$3"; D="$4"; H="$5"; RATE="$6"
cd "/home/nvidia/ncr"
[ -f "/home/nvidia/ncr/results_wcap_diag/STOP" ] && exit 3
CUDA_VISIBLE_DEVICES=$GPU /home/nvidia/tdenv/bin/python run_ncr.py --cell ncr --K "$K" --d "$D" --h "$H" \
    --seed 0 --steps 80000 --outdir "/home/nvidia/ncr/results_wcap_diag" \
    --stop-file "/home/nvidia/ncr/results_wcap_diag/STOP" --rate-gpuh-80k "$RATE" \
    >> "/home/nvidia/ncr/results_wcap_diag/wcap_cell_${LABEL}.log" 2>&1
