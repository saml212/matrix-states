#!/usr/bin/env bash
cd "/home/nvidia/ncr"
while [ ! -f "/home/nvidia/ncr/results/STOP" ]; do
    for arm in ncr loopedvec fwm; do
        CUDA_VISIBLE_DEVICES=6 /home/nvidia/tdenv/bin/python run_ncr.py --cell $arm --K 8 --seed 0             --steps 40000 --outdir "/home/nvidia/ncr/results" --stop-file "/home/nvidia/ncr/results/STOP"             --rate-gpuh-80k 4.8             >> "/home/nvidia/ncr/results/phase0_cell_${arm}.log" 2>&1 &
    done
    wait
    CUDA_VISIBLE_DEVICES=6 /home/nvidia/tdenv/bin/python run_ncr.py --phase0 --steps 40000         --outdir "/home/nvidia/ncr/results" --stop-file "/home/nvidia/ncr/results/STOP"         >> "/home/nvidia/ncr/results/phase0_supervisor.log" 2>&1
    rc=$?
    echo "[supervisor] gate pass exited rc=$rc at $(date -u +%FT%TZ)"         >> "/home/nvidia/ncr/results/phase0_supervisor.log"
    if [ $rc -eq 0 ] || [ $rc -eq 3 ]; then break; fi
    sleep 15
done
echo '[supervisor] done' >> "/home/nvidia/ncr/results/phase0_supervisor.log"
