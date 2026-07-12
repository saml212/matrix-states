#!/usr/bin/env bash
cd "/home/nvidia/ncr"
while [ ! -f "/home/nvidia/ncr/results_opbank_recal/STOP" ]; do
    CUDA_VISIBLE_DEVICES=5 /home/nvidia/tdenv/bin/python run_ncr_opbank.py --cell ncr-bank --seed 0 \
        --steps 80000 --train-batch 256 --ceiling-gpuh 3.0 \
        --outdir "/home/nvidia/ncr/results_opbank_recal" --stop-file "/home/nvidia/ncr/results_opbank_recal/STOP" --device cuda \
        >> "/home/nvidia/ncr/results_opbank_recal/recal_cell_ncr-bank.log" 2>&1
    rc=$?
    echo "[recal] cell exited rc=$rc at $(date -u +%FT%TZ)" >> "/home/nvidia/ncr/results_opbank_recal/recal_supervisor.log"
    if [ $rc -eq 0 ] || [ $rc -eq 3 ]; then touch "/home/nvidia/ncr/results_opbank_recal/DONE"; break; fi
    sleep 15
done
echo '[recal] done' >> "/home/nvidia/ncr/results_opbank_recal/recal_supervisor.log"
