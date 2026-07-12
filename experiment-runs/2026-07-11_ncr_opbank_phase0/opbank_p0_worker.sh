#!/usr/bin/env bash
cd "/home/nvidia/ncr"
while [ ! -f "/home/nvidia/ncr/results_opbank_phase0/STOP" ]; do
    CUDA_VISIBLE_DEVICES=2 /home/nvidia/tdenv/bin/python run_ncr_opbank.py --cell ncr-bank --seed 0 \
        --steps 20000 --outdir "/home/nvidia/ncr/results_opbank_phase0" --stop-file "/home/nvidia/ncr/results_opbank_phase0/STOP" \
        --ceiling-gpuh 2.0 --device cuda \
        >> "/home/nvidia/ncr/results_opbank_phase0/opbank_p0_cell_ncr-bank.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=3 /home/nvidia/tdenv/bin/python run_ncr_opbank.py --cell loopedvec-bank --seed 0 \
        --steps 20000 --outdir "/home/nvidia/ncr/results_opbank_phase0" --stop-file "/home/nvidia/ncr/results_opbank_phase0/STOP" \
        --ceiling-gpuh 2.0 --device cuda \
        >> "/home/nvidia/ncr/results_opbank_phase0/opbank_p0_cell_loopedvec-bank.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=4 /home/nvidia/tdenv/bin/python run_ncr_opbank.py --cell fwm-bank --seed 0 \
        --steps 20000 --outdir "/home/nvidia/ncr/results_opbank_phase0" --stop-file "/home/nvidia/ncr/results_opbank_phase0/STOP" \
        --ceiling-gpuh 2.0 --device cuda \
        >> "/home/nvidia/ncr/results_opbank_phase0/opbank_p0_cell_fwm-bank.log" 2>&1 &
    wait
    CUDA_VISIBLE_DEVICES=2 /home/nvidia/tdenv/bin/python run_ncr_opbank.py --phase0 --steps 20000 \
        --outdir "/home/nvidia/ncr/results_opbank_phase0" --stop-file "/home/nvidia/ncr/results_opbank_phase0/STOP" --device cuda \
        >> "/home/nvidia/ncr/results_opbank_phase0/opbank_p0_supervisor.log" 2>&1
    rc=$?
    echo "[supervisor] gate pass exited rc=$rc at $(date -u +%FT%TZ)" \
        >> "/home/nvidia/ncr/results_opbank_phase0/opbank_p0_supervisor.log"
    if [ $rc -eq 0 ] || [ $rc -eq 3 ]; then
        touch "/home/nvidia/ncr/results_opbank_phase0/DONE"
        break
    fi
    sleep 15
done
echo '[supervisor] done' >> "/home/nvidia/ncr/results_opbank_phase0/opbank_p0_supervisor.log"
