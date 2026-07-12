#!/usr/bin/env bash
cd "/home/nvidia/ncr"
while [ ! -f "/home/nvidia/ncr/results_opbank_recover/STOP" ]; do
    for arm in baseline warmup earlyln curriculum; do
        case $arm in
            baseline) gpu=0;; warmup) gpu=1;; earlyln) gpu=6;; curriculum) gpu=7;;
        esac
        CUDA_VISIBLE_DEVICES=$gpu /home/nvidia/tdenv/bin/python ncr_opbank_recover.py --cell $arm --seed 0 \
            --steps 80000 --train-batch 256 --ceiling-gpuh 2.0 \
            --outdir "/home/nvidia/ncr/results_opbank_recover" --stop-file "/home/nvidia/ncr/results_opbank_recover/STOP" --device cuda \
            >> "/home/nvidia/ncr/results_opbank_recover/recover_cell_${arm}.log" 2>&1 &
    done
    wait
    # DONE iff all 4 cells wrote a COMPLETED json
    n_done=$(/home/nvidia/tdenv/bin/python -c "import json,glob; print(sum(1 for f in glob.glob('/home/nvidia/ncr/results_opbank_recover/ncropbank_recover_*.json') if json.load(open(f)).get('status')=='COMPLETED'))")
    echo "[recover] round complete, $n_done/4 COMPLETED at $(date -u +%FT%TZ)" >> "/home/nvidia/ncr/results_opbank_recover/recover_supervisor.log"
    if [ "$n_done" -ge 4 ]; then touch "/home/nvidia/ncr/results_opbank_recover/DONE"; break; fi
    sleep 15
done
echo '[recover] done' >> "/home/nvidia/ncr/results_opbank_recover/recover_supervisor.log"
