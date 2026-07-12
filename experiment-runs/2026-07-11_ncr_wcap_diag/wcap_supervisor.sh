#!/usr/bin/env bash
log() { echo "[supervisor] $1 $(date -u +%FT%TZ)" >> "/home/nvidia/ncr/results_wcap_diag/wcap_supervisor.log"; }
log "launch GPUs=(2 3 4 5) steps=80000 subdir=results_wcap_diag cells=spareK14:14:16:64:2.832 spareK15:15:16:64:3.027 condA_K16:16:32:64:3.527 condB_K16:16:32:128:13.424"
EXPECT=(ncr_ncr_K14_s0 ncr_ncr_K15_s0 ncr_ncr_K16_d32_h64_s0 ncr_ncr_K16_d32_h128_s0)
while [ ! -f "/home/nvidia/ncr/results_wcap_diag/STOP" ]; do
    PIDS=()
    bash "/home/nvidia/ncr/results_wcap_diag/wcap_worker.sh" 2 spareK14 14 16 64 2.832 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_wcap_diag/wcap_worker.sh" 3 spareK15 15 16 64 3.027 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_wcap_diag/wcap_worker.sh" 4 condA_K16 16 32 64 3.527 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_wcap_diag/wcap_worker.sh" 5 condB_K16 16 32 128 13.424 & PIDS+=($!)
    wait "${PIDS[@]}"
    N=0
    for e in "${EXPECT[@]}"; do
        f="/home/nvidia/ncr/results_wcap_diag/$e.json"
        if [ -f "$f" ] && /home/nvidia/tdenv/bin/python -c "import json,sys; sys.exit(0 if json.load(open('$f')).get('status')=='COMPLETED' else 1)" 2>/dev/null; then
            N=$((N+1))
        fi
    done
    log "pass complete: $N/4 cells COMPLETED"
    if [ "$N" -eq 4 ]; then
        date -u +%FT%TZ > "/home/nvidia/ncr/results_wcap_diag/WCAP_DIAG_DONE"
        log "WCAP_DIAG_DONE written"
        break
    fi
    sleep 15
done
log "supervisor exiting"
