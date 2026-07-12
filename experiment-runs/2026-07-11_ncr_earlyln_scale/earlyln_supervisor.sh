#!/usr/bin/env bash
log() { echo "[supervisor] $1 $(date -u +%FT%TZ)" >> "/home/nvidia/ncr/results_earlyln_scale/earlyln_supervisor.log"; }
log "launch GPUs=(0 1 2 3 4 5 6 7) steps=80000 subdir=results_earlyln_scale cells=14:0 14:1 14:2 14:3 15:0 15:1 15:2 15:3 16:0 16:1 16:2 16:3 24:0 24:1 24:2 24:3"
while [ ! -f "/home/nvidia/ncr/results_earlyln_scale/STOP" ]; do
    PIDS=()
    bash "/home/nvidia/ncr/results_earlyln_scale/earlyln_worker.sh" 0 14:0 14:1 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_earlyln_scale/earlyln_worker.sh" 1 14:2 14:3 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_earlyln_scale/earlyln_worker.sh" 2 15:0 15:1 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_earlyln_scale/earlyln_worker.sh" 3 15:2 15:3 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_earlyln_scale/earlyln_worker.sh" 4 16:0 16:1 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_earlyln_scale/earlyln_worker.sh" 5 16:2 16:3 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_earlyln_scale/earlyln_worker.sh" 6 24:0 24:1 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_earlyln_scale/earlyln_worker.sh" 7 24:2 24:3 & PIDS+=($!)
    wait "${PIDS[@]}"
    N=$(/home/nvidia/tdenv/bin/python - <<PYEOF
import glob, json
n = 0
for p in glob.glob("/home/nvidia/ncr/results_earlyln_scale/earlyln_K*_s*.json"):
    if ".axis_c_lock." in p: continue
    try:
        if json.load(open(p)).get("status") == "COMPLETED": n += 1
    except Exception: pass
print(n)
PYEOF
    )
    log "pass complete: $N/16 cells COMPLETED"
    if [ "$N" -eq 16 ]; then
        date -u +%FT%TZ > "/home/nvidia/ncr/results_earlyln_scale/EARLYLN_SCALE_DONE"
        log "EARLYLN_SCALE_DONE written"
        break
    fi
    sleep 15
done
log "supervisor exiting"
