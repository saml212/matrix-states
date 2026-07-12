#!/usr/bin/env bash
log() { echo "[supervisor] $1 $(date -u +%FT%TZ)" >> "/home/nvidia/ncr/results_k12ext/k12ext_supervisor.log"; }
log "launch GPUs=(2 3 4 5 6) K=12 steps=80000 rate=1.678 seeds=5,6,7,8,9 (ncr arm only)"
while [ ! -f "/home/nvidia/ncr/results_k12ext/STOP" ]; do
    PIDS=()
    bash "/home/nvidia/ncr/results_k12ext/k12ext_worker.sh" 2 ncr:5 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_k12ext/k12ext_worker.sh" 3 ncr:6 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_k12ext/k12ext_worker.sh" 4 ncr:7 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_k12ext/k12ext_worker.sh" 5 ncr:8 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_k12ext/k12ext_worker.sh" 6 ncr:9 & PIDS+=($!)
    wait "${PIDS[@]}"
    N=$(/home/nvidia/tdenv/bin/python - <<'PYEOF'
import glob, json
n = 0
for p in glob.glob("/home/nvidia/ncr/results_k12ext/ncr_ncr_K12_s*.json"):
    if ".axis_c_lock." in p: continue
    try:
        if json.load(open(p)).get("status") == "COMPLETED": n += 1
    except Exception: pass
print(n)
PYEOF
)
    log "pass complete: $N/5 cells COMPLETED"
    if [ "$N" -eq 5 ]; then
        date -u +%FT%TZ > "/home/nvidia/ncr/results_k12ext/K12EXT_DONE"
        log "K12EXT_DONE written"
        break
    fi
    sleep 15
done
log "supervisor exiting"
