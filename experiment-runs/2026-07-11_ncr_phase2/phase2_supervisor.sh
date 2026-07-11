#!/usr/bin/env bash
log() { echo "[supervisor] $1 $(date -u +%FT%TZ)" >> "/home/nvidia/ncr/results_phase2/phase2_supervisor.log"; }
log "launch GPUs=(2 3 4 5 6 7) K=12 steps=80000 rate=1.678"
while [ ! -f "/home/nvidia/ncr/results_phase2/STOP" ]; do
    PIDS=()
    bash "/home/nvidia/ncr/results_phase2/phase2_worker.sh" 2 ncr:0 ncr:1 ncr:2 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_phase2/phase2_worker.sh" 3 ncr:3 ncr:4 loopedvec:0 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_phase2/phase2_worker.sh" 4 loopedvec:1 loopedvec:2 loopedvec:3 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_phase2/phase2_worker.sh" 5 loopedvec:4 fwm:0 fwm:1 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_phase2/phase2_worker.sh" 6 fwm:2 fwm:3 fwm:4 & PIDS+=($!)
    bash "/home/nvidia/ncr/results_phase2/phase2_worker.sh" 7 cmlp:0 cmlp:1 cmlp:2 & PIDS+=($!)
    wait "${PIDS[@]}"
    N=$(/home/nvidia/tdenv/bin/python - <<'PYEOF'
import glob, json
n = 0
for p in glob.glob("/home/nvidia/ncr/results_phase2/ncr_*_K12_s*.json"):
    if ".axis_c_lock." in p: continue
    try:
        if json.load(open(p)).get("status") == "COMPLETED": n += 1
    except Exception: pass
print(n)
PYEOF
)
    log "pass complete: $N/18 cells COMPLETED"
    if [ "$N" -eq 18 ]; then
        date -u +%FT%TZ > "/home/nvidia/ncr/results_phase2/PHASE2_DONE"
        log "PHASE2_DONE written"
        break
    fi
    sleep 15
done
log "supervisor exiting"
