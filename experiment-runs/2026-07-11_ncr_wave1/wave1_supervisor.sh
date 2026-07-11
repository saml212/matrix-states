#!/usr/bin/env bash
log() { echo "[supervisor] $1 $(date -u +%FT%TZ)" >> "/home/nvidia/ncr/results_wave1/wave1_supervisor.log"; }
log "launch GPUs=6/7 steps=80000 rate=1.1185 shards: A=[ncr:0 ncr:2 ncr:4 loopedvec:1 loopedvec:3 fwm:0 fwm:2 fwm:4 cmlp:1 ] B=[ncr:1 ncr:3 loopedvec:0 loopedvec:2 loopedvec:4 fwm:1 fwm:3 cmlp:0 cmlp:2 ]"
while [ ! -f "/home/nvidia/ncr/results_wave1/STOP" ]; do
    bash "/home/nvidia/ncr/results_wave1/wave1_worker.sh" 6 ncr:0 ncr:2 ncr:4 loopedvec:1 loopedvec:3 fwm:0 fwm:2 fwm:4 cmlp:1  &
    PA=$!
    bash "/home/nvidia/ncr/results_wave1/wave1_worker.sh" 7 ncr:1 ncr:3 loopedvec:0 loopedvec:2 loopedvec:4 fwm:1 fwm:3 cmlp:0 cmlp:2  &
    PB=$!
    wait $PA $PB
    N=$(/home/nvidia/tdenv/bin/python - <<'PYEOF'
import glob, json
n = 0
for p in glob.glob("/home/nvidia/ncr/results_wave1/ncr_*_K8_s*.json"):
    try:
        if json.load(open(p)).get("status") == "COMPLETED": n += 1
    except Exception: pass
print(n)
PYEOF
)
    log "pass complete: $N/18 cells COMPLETED"
    if [ "$N" -eq 18 ]; then
        date -u +%FT%TZ > "/home/nvidia/ncr/results_wave1/WAVE1_DONE"
        log "WAVE1_DONE written"
        break
    fi
    sleep 15
done
log "supervisor exiting"
