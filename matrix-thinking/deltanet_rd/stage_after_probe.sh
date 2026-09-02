#!/bin/bash
# Gate for the recall baseline-strengthening sweep (HEAD_TO_HEAD_DEMO_DESIGN.md §1.46):
# wait for the 0599 CUDA probe to COMPLETE, re-price with strengthen_reprice.py, and only on
# `REPRICE: PASS` copy the 27 staged specs from ~/strengthen_staging/ into the queue.
# Any other outcome writes ~/strengthen_staging/STOP.<reason> and exits without staging.
set -u
Q=/home/nvidia/queue; ST=/home/nvidia/strengthen_staging; LOG=$ST/stage_after_probe.log
RD=/home/nvidia/chapter2/deltanet_rd; PROBE_DIR=$RD/results/h2h_rung1/strengthen_probe
mkdir -p "$ST"; log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
log "watcher start; waiting for 0599 to complete"
while true; do
  [ -f "$ST/STOP" ] && { log "manual STOP present; exiting"; exit 0; }
  if [ -f "$Q/failed/0599_h2h_strengthen_probe_C2.json" ]; then log "0599 FAILED validity -> STOP.probe_failed"; touch "$ST/STOP.probe_failed"; exit 1; fi
  [ -f "$Q/completed/0599_h2h_strengthen_probe_C2.json" ] && break
  sleep 60
done
log "0599 COMPLETED; re-pricing"
if [ ! -f "$RD/strengthen_reprice.py" ]; then log "strengthen_reprice.py missing -> STOP.no_reprice"; touch "$ST/STOP.no_reprice"; exit 1; fi
cd "$RD" && /home/nvidia/tdenv/bin/python3 strengthen_reprice.py --probe-dir "$PROBE_DIR" --joblog "$Q/logs/0599_h2h_strengthen_probe_C2.log" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" != "0" ]; then log "REPRICE STOP (rc=$rc) -> nothing staged"; touch "$ST/STOP.reprice"; exit 1; fi
n=$(ls "$ST"/specs/06[0-2][0-9]_h2h_strengthen_*.json 2>/dev/null | wc -l)
if [ "$n" != "27" ]; then log "expected 27 specs in $ST/specs, found $n -> STOP.spec_count"; touch "$ST/STOP.spec_count"; exit 1; fi
cp "$ST"/specs/06*.json "$Q/pending/" && log "STAGED 27 specs into pending: $(ls "$Q/pending" | grep -c h2h_strengthen)"
touch "$ST/STAGED"
