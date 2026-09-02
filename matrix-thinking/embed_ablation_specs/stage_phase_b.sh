#!/bin/bash
# Phase-B gate for the embedding ablation (EMBEDDING_ABLATION_DESIGN.md §6, audit F2/MJ-2):
# wait until all 6 phase-A probes are in queue/completed, run --check-admission, and ONLY on exit 0
# copy the 24 phase-B specs from ~/embed_staging/specs into the queue. Otherwise write STOP.* and stage nothing.
set -u
Q=/home/nvidia/queue; ST=/home/nvidia/embed_staging; LOG=$ST/stage_phase_b.log
mkdir -p "$ST"; log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
log "watcher start; waiting for 6 phase-A probes"
while true; do
  [ -f "$ST/STOP" ] && { log "manual STOP; exiting"; exit 0; }
  nf=$(ls "$Q"/failed/064[0-5]_embed_ablation_probe_*.json 2>/dev/null | wc -l)
  [ "$nf" != "0" ] && { log "$nf probe(s) FAILED validity -> STOP.probe_failed"; touch "$ST/STOP.probe_failed"; exit 1; }
  nc=$(ls "$Q"/completed/064[0-5]_embed_ablation_probe_*.json 2>/dev/null | wc -l)
  [ "$nc" = "6" ] && break
  sleep 120
done
log "6 probes COMPLETED; running --check-admission"
cd /home/nvidia/embed_ablation && /home/nvidia/tdenv/bin/python3 embed_ablation_rd.py --check-admission --probe-results-dir /home/nvidia/embed_ablation/results/probes --intended-steps 2000 --intended-batch 64 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" != "0" ] && { log "ADMISSION FAIL (rc=$rc) -> nothing staged; re-derive STEPS_B per phase_B README"; touch "$ST/STOP.admission"; exit 1; }
n=$(ls "$ST"/specs/06[4-6][0-9]_embed_ablation_*.json 2>/dev/null | wc -l)
[ "$n" != "24" ] && { log "expected 24 specs, found $n -> STOP.spec_count"; touch "$ST/STOP.spec_count"; exit 1; }
cp "$ST"/specs/06*.json "$Q/pending/" && log "STAGED 24 phase-B specs: $(ls "$Q/pending" | grep -c embed_ablation)"; touch "$ST/STAGED"
