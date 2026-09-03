#!/bin/bash
# Phase-B gate, second attempt (2026-09-03): the 500-step probes failed admission on flat_M (T=1 rising over the
# last three evals while train loss fell). Six 1000-step probes (0670-0675, own results dir probes_1000, own ckpt
# dir) re-run the SAME admission check; ONLY on exit 0 are the unchanged 24 phase-B specs (0646-0669, STEPS_B=2000)
# copied into the queue. Otherwise STOP.admission_1000 and nothing is staged (cluster 4 then reads "drop").
set -u
Q=/home/nvidia/queue; ST=/home/nvidia/embed_staging; LOG=$ST/stage_phase_b_1000.log
mkdir -p "$ST"; log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
log "watcher start; waiting for 6 phase-A 1000-step probes (0670-0675)"
while true; do
  [ -f "$ST/STOP" ] && { log "manual STOP; exiting"; exit 0; }
  nf=$(ls "$Q"/failed/067[0-5]_embed_ablation_probe_*.json 2>/dev/null | wc -l)
  [ "$nf" != "0" ] && { log "$nf probe(s) FAILED validity -> STOP.probe_failed_1000"; touch "$ST/STOP.probe_failed_1000"; exit 1; }
  nc=$(ls "$Q"/completed/067[0-5]_embed_ablation_probe_*.json 2>/dev/null | wc -l)
  [ "$nc" = "6" ] && break
  sleep 120
done
log "6 probes COMPLETED; running --check-admission on probes_1000"
cd /home/nvidia/embed_ablation && /home/nvidia/tdenv/bin/python3 embed_ablation_rd.py --check-admission --probe-results-dir /home/nvidia/embed_ablation/results/probes_1000 --intended-steps 2000 --intended-batch 64 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" != "0" ] && { log "ADMISSION FAIL (rc=$rc) -> nothing staged; cluster 4 reads DROP unless the author overrides"; touch "$ST/STOP.admission_1000"; exit 1; }
n=$(ls "$ST"/specs/06[4-6][0-9]_embed_ablation_*.json 2>/dev/null | wc -l)
[ "$n" != "24" ] && { log "expected 24 specs, found $n -> STOP.spec_count"; touch "$ST/STOP.spec_count"; exit 1; }
cp "$ST"/specs/06*.json "$Q/pending/" && log "STAGED 24 phase-B specs: $(ls "$Q/pending" | grep -c embed_ablation)"; touch "$ST/STAGED_1000"
