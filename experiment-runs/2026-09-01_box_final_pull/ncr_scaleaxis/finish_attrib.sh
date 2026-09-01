#!/usr/bin/env bash
# Waits for the 4 in-flight attribution cells, re-adjudicates them with the
# CORRECTED validity_check, promotes failed->completed ONLY on a clean pass,
# then battery- and depth-scores them at ckpt_step 40000.
set -u
cd /home/nvidia/ncr_scaleaxis
PY=/home/nvidia/tdenv/bin/python3
CELLS="K16_compB_s0 K16_compB_s1 K24_compB_s0 K24_compB_s1"

echo "=== waiting for the 4 in-flight cells ==="
while :; do
  n=$("$PY" - <<'PY'
import json
n=0
for c in ("K16_compB_s0","K16_compB_s1","K24_compB_s0","K24_compB_s1"):
    try:
        d=json.load(open(f"/ephemeral/scaleaxis/attribution/results/attrib40k_{c}.json"))
        if d.get("status")=="COMPLETED" and d.get("step")==40000: n+=1
    except Exception: pass
print(n)
PY
)
  echo "$(date -u +%H:%M:%S) completed=$n/4"
  [ "$n" = "4" ] && break
  sleep 120
done
echo "=== all 4 landed; queue state before re-adjudication ==="
for d in claimed failed completed; do echo "  $d: $(ls ~/queue/$d 2>/dev/null | wc -l)"; done

echo "=== RE-ADJUDICATION (corrected check on artifact), dry run ==="
"$PY" readjudicate.py --cells K16_compB K24_compB
echo "=== RE-ADJUDICATION with promotion (promotes ONLY on clean pass) ==="
"$PY" readjudicate.py --cells K16_compB K24_compB --promote
echo "=== queue state after ==="
for d in claimed failed completed; do echo "  $d: $(ls ~/queue/$d 2>/dev/null | wc -l)"; done

echo "=== scoring the 4 at their extended checkpoints ==="
"$PY" build_attrib_manifest.py --completed-only > attrib_manifest_all.tsv 2>/dev/null
grep -E "K16_compB|K24_compB" attrib_manifest_all.tsv > attrib_manifest_last4.tsv
wc -l < attrib_manifest_last4.tsv
idx=0
for g in 0 2 5 7; do
  tmux kill-session -t az_shard$g 2>/dev/null
  tmux new-session -d -s az_shard$g \
    "ATTRIB_MANIFEST=/home/nvidia/ncr_scaleaxis/attrib_manifest_last4.tsv bash /home/nvidia/ncr_scaleaxis/run_attrib_score.sh $idx 4 $g > /home/nvidia/ncr_scaleaxis/az_shard$g.log 2>&1"
  idx=$((idx+1))
done
while [ "$(grep -l 'SHARD .* DONE' /home/nvidia/ncr_scaleaxis/az_shard*.log 2>/dev/null | wc -l)" != "4" ]; do sleep 20; done
grep -h "SHARD .* DONE" /home/nvidia/ncr_scaleaxis/az_shard*.log
grep -h "FAILED\|LOUD FAILURE\|SELF-CHECK FAIL\|SKIP+FLAG\|MISMATCH" /home/nvidia/ncr_scaleaxis/az_shard*.log || echo "NO SCORING FAILURES"
echo "ATTRIB_FINISH_DONE"
