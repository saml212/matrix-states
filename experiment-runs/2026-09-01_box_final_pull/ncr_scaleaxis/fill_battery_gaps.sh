#!/usr/bin/env bash
# Score the sweep1b cells that the original score_sweep1b.sh pass missed.
# Waits for the depth-ext wave so only ONE ~30 GB checkpoint is resident on the
# GPU at a time. Written as a file (not an inline nested-quoted command) because
# the first attempt mangled its own variable expansion.
set -u
cd /home/nvidia/ncr_scaleaxis
export CUDA_VISIBLE_DEVICES=0
export NCR_SCALE=1310m
PY=/home/nvidia/tdenv/bin/python3

while ! grep -q "DEPTHEXT1B DONE" /home/nvidia/ncr_scaleaxis/h1b.log 2>/dev/null; do sleep 60; done
echo "=== depth-ext wave finished; filling battery gaps ==="

fail=0
for cell in scaleaxis1310m_K16_primary_s0 scaleaxis1310m_K16_primary_s1; do
  K=16
  OUT="/home/nvidia/ncr_scaleaxis/results/sweep1b_${cell}_kscaling.json"
  if [ -f "$OUT" ]; then echo "already scored: $cell"; continue; fi
  CK="/ephemeral/scaleaxis1b/ckpts/${cell}/${cell}.ckpt.pt"
  RES="/ephemeral/scaleaxis1b/results/${cell}.json"
  if [ ! -f "$CK" ]; then echo "!!! MISSING-CKPT $cell: $CK"; fail=1; continue; fi
  echo "=== scoring $cell ==="
  NCR_K=$K "$PY" kscaling_battery.py --k $K --ckpt "$CK" --cellcfg "$RES" \
      --tag "sweep1b_${cell}" || { echo "!!! FAILED $cell"; fail=1; }
done

# fail loudly if the sweep is still not 22/22
"$PY" - <<'PY'
import json, glob, sys
have=set()
for p in glob.glob("/home/nvidia/ncr_scaleaxis/results/sweep1b_*.json"):
    d=json.load(open(p))
    have.add((d.get("K") or d["kscaling"]["K"], bool(d["freeze_entity_adapter"]), int(d["ckpt_seed"])))
miss=[f"K{k}_{'primary' if f else 'compB'}_s{s}"
      for k in (16,24,32,40) for f in (True,False) for s in (0,1,2)
      if not (k==24 and s==0) and (k,f,s) not in have]
print(f"SWEEP1B COVERAGE: {len(have)}/22")
if miss:
    print("!!! SELF-CHECK FAIL: sweep cells still unscored:", miss); sys.exit(1)
print("SELF-CHECK PASS: all 22 sweep cells scored")
PY
echo "GAPFILL_EXIT=$?"
echo GAPFILL_DONE
