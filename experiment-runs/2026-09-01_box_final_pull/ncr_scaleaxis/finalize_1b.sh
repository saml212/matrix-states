#!/usr/bin/env bash
# Autonomous finaliser: waits for the reserved harvest to drain the manifest,
# assembles the four datasets into one staging dir, runs the pinned-test
# aggregator, and writes FINAL_TABLES.txt. Runs on the box so the chapter's
# verdict-of-record tables exist whether or not any interactive session does.
set -u
cd /home/nvidia/ncr_scaleaxis
PY=/home/nvidia/tdenv/bin/python3
STAGE=/home/nvidia/ncr_scaleaxis/harvest1b
OUT=/home/nvidia/ncr_scaleaxis/results_1b_depthext
BATT=/home/nvidia/ncr_scaleaxis/results

while ! grep -qE "RESERVED_RUN_DONE|ESCALATE" /home/nvidia/ncr_scaleaxis/h1b3.log 2>/dev/null; do sleep 60; done
if grep -q "ESCALATE" /home/nvidia/ncr_scaleaxis/h1b3.log; then
  echo "!!! harvest escalated (lost the GPU race twice) -- NOT aggregating"; echo FINALIZE_ABORTED; exit 9
fi

echo "=== assembling datasets ==="
cp -f "$OUT"/depthext6_1310m_*_depthext.json "$STAGE"/ 2>/dev/null
cp -f "$BATT"/sweep1b_*_kscaling.json "$BATT"/calib1b_*_kscaling.json "$STAGE"/ 2>/dev/null
echo "  1310M depth  : $(ls $STAGE/depthext6_1310m_*_depthext.json 2>/dev/null | wc -l)/24"
echo "  1310M battery: $(ls $STAGE/sweep1b_*.json $STAGE/calib1b_*.json 2>/dev/null | wc -l)/24"
echo "  392M depth   : $(ls $STAGE/ref392m_depth/*.json | wc -l)/24"
echo "  392M battery : $(ls $STAGE/ref392m_battery/*.json | wc -l)"

# fail loudly rather than aggregate a partial verdict table
n1=$(ls "$STAGE"/depthext6_1310m_*_depthext.json 2>/dev/null | wc -l)
n2=$(ls "$STAGE"/sweep1b_*.json "$STAGE"/calib1b_*.json 2>/dev/null | wc -l)
if [ "$n1" -ne 24 ] || [ "$n2" -ne 24 ]; then
  echo "!!! INCOMPLETE: depth $n1/24, battery $n2/24 -- aggregating anyway but the tables are PARTIAL"
  echo "!!! DO NOT ADJUDICATE A PARTIAL TABLE SET"
fi

echo "=== running the pinned tests ==="
"$PY" aggregate_1b.py "$STAGE" > /home/nvidia/ncr_scaleaxis/FINAL_TABLES.txt 2>&1
echo "aggregator exit=$?  lines=$(wc -l < /home/nvidia/ncr_scaleaxis/FINAL_TABLES.txt)"
echo FINALIZE_DONE
