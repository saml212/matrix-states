#!/bin/bash
# GPU-6 sequence (launch-audited 2026-07-04): rung-3 two-point calibration with
# cold-Triton-cache retry, then the mixcontrol wave. Stop: touch STOP_trackc_g6.
cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
for attempt in 1 2 3; do
  $PY run_lm_rd_trackc_sweep.py --wave calibration --calib-rungs 3 \
    --out-dir results/lm_rd_trackc --gpus 1 --gpu-offset 6 \
    >> results/lm_rd_trackc/rung3_calib.log 2>&1
  ok=$($PY -c "import json; d=json.load(open(\"results/lm_rd_trackc/calibration.json\")); tc=d.get(\"timing_constants\",{}).get(\"rung3\",{}); print(1 if tc.get(\"per_step_s\",0)>0 else 0)" 2>/dev/null)
  if [ "$ok" = "1" ]; then echo "rung3 calibration VALID on attempt $attempt" >> results/lm_rd_trackc/rung3_calib.log; break; fi
  echo "rung3 calibration attempt $attempt invalid (cold-cache) -- discard + retry" >> results/lm_rd_trackc/rung3_calib.log
  for pt in A B; do
    f=$(ls results/lm_rd_trackc/calibration/calib_rung3_pt${pt}_*.json 2>/dev/null | head -1)
    [ -n "$f" ] && mv "$f" "$f.coldcache-invalid"
  done
done
while [ ! -f STOP_trackc_g6 ] && [ ! -f results/lm_rd_trackc/mixcontrol/ALL_DONE ]; do
  $PY run_lm_rd_trackc_sweep.py --wave mixcontrol --out-dir results/lm_rd_trackc \
    --gpus 1 --gpu-offset 6 >> results/lm_rd_trackc/mixcontrol_launch.log 2>&1
  sleep 15
done
