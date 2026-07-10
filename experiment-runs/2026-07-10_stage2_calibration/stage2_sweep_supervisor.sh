#!/bin/bash
# stage2_sweep_supervisor.sh -- S2.30: the 51-distinct-cell (registry "57")
# Stage-2 remainder sweep. ONE tmux session (stage2_sweep) runs this parent;
# it spawns 7 self-healing worker loops, one per GPU 0-6 (GPU 7 left free for
# the task2-diagnosis agent per the standing allocation; max 7 concurrent
# cells, well under the 25-cell concurrency cap). CLAUDE.md pattern:
# while [ ! -f STOP ] supervisor loops, resume-safe workers
# (is_valid_output(strict_real=True)), per-worker 20-fail bail.
# PI signoff: CAPABILITY_SEP_STAGE2_PI_SIGNOFF=1 citing S2.24 (build chain
# CLEARED) + S2.26 (fla cross-check 3/3) + S2.28/S2.29 (in-flight fixes,
# independently audited) + S2.30 (gate verdict SWEEP-READY: 2(e) 11/11 pass,
# rate 0.0433 in-band, projection 2.47 GPU-h vs cap 25).
cd /home/nvidia/chapter2/capability_separation || exit 1
export CAPABILITY_SEP_STAGE2_PI_SIGNOFF=1
PY=/home/nvidia/tdenv/bin/python
RD=stage2_results
NSHARDS=7
rm -f STOP_stage2_sweep
rm -f $RD/SWEEP_DONE $RD/SWEEP_FAILED

worker_loop() {
  local k=$1
  local fails=0
  while [ ! -f STOP_stage2_sweep ]; do
    if CUDA_VISIBLE_DEVICES=$k $PY stage2_sweep_worker.py --shard $k --nshards $NSHARDS \
        --results-dir $RD >> stage2_sweep_shard${k}.log 2>&1; then
      echo "[sweep-supervisor $(date -u +%FT%TZ)] shard $k COMPLETE" >> stage2_sweep_supervisor.log
      touch $RD/SWEEP_SHARD_${k}_DONE
      return 0
    fi
    fails=$((fails + 1))
    echo "[sweep-supervisor $(date -u +%FT%TZ)] shard $k exited nonzero (fail $fails); retry in 15s" >> stage2_sweep_supervisor.log
    if [ "$fails" -ge 20 ]; then
      echo "[sweep-supervisor $(date -u +%FT%TZ)] shard $k: 20 fails -- BAILING for diagnosis" >> stage2_sweep_supervisor.log
      touch $RD/SWEEP_SHARD_${k}_FAILED
      return 1
    fi
    sleep 15
  done
  echo "[sweep-supervisor $(date -u +%FT%TZ)] shard $k stopped via STOP sentinel" >> stage2_sweep_supervisor.log
  return 1
}

echo "[sweep-supervisor $(date -u +%FT%TZ)] launching $NSHARDS shard loops (GPUs 0-6)" >> stage2_sweep_supervisor.log
pids=()
for k in $(seq 0 $((NSHARDS - 1))); do
  worker_loop $k &
  pids+=($!)
done

status=0
for p in "${pids[@]}"; do
  wait "$p" || status=1
done

if [ "$status" -eq 0 ]; then
  echo "[sweep-supervisor $(date -u +%FT%TZ)] ALL SHARDS COMPLETE" >> stage2_sweep_supervisor.log
  touch $RD/SWEEP_DONE
else
  echo "[sweep-supervisor $(date -u +%FT%TZ)] one or more shards failed/stopped" >> stage2_sweep_supervisor.log
  touch $RD/SWEEP_FAILED
fi
