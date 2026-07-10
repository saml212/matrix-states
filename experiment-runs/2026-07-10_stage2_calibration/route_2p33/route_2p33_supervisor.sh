#!/bin/bash
# S2.33 route item (2) supervisor -- S5 decisive-config seed extension (2 cells).
# tmux session: stage2_route_2p33 (exact-name kill only, never pkill).
# GPU 0 ONLY via CUDA_VISIBLE_DEVICES=0 -- GPUs 6/7 (NCR Phase-0,
# task2-diagnosis) are never visible to this process tree.
# Self-healing loop (house convention): crash auto-restarts; the runner is
# resume-safe (run_cell_resume_safe skips valid real outputs), so a restart
# loses nothing. STOP sentinel: STOP_route_2p33.
cd /home/nvidia/chapter2/capability_separation || exit 1
export CUDA_VISIBLE_DEVICES=0
export CAPABILITY_SEP_STAGE2_PI_SIGNOFF=1  # citing S2.30's authorization + S1.4.2's escalation-to-5 trigger (S2.33)
N=0
while [ ! -f STOP_route_2p33 ]; do
  /home/nvidia/tdenv/bin/python route_2p33_s5ext_box.py >> route_2p33_s5ext.log 2>&1
  if grep -q ROUTE_2P33_S5EXT_DONE route_2p33_s5ext.log; then
    touch ROUTE_2P33_DONE
    exit 0
  fi
  N=$((N+1))
  if [ "$N" -ge 10 ]; then
    touch ROUTE_2P33_FAILED
    exit 1
  fi
  sleep 15
done
