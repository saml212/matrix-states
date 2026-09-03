#!/usr/bin/env bash
# §1.36c S5 razor re-test at 20k steps: seeds 0-3, one tmux session per seed, GPUs 0-3.
set -u
CAP_DIR=/home/nvidia/chapter2/capability_separation
PY=/home/nvidia/tdenv/bin/python
cd "$CAP_DIR" && mkdir -p results_m3fix_s5_20k logs_m3fix_s5_20k
for SEED in 0 1 2 3; do
  NAME="m3s5_20k_s${SEED}"; GPU=$SEED
  tmux new-session -d -s "$NAME" "cd $CAP_DIR && export CUDA_VISIBLE_DEVICES=$GPU CAPABILITY_SEP_PI_SIGNOFF=1; echo \"[$NAME] start \$(date -u) gpu=$GPU\" >> logs_m3fix_s5_20k/$NAME.log; until $PY run_capability_sep.py --m3fix --m3fix-seed $SEED --m3fix-groups S5 --steps 20000 --device cuda --results-dir results_m3fix_s5_20k/ >> logs_m3fix_s5_20k/$NAME.log 2>&1; do echo \"[$NAME] nonzero exit, retry in 15s \$(date -u)\" >> logs_m3fix_s5_20k/$NAME.log; [ -f STOP_s5_20k ] && break; sleep 15; done; touch results_m3fix_s5_20k/DONE_s${SEED}; echo \"[$NAME] done \$(date -u)\" >> logs_m3fix_s5_20k/$NAME.log"
done
sleep 25; tmux ls | grep -c m3s5_20k; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -4; tail -2 logs_m3fix_s5_20k/m3s5_20k_s0.log
