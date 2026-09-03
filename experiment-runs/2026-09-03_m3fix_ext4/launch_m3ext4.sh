#!/usr/bin/env bash
# §1.36b razor seed extension: S4/A5/S5/A6 × seeds 1,2,3; 12 tmux sessions over 8 GPUs.
set -u
CAP_DIR=/home/nvidia/chapter2/capability_separation
PY=/home/nvidia/tdenv/bin/python
cd "$CAP_DIR" && mkdir -p results_m3fix_ext4 logs_m3fix_ext4
i=0
for SEED in 1 2 3; do
  for G in A6 S4 A5 S5; do
    case "$G$SEED" in
      A61) GPU=0;; A62) GPU=1;; A63) GPU=2;;
      S41) GPU=3;; S42) GPU=4;; S43) GPU=5;;
      A51) GPU=6;; A52) GPU=7;; A53) GPU=3;;
      S51) GPU=4;; S52) GPU=5;; S53) GPU=6;;
    esac
    NAME="m3ext4_${G}_s${SEED}"
    tmux new-session -d -s "$NAME" "cd $CAP_DIR && export CUDA_VISIBLE_DEVICES=$GPU CAPABILITY_SEP_PI_SIGNOFF=1; echo \"[$NAME] start \$(date -u) gpu=$GPU\" >> logs_m3fix_ext4/$NAME.log; until $PY run_capability_sep.py --m3fix --m3fix-seed $SEED --m3fix-groups $G --device cuda --results-dir results_m3fix_ext4/ >> logs_m3fix_ext4/$NAME.log 2>&1; do echo \"[$NAME] nonzero exit, retry in 15s \$(date -u)\" >> logs_m3fix_ext4/$NAME.log; [ -f STOP_m3ext4 ] && break; sleep 15; done; touch results_m3fix_ext4/DONE_${G}_s${SEED}; echo \"[$NAME] done \$(date -u)\" >> logs_m3fix_ext4/$NAME.log"
    i=$((i+1))
  done
done
sleep 20; tmux ls | grep -c m3ext4; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; tail -2 logs_m3fix_ext4/m3ext4_A6_s1.log
