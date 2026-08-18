#!/usr/bin/env bash
# Multi-seed premise-battery evaluation: run the AUDITED pbe_repl instrument
# (identical to the 2026-08-13 replication: eval_arm_at_hops, seed 90210, n=256,
# both write modes) over every freshly-trained reseed checkpoint.
# Eval-only, no training. Runs on ONE gpu, sequentially, ~0.02 GPU-h per ckpt.
set -u
cd /home/nvidia/ncr_writecond
export CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-0}
R=/home/nvidia/ncr_g3b31_contrastive/results
for tag in compA compB primary; do
  # compB trained with a TRAINABLE entity adapter; compA/primary frozen.
  if [ "$tag" = "compB" ]; then FZ=""; else FZ="freeze"; fi
  for s in 1 2 3 4 5 6; do
    CK="$R/mob_g3b31_${tag}_s${s}_ckpts/mob_g3b31_${tag}_s${s}.ckpt.pt"
    [ -f "$CK" ] || { echo "SKIP ${tag}_s${s} (no checkpoint yet)"; continue; }
    OUT="/home/nvidia/ncr_writecond/results/writecond_premise_REPL_${tag}_s${s}.json"
    [ -f "$OUT" ] && { echo "SKIP ${tag}_s${s} (already scored)"; continue; }
    echo "=== scoring ${tag}_s${s} ==="
    /home/nvidia/tdenv/bin/python3 pbe_repl "$CK" "${tag}_s${s}" $FZ || echo "FAILED ${tag}_s${s}"
  done
done
echo REPL_WAVE_DONE
