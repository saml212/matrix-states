#!/usr/bin/env bash
# Multi-seed premise-battery evaluation (audited pbe_repl instrument: seed 90210,
# n=256, both write modes). Eval-only.
# 2026-08-18: checkpoints now live on /ephemeral (disk policy after the root-full
# incident); older ones remain under results/. Search BOTH, and FAIL LOUDLY if a
# completed cell's checkpoint is found in neither -- silently scoring 0 cells has
# now cost two ticks.
set -u
cd /home/nvidia/ncr_writecond
export CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-0}
OLD=/home/nvidia/ncr_g3b31_contrastive/results
NEW=/ephemeral/reseed_ckpts
scored=0; missing=0
for tag in compA compB primary; do
  if [ "$tag" = "compB" ]; then FZ=""; else FZ="freeze"; fi
  for s in $(seq 1 24); do
    NAME="mob_g3b31_${tag}_s${s}"
    CK=""
    for cand in "$NEW/${NAME}_ckpts/${NAME}.ckpt.pt" "$OLD/${NAME}_ckpts/${NAME}.ckpt.pt"; do
      [ -f "$cand" ] && { CK="$cand"; break; }
    done
    RES="$OLD/${NAME}.json"
    OUT="/home/nvidia/ncr_writecond/results/writecond_premise_REPL_${tag}_s${s}.json"
    [ -f "$OUT" ] && continue                      # already scored
    if [ -z "$CK" ]; then
      # only complain if the cell actually finished training
      [ -f "$RES" ] && { echo "MISSING-CKPT ${tag}_s${s} (results JSON exists but no checkpoint in either location)"; missing=$((missing+1)); }
      continue
    fi
    echo "=== scoring ${tag}_s${s} ($CK) ==="
    /home/nvidia/tdenv/bin/python3 pbe_repl "$CK" "${tag}_s${s}" $FZ && scored=$((scored+1)) || echo "FAILED ${tag}_s${s}"
  done
done
echo "SCORED=$scored MISSING=$missing"
echo REPL_WAVE_DONE
