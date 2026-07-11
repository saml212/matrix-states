#!/bin/bash
# REASONING-LINK INSTRUMENT VERIFIER, chain step 4 -- THE BIG RE-METRIC
# (REASONING_LINK_DESIGN.md sec 17.4, 2026-07-11).
#
# Re-runs the FULL committed Phase-1 grid (reasoning_link_chain.sh's own
# steps 4-9, 80 cells: 60 Leg A + 20 Leg B across all 4 rungs) under the
# FIXED squeeze_state_head (sec 17.1/17.2 -- the [K,V]-vs-[V,K] state-layout
# transpose fix). No raw states/Z-dumps were ever archived by the original
# harvest (only derived scalar JSONs, all computed via the pre-fix, buggy
# instrument) -- so nothing is "free" to re-score; this reruns fresh forward
# passes over the SAME already-archived checkpoints, using the SAME
# deterministic episode_seed() formula (mixed-radix, collision-free), which
# reproduces BIT-IDENTICAL synthetic episodes to the original harvest. Cost
# is essentially unchanged from the original run (the fix only changes
# postprocessing of an already-computed final_state, not any forward-pass
# compute) -- the original 80-cell grid realized ~=0.307 GPU-h total
# (REASONING_LINK_DESIGN.md sec 15.7: 0.29 GPU-h for Stage-1's 78 cells +
# sec 16.11: 0.017 GPU-h for rung-3's 2 cells), well under this dispatch's
# own 2 GPU-h authorization ceiling -- priced BEFORE launch, not after.
#
# Skips Stage -1/0/0.5 (already independently validated this session via
# Stage -1 item 20's closed-form check + the standalone real-kernel
# adjudication script + the poscontrol re-run, all recorded at sec
# 17.1-17.3) -- this script reruns ONLY the committed grid cells
# (--mode cell), which never pass through the CPU stub at all (no
# REASONING_LINK_FORCE_CPU_STUB anywhere in this script, matching
# reasoning_link_chain.sh's own steps 4-9), so the unrelated, pre-existing
# fla.ops.gated_delta_rule stub gap (discovered this session, reported
# separately, NOT fixed here -- out of scope) does not block this rerun.
set -euo pipefail

cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
OUTDIR=results/reasoning_link_remetric
mkdir -p logs "$OUTDIR"

REASONING_LINK_GPU=${REASONING_LINK_GPU:-1}

FROZEN_BIAS_CKPT_ROOT=${FROZEN_BIAS_CKPT_ROOT:-/data/deltanet_rd_frozenbias_ckpts}
TRACKC_CKPT_ROOT=${TRACKC_CKPT_ROOT:-/data/lm_rd_trackc_ckpts}

# ---------------------------------------------------------------------------
# Leg A committed grid (reasoning_link_chain.sh steps 4-7, identical loop
# structure/params -- 60 cells: 3 arms x 2 corpora x 3 seeds x 2 K x
# applicable surgery).
# ---------------------------------------------------------------------------
LEG_A_ARMS=(off per_token global)
LEG_A_CORPORA=(openr1-mix-ext wikitext-mix-ext)
LEG_A_SEEDS=(0 1 2)
LEG_A_KS=(20 32)

t_start=$(date +%s)
n_leg_a=0
for arm in "${LEG_A_ARMS[@]}"; do
  for corpus in "${LEG_A_CORPORA[@]}"; do
    for seed in "${LEG_A_SEEDS[@]}"; do
      for k in "${LEG_A_KS[@]}"; do
        for surgery in native off; do
          if [ "$arm" = "off" ] && [ "$surgery" = "off" ]; then
            continue
          fi
          out="$OUTDIR/leg_a_${arm}_${corpus}_s${seed}_k${k}_${surgery}.json"
          log="logs/95_remetric_leg_a_${arm}_${corpus}_s${seed}_k${k}_${surgery}.log"
          if [ -s "$out" ] && $PY -c "import json,sys; d=json.load(open('$out')); sys.exit(0 if d.get('forward_counts') else 1)" 2>/dev/null; then
            echo "SKIP (already complete): $out"
            n_leg_a=$((n_leg_a+1))
            continue
          fi
          CUDA_VISIBLE_DEVICES=$REASONING_LINK_GPU $PY reasoning_link_probe.py \
              --mode cell --family leg_a --arm "$arm" --corpus "$corpus" --ckpt-seed "$seed" \
              --ckpt-base-dir "$FROZEN_BIAS_CKPT_ROOT" --k "$k" --hops 1,2,3,4 \
              --surgery "$surgery" --batch-size 16 --device cuda --out "$out" \
              2>&1 | tee "$log"
          n_leg_a=$((n_leg_a+1))
        done
      done
    done
  done
done
echo "Leg A re-metric: $n_leg_a/60 cells done."

# ---------------------------------------------------------------------------
# Leg B committed grid (reasoning_link_chain.sh steps 8-9, identical loop
# structure/params -- 20 cells: rungs 0-2 = 2 corpora x 3 seeds each (18),
# rung 3 = 2 corpora x 1 seed (2, sec 6.1's PINNED rung-3 configuration).
# ---------------------------------------------------------------------------
LEG_B_CORPORA=(openr1-mix-ext wikitext-mix-ext)

n_leg_b=0
for rung in 0 1 2 3; do
  if [ "$rung" -lt 2 ]; then k=32; else k=64; fi
  if [ "$rung" = "3" ]; then seeds=(0); else seeds=(0 1 2); fi
  if [ "$rung" = "3" ]; then cell_batch=4; elif [ "$rung" = "2" ]; then cell_batch=8; else cell_batch=16; fi
  for corpus in "${LEG_B_CORPORA[@]}"; do
    for seed in "${seeds[@]}"; do
      out="$OUTDIR/leg_b_rung${rung}_${corpus}_s${seed}_k${k}.json"
      log="logs/95_remetric_leg_b_rung${rung}_${corpus}_s${seed}_k${k}.log"
      if [ -s "$out" ] && $PY -c "import json,sys; d=json.load(open('$out')); sys.exit(0 if d.get('forward_counts') else 1)" 2>/dev/null; then
        echo "SKIP (already complete): $out"
        n_leg_b=$((n_leg_b+1))
        continue
      fi
      CUDA_VISIBLE_DEVICES=$REASONING_LINK_GPU $PY reasoning_link_probe.py \
          --mode cell --family leg_b --rung "$rung" --corpus "$corpus" --ckpt-seed "$seed" \
          --ckpt-base-dir "$TRACKC_CKPT_ROOT" --k "$k" --hops 1,2,3,4 \
          --surgery native --batch-size "$cell_batch" --device cuda --out "$out" \
          2>&1 | tee "$log"
      n_leg_b=$((n_leg_b+1))
    done
  done
done
echo "Leg B re-metric: $n_leg_b/20 cells done."

t_end=$(date +%s)
echo "TOTAL WALL: $((t_end - t_start))s"
touch "$OUTDIR/REASONING_LINK_REMETRIC_DONE"
echo "REASONING-LINK RE-METRIC (sec 17.4): all $((n_leg_a + n_leg_b))/80 cells complete."
