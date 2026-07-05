#!/usr/bin/env bash
set -e
cd /home/nvidia/chapter2/deltanet_rd
export CUDA_VISIBLE_DEVICES=7
CKPT_DIR=/data/lm_rd_trackc_ckpts/wave1ext
OUT_DIR=results/lm_rd_trackc/wave1ext/attractor_probe
mkdir -p $OUT_DIR

echo "=== RUNG1EXT (98M, dm768/L12, ext mixes) FINAL pooled, 6 checkpoints @ step67547 ==="
/home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \
  --checkpoints $CKPT_DIR/lmC_openr1-mix-ext_dm768_ds64_L12_s0_step67547.pt \
                $CKPT_DIR/lmC_openr1-mix-ext_dm768_ds64_L12_s1_step67547.pt \
                $CKPT_DIR/lmC_openr1-mix-ext_dm768_ds64_L12_s2_step67547.pt \
                $CKPT_DIR/lmC_wikitext-mix-ext_dm768_ds64_L12_s0_step67547.pt \
                $CKPT_DIR/lmC_wikitext-mix-ext_dm768_ds64_L12_s1_step67547.pt \
                $CKPT_DIR/lmC_wikitext-mix-ext_dm768_ds64_L12_s2_step67547.pt \
  --out $OUT_DIR/rung1ext_final_pooled.json
echo "=== RUNG1EXT FINAL pooled DONE ==="

echo "=== RUNG1EXT TRAJECTORY (seed0 only, both corpora, 8 steps ~every 10th of 68) ==="
for STEP in 1000 11000 21000 31000 41000 51000 61000 67547; do
  echo "--- trajectory step $STEP ---"
  /home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \
    --checkpoints $CKPT_DIR/lmC_openr1-mix-ext_dm768_ds64_L12_s0_step${STEP}.pt \
                  $CKPT_DIR/lmC_wikitext-mix-ext_dm768_ds64_L12_s0_step${STEP}.pt \
    --out $OUT_DIR/rung1ext_traj_step${STEP}.json
done
echo "=== RUNG1EXT TRAJECTORY DONE ==="

echo ALL_DONE
