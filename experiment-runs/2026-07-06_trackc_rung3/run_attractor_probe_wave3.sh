#!/usr/bin/env bash
set -e
cd /home/nvidia/chapter2/deltanet_rd
export CUDA_VISIBLE_DEVICES=0
CKPT_DIR=/data/lm_rd_trackc_ckpts/wave3
OUT_DIR=results/lm_rd_trackc/wave3/attractor_probe
mkdir -p $OUT_DIR

echo "=== RUNG3 (1.31B, dm2560/L22/ds128, ext mixes) FINAL pooled, 2 checkpoints @ step155000 ==="
/home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \
  --checkpoints $CKPT_DIR/lmC_openr1-mix-ext_dm2560_ds128_L22_s0_step155000.pt \
                $CKPT_DIR/lmC_wikitext-mix-ext_dm2560_ds128_L22_s0_step155000.pt \
  --out $OUT_DIR/rung3_final_pooled.json
echo "=== RUNG3 FINAL pooled DONE ==="

echo "=== RUNG3 PLATEAU CHECK (both corpora, 4 late steps 130k/140k/150k/155k) ==="
for STEP in 130000 140000 150000 155000; do
  echo "--- plateau step $STEP ---"
  /home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \
    --checkpoints $CKPT_DIR/lmC_openr1-mix-ext_dm2560_ds128_L22_s0_step${STEP}.pt \
                  $CKPT_DIR/lmC_wikitext-mix-ext_dm2560_ds128_L22_s0_step${STEP}.pt \
    --out $OUT_DIR/rung3_plateau_step${STEP}.json
done
echo "=== RUNG3 PLATEAU CHECK DONE ==="

echo ALL_DONE
