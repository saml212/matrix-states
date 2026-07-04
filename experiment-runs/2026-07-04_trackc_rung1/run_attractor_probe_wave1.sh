#!/usr/bin/env bash
set -e
cd /home/nvidia/chapter2/deltanet_rd
export CUDA_VISIBLE_DEVICES=0
CKPT_DIR=results/lm_rd_trackc/wave1/checkpoints
OUT_DIR=results/lm_rd_trackc/wave1/attractor_probe
mkdir -p $OUT_DIR

echo '=== RUNG1 (98M, dm768/L12) pooled, 6 checkpoints ==='
/home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \
  --checkpoints $CKPT_DIR/lmC_openr1-mix_dm768_ds64_L12_s0_step67547.pt \
                $CKPT_DIR/lmC_openr1-mix_dm768_ds64_L12_s1_step67547.pt \
                $CKPT_DIR/lmC_openr1-mix_dm768_ds64_L12_s2_step67547.pt \
                $CKPT_DIR/lmC_wikitext-mix_dm768_ds64_L12_s0_step67547.pt \
                $CKPT_DIR/lmC_wikitext-mix_dm768_ds64_L12_s1_step67547.pt \
                $CKPT_DIR/lmC_wikitext-mix_dm768_ds64_L12_s2_step67547.pt \
  --out $OUT_DIR/rung1_pooled.json

echo '=== CONTROL (14M, dm256/L2) pooled, 6 checkpoints ==='
/home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \
  --checkpoints $CKPT_DIR/lmC_openr1-mix_dm256_ds64_L2_s0_step6103.pt \
                $CKPT_DIR/lmC_openr1-mix_dm256_ds64_L2_s1_step6103.pt \
                $CKPT_DIR/lmC_openr1-mix_dm256_ds64_L2_s2_step6103.pt \
                $CKPT_DIR/lmC_wikitext-mix_dm256_ds64_L2_s0_step6103.pt \
                $CKPT_DIR/lmC_wikitext-mix_dm256_ds64_L2_s1_step6103.pt \
                $CKPT_DIR/lmC_wikitext-mix_dm256_ds64_L2_s2_step6103.pt \
  --out $OUT_DIR/control_pooled.json

echo ALL_DONE
