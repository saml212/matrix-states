#!/usr/bin/env bash
set -e
cd /home/nvidia/chapter2/deltanet_rd
export CUDA_VISIBLE_DEVICES=7
CKPT_DIR_W2=/data/lm_rd_trackc_ckpts/wave2
CKPT_DIR_MC=/data/lm_rd_trackc_ckpts/mixcontrol
OUT_DIR_W2=results/lm_rd_trackc/wave2/attractor_probe
OUT_DIR_MC=results/lm_rd_trackc/mixcontrol/attractor_probe
mkdir -p $OUT_DIR_W2 $OUT_DIR_MC

echo "=== RUNG2 (392M, dm1536/L16/ds128) FINAL pooled, 6 checkpoints @ step91552 ==="
/home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \
  --checkpoints $CKPT_DIR_W2/lmC_openr1-mix-ext_dm1536_ds128_L16_s0_step91552.pt \
                $CKPT_DIR_W2/lmC_openr1-mix-ext_dm1536_ds128_L16_s1_step91552.pt \
                $CKPT_DIR_W2/lmC_openr1-mix-ext_dm1536_ds128_L16_s2_step91552.pt \
                $CKPT_DIR_W2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step91552.pt \
                $CKPT_DIR_W2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step91552.pt \
                $CKPT_DIR_W2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step91552.pt \
  --out $OUT_DIR_W2/rung2_final_pooled.json
echo "=== RUNG2 FINAL pooled DONE ==="

echo "=== MIXCONTROL (14M, dm256/L2/ds64, ext mixes) FINAL pooled, 6 checkpoints @ step6103 ==="
/home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \
  --checkpoints $CKPT_DIR_MC/lmC_openr1-mix-ext_dm256_ds64_L2_s0_step6103.pt \
                $CKPT_DIR_MC/lmC_openr1-mix-ext_dm256_ds64_L2_s1_step6103.pt \
                $CKPT_DIR_MC/lmC_openr1-mix-ext_dm256_ds64_L2_s2_step6103.pt \
                $CKPT_DIR_MC/lmC_wikitext-mix-ext_dm256_ds64_L2_s0_step6103.pt \
                $CKPT_DIR_MC/lmC_wikitext-mix-ext_dm256_ds64_L2_s1_step6103.pt \
                $CKPT_DIR_MC/lmC_wikitext-mix-ext_dm256_ds64_L2_s2_step6103.pt \
  --out $OUT_DIR_MC/mixcontrol_final_pooled.json
echo "=== MIXCONTROL FINAL pooled DONE ==="

echo "=== RUNG2 TRAJECTORY (seed0 only, both corpora, 11 steps ~every 10th of 92 ckpts) ==="
for STEP in 1000 11000 21000 31000 41000 51000 61000 71000 81000 91000 91552; do
  echo "--- trajectory step $STEP ---"
  /home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \
    --checkpoints $CKPT_DIR_W2/lmC_openr1-mix-ext_dm1536_ds128_L16_s0_step${STEP}.pt \
                  $CKPT_DIR_W2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step${STEP}.pt \
    --out $OUT_DIR_W2/rung2_traj_step${STEP}.json
done
echo "=== RUNG2 TRAJECTORY DONE ==="

echo ALL_DONE
