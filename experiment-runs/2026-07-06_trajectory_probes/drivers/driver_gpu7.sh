#!/usr/bin/env bash
set -e
cd /home/nvidia/chapter2/deltanet_rd
export CUDA_VISIBLE_DEVICES=7
mkdir -p results/lm_rd_trackc/trajectory_probes/wave1 results/lm_rd_trackc/trajectory_probes/wave1ext results/lm_rd_trackc/trajectory_probes/wave2 results/lm_rd_trackc/trajectory_probes/mixcontrol
echo "=== GPU7: wave2 wikitext-mix-ext_s0 (13 ckpts) ==="
/home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \
  --checkpoints \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step1000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step9000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step17000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step25000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step33000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step41000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step49000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step57000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step65000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step73000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step81000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step89000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s0_step91552.pt \
  --out results/lm_rd_trackc/trajectory_probes/wave2/wikitext-mix-ext_s0.json
echo "=== GPU7: wave2 wikitext-mix-ext_s0 DONE ==="
echo "=== GPU7: wave2 wikitext-mix-ext_s1 (13 ckpts) ==="
/home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \
  --checkpoints \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step1000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step9000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step17000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step25000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step33000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step41000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step49000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step57000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step65000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step73000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step81000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step89000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s1_step91552.pt \
  --out results/lm_rd_trackc/trajectory_probes/wave2/wikitext-mix-ext_s1.json
echo "=== GPU7: wave2 wikitext-mix-ext_s1 DONE ==="
echo "=== GPU7: wave2 wikitext-mix-ext_s2 (13 ckpts) ==="
/home/nvidia/tdenv/bin/python3 lm_attractor_probe_rd.py \
  --checkpoints \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step1000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step9000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step17000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step25000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step33000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step41000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step49000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step57000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step65000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step73000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step81000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step89000.pt \
    /data/lm_rd_trackc_ckpts/wave2/lmC_wikitext-mix-ext_dm1536_ds128_L16_s2_step91552.pt \
  --out results/lm_rd_trackc/trajectory_probes/wave2/wikitext-mix-ext_s2.json
echo "=== GPU7: wave2 wikitext-mix-ext_s2 DONE ==="
echo ALL_DONE_GPU7
