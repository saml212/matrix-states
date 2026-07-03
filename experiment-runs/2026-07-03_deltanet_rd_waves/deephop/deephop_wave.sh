#!/bin/bash
# Deeper-hop training probe (2026-07-03): does training at h in {1..5} extend
# the real-data exact-composition regime vs the {1,2,3} baseline (RD Wave A)?
# Attacks the K-exactness frontier with audited code. Bounded: 6 runs.
# Residue validity: K=8 -> test {6,7} (6,7 mod 8 not in {0,1..5});
# K=16 -> test {6,7,9} (all valid mod 16). h-extra dropped (21 mod 8 = 5 is
# now a TRAINED residue at K=8 -- invalid; use h-extra 15 for K=16 only:
# 15 mod 16 = 15, valid).
cd /home/nvidia/chapter2/deltanet_rd
PY=/home/nvidia/tdenv/bin/python
OUT=results/deltanet_rd_w0b/deephop
mkdir -p $OUT/logs
i=0
launch () {  # K seed h_test... (h-extra per K)
  local K=$1 s=$2; shift 2
  local gpu=$((i % 6))
  local name="deephop_rd_K${K}_htrain12345_s${s}"
  CUDA_VISIBLE_DEVICES=$gpu $PY run_deltanet_rd.py --K $K --steps 25000 --seed $s \
    --h-train 1 2 3 4 5 --trunc-impl svd_lowrank --internal-timeout 4000 \
    $@ --out $OUT/${name}.json > $OUT/logs/${name}.log 2>&1 &
  i=$((i+1))
}
launch 8 0 --h-test 6 7 --h-extra 7
launch 8 1 --h-test 6 7 --h-extra 7
launch 8 2 --h-test 6 7 --h-extra 7
launch 16 0 --h-test 6 7 9 --h-extra 15
launch 16 1 --h-test 6 7 9 --h-extra 15
launch 16 2 --h-test 6 7 9 --h-extra 15
wait
echo done > $OUT/ALL_DONE
