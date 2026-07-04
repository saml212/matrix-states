# H_e full-manifest completion at the 40K working budget (pre-registered rule
# fired: vector composed at 40K). 4 remaining cells, sequential on GPU 7.
cd /home/nvidia/stageg
PY=/home/nvidia/tdenv/bin/python
run () { # family variant seed
  CUDA_VISIBLE_DEVICES=7 $PY run_stageg_he.py --family $1 --variant $2 --steps 40000 --seed $3 \
    --answer-loss-weight 5.0 --ckpt-every 2000 --internal-timeout 42000 \
    --out results/he_calib/he40k_$1_$2_alw5_s$3.json > logs/he40k_$1_$2_s$3.log 2>&1
}
run matrix baseline 1
run matrix h_b_factored_r4 0
run matrix h_b_factored_r4 1
run vector baseline 1
echo done > results/he_calib/MANIFEST40K_DONE
