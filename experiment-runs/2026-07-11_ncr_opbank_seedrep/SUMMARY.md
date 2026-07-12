# NCR operator-bank earlyln SEED REPLICATION (§8.10) — 2026-07-12 UTC (2026-07-11 PDT)

Pre-registered §8.9 follow-on: seed replication of the earlyln RECOVERED
cell. Script `ncr_opbank_recover.py` byte-identical to the §8.9 audited
version (md5 `6007b092fb7e860757b45a20f233b6d5`, verified against box +
`experiment-runs/2026-07-11_ncr_opbank_recover/md5_manifest.txt`). No
logic changes. Seeds 1..8 (seed 0 = §8.9's original), one seed per GPU
0-7, identical config: `--cell earlyln --steps 80000 --train-batch 256
--ceiling-gpuh 2.0`, per-seed `--outdir results_opbank_seedrep/seed<N>`.

Launch (per seed N, GPU N-1), as preemptible filler under the
`opbank_earlyln_` tmux-prefix preemption contract, no supervisor loop:

    tmux new-session -d -s opbank_earlyln_s<N> \
      "CUDA_VISIBLE_DEVICES=<N-1> ~/tdenv/bin/python ncr_opbank_recover.py \
       --cell earlyln --seed <N> --steps 80000 --train-batch 256 \
       --ceiling-gpuh 2.0 --outdir ~/ncr/results_opbank_seedrep/seed<N> \
       --stop-file ~/ncr/results_opbank_seedrep/seed<N>/STOP --device cuda \
       2>&1 | tee <outdir>/opbank_earlyln_s<N>.log"

Launched 04:46:59Z, all terminal 05:13-05:18Z (26-31 min/seed). No seed
was preempted (the §11 K-scaling session appeared 05:20:55Z, after all
seeds finished). A first launch at 04:44Z shared one outdir — the
script's output filename is seed-agnostic, so 8 parallel seeds would
have overwritten each other; killed at ~2 min (~0.06 GPU-h discarded),
relaunched with per-seed outdirs.

## Per-seed results (raws: seed<N>/ncropbank_recover_earlyln.json)

All 8 COMPLETED at 80000 steps, 0 ABORTED-BUDGET, 0 missing. blank-out
P=1 passed 8/8. swap gap > 0.3 in 8/8.

| seed | final loss | in-dist min rec@0.9 (h=1/2/3 × r) | A_eff_rank | phase_resid | swap gap | h*=61 rec@0.9 per r (min/med/max) | h*=61 min mean_cos | GPU-h |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.00033 | 1.000 | 7.999-8.000 | 0.0052-0.0062 | 0.9407 | 0.9180 / 0.9297 / 0.9375 | 0.9431 | 0.444 |
| 2 | 0.01037 | 1.000 | 7.954-7.986 | 0.0220-0.0278 | 0.4470 | 0.0000 / 0.0000 / 0.0039 | 0.4703 | 0.522 |
| 3 | 0.00799 | 1.000 | 7.963-7.996 | 0.0100-0.0311 | 0.6276 | 0.0000 / 0.0117 / 0.2422 | 0.4196 | 0.451 |
| 4 | 0.00491 | 1.000 | 7.982-7.998 | 0.0103-0.0419 | 0.6537 | 0.0000 / 0.0410 / 0.1133 | 0.5789 | 0.523 |
| 5 | 0.00049 | 1.000 | 7.999-8.000 | 0.0042-0.0073 | 0.8979 | 0.8105 / 0.8633 / 0.9844 | 0.9207 | 0.467 |
| 6 | 0.00034 | 1.000 | 7.999-8.000 | 0.0069-0.0086 | 0.9001 | 0.6152 / 0.8535 / 0.8672 | 0.9093 | 0.497 |
| 7 | 0.00531 | 1.000 | 7.985-7.997 | 0.0142-0.0188 | 0.6164 | 0.0117 / 0.0508 / 0.0586 | 0.6714 | 0.481 |
| 8 | 0.00021 | 1.000 | 8.000-8.000 | 0.0032-0.0046 | 0.9585 | 0.9844 / 1.0000 / 1.0000 | 0.9682 | 0.494 |

Reference (§8.9 seed 0): in-dist 1.000, far-61 0.004-0.049, phase_resid
0.016-0.020, swap gap 0.5526.

## Counts (n=9 incl. seed 0)

- Convergence (in-dist min-over-9-cells rec@0.9 ≥ 0.9): **9/9**. Dead
  seeds: **0/9** (K=12 single-relation precedent: 2/10).
- Far-depth h*=61, min-over-r rec@0.9, all 9 seeds sorted:
  0.000, 0.000, 0.000, 0.004(s0), 0.012, 0.615, 0.811, 0.918, 0.984.
  ≥0.9: 2/9 (s1, s8). (0.5, 0.9): 2/9 (s5 0.811, s6 0.615). <0.5: 5/9
  (s0, s2, s3, s4, s7).
- Per-seed phase_resid vs far-61 (observed pairing, numbers only): the
  four seeds with max phase_resid ≤ 0.0086 (s1/s5/s6/s8) are the four
  with far-61 min-over-r ≥ 0.615; the five with max phase_resid ≥ 0.019
  (s0/s2/s3/s4/s7) all have far-61 min-over-r ≤ 0.049.

Ledger: 3.880 GPU-h (8 cells) + ~0.06 discarded first launch ≈ 3.94.
Box raws: `youthful-indigo-turkey:/home/nvidia/ncr/results_opbank_seedrep/seed{1..8}/`
(JSONs + axis_c_locks + logs mirrored fully in this directory; total <1MB,
all within the ≤25MB repo tier — no SSD-only payloads).
