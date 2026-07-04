# Track C Wave 2 (rung-2, 392M) — launch record, 2026-07-04 06:10 UTC

Spec: `SCALE_TRANSFER_DESIGN.md` §5 incl. the §5.6 Rev 2.1 amendment (registered
before launch, no rung-2 data existed). Code: commits `6c7a119` + `d2ee26b`.
Independent launch audit: fresh agent, verdict initially BLOCKED on one FATAL
(see below), LAUNCH-READY after remediation.

## Wave composition
- 6 runs = 2 extended mixes (`openr1-mix-ext`, `wikitext-mix-ext`) × 3 seeds
  (0,1,2), rung-2 config d_model=1536 / n_layers=16 / d_state=128 =
  391,869,440 params, batch 32, seq 512, 91,552 steps ≈ 1.5B tokens/run.
- Timing (measured two-point, warm pair): per_step_s=0.8358, per_ckpt_s=11.96
  → ≈21.6 h/run, 129.36 GPU-h wave total, cumulative 162.46/300 GPU-h ceiling.
- Gates at launch (from `wave2_launch.log`): memory headroom 43.9GB reserved =
  54.9% of 80GB (≤85% ✓); epoch-cap planned 1.50B vs ceilings 1.72B/2.09B ✓;
  budget guard ✓; smoke gate 4/4 PASS; timeout 1.6× projection.

## FATAL caught by the launch audit (before any GPU spend)
Root disk was 94% full (13.5 GiB free); wave-2 checkpoints = 6 runs × 92 ckpts
× 1.5676 GB ≈ 865 GB would have ENOSPC'd every cell within ~30 min and taken
GPU 7's live H_e experiment down as collateral. No gate in the (a)-(e) chain
checks disk space. Remediation (2026-07-04 06:08 UTC): moved
`results/lm_rd_trackc/{calibration,wave1}/checkpoints` (155 GB) to
`/data/lm_rd_trackc_ckpts/` and symlinked; pre-created wave2/mixcontrol
checkpoint symlinks the same way. Root: 93% → 13% used.

## Box processes (youthful-indigo-turkey)
- tmux `trackc2` — wave-2 supervisor: GPUs 0-5, `--wave 2 --gpus 6
  --gpu-offset 0 --rung2-steps 91552`, self-healing while-loop, stop via
  `touch STOP_trackc2`, log `results/lm_rd_trackc/wave2_launch.log`.
- tmux `trackc_g6` — `g6_sequence.sh` (archived here): rung-3 two-point
  calibration (`--calib-rungs 3`) with up-to-3-attempt cold-Triton-cache
  discard/retry, then `--wave mixcontrol` (6 × 14M control on ext mixes,
  ≈0.46 GPU-h). Stop via `touch STOP_trackc_g6`.
- tmux `he40kman` — GPU 7, unrelated Stage-G H_e 40K manifest (untouched).

## Harvest plan (due ~04:00 UTC 2026-07-05)
Attractor probe over the 6×92 checkpoints vs the rung-1/14M bands
(SCALE_TRANSFER_DESIGN §5.7 monotonic-trend criterion, stated as a joint
scale+data-mix result until the mixcontrol cells isolate the mix axis);
val-loss curves; rung-3 timing constants + go/no-go inputs (rung-3 needs
corpus augmentation ≥556M/482M more tokens OR fewer steps — user decision).
