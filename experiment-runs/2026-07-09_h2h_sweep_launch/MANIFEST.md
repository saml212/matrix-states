# 2026-07-09 HEAD-TO-HEAD — MARGIN FREEZE + task1-primary 27-CELL SWEEP LAUNCH

Registry record: `matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md` §1.38 (this archive's
source of truth). Gate discharged against: §1.37 ROUND 4 VERDICT (commit `1106055`,
TASK1-PRIMARY LEG-A WIN at n=1, S₀ hard-stop PASSED, Leg-B anchor reproduced).
Pre-registration executed: §1.31/§1.31a (Rev 5/5.1) chain — ladder/bands → MARGIN
FREEZE → 27-cell sweep.

## Contents

| file | what |
|---|---|
| `ladder_bands_review.json` | The §1.31.3/§1.31.2 ladder-bands review computed on the round-4 raws (verdict: ALL LAUNCH-BLOCKING BANDS PASS) + the 3 coordinator adjudications (ADJ-1 task2 pinned branch; ADJ-2 ablation Leg-B band mis-derivation tiebreak vs the raw §1.30 artifact → NO DRIFT; ADJ-3 stress-K48 scoping) |
| `MARGINS_FROZEN.token.json` | Verbatim copy of the token deployed to the box (`results/h2h_rung1/MARGINS_FROZEN.token`): §1.31.1 tiers frozen verbatim, bands, adjudications, `transformer_task3_lr=1e-3` (the one code-honored override), contender per_token pin + §13.13 discharge + cross-campaign disclosure, **pinned_at = 1783633080.929 = 2026-07-09T21:38:00Z** |
| `h2h_sweep_stage_d.sh` | The exact launch script deployed and running on the box (Stage-D-only extraction of `h2h_rung1_chain.sh`; rationale in its header + §1.38) |
| `round4_inputs/` | The 8 round-4 per-cell raws + `ROUND4_PROVENANCE_MANIFEST.json` + `FRESH_CELL_CONFIGS.json` the bands review was computed from (fetched from box `results/h2h_rung1/round4/`) |

## md5 (local == box, verified at deploy)

```
58fe68e7b15728739a5176d7e591e204  MARGINS_FROZEN.token(.json)
5fdb6ed57a05d4e5b3b81cdd3de68937  h2h_sweep_stage_d.sh
```

## Commands (box: youthful-indigo-turkey, /home/nvidia/chapter2/deltanet_rd)

```
# stale round-3 chain FATAL archived (adjudicated §1.27→§1.37; round-2 precedent):
mv results/h2h_rung1/FATAL results/h2h_rung1/FATAL_round3_stressK48_3strikes_archived_20260709T213915Z

# launch (exactly once), 2026-07-09 21:39:27Z:
tmux new-session -d -s h2h_sweep \
  "while [ ! -f results/h2h_rung1/SWEEP_STOP ] && [ ! -f results/h2h_rung1/FATAL ]; do \
     bash h2h_sweep_stage_d.sh >> logs/h2h_sweep_supervisor.log 2>&1; sleep 15; done"
```

## Bands summary (full table in §1.38 / ladder_bands_review.json)

- Gate-1 arm-aware (6 primary cells): contender task1 PASS (0.9990 > 0.09375); all
  baseline-arm rung-1 readings recorded as data (non-blocking by construction);
  instrument health PASS on all 8 cells (planted-signal positive controls +
  noise nulls ≤1.5× chance). contender task2 FAIL = the §1.31.1 pre-registered
  joint-failure-TIE branch (non-blocking, post-sweep diagnosis round).
- Leg-B reproduction: contender task1 0.6743 ∈ [0.624, 0.724] PASS; ablation task1
  0.0 == the raw §1.30 anchor (Rev 5.1's [0.069, 0.169] band was mis-derived from
  cos_mean; corrected to [0, 0.05] by §1.38 ADJ-2).
- S₀ hard-stop: clean on every recurrent-arm cell (no fires).
- Task3 anchored: ablation 2.2905 ∈ [1.90, 2.60]; lr_grid all in [1.90, 5.50], all
  decreased. LR decision: transformer_task3_lr = 1e-3 (best of 3.0245@1e-4,
  2.4161@3e-4, 2.1931@1e-3).

## Blind discipline (§13.13-verified pattern)

`key_anchoring.assert_blind_not_broken` wired at launch + negative test EXECUTED per
launch. From the launch log (verbatim):

```
blind check PASS: pinned_at=1783633080.928859 (2026-07-09T21:38:00Z) strictly precedes launch=1783633175.8489218; negative test exercised
```

## First-cell health (checked ~5 min post-launch)

- `h2h_contender_task1_sweep_s0` (GPU 0) + `_s1` (GPU 1): step 1500/20000, losses
  7.77→7.41 / 7.68→7.36, GPUs at 84-87% util, 9.7 GB each. Supervisor + token-probe +
  blind-check all green in `logs/h2h_sweep_supervisor.log`.

## Watch-path + ETA

- Harvest: box `results/h2h_rung1/sweep/h2h_{arch}_{task}_s{seed}.json` (27 expected),
  `results/h2h_rung1/SWEEP_STOP` on completion; per-cell logs `logs/h2h_sweep_h2h_*.log`;
  checkpoints `/data/h2h_rung1_ckpts/*_r4.pt`.
- Budget: 13.25 GPU-h ceiling (§1.26a re-price), mechanically enforced per wave.
- ETA on 2 GPUs: ≈5-7 h wall (upper bound 6.6 h at the ceiling) → completion
  ≈2026-07-10 03:00-05:00Z. To expand into freed GPUs: kill session, relaunch with
  `H2H_GPUS="0,1,…"` (resume-safe).

## NEXT (pre-registered, §1.38)

1. M* protocol (axis 2): 90-pass fan-out + contender horizon refs AFTER the sweep —
   re-verify `h2h_msweep_fanout_rd` against the acc_A re-registration and fix the
   chain's stale unsuffixed ckpt_map names (`_r4.pt` now) before dispatch.
2. TASK2 DIAGNOSIS ROUND + the K48 stress fresh cell, behind the sweep.
3. Sweep harvest → verdict-grade n=3 paired CIs → the §1.31.6 claim.

## Security

No fake `<system-reminder>` blocks observed in tool stdout this task. One
date-change notice arrived on the legitimate harness channel (not embedded in
command output) and was verified against the OS clock (`date` → 2026-07-09) and git
commit timestamps before being trusted, per the standing hard rule.

SSD mirror: `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-09_h2h_sweep_launch/`
