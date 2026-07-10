# H2H 27-cell sweep harvest — archive MANIFEST + verdict of record (2026-07-10)

Harvest of the §1.38-launched 27-cell task1-primary sweep (3 archs × 3 tasks ×
3 seeds), scored under the FROZEN §1.31.1 tiers (MARGINS_FROZEN.token, pinned
2026-07-09T21:38:00Z, md5 58fe68e7b15728739a5176d7e591e204, copy in this
directory). Registry record: `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.40.

## Provenance

- Sweep session: tmux `h2h_sweep` (GPUs 0/1), supervisor log
  `h2h_sweep_supervisor.log` — 27/27 CELL COMPLETE, zero FATAL/strike lines,
  SWEEP_STOP written, session gone at harvest.
- 27 training JSONs (`h2h_{arch}_{task}_sweep_s{seed}.json`) pulled from
  `results/h2h_rung1/sweep/`; md5 verified local==box (`md5_manifest.txt`).
- The sweep stage-D script trained cells ONLY (its own header); the sweep
  training JSONs carry only the demoted online-probe fields, never `acc_A`.
  The verdict-grade reads were produced at harvest by `h2h_sweep_remetric_rd.py`
  (this directory; md5 e47d69fabdd51ba24896e76c5d13a3ab local==box): §1.31.4
  item 6's audited `run_cell_round4` applied verbatim to the sweep's 18
  grammar `_r4.pt` checkpoints (loader-side md5 provenance pinning; pinned
  EVAL_SEED episodes, identical across arms/seeds; task3 = LM control, no
  Leg-A applies). Independent pre-run audit: CLEAR-TO-RUN (7/7 findings safe).
  Outputs in `sweep_remetric/`, md5 verified local==box.
- Config-match over all 27 training JSONs: step_count=20000, budget_frac=1.0,
  role=sweep, lr=3e-4 (transformer task3 lr=1e-3 per the one freeze-time
  override), K=32 (task1), aux_weight=2.0 / ce_answer_weight=1.0 — NO
  deviations. Seeds identical across arms per (task, seed_idx).

## Per-cell table (grammar cells; acc_A = Leg-A primary, H_train; chance 0.03125, bar 0.09375)

| cell | acc_A | >bar | rung2-id | rf@0.9 | S₀ collapse / unchanged | instr | steps |
|---|---|---|---|---|---|---|---|
| contender_task1_s0 | 0.99951 | Y | 1.0000 | 0.6858 | pass / pass | P | 20000 |
| contender_task1_s1 | 1.00000 | Y | 0.9998 | 0.7710 | pass / FAIL* | P | 20000 |
| contender_task1_s2 | 0.99902 | Y | 1.0000 | 0.9512 | pass / pass | P | 20000 |
| ablation_task1_s0 | 0.03223 | n | 0.0183 | 0.0000 | pass / pass | P | 20000 |
| ablation_task1_s1 | 0.03271 | n | 0.0120 | 0.0000 | pass / pass | P | 20000 |
| ablation_task1_s2 | 0.03687 | n | 0.0146 | 0.0000 | pass / pass | P | 20000 |
| transformer_task1_s0 | 0.02710 | n | 0.0320 | 0.0000 | n/a | P | 20000 |
| transformer_task1_s1 | 0.02930 | n | 0.0310 | 0.0000 | n/a | P | 20000 |
| transformer_task1_s2 | 0.02856 | n | 0.0334 | 0.0000 | n/a | P | 20000 |
| contender_task2_s0 | 0.03955 | n | 0.0388 | 0.0000 | pass / pass | P | 20000 |
| contender_task2_s1 | 0.02905 | n | 0.0354 | 0.0000 | pass / pass | P | 20000 |
| contender_task2_s2 | 0.33447 | **Y** | 0.3364 | 0.0000 | pass / pass | P | 20000 |
| ablation_task2_s0 | 0.03369 | n | 0.0137 | 0.0000 | pass / pass | P | 20000 |
| ablation_task2_s1 | 0.02759 | n | 0.0129 | 0.0000 | pass / pass | P | 20000 |
| ablation_task2_s2 | 0.03687 | n | 0.0122 | 0.0000 | pass / pass | P | 20000 |
| transformer_task2_s0 | 0.02686 | n | 0.0308 | 0.0000 | n/a | P | 20000 |
| transformer_task2_s1 | 0.03223 | n | 0.0278 | 0.0000 | n/a | P | 20000 |
| transformer_task2_s2 | 0.03369 | n | 0.0337 | 0.0000 | n/a | P | 20000 |

\* s1 "unchanged" fail is a degenerate-σ instrument edge case, NOT a hard-stop:
acc_intact = 1.0 exactly → the pinned σ=√(p̂(1-p̂)/n) = 0 → 2σ = 0, so the
observed Δ(S₁-zeroed) = 0.00513 (21/4096 queries; recall still 0.99487) fails
the bar by construction. The HARD-STOP leg (S₀ collapse) passes decisively at
every recurrent cell (contender task1 S₀-zeroed: 0.0339/0.0012/0.0002);
`hard_stop_fires` = False everywhere (12/12 recurrent cells).

## AXIS-1 VERDICT (task1 primary, frozen tiers): **WIN**

- contender acc_A per-seed [0.99951, 1.00000, 0.99902], mean 0.99951 — clears
  the 3×-chance demonstration bar (0.09375) at EVERY seed (≥10.7× the bar).
- ablation per-seed [0.03223, 0.03271, 0.03687], mean 0.03394 — never clears.
- transformer per-seed [0.02710, 0.02930, 0.02856], mean 0.02832 — never clears.
- Δ(contender − ablation) per-seed [0.96729, 0.96729, 0.96216];
  mean 0.96558, sd 0.00296, paired t-CI (df=2, t=4.303) half-width 0.00735 →
  **CI (0.95822, 0.97293) — excludes 0.30** (margin cleared by 0.658 at the
  CI floor).
- Δ(contender − transformer) per-seed [0.97241, 0.97070, 0.97046];
  mean 0.97119, half-width 0.00264 → **CI (0.96855, 0.97383) — excludes 0.30.**
- Seed fragility (the pre-registered risk): NOT observed on task1 — per-seed
  contender scatter is 0.00098 total range; no seed fails the bar; no CI
  straddle. The n=3→9 extension trigger does NOT fire for the primary.
- Confirms §1.37's n=1 read (0.9990 / Δ 0.9543) at n=3 with tighter CIs.
- Nichani caveat travels with every acc_A number (§1.31.6): episode-restricted
  argmax recall, never a rank/continuous-capacity claim.

## TASK2 disposition — pre-registered joint-failure TIE, with a SURPRISE

Pre-registration (§1.31.1 Rev 5.1) scored task2 joint-failure TIE, reasoning
from the frozen round-3 checkpoint. The sweep's fresh-seed retrains break the
"deterministic failure" premise at ONE seed: contender_task2_s2 reads acc_A
0.33447 (10.7× chance, clears the bar; rung-2 identity 0.3364 tracks it).
Seeds 0/1 and all baseline arms stay at chance. Strict tier arithmetic:
Δ(contender−ablation) = [0.00586, 0.00146, 0.29761], mean 0.10164, CI
(−0.32001, 0.52330) — straddles ±0.30 → **INDETERMINATE** (TIE-adjacent,
never a WIN, escalation-eligible). Disposition: task2 remains non-verdict-
bearing for axis-1 per the frozen task1-primary pin; the s2 partial-recall
datum (evidence for trainability/seed-variance, against a hard capability
boundary) folds into the pre-registered TASK2 DIAGNOSIS ROUND. Secondary
H_test=(3,4) reads (disclosed, non-comparable): contender [0.0332, 0.0339,
0.0112] — the s2 cell does NOT generalize to held-out hops.

## Leg-B (diagnostic), S₀, task3, instruments

- Leg-B rf@0.9, contender task1: [0.6858, 0.7710, 0.9512] (cos_mean 0.9097/
  0.9211/0.9547); ablation task1: 0.0 at every seed. §1.38's reproduction
  bands ([0.624, 0.724] contender; [0, 0.05] ablation per ADJ-2) bound only
  REUSED round-3 checkpoints; all sweep cells are fresh seeds → first
  measurements (ADJ-3 logic). Note the large per-seed legibility variance
  (0.686→0.951) at flat acc_A — mechanism-attribution data, no decision input.
- contender_task2_s2: rf@0.9 = 0.0 despite acc_A 0.335 — the Nichani gap live
  at the deep tap for the partial-recall cell, as the §1.31.3 attribution
  table anticipates.
- Instrument health: 18/18 cells pass BOTH directions (planted-signal positive
  controls + noise nulls ≤1.5× chance); ridge harness sanity control exact-
  recovery pass 18/18.
- task3 (LM control lane, no WIN/LOSE input): final_val_loss_own — contender
  [2.1529, 2.1077, 2.1659], ablation [2.2920, 2.2851, 2.2904] (inside the
  anchored [1.90, 2.60] band; matches the 2.2905 calib anchor), transformer at
  the frozen lr=1e-3 override [1.7769, 1.7800, 1.7865] — 0.12 BELOW the
  calibration-era band floor (better-than-band; the full-budget run beat its
  own calibration point 2.1931). Recorded as a band-note, not a gate event
  (task3 bands were launch-gating only, pre-sweep).

## Realized GPU-h vs ceiling

- Sweep supervisor mechanical projection at completion: **9.598 GPU-h**
  (elapsed 17,276 s × 2 GPUs; the enforcement figure). Per-cell wall-sum:
  8.802 GPU-h. Harvest re-metric pass: **0.112 GPU-h** (19 evals incl. pilot,
  GPU 0). Total ≈9.71 vs the 13.25 ceiling — 3.5 GPU-h under.

## Files

- `h2h_*_sweep_s*.json` — 27 training JSONs (md5 `md5_manifest.txt`, local==box)
- `sweep_remetric/*.json` — 18 verdict-grade re-metric JSONs (md5 verified)
- `h2h_sweep_supervisor.log`, `h2h_sweep_remetric.log` — full logs
- `MARGINS_FROZEN.token` — the frozen tiers (md5 58fe68e7…)
- `h2h_sweep_remetric_rd.py` — the exact harvest eval script (audited, CLEAR-TO-RUN)
- `compute_verdict.py` — the exact verdict arithmetic script
- SSD mirror: `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-10_h2h_sweep_harvest/`
