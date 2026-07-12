# 2026-07-11 — h2h FIX-5 transformer-task1 LEARNING-RATE GRID

Registry: `matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md` **§1.44** (pre-run
record, committed `11bc5aa` BEFORE launch) and **§1.45** (the verdict of
record). Charter: flagship rebuttal FIX-5
(`papers/flagship/gauntlet/round-1/04_rebuttal_report.md`, lines 83-93) —
the pre-specified measurement resolving A1's SERIOUS residual (the
round-3/4 sweep's transformer arm on task1/recall trained at the untuned
shared default lr=3e-4, never LR-searched on the recall task itself).

**VERDICT: OUTCOME A — TUNED_TRANSFORMER_STILL_BELOW_BAR at every LR.**
No LR's 3-seed mean acc_A clears the frozen §1.31.1 demonstration bar
(0.09375 = 3× chance at K=32); 0 of 12 individual (lr, seed) cells clear
it either. Per FIX-5's pre-stated branch, the capability separation
STRENGTHENS to a two-baseline result — the transformer joins the
parameter-matched vector-state ablation as a second baseline that fails
recall even after a 4-point LR search (10⁻⁴–3×10⁻³) × 3 seeds × 20,000
matched steps under the identical frozen protocol. The §1.40 axis-1 WIN
verdict is UNCHANGED (this round was diagnostic/disclosure-grade by
pre-registration; the ablation comparison remains the registered carrier).

## The 12-cell acc_A table (chance 0.03125, bar 0.09375; ★ = reused verbatim from the §1.40 sweep)

| lr | s0 | s1 | s2 | mean | n clearing | vs chance |
|---|---|---|---|---|---|---|
| 1e-4 | 0.0110 | 0.0186 | 0.0132 | 0.0142 | 0/3 | 0.46× (6.3σ BELOW) |
| 3e-4 ★ | 0.0271 | 0.0293 | 0.0286 | 0.0283 | 0/3 | 0.91× |
| 1e-3 | 0.0352 | 0.0298 | 0.0310 | 0.0320 | 0/3 | 1.02× (best) |
| 3e-3 | 0.0291 | 0.0264 | 0.0295 | 0.0283 | 0/3 | 0.91× |

**Key finding beyond the binary:** optimization and recall dissociate —
lr=1e-4 optimizes the LM objective best (final loss 6.54-6.60 vs the
default's ~7.5) yet reads acc_A 6.3σ BELOW chance, directly falsifying
FIX-1's "under-optimized arm" as the explanation for the chance-level
recall.

## Contents

- `results/` — 9 raw training JSONs (`h2h_fix5_transformer_task1_lr{1e-04,1e-03,3e-03}_s{0,1,2}.json`,
  full 40-point curves), `results/remetric/` 9 verdict-grade re-metric
  JSONs (audited `run_cell_round4`, md5 provenance-pinned; instrument
  health + ridge sanity PASS on all 9), `FIX5_LRGRID_VERDICT.json`
  (the harvested per-lr table + pre-registered decision rule; md5
  `c71eacc8185bcec1ea27e86b20afd559`), the smoke record.
  The 3e-4 column is NOT in this dir — it is cited verbatim (per §1.44's
  provenance record) from `experiment-runs/2026-07-10_h2h_sweep_harvest/`
  (raw `h2h_transformer_task1_sweep_s{0,1,2}.json` + remetric
  `sweep_remetric/..._round4.json`).
- `logs/` — supervisor + token-probe + 9 per-cell + smoke + remetric logs.
- `md5_manifest.txt` (box-side) / `md5_manifest_local.txt` (local);
  diff verified 19/19 MATCH at harvest.

## Deployed scripts (in the repo, not duplicated here)

- `matrix-thinking/deltanet_rd/h2h_fix5_lrgrid_rd.py` (md5
  `927ee15b1374ffd2d0e4119ff527f2a1`), `h2h_fix5_stage.sh` (md5
  `4d6bf84e69223ec4cc9d0c6f11aa001c`) — the EXACT deployed scripts,
  scp + md5-verified local==box at deploy. Both committed (`a5d2c7b`).

## Run provenance

- Box: youthful-indigo-turkey, GPUs 0-1 only (GPUs 2-7 = NCR Phase-2,
  untouched — confirmed idle before launch; the stage script hard-refuses
  any GPU in 2-7). tmux `h2h_fix5_grid` with the self-healing supervisor
  loop; 9 fresh cells 2-way parallel; zero failcount strikes.
- Reuse (disclosed, not silent): the lr=3e-4 × {s0,s1,s2} column was NOT
  relaunched — it is byte-identical in every FIX-5-fixed dimension to the
  already-completed round-4 sweep's own transformer×task1_sweep cells, so
  it is READ from those artifacts. Only the other 3 LR columns (9 cells)
  trained fresh. Seeds byte-identical to the reused cells
  (`rd_episode_seed("task1_sweep", ·)`: 1,000,000/1,010,000/1,020,000) —
  a paired grid isolating LR as the sole treatment variable per seed lane.
- Gates honored: gate-5 tokens (GATE6/GATE7 present), MARGINS_FROZEN
  present, code-level `require_launch_tokens` + `require_margins_frozen`
  per cell, `H2H_DIAL_ROUND=4` (filename versioning; role="sweep" → the
  three-loss dial structurally never evaluated, AUD2-F4).
- Two-stage measurement (identical to every prior h2h acc_A number):
  TRAIN via `train_grammar_cell` (persists curve + checkpoint,
  save-before-rung2 ordering) → RE-METRIC via `run_cell_round4`
  (fresh=False, provenance-pinned load) for the audited acc_A.
- Checkpoints (57.8 MB each, BOX/SSD-only per the >25MB hybrid archive
  policy): `/data/h2h_rung1_ckpts/h2h_fix5_transformer_task1_lr{1e-04,1e-03,3e-03}_s{0,1,2}_r4.pt`.
- Realized ≈2.53 GPU-h vs the 6.0 ceiling (§1.44 projected ≈2.44).
- Does NOT edit `papers/flagship/` — the §7 disclosure upgrade routes on
  this verdict via a separate dispatch.
- Security: zero fake system-reminder blocks in tool stdout this round.
