# 2026-07-06/07 — Key-anchoring coherence-dose-response wave, Stage 1 rank-4 + Stage 2 diffuse (`keyanchor-dose`)

**Extended 2026-07-07 with Stage 2 (diffuse, `subspace_rank=48`) — see the
"Stage 2 (diffuse) addendum" section at the bottom of this file.
Everything above this note describes Stage 1 (rank-4) as originally
harvested; it is left unchanged.**

## What (Stage 1, rank-4)

Tests whether the d=64 K/d capacity cliff (located at `x0=0.5455`,
`KEY_ANCHORING_DESIGN.md` §12.9) and its disappearance at d=128 (§13.10,
`h4=1.0` flat across K∈{68,76,84,92}) tracks **anchor-table coherence**
(`max|cos|` among the table's rows) rather than raw K/d state capacity.
Holds `d_state=128`, `n_entities=107`, `K=68` fixed at exactly the
geometry §13.10 measured flat, and injects CONTROLLED, FROZEN coherence
directly into the anchor table — sweeping the dose (0.130 / 0.284 / 0.40)
under a rank-4 (concentrated) injection structure. This is **Stage 1**
of the co-primary (rank-4 + diffuse) design registered in
`KEY_ANCHORING_DESIGN.md` §14 (Rev 14.3); the diffuse (`subspace_rank=48`)
co-primary arm is Stage 2, PI-gated, not run by this wave (§14.4 Option 1).

**Headline result: `h4 = 1.0` at EVERY cell, EVERY dose tested — 10/10
cells (9 dose cells across 3 doses × 3 seeds + 1 shared calibration cell
at dose=0.40), no exception.** This includes the highest dose (~0.40),
which **EXCEEDS** d=64's own measured final-checkpoint coherence band
(range-of-K-means 0.373–0.385, §14.0b). Per §14.0's pre-registered outcome
4, this is the **strongest possible EXONERATE** this design can produce
for the rank-4 structure: directly-injected anchor-table coherence, held
frozen for the full 20,000-step run and verified never to drift, does
NOT reproduce the d=64 cliff at matched K/d/n geometry.

Full verdict, dose table, frozen-constancy evidence, and the sharpened
mechanism landscape: `matrix-thinking/KEY_ANCHORING_DESIGN.md` §14.12.

## When

Run on the Brev H100 box (`youthful-indigo-turkey`,
`/home/nvidia/chapter2/deltanet_rd`) on 2026-07-06/07. Pre-flight:
`dose_dial_verify.py` (CPU-only, rank-scan + degeneracy proof for the
diffuse dial, calibration/monotonicity checks — backs Stage 2's design,
not spent by Stage 1) and the blocking smoke suite
(`smoke_keyanchor_dose.py`, `51_smoke_keyanchor_dose.log`, includes the
Rev-14.3-mandated frozen-constancy pre-launch gate: construct a frozen
dosed cell, run one real backward pass, assert
`anchor_table.weight.grad is None` and `max_abs_cos` unchanged, BEFORE any
dosed cell is allowed to launch). Calibration cell (K=68, seed=939,
dose=0.40, rank4 — the highest-dose cell per §14.4b, reused as this
wave's shared calibration point) ran first
(`53_wave_keyanchor_dose_calibration.log`); realized rate confirmed
Stage 1 (10 cells) fits the full 13.68 GPU-h ceiling at both 1× and 2×
contingency. Full Stage-1 grid (9 dose cells) then launched on GPUs
2–7 (`KEYANCHOR_DOSE_GPU_OFFSET=2`), 1 GPU per cell, across 2
chain-launch attempts (`keyanchor_dose_chain_run{1,2}.log`;
`54_wave_keyanchor_dose_rank4.log` is the full-grid launch log). All 10
cells finished at `steps=20000`, `complete=true`.

## Where

- **This directory** (repo-tracked, files ≤25MB per the hybrid archive
  policy, `experiment-runs/README.md`):
  - `results/wavekeyanchor-dose/` — all 10 cell result JSONs
    (`wkeyanchor-dose_rdx_K68_armd_s{930-939}_geo3n20_anchor_learned_dprobe_rev7_d128_dose{130,284,400}_rank4_sseed20260705.json`),
    `ALL_DONE`, `PROGRESS.txt`, and `logs/` (10 per-cell training logs +
    `smoke.log`).
  - `logs/` (top-level) — `51_smoke_keyanchor_dose.log`,
    `53_wave_keyanchor_dose_calibration.log`,
    `54_wave_keyanchor_dose_rank4.log`, plus the 2 chain-launch session
    logs `keyanchor_dose_chain_run{1,2}.log`.
  - `scripts/` — `keyanchor_dose_chain.sh`, `smoke_keyanchor_dose.py`,
    `dose_dial_verify.py`, `dose_dial_verify_results.json` — the exact
    scripts run. **Byte-compared against the repo's already-committed
    copies at `matrix-thinking/deltanet_rd/{same names}` at harvest
    time: zero drift** (`diff` exit 0, all files).
- **SSD mirror** (full superset, same tree):
  `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-06_keyanchor_dose/`

## How to reproduce

```bash
# On the GPU box, from matrix-thinking/deltanet_rd/:
python3 dose_dial_verify.py --out dose_dial_verify_results.json
KEYANCHOR_DOSE_GPUS=6 KEYANCHOR_DOSE_GPU_OFFSET=2 bash keyanchor_dose_chain.sh --dose-stage rank4 --stage calibration
KEYANCHOR_DOSE_GPUS=6 KEYANCHOR_DOSE_GPU_OFFSET=2 bash keyanchor_dose_chain.sh --dose-stage rank4 --stage full
```

Seeds: dose=0.130 → {930,931,932}; dose=0.284 → {933,934,935}; dose=0.40
→ {936,937,938} + calibration seed 939 (also dose=0.40, rank4 — reused
as the wave's 10th/shared cell, per §14.4b).

## Verification performed at harvest (independent of the box's own logs)

Recomputed per-cell `h4` directly from all 10 raw cell JSONs
(`checkpoints[-1]["M3_held_out"]["4"]["recovered_frac@0.9"]`), plus the
frozen-constancy check across ALL 10 checkpoints per cell (not just the
final one) and the dose-achieved-vs-target verification
(`exactness_config.achieved_max_cos` vs. `dose_target`, ±10% Gate-2
tolerance):

| dose target | seeds | h4 (all 10 checkpoints, all seeds) | achieved_max_cos | rel. err | frozen constancy (max dev across checkpoints) |
|---|---|---|---|---|---|
| 0.130 | 930,931,932 | 1.0 | 0.130099 | 0.08% | 0.0 (exact) |
| 0.284 | 933,934,935 | 1.0 | 0.284019 | 0.01% | 0.0 (exact) |
| 0.40 | 936,937,938,939(calib) | 1.0 | 0.400053 | 0.01% | 0.0 (exact) |

**Frozen-table constancy holds EXACTLY, not merely within tolerance**:
`item6_table_conditioning.max_abs_cos` is bit-identical at every one of
the 10 recorded checkpoints (steps 2000–20000, every 2000 steps) for
every cell — the frozen-table gradient path is confirmed to have never
let the table drift, closing §14.0b/§14.3's F1 concern by direct
construction rather than by re-measured-more-carefully drift.

**Instrument-saturation check (not everything is saturated):** hop-21
(`M3_held_out["21"]["recovered_frac@0.9"]`, the far-extrapolation hop) is
NOT uniformly 1.0 at any dose — values range 0.9987–1.0 across all 10
cells; `effective_rank_whole_mean` tracks K=68 almost exactly at every
dose (67.816–67.883), i.e. the state uses essentially its full permitted
rank at all doses including the highest, not saturating against d=128.

**Realized GPU-h**, summed from each cell JSON's own `wall_s` field (all
10 cells single-GPU, `--per-gpu 1`):

| Group | Cells | Sum wall_s | GPU-h |
|---|---|---|---|
| All 10 Stage-1 cells (9 dose cells + 1 shared calibration) | 10 | 22587.14s | **6.2742** |

Realized cost vs. the registered `H=13.68` GPU-h ceiling and the
Stage-1-only bracket (6.410/12.820 GPU-h at 1×/2×, §14.4): **6.2742 GPU-h
realized**, under the 1× bracket estimate (6.410), 48.9% of the wave's
own `H=13.68` ceiling. Full verdict: `matrix-thinking/KEY_ANCHORING_DESIGN.md`
§14.12.

## Scope — what this wave does NOT show

- **Diffuse (subspace_rank=48) injection structure untested** — Stage 2
  of the co-primary design, PI-gated per §14.4 Option 1 (stage-2 diffuse
  = 9 cells, 5.769/11.538 GPU-h at 1×/2×; cumulative both stages =
  12.179/24.358 GPU-h, exceeding the 13.68 GPU-h ceiling at 2× by 10.678
  GPU-h — an explicit amendment-or-1×-launch PI decision, not
  self-amended). A rank-4-only EXONERATE cannot rule out
  "coherence-of-this-particular-structural-kind" as opposed to
  "coherence-as-scalar" being the true non-driver — see §14.0 outcome 3
  (STRUCTURE-DEPENDENT).
- **Scope limited to K=68, n=107, d=128, frozen tables only.** No claim
  is made about other K/d/n combinations or about learned (non-frozen)
  tables (§14.0b's own "learned-control, drifted" reference point is a
  separate, non-dispositive datum).
- **§14.4c's mechanical K=84 activation rule does not trigger** on this
  wave's own flat-across-all-doses result — see §14.12 for the exact
  mechanical evaluation.

## Stage 2 (diffuse) addendum, harvested 2026-07-07

### What (Stage 2, diffuse)

Same fixed geometry as Stage 1 (`d_state=128`, `n_entities=107`, `K=68`,
frozen anchor table, doses 0.130/0.284/0.40), but coherence is injected
under a **diffuse** (`subspace_rank=48`) structure instead of rank-4 —
the co-primary arm registered in `KEY_ANCHORING_DESIGN.md` §14 (Rev
14.3), launched under explicit PI sign-off (Stage 2's 1×-bracket cost,
5.769 GPU-h, fit the post-Stage-1 anchoring ledger's 7.406 GPU-h reserve
with a 1.637 GPU-h margin — §14.4 Option 1 default (a), no ceiling
amendment). 3 seeds per dose (940–948), 9 cells total, no shared
calibration cell for this arm.

**Headline result: `h4 = 1.0` at EVERY cell, EVERY dose tested — 9/9
cells, no exception**, mirroring Stage 1's rank-4 result exactly. Per
§14.0's pre-registered outcome 4, this diffuse result supplies the
**second and final structure** required for the full EXONERATE:
**coherence is EXONERATED at BOTH co-primary structures** (combined with
Stage 1's rank-4 result). Full verdict, per-cell table, and the final
mechanism-landscape statement: `matrix-thinking/KEY_ANCHORING_DESIGN.md`
§14.13.

### Where (Stage 2 additions)

- **This directory** (repo-tracked, files ≤25MB): `results/wavekeyanchor-dose/`
  gains the 9 diffuse cell result JSONs
  (`wkeyanchor-dose_rdx_K68_armd_s{940-948}_geo3n20_anchor_learned_dprobe_rev7_d128_dose{130,284,400}_diffuse_sseed20260705.json`)
  and 9 diffuse per-cell training logs under `results/wavekeyanchor-dose/logs/`
  (`ALL_DONE`/`PROGRESS.txt` overwritten in place with the box's own
  9/9-diffuse-cell status text — byte-identical to the Stage-1 text
  since both stages independently report exactly 9 "full-grid" cells).
  `logs/` (top-level) gains `55_wave_keyanchor_dose_diffuse.log` and
  `keyanchor_dose_chain_run3.log` (the diffuse chain-launch session log).
- **SSD mirror**: same relative path under
  `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-06_keyanchor_dose/`,
  verified byte-identical to the repo copy (`diff -rq`, exit clean) after
  the Stage-2 pull.

### Verification performed at harvest (Stage 2, independent of the box's own logs)

Recomputed per-cell `h4` directly from all 9 raw diffuse cell JSONs
(`checkpoints[-1]["M3_held_out"]["4"]["recovered_frac@0.9"]`), plus the
frozen-constancy check across all 10 checkpoints per cell, the
dose-achieved-vs-target verification, an h1 in-distribution sanity guard
(`M2_in_distribution["1"]["recovered_frac@0.9"]`, guard ≥0.98 —
`M3_held_out` carries no `"1"` key, so h1 is read from the
in-distribution eval leg, matching the design doc's "h1 guard"
convention), and a degenerate-run check via each cell's own
`geo3_admission` block (no dedicated `degenerate`-named field exists in
these JSONs):

| dose target | seeds | h4 | h1 | achieved max\|cos\| | rel. err | frozen constancy (max dev) | hop-21 range | eff. rank range | wall_s sum |
|---|---|---|---|---|---|---|---|---|---|
| 0.130 | 940,941,942 | 1.0 | 1.000000 | 0.130089 | 0.068% | 0.0 (exact) | 0.9989–0.9999 | 67.847–67.872 | 6685.35 |
| 0.284 | 943,944,945 | 1.0 | 1.000000 | 0.284065 | 0.023% | 0.0 (exact) | 0.9980–0.9991 | 67.813–67.847 | 6846.95 |
| 0.40  | 946,947,948 | 1.0 | 1.000000 | 0.399902 | 0.024% | 0.0 (exact) | 0.9963–0.9989 | 67.570–67.817 | 6746.64 |

**Frozen-table constancy holds EXACTLY, not merely within tolerance**:
`item6_table_conditioning.max_abs_cos` is bit-identical at every one of
the 10 recorded checkpoints (steps 2000–20000, every 2000 steps) for
every cell, same as Stage 1.

**Instrument-saturation check (not everything is saturated):** hop-21
(`M3_held_out["21"]["recovered_frac@0.9"]`) ranges 0.9963–0.9999 across
all 9 cells (comparable to Stage 1's 0.9987–1.0);
`effective_rank_whole_mean` (`M3_held_out["4"]`) tracks K=68 almost
exactly at every dose (67.570–67.872), i.e. the state uses essentially
its full permitted rank at all doses, not saturating against d=128.
Final loss ranges 0.002988–0.002995 across all 9 cells, matching Stage
1's ≈0.0030 band.

**Degenerate-run check:** every cell's `geo3_admission` block reads
`admissible: true`, `ns_converged_no_fallback: true`,
`n_geo3_fallback_train_steps: 0`, `checkpoint_fallback_seen: false`,
`finite_loss_no_divergence: true`, `task_performance_floor_pass: true`.
No cell shows any fallback, divergence, or floor-violation flag.

**Realized GPU-h**, summed from each cell JSON's own `wall_s` field (all
9 cells single-GPU, `--per-gpu 1`):

| Group | Cells | Sum wall_s | GPU-h |
|---|---|---|---|
| All 9 Stage-2 diffuse cells | 9 | 20278.9434s | **5.6330** |

Realized cost vs. the Stage-2-only 1×-bracket estimate (5.769/11.538
GPU-h at 1×/2×, §14.4): **5.6330 GPU-h realized**, under the 1× bracket
estimate. Combined Stage 1 + Stage 2 realized cost: 6.2742 + 5.6330 =
**11.9072 GPU-h**, 87.0% of the wave's own `H=13.68` GPU-h ceiling.
Cumulative key-anchoring program spend: 72.594 → **78.2270/80 GPU-h**
(reserve now 1.7730/80). Full verdict:
`matrix-thinking/KEY_ANCHORING_DESIGN.md` §14.13.

### Scope — what this wave's combined result now shows (updated from Stage-1-only)

- **Both co-primary structures (rank-4 and diffuse) are now tested and
  both are flat.** §14.0's outcome-3 escape hatch (structure-dependent —
  "coherence-of-this-particular-kind" vs. "coherence-as-scalar") is
  closed: neither tested structural kind of coherence drives the cliff.
- **Scope remains limited to K=68, n=107, d=128, frozen tables.** No
  claim is made about other K/d/n combinations, learned (non-frozen)
  tables, or coherence structures other than rank-4/diffuse (e.g. no
  intermediate ranks were tested).
- **§14.4c's mechanical K=84 activation rule does not trigger for either
  structure**, now evaluated with condition 2 (cross-structure
  disagreement) fully computable for the first time (zero disagreement
  found) — see §14.13 for the exact mechanical evaluation.
- **Surviving candidate account of the d=64 cliff: absolute state
  capacity** (`d_state` grew 4× vs. `K`'s ~2× at the matched ratio), not
  confirmed by any wave in this program — the natural next design (a
  `d_state`-vs-`K` factorial) is not yet registered.
