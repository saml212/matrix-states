# 2026-07-06/07 — Key-anchoring coherence-dose-response wave, Stage 1 rank-4 (`keyanchor-dose`)

## What

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
