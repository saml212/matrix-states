# Paper brief — Three Bounds on a Null (reasoning-link null program)

## Venue

- **Name:** MOSS @ COLM 2026 (Methods and Opportunities at Small Scale),
  Small-Scale Frontier Track, **late window**.
- **Format:** 4 pages max main content (references and supplementary do not
  count); COLM 2026 official template (GitHub tag `2026`); single column —
  from `papers/reasoning-null-moss/venue-requirements.md`, live-fetched
  2026-07-10.
- **Requirements source:** `papers/reasoning-null-moss/venue-requirements.md`
  (evidence URLs + fetch date inside). Not a cache fallback; the CFP was live.
- **Review style:** double-blind (OpenReview) → anonymization grep required.
- **Archival:** non-archival, no proceedings; dual submission allowed.
- **Deadline:** PASSED (Jul 3, 2026 AoE). Entry is the capacity-gated late
  window; **BLOCKER: the late-add email to the organizers is a PI decision —
  not sent by this run.** Package prepared for immediate submission on a yes.

## Thesis (one falsifiable sentence)

> In DeltaNet-family language models spanning 14M to 1.31B parameters, a
> pre-registered instrument battery finds no evidence that fast-weight
> write geometry predicts or causally improves in-context multi-hop
> composition, and the null is bounded three ways — a readout-validity gate
> that fails at all 366 of 366 readings across three structurally different
> instruments, a pre-registered n=3 power floor of 1.5–1.7 loss units on the
> causal contrast, and a pre-registered batch-effect gate under which the
> program's single determinate transient fails to replicate at n=12 — with
> each bound naming the concrete observation that would overturn it.

Falsifiers, per bound: (1) any nonzero recovered fraction clearing its
correspondence nulls, or a premise-gate pass, anywhere; (2) a determinate
arm contrast exceeding the floor; (3) a clean pooled n=12 CI excluding
zero. None occurred.

**Claim-shape correction (2026-07-11, `REASONING_LINK_DESIGN.md` §17.1–§17.7,
routes to this revision by charter).** Bound 1's readout carried a
`[K,V]`-vs-`[V,K]` state-layout transpose defect, caught by a pre-submission
positive control (commit `8666aee`, gauntlet round-1 FIX-6 tier 2) that this
paper's own limitations section had flagged as missing. The defect was
root-caused, fixed, and independently audited (`fb9950c`); the fixed
instrument does **not** reproduce the pre-fix zero on the largest sub-grid
(78/320 nonzero, wave-1/wave-2 cells) — but two independent, pre-registered
correspondence nulls (a label-shuffle null and a positional derangement
null, `c3de797`/`b3c5c4e`) reproduce that apparent signal at every one of
those 320 readings. The old claim ("recovery reads exactly zero") is DEAD
for the re-verified 320 readings; the valid claim is "recovery is
statistically indistinguishable from a broken correspondence" at every
scale, corpus, and hop the corrected instrument was run on. Waves 3–4 (46
of the 366 readings) share the identical pre-fix function and were not
independently re-verified; their probe-invalid routing is unaffected
because it rests on the alignment-premise gates, which the defect never
touched, not on the raw zero reading — disclosed as an open item, not
silently extended by analogy. See evidence rows C10–C12 below.

## Contribution bullets

1. **A triply-instrumented geometric null that survived its own positive
   control.** Three structurally different deployments of a state-space
   composition readout (zero-shot marker template, zero-shot natural
   language, task-familiarized) return a recovered fraction of exactly 0.0
   at 366/366 readings on checkpoints from 14M to 1.31B parameters, with the
   readout's own pre-registered validity gates failing everywhere. A
   pre-submission positive control on the readout's production code then
   failed outright, exposing a state-layout defect; after the fix, the
   largest sub-grid (320 of the 366 readings) no longer reads zero, but two
   independent, pre-registered correspondence nulls reproduce that apparent
   signal at every reading — the corrected claim is that recovery is
   null-indistinguishable, not that it is absent, and the gate-failure
   routing to probe-invalid, never to an unlicensed refutation, stands
   either way.
2. **A power-bounded causal null with one bounded transient.** The
   behavioral (vocabulary-space) contrast bounds any causal effect of the
   write-geometry intervention below the pre-registered n=3 detection floor
   (≈1.5–1.7 loss units, the same order as the entire task-learning effect);
   the single determinate signal is a transient, harm-direction deviation at
   one (corpus, arm, checkpoint) cell that dissolves by the end of training.
3. **A replication attempt stopped by its own integrity gate.** A targeted
   n=12 seed extension of the transient triggers its pre-registered
   batch-effect gate (cohort variance ratio 4.47 against a pinned 4.0
   cutoff); the new n=9 cohort's own CI spans zero, and the design's routing
   rule reports the gate-fail rather than a silently pooled verdict —
   neither confirming nor refuting the transient at its original weight.

Distinction from nearest neighbors in one clause each: unlike capacity-law
work (Based), the object is an observable's downstream validity, not a
capacity frontier; unlike recall-gap diagnoses (Zoology), the checkpoints
are general-purpose LMs probed post hoc, and the finding is about the
instrument battery, not an architecture gap; unlike causal
Transformer-vs-SSM accounts (Okpekpe & Orvieto), the claim is a bounded
null on one named mechanism, not a positive mechanism claim.

## Per-section page budget (sums to 4.0, the venue main-content limit)

| Section | Pages | Purpose |
|---|---|---|
| 1 Introduction | 0.75 | the keystone question, why a bounded null is the honest deliverable, contributions, the instrument-fix falsifier |
| 2 Setting and instruments | 0.65 | models/ladder, task grammar, the three readouts, pre-registration + gate discipline, correspondence-null pointer |
| 3 Bound 1: the geometric readout never fires | 1.05 | the instrument-fix case study (positive control, re-metric, both correspondence nulls), premise gates, scale series, waves 3–4 disclosure, the gate-enforcement arc |
| 4 Bound 2: the behavioral contrast is power-bounded | 0.60 | 3/4 UNRESOLVED at the registered floor; the transient, sign-disciplined (unchanged, round-1 verdict) |
| 5 Bound 3: the replication gate | 0.55 | n=12 wave, batch-effect flag, cohort CIs, what stands (unchanged, round-1 verdict) |
| 6 Related work | 0.20 | Based / Zoology / Okpekpe–Orvieto / Nichani et al., by name |
| 7 What is and is not bounded | 0.20 | scope, alternative explanations, lessons for small-scale practice, conclusion |
| **Total** | **4.00** | |

The extra ≈0.35pp Section 3 needs is recovered by compressing the old
wave-1/wave-2 zero-reading prose it replaces (into a single short
paragraph) and by moving the full re-verification numeric detail (per-null
tables, mechanism covariates, audit trail) to Appendix A, which does not
count against the 4pp limit; Section 3 keeps the headline numbers and the
claim-shape conclusion in the budgeted body.

Abstract: 200–230 words (styleguide band), on page 1 inside the budget.

## Claims-to-evidence-to-figure map

Archive root: `experiment-runs/` (repo-committed copies; SSD mirror is the
superset archive). "Manifest-md5" = md5 of the sorted `md5  path` line list
over the named file set (computed 2026-07-10; the figure script re-asserts
per-file md5s at build time).

| Claim id | Claim (with the number) | Verdict record (§ / log entry) | Raw artifact (path + md5) | Figure / table |
|---|---|---|---|---|
| C0-ladder | Four-rung scale ladder 14M/98M/392M/1.31B (d_state 64/64/128/128; n_layers 2/12/16/22; d_model up to 2560), two corpora (openr1-mix-ext, wikitext-mix-ext) | REASONING_LINK_DESIGN.md §6.1; §16.11 (rung-3 config read from ckpt_config) | `ckpt_config` fields inside the C1/C2 raw JSONs (same files/md5s as those rows) | Table 1 |
| C1-phase1 | Zero-shot marker-template probe: recovered_frac@0.9 exactly 0.0 at 312/312 (cell,h) readings (78 cells × h∈{1..4}) **under the pre-fix instrument**; premise (iii) passes 0/78, premise (iv) 0/78 (premise gates read k/v/q hooks, not the state-extraction function C10 fixed, so this half is unaffected by the fix). **Claim-shape note:** the raw zero reading is superseded by C11/C12 for this same cell set — see the thesis's claim-shape correction | REASONING_LINK_DESIGN.md §15.1–§15.4; EXPERIMENT_LOG.md 2026-07-07 "REASONING-LINK PHASE 1 HARVEST"; superseded by §17.1–§17.7 | `experiment-runs/2026-07-07_reasoning_link_phase1/results/leg_*.json` (78 files, manifest-md5 `3c885fbe49ada8cb62afb33f3f6faf60`) | Fig 1; Table 1 |
| C2-rung3 | Rung-3 (1.31B) completes the scale series: 8/8 readings 0.0 **under the pre-fix instrument**; both premise gates fail at both corpora (unaffected by the fix, see C1). Combined with C1 this is the exact 80-cell/320-reading grid C10–C12 re-verify | REASONING_LINK_DESIGN.md §16.11; EXPERIMENT_LOG.md 2026-07-07 rung-3 entry; superseded by §17.1–§17.7 | `experiment-runs/2026-07-07_reasoning_link_rung3/results/leg_b_rung3_openr1-mix-ext_s0_k64.json` md5 `b75745af6ae31632af61b077dfdd539b`; `...wikitext-mix-ext_s0_k64.json` md5 `13ecd7c6b90cd0696771bcae474f0820` | Fig 1; Table 1 |
| C3-phase1b | Natural-language templates (2 candidates × 2 corpora): 16/16 readings 0.0, `gate_result_h1_probe_valid` False at 4/4 cells; the enforcement script refused the full 78-cell grid. **Disclosed limitation:** this wave calls the identical pre-fix state-extraction function C10 found defective and was NOT independently re-run under the fixed instrument or tested against either correspondence null; its probe-invalid routing is unaffected because it rests on the alignment-premise gates alone, but the raw zero reading itself carries the same unconfirmed status C1/C2 carried before re-verification | REASONING_LINK_DESIGN.md §16.8.1–§16.8.4; EXPERIMENT_LOG.md 2026-07-07 Phase-1b entry | `experiment-runs/2026-07-07_phase1b_gate/results/stage0_natural_*.json` (4 files, manifest-md5 `f3ec8b2a89f19645f84bcaa385b4efd4`); sentinel `results/STAGE0_GATE_REFUSED` | Fig 1; Table 1 |
| C4-dissoc | Task-familiarized: gate fails 30/30 (cell, checkpoint) readings (recovered_frac(h1)=0.0, 0/512 queries each) while the vocab-space `L_query` falls 21.8–46.4% (mean 35.9%) in 6/6 cells — a vocabulary/geometry dissociation. **Disclosed limitation:** same as C3-phase1b — identical pre-fix function, not independently re-verified; probe-invalid routing rests on the premise gates alone | REASONING_LINK_DESIGN.md §16.15.1–§16.15.2 | gate: `experiment-runs/2026-07-08_phase2_familiarization/ckpts/stage05_gate_*.json` (30 files, manifest-md5 `85d381404dd89769eda382beacbee673`); trajectories: `results/off_*.json` (6 files, manifest-md5 `091c33c4786048cd3f94bb75496465ec`) | Fig 2; Table 1 |
| C5-power | Behavioral contrast at n=3: 3/4 (corpus × arm) contrasts UNRESOLVED; pre-registered detectable floor ≈1.5–1.7 loss units (σ proxy 0.43–0.48, t(2,.975)=4.303), the same order as the OFF arm's whole familiarization effect ≈1.69 | REASONING_LINK_DESIGN.md §16.16.4 (registered pre-harvest); §16.18.6 | `experiment-runs/2026-07-08_phase2b/results/trajectory_wikitext-mix-ext_phase2b.json` md5 `5c727ac669aea02601790b4fb1dac8b4`; `trajectory_openr1-mix-ext_phase2b.json` md5 `bdbc2352172b79b206d9d7bf7be303fe`; `PHASE2B_SUMMARY.json` md5 `9f5b14f372187ab181c1981158a23f14` | text §4 + Fig 3 |
| C6-transient | The single determinate signal: wikitext×per_token classifies TRANSIENT — Δ(K=32, c=2500) = −0.500, CI [−0.624, −0.376] (harm direction: arm loss above control), dissolving by c=5000 (Δ=−0.795, CI spans zero); every determinate K=32 reading in the 20-checkpoint primary table is negative; the held-out-hop secondary readout fires at the same cell/checkpoint, same direction | REASONING_LINK_DESIGN.md §16.18.3–§16.18.6 | `trajectory_wikitext-mix-ext_phase2b.json` (md5 above; `per_arm.per_token.raw` + `holds_by_c`) | Fig 3 (left) |
| C7-replication | n=12 seed extension: batch-effect gate FLAGGED at the anchor (var_ratio 4.47 > 4.0 pinned; sd_old 1.154 vs sd_new 0.546); old-cohort CI [−0.624, −0.376] (the archived n=3), new-cohort (n=9) CI [−0.506, +0.357] spans zero; naive n=12 pool CI [−0.509, +0.147] spans zero (diagnostic only, non-decision-grade per the registered routing); outcome BATCH-EFFECT-FLAGGED, bucket null; across the full tables 7/10 primary and 7/10 held-out (checkpoint, K) pairs are computable and every computable pooled CI contains zero | REASONING_LINK_DESIGN.md §16.19.8 (pre-registered MECE partition + gate), §16.20.2/§16.20.5 | `experiment-runs/2026-07-08_phase2b_seedext/results/trajectory_seedext_wikitext_n12.json` md5 `989f9997c50973cc299f25d89efff64f`; `PHASE2B_SEEDEXT_SUMMARY.json` md5 `fe48bb35533c030ce5f352ad607e244e` | Fig 3 (right) |
| C8-teeth | Gate-enforcement arc: Phase 1's chain computed its Stage-0 abort gate but had no enforcing code path (≈0.29 GPU-h ran past a probe already flagged invalid); Phase-1b/Phase-2/Phase-2b chains enforce their gates at the process boundary and refused launches on real failing results | REASONING_LINK_DESIGN.md §15.2, §16.8.4, §16.15 headline; EXPERIMENT_LOG.md LEARN entry 2026-07-07 | `experiment-runs/2026-07-07_reasoning_link_phase1/reasoning_link_chain.sh` (grep: no gate reference); `experiment-runs/2026-07-07_phase1b_gate/results/STAGE0_GATE_REFUSED` + `logs/phase1b_run1.log` | prose (§3) |
| C9-compute | The complete program consumed ≈9.5 GPU-hours on at most two GPUs at a time (per-wave: 0.373 / 0.024 / 0.017 / 0.617 / 1.661 / 6.819) | Realized-cost records: REASONING_LINK_DESIGN.md §15.7, §16.8.5, §16.11, §16.15.6, §16.18.7, §16.20.8 | machine-readable: `PHASE2B_SUMMARY.json` (`elapsed_s=2833, n_gpus=2`), `PHASE2B_SEEDEXT_SUMMARY.json` (`elapsed_s=4119, n_gpus=2`), `rung1_seedext_*_summary.json` (Σ wall_s); log-timestamp-derived for the small waves (archived logs in the C1/C3 archives) | Appendix B (reproducibility compute disclosure) |
| C10-instrumentfix | Pre-submission positive control on the production readout FAILS pre-fix (0/256 recovered at every h∈{1..4}, cos_mean≈1.2e-5, while a deliberately-transposed variant recovers 1.0000/cos_mean≈0.99999 through the identical scoring code); closed-form adjudication PRE-FIX confirms `squeeze_state_head` returns fla's raw `[K,V]` layout instead of the `[V,K]` design layout (pre-fix: `S_squeeze_vs_closed_form_fla_rel_fro`=0.0, `S_squeeze_vs_closed_form_design_rel_fro`=1.4142=√2 — squeeze matches raw fla, differs from design by the pure-transpose signature); one-line transpose fix, independently audited (fresh Opus, no session context) as minimal/correctly-scoped/no other call site affected; POST-FIX the identical adjudication shows the numbers swap exactly as predicted (`S_squeeze_vs_closed_form_design_rel_fro`=0.0, `S_squeeze_vs_closed_form_fla_rel_fro`=1.4142 — squeeze now matches DESIGN, differs from raw fla), and the positive control recovers 1.0000 at every h (cos_mean≈0.99999), deliberately-transposed arm now reads 0.0000 (role-swap as predicted) | REASONING_LINK_DESIGN.md §17.1–§17.3 (2026-07-11); commits `8666aee` (pre-fix FAIL), `fb9950c` (fix + audit + PASS) | pre-fix poscontrol: `experiment-runs/2026-07-11_reasoning_null_poscontrol/results/reasoning_link_poscontrol_result.json` md5 `71e53be7953812a60004a5ee08d77e10` (manifest: `experiment-runs/2026-07-11_reasoning_null_poscontrol/MANIFEST.md5`); PRE-fix closed-form adjudication numbers: `experiment-runs/2026-07-11_reasoning_link_remetric/01_analytic_adjudication/logs/layout_adjudication_run_PREFIX.log` md5 `3e23fd60c32afef993ddf2db587a0e3f` (no PREFIX JSON was archived, only this log — disclosed as a minor provenance gap); POST-fix closed-form adjudication: `experiment-runs/2026-07-11_reasoning_link_remetric/01_analytic_adjudication/results/reasoning_link_layout_adjudication_result_POSTFIX.json` md5 `1daa4c80b706b610953fe45a38cebd23`; poscontrol re-run (post-fix): `experiment-runs/2026-07-11_reasoning_link_remetric/03_positive_control_rerun/results/reasoning_link_poscontrol_result_POSTFIX.json` md5 `e5abae881fa9bbcd3605bafecad3a745` | text §3 |
| C11-remetric | The wave-1/wave-2 grid (80 cells, 320 (cell,h) readings, identical archived checkpoints + deterministic episode seeds) re-run under the fixed instrument does NOT reproduce the pre-fix zero: 78/320 nonzero (30/60 Leg-A cells, 8/20 Leg-B cells carry ≥1 nonzero h); h=1 recovered_frac reaches 0.8691 at its strongest cell (per_token×wikitext×s1×K32), h≥2 reaches 0.6375 in one cell; premise (iii)/(iv) still fail 0/320 both, so no confirm is licensed by the registered routing regardless | REASONING_LINK_DESIGN.md §17.4 (2026-07-11); commit `fb9950c` | `experiment-runs/2026-07-11_reasoning_link_remetric/04_remetric/results/AGGREGATE_SUMMARY.txt` md5 `34186cbc6771bd4c8c631816ba1d90a5` (+ 80 per-cell JSONs in the same directory) | Table 1 |
| C12-nullvalidation | Two independent, pre-registered correspondence nulls reproduce the C11 fixed-lens signal at every one of the 320 readings: (a) label-shuffle null, 0/320 NULL-CLEARS (0/38 signal-bearing cells), mean real 0.3023 vs mean null 0.3010 at the 41 floor-clearing readings; found VACUOUS at h=1 by construction (`prev_slot` at h=1 never consults the shuffled permutation — real==null identically, e.g. 0.8691/0.8691 at the strongest cell; a healthy-draw check confirms the null cycle itself agrees with the true one on only 4.1% of entries, ruling out a degenerate shuffle); (b) positional derangement null (registered fallback, teeth at every h including h=1), 0/320 NULL-CLEARS, mean real 0.3023 vs mean deranged 0.2960; strongest cell 0.8691 real vs 0.8125 deranged (94% reproduction against the 50%-bar). Mechanism (descriptive, design-doc record only — no separate raw artifact locally archived, since the original harvest saved no raw state/Z-dumps to recompute from): cross-checkpoint prediction convergence 0.9996, value population mean pairwise \|cos\| 0.9648. Episode-resampling check: 20/21 floor-ON cells STABLE (35/38 at the raw-nonzero denominator) — the bimodality is a state property, not episode luck | REASONING_LINK_DESIGN.md §17.6–§17.7 (2026-07-11); commits `c3de797` (pre-registration), `895e0f1` (derangement addendum + mechanism numbers), `9744752`/`b3c5c4e` (verdict of record) | succ-shuffle: `experiment-runs/2026-07-11_reasoning_link_validation/01_item12_shuffle_resample/results/ANALYSIS_SUMMARY.txt` md5 `922bfc8b34d622ccdfe88236ac69731d`; derangement: `experiment-runs/2026-07-11_reasoning_link_validation/02_derange_control/results/DERANGE_SUMMARY.txt` md5 `2edc16daa9c604ae978e6e2d3e853da1`; mechanism covariates (0.9996/0.9648): **no raw JSON/log locally archived** — an on-box ad hoc single-checkpoint diagnostic, recorded only in REASONING_LINK_DESIGN.md §17.6a prose (commit `895e0f1`); disclosed here as a prose-only provenance gap, not backfilled | text §3; Table 1 |

Numbers in the draft cite rows via `<!-- evidence: <claim-id> -->` comments.
C10–C12 are new rows added by the 2026-07-11 claim-shape-correction revision
(§17); C1–C4 gained inline claim-shape/limitation notes above but keep
their original raw artifacts, which remain accurate records of what the
PRE-FIX instrument read.

## Figures to generate

Single versioned script `figures/figure-gen.py` (md5-asserts every source
file against the manifest above; loads only archived raws, no hand-entered
numbers).

- `fig1_validity_gates.pdf` — premise (iii) and (iv) medians vs their own
  null p95 across every cell of all three instruments (Phase 1's 78 cells,
  rung 3, Phase-1b's 4 cells, Phase-2's 6 terminal readings), one point per
  cell per premise, colored by instrument; the y=x line is the pass
  boundary. Takeaway: no point crosses the boundary anywhere, at any scale,
  under any surface form or training regime — the readout's validity
  precondition never holds, so the geometric null is an instrument verdict.
- `fig2_dissociation.pdf` — Phase-2 familiarization: per-cell `L_query`
  trajectories (6 cells, steps 250→5000, falling 21.8–46.4%) alongside the
  geometric gate's recovered fraction at the same five checkpoints (flat at
  exactly 0.0, 30/30). Takeaway: the model measurably learns the task in
  vocabulary space while the state-space geometric readout stays at floor —
  the two readouts dissociate.
- `fig3_transient_replication.pdf` — left: per-checkpoint Δ (off − arm) with
  n=3 CIs for wikitext×per_token at K=32, showing the single CI excluding
  zero at c=2500 (harm direction) and re-widening by c=5000; right: the
  anchor cell's 12 per-seed deltas, archived n=3 cohort vs new n=9 cohort,
  with each cohort's own CI and the (non-decision-grade) naive pool CI.
  Takeaway: the transient is real at n=3, and the pre-registered
  batch-effect gate stops the pooled n=12 read; the new cohort alone spans
  zero.

## Nearest prior work (distinguish by name)

- **Based (Arora et al., 2024, arXiv:2402.18668):** establishes the
  recall–state-size tradeoff and the param-matched ladder pattern; this
  paper's object is a geometric observable's downstream validity and its
  claim type is a bounded null, not a capacity law.
- **Zoology (Arora et al., 2023, arXiv:2312.04927):** diagnoses associative-
  recall gaps in fixed-state architectures trained for the probe setting;
  this paper probes general-purpose checkpoints post hoc and reports on the
  instrument battery, not an architecture gap.
- **Okpekpe & Orvieto (2025, arXiv:2508.19029):** a positive causal account
  of Transformer-vs-SSM recall differences; this paper bounds one named
  mechanism (write-geometry stabilization) and makes no positive mechanism
  claim.
- **Nichani, Lee & Bietti (ICLR 2025, arXiv:2412.06538):** associative-
  memory capacity theory whose argmax-decoding caveat shaped this program's
  exact-recovery readout choice; cited as readout-design provenance.

## Anonymization surface (double-blind)

Author/handle/org tokens for the grep (fill the styleguide's placeholders):
`Larson`, `samuellarson`, `samlarson`, `saml212`, `pebble`, `pebbleml`,
`learned-representations`, `youthful-indigo-turkey`, `Brev`, `anthropic`,
`Claude`. Plus the always-on patterns: `github.com/`, `huggingface.co/`,
`acknowledg`, `self-funded`, `funded by`. No absolute local paths (`/Users/`,
`/Volumes/`, `/root/`) anywhere in the PDF or supplementary.

## Dual output

- [x] Venue submission (anonymized LaTeX, 4pp COLM 2026 template)
- [ ] Public write-up — not requested for this paper.
