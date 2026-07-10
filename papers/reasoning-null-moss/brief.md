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

Falsifiers, per bound: (1) any nonzero recovered fraction or a premise-gate
pass anywhere; (2) a determinate arm contrast exceeding the floor; (3) a
clean pooled n=12 CI excluding zero. None occurred.

## Contribution bullets

1. **A triply-instrumented geometric null.** Three structurally different
   deployments of a state-space composition readout (zero-shot marker
   template, zero-shot natural language, task-familiarized) return a
   recovered fraction of exactly 0.0 at 366/366 readings on checkpoints from
   14M to 1.31B parameters, with the readout's own pre-registered validity
   gates failing everywhere — routing the outcome to probe-invalid, never to
   an unlicensed refutation.
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
| 1 Introduction | 0.70 | the keystone question, why a bounded null is the honest deliverable, contributions |
| 2 Setting and instruments | 0.65 | models/ladder, task grammar, the three readouts, pre-registration + gate discipline |
| 3 Bound 1: the geometric readout never fires | 0.90 | 366/366 zeros, premise gates, scale series, the gate-enforcement arc |
| 4 Bound 2: the behavioral contrast is power-bounded | 0.60 | 3/4 UNRESOLVED at the registered floor; the transient, sign-disciplined |
| 5 Bound 3: the replication gate | 0.60 | n=12 wave, batch-effect flag, cohort CIs, what stands |
| 6 Related work | 0.20 | Based / Zoology / Okpekpe–Orvieto / Nichani et al., by name |
| 7 What is and is not bounded | 0.35 | scope, alternative explanations, lessons for small-scale practice, conclusion |
| **Total** | **4.00** | |

Abstract: 200–230 words (styleguide band), on page 1 inside the budget.

## Claims-to-evidence-to-figure map

Archive root: `experiment-runs/` (repo-committed copies; SSD mirror is the
superset archive). "Manifest-md5" = md5 of the sorted `md5  path` line list
over the named file set (computed 2026-07-10; the figure script re-asserts
per-file md5s at build time).

| Claim id | Claim (with the number) | Verdict record (§ / log entry) | Raw artifact (path + md5) | Figure / table |
|---|---|---|---|---|
| C0-ladder | Four-rung scale ladder 14M/98M/392M/1.31B (d_state 64/64/128/128; n_layers 2/12/16/22; d_model up to 2560), two corpora (openr1-mix-ext, wikitext-mix-ext) | REASONING_LINK_DESIGN.md §6.1; §16.11 (rung-3 config read from ckpt_config) | `ckpt_config` fields inside the C1/C2 raw JSONs (same files/md5s as those rows) | Table 1 |
| C1-phase1 | Zero-shot marker-template probe: recovered_frac@0.9 exactly 0.0 at 312/312 (cell,h) readings (78 cells × h∈{1..4}); premise (iii) passes 0/78, premise (iv) 0/78; premise-iii medians span [−0.33, 0.76] and never cross their null p95 | REASONING_LINK_DESIGN.md §15.1–§15.4; EXPERIMENT_LOG.md 2026-07-07 "REASONING-LINK PHASE 1 HARVEST" | `experiment-runs/2026-07-07_reasoning_link_phase1/results/leg_*.json` (78 files, manifest-md5 `3c885fbe49ada8cb62afb33f3f6faf60`) | Fig 1; Table 1 |
| C2-rung3 | Rung-3 (1.31B) completes the scale series: 8/8 readings 0.0; both premise gates fail at both corpora; Leg-B ladder total 80/80 zeros across 14M→1.31B | REASONING_LINK_DESIGN.md §16.11; EXPERIMENT_LOG.md 2026-07-07 rung-3 entry | `experiment-runs/2026-07-07_reasoning_link_rung3/results/leg_b_rung3_openr1-mix-ext_s0_k64.json` md5 `b75745af6ae31632af61b077dfdd539b`; `...wikitext-mix-ext_s0_k64.json` md5 `13ecd7c6b90cd0696771bcae474f0820` | Fig 1; Table 1 |
| C3-phase1b | Natural-language templates (2 candidates × 2 corpora): 16/16 readings 0.0, `gate_result_h1_probe_valid` False at 4/4 cells; the enforcement script refused the full 78-cell grid | REASONING_LINK_DESIGN.md §16.8.1–§16.8.4; EXPERIMENT_LOG.md 2026-07-07 Phase-1b entry | `experiment-runs/2026-07-07_phase1b_gate/results/stage0_natural_*.json` (4 files, manifest-md5 `f3ec8b2a89f19645f84bcaa385b4efd4`); sentinel `results/STAGE0_GATE_REFUSED` | Fig 1; Table 1 |
| C4-dissoc | Task-familiarized: gate fails 30/30 (cell, checkpoint) readings (recovered_frac(h1)=0.0, 0/512 queries each) while the vocab-space `L_query` falls 21.8–46.4% (mean 35.9%) in 6/6 cells — a vocabulary/geometry dissociation | REASONING_LINK_DESIGN.md §16.15.1–§16.15.2 | gate: `experiment-runs/2026-07-08_phase2_familiarization/ckpts/stage05_gate_*.json` (30 files, manifest-md5 `85d381404dd89769eda382beacbee673`); trajectories: `results/off_*.json` (6 files, manifest-md5 `091c33c4786048cd3f94bb75496465ec`) | Fig 2; Table 1 |
| C5-power | Behavioral contrast at n=3: 3/4 (corpus × arm) contrasts UNRESOLVED; pre-registered detectable floor ≈1.5–1.7 loss units (σ proxy 0.43–0.48, t(2,.975)=4.303), the same order as the OFF arm's whole familiarization effect ≈1.69 | REASONING_LINK_DESIGN.md §16.16.4 (registered pre-harvest); §16.18.6 | `experiment-runs/2026-07-08_phase2b/results/trajectory_wikitext-mix-ext_phase2b.json` md5 `5c727ac669aea02601790b4fb1dac8b4`; `trajectory_openr1-mix-ext_phase2b.json` md5 `bdbc2352172b79b206d9d7bf7be303fe`; `PHASE2B_SUMMARY.json` md5 `9f5b14f372187ab181c1981158a23f14` | text §4 + Fig 3 |
| C6-transient | The single determinate signal: wikitext×per_token classifies TRANSIENT — Δ(K=32, c=2500) = −0.500, CI [−0.624, −0.376] (harm direction: arm loss above control), dissolving by c=5000 (Δ=−0.795, CI spans zero); every determinate K=32 reading in the 20-checkpoint primary table is negative; the held-out-hop secondary readout fires at the same cell/checkpoint, same direction | REASONING_LINK_DESIGN.md §16.18.3–§16.18.6 | `trajectory_wikitext-mix-ext_phase2b.json` (md5 above; `per_arm.per_token.raw` + `holds_by_c`) | Fig 3 (left) |
| C7-replication | n=12 seed extension: batch-effect gate FLAGGED at the anchor (var_ratio 4.47 > 4.0 pinned; sd_old 1.154 vs sd_new 0.546); old-cohort CI [−0.624, −0.376] (the archived n=3), new-cohort (n=9) CI [−0.506, +0.357] spans zero; naive n=12 pool CI [−0.509, +0.147] spans zero (diagnostic only, non-decision-grade per the registered routing); outcome BATCH-EFFECT-FLAGGED, bucket null; across the full tables 7/10 primary and 7/10 held-out (checkpoint, K) pairs are computable and every computable pooled CI contains zero | REASONING_LINK_DESIGN.md §16.19.8 (pre-registered MECE partition + gate), §16.20.2/§16.20.5 | `experiment-runs/2026-07-08_phase2b_seedext/results/trajectory_seedext_wikitext_n12.json` md5 `989f9997c50973cc299f25d89efff64f`; `PHASE2B_SEEDEXT_SUMMARY.json` md5 `fe48bb35533c030ce5f352ad607e244e` | Fig 3 (right) |
| C8-teeth | Gate-enforcement arc: Phase 1's chain computed its Stage-0 abort gate but had no enforcing code path (≈0.29 GPU-h ran past a probe already flagged invalid); Phase-1b/Phase-2/Phase-2b chains enforce their gates at the process boundary and refused launches on real failing results | REASONING_LINK_DESIGN.md §15.2, §16.8.4, §16.15 headline; EXPERIMENT_LOG.md LEARN entry 2026-07-07 | `experiment-runs/2026-07-07_reasoning_link_phase1/reasoning_link_chain.sh` (grep: no gate reference); `experiment-runs/2026-07-07_phase1b_gate/results/STAGE0_GATE_REFUSED` + `logs/phase1b_run1.log` | prose (§3) |
| C9-compute | The complete program consumed ≈9.5 GPU-hours on at most two GPUs at a time (per-wave: 0.373 / 0.024 / 0.017 / 0.617 / 1.661 / 6.819) | Realized-cost records: REASONING_LINK_DESIGN.md §15.7, §16.8.5, §16.11, §16.15.6, §16.18.7, §16.20.8 | machine-readable: `PHASE2B_SUMMARY.json` (`elapsed_s=2833, n_gpus=2`), `PHASE2B_SEEDEXT_SUMMARY.json` (`elapsed_s=4119, n_gpus=2`), `rung1_seedext_*_summary.json` (Σ wall_s); log-timestamp-derived for the small waves (archived logs in the C1/C3 archives) | Appendix B (reproducibility compute disclosure) |

Numbers in the draft cite rows via `<!-- evidence: <claim-id> -->` comments.

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
