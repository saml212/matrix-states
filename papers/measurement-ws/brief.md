# Paper brief — measurement-ws ("the instrument is the first suspect")

Stage 1 of the `paper` skill (repo mode). This paper is the
instrument-methodology story: how many apparent model failures in one
measurement-heavy empirical program were broken lenses, and the
adjudication discipline that separated them from the genuine model
failures. Base material: the compiled draft at
`matrix-thinking/submissions/measurement-2026/` (the tolerance-
miscalibration case study, "The Cliff That Wasn't"), which this paper
subsumes as its Case I and extends with five further incidents — most
importantly the two freshest: the wrong-layer instrument arc
(`HEAD_TO_HEAD_DEMO_DESIGN.md` §1.27–§1.30) and the primary-vs-crosscheck
broken-lens tiebreak (`CAPABILITY_SEPARATION_DESIGN.md`
§2.31/§2.31a/§2.32). Scope boundary inherited from the base draft: this
paper claims the measurement failure modes and the process, never the
substantive capacity/capability results themselves (those belong to the
companion papers and are cited, not re-claimed).

## Venue

- **Name:** NeurIPS 2026 measurement/evaluation/science-of-DL workshop —
  **PENDING the Jul-11 accepted-workshop list**; backup MOSS @ COLM 2026
  late window. See `papers/measurement-ws/venue-requirements.md`.
- **Format:** 4 pages main text excl. references/appendix (working
  assumption, strictest across the candidate set); NeurIPS 2025 kit as
  the flagged `UNVERIFIED — cache fallback` stand-in
  (`venue-requirements.md`), single column.
- **Requirements source:** `papers/measurement-ws/venue-requirements.md`
  (live-fetched 2026-07-10; workshop list 404, dates page live).
- **Review style:** double-blind assumed → anonymization grep gates the
  submission build.
- **Archival:** non-archival assumed; re-verify on the chosen CFP.
- **Deadline:** Aug 29 '26 AoE (suggested workshop-paper deadline, live).

## Thesis (one falsifiable sentence)

> In one measurement-heavy empirical research program, six consecutive
> pre-publication incidents in which a trained model appeared to fail
> traced, under a fixed adjudication discipline, to six mechanistically
> distinct broken instruments rather than broken models — and the same
> discipline is falsifiable rather than result-laundering, because its
> decisive crosscheck survived a pre-registered shuffled-target falsifier
> that would have voided the adjudication, and because it left the
> pre-registered endpoint verdict unflipped and two lens-independent
> genuine model failures standing.

The falsifier is concrete: if the shuffled-target control had read
recovered-fraction ≥ 0.5 on any converged checkpoint, the tiebreak was
pre-registered to be WRONG, voiding the adjudication and escalating to a
full instrument rebuild. It read 0.00/0.00/0.05.

## Contribution bullets

1. **A six-incident broken-lens catalogue**, each incident caught before
   publication, each with a distinct mechanism and a reusable diagnostic
   signature: a numerical tolerance calibrated on the wrong key
   population (near-miss-vs-wall resolution under an iteration sweep); a
   probe tap on a causally inert layer (task-metric/probe-metric
   dissociation, then causal state-zeroing localization); a gauge
   assumption measured on one architecture and silently carried to
   another (a 0-vs-1.0 primary/crosscheck contradiction concentrated
   exactly on converged cells); a degenerate uncentered-covariance lens
   (a synthetic perfect model failing production bars); a
   target-construction rank tax (rank-capped cells training to their
   exact theoretical ceiling, √(k/d_state)); and a transposed reference
   convention (a rel-Frobenius ≈ √2 signature resolved by a closed-form
   hand computation).
2. **A reusable adjudication discipline, stated as five rules:**
   instrument-health adjudication before any model verdict;
   pre-registered crosschecks graded where the primary lens's
   assumptions fail; falsifier teeth — negative and shuffled controls
   run (and mutated) to completion, with positive controls on every
   probe rung; raw-artifact tiebreaks when two reads contradict; and
   exact-recovery closure of argmax shortcuts wherever a claim depends
   on continuous recovery.
3. **The honest boundary — adjudication is not result-laundering:** under
   the corrected lens the pre-registered endpoint still read FALSIFY
   (for a different, more informative reason), and two genuine model
   failures (a config that never converges under either lens; a
   trainability-outlier seed) survived every instrument fix.

Nearest-work distinctions are in "Nearest prior work" below.

## Per-section page budget

| Section | Pages | Purpose |
|---|---|---|
| 1 Introduction | 0.60 | the question; the program; the six incidents; contributions |
| 2 The adjudication discipline | 0.50 | the five rules, each with its incident anchor |
| 3 Case I: the tolerance that manufactured a cliff | 0.55 | admission gate; wrong first diagnosis; repro instrument; unlock |
| 4 Case II: the instrument at the wrong layer | 0.65 | dissociation; refit refutation; causal localization; nonlinear storage |
| 5 Case III: the gauge that did not transfer | 0.80 | 0-vs-1 contradiction; four-ground tiebreak; teeth; re-metric |
| 6 Three more lenses, briefly | 0.35 | uncentered covariance; ambient rank tax; transposed convention |
| 7 What the discipline did not flip | 0.20 | FALSIFY stands; the two genuine failures |
| 8 Related work and limitations | 0.35 | by-name distinctions; scope |
| **Total** | **4.00** | = assumed venue limit (flagged) |

Appendix (not counted): the six-incident catalogue table + per-incident
evidence tables (the tap-variant table, the decision-rule walk, the
re-metric table, the teeth table; the 62-cell grid appears as Fig. 1,
not as a table). Gauntlet r1 FIX-4 disposition: the catalogue-into-body
move was attempted and measured at ~0.5pp over the 4pp working limit
even after the three named funding cuts, so the FIX-4 HARD FALLBACK was
exercised (table stays in the appendix; a source comment in
01_intro.tex records the reversal condition — if the Jul-11 venue
confirms ≥5pp main text, move `tab:catalogue` back into the body).

## Claims-to-evidence-to-figure map

Every row: the pre-registered verdict record (design-doc § / log entry,
fixed before this paper), the raw artifact (path + md5), and where it
shows. A claim with no row does not go in the paper.

### Case I — tolerance miscalibration (base draft, compressed)

| Claim id | Claim (with the number) | Verdict record | Raw artifact (path + md5) | Figure / table |
|---|---|---|---|---|
| I1 | d=96 wide grid: of 12 new d=96 cells only 1 admissible (K=78/84/90 all 0/3); wave-wide new-cell admissibility 5/16 incl. 4/4 clean d80-escalation cells; h1 training guard 1.0 everywhere | `KEY_ANCHORING_SCALING_DRAFT.md` §15.22 (per-cell table) | `experiment-runs/2026-07-07_keyanchor_scaling_wide/results/deltanet_rd_exactness/wavekeyanchor-scaling-wide/` (16 JSONs on disk: 12 new per §15.22's cell list — K∈{72,78,84,90}×3 — plus 3 reused K69 and one contingency seed 1733; admissibility tallied 1/12 over the 12 new by direct re-read) | §3 text |
| I2 | anchor-table NS residuals over 7,000× below the 0.01 tolerance (max 1.416e-06 → 7,060×; mean 1.195e-06 → 8,366×; the earlier "~7,800×" was loosely coupled to either statistic — re-paired to the conservative max per format-audit M2); failure 100% C17-pool-exclusive; first mechanistic claim retracted | §15.23 (MISDIAGNOSED-ARTIFACT) | `experiment-runs/2026-07-08_ns_admission_diag/results/diag_ns_admission_result.json` md5:d0d940d0afcdf30af8f45497168fb297 | §3 text |
| I3 | repro instrument: 107/107 reconstruction; 0 pool-membership violations over 36 events (12× floor); 0 live/offline disagreements; all 4,608 flagged episodes resolve by n_iter≤28 (4,313@24, 295@28) → TOLERANCE-MISCALIBRATION | §15.24 (4 audit rounds) + §15.25.1 (decision-rule walk) | `experiment-runs/2026-07-08_c17_repro/results/keyanchor_scaling_c17repro/diag_c17_repro_analysis_K84_s1940.json` md5:6f24447c9c063e19029b1bd3dd142266 (verdict + per-episode n_iter histogram re-tallied this session) | §3 text + Table 1 row |
| I4 | flag-only unlock at zero GPU; re-fit per-K mean h4 = 0.9592/0.9216/0.9326/0.9581/1.0000 (K=69/72/78/84/90) — no cliff through K/d=0.9375 | §15.25.4–.5 (THE UNLOCK) | `experiment-runs/2026-07-08_c17_repro/fits/fit_d96_unlocked_results.json` md5:61eaffe1744a56086af2f4115f9a9cf4 (curve_points block) | §3 text |
| I5 | frontier moves with load: K=90 stays fallback-inadmissible at n_iter=28; leg swap at K=84 (value-salvage 0.09307<0.10 with clean convergence); archived K=90 exact-1.0 ceiling read 0.9725 at a fresh seed; pool-restriction shift +0.0330 ≈ 12× the 0.0028 noise floor, correctly withheld (DEGENERATE_CELL) | §15.27 (3 escalated findings) | `matrix-thinking/deltanet_rd/results/keyanchor_poolmargin/poolmargin_verdict.json` md5:88adc0db4f4b6ad7639c671772c096ce (k84_standard 0.89388 vs restricted 0.92692; noise repeats 0.89609/0.89665) + `...poolmargin/wkeyanchor-scaling_rdx_K90_armd_s2043_geo3n28_anchor_learned_dprobe_rev7_d96.json` md5:7429579769803f0d61eb5c9352c3718e (rec@0.9 = 0.97253) | §3 + §7 text |
| I6 | economics (scoped to Case I, the one incident costed): repro instrument launch chain 0.50 GPU-h + pool-margin diagnostic 1.10 GPU-h, against 6.33 GPU-h of training protected. The previously quoted 0.82 (= 0.487 chain + 0.33 pre-launch smoke) is dropped from the paper: the 0.33 verification component has no named raw; the paper now rests on raw-sourced wall-clock only | §15.25.3 / §15.27 / §15.22 (realized-cost tables) | chain: `experiment-runs/2026-07-08_c17_repro/logs/box_verification_20260708.json` md5:df97dd392f30d6626b33d7e0eccecfec (chain_span_s 1782.3 → 0.4951 GPU-h); pool-margin: `matrix-thinking/deltanet_rd/results/keyanchor_poolmargin/poolmargin_run1.log` md5:3d19ac3d472a2bbe935e90658df3f046 (wrapper_wall_s 2608.7 + 1343.5 = 3952.2 s → 1.098 GPU-h); training: the 16 cell JSONs' wall_s fields in the I1 directory (sum 22,799.05 s → 6.3331 GPU-h, per §15.22's per-cell table) | §8 text |

### Case II — the wrong-layer instrument (h2h §1.27–§1.30)

| Claim id | Claim (with the number) | Verdict record | Raw artifact (path + md5) | Figure / table |
|---|---|---|---|---|
| W1 | round-3 dissociation: discrete rung-1 recall 0.9990 (chance 0.03125; ablation 0.0447; transformer 0.0295) while the continuous probe (rf@0.9) reads exactly 0.0 in all arms — three "failed" rounds carried this same plateau | `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.27 (round-3 verdict, recorded as FAIL + dissociation) | `experiment-runs/2026-07-09_h2h_tap_localization/results/tap_localization_SUMMARY.json` md5:0e73ee283b7208db10af5588fe1c6713 (`both_intact` 0.9990/0.0447 reproduces the §1.27 numbers bit-identically); transformer recall 0.0295: `experiment-runs/2026-07-09_h2h_calib3/results/h2h_calib_transformer_task1_calib_primary_K32_auxrev2.json` md5:a0a71816e4cb4f0086c6b9e9515fa227 (`final_rung1_accuracy` = 0.029541); rf@0.9=0.0: `experiment-runs/2026-07-09_h2h_decisive_probe/results/diagnosis_{contender,ablation,transformer}.json` md5:cdd3ea2ee2ac6136b1e70297f09f0d4c / e48b002e25365249ef863358853f55e7 / c0db74a59f5b234ba31c78a0e8d309f2 | §4 text |
| W2 | refit refutation: closed-form ridge (λ=100 fixed; optimum of the penalized objective — the over-regularization story is killed by the same rig reading 0.674 at tap iv, not by an optimality claim), pinned SGD, and MLP probes ALL reproduce rf@0.9=0.0; ridge barely clears its shuffled control (gap +0.059/+0.005/+0.021 cos); probe output ≈ the membership oracle (0.176 ≈ 99.5% of the 1/√32 ceiling; pred-vs-membership cos 0.896) | §1.28 (diagnosis) + §1.29 (decision rule 1 REFUTED) | `diagnosis_contender.json` (md5 above: ridge/SGD/MLP rf@0.9 blocks, probe_cos_mean 0.17596) + `oracles.json` md5:68ffeed919a8a0cd327bbb02f38d61cf (o1 0.187, o2 0.896/0.606/0.679) | §4 text |
| W3 | the answer exists in the network: LM-head route (full forward) episode-restricted top-1 = 0.9957 (31.9× chance) for the contender vs 0.0304 transformer / 0.0341 ablation — the §1.27 arm ordering reproduced end-to-end | §1.29 (decision rule 3 fires: tap placement) | `diagnosis_contender.json` (lm_head_route/q_pos/episode = 0.99573) + `diagnosis_transformer.json`, `diagnosis_ablation.json` (md5s above) | §4 text |
| W4 | causal localization: zeroing S0 collapses recall to chance (0.0286); zeroing S1 — the layer every instrument tapped — leaves it bit-unchanged (0.9990); no linear tap at EITHER state clears rf@0.9 (gaps +0.002–0.063); only the post-block-1 pre-LM-head hidden decodes linearly (rf@0.9=0.674, gap +0.800), and only for the contender (ablation iv: 0.0, gap +0.006) | §1.30 (Tables 1–2) | `tap_localization_SUMMARY.json` md5:0e73ee283b7208db10af5588fe1c6713 | Fig. 2 + §4 text |
| W5 | second instrument defect found in passing: the rung-2 classifier's tgt_slot labels are uniform given identity (slots drawn fresh per episode) — a PERFECT tap scores chance; corrective rule: every probe rung needs a planted-signal positive control | §1.28 (item 4, NEW INSTRUMENT DEFECT) | code facts: `matrix-thinking/deltanet_rd/h2h_cell_train_rd.py:686-727` (tgt_slot labels), `matrix-thinking/deltanet_rd/grammar_rd.py:434-436` (fresh entity order per episode) — cited as code, no number claimed | §4 text |

### Case III — the gauge that did not transfer (§2.31/§2.31a/§2.32)

| Claim id | Claim (with the number) | Verdict record | Raw artifact (path + md5) | Figure / table |
|---|---|---|---|---|
| X1 | 62/62-cell sweep clean; mechanical PRIMARY M-D3 endpoint = FALSIFY; primary multi-seed ceilings S3 0.51 / S4 0.29 / A5 0.09 / S5 0.10 / A6 0.02 | `CAPABILITY_SEPARATION_DESIGN.md` §2.31 (items 1–2) | `experiment-runs/2026-07-10_stage2_calibration/sweep_results/stage2_harvest_report.json` md5:7dddd19b3849fd907d0f99edbb31c540 (verdict block re-read this session: FALSIFY, verbatim reason) | §5 text |
| X2 | the broken-lens signature: primary and crosscheck agree (≈0 both) on every non-converged cell and contradict at 0-vs-1.0 exactly on converged cells (e.g. A6 nh4 seed0: final_loss 0.0001, primary rf90@64 0.050, crosscheck 1.000); all 25 Arm-2 baseline cells (5 groups × 5 seeds, 50 checkpoint values): crosscheck == primary exactly, zero divergence; threshold robustness: every cell with disagreement >0.3 is converged (16/16), but the COUNT is not cut-invariant (16/14/13/12/11 at cuts 0.3/0.4/0.5/0.6/0.7). NOTE: the design-doc §2.32 prose says "30 baseline cells" — internally inconsistent (its own 5×5=25); the raw is decisive at 25 (gauntlet r1 A1) | §2.31 (item 3, 6-cell table) + §2.32 (item 4, full-grid generalization) | `experiment-runs/2026-07-10_stage2_calibration/remetric_2p32/crosscheck_lens_verdict_output.json` md5:f26a769d5c263af224c91d39bd83710b (flat_per_cell_table, 62 rows = 37 arm3 + 25 arm2, recounted vs raw) + `experiment-runs/2026-07-10_stage2_calibration/instrument_health_adjudication.log` md5:fb58e8422e9bde2a2c8997cf314fa3b9 | **Fig. 1** |
| X3 | tiebreak on four grounds: crosscheck fit/eval-split leakage-guarded by construction; crosscheck discriminates (reads 0 on junk); the primary's Q̂≈I gauge was measured on Stage-1 DeltaNet states and does not transfer to the Householder-expanded composer (its basis-brittleness pre-registered: S4 per-seed primary mean_cos 0.03–0.69 vs crosscheck 0.86–0.95); oracle-injection precedent grades crosscheck where Q_true≠I | §2.31a (VERDICT: primary-degauge defect; mechanical FALSIFY VOID as model verdict) + §1.33 metric-health disclosure | grounds 1/4: `matrix-thinking/capability_separation/readout.py` (degauge_and_score fit/eval split; cited as code); ground 2: X2's artifacts; ground 3 (per-seed basis-brittleness numbers): `experiment-runs/2026-07-09_capability_sweep_harvest/results/S4__unconstrained__seed{0..4}.json` (seed0 md5:1f30c88318ba1c501d74265f47bf3113; primary mean_cos 0.478/0.694/0.330/0.028/0.112 → range 0.03–0.69, crosscheck_mean_cos 0.935/0.909/0.952/0.864/0.866 → range 0.86–0.95, re-read this session; the aggregate `harvest_summary.json` md5:7dce77dcba724cd1004419ac71fe5f2f does not carry the per-seed values — repointed per re-attack B3) | §5 text |
| X4 | teeth pass 3/3: shuffled-target crosscheck reads 0.00/0.00/0.05 (all ≪ the 0.5 void-bar) on three converged checkpoints while the real reads (1.00/0.80/1.00) reproduce the committed values bit-identically | §2.31a (teeth pinned) + §2.32 (item 2, PASS) | `crosscheck_lens_verdict_output.json` md5:f26a769d5c263af224c91d39bd83710b (teeth_control block: all_pass=true) | §5 text + Table 1 row |
| X5 | the re-metric verdict: crosscheck-lens M-D3 STILL FALSIFY (m_d3_verdict unmodified, shadow cells; harness first reproduces the committed primary verdict exactly; recomputed D=8 primary bit-identical <1e-9 in 62/62) — but for a different reason: A6's decisive n_h=2 config never converges under either lens (far64 0.00 all 5 seeds), and S5's decisive triad is dragged below its 0.735 bar (0.483) by the pre-classified seed-1 outlier while seeds 0/2 clear 80%/65% of their own ceiling vs an Arm-2 baseline at exactly 0.0 | §2.32 (items 3, 5, 6) | `crosscheck_lens_verdict_output.json` (verdict_crosscheck_lens per_group + verdict_primary_reproduction) + `remetric_2p32_box_output.json` (62-cell ceiling recompute), both in `experiment-runs/2026-07-10_stage2_calibration/remetric_2p32/`; md5 above for the verdict file | §5 + §7 text |

### The three brief lenses (§6)

| Claim id | Claim (with the number) | Verdict record | Raw artifact (path + md5) | Figure / table |
|---|---|---|---|---|
| B1 | uncentered-covariance degeneracy: for near-orthogonal targets ZZᵀ≈c²I is isotropic — on the production harness, an honest synthetic injection scored mean_cos 0.084 / rec90 0.05 under the OLD uncentered lens vs 0.9996 / 1.00 under the centered+crosscheck lens; the same file shows scale-only primary 0.046 vs full-Q crosscheck 0.9996 when Q_true≠I | `CAPABILITY_SEPARATION_DESIGN.md` §1.25 (DEFECT 1) + §1.26 (fix map); re-demonstrated at the §1.32-era calibration re-check | `experiment-runs/2026-07-09_capability_calib_recheck/gate1b_recheck.txt` md5:2d170cc03011cc56105adeae9929e481 (uncentered negative control + honest/centered rows quoted verbatim) | §6 text + Table 1 row |
| B2 | ambient-identity rank tax (D-AMB): the target was built as ρ ⊕ I₂ (rank d_min+2, all singular values 1), so every rank-capped arm bought the constant block first; 37/39 force-rank cells sit within 0.07 of the exact rank-k ceiling √(k/d_state) (mean abs dev 0.028) — the arms trained to their theoretical optimum against the WRONG target; the causal test was undelivered-by-instrument, not failed | §1.33 (D-AMB, P1–P5) | `experiment-runs/2026-07-09_capability_sweep_harvest/harvest_summary.json` md5:7dce77dcba724cd1004419ac71fe5f2f + `harvest_analysis_output.txt` md5:854a4bd7c46e626badcc0fbf05d0e07a | §6 text + Table 1 row |
| B3 | transposed reference convention: at D=1 the recurrence is closed-form (S₁=βvkᵀ); the composer matches the hand-computed form at rel-Fro 4.504e-08 and the TRANSPOSE at 1.405 ≈ √2 — the cross-check's fla reference was invoked on the wrong side (fla returns S=…+βkvᵀ); post-fix the cross-check passes 3/3 (2.80e-3/3.87e-3/4.52e-3) and a transposed-update mutant composer is KILLED at 1.405 | `CAPABILITY_SEPARATION_DESIGN.md` §2.26 (composer EXONERATED) | `experiment-runs/2026-07-10_stage2_calibration/analytic_step1_cpu_composer_vs_closed_form.log` md5:2b8fb480f7dabc77a5bc9207f611442a + `analytic_step2_box_fla_vs_closed_form.log` md5:29ef4e6300f9ec62a087e11af6d5f3e0 + `analytic_check_negative_teeth_test.log` md5:676a088505573d8c6f03336e9e30a731 + `fla_cross_check_fixed_pass.log` md5:c2c68f8a7a9ba4d4938cecd279cb07a6 | §6 text + Table 1 row |

### Cross-cutting

| Claim id | Claim | Verdict record | Raw artifact | Figure / table |
|---|---|---|---|---|
| N1 | argmax closure (no number claimed): under argmax decoding a rank-1 matrix recovers ≈d associations, so exact-continuous grading is required wherever a rank/capacity bound is claimed; the Case-II ladder pre-registered the discrete/continuous gap ("rung 1 does not imply rf@0.9") before it fired | Nichani, Lee & Bietti, ICLR 2025 (arXiv:2412.06538) + `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.28 item 4 | citation + design-doc pre-registration (no numerical claim) | §2 rule 5 |

## Figures to generate

Single versioned script `figures/figure_gen.py`, md5-asserted sources,
committed PDFs.

- `fig1_lens_contradiction.pdf` — 62-cell scatter: primary vs crosscheck
  recovered-fraction@0.9 at the far (8×) depth, point color = training
  convergence (final_loss, log scale). Takeaway visible at a glance: the
  two lenses agree along the diagonal at (0,0) wherever training did not
  converge, and the entire off-diagonal (0, 1) column is converged cells
  — the broken-primary signature. Source: `crosscheck_lens_verdict_output.json`
  (md5 asserted in-script).
- `fig2_tap_localization.pdf` — two panels from
  `tap_localization_SUMMARY.json` (md5 asserted): (a) causal
  state-zeroing (rung-1 accuracy: intact / S0-zeroed / S1-zeroed,
  contender vs ablation, chance line) — the probed layer is causally
  inert; (b) ridge rf@0.9 by tap variant (i)–(iv), contender vs ablation
  — only the post-nonlinearity tap exposes the recall, and only where
  recall exists.

## Nearest prior work (distinguish by name)

- **Silberzahn et al. (many-analysts):** crowd variance across analysts
  on one dataset; this paper is one team, serial incidents, with
  pre-registered per-incident adjudication rules and falsifier teeth —
  a process for deciding WHICH analysis is broken, not a demonstration
  that analyses vary.
- **D'Amour et al. (underspecification):** equally-good training
  pipelines diverging in deployment; here the MODELS are fixed and the
  MEASUREMENT pipeline is the underspecified object being adjudicated.
- **Henderson et al. / Bouthillier et al. (variance accounting):**
  seed/HP variance changing leaderboards; our incidents are structural
  instrument defects (wrong population, wrong layer, wrong gauge, wrong
  convention), not sampling variance — variance accounting would not
  catch any of the six.
- **Kapoor et al. (REFORMS):** reporting standards after the fact; this
  paper supplies the in-flight adjudication machinery (crosschecks with
  teeth, instrument-health gates) that a checklist presumes exists.
- **Nichani, Lee & Bietti:** the theoretical argmax/exact-recovery gap;
  we report it firing live inside a trained-system evaluation and the
  pre-registration that kept it from becoming a wrong claim.

## Anonymization surface (double-blind assumed)

Tokens per `venue-requirements.md`. The anonymized build must not name
the repo, the design-doc filenames, or internal § anchors in rendered
prose (source comments exempt; they are stripped in the bundle's
flattened submission TeX only if the venue requires source upload —
default: PDF-only, comments harmless but keep prose clean regardless).

## Dual output

- [x] Venue submission (anonymized, 4pp, NeurIPS-2025 stand-in kit)
- [ ] Public write-up — NOT produced this session (the pebbleml.com
  findings pages already carry the program's public material; a
  measurement-methodology page can consume this same evidence map later).
