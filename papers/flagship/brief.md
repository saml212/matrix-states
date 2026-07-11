# Paper brief — Flagship: The Matrix State Is Real (rank law + capability separation + the scale-worsening write-geometry attractor)

**[WORKING TITLE — PI]** — the title above is a drafting placeholder; the
final title is a PI decision. Drafting proceeds under thesis **T1** (the
PI-recommended candidate, the drafting thesis of record; final thesis
selection also remains with the PI).

Stage 0 of the `paper` skill (`/paper-draft`, repo mode). Per the skill's own
rule, **no section is drafted until this brief exists and every planned
numerical claim has a three-field evidence row** (pre-registered verdict
record + raw artifact path + figure). Sources (design docs, EXPERIMENT_LOG,
experiment-runs archives) are read-only; every paper artifact lands under
`papers/flagship/` and is committed as its persistence event.

**Authors:** [AUTHORS — PI decision pending]

## Venue

Drawn from `papers/flagship/venue-requirements.md` (Stage-0 artifact,
recorded 2026-07-10 from the live-verified `papers/VENUE_MAP.md`, commit
52eca3a): the ICLR 2027 CFP is NOT live (404 verified 2026-07-10); the
dates below are PROJECTED; the `iclr2026/` kit is the sanctioned stand-in.

- **Name:** arXiv preprint (~end July 2026), then ICLR 2027 (deadline ~late
  Sept 2026, PROJECTED — abstract ~Sep 19 / full ~Sep 24 per aggregators).
- **Format:** 9 pages main text (excluding references and appendix), ICLR
  LaTeX kit. Per `skills/ml-paper-writing/templates/VENUE_STATUS.md`, ICLR has
  not released a 2027 style file — draft in the `iclr2026/` kit as the
  sanctioned stand-in, swap in the official `iclr2027/` kit when it ships
  (never hand-edit year strings).
- **Review style:** double-blind (ICLR) → the anonymization grep applies to
  the submission build; the arXiv preprint is the named build.
- **Archival:** archival (ICLR proceedings); arXiv preprint precedes it.
- **Deadline:** arXiv target ~2026-07-31; ICLR 2027 ~late Sept 2026 AOE.

## Thesis (one falsifiable sentence — 3 candidates, PI chooses)

**T1 (unified mechanism — RECOMMENDED):**

> Delta-rule fast-weight matrices are a genuine representational medium —
> SGD recruits exactly the task-minimal rank d_min (recovery is 0.000 one
> rank below and returns at d_min), the layer-0 state causally carries an
> episodic-recall capability that parameter-matched transformer and
> vector-state baselines lack, and that capability becomes linearly legible
> only after downstream nonlinear processing — while the same write
> mechanism drives a population-geometry pathology that worsens
> monotonically from 14M to 1.31B parameters and is not removed by the
> community's standard mitigation.

Falsified by any of: the force-rank razor step failing to appear on the
repaired instrument (R3); S₀-zeroing not collapsing recall / S₁-zeroing
changing it (R5); a matched baseline closing the recall gap at equal budget
(R4, the n=3 sweep is exactly this test); the span-fraction ladder breaking
monotonicity or qk-norm removing the attractor (R6, R7). T1 is recommended
because it is the only candidate all nine evidence rows load onto; its
formerly weakest leg (recall) is now verdict-grade — the n=3 sweep WIN
(R4, §1.40) with paired CIs excluding the margin at both comparisons.

**T1 WORDING FLAG (2026-07-10, drafting-time verification — PI decision,
flagged not decided):** T1's opening clause attributes the rank law to
"delta-rule fast-weight matrices," but R1–R3's substrate is the Stage-1
MATRIX-STATE ENCODER family (`model_v4.py` BindingEncoder — an
nn.TransformerEncoder writing a single d_state×d_state matrix Z under the
hard P=1 single-state bottleneck; `CAPABILITY_SEPARATION_DESIGN.md` §1.4),
NOT a delta-rule model. The delta-rule substrate carries R4/R5/R10
(capability) and R6–R8 (pathology); R9 itself records the two families as
architecture-conditional (encoder-family c*·I scaffold EMPTY in
DeltaNet-family states). The draft therefore words the thesis with
per-leg substrate scoping ("matrix-valued state representations" as the
unifying object; the rank law on the matrix-state encoder family; the
capability and pathology legs on the delta-rule family) so that every
clause is supported by its own evidence row as written. If the PI wants
T1's literal "delta-rule" attribution for the rank law, that requires NEW
evidence (a rank-law replication on a delta-rule substrate), which no row
provides today.

**T2 (capability-first, per the PI's 2026-07-08 capability directive):**

> A two-layer delta-rule fast-weight model performs single-pass episodic
> recall that parameter- and data-matched transformer and vector-state
> baselines functionally cannot (0.999 vs ≤0.045 at equal budget), because
> the bindings are stored nonlinearly in the first layer's matrix state —
> zeroing that state collapses recall to chance while zeroing the second
> layer's changes nothing — and the state recruits exactly the
> task-minimal rank.

Falsified by the n=3 sweep landing TIE/LOSE, by the S₀/S₁ zeroing pattern
failing on fresh seeds, or by the razor step failing. Narrower and punchier;
drops the pathology arc to a supporting section, which wastes rows R6-R8.

**T3 (pathology-first, two-faces framing):**

> The same delta-rule write mechanism that provably stores task structure
> at minimal rank also drives its states toward a low-dimensional
> write-geometry attractor that worsens monotonically with scale
> (span-fraction 0.248→0.455, 14M→1.31B) and survives both qk-norm and
> gating — capability and pathology are two faces of one storage mechanism.

Falsified by the ladder breaking, by qk-norm/gating removing the attractor
at n=3, or by the rank/recall legs failing (which would sever the "two
faces" link). Strongest overlap with the existing `iclr-2027/` attractor
draft; weakest fit to the capability-first directive.

## Contribution bullets

1. **The rank law, causally closed:** across five permutation groups
   spanning the solvable/non-solvable divide, recruited effective rank
   tracks the minimal faithful representation dimension d_min (Spearman
   ρ=0.9747, the tie-capped maximum; 19/19 unconstrained cells in the
   pre-registered band), the designed S4-vs-A5 pair is TOST-equivalent
   (dimension, not solvability), and d_min is causally necessary AND
   sufficient (recovery exactly 0.000 at k=d_min−1 in all 5 groups and all
   4 S3 seeds; recovery returns at k=d_min in 5/5). Nearest prior work
   measures rank descriptively on frozen checkpoints; this is a train-time
   causal intervention with a pre-registered razor.
2. **A capability separation matched baselines lack:** at matched params
   and data, the matrix-state contender reads 0.9990 on episodic recall vs
   0.0447 (vector-state ablation) and 0.0295 (transformer) — >3× the
   pre-registered 0.30 WIN margin — and the capability is provably
   fast-weight-resident (S₀-zeroing collapses it to chance; S₁-zeroing
   changes nothing), with the recall geometry exposed only downstream of
   nonlinear processing (rf@0.9 = 0.674 at the pre-LM-head tap vs 0.0 at
   every state-level linear tap). CONFIRMED at n=3 (2026-07-10): per-seed
   contender [0.99951, 1.00000, 0.99902], paired CIs (0.958, 0.973) vs the
   ablation and (0.969, 0.974) vs the transformer, both excluding the 0.30
   margin — row R4 below is claim-grade.
3. **A scale-worsening pathology standard mitigations do not fix:** the
   write-geometry attractor's span-fraction climbs 0.248→0.344→0.389→0.455
   across a held-fixed-mix 14M→98M→392M→1.31B ladder (pure scale), was
   measured WITH qk-norm active throughout, and turning qk-norm off is a
   within-noise null at n=3 (Δ=−0.103, 0.05σ) while gating trends
   amplifying (not confirmed at the pre-registered bar). Prior qk-norm work
   conditions single-vector eigenvalue stability; this is cross-key
   population geometry — a different axis, measured causally.
4. **A mechanism scaffold tying them together:** S₀ nonlinear storage (the
   binding lives in layer 0's state, legible only after block 1's nonlinear
   transform), plus the c*·I complement law in the encoder family
   (Z ≈ c*·I + rank-(K−1) task correction at 0.5-2.9% residual, a loss-blind
   health signal at ρ=−0.973) as the appendix/mechanism candidate.

## Per-section page budget (ICLR 9.0 pages main text)

| Section | Pages | Purpose |
|---|---|---|
| 1 Introduction | 1.0 | The thesis, the three-leg structure, contribution bullets, the capability-first framing |
| 2 Background & setup | 1.0 | Delta-rule fast-weight models; the group-composition and episodic-recall tasks; the instruments (restricted effective rank, force-rank arms, acc_A, span-fraction) and their repair history in one paragraph |
| 3 The rank law | 2.0 | M1 observational (R1) + marquee TOST (R2) + the causal razor (R3), with the D-AMB instrument-defect-found-and-fixed methodology narrative |
| 4 Capability separation | 2.0 | Head-to-head recall WIN (R4) + S₀ localization / nonlinear storage (R5); the Nichani caveat carried on every Leg-A number |
| 5 The pathology at scale | 1.75 | The 4-point ladder (R6) + qk-norm exoneration & gating trend (R7) + fix-at-scale adjudication (R8, LANDED: no tested fix transfers) |
| 6 Related work | 0.5 | Distinguish BY NAME: Nichani-Lee-Bietti, the descriptive linear-attention rank papers, the qk-norm stability line, DeltaNet/Gated-DeltaNet |
| 7 Discussion & limitations | 0.5 | Scale limits of the capability result; geometry-leg-only scope of the ladder; synthetic-task scope; Stage-2 depth generalization explicitly flagged future work |
| 8 Conclusion | 0.25 | One paragraph |
| **Total** | **9.0** | equals the ICLR main-text limit |

Appendix (unlimited): full tables, the c*·I complement scaffold (R9), the
instrument-repair records, reproducibility pointers.

## Claims-to-evidence-to-figure map

One row per numerical claim. Every row names the pre-registered verdict
record AND the raw artifact (path + md5, computed 2026-07-09 with `md5 -q`).
A number with no raw artifact is a CRITICAL finding, not a claim. All nine
rows are now claim-grade: R4's sweep verdict LANDED 2026-07-10, and R8's
fix-at-scale harvest verdict LANDED 2026-07-10 (§13.22 — formerly the
reserved row).

| Id | Claim (with the number) | Verdict record (§ / log entry) | Raw artifact (path + md5) | Figure / table |
|---|---|---|---|---|
| R1 | Restricted effective rank tracks d_min across 5 groups: 1.877/2.852/2.832/3.591/4.736 vs d_min 2/3/3/4/5; Spearman ρ=0.9747 (tie-capped max, exact-null P(ρ≥0.8)=6.67%); 19/19 unconstrained cells in the [0.7,1.3]·d_min band | `CAPABILITY_SEPARATION_DESIGN.md` §1.33; `EXPERIMENT_LOG.md` 2026-07-09 capability-sweep entries | `experiment-runs/2026-07-09_capability_sweep_harvest/harvest_summary.json` md5:7dce77dcba724cd1004419ac71fe5f2f (+ `results/` 58 per-cell JSONs, `md5_local.txt` vs box verified 61/61) | fig1_rank_vs_dmin.pdf |
| R2 | Marquee S4-vs-A5 TOST DECLARE: diff +0.0194 rank-units, se 0.0368, df 7.83, t1=13.06/t2=14.12 vs t_crit=1.865 at margin ±0.5 — the pair lands together on dimension, not apart on solvability | §1.33 (same record; evaluated by the repo's pre-registered `tost_analysis.py`) | same archive as R1 (`harvest_summary.json` + `harvest_analysis_output.txt`) | table in §3 (or fig1 inset) |
| R3 | Causal razor 5/5: k=d_min−1 xrec90 = 0.000 in all 5 groups (and 0.000 in all 4 independent S3 seeds); k=d_min recovers past the fixed 0.9×anchor bar (S4 0.800/0.585, A5 0.700/0.630, S5 0.600/0.450, A6 0.650/0.585; S3 seed-mean 0.5625 ≥ 0.495) | §1.36 (CAUSAL-CONFIRM) + §1.36a (S3 seed extension CONFIRMED) | `experiment-runs/2026-07-09_m3fix_harvest/` (30 per-cell JSONs; `harvest_analysis_output.txt` md5:77be9c3b092c70e83ff08a0261575815) + `experiment-runs/2026-07-09_m3fix_s3ext/` (18 per-cell JSONs; `harvest_analysis_output.txt` md5:d4413385f9f45b71ff9707354ed6b055) | fig2_forcerank_staircase.pdf |
| R4 | **Head-to-head recall — THE n=3 VERDICT OF RECORD (frozen §1.31.1 tiers): AXIS-1 TASK1-PRIMARY LEG-A WIN.** Contender acc_A per-seed [0.99951, 1.00000, 0.99902] (mean 0.99951, every seed ≥10.7× the 0.09375 bar) vs ablation [0.03223, 0.03271, 0.03687] vs transformer [0.02710, 0.02930, 0.02856] (neither baseline ever clears); Δ(cont−abl) mean 0.96558, paired t-CI (df=2) (0.95822, 0.97293); Δ(cont−tfm) mean 0.97119, CI (0.96855, 0.97383) — both exclude the frozen 0.30 margin; n=3→9 extension trigger silent. Confirms the round-4 n=1 preview (0.9990/Δ 0.9543). DISCLOSED alongside: task2 = INDETERMINATE with a surprise (contender s2 = 0.33447 clears the bar, s0/s1 + all baselines at chance; no held-out-hop generalization, 0.0112) → diagnosis round; transformer read = degenerate-baseline datum (matched-budget caveat). Nichani caveat on every acc_A number per `MARGINS_FROZEN` | `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.40 (verdict of record) + §1.37/§1.38 (n=1 preview, freeze+launch) | `experiment-runs/2026-07-10_h2h_sweep_harvest/` — 27 training JSONs + 18 verdict-grade re-metric JSONs + `MARGINS_FROZEN.token` md5:58fe68e7b15728739a5176d7e591e204 + `md5_manifest.txt` (50 files, local==box) + exact eval script md5:e47d69fabdd51ba24896e76c5d13a3ab + `compute_verdict.py` (the CI arithmetic) | fig3_recall_separation.pdf (add per-seed panel at build) |
| R5 | S₀ localization + nonlinear storage: zeroing S₀ collapses contender recall 0.9990→0.0286 (≈chance 0.03125) while zeroing S₁ leaves it at 0.9990; no state-level linear tap clears rf@0.9 in either arm; the pre-LM-head tap reads rf@0.9=0.674 / cos 0.894 (contender only; ablation 0.0/0.119) | `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.30 (Tables 1-2) | `experiment-runs/2026-07-09_h2h_tap_localization/results/tap_localization_contender.json` md5:362333c89f4223c427fe8daf54f50fce; `tap_localization_ablation.json` md5:ff8e352a13c2bc1e177f53f8cef47c01 | fig4_tap_localization.pdf (zeroing table + tap bars) |
| R6 | Attractor scaling ladder, pure scale, held-fixed extended mix: span-fraction 0.248 (14M) → 0.344 (98M) → 0.389 (392M) → 0.455 (1.31B), monotone; PROMINENT DISCLOSURE: the 1.31B rung self-terminated at ~84.7% of the token-matched budget with span-frac flat-to-declining over the final window (0.4584→0.4554) | `EXPERIMENT_LOG.md` 2026-07-07 "Track C Wave 3 (rung-3, 1.31B) harvest" (line ~5463); `iclr-2027/NARRATIVE.md` Fig-9 spec + `sections/09/10.tex` | `experiment-runs/2026-07-06_trackc_rung3/probe_analysis_rung3.json` md5:6a627c315b0c8e35e084bbbe7730a2f8 + `rung3_final_pooled.json` md5:96597f682a0b972d4a1ad7922828efcc; lower rungs `experiment-runs/2026-07-04_trackc_rung1/`, `2026-07-05_trackc_rung2/`, `2026-07-05_wave1ext/`; dense per-seed regeneration `experiment-runs/2026-07-06_trajectory_probes/trajectories_tidy.json` md5:0fe53d8b40285b93fe81219fa6ff9606 (plot `archived4_span_frac`, the cross-scale convention) | fig5_attractor_ladder.pdf |
| R7 | Mitigation exoneration (14M, 2×2 qk-norm × gating): qk-norm OFF is a within-noise null at n=3 (Δ=−0.103 = 0.05σ vs the corrected same-corpus floor 2.244355) — the attractor is not a qk-norm artifact; gating reads +4.312 = 1.92σ, BELOW the pre-registered 2σ=4.489 bar — a direction-consistent trend (3/3 paired seeds positive; exploratory Welch p=0.062), NOT a confirmed amplification and NOT a null | `EXPERIMENT_LOG.md` 2026-07-09 "ATTRACTOR-ROBUSTNESS 2×2" entries: build-audit (noise-floor correction), n=1 screening harvest, n=3 escalation harvest | `experiment-runs/2026-07-09_attrrob_2x2_escalation_harvest/box_results/AGGREGATE.json` md5:d7e4d6b45c7a23d5b2a661257dca2c82 + `n3_recompute_summary.json` md5:b4bdffdf25bf85f84945a40b9170a467 (n=1 screening: `experiment-runs/2026-07-09_attrrob_2x2_harvest/`) | fig6_2x2_mitigations.pdf |
| R8 | **Fix-at-scale VERDICT (LANDED 2026-07-10): no tested frozen-bias construction stabilizes the attractor at scale — §5 must say "no tested fix transfers."** Deployed per_token arm (λ=0.58, n=3, vs artifact-matched arm_off′, pinned t(2,.975) CI): 98M Δspan_frac **+0.1133 [+0.0543,+0.1723]** openr1 / **+0.1011 [+0.0541,+0.1482]** wikitext (destabilizing, the 14M sign, both instruments incl. pre-blend co-primary +0.0796/+0.0606); 392M **+0.0189 [+0.0112,+0.0266]** wikitext (co-primary +0.0140), openr1 null [+0.0065, CI straddles 0]; reverses NOWHERE. Global-vector probe (n=1 exploratory, never a confirmatory bar): 14M's −0.33/−0.23 decays to −0.058/−0.034 at 98M and −0.012/**+0.019 (sign flip)** at 392M. Val-loss neutrality PASSES all 8 gates — the free-of-cost half transfers; the geometry half does not. Caveat: 392M = 20k-step budget (token-confounded vs 98M; within-scale claims only) | `FROZEN_BIAS_LM_DESIGN.md` §13.22 (verdict of record; §13.6 pre-registration = PARTIAL both scales); `EXPERIMENT_LOG.md` 2026-07-10 fix-at-scale harvest entry | `experiment-runs/2026-07-10_fixscale_harvest/fixscale_harvest_verdict.json` md5:f2f0aae84908c0db0a42b13c76a85158 (+ 16 comparator JSONs in `measure/`, 2 pins in `pins/`, 28 train/calib JSONs, `md5_manifest.txt` local==box) | fig7_fixscale_transfer.pdf (per-scale delta forest plot, TBD at build) |
| R9 | c*·I complement scaffold (appendix/mechanism candidate): in the Task-E encoder family the trained state obeys Z ≈ c*·I_d + rank-(K−1) task correction at 0.5-2.9% Frobenius residual (τ_identity ≥ 0.9997; per-example conformal lock ~0.5%); PER-LOAD effective rank of Z−c*I (amended 2026-07-10 per round-1 A3/FIX-8 — the earlier "within 0.3 at both loads" claim was false at K=12): converged K=8 runs (s1–s4) deviate 0.03–0.26 from the K−1=7 target; converged K=12 runs read 10.22/10.30/10.41 vs target 11, deviations 0.59–0.78 (K8_frN_s0 is non-converged, proc_resid 0.152, excluded by the converged-family scope); deviation-from-c*I is a loss-blind health signal (Spearman ρ=−0.973 all-11 runs); the complement channel is EMPTY (fD ≤ 3.2e-12) in DeltaNet-family delta-rule states — architecture-conditional | `EXPERIMENT_LOG.md` 2026-07-09 "Z-DUMP ORTHOGONAL-COMPLEMENT PIGGYBACK" entry (commit 64c59d9) | `experiment-runs/2026-07-09_zdump_complement/complement_results.json` md5:03208edab7adc7e433f2cad46ee975bb (+ `report.txt`, `complement_analysis.py`) | appendix figA1_complement_scaffold.pdf |
| R0 | Setup/configuration constants (design-pinned, verified in archived run configs): contender DeltaNetLM d_model=256, n_layers=2, d_state=64, 14,048,896 params; ablation 14,048,384 params (Δ<0.01%); transformer depth pinned =2, FLOP-matched ≤5%; LR REGIME (amended 2026-07-10 per round-1 FIX-1 — the earlier "lr frozen at calibration (1e-3)" wording conflated tasks): the 3-point LR grid + frozen 1e-3 override were TASK-3-ONLY (`transformer_task3_lr=1e-3`, HEAD_TO_HEAD_DEMO_DESIGN.md §1.38/line 4649); the transformer's task-1 (recall) and task-2 cells trained at the shared default 3e-4, identical to contender/ablation, never task-specifically gridded (verified `"lr": 0.0003` in all three task-1 training JSONs); all h2h arms 20,000 steps, lr 3e-4 (task-3 transformer excepted as above), K=32, T_bind=224 tokens (32×7), query_len=6, chance=1/32=0.03125, bar=0.09375; per-layer state 64×64 fp32 = 16,384 B, 2 layers = 32,768 B; Stage-1 encoder h=32, n_layers=3, d_state=d_min+2, step budgets S3/S4/A5/S5/A6 = 8K/20K/20K/8K/40K; ladder rungs 14M/98M/392M/1.31B = (256,2,64)/(768,12,64)/(1536,16,128)/(2560,22,128) measured params 14,048,896/97,618,176/391,869,440/1,311,135,488; span-frac anchors random √(K(K−1)/d), collapse √(K(K−1)) (7.94/63.50 at d=64, 5.61/63.50 at d=128, K=64) | `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.2/§1.3/§1.4/§1.31.1; `CAPABILITY_SEPARATION_DESIGN.md` §1.4; `SCALE_TRANSFER_DESIGN.md` §5.3; configs embedded in every archived run JSON | n_params/config fields in R4's 27 training JSONs, R6's `trajectories_tidy.json` (n_params, d_model, d_state, n_layers, anchor_random, anchor_collapse fields), R1's per-cell JSONs | setup text + appendix table |
| R1b | D-AMB diagnosis (the §1.33 instrument-defect record, methodology narrative): the eye-padded target ρ⊕I has rank d_state with all singular values 1, so a rank-k arm's direct-cosine optimum is √(k/d_state); 36/39 force-rank cells sat within 0.07 of that predicted ceiling (CORRECTED 2026-07-10 from 37/39 per round-1 A4/FIX-9, attack+defense+writer each independently recomputing 36/39 from the raws; the borderline third violator is S5__k_dmin_plus_1__seed0 at \|Δ\|=0.0716, alongside the two S5 k_dmin−1 outliers at 0.15/0.17; the harvest script's "only outliers" line was a hand-written string literal, never a computed count) — the first sweep's M3 arms were VOID as a causal test; the zero-pad fix (target rank exactly d_min) + pre-registered C1 crosscheck metric (full-Q Procrustes; the scale-only primary reads 0.01–0.22 cos for a flawless oracle) purchased R3 | `CAPABILITY_SEPARATION_DESIGN.md` §1.33 (D-AMB) + §1.34/§1.35 (fix build + C1 pin) | same archive as R1 (`harvest_summary.json` d_ambient fields + `results/` per-cell JSONs); fix-wave archive as R3 | §3 methodology paragraph (no figure) |
| R10 | **Axis-2 M\* (inference-memory-matched) — VERDICT OF RECORD: baseline non-competitive at matched params/tokens** (the pre-registered degenerate-baseline clause fired; NEVER certified M\*=∞/strongest-win). The two informative reads that stand: (i) the contender holds acc_A ≥0.998 per-seed at every horizon H2=454/H4=902/H8=1798 tokens (8× T_bind) with a FIXED 32,768-byte state — the constant-memory property demonstrated contender-side; (ii) KV-capping never rescues the transformer (every M∈{1,2,4,8,16,32} at chance, 0.020-0.033, alongside the uncapped read 0.029-0.036 — forced-locality answered: no). Every per-M H4 paired gap CI floor ≥0.958 vs the 0.20 crossover margin; no straddle, no extension. Task2 disclosed: joint-NO-RECALL TIE at all five points; contender s2's T_bind partial recall (0.334) collapses to 0.010 at every horizon. Matched-budget + Nichani caveats attached | `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.41 (verdict of record; pre-flight commits 8f825f4+be8cd3f) | `experiment-runs/2026-07-10_h2h_mstar/MSTAR_VERDICT.json` md5:4f115ad55d5301122f387df504efa35c (+ 90 fan-out JSONs + 3 reference JSONs + 10 stage logs + `md5_manifest.txt`, per-file local==box independently reverified) | fig3 companion panel (contender horizon flatline vs capped-M grid) — TBD at build |

**Rows the paper does NOT yet have (flagged, not claimed):** Stage-2
compositional depth generalization ("depth generalization incoming") has no
verdict record and no artifact — it is future work in §7, not a claim.
Likewise the h2h task2/task3 axes (task2 is the pre-registered joint-failure
TIE branch pending its diagnosis round — now seeded with §1.41's
horizon-collapse fact) are out of this brief until their records exist; the
M\* axis LANDED as R10 above (2026-07-10).

**K48 stress table + task2 diagnosis — LANDED as §1.43 (2026-07-10,
superseding this brief's earlier IN-FLIGHT flag; verified against the
raws by the writer AND independently by the round-1 gauntlet's attack and
defense agents):** row R11 below carries the record. The earlier flag
("the draft must not carry it as a number") is DISCHARGED by R11; the
gauntlet's A5 fix updates the draft's stale "in flight" language.

| Id | Claim (with the number) | Verdict record (§ / log entry) | Raw artifact (path + md5) | Figure / table |
|---|---|---|---|---|
| R4a | Transformer task-1 training curves (ADDED 2026-07-10 per round-1 FIX-1: the under-optimization disclosure): all three task-1 transformer cells trained at `"lr": 0.0003` (the shared default, never task-gridded); train_loss near flat across the run — s0 7.836→7.514 (min 7.277), s1 7.811→7.453 (min 7.392), s2 7.822→7.459 (min 7.423) from step 500→20,000; loss_final_mean5 7.472/7.446/7.464 | same record as R4 (§1.40; the curves are disclosure, not a verdict input) | `experiment-runs/2026-07-10_h2h_sweep_harvest/h2h_transformer_task1_sweep_s{0,1,2}.json` md5:427de1aaf3330f6f6d407fb12a49652a / 43d5a97707a797aae8448975f867d29e / 4d82e706b1d3c2cda3f9b4cb5dd7e195 (`curve` + `lr` fields) | §4.1 prose disclosure (no figure) |
| R11 | Task2 diagnosis + K48 stress — §1.43 VERDICT OF RECORD: task2 trainability/seed-variance CONFIRMED at pooled rate 3/9 (contender seeds clearing the 0.09375 bar: 0.33447/0.47949/0.39087, i.e. 10.7–15.3× chance; ablation 0/9 ever); the hard-capability-boundary hypothesis REJECTED for task2 at this scale/budget; pooled n=9 reading NON-DECISION-GRADE (batch-effect gate flagged, ablation var-ratio 6.14>4.0); strict-tier disclosed-only TIE; all bar-clearing seeds horizon- and held-out-hop-fragile; task2 stays NON-verdict-bearing for axis 1. K48 (K/d=0.75, chance 0.02083, locate-only bar 0.0625): contender 0.01888 / ablation 0.01953 / transformer 0.02181 (fresh cell) — all ≈chance, none clears; three-arm stress table COMPLETE | `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.43 (verdict of record; §1.42 pre-run record) | `experiment-runs/2026-07-10_h2h_task2diag/results/TASK2DIAG_VERDICT.json` md5:66d2291d8e65932d368d8978bfd16bdc + `transformer_task1_stress_K48_round4.json` md5:14e0c93f56c2a55983f929b7313eb5ac (+ 12 training and 12 re-metric JSONs, md5 manifest) | §4.5/§7 prose (no figure) |

**Instrument-validity methodology citations (qualitative, no numbers →
no evidence rows needed):** the §2 instruments paragraph and the §7
methodology sentence may cite, as process precedent, (a) the wrong-layer
instrument diagnosis in the recall program
(`HEAD_TO_HEAD_DEMO_DESIGN.md` §1.27–§1.30: three "failure" rounds were a
wrong-tap linear probe; the repaired §1.30-localized tap is what R5
reports) and (b) the Stage-2 primary-vs-crosscheck lens tiebreak
(`CAPABILITY_SEPARATION_DESIGN.md` §2.31a coordinator tiebreak: the
primary Procrustes lens is broken on converged cells, 0-vs-1.0
contradiction on 4 independent grounds; §2.32 crosscheck re-metric with
shuffled-target teeth 3/3). These are cited as records, qualitatively; no
Stage-2 model verdict is claimed (Stage 2 remains future work in §7).

## Figures to generate

All from the single versioned `figures/figure-gen.py` (filled from the
skill's template; asserts input md5s against the map above).

- `fig1_rank_vs_dmin.pdf` — recruited restricted effective rank vs d_min, 5
  groups with per-seed points and the [0.7,1.3]·d_min band; takeaway: rank
  tracks dimension at the tie-capped-maximum ρ.
- `fig2_forcerank_staircase.pdf` — xrec90 vs force-rank k per group; the
  step at d_min (0.000 below, anchor-class at/above); S3's 4-seed extension
  shown per-seed.
- `fig3_recall_separation.pdf` — acc_A for contender/ablation/transformer
  vs the 0.09375 demonstration bar and the 0.30 WIN margin; UPDATE at build:
  add the n=3 per-seed panel from R4's landed verdict (§1.40); S₀/S₁-zeroing inset.
- `fig4_tap_localization.pdf` — Table-1 zeroing results + Table-2 tap-variant
  rf@0.9 bars, both arms; takeaway: storage in S₀, legibility only
  post-nonlinearity.
- `fig5_attractor_ladder.pdf` — span-fraction vs params (4 points,
  log-x), per-rung random/collapse anchors, the 1.31B budget-shortfall
  disclosure in the caption; regenerate from `trajectories_tidy.json` +
  rung-3 pooled JSON.
- `fig6_2x2_mitigations.pdf` — 2×2 cell means ± sd (n=3), the corrected
  noise floor and 2σ bar drawn; takeaway: qk-norm exonerated, gating a
  trend below bar.
- `figA1_complement_scaffold.pdf` (appendix) — Procrustes residual vs
  health class + the ρ=−0.973 scatter.

## Nearest prior work (distinguish by name)

- **Nichani, Lee & Bietti (ICLR 2025, arXiv:2412.06538):** under argmax
  decoding a rank-1 matrix recovers ≈d associations — this paper's readouts
  force exact continuous recovery (the razor is provable, not
  decode-laundered), and the caveat is carried verbatim on every recall
  number per the frozen margins.
- **The Feb-2026 linear-attention rank papers (arXiv:2602.04852,
  2602.02195):** descriptive rank measurement on frozen checkpoints; this
  work intervenes at train time (force-rank arms, state-zeroing) and closes
  the causal loop in both directions.
- **The qk-norm stability line (Kimi Linear arXiv:2510.26692 §4,
  Qwen3-Next):** conditions single-VECTOR eigenvalue stability; the
  attractor is cross-key POPULATION geometry, measured with qk-norm active
  throughout and unchanged when it is removed (n=3).
- **DeltaNet / Gated-DeltaNet (Yang et al.):** the architecture substrate,
  benchmarked for quality/efficiency; this paper studies what the states
  REPRESENT and how that geometry scales, not leaderboard performance.
- **The program's own ICML 2026 MI-workshop paper ("The Gradient Does Not
  See Rank"):** the bolt-on negative result; this paper is the from-scratch
  positive+pathology arc that answers it.

## CHOPS note — workshop overlap, by design

Which evidence rows the sibling papers already use, so flagship overlap is
deliberate (the flagship is the ONE full paper over the program's chop
inventory, per the PI's 2026-07-08 many-workshops-one-flagship strategy):

- **`capability-ws-2026/RANK_LAW_SKELETON.md` (4pp target):** uses R1, R2,
  R3 verbatim (same §1.33/§1.36/§1.36a records, same three archives). The
  flagship's §3 is the superset telling of the same trilogy — shared by
  construction; the workshop paper is the fast non-archival flag-plant.
- **`iclr-2027/` (the existing attractor full-paper draft):** uses R6 (its
  Fig 9 / §9-§10 already carry 0.248→0.455) and has R7 QUEUED as a fold
  into `06_soft_fixes_fail` (the n=3 gating fold must carry
  trend-not-confirmed; qk-norm exoneration is fully foldable); R8's landed
  verdict (§13.22, no-tested-fix-transfers) is QUEUED for its
  07_the_fix/09_discussion folds via `NARRATIVE.md`. The flagship's §5 compresses that draft's Tier-2 spine. **PI
  decision flagged:** whether the flagship ABSORBS `iclr-2027/` (one ICLR
  submission) or the two stay separate submissions — they cannot both go to
  ICLR 2027 with the same §5 evidence.

  **DRAFTING ASSUMPTION OF RECORD (2026-07-10, per the flagship charter —
  the PI decides absorb-vs-separate; this only fixes how drafting
  proceeds until then): treat the two as SEPARATE.** The flagship is
  written self-contained: §5 cites the attractor results as evidence rows
  (R6–R8, from the raw archives) WITHOUT copying any paragraph from the
  `matrix-thinking/submissions/iclr-2027/` tree, and that tree stays
  untouched. §5 is structured as a self-contained module (its own
  figures, its own evidence citations, no forward/backward dependencies
  beyond the thesis sentence) so that if the PI later picks ABSORB, the
  section can be swapped to a short cross-reference without rewriting
  §§1–4, 6–8.
- **`neurips-ws-2026/` (Task D/E rank recruitment, NeurReps/UniReps EA
  target):** distinct evidence base (synthetic Task D/E archives) — zero
  row overlap with this map; adjacent narrative (rank recruitment) only.
- **`workshop-2026/` (capacity trilogy) and `measurement-2026/` (tolerance
  methodology):** zero row overlap (key-anchoring capacity archives).
- **Flagship-exclusive rows as of today:** R4, R5, R9 appear in NO sibling
  paper — the capability separation + mechanism sections are the flagship's
  novel territory.

## Anonymization surface (ICLR build only)

Author/handle/org tokens for the grep (extend when the author list lands):
`Sam Larson`, `samlarson16`, `samuellarson`, `learned-representations` (repo
name), `youthful-indigo-turkey` (Brev instance), Berkeley/Stanford
collaborator names once added. URL and acknowledgment patterns per
`references/styleguide.md` always apply.

## Dual output

- [x] Venue submission (anonymized LaTeX, 9pp, `iclr2026` kit as the 2027
  stand-in per VENUE_STATUS.md)
- [x] Public write-up (named arXiv preprint, ~end July 2026, full figures,
  real reproducibility pointers into `experiment-runs/`)

Both consume this same evidence map; a number fixed in one is fixed in the
other. The arXiv build precedes the ICLR build; R4's sweep verdict LANDED
(WIN, §1.40); R8's harvest LANDED (PARTIAL, §13.22 — §5's fix passage
reads "no tested fix transfers; val-loss neutrality does") — the evidence
map is freeze-ready.
