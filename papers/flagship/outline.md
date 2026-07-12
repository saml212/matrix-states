# Paper outline — Flagship "Fast-Weight Matrix States Are a Representational Medium, Not an Efficiency Trick" (drafting thesis T1)

One row per section, with its page budget from the brief, the claims it
carries (by id from the brief's evidence map), and the figures it shows.
Venue limit: 9.0 pages main text (ICLR, per
`papers/flagship/venue-requirements.md`; references and appendix excluded).

## Section plan

| # | Section | Pages | Claims (ids) | Figures | Notes |
|---|---|---|---|---|---|
| 1 | Introduction | 1.0 | previews of R1, R3, R4, R6 (evidence comments carried; primary placement below) | — | thesis T1 in one paragraph; the three-leg structure (rank law → capability → pathology); contribution bullets |
| 2 | Background and experimental setup | 1.0 | — (definitions only; no new numerical claims) | — | delta-rule fast-weight state; the group-composition and episodic-recall tasks; the four instruments (restricted effective rank, force-rank arms, acc_A, span-fraction); one instrument-repair paragraph citing §1.27–§1.30 and §2.31a/§2.32 qualitatively |
| 3 | The rank law | 2.0 | R1, R2, R3 | fig1_rank_vs_dmin, fig2_forcerank_staircase | observational law → marquee TOST → causal razor; the D-AMB defect-found-and-fixed narrative in one paragraph |
| 4 | Capability separation | 2.0 | R4, R5, R10 | fig3_recall_separation (+ M* horizon companion panel), fig4_tap_localization | n=3 WIN; S₀/S₁ zeroing localization; nonlinear legibility; constant-memory horizon table + KV-cap grid; Nichani + matched-budget caveats on every acc_A number |
| 5 | The pathology at scale | 1.75 | R6, R7, R8 | fig5_attractor_ladder, fig6_2x2_mitigations, fig7_fixscale_transfer | SELF-CONTAINED MODULE (swappable to a cross-reference if the PI picks ABSORB): ladder → 2×2 exoneration/trend → fix-at-scale "no tested fix transfers; val-loss neutrality does" |
| 6 | Related work | 0.5 | — | — | by name: Nichani–Lee–Bietti; the two Feb-2026 descriptive rank papers; the qk-norm stability line (Kimi Linear, Qwen3-Next); DeltaNet/Gated-DeltaNet; the program's own ICML-ws negative-result paper |
| 7 | Discussion and limitations | 0.5 | — (scope statements; K48 third arm disclosed as in-flight, no number) | — | scale limits of the capability result; geometry-leg-only scope of the ladder; synthetic-task scope; the 392M token-confound caveat; Stage-2 depth generalization = future work |
| 8 | Conclusion | 0.25 | — | — | one paragraph |
|   | **Total** | **9.0** | | | equals the ICLR main-text limit |
| A | Appendix A: the c*·I complement scaffold | n/a | R9 | figA1_complement_scaffold | mechanism candidate; architecture-conditional scope stated |
| B | Appendix B: full tables + instrument-repair records + reproducibility | n/a | — | — | per-seed tables for R1–R8, repair-history records, md5 manifest pointers |

## Outline sanity checks (run before drafting)

- [x] Page budgets sum to the venue limit: 1.0+1.0+2.0+2.0+1.75+0.5+0.5+0.25 = 9.0.
- [x] Every claim id in the brief's evidence map appears in exactly one
  primary section: R1/R2/R3 → §3; R4/R5/R10 → §4; R6/R7/R8 → §5; R9 →
  Appendix A. (§1's previews restate §3–§5 numbers with the same evidence
  comments; no number appears in §1 that is not owned by a later section.)
- [x] Every figure in the brief is placed: fig1/fig2 → §3; fig3(+panel)/fig4
  → §4; fig5/fig6/fig7 → §5; figA1 → Appendix A.
- [x] Related work distinguishes this paper from each nearest neighbor by
  name (§6 list mirrors the brief's "Nearest prior work").
- [x] No section carries a claim whose evidence row is missing or pending:
  all ten rows (R1–R10) are landed and md5-verified 2026-07-10 (14/14
  artifacts re-verified at outline time); the K48 transformer arm and
  Stage-2 depth generalization are flagged non-claims (brief, "Rows the
  paper does NOT yet have") and appear only as disclosed future/in-flight
  work in §7.

## Per-section beat sheet

### 1. Introduction (1.0 pp)
- Fast-weight/linear-attention models hold a d×d matrix state; the field
  benchmarks them for throughput and quality, and treats the state as an
  implementation detail. Whether that state is a representational medium
  in its own right (what it stores, at what dimensionality, with what
  failure modes) is unmeasured.
- Thesis T1 in one paragraph: SGD recruits exactly task-minimal rank; the
  layer-0 state causally carries a recall capability matched baselines
  lack, legible only after nonlinear processing; the same write mechanism
  drives a scale-worsening population-geometry pathology that the standard
  mitigation does not remove.
- Contribution bullets (four, from the brief): the rank law causally
  closed; the capability separation; the pathology ladder + mitigation
  adjudication; the mechanism scaffold (appendix).
- One-sentence method credo: pre-registered verdicts, causal
  interventions, instrument-validity checks; every number traces to an
  archived artifact.

### 2. Background and experimental setup (1.0 pp)
- Delta-rule fast-weight update (the DeltaNet-family state S ← S(I − βkkᵀ)
  + βvkᵀ shape), what "state" means at inference, and the two model
  families used: the Stage-1 single-state-bottleneck composer/encoder
  family and the two-layer recurrent contender used in the head-to-head.
- The group-composition task (five groups, d_min per group) and the
  episodic-recall task (K=32 bindings, episode-restricted argmax recall,
  chance 1/32); the LM setting for §5 (scale ladder, held-fixed data mix).
- Instruments: restricted effective rank; force-rank train-time arms;
  acc_A (metric of record, Nichani caveat stated once here and carried);
  span-fraction. One paragraph on instrument validity as a first-class
  methodology: the wrong-tap probe history (§1.27–§1.30) and the
  primary-vs-crosscheck lens tiebreak (§2.31a/§2.32), cited qualitatively.

### 3. The rank law (2.0 pp)
- Observation (R1, fig1): recruited restricted effective rank tracks d_min
  across five groups; ρ=0.9747 is the tie-capped maximum; 19/19
  unconstrained cells inside the pre-registered [0.7,1.3]·d_min band.
- The designed pair (R2): S4 vs A5 share d_min=3 but differ in
  solvability; TOST declares equivalence (both one-sided t's beat t_crit)
  — rank follows dimension, not solvability.
- Causation (R3, fig2): force-rank arms; recovery is a step function at
  exactly d_min — 0.000 at k=d_min−1 in all five groups and all four S3
  seeds; recovery returns at k=d_min in 5/5 past the 0.9×anchor bar.
- Methodology beat: the first sweep's M3 arms were voided by an
  eye-padding rank tax the harvest caught (D-AMB); the zero-pad fix wave
  purchased the causal test; the C1 crosscheck metric was pre-registered.
- Scope: nearest prior work measures rank descriptively on frozen
  checkpoints; this is a train-time causal intervention.

### 4. Capability separation (2.0 pp)
- The head-to-head design in three sentences: param- and data-matched
  contender / vector-state ablation / transformer; frozen margins
  (MARGINS_FROZEN) before the sweep; WIN/TIE/LOSE pre-registered.
- The verdict (R4, fig3): contender per-seed 0.99951/1.00000/0.99902 vs
  ablation ≈0.034 and transformer ≈0.028; paired CIs exclude the 0.30
  margin at both comparisons; extension trigger silent. Caveats verbatim:
  Nichani (argmax episode-restricted recall, never a rank/continuous-
  capacity claim) and matched-budget (transformer read is a
  degenerate-baseline datum; the WIN is carried by the ablation
  comparison).
- Localization (R5, fig4): S₀-zeroing collapses recall to ≈chance,
  S₁-zeroing changes nothing; no state-level linear tap reads it
  (rf@0.9=0.0 everywhere at state level); the pre-LM-head tap reads 0.674
  — storage is in S₀, linearly legible only after downstream nonlinear
  processing. The three "failure" rounds were a wrong-layer instrument.
- Constant memory (R10, fig3 companion panel): acc_A ≥0.998 per-seed at
  H2/H4/H8 = 454/902/1798 tokens on a fixed 32,768-byte state; KV-capping
  never rescues the transformer (all M∈{1..32} at chance); the
  pre-registered degenerate-baseline clause fired — scope the claim as
  "baseline non-competitive at matched params/tokens," never M*=∞.
- Disclosure beat: task2 (multi-hop) is INDETERMINATE with a one-seed
  surprise, under diagnosis; task1 is the primary per the frozen pin.

### 5. The pathology at scale (1.75 pp — self-contained module)
- The phenomenon (R6, fig5): span-fraction of the write geometry climbs
  0.248→0.344→0.389→0.455 across 14M→98M→392M→1.31B at a held-fixed data
  mix; monotone in scale; the 1.31B rung's budget shortfall disclosed
  prominently (84.7% of token-matched budget; span flat-to-declining over
  the final window).
- Not an artifact of the usual suspect (R7, fig6): qk-norm was ON
  throughout the ladder; turning it OFF is a within-noise null at n=3
  (Δ=−0.103, 0.05σ); gating reads +4.312 = 1.92σ, below the
  pre-registered 2σ bar — a direction-consistent trend, not a confirmed
  amplification and not a null.
- The standard mitigation does not remove it (R8, fig7): the deployed
  per_token frozen-bias arm is destabilizing at 98M on both instruments
  (CI excludes zero), attenuates but never reverses at 392M; the
  global-vector probe's 14M effect decays toward zero and sign-flips at
  392M; val-loss neutrality passes all 8 gates — the free-of-cost half
  transfers, the geometry half does not. Verbatim scope: "no tested fix
  transfers." 392M token-confound caveat stated.
- Two-faces closing beat: the same write mechanism that §3–§4 show
  storing task structure at minimal rank drives this geometry; capability
  and pathology are two faces of one storage mechanism.

### 6. Related work (0.5 pp)
- Nichani, Lee & Bietti: argmax decoding lets rank-1 store ≈d
  associations — our razor forces exact continuous recovery; the caveat
  travels on every recall number.
- Feb-2026 linear-attention rank papers (arXiv:2602.04852, 2602.02195):
  descriptive, frozen checkpoints; we intervene at train time, both
  directions.
- qk-norm stability line (Kimi Linear §4, Qwen3-Next): single-vector
  eigenvalue stability; ours is cross-key population geometry, measured
  with qk-norm active and unchanged without it.
- DeltaNet / Gated-DeltaNet: the substrate, benchmarked for
  quality/efficiency; we study what the states represent.
- The program's own ICML 2026 MI-workshop paper: the bolt-on negative
  result this paper's from-scratch positive arc answers.

### 7. Discussion and limitations (0.5 pp)
- Scale scope: the capability result is at task scale (small models,
  synthetic episodic recall), not LM scale; the pathology result is the
  scale story, but its geometry leg only (no downstream-capability tie).
- The K48 (K/d=0.75) stress table: contender and ablation arms read
  ≈chance; the transformer arm is in flight — disclosed, no number
  claimed.
- The 392M fix-at-scale rung is token-confounded vs 98M (20k-step
  budget); within-scale claims only.
- Instrument validity as methodology: every leg of this paper survived at
  least one instrument-defect-found-and-fixed round (D-AMB, wrong-tap,
  primary-vs-crosscheck); we treat that as evidence discipline, not
  weakness.
- Future work: Stage-2 compositional depth generalization (design cleared,
  no verdict — explicitly not claimed); task2 diagnosis; the absorb
  decision for the companion attractor paper.

### 8. Conclusion (0.25 pp)
- One paragraph: the matrix state is a measurable representational medium
  with a lawful capacity axis, a demonstrated capability separation, and
  a scale-worsening geometric cost; all three from one write mechanism.

### Appendix A: the c*·I complement scaffold (R9, figA1)
- Z ≈ c*·I + rank-(K−1) task correction in the Task-E encoder family;
  residual 0.5–2.9%; deviation-from-c*I as a loss-blind health signal
  (ρ=−0.973); the complement channel is EMPTY in DeltaNet-family
  delta-rule states — architecture-conditional, mechanism candidate only.

### Appendix B: tables, repair records, reproducibility
- Per-seed tables for R1–R8; the instrument-repair records; the md5
  manifest of every raw artifact behind every figure; pointer to
  `experiment-runs/` archives (named build only; anonymized pointer in the
  ICLR build).
