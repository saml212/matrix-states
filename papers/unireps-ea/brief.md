# Paper brief — UniReps 2026 Extended Abstract ("dimension, not solvability")

Stage 0 of the `paper` skill (repo mode). This EA presents the Stage-1
rank-law results (`CAPABILITY_SEPARATION_DESIGN.md` §1.33/§1.36/§1.36a,
banked 2026-07-09, after both pre-built submission drafts were written)
angled to UniReps' convergent-representations theme: five differently
structured groups, one law — the trained state's representational
dimension converges to the algebra's minimal faithful dimension, and a
solvable/non-solvable pair matched on that dimension converges to
statistically equivalent rank. The correlation (ρ=0.9747) and the
marquee TOST equivalence carry the paper; the causal razor is the
supporting leg. Sibling relationship disclosed: a companion EA
(papers/neurreps-ea/) presents the same experimental campaign with the
causal razor as its headline for NeurReps; both are non-archival
extended abstracts and neither burns the ICLR 2027 flagship. Content
decision note: `VENUE_DECISION.md`'s chop plan predates the banked
trilogy and slated the capacity-trilogy draft
(`matrix-thinking/submissions/workshop-2026/`) for this slot; per the
2026-07-09 directive the UniReps EA is instead carried by ρ=0.9747 +
the marquee TOST, so the capacity-trilogy draft is left untouched for
its own venue (e.g. the COLM Efficient Reasoning window).

## Venue

- **Name:** NeurIPS 2026 Workshop on Unifying Representations in Neural
  Models (UniReps) — Extended Abstract track.
- **Format:** 4 pages excluding references/appendix per the 2025 CFP
  language in `VENUE_DECISION.md`. **FLAG: the 2026 CFP is not yet
  published; 4pp+refs is the working assumption, re-check on the live
  CFP (list drops 2026-07-11).** House two-column scaffold pending the
  official style file.
- **Review style:** unknown until the CFP; both builds prepared
  (single-blind placeholder-author + anon). Anonymization grep gates
  the anon build.
- **Archival:** non-archival per 2025 CFP language; re-verify live.
- **Deadline:** CFP expected shortly after 2026-07-11; historically
  late August (2025: Aug 22 → Aug 29 extended).

## Thesis (one falsifiable sentence)

> When matrix-state models are trained end-to-end on the word problem
> of five finite groups spanning the solvable/non-solvable divide, the
> state's restricted effective rank converges to the group's minimal
> faithful real representation dimension d_min (Spearman ρ=0.9747, the
> design's tie-capped maximum) and the dimension-matched
> solvable/non-solvable pair S4/A5 lands statistically equivalent
> (TOST, ±0.5 rank-units) — so learned representational dimension is
> set by the task's algebra, not the group's computational complexity
> class, and either a group leaving the [0.7,1.3]·d_min band or the
> marquee pair separating would have falsified it.

The equivalence claim is falsifiable in the strong sense: the TOST was
pre-registered with a margin, n, and a power simulation (100% power to
reject equivalence at a 1.0 rank-unit gap) before the sweep ran.

## Contribution bullets

1. **Convergence law (headline).** Across S3/S4/A5/S5/A6 (d_min =
   2/3/3/4/5), unconstrained trained states recruit restricted
   effective rank 1.877/2.852/2.832/3.591/4.736; Spearman ρ=0.9747 is
   the exact tie-capped maximum achievable for this family; all 19
   seeds sit inside the pre-registered [0.7,1.3]·d_min band.
2. **Equivalence at matched dimension (headline).** S4 (solvable) vs A5
   (non-solvable, smallest), both d_min=3: Welch TOST at ±0.5
   rank-units DECLAREs equivalence decisively (|Δ|=0.019, se 0.037,
   both one-sided t = 13.06/14.12 vs critical 1.865) — the pair
   converges on dimension and does not separate on solvability.
3. **The causal check.** Force-rank at k=d_min−1 zeroes exact recovery
   in all 5 groups (0.000, including all 4 independent S3 seeds);
   recovery returns at k=d_min — d_min is load-bearing, not a
   correlate.
4. **A measurement lesson for convergence claims.** Two instrument
   defects (an isotropic uncentered-covariance lens; an ambient-identity
   target block taxing every rank budget) each produced plausible
   wrong readings before being caught against raw artifacts and fixed;
   the convergence law above is what survived.

## Per-section page budget

| Section | Pages | Purpose |
|---|---|---|
| 1 Introduction | 0.70 | convergence question; the groups-as-probes design; contribution bullets |
| 2 Task, model, instrument | 0.85 | word problem, encoder + P=1 bottleneck, restricted effective rank, degauged recovery, pre-registration |
| 3 Convergence to d_min | 0.90 | M1 table + headline figure; ρ analysis incl. tie-cap + exact null |
| 4 Equivalence at matched dimension | 0.60 | the marquee TOST; power design; what separation would have meant |
| 5 The causal check | 0.55 | razor table + support figure; S3 seed extension; D-AMB in brief |
| 6 Related work | 0.25 | by-name distinctions |
| 7 Limitations | 0.15 | scale, n, disclosures |
| **Total** | **4.0** | = assumed venue limit (flagged above) |

## Claims-to-evidence-to-figure map

| Claim id | Claim (with the number) | Verdict record (§ / log entry) | Raw artifact (path + md5) | Figure / table |
|---|---|---|---|---|
| U1 | M1 convergence: means 1.877±0.060 / 2.852±0.054 / 2.832±0.062 / 3.591±0.069 / 4.736±0.023 vs d_min 2/3/3/4/5; ρ=0.9747 = tie-capped max; exact-null P(ρ≥0.8)=8/120≈6.7%; 19/19 seeds in [0.7,1.3]·d_min | `CAPABILITY_SEPARATION_DESIGN.md` §1.33; `EXPERIMENT_LOG.md` 2026-07-09 sweep harvest | `experiment-runs/2026-07-09_capability_sweep_harvest/results/*__unconstrained__seed*.json` (19 files; md5s in `papers/unireps-ea/figures/figure_gen.py` SOURCE_MD5) + `harvest_summary.json` md5:7dce77dcba724cd1004419ac71fe5f2f | fig1_convergence.pdf + Table 1 |
| U2 | Marquee TOST: S4−A5 diff +0.0194, se 0.0368, df 7.83, t1/t2 = 13.06/14.12 vs tcrit 1.865, margin ±0.5, n=5 vs 5 → DECLARE | §1.33; design + power sim §1.4.2.1 | `harvest_summary.json` (marquee block) md5:7dce77dc… + the 10 S4/A5 unconstrained JSONs | §4 text + fig1 inset |
| U3 | Razor: k=d_min−1 xrec90 = 0.000 all 5 groups; k=d_min recovers (0.800/0.700/0.600/0.650 at S4/A5/S5/A6 vs 0.9×anchor bars 0.585/0.630/0.450/0.585; S3 seed-mean 0.5625 ≥ fixed 0.495 bar) | §1.36 + §1.36a | `experiment-runs/2026-07-09_m3fix_harvest/zero_pad__*.json` (20) + `experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__*.json` (12) (md5s in figure_gen.py) | fig2_razor_step.pdf + Table 2 |
| U4 | S3 necessity zero-noise: k=d_min−1 exactly 0.000 in all 4 independent seeds; per-seed k=d_min 0.450/0.550/0.600/0.650 | §1.36a | s3ext `zero_pad__S3__k_dmin*__seed{1,2,3}.json` + m3fix seed0 files | §5 text |
| U5 | L≥2 robustness: full-sample vs L≥2-only recovery near-identical per group (max \|Δ mean_cos\| 0.041 family-wide; per-group deltas ≤0.013) | §1.33 (schema deviation disclosed) | `harvest_analysis_output.txt` md5:854a4bd7c46e626badcc0fbf05d0e07a | §3 footnote |
| U6 | D-AMB tax (first sweep): 39 force-rank cells mean \|obs−√(k/d_state)\| = 0.028, max 0.166; first-sweep k≥d_min xrec90 = 0.000 everywhere | §1.33 (five-way diagnosis; INCONCLUSIVE registered) | `harvest_summary.json` md5:7dce77dc… + `harvest_analysis_output.txt` md5:854a4bd7… | §5 text |

U1–U3 and U5–U6 were independently recomputed from the raw JSONs during
this brief's preparation (Spearman enumeration 120-permutation exact
null included; Welch TOST re-derived; per-group means re-averaged) and
matched the verdict records exactly.

## Figures to generate

Single versioned script `papers/unireps-ea/figures/figure_gen.py`
(md5-asserting; colorblind-safe Okabe–Ito palette; loads only raw
JSONs):

- `fig1_convergence.pdf` (headline) — restricted effective rank vs
  d_min, all 19 per-seed points, [0.7,1.3]·d_min band, identity line;
  inset: S4 and A5 per-seed distributions at d_min=3 with the ±0.5
  TOST margin drawn. Takeaway: five groups converge onto the algebraic
  minimum, and the matched pair is indistinguishable.
- `fig2_razor_step.pdf` (support) — exact-recovery fraction at
  k ∈ {d_min−1, d_min, d_min+1} + anchor per group. Takeaway: the
  converged dimension is causally load-bearing (0.000 below d_min).

## Nearest prior work (distinguish by name)

- **Merrill et al., The Illusion of State (ICML 2024,
  arXiv:2404.08819):** frames state tracking by complexity class
  (solvable vs non-solvable word problems); the marquee result shows
  learned representational dimension does NOT sort by that divide but
  by d_min.
- **Grazzi et al. (ICLR 2025, arXiv:2411.12537) / Siems et al.
  DeltaProduct (NeurIPS 2025, arXiv:2502.10297):** expressivity of
  linear-RNN transition operators on exactly these group word problems
  (eigenvalue range, Householder count); neither measures what a
  trained state converges to, nor intervenes on it.
- **Nichani, Lee & Bietti (ICLR 2025, arXiv:2412.06538):** associative-
  memory rank capacity under argmax decoding; here decoding is exact
  continuous recovery, closing the loophole that would let a rank-1
  state fake convergence.
- **Nazari et al. (arXiv:2602.04852) / Sun et al. (arXiv:2602.02195):**
  observational state-rank dynamics in pretrained LLMs; no algebraic
  ground truth to converge to and no causal leg.
- **Mishra et al. M²RNN (arXiv:2603.14360):** trains matrix states on
  S3 composition; single group, no rank measurement, no
  dimension-vs-solvability contrast.

## Anonymization surface (anon build only)

Same token list as the sibling EA: `larson`, `samlarson`, `saml212`,
`pebble`, `pebbleml`, `rockie`, `github.com/`, `huggingface.co/`,
`.pebbleml.com`, `acknowledg`, `self-funded`, `funded by`. Expected
matches: zero. Anon code link: `https://anonymous.4open.science/`
placeholder.

## Project DO-NOT list

Identical to the sibling EA brief: no "audit" in prose; no GPU-hours /
dollars / fleet sizes / experiment-count bragging / funding language;
every number carries `% <!-- evidence: Ux -->` immediately after it.

## PI placeholders (deliberately left open)

- **Author block:** `Author Name(s) TBD`.
- **Title (two candidates):**
  - (A) "Dimension, Not Solvability: Trained Matrix States Converge to
    the Minimal Faithful Representation Dimension"
  - (B) "Five Groups, One Law: Learned State Rank Converges to the
    Algebraic Minimum Across the Solvable Divide"
- **Actual submission** — not performed here.
