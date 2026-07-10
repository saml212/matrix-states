# Paper brief — NeurReps 2026 Extended Abstract ("the causal rank law")

Stage 0 of the `paper` skill (repo mode). Base draft: the pre-built
`matrix-thinking/submissions/neurips-ws-2026/` Task D/E/frontier paper
(13pp), whose own abstract flags the group-composition causal force-rank
result as "still in flight." That result is now banked
(`CAPABILITY_SEPARATION_DESIGN.md` §1.36/§1.36a) and is this EA's
headline. This EA is the ~60% cut VENUE_DECISION.md §"Page-budget
reality check" called for, executed as cut Strategy A (Task D kept as
the foundation leg, Task E demoted to one sentence, frontier cut) plus
the new Stage-1 rank-law trilogy as the story.

## Venue

- **Name:** NeurIPS 2026 Workshop on Symmetry and Geometry in Neural
  Representations (NeurReps) — Extended Abstract track.
- **Format:** 4 pages excluding references/appendix, per the 2025 CFP
  language recorded in `VENUE_DECISION.md`. **FLAG: the 2026 CFP is not
  yet published (accepted-workshop list drops 2026-07-11); 4pp+refs is
  the working assumption and must be re-checked against the live CFP
  before submission.** No official 2026 style file yet; the house
  two-column scaffold is used pending the real kit (same convention as
  every sibling submission).
- **Review style:** unknown until the CFP; both builds prepared
  (`main.pdf` single-blind placeholder-author, `main-anon.pdf`
  double-blind). The anonymization grep gates the anon build.
- **Archival:** non-archival per 2025 CFP ("no restrictions on
  submissions appearing elsewhere") — dual-submission-safe for the
  ICLR 2027 flagship. Re-verify on the live CFP.
- **Deadline:** CFP expected shortly after 2026-07-11; historically
  late August (2025: Aug 29 → Sep 4 extended).

## Thesis (one falsifiable sentence)

> Gradient descent on a hard single-matrix-state bottleneck recruits
> exactly the effective rank the task's algebra demands — rank K on
> K-pair associative binding (provably necessary), and the group's
> minimal faithful real representation dimension d_min on
> group-composition state tracking — and d_min is causal: capping rank
> at d_min−1 drives exact recovery to zero in every group while
> restoring rank d_min brings it back, so a single below-d_min cell
> recovering (or recovery failing to return at d_min) falsifies the law.

The pre-registered falsifier existed before the data: HARD FALSIFY if
`k=d_min−1` reaches ≥0.9× ceiling at any group (§1.5 of the design doc);
the fix-wave grid ran against that criterion.

## Contribution bullets

1. **The causal razor (new; the headline).** Train-time force-rank at
   k=d_min−1 gives exact-recovery fraction 0.000 in all 5 groups (and in
   all 4 independent S3 seeds); recovery returns at k=d_min (≥0.9× the
   unconstrained anchor in S4/A5/S5/A6; S3 confirmed at seed-mean 0.5625
   vs the fixed 0.495 bar) and holds at k=d_min+1 — a step function at
   the minimal faithful representation dimension, 5/5 groups.
2. **The rank law, observed.** Restricted effective rank of the
   unconstrained state tracks d_min across S3/S4/A5/S5/A6 (means
   1.88/2.85/2.83/3.59/4.74 vs d_min 2/3/3/4/5); Spearman ρ=0.9747, the
   exact tie-capped maximum for this design; 19/19 seeds inside the
   pre-registered [0.7,1.3]·d_min band; the matched-dimension
   solvable/non-solvable pair (S4 vs A5) is TOST-equivalent within ±0.5
   rank-units (|Δ|=0.019, both one-sided t ≈ 7× critical).
3. **The foundation (from the base draft).** On K-pair associative
   binding under the same bottleneck and exact-continuous-recovery
   discipline, rank(Z) ≥ K provably, SGD recruits effective rank ≈ K
   (d=16: K=8→8.20, K=16→15.08), and force-rank confirms causality
   (d=8, K=4: rank ≤3 → 0.000, rank 4 → 0.940).
4. **Instrument honesty as methodology.** The first sweep's force-rank
   arms silently trained against a rank-(d_min+2) target (an ambient
   identity block); the cells sat at their rank-constrained optimum
   (mean |Δ|=0.028 from the √(k/d_state) ceiling over all 39 force-rank
   cells) and the verdict was registered INCONCLUSIVE, not spun; a
   zero-padded fix wave then landed the pre-registered prediction.

## Per-section page budget

| Section | Pages | Purpose |
|---|---|---|
| 1 Introduction | 0.65 | rank as the geometric currency of matrix memories; the question; contribution bullets |
| 2 Tasks, models, instrument | 0.85 | binding task + group word task; encoder; P=1 bottleneck; exact continuous recovery; restricted effective rank; degauging + C1 crosscheck pin |
| 3 Provable and recruited rank (binding) | 0.50 | Fact 1 (rank ≥ K); recruited rank tracks K; the d=8,K=4 causal step |
| 4 The rank law, observed (groups) | 0.50 | M1 table + ρ=0.9747; S4-vs-A5 TOST |
| 5 The causal razor | 1.00 | the C1 decisional table + figure; S3 seed extension; the D-AMB tax + fix-wave narrative |
| 6 Related work | 0.30 | by-name distinctions (nichani2025factual, nazari2026rank/sun2026staterank, mishra2026m2rnn, grazzi/siems, merrill2024illusion) |
| 7 Limitations and outlook | 0.20 | scale, n=1 fix-wave cells, soft-convergence disclosures, Stage-2 pointer |
| **Total** | **4.0** | = assumed venue limit (flagged above) |

## Claims-to-evidence-to-figure map

| Claim id | Claim (with the number) | Verdict record (§ / log entry) | Raw artifact (path + md5) | Figure / table |
|---|---|---|---|---|
| N1 | Razor step: xrec90 at k=d_min−1 → 0.000 all 5 groups; k=d_min → 0.450/0.800/0.700/0.600/0.650 (S3/S4/A5/S5/A6); k=d_min+1 → 0.550/0.950/0.750/0.550/0.700; anchors 0.550/0.650/0.700/0.500/0.650 | `CAPABILITY_SEPARATION_DESIGN.md` §1.36; `EXPERIMENT_LOG.md` 2026-07-09 m3fix harvest | `experiment-runs/2026-07-09_m3fix_harvest/zero_pad__*.json` (20 files; md5s in `papers/neurreps-ea/figures/figure_gen.py` SOURCE_MD5) | Table 2 + fig1_razor_step.pdf |
| N2 | Necessity leg zero-noise: k=d_min−1 xrec90 exactly 0.000 in all 4 independent S3 seeds | §1.36a | `experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__k_dmin_minus_1__seed{1,2,3}.json` + N1's seed0 file (md5s in figure_gen.py) | fig1 (S3 panel) |
| N3 | S3 sufficiency at seed-mean: k=d_min xrec90 seed-mean 0.5625 ≥ fixed bar 0.495 (per-seed 0.450/0.550/0.600/0.650) | §1.36a (pre-stated ±0.05 trigger + fixed-literal bar) | `experiment-runs/2026-07-09_m3fix_s3ext/zero_pad__S3__k_dmin__seed{1,2,3}.json` + seed0 file | text §5 + fig1 |
| N4 | M1: restricted eff. rank means 1.877±0.060 / 2.852±0.054 / 2.832±0.062 / 3.591±0.069 / 4.736±0.023 vs d_min 2/3/3/4/5; ρ=0.9747 (tie-capped max; exact-null P(ρ≥0.8)=8/120≈6.7%); 19/19 in band | §1.33; `EXPERIMENT_LOG.md` 2026-07-09 sweep harvest | `experiment-runs/2026-07-09_capability_sweep_harvest/results/*__unconstrained__seed*.json` (19 files) + `harvest_summary.json` md5:7dce77dcba724cd1004419ac71fe5f2f | fig2_rank_tracking.pdf + Table 1 |
| N5 | Marquee TOST: S4−A5 diff +0.0194 rank-units, se 0.0368, df 7.83, TOST t 13.06/14.12 vs tcrit 1.865, margin ±0.5 → DECLARE | §1.33 (design §1.4.2.1: n=5, margin, power sim) | `harvest_summary.json` (marquee block) md5:7dce77dc… + the 10 S4/A5 unconstrained JSONs | text §4 + fig2 inset |
| N6 | Binding foundation: d=16 eff-rank K=8→8.198, K=16→15.083; d=8,K=4 force-rank ≤3→ ≤0.0004, 4→0.940 | base draft §3 SOURCE traces (2026-07-03 integrity pass); `EXPERIMENT_LOG.md` Task D entries | `matrix-thinking/chapter2/results/overnight_snapshots/AGGREGATE_1234.json` md5:0134495e42e7549dd7d13a5753e69ce6 | text §3 |
| N7 | D-AMB: 39 force-rank cells mean \|obs−√(k/d_state)\|=0.028 (max 0.166; only the two S5 below-cells outliers); first-sweep M3 xrec90 0.000 at k≥d_min everywhere | §1.33 (five-way diagnosis) | `harvest_summary.json` (d_ambient fields, m3 block) md5:7dce77dc… + `harvest_analysis_output.txt` md5:854a4bd7c46e626badcc0fbf05d0e07a | text §5 |
| N8 | Fix-wave config integrity: 30/30 cells config-matched vs an independent-literal manifest; steps at Rev-7 pins; zero skipped steps | §1.36 (A3 discharge) | `experiment-runs/2026-07-09_m3fix_harvest/harvest_analysis_output.txt` md5:77be9c3b092c70e83ff08a0261575815 | reproducibility note |
| N9 | Composition: unconstrained K=8 operator reaches rec@0.9 = 1.0 at every held-out hop incl. depth 21 in 4/5 seeds (seed 0 the still-transitioning outlier, 0.0 at h=21) | base draft §4 (Task E); `EXPERIMENT_LOG.md` 2026-07-02 Task E entries | `experiment-runs/2026-07-02_task_e_40k/task_e_40k/t1_matrix_permutation_K8_frN_s{0..4}.json` md5s: f61f18b6…/5e70bb74…/1d54d016…/b80c6424…/02c0920b… (full values in reproducibility README) | text §3 |
| N10 | L≥2 robustness split: per-group mean_cos deltas ≤0.013 (max per-seed \|Δ\| 0.041), no group-selective divergence | §1.33 (disclosed schema deviation) | the 19 unconstrained JSONs (N4 row; `l_ge2_*` fields) + `harvest_analysis_output.txt` md5:854a4bd7c46e626badcc0fbf05d0e07a | §4 text |
| N11 | Centering defect: uncentered lens scores a flawless synthetic model 0.705261 on identical data the centered production lens scores 0.999594 | design doc §1.25 (Defect 1) + gate-1(b) recheck (`EXPERIMENT_LOG.md` 2026-07-09 SWEEP-READY entry) | `experiment-runs/2026-07-09_capability_calib_recheck/gate1b_recheck.txt` md5:2d170cc03011cc56105adeae9929e481 | §2 text |

Numbers N1–N5, N7, N9 and N10 were independently recomputed from the raw JSONs
during this brief's preparation (Spearman enumeration, Welch TOST, and
per-cell means re-derived; all matched the verdict records exactly).

## Figures to generate

Single versioned script `papers/neurreps-ea/figures/figure_gen.py`
(md5-asserting per the skill template; colorblind-safe Okabe–Ito
palette; loads only the raw JSONs named above):

- `fig1_razor_step.pdf` — per-group exact-recovery fraction
  (crosscheck rec@0.9) at k ∈ {d_min−1, d_min, d_min+1} with the
  unconstrained anchor and the 0.9×anchor bar; S3 shows all four seeds
  individually. Takeaway: a step function at d_min, floor exactly 0.000
  below it, in every group.
- `fig2_rank_tracking.pdf` — restricted effective rank (19 per-seed
  points) vs d_min with the [0.7,1.3]·d_min band and the identity line;
  S4/A5 visually coincident at d_min=3. Takeaway: the observational
  rank law and the dimension-not-solvability equivalence.

## Nearest prior work (distinguish by name)

- **Nichani, Lee & Bietti (ICLR 2025, arXiv:2412.06538):** hand-built
  rank-m argmax-decoding capacity construction in full attention; this
  work measures what SGD recruits under exact continuous recovery and a
  hard bottleneck, and intervenes on rank causally.
- **Nazari et al. (arXiv:2602.04852) / Sun et al. (arXiv:2602.02195):**
  descriptive effective-rank upper bounds in pretrained linear-attention
  LLMs; here the required rank is controlled by the task's group algebra
  and the necessity direction is causal (force-rank), not observational.
- **Mishra et al. M²RNN (arXiv:2603.14360):** matrix-state model on S3
  permutation composition with a representational theorem; no rank
  measurement, no causal rank intervention, and their own vector GRU
  matches — this work makes rank itself the measured and manipulated
  quantity across five groups spanning the solvable divide.
- **Grazzi et al. (ICLR 2025, arXiv:2411.12537) / Siems et al.
  (NeurIPS 2025, arXiv:2502.10297):** eigenvalue-range/Householder-count
  expressivity of the per-token transition operator; this work measures
  the trained state's own recruited rank against d_min.
- **Merrill et al. (ICML 2024, arXiv:2404.08819):** communication-
  complexity bounds on SSM state tracking (the solvability axis); the
  marquee TOST shows the learned geometry sorts by representation
  dimension, not by that complexity divide.

## Anonymization surface (anon build only)

Tokens: `larson`, `samlarson`, `saml212`, `pebble`, `pebbleml`,
`rockie`, `github.com/`, `huggingface.co/`, `.pebbleml.com`,
`acknowledg`, `self-funded`, `funded by`. Expected matches in
`main-anon.pdf` source closure: zero. Code link in anon build:
`https://anonymous.4open.science/` placeholder.

## Project DO-NOT list (carried from the house reference brief)

- No "audit" as a word in the prose (rename to "independent review" /
  "adversarial check").
- No GPU-hour, dollar, or fleet-size mentions; no experiment-count
  bragging; no funding language.
- Every number carries a `% <!-- evidence: Nx -->` comment in the tex
  immediately after the sentence that states it.

## PI placeholders (deliberately left open)

- **Author block:** `Author Name(s) TBD` (both builds; same open item
  as every sibling submission).
- **Title (two candidates):**
  - (A) "The Rank the Task Demands: A Causal Rank Law for Matrix
    Memories Trained on Group Composition"
  - (B) "Minimal Faithful Dimension Is What Gradient Descent Buys:
    Necessity and Sufficiency of d_min in Trained Matrix States"
- **Actual submission** (venue portal, final CFP compliance check) —
  not performed here.
