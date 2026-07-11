# Paper outline — NeurReps 2026 EA ("rank recruitment under teeth")

Stage 2 of the `paper` skill (repo mode). Budgets from `brief.md`; venue
limit 4pp excluding references/appendices (projected, flagged).

## Section plan

| # | Section (file) | Pages | Claims (ids) | Figures / tables | Notes |
|---|---|---|---|---|---|
| 1 | Introduction (`01_intro.tex`) | 0.60 | (frames R1--R8) | — | the published negative result, third person; the question; five contribution bullets |
| 2 | The bound and the teeth (`02_setup.tex`) | 0.90 | R9, R10 | — | task; Fact 1 (rank >= K); exact-readout pin vs argmax; P=1 bottleneck + blank-out; architecture + pre-registered controls |
| 3 | Recruitment and causal necessity (`03_recruitment.tex`) | 0.70 | R1, R1b, R2, R2b | Table 1 (M1 grid); fig_forcerank.pdf | observation (M1) -> causal test (M3) -> resolution of the companion's open confound |
| 4 | Exact composition and the mechanism (`04_composition.tex`) | 1.00 | R3, R4, R5 | fig_depth.pdf | headline depth-21; spectral prediction; subspace decomposition (per-seed table -> Appendix B). *Amended at drafting: the depth-curve table moved to Appendix A (the figure's left panel already carries it in the body; holds the 4pp budget).* |
| 5 | The exactness frontier and when to trust a dead cell (`05_frontier.tex`) | 0.35 | R6, R7, R8 | Table 3 (frontier) | K-wall reversal; frontier numbers; the 2--2.5x rule + counter-caution |
| 6 | Related work (`06_related.tex`) | 0.30 | — | — | seven by-name distinctions from the brief |
| 7 | Limitations and outlook (`07_limitations.tex`) | 0.15 | R9 | — | scope; falsification recap; anonymous companion pointer |
|   | Appendix A: depth-21 periodicity (`07_limitations.tex`, after \appendix) | excl. | R3, R4 | — | pi^21 = pi^5 under the single 8-cycle; what depth 21 does and does not test |
|   | Appendix B: per-seed subspace decomposition | excl. | R5 | Table B | four converged seeds, all block quantities |
|   | Appendix C: reproducibility and archive map | excl. | all | — | code pointer (anonymized), archive paths, md5 convention |
|   | **Total (body)** | **4.00** | | | = projected venue limit |

## Outline sanity checks

- [x] Page budgets sum to 4.00 (references and appendices excluded per CFP).
- [x] Every claim id in the brief's map appears in exactly one body section
      (R3/R4/R5 in §4 with appendix restatements; R6/R7/R8 in §5; R9 in §2
      scope sentence + §7; R10 in §2).
- [x] Both figures placed (§3, §4); three body tables placed (§3, §4, §5).
- [x] Related work distinguishes each neighbor BY NAME (seven, from brief).
- [x] No section carries a claim without an evidence row.
- [x] Zero overlap with the sibling EA's figures/tables/headline claims
      (checked against `papers/neurreps-ea/` sections and brief).

## Drafting amendments (2026-07-10, page-fit pass)

The first compiled draft ran ~5.5pp body against the 4pp limit; the
following structural amendments (all content-preserving at the claim
level, recorded per the repo-mode discipline) brought the body to
exactly 4pp:

- Table 1 (M1 grid) folded into §3 prose (grid endpoints + rho quoted;
  full grid remains in evidence row R1).
- Table 3 (frontier) folded into §5 prose (all cell values quoted).
- §6 Related work and §7 Limitations merged into one section
  ("Related Work, Limitations, and Outlook"); appendices unchanged.
- Intro contribution list compressed from five roman items to one
  sentence; the claim-level numbers moved wholly into §§3--5.
- The depth-curve table moved to Appendix A (as previously noted).
- Both figures regenerated at onecolumn-native size (5.6in wide).

## Per-section beat sheet

### 1. Introduction
- Matrix-valued latents make rank a structural observable; the published
  ICML-workshop negative result (third person): bolt-on matrix-CODI is
  rank-indifferent, but ProsQA is rank-1-solvable, so the confound stood.
- The question: with the solution provably rank-K and every shortcut
  closed, does SGD (a) recruit rank ~K and (b) need it causally?
- Why answering requires teeth: argmax tolerance, position
  decomposition, premature dead-cell verdicts.
- Contribution bullets (five, tied to R-ids).

### 2. The bound and the teeth
- Task D grammar (BIND x K, QUERY); fresh keys/values per sample.
- Fact 1: rank(Z) >= K for exact recovery; classical, not claimed novel.
- Teeth 1: exact continuous readout, absolute cosine bar; the Nichani
  rank-m -> md argmax construction and why it voids the bound.
- Teeth 2: P=1 single-state bottleneck; decoder reads only (Z, query);
  gradient blank-out verification (R10).
- Architecture (row-reader encoder, ~171K params, R9); pre-registered
  controls incl. train-time force-rank (C1) and decision bands.

### 3. Recruitment and causal necessity
- M1 table (R1): full grid, rho=1.0; band deviations named (K=1,2);
  d=8 ceiling; K>d decline; replication note (R1b).
- M3 staircase (R2, figure): step at k~K in all three cells;
  razor-sharp d=8,K=4; ragged d=16,K=12 ramp disclosed (R2b).
- Resolution: the earlier rank-blindness was task-rank-solvability,
  not a property of the gradient.

### 4. Exact composition and the mechanism
- Task E grammar: single Hamiltonian K-cycle (why general permutations
  collapse held-out hops -- one sentence, appendix A pointer).
- Headline (R3): 4/5 seeds >= 0.9996 at every hop through 21-fold
  self-application; outlier's decay signature stated from the archive.
- Depth amplification (R4, figure + small table): k=K-1 passes shallow,
  collapses at depth; the eigenspectrum-only prediction.
- Mechanism (R5): whole-matrix rank is the wrong instrument (16.0 full
  ambient; stable 14.7--15.6); restricted-to-subspace rank exactly K;
  leakage < 2.5%; inert full-rank complement; dead seeds fail the same
  test (collapse is real).

### 5. The exactness frontier and when to trust a dead cell
- K-wall reversal (R6): K=12 0/5@40K -> 3/3@80K; K=16 2/5 with one
  budget-immune stuck seed (the honest boundary case).
- Frontier (R7, table): monotone cosine decline 1.0 -> 0.39; recovered
  fraction 0 at d=64; all cells confirmed-flat before being read.
- The rule (R8): dead requires re-test at 2--2.5x budget; necessary,
  not sufficient (the d=32 plateau fails the bar at 100K).

### 6. Related work
- Seven by-name distinctions from the brief (Nichani; Barnfield;
  Nazari/Sun; Mishra M2RNN; Grazzi/Siems transition-vs-state rank;
  Zhu/Gozeten; Dziri/Wang caution). TPR/VSA lineage one sentence.

### 7. Limitations and outlook
- Synthetic, <= 1M params; why the mechanism claims still bind
  (provable ground truth) and what scale still costs (generality).
- Falsification conditions restated in one sentence each.
- Anonymous pointer to the concurrent companion submission
  (group-composition rank law) as the representation-theoretic
  extension of the same instrument.
