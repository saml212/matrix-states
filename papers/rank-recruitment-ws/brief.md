# Paper brief — NeurReps 2026 Extended Abstract ("rank recruitment under teeth")

Stage 1 of the `paper` skill (repo mode). Base draft: the 13pp
`matrix-thinking/submissions/neurips-ws-2026/` Task D/E/frontier paper.
This EA executes the page-cut plan in
`matrix-thinking/submissions/PAPER_SPRINT_PLAN.md` §1 as **cut Strategy B**
("the whole arc, compressed uniformly" — the plan's own recommended
default): Task D, Task E, and the exactness frontier each compressed to
headline number + one figure, related work woven tight, the
readout-pinning discipline kept intact (the plan marks it do-not-cut).

## Venue

- **Name:** NeurIPS 2026 Workshop on Symmetry and Geometry in Neural
  Representations (NeurReps) — Extended Abstract track.
- **Format:** 4 pages excluding references and appendices;
  `\documentclass[mlabstract,onecolumn]{jmlr}` (official NeurReps style
  zip, vendored via `papers/neurreps-ea/`); single flattened `.tex` in
  `bundle/`. From `papers/rank-recruitment-ws/venue-requirements.md`
  (live-fetched 2026-07-10), NOT memory.
- **Requirements source:** `papers/rank-recruitment-ws/venue-requirements.md`.
  **FLAG: the live page serves the 2025 CFP; the 2026 CFP is not
  published (accepted-workshop list drops 2026-07-11). All format fields
  are 2025-pattern projections; re-verify before submission.**
- **Review style:** double-blind (OpenReview, >= 3 reviews) — triggers
  the anonymization grep. Review build carries no author block and the
  draftwatermark track stamp.
- **Archival:** EA track non-archival; verbatim "There are no
  restrictions on Extended Abstract submissions" (live-fetched
  2026-07-10) — dual-submission-safe for the ICLR 2027 flagship.
- **Deadline:** PROJECTED late August 2026 (2025: Aug 29 -> Sep 4
  extended, AoE); re-verify when the 2026 CFP lands.

## Thesis (one falsifiable sentence)

> When a task's exact solution provably requires $\mathrm{rank}(Z)\geq K$
> and every rank-evading shortcut is closed by construction — an exact
> continuous readout (never argmax), a single-matrix-state bottleneck
> (never position decomposition), and a budget-extension rule (never a
> premature "dead" verdict) — gradient descent trained from scratch
> recruits effective rank tracking $K$, makes it causally necessary
> (a train-time force-rank step at $k \approx K$), and composes it
> exactly through 21-fold self-application; a force-rank-$(K{-}1)$ model
> reaching $\geq 0.9\times$ ceiling recovery at multiple $(d,K)$ points
> would falsify the causal claim.

The falsifier is pre-registered in the base program
(`TASK_D_PREREGISTRATION.md` decision bands; the 13pp draft §7 states it
verbatim) and has not occurred: the one converged force-rank-$(K{-}1)$
composition seed shows the predicted depth-decay signature, not ceiling.

## Contribution bullets

1. **The teeth (methodology).** A provable $\mathrm{rank}(Z)\geq K$
   lower bound stays a necessity result only under exact continuous
   recovery ($\mathrm{pred}=Z\cdot k_j$, absolute cosine bar) and a hard
   $P{=}1$ state bottleneck verified by a gradient blank-out test —
   under argmax decoding a rank-$m$ memory stores $\approx md$
   associations (Nichani et al. 2025) and the bound silently collapses.
2. **Recruitment and causal necessity.** Learned effective rank tracks
   $K$ across the full tested grid at $d{=}16$ (Spearman $\rho{=}1.0$,
   10 grid points), and train-time force-rank shows a step at
   $k\approx K$ in all three tested $(d,K)$ cells (razor-sharp at
   $d{=}8,K{=}4$: $k\leq 3$ gives $\leq 0.0004$ recovery, $k{=}4$ gives
   0.97).
3. **Exact composition and its mechanism.** 4/5 unconstrained seeds
   recover at $\geq 0.9996$ recovered-fraction at every hop through
   21-fold self-application; an entity-subspace decomposition shows the
   trained operator restricted to the $K$-dimensional key subspace is a
   scaled $K$-cycle with restricted rank exactly $K$, while a
   dynamically invisible full-rank complement absorbs the unconstrained
   degrees of freedom.
4. **Depth as a spectral-exactness amplifier.** A rank-starved
   ($k{=}K{-}1$) operator passes shallow hops (cosine $\approx 0.93$)
   and collapses only under composition depth (recovered fraction
   $0.88\to 0.06$ from $h{=}7$ to $h{=}21$), with its entire measured
   cosine curve predicted from its own eigenspectrum (within 0.008
   through $h{=}7$, no raw keys).
5. **The exactness frontier + the budget rule.** Converged recovery
   degrades monotonically with $d$ at fixed encoder width (cosine
   $\approx 1.0$ at $d{=}16$ to $0.39$ at $d{=}64$), and three
   independent axes of the program each mislabeled a slow cell "dead"
   at its first budget — a named methodological rule with a
   counter-example that keeps it from being a license to always re-test.

## Per-section page budget

| Section | Pages | Purpose |
|---|---|---|
| 1 Introduction | 0.60 | the companion negative result (third person); the question; contributions |
| 2 The bound and the teeth | 0.90 | task; Fact 1; exact-readout pin (anti-argmax); P=1 bottleneck + blank-out; architecture + force-rank control |
| 3 Recruitment and causal necessity | 0.70 | M1 grid table; force-rank staircase figure |
| 4 Exact composition and the mechanism | 1.00 | depth-21 headline; depth-amplification figure + spectral prediction; subspace decomposition |
| 5 The exactness frontier and when to trust a dead cell | 0.35 | frontier numbers; the 2--2.5x rule + counter-caution |
| 6 Related work | 0.30 | by-name distinctions |
| 7 Limitations and outlook | 0.15 | scope; companion cross-pointer; falsification recap |
| **Total** | **4.00** | = projected venue limit (flagged above) |

Appendices (excluded from the 4pp count per the CFP): A. depth-21
periodicity under the single $K$-cycle; B. per-seed subspace table;
C. reproducibility and archive map.

## Claims-to-evidence-to-figure map

Every number independently recomputed from the raw artifacts during this
brief's preparation (2026-07-10): M1 grid + Spearman enumeration, all
M3 staircase cells, all Task E per-hop recovered fractions and the
spectral prediction (via `matrix-thinking/chapter2/analyze_zdump.py`),
whole-matrix vs restricted rank from the archived Z-dumps, all frontier
cells, and both K-wall rounds. All matched the base draft except where
noted (R3 outlier phrasing; R5 whole-matrix quantity — see rows).

| Claim id | Claim (with the number) | Verdict record (§ / log entry) | Raw artifact (path + md5) | Figure / table |
|---|---|---|---|---|
| R1 | M1: d=16 effective rank tracks K over the full grid K in {1,2,3,4,6,8,10,12,14,16}: 2.42/3.01/3.92/4.74/6.40/8.20/9.89/11.78/13.47/15.09; Spearman rho=1.0; only K=1,2 exceed the pre-registered [0.7K,1.3K] band; d=8 tracks to ceiling (K=8 -> 7.83); K>d saturates and declines (K=20 -> 11.1, 24 -> 11.6, 32 -> 8.6). CORRECTION (gauntlet r1 format-audit M3, 2026-07-10): K=3 reads 3.9180 vs band top 3.90 — marginally OUT on the ten-point grid ("only K=1,2" was exact on the pre-registered decision grid only); the paper states "only at K <= 3, by overshoot" | `TASK_D_PREREGISTRATION.md` (M1 bands); `EXPERIMENT_LOG.md` 2026-07-01 "Chapter 2 — Task D"; base draft §3.1 (991-run snapshot) | `matrix-thinking/chapter2/results/overnight_snapshots/AGGREGATE_latest.json` md5:c0a7d27e33a606d81e1babfc5d674edb (n_runs=991) | Table 1 |
| R1b | The 1,234-run replication is directionally identical (d=16: K=8 -> 8.198, K=16 -> 15.083) | base draft §3.1 SOURCE trace (2026-07-03 integrity pass) | `matrix-thinking/chapter2/results/overnight_snapshots/AGGREGATE_1234.json` md5:0134495e42e7549dd7d13a5753e69ce6 (n_runs=1234, n_failed=3) | text §3 |
| R2 | M3 staircase: d=8,K=4: k in {1,2,3} -> {0.0, 0.0004, 0.0}, k=4 -> 0.97 (0.9675); d=16,K=8: k in {1,2} -> 0.0, k=6 -> 0.34, k=7 -> 0.91, k=16 -> 0.99; d=16,K=12: k in {1,2} -> 0.0, k=12 -> 0.94, k=16 -> 0.9994 | `TASK_D_PREREGISTRATION.md` (M3 primary causal test, C1 control); `EXPERIMENT_LOG.md` 2026-07-01 Task D; base draft §3.2 | same file as R1 (M3_recovered_frac@0.9_vs_forcerank) | fig_forcerank.pdf |
| R2b | Replication refines the d=16,K=12 transition to a ragged ramp: k=11 -> 0.75, k=13 -> 0.79, k=14 -> 0.85 | base draft §3.2 caption SOURCE trace | same file as R1b (M3...["d16_K12"]) | text §3 |
| R3 | Task E headline: 4/5 unconstrained K=8 seeds reach recovered_frac@0.9 >= 0.9996 at every hop (h=1..7 and 21; three of the four are 1.000 at every hop); the fifth seed decays 0.93 -> 0.16 (h=1 -> 7) and 0.0001 at h=21. NOTE: the base draft's §4.1 phrased the outlier as "1.00 -> 0.16"; the archived per-hop values are 0.9268 (h=1) -> 0.1613 (h=7). This EA states the archive's numbers. | `matrix-thinking/chapter2/NEXT_EXPERIMENT_DESIGN.md` (pre-registration); `EXPERIMENT_LOG.md` 2026-07-02 "Task E 40K-step round" | `experiment-runs/2026-07-02_task_e_zdump/task_e_40k_zdump/t1_matrix_permutation_K8_frN_s{0..4}.json` md5s: 5ecaaeb8fe649f209cd41c35ba95c082 / 20b615ad2ae1b33c734a06d4a4b2210f / 7908556e99a475e489a237772df7eb02 / eb5ee74e494854856296dccb54b5f50c / 313eda0fe2c05fc9aa7e857da913a1c0 | fig_depth.pdf (right panel) |
| R4 | Depth amplification (force-rank k=7=K-1, the one converged seed): measured mean cosine 0.9303/0.9259/0.9212/0.9163/0.8206 at h=1/3/5/7/21 vs the eigenspectrum-only prediction 0.9317/0.9258/0.9181/0.9089/0.7536 (|pred-meas| <= 0.008 through h=7); recovered_frac@0.9 falls 0.996 -> 0.881 (h=7) -> 0.060 (h=21) | `EXPERIMENT_LOG.md` 2026-07-02 Task E Z-dump entry; base draft §4.2 (C8 eigenstructure metric, pre-registered) | `experiment-runs/2026-07-02_task_e_zdump/task_e_40k_zdump/t1_matrix_permutation_K8_fr7_s2.json` md5:ccfafb45699d022da8f93a49b9d4b793; prediction computed by `matrix-thinking/chapter2/analyze_zdump.py` (depth_curve_for_run) | fig_depth.pdf (left panel) + Table 2 |
| R5 | Invariant-subspace mechanism (4 converged K=8 seeds, means over 4 eval examples): restricted operator eff-rank 7.999--8.000 (max 8); scale-corrected residual vs the ideal cycle 0.7--2.4%; leakage \|\|C\|\| < 2.5% of \|\|Z\|\|; complement eff-rank ~8 (full), spectral radius >= 1. Whole-matrix effective rank reads 16.0 (full ambient d=16; stable rank 14.7--15.6) — the wrong instrument, resolved by subspace restriction. NOTE: the base draft §4.3 quoted "whole-matrix effective rank ~14.7--15.6"; the archived Z-dumps give effective rank 16.0 and STABLE rank 14.66--15.59. This EA reports both quantities explicitly. | `EXPERIMENT_LOG.md` 2026-07-02 "Task E Z-dump entity-subspace analysis"; base draft §4.3 | same four frN_s{1..4} files as R3 (Z_dump field), analyzed by `analyze_zdump.py`; per-example fields A_eff_rank / A_minus_cPi_rel / normC / normZ / D_eff_rank / D_spectral_radius | Appendix Table B + text §4 |
| R6 | K-wall budget reversal: K=12 0/5 converged at 40K steps -> 3/3 at 80K (recovered_frac@0.9 = 1.000 at every hop incl. 21); K=16: 2 of 3 fresh 80K seeds converged (s0 h21=0.996; s2 through h=7 at 0.9997 with the depth signature at h21=0.26), 120K extension rescued zero further seeds (3/3 dead) -> K=16 tally 2/5 | `EXPERIMENT_LOG.md` 2026-07-02 "Task E K-wall resolution — 80K round" and "K=16's stuck-seed completion wave"; base draft §4.4 | `experiment-runs/2026-07-02_task_e_80k_kwall/t1_matrix_permutation_K12_frN_s{0,1,2}_80k.json` md5s: db5bca1e4033053fe97e9d1eac8ce4f1 / 758b7a69ce65a155e62680ec8f1b19c0 / a3fef398e3fec0fe24dfc4eeffbb78b5; `..._K16_frN_s{0,1,2}_80k.json` md5s: bce85e4957c3fc262d57e485f33dd6c2 / 9e2a13d642bf9d9b74a3cca062c62638 / 8408a4d196775002b86a806e8946dec3; `experiment-runs/2026-07-02_task_e_120k_k16/t1_matrix_permutation_K16_frN_s{1,3,4}_120k.json` md5s: 1480a547badd919443224cdf162465b4 / 62612572d35557e7d97a7b3f34e13500 / 0d48b58c017a930043053cc209b1e5f4 | text §5 |
| R7 | Exactness frontier (K=d/4, fixed h=64 encoder, confirmed-flat trailing checkpoints): d=32 (100K steps, 3 seeds) cosine 0.877/0.909/0.915, recovered 0.413/0.632/0.653 (all below the pre-registered 0.7 bar); d=48 (100K, 1 seed) cosine 0.7196, recovered 0.002; d=64 (150K, 1 seed) cosine 0.3882, recovered 0.0. Rank-vs-exactness separation at d=32: effective_rank_mean 7.762/7.296/7.680 against target K=8 = 91--97% of target while exact recovery stalls below the bar | `matrix-thinking/chapter2/STAGE0_DESIGN.md` (formal bar recovered_frac@0.9 > 0.7, pre-registered); `EXPERIMENT_LOG.md` 2026-07-02/03 Stage 0 entries; base draft §5.2 | `experiment-runs/2026-07-02_stage0_waves/probe_100k/p100k_baseline_d32_K8_s{0,1,2}.json` md5s: 9a0ad4dfc786d18de0c4f259ea89126f / f8a1b3df36157c8d68ab3f6c30324fa7 / 8e5aa8e102a09a713037b4cd9c508fe9; `probe_d48_100k/p100k_baseline_d48_K12_s0.json` md5:c8d3ed060d4947ce57089896fd0fc8ca; `probe_d64_150k/p150k_baseline_d64_K16_s0.json` md5:ab1b2c6e8eaccb89d232e3e4707193a6 | Table 3 |
| R8 | Three-budget-artifacts pattern: (i) Task E K=8: 1/5 seeds converged at 20K steps (the one at 1.00 -> 0.77 by h=7) vs 4/5 exact at 40K (=R3); (ii) Stage 0 d>=32: the 8K-step "wall" (rank ~1, recovery ~0) reversed — every baseline seed transitions at 20K, onset 6--16K; (iii) Task E K in {12,16} dead at 40K -> alive at 80K (=R6). Counter-caution: d=32's 100K plateaus are flat and still fail the 0.7 bar (=R7) | `EXPERIMENT_LOG.md` 2026-07-01 "Task E 20K-step round"; 2026-07-02 "Stage 0 — Wave -1/0 results"; 2026-07-03 "100K-step probe: formal pass criterion FAILS"; base draft §5.1 | (i) `experiment-runs/2026-07-01_task_e_20k/task_e_sweep/t1_matrix_permutation_K8_frN_s1.json` md5:76221743ff911c3bb603b73f74d16d78 (the converged-at-20K seed) + R3's files; (ii) `experiment-runs/2026-07-02_stage0_waves/wave0/AGGREGATE.json` md5:f23a536d025b6f1618939cb225325fa3; (iii) = R6 files | text §5 |
| R9 | Model scale: the d=16 composition model has 170,896 parameters (~171K); every model in the paper is under 1M parameters | base draft §2.5/§7 scope statements | `n_params` field of the R3/R4 archived JSONs (170896) | text §2/§7 |
| R10 | Bottleneck verification is behavioral, not a shape check: the decoder is a pure function of (Z, query); after encoding, the raw-input cache is corrupted and the decode path is confirmed bit-for-bit unchanged (blank-out test) | `TASK_D_PREREGISTRATION.md` (controls); base draft §2.4; `EXPERIMENT_LOG.md` 2026-07-01 Task D (smoke gate) | qualitative control (no plotted number); implementation in `matrix-thinking/chapter2/model_v4.py` / `run_task_d.py` smoke gate | text §2 |

## Figures to generate

Single versioned script `papers/rank-recruitment-ws/figures/figure_gen.py`
(md5-asserting per the skill template; Okabe--Ito colorblind-safe palette;
loads only the raw artifacts named above; the Task E spectral prediction is
computed by importing `analyze_zdump.py`, never re-derived by hand):

- `fig_forcerank.pdf` — R2: recovered_frac@0.9 under train-time
  force-rank k, three panels (d=8 K=4; d=16 K=8; d=16 K=12), dashed line
  at k=K. Takeaway: constraining rank below K destroys recovery; the
  step lands at k ~ K in every cell.
- `fig_depth.pdf` — R4 + R3: left, the rank-starved (k=K-1) operator's
  measured depth-decay cosine vs the prediction from its own
  eigenspectrum; right, recovered_frac@0.9 vs hop for rank-sufficient
  seeds (flat at 1.0 through h=21), the rank-starved operator
  (collapses at h=21), and the still-transitioning outlier seed.
  Takeaway: composition depth is a sharper rank probe than any
  single-hop tolerance.

Both figures differ from the base draft's committed figures only in
regeneration provenance (md5-asserted loader in this tree); neither
appears in the sibling `papers/neurreps-ea/` submission.

## Companion papers and overlap management

- **The published negative result** (ICML 2026 MI workshop, "The
  Gradient Does Not See Rank") is this paper's premise; cited in third
  person with its real bibliography entry (documented double-blind
  exception; see venue-requirements.md).
- **`papers/neurreps-ea/` (same venue, different EA):** headline = the
  group-composition causal rank law (d_min tracking, TOST equivalence,
  the five-group razor). Zero shared figures or tables with this paper.
  Its §"binding foundation" paragraph and abstract used to restate four
  numbers from the same Task D/E archives this paper is the primary
  carrier of (its N6/N9 rows quote the 1,234-run replication values
  8.198/15.083/<=0.0004/0.940 and the 4/5-seed depth-21 sentence); the
  final review (`gauntlet/round-1/07_final_review.md` §4, ESCALATION-1)
  found this substantive-at-the-surface and ordered de-dup edits S1-S3
  on the sibling, applied 2026-07-11 (see
  `../neurreps-ea/gauntlet/round-2/07_decision_record.md`): S1 replaced
  both restatements with a pointer to this paper as a companion
  submission (numbers dropped), S2 cut its parallel periodicity
  appendix to a pointer at this paper's Appendix A, and S3 added a
  mirror companion-disclosure to its Limitations (it previously had
  none — the final review's finding, now fixed). This paper reports the
  full grids from the 991-run pre-registered snapshot (0.9675 at the
  d=8,K=4 step; the full staircase and per-hop tables), reports the
  replication as a robustness note, and does NOT state any group-task
  claim (no d_min, no TOST, no five-group razor). The sibling
  relationship is now disclosed in both briefs and in each paper's
  related-work/limitations pointer, phrased anonymously
  ("a concurrent companion submission").
- **The 13pp base draft** remains the arXiv/source version; it is not
  submitted to this venue.

## Nearest prior work (distinguish by name)

- **Nichani, Lee & Bietti (ICLR 2025, arXiv:2412.06538):** hand-built
  rank-m argmax-decoding existence construction in full attention; this
  work pins the readout to exact continuous recovery precisely so that
  construction cannot satisfy the bound, then measures what SGD
  recruits and intervenes on rank at train time.
- **Barnfield et al. (arXiv, 2026):** asymptotically exact
  argmax-vs-listwise capacity thresholds for a static rank-constrained
  memory — an analytic reference, not a trained, composed, or causally
  ablated object.
- **Nazari et al. / Sun et al. (arXiv 2602.04852 / 2602.02195):**
  descriptive effective-rank dynamics and an upper bound
  rank(S_t) <= t in pretrained linear-attention LLMs; here the write
  count K is controlled, the bound is a lower bound, and the necessity
  direction is causal (train-time force-rank).
- **Mishra et al. M2RNN (arXiv:2603.14360):** matrix-state RNN with
  compose-beyond-horizon generalization on S3; their own vector-GRU
  control matches, no rank measurement, no causal intervention — this
  paper makes rank the measured and manipulated quantity under a
  provable per-task bound.
- **Grazzi et al. (ICLR 2025) / Siems et al. (NeurIPS 2025):**
  expressivity of the per-token transition operator (eigenvalue range /
  Householder count); this paper measures the trained state's own rank
  — a different object, and the source of a standing reviewer
  confusion the text names explicitly.
- **Zhu et al. / Gozeten et al. (2025):** existence constructions for
  continuous-CoT superposition bounded by embedding dimension; no
  measurement of what a trained objective recruits.
- **Dziri et al. (2023) / Wang et al. (2024):** the standing caution on
  compositional-generalization claims; this paper's composition claim
  is deliberately narrower — exact eigenstructure against a known
  analytic target, not held-out accuracy alone.

## Anonymization surface (review build)

Tokens: `larson`, `samlarson`, `saml212`, `pebble`, `pebbleml`,
`rockie`, `github.com/`, `huggingface.co/`, `.pebbleml.com`,
`acknowledg`, `self-funded`, `funded by`. Expected matches in the
review-build source closure: zero, EXCEPT the documented third-person
citation exception (the `larson2026gradient` refs.bib entry and its
rendered citation; see venue-requirements.md). Code link in the review
build: `https://anonymous.4open.science/` placeholder (swap for a real
anonymized snapshot before submission if time allows; otherwise an
explicit camera-ready deferral, same convention as the sibling EA).

## Project DO-NOT list (carried from the house reference brief)

- No "audit" as a word in the prose (use "independent review" /
  "adversarial check").
- No GPU-hour, dollar, or fleet-size mentions; no experiment-count
  bragging; no funding language.
- Every number carries a `% <!-- evidence: Rx -->` comment in the tex
  immediately after the sentence (or table/caption) that states it.
- Banned-word list and no-contractions rule per the skill styleguide.

## PI placeholders (deliberately left open)

- **Author block:** none in the review build (per template); camera-ready
  placeholder commented in `main.tex` (same open item as every sibling).
- **Title (default per PAPER_SPRINT_PLAN.md §4 Decision 2, PI-backed):**
  "When the Gradient Sees Rank: Provable Necessity, Causal Recruitment,
  and Exact Composition in Trained Matrix Memories". Alternates (B)/(D)
  from the base draft's title comment remain available.
- **Actual submission** (OpenReview portal, final 2026-CFP compliance
  check) — not performed here.
