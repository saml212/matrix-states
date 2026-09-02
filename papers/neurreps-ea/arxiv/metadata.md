# arXiv metadata — NeurReps EA named build (v1)

Package: `neurreps-ea-arxiv-v1.zip` (tex + bbl + jmlr.cls/jmlrutils.sty +
refs.bib + 2 figure PDFs; arXiv uses the included `.bbl`, it never runs
BibTeX). Local build: 3× tectonic passes, 8 pages, matches the review
build's pagination; render-inspected page-by-page 2026-07-10.

**REBUILD (2026-07-11):** regenerated from the review bundle after the
final-review de-dup edits landed (`gauntlet/round-1/07_final_review.md`
S1-S3 on the companion `papers/rank-recruitment-ws/`: abstract + §2
binding-numbers restatement replaced with a pointer, Appendix E cut to a
pointer, a mirror companion disclosure added to Limitations). Same
two-change de-anonymization pattern (watermark removed, real author
block added); still 8 pages, render-inspected clean, anonymization grep
clean pre-de-anonymization. Abstract field below updated to match.

## Title

The Rank the Task Demands: A Causal Rank Law for Matrix Memories Trained on Group Composition

## Authors

Sam Larson (Pebble AI)

Corresponding: samlarson16@gmail.com

## Abstract (paste into the arXiv abstract field)

Matrix-valued memories make rank the natural budget of a learned representation: the number of independent directions a state spans bounds what it can bind, compose, and track. We report causal evidence, on a group-composition testbed trained under a hard single-state bottleneck with a fixed decoder that cannot launder rank, that gradient descent recruits precisely the rank the task's algebra demands. A concurrent companion submission establishes the analogous recruitment and causal necessity pattern on a $K$-pair associative-binding testbed, where exact recovery provably requires state rank at least $K$; this paper inherits that instrument and extends the rank law from a scalar capacity bound to a representation-theoretic one. On group-composition state tracking over five finite groups spanning the solvable/non-solvable divide, the recruited rank equals the group's minimal faithful real representation dimension $d_{\min}$ (Spearman $\rho = 0.9747$, the design's tie-capped maximum), the dimension-matched solvable/non-solvable pair $S_4$/$A_5$ is statistically equivalent under a pre-registered test, and a pre-registered force-rank razor is exact in both directions: one rank below $d_{\min}$, recovery is capped by the target's tied unit spectrum at $\sqrt{(d_{\min}-1)/d_{\min}} \le 0.894$, below the $0.9$ threshold in every group by construction, with observed cells at 76-95% (mean 88%) of that ceiling; at $d_{\min}$, not guaranteed a priori, recovery clears the anchor-relative bar in all five groups. Representation theory, not computational complexity class, sets what gradient descent buys.

## Categories

- Primary: cs.LG (Machine Learning)
- Secondary: cs.NE (Neural and Evolutionary Computing)

## License

CC BY 4.0 (recommended).

## Comments field (suggested)

> 8 pages (4-page extended abstract plus references and appendices), 2
> figures. Companion paper: "When the Gradient Sees Rank: Provable
> Necessity, Causal Recruitment, and Exact Composition in Trained Matrix
> Memories"

After BOTH arXiv IDs exist, an optional v2 can add each paper's arXiv ID
to the other's comments field (not blocking).

## Notes

- The gauntleted body text is byte-identical to the pinned anonymized
  review bundle (`bundle/neurreps-ea-submission.tex`); the only changes
  are the removed review watermark and the real author block.
- The Related-Work self-citation to the ICML 2026 MI-workshop paper was
  cut for double-blind review (see `../brief.md`, anonymization section);
  it is NOT restored here to keep the gauntleted text frozen — restore it
  at workshop camera-ready and optionally in an arXiv v2.
- Expected arXiv-preview page count: 8. The source has a known aux
  fixed-point transient (12pp on a cold single pass); arXiv's AutoTeX
  runs multiple passes, but verify the preview before announcing.

## v2 named merged build (2026-09-02)

Source: `../arxiv-v2/` (modular: `main.tex` + `sections/01..08`, `refs.bib`,
`main.bbl`, `neurips.sty`, three figure PDFs). Built locally with
`pdflatex; bibtex; pdflatex; pdflatex` (TeX Live 2026 BasicTeX plus a
build-only local `environ.sty` shim on `TEXINPUTS`, NOT shipped; arXiv's
TeX Live carries the real `environ` package). 8 pages before the N15 amendment (9 after); zero undefined
references or citations; zero overfull boxes; `pdftotext` confirms
"Under Review", "Anonymous", "NeurReps", "UniReps" absent and
"Samuel Larson", "Pebble ML" present. Package:
`ranklaw-arxiv-v2.tar.gz` (tex, bbl, bib, sty, figures).

Amended the same day (author-approved addition relayed by the coordinator): the whole-matrix effective-rank evidence (N15) was added as a new Appendix C, one sentence in §5, and a half-sentence in Limitations; the build is now 9 pages (still zero undefined, zero overfull, string checks unchanged).

### Title

The Rank the Task Demands: A Causal Rank Law for Matrix Memories Trained on Group Composition

### Authors

Samuel Larson (Pebble ML)

Corresponding: samlarson@pebbleml.com

### Categories

- Primary: cs.LG (Machine Learning)
- Cross-list: cs.AI (Artificial Intelligence)

### License

CC BY 4.0

### Abstract (paste into the arXiv abstract field)

Matrix-valued memories make rank the natural budget of a learned representation: the number of independent directions a state spans bounds what it can bind, compose, and track. We report causal evidence, on a group-composition testbed trained under a hard single-state bottleneck with a fixed decoder that cannot launder rank, that gradient descent recruits precisely the rank the task's algebra demands. A companion paper (Larson, 2026) establishes the analogous recruitment and causal necessity pattern on a $K$-pair associative-binding testbed, where exact recovery provably requires state rank at least $K$; this paper inherits that instrument and extends the rank law from a scalar capacity bound to a representation-theoretic one. On group-composition state tracking over five finite groups spanning the solvable/non-solvable divide, the recruited rank equals the group's minimal faithful real representation dimension $d_{\min}$ (Spearman $\rho = 0.9747$, the design's tie-capped maximum), the dimension-matched solvable/non-solvable pair $S_4$/$A_5$ is statistically equivalent under a pre-registered test, and a pre-registered force-rank razor is exact in both directions: one rank below $d_{\min}$, recovery is capped by the target's tied unit spectrum at $\sqrt{(d_{\min}-1)/d_{\min}} \le 0.894$, below the $0.9$ threshold in every group by construction, with observed cells at 76-95% (mean 88%) of that ceiling; at $d_{\min}$, not guaranteed a priori, recovery clears the anchor-relative bar in all five groups. Representation theory, not computational complexity class, sets what gradient descent buys.

### Comments field (suggested)

> 8 pages, 2 figures. Companion paper in preparation.

### Every change made in v2 (relative to the v1 named build / the two review trees)

Structure and de-anonymization:
- Template: `\documentclass{article}` + `\usepackage[final]{neurips}` (neurips.sty copied from `unireps-ea/`), notice string blanked exactly as `unireps-ea/main.tex` does; same package set as unireps plus `amsthm` (`\theoremstyle{plain}` + `\newtheorem{fact}{Fact}` replaces the jmlr `\theorembodyfont`/`\theoremheaderfont` constructs). No draftwatermark. `\bibliographystyle{plainnat}`.
- Author block: `Samuel Larson` / `Pebble ML` / `samlarson@pebbleml.com`. Short-title argument of the jmlr `\title[...]` dropped (NeurIPS style has none).
- Keywords: the jmlr `\begin{keywords}` block rendered as one unnumbered line after the abstract (`\textbf{Keywords:} effective rank, fast weights, group representations, state tracking`).
- Section order: intro; setup (+ binding foundation paragraph from `03_binding.tex`); observed rank law; NEW "Equivalence at Matched Dimension" (= `unireps-ea/sections/04_equivalence.tex`, label `sec:equivalence`); causal razor; related work + limitations; appendices.
- The neurreps "Dimension, not solvability" paragraph was removed from the observed section (its numbers now live once, in the new equivalence section, tagged N5).
- The combined rank-tracking + razor-table figure (`fig:tracking`) moved verbatim from the razor section into the observed section (where it is first cited) so figures number in citation order; the razor-step figure widened from `0.80\textwidth` to `\textwidth` (as unireps typeset the same figure).
- `unireps-ea/figures/fig1_convergence.pdf` (same 19 points plus the TOST-margin inset) added as its own figure (`fig:convergence`) inside the equivalence section, with the unireps caption verbatim; the equivalence section's "Figure (inset)" reference resolves to it. NOTE FOR AUTHOR: its main panel duplicates the left panel of Figure 1; kept because the inset is the only visual of the ±0.5 margin. Drop one if the duplication reads badly.
- Appendices: A groups table (one copy); B per-seed observational rank (table + per-group deviation + S4/S5 anchor-exceed note); C NEW "Whole-Matrix Effective Rank" (`app:wmrank`, two tables, N15; see the amendment list below); D "Instrument Details and the Two Defects" = unireps `app:instrument` in full (centering, length-robustness split, ambient-identity tax, non-binding upper half, rank-constrained cosine ceiling with the von Neumann derivation, primary/crosscheck degauging) merged with neurreps `app:damb` (the tax paragraph is the neurreps three-signature version plus the unireps "registered as inconclusive / registered fix" sentence) and with the neurreps per-group ceiling-fraction sentence (N12) placed after the ceiling derivation; E S3 per-seed causal table (unireps `app:s3causal`); F soft convergence (neurreps gate1a); G depth-21 pointer (neurreps). Neurreps `app:damb` label retired; every `\ref{app:damb}` and the centering/length-split `\ref{app:m1}` pointers now point at `app:instrument`; the S3 per-seed pointer in the razor section now points at `app:s3causal`.
- De-duplicated (removed as duplicates, facts kept once): neurreps app:m1 centering sentence (N11 -> unireps centering paragraph), neurreps app:m1 length-split sentence (N10 -> unireps paragraph, identical wording), neurreps app:m1 S3 per-seed prose sentence (N3 -> the S3 table caption, which states the same numbers plus the anchor-mean recompute). Unireps `app:m1` table (identical to neurreps') kept once.
- Evidence tags: all 31 base tags survive next to their sentences; unireps U-tags converted (U1->N4, U2->N5, U3->N1/N12, U4->N2/N3, U5->N10, U6->N7, U7->N11); the ceiling paragraph carries `N1, N2, N12 (U3)` because its 0.61-0.84 cosines are the N12 cells' raw values rather than an N-row literal.
- Related work: neurreps paragraph kept; unireps' nichani2025factual sentence and huh2024platonic sentence appended verbatim; chughtai2023toy and chughtai2023universality verified byte-identical bib bodies, so only `chughtai2023universality` is kept and cited once. Self-citation restored VERBATIM from the pre-cut tree (commit 22d77cf^): "A bolt-on-latent negative result \citep{larson2026gradient} is this result's counterpart."
- Companion phrasing: every "concurrent companion submission" (abstract, binding paragraph, limitations, Appendix F) replaced by "a companion paper \citep{larson2026companion}"; in Appendix F the sentence was reordered to "Appendix A of a companion paper \citep{larson2026companion} performs this periodicity analysis ..." (same content). Binding-task numbers NOT restored.
- refs.bib: union of both trees (shared entries verified identical) plus `huh2024platonic`, `larson2026gradient` (author field `Larson, Samuel`; booktitle ICML 2026 Workshop on Mechanistic Interpretability; note as specified), and `larson2026companion` (`@misc`, note "Companion paper, in preparation for arXiv"). plainnat renders the two self-citations as Larson 2026a (companion) and 2026b (gradient).
- Intro cross-references: "(Section~\ref{sec:observed})" now sits after "predicts the recruited rank almost perfectly" and the equivalence clause points at `sec:equivalence`; the padding pointer points at `app:instrument`.

### Sentences/clauses authored by the agent (FLAG FOR AUTHOR REVIEW; nothing numeric)

1. Observed section, replacing the removed paragraph: "The dimension-matched pair $S_4$/$A_5$ is tested directly in Section~\ref{sec:equivalence}."
2. Observed section, the clause after "corroborating rather than independently decisive": ": the band's upper half is non-binding by construction, and a disclosed length-robustness split is reported alongside it (Appendix~\ref{app:instrument})." (replaces the base parenthetical "(a disclosed length-robustness split is in Appendix B)"; adapted from unireps §3's "the readout's [1, d_min] range makes the band's upper half non-binding by construction").
3. Convergence figure evidence tag `N4, N5` and the S3 table tag `N2, N3` (tag text only).
4. Appendix C degauging paragraph: "denoted crosscheck $\recninety$ throughout Section~\ref{sec:razor} and Figures~\ref{fig:tracking} and~\ref{fig:razor}" (unireps had "denoted rec@0.9 throughout Section 5 and Figure 2"; the word "crosscheck" and the second figure ref are the only additions, to match this paper's naming).
5. Appendix C tax paragraph: opening sentence is unireps' "The first 58-cell sweep's force-rank target was the eye-padded ..." spliced onto neurreps' body; the sentence "The sweep verdict was registered as inconclusive with the tax as mechanism; the zero-padded causal wave of Section~\ref{sec:razor} is the registered fix." is unireps' wording with "Section~\ref{sec:causal}" -> "Section~\ref{sec:razor}".
6. main.tex header comments (not compiled).

N15 amendment (author-approved; every number recomputed from the raw JSONs by the agent and matching the coordinator's figures to three decimals; all 51 raw files md5-verified against `figure_gen.py` SOURCE_MD5):
7. Appendix C, Table `tab:wmrank` (per group: d_min, d_state, observational-arm whole-matrix mean±sd (n), zero-padded unconstrained anchor; S3 four-seed mean±sd, others single seed) and its caption, written by the agent to the coordinator's specification: states the quantity has no subspace lens and ranges up to d_state; that the eye-padded arm recruits the target's rank d_min+2; that on the zero-padded target the unconstrained state recruits d_min and leaves the two spare dimensions unused.
8. Appendix C, second table `tab:wmrank_cells` (whole-matrix rank of every zero-padded razor cell, seed 0; S3 mean±sd over four seeds) with caption "The capped cells sit at or below their cap k; at k >= d_min they land where the unconstrained anchor does." NOT in the coordinator's table spec (the numbers were requested "for completeness"); drop the table if unwanted.
9. §5, after the sufficiency paragraph: "Table~\ref{tab:wmrank} adds an instrument-independent reading: on the zero-padded target the unconstrained anchor's whole-matrix effective rank, taken over the full $\dstate \times \dstate$ state with no subspace lens, lands at $1.82/2.95/2.88/3.55/4.73$ against $\dmin = 2/3/3/4/5$ with two ambient dimensions available, so the recruited rank equals $\dmin$ without any restriction to the model's dominant subspace."
10. Limitations: ", and the zero-padded whole-matrix anchors (Appendix~\ref{app:wmrank}) are single-seed outside $S_3$".
11. `papers/neurreps-ea/brief.md`: N15 row appended to the claims table (numbers, verdict record, raw paths, table/section).

Not resolved / for the author:
- Local build needed a scratchpad `environ.sty` shim because BasicTeX lacks the package; verify arXiv's AutoTeX preview (expected 8 pages).
- `larson2026gradient` author field written as "Larson, Samuel" (the original cut entry said "Larson, Sam"); change if the ICML listing uses "Sam".
- The main.tex header comment still names the two source directories by path (`papers/neurreps-ea/`, `papers/unireps-ea/`); no venue name reaches the compiled text.
