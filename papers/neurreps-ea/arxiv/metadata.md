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
