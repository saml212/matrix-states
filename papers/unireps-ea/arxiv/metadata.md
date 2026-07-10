# arXiv metadata — UniReps EA named build (v1)

Package: `unireps-ea-arxiv-v1.zip` (tex + bbl + neurips.sty + refs.bib +
2 figure PDFs; arXiv uses the included `.bbl`, it never runs BibTeX).
Local build: 3× tectonic passes, 7 pages, matches the review build's
pagination; render-inspected page-by-page 2026-07-10. The named build
uses the NeurIPS kit's `[preprint]` mode: real author block, no line
numbers, notice line blanked.

## Title

Dimension, Not Solvability: Trained Matrix States Converge to the Minimal Faithful Representation Dimension

## Authors

Sam Larson (Pebble AI)

Corresponding: samlarson16@gmail.com

## Abstract (paste into the arXiv abstract field)

When do differently structured tasks drive a learner to the same representation? We train identical matrix-state sequence models end-to-end on the word problems of five finite groups, $S_3$, $S_4$, $A_5$, $S_5$, $A_6$, spanning the solvable/non-solvable divide, and measure the effective rank of the learned state against each group's minimal faithful real representation dimension $d_{\min}$, the exact dimensionality representation theory assigns the task. The learned rank converges onto the algebraic minimum: group means 1.88/2.85/2.83/3.59/4.74 against $d_{\min} = 2/3/3/4/5$, all 19 seeds inside the pre-registered $[0.7, 1.3] \cdot d_{\min}$ band, Spearman $\rho = 0.9747$, the maximum this design permits under the $S_4$/$A_5$ tie. The designed head-to-head is that tie: $S_4$ (solvable) and $A_5$ (non-solvable) share $d_{\min} = 3$, so if learned dimension were set by computational complexity class the pair should separate, and if set by representation dimension it should coincide. A pre-registered Welch equivalence test at $\pm 0.5$ rank-units declares equivalence decisively (difference 0.019, both one-sided tests near seven times the critical value). A pre-registered causal force-rank razor adds a third leg: capping rank one below $d_{\min}$ pins recovery under 0.9 by the metric's own geometry ($\sqrt{(d_{\min}-1)/d_{\min}} < 0.9$ for every group tested), a floor no capped cell violates, and restoring $d_{\min}$ lifts it: recovery empirically clears the pre-registered $0.9\times$-anchor bar in every group. Learned representational dimension follows the task's algebra, not its complexity class.

## Categories

- Primary: cs.LG (Machine Learning)
- Secondary: cs.NE (Neural and Evolutionary Computing)

## License

CC BY 4.0 (recommended).

## Comments field (suggested)

> 7 pages (4-page extended abstract plus references and appendix), 2
> figures. Companion paper: "The Rank the Task Demands: A Causal Rank
> Law for Matrix Memories Trained on Group Composition"

After BOTH arXiv IDs exist, an optional v2 can add each paper's arXiv ID
to the other's comments field (not blocking).

## Notes

- The gauntleted body text is byte-identical to the pinned anonymized
  review bundle (`bundle/unireps-ea-submission.tex`); the only changes
  are the `[preprint]` style option (drops line numbers, renders the
  author block) and the real author block.
- Expected arXiv-preview page count: 7. Verify the preview before
  announcing.
