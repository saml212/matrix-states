# ICML 2026 MI Workshop — Submission #572

**Paper:** *The Gradient Does Not See Rank: Rank-Indifference in Matrix-CODI on ProsQA*
**Author:** Samuel Larson — Pebble Machine Learning — `samlarson@pebbleml.com`
**Venue:** ICML 2026 Mechanistic Interpretability Workshop (Seoul, 10 Jul 2026)
**Decision:** Accept (Virtual Poster). Acceptance rate ~44% (356 / ~800).
**OpenReview:** https://openreview.net/forum?id=Spof4PusVI  (submission id `Spof4PusVI`, number 572)
**Public code release:** https://github.com/saml212/matrix-codi-rank-blindness

## Status (as of 2026-06-21)

| Deliverable | Deadline | Status |
|---|---|---|
| Acceptance instructions form (Google) | 21 Jun 2026 | ✅ submitted |
| Virtual poster upload (Google) | 3 Jul 2026 | ✅ submitted |
| Camera-ready (OpenReview revision) | 21 Jul 2026, 15:59 PDT | ✅ submitted |

Next (not yet done): arXiv posting + Google Scholar indexing. See `ARXIV_PLAN.md`.

## Where everything is

### Paper source (this directory)
- `main.tex` — top-level paper. Uses `\usepackage[accepted]{icml2026}` (de-anonymized).
- `sections/01..10_*.tex` — body sections (intro, background, rank-blind readout,
  depth/scale, positive control, related work, discussion, conclusion,
  reproducibility, impact statement).
- `refs.bib` — bibliography.
- `icml2026.sty`, `icml2026.bst` — official ICML 2026 style files. **The footer in
  `icml2026.sty` (`\ICML@appearing`, ~line 159) was changed** from the PMLR
  proceedings string to the workshop string required by the organizers:
  *"Mechanistic Interpretability Workshop at the 43rd International Conference on
  Machine Learning, Seoul, South Korea, 2026. Copyright 2026 by the author(s)."*
- `figures/*.pdf` + `figures/generate_figures.py` — the 5 figures and their generator.
- `Makefile` — `make` builds `main.pdf`.

### Poster source
- `poster/poster.tex` — self-contained portrait poster, 24in W × 36in H (within the
  workshop's 36×24 limit). Plain `article` + `geometry` + `tikz`; no poster class
  needed. Reuses `../figures/*.pdf`.
- Build: `cd poster && pdflatex poster.tex`.

### Built PDFs (gitignored — rebuild from source)
- `main.pdf` (9 pp, US Letter) — rebuild with `make`.
- `poster/poster.pdf` (1 page, 24×36) — rebuild with `pdflatex poster.tex`.
- Archived copies of the exact submitted artifacts also live on the author's
  Desktop (`~/Desktop/ICML-MI-572-camera-ready.pdf`, `~/Desktop/ICML-MI-572-poster.pdf`)
  and on OpenReview (camera-ready) / the workshop poster repository (poster).

### Other docs here
- `README.md` — this file (authoritative index).
- `ARXIV_PLAN.md` — steps to post to arXiv + get Google Scholar indexing.
- `PAPER_READER_VIEW.md` — readable prose snapshot of the paper.
- `SUBMISSION_CHECKLIST.md` — **historical** pre-submission checklist (May 2026).

## How to rebuild
```
make            # builds main.pdf (needs pdflatex + bibtex; texlive)
make figures    # regenerates figures/*.pdf (needs matplotlib)
cd poster && pdflatex poster.tex   # builds poster/poster.pdf
```

## Key facts / decisions
- **Camera-ready vs. accepted version:** de-anonymized to Samuel Larson / Pebble
  Machine Learning; reproducibility link points to the public GitHub repo (was an
  anonymous 4open.science link); one reviewer-driven framing paragraph added to the
  intro (rank is functional only under specific conditions; we test whether
  matrix-CODI makes it so — it does not); minor prose sweep (US spelling, no
  em-dashes per author style, COCONUT bib entry fixed). Title, abstract, and all
  numerical results unchanged.
- **Reviews:** two reviewers, both accept. Shared note: justify why rank is the
  right observable — addressed by the new intro framing. The suggested
  provably-rank≥2 synthetic task is logged as future work, not run.
- **Open item to verify:** §6 quotes matrix-CODI at 82.03% while §4's γ=0 scale
  sweep puts it ~1.3pp below the SFT baseline; the prose treats these as different
  seed draws within seed noise (consistent with the §3 three-seed spread
  80.47–82.81). Confirm this is seed variance, not two different configs, before
  any arXiv revision.
- **Workshop is non-archival** (virtual posters go to a repository, not PMLR), so an
  arXiv version may differ from / extend the camera-ready freely.
