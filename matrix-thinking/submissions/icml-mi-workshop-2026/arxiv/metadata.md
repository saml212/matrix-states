# arXiv submission metadata

Package: `icml-mi-rankblind-arxiv-v1.zip`
Source tree: `matrix-thinking/submissions/icml-mi-workshop-2026/arxiv/`

## Title

The Gradient Does Not See Rank: Rank-Indifference in Matrix-CODI on ProsQA

## Authors

Sam Larson (Pebble AI) — `sam@pebbleml.com`

Note: OpenReview submission #572 was registered under the legal-name
variant "Samuel Larson / Pebble Machine Learning." This package uses
"Sam Larson / Pebble AI" per the standing instruction for this build,
which also matches the byline already live on pebbleml.com and the
repo's git author identity. Flagging so the two records can be
reconciled (or intentionally kept distinct) before upload — arXiv
listings are easiest to link to OpenReview/Scholar when the name
matches exactly.

## Abstract (plain text)

Continuous chain-of-thought models compress reasoning into latent tokens.
Matrix-valued variants introduce rank as a single-sample structural observable on
the latent matrix Z. If matrix latents carry parallel reasoning paths via
superposition, rank should track them, and truncating Z to low rank should hurt
accuracy on tasks whose solutions plausibly require multiple components. Across four
training regimes of a matrix-CODI model (three on ProsQA, one on GSM8K-Aug below the
learning threshold), the rank-k projection ablation curve is flat to within 0.6
percentage points. A three-seed replication yields 81.5 +/- 1.2 percentage points
accuracy while the final effective rank of Z spans {4, 12, 13}; the loss does not
reward any particular rank. To test whether rank-blindness arises from the
flatten-then-project readout alone, we trained four readouts: a bilinear
reparametrization, a bilinear-plus-GELU readout nonlinear in Z, an SVD-augmented
readout feeding singular values through an MLP, and a quadratic readout in Z Z^T. All
four rank-k curves remain flat (Spearman p-values 0.63, 0.14, 0.82, 0.46). The flat
curves persist for readouts nonlinear in Z. A linear probe on Z underperforms a raw
pretrained hidden state at target prediction (AUC 0.673 vs. 0.846). A negative
control on vanilla GPT-2 SFT (no matrix bottleneck, no Z, three seeds, n=500)
reproduces a flat rank-k curve under the same intervention paradigm with pooled-mean
range 0.20pp, and a random-h sensitivity floor lands at the same accuracy: the rank-k
ablation alone conflates rank-blindness with position-irrelevance.

## Categories

- **Primary:** `cs.LG` (Machine Learning)
- **Secondary (recommended):** `cs.AI` (Artificial Intelligence) — best fit of the
  two options in scope; the paper's subject (latent chain-of-thought reasoning,
  mechanistic-interpretability probing methodology) sits squarely in general AI/ML
  interpretability rather than neural-architecture or evolutionary computation, so
  `cs.NE` is not recommended.
- **Optional third cross-list to consider:** `cs.CL` (Computation and Language) —
  the object of study (COCONUT/CODI continuous chain-of-thought) is a language-model
  reasoning mechanism; most of the directly-cited adjacent work (Hao et al. COCONUT,
  Shen et al. CODI/SIM-CoT, Rizvi-Martel et al.) posts to cs.CL. Not one of the two
  options given, included here as a documented option for whoever finalizes the form.

## License recommendation

**CC BY 4.0.** Matches the license already set on the OpenReview camera-ready page
per the (unmerged) camera-ready branch's `README.md`.

## Comments field (suggested)

> Accepted at the ICML 2026 Workshop on Mechanistic Interpretability (virtual
> poster; the workshop is non-archival). 9 pages, 5 figures.

## Policy check (workshop archival status)

**Confirmed non-archival — no blocker.** Primary source: the workshop's own CFP
page, `https://mechinterpworkshop.com/cfp/`, states explicitly: *"The workshop is
non-archival."* (fetched 2026-07-10). Corroborated by this repo's own
`README.md` on the (unmerged) camera-ready branch: "Workshop is non-archival
(virtual posters go to a repository, not PMLR), so an arXiv version may differ
from / extend the camera-ready freely." No archival-transfer or copyright-assignment
clause was found anywhere in the submission tree or the CFP. Author retains full
rights; arXiv posting is unblocked.

## Build provenance (important — read before uploading)

The content in this `arxiv/` directory is **not** a copy of what's currently
committed on `main` at `matrix-thinking/submissions/icml-mi-workshop-2026/main.tex`
(that file is still the double-blind anonymized submission build, author hardcoded
to "Anonymous Authors" with no working de-anonymization toggle — see repo finding
below). The actual camera-ready content — de-anonymized author block,
`\usepackage[accepted]{icml2026}`, the workshop's non-archival footer text, the
public-repo reproducibility link, and a handful of reviewer-driven prose fixes —
exists only in commit `88924b3` ("paper: ICML 2026 MI Workshop camera-ready (#572)
+ poster + arXiv prep") on the **unmerged** remote branch
`origin/claude/compassionate-kare-3e3f19`. This `arxiv/` package was assembled by
pulling that commit's sections/refs/style content via `git show` and substituting
the author identity per the standing "Sam Larson, Pebble AI" instruction (see
Authors note above). **Recommend merging `88924b3` into `main`** (or cherry-picking
it) so the repo's `main.tex` actually reflects the paper's real, accepted,
camera-ready state — right now the true camera-ready lives only on a stranded
branch and the checked-in `paper.pdf`/`paper-anon.pdf` in the parent directory are
stale (compiled 2026-04-28, before ~7 rounds of post-April content fixes including
the ICML-style-file swap; they predate `88924b3` entirely).

## Build fix applied (toolchain-local, documented for reproducibility)

`main.tex` loads `\usepackage[T1]{fontenc}` before the rest of the preamble. Without
it, this environment's `tectonic` install resolves fonts under the default Unicode
(`TU`) encoding, for which the legacy `times.sty` (loaded internally by
`icml2026.sty`) has no `.fd` file — every bold/italic Times shape request silently
substitutes upright medium weight (`LaTeX Font Warning: Font shape 'TU/ptm/b/n'
undefined`), and the substituted metrics are just large enough to trip
`icml2026.sty`'s hardcoded running-title box-height check (6.25pt cutoff), which
replaces the page header with "Title Suppressed Due to Excessive Size" on every
page. `\usepackage[T1]{fontenc}` resolves proper Times metrics; the rebuild has zero
font warnings and the correct running header throughout. This is the only change
relative to the `88924b3` camera-ready `main.tex` beyond the author-identity swap.

## What's in the package

- `main.tex` — top-level source (camera-ready content, `[accepted]` style, T1
  fontenc fix)
- `sections/01..10_*.tex` — 10 body sections
- `refs.bib` — bibliography (camera-ready, includes the COCONUT `@article`→type fix)
- `main.bbl` — pre-generated bibliography (required; arXiv does not run BibTeX)
- `icml2026.sty`, `icml2026.bst` — official ICML 2026 style files, patched with the
  workshop's non-archival footer text (from `88924b3`)
- `algorithm.sty`, `algorithmic.sty`, `fancyhdr.sty` — bundled dependencies (not
  guaranteed present in every minimal TeX Live install; bundling avoids relying on
  arXiv's installed package set)
- `figures/*.pdf` — the 5 figures actually referenced by `\includegraphics`
  (`fig1_rank_curves`, `fig2_seed_decoupling`, `fig3_scale_sweep`,
  `fig5_linear_probe`, `fig6_negative_control`; `fig4_depth_sweep.pdf` is not
  included — it's generated but never referenced in the paper, a pre-existing
  orphan noted in this repo's own format-audit review)
- `main.pdf` — compiled output (9 pages, US Letter, tectonic)

## Compile check

Rebuilt clean with `tectonic --keep-intermediates --keep-logs main.tex`:
- **9 pages** (US Letter) — matches the camera-ready branch's own stated page count
  ("main.pdf (9 pp, US Letter)" in `88924b3`'s `README.md`) and is a plausible
  reduction from the stale `paper.pdf` in the parent directory (11 pages, compiled
  before the post-April compaction/style-guide-audit commits).
- Zero undefined citations, zero undefined references, zero `??` markers.
- Zero font warnings (after the T1 fontenc fix).
- Render-inspected all 9 pages as rasterized images: title/author block correct
  (Sam Larson / Pebble AI / sam@pebbleml.com), non-archival workshop footer correct
  ("Mechanistic Interpretability Workshop at the 43rd International Conference on
  Machine Learning, Seoul, South Korea, 2026. Copyright 2026 by the author(s)." —
  not the PMLR-proceedings string), running header correct on every interior page,
  no anonymization leftovers anywhere in the extracted text
  (`anonymous`/`4open.science`/placeholder-email all absent), figures and tables
  legible, references list complete and correctly formatted.
