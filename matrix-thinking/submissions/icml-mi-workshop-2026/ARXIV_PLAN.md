# arXiv + Google Scholar plan (everything short of submitting)

Goal: post the camera-ready to arXiv (so it is citable and Google-Scholar-indexed),
without actually clicking submit. Everything below is staged; the human does the
final clicks.

## ⚠️ Blocker to resolve first: arXiv endorsement
As of **21 Jan 2026**, a first-time arXiv submitter needs one of:
1. an institutional email **and** prior authorship on an existing arXiv paper in the
   target domain (cs.LG), **or**
2. a **personal endorsement** from an established cs.LG arXiv author.

Samuel Larson has no prior arXiv paper, so option 2 is the likely path: ask an
advisor / colleague who has published in cs.LG to endorse. Endorsement is requested
from inside the arXiv submission flow (arXiv gives a code to send the endorser).
Line this up before attempting the upload — it gates everything else.

## The submission package (already built)
`~/Desktop/arxiv-572-submission.tar.gz` (~120 KB). Verified to compile **clean-room**
with `pdflatex` twice and **no bibtex** (exactly how arXiv builds): 9 pp, US Letter,
no undefined references.

Contents: `main.tex`, `sections/*.tex`, `figures/*.pdf`, `main.bbl` (required — arXiv
does not run bibtex), `refs.bib`, and the bundled style files `icml2026.sty`,
`icml2026.bst`, `algorithm.sty`, `algorithmic.sty`, `fancyhdr.sty` (these last three
are bundled because they are not in a minimal TeXLive; including them guarantees the
build regardless of arXiv's TeXLive version).

Rebuild the tarball if the source changes:
```
cd matrix-thinking/submissions/icml-mi-workshop-2026
make                      # refreshes main.bbl + main.pdf
D=/tmp/arxiv572; rm -rf $D; mkdir -p $D/sections $D/figures
cp main.tex main.bbl refs.bib *.sty *.bst $D/
cp sections/*.tex $D/sections/; cp figures/*.pdf $D/figures/
( cd $D && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex )  # clean-room check
( cd $D && tar czf ~/Desktop/arxiv-572-submission.tar.gz main.tex main.bbl refs.bib *.sty *.bst sections figures )
```

## arXiv submission form — values to enter
- **Upload:** `~/Desktop/arxiv-572-submission.tar.gz`. Let arXiv recompile; check the
  generated PDF before finalizing.
- **Primary category:** `cs.LG` (Machine Learning).
- **Cross-list:** `cs.CL` (Computation and Language), `cs.AI`.
- **License:** **CC BY 4.0** — matches the license already set on OpenReview.
- **Title:** The Gradient Does Not See Rank: Rank-Indifference in Matrix-CODI on ProsQA
- **Authors:** Samuel Larson (Pebble Machine Learning)
- **Comments field:** `Accepted at the ICML 2026 Mechanistic Interpretability Workshop (virtual poster). 9 pages.`
- **Abstract (paste this plain-text version):**

```
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
```

Notes:
- The workshop is **non-archival**, so the arXiv version may differ from / extend the
  camera-ready. The footer already reads "Mechanistic Interpretability Workshop at
  the 43rd ICML"; that is fine to keep.
- Consider adding the public code link in the arXiv listing too (it is already in the
  paper's Reproducibility section): https://github.com/saml212/matrix-codi-rank-blindness

## Google Scholar — nothing to "submit"
Google Scholar **auto-indexes arXiv** (and OpenReview). Once the arXiv post is live
(usually next business-day announcement), Scholar typically picks it up within days.
To make sure it lands on the author's profile:
1. Create / sign in to a Google Scholar profile for Samuel Larson (scholar.google.com →
   "My profile"), affiliation Pebble Machine Learning, verify with an email.
2. Scholar auto-suggests papers by name match — confirm this paper when it appears.
   If it does not auto-appear, use **Add → Add articles** (search the title) or
   **Add article manually**.
3. Enable "automatically update my profile" so future versions/citations attach.
4. The OpenReview page is also indexable; the arXiv version is the stronger Scholar
   signal, so prioritize the arXiv post.

## Status checklist
- [x] arXiv source package built and clean-room compiled (`~/Desktop/arxiv-572-submission.tar.gz`)
- [x] Category / license / abstract / comments prepared (above)
- [ ] Endorsement secured (human — needs a cs.LG endorser)
- [ ] Upload to arXiv + verify recompiled PDF + finalize (human)
- [ ] Confirm Scholar indexing / add to profile (human, after arXiv goes live)
