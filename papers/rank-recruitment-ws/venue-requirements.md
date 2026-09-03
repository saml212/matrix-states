# NeurReps 2026 Extended Abstract track — venue requirements (Stage 0)

Live-fetched 2026-07-10 from https://www.neurreps.org/call-for-papers.
The page currently serves the **NeurIPS 2025 edition CFP**; the 2026 CFP
is expected after the NeurIPS 2026 accepted-workshop list drops on
2026-07-11 (per https://neurips.cc/Conferences/2026/Dates, live-verified
in `papers/VENUE_MAP.md`, commit 52eca3a, same day). **Every item below
is therefore a 2025-pattern PROJECTION and must be re-verified against
the live 2026 CFP before submission.**

## Requirements (2025 CFP language, fetched live 2026-07-10)

- **Track:** Extended Abstract. Verbatim: "Extended abstracts may be up
  to 4 pages long, excluding references and appendices."
- **Archival:** EA track is NON-archival (not included in PMLR; may be
  posted to arXiv). Proceedings track (9pp) is archival PMLR — not this
  submission's track.
- **Dual submission:** verbatim, fetched live 2026-07-10: "There are no
  restrictions on Extended Abstract submissions." (The 30%-new-material
  rule applies to the Proceedings track only.) ICLR-2027-flagship-safe.
- **Template (REQUIRED, official):** "All submissions must use the
  NeurReps 2025 LaTeX style files" — the NeurReps style zip (Google
  Drive id `12VSgMuQ-QH0V6sGl4KKnsXoJkkdfsCQR`, linked from the CFP), a
  JMLR/PMLR-based kit. `jmlr.cls` + `jmlrutils.sty` in this tree are
  copied verbatim from the sibling `papers/neurreps-ea/` tree, which
  vendored them from that zip (see its `VENUE_REQUIREMENTS.md`).
  Per the kit's INSTRUCTIONS.txt:
  - EA track: `\documentclass[mlabstract,onecolumn]{jmlr}`
  - use only the auto-loaded packages (amsmath, amssymb, natbib,
    graphicx, url, algorithm2e) plus the sample's own booktabs
  - references in a `.bib` file
  - the submission must be a **single `.tex` file** (this tree keeps
    modular `sections/` for drafting; `bundle/` holds the flattened
    single-file submission tex)
  - manuscript, data, and code anonymized during review — **no author
    block at all** in the review build; camera-ready adds it
  - keep the draftwatermark block (stamps the track); remove only at
    camera-ready
- **Review:** double-blind, OpenReview, >= 3 reviews per submission.
- **Deadline (2025 reference):** Aug 29 2025 -> Sep 4 2025 extended,
  AoE; 2026 PROJECTED similar (late August). NeurIPS 2026 suggested
  workshop-paper deadline: Aug 29 2026 AoE (live-verified in
  `papers/VENUE_MAP.md`).
- **Submission platform:** OpenReview portal.
- **Contact:** organizers@neurreps.org.

## Re-verify checklist (run when the 2026 CFP lands, expected after 2026-07-11)

1. Confirm NeurReps appears on the NeurIPS 2026 accepted-workshop list
   (https://neurips.cc/Conferences/2026/Workshops).
2. Diff the 2026 CFP against every item above (page limit, template zip,
   track structure, anonymization, deadlines, dual-submission language).
3. Swap the style files if the 2026 zip differs from the vendored 2025 kit.
4. Fill the real deadline into this file and `brief.md`.

## Sibling-submission disclosure

`papers/neurreps-ea/` is a DIFFERENT extended abstract targeting the same
venue (the group-composition causal rank law). This paper is the Task D/E
binding-and-composition program. Headline claims, figures, and tables are
disjoint by design; the overlap-management record is in `brief.md`
(§ "Companion papers and overlap management"). The 2025 CFP places no
restriction on multiple EA submissions; re-check on the 2026 CFP.

## Anonymization surface note

Double-blind review triggers the identity-leak grep (tokens recorded in
`brief.md`). One documented exception: the published companion negative
result (ICML 2026 MI workshop) is cited in third person with its real
bibliography entry, per standard double-blind self-citation practice;
the token `larson` is permitted ONLY inside that one `refs.bib` entry
and its rendered citation, nowhere else in the source or PDF.

## arXiv named build v1 (2026-09-03)

Tree: `papers/rank-recruitment-ws/arxiv-v1/` (main.tex, sections/, refs.bib,
neurips.sty, figures/, main.pdf, main.bbl). Built with pdflatex → bibtex →
pdflatex ×2 (TeX Live 2026); 6 pages, 0 undefined references, 0 BibTeX
warnings. Source tarball: `rankrecruit-arxiv-v1.tar.gz` (tex, bbl, bib,
sty, figures; no build shim shipped).

- **Authors:** Samuel Larson (Pebble ML, samlarson@pebbleml.com)
- **Title:** When the Gradient Sees Rank: Provable Necessity, Causal
  Recruitment, and Exact Composition in Trained Matrix Memories
- **Categories:** cs.LG (primary); cs.AI (cross-list)
- **License:** CC BY 4.0
- **Comments field:** 6 pages, 2 figures. Companion paper: The Rank the
  Task Demands (posted concurrently).

**Abstract (plain text, copied from main.tex):**

Matrix-valued fast-weight memories make rank a structural budget: the
directions a state spans bound what it can bind and compose. A recent
negative result reported that a matrix latent bolted onto a
vector-pretrained reasoner never uses its rank, but the probed task admits
a rank-1 solution, leaving the general question open. We answer it on
testbeds where the exact solution provably requires rank(Z) >= K and where
every rank-evading shortcut is closed by construction: the readout is
pinned to exact continuous recovery, since argmax decoding lets a rank-m
memory store roughly md associations; a single-matrix-state bottleneck
forecloses position decomposition; and no cell is called dead without
re-testing at 2-2.5x budget. Under these conditions gradient descent
recruits the rank the task demands in this architecture family: learned
effective rank tracks K across the tested grid (Spearman rho = 1.0 at
d = 16), train-time rank caps show a causal step at k ~ K (at d = 8,
K = 4: rank 3 gives at most 0.0004 recovery, rank 4 gives 0.97), and the
trained operator composes exactly through 21-fold self-application in four
of five seeds. An entity-subspace decomposition supplies the mechanism,
restricted rank exactly K on the key subspace, and in the single converged
rank-starved seed, the depth-decay curve is predicted from its
eigenspectrum alone.

**Every change from the review build (`papers/rank-recruitment-ws/`):**

- `main.tex` rewritten on the rank-law paper's arXiv template
  (`papers/neurreps-ea/arxiv-v2/main.tex`): `\documentclass{article}` +
  `\usepackage[final]{neurips}` with `\@noticestring` blanked; float,
  inputenc/fontenc, amsmath/amssymb, amsthm (plain-style `fact`
  environment, unused in this paper's text), graphicx, booktabs, xcolor,
  hyperref (hidelinks), url; `\raggedbottom`. Dropped: the draftwatermark
  block, `\jmlrvolume`, `\firstpageno`, `\editors{Under Review ...}`,
  `\jmlryear`, `\jmlrworkshop{...}`, the jmlr short-title option, the
  commented `\author{\Name{...} \Email{...} \addr ...}` placeholder, and
  every review-build header comment (TBD / PENDING-USER / anonymized).
- Author block added: `Samuel Larson / Pebble ML / samlarson@pebbleml.com`.
- `\begin{keywords}...\end{keywords}` replaced by the one unnumbered line
  `\noindent\textbf{Keywords:} effective rank, fast weights, associative
  memory, composition` (keyword list unchanged).
- Bibliography moved to follow the body and precede the appendix
  (`\bibliographystyle{plainnat}` + `\bibliography{refs}`, then
  `\input{sections/07_limitations}`), matching the rank-law build's order.
- `sections/06_related.tex`: "A concurrent companion submission" →
  "A companion paper \citep{larson2026ranklaw}"; the rest of that sentence
  is verbatim from the review build.
- `sections/07_limitations.tex` (Appendix C, Reproducibility): the
  anonymous.4open.science sentence replaced. Old: "Training, evaluation,
  and analysis code is available at \url{https://anonymous.4open.science/}
  (anonymized for review)." New: "Training, evaluation, and analysis code
  and md5-pinned raw artifacts accompany the repository release." No URL
  invented.
- `refs.bib`: `larson2026gradient` entry replaced by the arxiv-v2 version
  (author "Larson, Samuel" instead of "Larson, Sam"; added note "ICML 2026
  Mechanistic Interpretability Workshop; arXiv preprint submitted Sept
  2026"). New entry `larson2026ranklaw` (Samuel Larson, "The Rank the Task
  Demands: A Causal Rank Law for Matrix Memories Trained on Group
  Composition", 2026, note "Companion paper, arXiv preprint, posted
  concurrently"). The in-text citations of `larson2026gradient` (§1, §3)
  were already present in the review build in third person; `git log -p`
  over `sections/` shows no self-citation sentence was cut for
  double-blind review, so nothing was restored.
- Abstract, all section bodies, all numbers, figure/table captions, and
  every `% <!-- evidence: ... -->` tag are unchanged and remain adjacent to
  their sentences.

**Sentences authored for this build (verbatim):**

- "Training, evaluation, and analysis code and md5-pinned raw artifacts
  accompany the repository release." (`sections/07_limitations.tex`)
- "A companion paper \citep{larson2026ranklaw}" (fragment replacing "A
  concurrent companion submission" in `sections/06_related.tex`; the
  remainder of the sentence is the review build's)
- "Keywords: effective rank, fast weights, associative memory,
  composition" (`main.tex`, replacing the jmlr keywords environment)

**Open items:** `sections/06_related.tex` still says the companion
"restates this paper's recruitment, causal-necessity, and composition
numbers as background"; after the sibling's S1 de-dup edit
(`08_decision_record.md`) the rank-law paper points to this paper without
restating the numbers, so that clause may read stale — left verbatim per
the no-claim-change rule, flagged for the author. Appendix C's "every
number in this abstract" also carries over verbatim from the EA build.
