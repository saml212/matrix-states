# SUBMISSION PACKAGE — the two 2026 extended abstracts (target date 2026-07-11)

Assembled 2026-07-10 from the two gauntleted trees. Each venue section
records the final submission artifact, the compliance state verified
against that tree's `VENUE_REQUIREMENTS.md`, and the steps only a human
can take. Gauntlet history (attack/defense/rebuttal/style/format/render
reports) lives in each tree's `gauntlet/round-1/` and `gauntlet/round-2/`.

---

## 1. NeurReps 2026 — Extended Abstract track

- **Title (typeset, candidate A):** "The Rank the Task Demands: A Causal
  Rank Law for Matrix Memories Trained on Group Composition"
  (short title: "A Causal Rank Law for Matrix Memories")
- **Candidate B** (in `neurreps-ea/brief.md`): "Minimal Faithful
  Dimension Is What Gradient Descent Buys: Necessity and Sufficiency of
  d_min in Trained Matrix States"
- **Abstract (223 words source-convention, at most 228 under the
  strictest rendered-token count):** Matrix-valued memories make rank
  the natural budget of a learned representation [...] Representation
  theory, not computational complexity class, sets what gradient descent
  buys. (Full text: `neurreps-ea/main.tex`, `\begin{abstract}` block;
  trimmed 4 words 2026-07-10 to clear the 230 band under every counting
  convention, per format-audit-v2 S-C.)
- **Bundle (the submission artifact):**
  `papers/neurreps-ea/bundle/` — `neurreps-ea-submission.tex` (single
  flattened file, per CFP), `neurreps-ea-submission.pdf` (anonymized
  review build, 8 pages), `refs.bib`, `jmlr.cls`, `jmlrutils.sty`,
  `figures/fig1_razor_step.pdf`, `figures/fig2_rank_tracking.pdf`.

### Compliance checklist (verified against `neurreps-ea/VENUE_REQUIREMENTS.md`)

- [x] Official template: `\documentclass[mlabstract,onecolumn]{jmlr}`,
  `jmlr.cls`/`jmlrutils.sty` verbatim from the workshop style zip
- [x] EA page limit: body ends on page 4 (4pp excluding references and
  appendices); references + Appendices A-E follow
- [x] Single `.tex` file: bundle tex is byte-identical to a fresh
  flatten of `main.tex` + `sections/*.tex`
- [x] Anonymized review build: no author block; anonymization grep
  (larson/samlarson/pebble/rockie/github.com/huggingface.co/acknowledg/
  self-funded/funded by) zero hits across tex, bib, bbl, and bundle;
  the round-2 C1 self-citation leak was CUT (restore at camera-ready,
  note in `neurreps-ea/brief.md` anonymization section)
- [x] Draftwatermark track stamp kept (remove only at camera-ready)
- [x] Bibliography: 14 entries, 14 cited, zero orphans either
  direction; all 10 arXiv-numbered entries author-verified against the
  live arXiv API on 2026-07-10 (including the two previously fabricated
  entries: nazari2026rank = Philipp Nazari & T. Konstantin Rusch;
  sun2026staterank = Ao Sun et al., 12 authors)
- [x] Abstract in the 200-230 word band (223 source-convention; in
  band under every counting convention post-trim)
- [x] Compile: deterministic 3-pass tectonic build (see build note
  below), zero overfull hboxes, zero undefined references/citations
- [x] Gauntlet: round-2 style critique v2 — PASS (zero violations);
  round-2 format audit v2 — CLEAR, 0 critical (all 14 evidence rows
  recomputed exact from raws; v1 findings C1/S1/S2/S4/M1/M2 closed;
  v2's S-B fixed by retitling Section 2 to announce the binding
  foundation; v2's S-C fixed by the abstract trim; v2's S-A fixed by
  the Makefile build guard); render inspection — **PASS, 0 critical /
  0 serious / 0 minor, all 8 pages read**
  (`gauntlet/round-2/06_render_inspection.md`). **ACCEPT-READY.**

**BUILD NOTE (do not skip):** a clean-tree tectonic build of this paper
transits a mis-paginated 12-page layout and converges to the correct
dense 8-page layout only on the THIRD pass (aux fixed-point transient;
round-2 v2 audit finding S-A, empirically characterized 2026-07-10:
clean gives 12pp twice, then 8pp, then stable). `make` now runs three
passes and is deterministic from any starting state. The submission
artifact is the pinned `bundle/neurreps-ea-submission.pdf` (8 pages) —
upload that file; do not recompile with a single-pass command.

### Decisions of record (coordinator-adjudicated; PI delegated via impact-first directive 2026-07-10)

- [x] **Final title: CANDIDATE A** (as typeset): "The Rank the Task
  Demands: A Causal Rank Law for Matrix Memories Trained on Group
  Composition". Candidate B retired to `neurreps-ea/brief.md`.
- [x] **Authors: Sam Larson, Pebble AI (pebbleml.com) — solo;**
  corresponding: samlarson16@gmail.com.
- [x] **License (arXiv): CC BY 4.0** recommended.

### Remaining human-only steps (PI)

- [ ] Re-verify the 2026 CFP when it drops (the live page still serves
  the 2025 edition; the NeurIPS 2026 workshop list lands 2026-07-11) —
  page limit, template, deadline, dual-submission policy.
- [ ] Create/log into the OpenReview account and upload the bundle.
- [ ] At camera-ready: uncomment + fill the author block, restore the
  cut self-citation (see `brief.md` anonymization note), remove the
  draftwatermark block.

---

## 2. UniReps 2026 — Extended Abstract track

- **Title (typeset, candidate A):** "Dimension, Not Solvability:
  Trained Matrix States Converge to the Minimal Faithful Representation
  Dimension"
- **Candidate B** (in `unireps-ea/brief.md`): "Five Groups, One Law:
  Learned State Rank Converges to the Algebraic Minimum Across the
  Solvable Divide"
- **Abstract (224 words, as typeset):** When do differently structured
  tasks drive a learner to the same representation? [...] Learned
  representational dimension follows the task's algebra, not its
  complexity class. (Full text: `unireps-ea/main.tex`,
  `\begin{abstract}` block; the two comment-swallowed clauses found by
  the round-2 style critique v2 were restored 2026-07-10.)
- **Bundle (the submission artifact):**
  `papers/unireps-ea/bundle/` — `unireps-ea-submission.tex` (flattened
  single file), `unireps-ea-submission.pdf` (anonymized NeurIPS-template
  build with line numbers, 7 pages), `refs.bib`, `neurips.sty`,
  `figures/fig1_convergence.pdf`, `figures/fig2_razor_step.pdf`.

### Compliance checklist (verified against `unireps-ea/VENUE_REQUIREMENTS.md`)

- [x] Template: official NeurIPS kit (2025 stand-in per the CFP's "use
  the NeurIPS LaTeX template"; swap when the 2026 styles ship)
- [x] EA page limit: main text ends on page 4 (4pp excluding references
  and appendix); references start page 4, Appendix A starts page 5;
  7 pages total
- [x] Anonymized submission build: NeurIPS default mode (anonymous
  block + line numbers); anonymization grep zero hits across tex, bib,
  and bundle
- [x] Bibliography: 10 entries, 10 cited, zero orphans; all 8
  arXiv-numbered entries author-verified against the live arXiv API on
  2026-07-10 (nazari2026rank and sun2026staterank confirmed correct
  post-fix)
- [x] Evidence tokens: 30 `% <!-- evidence: Ux -->` tokens across
  abstract and sections; bundle byte-identical to a fresh flatten;
  bundle pdf byte-identical to `main.pdf`
- [x] Abstract in the 200-230 word band (224, post-restoration)
- [x] Compile: deterministic 3-pass tectonic build (mirrors the
  neurreps S-A guard), zero overfull hboxes, zero undefined
  references/citations
- [x] Gauntlet: round-2 format audit v2 — all 9 tracked findings
  CLOSED, 0/0/0 (`gauntlet/round-2/05_format_audit_v2.md`); style
  critique v2 FAIL(2) — both comment-swallowed abstract clauses
  restored 2026-07-10; render inspection v2 finding (open-marker
  legibility) and v3 findings (S5-vs-inset crowding, dashed-vs-dotted
  distinguishability) fixed in `figures/figure_gen.py` and
  regenerated from the md5-pinned raws; final render inspection —
  **PASS, 0 critical / 0 serious / 0 minor, all 7 pages read**
  (`gauntlet/round-2/06_render_inspection_v4.md`). **ACCEPT-READY.**

### Decisions of record (coordinator-adjudicated; PI delegated via impact-first directive 2026-07-10)

- [x] **Final title: CANDIDATE A** (as typeset): "Dimension, Not
  Solvability: Trained Matrix States Converge to the Minimal Faithful
  Representation Dimension". Candidate B retired to
  `unireps-ea/brief.md`.
- [x] **Authors: Sam Larson, Pebble AI (pebbleml.com) — solo;**
  corresponding: samlarson16@gmail.com.
- [x] **License (arXiv): CC BY 4.0** recommended.

### Remaining human-only steps (PI)

- [ ] Re-verify the 2026 CFP (the acquired requirements are from the
  2025 edition site; `unireps-<year>.netlify.app` pattern) — page
  limit, template file, deadline; swap in the NeurIPS 2026 style file
  if named.
- [ ] Create/log into the OpenReview account and upload the bundle.
- [ ] At camera-ready: fill the author block (5pp camera-ready EA
  limit per the 2025 CFP).

---

## 3. arXiv release packages (named builds, assembled 2026-07-10)

Upload-ready NAMED arXiv source packages for both EAs live in
`papers/neurreps-ea/arxiv/` and `papers/unireps-ea/arxiv/`
(`<slug>-arxiv-v1.zip` + compiled PDF + `metadata.md` each; Desktop
copies at `~/Desktop/arxiv-neurreps/`, `~/Desktop/arxiv-unireps/`, flow
in `~/Desktop/ARXIV_UPLOAD_STEPS.md`). Each named tex is the pinned
anonymized bundle with ONLY de-anonymization applied — neurreps:
watermark removed + jmlr author block filled; unireps: `[preprint]`
style mode (author block renders, line numbers drop) + author block
filled; body text diff-verified identical to the bundles. Both compile
clean standalone from the zips (3-pass; 8pp / 7pp, matching the review
builds) and were render-inspected page-by-page: named author block
correct, no watermark, no "Anonymous", no line numbers, no `??` refs,
figures intact. The neurreps cut self-citation stays cut in v1 (frozen
gauntleted text); restore at camera-ready / arXiv v2. The anonymized
`bundle/` artifacts remain pristine for the workshop submissions.

## Provenance notes

- Every figure regenerates from md5-pinned raw archives via each tree's
  `figures/figure_gen.py`; non-figure numbers trace to the evidence-row
  maps in each `brief.md` (N1-N14 / U1-U7), spot-recomputed from raw
  JSONs by the round-2 format audits.
- Bib metadata was verified programmatically against the live arXiv API
  (`export.arxiv.org/api/query`), never from memory — the two
  fabricated-author entries this program caught (nazari2026rank,
  sun2026staterank) are correct in both trees and both bundles.
