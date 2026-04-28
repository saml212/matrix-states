# Format audit

## Summary

The paper is structurally sound and nearly submission-ready. Citations, anonymization, banned
words, LaTeX hygiene, table style, and arXiv package compatibility are all clean. There is one
blocking cross-reference bug (`fig:depth` referenced but never defined), one serious bib
entry with a type/venue mismatch (`shen2025simcot` is an ICLR conference paper filed as
`@article` with the venue in the `journal` field), and a set of minor issues including a
literal placeholder "Table X" in the discussion, two dangling "appendix" references to
content that does not exist in the current submission, and the orphaned `fig4_depth_sweep.pdf`
which exists in `figures/` but is never included in the paper. Overall: **1 critical / 2
serious / 5 minor**. The compile-blocking `fig:depth` must be fixed before submission.

---

## Critical (would cause compile failure or be embarrassing at review)

**C1. Broken cross-reference: `fig:depth` — compile will emit `??`**

- File: `sections/09_reproducibility.tex`, line 49
- Text: `Figs.~\ref{fig:depth} and~\ref{fig:scale} from the vanilla SFT and vanilla CODI scripts`
- Issue: `\label{fig:depth}` is never defined anywhere in the paper. The depth-sweep section
  (`04_depth_scale.tex`) has text about a preliminary single data point but contains no figure
  environment and therefore no label. `fig:scale` is defined in the same file and is fine.
  `fig4_depth_sweep.pdf` exists in `figures/` but is not included anywhere.
- Fix: Either (a) remove `fig:depth` from the reproducibility sentence and replace with
  prose ("the depth-sweep results in §\ref{sec:depth-scale}"), or (b) add a figure
  environment in `04_depth_scale.tex` that includes `fig4_depth_sweep.pdf` and carries
  `\label{fig:depth}`. Option (a) is safer given that the section explicitly says the depth
  sweep is preliminary/deferred.

---

## Serious (would cause confusion or weaken the paper)

**S1. Bib type mismatch: `shen2025simcot` is a conference paper filed as `@article`**

- File: `refs.bib`, entry `shen2025simcot`
- Issue: The entry uses `@article` with `journal = {International Conference on Learning
  Representations (ICLR)}`. ICLR is a conference venue, not a journal. BibTeX will render
  this correctly in the text but the field name is semantically wrong and some citation
  styles will format it oddly. The parallel entry for `dong2021attention` (ICML) and
  `gunasekar2017implicit` (NeurIPS) correctly use `@inproceedings` with `booktitle`.
- Fix: Change to `@inproceedings` and move the venue to `booktitle`.

**S2. Literal placeholder in body text: "appendix Table X"**

- File: `sections/07_discussion.tex`, line 43
- Text: `(appendix Table X; deferred to camera-ready)`
- Issue: "Table X" is a literal placeholder that will appear verbatim in the compiled PDF.
  There is no appendix section in the submission. While the parenthetical marks it as
  deferred, the literal "X" is embarrassing in a reviewed submission.
- Fix: Replace with `(appendix, camera-ready)` or simply `(deferred to camera-ready)`.

---

## Minor (style nits)

**M1. Dangling "appendix" references — no appendix exists in the submission**

- Files: `sections/04_depth_scale.tex` line 72, `sections/07_discussion.tex` line 39–43
- Issue: Two locations refer to content "in the appendix" (sample-efficiency table;
  singular spectra). There is no `\appendix` block or appendix section in `main.tex` or
  any of the section files. The items are both marked as deferred, so this is unlikely to
  confuse a careful reader, but the word "appendix" implies the content is present.
- Fix: Rephrase to "deferred to camera-ready" or "omitted here for space; available in the
  full version."

**M2. Bib entry type mismatch: `schmidhuber1992fast` uses `@inproceedings` but has journal
fields (`volume`, `number`, `pages`, `booktitle = {Neural Computation}`)**

- File: `refs.bib`, entry `schmidhuber1992fast`
- Issue: *Neural Computation* is a journal; the entry should be `@article` with
  `journal = {Neural Computation}`. Using `@inproceedings` with journal-style fields will
  not break compilation but may produce unexpected formatting.
- Fix: Change to `@article`, rename `booktitle` to `journal`.

**M3. Bib entries `arora2019implicit` and `razin2020implicit` use `@article` but venue
is NeurIPS (a conference)**

- File: `refs.bib`, entries `arora2019implicit` and `razin2020implicit`
- Issue: NeurIPS is a conference; the correct type is `@inproceedings` with `booktitle`.
  `kobayashi2024weightdecay` has the same issue. These will compile fine but are
  inconsistent with the other NeurIPS entries (`gunasekar2017implicit`,
  `zhu2025superposition`, `li2024optimal`) which use `@inproceedings`.
- Fix: Change to `@inproceedings`, rename `journal` to `booktitle` for these three entries.

**M4. `fig4_depth_sweep.pdf` exists in `figures/` but is not included in the paper**

- The depth-sweep section (`04_depth_scale.tex`) states the full sweep is deferred.
  `fig4_depth_sweep.pdf` is present in `figures/` but never referenced by any
  `\includegraphics`. This is not a compile error, but the orphaned PDF in the submission
  bundle may confuse a reviewer who opens the archive.
- Fix: Either include the figure with appropriate label/caption (if the figure content is
  useful even as preliminary), or move `fig4_depth_sweep.pdf` out of the submission bundle.

**M5. Reproducibility section references `fig:depth` in a narrative context that implies
the figure contains reported results**

- File: `sections/09_reproducibility.tex`, line 49
- Text: `Figs.~\ref{fig:depth} and~\ref{fig:scale} from the vanilla SFT and vanilla CODI
  scripts in the same release.`
- Issue: This is the same root cause as C1, but the framing ("reported results trace to a
  specific checkpoint") implies `fig:depth` contains reported numbers. Since the depth sweep
  was explicitly flagged as incomplete in §4, the sentence overclaims. The fix for C1 should
  also reconcile this sentence to match what §4 actually says (single data point, deferred).

---

## Counts

- **Body word count** (`sections/*.tex` total, raw): 7,109 (includes LaTeX markup); stripped
  estimate ~4,850 prose words. Within the 4,800–5,600 ideal band for 8-page two-column ICML.
  Not over the 6,500 risk threshold.
- **Figures referenced in text**: 6 (`fig:seed-decoupling`, `fig:probe`, `fig:scale`,
  `fig:pc`, `fig:pc-negctrl`, `fig:depth`). **PDFs found**: 6
  (`fig1_rank_curves.pdf`, `fig2_seed_decoupling.pdf`, `fig3_scale_sweep.pdf`,
  `fig4_depth_sweep.pdf`, `fig5_linear_probe.pdf`, `fig6_negative_control.pdf`).
  Note: `fig4_depth_sweep.pdf` is present but **not included** in the paper body;
  `fig:depth` is referenced but **not defined** — these are two sides of the same problem
  (C1 above).
- **Citations**: 22 in text, 22 in bib. **Orphans in either direction**: none. All 22 bib
  keys are cited; all 22 cited keys resolve to bib entries.
- **Cross-refs**: 15 label references. **Broken**: 1 (`fig:depth` — no matching `\label`).
  Orphan labels (defined but not `\ref`'d): 5 section labels
  (`sec:intro`, `sec:background`, `sec:depth-scale`, `sec:conclusion`, `sec:repro`) — these
  are standard section labels not requiring explicit back-references; not a problem.
- **Anonymization matches**: none. No author names, institutional identifiers, or
  funding language found.
- **Banned words found**: none. `actually`, `obviously`, `clearly`, `frankly`,
  `essentially`, `wildly`, `literally`, `parsimonious`, `cleanest`, `sharpest`,
  `interestingly`, `nicely`, `honestly`, `simply`, `just` — all zero matches.
