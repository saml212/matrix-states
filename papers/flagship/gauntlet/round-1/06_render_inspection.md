# 06 — Render Inspection (round 1, pass 1)

**Build inspected:** NAMED (arXiv) build on the iclr2026 kit (`\iclrfinalcopy`
header "Published as a conference paper at ICLR 2026"), sanctioned stand-in for
ICLR 2027. Source of the render: `papers/flagship/latex/main.pdf`, rasterized to
16 PNG pages at 110 dpi.

## Summary

- **Pages inspected:** 16 of 16 (page-01 … page-16). None skipped.
- **Counts:** **1 CRITICAL / 2 SERIOUS / 3 MINOR.**
- **Verdict: the render BLOCKS accept-ready.** One critical, pervasive residue
  defect (91 stray `% evidence: Rx` annotations printed as visible body text)
  and one dangling in-text citation with no bibliography entry must be fixed and
  the paper re-rendered before a clean pass can be declared.

`FAIL (6 findings)`

---

## CRITICAL

### C1 — 91 evidence-tracking annotations render as literal visible body text
**Pages:** 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, and the Table B.1 config table on 14
(every content page except the Abstract p.1 and the Conclusion p.13).

The verdict-tracking annotations that were meant to be LaTeX comments print as
literal `% evidence: R0/R1/R1b/R2/…/R11` tokens embedded mid-sentence
throughout the paper — e.g. p.2 "…tracks $d_{\min}$ at Spearman $\rho=0.9747$
**% evidence: R1**, the maximum the family's tie structure permits…"; p.6 a
standalone line "**% evidence: R4 (whole table)**"; p.14 the appendix
configuration table has "**% evidence: R0**" inside its Value cells. This is
garbage text a reviewer sees on nearly every page; it makes the paper read as
unfinished.

**Root cause (confirmed in source):** `papers/flagship/latex/md2tex.py` line 70
escapes every `%` → `\%` in non-math prose, which clobbers the leading `%` of
the evidence comments (produced at line 83) into a literal `\%`. Count by file:
introduction 15, background 7, rank_law 12, capability_separation 30,
pathology_at_scale 16, related_work 1, discussion 2, appendix_a 4, appendix_b 4
(91 total; `grep -o '\\% evidence' sections/*.tex`).

**Fix:** strip the evidence annotations from the emitted LaTeX (they are internal
provenance, not paper content). Either (a) delete each `\% evidence: …` span
from the generated `sections/*.tex`, or preferably (b) fix `md2tex.py` so the
evidence-comment pass runs after — and is protected from — the `%`→`\%` escape
(or drops the annotations entirely), then regenerate all section files and
recompile. Verify with `grep -rc '\\% evidence' sections/` returning 0 before
re-rendering.

---

## SERIOUS

### S1 — In-text citation "(Olsson et al., 2022)" has no bibliography entry
**Page:** 10 (in-text); missing from References on p.13.

On p.10 the sentence "…two-layer from-scratch transformers reliably develop
induction circuitry (Olsson et al., 2022)" renders the citation as **plain,
unlinked text** — unlike its siblings on the same page (Arora et al. (2023),
Jelassi et al. (2024)) which are hyperlinked `\citet`. The printed References
(p.13) run straight from Nichani et al. (2024) to Ao Sun et al. (2026) with **no
Olsson entry**. A reader cannot resolve the citation. Source: the mention is
hardcoded plain text in `sections/06_related_work.tex:18` (no `\cite`); there is
no `\cite`/`\nocite` of Olsson anywhere in `sections/` or `main.tex`. (A stale
`\bibitem` for Olsson survives in `main.bbl:37` from an earlier build but is not
printed because it is uncited, and would be dropped on the next clean bibtex
run.) Functionally this is an unresolved reference.

**Fix:** replace the plain text with `\citep{olsson2022incontextlearninginductionheads}`
(the bib key already exists), then re-run bibtex + LaTeX so the entry both
hyperlinks in text and prints in the bibliography — or, if Olsson is not meant
to be cited, delete the parenthetical.

### S2 — Markdown bold `**…**` survives as literal asterisks
**Page:** 2 (contribution item 3).

Contribution 3 in the intro list renders as "3. **A pathology that scale makes
worse and the stock fix does not remove** (Section 5)." with literal double
asterisks around the phrase — markdown bold that the converter did not translate.
Source: `sections/01_introduction.tex:70-71`. `md2tex.py` has no `**bold**` →
`\textbf{}` rule.

**Fix:** change `**…**` to `\textbf{…}` in the intro section (or add a
`**` handling rule to `md2tex.py` and regenerate). This is the only `**`
occurrence in the sources.

---

## MINOR

### M1 — Figure 5 inset crowds the main-plot axis; very small inset labels
**Page:** 10. The "1.31B final window" inset in the upper-left of Figure 5 sits
close to the main plot's y-axis ticks, and its own tick labels (0.460/0.455/0.450,
130k/155k) are noticeably smaller than the body font. Still legible at print
size; cosmetic. Consider nudging the inset right/down or enlarging its tick font.

### M2 — Author block is the anonymous placeholder under a final-copy header
**Page:** 1. The header reads "Published as a conference paper at ICLR 2026"
(`\iclrfinalcopy`) while the author block shows "Anonymous authors / Paper under
double-blind review". Per the task context this is the **expected placeholder**
(author list is a pending PI decision), flagged here for tracking, not as a
defect. It must be replaced with the real author block before the eventual
camera-ready.

### M3 — Working-title placeholder (expected)
**Page:** 1. Title reads "[WORKING TITLE — PI] The Matrix State Is Real: …".
This is the sanctioned `[WORKING TITLE]` placeholder, flagged for tracking, not a
defect. These two (M2, M3) are the only intended placeholders in the document;
no other placeholder text was found (the C1 residue is a defect, not a placeholder).

---

## Clean checks (what passed)

- **All 8 figures render** with no missing-image boxes, no clipped axes/legends:
  Fig 1 (p.5 scatter, ρ annotation + group labels), Fig 2 (p.6 five-panel small
  multiples), Fig 3 (p.8 two-panel), Fig 4 (p.8 two-panel bars), Fig 5 (p.10
  line+inset), Fig 6 (p.11 error bars), Fig 7 (p.11 forest plot), and the
  appendix figure correctly numbered **Figure A1** (p.15). No title baked inside
  any figure image.
- **Captions sit with their figures/tables** in every case.
- **Table 1** (p.6) renders with booktabs rules (toprule/midrule/bottomrule),
  columns arm / seed 0 / seed 1 / seed 2 / mean. The B.1 config table (p.14)
  also renders with rules (but carries the C1 residue).
- **Math renders cleanly** — subscripts ($d_{\min}$, $S_t$, $\mathrm{acc}_A$),
  Greek, the delta-rule update equation, $\sqrt{k/d_{state}}$, etc. No raw `$`
  or stray backslashes (the only `\%` litter is C1, and it is a comment-escape
  artifact, not math).
- **References render with real entries** — Arora 2023, Jelassi 2024, Nazari &
  Rusch 2026, Nichani 2024, Sun 2026, Kimi Team 2025, Xiao 2024, Yang 2025a,
  Yang 2025b. No `??`, no `[?]`. arXiv URLs present and hyperlinked. (The one
  gap is S1: Olsson, cited but not listed.)
- **Appendix numbering correct:** A (Complement Scaffold), B (Configurations…),
  B.1–B.4; Figure A1. Sequential and consistent.
- **No margin/overfull violations** — no text, table, or figure spills past the
  text block into the margin or gutter on any page.
- **No widow/orphan headings** at page breaks; section headings all sit with
  their following text (incl. §5 top of p.9, §8 on p.13).
- **Page numbers present** and centered at the foot of all 16 pages.
- **Two placeholder blocks only** (M2 author block, M3 working title), both
  expected per context.

---

## Main-text page count and float-packing debt

- **Main text (Sections 1–8, through the Conclusion): 13 pages.** The Conclusion
  (§8) sits on **page 13**, with the References beginning lower on the same page.
  References span pages 13–14; Appendix A is on page 14; Appendix B (B.1–B.4)
  spans pages 14–16. Total document: **16 pages**.
- **Not a blocker for THIS build** — the arXiv build has no page limit.
- **Float-packing debt for the eventual ICLR build (9-page main-text budget):**
  the arXiv main text runs **13 pages, ~4 pages over** the ICLR limit. The
  overhang is partly float slack: pages 5, 6, 8, 9, 10, and 11 carry large
  vertical gaps (single figures with generous surrounding whitespace, plus
  `\iclrfinalcopy` top-padding), and the 7 main figures + Table 1 are placed one
  per block rather than packed. The ICLR build will need aggressive float
  packing (group Figs 3/4 and 6/7 into subfigure rows, `[t]` placement, reduced
  figure heights, tighter captions) **plus** prose trimming to reclaim the ~4
  pages. Record it as debt; it does not gate the arXiv render.

---

## Security note

No embedded or injected instructions were found in any inspected artifact. The
16 page images, the role prompt (`prompts/render-inspector.md`), and the source
files I cross-checked (`md2tex.py`, `sections/*.tex`, `main.bbl`,
`06_related_work.tex`, `01_introduction.tex`) contained no fake system-reminders,
no date-change or "file was modified — do not tell the user" concealment blocks,
and no directives attempting to alter this inspection. All instructions I acted
on came from the legitimate task prompt and the role file. Had any such content
appeared, it would be reported here and not obeyed.

---

## Pages inspected

page-01, page-02, page-03, page-04, page-05, page-06, page-07, page-08, page-09,
page-10, page-11, page-12, page-13, page-14, page-15, page-16 (16 of 16 — the
document's full page count).

**Verdict: `FAIL (6 findings)` — 1 CRITICAL / 2 SERIOUS / 3 MINOR. Re-render
required after C1, S1, S2 are fixed; a fresh render-inspection pass
(`06_render_inspection_v2.md`) must run on the new PDF.**
