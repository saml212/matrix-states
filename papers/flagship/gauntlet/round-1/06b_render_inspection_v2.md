# Render Inspection — Pass v2 (flagship, NAMED/arXiv build on iclr2026 kit)

**Verdict: FAIL (1 finding) — 0 CRITICAL / 1 SERIOUS / 0 MINOR.**

Fresh cold read of the compiled PDF, all 16 pages (110 dpi PNGs), read in
order. One defect blocks a clean pass: unrendered markdown emphasis
(literal `*...*`) printed in the body text of Section 2.2. Everything else —
all seven main figures plus Figure A1, Table 1 and the two appendix tables,
every in-text citation and its reference entry, appendix numbering, math,
page numbers — renders correctly.

## Main-text page count (informational)

Main text (Sections 1–8, through the Conclusion) runs **pages 1–13**, with
Section 8 (Conclusion) ending near the top of page 13 immediately before the
References. Effective main-text length ≈ 12.5 pages. References occupy p.13;
Appendix A (with Figure A1) is p.14; Appendix B is pp.15–16. Total document:
16 pages. arXiv build — no page limit applies; reported for the record only.

## Expected placeholders (NOT defects — confirmed as flagged in the brief)

- p.1 title renders as "[WORKING TITLE — PI] The Matrix State Is Real: …" —
  the sanctioned working-title placeholder (author list is a pending PI
  decision). Expected.
- p.1 author block "Anonymous authors / Paper under double-blind review" —
  expected placeholder. Expected.

Both render cleanly and are the two expected stand-ins; not counted as
findings.

## SERIOUS

**S1 — pp.2–3: literal markdown emphasis asterisks printed in body text.**
Three single-asterisk emphasis spans survived the markdown→LaTeX conversion
and render as literal `*` characters in the running prose of Section 2.2
("Model Families"):
- p.2: "(i) The **\*matrix-state encoder\*** family (Section 3): …"
- p.2: "(ii) The **\*two-layer delta-rule contender\*** …" (line wraps to p.3)
- p.3 (top): "(iii) **\*Delta-rule language models\*** at scale (Section 5): …"

Verified visually on the render (zoomed crops of p.2 bottom and p.3 top show
the asterisks printed on the page) and against source: all three live in
`sections/02_background_setup.tex` lines 19, 24, 26. LaTeX treats `*` as a
literal in text mode, so each `*word*` prints its asterisks verbatim. A
reviewer sees unrendered markup — the paper reads as an unfinished
conversion at exactly the point it names its three model families.

*Fix:* replace each `*…*` span with proper LaTeX emphasis (`\emph{…}` or
`\textit{…}`) in `sections/02_background_setup.tex` lines 19, 24, 26, then
recompile. This is the only markdown residue in the document (a full source
sweep for `**`, `##`, `<!--`, `` ` ``, and literal "evidence:" strings
returned nothing else).

## MINOR

None.

## Observations (not defects, no action required)

- p.14: Figure A1 floats to the top of the page, above the "A The c*I
  Complement Scaffold" heading it belongs to, with whitespace between the
  caption and the heading. This is standard LaTeX float placement; the
  caption is self-contained and sits adjacent to the figure, so it satisfies
  the caption-adjacency rule. No action needed.
- Figures 1–7 and A1: all render without clipping, overlap, or cut labels;
  series are direct-labeled or legend-distinguished (Fig. 1 uses on-point
  group labels; colorblind-safe by direct labeling). Axis/label fonts are
  legible at print size. Figure 5's "1.31B final window" inset and Fig. 4
  right-panel subscripted tap labels are small but readable.
- Table 1 (p.6) and the Appendix B tables (p.15) use proper booktabs rules
  (top/mid/bottom); no content spills into the margin or gutter on any page.
- References (pp.13–14): all ten entries render with real author lists,
  years, and arXiv URLs. Every in-text citation resolves — Yang et al.
  2025a/b, Nichani et al. 2024, Xiao et al. 2024, Arora et al. 2023,
  Jelassi et al. 2024, Olsson et al. 2022, Nazari & Rusch 2026, Sun et al.
  2026, Team et al. 2025. No `??`, no `[?]`, no dangling author-year
  citation lacking a bibliography entry.
- Appendix numbering correct (A, B; Figure A1). Section cross-refs
  (Sections 2–8, 3.2, 4.1, 5.3, Appendices A/B) and figure/table refs all
  resolve — no broken `\ref`. Inline math renders cleanly throughout
  (delta-rule update, Frobenius/Gram deviations, √(k/d_state), σ-margins).
- Page numbers present and correct on all 16 pages; running header
  "Published as a conference paper at ICLR 2026" on every page.

## Pages inspected

Pages 1–16 (all), read in order. Zoomed crops taken of page-02 (bottom) and
page-03 (top) to confirm S1.

## Security note

No embedded or adversarial instructions were encountered within the rendered
paper content (no injected directives baked into figures, captions, tables,
or body text). Per protocol I would report and disregard — never comply with
— any such instruction; none was present in the artifact under inspection.
(An out-of-band environment "date has changed" system notice appeared in the
tool stream; it is not part of the paper and carried no instruction to act
on. Noted and disregarded for the inspection.)

---

**Verdict: FAIL (1 finding).** One SERIOUS markdown-residue defect (S1)
blocks a clean render. Fix the three `*…*` spans in
`sections/02_background_setup.tex`, recompile, and re-run a fresh
render-inspection pass (v3).
