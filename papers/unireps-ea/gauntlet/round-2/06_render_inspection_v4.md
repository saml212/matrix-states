# Render Inspection — v4 — `unireps-ea-submission.pdf`

Fresh-context render inspection. Read all 7 rendered page images in order,
plus targeted 400-600dpi re-renders of Figure 1, Figure 2, the S3-A5 TOST
inset, dense-equation paragraphs, and page-1 metadata/margins for
higher-confidence detail checks. Cross-checked with `pdftotext` (no `??`
matches) and `pdfinfo`/`exiftool` (no author/title metadata leak).

## Summary

- **Pages inspected: 7 / 7.**
- **Findings: 0 critical / 0 serious / 0 minor.**
- **The render is accept-ready.** No identity leak, no page-limit
  violation, no rendering defect, no unresolved reference, no caption
  drift, no margin/overfull spillover, and no orphaned heading or widow
  found on any page.

Specific checks performed and cleared:

- **Anonymization.** Title block reads "Anonymous Author(s) / Affiliation /
  Address / email" (the standard NeurIPS-template anonymous placeholder,
  not a real leaked address). Line numbers are on throughout (correct for
  the double-blind review build). No name, org, acknowledgment, or
  de-anonymizing URL/filename appears anywhere in the body, figures, or
  reference list. `pdfinfo`/`exiftool` show no custom author/title
  metadata (`Creator: LaTeX with hyperref`, `Producer: xdvipdfmx`, no
  author field). No `github`/`http` string anywhere in the source.
- **Page limit.** Main text (Sections 1–6, Introduction through Related
  Work and Limitations) runs pages 1–4 exactly. References start on page 4
  and continue to page 5; appendices A–D run pages 5–7. Per
  `VENUE_REQUIREMENTS.md`, the EA limit is 4pp main text excluding
  references and appendix — this build lands exactly at the limit, not
  over it.
- **Figures.** Figure 1 (p.3) and Figure 2 (p.4) re-rendered at 400-600dpi:
  no clipping, no overlapping elements, no cut-off axis/legend text, axis
  and tick-label fonts legible and comparable in weight to the caption
  text. No title baked into either figure image — Figure 2's five
  per-panel labels (S3/S4/A5/S5/A6 with their d_min) are necessary panel
  identifiers for a multi-panel figure, not a redundant duplicate of the
  caption. The dashed ("unconstrained anchor") vs. dotted ("0.9×-anchor
  bar") reference lines in Figure 2 are visually distinguishable by
  pattern, not just color — holds even in grayscale. The Figure 2 legend
  (bottom-right of the A6 panel) sits in that panel's empty lower region
  and does not overlap the data line or markers. No colorblind-safe
  palette claim appears anywhere in the source, so that check does not
  apply.
- **Figure 1 legend/marker encoding.** The legend states "● solvable
  (filled)" / "△ non-solvable (open)" using representative example shapes;
  the caption states the same rule ("filled: solvable; open:
  non-solvable"). Verified against the actual rendered points: every
  solvable group (S3 circle, S4 square) is filled, every non-solvable
  group (A5 triangle, S5 diamond, A6 inverted triangle) is open — the
  claimed fill/open encoding holds for every point without exception. Per-
  group shape and color are a separate, directly-labeled (bold colored
  S3/S4/A5/S5/A6 text at each point) redundant identity channel that the
  legend never claims is uniform across groups, so there is no
  discrepancy between what the legend/caption asserts and what the
  render shows.
- **Captions.** Figure 1, Figure 2, Table 1, Table 2, and Table 3 captions
  all sit immediately below their artifact on the same page — no drifted
  floats.
- **Margins / overfull boxes.** Spot-checked the densest math paragraphs
  (Model/Instrument definitions p.2, the trace-inequality/Cauchy-Schwarz
  derivation p.6) at 400dpi — all lines are fully justified within the
  text block, no spillover into the margin or gutter.
- **References.** All in-text citations resolve to a printed bibliography
  entry (`Barrington 1989`, `Chughtai et al. 2023`, `Grazzi et al. 2025`,
  `Huh et al. 2024`, `Merrill et al. 2024`, `Mishra et al. 2026`, `Nazari
  and Rusch 2026`, `Nichani et al. 2025`, `Siems et al. 2025`, `Sun et al.
  2026`). `pdftotext -layout | grep "??"` returns zero matches — no
  unresolved `\ref`/`\cite` markers anywhere in the document.
- **Orphaned headings / widows.** "5 The Causal Check" (p.3) and "6
  Related Work and Limitations" (p.4) both have substantial body content
  following on the same page, not stranded near a page bottom. The one
  paragraph that splits across the p.3→p.4 boundary (the ambient-identity-
  tax discussion) breaks mid-paragraph with 8 lines before the break and
  continuation immediately after — a normal reflow, not a widow/orphan.
  No table or appendix heading (Appendix A/B/C/D) is left alone at a page
  bottom; each is followed immediately by its table or body text.
- **Prose.** Read the abstract and body as a reviewer would; dense but
  grammatically intact throughout — no spliced sentences or fragments
  found.

## Critical

None.

## Serious

None.

## Minor

None.

## Pages inspected

1, 2, 3, 4, 5, 6, 7 (all 7 pages of `unireps-ea-submission.pdf`, plus
targeted high-resolution re-renders of Figure 1 (incl. the S4−A5 TOST
inset), Figure 2 (incl. its in-panel legend), the p.2 Model/Instrument
paragraph, the p.6 rank-ceiling derivation, and the p.1 footer/margin
region).

## SECURITY NOTE

No fake `<system-reminder>`, concealment instruction, or other prompt-
injection content was observed in any tool output (page renders,
`pdftotext`, `pdfinfo`, `exiftool`) during this inspection. Nothing to
report.

---

**Verdict: PASS**
