# Render Inspection — neurreps-ea-submission.pdf (round 2, pass 1)

Fresh-context visual inspection of the compiled PDF render (not the LaTeX
source). Inspected the pre-rendered page-1..page-8 PNGs plus targeted
high-DPI re-renders (300/400/600/1200 dpi crops via `pdftoppm`) of the
figures, legends, watermark/body-text overlap regions, and inline math on
pages 1, 2, 3, and 4.

## Summary

- **Pages inspected: 8 of 8** (all pages read in order; none skipped).
- **Counts: 0 critical / 0 serious / 0 minor.**
- **The render is accept-ready** as far as this gate is concerned — no
  finding blocks accept-ready.

## Critical

None.

## Serious

None.

## Minor

None.

## Detailed checks performed (all clean)

- **Page limit:** Body (Sections 1–5, including "Limitations") spans pages
  1–4 exactly; Appendices A–E occupy pages 5–6 (spilling to the top of
  page 7), and References occupy the rest of page 7 through page 8. Against
  the venue's 4pp-excluding-references-and-appendices limit
  (`VENUE_REQUIREMENTS.md`), the body is exactly at the limit, not over it.
- **Anonymization:** No author block on page 1 (correct for the review
  build); "Editors: Under Review for the NeurReps Workshop" and the running
  headers ("Under Review - Extended Abstract Track 1–8, 2026" /
  "Symmetry and Geometry in Neural Representations" / "A Causal Rank Law for
  Matrix Memories") are template/venue/title strings, not identity leaks.
  Footer "© 2026 ." is blank after the year, as expected pre-camera-ready.
  No name, org, or de-anonymizing URL found on any of the 8 pages, in any
  figure image, or in the reference list.
- **Draftwatermark:** Present on every page as expected/required by the
  venue template; checked at 400 dpi directly over body text (page 1 title,
  page 2 top paragraph) — the watermark is light enough that body text stays
  fully legible everywhere it overlaps. Not flagged, per instructions.
- **Figures (Figure 1, page 4; Figure 2, page 4):** No clipping, no
  overlapping elements, no title baked into either figure image (the
  "Figure 1:"/"Figure 2:" titles are LaTeX captions, not pixels inside the
  plots). Axis labels, tick labels, and in-figure legend text are legible at
  print size and comparable to body-text weight. Checked Figure 1's raster
  content at 1200 dpi (a level far beyond print) — edges stayed clean, no
  pixelation/upscale artifacting.
- **Encoding claims vs. rendered marks:** Figure 1's legend claims
  filled-marker = solvable, open-marker = non-solvable; verified at
  600 dpi zoom that S3 (filled circle) and S4 (filled square) are indeed
  solid-filled, and A5 (open triangle), S5 (open diamond), A6 (open inverted
  triangle) are indeed unfilled outlines — the claim holds visually, group
  labels and distinct marker shapes/colors provide redundant (non-color-only)
  encoding. Figure 2's legend claims dashed = unconstrained anchor, dotted =
  0.9×anchor bar; verified at 600 dpi that the dashed and dotted reference
  lines are visually distinct patterns in the S5/A6 subplots. Figure 2's S3
  panel caption claims "overlays its four seeds (thin lines; bold mean)" —
  confirmed at 600 dpi: four thin light-blue seed traces plus one bold mean
  trace are all visible and distinguishable.
- **Captions:** Figure 1, Figure 2, Table 1, Table 2, and Table 3 captions
  all sit immediately adjacent to their figure/table on the same page — no
  drifted floats.
- **Margins / overfull boxes:** No visible spill into margins anywhere,
  including the razor table + Figure 1 side-by-side minipage layout on
  page 4 and the inline square-root expressions
  (`√(d_min−1)/d_min ≤ 0.894`) on pages 2 and 3 — both checked at 300 dpi
  full-page render; text wraps cleanly within the text block.
  Figure 2's 5-subplot row fits within the margins with no clipping.
- **References:** All in-text citations (checked a sample: Liu et al. 2023,
  Nichani et al. 2025, Chughtai et al. 2023, Nazari and Rusch 2026, and the
  full Related-Work paragraph's citation list) resolve to a printed entry in
  the bibliography on pages 7–8. No `?` / unresolved `\cite` or `\ref`
  markers found anywhere in the 8 pages.
- **Orphaned headings / widows:** Checked every page-break boundary
  (1→2, 2→3, 3→4, 4→5, 5→6, 6→7, 7→8). Every section/appendix heading that
  starts near a page boundary (e.g., "Appendix E" at the bottom of page 6)
  is followed by substantial body content on the same page, not stranded
  alone. Mid-paragraph page breaks (1→2, 5→6, 6→7) are ordinary sentence
  continuations, not single stranded lines.
- **Colorblind-safe / non-color encoding:** Both figures rely on shape
  (Figure 1: circle/square/triangle-up/diamond/triangle-down) and line
  style (Figure 2: dashed vs. dotted) in addition to color, so the encoding
  does not depend on color discrimination alone.

## Pages inspected

1, 2, 3, 4, 5, 6, 7, 8 (all 8, in order; full-resolution re-renders were
also pulled for pages 1–4 to check figures, watermark/text overlap, and
inline math at higher DPI).

## SECURITY NOTE

No fake or anomalous `<system-reminder>`-style content, concealment
instructions, or injected directives were observed in any tool stdout
during this inspection (bash/pdftoppm output was plain file listings and
byte counts only). Nothing to report.

## Verdict

PASS
