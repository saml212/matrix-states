# Render inspection — pass v2 (fresh context, post-trim)

Target: `papers/reasoning-null-moss/main.pdf` (MOSS @ COLM 2026; 4pp main-content
limit, references + supplementary uncapped; double-blind). Recompiled clean via
`make` (tectonic, 3 passes + BibTeX) before inspection. Read the compiled PDF
page by page with the visual Read tool; right-margin and table-edge regions
cropped at 300 DPI for the overfull-hbox check.

## Summary

- **Pages inspected: 10 of 10** (confirmed `pdfinfo` Pages: 10).
- **0 critical / 0 serious / 0 minor.**
- **The prior CRITICAL is FIXED.** The body (§1–§7) now ends at the bottom of
  page 4: §7 "Scope of the Bounds" — both the *Bounded* and *Not bounded*
  paragraphs — is fully contained on page 4, ending "…unblinded routing kept
  zeros and transients from becoming headlines." **References is the first
  content on page 5**, with only the running header above it; **no §7 (or any
  §1–§7) body text spills onto page 5.** The 4-page main-content limit is met.
- The render is accept-ready on the visual gate.

## Critical

None.

## Serious

None.

## Minor

None.

## Items explicitly checked that are NOT findings (recorded for the reviewer)

- **§3 opening-paragraph overfull hbox (page 2).** The LaTeX log reports
  `Overfull \hbox (8.46pt too wide)` in `sections/03_geometric_null` lines
  8–30 (the "The readout was defective…" paragraph, rendered on page 2). A
  300-DPI crop of that paragraph's right margin shows every justified line
  terminating at the same right boundary with full margin white-space beyond;
  **no glyph protrudes into the margin** — reads clean. Not a finding.
- **Table 1 (top of page 3).** Rightmost column ("Registered outcome"); the
  longest entries — "null-indistinguishable", "probe-invalid, refused",
  "batch-effect-flagged" — sit inside the table's right rule with margin to
  spare. No protrusion. Footnote markers †/‡/§ resolve to the caption legend.
- **Figures 1–3 render cleanly.** Fig 1 (page 9, Appendix B "Supplementary
  Figures"): two panels "(iii) query–key alignment" / "(iv) key–value
  alignment", 4-series legend distinguished by *shape* (circle/triangle/
  square/diamond) as well as color — colorblind-safe holds on the render;
  axis labels legible at body size; no title baked into the image; caption
  adjacent. Fig 1 is referenced from §3 (page 2: "the alignment-premise gates
  (Figure 1) fail at all 320 readings"). Figs 2 and 3 (page 10): two panels
  each, legible axes/annotations, no baked-in title, captions adjacent.
- **Figures live in the supplementary appendix (pages 9–10), not inline.** This
  is deliberate and venue-appropriate for a 4-page-main-content limit with
  uncapped supplementary; every in-text reference explicitly signposts
  "Appendix B", and each caption is adjacent to its own figure. Caption-
  adjacency rule satisfied. Not a finding.
- **No `?`/`??` or "undefined" markers** anywhere — visual scan of all 10 pages
  plus a text-layer grep of the rendered PDF returned none. Every citation
  resolves to a printed References entry (Schlag 2021; Yang 2024; Merity 2017;
  Nichani 2024; Arora 2023; Arora 2024; Okpekpe & Orvieto 2025; Grazzi 2024;
  Merrill 2024; Bertinetto 2021; Forde 2020). All Figure/Table/Section/Appendix
  cross-references print numerals.
- **Anonymization holds on every rendered page.** Header "Under review as a
  conference paper at COLM 2026"; byline "Anonymous authors / Paper under
  double-blind review". No author/organization name, handle, URL, email, or
  filename appears in any figure image, axis label, legend, table cell, or
  caption. Dataset labels ("openr1", "wikitext") are corpus names, not
  identifying. Appendix C defers artifact release "with the camera-ready
  version" — no identity leak.
- **Page-break hygiene.** §6→§7 (page 4): §7 heading + full content sit together
  on page 4 — no orphaned heading. Body→References (page 4→5): References opens
  fresh at page-5 top. References→Appendix A (page 5): flows without orphan.
  §3 spans page 2→3 with Table 1 leading page 3 — natural flow, no stranded
  heading or single-line widow at any page top/bottom. Appendix A flows pages
  5–8 (page 8 trails into white-space before figures begin on page 9 — normal
  float placement, not a widow). The two `Underfull \vbox (badness 10000)`
  page-glue warnings in the log produce no visible spacing defect on the page.

## Pages inspected (literal)

1, 2, 3, 4, 5, 6, 7, 8, 9, 10 — all 10 pages of `main.pdf`, in order.

PASS
