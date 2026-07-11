# Render Inspection v3 — Three Bounds on a Null (MOSS @ COLM 2026)

## Summary

- **Pages inspected: 8 of 8** (entire document, read in order, from the rendered
  page images at 150 dpi; pages 7–8 re-rendered at 300 dpi to verify figure text
  legibility and annotation/marker overlap).
- **Counts: 0 critical / 0 serious / 0 minor.**
- **The render does not block accept-ready.** Main content ends by page 4, all
  citations resolve, all three figures render cleanly with shape-distinguishable
  series and adjacent captions, and no author identity survives into any rendered
  page or figure pixel.

Verdict basis, checked page by page:

- **Page limit.** Main content is pages 1–4. Section 7 ("Scope of the Bounds")
  ends on page 4 at line 176; References begin on page 5. Exactly 4 pages of main
  content — within the venue's 4-page cap. References and appendices (pages 5–8)
  do not count. No spill onto page 5.
- **Citations.** No `??` markers anywhere. Every in-text `\cite` resolves
  (Schlag et al. 2021; Yang et al. 2024; Merity et al. 2017; Nichani et al. 2024;
  Grazzi et al. 2024; Merrill et al. 2024; Arora et al. 2023/2024; Okpekpe &
  Orvieto 2025; Bertinetto et al. 2021; Forde et al. 2020) and each appears as a
  printed entry in the References (page 5). Cross-refs resolve too: Table 1,
  Figures 1/2/3, and Appendix A/B all render with correct numbers/letters.
- **Anonymization.** Header on every page is the anonymized COLM boilerplate;
  page 1 shows "Anonymous authors / Paper under double-blind review". No author
  names, affiliations, or identifying URLs on any page. Figure pixels carry only
  generic labels (Phase 1/1b, Rung 3, Phase 2, openr1/wikitext, K=32/K=20,
  archived/new) — no file paths, hostnames, usernames, or institutional strings
  baked in. Reproducibility (Appendix C) defers artifact release to
  "camera-ready version" with no identifying link. Anonymization holds visually.
- **Figures render cleanly.**
  - *Figure 1* (page 7): two-panel scatter, four series distinguished by **marker
    shape** — circle / square / triangle / diamond — so they read without relying
    on color. Legend above the panels is clear; panel identifiers "(iii) query-key
    alignment" and "(iv) key-value alignment" are panel labels, not a baked-in
    figure title (the caption carries the title). The gray "fail region: median ≤
    null p95 (every point)" annotation sits in empty lower-right space in each
    panel, not over any marker. Dashed identity line renders. Axis labels and tick
    labels legible at print size. Caption directly beneath.
  - *Figure 2* (page 7): left panel series (openr1 = blue solid + circles;
    wikitext = orange dashed + squares) distinguished by **linestyle and marker
    shape** in addition to color; legend sits in the empty lower-left corner with
    no data marker over its text. Right panel green "×" markers all at 0.0 with the
    "h=1 absolute validity floor (0.10)" dashed reference line labeled clearly.
    Both panels legible; caption adjacent.
  - *Figure 3* (page 8): left panel K=32 (blue circles) vs K=20 (orange squares)
    with error-bar caps; the "c=2500, K=32: CI excludes 0 (harm direction)"
    callout and its leader line sit in empty space and point to the correct marker
    without overlapping any cap or the rotated y-axis label. Right panel
    archived-n=3 vs new-n=9 with the gray dotted pooled interval and the "naive
    n=12 pool (not decision-grade)" callout, leader line clear of the markers.
    Caption adjacent; Appendix C follows.
- **Captions adjacency.** Table 1 (page 2) and Figures 1–3 each sit immediately
  above their captions; no float drifted from its caption.
- **Margins / overfull boxes.** No text, table, or figure spills into the margin
  or gutter on any page; inline math and interval notation stay within the text
  block. Table 1 fits within the text width.
- **Widows / orphans.** No section heading is stranded at a page bottom (Section 3
  at the foot of page 2 keeps three lines of body beneath it; page-4 body ends on a
  complete sentence). No single line is widowed at a page top or bottom across any
  break.
- **Line numbers.** Present in submission mode throughout (1–271), including on the
  figure/appendix pages.

## Critical

None.

## Serious

None.

## Minor

None.

## Observations (non-blocking, not counted as findings)

- **Page header wording.** Every page carries "Under review as a conference paper
  at COLM 2026" — the COLM 2026 template's default submission header. This is the
  correct anonymized boilerplate and not a render defect or an identity leak, but
  the word "conference" (vs "workshop") is the template default rather than a
  MOSS-specific string. Whether to customize it is a format/content decision owned
  by the format auditor, not a visual-render issue; flagged here only so the writer
  is aware. It does not affect the verdict.

## Pages inspected

Pages 1, 2, 3, 4, 5, 6, 7, 8 (all 8 of 8; pages 7 and 8 additionally re-rendered
and read at 300 dpi).

---

**PASS**
