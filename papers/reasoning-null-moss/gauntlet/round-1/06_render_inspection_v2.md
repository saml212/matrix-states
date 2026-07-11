# Render inspection (pass v2) — reasoning-null-moss, MOSS @ COLM 2026

## Summary

Pages inspected: **8 of 8** (equals the document's total page count). Every page
read in order as a rendered image (150 dpi provided render, cross-checked with
300 dpi and 600 dpi re-renders of the figure pages for print-size legibility and
element-overlap checks).

Counts: **0 critical / 1 serious / 1 minor.**

The render is **not blocked on any CRITICAL ground** — no figure fails to render,
no page-limit violation, no unresolved reference, no margin violation, no
anonymization leak. It is **not yet a clean PASS**: one Serious annotation/marker
overlap in Figure 3 (right panel) visibly weakens the figure and should be fixed,
plus one Minor color-only series-distinguishability note. Per the pass/fail rule
(PASS only on zero findings), this pass is a FAIL pending the fixes below.

Structure verified against the venue rule (4 pp main content; refs + appendices
excluded): main content ends on **page 4** (Section 7 "Scope of the Bounds"
finishes on p4). Pages 5–6 = References + Appendix A; pages 7–8 = Appendix B
(Supplementary Figures) with Figures 1–3 + Appendix C (Reproducibility). **Page
limit respected.**

## Critical

None.

## Serious

1. **Page 8, Figure 3 (right panel) — annotation text overlaps data markers.**
   The gray annotation `naive n=12 pool` / `(not decision-grade)` is printed on
   top of the orange square markers of the "new n=9" cohort. At print size the
   collision obscures the final `l` of "pool" (merged into a square outline) and
   the closing `)` of "(not decision-grade)" (a square sits across the `e)`).
   The annotation remains mostly readable and the data markers are still
   visible, but the overlap is conspicuous and reads as a layout defect — it is
   exactly the "no element overlapping another (annotation vs. data)" failure the
   inspection checks for. Fix: nudge the annotation left/up into the empty region
   above the archived cohort (or shorten it) so no glyph intersects an orange
   square. Confirmed at 600 dpi (crop of the right panel), not merely at page
   scale.

## Minor

1. **Page 7 Figure 2 (left panel) and Page 8 Figure 3 (left panel) — two series
   distinguished by hue only.** In Fig 2-left the `openr1 cells` (blue) vs
   `wikitext cells` (orange) series are both solid lines with identical circle
   markers; in Fig 3-left the `K=32` (blue) vs `K=20` (orange) points are both
   circles. The blue/orange pair is the recommended colorblind-safe palette and
   survives the common (deutan/protan) forms, so this is not a hard
   accessibility failure — but the series are not distinguishable "without color
   alone" (e.g. under a grayscale print path both collapse to similar mid-grays).
   Optional hardening: give one series a dashed line style / distinct marker
   shape. Note Fig 1 and Fig 3-right already do the right thing (marker shape
   encodes the series), so this is a consistency gap, not a correctness bug.

## What was checked and passed

- **Figures render cleanly.** Fig 1 (both panels), Fig 2 (both panels), Fig 3
  (left panel): no clipping, no cut-off axes, no baked-in titles (captions carry
  the titles). Axis labels and legends legible at print size vs body text.
- **Fig 1 series are shape-encoded** (circles / squares / triangles / diamonds)
  and stay distinguishable independent of color; the "fail region: median ≤ null
  p95 (every point)" text sits in empty plot space, no data overlap.
- **Fig 2 right panel** ("all 30 readings = 0.0", green ×, `h=1 absolute validity
  floor (0.10)` dashed line) — no overlaps; floor label sits above the line in
  clear space.
- **Fig 3 left panel** — the `c=2500, K=32: CI excludes 0 (harm direction)`
  annotation sits in empty lower-left space with a clean leader line to the
  anchor point; no overlap.
- **Captions adjacent** to their figures (Fig 1 & Fig 2 captions on p7 directly
  below each figure; Fig 3 caption on p8 below the figure).
- **All citations resolve** — Schlag et al. 2021, Yang et al. 2024, Merity et al.
  2017, Nichani et al. 2024, Arora et al. 2023/2024, Okpekpe & Orvieto 2025,
  Grazzi et al. 2024, Merrill et al. 2024, Bertinetto et al. 2021, Forde et al.
  2020 — all render as blue links; **no `?` / `??` markers** anywhere; Table 1
  and Figures 1–3 cross-references resolve; bibliography present and formatted.
- **Anonymization holds on the page.** Title block reads "Anonymous authors /
  Paper under double-blind review"; running header "Under review as a conference
  paper at COLM 2026" on every page; no author names, affiliations, or
  identifying URLs in body or inside any figure's pixels; Appendix C promises
  artifact release "with the camera-ready version" (no live identifying link).
- **No margin violations / overfull boxes** visible on any page; single-column
  COLM template; submission line numbers present (1–271).
- **No orphaned headings or widowed lines** at page breaks: Section 4 heading
  (p3) is immediately followed by its content on the same page; Sections 5/6/7
  (p4), References + Appendix A (p5), Appendix B (p7), Appendix C (p8) all carry
  content directly under their headings; all page-crossing paragraphs break with
  ≥2 lines on each side.

## Pages inspected

Pages 1, 2, 3, 4, 5, 6, 7, 8 (all 8 pages; none skipped). Figure detail on pages
7–8 additionally verified via 300 dpi full-page and 600 dpi panel-crop
re-renders.

## Verdict

FAIL (2 findings)
