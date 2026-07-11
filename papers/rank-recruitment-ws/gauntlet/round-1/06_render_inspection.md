# Render Inspection — round 1, pass 1 (`06_render_inspection.md`)

Paper: `papers/rank-recruitment-ws/` — "When the Gradient Sees Rank" (NeurReps 2026
Extended Abstract track). Inspected the compiled render (page images), not the source.

## Summary

- **Pages inspected: 7 of 7** (the document's full page count).
- **Counts: 0 critical / 1 serious / 2 minor.**
- **Does the render block accept-ready?** No CRITICAL findings — the CRITICAL
  termination gate is NOT tripped. Body respects the 4-page limit, all figures
  render, all references resolve (no `?`), anonymization holds, no orphans/widows.
  The one SERIOUS + two MINOR findings are figure-polish items (legend/data
  overlap and in-image titles in Figure 2), all confined to page 3. They do not
  make any content unreadable but should be fixed before submission.

## Critical

None.

## Serious

- **[S1] Page 3, Figure 2 (both panels): in-plot legends collide with the
  plotted data.** Verified by zooming both panels to print scale.
  - *Left panel:* the legend rows "predicted (eigenspectrum only)" and
    "measured" sit inside the axes; the data curves cross through the label text
    — the orange square markers and the black curve run over "eigenspectrum
    only" near the top of the panel.
  - *Right panel:* the "rank-sufficient (4/5 seeds)" legend row sits directly on
    top of the flat blue line at rec@0.9 = 1.0, and the "force-rank K−1" /
    "still-transitioning seed" rows overlap the orange-dashed and gray-dotted
    series markers.
  - Every series remains individually traceable (each has a distinct
    color + marker + linestyle, so no value is actually lost), but the legend
    text sitting on the data is visible and crowds the deliberately
    reduced-height panels — exactly the "legends do not cover data" check the
    venue note called out for the page-fit pass. **Fix:** move both legends
    outside the axes (or into a genuinely empty region of each panel), or give
    the panels more height, so labels no longer overlap the data.

## Minor

- **[M1] Page 3, Figure 2 (both panels): descriptive claim-titles are baked into
  the figure image.** The left panel carries "force-rank k = 7 = K−1: spectrum
  predicts decay" and the right panel "exact-match recovery collapses under
  depth" as pixel text inside the plot. Per the ml-paper style rule the caption
  carries the title, not the figure, and these duplicate what the Figure 2
  caption already says (Left/Right descriptions). **Fix:** drop the in-image
  panel titles and rely on the caption, or reduce them to neutral condition
  labels. (Figure 1's "d = 8, K = 4" style labels are legitimate subplot
  *condition* identifiers, not titles — leave those.)

- **[M2] Page 3, Figure 1 (all three panels): the "k = K" dashed-line annotation
  is crowded against the panel title.** The small gray "k = K" text beside each
  dashed vertical marker nearly touches the panel condition title above it
  ("d = 8, K = 4", "d = 16, K = 8", "d = 16, K = 12"). Legible but tight.
  **Fix:** nudge the annotation down or away from the title text.

## Other checks (all clear)

- **Page limit:** body ends at the bottom of page 4 (Section 6 "Related Work,
  Limitations, and Outlook" closes there). Page 5 begins "Appendix A"; pages 5–7
  are Appendices A/B/C and References only. 4-page body limit respected.
- **References:** every in-text citation resolves to a printed bibliography entry
  (Anderson 1972, Barnfield et al. 2026, Dziri et al. 2023, Gozeten et al. 2025,
  Grazzi et al. 2025, Hao et al. 2024, Kohonen 1972, Larson 2026, Mezzadri 2007,
  Mishra et al. 2026, Nazari & Rusch 2026, Nichani et al. 2025, Schlag et al.
  2021, Siems et al. 2025, Sun et al. 2026, Wang et al. 2024, Yang et al. 2024,
  Zhu et al. 2025). No `?` markers anywhere.
- **Anonymization:** no author block on page 1 (only "Editors: Under Review for
  the NeurReps Workshop"); Appendix C uses `anonymous.4open.science` +
  "(anonymized for review)"; running header is the title only. The
  "Sam Larson / Larson (2026)" third-person citation (pages 1, 2, 7) is the
  documented double-blind self-citation exception. The "Under Review - Extended
  Abstract Track" side watermark and faded "…ed Abstrac…" template stamp are the
  intentional draftwatermark — not flagged.
- **Colorblind-safe:** holds. Every multi-series panel distinguishes series by
  marker + linestyle, not color alone — Fig 2 left: black solid circle
  (predicted) vs orange dashed square (measured); Fig 2 right: blue solid
  triangle vs orange dashed square vs gray dotted ×.
- **Captions adjacent:** Fig 1 and Fig 2 captions sit directly beneath their
  figures on page 3; Table 1 / Table 2 captions beneath their tables on page 5.
- **Figure legibility:** axis labels, tick labels, and legend text are legible at
  print size relative to the ~10pt body; no clipping of axis labels or legends at
  panel edges.
- **Orphans/widows:** no section heading is stranded at a page bottom; the
  page-2→3 and page-3→4 paragraph breaks each carry multiple lines on both sides.

## Pages inspected

Pages 1, 2, 3, 4, 5, 6, 7 — all 7 pages of the document (page-1.png through
page-7.png), read in order.

## Verdict

**FAIL (3 findings)** — 0 critical / 1 serious / 2 minor. No CRITICAL, so
accept-ready is not blocked by the termination rule; the SERIOUS legend/data
overlap and two MINOR figure-polish items (all on page 3, Figure 2 / Figure 1)
should be fixed and re-rendered for a clean pass.
