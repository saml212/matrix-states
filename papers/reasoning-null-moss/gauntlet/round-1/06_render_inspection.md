# Render Inspection — round 1, pass 1

## Summary

Pages inspected: **8 of 8** (full document read, in order, at 150 dpi).

Counts: **0 critical / 0 serious / 2 minor.**

The render does **not** block accept-ready. No critical or serious findings:
the 4-page main-content limit is respected (main text ends on page 4;
References begin on page 5), every in-text citation resolves to a printed
bibliography entry (no `??` markers), all three figures render cleanly with
captions adjacent, line numbers are present in submission mode, and
anonymization holds both in the body and inside the pixels of all three
figures. The two findings below are cosmetic only.

### Structure / limit check (passed)
- **Main content ends on page 4.** Section 7 ("Scope of the Bounds") closes at
  line 176 at the bottom of page 4; References start at the top of page 5. No
  main-content spill onto page 5. 4-page limit respected.
- **Page 5** = References + start of Appendix A (as expected).
- **Pages 6–8** = Appendix A (cont.), B Supplementary Figures, C
  Reproducibility; Figures 1–2 on page 7, Figure 3 on page 8.
- **Citations all resolve.** Schlag et al. 2021, Yang et al. 2024, Merity et al.
  2017, Nichani et al. 2024, Arora et al. 2023 & 2024, Okpekpe & Orvieto 2025,
  Grazzi et al. 2024, Merrill et al. 2024, Bertinetto et al. 2021, Forde et al.
  2020 — every one appears in the References list on page 5. No unresolved refs.
- **Table 1** (page 2) renders within margins; the `†` footnote marker resolves
  to its note in the caption.
- **Anonymization holds visually.** Header reads "Under review as a conference
  paper at COLM 2026"; byline is "Anonymous authors / Paper under double-blind
  review". No name, handle, path, or URL is baked into any of the three figures
  (checked legends, axis labels, panel labels, and in-plot annotations on
  pages 7–8). Dataset names "openr1"/"wikitext" in Figure 2 are not identity
  leaks.
- **Colorblind safety:** no explicit colorblind-safe *claim* appears in the
  captions or the Reproducibility section, so there is no claim to violate.
  Distinguishability nonetheless holds: Figure 1 separates its four series by
  marker shape (circle / square / triangle / diamond); Figure 3-right by shape
  (circle vs square); Figures 2 and 3-left use the blue/orange pairing (a
  deuteran/protan-safe pairing) reinforced by horizontal dodge / separate
  panels.
- **No orphaned headings or widowed lines.** Section 5 opens at the top of
  page 4 with its body following; Section 3 opens near the bottom of page 2 with
  three lines beneath it before the break; all paragraph breaks continue
  cleanly. Numbered list item 3 spans the page 1→2 break only because Table 1
  floated to the top of page 2, which is normal.

## Critical

None.

## Serious

None.

## Minor

1. **Page 8, Figure 3 (left panel) — legend text overlapped by error-bar
   whiskers.** The tall 95% error bars on the rightmost data points (notably the
   K=32 point at familiarization step 5000, whose interval reaches ≈+1.1) extend
   up into the top-right legend region, and a horizontal whisker cap crosses
   through the legend labels "K=32" / "K=20". The text stays legible, but the
   crossing lines read as untidy. Fix: nudge the legend (e.g. to a clear
   quadrant, or `loc` outside the tall-error region) or add headroom to the
   y-axis so the legend no longer sits on top of a whisker.

2. **Page 6 — Section B "Supplementary Figures" appears content-less on the
   page where its heading falls.** The "B Supplementary Figures" heading (line
   261) is immediately followed by "C Reproducibility" (line 262) with no body
   or figures between them on page 6; the section's three figures float to pages
   7–8. A reader on page 6 sees an empty section header. This is standard float
   behavior and the captions are correctly adjacent to their figures on the
   later pages, so it is purely cosmetic; if desired, a one-line pointer under
   the B heading ("Figures 1–3 follow.") or tighter float placement would remove
   the empty-heading impression.

## Pages inspected

Pages 1, 2, 3, 4, 5, 6, 7, 8 (all 8 of 8; none skipped).

---

FAIL (2 minor findings) — no critical or serious findings; both items are
cosmetic and do not block accept-ready.
