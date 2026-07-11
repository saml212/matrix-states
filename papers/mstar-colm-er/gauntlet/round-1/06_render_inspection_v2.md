# Render inspection (v2) — mstar-colm-er

Fresh-context visual inspection of the compiled PDF
(`papers/mstar-colm-er/main.pdf`, 10 pages). Verified by reading every rendered
page as an image; the flagged Figure 1 concern was re-checked at 400 DPI with a
cropped high-resolution render of the chance-line band. No prior inspection
report was read.

## Summary

- **Pages inspected:** 10 of 10 (1–10). None skipped.
- **Counts:** 0 critical / 0 serious / 0 minor.
- **Blocks accept-ready?** No. The render is clean and does not block
  accept-ready. No CRITICAL render finding exists.

Key gate checks, all passing:

- **Page limit (4–10 pp main text, references excluded — workshop rule).**
  Main text runs Sections 1–9; the Conclusion (§9) ends partway down **page 8**,
  References begin immediately below it on page 8 and run onto page 9,
  Appendices A–D occupy pages 9–10. Main text ≈ 7.5 rendered pages — comfortably
  inside the 4–10 window. PASS.
- **Figures render cleanly (fig1_horizon p5, fig2_szero p6, fig3_traincurve
  p10).** No clipping, overlap, cut-off labels, or illegible fonts. Legends and
  axis labels are legible at print size. No title baked into any figure's pixels
  (the caption carries the title in all three; Fig 2's top strip is a legend, not
  a title).
- **Fig 1 gray capped-KV crosses (specifically flagged).** Confirmed at 400 DPI:
  large gray "X" markers are clearly visible at all three x-positions (454, 902,
  1798), sitting on the chance line and overlapping the red uncapped-KV dashed
  line. Both transformer series read chance, as the caption states, and both are
  legible. Concern resolved.
- **Captions adjacent to floats.** Fig 1/2/3 captions sit directly beneath their
  figures (pp. 5/6/10); Table 1 caption under its table (p4); Tables 2–3 (p7),
  Table 4 (p9), Table 5 (p10) all captioned adjacent. No drifting floats.
- **No margin violations / overfull boxes.** All body text, the three-line
  title (p1), and every table (including the 10-column Table 2 on p7) stay
  within the text block. No spill into margin or gutter observed.
- **References render.** Full bibliography present (pp. 8–9), formatted in COLM
  style; every in-text citation resolves to a printed author–year link. No `?`
  markers from unresolved `\ref`/`\cite` anywhere.
- **Anonymization holds on the page.** Author block reads "Anonymous authors /
  Paper under double-blind review." No names or organizations baked into any
  figure's pixels; no identifying URLs. The running head names the workshop
  venue ("Under review at the Workshop on Efficient Reasoning, COLM 2026") —
  a venue name, not an identity leak, per the rule. Appendix D's promise to
  release artifacts "with the de-anonymized version" is a forward statement,
  not a leak.
- **No orphaned headings or widows.** Section headings 5 (p5), 6 (p6), 7 (p7),
  8–9 (p8), and Appendices B/C/D (p10) each carry following body text on the
  same page. Every page break lands mid-paragraph or between complete
  paragraphs; no single-line widows observed at any transition.

## Critical

None.

## Serious

None.

## Minor

None.

## Pages inspected

Pages 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (all 10 pages of the document). Page 5
Figure 1 additionally re-rendered at 400 DPI and cropped to the chance-line band
to confirm the gray capped-KV cross markers render.

## Security note

No `<system-reminder>`-style concealment or behavior-change block appeared in any
tool output during this inspection.

PASS
