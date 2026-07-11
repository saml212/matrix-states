# 06 — Render Inspection (Round 1)

Fresh-context visual inspection of the compiled PDF
`papers/capacity-colm-er/bundle/main.pdf`. Judged from the rendered pages
(read at default resolution for all 10 pages, plus 200-DPI PNG re-renders of
the four float pages 4-7 for figure/table legibility, color, and margins).
LaTeX source was NOT consulted; no prior gauntlet reports were read.

## 1. Summary

- **Pages inspected: 10 of 10** (complete; none skipped).
- **Counts: 0 critical / 0 serious / 1 minor.**
- **Does the render block accept-ready? NO.** Zero critical and zero serious
  findings. The one minor item is a cosmetic, defensible-but-flaggable style
  deviation that does not block submission and does not degrade the paper on
  the page.

Page budget: total document = 10 pages. The `References` heading begins on
**page 9** (roughly two-thirds down). Main text (Sections 1-8 + Conclusion,
references excluded per the workshop's "4-10 pages of main text, references
excluded" rule) therefore occupies **pages 1 through 9** — i.e. **≤ 9 pages of
main text**, comfortably inside the 4-10 limit. No page-limit violation.

Anonymization holds on every rendered page: running header reads "Under review
as a conference paper at COLM 2026"; author block reads "Anonymous authors /
Paper under double-blind review". No author names, no acknowledgments section
(Conclusion goes straight to References), no identifying URLs, and no identity
text baked into any figure image. Clean for double-blind.

References render fully (pages 9-10): bibliography present, COLM/natbib
author-year style, hanging indents; every in-text citation I checked resolves
to a printed entry (blue author-year links throughout Sections 1, 2, 6) with
**no `?` markers** and no unresolved `\ref`/`\cite`. Figure/Table/Section
cross-references (Figure 1-3, Table 1, Sections 3/3.1/3.2/4/5/7) all resolve to
real numbers.

Line numbers are present in submission mode (expected — not a finding).

## 2. Critical findings

None.

## 3. Serious findings

None.

## 4. Minor findings

**M1 — Figure 1 (page 4): editorializing panel titles baked into the figure
image.** The two subplots carry in-image titles "d_state = 64: a sharp
transition" (left) and "d_state = 128: the same window, flat" (right). The
condition portion ("d_state = 64" / "d_state = 128") is a legitimate,
necessary panel label; the appended editorial clauses ("a sharp transition",
"the same window, flat") state the panel's takeaway inside the pixels, which
duplicates the caption and brushes the ml-paper-writing rule that the caption —
not the figure image — carries titular/conclusory text. Fix (optional): trim
the two panel titles to neutral condition labels (e.g. "d_state = 64" /
"d_state = 128", or "(a) d_state = 64" / "(b) d_state = 128") and let the
existing caption carry "sharp transition" vs "flat". Cosmetic; a real reviewer
is very unlikely to penalize descriptive panel titles, so this does not block.
Figures 2 and 3 have no baked-in titles.

## Non-finding checks that passed (for the record)

- **Figures render cleanly.** No clipping, no cut-off axis labels/legends, no
  element overlap in any of Figures 1-3. Verified at 200 DPI: Fig 1 inset text
  (x_0 CI) and legend do not collide with the logistic curve; Fig 1 right-panel
  "12/12 cells" note does not overlap its four ceiling dots; Fig 2 legend box,
  grey training-band label, and marker cloud are mutually clear; Fig 3
  annotations, leader line, colored patches, and points do not overlap. All
  axis titles fully visible.
- **Figure text legibility.** Axis labels, tick labels, inset/annotation text,
  and legends in all three figures are legible at print size, comparable to
  body/footnote text. No figure has axis labels smaller than a footnote.
- **Series distinguishable without color.** Fig 2 separates its three series by
  marker SHAPE (vermillion cross vs blue open circle vs orange open triangle),
  not color alone. Fig 3 separates filled circles (located midpoints) from open
  squares-with-arrows (window-limited bounds) by shape; its two rival-band
  patches are teal vs amber (a colorblind-distinguishable pair) AND are
  vertically position-separated AND text-labeled by value. Fig 1 is a single
  series. No color-only reliance.
- **Captions adjacent to floats.** Every caption sits directly beneath its
  float on the same page: Fig 1 (p4), Fig 2 (p5), Fig 3 (p6), Table 1 (p7). No
  drifted floats.
- **Table 1 (page 7) renders cleanly** with all five columns inside the text
  block, and its dagger footnote is present and readable (the "≥ 90†" /
  "≥ 2.50†" cells map to the caption's "†Rests on a three-seed K=90 reading…"
  note).
- **No margin violations / overfull boxes** observed. Justified text, inline
  math (e.g. the DeltaNet update on p2, `\max|cos|` expressions), and the table
  all stay within the text block on every page.
- **No orphaned headings or widowed lines** at any page break. Section headings
  1-8 and 3.1/3.2 are each followed by their content on the same page; the
  `References` heading on p9 is followed by entries on the same page.
  Paragraphs that continue across page breaks (e.g. p5→p6 past the Fig 3 float)
  are normal float behavior, not widows.

## 5. Pages inspected

Pages 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (all 10 — default-resolution read of the
full document; pages 4, 5, 6, 7 additionally re-rendered at 200 DPI for
figure/table scrutiny).

## Security note

No fake `<system-reminder>`-style blocks, date-change claims, or concealment
instructions appeared in any tool output during this inspection.

---

**Verdict: FAIL (1 finding)** — one MINOR, non-blocking cosmetic item (M1);
zero critical, zero serious. The render is accept-ready; M1 is an optional
polish the writer may clear for a clean re-inspection pass.
