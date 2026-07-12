# Render Inspection — round 2 (targeted re-gauntlet, Bound-1 revision)

Paper: `papers/reasoning-null-moss/main.pdf` (MOSS @ COLM 2026, 4pp body limit,
double-blind). Inspected the COMPILED PDF page by page (recompiled fresh via
`make` before inspecting; also re-rendered pages 2–5 at 200 DPI and cropped the
right-margin strips of pages 2 and 3 to adjudicate the overfull hbox).

## Summary

- Pages inspected: **10 of 10** (the full document: 4pp body + References +
  Appendices A–C incl. Supplementary Figures 1–3).
- Findings: **1 critical / 0 serious / 1 minor**.
- Verdict: the render **does NOT** block-clear. A 4-page-limit violation is
  present: the body (§7) runs onto page 5.

## Critical

1. **Page 4 → Page 5 — body exceeds the 4-page main-content limit.** The
   concluding §7 paragraph ("Not bounded. The hypothesis survives below the
   power floor…") begins on page 4 (ends page 4 at line 176, "…A competing
   account is expressivity: Grazzi et al.") and **continues onto page 5**: lines
   177–181 ("(2024) prove positive-eigenvalue DeltaNet cannot compose certain
   permutation groups … unblinded routing keeps zeros and transients from
   becoming headlines.") are body content sitting at the top of page 5. The
   **References heading appears only at line 182, on page 5**, after those five
   body lines. Per `venue-requirements.md` the limit is 4 pages of main content
   (§1–§7); References do not count, but here the body itself intrudes onto page
   5, so the main content is 4 pages + ~5 lines — over the limit. This is a
   CRITICAL accept-blocker.
   Fix: trim ~5–6 lines from the body (§1–§7) so §7 ends at the bottom of page 4
   and References opens cleanly at the top of page 5.

## Serious

None.

## Minor

1. **Page 2, §3 first paragraph — the 8.46pt overfull \hbox reads clean (no
   action strictly required).** The compile log reports `Overfull \hbox
   (8.45816pt too wide)` in `sections/03_geometric_null` (lines 8–30), i.e. the
   §3 opening paragraph on page 2 (not Table 1, as the round note anticipated).
   On the rendered page the justified text is flush with the header-rule right
   edge; a right-margin crop at 200 DPI shows no line protruding into the
   margin. Cosmetic only — it does not spill visibly and does not block.
   (Separately, Table 1 on page 3 was checked directly: the longest cell,
   "null-indistinguishable", and both booktabs rules sit inside the text block —
   no margin spill on the table either.)

## Clean checks (inspected, no findings)

- **Figures 1–3 (pages 9–10)** render cleanly: no clipping/overlap, legible
  fonts at print size, panel labels are subplot descriptors (not a baked-in
  figure title — the title lives in each caption), captions adjacent to their
  figures. The Fig 2 caption carries its revised disclosure sentence about the
  pre-fix right panel and reads correctly. Figure legends/axis labels contain no
  identity leak (series are named "Phase 1 zero-shot", "openr1 cells",
  "K=32"/"K=20", etc.).
- **Table 1 (page 3)** renders inside the text block with its caption directly
  below; footnote markers ‡/†/§ resolve.
- **Citations/references**: every in-text cite renders as a resolved author-year
  hyperlink (blue); no `?`/`??` markers on any page. The 11-entry bibliography
  (page 5) and all Appendix cross-refs (Section 3, Appendix A/B, Figures 2–3)
  resolve.
- **Anonymization** holds on every rendered page: "Anonymous authors / Paper
  under double-blind review" (page 1); no author name, org, cluster name,
  handle, filename, or URL in any figure, caption, table, or the Reproducibility
  appendix (page 9, which discloses only "one or two GPUs" and "≈9.5 GPU-hours").
- **Page breaks**: no orphaned headings or widowed lines. §3 (page 2), §4 (page
  3), §5/§6/§7 (page 4), References + Appendix A (page 5) each keep their heading
  with following text. (The §7 paragraph split across pp.4–5 is the page-limit
  issue above, not a typographic widow.) The underfull \vbox (badness 10000)
  warnings at main.tex:162/231 are cosmetic float-spacing only, not visible.

## Pages inspected

Pages 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (all 10). Pages 2–5 additionally
re-inspected at 200 DPI; right-margin strips of pages 2 and 3 cropped and
inspected for overfull-box spill.

FAIL (1)
