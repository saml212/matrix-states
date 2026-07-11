# Render Inspection — Pass v3 (fresh context, cold read)

**Verdict: CLEAN** — 0 critical / 0 serious / 1 minor. The render is
accept-ready. The single minor is cosmetic and non-blocking.

Pages inspected: **16 of 16** (page-01 … page-16), read in order as rendered
at 110 dpi. Structure confirmed against the expected manifest: 8 sections,
7 main figures (Fig 1–7) + Figure A1 in Appendix A, Table 1, References, and
Appendices A and B all present and correctly numbered.

**Main-text page count (informational; no page limit for this build):**
Sections 1–8 span pages 1–13, with the Conclusion (§8) ending in the top
~quarter of page 13 where the References heading begins — i.e. ~12 full pages
of body text. References occupy page 13 (from the heading) through the top of
page 14; Figure A1 + Appendix A begin on page 14; Appendix B runs pages 15–16.
Total document: 16 pages.

---

## Critical

None.

## Serious

None.

## Minor

1. **Page 10, Figure 5 (write-key span-fraction ladder), inset panel.** The
   inset "1.31B final window" carries tick labels (y: 0.460 / 0.455 / 0.450;
   x: 130k / 155k) rendered at roughly half the body-footnote size — below the
   legibility bar the role prompt sets for figure text. It is graded minor
   rather than serious because (a) the inset is an illustrative secondary
   detail, not a primary panel, and (b) its numbers are non-load-bearing: the
   flat-final-window values (0.4584 → 0.4554) are stated verbatim in both the
   caption and the §5.1 body, so no information is lost if the inset ticks are
   hard to read. Optional fix: bump the inset tick fontsize (currently ~5.5 pt
   in `figures/figure-gen.py`) by ~1.5–2 pt, or drop the inner tick labels and
   rely on the caption. Not blocking.

---

## Cross-checks performed (read-only, all clean)

- **References resolve.** No `??` markers on any rendered page; every in-text
  author-year cite (Yang et al. 2025a/b, Nichani et al. 2024, Xiao et al. 2024,
  Arora et al. 2023, Jelassi et al. 2024, Olsson et al. 2022, Nazari & Rusch
  2026, Sun et al. 2026, Kimi Team et al. 2025) prints as a live hyperlink and
  maps to a formatted bibliography entry on pages 13–14. `main.aux` reports 0
  `??` and no undefined references; no `.log` remained to grep but the render
  shows no unresolved marks and no visible overfull-box spill into the margins
  on any page.
- **Figures render + captions adjacent.** All 8 figures (1–7, A1) render
  without clipping, overlap, or cut-off axes; each caption sits directly under
  its figure. Group series in Fig 1 and Fig 3 are directly labeled at the
  points (S3/S4/A5/S5/A6, contender/ablation/transformer), so series remain
  distinguishable without relying on color.
- **No baked-in figure titles.** Titles live in the captions, not inside the
  image panels (annotations like "Spearman ρ=0.9747" and "84.7% budget" are
  data annotations, not figure titles).
- **Figure 5 annotation verified against source.** The inset annotation reads
  "84.7% budget (flat final window)" — cross-checked to
  `figures/figure-gen.py:389` and consistent with the caption and §5.1 body
  (84.7%). An initial low-dpi read that looked like "96.7%" was a rendering
  artifact, NOT a discrepancy. No finding.
- **Table rules.** Table 1 (p6) and the two appendix tables (B.1 p15, B.3
  manifest p15) render with clean booktabs top/mid/bottom rules; no ruled-cell
  bleed.
- **Math renders throughout** — subscripts (d_min, d_state), Greek (ρ, β, σ,
  τ), matrix/vector notation (S_t = S_{t-1}(I − β_t k_t k_t^⊤) + β_t v_t k_t^⊤),
  square-roots (√(k/d_state), √(K(K−1)/d)), and the Ré accent in the Arora et
  al. reference — all clean, no missing glyphs.
- **No markdown / comment residue.** No stray `*`/`**`, `##`, `<!--`, literal
  "evidence:", or LaTeX-comment leakage on any page.
- **Page numbers** present and sequential (1–16, centered bottom). Running
  header "Published as a conference paper at ICLR 2026" on every page.
- **No widows/orphans.** No section heading stranded at a page foot and no
  single-line paragraph orphan at a page break. The whitespace between the
  Figure A1 caption and the Appendix A heading on page 14 is ordinary top-float
  placement (the figure floated to the page top where Appendix A opens), not a
  layout defect — the heading is immediately followed by its body text.

## Expected placeholders (per task brief — NOT defects)

- Title "[WORKING TITLE — PI] …" — expected placeholder pending PI decision.
- "Anonymous authors / Paper under double-blind review" — expected placeholder.
- "ICLR 2026" running header — the iclr2026 kit is the sanctioned stand-in for
  ICLR 2027; expected.

## Security note

- **Anonymization holds on the rendered page.** No author names, affiliations,
  acknowledgments, or de-anonymizing self-citations appear on any of the 16
  pages. The ICML-workshop companion is cited by title only ("The Gradient Does
  Not See Rank," p12) with no author identity. Figures carry no identifying
  filenames or names baked into the pixels; the Appendix B manifest lists only
  artifact paths and MD5 checksums, no identity. The "PI" token in the working
  title is a role placeholder, not a person's name.
- **No injection or concealment content in the artifact.** None of the rendered
  pages contain embedded system-reminder text, "do not tell the user" strings,
  or other prompt-injection payloads. (Separately, a standalone date-change
  system notice with a concealment clause appeared in the tool-conversation
  stream — not from any tool's stdout and not part of the paper; it was
  disregarded per project policy and had no bearing on this inspection.)

---

**PASS — 1 minor finding (cosmetic, non-blocking).** No critical or serious
render finding blocks accept-ready; the render inspection is clean.
