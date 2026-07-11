# 06 — Render Inspection (Stage 8, final visual gate)

**Paper:** `papers/measurement-ws/` — "The Instrument Is the First Suspect: Six Broken Lenses in One Empirical Program"
**Artifact inspected:** `main.pdf` (8 pages), rendered to page images at 110 dpi; Figure 1, Figure 2, and Table 1 re-rendered at 300–400 dpi for smallest-text legibility.
**Date:** 2026-07-11

## 1. Summary

Pages inspected: **8 of 8** (1–8, every page). Counts: **0 critical / 0 serious / 2 minor**.

The render **does not block accept-ready**. Every figure and table renders correctly, every in-text citation resolves to a printed bibliography entry (no `?`/`??` markers — confirmed visually and by a `pdftotext | grep '??'` cross-check that returned nothing), all math (√ radicals, subscripts, ⊕, ≈, ×10ⁿ, transpose ᵀ) renders cleanly, the anonymous author block is intact, and the body ends on page 4 within the 4-page limit. The two minor items below are optional cosmetic polish, not defects — every flagged element is legible at print size.

## 2. Critical

None.

## 3. Serious

None.

## 4. Minor

- **M1 — Figure 1 (page 3), in-plot text is small (borderline, acceptable).** The scatter is set at 0.40\linewidth in a side-by-side minipage with its caption. At print size (110 dpi render) every element is still readable: the axis titles ("primary lens: recovered fraction @0.9, depth 64" / rotated "crosscheck lens: recovered fraction @0.9, depth 64"), the tick labels 0.0–1.0, and the two-line legend ("not converged (final loss ≥ 0.02), n=45"; "converged (final loss < 0.02), n=17"). The in-plot font is slightly smaller than the body/footnote size — this is the size that a prior audit soft-flagged. It reads fine but sits at the low end of legible. *Optional fix:* bump the figure to ~0.45\linewidth or raise the matplotlib font size one step before re-export. Colorblind-safety holds independently of this: the two series are redundantly encoded by marker fill (open orange vs. filled blue) and by the legend, not by hue alone.
- **M2 — Figure 2(b) (page 7), x-axis category labels crowd between ticks (iii) and (iv).** The four two-line tick labels "(i) S₁ tap (deployed)", "(ii) S₀ tap", "(iii) S₁, true query path", "(iv) pre-LM-head hidden" are all legible, but "(iii)…" and "(iv)…" nearly abut (the "(iv)" starts immediately after "true" with "query path" wrapping below). No overlap that impairs reading; purely a spacing nicety. *Optional fix:* shorten labels (e.g. "(iii) S₁ query" / "(iv) pre-LM-head") or widen panel (b) / reduce label font.

## Per-page one-liners

- **Page 1** — Title, anonymous author block (Anonymous Author(s) / Affiliation / Address / email) and Abstract all render correctly; line-number gutter (1–35) clear of text; no orphan headings; §1 Introduction and §2 heading clean.
- **Page 2** — Rules 1–5, §3 Case I, §4 Case II; math (n_iter=20, K/d_state, d_state=96, 1/√32 ≈ 0.177, 1.4×10⁻⁶, 7,000×) renders cleanly; no margin spill; footer "2".
- **Page 3** — Figure 1 renders (scatter + adjacent caption in minipage; y=x reference line, legend present); §5 Case III text clean; Q̂ ≈ I and subscripts render; see M1. Footer "3".
- **Page 4** — §6, §7, §8 (Related Work and Limitations); √(k/d_state), √2 render; **body ends here within the 4-page limit**; no widow/orphan; footer "4".
- **Page 5** — References; all entries render, diacritics correct (Gaël Varoquaux, Štěpán Bahník); no unresolved `?` markers; footer "5".
- **Page 6** — Appendix A; dense math (Z ≈ c(ρ⊕I), ZZᵀ ≈ c²I, S₁ = βvkᵀ, ‖vkᵀ − kvᵀ‖_F/‖vkᵀ‖_F ≈ √2, 4.5×10⁻⁸) all renders; Appendix B heading followed by body text (no orphan); footer "6".
- **Page 7** — Table 1 (scriptsize six-incident catalogue — all cells legible at zoom, √(k/d_state) and √2 render, hyphenated wraps stay in-cell, no margin spill, caption adjacent), Figure 2 (two panels, y-axis titles fully present, chance line labeled, contender/ablation legend, caption adjacent — see M2), Table 2 (caption adjacent); footer "7".
- **Page 8** — Tables 3, 4, 5; grouped column headers (contender/ablation), bold cells (0.674 / +0.800), subscripts (n_h), all render with adjacent captions; bottom whitespace is normal appendix float placement; footer "8".

## 5. Pages inspected

Pages **1, 2, 3, 4, 5, 6, 7, 8** — all 8 pages of the document, in order. Figure 1 (p3), Table 1 (p7), and Figure 2 (p7) additionally re-rendered at 300–400 dpi and inspected at zoom for smallest-text legibility.

---

**Verdict: PASS (accept-ready).** 0 critical / 0 serious / 2 minor. No finding blocks accept-ready; M1 and M2 are optional cosmetic polish. (Recorded strictly: two minor cosmetic notes; neither is a render defect.)
