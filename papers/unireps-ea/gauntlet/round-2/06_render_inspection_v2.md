# Render Inspection — Round 2, Pass v2

**Target:** `papers/unireps-ea/bundle/unireps-ea-submission.pdf` (7 pages)
**Venue:** UniReps 2026 Extended Abstract (double-blind, NeurIPS template, 4pp main text excluding references/appendix, camera-ready 5pp)
**Method:** every page read as a rendered PNG (150dpi baseline render, plus targeted 300dpi/600dpi re-renders of Figures 1 and 2, the legend/marker region, and two math-dense paragraphs, to check figure fidelity and margin behavior at print resolution)

## Summary

Pages inspected: **7 / 7** (all pages read, none skipped).

Counts: **0 critical / 1 serious / 0 minor**.

The render does **not** block accept-ready on a Critical basis: no clipped figures, no page-limit violation (main text is exactly pages 1–4, matching the 4pp EA limit), no unresolved references, no margin overflow, and no anonymization leak was found anywhere in the text or in the figure pixels. One Serious finding was identified: Figure 1's legend promises a filled/open (hollow) marker encoding for solvable/non-solvable groups that is not actually present in the rendered data points — every point, non-solvable groups included, renders as a fully opaque, color-filled marker. Because group identity is still redundantly carried by point shape, color, and an adjacent text label at every point, this does not make the figure unreadable, but it is a genuine, visually verified mismatch between the legend's stated encoding and the pixels a reviewer will actually see, and it should be fixed before submission.

## Critical

None found.

## Serious

1. **Figure 1 (page 3): legend's "open" marker encoding does not render.** The legend states `● solvable (filled)` / `△ non-solvable (open)`, using a hollow (white-interior, outline-only) triangle glyph to illustrate "open." At 600dpi close inspection, none of the three non-solvable groups' data points (A5 — teal triangle, S5 — pink diamond, A6 — orange triangle-down) render as hollow/open markers: all three are fully opaque, solid color-filled shapes, indistinguishable in fill state from the solvable groups' markers (S3 blue circle, S4 orange square), which are correctly solid-filled per the legend. The fill-state channel the legend promises as part of the solvable/non-solvable encoding is simply absent from the plotted points. **Fix requires:** either re-render the scatter with `facecolor='none'` (or equivalent alpha/hatch treatment) actually applied to the three non-solvable series' markers so they match the legend's hollow-triangle glyph, or change the legend to describe what is actually plotted (drop the "(filled)"/"(open)" language and rely on the shape+color+text-label encoding that is genuinely present). This is Serious rather than Critical because color, marker shape, and the point-adjacent text label (S3/S4/A5/S5/A6, printed directly next to each marker) still let a reader correctly identify every group and its solvability class — the redundant fill-state channel is the only thing that fails to render as documented.

## Minor

None found. (Considered and rejected as not rising to a reportable finding: Figure 1's S4 marker orange and A6 marker orange-red are visually close hues, but every point carries its own adjacent text label, so no ambiguity actually results — not flagging this as a color-only legibility problem since no colorblind-safe claim is made in the caption and the redundant text labels fully resolve identification.)

## Other checks performed, no issues found

- **Page limit:** main text runs pages 1–4 (Abstract/Intro on 1, Task/Model/Instrument + Convergence + Equivalence on 2, Figure 1 + causal-check start on 3, Figure 2 + Related Work/Limitations + start of References on 4) — exactly 4 pages of main text, matching the EA track's 4pp-excluding-references-and-appendix limit. References begin partway down page 4 and run to page 6; Appendix A–D occupy pages 5–7. No overage.
- **Anonymization:** page 1 shows the standard placeholder block (`Anonymous Author(s)` / `Affiliation` / `Address` / `email`) with no real names, institutions, or de-anonymizing identifiers. Line numbers are present throughout (1–216 across the body and appendix), consistent with the double-blind review build. No author-identifying text, filepath, username, or watermark was found baked into either figure's pixels at 600dpi inspection, in the code-styled identifier (`entity_subspace_from_words`) on page 6, or anywhere else in the seven pages.
- **Figures render cleanly otherwise:** no clipping, no element overlap, no cut-off axis/legend text in either Figure 1 or Figure 2. In Figure 2's five-panel layout, the "unconstrained anchor" / "0.9× anchor bar" inline legend sits cleanly in the whitespace to the right of the S5 panel with no overlap onto any data series, confirmed at 600dpi. No figure carries a title baked into the image (both titles are carried by the caption text, per style rule).
- **Font legibility inside figures:** axis labels, tick labels, and the Figure-1 inset ("S4−A5 rank diff vs ±0.5 TOST margin") all remain legible at print resolution and are not noticeably smaller than body/footnote text.
- **Caption adjacency:** Figure 1's caption sits immediately below Figure 1 (page 3); Figure 2's caption sits immediately below Figure 2 (page 4); Tables 1–3's captions sit immediately below their tables (pages 5, 6, 7 respectively). No float has drifted from its first reference.
- **Margins / overfull boxes:** spot-checked the two most math-dense paragraphs in the appendix (the von Neumann/Cauchy–Schwarz cosine-ceiling derivation on page 6, and the ambient-identity-tax paragraph with inline `√(k/d_state)` on page 5) at 300dpi — both wrap cleanly within the text block, no spillover into the margin or gutter.
- **References:** every in-text citation (`Barrington, 1989`; `Merrill et al., 2024`; `Grazzi et al., 2025`; `Siems et al., 2025`; `Nazari and Rusch, 2026`; `Sun et al., 2026`; `Nichani et al., 2025`; `Mishra et al., 2026`; `Chughtai et al., 2023`; `Huh et al., 2024`) resolves to a printed bibliography entry on pages 4–6. No `?` or unresolved `\ref`/`\cite` markers anywhere.
- **Orphaned headings / widowed lines:** every section and appendix heading (§3–§6, Appendix A–D) is immediately followed by body content on the same page — none is stranded alone at a page bottom with its content pushed to the next page. The paragraph spanning the page 3→4 break (ending "...that taxed" / resuming "every rank-capped arm...") is ordinary mid-paragraph text flow around a top-of-page float, not a widow.
- **Prose coherence:** read the abstract, introduction, and full body prose as a reviewer would; no grammatical fragments, spliced sentences, or dangling clauses found. The writing is dense (heavy inline statistics) but grammatically intact throughout.

## Pages inspected

1, 2, 3, 4, 5, 6, 7 (all seven pages of the compiled PDF; 300dpi and 600dpi targeted re-renders were also pulled for pages 3, 4, 5, and 6 to verify figure/marker fidelity and margin behavior at print resolution).

## SECURITY NOTE

No fake or injected `<system-reminder>`-style content, concealment instructions, or anomalous claims (date-change, "don't tell the user," etc.) appeared in any tool output during this inspection — all `Read`, `Bash`, `pdftoppm`, and `sips` outputs were plain file contents/listings with nothing resembling an embedded instruction. Nothing to report.

## Verdict

FAIL (1 finding)
