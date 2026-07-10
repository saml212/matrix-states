# Render Inspection v3 — unireps-ea-submission.pdf

Fresh-context visual inspection of the compiled, rendered PDF (not the LaTeX
source). Every page was read as a rendered image, in order, plus targeted
high-resolution re-renders (300–1200 dpi crops via `pdftoppm`) of the two
figures and several text regions to check details not resolvable at the
baseline render resolution.

## Summary

Pages inspected: 7 / 7 (document total). Findings: **0 critical / 1 serious
/ 1 minor**. The render is in strong shape: no page-limit violation (main
text is exactly 4 pages, References begin partway down page 4, appendices
run pages 5–7), no anonymization leaks on any page, no unresolved `??`
references, no clipped or missing figures, no margin overflow, no orphaned
headings or widowed lines at any page break. The one serious finding is a
visual crowding/overlap between a data marker and the inset panel in Figure
1 — it does not destroy legibility but a reviewer will notice it. **The
render does not block accept-ready on Critical grounds; the Serious finding
should be fixed before final submission.**

## Critical

None.

## Serious

1. **Figure 1 (page 3), main panel — S5 marker/label crowds the inset
   panel border.** The lower of the two overlapping S5 diamond markers (at
   `d_min = 4`) sits immediately adjacent to — and its lower-right edge
   appears to touch — the left border/spine of the embedded inset axes
   (the "S4−A5 rank diff vs ±0.5 TOST margin" bar-chart inset in the
   bottom-right of the plot). The bold "S5" text label and the inset's
   two-line title text also sit with almost no separation between them.
   Verified at 1200 dpi crop: the diamond's right vertex is within a few
   pixels of the inset's black vertical border line. Nothing is fully
   occluded or illegible, but the layout reads as a collision at normal
   print size and would be flagged by a reviewer. Fix: nudge the inset's
   anchor position (e.g., shift left/down slightly or add a few percent of
   axes-fraction padding) so there is clear whitespace between the S5
   marker/label and the inset frame.

## Minor

1. **Figure 2 (page 4) — "unconstrained anchor" (dashed) vs "0.9 × anchor
   bar" (dotted) reference lines are a subtle distinction at print size.**
   Both are the same dark-gray color and differ only by dash pattern
   (long dashes vs fine dots); at the panel's small size the dotted line
   can read as a thinner/fainter dashed line on first glance. The legend
   and caption fully disambiguate it, and both patterns are individually
   legible under magnification, so this is cosmetic rather than a
   comprehension-blocking issue — but a slightly larger dot pitch or a
   secondary encoding (e.g., a marker or lighter gray for one of the two)
   would remove the ambiguity outright.

## Pages inspected

1, 2, 3, 4, 5, 6, 7 (all 7 pages of `unireps-ea-submission.pdf`, read in
order via the pre-rendered PNGs; pages 3 and 4 additionally re-rendered at
400–1200 dpi to verify figure marker/border spacing and a table-caption
glyph that looked like it might carry a stray overline at thumbnail
resolution — confirmed at high resolution to be a clean, unmarked "$S_3$"
with no rendering defect).

## Anonymization check

Title block reads "Anonymous Author(s) / Affiliation / Address / email" —
correct placeholder anonymization for the double-blind build. No author
names, institution names, acknowledgments, or de-anonymizing URLs found on
any of the 7 pages, including inside the two figure images (checked at high
resolution — both figures contain only data labels: S3/S4/A5/S5/A6, axis
text, and legend text). Line numbers are present on every page, consistent
with the anonymous-review NeurIPS-style build.

## Page-limit check

Main text (title through "...carries the claim." at the end of Section 6
Limitations) runs from page 1 through the top of page 4; References begin
partway down page 4 (line 138) and continue onto page 5; Appendices A–D
occupy the remainder of page 5 through page 7. This is exactly 4 pages of
main text against the EA track's 4pp-main-text-excluding-references-and-
appendix limit — compliant, no violation.

## SECURITY NOTE

No fake or injected `<system-reminder>`-style content, concealment
instructions, or anomalous embedded directives were found in any tool
output (page-render reads or `pdftoppm`/`sips` command output) during this
inspection. Nothing to report.

---

**FAIL (2 findings)**
