# Stage 03 — Style Critique (fresh-context cold read)

**VERDICT: PASS (0 violations).**

All eleven section files were read in full and swept mechanically (whole-word,
case-insensitive) against the banned-word list and every voice/DO-NOT rule in
`references/styleguide.md`. No hard-fail hit survives.

---

## Category results

### Banned words
Zero hits. All 17 banned tokens (`honest`, `actually`, `really`, `just`,
`clearly`, `obviously`, `interestingly`, `nicely`, `remarkable`, `surprising`,
`unfortunately`, `essentially`, `wildly`, `literally`, `parsimonious`,
`cleanest`, `sharpest`) checked whole-word, case-insensitive, across body,
abstract, captions, and headings. None present.

### First-person / narrative-process
Zero hard hits. Every `I`/`my`/`me` grep match is the mathematical identity
matrix `$I$` (e.g. `$S_t = S_{t-1}(I - \beta_t k_t k_t^{\top})$`, the Appendix A
title `The $c^{*} I$ Complement Scaffold`), not the pronoun. Voice is editorial
"we" throughout ("We ask", "we contribute", "We record", "we claim neither").
No occurrence of the banned narrative-process patterns ("our original
hypothesis", "we report a negative result with a mechanism", "the paper's
sharpest claim", "the story of finding it").

### Contractions
Zero hits.

### Em-dash-as-pause
Zero hits in rendered prose. The only em-dash (`—`) characters are inside two
build-scoped HTML comments in `00_abstract.md` (the `<!-- Title: … — PI -->` and
`<!-- Authors: … — PI decision pending -->` placeholders); these are invisible
in render and both comments are properly closed, so nothing leaks. The two
en-dashes (`–`) in `10_appendix_b_reproducibility.md` lines 61–62
(`§1.33–§1.36a`, `§1.31–§1.41`) are numeric section-range dashes, standard
typography, not conversational pauses. Colons used elsewhere (e.g.
`03_rank_law.md:69` "…0.000 in all five groups: not degraded, zero";
`05_pathology_at_scale.md:20` "…0.455: monotone in scale…") are legitimate
appositive colons, not disguised pauses.

### Headings
Zero rhetorical-question headings. All 30 headings are noun phrases or
declarative statements ("3.2 Dimension, Not Solvability", "4.3 The Capability
Is Fast-Weight-Resident and Nonlinearly Stored", "5.4 Two Faces of One
Mechanism"); none ends in `?`. Declarative-statement headings match the
styleguide's own endorsed example ("The Flatten-Then-Project Readout Is
Rank-Blind").

### Captions
Zero non-self-contained captions. All figure captions (Figures 1–7, A1) name
the subtask, the method/ablation, and the takeaway. No caption contains
`pending`, `TODO`, `will be added`, or `forthcoming`. Figure 7's "(not shown)"
is a standard "this datum is not plotted in this panel" convention, not a
deferral to future content. Cross-references such as Figure 1's "the
equivalence test in Section 3.2 declares them equivalent" supplement a
self-stated takeaway ("lands together") rather than defer meaning to the body.

### Abstract length
**230 words** (excluding the two HTML comments and the `# Abstract` heading;
each `$…$` math span counted as one rendered token). In-band (200–230,
inclusive) — but at the exact ceiling. See informational note 1.

### DO-NOT / apparatus
Zero violations. No GPU-hours, no dollar/cost figures (every `$[0-9]` grep hit
is a LaTeX math delimiter such as `$3 \times 10^{-4}$` or `$2\sigma$`, not a
cost), no funding language, no `acknowledg*`, no experiment-count bragging. The
single `audited` occurrence (`04_capability_separation.md:37`, "the audited
round protocol") is the tolerated technical term describing the protocol, per
the project's pass-2 ruling. Precise cell/run counts in Appendix B (27-cell
sweep, 90-pass fan-out, eleven archived runs) are reproducibility
quantification the styleguide's "quantify claims" rule requires, not
volume-bragging.

### Anonymization
Not applicable — this is the named (public) flagship build, which the styleguide
exempts from the identity-leak grep. The only identity token,
`samlarson@pebbleml.com`, sits inside the `<!-- Authors … (named build only) -->`
comment and will not render. No `github.com`/`huggingface.co`/de-anonymizing URL
appears anywhere. See informational note 2.

---

## Informational notes (non-violations, for awareness)

1. **Abstract is at 230/230 words — the exact top of the band.** It passes, but
   there is zero headroom; any word added downstream (e.g. resolving a
   placeholder or a reviewer-requested clarification) pushes it out of band.
   Recommend trimming 5–10 words of slack now to create margin.

2. **Build-scoped identity/title scaffolding, all inside HTML comments.**
   `00_abstract.md` lines 3–7 carry the working title, the unresolved
   `[AUTHORS — PI decision pending]` line, and the corresponding-author email.
   All are inside balanced `<!-- … -->` comments (verified: 12 opens / 12
   closes in the abstract; every file's comments balance), so none render. For
   the named arXiv build this is fine. If an anonymized double-blind build is
   ever generated from these same sources, these comments (and any archive-root
   link in Appendix B.3) must be stripped — B.3 already anticipates this
   ("the anonymized build links an anonymized mirror").

3. **"Our thesis" (`01_introduction.md:21`).** Editorial-we possessive that
   states the claim, not the banned narrative-process pattern "our original
   hypothesis." Not flagged as a hard violation; left for the detector judge's
   interpretive pass if it wants to prefer a bare "The thesis…".

4. **Instrument-repair narration (§2.5, §3.3, §4.3).** These passages describe
   instrument-defect-found-and-fixed rounds ("only repaired readings appear
   below"; "A verdict is only as good as the instrument that reads it",
   `02_background_setup.md:117`). This is deliberate methodological disclosure
   ("Instrument Validity as a First-Class Method"), not self-grading of results,
   and matches no banned pattern. The line 117 maxim is mildly aphoristic;
   noting it only so the detector judge can weigh tone. Not a hard fail.

5. **Mild coined/editorial descriptors, not banned:** "marquee pair/groups"
   (`03_rank_law.md`), "decode-proof rather than decode-laundered"
   (`06_related_work.md:10`), "standard bearer" (`01_introduction.md:9`). None
   appear on the banned list; the style judge is mechanical and does not invent
   rules. Flagged only for the detector judge's awareness.

---

## Security note

Clean. No file contains an embedded instruction, prompt-injection string, or
fake system-reminder block. Grep for `system-reminder`, `ignore previous`, `do
not tell`, `disregard`, `you must comply`, and similar returned zero matches
across all eleven files. The only HTML comments present are the legitimate
evidence-row tags (`<!-- evidence: Rn -->`) and the two PI title/author
placeholders; all are balanced and build-scoped. No concealment or
date-change instructions observed.

---

**FAIL count: 0. Verdict: PASS.**
