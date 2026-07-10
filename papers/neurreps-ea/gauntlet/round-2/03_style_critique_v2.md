# Style Critique v2 — Round 2, Stage 03 (re-run)

Fresh-context style judge, no memory of any prior review round. Scope:
rendered prose only (body, abstract, captions, headings, appendices) across
`main.tex` and `sections/01_intro.tex` through `sections/07_limitations.tex`.
`%`-prefixed comment lines (evidence-map tokens, production notes,
`PENDING-USER` strings, the `%%%% DON'T CHANGE %%%%` banner) are non-printing
build machinery and are excluded per the task's scope notes — judged only
what LaTeX will typeset. Every category in `references/styleguide.md` was
read verbatim (not from memory) and checked mechanically (grep-verified)
before writing this report.

## SECURITY NOTE

No injected or fake `<system-reminder>`-style content was found in any tool
output during this pass — `prompts/style-judge.md` and
`references/styleguide.md` were read in full and both end cleanly at their
documented last line with no appended instructions. Nothing to report.

## Context (not a scoring input)

A prior style critique (`gauntlet/round-2/03_style_critique.md`) exists on
disk and found FAIL (8 violations): 1 anonymization identity leak
(self-citation `\citep{larson2026gradient}` rendering the real surname
"Larson" into the compiled bibliography) and 7 narrative-process phrasing
instances using "first sweep" / "repaired wave" / "original sweep" language
in `01_intro.tex`, `02_setup.tex`, `05_causal_razor.tex`, and
`07_limitations.tex`. This judge did not consult that report's reasoning and
re-derived every finding independently from the current files. All 8
previously-identified locations were independently re-checked against the
current text and found already corrected (the self-citation is gone from
`06_related.tex`/`refs.bib`/`main.bbl` entirely, not merely anonymized; the
"sweep"/"wave"/"repaired" narrative language is gone from all four files,
replaced with direct statements of the mechanism/finding). No regressions
and no new instances of that pattern were found elsewhere in the draft.

---

### Banned words

Checked every word in the verbatim list (`honest, actually, really, just,
clearly, obviously, interestingly, nicely, remarkable, surprising,
unfortunately, essentially, wildly, literally, parsimonious, cleanest,
sharpest`), case-insensitive, whole-word, across `main.tex` and all seven
section files. **Zero hits.**

---

### First-person / narrative-process

Checked for first-person-about-the-author narration ("our original
hypothesis," "the paper's sharpest claim," lab-notebook process words like
"sweep," "wave," "repaired," "first sweep," "original sweep") across all
eight files. **Zero hits.** Specifically re-verified the phrasing at each of
the 7 locations named in the prior round's report:

- `01_intro.tex` (padding-fix sentence) now reads "Correcting the target's
  padding (Appendix~\ref{app:damb}) recovers the razor's sufficiency
  result." — states the finding, no process narration.
- `02_setup.tex` (target-embedding sentence) now names the two conditions as
  "the observational arm" / "the causal-razor arm," no "first sweep"/
  "repaired wave" framing.
- `05_causal_razor.tex` (confound paragraph) now reads "Identity-padding the
  target confounds the razor by handing every capped arm a free rank-2 loss
  reduction (Appendix~\ref{app:damb}); the zero-padded target above removes
  the confound." — states the mechanism, no sweep narrative.
- `07_limitations.tex` Appendix C, all three flagged sentences now read
  "The identity-padded force-rank target...", "...so no capped cell under
  identity-padding ever tested the law's confirm direction," and "...in the
  zero-padded grid, a deliberately eye-padded corroboration arm reproduces
  the failure on demand...," and "The zero-padded grid's 30 cells were
  configuration-verified..." — all direct, no "sweep"/"wave"/"repaired"
  language.

Also checked the remaining Appendix C/B prose for softer process-adjacent
phrasing not previously flagged ("Three raw-artifact signatures established
this: (i)...(ii)...(iii)...", "configuration-verified one-by-one against a
manifest re-derived independently from the design record," "an additional
disclosed optimization shortfall," "an unconfirmed hypothesis," "which is
why the main text uses the fixed pre-registered literal... rather than a
self-referential recompute"). None of these narrate a discovery story or
grade the paper's own work; they present enumerated evidence, disclose a
known limitation, or cross-reference which of two numbers is authoritative
and why — standard pre-registration/robustness reporting, not "the story of
finding it." Not flagged.

No first-person singular ("I"/"my"/"me") found anywhere in rendered prose.

---

### Contractions

Checked every `word'word` pattern across all eight files. All matches are
possessives ("task's," "group's," "encoder's," "instrument's," "law's,"
"model's," "seed's," "razor's," "pdfTeX's," etc.), not contractions. The one
true contraction-shaped token, "DON'T" (`main.tex:41`), is inside the
`%%%% DON'T CHANGE %%%%` build-machinery comment banner, excluded per scope.
**Zero hits in rendered prose.**

---

### Em-dash-as-pause

Checked for the literal Unicode em dash (—) and for LaTeX `---` across all
eight files. The only em-dash characters found are inside `%%%%`-prefixed
comment lines (`main.tex:1`, `main.tex:8`), excluded per scope. All `--`
(en-dash) occurrences in rendered prose are correct numeric-range usage
("76--95\%," "8k--40k," "4.9--10.2\%," "0.10--0.15," "0.876--0.879," "S3/S5
key--value"). **Zero hits.**

---

### Headings

All `\section{...}` headings (Introduction; Tasks, Models, and the Rank
Instrument; The Rank Law on Group Composition, Observed; The Causal Razor;
Related Work and Limitations; five appendix section titles) and all bold
paragraph lead-ins (`\textbf{Tasks.}`, `\textbf{Model.}`,
`\textbf{Instrument.}`, `\textbf{The provable foundation (binding).}`,
`\textbf{Dimension, not solvability.}`, `\textbf{Limitations.}`) are noun
phrases naming content, not questions. **Zero hits.**

Structural note (not a style-judge category, informational only):
`sections/03_binding.tex` has no `\section{}`/`\subsection{}` command — its
"The provable foundation (binding)" content renders as an unnumbered bold
lead-in continuing inside Section 2's flow in the compiled PDF (confirmed
against `main.pdf` text extraction). Not a style violation since no heading
is rhetorical or missing outright; flagged only as a possible structural
item for the format auditor, not scored here.

---

### Captions

All figure/table captions (`fig:tracking`, `fig:razor`, `tab:groups`,
`tab:m1`, `tab:gate1a`) name the subtask, method/ablation, and takeaway, and
are complete without deferring to body text. No "pending," "TODO," or "will
be added" strings appear in any caption. **Zero hits.**

---

### Abstract length

Word count of the abstract (`main.tex`, between `\begin{abstract}` and
`\end{abstract}`, comment tokens stripped, LaTeX macros/inline math counted
as the single typeset token they render to, e.g. `$K{=}8$` → "K=8"):
**227 words**, cross-checked against the compiled `main.pdf` text extraction
(identical prose, matches source). Within the required 200-230 band, 3 words
of margin. No trim needed.

---

### DO-NOT / apparatus

Checked for `gpu`, `h100`, `a100`, `$` (literal currency), `dollar`,
`funded`, `funding`, `acknowledg`, `self-funded`, `budget` (as compute-cost
language), and experiment-count bragging phrasing ("we ran N experiments")
across all eight files. All apparent hits are false positives: "step
budgets" refers to the pre-registered training-step allocation per group (a
methodological design parameter, not money), and `$`-signs are LaTeX math
delimiters. Cell/seed counts that do appear ("39 cells," "30 cells,"
"$n=5$ per group," "19 seeds") are precise methodological grid sizes
required to assess statistical power, not apparatus bragging. **Zero hits.**

---

### Anonymization (double-blind)

Ran the anonymization grep across `main.tex`, all seven section files,
`refs.bib`, and `main.bbl` for: `larson`, `samlarson`, `saml212`, `pebble`,
`pebbleml`, `rockie`, `github.com/`, `huggingface.co/`, `acknowledg`,
`self-funded`, `funded by`. **Zero matches anywhere**, including in
`refs.bib` (14 entries, all third-party citations — Nichani, Nazari, Sun,
Grazzi, Siems, Mishra, Kohonen, Anderson, Merrill, Barrington, Chughtai,
Liu, Delétang) and `main.bbl` (compiled bibliography, 14 `\bibitem`s,
confirmed no `Larson` entry present). The self-citation to the authors'
prior workshop paper that the earlier review round flagged as a hard
identity leak (`\citep{larson2026gradient}` in `sections/06_related.tex`, an
`author = {Larson, Sam}` entry in `refs.bib`, and a plain-text "Sam Larson"
bibitem in `main.bbl`) is absent from the current draft in its entirety —
the citation and its sentence were removed, not merely anonymized. The
author block in `main.tex` remains fully commented out (all lines prefixed
`%`), consistent with the template's documented review-build behavior.
**Zero hits.**

---

## Verdict

**PASS**
