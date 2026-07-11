# Style critique — pre-gauntlet pass (measurement-ws)

Judge: style-judge per `platform-skills/skills/paper/prompts/style-judge.md`,
contract `references/styleguide.md`. Scope: rendered prose of `main.tex` +
`sections/00_abstract.tex`–`09_appendix.tex`, including all figure/table
captions and section headings; LaTeX source comments excluded per the judge
brief. Checked 2026-07-10.

## VERDICT: PASS

Zero violations. Category-by-category record of what was checked and how:

### Banned words
Zero hits. All 17 verbatim tokens (`honest, actually, really, just, clearly,
obviously, interestingly, nicely, remarkable, surprising, unfortunately,
essentially, wildly, literally, parsimonious, cleanest, sharpest`) grepped
case-insensitive, whole-word, unfiltered, across `main.tex` and every file in
`sections/`. Adverbial near-variants (`remarkably`, `surprisingly`, `honestly`)
also checked: zero hits. Note: "cleanly" (03, 05) and "silently" (00, 03) are
not on the list and were not flagged.

### First-person / narrative-process
Zero violations. Editorial "we" used throughout ("We report", "We argue",
"Our contributions", "we cannot report"). Every whole-word `I` match is either
the identity matrix in math mode ($\hat{Q}\approx I$, $\rho\oplus I$,
$ZZ^\top\approx c^2 I$) or the Roman numeral in "Case I". No "my"/"me". The
banned narrative-process patterns ("our original hypothesis", "the paper's
sharpest claim", "we report a negative result with a mechanism") are absent;
the incident chronologies in §§3–5 are the paper's subject matter, written
impersonally, not authorial process narration.

### Contractions
Zero. All apostrophe forms in the prose are possessives ("fit's", "lens's",
"task's", "Case I's", "the paper's"). "Winner's Curse?" appears only as a
cited paper title in the bibliography.

### Em-dash-as-pause
Zero em-dashes in the entire draft (no `---`, no Unicode em-dash). En-dashes
appear only in "Newton--Schulz" and bibliography page ranges, both correct.

### Headings
All ten section/appendix headings are noun phrases; none is a rhetorical
question. ("Three More Lenses, Briefly" carries a trailing adverb but is
noun-headed and declarative; not a violation.)

### Captions
All seven captions (Fig. `fig:lens`, Fig. `fig:taploc`, Tables
`tab:catalogue`, `tab:walk`, `tab:taps`, `tab:remetric`, `tab:teeth`) are
self-contained: each names what is plotted/tabulated, the measurement and
condition, and states the takeaway. No caption defers to body text or to
content "to be added"; no "pending"/"TODO".

### Abstract length
201 words — inside the 200–230 band (verified by two independent counting
methods, both 201). Near the low edge; any trim during later rounds must not
drop it below 200.

### DO-NOT / apparatus
No experiment-count bragging, no fleet/hardware boasts, no dollar figures, no
funding language. The single compute-cost mention
(`sections/08_related_limits.tex`: "under 2 GPU-hours against the 6.33
GPU-hours of training") was judged on the boast axis per the task brief: it is
a cost-effectiveness ratio arguing that lens auditing is cheap, quantified and
small-scale — the opposite of a compute boast. Not a violation. This project's
brief/venue-requirements declare no project-specific banned term (the
styleguide's "audit"/GPU-hours bans are labeled EXAMPLES for a different
project), so "auditing" in the same sentence is likewise not a violation.

### Anonymization (double-blind)
Zero identity leaks in rendered prose or bibliography. The full project token
list from `venue-requirements.md` § anonymization was run (`Larson`,
`samlarson16`, `pebble`, `pebbleml`, `idastone`, `Anthropic`, `brev`,
`learned-representations`, `matrix-thinking`, `KEY_ANCHORING`,
`CAPABILITY_SEPARATION`, `HEAD_TO_HEAD`, design-doc filenames, internal §
anchors, `github.com/`, `huggingface.co/`, `acknowledg`, `funded by`,
`self-funded`): zero matches in rendered text. The author block is
"Anonymous Author(s)"; the companion self-citation (`companion2026capacity`)
is anonymized (`author = {Anonymous}`, @unpublished, no venue/URL).

**Non-blocking source note (out of scope for this verdict):** `main.tex`
line 15 contains the string `matrix-thinking/submissions/measurement-2026/`
inside a LaTeX comment. Comments are invisible in the render and exempt per
the brief ("source comments exempt"), but if the chosen venue requires TeX
source upload, the bundle flattening must strip comments as `brief.md`
already specifies.

### Quantified claims
Spot-checked: headline claims carry numbers and scope ("recall accuracy
0.9990 against chance 0.03125", "37 of 39 cells within 0.07", "residuals near
1.2e-6, roughly 7,800x below tolerance", "per-load mean recovery 0.92 to
1.00"). The one qualitative-sounding claim ("no measurable separation from
the baseline at the decisive groups", §5) is quantified in
Table `tab:remetric`. No unquantified performance claim found.

## VERDICT: PASS (0 violations)
