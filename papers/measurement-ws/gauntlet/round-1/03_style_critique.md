# Style Critique — measurement-ws (Gauntlet Round 1, Stage 03)

Judge: style judge (Stage 4). Scope: rendered prose only — `main.tex` plus
`sections/00_abstract.tex` through `sections/09_appendix.tex`, including all
figure/table captions. LaTeX comment lines/spans (after `%`) were excluded.
Venue: double-blind. Contract:
`platform-skills/skills/paper/references/styleguide.md`. Judged cold, as
written, after the 16-fix rebuttal wave and compression pass.

## Method

- Banned-word list read verbatim from the styleguide each run; grepped
  case-insensitive, whole-word, across all rendered `.tex` files.
- First-person, contractions, dashes, anonymization tokens, headings, and
  captions each swept mechanically; every hit inspected in context.
- Abstract extracted between `\begin{abstract}`/`\end{abstract}` with comment
  lines stripped, then word-counted.

## Findings by category

### Banned words
None. All 17 banned tokens (honest, actually, really, just, clearly,
obviously, interestingly, nicely, remarkable, surprising, unfortunately,
essentially, wildly, literally, parsimonious, cleanest, sharpest) return zero
whole-word hits in the prose.

### First-person / narrative-process
None. Every regex hit on `I`/`my`/`me` is the identity matrix `$I$`
(05_case_gauge lines 10, 59; 09_appendix lines 8, 32) or a "Case I/II/III"
roman-numeral case label (03_case_tolerance §title; 07_not_flipped line 18;
09_appendix line 150). No first-person singular pronoun appears. Editorial
"we"/"our" is used correctly throughout: "We report six..." (abstract),
"Our contributions:" (01_intro line 15, standard contributions list), "the
inverse of our thesis" (08_related_limits line 22, referring to the paper's
central claim, not narrating the discovery process), "we report it firing
live" (08_related_limits line 28). None crosses into banned narrative-process
phrasing ("our original hypothesis", "the paper's sharpest claim", etc.).

### Contractions
None. Every apostrophe in the prose is a possessive (task's, lens's, sweep's,
model's, primary's, group's, tiebreak's, ...). Negations are spelled out
throughout ("did not", "does not", "cannot", "is not").

### Em-dash-as-pause
None. No LaTeX `---` and no unicode em-dash/en-dash anywhere. The only `--`
occurrences are the compound proper name `Newton--Schulz`
(03_case_tolerance line 8), the numeric range `rows 4--6` (06_three_lenses
line 4), and HTML-style evidence comment markers, none of which is a
conversational pause.

### Headings
None. All section headings are noun phrases: "Introduction", "The
Adjudication Discipline", "Case I: The Tolerance That Manufactured a Cliff",
"Case II: The Instrument at the Wrong Layer", "Case III: The Gauge That Did
Not Transfer", "Three More Lenses, Briefly", "The Boundary: What the
Discipline Did Not Flip", "Related Work and Limitations", "The Three Brief
Lenses, in Full", "The Catalogue and Per-Incident Detail Tables". No
rhetorical-question heading. Run-in bold headers ("The signature.", "The
tiebreak.", "Teeth.", "The re-metric.", "Rule 1..5") are noun phrases as well.

### Captions
None. All captions are self-contained: fig:lens, tab:catalogue, fig:taploc,
tab:walk, tab:taps, tab:remetric, tab:teeth each name the case/subtask, the
method or ablation, and the takeaway. No caption defers to body text or says
"pending"/"TODO"/"will be added".

### Abstract length
211 words (comment lines stripped). Inside the 200-230 band. No action.

### DO-NOT / apparatus
None. No "audit" anywhere (the paper uses "adjudication"). No experiment-count
bragging in the DO-NOT sense (cell counts and episode counts describe the
experimental object, not a fleet), no dollar figures, no funding language, no
"we ran N"/"our large GPU" phrasing. The GPU-hour figures in
08_related_limits ("$0.50$ and $1.10$ GPU-hours ... $6.33$ GPU-hours") are the
deliberate, evidence-mapped economics claim the task brief explicitly
whitelists for this project; not counted. Machine verdict strings
(`\textsc{falsify}`) and metric names (rf@0.9, n_iter) are technical objects,
correctly presented; not counted.

### Anonymization (double-blind)
None in rendered prose. The only repo-identifying token found,
"matrix-thinking", occurs on `main.tex` line 15 inside a LaTeX comment (a
`%`-prefixed provenance note), which is not rendered and is explicitly
permitted by venue-requirements.md ("internal section anchors ... cited in
source comments only, never in rendered text"). No author surname, handle,
org name, de-anonymizing URL (github.com/, huggingface.co/), or
acknowledgment/funding language appears anywhere. Courtesy note for the
format auditor (Stage 05): if the submission bundle ships `.tex` source rather
than PDF-only, strip or verify that comment before the bundle grep.

## Verdict

PASS

Abstract word count: 211 (band 200-230).
