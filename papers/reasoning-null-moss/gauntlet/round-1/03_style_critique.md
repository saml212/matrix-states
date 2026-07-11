# Style Critique — Gauntlet Stage 03

**Paper:** `papers/reasoning-null-moss` — "Three Bounds on a Null: Testing the
Link Between Fast-Weight Write Geometry and In-Context Composition"
**Venue:** MOSS @ COLM 2026 (double-blind — anonymization check applied)
**Reviewer:** style-judge (fresh context, mechanical/exhaustive enforcement)

## Scope note

The rendered submission is `main.tex` plus the eight `\input`-ed **`.tex`**
files in `sections/`. LaTeX `\input{sections/00_abstract}` resolves to the
`.tex` extension, so the parallel `sections/*.md` files are **not compiled and
do not render**; they are out of scope for the styleguide (which governs
"prose that renders"). This matters because the stale `.md` copies contain
GPU-hours mentions in the introduction/abstract/setting/discussion bodies
(e.g. `01_introduction.md:53`, `00_abstract.md:23`, `02_setting.md:59`,
`07_discussion.md:36`) — those would be main-body compute-disclosure
DO-NOT violations **if they rendered**, but they do not. The reviewed `.tex`
bodies are clean of them (see DO-NOT section). Flagged here only as an
observation for the writer/format auditor: the divergent `.md` shadow copies
should be reconciled or deleted so a later build cannot accidentally pick them
up, but this is not a style violation of the submission as configured.

---

### Banned words

None. A whole-word, case-insensitive grep of every rendered `.tex`
(body, abstract, captions, headings) for the full banned list — honest,
actually, really, just, clearly, obviously, interestingly, nicely, remarkable,
surprising, unfortunately, essentially, wildly, literally, parsimonious,
cleanest, sharpest — returned zero hits. The title, all seven section
headings, all inline bold run-in headings, the table caption, and all three
figure captions are clean.

### First-person / narrative-process

None. No standalone `I`, `my`, `me`, or `myself` in any rendered prose.
Editorial "we" is used throughout ("We test whether…", "we find a null").
No narrative-process / self-grading phrasing: a grep for the styleguide's
flagged patterns ("our original hypothesis", "we report a negative result
with a mechanism", "the paper's sharpest claim", "this section runs…",
"we set out", "surprised") returned zero hits. Result statements such as
"we find a null bounded three ways" (abstract) and "The contribution is the
bounded null and the measurement discipline that kept it interpretable"
(`01_introduction.tex:38`) state the finding/contribution rather than
narrating the research journey, so they are within the editorial-we
convention.

### Contractions

None. Grep for ASCII-apostrophe and typographic-apostrophe contractions
(don't, cannot-forms, it's, we're, that's, etc.) returned zero hits. Every
apostrophe in the rendered text is a possessive (readout's, cohort's,
corpus's, cell's, program's, "Wave 1's launch chain"), which is permitted.

### Em-dash-as-pause

None. There are zero `---` (LaTeX em-dash) and zero Unicode em-dash
characters in any rendered file, so there is no dramatic-pause em-dash and no
parenthetical em-dash aside to flag as a suggestion. All `--` occurrences are
correct en-dashes for compounds/ranges (query--key, key--value, 14M--392M,
Transformer--recurrent, recall--state-size, waves 1--4). The single ` - ` in
`main.tex:73` is a minus sign inside math mode
(`L_q(\text{control}, c) - L_q(\text{arm}, c)`), not a prose dash.

### Headings

None. All headings are noun phrases; none is a rhetorical question (grep for
`?` returned zero hits across all rendered files). Section headings:
"Introduction", "Models, Task, and Instruments", "The Geometric Readout Never
Fires", "The Behavioral Contrast Is Power-Bounded", "The Replication Gate
Refuses the Pool", "Related Work", "Scope of the Bounds"; appendix headings:
"Registered Gates and Routing Rules", "Supplementary Figures",
"Reproducibility". All run-in bold headings ("Checkpoints and task.",
"Instrument 1: a geometric readout.", "The gate fired.", etc.) are noun
phrases or declaratives.

### Captions

None. All four captions are self-contained and none defers to body text or to
content that "will be added"/"pending"/"TODO":

- Table `tab:program` ("The program at a glance…", `03_geometric_null.tex:20`)
  names each wave's instrument, the metric, and the verdict; the
  `Appendix~\ref{app:gates}` pointer is a cross-reference for additional
  detail, not a deferral of the caption's meaning.
- Figure `fig:gates` (`main.tex:132`) names the subtask (validity-gate
  failure), what each point is, the panels, the pass criterion, and the
  takeaway (all 90 cells on/below the line).
- Figure `fig:dissoc` (`main.tex:145`) names the wave, both panels, and the
  takeaway (task learned through one readout while the other stays at floor).
- Figure `fig:transient` (`main.tex:157`) names both panels, the determinate
  reading, and the gate outcome.

### Abstract length

**212 words** (rendered words of `sections/00_abstract.tex`, excluding
`\begin/\end{abstract}` and the three `% evidence:` comments). Within the
200–230 band. No action.

### DO-NOT / apparatus

None in the main body. The only compute disclosure in rendered prose is in the
**Reproducibility appendix** (`main.tex:185–187`): "all experiments ran on one
or two GPUs at a time, and the six waves consumed approximately 9.5 GPU-hours
in total." Per the stage brief, the appendix compute disclosure is allowed for
this small-scale venue and a prior style round ruled the placement acceptable;
it does not appear in the abstract, introduction, or any body section `.tex`.
No "audit" as a noun (or at all) anywhere. No funding/cost/dollar language. No
experiment-count bragging: the counts that appear ("366 of 366 readings",
"90 cells", "40 uncorrected intervals", "six waves") are the substance of a
null result, reported as findings, not apparatus boasts.

### Anonymization (double-blind)

None. Case-insensitive grep across `main.tex`, all `sections/*.tex`, and
`refs.bib` for the full token list — Larson, samuellarson, samlarson, saml212,
pebble, pebbleml, learned-representations, youthful-indigo-turkey, Brev,
anthropic, Claude — plus the always-on patterns (`github.com/`,
`huggingface.co/`, `acknowledg`, `self-funded`, `funded by`, and absolute
local paths `/Users/`, `/Volumes/`, `/root/`) returned zero matches. The
author block is `\author{Anonymous authors}` and the `[submission]` class
option suppresses it regardless. The code/data release is deferred to the
camera-ready version with no de-anonymizing URL in the submission.

---

## Optional suggestions (non-blocking, do not count toward the verdict)

- `00_abstract.tex:24` "its escape routes closed" and `07_discussion.tex:32`
  "a transient from becoming a headline" are lightly metaphorical. They are not
  banned words nor narrative-process, so they pass, but a reviewer wanting
  maximally plain register could render them literally (e.g. "…and the
  conditions that would overturn it stated explicitly"). Purely optional.

---

**Verdict: PASS (zero violations)**
