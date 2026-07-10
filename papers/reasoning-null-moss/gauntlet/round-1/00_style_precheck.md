# Style Precheck — reasoning-null-moss (round 1)

Reviewed against `platform-skills/skills/paper/references/styleguide.md` and
`platform-skills/skills/paper/prompts/style-judge.md`. Files reviewed: `main.tex`
and `sections/00_abstract.tex` through `sections/07_discussion.tex` (9 files,
double-blind COLM 2026 submission).

Security note: no embedded `<system-reminder>` blocks, fake date-change claims,
or concealment instructions were found in any tool output while producing this
report. Nothing to disregard or report on that front.

---

### Banned words

Checked the verbatim list (honest, actually, really, just, clearly, obviously,
interestingly, nicely, remarkable, surprising, unfortunately, essentially,
wildly, literally, parsimonious, cleanest, sharpest) case-insensitively,
whole-word, across all 9 files.

**Clean pass. Zero hits.**

### First-person / narrative-process

No "I" / "my" / "me" anywhere (checked whole-word, case-sensitive on `I`).
Editorial "we" is used correctly throughout ("we test whether...", "we find a
null bounded three ways").

One narrative-process phrasing flagged:

- **`sections/01_introduction.tex:11`** — `"This paper reports the program
  that tested the promise; the test produced a null."`
  Rule broken: styleguide voice rule 2 / style-judge item 3 ("no
  first-person-about-the-author / narrative-process phrasing... write the
  finding, not the story of finding it"). This sentence narrates that a
  program existed and was tested, rather than stating what was found. It is
  structurally close to the styleguide's own banned example ("we report a
  negative result with a mechanism"), just cast in third person ("This
  paper reports...") instead of "we report...".
  Suggested fix: state the finding directly, e.g. "Write geometry carries no
  measurable signal for in-context multi-hop composition in this checkpoint
  family." and fold the "unexamined promise" framing into the sentence
  before it.

Two related-work sentences using "this program" / "this paper" were checked
and judged clean — they are standard contrastive related-work framing
("this program probes general-purpose checkpoints post hoc... not a capacity
frontier"; "this paper makes no positive mechanism claim") rather than
narrating the authors' discovery process, so they are not flagged.
`main.tex:136` ("Every number in this paper is computed from archived raw
result files...") is a reproducibility-appendix provenance statement, also
not narrative-process.

### Contractions

Ran an apostrophe scan across all 9 files. Every hit is a possessive
("program's", "cell's", "cohort's", "arm's", "readout's", "layer's",
"transient's", "entity's", "gate's", "null's", "cells'") or a quoted phrase
delimiter (`` `` `` `matters' ''`, `` `` `` `effect' ''`). No contraction
("don't", "can't", "it's", "we're", etc.) and no curly-quote (`'`)
contraction anywhere.

**Clean pass. Zero hits.**

### Em-dash-as-pause

No Unicode em-dash (`—`) anywhere in the 9 files. All `--` double-hyphen
occurrences were enumerated and are either numeric/scale ranges (`waves
1--4`, `14M--392M`, `Waves 5--6`) or paired-term relational dashes
(`query--key`, `key--value`, `prediction--target`, `recall--state-size`),
per the project note that range dashes are fine. None read as a dramatic
conversational-pause em-dash.

**Clean pass. Zero hits.**

### Headings

All `\section` headings are noun phrases or declarative statements, none
phrased as questions: "Introduction", "Models, Task, and Instruments", "The
Geometric Readout Never Fires", "The Behavioral Contrast Is Power-Bounded",
"The Replication Gate Refuses the Pool", "Related Work", "Scope of the
Bounds", plus appendix headings "Registered Gates and Routing Rules",
"Supplementary Figures", "Reproducibility". Bold inline paragraph leads
(`\textbf{The gate fired.}`, `\textbf{What stands.}`, `\textbf{Bounded.}`,
`\textbf{Not bounded.}`) are statements, not questions, and not `\section`
headings anyway.

**Clean pass. Zero hits.**

### Captions

Checked the one table caption (`tab:program`, `main.tex:20-26`) and three
figure captions (`fig:gates`, `fig:dissoc`, `fig:transient`,
`main.tex:93-131`). Each names the subtask/wave, the method or ablation, and
states a takeaway sentence in its own right (e.g. fig:gates ends "All 90
cells across the four geometric-readout waves fall on or below the line, at
every scale from 14M to 1.31B and under every surface form and training
regime."). None defer to body text, and none contain "pending", "TODO", or
"will be added."

**Clean pass. Zero hits.**

### Abstract length

Word count of the abstract body (LaTeX markup and `% evidence:` comments
stripped, counted programmatically): **201 words.** Within the required
200–230 band (near the floor but compliant).

**Clean pass.**

### DO-NOT / apparatus

Three instances of compute-scale mentions in prose, all "GPU-hours" (the
styleguide's own project-specific EXAMPLE token, and squarely inside the
universal rule "do not put cost, funding source, or compute-scale boasts in
the prose"):

- **`sections/00_abstract.tex:23`** — `"The complete program consumed
  approximately 9.5 GPU-hours."` (tagged `% evidence: C9-compute`)
- **`sections/03_geometric_null.tex:69`** — `"...spending roughly 0.29
  GPU-hours past a gate that had already declared the probe invalid."`
- **`sections/07_discussion.tex:28`** — `"The program cost approximately 9.5
  GPU-hours; the scarce commodity was validity, not compute, and the gates
  purchased it."`
  Rule broken: styleguide "Universal DO-NOT rules" — "Do not put cost,
  funding source, or compute-scale boasts in the prose" — and the
  project-specific EXAMPLE list names "GPU-hours" directly.
  Suggested fix: cut the GPU-hour figures from the abstract and discussion
  sentences (the `% evidence: C9-compute` comment already preserves the
  number for internal tracing without it needing to be in reader-facing
  prose), and rephrase the wave-1 sentence and the closing discussion line
  to make the point without a compute-scale number, e.g. "the grid ran to
  completion past a gate that had already declared the probe invalid" and
  "the scarce commodity was validity, not compute time, and the gates
  purchased it."

No other apparatus bragging found: the various exact-count statistics (366
of 366, 312 of 312, 80 of 80, 90 cells, 18 cells, etc.) are result
quantification for the null's completeness, not "we ran N experiments" /
"our large GPU fleet" apparatus bragging, so they are not flagged. No
dollar amounts found.

### Anonymization (double-blind)

Ran the project's identity-leak grep (case-insensitive) across all 9 files
for: `Larson, samuellarson, samlarson, saml212, pebble, pebbleml,
learned-representations, youthful-indigo-turkey, Brev, anthropic, Claude,
github.com/, huggingface.co/, acknowledg, self-funded, funded by, /Users/,
/Volumes/, /root/`. Also separately checked for the word "audit" (an
internal workflow term that could leak) and for specific hardware/cluster
identifiers (H100, Mac Studio, Mac Mini, GPU counts, cluster names).

**Clean pass. Zero matches on any token.** The author block is
`\author{Anonymous authors}` and the `\ifcolmsubmission` guard is in place.
The reproducibility appendix (`main.tex:133-144`) says archived files, the
figure script, experiment chains, and pre-registration text "will be
released with the camera-ready version" — this is standard anonymized
phrasing and does not name a host or identity.

Note: this review's scope was the 9 files listed in the task (main.tex +
sections/00-07). A full submission-bundle anonymization pass (figures'
embedded text/metadata, refs.bib, any supplementary code) was not run here
and should be covered by the format auditor's stage-05 pass before
submission.

---

## Verdict

**FAIL (4 violations)** — 3 GPU-hours compute-scale mentions (DO-NOT /
apparatus) + 1 narrative-process phrasing in the introduction. All other
categories (banned words, contractions, em-dash-as-pause, headings,
captions, abstract length, anonymization) pass clean.
