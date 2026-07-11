# Style judge — gauntlet round 1, stage 03 (`03_style_critique.md`)

Role: `paper` skill style judge (`prompts/style-judge.md`) enforcing
`references/styleguide.md`, run fresh against the CURRENT draft after the
FIX-1..FIX-10 rebuttal wave (byte-footprint paragraph, reasoning-proxy
paragraph, held-out-axis disclosure paragraph, four new citations, two
wording qualifiers). The pre-gauntlet pass record (`00_style_pregate.md`)
was NOT trusted; every category was re-checked from scratch.

Scope checked: `sections/00_abstract.tex` through `sections/08_conclusion.tex`
(all nine section files, in full), `bundle/main.tex` (front matter + assembly),
all four figure/table captions, and `bundle/refs.bib` for identity leaks.
Banned-word and identity greps were run whole-word, case-insensitive; LaTeX
command names and citation keys were excluded from banned-word scanning.

## Findings by category

### Banned words
None. Whole-word case-insensitive grep for all 17 verbatim banned tokens
(`honest, actually, really, just, clearly, obviously, interestingly, nicely,
remarkable, surprising, unfortunately, essentially, wildly, literally,
parsimonious, cleanest, sharpest`) returned zero prose hits across all
sections and `main.tex`. Stem-level near-matches were checked and cleared:
every `actual` occurrence is inside the citation key `nichani2025factual`
(LaTeX command argument, not prose); every `sharp` occurrence is the allowed
words "sharp"/"sharply"/"Sharp Transition" heading or the citation key
`barnfield2026sharp` — the banned token "sharpest" never appears. The
pre-gate's single hit ("honest", `05_frontier.tex`) is confirmed already
reworded to "The supportable summary".

### First-person / narrative-process
None. The only "I" match is the identity matrix `$I$` in the DeltaNet update
(`02_testbed.tex:8`, math mode). No "my"/"me". No contractions of first
person. All "we"/"our"/"this paper"/"the paper's" usages were reviewed
individually and are standard editorial-"we" (method: "we measure", "we
inject", "we exclude"; scoping: "we make no functional-form claim", "We
therefore scope") or conventional contribution framing ("This paper supplies
that frontier", "which is this paper's object"). None matches a banned
narrative-process pattern ("our original hypothesis", "we report a negative
result with a mechanism", "the paper's sharpest claim", "reports falsification
of our own hypothesis").

Reviewed-and-cleared (not counted; noted for the detector judge's interpretive
pass, both descriptive rather than self-grading, and neither uses a banned
token):
- `02_testbed.tex:24` "The paper's single load-bearing quantity is $h4$" —
  echoes the STRUCTURE of the banned "the paper's sharpest claim" but the
  descriptor ("single load-bearing quantity") is neutral, not a self-grading
  superlative. If the writer wishes to remove the self-reference entirely,
  "The single load-bearing quantity is $h4$" reads identically.
- `04_dose.tex:56` "this is the strongest exoneration the design can produce"
  — "strongest" is not a banned token; the phrase is methodologically scoped
  to the design's own limit rather than promotional. Optional tightening:
  "this is the maximal exoneration a frozen-dose design can produce".

### Contractions
None. Every apostrophe in the prose is possessive (`state's`, `paper's`,
`gate's`, `load's`, `d=64 table's`, `diagnostic's`, `frontier's`, etc.).
Targeted grep for `it's / there's / that's / what's / who's / let's` and for
the `n't`/`'re`/`'ll`/`'ve`/`'m` families returned zero hits.

### Em-dash-as-pause
None in rendered prose. The two Unicode em-dash (`—`) characters occur only in
non-rendering LaTeX comment lines (`00_abstract.tex:1`, `main.tex:1`) and do
not typeset. Every `--` in body text is a numeric-range or named-pair en-dash
(`1--3`, `1--2`, `3--4`, `$31$--$39$`, `$0.9963$--$1.0$`, `Newton--Schulz`) or
the compound `key--value cache` (`01_introduction.tex:5,80`) — correct
typography, not a conversational pause.

### Headings
None. All `\section`/`\subsection`/`\paragraph` headings are noun phrases or
declarative fragments ("The Transition, Located and Dissolved"; "The Capacity
Frontier Grows Super-Linearly"; "A Sharp Transition at $d_{\mathrm{state}}=64$";
"The Coherence Confound, Exonerated Twice"). Zero question-mark headings
(verified by grep on the heading commands). Note (reviewed, not a violation):
the abstract opens with a question ("How many independent bindings does a
fixed-size fast-weight state hold before recall collapses?") — this is body
prose, not a section heading; styleguide rule 5 bans rhetorical-question
HEADINGS only, and an abstract opening question is a conventional device.

### Captions
None. All four captions are self-contained with an explicit stated takeaway:
- fig:cliff (`03_cliff.tex:7`) — names task, readout, both panels, and
  "Takeaway: the same load ratio that collapses recall at
  $d_{\mathrm{state}}=64$ is loss-free at $d_{\mathrm{state}}=128$."
- fig:dose (`04_dose.tex:7`) — names geometry, all three marker families,
  the grey band, and "Takeaway: directly injected coherence ... does not move
  recall off ceiling."
- fig:frontier (`05_frontier.tex:7`) — names every plotted element and
  "Takeaway: the safe load ratio itself grows with state dimension."
- tab:bytes (`05_frontier.tex:104`) — defines $K^{*}$, the columns, and the
  `$\dagger$` caveat inline.
No caption defers to body text for meaning, and none contains
"pending"/"TODO"/"will be added". (Section cross-references present are for
provenance, not deferral of meaning.)

### Abstract length
In band. `00_abstract.tex` counts **215 words** with each `$...$` math span
counted as one word (13 spans), or **202 words** counting prose only and
excluding math spans — both inside the required 200-230 band. LaTeX comment
line 1 and the `% evidence: Cn` markers were excluded from the count. No
trim or expansion needed.

### DO-NOT / apparatus
None. No GPU-hour figures, no dollar/compute-cost figures, no funding-source
language, no hardware bragging, no experiment-count bragging in the prose.
(The DO-NOT cost grep's `$[0-9]` hits are all LaTeX math delimiters such as
`$1.000$`, `$0.6779$` — not currency.) The word "audit" does not appear and
is in any case explicitly NOT banned for this paper per `brief.md`.

### Anonymization (double-blind)
No author-identity leak. Full-tree case-insensitive grep of the anonymization
token list (`author surname/given name`, `saml212`, `samlarson16`, `pebble`,
`pebbleml`, `learned-representations`, `idastone`, `Brev`,
`youthful-indigo-turkey`, `github.com/`, `huggingface.co/`, `acknowledg`,
`self-funded`, `funded by`) across `sections/`, `bundle/main.tex`, and
`bundle/refs.bib` returned exactly one match:

- `bundle/main.tex:3` — `% (github.com/COLM-org/Template, release tag 2026)`.

Reviewed and cleared (NOT counted as a violation): this is (i) a
non-rendering LaTeX comment that does not appear in the compiled PDF, and
(ii) a pointer to the COLM organization's public template repository, not to
the author — it does not de-anonymize. Consistent with the pre-gate's
exemption. Recommendation for the format auditor (stage 05): strip or
neutralize the URL in this comment so the mechanical bundle grep runs clean,
but it is not a blocking identity leak. `refs.bib` author fields are all
external cited authors (Katharopoulos, Schlag, Yang, Nichani, Barnfield,
Nazari, Sun, ...); no self-citation carries the author's name. The
`author = {Anonymous Authors}` block (`main.tex:27`) is the kit's anonymized
`[submission]` placeholder. Identity tokens found in `brief.md` and
`venue-requirements.md` are internal planning docs, NOT part of the submission
bundle, and are the anonymization token list itself.

## Verdict

PASS
