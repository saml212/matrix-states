# Style Precheck v2 — `reasoning-null-moss` (round 1)

Reviewer: style judge (fresh context, no knowledge of prior review rounds).
Scope: `main.tex` + every `.tex` file in `sections/` (the `.md` files in
`sections/` are drafting scratch, not the LaTeX source under review, and
were excluded per the task instructions).

Files reviewed:
- `main.tex`
- `sections/00_abstract.tex`
- `sections/01_introduction.tex`
- `sections/02_setting.tex`
- `sections/03_geometric_null.tex`
- `sections/04_behavioral_contrast.tex`
- `sections/05_replication.tex`
- `sections/06_related.tex`
- `sections/07_discussion.tex`

Method: exhaustive case-insensitive, whole-word grep for every item in the
styleguide's verbatim banned-word list, first-person singular, contractions
(`'t/'s/'re/'ve/'ll/'d/'m`, filtered for possessive `'s` false positives),
literal em-dash and `---`/spaced-`--` conversational-pause patterns,
narrative-process phrase patterns, the project's identity-leak token list,
GPU/dollar/compute-cost mentions, apparatus-bragging phrases, and a
line-by-line read of every `\section` heading and every figure/table
caption. Abstract word count computed programmatically (LaTeX commands and
`% evidence:` comments stripped before counting).

## ### Banned words

Zero hits. Checked the full verbatim list (`honest, actually, really, just,
clearly, obviously, interestingly, nicely, remarkable, surprising,
unfortunately, essentially, wildly, literally, parsimonious, cleanest,
sharpest`) whole-word, case-insensitive, across all nine files. No matches.

**Clean pass.**

## ### First-person / narrative-process

Zero hits on `I`/`my`/`me`. Editorial "we" appears three times total
(`00_abstract.tex:5,7`, `01_introduction.tex:11`), all attributing findings
("we test", "we find"), matching the sanctioned "active we for findings"
pattern.

Checked narrative-process patterns explicitly (`our original/initial
hypothesis`, `we decided/chose/hypothesized/were surprised`, `we report a
negative result with a mechanism`, `the paper's sharpest/cleanest claim`,
`this paper/section runs...reports`): zero hits.

One borderline candidate inspected and cleared: `06_related.tex:6-7` uses
"this program" twice ("the parameter-matched ladder pattern this program's
scale series follows"; "this program probes general-purpose checkpoints
post hoc"). "The program" is used throughout the draft (including the
table caption "The program at a glance," `tab:program`) as the standing
name for the six-wave measurement study, functioning as a technical noun
naming the object under discussion, not as first-person narration of the
authors' research journey ("our original hypothesis was..."). This does
not match the styleguide's narrative-process examples, which single out
self-narration of belief-formation or self-grading language. Not flagged.

**Clean pass.**

## ### Contractions

Zero hits. The apostrophe-`'s` matches found by the search pattern were
all possessives on nouns (`cell's`, `entity's`, `null's`, `gate's`,
`venue's`, `cohort's`, `program's`, `layer's`, `arm's`), not contractions.
No instance of `it's`, `don't`, `can't`, `won't`, `we're`, `didn't`,
`isn't`, `doesn't`, or any other contraction form anywhere in the nine
files.

**Clean pass.**

## ### Em-dash-as-pause

Zero hits. No literal em-dash character (`—`) anywhere in the draft, and
no LaTeX triple-hyphen (`---`) or space-padded double-hyphen (` -- `,
the typical rendering of a conversational-pause em-dash in source). Every
`--` instance found is an unspaced range or word-pair dash per the
project's exemption (number ranges: `1--4`, `14M--392M`, `5--6`; term
pairs read the same way: `query--key`, `key--value`, `prediction--target`,
`recall--state-size`). None are dramatic pauses.

**Clean pass.**

## ### Headings

All ten `\section` headings are noun phrases or declarative statements
naming content, matching the styleguide's own example pattern (its
sanctioned example, "The Flatten-Then-Project Readout Is Rank-Blind," is
itself a full declarative sentence, not a bare noun phrase): "Introduction,"
"Models, Task, and Instruments," "The Geometric Readout Never Fires," "The
Behavioral Contrast Is Power-Bounded," "The Replication Gate Refuses the
Pool," "Related Work," "Scope of the Bounds," "Registered Gates and Routing
Rules," "Supplementary Figures," "Reproducibility." None phrased as a
question. All inline bold run-in headers ("Checkpoints and task.",
"Instrument 1: a geometric readout.", "The gate fired.", "What stands.",
"Bounded.", "Not bounded.", etc.) are likewise short statements or noun
phrases, never questions.

**Clean pass.**

## ### Captions

Three figure captions (`fig:gates`, `fig:dissoc`, `fig:transient` in
`main.tex` Appendix B) and one table caption (`tab:program` in
`03_geometric_null.tex`) inspected. Each names the subtask/instrument, the
method or wave, and states the takeaway inline, with no deferral to body
text and no "pending"/"TODO"/"will be added" language:

- `fig:gates` — states what each point represents (same-entity median
  cosine vs. shuffled-entity null p95) and the takeaway ("All 90 cells...
  fall on or below the line, at every scale... and under every surface
  form and training regime").
- `fig:dissoc` — states both panels' content and the takeaway ("The model
  learns the task through one readout while the other stays at floor").
- `fig:transient` — states both panels' content, the specific numbers, and
  the takeaway (variance-ratio gate failure, no decision-grade pooled
  verdict).
- `tab:program` — states what the table enumerates and the two-instrument
  takeaway.

**Clean pass.**

## ### Abstract length

Programmatic word count (LaTeX markup and `% evidence:` comments stripped,
`{,}` thousands-separator normalized before splitting): **208 words**.
Within the required 200-230 band. No trim or expansion needed.

**Clean pass.**

## ### DO-NOT / apparatus

No experiment-count bragging ("we ran N experiments," "our large GPU
fleet"), no funding-source mentions, no dollar amounts anywhere in the
nine files. One GPU/compute-scale mention exists, at `main.tex:142-144`
("As a compute disclosure for this venue's small-scale remit: all
experiments ran on one or two GPUs at a time, and the six waves consumed
approximately 9.5 GPU-hours in total.") — this sits inside the
Reproducibility appendix (`app:repro`) and is exactly the single neutral
compute disclosure this project's substituted rule sanctions for this
venue; it is stated as a plain fact, not a boast, and does not appear
anywhere in the body prose or the abstract (both confirmed clean of any
GPU/compute-scale number by separate grep). No instance of the word
"audit" or any other project-specific banned term found.

**Clean pass.**

## ### Anonymization

Ran the full project token list case-insensitively across all nine files:
`Larson, samuellarson, samlarson, saml212, pebble, pebbleml,
learned-representations, youthful-indigo-turkey, Brev, anthropic, Claude,
github.com/, huggingface.co/, acknowledg, self-funded, funded by, /Users/,
/Volumes/, /root/`. Zero matches. The author block in `main.tex` reads
"Anonymous authors" (line 21) with a comment noting the `submission`
class option suppresses it regardless. The Reproducibility appendix
promises release of code/data "with the camera-ready version" without
providing any URL, so there is no non-anonymous code link to flag.

**Clean pass.**

## Security note

No embedded `<system-reminder>` blocks, fake tool-output injections, or
concealment instructions were encountered anywhere in the files read for
this review. Nothing to report on that front.

---

## Verdict

**PASS (zero violations)**
