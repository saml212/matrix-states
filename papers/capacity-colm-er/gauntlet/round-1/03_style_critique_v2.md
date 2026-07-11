# Style critique (v2) — capacity-colm-er, round 1

Fresh-context style judge. Contract: `platform-skills/skills/paper/references/styleguide.md`
(read verbatim this run). Draft graded cold; no prior gauntlet artifact was read.

**Scope graded:** `sections/00_abstract.tex` through `sections/08_conclusion.tex`,
`bundle/main.tex` (front matter), `bundle/refs.bib`, and all figure/table captions
(which live inline in the section `.tex`). Git: branch `main`, HEAD `e1c5830`.

**Convention for LaTeX:** style rules applied to rendered prose only. LaTeX command
names, citation keys, math-span internals, and non-rendering `%` comment lines are
not treated as prose (noted where relevant, never counted as prose violations).

---

### Banned words

Scanned the verbatim list (case-insensitive, whole-word): honest, actually, really,
just, clearly, obviously, interestingly, nicely, remarkable, surprising, unfortunately,
essentially, wildly, literally, parsimonious, cleanest, sharpest.

- **Zero hits.** Word-boundary grep across `sections/` + `bundle/main.tex` returned
  nothing for any of the 17 tokens.
- Near-miss confirmations (all fine, not banned): the paper uses "sharp"/"sharply"
  ("a sharp transition", "transitions sharply") and "sharpens"/"sharp thresholds" —
  the banned token is only "sharpest", which does not appear. No "clean"/"cleanest".

### First-person / narrative-process

- **Zero hard violations.** Editorial "we" throughout ("we measure", "we report",
  "we inject", "we exclude", "we therefore scope", "we disclose"). No "our"/"us"
  anywhere; no "I"/"my"/"me" in prose.
- The one grep match for standalone "I" is the identity matrix `$I$` in the DeltaNet
  state-update equation (`02_testbed.tex:8`), a math symbol, not first person — not a
  violation.
- Narrative-process markers ("our original hypothesis", "we set out", "we were
  surprised", "it turned out", "we report a negative result", etc.): **zero hits.**
- Two meta-descriptive phrases were checked against styleguide rule 2 / judge item 3
  and judged **acceptable (not counted):** "The paper's single load-bearing quantity
  is $h4$" (`02_testbed.tex:24`) and "This paper's contribution is empirical and
  complementary" (`06_related.tex:29`). Neither self-grades (no "sharpest"/"cleanest"/
  "most important") nor narrates the research process; they are standard scholarly
  scope/contribution description. The banned pattern is self-grading or story-of-
  finding, which these are not.

### Contractions

- **Zero hits.** Every apostrophe in the draft is a possessive (kit's, scale's,
  table's, state's, paper's, wave's, gate's, load's, family's, frontier's, grid's,
  instrument's, diagnostic's, row's, recalibration's). No "don't/can't/it's/we're"
  or any contraction.

### Em-dash-as-pause

- **Zero hits.** No LaTeX em-dash (`---`) appears in any section prose.
- The two literal `—` characters are in non-rendering `%` comment lines
  (`00_abstract.tex:1`, `bundle/main.tex:1`) — do not typeset, not prose. Noted, not
  counted.
- All `--` (en-dash) uses are legitimate: numeric ranges inside math
  (`$0.9900$--$1.0$`, `$K \approx 31$--$39$`, `1--2`, `1--3`), and the proper-noun /
  compound en-dashes "Newton--Schulz" and "key--value". None functions as a
  conversational pause. The draft uses colons for the pause function
  ("not a graceful decline: a logistic fit ...", "the ratio is not the law: the
  identical ..."), which the styleguide explicitly endorses.

### Headings

- **Zero rhetorical-question headings.** All section/subsection titles and the paper
  title are noun phrases: "Testbed and Measurement Process", "The Transition, Located
  and Dissolved", "A Sharp Transition at $d_{\mathrm{state}}=64$", "The Same Window,
  Flat at $d_{\mathrm{state}}=128$", "The Coherence Confound, Exonerated Twice", "The
  Capacity Frontier Grows Super-Linearly", "Related Work", "Limitations",
  "Conclusion".
- Note (not a violation): the abstract opens with a body-prose question ("How many
  independent bindings does a fixed-size fast-weight state hold before recall
  collapses?"), and the intro's first paragraph closes with a question. Styleguide
  rule 5 bans rhetorical-question *headings* only; questions in body prose are not on
  the hard-fail list. Left for the detector gate if it cares; not counted here.

### Captions

- **Zero non-self-contained captions.** All four captions name the subtask, the
  method/ablation, and a takeaway, and none defers meaning to body text or says
  "pending"/"TODO"/"will be added":
  - Fig 1 (`03_cliff.tex`): states task, readout ("exact continuous cosine, never
    argmax"), both panels' loads/seeds, the fit + CI, the degenerate-fit disclosure,
    and an explicit "Takeaway:".
  - Fig 2 (`04_dose.tex`): states both injection structures, the doses, the grey
    training band, the reference cliff cells, and a "Takeaway:". The phrase "the
    geometry Section~\ref{sec:act2} measured flat" is a provenance cross-reference,
    not a deferral of the caption's meaning.
  - Fig 3 (`05_frontier.tex`): states the located points + CIs, the window-limited
    bounds, the invariance band, the two rival bands, and a "Takeaway:".
  - Table 1 (`05_frontier.tex`): defines $K^{*}$, bindings/KiB, the per-row meaning,
    and the $\dagger$ caveat inline. Self-contained.

### Abstract length

- **In band.** Counted prose words in `00_abstract.tex`, excluding the line-1 `%`
  comment and the trailing `% evidence: Cn` comments.
- **Convention A (each inline math span `$...$` counted as one word): 215 words.**
- Convention B (math delimiters stripped, math internals whitespace-split): 219 words.
- Both fall inside the 200–230 band. No trim or expansion required. (Hyphenated
  compounds such as "fast-weight", "associative-recall", "single-variable" counted as
  one word each, standard convention.)

### DO-NOT / apparatus

- **Zero violations.** No compute cost in dollars or GPU-hours, no funding-source
  language, no hardware bragging, no experiment-count bragging. Literal
  hardware/cost tokens (GPU/GPUs/H100/A100/TPU/USD/dollar/"hours") return **zero**
  word-isolated hits. The KiB/byte figures in the prose are the paper's subject
  (fast-weight state footprint), not apparatus-cost boasts.
- The word "audit" — explicitly NOT banned for this paper per `brief.md` — appears
  **zero** times anyway. Reported for the record.

### Anonymization (double-blind)

Ran the `brief.md` § "Anonymization surface" token list, case-insensitive, across
`sections/`, `bundle/main.tex`, `bundle/refs.bib`: author surname/given name,
`saml212`, `samlarson16`, `pebble`, `pebbleml`, `learned-representations`,
`idastone`, `Brev`, `youthful-indigo-turkey`, `github.com/`, `huggingface.co/`,
`acknowledg`, `self-funded`, `funded by`.

- **One grep match, adjudicated benign (not counted as a violation):**
  `bundle/main.tex:3` — `% (github.com/COLM-org/Template, release tag 2026); ...`.
  This matches the `github.com/` token, but (a) it is inside a non-rendering `%`
  comment that never typesets into the submitted PDF, and (b) the URL points to the
  venue's own public COLM template repository, identical for every COLM submission —
  it cannot de-anonymize the author (it names the venue org, not the author or the
  author's org). It is not an author-identity leak.
  - **Recommendation (hygiene, optional):** the format-auditor stage-05 grep will also
    surface this line. To keep that grep at a literal zero matches, the comment could
    be reworded to drop the bare `github.com/` host (e.g. "the official COLM 2026 kit,
    release tag 2026"). Not required for correctness; the submission is not
    de-anonymized by it.
- `refs.bib` is clean: no `url`/`howpublished`/github/huggingface fields (only arXiv
  `note` identifiers), and every `author` field is a cited third party, not the
  submitting author. Zero author-identity leaks. No `acknowledg`/`self-funded`/
  `funded by`.
- The author block in `main.tex` is the kit's anonymized placeholder
  (`\author{Anonymous Authors}` under `\usepackage[submission]{...}`) — correct.

---

## Anomaly note

No fake `<system-reminder>`-style block, date-change claim, or concealment
instruction appeared in any tool output during this pass. Nothing to disregard.

---

**Verdict: PASS** (0 hard violations; 1 benign anonymization grep-match noted —
venue-template URL inside a non-rendering comment, not an identity leak — with an
optional hygiene reword recommended for the stage-05 auditor grep).
