# Style Critique — Round 2 (Stage 03, post-rebuttal-fix prose)

Scope: `main.tex` + `sections/{01_intro,02_setup,03_convergence,04_equivalence,05_causal,06_related_limits,07_appendix}.tex`. Enforced against `references/styleguide.md` verbatim (banned-word list re-read from file this run, not from memory). `%`-comments in the `.tex` sources are build machinery and were excluded from the prose check per instructions; only rendered text was judged.

### Banned words

Checked the full verbatim list (`honest, actually, really, just, clearly, obviously, interestingly, nicely, remarkable, surprising, unfortunately, essentially, wildly, literally, parsimonious, cleanest, sharpest`), case-insensitive whole-word, across all seven files.

Zero hits.

### First-person / narrative-process

Two hits — both self-referential "the paper"/"this paper" constructions matching the styleguide's own banned example ("the paper's sharpest claim") almost verbatim.

1. `sections/07_appendix.tex:98-100` (subsection "The rank-constrained cosine ceiling"):
   > "At $k \geq \dmin$ the ceiling is 1.0 (exact recovery is representable), so recovery above 0 at $\dmin$ is not guaranteed by the metric and **is the paper's genuine causal finding** (sufficiency)."

   `is the paper's genuine causal finding` is structurally identical to the styleguide's own flagged example, `the paper's sharpest claim` — self-reference to the manuscript as the bearer of the claim, instead of stating the finding directly. Suggested rewrite: *"...is not guaranteed by the metric and constitutes the causal finding on sufficiency."* or *"...is not guaranteed by the metric; recovery above 0 at $\dmin$ is the causal evidence for sufficiency."*

2. `sections/07_appendix.tex:111-113` (subsection "Primary and crosscheck degauging"):
   > "...pins the fitted-$(c, Q)$ score, denoted $\recninety$ throughout Section~\ref{sec:causal} and Figure~\ref{fig:razor}, as decisional; **no conclusion in this paper reads** the scale-only score."

   `in this paper` narrates the manuscript as an actor ("this paper['s conclusions] read..."), the same pattern as the styleguide's banned "this section runs that control and reports falsification of our own hypothesis." Suggested rewrite: *"...as decisional; no reported conclusion reads the scale-only score."* or *"...as decisional; the scale-only score informs no reported conclusion."*

Not flagged (for contrast): `sections/06_related_limits.tex:7,17` use "this work measures..." / "this work asks..." to contrast with cited prior work inside the Related Work section. That is the standard, accepted convention for delineating a paper's contribution from cited baselines in a related-work paragraph, not narration of the discovery process — left as non-violating.

### Contractions

Checked `don't, can't, it's, we're, didn't, doesn't, isn't, won't, wasn't, weren't, haven't, hasn't, hadn't, wouldn't, couldn't, shouldn't, I'm, I've, I'll, I'd, that's, there's, let's`.

Zero hits. All possessives in the prose (`product's`, `group's`, `metric's`, `seed's`, etc.) are grammatical possessives, not contractions.

### Em-dash-as-pause

Two raw em-dash (—) characters found, both in `main.tex` lines 1 and 9 — both inside `%`-comment build notes ("UniReps Extended Abstract track — official NeurIPS LaTeX template...", "AUTHOR BLOCK: PENDING-USER — the style file prints..."), which do not render and are excluded per instructions. Zero em-dashes in rendered prose.

### Headings

All section and bolded-paragraph headings are noun phrases (e.g. "Convergence to the Algebraic Minimum," "Equivalence at Matched Dimension," "The Causal Check," "Instrument Details and the Two Defects," "The rank-constrained cosine ceiling"). Zero rhetorical-question headings.

### Captions

All four captions (Figure 1 `fig:convergence`, Figure 2 `fig:razor`, Table `tab:groups`, Table `tab:s3seeds`, Table `tab:m1`) are self-contained: each names the subtask/quantity, the method, and the takeaway inline, with no "pending"/"TODO"/"will be added" deferrals. `tab:s3seeds`'s caption references Section~\ref{sec:causal} for narrative context but fully states its own content (seed-level values, the pre-registered bar, the recompute-check) standalone — not a deferral of undefined content, so not flagged.

Note: the three `PENDING-USER` strings (`main.tex` lines 7, 9, 45) are inside `%`-comments (build/production notes to the author, not a caption or body text) and do not render — excluded per instructions, not a caption violation.

### Abstract length

224 words (counted by whitespace-delimited tokens between `\begin{abstract}` and `\end{abstract}`, which also splits at spaces inside inline math, e.g. `$\dmin = 2/3/3/4/5$` counts as three tokens as it renders). Within the 200-230 band. No trim/expand needed.

### DO-NOT / apparatus

Checked for the project DO-NOTs: `audit` (as a prose word), GPU-hours/dollar/fleet/experiment-count bragging, funding language. Zero hits. Numeric apparatus mentions present (seed counts, cell counts, step budgets "8k--40k") are all reporting sample sizes / hyperparameters in service of a statistical claim (e.g. "all 19 seeds inside band," "$n=5$ seeds per group"), not compute-scale bragging — not flagged.

### Anonymization

Ran the full token list (`larson, samlarson, saml212, pebble, pebbleml, rockie, github.com/, huggingface.co/, .pebbleml.com, acknowledg, self-funded, funded by`), case-insensitive, across all seven files. Zero matches. No author names, handles, orgs, de-anonymizing URLs, or acknowledgment language present. The author block correctly prints "Anonymous Author(s) / Affiliation / email@example.com" (template placeholder, not a leak).

---

## Verdict

**FAIL (2 violations)**
