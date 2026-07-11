# Style Critique — rank-recruitment-ws (gauntlet round 1, stage 4 / artifact 03)

Fresh-context copy-editor pass over the full draft (`main.tex` + all seven
section files + both appendix tables/captions + the rendered figure strings in
`figures/figure_gen.py`), enforcing `references/styleguide.md` verbatim plus the
paper's project-specific DO-NOT list. Rebuttal fixes re-checked from scratch; no
assumption of prior cleanliness.

Scope note: the style contract governs rendered prose — body, abstract, captions,
section titles, and (per the task) `set_title`/label strings baked into figures.
LaTeX comment lines (`%`) and Python docstrings/comments are out of that scope
except for identity leaks; they are checked separately below and reported as
advisories where relevant.

---

### Banned words
None. Full whole-word, case-insensitive sweep of the verbatim list (`honest,
actually, really, just, clearly, obviously, interestingly, nicely, remarkable,
surprising, unfortunately, essentially, wildly, literally, parsimonious,
cleanest, sharpest`) across `main.tex`, all sections, and `figure_gen.py`
returned zero hits in rendered prose. ("clean step" in `04_composition.tex:32`
is not "cleanest"; "sharper" in a figure title is addressed under Advisories —
it is not the banned token "sharpest".)

### First-person / narrative-process
None. Editorial "we" throughout ("We answer it", "We close the confound", "We
measure whether", "We therefore pin"). No standalone `I`/`my`/`me`. No
narrative-process slips: the intro "The contributions:" list is a noun-phrase
enumeration, not a research-journey narration; no "our original hypothesis",
"we report a negative result with a mechanism", or "the paper's sharpest claim".

### Contractions
None in rendered prose. The only apostrophe-contraction match is
`main.tex:36` — `%%%% DON'T CHANGE %%%%%%%%%` — a LaTeX comment, not rendered.
No action required for the style verdict. All other apostrophes are possessives
("state's", "operator's", "cycle's", "$K{=}16$'s", "paper's"), which are allowed.

### Em-dash-as-pause
None in rendered prose. Every `---`/unicode em-dash hit is a comment or code:
`main.tex:1`, `main.tex:7` (`%%%%` template comments) and `figure_gen.py`
docstring/comment lines (2, 12, 17, 18, 34, 38). No section file contains a
rendered em-dash. All `--` in the prose are numeric ranges (`2--2.5`,
`14.6--15.6`, `7.999--8.000`, `0.7--2.4\%`, `6--16K`, `91--97\%`, `0.92--0.93`),
not conversational pauses.

### Headings
All noun phrases; no rhetorical-question headings.
- "Introduction"
- "The Bound and the Teeth That Keep It a Bound"
- "Rank Tracks $K$ and Is Causally Necessary"
- "Exact Composition and the Invariant-Subspace Mechanism"
- "The Exactness Frontier, and When to Trust a Dead Cell"
- "Related Work, Limitations, and Outlook"
- Appendix: "Depth 21 Under the Single $K$-Cycle", "Per-Seed Subspace
  Decomposition", "Reproducibility"

### Captions
All self-contained; none defer to body text and none say "pending"/"TODO"/"will
be added". Each names its subtask, method/ablation, and takeaway:
- `fig:forcerank` — names the primary pre-registered causal test, the axis
  (train-time force-rank $k$), the marker ($k=K$ dashed line), and the takeaway
  (step at $k \approx K$).
- `fig:depth` — names the $(d{=}16, K{=}8)$ setting, both panels, and their
  contrasts.
- `tab:depthcurve` and `tab:subspace` — each states what is decomposed, the
  source (archived $Z$-dumps), and the reported quantities.

### Abstract length
216 words — inside the 200–230 band. PASS. (Counted from the rendered abstract
with the evidence-marker comment lines excluded and math clusters counted as a
reader would read them; the figure lands mid-band with comfortable margin either
way of small math-token counting choices.)

### DO-NOT / apparatus
No hard violation in rendered prose or rendered figure strings.
- No "audit" anywhere in the prose (project ban satisfied).
- No GPU-hour, dollar, or fleet-size mention anywhere.
- No experiment-count bragging in prose or captions. The paper's own prose uses
  qualitative descriptors ("a subsequent, larger replication sweep",
  `03_recruitment.tex:11`) rather than a bare run count. The rendered figure
  `set_title`/label strings contain no run counts.

### Anonymization (double-blind)
Clean — only the documented exception fires. Grep for `larson, samlarson,
saml212, pebble, pebbleml, rockie, github.com/, huggingface.co/, .pebbleml.com,
acknowledg, self-funded, funded by` returned hits ONLY on the sanctioned
`larson2026gradient` self-citation:
- `sections/01_intro.tex:13` `\citep{larson2026gradient}` — documented exception.
- `sections/03_recruitment.tex:34` `\citet{larson2026gradient}` — documented
  exception.
- `refs.bib:16,18` the `larson2026gradient` entry (`author = {Larson, Sam}`) —
  documented exception.
No other identity token, org name, de-anonymizing URL, or acknowledgment
language appears. The reproducibility appendix correctly uses the anonymous host
`https://anonymous.4open.science/` (`app:repro`), not a GitHub/HuggingFace URL.

---

## Advisories (not counted against the style verdict; flagged so a later stage does not lose a round)

These fall outside the style contract's rendered-prose scope but touch the
project DO-NOT list and the released submission bundle. They do NOT flip the
verdict; the format auditor (stage 05, which greps the whole bundle including
released code) should confirm the disposition.

1. **"991-run" in `figures/figure_gen.py`** — appears three times, all in
   non-rendered locations: the module docstring (`:17`), a source-map comment
   (`:44`), and an assert message + literal (`:107`,
   `assert agg.get("n_runs") == 991, "expected the 991-run pre-registered
   snapshot"`). None of these render into the paper prose, a caption, or a
   figure `set_title`/label, so this is not a style-contract violation. But
   `figure_gen.py` ships with the anonymized code at `anonymous.4open.science`,
   and the project brief names "991-run" as an experiment-count string to avoid.
   Recommend rephrasing the docstring/comment to a neutral description
   (for example "the pre-registered Task D snapshot") and keeping the numeric
   `991` only inside the assert as an integrity check, not narrated as a run
   count, before the bundle is released.

2. **"sharper" in the fig_depth right-panel title** (`figure_gen.py:188`,
   `ax2.set_title("recovered fraction is the sharper depth probe")`) — this is a
   rendered figure string the reader sees. "sharper" is NOT the banned token
   "sharpest" (whole-word list), so it is mechanically not a banned-word fail.
   It is, however, a comparative self-grading adjective in the same family the
   styleguide targets. Optional tightening: "recovered fraction resolves the
   depth decay more finely" or "recovered fraction is the more discriminative
   depth metric". Left as an advisory, not a violation.

---

## VERDICT

PASS
