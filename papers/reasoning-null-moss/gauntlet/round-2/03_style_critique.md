# Style Critique — Gauntlet Round 2, Stage 03 (TARGETED)

**Paper:** `papers/reasoning-null-moss` — "Three Bounds on a Null: Testing the
Link Between Fast-Weight Write Geometry and In-Context Composition"
**Venue:** MOSS @ COLM 2026 (double-blind — anonymization check applied)
**Reviewer:** style-judge (fresh context, mechanical/exhaustive enforcement)
**Scope:** only the 2026-07-11 Bound-1 claim-shape revision —
`sections/00_abstract.tex`, `sections/01_introduction.tex`,
`sections/02_setting.tex` (new correspondence-null sentence),
`sections/03_geometric_null.tex` (whole file), `sections/07_discussion.tex`,
and `main.tex` (rewritten Appendix A, Fig 2 caption, new Appendix C compute
sentence). Bounds 2/3 (sections 04, 05) are out of scope.

---

### Banned words

None. Whole-word, case-insensitive grep of every in-scope file for the full
verbatim list — honest, actually, really, just, clearly, obviously,
interestingly, nicely, remarkable, surprising, unfortunately, essentially,
wildly, literally, parsimonious, cleanest, sharpest — returned zero hits.

### "audit" (project DO-NOT — highest priority)

**None. HARD-PASS.** `grep -niE '\baudit' main.tex sections/*.tex` (all `.tex`,
not just the in-scope subset) returned **zero** matches — no "audit", "audits",
"auditor", or "auditing" anywhere. The reviser's claim to have removed every
occurrence is verified against the full source tree.

### First-person "I" / "my" / "me"

None. Grep for `\b(I|my|me|myself|mine)\b` across all in-scope files returned
zero. Editorial "we" is used throughout ("We test whether…", "we find a null").

### Narrative-process / self-grading phrasing

None (hard). Grep for the flagged patterns ("our original hypothesis", "we
report a negative result…", "the paper's sharpest claim", "we set out", "we
hypothesized", "marquee", "headline"-as-self-grade) surfaced only two
descriptive uses, both permitted for a MOSS methods paper whose instrument-fix
account IS the contribution:
- `03_geometric_null.tex:60` "The strongest-case test" — names the
  experimental design (the hardest bar for the readout to clear), not a grade
  of the paper's own result. Passed in round 1.
- `07_discussion.tex:23` "keeps zeros and transients from becoming headlines" —
  describes the methodological value, not the authors' journey. The near-
  identical phrase passed round 1 as an optional metaphor.
The Appendix A run-in headings ("The positive control … failed, root-caused,
fixed, independently reviewed.", "The readout was defective, and the fix did
not save it.") describe what the positive control found — legitimate content
under the stage brief — and contain no first-person process narration.

### Contractions

None. The three apparent hits (`main.tex:100` function's, `main.tex:104`
session's, `main.tex:295` dissociation's) are possessives ending in "n"
(false positives). Every apostrophe in the in-scope prose is a possessive
(readout's, cohort's, null's, kernel's, corpus's, etc.), which is permitted.

### Em-dash / dash-as-pause

**THREE violations.** All are spaced ` -- ` single dashes used as a
conversational pause, in the rewritten Appendix A. (Paired ` -- … -- ` dashes
bracketing an appositive that itself contains commas — e.g. `main.tex:117/119`,
`main.tex:181`, `main.tex:335`, and `03_geometric_null.tex:43/45,54/56,69/70` —
are tolerated per the stage note and are NOT flagged.)

1. **`main.tex:112`** — "…recovers 0.0000 `--` the exact role-swap the
   closed-form adjudication predicts." Single trailing dash introducing a
   dramatic restatement.
   *Fix:* replace with a colon — "recovers 0.0000: the exact role-swap…".
2. **`main.tex:173`** — "…not backed by a separately archived raw artifact}
   `--` no raw state tensors were saved by the original harvest…" Single dash
   joining two independent clauses.
   *Fix:* end the sentence and start a new one (period), or use a semicolon.
3. **`main.tex:186`** — "…0 of 72 for the global arm) `--` concentration of
   the artifact, not of composition, per the same two nulls." Single trailing
   dash introducing a summary appositive.
   *Fix:* replace with a colon or a period ("…for the global arm). This is
   concentration of the artifact, not of composition…").

### Rhetorical-question headings

None. Grep for `?` across all in-scope files returned zero. New Appendix A
headings ("Registered Gates and Routing Rules", "The re-metric (78/320
nonzero) and why it does not reopen the lane", "The corrected claim, binding
throughout this paper.") are noun phrases/declaratives, not questions.

### Captions (incl. changed Fig 2)

None. The changed Fig 2 caption (`fig:dissoc`, `main.tex:284–296`) is
self-contained: it names the subtask (vocabulary/geometry dissociation under
task familiarization, wave 4), both panels with their metrics and ranges, the
takeaway ("learns the task through one readout while the other stays at
floor"), and discloses the pre-fix caveat. It does not defer to
"pending"/"TODO"/"will be added"; the Section/Appendix pointers are cross-
references, not deferrals of the caption's meaning.

### Abstract length

**227 words** (rendered words of `sections/00_abstract.tex`, stripping `%`
comments and `\begin/\end{abstract}`). Within the 200–230 band. No action.

### DO-NOT / apparatus / GPU-hours-in-body

None. No GPU-hours, cost, funding, or dollar language in any body section.
The only compute disclosures are both in the **Reproducibility appendix**
(Appendix C): `main.tex:333` "approximately 9.5 GPU-hours" (accepted in round
1) and the new `main.tex:337` "a further $\approx$0.52 GPU-hours on one GPU" —
permitted in the reproducibility appendix per the stage brief; neither appears
in the abstract, introduction, setting, geometric-null, or discussion body.
Reported counts (256 recovered, 320 readings, 80 cells, six waves) are the
substance of the null / instrument-fix, not apparatus boasts.

### Anonymization (double-blind)

None. Case-insensitive grep of all in-scope `.tex` for the full token list
(Larson, samuellarson, samlarson, saml212, pebble, pebbleml,
learned-representations, youthful-indigo-turkey, Brev, anthropic, Claude,
`github.com/`, `huggingface.co/`, acknowledg, self-funded, funded by,
`/Users/`, `/Volumes/`, `/root/`) returned zero matches. The dated tag
"(2026-07-11)" in the Appendix A heading is a date, not an identity leak. Code
release is deferred to camera-ready with no de-anonymizing URL.

---

## Optional suggestions (non-blocking, do not count toward the verdict)

- `00_abstract.tex:29` "closed escape routes" and `07_discussion.tex:23`
  "becoming headlines" are lightly metaphorical (not banned, not narrative-
  process). A maximally plain register could render them literally. Optional.

---

**Verdict: FAIL (3 violations)** — all three are spaced ` -- ` single-dash
conversational pauses in the rewritten Appendix A (`main.tex:112`, `:173`,
`:186`); each is fixed by a colon, semicolon, or period. Every higher-priority
check passes: zero "audit" anywhere (verified across the full `.tex` tree),
zero banned words, zero first-person/contractions/rhetorical-question
headings, self-contained Fig 2 caption, no body-prose GPU-hours/apparatus/
funding, zero anonymization leaks, and a 227-word abstract inside the band.
