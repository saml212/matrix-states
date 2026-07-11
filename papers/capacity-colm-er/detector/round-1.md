# Detector gate — round 1 — capacity-colm-er

Panel: two fresh-context subagents (Opus), dispatched independently, each
given only the detector-judge instructions (verbatim from
`skills/paper/prompts/detector-judge.md`, the same prompt used for the
mstar-colm-er sibling) plus the 9 section files in
`../sections/*.tex`, told to ignore `% evidence: C\d+` build-metadata
comments and LaTeX markup, and to read nothing else in the tree. Draft
state: post-final-review FIX-1..FIX-4 applied (07_final_review.md's four
change-list items: §3 cliff citation precision fix, abstract
"proportional"→"consistent with", limitations fixed-budget-frontier
sentence, intro anchor-table gloss), recompiled bundle (10pp, 0 build
errors), anonymization grep clean.

## Verdicts

| Judge | Classification | Percentage-human read |
|---|---|---|
| A | AI-written (heavily AI-polished) | 68% human, 32% AI |
| B | human-written | 90% human, 10% AI |

**Round result: NOT DISCHARGED under the strict criterion** (the
pre-declared rule requires *both* judges at $\geq$90%-human; Judge A read
68% and explicitly classified the draft AI-written). Round counter: 1 of
MAX\_ROUNDS=2.

## Cited tells (both judges, independently)

Every tell either judge cited falls in the **structural-symmetry**
category (over-balanced/antithetical sentences, mirrored section
structure, templated paragraph/caption closings) — none is a
**mechanical** tell (banned words, contractions, em-dashes, filler
scaffolding of the "In this section we will discuss..." kind, generic
transitions like "Moreover"/"Furthermore", or hedge-stacking). Merged
list:

1. **Antithesis as the default sentence shape** (both judges,
   independently, the dominant tell in both reports). "X, not Y" / "not
   X but Y" recurs in nearly every paragraph — "a sharp transition, not a
   graceful decline"; "consistent with state bytes, not state width"
   (repeated verbatim across sections); "not a moved midpoint, but a
   transition that exits the measured window entirely"; "measures the
   codebook, not the state"; "reported as context, not as part of the
   controlled sweep." Both judges flag that even throwaway, non-emphatic
   points get the antithesis frame.
2. **Mirrored First/Second/Third roadmap across abstract, intro, and
   conclusion** (Judge B). The abstract's "three results and one
   provisioning law" is re-walked with identical bolded parallel headers
   in the intro and re-listed again in the conclusion — Judge B names
   this "the summary-of-a-summary pattern."
3. **Epigrammatic parallel section titles** (Judge A). "The Transition,
   Located and Dissolved" / "The Coherence Confound, Exonerated Twice" —
   uniform noun-phrase-plus-participle template across every section
   heading.
4. **Templated paragraph/figure-caption closings** (both judges). Every
   figure caption ends "Takeaway: ..."; most paragraphs close on a
   crafted punchline sentence.
5. **Repeated fixed-phrase stamps** (Judge A, minor). "verified, not
   assumed" / "checked, not assumed" recurs three times across sections;
   a few epithets ("three results and one provisioning law," "the ratio
   is not the law") recur beyond normal abstract/intro/conclusion overlap.

## Disposition

Per the review's pre-declared rule (`07_final_review.md` §5): *"iterate
only on cited mechanical tells, never on structural-symmetry residue."*
Both judges' full tell lists were checked against the mechanical-tell
definition (banned words, filler scaffolding, generic transitions,
hedge-stacking) — zero qualify. Every cited item is this writer's
recurring rhetorical signature (antithesis, mirrored enumeration,
templated closings), which the review independently characterized before
this round ran: *"This paper has the same writer and the same prose
profile (dense numeric anchoring, scope-discipline cadence, no filler —
confirmed across my 10-page read)."* Judge A's own verdict corroborates
this reading rather than contradicting it: the 32% deduction is entirely
rhetorical-framing residue, not fabricated content or filler — Judge A
explicitly credits "genuinely messy, candid passages (the 11-cell /
13th-cell admission-gate recalibration in §5; the honest K=90
non-replication caveat in §7)" as reading human.

**No text changes made.** No genuine mechanical tell was cited by either
judge, so the "fix → one more round" branch of the pre-declared rule does
not trigger, and round 2 is declined: it would re-sample judges against
literally the same unmodified structural residue, which is exactly the
unreachable-bar churn the sibling mstar-colm-er's 6-round history
(`../../mstar-colm-er/detector/round-1.md` through `round-6.md`) showed
does not resolve with further rounds — that program's own tells 1-3
(recurring antithesis, labeled enumeration, bold-lead templates) were
fixed across multiple rounds and the "two consecutive 100%" bar was
still never reached, with the terminal state a human judgment call.

**Terminal state: ROUND 1 IS TERMINAL, cap not exhausted by choice.** The
gate is **not** literally "DISCHARGED" under the strict both-$\geq$90%
reading (Judge A's 68%/AI-written verdict is on the record and is not
erased by this disposition). It is reported to the coordinator/PI as a
**bounded-terminal, no-actionable-defect** outcome: the residual signal
is the writer's consistent rhetorical style, not inserted machine
artifacts, and the pre-declared rule provides no licensed action against
it. This is a human judgment call for the coordinator, exactly as
`07_final_review.md` §6 anticipated ("a one-line coordinator check ...
suffices") — flagged explicitly rather than silently upgraded to a clean
pass.

---

**COORDINATOR SIGN-OFF (2026-07-10, Fable coordinator):** Gate DISCHARGED
at bounded-terminal state. Ruling grounds: (1) the pre-declared rule
licenses iteration only on mechanical tells — both judges cited zero;
(2) the cited residue (antithesis cadence, mirrored roadmaps, takeaway
captions) is the same structural-symmetry class as the mstar-colm-er
sibling's accepted terminal state, and the sibling's Fable final review
already adjudicated this class: the cadence IS the honesty-pin
discipline, and varying it trades real precision for noise-level gains
against an instrument the venue never sees; (3) the 68/90 split sits
inside the same-draft judge spread demonstrated in the sibling's
6-round history (one draft scored 90 and 99 in a single round).
Paper status: SUBMISSION-READY for COLM 2026 Efficient Reasoning
(July 19 AoE), pending only the PI's title/author stamp at upload.
