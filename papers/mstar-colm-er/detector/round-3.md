# Detector gate — round 3 — mstar-colm-er

Panel: two fresh-context subagents (Opus), independent, same dispatch shape.
Draft state: post round-2 fixes, commit 293e239.

## Verdicts

| Judge | Classification | Percentage-human read |
|---|---|---|
| E | human-written | ~95% human |
| F | human-written | ~93% human |

**Round result: NOT CLEAN.** Consecutive-clean counter: 0. Round counter:
3 of MAX_ROUNDS=6. Both judges rate every cited tell weak/defensible and
state the draft reads human; neither would reject it as AI-generated.

## Cited tells (merged)

1. **"X, not Y" contrastive-tail density** (judge E's main tell; judge F's
   tell 2 is the same pattern): seven quoted instances of a claim carrying a
   reflexive negated tail across §§1-7.
2. **Registered phrase repeated verbatim in four sections** (judge F):
   "baseline non-competitive at matched params/tokens" in abstract
   (near-verbatim), intro, §4, and conclusion.
3. **Conclusion re-lists the abstract** (judge E): the middle two-thirds of
   the closing paragraph restates the abstract's numbers nearly verbatim.
4. **Clause-stacked long sentences in §2** (judge F): the "three axes"
   sentence chains colon, triple, semicolon pivot, subordinate clause, and
   two citations without relief.
5. Tidy triples (judge F, minor, conceded as legitimate enumerations) and
   bold-lead regularity (judge E, explicitly not counted) — no action.

## Fixes applied before round 4

- Tell 1, shape varied where the content is not a pre-registered pin:
  §7 "not a contradiction but a different operating point" became "The two
  findings are consistent; ours sits at a different operating point";
  §5's band sentence restated positively ("there is no sampling process for
  the band to characterize, the 21-query change is a real and repeatable
  effect"); §5's forward-computation sentence inverted out of the "X, not
  Y" frame; §5 closer trimmed to "probe legibility is a property of the
  instrument"; §6 closer trimmed to "remains an open training problem."
  Kept unchanged, deliberately: the task2 "trainability finding, not a
  capability separation" pair (pre-registered pin language), §2's
  "descriptive and diagnostic rather than additional hypothesis tests"
  (FIX-6 statistical scoping), and §4's "What is not claimed." heading.
- Tell 2: conclusion's fourth verbatim repeat replaced with a reference
  ("the registered outcome of Section 4 is the paper's only comparative
  claim"); the verbatim phrase now appears where it is introduced (intro)
  and where it is adjudicated (§4), plus the abstract's near-verbatim form.
- Tell 3: conclusion recap compressed; the re-listed numbers (eight times
  the span, 32,768 bytes, thirty-two times) dropped in favor of one
  qualitative sentence plus interpretation; the closing flourish kept.
- Tell 4: the §2 "three axes" chain split into three sentences.

Recompiled after fixes: 0 errors, 0 `??`, 10 pages (conclusion compression
recovered a page), banned-word and contraction greps clean. Round 4 (fresh
panel) dispatched on the revised draft.
