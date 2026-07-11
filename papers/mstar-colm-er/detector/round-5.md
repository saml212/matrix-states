# Detector gate — round 5 — mstar-colm-er

Panel: two fresh-context subagents (Opus), independent, same dispatch shape.
Draft state: post round-4 fixes, commit 0119932.

## Verdicts

| Judge | Classification | Percentage-human read |
|---|---|---|
| I | AI-written (borderline, "high-human") | ~90% human |
| J | human-written | ~99% human |

**Round result: NOT CLEAN.** Consecutive-clean counter: 0. Round counter:
5 of MAX_ROUNDS=6.

**Gate arithmetic note (recorded for the hand-off):** with round 5 not
clean, two *consecutive* clean rounds are no longer reachable inside the
MAX_ROUNDS=6 cap — the best possible remaining outcome is a single clean
round 6. The gate is therefore certain to terminate as
`detector-cap-hit — human review needed`. Round 6 still runs: it determines
the best-scoring draft of record (the gate keeps the revision whose worst
judge gave the highest read; the floor to beat is round 2's worst=96%).

## Cited tells (merged)

- Judge I, tell A: the S0/S1 zeroing result restated in the same symmetric
  antithesis mold four times (abstract, intro bullet, §5 bold lead,
  conclusion).
- Judge I, tell B: "X, not Y" contrastive scaffolding as the paper's only
  scoping cadence (seven instances listed, several of which are
  pre-registered pin language).
- Judge I, tell C: tidy triple/quadruple enumerations at high density
  (conceded "partly forced by the content" — they enumerate the matched
  axes).
- Judge I, tell D (mild): conclusion re-lists the headline results.
- Judge J (99%, one regularity): the bold run-in headline + per-section
  disclaimer cadence — expressly read as "this paper's genuine
  pre-registration discipline showing through... a regularity, not a
  smell."

## Fixes applied before round 6

- Tell A: abstract instance replaced with a structurally different,
  shorter statement ("State-zeroing localizes the recall to the first
  block's fast-weight state."); intro bullet's second clause broken out of
  the parallel mold ("the second block's state proves inert"); the §5 bold
  lead keeps the memorable antithesis once; conclusion's already-varied
  phrasing kept.
- Tell B, non-pin instances: abstract opener tightened ("increasingly
  memory-bound"); "three axes, not one" became "three axes at once";
  limitations "as bounds, not claims" became "only as bounds". Pin
  instances (task2 trainability framing, FIX-6's descriptive-vs-tests
  sentence, §4's What-is-not-claimed heading) retained by design, as in
  every prior round.
- Tell C: retained — the enumerations name the matched experimental axes
  and judge I concedes they are content-forced; collapsing them would cost
  precision.
- Tell D: retained (already compressed in round 3; judge I weights it
  lightly).

Recompiled after fixes: 0 errors, 0 `??`, 10 pages, abstract 215 words (in
band). Round 6 (fresh panel, final round under the cap) dispatched.
