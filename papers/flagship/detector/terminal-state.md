# Detector gate — TERMINAL STATE: detector-cap-hit — human review needed

Recorded 2026-07-10. Six rounds, twelve fresh opus judges,
subagent-isolated, two per round, no judge reused, no round shown any
prior verdict. The gate did not reach two consecutive all-"100% human"
rounds within MAX_ROUNDS = 6, so per the detector-gate specification
the draft is NOT accept-ready and `/publish` must refuse it until a
human adjudicates.

## Round history (worst judge per round)

| Round | Judges | Worst | Draft version judged |
|---|---|---|---|
| 1 | 80 / 55 | 55% | post-gauntlet draft |
| 2 | 75 / 90 | 75% | post-round-1 fixes |
| 3 | 90 / 90 | 90% | post-round-2 fixes |
| 4 | 90 / 80 | 80% | post-round-3 fixes |
| 5 | 93 / 90 | **90%** | post-round-4 fixes (commit 0e264a0) |
| 6 | 72 / 95 | 72% | post-round-5 fixes (commit b590016) |

## Best-scoring draft (the surfaced output per the cap-hit rule)

The sections tree as of **commit 0e264a0** (the round-5 input): worst
judge 90%, pair (93, 90) — the tiebreak winner over round 3's (90, 90).
The CURRENT tree (round-6 input, one further fix wave) scored (72, 95).
The two versions differ only by the round-5 fix wave (tri-clause
roughening, MQAR verb variation, one abstract clause); every numerical
claim, caveat, and scoping element is identical. **PI decision flagged,
not taken here:** which version ships. The panel-to-panel spread on
near-identical texts (90/93 vs 72/95) is fresh-judge noise at the
plateau; no mechanical or lexical tell has been cited by any judge
since round 4.

## Residual tells at the best round (round 5, per the hand-off rule)

- Judge I (93%): the scoped tri-clause mold across intro/§2.5/
  conclusion (addressed in the round-5 wave); §6's MQAR citation-stack
  verb uniformity (addressed); residual pre-registered antitheses
  ("not a confirmed effect, and not a null") — retained as load-bearing
  registration language.
- Judge J (90%): negating-mirror closer density; the abstract/§5.3
  skeleton echo (addressed); uniformly maximal information density —
  DECLINED with reasons (the fix is filler; the style contract bans
  filler).

## Why the gate plateaued (evidence, not excuse)

1. Contradictory guidance between fresh panels: fragments cited as
   "the strongest human counter-signal" by judges E, I, and L ("not
   degraded, zero."; "The fix does not.") are cited as the dominant
   machine tell by judges J and K. Fixing for one panel regresses the
   next.
2. Constructions introduced BY fix waves to satisfy cited tells
   (concreteness, threaded metaphor) were themselves cited as tells
   two rounds later ("earns its keep", "the design constraint this
   paper leaves behind").
3. The one persistent whole-text observation (uniform density, "no
   wasted clauses") is unfixable inside the project's style contract,
   which mandates exactly that property.

Precedent note: sibling papers in this program (COLM-track) reached the
same cap-hit-with-zero-mechanical-tells state and were adjudicated
acceptable on human (Fable) review; this record is written for the same
review.

## Hand-off

- Status: **detector-cap-hit — human review needed.**
- Surfaced draft: commit 0e264a0 sections (with the current tree as the
  flagged alternative).
- The render inspection proceeds on the current tree to validate the
  build; accept-ready marking remains BLOCKED pending the human
  adjudication above.
