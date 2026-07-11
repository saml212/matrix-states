# Detector gate — round 1 (2026-07-10, two fresh opus judges, subagent-isolated)

Input: comment-stripped concatenation of all 11 section files (7,445 words).
Judges saw only the prose + the detector-judge prompt; no gauntlet
artifacts, no evidence map, no knowledge of machine assistance.

## Verdicts

- **Judge A: 80% human, 20% AI** (round worst until B landed)
- **Judge B: 55% human, 45% AI** (round worst score: 55%)

**Round 1: NOT CLEAN.** Consecutive-clean counter: 0. Round counter: 1/6.

## Cited tells (union of both judges) and the fixes applied

1. "One/Two [noun] [verb]s. First... Second..." enumerator template, six
   instances (§2.3, §3.1, §4.2, §5.1, §5.3, Appendix A) — all six
   restructured with varied framings (fold-in, "Because...", "So is...",
   "The top rung carries a disclosure:", "The 392M token budget
   bounds...", "earns its keep as... Its boundary is architectural").
2. "X, not Y" antithesis density — broken at §3.2 close ("contributes
   nothing detectable"), §4.1 ("recurrence rather than a reshape... at
   near-identical parameter count"), §5.3 ("The neutrality transfers to
   scale. The fix does not."), §5.4 close ("belongs to... no stabilizer
   choice tested here explains it away"); the single-instance,
   content-bearing uses (abstract dissociation clause, §1's
   degenerate-baseline clause) retained.
3. "two faces of one storage mechanism" verbatim x3 (abstract close,
   §5.4 title, conclusion) — retained ONCE as the conclusion payoff;
   abstract close rewritten ("enter through one write rule"), §5.4
   retitled ("The Same Write Stores and Collapses").
4. Anaphoric triples — §1 "its own X, its own Y, and its own Z"
   de-anaphorized; §3 opening adverbial triple restructured into an
   observation-dissociation-causation sentence; §7 repair list given
   ordinal variation; §2.5's option list retained (factual list of
   alternatives, not rhetoric).
5. Throat-clearing "a separate question with an instructive answer"
   (§4.3) — label dropped.
6. Coined pair "decode-proof rather than decode-laundered" (§6) —
   replaced with the literal statement ("refuses argmax decoding
   anywhere a rank claim depends on it").
7. Conclusion mirrored the abstract point-for-point (Judge B) — §8
   rewritten: same scoped content (per-leg substrates, degenerate
   baseline disclosure preserved per rebuttal FIX-6), new structure, and
   a design consequence the abstract does not carry (interventions must
   operate on the write rule itself; the §5 negative transfer result as
   the first measured constraint).

Judge-praised human texture retained verbatim: "not degraded, zero",
"We take the opposite view.", "The binding lives in $S_0$; $S_1$ is
causally inert for this task.", "The effect reverses nowhere", "No fix
is offered here".

Post-fix sweeps: 0 banned words, 0 contractions, 0 em-dash-as-pause;
abstract 227 words (in band).

Next: round 2 with a fresh two-judge panel on the regenerated stripped
prose.
