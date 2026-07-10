# Rebuttal report — NeurReps EA, gauntlet round 2 (SCOPED)

**Isolation mode: `single-process`.** The dispatched round-2 rebuttal
subagent stalled without producing an artifact; per the paper skill's
degraded single-process fallback (SKILL.md § "How roles are run"), this
adjudication was performed as a fresh reasoning pass by the coordinator
with a hard context reset (state: the round-2 attack report B1–B6, the
round-2 defense report, the current draft; no memory of writing the
draft carried into the adjudication). Single-process isolation is weaker
than a fresh subagent; read accordingly. Facts were re-verified against
the raws where numeric.

## 0. Is any CRITICAL open?

**No CRITICAL was raised in round 2 and none remains open.** Round 1's
CRITICAL (A1, necessity tautology) was re-attacked by a fresh round-2
reviewer and found resolved as written. The six round-2 findings are 4
SERIOUS / 2 MINOR wording-and-consistency residues.

## Final verdicts and fix list (ordered)

| Attack | Severity | Defense disposition | Final verdict | Fix |
|---|---|---|---|---|
| B1 5–11% range wrong (true 4.9–10.2%) | SERIOUS | CONCEDE+FIX | DEFENSE INSUFFICIENT → resolved by FIX-1 | FIX-1 |
| B2 vacuous conditional in §5 ¶1 | SERIOUS | CONCEDE+FIX | DEFENSE VALID BUT EDIT | FIX-2 |
| B3 gate1a S3 row "by margin" false uniformity | SERIOUS | CONCEDE+FIX | DEFENSE VALID BUT EDIT | FIX-3 |
| B4 Delétang mischaracterized | SERIOUS | CONCEDE+FIX | DEFENSE VALID BUT EDIT | FIX-4 |
| B5 brief.md N14 arithmetic (0.15 not 0.10) | MINOR | CONCEDE+FIX | DEFENSE VALID BUT EDIT | FIX-5 |
| B6 "(bold)" caption ambiguity | MINOR | PARTIAL | PARTIAL — resolved by caption tightening | FIX-6 |

### FIX-1 (SERIOUS) `sections/04_ranklaw_observed.tex`
Before: `restricted effective rank lands within 5--11\% of $\dmin$ at every group,`
After: `restricted effective rank lands within 4.9--10.2\% of $\dmin$ at every group,`
Why: B1; recomputed per-group deviations are 6.1/4.9/5.6/10.2/5.3%. Word-neutral.

### FIX-2 (SERIOUS) `sections/05_causal_razor.tex` ¶1
Before: `Pre-registered reading: if $\dmin$ is load-bearing, $k = \dmin{-}1$ must fail and $k \geq \dmin$ must recover past $0.9\times$ the anchor's crosscheck $\recninety$.`
After: `Pre-registered reading: below $\dmin$, a sound readout pins recovery to zero by geometry, making that arm an integrity control whose registered trigger (any recovery there) would indicate an instrument leak; the live prediction is that $k \geq \dmin$ recovers past $0.9\times$ the anchor's crosscheck $\recninety$.`
Why: B2; matches the design record's leak-first HARD-FALSIFY registration and the sibling paper's accepted resolution. Costs +17 words, funded by the named cuts in FIX-2a/2b below.

### FIX-2a (compensating cut) `sections/05_causal_razor.tex`
Before: `Both marquee groups confirm at $\dmin$; the causal criterion is met on the sufficiency leg.`
After: `Both marquee groups confirm at $\dmin$, meeting the causal criterion.`

### FIX-2b (compensating cut) `sections/05_causal_razor.tex`
Before: `observed cells still reach 76--95\% of that ceiling (mean 88\%, per-group breakdown in Appendix~\ref{app:m1}), evidence of near-optimal training under the cap rather than collapse.`
After: `observed cells still reach 76--95\% of that ceiling (mean 88\%; Appendix~\ref{app:m1}), evidence of near-optimal training under the cap.`

### FIX-3 (SERIOUS) `sections/07_limitations.tex` Appendix D table
Before: `fail (all 4, by margin)` After: `fail (all 4)`
Why: B3; the necessity cell misses by 0.255 vs 0.006–0.020 for the other three; the plain form matches the S5 row. The appendix's surrounding text already separates the geometric-floor cell.

### FIX-4 (SERIOUS) `sections/06_related.tex`
Before: `\citet{grazzi2025negative}, \citet{siems2025deltaproduct}, \citet{merrill2024illusion}, and \citet{deletang2023chomsky} characterize which word problems recurrent architectures can express; the marquee equivalence instead sorts by representation dimension. A bolt-on-latent negative result \citep{larson2026gradient} is this result's counterpart.`
After: `\citet{grazzi2025negative}, \citet{siems2025deltaproduct}, and \citet{merrill2024illusion} characterize which word problems recurrent architectures can express, and \citet{deletang2023chomsky} benchmark sequence architectures across the formal-language hierarchy; the marquee equivalence instead sorts by representation dimension. A bolt-on-latent negative result \citep{larson2026gradient} is the counterpart.`
Why: B4; Delétang et al. spans architectures and 15 tasks, not recurrent word problems.

### FIX-5 (MINOR) `brief.md` row N14
Before: `exceed their own unconstrained anchor by 0.10 ($S_4$: 0.65$\to$0.80) and 0.10 ($S_5$: 0.50$\to$0.60)`
After: `exceed their own unconstrained anchor by 0.15 ($S_4$: 0.65$\to$0.80) and 0.10 ($S_5$: 0.50$\to$0.60)`
Why: B5; 0.80−0.65 = 0.15. Brief-only, no page cost.

### FIX-6 (MINOR) `sections/05_causal_razor.tex` Figure 1 caption
Before: `$0.9\times$anchor bar at $\dmin$ in all five groups (bold).`
After: `$0.9\times$anchor bar at $\dmin$ in all five groups (bold; $S_3$ via seed-mean, asterisked).`
Why: B6; four of five k=d_min cells are bolded outright, S_3 confirms at seed-mean.

## Residual risk
Workshop-survivable only: n=1 causal cells (disclosed), the S3/S5
soft-convergence profile (disclosed in Appendix D), and the
now-integrity-control framing of the below arm (accurately registered).
None conference-blocking for a 4pp EA.

## Re-run instruction
No CRITICAL is open; per the gauntlet termination rule the SERIOUS/MINOR
fixes above do not force a third attack round. The affected sentences
re-enter review via the final style pass (stage 03) and format audit
(stage 05) on the revised text.
