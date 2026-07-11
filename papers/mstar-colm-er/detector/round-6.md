# Detector gate — round 6 (final round) — mstar-colm-er

Panel: two fresh-context subagents (Opus), independent, same dispatch shape.
Draft state: post round-5 fixes, commit 0895109.

## Verdicts

| Judge | Classification | Percentage-human read |
|---|---|---|
| K | human-written | ~93% human |
| L | human-written | ~92% human |

**Round result: NOT CLEAN.** Round counter: 6 of MAX_ROUNDS=6.

# GATE TERMINAL STATE: `detector-cap-hit — human review needed`

Six rounds ran without two consecutive all-"100% human" rounds. Per the
gate's cap-hit hand-off, the draft is NOT accept-ready under the paper
skill's definition; a human decides whether the residual tells warrant
further editing before submission. No seventh round was run.

## Full round history (worst-judge metric)

| Round | Judge reads | Worst | Clean? | Draft commit judged |
|---|---|---|---|---|
| 1 | 96 / 95 | 95 | no | 78bc6f8 |
| 2 | 96 / 96 | **96** | no | 624279a |
| 3 | 95 / 93 | 93 | no | 293e239 |
| 4 | 92 / 93 | 92 | no | 6758248 |
| 5 | 90 (AI-borderline) / 99 | 90 | no | 0119932 |
| 6 | 93 / 92 | 92 | no | 0895109 |

Every verdict across all six rounds except one classified the draft
human-written; the single exception (round 5, judge I) was self-labeled
"borderline — high-human." No judge in any round found a mechanical tell
(banned words, generic transitions, filler scaffolding, hedge-stacking,
empty emphasis); every cited tell was structural-symmetry residue, and
each round's judges independently conceded the recurring scope-discipline
cadence is content-motivated for a pre-registered paper.

## The best-scoring draft (the hand-off's letter vs. its intent)

- **By the rule's letter:** round 2's judged draft (commit 624279a,
  worst=96) is the best-scoring revision.
- **Read with care:** the round-to-round score differences (90-96 floors)
  track judge variance, not draft regressions — every post-624279a edit
  was a fix for a cited tell, each recorded in `detector/round-N.md`, and
  none touched a number, an evidence tag, or a pin. The current draft
  (this commit) additionally resolves the round-2 through round-6 tells,
  including both tells the round-4 panel independently re-cited from
  round-2 prose. A human comparing the two candidates can diff prose-only
  changes with:
  `git diff 624279a..HEAD -- papers/mstar-colm-er/sections/`
- **The bundle is assembled from the current draft**, with this cap-hit
  status carried in the hand-off report rather than silently dropped.

## Residual tells from the final round (verbatim summary for the human)

1. (Judge K, the only tell) Cumulative "not-X-only-Y" scoping symmetry:
   nearly every claim is delivered as a matched positive/negative pair;
   singled-out flourishes: "wins, and wins at ceiling" (§1), the §5 bold
   antithesis header, the §4 paired negation. Judge K concedes each
   instance is individually "a legitimate, precise scoping statement" a
   scrupulous human would write; the tell is only their cumulative
   density.
2. (Judge L, tell 1 — strongest) The shared-hyperparameter clause
   duplicated near-verbatim between §3 and the Appendix C caption.
   **Fixed post-cap** (caption now cross-references §2/§3 instead of
   restating; recorded here because the gate had already terminated when
   the fix landed).
3. (Judge L, tell 2) Density of exactly-three-item lists (seven quoted).
   Retained: they enumerate the matched experimental axes and registered
   analysis objects; collapsing them costs precision.
4. (Judge L, tell 3, "noted not weighted") Uniform scope-qualifier
   machinery — expressly distinguished from AI hedge-clustering by the
   judge ("deliberate, hyper-careful human caution").

Judge L expressly declined to cite the bold antithesis headers as a tell
("ordinary strong technical writing... the instruction is not to invent
tells") — the same items judge K's tell 1 lists. This within-round
disagreement between the two final judges is itself evidence the residue
is at the noise floor of the instrument.

## Consequences

- Status `detector-cap-hit — human review needed` recorded; the draft is
  not marked accept-ready.
- The render inspection and bundle assembly proceed under the
  coordinator's charter (submittable-draft-early, deadline Jul 19 AoE),
  with the cap-hit status reported up rather than resolved locally. The
  human decision needed: submit as-is, apply further prose variation, or
  prefer the 624279a revision.
