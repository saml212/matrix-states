# Detector gate — round 2 — mstar-colm-er

Panel: two fresh-context subagents (Opus), independent, same dispatch shape as
round 1 (judge prompt + the 11 section files only). Draft state: post
round-1 tell-fixes, commit 624279a.

## Verdicts

| Judge | Classification | Percentage-human read |
|---|---|---|
| C | human-written | ~96% human |
| D | human-written | ~96% human |

**Round result: NOT CLEAN.** Consecutive-clean counter: 0. Round counter:
2 of MAX_ROUNDS=6. Both judges explicitly state they would not reject the
draft as AI-generated; the residual tells are rated weak/borderline by both.

## Cited tells (merged)

1. **Doubled "(it does not; ...)" parenthetical** in §4's two-hypotheses
   sentence (both judges; the round's only tell cited by both).
2. **"license" as mono-verb** for what evidence permits, ~6 uses, three at
   short range in §3 (judge D).
3. **Claim-then-un-claim cadence recurring every section** (judge C, weighted
   low, "defensible as substance" for a pre-registered paper).
4. **Virtue-openers** at the head of Limitations ("Every claim in this paper
   is scoped to its evidence.") and Scope ("...we report both at the same
   evidentiary standard as the wins.") (judge D, weak).

## Fixes applied before round 3

- Tell 1: the two-hypotheses sentence broken into three sentences with
  differently-shaped rebuttals ("A larger cache does not eventually hold the
  bindings: ... Forced locality, which might have helped..., does not
  either; ..."). No number or scope changed.
- Tell 2: "license" reduced to 2 uses (intro's registered-phrasing sentence
  and §3's single "does not license"); the other four became
  supports/shows/permit, varying the §3 short-range repetition.
- Tell 3: honest-framing pins retained by design (the "what is not claimed"
  beats are pre-registered disclosure, not filler; judge C's own caveat
  concedes this); shape varied indirectly by the tell-1/2 rewrites.
- Tell 4: both virtue-openers cut ("Every claim... is scoped to its
  evidence." deleted, Limitations now leads with the scale limitation;
  Scope opener trimmed to "Two further results bound the claim.").

Recompiled after fixes: 0 errors, 0 `??`, 11 pages. Round 3 (fresh panel)
dispatched on the revised draft.
