# Detector gate — round 1 — mstar-colm-er

Panel: two fresh-context subagents (Opus), dispatched independently with only
`paper/prompts/detector-judge.md` plus the 11 section files (prose only, told
to skip `%` build-metadata comments and read nothing else in the tree). Draft
state: post-gauntlet-round-1 (all FIX-1..FIX-6 applied, style PASS, format
audit 0-critical with dispositions applied), commit 78bc6f8.

## Verdicts

| Judge | Classification | Percentage-human read |
|---|---|---|
| A | human-written | ~96% human |
| B | human-written | ~95% human |

**Round result: NOT CLEAN** (bar is every judge at "100% human").
Consecutive-clean counter: 0. Round counter: 1 of MAX_ROUNDS=6.

## Cited tells (merged; the two panels converged independently)

1. **Labeled enumeration in §3** (both judges). "Three readings of this table
   are licensed; two are not." followed by the four sentence-heads
   "Licensed: / Licensed, and narrower...: / Also licensed...: / Not
   licensed:" — a filled four-slot grid rendered in prose; a human would fold
   it into a flowing argument.
2. **Recurring semicolon antithesis** (both judges). The identical
   "X does everything; Y does nothing" balanced cadence three times: abstract
   ("Zeroing the first block's state collapses recall to chance; zeroing the
   second block's changes nothing."), §5 bold lead ("The first block's state
   is causally necessary; the second block's is inert."), §5 second bold lead
   ("The storage is nonlinear; the model's own forward pass is the
   decoder."). A human would break at least one out of the matched cadence.
3. **Scoping-qualifier density** (judge A). "under this matched budget / at
   this scale and budget / claimed only at this matched training budget"
   re-attached to nearly every result sentence; judge A weights it lightly
   (each instance is content-bearing) but flags the density.
4. **Bold-lead paragraph template** (judge B, flagged "only for
   completeness"): five sections open paragraphs with a bolded
   assertion-then-elaborate lead.

## Fixes applied before round 2

- Tell 1: §3's licensed-readings paragraph rewritten as a flowing argument
  ("The readings this table licenses are narrower than the raw gap
  suggests. ... A second reading is licensed but narrower... The transformer
  column carries its registered phrasing... What the table does not license
  is..."). Every FIX-1 content element (budget scoping, the three-axis
  confound disclosure, the narrowed claim, the registered transformer
  phrasing, both not-licensed claims, the Zoology contrast) preserved; no
  numerical claim or evidence tag touched.
- Tell 2: abstract instance subordinated ("Recall collapses to chance when
  the first block's state is zeroed, while zeroing the second block's leaves
  it unchanged."); §5's second bold lead varied ("The storage is nonlinear,
  and the model's own forward pass is its decoder."). The §5 first bold lead
  keeps the antithesis once, deliberately, as the section's headline
  contrast.
- Tell 3: one doubled scope qualifier removed in §3 (the transformer sentence
  carried "at this scale and budget" AND "claimed only at this matched
  training budget"; the registered phrase "non-competitive at matched
  params/tokens" plus "at this scale and budget" plus "never extrapolated..."
  retain the full honest-framing pin). All other scope qualifiers are
  load-bearing pre-registration language and stay.
- Tell 4: retained deliberately (both judges call the bold-lead convention
  legitimate; judge B rates it weakest, "flagged only for completeness"); the
  tell-2 fix already varies one of the five leads.

Recompiled after fixes: 0 errors, 0 `??`, 11 pages, abstract in the 200-230
band. Round 2 (fresh panel) dispatched on the revised draft.
