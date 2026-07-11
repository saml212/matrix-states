# Stage 01b — Scoped Re-Attack Report (reasoning-null-moss, round 1)

**Round type:** scoped re-attack (fresh context). Not a full re-attack. Adjudicates
whether the two originally-CRITICAL attacks (A1, A2) are now resolved below CRITICAL
after FIX-1..FIX-8, spot-checks the other six fixes, and reports any NEW attack the
fixes themselves introduced. All headline numbers were recomputed from the raw
artifacts, not taken from the draft on faith.

---

## Summary (for the coordinator)

- **A1 (was CRITICAL — provenance of the transient / registered unit of analysis):
  CLOSED.** FIX-1 landed in all three promised places (Table 1 dagger footnote, §4
  body, Appendix A full provenance paragraph). The disclosure is faithful to the raw:
  `PHASE2B_SUMMARY.json`'s `trajectories` field has exactly two corpus keys, both
  `UNRESOLVED`, and the paper now says exactly that and labels the four-way split a
  disclosed build-time re-derivation. Verified against design registry §16.18.3.
- **A2 (was CRITICAL, rebuttal down-rated to SERIOUS — multiplicity / denominator /
  compound-condition disambiguation): CLOSED (no longer CRITICAL).** FIX-2 discloses
  the 40-test denominator, the three K=32 CI exclusions, the "two expected by chance"
  framing, the two-hits-share-checkpoint-1,000 fact, and cleanly disambiguates the
  compound `holds()` condition (fires at exactly 1 of 20 cells) from "any CI excludes
  zero." Abstract/§1 wording is now precise ("the sole reading meeting the
  pre-registered differential condition").
- **FIX-3..FIX-8 spot-check: all six landed correctly.** No regressions in the
  variance-ratio framing, the Okpekpe characterization, the two new citations, the
  unit-level-validation limitation, the 250→5,000 measurement window, or the Merity
  year.
- **NEW findings introduced by the fixes: 2.**
  - **N1 (SERIOUS):** FIX-2's provenance paragraph (Appendix A) **mislabels the
    corpus of the first of the three K=32 CI-exclusion cells.** It calls it
    "encyclopedic×global at checkpoint 1,000," but the raw shows that Δ=−0.203,
    [−0.402,−0.004] cell is on the **reasoning-dense (OpenR1)** corpus, not
    encyclopedic (WikiText). A number-provenance error in the exact paragraph added
    to shore up rigor.
  - **N2 (MINOR):** the "three K=32 readings exclude zero, near the two expected by
    chance" framing pairs a 20-test numerator (K=32 only) with the false-positive
    expectation of the full 40-test family (2). The apples-to-apples comparison is
    3-of-20 K=32 vs ≈1 expected, or 5-of-40 total exclusions vs 2 expected — either
    way the excess is understated, mildly in the paper's own (null-friendly) favor.

**Neither originally-CRITICAL attack survives as CRITICAL.** Two new items are open
(1 SERIOUS, 1 MINOR); both are localized copy-edits, not structural. Final verdict at
bottom.

---

## Per-item detail

### A1 — Transient provenance / registered unit of analysis: **CLOSED**

**What the raw says.** `experiment-runs/2026-07-08_phase2b/results/PHASE2B_SUMMARY.json`,
field `trajectories`, has exactly two keys — `openr1-mix-ext` and `wikitext-mix-ext` —
both `"outcome": "UNRESOLVED"` with detail "holds() never fires and the terminal
CONVERGED-EQUIVALENT condition does not hold in full -- sub-case: power problem." That
is the registered pipeline's literal output: two corpus-level verdicts, both
unresolved. The per-arm four-way split is not in this summary's registered field; it
lives only in the per-arm `holds_by_c` blocks of the trajectory JSONs.

**Design registry corroboration.** REASONING_LINK_DESIGN.md §16.18.3 ("THE FOUR
PRIMARY (corpus × arm) CONTRASTS — a build-time scoping finding, disclosed") states
`analyze_corpus` computes only one hexachotomy classification per corpus (2 total),
using the **global** arm's own `holds_by_c` as the corpus's representative signal, and
that the four-way breakdown re-applies the same `classify_trajectory` primitives to
each arm's own trajectory. It explicitly calls this "a disclosed, intentional
build-time choice ... not a crash or silent bug."

**What the draft now says (all three FIX-1 sites present and accurate):**
- Table 1 wave-5 row carries `$^{\dagger}$` on "4 contrasts", with the caption footnote
  "Per-arm re-derivation; the registered corpus-level pipeline returned both corpora
  unresolved (Appendix~\ref{app:gates})." (03_geometric_null.tex L16, L26–27.)
- §4 body: "The registered corpus-level classifier, which uses the global arm as each
  corpus's representative, returns both corpora unresolved; a disclosed build-time
  per-arm re-derivation (folded into the analysis code, re-validated against the
  archived deltas; Appendix~\ref{app:gates}) splits this into four contrasts, three
  unresolved." (04_behavioral_contrast.tex L19–24.)
- Appendix A provenance paragraph (main.tex L85–94): "The registered pipeline computed
  one classification per corpus, using the global arm as that corpus's representative
  signal, and classified both corpora unresolved... This finer split was a build-time
  re-derivation performed after the corpus-level output was seen ... It is disclosed
  here as a re-derivation, not presented as the registered corpus-level verdict."

**Verdict.** The provenance is now disclosed exactly as the raw and the registry
describe it. The paper no longer presents "4 (corpus, arm) contrasts" as the registered
unit of analysis; it labels them a post-hoc re-derivation. The attack no longer rates
CRITICAL. **CLOSED.**

(Minor faithfulness note, not a finding: the paper simplifies "global arm as
representative" whereas §16.18.3 adds that `per_token` is folded in at the terminal
checkpoint for the #4/#5 split. This does not mislead — the global arm drives the
transient/persistent detection the sentence is about.)

---

### A2 — Multiplicity / denominator / compound-condition disambiguation: **CLOSED (no longer CRITICAL)**

**Raw recomputation of the primary det32 count.** From the two trajectory JSONs
(`trajectory_wikitext-mix-ext_phase2b.json`, `trajectory_openr1-mix-ext_phase2b.json`),
reading each `per_arm.<arm>.raw.<ckpt>.det32`, exactly **three** primary K=32 intervals
exclude zero, across all 4 contrasts × 5 checkpoints:

| Cell (corpus × arm @ ckpt) | Δ (K=32) mean | K=32 CI | det32 |
|---|---|---|---|
| **openr1** × global @ 1000 | −0.2030 | [−0.4025, −0.0036] | **true** |
| wikitext × per_token @ 1000 | −0.1683 | [−0.3006, −0.0361] | **true** |
| wikitext × per_token @ 2500 | −0.5000 | [−0.6241, −0.3758] | **true** |

All other 17 K=32 cells: det32 false. This matches the brief's expected set exactly
(openr1×global@1000, wikitext×per_token@1000, wikitext×per_token@2500). All three Δ are
negative → all in the harm direction (§4's "all in the harm direction" ✓). Two share
checkpoint 1,000 (§4's "two share checkpoint 1,000 across different cells" ✓ — and they
are genuinely different cells: different corpus *and* different arm).

**Compound `holds()` condition fires at exactly one cell.** Reading `per_arm.<arm>.holds_by_c`
across all 20 (contrast × checkpoint) cells, `holds=true` at exactly one: **wikitext ×
per_token @ 2500**. Hand-verified against the differential condition `det32 ∧ ¬det20 ∧
|Δ32|>|Δ20|` at that cell: det32=true, det20=false (Δ20 CI [−0.9200,+0.4164] spans
zero), |−0.500| > |−0.252|. ✓. At the other two det32-true cells the condition fails
because det20 *also* fires (openr1×global@1000: det20 CI [−0.3247,−0.0737]; wikitext×
per_token@1000: det20 CI [−0.2114,−0.0064]), so `¬det20` is false — matching
`holds_by_c=false` at both. So "the compound condition, not any single CI exclusion, is
the one determinate signal, firing at exactly 1 of 20 cells" is exactly right.

**What the draft now says.** §4 "Sign discipline and multiplicity" (L35–43): "Of 40
uncorrected 95-percent intervals in the primary table, three K=32 readings exclude
zero, near the two expected by chance, all in the harm direction ...; two share
checkpoint 1,000 across different cells... Only the checkpoint-2,500 cell also satisfies
the pre-registered differential condition ...; that compound condition, not any single
interval, is the one determinate signal." Appendix A repeats the 40-test denominator
and enumerates the three cells with Δ and CI. Abstract: "the sole reading meeting the
pre-registered differential condition (one of three that cross zero across forty
tests)." §1: "the sole reading meeting the pre-registered differential condition."

**Verdict.** Every element A2 demanded is now disclosed and the compound-vs-any-CI
disambiguation is explicit and correct against the raw. The count (3), the denominator
(40), the shared-checkpoint fact, and the harm direction all recompute correctly.
**CLOSED** below CRITICAL. (One residual framing imprecision → N2 below; and one
factual label error the fix introduced → N1 below.)

---

## FIX-3..FIX-8 spot-check (all landed)

- **FIX-3 (variance-ratio 4.0 = heuristic routing threshold, not calibrated test;
  F(2,8)≈4.46): landed, and the F value is correct.** §5 (L14–16), Appendix A
  (L112–114), and Fig 3 caption (L164–167) all describe 4.0 as a heuristic routing
  threshold and note the one-sided 5% F(2,8) ≈ 4.46. Independently, F_{0.05}(2,8) =
  4.459 ✓. The variance ratio itself is internally consistent: (1.154/0.546)² = 4.467
  ≈ 4.47, and "2.1 times" = 1.154/0.546 = 2.11 ✓. (Wave-6 cohort SDs 1.154/0.546 are
  not in the provided raws; not in scope for this round — internal consistency only.)
- **FIX-4 (Okpekpe & Orvieto = optimizer-choice confound): landed.** §6 (L9–11): "show
  that optimizer choice, not architecture alone, drives much of the reported
  Transformer--recurrent recall gap." Correct characterization.
- **FIX-5 (Grazzi 2024 arXiv:2411.12537 + Merrill 2024 arXiv:2404.08819 cited; Grazzi
  engaged in §7): landed.** §6 (L11–16) cites both; §7 (L17–21) engages Grazzi as a
  competing expressivity-ceiling account: "the null may partly reflect this variant's
  ceiling rather than a wrong observable; wave 4's dissociation points toward the
  observable." Both arXiv entries verified against the API (see below) — titles,
  authors, author order, and year all match.
- **FIX-6 (unit-level-only validation limitation + expanded Appendix A note): landed.**
  §7 (L21–26) states the readout's recovery math is "validated only at unit level,
  never end-to-end on a real forward pass with a known target"; Appendix A (L63–70)
  carries the expanded note ("no positive control run end-to-end through a real forward
  pass ... does not, on its own, rule out a systematic extraction or state-convention
  error").
- **FIX-7 (21.8–46.4% fall measured from checkpoint 250 to 5,000): landed and
  consistent.** §3 (L60–62) and Fig 2 caption (L143–147) both anchor the fall to
  "checkpoint 250 (first post-warmup) to checkpoint 5,000." (The dissociation raw is
  not in the provided artifacts; window-label consistency verified, value not
  recomputed — out of scope for this round.)
- **FIX-8 (Merity year 2017): landed.** refs.bib `merity2017pointer` year = 2017
  (arXiv 1609.07843; 2017 is the ICLR publication year — defensible).

### arXiv API verification of the two new entries

```
2411.12537 → "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues"
             Grazzi, Siems, Zela, Franke, Hutter, Pontil   published 2024-11-19
2404.08819 → "The Illusion of State in State-Space Models"
             Merrill, Petty, Sabharwal                     published 2024-04-12
```
Both match the bib entries verbatim (title, author list, author order, year 2024).
Note: Grazzi et al. was later published at ICLR 2025; the earlier gauntlet note said
"Grazzi et al. 2025," but the draft cites the arXiv preprint (2411.12537, year 2024),
which is internally consistent and correct as an arXiv citation. Not a finding — flag
only if the authors prefer the published-venue year.

---

## NEW findings introduced by the fixes

### N1 — Appendix A mislabels the corpus of the first K=32 CI-exclusion cell

**Severity:** SERIOUS
**Type:** number-provenance / factual mismatch vs. raw

**Attack.** FIX-2's provenance paragraph (main.tex, Appendix A, L97–102) reads:

> "three $K{=}32$ intervals exclude zero: encyclopedic$\times$global at checkpoint
> 1{,}000 ($\Delta{=}{-}0.203$, $[-0.402,-0.004]$), encyclopedic$\times$per-token at
> checkpoint 1{,}000 ($\Delta{=}{-}0.168$, $[-0.301,-0.036]$), and
> encyclopedic$\times$per-token at checkpoint 2{,}500 ($\Delta{=}{-}0.500$,
> $[-0.624,-0.376]$, the reported transient)."

The first cell is mislabeled. The paper's own §2 defines "encyclopedic" = WikiText-103
= `wikitext-mix-ext` and "reasoning-dense" = OpenR1-Math = `openr1-mix-ext`. The
Δ=−0.203, [−0.402,−0.004] value is the **openr1-mix-ext × global @ 1000** cell — the
reasoning-dense corpus, **not** encyclopedic. There is no encyclopedic×global cell that
excludes zero: `wikitext-mix-ext × global @ 1000` has Δ=−0.1357, CI [−0.5433, +0.2719],
which spans zero (det32=false). So "encyclopedic×global at checkpoint 1,000" is a cell
that does not exist as described; the number belongs to the other corpus.

**Supporting evidence.**
- `trajectory_openr1-mix-ext_phase2b.json`, `per_arm.global.raw.1000`: deltas
  [−0.1515, −0.2955, −0.1622], mean −0.20305, `ci_low` −0.40246, `ci_high` −0.00364,
  `det32` true. This is the Δ=−0.203, [−0.402,−0.004] cell — on **openr1** (reasoning-
  dense).
- `trajectory_wikitext-mix-ext_phase2b.json`, `per_arm.global.raw.1000`: mean −0.13571,
  ci [−0.54328, +0.27187], `det32` **false**. There is no encyclopedic×global exclusion.
- Design registry REASONING_LINK_DESIGN.md §16.18.4, table "**openr1-mix-ext × global**":
  row c=1000 reads "−0.2030 | [−0.4025, −0.0036] | **True**" — the registry itself
  attributes this exact cell to the openr1 (reasoning-dense) corpus.

**Why it matters.** The paper's Reproducibility appendix asserts "Every number in this
paper is computed from archived raw result files." A reviewer who follows that
invitation to the raws finds a data point attributed to the wrong corpus, in the one
paragraph FIX-2 added to demonstrate statistical candor. It also makes Appendix A
internally inconsistent with §4's own (correct) claim that the two checkpoint-1,000
hits are "across different cells": the truth is they are on *different corpora and
different arms* (reasoning-dense×global vs. encyclopedic×per-token), which is more
scattered — and more noise-like — than the "both encyclopedic" picture Appendix A
implies. The mislabel therefore also slightly weakens, rather than supports, the very
noise interpretation the paragraph argues.

**What would defuse it.** One-word fix: change "encyclopedic$\times$global at checkpoint
1{,}000" to "reasoning-dense$\times$global at checkpoint 1{,}000" (or the corpus token
the paper uses elsewhere for OpenR1). No number changes; Δ and CI are already correct.

---

### N2 — "near the two expected by chance" mixes a 20-test numerator with a 40-test denominator

**Severity:** MINOR
**Type:** statistical framing

**Attack.** §4 (L35–37): "Of 40 uncorrected 95-percent intervals in the primary table,
three $K{=}32$ readings exclude zero, near the two expected by chance." The count "three"
is drawn from the K=32 subset only (20 tests); the "two expected by chance" is the
false-positive expectation of the full family of 40 tests (40 × 0.05 = 2). These do not
share a denominator. The clean statements are either: (a) 3 of the 20 K=32 intervals
exclude zero vs ≈1 expected by chance for that subset — a ~3× excess, not "near"; or
(b) 5 of 40 total intervals exclude zero (the 3 K=32 plus 2 K=20: wikitext×per_token@1000
det20=true, openr1×global@1000 det20=true) vs 2 expected — also above, not "near."
Either framing shows a mild excess over chance; the current wording understates it, in
the direction that flatters the paper's null reading. The same slight imprecision rides
in the abstract's "(one of three that cross zero across forty tests)," which can be read
as "3 of 40 exclude zero" when 5 of 40 do.

**Supporting evidence.** det20=true cells (from the trajectory raws, `per_arm.<arm>.raw.<ckpt>.det20`):
wikitext×per_token@1000 (Δ20 CI [−0.2114,−0.0064]) and openr1×global@1000 (Δ20 CI
[−0.3247,−0.0737]); all other det20 false. So total CI exclusions across 40 tests = 5
(3 K=32 + 2 K=20).

**Why it is only MINOR.** The paper does not rest any conclusion on this cell being noise
vs. signal — it explicitly declines to interpret the transient as a real effect and
reports everything as measurement bounds. The framing is a common informal sanity check.
But since FIX-2's whole purpose was statistical candor, tightening the sentence to a
matched numerator/denominator (e.g. "three of the twenty K=32 intervals exclude zero,
about one more than the ~1 expected by chance; five of forty across both loads against
two expected") would remove the last soft spot.

**What would defuse it.** Rephrase to a single consistent denominator, or state both the
K=32-only comparison (3 of 20 vs ≈1) and the full-family comparison (5 of 40 vs 2).

---

## Cross-reference / build check

All `\ref` targets used by the fix edits resolve: `app:gates`, `app:figs`,
`tab:program`, `fig:gates`, `fig:dissoc`, `fig:transient`, `sec:setting`,
`sec:geometric`, `sec:behavioral`, `sec:replication`, `sec:scope`, `sec:related` are all
defined. The Table 1 `$^{\dagger}$` marker and its caption-embedded footnote text are
consistent. No broken reference introduced by the edits.

## Attacks I considered and dismissed

- **"The paper cites Grazzi as 2024 but the round-4 note said 2025."** Not a finding —
  the draft cites the arXiv preprint (2411.12537), whose posting year is 2024; the API
  confirms it. Internally consistent.
- **"366/366 and 90-cell geometric counts."** Out of scope for this scoped round
  (untouched by FIX-1..8); internal arithmetic checks out (312+8+16+30=366; 78+2+4+6=90).
- **"holds() should have fired at wikitext×per_token@1000 too."** No — det20 also fires
  there, so `¬det20` fails; `holds_by_c=false` at that cell, matching the raw. The
  compound condition correctly fires at exactly one cell.

---

## Verdict

**2 ITEMS OPEN** (N1 SERIOUS — corpus mislabel in Appendix A; N2 MINOR — multiplicity
denominator framing). Both originally-CRITICAL attacks (A1, A2) are **CLOSED**; FIX-3..8
all landed correctly; both new bib entries verified against the arXiv API.
