# QUARANTINE — R0 (second read, `855f548`) per-rung `DiD` / `gap_true` /
# `gap_placebo` / S1 / S2 / `acc_copy` values, and DiD trend-shape statements

## ⚠ DO NOT READ unless explicitly authorized, AND you are not the agent
## tasked with re-pre-registering or repairing T2 (`pick_t2_marker_tokens`
## and everything downstream of it — T2a's bar, T2b's admissibility rule,
## or any successor instrument that gates admission into the `DiD` ladder).

**This is a DIFFERENT hazard from `QUARANTINE_r0_void_values.md`.** That
file holds the *first* VOID build's values (`05de661`) — a build with a
load-bearing FATAL bug in the `DiD` machinery itself (shared-tensor mass
ablation), so those numbers never measured the right thing at all and are
inert. **This file holds the *second* build's values** — the `DiD`
machinery in `lm_recall_gap_probe_v2_rd.py` (commit `9ea3ce6`) is **sound
and independently audited**; R0 (`855f548`) is VOID only because a
*separate, independent* teeth-check (T2, `pick_t2_marker_tokens`) is
broken. Repairing T2 does **not** touch a single line of the `DiD`
computation. **That is exactly the danger:** if a repaired T2 later
passes its own gates, the `DiD` numbers already sitting in this file
(computed once, already final) become the verdict-grade read with **zero
recomputation**. Anyone who designs the T2 repair while able to see this
table — or even just the *shape* of the trend across rungs — could
(consciously or not) tune the repaired T2's bar, its admissible-Δ
selection, or its rung-admissibility rule so as to admit or exclude rungs
in a way that produces a preferred verdict (COUPLED / DECOUPLED /
FLAT-COUPLED / RECALL-TREND-ONLY). **A trend-shape statement (rising /
flat / declining) is an equivalent leak to naming the verdict outright** —
§9.5's map is a deterministic function of the trend shape (β's sign and
CI) crossed with span_frac licensing, so "I know DiD rises across the
ladder" is functionally "I know the verdict is DECOUPLED-leaning" for
anyone who has read §9.5. **This is the precise laundering failure the
whole pre-registration apparatus (§9, the blind-pin protocol that produced
`QUARANTINE_r0_void_values.md`) exists to prevent**, now recurring one
level downstream, at the T2-repair step instead of the §9.1-normalization
step.

**Status of the underlying instrument:** the `DiD`/placebo/S1/S2 machinery
is VALIDATED (two opus audit rounds, 63/63 smoke, §9's full pre-registration
survived independent attack). **The verdict is VOID** (§9.5's VOID row,
T2a fails) — not FLOOR, not a measured absence of mechanism. **No
COUPLED/DECOUPLED/FLAT-COUPLED/RECALL-TREND-ONLY verdict has been computed,
is licensed, or may be inferred from anything on this page or from having
read it.** These values are preserved here — not deleted, this is
quarantine, not destruction — purely as a provenance record for whoever
eventually IS authorized to read them (e.g. the PI adjudicating whether to
re-run under a repaired T2, or a re-read that is explicitly re-registered
as such per §9.6's stop rule). They are walled off because reading them
defeats a specific downstream blind step: designing the T2 repair without
already knowing (or being able to infer) what a passing T2 would reveal.

**Why this file exists (mechanism of the leak, stated plainly).** §10 of
`PARAM_AXIS_SCALING_DESIGN.md`, as committed at `855f548`, interleaves the
mandatory VOID-diagnosis narrative (§10.1-§10.2, which a T2-repair designer
*must* read — it names the exact bug, `pick_t2_marker_tokens`'s
frequency-first key selection) with the per-rung outcome table (§10.4) and
two further paragraphs (§10.3 points 2-3, and §10.4's T1b/Δ-profile
paragraphs) that state trend shape in prose, not just in the table. A
designer instructed to read "the methodological findings that explain why
R0 is VOID" cannot avoid the outcome values without this extraction,
exactly as `QUARANTINE_r0_void_values.md`'s own dispatch discovered one
level up.

## Pointer back to source

| Value moved from | Commit | Section / lines (as originally committed) |
|---|---|---|
| `matrix-thinking/PARAM_AXIS_SCALING_DESIGN.md` | `855f548` | §10.3, points 2 and 3 (the S1 utilization values and the "largest `DiD`" T2b-2 finding) |
| `matrix-thinking/PARAM_AXIS_SCALING_DESIGN.md` | `855f548` | §10.4, the full per-rung table (14M/98M/392M × openr1/wikitext) |
| `matrix-thinking/PARAM_AXIS_SCALING_DESIGN.md` | `855f548` | §10.4, the T1b `gap_placebo` scale-monotone paragraph |
| `matrix-thinking/PARAM_AXIS_SCALING_DESIGN.md` | `855f548` | §10.4, the "disclosed residual" Δ-profile paragraph |
| `EXPERIMENT_LOG.md` | `855f548` | the `## 2026-07-12 — PARAM-AXIS R0 READ` entry — the S1/"largest DiD" sentence, and the mirrored per-rung table |

**⚠ A THIRD, UNFIXABLE LOCATION: the commit message of `855f548` itself.**
Unlike every location above, this one **cannot be redacted in place** — git
commit messages are immutable without a history rewrite, which is out of
scope and dangerous (shared history, concurrent agents). The commit
message body **restates the S1 values verbatim** ("S1 (DiD/acc_copy) > 1
in EVERY openr1 cell — 6.562 / 1.255 / 1.117") **and the "largest DiD"
finding** ("the ONLY T2b-2-passing cell is the one with the LARGEST DiD").
**`git log` (even WITHOUT `-p`, i.e. without ever diffing a file) shows
this by default** — the hazard is not confined to `git show`/`git diff`.
See the no-read list update in `PARAM_AXIS_SCALING_DESIGN.md` §9.1 and
item 6 of the dispatch that produced this quarantine.

---

## 1. From `PARAM_AXIS_SCALING_DESIGN.md` §10.4 — the full per-rung table, verbatim as committed at `855f548`

`N_rows = 2048`, `C_max = 8`, 16,384 resolved candidates per cell, both
corpora, all cells carrying complete §9.1.5 S2 log-prob fields, all
checkpoints quiesced + md5-pinned, all six from the same arm (`frozen_bias
per_token, λ=0.58`) at the same forced 0.328B common token slice (step
20,000; `miss_tokens = 0` at every cell).

| rung | corpus | `DiD` | `DiD` 95% CI | `gap_true` | `gap_placebo` | `acc_copy` | **T1a** | **T2b-1** | **T2b-2** | S1 (`DiD/acc_copy`) | S2 (`DiD_logp`) | tok/param | **in primary fit?** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14M | openr1 | 0.1025 | [0.0973, 0.1075] | 0.1046 | 0.0020 | 0.0156 | PASS | **FAIL** | **FAIL** | **6.562** | +0.4650 | 23.32 | **NO** |
| 14M | wikitext | 0.0192 | [0.0167, 0.0219] | 0.0208 | 0.0016 | **0.0000** | PASS | **FAIL** | **FAIL** | undef. | +0.1948 | 23.32 | **NO** |
| 98M | openr1 | 0.1495 | [0.1432, 0.1555] | 0.1547 | 0.0052 | 0.1191 | PASS | PASS | **FAIL** | 1.255 | +0.7359 | 3.357 | **NO** |
| 98M | wikitext | 0.1360 | [0.1301, 0.1419] | 0.1379 | 0.0018 | **0.0000** | PASS | **FAIL** | **FAIL** | undef. | +1.0980 | 3.357 | **NO** |
| 392M | openr1 | 0.1680 | [0.1615, 0.1744] | 0.1749 | 0.0069 | 0.1504 | PASS | PASS | PASS | 1.117 | +0.8477 | **0.836** | **NO** (below floor) |
| 392M | wikitext | 0.1557 | [0.1494, 0.1618] | 0.1578 | 0.0021 | **0.0000** | PASS | **FAIL** | **FAIL** | undef. | +1.2375 | **0.836** | **NO** |

**Observed pattern (stated here, in quarantine, precisely because stating
it anywhere else is the leak):** on the openr1 corpus, `DiD` is monotone
increasing across the three rungs read (14M < 98M < 392M), and `acc_copy`
is also monotone increasing across the same three rungs; S1 = `DiD /
acc_copy` is > 1 at all three and is itself monotone *decreasing*
(14M > 98M > 392M) as `acc_copy`'s growth outpaces `DiD`'s. On wikitext,
`acc_copy` reads exactly 0.0000 at all three rungs (the `"The"`-never-
occurs degenerate case, §10.2), so S1 is undefined at every wikitext cell
and only `DiD`/`gap_true`/`gap_placebo` are informative there; `DiD` on
wikitext is *not* cleanly monotone (0.0192 → 0.1360 → 0.1557 — a large
jump 14M→98M, then a smaller further rise 98M→392M). **None of this is
verdict-grade** — the admissible set is empty (§9.6 item 6, both corpora
required) and T2a's failure means VOID takes precedence over any trend
reading regardless.

## 2. From `PARAM_AXIS_SCALING_DESIGN.md` §10.3, verbatim as committed at `855f548`

> 2. **S1 (§9.1.5's mandatory utilization ratio, `DiD/acc_copy`) exceeds 1 in
>    *every* openr1 cell** — 6.562 (14M), 1.255 (98M), 1.117 (392M). §9.1.5
>    expected S1 to sit near `[0,1]`, *"bounded by the already-pinned T2b-2
>    ceiling."* A ratio of **6.56** is not a model property; it is a broken
>    denominator.
> 3. **The T2b-2 failures track `acc_copy`'s brokenness, not `DiD`'s magnitude** —
>    the *only* cell that PASSES T2b-2 (392M/openr1, margin −0.0140) is the cell
>    with the *largest* `DiD` (0.1680). A ceiling that the largest effect clears
>    and the smallest effect violates is not a ceiling.

(Point 1 of that list — "T2a itself — 0.11/0.23 on models known to have
the mechanism" — is **not** quarantined; it is a reference-model value,
independent of our own rung ladder, and remains in place at §10.1/§10.3 of
the design doc.)

## 3. From `PARAM_AXIS_SCALING_DESIGN.md` §10.4, T1b paragraph, verbatim as committed at `855f548`

> **T1b (§9.3) — reported as pinned.** `gap_placebo` is **small but non-zero and
> scale-monotone on openr1** (0.0020 → 0.0052 → 0.0069, 14M → 98M → 392M): the
> "bigger models are more brittle to upstream context damage" effect §9.2 predicted
> is **real and does grow with scale**, but it is an order of magnitude smaller than
> `gap_true`. The placebo is load-bearing in *direction* and modest in *magnitude*.

## 4. From `PARAM_AXIS_SCALING_DESIGN.md` §10.4, "disclosed residual" Δ-profile paragraph, verbatim as committed at `855f548`

> **Disclosed residual, exactly as `summarize_delta_match` predicted it.** The
> placebo's realized Δ profile runs **systematically shorter** than the true arm's
> (14M/openr1: mean Δ 86.5 vs 121.6; 392M/wikitext: 109.5 vs 151.9 — a −35 to −42
> token shift). §9.2's rejection-resampling makes this **inherent to the pinned
> procedure**, not a deviation from it. It is a **report, not a gate** (the
> instrument's own docstring says so), and its direction is **conservative** (a
> nearer corrupted token should damage *more*, inflating `gap_placebo` and
> *shrinking* `DiD`).

## 5. From `EXPERIMENT_LOG.md`, the `2026-07-12 — PARAM-AXIS R0 READ` entry, verbatim as committed at `855f548`

The S1 / "largest DiD" sentence:

> **CONSEQUENCE — T2b-2's premise is FALSE.** §9.4 built the ceiling check
> (`DiD ≤ acc_copy + 2·SE`) on "acc_copy is an UPPER BOUND." It is not: models with
> demonstrated AR read 0.11/0.23 on it. Three confirmations in-record: (1) T2a itself;
> (2) **S1 (`DiD/acc_copy`) > 1 in every openr1 cell — 6.562 / 1.255 / 1.117** (§9.1.5
> expected ~[0,1]); (3) the *only* T2b-2-passing cell is the one with the *largest*
> DiD.

The mirrored per-rung table (identical values to §1 above, condensed
column headers, PASS/FAIL abbreviated to P/F):

| rung | corpus | DiD [95% CI] | gap_true | gap_plac | acc_copy | T1a | T2b-1 | T2b-2 | S1 | S2 (DiD_logp) | tok/param | in fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14M | openr1 | 0.1025 [.0973,.1075] | 0.1046 | 0.0020 | 0.0156 | P | **F** | **F** | 6.562 | +0.465 | 23.32 | no |
| 14M | wikitext | 0.0192 [.0167,.0219] | 0.0208 | 0.0016 | **0.0000** | P | **F** | **F** | — | +0.195 | 23.32 | no |
| 98M | openr1 | 0.1495 [.1432,.1555] | 0.1547 | 0.0052 | 0.1191 | P | P | **F** | 1.255 | +0.736 | 3.357 | no |
| 98M | wikitext | 0.1360 [.1301,.1419] | 0.1379 | 0.0018 | **0.0000** | P | **F** | **F** | — | +1.098 | 3.357 | no |
| 392M | openr1 | 0.1680 [.1615,.1744] | 0.1749 | 0.0069 | 0.1504 | P | P | P | 1.117 | +0.848 | **0.836** | no (floor) |
| 392M | wikitext | 0.1557 [.1494,.1618] | 0.1578 | 0.0021 | **0.0000** | P | **F** | **F** | — | +1.238 | **0.836** | no |

---

**End of quarantined values.** No COUPLED/DECOUPLED/FLAT-COUPLED/
RECALL-TREND-ONLY verdict was ever computed for R0 — the admissible set is
empty and T2a's failure VOIDs the read upstream of any trend
classification (§9.5: VOID → FLOOR → the table; here the precedence never
even reaches FLOOR). **Do not read this file, `git log`/`git show` on
`855f548` (including the commit message alone — see the pointer table
above), or the raw JSONs/log under
`experiment-runs/2026-07-12_param_axis_r0/` in an attempt to inform a T2
repair's bar, distance selection, or admissibility rule.** A repair
designer needs, and has, everything in `PARAM_AXIS_SCALING_DESIGN.md`
§10.1-§10.2 (the T2a failure and the `pick_t2_marker_tokens` diagnosis in
full, with exact mechanism and numbers) and §10.6-D1 (the §9.2/§9.6
spec inconsistency) without opening this file.
