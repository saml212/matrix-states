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

**End of quarantined values (second contamination round, `855f548`/`c106881`).**
No COUPLED/DECOUPLED/FLAT-COUPLED/
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

---

## THIRD CONTAMINATION QUARANTINE (2026-07-12) — code leak surface + elimination-leak sweep

Landed by the non-designing coordinator agent dispatched to close two vectors the
blind T2-repair designer (`c106881`) itself disclosed in §11.10's contamination
ledger, independently confirmed by its adversarial attack round. **Neither vector
exposes a NEW value beyond what `QUARANTINE_r0_void_values.md` or §1-§5 above
already quarantine — both are the SAME values escaping through a second door.**
Consolidated here rather than in a third file, per dispatch.

### 6. LEAK 1 — real values in a PERMITTED-to-read code file

The no-read list (§9.1 of `PARAM_AXIS_SCALING_DESIGN.md`, both quarantine rounds)
walls off *documents*. It never walled off *source code* — and a T2-repair
designer is explicitly required to read
`matrix-thinking/deltanet_rd/lm_recall_gap_probe_v2_rd.py` to do the job. Two
value-bearing spots were found, both belonging to the **FIRST, retracted VOID
build** (`05de661`), already quarantined in `QUARANTINE_r0_void_values.md` §1
("0.0454 → 0.1719 → 0.1882, rising monotonically with params" — the wikitext
recall-gap sequence at rungs with `acc_intact = 0.0000`, i.e. zero demonstrated
copy ability):

| # | Location (pre-redaction) | Commit introduced | Value leaked | Cross-ref |
|---|---|---|---|---|
| 6.1 | `lm_recall_gap_probe_v2_rd.py`, `check_t2b2_ceiling` docstring, ≈L1285 | `9ea3ce6` (carried forward unchanged since) | `"DiD=0.19 at rungs with acc_copy=0.0"` — a rounded restatement of the 392M wikitext figure (0.1882) | `QUARANTINE_r0_void_values.md` §1, the "0.19… rising with scale" sentence |
| 6.2 | `lm_recall_gap_probe_v2_rd.py`, in-module self-test, L1895 and L1897 | `9ea3ce6` (carried forward unchanged since) | `did_value=0.1882, acc_copy=0.0000` passed as literal call args, and restated in the assertion's failure-message string as `"(DiD=0.1882, acc_copy=0.0, regate 10.2)"` — the EXACT wikitext-392M figure, unrounded | `QUARANTINE_r0_void_values.md` §1, same sentence |

**Disclosure history:** 6.1 was hit unprompted by both the blind T2-repair designer
and its independent adversarial auditor (`c106881` §11.10, disclosure 1 — "a
residual redaction miss in a PERMITTED file… the independent attacker hit the
identical line"). 6.2 was found by this coordinator's own sweep (dispatch item 1,
"sweep all code, not just that line") and was not previously flagged by name in
`c106881`.

**Why this is inert, not a new hazard.** Both values belong to the first,
FATALLY-buggy build's shared-tensor mass-ablation numerator — a number that, per
`QUARANTINE_r0_void_values.md`'s own banner, "no longer exists" as a measurement of
anything (§9.1.8). Neither is a `855f548` R0 value and neither is per-rung
resolved at the granularity the live R0 table quarantines (6.1 says "rungs,"
plural). **Nonetheless both are redacted**, because (a) they are literal
duplicates of an already-quarantined value sitting in a file blind agents are
told to read, and (b) 6.2 in particular is MORE precise than 6.1 (exact `0.1882`
vs. rounded `0.19`) and was not the line either prior round flagged.

**REDACTED IN PLACE** (2026-07-12, this commit): both docstring/comment/test
sites in `lm_recall_gap_probe_v2_rd.py` now state the qualitative shape only
("a non-trivial recall gap reported at a rung with zero demonstrated copy
ability" / "a VOID-build-shaped case") with a pointer to this file, and the
self-test's literal call arguments were changed to a synthetic value
(`did_value=0.15`) that preserves the exact test assertion (`passes=False`
fires when `DiD` exceeds the `acc_copy + 2·SE` bound at `acc_copy=0`) without
carrying the real figure. **The test's pass/fail behavior is unchanged** —
`0.15 > 0.0 + 2×0.01 = 0.02` still trips `passes=False`, exactly as
`0.1882` did.

### 7. LEAK 2 — leak by elimination in §10.3 and its EXPERIMENT_LOG.md mirror

§10.3 of `PARAM_AXIS_SCALING_DESIGN.md` (as committed at `855f548`) quarantines
"the identity of the passing [T2b-2] cell" (its own item 3) in the same breath as
its closing sentence **names the two FAILING rungs explicitly** ("the T2b-2
rung-VOIDs at 14M and 98M"). Given the fitted rung set is public
(14M/98M/392M — the ladder itself, not a DiD outcome), **naming 2 of 3 rungs as
failing discloses the 3rd's status (passing) by elimination** — equivalent to
printing "the identity of the passing cell" that the very same paragraph claims
to withhold. Flagged by the blind T2-repair designer itself (`c106881` §11.10,
disclosure 2 — "the one that could have bitten me") and independently confirmed
by its adversarial attacker.

**Four sites carried the identical elimination-shaped phrase, all fixed this
commit:**

| # | Location | Commit | Pre-redaction text |
|---|---|---|---|
| 7.1 | `PARAM_AXIS_SCALING_DESIGN.md` §10.3, closing sentence | `855f548` | "the T2b-2 rung-VOIDs at 14M and 98M are artifacts of the broken ceiling" |
| 7.2 | `EXPERIMENT_LOG.md`, the `2026-07-12 — PARAM-AXIS R0 READ` entry (mirrors §10.3) | `855f548` | "the T2b-2 rung-VOIDs (14M, 98M) and the wikitext T2b-1 failures are the PROBE's failure" |
| 7.3 | `PARAM_AXIS_SCALING_DESIGN.md` §11.10, contamination-ledger disclosure 2 | `c106881` | quoted the same clause verbatim while describing the hazard: `*"the T2b-2 rung-VOIDs **at 14M and 98M**"*` |
| 7.4 | `PARAM_AXIS_SCALING_DESIGN.md` §11.10, "PROCESS FINDINGS FOR THE PI" item 1 | `c106881` | quoted it again in the designer's own recommended fix: `sealing *"at 14M and 98M"* → *"at two of the fitted rungs…"*` |

**7.3 and 7.4 matter as much as 7.1** — even a perfect fix to §10.3 leaves the
same disclosure sitting in the ledger that explains the fix, readable by anyone
who reads §11 (the PINNED, buildable T2-repair spec, not walled off the way the
quarantine files are). A future agent could learn nothing from §10.3 and
everything from §11.10.

**NOT touched:** `PARAM_AXIS_SCALING_DESIGN.md` §10.5, §11.8's second paragraph
("§9.6 item 2… admits only the 14M and 98M rungs"), and the mirrored sentence
in `EXPERIMENT_LOG.md`'s R0 entry ("Only 14M and 98M clear the ≥1.0 tok/param
floor"). **These are a DIFFERENT, non-quarantined gate** (§9.6 item 2's sample
floor, `tokens_seen / params ≥ 1.0` at the forced 0.328B common slice) — a fact
of the training *schedule* (total tokens run × param count), computable from
public, pre-registered numbers with no dependence on any measured `DiD` or
`acc_copy` value. It says nothing about T2b-2 (the ceiling check) and, alone,
discloses no rung's T2b-2 status. It only became load-bearing for the
elimination leak in combination with 7.1-7.4, which is why closing 7.1-7.4 is
sufficient without touching this gate's own disclosure.

**REDACTED IN PLACE** (2026-07-12, this commit): all four sites now read a
form that discloses only "two of the three fitted rungs (identities
QUARANTINED)" without naming which two — 7.3 was rewritten to describe the
leak's *shape* ("named the T2b-2 rung-VOIDs by specific rung") rather than
repeat the disclosing clause, and 7.4 was rewritten to describe the
*recommended fix* without repeating the clause it recommends replacing. A
`[COORDINATOR NOTE, landed 2026-07-12]` was added inline at 7.4 recording
that the designer's own recommendation was carried out.

---

**End of quarantined values (third contamination round).** No new numeric
value is disclosed anywhere in this section — every value named above was
already quarantined in this file or in `QUARANTINE_r0_void_values.md` before
this round began. This section exists to record WHERE duplicates were found
and closed, per the same provenance discipline as §1-§5.
