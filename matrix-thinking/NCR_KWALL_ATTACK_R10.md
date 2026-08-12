# NCR K-WALL — NARROW AUDIT ROUND 10 (scope: M1 + m1 ONLY, plus standing integrity; TERMINAL ON INSPECTION)

**Target:** `matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md` at
`3d339bf`, status `DRAFT-R9 — POST-AUDIT-9, AWAITING NARROW AUDIT
ROUND 10`, `5193` lines, whole-file
`md5 = e2ba60bd9b1a4fecafffa754bdc0f40c`.

**Charter (as adopted, `§A9-ADJUDICATION` + `NCR_KWALL_ATTACK_R9.md`
§6):** verify M1 (KW10.1, the rebuilt in-text `L6` payload) and m1
(KW10.2, the six-flip delta claim) by re-derivation and re-execution;
verify the standing integrity set (frozen zone, the `§R9` MD5 table,
the three disclosed scope boundaries, m2–m7's edits at INSPECTION
level); report anything release-unsound visible without expanding
scope. K1–K7's substance, both suites' figures as certified in R9, and
the settled sections are excluded and were not re-opened.

**VERDICT: REV-REQUIRED — 0 FATAL / 1 MAJOR / 3 MINOR.**

The MAJOR is at the M1 site and is a **one-number** correction: the
rebuilt `L6` is now arithmetically self-consistent (M1's chartered
discharge condition IS met, and the payload does reach `PASS`/`[]` by
execution) but its `COMPLETED` primary row declares
`elapsed_h=2.00`, which is **not reachable** under this design's own
per-attempt charging/enforcement rules (`1.20` enforced ceiling +
`0.0157` tail + `0.0053` startup allowance = `1.2210` maximum). KW10.1
charged a payload whose numbers *did not close*; the rebuild replaced
it with a payload whose numbers *close but cannot be produced*. Three
reachable replacements were executed and all `PASS` with `[]` — the
fix costs one line and no clause logic. m1 is **fully discharged**,
verified by an independent differential re-execution (set-equality of
the six flips confirmed, `B3-AMENDED` `PASS→PASS` confirmed, both
narrations confirmed against mechanism). Every integrity row
reproduces exactly.

---

## §0 METHOD

Everything below is executed or recomputed, never accepted from the
document or from any prior round's harness.

- **`r10_vcheck.py`** — this round's own transcription of
  `validity_check`, written from the current text (`:2319-2527`) and,
  for the OLD side, from the pre-Rev-8 text at `ad2bf48`
  (`:2254-2385`). Rev-8's/Rev-9's harnesses were not consulted (they
  do not exist on disk — see KW11.4). Rational arithmetic
  (`fractions.Fraction`) throughout, so no float artefact can excuse
  or manufacture a `1e-6` verdict.
- **`r10_payloads.py`** — the 24-payload suite reconstructed from the
  in-text test list (`:2529-2826`): `L1–L7,L7'`, `A1–A7,A6'`,
  `B1,B1',B2,B2',B3-OLD-STYLE,B3-AMENDED,B3-NEG,B4`, plus four
  rounding-literal variants.
- **`r10_probes.py`** — `D1/D1'` (m7) and `D2/D2'` (m3) re-run
  independently, to test m1's delta claim against Rev-9's OWN
  documented probes.
- **`r10_l6fix.py`** — three candidate reachable `L6` replacements,
  run to completion.

Scripts live in this session's scratchpad
(`/private/tmp/claude-501/.../scratchpad/`). They are ephemeral, like
every prior round's — see KW11.4.

---

## §1 SCOPE 1 — M1 (KW10.1): the rebuilt `L6`

### 1.1 Ledger arithmetic, re-derived independently — EXACT

The payload as written (`:2601-2617`): 12 primary `(K,seed)` pairs =
10 single-attempt `CRASHED-RECOVERED` (`ceiling_charged=true`, `1.20`
each) + 1 `COMPLETED` (measured `elapsed_h=2.00`, 1 canonical file) +
1 `GATE-REFUSED` at `attempt_n=1` (`0.0`).

| Figure | Claimed | Re-derived | ✓ |
|---|---|---|---|
| pairs | `10+1+1=12` | 12 | ✓ |
| `ceiling_charged_gpu_h` | `12.00` | `10 × 1.20 = 12.00` (exactly the 10 `ceiling_charged=true` rows) | ✓ |
| `realized_gpu_h_final` | `14.00` | `12.00 + 2.00 + 0.00 = 14.00`; U3 residual `\|14.00−14.00\| = 0` | ✓ |
| `ceiling_charged_fraction` | `12.00/14.00` exact | `0.857142857142857…`, `0.8571` to 4 dp | ✓ |
| U8 fraction half | passes | `\|12/14 − 12/14\| = 0 ≤ 1e-6` | ✓ |
| EBSO base 1 | `14.00>13.80` | true | ✓ |
| EBSO base 2 | `1<12` canonical | true (the single `COMPLETED` pair's file) | ✓ |
| EBSO base 3 | `attempt_n==1` primary `GATE-REFUSED` row exists | true | ✓ |
| EBSO mirror | `frac>0.50` | `0.8571>0.50` | ✓ |

The `0.71`-vs-`14.55` impossibility KW10.1 charged is genuinely gone:
the payload no longer borrows L5's disk state, and `12.00` is a
reachable member of `{1.20k + 2.32m}` — the set KW10.1 itself
identified.

**Single-row crashed pairs are reachable**, contrary to a plausible
objection: `:936-941` derives `PERSISTENTLY-ABORTED` explicitly from
"its attempt-1 row is non-`COMPLETED` and no attempt-2 row exists and
the retry gate closed it," so a crashed pair with exactly one row is a
defined terminal state. (Even were an attempt-2 `GATE-REFUSED` row
mandatory, it carries `elapsed_h=0.0` and changes no figure and no
verdict.)

### 1.2 Executed through the amended `validity_check` — PASS, `[]`

Transcribed and run, not trusted:

```
L6                 OLD=PASS  NEW=PASS   failure reasons: []
L6-literal0.8571   OLD=PASS  NEW=FAIL   ['U8-frac: declared 0.8571 != recomputed 0.8571428571428571']
```

The stated outcome reproduces exactly. The literal variant confirms
m4's rounding discipline is load-bearing for `L6` too
(`|0.8571 − 12/14| = 4.2857e-5 > 1e-6`) — the doc's "`12.00/14.00`
exactly (`0.8571` to 4 dp)" phrasing is correct and must not be
flattened to the bare literal.

### 1.3 Substitution history disclosed in-text — YES

`:2555-2558`: *"Every payload this suite REBUILDS from a prior round's
non-closing numbers is disclosed here, not asserted PASS in silence:
A6/A6' (§R8 K4, closing KW9.4) and L6 below (§R9 M1, closing KW10.1) —
no other payload among the 24 is a rebuild."* Both rebuilt payloads are
named in the live body; the `L6` bullet itself carries the reason
(`:2593-2600`); the `§R9` M1 row records that Rev-8's harness silently
ran a `frac=1.0000` variant. Checked: `L5`'s in-text bullet states only
`realized=14.55` + 9 canonical + a first-attempt `GATE-REFUSED` and IS
satisfiable as written (my `L5` transcription — 9 `COMPLETED` summing
`9.75` + 4 ceiling rows `4.80` — passes), so "no other payload is a
rebuild" is accurate.

### 1.4 Reachability under the charging rules — **FAILS** (KW11.1, MAJOR)

See §4. `elapsed_h=2.00` on a `COMPLETED` **primary** row exceeds the
design's own maximum reachable primary row (`1.2210`) by `0.779` h.

---

## §2 SCOPE 2 — m1 (KW10.2): the six-flip delta — DISCHARGED

### 2.1 Differential re-execution, 24 payloads + 4 variants

Independent transcription, both directions, no `"N/A(old had no rule)"`
auto-match anywhere:

| payload | OLD | NEW | NEW failure reasons |
|---|---|---|---|
| L1 | PASS | PASS | `[]` |
| L2 | PASS | PASS | `[]` |
| L3 | PASS | PASS | `[]` |
| L4 | PASS | PASS | `[]` |
| L5 | PASS | PASS | `[]` |
| **L6** | PASS | **PASS** | `[]` |
| L7 | FAIL | FAIL | `U1` (STOPPED-BY-OPERATOR excluded — by design) |
| L7' | PASS | PASS | `[]` |
| A1 | FAIL | FAIL | `COMPLETE/strict: canonical 0 != 12`, `strict-m7-J1a` |
| A2 | FAIL | FAIL | `EB base2`, `EB base3` |
| A3 | FAIL | FAIL | `U1` |
| A4 | FAIL | FAIL | `EB base1`, `EB base3` |
| A5 | FAIL | FAIL | `otherwise-J1a`, `otherwise-K2` |
| A6 | FAIL | FAIL | **exactly** `['EB J4: ceiling_charged_fraction not <=0.50']` |
| A6' | PASS | PASS | `[]` |
| A7 | FAIL | FAIL | `U7: qualifier_band with neither clause (a) nor (b)` |
| **B1** | **PASS** | **FAIL** | `COMPLETE/otherwise-K2: 0 COMPLETED primary pairs` |
| **B1'** | **PASS** | **FAIL** | `CD-K2: 0 COMPLETED primary pairs` |
| **B2** | **PASS** | **FAIL** | `U8-frac: declared 0.2 != recomputed 0.9295774647887324` |
| **B2'** | **PASS** | **FAIL** | `U8-ccgh: declared 2.84 != recomputed 13.20` (+ frac half) |
| B3-OLD-STYLE | FAIL | FAIL | `U7: neither clause (a) nor (b)` |
| **B3-AMENDED** | PASS | PASS | `[]` — **non-flip, confirmed** |
| **B3-NEG** | **PASS** | **FAIL** | `U7-mirror: cond_canon 4 not < 4` |
| **B4** | **PASS** | **FAIL** | `U7-mirror: cond_canon 4 != ledger cond COMPLETED 0` |

**0 mismatches** against every PASS/FAIL the current text claims.
**Flip set over the 24 = `{B1, B1', B2, B2', B3-NEG, B4}` — set
equality with the corrected six-flip claim CONFIRMED** (no extra, no
missing). `B3-AMENDED` traces `PASS(OLD)→PASS(NEW)`: the corrected
"newly **emittable**, verdict UNCHANGED" wording is executed-true.

Rounding variants (m4's own claim, re-confirmed): `A6-literal0.9296`
FAILs on `U8-frac` **and** `EB J4`; `A6'-literal0.9296` flips
`PASS→FAIL` on `U8-frac` alone. Exact quotient: `A6` dies on `EB J4`
alone with U1/U2/U3/U8 passing first; `A6'` passes with `[]`.

### 2.2 B2 / B2' narration matches mechanism — CONFIRMED

- **B2** (`ccgh=13.20` correct, `fraction=0.20` wrong): the `ccgh` half
  of U8 recomputes `13.20` and PASSES; the payload dies on the
  **FRACTION** half. Exactly what `:2781-2800` now narrates.
- **B2'** (`ccgh=2.84` WITH `fraction=0.20`): `2.84/14.20 = 1/5`
  **exactly** — the "internally self-consistent … but both wrong"
  claim is arithmetically true (verified as a rational identity); U8's
  `ccgh` half recomputes `13.20` and `|13.20−2.84| > 1e-6`, so it dies
  on the **`ccgh`** half. Exactly what `:2801-2808` now narrates. My
  transcription is non-short-circuiting and therefore reports both
  halves; under the text's own `assert`-then-`assert` phrasing the
  `ccgh` assert fires first, so "before the fraction half is even
  reached" is accurate as written. Verdict identical either way.

The crossed narration KW10.2 found is fixed, and each bullet now
describes its own payload's mechanism.

---

## §3 SCOPE 3 — INTEGRITY

| Check | Claimed (`§R9`) | Measured | ✓ |
|---|---|---|---|
| Whole file BEFORE (`9811cd6`) | `b97bd59757caf33992d0fd96f373f098`, `4960` | identical | ✓ |
| Live body BEFORE (`1..3351`) | `225eab951036e1575a2c5a317a760f4e` | identical | ✓ |
| Live body AFTER (`1..3445`) | `c87ca924b24e1c6943e3cb57afe4a7e0` | identical | ✓ |
| **Frozen zone BEFORE** (content-anchored `## §A1-ADJUDICATION…`, `9811cd6` `3352..4530`) | `3805e7dac8893f272f51fb62210e28be`, `1179` | identical | ✓ |
| **Frozen zone AFTER** (same anchor, now `3446..4624`) | `3805e7dac8893f272f51fb62210e28be`, `1179` | identical — **byte-identical, by content anchor, not line number** | ✓ |
| Tail (`§R7`-remainder+`§A8`+`§R8`+`§A9`) | `430` both sides, `diff` EMPTY | `430`/`430`, `diff` returns nothing | ✓ |
| Arithmetic | `3351+1179+430=4960`; `3445+1179+430=5054`; `3445−3351=94` | all three hold | ✓ |
| `§R9` block itself | not hashed (fixed-point convention) | `5193−5054 = 139` lines, consistent | ✓ |
| Live-body `diff` hunks | **15**, mapping to 11 sites | `diff` reports exactly **15**: `3c3`, `565,566c…`, `1079c…`, `1451c…`, `2394,2396c…`, `2430c…`, `2496,2509c…`, `2544,2545c…`, `2662c…`, `2665c…`, `2671c…`, `2705c…`, `2707,2710c…`, `2714,2720c…`, `2983,2987c…` | ✓ |
| No whole-file AFTER hash | correct per convention | correct | ✓ |

**Every row of the `§R9` MD5/line table reproduces.** No undisclosed
contact anywhere: the header line (`:3`) is the only edit outside the
eight items, exactly as disclosed.

### 3.1 m2–m7 at their named sites (INSPECTION level)

| # | Site | Present? | Plain reading correct? |
|---|---|---|---|
| m2 | `§5` qualifier-band headline, `:3071-3080` | ✓ | ✓ — headline scoped *"For the PAID branch (`K_trig∈{26,28,30}`)"*, and a new sentence exempts the `$0` `K_trig=32` archive branch, citing U7 clause (b). KW10.3's contradiction is gone. |
| m3 | U7 Otherwise arm, `:2405-2418` | ✓ | ✓ — the arm now asserts the conditional canonical directory contains **0** `COMPLETED` files. Re-executed: `D2`/`D2'` both flip `PASS→FAIL` on `U7-otherwise:cond_canon 4 != 0`. |
| m4 | A6/A6' fraction, `:2734`/`:2740`/`:2747` | ✓ (3 spots) | ✓ — all three state `13.20/14.20` **exactly**, never the bare `0.9296`. |
| m5 | charging rule `:1088-1091`; crash-window row `:1463` | ✓ | ✓ — both now cite `rec["elapsed_s"]` as the TOP-LEVEL field (`:302`), contrasted against the nested `:202`. (But see KW11.3 — the first of these sits inside `0.1`.) |
| m6 | `trigger()` comment `:565-575` | ✓ | ✓ — states `diag` is OVERLOADED, not split by copy; names both meanings and both return sites; instructs consumers to key off the RETURN SITE. The false per-copy split is gone. |
| m7 | `COMPLETE` strict branch, `:2452-2463` | ✓ | ✓ — strict branch now also asserts J1(a) and J1(b) (`12==12`, free). Re-executed: `D1` flips on `strict-m7-J1b`, `D1'` on `J1a`+`J1b`; `L1`, `L7'`, `B3-*` and every other legitimate payload still PASS. |

### 3.2 The three disclosed scope boundaries

1. **`§A8`/`§R8` historical framing untouched** — **accurate.** The
   tail `diff` is empty, and the stale "five payloads" framing and the
   rounded `0.9296` literal are both still present in that section, as
   disclosed. Flagged by Rev-9 for round-10's judgement: I agree it is
   a dated historical self-report and does NOT warrant a contact; the
   live body is the durable spec (KW10.1's own reasoning) and it is now
   correct. **No finding.**
2. **200-state composition suite not re-run** — the *consequence* is
   accurate (no figure moves), but the *stated reason* is not. See
   KW11.3.
3. **`A2–A5, A7` not re-derived** — **accurate**, and I re-derived all
   five anyway this round: none flips; all five FAIL under both
   transcriptions, on clauses that predate Rev-8. The exclusion cost
   nothing.

---

## §4 FINDINGS

### KW11.1 — MAJOR. The rebuilt `L6` declares a `COMPLETED` **primary** row of `elapsed_h=2.00`, which no primary attempt in this design can produce; if it could, `τ=0.0157` — and with it `R_N ≤ 15.0157` and `T ≤ 15.3737` — would be false.

**Quote** (`:2601-2604`, the M1 rebuild): *"12 primary `(K,seed)` pairs
— 10 single-attempt `CRASHED-RECOVERED` (`ceiling_charged=true`,
`1.20` each → `12.00`), 1 pair `COMPLETED` (measured, `elapsed_h=2.00`,
1 canonical file), 1 pair `GATE-REFUSED` at `attempt_n=1` (`0.0`)"*.

**The design's own bound on a primary `COMPLETED` row**, assembled from
four independent live-body sites:

| Term | Value | Source |
|---|---|---|
| enforced per-attempt ceiling, primary | `1.20` | `:460` (`--ceiling-gpuh 1.20`); `:977-984` — *"the charged value and the enforced value are the same number by definition"* |
| training-check granularity slip (`log_every=500`) | `+0.0031` | `:1905-1912` (max observed) |
| post-ceiling eval phase | `+0.0126` | `:2141-2156` — *"max observed 45.5s = 0.0126 GPU-h"*, D5/KW3.9 |
| ⇒ true elapsed of a RETURNING attempt | **`≤ 1.2157`** | `:1902-1912` — *"its true `elapsed_h` can exceed `ceiling(N)` by at most the single-attempt tail … `0.0126+0.0031=0.0157`"* |
| + startup allowance on a PROMOTED row | `+0.0053` | `:1091` |
| ⇒ **maximum reachable primary `COMPLETED` ledger row** | **`1.2210`** | |

`2.00` exceeds this by `0.779` GPU-h — **50× the entire `τ` tail**. For
scale, the design's own pricing puts a primary cell's nominal at
`≈0.55` h (`:2100`, `≈6.65`h/12) and its per-K `≥2×nominal` ceilings at
`1.0211/1.1073/1.1946` (`:2110-2112`); `:2122-2139` records that the
largest max/nominal ratio **ever observed in this program** is
`1.2069×`. A `2.00`-hour primary cell is `≈3.6×` nominal.

**Why this is a defect and not a nitpick.** The two claims cannot both
stand:

- If a primary `COMPLETED` attempt CAN measure `2.00`, then `:1902-1912`'s
  tail term is wrong, `R_N ≤ 15.0157` is wrong, `T ≤ 15.3737` is wrong,
  and the `15.50` pool declaration loses its derivation — the exact
  bound the build charter and the release statement quote.
- If it CANNOT (it cannot), then the in-text `L6` payload specifies a
  disk state the orchestrator can never write — and `:2529-2558` plus
  R7 §9 item 4(a) wire that payload list as a **build-stage unit test**
  (R9 charter addition (h) asserts its failure-reason list). The build
  agent has the prose and nothing else (KW10.1's own reasoning, still
  true: no `.py` exists — KW11.4).

**Severity, argued both ways, honestly.** Lighter than KW10.1 in one
respect: the payload reaches its stated verdict, so coverage of the
`-SUSPECT-OVERCHARGE` mirror clause is real, not negative, and no
`validity_check` clause reads a per-row ceiling, so no check behaviour
is affected. Heavier in another: it is the same failure class as
KW9.4/KW10.1 (an in-text payload whose numbers a real run cannot
produce), it was introduced BY the fix for that class, and it now
contradicts a §4 derivation the charter's release sentence quotes
verbatim. Classed MAJOR on the "durable spec must be producible" rule
the gauntlet has applied twice; the coordinator may reasonably adopt it
as a one-line contact rather than a full revision cycle.

**Discharge condition (cheap — three executed options).** Restate
`L6`'s composition with a reachable `COMPLETED` row (or none). All
three of the following were run to completion this round and PASS with
`[]`:

- **(A)** 10 single-attempt `CRASHED-RECOVERED` + 1 pair crashed twice
  (`+2.40`) + 1 `GATE-REFUSED` — `realized=14.40`, `ccgh=14.40`,
  `fraction=1.0000`, 0 canonical. (This is the composition Rev-8's
  harness actually ran; adopting it also retires the substitution.)
- **(B)** A6's own composition relabelled `-SUSPECT-OVERCHARGE` —
  `realized=14.20`, `ccgh=13.20`, `fraction=13.20/14.20` exactly. (This
  is literally `A6'`; distinctness from A6 would then be worth a word.)
- **(C)** 11 ceiling rows + 1 `COMPLETED` at the cap `1.2210` —
  `realized=14.4210`, `ccgh=13.20`, `fraction=13.20/14.4210`.

Whichever is chosen, keep the EXACT-quotient discipline (`L6-literal`
FAILs U8, verified above) and re-print the failure-reason list.

### KW11.2 — MINOR. The corrected "these SIX flips … are the ONLY behavioural deltas" is again unqualified, and Rev-9's OWN m3/m7 probes falsify it.

**Quote** (`:2542-2550`): *"…confirming **SIX** payloads (B1, B1', B2,
B2', B3-NEG, B4) flip PASS→FAIL and B3-AMENDED … these SIX flips plus
B3-AMENDED's newly-sanctioned emission are the ONLY behavioural
deltas; nothing else regressed"*.

**Executed:** `D1`, `D1'` (m7 probes) and `D2`, `D2'` (m3 probes) all
flip `PASS(OLD)→FAIL(NEW)` — reproduced independently this round, and
already reported by `§R9`'s own suite-figures section. So the current
text vs pre-Rev-8 differs on **ten** documented payloads, not six.

The **substance** is again fine — all four extra flips are the new m3/m7
clauses correctly catching adversarial payloads, and nothing legitimate
regressed (`L1–L7'`, `A6'`, `B3-AMENDED` all still PASS). Only the
unqualified scope is wrong, and it is wrong in exactly the way KW10.2
was, one revision later, because the revision that fixed the count also
added two clauses whose probes flip. **Discharge:** scope the sentence
— *"…are the ONLY behavioural deltas **among these 24 payloads** (the
m3/m7 clauses added this revision additionally flip the `D1/D1'/D2/D2'`
probes, `§R9`)"* — or drop "ONLY."

### KW11.3 — MINOR. `§R9`'s composition-suite exclusion is justified by a statement that is false: m5's charging-rule edit DOES land inside `0.1`.

**Quote** (`§R9`, `:5118-5123`): *"none of M1/m1–m7 touch
`G1`/`G2`/`0.0`/`0.1`/`0.2`/the recovery procedure … and no edit this
revision lands inside that procedure."*

**Measured:** the `diff` hunk `1079c1088,1091` lands at `:1088-1091`,
inside the **Charging rule** paragraph (`:1083-1092`) of item **0.1
Per-attempt-directory reconstruction** (`:1067`). One of m5's two edits
is inside the procedure the sentence says nothing touched.

The *conclusion* is nevertheless correct and I verified why: the edit
is pure citation/disambiguation (`rec["elapsed_s"]` = the TOP-LEVEL
field at `:302`, never the nested `:202`), changes no number, no row,
and no clause, and merely globalises the disambiguation KW9.11 already
settled at three other sites. No composition figure can move.
**Discharge:** one clause — *"…except m5's charging-rule
disambiguation at `:1088-1091`, which changes no number, row, or clause
and therefore cannot move a composition figure."*

### KW11.4 — MINOR. Half of m7's adopted disposition was not performed, and the residue is declared empty anyway.

`§6` item 8 of `NCR_KWALL_ATTACK_R9.md`, **adopted verbatim** by
`§A9-ADJUDICATION` (*"DISPOSITIONS: M1 + m1–m7 per the report's §6
adopted verbatim as the Rev-9 charter"*), reads: apply J1(a)/(b) to
`COMPLETE`'s strict branch **"and**, jointly with M1, copy the round's
`vcheck_*.py`/`recon_*.py` transcriptions into the repo
(`matrix-thinking/` or `experiment-runs/`, both ≤25 MB) so the suite
that certifies this design survives the session that wrote it."

**Measured:** `find matrix-thinking -name '*.py'` returns only
`stageg/*` — nothing for this design; `git ls-files | grep -i 'kwall.*\.py'`
is empty. Rev-9's own suite-figures header even confirms the state
(*"original `vcheck_r8_rev.py`/`drive_vcheck_r8_rev.py` confirmed gone,
no `.py` under `matrix-thinking/`"*), and Rev-9's own harness is now
gone the same way — as are this round's, which I am not permitted to
commit (report-only write). Yet `§R9`'s residue section declares
**"Empty for the eight chartered items."**

**Mitigation, stated fairly:** build-charter item (i) (R9 §8) makes the
commit binding *before the first dispatch*, so the requirement is
deferred, not lost — and the clause half of m7 IS correctly
implemented and verified (§3.1). **Discharge:** either commit the
transcriptions now (any round's; they are ≤25 MB text) or amend the
residue paragraph to disclose this half as deferred to charter item
(i) — but not both silences at once.

---

## §5 GATE SUMMARY

| Item | Status |
|---|---|
| M1 — `L6` arithmetic self-consistent, exact-quotient fraction | **PASS** (re-derived, §1.1) |
| M1 — `L6` runs to its stated outcome (`PASS`, `[]`) | **PASS** (executed, §1.2) |
| M1 — every rebuilt payload disclosed in-text (A6/A6' AND L6) | **PASS** (§1.3) |
| M1 — every figure reachable under the charging rules | **FAIL — KW11.1 (MAJOR)** |
| m1 — six flips, by name, set-equality | **PASS** (executed, §2.1) |
| m1 — `B3-AMENDED` `PASS→PASS`, "newly emittable" | **PASS** (§2.1) |
| m1 — B2/B2' narration matches each payload's own mechanism | **PASS** (§2.2) |
| m1 — no auto-match can hide a flip | **PASS** (my harness has none; 0 mismatches) |
| m1 — "ONLY behavioural deltas" as stated | **FAIL — KW11.2 (MINOR)** |
| Frozen zone byte-identical, 1179 lines, content-anchored | **PASS** |
| `§R9` MD5/line table reproduces in full | **PASS** |
| 15 `diff` hunks → 11 named sites, no undisclosed contact | **PASS** |
| Scope boundary 1 (`§A8`/`§R8` historical) accurately stated | **PASS** |
| Scope boundary 2 (composition suite) accurately stated | **FAIL — KW11.3 (MINOR)** |
| Scope boundary 3 (`A2–A5, A7`) accurately stated | **PASS** (and re-derived: no flips) |
| m2, m3, m4, m5, m6, m7 present + plain-reading correct | **PASS** (§3.1) |
| Residue declared empty | **FAIL — KW11.4 (MINOR)** |
| K1–K7 substance, both suites, settled sections | **NOT RE-OPENED** (out of scope, per charter) |

**Nothing in the science, the budget derivation, the reconstruction
procedure, the band logic, the trigger, or the report schema is
impugned by this round.** All four findings are in-text
specification/disclosure text. No clause of `validity_check` needs to
change.

---

## §6 WHAT REV 10 MUST DO (binding disposition proposal)

Four edits, each one to five lines. No re-derivation; a single
payload re-run.

1. **N1 (KW11.1, MAJOR):** restate `L6`'s composition with a reachable
   `COMPLETED` row — option (A), (B), or (C) of §4, all three executed
   PASS this round — keeping the exact-quotient fraction; re-run `L6`
   and print its failure-reason list.
2. **n1 (KW11.2):** scope the "ONLY behavioural deltas" sentence to the
   24-payload suite, or name the four m3/m7 probe flips alongside.
3. **n2 (KW11.3):** one clause disclosing m5's charging-rule edit
   inside `0.1` and why it cannot move a composition figure.
4. **n3 (KW11.4):** commit the transcription scripts, or disclose the
   deferral to build-charter item (i) in the residue paragraph.

**Round 11 scope, recommended (binding on the next audit):** N1 ONLY —
re-derive the replacement `L6` composition against `:460`/`:977-984`/
`:1091`/`:1902-1912`/`:2141-2156` and re-execute it — plus the
byte-range integrity re-check. n1–n3 are one-clause disclosure edits
verifiable by reading the diff. Everything this report marks PASS is
settled and excluded: M1's arithmetic and execution, m1 in full, both
narrations, all six m2–m7 sites, the frozen zone, the MD5 table, and
scope boundaries 1 and 3. **Round 11 should be terminal on inspection.**

---

## §7 SCRIPTS

This session's scratchpad, written independently this round:

- `r10_vcheck.py` — hand transcription of `validity_check`, OLD
  (pre-Rev-8, `ad2bf48`) and NEW (current), rational arithmetic.
- `r10_payloads.py` — the 24-payload suite + 4 rounding variants,
  reconstructed from `:2529-2826`; differential driver with no
  auto-match.
- `r10_probes.py` — `D1/D1'` (m7), `D2/D2'` (m3), plus the exact-quotient
  identities.
- `r10_l6fix.py` — the three candidate reachable `L6` replacements.

**Ephemeral, like every prior round's** — see KW11.4. (Run with the
repo's documented `DRY_RUN_BYPASS=1`; these are CPU-only logic
harnesses, no training.)

---

## §8 BINDING BUILD CHARTER (restated, **NOT released this round**)

Unchanged from `NCR_KWALL_ATTACK_R9.md` §8 (= R8 §8 = R7 §9 as adopted
by `§A7-ADJUDICATION`, plus R8's (f)/(g) and R9's (h)/(i)). **This
charter is not in force until a round clears the design.**

1. R5's conditional build-release checklist, in full.
2. R6's five additions, incl. every negative test RUN TO COMPLETION.
3. The 3 micro-smokes (K=26/28/30) pass before queue-eligibility.
4. R7's five additions (a)–(e), unchanged.
5. R8's (f) schema validation of every emitted `attempts[]` row incl.
   `attempt_n`, with a `bootstrap_n>2` positive fixture; (g) the build
   recomputes `charged_vs_measured` from `ledger.attempts` and asserts
   equality before `orchestrator_report.json` is written.
6. R9's (h) the payload suite wired as a build-stage unit test asserts
   the **failure-reason list**, not merely the verdict, for every
   forced-fail negative; (i) the transcription scripts are committed to
   the repo alongside the design before the first dispatch.
7. **R10 addition (new, proposed binding):** (j) every in-text payload
   used as a build fixture must be **producible** — each row's
   `elapsed_h` within `charged_ceiling(arm) + τ + s` for a `COMPLETED`
   row, or exactly `charged_ceiling(arm)` for a `ceiling_charged` row —
   asserted by the fixture loader, not by reading. KW9.4, KW10.1 and
   KW11.1 are three instances of the same class; this closes it by
   construction.

**What a clean round unlocks (unchanged, and one edit away).** Nothing
in this round bears on the science, the budget derivation, the
reconstruction procedure, or the band logic — all settled and
re-verified. Once N1 lands and round 11 confirms it, the charter above
may be released and the design becomes build-eligible: `validity_check`,
the recovery/reconstruction procedure, the trigger and band procedures,
the orchestrator report schema, and the ≤`15.00` / `15.3737`-bounded
dispatch loop, with the 12-cell primary sweep (K∈{26,28,30} × 4 seeds
on the live K=24 recipe) and the conditional 160K arm queue-eligible
behind the three micro-smokes, under the `15.50` GPU-h declared pool
ceiling.

---

*Round 10, `NCR_KWALL_ATTACK_R10.md`. Verdict: **REV-REQUIRED** —
0 FATAL / 1 MAJOR / 3 MINOR. The coordinator adjudicates; this report
writes nothing else in the repo.*
