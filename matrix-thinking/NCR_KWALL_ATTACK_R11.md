# NCR K-WALL — NARROW AUDIT ROUND 11 (scope: N1 + integrity ONLY; TERMINAL ON INSPECTION)

**Target:** `matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md` at
`e7d29f0`, `5264` lines, whole-file
`md5 = efd54b134fa43d6db7513502ccdf53bd`; predecessor `3d339bf`
(`5193` lines).

**Charter (binding, per `NCR_KWALL_ATTACK_R10.md` §6 and the design's
`§A10-ADJUDICATION`):** re-derive the restated `L6` composition (N1)
independently against `:460` / `:977-984` / `:1091` / `:1902-1912` /
`:2141-2156`, confirm it is R10 §4's option (A) exactly, re-execute it
through `validity_check` to `PASS`/`[]`, and confirm the in-bullet
clause trace; verify n1–n3 by diff inspection; verify the standing
integrity set. Everything R10 marked PASS is settled and was NOT
re-opened: M1's arithmetic, m1 in full, both B2/B2' narrations, the
m2–m7 sites, scope boundaries 1 and 3, K1–K7's substance, and both
suites' certified figures.

**VERDICT: CLEAR — 0 FATAL / 0 MAJOR / 0 MINOR.**

N1 re-derives exactly, is producible on every row with margin, is
byte-for-byte R10 §4's option (A), and executes to `PASS` with failure
reason list `[]` under a transcription written fresh this round from
the current text. Its in-bullet clause trace is correct as written,
clause by clause, against `:2319-2527`. More than the charter asked:
the declared figures are not merely per-row producible, they are
**trajectory-reachable** — a strictly-sequential all-crash wave under
the design's own gate rules lands on `realized=14.40`,
`ccgh=14.40`, `fraction=1.0000`, `0` canonical and a primary
`attempt_n==1` `GATE-REFUSED` row, *exactly* the five figures `L6`
declares. KW11.1's contradiction with `τ=0.0157` / `R_N ≤ 15.0157` /
`T ≤ 15.3737` is fully retired. n1–n3 are accurate as written; n3 is
discharged with byte-level evidence (all seven committed scripts are
`md5`-identical to the scratchpad originals R9-rev and R10 actually
ran, and re-executing them reproduces §R9's and §R10's recorded
figures with zero mismatches). Every integrity row reproduces; the
frozen zone is byte-identical; the diff contains exactly what §A10
discloses and nothing else.

Five **non-blocking observations** are recorded in §4. None is
release-unsound; none changes a figure, a clause, a verdict, or a
bound. Two of them (OBS-2, OBS-3) are one-word/one-line items that
fold naturally into the release edit the coordinator must make anyway.

---

## §0 METHOD

Everything below is executed or recomputed from the current text. No
prior round's harness was consulted before writing this round's own.

- **`r11_vcheck.py`** — this round's own transcription of
  `validity_check`, written from `e7d29f0`'s `:2319-2527` only, and
  deliberately structured differently from R10's (document-order named
  clauses, no OLD/NEW mode axis, different failure-string vocabulary)
  so that a shared transcription error is unlikely. Rational
  arithmetic (`fractions.Fraction`) throughout — no float can
  manufacture or excuse a `1e-6` verdict.
- **`r11_l6.py`** — independent construction of the restated `L6`
  payload from the bullet's own prose (`:2614-2633`); the figure
  re-derivation; the per-row producibility audit against the four
  cited charging sites; execution; seven negative controls; and a
  strictly-sequential orchestrator simulation built from §4's
  ORCHESTRATOR CONTRACT (`:832-999`).
- **`r11_variants.py`, `r11_label.py`** — row-set and status-literal
  sensitivity probes on the same payload.
- **The committed suites** (`matrix-thinking/kwall_suites/r9rev_*.py`,
  `r10_*.py`) re-executed as shipped, and `md5`-compared against the
  scratchpad originals.

Scripts for this round are in the session scratchpad; the durable
suites are the committed ones (n3).

---

## §1 SCOPE 1 — N1: the restated `L6`

### 1.1 It is R10 §4 option (A), exactly

| | R10 §4 option (A) | Design `:2614-2626`, as written |
|---|---|---|
| single-attempt `CRASHED-RECOVERED` | 10 | 10 (`ceiling_charged=true`, `1.20` each → `12.00`) |
| pair crashed on both attempts | 1 (`+2.40`) | 1 (`ceiling_charged=true` on each row, `1.20+1.20=2.40`) |
| `GATE-REFUSED` | 1 | 1, at `attempt_n=1` (`0.0`) |
| pairs | 12 | `10+1+1=12` |
| `realized` | `14.40` | `14.40` (`=12.00+2.40+0.00`) |
| `ccgh` | `14.40` | `14.40` (`=12×1.20`) |
| `fraction` | `1.0000` | `14.40/14.40` exactly (`=1.0000`) |
| canonical files | `0` | `0` total |

**Identical.** The design's version additionally states the
`ceiling_charged` flag per row, the `attempt_n` of the refusal, and
the universal-assertion attributions — strictly more information, no
divergence. §A10's stated reason for choosing (A) over (B)/(C) — that
(A) is the composition Rev-8's harness actually ran, so spec and
history become identical and the substitution lineage retires — is
consistent with `§R9`'s own M1 row (`:5093`: *"Rev-8's harness silently
substituted a rebuilt `frac=1.0000` variant"*) and with
`r10_l6fix.py`'s candidate A.

### 1.2 Figures re-derived independently — EXACT

Rational arithmetic, from the bullet's prose, not from any script:

| Figure | Declared | Re-derived | ✓ |
|---|---|---|---|
| distinct primary pairs | `10+1+1=12` | 12, and the set equals `{26,28,30}×{0,1,2,3}` exactly | ✓ |
| `ceiling_charged` rows | 12 | 12 rows × `1.20` | ✓ |
| `ceiling_charged_gpu_h` | `14.40` | `12 × 1.20 = 72/5 = 14.40` | ✓ |
| `realized_gpu_h_final` | `14.40` | `12.00 + 2.40 + 0.00 = 72/5`; U3 residual `\|14.40−14.40\| = 0` | ✓ |
| `ceiling_charged_fraction` | `14.40/14.40` exact | `Fraction(72,5)/Fraction(72,5) = 1` | ✓ |
| 4-dp literal vs exact quotient | `1.0000` | `\|1.0000 − 1\| = 0` — **exactly zero**, not merely `≤1e-6` | ✓ |
| EBSO base 1 | `14.40>13.80` | true | ✓ |
| EBSO base 2 | `0<12` canonical | true (no `COMPLETED` row ⇒ no canonical copy, G2) | ✓ |
| EBSO base 3 | primary `attempt_n==1` `GATE-REFUSED` row exists | true | ✓ |
| EBSO mirror | `frac>0.50` | `1 > 0.50` | ✓ |

Note a structural improvement the restatement buys for free: because
the exact quotient is `1`, the 4-dp literal `1.0000` **is** the exact
quotient. The rounding hazard m4 made load-bearing (and which the
superseded `0.8571` payload was exposed to, `L6-literal0.8571` FAILing
U8 by `4.2857e-5`) **cannot arise for this payload at all**. The
bullet's "per m4's same rounding discipline" phrasing remains correct
and is now trivially satisfied.

### 1.3 Per-row producibility — PASS, with margin (KW11.1 CLOSED)

The design's own bound on a primary row, re-assembled independently
from the four cited sites:

| Term | Value | Source |
|---|---|---|
| enforced/charged per-attempt ceiling, primary | `1.20` | `:460` (`--ceiling-gpuh 1.20`); `:977-984` — *"the charged value and the enforced value are the same number by definition"* |
| single-attempt tail τ (eval `0.0126` + `log_every=500` granularity `0.0031`) | `+0.0157` | `:1902-1912`, `:2141-2156` |
| startup allowance `s` on a promoted/`COMPLETED` row | `+0.0053` | `:1091` |
| ⇒ **max reachable primary `COMPLETED` ledger row** | **`1.2210`** | |

Executed row-by-row over all 13 rows of the payload:

- every `ceiling_charged` row is **exactly** `1.20` = `charged_ceiling("primary")` — the equality item (j) demands, not an inequality;
- the `GATE-REFUSED` row is exactly `0.0` (`:843-848` for an
  attempt-1 refusal, `:941-945` for a refused retry);
- **the payload contains ZERO `COMPLETED` rows**, so the `1.2210`
  cap — the term KW11.1's `2.00` violated by `0.779` h — is not even
  exercised. The maximum row in the whole payload is `1.20`, i.e.
  `0.0210` **below** the cap.

KW11.1's stated consequence is therefore fully retired: no in-text
fixture now asserts a row that would falsify `:1902-1912`'s τ term,
and with it `R_N ≤ 15.0157` (`:1926`) and `T ≤ 15.3737` (`:2007-2009`,
`:1146`, `:1590`) stand un-contradicted by the design's own test list.

### 1.4 Re-executed — PASS, failure-reason list `[]`

Through **this round's own** transcription (`r11_vcheck.py`), not
R10's:

```
band=INCOMPLETE-AT-K      resolution=TRIGGER-UNRESOLVED  -> PASS  failure reasons: []
band=FRONTIER-AT-K*=26    resolution=unanimous           -> PASS  failure reasons: []
```

(Both band/resolution framings run because the bullet does not fix
them; the verdict is invariant, as it must be — neither field is read
by any `EXHAUSTED-BUDGET*` clause beyond U6's membership test.)

Cross-check, the committed artifact as shipped:

```
$ python3 matrix-thinking/kwall_suites/r10_l6fix.py
candidate A (14.40/14.40, frac 1.0000): PASS []
candidate B (14.20/13.20, frac 13.20/14.20): PASS []
candidate C (realized 14.421, ccgh 13.2, frac 0.9153318077803204): PASS []
```

**Two independent transcriptions, same verdict, `[]` both ways.**

**Negative controls on the same payload — the instrument has teeth**
(each executed):

| Perturbation | Result |
|---|---|
| declared `fraction=0.50` | `U8-frac` **and** `EBSO J4-mirror` |
| declared `fraction=0.90` | `U8-frac` |
| declared `ccgh=12.00` (the superseded figure) | `U8-ccgh` |
| declared `realized=14.00` (the superseded figure) | `U3` **and** `U8-frac` |
| same ledger mislabelled plain `EXHAUSTED-BUDGET` | `EB J4: ceiling_charged_fraction not <=0.50` |
| same payload with 12 canonical primaries on disk | `EB base2` |
| `GATE-REFUSED` row removed | `EB base3` |

Every clause the bullet claims is load-bearing fails when, and only
when, its own premise is broken. In particular the two superseded
figures (`12.00`, `14.00`) are now *detected as wrong* by the check —
the restatement is not cosmetic.

### 1.5 The in-bullet clause trace — correct as written

Checked term by term against `:2319-2527`:

| Bullet's claim | Clause | Verified |
|---|---|---|
| "`ceiling_charged_gpu_h=14.40` … so universal assertion 8 PASSES" | U8 first half, `:2426-2429` — recompute over `ceiling_charged` rows | ✓ recomputes `72/5`, residual `0` |
| "`realized_gpu_h_final=14.40` … universal assertion 3 PASSES: `\|14.40−14.40\|=0`" | U3, `:2325-2329` | ✓ residual exactly `0` |
| "`ceiling_charged_fraction=14.40/14.40` exactly" | U8 second half, `:2430-2435` (guarded by `realized>0`; `14.40>0`, so the half fires) | ✓ residual exactly `0` |
| "the three base clauses hold (`14.40>13.80` ✓, canonical count `0<12` ✓, the `GATE-REFUSED` pair's row has `attempt_n==1` ✓)" | `:2518-2520` inheriting `:2501-2509` | ✓ all three, and base 3's full predicate (`arm=="primary"` **and** `attempt_n==1` **and** `status=="GATE-REFUSED"`) is satisfied by the single refusal row |
| "PLUS the mirror clause `ceiling_charged_fraction>0.50`" | `:2521` | ✓ `1 > 0.50` |
| "with U1/U2/U3/U8 and all three base clauses confirmed PASSING first" | U1 `:2320`, U2 `:2324` (`14.40 ≤ 15.50`) | ✓ — and the stronger claim implied by "failure-reason list `[]`" also holds: U4 (`d_override==K+1` on every row incl. `GATE-REFUSED`, `:2330-2337`), U5, U6, and U7's Otherwise arm (m3: conditional canonical directory contains 0 `COMPLETED` files) all pass too |

The trace is a *subset* assertion about the interesting clauses, and
it is true; the `[]` claim covers the rest and is also true, by
execution.

### 1.6 Trajectory reachability — the declared figures are an EXACT hit

The charter asked for per-row producibility. I also asked the stronger
question: can a real wave, run under §4's contract, arrive at this
ledger? Simulated directly from the contract — K-major cell order
(`:830-836`), exactly one attempt in flight (`:835`, `:437-444`), HARD
GATE `realized + 1.20 ≤ 15.00` before **every** dispatch (`:977-984`),
RETRY GATE `realized < 12.00` before an attempt-2 dispatch
(`:985-989`), a refused gate appending a `0.0` row (`:843-848`,
`:941-945`) — with every attempt crashing:

```
realized               : 14.40   (L6 declares 14.40)
ceiling_charged_gpu_h  : 14.40   (L6 declares 14.40)
fraction               : 1       (L6 declares 1.0000)
canonical primaries    : 0       (L6 declares 0)
>=1 primary attempt_n==1 GATE-REFUSED row: True
```

**All five declared quantities are hit exactly**, by a trajectory the
orchestrator can actually walk. This is a strictly stronger
producibility result than item (j) requires, and it is the first time
in this lineage an `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE` fixture has
had one. (See OBS-1 for the one respect in which the bullet's *pair
narration* differs from that trajectory's pair partition — a
difference that moves no figure and no verdict, confirmed by
execution.)

---

## §2 SCOPE 2 — n1–n3 by diff inspection

### 2.1 n1 (KW11.2) — the delta sentence, scoped, with four flips named

The sentence at `:2550-2558` now reads *"…are the ONLY behavioural
deltas **WITHIN THE 24-PAYLOAD SUITE** (§A10/n1 scoping, per KW11.2:
the four §R9 teeth-probes OUTSIDE the suite — `D1/D1'`, m7's
`COMPLETE`/strict ledger clause, and `D2/D2'`, m3's U7 Otherwise-arm
assertion — also flip PASS(OLD)→FAIL(NEW), by construction and by
design…)"*.

Verified, and every factual sub-claim re-executed this round:

| Sub-claim | Verified |
|---|---|
| six flips within the 24 | ✓ — over the 24 named payloads the flip set is exactly `{B1, B1', B2, B2', B3-NEG, B4}` under BOTH committed suites |
| **four** probes named, by name | ✓ — `D1`, `D1'`, `D2`, `D2'` |
| `D1/D1'` attributed to **m7's `COMPLETE`/strict ledger clause** | ✓ — executed: `D1` → `COMPLETE/strict-m7-J1b`; `D1'` → `strict-m7-J1a` + `J1b` |
| `D2/D2'` attributed to **m3's U7 Otherwise arm** | ✓ — executed: both → `U7-otherwise: cond_canon 4 != 0` |
| direction `PASS(OLD)→FAIL(NEW)` | ✓ all four |
| "by construction and by design, as the forced-fail demonstrations of their new clauses" | ✓ — each probe is the negative fixture for the clause its own revision item added |

The unqualified "ONLY" KW11.2 falsified is gone, and the scoped
sentence is now true as written.

### 2.2 n2 (KW11.3) — m5's textual contact with step `0.1`, disclosed

`§R9`'s composition-suite exclusion bullet (`:5133-5147`) now reads
*"none of M1/m1–m7 touch `G1`/`G2`/`0.0`/`0.1`/`0.2`/the recovery
procedure's **LOGIC**. (§A10/n2 correction, per KW11.3: one m5
clause — the `elapsed_s` disambiguation — does land TEXTUALLY inside
step `0.1`'s charging-rule sentence; it renames WHICH field is read…
without altering any arithmetic, threshold, or row rule, so it cannot
move a composition figure.)"*

Verified against the live text:

- **Containment:** step `0.1` spans `:1067` (heading) to `:1153`
  (step `0.2` begins `:1154`). The m5 clause sits at `:1088-1091`,
  inside `0.1`'s **Charging rule** paragraph (`:1083-1092`). The
  disclosure is factually correct — KW11.3's measurement reproduces.
- **The why-it-cannot-move-a-figure argument:** the clause reads
  *"charges MEASURED `rec["elapsed_s"]/3600` (§R9 m5 … the TOP-LEVEL
  field, `ncr_earlyln_scale.py:302`, never the nested
  `rec["train"]["elapsed_s"]` at `:202`…) PLUS a STARTUP ALLOWANCE
  `s=0.0053`"*. The formula, the threshold set, the row-selection rule
  and the `charged_ceiling(arm)` table are all textually unchanged;
  only the provenance of one field is disambiguated. The claim
  "renames WHICH field is read … without altering any arithmetic,
  threshold, or row rule" is exactly what the text does. ✓

### 2.3 n3 (KW11.4) — the scripts are committed, and they are the originals

Seven scripts now live at `matrix-thinking/kwall_suites/`. I compared
each against the surviving scratchpad originals:

| Script | `md5` | vs scratchpad original |
|---|---|---|
| `r9rev_vcheck.py` | `93d7c5f75f154552ac5619822292faeb` | **IDENTICAL** |
| `r9rev_payloads.py` | `9c3291165fddbe4719880ce7b3128aed` | **IDENTICAL** |
| `r9rev_drive.py` | `da34358d044aba53515fec8d1655c318` | **IDENTICAL** |
| `r10_vcheck.py` | `d8dc77946d3d2ee74d428f3751a214a4` | **IDENTICAL** |
| `r10_payloads.py` | `69d67dfeb307aa766c789b16f9fde12b` | **IDENTICAL** |
| `r10_probes.py` | `ae70c861f57bec52bf2a3f35df7007de` | **IDENTICAL** |
| `r10_l6fix.py` | `5b372af2f9092ff803431a05f87a68a7` | **IDENTICAL** |

(Originals timestamped `Aug 11 23:32–23:51`, i.e. written by the R9-rev
and R10 agents; the commit is `Aug 12 00:00:15`. These are the files
those rounds ran, not re-typings.)

**Re-executed as shipped**, and the figures match the recorded
results:

`r9rev_drive.py` vs `§R9`'s "Suite figures" section (`:5102-5145`):

| §R9 claim | Re-executed |
|---|---|
| 19 payloads reconstructed, **0/19 mismatches** | `Expectation mismatches: 0/19` ✓ |
| "Delta table: EXACTLY SIX flips — `{B1,B1',B2,B2',B3-NEG,B4}`", set-equality confirmed | `Total flips: 6` … `Flip set matches expected exactly: True` ✓ |
| `B3-AMENDED` traces `PASS(OLD)→PASS(NEW)`, a non-flip | `B3-AMENDED verdict OLD=PASS NEW=PASS` ✓ |
| `L6` (M1): `PASS`, `[]` | `L6 NEW verdict: PASS failures=[]` ✓ |
| `A6` fails on exactly `['EB J4: …not <=0.50']`; `A6'` passes `[]`; literal `0.9296` trips U8 on both, flipping `A6'` | reproduced verbatim ✓ |
| `D2/D2'` flip on `U7-otherwise`; `D1/D1'` flip on `strict-m7-J1a/J1b` | reproduced verbatim ✓ |

`r10_payloads.py` / `r10_probes.py` vs `NCR_KWALL_ATTACK_R10.md`
§2.1's 24-row table: **every OLD/NEW verdict and every NEW
failure-reason list reproduces**, `MISMATCHES vs stated expectation:
NONE`; flips over the 24 = the six, with `A6'-literal0.9296` and
`L6-literal0.8571` flipping as *rounding variants* outside the 24
exactly as R10's prose separately reports.

**n3 is discharged at a higher standard than either discharge option
R10 offered** (commit *or* disclose the deferral): the scripts are
committed, are provably the originals, and still reproduce their
recorded figures. The "no durable `.py`" fragility flagged since R9
is closed. Charter item (i) is satisfied in advance of the first
dispatch rather than deferred to it.

---

## §3 SCOPE 3 — INTEGRITY

### 3.1 Frozen zone — byte-identical

| Commit | Anchor (`## §A1-ADJUDICATION…`) | Lines | `md5` |
|---|---|---|---|
| `3d339bf` | `3446` | `1179` | `3805e7dac8893f272f51fb62210e28be` |
| `e7d29f0` | `3462` | `1179` | `3805e7dac8893f272f51fb62210e28be` |

**Byte-identical, by content anchor, not by line number** — the
declared hash reproduces on both sides. The `+16` anchor shift is
exactly the live-body growth accounted for below.

### 3.2 Byte accounting across the whole file

| Region | `3d339bf` | `e7d29f0` | Contact |
|---|---|---|---|
| Live body (`1 … anchor−1`) | `3445` lines, `md5 c87ca924b24e1c6943e3cb57afe4a7e0` (= `§R9`'s own table row, reproduced) | `3461` lines, `md5 862b63c3d6122e066c10b3ccc46f19a4` | **+16** = n1 (`+5`) + N1 (`+11`) |
| Frozen zone (`1179`) | `3446..4624` | `3462..4640` | **none** (`md5` above) |
| Tail (frozen-end+1 … `§R9` heading−1) | `4625..5057`, `433` lines, `md5 70aecee3ab27f9fb8fd1c7bc56558f86` | `4641..5073`, `433` lines, **same `md5`** | **none** |
| `§R9` block | `136` lines | `144` lines | n2 only (`+5`), plus `+3` trailing separator lines ahead of the appended block |
| `§A10` block | — | `5218..5264` | pure append (`+50`) |
| Whole file | `5193` | `5264` | `+71 = 16 + 5 + 50` ✓ |

### 3.3 Diff inventory vs §A10's process note — EXACT

`git diff 3d339bf e7d29f0` touches exactly:

| Path | Status | Content |
|---|---|---|
| `matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md` | M | 4 hunks: `@@ -2547 +2547 @@` (n1), `@@ -2598 +2603 @@` (N1), `@@ -5117 +5133 @@` (n2, inside `§R9`), `@@ -5191 +5212 @@` (`§A10`, pure append) |
| `matrix-thinking/NCR_KWALL_ATTACK_R10.md` | A | the R10 report |
| `matrix-thinking/kwall_suites/{r9rev_vcheck,r9rev_payloads,r9rev_drive,r10_vcheck,r10_payloads,r10_probes,r10_l6fix}.py` | A | n3 |
| `STATE.md`, `EXPERIMENT_LOG.md` | M | one tick / one entry each |

**Nothing else.** Restricting the diff to the live body gives exactly
6 change groups at exactly **two contiguous sites** — `:2550` (n1) and
`:2601-2617` (N1) — and no other live-body line is touched anywhere.
The `§R9` block's only change is the n2 hunk (diffed in isolation:
one replacement, nothing else). The frozen zone and the entire
`§R7`-remainder / `§A8` / `§R8` / `§A9` tail are byte-identical.
**§A10's process note is accurate.**

### 3.4 `STATE.md` / `EXPERIMENT_LOG.md`

Both entries were read in full. They state the R10 verdict
(`0F/1M/3m`), name KW11.1 and its `1.2210` bound correctly, record
charter addition (j)'s adoption, describe option (A) correctly
(including "spec and history now identical"), disclose the
coordinator-implemented / implementer≠verifier split, and state the
round-11 scope and the on-CLEAR release consequence. No claim in
either exceeds what R10 and §A10 support. ✓

---

## §4 OBSERVATIONS (all non-blocking; none release-unsound)

These are recorded because the charter asks for anything visible
without expanding scope, classed honestly. **None of them is a
finding**: no figure, clause, verdict, bound, or build artifact is
affected by any of them, and each statement below was verified by
execution rather than argued.

### OBS-1 — the bullet's *pair partition* is not the partition a sequential wave produces (the *figures* are)

Under §4's contract the retry is dispatched immediately after a
crashed attempt 1 whenever `realized < 12.00` (`:925-930`,
`:985-989`), so in a 12-charged-attempt all-crash wave the pair
partition is forced to `{5 pairs crashed twice, 2 pairs crashed once
with a 0.0 GATE-REFUSED attempt-2 row, 5 pairs refused at attempt 1}`,
not the `{10 single, 1 double, 1 refused}` the bullet narrates. R10's
§1.1 anticipated the neighbouring objection and answered it correctly
for row *existence* (*"even were an attempt-2 GATE-REFUSED row
mandatory, it carries `elapsed_h=0.0` and changes no figure and no
verdict"*); the partition itself is the residue of that argument.

**Why it is not a finding**, each point executed:

1. Every figure the fixture declares — and every clause any of them
   feeds — is identical under both partitions; the sequential
   trajectory hits all five exactly (§1.6).
2. All three row-sets `PASS` with `[]`: the bullet's as written; the
   bullet's plus ten zero-cost `GATE-REFUSED` attempt-2 rows; and the
   literal 19-row `5/2/5` trajectory.
3. Charter item (j), as adopted, is a **per-row** `elapsed_h`
   criterion, which this payload satisfies with margin and zero
   `COMPLETED` rows.
4. The property is pre-existing and class-wide across payloads this
   round is forbidden to re-open — `A6`'s certified composition
   (`:2733-2744`, R8 K4, re-certified by R9 and R10) has the same
   shape with 9 single-attempt crashed pairs. Treating it as a defect
   would reopen settled, twice-certified material on a question that
   moves no number.

*If* the coordinator ever wants this class closed by construction, the
lever is a strengthening of item (j) from a per-row check to a
trajectory replay of the gate rules over the fixture's rows — a build
convenience, not a design correction, and one that would require
re-narrating settled payloads. **Recorded, not recommended for this
release.**

### OBS-2 — `CRASHED` vs `CRASHED-RECOVERED` in one clause of the new bullet

`:2616-2618` reads *"1 pair `CRASHED` on BOTH attempts
(`ceiling_charged=true` on each row…)"*. `CRASHED` is a valid member
of the 6-value `attempts[].status` enum (`:1548-1549`, `:1600`), but
it denotes a **live** crash, whose `elapsed_h` is the orchestrator's
own measured wall-clock (`:1902-1912` explicitly groups `CRASHED` with
the *returning* attempts) — and `:1010` defines `ceiling_charged=true`
as meaning the value is a gate-admitted ceiling *rather than* a
measurement, naming `CRASHED-RECOVERED` and reconstruction rows. The
precise literal for a `ceiling_charged=true` row at exactly `1.20` is
`CRASHED-RECOVERED`, which is what A6's sibling bullet (`:2735`) and
the committed fixture (`r10_l6fix.py` candidate A) both use.

Executed: relabelling that pair — or all twelve ceiling-charged rows —
to `CRASHED` yields `PASS []` unchanged; no `validity_check` clause
reads either literal, and both are schema-valid, so charter items (f)
and (j) both pass either way. The build cannot be misled: the
committed fixture is unambiguous. **Discharge (optional, one word):
`CRASHED` → `CRASHED-RECOVERED` at `:2616`.**

### OBS-3 — the document's STATUS line is stale

`:3` still reads **"STATUS: DRAFT-R9 — POST-AUDIT-9, AWAITING NARROW
AUDIT ROUND 10"**. The document is post-audit-10 with the N1/n1–n3
layer applied and round 11 dispatched. (Rev-9 did update this line;
the §A10 layer did not — consistent with §A10's "no other live-body
contact", but now factually stale on the round number.) The clause
"not build-released, not queue-eligible" was correct at `e7d29f0`.
**Discharge: the release edit rewrites this line anyway.**

### OBS-4 — the §A10 layer publishes no MD5/line table of its own

Every prior revision published one; §A10 does not. In practice the
integrity of this layer is established more strongly by the commit
boundary itself (§3.2/§3.3 above are git-verifiable by anyone), and
`§R9`'s table remains correct *as a snapshot of Rev-9* under the
established fixed-point convention. **Non-blocking**; if the
coordinator wants the convention kept unbroken, the release edit can
carry the three numbers from §3.2.

### OBS-5 — a cosmetic gap in `r10_payloads.py`'s own self-check line

Its `set-equality vs claimed six-flip set` line filters one rounding-
variant family (`*literal0.8571`) but not the other
(`A6'-literal0.9296`), so the shipped script prints
`False | extra: [A6'-literal0.9296]` where R10's prose correctly reads
*"set equality over the 24 CONFIRMED"* and reports the variant flip
separately in the next paragraph. The **data** the script prints is
right and matches R10's table row-for-row; only its one-line summary
under-filters. Script hygiene, no design consequence. (Left as-is
deliberately: the committed scripts are byte-identical to what R9-rev
and R10 ran, which is worth more than a tidier summary line.)

---

## §5 GATE SUMMARY

| Item | Status |
|---|---|
| N1 is R10 §4 option (A) exactly | **PASS** (§1.1) |
| N1's figures re-derived independently, rational arithmetic | **PASS** — exact, U3 and U8 residuals both `0` (§1.2) |
| N1's exact-quotient/rounding discipline | **PASS** — `1.0000` **is** the exact quotient (§1.2) |
| Every row producible under `:460`/`:977-984`/`:1091`/`:1902-1912`/`:2141-2156` | **PASS** — max row `1.20` vs cap `1.2210`; zero `COMPLETED` rows (§1.3) |
| KW11.1's contradiction with τ / `R_N ≤ 15.0157` / `T ≤ 15.3737` | **RETIRED** (§1.3) |
| N1 re-executed to `PASS`, failure-reason list `[]` | **PASS** — two independent transcriptions (§1.4) |
| Negative controls on N1 have teeth | **PASS** — 7/7 (§1.4) |
| In-bullet clause trace correct as written | **PASS** — clause by clause vs `:2319-2527` (§1.5) |
| Declared figures trajectory-reachable under §4's contract | **PASS** — exact hit on all five (§1.6) |
| n1 — sentence scoped to the 24; four D-probe flips named + correctly attributed | **PASS**, re-executed (§2.1) |
| n2 — m5's `0.1` textual contact disclosed; why-it-cannot-move-a-figure accurate | **PASS**, containment re-measured (§2.2) |
| n3 — suites committed, byte-identical to the originals, figures reproduce | **PASS** (§2.3) |
| Frozen zone `1179` lines, `md5 3805e7dac…`, content-anchored, both commits | **PASS** (§3.1) |
| Live-body / tail / `§R9` / `§A10` byte accounting closes (`+71`) | **PASS** (§3.2) |
| §A10 process note accurate; no other live-body contact | **PASS** — exactly 2 live-body sites (§3.3) |
| `STATE.md` / `EXPERIMENT_LOG.md` accurate, no overclaim | **PASS** (§3.4) |
| M1 arithmetic/execution, m1, m2–m7, boundaries 1 & 3, K1–K7, both suites | **NOT RE-OPENED** (settled by R10, per charter) |

**Nothing in the science, the budget derivation, the reconstruction
procedure, the band logic, the trigger, or the report schema is
impugned by this round — through eleven rounds, none has ever been.**
No clause of `validity_check` needs to change. No further revision is
required.

---

## §6 VERDICT — **CLEAR**

**The design is BUILD-RELEASABLE.** The §8 build charter — as
restated in `NCR_KWALL_ATTACK_R10.md` §8, **including new item (j)
(every in-text payload used as a build fixture must be PRODUCIBLE:
each row's `elapsed_h` within `charged_ceiling(arm) + τ + s` for a
`COMPLETED` row, or exactly `charged_ceiling(arm)` for a
`ceiling_charged` row, asserted by the fixture loader rather than by
reading)** — is hereby **RELEASED**.

The charter now in force, in full:

1. R5's conditional build-release checklist, in full.
2. R6's five additions, including every negative test RUN TO
   COMPLETION.
3. The 3 micro-smokes (K=26/28/30) pass before queue-eligibility.
4. R7's five additions (a)–(e), unchanged.
5. R8's **(f)** schema validation of every emitted `attempts[]` row
   including `attempt_n`, with a `bootstrap_n>2` positive fixture;
   **(g)** the build recomputes `charged_vs_measured` from
   `ledger.attempts` and asserts equality before
   `orchestrator_report.json` is written.
6. R9's **(h)** the payload suite wired as a build-stage unit test
   asserts the **failure-reason list**, not merely the verdict, for
   every forced-fail negative; **(i)** the transcription scripts are
   committed to the repo alongside the design before the first
   dispatch — **already satisfied** at
   `matrix-thinking/kwall_suites/` (§2.3).
7. R10's **(j)** fixture producibility, as quoted above.

**What the build ceremony now covers:**

- `validity_check` (U1–U8 plus the four per-`run_status` branches),
  the recovery/reconstruction procedure (`G1`/`G2`/`0.0`/`0.1`/`0.2`),
  the trigger and band procedures, the orchestrator report schema, and
  the `≤15.00`-gated dispatch loop with its `T ≤ 15.3737` GPU-h
  honest bound;
- the **12-cell primary sweep — K∈{26,28,30} × 4 seeds — as a
  recovery-leg wall characterization on the live K=24 recipe**, plus
  the conditional 160K arm, queue-eligible behind the three
  micro-smokes;
- under the **`15.50` GPU-h declared pool ceiling**, one flat pool
  spec (the orchestrator), one GPU, strictly sequential cells.

The two one-line items in §4 (OBS-2's `CRASHED` → `CRASHED-RECOVERED`,
OBS-3's stale STATUS line) are offered for the release edit. Neither
gates the release, and neither requires another audit round.

---

## §7 SCRIPTS

This round's own, in the session scratchpad:

- `r11_vcheck.py` — independent transcription of `validity_check` from
  `e7d29f0`'s `:2319-2527`, rational arithmetic, document-order
  clauses.
- `r11_l6.py` — the restated `L6` rebuilt from the bullet's prose;
  figure re-derivation; per-row producibility audit against the four
  charging sites; execution under two band/resolution framings; seven
  negative controls; the strictly-sequential orchestrator simulation.
- `r11_variants.py` — the two alternative row-sets (`+10` zero-cost
  `GATE-REFUSED` rows; the literal `5/2/5` trajectory).
- `r11_label.py` — status-literal sensitivity (`CRASHED` vs
  `CRASHED-RECOVERED`).

Durable suites: the seven committed at `matrix-thinking/kwall_suites/`,
re-executed as shipped this round (§2.3). All harnesses are CPU-only
logic checks; no training, no GPU.

---

*Round 11, `NCR_KWALL_ATTACK_R11.md`. Verdict: **CLEAR** —
0 FATAL / 0 MAJOR / 0 MINOR, 5 non-blocking observations. The design
is BUILD-RELEASABLE and the §8 build charter including item (j) is
RELEASED. The coordinator adjudicates; this report writes nothing
else in the repo.*
