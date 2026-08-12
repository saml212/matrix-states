# NCR K-WALL — NARROW AUDIT ROUND 9 (scope: K1–K7 verification + both suites re-executed)

**STATUS: VERDICT = REV-REQUIRED.** 0 FATAL / 1 MAJOR / 7 MINOR.
Target: `matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md`, header
verified as **"DRAFT-R8 — POST-AUDIT-8, AWAITING NARROW AUDIT ROUND 9
(not build-released, not queue-eligible)"** (`:3-4`) — matches the
expected header exactly.

**Artifact pinning.** Design at commit `671d83a`, working tree CLEAN for
this file (`git status --porcelain` empty), on-disk md5
`b70242d9e7a2f9131c4dc2e25d7d756f`, `4919` lines. Every diff below is
pinned to the explicit pair `ad2bf48 → 671d83a` (`ad2bf48` = the
pre-Rev-8 file, byte-identical in the live-body range to Rev-7's
`7a0917d`, confirmed: both hash `55ba3e9a9289e10f5e7fde5864c21970` over
`1..3116`), never to a moving `HEAD~1`.

**Headline.** **The FATAL is genuinely dead.** K1's two-sided fix (§5's
4/4 precondition + universal assertion 7's mirror clause) was verified
not by reading but by an independent re-transcription and by tracing
every conditional-arm shape the scope named — 3/4-throttled, 4/4-
completed, 0/4-refused (both `launched` values), the $0 `K_trig=32`
branch, a ledger-count violation, and a non-throttle (crash) shortfall.
All land where the design says. K2/K3/K4/K6 are discharged by execution:
no zero-cost row combination reaches `completed/` at any of the four K2
sites; universal assertion 8's division guard is in the final text,
correct, and its disclosed blind spot is provably unreachable; the
rebuilt A6 dies on **exactly** J4's clause with U1/U2/U3/U8 and all
three base clauses passing first; `attempt_n` is consistent across every
site and both reachable bootstrap statuses. K5's MD5 reproduces
to the digit, the frozen zone is byte-identical, and Rev-8's own MD5
table is self-consistent in all four rows.

What survives is **not a soundness objection to any K disposition**. It
is one MAJOR bookkeeping-with-teeth defect — K3's own new universal
assertion 8 falsifies an in-text payload (`L6`) that the same section
still claims PASSES, and Rev-8's harness silently ran a *different* L6
while disclosing only A6/A6' as rebuilt — plus seven one-clause
precision/coverage residues. None of them can lose a legitimate run's
spend. A Rev-9 that fixes them is mechanical; see §6.

Scope discipline: this report addresses ONLY K1–K7, the two re-run
suites, the integrity block, and the one disclosed settled-section
contact. Everything the R8 report's §5 GATE SUMMARY named PASS (J2's
producer fix, J3 in full, J6's arithmetic/`s`/code citations/Class-2
cap, the frozen-zone identity, the three `§R7`-disclosed contacts) was
not re-opened, except where the 200-state re-run incidentally
re-verifies it (it does, exactly).

---

## §0 METHOD

Everything below is executed, hashed, or diffed. Nothing is asserted.
Rev-8's own scripts were re-run **and** independently cross-checked —
per this round's charge to re-derive rather than trust cached results.

- **`validity_check`:** all 8 universal assertions and all 5
  per-`run_status` branches transcribed **independently** from the
  amended text (`:2296-2509`) into `scratchpad/r9_indep.py`, written
  without using `vcheck_r8_rev.py`'s control flow as a template. Every
  one of Rev-8's 24 payloads was run through BOTH transcriptions:
  **0 / 24 disagreements** — Rev-8's transcription is faithful to the
  amended text. Rev-8's own differential driver was then re-run verbatim
  (`drive_vcheck_r8_rev.py`).
- **This round's own payloads:** 7 further constructions
  (`scratchpad/r9_probe.py` + §3 of `r9_indep.py`) — C1–C6 (the
  conditional-arm shapes the scope names) and D1–D5 (residual-hole
  probes an orchestrator BUG, not a forger, can produce).
- **Reconstruction:** `recon_r8.py` re-run verbatim; its validity as a
  transcription of the AMENDED text re-established by diffing the
  reconstruction range — Rev-8's four edits there (`:1092`, `:1130-1140`,
  `:1168-1174`, `:1246-1248`) are citation/disclosure text only and
  change no rule.
- **Arithmetic:** A6's composition re-derived independently in exact
  decimal; the `0.9296` rounding tested against U8's `1e-6` tolerance;
  the `ccgh ∈ {1.20k + 2.32m}` quantization argument checked against the
  design's own `charged_ceiling(arm)` definition (`:971`, `:1072-1073`,
  `:2019`).
- **Integrity:** `md5`, `wc -l`, byte-range `diff` against `git show
  ad2bf48:` and `git show 7a0917d:`.

---

## §1 SCOPE 1 — K1–K7 DISCHARGE VERIFICATION

### 1.1 K1 (the FATAL) — **FULLY DISCHARGED**, verified by tracing every path

Both halves are in the final text: §5's new paragraph (`:2983-3004`)
and universal assertion 7's mirror clause (`:2374-2396`), with the
"hypothetical future" scope note retracted at `:2374-2379`. The
retraction is correct on the facts — G4 does define the case, twice
(`:1701-1712`, sub-cases (ii)/(iii)).

Every path traced, by execution, against my independent transcription:

| Case | Report | Verdict | Clause that governs |
|---|---|---|---|
| **3/4 throttled** (C1 / Rev-8's B3-AMENDED) | `COMPLETE-DEGRADED`, 12/12 primary, 3 conditional canonical, band `null` | **PASS** | mirror: `3 == 3` ✓, `3 < 4` ✓ |
| 3/4 throttled, band still claimed (B3-OLD-STYLE) | band `SLOW-CONVERGENCE-AT-160K` | **FAIL** | U7 clause (a): `3 ≠ 4` |
| **4/4 completed** (C2) | `COMPLETE`, band set, 4 canonical, 4 ledger rows | **PASS** | U7 clause (a) |
| 4/4 completed, band `null` (B3-NEG) | band `null` | **FAIL** | mirror's `n_cond_canon < 4` half |
| **0/4 refused, `launched=true`** (C3) | 4 conditional `GATE-REFUSED` rows, 0 canonical, band `null` | **PASS** | mirror: `0 == 0` ✓, `0 < 4` ✓ |
| **0/4 refused, `launched=false`** (C3') | same disk state | **PASS** | "Otherwise" arm, no constraint |
| 3/4 with one ledger row missing (C5) | 3 canonical, 2 conditional `COMPLETED` rows | **FAIL** | mirror count `3 != 2` |
| 3/4 where the 4th **CRASHED** (not throttled) (D4) | `COMPLETE`, band `null` | **PASS** | §5's headline sentence covers the non-throttle shortfall too — the "If the conditional arm is THROTTLED" sentence alone would not have |

The G4 sub-case (ii)/(iii) prose was **not** edited this revision, and
does not need to be: it never mentions the qualifier band; the band rule
lives wholly in §5 and U7. Checked by grepping all 31 lines mentioning
`qualifier` in the live body — no site other than §5 mandates a band.

**One residue** (KW10.3, minor): §5's new headline sentence is written
unqualified — "The qualifier band is reported ONLY on 4/4 conditional
completion" — and thereby contradicts the pre-registered $0
`K_trig==32` branch three sentences later (`:3026-3031`: K=32 "is
**`CONFIRMED-WALL-AT-160K`** — reported at $0 incremental cost … on the
exact same footing as a paid `K_trig∈{26,28,30}` cell") and U7's own
retained clause (b). Bounded: both readings pass `validity_check` (C4
and C4' both PASS), so nothing is routed to `failed/` — only the
reported CONTENT of that branch is now ambiguous.

### 1.2 K2 (`>=1` distinct `COMPLETED` primary pair) — **DISCHARGED at all four sites**

All four edit sites present and identical in wording: G4 `COMPLETE`
prose `:1673-1681`, G4 `COMPLETE-DEGRADED` prose `:1727-1736`,
`validity_check` `COMPLETE`/OTHERWISE `:2443-2448`, `validity_check`
`COMPLETE-DEGRADED` `:2459-2462`.

**Can any zero-cost row combination still satisfy it? No.** The clause
is satisfiable by one `COMPLETED` row carrying `elapsed_h=0.0` (probe
D3), but at every site the clause sits beside the count clause
(`canonical count == distinct COMPLETED primary rows`), so passing
requires **≥1 genuine canonical `COMPLETED` file on disk** — which no
zero-cost path produces (a hard-gate comparison bug that refuses every
cell yields 0 canonical). Executed: B1 FAILS on exactly
`COMPLETE/otherwise K2`, B1' on exactly `CD K2`, L2 (11 real
completions) still PASSES. KW9.2's hole is closed.

Optional hardening (not a finding): nothing asserts a `COMPLETED` row's
`elapsed_h ≥ s`; the charging rule (`:1077-1079`) implies it.

### 1.3 K3 (universal assertion 8) — **DISCHARGED; the guard is in the final text and is correct**

Assertion 8 is at `:2397-2418`. The division guard Rev-8 disclosed
adding **is present** and correctly scoped: `ccgh` is recomputed and
asserted **unconditionally**; only the fraction half is skipped when
`realized_gpu_h_final == 0`.

The disclosed blind spot was probed directly, not accepted on the
argument. **D5** — `COMPLETE`, `realized=0.0`, fraction mis-declared
`0.99` — PASSES, exactly as disclosed. It is inert: `ceiling_charged_
fraction` has exactly one consumer, the `EXHAUSTED-BUDGET*` dichotomy
(G4 prose `:1779`/`:1812`, `validity_check` `:2479`/`:2487`), gated
behind `realized > 13.80`; the only other mention is a build-stage
red-team item exercising that same dichotomy (`:3231`). Verified by
grepping all 17 lines carrying the field. The guard costs nothing
reachable, as claimed.

B2 FAILS on exactly `U8: ceiling_charged_fraction 0.2000 != recomputed
0.9296`, before any per-`run_status` branch.

**One residue** (KW10.2, minor): the in-text trace of B2 (`:2705-2720`)
describes the failure as *"the report's self-declared `0.20` fraction
implies `ceiling_charged_gpu_h ≈ 2.84`, and `abs(2.84 − 13.20) > 1e-6`"*
— that is the **B2'** variant's mechanism (a report declaring
`ccgh=2.84`). The actual B2 payload declares `ccgh=13.20` and dies on
the fraction half. Both FAIL on U8; only the narration is crossed.

### 1.4 K4 (the A6 negative test) — **arithmetic re-derived independently, EXACT; teeth confirmed**

Re-derived from the design's own composition (`:2639-2674`), not from
the script:

```
9 pairs × 1 row  × 1.20 (CRASHED-RECOVERED, ceiling_charged)      = 10.80
1 pair  × 2 rows × 1.20 (retried, crashed again, ceiling_charged) =  2.40
1 pair  × 1 row  × 1.00 (COMPLETED, measured, not charged)        =  1.00
1 pair  × 1 row  × 0.00 (GATE-REFUSED)                            =  0.00
pairs 9+1+1+1 = 12 ✓   rows 9+2+1+1 = 13 ✓   CRASHED-RECOVERED rows = 11 ✓
realized = 14.20 ✓   ceiling_charged_gpu_h = 13.20 ✓   13.20/14.20 = 0.929577…
base clauses: 14.20 > 13.80 ✓ ; canonical 1 < 12 ✓ ; primary attempt_n==1 GATE-REFUSED ✓
```

Executed against my independent transcription: A6's failure list is
**`['EB/J4-frac']` and nothing else** — U1/U2/U3/U8 and all three base
clauses pass first. A6' (same ledger, `-SUSPECT-OVERCHARGE`) PASSES with
an empty failure list. The forced-fail negative now has teeth on exactly
the clause it exists to exercise. KW9.4's discharge condition is met.

**One residue** (KW10.5, minor): the in-text payload states the fraction
as `0.9296`. Transcribed literally, `|0.9296 − 13.20/14.20| = 2.254e-05
> 1e-6`, so A6 additionally trips U8 (dying on two clauses, one of them
the wrong one) and **A6' — a claimed PASS — flips to FAIL**. Executed
both ways.

### 1.5 K5 (the `§R7` MD5 correction) — **REPRODUCES**

```
git show 7a0917d:<design> | sed -n '1,3116p' | md5  →  55ba3e9a9289e10f5e7fde5864c21970
git show ad2bf48:<design> | sed -n '1,3116p' | md5  →  55ba3e9a9289e10f5e7fde5864c21970
```

Identical to the figure K5 installed at `:4552`. The line range is
correct by the section's own convention (`## §A1-ADJUDICATION` sits at
`:3117` in both commits). The stale `1f93fa4…` is gone. KW9.5 closed.

### 1.6 K6 (`attempt_n` schema/emitter consistency) — **DISCHARGED across ALL sites**

All 33 lines mentioning `attempt_n` in the live body inspected. Schema widened
at `:1530-1535` to `"attempt_n":int` with the reconstruction-label note;
the `attempts[].status` table gains the note on exactly the two rows a
bootstrap row can produce (`COMPLETED` `:1586`, `CRASHED-RECOVERED`
`:1589`). **Both rows are genuinely reachable** — instrumented tally over
the 200-state walk: of the 72 `bootstrap_n>2` states, **24 are
`COMPLETED` and 48 are `CRASHED-RECOVERED`**, max `attempt_n = 3`. No
other site constrains `attempt_n` to `{1,2}`: `:2474` reads
`attempt_n==1` (a base clause, correct), `:1325-1334` is the dispatch
numbering rule whose `max+1` is safe because every bootstrap cell derives
TERMINAL (re-confirmed: 0/200 non-terminal).

### 1.7 K7 (the five MINORs) — four discharged at their named sites, two residues

- **KW9.7** — closed by K1's mirror clause; B4 FAILS on exactly
  `U7 mirror: conditional canonical count 4 != ledger COMPLETED count 0`.
  **Residue KW10.4:** the clause is keyed on the report's own
  `conditional` block, so the same paid-arm-invisible outcome survives
  when the block is `null` or declares `launched:false` (probes D2, D2'
  both PASS with 4 conditional canonical files on disk).
- **KW9.8** — retracted correctly at `:1130-1140`; the replacement text
  is honest ("no term and no multiplicity cap in `T`'s expression … an
  honest bound on spend for which disk EVIDENCE survives"). Discharged.
- **KW9.9** — the `§9(d)` citation now names `NCR_KWALL_ATTACK_R7.md`
  §9 item 4(d) (`:1494-1499`). **Verified against the source:** R7 §9
  item 4(d) is *"the build asserts the conditional canonical directory
  is disjoint from the primary one … before the first conditional
  dispatch"* — exactly the assertion the citation site delegates. R8 §8
  item 4 does restate R7's (a)–(e) unchanged. Discharged.
- **KW9.10** — both pseudocode copies normalised; all 5 return sites are
  4-tuples with `resolution` at a fixed index, and the one consumer
  (`:655`) was re-keyed to `result[0]`. The `tie-break-min` payload now
  sets `resolution_detail` (`:2554`). **Residue KW10.7:** slot 4
  (`diag`) is semantically overloaded.
- **KW9.11** — all three named sites now cite `rec["elapsed_s"]`
  top-level with `:302` vs `:202` contrasted (`:1092`, `:1168-1174`,
  `:1246-1248`). Discharged at the named sites. **Residue KW10.6:** two
  further sites still say it unqualified.

---

## §2 SCOPE 2 — BOTH SUITES RE-EXECUTED

### 2.1 24-payload differential suite — reproduces, but the delta claim does not

Re-ran `drive_vcheck_r8_rev.py` verbatim: **OLD 24/24, NEW 24/24**, as
claimed. Independent cross-check: my own transcription agrees with
`validity_check_NEW` on **all 24** payloads — the transcription is
faithful, which is the load-bearing question for a suite that certifies
the amended text.

**But the behavioural delta is six payloads, not five, and the payload
the design names as the delta is not one.** Executed
(OLD verdict ≠ NEW verdict):

| Payload | OLD | NEW | Named in `§R8` / the live body? |
|---|---|---|---|
| B1 | PASS | FAIL | yes |
| B1' | PASS | FAIL | yes |
| B2 | PASS | FAIL | yes |
| **B2'** | **PASS** | **FAIL** | **no** |
| **B3-NEG** | **PASS** | **FAIL** | **no** |
| B4 | PASS | FAIL | yes |
| B3-AMENDED | PASS | PASS | named as "newly reachable as a genuine PASS" — **it does not flip** |

The design's live body asserts (`:2505-2509`) that the 4 flips plus
B3-AMENDED are *"the ONLY behavioural deltas — nothing else regressed."*
Executed, that sentence is false as written. Its **substance** holds:
both unnamed deltas are the new clauses correctly catching adversarial
payloads, and nothing legitimate regressed. The `24/24` tally hides the
sixth because the driver marks B3-NEG's OLD expectation
`"N/A(old had no rule)"` and auto-counts it as a match. See KW10.2.

### 2.2 200-state composition — reproduces to the digit

```
guard  states  orphans  abort-trips  boot n>2  max rows/cell
OLD    200     30       6            0         2
NEW    200     0        0            72        3
Max Class-2 rows/cell (NEW) = 2   (H3's ≤32 = 16×2 cap holds)
Non-terminal bootstrap_n>2 states = 0
```

Every figure matches the design's claim and the R8 audit's own
execution. The OLD-guard abort-trip table reproduces case-for-case (3
compositions × 2 arms, all resuming at `2`). Rev-8's claim that K1–K7 do
not touch the reconstruction procedure is verified by diff, not accepted:
the four edits inside `:1032-1250` are citation and disclosure text only.

---

## §3 SCOPE 3 — INTEGRITY

| Check | Result |
|---|---|
| Frozen zone (`§A1`→pre-`§R7`), current file `:3352-4530` | `3805e7dac8893f272f51fb62210e28be`, **1179 lines** |
| Same range at `ad2bf48` (`:3117-4295`) | `3805e7dac8893f272f51fb62210e28be`, 1179 lines — **byte-identical** |
| Same range at `7a0917d` | `3805e7dac8893f272f51fb62210e28be` — **byte-identical** |
| Live body before (`ad2bf48`, `1..3116`) | `55ba3e9a9289e10f5e7fde5864c21970` ✓ matches `§R8`'s row |
| Live body after (current, `1..3351`) | `225eab951036e1575a2c5a317a760f4e` ✓ matches `§R8`'s row |
| Whole file before Rev-8 (`ad2bf48`, 4509 lines) | `3cb6076062b40b578ff6f40b76f5b3d0` ✓ matches `§R8`'s row |
| `§R7`+`§A8` tail block, before vs after | 214 lines both sides; `diff` = **exactly one line**, the disclosed K5 MD5 row |
| Line arithmetic | `4509 = 3116+1179+214` ✓ ; `3351−3116 = 235` ✓ ; `3351+1179+214 = 4744` ✓ ; `4744+175 (§R8) = 4919` = `wc -l` ✓ |
| No whole-file "after" hash stated | correct, per the standing fixed-point convention |

**All four rows of Rev-8's own MD5 table are self-consistent and
reproduce.** The near-miss it disclosed (the "after" figure invalidated
by a later header edit) is genuinely resolved: the stated hash covers the
`DRAFT-R8` header line, verified by recomputation.

**Settled-section contacts: exactly one, and it is disclosed.** The
byte-range diff of the `§R7`+`§A8` tail returns a single changed line —
K5's authorized target. Nothing else in the frozen or tail range moved.
The live-body diff (`ad2bf48` → current) is 28 hunks, every one of which
maps to a K1–K7 disposition or the status header; no undisclosed edit.

---

## §4 FINDINGS

### KW10.1 — MAJOR. K3's own new universal assertion 8 falsifies the in-text `L6` payload, which still claims PASS; Rev-8's suite silently ran a different L6 and disclosed only A6/A6' as rebuilt.

**Quote, the in-text payload** (`:2544-2545`, **not edited this
revision**): *"- `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE`, same disk state
plus `ceiling_charged_fraction=0.71` — PASSES."* ("Same disk state" =
L5's, `:2541-2543`: `realized=14.55`, 9 canonical primaries, a primary
first-attempt `GATE-REFUSED`.)

**Quote, what `§R8` says it ran:** *"**24 payloads**: the design's
original 16 (L1–L7, L7', A1–A7 **with A6/A6' rebuilt** to the §R8 K4
composition) + this revision's 8 adversarial extensions."*

**Evidence, executed.** Under K3, `ceiling_charged_fraction` is no
longer a free self-report — it must equal `ccgh_recomputed /
realized_gpu_h_final` to `1e-6`. Taking the in-text payload as written
(the R8 audit's own faithful L5/L6 transcription, whose ledger is 9
measured `COMPLETED` rows + 3 `GATE-REFUSED` rows, hence
`ceiling_charged_gpu_h = 0.0`):

```
L5  realized=14.5500  true ccgh=0.0000  declared frac=0.00  ->  PASS  []
L6  realized=14.5500  true ccgh=0.0000  declared frac=0.71  ->  FAIL  ['U8-frac(0.710000!=0.000000)']
```

**And it is not merely under-specified — with `realized` held at the
in-text `14.55`, no ledger can make `0.71` true.** Every
`ceiling_charged==true` row carries exactly `charged_ceiling(arm)`
(`:999` "iff `elapsed_h` is a gate-admitted ceiling"; `:971`,
`:1072-1073`, `:2019` fix it at `1.20` primary / `2.32` conditional), so
`ceiling_charged_gpu_h ∈ {1.20k + 2.32m}` — a set whose every member has
at most two decimal places. `0.71 × 14.55 = 10.3305` is not in it. The
payload becomes satisfiable only by abandoning L5's stated
`realized=14.55` (e.g. `ccgh=10.80`, `realized=15.2113`).

Rev-8's harness did neither: it silently rebuilt L6 with a different
ledger, printing `# L6 composition check: realized=14.4000,
ccgh=14.4000, frac=1.0000` and labelling the payload **"L6 EBSO frac
1.0000."** The disposition table, the residue paragraph and the
`§R8` suite description disclose **only** A6/A6' as rebuilt.

**Why MAJOR, and why not minor.** Three reasons, all load-bearing:

1. **The in-text list is the only durable specification.** Every round's
   `vcheck_*.py`/`recon_*.py` lives in an ephemeral session scratchpad —
   confirmed: `find` over the repo returns **no** `.py` under
   `matrix-thinking/`. A build agent has the prose and nothing else.
2. **The standing charter wires it as a unit test** (R7 §9 item 4(a),
   adopted by `§A7-ADJUDICATION`; R6's "every negative test RUN TO
   COMPLETION"). A claimed-PASS payload that red-fails invites exactly
   the repair KW9.4 warned about — *relaxing the assertion* — and the
   assertion at risk is the one K3 just installed to close KW9.3.
3. **It is KW9.4's own failure class, in the positive direction**, and
   KW9.4 was adopted MAJOR. A payload whose stated numbers cannot close
   gives zero coverage; here it gives negative coverage.

**Discharge condition.** Rebuild L6 in-text the way K4 rebuilt A6: state
its ledger composition explicitly and give a fraction that is the exact
quotient of that ledger (or state that L5/L6's `charged_vs_measured`
block is recomputed from the ledger and drop the bare `0.71`). Update the
suite description to disclose every rebuilt payload. Re-run the list to
completion and print the failure-reason list, not just the verdict.

### KW10.2 — MINOR. The live body's "these five are the ONLY behavioural deltas" is executed-false; six payloads flip and B3-AMENDED is not one of them.

**Quote** (`:2505-2509`): *"confirming the 4 payloads (B1, B1', B2, B4)
that flip PASS→FAIL and the 1 (the legitimate 3/4-throttled report,
listed above) that is now reachable as a genuine PASS are the ONLY
behavioural deltas — nothing else regressed"*.

**Evidence, executed** (§2.1's table): B2' and B3-NEG also flip
PASS→FAIL; B3-AMENDED PASSES under both transcriptions. The `24/24`
tally does not expose the sixth because the driver's `want_OLD` for
B3-NEG is the string `"N/A(old had no rule)"`, which the match test
treats as an automatic pass.

The substance is fine — both unnamed flips are the new clauses catching
adversarial payloads, no legitimate outcome regressed, and "newly
reachable" is defensible for B3-AMENDED at the §5 level (pre-Rev-8, §5's
unconditional "is reported" gave an implementer no way to emit
`band=null`, which is precisely the FATAL). Only the count and the
attribution are wrong. Also crossed: the in-text B2 trace (`:2713-2717`)
narrates B2''s mechanism (`ccgh ≈ 2.84`) rather than B2's own (the
fraction half). **Discharge:** restate as six flips, name B2' and
B3-NEG, and say B3-AMENDED is newly *emittable* (its `validity_check`
verdict is unchanged) — or drop the enumeration.

### KW10.3 — MINOR. K1's unqualified §5 headline contradicts the pre-registered $0 `K_trig==32` branch and U7's own retained clause (b).

**Quote, K1's new headline** (`:2983`): *"**The qualifier band is
reported ONLY on 4/4 conditional completion**"*.
**Quote, §5, 43 lines later** (`:3026-3031`): *"**At `K_trig=32`:** the
qualifier is now read from the SAME ALREADY-ARCHIVED table … so by the
rule above K=32 is **`CONFIRMED-WALL-AT-160K`** — reported at $0
incremental cost … on the exact same footing as a paid
`K_trig∈{26,28,30}` cell."*
**Quote, U7 clause (b)** (`:2368-2371`, retained): *"OR (b)
`conditional["launched"]==False` AND `trigger["K_trig"]==32` (the
$0-branch archive citation…)"*.

The `K_trig=32` branch has **zero** conditional completions by
construction. Executed, both readings are accepted: C4 (band set,
`launched=False`, `K_trig=32`) PASSES via clause (b); C4' (band `null`)
PASSES via the "Otherwise" arm. So no run is lost — the defect is that
the pre-registered CONTENT of the design's own best-case branch is now
ambiguous, and a §5-literal implementer suppresses a band the same
section pre-registers. **Discharge:** scope the headline — "for the PAID
branch (`K_trig∈{26,28,30}`)"; the $0 archive citation reports its band
per U7(b), unchanged.

### KW10.4 — MINOR. The K1 mirror clause is keyed on the report's own `conditional` block, so a paid conditional arm is still invisible when the block is `null` or declares `launched:false`.

**Quote** (`:2393-2396`): *"Otherwise (`conditional is None`, or
`qualifier_band is None` and `launched` is `False` or absent): no
constraint from this assertion (**the conditional arm was never
dispatched** — genuinely absent, not throttled)."*

The parenthetical is an assumption, never a check: nothing reads the
conditional canonical directory on that path. **Executed:**

```
D2   conditional=None,          4 conditional canonical files on disk  ->  PASS
D2'  conditional launched=False, 4 conditional canonical files on disk  ->  PASS
```

Up to `2.32 × 4 = 9.28` GPU-h of real spend, invisible — the identical
outcome KW9.7 named, reached by a different field, and directly against
the mirror clause's own stated guarantee (`:2387-2389`: *"real
conditional spend can never be invisible to the ledger, whether or not a
band is claimed"*). Classed MINOR for parity with KW9.7, which the same
gauntlet classed MINOR. **Discharge:** one clause on the Otherwise arm —
the conditional canonical directory must contain **0** files.

### KW10.5 — MINOR. In-text A6's `0.9296` is a 4-dp display of `13.20/14.20`; transcribed literally it trips U8, and A6' — a claimed PASS — flips to FAIL.

**Quote** (`:2660-2662`): *"`ceiling_charged_fraction= 13.20/14.20=0.9296`"*.
`13.20/14.20 = 0.9295774647887324`; `|0.9296 − that| = 2.254e-05`, and
U8's tolerance is `1e-6`. **Executed:**

```
exact quotient   A6  -> FAIL ['EB/J4-frac']                                   A6' -> PASS []
literal 0.9296   A6  -> FAIL ['U8-frac(0.929600!=0.929577)', 'EB/J4-frac']    A6' -> FAIL ['U8-frac(...)']
```

The expression `13.20/14.20` is present, so a careful implementer
computes the quotient — hence MINOR, not a repeat of KW9.4. But it is
K4's own residue in the class K4 closed, and the charter wires these as
tests. **Discharge:** write "`= 13.20/14.20` exactly (`0.9296` to 4 dp)"
at both A6 and A6'.

### KW10.6 — MINOR. KW9.11's disambiguation is not global: two further sites still say `elapsed_s` unqualified, one of them the charging rule itself.

- `:1077-1079`, the charging rule: *"a row whose `status=="COMPLETED"` …
  charges MEASURED `elapsed_s/3600` PLUS a STARTUP ALLOWANCE"* — the
  single most-read statement of the rule, unqualified.
- `:1451`, the crash-window table's "before copy" row: *"`COMPLETED`
  row, MEASURED `elapsed_h` (**the subprocess's own `elapsed_s`**)"* —
  describing the very PROMOTE branch K7 disambiguated at `:1246`.

KW9.11's whole premise is that an implementer reading that phrase may
take `rec["train"]["elapsed_s"]` and under-charge by the entire
post-train instrument sequence. The three named sites are fixed; the
phrase survives verbatim at two more. **Discharge:** the same
parenthetical at both, or one global sentence in the charging rule.

### KW10.7 — MINOR. KW9.10 traded arity inconsistency for a semantically overloaded slot, and the disclosing comment mis-describes it.

Both `trigger()` copies now return `(K_trig, resolution,
resolution_detail, diag)`. In the **G5 copy** — the live one — `diag`
carries `blocking_K` at `:643` and `band_blocked_K_trig` at `:656`,
which the schema keeps as two **distinct** fields (`:1548-1549`). The
explanatory comment (`:565-566`) says diag *"carries `blocking_K` here /
`band_blocked_K_trig` in the G5 copy below"* — but the G5 copy has both.
Pre-Rev-8 the two were distinguishable by tuple arity; now they are
positionally identical, so an emitter driven by position alone can
populate the wrong schema field. Un-asserted and informational (universal
assertion 6 reads only `resolution`), hence MINOR. **Discharge:** return
a dict, or split `diag` into two slots, or correct the comment.

### KW10.8 — MINOR (PRE-EXISTING, outside the K1–K7 charter, flagged because K2 is a disclosed contact). `COMPLETE`'s STRICT branch carries no ledger clause at all.

J1's two clauses and K2's `>=1` clause live **only** in the OTHERWISE
branch (`:2430-2448`); the strict branch (`:2426-2430`) checks the 12
canonical files and nothing else. **Executed:**

```
D1   strict COMPLETE, 12 canonical, ledger = 12 GATE-REFUSED @0.0, realized 0.0  ->  PASS
D1'  strict COMPLETE, 12 canonical, ledger EMPTY,                  realized 0.0  ->  PASS
```

This is **not** KW9.2's no-op hole — it needs 12 genuine canonical
`COMPLETED` files, i.e. ~14.4 GPU-h of real work — but it is the
primary-arm twin of KW9.7/KW10.4: real spend with no ledger record, so
`realized_gpu_h_final` under-reports and universal assertion 3 passes
vacuously. It has been there since `§R5` and no round has named it; K1's
own rationale now makes the omission conspicuous. **Discharge (free):**
apply J1(a)/J1(b) to the strict branch too — any legitimate 12/12 run
satisfies them trivially (`12 == 12`).

---

## §5 GATE SUMMARY

| Scope item | Verdict |
|---|---|
| K1 — §5 4/4 precondition + U7 mirror; scope note retracted | **PASS** — every path traced (3/4, 4/4, 0/4 ×2, $0-32, count-violation, crash-shortfall); KW10.3 headline scope only |
| K2 — `>=1` distinct `COMPLETED` primary pair, 4 sites | **PASS** — all four sites present; no zero-cost combination passes |
| K3 — universal assertion 8 + division guard | **PASS** — guard in final text, correct, blind spot probed and provably inert |
| K4 — A6 rebuilt self-consistent, forced-fail wiring | **PASS** — arithmetic re-derived exact; dies on exactly `EB J4`; KW10.5 rounding |
| K5 — `§R7` live-body MD5 | **PASS** — `55ba3e9a…` reproduces against `7a0917d` and `ad2bf48` |
| K6 — `attempt_n` schema/emitter, all sites | **PASS** — 17 mentions checked; both noted rows reachable (24 + 48 of 72) |
| K7 — KW9.7 / 9.8 / 9.9 / 9.10 / 9.11 at their named sites | **PASS at the named sites**; residues KW10.4, KW10.6, KW10.7 |
| 24-payload differential suite re-executed | **PASS on execution** (24/24 both sides, 0/24 disagreement vs an independent transcription) — **FAIL on the delta claim**, KW10.2 |
| In-text test list as a build-stage unit test | **FAIL** — KW10.1 (L6 falsified by U8, undisclosed substitution) |
| 200-state composition re-executed | **PASS** — 30/6 → 0/0, 72, max rows 3, Class-2 cap 2, all exact |
| Frozen-zone byte identity (1179 lines, three commits) | **PASS** — md5 identical, `diff` empty |
| Rev-8's own MD5 table + line arithmetic | **PASS** — all four rows reproduce; every arithmetic identity checks |
| Settled-section contacts disclosed | **PASS** — byte-range diff returns exactly the one disclosed line |

**VERDICT: REV-REQUIRED.** Forcing finding: **KW10.1**. No FATAL. No
finding in this round can cost a legitimate run its spend or corrupt the
reconstruction, the charging bound, or the band logic.

**Coordinator note, offered plainly:** KW10.1 is classed MAJOR on parity
with KW9.4 and on the fact that the in-text list is the only durable
specification of the suite (the scripts are scratchpad-only). A
coordinator who judges the in-text list to be illustrative rather than
normative could adjudicate it down and release. That is a legitimate
call — but it should be recorded as an adjudication, not left implicit,
and it should come with the §6 item 8 fix (put the scripts in the repo),
because the two questions are the same question.

---

## §6 WHAT REV 9 MUST DO (binding disposition proposal)

Every item is a one- to five-line edit. No re-derivation, no re-run of
anything beyond the payload suite.

1. **M1 (KW10.1, MAJOR):** rebuild the in-text `L6` payload
   self-consistently (state its ledger, give the exact-quotient
   fraction), disclose EVERY rebuilt payload in the suite description,
   and re-run the list to completion printing failure-reason lists.
2. **m1 (KW10.2):** restate the behavioural delta as six, naming B2' and
   B3-NEG; say B3-AMENDED is newly *emittable*, not newly passing; fix
   the B2/B2' narration crossover at `:2713-2717`. Remove the driver's
   auto-matching `"N/A"` expectation so the tally cannot hide a flip.
3. **m2 (KW10.3):** scope §5's new headline to the paid branch.
4. **m3 (KW10.4):** U7's Otherwise arm asserts the conditional canonical
   directory is empty.
5. **m4 (KW10.5):** state A6/A6''s fraction as the exact quotient.
6. **m5 (KW10.6):** disambiguate `elapsed_s` at `:1077-1079` and `:1451`.
7. **m6 (KW10.7):** de-overload `diag` (dict, or two slots), or correct
   the comment.
8. **m7 (KW10.8):** apply J1(a)/(b) to `COMPLETE`'s strict branch —
   free for any legitimate run — **and**, jointly with M1, copy the
   round's `vcheck_*.py` / `recon_*.py` transcriptions into the repo
   (`matrix-thinking/` or `experiment-runs/`, both ≤25 MB) so the suite
   that certifies this design survives the session that wrote it.

**Round 10 scope, recommended (binding on the next audit):** M1 and m1
ONLY — re-verify the rebuilt L6 by execution and the corrected delta
enumeration — plus a byte-range integrity re-check. m2–m7 are
disclosure/wording edits verifiable by reading the diff; they do not
warrant a suite re-run. Everything this report's §5 marks PASS is
settled and excluded: all seven K dispositions' substance, both suites'
figures, the frozen-zone identity, the MD5 table, and the one disclosed
settled-section contact. **Round 10 should be terminal on inspection,
not on re-execution.**

---

## §7 SCRIPTS

All in this session's scratchpad. The first two are Rev-8's own,
re-run verbatim; the last two are this round's, written independently.

- `vcheck_r8_rev.py` + `drive_vcheck_r8_rev.py` — Rev-8's differential
  harness (OLD vs `§R8`-amended), 24 payloads. Re-run verbatim: 24/24
  both sides.
- `recon_r8.py` — 0.0 / 0.1 / 0.2, derived CELL state, step-3 resume,
  the 200-state composition under both guards. Re-run verbatim: every
  figure exact.
- `r9_indep.py` — **this round's independent transcription** of the
  amended `validity_check` (universal assertions 1–8 + all 5 branches),
  written from `:2296-2509` without using Rev-8's control flow as a
  template; cross-checks all 24 payloads (0 disagreements), traces C1–C6
  (the conditional-arm shapes), and tests A6/A6' under both the exact
  quotient and the literal `0.9296`.
- `r9_probe.py` — D1–D5, this round's residual-hole probes.

**These scripts, like every prior round's, are ephemeral.** See §6 item
8: the design cites scratchpad paths as the authority for its own
verification, and no `.py` exists anywhere under `matrix-thinking/`.

---

## §8 BINDING BUILD CHARTER (restated, NOT yet released)

Unchanged from `NCR_KWALL_ATTACK_R8.md` §8 (itself R7 §9 as adopted by
`§A7-ADJUDICATION`, plus R8's (f)/(g)), with two R9 additions. **This
charter is not in force until a round clears the design.**

1. R5's conditional build-release checklist, in full.
2. R6's five additions, incl. every negative test RUN TO COMPLETION.
3. The 3 micro-smokes (K=26/28/30) pass before queue-eligibility.
4. R7's five additions (a)–(e), unchanged — (d) is the conditional/
   primary canonical-directory disjointness assertion the design now
   cites correctly (KW9.9).
5. R8's additions (f) schema validation of every emitted `attempts[]`
   row incl. `attempt_n`, with a `bootstrap_n>2` positive fixture; (g)
   the build recomputes `charged_vs_measured` from `ledger.attempts` and
   asserts equality before `orchestrator_report.json` is written.
6. **R9 additions (new, binding):** (h) the payload suite wired as a
   build-stage unit test asserts the **failure-reason list**, not merely
   the PASS/FAIL verdict, for every forced-fail negative — a negative
   that dies on the wrong clause must fail the build (KW9.4/KW10.1 are
   both this rule firing); (i) the transcription scripts are committed
   to the repo alongside the design before the first dispatch, so the
   suite outlives the session that wrote it.

**What a clean round unlocks.** Nothing in this round bears on the
science, the budget derivation, the reconstruction procedure, or the
band logic — all of which are settled and re-verified. Once M1/m1 land
and Round 10 confirms them, the charter above may be released and the
design becomes build-eligible: `validity_check`, the recovery/
reconstruction procedure, the trigger and band procedures, the
orchestrator report schema, and the ≤15.00 / 15.3737-bounded dispatch
loop, with the 12-cell primary sweep and the conditional 160K arm
queue-eligible behind the three micro-smokes.
