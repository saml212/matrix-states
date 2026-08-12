# NCR K-WALL — NARROW AUDIT ROUND 8 (scope: J1–J6 verification + suite re-execution)

**STATUS: VERDICT = REV-REQUIRED.** 1 FATAL / 5 MAJOR / 5 MINOR.
Target: `matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md`, header
verified as **"DRAFT-R7 — POST-AUDIT-7, AWAITING NARROW AUDIT ROUND 8
(not build-released, not queue-eligible)"** (`:3-4`) — matches the
expected header exactly, no mismatch to report.

This round was dispatched 2026-08-06 (commit `7a0917d`); the original
auditor never ran. This is the re-dispatch, executed 2026-08-11.

**Artifact pinning.** A new commit (`e6ffe05`) landed mid-session. It
touches `EXPERIMENT_LOG.md`, `STATE.md`,
`matrix-thinking/CONSOLIDATION_POLICY_WATERFALL.md`,
`matrix-thinking/queue/auto_unpause.sh`, and
`research/consolidation-policy-novelty-2026-08-11.md` — **0 changes to
the design file**, verified by `git diff --name-only 7a0917d e6ffe05 --
<design>`. The design's on-disk md5 is `417b35bc56b6c39e5354690c599df33d`,
byte-identical to the snapshot taken at the start of this audit. Every
diff below is pinned to the explicit pair `cb08c47 → 7a0917d`, never to
a moving `HEAD~1`.

**Headline.** J2, J3 and J6's arithmetic are genuinely and completely
discharged — every claimed figure reproduced by direct execution, to
the digit, including the ones the design says it executed. J1, J4 and
J5 are each fixed against the SPECIFIC payload that exposed them and
remain open one step to the side. The forcing finding is in J5: the
disclosed "interval-decided partial-conditional qualifier" scope note
waves off a case that is not hypothetical and not future — it is
G4's own pre-registered `COMPLETE-DEGRADED` sub-case (ii), and under
the design as written it loses the whole ≤15 GPU-h run to `failed/`.

Scope discipline: this report addresses ONLY J1–J6, the two re-run
suites, the `s`/`15.3737` derivation, the frozen-zone identity, and
the three disclosed settled-section contacts. Everything
`§A7-ADJUDICATION` named NEWLY SETTLED (the 24-state derivation,
24/24 totality, quarantine placement, promotion preemption's ordering
guarantee, the 6 legitimate outcomes' passing, resume numbering) was
not re-opened.

---

## §0 METHOD

Everything below is executed, hashed, or diffed. Nothing is asserted.

- **`validity_check` (J1/J2/J4/J5):** all 7 universal assertions and
  all 5 per-`run_status` branches transcribed **independently** from
  `:2255-2396` into `scratchpad/vcheck_r8.py` (every clause carries its
  source line in a comment — this is a fresh transcription, not a re-run
  of `vcheck_r7.py`). **16 payloads** adjudicated: the design's own 14
  (L1–L7 + A1–A7), plus a negative control replaying Rev-6's f-string
  producer, plus a corrected A6. Then **7 further adversarial payloads
  of this round's own construction** (B1, B1', A6*, A6*', B2, B3, B4).
- **Reconstruction (J3):** 0.0 / 0.1 / 0.2, the derived CELL state
  (with KW8.7's precedence clause) and the step-3 resume rule
  transcribed from `:1032-1160` and `:1557-1571` into
  `scratchpad/recon_r8.py`, and the full 5×5×4×2 = 200-state cell
  composition walked under BOTH guards.
- **Arithmetic (J6):** `T`, `s`, the headroom figures and
  `⌊15.0157/1.20⌋` recomputed in exact decimal.
- **Code citations:** read directly out of
  `matrix-thinking/ncr/ncr_earlyln_scale.py`.
- **Integrity:** `md5`, `wc -l`, byte-range `diff`, and a brute-force
  prefix-range hash sweep.

---

## §1 SCOPE 1 — J1–J6 DISCHARGE VERIFICATION

### 1.1 J2 (`tie-break-min` bare literal) — FULLY DISCHARGED

Both cited pseudocode sites are corrected. `:563-570` and `:636-639`
now return the bare literal `"tie-break-min"` with the candidate list
in a fourth tuple slot; the `resolution_detail` field is added to the
schema at `:1511`. Universal assertion 6's text is unchanged, as
`§A7-ADJUDICATION` directed.

Executed:

| Payload | Expect | Actual |
|---|---|---|
| L7 `tie-break-min` bare literal | PASS | **PASS** ✓ |
| L7' same disk state, Rev-6 f-string producer (negative control) | FAIL | **FAIL on U6** ✓ |

The negative control fails with exactly the message KW8.2 predicted
(`trigger.resolution 'tie-break-min, candidates were [26, 28]' not in
enum`). The producer was the defect and the producer is fixed. **PASS.**

### 1.2 J3 (0.2 guard re-key) — FULLY DISCHARGED, EVERY FIGURE REPRODUCED

The 200-state composition, walked independently:

| Guard | Orphans | Abort-trips | bootstrap `n>2` | max rows/cell |
|---|---|---|---|---|
| OLD (Rev 6, "0.1 appended ZERO rows") | **30 / 200** | **6 / 200** | 0 | 2 |
| NEW (§R7 J3, "no `COMPLETED` row") | **0 / 200** | **0 / 200** | **72** | 3 |

Every number the design claims is reproduced exactly — 30, 6, 0, 0,
and the 72 `bootstrap_n>2` states. The 6 OLD-guard abort-trips resolve
to KW8.3's own worked table digit-for-digit (3 unique
`(a1, a2, canonical)` compositions × 2 arms, all resuming to attempt 2):

```
a1=parseable-nonCOMPLETED  a2=dir-absent  canon=parseable-COMPLETED  primary/conditional
a1=unparseable             a2=dir-absent  canon=parseable-COMPLETED  primary/conditional
a1=json-absent             a2=dir-absent  canon=parseable-COMPLETED  primary/conditional
```

The design's separate claim that "every one of those 72 derives
`COMPLETED` or `PERSISTENTLY-ABORTED` — both terminal" is also
confirmed by execution: **0 of the 72 is non-terminal.** G2's
"only way to trip it" sentence (`:1345-1365`) is reworded to the truth
and now names reconstruction as a second, now-closed producer.

Two consequences, one sound and one not disclosed:

- **Sound, and load-bearing for J6:** max Class-2
  (`ceiling_charged==false`) rows per cell is **2**, executed. H3's
  `≤32 = 16 cells × 2` Class-2 cap therefore survives the widened
  guard. Confirmed, not assumed.
- **Not disclosed — KW9.6**, below: max total rows per cell is now
  **3**, and `attempt_n` reaches 3.

**PASS on the mechanism.**

### 1.3 J6 (two-class charging + `s` + the `15.3737` bound) — ARITHMETIC EXACT, CODE CITATION EXACT

Recomputed in exact decimal:

```
R_N     = 15.00 + 0.0157                = 15.0157
12·τ    = 12 × 0.0157                   =  0.1884
32·s    = 32 × 0.0053                   =  0.1696
T       = 15.0157 + 0.1884 + 0.1696     = 15.3737   ✓
alt     = 15.00 + 13(0.0157) + 32(0.0053) = 15.3737 ✓  (both forms agree)
15.50 − 15.3737                         =  0.1263   ✓
15.3737 − 15.0157                       =  0.3580   ✓
15.3737 − 15.20                         =  0.1737   ✓
⌊15.0157 / 1.20⌋                        = 12        ✓
```

`s` derivation against `§R4`'s frozen KW5.7 range: `0.016 − 0.0157 =
0.0003`, `0.021 − 0.0157 = 0.0053`; the conservative upper end
`s = 0.0053` GPU-h = **19.08 s**. Exact.

**The `t0` placement citation is verified by direct read, and is
correct in a way the design does not spell out but needs.**

| Cited | Actual (`ncr_earlyln_scale.py`) | ✓ |
|---|---|---|
| `os.makedirs(outdir, exist_ok=True)` `:237` | line 237 | ✓ exact |
| resume-skip `:240-245` | `if os.path.exists(out_path)` :240 … `return prev` :245 | ✓ exact |
| `nt.claim_config` `:247` | line 247 | ✓ exact |
| `torch.manual_seed` `:248` | line 248 | ✓ exact |
| `NCREarlyLNModel(...).to(device)` `:249` | line 249 | ✓ exact |
| `rec` dict incl. `git_commit`/`n_params` `:250-255` | lines 250–255 | ✓ exact |
| `t0 = time.time()` `:257` | line 257 | ✓ exact |

**Independent corroboration the design did not offer:** the top-level
`rec["elapsed_s"] = time.time() - t0` for a `COMPLETED` record is at
`:302`, i.e. the field a reconstructed row reads genuinely spans
`:257` → past the entire post-train instrument sequence (`z_dump`,
deep probe, Axis-C lock, trust screen, `blank_out_check`, `eval_cell`).
The startup term really is the only thing it omits, so `s`'s
derivation rests on the right quantity. (One ambiguity this exposes —
KW9.11, below.)

The two-class structure itself is sound: Class-1 rows are each charged
`≥1.20` of an `R_N ≤ 15.0157` total, so `≤12` of them; Class-2 rows are
structurally capped at 2 per cell × 16 cells = 32, which this round
verified by execution rather than accepting. The false "I1 establishes
the premise" sentence is gone, replaced by honest accounting that says
plainly the residual is in the estimate, not the rule. **PASS on the
derivation.** One residual overstatement, KW9.8.

### 1.4 J1, J4, J5 — each fixed against its own payload, each still open one step aside

Executed against a fresh transcription:

| # | Payload | Expect | Actual |
|---|---|---|---|
| L1–L6 | the six §5-reportable outcomes | PASS | **PASS** ✓ (6/6) |
| A1 | R5 no-op | FAIL | **FAIL** ✓ |
| A2 | `EXHAUSTED-BUDGET` @ 12 canonical | FAIL | **FAIL** ✓ |
| A3 | `STOPPED-BY-OPERATOR` | FAIL | **FAIL** ✓ (U1) |
| A4 | ENHANCED no-op via `incomplete_at_K` | FAIL | **FAIL on J1(a)** ✓ |
| A5 | ENHANCED no-op via `interval_resolved_Ks` | FAIL | **FAIL on J1(a)** ✓ |
| A6* | well-formed suspect claimed `EXHAUSTED-BUDGET` | FAIL | **FAIL on J4** ✓ |
| A6*' | same state correctly claimed `-SUSPECT-OVERCHARGE` | PASS | **PASS** ✓ |
| A7 | fabricated conditional arm | FAIL | **FAIL on U7** ✓ |

So each of J1, J4, J5 does exactly what `§R7` says it does against the
payload that produced it. Then this round's own extensions:

| # | Payload | Expect | Actual |
|---|---|---|---|
| B1 | no-op with 12 zero-cost `GATE-REFUSED` rows, claimed `COMPLETE` | FAIL | **PASS** ✗ KW9.2 |
| B1' | the same ledger claimed `COMPLETE-DEGRADED` | FAIL | **PASS** ✗ KW9.2 |
| B2 | A6*'s ledger with the fraction mis-declared `0.20` | FAIL | **PASS** ✗ KW9.3 |
| B3 | **LEGITIMATE** conditional-throttled 3/4 + qualifier band | PASS | **FAIL** ✗ KW9.1 |
| B4 | paid conditional arm absent from the ledger | FAIL | **PASS** ✗ KW9.7 |

---

## §2 SCOPE 2 — THE THREE DISCLOSED SETTLED-SECTION CONTACTS

All three verified by direct diff against `cb08c47`. All three are
exactly what `§R7` discloses.

1. **H3 "True spend, worst case"** — the AUTHORIZED primary edit site
   for J6, correctly declared as such rather than smuggled. Full
   re-derivation, arithmetic shown, verified exact above. ✓
2. **Unified-enum-table precedence sentence** — the entire hunk is one
   line: ``(`T ≤ 15.2041h`, above).**`` → ``(`T ≤ 15.3737h`, §R7 J6's
   two-class re-derivation, above).**``. A single figure plus a
   six-word attribution. The sentence's LOGIC (that this derivation
   outranks `§R4`'s frozen KW5.1 row) is byte-unchanged, and the table
   itself is untouched. ✓
3. **§6 "Own cost ceiling" bullet** — `15.2041h` → `15.3737h` and
   `0.0041h` → `0.1737h`, plus a clarifying parenthetical stating that
   most of the process-startup term is now priced per-row rather than
   margin-absorbed. The bullet's STRUCTURE (`15.20 + 0.30 = 15.50`) and
   the `≤15.50 + ≤0.15 = ≤15.65` total-program disclosure are unchanged.
   ✓

The fourth, "Rounding conservatively" (disclosed in `§R7`'s line-count
paragraph rather than the numbered list), is likewise numeric-only:
`0.0041h` → `0.1737h`, `0.30 ≫ 0.1737`, `0.1263h` headroom. No new
logic. ✓

**None of the three disturbs what was settled.**

---

## §3 SCOPE 3 — INTEGRITY

| Check | Result |
|---|---|
| Frozen range `## §A1-ADJUDICATION` → end-of-pre-Rev-7 content, before (`cb08c47:2749-3927`) | 1179 lines, md5 `3805e7dac8893f272f51fb62210e28be` ✓ |
| Same range, after (`7a0917d:3117-4295`) | 1179 lines, md5 `3805e7dac8893f272f51fb62210e28be` — **`diff` returns empty, BYTE-IDENTICAL** ✓ |
| §R7 claim: whole file before = `4e03ed5d13b2139e123ab079bbc0517e` (3927 lines) | **VERIFIED** ✓ |
| §R7 claim: live body before = `68440ddc8fc7408168daa8ce4ef2f090` (2748 lines) | **VERIFIED** ✓ |
| §R7 arithmetic `3927 = 2748 + 1179`; `3116 − 2748 = 368`; `3116 + 1179 = 4295` | **VERIFIED**, all three ✓ |
| §R7 claim: live body after = `1f93fa4ca8ee7333d573d5d095b37453` (3116 lines) | line count ✓; **hash DOES NOT REPRODUCE — KW9.5** |
| §R7 uses `§R5`'s precise "→ end-of-pre-Rev-7 content" phrasing throughout, never "→ EOF" | **VERIFIED** — KW8.10 genuinely does not recur ✓ |
| Design file untouched by the intervening commit `e6ffe05` | **VERIFIED** ✓ |

The load-bearing half of the instrument — the frozen zone — verifies
byte-for-byte. **No settled section was disturbed by Rev 7.** The
live-body half does not verify; see KW9.5.

---

## §4 FINDINGS

### KW9.1 — FATAL. J5's disclosed scope note dismisses a LIVE pre-registered outcome as hypothetical; a partially-throttled conditional arm carrying its qualifier band is rejected after the full spend.

**Quote, the scope note** (`:2318-2325`): *"(Scope note, disclosed
rather than silently assumed: this assertion, as specified, requires
the conditional arm's full 4/4 completion for a PAID qualifier claim;
it does not yet cover a **hypothetical future** interval-decided
PARTIAL-conditional qualifier, **which this design does not currently
define** — out of J5's scope.)"*

**Quote, the design defining exactly that case** (`:1657-1663`, G4):
*"(ii) *conditional-throttled* — the trigger fired (`DECIDED`, and the
G5 band precondition held), but **1-4 of the conditional arm's 4 cells'
FIRST attempts were refused by the hard gate** before the 15.00 cap was
reached. (iii) **`conditional-retry-refused`** — the trigger fired and
every conditional cell got its first attempt, but a conditional cell's
RETRY was refused."*

**Quote, §5 mandating the qualifier unconditionally** (`:2764-2769`):
*"If the trigger (§4) fires at `K_trig∈{26,28,30,32}` …, its 4-cell
**160K** rate … **is reported** ALONGSIDE the PRIMARY 80K
classification above as a budget-verdict qualifier at `K_trig`."*
There is no 4/4 precondition anywhere in the rule; the three bands are
defined purely as thresholds on "`K_trig`'s rate" (`≤1/4`, `≥3/4`,
exactly `2/4`).

**Evidence, three-part.**

(a) **The case is reachable and is not exotic.** The primary baseline
is priced at 12 × 1.20 = 14.40 GPU-h against a 15.00 hard gate, and the
conditional arm at 4 × 2.32 = 9.28. The design says so itself
(`:2028-2031`): *"in the pathological case where all 12 primary cells
consume their full ceiling before the conditional arm is considered,
the hard gate correctly THROTTLES OR REFUSES **part or all** of the
conditional arm."* Partial throttle is a MAIN branch of the budget
model, pre-registered twice in G4.

(b) **The real code still yields a rate for a sub-4 directory.**
`harvest()` computes `rate = n_converged / n_seeds` at
`ncr_earlyln_scale.py:394-396` with `n_seeds = len(seeds)` from
`discover_seeds_by_K`'s glob. Only `gate_eligible` / `ladder_verdict`
are suppressed for a sub-4 rung (`gate_eligible = n_seeds >= 4` at
`:403`, consumed at `:404`/`:430`, and `excluded_sub4_Ks` at `:450`) —
`rate` is computed and returned regardless. A build implementer wiring §5's three threshold bullets to
`harvest()['per_K'][K_trig]['rate']` — the natural and only specified
implementation — emits `SLOW-CONVERGENCE-AT-160K` from a 3/3 read.

(c) **U7 then kills the run.** Executed, payload B3: a genuine
`COMPLETE-DEGRADED` sub-case (ii) report — 12/12 primary canonical, 3
of 4 conditional cells `COMPLETED` with real canonical files, 1
conditional `GATE-REFUSED`, `launched=true`,
`qualifier_band="SLOW-CONVERGENCE-AT-160K"`. Clause (a) requires
**exactly 4** conditional canonical files: `3 == 4` is False. Clause
(b) requires `launched==False`: also False. **NEITHER holds → FAILS
universal assertion 7 → routed to `failed/` after up to 15 GPU-h has
been spent.**

**Why this is FATAL and not a disclosed limitation.** The scope note
asserts two things that are both false against the live body: that the
partial-conditional case is "hypothetical future," and that "this
design does not currently define" it. G4 defines it, twice, as a
pre-registered `run_status`. The only text reconciling §5's "is
reported" with U7's rejection is a parenthetical inside U7 itself
(`:2326-2328`) listing *"throttled short of a qualifier"* as a state in
which `qualifier_band is None` — but **no rule anywhere says when that
obtains**, and §5 says the opposite. A build implementer has two live
sections in direct contradiction and no tie-break. This is the identical
failure shape as KW7.4 and KW8.2 — a pre-registered, §5-reportable
outcome that `validity_check` rejects after the full spend — both of
which this gauntlet classed FATAL.

**Discharge condition.** One clause in §5, and one in U7. In §5, state
explicitly that the 160K qualifier band is reported **only** when all
4 conditional cells reach `COMPLETED`; a conditional arm throttled to
`<4` completed cells reports `qualifier_band=null` with the completed
cells disclosed as data only (this is also the behaviour the runner's
own S9.5 sub-4 rule already implies, so cite it). In U7, add the mirror:
if `conditional.launched==True` and `qualifier_band is None`, the
conditional canonical count must equal
`len({(a["K"],a["seed"]) for a in ledger.attempts if
a["arm"]=="conditional" and a["status"]=="COMPLETED"})` and be `<4`.
Then re-run B3 to completion — it must PASS with `qualifier_band=null`
and FAIL with a band. Do not hand-check it.

### KW9.2 — MAJOR. J1 added evidence-of-a-ROW, not evidence-of-WORK; the no-op hole survives at zero GPU-h.

**Quote** (`:2337-2342`): *"**AND (§R7 J1 — closes KW8.1's FATAL
regression) every primary `(K,seed)` pair has ≥1 row in
`ledger.attempts`, AND the primary canonical count equals
`len({(a["K"],a["seed"]) for a in ledger.attempts if
a["arm"]=="primary" and a["status"]=="COMPLETED"})`**"*.

**Evidence.** Executed, payload B1: `run_status="COMPLETE"`,
`realized_gpu_h_final=0.0`, **zero canonical files**, `smoke` all
`PASS`, `band={"label":"INCOMPLETE-AT-K","interval_resolved_Ks":[],
"incomplete_at_K":[26,28,30]}`, and `ledger.attempts` = **12
`GATE-REFUSED` rows at `elapsed_h=0.0`**, one per primary `(K,seed)`.
U1–U7 all hold (U3: `|0.0 − 0.0| = 0`). The OTHERWISE branch fires; all
three K's read `0 < 4` ✓; J1's clause (a) is satisfied — every one of
the 12 pairs has a row ✓; J1's clause (b) reads `0 == 0` ✓.
**PASSES → routed to `completed/`.** Payload B1' shows the identical
ledger also passes `COMPLETE-DEGRADED` (the branch J1 copied its
clauses from), since a `GATE-REFUSED` row satisfies the
throttle-evidence clause too.

**Why this is the same hole.** KW8.1's characterisation was that the
branch *"is satisfiable by an empty filesystem and an empty ledger."*
Post-J1 it is satisfiable by an empty filesystem and a **zero-cost**
ledger. A row is free to emit; J1 requires a row, not work. This is
not purely adversarial: a hard-gate comparison bug that refuses every
cell produces exactly these 12 rows, and if the band logic (correctly,
from 0 completed cells) reports all three K's incomplete, a mislabel to
`COMPLETE` routes a 0-GPU-h run to `completed/`.

**Discharge condition.** One clause: `COMPLETE`'s OTHERWISE branch
additionally requires `len({(a["K"],a["seed"]) for a in
ledger.attempts if a["arm"]=="primary" and a["status"]=="COMPLETED"})
>= 1` — a run claiming `COMPLETE` must have completed something.
Verified against this round's suite: B1/B1' then FAIL while L2 (11
`COMPLETED`) still PASSES. Consider the mirror clause for
`COMPLETE-DEGRADED`, whose own definition requires the 12-cell baseline
to have completed.

### KW9.3 — MAJOR. J4's dichotomy is real but unenforced: `validity_check` trusts a self-reported float it has every input to recompute.

**Quote, the claim** (`:1726-1734`): *"**this makes the two labels a
DICHOTOMY, not merely disjoint-by-prose: for any ledger satisfying the
shared `>13.80`/`<12`/`GATE-REFUSED` base clauses,
`ceiling_charged_fraction` is a single real number that is EITHER
`≤0.50` … OR `>0.50` … so exactly one of the two `EXHAUSTED-BUDGET*`
labels can ever be the CORRECT claim for a given ledger**, closing the
mislabelling escape by construction, not by disclosure alone."*

**Quote, the definition** (`:1748-1753`): *"`charged_vs_measured.
ceiling_charged_fraction = ceiling_charged_gpu_h /
realized_gpu_h_final` (`ceiling_charged_gpu_h` = the sum of `elapsed_h`
over every `attempts[]` row with `ceiling_charged==true`)."*

**Evidence.** The quantity is a pure function of `ledger.attempts` —
one line of arithmetic over the `ceiling_charged` flag that **J6 itself
made authoritative this revision**. `validity_check` never computes it.
It reads `report["charged_vs_measured"]["ceiling_charged_fraction"]`, a
scalar the same report supplies. Executed, payload B2: A6*'s ledger
(11 `CRASHED-RECOVERED` rows at full ceiling `= 13.20` + one measured
`COMPLETED` row `= 1.00`, `realized = 14.20`, **true fraction 0.9296**)
declaring `ceiling_charged_fraction = 0.20` and claiming plain
`EXHAUSTED-BUDGET`. All three base clauses hold; J4's clause reads
`0.20 <= 0.50` ✓. **PASSES → routed to `completed/`**, evading
`-SUSPECT-OVERCHARGE`'s binding protection that *"resubmission is NEVER
automatic"* (`:1584-1585`) — the exact escape J4 exists to close, on a
ledger that is 93% environment-fault noise.

The dichotomy claim is true about *ledgers*. `validity_check` does not
adjudicate ledgers; it adjudicates *labels against reports*. Nothing
ties the two together.

**Discharge condition.** Replace the read with a recomputation, as one
universal assertion: `abs(charged_vs_measured.ceiling_charged_gpu_h −
sum(a["elapsed_h"] for a in ledger.attempts if a["ceiling_charged"]))
<= 1e-6`, and likewise tie `ceiling_charged_fraction` to
`ceiling_charged_gpu_h / realized_gpu_h_final`. This is the exact
sibling of universal assertion 3, which already does this for
`realized_gpu_h_final`. B2 then FAILS while L5/L6/A6*' still pass.

### KW9.4 — MAJOR. The in-text A6 negative test has no teeth: it dies on a bookkeeping assertion before reaching the J4 clause it exists to exercise.

**Quote** (`:2497-2505`): *"**Suspect run mislabelled plain
`EXHAUSTED-BUDGET`** (**9** primary-arm `CRASHED-RECOVERED` rows **at
full ceiling**, 3 primary first-attempt `GATE-REFUSED` rows,
**`realized=14.40`**, 0 canonical, **`ceiling_charged_fraction=0.93`**
…): all three base clauses hold (`14.40>13.80` ✓, `0<12` ✓, ≥1 primary
first-attempt `GATE-REFUSED` ✓) — **but §R7 J4's new clause,
`ceiling_charged_fraction<=0.50`, reads `0.93>0.50`** — **FAILS.**"*

**Evidence.** The payload's own numbers do not close. A primary-arm
full ceiling is `1.20`; `GATE-REFUSED` rows carry `elapsed_h=0.0`
(`:1553`). So the described ledger sums to `9 × 1.20 + 3 × 0.0 =
**10.80**`, not `14.40`. Two independent consequences, both executed:

- **Universal assertion 3** (`abs(realized_gpu_h_final −
  sum(elapsed_h)) <= 1e-6`) fails first: `|14.40 − 10.80| = 3.60`. The
  payload never reaches any per-`run_status` branch.
- Made self-consistent at `realized = 10.80` instead, it fails the
  **first** base clause (`10.80` is not `> 13.80`) — still never
  reaching J4's clause.
- The stated fraction is also wrong: with every non-zero row
  ceiling-charged, `ceiling_charged_gpu_h / realized_gpu_h_final =
  10.80/10.80 = 1.00`, not `0.93`.

So the design's traced clause-by-clause walk (`14.40>13.80 ✓`) is not
reproducible against the payload it describes. **J4 itself is fine** —
this round's rebuilt A6* (11 `CRASHED-RECOVERED` @ 1.20 + one measured
`COMPLETED` @ 1.00 + a `GATE-REFUSED`, `realized = 14.20`, fraction
`0.9296`) does exercise the clause and fails on it exactly as intended,
with A6*' passing as `-SUSPECT-OVERCHARGE`. The defect is in the test,
not the fix.

**Why this matters beyond bookkeeping.** The standing build charter
(R7 §9 item 4a, adopted by `§A7-ADJUDICATION`) mandates wiring this
exact in-text list *"as a build-stage unit test of the real
`validity_check`, with A1–A7 as forced-fail negatives."* A forced-fail
negative that fails for the wrong reason gives **zero coverage** of the
clause it was written to protect — and an implementer who "fixes" the
red test by relaxing U3 or the `>13.80` clause makes things actively
worse. This is precisely the repo's own standing rule that a negative
test must be run to completion and shown to have teeth.

**Discharge condition.** Replace the in-text A6 payload with a
self-consistent one (this round's A6* is a worked example) and state
which assertion it dies on. Re-run the whole list to completion.

### KW9.5 — MAJOR. §R7's own MD5 table's "live body, after" figure does not reproduce against the committed file.

**Quote** (`§R7` MD5 table): *"| Live body (§1–§7), after
(pre-`§R7`-append) | `1f93fa4ca8ee7333d573d5d095b37453` (3116 lines) —
**changed, as expected**: every J1–J7 edit landed here |"*.

**Evidence.** The live body is, by `§R7`'s own convention (verified
against the "before" figure, which reproduces exactly), lines 1 through
the line preceding `## §A1-ADJUDICATION`. In `7a0917d` that anchor is
at `:3117`, so the range is `1..3116` — **3116 lines, matching the
claim.** Its md5 is **`55ba3e9a9289e10f5e7fde5864c21970`**, not
`1f93fa4ca8ee7333d573d5d095b37453`.

This is not a range-convention quibble. A brute-force sweep over every
prefix range (start line 1–30 × end line 2900–3300, both with and
without a trailing newline, and against the `cb08c47` file as well)
found **no range whatsoever** that hashes to the claimed value.
The claimed figure is unreproducible, full stop.

**Why MAJOR, and how it differs from KW8.10.** KW8.10 was a *wording*
imprecision whose substance was correct — the hash reproduced for the
range the row actually meant. Here the number matches nothing. `§R7`
opens by asserting that the frozen sections are *"UNCHANGED as
historical record — verified, not asserted (MD5 block below)"*; the
frozen half of that block does verify, byte-for-byte, so **nothing
settled was disturbed and no finding here reopens a settled section**.
But the other half of the same instrument — the one that would let a
downstream reader confirm the live body being audited is the live body
Rev 7 says it produced — is broken, and the most likely benign
explanation (a live-body edit made after the hash was taken and after
`§R7` was appended, preserving line count) is exactly the thing the
instrument exists to detect. `§R7` explicitly claims KW7.10's failure
mode does not recur; on the wording it is right, on the figure it is not.

**Discharge condition.** Recompute and restate the live-body-after
figure from the committed file (`sed -n '1,3116p' <file> | md5` against
`7a0917d`), or, following KW7.10's own ratified precedent, decline to
restate it and direct the reader to `git show` — but do not leave a
figure that matches nothing.

### KW9.6 — MAJOR. J3's `bootstrap_n` emits `attempt_n=3`, which the Output JSON schema forbids; undisclosed in §R7's disposition table.

**Quote, J3** (`:1132-1145`): *"let `bootstrap_n = max(every attempt_n
already recorded by 0.1 for this cell, default 0) + 1` … **a bootstrap
row's `attempt_n` is a RECONSTRUCTION LABEL, not a future dispatch
number, and may exceed `2`** — harmless, because every cell a bootstrap
row lands on derives TERMINAL."*

**Quote, the schema** (`:1499`, **not edited this revision** — confirmed
by direct diff): *"`"attempt_n":1|2,"elapsed_h":float,`"*.

**Evidence.** Executed: **72 of 200** NEW-guard compositions produce a
bootstrap row with `attempt_n > 2` (reproducing the design's own 72),
and max rows per cell rises from 2 to 3. Every such row violates the
declared `attempts[]` schema. The `attempts[].status` table's
"Typical `attempt_n`" column (`:1548-1556`) likewise still reads "1 or
2" for every value, including the `CRASHED-RECOVERED` and `COMPLETED`
rows a bootstrap writes.

The design's terminality argument is correct and this round confirmed
it (0 of the 72 is non-terminal), so there is **no dispatch hazard** —
this is purely a schema-vs-emitter contradiction. But
`orchestrator_report.json` is the design's single pool artifact, the
schema is its contract, and `§R7`'s 12-row disposition table does not
list this contact at all. Any build-stage schema validator generated
from `:1499` rejects a legitimately reconstructed report.

**Discharge condition.** One character: `"attempt_n":int` (or
`1|2|…` with a note), plus one clause in the `attempts[].status` table
stating that reconstruction bootstrap rows may carry `attempt_n>2` as a
label. Disclose the contact in the disposition table.

### KW9.7 — MINOR. U7 is silent when `qualifier_band is None`, so a paid conditional arm can be wholly absent from the ledger.

Executed, payload B4: a 12/12 `COMPLETE` primary run with
`conditional={"launched":true,"per_seed":[0,1,2,3],
"qualifier_band":null}` and **4 conditional canonical files on disk**
but **zero conditional rows in `ledger.attempts`** — up to 9.248 GPU-h
of real spend invisible to the ledger. U7 does not fire (`qualifier_band
is None`); no per-`run_status` branch reads the conditional tree.
**PASSES.** This is the exact mirror of KW8.5 (fabricated evidence with
no spend) in the other direction (real spend with no record).
**Discharge:** the KW9.1 mirror clause already proposed above closes
this too — tie the conditional canonical count to the conditional
`COMPLETED`-row count whenever `launched==True`, band or no band.

### KW9.8 — MINOR. KW8.12's "bounded by the same derivation" is not established.

**Quote** (`:1112-1119`): *"a crash BEFORE `os.makedirs` … leaves no
attempt directory and is charged `0` … it is disclosed here rather than
charged … **and is bounded by the same derivation, "True spend, worst
case," below.**"*

The derivation is `T ≤ R_N + 12·τ + 32·s`. A crash before
`os.makedirs` produces **no row**, so it is neither Class 1
(`ceiling_charged==true`) nor Class 2 (`COMPLETED`, measured) and has
no term and no multiplicity cap anywhere in that expression. Under the
mandated supervisor loop the occurrence count is bounded only by the
restart count, which no section caps. `15.3737h` is therefore an exact
bound on *spend for which evidence survives*, not on all true spend.

Mitigating, and why this is MINOR not MAJOR: the window is `:1`–`:237`
(interpreter start, `torch`/CUDA import, arg parsing), a *systematic*
failure there is screened by the three mandated micro-smokes before
queue-eligibility, and each transient occurrence costs at most `s`
(≈19 s). The honest fix is one clause, not a re-derivation.
**Discharge:** delete "and is bounded by the same derivation," and say
instead that the term sits outside `T`'s bound, is screened
systematically by the micro-smoke gate, and is left to the `0.30h`
stated margin — the same honesty J6 applied everywhere else.

### KW9.9 — MINOR. The live body cites a `§9(d)` that does not exist in this document.

**Quote** (`:1467-1469`): *"This is a static, build-time invariant, not
a per-report runtime check; **the build asserts it directly (§9(d), R7
additions) before the first conditional dispatch**"*.

This design's sections are §1–§7 (live) and §A1–§A7/§R1–§R7 (frozen) —
verified by a full heading scan. There is no §9. The reference is to
`NCR_KWALL_ATTACK_R7.md` §9, which is not named at the citation site.
The half of J5's disjointness guarantee that the paragraph delegates to
"the build" therefore has no in-document referent. This is a recurrence
of the defect class the design's own KW2.8/KW3.13/KW4.6 close-out
records (*"the … 'recorded above' cross-reference FALSE: no live
section specified it"*). **Discharge:** name the document, or fold the
assertion into the live body's own build-gate list.

### KW9.10 — MINOR. J2's verification payload never exercises the field J2 created; the pseudocode's return arity is inconsistent.

The in-text `tie-break-min` payload (`:2443-2447`) carries
`trigger.candidate_set=[26,28]` — a pre-existing schema field — and
never sets `trigger.resolution_detail`, the field J2 was directed to
add and into which both pseudocode sites now write
`f"candidates were {sorted(K_trigs)}"`. `resolution_detail` is
un-asserted and unexercised by any of the 14 payloads. Separately, the
pseudocode now returns a 4-tuple on the `DECIDED` path
(`("DECIDED", K, "unanimous", None)`) but a 2-tuple
(`("TRIGGER-UNRESOLVED", blocking_K)`) and a 3-tuple
(`("TRIGGER-UNRESOLVED", None, band_blocked_K_trig)`) on the others, so
`resolution` sits at index 2 in one shape and index 0 in another.
**Discharge:** set `resolution_detail` in the payload and normalise the
tuple shape (or return a dict).

### KW9.11 — MINOR. "the JSON's own `elapsed_s`" is ambiguous — a nested field of the same name excludes the entire post-train instrument sequence.

0.1's rows 4–6, 0.2's OK bootstrap and the live PROMOTE branch all
charge *"`<the JSON's own elapsed_s>`/3600 + s"*. The record carries
**two** fields of that name: top-level `rec["elapsed_s"]`
(`ncr_earlyln_scale.py:302`, measured from the `:257` `t0` — the
correct one, and the one `s`'s derivation assumes), and nested
`rec["train"]["elapsed_s"]` (returned at `:202`, measured from a
**different** `t0` at `:175`), which excludes `z_dump`, the deep probe,
the Axis-C lock, the trust screen, `blank_out_check` and `eval_cell`.
An implementer reading the nested field under-charges by the whole
instrument sequence — a term orders of magnitude larger than `s`, and
one no leak class prices. **Discharge:** write `rec["elapsed_s"]`
(top-level) explicitly at all three sites and cite `:302`.

---

## §5 GATE SUMMARY

| Scope item | Verdict |
|---|---|
| J1 — `COMPLETE` OTHERWISE positive-evidence clauses | **PARTIAL** — A4/A5 killed ✓, but KW9.2 |
| J2 — bare-literal `trigger.resolution` + `resolution_detail` | **PASS** (KW9.10 wording/coverage only) |
| J3 — 0.2 guard re-key, `max+1` bootstrap, G2 reword | **PASS** — 30/6 → 0/0 and 72 reproduced exactly by execution |
| J4 — `EXHAUSTED-BUDGET*` label dichotomy | **PARTIAL** — the dichotomy is real, but KW9.3 (unenforced) + KW9.4 (toothless negative) |
| J5 — conditional-arm evidence + directory disjointness | **FAIL** — KW9.1 (scope note adjudicated: NOT hypothetical), KW9.7, KW9.9 |
| J6 — row-wise charging invariant, `s=0.0053`, `T ≤ 15.3737` | **PASS** on arithmetic, code citation, and the ≤32 Class-2 cap (executed); KW9.8 wording, KW9.11 ambiguity |
| 14-payload suite re-executed | **PASS** — all 14 reproduce as claimed |
| 200-state composition re-executed | **PASS** — every figure reproduced to the digit |
| Frozen-zone byte identity (1179 lines, both sides) | **PASS** — `diff` empty, md5 identical |
| Live-body-after MD5 | **FAIL** — KW9.5 |
| Three disclosed settled-section contacts (non-numeric / numeric-only) | **PASS**, all three exactly as disclosed |
| Schema consistency after J3 | **FAIL** — KW9.6 |

**VERDICT: REV-REQUIRED.** Forcing finding: **KW9.1**.

---

## §6 WHAT REV 8 MUST DO (binding disposition proposal)

- **K1 (KW9.1, FATAL):** define in §5 that the 160K qualifier band is
  reported ONLY on 4/4 conditional completion, and that a throttled
  conditional arm reports `qualifier_band=null` with its completed
  cells disclosed as data only; add U7's mirror clause for
  `launched==True, qualifier_band is None`. Retract the "hypothetical
  future" scope note — the case is G4 sub-case (ii)/(iii). Re-run the
  payload suite including a legitimate 3/4 conditional-throttled report.
- **K2 (KW9.2):** `COMPLETE`'s OTHERWISE branch additionally requires
  ≥1 distinct `COMPLETED` primary pair.
- **K3 (KW9.3):** recompute `ceiling_charged_gpu_h` /
  `ceiling_charged_fraction` from `ledger.attempts[].ceiling_charged`
  as a universal assertion, the sibling of universal assertion 3.
- **K4 (KW9.4):** rebuild the in-text A6 payload self-consistently and
  name the assertion each negative dies on; re-run the list to
  completion.
- **K5 (KW9.5):** recompute or withdraw the live-body-after MD5.
- **K6 (KW9.6):** widen the schema's `attempt_n` and disclose the
  contact.
- **K7:** the five MINORs (KW9.7–KW9.11), each a one-clause fix.

**Round 9 scope, recommended (binding on the next audit):** K1–K6
ONLY, plus one re-run of the payload suite (now including a legitimate
partial-conditional payload) and the 200-state composition.
Everything verified PASS in §5 is settled and excluded: J2's producer
fix, J3 in full (both guards, all four figures, the G2 reword, the
terminality of all 72 bootstrap states), J6's arithmetic / `s`
derivation / code citations / Class-2 cap, the frozen-zone identity,
and the three disclosed settled-section contacts.

---

## §7 SCRIPTS

Both transcriptions live in this session's scratchpad and are
independent of Rev 7's own `vcheck_r7.py`/`recon_r7.py` (written from
the design text, not adapted from them). They should be re-run verbatim
by Rev 8's audit rather than re-derived:

- `vcheck_r8.py` — all 7 universal assertions + 5 per-`run_status`
  branches, verbatim from `:2255-2396`, with 16 design payloads +
  7 R8 adversarial payloads (B1, B1', A6*, A6*', B2, B3, B4).
- `recon_r8.py` — 0.0 / 0.1 / 0.2, derived CELL state and step-3
  resume, verbatim from `:1032-1160` / `:1557-1571`; the 200-state
  composition under both guards, plus the Class-2-per-cell cap and the
  `attempt_n` range as instrumented outputs.

---

## §8 BINDING BUILD CHARTER (restated, NOT yet released)

Unchanged from `NCR_KWALL_ATTACK_R7.md` §9, which `§A7-ADJUDICATION`
adopted as the standing charter, plus two R8 additions. **This charter
is not in force until a round clears the design.**

1. R5's conditional build-release checklist, in full.
2. R6's five additions, incl. **every negative test RUN TO COMPLETION,
   never merely written** — see KW9.4, which is that rule firing.
3. The 3 micro-smokes (K=26/28/30) pass before queue-eligibility.
4. R7's five additions (a)–(e), unchanged.
5. **R8 additions (new, binding):** (f) the build asserts every emitted
   `attempts[]` row validates against the declared schema, `attempt_n`
   included, with a reconstructed `bootstrap_n>2` row as a positive
   fixture; (g) the build recomputes `charged_vs_measured` from
   `ledger.attempts` and asserts equality with the reported block
   before `orchestrator_report.json` is written.
6. **Unchanged ceilings:** `15.00` hard gate / `12.00` retry gate /
   `15.20` disclosed / `15.50` declared `gpu_h_estimate`, `+0.15`
   micro-smokes outside the ledger. J6's `T ≤ 15.3737h` true-spend
   bound is verified exact and changes none of them.
