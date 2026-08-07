# NCR K-WALL — NARROW AUDIT ROUND 7 (terminal round, scope: I1–I5)

**STATUS: VERDICT = REV-REQUIRED.** 2 FATAL / 4 MAJOR / 6 MINOR.
Target: `matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md`, header
verified as **"DRAFT-R6 — POST-AUDIT-6, AWAITING NARROW AUDIT ROUND 7
(not build-released, not queue-eligible)"** (`:3-4`) — matches the
expected header exactly, no mismatch to report.

This round was dispatched as the expected TERMINAL round. It is not.
Two forcing findings block CLEAR-FOR-BUILD; both are in the round's own
binding scope (I4's validity text), both are cheap to discharge, and
one of them (KW8.1) is a REGRESSION — it reopens a hole `§A6-ADJUDICATION`
named SETTLED. The build charter is nonetheless restated in §9 so a
Rev 7 can be adjudicated and released in one further pass.

Scope discipline: this report addresses ONLY the I1–I3 reconstruction
contract, the I4–I5 validity text, the three disclosed settled-section
contacts, the two collateral fixes, and integrity. The numeric sweeps
(partition, trigger, pricing), crash windows 1–8, the H3 arithmetic
itself, and the enum precedence mechanism are excluded and were not
re-run.

---

## §0 METHOD

Everything below is executed or diffed, not asserted:

- **Reconstruction (I1–I3):** the 0.0/0.1/0.2 rules and step-3 resume
  were transcribed verbatim into
  `scratchpad/recon.py` and run over (a) the 24-state per-attempt space
  and (b) the full 200-state cell-level composition
  (5 attempt-states × 5 attempt-states × 4 canonical-states × 2 arms).
- **Validity (I4–I5):** `validity_check` — all 6 universal assertions
  and all 5 per-`run_status` branches — was transcribed verbatim from
  `:2006-2091` into `scratchpad/vcheck.py`, and **13 payloads** (6
  legitimate, 7 adversarial) were adjudicated against it. The
  transcription follows the design line-for-line; every clause carries
  its source line number in a comment.
- **Integrity:** `git diff HEAD~1 HEAD`, `md5`, and byte-range `diff`.
- **Code citations:** read directly out of
  `matrix-thinking/ncr/ncr_earlyln_scale.py` and
  `matrix-thinking/ncr/run_ncr.py`.

---

## §1 SCOPE 1 — I1–I3 RECONSTRUCTION CONTRACT

### 1.1 State-space re-derivation — INDEPENDENTLY CONFIRMED

The design defines four dimensions (`:1057-1064`): attempt dir
(present/absent), attempt JSON (parseable/unparseable/absent),
`canonical_state` (OK/CORRUPT/ABSENT), `arm` (primary/conditional).

```
nominal      2 x 3 x 3 x 2                                       = 36
impossible   dir=absent x json in {parseable,unparseable} x 3 x 2 = 12
valid        36 - 12                                             = 24     ✓
core (dir,json,canonical) triples = 12  ->  12 x 2 arms          = 24     ✓
```

**The impossibility argument for the excluded 12 is SOUND and was
verified against the real code, not taken on the design's word.** The
attempt JSON path is
`.../K{K}_s{seed}_attempt{n}/earlyln_K{K}_s{seed}.json`; the basename
is produced by `cell_id`, `ncr_earlyln_scale.py:206-207`, which returns
exactly `f"earlyln_K{K}_s{seed}"`. The JSON is therefore a file INSIDE
the attempt directory, and cannot exist when that directory does not —
so `dir=absent × JSON∈{parseable,unparseable}` is genuinely
unreachable, for both canonical values and both arms. **24 is correct.**

### 1.2 Totality of 0.1 — CONFIRMED, 24/24

Every one of the 24 valid states maps to **exactly one** outcome row.
Rows 4–6 and 7–9 are an exhaustive 2-way sub-case split
(`status=="COMPLETED"` / not) inside the single `parseable` dimension
value, so they do not enlarge the space; rows 10 and 11 each cover all
three `canonical_state` values via their `any` cell. No state is
uncovered and none matches twice. **PASS.**

### 1.3 Conservativeness — PASS with one MAJOR (KW8.6) and one MINOR

Per-state charge vs. true spend:

| States | Charge | Verdict |
|---|---|---|
| 1–3 (dir absent) | none | correct **if** nothing ran; see KW8.12 |
| 4–6 (parseable COMPLETED) | MEASURED `elapsed_s/3600` | **under-charges the live path** — see KW8.6 |
| 7–9 (parseable non-COMPLETED) | MEASURED `elapsed_s/3600` | same, KW8.6 |
| 10–11 (unparseable / no JSON) | FULL `charged_ceiling(arm)` | conservative ✓ |

No state charges a NEGATIVE amount and no state reduces
`realized_gpu_h`, so §R6/KW7.7's literal claim ("every reconstructed
row charges a positive amount, never a reduction") is true. That claim
is not, however, the premise H3 actually needs — KW8.6.

### 1.4 "No state re-opens budget" — FAILS at the CELL level (KW8.3)

Per-attempt, no state re-opens budget. Composed into cells, **30 of 200
cell states leave a `status=="COMPLETED"` canonical file on disk with
NO `COMPLETED` row in the reconstructed ledger**, because 0.2's
bootstrap guard is "0.1 appended ZERO rows" when the condition that
actually matters is "0.1 appended no COMPLETED row." **6 of those 30
are additionally derived NON-terminal and are re-dispatched** — real
GPU-h spent on a cell whose science already exists. See KW8.3.

### 1.5 Quarantine rule — fires exactly where claimed. PASS

0.0 quarantine-renames iff the canonical file is present AND
(unparseable OR parses non-`COMPLETED`). It fires in no other state,
and it is load-bearing for rows 5/8: after the rename, `canonical_path`
is free, so row 5's PROMOTE lands cleanly and G2's later pre-copy
`os.path.exists(canonical_path)` cannot trip on the corrupt file
(`:1028-1032`). Verified across all 200 compositions: **no case where
quarantine fires spuriously, and no case where a CORRUPT canonical
blocks a legitimate later copy.** PASS.

### 1.6 Bootstrap fallback — fires where claimed, but its GUARD is too narrow

0.2 fires exactly on "0.1 appended zero rows AND `canonical_state ≠
ABSENT`", as written. That is the rule working as specified. The defect
is in the specification of the guard, not its execution — KW8.3.

### 1.7 Promotion preempts every PERSISTENTLY-ABORTED path — CONFIRMED

Both promotion sites were checked for ordering:

- **Live path, step 2.1** (`:1113-1133`): runs *before* both branches of
  step 2.2, so a provably-`COMPLETED` attempt is promoted before any
  `CRASHED-RECOVERED` row can be written, at attempt 1 **or** attempt 2.
  A `PERSISTENTLY-ABORTED` derivation is therefore unreachable for a
  cell whose open attempt's own JSON reads `COMPLETED`. ✓
- **Reconstruction, rows 5/6** (`:1049-1050`): a parseable `COMPLETED`
  attempt JSON produces a `COMPLETED` row regardless of
  `canonical_state`, so no `CRASHED-RECOVERED` row is ever written for
  an attempt that demonstrably completed. ✓

Across the 200 compositions there is **no state in which a cell holding
a parseable `COMPLETED` attempt JSON derives to
`PERSISTENTLY-ABORTED`.** KW7.2 is genuinely discharged. PASS.

### 1.8 Composition with the resume-numbering sentence — PASS

Over all 200 compositions:

- **`resume > 2`: 0 cases.** The step-3 derivation closes it exactly as
  claimed — any cell carrying an `attempt_n:2` row is either
  `COMPLETED` or `PERSISTENTLY-ABORTED`, both terminal, both skipped in
  full. ✓
- **A numbered attempt is never re-dispatched: 0 violations.** Resume is
  `max(recorded attempt_n)+1`, and reconstruction writes each attempt
  directory's OWN `n` (no longer the Rev-5 fixed `attempt_n:2`), so the
  two numbering spaces are genuinely single-sourced. I3's precedence
  sentence does the work it claims. ✓
- **Right attempt dispatched:** correct in 194/200. The 6 exceptions are
  KW8.3's orphan cases, where resume dispatches attempt 2 on a cell
  that is already complete on disk.

---

## §2 SCOPE 2 — I4–I5 VALIDITY TEXT

13 payloads adjudicated (`scratchpad/vcheck.py`). Result:

| # | Payload | Expect | Actual |
|---|---|---|---|
| L1 | `COMPLETE`, 12/12 canonical, nothing disclosed | PASS | **PASS** ✓ |
| L2 | `INCOMPLETE-AT-K` (one seed CRASHED×2, 11 canonical, `incomplete_at_K=[30]`) | PASS | **PASS** ✓ |
| L3 | `COMPLETE-DEGRADED` sub-case (i) primary-retry-refused | PASS | **PASS** ✓ |
| L4 | `COMPLETE-DEGRADED` sub-case (ii) conditional-throttled | PASS | **PASS** ✓ |
| L5 | `EXHAUSTED-BUDGET`, realized 14.55, 9 canonical | PASS | **PASS** ✓ |
| L6 | `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE`, fraction 0.71 | PASS | **PASS** ✓ |
| A1 | R5 no-op (`COMPLETE`, `attempts=[]`, 0 canonical) | FAIL | **FAIL** ✓ |
| A2 | R6 near-miss #1 (`EXHAUSTED-BUDGET` @ 12 canonical) | FAIL | **FAIL** ✓ |
| A3 | R6 near-miss #2 (`STOPPED-BY-OPERATOR`, no stop-file) | FAIL | **FAIL** ✓ |
| A4 | **ENHANCED no-op** (`COMPLETE`, `attempts=[]`, `incomplete_at_K=[26,28,30]`) | FAIL | **PASS** ✗ |
| A5 | **ENHANCED no-op** via `interval_resolved_Ks=[26,28,30]` | FAIL | **PASS** ✗ |
| A6 | Suspect run mislabelled plain `EXHAUSTED-BUDGET` (fraction 0.93) | FAIL | **PASS** ✗ |
| A7 | Fabricated conditional arm (`qualifier_band`, no conditional evidence) | FAIL | **PASS** ✗ |

**The legitimate side of I4 is genuinely fixed.** All six §5-reportable
outcomes pass, including both cases `§A6-ADJUDICATION` named for
direct walk-through (L2 and L3), each by the evidence clause §R6 claims
for it. KW7.4's FATAL is discharged on that side. The three named
adversarial JSONs still fail, by the assertions §R6 names. **The
adversarial side has four holes**, three of them new this revision.

A fourteenth check, not expressible as a JSON payload, is the
`trigger["resolution"]` clause of universal assertion 6 — see KW8.2.

---

## §3 SCOPE 3 — THE THREE DISCLOSED SETTLED-SECTION CONTACTS

All three verified by direct diff against `HEAD~1`. All three are
**genuinely non-numeric edits**, exactly as disclosed.

1. **Unified-enum-table precedence sentence** (hunk `@@ -1297,11 +1393,19`).
   KW7.8 replaces `` `:2590`-region ``/`` `:2615-2616`-region `` line
   pointers with name-only citations; KW7.12 appends one clause
   extending outranking from enumerations to arithmetic (naming §R4's
   KW5.1 row). **The table itself is byte-untouched and the mechanism's
   logic is unchanged.** Does not disturb what was settled. ✓
2. **Crash-window walk table** (hunk `@@ -1207,12 +1298,17`). Only rows 1
   ("Before copy starts") and 2 ("Mid-copy") are rewritten, to state
   I2's promotion outcome. Rows 3 ("Between copy and fold") and 4
   ("After fold") — KW6.2's actual settled FATAL closure — are
   **byte-identical** in the diff. The rewrite was FORCED: left alone,
   the table would have contradicted the procedure two paragraphs
   above it. Correctly disclosed and correctly scoped. ✓
3. **H3 "True spend, worst case"** (hunk `@@ -1531,16 +1677,26`). The
   edit is exactly what is claimed: `L_final` → `R_N` (KW7.6) plus one
   KW7.7 monotonicity sentence. **Zero numeric change.** Re-verified
   independently: `13 × 0.0157 = 0.2041`; `15.00 + 0.2041 = 15.2041`.
   H3's number stands. **However, the KW7.7 sentence makes a claim that
   is not supported — see KW8.6.** The EDIT is non-numeric as
   disclosed; the CLAIM inside it is not sound.

**The two collateral fixes** (both disclosed in §R6's line-count
paragraph) also verify:

- Hunk `@@ -815,9 +815,11` — the dispatch loop's stale forward-pointer
  into the old step-3 wording ("no row with a DISPATCHED status") is
  repointed to I3's derived-terminal-state rule. A direct, in-scope
  consequence of I3. ✓
- Hunk `@@ -1309,10 +1413,10` — the `attempts[].status` table's
  `COMPLETED` and `CRASHED-RECOVERED` "Reachable via" cells, which
  named the retired fixed-`attempt_n:2` convention and the old
  single-branch reconstruction path. Both now name I1's per-attempt
  table and I2's promotion branch. A direct, in-scope consequence of
  I1/I3, non-numeric. ✓

---

## §4 SCOPE 4 — INTEGRITY

| Check | Result |
|---|---|
| `git diff HEAD~1 HEAD` file set | `EXPERIMENT_LOG.md` + the design file only. No code, no launches, no other artifacts. ✓ |
| Hunk count | **13**, every one attributable (see below). ✓ |
| Frozen `§A1-ADJUDICATION` → end-of-pre-Rev-6 content | **BYTE-IDENTICAL.** Current `:2749-3708` vs `HEAD~1:2496-3455`: `diff` returns empty; both md5 `9d07f2879e25de26cab512465ba8aa90`, both 960 lines. ✓ |
| §R6 claim: whole file before = `cee7e8136a63028ab420f30ab2769cf4` | **VERIFIED** against `git show HEAD~1`. ✓ |
| §R6 claim: live body before = `e92b343202888e9a948769cd5ff5843b` (2495 lines) | **VERIFIED.** ✓ |
| §R6 claim: live body after = `68440ddc8fc7408168daa8ce4ef2f090` (2748 lines) | **VERIFIED** against the on-disk file. (KW7.10's failure mode is not repeated.) ✓ |
| §R6 arithmetic `3455 = 2495 + 960`; `2748 − 2495 = 253`; `2748 + 960 = 3708` | **VERIFIED**, all three. ✓ |
| §R6 claim: frozen "→ EOF, after ... IDENTICAL" | **imprecise — KW8.10.** |

**Hunk attribution (13/13):** `@@ -1,6` status header (disclosed);
`@@ -815,9` collateral #1 (disclosed); `@@ -999,121` recovery steps 0–4
= KW7.1/7.2/7.3/7.15; `@@ -1207,12` crash-window contact #2 (disclosed);
`@@ -1297,11` precedence contact #1 = KW7.8/7.12 (disclosed);
`@@ -1309,10` collateral #2 (disclosed); `@@ -1356,12` G4 `COMPLETE` =
KW7.4; `@@ -1372,34` G4 `COMPLETE-DEGRADED` = KW7.4; `@@ -1407,16` G4
`STOPPED-BY-OPERATOR`/`EXHAUSTED-BUDGET` = KW7.11/7.5; `@@ -1431,8` G4
`EBSO` = KW7.5; `@@ -1531,16` H3 contact #3 = KW7.6/7.7 (disclosed);
`@@ -1871,45` `validity_check` = KW7.4/7.5/7.14; `@@ -3453,3` the
§A6/§R6 append. **No unattributed hunk this revision** — the defect
KW5.8/KW7.9 recorded for earlier rounds does not recur.

**KW7.9 / KW7.10 addendum dispositions — CONSISTENT with the ratified
precedent.** `§A6-ADJUDICATION` explicitly adopted the REPLACEMENT
justification for KW6.9 ("editing frozen `§R4` would destroy the
`§A1→EOF` MD5 identity"). Both KW7.9 and KW7.10 apply that same trade to
`§R5`, and the integrity instrument it protects is REAL — this round
used it, and it verified. KW7.10 additionally **declines to manufacture**
a corrected `§R4→§R5` live-body-after figure it cannot reconstruct,
directing the reader to `git show` instead. That is the correct call and
is the opposite of the failure it is recording. Both dispositions stand.

**Code citations — all verified by direct read, none from memory:**

| Cited | Actual | ✓ |
|---|---|---|
| `rn.atomic_write_json`, `run_ncr.py:105-109` | `def` at `:105`; `os.replace(tmp, path)` at `:109` | ✓ exact |
| `os.makedirs(outdir, exist_ok=True)`, `ncr_earlyln_scale.py:237` | line 237, and it does precede the `status=="COMPLETED"` resume-skip at `:241-246` | ✓ exact |
| `sys.exit(3)`, `:196-197` | `if rn.stop_requested(...)` at `:196`, `sys.exit(3)` at `:197` | ✓ exact |
| `_cell_gate1`, `:317-329` | `def _cell_gate1` at `:317` | ✓ exact |
| `cell_id` → `earlyln_K{K}_s{seed}` | `:206-207` | ✓ exact |

---

## §5 FINDINGS

### KW8.1 — FATAL. I4's `COMPLETE` OTHERWISE branch REOPENS the settled no-op hole.

**Quote** (`:2052-2061`): *"`run_status=="COMPLETE"` ⇒ **(§R6 I4)** if
`band["interval_resolved_Ks"]==[]` and `band["incomplete_at_K"] is
None`: exactly 12 files … OTHERWISE (some K is disclosed incomplete):
for every `K∈{26,28,30}` named in either field, that K's canonical-file
count in the primary directory is `<4`; for every K named in NEITHER
field, that K's canonical-file count is exactly `4`."*

**Evidence.** Payload A4: `run_status="COMPLETE"`, `attempts=[]`,
`realized_gpu_h_final=0.0`, **zero canonical files**, `smoke` all
`PASS`, `band={"label":"INCOMPLETE-AT-K","interval_resolved_Ks":[],
"incomplete_at_K":[26,28,30]}`, `trigger.resolution="TRIGGER-UNRESOLVED"`.
Traced through the assertions as written: U1 ✓ (`COMPLETE` is in the
accept-set); U2 ✓ (`0.0 ≤ 15.50`); U3 ✓ (`|0.0 − sum([])| = 0`); U4 ✓
(vacuous over an empty `attempts`); U5 ✓; U6 ✓ (`INCOMPLETE-AT-K` and
`TRIGGER-UNRESOLVED` are both enum members). Per-`run_status`: the
OTHERWISE branch fires; all three K's are named, all three counts are
`0 < 4` ✓; the "named in NEITHER field" clause is vacuous ✓.
**PASSES → routed to `completed/`.** A5 is the same attack through
`interval_resolved_Ks` and also passes.

**Why this is a regression, not a residual.** `§R5` H4 closed the no-op
hole and `§A6-ADJUDICATION` listed *"the no-op rejection"* among the
items SETTLED and excluded from all future rounds. `§R6`'s own trace
re-derives the rejection but on a premise that begs the question:
*"`band` carries no disclosed incompleteness **in a genuine no-op**, so
`COMPLETE`'s strict-12 clause fires."* A no-op report is not obliged to
be genuine — that is the entire point of the check. The OTHERWISE
branch, unlike every other branch in the block, contains **no
positive-evidence clause whatsoever**: it is satisfiable by an empty
filesystem and an empty ledger. `COMPLETE-DEGRADED` carries the
"every primary `(K,seed)` has ≥1 row" clause precisely to close this;
`COMPLETE`'s new branch was not given it.

**Discharge condition.** Add to `COMPLETE`'s OTHERWISE branch the two
clauses `COMPLETE-DEGRADED` already carries: (a) every primary
`(K,seed)`, `K∈{26,28,30}`, `seed∈{0..3}`, has ≥1 row in
`ledger.attempts`; and (b) the primary canonical count equals
`len({(a["K"],a["seed"]) for a in ledger.attempts if a["arm"]=="primary"
and a["status"]=="COMPLETED"})`. Verified against this round's own
suite: A4/A5 then FAIL on clause (a) while L2 still PASSES (12 cells ×
≥1 row; 11 canonical == 11 distinct `COMPLETED` primary pairs). Re-run
the full 13-payload suite to completion after the edit — do not
hand-check it.

### KW8.2 — FATAL. Universal assertion 6 rejects the design's own pre-registered `tie-break-min` outcome.

**Quote** (`:2033-2034`): *"`trigger["resolution"] in
{"unanimous","tie-break-min","TRIGGER-UNRESOLVED"}`"*.

**Quote, the producer** (`:563`, and identically at `:628` inside the
G5-amended pseudocode): *"`return ("DECIDED", min(K_trigs),
f"tie-break-min, candidates were {sorted(K_trigs)}")`"*.

**Evidence.** The pseudocode emits the formatted string
`'tie-break-min, candidates were [26, 28]'`. That string is **not** a
member of the asserted set — exact membership, no prefix match.
Executed: `'tie-break-min, candidates were [26, 28]' in
{'unanimous','tie-break-min','TRIGGER-UNRESOLVED'}` → `False`.
**Every run that resolves `K_trig` by tie-break FAILS universal
assertion 6 and is routed to `failed/` after the full ≤15 GPU-h has
been spent** — the identical failure shape as KW7.4, one assertion
lower.

**Not vacuous.** The tie-break branch is a pre-registered, enumerated
outcome: KW4.5's discharge rests on 11 enumerated configurations that
*"all resolve by construction under `min()`"*, and KW5.6's own 729-space
sweep reports candidate-set sizes `{1:612, 2:102, 3:14, 4:1}` — 102
two-candidate configurations reach the tie-break. (The 15 wide-tie
cases do not, being `INCOMPLETE-AT-K`/`TRIGGER-UNRESOLVED`, per KW5.6.)

**Why this round.** KW7.14 rewrote **this same assertion** this
revision, expanding the band placeholder to seven literals. The
sibling clause in the same sentence, edited in the same hunk
(`@@ -1871,45 +2027,142`), was not checked against its producer. It is
in scope: I4's mandate is the validity text quantified over the full
reportable outcome space, and a legitimate outcome is being rejected.

**Discharge condition.** Either (a) change the pseudocode at both
`:563` and `:628` to return the bare literal `"tie-break-min"` and move
the candidate list into the already-declared `trigger.candidate_set`
field (`:1378` — it exists and is currently unwritten on this path),
or (b) weaken assertion 6 to `resolution.split(",")[0] in {...}`.
(a) is strictly better: it keeps the assertion an exact enum check and
fills a schema field that is otherwise dead. Add a legitimate
`tie-break-min` payload to the in-text test list and run it.

### KW8.3 — MAJOR. 0.2's bootstrap guard is too narrow; an orphaned canonical re-opens budget and trips G2's ABORT-LOUDLY.

**Quote** (`:1076-1080`): *"**0.2 Cell-level bootstrap fallback** (the
residual case: canonical evidence survives with NEITHER attempt
directory present for that cell — e.g. an attempt tree pruned after
copying). For any `(K, seed)` where 0.1 appended ZERO rows and
`canonical_state≠ABSENT` at 0.0 …"*

**Evidence.** The guard is "0.1 appended ZERO rows"; the condition that
matters is "0.1 appended no `COMPLETED` row." Over the 200 cell
compositions, **30 states end with a `COMPLETED` canonical file on disk
and no `COMPLETED` row in the ledger.** Of those, **6 are additionally
derived NON-terminal**, so step 3 dispatches an attempt on a cell whose
science already exists:

| attempt 1 | attempt 2 | `canonical_state` | reconstructed rows | resume |
|---|---|---|---|---|
| present, parseable non-`COMPLETED` | dir absent | `OK` | `[(1, ABORTED-BUDGET, measured)]` | **attempt 2** |
| present, unparseable | dir absent | `OK` | `[(1, CRASHED-RECOVERED, ceiling)]` | **attempt 2** |
| present, JSON absent | dir absent | `OK` | `[(1, CRASHED-RECOVERED, ceiling)]` | **attempt 2** |

(× 2 arms = 6.) Each costs a real ceiling of GPU-h; on completion the
orchestrator reaches G2's pre-copy `os.path.exists(canonical_path)`
check (`:1240-1242`), finds the orphaned canonical, and **ABORTS
LOUDLY (raises)** mid-run.

**Second face — a now-false claim in G2.** `:1242-1249`: *"**This
exists-check can only fire on a genuine invariant violation** — the
dispatch loop above only ever advances a cell to attempt 2 from
`ABORTED-BUDGET-1`/`CRASHED-1` … the only way to trip it is an operator
re-running the orchestrator against a dirty, pre-existing results
directory."* Post-I1 that is false: reconstruction itself manufactures
a state in which the **legitimate** `ABORTED-BUDGET-1 →` attempt-2
advance trips the check.

**Mitigating, and why this is MAJOR not FATAL.** The raise is
self-healing under the mandated supervisor loop: the next restart's
step 2.1 reads the attempt's own `COMPLETED` JSON, takes its explicit
*"skip the copy if a canonical file already exists there and itself
parses `COMPLETED`"* path (`:1121-1123`), writes the `COMPLETED` row,
and proceeds. Net damage is one wasted (but charged) attempt plus one
crash cycle; the harvest is uncorrupted. The trigger also requires
foreign interference (this design never prunes attempt dirs). But the
design itself names attempt-tree pruning as the motivating scenario for
0.2, so it is a case the contract claims to handle and handles only
half of.

**Discharge condition.** Widen 0.2's guard from "0.1 appended ZERO
rows" to "0.1 appended no row with `status=="COMPLETED"`", and write
the bootstrap row at `attempt_n = max(recorded attempt_n, 0) + 1`
rather than a hard `attempt_n:1` (otherwise it collides with an
existing attempt-1 row). Then correct G2's "only way to trip it"
sentence to name reconstruction as a second producer, or state that the
widened 0.2 makes it unreachable again. Re-run the 200-state
composition after the edit.

### KW8.4 — MAJOR. `EXHAUSTED-BUDGET`'s assertion set is a strict SUBSET of `-SUSPECT-OVERCHARGE`'s, so a suspect run evades the escape hatch by mislabelling itself.

**Quote** (`:2089-2091`): *"`run_status=="EXHAUSTED-BUDGET-SUSPECT-
OVERCHARGE"` ⇒ the same three clauses as `EXHAUSTED-BUDGET` (§R6 I5)
**PLUS** `charged_vs_measured.ceiling_charged_fraction > 0.50`."*

**Evidence.** Payload A6: 9 `CRASHED-RECOVERED` rows at full ceiling +
3 primary first-attempt `GATE-REFUSED` rows, `realized=14.40`, 0
canonical files, `ceiling_charged_fraction=0.93`, claimed
`run_status="EXHAUSTED-BUDGET"`. All three `EXHAUSTED-BUDGET` clauses
hold (`14.40 > 13.80` ✓, `0 < 12` ✓, ≥1 primary first-attempt
`GATE-REFUSED` ✓) and nothing tests the fraction. **PASSES** as plain
`EXHAUSTED-BUDGET` → routed to `completed/` **without** the
`-SUSPECT-OVERCHARGE` label's binding protection that *"resubmission is
NEVER automatic: only an explicit coordinator adjudication, with a
fresh ledger, may re-run the affected cells"* (`:1584-1585`). The pool's
ordinary "resubmitting resumes cleanly" advice then applies to a ledger
that is 93% environment-fault noise — precisely the failure KW6.8
identified and H5 exists to prevent.

**In scope.** I5's own mandate is *"label-vs-disk consistency … a label
contradicting a complete disk state is rejected."* This is the exact
mirror of KW7.5, which I5 fixed in the other direction; the two labels
are still not provably disjoint on disk.

**Discharge condition.** Add
`charged_vs_measured.ceiling_charged_fraction <= 0.50` to
`EXHAUSTED-BUDGET`'s branch. The two branches then partition the
`>13.80`/`<12`/`GATE-REFUSED` disk state exactly, and A6 fails while L5
(fraction 0.0) and L6 (fraction 0.71) both still pass.

### KW8.5 — MAJOR. No disk-evidence assertion covers the CONDITIONAL arm at all.

**Quote** (`:2046-2047`): *"Per-`run_status` (exactly one branch fires
… closes KW6.7's no-op hole by construction; rewritten §R6 I4/I5 —
quantified over the FULL pre-registered §5 outcome space …)"*.

**Evidence.** Every per-`run_status` branch counts files in the
**PRIMARY** canonical directory only. Nothing anywhere in
`validity_check` — universal or per-branch — reads
`/home/nvidia/ncr/results_kwall_characterization_160k/` (`:1325`), the
conditional arm's own canonical tree. Payload A7: a genuine 12/12
`COMPLETE` primary run whose report additionally carries
`conditional={"launched":true,"per_seed":[],"qualifier_band":
"SLOW-CONVERGENCE-AT-160K"}` with **zero conditional canonical files and
zero conditional ledger rows** — **PASSES**. The three 160K qualifier
bands are §5-reportable outcomes (`:2406-2422`), and the conditional arm
is priced at up to **9.248 of the ≤15 GPU-h** (`:1802`). The claim that
the block is "quantified over the FULL pre-registered §5 outcome space"
is therefore over-broad: it is quantified over the `run_status` × primary
outcome space only.

**Discharge condition.** Add one universal assertion: if
`conditional.launched` is `true`, then `conditional.qualifier_band` is
non-`null` **and** the conditional canonical directory holds a file
count equal to `len({(a["K"],a["seed"]) for a in ledger.attempts if
a["arm"]=="conditional" and a["status"]=="COMPLETED"})`; if
`conditional.launched` is `false`, then `qualifier_band is None` and the
conditional directory is empty. Or, if the coordinator rules conditional
evidence out of `validity_check`'s remit, say so explicitly and narrow
the "FULL pre-registered §5 outcome space" wording — but the ≤9.25
GPU-h arm should not be the one half of the study with no disk check.

### KW8.6 — MAJOR. KW7.7's H3 contact sentence claims a premise I1 does not establish, and I1 in fact weakens a different H3 premise.

**Quote, the new sentence** (`:1694-1698`): *"**This bound assumes the
ledger only ever accumulates across reconstructions (§R6 KW7.7); §R6
I1's per-attempt reconstruction (above) establishes exactly that —
every reconstructed row charges a positive amount, never a reduction —
so the assumption now holds by construction, not by disclaimer.**"*

**Quote, the premise H3 actually uses** (`:1699-1703`): *"Every LEAKING
attempt is, by definition, one the ledger charged its full
`charged_ceiling ≥ 1.20` for … so at most `⌊15.0157/1.20⌋ = 12`
attempts can have leaked."*

**Evidence, two faces.**

(a) **The `≥1.20`-per-leaking-attempt premise is now false.** Under Rev 5,
`§A5-ADJUDICATION` H1 charged **full ceiling** on every reconstruction
path (*"canonical files ⇒ `COMPLETED` rows charged at the full ceiling
(measured spend unavailable)"*). I1 changed that: rows 4–9 (6 of the 12
core triples), 0.2's `OK` bootstrap, and step 2.1's promotion now all
charge **MEASURED** `elapsed_s/3600`, which for a completing primary
cell is ≈0.5h, not 1.20h. The count of attempts that can leak inside the
budget is therefore no longer bounded by 12, and the `(1+12)` factor in
`T ≤ 15.00 + (1+12)(0.0157) = 15.2041` is no longer derived.

(b) **A new per-row leak is introduced.** The two timers are different
quantities. The live fold charges the ORCHESTRATOR's wall clock,
`attempt_elapsed_h=(t1-t0)/3600` spanning the whole `subprocess.run`
(`:833-835`). Reconstruction charges the SUBPROCESS's own `elapsed_s`,
whose `t0 = time.time()` is set at `ncr_earlyln_scale.py:257` — **after**
`os.makedirs` (`:237`), `nt.claim_config`, `torch.manual_seed`,
`NCREarlyLNModel(...).to(device)` (CUDA init + model build), and the
`git_commit`/`n_params` record assembly (`:247-256`). So a
measured-reconstructed row charges strictly LESS than the live path
would for the identical attempt, by the interpreter/CUDA-init/model-build
term. That term is exactly the one KW5.7's discharge assigned to the
**stated** `0.30h` supervisor margin — as ONE attempt's worth
(*"the unpriced process-startup term … the thing the STATED (not
derived) `0.30h` supervisor margin carries"*). Post-I1 the margin must
carry it once per reconstructed attempt, up to 32 attempts.

**Magnitude, honestly.** The leak is small and the run stays inside the
declared `15.50h`: the term is bounded by KW5.7's own residual
(`≈0.016–0.021h` true single-attempt tail vs `0.0157h` priced, so
`≈0.0003–0.0053h` each), and reconstruction rebuilds from disk rather
than accumulating, so it is one-time per attempt, not per restart. But
the **stated tight bound `15.2041h` is no longer the true-spend bound
post-I1**, and the sentence added this revision asserts the opposite —
it states that I1 *establishes* the premise when I1 is what weakened it.
Positivity of a row ("never a reduction") is a monotonicity property of
the ledger; it says nothing about the charge covering true spend.

**Discharge condition.** Replace the KW7.7 sentence with an honest one:
state that I1 trades conservativeness for accuracy on parseable-JSON
rows; re-derive the leak accounting with two classes (ceiling-charged
rows leak `≤ τ` and are still `≥1.20`-bounded, so `≤12` of them;
measured-charged rows leak only the process-startup term `s`, bounded by
`≤ 32` occurrences); and either state the resulting `T ≤ 15.2041 + 32s`
figure with `s` bounded from KW5.7's own range, or declare the term
absorbed by the `0.30h` margin **with the multiplicity made explicit**.
No numeric input changes; the declared `15.00/15.20/15.50` ceilings are
not affected. This is a bounded, one-paragraph fix — but it must be
made, because the current sentence tells a build implementer the bound
is safe by construction when it is safe by margin.

### KW8.7 — MINOR. The derived cell state is not a function: `COMPLETED` and `PERSISTENTLY-ABORTED` both derive in 24/200 states.

**Quote** (`:1424-1426`): *"`PERSISTENTLY-ABORTED` iff the cell's
attempt-2 row exists and is non-`COMPLETED`, OR its attempt-1 row is
non-`COMPLETED`, no attempt-2 row exists, and the retry gate closed
it."*

**Evidence.** Any cell whose attempt-1 row is `COMPLETED` and whose
attempt-2 row exists and is not (e.g. attempt 1 parseable-`COMPLETED`,
attempt 2 unparseable) satisfies BOTH the `COMPLETED` derivation and the
`PERSISTENTLY-ABORTED` derivation. 24 of 200 compositions. No dispatch
hazard — both are terminal, so the cell is skipped in full either way,
and `resume` is never reached. Two residual effects: (i) a build
implementer's derived-state function may return either answer; (ii) the
spurious `PERSISTENTLY-ABORTED`-deriving row satisfies
`COMPLETE-DEGRADED`'s throttle-evidence clause (`:2070-2072`), which
could let a run label itself `COMPLETE-DEGRADED` with nothing actually
throttled. I2 makes the state unreachable via this design's own dispatch
path going forward, so this is foreign-state hygiene.

**Discharge.** One clause: `COMPLETED` takes precedence — a cell with any
`COMPLETED` row is `COMPLETED`, never `PERSISTENTLY-ABORTED`.

### KW8.8 — MINOR. `canonical_state` is snapshotted once at 0.0 and never refreshed by 0.1's own PROMOTE writes.

**Quote** (`:1037-1039`): *"For each attempt directory `n∈{1,2}`, cross
this attempt's own evidence against the shared `canonical_state`
(0.0)"*.

**Evidence.** Attempt 1 taking row 6 (PROMOTE, `canonical_state=ABSENT`)
writes a canonical file, but attempt 2 is still evaluated against the
stale `ABSENT` snapshot and promotes again, overwriting it. 6 of 200
compositions produce a double-promote and two `COMPLETED` rows for one
cell. Benign: charging both is arithmetically correct (both attempts
really ran), and every downstream consumer counts **distinct
`(K,seed)` pairs** — `:1480-1481` and `:2067-2068` both use a set — so
the G2+H2 identity survives. Worth one sentence stating the snapshot is
deliberate and the set-based counting is what makes it safe.

### KW8.9 — MINOR. Rows 7–9 are undefined for a parseable JSON with no `status` key.

**Quote** (`:1051`): *"`status:<the JSON's own status>`"*. If a
parseable JSON lacks the key, the row's `status` is undefined and the
`attempts[].status` enum (6 values, `:1414-1421`) is violated. 0.0
handles the identical case for the canonical correctly by quarantining
it (`status!="COMPLETED"` is true for a missing key). **Discharge:** one
clause — a parseable JSON whose `status` is missing or not in the enum
is treated as unparseable (row 10, full ceiling).

### KW8.10 — MINOR. §R6's frozen-range MD5 row is imprecise as stated.

**Quote:** *"| Frozen range `## §A1-ADJUDICATION` → EOF, after |
`9d07f2879e25de26cab512465ba8aa90` (960 lines) — **IDENTICAL,
independently reproduced** |"*. Against the finished file this is false:
`§A1-ADJUDICATION` → EOF is now `:2749-3879`, 1131 lines, md5
`0abe0d617ed8a8f2b601143fafc4389c` — because §R6 appended §A6/§R6 inside
that range. The claim is TRUE for `§A1-ADJUDICATION` → end-of-pre-Rev-6
content (`:2749-3708`), which this round verified byte-identical by
`diff` as well as md5. `§R5`'s own wording was precise here (*"the byte
range from `## §A1-ADJUDICATION` to the end of the pre-Rev-5 file"*);
§R6 regressed it while fixing a different MD5 defect. **Discharge:**
restore §R5's phrasing. Substance is sound — no action beyond wording.

### KW8.11 — MINOR. "presented above as 12 rows" — the table has 11 physical rows.

**Quote** (`:1062-1063`): *"= 24 valid, presented above as 12 rows × the
2-way `arm` ceiling-value split"*. The markdown table has **11** rows.
12 is the count of core `(dir, JSON, canonical)` triples, which is the
figure the arithmetic actually needs and which is correct (`12 × 2 = 24`).
The mismatch arises because rows 10/11 merge three canonical values each
while the `parseable` triples are split into two sub-case rows each
(3 + 6 + 1 + 1 = 11). **Discharge:** say "12 core `(dir, JSON,
canonical)` states, rendered as 11 table rows."

### KW8.12 — MINOR. Rows 1–3 charge zero for a window in which real GPU time can be spent.

KW7.15's premise is **verified exact** — `os.makedirs(outdir,
exist_ok=True)` is at `ncr_earlyln_scale.py:237` and does precede both
training and the resume-skip (`:241-246`). But the subprocess does real
work before reaching it (interpreter start, `torch`/CUDA import, arg
parsing), so a crash in that window leaves no attempt directory and is
charged `0`. The design's softened wording (*"left no evidence this
design's dispatch path can produce"*) is epistemically correct and this
is the same process-startup term as KW8.6(b) — it should be priced in
the same paragraph, not separately.

---

## §6 GATE SUMMARY

| Scope item | Verdict |
|---|---|
| 1a. 24-state space re-derived independently (36−12=24; impossibility argument) | **PASS** |
| 1b. 0.1 totality — exactly one outcome row per state, 24/24 | **PASS** |
| 1c. Conservativeness per state | **PARTIAL** — KW8.6, KW8.12 |
| 1d. No state re-opens budget (cell-level composition) | **FAIL** — KW8.3 |
| 1e. Quarantine fires only where claimed | **PASS** |
| 1f. Bootstrap fires only where claimed | **PASS as executed; guard mis-specified** — KW8.3 |
| 1g. Promotion preempts every `PERSISTENTLY-ABORTED` path | **PASS** |
| 1h. Composition with resume numbering (never >2, never re-run) | **PASS** (194/200 dispatch correctly; 6 = KW8.3) |
| 2a. Every §5-reportable outcome accepted | **FAIL** — KW8.2 (`tie-break-min`) |
| 2b. Every adversarial payload rejected | **FAIL** — KW8.1, KW8.4, KW8.5 |
| 3. Three disclosed settled-section contacts non-numeric & non-disturbing | **PASS on the edits**; KW8.6 on contact #3's claim |
| 3b. Two collateral fixes | **PASS** |
| 4. Integrity (frozen block, hunks, addenda, code citations) | **PASS** (KW8.10 wording only) |

**VERDICT: REV-REQUIRED.** Forcing findings: **KW8.1** and **KW8.2**.

---

## §7 WHAT REV 7 MUST DO (binding disposition proposal)

- **J1 (KW8.1):** give `COMPLETE`'s OTHERWISE branch the two positive-
  evidence clauses `COMPLETE-DEGRADED` already carries. Re-run the
  13-payload suite to completion, including a forced-fail negative test.
- **J2 (KW8.2):** return the bare literal `"tie-break-min"` at `:563`
  and `:628`, moving the candidate list into `trigger.candidate_set`;
  add a `tie-break-min` payload to the in-text test list.
- **J3 (KW8.3):** widen 0.2's guard to "no `COMPLETED` row appended",
  write the bootstrap row at `max(attempt_n)+1`, and correct G2's
  "only way to trip it" sentence.
- **J4 (KW8.4):** add `ceiling_charged_fraction <= 0.50` to
  `EXHAUSTED-BUDGET`.
- **J5 (KW8.5):** add the conditional-arm disk-evidence assertion, or
  narrow the "FULL §5 outcome space" claim explicitly.
- **J6 (KW8.6):** replace the KW7.7 sentence with the honest two-class
  leak accounting. No declared ceiling changes.
- **J7:** the six MINORs (KW8.7–KW8.12), each a one-clause fix.

**Round 8 scope, recommended (binding on the next audit):** J1–J6 ONLY,
plus a re-run of this round's 13-payload suite and 200-state
composition. Everything verified PASS in §6 is settled and excluded:
the 24-state derivation, 0.1 totality, the quarantine rule, promotion
preemption, resume numbering, the three settled-section contacts, the
two collateral fixes, and integrity.

---

## §8 SCRIPTS

Both transcriptions live in this session's scratchpad and should be
re-run verbatim by Rev 7's audit rather than re-derived:

- `vcheck.py` — `validity_check` verbatim (`:2006-2091`) + 13 payloads.
- `recon.py` — 0.0/0.1/0.2 + step-3 resume verbatim (`:1004-1207`),
  24-state and 200-composition walks.

---

## §9 BINDING BUILD CHARTER (restated, NOT yet released)

Recorded here so Rev 7 can be adjudicated and released in one pass.
**This charter is not in force until a round clears the design.**

1. **R5's conditional build-release checklist** (`NCR_KWALL_ATTACK_R5.md`
   §9), accepted by `§A5-ADJUDICATION`, in full.
2. **R6's five additions**, adopted by `§A6-ADJUDICATION`: the
   per-attempt-dir reconstruction contract; the Rev-6 validity
   one-liner; red-team items i–xiv; **every negative test RUN TO
   COMPLETION, never merely written**; the `os.makedirs` build
   assertion.
3. **The 3 micro-smokes (K=26/28/30) pass before queue-eligibility**
   (`:2204-2209`), with the orchestrator's own startup re-read
   populating `report.smoke` (`:2210-2225`).
4. **R7 additions (new, binding):** (a) `vcheck.py`'s 13-payload suite
   is wired as a build-stage unit test of the real `validity_check`,
   with A1–A7 as forced-fail negatives; (b) `recon.py`'s 24-state and
   200-composition walks are wired as unit tests of the real
   reconstruction function, asserting `resume ≤ 2`, no orphaned
   canonical, and exactly one row per attempt directory; (c) the
   build asserts the orchestrator's emitted `trigger["resolution"]`
   is a bare enum literal; (d) the build asserts the conditional
   canonical directory is disjoint from the primary one
   (`results_kwall_characterization_160k/` vs
   `results_kwall_characterization/`) before the first conditional
   dispatch; (e) KW2.7's on-box `fallback_pool/`/`claimed/` sweep,
   still deferred and still listed.
5. **Unchanged ceilings:** `15.00` hard gate / `12.00` retry gate /
   `15.20` disclosed / `15.50` declared `gpu_h_estimate`, `+0.15`
   micro-smokes outside the ledger.
