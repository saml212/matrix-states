# NCR K-WALL — NARROW AUDIT ROUND 6 (crash-path / enums / H3 arithmetic)

**STATUS: REV-REQUIRED. 2 FATAL / 3 MAJOR / 10 MINOR.**
Scope-item tally — **(1) crash-path composition: FAIL**;
**(2) unified enum table + `validity_check`: FAIL**;
**(3) H3 arithmetic: PASS** (number and provenance both verify; one
prose defect and one stated dependency).
Integrity: **PASS** (frozen block byte-identical, line count reconciles,
diff hunks 20/22 inside the claimed edit list).

Target read: `matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md`
(3406 lines), header **"DRAFT-R5 — POST-AUDIT-5, AWAITING NARROW AUDIT
ROUND 6 (not build-released, not queue-eligible)"** — **matches the
expected header exactly, no mismatch to report.**

Scope honoured: the 125-outcome partition, the trigger's 1000-vector /
11-configuration sweeps, and the pricing chain were **not re-run and
not re-litigated**. Where a pricing figure is mentioned below it is
only to confirm that a Rev-5 edit is what §R5 claims it is.

---

## §0 PRE-CHECK — §R5's CLAIMED EDIT LIST vs THE ACTUAL DIFF

`git diff HEAD~1 HEAD` on the design file = **22 hunks** (840
insertions / 167 deletions on this file; the +14 in the stat line is
`EXPERIMENT_LOG.md`). Net +673 lines; `2733 + 673 = 3406` ✓.

| Hunks | Location | In §R5's "Where fixed" column? |
|---|---|---|
| 1 | status header | yes (explicit) |
| 17 | §4 (421–1983) | yes — every one maps to an H1–H6 "Where" entry |
| 1 | §6 "Own cost ceiling" bullet (→2285) | yes (H3) |
| **2** | **§6 resource/placement red-team (→2320, →2350)** | **NO — see KW7.6** |
| 1 | §R5 append | yes |

§5 (1984–2251) and §7 (2401–2495) are **untouched this revision** —
this is the round that finally leaves them alone, and it closes the
KW6.9 "unattributed §7 hunk" recurrence risk for Rev 5 specifically.

**Excluded-territory contact.** §R5 disclosed three edits touching
partition/trigger/pricing text; I verify all three are what they claim:

1. **H3 / `T≤15.2041`** — a pure arithmetic add-on built from `τ=0.0157`,
   the `1.20h` primary ceiling and the `15.00h` cap. No pricing INPUT
   changed; it sits in "Worst-case bound," not the ceiling-reference or
   margin-ratio subsections. ✓
2. **KW6.11 `0.5106→0.5105`** (`:784`) — one operand digit; the displayed
   product `0.9882` is byte-identical before and after. ✓ (Non-finding
   note, offered only so it is not re-raised later: `0.5105×1.9355`
   evaluates to `0.98807`, and `0.5536×1.9355` to `1.07152`, while the
   table shows `0.9882`/`1.0716` — the consistent upward difference is
   the signature of products computed from UNROUNDED operands, which is
   what §R5 says was executed. Excluded territory; **not** re-derived,
   **not** a finding.)
3. **KW6.12 comment fix** (`:552`, `:617`) — `# 1, 2, or 4 candidate` →
   `# 1, 2, 4, or 8 candidate ... (2^k for k AMBIGUOUS K's)`, both
   occurrences, code semantics untouched. ✓

A **fourth** live-body edit physically lands in that territory and is
NOT in the three-item contact disclosure: the KW6.16 attribution fix at
`:698-700` (`KW5.13` → `KW5.6`). It *is* claimed in the §R5 table
("§4 11-config table closing paragraph"), it changes zero numbers, and
I verified the new attribution against §R4's own rows: §R4's KW5.6 row
carries the lead-sentence scope restatement, §R4's KW5.13 row is about
the resolution-state-table collapse. **The reattribution is correct.**
Recorded here only so the contact list is complete.

---

## §1 SCOPE ITEM 1 — CRASH-PATH COMPOSITION: **FAIL**

Full walk of the revised state machine (`:806-924` dispatch loop,
`:957-1137` ledger persistence + recovery, `:1182-1217` G2 pair +
crash-window table).

### Windows that verify clean

| Crash window | Recovery outcome | ledger ≥ true spend? | budget re-opens? | cell lost / double-run? |
|---|---|---|---|---|
| Before the open record | no row, no charge, cell available | n/a (0 spend) | no | no |
| Between open record and `subprocess.run` | dangling ⇒ `CRASHED-RECOVERED`, full ceiling | **over**-charges ~1.20h for ~0h | no | no (retry available) |
| Mid-attempt | dangling, attempt dir exists ⇒ `CRASHED-RECOVERED`, full ceiling | under by ≤ `τ` only — exactly H3's priced leak | no | no |
| Mid-copy (`.tmp`, not renamed) | indistinguishable from "before copy" ⇒ `CRASHED-RECOVERED` | as above | no | see KW7.2 |
| Between rename and fold | canonical + dangling ⇒ **`COMPLETED`** row, full ceiling | over-charges | no | no — this is H2's fix, and it works |
| After fold | nothing dangling | exact | no | no |
| During recovery | step 0 is a pure function of disk ⇒ idempotent; step 2's row-append and `open_attempt=null` share one atomic persist (parallel to the fold's own wording at `:878-887`) | conservative | no | no |
| Double restart | each cycle closes exactly one dangling record before any gate check; attempt 1 rowed ⇒ walk advances to attempt 2 | monotone | no | no |

**Verified independently against the real code, not assumed:**

- `rn.atomic_write_json` is at `matrix-thinking/ncr/run_ncr.py:105-109`
  exactly as cited (`tmp = path + ".tmp"` … `os.replace(tmp, path)`).
  The citation is line-exact.
- **The mid-copy `.tmp` really is invisible to harvest.**
  `discover_seeds_by_K` globs `earlyln_K{K}_s*.json` and then re-filters
  with `re.search(rf"earlyln_K{K}_s(\d+)\.json$", p)`
  (`ncr_earlyln_scale.py:364-368`). A file ending `.json.tmp` matches
  neither. §R5's claim is CORRECT.
- **Mid-attempt crashes always leave disk evidence.**
  `os.makedirs(outdir, exist_ok=True)` is at `:237`, before training and
  before the resume-skip — so the attempt directory exists from the
  first second of every dispatched attempt. Reconstruction step 0.2 is
  therefore never evidence-blind for a normally-dispatched attempt.
  (This is load-bearing for H1 and is nowhere stated in the design;
  worth adding as a one-line premise.)
- Every other code citation in the touched text checks out: `sys.exit(3)`
  at `:196-197`, the `ABORTED-BUDGET` early-return write at `:262-266`
  (`atomic_write_json` at `:265`), the `COMPLETED` write at `:307`,
  `gpu_h` assigned only on the completed path at `:304`, the
  `status=="COMPLETED"` resume-skip at `:243-245`, `harvest()`'s
  non-recursive glob at `:358-380`.

### KW7.1 — **FATAL.** "Conservative reconstruction" is neither conservative nor total; the sentence "No path through this step re-opens budget" is false, and §A5's H1 directive is not discharged.

> **No path through this step re-opens budget:** every row it can
> possibly write charges a real, positive ceiling amount; the only way
> to add zero charge is to add no row, which happens only when there is
> nothing on disk to be conservative ABOUT. (`:1039-1043`)

The justification is a non-sequitur — positivity of each individual row
says nothing about whether the SUM is ≥ the ledger value it replaces.
Step 0.3 **replaces** `realized_gpu_h` outright ("`realized_gpu_h` = the
sum of every reconstructed row's `elapsed_h`", `:1032-1033`), so any
shortfall is a live budget re-open. Three concrete faces:

**(a) One row per CELL, not per ATTEMPT — under-charges every cell that
consumed two attempts.**

> append EXACTLY ONE row (never one per attempt-dir found — there is no
> way to know from a corrupted ledger how many independent runs it
> recorded, so charging once, at the full ceiling, is the conservative
> choice) (`:1019-1023`)

The stated premise is **directly contradicted by this design's own
naming convention**, quoted in the same sentence: the directories are
`K{K}_s{seed}_attempt{1,2}/`. The attempt number is encoded in the
name, and the design caps attempts at 2. Counting them is trivial.
Worked case, entirely inside the design's own rules: six primary cells
each `ABORTED-BUDGET` at attempt 1 and again at attempt 2 (the
pre-registered `PERSISTENTLY-ABORTED` path) ⇒ true spend ≈ `6×2×1.20 =
14.40h`, twelve `attempt{1,2}` directories on disk, zero canonical
files. Reconstruction writes **six** rows ⇒ `realized_gpu_h = 7.20h`.
**7.2 GPU-h of already-spent budget is handed back**, and the hard gate
will re-admit against it. The same under-charge hits every
attempt1-aborted/attempt2-completed cell (charged 1 ceiling, spent ~2).

**(b) The conditional arm can vanish entirely.** The reconstruction
universe is gated on the canonical directory:

> For every (K, seed) in the full cell order (primary 12, plus the
> conditional 4 **if that arm's canonical directory exists**) (`:1006-1008`)

The conditional canonical directory is created by the ORCHESTRATOR at
copy time. If the conditional arm was dispatched and crashed before its
first `COMPLETED` copy, that directory does not exist, its
`attempt{n}/` dirs are never examined, and up to `4×2.32 = 9.28 GPU-h`
of real spend is erased from the ledger.

**(c) The rule is NOT total.** Sub-step 1 fires only "if a canonical-path
file exists **and parses with `status=="COMPLETED"`**" (`:1008`);
sub-step 2 fires only for a cell with an attempt dir "**but NO canonical
file**" (`:1018-1019`). A cell whose canonical file EXISTS but is
unparseable, truncated, or carries a non-`COMPLETED` status matches
neither branch and gets **no row and no defined outcome**. Worse, the
fall-through leaves the cell marked available, so when its re-run
completes, G2's `os.path.exists(canonical_path)` check (`:1149-1151`)
**aborts the whole orchestrator loudly, mid-run**. Step 0 exists
precisely for "a manually-edited file, a filesystem fault, a foreign
process" (`:996-998`) — every one of those causes can hit a canonical
JSON as easily as it hits the ledger.

**Discharge condition (one edit closes all three).** Reconstruct
**per attempt DIRECTORY, not per cell**: for every `(K, seed)` and every
`attempt{n}/` directory present in EITHER tree (gate the conditional arm
on the presence of its ARCHIVAL directory, never its canonical one),
append one row at `charged_ceiling(arm)`, `ceiling_charged:true`, with
`status` = `COMPLETED` if that attempt's canonical file is present and
parses `COMPLETED`, else `CRASHED-RECOVERED`; and add the missing
else-branch — a canonical file that is present but unparseable or
non-`COMPLETED` is evidence of a dispatched attempt (charge it) and must
be quarantined (renamed aside) or the run aborted at reconstruction
time, never left to trip G2's exists-check mid-wave. Then replace the
false claim at `:1039-1043` with the argument that actually holds:
reconstruction charges ≥ one full ceiling per attempt for which disk
evidence exists, and every dispatched attempt provably leaves an
attempt directory (`ncr_earlyln_scale.py:237`).

### KW7.2 — **MAJOR.** Recovery discards a provably-`COMPLETED` attempt, and for an attempt-2 crash there is no retry left — the cell is lost and the harvest denominator shrinks.

The genuine-crash branch checks the CANONICAL path only (`:1048-1050`)
and never looks at the attempt-dir JSON, which for the "before copy"
window already reads `COMPLETED`. The disclosure:

> if the subprocess's OWN archival attempt-dir JSON in fact reads
> `COMPLETED` in this window — the "before copy" case — that completed
> science is not reused; **the retry re-runs the cell from scratch.**
> Wasteful, never unsafe (`:1075-1080`)

**The premise is false when the crash lands on attempt 2.** There is no
attempt 3. The `CRASHED-RECOVERED` row is written at `attempt_n=2`, the
derivation rule at `:904-907` fires (`attempt-2 row non-COMPLETED ⇒
PERSISTENTLY-ABORTED`), the cell is terminal, no canonical file is ever
produced — and `harvest()` reads that K at `n_completed=3`, pushing it
into interval logic or `INCOMPLETE-AT-K`. That is KW6.2's silent
denominator shrink surviving in a narrower window: a cell that
demonstrably completed is scored as never having completed. The same
hole exists in reconstruction step 0.2, which likewise ignores the
attempt-dir JSON's own status.

**Discharge.** In both the recovery-closure branch and reconstruction
step 0.2, read `.../K{K}_s{seed}_attempt{n}/earlyln_K{K}_s{seed}.json`
(deterministic from the fields `open_attempt` already carries). If it
parses `COMPLETED`, perform the canonical copy now and write a
`COMPLETED` row — this closes the "before copy" and "mid-copy" windows
into the same PROOF-OF-COMPLETION treatment "between copy and fold"
already gets, eliminates the disclosed waste, and removes the
denominator loss. If the design prefers to keep the current behaviour,
the disclosure must be corrected to state that an attempt-2 crash in
this window converts a completed cell into `PERSISTENTLY-ABORTED` and
can move the study's band to `INCOMPLETE-AT-K`.

### KW7.3 — **MAJOR.** The cell-level resume rule and reconstruction's `attempt_n=2` convention contradict each other; the wrong reading voids H1's "no retry credit" guarantee.

Reconstruction writes its `CRASHED-RECOVERED` row at `attempt_n:2`
deliberately, so the cell "derives TERMINAL immediately — **no retry
credit** — which is what guarantees reconstruction can only ever ADD
charged ceiling to the ledger, never re-open a dispatch slot"
(`:1025-1031`). But the resume rule that must honour that is written
attempt-wise and contradicts itself inside one sentence:

> Once every dangling record is closed and every already-terminal
> **cell** is skipped, dispatch resumes from the first **cell/attempt
> with no ledger row** — normal operation from here. (`:1114-1116`)

After reconstruction, such a cell has an attempt-2 row and **no
attempt-1 row**. Clause A (skip terminal cells) skips it; clause B
(resume at the first cell/attempt with no row) dispatches attempt 1 at
a fresh full ceiling. Step 3's own "never re-gated" test is likewise
attempt-scoped (`:1092-1098`), so it does not adjudicate. This document
explicitly requires a build-stage implementer to write the orchestrator
"from this design alone" (`:795-796`) — an ambiguity whose wrong
resolution re-opens a dispatch slot is not an implementer's judgement
call.

**Discharge.** State the skip CELL-wise and normatively: a cell whose
DERIVED state is terminal (`COMPLETED`, `PERSISTENTLY-ABORTED`,
`STOPPED-BY-OPERATOR`) is skipped in full regardless of which
`attempt_n` rows exist. KW7.1's per-attempt-directory reconstruction
also removes the row-less-attempt-1 state that creates the ambiguity —
one fix, two findings.

---

## §2 SCOPE ITEM 2 — UNIFIED ENUM TABLE + `validity_check`: **FAIL**

### 2a — The precedence sentence: mechanism sound, pointers stale

The three-way `attempts[].status` disagreement KW6.5 found is genuinely
resolved *in the live body*: the output-JSON schema (`:1270-1272`), the
resume rule's terminal test (`:1093-1097`) and the unified table
(`:1310-1317`) now carry the **same 6 values** — `COMPLETED`,
`ABORTED-BUDGET`, `CRASHED`, `CRASHED-RECOVERED`, `GATE-REFUSED`,
`STOPPED-BY-OPERATOR` — with `PERSISTENTLY-ABORTED` correctly demoted to
a derived CELL state with a stated derivation rule. **Named frozen
conflicts, verified by reading them:** §R4's KW5.3 row (current
`:3062`) enumerates only "all four reachable non-`COMPLETED` attempt
states," omitting `GATE-REFUSED`; §R4's "numbers that moved" line
(current `:3085-3088`) agrees on the 6 attempt statuses but says
`run_status`'s reachable values are **4**, now 5. Both are outranked,
not edited. **The live body never relies on either**: the only §R4
citations inside §1–§7 are `:700` (an attribution pointer) and `:1300`
(the precedence sentence itself); §R1 is cited twice and explicitly
called "frozen and non-operative" (`:1917`, `:1929`). The precedence
mechanism holds. See KW7.7 for the pointer defect.

### 2b — Reachable terminal states × the table: exhaustive, one gap

All eleven reachable terminal attempt outcomes (2 gate refusals, 4
classify branches incl. the new `(exit 0, no JSON)` arm, 2 recovery
branches, 2 reconstruction branches, 1 operator stop) map into the 6
values with no leftovers, and the 9-cell exit-code × JSON cross-product
at `:1329-1333` is correct against the code (the two exit-3 UNREACHABLE
cells are right: `sys.exit(3)` at `:196-197` strictly precedes both JSON
writes). The single gap is KW7.1(c) — the unparseable/non-`COMPLETED`
canonical file, which reaches **no** enum value.

### 2c — KW7.4 — **FATAL.** `validity_check` routes this design's own pre-registered reportable outcomes to `failed/`. This is KW5.4 reintroduced, and it does not discharge §A5's H4 directive.

§A5's binding directive: *"validity_check rewritten against the actual
schema, **accepting exactly the §5-reportable set**"* (`:3188-3189`).
§5's reportable set is unambiguous: *"**Every band is informative and
reportable** — none requires a follow-on wave to be publishable"*
(`:2242-2243`), and `INCOMPLETE-AT-K` is *"reported, disclosed with the
affected K(s) carried as a field"* (`:2206-2208`). `validity_check`'s
own universal assertion 6 admits it: `band["label"] in {…,
"INCOMPLETE-AT-K"}` (`:1874`).

But `INCOMPLETE-AT-K` arises **only** from an incomplete primary cell
(`:2194-2200`), i.e. **fewer than 12 canonical primaries** — and every
accept-set branch then fails. Adversarial near-miss #3, constructed and
traced: one primary seed `CRASHED` on both attempts (a deterministic
OOM/shape fault on one seed), 11 canonical primaries, `realized ≈ 8.0h`.

| Claimed `run_status` | Branch | Result |
|---|---|---|
| `COMPLETE` | 12 canonical primaries | 11 ≠ 12 → **fail** |
| `COMPLETE-DEGRADED` | same 12-canonical check | 11 ≠ 12 → **fail** |
| `STOPPED-BY-OPERATOR` | universal 1 excludes it | **fail** |
| `EXHAUSTED-BUDGET` | `realized > 13.80` | 8.0 → **fail** |
| `…-SUSPECT-OVERCHARGE` | same + fraction | **fail** |

**No `run_status` exists under which this run can be routed to
`completed/`.** A fully successful, publishable characterization is
certified `failed/` because one seed crashed twice.

Worse, the contradiction is internal to a single bullet.
`COMPLETE-DEGRADED`'s sub-case **(i) *primary-retry-refused*** is defined
as "a primary cell's attempt-2 retry was denied … that cell still
follows D5/E4's interval logic exactly as any other incomplete cell"
(`:1371-1375`) — an incomplete primary cell, hence ≤11 canonical
primaries — while the very same bullet asserts

> **`COMPLETE-DEGRADED` ⇔ the same 12-canonical-primaries condition as
> `COMPLETE`, PLUS at least one `GATE-REFUSED` or
> `PERSISTENTLY-ABORTED`-deriving row** (`:1387-1390`)

**Sub-case (i) can never satisfy its own disk-evidence assertion.** And
the attempt-2 `GATE-REFUSED` row that Rev 5 added at `:908-913` is
justified in the design's own words as "what gives
`COMPLETE-DEGRADED`'s *primary-retry-refused* sub-case the positive
on-disk evidence its `validity_check` disk-evidence assertion, §R5 H4,
depends on" — evidence for an assertion that is unsatisfiable by
construction. Note that §A5's directive said "COMPLETE ⇔ 12 canonical
primaries" and said nothing about `COMPLETE-DEGRADED`; extending the
condition to the degraded label is §R5's own move, and it is the move
that breaks.

**Discharge.** Keep the strict 12-canonical assertion for `COMPLETE`
only (it is correct and it is what §A5 asked for). For
`COMPLETE-DEGRADED`, assert the *identity* instead of the *count*:
every primary `(K, seed)` has a terminal disposition in
`ledger.attempts`, **and** the number of canonical primary files equals
the number of distinct primary `(K, seed)` pairs carrying a `COMPLETED`
row, **and** ≥1 `GATE-REFUSED`/`PERSISTENTLY-ABORTED`-deriving row
exists. That preserves everything H4 was buying (the no-op still fails:
0 canonical files vs 0 `COMPLETED` rows passes the identity but there is
no terminal disposition for any of the 12 cells, and no throttle row)
while admitting every pre-registered degraded and `INCOMPLETE-AT-K`
outcome. Re-run all three near-misses below against the replacement.

### 2d — The four `validity_check` payloads I ran

**Audit-R5 no-op** (`run_status="COMPLETE"`, `attempts=[]`,
`realized_gpu_h_final=0.0`): universals 1–4 pass vacuously (`|0.0−0|≤1e-6`
✓, `all(…)` over `[]` ✓); the `COMPLETE` branch finds 0 canonical files
≠ 12 ⇒ **correctly REJECTED**. §R5's claim is CONFIRMED, and it is the
disk-evidence branch — not any universal — that does the work.

**Near-miss #1 — 12 canonical primaries, `run_status="EXHAUSTED-BUDGET"`,
`realized=14.40`: INCORRECTLY ACCEPTED.** See KW7.5.

**Near-miss #2 — `STOPPED-BY-OPERATOR`, `attempts=[]`, no stop-file
marker: correctly REJECTED**, but by universal assertion 1 (the label is
outside the accept-set), never by the stop-file check. See KW7.11.

**Near-miss #3 — `COMPLETE-DEGRADED` with a degradation sub-case not in
the enumerated list (a twice-`CRASHED` primary cell): INCORRECTLY
REJECTED, and so is every other label it could claim.** See KW7.4 — this
is the FATAL.

### KW7.5 — **MAJOR.** The per-`run_status` branches are not mutually exclusive; a report whose label contradicts its own disk evidence is accepted.

`EXHAUSTED-BUDGET` is *defined* as "the hard gate refused a PRIMARY
cell's OWN FIRST ATTEMPT — the 12-cell baseline itself could not be
completed inside the ceiling" (`:1403-1405`), but its only enforced
evidence is `realized_gpu_h_final > 13.80` (`:1410-1411`). Twelve
primary cells running near their `1.20h` ceiling reach `realized ≈
14.40h` — so `> 13.80` and `12 canonical primaries` are **simultaneously
satisfiable**, and a report claiming `EXHAUSTED-BUDGET` while carrying a
complete 12-cell baseline passes every assertion and lands in
`completed/` with a label its own disk evidence refutes. The mutual
exclusivity the unified table asserts for `run_status` (`:1335-1338`) is
prose only; nothing enforces it.

**Discharge.** Add the negative half of the definition:
`EXHAUSTED-BUDGET` / `-SUSPECT-OVERCHARGE` ⇒ `realized_gpu_h_final >
13.80` **AND** fewer than 12 canonical primary files **AND** ≥1 primary
first-attempt `GATE-REFUSED` row. One clause each; it also makes the
five branches provably disjoint, which is what H4 claimed.

---

## §3 SCOPE ITEM 3 — H3 ARITHMETIC: **PASS**

**Recomputed from the design's own terms, independently:**
`13 × 0.0157 = 0.2041`; `15.00 + 0.2041 = **15.2041**` ✓ (`:1550`).
Stated deltas check: `15.2041 − 15.0157 = 0.1884` ✓ (`:1553`);
`15.2041 − 15.20 = 0.0041` ✓ (`:1554`).

**Provenance of the multiplier 13 = 1 + 12, verified:**
- The `1` is the truly-last attempt's own unpriced tail, `τ = 0.0126 +
  0.0031 = 0.0157` (`:1509-1516`) — the KW5.7-corrected SUM of the eval
  term and the `log_every=500` granularity term.
- The `12` is `⌊15.0157 / 1.20⌋` (`:1545`): every leaking attempt is a
  recovered one, each charged its full `charged_ceiling ≥ 1.20` (the
  primary arm's ceiling, correctly chosen as the MINIMUM so the count is
  maximised), against a ledger bounded by `15.0157h`. `15.0157/1.20 =
  12.513` → `12` ✓. Using the conditional `2.32` would yield 6, so `12`
  is the conservative choice ✓.
- The identity behind the factoring holds independently: `T ≤ R_N +
  Σleak_i ≤ (15.00 + τ) + 12τ = 15.00 + 13τ` ✓. `leak_i ≤ τ` is sound
  because the SUBPROCESS self-limits at its own `ceiling_s` check
  (`ncr_earlyln_scale.py:190-200`), so an orphaned attempt cannot run
  past `ceiling + τ` even with the orchestrator dead.
- The never-restarts case is covered rather than missed: the hard gate
  pre-verified `R_{N−1} + ceiling(N) ≤ 15.00`, so the unrecovered final
  attempt contributes at most `ceiling(N) + τ`, landing inside the same
  expression.

**Deleted-sentence check.** The false *"This is TIGHTER than the return
case"* is **gone from the live body**. The only three occurrences of
"TIGHTER" in the file are `:1522` (the H3 replacement text, which quotes
the phrase to mark it deleted) and `:3060` / `:3292` — both inside
frozen sections. See KW7.12 for `:3060`.

**Margin structure, checked at every appearance:** `15.20 + 0.30 =
15.50` at `:1615-1617` ✓; §6's own-cost-ceiling bullet at `:2287-2296`
carries the same structure plus the new third margin job ✓;
`gpu_h_estimate: 15.50` at `:1837` ✓; `validity_check`'s
`realized_gpu_h_final <= 15.50` at `:1855` ✓; program total
`15.50 + 0.15 = 15.65` at `:1938` and `:2309` ✓. `15.2041 < 15.50` ✓,
and the "three jobs" disclosure at `:1601-1612` is honest about the
`0.0041h` overshoot of the `15.20h` figure. **No inconsistency found.**

### KW7.6 — **MINOR.** The formula `T = L_final + Σ leak_i` is wrong as written; only the symbol is, and the number is right.

> Then `T = L_final + Σ leak_i`, where `L_final` is **the truly-last
> attempt's own cost**, `L_final ≤ 15.00 + τ` (`:1540-1542`)

A single attempt's own cost is bounded by `1.20 + τ` (or `2.32 + τ`),
not `15.00 + τ`; and `T = Σt_i` over *every* attempt cannot equal one
attempt's cost plus the leaks. The quantity actually used is the LEDGER
bound `R_N ≤ 15.00 + τ`, which is correct and yields the correct
`15.2041`. **Discharge:** rename `L_final` to the ledger's final value
and state `T = R_N + Σ leak_i`. Purely editorial — but an implementer or
reviewer taking the sentence literally derives a different (and wrong)
bound.

### KW7.7 — **MINOR.** The `15.2041h` bound is conditional on KW7.1 and does not say so.

The whole derivation rests on `R_N ≤ 15.00 + τ` and on "at most 12
attempts can have leaked **before the ledger itself would refuse the
next dispatch**" — both of which assume the ledger only ever accumulates.
KW7.1 shows reconstruction can reduce it. Under KW7.1(a)'s worked case
the ledger drops by ~7.2h and both premises fail. **Discharge:** either
land KW7.1's fix (after which the bound stands unchanged), or state the
bound as conditional on reconstruction being monotone.

---

## §4 §R5's 17-ROW DISPOSITION TABLE — 14 verify, 3 qualified

FATALs and MAJORs checked in full against the text; MINORs checked by
locating the fix.

| Row | §R5 claim | Round-6 verdict |
|---|---|---|
| KW6.1 | FIXED (H1) | **PARTIAL** — atomicity clean and correctly cited; recovery step 0 present but defective (KW7.1) |
| KW6.2 | FIXED (H2) | **PARTIAL** — copy-then-fold, the recovery canonical check and the crash-window table are all correct; the attempt-2 face survives (KW7.2) |
| KW6.3 | FIXED (H3) | **VERIFIED** — false sentence deleted, honest bound derived, margin structure consistent |
| KW6.4 | FIXED (H4) | **VERIFIED** — `d_override` added to the ledger row schema (`:1272`), recorded at gate-check time, defined for `GATE-REFUSED` rows (`:813`, `:909`); universal 4 reads it as pure ledger arithmetic |
| KW6.5 | FIXED (H4) | **VERIFIED for (i)(ii)(iii)**; **(iv) SUBSTITUTED** — the audit asked for the frozen rows to be corrected, §R5 outranks them instead (see KW7.13) |
| KW6.6 | FIXED (H4) | **VERIFIED as an enum fix** (`COMPLETE` single-criterion; third sub-case added) — but the disk assertion attached to it breaks sub-case (i), KW7.4 |
| KW6.7 | FIXED (H4) | **PARTIAL** — the no-op hole is genuinely closed; over-permissive in a new direction (KW7.5) and over-restrictive in another (KW7.4) |
| KW6.8 | FIXED (H5) | **VERIFIED** — `ceiling_charged` row field, `charged_vs_measured` report block, the `>0.50` rule, and the never-automatic-resubmission clause all present and coherent |
| KW6.9 | ACCEPTED-BY-ADDENDUM | **OUTCOME ACCEPTED, JUSTIFICATION REFUTED** (KW7.13) |
| KW6.10 | FIXED, subsumed by H2 | **VERIFIED** — classification strictly precedes any row write (`:878-887`) |
| KW6.11 | FIXED (1 digit) | **VERIFIED** — `:784`, product unchanged |
| KW6.12 | FIXED (comment) | **VERIFIED** — both occurrences, `:552` and `:617` |
| KW6.13 | FIXED | **VERIFIED** — startup smoke-read + refuse-to-dispatch at `:1957-1972` |
| KW6.14 | FIXED | **VERIFIED** — `abs(...) <= 1e-6` at `:1854-1858` |
| KW6.15 | FIXED | **VERIFIED** — `rate`/`gate_eligible`/`gate1_label` named as never-read at `:1801-1810` |
| KW6.16 | FIXED (attribution) | **VERIFIED** — `KW5.13`→`KW5.6` at `:698-700`, correct against §R4's own rows |
| KW6.17 | FIXED | **VERIFIED** — `nvidia-smi --query-compute-apps` reap + abort-loudly on both recovery branches (`:1081-1088`) |

### KW7.13 — **MINOR.** KW6.9's ACCEPTED-BY-ADDENDUM: outcome accepted, stated justification refuted, a better justification exists.

**Refuted.** Mitigation (iii) leans on the §A3/§A4 separator-append
precedent and concedes in the same breath that it is "one layer up,
applied to content instead of punctuation." That is an admission that
the precedent does not cover the case. The precedent that *does* cover
it points the other way, and KW6.9 quoted it: §R4's own KW5.8 row holds
that the cosmetic acceptance *"did NOT transfer to §R3 at the time of
the R4 audit, because §R3 was still the LIVE revision log for the round
then under review."* §R4 was the live revision log at the time of the R5
audit. Declaring it frozen at the moment a finding lands against it is
the precedent inverted.

**Accepted anyway, for a reason §R5 did not give.** Editing §R4's
"Where fixed" column would destroy the `§A1-ADJUDICATION → end-of-file`
MD5 identity that §R5 asserts and that I independently reproduced
(`df44dee31a86dc8afeed11f3d3e51024`, both sides). That identity is a
real integrity instrument for every future round; trading it for two
bookkeeping cells is a bad trade. The corrected attribution is recorded,
the residual is disclosed as attackable, and the impact on mechanism,
ceilings and numbers is nil. **Non-forcing. Discharge (optional, cheap):
replace mitigation (iii) with the MD5-identity argument, which is
sound.**

---

## §5 FURTHER MINORS

### KW7.8 — **MINOR.** The precedence sentence's line pointers are pre-Rev-5 numbers and now land in the wrong section.

> the FROZEN historical sections §R4 (KW5.3's row, `:2590`-region; the
> "numbers that moved" line, `:2615-2616`-region) (`:1300-1301`)

Verified: in the **pre-Rev-5** file those were correct (`prev:2590` is
KW5.3's row; `prev:2613-2616` is the "numbers that moved" enumeration).
Rev 5 shifted §R4 by **+472** lines. In the shipped file `:2590` and
`:2615-2616` are ordinary §R1 prose about the K=32 budget columns and
the Rev-1 re-derivation sources. Correct current targets: **`:3062`** and
**`:3085-3088`**. This document already has a house habit of flagging
renumbering (KW3.7 at `:24` and `:2005`); the precedence sentence is the
one place a stale pointer directly undercuts the mechanism it
establishes. **Discharge:** repoint, or drop the line numbers and cite
by name only (which cannot go stale).

### KW7.9 — **MINOR.** The two §6 red-team hunks are outside §R5's claimed edit list.

§R5 states "every rewrite is listed in the 'Where fixed' column"
(`:3215-3217`). Hunks `@@ -1869 +2320` and `@@ -1898 +2350` rewrite §6's
resource/placement red-team, adding items **(x)–(xiv)** and updating
item (viii)'s accept-set. No H1–H6 "Where" entry and no table row names
§6's red-team list — H3 names only §6's "Own cost ceiling" bullet. The
content is additive, correct, and materially good (it is the only place
that mandates a corrupt-ledger restart test, a copy-then-fold kill test,
a `GATE-REFUSED`-through-`validity_check` test, a GPU-reap test and a
forced-overcharge test). **This is the same defect class §R5 was
adjudicating in KW6.9, recurring in §R5's own edit list.** Automatic
finding per this round's mandate; non-forcing on substance.
**Discharge:** add "§6 (resource/placement red-team, items (x)–(xiv) and
item (viii))" to H1/H2/H4/H5's "Where" entries.

### KW7.10 — **MINOR.** §R5's post-revision live-body MD5 does not reproduce.

Claimed (`:3218-3223`): whole file before = `bd8d30b783e3e23b3eec587dc253fc05`
**✓ reproduces exactly**; live body before = `eeb62b6c236a0101759bb310fdcbe13d`
**✓ reproduces exactly** (lines `1..2023` of `HEAD~1`, i.e. header + §1–§7);
live body after = `ef8a45873b2c2455f12aa1db41879ab5` — **✗ does not
reproduce** at the corresponding boundary (`1..2495`) nor at any start
∈ {1, 27, 28} × end ∈ [2400, 2500] of the shipped file. The load-bearing
integrity claim (frozen block identical) and the "before" hashes are
sound; the one unreproducible figure is the non-load-bearing "after"
value, most likely computed against an intermediate state. **Discharge:**
recompute and restate, or drop it — a self-attestation that does not
verify is worse than no attestation.

### KW7.11 — **MINOR.** `STOPPED-BY-OPERATOR`'s stop-file disk-evidence assertion is dead inside `validity_check`.

Universal assertion 1 excludes the label before any per-`run_status`
branch runs, and the design says so itself: "this branch never fires
here" (`:1889-1891`). Yet §R5's H4 summary presents the marker check as
part of what closes the no-op hole ("fails under every OTHER
`run_status` label it could instead claim"). True, but by exclusion, not
by the marker. **Discharge:** relocate the stop-file check to the
orchestrator's own pre-write self-check (where it can actually fire and
prevent a false `STOPPED-BY-OPERATOR` report), and say so, so a build
implementer does not code an unreachable branch.

### KW7.12 — **MINOR.** The frozen §R4 KW5.1 row still carries the refuted TIGHTER claim, and the precedence sentence does not reach it.

`:3060` still reads: *"the crash case is actually TIGHTER (`R_N≤15.00`
exactly, no tail) than the completing case."* The precedence sentence at
`:1296-1304` is scoped to **enumerations** and names only KW5.3's row and
the "numbers that moved" line. A reader who consults §R4's disposition
table for the ledger-bound history reads the refuted claim with nothing
pointing at H3. **Discharge:** one clause in §R5's H3 paragraph naming
`§R4`'s KW5.1 discharge row as also carrying the now-refuted claim and
outranked by §4's "True spend, worst case" — the same addendum pattern
KW6.9 already establishes.

### KW7.14 — **MINOR.** `{<the 10 §5 partition labels>}` is an unexpanded placeholder and the count is unsupported.

Universal assertion 6 (`:1874`) asserts `band["label"] in {<the 10 §5
partition labels>, "INCOMPLETE-AT-K"}`. §5 defines **six** band labels —
`FRONTIER-AT-K*=24/26/28/30`, `GRADUAL-DECAY`,
`NON-MONOTONE-UNRESOLVED` — and §5 itself states the six-rule procedure
"returns exactly ONE label" (`:2201-2203`). The three 160K qualifier
bands (`CONFIRMED-WALL-AT-160K`, `SLOW-CONVERGENCE-AT-160K`,
`PARTIAL-IMPROVEMENT-AT-160K`) are a **different field**
(`conditional.qualifier_band`, `:1284-1285`). No reading of §5 yields
10. The check is over-permissive rather than dangerous, but as written a
build implementer cannot produce the literal set. **Discharge:** write
the six labels out, or cite §5's rule list by line. (`trigger
["resolution"]`'s enum in the same assertion **does** match the schema
at `:1280-1281` ✓.)

### KW7.15 — **MINOR.** H1's totality rests on an unstated code premise.

Reconstruction step 0.2 is only sound because every dispatched attempt
provably leaves an `attempt{n}/` directory — which is true
(`ncr_earlyln_scale.py:237`, `os.makedirs(outdir, exist_ok=True)`,
before training and before the resume-skip) but is stated nowhere in the
design. Without it, "a (K, seed) with NO disk evidence at all … remains
genuinely available for dispatch — reconstruction never charges for work
that **provably never ran**" (`:1036-1039`) is an unsupported inference:
absence of evidence is not proof of absence. **Discharge:** cite `:237`
as the premise, and soften "provably never ran" to "left no evidence
this design's dispatch path can produce."

---

## §6 INTEGRITY — **PASS**

- **Frozen sections byte-identical.** `§A1-ADJUDICATION` → end of the
  pre-Rev-5 file, `HEAD~1:2024-2733` vs `HEAD:2496-3205`, both 710 lines:
  MD5 **`df44dee31a86dc8afeed11f3d3e51024`** on both sides — matching
  §R5's own claim exactly. Independently reproduced, not taken on trust.
- **Line-count sanity.** `2733 + (840 − 167) = 3406` ✓; live body +472,
  §R5 append +201, sum 673 ✓.
- **Section skeleton intact:** §1 `:28`, §2 `:65`, §3 `:199`, §4 `:421`,
  §5 `:1984`, §6 `:2252`, §7 `:2401`, §A1 `:2496` … §A5 `:3160`,
  §R5 `:3209`.
- **No excluded-section hunk.** §5 and §7 untouched; the four
  pricing/trigger-adjacent contacts are accounted for in §0.
- Header string matches the expected `DRAFT-R5 — POST-AUDIT-5, AWAITING
  NARROW AUDIT ROUND 6`.

---

## §7 WHAT I COULD NOT BREAK

Recorded so the next round does not re-spend effort here.

- **H2's core ordering fix is correct.** Copy-then-fold + the recovery
  canonical check genuinely establish `COMPLETED ⇒ canonical`, and with
  G2's `canonical ⇒ COMPLETED` the harvest-patch-unnecessary claim is
  true BY THE PAIR exactly as §A5 directed. The "between copy and fold"
  window — KW6.2's actual FATAL — is closed.
- **Atomicity is real, not asserted.** `rn.atomic_write_json` exists at
  the cited lines and is the same helper the cell script already uses;
  the truncation window is closed by construction.
- **The mid-copy `.tmp` is genuinely invisible** to both the glob and
  the regex filter — checked against the code, not the prose.
- **The exit-code cross-product is right**, including both UNREACHABLE
  cells and the new `(exit 0, no JSON) → CRASHED` default arm.
- **`PERSISTENTLY-ABORTED` is now correctly a derived CELL state** with
  a stated derivation rule, and it no longer appears in any
  `attempts[].status` enumeration in the live body.
- **H5's escape hatch works** on the scenario it was built for: repeated
  dispatch-then-die cycles drive `ceiling_charged_fraction` above 0.50
  and the run reports `EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE` with
  resubmission blocked.
- **The no-op JSON is genuinely dead** under the rewritten
  `validity_check`.
- **Every code line-number citation in the touched text is accurate**
  (`:196-197`, `:237`, `:243-245`, `:262-266`, `:304`, `:307`,
  `:358-380`; `run_ncr.py:105-109`).
- **H3's 15.2041 is right**, by two independent routes.

---

## §8 VERDICT — **REV-REQUIRED**

Two FATALs, both inside the round-6 scope, both non-discharges of a
binding §A5 disposition rather than new complaints:

- **KW7.1** — §A5 H1 required "No path re-opens budget." Reconstruction
  charges one ceiling per CELL where up to two attempts were paid,
  excludes the conditional arm whenever its canonical directory is
  absent, and has no branch at all for a present-but-unreadable
  canonical file. A worked, in-rules case hands back **7.2 GPU-h**.
- **KW7.4** — §A5 H4 required `validity_check` to accept "exactly the
  §5-reportable set." As written it rejects every run containing a
  single twice-failed primary cell, i.e. every `INCOMPLETE-AT-K`
  outcome, and `COMPLETE-DEGRADED`'s own sub-case (i) cannot satisfy the
  assertion attached to it. This is KW5.4 in a new costume.

Both discharge with small, local, fully-specified edits — no
re-architecture, no re-derivation, and **no re-entry into the excluded
numeric sweeps**. H3 passes and needs no rework beyond one symbol
rename. Round 7 should be narrower still: **KW7.1 + KW7.2 + KW7.3
(reconstruct per attempt directory; cell-wise resume; read the
attempt-dir JSON) and KW7.4 + KW7.5 (`COMPLETE-DEGRADED` identity
assertion; `EXHAUSTED-BUDGET` negative clause) ONLY**, plus a
locate-the-fix pass on the ten MINORs. Everything in §7 above is
settled and must not be re-opened.

**The R5 build-release checklist is NOT yet in force** — it is
conditional on a clean design, and the design is not clean. It should be
re-stated as the binding charter at the moment Rev 6 clears, with the
additions §9 records now so they are not lost.

---

## §9 ADDITIONS THE R5 CHECKLIST IS MISSING (for the eventual charter)

To be merged into `NCR_KWALL_ATTACK_R5.md` §9's six items when the build
is finally released:

- Checklist item 1 must name the reconstruction contract explicitly:
  **per-attempt-directory** reconstruction, the canonical-file
  else-branch, and cell-wise (not attempt-wise) terminal skipping.
- Checklist item 3's `validity_check` one-liner must be the **Rev-6**
  version, with the `COMPLETE-DEGRADED` identity assertion and the
  `EXHAUSTED-BUDGET` negative clause — not Rev 5's.
- Checklist item 5 must cover §6's red-team items **(i)–(xiv)**, not
  (i)–(ix); items (x)–(xiv) are the only place the corrupt-ledger,
  copy-window, `GATE-REFUSED`-routing, GPU-reap and forced-overcharge
  tests are mandated, and R5's checklist predates them.
- Add a **negative test with teeth** (CLAUDE.md's standing rule): a
  synthetic 11-canonical-primary report must be ACCEPTED to `completed/`
  after KW7.4's fix, and the no-op report must still be REJECTED —
  both run to completion, not merely written.
- Add: the build must assert `os.makedirs(outdir, exist_ok=True)`
  still precedes training in `ncr_earlyln_scale.py`, since KW7.1's
  discharge depends on it (KW7.15).

---

*Narrow audit round 6, 2026-08-06. Written from direct reads of
`matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md` (§1–§7 live body
in the regions the diff touched, plus §5's band/`INCOMPLETE-AT-K`
sections, §6's ceiling/red-team bullets, §A5 and §R5 in full, §R4 by
targeted row read), `NCR_KWALL_ATTACK_R5.md` (§0, KW6.5, KW6.9, §9),
`matrix-thinking/ncr/ncr_earlyln_scale.py` (`:190-320`, `:351-385`) and
`matrix-thinking/ncr/run_ncr.py` (`:100-112`), plus `git diff`/`git
show` and `md5` section comparisons of `HEAD~1` vs `HEAD`. The
125-outcome partition, the trigger sweeps and the pricing chain were NOT
re-executed, per the binding round-6 scope. `validity_check` payloads
were traced by hand against the assertion list as written. No repo file
other than this one was created or modified; no command was run on the
box; no job was launched; no git mutation was made.*
