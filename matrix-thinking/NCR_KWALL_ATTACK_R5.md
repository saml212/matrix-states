# NCR K-WALL CHARACTERIZATION — AUDIT ROUND 5 (expected-terminal, contract-line verification)

**STATUS: COMPLETE. VERDICT = `REV-REQUIRED`.** 2 FATAL / 6 MAJOR /
9 MINOR. The round was dispatched as the expected TERMINAL verification
and it does not terminate: two forcing findings sit on the G1/G2 seam.
Both are contract-line surgery inside the existing ORCHESTRATOR
CONTRACT — **no seventh delivery model, no new mechanism, ~8 lines of
text** — and everything else this round could check verified clean,
including every number in the document.

Target: `matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md`
@ `8d0290f`, 2686 lines. Header verified as expected: **"DRAFT-R4 —
POST-AUDIT-4, AWAITING AUDIT ROUND 5 (not build-released, not
queue-eligible)"** — no mismatch to report.

Per-disposition: **G1 FAIL · G2 FAIL · G3 PARTIAL · G4 FAIL ·
G5 PASS · G6 PARTIAL.**

---

## §0 SUMMARY

Rev 4's fixes are, once again, **correct in isolation and unchecked
against each other and against the crash path** — the same shape round
4 diagnosed in §R3, one layer deeper. G1 (write-ahead pricing) and G2
(canonical-path harvest) are individually sound mechanisms that each
close the FATAL they were written for; what neither round has checked
is the *ordering seam between them*, and that seam is where both of
this round's FATALs live:

- **KW6.1** — the ledger, on which the entire budget bound now rests,
  has no atomicity or durability contract, and the recovery procedure
  has no branch for a ledger it cannot parse. A crash **during the
  ledger's own rewrite** (2 writes per attempt, ≤64 windows per run)
  leaves a truncated JSON; the natural build behaviour on an
  unparseable ledger — start fresh — re-opens the full 15.00 GPU-h
  budget. G1's headline guarantee ("a crash can only over-charge,
  never re-open budget") is false as written.
- **KW6.2** — the ledger fold is specified to happen *before* the cell
  JSON's status is inspected, and G2's canonical copy happens *after*.
  A crash in that window leaves a `COMPLETED` ledger row with **no
  canonical file**, and G1's own cell-level resume rule then skips the
  cell forever, so the copy is never retried. `harvest()` reads
  `n_completed=3` where the truth is 4 — silently, with no cross-check
  anywhere — and the study's band changes. G2's "the status-patch is
  unnecessary by construction" claim is true in the direction it states
  (canonical ⇒ COMPLETED) and **insufficient in the direction
  `harvest()`'s count actually needs** (COMPLETED ⇒ canonical).

The MAJORs are all on the same two seams plus the G4 job-spec contract:
the crash-recovery charge under-prices true spend by one tail per
recovered attempt (KW6.3, and the bolded "the crash case is TIGHTER"
claim is therefore false); the new `validity_check` cannot be
implemented over the ledger schema it is specified against and throws
on exactly the degraded runs G4 exists to protect (KW6.4); the three
enumerations of attempt status disagree with each other (KW6.5); the
`run_status` enum is neither exhaustive nor mutually exclusive
(KW6.6); the same `validity_check` is simultaneously over-permissive
enough to certify a total no-op as `completed/` (KW6.7); and repeated
crash-recovery over-charging can terminate the wave in a false
`EXHAUSTED-BUDGET` at ≈0 true spend with no escape hatch (KW6.8).

**What verified clean, by independent re-execution, not by citation
(recorded so round 6 does not re-spend it):** the entire numeric
apparatus. G5's amended sweep reproduces to the digit
(844/156 → 473/527, 371 → 0 paid-on-unresolved, 0/1000 reverse
direction, **0/1000 illegal dispatches at an excluded, unresolved, or
wide-tie K**); the 125-outcome partition (`18/4/12/8/12/8/42/15/4/2`,
Σ=125) and all 10 representative rows; the 729-space
candidate-set-size sweep `{1:612, 2:102, 3:14, 4:1}` with **15/15**
wide ties landing in `INCOMPLETE-AT-K`; the 11-configuration ambiguity
table, all 11 rows exact; KW4.7's six decide-rates (16/25, 17/25,
25/25, 43/80, 38/80, 36/80); the whole pricing chain including
`F(24,25,64)=8,636,672`, the five-K ratios, the nominals, the six
informational ceilings and `1.2685/1.0510=1.2069`; and
`0.0126+0.0031=0.0157`, `15.00+0.0157=15.0157`,
`15.20−15.0157=0.1843`. Additionally verified **against the real
code**, not the design's prose: the `earlyln_K{K}_s*.json` glob
(`ncr_earlyln_scale.py:364`) does **not** match
`ORCHESTRATOR_LEDGER.json` (a live collision risk the canonical
directory would otherwise have had), the `.axis_c_lock.json` sibling is
explicitly excluded at `:365-366`, `sys.exit(3)` is at `:196-197` as
G3 assumes, and the `blank_out` assert at `:300` fails *before* any
JSON is written and so lands correctly in G3's `CRASHED-n` branch.

---

## §1 G1 — WRITE-AHEAD ATTEMPT PRICING: **FAIL**

### What is right, verified not assumed

The mechanism does close both faces KW5.1 named. Walking the
state machine as written:

| Crash point | On-disk state | Recovery result | Ledger vs true spend |
|---|---|---|---|
| **(a)** after gate check, before `open_attempt` write | unchanged | `open_attempt` null → normal resume; cell has no row → re-gated, re-dispatched | ledger = true (both ≈0) ✓ |
| **(b)** after `open_attempt` write, before `subprocess.run` | dangling `open_attempt` | charged FULL `charged_ceiling` | ledger **over**-charges by 1.20 ✓ conservative |
| **(c)** mid-attempt (child dies with parent) | dangling `open_attempt` | charged FULL ceiling | ledger ≥ true ✓ |
| **(d)** after `subprocess.run` returns, before the fold | dangling `open_attempt` | charged FULL ceiling | **ledger < true by up to 0.0157h** ✗ (KW6.3) |
| **(e)** after the fold, before/during the canonical copy | terminal row, no/partial canonical file | cell skipped forever | **silent harvest loss** ✗ (KW6.2) |
| **(f)** during the ledger's own rewrite | **truncated JSON** | undefined | **budget re-opened** ✗ (KW6.1) |
| **(g)** during recovery, before its persist | on-disk unchanged | next restart re-charges from the same base | no accumulation ✓ |
| **(h)** double/N-fold restart | each cycle's dangling record priced before the next gate check | induction premise survives ✓ | but see KW6.8 |

Rows (a), (b), (c), (g), (h) are exactly as the design claims, and the
cell-level resume rule genuinely closes KW5.1's second face — a restart
at `realized≈13h` cannot turn an already-`COMPLETED` cell into
`GATE-REFUSED`, because the ledger is consulted before any gate. The
induction's cycle-count argument is sound: every dangling record is
priced before the next gate check, so the number of restarts is
irrelevant. **That much is a real fix and should stand.**

Arithmetic re-check: `0.0126 + 0.0031 = 0.0157` ✓;
`15.00 + 0.0157 = 15.0157` ✓; `15.20 − 15.0157 = 0.1843` ✓;
`12 × 1.20 = 14.40` ✓; `4 × 2.32 = 9.28` ✓.

### KW6.1 — **FATAL.** The ledger has no atomicity or durability contract, and no defined behaviour when it cannot be parsed — so the crash class G1 exists to survive includes one that re-opens the entire budget.

> The file is rewritten after EVERY ledger update — an `open_attempt`
> write BEFORE `subprocess.run`, and an `open_attempt` clear +
> `attempts[]` append + `realized_gpu_h` increment AFTER it returns —
> not only at the end. (`:910-913`)

> **Recovery procedure … 1. Read `ORCHESTRATOR_LEDGER.json`. If
> `open_attempt` is `null`, nothing is dangling — proceed to step 3.**
> (`:915-918`)

Grep confirms: **`atomic`, `fsync`, `os.replace`, `rename` appear
nowhere** in this document in a ledger context (`:1153`'s "atomic" is
about batching, `:1611` is unrelated). The literal build of `:910-913`
is `json.dump(ledger, open(path, "w"))`, which truncates the file
first. The write happens twice per attempt, up to 32 attempts — **≤64
truncation windows per run**, all of them straddling the exact events
(dispatch, return) at which the orchestrator is most exposed to the
crash classes KW5.1 enumerated (box teardown, OOM-kill, tmux kill).

If a crash lands in one, `ORCHESTRATOR_LEDGER.json` is truncated. The
recovery procedure's step 1 says only "Read" — there is **no branch
for an unparseable or missing ledger anywhere in the document.** The
two natural build behaviours are:

- treat it as absent and start fresh → `realized_gpu_h = 0` → **the
  full 15.00 GPU-h budget is re-opened**, which is precisely the
  failure mode G1's own summary says is now impossible ("A crash can
  now only over-charge, never re-open budget", `:2504-2505`); or
- crash on the `JSONDecodeError` → the supervisor loop restarts,
  crashes again, and the wave wedges silently.

Neither is specified, and the first is the one an implementer reaching
for `dict.get`-style defensiveness will write. Note the repo already
ships the right primitive and uses it for every cell JSON in the same
program — `rn.atomic_write_json` (`ncr_earlyln_scale.py:265`, `:307`,
and six further call sites across `ncr/*.py`) — so the design is
*less* durable about its budget ledger than the harness already is
about a single cell's result.

**Discharge condition.** Two contract lines in "Ledger persistence +
write-ahead recovery": (i) every ledger write is ATOMIC — serialize to
`ORCHESTRATOR_LEDGER.json.tmp` in the same directory, `fsync`,
`os.replace` onto the final path (name `rn.atomic_write_json` as the
existing house helper); (ii) recovery step 0: if
`ORCHESTRATOR_LEDGER.json` is missing while ANY output artifact for
this wave exists on disk, or is present but unparseable, the
orchestrator **HARD-ABORTS with `run_status="EXHAUSTED-BUDGET"` and
dispatches nothing** — it never treats an unreadable ledger as a fresh
start. Add to §6 red-team item (vi): corrupt the ledger mid-run and
confirm the restart aborts rather than resuming at zero.

### KW6.3 — **MAJOR.** "The crash case is actually TIGHTER (`R_N ≤ 15.00` exactly, no tail)" is false for TRUE spend; the worst-case crash schedule reaches ≈15.2041h, breaking both the tight bound and the disclosed 15.20h figure.

> - **It never returns** (the orchestrator itself dies): `R_N`, as
>   computed by the NEXT restart's recovery step, is `R_{N-1} +
>   charged_ceiling(N) ≤ 15.00` EXACTLY — no tail term at all, because
>   the recovery charge is the gate-admitted ceiling value, not a
>   measurement. **This is TIGHTER than the return case.** (`:1196-1200`)

`R_k` is *defined* as `ledger.realized_gpu_h` (`:1164`), and the
paragraph is correct about the LEDGER. It is then used as the bound on
GPU-hours consumed (`:1203`, and §6's ceiling derivation at
`:1841-1846`). Those are different numbers on the crash path.

Crash point (d) in the table above: the subprocess runs to its own
`COMPLETED` return having consumed `ceiling + 0.0157h` (the combined
tail KW5.7 established and §R4 correctly adopted), `subprocess.run`
returns, and the orchestrator dies before the fold. Recovery charges
`charged_ceiling` — a value that is now **less than the attempt truly
cost**. The induction's own stated premise — *"exact-or-conservative
in every case, never an omission"* (`:1169`) — fails here in the
non-conservative direction.

Worst-case true spend, re-derived: with `τ=0.0157`, `T = Σt_i =
L_final + Σ leak_i`, `leak_i ≤ τ` for a recovered attempt and `0`
otherwise. `L_final ≤ 15.00 + τ` (the last attempt's own tail). A
leaking attempt necessarily contributes `x_i = c_i ≥ 1.20` to the
ledger, so at most `⌊15.0157/1.20⌋ = 12` attempts can leak. Hence

```
T  ≤  15.00 + (1 + 12)(0.0157)  =  15.2041 GPU-h
```

which exceeds the stated tight bound `15.0157h` by `0.1884h` and the
**disclosed** conservatively-rounded figure `15.20h` by `0.0041h`. It
is still inside the declared `15.50h` pool ceiling — but only because
the `0.30h` supervisor margin absorbs it, and round 4's own §7 ruling
established that margin is already fully committed to the unpriced
process-startup term ("it is not slack, it is the only place that cost
is priced"). The margin is now double-booked.

**Discharge condition.** One of: (a) charge `open_attempt.
charged_ceiling + 0.0157` on recovery — one term, restores
`T ≤ 15.00 + 0.0157 = 15.0157` exactly as claimed, and is the cheapest
fix; or (b) keep the ceiling charge and restate the bound honestly as
`T ≤ 15.00 + (1 + n_recovered)·0.0157 ≤ 15.2041h`, delete "This is
TIGHTER than the return case," and state that the 0.30h margin carries
both this and the startup term.

### KW6.8 — **MAJOR.** Repeated crash-recovery over-charging terminates the wave in a false `EXHAUSTED-BUDGET` at ≈0 true spend, with no disclosure and no operator escape hatch.

The conservative charge is presented as unambiguously safe ("a crash
can only over-charge", `:2504`). Over-charging is safe for the
*ceiling*; it is not safe for the *study*. Each crash→restart cycle
that reaches the dispatch point burns `1.20h` of ledger for `≈0h` of
real compute. Twelve such cycles — the exact behaviour CLAUDE.md's own
mandated supervisor pattern (`while [ ! -f STOP ]; do <cmd>; sleep 15;
done`) produces against any systematic mid-attempt kill (kernel OOM
targeting the parent, box reboot loop, a bug in the orchestrator's own
post-dispatch path) — drive `realized_gpu_h` to `14.40h`. Every
subsequent cell is `GATE-REFUSED`; `run_status` reads
`EXHAUSTED-BUDGET`; and the design's own resubmission advice
(*"resubmitting the job resumes cleanly … at no cost beyond the ceiling
already spent"*, `:1136-1138`) is void, because the resumed run
gate-refuses everything. The wave is permanently dead having produced
no data and consumed ≈0 GPU-h.

Nothing in §4, §5 or §6 discloses this, and there is no specified
override (no ledger-reset procedure, no `--force-resume`, no cap on
cumulative recovery charges). `validity_check` passes it (the
self-consistency sum holds, since the recovered rows carry
`elapsed_h = charged_ceiling`), so the job lands in `completed/` and
the pool will not re-run it.

**Discharge condition.** One disclosure paragraph in G4's
`EXHAUSTED-BUDGET` definition naming this path explicitly ("an
`EXHAUSTED-BUDGET` verdict whose `attempts[]` is dominated by
`CRASHED-RECOVERED` rows indicates an environment fault, NOT a budget
result, and is not a §5-reportable outcome"), **plus** one operator
escape: a documented, disclosed manual ledger-reset procedure, or a
cap (e.g. `n_recovered ≥ 3` ⇒ abort with a distinct
`run_status="ENVIRONMENT-FAULT"` rather than continuing to burn
ledger). If `ENVIRONMENT-FAULT` is added it must be excluded from
`validity_check`'s accept-set so the job lands in `failed/` and is
re-runnable.

### KW6.10 — **MINOR.** The terminal ledger row is specified to be appended before the status that populates its own `status` field is inspected.

> On return, `ledger.realized_gpu_h += attempt_elapsed_h`
> UNCONDITIONALLY, `ledger.open_attempt` is cleared, and a terminal row
> is appended to `ledger.attempts` — **before the resulting JSON's
> `status` is even inspected.** (`:824-828`)

The row schema is `{K, seed, arm, attempt_n, elapsed_h, status,
outdir}` (`:906-908`). At append time `status` is by construction
unknown. The resume rule then tests that very field against a
six-value terminal enum (`:936-937`), so a row written in this order
has undefined resume semantics. **Discharge:** state the two-phase
write explicitly — append with `status="RETURNED-UNCLASSIFIED"`,
persist, classify, then patch the row and persist again (and add
`RETURNED-UNCLASSIFIED` to the recovery procedure as "re-classify from
the on-disk cell JSON, do not re-dispatch").

### KW6.17 — **MINOR.** Recovery does not reap or verify the death of the crashed attempt's child process.

Crash point (c) assumes the child dies with the parent. That holds for
a tmux-session kill (the design's own cited premise) but not for a
kernel OOM-kill of the Python parent alone, which orphans a live CUDA
process holding the GPU. The restarted orchestrator then re-dispatches
attempt 2 onto a GPU still running attempt 1: real contention, wall-clock
inflation, and two processes writing to different attempt dirs.
**Discharge:** one line in recovery step 2 — before closing a dangling
`open_attempt`, verify no live process holds the assigned GPU (e.g.
`nvidia-smi --query-compute-apps`) and abort loudly if one does.

---

## §2 G2 — CANONICAL-PATH HARVEST CONTRACT: **FAIL**

### What is right, verified against the real code

KW5.2's three named faces are genuinely closed, and I checked the
mechanism against `ncr_earlyln_scale.py` rather than against the
design's description of it:

- The flat glob is `os.path.join(outdir, f"{cell_id(K,'*')}.json")` =
  `earlyln_K{K}_s*.json`, non-recursive (`:364`) — the canonical
  copy lands exactly where it reads. **No recursive glob is needed**,
  as G2 claims.
- **No duplicate seed is possible.** The exists-check is a real
  guarantee, and the reachability argument is correct: the dispatch
  loop only advances to attempt 2 from `ABORTED-BUDGET-1`/`CRASHED-1`,
  never from `COMPLETED`, so no cell can produce two `COMPLETED`
  attempts. The loud abort therefore fires only on a dirty pre-existing
  results directory — the right case to fail on.
- The **copy-then-status-reclassification** path the mandate asked
  about is genuinely closed: `COMPLETED` is terminal in the state
  machine, and exit-3 (`STOPPED-BY-OPERATOR`) cannot coexist with a
  `COMPLETED` JSON because `sys.exit(3)` fires inside
  `train_earlyln_cell` (`:196-197`) before `run_earlyln_cell` writes
  anything.
- Two **live collision risks I went looking for and did not find**:
  `ORCHESTRATOR_LEDGER.json` sits inside the canonical directory
  (`:905`) but is NOT matched by the `earlyln_K{K}_s*.json` pattern;
  and `{cid}.axis_c_lock.json` (`:283`) — which *would* match that
  pattern under fnmatch — stays in the attempt dir and is in any case
  explicitly excluded at `:365-366`. Both clean.
- The second-`harvest()` ambiguity is resolved correctly (separate call
  per tree, merged after).

### KW6.2 — **FATAL.** A crash between the ledger fold and the canonical copy silently deletes a `COMPLETED` cell from `harvest()`'s count, and G1's own resume rule guarantees it is never repaired.

The two orderings are each stated unambiguously and are incompatible:

> On return, `ledger.realized_gpu_h += attempt_elapsed_h`
> UNCONDITIONALLY, `ledger.open_attempt` is cleared, and a terminal row
> is appended to `ledger.attempts` — before the resulting JSON's
> `status` is even inspected. (`:824-828`)

> On attempt ACCEPTANCE — i.e. **the instant a subprocess's cell JSON
> reads `status=="COMPLETED"`** — the orchestrator COPIES that JSON
> from its archival attempt directory … to the CANONICAL FLAT PATH
> (`:975-980`)

So the copy is strictly *after* the fold, and the fold clears
`open_attempt`. A crash in between leaves: `open_attempt = null`, a
`COMPLETED` row in `attempts[]`, and **no canonical file.** Recovery
step 1 sees a null `open_attempt` and skips to step 3; step 3 then
says:

> a cell/attempt with an existing TERMINAL row already in
> `ledger.attempts[]` (`COMPLETED`, …) is **never re-gated** — its state
> comes from the ledger record (`:933-940`)

The cell is skipped. **The copy is never retried, by design.** No step
of the recovery procedure reconciles `attempts[]` against the canonical
directory, and `harvest()` never looks in the attempt dirs ("never
globs inside them, recursively or otherwise", `:488-489`), where the
authoritative JSON is sitting intact.

Consequence: `harvest()` returns `n_seeds = 3` for that K. Per §4's own
resolution-state table that K drops from `EXACT` to
`DECIDED`/`AMBIGUOUS`; per D5/E4 it enters interval logic; and at
`r_known=2` — the value a sub-ROBUST rung is *expected* to produce, per
the design's own KW4.7 disclosure — the study fails to decide in
64–100% of surrounding configurations. **A single crash in a narrow
window silently changes the study's headline band, and nothing in the
report cross-checks the two views:** `ledger.attempts` would show 4
`COMPLETED` rows for that K while `primary.per_K["26"]` shows 3, and no
assertion anywhere compares them.

A second, distinct face of the same gap: the copy itself is not
specified as atomic. A crash mid-copy leaves a **truncated canonical
JSON**, which `harvest()` opens with a bare `json.load(open(p))`
(`:383`) → `JSONDecodeError` → the whole harvest dies after the full
≤15 GPU-h is spent. And the repair is blocked by G2's own exists-check,
which aborts loudly on any second write.

The load-bearing error is in the bonus claim §R4 uses to *delete* a
previously-specified build change:

> because a canonical-path file is written ONLY on `COMPLETED`
> acceptance, `discover_seeds_by_K`'s existing file-glob-presence count
> over the CANONICAL directory is now IDENTICAL to a status-based
> `n_completed` count by construction (`:1014-1020`)

The implication proved is **canonical ⇒ COMPLETED**. The identity
`glob-count == n_completed` requires the **biconditional**, and the
converse — **COMPLETED ⇒ canonical** — is exactly what the crash window
breaks. The claim is one-directional and the deleted patch was never
the thing that would have saved it; what is missing is a reconciliation
step, not a patch.

**Discharge condition.** Three lines:
(i) the canonical copy is ATOMIC (copy to `.tmp` in the canonical
directory, `os.replace`) so a truncated canonical file is unobservable;
(ii) **new recovery step 2b, run on every (re)start before the cell
walk:** for every `attempts[]` row with `status=="COMPLETED"`, verify a
parseable canonical file exists for that `(K, seed)` with
`status=="COMPLETED"`; if it is missing or unparseable, re-copy from
that row's recorded `outdir` (this reconciliation path is the ONE place
overwrite is permitted, and only from a source JSON that itself reads
`COMPLETED`); if the source is also missing or unparseable, rewrite the
row to `PERSISTENTLY-ABORTED` and disclose it in the report;
(iii) add to `validity_check`: for every K, the number of `COMPLETED`
rows in `ledger.attempts` equals `primary.per_K[K].n_seeds` — the
cross-check that makes this class of divergence loud instead of silent.
Then add each to §6 red-team item (ii), and extend item (vi)'s synthetic
kill to cover the post-fold/pre-copy window specifically.

### KW6.15 — **MINOR.** "`harvest()`'s existing code is the correct instrument AS-IS" needs to name WHICH fields, because two of them contradict the A4.9 fixed-denominator-4 guard on precisely the K's this design interval-resolves.

On an interval-resolved K (3 canonical files), `harvest()` computes
`rate = n_converged / n_seeds` = `n/3` (`:396`) and
`gate_eligible = n_seeds >= 4` → `False` →
`gate1_label = "SUB4-DISCLOSED-ONLY(n=3)"` (`:403-405`). The design's
own rule is *"a rate over the full fixed n=4"* (`:1573-1576`) and its
`classify()` consumes an integer COUNT `r∈{0..4}`, not a rate. The two
agree only on a complete K. An implementer told the existing code is
correct "AS-IS" and pointed at `per_K[K]["rate"]` gets `0.667` where
the design means `r=2`, and a `SUB4-DISCLOSED-ONLY` label where the
design means "apply interval logic." **Discharge:** one sentence — the
design consumes `per_K[K]["n_converged"]` (count) and `n_seeds` (as
`n_completed`) ONLY; `harvest()`'s own `rate`, `gate_eligible` and
`gate1_label` are computed against a 3-denominator on an incomplete K
and are never read by this design's band procedure.

---

## §3 G3 — EXIT-CODE CLASSIFICATION: **PARTIAL**

### Verified correct against the code

- exit `3` is `sys.exit(3)` from `rn.stop_requested(stop_file)` inside
  the training loop (`ncr_earlyln_scale.py:196-197`) — checked only at
  `log_every` boundaries during TRAINING, never during eval. G3's
  branch is exact and the "operator stop is terminal for the WHOLE
  wave" ruling is the right call.
- `status=="ABORTED-BUDGET"` returns at `:262-266` and exits **0** —
  G3's branch correctly keys on the JSON, not the exit code, so this
  case is caught.
- `assert rec["blank_out"]["passed"]` (`:300`) raises *before*
  `atomic_write_json` at `:307`, so a blank-out failure produces a
  non-zero exit with no JSON → `CRASHED-n`. Correct, and the right
  classification.
- A non-zero exit *after* a `COMPLETED` JSON is written falls into the
  `COMPLETED` branch — also correct (the science is done).

### KW6.5 — **MAJOR.** Three different enumerations of `attempts[].status` appear in the document and no two agree; and whether a `GATE-REFUSED` attempt produces a ledger row at all is asserted both ways.

| Location | Enumerated values |
|---|---|
| Output JSON schema, `:1069-1070` | `COMPLETED, ABORTED-BUDGET, CRASHED, CRASHED-RECOVERED, GATE-REFUSED, STOPPED-BY-OPERATOR` (6) |
| Recovery step 3 terminal test, `:936-937` | `COMPLETED, ABORTED-BUDGET, CRASHED, CRASHED-RECOVERED, **PERSISTENTLY-ABORTED**, STOPPED-BY-OPERATOR` (6 — `GATE-REFUSED` absent, `PERSISTENTLY-ABORTED` present) |
| §R4 KW5.3 row, `:2590` | "All four reachable non-`COMPLETED` attempt states (`ABORTED-BUDGET`, `CRASHED`, `CRASHED-RECOVERED`, `STOPPED-BY-OPERATOR`)" (5 incl. `COMPLETED`) |
| §R4 "numbers that moved", `:2615-2616` | `3 → 6 (COMPLETED/ABORTED-BUDGET/GATE-REFUSED + CRASHED, CRASHED-RECOVERED, STOPPED-BY-OPERATOR)` |

`PERSISTENTLY-ABORTED` is a CELL state (reached after two failed
attempts, or after a refused retry), not an attempt state; the recovery
rule asserts it appears as a **row in `ledger.attempts[]`**, which the
schema forbids. Conversely `GATE-REFUSED` is in the schema, but the
dispatch loop says a refusal leaves the **"ledger unchanged"**
(`:807-809`) and the resume rule says `GATE-REFUSED` "is only ever
produced for a cell/attempt with **no ledger row of any kind yet**"
(`:941-942`) — i.e. it explicitly has no row. Yet `run_status`'s own
`COMPLETE` definition is *"no `GATE-REFUSED` anywhere in the run"*
(`:1095-1097`), which requires it to be recorded somewhere, and
`attempts[]` is the only place the schema offers.

One further hole in the exit-code cross-product: of the 9 cells of
{exit 0, exit 3, other non-zero} × {no JSON, `COMPLETED` JSON,
`ABORTED-BUDGET` JSON}, the branch table covers 8. **(exit 0, no
JSON)** — reachable if `main` exits cleanly without dispatching, or if
a future CLI path returns early — matches no branch.

**Discharge condition.** (i) State that `GATE-REFUSED` DOES produce a
row, with `elapsed_h = 0.0` and `outdir = null`, and amend "ledger
unchanged" to "`realized_gpu_h` unchanged" and the resume test to "no
row with a DISPATCHED status"; (ii) state that `PERSISTENTLY-ABORTED`
is a derived CELL state, never an `attempts[]` value, and give the
derivation rule (a cell is `PERSISTENTLY-ABORTED` iff its attempt-2 row
is non-`COMPLETED`, or its attempt-1 row is non-`COMPLETED` and no
attempt-2 row exists and the retry gate was closed); (iii) add a
default branch `(exit 0, no JSON) → CRASHED-n`; (iv) correct KW5.3's
row and the "numbers that moved" line to the single reconciled list.

---

## §4 G4 — `run_status` ENUM + `validity_check`: **FAIL**

### KW6.4 — **MAJOR.** The new `validity_check`'s `d=K+1` assertion is unimplementable over the ledger schema it is specified against, and throws on exactly the degraded runs G4 exists to route to `completed/`.

> `all(a["K"]+1 == d_override_of(a) for a in ledger.attempts)`, i.e.
> every logged attempt's underlying cell carried `d==K+1`/
> `d_override==K+1`, **checked across the WHOLE ledger** rather than
> per-cell (`:1475-1479`)

The ledger row schema is `{K, seed, arm, attempt_n, elapsed_h, status,
outdir}` (`:906-908`) — **there is no `d` or `d_override` field.** So
`d_override_of(a)` must open the cell JSON under `a["outdir"]`. For a
`CRASHED`, `CRASHED-RECOVERED` or `GATE-REFUSED` row **no cell JSON
exists** (a crash dies before `atomic_write_json`; a gate refusal never
dispatched; and a `GATE-REFUSED` row has `outdir = null` at all).

`queue_worker.sh` runs `bash -c "$vcheck"` and routes on its exit
status alone (`:162-170`); job 108's own template is a
`python3 -c "…assert…"` one-liner. Any exception — `FileNotFoundError`,
`TypeError` on a null `outdir` — is a non-zero exit, so the job moves to
`failed/`. **Every run containing a single crash or a single gate
refusal is routed to `failed/`** — which is the precise defect class
KW5.4 identified and G4 was dispositioned to close, reintroduced by the
fix itself.

**Discharge condition.** Add `d_override` (or `d`) to the ledger row
schema, recorded at dispatch time from the CLI value the orchestrator
itself passes, and restate the assertion as
`all(a["K"] + 1 == a["d_override"] for a in ledger.attempts)` — pure
ledger arithmetic, no filesystem access, defined for every row
including `GATE-REFUSED`.

### KW6.6 — **MAJOR.** The `run_status` enum is neither mutually exclusive nor exhaustive over the terminal states the state machine reaches.

Cross-producting the state machine against the four definitions
(`:1095-1125`):

- **Not mutually exclusive.** `COMPLETE`'s definition carries two
  criteria that disagree: *"no `GATE-REFUSED` anywhere in the run"* and
  *"No budget-caused refusal anywhere."* A refused attempt-2 retry
  yields cell state `PERSISTENTLY-ABORTED`, **not** `GATE-REFUSED`
  (`:855-856`) — so it satisfies the first criterion while violating
  the second, and `COMPLETE-DEGRADED` sub-case (i)
  *primary-retry-refused* claims the same run. Two labels, one run.
- **Not exhaustive.** A run in which every primary AND every
  conditional first attempt is admitted but a **conditional cell's
  retry** is refused matches neither `COMPLETE` (a budget-caused
  refusal occurred) nor either enumerated `COMPLETE-DEGRADED` sub-case
  — (i) is scoped to a *primary* cell's retry, (ii) to a *conditional*
  cell's **first** attempt. The parent sentence covers it; the
  enumeration, which the disposition required be exhaustive
  ("the pre-registered degradation outcomes enumerated", `:2518-2519`),
  does not. An implementer with no label either invents one — which
  fails `validity_check`'s `run_status in {...}` membership test and
  lands the run in `failed/` — or silently mislabels.

**Discharge condition.** (i) Delete "no `GATE-REFUSED` anywhere in the
run" from `COMPLETE` and define it solely as "no budget-caused refusal
of any dispatch, first attempt or retry, anywhere in the run";
(ii) add sub-case (iii) *conditional-retry-refused* to
`COMPLETE-DEGRADED`, or reword "Two pre-registered sub-cases" to "the
sub-cases below, and any other budget-caused throttle strictly
downstream of the completed 12-cell primary baseline."

### KW6.7 — **MAJOR.** `validity_check` is over-permissive in the direction that matters: a total no-op passes all four assertions and is certified `completed/`.

The check asserts `run_status ∈ {COMPLETE, COMPLETE-DEGRADED,
EXHAUSTED-BUDGET}`, `realized_gpu_h_final <= 15.50`,
`realized_gpu_h_final == sum(elapsed_h)`, and the `d=K+1` clause
(`:1465-1479`). A report with `run_status="COMPLETE"`,
`realized_gpu_h_final=0.0`, `attempts=[]` satisfies **all four**
(`sum([]) == 0.0`, `all([]) == True`). The job moves to `completed/`,
and the pool will never re-run it.

That is not hypothetical: it is the exact shape of KW5.2's original
FATAL (a null harvest after full spend) and of KW6.2 above. Nothing in
the check ties the report to the work: no assertion on
`n_cells_attempted`, on `len(ledger.attempts)`, on `band.label` being a
member of §5's outcome set, on `trigger.resolution` being a member of
its enum, or on the smoke block being all-`PASS`. The check verifies
the report is *arithmetically self-consistent*, never that it is
*about anything*.

**Discharge condition.** Add, all cheap one-liners over the same
report: `n_cells_attempted == 12` (or `>= 12` with the conditional
arm); `len([a for a in attempts if a["arm"]=="primary"]) >= 12`;
`band["label"] in <the 10 partition labels + "INCOMPLETE-AT-K">`;
`trigger["resolution"] in {"unanimous","tie-break-min",
"TRIGGER-UNRESOLVED"}`; and `all(v=="PASS" for v in smoke.values())`.
Then update §6 red-team item (viii) to check the accept-set AND the
new content assertions.

### KW6.13 — **MINOR.** The `smoke` block is unfillable as specified.

`orchestrator_report.json` carries `"smoke": {"K26":"PASS"|"FAIL", …}`
(`:1072-1073`), but the 3 micro-smokes run "BEFORE the orchestrator is
queue-eligible" as a manual build/red-team gate (`:1504-1513`) — the
orchestrator does not run them and is given no way to learn their
results. **Discharge:** one clause — the orchestrator reads
`/home/nvidia/ncr/results_kwall_smoke/K{K}/earlyln_K{K}_s0.json` and
applies §4's own three-part pass criterion, or the results are passed
in via a CLI flag; and if any is missing or `FAIL`, the orchestrator
refuses to dispatch.

### KW6.14 — **MINOR.** `realized_gpu_h_final == sum(elapsed_h)` is an exact float-equality assertion used as a job-routing gate.

Exact equality survives only if the report's total is the same
IEEE-754 accumulation, in the same order, as the sum over the same
list, and if neither number is rounded for display anywhere in the
report path. That is plausible but nowhere stated, and the penalty for
being wrong is a real ≤15 GPU-h result routed to `failed/`.
**Discharge:** either write it as
`abs(realized_gpu_h_final - sum(...)) <= 1e-6`, or state in-text that
both figures are emitted unrounded from the same accumulation order.

---

## §5 G5 — TRIGGER PRECONDITION: **PASS**

Fully re-executed this round from the RULE TEXT alone (a fresh
~170-line enumeration written against §4's pseudocode and §5's six
rules, not adapted from any prior round's script). Per-K state space
modelled as the design specifies: `EXACT`×5, `DECIDED`×3
(`r_known∈{0,1,3}`), `AMBIGUOUS`×1 (`r_known=2`), `UNRESOLVED`×1 = 10
states, 10³ = 1000 vectors.

| Claim | Design | Re-executed | |
|---|---|---|---|
| old split (K-scan alone) | 844 / 156 | **844 / 156** | ✓ |
| paid-on-unresolved | 371 | **371** | ✓ |
| genuinely `(DECIDED, DECIDE)` | 473 | **473** | ✓ |
| new split under G5 | 473 / 527 | **473 / 527** | ✓ |
| paid-on-unresolved under G5 | 0 | **0** | ✓ |
| reverse direction (`TRIGGER-UNRESOLVED` while band decides) | 0/1000 | **0/1000** | ✓ |
| identity `473+371` | 844 | **844** | ✓ |
| identity `156+371` | 527 | **527** | ✓ |

**No conditional arm can dispatch at an excluded, unresolved, or
wide-tie K — verified exhaustively, 0 violations over all 1000
vectors.** Note this is now stronger than the pre-G5 rule: a vector
like `(EXACT r26=0, UNRESOLVED, ·)` did dispatch at `K_trig=26` under
Rev 3's rule while K=28 was unresolvable (the scan never reads 28); G5
blocks it because any `UNRESOLVED` K forces the band to
`INCOMPLETE-AT-K`. Good, and not claimed by §R4 — an unremarked bonus.

The 11-configuration table is **unaffected by G5**, as claimed: all 11
rows are band-agreeing at both candidates by construction, so
`classify_with_interval_logic` never returns `INCOMPLETE-AT-K` on them.
Independently re-derived, all 11 rows exact (incomplete K, `r_known`,
both candidate triples, band including `[NON-MONOTONE]` tag, candidate
`K_trig` set, and tie-break output).

The pseudocode's G5 block (`:629-634`) is correct as written: the
precondition is evaluated only after the K-scan decides, it overrides
to `TRIGGER-UNRESOLVED`, and it preserves the candidate as
`band_blocked_K_trig` rather than dropping it. The §5 rewrite at
`:1765-1784` describes the new asymmetry accurately (trigger-`DECIDED`
now implies band-decides; the converse still does not hold).

**G5 is the one disposition this round found nothing to say against.**

---

## §6 G6 / THE §R4 DISPOSITION TABLE: **PARTIAL** (11 of 13 rows verify)

Row-by-row against the revised text. All 13 rows' claims were checked
against the text they name.

| Row | Verdict |
|---|---|
| KW5.1 | claim matches text; mechanism as described — but see KW6.1/KW6.3/KW6.8 |
| KW5.2 | claim matches text; mechanism as described — but see KW6.2 |
| KW5.3 | text delivered; the row's own "all four reachable states" list is wrong (KW6.5) |
| KW5.4 | text delivered; enum defective (KW6.6), `validity_check` defective (KW6.4/KW6.7) |
| KW5.5 | ✓ verified exactly, both the mechanism and all six figures |
| KW5.6 | ✓ figures verified exactly (`{1:612,2:102,3:14,4:1}`, 15/15 wide ties → `INCOMPLETE-AT-K`); mis-attributed in the body (KW6.16) |
| KW5.7 | ✓ the sum `0.0126+0.0031=0.0157` is stated and used; the startup term is named as carried by the 0.30h margin, per the discharge |
| KW5.8 | ACCEPTED-COSMETIC is genuinely cosmetic **for §R3** — spot-checked: §7's E1/E4-uniformity bullet is real and correct. But §R4 repeats the defect in its OWN live table (KW6.9) |
| KW5.9 | ✓ all named fields present in the schema; `smoke` unfillable (KW6.13) |
| KW5.10 | ✓ `≤15.50 + ≤0.15 = ≤15.65` stated in both §4 and §6; the manual free-GPU-check disclosure is honest |
| KW5.11 | ✓ `NCR_ROOT=/home/nvidia/ncr` and all five paths absolute; the "informal shorthand" normative sentence is present |
| KW5.12 | ACCEPTED-COSMETIC, but its justification is **factually false** (KW6.11) |
| KW5.13 | ✓ scope note present with the audit's own counterexample reproduced and verified correct (`classify(1,0,0)=FRONTIER-AT-K*=24`, `classify(2,0,0)=GRADUAL-DECAY` — both re-executed); D5/E4 bullet amended |

### KW6.9 — **MINOR.** §R4's edit list is incomplete in exactly the way KW5.8 flagged for §R3 — and by §R4's own stated precedent, the ACCEPTED-COSMETIC ruling does not transfer to a LIVE table.

> §1–§7 … are UNCHANGED as historical record EXCEPT where a
> disposition explicitly required rewriting a section's content in
> place — **every such rewrite is listed in the "Where fixed" column
> below.** (`:2546-2550`)

`git diff -U0 HEAD~1 HEAD` = **45 hunks**, mapped to current sections:
1 header, **36 §4**, 1 §5, **5 §6**, **1 §7**, 1 (the appended §R4).

- **§7: 1 substantive hunk, unattributed.** New `:1996-2004` — the
  E1/E4-uniformity bullet gains `CRASHED`/`CRASHED-RECOVERED` and a new
  bolded G5 sentence. No §R4 row names §7. This is the *same bullet*
  KW5.8 was about, edited again, unattributed again.
- **§6: 5 hunks, 1 attributed.** KW5.10's row names "§6 ('Own cost
  ceiling' bullet)". Unnamed: the `15.0157h` update to that bullet
  (flows from KW5.1/KW5.7, neither of which names §6), and the
  resource/placement red-team rewrite adding items **(vi)–(ix)** for
  G1–G5 — a substantial, genuinely good addition that no row claims.

§R4's own KW5.8 row sets the governing precedent: the cosmetic
acceptance applies only once a table is frozen, and *"did NOT transfer
to `§R3` at the time of the R4 audit, because `§R3` was still the LIVE
revision log for the round then under review."* §R4 is the live
revision log now. **Discharge:** add "§7 (E1/E4-uniformity bullet)" to
the KW5.3/KW5.5 rows and "§6 (resource/placement red-team, items
(vi)–(ix); 'Own cost ceiling' bullet)" to the KW5.1/KW5.7/KW5.10 rows.
Two column edits, no prose change.

### KW6.11 — **MINOR.** KW5.12's "there is no live-body inconsistency to fix" is false, and half of KW5.12 is dropped without disposition.

> Re-verified this revision: the LIVE body … already uses ONE
> consistent digit, `1.0510`, in both places — **there is no live-body
> inconsistency to fix.** (`:2599`)

The `1.0510` half is correct — verified: `1.05105` appears only inside
frozen §R3 (`:2388`, `:2401`), and the live body uses `1.0510` at both
`:766` and `:1294`. But KW5.12 as filed also named the 160K nominals
being derived from a rounded `0.5106` rather than `0.510532`, and that
inconsistency **is live**: `:759` prints the K=26 80K nominal as
`0.5105` while `:782` uses `0.5106×1.9355=0.9882` for the same
quantity, 23 lines apart. Recomputed: `0.4680 × (9,421,568/8,636,672)
= 0.510532` → `0.5105`, so `:782`'s `0.5106` is the wrong digit. The
ceiling-table half of KW5.12 (`1.1073` vs `1.1072`, `1.1946` vs
`1.1945`, `2.1432` vs `2.1431`, `2.3121` vs `2.3120` — all
re-confirmed this round, all ≤0.0001h, all conservative) is not
mentioned by the row at all.

R4's discharge was "cosmetic; fix or ignore" so ignoring is licensed;
what is not licensed is a disposition that justifies itself with a
false statement of scope. **Discharge:** either change `:782` to
`0.5105` (one digit) or restate the row as "the live-body `0.5106`/
`0.5105` split and the ≤0.0001h ceiling-table deltas are both accepted
as cosmetic, in the conservative direction, and are left as printed."

### KW6.16 — **MINOR.** The 11-config table's scope correction is attributed to the wrong finding.

`:698` reads **"Scope correction (KW5.13, Rev 4)"** inside the
11-configuration table's closing paragraph. That paragraph is KW5.6's
fix, not KW5.13's — §R4's own "Where fixed" column assigns "§4
(11-configuration ambiguity table's closing paragraph, rewritten)" to
**KW5.6** and assigns the resolution-state scope note to **KW5.13**.
The body and the table contradict each other. **Discharge:** change
`:698` to `(KW5.6, Rev 4)`.

### KW6.12 — **MINOR.** The trigger pseudocode's branch-count comment is false, and the design contradicts it 100 lines later.

> `branches = cross_product_of_AMBIGUOUS(states)   # 1, 2, or 4
> candidate (r26,r28,r30) triples` (`:552`, and identically at `:617`)

Three simultaneously-`AMBIGUOUS` K's give `2³ = 8` branches. The design
knows this — `:706-708` describes "The 4-way case is exactly 'all three
K's `AMBIGUOUS` at `r_known=2`'", which is unreachable from ≤4
branches. Harmless to the results (`min()` is total over any candidate
set, and my 729-space sweep confirms the `{1:612,2:102,3:14,4:1}`
distribution), but it is a false comment in the one artifact a build
implementer transcribes literally. **Discharge:** change both copies to
`# 1, 2, 4, or 8 candidate triples (2^k for k AMBIGUOUS K's)`.

---

## §7 INTEGRITY — **PASS**

`git diff HEAD~1 HEAD` = 864 insertions / 213 deletions, 45 hunks
(`-U0`), one file. Section-level comparison of the `HEAD~1` and `HEAD`
texts, sliced identically in both:

| Section | Status |
|---|---|
| `§1`, `§2`, `§3` | **byte-identical** — matches the "not re-litigated" list ✓ |
| `§4` | CHANGED, 36 hunks — claimed extensively ✓ |
| `§5` | CHANGED, exactly one hunk (the `INCOMPLETE-AT-K` section) — claimed ✓ |
| `§6` | CHANGED, 5 hunks — **1 of 5 claimed** (KW6.9) |
| `§7` | CHANGED, 1 hunk — **unclaimed** (KW6.9) |
| `§A1`, `§R1`, `§A2`, `§R2`, `§A3`, `§R3` | **byte-identical** ✓ |
| `§A4-ADJUDICATION-KWALL` | body byte-identical; two lines appended (a blank + `---` separator ahead of the new §R4) |

**No frozen adjudication text was retconned.** The `§A4` delta is the
same separator-append artifact round 4 accepted for `§A3` ("nit only"),
and the §R4 MD5 table's claim is defensible under a body-scoped
section boundary — I record it here only so the identical pattern is on
record for a third consecutive revision. The `STATE.md` line-renumber
content (KW3.7) is unmoved. §2's `GRID_SHAPES`/`GRIDS` build note is
byte-unchanged and I independently re-confirmed its premise: `26`, `28`,
`30` are absent from `GRID_SHAPES` (`ncr_earlyln_scale.py:75-93`) and
from the `_gen_grid` extension loop (`ncr_task.py:161-162`), so the
additive build note is both necessary and correctly specified.

---

## §8 WHAT I COULD NOT BREAK — verified clean this round

Recorded so round 6 does not re-spend the effort.

1. **The 125-outcome partition.** Fifth consecutive independent
   regeneration: `18/4/12/8/12/8/42/15/4/2`, Σ=125/125, plus all 10
   representative rows including `(2,4,2)`'s `[NON-MONOTONE]` tag.
   Untouched by Rev 4.
2. **The whole pricing chain.** `F(24,25,64)=8,636,672`; ratios
   `1.0909/1.1829/1.2762/1.3706`; nominals `0.5105/0.5536/0.5973`;
   12-cell `6.6456` ("≈6.65"); the `1.9355×` maximum-ratio choice;
   160K nominals; all six informational ceilings; `1.2685/1.0510 =
   1.2069`; `2.00/1.2069 = 1.6571` ("≈1.66×"); `≈11.27h` nominal
   program. Every number correct to its stated digits modulo KW6.11's
   ≤0.0001h nits.
3. **KW4.7's decide-rates.** `16/25`, `17/25`, `25/25`, `43/80`,
   `38/80`, `36/80` → "45–54%" — fourth independent derivation, exact.
4. **The 11-configuration ambiguity table.** All 11 rows exact.
5. **G5's whole sweep** (see §5) — the single strongest part of Rev 4.
6. **The delivery model.** Strict sequencing genuinely removes KW4.2's
   concurrency hole; §6's pool-contract argument holds line by line
   against `idle_fallback_daemon.sh:10-16` and `queue_worker.sh`; the
   "self-enforced ceiling" honesty is intact; the new red-team items
   (vi)–(ix) are well-targeted (they would catch nothing in this
   round's findings, but they are the right tests for what they name).
7. **G2's duplicate-seed and flat-glob closure**, and the two
   collision risks I hunted and found closed (`ORCHESTRATOR_LEDGER.json`
   unmatched by the glob; `.axis_c_lock.json` excluded at `:365-366`).
8. **G3's exit-code branch against the real code** (`sys.exit(3)` at
   `:196-197`; `ABORTED-BUDGET` returning exit 0 at `:262-266`;
   `blank_out` assert pre-write at `:300`).
9. **The `d=K+1` collision defence's INTENT** — still well-targeted and
   worth keeping; only its implementation over the ledger schema is
   broken (KW6.4).
10. **The 0.30h supervisor margin.** Round 4's ACCEPTABLE-AS-DECLARED
    ruling is not reopened. I note only that KW6.3 now draws on it, so
    it is doing two jobs; that is an argument for fixing KW6.3, not for
    touching the margin.

---

## §9 VERDICT

**`REV-REQUIRED`.** 2 FATAL / 6 MAJOR / 9 MINOR.

**Forcing:** **KW6.1** (no ledger atomicity or unparseable-ledger
branch — the crash class G1 exists to survive includes one that
re-opens the full budget) and **KW6.2** (the post-fold/pre-copy crash
window silently deletes a `COMPLETED` cell from `harvest()`'s count,
permanently, and G2's "no patch needed" claim proves only the
direction that isn't the one `n_completed` needs).

**Also blocking a clean build, all in Rev-4-written text:** KW6.3 (the
crash-path bound claim is false for true spend), KW6.4 (`validity_check`
throws on every degraded run), KW6.5 (three disagreeing status enums),
KW6.6 (`run_status` neither exhaustive nor exclusive), KW6.7
(`validity_check` certifies a no-op as `completed/`), KW6.8 (false
`EXHAUSTED-BUDGET` with no escape hatch).

**Assessment for the adjudicator.** Rev 4 did the work: G5 is exact and
exhaustively verified, G6's figures reproduce to the digit, G3's
exit-code branch is right against the actual code, and G1's and G2's
mechanisms each close the FATAL they were written for. The round does
not terminate because the *composition* of G1, G2, G3 and G4 was again
not checked — the same diagnosis round 4 wrote about §R3, now one layer
in: the write-ahead ledger is durable in intent but not in its own file
format (KW6.1); the ledger fold and the canonical copy are each correct
and are ordered so that a crash between them is unrecoverable (KW6.2);
the status enum is defined three times and never reconciled (KW6.5);
and the `validity_check` written to stop rejecting good runs now
rejects them for a new reason (KW6.4) while accepting an empty one
(KW6.7).

None of this needs a seventh delivery model. **Every discharge is a
contract line inside the existing ORCHESTRATOR CONTRACT — an atomic
write, an abort-on-corrupt-ledger branch, one reconciliation step in
recovery, one field added to the ledger row schema, one reconciled
enum, and five one-liner assertions in `validity_check`.** A Rev 5 that
does those, plus the four bookkeeping MINORs, should clear a short
confirm round. I would recommend round 6 be scoped narrowly to the
crash-path composition (G1×G2×G3×G4) and the `validity_check`
one-liners, and NOT re-run the numeric sweeps — they are now five-fold
verified and untouched by anything above.

### Conditional build-release checklist (hand to the coordinator once Rev 5 clears)

Recorded now so it does not have to be reconstructed. Against the
frozen text of **§4 (ORCHESTRATOR CONTRACT: cell order, dispatch loop,
gate check points, ledger persistence + write-ahead recovery, G2
canonical-path contract, output JSON schema, job-spec template) and §6
(red-team items (i)–(ix))**, the build must produce:

1. **The orchestrator script** implementing §4's contract exactly —
   K-major cell order, write-ahead `open_attempt` (atomic), recovery
   with the KW6.2 reconciliation step, exit-code-exact branching,
   copy-on-accept with the exists-check, and the two `harvest()` calls.
2. **The additive grid registration** — `GRID_SHAPES[26/28/30]` and
   `for _K_new in (26,28,30): GRIDS[_K_new]=_gen_grid(_K_new)` (§2's
   build note), additive only, no existing key mutated.
3. **The single job-spec JSON** in job-108's 8-field format — absolute
   `cmd`, `output_dir=/home/nvidia/ncr/results_kwall_characterization`,
   `gpu_h_estimate: 15.50`, and the Rev-5 `validity_check` one-liner.
4. **The 3 per-K micro-smokes** (K=26/28/30, `--d-override K+1`,
   500 steps, `--ceiling-gpuh 0.05`, ≤0.15 GPU-h) run on a manually
   verified-idle GPU, all 3 passing §4's three-part criterion, BEFORE
   the orchestrator becomes queue-eligible.
5. **The pre-launch resource/placement red-team** covering §6's items
   (i)–(ix) plus the three new synthetic tests this round's discharges
   add: corrupt-ledger restart, post-fold/pre-copy kill, and a
   `GATE-REFUSED`-containing run through `validity_check`.
6. **The KW2.7 on-box sweep** of `~/queue/{fallback_pool,claimed}` for
   K∈{26,28,30} content — still outstanding, still mandatory.

*Focused audit round 5, 2026-08-06. Written from direct reads of
`matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md` (2686 lines:
§1–§7 in full, §A4-ADJUDICATION-KWALL and §R4 in full, §A1–§R3 by
section hash only), `NCR_KWALL_ATTACK_R4.md` (every FATAL/MAJOR/MINOR
heading, every discharge condition, §6–§9 in full),
`matrix-thinking/ncr/ncr_earlyln_scale.py` (`:75-93`, `:180-330`,
`:351-450`), `matrix-thinking/ncr/ncr_task.py` (`:120-169`),
`matrix-thinking/queue/queue_worker.sh` (`:150-175`), and
`matrix-thinking/queue/jobs/pending/108_laneA_main_K48_s0.json`. All
numeric claims come from one fresh session-local Python enumeration
written from the design's rule text alone (the six-rule `classify()`
over 125 triples; the trigger over 300 singly-, 240 doubly-, 729
non-`UNRESOLVED` and 1000 total state-space configurations; the
closed-form pricing chain), plus `git diff`/`git show` and a
section-level hash comparison of `HEAD~1` vs `HEAD` for the integrity
check. The enumeration script is a session-local scratch file, not
committed. No repo file other than this one was created or modified;
no command was run on the box; no job was launched; no git mutation was
made.*
