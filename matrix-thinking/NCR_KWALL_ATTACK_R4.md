# NCR K-WALL CHARACTERIZATION — FOCUSED AUDIT ROUND 4

**STATUS: COMPLETE — VERDICT `REV-REQUIRED`.**
**2 FATAL / 3 MAJOR / 8 MINOR. Forcing findings: KW5.1, KW5.2, KW5.3,
KW5.4, KW5.5.**

Target: `matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md`.
Header verified at `:3-4` — **`STATUS: DRAFT-R3 — POST-AUDIT-3, AWAITING
FOCUSED AUDIT ROUND 4 (not build-released, not queue-eligible).`** —
matches the expected string exactly. No mismatch to report.

Scope as charged: §A3's F1–F4 dispositions and everything §R3 touched.
Prior-settled material was checked for disturbance first (§Integrity)
and then re-verified opportunistically where re-execution was free
(the 125-partition, the decide-rates, the pricing chain) — all of it
reproduced to the digit and none of it is disturbed by §R3.

Method: the design's `classify()` six-rule procedure and F2's trigger
pseudocode were transcribed verbatim into Python and executed over
(a) all 125 complete triples, (b) all 300 singly-incomplete
configurations, (c) all 240 doubly-incomplete configurations, and
(d) the FULL reachable per-K state space (10 states per K → 1000 state
vectors, including the triply-incomplete case the design's own sweeps
excluded). The pricing chain (`F(K,d,h)`, nominals, ceilings, ratios)
was recomputed from the closed form. `queue/queue_worker.sh`,
`queue/idle_fallback_daemon.sh`, `queue/jobs/pending/108_laneA_main_K48_s0.json`
and `ncr/ncr_earlyln_scale.py` were read directly for the enforcement,
dispatch, and harvest semantics. Section-level byte hashes of the
`HEAD~1` and `HEAD` versions were compared for the integrity check.
No repo file other than this one was created or modified; no command
was run on the box; no job was launched; no git mutation was made.

---

## §0 SUMMARY

| # | Sev | One line |
|---|---|---|
| KW5.1 | **FATAL** | A mid-attempt orchestrator death zero-counts that attempt's spend; the ledger under-counts on the box's own documented restart path and nothing external backstops it — `15.0126h`/`15.50h` is not a bound. |
| KW5.2 | **FATAL** | The attempt-indexed outdirs (F1's KW4.1 fix) break `harvest()`'s discovery outright; the obvious build "fix" (recursive glob) corrupts the fixed-denominator-4 guard instead. The two R3 fixes are mutually incoherent. |
| KW5.3 | MAJOR | "or a non-zero subprocess exit → `ABORTED-BUDGET-1`" silently defeats the `--stop-file` kill switch (`sys.exit(3)`) and misfiles crashes as budget aborts; the report schema has no status value for either. |
| KW5.4 | MAJOR | `run_status`'s two values are never defined, and the job spec's own `validity_check` asserts `run_status=="COMPLETE"` — routing the design's pre-registered graceful-degradation outcome to `failed/`. |
| KW5.5 | MAJOR | The trigger dispatches a PAID 4-cell arm (≤9.28 GPU-h, >60% of the hard gate) in 371/1000 reachable state vectors where the primary band is `INCOMPLETE-AT-K` — which §5 EXCLUDES from frontier claims. Never pre-registered either way. |
| KW5.6 | MINOR | "no 3-way tie is ever produced by this rule" is false: 14 state vectors yield 3 candidates, 1 yields 4. The narrower band-agreeing parenthetical is true. `min()` still resolves everything. |
| KW5.7 | MINOR | "Only one of the two can apply to a single attempt" is false; the single-attempt tail also omits interpreter/CUDA-init/model-build time. True tail ≈0.016–0.021h, not 0.0126h. |
| KW5.8 | MINOR | §R3's "every such rewrite is listed in the 'Where fixed' column" is false — §7 was rewritten this revision and no row names §7. Same class as KW4.11, but in a LIVE table, so the frozen-section precedent does not apply. |
| KW5.9 | MINOR | `orchestrator_report.json` carries no field for the KW4.8 fix (`INCOMPLETE-AT-K`'s affected-K disclosure, the candidate bands), nor for the F3 smoke results or per-attempt exit codes. |
| KW5.10 | MINOR | The 3 micro-smokes (≤0.15 GPU-h) run outside the 15.50h ceiling, outside any pool spec, with no stated GPU or free-GPU-gate discipline. |
| KW5.11 | MINOR | Every results/ledger/smoke path is RELATIVE while the house spec convention (job 108) is absolute everywhere and `queue_worker.sh` runs both `cmd` and `validity_check` from its own CWD. |
| KW5.12 | MINOR | Rounding nits in the ceiling/nominal tables (≤0.0001h, all conservative); `1.05105` vs `1.0510` used as the same denominator two lines apart. |
| KW5.13 | MINOR | The §4 resolution-state table is headed "shared with D5/E4" but its `DECIDED → one fixed r-value` collapse is valid for the trigger's ROBUST-only scan ONLY; the band side must evaluate both candidates at every `r_known`. Concrete counterexample below. |

Per-disposition: **F1 NOT-DISCHARGED**, **F2 PARTIAL**,
**F3 DISCHARGED**, **F4 DISCHARGED-WITH-ONE-DEFECT**.
Integrity: **PASS** (one unattributed hunk, KW5.8).

---

## §1 F1 — ORCHESTRATOR CONTRACT: **NOT-DISCHARGED**

### What is right, verified not assumed

Before the defects, the parts that hold up under direct attack, because
they narrow what a revision has to touch:

- **(b) Ledger-before-status ordering is correctly specified in the
  contract text.** `:687-690`: *"`ledger.realized_gpu_h +=
  attempt_elapsed_h` UNCONDITIONALLY — before the resulting JSON's
  `status` is even inspected. Then branch on `status`…"* The ordering
  KW4.1 demanded is genuinely there, in that order, in the normative
  step-1 text. ✓
- **The wall-clock instrument is the right one.** `:684-687` measures
  `t0`/`t1` around `subprocess.run` and explicitly refuses the cell
  JSON's `gpu_h`. Re-verified against the code: `gpu_h` is assigned at
  `ncr_earlyln_scale.py:304` on the COMPLETED path only, and the
  `ABORTED-BUDGET` early return at `:262-266` writes `elapsed_s` but
  never `gpu_h`. The design's characterization of the harness is
  exact. ✓
- **(c) The `15.0126` induction is arithmetically correct for its
  stated premise.** `R_{N-1} + ceiling(N) ≤ 15.00` ⟹
  `R_N = R_{N-1} + elapsed(N) ≤ 15.00 − ceiling(N) + ceiling(N) + δ =
  15.00 + δ`. With `δ=0.0126` that is `15.0126`. The claim that
  sequencing converts Rev 2's N-attempt unpriced term into a 1-attempt
  term is correct reasoning, not hand-waving. (The `δ` value itself is
  wrong — KW5.7 — but only by ~0.008h.) ✓
- **(c) No crash-free path dispatches without its ceiling being
  checked.** Walked: attempt 1 → HARD GATE; attempt 2 → HARD GATE *and*
  RETRY GATE, both against a ledger already updated by attempt 1
  (`:693-698`); a refused retry spends nothing (`:699-701`). The
  conditional arm shares the same ledger and gates (§7's uniformity
  bullet, `:1494-1507`). There is no fourth dispatch site. ✓
- **(d) The declarative-ceiling question is answered HONESTLY by the
  design.** Read `queue/queue_worker.sh` end to end: it parses only
  `cmd`, `validity_check`, `output_dir` (`:139-141`), runs
  `( CUDA_VISIBLE_DEVICES="$GPU" bash -c "$cmd" )` with **no timeout,
  no wall-clock cap, and no read of `gpu_h_estimate` at all** (`:157`).
  `idle_fallback_daemon.sh` promotes specs by filename and likewise
  never reads a cost field. **Nothing anywhere enforces a job-level
  GPU-h ceiling — it is purely declarative.** The design does NOT claim
  otherwise: §6 `:1374-1381` states the ceiling is *"enforced ENTIRELY
  inside the orchestrator's own process … no external launcher … and no
  dependence on `queue_worker.sh`/`idle_fallback_daemon.sh` carrying any
  budget state (they carry none, and now need none)."* That is exactly
  right. ✓ **No finding.** (It does, however, mean KW5.1 has no
  external backstop — see below.)
- **(e) No single-GPU/`CUDA_VISIBLE_DEVICES` mismatch.** The worker
  exports `CUDA_VISIBLE_DEVICES="$GPU"` for the job's `cmd` (`:157`);
  the orchestrator's `subprocess.run` children inherit it, so every
  cell sees exactly one device. The sequential-single-GPU premise the
  worst-case derivation rests on is not merely a convention the build
  could violate — it is enforced by the dispatch environment. ✓ (Worth
  saying so in §6's red-team item (iii), which currently asks the
  red-team to confirm by inspection what the environment already
  guarantees.)
- **§6's structural fix is real.** The pool now holds one flat spec; the
  cumulative-cap-across-independent-specs contradiction KW4.3 found
  genuinely dissolves, because the two claims are no longer about the
  same object. Checked against the pool contract text at
  `idle_fallback_daemon.sh:10-16` verbatim. ✓

### KW5.1 — **FATAL.** A mid-attempt death zero-counts that attempt's spend; the ledger under-counts on the box's own documented restart path, with no external backstop.

**(a) of the charge, and it breaks.** The contract updates the ledger
only *after* `subprocess.run` returns:

> `t0` immediately before `subprocess.run(...)`, `t1` immediately after
> it returns, `attempt_elapsed_h=(t1-t0)/3600` … `ledger.
> realized_gpu_h += attempt_elapsed_h` UNCONDITIONALLY — (`:683-688`)

and persists it *after* every such update:

> `ORCHESTRATOR_LEDGER.json` … is rewritten after EVERY ledger update,
> not only at the end. A restarted orchestrator reads this file FIRST
> and resumes from the true recorded spend — never resets
> `realized_gpu_h` to 0 — (`:726-732`)

If the orchestrator process dies **between** `t0` and `t1` — SIGKILL,
OOM-killer, node hiccup, or the house's own documented preemption
contract — `t1` is never taken, `attempt_elapsed_h` is never computed,
no ledger row is written, and the file on disk still reflects the state
*before* that attempt started. The spend is **zero-counted**. On
restart the orchestrator reads a ledger that is short by up to one full
ceiling and re-opens budget it has already consumed. This is the SAME
class of defect KW4.1/KW4.2 were FATAL for — spend invisible to the
gate, a false "this IS the true cumulative spend" premise — reintroduced
through the one path §R3 explicitly claims to have covered.

The restart path is not hypothetical. It is the documented normal
operation of this box:

- `queue_worker.sh:62-64` documents the supervisor wrapper
  `while [ ! -f $QROOT/STOP ]; do bash queue_worker.sh <N>; sleep 15; done`;
- `queue_worker.sh:81-95` reclaims a crashed worker's own stale claims
  from `claimed/` back to `pending/` for a **from-scratch retry**, with
  the limitation stated in its own comment: *"this moves the spec back
  to pending/ for a FROM-SCRATCH retry, not a resume-from-checkpoint"*;
- `queue_worker.sh:34-36` documents that killing the worker's tmux
  session kills its in-flight job too — *"the intended preemption
  contract"*.

So: worker dies or is preempted mid-attempt → orchestrator dies
mid-attempt → spec returns to `pending/` → re-claimed → orchestrator
restarts → reads an under-counted ledger → keeps spending. Each cycle
loses up to `1.20h` (primary) or `2.32h` (conditional) from the ledger,
and **nothing caps the number of cycles**, because — per the verified
finding above — no external component enforces `15.50` or any timeout.
The stated `0.30h` supervisor margin cannot absorb a `1.20h` accounting
hole, and the `15.00` hard gate cannot fire on spend it cannot see.

The design's own worst-case sentence — *"`realized_gpu_h` at every gate
check IS the true cumulative spend, by construction of strict
sequencing, not by a convention that could be violated"* (§R3 KW4.2 row,
`:1896`; same claim at `:446-448`, `:788-792`) — is false on this path.
Strict sequencing makes the ledger exact **for attempts that return**;
it says nothing about attempts that do not.

There is a second, smaller face of the same gap: the contract specifies
budget resume but **no cell-level resume rule at all**. On restart, does
the orchestrator re-walk the full cell order? If yes, a re-dispatch into
the same `attempt1/` outdir is cheap (the harness's own
`status=="COMPLETED"` skip, `ncr_earlyln_scale.py:240-245`, fires) — but
the HARD GATE is checked *before* dispatch and charges the full `1.20`,
so a restart at `realized≈13h` turns already-COMPLETED cells into
`GATE-REFUSED`, a state the contract says is *"treated identically to
MISSING … by `harvest()`"* (`:679-680`) — which is itself false, since
`harvest()` reads the on-disk `COMPLETED` JSON and cannot see the
orchestrator's in-memory refusal. Two sources of truth that disagree
exactly when a restart happens.

**Discharge condition.** Write a PENDING ledger row —
`{K, seed, arm, attempt_n, dispatch_ts, charged_ceiling, status:
"IN-FLIGHT"}` — and flush it **before** `subprocess.run`, then replace
it with the measured row after `t1`. On restart, any row still
`IN-FLIGHT` is charged its FULL ceiling (the conservative choice) before
any gate check runs. Separately, specify the cell-level resume rule
explicitly: which cells are re-walked, whether a cell with a recorded
terminal state is re-dispatched at all, and that a cell with an on-disk
`COMPLETED` JSON is never re-gated (its state comes from the record, not
from a fresh HARD-GATE decision). Reconcile the
`GATE-REFUSED`-vs-`harvest()` claim at `:679-680` with whichever answer
is chosen.

### KW5.2 — **FATAL.** The attempt-indexed outdirs break `harvest()`'s discovery; the obvious build fix corrupts the fixed-denominator-4 guard instead.

F1's KW4.1 fix and F1's harvest patch are mutually incoherent, and the
design contains no text reconciling them (verified by grep: `glob`,
`discover_seeds_by_K`, `attempt1`, `attempt2`, `recurs` — no occurrence
addresses this).

The cells are written to per-attempt subdirectories:

> `--outdir results_kwall_characterization/K{K}_s{seed}_attempt{n}`
> (`:461`)
> `{n}∈{1,2}`: attempt 1's outdir is never reused for attempt 2 …
> `run_earlyln_cell`'s existing `--outdir` flag … is the ONLY mechanism
> needed. (`:465-475`)

and the harvest is invoked on the PARENT:

> Once all 12 primary cells are terminal, `harvest()` runs ONCE over
> `results_kwall_characterization/` … (`:735-737`)

Against the actual code that is a null read:

- `discover_seeds_by_K(outdir)` globs
  `os.path.join(outdir, f"{cell_id(K,'*')}.json")` — i.e.
  `outdir/earlyln_K{K}_s*.json`, **non-recursive**
  (`ncr_earlyln_scale.py:358-370`);
- `harvest(outdir, seeds_by_K)` opens
  `os.path.join(outdir, f"{cell_id(K,seed)}.json")` — same flat
  construction (`:376-380`).

Every cell JSON lives one directory deeper, at
`results_kwall_characterization/K26_s0_attempt1/earlyln_K26_s0.json`.
Discovery returns the empty tuple for all three K's. `n_completed=0 ≤ 2`
at every K, so §4's own rule (`:1008`) returns `INCOMPLETE-AT-K`
unconditionally for the study, the trigger returns
`TRIGGER-UNRESOLVED`, and the `validity_check` fails — **after the full
≤15 GPU-h has been spent.** The data survives on disk, so this is
recoverable by a re-harvest, but the specified artifact, built exactly
as specified, produces no verdict.

The natural build-stage repair is worse than the disease. Switching to a
recursive glob makes `discover_seeds_by_K` match *both* attempts of a
retried cell — two files with the SAME basename in different
directories. `seeds_by_K[K].append(int(m.group(1)))` (`:365`) does not
dedupe, and `tuple(sorted(...))` preserves duplicates, so a K with one
retried seed reports `n_seeds = 5`. `rate = n_converged / n_seeds`
(`:392`) then divides by 5. That silently breaks the A4.9 fixed-
denominator-4 guard the design leans on in three separate places
(`:934-938`, `:1116-1119`, `:1306-1307`) — and it fails *quietly*, with
a plausible-looking wrong rate, rather than loudly with a null read.

The design's build-stage harvest instruction (`:1002-1008`) specifies
only the `n_seeds → n_completed` status change. It never specifies the
discovery path, recursion, per-seed dedupe, or which attempt is
authoritative when both exist. §6's red-team item (ii) likewise only
checks *"E4's `harvest()` `n_completed`-vs-`n_seeds` patch is actually
applied."* Neither would catch this.

A third, smaller instance of the same gap: *"`harvest()` runs a SECOND
time over the combined results"* (`:742-744`) is not expressible in
`harvest(outdir, seeds_by_K)`'s single-`outdir` signature — the two arms
write to `results_kwall_characterization/` and
`results_kwall_characterization_160k/`.

**Discharge condition.** Specify, in the ORCHESTRATOR CONTRACT: (i) the
discovery mechanism for the attempt-subdir layout (recursive glob, or a
symlink/copy of each cell's authoritative JSON into a flat harvest
directory, or discovery driven off the ledger's `attempts[]` rather than
the filesystem — the ledger is the cleaner source of truth and already
carries `outdir` per attempt); (ii) the attempt-precedence rule
(highest `attempt_n` wins) and an explicit per-seed dedupe so `n_seeds`
can never exceed 4; (iii) how the second, combined harvest spans two
result trees. Then add each to §6's red-team item (ii).

### KW5.3 — **MAJOR.** Non-zero exit is folded into `ABORTED-BUDGET`, defeating the STOP kill switch and misfiling crashes.

> `ABORTED-BUDGET` (or a non-zero subprocess exit) → state
> `ABORTED-BUDGET-1`, proceed to step 2. (`:690-692`)

The runner's STOP path is `sys.exit(3)`:

> `if rn.stop_requested(stop_file): print("  STOP file seen -- exiting
> 3"); sys.exit(3)` (`ncr_earlyln_scale.py:196`)

and the design itself creates the sentinel path it reacts to
(`--stop-file results_kwall_characterization/STOP`, `:462`). Under the
quoted rule, an operator touching that STOP file does not stop the
program — the orchestrator reads exit 3 as `ABORTED-BUDGET-1`,
immediately re-dispatches attempt 2 (which exits 3 again in seconds),
marks the cell `PERSISTENTLY-ABORTED`, and moves to the next cell. All
16 cells burn through to `PERSISTENTLY-ABORTED` in under a minute. The
house's own kill switch is converted into a mass-abort. (Note also that
on the `sys.exit(3)` path no cell JSON is written at all, so the
orchestrator has nothing but the exit code to branch on.)

The same conflation misfiles genuine crashes — a shape bug at K=30, an
OOM, an import failure — as budget aborts: each is charged a retry it
cannot benefit from, and is recorded in `orchestrator_report.json` as
`"status":"ABORTED-BUDGET"`, a value whose enum (`:761`) has no other
option to offer. Downstream, `ABORTED-BUDGET`→`PERSISTENTLY-ABORTED`
feeds the interval logic as an *unknown-numerator* seed, which is the
right treatment for a budget abort and the wrong one for a deterministic
crash (a crash at K=30 is not a coin-flip seed, it is a broken config).

**Discharge condition.** Branch on the exit code, not on its
non-zero-ness: `3` → STOP, orchestrator flushes the ledger and exits
without dispatching anything further; any other non-zero with no cell
JSON → a distinct `CRASHED` state with its own retry policy (0 or 1,
stated) and its own value in the report `status` enum; `ABORTED-BUDGET`
read from the cell JSON keeps the current path. State how `CRASHED`
feeds the resolution-state table.

### KW5.4 — **MAJOR.** `run_status` is undefined, and the spec's own `validity_check` rejects the design's pre-registered degraded outcome.

The schema offers two values and defines neither:

> `"run_status": "COMPLETE" | "GATE-EXHAUSTED-PARTIAL"` (`:755`)

Nothing in §4 says when the second is emitted — any `GATE-REFUSED`? only
conditional-arm refusals? a refused retry? — and the design's worst-case
paragraph declares that branch *correct behaviour*:

> the hard gate correctly THROTTLES OR REFUSES part or all of the
> conditional arm rather than exceeding 15.00 — degrading gracefully to
> `TRIGGER-UNRESOLVED`/reduced conditional coverage, never to a silent
> overrun. (`:848-852`)

Yet the job spec written three paragraphs later asserts the opposite:

> `validity_check` asserts … `run_status=="COMPLETE"`,
> `ledger.realized_gpu_h_final <= 15.50`, and … (`:1024-1031`)

`queue_worker.sh:161-169` uses exactly that check to decide
`completed/` vs `failed/` — *"That check, not the cmd's exit code,
decides completed/ vs failed/"* (`:26-30`). So the design's own
graceful, disclosed, publishable degradation is filed by the queue as a
FAILED job. (There is no auto-requeue of `failed/` anywhere in
`queue/` — verified — so this does not cause a re-run loop; the cost is
that a legitimate result is recorded as a failure and, by house
convention, not trusted.)

**Discharge condition.** Define both `run_status` values normatively in
the ORCHESTRATOR CONTRACT, and make `validity_check` accept both
(assert on ledger self-consistency and attempt-record completeness
instead — e.g. `run_status in {"COMPLETE","GATE-EXHAUSTED-PARTIAL"}`,
`realized_gpu_h_final <= 15.50`, `realized_gpu_h_final == sum(elapsed_h)`,
plus the existing `d==K+1` check).

---

## §2 F2 — TRIGGER RULE: **PARTIAL**

### Re-execution results (the mandated checks, all passed)

The decision rule was implemented from the pseudocode at `:517-531`
(scan `K=26,28,30,32` in order; `r24=4`, `r32=0`; `AMBIGUOUS` states
branch into a candidate cross-product; an `UNRESOLVED` K encountered
before the trigger K returns `TRIGGER-UNRESOLVED`; ties resolve by
`min()`), and run against every configuration the state machine permits.

- **The 11-configuration table reproduces EXACTLY** — all 11 rows, in
  order, with the same incomplete K, the same `r_known=2`, the same
  candidate triples, the same bands, the same `K_trig` candidate sets
  (`{26,28}` ×9, `{28,30}` ×2), and the same tie-break results. Nothing
  in the printed table is wrong. ✓
- **`K_trig` is TOTAL.** Over the full reachable state space — each K
  independently in `EXACT(r=0..4)`, `DECIDED(r_known∈{0,1,3})`,
  `AMBIGUOUS(r_known=2)`, or `UNRESOLVED`; 10³ = **1000 state
  vectors** — the rule returns `DECIDED` in 844 and
  `TRIGGER-UNRESOLVED` in 156. **Zero exceptions, zero non-terminating
  cases, and no waiting state anywhere.** The Rev-1/Rev-2 deadlock is
  genuinely retired. ✓
- **No conditional arm ever runs at an excluded K.** `DECIDED` with
  `K_trig ∈ {26,28,30}` where that K is itself `UNRESOLVED`: **0 of
  1000**. F2's exclusion rule is airtight — the scan cannot reach past
  an `UNRESOLVED` K without returning. ✓
- **The `TRIGGER-UNRESOLVED`-while-band-decides claim at `:1310-1313`
  is true.** Joint distribution over 1000 vectors:
  `(DECIDED, DECIDE)` 473, `(DECIDED, INCOMPLETE-AT-K)` 371,
  `(TRIGGER-UNRESOLVED, INCOMPLETE-AT-K)` 156,
  `(TRIGGER-UNRESOLVED, DECIDE)` **0**. ✓
- **KW4.7's decide-rates reproduce exactly**: 16/25, 17/25, 25/25 at
  `r_known=2`; and 43/80 (53.8%), 38/80 (47.5%), 36/80 (45.0%) for the
  two-K case — the design's "45–54%". ✓
- **E5's 125-outcome partition reproduces exactly** (10 band rows,
  18/4/12/8/12/8/42/15/4/2, Σ=125). Untouched by §R3 and still correct.
  ✓

### KW5.5 — **MAJOR.** The paid conditional arm fires in state vectors where the primary band is `INCOMPLETE-AT-K`, and the design never pre-registers whether it should.

§5 is unambiguous that `INCOMPLETE-AT-K` is not a publishable frontier
read: *"explicitly EXCLUDED from frontier claims"* (`:958`, `:1305-1306`).
§4 is equally clear that the trigger is evaluated independently of the
band and that the two *can* disagree (`:499-503`, `:939-943`). What the
design never states is what happens in the intersection.

Measured over the design's own state machine: **371 of 1000 reachable
state vectors dispatch a PAID 4-cell 160K arm while the primary band is
`INCOMPLETE-AT-K`.** In **122** of those, the arm is dispatched *at a K
whose own 80K rate is itself unresolved* (`AMBIGUOUS`). Smallest
concrete example: `states = (EXACT, EXACT, AMBIGUOUS)`,
`r = (0, 0, r_known=2)` → trigger `DECIDED, K_trig=26, unanimous`, band
`INCOMPLETE-AT-K` (candidates `FRONTIER-AT-K*=24` vs
`FRONTIER-AT-K*=30 [NON-MONOTONE]`).

The spend at stake is not marginal: 4 cells × `--ceiling-gpuh 2.32` =
**9.28 GPU-h**, more than 60% of the 15.00 hard gate, committed to
budget-qualifying a `K_trig` inside a study whose headline verdict is,
by the design's own rule, unreportable as a frontier claim. Whether that
is a good trade is genuinely arguable — a 160K rate at `K_trig` is a
standalone datum, and §5's qualifier bands are defined purely on the
160K rate with no dependence on the 80K band. But a >9 GPU-h branch of
a 15 GPU-h design must be *pre-registered*, and this one is silent.

A related, milder case: in **37** state vectors the `min()` tie-break
selects a PAID K while a live candidate branch says `K_trig=32` — the
`$0`, already-archived branch. This one IS covered by the stated
rationale (`:547-551`, "the SMALLEST disagreeing candidate K runs"), so
it is disclosed, not a defect; it is recorded here because a reviser
choosing a `$0`-preferring tie-break should know the size of the
affected set.

**Discharge condition.** Pre-register one sentence, either way:
(a) the conditional arm is SUPPRESSED when the primary band is
`INCOMPLETE-AT-K` (and the run reports the trigger's `K_trig` as a
disclosed non-dispatched candidate); or (b) it runs, with a stated
sentence on what the 160K qualifier buys when the 80K band is
unreportable. Either is attackable and defensible; the silence is not.

### KW5.6 — **MINOR.** "no 3-way tie is ever produced by this rule" is false.

> every disagreement is between exactly two adjacent K's, so `min()`
> always resolves it; no 3-way tie is ever produced by this rule
> (verified by an extended sweep of the two-K-simultaneously-incomplete
> case … the maximum number of distinct `K_trig` values in any
> band-agreeing configuration is **2**, never 3+). (`:579-585`)

The **parenthetical is TRUE** — I re-swept it: over every resolvable
state vector, configurations whose band agrees at all candidates and
whose `K_trig` candidate set has ≥3 members number **0**. ✓

The **lead sentence is FALSE**. The design's sweeps stopped at two
simultaneously-incomplete K's; the state machine permits three. Over the
full space: candidate-set sizes are `{1: 612, 2: 102, 3: 14, 4: 1}`.
The 4-way case is `all three K's AMBIGUOUS at r_known=2`, whose
candidate set is exactly `{26,28,30,32}`. Since the trigger is
evaluated *independently of the band* — the central point of F2 — the
band-agreeing scoping does not rescue the unqualified sentence, and
"every disagreement is between exactly two adjacent K's" is wrong on 15
state vectors.

Nothing breaks: `min()` is total over any candidate set, and all 15
cases already land in `INCOMPLETE-AT-K` for the band. The finding is
that a freshly-computed R3 verification claim overstates its own
coverage.

**Discharge condition.** Either restate the sentence with its true
scope (band-agreeing, ≤2 incomplete K's) or extend the sweep to the
triply-incomplete case and report `{1:612, 2:102, 3:14, 4:1}`. Note
that the tie-break rationale ("nearest the live rung") still reads fine
for a 4-way set, so no rule change is needed.

### KW5.13 — **MINOR.** The shared resolution-state table's `DECIDED` collapse is trigger-only, and is not labelled as such.

The table at `:508-513` is introduced as *"Per-K resolution state
(shared with D5/E4…)"* and assigns `n_completed=3, r_known∈{0,1,3}` the
state `DECIDED` with *"one fixed `r`-value — `ROBUST(r_known)==ROBUST
(r_known+1)` for every value except 2"*. That collapse is valid for the
trigger's K-scan, which reads only `ROBUST`. It is **invalid for band
classification**, whose rules 1–4 test `r ≤ 1`, not `ROBUST(r)`.

Counterexample, executed: K=26 incomplete with `r_known=1`, `r28=r30=0`.
`classify(1,0,0)` → rule 4 fires (`r26=1≤1`) → `FRONTIER-AT-K*=24`.
`classify(2,0,0)` → rule 4 fails (`2>1`), rule 5 fires → `GRADUAL-DECAY`.
Different bands, so D5/E4 (`:944-956`) correctly requires
`INCOMPLETE-AT-K` — but a build implementer treating the "shared" table
as authoritative would collapse to one candidate and report a decided
band.

The third column IS headed "Contribution to the trigger scan", so a
careful reader gets it right; the header "shared with D5/E4" and the
"one shared computation" sentence at `:939-943` point the other way.
Since F1 explicitly asks the build stage to implement from this design
alone, the ambiguity is load-bearing.

**Discharge condition.** One clause: the `DECIDED`-collapse applies to
the trigger's `ROBUST`-only scan ONLY; band classification evaluates
`classify()` at BOTH interval candidates for every `r_known`, including
0, 1, and 3. What is genuinely shared is the `n_completed` count.

---

## §3 F3 — PER-K MICRO-SMOKE: **DISCHARGED**

Buildable as specified, and the smoke/`validity_check` separation is
clean.

- **Command shape** (`:1048-1051`): `ncr_earlyln_scale.py --cell --K {K}
  --d-override {K+1} --seed 0 --steps 500 --ceiling-gpuh 0.05 --outdir
  results_kwall_smoke/K{K}`. Every flag exists on the R0 CLI surface
  (verified at `ncr_earlyln_scale.py:854-882`). The `--K` `choices=
  sorted(GRID_SHAPES)` constraint that would reject `26` is correctly
  handled upstream by §2's additive build note (`:178-195`), which adds
  `GRID_SHAPES[26/28/30]` and the matching `GRIDS[_K_new]=_gen_grid(...)`
  entries — and `GRID_SHAPES[26]=dict(d=52,h=64)` satisfies the file's
  own `d=2K, h=64` self-test assertion at `:564`. The chain closes. ✓
- **Pass criterion** (`:1057-1063`): exit without uncaught exception,
  `status ∈ {COMPLETED, ABORTED-BUDGET}`, and `K`/`d`/`d_override`
  equal to `K`/`K+1`/`K+1` in the JSON. All three fields are genuinely
  written by `run_earlyln_cell` (`rec = dict(cell_id=..., K=K, d=d_eff,
  d_default=..., d_override=d_override, ...)`, `:239-246`). Mechanically
  checkable, no interpretation required. ✓ The 500-step/0.05h sizing is
  sound: at the K=24 measured throughput (0.468h / 80,000 steps) 500
  steps is ≈11s against a 180s ceiling, and the post-train instrument
  sequence adds ≈6–27s at the design's own measured eval fraction. ✓
- **Gate placement** (`:1064-1069`): build-release gate — *"the
  orchestrator script is not queue-eligible until all 3 micro-smokes
  (K=26,28,30) pass"* — and re-stated as red-team item (iv) in §6
  (`:1410-1412`). Two independent placements, both in the live body. ✓
- **`validity_check` is never called a smoke.** All 12 occurrences of
  "smoke" in the document were read; the only ones touching
  `validity_check` say the opposite explicitly (`:1041-1044`:
  *"a harvest-time assertion on a COMPLETED production cell, not a
  smoke test … it is never called a smoke anywhere in this document"*).
  KW4.6's false cross-reference is gone and no new one replaces it. ✓

Two MINORs attach, neither blocking:

### KW5.10 — **MINOR.** Smoke spend is unbudgeted and unplaced.

3 cells × `--ceiling-gpuh 0.05` = **0.15 GPU-h** run *before* the
orchestrator is queue-eligible — therefore outside the 15.50h declared
ceiling, outside any pool spec, and outside the orchestrator's ledger.
The design never says which GPU they run on or that they must respect
`queue_worker.sh:107-115`'s free-GPU gate (which governs pool jobs
only). Small, but the design's whole §6 posture is "one flat spec, one
declared ceiling."
**Discharge:** one sentence disclosing total program spend as
`15.50 + ≤0.15` and stating the smokes run on a verified-idle GPU (or
inside the orchestrator itself as a pre-flight step, charged to the
ledger — which would also fold them under the single declared ceiling).

### KW5.11 — **MINOR.** Relative paths against an absolute-path house convention.

Every path in §4's command blocks is relative:
`results_kwall_characterization/…` (`:461-462`),
`results_kwall_characterization_160k/…` (`:596-597`),
`results_kwall_smoke/K{K}` (`:1050`),
`results_kwall_characterization/ORCHESTRATOR_LEDGER.json` (`:726`).
The design requires absolutes only for the orchestrator's own `cmd`
(`:1019-1022`). The house convention is absolute everywhere —
job 108's `cmd` is `cd /home/nvidia/ncr && /home/nvidia/tdenv/bin/python3
… --outdir /home/nvidia/ncr/results_earlyln_scale …` with an absolute
`output_dir` and an absolute path inside `validity_check`. And
`queue_worker.sh` runs both `bash -c "$cmd"` (`:157`) and
`bash -c "$vcheck"` (`:162`) from the *worker's* CWD, and `mkdir -p`s
the spec's `output_dir` (`:148`), so a relative results tree lands
wherever the supervisor happened to start.
**Discharge:** require absolute paths for the results trees, the ledger,
the report, and the smoke outdirs, matching job 108.

---

## §4 F4 — §R3 DISPOSITION TABLE: **DISCHARGED-WITH-ONE-DEFECT**

**Coverage: complete.** All 11 findings from `NCR_KWALL_ATTACK_R3.md`
(KW4.1–KW4.11, confirmed against that file's own section headings) plus
both round-3 PARTIALs (E4, E6) have rows — 13 rows, no silent leftovers.
✓

**FATAL/MAJOR rows, checked in full against the revised text:**

| Row | Claim | Verified |
|---|---|---|
| KW4.1 FATAL | wall-clock timer, unconditional ledger update, attempt-indexed outdirs | All three mechanisms present at `:683-692`, `:465-475`. Timer/ordering ✓. **But "all three grounds … closed by construction" overstates**: the overwrite ground is closed at the cost of KW5.2, and the ledger ground is open on the KW5.1 path. |
| KW4.2 FATAL | concurrency removed; ledger exact by construction | Strict sequencing stated `:667-673`, `:785-792`; induction recomputed and arithmetically correct. **"IS the true cumulative spend, by construction" false on the restart path (KW5.1).** |
| KW4.3 FATAL | one orchestrator job; §6 rewritten | §6 rewritten in full and checked against `idle_fallback_daemon.sh:10-16` verbatim; the pool now sees one flat spec; `queue_worker.sh` confirmed to carry no budget state, exactly as claimed. **DISCHARGED, no residue.** ✓ |
| KW4.4 MAJOR | gate charges the CLI value directly | `:711-718` charges `1.20`/`2.32`; the per-K `max(2×nominal,1.0)` table relabelled INFORMATIONAL at `:857-873` and feeds nothing. Recomputed: 1.0211/1.1072/1.1945 and 1.9763/2.1431/2.3120 — `1.20` clears every primary floor, `2.32` clears `2.3120`. **DISCHARGED.** ✓ |
| KW4.5 MAJOR | new trigger rule, tie-break, 11 configs enumerated | Re-executed: 11/11 rows exact, totality 1000/1000, deadlock retired. **DISCHARGED** as to the defect it names; residue is KW5.5/KW5.6, which are new. ✓ |
| KW4.6 MAJOR | real per-K micro-smoke, build-gated | See §3. **DISCHARGED.** ✓ |

**MINOR rows, fix located in each case:** KW4.7 — the decide-rate bullet
is at `:961-974` and its six figures reproduce exactly ✓. KW4.8 —
`INCOMPLETE-AT-K` restated as a study-level verdict at `:953-960` and
`:1297-1301` ✓ (but see KW5.9). KW4.9 — the `log_every` overshoot is
priced at `:804-807` ✓ (but see KW5.7). KW4.10 — `1.2069`/`1.207` at
`:882-885`, recomputed `1.2685/1.05105 = 1.206888…` ✓, and
`2.00/1.2069 = 1.65714` → "≈1.66×" ✓. KW4.11 — accepted-cosmetic, no
edit made, and §R2 confirmed byte-identical by hash ✓.

### KW5.9 — **MINOR.** The output schema does not carry the KW4.8 fix.

§4 and §5 both require `INCOMPLETE-AT-K` to be *"disclosed carrying the
affected K(s) as a field"* (`:958-960`, `:1304-1305`) and, in the
multi-K case, *"both/all candidate bands disclosed"* (`:986-987`). The
`orchestrator_report.json` schema written in the same revision offers
only `"band": {"label": str, "non_monotone_tag": bool,
"interval_resolved_Ks": [int,...]}` (`:769-770`) — no affected-K field,
no candidate-band list. Over the state space this is the outcome in
527/1000 reachable vectors. Also absent: the F3 micro-smoke results,
per-attempt exit codes (needed once KW5.3 is fixed), and the assigned
GPU/git commit.
**Discharge:** add `"incomplete_at_K": [int,...]|null` and
`"candidate_bands": [str,...]|null` under `band`, and a
`"smoke": {...}` block.

### KW5.8 — **MINOR.** §R3's edit list omits §7; the "every such rewrite is listed" sentence is false.

> `§1–§7`, `§A1-ADJUDICATION`, `§R1`, `§A2-ADJUDICATION`, `§R2`, and
> `§A3-ADJUDICATION` are UNCHANGED as historical record EXCEPT where a
> disposition explicitly required rewriting a section's content in
> place — **every such rewrite is listed in the "Where fixed" column
> below.** (`:1868-1872`)

§7 WAS rewritten this revision — the E1/E4-uniformity bullet at
`:1494-1507` is substantively rewritten (verified by section diff:
"the launcher's cumulative-GPU-h check (§4) gates every launch/retry" →
"the orchestrator's single ledger and HARD/RETRY gates (§4) govern every
dispatch in the whole program, in the SAME sequential stream … no
separate ledger, no separate gate, and no separate GPU for the
conditional arm"). No row's "Where fixed" column names §7. (§R2's own
table did name §7 for the same bullet, so the omission is a regression
in bookkeeping, not a house-convention difference.)

This is the KW4.11 defect repeated — but KW4.11 landed in a **frozen**
`§R2` table, which is precisely why R3 accepted it as cosmetic
("editing a frozen table … trades a documented correction for an
invisible retcon"). §R3 is the **live** revision log for the round now
under review; that precedent does not transfer.
**Discharge:** add `§7` to the E4 or E6 row's "Where fixed" column.

---

## §5 WORST-CASE ARITHMETIC — recomputed

### KW5.7 — **MINOR.** The single-attempt tail is understated on two counts.

> Attempt `N`'s own true `elapsed_h` can exceed `ceiling(N)` by at most
> ONE of: eval overhead, if `N` reaches `COMPLETED` (max observed
> **0.0126 GPU-h** …); or the `log_every=500`-step training-
> ceiling-check granularity … (max observed **0.0031 GPU-h** …), if `N`
> instead aborts. **Only one of the two can apply to a single
> attempt.** (`:800-808`)

**(i) Both can apply to the same attempt.** The ceiling test fires only
inside the `if step % log_every == 0 or step == 1:` block
(`ncr_earlyln_scale.py:191-199`). A training run whose last check at
step 79,500 reads `elapsed = ceiling_s − ε` proceeds through 500 more
steps, crosses `ceiling_s` with no further check, exits the loop at
`steps` and returns **`COMPLETED`** — and then the unbounded eval phase
runs. Overshoot **and** eval, on the same attempt: `0.0031 + 0.0126 =
0.0157`.

**(ii) A third term is unpriced.** The ledger measures orchestrator
wall clock around `subprocess.run`, while `ceiling_s` bounds only the
training loop's own elapsed (`train_earlyln_cell`'s internal `t0`, and
`run_earlyln_cell`'s `t0` is set only after `claim_config`/`manual_seed`/
model construction, `:236-249`). Interpreter start, `torch` import, CUDA
context init and model build sit outside every priced term. The design
acknowledges this class of cost but folds it into the *supervisor
margin* (`:832-836`) rather than the tail, while presenting `15.0126h`
as the tight derived bound.

True single-attempt tail ≈ `0.0157 + startup` ≈ **0.016–0.021 GPU-h**,
so the internal worst case is ≈`15.016–15.021h`, not `15.0126h`.

**Impact: nil on the declared numbers** — the disclosed `15.20h` and the
`15.50h` pool ceiling both absorb it with room to spare. The finding is
that a claim R3 introduced as a fresh derivation ("derived by induction
… not asserted") contains a false exclusivity premise.
**Discharge:** replace "Only one of the two can apply" with the sum, and
either price the process-startup term explicitly in the tail or state
that it is deliberately carried by the 0.30h margin.

### KW5.12 — **MINOR (cosmetic).** Rounding nits.

Recomputed from `F(K,d,64)=76Kh²+4dh²+12K²h+4Kdh+4d²h` and the K=24
measured mean 0.4680: FLOP ratios `1.0909/1.1829/1.2762` (design
`1.091/1.183/1.276` ✓); 80K nominals `0.5105/0.5536/0.5973` ✓; 12-cell
total `6.6456` ("≈6.65" ✓). Deltas: ceiling table K=28 `1.1073` vs
computed `1.10724→1.1072`, K=30 `1.1946` vs `1.19452→1.1945`; the 160K
nominals are derived from the *rounded* `0.5106` rather than `0.510532`
(`0.9882` vs `0.9881`), and the conditional ceilings inherit it
(`1.9764/2.1432/2.3121` vs `1.9763/2.1431/2.3120`). All ≤0.0001h and all
in the conservative direction; `2.32` still clears `2.3120` (by 0.0080h,
a 0.35% margin worth knowing). Separately, `1.05105` (`:883`) and
`1.0510` (`:640`, `:883`) are both used as the K32 seed-3 denominator
two lines apart; both round to `1.207`.
**Discharge:** cosmetic; fix or ignore, but pick one denominator digit.

---

## §6 INTEGRITY — **PASS** (one unattributed hunk)

`git diff HEAD~1 HEAD -- matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md`
= 767 insertions / 305 deletions across 9 hunks. Section-level MD5
comparison of the `HEAD~1` and `HEAD` texts:

| Section | Status | Claimed by §R3? |
|---|---|---|
| preamble (mandate) | CHANGED — status header only, `DRAFT-R2 → DRAFT-R3` | yes (status line) |
| §1, §2, §3 | **byte-identical** | matches "not re-litigated" list ✓ |
| §4 | CHANGED | yes, extensively ✓ |
| §5 | CHANGED — **exactly one paragraph**, the `INCOMPLETE-AT-K` section | yes (KW4.8 row) ✓ |
| §6 | CHANGED — rewritten | yes (KW4.3, E6 rows) ✓ |
| §7 | CHANGED — E1/E4-uniformity bullet | **NO — KW5.8** |
| §A1, §R1, §A2, §R2 | **byte-identical** | yes, "UNCHANGED as historical record" ✓ |
| §A3 | body byte-identical; two lines appended (a blank line + `---` separator ahead of the new §R3) | substantively unchanged ✓ (nit only) |

So: **§A1/§A2/§A3/§R1/§R2 are unchanged** as §R3 claims — §A1/§R1/§A2/§R2
to the byte, §A3 to the byte plus a horizontal-rule separator. §R3's
claimed edit list matches the actual hunks **except for §7**. The
`STATE.md` line-renumber content from KW3.7 is unmoved. No frozen
adjudication text was retconned.

---

## §7 THE FLAGGED-OPEN ITEM — the 0.30h supervisor margin

**Ruling: ACCEPTABLE AS-DECLARED. Do not tighten it pre-build. It is not
the thing that needs work.**

§R3 (`:1956-1964`) flags the margin as *"STATED, not derived"* and
invites Round 4 to tighten it once real orchestrator overhead is
measured. Three reasons not to:

1. **It is not a runtime bound, so tightening it buys nothing
   operationally.** Verified directly: `queue_worker.sh` never reads
   `gpu_h_estimate`, applies no timeout, and runs `bash -c "$cmd"` to
   completion (`:157`); `idle_fallback_daemon.sh` promotes by filename
   and reads no cost field. The ONLY place `15.50` bites is the
   `validity_check`'s `realized_gpu_h_final <= 15.50` assertion — a
   post-hoc pass/fail on a number the orchestrator itself computed.
   The real enforcement is the internal `15.00` hard gate, which the
   margin does not touch.
2. **The gap it covers is currently doing real work.** KW5.7 shows the
   tail is ≈0.016–0.021h rather than 0.0126h, and the process-startup
   term is carried *by this margin* — it is not slack, it is the only
   place that cost is priced. Between the true internal bound
   (≈15.02h) and the declared 15.50h there is ≈0.48h, of which the
   design already spends ≈0.18h on conservative rounding to 15.20h. A
   generous, round, disclosed policy number here is the correct
   engineering choice for a 15 GPU-h job on a contended box, and the
   design's own KW4.9 citation ("proportionally more under exactly the
   contention the ceiling exists to survive") justifies it in-text.
3. **The margin cannot fix what actually threatens the ceiling.** A
   0.30h cushion is irrelevant against KW5.1's ≥1.20h-per-crash
   accounting hole. Spending a pre-build round measuring subprocess
   spawn latency, while the ledger can silently lose a full cell's
   spend, would be optimizing the wrong term by two orders of
   magnitude.

**Recommendation:** keep `0.30h` exactly as written and as disclosed;
close KW5.1 instead. If a reviser wants the margin to be *derived*
rather than *stated*, the cheapest honest route is to fold the
micro-smokes (KW5.10) and the measured per-attempt startup (KW5.7) into
the ledger as real charges, at which point the margin covers only
genuine variance and can be re-sized from data at harvest time — a
post-run refinement, not a pre-build gate.

---

## §8 WHAT I COULD NOT BREAK — verified clean this round

Recorded so a future round does not re-spend effort here.

1. **The 125-outcome partition (E5).** Regenerated by execution:
   10 band rows, counts 18/4/12/8/12/8/42/15/4/2, Σ=125/125, exact
   match including the `NON-MONOTONE-UNRESOLVED` split into 4 + 2 and
   the `(2,4,2)` tag correction. Fourth consecutive round. Untouched by
   §R3.
2. **The whole pricing chain.** Closed-form FLOPs, the five-K ratio
   table, 80K nominals, the 12-cell ≈6.65h total, the 1.9355× maximum
   budget ratio choice, all six informational ceilings, the 1.2069×
   archive spike and the 1.657× headroom — every number recomputed
   independently, all correct to the stated digits (modulo KW5.12's
   ≤0.0001h rounding).
3. **KW4.7's decide-rates.** 16/25, 17/25, 25/25, 43/80, 38/80, 36/80 —
   third independent derivation, exact.
4. **The 11-configuration ambiguity table.** All 11 rows exact,
   including candidate triples, bands, `[NON-MONOTONE]` tags and
   tie-break outputs.
5. **The trigger's totality and its exclusion guarantee.** 1000/1000
   state vectors terminate; 0/1000 dispatch at an excluded K; 0/1000
   produce `TRIGGER-UNRESOLVED` while the band decides (the
   `:1310-1313` claim).
6. **§6's pool-eligibility rewrite.** Checked line by line against the
   pool contract text and against both dispatch scripts. The KW4.3
   contradiction is genuinely dissolved, and the design is honest
   about the ceiling being self-enforced.
7. **§2's config family and the additive `GRID_SHAPES`/`GRIDS` build
   note.** Byte-unchanged this revision; independently re-verified that
   `--K` `choices=sorted(GRID_SHAPES)` (`ncr_earlyln_scale.py:857`)
   would reject 26/28/30 today and that §2's build note closes exactly
   that gap, including compatibility with the file's own `d=2K,h=64`
   self-test assertion at `:564`.
8. **The `d=K+1` collision defence in `validity_check`** (`:1026-1031`)
   is well-targeted and cheap; keep it through any revision.

---

## §9 VERDICT

**`REV-REQUIRED`.** 2 FATAL / 3 MAJOR / 8 MINOR.

Forcing: **KW5.1** (ledger loses spend on mid-attempt death; the
declared ceiling is not a bound and has no external backstop),
**KW5.2** (attempt-indexed outdirs vs `harvest()`'s flat glob — the two
F1 fixes are mutually incoherent, and the obvious repair corrupts the
fixed-denominator-4 guard), **KW5.3** (non-zero exit folded into
`ABORTED-BUDGET`, defeating the STOP kill switch), **KW5.4**
(`run_status` undefined; the spec's own `validity_check` rejects the
design's pre-registered degraded outcome), **KW5.5** (a ≤9.28 GPU-h
conditional branch fires in 371/1000 state vectors with no
pre-registration either way).

Not forcing, but all cheap and all in R3-written text: KW5.6–KW5.13.

**Assessment of the round, for the adjudicator.** The delivery-model
change is the right call and it worked: KW4.3 is closed with no residue,
KW4.2's concurrency hole is structurally gone rather than patched, and
F2's trigger rule survived an exhaustive sweep over ten times the
configuration space the design itself checked — total, terminating, with
its exclusion guarantee airtight and its 11-row table exact to the digit.
F3 is buildable as written. F4's coverage is complete and its
MAJOR/FATAL rows hold up.

What round 4 finds is a consistent shape: **§R3's fixes are correct in
isolation and were not checked against each other or against the crash
path.** The wall-clock ledger is exact for attempts that return but not
for the one that does not (KW5.1); the attempt-indexed outdir closes the
overwrite but opens the harvest (KW5.2); the retry state machine
handles budget aborts but swallows the kill switch (KW5.3); the graceful
throttle is declared correct in §4 and marked FAILED by §4's own job
spec (KW5.4); the trigger and the band are correctly independent and
their intersection is unpriced (KW5.5). None of these needs a fifth
delivery model — every one has a discharge condition of a few lines
inside the existing ORCHESTRATOR CONTRACT. A Rev 4 that writes an
in-flight ledger row, names the harvest discovery/precedence rule,
branches on exit code, defines `run_status`, and pre-registers the
`INCOMPLETE-AT-K` spend policy should clear.

*Focused audit round 4, 2026-08-06. Written from direct reads of
`matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md` (1988 lines,
in full), `NCR_KWALL_ATTACK_R3.md` (section headings + every
FATAL/MAJOR/MINOR heading), `matrix-thinking/ncr/ncr_earlyln_scale.py`
(`:75-93`, `:165-266`, `:270-420`, `:545-600`, `:850-912`),
`matrix-thinking/queue/queue_worker.sh` (in full),
`matrix-thinking/queue/idle_fallback_daemon.sh` (`:1-80`),
`matrix-thinking/queue/jobs/pending/108_laneA_main_K48_s0.json`, and
`matrix-thinking/ncr/ncr_task.py` (`:64`, `:126-162`). All numeric
claims in this report come from fresh Python execution — the six-rule
`classify()` over 125 triples; the F2 trigger over 300 singly-,
240 doubly-, and 1000 total state-space configurations; the closed-form
pricing chain — plus `git diff`/`git show` and section-level MD5
comparison for the integrity check. The enumeration scripts are
session-local working files, not committed. No repo file other than
this one was created or modified; no command was run on the box; no job
was launched; no git mutation was made.*
