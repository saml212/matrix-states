# K-wall build audit R1 + pre-launch resource/placement red-team

**Scope:** `matrix-thinking/kwall_build/` at commit `4d9a6b9`, audited
against `NCR_KWALL_CHARACTERIZATION_DESIGN.md` (STATUS: RELEASED, commit
`1c99cc5`) §4 ORCHESTRATOR CONTRACT / §5 / §6, the §8 build charter
(incl. item (j)), and `BUILD_REPORT.md` §6/§7. Independent pass — the
implementer did not review this work. Combined 10–50 GPU-h-tier gate
(build audit + resource/placement red-team).

**VERDICT: FAIL** — 1 FATAL, 4 MAJOR, 8 minor.
Do NOT deploy, do NOT run the box-side micro-smokes, do NOT promote the
job spec until the FATAL and MAJORs are adjudicated. The FATAL is a
launch-losing defect of exactly the KW8.2 class the design's own
gauntlet already killed once: a legitimate, pre-registered outcome
emits a report that its OWN `validity_check` rejects, so
`queue_worker.sh` routes the run to `failed/` **after** the full
≤15.50 GPU-h is spent.

All findings below are executed, not asserted. Harnesses:
`/private/tmp/claude-501/-Users-samuellarson-Experiments-learned-representations/be705417-f189-4cd8-8024-24cf6a0130a0/scratchpad/`
(`h1_band_disclosure.py`, `h2_trigger_diag.py`, `h3_full_grid.py`,
`h4_gates_and_diag.py`, `h5_adversarial_disk.py`, `stub2.py` — a
per-cell-controllable CPU stub written for this audit).

---

## 0. What was re-run from a clean checkout (all three suites + the 33-payload identity)

| Suite | Result |
|---|---|
| `tests/test_reconstruction.py` | **PASS 4/4** — 24-state totality; OLD guard 30/200 orphans + 6/200 abort-trips; NEW guard 0/200 + 0/200 (design's own KW8.3 / §R7 J3 figures reproduced exactly) |
| `tests/test_validity_check_suite.py` | **PASS 33/33**, 0 divergences vs the unmodified `kwall_suites/r10_vcheck.py` in both NEW and OLD modes |
| `tests/test_orchestrator_integration.py` | **PASS 10/10** (44 individual checks), including the real-SIGKILL mid-attempt recovery test |

Two claims verified more strictly than the build report states them:

- **"Byte-for-byte port" of `validity_check` — CONFIRMED by diff, not
  only by behaviour.** `inspect.getsource` diff of
  `r10_vcheck.validity_check` (132 lines) vs
  `kwall_lib.validity_check_core` (135 lines) yields exactly ONE hunk:
  the signature line and an added docstring. Every module constant is
  equal (`BAND_LABELS`, `RESOLUTIONS`, `ACCEPT`, `EPS=Fraction(1,10**6)`,
  `PRIMARY_CEIL=Fraction(6,5)`, `COND_CEIL=Fraction(58,25)` — all
  `port_equal=True`). `kwall_suites/` is committed and clean
  (`git status --short kwall_suites/` empty).
- **The in-build GATE-REFUSED fix's negative test HAS TEETH** (charter
  requirement, verified by forced-fail): monkey-reverting
  `derive_cell_state` to the literal `attempt_n==2`-only reading and
  re-running the same scenario yields
  `rows=[(1,'GATE-REFUSED'),(2,'GATE-REFUSED')]`, so
  `test_hard_gate_refused`'s "exactly one row" assertion FAILS. The fix
  is load-bearing and correct per the design's derived-cell-state rule.

---

## FATAL

### F1 — `INCOMPLETE-AT-K` under-discloses its affected K's; the emitted report FAILS its own `validity_check` in 54 of 125 reachable outcome shapes

`kwall_lib/classify.py:102-104`:

```python
if any(s == "UNRESOLVED" for s in states.values()):
    incomplete = sorted(int(K) for K, s in states.items() if s == "UNRESOLVED")
    return None, None, "INCOMPLETE-AT-K", incomplete
```

The early return lists **only** the `UNRESOLVED` K's (`n_completed ≤ 2`).
A K sitting at `n_completed == 3` (`AMBIGUOUS`) in the same run is
dropped: it appears in neither `band.incomplete_at_K` nor
`band.interval_resolved_Ks` (the latter is forced to `[]` on this path
by `orchestrator.py:354-358`).

`validity_check`'s `COMPLETE`/OTHERWISE branch then requires *"for every
K named in NEITHER field, that K's canonical-file count is exactly 4"*
(design `:2467-2471`; port at `validity_check.py:145-153`). The dropped
K reads 3. The report fails its own check.

**Executed, end-to-end through the real `orchestrator.run()`**
(`h3_full_grid.py`, real subprocess dispatch, real
`harvest_bridge`→repo `harvest()`, real report emission, real
`python3 -m kwall_lib.validity_check` CLI):

```
--- C_mixed_unres_amb   (K=30 s2,s3 crash twice; K=28 s3 crashes twice)
    run_status=COMPLETE canonical=9
    per_K=26:(4,4), 28:(3,3), 30:(2,2)
    band=INCOMPLETE-AT-K interval_resolved=[] incomplete_at_K=[30]     <-- 28 dropped
    VALIDITY_CHECK: FAIL ['COMPLETE/otherwise: K=28 count 3 != 4']
    CLI exit=1  ->  queue_worker.sh routes to failed/
```

Control cases from the same harness all PASS: `A` (12/12 →
`FRONTIER-AT-K*=30`), `B` (K=30 only → `incomplete_at_K=[30]`), `D`
(K=28 only → `interval_resolved_Ks=[28]`).

**Blast radius (`h1_band_disclosure.py`, full enumeration of every
reachable `(n_completed, n_converged)` triple, 3375 states):**

- **1440 / 3375 states**, spanning **54 / 125** `n_completed` shapes,
  emit a report that fails its own `validity_check` on this clause.
- The failing class is exactly `{≥1 K at n_completed==3} ∧ {≥1 K at
  n_completed≤2}` — 54 shapes, matching the enumeration to the count.
- Only one other state fails anywhere in the space: `(0,0,0)` via the
  §R8 K2 `>=1 COMPLETED primary pair` clause. That one is
  **design-inherited**, not a build defect (K2 is unconditional in the
  design's own text), and is noted as observation O1 below.

**Reachability is not exotic.** The class needs crash-caused (not
budget-caused) incompleteness — precisely the cheap failure mode: a
deterministic crash exits in seconds, so `realized_gpu_h` stays low, no
gate ever refuses, `determine_run_status` returns `COMPLETE`, and the
per-K clause fires. A systematic failure at one K (all 4 seeds) plus one
flaky seed at another K lands here directly.

**Fix direction (coordinator adjudicates; not applied by this audit):**
the early return must disclose the union of incomplete K's —
`UNRESOLVED ∪ AMBIGUOUS` — as `incomplete_at_K`. Re-checked against the
same enumeration, that union satisfies the OTHERWISE branch in every one
of the 54 shapes (named K's read `<4`; unnamed read exactly `4`), and
leaves cases B/D byte-identical. It is also what the design's own D5/E4
text says ("`INCOMPLETE-AT-K` for the affected K's, both/all candidate
bands disclosed").

---

## MAJOR

### M1 — the trigger `diag` return-site disambiguation is wrong in 115/1000 reachable state vectors (BUILD_REPORT §7 item 5)

`kwall_lib/classify.py:156-169` blocks the K-scan on **any** `UNRESOLVED`
K. The design blocks only when the scan must **read** that K
(`:558-579`: `kt = smallest_K_with_rate_below_3(triple)`; the early
return fires only `if kt requires reading an UNRESOLVED K's status`). If
a smaller K is already non-ROBUST, the scan decides without ever
touching the unresolved K, and G5's band precondition then returns
`band_blocked_K_trig = result[0]`.

I wrote an independent reference implementation of the design's
pseudocode and validated it against the design's own disclosed figures
before comparing: pre-G5 K-scan alone → **DECIDED=844 /
TRIGGER-UNRESOLVED=156** (design `:676`), post-G5 → **473 / 527**
(design `:682-683`). Both MATCH.

Per-vector comparison over the full 1000-vector space
(`h2_trigger_diag.py`):

```
vectors where (K_trig, resolution) DIFFER build-vs-design:   0 / 1000
vectors where (blocking_K, band_blocked_K_trig) DIFFERS:   115 / 1000
   states=(0, 0, 'UNRESOLVED')  design=(None, 26)  build=(30, None)
```

`115 = 844 − 729` — exactly the vectors carrying an `UNRESOLVED` K whose
K-scan nevertheless decided. Confirmed end-to-end through the real
orchestrator (`h4_gates_and_diag.py` case E: `r26=0/4`, K=30
`UNRESOLVED` → report carries `blocking_K=30, band_blocked_K_trig=null`;
the design requires `blocking_K=null, band_blocked_K_trig=26`).

No dispatch decision changes (resolution is identical in all 1000
vectors, and the build's import-time 473/527 assertion still passes —
which is why this survived the build's own self-check). The damage is
to the disclosure G5 exists to make: the K-scan's own candidate
`K_trig` is silently dropped in exactly the cases the design says must
never drop it, and an unrelated `blocking_K` is reported in its place.
Neither field is asserted by `validity_check`, so this is MAJOR, not
FATAL.

### M2 — job-spec `cmd` uses `$(git rev-parse HEAD)`; if `/home/nvidia/ncr` is not a git working tree the orchestrator dies at argparse before dispatching anything

`job_specs/kwall_char_orchestrator.json` ends with
`--gpu-id ${CUDA_VISIBLE_DEVICES:-0} --git-commit $(git rev-parse HEAD)`.
`queue_worker.sh:157` runs the spec's `cmd` through
`( CUDA_VISIBLE_DEVICES="$GPU" bash -c "$cmd" )`, so both expansions are
live. Evidence that the git substitution is unsafe here:

- **0 of 366** existing `queue/jobs/pending/*.json` specs use any `$(…)`
  command substitution — this spec is the only one.
- The box's `~/ncr` is populated by `deploy.sh`'s staged `scp` +
  md5-verify path; neither `deploy.sh` nor `H100_SETUP.md` contains a
  `git clone`/`git init` anywhere.
- `run_ncr.git_commit()` (`:93-98`) wraps the identical command in
  `try/except → "UNKNOWN"` — the repo's own code already defends against
  this exact failure at this exact path.

Executed failure mode (non-git cwd, same expansion shape):

```
fatal: not a git repository (or any of the parent directories): .git
usage: toy.py [-h] [--gpu-id GPU_ID] [--git-commit GIT_COMMIT]
toy.py: error: argument --git-commit: expected one argument
```

The empty substitution leaves `--git-commit` with no value, argparse
exits 2, nothing is dispatched, no report is written, the job routes to
`failed/` and the claimed GPU goes idle. GPU-h cost ≈ 0, but the wave
does not run. Fix is one token:
`--git-commit "$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"`, or
verify box-side that `~/ncr` is a working tree. `${CUDA_VISIBLE_DEVICES:-0}`
itself is fine (`queue_worker.sh` sets a single index; `nvidia-smi -i`
takes the physical index, which is what `gpu_reap_check` needs).

### M3 — G4's mandated `STOPPED-BY-OPERATOR` pre-write self-check is absent, and the field it protects can be affirmatively wrong

Design `:1754-1763` relocates the stop-file evidence check *out* of
`validity_check` and makes it **the orchestrator's own pre-write
self-check**: *"`report["stop_file_path"] is not None and
os.path.exists(report["stop_file_path"])` … asserted before
`orchestrator_report.json` is ever written with this label, where it can
actually fire and prevent a false report from existing at all."*

`orchestrator.py:454-464` and `:479-485` write the
`STOPPED-BY-OPERATOR` report with no such assertion anywhere in the
file (`grep -n "stop_file_path" orchestrator.py` → only the
`build_report` field assignment, `:378`). Two compounding defects on the
same path:

1. On a **conditional-arm** stop, `build_report` still sets
   `stop_file_path = state.args.stop_file` — the PRIMARY sentinel path,
   not the conditional one that was actually seen.
2. On a **primary** stop, `resolution` is passed as the fabricated
   literal `{26:(0,0), 28:(0,0), 30:(0,0)}` (`:460`), so
   `primary.per_K` claims zero completions for all three K's regardless
   of how many cells actually completed before the stop.

`STOPPED-BY-OPERATOR` is outside the accept-set, so nothing is
mis-routed — but the design put a named enforcement point here
precisely so a false artifact never exists, and the build silently
omits it while persisting two wrong fields.

### M4 — three schema fields the design defines (and one it *requires*) are never populated

| Field | Design status | Build |
|---|---|---|
| `trigger.candidate_set` | schema `:1563`; the design's own tie-break verification payload sets `[26,28]` (`:2644`) | hardcoded `None` (`orchestrator.py:385`) even on `tie-break-min` |
| `band.candidate_bands` | schema `:1570`; D5/E4 `:2250` — *"otherwise `INCOMPLETE-AT-K` for the affected K's, **both/all candidate bands disclosed**"* | hardcoded `None` on both branches (`:358`, `:362`) |
| `conditional.per_seed` | schema `:1565`; §5 `:3104-3107` — a throttled arm's completed cells *"are disclosed as DATA ONLY (their raw per-seed `gate1`/`indist_min` records…)"* | hardcoded `[]` (`:388`) |

`band.candidate_bands` and `conditional.per_seed` are not optional
niceties: they are the disclosure the design substitutes for the
suppressed band in exactly the two outcomes (`INCOMPLETE-AT-K`, a
throttled 160K arm) where the headline is withheld. Emitting the
suppression without the disclosure loses the only reportable content
those branches were designed to carry. None is asserted by
`validity_check`, hence MAJOR not FATAL.

---

## minor

- **m1 (item (j)).** `tests/test_orchestrator_integration.py:329-333`
  pads the ledger with 12 synthetic rows at `(K=30, seed=8+i)`,
  i.e. seeds 8–19. The `elapsed_h` values are producible (exactly
  `1.20 = PRIMARY_CEILING`, `ceiling_charged=true` — the Class-1 rule,
  correctly fixed per BUILD_REPORT §6), but the `(K,seed)` identifiers
  are not: `SEEDS=(0,1,2,3)`, so no such pair can exist. The same
  ledger shape is producible with real pairs (11 pairs × [attempt-1
  `CRASHED-RECOVERED` 1.20 + attempt-2 `COMPLETED`] + 1 pair ×
  [`CRASHED-RECOVERED` 1.20 + `GATE-REFUSED` 0.0] = 12 pairs, 12
  ceiling-charged rows, 14.40). Item (j) says fixtures must be
  producible; this one is producible in its numbers and not in its
  keys.
- **m2.** `reconstruction.reconstruct_attempt_row` crashes with
  `TypeError: unsupported operand type(s) for /: 'NoneType' and 'float'`
  on a parseable `COMPLETED` cell JSON that lacks the top-level
  `elapsed_s` (executed, `h5_adversarial_disk.py`). §R7 KW8.9 makes a
  missing/invalid `status` behave as UNPARSEABLE but says nothing about
  `elapsed_s`. Unreachable from `ncr_earlyln_scale.py`'s own writer
  (`:275`/`:314` always set it), reachable from a foreign/hand-edited
  file — and it lands in the RECOVERY path, i.e. inside a
  supervisor restart loop.
- **m3.** `ledger.reconstruct_ledger_from_disk` reconstructs the
  conditional 4 only when `_infer_conditional_K()` finds a
  `K{K}_s{seed}_attempt{n}/` directory. The design (`:1047-1054`) says
  the conditional 4 reconstruct **UNCONDITIONALLY**. Gating on archival
  attempt dirs (rather than on the conditional CANONICAL directory) is
  the right *spirit* of KW7.1(b), but 0.2's own named residual case —
  "an attempt tree pruned after copying" — then leaves surviving
  conditional canonical evidence uncharged. Inferring K from the
  canonical filenames as a fallback closes it.
- **m4.** `BUILD_REPORT.md` §5 item 2 and the job spec's `notes`
  precondition (2) both say *"see BUILD_REPORT.md for the exact deploy
  commands"* — BUILD_REPORT.md contains no deploy commands (no `scp`,
  no `rsync`, no `ssh` anywhere in the file). Dangling self-reference on
  the one step that must not be improvised.
- **m5.** BUILD_REPORT §4 says the pre-fix bug "would have caused
  `dispatch_cell`'s loop to attempt a SECOND dispatch after a refused
  FIRST attempt." Executed: with the fix reverted, the second attempt is
  itself GATE-REFUSED (the ledger is unchanged, so the same gate refuses
  again) — two `elapsed_h=0.0` rows, no subprocess, no GPU-h. The fix is
  correct and necessary for contract fidelity; the consequence was
  cosmetic, not a real second dispatch.
- **m6.** `smokes/MICRO_SMOKE_SPEC.md` adds `--stop-file
  .../K${K}/STOP` to the design's own smoke command (design `:2862-2864`
  has no `--stop-file`). Harmless. Separately: `--ceiling-gpuh 0.05`
  bounds only the training loop (`train_earlyln_cell:209`), never the
  post-train instrument sequence, so `≤0.15 GPU-h` is a training-only
  bound — the design's own D5 discloses the mechanism, the smoke spec
  restates the figure without the caveat.
- **m7.** When `run()` refuses on the startup smoke gate (returns 3) or
  aborts loudly on G2, no report is written and the spec's
  `validity_check` dies with an uncaught `FileNotFoundError`. Routing is
  still correct (non-zero → `failed/`) but the job log gets a traceback
  instead of a diagnosis.
- **m8.** Design §6 red-team item (v) — the on-box
  `~/queue/{fallback_pool,claimed}` sweep for K∈{26,28,30} content
  (KW2.7) — is **still UNDISCHARGED**; it needs box access this audit
  does not take. The repo half is clean: **0 of 366**
  `queue/jobs/pending/*.json` specs reference `--K 26/28/30`.

---

## Observations (design-inherited, NOT build defects — recorded so they are not re-found later)

- **O1.** A run where every primary cell crashes twice (0 canonical,
  no budget refusal) is `COMPLETE` by G4's own definition and is then
  rejected by §R8 K2's unconditional `>=1 COMPLETED primary pair`
  clause → `failed/`. Claiming it `COMPLETE-DEGRADED` fails the mirror
  clause. This is the design's own text, faithfully implemented; it is
  the only other report class in the whole reachable space that fails
  its own check (1 state of 3375).
- **O2.** `COMPLETE-DEGRADED`'s throttle-evidence clause tests
  `a["status"] in ("GATE-REFUSED","PERSISTENTLY-ABORTED")`, but
  `PERSISTENTLY-ABORTED` is provably never an `attempts[].status`
  (§R5 KW6.5(ii)) — that half is dead. It is in the R11-audited
  reference verbatim; the build's byte-for-byte port inherits it, and
  the live half (`GATE-REFUSED`) is exactly what
  `determine_run_status` keys on, so the two stay consistent.
- **O3.** The crash-window table's row 3 ("between copy and fold ⇒ full
  ceiling charged") is unreachable as written, because §R6 I2's step 2.1
  runs FIRST and charges measured + `s` whenever the attempt JSON is
  also readable-`COMPLETED`. The build follows the *procedure* text
  (2.1 precedence), which is correct. The `T ≤ 15.3737` bound survives:
  moving rows from Class 1 (≤12 rows × τ) to Class 2 (≤32 rows × s) can
  only reduce the Class-1 count, and the ledger bound `R_N` is
  unchanged.

---

## Contract-fidelity checks that PASSED (recorded with their evidence)

**Cell order / concurrency.** K-major `26→28→30`, seeds `0,1,2,3`,
conditional at `K_trig` after (`orchestrator.py:449-478`) — matches
design `:833-839`. Strictly serial: `grep` for
`thread|Thread|multiprocessing|Pool|Popen` across `orchestrator.py` +
`kwall_lib/*.py` returns **nothing**; the only dispatch primitive is a
blocking `subprocess.run`. §6 red-team item (iii) discharged by
inspection.

**Gate arithmetic, boundary-probed** (`h4` case G):

```
hard_gate_ok(13.80,      1.20) = True     (13.80+1.20 = 15.00 <= 15.00)
hard_gate_ok(13.8000001, 1.20) = False
hard_gate_ok(12.68,      2.32) = True     (= 15.00 exactly)
hard_gate_ok(12.681,     2.32) = False
retry_gate_ok(11.999999) = True ; retry_gate_ok(12.0) = False ; (12.000001) = False
```

Both gates are evaluated **before** dispatch and before the write-ahead
(`dispatch_attempt:84-100`, ahead of `:104-107`); the retry gate is
subordinate to the hard gate and applies only to `attempt_n==2`. No path
dispatches past 15.00 or re-dispatches past 12.00. Conditional attempts
charge 2.32, primary 1.20 — charged == enforced (KW4.4).

**Write-ahead ordering and the crash-window walk.** `open_attempt`
persisted atomically before `subprocess.run` (`:104-107`); classify →
copy → fold strictly in that order (`:119-162`), so a `COMPLETED` row
can never precede its canonical file. All four windows walked against
the code: before-copy and mid-copy → 2.1 promotes and charges measured
+ `s`, `ceiling_charged=false`; between-copy-and-fold → 2.1's
skip-the-copy branch; after-fold → no action. 2.2's YES branch (full
ceiling, `ceiling_charged=true`) is reached only when the attempt JSON
is absent/unparseable/non-`COMPLETED` while the canonical is
`COMPLETED`. `atomic_write_json` is byte-identical to
`run_ncr.py:105-109`. The real-SIGKILL test produces exactly one
`CRASHED-RECOVERED` row at exactly `1.20`, `ceiling_charged=true`, no
gap, no double-charge.

**`determine_run_status`'s row-coverage invariant (BUILD_REPORT §7 item
2) — reliance is sound, and the failure mode on a violated invariant is
loud, not silent.** `dispatch_cell` returns only after appending ≥1 row
or on an already-terminal derived state (which implies rows exist); its
`next_n > 2` defensive branch is unreachable with a `NON-TERMINAL`
derivation (a `NON-TERMINAL` cell provably has no `attempt_n==2` row, so
`max+1 ≤ 2`); an operator stop short-circuits before the function is
reached. Adversarial disk states (`h5`):

```
foreign_ledger_no_rows (realized=14.9, attempts=[])
    -> run_status=EXHAUSTED-BUDGET   VALIDITY_CHECK: FAIL ['U3: realized 149/10 != sum 0']
dirty_dir_empty_ledger (12 canonical, empty ledger)
    -> ABORTED LOUDLY: RuntimeError G2 INVARIANT VIOLATION; report written? False
ledger_missing_attempts_key           -> ABORTED LOUDLY: KeyError 'attempts'
no_ledger_canonical_only              -> COMPLETE, realized 0.0669, VALIDITY_CHECK PASS
truncated_ledger_canonical_only       -> COMPLETE, realized 0.0669, VALIDITY_CHECK PASS
```

A violated invariant is caught downstream by U3 (bookkeeping identity)
or aborts with an exception — never a silent misclassification.
Reconstruction from a missing/truncated ledger resumes correctly and
never resets to 0 (red-team item (x) reconfirmed independently).

**`harvest_bridge`'s no-patch claim (BUILD_REPORT §7 item 3) — verified
by reasoning against the code, not only by the one passing test.**
`discover_seeds_by_K` (`ncr_earlyln_scale.py:362-382`) globs
`earlyln_K{K}_s*.json` non-recursively in the canonical directory and
returns `n_seeds = len(seeds)`; G2 writes a canonical file ONLY on
`COMPLETED` acceptance and §R5 H2 gives the converse, so file-presence
== `n_completed` **by the pair**. The bridge reads only `n_seeds` and
`n_converged` — never `rate`/`gate_eligible`/`gate1_label` (§R5 KW6.15's
exact scope). One real trap correctly handled: `harvest()` keys
`per_K` by **`str(K)`** (`:448`), and the bridge tries `get(K) or
get(str(K))`. Attempt dirs are subdirectories (never matched by a flat
glob), `.axis_c_lock.json` siblings are written into attempt dirs and
are skipped by both globs, `.CORRUPT-<ts>` and `.tmp` files fail the
`.json` suffix, and `ORCHESTRATOR_LEDGER.json` / `orchestrator_report.json`
fail the `earlyln_K` prefix.

**G2 exists-check + full-grid disk glue (BUILD_REPORT §7 item 4).**
Exercised against files `run()` itself produced on the FULL 12-cell grid
for the first time (`h3`): 12/12 canonical, `_glob_canonical` +
`build_disk_view` + the real
`python3 -m kwall_lib.validity_check <report> <primary> <cond>` CLI →
`exit=0 VALIDITY_CHECK PASS`. The `K_trig==32` `$0` branch fires in the
all-converged case and correctly satisfies U7 clause (b)
(`launched=False`, `K_trig=32`, `qualifier_band=CONFIRMED-WALL-AT-160K`).
A budget-exhausted mid-run case with real HARD-GATE refusals
(`h4` case F, pre-seeded from a producible ledger:
`9×1.2210 + 3×1.20 = 14.5890`) emits `run_status=EXHAUSTED-BUDGET`,
`frac=0.2468 ≤ 0.50` → **VALIDITY_CHECK PASS**, exercising the §R7 J4
dichotomy on the correct side.

**Item (j) spot-checks (≥6, arithmetic re-derived).**
(1) `PRIMARY_COMPLETED_REACHABILITY_CAP = 1.20 + 0.0157 + 0.0053 =
1.2210` — asserted at import.
(2) `TRUE_SPEND_WORST_CASE = 15.0157 + 12(0.0157) + 32(0.0053) =
15.3737 < 15.50` — asserted at import.
(3) reconstruction `COMPLETED` fixture `elapsed_s=3600 → 1.0 + s =
1.0053 ≤ 1.2210` ✓.
(4) reconstruction `CRASHED-RECOVERED` fixtures at exactly `1.20`
(primary) / `2.32` (conditional) — the Class-1 rule ✓.
(5) suite L6 (option A): `12 × 1.20 = 14.40`, `ccgh=14.40`,
`frac=14.40/14.40=1.0 > 0.50` ✓ (printed by the suite run).
(6) suite A6: `9×1.20 + 2.40 + 1.00 + 0.0 = 14.20`, `ccgh=13.20`,
`frac=13.20/14.20=0.9295774…` (exact quotient, not the 4-dp literal —
the suite prints `|0.8571 − 12/14| = 4.29e-5 > 1e-6`, confirming m4's
rounding discipline has teeth) ✓.
(7) B2′: `2.84/14.20 = 0.20` exactly ✓.
Only m1's `(K,seed)` keys fail producibility; every `elapsed_h` in every
fixture I checked is reachable under the charging rules.

**Recipe patches (task item 4) — additive-only claim CONFIRMED.**
`git show 4d9a6b9 -- matrix-thinking/ncr/{ncr_earlyln_scale.py,ncr_task.py}`:
`11 +`/`12 +`, **zero deletions, zero modified lines**. Values match the
design's §2 Build note verbatim: `GRID_SHAPES[26]=dict(d=52,h=64)`,
`[28]=dict(d=56,h=64)`, `[30]=dict(d=60,h=64)` (design `:187-188`) and
`for _K_new in (26,28,30): GRIDS[_K_new] = _gen_grid(_K_new)` (design
`:189-190`). No existing K=24 behaviour can change:
`_gen_grid` is untouched; the `:150-152` regression assert
(K∈{14,15,16,24}) and the `_K, _g in GRIDS.items()` residue assert both
still hold for the new keys (`(m·K−3) % K = K−3` ✓); `t4b`'s
`GRID_SHAPES[_K] == dict(d=2K, h=64)` convention holds
(`52=2·26, 56=2·28, 60=2·30`) though it does not iterate the new keys;
`--K choices` only widens; the only cross-campaign effect is 3 extra
`SUB4-DISCLOSED-ONLY(n=0)` rows in every `--harvest` report, which are
`gate_eligible=False` and therefore excluded from `scored_Ks` and
`pooled_verdict_K_gt_14` — exactly the artifact §5's KW2.9 paragraph
pre-discloses. `_self_test()` t5 and `--smoke` now build 3 extra micro
cells (slower, not failing).

---

## Resource / placement red-team (task item 5)

| Item | Finding |
|---|---|
| GPU count / exclusivity | The orchestrator occupies exactly ONE GPU. `queue_worker.sh:157` runs `cmd` synchronously as its own child under `CUDA_VISIBLE_DEVICES="$GPU"`, which every dispatched subprocess inherits. No concurrency primitives exist in the build (grep, above). The other 7 GPUs stay claimable. **PASS.** |
| Memory | The design pins no VRAM figure; the spec's `notes` claim is sound by construction — `n_params ≈ 174K` at `d=K+1∈{27,29,31}`, `h=64`, `TRAIN_BATCH=256`. Nowhere near the 80 GB card, and well under `queue_worker.sh`'s 2048 MiB free-GPU threshold *between* cells. **PASS.** |
| Worker claim vs `gh_*` vLLM engines | The free-GPU gate is per-GPU (`nvidia-smi … -i "$GPU"`), so engines on other cards neither block this job nor are blocked by it; an engine on the SAME card prevents the claim outright. **PASS.** |
| PAUSE / STOP / HOLD | `~/queue/PAUSE` stops new claims (a live orchestrator continues); `~/queue/STOP` makes the worker exit after the current job. Both are distinct paths from the orchestrator's own `results_kwall_characterization/STOP` — no collision. **Operational note:** touching `~/queue/STOP` does NOT stop a running orchestrator; the ≤15.5 h job runs to completion. The orchestrator's own kill switch is its results-tree `STOP`. |
| Coordinator-death / requeue safety | Killing the worker's tmux session kills the orchestrator (intended preemption). On worker restart, `queue_worker.sh`'s `.g<GPU>.json` reclaim returns the spec to `pending/`; the re-dispatched orchestrator recovers via write-ahead + reconstruction. Verified locally by the SIGKILL test and by the missing/truncated-ledger cases. **PASS.** |
| Pool contract (≤15.50 vs queue accounting) | `gpu_h_estimate` is read by neither `queue_worker.sh` nor `idle_fallback_daemon.sh`; the 15.50 ceiling is enforced solely inside the orchestrator (hard gate) and re-asserted by U2 — exactly as §6 states. One flat, independent spec; `idle_fallback_daemon.sh` promotes it in filename order with no intra-wave dependency. **PASS.** |
| Box paths | `/home/nvidia/ncr`, `/home/nvidia/tdenv/bin/python3` match `H100_SETUP.md` and job-108 verbatim. `python3 -m kwall_lib.validity_check` requires `kwall_lib/` to sit in the `cd`-target `/home/nvidia/ncr` — consistent with the deploy step, and exercised locally (`exit=0 VALIDITY_CHECK PASS`). |
| Job-spec `cmd` shell semantics | **See M2** — the `$(git rev-parse HEAD)` substitution is the single launch-losing risk in the spec. |
| Micro-smoke commands | `--outdir .../results_kwall_smoke/K{K}` writes `earlyln_K{K}_s0.json` (`cell_id` has no `d` suffix — verified at `ncr_earlyln_scale.py:217-218,250`), exactly the path `check_smoke` reads. `check_smoke`'s criterion (`status ∈ {COMPLETED, ABORTED-BUDGET}` ∧ `K`,`d`,`d_override` = `K`,`K+1`,`K+1`) is the design's §4 F3 criterion verbatim, and `rec["d"]`/`rec["d_override"]` are set at `:261` before either write path. The commands will run **only after** the two recipe patches reach the box (`--K` `choices=sorted(GRID_SHAPES)` rejects 26/28/30 otherwise; `nt.GRIDS[K]` raises `KeyError` inside `_cell_gate2`). See m6 for the two nits. |
| Archive policy | The build writes nothing to `experiment-runs/`; all artifacts are ≤27 KB source files under `matrix-thinking/kwall_build/`. Results land box-side. No ≤25 MB / SSD split issue at this stage. **PASS.** |
| Outstanding | §6 item (v) on-box `fallback_pool`/`claimed` sweep (m8) — must be done before promotion. |

---

## Deploy sequence I verified (blocked until the FATAL + MAJORs are adjudicated)

Recorded so the next round can execute it directly once cleared; every
step below was checked against the scripts/paths named, but **none of it
may run under this verdict**:

1. Fix F1 (`classify_with_interval_logic`'s UNRESOLVED early return) and
   M1 (`trigger`'s scan-reached blocking condition); adjudicate M2–M4.
2. Re-run all three suites + `h1`/`h2`/`h3` from a clean checkout;
   `h1` must report **0** reports failing their own `validity_check`
   outside O1's single `(0,0,0)` state, and `h2` must report **0/1000**
   diag divergences.
3. `bash matrix-thinking/queue/deploy.sh` (or its staged
   `scp`+md5-verify path for `ncr_task.py`/`ncr_earlyln_scale.py` only)
   to land the two additive recipe patches on the box; verify md5
   equality box-side before publish, as `deploy.sh` already does.
4. `scp orchestrator.py` and the `kwall_lib/` package to
   `/home/nvidia/ncr/`; confirm `kwall_lib/__init__.py` is present so
   `-m kwall_lib.validity_check` resolves.
5. Sweep `~/queue/fallback_pool/` and `~/queue/claimed/` for
   K∈{26,28,30} content (§6 item (v), m8).
6. On a GPU verified idle by `nvidia-smi`, run the 3 micro-smokes from
   `smokes/MICRO_SMOKE_SPEC.md`, then the spec's own verification
   snippet (`orch.check_smoke(...)` must read `PASS` for all three).
7. Only then place the (M2-corrected) job spec in
   `~/queue/fallback_pool/`.

---

**Counts: 1 FATAL, 4 MAJOR, 8 minor, 3 design-inherited observations.**
Report path: `matrix-thinking/kwall_build/BUILD_AUDIT_R1.md`.
This audit modified nothing else — no build files, no `STATE.md`, no
`EXPERIMENT_LOG.md`, no box contact.
