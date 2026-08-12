# K-wall wall-characterization — BUILD REPORT

Built against `NCR_KWALL_CHARACTERIZATION_DESIGN.md` (STATUS: RELEASED,
commit `1c99cc5`, §A11-ADJUDICATION) per the §8 build charter restated in
`NCR_KWALL_ATTACK_R8.md` §8, extended through R9's (h)/(i) and R10's item
(j) (`NCR_KWALL_ATTACK_R10.md` §8). This build does not launch anything
and did not touch the box. Everything below lives in
`matrix-thinking/kwall_build/` plus two additive patches to
`matrix-thinking/ncr/{ncr_earlyln_scale.py,ncr_task.py}` (both disclosed
in full below — these are the ONE build-stage code edit the design's own
§2 "Build note" names as required and pre-authorizes).

## 1. What was built

| File | Design anchor | Purpose |
|---|---|---|
| `orchestrator.py` | §4 ORCHESTRATOR CONTRACT | The ONE pool artifact — cell order, dispatch/retry state machine, HARD/RETRY gates, write-ahead ledger + crash recovery, G2 copy-then-fold, trigger+band evaluation, `orchestrator_report.json` emission |
| `kwall_lib/constants.py` | §4 pricing/ceilings | Every ceiling/τ/s/threshold number, each derived in a comment (item (j)) |
| `kwall_lib/classify.py` | §5 six-rule `classify()`; §4 F2/G5 trigger | Band classification, interval logic, the conditional-arm trigger |
| `kwall_lib/reconstruction.py` | §4 recovery steps 0.0/0.1/0.2 | Pure-function 24-state reconstruction + derived-cell-state rule |
| `kwall_lib/disk_io.py` | §4 (G1 atomic writes, G2 canonical contract) | Real-filesystem I/O: atomic JSON writes, attempt/canonical readers, quarantine, promote |
| `kwall_lib/ledger.py` | §4 ledger schema + reconstruction driver | Ledger read/save, `reconstruct_ledger_from_disk` (drives `reconstruction.py` against real disk via `disk_io.py`) |
| `kwall_lib/harvest_bridge.py` | §4 "Enforcement point, named" | Calls the repo's own `ncr_earlyln_scale.harvest()`/`discover_seeds_by_K` — no `harvest()` patch, as the design specifies |
| `kwall_lib/validity_check.py` | §4 job-spec `validity_check` | Byte-for-byte port of the audited `kwall_suites/r10_vcheck.py` core + real disk-reading glue + CLI entry point |
| `cell_runner_stub.py` | — (build-only) | CPU-STUB emulating `ncr_earlyln_scale.py --cell`'s exit-code/JSON contract, for integration tests only |
| `tests/test_validity_check_suite.py` | charter: "pass the committed 24-payload suite VERBATIM" | Cross-checks `validity_check_core` against the unmodified `kwall_suites/` scripts |
| `tests/test_reconstruction.py` | charter R7(b): "24-state and 200-composition walks... unit tests of the real reconstruction function" | Re-derives the design's own 24-state totality + 200-state OLD/NEW-guard figures against the REAL `reconstruction.py` |
| `tests/test_orchestrator_integration.py` | §6 red-team items (vi)/(x)/(xii) + G1-G4 mechanics | Real-subprocess integration tests of `orchestrator.py`, including a genuine SIGKILL mid-attempt-crash-recovery test |
| `smokes/MICRO_SMOKE_SPEC.md` | §4 F3 micro-smoke gate | Exact box commands + pass criterion for the 3 CUDA-bound 500-step smokes |
| `job_specs/kwall_char_orchestrator.json` | §4 job-spec template, job-108 format | The ONE orchestrator pool spec (see §3 below for why one, not sixteen) |

## 2. Test results — everything below was actually RUN, not merely written

All commands run from `matrix-thinking/kwall_build/` with
`DRY_RUN_BYPASS=1` (this repo's own `pre-train-gate` hook otherwise
treats any `python3 *.py` invocation as a candidate training launch;
none of the below is training).

### 2a. `kwall_lib.classify` — import-time regression (`python3 -c "import kwall_lib.classify"`)
- Re-executes the design's own 125-outcome `classify()` partition table
  by construction: **matches exactly** — `FRONTIER-AT-K*=24: 18`,
  `...=24 [NON-MONOTONE]: 4`, `...=26: 12`, `...=28: 8`,
  `...=28 [NON-MONOTONE]: 12`, `...=30: 8`, `...=30 [NON-MONOTONE]: 42`,
  `GRADUAL-DECAY: 15`, `NON-MONOTONE-UNRESOLVED: 4`,
  `...[NON-MONOTONE]: 2`, Σ=125/125. An `assert` fires (hard import
  failure) on any divergence.
- Re-executes the design's own G5-gated 1000-vector trigger split:
  **DECIDED=473, TRIGGER-UNRESOLVED=527** — matches exactly.
- (Ad hoc, not a committed test file, run once for confidence): the
  11-configuration ambiguity table (KW4.5's finding) was independently
  re-derived against `classify.py`'s own functions and reproduced
  **exactly 11** rows, every one at `r_known=2`, every disagreement
  between exactly two adjacent K's — matching the design's printed table
  row-for-row.

### 2b. `tests/test_reconstruction.py` — PASS (4/4)
```
test_24_state_totality: PASS (24 states, 24 distinct (state,outcome) keys, every state resolved)
test_200_composition_old_guard: orphans=30/200 abort_trips=6/200 (design's own KW8.3 figures: 30/200, 6/200)
test_200_composition_new_guard: orphans=0/200 abort_trips=0/200 (design's own §R7 J3 released figures: 0/200, 0/200)
test_derive_cell_state_precedence: PASS
```
Both the OLD-guard defect figures (30/6) and the NEW-guard released
figures (0/0) reproduce the design's own disclosed numbers exactly,
against the REAL `reconstruct_cell`/`derive_cell_state` functions
`orchestrator.py` actually calls (via `kwall_lib/ledger.py`) — not a
hand-transcription.

### 2c. `tests/test_validity_check_suite.py` — PASS (33/33 checks, 0 divergences)
Imports `matrix-thinking/kwall_suites/r10_vcheck.py`, `r10_payloads.py`,
`r10_probes.py` **UNCHANGED** (no edits, no copies) and cross-checks every
payload's verdict, in both `NEW` and `OLD` mode, between the audited
`r10_vcheck.validity_check` and this build's `kwall_lib.validity_check_core`:
- 28 payloads from `r10_payloads.py` (the full committed suite, including
  every literal-variant negative control: `L6-literal0.8571`,
  `A6-literal0.9296`, `A6'-literal0.9296`, `L7'-fstring(pre-J2)`).
- 4 teeth-probes from `r10_probes.py` (D1/D1'/D2/D2', §R9's m3/m7 forced-fail
  demonstrations).
- 1 candidate from `r10_l6fix.py` (candidate A — the fixture R10/R11
  actually adopted as the released L6, per the design's live text and
  §A11-ADJUDICATION's N1 re-derivation): `PASS []` under both.

**Every one of the 33 checks is verdict-identical between the audited
reference and this build's module.** This is the charter's "must pass the
committed 24-payload suite VERBATIM, including every negative control"
requirement, satisfied by construction (the core logic is a byte-for-byte
port, not a re-derivation) and PROVEN by execution against the unmodified
suite scripts, not merely asserted.

`kwall_suites/r9rev_payloads.py` (the Rev-9 agent's own now-superseded
transcription, pre-dating the R10 rewrite; incompatible calling
convention — flat `realized`/`primary_canonical` keys vs. the
`ledger`/`disk` shape every later round uses) was also run unchanged to
confirm it still executes cleanly (it does, no import errors) but was
**not** cross-checked against this build's module — it is historical
record, not part of "the committed 24-payload suite" the charter names.

### 2d. `tests/test_orchestrator_integration.py` — PASS (10/10 tests, all real subprocess dispatch)
```
test_completed_cell                              6/6 checks PASS
test_hard_gate_refused                           6/6 checks PASS
test_retry_then_persistently_aborted             4/4 checks PASS
test_retry_gate_refused                          4/4 checks PASS
test_stop_file                                   4/4 checks PASS
test_g2_exists_check_aborts_loudly               2/2 checks PASS
test_ledger_corruption_recovery                  3/3 checks PASS
test_mid_attempt_sigkill_recovery                6/6 checks PASS
test_gate_refused_report_passes_validity_check   2/2 checks PASS
test_end_to_end_run_reduced_grid                 3/3 checks PASS (+1 informational)
```
Every test drives the REAL `orchestrator.py` functions
(`dispatch_attempt`/`dispatch_cell`/`OrchestratorState.recover`/`run`)
through real `subprocess.run`/`subprocess.Popen` calls against
`cell_runner_stub.py` (a CPU-STUB — flagged per CLAUDE.md's CPU-stub hard
rule; it emulates only the OBSERVABLE contract of `ncr_earlyln_scale.py
--cell`: exit codes, on-disk JSON schema, `--stop-file` handling — no
torch/CUDA training). Highlights:
- **`test_mid_attempt_sigkill_recovery`** (red-team item vi, forced to
  completion, not just written): launches a real driver subprocess that
  write-aheads an `open_attempt` and blocks inside a real
  `subprocess.run` call to a hanging stub; the test polls the on-disk
  ledger until `open_attempt` is observed non-null, then `SIGKILL`s the
  entire process group. A fresh `OrchestratorState.recover()` in a new
  process then correctly closes the dangling attempt as exactly one
  `CRASHED-RECOVERED` row charged at the full `1.20` ceiling — no gap, no
  double-charge.
- **`test_ledger_corruption_recovery`** (red-team item x): truncates
  `ORCHESTRATOR_LEDGER.json` mid-file after one real COMPLETED dispatch,
  then confirms fresh recovery does NOT reset to `realized_gpu_h=0` and
  correctly recovers the COMPLETED row from the canonical file.
- **`test_g2_exists_check_aborts_loudly`**: pre-seeds a canonical file,
  confirms `dispatch_attempt` raises `RuntimeError` (ABORTS LOUDLY) rather
  than silently overwriting.
- **`test_hard_gate_refused`**/**`test_retry_gate_refused`**: confirm the
  bug found and fixed during this build (§4 below) — a `GATE-REFUSED`
  attempt-1 row is now correctly terminal (`PERSISTENTLY-ABORTED`, no
  second attempt ever dispatched), and a retry specifically refused by
  the `<12.00` RETRY gate (while the `<=15.00` HARD gate would have
  admitted it) is distinguished correctly.
- **`test_end_to_end_run_reduced_grid`**: the only test that runs the full
  `run()` entrypoint (startup smoke gate → primary sweep → REAL
  `harvest()`/trigger/band evaluation via `kwall_lib.harvest_bridge`,
  which imports and calls the repo's own `ncr_earlyln_scale.harvest` →
  report emission), against a monkeypatched 1×2 reduced grid (K=26,
  seeds={0,1}) for speed. `run_status=COMPLETE`, 2/2 canonical files,
  `run()` exits 0.

### 2e. CPU-runnable smoke of the two code-path patches (§3 below) + a real forward/backward/grad-finite/optimizer.step() proxy
```
$ cd matrix-thinking/ncr && python3 -c "... GRID_SHAPES/GRIDS resolve for K in (26,28,30) ..."
26 GRID_SHAPES {'d': 52, 'h': 64} GRIDS h_star 205 ladder_residue 23
  claim_config OK, d= 27 K= 26
  eval_points OK, n= 41
28 GRID_SHAPES {'d': 56, 'h': 64} GRIDS h_star 221 ladder_residue 25
  claim_config OK, d= 29 K= 28
  eval_points OK, n= 43
30 GRID_SHAPES {'d': 60, 'h': 64} GRIDS h_star 237 ladder_residue 27
  claim_config OK, d= 31 K= 30
  eval_points OK, n= 45
ALL OK -- imports clean, no KeyError, periodicity asserts pass for K in {26,28,30}
```
Followed by a REAL (not stubbed) forward + backward + gradient-finite +
`optimizer.step()` check against the actual `NCREarlyLNModel` class, at
`d=K+1`, batch size 8 (reduced from the harness's real `TRAIN_BATCH=256`
purely for CPU wall-clock feasibility — see §5):
```
K=26 d=27 n_params=173723 loss=1.0082 grads_finite=True params_changed=True
K=28 d=29 n_params=174237 loss=0.9983 grads_finite=True params_changed=True
K=30 d=31 n_params=174751 loss=0.9971 grads_finite=True params_changed=True
REAL CPU SMOKE PASS: forward + backward + grad-finite + optimizer.step() for K in {26,28,30} at d=K+1
```
This satisfies CLAUDE.md's smoke-test hard rule (forward/backward/gradient
check) on the ONE model-touching code path this build's own edits affect.
It is **not** the design's own exact 500-step/`TRAIN_BATCH=256`
micro-smoke pass criterion — see §5 for why, and `smokes/MICRO_SMOKE_SPEC.md`
for the exact box commands that ARE that criterion.

## 3. The one code-stage edit: additive `GRID_SHAPES`/`GRIDS` patches

The design's §2 "Build note" names this explicitly as required and
pre-authorized (not a design gap — implementation deferred to build):
`GRID_SHAPES[K]` (`ncr_earlyln_scale.py`) and `GRIDS[K]`
(`ncr_task.py`, via `nt._gen_grid`) have no entries for K∈{26,28,30};
`nt.claim_config(K,...)` and the CLI's `--K choices=sorted(GRID_SHAPES)`
both require K already be a dict key, and `_cell_gate2`'s
`nt.GRIDS[K]["h_star"]` lookup (reached whenever ANY stub/real cell at a
new K converges during `harvest()`) has the identical requirement — this
build found and confirmed that second call site independently (the
design's own text names only `claim_config`/`--K choices`; the
`_cell_gate2` KeyError risk was verified by direct code read during this
build, not previously spelled out).

```diff
--- matrix-thinking/ncr/ncr_earlyln_scale.py  (GRID_SHAPES, +11 lines, additive only)
+++
     192: dict(d=384, h=64),
     256: dict(d=512, h=64),
+    # NCR_KWALL_CHARACTERIZATION_DESIGN.md §2 "Build note" ...
+    26: dict(d=52, h=64),
+    28: dict(d=56, h=64),
+    30: dict(d=60, h=64),
 }
```
```diff
--- matrix-thinking/ncr/ncr_task.py  (GRIDS, +12 lines, additive only)
+++
 for _K_new in (20, 32, 48, 64, 96, 128, 192, 256):
     GRIDS[_K_new] = _gen_grid(_K_new)
+
+# NCR_KWALL_CHARACTERIZATION_DESIGN.md §2 "Build note" ...
+for _K_new in (26, 28, 30):
+    GRIDS[_K_new] = _gen_grid(_K_new)
```
`git diff --stat`: `ncr_earlyln_scale.py | 11 +++++++++++`,
`ncr_task.py | 12 ++++++++++++` — **0 deletions, 0 modifications to any
existing key**, matching the design's own "additive only, no existing
key mutated" requirement (verified: the file's own regression assert at
`ncr_task.py:150-152`, which checks K∈{14,15,16,24} against `_gen_grid`,
still runs and passes at import time — confirmed above, "imports clean").
`d` values use the same Condition-A `d=2K` convention as every other
`GRID_SHAPES` entry (informational only — every kwall cell command passes
`--d-override K+1`, which overrides `d_eff` regardless).

**This patch is committed to the repo (`matrix-thinking/ncr/`) but NOT
deployed to the box** — "do not touch the box" per the build charter. The
box's own copies of these two files must receive the SAME additive diff
before the micro-smokes or the orchestrator can run there (see
`smokes/MICRO_SMOKE_SPEC.md` "Preconditions" and the job spec's own
`notes` field).

## 4. A real bug found and fixed during this build

Independent review of the ORCHESTRATOR CONTRACT's derived-cell-state rule
("PERSISTENTLY-ABORTED iff (no row COMPLETED) and [attempt-2 row exists
and non-COMPLETED, OR attempt-1 row is non-COMPLETED, no attempt-2 row
exists, and the retry gate closed it]") found that a literal
`attempt_n==2`-only reading of that rule left a `GATE-REFUSED`
attempt-1 row (a first attempt refused outright by the HARD gate) as
`NON-TERMINAL` — which would have caused `dispatch_cell`'s loop to
attempt a SECOND dispatch after a refused FIRST attempt, contradicting
the dispatch loop's own explicit text ("Refused → append a GATE-REFUSED
row ... move to the next cell"). Traced this precisely: the "no
attempt-2 row exists, and the retry gate closed it" disjunct is exactly
the attempt-1-`GATE-REFUSED` case (the ONLY way that state is reached
terminally in normal operation — a genuinely-run attempt-1 that fails
with `ABORTED-BUDGET`/`CRASHED` is instead legitimately `NON-TERMINAL`,
owed a retry). Fixed in `kwall_lib/reconstruction.py::derive_cell_state`;
re-verified the fix does not change the 200-state composition sweep's
30/6 (OLD) and 0/0 (NEW) figures (GATE-REFUSED never appears among
reconstructed-row statuses, only in LIVE dispatch, so the reconstruction
sweep was unaffected) and added `test_hard_gate_refused` as the negative
test proving the fix (a refused attempt-1 now correctly produces exactly
ONE row and immediate `PERSISTENTLY-ABORTED`, never a second dispatch).

## 5. Deferred to the box, and why

1. **The 3 exact micro-smokes** (K=26/28/30, 500 steps, `TRAIN_BATCH=256`,
   the design's own F3 pass criterion) — `smokes/MICRO_SMOKE_SPEC.md` has
   the exact commands. **Why CPU-infeasible in build-agent time:** a
   timed probe of ONE full `run_earlyln_cell` call at K=24 (20 steps,
   `device="cpu"`) did not complete within 120s; a further probe isolated
   the cost to the harness's default `TRAIN_BATCH=256` (individually,
   `claim_config`/model-construction/`sample_batch(bs=8)` each measured
   <0.01s — the slowdown is specifically the real batch size, not a
   shape/import problem). §2e's reduced-batch (bs=8) proxy is the
   CPU-runnable substitute; it is NOT the design's own literal pass
   criterion (which needs the real batch size and 500 real steps to
   "exercise one full forward/backward pass and one optimizer step... at
   each new K's actual d=K+1 shape" at the shape the box will really run).
2. **Deploying `orchestrator.py` + `kwall_lib/` to `/home/nvidia/ncr/`
   on the box**, and deploying the two additive patches (§3) to the
   box's own `ncr_earlyln_scale.py`/`ncr_task.py` — "do not touch the
   box" is an explicit constraint of this build stage.
3. **Red-team items (vii), (viii), (ix), (xi), (xiii), (xiv)** (§6) —
   `STOPPED-BY-OPERATOR`-mid-run-vs-`ABORTED-BUDGET` misclassification
   is covered by `test_stop_file` (CPU-runnable, done); a genuine
   `nvidia-smi`-based GPU-reap check (xiii) is implemented in
   `orchestrator.py::gpu_reap_check` but degrades to a disclosed,
   printed no-op when `nvidia-smi` is unavailable (this Mac) rather than
   silently claiming the GPU is free — exercising it for real (an
   orphaned CUDA process surviving a parent-only kill) needs a real GPU.
   Items (viii)/(ix) (the full `validity_check` accept-set match, the G5
   trigger precondition) are exercised end-to-end by
   `test_gate_refused_report_passes_validity_check` and
   `test_end_to_end_run_reduced_grid` respectively at the CODE level;
   the design's own staging (§A11-ADJUDICATION: "then the build's OWN
   audit + pre-launch resource/placement red-team... specs enter the box
   `fallback_pool/` ONLY after that audit passes") treats the FULL
   red-team ceremony as the NEXT stage after this build, not this
   build's own deliverable.
4. **KW2.7's on-box `~/queue/{fallback_pool,claimed}` sweep** for
   K∈{26,28,30} content — needs box access this build stage does not
   have; still outstanding per every prior round's own disposition.

## 6. Disclosed deviations from the literal build-dispatch instruction

**"The box job-spec JSONs for the 12 cells + the conditional arm"** was
read as **one** job-spec JSON that internally dispatches all 12 primary +
up to 4 conditional cells — not 12(+4) separate pool-eligible specs. This
is not a simplification; it is the design's own, heavily-audited delivery
model, changed at Rev 3 specifically because 16 independent pool specs
sharing a cumulative budget gate is a **FATAL**, structurally
unfixable violation of the pool's own no-intra-wave-dependency contract
(`§A3-ADJUDICATION`, `NCR_KWALL_ATTACK_R3.md` — three independent FATALs
all traced to this one root cause). §6 POOL-ELIGIBILITY STATEMENT states
outright: "pool-eligibility attaches to the ORCHESTRATOR SPEC, not to the
16 cells... No per-cell job specs are created — the orchestrator is the
only pool artifact this design produces." Building 12+4 separate specs
would have reintroduced the exact defect 3 audit rounds killed and
reproduced no other artifact this design's own text authorizes.
`job_specs/kwall_char_orchestrator.json` is that one spec, `notes`-field
disclosing every precondition (micro-smokes, box deployment, red-team)
before it may be promoted to `queue/jobs/pending/`.

**A small number of test-only fixtures were corrected mid-build for
item-(j) producibility** (a `ceiling_charged=true` synthetic ledger row
in `test_gate_refused_report_passes_validity_check` originally used an
arbitrary padding `elapsed_h` rather than the exact `1.20` ceiling value
— fixed to use N rows at exactly the ceiling, per the same rule
KW9.4/KW10.1/KW11.1 closed for the design's own fixtures). No committed
`kwall_suites/` payload needed any correction — all 33 cross-checks in
§2c passed on the first run.

## 7. What the build audit must check

1. **`orchestrator.py`'s fidelity to the ORCHESTRATOR CONTRACT line by
   line** — cell order, the write-ahead/classify/copy-then-fold sequence,
   the HARD/RETRY gate check points, G2's exists-check-aborts-loudly, the
   recovery procedure's steps 0/1/2/3/4 (this build's own review already
   found and fixed one real gap — §4 above — a second independent pass is
   exactly the point of a separate audit).
2. **`determine_run_status`'s G4 encoding** — it deliberately relies on
   `dispatch_cell`'s own loop invariant (every primary pair gets ≥1 row
   before `run()` reaches trigger evaluation) rather than re-deriving
   that invariant defensively; confirm this reliance is sound (i.e., that
   no code path can reach `determine_run_status` without the invariant
   holding) rather than assuming this report's own claim.
3. **`kwall_lib/harvest_bridge.py`'s reliance on `ncr_earlyln_scale.harvest()`
   as-is** — the design says no `harvest()` patch is needed; confirm this
   build did not silently need one (it did not, per `test_end_to_end_run_reduced_grid`,
   but the audit should verify the reasoning, not just the one passing test).
4. **Whether `kwall_lib/validity_check.py`'s disk-reading glue
   (`_glob_canonical`/`build_disk_view`) correctly matches G2's real
   canonical-directory shape** — this was not exercised against files
   `run()` itself produced with a FULL 12-cell grid (only the 2-cell
   reduced grid and the direct-construction `disk` dicts the r10_
   payloads suite uses) — box-side or a larger local run should confirm.
5. **The trigger `diag` disambiguation in `evaluate_trigger_and_band`**
   (`blocking_K` vs `band_blocked_K_trig`, keyed off whether any per-K
   state is the raw `"UNRESOLVED"` sentinel) — this build's own
   resolution of the design's "key off the RETURN SITE, never tuple
   position alone" note; worth an independent re-derivation.
6. **`gpu_reap_check`'s no-op-on-missing-`nvidia-smi` behavior** — verify
   this degrades safely (never silently claims a GPU is free when one
   genuinely cannot be checked, only when the check itself is
   unavailable) and that the box-side execution path (where `nvidia-smi`
   IS available) actually exercises the raise branch, not just the
   no-op branch this build's own test environment always takes.
7. **Item (j) compliance across every fixture this build wrote** — §2c/§6
   above disclose the one fixture this build itself had to correct;
   confirm no other fixture (in `test_orchestrator_integration.py` or
   elsewhere) carries an unreachable `elapsed_h` for its `status`/
   `ceiling_charged` combination.

## 8. What this build did NOT do

- Did not modify `NCR_KWALL_CHARACTERIZATION_DESIGN.md`, `STATE.md`, or
  `EXPERIMENT_LOG.md`.
- Did not touch the box (no SSH, no deploy, no queue mutation).
- Did not launch anything.
- Did not re-derive any pinned equation from the design (τ, s, the
  15.3737/15.00/12.00/15.50 figures, the 125-outcome and 1000-vector
  tables) by hand — every one is either imported as a constant with its
  derivation cited in a comment, or independently RE-EXECUTED at
  import/test time against the design's own disclosed figures (§2a/§2b),
  never merely copied.

## REV-1 — fixes for `BUILD_AUDIT_R1.md` (1 FATAL, 4 MAJOR, 8 minor, 3
design-inherited observations)

Scope discipline: every edit below lives under `matrix-thinking/kwall_build/`
(`git diff --stat` confirms — 10 files, 0 touched outside this directory).
Did not modify `NCR_KWALL_CHARACTERIZATION_DESIGN.md`, any `matrix-thinking/
ncr/` recipe file, `STATE.md`, `EXPERIMENT_LOG.md`, or the box.

### Per-finding disposition table

| Finding | Fix | Test | Result |
|---|---|---|---|
| **F1 (FATAL)** — `classify_with_interval_logic`'s UNRESOLVED early return disclosed only `UNRESOLVED` K's, dropping any co-occurring `AMBIGUOUS` (n_completed==3) K from BOTH `incomplete_at_K` and `interval_resolved_Ks`; `validity_check`'s COMPLETE/otherwise clause ("every K named in NEITHER field reads exactly 4") then failed on the dropped K's true count of 3 | `kwall_lib/classify.py::classify_with_interval_logic` — the early return now discloses `sorted(UNRESOLVED ∪ AMBIGUOUS)` as `incomplete_at_K`, not `UNRESOLVED` alone | (1) audit's own `h3_full_grid.py` re-run unmodified, real `orch.run()` + real `python3 -m kwall_lib.validity_check` CLI; (2) `h1_rev1_band_disclosure.py`, full 3375-state `(n_completed,n_converged)` enumeration | (1) `C_mixed_unres_amb`: `incomplete_at_K=[28,30]` (was `[30]`), `VALIDITY_CHECK: PASS`, CLI `exit=0` (was FAIL/exit=1); A/B/D controls unchanged PASS. (2) **1/3375** states fail their own `validity_check` — exactly the O1 `(0,0,0)` design-inherited residual, **0 build-defect failures** (was 1440/3375, 54 shapes) |
| **M1 (MAJOR)** — the trigger `diag` return-site disambiguation was wrong in 115/1000 reachable state vectors: the raw K-scan blocked on ANY `UNRESOLVED` K regardless of scan position, and the consumer (`evaluate_trigger_and_band`) inferred the return site from "any state is UNRESOLVED" | Two-part: (a) `classify.py::_smallest_K_with_rate_below_3_or_blocking`/`_compute_K_trigs` — the scan now blocks only when it actually REACHES an UNRESOLVED K before deciding; (b) new `classify.trigger_raw_scan_blocked()` — **found during this fix's own re-verification**: (a) alone leaves the consumer's old "any UNRESOLVED present" heuristic wrong post-fix (an UNRESOLVED K past the decision point no longer blocks the scan, so the heuristic mis-attributes `diag`); `orchestrator.py::evaluate_trigger_and_band` now calls `trigger_raw_scan_blocked()` instead | `h2_rev1_trigger_diag.py` (independent reference re-implementation, full 1000-vector sweep) + `h2_teeth_revert.py` (revert-style: reconstructs the pre-fix scan inline, confirms it STILL reproduces 115/1000) | pre-G5 844/156 MATCH, post-G5 473/527 MATCH, `(K_trig,resolution)` 0/1000 diffs, **`(blocking_K, band_blocked_K_trig)` 0/1000 diffs** (was 115/1000); teeth check reproduces 115/1000 on the reconstructed pre-fix code, confirming the diagnostic has teeth; audit's own named example `(0,0,'UNRESOLVED')` now reads `blocking_K=None, band_blocked_K_trig=26` exactly as the design requires |
| **M2 (MAJOR)** — job-spec `cmd` used `$(git rev-parse HEAD)`; dies at argparse if `/home/nvidia/ncr` is not itself a git working tree, refusing all dispatch before any GPU-h is spent | `job_specs/kwall_char_orchestrator.json`: `--git-commit $(git rev-parse HEAD)` → `--git-commit __DEPLOY_GIT_COMMIT__` (a literal placeholder token, zero command substitution); the deploy step stamps it (see corrected deploy sequence below, step 3b); `notes` field updated to state this precondition explicitly | (1) grep of `cmd`/`validity_check` fields for `$(` — the two fields ever shell-executed by `queue_worker.sh`; (2) stamp-simulation (placeholder → literal, re-grep); (3) revert-style teeth: reproduced the audit's exact non-git-cwd failure with the OLD `$(git rev-parse HEAD)` shape, then re-ran the SAME shell invocation with a stamped literal | (1) 0 occurrences of `$(` in `cmd` or `validity_check` (the `notes` field's ONE `$(...)` mention is prose citing the removed pattern by name — never shell-executed); (2) stamped spec is `$(`-free and placeholder-free; (3) OLD shape: `fatal: not a git repository`, argparse `error: --git-commit: expected one argument`, exit 2 — reproduced exactly; NEW stamped literal: clean `Namespace(...)`, exit 0 |
| **M3 (MAJOR)** — G4's mandated pre-write `stop_file_path` self-check was absent; a conditional-arm stop reported the PRIMARY sentinel path; a primary-arm stop reported the fabricated literal `{26:(0,0),28:(0,0),30:(0,0)}` for `per_K` regardless of real completions | `orchestrator.py`: new `_assert_stop_file_evidence()`, called from `build_report` whenever `run_status=="STOPPED-BY-OPERATOR"`, before `atomic_write_json`; `build_report` now takes an explicit `stop_file_path` param (caller supplies `args.stop_file` or `args.stop_file_conditional`, never a hardcoded default); the primary-loop stop now reads REAL `harvest_bridge.per_K_resolution(...)` instead of the fabricated zeros dict | `test_m3_stop_file_self_check_and_real_per_K` (real `run()`, dispatches (26,0) to a genuine COMPLETED cell, THEN stops — confirms `per_K["26"].n_completed==1`, not 0) + `test_m3_conditional_stop_uses_conditional_path` (real 12-cell sweep to a decided `K_trig=26`, conditional-loop stop, confirms `stop_file_path == args.stop_file_conditional ≠ args.stop_file`) + direct self-check teeth (`_assert_stop_file_evidence` raises on a nonexistent path and on `None`) | All PASS: real per_K disclosure (`26:(1,·)`, `28:(0,·)`, `30:(0,·)`) on a primary stop; `stop_file_path` correctly the CONDITIONAL sentinel on a conditional stop, self-check never spuriously fires on the genuine path, and DOES raise `AssertionError` on the two adversarial inputs |
| **M4 (MAJOR)** — `trigger.candidate_set`, `band.candidate_bands`, `conditional.per_seed` hardcoded `None`/`None`/`[]` regardless of outcome | `classify.py`: `classify_with_interval_logic` now returns a 5-tuple (adds `candidate_bands`, populated only when interval candidates disagree — design `:2249-2250`); new `trigger_candidate_set()` (shares `_compute_K_trigs` with `trigger()`, so it can never drift). `kwall_lib/harvest_bridge.py`: new `per_seed_gate1_records()` (deliberately wider-scoped than `per_K_resolution`'s decision-only contract — disclosure only, never consumed by classify.py). `orchestrator.py`: `evaluate_trigger_and_band`/`build_report`/`run()` thread all three through; new `_conditional_per_seed()` populates the throttled-arm disclosure exactly when `qualifier_band is None` | `test_m4_schema_fields` — pure-function checks against the design's own KW4.5 row 1 (tie-break, `candidate_set==[26,28]`) and an independently-found disagreeing-candidates case (`candidate_bands==["FRONTIER-AT-K*=24","GRADUAL-DECAY"]`), PLUS end-to-end through real canonical files → `evaluate_trigger_and_band` → `build_report`, PLUS a throttled 2/4 conditional-arm `per_seed` disclosure (real gate1 records, correct per-seed CONVERGED/DEAD split), PLUS revert-style teeth (the pre-fix hardcoded values are confirmed to DIFFER from every real value produced) | All PASS (17 checks) |
| **m1** — `test_orchestrator_integration.py`'s ledger-padding fixture used unproducible `(K=30, seed=8..19)` pairs (`SEEDS=(0,1,2,3)` only) | `test_gate_refused_report_passes_validity_check` rebuilt to use exclusively the 12 valid `(K,seed)` pairs: 11 carry `[attempt-1 CRASHED-RECOVERED@1.20, attempt-2 COMPLETED@0.01]`, 1 carries `[attempt-1 CRASHED-RECOVERED@1.20, attempt-2 GATE-REFUSED@0.0]` | re-run to completion | PASS — `ccgh==14.40` (12×1.20, item (j) arithmetic asserted), `validity_check_core` PASS, 11 distinct COMPLETED pairs |
| **m2** — `reconstruct_attempt_row` crashed (`TypeError`) on a parseable `COMPLETED` record missing top-level `elapsed_s` | Two layers: `reconstruction.py::reconstruct_attempt_row`'s COMPLETED branch now guards `isinstance(elapsed_s,(int,float))`, bootstrapping to `CRASHED-RECOVERED@ceiling` instead of crashing; `disk_io.py::read_attempt_evidence` also classifies such a record as `"unparseable"` at the READ boundary (defense in depth, never even constructing the dangerous combination from a real file) | `test_m2_completed_missing_elapsed_s_no_crash` (both layers) + revert-style teeth (reproduces the exact pre-fix `TypeError`) | PASS — no crash at either layer; teeth check confirms the pre-fix code path does TypeError on the identical input |
| **m3** — `reconstruct_ledger_from_disk`'s conditional-4 reconstruction only fired when `_infer_conditional_K` found a `K{K}_s{seed}_attempt{n}/` directory; an attempt tree pruned after copying (0.2's own named residual case) left surviving conditional canonical evidence uncharged | `orchestrator.py::OrchestratorState._infer_conditional_K` — added a fallback that infers K from canonical filenames (`earlyln_K{K}_s{seed}.json`) when no attempt directory survives | `test_m3_infer_conditional_K_pruned_attempt_tree` (canonical-only directory, no attempt dirs at all) | PASS — K correctly inferred as 28 from canonical evidence alone; empty-directory case still correctly returns `None` (no regression) |
| **m4** — `BUILD_REPORT.md`/job-spec `notes` both said "see BUILD_REPORT.md for the exact deploy commands," but no deploy commands existed in this file | Discharged by this REV-1 section itself — see "Corrected deploy sequence" below, the exact commands the `notes` field now points to | inspection | Discharged |
| **m5** — BUILD_REPORT §4 overstated the pre-fix bug's consequence as "a SECOND dispatch" | Corrected below (see "m5 wording correction") to the audit's own executed finding: two `elapsed_h=0.0` `GATE-REFUSED` rows, no subprocess, no GPU-h — cosmetic, not a real second dispatch | n/a (documentation correction) | Discharged |
| **m6** — `MICRO_SMOKE_SPEC.md` added an undesigned `--stop-file` arg; `≤0.15 GPU-h` restated without the design's own training-loop-only caveat | `smokes/MICRO_SMOKE_SPEC.md`: removed `--stop-file` (byte-for-byte the design's own command now); added the D5 training-loop-only caveat to the budget paragraph | diff against design `:2861-2864` (inspection) | Discharged |
| **m7** — `validity_check` CLI raised an uncaught `FileNotFoundError` (traceback, not a diagnosis) when no report was ever written (startup smoke refusal / G2 abort) | `kwall_lib/validity_check.py::main` — explicit `os.path.exists` check before opening, prints a clean `VALIDITY_CHECK FAIL` diagnosis and returns 1 (same routing as before) | `test_m7_validity_check_cli_missing_report_no_traceback` (real CLI subprocess against a nonexistent report path) | PASS — exit 1 (unchanged routing), stderr carries the diagnosis, zero `Traceback` |
| **m8** — on-box `~/queue/{fallback_pool,claimed}` K∈{26,28,30} sweep still undischarged (needs box access) | Not discharged by this build stage — "do not touch the box" is an explicit constraint here too. Carried forward as step 6 of the corrected deploy sequence below (unchanged from the audit's own step 5) | n/a | **Carried forward, not a code fix** |

### Design-inherited observations O1–O3 — dispositions (NOT build defects; recorded, not silently absorbed or dropped)

- **O1** (a run where every primary cell crashes twice — 0 canonical, no
  budget refusal — is `COMPLETE` by G4's own definition, then rejected by
  §R8 K2's unconditional `>=1 COMPLETED primary pair` clause). **Disposition:
  NOT FIXED, per the audit's own reasoning** — "this is the design's own
  text, faithfully implemented." K2 is unconditional in the design's own
  clause; changing this behavior would mean editing
  `NCR_KWALL_CHARACTERIZATION_DESIGN.md`, explicitly out of scope for a
  build stage. This is the ONE state (`(0,0,0)`, 1/3375) still present in
  the F1 re-sweep above, exactly as the audit predicted it would remain.
- **O2** (`COMPLETE-DEGRADED`'s throttle-evidence clause tests for
  `PERSISTENTLY-ABORTED` among `attempts[].status`, but that value is
  provably never an attempt status — only `GATE-REFUSED` is live).
  **Disposition: NOT FIXED, per the audit's own reasoning** — "it is in the
  R11-audited reference verbatim; the build's byte-for-byte port inherits
  it, and the live half (`GATE-REFUSED`) is exactly what
  `determine_run_status` keys on, so the two stay consistent." Touching
  `validity_check.py`'s ported core here would break the byte-for-byte-port
  claim §2c's 33/33 cross-check depends on, for a dead clause with no
  behavioral consequence. Untouched.
- **O3** (the crash-window table's row 3, "between copy and fold ⇒ full
  ceiling charged," is unreachable as written because §R6 I2's step 2.1
  precedence always fires first when the attempt JSON is readable-COMPLETED).
  **Disposition: NOT FIXED, per the audit's own reasoning** — "the build
  follows the *procedure* text (2.1 precedence), which is correct,"
  and "the `T ≤ 15.3737` bound survives" regardless. This is a design-doc
  narrative-vs-procedure inconsistency, not a build defect; `reconstruction.py`
  already implements the (correct) procedure. Untouched.

### m5 wording correction

BUILD_REPORT.md §4 above (unedited, left as historical record of what was
found) says the pre-fix bug "would have caused `dispatch_cell`'s loop to
attempt a SECOND dispatch after a refused FIRST attempt." The audit
executed this claim and found it overstated: with the fix reverted, the
second attempt is ITSELF `GATE-REFUSED` (the ledger is unchanged, so the
same gate refuses again) — two `elapsed_h=0.0` rows, no subprocess ever
launched, no GPU-h spent. **Corrected statement:** the fix is still
correct and necessary for `attempts[].status` contract fidelity (a refused
attempt-1 must derive `PERSISTENTLY-ABORTED` in ONE row, not two), but its
pre-fix consequence was a cosmetic double `GATE-REFUSED` row, never a real
second subprocess dispatch or a real GPU-h leak.

### Re-run suite figures (all green after REV-1)

| Suite | Result |
|---|---|
| `tests/test_reconstruction.py` | **PASS 5/5** (added `test_m2_completed_missing_elapsed_s_no_crash`) — 24-state totality; OLD guard 30/200 orphans + 6/200 abort-trips; NEW guard 0/200 + 0/200; m2 fix confirmed at both layers, teeth confirmed |
| `tests/test_validity_check_suite.py` | **PASS 33/33**, 0 divergences vs the unmodified `kwall_suites/r10_vcheck.py` (untouched by REV-1 — `validity_check_core`'s byte-for-byte port was never edited) |
| `tests/test_orchestrator_integration.py` | **PASS 15/15 tests, 80 individual checks, 0 failures** (added `test_m3_stop_file_self_check_and_real_per_K`, `test_m3_conditional_stop_uses_conditional_path`, `test_m4_schema_fields`, `test_m3_infer_conditional_K_pruned_attempt_tree`, `test_m7_validity_check_cli_missing_report_no_traceback`) |
| **33-payload verdict-identity check** | Included in `test_validity_check_suite.py` above — **33/33 unchanged**, since `validity_check.py`'s ported core (`validity_check_core`) was not touched by any REV-1 fix (only its CLI's `main()` gained the m7 fix, which is outside the ported core) |
| **F1 full-enumeration re-sweep** (`h1_rev1_band_disclosure.py`, 3375 states) | **1/3375 fail their own `validity_check`** — exactly the O1 `(0,0,0)` design-inherited residual; **0 build-defect failures** (was 1440/3375, 54 shapes) |
| **M1 full 1000-vector diag re-sweep** (`h2_rev1_trigger_diag.py`) | pre-G5 844/156 MATCH, post-G5 473/527 MATCH, **0/1000** `(blocking_K, band_blocked_K_trig)` divergences (was 115/1000) |
| **Audit's own end-to-end demonstration** (`h3_full_grid.py`, unmodified, real `orch.run()` + real CLI) | `C_mixed_unres_amb`: `incomplete_at_K=[28,30]`, **`VALIDITY_CHECK: PASS`, CLI `exit=0`** (was FAIL, exit=1); A/B/D controls unchanged PASS |

### Corrected deploy sequence

Incorporates the audit's own verified sequence (`BUILD_AUDIT_R1.md`,
"Deploy sequence I verified") plus the M2 stamping step (new, 3b) — still
blocked until build audit R2 passes:

1. ~~Fix F1 and M1; adjudicate M2–M4.~~ **DONE this round** (F1/M1/M2/M3/M4
   all fixed, see disposition table above).
2. Re-run all three suites + `h1`/`h2`/`h3` from a clean checkout — **DONE
   this round**: `h1` reports 1/3375 (O1 only, 0 build-defect), `h2`
   reports 0/1000 diag divergences. (Superseded acceptance criterion: the
   audit's own text said "`h1` must report 0 outside O1's single state" —
   satisfied exactly.)
3. `bash matrix-thinking/queue/deploy.sh` (or its staged `scp`+md5-verify
   path) to land the two additive recipe patches
   (`ncr_task.py`/`ncr_earlyln_scale.py`) on the box; verify md5 equality
   box-side before publish, as `deploy.sh` already does.
   **3b. (NEW, M2 fix)** Before `job_specs/kwall_char_orchestrator.json`
   is ever placed on the box: stamp the literal commit hash into it
   locally (where `git` IS available — this repo IS a working tree),
   e.g. `sed "s/__DEPLOY_GIT_COMMIT__/$(git rev-parse HEAD)/" \
   job_specs/kwall_char_orchestrator.json > /tmp/kwall_char_orchestrator.stamped.json`,
   then verify the stamped copy has ZERO occurrences of
   `__DEPLOY_GIT_COMMIT__` and ZERO occurrences of `$(` in its `cmd`/
   `validity_check` fields before it is ever scp'd to the box. Never scp
   the placeholder-carrying original — the box's `~/ncr` is not
   guaranteed to be a git working tree (confirmed: `deploy.sh` contains
   no `git clone`/`git init` anywhere), which is exactly what M2 closes.
4. `scp orchestrator.py` and the `kwall_lib/` package to
   `/home/nvidia/ncr/`; confirm `kwall_lib/__init__.py` is present so
   `-m kwall_lib.validity_check` resolves.
5. Sweep `~/queue/fallback_pool/` and `~/queue/claimed/` for K∈{26,28,30}
   content (§6 item (v), m8) — **still undischarged, needs box access this
   build stage does not have.**
6. On a GPU verified idle by `nvidia-smi`, run the 3 micro-smokes from
   `smokes/MICRO_SMOKE_SPEC.md` (m6-corrected: no `--stop-file`, training-
   loop-only budget caveat), then the spec's own verification snippet
   (`orch.check_smoke(...)` must read `PASS` for all three).
7. Only then place the **stamped** (M2-corrected, per step 3b) job spec in
   `~/queue/fallback_pool/`.
