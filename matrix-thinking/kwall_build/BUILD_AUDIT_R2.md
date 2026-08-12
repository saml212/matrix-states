# K-wall build audit R2 — narrow re-verification of Rev-1

**Scope:** `matrix-thinking/kwall_build/` at commit `d918074`, against
`BUILD_AUDIT_R1.md` (1 FATAL, 4 MAJOR, 8 minor, 3 design-inherited
observations) and `BUILD_REPORT.md` `## REV-1`. NARROW round by charter:
verify the Rev-1 fixes and the standing green state only. Everything
`BUILD_AUDIT_R1.md` marked PASS with evidence is settled and was NOT
re-audited. Independent pass — neither the implementer nor the R1 auditor
wrote this. No box contact; no `STATE.md` / `EXPERIMENT_LOG.md` edit; this
report is this audit's only repo write.

**VERDICT: PASS — the build is DEPLOY-RELEASED.**
0 FATAL, 0 MAJOR, 3 new minor, 3 observations. Every R1 FATAL/MAJOR is
discharged, and each discharge was re-verified through the **real
production functions**, not through the reviser's own mirrors. The
verified deploy sequence is enumerated at the end; three of its steps
needed correction (all minor, all in the sequence text, none in the code).

All findings below are executed, not asserted. My own harnesses live in
`/private/tmp/claude-501/-Users-samuellarson-Experiments-learned-representations/be705417-f189-4cd8-8024-24cf6a0130a0/scratchpad/`
(`r2_a_realpath_3375.py`, `r2_b_realpath_diag_1000.py`, `r2_c_m2_stamp.py`,
`r2_d_m3_m4.py`, `stub3.py` — an arm-aware CPU stub written for this
round); the R1 and Rev-1 harnesses (`h3_full_grid.py`,
`h1_rev1_band_disclosure.py`, `h2_rev1_trigger_diag.py`,
`h2_teeth_revert.py`) were also re-run unmodified.

---

## 1. F1 (FATAL) — DISCHARGED

### 1a. R1's own end-to-end demonstration, re-run unmodified

`h3_full_grid.py` (real `orch.run()`, real subprocess dispatch, real
`harvest_bridge` → repo `harvest()`, real report emission, real
`python3 -m kwall_lib.validity_check` CLI). All four cases:

```
A_all_complete     per_K 26:(4,4) 28:(4,4) 30:(4,4)  band=FRONTIER-AT-K*=30
                   VALIDITY_CHECK: PASS   CLI exit=0
B_K30_two_crash    per_K 26:(4,4) 28:(4,4) 30:(2,2)  incomplete_at_K=[30]
                   VALIDITY_CHECK: PASS   CLI exit=0
C_mixed_unres_amb  per_K 26:(4,4) 28:(3,3) 30:(2,2)  incomplete_at_K=[28, 30]
                   VALIDITY_CHECK: PASS   CLI exit=0        <-- was FAIL / exit=1
D_single_amb       per_K 26:(4,4) 28:(3,3) 30:(4,4)  interval_resolved=[28]
                   VALIDITY_CHECK: PASS   CLI exit=0
```

The K=28 drop R1 demonstrated is gone; the three controls are unchanged.

### 1b. Full 3375-state enumeration — **through the REAL emitter**

The reviser's `h1_rev1_band_disclosure.py` re-implements
`build_report`'s `band_dict`/`trig_info` construction as a **mirror**. A
mirror can drift from the emitter, so I did not rely on it. My
`r2_a_realpath_3375.py` drives
`orchestrator.evaluate_trigger_and_band()` → `orchestrator.build_report()`
→ `validity_check.validity_check_core()` unmodified, stubbing only the one
non-pure dependency (`harvest_bridge.per_K_resolution`, the disk read):

```
[R2-A] enumerated 3375 (n_completed, n_converged) states through the REAL
       evaluate_trigger_and_band + build_report + validity_check_core
[R2-A] (1) reports failing their OWN validity_check: 1
        n_completed=[0, 0, 0] n_converged=[0, 0, 0] band=INCOMPLETE-AT-K
        interval_resolved=[] incomplete_at_K=[26, 28, 30]
            -> COMPLETE/otherwise-K2: 0 COMPLETED primary pairs
[R2-A] (2) SHARP-INVARIANT violations (named != {K : n_completed<4}): 0
```

**Exactly 1 failing state, and it is O1.** The failing reason string is
`COMPLETE/otherwise-K2`, i.e. §R8 K2's unconditional `>=1 COMPLETED
primary pair` clause firing on `(0,0,0)` — verbatim
`BUILD_AUDIT_R1.md`'s own O1 description (every primary cell crashes
twice ⇒ 0 canonical, no budget refusal ⇒ `COMPLETE` by G4's own
definition ⇒ rejected by an unconditional design clause). It is
design-inherited: removing it means editing
`NCR_KWALL_CHARACTERIZATION_DESIGN.md`, out of scope for a build stage.
**Not a build defect.** The reviser's own sweep independently reports the
same 1/3375 with the same reason.

### 1c. Adversarial hunt for a disclosure-dropping state the fix shape could miss

I did not settle for "validity_check passes." The COMPLETE branch's per-K
clause is satisfiable **iff**

> `named := set(interval_resolved_Ks) | set(incomplete_at_K)` equals
> exactly `{K : n_completed[K] < 4}`

(K in `named` ⇒ count `<4`; K not in `named` ⇒ count `==4`; and the STRICT
arm is the `named == ∅` case requiring all-4). That is necessary AND
sufficient, and strictly stronger than "the check passed" — it cannot be
satisfied vacuously by a state whose other clauses mask the per-K one.
**0 violations across all 3375 states**, i.e. no disclosure-dropping state
exists anywhere in the reachable space. Path-by-path, this holds by
construction:

| `classify_with_interval_logic` path | `named` | `{K : n_c<4}` |
|---|---|---|
| any-UNRESOLVED early return (**the F1 site**) | `sorted(UNRESOLVED ∪ AMBIGUOUS)` | identical |
| no UNRESOLVED, no AMBIGUOUS | `∅` (STRICT) | `∅` (all n_c==4) |
| AMBIGUOUS, cross-product AGREES | `interval_resolved_Ks = ambiguous_Ks` | identical |
| AMBIGUOUS, cross-product DISAGREES | `incomplete_at_K = ambiguous_Ks` | identical |

The specific shapes the task named were probed and are all inside that
sweep: mixed-AMBIGUOUS-only K's (e.g. `(3,3,3)`, both agree and disagree
sub-cases), and every boundary `n_completed ∈ {0,1,2,3,4}` — `resolution_state`
maps `≤2 → UNRESOLVED`, `3 → AMBIGUOUS`, `4 → EXACT`, so the 5×5×5 ×
`n_converged` enumeration is the complete reachable space. I also confirmed
by reading `validity_check_core` that the per-K disclosure clause exists
**only** on the `COMPLETE` branch — `COMPLETE-DEGRADED` and the two
`EXHAUSTED-BUDGET` branches carry no per-K clause at all — so no other
`run_status` can be dropped into by this fix.

---

## 2. M1 (MAJOR) — DISCHARGED

`r2_b_realpath_diag_1000.py` drives the REAL consumer
`orchestrator.evaluate_trigger_and_band` (not a re-implementation of its
`trig_info` construction) against a reference I wrote from the design's
pseudocode, whose **G5 band precondition is also re-implemented
independently** (the reviser's reference calls
`classify_with_interval_logic` for that half; mine does not, so the only
thing my reference shares with the build is `classify()` itself — which is
pinned by its own import-time 125-outcome checksum against the design's
printed table).

```
[R2-B] reference PRE-G5  : DECIDED=844 TRIGGER-UNRESOLVED=156  (design 844/156) -> MATCH
[R2-B] reference POST-G5 : DECIDED=473 TRIGGER-UNRESOLVED=527  (design 473/527) -> MATCH
[R2-B] (K_trig, resolution) divergences REAL-orchestrator vs reference: 0 / 1000
[R2-B] (blocking_K, band_blocked_K_trig) divergences:                   0 / 1000
[R2-B] R1's named example states=(0,0,'UNRESOLVED') via REAL orchestrator:
       blocking_K=None band_blocked_K_trig=26   (design requires None / 26)
```

Both design figures reproduce; the field R1 found wrong in 115/1000
vectors is now right in 1000/1000, measured on the production path.

**Shared `_compute_K_trigs` — proven, not asserted.** One definition in
`classify.py`; all three consumers (`trigger`, `trigger_raw_scan_blocked`,
`trigger_candidate_set`) call it. Mutation test — patch the shared symbol
to `lambda a,b,c: (None, 999)` and re-probe a vector that normally decides
cleanly:

```
trigger(4,4,4)                  -> (None,'TRIGGER-UNRESOLVED',None,999)  (unpatched: (32,'unanimous',None,None))
trigger_raw_scan_blocked(4,4,4) -> True                                  (unpatched: False)
trigger_candidate_set(4,4,4)    -> None                                  (unpatched: None)
```

All three flipped together ⇒ single source of truth; there is no second
copy of the scan that can drift.

**Revert teeth.** `h2_teeth_revert.py` (reconstructs the pre-fix scan
inline) re-runs to `115 / 1000` — the diagnostic correctly flags the OLD
code as broken before its 0/1000 verdict on the new code is trusted.

---

## 3. M2 (MAJOR) — DISCHARGED

**(a) Shell-executed fields are the only ones that matter, and they are
clean.** `queue_worker.sh:157` (`bash -c "$cmd"`) and `:162`
(`bash -c "$vcheck"`) are the only re-evaluating contexts; `output_dir` is
consumed as `mkdir -p "$outdir"`, a quoted variable expansion the shell
never re-expands (and it contains no `$(` anyway).

```
cmd             : 0 occurrences of $( or backtick  -> CLEAN
validity_check  : 0 occurrences of $( or backtick  -> CLEAN
output_dir      : 0
notes  (PROSE, never executed): 1 occurrence
${...} parameter expansions in cmd: ['${CUDA_VISIBLE_DEVICES:-0}']
```

The lone `$(` is inside `notes`, prose citing the removed pattern by name.
Cross-check against the 366 live pending specs: **0/366** use `$(` in
either shell-executed field — this spec is now consistent with the fleet.

**(b) The stamping step, executed verbatim** (`BUILD_REPORT` REV-1 step 3b,
run as written):

```
exit=0 ; stamped file parses as JSON: YES
remaining __DEPLOY_GIT_COMMIT__ anywhere: 0
stamped cmd carries the real HEAD d918074bfd99...: True
stamped cmd / validity_check: 0 $( or backtick -> CLEAN
```

**(c) The stamped `cmd` through a real `bash` word-splitting pass, then the
REAL `orchestrator.parse_args`:**

```
CUDA_VISIBLE_DEVICES=3      -> 26 argv tokens, parse_args OK, gpu_id=3
CUDA_VISIBLE_DEVICES unset  -> 26 argv tokens, parse_args OK, gpu_id=0
git_commit='d918074bfd99ca2979231924028612f96c670cf6'
steps_primary=80000  steps_conditional=160000
primary_outdir=/home/nvidia/ncr/results_kwall_characterization
cell_runner=/home/nvidia/ncr/ncr_earlyln_scale.py  python=/home/nvidia/tdenv/bin/python3
```

**(d) Teeth, both directions, live:**

```
OLD $(git rev-parse HEAD)  exit=2  fatal: not a git repository ...
                                   toy.py: error: argument --git-commit: expected one argument
NEW stamped literal        exit=0  Namespace(gpu_id='0', git_commit='d918074...')
```

The placeholder and its stamping step are documented in the deploy
sequence (`BUILD_REPORT.md` REV-1 step 3b) and re-stated in the spec's own
`notes` precondition (2b). See **r2-m2** below for one cosmetic side effect
of the `sed`.

---

## 4. M3 + M4 (MAJOR) — DISCHARGED

`r2_d_m3_m4.py`, written independently of
`tests/test_orchestrator_integration.py`: **27 checks, 0 failures.**

**M3(a) — the self-check has teeth.**

```
stop_file_path=None      : raised=True   (want True)
nonexistent path         : raised=True   (want True)
a REAL sentinel on disk  : raised=False  (want False)
interpreter assertions ENABLED (sys.flags.optimize=0); the box `cmd` passes
no -O and sets no PYTHONOPTIMIZE, so `assert` is live there too
```

**M3(b) — it PREVENTS a false artifact, which is the point of relocating it
out of `validity_check`.** With the sentinel deleted between detection and
the report build (the real operator race), through the REAL `run()`:
`AssertionError` raised, and **no `orchestrator_report.json` was written to
disk**. This is the design's `:1758-1763` requirement demonstrated, not
merely present.

**M3(c) — conditional sentinel path.** A real 12-cell sweep to a decided
`K_trig=26`, stopped inside the conditional loop:
`stop_file_path == args.stop_file_conditional` and `!= args.stop_file`
(the pre-fix bug), with `trigger.K_trig == 26` confirming the arm genuinely
launched.

**M3(d) — `per_K` is real.** Primary-loop stop after one completed cell:

```
per_K = {26:{n_completed:1, n_converged:1}, 28:{0,0}, 30:{0,0}}
canonical files on disk = ['earlyln_K26_s0.json']
```

Matches disk exactly; the pre-fix fabricated literal
`{26:(0,0),28:(0,0),30:(0,0)}` would have read all-zero here — teeth
confirmed by construction.

**M4 — the three schema fields, against the design's own table.** I re-typed
the design's printed KW4.5 11-configuration table (`:701-713`) into the
harness and re-executed it:

- **all 11 rows reproduce EXACTLY** — `K_trig` tie-break, `resolution ==
  "tie-break-min"`, `candidate_set` (`[26,28]` rows 1-9, `[28,30]` rows
  10-11), band label + non-monotone tag, `candidate_bands is None` (the
  band decides on these rows), and `resolution_detail` as the **bare
  literal** `"candidates were [26, 28]"` the design's `:2644` payload
  requires. Mismatches: **0**.
- the design's own 300-config sweep, re-executed independently: **exactly
  11** band-agreeing configs with >1 `K_trig` candidate; every one has
  `r_known == 2`; max distinct candidates = **2**, never 3+ — all three
  design-stated properties reproduce.
- **91** singly-AMBIGUOUS disagreeing configs found independently; **every
  one** populates `candidate_bands` with ≥2 labels (never the pre-fix
  `None`), and `candidate_bands` equals the independently recomputed
  cross-product label set in **all 91**. Example:
  `states=(2, ('AMBIGUOUS',0), 1) → incomplete_at_K=[28],
  candidate_bands=['GRADUAL-DECAY','NON-MONOTONE-UNRESOLVED']`.
- **`conditional.per_seed`, end-to-end through the real `run()`**, in both
  directions: a throttled 2/4 arm ⇒ `qualifier_band=None` and `per_seed`
  discloses exactly the two completed cells with their raw `gate1` records
  (§5 `:3104-3107`); a full 4/4 arm ⇒ `qualifier_band=
  SLOW-CONVERGENCE-AT-160K` and `per_seed=[]` (the headline is not
  withheld, so no substitute disclosure).

**Both conditional reports pass their own `validity_check` CLI (exit 0).**
This exercises U7 clause **(a)** (`launched=True`, 4 canonical, 4 ledger
COMPLETED) and the U7-mirror throttled branch end-to-end — neither was
reached by R1, whose only conditional case was clause (b) (`K_trig=32`,
`launched=False`).

---

## 5. Minors m1–m8 — all confirmed at their named sites

| # | Site | Verified |
|---|---|---|
| m1 | `tests/test_orchestrator_integration.py:333` | `K_seeds = [(K,s) for K in (26,28,30) for s in range(4)]` — the only 12 valid pairs; 11 × [`CRASHED-RECOVERED@1.20` + `COMPLETED`] + 1 × [`CRASHED-RECOVERED@1.20` + `GATE-REFUSED@0.0`]; suite asserts `ccgh == 12×1.20 = 14.40`. Producible in keys AND numbers. |
| m2 | `reconstruction.py:75-87` + `disk_io.py:94-95` | Both layers present: `isinstance(elapsed_s,(int,float))` guard bootstrapping to `CRASHED-RECOVERED@ceiling`, and a READ-boundary `"unparseable"` classification. Suite reports no crash at either layer and confirms the pre-fix path DOES `TypeError` on the identical input. |
| m3 | `orchestrator.py:226-238` | Canonical-filename fallback (`re.match(r"earlyln_K(\d+)_s\d+\.json$")`) after the attempt-dir scan; K inferred as 28 from canonical evidence alone, `None` still returned on a genuinely empty dir. |
| m4 | `BUILD_REPORT.md:460-500` | The "Corrected deploy sequence" now exists, with real `sed`/`scp` commands — the dangling self-reference is discharged. |
| m5 | `BUILD_REPORT.md:434-446` | Wording corrected to R1's executed finding (two `elapsed_h=0.0` `GATE-REFUSED` rows, no subprocess, no GPU-h); §4 left intact as historical record, which is the right call. |
| m6 | `smokes/MICRO_SMOKE_SPEC.md:44-76` | `--stop-file` removed (byte-for-byte the design `:2861-2864` command); the D5 training-loop-only caveat added to the budget paragraph, with the instrument sequence named. |
| m7 | `validity_check.py:256-261` | `os.path.exists` pre-check; suite confirms exit 1 (routing unchanged), clean diagnosis on stderr, zero `Traceback`. |
| m8 | deploy sequence step 5 | **Legitimately a deploy-stage step, and it does appear** — `~/queue/{fallback_pool,claimed}` sweep for K∈{26,28,30}, needs box access neither the build nor this audit takes. Repo half reconfirmed independently: **0 of 366** `queue/jobs/pending/*.json` reference `--K 26/28/30`, and the kwall spec is NOT in that tree. |

---

## 6. Suites from clean state, and scope discipline

| Suite | Result |
|---|---|
| `tests/test_reconstruction.py` | **PASS 5/5** — 24-state totality; OLD guard 30/200 orphans + 6/200 abort-trips; NEW guard 0/200 + 0/200; m2 fix + teeth |
| `tests/test_validity_check_suite.py` | **PASS 33/33**, 0 divergences vs the unmodified `kwall_suites/` in both NEW and OLD modes |
| `tests/test_orchestrator_integration.py` | **PASS 15 tests / 80 individual checks / 0 failures** (counted from the run, not from the report) |

`git status --short matrix-thinking/kwall_suites/` is empty — the reference
suite is committed, clean, and untouched by Rev-1.

**Byte-for-byte port re-verified, not carried over from R1.**
`inspect.getsource` diff of `r10_vcheck.validity_check` (132 lines) vs
`kwall_lib.validity_check_core` (135 lines) yields **exactly 1 unified-diff
hunk** — the signature line plus an added docstring — and all six module
constants (`BAND_LABELS`, `RESOLUTIONS`, `ACCEPT`, `EPS`, `PRIMARY_CEIL`,
`COND_CEIL`) compare equal. Rev-1's `validity_check.py` change is confined
to `main()`, outside the ported core (confirmed by reading the full diff).

**Scope discipline.** `git diff e155a6c d918074` touches 11 files under
`matrix-thinking/kwall_build/` plus `STATE.md` and `EXPERIMENT_LOG.md` —
the latter two being the coordinator's own round-#13 bookkeeping rows, not
build edits (read and confirmed). Nothing else in the repo moved; the
working tree is clean apart from pre-existing untracked LaTeX artifacts. I
read the complete Rev-1 diffs of `disk_io.py`, `harvest_bridge.py`,
`reconstruction.py`, `validity_check.py`, `classify.py` and
`orchestrator.py`: **every changed line is attributable to a named
finding**; nothing unrelated rode along.

---

## 7. New findings (all minor, all in the deploy sequence or in a
non-accept-set report)

### r2-m1 (minor) — deploy-sequence step 3 as written copies nothing

`BUILD_REPORT` REV-1 step 3 (inherited verbatim from R1's step 3) says
`bash matrix-thinking/queue/deploy.sh`. **`DRY_RUN=1` is that script's
default** (`deploy.sh:182`), and the `DRY_RUN` gate at `:421-434` prints
the ship/skip plan and `exit 0`s **before** the static-file block at
`:436` that actually stages, md5-verifies and atomically renames
`ncr_task.py` / `ncr_earlyln_scale.py` into `~/ncr/`. Run as written, step
3 exits 0 having deployed neither recipe patch.

*Blast radius is bounded and self-correcting:* the omission first surfaces
at step 6, where `ncr_earlyln_scale.py --K 26` is rejected by
`--K choices=sorted(GRID_SHAPES)` and no smoke JSON is written, so
`check_smoke` reads FAIL and pool promotion (step 7) stays gated. Cost ≈ 0
GPU-h — one wasted deploy cycle, not the wave. Not a code defect. Resolved
in the sequence below (step 3 uses the narrow, zero-side-effect
stage→md5→atomic-rename path).

*Second-order caution, also resolved below:* `DRY_RUN=0 bash deploy.sh`
would ALSO sync every local `queue/jobs/pending/*.json` whose basename is
absent from the box straight into `~/queue/pending/`, where a worker
claims it within ~60 s. That is deploy.sh's normal job, but it is not
something this wave's deploy should trigger blind.

### r2-m2 (minor) — the step-3b `sed` also rewrites the `notes` prose

`__DEPLOY_GIT_COMMIT__` appears twice in the spec: once in `cmd`, once in
the `notes` precondition (2b) sentence that names the token. The
whole-file `sed` (no `g` flag, but one occurrence per line) substitutes
both, so the deployed spec's own precondition reads "…still carries the
literal placeholder token `d918074…` in place of a real commit hash" —
self-contradicting prose. Never shell-executed, so zero functional
impact; and the whole-file `sed` is precisely what makes step 3b's own
stated "ZERO occurrences of `__DEPLOY_GIT_COMMIT__`" check pass. Keep the
`sed`; treat the stamped `notes` as historical. (A `cmd`-only stamp would
need step 3b's acceptance criterion re-scoped to `cmd`.)

### r2-m3 (minor) — a primary-loop operator stop reports `incomplete_at_K: []`

`orchestrator.py:525` passes the hardcoded band tuple
`(None, False, "INCOMPLETE-AT-K", [], None)` on a primary-loop stop, so the
emitted report carries:

```
band  : {'label':'INCOMPLETE-AT-K', 'interval_resolved_Ks':[], 'incomplete_at_K':[], ...}
per_K : {'26':{n_completed:1,...}, '28':{0,...}, '30':{0,...}}
```

A consumer reading `incomplete_at_K` sees "no K is incomplete" under an
`INCOMPLETE-AT-K` label, while the adjacent `primary.per_K` — the field
M3's own fix made real — says otherwise. Same class as the M3 defect R1
named ("the design put a named enforcement point here precisely so a false
artifact never exists"), at strictly lower impact: `STOPPED-BY-OPERATOR` is
excluded from `validity_check`'s accept-set by universal assertion 1, which
returns before any band clause is read, so nothing is mis-routed and the
truth is present one field away. The **conditional**-loop stop path is
unaffected (it passes the real evaluated band). One line to fix
(`sorted(K for K in (26,28,30) if real_resolution[K][0] < 4)`); recorded,
not blocking, coordinator's call whether to take it before or after deploy.

---

## 8. Observations (recorded so they are not re-found)

- **O-R2-1.** The reviser's `h1_rev1_band_disclosure.py` mirror is stale
  w.r.t. its own M1 fix: its `trig_info` construction (lines 39-40) still
  uses the pre-fix `any_unresolved` heuristic rather than
  `trigger_raw_scan_blocked`. Harmless for F1's conclusion —
  `validity_check_core` reads none of `candidate_set`, `blocking_K`,
  `band_blocked_K_trig`, `candidate_bands`, `conditional.per_seed`
  (verified by reading the function) — but it means that sweep did not
  exercise the real emitter. `r2_a_realpath_3375.py` does, and agrees to
  the digit.
- **O-R2-2.** `BUILD_REPORT` REV-1 says "10 files, 0 touched outside this
  directory"; the commit's diff shows 11 under `kwall_build/` (it does not
  count itself) plus the coordinator's `STATE.md`/`EXPERIMENT_LOG.md` rows.
  Both claims are true in substance; the count is off by one.
- **O-R2-3 (operational, not a defect).** `~/queue/fallback_pool/` is the
  *starvation* path, not a run queue. `idle_fallback_daemon.sh` promotes
  from it only when **all** of: `~/queue/idle_launcher.DONE` exists
  (`:43`), no `~/queue/idle_launcher.HOLD`, 37 consecutive 5-min samples
  with zero compute apps box-wide (≥3 h), and `pending` + `claimed` both
  empty (`:63-70`). With the box saturated, a spec placed there will sit as
  runway indefinitely — which is what §6 pool-eligibility intends. If the
  coordinator wants this wave to RUN rather than bank runway, the same
  stamped spec goes to `~/queue/pending/` after the identical gates (a
  worker claims it within ~60 s on a GPU showing zero compute apps).
- **O1/O2/O3 (R1's design-inherited observations)** are correctly
  dispositioned NOT FIXED by Rev-1, each with R1's own reasoning quoted.
  O1's residual is the single `(0,0,0)` state in §1b and reads exactly the
  clause R1 predicted. O2's dead `PERSISTENTLY-ABORTED` half remains inside
  the untouched byte-for-byte port, as it must for the 33/33 identity to
  hold. O3's procedure-vs-narrative gap is in the design doc, not the code.

---

## 9. Verified deploy sequence — execute in this order

Every path, binary and command below was checked against
`matrix-thinking/H100_SETUP.md`, `HANDOFF_BOX_ACCESS.md`,
`matrix-thinking/queue/{deploy.sh,queue_worker.sh,idle_fallback_daemon.sh}`
and the 366 live specs. Box facts confirmed: SSH alias
`youthful-indigo-turkey` (non-interactive, `~/.brev/ssh_config`); default
user `nvidia`, home `/home/nvidia`; venv `/home/nvidia/tdenv/bin/python3`
(**366/366** live specs use it); code root `/home/nvidia/ncr` (**111**
specs `cd` there; `deploy.sh:516` publishes the two recipe files there);
queue root `$HOME/queue` with `pending/ claimed/ completed/ failed/
fallback_pool/`. This is paper verification — no box contact was made.

**Step 1. Land the two additive recipe patches on the box** (replaces
Rev-1 step 3; see **r2-m1**). Use the narrow stage → md5 → atomic-rename
path — it is deploy.sh's own verified pattern with none of its job-spec
side effects, and the atomic rename is load-bearing
(`ncr_earlyln_scale.py` is imported by every queued job's `cmd`; a direct
`scp` truncates the live inode under a mid-read worker):

```bash
cd /Users/samuellarson/Experiments/learned-representations/matrix-thinking/ncr
ssh youthful-indigo-turkey 'mkdir -p ~/ncr/.kwall_stage'
scp ncr_task.py ncr_earlyln_scale.py youthful-indigo-turkey:~/ncr/.kwall_stage/
ssh youthful-indigo-turkey 'cd ~/ncr/.kwall_stage && md5sum ncr_task.py ncr_earlyln_scale.py'
#   MUST equal, in this order:
#     ncr_task.py           6e461d6e598b857c67020ccc9c884aef
#     ncr_earlyln_scale.py  dc50c82d15eb6c00681aca81f2f31c25
#   (verified locally this round; abort if either differs)
ssh youthful-indigo-turkey 'mv -f ~/ncr/.kwall_stage/ncr_task.py ~/ncr/ncr_task.py \
  && mv -f ~/ncr/.kwall_stage/ncr_earlyln_scale.py ~/ncr/ncr_earlyln_scale.py \
  && rmdir ~/ncr/.kwall_stage'
```

*(If you prefer `deploy.sh` instead: run the default `bash deploy.sh` FIRST
to read its ship/skip plan, require the ship list to be EMPTY, then re-run
as `DRY_RUN=0 bash deploy.sh`. Never `DRY_RUN=0` without that check —
r2-m1.)*

Patch content confirmed present locally and unchanged since `4d9a6b9`:
`GRID_SHAPES[26]=dict(d=52,h=64)`, `[28]=dict(d=56,h=64)`,
`[30]=dict(d=60,h=64)` (`ncr_earlyln_scale.py:101-103`) and
`for _K_new in (26, 28, 30): GRIDS[_K_new] = _gen_grid(_K_new)`
(`ncr_task.py:173-174`).

**Step 2. Deploy the orchestrator + its package.** `deploy.sh` does NOT
ship these — this step is genuinely manual. Nine files, no `__pycache__`
present locally (verified):

```bash
cd /Users/samuellarson/Experiments/learned-representations/matrix-thinking/kwall_build
ssh youthful-indigo-turkey 'ls -la ~/ncr/orchestrator.py ~/ncr/kwall_lib 2>&1 | head'
#   expect "No such file" for both -- if either EXISTS, stop and adjudicate
#   before clobbering someone else's file
scp orchestrator.py youthful-indigo-turkey:~/ncr/
scp -r kwall_lib youthful-indigo-turkey:~/ncr/
ssh youthful-indigo-turkey 'ls ~/ncr/kwall_lib/__init__.py && cd ~/ncr \
  && /home/nvidia/tdenv/bin/python3 -c "import orchestrator, kwall_lib.validity_check; print(\"import OK\")"'
```

`kwall_lib/__init__.py` must be present for `-m kwall_lib.validity_check`
to resolve from the `cd` target. The import probe is safe and free: no
`kwall_lib` module imports `torch`/`numpy`/`ncr_*` at module level
(verified), and `classify.py`'s import-time 125-outcome and 473/527
self-checks run in milliseconds — if either assert fires, stop.

**Step 3. Discharge m8 — the on-box K∈{26,28,30} sweep** (design §6
red-team item (v); repo half already clean, 0/366):

```bash
ssh youthful-indigo-turkey 'grep -l "\-\-K 26\|\-\-K 28\|\-\-K 30" \
  ~/queue/fallback_pool/*.json ~/queue/claimed/*.json 2>/dev/null; echo "sweep done"'
```
Expect no filenames. Any hit must be adjudicated before step 5.

**Step 4. The three micro-smokes — WHERE THEY RUN and what they cost.** On
a GPU verified idle by `nvidia-smi` (NOT covered by `queue_worker.sh`'s
free-GPU gate — the design makes this a manual pre-launch check), from
`smokes/MICRO_SMOKE_SPEC.md` verbatim (m6-corrected: no `--stop-file`):

```bash
ssh youthful-indigo-turkey 'nvidia-smi --query-compute-apps=pid,used_gpu_memory \
  --format=csv,noheader'      # pick a GPU index with NOTHING on it
ssh youthful-indigo-turkey 'cd /home/nvidia/ncr && CUDA_VISIBLE_DEVICES=<IDX> bash -lc "
for K in 26 28 30; do
  D=\$((K + 1))
  /home/nvidia/tdenv/bin/python3 ncr_earlyln_scale.py --cell \
    --K \"\$K\" --d-override \"\$D\" --seed 0 --steps 500 --ceiling-gpuh 0.05 \
    --outdir \"/home/nvidia/ncr/results_kwall_smoke/K\${K}\"
done"'
```

Budget **≤0.15 GPU-h, a TRAINING-LOOP-ONLY bound** (m6's own caveat:
`--ceiling-gpuh` gates `train_earlyln_cell`'s loop, never the post-train
instrument sequence). These run OUTSIDE the orchestrator's 15.50 h ledger
and outside any pool spec. Then the spec's own verification snippet:

```bash
ssh youthful-indigo-turkey 'cd /home/nvidia/ncr && /home/nvidia/tdenv/bin/python3 -c "
import sys; sys.path.insert(0, \"/home/nvidia/ncr\")
import orchestrator as orch
smoke = orch.check_smoke(\"/home/nvidia/ncr/results_kwall_smoke\", (26, 28, 30))
print(smoke); assert all(v == \"PASS\" for v in smoke.values()), smoke
print(\"ALL 3 MICRO-SMOKES PASS\")"'
```

Step 2 must precede this (the snippet imports `orchestrator`). Step 1 must
precede the three runs (`--K 26` is not a valid choice without the patch —
this is exactly where an r2-m1 miss would surface).

**Step 5. Stamp the spec, then promote — the gate on pool promotion.**

```bash
cd /Users/samuellarson/Experiments/learned-representations/matrix-thinking/kwall_build
sed "s/__DEPLOY_GIT_COMMIT__/$(git rev-parse HEAD)/" \
  job_specs/kwall_char_orchestrator.json > /tmp/kwall_char_orchestrator.stamped.json
grep -c '__DEPLOY_GIT_COMMIT__' /tmp/kwall_char_orchestrator.stamped.json   # MUST be 0
python3 -c "import json,re;s=json.load(open('/tmp/kwall_char_orchestrator.stamped.json'));
print('dirty' if any(re.search(r'\\\$\\(|\`',s[k]) for k in ('cmd','validity_check')) else 'CLEAN')"
scp /tmp/kwall_char_orchestrator.stamped.json \
  youthful-indigo-turkey:~/queue/fallback_pool/kwall_char_orchestrator.json
```

**Pool promotion is gated on ALL of:** (i) steps 1-2 landed and the import
probe passed; (ii) step 3's sweep returned nothing; (iii) step 4 printed
`ALL 3 MICRO-SMOKES PASS`; (iv) the stamped spec shows 0 placeholders and
`CLEAN`. Never `scp` the placeholder-carrying original — the box's `~/ncr`
is not a git working tree (`deploy.sh` contains no `git clone`/`git init`),
which is exactly what M2 closes.

See **O-R2-3** before choosing `fallback_pool/` vs `pending/`: the former
is runway that fires only after ≥3 h of box-wide idle with an empty queue;
the latter runs within ~60 s. Both are legitimate; the choice is the
coordinator's and should be made deliberately, not by inheriting R1's
wording.

---

**Counts: 0 FATAL, 0 MAJOR, 3 minor, 3 observations (+ R1's O1-O3 re-confirmed).**
The build is **DEPLOY-RELEASED**. Report path:
`matrix-thinking/kwall_build/BUILD_AUDIT_R2.md`. This audit modified
nothing else — no build files, no `STATE.md`, no `EXPERIMENT_LOG.md`, no
box contact.
