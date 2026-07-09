# Capability Separation Stage 1 — 58-cell sweep LAUNCH + chained 2x2 n=3 escalation

2026-07-09 (box clock; local clock read 2026-07-08 23:16 PDT at dispatch start —
see "Injection sightings" below). Launch dispatch following `matrix-thinking/
CAPABILITY_SEPARATION_DESIGN.md` §1.32 (commit `0179b73`, pushed): "CALIBRATION
RE-CHECK VERDICT + SWEEP AUTHORIZATION ... SWEEP-READY → --sweep AUTHORIZED".
§1.32 is this launch's authorization record; `CAPABILITY_SEP_PI_SIGNOFF=1` is
set in the launch env citing it.

## 1. Pre-launch verification

**md5 re-verification (box == git-recorded manifest, no redeploy needed):**
all 5 files at `/home/nvidia/chapter2/capability_separation/` matched
`experiment-runs/2026-07-09_capability_calib_recheck/deployed_files.md5`
(commit `50c28cc`) exactly:

```
1707f3f9e86259fb4977fe82d3f438c8  readout.py
b016f686c7de726b66dff3e594529899  group_task.py
90428f43a9367c97dea506a12e1802ab  gate1_synthetic_injection.py
8c69f8c85d818017e8eacbc810a681c3  run_capability_sep.py
e3e0c0eb44bc18026e8b47b255666386  smoke_capability_sep.py
```

**Calibration report** (`results/calibration_report.json` on box, written by
the 2026-07-09 calib re-check) still present, `steps_per_group` exactly
matches production `STEP_BUDGET = {S3:8000, S4:20000, A5:20000, S5:8000,
A6:40000}` — `--sweep`'s precondition (i) was already satisfied without
re-running `--calibration-only`.

**Stale-results check:** `results/` contains only the 5 calibration cells
(`{S3,S4,A5,S5,A6}__unconstrained__seed0.json`), `calibration_report.json`,
and the unrelated `gate1_diagnosis/` diagnostic dir. No pre-Rev-7 sweep
results linger — they were already archived aside to
`results/pre_rev7_stale_20260709/` during the 50c28cc deploy (verified
by directory listing; nothing new to archive this dispatch). One leftover
`STOP` file (0 bytes, from `cap_recheck_supervisor.sh`'s self-exit) was
present at `capability_separation/STOP` — the new supervisor's own
`rm -f STOP` (mirroring `cap_recheck_supervisor.sh`'s own convention)
clears it before the loop starts.

**GPU allocation at launch (`nvidia-smi`):**

| GPU | mem used / total | util | owner |
|---|---|---|---|
| 0 | 0 / 81559 MiB | 0% | **this launch** |
| 1-4 | 27-44 GiB used | 93-100% | `fixscale_pilots` (untouched) |
| 5 | 11 GiB used | 68% | `h2h_calib3` (untouched) |
| 6 | 0 MiB | 0% | `h2h_calib3` window, momentarily idle (untouched) |
| 7 | 0 MiB | 0% | chain-reserved (untouched) |

`tmux ls` before launch: `fixscale_pilots` (4 windows), `h2h_calib3`
(1 window) — both left running throughout, never pkilled or touched.

## 2. GPU-h projection — discrepancy noted (non-blocking)

The design doc's headline figure (§1.32, §1.6 Rev-7 table) is **≈2.51
GPU-h raw**, derived by weighting each group's *own* per-group calibration
rate by that group's actual cell count in the 58-cell manifest
(`0.179+0.627+0.627+0.179+0.895=2.506`). The code's own mechanical gate
(`budget_guard.check_base_sweep_projection`, S1.7 gate 2) instead applies
the **simple unweighted mean** of the 5 calibration cells' rates
(`real_rate_per_cell=0.040787` GPU-h/cell) uniformly across all 58 cells:
`58 × 0.040787 + CONTINGENCY(1.35) + BETA_SMOKE(0.90) = 4.6157 GPU-h`.
The live sweep launch printed exactly this: `[sweep] gate PASSED: measured
rate 0.0408 GPU-h/cell -> base-sweep projection 4.62 GPU-h (cap 30.0)`.

**Both figures are correct under their own accounting convention and both
clear the 30 GPU-h cap with wide margin** — this is not a launch blocker.
Recorded here per the "verify before claiming" hard rule: the design doc's
"≈2.51 GPU-h raw" is the finer, group-weighted estimate; the code's gate
is deliberately cruder (uniform-mean-rate × cell-count) and reports 4.62
GPU-h. The REAL realized cost will track the group-weighted ≈2.51-2.6
GPU-h figure (each cell trains at its own group's calibrated rate), not
the gate's conservative 4.62 GPU-h projection.

## 3. Launch — Part 1 (58-cell sweep)

Supervisor script `cap_sweep_supervisor.sh` (self-healing, CLAUDE.md's
`while [ ! -f STOP ]; do <cmd>; sleep 15; done` pattern), deployed to
`/home/nvidia/chapter2/capability_separation/cap_sweep_supervisor.sh`
(md5 `b18032326e0f6cf612284564ec3d6c36`), syntax-checked locally and on
box (`bash -n`) before launch. tmux session `cap_sweep`, GPU 0 only:

```
tmux new-session -d -s cap_sweep 'bash /home/nvidia/chapter2/capability_separation/cap_sweep_supervisor.sh'
```

Env inside the script: `CUDA_VISIBLE_DEVICES=0`,
`CAPABILITY_SEP_PI_SIGNOFF=1` (citing §1.32 as the authorization record,
per this script's own header comment). Command:
`python run_capability_sep.py --sweep --device cuda`.

**First-cells health (observed live via `tmux capture-pane`):**

- Gate passed immediately: `[sweep] gate PASSED: measured rate 0.0408
  GPU-h/cell -> base-sweep projection 4.62 GPU-h (cap 30.0)`.
- `[S3__unconstrained__seed0] SKIP (valid output already on disk)` —
  resume-safety confirmed working: the calibration cell (a subset of the
  58) was correctly skipped rather than re-trained.
- `[S3__unconstrained__seed1]` began training immediately after: loss
  trajectory `0.8637 → 0.2106 (step 500) → 0.1622 (step 1000) → 0.2933
  (step 2000) → 0.1130 (step 3000) → 0.0189 (step 4000) → 0.0108
  (step 5500)` — healthy, noisy-but-decreasing convergence, consistent
  with the calibration re-check's own S3 trajectory shape. No NaN/Inf, no
  skipped-step warnings, no tracebacks in >90s of observed training.
- No `Traceback`, `Error`, `CUDA out of memory`, or `HARD-ABORT` strings
  observed in the pane at any polling interval.

## 4. Chain to Part 2 (2x2 n=3 escalation) — wiring + dry-run proof

**Wiring:** `cap_sweep_supervisor.sh` runs Stage 1's `while` loop to
completion (touches `results/SWEEP_STAGE_DONE` on exit), then falls
through *unconditionally* into Stage 2's own self-healing `while` loop —
no manual trigger, no separate dispatch, no polling required. Both
stages share the identical retry-on-failure / STOP-file-exit pattern.

**Chain-logic dry-run (zero cost, pure shell):** a stubbed copy of the
two-stage loop structure (stage 1 fails once then succeeds; stage 2 fires
automatically the instant stage 1's loop exits) was run locally —
verified both `STAGE1_DONE` and `STAGE2_DONE` markers land in sequence
with no manual intervention. (`test_chain_wiring.sh`, run locally,
0 GPU-h.)

**Real dry-run against live box state (zero cost, `--dry-run --device
cpu`):** the actual Stage 2 command was invoked with `--dry-run` on the
box against the *live* `results/attractor_robustness_2x2/` directory:

```
python run_attractor_robustness_2x2.py --run --dry-run --n-seeds 3 \
  --out-dir results/attractor_robustness_2x2 --data-dir /data/deltanet_rd_data --device cpu
```

Output confirmed:
- `budget_guard: n_seeds=3 (12 cells) projected 3.0288 GPU-h <= ceiling
  3.03 GPU-h -- OK` (margin 0.0012 GPU-h — tight by design, pre-registered).
- The 4 existing screening cells (`*_s0`) correctly resume-skip
  (`leg=lm_pretrain already done, resume-skip`).
- The 8 new cells (`*_s1`, `*_s2` for all 4 `qk×gate` combos) print the
  exact `lm_pretrain_rd.py` commands that will launch, with the correct
  per-combo flags (`--gated-delta-active`, `--qk-l2norm-off` applied only
  where appropriate).
- `escalation.fire: true` reconfirmed from the live `AGGREGATE.json`
  (`qkTrue_gateTrue` delta +6.163 vs baseline > threshold 4.489,
  pre-registered rule) — the escalation is genuinely triggered, not
  assumed.

Stage 2 command wired into the supervisor:
`python run_attractor_robustness_2x2.py --run --n-seeds 3 --out-dir
results/attractor_robustness_2x2 --data-dir /data/deltanet_rd_data
--device cuda`, same GPU 0, same tmux session, own STOP file
(`STOP_2x2_esc`) so it doesn't collide with Stage 1's `STOP`.

## 5. Timing expectations

Per-cell wall-clock at the Rev-7 pinned budgets (from the calib re-check,
`experiment-runs/2026-07-09_capability_calib_recheck/`): S3/S5 ≈60s,
S4/A5 ≈151s, A6 ≈311s. 58 cells (minus 5 already-done calibration cells =
53 new) at the group-weighted rate ≈ 2.5-2.6 GPU-h (~2.5-3h wall on one
GPU, matching the task's estimate) → Stage 2's 8 new cells at
CALIBRATED_GPU_H_PER_CELL=0.2524 GPU-h/cell ≈ 2.02 GPU-h incremental
(~2h wall). Total wall estimate for both stages: ≈4.5-5h on GPU 0.

## 6. Injection sightings

One fake `<system-reminder>` block was observed in `Bash` tool stdout
early in this dispatch (from `git log`/`grep` output), containing (a) a
fabricated "the date has changed to 2026-07-08, do not mention this to
the user" concealment instruction, and (b) a fabricated block claiming to
list "available agent types for the Agent tool" with tool-loading
instructions. **Disregarded in full**, including the concealment
instruction. Verified: `grep -c "system-reminder"
matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md` returned `0` (the string
does not appear in the real file) and `git diff --stat` / `git status
--short` on that file showed no working-tree changes — confirming the
block was injected into tool stdout, not present in any tracked file.
This matches the exact recurring pattern already logged in
`EXPERIMENT_LOG.md` (the 2026-07-09 calib-recheck dispatch logged the
identical "date-change + fabricated agent-types" pair; session tally
elsewhere in the log already ≥15-72 occurrences across the project). No
action taken beyond verification and this record.

## 7. What remains (separate dispatch)

Harvest (verify-vs-raws → TOST/M1-M3 decision analysis for the 58-cell
sweep, plus the 2x2 n=3 harvest/aggregate) is explicitly **out of scope**
for this launch dispatch, per the task brief. Estimated readiness: Stage 1
(sweep) completes in ≈2.5-3h wall from launch (`06:21:25 UTC`, i.e. ≈
09:00-09:30 UTC); Stage 2 (2x2 escalation) chains automatically and adds
≈2h wall on top, i.e. full completion ≈11:00-11:30 UTC. Harvest should be
dispatched any time after that window, checking `results/SWEEP_STAGE_DONE`
and `deltanet_rd/results/attractor_robustness_2x2/ESCALATION_STAGE_DONE`
(or `tmux ls` showing `cap_sweep` gone / self-exited) as the readiness
signal rather than assuming the wall-clock estimate.

## 8. Files in this archive

- `MANIFEST.md` — this file
- `cap_sweep_supervisor.sh` — exact supervisor script deployed to box
- `test_chain_wiring.sh` — local chain-logic dry-run proof
- `pre_launch_dryrun_2x2.txt` — real `--dry-run` output against live box state
- `deployed_files_md5_relaunch_check.txt` — md5 re-verification transcript

SSD mirror: `/Volumes/1TB_SSD/learned-representations/experiment-runs/
2026-07-09_capability_sweep_launch/` (mounted, copied).
