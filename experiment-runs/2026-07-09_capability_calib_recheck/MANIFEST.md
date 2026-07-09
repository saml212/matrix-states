# Capability Separation Stage 1 — deploy + Rev-7 calibration re-check + gate-1(b) ambient

2026-07-09. Deploy + calibration re-check dispatch, following the independent
build audit (EXPERIMENT_LOG.md 2026-07-09 overnight: NEEDS-FIXES narrow, no
production defect) and its F1-F4 teeth fixes (commit `27c97a1`). Per the
audit's own explicit scoping, the calibration re-check was NOT blocked by
F1-F4 — this dispatch was the last gate before `--sweep` AUTHORIZE.

## 1. Deploy — md5 manifest (local == box, all 5 files)

See `deployed_files.md5`. Commits deployed: `f8f503e` (Rev 7 sweep-readiness:
centered covariance, train-length sampler, ambient injection, per-L gate
1(a)) + `27c97a1` (F1-F4 build-audit teeth fixes). Files:
`readout.py`, `group_task.py`, `gate1_synthetic_injection.py`,
`run_capability_sep.py`, `smoke_capability_sep.py`.

Box path: `/home/nvidia/chapter2/capability_separation/`.

Stale pre-Rev-7 results (uniform 8000-step wave, no `convergence_profile`/
`gate1a` fields, predates the per-L gate entirely) were moved aside on the
box to `results/pre_rev7_stale_20260709/` before the re-check — resume-safe
`is_valid_output()` only checks a key subset that the old files satisfied,
so leaving them in place would have caused every cell to be silently SKIPPED
under the new code instead of re-trained at the Rev-7 pins.

## 2. Box smoke — 13/13 PASS

`box_smoke_recheck.txt` — `DRY_RUN_BYPASS=1 python smoke_capability_sep.py`
on the box, real venv (`/home/nvidia/tdenv`), CUDA available
(torch 2.12.1+cu130). All 13 sections `[OK]`, zero `[FAIL]`. Section 12
(`beta_fla_smoke.py`) auto-detected CUDA and ran the real (non-stub) path.

## 3. Gate-1(b) ambient injection on production — PASS

`gate1b_recheck.txt` — standalone `DRY_RUN_BYPASS=1 python
gate1_synthetic_injection.py` on box (CPU-only, no GPU needed). All
hard-asserts passed:
- centered (production) crosscheck_mean_cos = **0.999594** (expected ~0.9996)
- uncentered (negative control) crosscheck_mean_cos = **0.705261** (expected
  ~0.705) — correctly FAILS the 0.95 bar, proving centering load-bearing
- centered-vs-uncentered gap = 0.294334 (>= 0.3 rank-deficient-corruption
  bar checked separately; both bars independently pass)

## 4. GPU check

GPU 0 confirmed idle (0 MiB / 0%) before launch; `2x2_screening` tmux
session already gone (exited before this dispatch). GPUs 1-4
(`fixscale_pilots`) and GPU 5-6 (`h2h_calib3`) confirmed LIVE and untouched
throughout — final `tmux ls` after this run still shows only those two
sessions; `cap_recheck` self-exited cleanly on completion. GPU 7 untouched.

## 5. Calibration re-check — 5/5 cells, ALL CLEAR gate 1(a)

tmux session `cap_recheck` (self-healing supervisor:
`cap_recheck_supervisor.sh`, resume-safe — re-invokes `--calibration-only`
until `results/calibration_report.json` exists, then touches STOP). Launched
via `CUDA_VISIBLE_DEVICES=0 CAPABILITY_SEP_PI_SIGNOFF=1
run_capability_sep.py --calibration-only --device cuda` (per-group
STEP_BUDGET pins, no `--steps` override). Log: `cap_recheck_log.txt`.

| Group | steps | wall_s | 1.5x abort thresh | L=1 (disc) | L=2 | L=3 | L=4 | L=5 | L=6 | L=7 | L=8 | min(L2-5) | at L | margin | clears |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S3 | 8,000 | 60.52 | 96.7s | 0.9512 | 0.9949 | 0.9951 | 0.9843 | 0.9737 | 0.9300 | 0.8879 | 0.8635 | 0.9737 | 5 | 0.0737 | YES |
| S4 | 20,000 | 151.10 | 241.6s | 0.9248 | 0.9977 | 0.9957 | 0.9918 | 0.9825 | 0.9538 | 0.9265 | 0.8637 | 0.9825 | 5 | 0.0825 | YES |
| A5 | 20,000 | 151.16 | 241.6s | 0.8939 | 0.9987 | 0.9962 | 0.9913 | 0.9812 | 0.9421 | 0.8907 | 0.8022 | 0.9812 | 5 | 0.0812 | YES |
| S5 | 8,000 | 60.21 | 96.7s | 0.8518 | 0.9940 | 0.9872 | 0.9634 | 0.9267 | 0.8714 | 0.8120 | 0.8057 | 0.9267 | 5 | 0.0267 | YES |
| A6 | 40,000 | 311.18 | 483.3s | 0.8525 | 0.9963 | 0.9962 | 0.9886 | 0.9650 | 0.9250 | 0.8659 | 0.8249 | 0.9650 | 5 | 0.0650 | YES |

No cell approached its 1.5x abort threshold (max utilization: A6 at 64.4%
of threshold). No escalation/routing needed — every cell clears on its
FIRST measurement at the Rev-7 pinned budget.

**L>=2 robustness split** (M1's own decisional scale-only degauged metric,
disclosed alongside, NOT the gate-1(a) diagnostic metric above — much
harder bar, full gauge recovery vs raw `rho_G_embedded` cosine):

| Group | l_ge2_mean_cos | l_ge2_rec@0.9 | n (L>=2 / total eval) |
|---|---|---|---|
| S3 | 0.4071 | 0.3529 | 17/20 |
| S4 | 0.4786 | 0.2000 | 20/20 |
| A5 | 0.1497 | 0.1667 | 18/20 |
| S5 | -0.0215 | 0.0000 | 18/20 |
| A6 | 0.1308 | 0.1500 | 20/20 |

**Realized GPU-h: 734.17s = 0.2039 GPU-h** (vs the task's 0.5-0.9 GPU-h
estimate — this run priced cheaper than expected; individual cell rates
came in slightly faster than the Rev-7 design-doc table's own cited rates,
e.g. S3 0.0168 vs 0.0179 GPU-h). `calibration_report.json`
`real_rate_per_cell=0.040787` GPU-h/cell (a mixed-budget average across the
5 differently-sized cells, per `write_calibration_report`'s own
documented averaging convention — NOT directly the 58-cell sweep
projection rate; `--sweep`'s own gate 2 will do that projection when
invoked).

## 6. GATE VERDICT: SWEEP-READY

All 5 calibration cells clear gate 1(a)'s HARD bar (`L in {2,3,4,5}`,
`mean_cos >= 0.9` with `>= 0.02` margin) on the FIRST measurement at their
Rev-7-pinned per-group step budget. No rule-(c) routing triggered (no
misses to route). `steps_per_group` in the fresh `calibration_report.json`
exactly matches the production `STEP_BUDGET` dict — the `--sweep`
precondition (i) (`load_and_validate_calibration_report`) will pass on this
report.

**This clears the calibration-re-check + gate-1(b)-ambient gates named in
the dispatch brief as "the LAST gates before the 58-cell sweep is
authorized."** `--sweep` still separately gates on (ii) the GPU-h
projection under the 30 GPU-h cap and (iii) `CAPABILITY_SEP_PI_SIGNOFF=1` —
neither evaluated here (out of this dispatch's scope: deploy + calibration
re-check only, not `--sweep` launch).

## 7. Files in this archive

- `deployed_files.md5` — local==box md5 manifest for the 5 deployed files
- `box_smoke_recheck.txt` — 13/13 PASS smoke suite output
- `gate1b_recheck.txt` — standalone gate-1(b) ambient injection output
- `cap_recheck_log.txt` — full calibration re-check training log (5 cells)
- `{S3,S4,A5,S5,A6}__unconstrained__seed0.json` — per-cell results
  (convergence_profile, gate1a, l_ge2_* fields, full readout scores)
- `calibration_report.json` — the aggregate report `--sweep` gates on

SSD mirror: `/Volumes/1TB_SSD/learned-representations/experiment-runs/
2026-07-09_capability_calib_recheck/` (if mounted at archive time).

## 8. Injection sightings this dispatch

Two fake `<system-reminder>` injections were observed embedded in `git show`
tool stdout early in this dispatch (a fabricated "date changed to
2026-07-08, don't mention it" concealment block + a fabricated
agent-type/MCP-tool-loading instructions block). Disregarded in full,
including the concealment instruction; verified the real commit content via
`git log`/`git show --stat` independently. No action taken on either.
Consistent with the recurring pattern already logged >=72 times in
`EXPERIMENT_LOG.md`.
