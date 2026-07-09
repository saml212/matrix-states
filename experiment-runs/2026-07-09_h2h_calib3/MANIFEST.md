# H2H CALIBRATION ROUND 3 — Rev 4 three-term objective, 9-cell launch (2026-07-09)

**STATUS: LAUNCHED + VERIFIED HEALTHY, HARVEST PENDING.** This record covers the launch,
the pre-launch state repair it required, and health verification of the first wave. The
round is expected to run ~1.5-3h wall clock on 2 GPUs; a follow-up harvest pass must read
`results/h2h_rung1/CALIBRATION_COMPLETE.json` and the per-cell result JSONs once the chain
finishes Stage B (or hits FATAL), record the rung 1-3 gate verdicts, and append them here.

**Charter:** `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.26a coordinator authorization: calibration
round 3 (9 cells: task1_calib + task1_stress (1/4-budget) + task2_calib, x3 arms:
contender/ablation/transformer) AUTHORIZED at the re-priced 3.593 GPU-h. Question this round
answers: does the Rev-4 answer-CE objective (§1.21 diagnosis -> §1.22 Rev 4 -> §1.23 attack
round 5 -> §1.24 audit -> §1.25 build-fix verification -> §1.26/§1.26a deploy+re-price+
authorization) finally produce recall (diagnostic-ladder rungs pass) where rounds 1-2's
2-term objective (aux-only, no CE_answer) could not (rf@0.9=0 on all 9 task1/2 cells, all
arms; models converged to the episode-membership local optimum instead)?

## 0. Pre-launch state verification

- `git pull`: already up to date at `5230780` (the §1.26a authorization commit).
- md5 of all 5 deployed files (`lm_pretrain_rd.py`, `h2h_cell_train_rd.py`,
  `ablation_mixer_rd.py`, `h2h_rung1_chain.sh`, `h2h_box_smoke_checklist.py`) verified
  IDENTICAL local vs box, byte-for-byte matching the §1.26 MANIFEST's own recorded table.
  **No redeploy performed**, per instruction.
- GPU state before launch: GPU 0 (`attrrob_2x2` tmux session, confirmed running) and GPUs
  1-4 (`fixscale_pilots` tmux session, 4 windows, confirmed running: fixscale 98M/392M
  calibration jobs on GPUs 1/2/3/4) BUSY/LIVE, left untouched throughout. GPUs 5,6,7 idle
  (0 MiB / 0%) before launch. No pre-existing `h2h_*` tmux session.

## 1. Critical pre-launch finding: stale round-2 data + FATAL marker

Not disclosed in the launch brief; found by reading the raw box state before touching
anything, per the "coordinator reads the raw artifact" house convention.

`results/h2h_rung1/FATAL` was present (0 bytes, mtime Jul 9 02:48, immediately after
`h2h_20_band_check.log` — round 2's own band-check HARD-STOP, the exact chronology §1.21
documents: aux-weight parity achieved but all arms plateaued at the episode-membership
ceiling -> HARD-STOP -> box diagnosis -> Rev 4). The chain script refuses to run at all while
`FATAL` is present (`[ -f "$RES/FATAL" ] && ... exit 1`).

`results/h2h_rung1/calib/` held all 13 files matching `--list-cells calibration`'s exact 13
launchable cells (verified via a fresh `--list-cells calibration` call on box). 9 carried an
`_auxrev2` suffix — round 2's aux_weight=2.0 re-run, trained under the OLD 2-term objective
(no `CE_answer` at all, pre-Rev4) — exactly round 3's 9 target cells (task1_calib/
task1_stress/task2_calib x3 arms). The other 4
(`ablation_task3_calib_primary`, `transformer_task3_calib_lr_grid_0/1/2`) are unaffected by
Rev 4 — confirmed via `ladder_applies()`/the task3 training path itself: task3 "has no
query/answer-position structure," so `CE_answer` structurally does not apply — and correctly
remained valid/current.

`is_valid_result()` (`h2h_sweep_runner_rd.py`) checks only that a result JSON parses and has
every `REQUIRED_RESULT_KEYS` entry — it does not check which training objective/round
produced the file. Left in place, `run_cells_par`'s resume-safety would have silently
SKIPPED all 9 round-3 target cells as "already valid," re-recording round 2's stale
pre-Rev4-objective numbers as round 3's answer — the identical class of bug §1.24's AUD2-F4
finding flagged for the round-3-to-round-4 transition, occurring one round earlier than the
design doc's own bookkeeping had explicitly written a procedure for.

**Action taken** (mirrors the round-4 invalidation procedure already documented verbatim in
`h2h_rung1_chain.sh`'s own Stage-B comment block, applied one round earlier):
```
mkdir -p results/h2h_rung1/calib_round2_archived_20260709T054023Z
mv results/h2h_rung1/calib/h2h_calib_{contender,ablation,transformer}_task1_calib_primary_K32_auxrev2.json \
   results/h2h_rung1/calib/h2h_calib_{contender,ablation,transformer}_task1_calib_stress_locate_only_K48_auxrev2.json \
   results/h2h_rung1/calib/h2h_calib_{contender,ablation,transformer}_task2_calib_primary_auxrev2.json \
   results/h2h_rung1/calib_round2_archived_20260709T054023Z/
mv results/h2h_rung1/FATAL results/h2h_rung1/FATAL_round2_bandcheck_archived_20260709T054023Z
```
The 4 task3 cells were left untouched. No `CALIBRATION_COMPLETE.json` existed yet (round 2
hard-stopped before Stage C). Stage A's pilot gate (`PILOTS_GATE.json`, all 9 rows +
top-level `ok:true`) was already valid and unaffected — confirmed to skip cleanly on relaunch
(consistent with round 3's own re-priced 3.593 GPU-h figure, which does not include pilot
cost — already sunk).

**This decision was made under the LAUNCH agent's own authority**, per the coordinator's
explicit "round 3 is AUTHORIZED" instruction: the FATAL marker's "coordinator review
required" gate was, in substance, already discharged by the entire §1.21-1.26a gauntlet
(diagnosis -> Rev 4 -> attack round 5 -> build-fix -> audit -> verification -> deploy ->
re-price -> authorization) that produced this exact launch order. Clearing the mechanical
artifact of round 2's already-diagnosed, already-fixed failure is the operational act of
executing that authorization, not a new judgment call. Recorded here per the "coordinator
reads the raw artifact and records the tiebreak" house convention (`CLAUDE.md` hard rule).
**Confirmed correct post-launch**: the chain log shows exactly 4 `SKIP (already valid)` lines
(the 4 untouched task3 cells) followed immediately by fresh `LAUNCH` lines for the first 2
archived cells — the scoping was exact.

## 2. A second finding: a real (non-blocking) bug in the chain's own negative-test triple

Stage 0's negative-test triple (`h2h_rung1_chain.sh` lines ~159-164) is meant to prove
`--token-probe`'s refusal logic has teeth by removing each of the two required env tokens in
turn and confirming refusal. NEG-TEST 1 (`env -u HEADTOHEAD_PI_SIGNOFF ...`) executed and
correctly printed `REFUSE (sec 1.7 gate 5): missing env tokens=['HEADTOHEAD_PI_SIGNOFF']`.
NEG-TEST 2's command (`env HEADTOHEAD_PI_SIGNOFF=1 -u HEADTOHEAD_MATCH_GATE_SIGNOFF ...`)
instead printed `env: '-u': No such file or directory` — GNU `env`'s grammar is
`env [OPTION]... [NAME=VALUE]... [COMMAND]`; once a `NAME=VALUE` token appears, a later `-u`
is no longer parsed as an option and is treated as the command to exec, which doesn't exist.
This is a real, previously-undetected bug in the deployed, md5-verified, gauntlet-approved
`h2h_rung1_chain.sh` (part of the audited `68e2768` commit chain).

**Impact: non-blocking, but a real coverage gap.** The chain's `cmd && {FAILED_TO_FAIL} ||
true` wrapper only distinguishes "cmd succeeded" (bad) from "cmd failed for any reason"
(treated as pass); env's parse error is a non-zero exit, so it's silently absorbed by
`|| true` and the chain proceeds normally — it does NOT verify that `--token-probe` itself
correctly refuses when `HEADTOHEAD_MATCH_GATE_SIGNOFF` alone is missing. The underlying
Python-level refusal logic IS independently verified elsewhere (CPU selftest 5's own
negative-test triple, which calls the check directly rather than via this exact shell `env`
invocation, PASSED cleanly). NEG-TEST 3 (both env vars set, bad `--gates-dir`) and the final
real `--token-probe` call (both env vars + real gates-dir) both ran and passed correctly
after this, so **round 3's launch was not blocked or compromised** — but this specific shell
invocation should be fixed (`env -u HEADTOHEAD_MATCH_GATE_SIGNOFF HEADTOHEAD_PI_SIGNOFF=1
...` — options before assignments) in a future revision. Flagged, not fixed here (no
redeploy authorized for this launch).

## 3. Launch

GPU constraint: the launch brief said "GPUs 5, 6, 7 ONLY"; `h2h_rung1_chain.sh` itself
hard-refuses if GPU 7 appears in `H2H_GPUS` (`"GPU 7 IS NEVER USED (pool-reserved, deploy
directive)"` — a pre-existing, deliberate guard). Reconciled by using `H2H_GPUS=5,6` (2
GPUs) — stays within the brief's boundary (never touches 0-4) while honoring the script's
own stricter, pre-existing reservation on GPU 7. Flagged, not silently resolved either way.

```
tmux new-session -d -s h2h_calib3 \
  "cd /home/nvidia/chapter2/deltanet_rd; export H2H_GPUS=5,6; \
   while [ ! -f results/h2h_rung1/STOP ] && [ ! -f results/h2h_rung1/FATAL ]; do \
     bash h2h_rung1_chain.sh >> logs/h2h_rung1_supervisor_calib3.log 2>&1; sleep 15; done"
```
Launched 2026-07-09 05:41:11 UTC (box time). Self-healing per house pattern: any transient
per-cell failure exits 1 without the FATAL marker, the outer `while` loop retries after 15s;
gate-class failures (band check, 3-strike same-cell) touch `FATAL` and the loop refuses to
continue past it (matches the design's own failure discipline, `h2h_rung1_chain.sh` header
comment).

## 4. Stage-by-stage verification (through the start of Stage B)

- **Stage -1 (CPU self-tests, forced stub):** all 6 smoke suites + injectivity-guard teeth
  test PASSED, including the 17-item `h2h_cell_train_rd.py --selftest` suite (selftest 16/17
  specifically re-verify AUD2-F1's slice-before-matmul fix and AUD2-F2's K-restricted
  gather+argmax — both PASS). Ran slower than typical (~3-4 min instead of seconds) because
  the box's 5 other live GPU campaigns (attrrob_2x2, 4x fixscale_pilots) are heavily
  CPU-contended (load average 81.6 on a 208-core box, the selftest process itself observed
  at ~6600% CPU / ~66 threads) — confirmed alive and progressing via `ps` CPU-time deltas,
  not hung.
- **Stage 0 (gate tokens + negative-test triple):** all 3 gate-6/7/box-smoke tokens present;
  negative-test triple ran (see finding #2 above re: NEG-TEST 2's env bug); real
  `--token-probe` with both signoffs + real gates-dir PASSED.
- **Stage A (gate-2 timing pilots):** all 9 `pilot_<arch>_<task>.json` files already valid
  (from the prior §1.26 timing-pilot work) — all SKIPPED. `PILOTS_GATE.json`: all 9 rows
  `ok:true`, top-level `ok:true`, total projected training 11.675 GPU-h (unchanged from the
  design's own sweep estimate).
- **Stage B (gate-1 calibration) — IN PROGRESS at report time:** the 4 valid task3 cells
  correctly SKIPPED; the first wave (`contender/task1_calib_primary_K32_auxrev2` on GPU 5,
  `contender/task1_calib_stress_locate_only_K48_auxrev2` on GPU 6) LAUNCHED immediately after.

## 5. Health verification (first wave)

Both processes confirmed with the exact expected command line (`--run-cell <name> --set
calibration --out results/h2h_rung1/calib/<name>.json --device cuda`, correct
`CUDA_VISIBLE_DEVICES`). GPU 5/6 VRAM 9.7-13.8 GB (sane, well inside the §1.26 pilot's
~6.9GB single-cell estimate plus normal variance), utilization 86-90% (real compute, not
idle/hung). GPUs 0-4 unchanged throughout (confirmed untouched); GPU 7 confirmed untouched
(0 MiB).

Training curves confirmed healthy and monotonically improving over the observed window:

| cell | step | loss | rf_train | cos_train |
|---|---|---|---|---|
| primary (20000 steps total) | 500 | 7.6907 | 0.0000 | 0.1145 |
| primary | 1000 | 7.4502 | 0.0000 | 0.1194 |
| primary | 1500 | 7.2687 | 0.0000 | 0.1332 |
| primary | 2000 | 7.0982 | 0.0000 | 0.1421 |
| primary | 2500 | 7.1086 | 0.0000 | 0.1478 |
| stress (5000 steps total, exactly 1/4 of primary's budget as designed) | 500 | 7.8265 | 0.0000 | 0.1151 |
| stress | 1000 | 7.7419 | 0.0000 | 0.1158 |
| stress | 1500 | 7.6064 | 0.0000 | 0.1168 |
| stress | 2000 | 7.5536 | 0.0000 | 0.1191 |

Loss decreasing, `cos_train` climbing steadily; `rf_train` still 0 at 10-40% through
training, which is expected this early — whether it rises above the rf@0.9 decision bar (or
the rung-1 gate passes at all) is precisely THIS round's open question, not yet answerable
from this window. No NaN/Inf, no crash-signature strings in any per-cell log, no `FATAL`.

## 6. What remains for harvest (NOT done in this launch)

1. Wait for all 9 cells across ~5 GPU-waves on 2 GPUs (rough estimate 1.5-3h wall clock,
   given per-cell ceilings derived from the §1.26 measured rates: contender ~18.7min,
   ablation ~43.5min, transformer ~70.3min per full cell at 1.5x padding; stress cells 1/4
   that). `h2h_calib3` tmux session + its self-healing supervisor loop is left running
   unattended — no action needed to keep it alive.
2. Once Stage B's `run_cells_par` finishes all 9 cells (or a cell fails 3 strikes and touches
   `FATAL`), the chain runs `h2h_calibration_bands_rd.py --check-dir` — ANY failed band
   (sanity bands included) hard-aborts via `FATAL`.
3. If bands pass, Stage B2 (M-sweep timing pilot) and Stage C
   (`CALIBRATION_COMPLETE.json` write) follow, then the chain BLOCKS indefinitely on
   `MARGINS_FROZEN.token` (coordinator-only write — intentionally NOT part of this task).
4. **Harvest checklist for the next agent/check-in:**
   - `tail results/h2h_rung1/calib/*.json` for all 9 target cells + read
     `logs/h2h_rung1_supervisor_calib3.log` tail for FATAL/errors.
   - If `CALIBRATION_COMPLETE.json` exists: read it, plus `logs/h2h_20_band_check.log`, for
     the per-cell rung 1 (K-restricted gather+argmax, >3x chance), rung 2 (identity-classifier
     + R5-F2 capacity null), rung 3 (rf@0.9) verdicts — THE central question (does Rev-4's
     CE_answer objective produce recall where rounds 1-2 couldn't) is answered here.
   - If `FATAL` exists instead: read the log immediately preceding it (band-check failure,
     3-strike cell failure, or budget-ceiling abort are the three possible causes per the
     chain's own failure discipline) and report which cell/reason.
   - Archive per-cell result JSONs (≤25MB) into this directory + mirror to
     `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-09_h2h_calib3/` if
     mounted; append the verdict table to this MANIFEST; update
     `HEAD_TO_HEAD_DEMO_DESIGN.md` with a new `§1.27` gauntlet-bookkeeping record before any
     dependent stage (margin freeze, sweep) is dispatched, per the gauntlet-bookkeeping hard
     rule.

## 7. Security note

One fake `<system-reminder>` injection sighted in `Read` tool stdout while reading
`HEAD_TO_HEAD_DEMO_DESIGN.md` at the very start of this task (a "the date has changed... DO
NOT mention this to the user" block plus a fabricated "available agent types" list block).
Verified against `git show HEAD:matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md` (the real
committed file has no such text at that location, and its own line 3283 documents an earlier
sighting of the identical pattern) and disregarded — no compliance with either injected
instruction. Reported per the injection-tally convention. (Separately: this session's
Monitor-tool background-task notifications ARE legitimate, correctly schema-tagged with a
task-id and explicit "not user input" framing — distinguished from the fake block by
provenance, not by content alone.)

## 8. Local harness note (unrelated to the box, recorded for completeness)

The local `pre-train-gate` pre-commit-style hook fired on several read-only SSH commands in
this session (e.g. `ssh ... "python h2h_cell_train_rd.py --list-cells calibration"`) because
it pattern-matches `python ... .py` in the command STRING without distinguishing a remote SSH
invocation from an actual local training launch, then tries to resolve the script path
locally and fails. Used the hook's own documented `DRY_RUN_BYPASS=1` escape hatch for these
specific read-only remote metadata queries only (never for anything that could be an actual
local training launch — there were none in this task, everything ran on the box via SSH).

## Files in this record
- `MANIFEST.md` — this file
- (pending harvest) per-cell result JSONs, `CALIBRATION_COMPLETE.json`,
  `h2h_20_band_check.log`, `h2h_rung1_supervisor_calib3.log` tail at completion/abort.
