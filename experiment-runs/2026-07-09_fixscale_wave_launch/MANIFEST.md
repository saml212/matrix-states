# FIX-AT-SCALE full wave — deploy + box checklist + LAUNCH (2026-07-09, box time)

**Wave:** FIX-AT-SCALE full sweep (`FROZEN_BIAS_LM_DESIGN.md` §13, Rev 1). Orchestrator:
`fixscale_wave.py` + `fixscale_supervisor.sh` (commit `bd40ebb`, cleared for deploy at
`a593e9f` §13.19: "DEPLOY+LAUNCH DISPATCHED with the 11-item box checklist"). 28-cell manifest
(12 arm_off + 12 arm_per_token + 4 arm_global_probe), ~281/300 GPU-h at 2× contingency.

**Authorization:** §13.19 CLEARED-FOR-DEPLOY (`a593e9f`); PI GPU-saturation directive
(`STATE.md` GOALS item 5, 2026-07-09, verbatim in `FROZEN_BIAS_LM_DESIGN.md` §13); the
300 GPU-h `fix-at-scale` ledger.

## md5 table (checklist item 1)

Local git HEAD (`bd40ebb`, working tree clean) vs box `/home/nvidia/chapter2/deltanet_rd/`:

| file | md5 |
|---|---|
| `fixscale_wave.py` | `402f0928cab3b5d2cabc987f22a51f38` |
| `fixscale_supervisor.sh` | `b931cb4502989de28352fc929e324672` |
| `smoke_fixscale.py` | `c892298c2e74e50ad05ecb2c2e3f49a1` |
| `bands_pinned_frozenbias.py` | `9e12e8b376e2b9c574f2fad594b4ee9f` |

All 4 EXACT MATCH (deployed via `scp`; box's pre-deploy `bands_pinned_frozenbias.py` was stale —
Jul 6, pre-l12 atomic-write fix, md5 `1bb7cbcfe7056f431d795d65ab0d9199` — diffed before overwrite;
confirmed the delta was exactly the l12 atomic tmp+fsync+os.replace pin-write fix and that no live
process imports this module directly, safe to overwrite). **Item 1: PASS.**

## Box smoke checklist (§13.19's 11 items)

**(2) Real-kernel training subprocess smoke — PASS.** GPU 7, real fla/CUDA, toy config
(d_model=128, d_state=64, n_layers=2, seq_len=512, batch=4). 8 steps, ckpt-every 4: loss
10.8598→10.8387, checkpoints written at step 4 and step 8. Resume/ckpt-load: a second run with
`--init-checkpoint` on the step-8 checkpoint loaded successfully (`strict=True` + config-equality
assert), trained 4 more steps to completion. Forward, backward, checkpoint-write, and
checkpoint-load(resume) all confirmed on real kernels.

**(3) Corpus loading against /data — PASS.** Same run loaded real `openr1-mix-ext`
(344,736,296 train tok / 230,074 docs) and `wikitext-mix-ext` (247,349 val tok / 60 docs) —
correct token/doc counts, no errors.

**(4) Real peak-VRAM at both scales — PASS (cited, not re-run).** Gate tier's own
`PILOT_98M_VERDICT.json` / `PILOT_392M_VERDICT.json` (verdict PASS both) measured at these
EXACT configs:
- 98m off: 23.22 GB alloc / 25.62 GB reserved; per_token: 23.46 / 25.86 GB.
- 392m off: 38.35 GB alloc / 41.98 GB reserved; per_token: 38.98 / 42.63 GB.
Both comfortably under 80GB H100 budget.

**(5) Supervisor tmux runtime smoke — PASS.** Stub `tmux new-session -d` session confirmed:
session creation, background command execution, `capture-pane` output readback, clean
termination on command exit. STOP_FILE honor confirmed separately: a deliberately-failing stub
scale (`bogus_stub_scale`) via `fixscale_supervisor.sh armoff-loop`, touched
`STOP_fixscale_wave`, session terminated ~60s later (matches `run_until_terminal`'s 60s retry
cadence, STOP_FILE checked after the nonzero-exit retry boundary).

**(6) Verify-pin determinism — PARTIAL / substituted (see Deviations below).** No arm_off cell
is complete anywhere on the box yet (verified directly against the gate tier's own result JSONs,
contradicting the task brief's assumption — see Deviations). `verify-pin`/`write_wave_pin` hard-
require all 6 arm_off cells complete, so the literal CLI check could not run. Substituted: ran
the standalone `comparator` subcommand (the exact function `_collect_pin_inputs`/`verify_pin`
depend on) TWICE on the same real-CUDA checkpoint from item (2)'s smoke run — byte-identical
JSON output both times. **Determinism of the underlying mechanism confirmed; the full CLI-level
`verify-pin` re-check is deferred until a real arm_off cell completes** (flagged for a follow-on
check, not blocking — the pin itself won't be written for hours regardless).

**(7) `wave-minus1-check --d-state 128 --device cuda` — PASS.** `BIT-IDENTICAL`. Per §13.19's N2
caveat, explicitly noted: this is a d128 construction/determinism check ONLY, not off-path
purity evidence (`check_off_path_bit_identity`'s two constructions both default-resolve to
`arm="off"`; real off-path purity teeth are smoke [2] + the forward's `!="off"` guard).

**(8) startup-to-step-1000, HARD GATE (98m<118s, 392m<418s) — PASS both scales, large margin.**
Gate-tier's own rate-watch CSVs use wall-clock-interval polling (not exactly step-1000-aligned),
so this was measured directly and precisely instead: real production configs, GPU 7, `--steps
1000`, dense (~3s) polling replicating `_budget_watchdog`'s own `elapsed_since_launch / step`
formula.
- **98m:** step 1000 at elapsed=246.404s → **cum_rate = 0.2464 s/step** (bound: 1.5×0.236=0.354).
  Fitted startup overhead ≈ 12.6–13.5s (from steps 100→1000 and 200→1000 linear fits) — **~9×
  margin under the 118s bound.**
- **392m:** step 1000 at elapsed=841.320s → **cum_rate = 0.8413 s/step** (bound:
  1.5×0.836=1.254). Fitted startup overhead ≈ 13.3–15.0s (steps 100→1000 and 200→1000 linear
  fits; first step-line at elapsed 15.0s) — **~28× margin under the 418s bound.**
No false-abort risk at either scale; the F2 fix's step≥1000 gate has ample headroom.

**(9) disk-check both scales — PASS.** `disk-check` initially reported `free_bytes=0`/`ok=false`
because `/data/fixscale_ckpts/train` (CKPT_ROOT) didn't exist yet — the code's own documented
"not-yet-created dir reports free_bytes=0, refuse don't crash" behavior, not a real shortfall
(`df -h /data` showed 15TB free throughout). Created the directory (which `run_cell` creates
anyway on first launch) and re-ran:
- 98m: required 597.99 GB (1.5×) — OK.
- 392m: required 743.43 GB (1.5×) — OK.
- Combined ≈1.34 TB required vs 15.5 TB free (14.14 TiB) — large margin. (The task brief's
  "~1.07 TB combined" estimate was in the right ballpark; the live-measured figure above is
  authoritative and still passes with wide margin either way.)

**(10) Clean-slate assert — PASS.** `results/fixscale/train/` empty (no `.REFUSED` markers
anywhere under `results/fixscale/`, no post_pin result JSONs) before any cell launch.

**(11) Gate-tier calib cells' terminal state — CONFIRMED NOT TERMINAL (shapes the launch plan).**
Directly checked all 4 gate-tier calib result JSONs' `complete` field (not inferred from `ps`
or elapsed time): 98m openr1/wikitext both `complete=false`, `steps_completed=22000/67547`
(~33%); 392m openr1/wikitext both `complete=false`, `steps_completed=4000/20000` (~20%). **This
contradicts the task brief's "the gate tier has completed 98M cells" assumption for item 6 —
see Deviations.** Since none are terminal, the launch plan below explicitly avoids the arm_off
seed=0 partition for both scales (see "Why slot0 is skipped").

## Deviations from the task brief (verified against the box, not assumed)

1. **Item 6's premise was false at dispatch time.** The brief stated "the gate tier has completed
   98M cells" — live JSON check showed 0/4 gate-tier calib cells complete. Substituted a
   comparator-mechanism determinism check (see item 6 above) and deferred the full CLI check.
2. **Deployment gap I introduced and fixed:** `fixscale_supervisor.sh` defaults
   `PYBIN=python3` (system Python, no torch). My first launch of all 7 tmux sessions failed
   immediately (`ModuleNotFoundError: No module named 'torch'`), retrying harmlessly every 60s
   via the existing transient-retry path (no terminal marker written — confirms F1's fix holds
   even under an unrelated failure mode). Killed and relaunched all sessions with
   `FIXSCALE_PYBIN=/home/nvidia/tdenv/bin/python3` (the script's own documented override).
   Verified working immediately after (real training log appeared for the seed=1 cell within
   seconds).

## Why armoff slot0 is deliberately not launched for either scale

The 3-way `--gpus 3` partition puts BOTH gate-tier-owned seed=0 cells (openr1 + wikitext) for a
scale into slot0. Gate-tier trains those two cells on TWO DIFFERENT physical GPUs (98m:
GPU1/GPU2; 392m: GPU3/GPU4) — pinning slot0 to a single GPU only protects ONE of its two cells
from the M4 occupancy guard's blind spot: if slot0 is launched on any GPU other than the one
still busy with a given corpus's gate-tier cell, `out_path()`'s `cell_state` sees `absent` (gate-
tier JSON not yet `complete`) and `run_cell` launches a genuine DUPLICATE ~4.5-4.7 GPU-h training
run for a cell gate-tier is already training. **Functionally, slot0 does not need to be launched
at all** — `await_armoff_and_pin`'s barrier polls `cell_state()` directly for all 6 arm_off
cells, and `out_path()` transparently resolves a gate-tier seed=0 cell to `complete` once its
JSON reports so, with zero action needed from this wave. A small `fixscale_slot0_waiter.sh`
(operational convenience script, not part of the audited codebase — polls both gate-tier JSONs
for `complete=true`, then invokes `armoff-loop <scale> 3 <offset> 0`) is wired per scale so
slot0 gets an explicit resume-skip run once safe, purely for bookkeeping parity with slots 1/2;
this closes the wave's own manifest/log symmetry without any duplicate-compute risk.

## Launch plan / slot→GPU map

**98m** (`--gpus 3`, cells: slot0={openr1 s0,wikitext s0} SKIPPED — see above; slot1={openr1
s1,wikitext s1}; slot2={openr1 s2,wikitext s2}):

| tmux session | command | target GPU | initial state at launch |
|---|---|---|---|
| `fixscale_98m_armoff_1` | `armoff-loop 98m 3 5 1` | 6 | GPU free → **launched immediately, real training confirmed** (97.6M params, seed=1, loss 10.99→5.54 over 400 steps) |
| `fixscale_98m_armoff_2` | `armoff-loop 98m 3 5 2` | 7 | GPU busy (my own item-8 measurement) → M4-refused, retrying 60s |
| `fixscale_98m_armoff_0_waiter` | `fixscale_slot0_waiter.sh 98m 5 5` | 5 (eventual) | polling gate-tier JSONs |
| `fixscale_98m_pin` | `CUDA_VISIBLE_DEVICES=7 pin-loop 98m` | 7 (comparator only, when it fires) | polling arm_off cell states |
| `fixscale_98m_post_0` | `post-loop 98m 3 5 0` | 5 | blind-gate-blocked, retrying 60s |
| `fixscale_98m_post_1` | `post-loop 98m 3 5 1` | 6 | blind-gate-blocked, retrying 60s |
| `fixscale_98m_post_2` | `post-loop 98m 3 5 2` | 7 | blind-gate-blocked, retrying 60s |

**392m** (`--gpus 3`, same slot0-skip logic; slot1={openr1 s1,wikitext s1};
slot2={openr1 s2,wikitext s2}) — launched after item 8's 392m measurement cleared:

| tmux session | command | target GPU |
|---|---|---|
| `fixscale_392m_armoff_1` | `armoff-loop 392m 3 2 1` | 3 |
| `fixscale_392m_armoff_2` | `armoff-loop 392m 3 2 2` | 4 |
| `fixscale_392m_armoff_0_waiter` | `fixscale_slot0_waiter.sh 392m 2 2` | 2 (eventual) |
| `fixscale_392m_pin` | `CUDA_VISIBLE_DEVICES=7 pin-loop 392m` | 7 (comparator only) |
| `fixscale_392m_post_0` | `post-loop 392m 3 2 0` | 2 |
| `fixscale_392m_post_1` | `post-loop 392m 3 2 1` | 3 |
| `fixscale_392m_post_2` | `post-loop 392m 3 2 2` | 4 |

**Launch-state verified live (07:11-07:12 box time):** 392m armoff slots 1/2 correctly
M4-occupancy-refused (GPUs 3/4 at 43.8/44.4 GB, gate-tier live) and retrying 60s; waiter/pin
polling; post_pin slots blind-gate-blocked and retrying. 98m slot2 SELF-RECOVERED onto GPU 7
within one 60s retry of the item-8 measurement subprocess exiting — the occupancy-guard staged
expansion demonstrated end-to-end live (`GPU 7: 0 MiB used... clear to launch` → real training
at 100% util).

All sessions launched with `FIXSCALE_PYBIN=/home/nvidia/tdenv/bin/python3` (the venv fix).
`pin-loop` sessions additionally wrapped with `CUDA_VISIBLE_DEVICES=7` — `write_wave_pin`'s
comparator step has no `--gpu` argument and would otherwise default to CUDA's ambient device 0,
which is off-limits (GPU 0 belongs to `cap_sweep`, never to be scheduled by this wave).

## Staged-expansion wiring

- **GPU 6** freed early (h2h_calib3's stress-task cell finished ahead of its primary cell) —
  98m slot1 started training immediately rather than waiting.
- **98m slot2 (GPU7)** retries every 60s until my own item-8 392m measurement subprocess exits
  (expected within ~15 min of its own launch); no manual re-launch needed — M4 occupancy guard
  handles it.
- **392m slots 1/2 (GPUs 3/4)** retry every 60s until `fixscale_pilots`' remaining 392m
  calibration cells go terminal (~10:30-10:45 box time per the live estimate).
- **98m/392m post_pin slots** all launched now; harmless blind-gate retries until each scale's
  pin lands (requires all 6 arm_off cells — the wave's own 4 + gate-tier's 2 reused — complete).
- **slot0 waiters** launched now for both scales; will invoke a resume-skip-only `armoff-loop`
  once gate-tier's own cells are confirmed complete.
- GPU 0 (`cap_sweep` + the chained 2×2 escalation) was never touched or scheduled.

## Security

Two fake `<system-reminder>` injections this session, both disregarded in full (including any
concealment instruction), both verified against ground truth before dismissal:
1. A bundled date-change+concealment block + fabricated "available agent types" list +
   fabricated MCP-server-instructions block, attached to a local `git show --stat bd40ebb` tool
   result. Verified: `git log -1 --format=%B` on the real commit contains none of that text —
   pure stdout injection, not repo content.
2. A "the date has changed... today is 2026-07-09... don't mention this" block attached to a
   `tmux capture-pane` result for the `fixscale_98m_pin` session. Verified: box `date -u`
   independently confirmed 2026-07-09 at session start, matching git commit timestamps; the
   instruction to conceal was not followed (reported here explicitly).
3. The same date-change-with-concealment pattern + fabricated "available agent types" list +
   fabricated MCP-server-instructions block, appended to the item-8 392m background-task
   completion notification. Same disposition: not complied with, reported here.

Tally 75→78 (project-wide running count; prior entry `EXPERIMENT_LOG.md` "Tally 74→75").

## Expected completion timeline

- 98m: 4 wave-trained arm_off cells (~4.478 GPU-h each) + 2 gate-tier-reused ≈ hours, gated by
  the slowest of 6; then 8 post_pin cells (~4.478 GPU-h each, 3-way parallel) ≈ ~12 GPU-h wall
  once the pin lands.
- 392m: gated on `fixscale_pilots` freeing GPUs 3/4 (~10:30-10:45 box time) before its 2
  wave-trained arm_off cells (~4.671 GPU-h each) can even start; then 8 post_pin cells similarly.
- Full 28-cell wave: ~281 GPU-h at 2× contingency, spread across the staged GPU pool as it frees;
  no single-session wall-clock estimate is meaningful given the staggered start times documented
  above — the supervisor's self-healing loops carry it to completion unattended.

## Archive contents

- `MANIFEST.md` (this file)
- `deployed_files_box.md5` — post-deploy box md5 of all 4 wave files + the waiter script
- `smoke1.json` / `smoke2_resume.json` — item (2)'s real-kernel train + resume smoke results
- `comp_run1.json` — item (6)'s comparator determinism run (run2 byte-identical, kept on box)
- `startup_98m.json` / `startup_392m.json` — item (8)'s full result JSONs (real production
  configs, 1000 steps each)
- `startup_measure_98m_poll.txt` / `startup_measure_392m_poll.txt` — the dense-poll cum_rate
  traces backing item (8)'s numbers (`.txt` because `experiment-runs/**/*.log` is gitignored)
- `startup_measure.sh` — the exact measurement script run (archived; removed from the box's
  code dir post-run)
- `fixscale_slot0_waiter.sh` — the slot0 duplicate-compute-race waiter (md5
  `5088b52a291423908adec435def7938f`, matches box; LIVE on box, 2 sessions)
- `resume_cell_with_checkpoint.py` — the §13.21 ABORTED_BUDGET→`--init-checkpoint` resume
  script (md5 `94e68c8cf6e40a55855da352586c192b`, matches box; LIVE on box in tmux session
  `fixscale_98m_resume_s1`)
- Box-side provenance kept at `/data/fixscale_smoke_test/` (JSONs + raw logs; smoke checkpoint
  dirs deleted post-verification to reclaim ~3 GB).

## Update 2026-07-09 ~16:22 box (see `FROZEN_BIAS_LM_DESIGN.md` §13.21 for full detail)

`fixscale_train_arm_off_98m_openr1-mix-ext_s1` (GPU 6) hit the 1.5× breaker at step 14200
(contention artifact — `h2h_calib3` co-scheduled GPU 5/6; loss healthy). Marker + partial JSON
archived aside (`.superseded_20260709T162229Z`), checkpoint step14000.pt backed up
(`.pre_resume_backup_20260709T162229Z`), resumed via a direct `lm_pretrain_rd.py --init-checkpoint`
invocation (`resume_cell_with_checkpoint.py`, reuses `fixscale_wave.py`'s own audited
`_cell`/`base_cmd`/`_budget_watchdog` — the `cell` CLI has no `--init-checkpoint` passthrough)
with `--steps 53547` (the remainder — `--init-checkpoint` restarts the step counter, doesn't
continue it) in tmux session `fixscale_98m_resume_s1` on GPU 6. First 500 steps confirmed healthy
and uncontended (0.2067 s/step steady-state, faster than the 0.236 s/step reference). Unblocks
`fixscale_98m_pin`'s barrier and the entire 98m post_pin phase once this cell completes.
