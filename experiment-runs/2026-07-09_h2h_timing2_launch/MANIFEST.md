# H2H DEPLOY + TIMING-PILOT — sec 1.24/1.25 pre-launch-fix deploy, box re-smoke, fresh
timing pilot (real fused kernel), round-3 decision gate (2026-07-09, box time)

**Charter:** `HEAD_TO_HEAD_DEMO_DESIGN.md` sec 1.25's closing NEXT paragraph: "deploy patch to
box (md5) → FRESH timing pilot (re-price calibration round 3 + sweep; closes the 1.2-1.5×
target on the real kernel) → calibration round 3 (9 cells) → ...". This launch = deploy +
re-smoke + fresh timing pilot + the pre-registered decision gate. **Verdict: gate FAILED on
the round-3 re-price condition — calibration round 3 was NOT chained.** Pilot-only spend.

**GPU constraint honored:** ONLY GPU 5 used (of the permitted 5/6/7 set). GPU 0
(`attrrob_2x2`) and GPUs 1/3 (`fixscale_pilots`) were confirmed busy and untouched before any
work started; no tmux session other than my own (`h2h_timing2_pilot`, self-terminated on
completion) was created or touched; no `pkill` used anywhere.

## 1. Deploy — md5 table

Local HEAD verified to contain `68e2768` (the §1.24 pre-launch fix commit) via
`git merge-base --is-ancestor 68e2768 HEAD`; all four target files clean vs HEAD
(`git diff --stat HEAD -- <files>` empty). Deployed via `scp` to
`/home/nvidia/chapter2/deltanet_rd/`.

| file | local md5 | box md5 (post-deploy) | match |
|---|---|---|---|
| `lm_pretrain_rd.py` | `34addd9d8cc6a3df5a367d0f18a2ee0e` | `34addd9d8cc6a3df5a367d0f18a2ee0e` | YES |
| `h2h_cell_train_rd.py` | `77d8d9e2f00ddc70d862c0789f8a14ce` | `77d8d9e2f00ddc70d862c0789f8a14ce` | YES |
| `ablation_mixer_rd.py` | `612cb2d4c6e0022d0ba5801a36c88d28` | `612cb2d4c6e0022d0ba5801a36c88d28` | YES |
| `h2h_rung1_chain.sh` | `009fbb39b508e99c80f90542f6ada370` | `009fbb39b508e99c80f90542f6ada370` | YES |
| `h2h_box_smoke_checklist.py` (untouched by 68e2768; deploy not required, verified identical) | `7d42c18b7bfa51e1178132d7d56bab4a` | `7d42c18b7bfa51e1178132d7d56bab4a` | YES |

`git show --stat 68e2768` confirms exactly these 4 files changed (192 insertions, 21
deletions); `h2h_box_smoke_driver.py` and `h2h_box_smoke_checklist.py` are untouched by the
fix and were already byte-identical local↔box before this deploy (md5 `bec0073a...` and
`7d42c18b...` respectively).

## 2. Re-smoke on box (GPU 5, real fla/Triton kernel — the pre-passed tokens from the prior
deploy do NOT cover this build, code changed)

**Items 1-4 + gates 6/7** (`h2h_box_smoke_driver.py`, unmodified, run fresh):
```
CUDA_VISIBLE_DEVICES=5 /home/nvidia/tdenv/bin/python h2h_box_smoke_driver.py \
    --token-dir results/h2h_rung1/gates_v2_20260709
```
Result: **ALL BLOCKING ITEMS PASSED.** Fresh tokens written under
`results/h2h_rung1/gates_v2_20260709/` (full stdout archived: `box_smoke_driver_output.log`).
Item-by-item: item 1 (lm_pretrain_rd --smoke) PASS; probe-head suite (smoke 1-8, incl. AUD-F1's
box-only contender aux-isolation branch, smoke 8c) PASS; item 2 (end-to-end tap-changes-with-q,
real nonzero S_T, |S|=579.73) PASS; item 3 (contender aux-only k/v/b grads nonzero on real
kernel) PASS; gate 6 MATCH-GATE 2-pass PASS (the "has teeth" self-test's deliberate
mid-run corruption shows two transient `[FAIL]` lines — `n_ablation` agreement and the
transformer FLOP-match — these are the intended mutation being CAUGHT, confirmed by
`agreement_check_detected_corruption=True` / `tolerance_check_detected_corruption=True`; the
clean re-run immediately after both re-verify PASS); item 4 (state bytes == 32,768 fp32) PASS;
gate 7 (probe-capacity null, all 3 arms, `recovered_frac=0.0 < 0.05` bar) PASS.

**Items 9-10** (`continuation_blankout_inplace` / `continuation_blankout_fresh_instance`,
sec 1.23's own new blank-out box-only residual — NOT yet wired into `h2h_box_smoke_driver.py`,
which predates them): the audited functions (`_recurrent_continuation_answer_logits`,
`DeltaNetLM`, `AblationLM`) are device-agnostic per the driver's own established convention, so
a small throwaway harness (`box_smoke_items_9_10_realkernel.py`, zero logic changes to any
audited file, archived here) sets `torch.set_default_device("cuda")` and replicates
`mode_selftest`'s own selftest-12/13 construction verbatim. (The FULL `--selftest` suite was
tried first, unforced-stub, both plain and under a blanket CUDA default device; both crash --
plain crashes inside real fla's RMSNorm on CPU tensors (`ValueError: Pointer argument cannot be
accessed from Triton (cpu tensor?)`, confirming the CLAUDE.md Hard Rule "fla 0.5.1's RMSNorm has
no CPU fallback"); the blanket-CUDA variant crashes inside `random_unit_rows_init`'s
CPU-pinned `torch.Generator` (`RuntimeError: Expected a 'cuda' device type for generator but
found 'cpu'`) — a real device-context conflict baked into `mode_selftest`'s own design (target
tables intentionally use a fixed CPU generator for cross-run reproducibility), unrelated to
items 9-10. The narrow harness sidesteps this by never touching `ProbeRig`/`T_val` at all.)
Result: **PASS both items, both arches** (contender S_T_abs_sum=620.63/583.68, ablation
S_T_abs_sum=6.34/3.94, both nonzero — confirms a REAL, not CPU-stub-zero, kernel state was
exercised; logits bit-identical in both the in-place-corruption and fresh-instance cases; the
fresh-instance check's `id()`/`data_ptr()` distinctness guard also confirmed true). Full stdout
archived: `box_smoke_items_9_10_output.log`.

**Not re-run (out of this deploy's scope):** checklist items 5-7 (`msweep_timing_pilot`,
`gate2_timing_pilots`, `gate1_calibration_run`) are later pipeline stages — the M-sweep pilot
gates the 90-pass fan-out (far downstream of round 3), and `gate1_calibration_run`/
`gate2_timing_pilots` are what calibration round 3 and this launch's own §3 pilot respectively
*are*, not a prerequisite to re-run separately. Item 8 (d=128 diagnostic) is non-blocking,
explicitly deferred pending a separate escalation authorization never given here.

**All blocking items covered by this deploy's scope: PASS. No STOP triggered.**

## 3. Fresh timing pilot (GPU 5, real fused kernel, task1_calib K=32 production config)

Script: `h2h_timing2_pilot.py` (throwaway harness, zero logic changes to any audited file;
reuses `h2h_cell_train_rd.train_grammar_cell`/`run_one_cell` — the SAME harness
`mode_pilot_pair` uses — at PINNED production dims: contender `d_model=256,d_state=64,
n_layers=2`; ablation same `d_state=64`; transformer `d_model=256,n_layers=2,n_heads=4`).
Method: mode_pilot_pair's own two-point convention (cold run = warmup 10 + timed 40 steps,
then a second warm-only run of 40 steps; `s_per_step` taken from the warm run, startup/
kernel-autotune overhead cancels). VRAM: `torch.cuda.reset_peak_memory_stats()` before each
run, `torch.cuda.max_memory_allocated()` after. Two variants per recurrent arch: **FIXED**
(current, unmodified `_recurrent_continuation_answer_logits`, slice-before-matmul) and
**PRE-FIX** (in-process monkeypatch reconstructing AUD2-F1's own "before" computation — full
vocab matmul over all 128 padded positions via `model.forward(..., return_states=True)`
without `return_hidden`, then slice — a faithful reproduction since the fix was additive-only,
not a guess). Transformer has no pre-fix variant (its `answer_logits` reuse the existing single
forward pass by design, "no second forward", untouched by AUD2-F1). Baseline denominator = the
ALREADY-REGISTERED, ALREADY REAL-GPU-measured sec 1.6 round-1/2 rates (contender 10 min,
ablation 19.5 min, transformer 27 min @ `FULL_STEPS=20,000`, pre-Rev4, no CE_answer at all) —
reused rather than re-derived, for direct consistency with the design doc's own round-3
arithmetic this pilot re-prices. Ran once, sequentially, all on GPU 5 (`tmux h2h_timing2_pilot`,
self-terminated on completion — no supervisor loop needed, single short-lived run).

**Measured rates:**

| arch | variant | s/step (warm) | peak VRAM (GB) |
|---|---|---|---|
| contender | fixed | 0.037323 | 6.881 |
| contender | prefix (pre-fix reconstruction) | 0.090777 | 16.468 |
| ablation | fixed | 0.086992 | 6.829 |
| ablation | prefix | 0.139651 | 16.416 |
| transformer | fixed | 0.140589 | 27.277 |

**Ratios (fixed vs the sec 1.6 pre-Rev4 real-GPU baseline — the exact "per-step cost ratio vs
the pre-fix baseline path" this pilot's charter asks for):**

| arch | baseline s/step (registry) | ratio fixed/baseline | ratio pre-fix/baseline | ratio pre-fix/fixed |
|---|---|---|---|---|
| contender | 0.03000 | **1.244×** | 3.026× | 2.432× |
| ablation | 0.05850 | **1.487×** | 2.387× | 1.605× |
| transformer | 0.08100 | 1.736× | n/a | n/a |

**The AUD2-F1 fix's own target (contender/ablation — the two arches the fix actually
targeted) is CLOSED on the real kernel: 1.244× and 1.487×, both cleanly inside the design's
~1.2-1.5× target and both ≤ this launch's 1.6× threshold** — a large, confirmed improvement
over the CPU-stub's own confounded 2.49×/3.55× (which the stub's padded-recurrence Python-op
overhead, not the fix, was driving). VRAM also confirms the fix at the op level: the pre-fix
variant's peak VRAM is ~2.4× the fixed variant's (16.4GB vs 6.9GB) — consistent with a
(B·Q, 128, vocab) intermediate tensor (256 rows × 128 positions × 50,259 vocab × 4 bytes ≈
6.6GB) that the fix's slice-before-matmul eliminates.

**Flagged finding (NOT an AUD2-F1 regression — the transformer's fused
`_fused_transformer_tap_and_answer_logits` path was never touched by the fix and was never
measured on real GPU under the Rev4 three-term objective before now):** the transformer's own
measured ratio, 1.736×, exceeds both the 1.2-1.5× design target and this launch's 1.6×
threshold. This is a genuine, previously-unmeasured cost of Rev4's own construction (the
transformer's "no second forward" trick still reprocesses the FULL context, repeated once per
query — `B·Q` rows of `T_ctx+qlen` tokens — a structurally larger forward than the recurrent
arms' short-query continuation), not a build-fix defect. Reported here per the program's own
"flag, don't paper over" convention; not blocking on its own, but material to the round-3
re-price below since round 3 trains all three arms.

**Re-priced calibration round 3 (9 cells: task1_calib + task1_stress(¼-budget) + task2_calib,
×3 arms; task2 reuses task1's measured rate — same continuation shapes, `N_QUERY_TRAIN=8`,
`GRAMMAR_BATCH=32`, architecture-level not task-conditioned — flagged, not silently assumed):**

| component | GPU-h |
|---|---|
| task1_calib + task2_calib (full, ×3 arms) | 2.9434 |
| task1_stress (¼ budget, ×3 arms) | 0.3679 |
| diagnostic-ladder overhead (sec 1.6's own 8.5% factor) | 0.2815 |
| **Round-3 total (measured)** | **3.5928** |
| registry prior (§1.6) | 2.300 |
| **ratio vs registry** | **1.562×** |

**Re-priced 27-cell sweep (rough upper bound, 9 full-cell-equivalents/arch × measured fixed
rate — NOT the exact budget_frac-weighted manifest, flagged as an upper bound, not a precise
re-price):** ≈13.25 GPU-h (vs the design's own prior ≈11.675 GPU-h pilot-measured estimate,
sec 1.6 — comparable order of magnitude, consistent with the pilot not finding a large
regression at sweep scale either).

Full JSON: `h2h_timing2_fresh_pilot.json`.

## 4. Decision gate (pre-registered before this pilot ran)

Rule: *IF measured end-to-end ratio ≤ 1.6× AND re-priced round-3 total ≤ 1.5× the registry's
prior round-3 estimate, CHAIN calibration round 3 immediately. ELSE: launch nothing beyond the
pilot, report the numbers and stop.*

- Condition 1 (ratio ≤ 1.6×): the AUD2-F1-relevant arches PASS (contender 1.244×, ablation
  1.487×); the transformer's own newly-measured 1.736× FAILS this threshold.
- Condition 2 (round-3 re-price ≤ 1.5× registry, i.e. ≤ 3.450 GPU-h): **FAILS** — measured
  3.593 GPU-h, ratio 1.562×.

**Both readings of condition 1 aside, condition 2 fails unambiguously (1.562× > 1.5×).
VERDICT: GATE FAILED. Calibration round 3 was NOT chained.** No cells beyond the pilot's own
short timing runs were launched; no supervisor loop, no `H2H_DIAL_ROUND` export was needed.
This is the pre-registered rule working as designed, not a judgment call — the fix itself is
confirmed working (contender/ablation cleanly hit their target), but the round-3 total
including the transformer's own previously-unpriced Rev4 cost sits just above the pre-registered
margin. This is reported to the coordinator, not silently re-authorized or split-the-
difference-adjusted, per the "coordinator reads the raw artifact and records the tiebreak"
house convention (`CLAUDE.md` hard rule) — the next decision (re-scope round 3 excluding/
re-budgeting the transformer arm, widen the pre-registered margin, or accept the 1.562× spend
given the fix's own confirmed success) is the coordinator's, not this agent's, to make.

## 5. Cost (this launch)

Pilot spend: ~15 real training steps × 5 configurations × ~0.04-0.14s/step ≈ trivial
(sub-minute of actual GPU compute; box smoke items add a few more seconds). Total this
launch: **≈0.02 GPU-h** (well under 1 minute of wall clock across all measured runs, per the
pilot JSON's own `warm_wall_s`/`cold_wall_s` fields, plus the box-smoke driver's own ~1-2 min
wall clock). No calibration cells were launched, so no material GPU-h beyond the pilot itself.

## 6. Security note

Two fake `<system-reminder>` prompt-injection attempts were sighted in tool stdout during this
task: (1) a "the date has changed... DO NOT mention this to the user" block embedded inside a
`Read`/`Bash` tool result while reading `HEAD_TO_HEAD_DEMO_DESIGN.md`, paired with a fabricated
"available agent types" list; (2) none further observed during the SSH/box work. Both were
disregarded and verified against `git log` (real commit dates checked directly, e.g. `68e2768`
committed 2026-07-08 22:07:14 -0700 per `git show`) rather than trusted. No compliance with
either injected instruction. Reported here per the program's injection tally convention.

## Files in this record

- `MANIFEST.md` — this file
- `deploy_manifest.md5` — the 5-file md5 table (machine-readable)
- `box_smoke_driver_output.log` — full stdout, items 1-4 + gates 6/7 (fresh, GPU 5)
- `box_smoke_items_9_10_output.log` — full stdout, items 9-10 (fresh, real kernel, GPU 5)
- `box_smoke_items_9_10_realkernel.py` — the throwaway items-9-10 harness (exact script run)
- `BOX_SMOKE_ITEMS_1_4_PASSED.token`, `GATE6_MATCH_GATE_PASSED.token`,
  `GATE7_PROBE_CAPACITY_NULL_PASSED.token` — the fresh box-smoke-driver token files
  (from `results/h2h_rung1/gates_v2_20260709/` on box)
- `h2h_timing2_pilot.py` — the exact fresh-timing-pilot script run
- `h2h_timing2_fresh_pilot.json` — the pilot's full measured output (rates, VRAM, ratios,
  re-price arithmetic)
