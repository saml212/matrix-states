# FIX-AT-SCALE gate-tier launch — timing pilots + arm_off calibration cells (2026-07-09, box time)

**Wave:** FIX-AT-SCALE (`FROZEN_BIAS_LM_DESIGN.md` §13, Rev 1). **This launch = the GATE TIER
ONLY** — §13.10 item 3 (timing pilot per scale, folding in §13.13 Rev-1 item 2's VRAM-logging
mandate) and §13.10 item 4 (calibration-first per scale; task-directed **arm_off** control cells,
1 per scale per corpus). Both gates are REQUIRED before the full sweep and are INDEPENDENT of
Rev 1's additive delta (the `arm_global_probe`). Everything below runs on lm_pretrain_rd.py's
EXISTING committed CLI — no new training code; the full-sweep launcher (blind-pin enforcement via
`assert_blind_not_broken`, the §13.8 1.5× per-cell circuit breaker, the global-probe arm, manifest
orchestration across 24+ cells, frozen_bias_lm_sweep.py scale-parametrization) is the BUILD stage,
explicitly not launched here.

## Authorization

1. **PI GPU-saturation directive, verbatim** (`STATE.md` GOALS item 5, 2026-07-09): *"how will
   [we] ensure that all these 8 gpu's are hot for the next few days. I don't want these sitting
   idle anymore."* Response recorded there as the authorization the never-self-amend rule
   requires: FIX-AT-SCALE chartered as one of two compute-heavy waves, all 8 GPUs in play.
2. **300 GPU-h `fix-at-scale` ledger — PI-ratified 2026-07-09** per the launch task's recorded
   citation: *"300 GPU-h ledger PI-ratified 2026-07-09 'the more and larger experiments the more
   data'"* (relayed by the coordinating agent in the pilot-launch task; at this record's writing,
   `STATE.md`'s LEDGERS section still shows the cap as "proposed" pending its next consolidation
   pass — the ratification citation above is the operative authorization, and STATE.md's ledger
   line should be updated to "ratified" at the next STATE touch).
3. Design authority: `FROZEN_BIAS_LM_DESIGN.md` §13.7 (corrected realized rates), §13.9 (GPU
   layout), §13.10 (gates), §13.13 attack-round-1 verdict (NEEDS-REVISION resolved into Rev 1;
   the wrong-direction disclosure §13.2/§13.11-1 stands — these arm_off cells presuppose nothing
   about the per_token sign).

## Inventory verdict (task step 1)

- **`lm_pretrain_rd.py` (committed md5 `a16fff66bb121b60ecd906ae6f9ef018`) runs BOTH arms at BOTH
  new scales TODAY.** The frozen-bias mechanism is model-side and config-driven:
  `--frozen-bias-arm {off,per_token,global}`, `--frozen-bias-lambda` (default 0.58),
  `--frozen-bias-seed` (default ANCHOR_INIT_SEED), with `build_frozen_bias_table(vocab_size,
  d_state, seed)` parametrizing the table shape per `d_state` (64 at 98M, 128 at 392M) — no
  14M-hardcoded assumption anywhere on that path. `--d-state` choices already include 128.
  Box smoke (this launch, GPU 1): ALL PASS, incl. item [17] (every arm forward/backward/
  grad-finite + off-arm bit-identity) and items [12]/[13] (rung-2/3 shapes).
- **`frozen_bias_lm_sweep.py` is 14M-hardcoded** — needs build work before it can orchestrate
  98M/392M: `RUNG1_CFG = {"d_model": 256, "d_state": 64, "n_layers": 2, ...}` (line 77), threaded
  through `cell_name` (179–181), `_spec` (188–189), `find_ckpt_size_bytes` call sites (725, 840);
  `--wave` choices are `rung1`/`rung1-seedext` only (line 873).
- **`run_lm_rd_trackc_sweep.py` has zero frozen-bias support** (0 grep hits) and its program
  ledger is closed (`PROGRAM_SPENT_GPUH = 312.23` > 300 ceiling, disclosed) — not a launch vehicle
  for this wave; its conventions (BATCH_SIZE_BY_RUNG, build_cmd_cell flag set, two-point
  calibration method) are inherited by this launch's driver instead.
- **`lm_rd_rung_configs.py` RUNGS[1]/[2]** verified on-box this launch: 97,618,176 (rung 1) and
  391,869,440 (rung 2) params, both within tolerance.
- **Build-stage finding:** `lm_pretrain_rd.py`'s `run_name` (line 3312) carries NO frozen-bias-arm
  tag — two arms at one config/corpus/seed collide in a shared ckpt dir. This launch uses per-cell
  ckpt dirs (`/data/fixscale_ckpts/calib/<cell>/`); the build-stage launcher must do the same
  (frozen_bias_lm_sweep.py's own per-cell-dir convention).

## What launched (box `youthful-indigo-turkey`, tmux session `fixscale_pilots`, 2026-07-09 ~05:02 box time)

Deployed (md5-verified local=box): `fixscale_gate_tier.py` (`b5dc273b0a9fd639d67af007df3c9355`),
`fixscale_bank.sh` (`7414d6fbbbd41a8f9836a93b41894a8b`) — archived verbatim in this directory.
Pre-launch deploy-closure check: the 6 reused scripts byte-verified vs the repo's COMMITTED copies
(all match git HEAD; the local worktree's uncommitted `lm_pretrain_rd.py` edit is the h2h
campaign's in-progress AUD2-F1 fix — additive-only `return_hidden` kwarg, NOT deployed, not this
wave's to touch). GPU occupancy re-verified live immediately before launch: GPU 0 = `attrrob_2x2`
(untouched), GPUs 1–7 idle.

| tmux window | GPU | Work |
|---|---|---|
| bank98 | 1 | 98M timing pilot (two-point: off/per_token × 200/1000 steps, openr1-mix-ext, s0, no ckpts) → verdict JSON → 98M openr1-mix-ext arm_off calibration (67,547 steps, Track-C-matched) |
| calib98w | 2 | waits for `PILOT_98M_VERDICT.json` = PASS → 98M wikitext-mix-ext arm_off calibration (67,547 steps) |
| bank392 | 3 | 392M timing pilot (same construction) → verdict → 392M openr1-mix-ext arm_off calibration (20,000 steps, §13.7's reduced budget) |
| calib392w | 4 | waits for `PILOT_392M_VERDICT.json` = PASS → 392M wikitext-mix-ext arm_off calibration (20,000 steps) |

GPUs 5–7 left free for the concurrent CAPABILITY STAGE 2 wave (§13.9's cross-wave ordering).
Supervisor: self-healing retry loop per bank (transient failures retry at 60 s; terminal states —
completed, timed_out, pilot-FAIL refusal — never spin); resume-safe by output-JSON validity, not
existence; clean stop via `touch /home/nvidia/chapter2/deltanet_rd/STOP_fixscale` (never pkill).

**Timing-pilot method:** the house two-point convention (rate = (wall₁₀₀₀−wall₂₀₀)/800, startup
overhead cancels). Gate: BOTH arms' steady-state s/step ≤ 1.5× the archived Track-C realized rate
(§13.7 corrected: 98M **0.236 s/step**, 392M **0.836 s/step** — the 6× per-step error is NOT
inherited). Verdict JSON records per-arm s/step, peak VRAM GB (trainer's own
`peak_memory_*_bytes`), and the per_token/off blend-overhead ratio. Pilot outputs are
timing-measurement throwaways — excluded from every span_frac/val-loss analysis by construction
(the per_token pilots precede any blind pin but are never read as measurement cells).

**Calibration cells (arm_off, seed 0):** anchored to Track C's archived val-loss bands
(EXPERIMENT_LOG.md:3980-3984, 4046-4047): 98M openr1 **1.290** / wikitext **3.189**; 392M openr1
**1.135** / wikitext **2.847** — 392M cells run 20k of Track C's 91,552 steps, so their final val
loss should read HIGHER than the archived anchor, not at it (§13.10 item 4's own caveat).
Manual watchdog (the §13.8 1.5× breaker is NOT built yet — task-acknowledged): a thread appends
(unix_time, step) to `<cell>_rate_watch.csv` every 120 s (~500 steps at the 98M reference rate),
logging a loud WARN row on any >1.5× interval rate, never aborting; the harvest re-checks every
row. Crude runaway ceiling: `--internal-timeout 36000` (10 h ≈ 2.2× expected wall) per cell —
the trainer self-terminates cleanly (`timed_out: true`, flagged, not silently re-run).
Exact flag set mirrors `run_lm_rd_trackc_sweep.py`'s `build_cmd_cell` (batch 32, seq 512,
ckpt-every 1000, stock lr/warmup/eval defaults) for band comparability; `--frozen-bias-arm off`
passed explicitly (bit-identical to the pre-frozen-bias path per smoke [17]).

## Cost (vs the 300 GPU-h fix-at-scale ledger)

| Item | GPU-h (projected at §13.7 rates) |
|---|---|
| Timing pilots, 8 runs (2 scales × 2 arms × 2 points) | ≈0.9 (incl. startup overhead) |
| 98M arm_off calibration × 2 corpora (67,547 steps) | ≈8.96 |
| 392M arm_off calibration × 2 corpora (20,000 steps) | ≈9.34 |
| **This launch, committed total** | **≈19.2** |

≈6.4% of the 300 cap. The 4 calibration cells are not throwaway spend: they are 4 of the full
manifest's own `arm_off` training cells (seed 0, both corpora, both scales) — the build-stage
sweep skips them as already-complete (resume-safe convention), so this spend sits INSIDE §13.7's
≈244.44 GPU-h (2×) primary-default projection, not on top of it. Realized rates land in
`PILOT_*_VERDICT.json` and supersede these projections for the build-stage re-price.

## Expected wall / monitor lines

Expected: 98M pilot verdict ≈25 min after launch; 392M ≈55 min; calibration cells ≈4.5–4.8 h each
(all four done ≈5.5–6 h after launch, all four GPUs releasing by then).

```
ssh youthful-indigo-turkey 'tmux ls; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
ssh youthful-indigo-turkey 'cat /home/nvidia/chapter2/deltanet_rd/results/fixscale/pilots/PILOT_98M_VERDICT.json /home/nvidia/chapter2/deltanet_rd/results/fixscale/pilots/PILOT_392M_VERDICT.json'
ssh youthful-indigo-turkey 'tail -5 /home/nvidia/chapter2/deltanet_rd/results/fixscale/calib/*.log; tail -3 /home/nvidia/chapter2/deltanet_rd/results/fixscale/calib/*_rate_watch.csv'
ssh youthful-indigo-turkey 'ls -la /home/nvidia/chapter2/deltanet_rd/results/fixscale/calib/'   # *.json complete | *.REFUSED
# clean stop (never pkill): ssh youthful-indigo-turkey 'touch /home/nvidia/chapter2/deltanet_rd/STOP_fixscale'
```

## ADDENDUM — realized pilot results (gate §13.10 item 3 DISCHARGED, both scales, same session)

Both timing pilots completed and **PASSED**; the four calibration cells auto-launched behind
their verdicts (verified live: GPUs 1–4 all training, `attrrob_2x2` on GPU 0 undisturbed).
Verdict JSONs (source of truth, on box): `results/fixscale/pilots/PILOT_{98M,392M}_VERDICT.json`.

| Scale | arm | two-point s/step | vs Track-C ref | 1.5× gate | peak VRAM alloc (GB) |
|---|---|---|---|---|---|
| 98M | off | **0.2361** | 0.236 (+0.05%) | PASS | 23.2 |
| 98M | per_token | **0.2379** | +0.8% over off | PASS | 23.5 |
| 392M | off | **0.8215** | 0.836 (−1.7%) | PASS | 38.3 |
| 392M | per_token | **0.8311** | +1.2% over off | PASS | 39.0 |

**§13.7's no-overhead-from-the-blend assumption CONFIRMED at both new scales** (blend overhead
+0.8%/+1.2%, far inside the 1.5× band; VRAM headroom ≥41 GB on 80 GB cards both scales — the
§13.13 Rev-1 item-2 VRAM mandate is satisfied and batch=32 is nowhere near the memory wall).
Realized pilot spend: **0.724 GPU-h** (8 runs, sum of trainer `wall_s`). Re-priced calibration
projection at measured rates: 98M ≈4.43 GPU-h/cell, 392M ≈4.56 GPU-h/cell → this launch's
committed total ≈ **18.7 GPU-h** (≈6.2% of the 300 cap). Build-stage re-price of the full
manifest should use the verdict-JSON rates, which slightly UNDERCUT the §13.7 planning rates.

## Still needs the BUILD stage (not launched, per scope)

1. Scale-parametrized sweep launcher (or frozen_bias_lm_sweep.py generalization) — the 24-cell
   n=3/n=3 primary manifest + 4 `arm_global_probe` cells (§13.3 Rev 1).
2. Blind-pin writers `BANDS_PINNED-FrozenBias-{98M,392M}.json` + `assert_blind_not_broken` wired
   into the launcher (§13.10 item 5 — pin BEFORE any manifest `arm_per_token` training starts;
   §13.13 binding item 3).
3. The 1.5× per-cell circuit breaker (§13.8) — supervisor-level, ABORT_1.5X.json sentinel.
4. Wave −1 instrument-validity + code-path-equality smoke at `d_state=128` (§13.10 item 1) — the
   pre/post-blend bit-identity check at the genuinely new shape; the 392M per_token pilot
   exercises the arm's forward/backward for real but is NOT that bit-identity check.
5. Eval-only retrofit passes (arm_off′) + their own timing check (§13.11 item 7).
