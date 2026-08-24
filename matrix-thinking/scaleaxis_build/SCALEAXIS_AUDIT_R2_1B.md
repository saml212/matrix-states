# SCALEAXIS AUDIT R2 — THE 1.31B RUNG-3 EXTENSION (DELTA ONLY)

**Auditor:** independent audit agent (not the builder). Everything below was
RE-EXECUTED or RE-DERIVED from raw artifacts; nothing is taken from the
builder's prose.
**Scope:** the DELTA introduced by `79b3d41` / `d0097ca` / `db9705f`. The
scale-axis machinery through 392M is out of scope (SCALEAXIS_AUDIT_R1 +
its conditions).
**Box evidence sampled:** 2026-08-23T23:54–23:56Z, read-only.
**Gates:** the 24-cell 1B sweep, ids 0370–0393, ~269 GPU-h as declared.

---

## VERDICT: **REV-REQUIRED (narrow)**

The rung-3 machinery is sound. Config provenance, the parameter formula, the
B3 fix, command fidelity, validity paths, ID hygiene and the thickening
ceilings all re-derive **exactly**. There is **no burn risk, no OOM risk and
no ceiling-abort risk** anywhere in the wave.

Three things must be resolved before the sweep is queued or harvested. Only
one of them is spec content, and it has a **zero-code fix**:

| # | Finding | Class | Fix cost |
|---|---|---|---|
| **R2-1** | Specs **0382 / 0385 alias the calibration pair** — identical `--cell-id`, `--out`, `--ckpt-dir` | spec content | 0 lines (elect not to queue 2 files) or 1 field |
| **R2-2** | **6 GPUs go idle ~04:25Z Aug 24** — holdback 0308–0323 never promoted | operational, live | promote 16 files |
| **R2-3** | The calibration pair **is not running** — the recorded schedule did not execute | record correction | log line |

**If the coordinator elects R2-1 option (b) — simply do not queue 0382 and
0385 — this audit converts to PASS-LICENSE-SWEEP with no code change at
all**, subject to the ledger being restated (R2-4) and the record corrected
(R2-3).

---

## 1. RUNG-3 CONFIG PROVENANCE — **PASS**

`matrix-thinking/deltanet_rd/lm_rd_rung_configs.py` `RUNGS[3]`:

```
3: {"d_model": 2560, "n_layers": 22, "d_state": 128,
    "approx_params": 1_310_000_000, "target": "~1.3B"}
```

`patched/kscaling_config.py:177` `RUNGS[3]`:

```
3: dict(d_model=2560, d_state=128, n_layers=22, conv_size=4, num_heads=1, ffn_mult=4)
```

Byte-identical on all three load-bearing values. The three extra keys
(`conv_size=4, num_heads=1, ffn_mult=4`) are the NCR dict's own shape fields;
`lm_rd_rung_configs.verify_param_count()` passes exactly those same values as
`DeltaNetLM(..., conv_size=4, num_heads=1, ffn_mult=4)`, so they are the
repo's values too, not new literals. **Config is the repo's own, verbatim.**

### Hand re-derivation of sec 3.4 at rung 3

```
per_layer = 2·4·2560²  +  4·2560·128  +  3·128·4  +  2560  +  (2·2560 + 128)
          = 52,428,800 +   1,310,720  +    1,536  +  2,560  +      5,248
          = 53,748,864

backbone(50259) = 50,259·2560 + 22·53,748,864 + 2560
                = 128,663,040 + 1,182,475,008 +  2,560   =  1,311,140,608

ncr(h=64, K)  = 40·64² + 4·(K+1)·64 + 46·64 + (K+1)  =  166,784 + 257·(K+1)
integ(K)      = 2·2560·(K+1)
```

| K | ncr head | integ | **total/arm** | claimed |
|---|---|---|---|---|
| 16 | 171,153 | 87,040 | **1,311,398,801** | 1,311,398,801 ✓ |
| 24 | 173,209 | 128,000 | **1,311,441,817** | 1,311,441,817 ✓ |
| 32 | 175,265 | 168,960 | **1,311,484,833** | 1,311,484,833 ✓ |
| 40 | 177,321 | 209,920 | **1,311,527,849** | 1,311,527,849 ✓ |

**Independent validation of the arithmetic path:** the same formula, run at
rung 2, reproduces `TOTAL_PARAM_TABLE_392M` (392,095,889 / 392,122,521 /
392,149,153 / 392,175,785) at all four K exactly — so the code path that
produced the rung-3 numbers is the one already validated against four
independently-measured endpoints.

**Hardware validation:** `smoke_results/gates_1b_K16.json` / `_K40.json` item
`B3_measured_param_counts_vs_formula` records
`measured_total_per_arm = 1311398801 / 1311527849`,
`measured_backbone = 1311140608` both, PASS. Real `nn.Module` numel on CUDA,
not a projection.

---

## 2. THE B3 FIX — **PASS** (two hygiene residues, §9)

The defect was `scaleaxis_gates.py:275`, a two-rung ternary
`(TOTAL_PARAM_TABLE_392M if SCALE == "392m" else TOTAL_PARAM_TABLE_98M)`,
which fell through to the 98M table at any new scale. The fix routes through
`KS.total_param_table()`:

```python
def total_param_table() -> dict:
    return {1: TOTAL_PARAM_TABLE_98M, 2: TOTAL_PARAM_TABLE_392M,
            3: TOTAL_PARAM_TABLE_1310M}[RUNG]
```

### Re-executed: what EVERY rung's selector returns

Imported `kscaling_config` fresh at all 12 (scale, K) combinations and
compared the selected table against `total_param_exact()`:

| scale | rung | table returned | K=16 | K=24 | K=32 | K=40 | table == total_param_exact() |
|---|---|---|---|---|---|---|---|
| 98m | 1 | `TOTAL_PARAM_TABLE_98M` | 97,816,977 | 97,831,321 | 97,845,665 | 97,860,009 | ✓ 4/4 |
| 392m | 2 | `TOTAL_PARAM_TABLE_392M` | 392,095,889 | 392,122,521 | 392,149,153 | 392,175,785 | ✓ 4/4 |
| 1310m | 3 | `TOTAL_PARAM_TABLE_1310M` | 1,311,398,801 | 1,311,441,817 | 1,311,484,833 | 1,311,527,849 | ✓ 4/4 |

**12/12 correct. No rung can now reach another rung's table.**

### Re-executed: does a hypothetical rung 4 fail loudly?

Injected `RUNGS[4]`, `RUNG_OF_SCALE["2800m"]=4`, `RUNG=4`:

```
LOUD: KeyError 4
```

The process dies; it cannot silently use a wrong table. Two adjacent guards
also re-fired correctly:

- unknown `NCR_SCALE=2800m` → `AssertionError: NCR_SCALE='2800m' is not one of
  ['1310m','392m','98m'] … refusing to guess.`
- unported `NCR_K=36` at `1310m` → `AssertionError: NCR_K=36 is not one of the
  FOUR ported K (16, 24, 32, 40)`

The smoke's backbone assert is genuinely table-driven now
(`RUNG1_BACKBONE == _KSCFG.RUNGS[_KSCFG.RUNG]`, plus an explicit
`{1:(768,64,12), 2:(1536,128,16), 3:(2560,128,22)}[RUNG]` cross-check) — so
rung 3 introduced no new backbone literal, as claimed.

---

## 3. PROBE NUMBERS — **PASS**, with a ledger caveat (R2-4)

### The measured points are real

`smoke_results/vram_1b_K16.json` / `_K40.json`, real CUDA, batch 32, two arms:

| K | t_in | s/step | peak GB (train == with-eval) | proj GPU-h/20k | oom |
|---|---|---|---|---|---|
| 16 | 128 | 1.3355 | 49.913 | 7.419 | false |
| 40 | 286 | 2.7445 | 64.17 | 15.247 | false |

`measured_params_per_arm == total_param_exact_per_arm`, `params_match_formula:
true` in both.

### FLOPs plausibility vs the 392M measurements — **strongly consistent**

Achieved throughput, computed as `2 arms · 6 · N · T · B / s_per_step`:

| model | K | N | T | s/step | achieved |
|---|---|---|---|---|---|
| 392M | 40 | 3.9218e8 | 286 | 0.8280 (realized) | **52.0 TFLOP/s** |
| 1.31B | 40 | 1.3115e9 | 286 | 2.7445 (probe) | **52.5 TFLOP/s** |

Same harness, same efficiency. Param ratio is 3.344; measured s/step ratio is
3.23 (K=16) / 3.31 (K=40) — mildly *sub*-linear, exactly what larger GEMMs
should give. Peak-VRAM ratios 49.913/17.094 = 2.92× and 64.17/23.460 = 2.74×
against 3.34× params — sub-linear because the embedding/vocab share carries no
activation. **Nothing implausible. Nothing to flag on the physics.**

### K=24 / K=32 are interpolations, correctly labelled

`RATE = {16: 1.3355, 24: 1.7457, 32: 2.2451, 40: 2.7445}` is exactly linear in
`t_in`:

```
slope = (2.7445 − 1.3355)/158 = 0.0089177
K=24:  1.3355 + 0.0089177·46  = 1.74572 → 1.7457 ✓
K=32:  1.3355 + 0.0089177·102 = 2.24511 → 2.2451 ✓
```

Labelled as interpolations in four places per spec: the `hypothesis`
(`{measured}` → "INTERPOLATED in t_in from the K=16/K=40 measurements, FLAGGED
as an interpolation"), `notes` (`~54.1 GB (interpolated)`), `rate_basis`, and
`ceiling_provenance` ("1.5 x this cell's own **interpolated** 20,000-step
projection"). `peak_gb_batch32` is `null` at K=24/32 rather than carrying a
fabricated measurement. **This is honest labelling, and audit m5's lesson is
respected — this genuinely is an interpolation, not a re-labelled assignment.**

**The interpolation METHOD is empirically validated against 392M's four
measured points**, which the builder did not claim but which strengthens it:

| K | linear-in-t_in prediction from K16/K40 endpoints | 392M measured | error |
|---|---|---|---|
| 24 | 0.53437 | 0.51620 | **+3.5% (conservative)** |
| 32 | 0.68120 | 0.68455 | −0.5% |

### Ceilings are per-cell probe-derived — **PASS**

| K | own 20k projection | ×1.5 → `--ceiling-gpuh` |
|---|---|---|
| 16 | 7.419 | **11.128** |
| 24 | 9.698 | **14.547** |
| 32 | 12.473 | **18.71** |
| 40 | 15.247 | **22.87** |

`ceiling_provenance` on all 26 specs reads "Priced from THIS BUILD'S OWN
real-CUDA probe, **not from any 98M/392M multiplier**". Verified: no 98M/392M
factor appears in the ceiling path. Confirmed by reading the emitted field on
all 26, not by reading the generator alone.

---

## 4. SPECS 0360–0361 + 0370–0393 — **PASS except R2-1 / R2-6 / R2-7**

### Command cross-check vs the 392M spec of record — **PASS**

Mechanically parsed the flag-map of all 26 1B commands and diffed against
392M `0200_ncr_scaleaxis_392m_K40_primary_s0.json`.

- **Extra flags: 0 in all 26.**
- **Missing flags: only `--freeze-entity-adapter` on the 13 compB cells** —
  correct, compB *is* the trainable-adapter recipe.
- **Every differing value lies in
  `{--scale, --ceiling-gpuh, --cell-id, --out, --ckpt-dir, --k, --seed,
  --ckpt-every}`.** Zero unexpected differences.

Byte-identical across all 26 and the 392M reference:
`--mode calibration --device cuda --steps 20000 --batch-size 32
--eval-batch-size 64 --warmup-steps 200 --lr 3e-4 --aux-read-loss-weight 0.5
--ortho-reg-weight 0.1 --eval-every 1000 --aux-loss-type contrastive+cosine
--contrastive-temperature 0.07`. `--ckpt-every` differs only on the
calibration pair (5000 vs 10000), which is stated and intended.

### validity_check paths — **PASS, one clause lost (R2-7)**

- Top-level `steps_target == 20000` present on all 26 ✓. Verified the field
  really is top-level in the runner's record: `ncr_lm_wave1_runner.py:1326`
  writes `steps_target=steps` at record top level (with its own comment saying
  so), and a real result JSON
  (`smoke_results/LIT3_scaleaxis392m_K40_primary_s0.json`) has
  `steps_target` at top level, not under `config`. #1's field-path bug cannot
  recur.
- From-scratch Gate-0 `h[-1][1] < h[0][1]` on all 26 ✓ — correct, these are
  fresh runs, so #2's resumed-segment scoping does not apply.
- New rung-3 clauses present: `ks['rung'] == 3`,
  `(d_model, d_state, n_layers) == (2560, 128, 22)`,
  `params.per_arm == <rung-3 table value>` ✓.
- Every asserted key exists in real runner output (checked field-by-field
  against a real result JSON): `status`, `step`, `steps_target`, `runner_tag`,
  `kscaling.{K,d_ncr,h_top,deep_ladder,scale,rung,backbone}`, `params.per_arm`,
  `loss_history.{full_graft,backbone_only}` ✓.

**R2-7 (LOW):** the 1B validity **drops** `assert ks.get('d_equals_k_plus_1')
is True`, which both 392M `0200` and thickening `0300` carry. One clause fewer
than the spec of record. Minimal fix: re-add the clause in
`gen_1b_specs.validity()`.

### LICENSE_SWEEP_1B sentinel — **structurally PASS, textually leaky (R2-6)**

The machine-readable field is `scaleaxis1b.gate`:

| specs | `scaleaxis1b.gate` |
|---|---|
| 0360, 0361 (calibration) | `"build audit + memory/ledger ruling"` — **sentinel ABSENT** ✓ |
| 0370–0393 (24 sweep) | `"build audit + memory/ledger ruling + LICENSE_SWEEP_1B"` — **present, 24/24** ✓ |

**R2-6 (LOW-MED):** the literal token `LICENSE_SWEEP_1B` nonetheless appears
**once** in the calibration specs' `hypothesis` prose (`CAL_NOTE`, describing
what the calibration produces), so
`grep -l LICENSE_SWEEP_1B job_specs_1b/*.json` matches **all 26**. The 392M
precedent kept this clean: `0190` contains **zero** occurrences of
`LICENSE_SWEEP_SCALEAXIS`; `0200` contains 2, both in `notes`. Any text-grep
selection step that worked at 392M now silently includes the calibration pair.
Minimal fix: in `CAL_NOTE`, replace the literal token with "the sweep license
sentinel".

### ID collisions vs full queue history — **PASS**

Box queue at `/home/nvidia/queue/` — pending 2, claimed 8, completed 534,
failed 0, cancelled 0, holdback 16, parked_k24plus 38 (598 spec files;
**188 unique 4-digit ids**).

| range | on box | verdict |
|---|---|---|
| 0300–0323 | claimed(8) + holdback(16) | this build's own files, **md5-identical 24/24** to `job_specs_thicken/` — deployment, not collision |
| 0360–0361 | pending(2) | md5-identical to `job_specs_1b/` — deployment, not collision |
| **0370–0393** | **absent** | **collision-free; nothing queued** ✓ |
| 0330–0341 | absent | live in `matrix-thinking/kscaling_build/job_specs_thicken/` (a different build tree) |
| 0350–0352 | absent | live in `job_specs_v2prime_thicken/` |

Also confirmed no 0370–0393 anywhere in the local repo outside
`job_specs_1b/`. The generator's own `--queue-ids` assert **has teeth** —
re-executed against a synthetic collision file: `AssertionError: ID COLLISION:
['0370']`.

*Informational:* a legacy 3-digit id namespace (`000_`–`723_`, incl. `300_`–
`315_`) coexists in `completed/`. Filenames are distinct strings (`0300_` vs
`300_`), so there is no file-level collision — but any tool parsing ids with a
loose numeric regex could conflate them.

### Longest-first naming — **PASS**

Filename sort of `job_specs_1b/` gives sweep K order
`40,40,40,40,40,40, 32,32,32,32,32,32, 24,24,24,24,24,24, 16,16,16,16,16,16`.
Filename order *is* the schedule control, as sec 8.3 requires.

### Generator determinism — **PASS**

Re-ran `gen_1b_specs.py` into a scratch directory: **26/26 files byte-identical**
to the committed ones. No hidden state, no environment dependence, nothing
that drifts on regeneration.

---

### ⚠ R2-1 (HIGH) — 0382 and 0385 ALIAS the calibration pair

Four specs share two cell identities:

| spec | tier | `--cell-id` | `--out` | `--ckpt-dir` |
|---|---|---|---|---|
| **0360** | calibration | `scaleaxis1310m_K24_primary_s0` | `…/results/scaleaxis1310m_K24_primary_s0.json` | `…/ckpts/scaleaxis1310m_K24_primary_s0` |
| **0382** | **sweep** | **identical** | **identical** | **identical** |
| **0361** | calibration | `scaleaxis1310m_K24_compB_s0` | `…/results/scaleaxis1310m_K24_compB_s0.json` | `…/ckpts/scaleaxis1310m_K24_compB_s0` |
| **0385** | **sweep** | **identical** | **identical** | **identical** |

**Root cause.** At 392M, K=24 was the calibration *sextet* (0190–0195, both
recipes × seeds 0,1,2) and was **excluded from the sweep** — re-running the
392M generator confirms its sweep tier is `[40×6, 32×6, 16×6]`, 18 cells, no
K=24. At 1B the builder shrank the calibration to K=24 × 2 recipes × **seed 0**
and then also put K=24 **seeds 0,1,2** into the sweep. Seed 0 collides.

**Runtime behaviour — verified, not assumed.**
`ncr_lm_wave1_runner.py:1249-1252`, the first statement of
`run_two_arm_cell()`:

```python
if os.path.exists(out_path):
    prev = json.load(open(out_path))
    if prev.get("status") == "COMPLETED":
        print(f"[{cell_id}] already COMPLETED -- skipping (resume-safe)")
        return prev
```

So 0382/0385 will **not** clobber and **not** re-train. They return the
calibration record at ~zero cost. **This is not a burn risk and not a data-loss
risk.** What it *is*:

1. **The declared ledger is wrong.** `"24-cell sweep = 269.0 GPU-h"` (in every
   spec's `scaleaxis1b.ledger_note`, in the commit message, and in
   EXPERIMENT_LOG 2026-08-24 #2) overstates by **19.40 GPU-h**. True
   probe-rate cost is **249.67**.
2. **Two of twenty-four "sweep cells" produce no new data.** A harvest keyed on
   spec id would read 0360 and 0382 as two independent records when they are
   one file. Under Ruling 2 the K=24 stratum feeds the pinned pairwise
   instruments — this must be declared before, not after, the harvest.
3. **0382/0385's own `--ckpt-every 10000` is never honoured** — the artifact
   that satisfies them was written at the calibration's 5000 cadence.

**Minimal fixes (elect one):**

- **(b) — zero code.** Do not queue 0382 and 0385; queue 22 sweep specs.
  Record: "the sweep's K=24 seed-0 arm IS the calibration pair 0360/0361."
  Ledger becomes 249.67 GPU-h. **Recommended** — it needs no regeneration and
  no re-audit.
- **(a) — one field.** In `gen_1b_specs.spec()`, for `k == 24 and seed == 0 and
  tier == "sweep"`, set `gpu_h_estimate = 0.0` and add
  `"aliases_calibration_cell": "0360"` / `"0361"`; restate the ledger.

Do **not** re-seed 0382/0385 to seeds 3/4 — that changes the pre-registered
design (n=3 per cell) and would need its own ruling.

---

## 5. THICKENING-WAVE DELTA

### Ceilings vs measured sibling rates — **PASS, all 8 exact**

Recomputed `1.5 × R₈(1.0026) × mean(gpu_h of the three n=3 siblings)` from the
24 archived cells in `experiment-runs/2026-08-22_scaleaxis_sweep/`:

| (K, recipe) | n | measured mean gpu_h | recomputed ceiling | spec `--ceiling-gpuh` | |
|---|---|---|---|---|---|
| (16, compB) | 3 | 2.2937 | 3.449 | 3.449 | ✓ |
| (16, primary) | 3 | 2.3032 | 3.464 | 3.464 | ✓ |
| (24, compB) | 3 | 2.8663 | 4.311 | 4.311 | ✓ |
| (24, primary) | 3 | 2.8692 | 4.315 | 4.315 | ✓ |
| (32, compB) | 3 | 3.8214 | 5.747 | 5.747 | ✓ |
| (32, primary) | 3 | 3.7842 | 5.691 | 5.691 | ✓ |
| (40, compB) | 3 | 4.5948 | 6.910 | 6.910 | ✓ |
| (40, primary) | 3 | 4.6056 | 6.926 | 6.926 | ✓ |

**8/8 exact.** The corrected ledger also reproduces: sum over 24 thickening
cells = **81.415 GPU-h**, mean **3.3923/cell** — the builder's ledger
correction (fresh ≠ resume) is **confirmed against the raw archive**. Note for
the record: 3.392 is the four-K *average*; actual per-cell cost runs 2.298
(K=16) → 4.600 (K=40). Their `validity_check` correctly appends the top-level
`steps_target` clause **and** retains `d_equals_k_plus_1`.

### ⚠ R2-5 (MEDIUM) — the near-miss "guard" is INERT

The claim (commit `d0097ca`, EXPERIMENT_LOG #2): *"caught by diffing before
commit, restored via git checkout, and a **docstring guard** added naming the
hazard."*

**Re-executed in a scratch copy of the whole build tree.** Ran
`gen_scaleaxis_specs.main()` bare (no `--ceilings-from`), exactly as the
near-miss did:

```
wrote 24 CANDIDATE specs to …/job_specs (6 calibration sextet + 18 sweep)
ceiling provenance: PROJECTED-NOT-LAUNCH-READY: 3.795 x projected solo …
```

**All 24 as-run specs were overwritten and their ceilings reverted:**

| spec | as-run (repo) | after bare regen | |
|---|---|---|---|
| 0190 | 3.989 | **11.772** | REVERTED |
| 0200 | 6.586 | **16.095** | REVERTED |
| 0206 | 4.621 | **13.639** | REVERTED |
| 0212 | 3.867 | **11.412** | REVERTED |

**The "guard" is a docstring at `gen_scaleaxis_specs.py:451-458`. It names the
hazard and prevents nothing.** The hazard is exactly as live as it was before
the near-miss.

What *does* work, and was verified separately: the **separation**.
`main_thicken()` with a bad `--sweep-dir` exits cleanly
(`SystemExit: --sweep-dir '/nonexistent' does not exist …`) and left
`job_specs/` md5-unchanged. The thickening wave genuinely cannot touch the
completed wave's directory.

This is **not on the 1B critical path** (different generator, different
directory) — it is ranked MEDIUM because the record claims a protection that
does not exist.

**Minimal fix — 5 lines, immediately after `args = ap.parse_args()` in
`main()`:**

```python
if not args.ceilings_from and glob.glob(os.path.join(OUT, "*.json")):
    raise SystemExit(
        "REFUSING to regenerate job_specs/ without --ceilings-from: it holds the AS-RUN "
        "0190-0217 specs with Stage-A0 RE-PRICED ceilings (3.867-6.586). Re-running "
        "without --ceilings-from reverts them to the PROJECTED placeholder and destroys "
        "the as-run record. Pass --ceilings-from <stage-A0 dir>, or move job_specs/ aside.")
```

---

## 6. THE WALL-CLOCK MATH

### The ~36.1 h figure is arithmetically right for the wrong quantity

`gen_1b_specs.main()` prints `wall on 8 GPUs ~{(cal+swp)/8:.1f} h` =
288.418/8 = **36.05 h**. That is *(calibration + sweep)/8*, an average-load
figure, **not** the sweep's makespan. EXPERIMENT_LOG #2 reports it as "the
24-cell 1B sweep … takes the full box ~36h". Sweep-alone/8 is **33.63 h**.

**True longest-first makespan on 8 GPUs (list-scheduled in filename order):**

| basis | sweep GPU-h | makespan |
|---|---|---|
| probe rates, no alias | 269.02 | **35.14 h** |
| probe rates, with R2-1 alias | 249.67 | **32.38 h** |
| β-corrected, no alias | 286.45 | **37.28 h** |
| β-corrected, with R2-1 alias | 265.57 | **34.55 h** |

So "~36 h" lands close to the truth, but by two errors partly cancelling
(total/8 understates the makespan; the probe understates the rate; the alias
overstates the work). Longest-first is doing its job — the counterfactual
shortest-first makespan is 37.42 h, so the ordering buys ~2.3 h.

### ⚠ R2-4 (MEDIUM) — the ledger is a FLOOR, not an estimate

The specs are priced from the **probe** instrument. At 392M the probe
systematically under-reads realized cost, measured against the 24 archived
cells:

| K | probe s/step | realized s/step | β = probe/realized |
|---|---|---|---|
| 16 | 0.3793 | 0.413716 | **0.9149** |
| 24 | 0.4815 | 0.516199 | **0.9275** |
| 32 | 0.6470 | 0.684507 | **0.9440** |
| 40 | 0.7911 | 0.828036 | **0.9552** |

Applying β to rung 3:

| K | priced GPU-h | β-corrected | ceiling | ceiling utilisation |
|---|---|---|---|---|
| 16 | 7.419 | **8.11** | 11.128 | 72.9% |
| 24 | 9.698 | **10.46** | 14.547 | 71.9% |
| 32 | 12.473 | **13.21** | 18.71 | 70.6% |
| 40 | 15.247 | **15.96** | 22.87 | 69.8% |

**Ceilings hold with 27–30% headroom at every K — there is no abort risk.**
This is a planning correction only. Read the sweep as **~265–286 GPU-h**, not
269.02.

Two smaller biases point the same direction and are worth naming since all
three compound:

- the two 1.31B instruments disagree — vram probe 1.3355/2.7445 vs gates B8
  1.3375/2.7626 (+0.15%/+0.66%) — and the build quoted the **lower**;
- the build wrote `64.17` where B8 measured `64.177` (truncation, not
  rounding).

β was never measured at rung 3, and rung-3 checkpoints are ~31.5 GB against
392M's 8.76 GB, so checkpoint I/O could push β *down* at this rung, not up.

### ⚠ R2-3 (HIGH, record correction) — the calibration pair is NOT running

EXPERIMENT_LOG #2's schedule: *"NOW — calibration pair 0360/0361 on 2 GPUs
(~14h) + the 392M thickening block 0300-0323 on the other 6."*

**Box state at 2026-08-23T23:56Z:** all 8 workers claimed **thickening** cells
0300–0307 at 23:49–23:50Z. `0360`/`0361` sit in `queue/pending/`.
`/ephemeral/scaleaxis1b/results/` and `/ephemeral/scaleaxis1b/ckpts/` are
**empty**. No process, no log, **zero steps executed**.

The "~14 h" calibration window is wrong twice over: 13.6 h is the *thickening
block's* wall, and the calibration pair's own wall is **9.70 h** (probe) /
**10.46 h** (β) on 2 GPUs.

### ⚠ R2-2 (HIGH, live operational) — 6 GPUs idle at ~04:25Z Aug 24

`queue/pending/` holds **only** 0360/0361. `queue/holdback/` holds 0308–0323
(16 cells). Using the measured per-cell 392M costs:

| event | cells | duration | ETA (from 23:49Z Aug 23) |
|---|---|---|---|
| 0306/0307 (K32 primary) finish → claim 0360/0361 | 2 | 3.784 / 3.787 h | **~03:36Z Aug 24** |
| 0300–0305 (K40) finish → **pending is EMPTY** | 6 | ~4.60 h | **~04:25Z Aug 24** |
| calibration 0360/0361 complete | 2 | 9.70–10.46 h | **~13:20–14:05Z Aug 24** |

Unless 0308–0323 are promoted holdback→pending, **six H100s idle for ~9 h** —
a direct breach of the never-idle standing rule adopted in the same log entry.
The 16 held-back cells are ~46.2 GPU-h ≈ 5.8 h of box time, which fits the gap
almost exactly.

### Realistic completion

| milestone | earliest | latest |
|---|---|---|
| calibration complete | 13:20Z Aug 24 | 14:05Z Aug 24 |
| license adjudicated + 2 cells scored | ~15:00Z Aug 24 | ~16:00Z Aug 24 |
| sweep makespan | 32.4 h | 37.3 h |
| **last sweep cell lands** | **~23:30Z Aug 25** | **~05:20Z Aug 26** |
| + per-cell battery scoring tail | — | — |

**"~Aug 26" HOLDS — but with no slack.** It holds only if (i) the license
handoff is immediate, (ii) the holdback is promoted so the box does not idle,
and (iii) β does not degrade at rung 3. **A single re-run of one K=40 cell
costs ~16 h of wall on one GPU** — there is no room for one.

---

## 7. GATE COVERAGE — R2-8 (LOW)

392M ran `scaleaxis_gates` at **all four** K
(`smoke_results/gates_K{16,24,32,40}.json`). 1B ran **two**
(`gates_1b_K16.json`, `gates_1b_K40.json` — 20 PASS / 0 FAIL / 0 N/A each,
every negative test fired, including `B5` refusing a real 98M checkpoint under
the 1310m config with its matched-scale positive control, and `A0.2`
re-measuring `MIN_KERNEL_T` at rung 3).

The two that ran **bracket the range**, and every gated quantity at K=24/32 is
strictly interior:

- `A0.2` / `MIN_KERNEL_T`: K=16's `t_in` is **exactly 128**, the tightest
  possible case, and it passed. K=24 (174) and K=32 (230) have wide margin.
- `B8` / P4 memory: 49.913 → 64.177 GB brackets the interpolated 54.1 / 59.1.
- `B3` param arithmetic at K=24/32: re-derived by hand above, exact.
- `B1/B2/B4/B5/B6/B7`: K-independent.

**Acceptable on the merits.** But the cell the LICENSE rests on — K=24 — has
no gate record at its own K, and the narrowing is not recorded anywhere.
Minimal fix: either run
`NCR_SCALE=1310m NCR_K=24 python3 scaleaxis_gates.py` on the first free GPU
before licensing (~1 min of GPU per the K=16/40 elapsed times), or record the
narrowing plus the interiority argument in `DEVIATIONS.md`.

---

## 8. THE THREE "DOES NOT TRANSFER CLEANLY" FLAGS — all correctly raised

The builder flagged three items for ruling rather than assuming them. All
three are real, all three were correctly characterised, and all three were
ruled on in EXPERIMENT_LOG 2026-08-24 #2 **before** this audit. Confirmed:

- **(i) P4's 40 GB gate FIRES.** `P4_reading_gb` 49.913 / 64.177, both > 40.
  Correctly called not-a-blocker on an 80 GB part (~15.8 GiB headroom at
  K=40) and correctly characterised as *foreclosing* packing rather than
  declining it. Ruling 1 accepted it as informational. ✓
- **(ii) sec 5's cross-scale instruments are PAIRWISE.** Confirmed: TEST-X is
  98M-vs-392M over 8 strata, δ=0.05 is a two-point equivalence margin, and no
  three-point trend statistic is pre-registered. The specs assert **no**
  cross-scale verdict — verified by reading `cross_scale_instrument` on all
  26: `"NOT PRE-REGISTERED for three rungs (sec 5 is pairwise)"`. Ruling 2
  pinned the instruments as pairwise-identical pre-harvest, which is the
  correct disposition. ✓
- **(iii) the ledger is a new envelope.** Confirmed: 269 GPU-h is 2.4× the
  392M chapter and far past sec 8.2's ~112 tier-(c) boundary and its 130 GPU-h
  gate. Ruling 3 recorded the PI directive as the authorization the §8.2 tiers
  require. ✓ (Subject to the R2-1 and R2-4 restatements.)

**This flagging is the discipline working.** A builder that surfaced three
non-transferring items instead of quietly inheriting them is the reason this
audit is narrow.

---

## 9. HYGIENE RESIDUES — R2-9 / R2-10 / R2-11 (LOW)

**R2-9(a) — the B3 bug class survives at `scaleaxis_gates.py:311`:**

```python
KS.provenance(R.H_NCR, 768 if SCALE == "392m" else 1536)
```

The same two-branch ternary that caused B3's defect. It is **not a live
defect** — re-checked at all three rungs, the negative test fires correctly in
every case (98m: 1536 vs 768; 392m: 768 vs 1536; 1310m: 1536 vs 2560) — but it
is the un-generalized pattern, one rung away from silently passing. This is
now the **only** scale-conditional ternary left in the tree (grepped:
`SCALE ==`, `RUNG ==`, `scale ==`, `rung ==` across all `*.py`). Fix: derive
the deliberately-wrong `d_model` from the table, e.g.
`KS.RUNGS[(KS.RUNG % 3) + 1]["d_model"]`.

**R2-9(b) — `PORTED_K_GRID_1310M` is dead.** Defined at
`kscaling_config.py:184` and **never read**; the `1310m` branch asserts against
`PORTED_K_GRID_392M`. Harmless today (identical tuples) but it is a second
source of truth that nothing honours. Fix:
`assert K_NCR in {"392m": PORTED_K_GRID_392M, "1310m": PORTED_K_GRID_1310M}[SCALE]`.

**R2-9(c) — bare `KeyError: 4`.** `total_param_table()` fails loudly at a
hypothetical rung 4, which is what matters, but with no message — unlike every
other guard in the file, which explains itself. Fix: `assert RUNG in tbl, …`.

**R2-10 — `BOX_TREE_MD5.txt` is stale for two files.**
`gen_scaleaxis_specs.py` (pin+box `c17e42e6…`, local `17e2e3c9…`) and
`gen_v2prime_specs.py` (pin+box `9c36f546…`, local `b6e8c2d6…`) — the two files
`d0097ca` edited. `79b3d41`'s "repo mirror md5-identical to the box (6/6)" was
true for the six files the 1B build touched, but the manifest now carries two
stale rows. **Not a correctness issue**: generators run locally, `gen_1b_specs.py`
itself matches box+pin (`232d5095…`), the 28 runtime files under
`/home/nvidia/ncr_scaleaxis` are **28/28 clean**, and all 26 deployed job specs
are md5-identical box↔repo.

**R2-11 — "graft `bc105af6…`" is mislabeled.** That md5 belongs to
`~/ncr_g3b31_contrastive/ncr_lm_wave1_smoke.py`; no file named `*graft*` exists
in that directory. The runner pin `9a93198b…` is correct. Cosmetic, but the
mislabel is repeated verbatim in every one of the 26 specs' `notes`.

---

## 10. PROCESS NOTE — gate (a) ordering

Both calibration specs declare themselves **DOUBLE-GATED** on "(a) the build
audit" and "(b) the memory/ledger adjudication". Ruling (b) was recorded in
EXPERIMENT_LOG 2026-08-24 #2; gate (a) is **this document**. The pair was
staged to `queue/pending/` at 23:49Z, before (a) was discharged — permitted by
the coordinator's own recorded schedule ("the 1B delta AUDIT runs *during*
calibration and must land before licensing"), and moot in practice because the
pair has not started (R2-3). **This audit discharges gate (a) for 0360/0361.**
No deviation to record beyond noting the ordering.

---

## FINDINGS, RANKED

| # | Finding | Severity | Blocks sweep? | Minimal fix |
|---|---|---|---|---|
| **R2-1** | 0382/0385 alias calibration 0360/0361 (same cell-id/out/ckpt-dir); ledger overstated 19.40 GPU-h | **HIGH** | **YES** | Do not queue 0382/0385 (0 lines), or add an alias field + restate ledger |
| **R2-2** | 6 GPUs idle ~04:25Z Aug 24; holdback 0308–0323 never promoted | **HIGH** | no (breaches never-idle) | promote 16 files holdback→pending **now** |
| **R2-3** | Calibration pair not running (0 steps); "~14h window" wrong | **HIGH** | no (delays license ~3.7 h) | correct the log; license ETA 13:20–14:05Z Aug 24 |
| **R2-4** | Ledger is a floor: probe under-reads realized cost by 4.5–8.5% (β measured at 392M) | MED | no (ceilings hold, 70–73% util) | restate sweep as ~265–286 GPU-h |
| **R2-5** | The near-miss "docstring guard" is inert — re-executed, still reverts all 24 as-run ceilings | MED | no | 5-line `SystemExit` in `main()` |
| **R2-6** | `LICENSE_SWEEP_1B` literal leaks into calibration `hypothesis`; grep matches 26/26 | LOW-MED | no | drop the token from `CAL_NOTE` |
| **R2-7** | `validity_check` drops `d_equals_k_plus_1` vs the spec of record | LOW | no | re-add one clause |
| **R2-8** | Gate coverage narrowed 4 K → 2 K without a recorded deviation | LOW | no | run K=24 gate, or record in `DEVIATIONS.md` |
| **R2-9** | B3 bug class survives in `_b3n2` ternary; dead `PORTED_K_GRID_1310M`; bare `KeyError` | LOW | no | three one-liners |
| **R2-10** | `BOX_TREE_MD5.txt` stale for 2 generator files | LOW | no | re-pin |
| **R2-11** | "graft `bc105af6…`" is actually `ncr_lm_wave1_smoke.py` | INFO | no | fix the label in `NOTES` |

## WHAT RE-EXECUTED CLEAN

Rung-3 config provenance (byte-identical to `lm_rd_rung_configs.RUNGS[3]`) ·
sec 3.4 hand-derivation at all four K, exact · the same formula reproducing the
392M table exactly · B3's generalized selector at 12/12 (rung, K) combinations ·
rung-4 loud failure · unknown-scale and unported-K guards · probe rates
plausible against 392M at matched achieved throughput (52.5 vs 52.0 TFLOP/s) ·
K=24/32 interpolations exact, labelled in four places, and the method validated
against 392M's measured points · ceilings per-cell probe-derived with no
cross-scale multiplier · 26/26 command flag-maps clean against the 392M spec of
record · top-level `steps_target` verified against real runner output ·
from-scratch Gate-0 correct for fresh cells · sentinel structurally correct
24/24 present, 2/2 absent · IDs collision-free against 188 box ids, assert has
teeth · longest-first ordering correct · generator regenerates 26/26
byte-identically · all 8 thickening ceilings exact against measured siblings ·
81.415 GPU-h ledger correction confirmed against the raw archive · 28/28 box
runtime files md5-clean · gates 20 PASS / 0 FAIL at both K probed.
