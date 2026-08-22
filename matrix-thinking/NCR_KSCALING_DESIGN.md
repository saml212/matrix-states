# NCR K-SCALING SWEEP — DESIGN + BUILD, DRAFT-R0

**Status:** DRAFT-R0, BUILT AND SMOKE-CLEAN, **NOT LAUNCH-RELEASED.**
Awaiting the audit round (ceremony tier: 10–50 GPU-h ⇒ audit **+** pre-launch
resource/placement red-team). No cell has been queued. Nothing beyond the
smoke has been run.

**Build agent:** Opus, 2026-08-21. **Base commit:** `ad52dcf`.
**Gate:** `research/kscaling-novelty-2026-08-21.md` — ADJUDICATED CLEAR 3/3
(internal NOVEL-TO-US, by-mechanism NOVEL, by-task NOVEL). All five carried
requirements are discharged below; §2 maps each to where.

---

## 1. Hypothesis (one sentence)

**Exact in-context-written operator composition (closed-form `V·K†` writes,
O(log h) repeated-squaring reads) inside a 98M-parameter DeltaNet LM retains
its capability as the binding breadth grows over the pair (K, d = K+1) for
K ∈ {12,16,20,24,28,32}, while the model's own learned writes stay at chance
(1/K) at every K** — i.e. the capability curve is flat and the wall curve is
flat, and the gap between them is the capability separation.

Falsifiable both ways: a K at which the exact-write read falls below the band
is a **frontier finding** (the capability has a measurable breadth limit),
and a K at which the learned write rises off chance retracts the wall.

---

## 2. Gate-memo carried requirements → discharge map

| # | Carried requirement | Discharged in |
|---|---|---|
| 1 | Closed-form-vs-learned contrast at every K (P1b **and** P0) | §6 scorer evaluates both regimes on every cell; §7 bands cover both curves |
| 2 | (K, d=K+1) **pair** framing; chance-normalized metrics | §3, §7.1; every JSON carries `chance = 1/K`, `margin_over_chance`, `kappa` |
| 3 | Fresh residue-verified deep ladder per K, no carried ladders | §4; `kscaling_config.derive_ladder`; §8 items B–E prove the guard has teeth |
| 4 | Matched pool seeds in ALL scoring from day one | §6; `kscaling_battery.py` has **no** unmatched mode; `pool_seed`/`ckpt_seed`/`matched` on every block |
| 5 | Cite toy K-frontier + the five verified externals; never from memory | §9 (all citations copied from the gate memo, not recalled) |

---

## 3. The variable is the PAIR, not K

House precedent (gate memo leg 1): **K=24 is dead at d=48 and healthy at
d=25** — the operator dimension, not the binding count, was doing the work.
Every curve in this design is therefore plotted and reported against the
**pair (K, d=K+1)**, never against K alone, and every axis label must say so.

Three quantities co-vary with K by construction. All three are recorded in
every output JSON so none of them can be silently confounded with the result:

| Quantity | K=12 | K=16 | K=20 | K=24 | K=28 | K=32 |
|---|---|---|---|---|---|---|
| `d_ncr = K+1` | 13 | 17 | 21 | 25 | 29 | 33 |
| chance `1/K` | 0.0833 | 0.0625 | 0.0500 | 0.0417 | 0.0357 | 0.0312 |
| `t_in = max(128, 7K+6)` | 128 | 128 | 146 | 174 | 202 | 230 |
| effective distance at `h_top` = K/2 | 6 | 8 | 10 | 12 | 14 | 16 |

The fourth row is the sharpest confound: at the primary readout depth the
**effective composition distance grows with K**, so a decline could be
breadth (more bindings to hold) or depth (a longer composition). §4.4's
fixed-distance control separates them. The third row has a **kink at K=20**
(the T-floor pad, §5.3) and is read per §7.5 (measured clean; conditional control).

---

## 4. The ladders

### 4.1 Why the pinned ladder cannot be carried

Task-1's ground truth is a **single Hamiltonian K-cycle**
(`grammar_rd._permutation_graph`), so `π^h` depends only on `h mod K`. A
ladder depth is informative only if its residue is (a) non-zero, (b) not a
train residue {1,2,3}, and (c) **not shared with another rung**. The pinned
`_assert_ladder_sound` enforces (a) and (b) only. Evaluating the pinned
ladder `(5,12,20,29,40,61)` against each K — reproduced mechanically by the
build, not asserted from the memo:

| K | residues of (5,12,20,29,40,61) | outcome |
|---|---|---|
| 12 | 5, **0**, 8, 5, 4, **1** | guard CRASHES (identity at h=12; train residue at h=61) |
| 16 | 5, 12, 4, **13**, 8, **13** | **SILENT** collision — 29 and 61 measure one ground truth |
| 20 | 5, 12, **0**, 9, **0**, **1** | guard CRASHES (identity at h=20 and h=40; train residue at h=61) |
| **24** | 5, 12, 20, **5**, 16, 13 | **SILENT** collision — 5 and 29 share residue 5 |
| 28 | 5, 12, 20, **1**, 12, 5 | guard CRASHES (train residue at h=29) + 2 silent collisions |
| 32 | 5, 12, 20, **29**, 8, **29** | **SILENT** collision — 29 and 61 |

The K=12/20/28 crashes and the K=16/32 degradations are the gate memo's own
finding. **The K=24 row is new and is a disclosure about the harness of
record**: the ladder all 55 existing K=24 cells were evaluated on measures
**5 distinct residues at 6 rungs**. It does not invalidate any K=24 number
(each rung's reading is correct for its own residue) but it does mean the
K=24 "6-point depth profile" of record is really a 5-point profile. §4.3
says how the K=24 anchor is handled.

### 4.2 Derivation rule and the per-K table

One rule, applied to every K (`kscaling_config.derive_ladder`):

1. **Band profile.** Six rungs with `binexp_read` squaring counts
   **(2,3,4,4,5,5)** — exactly the pinned K=24 ladder's own profile. Since
   `n_squarings = floor(log2 h)`, this pins the bands to
   `[4,7] [8,15] [16,31] [16,31] [32,63] [32,63]`. **Holding this fixed
   across K is load-bearing**: EXPERIMENT_LOG 2026-08-21 #3 result B measured
   a real monotone fp-depth DRIFT in the in-LM binexp read (1.0000 → 0.9219
   over 3→11 squarings), so an unmatched squaring count would confound the K
   axis with numerical depth.
2. **Top rung.** `h_top(K)` = the smallest `h ∈ [32,63]` with `h ≡ K/2 (mod K)`.
   Residue K/2 is the **antipodal point of the cycle** — the maximum
   reachable effective distance, i.e. the hardest query — and is admissible
   for every K ≥ 12. Every K here is even.
3. **Rungs 1–5.** In band order, the smallest `h` in the band whose residue
   is admissible and unused, strictly increasing.

| K | d | deep ladder | residues mod K | n_sq | n_applies | `h_top` | residue(`h_top`) = K/2 | `h_fix` |
|---|---|---|---|---|---|---|---|---|
| 12 | 13 | 4, 8, 17, 19, 33, **42** | 4, 8, 5, 7, 9, **6** | 2,3,4,4,5,5 | 1,1,2,3,2,3 | 42 | 6 ✓ | 40 |
| 16 | 17 | 4, 9, 21, 22, 39, **40** | 4, 9, 5, 6, 7, **8** | 2,3,4,4,5,5 | 1,2,3,3,4,2 | 40 | 8 ✓ | 36 |
| 20 | 21 | 4, 8, 16, 17, 32, **50** | 4, 8, 16, 17, 12, **10** | 2,3,4,4,5,5 | 1,1,1,2,1,3 | 50 | 10 ✓ | 44 |
| 24 | 25 | 4, 8, 16, 17, 33, **36** | 4, 8, 16, 17, 9, **12** | 2,3,4,4,5,5 | 1,1,1,2,2,2 | 36 | 12 ✓ | 52 |
| 28 | 29 | 4, 8, 16, 17, 33, **42** | 4, 8, 16, 17, 5, **14** | 2,3,4,4,5,5 | 1,1,1,2,2,3 | 42 | 14 ✓ | 32 |
| 32 | 33 | 4, 8, 17, 18, 37, **48** | 4, 8, 17, 18, 5, **16** | 2,3,4,4,5,5 | 1,1,2,2,3,2 | 48 | 16 ✓ | 36 |

Every row: 6 distinct residues, none 0, none in {1,2,3}, strictly increasing
depth, squaring profile identical across K, top rung antipodal.
`n_applies` (= popcount) is **not** matched — it ranges 2–3 at `h_top` and
cannot be matched at every K simultaneously (K=32's only admissible antipodal
depths in-band force popcount 2). It is recorded per rung and is a disclosed
residual; the squaring count, which the #3 DRIFT finding actually implicates,
**is** matched at 5 for every K.

Hand-check of the top rungs (the only rungs the primary band reads):
42 = 3·12 + 6; 40 = 2·16 + 8; 50 = 2·20 + 10; 36 = 1·24 + 12; 42 = 1·28 + 14;
48 = 1·32 + 16. Each is in [32,63] ⇒ `floor(log2 h) = 5`.

`kscaling_config.LADDER_TABLE` is the literal table above; `assert_ladder_table()`
runs at **import** and fails loudly if the table and the rule disagree, so a
typo in either cannot silently change what gets evaluated.

> The table above is the **six K of the curve of record**. K=36 and K=40 were
> added later by the same rule — see **§14.1** for their rows; K=44 is
> construction-impossible and is dropped there.

### 4.3 The K=24 anchor

K=24 is not trained in this sweep (55 cells exist). But its ladder of record
differs from the derived one, so the K=24 curve point must be re-scored on
the **new** K=24 ladder to be commensurate with every other K. This is
eval-only on existing checkpoints (~0.02 GPU-h, 6 cells: 3 frozen + 3
trainable seeds at `ckpt_step == 20000`):

```
NCR_K=24 ~/tdenv/bin/python3 ~/ncr_kscaling/kscaling_battery.py --k 24 \
    --ckpt <existing mob_g3b31_{primary,compB}_s{0,1,2} ckpt> \
    --cellcfg <its training results JSON> --tag k24_anchor_<arm>_s<seed>
```

**Pre-registered before the sweep runs:** the K=24 anchor is scored with the
identical instrument, ladder, pool policy, n and seed as every other K. It is
NOT carried over from the #6 battery numbers (those were read at h=61, a
depth this design does not use). Specs for the anchor are **not** among the
32 in `job_specs/` — the audit releases them, since they touch the 55 cells of
record.

### 4.4 The fixed-effective-distance control

`h_fix(K)` = smallest `h ∈ [32,63]` with `h ≡ 4 (mod K)`: effective distance
**4 at every K**, at the **same squaring count (5)** as `h_top`. It
deliberately shares residue 4 with ladder rung 1 (which sits at squaring count
2), so it is carried as its **own labelled probe** and kept out of
`DEEP_LADDER` rather than smuggled past the distinctness assert — the same
discipline as EXPERIMENT_LOG 2026-08-21 #2 experiment B ("SAME ground truth by
construction, labeled as such"). It is what separates breadth from depth in
§7.4.

---

## 5. The build

Pinned files are **never** written to. `patch_kscaling.py` reads them,
verifies `ncr_lm_wave1_runner.py` md5 = `9a93198b642242f512ff8489e32b0a53`,
and writes patched copies to `~/ncr_kscaling/`. Every edit is an exact-string
replacement whose anchor must occur **exactly once** — a moved anchor aborts
the patch rather than applying somewhere else.

| file | md5 |
|---|---|
| `~/ncr_g3b31_contrastive/ncr_lm_wave1_runner.py` (PINNED, untouched) | `9a93198b642242f512ff8489e32b0a53` |
| `~/ncr_g3b31_contrastive/ncr_lm_wave1_smoke.py` (PINNED, untouched) | `bc105af69661e488ff95f5046e2bcd8a` |
| `~/ncr_kscaling/ncr_lm_wave1_runner.py` (patched) | `ee5833743049e1bb1864124ad5d3fbf6` |
| `~/ncr_kscaling/ncr_lm_wave1_smoke.py` (patched) | `74ee84fc920b024901d11add66cc5c2d` |
| `~/ncr_kscaling/kscaling_config.py` | `6eaf8384a3ef6e9e43b3947720291024` |
| `~/ncr_kscaling/kscaling_battery.py` | `27e691b78252b72d8c0ffbc9af7f7ead` |
| `~/ncr_kscaling/kscaling_smoke.py` | `23fabe2c8b8fda7143d2a08d57f03d71` |

Repo mirror: `matrix-thinking/kscaling_build/` —

```
kscaling_config.py     single source of truth (K, d=K+1, ladders, pad, chance)
patch_kscaling.py      the patch generator (md5-verifies the pinned sources)
kscaling_battery.py    the matched-pool P1b/P0 scorer
kscaling_smoke.py      the real-CUDA smoke
gen_job_specs.py       the 32-spec generator
job_specs/             the 32 specs (NOT queued)
patched/               the two patched files, md5-identical to the box copies,
                       so the build is reconstructable if the box is lost
smoke_results/         the six per-K smoke JSONs
```

`kscaling_config.py` is byte-identical in both places, and `patched/` md5s
match the box exactly (`ee5833743049e1bb1864124ad5d3fbf6`,
`74ee84fc920b024901d11add66cc5c2d`).

### 5.1 S1 — K and d are DERIVED (gate requirement, PATCH-not-config)

The pinned graft carries `K_NCR = 24` and `D_NCR = 25` as **two independent
hand-set constants**, and asserts the K=24 literal `NCR_PARAM_EXACT ==
173_209`. Patched: both come from `kscaling_config`, where

```
D_NCR = K_NCR + 1          # the ONLY definition of d_ncr in the program
NCR_PARAM_EXACT = 40*h² + 4*d*h + 46*h + d      # re-derived per K
```

re-asserted against the formula at import, with a back-compat anchor
(`K==24 ⇒ 173_209`). **Verified by measurement, not formula** — smoke item F
counts the actual `nn.Module` parameters at every K:

| K | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|
| `ncr_param_exact` (derived) | 170,125 | 171,153 | 172,181 | **173,209** | 174,237 | 175,265 |
| measured NCR params | 170,125 | 171,153 | 172,181 | 173,209 | 174,237 | 175,265 |
| `integ_param_exact = 2·768·d` | 19,968 | 26,112 | 32,256 | 38,400 | 44,544 | 50,688 |
| **total params / arm** | 97,809,805 | 97,816,977 | 97,824,149 | 97,831,321 | 97,838,493 | 97,845,665 |

Total parameters vary by **0.037%** across the whole K range — the sweep is
param-matched to within a rounding error, so the curve is not a capacity
curve in disguise.

### 5.2 S2 — the task config crashed at K=20

`DeltaNetRDTaskConfig.__post_init__` runs its own periodicity guard on
`H_test`/`H_extra`, whose defaults are `(4,5,6)` / `(7,21)`. At K=20,
`21 mod 20 == 1`, a train residue — so **constructing the config at all**
raised `AssertionError` at K=20, before any model existed. Patched to pass
this K's own ladder, which both fixes the crash and turns `grammar_rd`'s
independent guard into a **second, external check** on the ladder actually
evaluated.

### 5.3 S3 — the T-floor pad (a launch-losing FATAL, measured)

The document is `7K+7` tokens and the backbone is fed `doc[:, :-1]`, i.e.
`T_in = 7K+6`. `lm_pretrain_rd.DeltaNetLMMixer.forward` hard-asserts
`T >= 128` (chunk_delta_rule's backward crashes below it; F15-LM, measured
2026-07-02). **Measured on this box 2026-08-21, fresh process per T, rung-1
backbone:**

| `T_in` | 90 (K=12) | 118 (K=16) | 128 | 146 (K=20) | 174 (K=24) | 230 (K=32) |
|---|---|---|---|---|---|---|
| result | **AssertionError** | **AssertionError** | PASS | PASS | PASS | PASS |
| peak mem | — | — | 0.96 GB | 1.04 GB | 1.14 GB | 1.37 GB |

**K=12 and K=16 were unrunnable — an immediate crash on step 1, not a
quality question.** Twelve of the thirty sweep cells would have died. Fix: a
**left pad** of `max(0, 128 − (7K+6))` inert **BUFFER** tokens (the same
reserved token `grammar_rd` already uses as intra-clause filler — no new token
type enters the vocabulary), with all four position fields
(`key_pos`, `val_pos`, `query_key_col`, `query_mark_col`) shifted by the same
amount. Pad = 38 at K=12, 10 at K=16, **0 for every K ≥ 20**, so K ≥ 20 —
in particular the **K=24 anchor** — stays byte-identical to the pinned
construction. Smoke item G verifies by content (not shape) that the KEY, VALUE
and query-KEY positions still index this row's own entities after the shift;
item K verifies the unpadded length still crashes.

### 5.4 R1–R7 — runner patches

* **R1** `RUNNER_TAG` → `ncr_kscaling_runner_v1` (so a K-scaling checkpoint
  can never be silently resumed by, or confused with, the pinned K=24 wave —
  `load_checkpoint` asserts on this field); `DEEP_LADDER` from `kscaling_config`.
* **R2** `_assert_ladder_sound` delegates to the strengthened check: both
  pinned checks **plus** pairwise residue distinctness, strictly-increasing
  depth, and a matched squaring profile.
* **R3** the fixed-distance control is evaluated into its **own** `fixed_dist`
  block, never merged into `deep`.
* **R4/R5** `KS.provenance(...)` — K, d, ladder, residues, squarings, applies,
  chance, `h_top`, pad, `t_in` — is written into every results JSON.
* **R6/R6b/R7** a **mandatory `--k` flag** that must equal the `NCR_K` env var
  `kscaling_config` actually reads, asserted immediately after `parse_args` on
  **every** mode. Thirty specs each carrying an env var and a flag is the
  single easiest way to launch a wave at the wrong K; this dies before any GPU
  work rather than producing plausible, wrongly-labelled numbers.
* **R8/R9** — see §5.5.

### 5.5 R8 — a second launch-losing FATAL, found only by the end-to-end run

`build_attribution` indexed its two headline fields by a **K=24-ladder
literal**:

```python
"primary_signal_deepest_gap_h61":    deep_gap["h=61"],
"primary_signal_v2_deepest_gap_h61": retrieval24_gap_deep["h=61"],
```

`h=61` is in **no** derived ladder, so **every cell at every K would have
raised `KeyError('h=61')` at its first eval** (step 1000 of 20000) — all 32
cells lost, ~28 GPU-h burned to nothing.

**The module-level smoke could not see this**: it exercises the model, the
ladder, the document, the checkpoint and the guards, but never calls
`build_attribution`, which only runs inside the training loop's eval. It was
caught by this build's **end-to-end 3-step production run** through the actual
spec command line (§8.1). Re-keyed to this K's own `H_TOP`, with the depth and
residue emitted as their own fields rather than frozen into a field name, plus
`chance` and the fixed-distance probe depth. R9 re-keys the accompanying prose
so it states this K's ladder instead of the K=24 one.

**Process note for the audit:** this is the second launch-losing FATAL in this
build (the first being §5.3's T-floor), and the two were caught by *different*
instruments. A module smoke and an end-to-end run through the literal spec
command are not substitutes for each other. Any later revision of this build
must re-run **both**.

---

## 6. Scoring — `kscaling_battery.py`

Descends from `ncr_writecond/poolmatched_battery.py`
(md5 `7c3610cc09303627234b0cd6c9977014`), the instrument that produced the
2026-08-21 #6 adjudication of record. Three changes, all forced by K-scaling:

1. **Matched pools are the only mode.** The ancestor scored under both the
   checkpoint's seed and legacy seed-0 because its job was to adjudicate a
   retraction; that question is settled. Every pool here is built with
   `build_grammar_pools_and_cfg(seed = ckpt["seed"])`, and
   `pool_seed`/`ckpt_seed`/`matched` are recorded on every block so the
   property is auditable, never assumed (gate requirement 4).
2. **Chance is 1/K.** The ancestor hard-codes `CHANCE = 1/24`. Every accuracy
   is reported with `margin_over_chance = acc − 1/K` and the per-K binomial
   wall band `1/K ± 3·sqrt(p(1−p)/n)`.
3. **Hops are this K's own ladder** — 3 train + 6 ladder + 1 fixed-distance
   control, both regimes (P1b and P0), n=256, eval seed 90210.

**The K guard.** The checkpoint records `ncr_config.d`. The scorer refuses to
run unless `ckpt d == K+1` for the K it was invoked with — it is therefore
impossible to score a K=24 checkpoint as a K=32 curve point.

**Metric name.** `retrieval24_acc` is kept despite the "24": it is computed as
`cos_all.argmax(1) == tgt_slot` over exactly K slots and is already K-generic.
Renaming it would break comparability with the 55 K=24 cells of record. The
JSON annotates this.

Per-K wall bands at n=256 (`WALL-HOLDS` is checked against these):

| K | chance | sd | band (chance ± 3sd) |
|---|---|---|---|
| 12 | 0.0833 | 0.0173 | [0.0315, 0.1352] |
| 16 | 0.0625 | 0.0151 | [0.0171, 0.1079] |
| 20 | 0.0500 | 0.0136 | [0.0091, 0.0909] |
| 24 | 0.0417 | 0.0125 | [0.0042, 0.0791] |
| 28 | 0.0357 | 0.0116 | [0.0009, 0.0705] |
| 32 | 0.0312 | 0.0109 | [0.0000, 0.0639] |

---

## 7. Pre-registered bands

All bands read **P1b/P0 accuracy at `h_top(K)`, matched pools, n=256,
`ckpt_step == 20000`**, unless stated. Three chance-normalizations are
recorded on every number:

* `acc` — raw
* `margin_over_chance = acc − 1/K` — the brief's literal quantity
* `kappa = (acc − 1/K)/(1 − 1/K)` — chance-corrected; **the only one that is
  strictly comparable across K**, since `margin`'s ceiling is itself `1 − 1/K`
  and so a margin bar of 0.90 is `1/K` stricter at small K (at K=12 it demands
  raw acc ≥ 0.9833; at K=32, ≥ 0.9313).

**Primary band is stated on `kappa` — ELECTED by KSCALING_AUDIT_R1 (M2),
2026-08-21.** The audit measured that a margin-0.90 bar is monotone-in-K
(raw acc 0.9833 at K=12 vs 0.9313 at K=32, span 0.052): a flat
0.980-accuracy model would FAIL only at K=12 and the sweep would
manufacture `FRONTIER-AT-K*=12` as a pre-registered output. κ's raw-acc
span across the range is 0.0052. `margin_over_chance` remains recorded
alongside in every JSON. Existing matched-pool K=24 evidence (#6:
primary 1.0000, compB 0.9922 at h=61) clears both formulations.

### 7.1 CURVE 1 — CAPABILITY (P1b, frozen arms) — PRIMARY

* **CAPABILITY-HOLDS(K)** = `kappa ≥ 0.90` at `h_top(K)` for
  **≥ 2/3 seeds** on the FROZEN arm.
* **CAPABILITY-HOLDS (curve)** = CAPABILITY-HOLDS(K) at **every** K
  ∈ {12,16,20,24,28,32}. This is the headline: exact composition breadth does
  not degrade over a 2.7× range of K at matched parameters.
* **FRONTIER-AT-K\*** = the smallest K at which CAPABILITY-HOLDS(K) fails.
  **This is reported as a positive frontier finding — the capability has a
  measurable breadth limit at the pair (K\*, d=K\*+1) — not as a failure**,
  and it is the more interesting of the two outcomes for the flagship. It
  must be reported with the §7.4 breadth-vs-depth attribution attached.
* **PARTIAL** = holds at some K, fails at others non-monotonically ⇒ an
  instrument or convergence problem, not a capability curve; diagnose before
  reporting.

### 7.2 CURVE 2 — THE WALL (P0, all arms)

* **WALL-HOLDS(K)** = **every** P0 reading (10 hops × 6 cells at that K)
  inside the §6 band.
* **WALL-HOLDS (curve)** = WALL-HOLDS(K) at every K. Prior: 165/165 P0
  readings at chance under matched pools (#6) — this extends it across K.
* **WALL-BREACHED-AT-K** = any P0 reading above the band at any K, replicated
  across ≥2 seeds. Retracts the wall at that K and is a major finding either
  way; a single-seed excursion is an outlier to re-measure, not a breach.

### 7.3 CURVE 3 — the residual frozen-vs-trainable ordering (SECONDARY)

The question inherited from #6: the matched-pool gap is +0.0098 at K=24, h=61
and **ceiling-compressed**; does it open at larger K?

Δ(K) = median κ(frozen, 3 seeds) − median κ(trainable, 3 seeds) at `h_top(K)`.

> **POWER DISCLOSURE — READ BEFORE ADJUDICATING.** At 3 seeds per (K, recipe),
> a per-K Mann–Whitney U on 3-vs-3 has a **minimum attainable two-sided
> p of 0.10**. A per-K band of the #7-style form ("gap > 0.05 **and**
> p < 0.01") is therefore **mathematically unreachable in this sweep** and is
> deliberately NOT pre-registered per K. This is a real limitation of the
> sweep as sized, surfaced here rather than discovered at harvest.

Pre-registered instead:

* **Primary inferential test — REPLACED by KSCALING_AUDIT_R1 (M1),
  2026-08-21.** The DRAFT-R0 pooled 15v15 Mann–Whitney analyzed a blocked
  design unblocked: the audit measured that a UNANIMOUS within-K
  frozen>trainable ordering (44.5/45 within-K wins) still yields pooled
  p = 0.361 (because κ varies across K far more than between recipes
  within K), while a pure single-K artifact yields p = 0.016 — the pooled
  test is anti-powered in exactly the regime the sweep exists to find.
  **Pre-registered instead: stratified within-K exact permutation test.**
  T = Σ_K U_K where U_K = # of (frozen, trainable) within-K pairs with
  κ_frozen > κ_trainable (ties count ½; 9 pairs per stratum). Audit-
  precomputed exact thresholds for p < 0.01: **T ≥ 36/45** with the five
  sweep K, **T ≥ 42/54** if the K=24 anchor re-score (its own stratum —
  which also dissolves the cross-harness exchangeability objection)
  is released.
  * **ORDERING-CONFIRMED** = median within-K gap > 0.05 (median over the
    ≥5 per-K median-gaps) **and** T at or above the exact threshold.
  * **ORDERING-NEGLIGIBLE** = median within-K gap ≤ 0.05.
  * **ORDERING-INVERTED** = median within-K gap < −0.05 with T at or
    below the symmetric lower threshold (9/45, resp. 12/54)
    (trainable beats frozen — publishable as a reversal, and consistent with
    #6's reinterpretation that freezing buys pool-agnosticism, not composition).
* **Trend (descriptive, explicitly underpowered)** — Spearman ρ between K and
  Δ(K) over the 6 curve points. At n=6, p < 0.05 requires |ρ| ≥ 0.829; this is
  reported **with** that caveat inline and **cannot alone** license an
  "ordering opens with K" claim.
* **Escalation, pre-registered** — if the pooled test is CONFIRMED **and**
  Δ(K) is largest at the two highest K, that licenses a **seed-extension wave**
  (n=12 per recipe at those two K, ~10 GPU-h) which *would* be powered for a
  per-K claim. Declaring "the ordering opens at larger K" is gated on that
  wave, not on this one.

### 7.4 CURVE 4 — breadth vs depth (attribution, pre-registered)

`h_top` grows in effective distance with K (K/2); `h_fix` does not (always 4),
at the same squaring count.

* **DEPTH-DRIVEN** = κ(`h_fix`) flat in K (range ≤ 0.05) while κ(`h_top`)
  declines ⇒ the decline is composition depth, not binding breadth. The
  headline becomes a *depth* frontier, and the breadth claim survives.
* **BREADTH-DRIVEN** = κ(`h_fix`) declines with K comparably to κ(`h_top`)
  ⇒ holding K bindings is itself the limit. This is the stronger and more
  surprising result.
* **BOTH-FLAT** = neither declines ⇒ CAPABILITY-HOLDS, and this control is
  the evidence that flatness is not an artifact of an easy probe.

### 7.5 The T-floor pad — measured clean; conditional control (added per KSCALING_AUDIT_R1)

§3/§5.3's promised pad read lives here. The audit MEASURED pad-invariance
rather than arguing it: a trained K=24 checkpoint scored at pads 0/10/38
returns identical accuracy to the last digit at all 10 hops — as expected
by construction, since NCR keys/values are extracted from raw token ids,
not from backbone hidden states, so inert BUFFER tokens cannot enter the
write or the read. A common-T pad across all K is therefore NOT
warranted. **Conditional control, pre-registered:** only if K∈{12,16}
(pad > 0) shows an anomaly absent at K≥20 (pad = 0) that §7.4's
breadth-vs-depth attribution cannot explain, run a pad-titration at one
affected K (3 pads × 1 seed, ≤0.8 GPU-h) before any frontier claim
naming those K.

Every outcome of every curve above is publishable and pre-specified. No
combination is a null that ends the lane.

**Wave-0 rule (M6, adjudicated 2026-08-21):** all six K=32 cells
(0100/0101 calibration + the four sweep seeds) launch together as wave 0,
so `FRONTIER-AT-K*=32` can never be declared from n=1; the calibration
legs license the sweep, and the K=32 curve point is read at full n=3 per
recipe like every other K.

---

## 8. Smoke — REAL CUDA, all six K, every negative test FIRED

`kscaling_smoke.py`, one process per K, one GPU each, run 2026-08-21.
Results: `/ephemeral/kscaling/smoke/kscaling_smoke_K{12,16,20,24,28,32}.json`.

**12 PASS / 0 FAIL at K=12 and K=16; 11 PASS / 0 FAIL / 1 N/A at K=20,24,28,32
(the N/A is item K, which is only meaningful where the pad is load-bearing).**

| item | kind | what it proves | result |
|---|---|---|---|
| A | positive | D=K+1; derived param formula; ladder sound; `h_top` antipodal; `h_fix` matches `h_top`'s squaring count | PASS ×6 |
| **B** | **negative** | the carried pinned ladder `(5,12,20,29,40,61)` is **REJECTED** at every K | **FIRED ×6** |
| **C** | **negative** | a ladder that **passes** the pinned guard but has a duplicate residue is **REJECTED** — the fixture is first proven to pass the pinned checks, so this demonstrates the pinned guard's blind spot rather than assuming it | **FIRED ×6** |
| **D** | **negative** | identity-residue rung rejected (pinned check still has teeth) | **FIRED ×6** |
| **E** | **negative** | train-residue rung rejected (pinned check still has teeth) | **FIRED ×6** |
| F | positive | **measured** NCR/integ parameter counts == derived formulas, exactly | PASS ×6 |
| G | positive | doc shape, `t_in`, pad; KEY/VALUE/query-KEY positions still index this row's own entities after the shift; pad is BUFFER tokens | PASS ×6 |
| H | positive | fwd+bwd+grad, **both arms × both regimes** (P1b/P0), finite loss, non-zero finite backbone grads, `o_raw` width == d | PASS ×6 |
| **K** | **negative** | the **unpadded** `7K+6` length still crashes the kernel floor | **FIRED ×2** (K=12,16); N/A ×4 |
| I | positive | checkpoint → load → restore → **bit-identical** logits (`max|Δ| == 0.0`); ckpt records d == K+1 | PASS ×6 |
| J | positive | the pinned read-ablation exact-zero invariant still holds | PASS ×6 |
| L | positive | throughput, peak memory, sampled SM utilisation | PASS ×6 |

A negative test that returns cleanly is recorded **FAIL — "did not fire"**, and
one that raises the *wrong* error is recorded **FAIL — "fired with the WRONG
error"**. All eight negative-test instances fired with the expected message.
Verbatim, at K=12:

```
B: AssertionError: deep-ladder h=12 is IDENTITY mod K=12 (h%K=0) -- confounded, not held-out
C: AssertionError: deep-ladder (4, 8, 16, 17, 19, 33) at K=12 has PAIRWISE residue
   collisions at [4] (residues [4, 8, 4, 5, 7, 9]) -- two rungs measure the SAME ground truth
D: AssertionError: deep-ladder h=24 is IDENTITY mod K=12 (h%K=0) -- confounded, not held-out
E: AssertionError: deep-ladder h=25 has h%K=1 colliding with a train-residue [1, 2, 3]
   -- secretly in-distribution, not held-out
K: AssertionError: sequence length 90 < _MIN_KERNEL_T=128 -- chunk_delta_rule's backward
   crashes below this floor (F15-LM, measured 2026-07-02)
```

### 8.1 End-to-end run through the literal spec command line

A 3-step cell at K=12 was run through the **exact** `cmd` shape the specs
carry (`NCR_K=12 … --k 12 --mode calibration …`), writing to `/ephemeral`:

* Ran to `status=COMPLETED`, `step=3`; read-ablation exact-zero and the
  same-op assertion passed **pre- and post-train**; loss finite and falling
  (11.187 → 10.917).
* The results JSON carries the full `kscaling` provenance block (K, d,
  `d_equals_k_plus_1`, chance, ladder, residues, squarings, applies, `h_top`,
  `h_top_is_antipodal`, `fixed_dist_probe`, `doc_len`, `doc_left_pad`, `t_in`)
  and the re-keyed attribution fields (`primary_signal_v2_deepest_gap_h = 42`,
  `…_residue = 6`).
* **This is what caught §5.5's R8 FATAL.**

### 8.2 Scorer, end to end, with both K-guards fired

Against the checkpoint that run produced:

* **Positive** — `kscaling_battery.py --k 12`:
  `P1b@h_top=42 acc=1.0000 margin=+0.9167 | P0max=0.1250 (chance 0.0833,
  band [0.0000, 0.1870])`. Matched pool (`ckpt_seed=0`), self-check PASS.
  *(A 3-step model: this says the instrument reads, not that the science
  holds. No verdict is implied or recorded.)*
* **Negative — wrong-K scoring REJECTED:**
  `K MISMATCH [e2e_wrongK]: checkpoint records ncr d=13 / integ d_ncr=13, but
  this process is configured for K=32 -> d=33. The checkpoint was trained at
  K=12. Refusing to score.`
* **Negative — env/flag drift REJECTED:**
  `AssertionError: --k 32 disagrees with NCR_K='12' (kscaling_config resolved
  K_NCR=12). Refusing to run: one of the two is a typo and the results would
  be silently mislabelled.`

---

## 9. Prior art — cited, never recalled

All citations copied from `research/kscaling-novelty-2026-08-21.md`
(agent-web-verified, coordinator spot-checked). **None is cited from memory.**

**Internal (PRIOR, not to be rediscovered).** The toy-harness K record is
extensive: K ∈ {8..32}, **FRONTIER-AT-K\*=30**, **CONFIRMED-WALL-AT-160K**,
and **toy K=32 far-depth death at h ≈ 5–6 in the FREE-write regime**. No
LM-graft arm has ever run at K ≠ 24.

> **How the K=32 toy prior bears on this design.** It is a **FREE-write**
> result. Our primary claim (P1b) is about **EXACT-write** reads, so the prior
> does not directly bar it. But P0 *is* the free-write analogue — so the toy
> prior **predicts P0 death at K=32, which is exactly what WALL-HOLDS
> predicts.** The prior therefore acts as a **positive control on the wall
> curve**, not as a threat to the capability curve. §10's calibration gate
> nonetheless treats K=32 as the riskiest cell and runs it first.

**External (verified).** Schlag/Irie/Schmidhuber [arXiv:2102.11174] (fast
weight programmers); Wang et al. [arXiv:2501.12352] (test-time regression —
pseudoinverse writes exist as a framework, `M_t = V_t(K_t†)ᵀ`, but no
composition reads and no contrast curve); Grazzi et al. [arXiv:2411.12537] and
Siems et al. [arXiv:2502.10297] (DeltaProduct — group word problems via
SGD-learned serial transitions, no closed-form writes, no `Z^h` reads); Liu et
al. [arXiv:2210.10749] (shortcuts to automata — O(log T) depth via attention
layers, not fast-weight powers); Arora et al. [arXiv:2312.04927]
(Zoology/MQAR — dimension-vs-K curves for **flat recall only**); Li/Guo/Andreas
[arXiv:2503.02854] (LMs spontaneously learn associative-scan state tracking —
emergent, not engineered, K not the organizing axis); Log-Linear Attention
[arXiv:2506.04761] (O(log T) reads over **time**, not hop-depth of a written
relation).

**Instrument note carried from the gate:** one WebFetch PDF summary
(arXiv:2601.04254) was a **confabulated match**, caught by re-verification
against the arXiv abstract. Treat single-fetch summaries as unverified until
cross-checked.

---

## 10. Calibration gate — MANDATORY before the sweep

Two cells, **K=32 (the riskiest K), both recipes, seed 0**, run first and
alone. Specs `0100`/`0101` — **these two are queue-eligible candidates; the
30 sweep specs are NOT.**

**LICENSE-SWEEP requires all three:**

1. **Gate-0 convergence.** Final CE < initial CE on the `full_graft` arm, loss
   finite throughout, run reaches `step == 20000` with `status == COMPLETED`.
2. **In-distribution recovery.** P1b `kappa ≥ 0.90` (M2 election) at the train
   hops h ∈ {1,2,3} on the **frozen** calibration cell.
3. **Deep capability.** P1b `kappa ≥ 0.90` (M2 election) at `h_top(32) = 48` on
   the **frozen** calibration cell.

**If (3) fails but (1) and (2) pass:** K=32 is the frontier. Do **not** launch
the 30 blindly — re-scope the sweep to K ≤ 28 (24 cells), report K=32 as
FRONTIER-AT-K\*=32, and run the §7.4 attribution at K=28 and K=32 to say
whether it is breadth or depth.

**If (1) or (2) fails:** an instrument/convergence problem, not a science
result. Diagnose before any sweep GPU-hour.

**Trainable calibration cell (`0101`):** informational only — it does not gate.
Its purpose is to price the trainable recipe's cost and to give the §7.3
ordering its K=32 anchor early.

---

## 11. Ledger, placement, and predicted utilisation

### 11.1 Measured per-K cost (this build, 30 timed steps after 5 warmup, batch 32, both arms)

| K | `t_in` | s/step | projected GPU-h @20K (train only) | ×1.17 eval/ckpt overhead | peak mem | SM util (median/max) |
|---|---|---|---|---|---|---|
| 12 | 128 | 0.1268 | 0.704 | **0.824** | 5.21 GB | 72 / 74 |
| 16 | 128 | 0.1205 | 0.669 | **0.783** | 5.22 GB | 72 / 74 |
| 20 | 146 | 0.1195 | 0.664 | **0.777** | 5.54 GB | 86 / 88 |
| 24 | 174 | 0.1289 | 0.716 | **0.838** | 5.96 GB | 89 / 90 |
| 28 | 202 | 0.1372 | 0.762 | **0.892** | 6.38 GB | 94 / 96 |
| 32 | 230 | 0.1529 | 0.850 | **0.995** | 6.78 GB | 93 / 100 |

The **1.17 overhead multiplier is calibrated, not guessed**: the K=24 cells of
record measured 0.82–0.83 GPU-h wall for identical steps/batch/backbone, and
this build's K=24 train-only projection is 0.716 ⇒ 1.145 measured overhead,
scaled by 10/9 for this design's extra eval hop ⇒ 1.164, rounded to 1.17. The
resulting K=24 estimate (0.838) reproduces the measured 0.82–0.83 — the cost
model is validated against a real completed cell, not asserted.

### 11.2 Ledger

| item | cells | GPU-h |
|---|---|---|
| Calibration gate (K=32, both recipes, seed 0) | 2 | **1.99** |
| Sweep: K ∈ {12,16,20,28,32} × {frozen, trainable} × 3 seeds | 30 | **25.63** |
| Scoring (`kscaling_battery`, eval-only, 32 cells) | — | **0.12** |
| K=24 anchor re-score (eval-only, existing ckpts, audit-released) | 6 | **0.02** |
| **TOTAL** | **32 trained** | **≈ 27.8 GPU-h** |

Ceremony tier: 10–50 GPU-h ⇒ **audit + pre-launch resource/placement
red-team** (this document is the input to both).

### 11.3 Placement — 8 GPUs, one cell per GPU

The `~/queue/` worker contract is **one job per GPU**: each worker checks
`nvidia-smi --query-compute-apps` for its own GPU and treats *any* listed PID
as busy, so a second cell can never be claimed onto an occupied GPU. Placement
is therefore one cell per GPU, dispatched by the existing 8 workers
(`CUDA_VISIBLE_DEVICES` is set by `queue_worker.sh`, so **no spec hardcodes a
GPU**).

* **Calibration:** 2 cells, 2 GPUs, ~1.0 h wall.
* **Sweep:** 30 cells / 8 GPUs ≈ 3.75 waves × ~0.85 h ≈ **3.2 h wall**.
* **Total ≈ 4.2 h wall** for the whole payload.

**Predicted utilisation: 72–94% median SM (measured, above), i.e. every cell
clears the doctrine's <50%-is-a-bug threshold**, and K ≥ 20 (18 of the 30
cells) runs at 86–94%.

**Disclosed low-utilisation cells and a measured packing option.** K=12 and
K=16 (12 of 30 cells) sit at 72%. Two K=12 cells packed on one GPU were
measured: SM util **72% → 99%**, per-cell s/step 0.1268 → 0.150–0.162
(+23% GPU-h per cell), **1.63× wall throughput** for the pair. Packing is
therefore the doctrine-preferred trade on an uptime-metered grant (wall time
and utilisation are the binding resources, not GPU-h). **It is NOT built into
these specs**, because it requires paired specs (two backgrounded processes
plus `wait` in one `cmd`) to get past the worker's one-job-per-GPU gate, which
couples two cells' fates and their validity checks. **Flagged for the
resource/placement red-team to elect or decline** — the measurement is
recorded here so the decision is priced, not guessed.

**FLOP-efficiency disclosure.** SM occupancy is high (72–94%) but arithmetic
intensity is low: ≈ 7.9e16 FLOPs per cell against 0.72 GPU-h ⇒ ≈ 31 TFLOP/s,
~6% of dense bf16 peak. This is inherent to the task (sequences of 128–230
tokens, `d_state`=64, many small kernels), not a bug. Occupancy is the metric
the doctrine specifies and it is met; the FLOP number is disclosed so nobody
reads 94% util as 94% MFU.

### 11.4 Disk

All checkpoints go to **`/ephemeral/kscaling/...`** (5.9 TB, 5% used). Never
the root filesystem — it filled to 100% once and is currently at 68%.
32 cells × 2 arms × ~1.2 GB ≈ 77 GB, comfortably on `/ephemeral`.

---

## 12. Job specs

`matrix-thinking/kscaling_build/job_specs/` — 32 files, none queued.

* `0100`, `0101` — **calibration**, K=32, frozen + trainable, seed 0.
  Marked **`CANDIDATE -- queue-eligible after audit`**.
* `0110`–`0139` — **sweep**, 30 cells. Marked
  **`CANDIDATE -- NOT queue-eligible until the calibration gate LICENSES the sweep`**
  — double-gated: audit *and* calibration.

Every spec carries `NCR_K=<K>` in its `cmd` **and** `--k <K>`, which the runner
asserts are equal (§5.4 R6b). Each writes to `/ephemeral/kscaling/...` and
carries a `validity_check` asserting `status == COMPLETED`, `step >= 20000`,
**and** that the recorded `kscaling.K` and `kscaling.d_ncr` match the spec's
own K — so a mislabelled cell fails its own validity check rather than
entering the curve.

### 12.1 Sweep structure

Per-cell try/except sequencing is provided by the queue itself: one cell per
spec, a failed cell routes to `failed/` and is **not** auto-retried, and the
other 29 are unaffected. There is no shared driver whose crash could take down
the wave.

| | frozen-contrastive (primary recipe) | trainable-contrastive (compB recipe) |
|---|---|---|
| flags | `--aux-loss-type contrastive+cosine --freeze-entity-adapter --contrastive-temperature 0.07` | `--aux-loss-type contrastive+cosine --contrastive-temperature 0.07` |
| K=12 | 0110, 0111, 0112 | 0113, 0114, 0115 |
| K=16 | 0116, 0117, 0118 | 0119, 0120, 0121 |
| K=20 | 0122, 0123, 0124 | 0125, 0126, 0127 |
| K=28 | 0128, 0129, 0130 | 0131, 0132, 0133 |
| K=32 | 0134, 0135, 0136 | 0137, 0138, 0139 |

Seeds 0, 1, 2 in each triple. All other hyperparameters are held at the
audited G3-B31 values (20000 steps, batch 32, eval batch 64, lr 3e-4, warmup
200, `--aux-read-loss-weight 0.5`, `--ortho-reg-weight 0.1`, ceiling 6.0
GPU-h) — **the recipe is not a variable in this sweep; K is.**

---

## 13. Open items for the audit

1. **§7.3 power.** Per-K frozen-vs-trainable inference is unreachable at n=3.
   The pooled test and the pre-registered seed-extension escalation are the
   proposed remedy. Ratify or resize the sweep.
2. **§11.3 packing.** Elect or decline 2-per-GPU packing for the 12 K=12/16
   cells (measured: 72%→99% util, 1.63× wall, +23% GPU-h/cell, requires paired
   specs).
3. **§4.3 K=24 anchor.** Release (or decline) the 6 eval-only re-score specs
   against the cells of record.
4. **§7 band choice.** `margin ≥ 0.90` (the brief's literal band, `1/K`
   stricter at small K) vs `kappa ≥ 0.90` (strictly cross-K comparable). Both
   are recorded; pick the primary.
5. **§4.2 `n_applies` residual.** Squaring count is matched at 5 across all K;
   popcount is not (2–3) and cannot be. Confirm this is an acceptable residual.
6. **§4.1 K=24 disclosure.** The ladder of record for the 55 existing K=24
   cells has a silent residue collision (29 ≡ 5 mod 24), so its 6-point depth
   profile is really 5-point. Decide whether this needs a note in
   EXPERIMENT_LOG independent of this sweep.

---

## 14. K=36/40 extension

Added 2026-08-22 (repo commit `add3239`), after the sweep of §12 completed
36/36 cells with 0 failures. This section is the **delta only** — every
construction, band, recipe, instrument and pool policy above applies
unchanged. Gate verdict of record: **CLEAR-WITH-CONSTRAINTS**, EXPERIMENT_LOG
2026-08-22 **#3**.

### 14.1 The two new ladders

Derived by §4.2's rule, unchanged — squaring profile (2,3,4,4,5,5), top rung
= smallest `h ∈ [32,63]` with `h ≡ K/2 (mod K)`, rungs 1–5 the smallest
admissible unused residue in each band, strictly increasing. This table
**extends** §4.2's; it does not replace it.

| K | d | deep ladder | residues mod K | n_sq | n_applies | `h_top` | residue(`h_top`) = K/2 | `h_fix` | `t_in` | pad |
|---|---|---|---|---|---|---|---|---|---|---|
| 36 | 37 | 4, 8, 16, 17, 32, **54** | 4, 8, 16, 17, 32, **18** | 2,3,4,4,5,5 | 1,1,1,2,1,**4** | 54 | 18 ✓ | 40 | 258 | 0 |
| 40 | 41 | 4, 8, 16, 17, 32, **60** | 4, 8, 16, 17, 32, **20** | 2,3,4,4,5,5 | 1,1,1,2,1,**4** | 60 | 20 ✓ | 44 | 286 | 0 |

Hand-check of the top rungs: `54 = 1·36 + 18`, and no `h ∈ [32,53]` has
residue 18 mod 36 (32–53 walk residues 32…35, 0, 1, …, 17); `60 = 1·40 + 20`,
and 32–59 walk residues 32…39, 0, 1, …, 19. Both are in [32,63] ⇒
`floor(log2 h) = 5`, the same squaring count as every other K's top rung.
`h_fix`: smallest `h ∈ [32,63]` with `h ≡ 4`, i.e. `40 = 1·36 + 4` and
`44 = 1·40 + 4`; both at squaring count 5, effective distance 4. Both pads
are 0 (`t_in = 7K+6 ≫ 128`), so both K are byte-identical to the pinned
construction like every K ≥ 20. The full hand-derivation is reproduced as a
comment in `kscaling_config.LADDER_TABLE`, and `assert_ladder_table()` proves
the literal table equals `derive_ladder()` output **at import for all 8 K**.

The grid guard is extended without disturbing the six K of record:
`SWEEP_K_GRID` is unchanged, `FRONTIER_K_GRID = (36, 40)`, and only the
`NCR_K` env guard and `assert_ladder_table()` read
`ADMITTED_K_GRID = SWEEP_K_GRID + FRONTIER_K_GRID`.

**K=44 is DROPPED, not deferred silently** (#3). An antipodal top rung needs
`3K/2 = 66 ≤ 63`, so no `h ∈ [32,63]` has residue 22 mod 44 and
`derive_ladder(44)` raises. The nearest reachable rung is `h=63`, residue 19 —
a 13.6% effective-distance reduction that would break the antipodal
convention. Addable only as an explicit disclosed deviation; not this wave.

**Two disclosed residuals.** (i) `n_applies` (popcount) at `h_top` is **4**
here (54 = 0b110110, 60 = 0b111100) versus 2–3 across K=12…32. §4.2 already
discloses that popcount is not matchable across K; this extends the spread.
`n_squarings` — the axis the 2026-08-21 #3 result-B fp-DRIFT finding actually
implicates — is 5 at every K, unchanged. (ii) At **K=36 only**, the pinned
reference ladder `(5,12,20,29,40,61)` is *genuinely sound* (residues
5, 12, 20, 29, 4, 25 — six distinct, none 0, none in {1,2,3}, profile intact),
because past K=32 only `h=61` wraps and `61 mod 36 = 25` collides with
nothing. §8's negative item B therefore has nothing to fire on at K=36; see
§14.4.

### 14.2 The 8-strata stratified-T threshold

§7.3's stratified within-K exact permutation test gains two strata (K=36, 40),
giving **8 strata × 9 within-K (frozen, trainable) seed pairs = 72**.
Recomputed at build from the *same* construction the audit used:

* Per stratum, `U_K` = # of the 9 pairs with `κ_frozen > κ_trainable` (ties ½).
  Under within-stratum exchangeability the null is the exact Mann–Whitney
  3-vs-3 distribution over the `C(6,3) = 20` equally likely assignments:
  counts `1,1,2,3,3,3,3,2,1,1` for `U = 0…9`.
* `T = Σ_K U_K`; the null is the 8-fold convolution over `20⁸` outcomes,
  symmetric about 36.
* Criterion, as used by the audit: **two-sided p < 0.01**, i.e. upper tail
  < 0.005.

| strata | max T | threshold | exact one-sided P(T ≥ thr) | two-sided | next-lower T | its two-sided p |
|---|---|---|---|---|---|---|
| 5 (audit) | 45 | **T ≥ 36** | 0.004733 | 0.009467 | 35 | 0.017284 |
| 6 (audit) | 54 | **T ≥ 42** | 0.004216 | 0.008433 | 41 | 0.014635 |
| **8 (this extension)** | **72** | **T ≥ 53** | **0.004934** | **0.009868** | 52 | 0.015640 |

Rows 1–2 **reproduce the audit's published 36/45 and 42/54 exactly**, which is
the receipt that row 3 comes from the same construction and not a new one.

* **ORDERING-CONFIRMED** = median within-K gap > 0.05 **and** `T ≥ 53`.
* **ORDERING-NEGLIGIBLE** = median within-K gap ≤ 0.05.
* **ORDERING-INVERTED** = median within-K gap < −0.05 with `T ≤ 19`
  (the symmetric lower threshold, 72 − 53).

Observed `T` may be half-integral (ties count ½) and is compared to the
integer threshold as `T ≥ 53`, the same convention the audit's 36/45 and
42/54 already use.

For reference, the sweep of record read `T = 32/54` ⇒ ORDERING-NEGLIGIBLE at 5
squarings (EXPERIMENT_LOG 2026-08-22 #2, curve 3).

**This threshold is also the one EXPERIMENT_LOG 2026-08-22 #4 defers to.**
The depth-extension harvest read `T = 43.5/54 ≥ 42` at 11 squarings ⇒
ORDERING-AT-DEPTH-CONFIRMED, with a disclosed 1.5-pair margin and a
leave-one-stratum-out fragility, and explicitly routed the robustness
adjudication to "the in-build K=36/40 wave … under a build-time pre-registered
8-strata threshold." That threshold is the `T ≥ 53 / 72` derived above,
pre-registered here **before** any K=36/40 cell runs, and it applies to both
readouts on which the 6-strata test was run (`h_top` at 5 squarings, §7.3;
the fixed-residue 11-squaring rung, #4) — the construction is identical, only
the κ values fed in differ.

### 14.3 Pre-registered null expectations — carried verbatim by reference

**Locked in EXPERIMENT_LOG 2026-08-22 #3 BEFORE this build; that entry is the
authority and nothing here restates or softens it.** In brief, for
navigation only:

* **(a)** CAPABILITY-HOLDS continues at K=36/40 (κ ≥ 0.90, likely ≥ 0.95, both
  recipes) — a frontier here would be a FINDING, not the expectation.
* **(b)** WALL-HOLDS 0/6 at both K.
* **(c)** The **LIVE RISK**, to watch and not assume away: trainable/
  contrastive Gate-0 convergence (CE finite + falling) in the K≈36–40 regime.

Bands otherwise identical to §7 as amended (κ bar 0.90 on ≥ 2/3 seeds).
(c) is instrumented, not merely watched: every frontier spec's
`validity_check` asserts Gate-0 directly on the run's own `loss_history` —
both arms logged, every logged CE finite, final CE strictly below initial —
so a cell whose optimisation collapsed fails its own validity check and routes
to `failed/` instead of entering the curve as a spurious frontier point.

### 14.4 Smoke — REAL CUDA, both new K

`kscaling_smoke.py` (md5 `50eb09c03952b81f70df18eed3c3f05e`), one process per
K, H100 80GB HBM3, torch 2.12.1+cu130, 2026-08-22. Results archived at
`kscaling_build/smoke_results/kscaling_smoke_K{36,40}.json`.

**11 PASS / 0 FAIL / 1 N/A at each K.** Every applicable negative fired with
its hand-predicted message:

| item | K=36 | K=40 |
|---|---|---|
| B — pinned ladder rejected | **N/A** (see below) | FIRED, `h=40 is IDENTITY mod K=40` |
| C — silent residue collision rejected | FIRED, `PAIRWISE residue collisions` on (4,8,16,17,32,40) | FIRED, same on (4,8,16,17,32,44) |
| D — identity residue rejected | FIRED, `h=72 is IDENTITY mod K=36` | FIRED, `h=80 is IDENTITY mod K=40` |
| E — train residue rejected | FIRED, `h=73 … h%K=1` | FIRED, `h=81 … h%K=1` |
| K — unpadded T crashes floor | N/A (pad 0, as at every K ≥ 20) | N/A |

**Item B at K=36 — the one deviation, disclosed.** By §14.1(ii) the pinned
ladder is genuinely sound at K=36, so a rejection test has nothing to fire on.
Recording it as a FAIL ("did not fire") would be false, and inventing a
different fixture to force a firing would be theatre. It is recorded instead
as a **positive** item, `B_POS_pinned_ladder_is_SOUND_at_this_K`, which
asserts that `assert_ladder_sound` does **not** raise and that the six
residues are distinct/admissible. The pairwise-distinctness guard is still
exercised at K=36 by item C, which builds a colliding fixture on purpose, so
no guard goes untested at that K. `_B_EXPECT` in the smoke carries the
hand-derived first-firing substring for K=40 (`IDENTITY mod K`, since the
loop reaches `h=40` before any other rung offends) and deliberately has **no**
K=36 entry.

Measured (item F/L), quoted into the specs rather than re-derived:

| K | params/arm | s/step | proj. GPU-h @20K | ×1.17 | peak mem | SM util med/max |
|---|---|---|---|---|---|---|
| 36 | 97,852,837 | 0.1621 | 0.901 | **1.054** | 7.22 GB | 97 / 100 |
| 40 | 97,860,009 | 0.1806 | 1.003 | **1.174** | 7.69 GB | 97 / 100 |

Param spread over K=12…40 is now 0.051% — still not a capacity curve in
disguise. Util at 97% median at both K ⇒ 1 cell/GPU, no packing warranted
(consistent with the standing declined-packing ruling).

**Spec-level end-to-end check.** Module smokes were proven blind to
spec-level defects on the previous build (the `deep_gap['h=61']` KeyError),
so spec `0146` was additionally run through its **literal `cmd`** at K=40 for
3 steps — identical flags, env, workdir and interpreter, with only `--steps`,
`--out` and `--ckpt-dir` redirected so a 3-step artifact can never sit at a
production path and block the real cell's resume logic (verified after the
run: `/ephemeral/kscaling/{results,ckpts}` contain no K=36/40 file).

**Result: COMPLETED, 6 s.** The banner resolved
`K=40 d_ncr=41 chance=0.0250 ladder=(4,8,16,17,32,60) h_top=60 (residue 20 == K/2)
fixed_dist_probe=44 t_in=286 doc_left_pad=0`; the read-ablation exact-zero
check passed pre- **and** post-train (`max_abs_diff = 0.00e+00`); the same-op
assertion passed for both arms; and — the path that hid the previous build's
launch-losing `KeyError` — the **eval and `build_attribution` fired at step 3**
(`step % eval_every == 0 or step == steps`) without touching a K=24 ladder
literal. Losses 11.0649 → 11.0170 (full_graft), finite throughout.
Artifact: `kscaling_build/smoke_results/spec0146_literal_3step_K40.json`.

The spec's **own `validity_check` string** was then run verbatim (path
redirected) against that artifact, three ways: (1) unmodified ⇒ **FAILS**
`AssertionError: 3` on `step >= 20000` — it has teeth; (2) with *only* the two
count thresholds relaxed (`20000→3`, `100→2`) ⇒ **PASSES**, so the K-identity,
`d = K+1`, `h_top`, `deep_ladder` **and the new Gate-0 `loss_history` clause**
(both arms present, all CE finite, final < initial) all evaluate correctly on
a real runner output; (3) with one ladder digit corrupted (`60 → 61`) ⇒
**FAILS** on the ladder clause.

### 14.5 Job specs and ledger

`matrix-thinking/kscaling_build/job_specs/` — **12 new files, `0140`–`0151`,
none queued**, generated by `gen_job_specs.py frontier`. IDs start at 0140 so
they cannot collide with 0100–0139 (including the retired 0134/0137, whose
cells ran as the calibration pair) or with anything in the queue's history
(verified against `~/queue/{pending,claimed,completed,failed,cancelled}`).

| | frozen-contrastive (primary) | trainable-contrastive (compB) |
|---|---|---|
| K=36 | 0140, 0141, 0142 | 0143, 0144, 0145 |
| K=40 | 0146, 0147, 0148 | 0149, 0150, 0151 |

Seeds 0, 1, 2 per triple. All other hyperparameters held at the audited
G3-B31 values — the recipe is not a variable here either.

Marked **`CANDIDATE -- NOT queue-eligible until audited`**. **SINGLE-gated:
no LICENSE sentinel**, because this wave is not calibration-gated — the sweep
it extends already ran to completion and its internal gate returned
CLEAR-WITH-CONSTRAINTS. It launches directly after its audit.

**Ledger: 13.37 GPU-h** (6 × 1.054 at K=36 = 6.32; 6 × 1.174 at K=40 = 7.04),
plus ~0.05 GPU-h eval-only scoring. **This is above #3's ≈10 GPU-h estimate**,
which was extrapolated from the K≤32 slope before the frontier K were timed;
the measured per-cell cost is used instead. Ceremony tier is unchanged
(10–50 GPU-h ⇒ one audit round on the extension delta). Wall: 12 cells / 8
GPUs = 2 waves × ~1.1 h ≈ **2.2 h**.

`kscaling_battery.py` is **unchanged by this extension** (md5
`5735c788563d9a21f2198c9f5b4793d5`, the battery of record): it reads the
ladder, chance and `d = K+1` guard from `kscaling_config`, so admitting two
more K required no edit to the scorer.
