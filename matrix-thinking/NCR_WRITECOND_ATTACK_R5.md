# NCR write-conditioning — ATTACK ROUND 5

**Target:** `matrix-thinking/NCR_WRITE_CONDITIONING_DESIGN.md` §`DRAFT-R5 (Rev-5,
2026-08-14)` (lines 4019–4589) **plus** the 14 build artifacts in
`matrix-thinking/writecond_build/`, at commit `c77d6ef`, ONLY.
**Date:** 2026-08-14. **Agent:** round-5 adversarial audit (independent of the reviser).
**Frozen, not re-litigated:** the mechanism (R4's V1–V13 and its CERTIFIED-CLEAN
verdict on D1–D3); the novelty verdicts; the premise-battery facts; everything
`§A4-ADJUDICATION` adopted.
**Prior reports:** R1 (BLOCKED 5F/11M/7m), R2 (BLOCKED 5F/12M/8m), R3 (BLOCKED
3F/8M/8m), R4 (REV-REQUIRED 4F/11M/10m).

---

## VERDICT: **REV-REQUIRED** (4 FATAL / 6 MAJOR / 9 minor)

This round was expected terminal on the instrument layer. It is not, and the
reason is narrow and mechanical: **the three artifacts that were never executed
end-to-end are the three that are broken.** Everything Rev-5 *tested* is right —
I re-derived F1's repair independently and it is stronger than the design's own
demonstration; I re-enumerated the partition with my own from-text predicate over
51,170 outcomes and found zero disagreements; all 131 shipped assertions execute
to completion, zero failures. But `stage0prime_eval.py` still **cannot run on the
box** (F2, F3), and the one field the coordinator reads to spend 24.94 GPU-h
**disagrees with its own pre-registration** (F1).

**FATAL** here keeps R4's definition: *would produce an uninterpretable verdict,
void a pre-registered cell, or spend GPU-h that cannot answer the question.*

The four FATALs are: a GO/NO-GO gate strictly weaker than the card that registers
it; a wave-deciding probe whose fresh encoder is built on CPU while every tensor
it is handed lives on `cuda`; a checkpoint restore that dies on the first of three
checkpoints because a per-checkpoint flag was hardcoded to one value; and CONTROL
B, whose R4-repaired Band-0 branch is now satisfiable only by an artifact the
pinned runner refuses to produce.

**Nothing found this round touches the mechanism.** D1–D3 remain clean. Every
FATAL below is a two-to-five-line repair, and none of them re-opens a frozen item.

---

## What is VERIFIED CLEAN (executed by me this round, independently)

| # | item | how verified |
|---|---|---|
| **V1** | **F1's repair is correct and the c-sweep reproduces at real dims.** My own driver (not their test), `K=24, d=25`, compB-matched geometry (measured mean pairwise cos **0.2168** vs R4 V1's 0.2200), real `pinv` `teacher_force_operator`: the repaired gate (i) `L_key(c*·Z) ≤ 3e-4` **passes at every** `c ∈ {0.01, 0.1, 1, 1.5, 10, 100}` (max `L_key_cstar` `3.4e-10 … 3.9e-11`) while the raw gate fails at every `c ≠ 1` (`2.500e-01` at 1.5, `9.801e+03` at 100 — F1's own table). | executed, `r5_A_f1.py` |
| **V2** | **F1's LITERAL impostor is closed — and my construction is harder than the design's.** I built F1's own shape (`‖Z‖_F = 1.0198`, `‖Zw‖ = 0.2000`, transverse norm *fixed*): the old bare `‖Zw‖ ≤ 3.0` gate passes **100% of episodes at every c ≤ 10**; the repaired ratio gate fails **100%** at every c. The shipped test only exhibits the loophole at `c=0.01` (81%); mine exhibits it at `c=1`. | executed, `r5_A_f1.py` |
| **V3** | **Both gates are exactly scale-invariant.** Over a 10⁶× rescale: `max\|ΔZw_ratio\| = 2.2e-07`, `max\|ΔL_key_cstar\| = 1.8e-07`, gate booleans identical. | executed |
| **V4** | **The 0.12 provenance checks out.** `inspect.signature(band2_check)` → `zw_ratio_bound = 0.12` **== `3/25` exactly**, `l_key_bound = 3e-4`; `3/25` is R3's frozen *"‖Zw‖ ≲ 3 at ‖Z‖_F ≈ 25, i.e. ≤ ~12% of the operator norm"* with its conditioning restored. | executed |
| **V5** | **The partition is hole-free and double-fire-free, and the committed `classify` agrees with the design text.** I wrote my **own** predicate from R5.3's prose and ran **51,170 constructed outcomes** (40,936 compB across x × GAP-boundary × C-margin-boundary incl. both exact 0.15 edges; 10,234 compA/primary): **0 holes, 0 double-fires, 0 label disagreements, 0 `ortho_confounded_disclosed` disagreements.** M3's counter-example reproduced exactly (C=0.0664 → WIN, C=0.7000 → PARTIAL, `win_base` identical in both). `classify` raises on compB without `control_c_reading`. | executed, `r5_C_partition.py` |
| **V6** | **Band-0's CONTROL-B repair has teeth.** All 13 shipped checks executed: the pre-repair gate VOIDs a well-formed CONTROL B, the repaired gate passes it, and all six negatives fire (missing clean-eval, mis-flagged clean-eval, `active=False`, 1999/2000, missing `steps_run`, leaked non-B cell). | executed |
| **V7** | **131/131, exactly as claimed.** `35 + 13 + 12 + 18 + 10 + 43 = 131` PASS assertions, all six suites run to completion from the committed tree (`git diff HEAD` empty for `writecond_build/`), zero failures, torch 2.8.0, CPU, fp32, no box contact. | executed |
| **V8** | **Every runner/model signature the two box-only scripts use is real and correctly called.** Pinned runner md5 **`9a93198b642242f512ff8489e32b0a53` re-confirmed**. AST-extracted every call site and checked each against source: `build_task1_document(cfg,pools,gen,batch_size,hop_value,device)`; `query_key(token_ids, query_key_col:int, embed)` (and `query_key_col` **is** a python int, `smoke:466`); `teacher_force_operator(keys_v, values_v)`; `ncr_lm_forward_ablatable(...)` → 7-tuple, indices 4/5/6 = `Z, keys_v, values_v` ✓; `discriminability_metrics(integ, embed, o, entity_ids, tgt_slot)` (480–537); `eval_arm_at_hops(arm,pools,cfg,hops,bs,device,base_seed,read_ablate,teacher_force)`; `load_checkpoint`; `restore_arms_and_opts` → `(arms,opts,gen)`; `R.build_grammar_pools_and_cfg` (module-level alias, runner:277); `R.graft` (runner:263); `nm.binexp_read(Z,q,h)` with its `isinstance(h,int)` assert satisfied; `els.NCREarlyLNModel(d,h).encode(keys,values)`; `ckpt["step"]` / `ckpt["freeze_entity_adapter"]`. Both scripts `py_compile` clean. | executed |
| **V9** | **The checkpoint PATHS are right, verbatim against the recorded convention.** `~/ncr_g3b31_contrastive/results/mob_g3b31_{tag}_s0_ckpts/mob_g3b31_{tag}_s0.ckpt.pt` matches `pbe_supplement.py.txt:16` and `writecond_premise_REPL_{compA,primary}.json`'s own `ckpt` fields character-for-character. (The *flag* that travels with them does not — F3.) | raw artifacts |
| **V10** | **M11's `d−K` generalization reproduces independently at `K=3, d=6`.** `L_write(Z_ideal) = 1.07e-13` (design: 1.03e-13); a null-space-only perturbation leaves `L_key` flat (`1.08e-13`) and drives `L_transverse` to `2.65`; a span(keys)-only perturbation does the exact reverse (`L_key 6.5e-02`, `L_transverse 7.1e-15`). | executed, `r5_I_degenerate.py` |
| **V11** | **R5.6's timing anchor is honest.** The battery's own `elapsed_s`: SUPP **5.75**, REPL_compA 6.26, REPL_primary 10.29, P0P1b **15.02** — the design's "5.75–15.02 at n=256" is exact, and correctly excludes the n=8 smoke (3.70). | raw JSONs |
| **V12** | **The box-only justification is true.** `import ncr_lm_wave1_runner` with the full flat layout on `sys.path` raises `ModuleNotFoundError: No module named 'fla'`, exactly as claimed. The 173,209-param count and the `.encoder.row_out` route both verified on the real classes. | executed |
| **V13** | **R5.7's item-1 disclosed risk is SETTLED, not merely disclosed — and the build's inference is CORRECT.** See §"A risk the design carried that I can close" below. | executed, `r5_J2_hop.py` |

---

# FATAL FINDINGS

### F1 (FATAL — the wave-deciding gate is strictly weaker than the card that registers it)

DRAFT-R5 §R5.4 pre-registers, verbatim:

```
   stage1_gate = "GO" iff ANY (lr, lambda_t) cell reaches BOTH of Band 2's
      repaired gates (L_key_cstar<=3e-4 AND Zw_ratio<=0.12), at BOTH the
      median AND the p90, AT EVERY SCORED HOLD-OUT HOP.
```

The shipped code (`stage0prime_helpers.py:406`) is:

```python
stage1_gate="GO" if any_median else "NO-GO-ON-CURRENT-BAND"
```

`any_p90` is computed, emitted — and **never used in the gate**.

**Executed** (`r5_D_gate.py`, driving the real committed
`item_6_achievability_probe`, stubbing only `band2_check` so the population
statistics land on the card's edge):

```
  any_lambda_t_reaches_band2_median = True
  any_lambda_t_reaches_band2_p90    = False
  stage1_gate EMITTED BY THE CODE   = GO
  stage1_gate REQUIRED BY THE CARD  = NO-GO-ON-CURRENT-BAND
```

This is not a corner case. `quantile(0.90) ≥ median` for any sample, so
`reaches_band2_p90 ⟹ reaches_band2_median` **always**; the card's predicate is
therefore *exactly* `any_p90`, and the shipped one is *strictly weaker* on
precisely the outcome shape R4's M2 predicts is most likely — half the episodes
clean, the tail not. The field is the single string a harvest agent reads to
decide whether 24.94 GPU-h is spent.

**Repair (one line).** `stage1_gate = "GO" if (any_median and any_p90) else
"NO-GO-ON-CURRENT-BAND"`. (Equivalently `if any_p90`, but the conjunctive form
reads back against the card without an implication argument.)

---

### F2 (FATAL — Stage 0′'s wave-deciding item cannot run on the box: CPU model, CUDA data)

`build_fresh_encoder()` (`stage0prime_helpers.py:78-83`) is
`return els.NCREarlyLNModel(d=D_NCR, h=H_NCR)` — **no `.to(device)`**.
`item_6_achievability_probe` takes no device argument, contains no `.to(`, and
`stage0prime_eval.py:141-143` passes none. Every tensor it is handed
(`keys_v`, `values_v`, the per-hop held-out sets) comes from
`run_one_checkpoint(..., device)` with `--device` defaulting to `"cuda"`.

**Executed** (`r5_E_device.py`, reproduced live on this machine's MPS device
because it is the available non-CPU device; the CUDA error is the same class):

```
  build_fresh_encoder() -> parameters live on: cpu
  n_params = 173,209   (== NCR_PARAM_EXACT)
  m.encode(<mps tensors>) raises:
      RuntimeError: Tensor for argument weight is on cpu but expected on mps
```

The job dies on item 6's **first** `encoder_module.encode(keys_v_train, ...)`, i.e.
the wave-deciding gate never produces a reading. The off-box suite is CPU-only by
construction, so it structurally cannot catch this — the same blind spot that let
F3.1/F3.2/F3.3 through in R4.

**Repair (two lines).** `build_fresh_encoder(device)` → `.to(device)`, and pass
`keys_v_train.device` from `item_6_achievability_probe` (which already has the
tensor in hand, so no new plumbing through `stage0prime_eval.py` is needed).

---

### F3 (FATAL — Stage 0′ dies on its FIRST checkpoint: `freeze_entity_adapter` is per-checkpoint and was hardcoded)

`stage0prime_eval.py:109`:

```python
arms, _, _ = R.restore_arms_and_opts(ckpt, pool_report["vocab_size_total"], lr=3e-4,
                                      device=device, freeze_entity_adapter=False)
```

— **for all three tags.** The archived run configs say otherwise:

| cell | recorded `config.freeze_entity_adapter` |
|---|---|
| `mob_g3b31_primary_s0` | **True** |
| `mob_g3b31_compA_s0` | **True** |
| `mob_g3b31_compB_s0` | False |

`save_checkpoint` records that same arg into the checkpoint (`runner:1108`), and
`restore_arms_and_opts` passes it to `build_optimizer` — which **excludes**
`integ.entity_adapter`'s params from the optimizer's param group when True
(`runner:918`) — then unconditionally calls
`opt.load_state_dict(ckpt[arm_name]["opt_state"])` (`runner:1164`).

**Executed** (`r5_G_optshape.py`, reproducing the exact param-group shapes):

```
saved (freeze=True)   param count in group 0: 6
restore (freeze=False) param count in group 0: 8
  load_state_dict RAISES: ValueError: loaded state dict contains a parameter
                          group that doesn't match the size of optimizer's group
```

`CKPTS` is iterated in insertion order — `primary` first — so the job raises
**before item 1 of the first checkpoint**. This is F3.1's failure mode re-created
through a different field: A1 correctly carried over the battery's *path*
convention and dropped the *flag* convention that travels with it. The battery
itself knew: `pbe_repl.py.txt:16` takes `freeze` as `argv[3]` **precisely because
the sibling checkpoints differ**, while `pbe_supplement.py.txt:28` hardcodes
`False` because it only ever touches compB.

**Repair (three lines).** Read it off the checkpoint rather than hardcoding:
`fe = bool(ckpt.get("freeze_entity_adapter", False))`, then pass `fe`. Extend
A1's pre-flight beyond `ls` to print that field per checkpoint (the battery's
recorded configs are a proxy; the authoritative value is inside the `.ckpt.pt`,
which is on the box and was not read this round — see the caveat in §Residual
uncertainty).

---

### F4 (FATAL, Stage-1-scoped — CONTROL B is still unproducible; F4's void-by-construction recurs one layer down)

R4's F4 was closed at the *gate* (V6 confirms the branch and its six negatives).
It is now blocked at the *artifact*.

The chain, as Rev-5 ships it:

1. `band0_checker.py`'s CONTROL_B branch PASSES only if `clean_eval_rec` exists
   and carries `config.teacher_force_operator == False`.
2. That artifact comes from `control_b_clean_eval.py`, which **hard-asserts**
   `ckpt.get("freeze_entity_adapter", False) is True` (line 66) — following D7's
   own registered repair (design:3535-3540, `--freeze-entity-adapter` "for this
   continuation specifically, disclosed as no longer compB's exact recipe").
3. CONTROL B is a **warm-start continuation of the compB checkpoint** (design:2610,
   3453), and compB recorded `freeze_entity_adapter: False`.
4. The pinned runner's own resume gate (`run_two_arm_cell`, runner:1228-1232)
   asserts `ckpt.get("freeze_entity_adapter", ...) == freeze_entity_adapter` and
   **aborts the launch** on mismatch.

So: launch the continuation **with** `--freeze-entity-adapter` → the runner's
resume assert aborts it. Launch **without** → the continuation runs, records
`freeze_entity_adapter=False`, and `control_b_clean_eval.py`'s assert aborts → no
`clean_eval_rec` → `band0_checker` VOIDs CONTROL B. Hand-patching the field in a
copied checkpoint does not rescue it either: the optimizer param-group mismatch
proved in F3 raises inside `restore_arms_and_opts`.

**0.486 GPU-h is again funded for a cell that cannot be produced**, and the one
control answering "does read-side adaptation alone explain the gain" again leaves
no record — R4's F4 sentence, verbatim, one layer down.

**Repair (adjudication, not code).** Three options, all pre-registrable now, none
requiring a runner edit: (a) run CONTROL B **without** the freeze and carry R3's
M4(b) drift confound as a disclosed weakening of the "isolates the read side"
claim — then drop the assert in `control_b_clean_eval.py` to a **warning field**
(`freeze_entity_adapter_verified: False`) rather than an abort; (b) re-run compB
itself with `--freeze-entity-adapter` (a new 4.861 GPU-h cell — expensive, and it
changes the anchor `P0 = 0.0664` every band is calibrated against); (c) drop
CONTROL B. **(a) is the only one that fits the registered budget**, and it is
strictly better than the status quo, in which the cell is bought and discarded.
This is **not** a Stage-0′ blocker — it must be settled before CONTROL B's cell is
launched, not before the probe runs.

---

# MAJOR FINDINGS

### M1 — the timing probe R5.6 recommends cannot be run from the shipped script

R5.6's own recommendation: *"before committing to the full 8-cell/8000-step grid,
run a short timing probe (e.g. `n_steps=500` on one cell)."* The shipped script has
**no knob for it**. `--smoke-only` reduces only `n` (episodes: 256 → 8);
`lambda_t_grid=(0.0,0.1,1.0,3.0)`, `lr_grid=(3e-4,1e-3)` and `n_steps=8000` are
hardcoded at `stage0prime_eval.py:143`. A `--smoke-only` run therefore still
executes **all 64,000 Adam steps**. Running the recommended probe requires editing
the deployed copy on the box — exactly the drift the "deploy the tested module
unmodified" discipline (R5.5) exists to prevent.

**Repair.** `--n-steps`, `--lr-grid`, `--lambda-t-grid` CLI args (or a
`--timing-probe` mode = 1 cell × 500 steps, write JSON, exit). This is the
pre-item my Stage-0′ ruling below turns on.

### M2 — item 6 recomputes a batched SVD of an **unchanging** tensor 64,000 times

`write_supervision_loss` exposes `W=` *specifically* so a caller with fixed keys
can hoist the SVD ("Exposed so a caller computing W once … can reuse it", its own
docstring). `item_6_achievability_probe:372` does not pass it:

```python
out = write_supervision_loss(Z, keys_v_train, values_v_train, lambda_t=lam_t, K=K)
```

so `null_directions(keys_v_train)` — a batched `torch.linalg.svd` over 256
`24×25` matrices — runs on **every one of the 8,000 steps × 8 cells**, on a tensor
verified bit-identical throughout (`max|W(step 1) − W(step 8000)| = 0.000e+00`).
`band2_check` in the curve-logging branch does the same.

**Measured** (`r5_F_cost.py`, CPU, B=256, K=24, d=25, fp32):

| | per-step |
|---|---|
| as committed (SVD every step) | **49.0 ms** |
| with `W` hoisted (one line) | 40.0 ms |
| the batched SVD alone | 9.2 ms = **19% of every step** |

19% of the wave-deciding probe's entire cost is recomputing a constant. Batched
small SVD is *relatively worse* on cuSOLVER than a 173K-param transformer forward,
so the H100 fraction is plausibly higher, not lower. This is a direct, one-line
contributor to the very cost overrun R5.6 flags.

### M3 — a degenerate hop clears Band 2 **vacuously**, and `item_6`'s all-hops-must-pass gate then reads GO off a hop that measured nothing

`L_key`'s normaliser is `‖vᵢ‖² + eps`. With all-zero held-out keys/values the
residual is exactly zero, so the gate passes on nothing.

**Executed** (`r5_I_degenerate.py`):

```
  all-zero keys/values, random Z:
    L_key_cstar_median = 0.000e+00  -> gate(i) pass: True
    Zw_ratio_median    = 0.1935     -> gate(ii) pass: False
  all-zero keys/values, Z with its W-component removed:
    L_key_cstar_median=0.000e+00  Zw_ratio_median=0.000e+00
    band2_at_median=True  band2_at_p90=True      <-- BOTH gates pass
```

Since the gate is `all(hops)`, a hop whose extraction silently produced empty
tensors contributes a **free PASS** rather than a failure. The build's own suite
uses an all-zero held-out set as a fixture (`test_item_6_per_hop_scoring_uses_the_
matching_held_out_set`) but asserts only that the two `o` tensors differ — it never
asserts the gate *rejects* it, so the fixture proves plumbing, not soundness.

**Repair (one line).** Emit `n_valid_episodes` per hop and require
`values_v.pow(2).sum(-1).min() > 0` before scoring; a hop that fails it VOIDs
rather than passes.

### M4 — the `0.12` bound is carried unchanged into M11's `d−K` generalization, where it is a materially different requirement

`band2_check`'s `zw_ratio_bound=0.12` is `3/25`, R3's calibration **at `d−K = 1`**.
M11's generalization aggregates `‖ZWᵀ‖_F` over `d−K` directions, and the
no-suppression baseline scales as `√((d−K)/d)`.

**Executed** (`r5_H_misc.py`, generic `Z`, B=512):

| (K, d) | d−K | `E[‖ZW‖_F/‖Z‖_F]` | `√((d−K)/d)` |
|---|---|---|---|
| 24, 25 | 1 | **0.1984** | 0.2000 |
| 20, 25 | 5 | 0.4463 | 0.4472 |
| 12, 25 | 13 | **0.7206** | 0.7211 |
| 3, 6 | 3 | 0.7011 | 0.7071 |

At `d−K=1` the `0.12` bound asks for ~40% suppression below an untrained
operator's own baseline — a real, calibrated gate. At `d−K=13` the same number
asks for **~6× more** suppression than the calibration ever established. M11
exists *for the spearhead's K-ladder* (`NCR_KLADDER_DESIGN.md` is live), and the
module's docstring correctly notes it reduces to the carded form at `K=d−1` while
saying nothing about the threshold's validity beyond that point.

**Repair.** Either scale the default (`0.12·√(d−K)`) or — better, since no
calibration exists for `d−K>1` — `assert d - K == 1` whenever the default is used,
forcing an explicit, disclosed re-derivation at the first K-ladder rung.

### M5 — item 3 is a registered STOP gate that does not short-circuit the most expensive item

The card: *"Item 3 (reachability): GATES. FAIL … ⇒ STOP, escalate — rescope before
Stage 1."* `run_one_checkpoint` computes `item3` (line 122) and then, for compB,
runs the 8-cell × 8,000-step item 6 (line 141) **regardless of its verdict**. If
item 3 FAILs, the entire item-6 budget — the whole cost overrun R5.6 flags — is
spent on a run the card has already stopped. This is not hypothetical: item 3
gates `cond(row_out) ≥ within_episode_ratio_p99`, and R4's own measured p99 at
compB geometry is **19.0**, a value a `(25,64)` weight matrix's condition number
does not obviously clear.

**Repair.** Either branch on `item3["gate_pass"]` before item 6, or state
explicitly in the card that item 6 runs regardless **because** it measures
achievability directly and supersedes item 3's proxy. Both are defensible; the
silence is not.

### M6 — `required_row_norm_p99` contains the **max**, and A3's p99 statistic is never computed at all

`item_3_reachability:239` emits `required_row_norm_p99=keygeom["row_norm_max"]`.
`item_1_2_keygeom` computes only `row_norm_med` and `row_norm_max` — there is no
row-norm p99 anywhere in the build. A3 asks for the absolute ceiling *"vs
`‖Z_ideal‖` row-norm **p99**/max"*; the build ships one of the two under the
other's name. It biases conservative (max ≥ p99) so it cannot manufacture a false
GO — but item 1/3's readings are explicitly *"the coordinator's own reading, a
judgment call"*, and this is a mislabeled number inside that judgment.

---

# minor findings

**m1 — R5.1 mis-transcribes its own executed output.** The design states the
c-sweep's `L_key_cstar` max ran *"`4.35e-11` to `4.27e-11` across all six."* The
actual printed values are `3.386e-10` (c=0.01), `4.350e-11`, `3.950e-11`,
`4.268e-11`, `4.239e-11`, `3.916e-11` — the stated range is two of the six, and
the `c=0.01` value is an order of magnitude outside it. Verdict-immaterial (all
≤3e-4 by ≥7 orders), but F1 itself *was* a transcription drop; this is the same
class of hygiene failure in the paragraph fixing it.

**m2 — the `c*` gate's scale-invariance has a floor, undocumented.** With
`eps=1e-6` inside `den = Σ‖Zkᵢ‖² + eps`, a perfect-direction operator stops
clearing gate (i) once `‖Z‖_F` falls below ≈`3e-2`. Executed at `‖Z_ideal‖_F ≈
28`: passes at `c=1e-3` (`L_key_cstar = 2.8e-06`), fails at `c=1e-4`
(`2.0e-02`). The valid domain covers F1's own regime (`‖Z‖_F ≈ 1–26`) with 1.5
orders of headroom, so nothing is at risk — but the design claims exact scale
invariance without a domain, and one should be recorded.

**m3 — `band0_check`'s CONTROL_B branch PASSES a `steps_run = 0` record.**
Executed: `passed=0, steps_run=0` → `{'verdict': 'PASS', ...}`. A continuation
that never trained satisfies `passed == steps_run`. Add `steps_run > 0`.

**m4 — `band0_check` raises `KeyError` rather than VOIDing on a malformed
record.** Executed: a `rec` missing `teacher_force_check`, or a `config` missing
`teacher_force_operator`, crashes the harvest. Loud (so not dangerous), but the
band's own registered remedy is VOID.

**m5 — `control_b_clean_eval.py` cites `M4(a)`/`M4(b)` with no round qualifier.**
Those are **R3**'s M4 (Control B's readout + adapter drift). **R4** also has an M4
— about item 3's statistic mismatch — so in a five-round record the bare ID
collides. Qualify every finding ID with its round.

**m6 — `HOPS = (1, 13, 37, 61)` in `stage0prime_eval.py` is dead.** Item 6 scores
`(1, 61)`, hardcoded at line 140; nothing reads `HOPS`. (In
`control_b_clean_eval.py` the same constant *is* used and is correct.)

**m7 — `item_6` never calls `.eval()` before held-out scoring.** Harmless *here*,
and I verified why rather than assuming: `BindingEncoder` sets `dropout=0.0` on
both its `TransformerEncoderLayer` and its `MultiheadAttention`
(`model_v4.py:41-51`), and LayerNorm is mode-independent. But this is an unguarded
dependency on a module the program is actively editing. One line: `.eval()` before
the scoring loop, `.train()` after.

**m8 — the C-margin boundary is a float subtraction.** `(x - control_c) > 0.15`
means "exactly 0.15" is decided by the representation of a difference (my own
probe hit `0.15000000000000002` → WIN). Immaterial on real data — but the design's
claim to have swept *"the exact `0.15` boundary"* is not literally checkable in
this form.

**m9 — three redundant `build_grammar_pools_and_cfg(seed=0)` calls.**
`run_one_checkpoint` rebuilds the GPT-2 tokenizer and entity pools once per
checkpoint. Hoist it out of the loop (the battery's own scripts each build it once
because each touches one checkpoint).

---

# A risk the design carried that I can CLOSE (positive finding)

**R5.7 item 1 — "train once, score per hop" — is not a risk. The inference is
correct, and it is verifiable off-box.**

The design flags this as *"this build's single most exposed engineering judgment
call … read off `build_task1_document`'s own code, never verified empirically",*
and offers two ways to settle it: (a) read `grammar_rd.sample_batch_rd`'s
construction directly, or (b) an on-box A/B. **Option (a) settles it decisively,
at zero GPU cost.** In `sample_batch_rd`, `hop_set` is referenced at exactly two
lines of the body:

```
   +53: hops_pool = torch.tensor(hop_set, ...)
   +54: hops = hops_pool[torch.randint(0, len(hop_set), (B, Q), generator=gen, ...)]
```

Every quantity that determines `keys_v`/`values_v` is computed strictly **before**
line +53: `entity_ids` (+20), `succ` (+22), `key_ids` (+23), `value_ids` (+24),
`item_pos`/`key_pos` (+36), and both `token_ids.scatter_` calls that write the KEY
and VALUE tokens (+41, +43). `integ.extract_kv` gathers exactly those positions.

**Executed** (`r5_J2_hop.py`, real `grammar_rd`, synthetic pools, same generator
seed, `hop_set=(1,)` vs `(61,)`):

```
  entity_ids / succ / key_ids / value_ids / item_pos / token_ids / rel_id
      -> bit-identical across hop values: True (all seven)
  hops, tgt_slot  -> differ (hop-conditioned, as expected)
```

The K bind-clause key/value token ids — the sole inputs to `keys_v`/`values_v` —
are **provably hop-independent**. R5.4's disclosed scoping decision is therefore
**correct**, its half-cost saving is real, and the risk can be struck from the
record rather than carried into the harvest. (The design should replace the
disclosure with this citation: `grammar_rd.py:494-495` vs `:436-440`.)

---

# RULINGS

### Stage 0′ — **MAY NOT RUN AS CARDED. CLEARED after F1+F2+F3+M1 (≤ 10 lines total).**

As shipped it raises inside `restore_arms_and_opts` on the first checkpoint (F3),
would raise again inside item 6's first `encode` if it got past that (F2), and its
one wave-deciding output field would misreport if it got past both (F1). None of
this touches the item structure, the bands, or the mechanism. After the four
repairs the card is sound and Stage 0′ should run — its gate is genuinely
load-bearing and correctly ordered above the 24.94 GPU-h wave.

**Everything else about the card is kept verbatim**, including A1's paths (V9),
A2's `.encoder` route, A3's spec statistic, A4's polarity fix, A5's `1/B`
correction and its disclosure of the un-reproduced `λ_w` curve, A7's cliff check,
A8's seed correction, the single-process/three-checkpoint structure, the
tmux-by-name discipline, and the `timeout 2400` wrapper.

### The timing probe — **RULED IN. It gates the full grid.**

Run **1 cell × 500 steps first**, read real H100 ms/step, re-derive Stage 0′'s
ceiling from it, *then* launch the 8-cell grid under `timeout`. Two reasons this
is not ceremony: (i) the probe **cannot currently be run at all** without an
on-box edit (M1) — ruling it in forces the CLI knob to exist, which is itself the
repair; (ii) M2 shows 19% of every step is a recomputed constant, which a real
ms/step reading would expose immediately and a one-line hoist would remove. The
probe costs minutes and removes the last unverified number in the build, exactly
as R5.6 argues.

### The two disclosed risks

| risk (R5.6/R5.7) | ruling |
|---|---|
| "train once, score per hop" inference unverified | **NOT A RISK — CLOSED.** Settled by their own option (a), executed off-box (§above). Strike the disclosure, cite the source lines. |
| Stage 0′ cost overruns the "~0.2 GPU-h" estimate | **NEEDS A PRE-ITEM: the timing probe, ruled in above, plus M1's CLI knob and M2's one-line SVD hoist.** Not blocking beyond that — even a generously priced ~1 GPU-h Stage 0′ is negligible against the ≤35 GPU-h hard-capped wave and does not change the ceremony tier. The design's own honest correction here was right; it just shipped without the means to act on it. |

### What the build ceremony MAY prepare for Stage 1, pending Stage 0′'s reading

Everything downstream of the *verified-clean* mechanism and the *verified-clean*
adjudication layer (V1–V13) — none of it can be invalidated by F1–F4, which touch
one boolean, two device/flag lines, and one control's launch recipe:

- `write_supervision_loss.py` **as-is** (V1–V4, V10) — including the `W=` reuse
  path, which M2 says to start using.
- `band_partition.py` **as-is** (V5) — the partition and the M3 margin predicate
  are independently confirmed correct.
- `band0_checker.py`'s gate logic **as-is** (V6), plus m3's `steps_run > 0` and
  m4's malformed-record VOID.
- `write_diag.py` and `config_provenance.py` **as-is**, and the named
  `eval_arm_at_hops` wiring point (runner:942) — the one-line call-site edit into
  the archived runner remains out of scope until the Stage-1 build.
- Per-step separated `‖∇_Z L_key‖` / `‖∇_Z L_transverse‖` logging (D7/F2's
  monitored risk) — the loss module already returns both sub-losses un-summed.

**May NOT be prepared:** any Stage-1 cell; any committed `λ_t`; any band threshold;
and **CONTROL B's launch card**, which is blocked on F4's adjudication.

---

# Residual uncertainty (stated, not smoothed over)

- **F3's checkpoint field is inferred, not read.** `config.freeze_entity_adapter`
  in the archived result JSONs is the run's own CLI arg, and `save_checkpoint`
  writes that same arg into the `.ckpt.pt` (`runner:1108`); the `.ckpt.pt` files
  themselves are on the box and were not opened this round (no box contact, per
  the charter). The inference is tight — the runner's own resume assert makes a
  divergence between the two impossible — but A1's pre-flight should print the
  field rather than trust the chain.
- **`ncr_models.py` on the box is assumed identical to `matrix-thinking/ncr/
  ncr_models.py`.** `stage0prime_helpers`' `try/except` import resolves to the box
  copy when deployed flat; only the repo copy was verifiable here.
- **R5.7's items 2–5** (the `c*` proxy's non-identity with `L_key`'s true
  minimizer; item 5's `1e3` predicate inherited as an "e.g."; the
  verbatim-duplicate drift check; the all-hops-vs-any-hop strictness) are all
  correctly self-flagged and none is falsified by anything I ran. On the fourth:
  I confirm the `discriminability_metrics` / `ortho_regularization_loss` bodies in
  `stage0prime_helpers.py` currently match runner lines 480-537 / 714-742, and
  endorse the registered mitigation (assert the runner's md5 in the test).

---

# Reproduction

All sims pure-CPU (one MPS device-mismatch demonstration), fp32, torch 2.8.0, no
GPU, no box contact, no SSH. They import the **actual committed** build modules
(`write_supervision_loss`, `band_partition`, `band0_checker`,
`stage0prime_helpers`) and the **real** `grammar_rd` / `ncr_models` /
`chapter2/model_v4`, never a transcription.

```
scratchpad/r5_A_f1.py          V1-V4: F1 c-sweep + F1's LITERAL impostor + 0.12 provenance
scratchpad/r5_B_eps.py         m2: the eps-floor domain of the c*-rescale
scratchpad/r5_C_partition.py   V5: my own from-text predicate, 51,170 outcomes
scratchpad/r5_D_gate.py        F1: the GO/NO-GO divergence, executed on the real function
scratchpad/r5_E_device.py      F2: CPU-model/CUDA-data, reproduced on MPS
scratchpad/r5_F_cost.py        M2: the 64,000 redundant SVDs, timed
scratchpad/r5_G_optshape.py    F3: the optimizer param-group ValueError
scratchpad/r5_H_misc.py        m3/m4 (Band-0 negatives), M4 (0.12 at d-K>1), V11 (timing)
scratchpad/r5_I_degenerate.py  M3 (vacuous Band-2 pass), V10 (M11 at K=3,d=6)
scratchpad/r5_J_hop.py         V13: sample_batch_rd's hop_set reference sites
scratchpad/r5_J2_hop.py        V13: bit-identity of bind-clause content across hops
scratchpad/r5_sigcheck.py      V8: AST extraction of every runner call site
```

(scratchpad root: `/private/tmp/claude-501/-Users-samuellarson-Experiments-learned-representations/be705417-f189-4cd8-8024-24cf6a0130a0/scratchpad`)

**Raw artifacts read this round:**
`matrix-thinking/writecond_build/*` (all 14 files, in full);
`experiment-runs/2026-07-30_ncr_g3b31_contrastive_grid/ncr_lm_wave1_runner.py`
(md5 `9a93198b642242f512ff8489e32b0a53`, re-confirmed) and its three
`mob_g3b31_*_s0.json` (the `freeze_entity_adapter` configs — F3);
`experiment-runs/2026-07-17_ncr_gate3_wave1/ncr_lm_wave1_smoke.py`;
`experiment-runs/2026-08-13_ncr_writecond_premise_battery/{pbe_repl,pbe_supplement}.py.txt`
and all six result JSONs (paths, `elapsed_s`);
`matrix-thinking/{ncr/ncr_models.py, ncr/ncr_earlyln_scale.py, chapter2/model_v4.py,
deltanet_rd/grammar_rd.py}`; `NCR_WRITECOND_ATTACK_R4.md` in full;
`NCR_WRITECOND_ATTACK_R3.md` (M4's own text, for m5).

**Not re-litigated:** the mechanism (R4 V1–V13, D1–D3), the novelty verdicts, the
premise-battery facts, anything `§A4-ADJUDICATION` adopted. **No repo file other
than this report was written; no STATE.md / EXPERIMENT_LOG.md edit; no commit; no
box contact; no GPU spend.**
