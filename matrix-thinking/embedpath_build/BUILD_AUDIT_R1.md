# EMBED-PATH INTERVENTION — BUILD AUDIT R1 (independent)

**Auditor: independent agent (not the implementer), 2026-08-21.** Repo commit
at audit time: `6f37b7b`. Audited artifacts: `matrix-thinking/embedpath_build/`
(BUILD_REPORT.md, `embed_path_runner.patch`, `job_specs/*.json` ×12,
`run_repl_wave3.sh`, `verification/`, `logs/`) against the binding spec
`matrix-thinking/NCR_EMBED_PATH_DESIGN.md` `## DRAFT-R1`, the executed attack
round `NCR_EMBED_PATH_ATTACK_R1.md` (591301a), and `EXPERIMENT_LOG.md`
2026-08-18 #14–#17 / 2026-08-21.

**Every finding below is backed by a command this auditor ran.** Box work was
confined to GPUs 2/3/4 (`youthful-indigo-turkey`); GPUs 0/1 (PI vLLM, 73 733
MiB each) were never touched. The pinned runner's md5 was re-verified as
`9a93198b642242f512ff8489e32b0a53` **after** every box operation in this audit;
the patched copy is unchanged at `d0ba4712f026dfdb00ba14d277a4d486`;
`~/queue/pending/` is still empty and no `PAUSE`/`STOP` sentinel exists. All
audit scratch (≈13 GB of probe checkpoints) was deleted from the box.

**Counts: 0 FATAL · 6 MAJOR (3 requiring pre-launch action, 3 raised-and-
discharged by this audit's own execution) · 8 minor.**

---

## 0. Executed verification ledger (what this audit actually ran)

| # | Check | Command / artifact | Result |
|---|---|---|---|
| V1 | Patch applies to the pinned runner and reproduces the deployed file | `patch -p1` on a clean copy of `9a93198b…` | md5 → `d0ba4712f026dfdb00ba14d277a4d486`, **exact match** to the box's `~/ncr_embedpath/ncr_lm_wave1_runner.py` |
| V2 | `_non_ce_term` is exact by linearity | direct read `pinned_runner.py:827-852` | `total_loss = ce + aux_w*aux + ortho_w*ortho`, no other terms → the split is algebraically exact, not an approximation |
| V3 | CPU synthetic suite, clean state, local | `python3 verify_cpu_synthetic.py` (torch 2.8.0, macOS) | reproduces the archived log **bit-for-bit** (ratios 0.180913 / 0.147566, clip 0.071827, 7/7 scope-exact, forced-fail fired) |
| V4 | **Production-path** cut-assert negative test | `/tmp/audit_negatives.sh` N2 — deleted the shipped `target_w.grad.sub_(...)` line from the *shipped* function, ran both targets on real CUDA | `AssertionError` fired for **both** `embed` and `entity_adapter` |
| V5 | `RUNNER_TAG` guard negative test | N1 — mutated the constant in a copy of the shipped file, ran `main()` | fired: `AssertionError: RUNNER_TAG changed to 'ncr_gate3_wave1_runner_v2_TAMPERED' …`, exit 1. N1b: unmutated file passes the assert and proceeds |
| V6 | `assert_conduit_has_teeth` behaviour | N3 on the shipped function | ratio 1e-6 vs floor 5e-5 → **FIRED**; ratio 1.0 → passed; `cut_active=False` → passed (no assertion, per m2's fix) |
| V7 | Full entry point from clean state (`/tmp/embedpath_verify` wiped first) | `verify_run_two_arm_cell.py` on GPU 2 | A ABORTED-BUDGET@10 PASS (peak VRAM **6.11 GB**), B matched resume→COMPLETED PASS, C close-target mismatch fired, C2 seed-trap fired, D parity PASS |
| V8 | **D-M6 flag-OFF parity as R1.11 literally specifies** (200 steps, seed 9999, loss trajectory **and both arms' final `state_dict()`s** `torch.equal`) | `/tmp/audit_parity_strict.sh`, GPU 4, patched vs pinned as separate processes | loss trajectory identical (9 pts); **414 / 414 tensors `torch.equal`, 0 different** |
| V9 | Scoring path compatibility (the build patched a *copy*, so the scorer is now different code) | `/tmp/audit_scoring_compat.sh` — the **unpatched** pinned runner (what `pbe_repl` imports through the `~/ncr_writecond` symlink) loading a patched-runner checkpoint | `load_checkpoint` OK, extra `close_target` key ignored, `restore_arms_and_opts` OK, `eval_arm_at_hops` OK (h=1 1.0, h=61 1.0) |
| V10 | Throughput root-cause | `/tmp/audit_throughput_rootcause.py`, GPU 2 | see MAJOR-2 — decisive |
| V11 | 2 000-step conduit_ratio probe, both targets (10× the build's sample) | `/tmp/audit_long_conduit.py`, GPU 3 | see MAJOR-1 |
| V12 | 12 job specs vs the queue's real schema, the runner's real CLI, and `run_repl_wave3.sh`'s real path grammar | scripted cross-check + box `queue_worker.sh` read | clean — see §4 |
| V13 | Placement | `nvidia-smi --query-compute-apps`, `~/queue/queue_worker.sh:107-112` | GPUs 0/1 each hold a compute-app PID; worker gate `napps>0 \|\| mem>=2048` per its **own** GPU → g0/g1 back off automatically |

---

## 1. The three disclosed judgment calls — adjudicated

### (a) The two corrections to DRAFT-R1's own text — **FAITHFUL TO INTENT, both are bug fixes, neither is scope creep. RATIFIED.**

**The `opt` parameter / `opt.step()`.** DRAFT-R1 R1.6's training-loop snippet
states *"clip_grad_norm_ and opt.step() are ALREADY DONE inside
assemble_closed_grads_ for this arm — do NOT re-run the shared finite/clip/
opt.step() block below for it"*, while the function body in the same section
receives no `opt` and never steps. Re-deriving what the loop **must** do from
D-F5 rather than from either text: D-F5 requires the clip coefficient to be the
one compB's own `backward()` would produce, i.e. `clip_grad_norm_` must run
**exactly once**, over the **full combined** gradient, **before** the cut. If
the caller fell through to the shared block, the gradient would be clipped a
*second* time — and that second clip would be computed over a **post-cut**
global norm, which is precisely the every-other-parameter rescaling confound
D-F5 exists to eliminate. Threading `opt` in and stepping inside the helper is
therefore the *only* construction consistent with D-F5. The patch does exactly
this (`_finite_clip_step()`; `opt.step()` at patch line 172) and correctly
guards the shared block with `if not close_this_arm:` (patch line 342).

**The `"stepped"` bool.** R1.6's two early-return branches call
`total_loss.backward()` and return **without** finite-check, clip, or step —
and, given the same section's "do NOT re-run the shared block" instruction,
that silently discards the step's parameter update *and* leaves `n_skipped`
unincremented. The invariant the shared block enforces is: on every step, for
every arm, **either** the optimizer steps **or** `n_skipped` increments. The
build's fix restores exactly that invariant and reports it to the caller
(`if not grad_diag["stepped"]: n_skipped[arm_name] += 1`, patch lines 324-325).
Bookkeeping semantics verified against the pinned runner's own
`n_skipped`/`loss_hist` conventions (`pinned_runner.py:1298`, `1354-1360`) —
identical.

Both branches are provably dead in this wave's launch config, and this audit
confirmed that empirically rather than by assertion: `aux_read_loss_weight=0.5`
and `ortho_reg_weight=0.1` are both `>0` so `_non_ce_term` always returns a
tensor, and `grad_rest_target` was non-`None` on **4 000 / 4 000** real steps
across both targets in V11 (`all_cut_active: True` in every block). Fixing
unreachable code is still correct; leaving it broken would have been a latent
trap for the deferred `--freeze-entity-adapter --close-target` follow-on.

### (b) `PINNED_MIN_RATIO` at measured-min/10 — **the has-teeth framing is wrong in the build report; the value is not fatal, but it must be lowered. MAJOR-1.**

The audit brief asks: *at 5e-5, can the assert still detect a broken cut?*
**It cannot — and it could not at any value, including DRAFT-R1's 0.5×-mean.**
This is the central adjudication and it is a structural fact, not a tuning
question:

`conduit_ratio` is computed from `grad_rest_target = autograd.grad(non_ce,
[target_w])` and `grad_ce_target = combined_before_clip − grad_rest_target`,
**both before the cut** (patch lines 129-149). The `sub_` at patch line 164
happens *after*. Therefore, if the cut silently failed — `sub_` removed,
no-op'd, or the aux gradient still flowing — `conduit_ratio` reads **exactly
the same number**. A floor on it is mathematically incapable of separating the
two worlds. So 5e-5 does not "lose the teeth for detecting a broken cut"; that
was never this assert's job.

The detector that **is** responsible for "did the cut actually happen" is the
per-step `torch.allclose(target_w.grad, grad_ce_target_clipped, rtol=1e-5,
atol=1e-6)` at patch line 170. This audit tested it **on the shipped code path**
(V4/N2 — deleting the real `sub_` line from the real function, on real CUDA
tensors) and it fired for both `embed` and `entity_adapter`. **The cut has
teeth, and they are proven on the production path.** (The build's own
"forced-fail" test did *not* prove this — see MAJOR-6.)

What the floor *is* responsible for is D-A2's **vacuity** question: is the flag
closing a path that carries meaningful gradient? At 5e-5 the answer is that the
check is degenerate — 5e-5 means the aux+ortho share is 0.005 % of CE's, which
would pass a conduit that is vacuous by any reasonable standard. The teeth for
*that* question are gone at 5e-5. But they are also unrecoverable **by a
per-step floor**, because the measured per-step ratio spans five orders of
magnitude and vacuity is a property of the *run*, not of a step:

**V11, 2 000 real training steps per target (10× the build's sample, LR
schedule mirrored, seed 1 = compE_s1's own seed):**

| target | mean | median | p5 | p1 | p0.1 | min | max | frac < 1.0 | shipped pin | **min / pin** |
|---|---|---|---|---|---|---|---|---|---|---|
| `embed` | 0.9939 | 0.1327 | 3.597e-3 | 8.400e-4 | 2.766e-4 | **1.151e-4** | 88.76 | 76.9 % | 5e-5 | **2.30×** |
| `entity_adapter` | 16.575 | 12.896 | 3.200 | 1.339 | 4.654e-1 | **3.624e-1** | 312.9 | 0.45 % | 0.049 | **7.40×** |

Three consequences:

1. **The run-level has-teeth question is already answered, affirmatively, and
   the data to answer it per-cell is already being recorded.** Embed's mean
   conduit_ratio ≈ 1.0 and adapter's ≈ 16.6: the aux+ortho gradient share into
   each target is of the same order as (or far larger than) CE's own. Not
   vacuous. And `rec["close_target_diag"]["history"]` logs
   `[step, conduit_ratio, clip_coef, cut_active, stepped]` at `LOG_EVERY=25`,
   i.e. **≈800 points per 20 000-step cell** — ample for a run-level verdict.
   The has-teeth **gate belongs at harvest**, over that history, with the
   sample size the question actually needs. No code change is required to do
   this; the artifact already exists.
2. **The per-step floor's only remaining live effect is abort risk.**
   `assert_conduit_has_teeth` is called every step with no exception handling
   (patch line 322); an `AssertionError` propagates out of `run_two_arm_cell`
   and kills the cell. The build's 10× margin (measured min 5.5e-4 over 200
   steps) shrank to **2.30× at 2 000 steps** — i.e. the margin is a function of
   how long you look, which is exactly the signature of an unbounded lower
   tail. The decile running-minimum shows *where* the risk lives: deciles 2-5
   (steps ≈200-1000, right after warmup ends while `ortho_loss` collapses)
   reach 1.15e-4 – 4.3e-4, while deciles 6-10 sit at ≈2e-3. The dangerous
   window is the first ~1 000 steps of every cell, and my probe sampled that
   window at exactly one seed. Extrapolating the observed lower tail
   (p1 → p0.1 → min) across 8 compE cells, this auditor's estimate is a
   **30–60 % probability that at least one compE cell aborts.** The attrition
   rule (R1.2) tolerates **one** loss out of 8; two voids the compE read.
3. **A crash is not silently absorbed but it is expensive.** The record on disk
   would carry `status: "RUNNING"` (set at `rec` construction, only overwritten
   at `final_status`), so the spec's `validity_check` correctly routes it to
   `failed/` — no false COMPLETED. But with `--ckpt-every 10000` a crash at
   e.g. step 14 000 loses 4 000 steps, and a naive requeue resumes from the
   step-10 000 checkpoint with the data generator restored and re-hits the same
   deterministic batch. Recovery requires a human lowering the floor — during a
   grant that closes in ~5 days.

**Adjudication:** the shipped value is *not* FATAL (it removes no detection
capability that a higher value would have had, and the real cut detector is
independently proven). It *is* a MAJOR: it buys nothing and risks a cell.
**Lower it to a pure degenerate-zero/NaN tripwire and move the has-teeth
verdict to a pre-registered harvest aggregate** (numbers pinned in §6 below,
from V11, before any wave data exists). `NaN > floor` is `False` at any floor,
so NaN detection is preserved.

### (c) The 10× throughput discrepancy — **ROOT-CAUSED. The build's hypothesis is wrong; its conservative choice is right. MAJOR-2.**

BUILD_REPORT §3.4 attributes the second invocation's 10× speed-up to "almost
certainly a Triton JIT-kernel-cache effect." **It is not.** `verify_throughput.py`
deletes `out_path` (line 25) but **not** the checkpoint directory, and
`run_two_arm_cell(steps=300, ckpt_every=1000)` saves a checkpoint at
`step == steps` (`pinned_runner.py:1394`). The second invocation therefore
resumes at step 300 and executes `for step in range(301, 301)` — **zero
training steps.** It is a vacuous re-run, not a faster one.

Executed proof (V10, GPU 2, same script shape, `/tmp/audit_throughput_rootcause.py`):

```
### invocation 1 (fresh ckpt dir)
=== first_fresh === elapsed_s=53.33 per_step_s=0.1778 gpuh20k=0.9876
    status=COMPLETED step=300 peak_vram_gb=6.11 n_loss_hist_points=13
### invocation 2 (ckpt from invocation 1 still present)
[tp_probe] RESUMING from checkpoint at step 300 ...
[tp_probe] read-ablation exact-zero check PASSED (pre-train, ...)
[tp_probe] read-ablation exact-zero check PASSED (post-train, ...)   <-- no step lines at all
[tp_probe] COMPLETED at step 300/300 in 47s
KeyError: 'loss_history'                                             <-- rec never got one
```

`peak_vram_gb` 6.11 → 3.20 in the build's own two runs is the same tell (no
backward pass on the second). The archived `verify_throughput.log` likewise
contains **no `step N/300` lines**.

**Settlement:** the slow numbers are real and are reproduced independently —
this audit measured **0.1778 s/step → 0.988 GPU-h @ 20 000 steps** for
`close_target=embed` against the build's 0.1772 → 0.9843. Cross-validated a
second way: V11's single-arm training loop ran at 0.0847 s/step, and
`run_two_arm_cell` trains **two** arms per step (2 × 0.0847 ≈ 0.169 ≈ 0.178
with loop overhead). Cross-validated a third way against the real archive:
`mob_g3b31_compB_s{0,1,10,11}` completed 20 000 steps in 2 924–3 016 s =
**0.812–0.838 GPU-h**, so the specs' 0.9843/0.9423 are ~15-20 % conservative,
not an order of magnitude wrong.

**Budget consequence: the 11.6436 GPU-h ledger stands as a conservative
over-estimate; the realistic wave cost is ≈10.0–10.5 GPU-h.** No re-derivation
is needed and no spec `gpu_h_estimate` needs changing. The `--ceiling-gpuh 6.0`
per-cell cap is ~6× above the true rate, i.e. inert as a constraint but present
as a runaway guard. This was the correct call by the build; only the stated
*reason* was wrong.

**Secondary finding (reproducibility):** the retained artifact
`matrix-thinking/embedpath_build/logs/verify_throughput.log` is the **vacuous
zero-step run**, and it contradicts the table in BUILD_REPORT §3.4 that it is
filed as evidence for. The real first-invocation measurement has no retained
log anywhere in the repo. Under this repo's own "save the exact script that was
run alongside experiment results" rule that log should be annotated or replaced
with V10's output.

---

## 2. The patch, line by line against the pinned runner

**Provenance (V1).** `patch -p1` of `embed_path_runner.patch` onto a fresh copy
of the pinned `9a93198b…` reproduces `d0ba4712f026dfdb00ba14d277a4d486`
byte-for-byte — the diff in the repo *is* the deployed change, with nothing
applied out of band. The pinned file itself is untouched (re-verified after all
audit operations).

**Exactness of the split (V2).** `compute_arm_losses` (`pinned_runner.py:827-852`)
builds `total_loss = ce_loss + aux_read_loss_weight*aux_loss +
ortho_reg_weight*ortho_loss` with no other terms and no reweighting, so
`_non_ce_term`'s reconstruction is exact by linearity of differentiation and
`grad_ce_target = combined − grad_rest` is correct up to float
non-associativity — which is exactly the tolerance tier D-F3 pinned. The
construction never takes `total_loss − ce_loss` (F2's killed form) and never
takes a second `autograd.grad(ce_loss, …)` (D-F3's non-associativity source).
`retain_graph=True` on the single targeted call, `backward()` last — correct
ordering, matches R1.6.

**Both close targets.** `target_w` is selected by dict literal, no positional
`list.index` (R1.14 m1 discharged by construction). V4 confirms the production
assert fires for both; V7/V11 confirm both run clean at scale. `cut_active` was
`True` on 4 000/4 000 probe steps for both targets.

**Flag OFF.** `close_this_arm = close_target is not None and arm_name ==
"full_graft"` (patch line 317); with `close_target=None` the original
`total_loss.backward()` and the original shared finite/clip/step block both run
unmodified. The only other OFF-path deltas are inert: a `close_diag_hist = []`,
three `if close_target is not None:` guards, an extra `close_target: None` key
in the checkpoint dict, and the `RUNNER_TAG` assert (a no-op when the tag is
unchanged — V5 confirms it passes). **V8 discharges D-M6 as literally
specified: 200 steps, seed 9999, patched vs pinned as separate processes —
identical loss trajectory and 414/414 tensors of both arms' final
`state_dict()`s `torch.equal`.** The prior archive is not at risk.

**Resume mid-run.** `ckpt_close_target = ckpt.get("close_target", close_target)`
followed by `assert ckpt_close_target == close_target` (patch lines 260-266),
mirroring the seed-trap and freeze-trap asserts exactly, and `close_target` is
threaded into all four `save_checkpoint` call sites (periodic, STOP,
ABORTED-BUDGET, final). **V7 test C confirms the mismatch assert fires on a
real resume; test C2 confirms the pre-existing seed trap still fires; test B
confirms a matched resume completes with `close_target_diag` present.** So a
restart cannot silently change condition.

**Ordering interactions with the pre-existing per-step checks.** Under
`close_this_arm`, `opt.step()` now happens *inside* the helper, i.e. **before**
the `if tf_this_arm:` ncr-zero-grad proof and the `if freeze_entity_adapter:`
`assert_entity_adapter_grad_none` (`pinned_runner.py:1336-1353`). Both inspect
`.grad`, which survives `opt.step()` (zeroing happens at the next arm
iteration), and both test properties (`grad is None`, `grad == 0`) that
`clip_grad_norm_`'s scalar rescale preserves. **No live exposure this wave** —
neither `--teacher-force-operator` nor `--freeze-entity-adapter` appears in any
of the 12 specs (verified by scripted scan). Recorded as m4 for the deferred
`--freeze-entity-adapter --close-target` follow-on, which the CLI help already
flags as untested in combination.

**`RUNNER_TAG` guard (V5, negative run).** Mutating the constant in a copy of
the shipped file and invoking `main()` raises the pinned `AssertionError` and
exits 1; the unmutated file passes it and proceeds. The guard fires. (m3: R1.11
also asked for a `# PINNED.` comment at the definition itself, line 281 — only
the assert shipped.)

**`clip_coef` derivation.** `clip_coef = ‖after‖/‖before‖` is the *applied*
ratio rather than PyTorch's internal formula re-derived by hand — correct, and
robust in both limits: when the clip does not fire the two norms are the
identical value so the quotient is exactly 1.0, and when it does,
`clip_grad_norm_`'s `_foreach_mul_` is a uniform elementwise scale so the
quotient recovers it. The residual `(c − ĉ)·b` term in the post-cut identity is
~5e-9 at the observed magnitudes, four orders inside the `atol=1e-6` bound —
consistent with the assert surviving 4 000 audit steps and 800 build steps
without a spurious fire.

**Scope preservation.** Independently reproduced: V7's re-run and the build's
own `2b_scope_preserved` both report **206 / 206 non-target parameters
`torch.equal`, `worst_rel_diff = 0.0`, for both targets**. The mechanism is
sound and is a genuine property of R1.6's construction rather than a
measurement artifact: exactly **one** `total_loss.backward()` populates every
parameter's `.grad`, and only `target_w.grad` is subsequently modified — there
is no per-parameter re-assembly anywhere, so there is no non-associativity to
introduce. The build's own audit note is correct: D-F3's `EXACT_TIER`/`TOL_TIER`
split is unnecessary for this construction; only the cut target needs the
tolerance tier. **Confirmed, ratified.**

---

## 3. Findings

### FATAL — none.

### MAJOR

**MAJOR-1 — `--min-conduit-ratio 5e-05` (compE) is a live abort risk that buys
nothing.** Full adjudication in §1(b). 2.30× margin at 10× the build's sample,
with the tail concentrated in every cell's first ~1 000 steps; ~30-60 % chance
of losing ≥1 of 8 compE cells against an attrition rule that tolerates exactly
one. The floor cannot detect a broken cut at any value (the production
`allclose` assert does that, and V4 proves it fires). **Action: G1 in §6** —
lower to a degenerate-zero tripwire in all 12 specs; carry the D-A2 has-teeth
verdict at harvest over the already-recorded `close_target_diag.history`.
*Coordinator-implementable value transcription; no code change.*

**MAJOR-2 — the throughput anomaly is a vacuous zero-step re-run, not a JIT
cache effect; the repo's retained log is the vacuous run.** Full adjudication
in §1(c). Budget conclusion **unchanged and correct** (11.64 GPU-h conservative;
true ≈10.0-10.5; independently reproduced at 0.1778 s/step and cross-checked
against compB's real 0.812-0.838 GPU-h archive). **Action: none required before
launch.** Annotate `logs/verify_throughput.log` at codify time so the archive
does not carry a measurement that contradicts the report it supports.

**MAJOR-3 — `run_repl_wave3.sh`'s new self-check has zero discriminating power
(this is the 6×-recurring scoring bug class).** The `SEEDS` map declares
`1 24` for `compA compB compD primary`, but the real archive is
`compA` s0-s8 (9), `compB` s0-s20 (21), `compD` s1-s8 (8), `primary` s0-s16
(17) — measured on the box. Consequence: **`fail=1` is guaranteed on every
invocation and `SELF-CHECK PASS` can never print.** The build's own real run
(`logs/run_repl_wave3_real.log`) shows **56** `SELF-CHECK FAIL` lines, of which
**44 are permanently spurious** and 12 are the legitimately-not-yet-run compE
cells. After the wave, a genuinely missing compE cell would be line 45 of 45 —
the loud-fail signal D-F4 asked for is destroyed. The build ran D-F4's
`compZZZ` negative test and saw it fire, but never checked that the same check
also fires unconditionally for pre-existing arms, i.e. that it *discriminates*.
(The over-broad ranges and the `seq 1 24` that silently drops seed 0 are
inherited verbatim from `run_repl_wave2.sh`; the **self-check that they
neutralise is new in wave3**.) The scoring **loop** itself is correct — verified
end to end: `PREFIX=mob_gembed_compE`, seeds 1-8 / 9-12, `FZ=""` for both,
`/ephemeral/embed_path_ckpts` first in the search roots, path grammar
`$NEW/${NAME}_ckpts/${NAME}.ckpt.pt` matching the runner's own
`os.path.join(ckpt_dir, f"{cell_id}.ckpt.pt")` (`pinned_runner.py:1873`) and
the specs' `--ckpt-dir`, and `pbe_repl`'s output name
`writecond_premise_REPL_${tag}_s${s}.json` matching `OUT`. **Action: G2 in §6.**
*Coordinator-implementable, one loop header.*

**MAJOR-4 — `run_repl_wave3.sh` defaults to `CUDA_VISIBLE_DEVICES=0`, which is
the PI's vLLM GPU.** Line 13: `export CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-0}`.
GPU 0 currently reports 73 733 / 81 559 MiB used by a live compute-app PID —
≈7.8 GiB free. The scoring pass restores **both** arms plus optimizers and
evaluates at n=256 across four hop values; the training peak on this model is
6.11 GiB, so an eval-only scoring pass on GPU 0 is at best marginal and at worst
degrades the PI's server. Inherited from `run_repl_wave2.sh` (written before the
vLLM servers existed), but live now. BUILD_REPORT §3 asserts "all commands …
executed on GPU 2 (never GPUs 0/1)", yet §1.4/§4.1 record that
`run_repl_wave3.sh` was run for real and the report does not state that
`SMOKE_GPU` was set — so it is unresolved whether the build's own re-scoring of
6 `primary_s*` cells landed on GPU 0. **Action: G3 in §6.**
*Coordinator-implementable, one default.*

**MAJOR-5 (raised and DISCHARGED by this audit) — the build's flag-OFF parity
test was materially weaker than D-M6 specifies.** R1.11 requires "the full loss
trajectory (`ce_loss.item()` per step) **and both arms' final `state_dict()`s**
are `torch.equal`" over ~200 steps. `verify_run_two_arm_cell.py` part D runs
**60** steps, compares **4** `LOG_EVERY` points, and never compares
`state_dict()`s; its second check, `D_ce_loss_values_bitwise_equal`, re-reads
the same four numbers from the same array, so it is not the independent
per-step list the report describes. This is the gate protecting the *entire*
archived compB/compA/compD/primary corpus. **Discharged by execution: V8 ran
the specified version** (200 steps, seed 9999, separate processes) — identical
trajectory and **414/414 tensors `torch.equal` across both arms**. No further
action.

**MAJOR-6 (raised and DISCHARGED by this audit) — the build's "forced-fail"
negative test did not exercise the shipped function.**
`verify_embed_path_real.py` sub-test 4 defines a local `broken_assemble` that
*re-implements* the assembly with the subtraction omitted, then asserts that
*that copy's* assert fires. Under this repo's own standing rule (CPU-stub
suites test logic only; real-kernel coverage needs a smoke of the *production*
path), that does not establish that the shipped `assemble_closed_grads_`'s
assert has teeth, and it would not catch drift between the copy and the
original. **Discharged by execution: V4/N2 mutated the shipped `sub_` line out
of the shipped file and ran both targets on real CUDA — the production
`AssertionError` fired for `embed` and for `entity_adapter`.** No further
action.

### minor

- **m1** — BUILD_REPORT §2 claims `verify_cpu_synthetic.py` contains "the
  **literal same** `_non_ce_term`/`assemble_closed_grads_`/
  `assert_conduit_has_teeth` code as the patch." Mechanically diffed (AST,
  docstrings/comments stripped): it is a **reimplementation** — different
  signature (`all_params`/`target_w` passed in), no `target` return key,
  different assert messages, inlined `stepped`. **Logic is identical** (verified
  line by line, no semantic difference), so V3's conclusions transfer; but the
  claim overstates and the copy gives no drift protection.
- **m2** — Three different pin formulas exist across the artifacts without the
  divergence being named: `verify_embed_path_real.py`'s docstring says "0.5×
  the measured mean over 50 steps"; its body computes and prints
  `pinned_min_conduit_ratio` = **p5/2** (9.263e-4 embed, 1.257 adapter) over
  200 steps; the job specs ship **min/10** (5e-5, 0.049). BUILD_REPORT discloses
  the third but not that it differs from the script's own printed suggestion by
  18× (embed) and 26× (adapter).
- **m3** — `1_has_teeth_baseline[embed]` is retained in the archive with
  `PASS: False`. The criterion (`abs(‖combined‖ − ‖CE-alone‖) > 1.0`) is the
  wrong instrument for a ratio question and is nearly blind when the two
  gradients are close to orthogonal (‖combined‖ ≈ √(‖ce‖²+‖r‖²); at that
  batch's ratio 0.0346 and ‖ce‖ 66.32 the predicted difference is 0.040 vs the
  0.048 observed — the "failure" is arithmetic, not a defect). The build's prose
  reinterpretation is correct; the red `PASS` field is unannotated in the log.
- **m4** — R1.11 asked for a `# PINNED.` comment at the `RUNNER_TAG` definition
  (`pinned_runner.py:281`) in addition to the assert; only the assert shipped.
- **m5** — `--close-target` composed with `--freeze-entity-adapter` is
  unguarded: `close_target="entity_adapter"` + freeze would call
  `autograd.grad` against a `requires_grad=False` tensor. Not reachable this
  wave (no spec sets either flag), and the CLI help discloses it; a one-line
  `assert not (freeze_entity_adapter and close_target == "entity_adapter")`
  would close it before the deferred §6.4 follow-on.
- **m6** — `min_conduit_ratio` is written to `rec["config"]` but **not** to the
  checkpoint and is not resume-asserted, unlike `close_target`. A resume with a
  different floor is silent. (This is what makes MAJOR-1's crash recoverable, so
  it is a double-edged minor; note it rather than close it.)
- **m7** — `ckpt.get("close_target", close_target)` makes the resume assert
  vacuous for a checkpoint written by the *unpatched* runner. Faithful to the
  seed/freeze pattern it mirrors and unreachable this wave (fresh per-cell ckpt
  dirs under `/ephemeral/embed_path_ckpts/`), but it means the trap only
  protects checkpoints the patched runner itself wrote.
- **m8** — **A cell whose `out` JSON is lost while its step-20 000 checkpoint
  survives re-runs to `status=COMPLETED, step=20000` with no `loss_history` and
  no `close_target_diag`, and the current `validity_check` would still PASS.**
  Demonstrated incidentally by V10 (invocation 2's `rec` raised
  `KeyError: 'loss_history'`). Also: on any resume, `close_diag_hist` restarts
  empty (same convention as `loss_hist`, `pinned_runner.py:1297`), so a resumed
  cell's history covers only post-resume steps. **Folded into G1's harvest gate
  as a `len(history) >= 100` assertion.**

---

## 4. The 12 job specs — clean

Scripted cross-check of every spec against the runner's real CLI, the queue's
real schema, and `run_repl_wave3.sh`'s real path grammar:

| property | result |
|---|---|
| ids | `130800`–`130811`, contiguous, filename prefix = `"id"` field, above the highest existing id anywhere on the box (`130708` across `completed/ failed/ cancelled/ parked_k24plus/ pending/`) |
| arms / seeds | 8 × `--close-target embed` at seeds **1-8**; 4 × `--close-target entity_adapter` at seeds **9-12** — matches R1.9/R1.10's paired design exactly |
| cell ids | `mob_gembed_compE_s{1..8}` / `mob_gembed_compE_adapter_s{9..12}` — matches `run_repl_wave3.sh`'s `PREFIX="mob_gembed_${tag}"` |
| ckpt paths | `/ephemeral/embed_path_ckpts/<cell_id>_ckpts` → runner writes `<ckpt_dir>/<cell_id>.ckpt.pt`, which is exactly the scorer's first search candidate `$NEW/${NAME}_ckpts/${NAME}.ckpt.pt` |
| base flags | byte-identical to compB's archived launch across all 12 (`--steps 20000 --batch-size 32 --eval-batch-size 64 --warmup-steps 200 --lr 3e-4 --aux-read-loss-weight 0.5 --ortho-reg-weight 0.1 --aux-loss-type contrastive+cosine --contrastive-temperature 0.07 --ckpt-every 10000 --eval-every 1000 --ceiling-gpuh 6.0`) |
| stray flags | none — no `--freeze-entity-adapter`, no `--teacher-force-operator` in any spec |
| shell substitution | none — no `$(` and no backticks in any `cmd` |
| `validity_check` | asserts `status=='COMPLETED'`, `step>=20000`, **and** `config.close_target` matches the arm — D-M7's bookkeeping made load-bearing. (One gap: m8. Closed by G1.) |
| schema | exactly the queue's 8 fields (`id lane hypothesis cmd gpu_h_estimate output_dir validity_check notes`), matching the box's own completed specs |
| `gpu_h_estimate` | 0.9843 × 8 + 0.9423 × 4 = **11.6436** — conservative vs the real ≈0.83/cell archive (see MAJOR-2) |
| runnability | `cd /home/nvidia/ncr_embedpath` exists with the patched runner + the `ncr_lm_wave1_smoke.py` symlink; `/home/nvidia/tdenv/bin/python3` is the interpreter every other lane on this box uses; `mkdir -p …/results` is in the `cmd`; `/ephemeral/embed_path_ckpts` does not exist yet and is created by the runner's own `os.makedirs(ckpt_dir, exist_ok=True)` |
| `notes` | all 12 carry `CANDIDATE -- NOT queue-eligible until audited` (must be updated at launch, see G4) |

---

## 5. Placement / red-team

- **GPUs 0/1 are occupied and the queue handles it without any change to the
  specs.** `nvidia-smi --query-compute-apps` shows one PID at 73 724 MiB on each
  of GPU 0 (`GPU-2413337d…`) and GPU 1 (`GPU-c1bf0056…`). `queue_worker.sh:108-112`
  gates each worker on **its own** GPU: `napps=$(… --query-compute-apps=pid -i
  $GPU | grep -c '[0-9]')`, `mem=$(… memory.used -i $GPU)`, `if [ "$napps" -gt 0
  ] || [ "$mem" -ge 2048 ]` → sleep 60 and re-poll. Workers g0/g1 will back off
  indefinitely; the wave lands on g2-g7. No spec assumes 8 GPUs and none pins a
  GPU (`CUDA_VISIBLE_DEVICES` is injected by the worker at line 157). All 8
  worker tmux sessions are alive.
- **Wall-clock:** 12 cells / 6 free GPUs = 2 sequential slots × ≈0.85-1.0 h ≈
  **2.0-2.5 h**, one cell per GPU (G3-B31's no-packing ruling carried forward;
  measured peak VRAM 6.11 GiB of 80 GiB leaves packing headroom that this audit
  does **not** authorise — a packing decision needs its own contention-pricing
  pass and would perturb the timing baseline against archived compB, which is
  the paired comparator).
- **Disk:** `/ephemeral` 5.6 T available; 12 × 2.2 GiB checkpoints ≈ **26 GiB**
  (measured against `mob_g3b31_compB_s1_ckpts`) — 0.5 % of free space. Result
  JSONs go to `/` (63 G free after audit cleanup) at ~1-2 MB each. No headroom
  concern.
- **Grant runway:** an ≈10-11.6 GPU-h wave with a ≈2.5 h critical path against
  ~5 remaining days is comfortably affordable and leaves the box free for the
  spearhead queue immediately after. There is no competing pending job
  (`~/queue/pending/` is empty), so this wave will not displace anything.
- **Side effects from the build (BUILD_REPORT §4) — assessed acceptable, no
  remediation needed.** The 6 re-scored `primary_s*` cells were re-evaluated
  from the same checkpoints with the same deterministic `pbe_repl` instrument at
  the same `seed=90210`, so the overwritten JSONs are numerically identical; the
  re-fire was caused by copy-refreshed mtimes from the Aug-18 disk remediation,
  and a second run confirmed `SCORED=0 RESCORED=0`. The only open question is
  which GPU it ran on — see MAJOR-4.

---

## 6. VERDICT

# PASS — LAUNCH-RELEASED, conditional on three mechanical pre-launch transcriptions (G1-G3)

The science construction is sound and the code is correct. Every load-bearing
invariant was re-verified by execution on the production path, not accepted on
the build's word: the patch is provably the deployed diff; the split gradient is
algebraically exact against `compute_arm_losses`; the cut's own detector fires
when the shipped `sub_` is removed, for both targets; the `RUNNER_TAG` guard
fires; the resume trap fires; the seed trap still fires; flag-OFF is bit-identical
to the pinned runner across 414/414 tensors of both arms; 206/206 non-target
parameters are `torch.equal` under the cut; and the unpatched scorer reads
patched checkpoints. **Nothing found requires an Opus rebuild, and no change to
`ncr_lm_wave1_runner.py` is required.** G1-G3 are literal-value edits to spec
JSONs and one shell script — coordinator-implementable transcriptions that need
no re-audit of the runner.

### G1 (mandatory) — retune the has-teeth floor and move its verdict to harvest

In all 12 job specs, replace the `--min-conduit-ratio` value with **`1e-8`**
(both arms). Rationale in §1(b): the floor cannot detect a broken cut at any
value, it has no vacuity teeth at 5e-5 either, and its only live effect is a
30-60 % chance of aborting a compE cell. `1e-8` sits four orders below the
lowest ratio ever observed (1.151e-4 over 2 000 steps) and four orders above the
`clamp_min(1e-12)` degenerate floor, and preserves NaN detection
(`NaN > 1e-8` is `False`).

In the same edit, strengthen each spec's `validity_check` by appending
(closing m8):

```python
h = d['close_target_diag']['history']; assert len(h) >= 100, len(h); assert all(r[3] for r in h)
```

**Pre-registered harvest has-teeth gate** (this replaces D-A2's per-step check;
thresholds pinned **now**, from V11's 2 000-step probe, before any wave data
exists — embed measured median 0.1327 / mean 0.9939, adapter median 12.896 /
mean 16.575):

- every cell: `len(history) >= 100` and `all(cut_active)` and no logged
  `conduit_ratio <= 1e-8`;
- `compE`: per-cell **median `conduit_ratio` > 0.01** and **mean > 0.1**
  (≈13× and ≈10× below measured);
- `compE_adapter`: per-cell **median > 1.0** and **mean > 5.0** (≈13× and ≈3×
  below measured);
- any cell failing these is reported as an **instrument-validity failure**
  (like R1.2's h=1 co-condition), not folded into the band ladder.

### G2 (mandatory) — restore the scoring self-check's discriminating power

In `run_repl_wave3.sh` (repo copy **and** `~/ncr_writecond/run_repl_wave3.sh`,
which must stay md5-identical), scope the final self-check loop to the arms this
wave is responsible for, leaving the scoring loop untouched:

```bash
for tag in compE compE_adapter; do          # was: "${!SEEDS[@]}"
```

This restores D-F4's intent exactly (loudly name any missing compE/compE_adapter
arm/seed) and lets `SELF-CHECK PASS` actually print. Re-run the `compZZZ`
negative afterwards to confirm the check still fires — it must be **inside** the
narrowed tag list to be exercised. (Optionally, separately: correct the `SEEDS`
map to the real archived ranges `compA 0 8`, `compB 0 20`, `compD 1 8`,
`primary 0 16` — this also stops silently dropping seed 0 from re-scoring. Not
required for this wave's read, which uses only archived compB s1-s12.)

### G3 (mandatory) — never default scoring onto the PI's GPU

`export CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-2}` (or drop the default entirely and
require `SMOKE_GPU`). Always invoke as `SMOKE_GPU=<2..7> bash
~/ncr_writecond/run_repl_wave3.sh`.

### G4 — launch sequence (exact)

1. Apply **G1** to all 12 specs; apply **G2** and **G3** to
   `matrix-thinking/embedpath_build/run_repl_wave3.sh` and re-deploy it to
   `~/ncr_writecond/run_repl_wave3.sh` (verify md5 match after copy).
2. Flip each spec's `notes` prefix from `CANDIDATE -- NOT queue-eligible until
   audited` to a launch-released marker citing this file.
3. Pre-flight (all four must hold):
   `md5sum ~/ncr_g3b31_contrastive/ncr_lm_wave1_runner.py` = `9a93198b642242f512ff8489e32b0a53`;
   `md5sum ~/ncr_embedpath/ncr_lm_wave1_runner.py` = `d0ba4712f026dfdb00ba14d277a4d486`;
   no `~/queue/PAUSE` and no `~/queue/STOP`; `tmux ls` shows
   `queue_worker_g0..g7`.
4. Copy all 12 specs into `~/queue/pending/` in **one** operation, **under their
   existing filenames** — do not renumber or rename (QUEUE_README's documented
   "Incarnation 2" landmine: a rename that does not change the JSON `id`).
   Claim order will be `130800 … 130811`, i.e. compE s1-s6 in the first slot on
   g2-g7 and compE s7-s8 + compE_adapter s9-s12 in the second. Both arms
   complete inside ≈2.5 h.
5. Watch: `ls ~/queue/{claimed,completed,failed} | wc -l` and, on any
   `failed/`, read `~/queue/logs/<id>.log` first for a `HAS-TEETH FAILED` line
   (should now be impossible under G1) or a `post-cut grad does not match`
   line (a real cut failure — stop the wave and escalate).

### G5 — what the harvest gates on

1. **Completion:** ≥7/8 compE and ≥3/4 compE_adapter in `completed/` (R1.2's
   attrition rule); below that the corresponding read is **void**, not reported.
2. **Instrument validity:** G1's has-teeth aggregate per cell, **and** R1.2's
   co-condition `median(P1b h=1) ≥ 0.95` per cell.
3. **Scoring:** `SMOKE_GPU=<2..7> bash ~/ncr_writecond/run_repl_wave3.sh` →
   12 new `writecond_premise_REPL_compE*_s*.json` **and** a green
   `SELF-CHECK PASS` (only meaningful after G2).
4. **Read rule:** R1.9's strict first-match ladder for compE (WIN / PARTIAL at
   the pinned 0.85195 / NULL), paired Wilcoxon against archived compB s1-s8
   (median 0.75195), with the unpaired n=8-vs-n=20 Mann-Whitney cross-check
   reported alongside; then R1.9's ADAPTER-LEVERAGE-CONFIRMED / EMBED-SPECIFIC /
   AMBIGUOUS rule for compE_adapter. **All numbers labelled P1b explicitly**
   (R1.2, the twice-caught defect).
5. **Mandatory disclosure at harvest (new, from V11 — a science finding, not a
   build defect).** The two arms' interventions are **not of comparable
   magnitude**: `embed`'s conduit_ratio has median **0.133** with **76.9 %** of
   steps below 1.0, while `entity_adapter`'s has median **12.9** with only
   **0.45 %** below 1.0 — roughly a **100× difference in the size of the
   gradient share being removed**, in the adapter's favour. R1.9's read rule
   compares the two arms' *outcome* medians directly (ADAPTER-LEVERAGE CONFIRMED
   if `compE_adapter` median ≥ `compE` median − 0.05), and that comparison is now
   known **in advance** to be confounded by intervention strength. An
   EMBED-SPECIFIC or ADAPTER-LEVERAGE label reported without this disclosure
   would be misread. It should be stated in the same paragraph as the label,
   every time.

---

## 7. Deferred / explicitly not covered by this audit

- No SM-utilisation or contention-pricing pass was run; G3-B31's one-cell-per-GPU
  ruling is carried forward. The measured 6.11 GiB/cell leaves real packing
  headroom, but packing would perturb the per-step timing baseline against the
  archived compB comparator, so it is **not** authorised here.
- M1's fourth 2×2 cell (frozen adapter + closed embed) and the §6.4 `TPC_fg`
  follow-on remain out of scope, unchanged by this audit (m5 flags the guard
  they will need).
- The extrapolated abort probability in MAJOR-1 is an order-of-magnitude
  estimate from a 2 000-step, single-seed lower tail, not a measurement over
  20 000 steps × 8 seeds. G1 makes the estimate moot rather than resolving it.
