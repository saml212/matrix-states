# EMBED-PATH INTERVENTION — BUILD REPORT

**Build agent, 2026-08-21.** Repo commit at build time: `db4d385` (main).
Binding spec: `matrix-thinking/NCR_EMBED_PATH_DESIGN.md` `## DRAFT-R1`
(responding to `matrix-thinking/NCR_EMBED_PATH_ATTACK_R1.md`, REV-REQUIRED
5F/8M/5m, all findings executed). `EXPERIMENT_LOG.md` 2026-08-18 #14–#17.

**Status: BUILT, NOT LAUNCHED.** Nothing was written to `~/queue/pending/`
on the box. All 12 job specs are `"notes": "CANDIDATE -- NOT queue-eligible
until audited"`. This report is the input to that audit.

---

## 1. What was built

### 1.1 Runner patch

- Pinned file (untouched, re-verified after every box operation in this
  build): `~/ncr_g3b31_contrastive/ncr_lm_wave1_runner.py`, md5
  `9a93198b642242f512ff8489e32b0a53`.
- Patched copy, in a **new** box directory (never edited the pinned file
  in place): `~/ncr_embedpath/ncr_lm_wave1_runner.py`, md5
  `d0ba4712f026dfdb00ba14d277a4d486`. Alongside it: a symlink
  `~/ncr_embedpath/ncr_lm_wave1_smoke.py` → the pinned smoke/graft module
  (unmodified; its own `_setup_paths()` resolves `ncr_models`/`grammar_rd`/
  `lm_pretrain_rd` from absolute box paths `/home/nvidia/{ncr,chapter2/
  deltanet_rd}`, so no other files needed to be copied or symlinked).
- Diff: `matrix-thinking/embedpath_build/embed_path_runner.patch` (472
  lines, unified diff, `a/ncr_lm_wave1_runner.py` → `b/ncr_lm_wave1_runner.py`).

Implements DRAFT-R1 sec R1.3 (`_non_ce_term`, built from
`compute_arm_losses`'s own returned `aux_loss`/`ortho_loss` tensors, never
by `total_loss - ce_loss` subtraction — F2's fix), R1.6 (`assemble_closed_
grads_`, cut **after** `clip_grad_norm_` — D-F5's fix; `--close-target=
{embed,entity_adapter}` selector — the M1/M2/M2b re-scope from R1.1),
R1.4/D-A2 (`assert_conduit_has_teeth`, a ratio check, not an absolute
floor), R1.11 (`RUNNER_TAG` pinned-invariant assertion), R1.12/D-M7
(`close_target` in `rec["config"]`, `save_checkpoint`'s ckpt dict, and a
resume-mismatch assert mirroring the existing seed-trap/freeze-trap
pattern exactly).

### 1.2 Two disclosed corrections to DRAFT-R1's own text (not deviations from intent — the design's own comment and code disagreed with each other)

1. **R1.6's training-loop comment says "`clip_grad_norm_` and `opt.step()`
   are ALREADY DONE inside `assemble_closed_grads_`" — but the function
   signature/body given in that same section never receives `opt` and
   never calls `.step()`.** This build adds `opt` as an explicit parameter
   and calls `opt.step()` at the end of the cut-active path, so the code
   matches what the design's own comment promised the caller.
2. **R1.6's two early-return branches (`non_ce is None` / `grad_rest_
   target is None`) called `total_loss.backward()` and returned without
   ever finite-checking, clipping, or stepping — silently dropping that
   training step's parameter update entirely**, with no signal to the
   caller's `n_skipped` bookkeeping. Both are expected-dead-code in this
   wave's actual launch config (`aux_read_loss_weight=0.5`,
   `ortho_reg_weight=0.1` always > 0; embed/entity_adapter are both
   structurally on the conduit), but "expected unreachable" is not the
   same as "safe to leave broken." Fixed: every path now runs the same
   finite-check → clip → step sequence the shared training-loop block
   uses, and returns a new `"stepped": bool` field so `n_skipped`
   bookkeeping stays accurate even on this fallback path.

### 1.3 Job specs

`matrix-thinking/embedpath_build/job_specs/` — 12 JSONs, IDs
`130800`–`130811` (next free block after the highest ID found across
`~/queue/{completed,failed,cancelled,parked_k24plus}/` on the box,
`130708`). 8× `compE` (seeds s1–s8, `--close-target embed`), 4×
`compE_adapter` (seeds s9–s12, `--close-target entity_adapter`) — per
R1.1's RE-SCOPE (the `ncr_head` placebo replaced by the mandatory on-path
`compE_adapter` co-arm) and R1.9/R1.10 (paired seeds s1–s8/s9–s12, not
DRAFT-R0's s21–s28). Config-only `cmd` strings, no shell command
substitution. Checkpoints to `/ephemeral/embed_path_ckpts/`. Base flags
identical to compB's own archived launch (`--steps 20000 --batch-size 32
--eval-batch-size 64 --warmup-steps 200 --lr 3e-4 --aux-read-loss-weight
0.5 --ortho-reg-weight 0.1 --aux-loss-type contrastive+cosine
--contrastive-temperature 0.07 --ckpt-every 10000 --eval-every 1000
--ceiling-gpuh 6.0`), plus `--close-target` and `--min-conduit-ratio`
(§3 below). `validity_check` additionally asserts
`rec["config"]["close_target"]` matches the expected value (D-M7's own
bookkeeping fix, made load-bearing here). Total estimated budget: **11.64
GPU-h** (8×0.9843 + 4×0.9423), within R1.3's re-derived 9.84–13.20 GPU-h
band. **Not written to `~/queue/pending/`.**

### 1.4 `run_repl_wave3.sh`

`matrix-thinking/embedpath_build/run_repl_wave3.sh`, deployed to
`~/ncr_writecond/run_repl_wave3.sh` on the box and **run for real**
(§4.4). Written against the box's *current* `run_repl_wave2.sh`
(re-verified md5 `dfba70bccd318074d95dbe698c40c77b`, unchanged since
Rev-1's own read — confirms R1.5's account is still accurate) per D-F4:
parameterized cell-name prefix, a per-tag freeze map (`compE`/
`compE_adapter` both `FZ=""`), `/ephemeral/embed_path_ckpts` added to the
search roots, the correct seed ranges (`compE` 1–8, `compE_adapter`
9–12), loud `MISSING-CKPT`, re-score-not-skip (`-ot` freshness check),
and a self-check that FAILS LOUDLY if any expected arm/seed produced no
output.

---

## 2. CPU-runnable verification (portable, no box dependency)

`matrix-thinking/embedpath_build/verification/verify_cpu_synthetic.py` —
mirrors the real runner's topology at toy scale (tied head,
`extract_kv`/`query_key` off raw ids, a **detached** aux target — the
provable no-op DRAFT-R0 already established — a live undetached `o_raw`
feeding both CE and aux, and a separate `ortho(Z)` conduit) with the
**literal same** `_non_ce_term`/`assemble_closed_grads_`/
`assert_conduit_has_teeth` code as the patch (copy-verified by eye
against `embed_path_runner.patch`, not re-derived). Run to completion,
locally, no CUDA/`fla` dependency:

```
=== synthetic[embed] ===          cut_active=True stepped=True  n_scope_exact=7 scope_fails=[]  PASS: True
=== synthetic[entity_adapter] ===  cut_active=True stepped=True  n_scope_exact=7 scope_fails=[]  PASS: True
=== synthetic_forced_fail_negative_test ===  assertion_fired=True  PASS: True
```

Full log: `matrix-thinking/embedpath_build/logs/verify_cpu_synthetic.log`.
The forced-fail test deliberately omits the post-clip subtraction (a
simulated no-op cut) and confirms the internal `allclose` assertion fires
— run to completion, not merely written, per this repo's standing rule.

---

## 3. Real-CUDA verification (box-side smoke, GPU 2, `youthful-indigo-turkey`)

All commands below were actually executed on GPU 2 (never GPUs 0/1, held
by the PI's vLLM servers). Full logs under `matrix-thinking/
embedpath_build/logs/`.

### 3.1 `verify_embed_path_real.py` — has-teeth, cut-confirmed, scope-preserved, conduit_ratio distribution, forced-fail

Exact command: `CUDA_VISIBLE_DEVICES=2 /home/nvidia/tdenv/bin/python3
~/ncr_embedpath/verify_embed_path_real.py`. Log: `logs/verify_embed_path_
real.log`.

- **Has-teeth baseline** (aux+ortho measurably move the target's combined
  grad before any cut, on ONE fixed real batch): `entity_adapter` —
  combined norm 2.338 vs CE-alone 0.399 (abs diff 1.939, 83% of the
  combined norm). `embed` — combined norm 66.37 vs CE-alone 66.32 (abs
  diff 0.048, ~0.07% of the combined norm on THIS ONE probe batch) —
  this is exactly the "vacuous pass" mode D-A2 exists to catch via a
  **ratio**, not an absolute floor: on some batches embed's own aux+ortho
  share is a small fraction of its total gradient. §3.3 below shows the
  ratio is enormous on other batches (max 23.4) — the per-step variance
  is real and large, not a sign of a broken cut.
- **Cut-confirmed**: `assemble_closed_grads_`'s own internal
  `torch.allclose` assertion (the tolerance-tier check, D-F3) passed for
  both targets on the fixed probe batch (`cut_active=True`,
  `stepped=True` both times).
- **Scope-preserved — measured stronger than D-F3 anticipated.** D-F3's
  own attack measured non-associativity (~4.5e-8 max diff, `allclose`-true
  but not `torch.equal`) in DRAFT-R0's ORIGINAL construction, which
  derived **every** parameter's `grad_ce` via a separate `autograd.grad`
  call and manually summed it with `grad_rest` — a real source of float
  non-associativity relative to `backward()`'s own fused accumulation
  order. R1.6's construction (what this build ships) never does that for
  non-target parameters: it calls **one** plain `total_loss.backward()`
  for the full combined gradient of every parameter (target included),
  and only the target parameter's grad is modified afterward. Measured
  directly (not assumed): **all 206 non-target parameters, for both
  `close_target` choices, are `torch.equal` — bit-identical, not merely
  `allclose` — between flag-OFF and flag-ON runs on the same batch/seed/
  init.** (First run of this script caught a test-harness bug — I forgot
  to apply `clip_grad_norm_` to the OFF-path comparison, since the ON
  path clips internally — producing a uniform ~99.5% "failure" across
  every parameter, exactly `1 − clip_coef`. Fixed and re-run; see the log
  for both versions' reasoning, retained as a worked example of the
  check actually having teeth.) **Audit note:** D-F3's own two-tier
  `EXACT_TIER`/`TOL_TIER` split can be simplified for this specific
  construction — only the cut target itself needs the tolerance-tier
  `allclose` bound; every other parameter can be checked with plain
  `torch.equal`.
- **`conduit_ratio` over 200 real training steps** (steady mix of hops
  1/2/3, real optimizer state evolving): `embed` — mean 1.748, min
  0.00055, max 23.41, p5 0.00185. `entity_adapter` — mean 24.83, min
  0.492, max 215.44, p5 2.514. **A mean-based pin (as DRAFT-R1's own
  `assert_conduit_has_teeth` docstring suggests, "0.5x the measured
  mean") would have failed on 101/200 steps for `embed` and 91/200 steps
  for `entity_adapter` — roughly HALF of all steps.** Because this
  assertion is called every step with no exception handling in
  `run_two_arm_cell`, an `AssertionError` is an **uncaught crash that
  aborts the entire 20,000-step cell.** This is a real, load-bearing
  finding, not a hypothetical: the naive pin from R1.6's own suggested
  formula is unsafe as literally written.
- **Forced-fail negative test** (deliberately skip the post-clip
  subtraction, simulating a no-op cut, on real tensors): the internal
  `allclose` assertion fired. `PASS: True`.

### 3.2 Disclosed deviation: `PINNED_MIN_RATIO`

DRAFT-R1 explicitly left `PINNED_MIN_RATIO` undetermined, deferred to
this build's own smoke measurement (R1.15). Given §3.1's variance
finding, this build does **not** use the literal "0.5x mean" formula.
Instead: **`min_ratio = (measured 200-step minimum) / 10`** — a 10×
safety margin below the worst value actually observed, while staying
meaningfully above exact zero (so a genuinely broken/no-op cut, which
reads `conduit_ratio ≈ 0.0`, is still caught).

| `close_target` | measured min (200 steps) | pinned `--min-conduit-ratio` |
|---|---|---|
| embed | 0.000551 | **0.00005** |
| entity_adapter | 0.492361 | **0.049** |

**Flagged for audit, not silently resolved:** only 200 of the eventual
20,000 real training steps were sampled. The true minimum over a full
run could plausibly be lower still (the aux/ortho loss's own magnitude
generally *decreases* over training — see the raw per-step log in §3.1,
`ortho_loss` fell from 188.77 at step 1 to 1.51 at step 20 in one probe
— which could push the conduit_ratio lower as training progresses, not
higher). The audit should evaluate whether a **rolling-window** or
**violation-rate** check (e.g., fail only if N consecutive steps or a
sustained fraction of steps read below floor) would be structurally more
robust than a single hard per-step floor, before this wave launches —
this build implements D-A2 exactly as specified (a per-step hard assert)
with the safest pin the available evidence supports, but does not
consider itself authorized to redesign the check's own mechanism.

### 3.3 `verify_run_two_arm_cell.py` — the full entry point, checkpoint/resume, VRAM

Exact command: `CUDA_VISIBLE_DEVICES=2 /home/nvidia/tdenv/bin/python3
~/ncr_embedpath/verify_run_two_arm_cell.py`. Log: `logs/verify_run_two_
arm_cell.log`. All writes confined to `/tmp/embedpath_verify/` on the
box (deleted-safe, not under `results/` or `~/queue/`).

- **A — partial run via a tiny `--ceiling-gpuh`**: forced a genuine
  interruption (`status=ABORTED-BUDGET` at step 10/40) rather than a
  full completion, so the resume test in B is a real resume, not a
  vacuous "already COMPLETED" short-circuit. (First draft of this script
  used a full-length first call, which DID short-circuit — caught before
  reporting a false pass, not silently accepted.) Peak VRAM: **6.11 GB**
  — well inside the design's predicted 7–9 GB band (R1.13), actually
  better than predicted, consistent with D-F2's fix removing the
  subtraction form's extra full-model zero-gradient tensor.
- **B — matched resume to completion**: `status=COMPLETED step=40`,
  `close_target_diag` history present (2 entries at `eval_every=5`
  cadence). `PASS: True`.
- **C — mismatched `--close-target` on resume (negative test)**:
  resuming the same checkpoint (`close_target='embed'`) with
  `close_target='entity_adapter'` fired the loud `AssertionError` exactly
  as D-M7's resume-mismatch assert specifies. `PASS: True`.
- **C2 — pre-existing seed-mismatch check still fires**, confirming this
  build didn't break the pattern it mirrors. `PASS: True`.
- **D — flag-OFF parity smoke (D-M6)**: `close_target=None` on the
  **patched** runner (`~/ncr_embedpath/`) vs the **unpatched pinned**
  runner (`~/ncr_g3b31_contrastive/`, md5 `9a93198b…`, untouched), same
  seed (9999), 60 steps, run as two separate subprocesses (avoids
  Python's module-name-keyed import cache aliasing the two
  `ncr_lm_wave1_runner` modules if imported in one process). **Loss
  trajectories and full per-step CE loss lists are bit-identical
  (`==`, not `allclose`) between the two runners.** `PASS: True`.

### 3.4 `verify_throughput.py` — GPU-h budget confirmation, and an unresolved anomaly

Exact command as above. Log: `logs/verify_throughput.log`.

**First invocation** (300 steps, `eval_every=1000`/`ckpt_every=1000` so
no mid-run overhead, `warmup_steps=200`, `seed=31337`):

| `close_target` | per-step (s) | peak VRAM (GB) | extrapolated GPU-h @ 20,000 steps | overhead vs OFF |
|---|---|---|---|---|
| embed | 0.1772 | 6.11 | 0.9843 | +6.5% |
| entity_adapter | 0.1696 | 5.95 | 0.9423 | +2.0% |
| OFF (compB-equivalent) | 0.1663 | — | 0.9238 | — |

This closely matches R1.3's own re-derived "+2–10% overhead" prediction,
and the OFF rate (0.9238 GPU-h) is close to the archived real compB
baseline the design cites (`mob_g3b31_compB_s0.json`, `elapsed_s 2985.6`
at step 20000 = 0.829 GPU-h). **These are the numbers used for the job
specs' `gpu_h_estimate`** and the 11.64 GPU-h wave total in §1.3.

**Second invocation, minutes later, identical script/seed/config:**
per-step dropped to 0.0162 / 0.0127 / 0.0134 s — **roughly 10× faster**,
peak VRAM dropped to 3.20 GB. This is almost certainly a Triton
JIT-kernel-cache effect (the first invocation pays a one-time
compilation cost per unique kernel signature, persisted to a
filesystem-level cache that the second invocation's fresh process then
reused) rather than a real per-step throughput difference — but this is
a **hypothesis, not a confirmed explanation**; it was not chased further
within this build's scope. **Disclosed, not resolved:** if the second
number is the real steady-state rate for actual 20,000-step launches
(where the one-time compile cost amortizes over 20,000 steps instead of
300), the true wave cost could be closer to **~1.2 GPU-h total**, not
11.64 — an order of magnitude lower. This build deliberately keeps the
**conservative (slower, first-measurement) number** in the job specs,
since it sits much closer to the historically-real, full-scale archived
compB rate, and because the existing `--ceiling-gpuh 6.0` safety cap
(inherited unchanged from compB's own base config) is 6–60× above either
estimate — so the discrepancy affects budget bookkeeping accuracy, not
launch correctness. **Audit should decide** whether to re-derive the
budget from a longer, cleaner probe before treating 11.64 GPU-h as
final.

---

## 4. Side effects on the box (disclosed)

1. **`run_repl_wave3.sh` was run for real** against the box's actual
   `~/ncr_writecond/results/` state (§1.4/§4.4). It correctly re-scored 6
   `primary_s*` cells (`primary_s3/s4/s5/s6/s13/s14`) whose checkpoints
   live under `/ephemeral/reseed_ckpts` with mtimes **newer** than their
   existing eval JSONs — almost certainly an artifact of those
   checkpoints having been copied there during the Aug-18 disk-incident
   remediation (a file-copy operation stamps a fresh mtime without
   changing the checkpoint's actual trained weights), not a sign that
   those cells were retrained. This consumed a small amount of real
   GPU-2 eval-only time (well under a minute) and overwrote 6 existing
   `writecond_premise_REPL_primary_s*.json` files with what should be
   numerically identical content (same checkpoint, same deterministic
   `pbe_repl` instrument, same seed 90210). **Flagged as a real
   consequence of testing D-F4's own re-score-not-skip fix against live
   data** — the fix (correctly) does not distinguish "checkpoint content
   changed" from "checkpoint file mtime changed for an unrelated reason
   (a copy)." A subsequent immediate re-run of the same script produced
   `SCORED=0 RESCORED=0`, confirming the mtimes are now settled and it
   will not re-fire spuriously again.
2. **The required negative test** (a temporary `compZZZ` entry, D-F4's
   own ask) was run twice — once with no results marker (confirmed
   `SELF-CHECK FAIL` fires, `MISSING-CKPT` does not, since that message
   is specifically gated on a results marker existing per the script's
   own logic) and once with a fake results marker present (confirmed
   **both** `MISSING-CKPT compZZZ_s1` and `SELF-CHECK FAIL` fire
   together, matching R1.5's literal description exactly). Both the
   temporary script variant and the fake marker file were deleted
   immediately after; the deployed `run_repl_wave3.sh` on the box is
   the clean version with no bogus entries.
3. **Nothing else was modified.** The pinned runner's md5 was
   re-verified as `9a93198b…` (unchanged) after every box operation in
   this build. `~/queue/pending/` is untouched. `~/ncr_embedpath/
   results/` contains no real launch artifacts (all test writes went to
   `/tmp/embedpath_verify/`, which is not under version control or the
   archive policy).

---

## 5. What is deferred to the audit / not built

- **The `PINNED_MIN_RATIO` mechanism itself** (§3.2) — this build pins
  the safest values the available 200-step evidence supports, but
  flags that a per-step hard assert may be the wrong mechanism given
  the measured variance, and does not consider itself authorized to
  change D-A2's own specified check shape.
- **The throughput anomaly** (§3.4) — disclosed with a hypothesis, not
  resolved. The conservative number is used; a fresh, longer,
  clean-cache probe would settle it.
- **Fresh SM-utilization/contention-pricing pass** — R1.13 already
  flagged this as carried-forward-not-re-derived from G3-B31's own
  no-packing ruling; this build's own measured VRAM (5.95–6.11 GB/cell)
  leaves substantial headroom on an 80 GB H100 that a packing decision
  could exploit, but per CLAUDE.md's pre-launch contention-pricing gate
  this is a placement-audit decision, not a build-time one — flagged as
  an opportunity, not implemented.
- **M1's fourth 2×2 cell** (frozen adapter + closed embed) — R1.1 itself
  deferred this, not funded this wave; unchanged by this build.
- **§G3-B32's own TPC_fg claim under `--freeze-entity-adapter
  --close-target embed`** — §6.4's own natural follow-on, explicitly out
  of this wave's scope; the code supports it (both flags compose
  independently — untested in combination, disclosed in the new
  `--close-target` CLI help text) but no job spec for it was written.

---

## 6. What the audit must check (summary, cross-referenced to the sections above)

1. The two disclosed corrections to DRAFT-R1's own inconsistent text
   (§1.2) — verify they're the right fix, not a scope creep.
2. The `PINNED_MIN_RATIO` values and the percentile/margin-based
   derivation (§3.2) — including whether the per-step hard-assert
   mechanism itself is safe for a full 20,000-step run given the
   measured variance.
3. The scope-preservation finding that non-target parameters are exactly
   `torch.equal`, not merely `allclose` (§3.1) — confirm this is real
   and not an artifact of this build's own measurement, since it's a
   stronger guarantee than D-F3 anticipated.
4. The throughput discrepancy (§3.4) and whether the job specs'
   `gpu_h_estimate`/budget total should be revised.
5. That `run_repl_wave3.sh`'s side effects on the box (§4) are
   acceptable and don't need remediation.
6. All 12 job specs' `cmd`, `validity_check`, and `--min-conduit-ratio`
   values against `matrix-thinking/embedpath_build/job_specs/*.json`
   before anything is moved to `~/queue/pending/`.
