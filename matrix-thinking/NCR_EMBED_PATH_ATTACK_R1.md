# EMBED-PATH INTERVENTION — ROUND-1 ATTACK

**Target:** `matrix-thinking/NCR_EMBED_PATH_DESIGN.md` (DRAFT-R0), repo commit `f04eb29`.
**Date:** 2026-08-18. **Round:** 1 (adversarial, pre-build).
**Verdict: REV-REQUIRED — 5 FATAL / 8 MAJOR / 5 minor.**

**Is the split-backward correct as specified? Partly.** The gradient *values* it
assembles are correct (verified by execution). Three things about it are not: the
central cost claim is false (the second pass traverses the entire backbone), the
"bit-for-bit identical to `backward()`" claim is false (float non-associativity),
and the intervention is not gradient-routing-only (it silently rescales every
other parameter's step through `clip_grad_norm_`). All three are repairable.

**Method.** Every autograd, arithmetic, and statistical claim below was EXECUTED,
not reasoned from memory: PyTorch 2.8.0 reproductions in
`.../scratchpad/repro_split_backward.py` and `stats_bands.py` / `stats_fix.py`.
Code claims are checked against the repo's byte-copy of the pinned runner,
`experiment-runs/2026-07-30_ncr_g3b31_contrastive_grid/ncr_lm_wave1_runner.py`,
md5 `9a93198b642242f512ff8489e32b0a53` — verified identical to the design's pin.

---

## §0 What the design gets RIGHT (verified, do not re-litigate)

Recording these so the revision does not lose them:

1. **The refutation of the coordinator's detach fix is CORRECT.** Verified at the
   byte level — line 630 `target_o = integ.entity_adapter(embed(answer_token).float()).detach()`
   and line 690 `T = integ.entity_adapter(embed(entity_ids).float()).detach()`.
   Both aux targets are fully detached. The proposed fix was a provable no-op.
2. **The second (ortho/Z-side) conduit is REAL and correctly identified.**
   `ortho_regularization_loss(Z)` (called line 850) → `Z = ncr_head.encode(keys_v, values_v)`
   → `integ.extract_kv(..., backbone.embed)` → `entity_adapter(embed(ids))`. It
   reaches `embed` without passing through `o`. R2's account of the leak was
   incomplete; this is a genuine new finding.
3. **`retain_graph=True` on the first `autograd.grad` is REQUIRED.** Executed
   control (T6): omitting it makes the second call raise
   `RuntimeError: Trying to backward through the graph a second time`.
4. **Manual `.grad` assembly interacts correctly with the optimizer.** Executed
   (T8): given identical `.grad`, `AdamW.step()` produces bit-identical parameters
   whether `.grad` came from `backward()` or from direct assignment; a param left
   at `.grad = None` is silently skipped and creates **no** optimizer state entry.
   The design's "no change to `build_optimizer`'s param-group structure" reasoning
   (§2.5) is sound.
5. **`allow_unused=True` cannot mask a should-be-reachable parameter here.**
   Everything reachable from `aux_loss`/`ortho_loss` is also reachable from
   `ce_loss` (because `o_injected = o_raw` feeds `logits`, line 396–397), so
   `grad_ce` returns `None` exactly where `backward()` would also have produced
   nothing. The §6.5 worry is discharged for `grad_ce`; it re-appears in a
   different form under F2's fix (see D-F2).
6. **No AMP / `GradScaler` / `autocast` anywhere in the runner** (grepped). That
   attack surface is empty; the split-backward needs no scaler bookkeeping.
7. **The subtraction node introduces no numerical error in the gradient.**
   Executed (T3): `grad(total-ce, embed)` and `grad(w_aux*aux + w_ortho*ortho, embed)`
   are `torch.equal`. Catastrophic cancellation in the *value* of `total - ce` is
   irrelevant — autograd differentiates the `Sub` node symbolically (+1/−1). An
   auditor flagging cancellation here would be wrong.
8. **Weight tying is handled correctly.** `embed.weight` is yielded once by
   `parameters()`; `grad_ce[embed]` includes all three CE routes (embedding
   lookup, tied head `F.linear(x, self.embed.weight)` at
   `lm_pretrain_rd.py:1310`, and the o-side path). That is what the design wants
   and §2.2's scope statement says so explicitly.

---

## §1 FATAL

### F1 — The scored metric's REGIME is never named, and the two candidates invert the verdict. Exact repeat of the adopted #7 F1 finding.

The design specifies the metric as `retrieval24_acc` @ h=61, "pbe_repl instrument",
`ckpt_step==20000` — in §1, §3 (bands), and §4. **It never says P1b or P0.**

`pbe_repl.py` writes BOTH regimes into every record, at sibling keys
(`experiment-runs/2026-08-13_ncr_writecond_premise_battery/pbe_repl.py.txt:26-33`):

```
p1b = R.eval_arm_at_hops(..., read_ablate=False, teacher_force=True)
p0  = R.eval_arm_at_hops(..., read_ablate=False, teacher_force=False)
rec = dict(..., P1b=dict(teacher_force=True, result=p1b),
                P0 =dict(teacher_force=False, result=p0), ...)
```

EXECUTED against the archived record for compB s6
(`experiment-runs/2026-08-18_premise_multiseed/writecond_premise_REPL_compB_s6.json`,
`ckpt_step 20000`, `n 256`) — **the same checkpoint, the same field name**:

```
P1b (teacher_force=True) : h=1 1.0000  h=13 0.9844  h=37 0.9766  h=61 0.9727
P0  (teacher_force=False): h=1 0.0430  h=13 0.0312  h=37 0.0508  h=61 0.0508   (chance 0.0417)
```

Under P1b this seed is compB's *maximum*; under P0 it is at chance. A verdict read
off the unnamed field is a coin flip. This is the identical failure the #7 audit
called F1 ("the metric was never named ... the two candidates INVERT the verdict"),
which this campaign already paid for once.

Two adopted #7 findings are also regressed in the same bands:

- **No h=1 co-condition.** #7 F3 established h=1 as the discriminator and #9 scored
  compD with `median h=1 ≥ 0.95` as a co-condition. Here it is absent. The failure
  mode it guards is live in *this* design and in the opposite direction: if closing
  the aux+ortho conduit damages the read at all depths (h=1 falls off 1.0000), the
  h=61 number drops and the wave scores **NULL** — recording "the embed-interaction
  mechanism is not supported" when the true reading is "the intervention broke the
  read." §6.3's disclosed null-ambiguity does not cover this; it is a *validity*
  failure, not a mechanism ambiguity.
- **No attrition rule.** WIN is a `min`-based criterion at n=8. #7 flagged exactly
  this ("no attrition rule for a max-based criterion at n≥8") as an adopted MAJOR.
  If 2 of 8 cells die (this campaign has lost 12 cells to one disk incident), the
  design does not say whether n=6 is scored, refilled, or voided.

**Disposition D-F1 (binding).** Pre-register, verbatim: metric =
`P1b.result["h=61"]["retrieval24_acc"]`, regime **P1b / teacher_force=True /
exact-write substitution**, instrument `pbe_repl` pinned at seed 90210, n=256, both
write modes computed; guard `ckpt_step == 20000`; **co-condition
`median(P1b h=1) ≥ 0.95`, and any cell with `P1b h=1 < 0.95` is reported
separately as an instrument-validity failure, not folded into the h=61 median**;
attrition rule: verdict is read at n≥7, voided below.

---

### F2 — `non_ce_term = total_loss − ce_loss` re-traverses the ENTIRE backbone. The design's central cost claim is false and the budget breaks the ≤15 GPU-h cap.

§2.3 claims: *"that second pass's backward graph **does not traverse the backbone's
transformer stack at all**. The expensive part of backward ... is paid once."*
That is the sole justification for the "+10–30%" overhead estimate and therefore
for the whole budget table.

It is false. Autograd accumulates gradient at the shared `ce_loss` node: +1 via
`total_loss`'s `Add`, −1 via the `Sub`. The sum is exactly `0.0` — and the engine
**still executes** `CrossEntropyBackward` and everything downstream of it, seeded
with zero, all the way through the transformer stack.

EXECUTED (`repro_split_backward.py`, a graph with the runner's exact topology —
tied head, `extract_kv`/`query_key` off raw ids, detached aux targets, `ortho(Z)` —
instrumented with a counting `autograd.Function` inside the backbone stack):

```
T0  plain total_loss.backward()                 backbone traversals = 1
T1  grad(ce, retain=True); grad(total-ce)       grad_ce pass = 1,  grad_rest pass = 1
      -> DESIGN CLAIM 'second pass does not traverse the backbone stack': *** FALSE ***
      grad_rest on backbone blocks: None for 0, EXACT-ZERO tensor for 8
T3  grad(ce, retain=True); grad(w_aux*aux + w_ortho*ortho)
                                                grad_ce = 1,  grad_rest(direct) = 0
      grad_rest[embed]: subtraction form 4.06434155, direct form 4.06434155
      values equal (torch.equal)? True    max|diff| = 0.000e+00
```

So the correct construction is available at zero cost in fidelity — build
`non_ce` from the `aux_loss`/`ortho_loss` tensors `compute_arm_losses` **already
returns**, not by subtraction. Identical gradients, zero backbone traversal, and
backbone params come back `None` instead of full-size zero tensors.

Budget consequence, re-derived from measured rates (`mob_g3b31_compB_s0.json`:
`step 20000, elapsed_s 2985.6` = **0.829 GPU-h**; primary_s0 = 0.840; the design's
0.8–1.0 band is sound):

| construction | per-cell | 12 cells | ≤15 cap |
|---|---|---|---|
| design's estimate (+10–30%) | 0.88–1.30 | 10.6–15.6 | already at/over |
| **as specified** (second FULL backward; fwd+2·bwd with bwd≈2·fwd ⇒ +50–67%) | 1.20–1.67 | **14.4–20.0** | **OVER** |
| direct construction (aux/ortho subgraph only, +2–10%) | 0.82–1.10 | 9.8–13.2 | within |

The design's own contingency ("if the build-time smoke measures overhead above ~20%,
trim the placebo to n=3") is triggered by construction and does not save the cap.

Secondary consequences of the subtraction form, both removed by the fix:
- **Memory.** `retain_graph=True` holds the full activation graph across both
  passes, and the subtraction form materializes a *second complete* model-sized
  gradient (exact-zero tensors for every backbone param) before summation — ~3
  live copies of a 98M-param gradient plus retained activations, against §G3-B31's
  measured 6.86 GB baseline. Neither estimated nor measured anywhere in the design.
- **NaN surface.** The subtraction form multiplies the CE subgraph's local
  Jacobians by exact zero; where any local Jacobian is non-finite, `0 × inf = NaN`
  (executed, T7) and the runner's `finite` gate (line 1355) then **skips the whole
  optimizer step for that arm**. Honest caveat: in the case I constructed
  `grad_ce` was *also* NaN, so this is not a demonstrated *differential* failure
  versus plain `backward()` — but the direct construction is structurally immune
  (`None`, not zeros), so the surface is free to eliminate.

**Disposition D-F2 (binding).** Replace `non_ce_term = total_loss - ce_loss` with
`non_ce = aux_read_loss_weight * aux_loss + ortho_reg_weight * ortho_loss`, built
from the tensors already returned by `compute_arm_losses`, with an explicit guard
for the `aux_loss is None` / `ortho_loss is None` cases (do not rely on
`total_loss is ce_loss`). Note this makes `allow_unused=True` genuinely
load-bearing (backbone params become unreachable ⇒ `None`), so §6.5's masking worry
must be discharged by an explicit assertion that the set of `None` entries in
`grad_rest` is exactly the expected set (backbone blocks + `read_injector`), not
merely tolerated. Re-derive and re-table the budget from the measured overhead.

---

### F3 — The verification plan cannot execute as written. Sub-test (b) raises; sub-test (c) is provably false.

§5(2) is the only thing standing between this new autograd code and a silently
meaningless arm. Both of its load-bearing sub-tests fail.

**(b) crashes.** As specified, it calls `assemble_embed_closed_grads_` and then
*"independently recompute `grad_ce_only = torch.autograd.grad(ce_loss,
[embed.weight], retain_graph=True)[0]` **on the same graph**"*. The helper's second
`autograd.grad` runs with `retain_graph=False`, which frees the graph. EXECUTED:

```
--- smoke sub-test (b) as written: recompute grad_ce_only on the same graph ---
  *** RuntimeError: Trying to backward through the graph a second time (or directly
      access saved tensors after they have already been freed)...
```

**(c) is guaranteed to fail.** It asserts `torch.equal` between the flag-ON assembly
and the flag-OFF `backward()` for *every* parameter except `embed.weight`, and the
helper's docstring makes the same claim ("bit-for-bit what a normal
`total_loss.backward()` would have produced"). Floating-point addition is not
associative: `backward()` sums CE's and aux's contributions **at `o_raw`** and
propagates one tensor down the shared chain; the split propagates each alone and
sums **at the parameter**. `J^T(a+b) ≠ J^T a + J^T b` in float32. EXECUTED:

```
T2  bit-identical (torch.equal) params : 9  -> [blocks.0..3.{weight,bias}, inject.weight]
    NOT bit-identical                  : 2
       entity_adapter.weight  max|diff| = 4.470e-08   allclose=True
       ncr.weight             max|diff| = 1.490e-08   allclose=True
```

The parameters that fail are exactly the ones downstream of the shared `o_raw`/`Z`
node — in the real runner, `integ.entity_adapter` and every `ncr_head` parameter.
Backbone blocks pass (aux never reaches them). This is not a bug in the mechanism;
it is a false claim in the design, wired into an assertion that will halt the build.

The danger is not the crash, it is the repair: a build agent that "fixes" (c) by
loosening `torch.equal` → `allclose` silently deletes the only check that the aux
signal to `entity_adapter`/`ncr_head` was preserved, and the design's own
CLAUDE.md-cited "structural checks need exact thresholds, not tolerance" rule then
reads as having been satisfied when it has been abandoned.

**Disposition D-F3 (binding).** (i) Add `retain_graph=True` to the helper's second
`autograd.grad` when the smoke path is active, or restructure (b) to rebuild the
forward graph — and state which. (ii) Rewrite (c) as a **two-tier** check, both
pre-registered: `torch.equal` for the parameters that are provably order-invariant
(all backbone-block params and `read_injector` — aux/ortho never reach them), and
a pinned numerical bound for `entity_adapter` / `ncr_head` (`allclose` at explicit
rtol/atol AND `max|diff| / max|ref| < 1e-5`), with a **forced-fail negative test**
(inject a deliberate mis-assembly and confirm the bound fires) run to completion,
per this repo's standing rule. (iii) Delete "bit-for-bit" from the docstring.

---

### F4 — compE is invisible to the pinned scorer, and the obvious one-line fix mis-scores it. Fifth and sixth instances of a bug class this log has already counted to four.

`experiment-runs/2026-08-18_premise_multiseed/run_repl_wave2.sh` is the scoring
instrument of record. Read directly:

```bash
OLD=/home/nvidia/ncr_g3b31_contrastive/results
NEW=/ephemeral/reseed_ckpts
for tag in compA compB primary; do
  if [ "$tag" = "compB" ]; then FZ=""; else FZ="freeze"; fi
  for s in $(seq 1 24); do
    NAME="mob_g3b31_${tag}_s${s}"
    for cand in "$NEW/${NAME}_ckpts/${NAME}.ckpt.pt" "$OLD/${NAME}_ckpts/${NAME}.ckpt.pt"; do ...
```

Against the design's §3 as written, **four independent mismatches**:

1. **Arm loop** is `compA compB primary`. `compE` is not in it — the exact scope bug
   EXPERIMENT_LOG #7 called "**fourth instance of this scope/path bug class in four
   ticks**" when it hid compD.
2. **Cell name** is hardcoded `mob_g3b31_${tag}_s${s}`. The design's cells are
   `mob_gembed_compE_s{seed}`. No match, and — because the loop never reaches the
   tag — the `MISSING-CKPT` loud-fail that was added specifically to stop silent
   zero-scoring **cannot fire**.
3. **Checkpoint roots** are `/ephemeral/reseed_ckpts` and
   `~/ncr_g3b31_contrastive/results`. The design writes to
   `/ephemeral/embed_path_ckpts/` — in neither.
4. **Seed range** is `seq 1 24`. The design's test arm is **s21–s28**; s25–s28 fall
   outside the loop even after the tag is added.

And the landmine: `FZ="freeze"` is applied to **every tag except compB**. Appending
`compE` to the loop in the natural way scores it with
`restore_arms_and_opts(..., freeze_entity_adapter=True)` — the wrong instrument
configuration for a trainable-adapter arm, applied **silently**. `pbe_repl.py:16,24`
confirms `freeze` is consumed and passed straight into the restore.

Adjacent, same class: `[ -f "$OUT" ] && continue` skips any cell whose eval JSON
already exists. That is precisely the mechanism that produced the stale-eval
incident (#12/#13) — an eval record that survives while its checkpoint moves on.

**Disposition D-F4 (binding).** Ship the scorer patch **as part of this design**,
not as a build-time detail: a `run_repl_wave3.sh` that (a) parameterises the cell
name prefix, (b) carries a per-tag `FZ` map (compE → `""`, compE_frozen → `"freeze"`),
(c) adds `/ephemeral/embed_path_ckpts` to the search roots, (d) covers the actual
seed range, (e) FAILS LOUDLY on any completed cell with no checkpoint, and (f)
re-scores rather than skips when the checkpoint is newer than the eval record.
Include a negative test: run it against a deliberately-misnamed cell and confirm
`MISSING-CKPT` fires.

---

### F5 — compB and compE do NOT differ only in gradient routing: `clip_grad_norm_` couples the embed cut to every other parameter's effective step.

§2.4 states that everything downstream — `finite`, `clip_grad_norm_`, `opt.step()`
— "is unchanged — it only reads `.grad`". That is true and it is the problem. Line
1357 is `torch.nn.utils.clip_grad_norm_(all_params, 1.0)` over the **global** norm.
Removing the aux+ortho contribution from `embed.weight` changes the global norm,
which changes the clip coefficient, which rescales **every parameter's** post-clip
gradient. With a clip bound of 1.0 and a global norm ≫1, both arms are effectively
doing normalised-gradient descent to a fixed budget of 1.0 — so the cut does not
merely remove embed's aux share, it **redistributes the entire fixed step budget**
toward every other parameter.

EXECUTED (T4), demonstrating the mechanism (magnitudes are the toy's, not the real
model's — that is the point: nobody has measured the real one):

```
global grad norm, compB (no cut)  = 4.174178 -> clip coef @1.0 = 0.239568
global grad norm, compE (cut)     = 0.959370 -> clip coef @1.0 = 1.000000
  -> every OTHER parameter's post-clip gradient is rescaled by 4.3510x
     purely as a side effect of the embed cut
```

AdamW absorbs a *constant* global rescale but not a step-varying one (β1=0.9 and
β2=0.999 have different effective time constants). So the causal licence the design
claims in §1 — that a positive result attributes the change to embed's gradient
routing, because "compB vs compE differ ONLY in the gradient routing" — is not
established by the construction as specified. A positive compE would be confounded
with a global effective-learning-rate change across the whole model.

There is a clean fix that costs nothing.

**Disposition D-F5 (binding).** Do the cut **after** clipping, not before: assemble
the full combined gradient exactly as `backward()` would, run
`clip_grad_norm_(all_params, 1.0)` so the clip coefficient is computed from the
*same* global norm compB used, and only then subtract the (identically-scaled)
`grad_rest[embed]` from `embed.weight.grad`. Every non-embed parameter then
receives compB's post-clip gradient to within F3's numerical bound, and the arms
differ in `embed.weight.grad` alone. If the design prefers the current ordering, it
must instead (i) log `||g||` pre-clip and the clip coefficient every step in BOTH
arms, (ii) pre-register a tolerance for the coefficient distribution shift, and
(iii) add a control arm. The post-clip ordering is strictly cheaper and strictly
cleaner.

---

## §2 MAJOR

### M1 — The claim is registered as an INTERACTION but the 2×2 is only 3/4 built, and the argument that would license it is never made.

The factor grid is (embed leak: open/closed) × (adapter: trainable/frozen):

| | adapter FROZEN | adapter TRAINABLE |
|---|---|---|
| **embed leak OPEN** | primary/compA, 0.9844–1.0000 | compB, median 0.7246 |
| **embed leak CLOSED** | **not run** (§6.4, out of budget) | compE (this wave) |

A positive compE establishes the **simple effect** of closing the leak *at* a
trainable adapter. It does not establish an interaction — that requires showing the
effect *differs* across the adapter levels, i.e. the fourth cell. There *is* a
ceiling argument that mostly rescues it (the frozen arms sit at 0.9844–1.0000, so
the closed-frozen cell cannot move up by more than 0.0156), but it is one-sided:
if closing the leak **degrades** the frozen arm, the interaction's sign is not what
§1/§4 assert. The design never makes this argument at all; it simply asserts the
interaction. §6.4 correctly names the fourth cell and then puts it out of scope.

**Disposition D-M1.** Reallocate the placebo's 4 cells to the fourth cell:
`primary` config + `--close-embed-aux-path`, n=4, seeds matched to existing primary
seeds. Same budget, completes the 2×2, licenses the registered claim, and
discharges §6.4 in-wave instead of deferring it. Pre-register its read as a
one-sided bound: "closed+frozen median ≥ 0.95 ⇒ ceiling argument holds; < 0.95 ⇒
the interaction framing is void and the wave reports the simple effect only."

### M2 — The placebo target is not on the measurement graph at all. Its stated job is undischargeable.

§4 justifies `ncr_head` as the placebo because it "sits on the same conduit". True
of the **training** graph. False of the **measurement** graph, which is what the
verdict is read from. Traced through the code:

Under P1b (`teacher_force=True`), `ncr_lm_forward_ablatable` (line 390–391) takes
`Z = integ.teacher_force_operator(keys_v, values_v)` — a `pinv` least-squares fit on
**detached** `entity_adapter(embed(ids))` outputs
(`ncr_lm_wave1_smoke.py:348-364`), which *"bypasses ncr_head's own BindingEncoder
entirely"*. Then `q_key = integ.query_key(input_ids, ..., backbone.embed)` =
`entity_adapter(embed(q_ids))`; `o_raw = binexp_read(Z, q_key, h)`; and
`retrieval24_acc` is `argmax_k cos(o, entity_adapter(embed(entity_ids_k)))`
(`discriminability_metrics`, lines 480–540).

**So the scored quantity is a pure function of exactly two trained tensors:
`integ.entity_adapter.weight` and `backbone.embed.weight`.** `ncr_head`, every
DeltaNet block, `norm_f` and `read_injector` are all causally absent from it.

Consequences:
- Cutting aux+ortho into `ncr_head` cannot act on the metric except second-order
  (through altered CE dynamics feeding back into embed/adapter). It therefore
  cannot discharge its registered job — ruling out *"any comparably-sized gradient
  ablation improves retrieval by regularization alone"* — because the ablation is
  not comparable; it is off-path.
- §6.6's disclosed weakness ("`ncr_head` is plausibly load-bearing for the aux
  loss's own purpose") is the *wrong worry*. The real problem is the opposite.
- The genuinely informative on-path control is cutting aux+ortho into
  **`entity_adapter`** (trainable in compB, same conduit, and one of the only two
  tensors the metric depends on). That decomposes the damage between the two
  on-path factors instead of testing an irrelevant third.

**Disposition D-M2.** Drop the `ncr_head` placebo. If M1's fourth cell is not
adopted, spend the 4 cells on the `entity_adapter`-target cut instead. Either way,
§4's rationale must be rewritten to distinguish the training graph from the
measurement graph.

### M2b — A near-free eval-only pre-test exists and is not considered.

Because the metric is a function of `(entity_adapter.weight, embed.weight)` alone,
and `COMPB_DRIFT_ANALYSIS.md` already **verified bit-exact seeded reconstruction of
init weights** via `build_arm` (three seeds, `torch.equal` true), a factor-swap
ablation is available on the 20 archived compB checkpoints at *eval cost only*
(~0.02 GPU-h/cell per #13's remediation, ≈0.5 GPU-h total, no training):
re-score each compB checkpoint with (a) `embed.weight` reset to its own seeded
init, (b) `entity_adapter.weight` reset to its own seeded init. If (a) does not
recover retrieval, the embed factor is not carrying the damage and the 10–15 GPU-h
wave is dead for free. Caveat to disclose: resetting embed also changes the target
set, so a rescue is not by itself proof — but a **non**-rescue is close to
dispositive against the hypothesis, at ~4% of wave-1's cost.

**Disposition D-M2b.** Run this before wave-1 launches, or state in writing why a
result that could kill the hypothesis for 0.5 GPU-h is not worth 0.5 GPU-h.

### M3 — The frame ignores three recorded facts that lower the prior, including §G3-B32's own headline.

The design cites §G3-B32 for compA's TPC_fg 0.797–0.814 (§6.4) and never engages
that section's stated verdict, three lines further down
(`NCR_REAL_LM_DESIGN.md:7077-7080`): *"The §G3-B26 target-space mechanism is
therefore **NOT the binding block**: the depth path itself (binexp = power iteration
toward Z's top singular direction) destroys read discriminability in-LM independent
of the aux loss."* Two more facts point the same way:

- §G3-B32 measured **compB's own** TPC_fg at **0.196–0.228** — far below the 0.50
  tripwire and characterised as "both-arms-drifting regime, not the B26 pathology."
  The collapse route the embed leak is supposed to re-open is largely *not firing*
  in the arm this intervention is run on.
- `COMPB_DRIFT_ANALYSIS.md` leg (a) measured the within-compB association between
  target collapse and deep retrieval as **positive** (ρ=+0.4643, p=0.0392 at n=20;
  #13) — more collapse goes with *better* composition, the opposite sign.

**In fairness (the design deserves this and doesn't get it right either):** §G3-B32's
structural-block conclusion was drawn in the OWN-WRITE (P0) regime with an
instrument that did not exist until 2026-08-13 (#9), whereas 0.7246/0.9844 are
teacher-forced P1b. So it is not a strict contradiction. But the design cites the
section selectively, never notes the regime difference, and never states the causal
chain that survives all three facts.

There *is* a chain that survives, and M2's trace supplies it: under P1b the metric
depends on `(entity_adapter, embed)` only; the frozen arms hold `entity_adapter`
fixed and score ~1.0 with embed open, so embed's leak alone is harmless; closing it
under a trainable adapter is therefore the sharp, well-posed remaining test. The
design should write **that**, explicitly, rather than leaning on R2's prose.

**Disposition D-M3.** Add a §1 paragraph stating the P1b measurement-graph chain,
citing §G3-B32's structural-block verdict and leg (a)'s sign, and registering the
prior honestly (this is a plausible-but-contra-indicated mechanism, not a
likely-positive).

### M4 — The bands are not a partition. Executed counterexample.

NULL fires on `median < 0.80` **OR** `MW p > 0.05` **OR** `median inside compB's
IQR`. PARTIAL fires on `median ∈ [0.80, 0.90)`. These overlap whenever a
materially-shifted median coexists with a spread wide enough to lose significance —
which is precisely the regime compB's own 0.617–0.973 spread makes likely.

EXECUTED (`stats_bands.py`, exact Mann–Whitney by the standard DP recurrence):

```
A: compE = [0.6172, 0.6484, 0.8438, 0.8516, 0.8594, 0.8750, 0.9531, 0.9727]
   median = 0.8555   U = 114/160   exact two-tailed p = 8.869e-02
   -> ['PARTIAL', 'NULL']   <<< AMBIGUOUS (>1 band fires)
B: compE = [0.6250, 0.6797, 0.8398, 0.8477, 0.8516, 0.8672, 0.9375, 0.9688]
   median = 0.8497   p = 7.046e-02  -> ['PARTIAL', 'NULL']   <<< AMBIGUOUS
```

Monte Carlo over 20,000 bootstrap draws from compB's own distribution: ambiguous
labels occur in 0.9% of outcomes at zero effect and 3.8% at a +0.05 shift.

Related, same section: `"compB's own IQR"` is never numerically defined (the
median-of-halves reading gives [0.6797, 0.7695]); PARTIAL's justification
("≥0.10 absolute over compB's 0.7246 median") contradicts its own numeric floor of
0.80 (= +0.0754).

**Disposition D-M4.** Re-write as a strict priority ladder evaluated in order
(WIN → PARTIAL → NULL, first match wins, no ORs), pin the IQR numerically or delete
that clause, and reconcile the 0.80 floor with the stated ≥0.10 rationale.

### M5 — Power: WIN is nearly unreachable, n=8 is underpowered for the effect the design itself anticipates, and the headline p is wrong by 3×.

Exact Mann–Whitney, n=8 vs n=20 (`stats_fix.py`):

```
compE n=8 vs compB n=20   C(28,8) = 3,108,105   best two-tailed p = 6.435e-07
   upper-tail critical U >= 119/160 (74.4% of cross-pairs) for p <= 0.05
DESIGN CLAIMS 2/C(28,8) ~= 1.9e-6  ->  actual 2/C(28,8) = 6.435e-07 (off by 3.0x;
   1.9e-6 is 2/C(25,8) = 1.849e-06)
```

Label probabilities, bootstrapping compB's own distribution under an additive shift
(20,000 reps):

```
 shift   P(WIN)  P(PARTIAL)   P(NULL)
  0.00    0.000       0.013     0.996
  0.05    0.000       0.163     0.875
  0.10    0.000       0.735     0.272
  0.15    0.000       0.996     0.004
  0.20    0.000       1.000     0.000
  0.25    0.005       0.995     0.000
  0.30    0.171       0.829     0.000
```

Two readings. (i) **WIN is a dead band** for anything short of a near-total rescue
to frozen-arm levels — P(WIN) ≈ 0 up to a +0.25 median shift, because WIN requires
`min > 0.9727`. The design's own §1 language ("raise ... *toward* the frozen floor")
describes an outcome that lands in PARTIAL essentially always. (ii) n=8 has
P(NULL) = 0.88 against a real +0.05 shift and 0.27 against +0.10 — underpowered for
exactly the modest effects §6.1/§6.3 anticipate. Type-I is fine (1.3% non-NULL at
zero effect).

**The unpaired choice discards the largest available lever.** Seeds are exact
blocking factors in this runner: `build_arm(vocab, seed, device)` calls
`torch.manual_seed(seed)` before construction (init is seed-determined) and
`data_gen = torch.Generator(...).manual_seed(seed + 777)` fixes the training data
stream — both identical under the flag. Running compE on **s1–s8** rather than
s21–s28 permits a paired test that removes compB's sd = 0.0951 seed variance. The
design's stated reason for avoiding s1–s8 ("so no seed number collides with an
existing archived cell") is a naming concern, fully solved by the distinct cell_id
and ckpt path it already specifies.

**Disposition D-M5.** (i) Correct the p arithmetic to 6.435e-07. (ii) Either run
compE on s1–s8 and pre-register a paired exact test (Wilcoxon signed-rank, plus the
sign test, best two-sided p = 2/2^8 = 0.0078) **in addition to** the unpaired MW
against the archived n=20, or state in writing why unpaired is preferred despite
the variance cost. (iii) Re-label the bands so the design's own predicted outcome is
reachable: either lower WIN's separation requirement or rename PARTIAL to reflect
that it is the expected success band.

### M6 — No runner-parity check; two runner copies; `RUNNER_TAG` unguarded.

compE will be trained by a **patched** binary and compared against compB cells
trained by md5 `9a93198b`. §G3-B31's own build ran a legacy-parity smoke ("all diffs
0.0"); this design has no equivalent. Three specific exposures:

- **No flag-OFF parity run.** Nothing verifies the patched runner reproduces the
  unpatched one when `--close-embed-aux-path` is absent.
- **Two copies exist on the box.** `pbe_repl.py` does
  `sys.path.insert(0, dirname(abspath(__file__)))` then `import ncr_lm_wave1_runner`
  — i.e. the scorer imports the copy sitting next to it in `~/ncr_writecond/`,
  while the design pins and patches `~/ncr_g3b31_contrastive/`. The design names one
  md5 and never says which copies are patched or re-pinned.
- **`RUNNER_TAG` must not change.** `load_checkpoint` (line 1129) asserts
  `ckpt["runner_tag"] == RUNNER_TAG`; on mismatch it returns `None` and
  `pbe_repl.py:22` then fires `assert ckpt is not None, "checkpoint missing"`.
  A build agent bumping the tag for a new code path silently un-loads **every
  archived compB and frozen checkpoint**. The design is silent.

**Disposition D-M6.** Add to §5: (a) a flag-OFF parity smoke — one seed, ~200 steps,
`torch.equal` on the full loss trajectory and on both arms' state dicts against the
unpatched runner; (b) an explicit statement of which runner copies are patched, with
new md5s pinned for each; (c) a pinned invariant `RUNNER_TAG == "ncr_gate3_wave1_runner_v1"`
with a comment saying why it must never change.

### M7 — Bookkeeping regressions against already-adopted findings.

- **`close_embed_aux_path` is not added to `rec["config"]`.** Lines 1257–1264 record
  every existing flag; the design records only the per-step diagnostic. The results
  JSON would not say which condition the cell ran under — in a campaign whose last
  three ticks were a retrodiction, a stale-eval incident, and a flag-scope bug.
- **The placebo's target selector has no checkpoint key and no resume assert.** §3
  defers `--close-target=ncr_head|embed` vs a second flag to "audit's call at build
  time" — but the seed-trap and freeze-trap precedents exist *because* an
  unrecorded flag corrupts a resume silently. Whichever form is chosen needs the
  same `save_checkpoint` key + resume assert §2.5 writes for the main flag.
- **Rounded literals reintroduced.** #7 adopted the finding that `0.9844` is a
  rounded literal of `252/256 = 0.984375` that "rounds the wrong way on both bands";
  this design uses `0.9844` and `0.9727` (true value `249/256 = 0.97265625`). No
  achievable value falls in the induced gap (grid spacing 1/256), so it is not
  live — but it is a reverted fix.
- **No single source of record for n.** §1 says the frozen floor is n=18; §4 says
  "primary n=14 + compA n=6" (= 20); #10 records compA at n=8. #13 is the correct
  source (frozen n=18, compB n=20). Cite it once.

**Disposition D-M7.** Fix all four; add `close_embed_aux_path` (and the placebo
selector) to `rec["config"]`, `save_checkpoint`, and the resume assert; use
`249/256` and `252/256`; cite #13 as the sole numeric source.

### M8 — No placement / VRAM / utilization figures. Violates the 07-16 pre-launch design gate.

CLAUDE.md makes saturation-packing a **design gate**: "predicted SM util + memory in
every design." §G3-B31 supplied 6.86 GB, 73–80% SM, 0.83 GPU-h/cell, one cell per
GPU with an explicit no-packing ruling. This design supplies none of it, for a
12-cell wave whose memory profile it has changed (F2's retained graph + second
full-model gradient) and whose per-cell time it has mis-estimated.

**Disposition D-M8.** Add a placement section: measured peak VRAM under the flag
(smoke), predicted SM util, cells-per-GPU, wall-clock critical path, and the ≤15
GPU-h reconciliation after D-F2's re-derivation.

---

## §3 minor

- **m1 — `all_params.index(embed_w)` is a footgun that works only by accident.**
  `list.index` short-circuits on identity, so this returns 0 *because*
  `self.embed = nn.Embedding(...)` is DeltaNetLM's first registered parameter
  (`lm_pretrain_rd.py:1223-1229`, verified: `param order[0] = embed.weight`). Any
  future reordering makes it compare tensors elementwise. EXECUTED:
  `[t1, t2].index(t2)` → `RuntimeError: The size of tensor a (4) must match the size
  of tensor b (5)`. Use an identity scan (`next(i for i,p in enumerate(...) if p is embed_w)`).
- **m2 — the `total_loss is ce_loss` branch is self-contradictory.** The helper
  returns `cut_active: False`; the caller unconditionally asserts `cut_active`.
  Unreachable in the compE config (aux 0.5 and ortho 0.1 are both > 0 on
  `full_graft`), but as written that path is a guaranteed `AssertionError`. Also
  note D-F2 removes the `is`-identity test entirely.
- **m3 — line-number drift**, despite §0's claim that all line numbers were read off
  the box. `ortho_regularization_loss(Z)` is applied at **line 850**, not 743 (743
  is blank); the additive sum is lines **848 and 851**, not "845–850"; the per-arm
  loop header is **1317**, not ~1315; `nn.init.normal_(self.embed.weight, ...)` is
  **1229**, not ~1223. Verified correct: 630, 690, 360, 768, 752, 908, 1086, 1136,
  1185, 1334–1335, 1835, and `lm_pretrain_rd.py:1310`.
- **m4 — the has-teeth check's `min_conduit_norm = 1.0` is an arbitrary absolute
  floor** against an R2 reference of 110.13 measured on a different arm at an
  unstated step — 100× slack, and it tests the wrong thing (see M-extra below).
- **m5 — per-step `.item()` on the conduit norm forces a host sync every step**
  (20,000 steps × 12 cells), unmeasured. Log every N steps, or accumulate on device.

---

## §4 Direct answers to the brief's numbered surfaces

**(1) Is the split-backward correct PyTorch?** The assembled *values* are correct;
`retain_graph=True` on the first call is required and present; nothing is
double-counted or dropped; `allow_unused=True` is safe for `grad_ce`; it composes
correctly with AdamW's param groups (bit-identical step, T8); there is no
AMP/GradScaler to interact with; and `grad_ce[embed]` correctly includes **both** of
embed's roles (lookup **and** tied head) plus the o-side route — which is what the
design wants and states. **Three things are wrong:** the second pass *does* traverse
the backbone (F2), the result is *not* bit-identical to `backward()` (F3), and the
downstream `clip_grad_norm_` breaks the "gradient routing only" premise (F5).

**(2) Does the intervention cut what it claims — and can the assertion pass
vacuously?** The cut is real; both conduits (o-side aux, Z-side ortho) close. But
the R2 target of 110.13 is not a valid reference (frozen-adapter arm, bare-cosine
config, unstated step) and the design correctly declines to pin to it — then
replaces it with an *absolute* floor of 1.0 that cannot detect the failure mode that
matters. **The vacuous-pass mode:** CE's own gradient reaches `embed` through three
routes (lookup, tied head, o-side) and none of them are cut. If the aux+ortho share
is a small fraction of `‖embed.grad‖`, the assertion passes (norm ≫ 1) while the
intervention is a rounding error on embed's trajectory — a guaranteed NULL that says
nothing about the mechanism. My toy measured a 95.9% aux share; the real model's
share is unknown and unmeasured. **The converse mode:** the assertion never checks
that `embed.grad` actually *equals* `grad_ce` — only that the discarded quantity was
nonzero. Passing `total_loss` where `non_ce_term` was intended would sail through.
**Disposition D-A2 (binding):** replace the absolute floor with (i) a measured,
pre-registered **ratio** `‖grad_rest[embed]‖ / ‖grad_ce[embed]‖` with an abort rule
if it is below a pinned threshold at smoke time, and (ii) a per-step (or per-N-step)
`torch.equal(embed.grad, grad_ce[embed])` identity assertion so the check has teeth
in both directions.

**(3) Claim logic.** Existing arms pin three of four cells; the fourth
(closed-leak + frozen adapter) is missing and the interaction claim is not licensed
without it or without an explicit one-sided ceiling argument (M1). compB and compE
do **not** differ only in gradient routing (F5). The `entity_adapter` effective-LR
worry the brief raises is real but indirect — `entity_adapter`'s own gradient is
preserved exactly (to F3's numerical bound), so the divergence enters through the
clip coefficient and, downstream, through Adam's moment state diverging as the
trajectories separate; that is intrinsic to any intervention and is fine **once F5's
global rescale is removed**.

**(4) Bands and power.** Not a partition (M4, executed counterexample). n=8's best
achievable exact two-tailed p is **6.435e-07**, not 1.9e-6; p ≤ 0.05 needs
U ≥ 119/160. n=8 is adequate for a ≥0.15 shift, underpowered below +0.10, and WIN is
effectively unreachable below a +0.30 shift. compB's 0.617–0.973 spread is exactly
why the median-shift criterion is weak, and the unpaired-fresh-seed choice
gratuitously forgoes the blocking that would fix it (M5). **The placebo at n=4 is
not statistically unusable** — best exact p = 2/C(24,4) = 1.88e-4, critical
U ≥ 66/80 — but the design registers **no band for it at all**, in a program whose
recurring failure mode is post-hoc band drift. If it survives M2, give it a read
rule.

**(5) Budget/placement.** Re-derived from `mob_g3b31_compB_s0.json` (0.829 GPU-h at
step 20000): the design's floor is right, its overhead multiplier is not. As
specified the wave is **14.4–20.0 GPU-h** (over the cap); with D-F2's fix it is
**9.8–13.2**. The `/ephemeral` checkpoint policy (#4) is correctly honoured for
*writing* and silently broken for *reading* (F4). Split-backward overhead is
**guessed**, explicitly flagged as an estimate by the design — but the estimate rests
on a claim that is false, so it is not merely imprecise, it is wrong-signed. No
placement/VRAM/util figures at all (M8).

**(6) The designer's three disclosed weaknesses.** *Ambiguous null* — under-stated:
§6.3 covers "mechanism wrong vs mechanism swamped" but not the third reading F1's
missing h=1 co-condition admits ("the intervention broke the read"), nor the fourth
that the vacuous-pass mode admits ("the cut was numerically negligible"). Both are
closable by D-F1 and D-A2. *Unaudited autograd engineering* — honestly disclosed and
the disclosure was correct: this round found three real defects in it (F2, F3, F5).
*Non-inert placebo* — disclosed, but the disclosed worry is the wrong one; the real
defect is that the target is off the measurement graph entirely (M2). All three are
**adequately disclosed and inadequately mitigated**; none is fatal in itself.

---

## §5 Verdict and binding disposition list

**REV-REQUIRED.** The hypothesis is well-posed, the conduit analysis is correct and
genuinely advances R2, and the chosen mechanism is the right one. But as specified
the wave would run over budget, on a code path whose verification plan cannot
execute, scored on an unnamed regime by an instrument that cannot see it, with a
global optimization confound that voids the causal licence. Do not build against
DRAFT-R0.

**Binding on Rev-1 (all must be discharged in the document before any build agent
is dispatched):**

| # | Disposition | Source |
|---|---|---|
| D-F1 | Pin metric to `P1b.result["h=61"]["retrieval24_acc"]`, teacher-forced regime, seed 90210 / n=256; add the `median(P1b h=1) ≥ 0.95` validity co-condition; add an attrition rule (read at n≥7, void below) | F1 |
| D-F2 | Build `non_ce` from `aux_loss`/`ortho_loss` directly, never by subtraction; assert the exact expected set of `None` entries in `grad_rest`; re-derive the budget table | F2 |
| D-F3 | Fix sub-test (b)'s `retain_graph`; rewrite sub-test (c) as a two-tier exact/bounded check with a forced-fail negative test run to completion; delete the "bit-for-bit" docstring claim | F3 |
| D-F4 | Ship `run_repl_wave3.sh` with the design: parameterised cell prefix, per-tag freeze map, `/ephemeral/embed_path_ckpts` in the search roots, correct seed range, loud MISSING-CKPT, re-score-not-skip; include its negative test | F4 |
| D-F5 | Move the cut to **after** `clip_grad_norm_`, so every non-embed parameter receives compB's post-clip gradient and the arms differ in `embed.weight.grad` alone | F5 |
| D-A2 | Replace the absolute has-teeth floor with a pre-registered `‖grad_rest[embed]‖ / ‖grad_ce[embed]‖` ratio + smoke-time abort rule, plus a `torch.equal(embed.grad, grad_ce[embed])` identity assertion | brief (2) |
| D-M1 | Reallocate 4 cells to the fourth 2×2 cell (frozen adapter + closed leak) with a pre-registered one-sided read, or drop the INTERACTION framing | M1 |
| D-M2 | Drop the `ncr_head` placebo; rewrite §4 to separate training graph from measurement graph; if a control survives, target `entity_adapter` | M2 |
| D-M2b | Run the eval-only init-swap pre-test on the 20 archived compB checkpoints (~0.5 GPU-h) before wave-1, or justify in writing why not | M2b |
| D-M3 | Add the P1b measurement-graph causal chain to §1; cite §G3-B32's structural-block verdict, compB's own TPC_fg 0.196–0.228, and leg (a)'s positive sign; register the prior honestly | M3 |
| D-M4 | Re-write bands as a strict first-match ladder; pin or delete the IQR clause; reconcile the 0.80 floor with the ≥0.10 rationale | M4 |
| D-M5 | Correct the exact p to 6.435e-07; adopt seed-paired s1–s8 with a pre-registered paired test, or justify unpaired; make the design's own predicted outcome land in a reachable success band | M5 |
| D-M6 | Add a flag-OFF parity smoke vs the unpatched runner; state which runner copies are patched and re-pin every md5; pin `RUNNER_TAG` as invariant | M6 |
| D-M7 | Add the flag (and any placebo selector) to `rec["config"]`, `save_checkpoint`, and the resume assert; use `249/256` / `252/256`; cite #13 as the sole numeric source | M7 |
| D-M8 | Add measured peak VRAM under the flag, predicted SM util, cells-per-GPU, wall-clock critical path, and the ≤15 GPU-h reconciliation | M8 |
| D-m1..m5 | Identity scan instead of `list.index`; resolve the `cut_active` contradiction; correct the four drifted line numbers; log the conduit norm every N steps rather than every step | minor |

**Recommended round-2 scope (narrow):** verify D-F2/D-F3/D-F5 against a real-CUDA
smoke on the box (the CPU reproductions here establish the semantics; only a real
run establishes the overhead, VRAM, and conduit-ratio numbers), re-check the band
ladder for partition, and confirm the scorer patch's negative test fires.

---

### Appendix — executed artifacts

- `.../scratchpad/repro_split_backward.py` — T0–T6: backbone-traversal counting,
  bit-identity, subtraction-vs-direct equivalence, sub-test (b) crash,
  `clip_grad_norm_` coupling, `list.index`, `retain_graph` control. PyTorch 2.8.0.
- `.../scratchpad/stats_bands.py`, `stats_fix.py` — exact Mann–Whitney null by DP
  recurrence, band-partition counterexamples, 20,000-rep label-probability
  bootstrap, n=8 / n=12 / n=4 critical values, budget re-derivation. scipy 1.13.1.
- Code read at byte level: `ncr_lm_wave1_runner.py` (md5 `9a93198b…`, repo copy),
  `pbe_repl.py.txt`, `run_repl_wave2.sh`, `ncr_lm_wave1_smoke.py` (NCRIntegration),
  `lm_pretrain_rd.py` (DeltaNetLM), `mob_g3b31_{compB,primary}_s0.json`,
  `writecond_premise_REPL_compB_s6.json`.
