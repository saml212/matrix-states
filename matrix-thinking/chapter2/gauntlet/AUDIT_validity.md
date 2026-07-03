# AUDIT: Experimental Validity of Task D (Chapter 2 Rank Gate)

Auditor mandate: not "does the code run" but "if it runs, is a PASS or FAIL
scientifically interpretable as evidence for/against H_primary?" Reviewed
`TASK_D_PREREGISTRATION.md`, `task_d.py`, `model_v4.py`, `run_task_d.py`,
`rank_utils.py`, both ATTACK files. All claims below were checked against the
actual code (instantiated with real `torch`) or against a numerical
reproduction of the pre-registered objective, not just read by eye.

---

## Item 1 — Blank-out test: vacuous as written

**Severity: MAJOR.** Bottleneck holds in the *current* code (verified below),
but the test that is supposed to verify it cannot fail regardless.

`run_task_d.py` smoke step 5:
```python
Z = model.encode(b["keys"], b["values"])
pred_a = model.unbind(Z, b["query_keys"])
b_corrupt = dict(b)
b_corrupt["keys"] = torch.randn_like(b["keys"])      # never read by unbind
b_corrupt["values"] = torch.randn_like(b["values"])  # never read by unbind
pred_b = model.unbind(Z, b_corrupt["query_keys"])     # SAME tensor as b["query_keys"]
assert torch.equal(pred_a, pred_b)
```
I instantiated the real model and ran this exact sequence:
`b_corrupt["query_keys"] is b["query_keys"]` → **`True`** (a shallow `dict(b)`
copy never touches that key). So the test calls `unbind(Z, X)` and then
`unbind(Z, X)` again with byte-identical `Z` and `X`. `torch.equal` is
guaranteed to hold by determinism alone — the assertion cannot fail no matter
what the architecture does, because `unbind`'s signature is `(Z, query_keys)`
and simply never receives `keys`/`values` as arguments in the first place.
This is not "testing a bottleneck that happens to hold" — it is a tautology
that would pass identically even for a deliberately leaky decoder, as long as
that decoder's Python signature is `(Z, query_keys)`.

**Does the bottleneck nonetheless hold?** Yes, by construction, independent of
this test: `unbind` is a bare `@staticmethod` with exactly two parameters, no
`self` access, and `MatrixMemoryModel`/`BindingEncoder.forward` set no
persistent attributes. There is no KV-cache, no closure, no second channel —
the Arch-1 leak this test was modeled on (a full-attention transformer decoder
re-attending raw prefix tokens, per `ATTACK_task_shortcuts.md`) structurally
cannot occur in this architecture, because there is no attention over raw
tokens at decode time at all — decode is a single batched `einsum`. So the
*property* is real; the *test* is decorative.

**Why this matters:** the pre-registration calls this "mandatory,
pre-training" and treats its pass as licensing everything downstream (§4:
"If they change, the bottleneck leaks and every downstream number is
meaningless — fix before training"). As written, "blank-out PASSED" in a run
log is not evidence of anything and should not be cited as verification. It
also provides zero protection against a plausible future regression — e.g.
someone "helpfully" refactors `unbind` to take the whole `batch` dict (a
natural-looking change given the dict-based batch pattern used everywhere
else in this codebase) and reintroduces Arch-1 silently; this test would keep
passing.

**Fix:**
1. Replace/augment with a genuine behavioral check: compute `Z_a` from `b`
   and `Z_b` from a *different* sampled batch, and confirm
   `unbind(Z_a, q) != unbind(Z_b, q)` for the same query — i.e. assert Z
   actually matters (currently untested; the existing test only asserts raw
   bindings *don't* matter, never that Z *does*).
2. Add a structural/static assertion, e.g. `inspect.signature(unbind)` has
   exactly `("Z", "query_keys")` and `isinstance(vars(MatrixMemoryModel)["unbind"], staticmethod)`,
   and document that this codifies the no-leak guarantee as a property of the
   architecture, not something the runtime test discovers.
3. Note the pre-registration's §4 phrase "causally among themselves, for
   autoregressive multi-item decode" describes a sequential decode loop; the
   actual implementation decodes all `Q` queries in one parallel `einsum`
   call. This is *safer* (no per-step re-encoding path for Arch-2 to exploit)
   but is a documentation mismatch worth a one-line note in the prereg.

---

## Item 2 — Readout pinning: CONFIRMED CORRECT, no issue

`MatrixMemoryModel.unbind` = `torch.einsum("bij,bqj->bqi", Z, query_keys)` —
a single linear matrix–vector product, zero extra learned parameters, zero
nonlinearity. `recovery_cosine` / `cosine_loss` in `task_d.py` /
`run_task_d.py` score **absolute** cosine similarity between `pred` and the
*true* `v_j`; there is no comparison against the sample's other `K-1` values,
no softmax/argmax over a candidate set anywhere in the training loss or
`evaluate()`. This correctly closes the Nichani et al. (2412.06538) shortcut
as pre-registered in §2–3. **No fix needed — flag as a verified positive.**

---

## Item 3 — rank ≥ K premise: generator is sound, but the default weakens the operational test

**Severity: MAJOR** (ties directly to Item 4).

`_random_directions` gives i.i.d. Gaussian unit vectors (`orthogonal=False`,
the default) or exact QR-orthonormal vectors (`orthogonal=True`). Both are
linearly independent for `K ≤ d` (a.s. for Gaussian, exactly for QR), so the
theorem "exact recovery (cos=1) of K independent bindings forces rank(Z)≥K"
holds under either setting — the *theorem* is not at risk.

Two gaps:
- The self-test's rank check (`vrank >= min(K,d) - 1`) is looser than
  necessary — should assert `== K` for a tight verification; currently would
  silently tolerate a rank-deficient batch. Minor.
- **The near-orthogonality slack is not cosmetic.** Pairwise key/value cosine
  under `orthogonal=False` is ~`N(0, 1/d)` (std ≈ 0.25 at d=16) — real
  correlation that a rank-deficient `Z` can exploit as "free" partial credit
  under a τ=0.9 *approximate* (not exact) recovery bar. Quantified in Item 4:
  this measurably moves the observed knee away from K. `orthogonal=True`
  should be the default for the K=8 causal-test operating point, not an
  opt-in flag.

**Fix:** set `--orthogonal` as the default for M2/M3's fixed operating point
(§6), or explicitly re-register the corrected (weaker, τ-dependent) knee
prediction if `orthogonal=False` is kept.

---

## Item 4 — THE KEY QUESTION, quantified

**Severity: MAJOR — real, quantified false-negative/smearing risk.**

I directly probed the question with a numerical reproduction of the exact
pre-registered objective: for a batch of `K` fresh random keys/values, fit
`Z = U V^T` (`U,V ∈ R^{d×k}`, i.e. rank ≤ k by construction) via Adam,
minimizing `mean_j(1 − cos(Z k_j, v_j))` — byte-for-byte the same loss
`run_task_d.py.cosine_loss` uses — to convergence, per instance, averaged
over 20 fresh instances per (k, config). This answers the *existence*
question directly: does a rank-k<K matrix that clears τ exist and is it
trivially reachable by gradient descent on the pre-registered loss? (Method
and full sweep code available on request; verified no NaNs, verified k=K and
k=d recover cos=1.000 exactly as the theorem predicts, confirming the
optimizer is correctly finding the true rank-constrained optimum.)

**d=16, K=8 (the pre-registered operating point), 20 instances/cell:**

| k | orthogonal=**False** (current default) mean_cos | frac. instances where **ALL 8** bindings clear τ=0.9 | orthogonal=**True** mean_cos | frac. all-8 clear τ=0.9 |
|---|---|---|---|---|
| 4 | 0.873 | 0.00 | 0.707 | 0.00 |
| 5 | 0.927 | 0.15 | 0.791 | 0.00 |
| **6 (K−2)** | **0.964** | **0.90** | 0.866 | 0.00 |
| **7 (K−1)** | 0.988 | 1.00 | **0.935** | **1.00** |
| 8 (K) | 1.000 | 1.00 | 1.000 | 1.00 |

**Finding: under the current DEFAULT (`orthogonal=False`, τ=0.9), a
rank-(K−2) matrix already clears the pre-registered success bar on all K
bindings in 90% of instances**, and does so at mean cosine 0.964 — nowhere
near a marginal pass. This is two ranks below the pre-registered CONFIRM
window `k* ∈ [K−1, K+1] = [7,9]`. It is not a contrived adversarial
construction: it is exactly the loss-minimizing behavior the real
`BindingEncoder` is trained to find, and this project's own prior evidence
(`KILL_LIST.md` Lesson 2, quoted in `ATTACK_task_shortcuts.md` §0: "the model
always finds the lowest-rank representation that satisfies the loss") says
SGD is very likely to find it too.

**Under `orthogonal=True`, the knee is sharp and lands exactly at K−1 = 7**,
inside the pre-registered window — but only barely. The orthonormal case has
a clean closed form: `mean_cos(k) = sqrt(k/K)` (verified to 3 decimal places
against every k in the table: sqrt(6/8)=0.866, sqrt(7/8)=0.935, etc.). This
arises because the mean-cosine objective, applied to K *equal-weight*
orthogonal targets with an unconstrained rank-k budget, prefers to spread
imperfect coverage evenly across all K directions (cos ≈ sqrt(k/K) each)
rather than perfectly solve a k-subset and abandon the rest — a smooth,
concave-in-allocation optimum, not a hard "keep k / drop K−k" partition. This
means the knee location `k* = ceil(τ²·K)` is a direct, sensitive function of
τ even in the *idealized* exactly-orthogonal case: to land inside
`[K−1,K+1]` at all requires `τ > sqrt(1 − 2/K)` (= 0.866 at K=8). τ=0.9 clears
this by a margin of only 0.034 — i.e. the pre-registered default is only
barely inside the safe zone, and only for the orthogonal setting, and only at
K=8 specifically (the margin shrinks for smaller K in the M1 sweep).

**K/d ratio also matters for `orthogonal=False`:** repeating at K=4, d=16
(K/d=0.25 instead of 0.5) sharpens the transition (k=3=K−1 already at 0.939
mean cos / 45% all-pass, full pass only at k=4=K) — smaller K/d reduces the
cross-key correlation that lets low rank "cheat," but does not eliminate it.

**Bottom line for M2/M3 interpretability:** if SGD, exactly as the hypothesis
predicts, discovers rank ≈ K, then under the current defaults the *measured*
knee is likely to appear at k ≈ K−2, i.e. **outside** the pre-registered
CONFIRM window, and a force-rank-(K−2) M3 run is likely to reach *near-ceiling*
accuracy despite K−2 < K. Per §6 this would not be a HARD FALSIFY (only
force-rank-1 reaching ceiling triggers that), but it would break the
"monotone step at k=K(±1)" CONFIRM criterion and the M2 knee-window CONFIRM
criterion, landing the result in INCONCLUSIVE — a **false negative** relative
to the true underlying mechanism, exactly the failure mode the prompt asked
me to check for.

**Recommendation (concrete, before Stage 1):**
1. **Default to `orthogonal=True`** for the K=8 causal-test operating point.
2. **Raise τ to 0.95** (not 0.9). Under orthogonal=True this gives
   `k* = ceil(0.95² · 8) = ceil(7.22) = 8 = K` exactly — the cleanest
   possible signal, versus τ=0.9's thin, coincidental K−1 margin.
3. Report M2/M3 at a **second K/d ratio** (e.g. K=4,d=16) as a robustness
   check, since the smearing effect is K/d-dependent under `orthogonal=False`
   and worth knowing about even after switching the default.
4. Always report the **full mean_cos(k) curve**, not just the thresholded
   recovered_frac — the sqrt(k/K) shape is itself diagnostic and should be
   checked against the trained model's actual curve as a sanity/consistency
   check independent of τ.

---

## Item 5 — Metrics vs. decision criteria: M2's grid is too coarse to test its own criterion

**Severity: MAJOR** (interacts directly with Item 4's finding).

- **M1** (`effective_rank_mean/std`) is correctly computed and matches the
  §6 metric definition. Gap: no code in `chapter2/` computes the actual
  decision statistic (Spearman ρ(K, eff_rank) across the K-sweep, or the
  `eff_rank ∈ [0.7K,1.3K]` check) — `evaluate()` only returns raw per-run
  numbers. This aggregation script does not yet exist. Not fatal (raw data is
  saved), but it should be written and pre-registered *before* Stage 1
  results come back, not improvised at read time. MINOR.

- **M2** (`rankk_curve`): `evaluate()`'s default `rank_ks=(1,2,4,8,16)` skips
  every value between 4 and 8 — **it never probes k=5, 6, or 7 for K=8**, so
  it structurally cannot report the pre-registered knee `k* ∈ {7,8,9}` at any
  resolution finer than "first tested point that clears the bar." Combined
  with Item 4's finding that the *true* (finer-resolution) knee under
  defaults sits at k≈6, this coarse grid would report "k*=8 confirms" while
  masking that a finer sweep shows the real capacity boundary is two ranks
  earlier — the opposite of what the instrument should do. **Fix: densify
  `rank_ks` to include every integer 1..K+2** (e.g.
  `{1,2,3,4,5,6,7,8,9,10,12,16}`) so the knee is actually measured, not
  inferred from a coincidence of which coarse grid points happen to be
  tested.

- **M3** (`--force-rank-k`, the primary causal test): the CLI correctly
  routes to the same `train()` call for every condition, so results *would*
  be comparable across k **provided** an external launcher holds
  steps/batch_size/lr/seed fixed while sweeping only `force_rank_k`. I
  searched the repo and found **no such launcher exists yet** for
  `run_task_d.py` (`matrix-thinking/scripts/run_sweep.py` and
  `pebble_launcher.sh` do not reference `task_d`/`run_task_d` at all — they
  belong to the separate, older `run_matrix_codi.py` pipeline). §7's Stage 1
  plan ("~20 short runs") currently has no driver script producing them.
  MINOR/procedural, but a real pre-flight gap: write and archive this
  launcher (with fixed seeds and hparams per the compute plan) before
  claiming M3 results are comparable across k.

---

## Item 6 — HRRVectorMemory baseline: under-powered on both axes C2 requires

**Severity: MAJOR overall; FATAL for interpreting M4 as currently
implemented** (M4 is explicitly non-gating per §6, so this does not sink the
overall PASS/FAIL verdict, but any M4 claim drawn from the current code would
be confounded and should not be reported as-is).

I instantiated both arms directly:

| | matrix arm encoder (`BindingEncoder`, d=16,h=64,3 layers) | vector arm (`HRRVectorMemory`, d=16,h=64) |
|---|---|---|
| **Total learned params** | **170,896** | **4,256** |
| **Bundled state dimensionality** | 256 (`Z ∈ R^{16×16}`) | **16** (`s ∈ R^{16}`) |

**Two direct violations of C2** ("Reshape-equivalent, weight-tied VECTOR
control arm at **matched params** (**d²-dim vector state** + vector
binding...)"):
1. **Param count is off by 40.15×** (170,896 vs 4,256), not "matched."
2. **State dimensionality is off by 16×** (16-dim vs the required d²=256-dim).
   A 16-dim HRR bundle holding 8 items is deep in the lossy/crosstalk regime
   by construction (Plate 1995 capacity scales with bundle dimensionality);
   the pre-registration's own C2 spec called for 256 dims specifically to
   avoid this and give the vector arm a fair shot.

There is also a **mechanism asymmetry** layered on top of the two
quantitative gaps: the matrix arm's encoder is a 3-layer self-attention
transformer plus an iterative cross-attention "row-reader" — a rich, learned,
permutation-invariant set-aggregation function. The vector arm's binding
operator (`circconv`) is deliberately fixed/non-learned (a legitimate,
principled choice — it is the correct classical HRR baseline per Plate 1995,
and the code's own comment says so), but this means the *only* learned
components are the two shallow 2-layer `enc_k`/`enc_v` MLPs. Even granting
that the binding operator itself should stay parameter-free by design, those
surrounding per-item transforms are ~40× smaller than what they're being
compared against.

**Net effect:** as implemented, M4 would almost certainly show the matrix arm
"winning" — but that result would be uninterpretable, because the vector arm
never received the state budget or learned capacity the pre-registration
specified. This is exactly the confound `ATTACK_baseline_confound.md` §1
warned about (reshape-equivalence / weight-sharing masquerading as
"matrixness"), reappearing in a cheaper form.

**Fix (both required before trusting any M4 number):**
1. Resize the HRR bundle to d²=256 dims (expand k/v through a learned
   d→d² projection before circular convolution, or use a genuinely d²-dim
   binding), matching C2's literal spec.
2. Param-match the learned components to ~170K (grow `enc_k`/`enc_v`, and/or
   add a learned attention-based set aggregator ahead of the fixed `circconv`
   bind, keeping the bind operator itself parameter-free as the principled
   choice) — or, more cheaply, shrink `BindingEncoder` to ~4K params and
   bracket the comparison in both directions.
3. Report an explicit param-count table for all arms (including the
   reshape-equivalent "block-weight-tied vector MLP" arm
   `ATTACK_baseline_confound.md` recommends) before any M4 claim is made.

---

## Overall verdict

**As currently written, this would NOT reliably yield an interpretable
go/no-go**, primarily because of Item 4 (quantified) compounded by Item 5's
coarse M2 grid:

- Under the pre-registered **defaults** (`orthogonal=False`, τ=0.9), a
  numerical reproduction of the exact training objective shows a rank-(K−2)
  matrix robustly clears the success bar on all K=8 bindings (90% of
  instances, mean cos 0.964) — two ranks below the pre-registered CONFIRM
  window. This is not a hypothetical; it is the loss-minimizing behavior the
  real encoder is trained toward, consistent with this project's own prior
  finding that SGD gravitates to the lowest rank the loss tolerates. A TRUE
  hypothesis run could therefore land as INCONCLUSIVE rather than CONFIRM
  purely from this measurement-design gap — a real false-negative risk, not
  a nitpick.
- Item 5's coarse `rank_ks=(1,2,4,8,16)` grid could actively mask this: it
  cannot probe k=5,6,7, so it may report a falsely clean "k*=8" pass without
  ever revealing the true knee sits earlier.
- Item 6 makes any M4 (vector-baseline) claim currently uninterpretable
  (40× param gap, 16× state-size gap, both favoring the matrix arm) — though
  M4 is correctly scoped as non-gating in §6, so this alone does not sink the
  primary M1/M3 verdict.
- Item 1's blank-out test provides no real protection (it is tautological by
  construction) but the bottleneck it's meant to verify does hold in the
  current code by architectural inspection — so it does not currently
  compromise results, only future-regression safety.
- Items 2 and 3's core theorem are sound and correctly implemented — no
  changes needed there beyond the `orthogonal` default (Item 3/4).

**Recommended before Stage 1 launch (all small, hours not days):** default
to `--orthogonal`, raise τ to 0.95, densify `rank_ks` to every integer
1..K+2, fix `HRRVectorMemory`'s state dim and param count to match C2 (or
formally re-scope M4's interpretive weight down further), replace the
blank-out test with a genuine behavioral + structural check, and write the
M3 sweep launcher with fixed seeds/hparams before running. With those fixes,
the underlying design (hard bottleneck, pinned linear readout, provable
rank≥K theorem, train-time force-rank-k causal mechanism) is sound and would
produce a genuinely decisive result either way.
