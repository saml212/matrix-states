# ROUND-2 AUDIT: Experimental Validity + Adversarial Review of Task D Rank Gate

**Mandate:** verify the round-1 fixes (orthonormal keys+values default, τ reported
at {0.9,0.95,0.99} with the knee test at τ=0.99, dense rank grid k=1..d,
gradient-based blank-out test, resolution pre-flight, vector baseline descoped)
actually make the gate interpretable and shortcut-free; hunt for anything new.
Reviewed `gauntlet/AUDIT_validity.md`, `gauntlet/AUDIT_adversarial.md` (round-1),
`TASK_D_PREREGISTRATION.md`, `task_d.py`, `model_v4.py`, `run_task_d.py`,
`rank_utils.py`.

**Method.** No `torch` available in this sandbox (same constraint round-1 hit).
All quantitative claims below are backed by a faithful `numpy` re-implementation
of the exact generator (`_random_directions`, both `orthogonal=True/False` paths)
and the exact rank projection (`truncate_to_rank` via `eigh(ZZ^T)` — same
operation as `rank_utils.py`, reimplemented line-for-line), plus an independent
from-scratch Adam optimizer (manual gradients, verified against the analytic
cosine-similarity gradient) that reproduces `run_task_d.py`'s `cosine_loss`
exactly, to compute the *true* rank-k optimum the training objective would push
toward. All scripts are in scratchpad; every numeric claim is reproducible.
Numpy-on-Accelerate spuriously warns `RuntimeWarning: invalid value encountered
in matmul` on ordinary finite matmuls (same benign macOS/BLAS quirk round-1
flagged) — confirmed no actual NaN/Inf in any output via explicit checks.

---

## Headline result: the τ=0.9 false-pass is CLOSED, and by a wide margin

Round-1's Finding 1 (`AUDIT_adversarial.md`): under `orthogonal=False, τ=0.9`, a
rank-(K−2) matrix cleared the success bar on **all 8** bindings in 90% of
instances. I reran the identical "fraction of instances where all K bindings
clear τ" statistic under the **new** default (`orthogonal=True`), using the
exact Eckart-Young truncation of the true correlation-memory solution
`Z* = Σ vⱼkⱼᵀ` (what M2's post-hoc truncation curve measures), n=3000 instances:

| k | frac. instances ALL 8 clear τ=0.9 | τ=0.95 | τ=0.99 |
|---|---|---|---|
| 6 (K−2) | **0.0000** | 0.0000 | 0.0000 |
| 7 (K−1) | **0.0000** | 0.0000 | 0.0000 |
| 8 (K) | 1.0000 | 1.0000 | 1.0000 |

Zero out of 3000 instances show a sub-K-rank matrix clearing **even τ=0.9** on
all K bindings — a complete reversal of round-1's 90%/100% false-pass rates.
The per-item `recovered_frac@τ` (not requiring all-K, matching the actual
`_recovery_stats` metric in `run_task_d.py`) is similarly sharp at τ=0.99:
0% at k=6, 0% at k=7, 100% at k=8 (n=16,000 items, §1 below). **This is the
single most important round-2 result: the orthonormal + τ=0.99 + dense-grid
combination genuinely closes round-1's Finding 1/Item 4, and does so with a
much larger margin than the pre-registration's own arithmetic assumed** (see
Finding A below for why the margin is even bigger than expected).

---

## Finding A — MODERATE: the preregistration's own "(K−1)/K" mental model is analytically wrong (mechanism, not conclusion)

**Round-2 hunt item 1.** The task brief (and by extension whoever wrote
`TASK_D_PREREGISTRATION.md` §6's rationale) assumes rank-(K−1) truncation of
the exact `Z*` "drops exactly one binding," giving per-item mean cos ≈
(K−1)/K = 0.875. **This is not what happens.** I verified it two independent
ways:

1. **Eckart-Young truncation of exact orthonormal `Z*`** (2000 trials × 8
   items): mean cos at k=7 = **0.9271** (not 0.875), and the *shape* of the
   full curve matches `sqrt(k/K)` almost exactly at every k (k=6: 0.848 vs
   predicted 0.866; k=7: 0.927 vs predicted 0.935; round-1 had already derived
   this `sqrt(k/K)` closed form for the *training-objective* optimum under
   `orthogonal=True` — it turns out to also govern the *post-hoc-truncation*
   curve, not just the trained-from-scratch one).
2. **Direct gradient optimization of the true `cosine_loss`** on a free
   rank-k factorization (my own Adam implementation, manually-derived and
   verified gradient, 15 trials × 8 items): mean cos at k=7 = **0.9354 =
   √(7/8) exactly, std = 0.0000** — every single item, every single random
   instance, converges to the *identical* cosine. This is not approximate; it
   is the provable global optimum of the mean-cosine loss under exchangeable
   orthonormal targets (concavity of the coverage objective favors spreading
   an imperfect rank-k budget evenly across all K queries over perfectly
   solving K−1 of them and abandoning the K-th — round-1 already intuited
   this qualitatively; I've now pinned it down to 4 decimal places with zero
   variance).

**Why this doesn't hurt the gate (it helps it):** because `eigh`'s top-k
eigenvector selection, applied to the *exactly degenerate* spectrum of `Z*`
(all K singular values of the correlation matrix equal exactly 1.0 when both
keys and values are exactly orthonormal — verified via `ZZᵀ = Σvᵢvᵢᵀ`, a rank-K
projector), does **not** correspond to "keep K−1 keys / drop 1" — any k-dim
subspace of the tied K-dim eigenspace is a valid top-k choice, and the
"spread evenly" pattern is what both the truncation and the true optimum
independently land on. Per-item `recovered_frac@0.99` at k=K−1 is governed by
this smeared distribution, and is **0.0%** (headline result above) — sharper
than the pre-registration's own (incorrect) 0.875 mental model would predict.

**Action:** correct §6 of the preregistration before writeup — replace the
"drops exactly one binding" framing with the verified "smears evenly,
`mean_cos(k) = √(k/K)`" mechanism. This is a documentation-correctness issue,
not a gate-validity issue (the pass/fail conclusion is unaffected and, if
anything, was under-selling its own margin).

---

## Finding B — MODERATE, NEW: raw `torch.linalg.qr` orthonormal generation is not Haar-uniform — a real, quantified, previously-untested statistical defect in the "fix" itself

**Round-2 hunt item 6 (generator statistics).** `_random_directions`'s
`orthogonal=True` path does `q, _ = torch.linalg.qr(x.transpose(-1,-2))` with
**no sign correction**. This is the textbook non-Haar-uniform QR pitfall
(Mezzadri 2007, arXiv:math-ph/0609050): LAPACK's Householder construction
picks a numerically-motivated, not randomly-motivated, sign for each pivot,
so the resulting orthonormal frame is **not** uniformly distributed over
O(d) — it has a systematic per-column bias.

I confirmed this directly (numpy `linalg.qr`, same LAPACK-backed algorithm,
N=20,000 draws, d=16, K=8): the **mean** of the j-th generated
key/value slot, across draws, is **not zero** — it is a vector of magnitude
**0.17–0.20** (>100σ from zero; sampling noise floor at N=20,000 is ~0.007),
and the bias is **perfectly diagonal**: slot j's mean vector is ≈ −0.19·eⱼ
(concentrated almost entirely in ambient dimension j, deterministic sign,
cross-slot cosine ≈ 0, i.e. noise-level). Applying the standard
sign-correction (`q *= sign(diag(r))`) removes the bias to the sampling-noise
floor (verified: max residual bias entry 0.006 after correction, vs. 0.20
before). This is the confirmed root cause and the confirmed fix.

**Why this matters:** round-1 Finding 8 ("generator statistical shortcuts:
not found — closed") checked *cross-correlation between draws* (keys vs.
values, query-index independence) but never tested the **marginal**
Haar-uniformity of the `orthogonal=True` path itself — this is a genuinely
new gap, not a re-litigation of a closed item. It matters because the
network's linear layers (`in_proj`, `row_out`, etc.) *do* see raw ambient
dimension index (unlike the K-slot axis, which is genuinely permutation-blind
— no positional embedding exists over bindings), so a systematic,
per-ambient-dimension bias is in principle learnable/exploitable as a
population-level "free" partial-credit direction, independent of and on top
of whatever the model achieves via genuine per-instance binding.

**Severity in practice:** not fatal to the current τ=0.99 conclusion. The
bias magnitude (‖bias‖≈0.19) is far short of what's needed to bridge the
`√(7/8)=0.935 → 0.99` gap at k=K−1 (confirmed: recovered_frac@0.99 stays at
0.0% in my simulations, which already reflect this generator as-implemented).
But it is a **confirmed, reproducible, previously-uncaught defect in the very
generator round-1 recommended as the fix**, it violates the "fresh random, no
memorizable population structure" design intent the whole task depends on
(§2, §10's Kohonen/Anderson framing assumes generic/exchangeable targets),
and the fix is a one-line, zero-downside change. **Recommend fixing before
Stage 1** — not because it currently breaks the pass/fail line, but because
leaving a known, quantified statistical defect in the "airtight" generator
undermines the round-2 exercise's own goal, and because it could matter more
at smaller K (where a fixed ~0.19 bias is a larger fraction of the achievable
margin) which the M1 sweep (K∈{1,2,4,8,12,16}) does test.

---

## Finding C — MODERATE: the resolution pre-flight (smoke [6]) validates a proxy, not the actual M2 decision statistic, and is not parametrized by the run it's meant to validate

**Round-2 hunt item 3.** Current smoke[6]:
```python
cos_full = recovery_cosine(...).mean().item()          # mean cosine, not recovered_frac
cos_tr = recovery_cosine(...).mean().item()             # only k=K-1, not the full [K-2,K+1] window
assert cos_tr < cos_full - 0.05
```
Three gaps, none individually fatal, together worth tightening:

1. **Proxy vs. criterion.** The actual M2/M3 CONFIRM criterion (§6) is
   `k* = smallest k with recovered_frac@0.99(k) ≥ 0.9·recovered_frac@0.99(K)`.
   Smoke[6] instead checks whether the *mean* cosine drops by ≥0.05 — a
   different, looser statistic. In this specific config the two happen to
   agree (I verified both: mean-cos drop = 0.073 ≫ 0.05, **and**
   recovered_frac@0.99 drop = 100%→0% ≫ the 90%-of-ceiling bar), so the
   smoke test's conclusion is correct, but it is not formally proving what it
   claims to prove. A configuration could exist where mean cos drops
   comfortably while `recovered_frac@τ` barely moves (e.g. a bimodal
   per-item distribution). **Fix:** have smoke[6] directly compute and
   assert on `recovered_frac@0.99`, matching §6's literal wording.
2. **Threshold is not K-scaled.** The analytically-expected mean-cos drop at
   rank K−1 is `1 − √((K−1)/K)`, which **shrinks** as K grows: 0.065 at K=8,
   but only **0.032 at K=16** — below the fixed `0.05` threshold. Smoke[6]
   hardcodes `K=8` and is never re-run for the other M1-sweep operating
   points (K∈{1,2,4,8,12,16}); if this exact pattern is copy-pasted for a
   different K without adjusting the constant, it would risk a **spurious
   failure** at large K, not a spurious pass. **Fix:** either scale the
   threshold (e.g. `0.5·(1−√((K−1)/K))`) or explicitly restrict the "knee is
   resolvable" claim to the tested K=8 point only, and add the same check for
   K=16 (the boundary case, K=d) before trusting that leg of the M1 sweep.
3. **Not parametrized by `args`.** `smoke(device)` hardcodes
   `orthogonal=True, K=8, d=16` internally and ignores `--gaussian`/`--K`/
   `--d`. The `--gaussian` CLI flag still exists ("weaker knee; dev only")
   — a user could pass `--smoke` (passes, silently always tests the
   orthonormal config) then separately launch a real run with `--gaussian`,
   wrongly believing the smoke gate already validated that regime. **Fix:**
   either have `smoke()` accept `args` and test the actually-requested
   config, or remove the vestigial `--gaussian` flag now that orthogonal=True
   is mandatory for the gate (simplest, lowest-risk option).

---

## Finding D — MODERATE, carried forward from round-1 Finding 3 (STILL-OPEN, not resolved by the round-2 fixes)

**Round-2 hunt items 4/6.** Round-1 flagged that `eigh`'s backward pass is
ill-conditioned when the k-th and (k+1)-th singular values of the encoder's
*raw* (pre-truncation) output are near-tied, and that this risk is highest
"at low k early in training." I attempted to sharpen this by checking whether
the **true optimal** rank-k solution is *forced* into a near-degenerate raw
spectrum (which would make this a structural, not just transient, risk under
the new orthonormal config specifically). Using a free `Z=A·Cᵀ`
parameterization directly optimized on `cosine_loss` (not going through
`truncate_to_rank`, so this only bounds the question, doesn't settle it): the
found solutions have **well-separated** singular values (e.g. at k=7:
11.3, 9.1, 8.0, 7.7, 7.4, 6.1, 5.7 — no tight clustering). This is mild
reassurance that the optimum itself doesn't force degeneracy, but it does
**not** replicate the actual `force_rank_k` training loop (raw full-rank `Z`
→ `eigh(ZZᵀ)` truncation → backprop through the projection every step),
which I could not run without `torch`. **Status: unresolved, unchanged from
round-1 — the round-2 fixes (orthogonal default, τ=0.99, dense grid) address
the *resolution/threshold* question, not this *numerical-stability* question,
which is orthogonal to them.** Recommend round-1's original mitigation
(log σ_k−σ_{k+1} gap and grad-norm at low `force_rank_k` during real
training) unchanged.

---

## Finding E — MODERATE nuance, NEW: M2 and M3 are expected to disagree quantitatively at k=K−1, and a real (non-global-optimum) trained model could land in a thinner-margin regime than the idealized analysis suggests

**Round-2 hunt item 4 (residual shortcuts / optimization noise).** Two
different "how good can rank-(K−1) get" numbers came out of my simulations,
and the gap between them is itself informative:

| method | mechanism | recovered_frac@0.99 at k=K−1 |
|---|---|---|
| Eckart-Young truncation of unconstrained-optimal `Z*` (≈ what **M2** measures: post-hoc SVD of a trained, unconstrained model) | fixed target, truncate after the fact | **38.5%** (n=16,000 items) |
| Free rank-k parameterization, directly optimized on `cosine_loss` (≈ ceiling of what **M3** could reach: train-time-adaptive rank constraint) | encoder adapts *around* the constraint every step | **0.0%** (deterministic, std=0.0000) |

Both comfortably clear the CONFIRM criterion (`< 0.9×ceiling`, i.e. <90%), so
this is **not** a false-pass risk as currently thresholded — but it is a real,
previously-unquantified difference, and it exists precisely because M2
(post-hoc truncation of a matrix that was never optimized *for* that rank)
has per-instance variance that lets some items get "lucky" alignment, while
M3's train-time-adaptive optimum is deterministic and uniform. This is
exactly the design rationale C1 already gives for treating M3 as PRIMARY and
M2 as merely corroborating — my numbers now make that rationale concrete and
quantitative rather than qualitative.

**A sharper, related risk:** the *global* mean-cosine optimum spreads
recovery evenly (0% at τ=0.99, huge margin). But a **different, worse-mean
but still-plausible** allocation exists — "perfectly recover K−1 items,
abandon the K-th" — which gives per-item cosines {1,1,...,1,0}, i.e.
`recovered_frac@0.99 = (K−1)/K = 87.5%` — dangerously close to the 90%
CONFIRM cutoff. My free-parameterization Adam runs reliably found the
smooth/global optimum, not this "axis-aligned" one, from every random
initialization tried — but a real encoder's architecture (the row-reader's
per-row cross-attention, which processes each of the d output rows somewhat
independently) is not obviously immune to converging toward a more
partitioned solution instead of the smooth global one. **Recommend**
pre-registering an explicit check on the *shape* of the per-item cosine
distribution at k=K−1 during real M2/M3 runs (uniform/spread vs.
bimodal/partitioned), not just its mean or thresholded rate — the
preregistration's "always report the full cosine distribution" already
captures the data needed; this just asks that the *interpretation* of that
shape (spread=robust vs. partitioned=fragile-margin) be named as a specific
pre-registered diagnostic rather than left implicit.

---

## Finding F — MINOR: blank-out test (smoke [5]) is substantially fixed, closes the exact regression round-1 named, but still has one narrow residual gap

**Round-2 hunt item 2.** This is a real improvement over round-1, verified
by tracing the actual computation graph semantics:

- **Part 1** (`gk = grad(pred_g.sum(), keys)`, asserted nonzero) is a genuine
  behavioral test — `keys` is a real input to `model.encode()`, so this
  positively confirms bindings actually affect the output (round-1's
  original recommendation #1, achieved via gradient rather than a second
  forward pass — arguably stronger, since gradient-nonzero implies local
  sensitivity, not just "these two particular forward passes differed").
- **Part 2** (`Z_leaf = Zg.detach()...`; `g_leak = grad(pred_leaf.sum(), keys)`
  asserted `None`) genuinely catches the exact regression round-1 warned
  about ("someone refactors `unbind` to read the whole batch dict") — I
  traced the specific failure mode: if `unbind` were changed to read
  `self._last_keys` (a side-effecting cache set during `encode()`) instead of
  its declared arguments, `self._last_keys` would still hold a live,
  differentiable reference to the *same* `keys` tensor used in this test at
  the time `pred_leaf` is computed, so `g_leak` would correctly become
  non-`None` and the assertion would fire. Combined with the
  `inspect.signature(unbind) ⊆ {Z, query_keys}` static check (directly
  implementing round-1's recommendation #2), this closes the "signature
  grows to leak raw bindings" regression class concretely, not just
  architecturally-by-assumption as round-1's version was.
- **Residual gap:** this test only detects **differentiable** leaks. A
  hypothetical leak through a **detached** side channel (e.g. a raw,
  non-differentiable buffer/cache read for its *value*, not its gradient —
  `self._last_keys.detach()`) would produce zero gradient contribution and
  would **not** be caught by part 2, only by the static signature check (and
  only if the leak required an actual extra parameter, not a captured
  `self.*` reference on an already-2-parameter method). Not a live bug given
  the current bare-function `encode`/`unbind` implementation, but round-1's
  original recommendation #1 (compute `Z_a` from batch `b`, `Z_b` from a
  *different* batch, assert `unbind(Z_a,q) ≠ unbind(Z_b,q)`) is a
  complementary, purely value-level check that doesn't currently exist
  alongside the gradient test. **Recommend adding it as defense-in-depth**
  (cheap, a few lines), not because a leak is suspected now.

---

## Item 5 — Is the Stage-1 gate (M1+M2+M3, matrix-only) interpretable without C2?

**Yes, for the question as pre-registered — and this is unchanged by round-2,
already correctly scoped in the current docs.** `H_primary` (§1) is a claim
about the matrix model's *own* rank-recruitment behavior under a hard
bottleneck ("a from-scratch matrix-native model... develops effective rank ≈
K... and its accuracy collapses under rank-k truncation for k<K") — not a
comparative "matrix beats vector" claim. §6's actual gate logic already
reflects this: `PASS if M1 CONFIRM and M3 CONFIRM (M2 corroborating)`; M4 is
explicitly "interpretation input... not a pass/fail gate."

**What a PASS from M1+M2+M3 alone establishes:** that SGD, under a
provably-necessary-rank task and an architecturally-pinned single-matrix
bottleneck with a linear unbind readout, *does* recruit and use rank ≈ K —
resolving the sharpened §1 question of whether the workshop paper's
CODI rank-blindness result was task-specific (ProsQA being rank-1-solvable)
rather than a fundamental gradient-descent pathology.

**What it does NOT establish without C2:** whether this rank-recruitment
behavior is a distinctive property of *matrix-native* structure, versus
something an equally-parameter/state-matched vector representation (HRR or
otherwise) would also exhibit under an analogous bottleneck+exact-recovery
task — the reshape-equivalence confound `ATTACK_baseline_confound.md`
raised. It also says nothing about whether rank *helps real reasoning*
(explicitly out of scope, §9) or whether matrix structure has any
capacity/generalization edge over vectors at matched budget (M4's role).

**Verdict on this question:** descoping C2 does not weaken the PASS/FAIL
decision as literally defined; it narrows what a PASS can be cited to support
in a writeup. That narrowing is already honestly stated in the current docs
(model_v4.py's C2 comment block, run_task_d.py's module docstring) — the only
actionable recommendation is to make sure the eventual Chapter-2 writeup
repeats this scope caveat explicitly rather than let a PASS be read as
"matrix beats vector," which this gate does not test.

---

## Round-1 concern status

| Round-1 item | Status | Note |
|---|---|---|
| `AUDIT_validity` Item 1 — blank-out vacuous | **CLOSED** | Genuinely behavioral now (Finding F); one narrow non-differentiable-leak gap remains, non-blocking |
| `AUDIT_validity` Item 2 — readout pinning | **CLOSED** | Re-confirmed, unchanged |
| `AUDIT_validity` Item 3 — orthogonal default | **CLOSED** | `--orthogonal` is now the CLI default |
| `AUDIT_validity` Item 4 — quantified false-negative risk | **CLOSED** | Re-verified with a stronger test (0/3000 instances clear τ at k<K, all three τ); margin is *larger* than the preregistration's own (incorrect) arithmetic predicted — see Finding A |
| `AUDIT_validity` Item 5 — M2 grid too coarse | **CLOSED** | `evaluate()` defaults to dense `range(1, d+1)` |
| `AUDIT_validity` Item 5 — M3 launcher missing | **STILL-OPEN** | Confirmed: no script in `matrix-thinking/scripts/` or `chapter2/` references `task_d`/`run_task_d`; Stage-1's "~18 short runs" still has no driver script |
| `AUDIT_validity` Item 6 — vector baseline underpowered | **CLOSED (by descoping)** | Explicitly out of Stage-1 scope now, correctly documented in `model_v4.py`/`run_task_d.py` |
| `AUDIT_adversarial` Finding 1 — τ=0.9 false pass | **CLOSED** | Strongly reconfirmed, see headline result |
| `AUDIT_adversarial` Finding 2 — rank-1 shortcut | **CLOSED** | Unchanged, reconfirmed |
| `AUDIT_adversarial` Finding 3 — eigh degenerate-boundary risk | **STILL-OPEN** | See Finding D; round-2 fixes don't address this axis |
| `AUDIT_adversarial` Finding 4 — `n_query<K` weakens bound | **PARTIALLY CLOSED** | The warning fix was applied (`run_task_d.py:256-258`); the recommended `evaluate()` result-dict fix (add `Q`/`queries` key) was **not** applied — `out = {"K":..., "d":..., "model":..., "orthogonal":..., "force_rank_k":...}` still omits `Q` |
| `AUDIT_adversarial` Finding 5 — stale role-embedding docstring | **CLOSED** | `ROLE_KEY`/`ROLE_VALUE`/`ROLE_QUERY` constants no longer present in `task_d.py` |
| `AUDIT_adversarial` Finding 6 — query/target leakage | **CLOSED** | Re-confirmed, unchanged |
| `AUDIT_adversarial` Finding 7 — force_rank_k routing leak | **CLOSED** | Re-confirmed, unchanged; no cross-step state, exact geometric projection |
| `AUDIT_adversarial` Finding 8 — generator statistical shortcuts | **PARTIALLY REOPENED** | Cross-correlation/independence still closed; **new** sub-issue found — marginal Haar-uniformity of `orthogonal=True` was never tested and is violated (Finding B) |

---

## Severity-ranked new/residual findings (round 2)

| # | Finding | Severity | Blocks Stage-1? |
|---|---|---|---|
| B | QR generator not Haar-uniform (~0.19-magnitude diagonal bias per slot) | MODERATE | No (doesn't flip pass/fail), but cheap+recommended fix before running |
| C | Resolution pre-flight validates a proxy, not the literal M2 criterion; threshold not K-scaled; not parametrized by `--gaussian`/other K | MODERATE | No, but strengthen before trusting the pre-flight for K≠8 or `--gaussian` |
| A | Preregistration's "(K−1)/K=0.875" mechanism is analytically wrong (true value √(k/K)) | MODERATE (docs) | No — conclusion unaffected, correct the writeup |
| D | `eigh` near-degenerate-boundary numerical risk (round-1 Finding 3) | MODERATE | STILL-OPEN, unresolved by round-2, recommend monitoring during real training |
| E | M2 vs M3 expected-curve gap (38.5% vs 0% at k=K−1); "axis-aligned" allocation margin-fragility risk | MODERATE (nuance) | No — both clear the CONFIRM bar; pre-register the distribution-shape diagnostic |
| — | M3 sweep launcher still doesn't exist | MINOR/procedural | Blocks *running* Stage-1, not its interpretability |
| — | `evaluate()` omits `Q`/`queries` from result dict | MINOR | No |
| F | Blank-out test doesn't catch non-differentiable leak channels | MINOR | No |

---

## Verdict

**Yes — this is now an interpretable, shortcut-free go/no-go gate, materially
stronger than round-1's version, and the core round-1 concern (τ=0.9 false
pass) is closed with a wide, quantitatively-verified margin (0/3000 instances
clear even τ=0.9 at k<K under the new defaults, vs. 90-100% under the old
ones).** The rank≥K theorem, the pinned linear readout, the exact train-time
`force_rank_k` projection, and the encode/decode boundary are all sound and
re-confirmed. The blank-out test now does real work. The dense grid removes
the coarse-grid blind spot. M4's descoping is honest and doesn't weaken the
pre-registered PASS/FAIL logic for the question it's actually meant to
answer.

**Not fully airtight, and not yet actually runnable as claimed:** (1) the
orthonormal generator itself has a confirmed, previously-untested statistical
defect (Finding B) that should be patched — cheap, and leaving it in
undercuts the round-2 exercise's own standard for "no exploitable
distributional shortcut"; (2) the resolution pre-flight proves a proxy
statistic, not the literal §6 decision rule, and silently only covers the
K=8/orthogonal=True point (Finding C); (3) round-1's numerical-stability
concern about `eigh`'s backward near a near-degenerate spectrum remains
completely unaddressed by the round-2 fixes and unverified either way
(Finding D) — this needs a `torch`-based check, not a numpy proxy; (4) no
launcher script exists yet for the pre-registered Stage-1 sweep, so
"ready to run" is true for the *design* but not yet for the *harness*.

**Recommendation:** apply the QR sign-correction (Finding B, ~3 lines),
strengthen smoke[6] to assert the literal `recovered_frac@0.99` criterion and
either scale its threshold or explicitly limit its claim to K=8 (Finding C),
correct §6's "(K−1)/K" arithmetic to the verified `√(k/K)` form (Finding A),
and write the M3 launcher before declaring Stage-1 "ready." None of these are
multi-day fixes. The `eigh` numerical-stability question (Finding D) should
be checked with real `torch` during the first Stage-1 dry run (log the
σ_k−σ_{k+1} gap and gradient norm at `force_rank_k∈{1,2}`) rather than
blocking on it further — it is a monitoring item, not a redesign item, at
this point.
