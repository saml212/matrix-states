# ADVERSARIAL AUDIT — Task D Rank Gate (`chapter2/`)

**Role:** adversarial auditor. Goal: find a way the trained model gets high
recovery cosine WITHOUT genuinely using rank ≥ K, i.e. a way the gate is
misleading (false pass or false fail). Config audited: `d=16, K=8, τ=0.9`
(the pre-registered operating point, §6 of `TASK_D_PREREGISTRATION.md`).

**Method note.** No `torch` was available in the sandbox. All quantitative
claims below were verified with a faithful **numpy re-implementation** of
the exact generator (`task_d.sample_batch`'s near-orthogonal-Gaussian draw)
and the exact rank projection (`rank_utils.truncate_to_rank`'s Eckart-Young
top-singular-subspace projection — mathematically identical operation,
reimplemented via `svd` instead of `eigh(ZZ^T)`, same output). Scripts are
in the scratchpad; numbers are reproducible and I recommend re-running the
equivalent check with the real torch code as a cheap pre-flight addition to
`run_task_d.py --smoke` (concrete patch suggested in Fix 1 below). One
environment artifact: numpy-on-Accelerate (Apple Silicon) throws spurious
"invalid value encountered in matmul" `RuntimeWarning`s on ordinary matmuls
with no actual NaN/Inf in the output (confirmed by explicit `isfinite`
checks) — a known Accelerate/BLAS quirk, unrelated to the codebase under
audit, not a real numerical bug.

---

## FINDING 1 — τ=0.9 cosine threshold cannot distinguish "needs rank K" from "rank K−1 (or K−2) already suffices"

**Severity: MAJOR, functionally FATAL to the gate's decisiveness as configured. Causes a FALSE PASS.**

**Hunt items covered:** #3 (cosine-metric gaming — construct the cheat) and
the resolution half of #4/#5.

### The exploit

The theorem (§3) proves `rank(Z) ≥ K` is necessary for **exact** (cosine
= 1) recovery. The *operational* test relaxes this to `cos > 0.9`. Because
targets are near-orthogonal Gaussian and cosine ignores magnitude, a matrix
that is **one or two ranks short of K** already reconstructs each query's
*direction* almost correctly — dropping the smallest singular value(s) of
the exact solution perturbs each individual predicted direction only
slightly, because the perturbation is spread across K near-orthogonal
directions rather than concentrated on any one query.

I constructed two independent estimates of "how good can rank-r do," both
using the exact pre-registered geometry (d=16, K=8, i.i.d. Gaussian
unit vectors):

**(B) Eckart-Young truncation of the exact least-squares solution Z\*** (the
literal operation `rank_utils.truncate_to_rank` performs), averaged over
3000 fresh instances — this is exactly what M2's post-hoc truncation curve
measures:

| rank r | mean cos | **per-item recovered_frac @ τ=0.9** (= M2/M3's reported metric) |
|---|---|---|
| 1 | 0.392 | 0.001 |
| 2 | 0.597 | 0.010 |
| 3 | 0.735 | 0.051 |
| 4 | 0.833 | 0.215 |
| 5 | 0.904 | 0.597 |
| **6 (K−2)** | 0.953 | **0.924** |
| **7 (K−1)** | 0.984 | **0.998** |
| 8 (K) | 1.000 | 1.000 |

**(C) Direct gradient optimization of a free rank-r factorized `Z=U@V`
under the actual training objective** (`cosine_loss` from `run_task_d.py`,
per-instance overfit — a ceiling on what ANY encoder, even one with
unlimited expressivity, could do at rank r, i.e. the tightest bound on the
cheat): at r=6 this reaches **93.3%** of instances with *all 8* queries
simultaneously above τ=0.9 (vs. 58.6% for the naive Eckart-Young estimate
at the same rank) — direct optimization of the real cosine objective is
*more* forgiving of low rank than the L2-optimal truncation, not less.

**Chance-level control:** `P(cos > 0.9)` for two independent random unit
vectors in R^16 is **0 / 2,000,000** in a Monte Carlo run — τ=0.9 is not
trivially gameable by a rank-1 `u⊗v0` shortcut (see Finding 4, closed). The
problem is specifically the **K−1 / K−2 near-miss regime**, not chance.

**Does raising τ fix it?** Partially, and not enough on its own:

| τ | recov@r=5 | recov@r=6 | recov@r=7 | recov@r=8 |
|---|---|---|---|---|
| 0.90 | 0.597 | 0.924 | 0.998 | 1.000 |
| 0.95 | 0.193 | 0.603 | 0.943 | 1.000 |
| 0.99 | 0.008 | 0.100 | 0.491 | 1.000 |
| 0.999 | 0.0001 | 0.009 | 0.150 | 1.000 |

Even at τ=0.99, rank K−1 recovers ~49% per-item — not "near-chance." Only
at τ≈0.9999 does K−1 drop toward chance-adjacent levels.

### Why this breaks the gate as written

Pre-registered M3 CONFIRM text: *"best eval accuracy is near-chance for
k < K ... with a monotone step at k = K (±1)."* This literally requires
near-chance accuracy at **k = K−1**. My numbers show k = K−1 gives ~100%
per-item recovery at the default τ=0.9 — the **opposite** of near-chance.
But the actual test grids never probe this:

- `run_task_d.py`'s `evaluate()` default `rank_ks=(1, 2, 4, 8, 16)` — skips
  5, 6, 7.
- Preregistration §7 Stage 1: `force-rank-k at K=8, k ∈ {1, 4, 8}` — even
  coarser, skips 2, 5, 6, 7.

So the code as configured will show "near-chance at k=4 (21%), ceiling at
k=8 (100%)" and this will read as a clean monotone step — satisfying M3
CONFIRM — while the fine-grained region that actually falsifies the strict
reading of CONFIRM (k=6 at 92%, k=7 at 99.8%) is never measured. The report
would legitimately be able to say "SGD discovers rank ≈ K and force-rank-k
< K craters" while the true minimal sufficient rank under this τ is K−1 or
even K−2, not K. That is a **false pass** of H_primary as stated, not
because the theorem is wrong, but because the operational metric doesn't
have the resolution the theorem's proof (which is about *exact* recovery)
assumed.

A second-order consequence: if the trained (not oracle) encoder settles at
*true* rank K−1 or K−2 (plausible — SGD under this soft metric has no
pressure to close the last small singular value), M1's effective-rank CONFIRM
band `[0.7K, 1.3K] = [5.6, 10.4]` **also** swallows K−1=7 and K−2≈6.4
without flagging anything unusual — both of the sharp instruments (M1's
band, M2/M3's coarse grid) have blind spots tuned exactly to miss this
failure mode.

### Fix

1. **Widen the rank grids to straddle K tightly.** Change
   `run_task_d.py`'s `evaluate()` default `rank_ks=(1, 2, 4, 8, 16)` →
   include `{K-2, K-1, K, K+1}` always (e.g. `(1, 2, 4, 6, 7, 8, 9, 16)` at
   K=8), and change §7's Stage 1/Stage 2 force-rank-k sweep from
   `{1, 4, 8}` to include `6, 7, 9` at minimum. This is a config-only fix,
   not a redesign.
2. **Pre-register a second, stricter τ for the knee test specifically**
   (e.g. τ=0.999, per the table above) alongside the headline τ=0.9, and
   require the knee `k*` be reported for both. If `k*` differs materially
   between τ=0.9 and τ=0.999, report that gap explicitly rather than
   picking whichever τ gives the cleaner story.
3. **Add a pre-flight "resolution check" to `run_task_d.py --smoke`**: for
   a batch of instances, compute the exact least-squares `Z*` (torch
   equivalent of `numpy.linalg.lstsq`), Eckart-Young-truncate to `K-1` and
   `K-2` via the real `truncate_to_rank`, and print/assert the per-item
   recovered_frac at τ=0.9. This turns "is the metric sharp enough at this
   (d,K,τ)" into a checked, printed number before any GPU time is spent,
   the same spirit as the existing blank-out test.
4. When writing up results, report whether the "clean step at K" survives
   the finer grid — if k=K−1 or K−2 already clear ~0.9× ceiling, say so
   plainly rather than rounding to "rank K required."

---

## FINDING 2 — Rank-1 `u⊗v0` master shortcut is genuinely closed

**Severity: none (informational — confirms a control works). Not a false pass or false fail.**

**Hunt item covered:** #2.

`MatrixMemoryModel.unbind` (`model_v4.py:81-87`) is a bare
`torch.einsum("bij,bqj->bqi", Z, query_keys)` — no learned map is applied to
`query_keys` anywhere in the matrix arm. Given this, `Z = u v0^T` (rank 1)
produces `Z k_q = u (v0 · k_q)` — a **scalar multiple of the single fixed
direction `u` for every query**, regardless of which key is asked. Since
targets are near-orthogonal random directions, the Monte Carlo control above
(`P(cos > 0.9)` for a random direction vs. a fixed target ≈ 0/2,000,000)
shows that even matching **one** of the K targets this way is astronomically
unlikely by chance, let alone all K simultaneously with a fixed `u`. This
closes §0 of `ATTACK_task_shortcuts.md` for this specific readout: the
architecturally-pinned `Z @ key` (no MLP, no learned query transform)
genuinely blocks the free-side rank-1 shortcut, exactly as the
pre-registration claims. No fix needed; flagging so it isn't re-litigated.

---

## FINDING 3 — `truncate_to_rank`'s `eigh` backward near-degenerate-spectrum risk is plausible but not independently verified here

**Severity: MINOR–MODERATE (unresolved — recommend a monitoring addition, not a blocking fix). Could produce a spurious HARD FALSIFY of M3 at low k (numerical, not "rank-averse gradient").**

**Hunt item covered:** #5 (the numerical-stability half; the "leaks around
the constraint" half is closed — see Finding 6).

`rank_utils.py`'s docstring claims *"eigh backward is numerically stable
even when singular values coincide... verified NaN-free on a constructed
[3,1,1,0] spectrum."* This is narrower than it sounds: `truncate_to_rank`'s
actual output, `Zk = U_k (U_k^T Z)`, is invariant to the internal
orthogonal-basis gauge freedom within a *tied* eigenvalue cluster **as long
as `k` does not split the cluster** (all tied eigenvectors fully in or fully
out of the top-k). The `[3,1,1,0]` test case has its tie in the excluded
tail (values 1,1 both below the cut for k≤2) — a favorable case. The
genuinely dangerous case is when `k` lands **exactly at the boundary of a
near-tied pair** (the k-th and (k+1)-th singular values of `Z_raw` are
close but not equal) — there the top-k *subspace itself* is ill-conditioned
as a function of Z (a small perturbation can swap membership), and this is
a property of the Eckart-Young projection operator itself, not an
eigh-vs-SVD implementation detail — any correct autodiff of a true
sensitivity will show large/noisy gradients there. This is most likely to
occur at **low k** (k=1, k=2) early in training, when the encoder's raw
output is close to isotropic and nothing in the architecture forces
singular-value separation.

I could not install `torch` in this sandboxed environment to directly
measure `torch.linalg.eigh` backward gradient norms near a controlled
near-tied spectrum (attempted; a full torch install was impractical within
the audit's scope). This finding is therefore **analytical, not empirically
confirmed** — flagged as a risk to monitor, not a proven bug.

### Fix

- During `force_rank_k` training (especially k=1, 2), log the gap between
  the k-th and (k+1)-th singular values of `Z_raw` (pre-truncation) and the
  gradient norm at each step. If k=1 training shows anomalous loss spikes
  or unusually high seed-to-seed variance compared to k=4/8, check whether
  it correlates with small singular-value gaps before concluding "rank-1
  training genuinely cannot recover K=8 bindings" — rule out optimization
  noise before writing up a HARD FALSIFY.
- If this shows up as a real problem, the existing NaN check in `train()`
  (`run_task_d.py:103-105`) catches NaN but not "large but finite" gradient
  spikes from a near-boundary crossing — consider adding a gradient-norm
  ceiling/skip-step as a secondary guard, or switching to a soft nuclear-norm
  penalty as a cross-check arm if hard per-step SVD projection proves noisy.

---

## FINDING 4 — `n_query < K` silently weakens the theorem to `rank ≥ Q`, not `rank ≥ K`

**Severity: MINOR (currently inert at the default operating point; a footgun for future runs).**

**Hunt item covered:** #1 (leakage/scoping), #6 (generator config surface).

The proof in §3 requires exact recovery of **all queried** bindings with
linearly-independent target vectors; the actual necessary condition is
`rank(Z) ≥ Q` where `Q = cfg.queries`, not `rank(Z) ≥ K`. `TaskDConfig`
defaults `n_query=None → queries=K`, which matches the pre-registered
K=8 operating point exactly, so this is **not live** at the current
config. But nothing in `run_task_d.py` enforces `Q=K` — a future sweep run
with e.g. `--K 8 --n-query 4` would silently test (and could report results
against) a `rank ≥ 4` requirement while being logged/labeled as a "K=8"
experiment, since `result["K"]` is recorded but the actually-binding `Q` is
not surfaced in `evaluate()`'s output dict at all.

### Fix

Add `Q` (`cfg.queries`) to the `evaluate()` result dict alongside `K`, and
add a printed warning (mirroring the existing `K > d` warning in `main()`)
when `n_query is not None and n_query < K`, noting the weakened bound
explicitly.

---

## FINDING 5 — Stale docstring: claimed role embeddings don't exist in `model_v4.py`

**Severity: MINOR (documentation only, not a leak).**

**Hunt item covered:** #1 (sanity-checking the encoder's actual inputs).

`task_d.py`'s module docstring and the `ROLE_KEY`/`ROLE_VALUE`/`ROLE_QUERY`
constants (`task_d.py:19-24`) describe "The model adds a learned role
embedding per position," but `BindingEncoder` (`model_v4.py:25-63`) never
references these constants and has no role-embedding parameter — key vs.
value role is communicated purely by concatenation order in
`self.in_proj(torch.cat([keys, values], dim=-1))`. Functionally this is
fine (not a leak — keys/values are still only reaching the encoder through
the two fixed argument channels), but it's a genuine mismatch between the
documented architecture and the implemented one that could mislead a
reviewer checking "how does the encoder know which vector is a key vs a
value." Recommend either wiring up the role embedding as documented or
deleting the unused constants and correcting the docstring.

---

## FINDING 6 — Query/target leakage into `Z`: not found (closed)

**Severity: none (informational).**

**Hunt item covered:** #1, primary ask.

Traced every call site: `MatrixMemoryModel.forward` →
`self.encode(batch["keys"], batch["values"], force_rank_k=...)` →
`BindingEncoder.forward(self, keys, values)`. Neither `query_idx`,
`query_keys`, nor `targets` is passed into `encode()`/`BindingEncoder` at
any point in `run_task_d.py`'s `train()` or `evaluate()`. The
`HRRVectorMemory` baseline (M4 arm) has the same property: its
bundle/encode step (`kk = enc_k(keys); vv = enc_v(values); s =
circconv(kk,vv).sum(1)`) only touches `keys`/`values`; `query_keys` is only
read afterward, at the unbind step, alongside the already-frozen `s`. No
shared mutable state, buffers, or module-level caches exist that could leak
across the encode/decode boundary. This is also empirically checked at
runtime by the blank-out test in `smoke()` (`run_task_d.py:157-169`), which
corrupts `keys`/`values` after `Z` is computed and asserts bit-identical
decode output — a real, executable check, not just an architectural
argument. Confirmed sound.

---

## FINDING 7 — `force_rank_k` gradient routing around the constraint: not found (closed)

**Severity: none (informational).**

**Hunt item covered:** #5, the "leaks around the constraint" half.

Two independent reasons this can't happen as implemented:

1. **No cross-step state.** `train()` (`run_task_d.py:91-111`) draws a
   **fresh** random batch every step via `sample_batch` and computes `Z`
   fully from scratch each forward pass — there is no persistent buffer,
   RNN-style hidden state, or momentum term on `Z` itself that could carry
   information between steps. "Routing information around the constraint
   across steps" has no channel to travel through; each step's `force_rank_k`
   projection is independently and fully enforced on that step's `Z`.
2. **The projection is an exact geometric operation, not a soft penalty.**
   `truncate_to_rank` (`rank_utils.py:21-39`) computes `Zk = U_k(U_k^T Z)`
   via `eigh(ZZ^T)` — this is mathematically guaranteed rank ≤ k for
   *any* input `Z`, regardless of what the encoder's weights are doing.
   There is no way for gradient descent to make this projection "leak"
   higher rank through to the decoder; the decoder only ever sees the
   already-projected `Zk`. This is verified at runtime too: `smoke()`
   step [4] asserts `effective_rank(Zk).mean() <= k + 1e-2` after a real
   forward+backward pass.

The adjacent, genuinely open question is **not** "does information leak
around the constraint" (closed) but "is the gradient *through* the
constraint numerically well-behaved at low k" — that's Finding 3, a
different and softer concern.

---

## FINDING 8 — Generator statistical shortcuts: not found (closed)

**Severity: none (informational).**

**Hunt item covered:** #6.

`sample_batch` (`task_d.py:65-98`) draws `keys` and `values` as two
**sequential, independent** calls to `_random_directions` on the same
`torch.Generator` stream. Sequential draws from a PRNG stream are, by
construction, statistically independent samples from the target
distribution (that is the definition of a PRNG) — there is no shared
tensor, no correlated noise source, and no reuse of the keys' random draw
in constructing values. Query-index selection (`noise = torch.rand(B,K,
gen); perm = noise.argsort(...)`) draws yet another independent stream and
is uniform over which of the K bindings get queried — unrelated to key or
value content, so there's no bias toward "easy" or highly-aligned bindings
being preferentially queried. `orthogonal=True` mode (QR-based) applies the
same construction independently to keys and values and introduces no
cross-correlation either. No exploitable distributional shortcut found.

---

## Ranked Summary

| # | Finding | Severity | Effect if unfixed |
|---|---|---|---|
| 1 | τ=0.9 + coarse rank grid can't distinguish rank K from K−1/K−2 | **MAJOR / functionally FATAL to decisiveness** | **FALSE PASS** — a clean-looking "CONFIRM at K" that a finer grid/tighter τ would show is really "K−1 or K−2 already ~ceiling" |
| 3 | `eigh` backward near-degenerate-spectrum risk at low force_rank_k | MINOR–MODERATE (unverified, analytical) | possible spurious **HARD FALSIFY** at k=1/2 from optimization noise misread as "rank-averse gradient" |
| 4 | `n_query < K` silently weakens the bound to rank≥Q | MINOR (inert at default config) | future footgun, mislabeled results if config drifts |
| 5 | Stale role-embedding docstring vs. actual architecture | MINOR (docs only) | reviewer confusion, no functional risk |
| 2, 6, 7, 8 | Rank-1 shortcut, query/target leakage, force_rank_k routing, generator correlation | none — all closed | — |

---

## Verdict

**The code closes the master shortcut (`Z = u⊗v0`) and the encode→decode
boundary is genuinely leak-free** — Findings 2, 6, 7, 8 are all clean, and
2/6 are backed by both architectural inspection and an executable
runtime test (blank-out) or a direct numerical control (chance-level
Monte Carlo). The provable `rank(Z) ≥ K` theorem is correctly reflected in
the architecture (fixed linear `Z @ key` readout, no learned map on the
query, exact per-step rank projection for the causal test).

**But there is a residual cheat that must be fixed before running Stage
1, and it is quantified, not speculative: the pre-registered τ=0.9
threshold combined with the current rank grids (`{1,2,4,8,16}` at eval,
`{1,4,8}` at train) cannot tell "genuinely needs rank K" apart from "rank
K−1 or K−2 already clears the bar" (numeric estimate: ~92-99.8% per-item
recovered at K−2/K−1 vs. 100% at K, against ~0% chance level).** This is
exactly the failure mode the pre-registration was designed to rule out
(the CODI rank-blindness result), reintroduced one level down by the
choice of operational threshold and test resolution rather than by the
task design itself. Do not run Stage 1 before applying Fix 1 (widen the
rank grids to include K−2, K−1, K+1; add a stricter secondary τ for the
knee test; add the resolution pre-flight check to `--smoke`) — otherwise a
"PASS" result from this gate cannot be trusted to mean what H_primary
claims it means.
