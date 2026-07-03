# ATTACK_RANK_AWARE — Rank-Aware Patch Audit

Patch reviewed: `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/scripts/run_matrix_codi.py`
Smoke test reviewed: `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/scripts/_smoke_rank_aware.py`
Smoke test status: PASSES on a single random init (CPU, fp32).

This file enumerates 10 specific attacks. For each: failure mode, line citation,
verdict (LANDS / DOES NOT LAND / NEEDS FIX), and proposed fix.

---

## Attack 1 — Default-path byte-identity

**Lines:** `run_matrix_codi.py:599-603`, `:618`, `:664`, `:1093-1102`, `:1168`, `:1179`.

**Failure mode:** The agent's report claimed defaults give identical training to
pre-patch. The new code adds (a) an unconditional empty list `Z_list_grad`
inside `student_forward`, (b) an unconditional `Z_list_grad.append(Z_t)` for
matrix-CODI, (c) a new dict key in the student return, (d) cfg.get probes in
`compute_codi_loss`, (e) `L_rank = torch.zeros(1, device=device, dtype=L_total.dtype)`
allocated EVERY step regardless of flags, (f) a new `"L_rank"` key in the parts
dict.

None of these consume RNG. None affect the value of `L_total`. The Z_t
references retained in `Z_list_grad` are already kept alive by the autograd
graph through `running_embeds → out_full`, so no extra activation memory.
`torch.zeros(1)` is a per-step 4-byte allocation; in fp16/bf16 it is 2 bytes.

**Verdict: DOES NOT LAND** for training-trajectory equivalence. The optimizer
sees identical gradients. There is a microscopic per-step allocation overhead
(`torch.zeros(1)`) that does not affect bit-for-bit numeric equivalence.

**Proposed fix (optional, defensive):** guard the `L_rank` placeholder allocation
behind `_need_rank_loss`, and skip populating `Z_list_grad` when
`_need_rank_loss is False AND _force_rank_k is None`. This makes the
default path provably zero-overhead and easier to audit:

```python
# in student_forward, replace line 603:
Z_list_grad = [] if (self.use_matrix_bottleneck and need_grad_Z) else None
# need_grad_Z must be threaded as a kwarg or implicitly True during training.
```

---

## Attack 2 — SVD backward at near-coincident singular values  *** CRITICAL ***

**Lines:** `run_matrix_codi.py:312` (force-rank path) and `:374` (truncate_to_rank).

**Failure mode:** Both new sites use `torch.linalg.svd(full_matrices=False)`.
PyTorch's autograd through full SVD has a documented instability: the U/V
gradient contains a term proportional to `1/(σ_i² − σ_j²)`. When two singular
values coincide, the term is `1/0` → NaN. When they are merely close, the
term explodes.

**Empirical reproduction (fp32, CPU):**

| input spectrum                | k | grad max abs | NaN? |
|-------------------------------|---|--------------|------|
| `[2.0, 2.0, 0.5, 0.1]` exact  | 2 | NaN          | yes  |
| `[2.0, 2.0, 0.5, 0.1]` exact  | 1 | NaN          | yes  |
| `[2.0, 2.0−1e-6, 0.5, 0.1]`   | 1 | 1.2e+5       | no   |
| `[2.0, 2.0−1e-12, 0.5, 0.1]`  | 1 | NaN          | yes  |

The smoke test passed only because random init has a 0% chance of producing
exactly-coincident σ at d=4. During real training:
1. `--force-rank-during-training k`: k cuts inside the spectrum at every
   forward step. As training drives Z toward the rank-k subspace, σ_{k-1} and
   σ_k will frequently approach each other near convergence — exactly when the
   1/(σ²−σ²) blow-up activates.
2. `--rank-loss entropy`: `compute_rank_loss` only uses `svdvals` (no U, V),
   whose backward IS stable through eigh of `Z @ Zᵀ`. This path is safe.
3. `--rank-loss nuclear`: `svdvals` only — also safe.

The danger is concentrated in `force_rank_during_training`, with secondary
exposure in `truncate_to_rank` (only used in eval-time rank projection, less
critical).

**Verdict: LANDS HARD on `--force-rank-during-training` only.** This is the
most severe finding. A long training run with this flag has a non-trivial
probability of NaN gradient followed by silent training divergence (caught by
grad clip + adam epsilon as inflated parameter updates rather than a hard
crash). Could explain a "rank=k task fails to learn even with rank-k forced"
result that is actually an optimization artifact.

**Proposed fix:** replace full-SVD-based truncation with a numerically stable
projection. Two acceptable options:

A. **eigh-based truncation** of the symmetric `Z Zᵀ`:
```python
def truncate_to_rank(Z, k):
    # Z: (B, d, d). Eigendecompose Z Zᵀ (Hermitian, stable backward).
    # Top-k eigenvectors of Z Zᵀ span the column space of the rank-k truncation.
    # P_k = U_k U_kᵀ where U_k are top-k eigvecs of Z Zᵀ.  Z_k = P_k Z.
    Zf = Z.float()
    A = Zf @ Zf.transpose(-1, -2)              # (B, d, d), Hermitian PSD
    eigvals, eigvecs = torch.linalg.eigh(A)    # ascending order
    # Take the LAST k columns (largest eigenvalues).
    U_k = eigvecs[..., :, -k:]                 # (B, d, k)
    # Z_k = U_k U_kᵀ Z
    Zk = U_k @ (U_k.transpose(-1, -2) @ Zf)
    return Zk.to(Z.dtype)
```
`torch.linalg.eigh` backward is stable as long as eigenvalue gaps are positive.
Eigenvalues of `Z Zᵀ` are `σᵢ²`; the same coincidence still produces a small gap,
but eigh's gradient pathology is `1/(λ_i − λ_j)` (gap, not gap-of-squares),
which is strictly larger than `1/(σ_i² − σ_j²)` near the boundary, mitigating
the blow-up by a factor of `(σ_i + σ_j)`.

B. **add a small spectral perturbation before SVD** (less principled): mix
`Zf + ε · I` with `ε = 1e-4 · ‖Z‖_F / d` to break degeneracies. Cheap and
empirically robust but biases the rank.

Option A is preferred. Add a smoke test that drives Z toward a degenerate
spectrum (e.g. `Z = u uᵀ + ε · v vᵀ` with ε small) and asserts NaN-free
gradients.

---

## Attack 3 — Dtype handling under bf16 / autocast

**Lines:** `run_matrix_codi.py:309-317` (force-rank path), `:401-426` (rank loss).

**Failure mode:**
- The force-rank block does `Zf = Z.float()` then SVD then `Z = Zf_trunc.to(orig_dtype)`.
  The cast back to bf16 is fine for stability, BUT the `.float()` at line 311
  is NOT wrapped in `torch.autocast("cuda", enabled=False)`. Compare with the
  existing `rank_project_k` block at line 322 which DOES wrap with `with torch.autocast(...)`
  and is annotated as "Fix SERIOUS-13". The new force-rank block re-introduces
  the SERIOUS-13 condition.
- Same omission in `compute_rank_loss` (lines 408, 414) — `Z.float()` inside
  what may be an autocast region. With autocast enabled, the matmuls inside
  `linalg.svd`'s forward and the entropy computation can be silently re-cast
  back to bf16 mid-graph, depending on the op.
- Returned gradients: with bf16 model, `Z.grad` comes back in bf16 (verified
  empirically). Precision loss in the rank-loss gradient is real but probably
  acceptable given the loss is auxiliary and small-coefficient.

**Verdict: LANDS (partial).** Will not crash on H100 with bf16 in most cases
but matches the exact pattern that "Fix SERIOUS-13" was added to fix
elsewhere, so it inherits the same stability risk on certain CUDA driver
versions.

**Proposed fix:**
```python
# in MatrixBottleneck.forward, force_rank block:
if force_rank_k is not None and force_rank_k > 0:
    orig_dtype = Z.dtype
    with torch.autocast(Z.device.type, enabled=False):
        Zf = Z.float()
        U, S, Vh = torch.linalg.svd(Zf, full_matrices=False)
        k_clamped = min(force_rank_k, S.shape[-1])
        Zf_trunc = (U[..., :, :k_clamped] * S[..., :k_clamped].unsqueeze(-2)) \
                   @ Vh[..., :k_clamped, :]
    Z = Zf_trunc.to(orig_dtype)
```
Same wrapping in `compute_rank_loss` around the `svdvals` call:
```python
with torch.autocast(Z.device.type, enabled=False):
    sigma = torch.linalg.svdvals(Z.float())
```

---

## Attack 4 — Entropy-loss sign

**Lines:** `run_matrix_codi.py:416-418`.

**Failure mode:** `H = -Σ p̃·log(p̃)`. Code returns `-H.mean()` per position.
`compute_codi_loss` then does `L_total = L_total + _rank_lambda * L_rank`.
So total contribution is `λ · (−H̄)`. Optimizer minimizes L_total →
minimizes `−H̄` → maximizes `H̄`. Sign is correct.

**Empirical verification:** one descent step on `L = -H.mean()` with random
init, `H` increased from 1.8411 to 1.8455.

**Verdict: DOES NOT LAND.**

---

## Attack 5 — Nuclear-loss sign

**Lines:** `run_matrix_codi.py:419-423`.

**Failure mode:** Code returns `-nuc.mean()`; total contribution `λ · (−|Z|*)`.
Optimizer minimizes → maximizes `|Z|*`. Sign is correct.

**Empirical verification:** one descent step on `L = -|Z|*.mean()` increased
`|Z|*` from 18.16 to 19.16 as expected.

**Verdict: DOES NOT LAND.**

NOTE (out-of-scope warning, not a bug): `|Z|*` does not control rank spread; it
primarily inflates σ_1. The COMBINED_PLAN already CUT the nuclear-norm column
from the experiment matrix. Including it in the CLI is fine for ablations but
do not run it as a primary condition.

---

## Attack 6 — Per-position averaging

**Lines:** `run_matrix_codi.py:418, 423, 426`.

**Failure mode:** code computes `(1/n_latents) · Σ_t [(1/B) · Σ_b (−H_{t,b})]`.
This equals `(1/(n_latents · B)) · Σ_{t,b}(−H_{t,b})`. Verified numerically
identical to a flat-average reference. Gradient magnitude scales as
`1/(n_latents · B)`, which is the desired per-sample average.

**Verdict: DOES NOT LAND.**

---

## Attack 7 — `Z_list_grad` lifetime / memory leak into default path

**Lines:** `run_matrix_codi.py:603, 618, 664, 1170`.

**Failure mode:** `Z_list_grad` is populated unconditionally during matrix-CODI
training even when no rank loss is requested. The Z_t tensors are appended to
this list, which is returned in the student dict, which is held by
`compute_codi_loss` until function exit. Concern: does this prevent autograd
from freeing intermediate Z buffers earlier than before?

**Analysis:** Z_t is already alive in the autograd graph because h_next
(produced from Z_t) feeds into running_embeds → out_full, whose graph is
needed for backward. So Z_list_grad adds a Python reference but no NEW
GPU memory. At B=16, d=16, n_latents=6, even doubling would be 24KB. Negligible.

There IS a defensive concern: `generate_answer` calls student_forward inside
`@torch.no_grad()`. Z_list_grad still gets populated but contains tensors
without grad_fn. No memory leak (no_grad path), no functional issue.

**Verdict: DOES NOT LAND** in practice. **NEEDS FIX (cosmetic):** when
rank loss is OFF, do not populate Z_list_grad — purely for code clarity and
to make the default-path identity claim defensible. See Attack 1 fix.

---

## Attack 8 — Force-rank gradient correctness

**Lines:** `run_matrix_codi.py:309-317`.

**Failure mode (claimed):** does the gradient correctly accumulate the rank-k
constraint back to w_up and the thinker?

**Empirical verification:** a minimal model with `w_up`, force_rank_k=2,
loss `Z.sum()`. Gradients are non-NaN, finite, and differ from the unconstrained
gradient (as expected — high-σ subspace is projected out).

**Verdict: DOES NOT LAND on random init.** However, it INHERITS the Attack 2
defect: under near-coincident singular values during training, gradients become
NaN. Mitigated by Attack 2's proposed eigh-based fix.

---

## Attack 9 — Parts dict shape stability

**Lines:** `run_matrix_codi.py:1175-1180`.

**Failure mode:** consumers grep `parts['L_teacher' | 'L_student' | 'L_kd' |
'L_rank']`. The new `L_rank` key is ALWAYS present (set to zeros on default
path). Reverse direction: removing `L_rank` from parts is not done.

**Verdict: DOES NOT LAND.** All pre-existing keys still present with same
shape and dtype. New key is additive and harmless.

---

## Attack 10 — Argparse defaults

**Lines:** `run_matrix_codi.py:2381-2410`, `:2437-2442`, CONFIG dict
`run_matrix_codi.py:71-123` (no rank_loss/rank_lambda/force_rank_during_training keys).

**Failure mode:** the agent's report stated:
- `--rank-loss` default `'none'`
- `--rank-lambda` default `0.0`
- `--force-rank-during-training` default `0`

**Actual code:** all three argparse `default=None`. The CONFIG dict does NOT
contain these keys at all. Effective defaults come from `cfg.get("rank_loss", "none")`,
`cfg.get("rank_lambda", 0.0)`, `cfg.get("force_rank_during_training", 0)`
inside `compute_codi_loss` and the logger.

**Functional equivalence:** YES. When the user passes none of the flags, the
behaviour matches the agent's claimed defaults.

**Verdict: DOES NOT LAND functionally, NEEDS FIX (cosmetic) for spec compliance.**
The mismatch between agent report and code is small but a future maintainer
inspecting CONFIG will not see these keys exist. Either:

A. set the argparse defaults explicitly:
```python
parser.add_argument("--rank-loss", default="none", choices=[...])
parser.add_argument("--rank-lambda", default=0.0)
parser.add_argument("--force-rank-during-training", default=0)
# and unconditionally set cfg[...] = args.X
```

B. add the keys to CONFIG with their defaults so they are discoverable.

(B) is preferred — it documents the new flags as part of the canonical config.

---

## Additional note — input validation

`--force-rank-during-training` accepts negative ints silently. `min(force_rank_k, d)`
with negative input yields negative; the ` > 0` guard correctly disables in that
case, but a user passing `-1` thinking it means "off" would not be told they are
mis-using. **Proposed fix:** `assert args.force_rank_during_training >= 0` after
parse, with a clear error.

---

## Summary

| # | Attack                              | Verdict                | Severity |
|---|-------------------------------------|------------------------|----------|
| 1 | Default-path byte identity          | DOES NOT LAND          | —        |
| 2 | SVD backward, coincident σ          | **LANDS HARD**         | CRITICAL |
| 3 | Dtype / autocast wrapping omitted   | LANDS (partial)        | medium   |
| 4 | Entropy-loss sign                   | DOES NOT LAND          | —        |
| 5 | Nuclear-loss sign                   | DOES NOT LAND          | —        |
| 6 | Per-position averaging              | DOES NOT LAND          | —        |
| 7 | Z_list_grad memory leak (default)   | DOES NOT LAND          | cosmetic |
| 8 | Force-rank grad correctness         | DOES NOT LAND (random init); INHERITS #2 | (rolled into #2) |
| 9 | Parts dict shape                    | DOES NOT LAND          | —        |
| 10| Argparse defaults vs report         | DOES NOT LAND (cosmetic mismatch) | cosmetic |

**Two lands, one of them critical.**

Critical action item: replace full-SVD truncation in `MatrixBottleneck.forward`
(force_rank path) and `truncate_to_rank` with eigh-based projection BEFORE
running any training that uses `--force-rank-during-training`. The smoke test
should add a degenerate-spectrum case to detect regression.

Medium action item: wrap both new SVD/svdvals call sites in
`torch.autocast(..., enabled=False)`.

The rank-LOSS path (entropy / nuclear) is structurally safe — it uses only
`svdvals` whose backward goes through `Z Zᵀ` eigendecomposition and avoids
the U/V instability. Run those experiments first.
