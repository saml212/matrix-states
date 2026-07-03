# Task D — Mechanical Correctness Audit

Static analysis only (no torch/numpy available in this environment). Findings
split into (A) will crash / silently produce a wrong result, and (B) needs
runtime confirmation in the smoke gate. Line numbers refer to the files as of
2026-07-01.

---

## A. Will crash or produce a wrong result

### A1 — FATAL — `evaluate()` never applies `force_rank_k`; M3's headline metrics are wrong for every `--force-rank-k` run

`run_task_d.py:46-88` (`evaluate`) and `run_task_d.py:91-111` (`train`).

`evaluate()` has no `force_rank_k` parameter. Line 57 always computes the
**unconstrained** encoder output:

```python
Z = model.encode(b["keys"], b["values"])                 # unconstrained
pred = model.unbind(Z, b["query_keys"])
cos_all.append(recovery_cosine(pred, b["targets"]))       # -> mean_cos, recovered_frac
```

`train()` (the M3 causal-test entry point) trains with `force_rank_k` applied
every step (line 99: `model(b, force_rank_k=force_rank_k)`), but its final
call is:

```python
return evaluate(model, cfg, eval_gen, device, model_type)   # line 111, no force_rank_k
```

Consequence: for a model **trained** with e.g. `force_rank_k=1`, gradients
only ever flowed through the rank-1-projected `Zk = U_k(U_kᵀZf)`; the
orthogonal residual of the raw encoder output `Zf` was never optimized to be
task-informative (it received gradient only incidentally, through `eigh`'s
dependence on the whole matrix, not through the task loss directly). Reading
`pred = model.unbind(Zf_full, query)` at eval time therefore evaluates a
function the model was **never trained to compute** — the untrained residual
adds noise on top of the trained rank-1 signal. The top-level `mean_cos`,
`recovered_frac`, `cos_p10/50/90` fields — the numbers a human or a results
script will read first, and the ones written to `--out` — do **not** reflect
"how well does the rank-k-constrained model perform," which is exactly what
§6 M3's CONFIRM/FALSIFY criteria require ("best eval accuracy ... for k < K
... for k ≥ K").

The *correct* number does exist in the output, buried at
`result["rankk_curve"][str(force_rank_k)]["recovered_frac"]` (computed at
line 62-64 by re-truncating the same raw `Z` to each `k` in `rank_ks`) — but
nothing in the code promotes it, and the JSON round-trip turns the `int` key
into a string, an easy place for an analysis script to silently look up the
wrong (or a missing) key.

This does not crash. It silently corrupts the primary, pre-registered
decision metric (M3) for every force-rank-k run — rate it FATAL to the
experiment even though it is not a FATAL-to-execute bug.

**Fix** — thread `force_rank_k` through `evaluate()` and use it for the
headline prediction while still reporting the full curve for comparison:

```python
@torch.no_grad()
def evaluate(model, cfg, gen, device, model_type,
             tau: float = 0.9, rank_ks=(1, 2, 4, 8, 16), n_batches: int = 8,
             batch_size: int = 256, force_rank_k: int | None = None) -> dict:
    ...
    for _ in range(n_batches):
        b = sample_batch(cfg, batch_size, gen, device=device)
        if model_type == "matrix":
            Z = model.encode(b["keys"], b["values"])          # raw, for M1/M2/curve
            Z_eval = Z if force_rank_k is None else truncate_to_rank(Z, force_rank_k)
            pred = model.unbind(Z_eval, b["query_keys"])        # what the trained model actually emits
            cos_all.append(recovery_cosine(pred, b["targets"]))
            er_all.append(effective_rank(Z))
            for k in rank_ks:
                Zk = truncate_to_rank(Z, k)
                pk = model.unbind(Zk, b["query_keys"])
                rk_cos[k].append(recovery_cosine(pk, b["targets"]))
        ...
```

and in `train()`:

```python
    return evaluate(model, cfg, eval_gen, device, model_type, force_rank_k=force_rank_k)
```

---

### A2 — MAJOR — `stable_rank` is implemented but never wired in; violates C3

`rank_utils.py:56-64` defines `stable_rank`, pre-registered in
`TASK_D_PREREGISTRATION.md` §5 C3 as the required **secondary** rank metric
("Both pre-registered; no post-hoc metric shopping"). `run_task_d.py:28`
imports only `truncate_to_rank, effective_rank`; grep confirms `stable_rank`
is referenced nowhere outside its own definition. `evaluate()`'s output dict
(lines 78-87) reports `effective_rank_mean/std` but no `stable_rank_*`. As
shipped, M1 cannot be reported per protocol.

**Fix**:

```python
from rank_utils import truncate_to_rank, effective_rank, stable_rank
...
        if model_type == "matrix":
            ...
            er_all.append(effective_rank(Z))
            sr_all.append(stable_rank(Z))                      # add
...
    if model_type == "matrix":
        er = torch.cat(er_all)
        sr = torch.cat(sr_all)                                  # add
        out["effective_rank_mean"] = er.mean().item()
        out["effective_rank_std"] = er.std().item()
        out["stable_rank_mean"] = sr.mean().item()               # add
        out["stable_rank_std"] = sr.std().item()                 # add
```

(remember to initialize `sr_all = []` alongside `cos_all, er_all = [], []`).

---

### A3 — MAJOR — NaN-only gradient guard misses `Inf`

`run_task_d.py:103-105`:

```python
if any(p.grad is not None and torch.isnan(p.grad).any()
       for p in model.parameters()):
    raise FloatingPointError(f"NaN grad at step {step}")
```

`truncate_to_rank`'s `eigh` backward (see B1 below) can diverge to `Inf`
before it becomes `NaN` when eigenvalues are near-degenerate — a scenario
`force_rank_k` training is specifically designed to court (forcing rank
constrains the spectrum, increasing the chance of near-tied singular values
among the discarded directions as training progresses). An `Inf` gradient
passes this check silently, `opt.step()` applies it, and the corruption may
not manifest as a `NaN` loss until several steps later (or never, if Adam's
moment estimates absorb it), making the eventual failure hard to trace back
to its origin step.

**Fix** (also resolves the per-parameter CUDA sync cost noted in B4 below):

```python
grads = [p.grad for p in model.parameters() if p.grad is not None]
if grads and not torch.isfinite(torch.cat([g.reshape(-1) for g in grads])).all():
    raise FloatingPointError(f"non-finite grad at step {step}")
```

---

### A4 — MAJOR — the mandatory "blank-out" bottleneck test is vacuous

`run_task_d.py:157-169`:

```python
with torch.no_grad():
    Z = model.encode(b["keys"], b["values"])
    pred_a = model.unbind(Z, b["query_keys"])
    b_corrupt = dict(b)
    b_corrupt["keys"] = torch.randn_like(b["keys"])
    b_corrupt["values"] = torch.randn_like(b["values"])
    pred_b = model.unbind(Z, b_corrupt["query_keys"])   # same Z, same query
    assert torch.equal(pred_a, pred_b), "LEAK: decode changed when bindings changed"
```

`b_corrupt = dict(b)` is a shallow copy; only `"keys"` and `"values"` are
overwritten. `b_corrupt["query_keys"]` is **never corrupted** — it is the
same tensor object used for `pred_a`. `pred_b` is computed from the exact
same `(Z, query_keys)` pair as `pred_a`. Since
`MatrixMemoryModel.unbind(Z, query_keys)` (model_v4.py:82-87) is a
`@staticmethod` whose signature textually excludes `keys`/`values`, it is
*impossible* for this call to be affected by the corruption regardless of
whether the architecture has a real leak — the assertion is true by Python
argument-passing semantics alone, independent of model behavior. It cannot
fail today, and would not catch a real regression unless that regression
also happened to feed a stale, previously-corrupted `query_keys` into
`unbind` — a scenario the current code doesn't construct.

This is the test §4 of `TASK_D_PREREGISTRATION.md` calls "mandatory... If
this ever fails, the bottleneck leaks and every downstream number is
meaningless" — as implemented it can never fail, so it currently supplies
zero evidence toward that mandatory gate.

**Fix** — also assert the encoder is *sensitive* to its inputs (a real
regression guard against a disconnected/dead subgraph), keeping the existing
check as a signature-level regression guard:

```python
with torch.no_grad():
    Z = model.encode(b["keys"], b["values"])
    pred_a = model.unbind(Z, b["query_keys"])
    b_corrupt = dict(b)
    b_corrupt["keys"] = torch.randn_like(b["keys"])
    b_corrupt["values"] = torch.randn_like(b["values"])
    b_corrupt["query_keys"] = torch.randn_like(b["query_keys"])

    # sanity: the encoder DOES read its inputs (catches a dead/disconnected graph)
    Z_corrupt = model.encode(b_corrupt["keys"], b_corrupt["values"])
    assert not torch.equal(Z, Z_corrupt), "encoder ignored keys/values -- dead graph"

    # bottleneck: decode of a FIXED Z never depends on which bindings produced it
    pred_b = model.unbind(Z, b["query_keys"])
    assert torch.equal(pred_a, pred_b), "LEAK: decode changed for identical (Z, query)"
```

---

## B. Needs runtime confirmation in the smoke test

### B1 — `truncate_to_rank`'s "stable backward" claim is untested by Task D's own smoke gate

`rank_utils.py:9-14, 25-27` claims eigh backward is "numerically stable even
when singular values coincide" and was "Verified NaN-free on a constructed
[3,1,1,0] spectrum" — but that verification happened in a **different**
codebase (`_smoke_rank_aware.py`), not here. PyTorch's own docs for
`torch.linalg.eigh` state the backward is only well-conditioned when
eigenvalues are distinct; the vjp involves a `1/(λ_i − λ_j)` term for
`i ≠ j`, which for `Z Zᵀ` is `1/(σ_i² − σ_j²)` — the *same* expression the
docstring says this approach avoids relative to full-SVD backward. Whether
`eigh`'s implementation actually zeroes that term when the corresponding
gradient contribution is zero (rather than propagating `inf·0 = NaN`) is an
implementation detail that needs to be re-checked against PyTorch 2.9, not
assumed from a comment.

`run_task_d.py`'s current smoke gate never exercises this path:
- Step `[2]` (lines 125-131) calls `truncate_to_rank` on `torch.randn(4,16,16)`
  — a random Gaussian matrix has non-degenerate eigenvalues with probability
  1 — and never calls `.backward()` on the result (no `requires_grad=True`,
  pure forward check).
- Step `[4]` (lines 148-155) does call `.backward()` through `force_rank_k=2`,
  but on `Z` produced by an untrained, randomly-initialized encoder — again
  generically non-degenerate.

Neither exercises the one spectrum shape the docstring's safety claim rests
on. **Recommend adding an explicit degenerate-spectrum backward check**
before trusting `force_rank_k` training runs at scale:

```python
print("\n[2b] rank_utils degenerate-spectrum backward (eigh NaN/Inf risk)")
d = 4
U, _ = torch.linalg.qr(torch.randn(d, d, device=device))
V, _ = torch.linalg.qr(torch.randn(d, d, device=device))
sigma = torch.tensor([3.0, 1.0, 1.0, 0.0], device=device)   # repeated singular value
Zdeg = (U * sigma) @ V.T
Zdeg = Zdeg.unsqueeze(0).clone().requires_grad_(True)
for k in (1, 2, 3):
    out = truncate_to_rank(Zdeg, k).sum()
    (grad,) = torch.autograd.grad(out, Zdeg, retain_graph=False)
    assert torch.isfinite(grad).all(), f"non-finite grad at k={k} on degenerate spectrum"
print("  degenerate-spectrum backward finite for k in (1,2,3)")
```

If this fails on the H100, `force_rank_k` training is unsafe as-is and needs
either a full-SVD-with-epsilon-regularized-denominator backward, a
stop-gradient on the projection basis (`U_k.detach()`), or gradient clipping
around the truncation step.

### B2 — `--orthogonal`'s QR path is never exercised on the target CUDA device

`task_d.py:58-61` (the `orthogonal=True` branch of `_random_directions`,
using `torch.linalg.qr`) is only ever run from `_self_test()`
(`task_d.py:115-159`), which is hardcoded to CPU (see B3). `run_task_d.py`'s
`smoke()` step `[1]` calls `td._self_test()` (CPU only) and step `[3]`'s
`sample_batch` call (`run_task_d.py:137`, `device=device`) uses the default
`TaskDConfig(d=16, K=8)` → `orthogonal=False`, so the QR branch never runs on
CUDA anywhere in the smoke gate. `torch.linalg.qr` on CUDA goes through
cuSOLVER, a different numerical path than CPU LAPACK; batched-QR shape
handling should be fine in 2.9 but has not been exercised on-device by this
code. **Recommend**: before any `--orthogonal` run on the H100, add a smoke
line `sample_batch(TaskDConfig(d=16, K=8, orthogonal=True), 32, torch.Generator(device=device).manual_seed(2), device=device)` and assert orthonormality (mirrors the `_self_test` check) on-device.

### B3 — `task_d._self_test()` hardcodes CPU; latent device-mismatch trap if ever parameterized

`task_d.py:115-159`. `torch.manual_seed(0)` / `torch.Generator().manual_seed(0)`
(no `device=`) and `b = sample_batch(cfg, batch_size=64, gen)` (no
`device=` kwarg, defaults to `"cpu"` per `sample_batch`'s signature) are
self-consistent today — everything in this function is CPU. But line 146,
`eye = torch.eye(K).expand(B, K, K)`, has **no explicit device**, unlike
`torch.ones_like(kn)` at line 133 (which is always device-safe because it
mirrors its input tensor). If `_self_test()` is ever extended to accept a
`device` parameter (a natural-looking refactor given the rest of the file is
device-aware) and only the `gen`/`sample_batch` calls are updated, `eye`
would remain on CPU while `gram` moved to CUDA, and
`gram - eye * gram` at line 147 would raise a device-mismatch
`RuntimeError`. Not a bug today; flagging because the audit brief
specifically calls out this exact `torch.eye` pattern, and it is the one
place in the file where `ones_like`'s automatic device inference isn't
mirrored by `eye`.

**Fix** (defensive, no behavior change today): `eye = torch.eye(K, device=b["keys"].device).expand(B, K, K)`.

### B4 — per-step NaN check forces a CUDA sync per parameter, per step

`run_task_d.py:103-105`. `any(... torch.isnan(p.grad).any() ...)` is a
Python `any()` over a generator; each `.any()` on a CUDA tensor forces a
device→host sync to evaluate its truthiness. With ~a dozen parameter tensors
× up to 3000 steps (§7 compute plan), this is a real number of syncs on the
H100 — a throughput concern, not a correctness bug. The single-`isfinite`
fix in A3 also collapses this to one sync per step.

### B5 — `force_rank_k=0` silently means "unconstrained," not "rank 0"

`model_v4.py:77`: `if force_rank_k is not None and force_rank_k > 0:`. Passing
`--force-rank-k 0` skips truncation entirely rather than producing a
rank-0 (all-zero) `Z`. Not exercised by §7's sweep grids (`k ∈ {1,2,4,8,16}`
never includes 0), so unlikely to bite the pre-registered runs, but worth a
one-line docstring/CLI-help note so a future sweep script doesn't
misinterpret `0` as a "no signal" control arm and silently get the
unconstrained model instead.

### B6 — `--model vector --force-rank-k K` silently ignores the flag

`model_v4.py:133-135`, `HRRVectorMemory.forward` accepts and discards
`force_rank_k` by design (a vector state has no matrix rank). Correct
behavior, but `run_task_d.py`'s CLI prints `force_rank_k={args.force_rank_k}`
in its startup banner (line 215) regardless of `--model`, which could read as
"this vector run is rank-constrained." **Recommend** a one-line warning in
`main()`: `if args.model == "vector" and args.force_rank_k is not None: print("WARNING: --force-rank-k is ignored for --model vector")`.

---

## Verdict

One FATAL-to-the-experiment issue (A1: `evaluate()` doesn't apply
`force_rank_k`, so M3's headline `mean_cos`/`recovered_frac` are wrong for
every force-rank-k run — the correct number is present but buried in
`rankk_curve`) plus one pre-registration compliance gap (A2: `stable_rank`
never wired in) must be fixed before any M3 run's results are trusted or
reported. The einsum unbind, the rank-truncation projection math, the HRR
circular-convolution/involution, the generator's gather/argsort/QR paths,
and the `nn.TransformerEncoderLayer`/`MultiheadAttention` usage are all
mechanically correct on static inspection. Fix A1–A4 (all small, localized
patches), then smoke-test — add B1 and B2's device/degenerate-spectrum
checks to the smoke gate itself before trusting `force_rank_k` and
`--orthogonal` runs on the H100.
