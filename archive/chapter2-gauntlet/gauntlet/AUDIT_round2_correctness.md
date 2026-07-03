# Round-2 Audit — Task D Rank Gate (`chapter2/`)

Static analysis only (no torch available). Verifies round-1 findings against the
current code, then does a fresh adversarial pass on the rewritten `run_task_d.py`.
Line numbers refer to the files as read 2026-07-01.

---

## Part A — Round-1 findings, verified against current code

### `AUDIT_correctness.md`

| # | Finding | Sev | Status | Evidence |
|---|---|---|---|---|
| A1 | `evaluate()` never applied `force_rank_k`; M3 headline metrics wrong | FATAL | **RESOLVED** | `run_task_d.py:68` adds `force_rank_k=None` param; `:83` `Z = model.encode(b["keys"], b["values"], force_rank_k=force_rank_k)` — headline Z now uses the trained constraint; `train()` final call `:134` `evaluate(model, cfg, eval_gen, device, model_type, force_rank_k=force_rank_k)` threads it through. |
| A2 | `stable_rank` defined but never wired in (violates C3) | MAJOR | **RESOLVED** | `:36` imports `stable_rank`; `:77` `sr_all=[]`; `:87` `sr_all.append(stable_rank(Z))`; `:105` `out["stable_rank_mean"]`. Minor gap: no `stable_rank_std` (asymmetric with `effective_rank_std`) — not required by C3's text, see Part B. |
| A3 | NaN-only grad guard misses Inf | MAJOR | **RESOLVED** | `:126-127` `if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()): raise FloatingPointError(...)`. Correctness fixed; introduces a mild performance regression vs. the suggested single-`isfinite` patch — see Part B §6. |
| A4 | Blank-out test vacuous (query_keys never corrupted, tautological) | MAJOR | **RESOLVED** | Rewritten as a real gradient-flow test, `:187-202` (smoke `[5]`). Verified sound in Part B §1 below; one residual blind spot noted (Part B §2). |
| B1 | `eigh` degenerate-spectrum backward untested | needs-runtime | **RESOLVED (test added)** | `:147-161` (smoke `[2]`) constructs `Zdeg` with spectrum `[3,1,1,0]`, backward through `truncate_to_rank(Zdeg,2)`, asserts no NaN/Inf. Verified the construction correctly lands the tie exactly at the k=2 cut boundary (Part B §4). Outcome still needs an actual H100 run — code-level gap is closed, numerical outcome is not yet observed. |
| B2 | `--orthogonal` QR path never exercised on CUDA | needs-runtime | **RESOLVED** | Gate default flipped to `orthogonal=True` (`:233`) and smoke steps `[3]-[6]` build `TaskDConfig(..., orthogonal=True)` with `sample_batch(..., device=device)` — QR path now runs on-device in the standard smoke gate, not CPU-only. |
| B3 | `task_d._self_test()` hardcodes CPU; `torch.eye(K)` has no device | latent | **NOT RESOLVED (unchanged, still non-blocking)** | `task_d.py:138` still `eye = torch.eye(K).expand(B, K, K)` with no `device=`. `_self_test()` is still fully CPU-self-consistent (`task_d.py:108-109`), so this remains inert today — same status as round 1. |
| B4 | Per-step NaN check forces a CUDA sync per parameter | perf | **NOT RESOLVED, mildly worse** | Still loops `for p in model.parameters(): ...` (`:125-127`). The A3 fix now evaluates `isnan().any()` **and**, when that's `False` (the common case), also `isinf().any()` — i.e. 2 syncs/param/step instead of 1. See Part B §6. |
| B5 | `force_rank_k=0` silently means "unconstrained" | minor | **NOT RESOLVED** | `model_v4.py:77` unchanged: `if force_rank_k is not None and force_rank_k > 0:`. No docstring/CLI note added. Inert at current grids (never includes 0). |
| B6 | `--model vector --force-rank-k K` silently ignored, no warning | minor | **NOT RESOLVED** | Grepped `run_task_d.py` for a vector+force_rank_k warning — none exists. Only `n_query<K` (`:256-258`) and `K>d` (`:260-261`) warnings were added. |

### `AUDIT_validity.md`

| # | Finding | Sev | Status | Evidence |
|---|---|---|---|---|
| Item 1 | Blank-out test vacuous | MAJOR | **RESOLVED** | Same fix as A4 above. |
| Item 2 | Readout pinning correct | none | **STILL TRUE** | `model_v4.py:81-87` unchanged, `unbind` is still a bare 2-arg `@staticmethod` einsum. |
| Item 3 | `orthogonal` should default True for the K=8 operating point | MAJOR | **RESOLVED** | `run_task_d.py:233-236` argparse default `orthogonal=True`; `TASK_D_PREREGISTRATION.md:162-164` locks it in as the fixed operating point. Argparse dest-sharing logic verified correct — see Part B §5. |
| Item 4 | τ=0.9 default + Gaussian smears the knee (quantified) | MAJOR | **RESOLVED** | Default flipped to orthonormal; `TAUS = (0.9, 0.95, 0.99)` (`:38`) reports all three; preregistration `§6` sets knee test primary τ=0.99 (not the audit's suggested 0.95/0.999, but in the same direction and locked in explicitly — a defensible judgment call, not a gap). |
| Item 5 (M2 grid) | Coarse `{1,2,4,8,16}` grid can't resolve the knee | MAJOR | **RESOLVED** | `evaluate()` default `rank_ks = tuple(range(1, cfg.d+1))` (`:76`) — every integer 1..d, denser than the audit's suggested subset. |
| Item 5 (M1 aggregation script) | No script computes Spearman ρ / eff-rank CONFIRM check | minor | **NOT RESOLVED** | Still no aggregation script in `chapter2/`; raw data is saved so this is non-blocking but still an open procedural gap. |
| Item 5 (M3 launcher) | No driver script exists for the k-sweep with fixed seeds/hparams | minor/procedural | **NOT RESOLVED** | `grep -rl "task_d" matrix-thinking/scripts/` returns nothing; `pebble_launcher.sh` (new, untracked) is for the ProsQA/matrix-CODI pipeline only, does not reference `task_d`/`run_task_d`. Must be written before Stage 1's `~18 short runs` are launched, per `TASK_D_PREREGISTRATION.md §7`. |
| Item 6 | HRR vector baseline under-powered (40x param gap, 16x state gap) | MAJOR/FATAL-for-M4 | **RESOLVED (via descope)** | `model_v4.py:96-106` docstring + `TASK_D_PREREGISTRATION.md §5 C2` explicitly descope M4 to Stage-1b, state it is dev-only and must not be reported as the C2 control. `run_task_d.py` smoke `[7]` (`:216-221`) prints the same caveat. This resolves the finding by descoping rather than fixing the mismatch, which is an accepted, explicitly-documented resolution — not a code bug. |

### `AUDIT_adversarial.md`

| # | Finding | Sev | Status | Evidence |
|---|---|---|---|---|
| Finding 1 | τ=0.9 + coarse grid can't tell rank K from K-1/K-2 (functionally FATAL) | MAJOR | **RESOLVED** | Same fixes as Items 4/5 above, plus the exact "resolution pre-flight" check this finding requested is now smoke `[6]` (`:204-214`) — verified sound in Part B §2. |
| Finding 2 | rank-1 `u⊗v0` shortcut closed | none | **STILL TRUE** | `unbind` unchanged. |
| Finding 3 | `eigh` backward near-degenerate-spectrum risk (unverified) | MINOR-MODERATE | **RESOLVED (test added, outcome pending H100 run)** | Same as B1 above; the constructed spectrum lands the tie exactly at the truncation boundary, which is the scenario this finding specifically worried about — see Part B §4. |
| Finding 4 | `n_query < K` silently weakens bound to rank≥Q; recommend warning + surfacing Q in output | MINOR | **PARTIAL** | Warning added (`:256-258`). `Q`/`cfg.queries` is **not** added to `evaluate()`'s output dict (`out = {"K":..., "d":..., "model":..., "orthogonal":..., "force_rank_k":...}`, `:98-99` — no `Q` key). A saved JSON result alone still can't reveal `n_query<K` was used. |
| Finding 5 | Stale `ROLE_KEY/ROLE_VALUE/ROLE_QUERY` docstring/constants | MINOR | **RESOLVED** | Grepped `chapter2/*.py` — no `ROLE_KEY`/`ROLE_VALUE`/`ROLE_QUERY` remain (only in the round-1 audit `.md` files themselves). `task_d.py`'s module docstring (`:1-12`) no longer mentions role embeddings. Resolved via deletion + doc correction, as the audit's fix option 2 suggested. |
| Findings 6,7,8 | Query/target leakage, force_rank_k routing, generator correlation — all closed | none | **STILL TRUE** | Underlying mechanisms (`encode`/`unbind` call sites, `truncate_to_rank`'s exact projection, `sample_batch`'s independent draws) are unchanged. |

---

## Part B — Fresh findings on the rewritten `run_task_d.py`

### 1. Blank-out test (smoke `[5]`) — verified genuinely non-tautological. No bug found.

Traced the full autograd path: `keys`/`values` are cloned into fresh leaves
(`:188-189`), `Zg = model.encode(keys, values)` builds a real graph, and
`gk = torch.autograd.grad(pred_g.sum(), keys, retain_graph=True, allow_unused=True)[0]`
followed by `assert gk is not None and gk.abs().sum() > 0` (`:192-193`) is a genuine
sensitivity check — it would fail if the encoder ever became a dead/disconnected
subgraph w.r.t. its inputs, which the round-1 test could never catch. The second half
(`Z_leaf = Zg.detach().clone().requires_grad_(True)`, `:195`) correctly produces a leaf
with **no** path back to `keys`, so `torch.autograd.grad(pred_leaf.sum(), keys,
allow_unused=True)[0]` correctly returns `None` (`:197`) rather than erroring, because
`pred_leaf` (built from `Z_leaf`, which does require grad) legitimately requires grad
while having zero graph-overlap with `keys`. This closes round-1 Item 1's core
complaint. `retain_graph=True` on the first `grad()` call is unnecessary (nothing else
consumes that graph afterward) but harmless.

### 2. [MINOR] Structural leak-guard silently drops `self`, weakening its own future-regression value

`:199-201`:
```python
sig = set(inspect.signature(MatrixMemoryModel.unbind).parameters) - {"self"}
assert sig <= {"Z", "query_keys"}, ...
```
Today `unbind` is `@staticmethod`, so `"self"` was never in the signature and the
subtraction is a no-op — the check is correct as written. But if a future refactor
turned `unbind` into a bound instance method that reads cached raw keys/values off
`self` (e.g. `self._last_keys`) instead of taking them as parameters — exactly the
"helpfully refactored" regression scenario `AUDIT_validity.md` Item 1 warned about —
this check would still pass, because `self` is unconditionally stripped before the
`<=` comparison. The round-1 fix suggestion also proposed
`isinstance(vars(MatrixMemoryModel)["unbind"], staticmethod)`; that half of the
suggested patch was not carried over. **Fix:** add the `isinstance(..., staticmethod)`
assertion alongside the signature check.

### 3. Resolution pre-flight (smoke `[6]`) — math verified correct, but the safety margin is thinner than the code's own comment implies

`Zexact = torch.einsum("bki,bkj->bij", bb["values"], bb["keys"])` (`:207`) correctly
computes `Σ_k v_k k_k^T`, and for exactly orthonormal keys, `Zexact @ key_q = v_q`
exactly (`k_k·k_q = δ_kq`), so `cos_full > 0.99` (`:211`) is safe.

The subtler issue: because `bb`'s **values** are *also* drawn with `orthogonal=True`
(`cfgo = TaskDConfig(d=16, K=8, orthogonal=True)`, `:205`), `Zexact` has *all K*
nonzero singular values **exactly tied at 1** (it's `V K^T` with both `V,K`
orthonormal-column, so `Z Z^T = V V^T`, eigenvalues `{1 (×K), 0 (×(d−K))}` — a fully
degenerate top cluster, not merely "near-degenerate"). Truncating to rank `K−1`
(`:209`) therefore does **not** reliably "drop one full binding" the way the
surrounding narrative (`TASK_D_PREREGISTRATION.md §6`: "rank K−1 then must drop a
full value direction, so exactly one binding fails") suggests — `eigh` returns an
*arbitrary* orthonormal basis of the tied 8-dim eigenspace, so the dropped direction
`n` can be axis-aligned to one `v_i` (gap ≈ 0.125, "one binding fails completely") or
spread evenly across all 8 (gap ≈ 0.065, "every binding degrades a little"), or
anywhere between.

I derived the general result: if `n = Σc_i v_i` (`Σc_i²=1`) is the dropped direction,
`cos(pred_q, v_q) = sqrt(1−c_q²)` per query, and by Jensen's inequality (`sqrt(1−x)` is
concave) the mean over 8 queries is **maximized** — i.e. the assertion margin is
**smallest** — at the uniform-spread extreme `c_q²=1/8 ∀q`, giving
`mean_cos_tr = sqrt(7/8) ≈ 0.9354`, a gap of `1 − 0.9354 ≈ 0.0646` from `cos_full≈1`.
This is still `> 0.05` (margin ≈ 0.015), so the assertion `cos_tr < cos_full − 0.05`
(`:212-213`) **should reliably pass** even in the theoretical worst case, and averaging
over `B=128` independent instances (`:206`) further dilutes any single unlucky draw.
**Not a bug — the check is sound — but severity MINOR: the safety margin (~0.015) is
much thinner than a reader would assume from the accompanying comment, and it rests on
a fully-degenerate (not just near-degenerate) construction.** Recommend either (a)
loosening the assert threshold to something like `0.03` for extra headroom, or (b)
correcting the comment to say "degrades across bindings, not necessarily drops exactly
one" so a future reader doesn't mis-debug a marginal failure.

### 4. Degenerate-spectrum backward test (smoke `[2]`) — construction verified correct and adversarially well-targeted

`:153-161`. `U, V = qr(randn(4,4))` are built from `torch.randn` tensors that never
had `requires_grad=True`, so `U`, `V`, and the product `U @ diag([3,1,1,0]) @ V.T` all
have `requires_grad=False` (no `grad_fn`). Calling `.unsqueeze(0).requires_grad_(True)`
on that plain tensor correctly converts it into a **leaf** tensor with
`requires_grad=True` — this is the standard, correct idiom (it would only raise if
applied to a non-leaf, i.e. a tensor with an existing `grad_fn`, which is not the case
here). `Zdeg.grad` will be populated by `.backward()` as expected — no leaf/non-leaf
bug.

More importantly: `truncate_to_rank(Zdeg, 2)` computes `eigh(Z Z^T)`, whose eigenvalues
are `[σ_i²] = [9, 1, 1, 0]` (ascending: `[0,1,1,9]`). Truncating to `k=2` keeps the top
2 (`eigvecs[..., -2:]`), which is exactly `{9, 1}` — meaning the cut lands **exactly at
the boundary of the tied pair** (one `1` kept, one `1` dropped). This is precisely the
"genuinely dangerous case" `AUDIT_adversarial.md` Finding 3 flagged as unverified
(boundary-tied, not just tied-in-the-tail like round-1's `[3,1,1,0]` self-test which
only exercised `k≤2` with the tie fully excluded). The test is well-constructed to
exercise the real risk. **Whether it actually passes on real hardware is still
unverified** — this is a legitimate "flag for smoke" item, not a static-analysis
verdict; recommend running `python run_task_d.py --smoke` on the H100 as the very
first step before any `force_rank_k` training.

### 5. argparse `--orthogonal`/`--gaussian` dest-sharing — verified correct

`:233-236`. `--orthogonal` (`action="store_true", default=True`) and `--gaussian`
(`action="store_false"`, dest shared) both target `dest="orthogonal"`. Argparse
initializes namespace defaults by iterating registered actions in order and only
applying an action's `default` if the dest isn't already set (`hasattr` check) — so
the first-registered action's `default=True` wins regardless of the second action's
implicit default, and explicit `--gaussian` on the command line still correctly
overrides to `False` at parse time (explicit flags always win over defaults). This is
the standard, correct pattern for paired boolean flags; no bug.

### 6. [MINOR, new] The A3 fix (NaN→NaN/Inf) roughly doubles per-step CUDA sync overhead, reintroducing part of round-1's B4 as a live cost

`:125-127`:
```python
for p in model.parameters():
    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
        raise FloatingPointError(f"NaN/Inf grad at step {step}")
```
Correctness-wise this resolves A3 (Inf now caught). But in the common case (no NaN),
Python's `or` still evaluates `torch.isinf(p.grad).any()`, so **every** parameter now
costs 2 device→host syncs per step instead of round-1's 1 — the exact overhead B4
flagged, now doubled. Round-1's suggested fix (`torch.isfinite(torch.cat([g.reshape(-1)
for g in grads])).all()`, one sync total per step) was not adopted. Over `~3000` steps
× a dozen parameter tensors (`TASK_D_PREREGISTRATION.md §7`), this is a real,
avoidable throughput tax on the H100, not a correctness bug. **Fix:** adopt the
single-tensor `isfinite` pattern.

### 7. [MINOR, informational] Two round-1 "fix" suggestions were partially applied

- `A2`'s suggested fix included `stable_rank_std`; only `stable_rank_mean` was added
  (`:105`), asymmetric with `effective_rank_mean/std`. Not required by C3's text
  ("secondary = stable rank" is satisfied by reporting the mean), but cheap to add for
  parity.
- `Finding 4`'s fix asked for both a CLI warning **and** surfacing `Q` in the result
  dict; only the warning shipped (see Part A table). A saved-JSON-only consumer (e.g.
  a future results-aggregation script) can't detect `n_query<K` runs without also
  parsing the run's CLI invocation.

---

## Verdict

All FATAL and MAJOR round-1 findings that block correctness are **RESOLVED**:
`evaluate()`/`train()` now correctly thread `force_rank_k` end-to-end with M2 properly
gated to unconstrained-only runs; the blank-out test is a real gradient-flow check, not
a tautology; `stable_rank` is wired in; the NaN guard now also catches Inf; the rank
grid is dense (1..d); `orthogonal=True` is the default and its QR path is now exercised
on-device; the τ set and knee-test τ=0.99 are locked into the preregistration; the
degenerate-spectrum and resolution-preflight smoke checks that round 1 said were
missing now exist and, on static/mathematical inspection, are correctly targeted and
should pass. The vector baseline (M4/C2) is honestly descoped rather than
misreported. Stale `ROLE_*` constants are gone.

What's left is exclusively **MINOR / procedural**, and none of it blocks a smoke run:
the M1-aggregation and M3-sweep-launcher scripts still don't exist (write before Stage
1, not before smoke), `Q` isn't surfaced in the output dict, `--model vector
--force-rank-k` still warns nothing, `task_d.py`'s `torch.eye` still lacks an explicit
device (inert, CPU-only function), and the Inf-check fix costs an extra sync/param/step
(throughput, not correctness). One new substantive-but-passing finding: the resolution
pre-flight's assert margin is thinner (~0.015) than its comment implies, because the
test's own construction is *exactly* rank-degenerate — verified by direct calculation
that it should still hold reliably.

**The code is correct enough to smoke-test on the H100 now.** Run
`python run_task_d.py --smoke` first; the two runtime-dependent items to watch in that
output are (a) the degenerate-spectrum backward in step `[2]` (finite grad, no NaN/Inf)
and (b) the resolution pre-flight in step `[6]` (`cos_tr` should land noticeably below
`cos_full − 0.05`, per Part B §3's derivation — if it's ever a near-miss, that's the
signal to revisit that assert's margin, not a sign of a code bug). Write the M3 sweep
launcher and add the `Q` field before trusting any aggregated Stage-1 result.
