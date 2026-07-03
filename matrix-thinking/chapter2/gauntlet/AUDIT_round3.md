# Round-3 Audit — Final Static Pass Before H100 Smoke (`chapter2/`)

Static analysis only (no `torch` available). Verifies the round-2 fixes against
current code, then does a fresh adversarial pass on the new `run_stage1.py`
launcher. Line numbers refer to files as of 2026-07-01.

---

## Part A — Round-2 fix verification

### A1. Haar sign-correction (`task_d._random_directions`) — **CONFIRMED-FIXED**

```python
q, r = torch.linalg.qr(x.transpose(-1, -2))    # x:(B,n,d) -> q:(B,d,n), r:(B,n,n)
s = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))   # (B, n)
s = torch.where(s == 0, torch.ones_like(s), s)         # sign(0) -> +1
q = q * s.unsqueeze(-2)                                # scale columns
return q.transpose(-1, -2).contiguous()                # (B, n, d)
```
`x.transpose(-1,-2)` is `(B,d,n)` with `d>=n`, so reduced QR gives `q:(B,d,n)`
with **orthonormal columns** (indexed by the last axis) and `r:(B,n,n)`.
`torch.diagonal(r, dim1=-2, dim2=-1)` correctly extracts `diag(R)` per batch
→ `(B,n)`. `s.unsqueeze(-2)` → `(B,1,n)`, broadcast-multiplied against
`q:(B,d,n)` scales **column `j`** (last axis) by `sign(R_jj)` for every row —
this is exactly the Mezzadri (2007) correction applied to the correct axis
(columns, pre-transpose), not rows. `sign(0)->1` is a benign, measure-zero
tie-break. This is the textbook fix and removes the diagonal bias round-2
quantified (~0.19 magnitude, >100σ) down to sampling noise. **No bug.**

### A2. Resolution pre-flight (`run_task_d.py` smoke `[6]`) — **CONFIRMED-FIXED**

```python
Zexact = torch.einsum("bki,bkj->bij", bb["values"], bb["keys"])   # Σ v_k k_k^T
pred_full = model.unbind(Zexact, bb["query_keys"])
pred_tr   = model.unbind(truncate_to_rank(Zexact, cfgo.K - 1), bb["query_keys"])
frac_full = (recovery_cosine(pred_full, bb["targets"]) > 0.99).float().mean().item()
frac_tr   = (recovery_cosine(pred_tr,   bb["targets"]) > 0.99).float().mean().item()
assert frac_full > 0.95
assert frac_tr < 0.5
```
- **`Zexact` einsum:** `"bki,bkj->bij"` sums over `k`, giving
  `Zexact[b,i,j] = Σ_k values[b,k,i]·keys[b,k,j] = (Σ_k v_k k_k^T)[i,j]` —
  correct, matches Σvₖkₖᵀ literally.
- **`pred_full`/`pred_tr` via `model.unbind`:** `unbind` is a bare
  `@staticmethod` pure function of `(Z, query_keys)` — calling it off an
  arbitrary `model` instance is valid (no dependence on model parameters),
  and `Zexact @ key_q = Σ_k v_k(k_k·key_q) = v_q` for orthonormal keys, so
  `cos(pred_full, target) ≈ 1` — correct.
- **The asserts, given the literal `recovered_frac@0.99` criterion:** this
  now directly checks the §6 decision statistic (a real fix over round-2's
  mean-cosine proxy, which round-2-correctness Part B §3 flagged as a
  thinner-than-advertised margin — that concern is now moot since the proxy
  is gone). Analytically, `ZZᵀ` for `Zexact` is *exactly* rank-`K` degenerate
  (eigenvalues `{1×K, 0×(d−K)}`), so `eigh` must pick *some* orthonormal
  basis of the tied `K`-eigenspace when truncating to `K−1`; round-2's
  independent numpy re-implementation measured `recovered_frac@0.99 = 0%`
  across 3000 trials at this exact operating point (`d=16,K=8`), giving
  `frac_tr<0.5` an enormous empirical margin, and `frac_full>0.95` is
  essentially guaranteed by exact reconstruction. **Math and code are sound.**
  One caveat carried over as a genuine runtime unknown (not a bug): the
  *specific* eigenvector basis LAPACK/cuSOLVER picks for an exactly-tied
  eigenvalue is an implementation-defined tie-break. Round-2's verification
  used a CPU/numpy LAPACK path; the H100 run goes through cuSOLVER (GPU
  `eigh`). If that tie-break ever landed axis-aligned with one query's `v_k`
  on many of the `B=256` trials (the theoretical worst case derived in
  round-2-correctness Part B §3: per-item cos up to 1 for 7/8 items, 0 for
  the 8th → `frac_tr` could reach `(K−1)/K=0.875`, failing `frac_tr<0.5`),
  the assert would fail. This is architecturally implausible (floating-point
  perturbation of an exactly-degenerate matrix generically produces a
  "generic-looking" basis, not an axis-aligned one, and it's diluted over
  256 independent draws) and round-2's simulation is strong evidence it
  won't happen — but it is GPU-`eigh`-implementation-dependent and cannot be
  proven from statics alone. **Flag for smoke:** watch step `[6]`'s printed
  `frac_tr` value on the actual H100 run; it should print ≈0.0, not merely
  scrape under 0.5.

### A3. Minor fixes — **ALL CONFIRMED-FIXED**

- `n_query` surfaced in output: `run_task_d.py:98`, `out = {..., "n_query":
  cfg.queries, ...}` — present, closes round-2-adversarial Finding 4's
  residual gap.
- `--model vector --force-rank-k` warning: `run_task_d.py:269-270` —
  `if args.model == "vector" and args.force_rank_k is not None: print("WARNING: ...")`
  — present, closes round-2-correctness B6.
- No leftover `if False else`: grepped the full `smoke()` function and
  `main()` — none present. Clean.

**Part A verdict: all three round-2 items are genuinely fixed.** One
non-blocking runtime watch-item remains on A2 (GPU `eigh` tie-break), flagged
above for the smoke run, not a code defect.

---

## Part B — Fresh audit of `run_stage1.py`

### B1. Aggregation key types — **CONFIRMED CORRECT**

- `rankk_curve` keys: `evaluate()` builds `{int(k): ... for k in rank_ks}`
  (`run_task_d.py:107-110`). `aggregate()`'s `ks_curve = sorted(curves[0])`
  sorts a dict → sorted **int** keys; `avg = {k: mean([c[k][frac] for c in
  curves]) for k in ks_curve}` indexes `c[k]` with those same int keys — types
  match throughout the in-process path. No str/int mismatch.
- `frac = f"recovered_frac@{TAU}"` with `TAU = 0.99` (float): verified by
  direct interpreter check, `f"recovered_frac@{0.99}"` → `"recovered_frac@0.99"`,
  identical to `run_task_d.py`'s `_recovery_stats`, which builds the same key
  from `TAUS = (0.9, 0.95, 0.99)` via `f"{prefix}recovered_frac@{tau}"` with
  `tau=0.99` from the same tuple. Python's shortest-round-trip float repr
  renders `0.99` as `"0.99"` in both files — **no formatting mismatch.**
  (The feared `0.99` vs `0.9` collision does not occur; `0.9` and `0.99` are
  visually similar but format to distinct, non-prefix-colliding strings.)

### B2. M1/M2/M3 logic

**M2 knee extraction — CONFIRMED CORRECT.** `acc_d = avg[max(ks_curve)]` is
`acc(k=d)`; the loop over ascending `ks_curve` returning the first `k` with
`avg[k] >= 0.9*acc_d` is exactly §6's "smallest k with acc(k) ≥ 0.9·acc(k=d)".
`m2_ok` checks `K_GATE-1 <= m2_knee <= K_GATE+1`, matching §6's `[K−1,K+1]`
window. `FORCE_RANKS=(2,4,6,7,8)` and `UNCONSTRAINED_KS=(1,4,8,16)` are a
byte-for-byte match to §7's Stage-1 grid — **no off-by-one.**

**M1 `m1_trends_up` — [MODERATE] a much weaker proxy than §6's literal M1
CONFIRM criterion; risk of a misleading advisory PASS.**
```python
m1_trends_up = all(er_seq[i] <= er_seq[i+1] + 0.5 for i in range(len(er_seq)-1)) \
    and er_seq[-1] > er_seq[0] + 1.0
```
This only requires (a) no single consecutive-K drop exceeding 0.5, and (b)
*total* growth from `K=1` to `K=16` exceeding 1.0. §6's actual M1 CONFIRM is
`Spearman ρ(K, eff_rank) ≥ 0.8` **and** `eff_rank ∈ [0.7K, 1.3K]` for each
`K≤d`. A sequence like `eff_rank ≈ [1.0, 1.3, 1.6, 2.5]` at `K=[1,4,8,16]`
would pass `m1_trends_up` (final−first = 1.5 > 1.0, no big drops) while
utterly failing the pre-registered range check (`eff_rank(16)=2.5` is nowhere
near `[11.2, 20.8]`) — i.e., the launcher could print a "PASS" while the real
M1 criterion FALSIFYs. The docstring does call the verdict "advisory," which
covers this legally, but the gap is large enough that a human skimming
`PRELIMINARY_VERDICT` could be misled. **Fix (cheap, recommended before
trusting any Stage-1 "PASS"):** compute the literal statistic in `aggregate()`
— Spearman ρ (a manual rank-correlation is ~5 lines if avoiding a `scipy`
dependency) and `all(0.7*k <= m1[k]["effective_rank"] <= 1.3*k for k in ks)`
— and gate `m1_trends_up` on that instead of the current heuristic. This also
finally closes round-2-validity's still-open "M1 aggregation script" item
properly (a script now exists, but doesn't yet implement the pre-registered
statistic).

**M3 step / hard-falsify — [MINOR-MODERATE] uses non-literal thresholds, but
defensibly so.**
```python
acc_atK = m3[K_GATE]["recovered_frac"]                 # force_rank_k=8 run's own acc
acc_belowK = max(m3[k]["recovered_frac"] for k in FORCE_RANKS if k < K_GATE)
m3_step = (acc_atK >= 0.9 and acc_belowK <= acc_atK - 0.3)
m3_hard_falsify = acc_belowK >= 0.5
```
Two deviations from §6's literal wording, both defensible but worth
surfacing:
1. §6 defines "ceiling" as the **unconstrained** model's converged accuracy,
   but this code substitutes the `force_rank_k=8` run's own accuracy as both
   the "at-K" value and the ceiling proxy (using `acc_atK >= 0.9` as an
   absolute bar rather than `>= 0.9 * true_ceiling`). Since the true
   unconstrained `K=8` run's data is *also* collected (`unconstrained` list
   has `K=8` entries) but never consulted here, if real training only
   converges to e.g. 0.85 ceiling, `acc_atK>=0.9` is a *stricter* bar than
   the pre-registered `0.9×ceiling` and could produce a false
   INCONCLUSIVE/FAIL where the literal §6 criterion would say CONFIRM. Not
   likely at this task's scale (recovery should approach ~1.0 given the
   provable-solution setup), but worth a one-line fix: pull
   `ceiling = mean(r[frac] for r in unconstrained if r["K"]==K_GATE)` and use
   `acc_atK >= 0.9*ceiling`.
2. `m3_hard_falsify` uses an absolute `0.5` bar at the smallest **tested**
   rank (`k=2`, since Stage-1's grid per §7 already excludes `k=1` — that
   grid choice is pre-registered, not a launcher bug), where §6's literal
   HARD FALSIFY is `≥0.9·ceiling` at `k=1` specifically. `0.5` is a
   conservative, earlier-tripping proxy (good for an advisory "stop and
   look" signal) but is not the same statement as §6's; a human should not
   read "hard_falsify=False" here as literally clearing §6's bar without
   checking the saved `k=2` number against `0.9×ceiling` by hand.

Neither issue blocks the launcher from running or corrupts the underlying
per-run JSON (all raw numbers are saved regardless), but both should be
tightened before the aggregator's advisory verdict is trusted at face value.

### B3. Run plumbing

- **Seeding — CONFIRMED CORRECT.** `one_run` calls `torch.manual_seed(seed)`
  (global RNG) immediately before `make_model(...)`; `BindingEncoder`'s
  submodules (`nn.Linear`, `nn.TransformerEncoderLayer`, the `row_queries`
  `nn.Parameter(torch.randn(...))`, `nn.MultiheadAttention`, `nn.LayerNorm`)
  are all constructed on CPU using the default global generator before
  `.to(device)` moves weights — so model init is reproducibly seeded.
  `train(..., seed=seed)` additionally builds its own `torch.Generator(...)`
  objects for data sampling (`gen`) and held-out eval (`eval_gen`, seed+10000)
  — these are independent generator *objects*, not the global stream, so data
  sampling is decoupled from however many global-RNG draws model init
  consumed. No correctness bug.
- **`train()`'s return dict — CONFIRMED matches what `aggregate()` needs.**
  For `force_rank_k=None` runs: `K, d, n_query, model, orthogonal,
  force_rank_k` + `_recovery_stats` fields (incl. `recovered_frac@0.99`) +
  `effective_rank_mean`, `effective_rank_std`, `stable_rank_mean` +
  `rankk_curve` (present because `force_rank_k is None` gates that branch in
  `evaluate()`). For forced runs, same base fields minus `rankk_curve` —
  exactly what `aggregate()`'s M1/M3 paths consume; the M2 path only ever
  looks at the `unconstrained` list, so the absence of `rankk_curve` on
  forced runs is correctly never dereferenced.
- **K=8 unconstrained run produces `rankk_curve` — CONFIRMED.** The M1 loop
  calls `one_run(..., force_rank_k=None)` for every `K` in
  `UNCONSTRAINED_KS` including `K=8`; `evaluate()`'s `if force_rank_k is
  None:` branch is keyed only on the force-rank argument, not on `K`, so the
  `K=8` unconstrained run does compute and return `rankk_curve` as required
  by the M2 aggregation path.
- **[MINOR] File-handle leak — CONFIRMED, real, and a regression vs. its own
  sibling script.** `run_stage1.py:136, 144, 147`:
  ```python
  json.dump(r, open(f"{args.out_dir}/uncon_K{K}_s{seed}.json", "w"), indent=2)
  json.dump(r, open(f"{args.out_dir}/force{k}_s{seed}.json", "w"), indent=2)
  json.dump(report, open(f"{args.out_dir}/AGGREGATE.json", "w"), indent=2)
  ```
  `open(...)` is never closed or used as a context manager. On CPython this
  is usually masked by prompt refcount-based GC, but it's a real anti-pattern:
  no guaranteed flush/close on exception mid-write (risk of a truncated JSON
  file if the process is killed/OOMs right after one of these lines), and an
  unnecessary FD leak across ~18 runs. Notably, `run_task_d.py:285-286`
  already does this correctly (`with open(args.out, "w") as f: json.dump(...)`)
  — `run_stage1.py` should match that pattern. **Fix:**
  ```python
  with open(f"{args.out_dir}/uncon_K{K}_s{seed}.json", "w") as f:
      json.dump(r, f, indent=2)
  ```
  (and same for the other two call sites).
- **[MODERATE] No try/except around the sweep loop — violates the project's
  own explicit hard rule and is a real risk to ~18 sequential GPU runs.**
  `main()`'s two `for` loops call `one_run(...)` with no exception handling.
  A single crashing config — e.g. the NaN/Inf `FloatingPointError` that
  `train()`'s per-step grad guard can raise (round-2-correctness Part A3), or
  a CUDA OOM — kills the entire remaining Stage-1 sweep. Per-run JSONs
  already written to disk *before* the crash are safe (good), and Stage-1 is
  cheap to re-run (~5–10 GPU-h total per §7), but this directly matches
  `CLAUDE.md`'s standing instruction: *"Sweep experiments... Add try/except
  so one crash doesn't kill remaining configs."* **Fix (cheap, recommended
  before launch, not launch-blocking):** wrap each `one_run(...)` call in
  `try/except Exception as e: print(f"FAILED: ...{e}"); continue`, and have
  `aggregate()` tolerate missing `(K,seed)`/`(k,seed)` cells (it mostly
  already does via `mean([])→nan`, but the M2 path's `curves[0]` would
  `IndexError` if *both* `K=8` unconstrained runs failed — worth a guard).
- **[LOW / informational]** `TAU = 0.99` is redefined as an independent
  literal in `run_stage1.py` rather than imported from `run_task_d.TAUS[-1]`.
  Currently consistent (verified above), but a future edit to one file's
  constant without the other would silently produce a `KeyError` (loud, not
  silent) rather than a silent mismatch — low risk, not worth blocking on.

---

## Round-2 status carried forward (unchanged by round-3, for completeness)

| Item | Status |
|---|---|
| `task_d.py` `torch.eye(K)` no device in `_self_test` | Still latent/inert (CPU-only self-test), non-blocking |
| NaN-only→NaN/Inf guard costs 2 syncs/param/step | Still unresolved, throughput-only, non-blocking |
| `unbind` `isinstance(..., staticmethod)` guard not added alongside signature check | Still open, MINOR future-regression gap |
| Round-2-validity Finding D (`eigh` backward near-degenerate risk under real `force_rank_k` training) | Still unresolved by any round; genuinely needs the H100 run — monitor `σ_k−σ_{k+1}` gap and grad-norm at low `force_rank_k`, per round-2's original recommendation |
| Preregistration §6 "(K−1)/K" arithmetic | Already corrected in the current `TASK_D_PREREGISTRATION.md` to the verified `√((K−1)/K)` form — confirmed while re-reading §6 for this audit, no action needed |

---

## Verdict

**Both round-2 fixes are correct.** The Haar sign-correction scales the right
axis (orthonormal columns, pre-transpose) and removes the documented bias; the
resolution pre-flight now asserts the literal `recovered_frac@0.99` criterion
with a mathematically sound and empirically wide margin — its only residual
uncertainty (GPU `eigh` tie-break on an exactly-degenerate spectrum) is a
genuine runtime unknown, not a static defect, and is flagged for the first
smoke run to watch, not block on. All three named minor fixes (n_query
surfaced, vector+force-rank-k warning, cleaned pre-flight) are present.

**`run_stage1.py` is structurally correct** — key types, seeding, the
`train()`/`evaluate()` contract, and the FORCE_RANKS/UNCONSTRAINED_KS grid all
match the pre-registration exactly, and M2's knee extraction is a faithful
implementation of §6. It has one real regression (unclosed file handles,
trivial fix) and one real process-risk (no try/except around the sequential
sweep, contradicting the project's own hard rule — cheap fix, recommended
before the actual Stage-1 launch though not before the smoke test). Its
biggest substantive gap is that `m1_trends_up` is a materially weaker,
non-literal proxy for the pre-registered M1 CONFIRM criterion (no Spearman ρ,
no `[0.7K,1.3K]` range check) — the "advisory" framing in the docstring covers
this honestly, but a human should not treat a launcher "PASS" as the real M1
CONFIRM without recomputing it from the saved per-K JSONs, and it's a cheap
fix to close properly.

**Is the full pipeline correct enough to run on the H100? Yes.** Run
`python run_task_d.py --smoke` first as already planned — the two watch-items
are step `[2]`'s degenerate-spectrum backward and step `[6]`'s `frac_tr`
value (should be ≈0, not just <0.5). Then `run_stage1.py` is safe to launch
as-is; the try/except and file-handle fixes are cheap and worth 10 minutes
before the real Stage-1 run, but nothing found here is a correctness blocker
that would invalidate the resulting numbers.
