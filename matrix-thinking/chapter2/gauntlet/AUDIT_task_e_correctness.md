# AUDIT: Task E code correctness (task_e.py, model_e.py, run_task_e.py, eigen_utils.py)

**Auditor:** fresh-context adversarial static audit (no torch available locally; no
execution). Cross-checked against `NEXT_EXPERIMENT_DESIGN.md` and the Task D
substrate (`task_d.py`, `model_v4.py`, `rank_utils.py`, `run_task_d.py`).

**Method note:** the four files under audit were mid-edit when this audit started
(sizes/mtimes changed between the first and second read — an `assert_injective`
perf toggle, an MLP-contiguous-H_train CLI guard, and a chain-variant feasibility
preflight were added). Everything below is audited against the **final, stable**
versions (confirmed unchanged on a second pass). All `ast.parse` checks pass; all
imports are `torch` + stdlib + same-directory siblings only (no `src/` imports) —
verified by walking the AST, not by inspection.

---

## FATAL — will crash the smoke gate, and defeats the load-bearing C6 check

### F1. `_assert_injective`'s rank threshold is one too loose to catch a single merge

`task_e.py:236-241`:
```python
K = values.shape[1]
K_eff = min(K, values.shape[-1])
vrank = torch.linalg.matrix_rank(values[0].float())
assert vrank >= K_eff - 1, ...
```

A single accidental merge (two sources → one target, in-degree 2 for one entity)
reduces the rank of the stacked value set by **exactly 1** (one row becomes a
linear combination — here an exact duplicate — of another). That means a merged
value set has `vrank == K_eff - 1`, which **passes** `vrank >= K_eff - 1`. The
check as written cannot detect the single most likely, most realistic failure
mode it exists to catch — the "-1" slack and the merge-detection contract are
mutually incompatible.

This isn't a theoretical concern; the codebase carries its own negative test for
exactly this, and it fails. `task_e.py:331-350`:
```python
def _test_injectivity_guard_detects_merge() -> None:
    ...
    values[:, 1, :] = values[:, 0, :]          # force a merge: keys 0 and 1 -> same value
    raised = False
    try:
        _assert_injective(values)
    except AssertionError:
        raised = True
    assert raised, "injectivity guard FAILED to detect a merged (non-injective) value set"
```

**Verified independently** (pure-Python Gaussian-elimination rank, no numpy/torch
needed, since neither is available in this sandbox):

```
rank before merge: 8   K: 8
rank after single-pair merge: 7
assert vrank >= K_eff - 1 (current code): True  -> does NOT raise
assert vrank >= K_eff     (proposed fix): False -> DOES raise
```

So `_assert_injective(values)` does not raise on the deliberately-merged input,
`raised` stays `False`, and the test's own final `assert raised, ...` fires —
**`task_e._self_test()` crashes with `AssertionError` before printing "task_e
self-test PASSED"**, and therefore **`run_task_e.py --smoke` crashes at step
[1]**, before any other check runs.

Beyond the crash: this is exactly the merge/miscounting attack the design doc
calls "the strongest attack found" and marks C6 as mandatory/load-bearing to
close (`NEXT_EXPERIMENT_DESIGN.md` §2, §5, §11). As implemented, the guard would
silently pass a real, single-element merge in production if a future edit to
`_permutation_graph`/`_chain_graph` ever introduced one — it currently only
"works" because the generators are structurally guaranteed injective by
construction, making the runtime check a no-op safety net rather than the
"asserted at generation-time... not just entries" guarantee the design promises.

**Fix:** tighten to `assert vrank >= K_eff` (verified safe for genuine,
non-merged batches — orthonormal-QR-constructed values have `vrank == K_eff`
exactly, confirmed above). Do **not** reintroduce a `-1` slack "for floating
point safety" — it was almost certainly copied from `task_d.py:155`'s benign
general sanity check (which never has to survive a dedicated single-merge
negative test), and that assumption doesn't transfer here. Applies to both the
permutation (`task_e.py:292-293`) and chain (`task_e.py:308-309`) call sites,
and ideally to the redundant inline rank checks inside `_self_test` itself
(`task_e.py:373`, `396`) for consistency, though those aren't the ones that
crash.

**Confidence:** high without execution — verified via an independent from-scratch
rank computation (Gaussian elimination), not just symbolic reasoning. Still worth
a 10-second on-cluster confirmation that `torch.linalg.matrix_rank`'s default
tolerance on GPU/float32 doesn't do anything surprising after the fix.

---

## MAJOR

### M1. Output JSON omits the run configuration — blocks a downstream orchestrator

`run_task_e.py:365-374` (`main()`):
```python
result = train(model, cfg, device, args.model, steps=args.steps,
               batch_size=args.batch_size, lr=args.lr,
               force_rank_k=args.force_rank_k, seed=args.seed)
result.update({"n_params": n_params, "seed": args.seed, "steps": args.steps})
```
`train()` itself only returns `{"n_skipped_steps", "M2_in_distribution",
"M3_held_out"}`. Unlike Task D's `evaluate()` (`run_task_d.py:98-99`, which
embeds `K, d, n_query, model, orthogonal, force_rank_k` directly into every
result dict), Task E's `--out` JSON never records `K`, `d`, `variant`, `H_train`,
`H_test`, `H_extra`, `orthogonal`, `force_rank_k`, or `model_type` — only
`n_params`, `seed`, `steps`. A directory of Task E sweep result files is
unparseable on its own; a future orchestrator (or you, in three weeks) would
have to re-derive the config from stdout logs or a filename convention.

**Fix:** in `main()`, before dumping:
```python
result.update({
    "model_type": args.model, "variant": args.variant, "d": args.d, "K": args.K,
    "N": cfg.pool_size, "H_train": list(cfg.H_train), "H_test": list(cfg.H_test),
    "H_extra": list(cfg.H_extra), "orthogonal": args.orthogonal,
    "force_rank_k": args.force_rank_k, "n_params": n_params,
    "seed": args.seed, "steps": args.steps,
})
```
(Point 7 of the audit brief: this is exactly the "future orchestrator" concern —
current state fails it.)

### M2. Smoke gate never exercises the model's *own* learned Z at extrapolated hops — the exact numerical-explosion risk the design flags is untested

The audit brief explicitly asks whether repeated matmul over h hops can
explode/vanish/NaN and whether there's eigenvalue control. There is **no**
normalization, clipping, or spectral-radius regularization anywhere in
`MatrixCompositionModel`/`BindingEncoder` — `compose()` (`model_e.py:57-82`) is
literally `h` un-normalized matmuls of whatever Z SGD happens to produce, and
nothing in the training objective (which only ever sees `h <= 3`, per
`cfg.H_train`) penalizes a Z with spectral radius meaningfully above 1. At eval
time (`evaluate_at_hops`, called from `train()` for `M3_held_out`), the model's
*actual* Z is genuinely raised to h=8 and h=10.

Smoke step [8] (`run_task_e.py:263-277`, "RESOLUTION pre-flight") tests
`compose()` only against `z_ideal` (provably spectral-radius-1 by construction)
and a fresh `torch.randn_like` `Z_broken` (well-conditioned, spectral norm
~2√d ≈ 8 for d=16) at a single held-out hop (h=5). **It never calls the actual
`model(...)` forward pass at any hop beyond `max(H_train)=3`** — steps [4]/[5]
only build `b` with `hop_set=cfg.H_train`. So a poorly-scaled encoder output
(nothing bounds `BindingEncoder.row_out`'s output magnitude) could produce
Inf/NaN at h=8/10 and this would not surface until a real, expensive training
run reaches the M3_E eval — silently, too: `recovery_cosine`'s normalization
(`pred / pred.norm(...).clamp(min=1e-8)`) turns `Inf/Inf` into `NaN`, and
`NaN > 0.9` is `False` in IEEE comparisons, so `recovered_frac@0.9` would just
read as a quiet 0 with no signal that the cause was numerical blowup rather than
"the model didn't learn to compose."

**Fix (add to smoke, cheap):** after step [4], forward the (randomly
initialized, and ideally after a handful of dummy training steps) model at
`hop_set=(max(cfg.H_extra),)` and assert `torch.isfinite(pred).all()`. Separately,
consider logging `torch.linalg.matrix_norm(Z, ord=2)` (spectral norm) alongside
`effective_rank`/`stable_rank` in `evaluate_at_hops` as a cheap always-on
diagnostic — it directly explains any M3_E cliff/NaN that shows up later and
costs one extra SVD per eval batch.

**Runtime-confirmation item:** whether trained Z in practice stays well-scaled
(self-regularizing via the bounded h≤3 training signal) is an empirical question
this audit cannot answer statically — flag for first on-cluster run: watch
`n_skipped_steps` and the M3_E `mean_cos`/`ideal_mean_cos` gap at h=8,10 for
NaN or a sudden discontinuity vs. h=4-6.

---

## MINOR

1. **`eigenvalue_fidelity` output tensor is always CPU** (`eigen_utils.py:113`):
   `out = torch.zeros(B, dtype=torch.float32)` has no `device=`, so it silently
   returns a CPU tensor regardless of `Z`'s device. Harmless today (only
   consumed via `torch.cat` within its own homogeneous list, then `.item()`'d,
   under `@torch.no_grad()`), but inconsistent with the rest of the file and a
   latent trap if ever combined with a GPU tensor list. Fix: add
   `device=Z.device`.

2. **`MLPShortcutModel._one_hot_h`'s OOV logic checks a range, not set
   membership** (`model_e.py:133`: `in_vocab = (hops >= 1) & (hops <=
   self.h_train_max)`). A non-contiguous `H_train` (e.g. `{1,3}`) would silently
   mark a *skipped, untrained* hop (e.g. `h=2`) as in-vocabulary. This is now
   **mitigated at the CLI layer** — `run_task_e.py:319-327` refuses
   `--model mlp` with a non-contiguous `--h-train` and exits with a clear error
   — but the underlying `MLPShortcutModel` class itself has no such guard if
   instantiated directly (future script/notebook/orchestrator bypassing the
   CLI). Consider moving the check into `MLPShortcutModel.__init__` for
   defense-in-depth.

3. **M1_E isn't a distinct output key.** "Learned effective rank vs K" is
   redundantly recomputed per-hop inside both `M2_in_distribution` and
   `M3_held_out` (same statistic on fresh random batches per hop-group, since Z
   doesn't depend on h). Not wrong, just means a downstream reader has to know
   to average across those entries to reconstruct M1_E rather than reading one
   key. Documentation/consistency note only.

4. **Integer hop keys get stringified by `json.dump`.** `per_hop[h] = entry`
   (`run_task_e.py:117`) uses `int` keys; `json.dump`/`json.dumps` silently
   converts dict keys to strings on serialization. Round-tripping the `--out`
   file requires the reader to know to re-cast `"4"` → `4`. Not a crash, just a
   parsing gotcha worth documenting for the orchestrator.

5. **Chain-variant CLI preflight is incomplete.** `run_task_e.py:335-351`
   checks `K < H_max`, `N-1 < H_max`, `N > d`, but not the `K < N` (strict)
   requirement `TaskEConfig.__post_init__` enforces. A user-supplied `--N <=
   --K` falls through to a less-friendly raw `AssertionError` from inside
   `TaskEConfig` instead of the nicely formatted preflight message. Low
   priority — the safety net (the assertion itself) still fires correctly.

6. **No `--n-query` CLI flag** (present in `run_task_d.py`, absent from
   `run_task_e.py`); `TaskEConfig.n_query` can only be swept by editing code.
   Not required by the design doc's measurement plan, but worth noting for
   completeness if a future orchestrator wants to vary query count.

---

## Verified correct (adversarial checks that held up)

- **Readout is the pinned linear composition, not a learned/MLP/argmax path.**
  `MatrixCompositionModel.compose` (`model_e.py:57-82`) is a `@staticmethod`
  with zero parameters; smoke step [6] structurally confirms this two ways —
  `inspect.signature(...) <= {"Z","query_keys","hops"}` and
  `n_model == n_encoder` (no params beyond the encoder) — both are genuine,
  non-vacuous checks (they'd fail if a per-hop weight were ever added).
- **The multi-hop iteration itself is correct.** `compose()`'s masked-loop
  pattern (`cur` advances every step, `result` latched via `torch.where(hops==t,
  cur, result)`) is independently re-derived and cross-checked against a
  brute-force per-(row,query) Python loop in smoke step [6]
  (`run_task_e.py:237-247`) — a real adversarial check, not vacuous.
- **Chain-variant generator arithmetic is sound.** Traced `_chain_graph`'s
  `run_len` recurrence (trailing-run-length via backward DP), the forced
  contiguous block guaranteeing a run ≥ H_max, and `_sample_valid_start`'s
  existence assertion — all internally consistent; index bounds
  (`tgt_chain_pos = chosen_pos + hops`) cannot exceed `N-1` given the run-length
  invariant. `_chain_edge_positions`' stable-sort trick correctly extracts the
  K active (source, source+1) pairs in original order.
- **`_iterate_permutation`'s batched lockstep hop advance is correct** —
  independent of and consistent with `compose()`'s pattern; the ground-truth
  label path never touches floating point (pure integer `gather`), so it's
  immune to the numerical-explosion concern in M2 above.
- **Z_ideal (C7) is provably exact and provably well-conditioned** for the
  permutation variant (spectral radius exactly 1, eigenvalues on the unit
  circle by construction) and nilpotent-safe for the chain variant — confirmed
  by `task_e._self_test()`'s cross-check against an independently
  re-implemented `_iterate_matvec` up to h=20, and by `eigen_utils._self_test()`'s
  empirical roots-of-unity check.
- **Reused utilities are genuinely reused, not re-copied-and-drifted.**
  `rank_utils.truncate_to_rank`/`effective_rank`/`stable_rank` and
  `task_d._random_directions`/`recovery_cosine` are imported directly (not
  duplicated), so Task E cannot silently diverge from Task D's already-audited
  implementations. The non-finite-grad skip/clip/`n_skipped`-report training
  loop (`run_task_e.py:141-150`) is carried over verbatim from
  `run_task_d.py:131-140`.
- **`force_rank_k` composes the SAME truncated Z across all h hops** (projected
  once in `encode()`, iterated in `compose()`, not re-truncated per hop) —
  matches the design's M4_E intent exactly (`model_e.py:51-55` docstring and
  code agree).
- **All four files parse cleanly and import only torch + stdlib + same-directory
  siblings** (verified via `ast.parse` / `ast.walk`, not inspection) — no
  `src/` imports, matching the pod-safety convention.

---

## Verdict

**Fix F1 before smoke-testing on the H100.** As written, `run_task_e.py --smoke`
will crash at step [1] (`te._self_test()`) on the very negative test that's
supposed to prove the C6 injectivity guard has teeth — a one-line threshold
change (`K_eff - 1` → `K_eff`), verified safe against genuine orthonormal
batches via an independent rank computation. This is not a "maybe" — it will
crash deterministically given the seeded self-test.

Also fix M1 (config-in-JSON) before any real sweep — cheap, and every result
file produced before the fix is otherwise orphaned from its own configuration.
M2 (extend smoke to forward the real model through H_extra) is strongly
recommended before launching Stage 2's longer 20-30K-step runs, since an M3_E
NaN discovered after a multi-hour training run is a much more expensive way to
learn the same thing a 5-line smoke addition would catch in seconds.

None of the MINOR items block a smoke run. After F1 is patched, this codebase
is correct enough to smoke-test on the H100; strongly recommend also patching
M1/M2 in the same pass since they're small and cheap relative to the compute
this is about to consume, and re-running the full `--smoke` gate once more
before Stage 2 launches.
