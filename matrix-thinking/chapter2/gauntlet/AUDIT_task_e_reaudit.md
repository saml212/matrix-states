# RE-AUDIT: Task E post-fix verification (2026-07-01)

**Scope.** Fresh-context adversarial re-audit of `task_e.py`, `model_e.py`,
`eigen_utils.py`, `run_task_e.py` after the 2 FATAL + 2 MAJOR findings from
`gauntlet/AUDIT_task_e_correctness.md` and `gauntlet/AUDIT_task_e_validity.md`
were fixed. No `torch` available locally (confirmed:
`ModuleNotFoundError: No module named 'torch'`, and no `numpy` either). Every
load-bearing claim below is backed by a from-scratch pure-Python reproduction
(Gaussian-elimination rank, `random.shuffle`-based permutation sampling, exact
re-implementations of the guard/generator control flow) rather than symbolic
reasoning alone, matching the methodology of the two prior audits.

---

## Part A — Verify each fix

### 1. Injectivity threshold: exact `>= K_eff` (no `-1` slack) — **CONFIRMED**

`task_e.py:308-320` (`_assert_injective`) now asserts `vrank >= K_eff` with an
explicit comment ruling out reintroducing slack. The inline checks inside
`_self_test()` (permutation branch `task_e.py:452`, chain branch `:475`) were
also tightened to `assert vr >= K` — the correctness audit's "ideally also fix
the redundant inline checks" recommendation was applied, not just the
function itself.

Reproduced the exact scenario `_test_injectivity_guard_detects_merge` drives
(orthonormal K=8, d=16 values, force a merge by overwriting row 1 with row
0, Gaussian-elimination rank in pure Python):

```
rank before merge: 8   K_eff: 8
rank after single-pair merge: 7
OLD check (>= K_eff-1): True  -> does NOT raise (the bug)
NEW check (>= K_eff):   False -> DOES raise (correct)
```

Also swept 200 independently-sampled genuine (non-merged) orthonormal
batches through the new exact threshold: **0 false-positive rejections** —
confirms the tightened check is safe, not just tighter. `_test_injectivity_guard_detects_merge`
will now correctly set `raised=True`, its own `assert raised` will pass, and
`_self_test()` reaches the final `print("task_e self-test PASSED")` line
(traced by inspection of the full control flow in `task_e.py:410-493`; no
other early-return/exception path exists between the two variant blocks and
this call).

### 2. Periodicity fix (single Hamiltonian K-cycle) — **CONFIRMED**

**2a. `_permutation_graph` produces one K-cycle, not a general permutation.**
Re-derived the tensor ops by hand (`order = argsort(rand)`, `next_order =
roll(order, -1)`, `succ.scatter_(1, order, next_order)` ⇒ `succ[order[i]] =
order[(i+1) mod K]`) and reproduced it in pure Python (`random.shuffle` as
the argsort-of-noise equivalent — same distribution). Swept K ∈
{1,2,3,4,8,12,16} × 500 seeds each: every sampled `succ` is (a) a bijection
(`sorted(succ) == range(K)`) and (b) decomposes into **exactly one** cycle of
length K — 0 counterexamples across 3,500 trials. The construction is
provably correct, not just empirically lucky: since `order` is itself a full
permutation traversed in sequence and consecutive elements are chained
cyclically, no shorter sub-cycle can form by construction (this matches the
module docstring's own reasoning, and the reproduction confirms it holds in
practice too).

**2b. `TaskEConfig.__post_init__`'s periodicity guard.** Re-implemented the
exact guard logic (`task_e.py:121-134`): reject any `H_test`/`H_extra` hop
with `h % K == 0` or `h % K` in `{h % K for h in H_train}`. Verified:
- Default `H_extra=(7,21)` passes the guard for **all** of K ∈ {8,12,16}
  (the full Stage-2 sweep grid) — confirmed by direct computation, not
  spot-checked: `7%8=7, 21%8=5` (K=8); `7%12=7, 21%12=9` (K=12); `7%16=7,
  21%16=5` (K=16) — none land on 0 or {1,2,3}.
- The **old** buggy default `(8,10)` is correctly rejected exactly where the
  validity audit said it was self-defeating: K=4 (`8%4=0`) and K=8
  (`8%8=0`), and correctly *passes* at K=12/16 where it wasn't actually
  broken — confirms the guard's discriminating power is real, not
  over-broad.
- **K=4 is provably rejected** under the codebase's actual default
  (`H_train=(1,2,3)`): `train_residues = {1,2,3}`, and combined with the
  `r==0` ban that's all four residues mod 4 (`{0,1,2,3}`) — brute-forced over
  `h = 1..99999` and confirmed every single value is banned. One nuance
  worth stating precisely: this is a consequence of the **default**
  `H_train=(1,2,3)` saturating all non-zero residues mod 4, not an
  *intrinsic* property of K=4 in isolation (verified: `K=4, H_train=(1,),
  H_test=(2,)` passes the guard fine). The design doc and code comments are
  careful to phrase it this way ("even under the single-K-cycle fix, K=4
  leaves only residues {0,1,2,3}... with H_train={1,2,3}"), so this is not a
  misstatement — just flagging that "provably rejected" is conditional on
  the pinned operating point, which is exactly what the codebase enforces.
- `effective_hop = h % K` (`run_task_e.py:123`) is correctly a scalar per
  hop-group entry (not a per-query distribution) — this is *sound*
  specifically because the single-K-cycle fix makes the period identical for
  every entity, eliminating the per-query heterogeneity the old
  general-permutation generator had. No stratification bug found.

### 3. Config-in-JSON — **CONFIRMED**

`run_task_e.py:416-422` now includes `model_type, variant, d, K, N,
H_train, H_test, H_extra, orthogonal, force_rank_k, n_params, seed, steps` —
matches the correctness audit's M1 fix suggestion field-for-field.

### 4. Real-Z high-h smoke step — **CONFIRMED**

`run_task_e.py:221-242` (new step **[5]**): trains a fresh
`MatrixCompositionModel` for 20 real SGD steps, then forwards it (full
`stab_model(bh)` call — the actual `encode()` → `compose()` path, not
`z_ideal` or `torch.randn_like`) through **every** hop in `cfg.H_extra`
(`(7, 21)` by default) under `torch.no_grad()`, asserting
`torch.isfinite(pred_h).all()`. This is a strict superset of the
correctness audit's minimal suggested fix (which only asked for one probe at
`max(H_extra)`).

---

## Part B — New bugs from the fixes / residual assessment

**No new FATAL or MAJOR bugs found.** The K-cycle rewrite — correctly
identified as the highest-stakes change — holds up under adversarial
re-derivation and 3,500-trial simulation: it is a true bijection, always
exactly one cycle, and the guard that depends on that invariant is logically
sound and does not over-reject (verified both a true-positive rejection of
the old confounded default and zero false rejections of the new safe
default across the full sweep grid).

Specific hunts and results:

- **Guard over-rejection / off-by-one:** none found. The modulo check is a
  single clean `r == 0` / `r in train_residues` test with no boundary
  arithmetic (`K >= 1` is asserted earlier, ruling out mod-by-zero); swept
  exhaustively for K=4 and spot-checked for K∈{8,12,16} with both old and
  new `H_extra` values with matching expected outcomes in every case.
- **Smoke-step renumbering:** verified the new step [5] is purely additive —
  prints are sequential `[1]`...`[9]` with no gaps, no duplicate numbers, and
  no step became unreachable or vacuous. Steps [6]-[9] are the old [5]-[8]
  shifted, logic otherwise byte-identical.
- **`effective_hop` stratification:** sound, see Part A.2b.
- **MLP membership change:** the smoke gate's step [8] instantiates
  `MLPShortcutModel(..., h_train=cfg.H_train)` explicitly, so it genuinely
  exercises the new exact-set-membership `_one_hot_h` path (not the old
  range-check fallback) — this is a real, non-vacuous confirmation of the
  MINOR fix, not just a code-path that's present but untested by smoke.
- **Chain-variant `K<N` preflight gap** (flagged as a residual, not fixed):
  confirmed still real. Constructed a concrete counterexample — `--variant
  chain --K 10 --N 10 --h-extra 7 8` (H_max=8) — that **passes** the CLI
  preflight (`run_task_e.py:377-393` only checks `K<H_max`, `N-1<H_max`,
  `N>d`) but then raises a bare `AssertionError: chain variant requires K <
  N` from inside `TaskEConfig.__post_init__`. Severity: low — the safety net
  still fires, just with a worse error message; this predates the current
  fix pass and was already correctly flagged as MINOR item 5 in the
  correctness audit. Unchanged status.
- **Stringified-int JSON keys** (flagged as a residual, not fixed):
  confirmed still real by direct `json.dumps({4: ..., 21: ...})` round-trip
  — keys come back as `"4"`, `"21"` (Python `str`, not `int`). Severity: low,
  a documented parsing gotcha for a downstream orchestrator, not a crash.
  Unchanged status, matches MINOR item 4 from the correctness audit.
- **Trivial, out-of-scope observation (not a bug):** `task_d.py:155` still
  carries the original `vrank >= min(K, d) - 1` slack. Both prior audits
  treated fixing this as optional/lower-priority (task_d.py has no dedicated
  merge-negative-test that would crash on it, unlike task_e.py), and it is
  outside the four files under fix in this pass. Not a regression — flagging
  only for completeness since the re-audit brief asked about consistency.
- **Trivial, not a bug:** `run_task_e.py:361-369`'s CLI-level "refuse
  `--model mlp` with non-contiguous H_train" guard is now stricter than
  necessary, since `MLPShortcutModel` itself correctly handles non-contiguous
  `H_train` via exact set membership when `h_train` is passed (which the CLI
  always does). This blocks a now-safe use case rather than causing
  incorrect behavior — belt-and-suspenders, not a defect.

---

## Runtime-only items (cannot verify without torch — flag for the H100 smoke run)

1. **Run `python run_task_e.py --smoke` end-to-end and confirm it actually
   prints "ALL SMOKE CHECKS PASSED."** Static analysis and pure-Python
   reproduction give high confidence every fix is correct, but this codebase
   has a documented history (both prior audits) of the smoke gate crashing
   on its very first check — actually executing it is the only way to be
   sure nothing else was mid-edit.
2. **`torch.linalg.matrix_rank`'s default tolerance on GPU/float32**, now
   that the check has zero slack — both prior audits flagged this as an
   outstanding runtime-confirmation item and nothing in this fix pass
   addresses it (it's inherently a numerical-backend question, not a logic
   question). Watch for any spurious `_assert_injective` failure on
   genuinely valid batches in the first real run.
3. Confirm `evaluate_at_hops`'s M2_E/M3_E numbers don't show NaN/discontinuity
   at h=7,21 during a real multi-thousand-step training run (the new smoke
   step [5] screens the *architecture*, but only after 20 steps — watch
   `n_skipped_steps` and the `mean_cos`/`ideal_mean_cos` gap in the first
   real run per the original M2 finding's runtime-confirmation note).

---

## Verdict

**All 4 required fixes are CONFIRMED**, independently verified via
from-scratch pure-Python reproduction rather than inspection alone:
injectivity threshold (exact, no false positives across 200 trials),
single-Hamiltonian-K-cycle generator (bijection + exactly-one-cycle across
3,500 trials), the config-time periodicity guard (correct accept/reject
across the full sweep grid, K=4 provably unusable under the default
operating point, old buggy `H_extra=(8,10)` correctly identified as
self-defeating exactly where the validity audit said it was), config-in-JSON,
and the real-learned-Z high-h smoke probe.

**No new FATAL or MAJOR bugs were introduced by the fixes.** The K-cycle
rewrite — the highest-stakes change, since getting it wrong would
re-confound the hop-depth-generalization claim exactly like before — holds
up under adversarial re-derivation. The two residuals (chain-variant `K<N`
preflight gap, stringified JSON keys) are pre-existing, low-severity, and
already correctly triaged as non-blocking in the prior audits; their status
is unchanged.

**Task E is now audited-clean and correct enough to smoke-test on the
H100.** Recommend running `python run_task_e.py --smoke` next and confirming
it reaches "ALL SMOKE CHECKS PASSED" before launching any Stage 0/2 training,
per the three runtime-only items above.
