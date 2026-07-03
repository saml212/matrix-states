# Adversarial Validity Audit — Task E (Compositional Multi-Hop Relational Recall)

**Scope.** `NEXT_EXPERIMENT_DESIGN.md`, `task_e.py`, `model_e.py`, `run_task_e.py`,
against the Task D substrate (`task_d.py`, `model_v4.py`) they build on. Question:
not "does it run" but "would a PASS or FAIL be scientifically interpretable as
evidence for/against H_E (does the rank-K matrix Z compose at unseen hop depths)."

**Method note.** No `torch` is installed on this dev machine. Two of the findings
below (Finding A and Finding B) are load-bearing enough that I verified them
empirically rather than by inspection alone: I installed `numpy` into the
scratchpad (`pip install --target .../scratchpad/pylibs numpy`) and reproduced
the *exact* control flow of `_assert_injective` / `_test_injectivity_guard_detects_merge`
(Finding A) and the *exact* random-permutation sampling `_permutation_graph`
performs (Finding B) using `numpy` as a drop-in for the relevant `torch` calls
(`torch.linalg.matrix_rank` → `np.linalg.matrix_rank`, `argsort`-of-noise
permutation → `rng.permutation`). Both reproduce standard, seed-independent
combinatorial/linear-algebra facts, so the numpy substitution does not change
the conclusions — this is not a torch-specific numerical quirk.

---

## Verdict (read this first)

**As written, Task E would NOT yield an interpretable go/no-go.** Two FATAL
findings, independent of each other:

1. The experiment **cannot currently run at all** — the mandatory Stage-1 smoke
   gate crashes on its first check (`task_e.py::_self_test`), before any of the
   later composition-purity / blank-out / C_MLP checks execute. This is
   verified, not suspected (§Finding A).
2. Even if the smoke gate is patched to pass, **the primary measurement (M3_E)
   is confounded by cycle-length periodicity of the sampled permutation**: a
   large, K-dependent fraction of nominally "held-out-depth" queries are, by
   elementary group theory, computationally identical to in-distribution or
   trivial-identity queries. This is not a hypothetical — I measured the
   collapse fraction by direct simulation of the actual generator across the
   entire planned Stage-2 sweep grid (§Finding B). At the primary operating
   point (K=8), the `H_extra=8` "further-out probe" the design doc specifically
   added to check "graceful degradation vs. hard cliff" (§6) is **useless by
   construction**: 100% of queries at h=8 reduce to an already-trained or
   trivial hop, for *every* possible cycle length. At K=4 (in the planned Stage
   2 grid), **all five held-out hops (4,5,6,8,10) collapse 100% of the time** —
   M3_E at K=4 measures nothing but periodicity.

Both are fixable (concrete fixes below) and neither indicts the paper's other
strengths — the literal-Zʰ composition, the blank-out/purity test design, and
the pinned (non-argmax, non-MLP) readout are all sound as designed (Findings
2, 3, 5 below fire clean). But as shipped, a PASS on M3_E cannot be
distinguished from "SGD learned to get short cycles right," and a FAIL cannot
be trusted either, because the smoke gate that would certify the harness is
correct has never successfully completed.

---

## Finding A — Injectivity guard (C6) fails its own negative unit test; blocks the entire smoke gate

**Hunt item 1.** SEVERITY: **FATAL** (both operationally and epistemically).

`task_e.py::_assert_injective`:

```python
def _assert_injective(values: torch.Tensor) -> None:
    K = values.shape[1]
    K_eff = min(K, values.shape[-1])
    vrank = torch.linalg.matrix_rank(values[0].float())
    assert vrank >= K_eff - 1, ...
```

The design doc (§2, §5 C6, §11) claims this check is "load-bearing," is
"asserted at generation time... a required smoke-gate addition, not an
optional nicety," and — critically — that `_test_injectivity_guard_detects_merge`
"confirms the check detects" a deliberately constructed merge, proving the
guard "has actual discriminating power (not a vacuous pass)."

**This claim is false as implemented.** The `-1` slack means the check only
fires when rank drops by **two or more**. A single merge (two of K entities
mapping to the same value — the exact MNNS/B-4 failure mode this machinery
exists to catch) drops rank from K to exactly K-1, which satisfies
`vrank >= K_eff - 1` and passes silently. I confirmed this by running the
exact logic `_test_injectivity_guard_detects_merge` executes (numpy substitute
for the linear algebra, identical control flow):

```
rank after single merge: 7   (K_eff - 1 = 7)
_assert_injective check: vrank >= K_eff - 1 -> True  (merge NOT detected)
_test_injectivity_guard_detects_merge: FAILED with AssertionError:
  injectivity guard FAILED to detect a merged (non-injective) value set
```

So `_self_test()` — called as smoke-gate step [1] by `run_task_e.py::smoke()`,
and as `task_e.py`'s own `__main__` — **raises an uncaught `AssertionError` and
aborts**, before reaching the chain-variant's own checks are fine, but the
*negative* test at the end of `_self_test()` crashes the whole function. This
means `run_task_e.py --smoke` steps [2]–[8] — the MLP-forward/backward check,
the `force_rank_k` check, the blank-out/composition-purity check, and the
"M3_E metric is resolvable" pre-flight — **have never been exercised end to
end**. Per `CLAUDE.md`'s own hard rule ("smoke test every model... before
training"), Task E is not currently smoke-clean; nothing downstream of this
line has been empirically verified to work, only reviewed by inspection.

**Nuance on blast radius.** The *generators themselves* (`_permutation_graph`
via a true bijection; `_chain_graph` via forward-only edges over a topological
order) are injective by a separate, sound combinatorial argument independent
of this runtime check — a bijection cannot merge, and `order[i]→order[i+1]`
edges have distinct sources and distinct targets by construction. So this bug
has probably not silently corrupted any data *yet*. But the check exists
precisely as a regression guard against a *future* generator bug (e.g. a
Stage-2 refactor of `_chain_graph`'s "extra edge" sampling, or a new
K/N/variant combination) — and as shown, it would not catch the single-merge
case, which is the realistic failure mode (an accidental off-by-one or
duplicate index), not a two-or-more-merge case. The safety net has a
person-sized hole in exactly the place it needs to hold weight.

The identical pattern (`vrank >= min(K, d) - 1`) already exists in
`task_d.py::_self_test`, inherited from there. In Task D it is a genuinely
harmless numerical-tolerance idiom (Gaussian near-orthogonal keys, no claim of
"this check has discriminating power against a constructed adversarial
example"). Copy-pasting it into Task E's stricter context — where the doc
explicitly claims a proven negative-test guarantee — turned a harmless
tolerance into a false assurance.

**Fix.**
- Change `_assert_injective` to `assert vrank >= K_eff` (no slack) or, if
  floating-point margin is genuinely needed for `orthogonal=False` mode, pass
  an explicit numerical tolerance to `torch.linalg.matrix_rank(..., rtol=...)`
  rather than subtracting 1 from the required integer rank — conflate
  "floating-point conditioning" and "count of structural violations" no
  further.
- Re-run `_self_test()` after the fix and confirm it reaches the final
  `print("task_e self-test PASSED")` line before relying on any of it.
- Apply the same fix to `task_d.py::_self_test`'s analogous check for
  consistency, even though its stakes are lower there.

---

## Finding B — Cycle-length periodicity confounds M3_E's "held-out hop" claim (permutation variant)

**Hunt item 4 (primary), with consequences for items 3 and 6.**
SEVERITY: **FATAL** for M3_E as currently measured and configured.

The permutation variant's `π` (`_permutation_graph`) is sampled as a **general
uniform random permutation** of the K-entity subset — i.e., it can and
typically does decompose into *several* cycles of varying, generally short,
length, not one K-length cycle. This matters because `Zʰ`, applied to an
entity `a` belonging to a cycle of length `ℓ`, is periodic with period `ℓ`:
`π^h(a) = π^{h mod ℓ}(a)`. So a query nominally at "held-out depth" `h ∈
H_test ∪ H_extra` is, for any entity whose own cycle is short, secretly
equivalent (in terms of what the *ground-truth relation* actually requires)
to an in-distribution or even trivial-identity query.

**This is not hypothetical — I simulated the actual generator's sampling
procedure** (uniform random permutation, matching `_permutation_graph`'s
argsort-of-noise trick) across 100k–200k trials at every K value in the
planned Stage-2 sweep (`NEXT_EXPERIMENT_DESIGN.md` §6, §8: K ∈ {4,8,12,16}),
and computed the fraction of queries at each held-out h whose *effective* hop
(`h mod cycle_length`) lands on 0 (identity) or an in-distribution H_train
value:

```
K= 4 -> h=4:1.00  h=5:1.00  h=6:1.00  h=8:1.00  h=10:1.00
K= 8 -> h=4:0.50  h=5:0.63  h=6:0.75  h=8:1.00  h=10:0.88
K=12 -> h=4:0.33  h=5:0.42  h=6:0.50  h=8:0.66  h=10:0.75
K=16 -> h=4:0.25  h=5:0.31  h=6:0.37  h=8:0.50  h=10:0.56
```

(Sanity check on the mechanism: the cycle length containing a uniformly
random element of a uniform random permutation of size K is itself uniform on
{1,...,K} — a standard combinatorial fact, confirmed empirically at K=8 to
within 0.1% per length bucket.)

Consequences, concretely:

- **At the pre-registered primary operating point (K=8, §6), the `H_extra=8`
  "further-out probe"** — added specifically "to check whether degradation is
  graceful or a hard cliff" (§6) — **collapses 100% of the time**, for every
  possible cycle length 1–8 (verified by direct enumeration: `8 mod ℓ ∈
  {0,1,2,3}` for every `ℓ ∈ {1,...,8}`, which is not a coincidence — it falls
  out of `H_extra` literally equaling `K`). This probe point carries **zero**
  information about extrapolation. It was chosen without cross-checking
  against `K`.
- **At K=4** (also in the planned Stage-2 grid, §8), **every held-out hop
  collapses 100% of the time**. An M3_E run at K=4 would measure pure
  periodicity, not composition, at any hop tested.
- Even at the largest planned K=16, 25–56% of "held-out" queries are secretly
  in-distribution or trivial. The confound shrinks with K but never vanishes
  while `π` is a general permutation (a fixed-point/identity entity exists
  with probability exactly `1/K` regardless of K).

**Why this makes a PASS uninterpretable.** M3_E's CONFIRM criterion (§6.i) is
"held-out accuracy significantly above chance AND the C_MLP floor." A model
that has only ever learned to correctly represent *short* cycles (which is a
much easier sub-problem — low-order rotations are easier to fit from
H_train={1,2,3} supervision than a genuinely 7- or 8-order rotation) could
clear this bar on the *pooled* metric without ever having learned to compose
correctly at genuinely unseen depth for the *long*-cycle entities the H_E
hypothesis is actually about. The aggregate `recovered_frac@0.9` at h=6, for
example, is a 75%-periodicity / 25%-genuine mixture at K=8 — a model could
plausibly get all the periodicity-favored queries right and half the genuine
ones and still clear a naive significance bar.

**Why this also weakens the FALSIFY side.** If a real model shows apparent
"graceful decay then a cliff" as h grows through {4,5,6,8,10}, that curve
shape is exactly what you'd expect from *pure periodicity dilution* even with
zero genuine long-range composition ability (more queries hit unlucky long
effective distances as h grows and coprimality with short cycles becomes less
likely) — so the curve shape itself, which §6.ii treats as diagnostic
("degrades gracefully... not a hard cliff"), cannot currently distinguish
"genuine graceful eigenstructure decay" from "an artifact of which fraction of
queries happen to land on short cycles at each h." Both outcomes are
consistent with the same observed curve.

**Secondary consequence for the C_MLP control (hunt items 3 & 6).** C_MLP
predicts a single deterministic `f(flatten(Z), key_a)` for *all* held-out h
(one-hot(h) is all-zero/OOV, so it cannot distinguish h=4 from h=6 for the
same query). Normally this should force it to a low floor. But for a
short-cycle entity where multiple held-out h's share the same true target
(e.g., `ℓ=2` ⇒ h=4,6,8,10 all have target = `a` itself), C_MLP's single fixed
guess can be simultaneously "correct" for several held-out h's purely because
the *targets themselves* coincide under periodicity — not because it learned
anything about h. This means C_MLP is not a clean "rank-blind, h-blind"
floor; it inherits the same periodicity benefit as the primary model,
compressing the gap the M3_E-vs-C_MLP comparison is supposed to measure, in
either direction.

**Tie-in to C8 (eigen_utils).** `eigen_utils.py`'s own self-test motivates the
eigenvalue-fidelity metric using a **single pure K-cycle** ("for a pure K-cycle
permutation these [eigenvalues] are exactly the K-th roots of unity" — §5 of
the design doc, reproduced almost verbatim in the self-test). But the actual
generator (`_permutation_graph`) does not produce pure K-cycles by default —
it produces general permutations with multiple, generally-short cycles. So
the clean "roots of unity" interpretive story that motivates C8 doesn't
describe the typical sampled instance; the eigenvalue-fidelity number reported
alongside M3_E is averaging over a mixture of different-order sub-rotations
whose expected spectral structure isn't the thing C8's own justification
describes.

**Fix (design + code, in order of preference).**
1. **Force `π` to be a single full K-cycle** (uniformly random Hamiltonian
   cycle on the K-subset, not a general permutation) in the permutation
   variant. This is a small change to `_permutation_graph` (sample a random
   ordering of the K indices, chain them into one cycle) and has two
   benefits: it removes the "many small independent cycles" escape hatch
   entirely (periodicity now only occurs at multiples of K itself), and it
   makes the sampled data match the roots-of-unity story C8 is built around.
2. **Choose H_test/H_extra so they are never ≡ 0 mod K and never ≡ any
   H_train value mod K** for the K's actually used — after fix (1), this
   reduces to picking H_extra values that are not multiples of K and not
   within `max(H_train)` of a multiple of K. At minimum, replace the current
   `H_extra=(8,10)` default, which is self-defeating at K=8 (literally `K`
   itself) and at K=4 (both 8 and 10 are multiples of 4).
3. Even with (1)+(2), a clean design should **stratify M3_E by the true cycle
   length / effective distance from H_train (`h mod ℓ` after fix 1, or the
   raw `h` before it), not just raw nominal h** — report accuracy on the
   subset of queries where genuine composition depth (distance from the
   nearest periodicity-equivalent in-distribution hop) actually exceeds
   H_train, alongside the pooled number. This is the only way to make a PASS
   mean "the model composes at genuinely unseen depth" rather than "the model
   is good at the queries that happen to be easy this batch."
4. Cheapest interim mitigation if (1)–(3) can't land before Stage 2: increase
   K well beyond the planned grid's low end (drop K=4 from the sweep; it is
   uninterpretable as shown above) and report, per eval batch, the measured
   fraction of periodicity-collapsed queries as a mandatory diagnostic next to
   every M3_E number, so readers can see how contaminated each number is.

---

## Finding C — remaining hunt items (clean or minor)

**Hunt item 2 — literal Zʰ composition.** SEVERITY: N/A (fires clean).
Traced every path from raw bindings to `pred`: `BindingEncoder.forward` (from
`model_v4.py`, reused verbatim) only ever sees `(keys, values)`, never `h` or
the query. `MatrixCompositionModel.compose` is a `@staticmethod` taking only
`(Z, query_keys, hops)`, implements literal iterated `einsum("bij,bqj->bqi",
Z, cur)` with the *same* `Z` reused every step (no re-derivation, no
re-attention over raw tokens, no learned per-hop weight). No hidden pathway.
This part of the design is sound as written. (Caveat: this has been *reviewed*
but not yet *run*, per Finding A — smoke step [4]/[6], which would exercise
this at runtime, never executes.)

**Hunt item 3 — pinned readout / shortcuts on the headline model.**
SEVERITY: MINOR on `MatrixCompositionModel` itself (fires clean: no argmax, no
free-side `Z=uv₀ᵀ` construction, no MLP headroom — `n_model == n_encoder` is
explicitly asserted in the smoke gate to rule out hidden per-hop parameters).
The one real issue under this heading is C_MLP's periodicity contamination,
folded into Finding B above (not a flaw in the headline model, but it weakens
the control that's supposed to bound false positives on the headline model).

**Hunt item 5 — bottleneck / blank-out.** SEVERITY: N/A (fires clean, by
inspection). The smoke gate's blank-out check (grad flows from `keys` to
`pred` through `Z`, but is `None` when `Z` is detached to a leaf) plus an
independent per-(row, query) manual-loop cross-check against `compose()`'s
own output is a genuinely good, meaningful test — appropriately extended
from Task D's original blank-out test to also confirm h-purity (no
cross-query leakage, exactly `h` sequential matmuls). As with Finding
C/item 2, this has not yet been *executed* due to Finding A.

**Hunt item 6 — success without mechanism.** Covered by Finding B (the
dominant vector found). Other candidate shortcuts checked and ruled out:
small-vocab memorization (entities are continuous, freshly QR-sampled per
batch — no fixed vocabulary), positional shortcuts (no token-position
channel into `Z`; encoder is a permutation-invariant set encoder over
bindings, reused from the already-audited Task D), chance-level cosine
matching at `d=16` (astronomically low, established in Task D's own
calibration and reused here unchanged).

**Minor residual note.** `_assert_injective` only checks `values[0]` (row 0)
per call — reasonable given the stated rationale (injectivity here is a
structural property of the code, not a per-row probabilistic one), but this
reasoning is only as good as the `-1` tolerance is tight; once Finding A is
fixed, spot-checking row 0 becomes a meaningful guard again. `K_eff = min(K,
values.shape[-1])` is currently dead code — `TaskEConfig.__post_init__`
already guarantees `K <= N <= d` for both variants, so `K_eff == K` always in
practice; harmless, but worth knowing it isn't doing anything.

---

## Summary table

| # | Item | Fires? | Severity | Uninterpretable? |
|---|------|--------|----------|-------------------|
| 1 | Injectivity guard / MNNS trap | **Yes** | **FATAL** | Blocks the smoke gate entirely; nothing downstream verified to run |
| 2 | Literal Zʰ composition (no re-attention) | No | — | Clean by inspection; unexecuted by Finding A |
| 3 | Pinned readout shortcuts on headline model | No (headline); Yes (C_MLP control) | MINOR / MAJOR (via B) | C_MLP's floor is softened by periodicity, weakening the M3_E comparison |
| 4 | Unseen-hop-depth generalization (train/eval split) | **Yes** | **FATAL** | Label-level split is fine; effective computational split is not — M3_E pools genuine and periodicity-trivial queries indistinguishably |
| 5 | Bottleneck / blank-out | No | — | Well-designed test; unexecuted by Finding A |
| 6 | Success without mechanism | **Yes** (= Finding B) | **FATAL** | Periodicity is exactly this: high M3_E score achievable via short-cycle competence alone |

---

## Overall verdict

**Not yet interpretable, and not yet run.** Two independent FATAL issues:
(A) the harness cannot currently pass its own mandatory smoke gate (verified
by reproducing the exact crash), so no PASS or FAIL has actually been
produced yet despite the code being "freshly built" and reviewed; and (B) even
once smoke is fixed, the primary metric M3_E as configured (general random
permutation, `H_extra` chosen without checking it against `K`) is
substantially confounded by cycle-length periodicity across the *entire*
planned Stage-2 sweep, most severely at the low end (K=4: total confound;
K=8 primary operating point: `H_extra=8` probe is dead, H_test 50–75%
confounded) and still materially at the high end (K=16: 25–56% confounded).

Everything else audited — the literal-Zʰ readout, the composition-purity /
blank-out test, the pinned (non-argmax, non-MLP) headline readout, the C7
idealized-Z ceiling, and the chain-into-sink variant (which, note, is *not*
subject to Finding B at all — it's acyclic, no periodicity possible, and is
arguably the more valid test of the two despite being labeled "secondary") —
is sound as designed. This is a fixable, not a fatal-to-the-hypothesis,
situation: fix Finding A (tighten the rank check, confirm `_self_test`
actually reaches its PASSED line), fix Finding B (force single-K-cycles,
re-pick `H_extra` relative to `K`, stratify M3_E by effective composition
distance, drop K=4 from the sweep), re-run the smoke gate to completion, and
only then proceed to Stage 0/2 training runs.
