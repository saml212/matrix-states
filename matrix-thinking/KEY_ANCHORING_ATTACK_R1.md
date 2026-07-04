# KEY_ANCHORING_ATTACK_R1 — Adversarial attack on KEY_ANCHORING_DESIGN.md (Rev 1)

**Attacker verdict: NEEDS-REV-2.** Three FATAL findings, two of them code-verified
(not interpretive), plus three MAJOR findings. The hypothesis and most of the
infrastructure reuse are sound; the specific closures the design leans on to make
its headline claim *interpretable* (item 6's manipulation check, the λ→1
degenerate-limit story) are broken as written, and one gap (C17 held-out-entity
generalization) is undisclosed and newly load-bearing for this design in a way it
was not for bare geo3. All three FATALs are concretely fixable without abandoning
the wave.

---

## FATAL findings

### F1 — Item 6 (the manipulation-check closure) measures a tensor that is tautologically full-rank; it cannot detect the anchor-collapse failure it claims to close

**The flaw.** §3 item 6 (line ~362-373 of `KEY_ANCHORING_DESIGN.md`) is registered as the
closure for attack surface #2 ("a collapsed anchor could inflate the drift statistic
without genuine stability"): `σ_K/σ_1 ≥ 0.1` "computed on the **final `k_eff_items`
tensor** the same way item 1 already computes it for `v_eff_items`."

`k_eff_items` under `geo3_active` (and under every anchoring candidate, which all
route through the *same* `geo3_orthogonalize_logged` call, §2.2's own "UNCHANGED
call" framing) is the **output of Newton-Schulz**, which enforces `Q Q^T ≈ I_K`
by construction, every step, for *any* input — collapsed, degenerate, or not
(`model_rd.py` L878-879: `geo3_orthogonalize_logged(k_eff_raw, ...)`, the same
mechanism documented in `DELTANET_RD_EXACTNESS_DESIGN.md` §14.1's convergence proof).
A converged NS output has **all singular values ≈1 by definition**, so
`σ_K/σ_1 ≈ 1` **identically**, regardless of whether the pre-NS input (the raw
anchor blend) was well-conditioned or fully collapsed onto a single direction.

The design's own text says this explicitly, one sentence before proposing the
check: *"geo3 itself never needed this (its key-Gram is forced to ~0 IDENTITY by
construction, **which already implies full rank**)"* — then specifies item 6 to
measure the **same kind of tensor** (post-NS output) for the **new** mechanism,
walking directly into the identical tautology it just described. This is the
exact failure pattern `DELTANET_RD_EXACTNESS_DESIGN.md`'s own attack round already
found once, for a different pair of instruments (patched `q_self ≡ k_eff_items` →
alignment ≡ 1.0 — attack **F3**, §14.9 item 1 / §14.10, "the finding-5 admission
gate becomes tautological"). Item 6 reintroduces the same bug under a new name.

**Concrete failure scenario.** Train candidate (d) with the anchor table collapsed
onto (say) two dominant directions across the whole vocabulary (a real degenerate
solution — every entity's raw anchor points close to one of two axes). The blend
`k_blend_raw[j] = normalize((1-λ)k_eff_raw[j] + λ·A[key_ids[j]])` still varies
per-episode (different `k_eff_raw` per draw), so NS still receives a somewhat
different `(K,d)` input each episode and — as it always does — returns something
with `Q Q^T ≈ I_K` exactly. Item 6, read off this output, reports `σ_K/σ_1 ≈ 1.0`,
comfortably clearing `≥0.1`, **even though the anchor is fully collapsed**. The
cross-episode drift statistic could plausibly read favorably too (same nearly-fixed
pair of directions every draw "looks" stable), exactly attack surface #2's own
scenario — and nothing in the current admission stack would catch it.

**Fix required.** Item 6 must be computed on a tensor that is **not** already
forced through NS — e.g., `σ_K/σ_1` of the raw anchor table `A` itself (restricted
to `pools.train_name_ids`, an SVD of the weight matrix), or of the pre-NS blended
input `k_blend_raw` for a batch of episodes, analogous to how item 1's
`v_eff_items` check is genuine specifically *because* `v_conv` is never passed
through a rank-forcing transform.

---

### F2 — Candidate (d)'s "λ→1 ⇒ orthogonalized i-strong" claim is mathematically false for the stated construction; the cited init reuse crashes on the actual pool size

**The flaw, code-verified.** §2.2 states `A` is initialized "via the **same
QR-orthonormal construction** `strong_pin_table` already uses at init, lines
715–722." That construction is `embed_arms.py`'s `_qr_orthonormal_rows(n, d, seed)`
(L76-85), which has a **hard precondition, asserted in code**:

```
assert n <= d, f"cannot build {n} mutually-orthonormal rows in R^{d} (n>d)"
```

(`embed_arms.py` L81.) Arm (i-strong) satisfies this only because
`restrict_pools_for_strong_pin` (L276) deliberately shrinks the pool to **32 train
+ 32 held-out names** ("arm (i-strong)'s 32+32 name split," `_I_STRONG_POOL_SEED`
comment L28; confirmed by `embed_arms.py` L377-392's own self-test print: "32+32
disjoint from each other AND internally orthonormal **within each side**"). `n=32
≤ d=64=d_state` — the assert passes trivially, with 32 dimensions to spare.

The **train name pool** candidate (d)'s anchor table is meant to cover is **107
names** (`pool_report.n_train_names = 107`, verified directly from
`w1_rdx_K32_armiii-beta_s0.json`'s own `pool_report` field, and independently
from `DELTANET_RD_EXACTNESS_DESIGN.md` §14.7: "213 verified names split at
`heldout_frac=0.5` into ~106/107 train/held-out"). `107 > 64 = d_state`. Literally
reusing `_qr_orthonormal_rows` over the full train pool **raises `AssertionError`
at construction time** — this is not a hypothetical, it is the documented
precondition of the exact function the design cites.

If the build instead substitutes some *other*, unspecified init to dodge the
assert (the design never says which), the deeper problem survives regardless: **it
is mathematically impossible for 107 unit vectors in R^64 to be pairwise
orthonormal.** The Welch bound gives a hard floor on the best achievable packing:
`max|cos| ≥ sqrt((n-d)/(d(n-1))) = sqrt(43/6784) ≈ 0.0796` for `n=107, d=64` — i.e.
even the *optimal* arrangement has at least one pair of entities whose raw anchor
rows are ≥8% correlated. Newton-Schulz will still have to perform a genuine,
subset-dependent correction every episode (which of the other 31 co-drawn
entities happen to be the correlated ones varies by draw) — so **candidate (d) at
λ→1 cannot reach i-strong's literal zero-drift ceiling**, because i-strong's
ceiling is only achievable *because* its pool size (32) was deliberately kept
`≤ d_state`, removing the very phenomenon (subset-composition-dependent NS
correction) that causes cross-episode drift in the first place. This is not an
optimization/training issue; it is a fixed geometric fact about packing >d_state
vectors into d_state dimensions.

**Consequence for the design.** §2.0's "now-quantified 2×2" and §2.2's "the family
interpolates smoothly between the two already-measured endpoints" both rest on
treating i-strong as a valid λ=1 reference point for a *full-train-pool* anchor
table. It is not: i-strong changes **two** variables at once relative to bare
geo3 — (a) context-free stability *and* (b) pool size collapsed to ≤ d_state — and
the design's own §2.0 table only tests for (a). The λ→1 endpoint candidate (d)
can actually reach is an **unmeasured, structurally-lower ceiling** than i-strong's
archived 1.0000, and the design currently has no estimate of what that ceiling is
(the exact kind of reachability computation §4 already performs for the *value*-Gram
axis — Welch/i.i.d.-ceiling style — was never done for the *key*-anchor-table axis,
which is arguably the more decision-relevant one before spending any GPU-h on
candidate (d)).

**Fix required.** Before Wave −1: (1) specify exactly which init function `A` uses,
confirming it does not crash on `n_train_names=107 > d_state=64`; (2) either
restrict `A`'s coverage to a ≤64-entity dedicated subset (mirroring i-strong, but
then candidate (d) forfeits the "generalizes beyond a closed pool" advantage §2.1
explicitly claims over i-strong) or explicitly retract the "λ→1 ⇒ i-strong" claim
and, analogous to §4's value-Gram reachability check, compute the Welch-bound-style
achievable-ceiling estimate for the actual pool size so the outcome table has a
real target to compare against instead of the wrong one (i-strong's 1.0000).

---

### F3 — No pre-registered λ-trajectory criterion distinguishes "genuine learned interior-point anchoring" from "SGD saturates λ→1"; both report identically as Outcome A

**The flaw.** §2.2 discusses the λ→0 degenerate limit at length ("what failure
would mean... the informative negative"), but is silent on what a λ→1 (or
λ→0.95+) **success** means for interpretation. §3's Outcome A is defined purely
behaviorally: "drift ≥0.95 AND h4 ≥0.5, 3/3 admissible under items 1–6... KEY_
ANCHORING is the winning mechanism." Nothing in this criterion — nor anywhere else
in the design — distinguishes a result driven by an **interior** λ (e.g. 0.3-0.6,
genuinely blending learned and stable signal, the interaction the hypothesis in
§1 is actually about) from a result driven by **λ saturating near 1** (the model
using gradient descent to slowly rebuild something close to the already-known,
already-archived i-strong pin — reported in §2.0/§2.1 as *already confirmed,
zero new cost*, and explicitly **not** this design's own deliverable: "[i-strong]
is not a trainable fix... its role here is the pre-registered ceiling every
trainable candidate below is read against").

Per F2, λ→1 does not even reach i-strong's literal ceiling, but a λ trajectory
that saturates toward 1 across all three seeds would still be scientifically much
weaker evidence for "a mechanism that keeps the learned key path in the loop can
get anywhere close" (candidate (d)'s own stated goal, §2.2) than an interior-λ
win — because at high λ the learned key path is, by construction, barely in the
loop at all. The design has no pinned band (the same discipline
`DELTANET_RD_EXACTNESS_DESIGN.md` §14.5/§14.13 applied to drift — "adjectives
cannot gate a mechanism claim," numeric bands pinned before data exists) for what
counts as "genuinely learned mid-point" vs. "λ collapsed toward the trivial
endpoint." Outcome A currently allows both to be written up as the same claim.

**Fix required.** Pre-register, now: report the mean converged λ (last N steps,
every seed) at every Wave 1 checkpoint; pin a numeric band analogous to §14.5's
drift bands (e.g., "λ̄ < 0.8 at convergence, ≥2/3 seeds ⇒ eligible for the
interior-point claim; λ̄ ≥ 0.8 ⇒ relabel as 'Outcome A′: SGD re-derives the
i-strong-adjacent regime via gradient descent — confirms reachability, not a new
interior mechanism, and does not support the interaction hypothesis of §1 as
framed'"). This is the single highest-priority fix — it is exactly the load-bearing
distinction the task brief asked this attack to test for, and the design as
written cannot make it.

---

## MAJOR findings

### M1 — C17 held-out-entity generalization is neither gated nor reported for any anchoring candidate, despite being a newly live risk this design itself creates

**The flaw.** `run_deltanet_rd.py`'s own docstring (L11-14) and `evaluate_pool`'s
`use_heldout_entities` flag (only set `True` for the dedicated C17 pass, L564)
confirm: `M2_in_distribution` and `M3_held_out` — the **only** metrics §3's
primary/secondary/guard bars and the substitute admission stack (items 1-4) read
— are computed exclusively over `pools.train_name_ids`. `C17_heldout_entities`
(the disjoint name pool, `grammar_rd.py` L138-139/L423) is logged in every
checkpoint but appears in **no** bar, in either this design or its parent
(`DELTANET_RD_EXACTNESS_DESIGN.md` §16.2-§16.4's tables have no C17 row either —
the one mention, §8 item 9, is a general aside, not a pinned threshold).

For bare geo3 this omission is low-risk: Newton-Schulz orthogonalization is a pure
function of whichever raw keys are gathered — it has no entity-identity state and
generalizes trivially to any entity, held-out or not. For candidates (d)/(b)/(c)
this is **not** true: the anchor table `A` is indexed by `key_ids`, and attack
surface #1's own closure (correctly) restricts every gradient/EMA update to
`pools.train_name_ids` rows. That means **held-out-entity anchor rows are never
trained** — they sit at whatever `A`'s init leaves them (see F2: not even
guaranteed well-conditioned). If a C17 batch is ever run with `anchor_active=True`
and λ materially >0, the blend replaces part of a held-out entity's raw,
model-computed key with an untrained, identity-arbitrary direction — a plausible,
currently completely unmeasured degradation path for exactly the generalization
axis attack surface #1 is trying to protect. §3's own bars would not see this
regardless of how badly it degrades, because M2/M3 never touch the C17 pool.

**Fix required.** Add C17_heldout_entities as a required, reported diagnostic
(need not be a hard gate) for every Wave 1 anchoring cell, at the same hop depths
M2 already covers — this is a zero-marginal-cost addition (the field is already
computed every checkpoint) and directly closes the gap attack surface #1 opened
but did not finish closing.

### M2 — K=32 (the wave's actual headline bet) has no real go/no-go gate; the reused K=16 gate is expected to pass almost trivially

**The flaw.** §4 registers exactly one hard launch gate: predicted K=16 h=4
`rec@0.9 ≥ 0.8` under the (already-validated-at-K=16) tilt-only simulator. But
geo3's own **bare**, un-anchored K=16 drift is already 0.9416 (§16.1), which the
same registered simulator already maps to a predicted `rec@0.9` of **1.0000** —
comfortably above the 0.8 bar with zero anchoring contribution. Any anchoring
candidate that does not actively make K=16 *worse* will clear this gate
regardless of whether the anchor mechanism does anything useful at all. The
K=32 cell — the one the primary bar (§3) is actually about — is explicitly
**non-gating** ("both naive bounds are reported non-gating... neither bound
should be read as a tight prediction," §4). Net effect: Wave 1 can launch, spend
its full ~4-7 GPU-h, and only discover post-hoc whether the mechanism did
anything on the cell that matters — there is no pre-registered stop/no-stop
decision point tied to K=32 at all, unlike geo3's own Wave −1 (where the K=16
gate was a genuine test of a brand-new, previously-unmeasured mechanism).

**Fix required (the "minimal defensible gate" the task asks for).** Reuse the
Wave −1 5,000-step probe infrastructure the design already runs, but read it at
K=32 too: pre-register that if the probe-stage anchored drift at K=32 does not
improve by at least some pre-registered margin (e.g. +0.02-0.03) over geo3's own
measured probe-stage K=32 baseline (comparable probe-to-probe, not probe-to-final,
to avoid a training-stage confound), that is evidence the mechanism is not moving
the one quantity it targets, and Wave 1's K=32 cells should be deferred rather
than run to completion blind.

### M3 — The design's own fresh value-Gram evidence argues key-anchoring may be capped well below the bar regardless of outcome, with no early-stop to catch it

**The flaw.** §4/§6 already compute, honestly, that geo3's K=32 value-Gram
deviation is 1.3-1.8× *worse* than baseline (5.9274 vs. 3.18-3.58), and that a
value-Gram-corrected simulator predicts a ceiling of only **~0.19** at K=32 h=4
even under a **perfect** (`c=1.0`) key-drift fix — well under the 0.5 bar. This is
a real, quantified, pre-existing signal (computed *before* this wave's own data)
that value geometry, not key drift, may already be the binding constraint. The
design correctly pre-registers the re-attribution *if* Wave 1 fails (§6's Outcome
B block) — but does not act on this evidence going in: Wave 1 runs all mandatory
cells to 20,000 steps with no intermediate checkpoint rule that could catch "value-
Gram-capped" mid-run. Given the corrected simulator's own ~0.19 ceiling estimate,
an early checkpoint (e.g. 5,000-8,000 steps) showing drift already ≥0.95 but h4
tracking near ~0.15-0.25 would be a strong, cheap, mid-run confirmation of Outcome
B, saving the remainder of the run's compute and, more importantly, avoiding
running the *ablation* candidate (c) and the *fallback* candidate (b) — both gated
on (d)'s specific failure mode, §6 — under a mis-specified expectation.

**Fix required.** Add a pre-registered early-checkpoint read (drift + h4, at the
same probe-length used for the Wave −1 gate) with a non-binding but *reported*
comparison against the corrected-simulator's ~0.19 ceiling, so the wave's own
write-up can say, mid-flight, "this looks like Outcome B materializing" rather
than only after all cells complete.

---

## MINOR findings

**m1 — i-strong h=21 range misquoted.** The document's own opening summary states
h=21 "climbs from 0.0 at step 2000 to **0.9987–0.9994** by step 20000." Verified
directly against `w1_rdx_K32_armi-strong_s{0,1,2}_sp.json`'s final checkpoints
(step 20000): `recovered_frac@0.9` = 0.99939 (s0), 0.99878 (s1), 0.99994 (s2) —
i.e. **[0.9988, 0.9999]**. The upper bound is understated by 0.0005 and, more
importantly, the range as quoted excludes seed 2's actual value entirely. Trivial
to the argument (all three values are still ≈1.0), but a real transcription error
in a design that explicitly holds itself to a no-invented-numbers standard.

**m2 — §2.0's "baseline/arm iii-beta" key-Gram-dev range doesn't match its own
cited files.** The 2×2 table (§2.0, row "NOT stable × NOT orthogonal") states
"dev 2.72–2.77" for "baseline / arm iii-beta." Verified directly against
`w1_rdx_K32_armiii-beta_s{0,1,2}.json`: `key_gram_deviation_mean` = 2.7176, 2.7652,
2.7143 → actual range **2.71–2.77**, not 2.72–2.77. The stated lower bound matches
the *older*, different-arm figure from `DELTANET_RD_EXACTNESS_DESIGN.md` §15.1
("learned-baseline key-Gram dev (arm iii): 2.71–2.77" — note: 2.71, not 2.72
either, so even this doesn't match cleanly) rather than the freshly-extracted
iii-beta files the design says it read this session. Sloppy provenance, not
consequential to any conclusion.

**m3 — §5's Wave −1 budget row is internally inconsistent.** Prose: "anchor-table
init/blend/backward smoke (**~6–8** short probes...)." Same row's "New runs"
column: "**~8–10** short + 2 probes." This ~2-run discrepancy also slightly
understates the stated "~22–24" baseline run total (the column figures actually
sum to ~22–26). Does not change the GPU-h arithmetic, which is internally
consistent and correctly totals to the stated ~4.1-5.0 / ~5.6-6.7 GPU-h figures.

**m4 — Candidate (b)'s EMA-circularity risk has no pre-registered numeric
threshold.** §2.3/§6 honestly name the risk (the anchor chases a non-stationary
`W_k`) and propose a diagnostic ("comparing the anchor table's own step-to-step
movement against `W_k`'s gradient norm") but pin no numeric criterion for
"stabilized" vs. "still chasing." Low priority since (b) is a conditional
fallback, not mandatory — but the same "adjectives cannot gate a mechanism claim"
standard the parent design applied to drift bands (§14.5/§14.13) should apply here
too, before (b) is ever triggered.

---

## Verdict

**NEEDS-REV-2.** Required changes before this design is build-ready:

1. Rewrite item 6 (§3) to measure a genuinely non-tautological tensor (the raw
   anchor table or the pre-NS blend), not `k_eff_items` (F1).
2. Specify `A`'s actual init function; confirm it does not crash on
   `n_train_names=107 > d_state=64`; retract or replace the "λ→1 ⇒ i-strong"
   equivalence in §2.2 with a real, computed ceiling estimate for the actual pool
   size (F2).
3. Pre-register a numeric λ-trajectory band splitting "interior-point learned
   win" from "λ saturates toward the trivial endpoint," before any Wave 1 data
   exists (F3).
4. Add `C17_heldout_entities` as a required, reported diagnostic for every
   anchoring candidate (M1).
5. Add a real, pre-registered go/no-go read on K=32 itself, not only K=16 (M2).
6. Add an early-checkpoint stop/re-attribution rule tied to the corrected-
   simulator's ~0.19 ceiling estimate (M3).
7. Fix the two numeric misquotes and the Wave −1 run-count arithmetic (m1-m3).

None of these require abandoning the hypothesis or the candidate ranking — the
underlying interaction hypothesis (§2.0's 2×2 read) is reasonable, the parent
design's infrastructure reuse is careful and mostly correctly cited (verified:
`bind()` insertion sites at `model_rd.py` L871-888/877-878/926-990 all check out
exactly against the live code; the §5 budget correctly anchors on
`DELTANET_RD_EXACTNESS_DESIGN.md`'s own 80 GPU-h cap rather than the unrelated
300 GPU-h SCALE_TRANSFER figure, verified by direct grep — this specific
self-flagged concern clears). The three FATALs are all concrete, scoped, one-to-
a-few-paragraph fixes, not a redesign.
