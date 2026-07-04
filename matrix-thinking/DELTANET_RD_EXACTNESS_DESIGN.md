# DeltaNet RD Exactness Design — Why Per-Binding Operator Exactness Degrades With K on Real Tokenized Data

> **Rev 3 — 2026-07-03, post-round-2 verification.** Round-2 verification
> of Rev 2 returned NEEDS-REVISION: **9/10 round-1 findings verified
> cleanly addressed, budget arithmetic recomputed exact, and the
> verifier's own Welch-bound attack on Wave F's headline bar FAILED**
> (K=32 ≤ d_state=64 leaves no coherence obstruction — 32 exactly
> orthonormal directions exist in R⁶⁴ — and the synthetic harness already
> achieves `rec@0.9 = 1.0` at this exact (d,K) with Gram deviation
> 0.005–0.01; the ≥0.5 bar is achievable in principle and is **kept
> unchanged**, §5.5). Three NEW blocking MAJORs, all in Rev 2's own new
> material, fixed in this revision (finding→change map: **§12**):
> **BLOCK-1** — arm (iv) calibrated the RAW embedding rows (d_model=256)
> against a Gram band MEASURED on `k_eff` (post-conv, post-learned-`W_k`,
> d_state=64), with no safeguard against the same de-orthogonalization
> §4.2 itself warns about; the demanded reachability calculation was run
> (verified numerically this session) and **confirms the α∈[0,1] blend
> cannot even span the target band** — E‖G−I‖_F for i.i.d. unit vectors
> in d=256 tops out at ≈0.97 (K=16) / ≈1.97 (K=32) vs. the 1.26–2.77
> target. Fixed: a two-parameter construction (blend + shared-direction
> ρ, spans the band with margin), **closed-loop calibration against
> trained `k_eff`** (≤3 bounded probe iterations per K), and a
> pre-registered **demotion gate** on the (i)-vs-(iv) contrast (§4.5).
> **BLOCK-2** — the ≥50%-dominance Wave F gate had no near-miss path,
> although §1's own H_RD-EXACT predicts compounding mechanisms; a
> **40–50% reduced-scope fallback** (single top mechanism, trimmed
> cells, **same success bars**) and an **interventional-evidence-wins
> tie-break** are pre-registered (§5.5). **BLOCK-3** — F-geo-2's ZCA
> layer is now buildably specified: **content-position-only covariance
> pooling mandated** (BUFFER positions are exact-zero conv outputs, ~75%
> of tokens at K=8's padded T=128 — pooling them would degenerate the
> covariance), EMA statistics + ε-regularized inverse-sqrt with this
> codebase's known eigh-failure mitigations, a warm-up schedule, and a
> **required Wave −1 smoke** (§5.5). Minors: §4.3's "≤252" corrected to
> **251**; the cut-order "~11.6 GPU-h freed" figure was itself
> miscounted — decomposed per-step it is **~15.1**, with Wave 3's ~6
> flagged as unmeasured (§6). Budget: baseline 36.5 → **~37.5** (arm-(iv)
> calibration probes); **every realized path still closes ≤80 GPU-h**
> (§6).

> **Rev 2 — 2026-07-03, post-attack-round.** An independent adversarial
> review of Rev 1 returned **NEEDS-REVISION (no FATALs — core structure
> sound)**: 8 MAJOR, 1 MODIFIES, 1 MINOR. Every finding is addressed in
> this revision; the finding→change map is **§11** (placed at the end so
> §1–§10 read as the current design, not a diff). The load-bearing
> changes: fresh measured-β arm-(iii) reruns so mechanism (g)'s
> attribution never rests permanently on the β≈1 idealization (finding 1);
> §3.3 split into **Test A** (β-idealization adequacy — near-circular on
> geometry by construction, evidentiary weight downgraded) and **Test B**
> (the actual a+g sufficiency test, against the independently measured
> §16.2 frontier) (findings 2/3); arm (i-strong) promoted from conditional
> fallback to **co-primary at K=32** (finding 4); a pre-registered
> **≥2-alignment-clean-seeds** gate with an add-seeds-not-steps
> contingency for every K=32 cross-arm headline (finding 5); arm (ii)'s
> JL random projection **replaced by an exact span-projection** — zero
> distortion on the rows actually used, strictly better than the review's
> own suggested fix (finding 6); a **Gram-matched non-orthonormal frozen
> control (arm (iv))** built into Wave 1 proper, unbundling geometry from
> trainable-param count (finding 7); the value-Gram-vs-λ_nce dependency
> readout added to test whether mechanism (c) is upstream of (b)
> (finding 8); the deephop probe (`DELTANET_REALDATA_DESIGN.md` §18)
> folded in — mechanism (d)'s hop-supervision axis is already answered
> negatively, Wave 2's trigger bar raised accordingly (finding 9); Wave
> 3's timing gate restated as budget-anchored rather than inheriting the
> force-rank-calibrated ratio buckets, and §17.5's small-n caveat
> extended to §3's instrument (finding 10).
>
> **PI addendum (same revision cycle, incorporated into Rev 2):** the
> program must not end at attribution. **Wave F** — a pre-registered,
> mechanism-gated **fix-demonstration wave** (§5.5) — is added: for each
> plausible Wave-1 mechanism verdict the matching intervention arm is
> registered NOW, before any Wave 1 data exists (geometry → two
> key-conditioning fixes chosen from three candidates, rejection
> justified; NCE-crowding → a crowding-normalized contrastive term;
> kernel/precision → the fp32-path demonstration), with **frontier-shift
> success criteria** (headline: K=32 h=4 `rec@0.9` 0.009 → ≥0.5; minimum
> publishable: K=16 h=4 ~0.45 → ≥0.8; a no-h=1-sacrifice guard), the
> same §14.7 validity stack + finding-5 seed discipline, and **15–25
> GPU-h reserved, cut-protected last, not first** — Wave F is the
> program's deliverable; the attribution waves are its means. The goal
> statement (§1) is reframed accordingly: **name the mechanism, then
> demonstrate the fix that moves the frontier.** Expected-path spend is
> ~54.5 GPU-h; §6 pre-registers how every realized path stays ≤80 GPU-h
> with Wave F protected.

**Drafted 2026-07-03 (Rev 1); revised same day (Rev 2, Rev 3). Design only — no model/training code is written here.**
This document does not propose a new causal claim. RD-1 (real-data
train-time causal rank necessity) is already **CONFIRMED**
(`DELTANET_REALDATA_DESIGN.md` §17.11) via eval-time truncation at all four
K tested. What is *not yet explained* is **why** the same architecture, the
same delta rule, the same provable `rank(S_T)≥K` bound, produces a
**razor-cliff, perfectly-exact** operator at every `(d,K)` cell tested up to
`d=K=64` on hand-built orthonormal keys (`DELTANET_CAUSAL_RANK_DESIGN.md`
§12.8.3), and a **graded, sub-exact, monotonically-degrading-in-K** operator
at the identical `(d_state=64, K=32)` operating point on real GPT-2-tokenized
language (`DELTANET_REALDATA_DESIGN.md` §16.2, §17.5). This is a mechanism
study **with a constructive deliverable** (Rev 2, PI addendum): **name the
mechanism, then demonstrate the fix that moves the frontier.** The
attribution waves (§2–§5.4) are the means; the pre-registered,
mechanism-gated fix demonstration (Wave F, §5.5) is the deliverable.
Neither half re-litigates whether the phenomenon is real.

---

## 0. Reading list this design builds on (context, not repeated here)

- `matrix-thinking/DELTANET_REALDATA_DESIGN.md` §14.7 (the R2-4 gate
  decision: the validity stack for real-data arms — salvage tier both sides
  `σ_K/σ_1 ≥ 0.1` + per-item alignment `cos ≥ 0.9` (unchanged) + R2-8
  held-out-pool classification; `τ/τ_v = 0.03` retired to descriptive-only),
  §16 (Wave A — the graded K-axis frontier at fixed `d_state=64`: ID
  h=1/h=2/h=3 = 1.000/0.999/0.978 (K=8) → 0.997/0.90/0.68 (K=16) →
  0.94/0.58/0.27 (K=24) → 0.78/0.26/0.05 (K=32); entity-subspace rank
  94–99% of K at every cell, §16.4), §17 (the h=1 eval-truncation staircase:
  ceiling reached **exactly** at k=K at all four K, but the transition is a
  **graded window several ranks wide** (2–3 ranks at K=8, 12–13 ranks at
  K=32, §17.5), not the synthetic design's single-step cliff; §17.9's
  CONFIRM verdict and its "graded-not-razor" qualification that must travel
  with it), **§18 (Rev 2 — the deeper-hop training probe, complete
  2026-07-03: K∈{8,16} × 3 seeds trained at h_train={1..5} produce h=4–7
  recovery statistically indistinguishable from Wave A's held-out
  extrapolation (K=16 h=4: 0.390–0.503 trained-at vs. 0.419–0.465
  extrapolated); the depth-amplification signature reproduces a second
  time at h=14/15; "the lever, if it exists, lives at the write/geometry
  level (key crosstalk), not the supervision level" — §18's own reading,
  which this design inherits as a pre-answered negative on mechanism (d)'s
  supervision axis, §2/§5.2)**.
- `matrix-thinking/DELTANET_CAUSAL_RANK_DESIGN.md` §12.3–12.4 (Wave 0: the
  unconstrained arm saturates to `rec@0.9=1.0000` at **every** `(d,K)` cell
  tested, `d∈{64,128}`, `K∈{16,32,64}`, key-Gram deviation 0.0052–0.0097),
  §12.8.3 (the eval-truncation staircase: razor cliff at k=31→32 exactly at
  K=32, `d=64` — the identical `(d,K)` cell RD Wave A shows graded/sub-exact
  behavior on), §12.8.5 (the cross-architecture train-time-forcing-breaks-SGD
  finding, transplanted into this design's Wave 3/4 budget discipline).
- `matrix-thinking/chapter2/TASK_E_FINDINGS.md` §3 (depth-amplification: raw
  iteration depth is a sharper exactness probe than single-hop cosine
  tolerance — `sqrt(1-1/K)` per-mode-drop heuristic, confirmed exactly by
  §9's eigenmode analysis), §9 (the entity-subspace restriction + the
  scale-invariant, eigenmode-level verification method this design's Wave 0
  reuses for the crosstalk reconstruction below).
- `matrix-thinking/chapter2/STAGE0_DESIGN.md` §14–15 (the bespoke encoder's
  own exactness frontier, stressed along `d` at fixed `h=64`: d=16 exact,
  d=32 sub-exact plateau, d=64 far sub-exact, with rank recruitment itself
  becoming partial only at the deepest cell — a **third, architecturally
  distinct** instance of the same rank-recruited/exactness-scarce
  separation, on a bespoke attention-set encoder with no delta rule and no
  conv/chunked kernel at all).
- Harness code read directly (paths cited throughout, not re-derived):
  `matrix-thinking/deltanet_rd/{model_rd.py, run_deltanet_rd.py,
  grammar_rd.py, rank_utils.py, f15_lm_checkpoint.py,
  analyze_eval_truncation_rd.py}`.

---

## 1. The sharpened question

At the **identical** operating point `d_state=64, K=32`, two runs of the same
architecture (`fla.ops.delta_rule.chunk_delta_rule`, the delta rule
`S_t=S_{t-1}(I-β_t k_t k_t^T)+β_t v_t k_t^T`) produce categorically different
operators:

| | key source | `rec@0.9`, h=1 | `rec@0.9`, h=2 | key-Gram dev `‖K^TK−I‖_F` | eval-trunc transition |
|---|---|---|---|---|---|
| Synthetic (`DELTANET_CAUSAL_RANK_DESIGN.md` §12.3/§12.8.3) | hand-built QR-orthonormal, 64-dim, fed raw | **1.0000** | **1.0000** | 0.0053–0.0056 | razor cliff, k=31→32 (0.9681→1.0000) |
| Real (`DELTANET_REALDATA_DESIGN.md` §16.2, §17.4) | GPT-2 tokenizer → learned embedding → conv → `W_k` | **0.780–0.790** | **0.259–0.265** | 1.26–2.77 (pooled K=16∪32 band, §14.7 item 2 — not yet broken out by K, §3.1's free extraction fixes this) | graded, ~12–13 ranks wide (§17.5) |

Both runs recruit essentially the full entity-subspace rank (synthetic:
31.9993–31.9994/32; real: 29.93/32, 94%, §16.4). **Capacity is not the
difference.** Exactness is. This design exists to explain that gap with
discriminating cells, not another restatement of the gap itself.

**H_RD-EXACT (the sharpened question, restated as a falsifiable set of
mechanism hypotheses, §2).** The graded, K-dependent exactness deficit on
real tokenized data is attributable — quantitatively, not just
qualitatively — to one or more of a small number of concrete, measurable
mechanisms that have no analog (or a controllably different analog) in the
hand-built-orthonormal-key synthetic harness. Each mechanism below is
independently falsifiable by a specific cell; the design's headline
deliverable is an **attribution**, not a single verdict — "X% of the gap is
key/value geometry, Y% is Z" — because the honest prior, given this
program's own repeated finding that multiple independent factors compound
(late-transition budget artifacts, train-time-forcing interference,
depth-amplification), is that more than one mechanism contributes.

**Goal statement (Rev 2, PI addendum — the frame every wave below serves):
name the mechanism, then demonstrate the fix that moves the frontier.**
The attribution (§2–§5.4, Waves 0–4) earns its budget only as the gating
evidence for Wave F (§5.5): a pre-registered intervention, matched to the
dominant mechanism, whose success criterion is a **frontier shift** —
held-out-hop exact composition at K=32 lifted from floor, K=16 lifted to
near-K=8 levels — under the same validity stack as everything else. An
attribution with no demonstrated fix is an incomplete deliverable for
this design (though §5.5's negative branch records that a fix-refusing
attribution is itself publishable); a fix with no attribution would be
untargeted engineering. The design runs them in that order, gated.

---

## 2. Mechanism hypotheses (one sentence each)

Lettered (a)–(e) are the task brief's candidates; (f) and (g) are two
mechanisms this design surfaces during construction that have no synthetic
analog and are directly buildable from the same harness — flagged as
**found during design, not originally enumerated**, per this project's
standing transparency norm.

- **(a) Key geometry.** Real effective keys are linearly independent but not
  orthonormal (measured Gram deviation 1.3–2.8 vs. synthetic's 0.005–0.01);
  the delta rule's own sequential write mechanism (not an SGD-optimization
  artifact — see (g)) produces genuine cross-item leakage whenever
  consecutive keys are non-orthogonal, and that leakage should be
  **quantitatively predictable** from the measured Gram matrix alone (§3).
- **(b) Value geometry.** Key-side leakage only *shows up* as a cosine
  error to the extent the leaking value differs from the target value;
  value-side correlation structure is a second, separable multiplier on
  the same leakage, testable by an ablation inside the same closed form
  (§3.4).
- **(c) NCE-crowding.** The training objective's discriminative term
  (`L_nce`, in-episode softmax over K candidates, §14.4 of the RD design)
  only needs `pred_j` to outrank K−1 other candidates, not to reach
  cos≥0.9 in absolute terms; as K grows this ranking-vs-absolute-threshold
  gap should widen, diluting the pressure toward the eval metric's stricter
  bar.
- **(d) Optimization budget vs. K.** Step count was held constant (20,000,
  Wave A) across K; per the project's own late-transition law, any
  K-dependent "ceiling" must be shown to be a converged plateau, not a
  budget-starved snapshot, before it is trusted. **(Rev 2, finding 9:
  mechanism (d)'s *hop-supervision* axis is already answered negatively
  by `DELTANET_REALDATA_DESIGN.md` §18's deephop probe — training AT
  h∈{1..5} leaves the per-hop decay curve unchanged at K∈{8,16}, 3
  seeds, complete — so (d) survives on its *step-budget* axis only, and
  Wave 2's trigger bar is raised accordingly, §5.2. §18 does NOT bear on
  mechanism (c): K was held fixed within each deephop cell, so it says
  nothing about K-dependent NCE-crowding.)**
- **(e) Conv/window and chunk-boundary collisions at higher K.** Longer
  episodes (`T_bind = 7K` at the F15-LM-verified `conv_size=4` default)
  cross more `chunk_delta_rule` internal chunk boundaries (`chunk_size=64`)
  as K grows — an effect that could be confounded with K itself, since both
  increase together in the existing Wave A grid.
- **(f) — new — kernel-path / numerical precision.** The real-data path
  uses the production **chunked, bf16-only** kernel (`chunk_delta_rule`);
  the synthetic design used a **pure-PyTorch, fp32-capable sequential**
  recurrence (`deltanet_core.py`) specifically because it needed fp64
  gradcheck feasibility (`DELTANET_REALDATA_DESIGN.md` §4.3). This is a
  **bundled, untested confound** between the two lineages that has nothing
  to do with tokenization or embedding geometry at all — it has never been
  isolated because no cell in either lineage has varied kernel path at
  fixed data/grammar.
- **(g) — new — the β-gate's episode-context blindness.** `β_t =
  σ(W_β(x_t))` is computed from token `t`'s own (pre-conv) embedding alone
  (`model_rd.py`'s `bind()`, "from RAW pre-conv x") — a pure function of
  **entity identity**, never of how much a key's direction has already been
  "used" by earlier writes in the same episode. Exact recovery for
  non-orthonormal-but-independent keys generally requires a per-write gain
  that compensates for accumulated overlap (an RLS/Gram-Schmidt-style
  adaptive β); this architecture cannot express that function at all. This
  is not really a sixth independent mechanism — it is the reason mechanism
  (a)'s naive closed-form prediction (§3) should be **expected to fit well**
  rather than a coincidence: if the model had no way to learn around
  non-orthonormality, the delta rule's own uncorrected sequential-write
  mathematics should describe what SGD actually finds.

---

## 3. The analytical instrument — closed-form crosstalk reconstruction (mechanisms a, b, g)

This is the tool that turns "key geometry is probably involved" into a
number. It reuses data that is **already archived, zero new GPU-hours**.

### 3.1 What is already available

`Z_dump` (dumped by default on every RD run since `--save-z` defaults to
true, `DELTANET_REALDATA_DESIGN.md` §17.1) carries, per run, per example:
`S_T_raw` (the trained, unconstrained final state), `k_eff_items` (B,K,d,
L2-normalized — exactly what the kernel consumed), `v_eff_items` (B,K,d,
raw). These exist today for K∈{8,16,24,32} across
`experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA,waveBprobe}/`.
Per-checkpoint `key_gram_deviation_mean` / `value_gram_deviation_mean` are
also already logged in every result JSON's `checkpoints[i]` entries — they
have simply never been tabulated **broken out by K** (§14.7 item 2's
1.26–2.77 / 1.32–3.31 bands pool K=16 and K=32 together). Wave 0 (§6)
extracts these per-K for free, before any new analysis is trusted.

### 3.2 The reconstruction

The delta rule's update, restricted to the K real write events (every other
position has `β=0` and leaves `S` unchanged, so this K-step reduction is
exact, not an approximation), in this harness's own `S@k`-retrieves
convention (`model_rd.py::kernel_state_design_layout`'s documented
convention):

```
S_0 = 0                                            # (d_state, d_state)
for j = 0 .. K-1  (write/slot order = clause order = row order in
                    k_eff_items/v_eff_items, since VALUE tokens appear
                    in ascending slot order in the token stream):
    S_{j+1} = S_j @ (I - β_j · k_j k_j^T) + β_j · v_j k_j^T
S_pred := S_K
```

This is **fully specified, unambiguous, and directly smoke-testable**
against a hand-built K=2/K=3 case by exact arithmetic (matching this
project's own "hand-built key sets with known Gram deviation... fed through
the code path and must classify correctly" discipline, §5.2 of the RD
design) — no dependence on any particular closed-form matrix expression is
required for correctness. (A closed-form matrix version of the same
recursion — the "UT transform" used in DeltaNet's own chunked-parallel
derivation — likely exists in the literature (Yang et al., the DeltaNet
chunked-parallelization paper; **exact arXiv ID not re-verified in this
session, verify before external citation**, per this project's own
citation-confidence convention) and would let the reconstruction run in
`O(K^3)` instead of `O(K)` sequential outer-product updates; this is a pure
implementation-speed question, not a correctness one, and is not required
to build this design.)

**β. (REWRITTEN, Rev 2 — findings 1/3.)** Not currently dumped. Wave 0's
first pass uses the **idealized `β_j=1`** approximation — but Rev 1's
justification for it was imported from the wrong regime, and Rev 2 says
so plainly: the synthetic design's `β≈1` finding
(`DELTANET_CAUSAL_RANK_DESIGN.md` §3.6, §12.4) was measured under
**near-orthonormal keys, where β=1 is the exact fixed point of exact
recovery**. With overlapping real keys, a β=1 write **over-erases the
legitimate content other bindings have already stored along shared
directions** — so SGD may well prefer systematically smaller (and
possibly K-dependent) β on real data, and the β≈1 idealization is a
**hypothesis to be measured, not an anchor**. Consequences, wired into
the design rather than footnoted:

1. **The Wave 0 β≈1 pass is PROVISIONAL** — labeled as such in every
   output it produces, and conditioned on item 2 below before any
   mechanism-(g) attribution is written. (Rev 1 called this pass "the
   single most decisive free result this design can produce"; that
   phrasing is retracted — §9's open question is restated accordingly.)
2. **Fresh measured-β baseline runs are a required Wave 1 cell, not an
   incidental by-product (finding 1):** arm (iii-β), §4.1 — 5 fresh
   reruns of the unchanged learned baseline (K=16 ×2 seeds, K=32 ×3
   seeds) with the β-dump active, so the primary anomaly's mechanism-(g)
   attribution rests on **measured** β at exactly the cells where it
   matters, never permanently on the idealization. Without this cell,
   arm (iii)'s zero-cost archived reuse (§4.1) would have left the
   learned baseline as the *only* arm whose β is never observed.
3. **Required build item** (not "no code" for this document — a build
   note for the build phase): add one field to the `Z_dump` path
   (`run_deltanet_rd.py`/`model_rd.py`) dumping `beta[item_pos]` (B,K)
   alongside `k_eff_items`/`v_eff_items`. The added logging must not
   touch the compute graph — Wave −1 smoke: training loss bitwise
   identical over a few hundred steps at fixed seed, with and without
   the dump field (§6).

### 3.3 What the reconstruction tests (REWRITTEN, Rev 2 — finding 2: Rev 1's item 1 was near-circular as framed)

The attack round's objection, accepted in full: the Z_dump-internal
state-level comparison feeds the **same** measured `k_eff`/`v_eff` that
produced `S_T_raw` through a recursion **mathematically identical** to
`chunk_delta_rule` — the geometry is identical on both sides of the
comparison **by construction**, and the only free parameter is β. Given
the true β, `S_pred ≡ S_T_raw` up to kernel numerics, tautologically. Rev
1's claim that a close state-level fit is "direct, structural evidence for
mechanism (g)" over-read a near-tautology. The two checks are therefore
split, with honestly different evidentiary weights:

- **Test A — β-idealization adequacy (Z_dump-internal; DIAGNOSTIC weight
  only, downgraded from Rev 1).** `‖S_pred − S_T_raw‖_F / ‖S_T_raw‖_F`
  under β≈1. This measures exactly two things — (i) how far the learned β
  departs from 1 (the §3.2 regime-import question), plus (ii)
  kernel-vs-recursion numerics (mechanism f's shadow; expected small per
  the existing Tier-1 cross-check) — and **nothing about whether geometry
  explains the frontier**. Once measured β exists (arm (iii-β), §4.1),
  Test A splits further: reconstruction-with-measured-β vs. `S_T_raw`
  becomes a pure numerics/write-order check (expected ≈0, and a genuine
  smoke of the instrument), and the **β≈1-vs-measured-β gap becomes the
  direct measurement of how far SGD's real-data β sits from the
  orthonormal-regime fixed point** — the concrete deliverable Test A
  earns, no more.
- **Test B — mechanism (a)+(g) joint sufficiency (against the
  INDEPENDENTLY measured frontier; this carries the section's real
  weight).** From `S_pred`, compute the predicted per-item recovery
  `cos(S_pred @ k_i, v_i)` per K, and compare the **predicted**
  degradation-vs-K curve against the **measured**
  `recovered_frac@0.9`-vs-K frontier from §16.2/§17.4 — numbers produced
  by the live training-eval pipeline on fresh batches with the real query
  path, not by the dump the reconstruction consumes. **Hop scope, per
  §17.2's own principled boundary:** on archived dumps Test B is
  **h=1-only** (`succ` is not dumped, and reconstructing it via
  nearest-neighbor is the argmax-decoding pattern the house rules
  prohibit near any rank/exactness claim); the multi-hop extension
  (`cos(S_pred^h @ k_i, v_{π^h(i)})` via `apply_state_power`, against
  §16.2's full per-hop table) becomes available on Wave 1's new runs by
  adding `succ`/`tgt_slot` to the `Z_dump` — a second one-field build
  item alongside §3.2's β-dump, same no-compute-graph-impact smoke. What Test B
  isolates, stated precisely: whether **write-time geometry + the delta
  rule's uncorrected sequential-write algebra alone** — no appeal to
  training dynamics, readout-path effects, or optimization — suffice to
  generate the frontier's shape and magnitude. If the predicted curve
  matches at every K with a residual that does **not** grow with K,
  mechanisms (a)+(g) quantitatively explain the frontier. If the residual
  grows with K, something else (c/d/e/f) contributes on top. Honest
  residual dependence, named: prediction and measurement share the same
  trained weights (they cannot not — the mechanism under test lives in
  those weights); what they do not share is the evaluation path, the
  example set, or the recovery bookkeeping.

**Small-n caveat (Rev 2, finding 10 — §17.5's own caveat extended to this
instrument):** the dump carries only 4 eval examples × K items per run
(32–128 item-scores per seed). Mitigation, matching §17.5's own: aggregate
across every archived seed (10 at K=16, 7 at K=32, 2 each at K=8/24) and
report per-seed spread alongside every Test A/B number — the frontier
being explained showed seed-to-seed std ≤0.06 under the same n, so the
instrument's resolution is adequate for the effect size at issue, but
per-cell numbers below that resolution are not to be over-read.

### 3.4 The value-geometry ablation (mechanism b)

Two variants of the *same* recovery-curve prediction: **(b-blind)**
replace the measured `v_eff_items` with a hand-built orthonormal value set
of the same K (same magnitude, no cross-item correlation) before running
the recursion, isolating "how bad would key-side leakage look if value
geometry couldn't hide or expose it"; **(b-full)** use the actual measured
`v_eff_items`. If (b-full) tracks measured recovery meaningfully better
than (b-blind), value geometry is a real, separable contributor beyond key
geometry alone — not redundant with (a).

**Non-additivity caveat (Rev 2, finding 8): mechanism (c) is plausibly
UPSTREAM of (b), not parallel to it.** The InfoNCE term's separating force
is exactly the kind of pressure that would *set* `v_eff`'s Gram geometry
in the first place — in which case a (b-full)-vs-(b-blind) contrast
attributes variance to value geometry while the root cause is the
objective that shaped that geometry. Consequences: (i) the direct
dependency test is §5.1's zero-cost value-Gram-vs-λ_nce readout (does
`value_gram_deviation_mean` move systematically with λ_nce?); (ii) Wave
5's attribution table must present (b) and (c) as **potentially
non-additive** — a joint "(b|c)" line, not two independent percentages —
unless the dependency test comes back null, in which case they may be
treated as dissociated.

---

## 4. The controlled interpolation — the key design idea

**Same grammar (`grammar_rd.py`, unchanged), same model (`model_rd.py`'s
`DeltaNetRDBlock`, unchanged), same loss (§14.4's `L_cos + λ_nce·L_nce`,
unchanged), same C16/R2-8 diagnostics, same K-cycle generator.** Across
arms **(i)/(ii)/(iv)/(iii)** the only variable is where the entity/relation
token's `d_model=256` embedding row comes from — with one bundle named
rather than glossed (Rev 2, finding 7): freezing an embedding table changes
**geometry AND trainable-parameter count together**, which is exactly why
arm (iv) (Gram-matched non-orthonormal frozen, §4.5) exists — the
(i)-vs-(iv) contrast is the geometry-only comparison at fixed frozen-param
count, and §7's claim-tier language cites it, not any frozen-vs-learned
contrast, for the mechanism-(a) headline. Arm **(i-strong)** (§4.4) is the
deliberate exception to the one-variable statement: it additionally
bypasses the learned `W_k`/conv path at key and query positions — a
two-variable surgical arm by design, labeled as such everywhere it
appears, never folded into the one-variable claim.

### 4.1 Arm (iii) — learned (current, baseline) + arm (iii-β) — fresh measured-β reruns (Rev 2, finding 1)

**Arm (iii), archived:** zero new compute — reuses archived checkpoints
exactly: K=8 (waveA, 2 seeds), K=16 (wave0_rerun 5 + waveA 5 = 10 seeds),
K=32 (wave0_rerun 5 + waveA 2 = 7 seeds). These remain the recovery/rank
baseline for every cross-arm comparison.

**Arm (iii-β), new (Rev 2):** 5 fresh reruns of the **unchanged** learned
baseline — same config as Wave A's arm-(iii) cells to the flag — with the
§3.2 β-dump (and §3.3's `succ`/`tgt_slot` dump) active: **K=16 ×2 seeds,
K=32 ×3 seeds** (~2.9 GPU-h at the §6 anchor). Why it exists: without it,
Wave 1's zero-cost archived reuse would have left the learned baseline as
the only arm whose β is never observed, and the primary anomaly's
mechanism-(g) attribution would rest **forever** on the β≈1 idealization
(§3.2's regime-import problem). Why 5 rather than the review's suggested
2–3: K=32 is the primary anomaly cell and finding 5's
≥2-alignment-clean-seeds minimum (§6) applies to it — at the observed
K=32 alignment-clean base rate (~1–2 of 7 archived seeds), 3 seeds is the
minimum credible first draw, with the §6 add-seeds contingency behind it.
Wave −1 smoke: dump fields provably outside the compute graph (bitwise-
identical training loss over a few hundred steps at fixed seed, §3.2 item
3).

### 4.2 Arm (i) — frozen orthonormal ("synthetic-in-RD-harness")

**Feasibility, checked before committing to this arm (not assumed):** the
entity/relation vocabulary this design touches is 213 verified names + 21
rel-A + 16 rel-B + 1 period = **251 identities**
(`grammar_rd.py::build_entity_pools`'s own measured counts, "Expected
runtime outcome... 213/213 names, 21/21 rel-A, 16/16 rel-B"), all of which
must be **simultaneously** mutually orthonormal for this arm to hold at
every possible K-draw. `251 ≤ d_model = 256` — QR-orthogonalize a single
`256×251` random Gaussian once, frozen forever, 5 spare directions. **This
fits with no scope reduction; verify the arithmetic held at exactly this
Wave 0 → Wave 1 gate before building (§6), since it is one 2026-07-02
measurement away from not fitting (e.g. `heldout_frac` retuning).**

Buffer row stays zero-pinned+frozen exactly as in every prior arm (R2-3
unchanged). The `<Q>` marker row is the **one deliberate exception**: it is
not an entity/relation identity under test, it is a structural marker, and
forcing it to a frozen-random direction would inject noise without testing
any of §2's hypotheses — it is **left learned** in arms (i) and (ii) alike,
recorded explicitly so a reviewer does not read "frozen" as "100% of the
table," which it is not.

**What this arm does and does not guarantee.** Freezing the **raw
embedding row** to orthonormal does **not** guarantee `k_eff` (post-conv,
post-`W_k`) is orthonormal — `W_k`/the causal conv are still learned and
can, in principle, learn to de-orthogonalize a clean input. This is the
single largest open risk in this design (§8 item 1) and the reason §4.4
runs **co-primary at K=32, in parallel** (Rev 2, finding 4 — no longer a
serially-gated fallback).

### 4.3 Arm (ii) — frozen real GPT-2 embeddings (promotes C18 from Reserve to primary mechanism evidence)

`DELTANET_REALDATA_DESIGN.md`'s own C18 ("frozen-pretrained-GPT-2-embedding
variant... explicitly-labeled Reserve-wave robustness check," §5.2, §5.4,
§7) was scoped as a naturalness robustness check, never run. This design
promotes it to a **primary mechanism-diagnostic arm**, not a robustness
check — the framing differs even though the artifact (a frozen GPT-2
embedding table) is the same.

**Dimensionality mismatch, resolved explicitly (REWRITTEN, Rev 2 —
finding 6).** GPT-2-small's `wte` weight is `(50257, 768)`; this
harness's `d_model=256`. Rev 1 proposed a fixed random Gaussian (JL)
projection and claimed it "approximately preserves pairwise angles" — the
attack round correctly priced that claim: at `m=251` points and `k=256`
target dimensions the Dasgupta–Gupta bound gives `ε ≈ sqrt(8·ln m / k) ≈
0.5`, i.e. the worst-case guarantee is nearly vacuous, and the sentence
as written was not supported. **Rev 2 replaces the construction outright
with one that is exact rather than approximate:** the grammar only ever
consumes **251** real-vocabulary rows (213 names + 21 rel-A + 16 rel-B +
1 period = 251; buffer/`<Q>` are reserved IDs outside GPT-2's table —
Rev 2's "≤252" was a miscount, corrected at round-2's direction). Those rows
span a subspace of `R^768` of dimension `r ≤ 251 ≤ 256 = d_model`. Build
an orthonormal basis `Q ∈ R^{768×r}` of that span once (QR/SVD of the
used-row matrix, deterministic, frozen forever) and set
`embed_frozen[t] = Q^T · wte[t]` (zero-padded to 256 if `r < 256`).
Restricted to the used rows this is an **isometry** — every pairwise
inner product, angle, and norm among the rows the task can ever see is
preserved **exactly**, not to `(1±ε)`. Rows outside the span (the other
~50,000 vocabulary entries) are distorted; they are never consumed by
the grammar, so the distortion is irrelevant by construction. This keeps
precisely the property the arm needs — GPT-2's own anisotropy and
near-collinearity arrive untouched, so arm (ii) is a genuine
intermediate geometry, not a disguised copy of arm (i) (§8 item 2).
**Wave −1 smoke (kept from the review's suggested fix, now as a
verification of an exact map rather than a measurement of a lossy one):**
compute all `(251·250)/2` pairwise cosines raw-768 vs. projected-256 and
assert max discrepancy < 1e-5. `<Q>`-row and buffer-row handling:
identical to §4.2.

### 4.4 Arm (i-strong) — surgical `k_eff` pinning (CO-PRIMARY at K=32 — Rev 2, finding 4)

**Promotion rationale (Rev 2).** Rev 1 gated this arm serially on arm
(i)'s K=32 outcome. The attack round argued — and this revision accepts —
that the gate was backwards: `W_k` is a learned, unconstrained 256→64 map
with **no isometry incentive anywhere in the loss**, and arm (iii)'s own
learned embeddings already converge to key-Gram deviation 1.26–2.77 —
structural evidence is already in hand that this path de-orthogonalizes
whatever cleanliness its input has. The prior probability that arm (i)
alone is un-interpretable at K=32 (frozen-orthonormal input, but `k_eff`
degraded anyway) is therefore high enough that serial gating buys a
likely wasted round-trip for a saving of only ~1.8 GPU-h. **Arm
(i-strong) runs in Wave 1 proper, in parallel with arm (i), K=32 × 3
seeds.** The arm's mechanics are unchanged from Rev 1:

Bypasses `W_k`/conv entirely at the K write positions **and** at query
time: `k_eff_items[j] := u_{entity(j)}` and `q_eff := u_{entity(a)}` for a
**fixed, hand-built** per-entity lookup `u: \text{entity} \to R^{64}`
(64-dim, matching `d_state`, not `d_model` — a genuinely different object
from arm (i)'s frozen embedding row), drawn once via QR. Only `v_eff`
(through `W_v`/conv) and `β` (through `W_β`) remain learned. This is
architecturally the closest possible transplant of the synthetic design's
own premise (fixed keys, learned values, learned write-strength) into the
real grammar/loss/optimizer, at the cost of bypassing the tokenizer's own
key-side contribution entirely for this one diagnostic arm.

**Pool-size correction, scoped honestly.** `d_state=64` caps the number of
*simultaneously* mutually-orthonormal directions at 64 — far short of the
251-identity full pool arm (i) uses. This arm therefore runs on a
**reduced, dedicated pool**: 32 train names + 32 disjoint held-out names
(64 total, two independent QR draws of 32-dim orthonormal sets each — train
and held-out episodes never mix names in one K-cycle, so cross-set
orthonormality between the two 32-sets is not required). This supports
K≤32 exactly at the primary anomaly cell and nothing wider; **this arm is
not a general-purpose replacement for arm (i), it is a single-cell tie-breaker**,
run only at K=32 (§6) — co-primary there per finding 4, in parallel with arm (i).

**What each outcome would mean, stated before any data exists:**
- Arm (i-strong) closes the gap to synthetic exactness, arm (i) did not →
  `W_k`/conv learning itself de-orthogonalizes an already-clean input; a
  new, itself-interesting finding ("SGD detunes a frozen-orthonormal
  embedding via the learned projection it sits under" — worth a dedicated
  follow-up, not chased in this design's budget).
- Neither arm closes the gap → key geometry (mechanism a) is **decisively
  ruled out** as sufficient on its own; the mechanism is elsewhere (c/d/e/f)
  or a genuine interaction §9 must name explicitly.
- Both close the gap → mechanism (a) confirmed at the input-embedding
  level, no further surgery needed; arm (ii)'s intermediate position
  becomes directly interpretable (§7).

### 4.5 Arm (iv) — Gram-matched non-orthonormal frozen control (NEW, Rev 2 — finding 7)

**What it unbundles.** Arms (i)/(ii) change two things relative to arm
(iii) at once: the embedding **geometry** and the **trainable-parameter
count** (frozen vs. learned table). Rev 1 acknowledged this only as a
"documented, not-built follow-on" (§8 item 1, Rev 1 text); the attack
round required it built into Wave 1 proper, and this revision agrees —
without it, a Wave 1 result attributing the frontier to geometry could
equally be an optimization-dynamics effect of freezing ~50K×256
parameters.

**The calibration-layer problem Rev 2 missed (REWRITTEN, Rev 3 —
BLOCK-1).** Rev 2 calibrated the **raw embedding rows** (d_model=256) to
match a Gram-deviation band that was **measured on `k_eff`** — post-conv,
post-learned-`W_k`, in d_state=64. Two independent holes, both caught by
round-2 verification: (1) §4.2's own warning applies here too — the
learned `W_k`/conv can move arm (iv)'s realized `k_eff` geometry
arbitrarily far from whatever the raw rows were calibrated to, and Rev 2
had no safeguard; (2) **reachability**: the demanded calculation was run
(verified numerically this session, analytic + Monte Carlo agreeing to
3 significant figures) — for K i.i.d. uniform unit vectors in `R^d`,
`E‖G−I‖_F ≈ sqrt(K(K−1)/d)`, so the α∈[0,1] blend's maximum-disorder
endpoint (α=1, pure Gaussian) tops out at **≈0.97 at K=16 and ≈1.97 at
K=32 in d=256** — the blend **cannot reach the 1.26–2.77 target band at
K=16 at all**, and covers only the band's lower half at K=32. (For
scale: the same formula in d_state=64 gives 1.94/3.94 — the measured
k_eff band actually sits *below* i.i.d.-random at K=32, i.e. trained
keys are more orthogonal than chance in the state space; a raw-row
calibration in d=256 was aiming at the wrong object in the wrong
dimension.)

**Construction (Rev 3): two parameters, calibrated closed-loop on
trained `k_eff`, not open-loop on raw rows.**

- **Family:** `x_j = normalize( sqrt(1−ρ²)·[(1−α)·u_j + α·g_j]_norm +
  ρ·c )`, with `{u_j}` = arm (i)'s QR-orthonormal scaffolding, `g_j`
  i.i.d. Gaussian, and `c` a single shared unit direction. The shared
  component adds ~ρ² to every pairwise inner product, so `‖G−I‖_F`
  scales toward `sqrt(K(K−1))·ρ²` — measured this session: ρ=0.3 alone
  already yields ≈1.72 (K=16) / ≈3.44 (K=32); the family **spans the
  target band with margin at both K** (and deliberately mimics the
  anisotropy/common-direction structure real embedding tables are known
  for, rather than only i.i.d. crosstalk).
- **Calibration target: the trained `k_eff` Gram deviation** (the same
  object, computation, and dimension as the arm-(iii) band it must
  match), via a documented, bounded monitoring loop: (1) initial `(α,ρ)`
  from the raw-row arithmetic above; (2) one **5,000-step calibration
  probe** per K (k_eff geometry is meaningless without a trained
  `W_k`/conv — the probe trains one); (3) if the probe's final-checkpoint
  `k_eff` Gram deviation misses the per-K target band, adjust `(α,ρ)` by
  deterministic bisection and re-probe; **≤3 iterations per K, hard cap,
  pre-registered** — this is *instrument calibration against a
  baseline-matching criterion fixed by arm (iii)'s measured band*, not
  outcome tuning (the arm's recovery/rank results are never consulted
  during calibration); (4) freeze the table. Cost: ≤6 probes ≈ ~1 GPU-h,
  priced in Wave −1's row (§6). Sequencing unchanged: Wave 0's per-K
  band extraction still runs first.
- **Re-verification gate (pre-registered demotion rule):** at arm (iv)'s
  full 20,000-step runs, the trained `k_eff` Gram deviation is measured
  again (C16's standing instrument, zero extra cost). If it falls
  outside the per-K target band by **>25% of the band midpoint** in
  either direction, the (i)-vs-(iv) contrast is **demoted to descriptive
  tier** for the mechanism-(a) headline — §7's geometry-only citation
  then falls back to the bundled contrasts with the bundle named, and
  the miss itself is reported (it is evidence about how strongly the
  learned path re-shapes input geometry, the same phenomenon arm
  (i)/(i-strong) probe from the other side). X=25% is set now, before
  any data, to keep the demotion rule out of post-hoc reach.

**What each contrast isolates:** (i)-vs-(iv) = geometry alone, at fixed
frozen-param count — **the clean geometry-only causal comparison, and
the contrast §7's mechanism-(a) headline must cite (conditional on the
re-verification gate above)**; (iv)-vs-(iii) = learnedness/trainability,
at approximately matched scalar geometry.

**Cells:** K∈{16,32} × 3 seeds = 6 runs (~3.5 GPU-h) + ≤6 calibration
probes (~1 GPU-h, Wave −1). **Wave −1 smoke (raw-table sanity, kept but
now explicitly NOT the calibration criterion):** realized K-draw raw-row
Gram deviation finite, reproducible across 100 sampled draws, and
monotone in `(α,ρ)` — the pass/fail calibration criterion lives at the
`k_eff` level per the closed loop above.

**Honest residual (carried to §8):** matching the *scalar* `k_eff` Gram
deviation does not match the full Gram *spectrum/structure* — learned
geometry may be structured (e.g. frequency- or co-occurrence-correlated)
in ways the (α,ρ) family is not (the shared-direction term mimics one
common structure, not all). Arm (iv) matches the one statistic mechanism
(a)'s closed form (§3) says is first-order load-bearing; it does not
reproduce arm (iii)'s geometry in full, and the write-up must say
"k_eff-Gram-deviation-matched," never "geometry-matched."

---

## 5. Other active discriminating cells (mechanisms c, d, e, f)

### 5.1 Mechanism (c) — NCE-crowding, ranking-vs-absolute-threshold gap

**Free, Wave 0.** For every dumped example (existing archives, no new
runs): compute both **top-1 ranking accuracy** (`argmax_j cos(pred_i,
v_eff_j) == i`, the quantity `L_nce`'s gradient actually optimizes) and the
**already-logged** `recovered_frac@0.9` (absolute-cosine bar, the eval
metric) — per K. Report the **gap** (ranking-correct-but-sub-0.9 fraction)
as a function of K. **This never enters any headline recovery number** —
per `CLAUDE.md`'s standing argmax-decoding rule (Nichani, Lee & Bietti,
arXiv:2412.06538) and §14.3's identical discipline for `L_nce` itself,
ranking accuracy is a **diagnostic only**, computed and reported
separately, never blended into or substituted for the absolute-cosine
recovery metric a rank-necessity claim depends on.

**Active follow-on, gated on the free pass showing a live signal
(trigger WIDENED, Rev 2 — finding 8).** If the ranking-vs-threshold gap
widens materially with K, **or** if §3.4's (b-full)-vs-(b-blind) contrast
shows value geometry to be a material contributor (either signal now
triggers), an ablation on `λ_nce` (§14.4's fixed `λ_nce=1.0`) — rerun
K=32 at `λ_nce∈{0.3,3.0}` (already pre-registered as a sensitivity check
in `DELTANET_REALDATA_DESIGN.md` §14.4, never yet exercised) — tests
whether shifting the relative weight toward `L_cos` narrows the K=32
exactness deficit. 2 values × 3 seeds = 6 runs.

**Zero-cost dependency readout attached to the same ablation (Rev 2,
finding 8):** `value_gram_deviation_mean` is already collected at every
checkpoint of every run — tabulate it against `λ_nce` across the ablation
arms (plus the archived `λ_nce=1.0` cells). This is the **direct test of
whether mechanism (c) is upstream of (b)**: if the InfoNCE separating
force is what sets value geometry, value-Gram deviation should move
systematically with `λ_nce`; if it does not move, (b) and (c) are
dissociated and Wave 5's attribution may treat them additively (§3.4's
non-additivity caveat then relaxes).

### 5.2 Mechanism (d) — optimization budget vs. K (REVISED, Rev 2 — finding 9: the supervision axis is already closed, and the trigger bar rises)

**The half of (d) that is already answered — negatively — before this
design runs.** `DELTANET_REALDATA_DESIGN.md` §18's deephop probe
(complete, 2026-07-03, ~2 GPU-h, K∈{8,16} × 3 seeds, h_train={1..5} at
25K steps on the audited Wave-1 harness): training **at** h=4,5 produces
h=4–7 recovery statistically indistinguishable from — if anything a hair
below — Wave A's held-out extrapolation from h_train={1,2,3} (K=16 h=4:
0.390–0.503 trained-at vs. 0.419–0.465 extrapolated; K=8 h=4: 0.921–0.941
vs. 0.944–0.956), with in-distribution h=1–3 unchanged and the
depth-amplification signature reproduced a second time at h=14/15
(0.755–0.768 → 0.051–0.063 at identical effective hop). §18's own
reading, inherited here verbatim: per-binding inexactness ε is set at
write time by K, composition inherits ~ε^h regardless of the hop
distribution the readout was trained on — **the "train deeper" lever is
dead**. (Scope note, per the coordinator's own review: §18 does *not*
bear on mechanism (c) — K was fixed within each deephop cell. A K=32
deephop extension was in flight at §18's write-up; if it lands before
this design's Wave 2 gate, its verdict folds in here.)

**Free pre-check (Wave 0), unchanged in substance.** Re-pull
per-checkpoint trajectories (already archived, no new runs) for: K=24's
`waveA` cells (never checked for a flat tail) and K=16's **held-out-hop**
(h=4–7) trajectories specifically, not just h=1. Standing partial answer
already on record: K=32's h=1 recovery was already shown **flat for the
final 21,000 of 25,000 steps** (`DELTANET_REALDATA_DESIGN.md` §15.1 item
4, §14.7 item 4) — that cell needs no re-check.

**Active cell — trigger bar RAISED (Rev 2, finding 9).** With the
supervision axis dead and K=32's h=1 flatness already on record, (d)'s
remaining prior is low, and Wave 2's 2.5× extension (50,000 steps, 3
seeds, at whichever of K=16/K=24 the free check flags) now triggers only
on a pre-registered, concrete climbing-tail criterion — not on "looks
maybe non-flat": **net rise ≥ +0.02 in `rec@0.9` at the flagged hop over
the final 5 logged checkpoints (10,000 steps), with a positive trend
rather than oscillation** — calibrated against §14.2/§15.2's own
conventions, where "flat" tails oscillate within a ±0.02 band with no
trend and "still climbing" tails moved +0.03 or more over comparable
windows. Up to 2 K values × 3 seeds = 6 runs, expected NOT to trigger.

### 5.3 Mechanism (e) — conv/window and chunk-boundary collisions

**Free structural audit first (Wave 0).** `T_bind = 7K` at `conv_size=4`
(F15-LM-verified default); `_MIN_KERNEL_T = 128`
(`model_rd.py`, the F15-LM-driven short-sequence padding floor). Exact
numbers: K=8 → `T_bind=56`, **padded to 128** (2 chunk boundaries at
`chunk_size=64`); K=16 → `T_bind=112`, **also padded to 128** (2 chunk
boundaries, **identical** to K=8's padded length); K=24 → `T_bind=168`, not
padded, 3 chunks; K=32 → `T_bind=224`, not padded, 4 chunks. **This is
already a free natural experiment sitting in the existing data**: K=8 and
K=16 run at the *same* padded length and chunk count, yet show sharply
different exactness (K=8 near-perfect through several held-out hops; K=16
partial from h=2 onward) — this already argues chunk-count-at-2 does *not*
explain the K=8→K=16 step, since it is held fixed across it. The open part
of mechanism (e) is confined to the K=16→24→32 leg, where K and chunk-count
increase together and are genuinely confounded.

**Active decoupling cell.** Fix K=16 (so the binding *count* is unchanged)
and pad the BIND sequence with additional trailing, zero-embedding,
`β=0` buffer tokens (state-neutral by construction, same mechanism the
harness already uses for short-sequence padding) to reach `T_bind≈192`
(3 chunks) and `T_bind≈256` (4 chunks — **matching K=32's own unpadded
length exactly**, the sharpest single comparison this cell can produce: "K
still 16, but now crossing as many chunk boundaries as the real K=32 cell
does — does exactness degrade toward K=32's level anyway?"). Structural
smoke test required before trusting any result: corrupt the padding region,
assert every real clause's `k_eff`/`v_eff` are bit-identical with and
without padding (a direct extension of C14's two-sided blank-out
discipline, `DELTANET_REALDATA_DESIGN.md` §5.2). 2 pad-length conditions × 3
seeds = 6 runs.

### 5.4 Mechanism (f) — kernel path / numerical precision

**A 3-arm grid, not a single swap** (an upgrade over the naive "just try
fp32" cell, made necessary by an attack on the design's own first draft,
§8 item 5): `{chunked-bf16 (current/baseline), naive-bf16, naive-fp32}` ×
`K∈{16,32}` × 3 seeds. The **naive** arms substitute
`f15_lm_checkpoint.py::patched_delta_rule_recurrence` (already built,
already Tier-0 fp64-gradcheck-verified and Tier-1 cross-checked against
`chunk_delta_rule` on generic random inputs, `DELTANET_REALDATA_DESIGN.md`
§4.4) for `chunk_delta_rule` in the BIND phase — this function is
**dtype-preserving**, so it runs identically at bf16 or fp32 with no other
code change, decoupling **"chunked/WY-parallel algorithm vs. literal
sequential recursion"** from **"bf16 vs. fp32 precision"** in one 2×2 (plus
the existing baseline) design:

| | sequential (naive) | chunked (production) |
|---|---|---|
| **bf16** | naive-bf16 (new) | chunked-bf16 (existing baseline) |
| **fp32** | naive-fp32 (new) | *impossible — kernel rejects fp32 categorically (§4.3 of `DELTANET_REALDATA_DESIGN.md`)* |

If **naive-fp32** alone recovers exactness (naive-bf16 does not) →
precision is the mechanism, not the chunking algorithm. If **naive-bf16**
also recovers exactness → the chunked/WY-transform algorithm itself is
implicated, independent of dtype. If **neither** naive arm improves on the
chunked-bf16 baseline → mechanism (f) is ruled out entirely; the frontier
is intrinsic to the geometry/optimization story (a/b/c/g), not the
numerics or the kernel implementation. `chunked-bf16` reuses archived data
(zero new runs for that cell); new runs: 2 arms × 2 K × 3 seeds = 12.
**Targeted re-verification required before trusting the naive-reference
training substitution** (beyond the existing generic-input Tier-1 check):
smoke-test forward-output agreement between `chunk_delta_rule` and
`patched_delta_rule_recurrence` on THIS task's own **structured** (β mostly
zero, keys drawn from a trained/frozen table, not i.i.d. Gaussian) input
statistics — cheap, folded into Wave −1 (§6).

**Timing gate, restated for this wave specifically (Rev 2, finding 10).**
The ≤10×/10–15×/>15× bucket rule this lineage inherits was calibrated on
the `svd_lowrank` **force-rank** overhead — a different overhead class
from replacing a fused Triton kernel with a pure-PyTorch **per-token
sequential Python loop**, whose slowdown at `T=128–224` (`_MIN_KERNEL_T`
to K=32's `T_bind`) is unmeasured and could exceed any calibrated band.
Wave −1 therefore prices the naive arms **directly** (~500-step probes at
K=16 and K=32), and Wave 3's own gate is **budget-anchored, not
ratio-anchored**: proceed if Wave 3's projected total fits its priced
~12 GPU-h band × 1.5; if 1.5–2.5× over, apply cut-order step 2 (drop
`naive-bf16`, keep `naive-fp32`, halving the wave); if >2.5× over, shrink
Wave 3 to a single `naive-fp32` K=32 × 3-seed cell and record the rest as
priced out — the K=32 fp32 cell alone still answers the sharpest form of
mechanism (f)'s question at the primary anomaly cell.

### 5.5 Wave F — from attribution to demonstration (PI addendum, Rev 2)

Housed under §5 for Rev 2 numbering stability, but it is not another
discriminating cell — **it is the program's deliverable** (§1's goal
statement points here). Purpose: once the attribution names the dominant
mechanism, **demonstrate the intervention that extends the
exact-composition frontier**, not just explain the failure. Everything
below is registered NOW, before any Wave 1 data exists.

**Launch gate — three bands, pre-registered (REVISED, Rev 3 — BLOCK-2:
Rev 2's single ≥50% gate had no near-miss path, although §1's own
H_RD-EXACT names compounding mechanisms as the honest prior — the modal
outcome, e.g. a 45/35/20 split, would have structurally starved the
program's stated deliverable).** Wave F's gate is decided at Wave 5, on
two pre-registered dominance measures per mechanism: (M-attr) its share
of the K=32 h=1 exactness deficit in §3 Test B's measured-β attribution,
and (M-arm) the fraction of the gap between arm (iii)'s 0.78 and the
synthetic 1.00 ceiling that the corresponding Wave-1 arm closes.

- **≥50% band (clear dominance, either measure):** full track as
  designed below.
- **40–50% band (near-miss — the compounding-mechanisms modal case):**
  **reduced-scope fallback, registered now** — a single-top-mechanism
  demonstration: ONE fix candidate (for F-geo: **F-geo-2/whitening** by
  default, per the house structural-over-penalty preference, unless
  Wave 1 localized the de-orthogonalization to `W_k` — then the
  QR/Cayley swap already mandated below), K∈{16,32} only (K=24
  dropped), 3 seeds each (~6 runs ≈ 3.5 GPU-h + contingency). **Success
  bars UNCHANGED** — reduced scope trims cells, never standards: a
  reduced bar would blur the deliverable and open a Goodhart door; if
  the single fix passes the same bars at fewer cells the demonstration
  stands, and if it fails, the negative branch reads identically.
- **<40% for every mechanism (genuinely split):** Wave F does **not**
  launch on schedule — a combined-fix design goes back through the full
  design → attack → verify loop as a new revision, never improvised
  (the same anti-burn discipline as `DELTANET_REALDATA_DESIGN.md`
  §14.6's one-iteration cap: **no fix-fishing** inside this budget).

**Tie-break, pre-registered (BLOCK-2's second half):** if M-attr and
M-arm name **different** winners, **the interventional evidence (M-arm)
selects the track** — Wave 1's arm-gap closure is a causal intervention;
Test B's attribution is structural and non-interventional, and §7's own
tier hierarchy already ranks them exactly that way. The disagreement is
itself recorded in Wave 5's summary, the non-selected mechanism's
attribution line is flagged, and the disagreement is named in any
write-up (it is evidence the attribution model is incomplete, worth
reporting regardless of Wave F's outcome).

**Track mapping (one track runs, per the verdict):**

- **(a)/(g) geometry dominates → F-geo.** Two candidates, chosen from the
  three named in the addendum brief, both trained end-to-end on the SAME
  grammar/loss/model at K∈{16,24,32}:
  - **F-geo-1 — per-episode differentiable orthogonality penalty** on the
    effective-key Gram: `L_orth = λ_orth · ‖K_effᵀK_eff − I_K‖_F²` over
    the K in-episode `k_eff_items` (the exact tensor C16 already gathers
    — minimal diff, loss-side, directly targets the measured premise
    object). `λ_orth ∈ {0.1, 1.0}` pre-registered pair at K=32, single
    `λ_orth = 1.0` at K=16/24 — **no further tuning** (anti-Goodhart cap;
    same no-iterative-fishing rule as everything else here).
  - **F-geo-2 — whitening/decorrelation layer on the key path**: a
    ZCA-style whitening module on the post-conv key features
    (running population statistics, per-token, causal-safe — no
    within-episode lookahead), driving the *population* distribution of
    effective keys toward decorrelation so any K-draw's expected Gram
    approaches `I`. Structural (an architecture change that shapes
    geometry by construction) where F-geo-1 is a soft penalty — the pair
    deliberately spans the structural-vs-penalty dichotomy §14.4 already
    navigated for the collapse problem (there the structural option won;
    testing both classes on THIS problem is the point of picking these
    two).

    **Buildable specification (NEW, Rev 3 — BLOCK-3; Rev 2 left this at
    one sentence, under-specified for a codebase that has already had
    `eigh` fail to converge on real-embedding-derived matrices — the
    exact failure `truncate_to_rank_svd_lowrank` exists to work
    around):**
    - **(a) Statistics pooling: content positions ONLY, mandated.**
      BUFFER positions produce **exactly-zero** conv outputs by
      construction (zero-pinned rows + `β=0`), and they dominate the
      stream — at K=8's padded `T=128`, ~75% of positions are
      buffer/padding. Pooling them would mix a point-mass at zero into
      the covariance, shrinking and degenerating it. The running
      mean/covariance are therefore accumulated **only over positions
      whose token id is not BUFFER** (mask by token id — KEY/REL/VALUE/
      PERIOD content and query-window content), asserted by a unit
      check that a buffer-only batch contributes zero mass.
    - **(b) Numerical mechanics, pre-registered:** EMA running moments
      (momentum 0.99, bias-corrected — the BN convention), statistics
      held in fp32 buffers with **no gradient into the statistics**
      (BN-style: gradients flow through the applied linear map only);
      transform `W_zca = (Σ + εI)^{−1/2}` with `ε = 1e-4 · tr(Σ)/d`
      (scale-invariant, strictly-PD by construction); recomputed every
      **100 steps** from the current EMA (cached in between — a 64×64
      eigh is cheap, but re-deriving per step buys nothing);
      **eigh-failure mitigations inherited from this codebase's own
      history**: the deviation-#6 jitter-retry (+1e-6·tr/d on
      `LinAlgError`, one retry) and, if eigh still fails, eigenvalue
      clamping via `svd_lowrank`-style fallback — never an unhandled
      crash mid-run; **warm-up**: the layer applies identity for the
      first 500 steps while statistics accumulate, then switches to the
      live transform (hard switch, recorded step; no annealing
      schedule to tune).
    - **(c) Required Wave −1 smoke (build item, blocking):** run the ZCA
      layer over real probe batches (untrained + a few-hundred-step
      briefly-trained model): assert **finite outputs everywhere**;
      `cond(Σ + εI)` below 1e6 on content-pooled statistics;
      post-whitening content-position covariance within `‖Σ_w − I‖_F <
      0.1·d_state` after warm-up; and the pooling mask verified (buffer
      positions contribute zero statistics mass and receive an
      unchanged/identity-consistent transform).
  - **Rejected third candidate, justification recorded:**
    QR/Cayley-orthogonal reparameterization of `W_k`. An orthogonal `W_k`
    is an isometry — it **cannot repair non-orthonormal upstream
    embedding geometry, only avoid adding damage** — so at learned (arm
    (iii)) embeddings it attacks the wrong link unless Wave 1 localizes
    the de-orthogonalization to the `W_k`/conv path specifically (the
    "arm (i) degraded but arm (i-strong) exact" outcome, §4.4). It is
    pre-registered as the **conditional substitution for F-geo-2 in
    exactly that outcome**, not as a default candidate.
- **(c) NCE-crowding dominates → F-nce.** Crowding-normalized contrastive
  term: negatives subsampled to a **fixed count m=7 per query regardless
  of K** (uniform over the K−1 in-episode non-targets; K-invariant
  discrimination difficulty **by construction** — the structural fix,
  house preference), temperature held at T=0.1, chance floor pinned at
  `log(m+1) = log 8` for every K. A `λ_nce(K)`/temperature-scaling
  variant is the secondary knob, run only if the structural fix moves the
  frontier partially (one pre-registered pair, same anti-fishing cap).
  Training-only, per §14.3's standing rule — nothing here touches eval
  scoring.
- **(f) kernel/precision dominates → F-fp32.** Extend Wave 3's winning
  naive-fp32 arm to the full frontier-shift criterion at K∈{16,24,32}.
  The deliverable **includes the measured wall-clock multiplier,
  disclosed** — a fix that costs sequential-fp32 throughput is a
  diagnosis-grade demonstration, not a production recommendation, and the
  write-up says so plainly.
- **(d)/(e) dominates (expected not, per §5.2/§5.3):** the "fix" is
  trivial (more steps / re-chunked sequence layout) and Wave F reduces to
  confirming the corresponding Wave 2/4 cell at the frontier-shift
  criterion — cheapest possible track.

**Success criteria (pre-registered NOW, per the addendum brief).**
Achievability note (Rev 3): the round-2 verifier attacked the headline
bar with a Welch/coherence-bound argument and the attack **failed** —
at K=32 ≤ d_state=64 there is no coherence obstruction (32 exactly
orthonormal directions exist in R⁶⁴), and the synthetic harness already
achieves `rec@0.9 = 1.0` at this exact (d,K) with Gram deviation
0.005–0.01. The bar is information-theoretically and architecturally
achievable; whether a *trainable fix* achieves it is precisely what
Wave F measures. **The bar stands unchanged.**

| Tier | Cell | Baseline (measured, §16.2) | Bar |
|---|---|---|---|
| **Headline demo** | K=32, h=4 (first held-out hop) | 0.009 | `rec@0.9` **≥ 0.5** — into the current K=8 band's territory (K=8's own h=4 is 0.944–0.956, the aspirational ceiling, not the bar) |
| **Minimum publishable shift** | K=16, h=4 | 0.419–0.465 | `rec@0.9` **≥ 0.8** |
| **Guard (both tiers)** | same arm's h=1, same K | arm's own unconstrained baseline | within **−0.02** — a fix that buys held-out depth by sacrificing in-distribution binding fails regardless of the shift |

Validity: the full §14.7 stack + finding-5's ≥2-alignment-clean-seeds
rule at **every** headline cell (K=16 and K=32), same add-seeds-not-steps
contingency. One F-geo-specific instrument caveat, pre-registered: in
F-geo-1 arms the key-Gram diagnostic is **directly optimized by the
loss** and therefore loses its independence as a premise diagnostic — the
success criterion is the *behavioral* frontier shift, never the Gram
number itself, and premise validity in those arms leans on the
(un-optimized) salvage/alignment/value-side instruments.

**Negative branch, named:** a dominant mechanism whose matching fix does
NOT move the frontier is itself a publishable finding — it means the
attribution was incomplete (the mechanism is a correlate, not the
binding constraint) — and routes back through design → attack as a new
revision, not through fix-iteration.

**Budget:** nominal F-geo track (the widest): F-geo-1 = K16/K24 ×3 + K32
×3 ×2λ = 12 runs; F-geo-2 = 3K ×3 = 9 runs; 21 runs ≈ 12.2 GPU-h at the
§6 anchor, plus alignment contingency (≤4 runs ≈ 2.3) and flat-tail
2.5×-extension exposure → **nominal ~15, hard band ≤25 GPU-h** (F-nce and
F-fp32 tracks are strictly smaller). Reserved in §6's table, **cuttable
last, not first**, with an internal trim order (§6) that never goes below
the minimal demonstrable cell.

---

## 6. Manifest — waves, gates, budget

Calibration-first throughout, per `STAGE0_DESIGN.md` §12's and
`TASK_E_FINDINGS.md` §10's repeatedly-relearned late-transition law: every
active (GPU-spending) wave below either reuses a cell already shown flat
(§5.2), or carries its own flat-tail check with a pre-authorized 2.5×
extension before any "no effect" verdict is written. Measured RD-speed
anchor: Wave A's own 11 runs averaged **2,091s/run (~0.58 GPU-h, ~35 min)**
at 20,000 steps (`DELTANET_REALDATA_DESIGN.md` §16.9) — used as the primary
unit-cost anchor below; cells with a genuinely new code path (naive
reference, padded sequences) carry a wider, explicitly-uncertain band until
Wave −1 measures them directly, never a false-precision point estimate.

| Wave | Purpose | New runs | Est. GPU-h | Gate |
|---|---|---|---|---|
| **−1 (blocking — EXTENDED, Rev 3)** | Build/smoke every new code path: arm (i) init sanity (near-zero Gram deviation on entity rows **at step 0**, before any training); arm (ii) span-projection **exactness assert** (max pairwise-cos discrepancy raw-768 vs. projected-256 over all 251 used rows < 1e-5, §4.3); **arm (iv) closed-loop `(α,ρ)` calibration probes (BLOCK-1): ≤3 × 5,000-step probes per K∈{16,32}, calibrating against trained `k_eff` Gram deviation, ~1 GPU-h** plus the raw-table sanity smoke (§4.5); β/`succ` dump fields provably outside the compute graph (bitwise-identical loss at fixed seed, §3.2/§3.3); naive-reference structured-input Tier-1 re-check (§5.4); **F-geo-2 ZCA smoke (BLOCK-3): finite outputs, `cond(Σ+εI)` < 1e6, post-warm-up whitened covariance check, buffer-mask verification (§5.5)**; trailing-pad two-sided blank-out assert (§5.3); timing probes (~500–1,000 steps) for every new arm at K=16 **and the naive arms additionally at K=32** (finding 10 — their cost class is unknown). **9 short probes + ≤6 calibration probes.** | 9 (short) + ≤6 (calib) | ~3 | All smoke asserts pass; arm (iv)'s `(α,ρ)` frozen with its calibration trace recorded. Timing: interpolation arms bucketed against the §16.9 anchor as before; **Wave 3's naive arms use their own budget-anchored gate (§5.4), not the force-rank-calibrated ratio buckets** |
| **0 (free, always runs)** | §3 Test A (β≈1, PROVISIONAL) + per-K Gram band extraction (§3.1 — also feeds arm (iv)'s α-calibration); §3 Test B (h=1, archived dumps); §3.4 (b-blind)/(b-full); §5.1 ranking-vs-threshold gap; §5.2 flat-tail pre-check (K=24, K=16 held-out-hop trajectories); §5.3 T_bind/chunk-boundary audit. Pure numpy on already-archived `experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA,waveBprobe}/`. | 0 | **0** | Informs the conditional waves; §3's reconstruction residual computed **before** Wave 1 launches, so Wave 1's arms are read against a pre-registered baseline residual; arm (iv)'s per-K α targets frozen here |
| **1 (primary, the interpolation — EXPANDED, Rev 2)** | Arms (i)/(ii) × K∈{8,16,32} × 3 seeds (18); **arm (iv)** × K∈{16,32} × 3 seeds (6, §4.5); **arm (i-strong)** K=32 × 3 seeds co-primary (3, §4.4); **arm (iii-β)** K=16 ×2 + K=32 ×3 with β/`succ` dumps (5, §4.1); arm (iii) archived reuse at zero cost. Same loss/grammar/model/C16 stack throughout. 20,000 steps primary, flat-tail-checked with pre-authorized 2.5× extension per cell. | 32 | ~18.6 | See the **K=32 headline gate** below the table (finding 5) — the Rev 1 "≥1 premise-valid seed" gate survives only as the per-cell *read-at-all* floor, not as license for a cross-arm headline |
| **2 (conditional — trigger bar RAISED per §5.2, finding 9)** | 2.5× budget extension for whichever of K=16/K=24 Wave 0 flags as genuinely climbing (pre-registered criterion: net rise ≥ +0.02 over the final 5 checkpoints with positive trend), 3 seeds each, up to 2 cells. Expected NOT to trigger given §18 + K=32's recorded flatness | ≤6 | ≤8.7 | — |
| **3 (kernel path/precision)** | The 3-arm grid (§5.4): naive-bf16 + naive-fp32 × K∈{16,32} × 3 seeds; chunked-bf16 reused. Own budget-anchored timing gate (§5.4) | 12 | ~12 (wide band, Wave −1-priced) | — |
| **4 (chunk-boundary decoupling)** | K=16 fixed, trailing-pad to 3/4 chunks × 3 seeds (§5.3) | 6 | ~3.9 | — |
| **NCE ablation (conditional — trigger WIDENED per §5.1, finding 8)** | `λ_nce∈{0.3,3.0}` at K=32, 3 seeds each; carries the zero-cost value-Gram-vs-λ_nce dependency readout regardless of which trigger fired | ≤6 | ≤5 | — |
| **Alignment-clean contingency (finding 5)** | +2 seeds for any K=32 arm below 2 alignment-clean seeds (ONE iteration per arm; see the gate below) | ≤10 | ≤5.8 | — |
| **5 (attribution synthesis, free)** | Re-run §3 Tests A/B with **measured** β (and multi-hop Test B via the dumped `succ`) on Waves 1/3/4's own new data — this supersedes Wave 0's provisional β≈1 pass; combine every wave's verdict into one attribution table (§7), with (b)/(c) presented jointly unless §5.1's dependency readout dissociates them. **Outputs the Wave F gate verdict (§5.5's ≥50% dominance rule), recorded in the wave summary before any Wave F manifest generates** | 0 | **0** | Wave F launches iff a mechanism dominates; split attribution → design→attack loop, no fix-fishing |
| **F (fix demonstration — PI addendum; gated on Wave 5's verdict; THE DELIVERABLE)** | One track only per §5.5's pre-registered mapping (F-geo: 21 runs nominal; F-nce/F-fp32: smaller); frontier-shift success criteria + no-h=1-sacrifice guard (§5.5's table); §14.7 stack + finding-5 seed rule at every headline cell | ≤21 (+≤4 contingency) | **15 nominal, ≤25 reserved** | Frontier-shift verdict recorded per §5.5's pre-registered bars — headline / minimum-publishable / negative-branch |
| **Total (required baseline, attribution only)** | Wave −1 + Wave 0 + Wave 1 + Wave 3 + Wave 4 + Wave 5 | 58 + ≤7 probes | **~37.5** (Rev 3: +1 for BLOCK-1's calibration probes) | |
| **Expected path (baseline + Wave F nominal, conditionals unfired)** | baseline + F | ~80 | **~52.5–55.5** | |
| **All-conditionals-max planning sum** | + Wave 2 + NCE ablation + alignment contingency + Wave F at its ≤25 band + ~8 general Reserve | ~106 | **~90 on paper — exceeds the 80 cap; resolution pre-registered below, every realized path stays ≤80** | |

**How the 80 GPU-h cap is honored (pre-registered, not improvised):** the
~90 planning sum is the sum of *allocations*, several of which are
mutually exclusive or expected-unfired in any realized path. Rules, in
order: (i) **unfired conditional allocations roll into Wave F first** —
Wave 2 (≤8.7) is expected not to trigger (§5.2) and the NCE ablation
(≤5) fires only on its §5.1 signals; (ii) if realized spend plus the
chosen Wave F track's Wave −1-priced cost projects past 80, **cut-order
steps 1–4 fire before Wave F is touched** — decomposed per-step (Rev 3
minor fix: Rev 2 quoted "~11.6," which miscounted by omitting step 3):
step 1 = ~3.9 (Wave 4, anchor-priced), step 2 = **~6 (Wave 3's
naive-bf16 half — flagged: unit cost UNMEASURED until Wave −1's naive
probes; this is the soft entry in the sum)**, step 3 = ~3.5 (Wave 1
K=8 cells, anchor-priced), step 4 = ~1.7 (arm (iv) K=16, anchor-priced)
— **total ~15.1 GPU-h**; (iii) only after steps 1–4 are exhausted does
Wave F trim **internally**: (F-i) drop the K=24 fix cells (−~3.5),
(F-ii) drop the second fix candidate everywhere except K=32 (−~3.5),
(F-iii) **floor = the winning candidate × K∈{16,32} × 3 seeds +
finding-5 contingency (≈10–13 GPU-h) — never cut below this while the
program claims a demonstration**; if even the floor cannot be funded,
Wave F is **deferred to a follow-on design, not run underpowered** — a
deferred demo, never a broken one. (BLOCK-2's 40–50% reduced-scope
fallback is itself ~3.5–6 GPU-h — comfortably under the floor — so the
near-miss band never stresses the cap.) General Reserve is trimmed from
Rev 2-interim's 12 to **~8** (Wave F's own 15–25 band carries its
headroom). **Worked worst-case check (Rev 3):** even if every
conditional fires at its ceiling (37.5 + 8.7 + 5 + 5.8 = 57.0 realized
attribution spend), Wave F at nominal 15 lands at 72.0 ≤ 80 with no
cuts; Wave F needing its full 25 projects to 82.0, at which point
cut-order step 1 alone (−3.9) restores ≤80 — **no realized path
requires touching Wave F's floor to stay under the cap.**

**K=32 headline gate (Rev 2, finding 5 — pre-registered before any Wave 1
data exists).** The observed alignment-clean rate at K=32 in the archived
learned arm is **1–2 of 7 seeds** (`DELTANET_REALDATA_DESIGN.md` §14.7
item 3: 1/5 in the Wave 0 rerun; §16.5: 1/2 in Wave A), and the
`align_min` decay appears unrelated to convergence quality (recovery holds
flat-to-rising while alignment decays, §16.5) — so Rev 1's "≥1
premise-valid seed" gate was too permissive for a **comparative** claim
across arms. Rule: **any cross-arm K=32 headline comparison requires ≥2
alignment-clean seeds per arm** (strict §14.7 stack: salvage both sides +
per-item alignment ≥0.9). Contingency is **add seeds, not steps**: an arm
below 2 clean seeds after its first 3 gets +2 fresh seeds (same config,
new seeds — the decay phenomenon is seed-stochastic, not step-curable on
the evidence of §16.5's flat-recovery-during-decay trajectories), **one
iteration only**; an arm still below 2 clean after the extension is
reported at the **descriptive tier** per §14.7's conventions — never
silently promoted, and never rescued by moving the 0.9 threshold (R2-2's
symmetric-handling rule binds here too). Archived arm (iii) already meets
the bar (2 clean of 7); arm (iii-β)'s 3 fresh K=32 seeds are subject to
the same rule.

**Pre-registered cut order**, applied in this sequence if measured costs
run materially over the table above (same discipline as
`DELTANET_REALDATA_DESIGN.md` §7's cut order):
1. Drop Wave 4 (chunk-boundary decoupling) — the least central to the
   headline anomaly; §5.3's free K=8-vs-K=16 padding-matched natural
   experiment already delivers a partial answer at zero cost.
2. Drop Wave 3's `naive-bf16` arm specifically, keep `naive-fp32` — the
   sharper of the two precision/algorithm hypotheses; the
   precision-vs-chunking-algorithm distinction becomes a documented open
   question (§9) instead of a closed one. (6 runs cut; this is also
   §5.4's own 1.5–2.5×-over response.)
3. Drop Wave 1's K=8 cells (arms (i)/(ii)) — real data is already
   near-synthetic-exact at K=8; the interpolation is most diagnostic at
   K=16/32 where the baseline gap is largest. (6 runs cut.)
4. Drop arm (iv)'s K=16 cell, keep its K=32 cell — the geometry-vs-param-
   count unbundling matters most at the primary anomaly cell. (3 runs
   cut.)
5. Every conditional item (Wave 2, NCE ablation) is already gated and is
   cut first in a genuine squeeze, before any of 1–4 above.
6. **Never cut:** Wave 0 (free); Wave 1's K=16/K=32 cells for arms
   (i)/(i-strong)/(iii-β); arm (iv)'s K=32 cell. **The finding-5
   contingency seeds are never cut while their arm's headline is claimed**
   — if budget genuinely forces skipping the contingency for an arm, that
   arm's result drops to the descriptive tier; the cut degrades the claim
   tier, it never silently preserves it.
7. **Wave F is cut-protected LAST among all new spend (PI addendum):**
   steps 1–5 fire before Wave F is trimmed; Wave F then trims only via
   its own internal order (F-i/F-ii/F-iii above) and is **deferred, not
   run underpowered**, if its floor cannot be funded. The attribution
   waves are the means; the demonstration is the deliverable — the cut
   order is built to reflect that priority, not to invert it under
   pressure.

---

## 7. Claim-tier language — consistent with §14.7's stack, not a new standard

Every new headline number this design produces inherits
`DELTANET_REALDATA_DESIGN.md` §14.7's dated, measured validity stack
**exactly**, not the stale τ=0.03 unconstrained-arm check
(§16.6's own documented aggregator bug is a standing warning: any new
sweep/aggregation tooling this design's build phase writes must implement
the salvage-tier rule from the start, not inherit that bug a second time):

- Headline validity = salvage tier both sides (`σ_K/σ_1 ≥ 0.1`, key **and**
  value) + per-item alignment (`cos ≥ 0.9`, unchanged, symmetric handling
  per R2-2) + R2-8's held-out-pool classification. `τ/τ_v = 0.03` logged,
  descriptive only, never gating.
- Non-premise-valid checkpoints are reported in their own category, never
  blended into a headline mean, exactly the companion rule
  (`DELTANET_REALDATA_DESIGN.md` §5.2's Rule, §14.7 item (1)/(2)).
- Ranking accuracy (§5.1) and `L_nce` are diagnostics; per §14.3's
  standing rule, neither may enter, substitute for, or be blended into any
  exact-recovery headline number.

**What tier each wave's verdict earns, stated once so it cannot drift at
write-up time:**

- **§3's reconstruction (REVISED, Rev 2 — finding 2):** Test A
  (Z_dump-internal state fit) is a **β-idealization adequacy diagnostic
  only** — geometry is identical on both sides by construction, so a close
  fit earns no explanatory claim about the frontier and must never be
  quoted as one. Test B (predicted-vs-measured frontier) is the
  **structural, deterministic reproduction** test: if the predicted
  degradation-vs-K curve matches the independently measured §16.2 frontier,
  that is as strong an *explanatory* claim as a non-interventional analysis
  can earn — "the measured write-time geometry, run through the delta
  rule's own uncorrected mathematics, is sufficient to generate the
  frontier." Still **not** a causal claim (nothing was intervened on), and
  **provisional until re-run with measured β at Wave 5** (finding 3).
- **Wave 1's interpolation arms (§4) — bundle named (Rev 2, finding 7):**
  a genuine **intervention** on the embedding table, grammar/model/loss
  held fixed — earning **"causal, premise-conditional"** in the same sense
  RD-1 itself is labeled (`DELTANET_REALDATA_DESIGN.md` §1's MAJOR-8
  qualifier). But the causal variable must be named honestly per contrast:
  any **frozen-vs-learned** contrast ((i)/(ii)/(iv) vs. (iii)) intervenes
  on **geometry AND trainable-param count bundled** and is labeled
  "geometry+trainability, bundled"; the **(i)-vs-(iv)** contrast — frozen
  orthonormal vs. frozen Gram-matched-non-orthonormal at identical
  trainable-param count — is the **geometry-only** causal comparison, and
  is the contrast any mechanism-(a) headline must cite **(Rev 3,
  BLOCK-1: conditional on arm (iv) passing its §4.5 trained-`k_eff`
  re-verification gate — a >25% band miss demotes this contrast to
  descriptive tier and the headline falls back to the bundled contrasts,
  bundle named)**. Arm **(i-strong)**
  is two-variable by design (embedding source AND `W_k`/conv bypass) and
  earns only its diagnostic tie-breaker role (§4.4's outcome table), never
  a standalone causal headline. "Premise-conditional" additionally means:
  conditional on §4.2's open risk (does frozen-orthonormal input propagate
  to `k_eff` orthonormality) being resolved by the arm-(i)-vs-(i-strong)
  comparison, not assumed.
- **Wave 2/3/4's active cells:** each is its own single-variable causal
  intervention (budget, kernel path, sequence length) on top of the
  already-CONFIRMED RD-1 result — none of them re-opens or re-tests RD-1
  itself; a null result on any of them narrows the explanation space, it
  does not weaken RD-1's own eval-truncation-based verdict.
- **Wave 5's synthesis:** an attribution, explicitly **not** a single
  number — the honest output is a table of "% of the K=32 exactness deficit
  each confirmed mechanism plausibly accounts for," with an acknowledged
  residual if the mechanisms tested do not sum to the full gap, and with
  **(b) and (c) presented as a joint, potentially non-additive line**
  unless §5.1's value-Gram-vs-λ_nce dependency readout dissociates them
  (Rev 2, finding 8).
- **Wave F's demonstration (PI addendum):** a genuine intervention with a
  pre-registered behavioral success bar — the frontier-shift claim earns
  **"causal, premise-conditional"** on the intervention variable (the fix)
  under the §14.7 stack. **Two tiers kept strictly separate at write-up:**
  (1) "the fix moves the frontier" (the demonstration claim — what Wave F
  measures); (2) "therefore the attribution was right" (a *supported
  inference*, never proven by the demo alone — a fix can work through a
  side channel its mechanism story did not predict; conversely §5.5's
  negative branch shows a dominant-mechanism attribution can fail to
  yield a working fix). The demo claim and the attribution claim carry
  their own labels and are never collapsed into each other. F-geo-1's
  additional caveat travels with any of its numbers: the key-Gram
  diagnostic is optimized by that arm's loss and is not premise-evidence
  there (§5.5).

---

## 8. Attack-yourself

1. **Arm (i)'s open risk is real and now attacked from two sides at once
   (REVISED, Rev 2 — findings 4/7).** Freezing the raw embedding row does
   not architecturally guarantee `k_eff` orthonormality; `W_k`/conv remain
   learned and could de-orthogonalize a clean input. Rev 2's structure:
   (i) arm (i)'s own `k_eff` Gram deviation at convergence is measured
   directly (C16's instrument, zero extra cost), never assumed clean;
   (ii) arm (i-strong) runs **co-primary in parallel** at K=32 (finding
   4), so the `W_k`-detuning question is answered in the same wave, not a
   round-trip later; (iii) Rev 1's "fewer trainable parameters changed
   the dynamics, not geometry" ambiguity is no longer a documented-only
   follow-on — it is **built**, as arm (iv) (finding 7): the (i)-vs-(iv)
   contrast holds frozen-param count fixed and varies geometry alone.
   Residual, still honest: arm (iv) matches the *scalar* Gram deviation,
   not the full learned Gram spectrum/structure (§4.5) — "geometry-only"
   means "Gram-deviation-level geometry," and the write-up must say so.
2. **Arm (ii)'s "intermediate point" framing is an empirical claim, not a
   guarantee.** GPT-2's own embedding geometry is known to exhibit
   anisotropy (embeddings clustering in a narrow cone) that could make its
   *effective* Gram deviation **worse**, not better, than the RD harness's
   own from-scratch learned embeddings — arm (ii) landing outside the
   (i)/(iii) interval entirely, in either direction, is a real possible
   outcome and would itself be reportable, not a design failure. (Rev 2
   note: the span-projection replacement (§4.3, finding 6) removes the
   Rev 1 worry that projection distortion could *manufacture* this
   outcome — whatever geometry arm (ii) exhibits is now exactly GPT-2's
   own, so an out-of-interval result is a fact about GPT-2's table, not
   about the dimensionality bridge.)
3. **The β≈1 idealization is now bounded, not just flagged (REVISED,
   Rev 2 — findings 1/3).** Rev 1 relied on a β≈1 anchor imported from
   the orthonormal regime where β=1 is the exact fixed point; with
   overlapping real keys, β=1 over-erases shared-direction content and
   SGD may prefer smaller, possibly K-dependent β — which would
   systematically bias Test A/B at exactly the higher-K cells this design
   cares most about. Rev 2's answer: arm (iii-β)'s fresh measured-β runs
   at K=16/32 (§4.1) plus Wave 5's mandatory measured-β re-run of Tests
   A/B. The Wave 0 β≈1 pass is labeled PROVISIONAL in every output and
   carries no mechanism-(g) attribution on its own.
4. **Post-hoc cell selection.** K=32 (and to a lesser extent K=16) were
   chosen specifically because they show the largest synthetic-vs-real
   gap — a genuine multiple-comparisons/selection risk in principle.
   Mitigated by the underlying data's own low seed-to-seed variance (≤0.06
   at every reported cell in `DELTANET_REALDATA_DESIGN.md` §17.5, mostly
   ≤0.02) and by 7–10 seeds' worth of existing archived data at exactly
   this cell (not a single noisy run) — the anomaly being explained is
   itself well-replicated, not cherry-picked from one seed.
5. **Wave 3's 3-arm grid still bundles more than the two axes it names.**
   Swapping `chunk_delta_rule` for `patched_delta_rule_recurrence` changes
   the recurrence's own parallelization structure (chunked/WY vs. literal
   sequential) **and** removes any GPU-kernel-specific numerical
   optimizations Triton's autotuner applies — "algorithm" and
   "implementation-specific numerics" are not perfectly separated by this
   2×2 design even after the upgrade from a single swap (§5.4). Flagged as
   a residual, not fully closable within this design's scope without a
   third, hand-written chunked-fp32 reference (out of scope, no such
   reference exists or is planned to be built here).
6. **No fresh novelty/positioning check was run for this specific
   mechanism-diagnosis document.** RD-1's own underlying causal-necessity
   claim is already positioned (`DELTANET_REALDATA_DESIGN.md` §10,
   2026-07-02, medium-to-high confidence). Whether "why does a production
   fast-weight kernel's exactness degrade with binding count on real
   tokenized data, and is it geometry/optimization/kernel-precision"
   has independent prior art was **not** checked before writing this
   design — a research-agent novelty pass against this exact framing is a
   pre-registered prerequisite before any external write-up, not assumed
   clear by inheritance from RD-1's own (differently-scoped) novelty check.
7. **The trailing-pad cell (§5.3) assumes padding is truly state-neutral
   at the positions it touches.** This is asserted, not merely hoped, by
   the required two-sided blank-out smoke test (§5.3) — but if that smoke
   test is skipped or weakened under time pressure, a positive result on
   this cell could be an artifact of the padding leaking non-zero signal
   into the recurrence rather than genuine evidence for mechanism (e).
   The smoke test is listed as **required**, not optional, in Wave −1's
   gate for exactly this reason.
8. **Wave F's fix-success ≠ mechanism-proof circularity (PI addendum).**
   A working fix confirms the intervention moves the frontier; it does
   not by itself prove the mechanism story that motivated it (side-channel
   risk). Handled by §7's two-tier labeling — the demo claim and the
   attribution claim never collapse into one — and by the fact that the
   attribution rests on Waves 0–5's own evidence, not on Wave F's outcome.
9. **F-geo-1 is a Goodhart magnet.** It optimizes the very diagnostic
   (key-Gram deviation) the premise stack uses, so (i) that diagnostic is
   excluded from premise-evidence in F-geo-1 arms (§5.5), (ii) the success
   bar is behavioral (frontier shift), never the Gram number, and (iii)
   `λ_orth` is capped at a pre-registered pair — no iterative tuning. The
   residual risk that a penalty-shaped geometry passes the behavioral bar
   in some degenerate way is bounded by the unchanged eval readout
   (absolute cosine, pinned, §14.3's no-leak discipline) and the C17/C19
   held-out pools, which a degenerate solution would have to fool too.
10. **Two fix candidates = a small multiple-testing surface.** Both
    F-geo candidates are reported against the same pre-registered bars;
    if only one passes, the write-up reports both outcomes (the failed
    candidate is a result, not noise) and never presents the winner as
    if it had been the sole pre-registered fix.

---

## 9. Open questions (explicit, for the attack round)

- Does §3's Test B — provisionally under β≈1 at Wave 0, definitively with
  measured β at Wave 5 (Rev 2, finding 3: the Rev 1 phrasing "single most
  decisive free result" is retracted; the β≈1 pass carries no standalone
  attribution) — explain most of the K=32 gap? Everything downstream (how
  much weight Waves 3/4 deserve, whether the residual demands new
  mechanisms) is conditioned on the *measured-β* version, and the
  measured-β version arrives only after arm (iii-β) runs. Should Wave 0
  be split further (e.g., a cheap per-example rather than per-run
  aggregate reconstruction) if the aggregate residual is ambiguous?
- Is arm (i)'s 251-identity simultaneous-orthonormality budget (§4.2)
  actually still valid at build time, or has `heldout_frac`/the verified
  name-pool count drifted since the 2026-07-02 measurement this design
  cites? A one-line re-check, but currently unverified for this exact
  build.
- Should Wave 3's naive-reference training substitution be extended to a
  **fourth** arm — the production kernel's own bf16 chunked path but with
  a **larger** `chunk_size` override (if the kernel exposes one), to test
  "fewer, bigger chunks" as a cheaper proxy for the chunked-vs-sequential
  axis without building a second reference implementation? Not designed
  here; flagged as a possible Wave 3 extension if the attack round thinks
  the naive-reference confound (§8 item 5) is not adequately closed.
- Does mechanism (g)'s claim — that β is architecturally blind to
  within-episode write history — actually matter in practice, or does the
  entity-identity-conditioned β already implicitly encode "how often does
  this specific name co-occur with high-overlap neighbors across the
  training distribution," which could be a weaker but real substitute for
  true per-instance novelty-conditioning? This is not directly tested by
  any cell in this design (it would require a targeted probe of what `β`
  actually correlates with across the trained vocabulary) — named as a
  possible Wave 6 if Wave 5's synthesis leaves mechanism (g) as a live,
  unresolved contributor.
- If the attribution (Wave 5) does not sum to the full observed gap — some
  residual unexplained by (a) through (g) — what is the pre-registered
  next step? Not decided here; the attack round should weigh in on whether
  an unattributed residual below some threshold (e.g. <15% of the gap) is
  an acceptable place to stop, or whether it demands a further design
  iteration before any write-up.
- **Can any within-architecture fix reach Wave F's K=32 headline bar at
  `d_state=64` at all, given that K=32 = d/2?** §14.7 item 4 documents a
  sub-exactness plateau at exactly `K = d/2` as a cross-program echo of
  the bespoke encoder's own frontier (Stage 0's d-axis) — if the d/2
  boundary is itself binding, a geometry/loss/precision fix might hit the
  K=16 minimum-publishable bar while structurally unable to reach the
  K=32 headline. Pre-registered reading if that pattern lands: the result
  is interpreted against the d/2 boundary phenomenon (partial success,
  not fix-failure), and a `d_state=128, K=32` rider cell (~3 runs,
  Reserve-eligible) is the named follow-on probe — explicitly not run
  inside Wave F's own budget without a documented deviation.

---

## 10. Reproducibility pointers

- This design: `matrix-thinking/DELTANET_RD_EXACTNESS_DESIGN.md`
  (**Rev 3**, 2026-07-03 — Rev 1/Rev 2 same day; round-1 finding→change
  map is §11, round-2 map is §12).
- Builds on (read, not modified): `matrix-thinking/DELTANET_REALDATA_DESIGN.md`
  §14.7/§16/§17/**§18 (deephop, folded in at Rev 2)**,
  `matrix-thinking/DELTANET_CAUSAL_RANK_DESIGN.md` §12,
  `matrix-thinking/chapter2/TASK_E_FINDINGS.md` §3/§9,
  `matrix-thinking/chapter2/STAGE0_DESIGN.md` §14–15.
- Harness to extend (read in full this session, not modified):
  `matrix-thinking/deltanet_rd/{model_rd.py, run_deltanet_rd.py,
  grammar_rd.py, rank_utils.py, f15_lm_checkpoint.py,
  analyze_eval_truncation_rd.py}`. Key constants verified directly from
  source: `C16_SALVAGE_RATIO=0.1`, `C16_ALIGNMENT_COS=0.9`,
  `C16_VALUE_SALVAGE_RATIO=0.1`, `NCE_LAMBDA=1.0`, `NCE_T=0.1`,
  `_MIN_KERNEL_T=128` (`run_deltanet_rd.py`, `model_rd.py`).
- Data for Wave 0 (free, already local): `experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA,waveBprobe}/`.
- Next: ~~independent attack round~~ ~~round-2 verification~~ **(BOTH
  DONE — Rev 1 → attack round → Rev 2 → round-2 verification → this
  Rev 3; maps in §11/§12)** → build (Wave −1's new arms — (i)/(ii)/(iv)/
  (i-strong) tables, the F-geo-2 ZCA layer per §5.5's BLOCK-3 spec,
  naive-reference substitution, trailing-pad grammar variant, the §3.2
  β-dump + §3.3 `succ`/`tgt_slot`-dump additions) → audit by a
  fresh-context agent → Wave −1 on the cluster (incl. arm (iv)'s
  closed-loop `(α,ρ)` calibration probes) → Wave 0 (free, run first
  regardless of cluster queue state; extracts arm (iv)'s per-K target
  bands) → Wave 1 (incl. arm (iii-β) and co-primary (i-strong)) / Waves
  3/4 (parallel-launchable) → conditional waves per their gates → Wave 5
  measured-β synthesis + Wave F three-band gate verdict → **Wave F (full
  track, or the 40–50% reduced-scope fallback, §5.5) or the
  split-attribution design→attack loop**.

---

## 11. Rev 2 — attack-round responses (finding → change map)

The independent adversarial review of Rev 1 (2026-07-03) returned
NEEDS-REVISION: no FATALs, 8 MAJOR (1–8), 1 MODIFIES (9), 1 MINOR (10).
A **PI addendum** arrived in the same revision cycle (Wave F — the fix
demonstration) and is mapped as row **PI-A** below, incorporated into
this same Rev 2 rather than deferred to a Rev 3.
Disposition: **findings 1–5 and 7–10 accepted and implemented as
specified; finding 6 accepted with a stronger fix than the review
proposed** (exact span-projection instead of a measured-distortion JL
map); **PI-A accepted and implemented in full**. One scoping deviation,
declared: finding 1 suggested 2–3 fresh
arm-(iii) reruns; this revision runs **5** (K=16 ×2, K=32 ×3, ~2.9 GPU-h,
still inside the review's own 2–4 GPU-h estimate's spirit at the §6
anchor) because finding 5's ≥2-alignment-clean-seeds minimum applies to
arm (iii-β)'s K=32 cell and 3 seeds is the minimum credible first draw at
the observed ~1–2/7 clean rate. No finding is ignored.

| # | Finding (condensed) | Change made | Where |
|---|---|---|---|
| 1 | MAJOR — arm (iii) never gets the β-dump; mechanism (g) attribution would rest forever on β≈1 | Arm (iii-β) added to Wave 1 proper: 5 fresh learned-baseline reruns (K=16 ×2, K=32 ×3) with β (and `succ`/`tgt_slot`) dumps; Wave 5's synthesis re-runs Tests A/B with measured β and supersedes the β≈1 pass | §3.2 (items 2–3), §4.1, §6 (Wave 1 row, Wave 5 row) |
| 2 | MAJOR — §3.3 item 1 near-circular: same measured k/v on both sides, recursion ≡ kernel, only β free | §3.3 rewritten: Test A = β-idealization adequacy, Z_dump-internal, DIAGNOSTIC weight only (Rev 1's "structural evidence for (g)" claim withdrawn); Test B = a+g sufficiency vs. the independently measured §16.2 frontier, h=1-scoped on archived dumps per §17.2's own boundary, multi-hop on new dumps | §3.3, §7 (first tier bullet) |
| 3 | MAJOR — β≈1 imported from the orthonormal regime where β=1 is the exact fixed point; over-erasure argument on real keys | §3.2 rewritten: regime-import problem stated; Wave 0 β≈1 pass labeled PROVISIONAL everywhere, carries no standalone attribution; "single most decisive free result" retracted in §9; conditioned on finding-1's measured-β runs | §3.2, §8 item 3, §9 (first bullet) |
| 4 | MAJOR — promote arm (i-strong) to co-primary at K=32, parallel not serial (W_k has no isometry incentive; learned-arm Gram 1.26–2.77 is structural evidence of de-orthogonalization) | Promoted: (i-strong) K=32 ×3 runs in Wave 1 proper, in parallel with arm (i); Rev 1's conditional "1-strong" wave row deleted; §4.2's "conditional fallback" cross-reference updated | §4.4, §4.2, §6 (Wave 1 row) |
| 5 | MAJOR — "≥1 premise-valid seed" too permissive for a comparative K=32 headline at an observed ~1/7 alignment-clean rate | Pre-registered K=32 headline gate: ≥2 alignment-clean seeds per arm; contingency = add seeds not steps (+2, one iteration); failing arms report at descriptive tier, never threshold-rescued; ≤10-run/≤5.8 GPU-h contingency line added; cut-order rule 6 forbids silently keeping a headline while cutting its contingency | §6 (gate paragraph, contingency row, cut order 6) |
| 6 | MAJOR — JL at m=251, k=256 gives ε≈0.5; "approximately preserves angles" unsupported; measure or state honestly | Construction replaced outright: exact span-projection (orthonormal basis of the ≤251 used rows' span, r ≤ 251 ≤ 256) — an isometry on every row the task can see, zero distortion by construction; Wave −1 smoke retained as an exactness assert (<1e-5) rather than a distortion measurement | §4.3, §6 (Wave −1 row), §8 item 2 |
| 7 | MAJOR — arms (i)/(ii) bundle geometry with trainable-param count; build the Gram-matched control; patch "the ONLY variable" sentence | Arm (iv) built into Wave 1 proper (K∈{16,32} ×3, α-mix Gram-deviation-matched to Wave 0's per-K extraction, frozen = param-matched to (i)/(ii)); §4 intro rewritten — bundle named, (i)-vs-(iv) designated the geometry-only contrast, (i-strong) explicitly excepted as two-variable; §7 claim tiers relabeled per contrast | §4 intro, §4.5 (new), §6, §7, §8 item 1 |
| 8 | MAJOR — (c) may be upstream of (b): InfoNCE plausibly SETS v_eff geometry; attribution non-additive | Non-additivity caveat added to §3.4; zero-cost `value_gram_deviation_mean`-vs-λ_nce dependency readout attached to the NCE ablation (trigger widened so the readout also fires on a (b) signal); Wave 5's table presents (b)/(c) jointly unless dissociated | §3.4, §5.1, §6 (NCE row), §7 (Wave 5 bullet) |
| 9 | MODIFIES — fold in deephop (§18): supervision axis of (d) already negative; raise Wave 2's trigger bar; deephop does not bear on (c) | §18 added to §0 reading list and §2(d); §5.2 rewritten — supervision axis recorded as closed-negative with §18's numbers, Wave 2 trigger raised to a concrete pre-registered climbing-tail criterion (net rise ≥ +0.02 over final 5 checkpoints with positive trend), expected not to trigger; (c)-scope note included | §0, §2(d), §5.2, §6 (Wave 2 row) |
| 10 | MINOR — Wave 3's timing gate is a different overhead class than the svd_lowrank-calibrated buckets; extend §17.5's small-n caveat to §3 | Wave 3 gate restated as budget-anchored (×1.5 proceed / 1.5–2.5× drop naive-bf16 / >2.5× shrink to K=32 naive-fp32 only), with naive arms probed at both K in Wave −1; small-n caveat (4 examples/run) added to §3.3 with the aggregate-across-seeds + per-seed-spread mitigation | §5.4, §6 (Wave −1 row), §3.3 |
| PI-A | PI ADDENDUM (same revision cycle) — do not end at attribution: pre-register a mechanism-gated fix-demonstration wave with frontier-shift success criteria, §14.7 stack + finding-5 seed discipline, 15–25 GPU-h reserved cuttable last; reframe the goal statement | Wave F added (§5.5): per-mechanism intervention arms registered pre-data (F-geo: orthogonality penalty + key-path whitening, QR/Cayley rejected with justification and kept as F-geo-2's conditional swap; F-nce: fixed-m=7 crowding-normalized negatives; F-fp32: Wave 3's winner at frontier criterion); ≥50%-dominance launch gate + no-fix-fishing split branch; success bars pre-registered (K=32 h=4 0.009→≥0.5 headline; K=16 h=4 →≥0.8 minimum; h=1 no-sacrifice guard); goal statement reframed in the intro and §1; §6 gains the Wave F row, the 80-cap resolution paragraph, and cut-order rule 7 (Wave F protected last, deferred rather than run underpowered); §7 two-tier demo-vs-attribution labeling; §8 items 8–10; §9 K=d/2 boundary question | §1, §5.5 (new), §6, §7, §8, §9 |

**Net budget effect (Rev 1 → Rev 2, including the PI addendum):**
required attribution baseline 39 → 58 runs (~28.5 → ~36.5 GPU-h);
expected path (baseline + Wave F nominal, conditionals unfired)
**~51.5–54.5 GPU-h**; all-conditionals-max planning sum ~89 GPU-h on
paper — over the 80 cap by allocation, with the pre-registered §6
resolution (unfired conditionals roll into Wave F; cut-order 1–5 fire
before Wave F; Wave F trims internally to a ≈10–13 GPU-h floor or defers)
keeping **every realized path ≤80**. General Reserve 15 → 8 (Wave F's
15–25 band carries its own headroom). Cut order extended (arm (iv) K=16
as step 4; contingency-seed protection as rule 6; Wave F last-protection
as rule 7). *(Rev 3 supersedes some of these figures — see §12's budget
note; this paragraph is retained as the Rev 2 record.)*

---

## 12. Rev 3 — round-2 verification responses (finding → change map)

Round-2 verification of Rev 2 (2026-07-03) returned NEEDS-REVISION with a
substantial verified-clean core: **9/10 round-1 findings confirmed
genuinely addressed; the budget arithmetic recomputed exact; and the
verifier's own adversarial Welch-bound attack on Wave F's headline bar
FAILED** — at K=32 ≤ d_state=64 no coherence obstruction exists, and the
synthetic harness already achieves `rec@0.9 = 1.0` at this exact (d,K)
(§12.3 of `DELTANET_CAUSAL_RANK_DESIGN.md`), so the ≥0.5 bar is
achievable in principle and **stands unchanged** (recorded as an
achievability note in §5.5). Three new blocking MAJORs — all in Rev 2's
own new material — plus two minors. Disposition: **all three BLOCKs and
both minors accepted and implemented**; the BLOCK-1 reachability
calculation was executed and numerically verified this session (analytic
`sqrt(K(K−1)/d)` vs. Monte Carlo, agreement to 3 significant figures)
before being written into §4.5. No finding is ignored.

| # | Finding (condensed) | Change made | Where |
|---|---|---|---|
| BLOCK-1 | MAJOR — arm (iv) calibrates RAW rows (d=256) against a band MEASURED on `k_eff` (post-conv/W_k, d=64); no safeguard against §4.2's own de-orthogonalization warning; reachability unverified | (a) Calibration retargeted to **trained `k_eff`** via a closed-loop protocol: initial `(α,ρ)` from the raw arithmetic → 5,000-step calibration probe per K → deterministic-bisection adjustment → **≤3 iterations per K, hard cap** (instrument calibration against a fixed baseline-matching criterion, never outcome tuning; ~1 GPU-h, priced in Wave −1); (b) **re-verification demotion gate**: trained `k_eff` deviation off the per-K band by **>25% of band midpoint** at the full runs → (i)-vs-(iv) demoted to descriptive tier, §7's headline citation falls back to the bundled contrasts; (c) **reachability calc run and recorded**: `E‖G−I‖_F ≈ sqrt(K(K−1)/d)` → α∈[0,1] tops out at ≈0.97 (K=16) / ≈1.97 (K=32) in d=256 — cannot span the 1.26–2.77 band at K=16 — so the family is extended to two parameters (blend α + shared-direction ρ); measured: ρ=0.3 already gives ≈1.72/≈3.44, spanning the band with margin | §4.5 (rewritten), §6 (Wave −1 row, baseline total), §7 (conditional citation) |
| BLOCK-2 | MAJOR — the ≥50% dominance gate has no near-miss path; §1's own compounding-mechanisms prior makes a 45/35/20 split the modal outcome, structurally starving the deliverable; winner-disagreement unhandled | Three-band gate pre-registered: ≥50% → full track; **40–50% → reduced-scope fallback** (single top mechanism, ONE candidate — F-geo-2 by default per the structural-over-penalty house preference, unless Wave 1 localized damage to `W_k` — K∈{16,32} only, 3 seeds, **success bars UNCHANGED**: reduced scope trims cells, never standards); <40% everywhere → split-attribution design→attack loop (unchanged). **Tie-break**: when the Test-B attribution winner and the Wave-1 arm-gap-closure winner disagree, **the interventional evidence (arm-gap closure) selects the track** — consistent with §7's own tier hierarchy — and the disagreement is recorded and reported as evidence of attribution-model incompleteness | §5.5 (launch gate rewritten), §6 (cap paragraph notes the fallback's cost) |
| BLOCK-3 | MAJOR — F-geo-2's ZCA layer under-specified for buildability: pooling over ~75%-buffer streams would degenerate the covariance; no numerics (ε, update rule, warm-up) in a codebase with documented eigh failures; no smoke test | Buildable spec added: (a) **content-position-only pooling mandated** (token-id mask; buffer-only batches contribute zero mass, unit-checked); (b) numerics pre-registered — fp32 EMA moments (momentum 0.99, bias-corrected, BN-style stop-grad on statistics), `ε = 1e-4·tr(Σ)/d` scale-invariant regularization, transform recomputed every 100 steps from cached EMA, deviation-#6 jitter-retry + clamping fallback on eigh failure, 500-step identity warm-up with a hard recorded switch; (c) **required Wave −1 smoke**: finite outputs, `cond(Σ+εI) < 1e6`, post-warm-up `‖Σ_w−I‖_F < 0.1·d_state` on content positions, pooling-mask verification | §5.5 (F-geo-2 spec), §6 (Wave −1 row) |
| min-1 | MINOR — §4.3's "≤252" is a miscount | Corrected to **251** (213+21+16+1), with the miscount noted inline | §4.3 |
| min-2 | MINOR — the "~11.6 GPU-h freed by steps 1–4" figure was aggregated without decomposition (and was itself miscounted) | Decomposed per-step: 3.9 (Wave 4) + **~6 (Wave 3 naive-bf16 — flagged UNMEASURED until Wave −1's naive probes)** + 3.5 (Wave 1 K=8) + 1.7 (arm (iv) K=16) = **~15.1**; Rev 2's 11.6 had omitted step 3 | §6 (cap paragraph) |
| (kept) | Round-2's Welch-bound attack on the K=32 h=4 ≥0.5 bar FAILED — bar achievable in principle | Bar **kept unchanged**; achievability note recorded at the success-criteria table so the write-up can cite why the bar was not softened | §5.5 |

**Budget note (Rev 2 → Rev 3):** required baseline ~36.5 → **~37.5
GPU-h** (BLOCK-1's ≤6 closed-loop calibration probes, ~1 GPU-h, in Wave
−1); expected path **~52.5–55.5 GPU-h**; all-conditionals-max planning
sum ~90 on paper, with §6's worked worst-case check showing **no
realized path requires touching Wave F's floor to stay ≤80** (worst
realistic case: 57.0 attribution spend + Wave F at 25 = 82.0 → cut-order
step 1 alone (−3.9) restores ≤80). BLOCK-2's reduced-scope fallback
(~3.5–6 GPU-h) is strictly cheaper than the full track and never
stresses the cap.

---

## 13. Build-time clarifications (round-3 verifier, pinned before build)

Three points the round-3 verifier flagged as under-determined by Rev 3's
text alone — resolved here, at build time, rather than left for the
implementer to guess silently. None changes any pre-registered bar,
budget, or gate threshold; each is a reading of already-agreed intent.

1. **Wave F band-edge same-winner case (§5.5's three-band launch gate).**
   BLOCK-2 registered ≥50% (full track), 40–50% (reduced-scope), <40%
   (split-attribution loop), plus a tie-break for when M-attr and M-arm
   name *different* winners. Left open: what happens when the two
   measures name the **same** winner but land in *different* bands (e.g.
   M-attr = 47%, M-arm = 55%, same mechanism). Resolution: **the ≥50%
   generosity clause extends down to the same-winner case** — if
   **either** measure places the agreed winner at ≥50%, the full track
   launches; otherwise, if **either** measure places it in the 40–50%
   band, the reduced-scope fallback launches. (This is the natural
   extension of the tie-break's own logic — interventional evidence,
   M-arm, is already privileged when the two measures disagree on
   *who* wins; this clarification only says the more generous of the two
   *bands* is honored when they agree on *who*, so a mechanism is never
   starved of its full track by the more conservative of two
   non-contradictory measurements.) The <40%-for-every-mechanism branch
   is unaffected — it already requires both measures (and every
   mechanism) to miss 40%.
2. **ZCA insertion point (§5.5, F-geo-2's buildable spec).** The spec
   names "the post-conv key features" but does not pin the exact line.
   Resolution (the minimal-diff reading against `model_rd.py`'s existing
   code): the whitening transform sits **immediately post-conv, BEFORE
   the existing `F.normalize(k_conv)` step** — i.e. `k_conv → ZCA →
   L2-normalize`, in both `bind()`'s BIND-phase path and
   `effective_key_window()`'s query-phase path (the two sites that
   currently each compute `k_conv` then normalize). This keeps the
   L2-normalize's zero-safety contract as the last step before the
   kernel/comparison, unchanged from every existing arm.
3. **§6 Total row bookkeeping (required-baseline run count).** Rev 3's
   §6 table literally reads "58 + ≤7 probes." Summing the table's own
   **New runs** column for the waves the Total row claims to include
   (Wave −1's 9 short probes + Wave 1's 32 + Wave 3's 12 + Wave 4's 6;
   Wave 0 and Wave 5 are free) gives **9 + 32 + 12 + 6 = 59**, not 58;
   and BLOCK-1 pins arm (iv)'s calibration at **≤3 iterations × 2 K
   values (16, 32) = ≤6** probes, not ≤7. Corrected: **Total (required
   baseline, attribution only) = 59 runs + ≤6 probes**, ~37.5 GPU-h
   (the GPU-h figure itself was already computed correctly in Rev 3's
   budget note above and is unchanged by this correction — only the run
   *count* was mis-added).

---

## 14. F-geo-3 — per-episode differentiable key orthogonalization at the `k_eff` site

**STATUS: documented addendum to a BUILD-READY design — Rev B
(post-round-2-verification). This section is itself NOT build-ready.**
Two review rounds are DONE. **Attack round (→ Rev A):** NEEDS-REVISION,
no FATALs; verified clean by the attacker's own GPU probes: the
Newton-Schulz update rule's math, the `sqrt(K)` pre-scaling basin
proof, the β-mask inertness argument for non-write positions, and the
`bind()→readout()` threading; adversarial near-duplicate-row inputs
correctly triggered the eigh fallback with finite gradients throughout.
The load-bearing attack result was **F1** (MEASURED cross-episode
orthogonalized-key drift — a third failure mechanism the Rev-0 outcome
table conflated with incomplete orthogonalization); map in **§14.11**.
**Round-2 verification (→ this Rev B):** NEEDS-REVISION on CHECKs 1(b)
(the launch read cited an "F2 simulator" that existed only as the
attacker's session script, defined nowhere in the doc) and 2(a)/(c)
(the substitute admission stack's selectivity was asserted rather than
grounded, and it lacked any task-performance floor); CHECKs 3/4 PASS.
All four fixes are in this revision: the simulator now lives in the
repo (`matrix-thinking/deltanet_rd/geo3_simulator.py`,
CPU-smoke-tested) with the drift→simulator mapping **written down and
registered** (§14.6), the D/E/F drift split is numerically pinned
(§14.5), and the substitute gate gains a task-performance criterion
plus an honest comparability caveat (§14.10); map in **§14.13**. **One
final verification round (convergence check on Rev B) is expected
before any Wave −1 code is written** — the not-build-ready flag does
NOT clear until it passes. Nothing in §1–§13 is modified by this
section; F-geo-3 is a **third F-geo candidate**, proposed after the
two pre-registered ones (F-geo-1 penalty, F-geo-2 ZCA) both ran and
both missed the frontier-shift bars (`EXPERIMENT_LOG.md`, "Exactness
Wave F (soft arms)," 2026-07-04).

### 14.0 Why this candidate, stated against the fresh evidence

Two facts, both measured, both already on record:

1. **The exact solution exists and composes perfectly.** Arm (i-strong)
   (§4.4) — a non-trainable, hand-built, per-identity-fixed `k_eff`
   lookup that bypasses `W_k`/`k_conv1d` entirely — achieves key-Gram
   deviation **0.000** and `rec@0.9` **1.00/1.00/1.00** at h=1/2/3, K=32
   (`EXPERIMENT_LOG.md`, "Wave 1 ATTRIBUTION VERDICT," 2026-07-04). The
   architecture and the delta rule's own algebra can represent and use
   the perfect solution. This is not in question.
2. **SGD does not find it under either soft push tried so far.** F-geo-1
   (`L_orth` penalty, `λ∈{0.1,1.0}`) and F-geo-2 (population-level ZCA
   whitening) both land at K=32 key-Gram deviation **2.51–2.57** (vs.
   baseline 2.71–2.77) — a **3–8% cleanup**, nowhere near i-strong's
   0.000 — and recovery moves proportionally, not categorically: K=32 h=4
   `rec@0.9` 0.009→0.09–0.10 vs. the ≥0.5 headline bar; K=16 h=4
   0.42–0.47→0.47–0.52 vs. the ≥0.8 minimum-publishable bar
   (`EXPERIMENT_LOG.md`, "Exactness Wave F (soft arms)," 2026-07-04).
   `λ=0.1` vs. `1.0` are nearly identical (the penalty **saturates**);
   ZCA converges to the same basin. Both are population-level or
   soft-gradient nudges on a training objective that never *requires*
   exact orthonormality to reduce loss — SGD settles for "somewhat
   cleaner," not "exact," because nothing in either loss forces more.

F-geo-3 closes that gap the way i-strong does — by **structurally
guaranteeing** orthonormality of the K keys actually written into `S_T`,
every step — while remaining trainable end-to-end, unlike i-strong. It is
the deployable version of the surgical pin: the mechanism, not the oracle.

### 14.1 Design question 1 — which orthogonalization

**Setup.** At `bind()`'s existing gather site, `k_eff_items` is `(B,K,d)`,
`K≤d_state` (`K∈{16,32,48}` vs. `d_state=64`), rows already L2-normalized
by the existing `F.normalize(k_conv, dim=-1)` (zero-safe) call — this is
an **invariant F-geo-3's own numerics lean on**, see the pre-scaling
argument below. The problem: find `Q∈R^{K×d}` with `Q Q^T = I_K`
("orthonormal rows"), a smooth, differentiable function of the model's
own raw `k_eff_items`, so gradient still reaches `embed`/`k_proj`/
`k_conv1d`.

**Three candidates, one chosen.**

- **Gram-Schmidt — rejected outright.** Numerically poor (well known,
  and this codebase's own standing rule already treats sequential
  correction as suspect — see below), inherently sequential (no
  standard batched-GPU form), and **bakes in an arbitrary row order**:
  the first key of the K is left untouched, the last absorbs the most
  correction. There is a sharper, project-specific reason to reject it
  beyond "numerically poor": mechanism (g) (§2) already names the
  β-gate's *inability* to express a sequential, write-history-aware
  correction as the reason the delta rule's own uncorrected algebra
  degrades with K. Implementing the orthogonalizer itself as a
  sequential, order-dependent correction would silently reintroduce the
  same asymmetry one level up the stack — worse, it would make the
  orthogonalized key's dependence on `item_pos` order (an artifact of
  how `grammar_rd.py` lays out clauses, not of entity content) an
  explicit, differentiated part of the loss landscape.
- **QR (`torch.linalg.qr`, has autograd) — rejected as the primary
  mechanism, kept as a one-shot Wave −1 numerical cross-check only.**
  Two independent problems, not one: (i) **backward instability near
  rank-deficient input.** `torch.linalg.qr`'s documented gradient
  formula is well-conditioned only away from repeated/near-zero `R`
  diagonal entries — exactly the regime a barely-trained embedding table
  produces (K raw keys for K distinct entities that the model has not
  yet learned to separate are close to collinear at initialization and
  for a nontrivial fraction of early training). A `LinAlgError`-free but
  numerically exploding QR gradient here would look like ordinary
  training instability with no obvious cause. (ii) **QR-of-A^T
  (Gram-Schmidt under the hood) is order-dependent for the same reason
  Gram-Schmidt is** — it returns *a* valid orthonormal basis of `A`'s row
  space, not the one *nearest* to `A` (the orthogonal-Procrustes/polar
  solution). For a single fixed episode this does not break exact
  single-hop recovery (any orthonormal set satisfies `S_T @ k_j = v_j`
  for the delta rule's own algebra — nothing requires the *closest*
  rotation). But it **does** make the effective supervision target for a
  given entity's raw key direction depend on an arbitrary artifact of
  clause layout order, which is a worse optimization target than an
  order-independent map, for the same reason Gram-Schmidt is rejected
  above.
- **Cubic Newton-Schulz / Björck-Bowie iteration — chosen as primary.**
  The classical iteration for the orthogonal polar factor of a
  rectangular, full-row-rank matrix (`m≤n`; here `m=K≤n=d`):

  ```
  X_{t+1} = 1.5 X_t − 0.5 (X_t X_t^T) X_t
  ```

  converges quadratically to the `Q` with `Q Q^T = I_K` **nearest** to
  the pre-scaled input, provided the input's operator norm is `≤1` at
  `t=0`. Three properties make this the favorite, not just "the third
  option": (1) **order-equivariant, not order-dependent** — every step
  involves only `X_t` and `X_t X_t^T` (`K×K`); permuting `A`'s rows
  permutes `X_t X_t^T`'s rows and columns identically and permutes every
  iterate the same way — no row is privileged, closing the exact
  objection raised against Gram-Schmidt/QR above. (2) **Differentiable
  through ordinary matmuls, no LinAlg-specific backward formula** — the
  failure mode near-degenerate input produces is *slow convergence*
  (bounded), not an exploding/NaN gradient the way QR's backward can.
  (3) **Controllable strength** — the iteration count `n_iter` is a
  single, cheap, pre-registerable knob (fewer iterations = a softer,
  partial orthogonalization; more = closer to i-strong's exactness),
  unlike QR/Gram-Schmidt which are one-shot, all-or-nothing operations.

  **Pre-scaling (fixed, data-independent — no new backward dependency).**
  Because every row of `A = k_eff_items` already has unit norm (the
  invariant noted above), `σ_max(A) ≤ ‖A‖_F = sqrt(K)` unconditionally
  (`A A^T`'s trace is `K`; its top eigenvalue is at most the trace).
  `X_0 := A / sqrt(K)` therefore has `σ_max(X_0) ≤ 1` **always**, inside
  Newton-Schulz's convergence basin (squared singular values in `(0,3)`)
  with no data-dependent norm estimate computed at all — one fewer
  differentiable operation in the critical path than a naive
  `A/‖A‖_2`-style scaling would need.

  **What this buys, stated plainly (closes the "readout assumption"
  question, design question 4):** the delta rule's readout is `S_T @ k_j`
  — exact recovery requires the K written keys to be mutually
  orthonormal. With `Q Q^T = I_K` enforced by construction (up to the
  iteration's residual), the unbind is **exactly the synthetic harness's
  own regime** — this is the entire point of the intervention, not an
  incidental side effect.

### 14.2 Design question 2/3 — exact insertion spec (against `model_rd.py`'s real structure)

**Why this cannot be a `ZCAWhiten`-style shared transform.** F-geo-2's
insertion lesson was: apply the *same* transform (a single cached
`W_zca`, a population-level statistic, entity-context-free) at both
`bind()`'s write site and `effective_key_window()`'s query site. F-geo-3
**cannot reuse that pattern** — the orthogonalizing map is not a function
of key content alone, it is a **joint** function of the K keys that
happen to co-occur in *this* episode. There is no stable, context-free
`W` to cache and share. The consistency discipline the ZCA lesson
teaches — "the write site and the query site must agree" — still applies,
but the *mechanism* enforcing it must change: not a shared cached matrix,
but **literal reuse of the one already-computed per-episode tensor**,
threaded from `bind()` into the query path, never recomputed.
`effective_key_window()`'s existing contract ("compute `k_eff` from a
window in isolation") is broken by construction for this arm; F-geo-3
introduces a **parallel** query-resolution path rather than a fourth
branch inside `effective_key_window()` itself, which stays byte-identical
for every existing arm (additive, off-by-default, matching this file's
own standing regression discipline for every prior extension).

**Write side — inside `bind()`, a third branch alongside the existing
`strong_pin_active` branch (§4.4, mutually exclusive with it —
asserted at `__init__`):**

```python
# model_rd.py -- new module-level utilities, sec 14.1/14.2

def newton_schulz_orthogonalize(A: torch.Tensor, n_iter: int = 12
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """A: (B,K,d), K<=d, rows ALREADY L2-normalized (bind()'s existing
    zero-safe F.normalize -- the pre-scaling proof below relies on this).
    Returns (Q, resid): Q (B,K,d) with Q@Q^T ~= I_K per batch element;
    resid (B,) = ||Q Q^T - I_K||_F AFTER n_iter, detached (sec 14.1's
    Wave -1 fallback trigger, no_grad -- does not affect Q's own
    gradient path)."""
    B, K, d = A.shape
    X = A / (K ** 0.5)                            # sec 14.1 pre-scale: sigma_max(X)<=1 always
    I_K = torch.eye(K, device=A.device, dtype=A.dtype)
    for _ in range(n_iter):
        G = X @ X.transpose(-1, -2)               # (B,K,K)
        X = 1.5 * X - 0.5 * (G @ X)               # cubic Newton-Schulz / Bjorck-Bowie
    with torch.no_grad():
        resid = (X @ X.transpose(-1, -2) - I_K).norm(dim=(-2, -1))
    return X, resid


def _polar_via_eigh(A: torch.Tensor, eps_scale: float = 1e-4) -> torch.Tensor:
    """Fallback path (sec 14.4). REFACTOR NOTE: mathematically and
    procedurally the SAME jitter-retry-then-clamp discipline as
    ZCAWhiten._recompute_transform (sec 5.5/BLOCK-3) -- applied here to
    the per-episode (B,K,K) key Gram matrix instead of ZCA's (d,d)
    population feature covariance. Build item: refactor the shared
    eigh+jitter+clamp core into ONE utility both call, rather than
    duplicating it -- same math, same failure class (eigh non-convergence
    on a near-singular Gram matrix), same fix."""
    B, K, d = A.shape
    Sigma = A @ A.transpose(-1, -2)                            # (B,K,K)
    eye = torch.eye(K, device=A.device, dtype=A.dtype)
    eps = eps_scale * Sigma.diagonal(dim1=-2, dim2=-1).sum(-1) / K       # (B,)
    try:
        eigvals, eigvecs = torch.linalg.eigh(Sigma + eps.view(B, 1, 1) * eye)
    except _LinAlgError:
        jitter = eps + 1e-6 * Sigma.diagonal(dim1=-2, dim2=-1).sum(-1) / K
        eigvals, eigvecs = torch.linalg.eigh(Sigma + jitter.view(B, 1, 1) * eye)  # ONE retry,
        # matching bind()'s own force_rank_k retry-once-then-raise discipline (L771-780);
        # a second failure raises LinAlgError LOUDLY (kills the run) — RULED correct at
        # build audit (2026-07-04): geo3's orthogonalization IS the mechanism under test;
        # a silent skip would mask systematic degeneracy, and admission item 2 already
        # excludes fallback-leaning runs. (This comment previously claimed a
        # TruncationError/skip-step conversion that was never designed nor built.)
    eigvals = eigvals.clamp(min=eps.view(B, 1).expand(-1, K))
    inv_sqrt = eigvecs @ torch.diag_embed(eigvals.rsqrt()) @ eigvecs.transpose(-1, -2)
    return inv_sqrt @ A


def geo3_orthogonalize(k_eff_items: torch.Tensor, n_iter: int = 12,
                        resid_tol: float = 1e-2) -> torch.Tensor:
    """bind()'s F-geo-3 call site. Primary: batched, differentiable
    Newton-Schulz. Fallback: WHOLE-BATCH retry via the shared eigh-polar
    route if ANY episode's residual exceeds resid_tol after n_iter --
    batch-level, not episode-level, granularity (sec 14.4's documented
    scoping tradeoff -- matches bind()'s own retry-the-whole-computation
    convention for force_rank_k rather than introducing a new per-element
    dynamic-branching primitive)."""
    Q, resid = newton_schulz_orthogonalize(k_eff_items, n_iter=n_iter)
    if bool((resid > resid_tol).any()):
        Q = _polar_via_eigh(k_eff_items)
    return Q
```

```python
# model_rd.py -- inside DeltaNetRDBlock.bind(), replacing the existing
# single learned-path branch with a THIRD branch (i-strong's branch,
# sec 4.4, is unchanged above it):

if self.strong_pin_active:
    ...                                            # unchanged, sec 4.4, L737-751
elif self.geo3_active:
    k_norm_raw = F.normalize(k_conv, dim=-1)        # existing learned path, UNCHANGED up to here --
                                                     # F-geo-3 does NOT bypass k_proj/k_conv1d/embed
                                                     # (unlike i-strong); it operates strictly
                                                     # downstream of the model's OWN learned k_conv
    k_eff_raw = _gather_at(k_norm_raw, item_pos)     # (B,K,d) -- this episode's raw item keys
    k_eff_items = geo3_orthogonalize(k_eff_raw, n_iter=self.geo3_n_iter)  # (B,K,d), Q Q^T ~= I_K
    k_norm = k_norm_raw.clone()                      # non-write positions: the model's own raw
                                                       # conv output, architecturally INERT
                                                       # (beta=0 there -- kernel_state_design_layout's
                                                       # own math zeroes the entire update term)
    k_norm.scatter_(1, item_pos.unsqueeze(-1).expand(-1, -1, self.d_state),
                     k_eff_items.to(k_norm.dtype))    # substitute AT item_pos ONLY -- IDENTICAL
                                                       # scatter pattern to arm (i-strong), L748-751
else:
    k_norm = F.normalize(k_conv, dim=-1)              # existing learned baseline, byte-unchanged
    k_eff_items = _gather_at(k_norm, item_pos)

# unchanged below: S_T = kernel_state_design_layout(k_norm, v_conv, beta)
# now consumes the substituted k_norm; bind() returns k_eff_items (now the
# ORTHOGONALIZED set under F-geo-3) via the SAME 3/4-tuple contract every
# existing caller already expects ("EXACTLY the rows the kernel consumed" --
# C16's own instrument's premise, preserved).
```

**Query side (design question 3 — eval-path consistency).** `readout()`
gains two optional, default-`None` arguments; default behavior (both
`None`) is byte-identical to today, preserving the "additive,
off-by-default, regression-checked" discipline this file already applies
to every extension since §4:

```python
def readout(self, S_T, query_tokens, hops, k_eff_items=None, a_slot=None):
    if self.geo3_active:
        assert k_eff_items is not None and a_slot is not None, (
            "F-geo-3 requires bind()'s own k_eff_items and the batch's "
            "a_slot at query time -- effective_key_window() CANNOT "
            "recompute the correct (per-episode-joint) value from a "
            "window in isolation (sec 14.2)")
        d = k_eff_items.shape[-1]
        idx = a_slot.unsqueeze(-1).expand(-1, -1, d)
        q_eff = torch.gather(k_eff_items, 1, idx)     # (B,Q,d) -- DIRECT reuse of bind()'s own
                                                        # orthogonalized value; NEVER recomputed
                                                        # via embed/k_proj/k_conv1d
    else:
        B, Q, L = query_tokens.shape                  # unchanged, existing path
        q_eff = self.effective_key_window(query_tokens.reshape(B * Q, L)).reshape(B, Q, -1)
    return apply_state_power(S_T, q_eff, hops)


def forward(self, batch, force_rank_k=None):
    S_T, k_eff_items, v_eff_items = self.bind(batch, force_rank_k=force_rank_k)
    pred = self.readout(S_T, batch["query_tokens"], batch["hops"],
                         k_eff_items=k_eff_items if self.geo3_active else None,
                         a_slot=batch.get("a_slot") if self.geo3_active else None)
    ...                                                # unchanged target-gathering below
```

**Build item — `grammar_rd.py`, one new field.** `sample_batch_rd`
already computes the query's *source* slot at `a_slot =
torch.randint(0, K, (B, Q), ...)` (current file, line 452) and uses it to
derive `tgt_slot` and `query_key_ids` — but never returns it. Add
`"a_slot": a_slot,` to the function's returned dict (line ~462–466,
alongside the existing `tgt_slot`). Zero new compute (the tensor already
exists); Wave −1 smoke: byte-identical `token_ids`/`beta_mask`/every
other returned field with and without consuming the new key (trivial,
since it is a pure dict addition with no new tensor computed).
`a_slot`'s range is `[0,K)` by construction (`torch.randint(0, K, ...)`)
— it always indexes one of THIS episode's own K bound entities (queries
only ever ask about hops from an entity that was itself part of the
K-cycle), so `k_eff_items` is guaranteed to contain the row the query
needs; no out-of-episode lookup is ever attempted.

**Build item — the self-query C16 diagnostic (`run_deltanet_rd.py`,
current lines 212–218) must be patched, and this is load-bearing, not
cosmetic (§14.9 item 1 explains why; its Rev-A note records the F3
flip-side, resolved in §14.10).**

```python
# CURRENT (every existing arm):
sq = self_query_tokens(cfg, pools, b["key_ids"], b["rel_id"])
Bq, Kq, Lq = sq.shape
q_self = model.effective_key_window(sq.reshape(Bq * Kq, Lq)).reshape(Bq, Kq, -1)

# F-geo-3 branch (NEW): self_query_tokens(cfg, pools, key_ids, rel_id)
# builds windows DIRECTLY from key_ids IN SLOT ORDER (grammar_rd.py
# L469-482) -- identical order to item_pos/k_eff_items' own rows -- so
# "self a_slot" is trivially arange(K) per row; no new grammar_rd field
# needed for THIS call site, only this branch:
if model.geo3_active:
    q_self = k_eff_items          # IS the query-time value by construction
                                    # (slot j <-> row j, identity mapping) --
                                    # never independently recomputed
else:
    sq = self_query_tokens(cfg, pools, b["key_ids"], b["rel_id"])
    Bq, Kq, Lq = sq.shape
    q_self = model.effective_key_window(sq.reshape(Bq * Kq, Lq)).reshape(Bq, Kq, -1)
```

**Constructor additions** (`DeltaNetRDBlock.__init__`, additive, default
off, mirroring `use_zca`'s own pattern): `geo3_active: bool = False`,
`geo3_n_iter: int = 12`, `geo3_resid_tol: float = 1e-2`. Assert
`not (strong_pin_active and geo3_active)` — the two arms both replace
`k_eff_items` at write time by construction and combining them has no
defined meaning. `use_zca` composes with `geo3_active` in principle
(ZCA transforms the population distribution of `k_conv` **before**
F-geo-3's per-episode gather+orthogonalize) but this design does **not**
test the combination — explicitly out of scope, named as a possible
follow-on only if either arm alone partially helps.

### 14.3 Design question 4 — what can break

- **Information loss (key identity vs. value-side).** No. `v_conv` is
  never touched by `geo3_orthogonalize` — the delta rule's write term
  `β_j v_j k_j^T` uses the model's own learned `v_j` unchanged; only the
  key-side *direction* used to address that value is rotated. The
  orthogonalized key is a smooth, differentiable function of the raw key
  (which is itself a deterministic function of entity identity via
  `embed→k_proj→k_conv1d`), so gradient still reaches the raw
  key-producing weights — nothing is severed, only jointly rotated per
  episode before use.
- **Training instability through the orthogonalization's backward.**
  Addressed structurally (§14.1's order-equivariance and
  no-LinAlg-backward properties) and operationally (§14.4's stability
  plan: fixed data-independent pre-scaling, a bounded fallback, and a
  required Wave −1 gradient-finiteness smoke matching this file's
  standing "forward/backward/grad-finite" convention for every arm).
- **The readout's linear-unbind assumption.** `S_T @ k_j` recovers `v_j`
  exactly iff the K written keys are mutually orthonormal. F-geo-3
  enforces `Q Q^T = I_K` by construction (up to the iteration's
  residual) — the unbind becomes exactly the synthetic harness's own
  regime. This is restated from §14.1 because it is the mechanism's
  entire justification, not a side detail.
- **`T_bind` position bookkeeping.** No new risk beyond what the
  existing learned baseline already handles: F-geo-3's gather happens at
  `item_pos` (the VALUE/write positions), the SAME site the baseline and
  i-strong both already gather at — not `key_pos` (`item_pos−2`, the
  KEY token's own, separate, non-write position two slots earlier in the
  same clause). The scatter-back substitution targets `item_pos`
  exclusively, an exact copy of i-strong's own scatter call (§4.4,
  `model_rd.py` L748–751). Padding (`_MIN_KERNEL_T=128` for `K∈{8,16}`)
  appends trailing BUFFER positions **after** every real clause via
  `torch.cat` on the time dimension — `item_pos` indices are computed
  before padding and are unaffected by it (no reindexing needed).

### 14.4 Stability plan

- **Pre-scaling is fixed and data-independent** (§14.1's proof:
  `σ_max(A) ≤ sqrt(K)` unconditionally, given the pre-existing
  unit-row-norm invariant) — no extra differentiable norm estimate in
  the critical path, and no failure mode where the pre-scale itself
  could misfire.
- **Iteration count is a single pre-registered default, escalated at
  most once, never searched.** `n_iter=12` is the starting default,
  calibrated (not assumed) at Wave −1 against realistic — not only
  adversarial — inputs: an untrained model's K raw keys (the hardest
  realistic case, since entities are least differentiated at init) at
  `K∈{16,32,48}`. If `n_iter=12` does not reach `resid_tol=1e-2` on that
  probe, **one pre-registered escalation to `n_iter=20`, hard cap** —
  the same "instrument calibration against a fixed criterion, never
  outcome tuning" discipline arm (iv)'s `(α,ρ)` calibration already
  established (§4.5), not a new precedent. `resid_tol=1e-2` is chosen
  against the measured reference points already on record: i-strong's
  own Gram deviation is 0.000; the two soft arms' best cleanup left
  2.51–2.57 — a residual of `1e-2` after F-geo-3 would be a
  ~100–250× tighter fit than the soft arms achieved, close to
  i-strong's exactness without literally requiring bit-for-bit `Q Q^T=I`.
  **Now numerically grounded — and honestly relabeled conservative-not-
  derived (attack F2, kept as evidence).** The attack round measured, on
  GPU, how much final-key-set Gram residual the pre-registered bars
  themselves tolerate under idealized (drift-free) alignment: **K=16 h=4
  `rec@0.9` = 0.863 and K=32 h=4 = 1.000 at Gram residual ≈ 1.0**,
  cratering only by residual ≈ 2.5. So `resid_tol=1e-2` is **~100–300×
  tighter than the bars require** — it was chosen by analogy to
  i-strong's exactness, not derived from the bars, and is kept anyway
  (the tightness is nearly free at `(B,K,K)` sizes). The margin has a
  second, load-bearing consequence for attack F1 (§14.5): **moderate
  cross-episode key drift can in principle still clear the behavioral
  bars** — whether the measured drift at these K actually does is
  exactly what §14.6's F1 gating diagnostic answers before any Wave 1
  spend.
- **Fallback granularity is batch-level, not episode-level, and this is
  a documented tradeoff, not an oversight.** A per-episode dynamic
  branch (some rows in a batch use the Newton-Schulz output, others the
  eigh fallback) would require boolean-indexed gather/scatter under
  autograd — a pattern this codebase does not otherwise use. Instead,
  ANY episode in the batch exceeding `resid_tol` after `n_iter` triggers
  a WHOLE-BATCH retry via the shared eigh-polar route — matching
  `bind()`'s own existing retry-the-whole-forward-pass convention for
  `force_rank_k` (L771–780) rather than introducing a new primitive.
  Cost tradeoff, named: one pathological episode forces every other
  episode in that batch onto different (still correct, not
  bit-identical) numerics for that step — log a per-step
  fallback-triggered flag so any resulting loss-curve noise is
  diagnosable, not mysterious, if observed.
- **Gradient finiteness is a required Wave −1 smoke**, matching this
  file's own standing per-arm convention (i-strong's model-13 test,
  ZCA's model-14 test): forward+backward through `bind()` with
  `geo3_active=True`, assert every parameter's gradient is finite,
  at both a well-conditioned (mid-training-like) and an adversarial
  (near-duplicate-rows) synthetic input.
- **Interaction with `force_rank_k`: orthogonal mechanisms, not
  composed in Wave 1.** `force_rank_k` truncates `S_T` **after** the
  kernel call; F-geo-3 modifies the keys **before** it. They do not
  conflict architecturally, but F-geo-3's Wave 1 cells run at
  `force_rank_k=None` throughout, matching every other Wave F arm's
  own unconstrained-rank convention (§5.5) — combining an additional
  external rank cap with an already-structurally-forced orthonormal
  write geometry is a different, unregistered experiment.

### 14.5 Design question 6 — the i-strong-vs-F-geo-3 distinction, and what a failure would actually look like

**Name the distinction precisely.** Arm (i-strong) pins `k_eff_items[j]
:= u_{entity(j)}` — a **fixed, global, context-free** lookup: the same
64-dim vector for entity X in every episode, train and eval alike,
independent of which other entities co-occur with X in any given
K-cycle. F-geo-3's `k_eff_items[j]` is the output of a **joint**
transform over the K entities that happen to co-occur in *this specific
episode* — entity X's orthogonalized key is a function of `(X, the other
K−1 co-occurring entities)`, not of X alone. **Cross-episode key
consistency is architecturally guaranteed under i-strong and only
*learned* (indirectly, via the raw key projection's training-set-wide
gradient pressure) under F-geo-3.** This is the correct name for what the
user's design brief called "cross-episode key consistency is now
learned" — worth stating in exactly this form because it is the single
largest structural difference between the surgical pin and its
deployable version.

**Does this threaten composition (h≥2)? Check the claim against
`TASK_E_FINDINGS.md` §3's own mechanism, by name — REVISED (Rev A,
attack F1: Rev 0's answer here was incomplete, and the attacker
measured the missing piece).** §3's finding: a rank-`K−1` operator (one
missing eigenmode out of K) looks behaviorally correct at shallow hop
depth under a fixed cosine bar (`sqrt(1−1/K)≈0.94>0.9`) and is only
exposed by **repeated self-application of the same operator** (`Z^h`).
Rev 0 argued from this that within-episode orthonormality — which
F-geo-3 guarantees — is the load-bearing property for composition, and
concluded composition was safe. That argument covers the operator's
**spectral completeness** (the missing-rank channel §3 measured) and
that half stands. What it missed: `S_T^h` composition ALSO requires
each hop's retrieved **value** to function as the next hop's **key** —
`S_T @ q_a` returns (≈) the value-side representation of entity
`π(a)`, and for hop 2 that vector must behave like `k_eff_{π(a)}` when
`S_T` is applied again. Per-hop exactness therefore carries a
**value-to-key cross-alignment** factor `cos(v-rep_X-direction,
k_eff_X)` for the same entity X. `v_eff_X` is a pure per-identity
function (`v_proj`/`v_conv1d`, untouched by F-geo-3); under i-strong,
`k_eff_X` is also per-identity-fixed, so `W_v` has a **stable** target
and SGD demonstrably nails it (1.00/1.00/1.00). Under F-geo-3,
`k_eff_X` is **episode-conditional** — and the attack round measured
how much, directly: **fixing one entity's raw key and resampling its
K−1 episode-mates, the ORTHOGONALIZED key for that entity drifts to
pairwise cosine 0.94–0.95 (K=16), 0.87–0.89 (K=32), 0.80–0.82 (K=48)
across episodes, even with within-episode `resid ≈ 0`.** `W_v` can at
best learn the population-mean direction of a moving target it cannot
see (it never observes the episode context), so per-hop alignment
carries an ε set by the drift — and `ε^h` amplification (§3's own
mechanism, arriving through the cross-alignment channel rather than
missing rank) produces **graded h-decay at `resid ≈ 0`**. The honest
prediction is therefore **conditional**, not categorical: IF training
shrinks the drift (F-geo-3's gradient explicitly rewards raw keys that
are already near-orthogonal per episode — the smaller the needed
correction, the smaller its context-dependence), converged runs should
look i-strong-like; IF drift stays near the probe-measured levels,
graded h-decay at rates predictable from the measured cosines is
expected — and per §14.4's F2 margin measurement, **moderate drift may
still clear the bars**. §14.6's F1 gating diagnostic (per-checkpoint,
per-entity cross-episode spread) decides which regime is live before
Wave 1 spends its budget.

**The full risk picture — THREE separable failure signatures, not two
(Rev A; Rev 0's two-way separation is superseded).** `v_eff` and `β`
remain pure per-token functions of entity identity (via
`v_proj`/`W_beta`, unaffected by F-geo-3), so those stay stable across
episodes; only the key side becomes episode-relative. The signatures:

1. **Incomplete orthogonalization** (`n_iter` too low / `resid_tol` too
   loose): within-episode `resid > 0`, measurable directly on the
   logged residual — no other evidence needed.
2. **Cross-episode drift bottleneck (attack F1's named mechanism —
   "stable-not-just-orthogonal geometry")**: within-episode
   `resid ≈ 0` **+** high per-entity cross-episode spread (§14.6's F1
   diagnostic) **+** graded h-decay inside converged runs, steeper at
   higher K (matching the measured drift ordering 0.94-0.95 →
   0.87-0.89 → 0.80-0.82 at K=16/32/48). Rev 0 would have misread this
   signature as (1) — they are **decoupled from `resid_tol`** and only
   the drift diagnostic separates them.
3. **Optimization noise from the episode-conditional gradient target**:
   the effective gradient target for an entity's *raw* key direction
   varies with its K−1 co-drawn episode-mates, potentially making
   `W_k`/`k_conv1d`'s training signal noisier than the soft arms'
   (whose raw keys keep a stable identity-only target). Signature:
   seed-to-seed variance in recovery materially larger than
   F-geo-1/F-geo-2's own spread, or many seeds failing the §14.10
   substitute admission stack — **while any cleanly-converged run shows
   h=1/2/3 near-exact together AND low measured drift**.

**Numeric drift split, pinned before any data exists (Rev B, round-2
CHECK 1(c) — "high" and "low" were adjectives in Rev A; adjectives
cannot gate a mechanism claim).** Measured on the §14.6 F1 diagnostic's
pooled pairwise-cosine statistic at the TRAINED final checkpoint, per
K:

- **HIGH drift** = mean pooled pairwise cos **< 0.95** (at-or-worse-
  than the attacker's measured K=16 probe level — if a trained model
  still drifts as much as the worst untrained-probe cell, the drift is
  live) → eligible for signature 2 / outcome F.
- **LOW drift** = mean **≥ 0.98** → eligible for signature 3 / outcomes
  D and E.
- **0.95 ≤ mean < 0.98 = INDETERMINATE**: no mechanism claim is made in
  either direction — the run's behavioral numbers are still reported,
  but the D/E/F attribution line reads "indeterminate drift band" and
  any follow-on design must treat the drift mechanism as unresolved.

The write-up must report which signature was observed, not a blended
"it didn't work" — the three route to different follow-ons
(re-calibrate the iteration; a stability-targeted follow-on design,
§14.8 outcome F; accept-and-report per §14.8 outcome D).

### 14.6 Wave −1 smoke tests (required, blocking, before any Wave 1 spend)

1. **Convergence smoke, realistic and adversarial inputs.** Untrained
   (random-init) and briefly-trained (few-hundred-step) models' raw
   `k_eff_items` at `K∈{16,32,48}`; assert `resid ≤ resid_tol` within
   `n_iter=12` on the realistic case, and separately probe a
   hand-constructed adversarial case (two near-duplicate rows) to
   confirm the fallback triggers and itself converges (never an
   unhandled crash) — same "never crash mid-run" discipline as every
   other `_LinAlgError`-guarded path in this file.
2. **Gradient finiteness.** Forward+backward through `bind()` with
   `geo3_active=True`; assert every parameter gradient finite, at both
   inputs from item 1.
3. **Order-equivariance, numerically verified, not just argued.**
   Permute a test `A`'s K rows; confirm `geo3_orthogonalize`'s output
   permutes identically (closes §14.1's equivariance claim, which is
   otherwise asserted-not-measured, matching the same "asserted, not
   merely hoped" bar §8 item 7 already applies to the padding-neutrality
   claim elsewhere in this design).
4. **Self-consistency (the i-strong model-13 analog).** For a
   `geo3_active` model, `self_query_tokens` at slot `j`, routed through
   the §14.2 `q_self = k_eff_items` branch, must equal
   `k_eff_items[:, j, :]` **exactly** (a triviality check on the
   indexing, not a numerical-tolerance check — both sides are the
   identical underlying tensor by construction).
5. **`a_slot` bookkeeping.** Assert `a_slot` is always in `[0,K)`
   (trivial from `torch.randint`'s own range, checked anyway) and that
   `key_ids.gather(1, a_slot) == query_tokens[:, :, -3]` for every
   sampled batch — closes design question 4's `T_bind`-bookkeeping
   concern at the query path specifically (the write-path check is item
   6 below).
6. **QR cross-check (non-differentiable, one-off, `torch.no_grad()`
   only).** On a moderately-conditioned random test matrix, confirm
   `geo3_orthogonalize`'s converged output spans the **same row space**
   as `torch.linalg.qr(A.T).Q.T` (projector distance, not raw value
   comparison — the two methods are not expected to return the same
   rotation, §14.1) — verifies Newton-Schulz is solving the right
   problem (A's row space), not merely "some" orthonormal matrix.
7. **Regression / off-by-default.** With `geo3_active=False` (the
   default), `bind()`/`readout()`/`forward()` produce byte-identical
   output to the pre-extension baseline — the same standing check this
   file requires of every prior additive arm.
8. **Padding boundary.** `K=16` (`T_bind=112`, padded to 128): confirm
   the write-side scatter lands at the correct (pre-padding) `item_pos`
   indices post-`torch.cat` — a direct extension of the existing
   two-sided blank-out discipline (§5.3) to this new write path.
9. **R2-8 cross-clause leak invariant, RE-SCOPED (Rev A, attack F4).**
   The existing leak check asserts clause `j`'s effective key does not
   depend on clause `i≠j`'s tokens. Under `geo3_active` that invariant
   is **violated by design** at the post-orthogonalization level —
   `k_eff_items[j]` is a joint function of all K clauses, expected and
   intended. Re-scope: assert the invariant at the **raw,
   pre-orthogonalization `k_conv` level** (where the per-clause causal
   independence claim still holds and must keep holding — the causal
   conv is the only cross-position mixer on that path, window
   `conv_size=4`), and separately **assert-and-document the expected
   global coupling** of the post-orth tensor (corrupting clause `i`'s
   tokens MUST change `k_eff_items[j]`, `j≠i` — a positive control that
   the joint transform is actually live, the same
   negative-test-must-fail discipline as [model 10]'s transpose
   control).
10. **v-path blank-out / zero-gradient invariant under the new readout
    mechanics (Rev A, attack F4).** The F-geo-3 readout replaces
    `effective_key_window()`'s embed→k_proj→k_conv1d query forward with
    a gather from `bind()`'s own `k_eff_items` — a NEW gradient route
    from the recovery loss into the key path via the query side.
    Re-verify: (a) `v_eff_items` bit-identical with `geo3_active` on vs.
    off at fixed weights/batch (the v path is untouched by
    construction — assert it, don't assume it); (b) the existing
    two-sided blank-out on the v path (corrupt non-clause positions,
    v_eff unchanged) passes under the new forward; (c) gradient
    presence/absence pattern is as designed: `v_proj`/`W_beta` receive
    gradients, buffer rows receive none, and the query-side gather
    contributes finite (not exploding) gradient to
    `embed`/`k_proj`/`k_conv1d` — the analog of i-strong's model-13
    gradient-pattern assertions for THIS arm's different (non-bypass)
    topology.

**Wave −1 GATING DIAGNOSTIC (attack F1 — blocks Wave 1 spend; distinct
from the pass/fail smokes above. REWRITTEN, Rev B — round-2 CHECK 1(b):
Rev A's launch read cited "the F2 simulator," which existed only as the
attacker's session script. The simulator now lives in the repo, and the
mapping is written down here, registered before Wave −1 runs.)**

- **The simulator:** `matrix-thinking/deltanet_rd/geo3_simulator.py` —
  a minimal importable refactor of the attack round's own preserved
  probe script (`/home/nvidia/geo3_ns_check.py` on the box; also in the
  2026-07-03 session scratchpad as `geo3_ns_check.py`), the script that
  produced both the F2 bar-tolerance numbers and the F1 drift numbers.
  It simulates the EXACT delta-rule recursion (S@k design convention,
  β=1) over K keys at an injected key-Gram residual, with
  harness-faithful scoring (targets are the value representations, the
  `rec@0.9` absolute-cosine bar). CPU-smoke-tested at write-in: exact
  regime (c=1, resid=0) gives rec 1.0 at h=1–4; drift degrades deep
  hops while h=1 stays 1.0 (targets are v_eff — drift-immune at one
  hop, exactly §14.5's channel).
- **The REGISTERED drift→simulator mapping** (the formula, written
  before any Wave −1 data): cross-episode drift enters as the
  **value→key cross-alignment factor**. Each entity's value-side
  representation is a fixed unit vector tilted to cosine `c` from that
  entity's own episode key (deviation direction i.i.d. random
  orthogonal — `geo3_simulator.tilt_to_cos`), where `c` = the MEASURED
  mean pooled pairwise drift cosine; a second, worst-case run uses the
  p10 pairwise cosine. Rationale: `v_eff_X` is per-identity-only and
  cannot condition on episode context, so its achievable alignment to
  the episode-specific orthogonalized key is bounded by that key's own
  cross-episode consistency; the mean pairwise cosine is a conservative
  stand-in for cos-to-population-mean (which is ≥ mean pairwise cos for
  concentrated direction distributions). Compounding: hop 1 scores
  directly against the value representation (no `c` factor — eval
  targets ARE `v_eff`); each subsequent hop re-enters the retrieved
  value as a query, paying ~`c` per re-entry → ~`c^(h−1)` on top of the
  Gram-residual term, PLUS interference from the orthogonal deviation
  component re-reading `S`, which the simulation captures and no closed
  form does. (The round-2 instruction's "`c^h`" shorthand is the
  conservative analytic bound; this simulation-based mapping is
  registered as the refinement, per that instruction's own allowance —
  the load-bearing registered object is the SIMULATION with the tilt
  construction above, not either closed form.) Keys in the launch-read
  simulation sit at the post-orthogonalization guaranteed residual,
  `gram_resid = resid_tol = 1e-2`.
- **The measurement (statistic + sampling spec, pinned):** **≥8
  entities per K**, drawn randomly WITHOUT replacement from the train
  name pool; **≥32 episode-context resamples per entity**; per entity,
  orthogonalize each resampled episode and collect that entity's output
  row; aggregation = pairwise cosines pooled WITHIN entity across
  resamples, then pooled across entities; report **mean and p10** of
  the pooled distribution (`geo3_simulator.pairwise_drift_stats` per
  entity). Run per `K∈{16,32,48}`, at init and at a briefly-trained
  probe checkpoint — and logged per-checkpoint during any Wave 1 run
  that launches, so §14.5's "does training shrink drift" question gets
  a measured answer. Attack-round reference values (probe conditions,
  single entity, 200 resamples): 0.94–0.95 (K=16), 0.87–0.89 (K=32),
  0.80–0.82 (K=48), at within-episode `resid ≈ 0`.
- **The launch read (`geo3_simulator.launch_read`):** run the
  simulation at the measured **mean** drift and at the **p10** drift,
  K∈{16,32}. **GATE: predicted K=16 h=4 `rec@0.9` ≥ 0.8 under the MEAN
  mapping** (the minimum-publishable cell). The p10 run and both K=32
  predictions are reported alongside, non-gating — K=32 because §14.5's
  shrinking-drift-under-training hypothesis is precisely what only a
  real run can test and F2's margin leaves it plausible; p10 because
  gating on a tail statistic of an untrained-probe measurement would
  double-count conservatism already built into the mean-pairwise choice.
  If K=16's prediction fails at the measured drift, Wave 1 does not
  launch as-is: the stability question routes to a follow-on design
  (§14.8 outcome F's named follow-on), never to on-the-fly fix
  iteration — the base design's standing no-fix-fishing rule.

### 14.7 Design question 5 — cells + budget

**Scoping note.** Wave F's **full** F-geo track (both pre-registered
candidates) already ran and both missed the bars — the §5.5/BLOCK-2
three-band dominance gate that would normally set cell count is not the
live question here (dominance was already resolved: "same winner, ≥50%
on both measures → WAVE F FULL TRACK," `EXPERIMENT_LOG.md`, 2026-07-04).
F-geo-3 borrows **BLOCK-2's reduced-scope-fallback template** — one
candidate, `K∈{16,32}` only, 3 seeds, **success bars unchanged** — as the
appropriate budget-scoped precedent for "one more candidate, minimal
cells, standards never relaxed," even though the gate that originally
produced that template's shape was about mechanism dominance, not
candidate exhaustion. This is a deliberate reuse of an already-attacked
scoping decision, not a new one.

| Cell | Seeds | Steps | Notes |
|---|---|---|---|
| K=16 | 3 | 20,000 | mandatory — minimum-publishable headline |
| K=32 | 3 | 20,000 | mandatory — primary anomaly / headline |
| K=48 (stretch) | 3 | 20,000 | **optional, Reserve-eligible** — only 16 spare dimensions of `d_state=64`; feasibility (name-pool ≥48 **per pool, not in total** — the 213 verified names split at `heldout_frac=0.5` into ~106/107 train/held-out, and each pool alone must cover a K=48 draw; both do, ~2.2× margin. Rev 0 cited "251 available," which counted rel/period rows and both pools together — the wrong object, corrected per attack F9; `_MIN_KERNEL_T`/chunk-boundary safety at `T_bind=7·48=336`) verified at Wave −1, not assumed; the single hardest cell for Newton-Schulz convergence (§14.9 item 3) **and the highest-drift cell in the F1 diagnostic (measured 0.80–0.82)** |

Same `force_rank_k=None`, same 20,000 steps, same §5.5 success bars:
**K=32 h=4 `rec@0.9` ≥0.5 headline; K=16 h=4 ≥0.8 minimum publishable;
h=1 no-sacrifice guard within −0.02 of the arm's own baseline.**
Per-seed admission: the finding-5 `≥2`-clean-seeds count and
add-seeds-not-steps contingency apply unchanged, but against the
**§14.10 substitute admission stack** (Rev A, attack F3) — the standard
alignment/key-Gram stack is tautological under this arm.

**Budget.** 6 mandatory runs at the §6 anchor (~0.58 GPU-h/run) ≈ **3.5
GPU-h**; +3 optional K=48 runs at a modestly higher per-run cost (longer
`T_bind`) ≈ **1.7–2.0 GPU-h** more; Wave −1's 10 smoke items + the F1
gating diagnostic (short probes, cheap) ≈ **0.5–1 GPU-h**. **Total:
~4–6.5 GPU-h** (mandatory-only floor ~4.0–4.5; with the K=48 stretch,
~5.7–6.5 — Rev 0's "7.0" upper figure was a mis-add, corrected per
attack F8: 3.5 + 2.0 + 1.0 = 6.5) — comfortably inside this design's
already-reserved Wave F headroom (§6), no new manifest-row budget
request needed.

### 14.8 Pre-registered predictions per outcome

- **A — bars hit at both K, h=1 preserved, seed variance comparable to
  i-strong's tightness.** Mechanism (a)/(g)'s attribution fully
  vindicated as *fixable by a trainable structural intervention*, not
  just a soft-push-resistant attractor. F-geo-3 becomes Wave F's winning
  candidate; the program's constructive deliverable is met.
- **B — bars hit at K=16 but not K=32.** Read against §9's own
  already-flagged `K=d/2` boundary question (`K=32=d_state/2`), **not**
  as a fix failure — partial success at the structural boundary. The
  already-named `d_state=128, K=32` rider cell (§9) becomes the natural,
  cheap follow-on, not a new design.
- **C — composition (h≥2) exact/near-exact within converged runs, but
  the recovery bars still miss because of a different bottleneck
  (e.g. mechanism (c) NCE-crowding or (b) value geometry reasserting
  itself once key geometry stops being the binding constraint).**
  Informative, not a failure: key geometry was necessary but not
  sufficient — consistent with §1's own standing prior that multiple
  mechanisms compound. Next lever is (b)/(c), already-designed cells
  (§5.1/§3.4).
- **D — gradient/convergence instability** (Wave −1 or Wave 1 shows
  non-finite grads, divergent loss, or a large fraction of seeds failing
  the §14.10 substitute admission stack) **with LOW measured drift in
  cleanly-converged runs** (§14.5's pinned band: mean pooled pairwise
  cos ≥ 0.98 at the trained checkpoint — the drift diagnostic is what
  separates D from F; 0.95–0.98 is indeterminate, no mechanism claim,
  per §14.5). The §14.5 "noisy per-identity target" risk
  materializes as seed-level noise, not within-run decay. Publishable
  negative branch per §5.5's own discipline — read as evidence that
  **structural forcing trades attractor-mismatch for optimization
  noise**, not a free lunch relative to the soft arms.
- **E — orthogonality achieved (confirmed near-0 residual by
  construction), LOW cross-episode drift (F1 diagnostic, §14.5's pinned
  band: mean ≥ 0.98), but recovery does NOT improve categorically more
  than the soft arms already showed.** The most scientifically puzzling possible outcome: with
  both orthogonality AND stability measured in hand, the i-strong gap
  would have to live somewhere not yet named. Would require its own
  follow-on design, not an iteration inside this one (no fix-fishing,
  per the base design's standing anti-Goodhart rule). *(Rev A note:
  Rev 0's E bundled this with what is now outcome F — the F1
  diagnostic is what splits them.)*
- **F — stable-not-just-orthogonal geometry is the bottleneck (NEW,
  Rev A — attack F1's named mechanism).** Discriminating signature,
  pre-registered (§14.5 signature 2): within-episode `resid ≈ 0`
  (orthogonalization complete — NOT distinguishable from success on the
  residual alone) **+** HIGH per-entity cross-episode key spread
  (§14.5's pinned band: mean pooled pairwise cos < 0.95 at the trained
  checkpoint; probe reference 0.94-0.95 / 0.87-0.89 / 0.80-0.82 for
  K=16/32/48) **+** graded h-decay within converged runs,
  steeper at higher K, at rates predictable from the measured cosines
  via the registered simulator (`geo3_simulator.simulate_recovery`,
  §14.6). Read: within-episode orthonormality is real but
  insufficient — the property i-strong actually demonstrated bundles
  orthogonality WITH **cross-episode key stability**, and joint
  per-episode orthogonalization supplies only the first while `W_v`
  cannot chase the second. This isolates "a *stable, entity-fixed* key
  identity" as the binding requirement — something no
  purely-within-episode intervention can provide — and routes to a
  stability-targeted follow-on design (e.g. an EMA-anchored or
  identity-registered orthogonalization — named as a direction, NOT
  designed here). Without the F1 diagnostic this outcome would
  masquerade as incomplete orthogonalization; with it, the verdict is
  clean either way.

### 14.9 Attack-yourself (F-geo-3-specific — new items, continuing the base design's numbering scope as its own local list)

1. **A found bug, not a hypothetical one: the existing self-query
   alignment diagnostic silently poisons F-geo-3's own premise-validity
   gate unless patched.** `run_deltanet_rd.py`'s C16 alignment
   instrument (`cos_align = F.cosine_similarity(k_eff_items, q_self)`,
   feeding `align_mean`/`align_min`/`align_valid` — the exact quantity
   §6's finding-5 `≥2`-alignment-clean-seeds gate reads) computes
   `q_self` via `effective_key_window(self_query_tokens(...))` — the
   **raw, pre-orthogonalization** path — for every existing arm. Left
   unpatched under F-geo-3, `q_self` would differ from
   `k_eff_items` (orthogonalized) by exactly the orthogonalization
   correction, on **every** batch, for **every** seed — every F-geo-3
   run would appear to have near-zero alignment-clean checkpoints and
   be demoted to descriptive tier by §6's own gate **even if the fix
   works perfectly**. §14.2's patch (route `geo3_active` through
   `q_self = k_eff_items` directly, using the slot-order identity that
   `self_query_tokens` already guarantees) closes this — but it is a
   **build-blocking correctness item**, not a nice-to-have. *(Rev A:
   the attack round re-derived the patch as necessary — and surfaced
   its flip-side, attack F3: the patched diagnostic is tautologically
   1.0, which compromises the finding-5 admission GATE itself, not just
   the instrument. The substitute per-seed admission stack is
   pre-registered in §14.10.)*
2. **`effective_key_window()`'s broken contract is a standing landmine
   for future extensions, not just today's two call sites.** Any FUTURE
   diagnostic or analysis script that calls `effective_key_window()`
   directly on an in-episode query window, under `geo3_active`, will
   silently recompute the **wrong** (pre-orthogonalization) value with
   no error raised. §14.2 closes the two call sites this design touches
   (`readout()`, the self-query diagnostic); it does not — and cannot —
   close every future one. Recommend (not built here, flagged for the
   attack round to weigh in on): a cheap, logged-not-asserted debug
   cross-check comparing `effective_key_window()`'s raw output against
   the slot-gathered `k_eff_items` value whenever both are available, so
   a future silent-mismatch bug surfaces as a visible log line rather
   than a mystery.
3. **K=48 is the single most likely Wave −1 failure point, by
   construction, not by bad luck.** It has the least spare dimensionality
   (16 of 64), sits past the already-documented `K=d/2` hard boundary
   (§9), and combines with early-training near-collinear raw keys — the
   regime Newton-Schulz converges slowest in. If Wave −1's convergence
   smoke (§14.6 item 1) is going to fail anywhere, it is expected to fail
   here first; budget the eigh-fallback path to be genuinely exercised
   and verified at this cell specifically, not only at a synthetic
   adversarial matrix.
4. **Batch-level fallback granularity is a training-dynamics wrinkle,
   named in §14.4, restated here as an attack surface**: numerics
   silently alternate step-to-step between two mathematically-equivalent
   but not bit-identical solvers whenever any one episode in a batch is
   pathological. Could manifest as loss-curve noise with no obvious
   correlate. Mitigation is logging only (the fallback-triggered flag,
   §14.4) — if this design is attacked and the reviewer judges
   batch-level granularity too coarse, the ONE alternative already
   considered and rejected (per-episode dynamic branching) should be
   re-litigated there, not silently reintroduced at build time.
5. **Goodhart / diagnostic independence, sharper than F-geo-1's own
   caveat.** F-geo-1 merely *optimizes toward* low key-Gram deviation
   (§8 item 9); F-geo-3 **guarantees it by construction, every step**.
   The key-Gram diagnostic is not just compromised as premise-evidence
   under this arm, it carries **zero** information about whether the
   fix works — it is a near-constant by design. Only the behavioral
   frontier-shift metric and the h=1 no-sacrifice guard carry any
   evidentiary weight here; this must be stated with more force than
   §7's existing Wave F caveat, not merely inherited verbatim.
6. **Order-equivariance is asserted in §14.1, not yet measured.** §14.6
   item 3 is the required close; until it is run, treat the claim as
   argued-not-verified, in the same bucket as §8 item 7's
   padding-neutrality claim before its own required smoke closed it.

### 14.10 Claim-tier language + the substitute admission gate (Rev A, attack F3)

Inherits §7 exactly, no new standard. Wave F's own two-tier separation —
"the fix moves the frontier" (the demonstration claim) vs. "therefore the
attribution was right" (a supported inference, never proven by the demo
alone) — applies verbatim to F-geo-3. One item is **sharpened, not
new** (§14.9 item 5), and extended per attack F9: F-geo-3's key-Gram
diagnostic is excluded from premise-evidence **more completely** than
F-geo-1's own carve-out — it is architecturally forced to ≈0 by
construction, so it is not merely downweighted, it is uninformative —
**and the same is true of the per-item alignment instrument** under
§14.2's q_self patch (identically 1.0, the two sides are one tensor).
Both are logged, both are labeled non-evidence in every output. Success
is the behavioral bar (`rec@0.9` vs. K, h) alone, exactly as §5.5
already requires of every F-geo arm.

**The admission GATE, not just the diagnostics, is compromised — and a
substitute is pre-registered NOW (attack F3, MAJOR, accepted).** §6's
finding-5 rule admits seeds to headline eligibility via ≥2
alignment-clean seeds under the `DELTANET_REALDATA_DESIGN.md` §14.7
stack (the base design's usage; not this section's own §14.7, an
unfortunate numbering collision) — a bar every other arm
must *earn* (the archived K=32 learned arm passes it at only ~1–2 of 7
seeds). Under F-geo-3, alignment ≡ 1.0 and key-Gram ≡ resid ≈ 0 **by
construction**: 100% of geo3 seeds would pass automatically, making
any cross-arm K=32 headline comparison structurally asymmetric — a
gate that filters one arm's seeds and rubber-stamps the other's is not
one gate. **Substitute per-seed admission criterion for `geo3_active`
runs, pre-registered before any data exists:**

1. **Value-side salvage tier** `σ_K/σ_1 ≥ 0.1` on `v_eff_items` — the
   one base-stack instrument F-geo-3 leaves fully independent
   (`v_conv` is untouched by the orthogonalization, §14.3);
2. **Newton-Schulz converged WITHOUT the eigh fallback at every logged
   checkpoint** — read from the per-step fallback flag §14.4 already
   mandates logging (a seed that leaned on the fallback is not
   premise-clean for a claim about the *differentiable* mechanism);
3. **Finite loss throughout, no divergence** (standard, but now
   load-bearing since the alignment instrument can no longer catch a
   degenerating key path);
4. **Task-performance floor (NEW, Rev B — round-2 CHECK 2(c): items
   1–3 alone admit a run that never learned the task).** Final train
   loss must improve on the untrained-model loss (same seed, step-0
   measurement, already logged at every run's first checkpoint) by
   **≥50%**, AND h=1 in-distribution `rec@0.9` **≥ 0.5** at the final
   checkpoint. An F-geo-3 run that cannot do single-hop lookup is not
   admissible evidence about composition regardless of finite loss —
   0.5 is deliberately a *floor for admissibility*, far below the
   learned baseline's own h=1 (0.78 at K=32), never a success bar.

R2-8's held-out-pool classification is **kept** (it reads behavior on
held-out entities, not key geometry — uncompromised). The finding-5
**count** (≥2 clean seeds per arm for any K=32 cross-arm headline) and
the add-seeds-not-steps contingency apply unchanged against this
substitute stack; failing arms report at descriptive tier exactly as
§6 already prescribes.

**Comparability is UNVERIFIED, stated plainly (Rev B — round-2 CHECK
2(a): Rev A implied the substitute stack restores cross-arm symmetry;
it does not, yet).** Per the archived data, value-side salvage is a
**low-selectivity** criterion — post-fix baseline runs pass it
near-universally; the old gate's actual selectivity came from the
per-item **alignment** criterion, which is exactly the instrument
F-geo-3 tautologizes and excludes. Items 2–4 add selectivity of a
different KIND (solver health, task competence), but whether the
substitute stack's realized pass rate is comparable to the standard
stack's (~1–2 of 7 at K=32) is an **empirical question with no data
until Wave −1/Wave 1 produce pass rates** — it is not assumed. Until
then: any cross-arm headline comparison involving a geo3 arm must
state, explicitly and wherever a seed count appears, that the two
arms' admission stacks differ, what each gate filters, and the
realized pass rate under each — a geo3 "≥2 clean seeds" is not the
same evidentiary event as a learned-arm "≥2 clean seeds," and the
write-up must never present them as interchangeable.

### 14.11 Rev A — attack-round responses (finding → change map)

The independent attack round on §14 (Rev 0) returned **NEEDS-REVISION,
no FATALs**. The attacker ran GPU probes, not just a read: the
Newton-Schulz update rule's math, the `sqrt(K)` pre-scaling basin
proof, the β-mask inertness of non-write positions, and the
`bind()→readout()` threading were all **verified numerically** (no
change needed), and adversarial near-duplicate-row inputs correctly
triggered the eigh fallback with finite gradients. Finding numbers
below are the attacker's own, as relayed — non-contiguous because
findings that verified clean carry no change row. Disposition: **every
relayed finding accepted and implemented; F2 is a positive
measurement, kept as evidence rather than triggering a change to the
tolerance itself.** Rev 0's own §14.11 attack-checklist items are
answered: (i) the q_self patch was re-derived as necessary — and its
tautology flip-side became F3; (ii) the Newton-Schulz-vs-QR choice
survived numerical verification; (iii) `n_iter`/`resid_tol` are now
grounded by F2 (conservative-not-derived, kept); (iv) the
within-run-vs-across-seed separation did NOT fully hold up — F1 found
the third signature it was missing.

| # | Finding (condensed) | Change made | Where |
|---|---|---|---|
| F1 | MAJOR — cross-episode key-identity instability, **measured**: fixing one entity's raw key and resampling its K−1 episode-mates, the orthogonalized key drifts to pairwise cos 0.94–0.95 (K=16) / 0.87–0.89 (K=32) / 0.80–0.82 (K=48) across episodes at within-episode `resid ≈ 0`; joint orthogonalization makes `k_eff_X` episode-conditional and per-identity `v_eff_X` cannot chase it; masquerades as "incomplete orthogonalization" (graded h-decay) in Rev 0's outcome table while being **decoupled from `resid_tol`** | (a) Wave −1 **gating diagnostic** added — per-entity cross-episode orthogonalized-key cosine spread (≥32 context resamples, per K, per checkpoint), with a pre-registered launch read via the F2 simulator (K=16 h=4 prediction must clear ≥0.8, else no launch, route to follow-on design); (b) §14.8 gains outcome **F** ("stable-not-just-orthogonal geometry is the bottleneck") with the three-part discriminating signature (resid≈0 + high spread + graded h-decay steeper in K), D and E amended to cite the drift level that separates them from F; (c) §14.5's composition-safety answer rewritten — the §3-based argument covers spectral completeness only; the value→key cross-alignment channel is named, the prediction is now conditional on trained drift, and the two-way failure separation is superseded by a three-way one | §14.6 (gating block), §14.8 (D/E/F), §14.5 |
| F2 | Positive evidence, kept — bars tolerate final-key-set Gram residual ≈1.0 under idealized alignment (K=16 h=4 0.863; K=32 h=4 1.000 at resid 1.0; craters by ≈2.5) | `resid_tol=1e-2` relabeled **conservative-not-derived, now numerically grounded** (~100–300× tighter than the bars require); the margin's F1 consequence recorded — moderate cross-episode drift may still clear the bars, which is exactly what the F1 gating diagnostic decides pre-spend | §14.4, §14.5, §14.6 |
| F3 | MAJOR — the finding-5 admission gate becomes tautological for geo3 runs (patched `q_self ≡ k_eff_items` → alignment ≡ 1.0; key-Gram ≡ 0 by construction): 100% of geo3 seeds would pass a gate other arms must earn | Substitute per-seed admission criterion pre-registered: value-side salvage tier (untouched by geo3) + Newton-Schulz-converged-without-fallback at every logged checkpoint + finite-loss/no-divergence; R2-8 kept (behavioral); finding-5 count + add-seeds contingency unchanged against the substitute stack; gate-compromise (not just diagnostic-compromise) stated in the claim-tier section; §14.7's admission sentence repointed | §14.10 (rewritten), §14.7, §14.9 item 1 |
| F4 | Smoke coverage gaps: R2-8 cross-clause leak invariant is violated by design post-orth; v-path blank-out/zero-grad never re-verified under the new readout mechanics | Smoke item 9 added — R2-8 re-scoped to the raw pre-orthogonalization `k_conv` level, with the post-orth global coupling asserted-and-documented as a positive control; smoke item 10 added — v_eff bit-identity geo3-on-vs-off, v-path blank-out re-run, and the gradient presence/absence pattern (incl. the NEW query-side gradient route into the key path) asserted | §14.6 items 9–10 |
| F8 | MINOR — K=48 stretch budget upper bound mis-added (7.0; correct sum 3.5+2.0+1.0 = 6.5) | Corrected to **~5.7–6.5**, headline total to ~4–6.5 GPU-h, mis-add noted inline | §14.7 |
| F9 | MINOR — K=48 name-pool feasibility cited "251 available" (rel/period rows + both pools — the wrong object); compromised-instrument language too narrow (key-Gram only) | Feasibility restated per pool: 213 names split at `heldout_frac=0.5` → ~106/107 train/held-out, each ≥48 with ~2.2× margin; §14.10's exclusion extended to the alignment instrument (identically 1.0 under the patch), both logged-but-non-evidence | §14.7, §14.10 |

### 14.12 Sequencing

This section still does not authorize any spend. Two review rounds are
done (attack → §14.11; round-2 verification → §14.13 — the round-2
scrutiny targets were exactly the two Rev-A additions this section
pre-registered for it: the launch-read rule and the substitute stack's
comparability claim, and both drew findings). Remaining sequence:
**final verification round on Rev B (convergence check — confirm each
§14.13 change closes its CHECK; expected the last round before build)**
→ build (the §14.2 diffs + the §14.6 smoke suite;
`geo3_simulator.py` already exists in the repo, CPU-smoke-tested, and
is itself audit-scope) → independent code audit per house rule →
Wave −1 (smokes 1–10, then the F1 gating diagnostic and its registered
`launch_read`) → Wave 1 mandatory cells only if the launch read passes
→ assess against §14.8's A–F outcome table, with the D/E/F attribution
read against §14.5's pinned drift bands. The K=48 stretch launches only
after both mandatory cells complete without tripping §14.8 outcome D.

### 14.13 Rev B — round-2 verification responses (finding → change map)

Round-2 verification of Rev A returned **NEEDS-REVISION on CHECKs 1(b)
and 2(a)/(c); CHECKs 3/4 PASS**. Disposition: all four fixes accepted
and implemented; one registered refinement to the verifier's own
proposed mapping formula, declared per that instruction's explicit
allowance (row 1 below). This round is expected to be the convergence
round — a final verification pass on this Rev B precedes build
(§14.12); the not-build-ready flag stands until it passes.

| # | Finding (condensed) | Change made | Where |
|---|---|---|---|
| 1(b) | THE LOOPHOLE — the launch read cited "the F2 simulator," which was never defined in the doc and existed only as the attacker's session script; an undefined instrument cannot gate a spend decision | (a) The attacker's preserved script (`/home/nvidia/geo3_ns_check.py`; scratchpad copy) refactored minimally into the repo as **`matrix-thinking/deltanet_rd/geo3_simulator.py`** — importable functions (`newton_schulz`, `delta_rule_exact`, `tilt_to_cos`, `simulate_recovery`, `pairwise_drift_stats`, `launch_read`), original probe sections preserved under `__main__` for provenance, CPU-smoke-tested at write-in (exact regime → rec 1.0 at h=1–4; h=1 drift-immune; drift degrades deep hops); (b) the drift→simulator mapping **written down and registered** in §14.6: drift = value→key cross-alignment factor, value-rep tilted to the MEASURED mean pooled pairwise drift cosine `c` (p10 worst-case run alongside), keys at `gram_resid = resid_tol = 1e-2`, gate = predicted K=16 h=4 `rec@0.9 ≥ 0.8` under the MEAN mapping; statistic (mean + p10) and sampling spec (≥8 entities per K, random without replacement from the train pool, ≥32 context resamples per entity, pairwise cosines pooled within entity then across entities) pinned. Registered refinement, declared: the simulation's emergent compounding (~`c^(h−1)` — hop 1 scores against the value directly — plus captured interference) supersedes the instruction's `c^h` shorthand as the registered form; the load-bearing object is the simulation with the tilt construction, not either closed form | §14.6 (gating block rewritten), `geo3_simulator.py` (new file) |
| 1(c) | The D/E/F split hinged on unquantified "high"/"low" drift | Numeric bands pinned: **HIGH = mean pooled pairwise cos < 0.95** at the trained checkpoint for that K (at-or-worse-than the attacker's measured K=16 probe level); **LOW = ≥ 0.98**; **0.95–0.98 = indeterminate, no mechanism claim** (behavioral numbers still reported; attribution line reads "indeterminate drift band"); §14.8's D/E/F rows repointed at the bands | §14.5 (pinned-split block), §14.8 (D/E/F) |
| 2(c) | The substitute admission stack had no task-performance criterion — a run that never learned the task could pass items 1–3 | Item 4 added: final train loss ≥50% improvement over the untrained-model (step-0) loss AND h=1 in-distribution `rec@0.9 ≥ 0.5` at the final checkpoint — explicitly an admissibility floor (learned baseline's own h=1 is 0.78 at K=32), never a success bar | §14.10 (item 4) |
| 2(a) | Rev A implied the substitute stack restores cross-arm gate symmetry; value-salvage is low-selectivity per archived data (post-fix baselines pass near-universally — the old gate's selectivity lived in the now-excluded alignment criterion), so comparability is an unverified empirical claim | Stated plainly in §14.10: substitute-stack comparability is **UNVERIFIED until Wave −1/Wave 1 produce pass-rate data**; items 2–4 add selectivity of a different kind, not demonstrated-equal selectivity; every cross-arm headline involving a geo3 arm must name the differing admission stacks, what each filters, and the realized pass rate under each — geo3 "≥2 clean seeds" and learned-arm "≥2 clean seeds" are not interchangeable evidentiary events | §14.10 (comparability paragraph) |

---

## 15. Wave F results (soft arms): bars NOT met — proportionality law + attractor robustness

**18/18 cells complete** (`wF_geo1_l01`, `wF_geo1_l10`, `wF_geo2_zca` ×
K∈{16,32} × seeds 0–2, 20,000 steps each, zero skipped steps, zero
timeouts). This is the §5.5 launch-gate's full track (M-attr and M-arm
agreed on geometry as the ≥50%-dominant mechanism, `EXPERIMENT_LOG.md`
"Wave 1 ATTRIBUTION VERDICT"), run to completion with **no further
tuning beyond the two pre-registered `λ_orth` values and the one ZCA
construction** — the anti-Goodhart cap §5.5 sets, honored. Verdict,
stated first: **every pre-registered bar was missed; the h=1 guard held
everywhere.** The result is a real, measurable, directionally-consistent
effect that is nowhere close to the frontier-shift the demonstration
needed.

### 15.1 Full table — mean [range] over 3 seeds, per arm × K

**Write-time key/value geometry (trained, final checkpoint):**

| Arm | K | key-Gram dev | value-Gram dev | learned-baseline key-Gram dev (§1, arm iii) |
|---|---|---|---|---|
| F-geo-1, λ=0.1 (`wF_geo1_l01`) | 16 | 1.2337 [1.2329–1.2344] | 1.2391 [1.2357–1.2455] | 1.26–1.31 |
| F-geo-1, λ=0.1 (`wF_geo1_l01`) | 32 | 2.5083 [2.5071–2.5099] | 2.5126 [2.5111–2.5147] | 2.71–2.77 |
| F-geo-1, λ=1.0 (`wF_geo1_l10`) | 16 | 1.2342 [1.2281–1.2395] | 1.2540 [1.2437–1.2691] | 1.26–1.31 |
| F-geo-1, λ=1.0 (`wF_geo1_l10`) | 32 | 2.5163 [2.5074–2.5333] | 2.5593 [2.5414–2.5950] | 2.71–2.77 |
| F-geo-2, ZCA (`wF_geo2_zca`) | 16 | 1.2431 [1.2411–1.2450] | 1.2383 [1.2371–1.2402] | 1.26–1.31 |
| F-geo-2, ZCA (`wF_geo2_zca`) | 32 | 2.5375 [2.5197–2.5707] | 2.5814 [2.5141–2.7141] | 2.71–2.77 |

**In-distribution recovery, `rec@0.9`, M2 (h=1,2,3):**

| Arm | K | h=1 | h=2 | h=3 | learned baseline h1/h2/h3 |
|---|---|---|---|---|---|
| λ=0.1 | 16 | 0.9960 [0.9954–0.9967] | 0.9100 [0.9081–0.9111] | 0.7233 [0.7158–0.7297] | ≈1.00 / 0.87–0.90 / 0.64–0.70 |
| λ=0.1 | 32 | 0.8131 [0.8111–0.8160] | 0.3639 [0.3622–0.3668] | 0.1042 [0.1006–0.1082] | ≈0.79 / 0.26–0.29 / 0.05–0.06 |
| λ=1.0 | 16 | 0.9875 [0.9860–0.9899] | 0.8930 [0.8861–0.8975] | 0.7051 [0.6948–0.7212] | ≈1.00 / 0.87–0.90 / 0.64–0.70 |
| λ=1.0 | 32 | 0.7970 [0.7866–0.8037] | 0.3498 [0.3293–0.3622] | 0.0948 [0.0860–0.1016] | ≈0.79 / 0.26–0.29 / 0.05–0.06 |
| ZCA | 16 | 0.9978 [0.9976–0.9979] | 0.9120 [0.9073–0.9207] | 0.7133 [0.7098–0.7169] | ≈1.00 / 0.87–0.90 / 0.64–0.70 |
| ZCA | 32 | 0.8063 [0.7986–0.8159] | 0.3504 [0.3314–0.3633] | 0.0948 [0.0878–0.1000] | ≈0.79 / 0.26–0.29 / 0.05–0.06 |

**Held-out recovery, `rec@0.9`, M3 (h=4,5,6,7,21):**

| Arm | K | h=4 | h=5 | h=6 | h=7 | h=21 |
|---|---|---|---|---|---|---|
| λ=0.1 | 16 | 0.4995 [0.4883–0.5155] | 0.3196 | 0.1920 | 0.1108 | 0.0000 |
| λ=0.1 | 32 | 0.0232 [0.0204–0.0258] | 0.0038 | 0.0010 | 0.0000 | 0.0000 |
| λ=1.0 | 16 | 0.4909 [0.4652–0.5118] | 0.3238 | 0.1863 | 0.1069 | 0.0000 |
| λ=1.0 | 32 | 0.0169 [0.0141–0.0201] | 0.0027 | 0.0006 | 0.0000 | 0.0000 |
| ZCA | 16 | 0.4874 [0.4811–0.4979] | 0.3047 | 0.1784 | 0.0948 | 0.0000 |
| ZCA | 32 | 0.0191 [0.0173–0.0209] | 0.0028 | 0.0003 | 0.0001 | 0.0000 |

**Premise/alignment gate (finding-5 stack, §7):** all 18 cells pass
key- and value-side salvage tier (`σ_K/σ_1 ≥ 0.1`) at every checkpoint —
premise validity holds universally. Per-item alignment (`cos ≥ 0.9`,
all items) holds for **16/18 seeds**; it fails for exactly 2 —
`wF_geo1_l10_K16_s1` and `wF_geo1_l10_K32_s1` (same seed index, same
`λ_orth=1.0` arm) — where `alignment_cos_mean` stays ≈0.991–0.992 but
`alignment_cos_min` collapses to 0.047–0.082 for a small item minority
(`alignment_valid_frac` ≈0.99). Net effect: the λ=1.0 arm retains
**2/3 alignment-clean seeds at each K**, meeting finding-5's ≥2-clean
floor but not the full 3/3 the other two arms achieve — noted here so
any downstream mean over "all 3 seeds" for that one arm is read as
including one alignment-flagged seed, not silently dropped.

### 15.2 Verdict vs. each pre-registered bar (§5.5)

| Tier | Cell | Baseline | Bar | Measured (Wave F, all 3 arms pooled) | Verdict |
|---|---|---|---|---|---|
| Headline demo | K=32, h=4 | 0.009 | ≥0.5 | 0.020 mean, 0.014–0.026 range | **MISSED — ~20–25× below bar** |
| Minimum publishable | K=16, h=4 | 0.419–0.465 | ≥0.8 | 0.493 mean, 0.465–0.516 range | **MISSED — bar ~1.6–1.7× above achieved** |
| Guard, K=16 h=1 | same arm's own h=1 | ≈1.00 (arm iii) | within −0.02 | 0.988–0.998 | **SATISFIED** |
| Guard, K=32 h=1 | same arm's own h=1 | ≈0.79 (arm iii) | within −0.02 | 0.797–0.813 | **SATISFIED** |

No arm, no seed, at either K, comes within an order of magnitude of the
headline bar. The minimum-publishable bar is missed by a smaller but
still decisive margin. Every arm satisfies the no-sacrifice guard —
whatever these interventions do, they do not trade in-distribution
binding for held-out reach; they simply do not reach far enough.

### 15.3 The proportionality reading — real, directionally consistent, far from the bars

The Gram-deviation cleanup at K=32 is **3–8%** relative to the arm-iii
learned baseline (baseline 2.71–2.77 → Wave F 2.51–2.57 pooled across
all three arms and 9 seeds). This is not noise: every one of the 9 K=32
seeds lands below the baseline band, and recovery moves with it in the
same direction at every hop measured:

- **h=2:** baseline 0.26–0.29 → Wave F 0.33–0.36, a gain of **+0.07 to
  +0.10**.
- **h=3:** baseline 0.05–0.06 → Wave F 0.09–0.10, a gain of **+0.04 to
  +0.05**.
- **h=4 (the headline cell):** baseline 0.009 → Wave F 0.014–0.026, a
  gain of only **+0.005 to +0.017** in absolute terms.

The gain **shrinks at every successive hop** even as the underlying
Gram cleanup is a single fixed number per run. This is exactly the
shape §0's ε^h compounding law predicts (`DELTANET_REALDATA_DESIGN.md`
§18, inherited verbatim into this design at §2 mechanism (g) and §14.0):
per-binding write-time inexactness ε is set once, and composition
inherits `~ε^h` regardless of hop distribution — a modest, real
reduction in ε produces a compounding-in-reverse shrinkage of its own
benefit as h grows, because `ε^h` is far more sensitive to `h` than to
a few-percent change in `ε` at fixed h. A **3–8% cleaner** write-time
Gram matrix was never going to move a metric measured 4 compositions
deep from 0.009 to 0.5 — the arithmetic of the compounding law itself
explains why the proportional, real effect at h=2/h=3 evaporates by
h=4, independent of any story about *why* SGD only cleans up 3–8%.

### 15.4 The saturation observation — one attractor, three approach angles

`λ_orth=0.1` and `λ_orth=1.0` land at statistically indistinguishable
K=32 key-Gram deviation (2.5083 vs. 2.5163 mean — inside each other's
seed range) and statistically indistinguishable recovery at every hop.
A 10× change in penalty weight bought essentially nothing — **the
penalty saturates**. ZCA whitening — an architecturally different,
population-level, structural intervention rather than a per-episode
soft penalty — lands in the *same* band (K=32 key-Gram 2.5375, h=4
0.0191) as both penalty weights. Three mechanistically distinct
pressures (weak soft push, strong soft push, population-level
structural decorrelation) converge to the same narrow basin
(2.51–2.57 Gram deviation, 0.33–0.36 at h=2, 0.09–0.10 at h=3,
0.014–0.026 at h=4). This is the signature of a single attractor, not
three different partial successes — the write-geometry basin SGD
settles into under this objective is robust to at least this much
variation in how hard and by what mechanism it is pushed.

### 15.5 Honest conclusion

Soft, gradient-level or population-level geometry pressure cannot pull
SGD's write-time key geometry out of its non-orthonormal attractor on
real tokenized data. The effect is real (§15.3), directionally
consistent across every hop and every arm, and structurally robust
(§15.4) — but an order of magnitude too small to matter at the depth
the headline bar is set. This is not a story about a broken
intervention; F-geo-1 and F-geo-2 did exactly what their construction
promises (a soft loss-side nudge, a population-level decorrelation
layer), and both did it correctly (premise validity holds universally,
§15.1). It is a story about what soft pressure *cannot* do to an SGD
optimization that never requires exact orthonormality to reduce loss.

**Arm i-strong's existence proof stands, unweakened, as the other half
of this picture** (§4.4, `EXPERIMENT_LOG.md` "Wave 1 ATTRIBUTION
VERDICT"): a non-trainable, hand-built, per-identity `k_eff` pin
bypassing `W_k`/conv achieves key-Gram deviation **0.000** and
`rec@0.9` **1.00/1.00/1.00** at h=1/2/3, K=32 — exact composition, at
the identical operating point Wave F's soft arms plateau at 3–8%
cleanup. The architecture and the delta rule's own algebra can
represent and use the perfect solution. Ordinary training pressure,
soft-nudged or not, does not find it. The gap between "exists and is
reachable in principle" and "SGD finds it under any loss-side or
population-level pressure tried so far" is now measured, not assumed.

### 15.6 F-geo-3 handoff note

§14 (per-episode differentiable key orthogonalization *at* the `k_eff`
site — the structural, trainable version of the i-strong pin) is the
constructive response to §15.5's gap, commissioned after F-geo-1/F-geo-2
both landed here. As of this entry, §14 has completed its **own**
independent attack round (Rev A, verdict NEEDS-REVISION, no FATALs —
the Newton-Schulz update math, the `sqrt(K)` pre-scaling basin proof,
the β-mask inertness argument, and the `bind()→readout()` threading all
verified clean under adversarial GPU probes; every relayed finding
addressed, load-bearing new result F1: measured cross-episode
orthogonalized-key drift as a third failure mechanism, folded into a
pre-registered Wave −1 gating diagnostic with its own launch-read rule,
§14.6/§14.11). **§14 is still explicitly NOT build-ready** — per the
base design's own two-round discipline, a fresh verification round on
Rev A is required before any Wave −1 code is written (§14.12). Nothing
in §1–§13 or this section is modified by §14; it remains a documented
addendum, not an authorization to spend.

### 15.7 Claim-tier language (§7) — this is the pre-registered negative branch, not a failure to iterate

Per §5.5's own text, written before any Wave F data existed: **"a
dominant mechanism whose matching fix does NOT move the frontier is
itself a publishable finding... and routes back through design → attack
as a new revision, not through fix-iteration."** That is exactly what
happened: both pre-registered F-geo-1 `λ` values and the one
pre-registered F-geo-2 construction ran to completion with no further
tuning (the anti-Goodhart cap honored — this was the single registered
soft-arm shot, not the first of an open-ended search), both missed
every bar, and the response is §14's new design-and-attack cycle, not a
fourth soft-pressure rerun inside this budget.

Tier discipline, per §7: this section earns the **demonstration-claim
tier** — "the fix moves the frontier" — and that claim is **false** for
all three arms, stated plainly. The weaker, true claim earned instead:
a genuine, causal (premise-conditional) but small write-geometry
cleanup, with recovery moving proportionally in the direction the
mechanism predicts. §5.5's F-geo-1 instrument caveat applies exactly as
pre-registered: in the `λ=0.1`/`λ=1.0` arms, the key-Gram diagnostic is
directly optimized by `L_orth` and is excluded from premise-evidence
status — the verdict above rests on the *behavioral* `rec@0.9` bars,
never on the Gram number itself, for those two arms. F-geo-2 (ZCA) has
no such carve-out — its key-Gram number is a genuine, independent
measurement — and it lands in the same basin as the two compromised
numbers anyway (§15.4), which is itself evidence the F-geo-1 caveat
is not hiding a materially different outcome.

**Archive:**
`experiment-runs/2026-07-03_deltanet_rd_waves/exactness/waveF/` — the
18 Wave F run JSONs, the 5 arm-(iii-β) fresh measured-β reruns
(`w1_iiibeta_K{16,32}_s{seed}.json`), `W0_REPORT.json` +
`W0_REPORT_iiibeta.json`, and `CALIBRATION_SUMMARY.json` (26 files, 37 MB
total, verified against the box byte-for-byte by `du` cross-check).
Extraction script: `extract_wavef.py` (dumps the raw per-run fields) +
`summarize_wavef.py` (mean/range aggregation), archived alongside the
data in the same `waveF/` directory — run CPU-only on
`youthful-indigo-turkey` against the box's live results dir, no GPU
spend; the 18-cell table in §15.1 is their direct output.

---

## 16. F-geo-3 results (Wave geo3 + escalation): K=16 bar HIT 3/3; K=32 45x at full tier, bar narrowly missed, residual attributed to outcome F (cross-episode drift)

**9/9 cells complete** — the original Wave 1 batch (`geo3_n_iter=12`:
K=16 ×3 seeds, K=32 ×3 seeds) plus the audit-verified escalation
(`geo3_n_iter=20`: K=32 ×3 seeds, same seed indices, same everything
else), 20,000/20,000 steps each, zero skipped steps, zero timeouts,
`d_model=256`, `d_state=64`, `n_params=12,899,841`. Verdict, stated
first, per §14.8's pre-registered outcome table: **K=16 hits the
minimum-publishable bar on 3/3 admissible seeds; K=32 improves ~43–56×
over its learned-arm baseline (mean ≈48×, consistent with the ~45×
figure in this section's own title) but falls short of its headline
bar on the mean, with the residual gap attributed to pre-registered
outcome F — stable-not-just-orthogonal geometry — not to a broken fix.**
The escalation's own function was narrower than "hit the bar": it
existed to remove the original wave's admissibility confound (the
Newton-Schulz eigh fallback triggering on a small step fraction) so the
K=32 number could be read as premise-clean evidence at all. It
succeeds at exactly that — 0/3 → 3/3 admissible — and the underlying
behavioral numbers do not move, which is itself the section's second
finding (§16.5).

### 16.1 Wave −1 gate record (pre-registered, before any Wave 1 spend)

The §14.6 gating diagnostic (8 entities, 32 context resamples each,
7,936 pooled pairs per cell, `probe_steps=5,000`) measured cross-episode
orthogonalized-key drift before any Wave 1 seed ran, and the registered
`geo3_simulator.launch_read` (§14.13 finding 1(b)) converted that drift
into a pre-spend prediction at the gate cell (K=16, h=4, bar ≥0.8):

| K | drift at init (mean / p10) | drift after 5K-step probe (mean / p10) | drift band (§14.5: HIGH <0.95, LOW ≥0.98) | predicted `rec@0.9` h=4, mean mapping | predicted `rec@0.9` h=4, p10 mapping |
|---|---|---|---|---|---|
| 16 (gate cell) | 0.4676 / 0.3084 | **0.9416** / 0.9186 | HIGH | **1.0000** | 0.9551 |
| 32 (informational only) | 0.4350 / 0.2738 | **0.9037** / 0.8713 | HIGH | 0.7734 | 0.2227 |

`gate_bar = 0.8` at K=16 h=4; `predicted_gate_value = 1.0000 ≥ 0.8` →
**`launch = true`**. Training shrinks drift substantially at both K
(0.47→0.94 at K=16, 0.44→0.90 at K=32) relative to the untrained-key
baseline — the attack's cross-episode-drift risk resolved favorably
enough to authorize spend — but both trained-checkpoint drift levels
still land in the pre-registered **HIGH** band (<0.95), not the LOW
band (≥0.98) that would put this wave in outcome D/E territory. That
placement is exactly what materializes in §16.6.

### 16.2 Full per-cell table — `geo3_n_iter=12` (original wave, 6 cells)

**Write-time key/value geometry (trained, final checkpoint):**

| K | key-Gram dev (all 3 seeds) | value-Gram dev, mean [range] |
|---|---|---|
| 16 | ~3×10⁻⁷ (2.97–3.28×10⁻⁷) — non-evidence, forced ≈0 by construction (§14.9 item 5, §14.10) | 2.1948 [1.6872–2.6686] |
| 32 | ~7–8×10⁻⁷ (6.70–8.20×10⁻⁷) — non-evidence, same reason | 5.9274 [4.5512–6.7564] |

**In-distribution recovery, `rec@0.9`, M2 (h=1,2,3), mean [range]:**

| K | h=1 | h=2 | h=3 |
|---|---|---|---|
| 16 | 1.0000 [1.0000–1.0000] | 1.0000 [1.0000–1.0000] | 0.9990 [0.9971–1.0000] |
| 32 | 1.0000 [1.0000–1.0000] | 0.9999 [0.9998–1.0000] | 0.9014 [0.8701–0.9479] |

**Held-out recovery, `rec@0.9`, M3 (h=4,5,6,7,21), mean [range]:**

| K | h=4 | h=5 | h=6 | h=7 | h=21 |
|---|---|---|---|---|---|
| 16 | 0.9767 [0.9525–0.9969] | 0.8967 [0.8402–0.9534] | 0.7626 [0.7006–0.8485] | 0.6007 [0.5497–0.6729] | 0.0074 [0.0048–0.0089] |
| 32 | 0.4376 [0.3890–0.5040] | 0.1485 [0.1477–0.1493] | 0.0463 [0.0374–0.0538] | 0.0177 [0.0093–0.0261] | 0.0000 [0.0000–0.0000] |

**Substitute admission stack, per seed (§14.10 items 1–4):**

| K | seed | (1) value-salvage pass, ratio | (2) NS no-fallback | fallback steps / 20,000 (%) | checkpoint fallback seen | (3) finite loss | (4) task floor: %-improve / h1@0.9 | **admissible** |
|---|---|---|---|---|---|---|---|---|
| 16 | 0 | pass, 0.4547 | pass | 0 (0.00%) | no | pass | 99.52% / 1.0000 | **yes** |
| 16 | 1 | pass, 0.3229 | pass | 0 (0.00%) | no | pass | 99.51% / 1.0000 | **yes** |
| 16 | 2 | pass, 0.3821 | pass | 0 (0.00%) | no | pass | 99.47% / 1.0000 | **yes** |
| 32 | 0 | pass, 0.2056 | **fail** | 56 (0.28%) | no | pass | 99.00% / 1.0000 | **no** |
| 32 | 1 | pass, 0.1317 | **fail** | 11 (0.055%) | no | pass | 98.92% / 1.0000 | **no** |
| 32 | 2 | pass, 0.1132 | **fail** | 374 (1.87%) | yes | pass | 98.79% / 1.0000 | **no** |

K=16: **3/3 admissible**. K=32 (`n_iter=12`): **0/3 admissible** — every
seed fails item 2 alone (items 1/3/4 pass universally); this is the
admissibility failure the escalation targets.

### 16.3 Full per-cell table — `geo3_n_iter=20` (escalation, K=32 ×3 seeds)

**Write-time key/value geometry (trained, final checkpoint):**

| K | key-Gram dev (all 3 seeds) | value-Gram dev, mean [range] |
|---|---|---|
| 32 | ~5×10⁻⁷ (5.14–5.45×10⁻⁷) — non-evidence, same reason as §16.2 | 5.9271 [4.5514–6.7556] |

**In-distribution recovery, `rec@0.9`, M2 (h=1,2,3), mean [range]:**

| K | h=1 | h=2 | h=3 |
|---|---|---|---|
| 32 | 1.0000 [1.0000–1.0000] | 0.9999 [0.9998–1.0000] | 0.9021 [0.8721–0.9477] |

**Held-out recovery, `rec@0.9`, M3 (h=4,5,6,7,21), mean [range]:**

| K | h=4 | h=5 | h=6 | h=7 | h=21 |
|---|---|---|---|---|---|
| 32 | 0.4368 [0.3903–0.5045] | 0.1480 [0.1450–0.1500] | 0.0462 [0.0373–0.0539] | 0.0171 [0.0092–0.0245] | 0.0000 [0.0000–0.0000] |

**Substitute admission stack, per seed:**

| K | seed | (1) value-salvage pass, ratio | (2) NS no-fallback | fallback steps / 20,000 (%) | checkpoint fallback seen | (3) finite loss | (4) task floor: %-improve / h1@0.9 | **admissible** |
|---|---|---|---|---|---|---|---|---|
| 32 | 0 | pass, 0.2056 | **pass** | 0 (0.00%) | no | pass | 99.00% / 1.0000 | **yes** |
| 32 | 1 | pass, 0.1320 | **pass** | 0 (0.00%) | no | pass | 98.92% / 1.0000 | **yes** |
| 32 | 2 | pass, 0.1133 | **pass** | 0 (0.00%) | no | pass | 98.79% / 1.0000 | **yes** |

K=32 (`n_iter=20`): **3/3 admissible** — item 2 now clears at every
seed with zero fallback steps triggered, items 1/3/4 unchanged (as
expected — nothing about the admission floor changed, only the solver
iteration budget did).

### 16.4 Verdict vs. each pre-registered bar (§5.5)

| Tier | Cell | Baseline (learned arm) | Bar | Measured (geo3, admissible seeds only) | Verdict |
|---|---|---|---|---|---|
| Minimum publishable | K=16, h=4 | 0.419–0.465 | `rec@0.9` **≥ 0.8** | 0.9767 mean, 0.9525–0.9969 range (3/3 admissible) | **HIT — 3/3, ~2.1× the bar on the low seed alone** |
| Headline demo | K=32, h=4 | 0.009 | `rec@0.9` **≥ 0.5** | 0.4368 mean, 0.3903–0.5045 range (3/3 admissible, `n_iter=20`) | **NOT MET on the mean (0.4368 < 0.5, ~0.06 short) — narrow miss; the s0 seed individually clears the bar at 0.5045** |
| Guard, K=16 h=1 | same arm's own h=1 | ≈1.00 (arm iii) | within −0.02 | 1.0000 | **SATISFIED** |
| Guard, K=32 h=1 | same arm's own h=1 | ≈0.79 (arm iii) | within −0.02 | 1.0000 | **SATISFIED — exceeds baseline by +0.21, not merely within tolerance** |

K=32's `n_iter=12` cell (0/3 admissible, §16.2) is excluded from this
table per §14.10's own admission-gate discipline — a non-admissible
seed does not enter a headline comparison regardless of what its raw
number reads. Its behavioral numbers are reported in §16.2 for
completeness and turn out to be nearly identical to the admissible
`n_iter=20` numbers (§16.5) — the exclusion changes admissibility, not
the measured effect.

### 16.5 The fallback-irrelevance observation

`n_iter=12` (0/3 admissible, fallback-contaminated) and `n_iter=20`
(3/3 admissible, zero fallback) land within seed-level noise of each
other at every hop, for every seed:

| seed | fallback steps (n12) | h=4 (n12 → n20) | h=7 (n12 → n20) | value-Gram dev (n12 → n20) | task-floor %-improve (n12 → n20) |
|---|---|---|---|---|---|
| 0 | 56 | 0.5040 → 0.5045 (Δ+0.0005) | 0.0093 → 0.0092 (Δ−0.0001) | 4.5512 → 4.5514 | 99.00% → 99.00% |
| 1 | 11 | 0.4199 → 0.4157 (Δ−0.0042) | 0.0179 → 0.0177 (Δ−0.0002) | 6.4746 → 6.4742 | 98.92% → 98.92% |
| 2 | 374 | 0.3890 → 0.3903 (Δ+0.0013) | 0.0261 → 0.0245 (Δ−0.0016) | 6.7564 → 6.7556 | 98.79% → 98.79% |

Every delta is at or below seed-level noise (largest single move is
Δ0.0042 on a metric with a 0.39–0.50 range), even for seed 2, whose
`n_iter=12` run leaned on the fallback solver for **1.87% of steps**
(374/20,000) and had a checkpoint-level fallback flag raised — the
single most fallback-contaminated run in the wave. **The 56 + 11 + 374
fallback steps pooled across the original wave never measurably
degraded training** — the admission failure at `n_iter=12` was a
premise-cleanliness problem (a run that leaned on the non-differentiable
solver path is not clean evidence about the *differentiable* mechanism,
per §14.10 item 2's own rationale), not a sign the mechanism was
partially broken. The escalation's entire value is closing that
premise gap, not changing the number.

### 16.6 Outcome-F attribution: stable-not-just-orthogonal geometry is the bottleneck

§14.8's outcome F requires three co-occurring signatures, all present:

1. **`resid ≈ 0`** — key-Gram deviation is ~3–8×10⁻⁷ at both K, every
   seed (§16.2/§16.3) — orthogonalization is complete, by construction.
2. **HIGH cross-episode drift** (mean pooled pairwise cos < 0.95 at the
   trained checkpoint, §14.5's pinned band) — measured **0.9037** at
   K=32 and **0.9416** at K=16 (§16.1), both below the 0.95 HIGH
   threshold, both measured *before* the wave ran, not fit after the
   fact.
3. **Graded h-decay, steeper at higher K** — K=16 falls 0.9767→0.6007
   from h=4 to h=7 (a 38.5% relative drop); K=32 falls 0.4376→0.0177
   over the same span (a 96.0% relative drop) — the decay is
   categorically steeper at the higher K, exactly the ordering §14.8's
   outcome F predicts.

All three hold simultaneously, so the verdict is clean per §14.8's own
framing: **within-episode orthonormality is real but insufficient** —
joint per-episode orthogonalization supplies orthogonality, not the
*stable, entity-fixed* key identity across episodes that exact
composition also requires, and `W_v` cannot chase a moving key target.
The K=32 residual gap against its ≥0.5 bar is attributed to this named
mechanism, not read as evidence the fix is broken. One subtlety worth
flagging plainly: K=16's drift (0.9416) is *also* inside the HIGH band,
yet K=16 still clears its own (lower, K-specific) bar with a wide
margin — outcome F's mechanism is present at both K, but it only binds
K=32 against its harder ≥0.5 bar. This is consistent with, not a
counterexample to, the attribution: the same drift level produces a
much smaller relative composition penalty at h=4 when K is smaller
(fewer competing episode-conditional key directions to confuse).

### 16.7 Simulator calibration note

The registered `launch_read` mean-mapping prediction (§16.1) was
accurate at the gate cell and imprecise off it:

| K | predicted `rec@0.9` h=4, mean mapping | predicted `rec@0.9` h=4, p10 mapping | measured `rec@0.9` h=4 (admissible seeds) | prediction error (mean mapping) |
|---|---|---|---|---|
| 16 | 1.0000 | 0.9551 | 0.9767 | **−0.023** (essentially exact) |
| 32 | 0.7734 | 0.2227 | 0.4368 | **+0.337** (overestimate, ~1.8× the measured value) |

K=16's prediction is accurate to within the launch-read's own gate
tolerance. K=32's mean-mapping prediction overshoots by a wide margin —
but the measured value (0.4368) falls **inside** the registered
`[p10, mean]` bracket the launch-read itself logged before the wave
(0.2227–0.7734), so the tool was not wrong in kind, only imprecise in
magnitude at the harder cell. Attribution: the registered drift→
simulator mapping (§14.13 finding 1(b)) tilts *only* the value
representation by a single scalar (the mean pooled pairwise drift
cosine `c`) against otherwise-idealized keys (`gram_resid` pinned at
`resid_tol=0.01`); it carries no separate term for the value-Gram
deviation the real trained model actually exhibits, which is itself
2.7× larger at K=32 than at K=16 (5.9274 vs. 2.1948 mean, §16.2/§16.3)
— a real, K-dependent degradation channel the single-parameter tilt
construction does not model. Two compounding effects (cross-episode key
drift AND value-geometry deviation) are live at K=32; the registered
simulator carries only the first, which is consistent with — not a
retraction of — the outcome-F attribution: the drift mechanism is
confirmed as real and dominant enough to correctly separate
K16-clears from K32-misses pre-spend, but a single-scalar tilt is not a
complete quantitative account of the K=32 residual specifically.

**CORRECTION (2026-07-04, GPU re-measurement — supersedes the K=32 row and
its attribution paragraph above).** The K=32 "0.7734 predicted" figure was
computed with the WRONG drift input: `geo3_drift_diagnostic.py::main()`
extracts only K=16's measured drift (`per_k[16]["after_probe"].mean` =
0.9416) and `geo3_simulator.launch_read()` applies that single scalar `c`
to BOTH K's in its `for K in (16, 32)` loop — K=32's own separately-measured
drift (0.9037, logged in the same JSON) was never wired into K=32's
prediction. Verified by direct re-run on the box (GPU 6, 3 seeds ×
{GPU, CPU} × both `c` values; archive:
`experiment-runs/2026-07-04_geo3_simulator_recheck/`): with c=0.9416 the
simulator reproduces the recorded 0.7734 to the last digit (seed 0); with
K=32's own c=0.9037 it predicts **0.06–0.09** — an UNDERESTIMATE of the
measured 0.4368, the opposite direction from the paragraph above. GPU vs
CPU agree everywhere (spread ≈ seed noise); no platform effect exists.
Consequences: (1) the "+0.337 overestimate" row is an input-mismatch
artifact, and the measured value falling inside the logged [p10, mean]
bracket was partly luck, since that bracket was also computed from K=16's
drift; (2) the correct statement is that the single-scalar mean-drift
mapping, fed K=32's own drift, is strongly CONSERVATIVE at K=32 — the
trained model recovers substantially more than the idealized-drift model
predicts; the value-Gram-deviation channel invoked above is therefore not
needed to explain an overshoot (there is none), though it remains a real,
unmodeled channel; (3) outcome F's attribution (drift is the residual
bottleneck) is UNAFFECTED — it rests on the measured drift statistic and
the measured h4 miss, not on the simulator; (4) the K16-clears/K32-misses
pre-spend separation claim for the gate stands empirically but must be
restated: it separated the cells with a mis-wired input, so it cannot be
cited as validation of the drift→recovery mapping at K=32. Any future gate
must thread each K's own measured drift (the shared-`c` API is the bug;
fix registered for the key-anchoring wave build). Found by attack round 2
on `KEY_ANCHORING_DESIGN.md` (platform-sensitivity hypothesis) and pinned
to the true cause by the archived GPU recheck.

### 16.8 h=1 no-sacrifice guard and h=21 literal-depth decay

**h=1 guard:** K=16 measures 1.0000 at every seed against an arm-iii
baseline of ≈1.00 — trivially within the −0.02 guard. K=32 measures
1.0000 at every seed (both `n_iter` tiers) against an arm-iii baseline
of ≈0.79 (§15.1) — not merely inside the guard but **+0.21 above
baseline**, echoing i-strong's own h=1 result (§4.4) and confirming the
fix does not trade single-hop binding away to reach for deeper
composition.

**h=21:** K=16 lands at 0.0074 mean [0.0048–0.0089]; K=32 is flat at
0.0000 across all 6 runs (both tiers). Wave F's own h=21 column was
0.0000 at every arm, both K (§15.1) — so K=16's small nonzero reading
is a marginal move off that floor, K=32's is unchanged. Both remain
effectively at the noise floor: **the h=21 literal-depth collapse is
unchanged by this fix**, consistent with the original wave verdict
(`EXPERIMENT_LOG.md`, "F-geo-3 WAVE VERDICT") — per-episode key
orthogonalization repairs cross-item write interference (the h=4–7
frontier), it does not touch the separate iteration-compounding failure
mode that dominates at h=21.

### 16.9 Claim-tier language (§7/§14.10) — substitute admission stack footnote

Per §7's Wave F two-tier separation, inherited verbatim by F-geo-3
(§14.10): **(1) "the fix moves the frontier"** — TRUE at K=16 (bar HIT,
3/3 admissible); **partially true** at K=32 — a large, admissible,
premise-clean shift (43–56× over the 0.009 baseline) that still misses
the pre-registered ≥0.5 bar on the mean. **(2) "therefore the
attribution was right"** — a supported inference, not proven by the
demo alone: here it is corroborated independently, because the §14.6
drift diagnostic that predicts the K=32 shortfall was measured *before*
the wave ran (§16.1), and the measured checkpoint drift lands squarely
in outcome F's pre-registered three-part signature (§16.6) rather than
being a story assembled after seeing the miss.

**Substitute admission stack, comparability restated per §14.10's own
requirement** (never present a geo3 admissible-seed count as
interchangeable with a learned-arm finding-5-clean-seed count without
stating both gates' realized pass rates): geo3 K=16 realizes **3/3
(100%)** admissible seeds under the §14.10 substitute stack; geo3 K=32
realizes **0/3 (0%)** at `n_iter=12` and **3/3 (100%)** at
`n_iter=20`. The standard finding-5 gate the learned (arm iii) K=32
baseline is held to realizes only **~1–2/7 (~14–29%)** clean seeds
per the archived data (§14.10). These are different evidentiary
events measured by different criteria (value-salvage tier + NS
convergence + finite loss + task floor, vs. alignment-clean + salvage
tier) — comparability remains **UNVERIFIED**, exactly as §14.10
pre-registered, and this section does not treat a 100% geo3 pass rate
as a stronger or weaker claim than the learned arm's ~14–29%, only as
a differently-gated one.

**Non-evidence instruments, restated per §14.9 item 5:** the key-Gram
deviation (~3–8×10⁻⁷ at every seed, every K, both tiers) and the
per-item alignment instrument are architecturally forced to ≈0 and
≈1.0 respectively under `geo3_active` — logged in §16.2/§16.3 for
completeness, carrying **zero** evidentiary weight about whether the
fix works. Every verdict in §16.4–§16.8 rests exclusively on the
behavioral `rec@0.9` bars and the substitute admission stack, never on
these two instruments.

### 16.10 Archive

`experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wavegeo3/` —
10 files: the original 6 `geo3_n_iter=12` run JSONs
(`wgeo3_rdx_K{16,32}_armgeo3_s{0,1,2}_geo3n12.json`) and
`GEO3_DRIFT_DIAGNOSTIC.json` (archived at the original wave verdict,
2026-07-04), plus the 3 new `geo3_n_iter=20` escalation run JSONs
(`wgeo3_rdx_K32_armgeo3_s{0,1,2}_geo3n20.json`, archived with this
entry) — 15 MB total, mirrored byte-for-byte to
`/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wavegeo3/`.
Total wave compute: ~1.67 GPU-h across all 9 runs (~0.58 GPU-h for the
3-seed escalation alone). Every number in §16.1–§16.8 is read directly
from each run's final checkpoint (`M2_in_distribution`,
`M3_held_out`, `geo3_admission` dicts) and `GEO3_DRIFT_DIAGNOSTIC.json`'s
`launch_read` block — no aggregation script beyond direct field
extraction was required for this wave's small cell count (contrast
Wave F's 18-cell `extract_wavef.py`/`summarize_wavef.py`, §15.7).
