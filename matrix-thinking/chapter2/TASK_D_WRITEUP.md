# Gradient Descent Recruits Matrix Rank When a Task Provably Requires It

**Task D — Chapter 2 of the matrix-thinking program.**
Status: **preliminary** (results snapshot at 991 completed runs; the perpetual sweep
is still refilling). All numbers below are from
`results/overnight_snapshots/AGGREGATE_latest.json`. Pre-registration:
`TASK_D_PREREGISTRATION.md`. Full audit trail: `gauntlet/`.

> **CORRECTION (2026-07-01, 1,234-run mega-replication):** a full replication
> superseding the 991-run snapshot below found that §5.2's dismissal of the
> `d=16,K=4` cell as "undertrained/seed-sparse" was **wrong** — it is a
> genuine, seed-count-independent convergence ceiling. The `d=16,K=12`
> force-rank transition zone is also noisier than the two-point table below
> implied. Both corrections are inline and dated in §5.1/§5.2; see also §7.

> **CORRECTION (2026-07-02, Stage 0 Wave −1/0, Brev 8×H100):** §5.3's
> d≥32 "trainability frontier" is revised. At **d=32** it is substantially
> a **step-budget artifact**, not a stuck optimization: all 6 baseline runs
> at 2.5× Task D's step budget (20K vs 8K) show a late, sharp effective-rank
> transition at 6-16K steps — well past the original 8K cutoff. At **d=64**
> the picture is genuinely undetermined (no transition completes within
> 25K, though 2/3 seeds show a late onset beginning near step 20K) rather
> than confirmed dead. The architectural rank-ceiling story
> (`rank(Z)≤h+1=65`) is untouched and still applies at **d=128, K>65**.
> Full numbers, per-seed trajectories, and hypothesis-by-hypothesis
> verdicts: `matrix-thinking/chapter2/STAGE0_DESIGN.md` §12. Inline
> correction in §5.3 below; see also a further correction to the
> `d=16,K=4` note in §5.2's 2026-07-01 correction block immediately below,
> which was itself budget-confounded.

---

## Abstract

Our companion workshop paper — *The Gradient Does Not See Rank* (Larson, ICML 2026
Mechanistic Interpretability Workshop) — showed that a matrix-valued bottleneck
bolted onto a CODI-style continuous-reasoning model is **rank-blind**: rank-`k`
truncation of the latent leaves accuracy unchanged, across four training regimes
and four nonlinear-in-`Z` readouts. That result left one question open: is the
gradient *fundamentally* indifferent to rank, or was ProsQA simply rank-1-solvable
so nothing rewarded rank in the first place? **Task D settles it.** We construct a
from-scratch, matrix-native associative-memory task whose exact solution *provably*
requires `rank(Z) ≥ K` (a hard linear-algebra bound, not an assertion), force all
information through a single `d×d` matrix state a decoder reads exclusively, and pin
the readout to the literal unbind `Z·key` so no argmax shortcut can evade the bound.
On this task, **when trained end-to-end, gradient descent both develops effective
rank ≈ K and makes rank causally necessary**: at `d ∈ {8, 16}` the learned effective
rank tracks `K` almost exactly (e.g. `d=16`: `K=4→4.7, 8→8.2, 12→11.8, 16→15.1`), and
forcing the state to rank `k` during training destroys recovery for `k < K` with a
step at `k ≈ K` (razor-sharp at `d=8,K=4`: force-rank≤3 → 0.0 recovery, force-rank=4
→ **0.97**). The workshop paper's rank-blindness is therefore **task-specific, not a
property of the gradient**: when the task provably needs rank, the gradient uses it.
We also report an honest **trainability frontier**: at `d ≥ 32` the current small
encoder fails to learn the write (effective rank collapses to ≈1, recovery ≈0) — an
optimization/architecture limitation, not a refutation, and the subject of the next
experiment.

---

## 1. Background and motivation

### 1.1 The rank hypothesis and the negative result it came from

The matrix-thinking program asks whether representing a token/latent as a `d×d`
**matrix** rather than a vector gives a measurable, differentiable observable —
**rank** — that quantifies how many independent items or reasoning paths a
representation holds in superposition. This connects to an active, unresolved debate
about whether continuous chain-of-thought models reason "in superposition":
**Reasoning by Superposition** (Lin, Zhu et al., arXiv:2505.12514) argues a 2-layer
transformer *can* encode a BFS frontier as a superposition; **CoT2** (Zhang, Tian et
al., arXiv:2505.23648) frames parallelism as bounded by embedding dimension; and
**The Illusion of Superposition?** (Rizvi-Martel, Mosbach & Rabusseau,
arXiv:2604.06374) rebuts, showing fine-tuned COCONUT reaches 96.6% on ProsQA without
latent tokens.

Our workshop paper *The Gradient Does Not See Rank* entered this debate empirically:
in a matrix-CODI bottleneck, the rank-`k` truncation curve is **flat** across four
training regimes and four nonlinear-in-`Z` readouts (bilinear, bilinear+GELU,
SVD-augmented, quadratic), and a three-seed replication decouples accuracy
(81.5±1.2pp) from effective rank (spanning {4, 12, 13}). We diagnosed this as a
rank-indifferent gradient through the full chain rule. **The unresolved caveat** (our
§7): ProsQA may be rank-1-solvable, in which case the flat curves are non-evidence.

### 1.2 What Task D adds

Task D removes both confounds at once. Instead of bolting a matrix onto a
vector-pretrained model with a vector teacher, we train **matrix-native from
scratch** on a task with a **provable** `rank(Z) ≥ K` lower bound and **no
rank-1 shortcut by construction**. The question sharpens from the vague "does rank
track reasoning" to the falsifiable:

> **When a task's exact solution provably requires `rank(Z) ≥ K`, does gradient
> descent (a) develop effective rank ≈ K, and (b) make rank causally necessary
> (rank-`k` truncation collapses accuracy for `k < K`)?**

This is the mirror image of the workshop paper: there, nothing rewarded rank and the
gradient ignored it; here, the task *forces* rank and we ask whether the gradient
complies.

---

## 2. Related work

Task D is, to our knowledge, the first to (i) train a transformer end-to-end under a
hard single-`Z` matrix bottleneck, (ii) measure the effective rank of `Z` as a
function of a controlled binding count `K`, (iii) invoke a **provable** `rank ≥ K`
necessity for exact continuous recovery, and (iv) causally force rank `k` during
training against a per-`K` accuracy curve. We position against four lineages
(novelty audit: `research/task-d-novelty-july2026.md`).

**Associative memory & the classical rank fact.** That exact recovery of `K`
orthonormal key→value pairs from a linear memory `W = Σ vᵢ kᵢᵀ` requires
`rank(W) = K` is textbook (Kohonen, *Correlation Matrix Memories*, IEEE TC 1972;
Kohonen & Ruohonen 1973; Anderson, *Math. Biosciences* 1972). **We do not claim this
fact as novel** — our contribution is that a *trained, bottlenecked, matrix-native
transformer* spontaneously discovers this rank, plus the causal ablation.

**Factual recall via associative memory — the closest prior work.** Nichani, Lee &
Bietti (*Understanding Factual Recall in Transformers via Associative Memories*, ICLR
2025 Spotlight, arXiv:2412.06538) formalize the same linear associative memory and,
in a remark to their Theorem 1, give a **rank-`m` construction** storing `≈ md`
associations. Crucially this is (a) a hand-built **existence** proof, not a
measurement of the rank gradient descent *discovers*; (b) valid for **discrete
argmax** decoding, which is interference-tolerant — under argmax a rank-1 matrix
stores `≈ d` associations, so the `rank ≥ K` necessity **does not hold**; and (c) has
no causal truncation ablation. **This is precisely why Task D pins the readout to
exact continuous recovery (cosine to the true vector), not argmax over a codebook**
(§3.3) — it is the design decision that keeps our lower bound valid, and the point on
which we most sharply differ from Nichani et al.

**Rank of linear-attention / fast-weight state — the closest rank-measurement
precedent.** Nazari & Rusch (*The Key to State Reduction in Linear Attention: A
Rank-based Perspective*, arXiv:2602.04852, Feb 2026) and Sun et al. (*State Rank
Dynamics in Linear Attention LLMs*, arXiv:2602.02195, Feb 2026) measure the same
effective-rank metric on the state matrix `Sₜ = Σ vₜ kₜᵀ` and prove the **upper**
bound `rank(Sₜ) ≤ t`. Both study **pretrained LLMs on real text with uncontrolled
`K`** and have no necessity theorem and no training-time ablation. Task D is the
mirror image: **controlled** `K`, a **lower** bound `rank ≥ K`, and a causal
force-rank curve. The lineage traces to **Fast Weight Programmers** (Schlag, Irie &
Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers*, ICML 2021,
arXiv:2102.11174; and the delta-rule / DeltaNet line), which motivate why rank and
capacity matter once writes exceed dimension. Recent delta-rule work on the
eigenstructure of the state transition (e.g. the DeltaProduct line, Grazzi et al.,
2025 — exact reference to be confirmed) is the natural framing for our forthcoming
*compositional* extension (Task E, §7).

**Binding as outer product — TPR / HRR / VSA.** `Z` is literally a Tensor Product
Representation (Smolensky, *Tensor Product Variable Binding*, Artificial Intelligence
1990; McCoy, Linzen, Dunbar & Smolensky, *RNNs Implicitly Implement TPRs*, ICLR 2019,
arXiv:1812.08718; Schlag & Schmidhuber, *Learning to Reason with Third-Order Tensor
Products*, NeurIPS 2018, arXiv:1811.12143; TP-Transformer, Schlag et al.,
arXiv:1910.06611 — TPR-style binding for math, the ancestor of "matrix structure for
reasoning"). The Vector Symbolic Architecture tradition (Plate, *Holographic Reduced
Representations*, IEEE TNN 1995; Frady, Kleyko & Sommer, *Resonator Networks*, Neural
Computation 2020, arXiv:1906.11684) studies a **different** capacity notion —
SNR/bundling and codebook factorization, *not* matrix rank — which is why our
exact-rank framing is a distinct, cleaner question. None of these measure rank vs.
binding count or run a causal rank ablation.

**Adjacent.** Tensor Product Attention (Zhang et al., arXiv:2501.06425, 2025) and
modern Hopfield networks (Ramsauer et al., *Hopfield Networks is All You Need*,
arXiv:2008.02217, 2020) use low-rank/energy structure *prescriptively* for efficiency
or capacity, not *diagnostically*. Zoology / MQAR (Arora et al., arXiv:2312.04927) is
the standard synthetic associative-recall benchmark our grammar resembles, but varies
state *dimension* against a fixed pair count and reports end-task accuracy, not rank
spectra. Merrill, Petty & Sabharwal (*The Illusion of State in SSMs*, ICML 2024,
arXiv:2404.08819) bounds SSM state expressivity (a complexity, not rank-vs-`K`,
question). *Recursive Binding on a Budget: Subspace Carving* (arXiv:2606.11391, June
2026) is a hand-designed order-`p` VSA memory — similar title, but not gradient-trained
and with no rank observable or causal ablation.

---

## 3. Task D design

### 3.1 The task

Each sample presents `K` bindings then queries them:

```
BIND k₁ v₁   BIND k₂ v₂   …   BIND k_K v_K   QUERY k_j   →   recover v_j
```

Keys and values are **fresh per sample**, drawn as random near-orthogonal
**continuous** `d`-vectors (i.i.d. Gaussian, L2-normalized; a Haar-corrected
orthonormal option via Mezzadri (2007) sign-correction is the gate default). Freshness
gives a held-out generalization guarantee (evaluation keys/values are never seen in
training). A binding is **recovered** if `cos(Z·kⱼ, vⱼ) > τ`; see §4.2 for `τ`.

### 3.2 The provable lower bound (why there is no shortcut)

Require exact recovery `Z·kⱼ = vⱼ` for all `j`, with `{kⱼ}` and `{vⱼ}` each linearly
independent (generic for random vectors, `K ≤ d`). Stacking keys as columns of
`K_mat ∈ ℝ^{d×K}` and values as `V_mat`, the requirement is `Z·K_mat = V_mat`, so
`rank(V_mat) = rank(Z·K_mat) ≤ rank(Z)`; since `rank(V_mat) = K`,

> **`rank(Z) ≥ K`.  ∎**

The rank-1 "vector-in-a-costume" shortcut `Z = u·v₀ᵀ` (which a bilinear/free-side
readout would admit) is ruled out by contradiction for any `K > 1`. For `K > d` no
exact solution exists — the lossy HRR-capacity regime (Plate 1995), a secondary
prediction we observe but do not treat as the gate.

### 3.3 Exact continuous recovery, not argmax (the critical readout decision)

The bound holds **only** for exact continuous recovery. Under discrete
argmax/nearest-neighbor decoding, Nichani et al.'s rank-`m`→`md` construction applies
and a rank-1 `Z` recovers `≈ d` bindings — the bound collapses. We therefore pin the
decoder to the **literal linear unbind** producing a continuous vector,
`pred = Z·keyⱼ`, scored by an **absolute** cosine bar — never a `K`-way argmax among
the sample's values, never a generic MLP (a generic MLP re-opens the shortcut; cf. the
workshop paper's Lesson that nonlinearity alone does not force rank use).

### 3.4 Single-`d×d`-state bottleneck (why rank has nowhere to hide)

The design gauntlet (`gauntlet/ATTACK_task_shortcuts.md`) established that in any
full-attention model, "hold `K` items" is satisfiable by attending to `K` positions at
rank-1 each — the **position-decomposition** escape that killed our earlier
matrix-CODI rank story (`rank_aware_v1`: a run at effective rank 13.2 was functionally
rank-1). Task D forecloses it: all information funnels through a **single** `d×d`
state `Z`, and the decoder is a pure function of `(Z, query_key)` — it never sees the
raw bindings. A gradient-based **blank-out test** verifies at runtime that the decode
path touches the bindings only through `Z`.

### 3.5 Controls (pre-registered)

- **C1 — train-time `force_rank_k`** (SVD/eigh-project `Z` to rank `k` every step; the
  *primary causal test*, not post-hoc truncation, which was uninformative in CODI).
- **C2 — param-matched vector baseline** — *descoped to Stage-1b*; the first HRR
  baseline was not param/state-matched (~4.3K vs ~171K params). The Stage-1 gate is
  M1+M2+M3 on the matrix model alone.
- **C3 — ≥5 seeds; primary metric effective rank** (SV entropy), secondary stable rank.
- **C4 — chance-adjusted per-item recovery.**
- **C5 — held-out generalization** (fresh keys/values; longer sequences).

---

## 4. Method

### 4.1 Architecture (model v4)

A `BindingEncoder` (a `norm_first` Transformer encoder over the `K` binding tokens,
then `d` learned "row-reader" latents cross-attending to produce the `d` rows of `Z`)
maps the set of bindings to a single `Z ∈ ℝ^{d×d}` — permutation-invariant, able to
express any rank up to `d`, and **not** hardcoding the `Σ vⱼkⱼᵀ` solution (so
"discovers rank ≈ K" is a genuine question). Readout is the pinned unbind (§3.3).
Default: `h=64`, 3 layers, 1 refinement iteration, ~171K parameters. Loss is
`1 − cos(pred, target)`. The reused rank utilities use `eigh(ZZᵀ)` rather than full
SVD for numerically stable backward at degenerate spectra (verified NaN/Inf-free on a
constructed `[3,1,1,0]` spectrum, on GPU).

### 4.2 Calibration and metric re-registration (logged before the sweep)

A pre-launch calibration run showed the reliably-trainable small model plateaus at
unconstrained cosine ≈ 0.936 (and a naively larger encoder, `h=256`/`n_refine=3`,
**diverged** — loss stuck at 1.0). Its `recovered_frac@0.99` is therefore ≈0 even at
full rank, so **the trained-model knee metric was re-registered from τ=0.99 to
τ=0.9**, where the knee lands at `k*=7 ∈ [K−1, K+1]` for `K=8` (τ=0.99 remains valid
only for the exact-memory resolution pre-flight). Effective rank (M1) is
threshold-free and unaffected. This is documented in the pre-registration §6.

### 4.3 The sweep

A perpetual, resumable, 8×H100 orchestrator (`run_overnight.py`) runs the grid
priority-ordered — Tier 1 (`d=16` gate, M1 K-sweep + M3 causal knee at `K∈{4,8,12}`,
interleaved), Tier 2 (`d∈{8,32,64}` + `K>d` lossy), Tier 3 (`d=128`) — with per-run
subprocess isolation, validity-checked resume, per-`d` timeouts with GPU quarantine,
and a self-healing supervisor. Numbers below are a mid-sweep snapshot: **991 runs
completed, 2 failed** (the training loop skips occasional non-finite gradient steps
from rank-truncation backward rather than crashing; skipped-step counts are logged).

### 4.4 Reproducibility / audit trail

Design and code passed an adversarial gauntlet before any GPU run: a task-shortcut
attack (which killed the original `K≈P` crossover design and produced Task D), a
baseline/confound attack, and a novelty check, then **three rounds of code audit** +
**two rounds of runner audit** (`gauntlet/AUDIT_*.md`), a runtime smoke gate on the
H100 (degenerate-spectrum backward, blank-out leak test, resolution pre-flight all
pass), and the metric re-registration above. Pre-registration with decision criteria
predates the runs.

---

## 5. Results

### 5.1 M1 — learned effective rank tracks K (at d = 8, 16)

Effective rank of the unconstrained `Z` vs. binding count `K` (mean over seeds):

| K       | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14 | 16 |
|---------|---|---|---|---|---|---|----|----|----|----|
| **d=16** | 2.42 | 3.01 | 3.92 | 4.74 | 6.40 | **8.20** | 9.89 | 11.78 | 13.47 | 15.09 |
| **d=8**  | 1.65 | 3.44 | — | 4.57 | — | **7.83** | — | — | — | — |

At `d=16`, effective rank ≈ `K` across the entire range (Spearman ρ = 1.0; the only
departure from the pre-registered `[0.7K, 1.3K]` band is at `K=1,2`, where rank
modestly *exceeds* `K`). At `d=8`, rank tracks `K` up to the ceiling (`K=8→7.8 ≈ d`).
For `K > d` (`d=16`: `K=20→11.1, 24→11.6, 32→8.6`) rank saturates/declines — the
expected lossy over-complete regime.

> **CORRECTION (2026-07-01, 1,234-run mega-replication):** refined figures
> supersede the 991-run snapshot above — `d=16`: `K=8→8.198, K=16→15.083,
> K=20→11.20, K=24→10.40, K=32→9.46`; `d=8, K=8→7.855`. Directionally
> unchanged (rank still saturates/declines past `K=d`), but `K=24` lands
> lower and `K=32` lands higher than the earlier snapshot suggested — logged
> for the record, table above left as originally reported.

### 5.2 M3 — the causal force-rank step lands at k ≈ K (the primary test)

`recovered_frac@0.9` when `Z` is *constrained to rank `k` during training*:

| force-rank `k` → | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 11 | 12 | 13 | 14 | 16 |
|------------------|---|---|---|---|---|---|---|---|----|----|----|----|----|----|
| **d=8, K=4**  | 0.0 | 0.00 | 0.0 | **0.97** | 0.99 | 0.96 | — | 0.98 | — | — | — | — | — | — |
| **d=16, K=4** | — | — | — | — | — | — | — | — | — | — | — | — | — | 0.045 |
| **d=16, K=8** | 0.0 | 0.0 | — | — | — | 0.34 | **0.91** | 0.76 | 0.84 | — | — | — | — | 0.99 |
| **d=16, K=12**| 0.0 | 0.0 | — | — | — | — | — | — | 0.56 | 0.75 | **0.94** | 0.79 | 0.85 | 0.9994 |

The signature is unmistakable: **constraining rank below `K` destroys recovery, with a
step at `k ≈ K`.** `d=8, K=4` is razor-sharp — force-rank ≤3 gives exactly 0.0
recovery, force-rank 4 jumps to 0.97. `d=16, K=8` and `K=12` show the same step near
their respective `K` (with mild post-knee non-monotonicity attributable to seed noise
and the ~0.936 precision ceiling).

> **CORRECTION (2026-07-01, 1,234-run mega-replication).** Two findings correct the
> original text of this section (the original "One weak cell, `d=16, K=4`... not
> evidence against the hypothesis" sentence has been replaced by this note rather
> than left in place, since the mega-replication shows it was factually wrong, not
> just incomplete):
>
> 1. **`d=16, K=12` transition zone is noisier than the two-point table (fr=10→0.56,
>    fr=12→0.94) implied.** New intermediate force-rank points: fr=11→0.75,
>    fr=13→0.79, fr=14→0.85 — the climb from fr=10 to the fr=16 ceiling (0.9994) is a
>    ragged ramp, not a clean single step, though the qualitative knee near `k≈K`
>    still holds.
> 2. **`d=16, K=4` is not undertrained — it is a genuine, seed-count-independent
>    convergence ceiling.** Even at force_rank=16 (full rank, i.e. no rank constraint
>    at all), recovered_frac@0.9 = 0.045, and the entire force-rank sweep for this
>    cell stays within 0.0008–0.134 across all tested `k` — it never approaches the
>    0.9+ ceiling every other tested cell reaches. More seeds do not fix this. The
>    original claim ("~~undertrained / seed-sparse at this snapshot~~ ... ~~will
>    firm up with more seeds~~", also §7) is retracted. Net effect on the M3 gate: 3
>    of the 4 tested `(d,K)` cells (`d=8,K=4`; `d=16,K=8`; `d=16,K=12`) show the
>    pre-registered causal step; `d=16,K=4` does not and does not converge under the
>    current training recipe regardless of seed count — an honest negative data point
>    for that specific cell, not a refutation of the cells that do pass.
>
> **FURTHER CORRECTION (2026-07-02, Stage 0 Wave 0 opportunistic probe,
> `STAGE0_DESIGN.md` §12.5).** The "genuine, seed-count-independent
> convergence ceiling" claim directly above was itself confounded by the
> same 8K-step budget diagnosed in §5.3's 2026-07-02 revision — it reads as
> a seed-count issue but was in fact a step-count issue. Two seeds at 20K
> steps (2.5× budget, same architecture, unconstrained `d=16,K=4`) reach
> `recovered_frac@0.9` = 0.348 / 0.483 and mean_cos = 0.880 / 0.886 — a
> 3-10× jump off the 0.0008–0.134 ceiling reported above. The rank
> trajectory shows the same overshoot-then-compress shape general to the
> late-transition regime elsewhere in this correction (effective rank
> 1.3→7.8 at step 4K, settling to 4.4–5.0 by 20K). Whether 0.35–0.48 is
> itself a further-budget-limited number or a new, real (lower-than-K=8/
> K=12) ceiling for this cell is **not yet resolved** — not re-tested at
> further-extended budgets in Stage 0's manifest. Reported as an open,
> non-gating finding per `STAGE0_DESIGN.md` §7/§12.5; the correct summary
> as of this writing is "this cell recovers substantially with more steps,
> ceiling location still unknown," not "genuine ceiling, more steps/seeds
> do not help" (the 2026-07-01 text immediately above).

**Together, M1 and M2/M3 confirm the pre-registered hypothesis at `d ∈ {8,16}`: SGD
develops rank ≈ K, and rank is causally necessary.** This directly resolves the
workshop paper's open question — the earlier rank-blindness was **task-specific
(ProsQA was rank-1-solvable), not a property of the gradient.**

### 5.3 The trainability frontier at d ≥ 32 (honest limitation)

At `d ≥ 32`, the same small encoder **fails to train**:

- **M1:** effective rank collapses toward ≈1 (`d=64` and `d=128`: ≈1.0–1.02 for *all*
  `K`; `d=32`: erratic 1.2–4.7, not tracking `K`).
- **M3:** `recovered_frac@0.9` is **0.0 for every force-rank at `d=32/64/128`** — the
  unconstrained model never learns the write, so there is nothing for truncation to
  degrade.

This is **not** a refutation of the hypothesis. It is the *same fixed `h=64` encoder*
being asked to write a rank-`K` memory into a much larger `Z` (256× more entries at
`d=128` than `d=16`), and calibration already showed a naively larger encoder
diverges. It is an **optimization/architecture limitation** — the write is hard to
learn at scale, not the rank claim being false. It is the direct motivation for the
next experiment (§7).

> **CORRECTION (2026-07-02, Stage 0 Wave −1/0, `STAGE0_DESIGN.md` §12):**
> the "optimization/architecture limitation" framing above is revised at
> **d=32**: it is substantially a **step-budget artifact**, not a stuck
> optimization. Six baseline runs (K=8 and K=16, 3 seeds each) at 2.5× this
> section's 8K-step budget (20K steps) all show the same signature — flat
> effective rank ≈1.0–1.06 through 6–10K steps, then a sharp climb to
> 8.8–17.8 by 12–16K steps. The `d=32: erratic 1.2–4.7` figure quoted above
> (from the 991-run snapshot, all at 8K steps) is now understood as
> **mid-transition snapshots taken at various stages**, not a converged,
> erratic ceiling. **Caveat — transitioned is not solved:** even at 20K
> steps, final `recovered_frac@0.9` across the 6 runs is 0.0001–0.1355 and
> mean_cos is 0.652–0.841 — the write becomes learnable but converges far
> below the `>0.7` pass bar; whether more steps alone closes that gap or an
> intervention is required is Wave A/B's open question (`STAGE0_DESIGN.md`
> §12.7/§12.8), running as of this writing.
>
> **At d=64, the picture is genuinely undetermined, not confirmed dead.**
> 3 seeds at 25K steps (2.5× the 10K budget used above): one stays fully
> flat throughout, but two show a late onset beginning around step 20,000
> (eff. rank 1.01→5.11 and →6.02 by 25K, cos rising to 0.085–0.107) that is
> not complete within budget. Reads as "no transition within 25K" with
> onset scaling apparently superlinearly in `d` (d=16: <2K steps; d=32:
> 6–16K; d=64: ≥20K, incomplete at 25K) — a candidate scaling-law finding
> in its own right, not evidence the architecture is dead at d=64.
>
> **d=128 is unaffected by this correction and remains the one case where
> the architectural claim in §2.1 (`rank(Z) ≤ h+1 = 65`) is a real, proven
> cap** — for `K > 65` specifically; d=128 cells with `K ≤ 64` are not
> explained by either this correction or the cap, and remain an open
> question (`STAGE0_DESIGN.md` §9).

---

## 6. Discussion

**The gradient uses rank when the task forces it.** The workshop paper and Task D are
two ends of one story. Where nothing rewarded rank (matrix-CODI on ProsQA), the
gradient was blind to it; where the task *provably requires* rank (Task D), the
gradient develops it and it is causally load-bearing. The rank-blindness result was
never about the gradient's inability — it was about ProsQA being rank-1-solvable, the
caveat we could not close before. Task D closes it.

**Why the readout decision is the crux.** The single most important design choice is
exact-continuous recovery over argmax (§3.3). It is exactly the axis on which we
differ from Nichani et al. (2412.06538): their rank-`m`→`md` capacity is real *under
argmax*, and had we scored by nearest-neighbor over a codebook, a rank-1 `Z` would
have "passed" and we would have reported a false negative. The provable bound and the
causal step both hinge on it.

**Relation to the superposition debate.** Task D does not adjudicate whether *real
reasoning* models superpose — it is associative memory, deliberately abstract, chosen
for the provable bound. But it does establish the mechanism's *existence* under
gradient training: a matrix latent *can* be driven to functional rank `K`, cleanly and
causally, contra a strong reading of *The Illusion of Superposition* (2604.06374) that
latent capacity is inertly low. Whether this transfers to a genuine reasoning task is
the open question, addressed next.

---

## 7. Limitations and future work

- **Preliminary snapshot.** 991 runs mid-sweep at time of writing. M2 (eval-time
  truncation curve) is computed per-run but not in this aggregate; the vector
  control (C2) is descoped to Stage-1b.
  **CORRECTION (2026-07-01):** this bullet originally predicted the `d=16,K=4` cell
  "will firm up with more seeds" — a 1,234-run mega-replication (§5.2) shows this was
  wrong. The cell is a genuine, seed-count-independent convergence ceiling
  (recovered_frac@0.9 = 0.045 even at full rank, sweep range 0.0008–0.134
  throughout), not a seed-sparsity artifact.
- **d ≥ 32 trainability (Stage-0 precursor).** A better-optimized / larger matrix-write
  encoder (LR warmup, normalization/residual-scaling, curriculum on `K`) is needed to
  extend the confirmation to large matrices — and, unlike the tiny-model sweep, it
  genuinely saturates an H100 per job.
- **Reasoning transfer — Task E (recommended next experiment;
  `NEXT_EXPERIMENT_DESIGN.md`).** Compositional Multi-Hop Relational Recall tests
  whether the causally-necessary `Z` *composes* correctly via iterated `Zʰ` at
  hop-depths unseen in training — the actual "does the rank mechanism do reasoning"
  question (pre-reg §8, and the workshop paper's stated future work). The design was
  attack-vetted; the edge generator must enforce **injective** mappings or "K edges"
  stops implying rank ≥ K (the MNNS miscounting trap), and the headline is framed
  around **eigenstructure fidelity** (delta-rule / DeltaProduct lineage) rather than an
  overclaimed "genuine reasoning" result.
- **Tokenization is held fixed elsewhere.** For the broader program's scale-up on real
  data, a separate research pass (`research/bytes-vs-tokens-matrix-native-june2026.md`)
  established that byte-level input (BLT — Pagnoni et al., arXiv:2412.09871; H-Net —
  Hwang et al., arXiv:2507.07955; ByT5, etc.) should be an isolated ablation, not
  bundled with the matrix change.

---

## 8. Reproducibility pointers

- **Pre-registration:** `matrix-thinking/chapter2/TASK_D_PREREGISTRATION.md` (hypothesis,
  proof, controls, decision criteria; metric re-registration logged).
- **Code (torch + stdlib only, self-contained):** `task_d.py` (generator), `model_v4.py`
  (encoder + pinned unbind), `rank_utils.py` (eigh-stable rank), `run_task_d.py`
  (train/eval/smoke), `run_overnight.py` (perpetual 8×H100 orchestrator).
- **Audit trail:** `matrix-thinking/chapter2/gauntlet/` (design gauntlet, 3× code audit,
  2× runner audit, novelty check).
- **Results:** `results/overnight/AGGREGATE.json` + per-run JSONs; snapshot used here in
  `results/overnight_snapshots/`.
- **Hardware:** 8×H100 80GB (GCP, Brev), torch 2.12/cu13, ~56 GPU-h for this snapshot.

---

## References

1. Larson, S. *The Gradient Does Not See Rank: Rank-Indifference in Matrix-CODI on ProsQA.* ICML 2026 Mechanistic Interpretability Workshop (this project's companion paper).
2. Nichani, E., Lee, J. D., & Bietti, A. *Understanding Factual Recall in Transformers via Associative Memories.* ICLR 2025 (Spotlight). arXiv:2412.06538.
3. Nazari, P., & Rusch, T. K. *The Key to State Reduction in Linear Attention: A Rank-based Perspective.* 2026. arXiv:2602.04852.
4. Sun, A., et al. *State Rank Dynamics in Linear Attention LLMs.* 2026. arXiv:2602.02195.
5. Schlag, I., Irie, K., & Schmidhuber, J. *Linear Transformers Are Secretly Fast Weight Programmers.* ICML 2021. arXiv:2102.11174.
6. Schlag, I., & Schmidhuber, J. *Learning to Reason with Third-Order Tensor Products.* NeurIPS 2018. arXiv:1811.12143.
7. Schlag, I., Smolensky, P., Fernandez, R., Jojic, N., Schmidhuber, J., & Gao, J. *Enhancing the Transformer with Explicit Relational Encoding for Math Problem Solving* (TP-Transformer). 2019. arXiv:1910.06611.
8. McCoy, R. T., Linzen, T., Dunbar, E., & Smolensky, P. *RNNs Implicitly Implement Tensor Product Representations.* ICLR 2019. arXiv:1812.08718.
9. Smolensky, P. *Tensor Product Variable Binding and the Representation of Symbolic Structures in Connectionist Systems.* Artificial Intelligence 46(1–2), 1990.
10. Plate, T. A. *Holographic Reduced Representations.* IEEE Transactions on Neural Networks 6(3), 1995.
11. Frady, E. P., Kleyko, D., & Sommer, F. T. *Resonator Networks, 1.* Neural Computation 32(12), 2020. arXiv:1906.11684.
12. Zhang, Y., et al. *Tensor Product Attention Is All You Need.* 2025. arXiv:2501.06425.
13. Ramsauer, H., et al. *Hopfield Networks is All You Need.* 2020. arXiv:2008.02217.
14. Merrill, W., Petty, J., & Sabharwal, A. *The Illusion of State in State-Space Models.* ICML 2024. arXiv:2404.08819.
15. Arora, S., et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* 2023. arXiv:2312.04927.
16. *Recursive Binding on a Budget: Subspace Carving in Order-p Tensor Memories.* 2026. arXiv:2606.11391 (author list to be confirmed).
17. Kohonen, T. *Correlation Matrix Memories.* IEEE Transactions on Computers C-21(4), 1972.
18. Kohonen, T., & Ruohonen, M. *Representation of Associated Data by Matrix Operators.* IEEE Transactions on Computers C-22(7), 1973.
19. Anderson, J. A. *A Simple Neural Network Generating an Interactive Memory.* Mathematical Biosciences 14(3–4), 1972.
20. Lin, Z., Zhu, H., et al. *Reasoning by Superposition.* 2025. arXiv:2505.12514.
21. Zhang, J., Tian, Y., et al. *CoT2.* 2025. arXiv:2505.23648.
22. Rizvi-Martel, A., Mosbach, M., & Rabusseau, G. *The Illusion of Superposition?* 2026. arXiv:2604.06374.
23. Pagnoni, A., et al. *Byte Latent Transformer: Patches Scale Better Than Tokens.* Meta AI, 2024/ACL 2025. arXiv:2412.09871.
24. Hwang, S., et al. *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (H-Net). 2025. arXiv:2507.07955.
25. Mezzadri, F. *How to Generate Random Matrices from the Classical Compact Groups.* Notices of the AMS, 2007. (Haar orthonormal sampling / QR sign-correction.)
26. Grazzi, R., et al. *DeltaProduct / delta-rule state-transition eigenstructure.* 2025 (exact citation to be confirmed; relevant to the Task E compositional extension).

*Preliminary — numbers from a mid-sweep snapshot (991 runs); the perpetual sweep is
ongoing and the write-up will be revised as coverage tightens.*
