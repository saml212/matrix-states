# Task D Pre-Registration — Chapter 2 Rank Gate

> **STATUS: CLOSED — CONFIRMED (2026-07-04 consolidation header). Results:
> `TASK_D_WRITEUP.md`.** This pre-registration is kept verbatim below,
> unmodified, per scientific-integrity convention for locked pre-registered
> designs — do not edit the body to match results after the fact.

**Pre-registered 2026-07-01, before any training.** This document locks the
hypotheses, design, controls, and decision criteria *in advance* so a result
cannot be re-interpreted after the fact. It **supersedes the K≈P crossover
design** in `CHAPTER_2_DESIGN.md`, which was killed by the 2026-07-01 design
gauntlet (see `archive/chapter2-gauntlet/gauntlet/`): a rank-1 matrix `Z=u⊗v₀` holds `d` items
via its free vector side, so rank was never the binding capacity constraint and
the crossover would appear at K≈P·d, not K≈P — flat everywhere testable.

Novelty check: **cleared** — Task D is novel (`research/task-d-novelty-july2026.md`).
The check forced one correctness fix folded in below: the readout must require
**exact continuous recovery**, not argmax over a codebook, or the rank≥K bound
collapses (Nichani et al. 2412.06538). See §2, §3, §10.

---

## 1. The sharpened question

The workshop paper ("The Gradient Does Not See Rank") showed bolt-on matrix-CODI
is rank-blind on ProsQA, but left open whether that was the *objective* being
rank-blind or the *task* (ProsQA) being rank-1-solvable. Task D removes both
ambiguities by testing in a setting where rank is **provably necessary** and
**no rank-1 shortcut exists by construction**:

> **H_primary.** When a task's exact solution provably requires `rank(Z) ≥ K`,
> a from-scratch matrix-native model trained by plain SGD (a) develops effective
> rank ≈ K in its matrix state, and (b) its accuracy collapses under rank-k
> truncation for k < K.

- **CONFIRM** ⇒ the CODI rank-blindness was task-specific (ProsQA was
  rank-1-solvable); gradient *can* recruit rank when the task demands it;
  matrix-thinking is alive → proceed to reasoning-transfer.
- **FALSIFY** (model fits with effective rank ≪ K, or force-rank-`k`<K training
  reaches ceiling accuracy) ⇒ the gradient is rank-averse *even when rank is
  provably required*; matrix-thinking's core premise is dead in this setting →
  pivot.

Both outcomes are decisive and publishable. This reframes Chapter 2 from the
vague "does rank track reasoning" to the sharp, falsifiable "**does SGD learn to
use rank when the task provably requires it.**"

---

## 2. Task D — tensor-product key/value binding

**Grammar (one sample):**
```
BIND k₁ v₁   BIND k₂ v₂   ...   BIND k_K v_K   QUERY k_j   SEP  → target v_j
```
- `k₁..k_K`, `v₁..v_K` are drawn **fresh per sample** as random near-orthogonal
  **continuous** `d`-dimensional vectors (i.i.d. Gaussian, L2-normalized). **Not a
  small discrete codebook** — see the readout note and §3 for why a codebook +
  argmax decoding would break the rank≥K necessity. Fresh-per-sample vectors close
  the static-vocabulary memorization shortcut and enable a held-out generalization
  test on never-seen key/value vectors.
- BIND order is shuffled (no positional-recency shortcut).
- The queried `j` is uniform over the K bindings. (Training may query several/all
  j per sample; see M-metrics.)

**Readout, architecturally pinned (this is load-bearing):** the decoder is
**not** a generic MLP over `flatten(Z)`, and **not** an argmax/nearest-neighbor
classifier over a value codebook. It is the literal unbind the hypothesis claims
the matrix performs, producing a **continuous** vector:
```
predicted_value = Z · key_j                  # matrix–vector product, continuous output
```
scored against the true `v_j` by an **absolute** criterion — cosine similarity
`cos(Z·key_j, v_j) > τ` (pre-registered high τ, e.g. 0.9) and/or reconstruction
MSE below threshold — **never a relative K-way argmax** among the sample's values.

**Why the readout must be exact continuous recovery (critical):** Nichani, Lee &
Bietti (ICLR 2025, arXiv:2412.06538, Thm 1 rank-*m* remark) prove a **rank-m**
matrix can store `≈ md` associations *under discrete argmax decoding* (random
Gaussian mixing, interference-tolerant). Under argmax, a rank-1 `Z` would recover
`K ≤ d` bindings and the `rank(Z)≥K` necessity **collapses** — reopening the
gauntlet's master shortcut. The lower bound holds **only** for exact continuous
recovery of the actual value vector. A generic MLP readout is likewise forbidden
(gauntlet Arch-4; KILL_LIST Lesson 5: nonlinearity alone does not force rank use).

---

## 3. The provable lower bound (why there is no shortcut)

Require exact recovery: `Z·k_j = v_j` for all `j=1..K`, with `{k_j}` linearly
independent (generic for random vectors, K ≤ d) and `{v_j}` linearly independent.
Stack keys as columns of `K_mat ∈ ℝ^{d×K}`, values as `V_mat ∈ ℝ^{d×K}`. Then
`Z·K_mat = V_mat`, and since `rank(V_mat) = rank(Z·K_mat) ≤ rank(Z)` while
`rank(V_mat) = K`:

> **rank(Z) ≥ K.**   ∎

The gauntlet's master shortcut `Z=u⊗v₀` (rank 1) is ruled out by contradiction
for any K>1. This is the "no shortcut" property that Task A′ and Task B both
lacked — it is a hard theorem, not a hoped-for construction.

**The bound is conditional on the readout (do not skip this).** `rank(Z)≥K` holds
for **exact continuous** recovery. It does **not** hold under discrete
argmax/nearest-neighbor decoding: Nichani et al. (2412.06538) show a rank-m matrix
stores `≈ md` associations in that regime. This is exactly why §2 pins the readout
to continuous `Z·key_j` scored by an absolute cosine/MSE bar, not a K-way argmax.
The two regimes are distinct predictions: exact-recovery ⇒ rank is the binding
constraint (the gate); argmax/interference-tolerant ⇒ rank stores `≈md` (a later,
separate test, not this gate).

**Honest scoping of novelty.** The linear-algebra fact itself — "exact recovery of
K independent orthonormal key→value pairs requires rank K" — is **classical**
(Kohonen 1972/1973; Anderson 1972; correlation-matrix / optimal linear associative
memory). We do **not** claim it as novel. The contribution is the *trained, causal,
bottlenecked, matrix-native-transformer* framing: whether SGD **spontaneously
discovers** rank ≈ K under a hard bottleneck, plus the forced-rank-k training
ablation and vector-state baseline — none of which exist in prior work (§10).

For **K > d**, no exact solution exists (over-complete binding); this is the
well-studied *lossy* HRR/VSA capacity regime (Plate 1995; resonator networks)
with known capacity-vs-dimension curves — a second, literature-grounded
prediction, not tested in the first gate.

---

## 4. Architecture (v4, single-state P=1 first)

```
input tokens ──▶ ENCODER (matrix-native blocks; may iterate/refine)
                    │  writes into
                    ▼
              Z  ∈ ℝ^{d×d}        ← the ONLY state that survives the bottleneck
                    │  hard attention mask
                    ▼
              DECODER: given key_j, emit decode(Z · key_j)
```

- **P = 1 (single matrix state) for the gate.** With one slot, position-
  decomposition is *architecturally impossible* — the failure mode that killed
  every prior rank result cannot occur.
- **Hard bottleneck mask (mandatory).** Decoder/readout positions attend to `Z`
  (and causally among themselves for multi-item decode) and **nothing else** —
  never to raw BIND/QUERY tokens. Enforced by an explicit mask, not hoped for.
- **Blank-out unit test (mandatory, pre-training).** After the encoder writes `Z`,
  zero/corrupt the raw-input KV cache and confirm decode logits are **bit-for-bit
  unchanged**, across the full decode sequence. If they change, the bottleneck
  leaks and every downstream number is meaningless — fix before training.
- `d = 16` to start (rank range 1..16 is meaningful and achievable). Encoder
  depth/width fixed at smoke-test time; ~1–5M params.

---

## 5. Mandatory controls

| # | Control | Closes |
|---|---------|--------|
| C1 | **Train-time `force_rank_k`** (SVD-project Z to rank k every step) — the **primary causal test**, not post-hoc SVD. Reuse `rank_aware_v1`'s existing implementation. | Post-hoc SVD gave uninformative flat curves in CODI; conflates "instrument insensitive" with "mechanism absent". |
| C2 | **Param-matched VECTOR control arm** (d²-dim vector state, reshape-INequivalent binding). **DESCOPED from Stage-1 → Stage-1b** — the round-1 HRR baseline was not param/state-matched (~4.3K vs ~171K params; 16-dim vs 256-dim state), so M4 is not yet interpretable. Needs its own design + audit. | Frobenius attention is reshape-equivalent to vector attention; proves any matrix advantage is "matrixness", not just weight-sharing. |
| C3 | **≥5 seeds**; primary rank metric = **effective rank** (entropy of singular values), secondary = stable rank. Both pre-registered; no post-hoc metric shopping. | rank_aware_v1 saw 3× rank spread across seeds at matched accuracy — rank is noisy. |
| C4 | **Chance-adjusted per-item (Hamming) accuracy**, not raw joint exact-match. | Raw exact-match drops ~(1/S)^K from combinatorics alone. |
| C5 | **Held-out generalization split**: eval on never-seen key/value vectors and longer sequences than trained. | Prevents "memorized instance statistics" masquerading as the mechanism. |

---

## 6. Pre-registered measurements & decision criteria

Fixed operating point for causal tests: `d=16`, `P=1`, `K=8`, **orthonormal
keys+values** (`--orthogonal`), **dense rank grid k = 1..d** (unless smoke forces a
change, logged + re-registered).

**Success metric (given the continuous readout, §2):** a binding is *recovered* if
`cos(Z·key_j, v_j) > τ`. Report at **τ ∈ {0.9, 0.95, 0.99}**; the **knee test uses
τ = 0.99** (primary). Round-1 audit finding (`archive/chapter2-gauntlet/gauntlet/AUDIT_validity.md`,
`AUDIT_adversarial.md`): with Gaussian near-orthogonal values a rank-(K−2) matrix
clears τ=0.9 on all K bindings ~90% of the time — smearing the knee. Two fixes
locked in: **(a) orthonormal keys+values** — the exact memory Σ vₖkₖᵀ then has a
degenerate spectrum (all σ=1), so rank-(K−1) truncation drops a *spread* direction
and every binding degrades to ≈√((K−1)/K) mean cosine (≈0.935 at K=8), pushing
*all* K bindings below τ=0.99 → `recovered_frac@0.99` collapses to ~0 (a sharp
knee); and **(b) high τ + dense grid**. "acc" =
fraction of K bindings recovered (per-item Hamming, C4); "ceiling" = unconstrained
model's converged recovery; "chance" ≈ 0 at high τ. Always report the **full cosine
distribution**, not just the thresholded rate.

> **CALIBRATION RE-REGISTRATION (2026-07-01, before the sweep, logged):** the
> pre-launch calibration run showed the *reliably-trainable* model (small h=64
> encoder; larger/deeper encoders diverged) plateaus at unconstrained cosine
> ≈ 0.936, so its `recovered_frac@0.99` is ≈ 0 even at full rank — τ=0.99 cannot
> resolve the *trained model's* knee. **The trained-model knee/decision metric is
> therefore re-registered to `recovered_frac@0.9`** (which lands the knee at
> k*=7 ∈ [K−1,K+1] for K=8), and this is what the aggregator reports. τ=0.99
> remains the criterion **only** for the exact-memory resolution pre-flight
> (smoke [6]), where the memory is exact and the knee is razor-sharp. Effective
> rank (M1) is threshold-free and unaffected.

**M1 — Learned effective rank vs K** (unconstrained model, K ∈ {1,2,4,8,12,16}).
- CONFIRM: effective rank grows with K, Spearman ρ(K, eff_rank) ≥ 0.8 across
  seeds AND eff_rank ∈ [0.7K, 1.3K] for K ≤ d.
- FALSIFY: eff_rank stays ≤ 2 while accuracy is high across K. (If observed with
  a *passing* blank-out test, this contradicts §3 and signals a readout leak to
  debug — not a valid result until the leak is ruled out.)

**M2 — Eval-time rank-k truncation curve** (unconstrained checkpoint at K=8;
truncate Z to **every** rank k = 1..d at inference; acc at τ=0.99).
- Define the knee k\* = smallest k with acc(k) ≥ 0.9·acc(k=d).
- CONFIRM: k\* ∈ [K−1, K+1] in ≥4/5 seeds. (The dense grid is required — the
  round-1 audit showed a coarse {1,2,4,8,16} grid skips k=5,6,7 and could miss a
  true knee at K−1.)
- FALSIFY: acc(k=1) ≥ 0.9·acc(k=d) (flat curve — the CODI signature).

**M3 — Train-time force-rank-k (PRIMARY causal test)** — train separate models
with Z constrained to rank k ∈ {1,2,4,8,16} throughout, at K=8.
- CONFIRM: best eval accuracy is near-chance for k < K and ≥ 0.9·ceiling for
  k ≥ K, with a monotone step at k = K (±1).
- HARD FALSIFY (premise dead): force-rank-1 reaches ≥ 0.9·ceiling at K=8 — a
  rank-1 solution exists despite the pinned unbind readout ⇒ investigate leak;
  if real, the setting is rank-blind and matrix-thinking is dead here.

**M4 — Vector control arm (C2)** — capacity = max K at ≥90% per-item accuracy;
also run M2 on the vector arm.
- Interpretation input (not a pass/fail gate): if the matrix arm sustains higher
  K and/or shows the M2 knee while the vector arm degrades gracefully with no
  knee, matrix structure is load-bearing. If they match, "rank" adds nothing
  over d²-dim vector capacity — report honestly.

**Overall gate:** PASS if M1 CONFIRM **and** M3 CONFIRM (M2 corroborating).
FAIL if M3 HARD FALSIFY or (M1 FALSIFY with a clean blank-out test). Anything
else = INCONCLUSIVE → diagnose (leak? optimization? metric?) before scaling.

---

## 7. Compute plan (8×H100)

- **Stage 1 — minimal gate (the user's chosen first cut, ~5–10 GPU-h; MATRIX
  MODEL ONLY):** unconstrained K ∈ {1,4,8,16} × 2 seeds (M1 + dense-grid M2);
  force-rank-k at K=8, k ∈ {2,4,6,7,8} × 2 seeds (M3 — grid straddles K per the
  audit). ~18 short runs. The vector control (C2/M4) is Stage-1b, not here.
  Decision: if M3 shows a step at k=K and M1 trends up, go to Stage 2; if
  flat/HARD FALSIFY, stop and pivot.
- **Stage 2 — full sweep (~60–90 GPU-h):** all K, k, and seed grids above at 5
  seeds; held-out generalization; effective-rank trajectories saved.
- Total ≤ ~100 GPU-h — trivial against the two-month cluster, thorough enough to
  publish either verdict.

---

## 8. Sequencing (what this gate unlocks)

1. **P=1 Task D gate** (this doc) — clean go/no-go on "does SGD use provably-
   required rank."
2. **If PASS →** P-slot sweep (reintroduces the P<K pigeonhole question; tests
   the HRR K≈c·P·d capacity regime), then **reasoning-transfer**: does the
   mechanism survive on an actual reasoning task (patched Task A′ with a
   relational query, or real data)? Mirrors the synthetic→real logic we set for
   tokenization.
3. **If FAIL →** matrix-thinking's premise is dead in the trainable regime;
   pivot (byte-level JEPA / other directions), write the negative up as a clean
   companion to the workshop paper.

---

## 9. Scope — what this does and does NOT show

- **Shows:** whether gradient descent recruits and uses matrix rank when a task
  *provably* requires it, with no shortcut and no positions to hide in. This is
  the cleanest possible test of the workshop paper's open question.
- **Does NOT show:** whether rank helps *real reasoning*. Task D is associative
  memory, deliberately abstract — chosen for provability, not realism. Reasoning
  relevance is the transfer step (§8), gated on this passing.

---

## 10. Related work & positioning (must-cite; from `research/task-d-novelty-july2026.md`)

Novelty check verdict: **no paper trains a transformer end-to-end under a hard
single-`Z` bottleneck, measures effective rank vs. a controlled binding count K,
uses a provable rank≥K necessity for exact recovery, AND causally forces rank-k
during training vs. a param-matched vector baseline.** Task D is that combination.
Must cite and distinguish, in priority order:

1. **Nichani, Lee & Bietti — arXiv:2412.06538 (ICLR 2025).** *The* paper to
   distinguish head-on: same linear associative memory, and their Thm 1 rank-m
   remark (rank-m stores ≈md under argmax) is the "isn't this already rank-aware?"
   reviewer attack. Our answer: they give a hand-built **existence** construction
   for **discrete-argmax** capacity; we measure the rank **gradient descent
   discovers**, prove a **necessity** bound for **exact** recovery, and run a
   **causal** rank-k ablation. Their transformer has full self-attention over raw
   tokens — **no hard bottleneck**.
2. **Nazari & Rusch — arXiv:2602.04852; Sun et al. — arXiv:2602.02195 (both Feb
   2026).** Closest *rank-measurement* precedent — same effective-rank metric on
   linear-attention state matrices, prove the **upper** bound `rank(S_t) ≤ t`. But
   on **pretrained LLMs over real text**, uncontrolled K, no necessity theorem, no
   training-time rank-k ablation. We are the mirror image: controlled K, a **lower**
   bound, a causal ablation.
3. **Schlag, Irie & Schmidhuber — arXiv:2102.11174 (DeltaNet/FWP).** Origin of
   "linear-attention state as associative memory"; motivates why rank/capacity
   matters once writes exceed dimension. (FWP lineage = Schlag, a Tier-1 outreach
   target.)
4. **TP-Transformer (Schlag et al., 1910.06611) + TPDN (McCoy et al., 1812.08718).**
   Binding-as-outer-product ancestry — `Z` is a TPR-style binding matrix. Neither
   measures rank vs. binding count.
5. **Plate 1995 (HRR); Frady/Kleyko/Sommer resonator networks (1906.11684).** Cite
   to scope "capacity" correctly — their VSA capacity is SNR/bundling/codebook
   factorization, **not** matrix rank; distinguishes our exact-rank framing.
6. **Arora et al. — arXiv:2312.04927 (Zoology / MQAR).** The standard synthetic
   associative-recall benchmark family our grammar resembles; they vary state
   *dimension* vs. fixed pair count and measure end-task accuracy, not rank spectra.
7. **Subspace Carving — arXiv:2606.11391 (June 2026).** Similar title; hand-designed
   VSA, not gradient-trained, no rank observable, no causal ablation — cite to
   preempt the comparison. (Author list unresolved — verify before citing.)
8. **Kohonen 1972/1973; Anderson 1972.** Classical linear associative memory —
   source of the rank-for-exact-recovery fact, so we don't overclaim it (§3).
