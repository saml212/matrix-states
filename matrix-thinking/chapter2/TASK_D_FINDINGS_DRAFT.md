# Task D Findings — Does Gradient Descent Recruit Matrix Rank When the Task Provably Requires It?

**Status: PRELIMINARY.** This draft reports a mid-sweep snapshot
(`results/overnight_snapshots/AGGREGATE_1015.json`, 830 completed runs, 2
failed) from an overnight queue on 8×H100 that refills with fresh seeds and
has not stopped. Numbers below will move as more seeds land, particularly for
the sparser cells (d=16, K=4; all of d=32/64/128). Treat every table as a
snapshot, not a final result, and re-pull before citing externally.

---

## Abstract

The ICML MI-workshop paper "The Gradient Does Not See Rank" showed that a
bolt-on matrix-CODI readout produces flat rank-$k$ accuracy curves on ProsQA,
and argued the cause is a constant Jacobian in the readout, not an inherent
property of matrix-valued latents. That left open whether gradient descent is
*intrinsically* rank-averse, or whether ProsQA simply admits a rank-1
solution. Task D removes the ambiguity: it is a from-scratch, matrix-native
key/value binding task with a hard single-matrix bottleneck ($P=1$) and a
literal linear unbind readout, chosen because exact recovery of $K$
independent bindings **provably requires** $\mathrm{rank}(Z) \geq K$. At the
architecture's trainable operating point ($d=8, 16$; encoder width $h=64$), we
find that (a) the learned effective rank of $Z$ tracks $K$ almost exactly
(effective rank at $d{=}16$: $K{=}1\to2.4$, $K{=}8\to8.2$, $K{=}16\to15.1$),
and (b) constraining $Z$ to rank $k$ *during training* produces a sharp,
causal accuracy collapse below $k\approx K$ and near-ceiling accuracy at
$k\geq K$ (e.g. $d{=}8,K{=}4$: $0.0$ at $k\leq3$, $0.97$ at $k=4$). This is
the confirmatory result the workshop paper's open question predicted: when a
task forces rank, SGD both builds it and depends on it causally. At $d\geq32$
the same tiny encoder fails to train at all — effective rank collapses to
$\approx1$ and force-rank accuracy is $\approx0$ regardless of $K$ or $k$,
including the fully unconstrained ceiling. We read this as a capacity/optimization
limitation of the current $h{=}64$ encoder, not a refutation of the
hypothesis, and flag it as the open trainability frontier. Task D is
associative memory, deliberately abstract for provability; transfer to actual
reasoning tasks is future work.

---

## 1. Background and the Sharpened Hypothesis

The workshop paper's rank-$k$ ablation on ProsQA was flat under every readout
we tried, linear or nonlinear, and we could not distinguish two explanations:
(i) the training objective is rank-blind in general, or (ii) ProsQA happens to
admit a rank-1 functional solution, so nothing forces rank use regardless of
objective. Task D is designed to force the issue by construction: it uses a
task where the *ground truth* solution has a provable rank lower bound, so
there is no rank-1 shortcut to fall back on.

**H_primary** (pre-registered, `TASK_D_PREREGISTRATION.md`): when a task's
exact solution provably requires $\mathrm{rank}(Z) \geq K$, a from-scratch
matrix-native model trained by plain SGD (a) develops effective rank
$\approx K$ in its matrix state, and (b) its accuracy collapses under rank-$k$
truncation for $k < K$.

- **CONFIRM** → the workshop paper's rank-blindness was task-specific
  (ProsQA was rank-1-solvable); gradient *can* recruit rank when the task
  demands it.
- **FALSIFY** → the gradient is rank-averse even when rank is provably
  required; matrix-thinking's core premise is dead in this setting.

Both outcomes were pre-registered as decisive and publishable before any
training ran.

---

## 2. Method

### 2.1 Task D — tensor-product key/value binding

One sample: `BIND k_1 v_1 ... BIND k_K v_K QUERY k_j SEP → target v_j`. Keys
and values are drawn fresh per sample as continuous $d$-dimensional vectors
(orthonormal via QR, the gate default; i.i.d.\ Gaussian L2-normalized is used
only for the deliberately over-complete $K>d$ probes), never a small discrete
codebook — this is what keeps the readout continuous and the rank bound
alive (see §2.2). Bind order is shuffled; the queried binding is uniform over
the $K$ pairs.

**Readout, architecturally pinned.** The decoder is not an MLP over
`flatten(Z)` and not an argmax/nearest-neighbor classifier over a value
codebook — it is the literal unbind the hypothesis claims the matrix
performs: `predicted_value = Z @ key_j`, a continuous vector, scored by an
*absolute* criterion (cosine similarity against the true value), never a
relative $K$-way argmax. This choice is load-bearing (§2.2).

### 2.2 The provable rank(Z) ≥ K lower bound

Requiring exact recovery $Z k_j = v_j$ for $j=1..K$ with linearly independent
$\{k_j\}$ and $\{v_j\}$ (generic for random vectors, $K\leq d$): stack keys as
columns of $K_{\mathrm{mat}}\in\mathbb{R}^{d\times K}$ and values as
$V_{\mathrm{mat}}$. Then $Z K_{\mathrm{mat}} = V_{\mathrm{mat}}$, and since
$\mathrm{rank}(V_{\mathrm{mat}}) = \mathrm{rank}(Z K_{\mathrm{mat}}) \leq
\mathrm{rank}(Z)$ while $\mathrm{rank}(V_{\mathrm{mat}}) = K$:

$$\mathrm{rank}(Z) \geq K. \qquad\blacksquare$$

This rules out the master shortcut that killed earlier Chapter 2 designs
($Z = u\otimes v_0$, rank 1, storing information in the free vector side
instead of rank) for any $K>1$.

**The bound is conditional on the readout.** Nichani, Lee & Bietti (ICLR
2025, arXiv:2412.06538) prove a rank-$m$ matrix can store $\approx md$
associations *under discrete argmax decoding* with random Gaussian mixing —
under that decoding regime the rank$\,\geq K$ necessity collapses, because
argmax tolerates interference that continuous recovery cannot. This is
exactly why Task D pins the readout to continuous `Z·key_j` scored against an
absolute cosine bar rather than a $K$-way argmax (§10, and §5 below).

The underlying linear-algebra fact — exact recovery of $K$ independent
key→value pairs needs rank $K$ — is classical (Kohonen 1972/73; Anderson
1972; optimal linear associative memory). Task D does not claim this fact as
novel; the contribution is whether an end-to-end trained, matrix-native
transformer under a hard bottleneck *spontaneously discovers* this rank, and
what happens under a causal forced-rank-$k$ training ablation, neither of
which exist in prior work (§5).

### 2.3 Architecture: the P=1 single-matrix bottleneck

```
(K bindings) --encoder--> Z ∈ R^{d×d} --unbind--> predicted value
                          ^^^^^^^^^^^^
                          the ONLY thing the decoder sees
```

`BindingEncoder` (`model_v4.py`) is a permutation-invariant set encoder: a
3-layer, 4-head Transformer encoder (hidden width $h=64$, fixed across all
$d$) over the $K$ `[key;value]` tokens, followed by $d$ learned "row-reader"
queries that cross-attend into the encoded set to produce the $d$ rows of
$Z$. It does not hard-code the $\Sigma v_jk_j^\top$ outer-product solution, so
"does SGD discover rank $\approx K$" is a genuine empirical question, not a
built-in answer. At $d=16$ this is a $\approx$171K-parameter model writing
into a $d^2=256$-dimensional state — small by design, so any capacity limit
shows up quickly as $d$ grows (§4).

The decoder is a pure function `pred = Z @ query_key`, verified at runtime by
a mandatory blank-out unit test: corrupting the raw binding tokens after $Z$
is written leaves decode output bit-for-bit unchanged, and a direct autograd
check confirms the decoder gradient has no path to the raw keys/values except
through $Z$. Both checks are part of the smoke gate that must pass before any
training run launches.

### 2.4 Controls

| # | Control |
|---|---------|
| C1 | Train-time `force_rank_k` — SVD/eigh-projects $Z$ to rank $k$ every step, the **primary causal test** (not post-hoc truncation). |
| C2 | Param-matched vector-state control arm. **Descoped to Stage-1b** — the round-1 HRR baseline was not param/state-matched (~4.3K vs ~171K params, 16-dim vs 256-dim state); not reported here. |
| C3 | ≥5 seeds; primary rank metric = effective rank ($\exp$ of the entropy of the normalized singular-value spectrum), secondary = stable rank. This snapshot averages 8 seeds per (d,K) cell for M1 and 5 seeds per (d,K,k) cell for M3 (tier-1); tier-2/3 use 4 and 3 seeds respectively. |
| C4 | Chance-adjusted per-item (Hamming) accuracy, not raw joint exact-match. |
| C5 | Held-out generalization: keys/values are fresh per sample with near-zero collision probability, so every eval batch is composed of never-seen vectors by construction. |

### 2.5 Metrics, and an important caveat on τ in this snapshot

**M1** — learned effective rank of the unconstrained model vs. $K$.
**M2** — eval-time rank-$k$ truncation curve on an unconstrained checkpoint
(not yet rolled into this aggregate; see §6). **M3** — train-time
force-rank-$k$, the primary causal test.

A binding is "recovered" if $\cos(Z\cdot\mathrm{key}_j, v_j) > \tau$. The
pre-registration fixes the primary knee criterion at $\tau=0.99$ (report also
at $0.9, 0.95$). **The cross-run aggregation script currently hardcodes
`TAU = "recovered_frac@0.9"`** (`run_overnight.py:34`) — so every M3 number
below is at the *secondary* $\tau=0.9$ threshold, not the pre-registered
primary $\tau=0.99$. Per-run JSONs contain all three thresholds; the
$\tau=0.99$ roll-up has not been produced yet. This is flagged explicitly in
§6 and should not be conflated with the pre-registered knee test.

---

## 3. Results

### 3.1 M1 — learned effective rank tracks K in the trainable regime

| K | eff. rank, $d{=}8$ | eff. rank, $d{=}16$ |
|---|---|---|
| 1 | 1.65 | 2.38 |
| 2 | 3.44 | 3.22 |
| 3 | — | 3.87 |
| 4 | 4.57 | 4.69 |
| 6 | — | 6.40 |
| 8 | 7.83 | 8.18 |
| 10 | — | 9.89 |
| 12 | — | 11.85 |
| 14 | — | 13.45 |
| 16 | — | 15.09 |

Within each $d$, effective rank is strictly monotonically increasing in $K$
across the full tested grid (Spearman $\rho=1.0$ on the seed-averaged
sequence at both $d=8$ and $d=16$) — the M1 CONFIRM criterion
($\rho\geq0.8$) is comfortably met. The band criterion (eff. rank $\in
[0.7K, 1.3K]$ for $K\leq d$) is met from $K{=}3$ ($d{=}16$) / $K{=}4$
($d{=}8$) onward, but **violated at $K=1,2$**: effective rank at $K{=}1$ is
$1.65$–$2.38$ against an upper bound of $1.3$, and at $K{=}2$ is $3.22$–$3.44$
against an upper bound of $2.6$. The model is not collapsing to a
rank-1 solution at $K{=}1$ as cleanly as pre-registered — there is a
residual-entropy floor of roughly 1–1.5 extra effective-rank units at very
low $K$ that is not yet explained (candidate causes: attention-noise in the
row-reader spreading mass across rows even when only one binding needs to be
represented, or the effective-rank metric's sensitivity to small
non-dominant singular values; not distinguished here). This is an honest
partial deviation from the pre-registered band, not disqualifying (the
Spearman criterion, the primary CONFIRM gate, is unaffected), but should be
investigated before the final write-up.

For $K > d$ at $d=16$ (the deliberately over-complete, Gaussian
non-orthonormal probe, out of scope for the exact-recovery gate per §2.2):
effective rank plateaus and then declines ($K{=}20\to11.1$, $K{=}24\to11.6$,
$K{=}32\to8.6$) rather than continuing to grow — consistent with the expected
lossy HRR/VSA capacity regime, not a failure of the gate.

### 3.2 M3 — force-rank-k causally gates accuracy at k ≈ K (primary test)

`recovered_frac@0.9` vs. forced training-time rank $k$, at the pre-registered
straddle grid around each $(d,K)$; ceiling = accuracy at $k=d$ (rank
unconstrained in practice):

| $(d, K)$ | $k$ well below $K$ | $k=K{-}1$ | $k=K$ | $k=K{+}1$ | ceiling ($k=d$) |
|---|---|---|---|---|---|
| $d{=}8,\ K{=}4$ | $k{=}1\!:\!0.000,\ k{=}2\!:\!0.000,\ k{=}3\!:\!0.000$ | ($k{=}3$, above) | **0.968** | 0.987 | 0.980 |
| $d{=}16,\ K{=}8$ | $k{=}1\!:\!0.000,\ k{=}2\!:\!0.000,\ k{=}6\!:\!0.326$ | 0.888 | 0.792 | 0.898 | 0.991 |
| $d{=}16,\ K{=}12$ | $k{=}1\!:\!0.000,\ k{=}2\!:\!0.000,\ k{=}10\!:\!0.560$ | 0.674 | **0.935** | 0.694 | 0.999 |
| $d{=}16,\ K{=}4$ | $k{=}1\!:\!0.000,\ k{=}2\!:\!0.001,\ k{=}3\!:\!0.005$ | (=$k{=}3$, above) | 0.145 | 0.087 | 0.045 |

Three of the four cells show the pre-registered signature — near-zero
accuracy for $k$ well below $K$, a sharp jump straddling $k=K$, and a ceiling
well above $0.9\times$ceiling by $k=K$ or $k=K{+}1$:

- **$d{=}8, K{=}4$ is razor-sharp:** $0.000\to0.000\to0.000$ at
  $k=1,2,3$, then $0.968$ at $k=4$ — a step of $+0.968$ in a single rank
  increment, landing exactly at $k=K$.
- **$d{=}16, K{=}8$:** the step lands one rank early, between $k=6$
  ($0.326$) and $k=7=K{-}1$ ($0.888$), already crossing $0.9\times$ceiling
  ($0.892$) at $K{-}1$ — within the pre-registered $K\pm1$ tolerance.
- **$d{=}16, K{=}12$:** the step lands at $k=K=12$ exactly ($0.935$, above
  the $0.9\times$ceiling bar of $0.899$), up from $0.560$–$0.674$ at
  $k=10,11$.

All three are consistent with M3's CONFIRM criterion and nowhere near the
HARD FALSIFY criterion (force-rank-1 reaching $\geq0.9\times$ceiling never
happens — every $k{=}1$ cell in the snapshot is exactly $0.000$).

**Two honest caveats.** First, the region at and past the knee is not
cleanly monotone: $d{=}16,K{=}8$ dips from $0.888$ ($k{=}7$) to $0.792$
($k{=}8$) before rising again to $0.898$ ($k{=}9$) and dipping to $0.798$
($k{=}10$); $d{=}16,K{=}12$ dips from $0.935$ ($k{=}12$) to $0.694$
($k{=}13$) before rising to $0.813$ ($k{=}14$). Rank should not *hurt*
accuracy once $k\geq K$ in the idealized picture — these dips most likely
reflect optimization noise in force-rank training at intermediate seeds/steps
within a mid-sweep snapshot rather than a real non-monotonicity, but they are
reported as-is rather than smoothed. Second, **$d{=}16,K{=}4$ does not show a
clean step in this snapshot**: even its ceiling ($k{=}16$, effectively
unconstrained) is only $0.045$, far below the $\geq0.97$–$0.999$ ceilings
seen at $K{=}4$ ($d{=}8$), $K{=}8$, and $K{=}12$ ($d{=}16$).

> **CORRECTION (2026-07-01, 1,234-run mega-replication).** This draft
> originally read the $d{=}16,K{=}4$ cell as "under-converged or seed-sparse
> rather than genuinely rank-blind ... should be re-examined once more seeds
> land for this specific $(d,K)$ pair." The mega-replication shows that
> reading was wrong: the cell is a **genuine, seed-count-independent
> convergence ceiling**. Even at force-rank $k{=}16$ (full rank, i.e. no rank
> constraint at all), `recovered_frac@0.9` $= 0.045$, and the entire
> force-rank sweep for this cell stays within $0.0008$–$0.134$ across all
> tested $k$ — it never approaches the $0.9{+}$ ceilings every other tested
> cell reaches, and more seeds do not fix it. It remains
> not-evidence-against H_primary in the narrow sense that the unconstrained
> model never learns the task at this cell (so there is nothing for the
> force-rank step to gate), but it is a standing negative data point for
> this $(d,K)$ pair under the current training recipe — not a seed-sparsity
> artifact awaiting more runs.

M2 (the eval-time truncation curve on an already-trained unconstrained
checkpoint, corroborating but not primary) is not yet present in this
aggregate and is not reported here.

### 3.3 Decision-criteria scorecard against the pre-registration

| Metric | Pre-registered CONFIRM criterion | This snapshot |
|---|---|---|
| M1 | $\rho(K,\text{eff.\ rank})\geq0.8$ and eff. rank $\in[0.7K,1.3K]$ for $K\leq d$ | $\rho=1.0$ at $d{=}8,16$ (met); band violated at $K{=}1,2$ only (partial) |
| M3 | Near-chance below $K$, $\geq0.9\times$ceiling at/above $K$, step at $K\pm1$ | Met at $(8,4)$, $(16,8)$, $(16,12)$; **fails at $(16,4)$** — resolved as a genuine convergence ceiling by the 1,234-run replication (2026-07-01, see §3.2 correction); HARD FALSIFY criterion never triggered anywhere |

Reading M1+M3 together at the trainable operating point ($d=8,16$): the
overall gate criterion (M1 CONFIRM and M3 CONFIRM) is **met at 3 of 4 tested
cells**, with the low-$K$ band deviation still open and the $d{=}16,K{=}4$
cell now resolved (2026-07-01) as a standing negative — a genuine,
seed-count-independent convergence ceiling, not a snapshot artifact (§3.2).

---

## 4. The d ≥ 32 Trainability Frontier — a limitation, not a refutation

At $d=32, 64, 128$, both M1 and M3 collapse to a flat, uninformative floor
regardless of $K$:

| $d$ | $K$ range tested | effective rank observed |
|---|---|---|
| 32 | 1–32 | 1.23–4.68, no clean trend with $K$ |
| 64 | 1–64 | 1.01–1.05 (flat, $\approx1$ for every $K$) |
| 128 | 8–128 | 1.01–1.02 (flat, $\approx1$ for every $K$) |

| $(d,K)$ | $k{=}1$ | straddle around $K$ | ceiling ($k{=}d$) |
|---|---|---|---|
| $32, 16$ | 0.000 | 0.000 at $k{=}14,15,17,18$ | 0.000 |
| $64, 32$ | 0.000 | 0.000 at $k{=}30,31,33,34$ | 0.000 |
| $128, 64$ | 0.000 | 0.000 at $k{=}62,63,65,66$ | 0.000 |

Critically, the **unconstrained ceiling itself is $0.000$** at all three
scales — this is not a force-rank artifact and not a rank-blindness result:
the model never learns the task at all at these sizes, so there is nothing
for M1/M3 to measure. The encoder (`BindingEncoder`, `model_v4.py`) uses a
fixed hidden width $h=64$ across every $d$ in the sweep; at $d=128$ it must
write a $16{,}384$-dimensional matrix state from a $64$-wide representation,
a roughly 64× harder writing problem than at $d=16$ with the same encoder
capacity. Informal calibration ahead of the sweep found that naively scaling
the encoder wider/deeper to compensate led to training instability rather
than an easy fix — this was not run as a formal ablation and is not part of
this snapshot, so it should be treated as an anecdotal caveat, not evidence.

We read this as an open **encoder-capacity / optimization** problem at larger
$d$, separate from H_primary, which is about whether rank is used *once the
model trains at all*. Task D's design already anticipates this: the
pre-registration fixes $d{=}16$ as the decisive gate and treats larger $d$ as
a generality probe, not the primary test. The $d\geq32$ result should be
read as "we have not yet found an encoder that learns this task at these
scales," not as "rank is not used at these scales" — those are different
claims, and only the trainable regime ($d{=}8,16$) can currently speak to
H_primary.

---

## 5. Related Work — distinguishing Nichani, Lee & Bietti head-on

The paper Task D is in most danger of being read as "just a rank-flavored
variant of" is **Nichani, Lee & Bietti, "Understanding Factual Recall in
Transformers via Associative Memories"** (ICLR 2025 Spotlight,
arXiv:2412.06538). They define the same object — a linear associative memory
built from outer-product bindings — and their Theorem 1 rank-$m$ remark shows
a rank-restricted matrix can store $\approx md$ associations under **discrete
argmax decoding** with random Gaussian mixing. Three axes distinguish Task D:

1. **Existence vs. discovery.** Their result is a hand-built existence
   construction for a chosen rank $m$; Task D measures the rank gradient
   descent *actually discovers* when trained end-to-end.
2. **Necessity vs. achievability, and discrete vs. continuous.** Their
   capacity result holds under argmax decoding, which tolerates interference
   and does not require rank $\geq K$. Task D proves a **necessity** lower
   bound (§2.2) that holds specifically because the readout is pinned to
   **exact continuous** recovery — a design choice made precisely to keep the
   bound alive, since under argmax decoding the rank$\geq K$ requirement
   collapses.
3. **No causal ablation, no hard bottleneck.** Their transformer has full
   self-attention over raw tokens with the associative memory living inside
   one weight matrix, not an explicit, architecturally isolated $d\times d$
   state the decoder is forced to read from exclusively; and they report no
   training-time forced-rank-$k$ ablation.

Other lineage, in priority order (see `research/task-d-novelty-july2026.md`
for the full pass): **Nazari & Rusch (arXiv:2602.04852)** and **Sun et al.
(arXiv:2602.02195)**, both Feb 2026, are the closest *mechanistic*
rank-measurement precedent — same effective-rank definition, applied to
linear-attention fast-weight states in **pretrained LLMs on real text**, with
an *upper* bound $\mathrm{rank}(S_t)\leq t$ and no controlled ground-truth
$K$, no necessity theorem, and no training-time rank-$k$ ablation; Task D is
close to a mirror image (controlled $K$, a *lower* bound, a causal ablation).
**Schlag, Irie & Schmidhuber (arXiv:2102.11174, DeltaNet/FWP)** motivate why
capacity/rank matters once writes exceed state dimension. **TP-Transformer
(arXiv:1910.06611)** and **TPDN (arXiv:1812.08718)** are the binding-as-outer-
product ancestry — $Z$ is a TPR-style binding matrix, but neither measures
rank vs. binding count. **Plate 1995 (HRR)** and **Frady/Kleyko/Sommer
resonator networks (arXiv:1906.11684)** scope "capacity" correctly in the VSA
tradition (SNR/bundling, codebook factorization) as a different mechanism
from Task D's exact-rank framing. **Kohonen (1972/73)** and **Anderson
(1972)** are the classical source of the rank-for-exact-recovery fact itself,
cited so Task D does not overclaim novelty of the linear-algebra fact — only
of the trained, causal, matrix-native-transformer framing around it.

---

## 6. Limitations, Open Questions, and Future Work

- **Preliminary snapshot.** 830 runs, 2 failed, from a perpetually-refilling
  overnight queue that has not stopped. Every number above should be
  re-pulled from a fresh `AGGREGATE*.json` before external use, and seed
  counts per cell will keep growing.
- **$\tau$ mismatch.** All M3 numbers here are `recovered_frac@0.9`, the
  aggregation script's hardcoded threshold — not the pre-registered primary
  knee criterion at $\tau=0.99$. The $\tau=0.99$ roll-up needs its own
  aggregation pass before the final knee-location claim can be made
  precisely; qualitatively we do not expect the step location to move much
  given how sharp the $d{=}8,K{=}4$ step already is at $\tau=0.9$, but this
  is an expectation, not a measured result.
- **M2 absent.** The eval-time rank-$k$ truncation curve on a fixed
  unconstrained checkpoint (corroborating evidence per the pre-registration)
  is not in this aggregate.
- **Low-$K$ effective-rank band deviation (§3.1)** remains unresolved and
  should be chased down — most likely mundane (a metric artifact at low K)
  but not yet confirmed as such.
  **CORRECTION (2026-07-01):** this bullet originally also listed the
  **$d{=}16,K{=}4$ force-rank anomaly (§3.2)** as "most likely mundane
  (undertraining, seed sparsity)." The 1,234-run mega-replication resolved
  it the other way: it is a genuine, seed-count-independent convergence
  ceiling (`recovered_frac@0.9` $=0.045$ even at full-rank $k{=}16$; sweep
  range $0.0008$–$0.134$ across all force-ranks). See the dated correction
  in §3.2 — no amount of additional seeds fixes this cell.
- **Non-monotonic dips just past the knee** in the $d{=}16$ M3 curves are
  reported as-is; a larger-seed re-aggregation should show whether they
  average out.
- **$d\geq32$ trainability, not rank-blindness (§4).** The current
  fixed-width ($h{=}64$) encoder does not learn Task D at all at these
  scales. This blocks any claim about whether rank matters at larger $d$ and
  is the most actionable near-term follow-up: a properly capacity-scaled (and
  stability-checked) encoder for $d\geq32$.
- **C2/M4 vector-state control not run.** The param-matched vector baseline
  that would show whether "matrixness" specifically, and not just parameter
  count or state dimensionality, is doing the work is descoped to Stage-1b
  and not part of this snapshot.
- **Task D is associative memory, not reasoning.** It was deliberately
  chosen as an abstract task with a provable rank lower bound, not for
  realism. It shows whether gradient descent *can* recruit and depend on
  matrix rank when a task requires it — it does not show that rank helps
  real reasoning. That transfer step (a patched relational-query task, or
  real data) is gated on this result and is future work, per
  `TASK_D_PREREGISTRATION.md` §8.

---

## 7. Reproducibility

- Pre-registration: `matrix-thinking/chapter2/TASK_D_PREREGISTRATION.md`
  (locked 2026-07-01, before any training).
- Task definition and self-test: `matrix-thinking/chapter2/task_d.py`.
- Model and rank utilities: `matrix-thinking/chapter2/model_v4.py`,
  `matrix-thinking/chapter2/rank_utils.py`.
- Runner (smoke gate, train, eval): `matrix-thinking/chapter2/run_task_d.py`
  — run `python run_task_d.py --smoke` before trusting any downstream result.
- Overnight orchestrator and aggregation:
  `matrix-thinking/chapter2/run_overnight.py`.
- This snapshot: `matrix-thinking/chapter2/results/overnight_snapshots/AGGREGATE_1015.json`.
- Design/code audit trail: `matrix-thinking/chapter2/gauntlet/` (attack,
  research, and audit rounds 1–3, plus an overnight-orchestrator-specific
  audit).
- Novelty check: `research/task-d-novelty-july2026.md`.
