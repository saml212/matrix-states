# CONSOLIDATION ATTACK — ROUND 2

Fresh Opus agent, 2026-08-12. Charter = §7 of
`CONSOLIDATION_POLICY_WATERFALL.md` at commit `65e7749`.
Target = §6 DRAFT-R1 (the falsifier-backwards re-derivation).

**VERDICT: BLOCKED.** 5 FATAL / 6 MAJOR / 5 minor.
The lane's sole surviving regime (ii) does not survive round 2.

All demonstrations executed in numpy on this machine, zero GPU-h. R2 scripts
live in the session scratchpad at stable `r2_*` filenames (inventory §R2.9),
each re-runnable via `DRY_RUN_BYPASS=1 python3 <file>.py`.

---

## §R2.0 Executive summary

DRAFT-R1's surviving claim is that a usage-gated compressive delta write beats
a byte-matched index-coded exact-KV store on dense aggregate queries under
overload, by **+0.230** at the registered cell (d=32, M=128, ρ=0.7, gated p=2),
"20× seed noise", with a "provable √(C/M) backbone".

Round 2 finds the margin is manufactured by a **numerical-conditioning defect in
the falsifier's readout**, and that the registered *policy* — the usage gate,
which is the lane's entire subject — has **no measurable effect on the registered
decision statistic at all**.

| what §6 registers | what R2 measures | ref |
|---|---|---|
| margin +0.230 vs steelman A7 | +0.227 reproduced — but A7's readout is min-norm `lstsq` at C=31, d=32, sitting on the Marchenko–Pastur hard edge. Free ridge (λ tuned **offline**, no oracle) lifts A7 0.0610 → **0.2216**; margin collapses to **+0.066**, below the registered 0.15 bar | R2-A |
| gate is "the same experiment" as the win | gated − shuffled = **+0.0068** (bar 0.013); ρ-sweep **flat** (+0.2299 at ρ=0 vs +0.2234 at ρ=1); a **constant** η at the gate's mean scores **0.4088 vs the gate's 0.2880** — the gate *costs* 0.121 | R2-B |
| fp16 risk "collapses to +0.005–0.05" | registered arm vs fp16-A7-ridge = **−0.035** (a loss); vs int8-A7 = **−0.193** | R2-C |
| "provable √(C/M) backbone" | bound = 0.4921 at the registered cell; the **matrix's own ceiling is 0.4911**. The bound sits *above* the arm it is meant to support. `C = d − 1` exactly, for every d tested | R2-D |
| dense random combinations as the query family | under **usage-weighted sparse** queries — the only workload where a usage policy can matter — **A7 wins**: k=1 → −0.149, k=2 → −0.065, k=4 → −0.031 | R2-E |

**The design contains an internal contradiction.** The query family that makes
the matrix win (dense, uniform, usage-*independent*) is exactly the family that
makes the usage gate pointless — which is why the ρ-sweep is flat. The query
family that gives the gate a job (usage-correlated, sparse) is exactly the family
where the cache wins. There is no cell where both the margin and the policy are
alive.

**M1 (the load-bearing item) is answered in §R2.2: NO** at the registered cell
and registered arm — with a precise account of what the symmetry lever really is.

---

## §R2.1 Charter item 1 — reproduction of §6's figures

All seven saved scripts run clean and reproduce their §6 figures **exactly**.

| §6 claim | script | reproduced | status |
|---|---|---|---|
| capacity table: naive 16 / index-fp32 31.78 / index-fp16 63.14 at d=32,M=128 | `r1_steelman_a7.py` | identical | ✅ |
| chance-floor cos 0.015–0.02 across d∈{16,32,64,128} | `r1_steelman_a7.py` | 0.0193/0.0152/0.0148/0.0152 | ✅ |
| regime (i): 4/192 cells clear | `r1_regime1_overload.py` | `4 / 192 cells clear` | ✅ |
| regime (i) best cell A7=0.254, matrix=0.339, margin +0.085 | `r1_regime1_overload.py` | 0.2537 / 0.3390 / +0.0853 | ✅ |
| regime (i-b): 1/48 cells; M=128,s=0,ρ=1 N-way +0.057, off-target −0.40 | `r1_regime1b_discriminative.py` | `1 / 48`; +0.0566 / −0.4047 | ✅ |
| regime (ii) frontier table, all 6 rows | `r1_regime2_aggregate.py` | identical to 4 dp | ✅ |
| regime (ii) seed noise 2×SEM = 0.0148 at n=20 | `r1_regime2_aggregate.py` | 0.0148 | ✅ |
| regime (iii): 0/45 cells; collapse T=200 P=0.99 → T=500 P=0.001 | `r1_regime3_correction.py` | `0/45`; 0.9905 → 0.0010 | ✅ |
| L2 coherence: M=64 +0.316→+0.009; M=128 +0.180→+0.010 | `r1_l2_coherence.py` | +0.3157→+0.0092; +0.1802→+0.0095 | ✅ |
| L3 info-free reader 0.011 vs matrix 0.205 at zero collapse | `r1_l3_l4_l6_discharge.py` | 0.0113 / 0.2049 | ✅ |
| L4 A4−A5 = +0.365 / +0.197 / +0.168 / −0.038 | `r1_l3_l4_l6_discharge.py` | identical | ✅ |
| L6 matrix 397,312 write FLOPs, 12.5× A7 | `r1_l3_l4_l6_discharge.py` | identical | ✅ |
| §6.1 entropy-coding ceiling buys A7 <1 slot (31.83 vs 31.78) | `r1_steelman_a7.cap_entropy` | confirmed | ✅ |

**The one exception — the registered headline cell itself.** §6.9 states the
gated cell and the fp16 sensitivity were *"single parametrized calls into
`r1_regime2_aggregate.py`'s `run_cell`"*. **They cannot have been.** That
function is `run_cell(d, M, n_queries, n_seeds)` with `eta=1.0` hardcoded at
line 56 and **no `rho`, no gate, no `val_bits` argument anywhere in the module**.
No script in the §6.9 inventory computes a *gated* aggregate cell — the only
gate implementation in the whole inventory (`r1_l3_l4_l6_discharge.gate`) is
applied to regime (i)'s per-item metric, never to the aggregate metric. So the
registered headline figure is **not reproducible from the stable scripts**
(finding R2-F).

I therefore rebuilt the cell from the §6.5 spec (`r2_repro_gated.py`):

```
  d=32, M=128, rho=0.7, gate = percentile_rank(u)^2, A7 admission = greedy-by-observed-usage
  n=20 seeds:  A7-lstsq 0.0610 +- 0.0051   MAT-gated 0.2880 +- 0.0042   margin +0.2270
  n=50 seeds:  A7-lstsq 0.0612 +- 0.0036   MAT-gated 0.2884 +- 0.0030   margin +0.2272
  §6.3 claims: A7      0.059  +- 0.005     MATRIX    0.289  +- 0.003     margin +0.230
```

**The figure itself is substantially correct** (+0.227 vs +0.230, well inside the
reconstruction's spec ambiguity). This is a provenance/bookkeeping failure, not a
fabrication, and I class it MAJOR rather than FATAL — flagging that under a
literal reading of the charter ("any figure that does not reproduce is a FATAL")
the coordinator may class it higher. The substantive kill comes from R2-A/B/C/E,
which apply to the correctly-valued figure.

---

## §R2.2 M1 — THE PRECISION-SYMMETRY ADJUDICATION

> §7: *"if A7 may halve value precision to double slots (31→63), then S may halve
> element precision to double its state… If the matrix STILL loses to fp16-A7
> when it is itself allowed fp16, the regime dies honestly."*

### R2.2a The amendment's premise is half wrong — and I fixed it in the matrix's favour

**"Double state" as more operator buys the matrix exactly zero.** For the
registered query family the Bayes-optimal read is *already* a d×d linear map:

> K, V fixed; a ~ Unif(S^{M−1}); q = Kᵀa, t = Vᵀa. Conditional on q, a is uniform
> on the affine slice {a : Kᵀa = q} of the sphere, whose mean is the min-norm
> solution K(KᵀK)⁻¹q. Hence **E[t | q] = Vᵀ K (KᵀK)⁻¹ q — exactly linear in q.**

So no 2-head split, no lifted dimension, no nonlinear feature map can beat a
single d×d operator; they can only add quantization noise. Measured at M=128
(`r2_precision_symmetry.py`):

```
  fp16 2-head (2 x d x d, distinct random key projections)  0.4909
  fp16 lifted d'=45 (2*45^2 B <= budget)                    0.4819
  fp16 nonlinear random features (d x 2d, ReLU)             0.1978   <- much worse
  fp32 single-operator ceiling (batch ridge over all M)      0.4911
```

**But there IS a legitimate symmetric lever, and I granted it.** Spend the
doubled fp16 budget on *second-order statistics* instead of more operator:
streaming RLS keeps S (operator) **and** P = (KᵀK+λI)⁻¹ (inverse Gram), both
d×d at fp16 = exactly the fp32 single-operator budget, updated per item by
Sherman–Morrison. It is genuinely single-pass and fixed-memory — a legal
consolidation writer — and it converges to the batch-ridge optimum
(`r2_symmetric_rls.py`):

```
  M=128:  registered gated delta 0.2880  ->  const-eta delta 0.4088
                                         ->  fp16 streaming RLS 0.4773   (ceiling 0.4911)
```

So the honest statement is **not** "extra bytes buy the matrix nothing". It is:
extra bytes buy the matrix a **better estimator** (0.288 → 0.477) but **cannot
raise its ceiling** (√(d/M)), whereas the same bytes raise A7's slot count C
without bound.

### R2.2b fp16 accumulation drift — the symmetric arm is real, with a stated horizon

Relative drift ‖S₁₆ − S₆₄‖/‖S₆₄‖ over stream length T:

| T | plain delta rule | streaming RLS |
|---|---|---|
| 128 | 1.36e-3 | 2.02e-3 |
| 512 | 1.56e-3 | 5.24e-3 |
| 2048 | 1.51e-3 | 2.05e-2 |
| 8192 | 1.56e-3 | **1.15e-1** |

The **delta rule is self-correcting** — each write recomputes the residual
against the current state, so quantization error is absorbed on the next touch;
drift saturates at ~1.6e-3 and does **not** grow with stream length. **RLS is
not** self-correcting and degrades to 11.5% at T=8192. Verdict: the symmetric
arm is **not fake** at the registered M=128 (drift 0.2%), but beyond T≈2000 it
needs error feedback or periodic fp32 refresh — which must be stated, not
assumed, if it is ever built.

### R2.2c The symmetric frontier (both arms optimized at 4096 fixed bytes)

λ tuned **offline** on synthetic replicates A7 generates from public knowledge
(d, M, C and the iid-Gaussian model) — never touching the evaluation queries.
Offline and oracle λ agree to 4 decimal places at every cell, so the ridge
steelman is not an oracle-tuning artifact (`r2_precision_frontier.py`).

**d=32, M=128 (the registered cell).** Registered bar: margin ≥ +0.15 AND ≥ 2×SEM.

| A7 value bits | C | A7 (offline λ) | margin, **registered arm** (gated delta 0.2880) | margin, **best matrix** (0.4911) |
|---|---|---|---|---|
| 32 | 31 | 0.2215 | **+0.066 ✗** | +0.270 ✓ |
| 16 | 63 | 0.3230 | **−0.035 ✗** | +0.168 ✓ |
| 12 | 83 | 0.3731 | **−0.085 ✗** | +0.117 ✗ |
| 10 | 100 | 0.4190 | **−0.131 ✗** | +0.072 ✗ |
| 8 | 124 | 0.4812 | **−0.193 ✗** | +0.010 ✗ |
| 6 | 128 | 0.4909 | −0.203 ✗ | +0.000 ✗ |

At **M=64** the best matrix already fails at fp16 (+0.011). At **M=256** it
survives to 12 bits (+0.157) and dies at 8 (+0.113).

### R2.2d M1 VERDICT — **NO**

**The registered arm loses under symmetric precision at every precision,
including fp32.** The registered gated delta write clears the +0.15 bar against
*nothing* once A7 is allowed a regularized readout.

The margin survives symmetric fp16 **only** under a conjunction of three
stipulations:

1. A7 is stipulated to ≥16-bit values (a definitional escape — and precisely the
   asymmetry §7 convened this adjudication to remove; nothing about int8 values
   with a per-vector scale is unfaithful under a cosine metric, and it takes A7
   to C=124 of M=128);
2. M/d ≥ 4;
3. the writer is swapped from the usage-gated delta write to **fp16 streaming
   RLS**, which contains **no gate at all**.

Stipulation (3) removes the lane's subject matter. What survives is "a streaming
compressive regression operator beats a byte-matched exact store on dense
aggregate queries when the store is slot-limited" — classical sketching /
streaming least-squares, with no consolidation policy in it.

---

## §R2.3 FATAL findings

### R2-A [FATAL] — the headline margin is an A7 conditioning artifact, not an information bound

`r2_a7_readout.py`. DRAFT-R1 gives A7 `np.linalg.lstsq(K_st, V_st, rcond=None)`
— the **minimum-norm interpolant** of its C=31 stored pairs in d=32. K_st is a
31×32 Gaussian: its smallest singular value sits at the Marchenko–Pastur hard
edge, √d(1−√(C/d)) ≈ 0.09, so `pinv(K_st)` carries a singular value ≈11×.
Applied to the 97 **unstored** items' contribution to a dense query, that emits a
hallucinated output ~11× oversized, which destroys the cosine. A7 scores 0.068
against its own √(C/M) = 0.492 — the gap is numerical, not informational.

Ridge is free, standard, costs zero extra bytes, and λ is tunable offline:

| M | C | A7-lstsq (§6's arm) | A7-ridge | A7-tSVD | MATRIX (ungated) | margin lstsq | **margin ridge** |
|---|---|---|---|---|---|---|---|
| 48 | 31 | 0.2246 | 0.5531 | 0.5255 | 0.5379 | +0.3133 | **−0.0152** |
| 64 | 31 | 0.1402 | 0.4183 | 0.3894 | 0.4395 | +0.2993 | **+0.0212** |
| 96 | 31 | 0.0675 | 0.2895 | 0.2686 | 0.3170 | +0.2495 | **+0.0275** |
| 128 | 31 | 0.0679 | 0.2182 | 0.2025 | 0.2308 | +0.1629 | **+0.0126** |
| 160 | 31 | 0.0525 | 0.1784 | 0.1659 | 0.1956 | +0.1431 | **+0.0172** |
| 256 | 31 | 0.0269 | 0.1100 | 0.1054 | 0.1198 | +0.0929 | **+0.0098** |

**Every cell of §6.3's headline table falls from "6–20× headroom" to inside or
barely outside seed noise, and none clears the registered +0.15 bar.** At the
registered gated cell the margin goes +0.227 → **+0.066**.

R1's M10 explicitly demanded: *"Pre-register A7's readout family explicitly and
argue why it is the **strongest** fair form."* §6.7 dispositions M10
**FIXED-BY-CONSTRUCTION**. It was not: the chosen readout is demonstrably not the
strongest fair form, and the entire headline margin is the difference.

### R2-B [FATAL] — the usage channel is null on the registered statistic, and the registered gate is strictly worse than a constant

`r2_repro_gated.py`, at the registered cell.

```
  MAT-gated  0.2880      MAT-shuffled 0.2812      gated - shuffled = +0.0068  (bar 0.0132)
  MAT-const  0.4088  <-- constant eta at the gate's mean BEATS the gate by +0.1208
  rho sweep:   rho=0.00  margin +0.2299      rho=0.70  margin +0.2270
               rho=0.40  margin +0.2258      rho=1.00  margin +0.2234
```

Three independent readings, one conclusion:

1. **Shuffling the gate across items changes nothing** (+0.007, inside the noise
   bar) — the usage→item assignment carries no signal on this metric.
2. **The margin is flat in ρ, and very slightly *decreasing***: it is marginally
   *larger* when the usage trace is pure noise (ρ=0) than when it is a perfect
   oracle (ρ=1). The registered ρ=0.7 is neither defensible nor cherry-picked —
   it is **inert**.
3. **A constant step size beats the gate by 0.121.** The gate's only effect on
   this metric is a step-size effect (η<1 turns a catastrophically-forgetting
   delta rule into an averaging one), and it is a *worse* step-size schedule than
   the trivial constant control.

§6.3 claims *"the registered novel contribution (the usage gate) and the
headline-win construction are the SAME experiment, not two decoupled ones."*
The opposite is true: they are perfectly decoupled, and the gate is a net
negative. §6.7 dispositions **F4 FIXED-BY-CONSTRUCTION** and **M5
FIXED-BY-CONSTRUCTION** on the strength of §6.6's L4 trace — but that trace was
run at d=32, **M=64, on regime (i)'s usage-weighted per-item cosine**, i.e. the
metric of a *killed* regime. The surviving regime's registered decision statistic
never received a usage-channel attribution test. It fails one.

### R2-C [FATAL] — regime (ii) does not survive symmetric precision

Full derivation in §R2.2. Registered arm: **−0.035 vs fp16-A7**, **−0.193 vs
int8-A7**, and only **+0.066 vs fp32-A7** — failing the registered bar at every
precision. §6.8 risk 1's own estimate ("collapses to +0.005–0.05") is optimistic:
it holds only against the *unregularized* fp16 A7 (I measure +0.0076); against a
regularized fp16 A7 the registered arm **loses**.

### R2-D [FATAL] — the √(C/M) bound is valid but vacuous as a separator, and §7 makes it load-bearing

The inequality itself is fine: for any f of A7's stored bytes,
E[cos(f(q), t)] ≤ √(C/M), and A7's achieved score sits below it at every M.
The defect is its **use**.

Both arms read through a **d×d linear map**, so both are capped by the same Bayes
ceiling ≈ √(d/M). Measured at the registered cell: bound √(C/M) = **0.4921**;
matrix ceiling = **0.4911**. *The bound on the falsifier lies above the arm it is
invoked to support*, by 0.001. Proving A7 ≤ 0.4921 establishes nothing whatsoever
about A7 < matrix.

The reason is an arithmetic accident, verified for every d (`r2_d_sweep.py`):

```
  C_fp32-index(d, N) = floor(d^2 / (d + ceil(log2 N)/32)) = d - 1   EXACTLY
  d= 16  C=15 (d-1)      d= 32  C=31 (d-1)
  d= 64  C=63 (d-1)      d=128  C=127 (d-1)
```

fp32 index coding parks the registered falsifier permanently **one slot below**
the crossover C = d, where an exact store stops being slot-limited. §7 elevates
this to *"a provable-by-construction backbone, which is the program's house
style."* It is not a backbone; it is a tautology at the chosen precision.

Deeper consequence for the lane's purpose: since the optimal answer to the
registered query family is a d×d linear operator that **both** arms can
represent, this task family cannot exhibit *capability separation* even in
principle. The only difference available is how many of the M pairs each arm gets
to fit the same operator on — a sample-efficiency difference. Under the repo's
capability-first directive that is a framing defect, not just a measurement one.

### R2-E [FATAL] — external validity: the margin inverts under the only workload where the policy could matter

`r2_external_validity.py`, Part A. Queries with exactly k nonzero coefficients.

**A1 — uniform support** (k items chosen uniformly). Registered arm vs ridge-A7:

```
  k=1 +0.032   k=2 +0.069   k=4 +0.072   k=8 +0.067   k=16 +0.069   k=128 +0.064
```

Flat and far below the +0.15 bar at every sparsity — the dense family is not
load-bearing *by itself*, R2-A is.

**A2 — usage-weighted support** (support drawn ∝ true future relevance w — i.e.
"usage predicts future queries", which is the *definition* of the setting in
which a consolidation policy has a job):

| k | A7-lstsq | **A7-ridge** | MAT-gated | MAT-const | best-matrix margin |
|---|---|---|---|---|---|
| 1 | 0.6097 | **0.6137** | 0.4651 | 0.4032 | **−0.1486** |
| 2 | 0.4658 | **0.5278** | 0.4633 | 0.4053 | **−0.0645** |
| 4 | 0.3260 | **0.4855** | 0.4545 | 0.4082 | **−0.0309** |
| 8 | 0.2197 | **0.4493** | 0.4368 | 0.4056 | **−0.0125** |
| 16 | 0.1669 | 0.4068 | 0.4181 | 0.4110 | +0.0113 |
| 128 | 0.0681 | 0.2173 | 0.2855 | 0.4035 | +0.1861 |

**A7 wins outright** for k ≤ 4 and ties through k=8. The charter asked whether
the dense construction is "a construction only a matrix could love"; the answer
is worse than that — it is the *unique* corner that both survives and voids its
own policy. §6.5 registers `a ~ Unif(S^{M−1})`, which is **usage-independent by
construction**; that is precisely why R2-B's ρ-sweep is flat. Make the workload
usage-correlated and the gate acquires a job — and the cache wins.

I could construct no query distribution in which the gate matters *and* the
matrix wins.

---

## §R2.4 MAJOR findings

### R2-F [MAJOR] — the registered headline cell is not reproducible from the stable scripts; §6.9's inventory claim is false

Detailed in §R2.1. `run_cell` has no gate/ρ/precision parameter, so §6.9's
"single parametrized calls into `run_cell`" is factually impossible, and "the
commands recorded in this section's git history" record no commands. The
reconstructed value (+0.227) matches the claim (+0.230), so the figure is sound
and this is a provenance failure. Both un-saved inline checks (gated cell, fp16
sensitivity) are affected — and the fp16 one turns out to have been the more
consequential of the two (R2-C).

### R2-G [MAJOR] — F6's attempt-vs-commit reading is load-bearing, disclosed but not measured

`r2_external_validity.py`, Part B, at the registered cell:

```
  READING-ATTEMPT (DRAFT-R1's):  matrix commits all M       margin +0.2270
  READING-COMMIT:                matrix commits only C=31   margin +0.1336
  DELTA between readings = 0.0934   seed-noise bar = 0.0132   -> 7.1x the bar: LOAD-BEARING
  commit-matched vs ridge-A7:  -0.0270
```

**Charter item 4 answer: the reading is *fair*, but load-bearing, and was chosen
rather than justified.** Attempt-matching is the defensible choice — a C-slot
store is *physically* capacity-rejected, and forcing the matrix down to C writes
handicaps it for a constraint it does not have. But §6.7 dispositions F6
**FIXED** with the claim that *"because mass/FLOPs are reported rather than
artificially equalized, that failure mode cannot recur."* It recurs — relocated
from mass-matching to commit-matching, at 7× the noise bar, and under commit
matching + ridge the margin goes **negative**. §6.8 risk 4 raises the question
honestly but never measures it; a load-bearing interpretive choice must ship with
its sensitivity.

### R2-H [MAJOR] — L2's coherence "enabling condition" is the same conditioning artifact

`r2_external_validity.py`, Part C. §6.6 reports that near-orthonormal keys crush
the margin (M=128: +0.180 at α=0 → +0.010 at α=1) and interprets this as
orthogonal keys making *"A7's C-item regression dramatically MORE informative"*.
It is not information — orthogonalizing the keys is simply the knob that fixes
A7's conditioning. With ridge-A7 the whole effect nearly vanishes:

| M | α | coherence | margin (lstsq A7) | **margin (ridge A7)** |
|---|---|---|---|---|
| 64 | 0.0 | 0.143 | +0.3157 | **+0.0221** |
| 64 | 1.0 | 0.072 | +0.0092 | **+0.0092** |
| 128 | 0.0 | 0.143 | +0.1802 | **+0.0274** |
| 128 | 1.0 | 0.107 | +0.0095 | **+0.0095** |
| 256 | 0.0 | 0.142 | +0.0930 | **+0.0020** |

The registered enabling condition ("keys must stay away from near-orthonormal,
coherence ≳0.10") and the disclosed NCR cross-lane coupling (§6.8 risk 5) are
both derived from the artifact. The cross-lane worry is, at least, unfounded in
the direction stated — which is good news for NCR and bad news for this lane's
claim to have measured anything.

### R2-I [MAJOR] — M9's disposition over-claims ("spec pinned now")

M9 asked for **A8's write op, its read op, and which quantity is matched**.
§6.7 supplies "explicit bilinear read, param-count-matched". The write op is
absent, and "param-count-matched" is exactly the quantity M9 proved incoherent
(the matrix read has **zero** parameters; a flat read head `U ∈ R^{32×1056}` has
33,792). The disposition restates the problem as its solution. Since A8 is the
mandatory param-matched flat-vector ablation under the repo's hard rules
("the param-matched flat-vector ablation blocks ALL downstream decisions"), this
is not a safe deferral.

### R2-J [MAJOR] — the L3-mandated discriminative diagnostics exist only for the killed regime

§5's L3 registers off-target margin and N-way retrieval as *"FIRST-CLASS
pre-registered metrics with bars"*. §6.6 discharges L3 by pointing at §6.2 —
regime (i), which is dead. The **surviving** regime carries no off-target or
N-way diagnostic and no bar for one. Given the aggregate metric was where the
info-free reader scored 0.0113 (clean) but also where the info-free reader *beats
the true matrix* at collapse=0.5 (0.1963 vs 0.1816), the diagnostics are not
optional decoration here.

### R2-K [MAJOR] — m5's "fully-parameterized pigeonhole cell set" is false; d is never swept

Every regime-(ii) figure in §6 is at d=32; the grid varies M only. I ran the
missing sweep (`r2_d_sweep.py`, M/d = 4 fixed, d ∈ {16,32,64,128}):

```
   d    M    C |  A7-lstsq  A7-ridge |  MAT-gated  MAT-best | mgn vs lstsq  mgn vs ridge  best vs A7@8bit
  16   64   15 |    0.0914    0.2157 |     0.2760    0.4929 |      +0.1846       +0.0603           +0.0128
  32  128   31 |    0.0583    0.2225 |     0.2856    0.4914 |      +0.2273       +0.0631           +0.0101
  64  256   63 |    0.0402    0.2240 |     0.2915    0.4971 |      +0.2513       +0.0675           +0.0055
 128  512  127 |    0.0357    0.2178 |     0.2919    0.4975 |      +0.2562       +0.0741           +0.0034
```

The disposition is inaccurate but substantively harmless **in both directions**:
the structure is invariant in d at fixed M/d, so the d=32-only evidence
generalizes — and so does the kill (+0.060…+0.074 vs ridge-A7 at every d; +0.003…
+0.013 for the best matrix vs int8-A7 at every d).

---

## §R2.5 minor findings

- **m-a** — §6.3's frontier table prints only the 6 winning rows. The losing
  M=32 row (**−0.108**, the matrix *loses*) and the sub-bar M=512 row (+0.048)
  are omitted, though the "6/8 cells" count discloses their existence. Print all
  eight.
- **m-b** — the seed-noise bar uses the *unpaired* pooled SEM although both arms
  share seeds. This is conservative (paired SEM would be smaller), so it errs
  honestly; but "20×/29× headroom" describes the noise bar, not the effect's
  robustness, and R2-A shows how little the latter followed from the former.
- **m-c** — §6.8 risk 1 understates its own severity: "+0.005–0.05" holds only
  against unregularized fp16-A7. Against regularized fp16-A7 the registered arm
  loses (−0.035).
- **m-d** — §6.1's Shannon/entropy-coding ceiling check is **correct and
  confirmed** (C = 31.83 vs plain index 31.78; ID cleverness buys A7 <1 slot).
  The section's own conclusion — that *value precision*, not ID coding, is the
  real lever — was right, and R2-C is that conclusion carried to its end.
- **m-e** — §6.6's L6 FLOP arithmetic checks out (matrix 397,312; A7 lstsq
  ≈31,744 ≈ C²d), and holds under ridge too (A7 ridge ≈ d³ = 32,768, still ~12×
  cheaper). What evaporates is the *rhetoric*: "the matrix uses 12.5× more
  compute and still wins by +0.23" becomes "A7 recovers the margin at unchanged
  cost by fixing one call."

---

## §R2.6 Charter item 5 — integrity

**Append-integrity: CLEAN.** `git diff 671d83a 65e7749` on the waterfall is
exactly two hunks — the §0 status block (2 lines replaced by 7) and a pure
append at line 209. Total deleted lines across the file: **2**, both in §0.
**§3 (lines 85–145) and §5 (152–217) are byte-untouched.**
`CONSOLIDATION_ATTACK_R1.md` is unmodified since `dee581f`, and the working tree
has no uncommitted changes under `matrix-thinking/`.

**Disposition table: COMPLETE by ID, MIXED on honesty.** 29 entries = F1–F6 (6)
+ M1–M16 (16) + m1–m7 (7), matching R1's 6 FATAL / 16 MAJOR / 7 minor verdict;
every ID appears exactly once. Spot-checked 9:

| ID | §6.7 disposition | R2 finding |
|---|---|---|
| F1 | FIXED-BY-CONSTRUCTION | Capacity steelman is genuine and correct. But the *substance* (A7 undefeated at the registered bar) is restored by R2-A. **Not fixed in substance.** |
| **F4** | **FIXED-BY-CONSTRUCTION** | **Over-claimed.** L4 discharged on the killed regime's per-item metric; on the surviving statistic the shuffle test reads +0.007 (null) — R2-B. |
| F6 | FIXED | Over-claimed: "that failure mode cannot recur" — it recurs at 7× the noise bar (R2-G). |
| **M5** | **FIXED-BY-CONSTRUCTION** | **Over-claimed.** ρ swept, but the margin is flat in ρ; registering ρ=0.7 "not 1.0" is cosmetic when ρ=0 gives the same margin (R2-B). |
| M9 | DEFERRED, "spec pinned now" | Over-claimed — spec not pinned (R2-I). |
| **M10** | **FIXED-BY-CONSTRUCTION** | **The load-bearing failure.** M10's explicit demand — argue the readout is the *strongest* fair form — was never discharged; the chosen readout is the artifact (R2-A). |
| M11 | FIXED | Partially. Codings pinned, but registering fp32 primary while disclosing that fp16 kills the result is the post-hoc choice M11 warned of; §7's M1 amendment already caught this, and R2-C settles it. |
| M15 | FIXED-BY-CONSTRUCTION | Partially — the pre-check exists, but M15's required off-target-margin leg is missing for the surviving regime (R2-J). |
| m1 | FIXED | **Verified true**: `eps=1e-12` pinned in `r1_steelman_a7.delta_write` and inherited by every script. |

Four of the six FATAL dispositions (F1, F4, F6, plus M10 which carries F1's
weight) do not hold on inspection. Two spot-checked FIXED-BY-CONSTRUCTION entries
(F4, M5) are over-claimed; one (m1) is exactly right.

**Scripts: reproduce from stable filenames**, with the single exception in R2-F.

---

## §R2.7 What I could not break

Recorded so the coordinator can weigh the kill honestly:

- **Both regime kills are sound.** Regime (i) (192 + 48 cells) and regime (iii)
  (45 cells + the exact-binomial survival table) reproduce exactly and I found no
  error in either construction. The regime-(iii) collapse co-location argument is
  correct and well-executed; retiring §1's marquee correction demo as UNVALIDATED
  was the right call and R2 does not disturb it.
- **The exact-binomial fix in regime (iii)** (order statistic via `binom.cdf`
  instead of Monte Carlo) is correct and a genuine improvement.
- **§6.1's capacity accounting and the entropy-coding ceiling are correct**, and
  its conclusion that value precision is the real lever was right.
- **fp16 delta-rule accumulation is safe** — self-correcting, drift bounded at
  ~1.6e-3 independent of stream length. Worth keeping as a durable result.
- **The math of the √(C/M) bound is valid.** Only its application is vacuous.
- **The reported headline number is honest** — my independent reconstruction
  landed at +0.227 against a claimed +0.230.
- **The dense-uniform construction genuinely does defeat a slot-limited exact
  store** when that store is at fp32 and M/d ≥ 4. That result is real; it is just
  classical streaming least-squares, contains no consolidation policy, and dies
  at int8.

---

## §R2.8 Verdict — **BLOCKED**

Regimes (i) and (iii) were killed by Rev-1's own executed sweeps, which I
reproduced. Regime (ii) — the lane's sole survivor — fails on five independent
grounds, any one of which is disqualifying:

1. Its headline margin is a conditioning defect in the falsifier's readout; a
   free, offline-tuned ridge collapses it to +0.066, below the registered 0.15
   bar, **at fp32, before precision symmetry is even considered** (R2-A).
2. Its registered policy — the usage gate, the lane's entire subject — has **no
   measurable effect** on its own registered decision statistic, and is strictly
   worse than a constant step size (R2-B).
3. It loses under the symmetric-precision comparison §7 convened (R2-C, M1).
4. Its theoretical backbone is vacuous by 0.001, because fp32 index coding puts
   the falsifier at exactly C = d − 1 (R2-D).
5. It inverts under the only workload class in which a consolidation policy could
   matter (R2-E).

A Rev-2 cannot repair these jointly. Fixing (1) requires a regularized A7, which
removes the margin. Fixing (2) requires a usage-correlated workload, which is
(5), where the cache wins. Fixing (3) requires either stipulating A7's precision
— the asymmetry §7 exists to forbid — or swapping in an RLS writer that has no
gate, which deletes the lane's subject. The parameter region where the claim is
both true and about consolidation policy is **empty**.

Per the charter, BLOCKED ⇒ **the lane's sole surviving regime is dead and the
lane parks.** Stage 4 validation and the Mac pilot build ceremony do **not** open.
Still zero GPU-h spent on this lane — the waterfall did its job, twice.

**Salvage worth recording** (not a rescue of this lane, and each needs its own
waterfall): (a) fp16 streaming RLS at fixed bytes is a real, budget-legal,
single-pass writer that reaches the linear ceiling — a candidate *writer*
upgrade for any lane that needs one, with a stated drift horizon of T≈2000 at
fp16; (b) the delta rule's self-correcting fp16 behaviour is a durable, reusable
finding; (c) any future memory-vs-cache falsifier in this program must pin the
control's readout as **regularized** and must sweep the control's value precision
— R2-A and R2-C are general instrument bugs, not local ones.

---

## §R2.9 R2 script inventory

Scratchpad, stable filenames, `DRY_RUN_BYPASS=1 python3 <file>.py`.

- `r2_a7_readout.py` — the ridge/tSVD steelman for A7 vs DRAFT-R1's min-norm
  `lstsq`, at fp32 and fp16, with the matrix's own batch-ridge ceiling and the
  √(C/M) / √(d/M) comparison. **The R2-A demonstration.**
- `r2_repro_gated.py` — reconstruction of the unreproducible registered headline
  cell from the §6.5 spec, plus the usage-channel controls (shuffled gate,
  constant-η) and the ρ sweep. **The R2-B demonstration.**
- `r2_precision_symmetry.py` — M1 part 1: A7 at fp32/fp16/int8 vs matrix arms
  (gated, const-η, fp16 delta, fp16 2-head, fp16 lifted d'=45, fp16 nonlinear
  random features, fp32 ceiling); fp16 delta-rule drift vs stream length.
- `r2_symmetric_rls.py` — M1 part 2: the strongest symmetric matrix arm (fp16
  streaming RLS, operator + inverse Gram at exactly the fp32 budget) with
  Sherman–Morrison drift tracked to T=8192.
- `r2_precision_frontier.py` — M1 part 3: A7 value precision swept 32→4 bits at
  M ∈ {64,128,256}, with **offline** (non-oracle) λ tuning as a fairness check.
- `r2_external_validity.py` — Part A query-sparsity sweep under uniform and
  usage-weighted support (R2-E); Part B the F6 attempt-vs-commit readings
  (R2-G); Part C coherence under a regularized A7 (R2-H).
- `r2_d_sweep.py` — the never-swept d axis; the C = d − 1 identity (R2-D, R2-K).
