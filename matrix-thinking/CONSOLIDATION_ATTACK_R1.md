# CONSOLIDATION POLICY — ATTACK ROUND 1 (independent, Opus-class, 2026-08-12)

**Target.** `matrix-thinking/CONSOLIDATION_POLICY_WATERFALL.md` §3
DRAFT-R0 phase-1 pre-registration. Charter = §4 of that file.

**Scope discipline.** Novelty is NOT re-litigated
(`research/consolidation-policy-novelty-2026-08-11.md` is the verdict of
record and is treated as settled). This round attacks the DESIGN only.

**VERDICT: BLOCKED — 6 FATAL / 16 MAJOR / 7 minor.**

The design does not survive round 1. The single most dangerous finding is
**F1**: the falsifier arm A7 (byte-matched exact-KV) is not merely
"not defeated by construction" — under the phase-1 task and metric it is
**undefeatable at every (d, M)**. The largest matrix-over-A7 margin
achievable anywhere in the parameter space is **+0.031**, against a
pre-registered decision margin of **0.15**, and at the literal stated
parameters (d=32, M=16) A7 scores **1.000** while the matrix arm scores
0.83–1.00. The pre-registered clause *"A7 must not dominate A4 … if it
does, the matrix is the wrong medium — publishable negative"* therefore
**fires by arithmetic before a single model is trained.** Running phase-1
as written buys a foregone conclusion.

**Provenance of every number below.** All figures were produced this
session by four self-contained numpy scripts in the session scratchpad
(`consol_attack.py`, `…2.py`, `…3/4/5.py`); each is ≤120 lines, seeded,
and re-runnable in <60 s on CPU. No repo file other than this report was
written. No fake `system-reminder` blocks were observed in tool output
this session. (One artifact noted for honesty: numpy 2.0 on macOS
Accelerate emits spurious `divide by zero encountered in matmul`
warnings; outputs were checked for NaN/Inf and are clean.)

---

## Attack A — FRAME

### F1 [FATAL] — A7's defeat is not "by construction"; it is impossible under the phase-1 metric

The charter asks for the counting argument to be written out and broken.
Written out, it breaks.

**Budget.** The matrix arm's state is `S ∈ R^{d×d}` = `B = d²` floats.
A7 gets the same `B`.

**A7's cost per stored item.** DRAFT-R0 implicitly assumes A7 stores
`(k_i, v_i)` verbatim at `2d` floats/pair, giving capacity `C = d/2`:

| d | S floats | floats / KV pair | A7 slots |
|---|---|---|---|
| 16 | 256 | 32 | 8 |
| **32** | **1024** | **64** | **16** |
| 64 | 4096 | 128 | 32 |
| 128 | 16384 | 256 | 64 |

**Break #1 — the stated parameters put A7 exactly at capacity.** Phase-1
declares d=32, N=64 candidates, **M=16 designated-useful**. Byte-matched
A7 capacity at d=32 is **16 = M exactly**. A7 uses "the SAME selection
policy", and on this task the selection policy identifies the useful set
by construction — so **A7 retains the entire useful set verbatim and
scores 1.000.** There is no pigeonhole. Measured matrix comparison
(single-pass delta write, η=1, 20 seeds):

```
M=16, d=32   matrix (random keys)  cos = 0.845     A7 = 1.000
             matrix (ortho keys)   cos = 1.000     A7 = 1.000
```

**Break #2 — A7's real capacity is ~2× the assumed one, and the design
never pins the coding.** The candidate set is finite (N=64), so A7 need
not store the key at all — a `ceil(log2 64) = 6`-bit index suffices:

```
naive (k,v) fp32 : 64.000  floats/pair -> 16 slots
(index, v) fp32  : 32.188  floats/pair -> 31 slots      <-- legitimate
(index, v) fp16  : 16.094  floats/pair -> 63 slots      <-- also legitimate
```

Byte-matching with no pinned representation lets the falsifier be made
arbitrarily strong (or, by charity, arbitrarily weak) after the fact.

**Break #3 — with the honest (index, v) coding, the matrix never wins by
the margin, at ANY M.** Matrix arm vs `A7 = min(1, C/M)` with `C=31`:

```
    M    M/d  matrix rand  matrix ortho   A7=min(1,C/M)      winner (margin)
   16   0.50       0.8238        1.0000          1.0000      A7   (-0.000)
   24   0.75       0.7486        1.0000          1.0000      A7   (-0.000)
   31   0.97       0.6776        1.0000          1.0000      A7   (-0.000)
   32   1.00       0.6671        1.0000          0.9688      MATRIX (+0.031)
   40   1.25       0.5944           n/a          0.7750      A7   (-0.181)
   64   2.00       0.4321           n/a          0.4844      A7   (-0.052)
  128   4.00       0.2494           n/a          0.2422      MATRIX (+0.007)
  160   5.00       0.1965           n/a          0.1938      MATRIX (+0.003)
```

The **global maximum matrix advantage is +0.031** (at M=d=32, and only if
the learned keys are *perfectly* orthonormal). Past M=d orthonormal keys
do not exist, and the two arms cross again only at M/d≈4 where **both are
in the 0.20–0.25 cosine range — i.e. both useless in absolute terms**, and
the margin is +0.007.

**Why the claim's own logic already implied this.** §1's separation leg 3
states the ratio honestly: `d²` numbers hold "up to d exact associations"
vs "d/2 verbatim KV pairs" — a **2× advantage at best**, and only at
perfect key orthogonality. Index-coding erases that factor of 2 entirely.
A 2× capacity ratio cannot produce a 0.15-cosine separation on a
per-item recovery metric at any operating point, because the two arms'
score curves (`f(M/d)` vs `min(1, C/M)`) cross where both are low.

**Failure scenario.** Phase-1 runs. A7 reads 1.000, A4 reads 0.90–0.95.
The pre-registered clause fires and the program records "the matrix is the
wrong medium." That verdict is an artifact of parameter choice and byte
accounting, not evidence about media. It would then be cited in future
design rounds as a settled negative.

**Required fix (attack-derived, not a redesign).** Either (a) pin A7's
representation adversarially (strongest fair form = index-coded, and state
the dtype), accept that per-item recovery cannot separate, and move the
falsifier onto a query family where partial information composes and
exact storage of *values* is itself impossible (e.g. values drawn from a
continuum with M ≫ d, scored by aggregate/compositional error, not
per-item cosine); or (b) drop A7's "defeat" from the pre-registration and
demote it to a reported control, removing the "publishable negative"
trigger. Do not run phase-1 with the trigger live.

### F2 [FATAL] — the A3 ≥0.95 kill switch and any A7-defeating regime are mutually exclusive; key orthogonality is an unstated, unmeasured enabling condition

Single-pass delta-rule recovery is a function of `M/d` **only** — verified
scale-invariant across d ∈ {32, 64, 128} (12 seeds each):

```
  M/d |  d=32  |  d=64  | d=128  | cache line 0.5·d/M
 0.50 | 0.8278 | 0.8433 | 0.8385 |   1.0000
 0.60 | 0.7993 | 0.8087 | 0.8017 |   0.8333
 0.70 | 0.7752 | 0.7707 | 0.7674 |   0.7143   <- crossover vs naive-coded A7
 0.75 | 0.7401 | 0.7582 | 0.7527 |   0.6667
 1.00 | 0.6654 | 0.6756 | 0.6732 |   0.5000
```

Even against the *weak* (naive-coded, C=d/2) A7, the matrix only pulls
ahead at M/d ≈ 0.72, where its own recovery is **0.75 — far below the
pre-registered A3 ≥ 0.95 kill switch.** There is no M/d at which both
"A7 is beaten" and "A3 clears 0.95" hold with random keys. The window is
empty, and it is empty at every scale.

The only escape is near-orthonormal learned memory keys. Quantified
(d=32, M=24, interpolating keys between random and orthonormal):

```
 alpha   mean |off-diag| key coherence   matrix cos   clears 0.95?
  0.00                          0.1430       0.7399       no
  0.50                          0.1213       0.8127       no
  0.75                          0.0606       0.9594       YES
  0.90                          0.0216       0.9953       YES
```

**Phase-1 therefore silently depends on the model learning a memory-key
basis with mean pairwise coherence ≲ 0.05 at M/d = 0.75.** DRAFT-R0 does
not require it, does not measure it, does not report it, and does not
register what happens if it fails. This is the *same* open problem the
NCR lane is currently spending its compute on (the §G3-B32 ortho-write
lever) — the consolidation lane inherits it un-flagged.

**Failure scenario.** A3 reads 0.74. The pre-registered kill switch fires
("the WRITER is the blocker, the policy question is moot") and the lane is
killed. The true cause is that the task was parameterized into the
capacity-limited regime with uncontrolled key coherence — a 60-second
numpy check, not a training run, is what should have caught it. This is
the §G3-B26 pattern exactly: ≈8 GPU-h burned on an instrument that could
not falsify, caught by a read-only check *after* the spend.

**Required fix.** A closed-form writer-capacity pre-check (the script in
this report is sufficient) must run and pass BEFORE any training cell,
and mean key coherence must be a pre-registered reported quantity with a
band. The A3 kill switch must be conditioned on it: coherence high ⇒
NOT-A-WRITER-VERDICT.

### F3 [FATAL] — the primary metric reproduces the §G3-B26 saturated instrument

DRAFT-R0's primary score is `cos(S k_i, v_i)` plus relative error, with
thresholds at 0.95 / 0.90 and a 0.15 margin. The memory value projection
is *learned* ("dedicated position-free memory projections"), and it is
trained against a reconstruction objective — the exact configuration in
which `NCR_REAL_LM_DESIGN.md` §G3-B26 measured a trained adapter collapse
the target space from pairwise cos **0.0837 → 0.9960**, after which
`recovered_frac@0.9` read 1.0 for an information-free read.

Reproduced here. A **rank-1 reader carrying zero per-item information**
(`S = v̄ k̄ᵀ / ‖k̄‖²`), scored with an optimal rescale (the most generous
reading of "relative error"):

```
 pairwise v cos   free-reader cos   best relerr   off-target margin   16-way NN acc
        -0.0026            0.2451        0.9451             -0.2437    0.062 (chance 0.062)
         0.8170            0.9102        0.4114             -0.0235    0.062
         0.9086            0.9562        0.2910             -0.0120    0.062
         0.9542            0.9783        0.2061             -0.0061    0.062
         0.9908            0.9957        0.0926             -0.0013    0.062
```

At the collapse level this program has already measured in its own
trained projections, an information-free reader scores **cos 0.956–0.996
and relative error 0.29–0.09**, clearing every pre-registered threshold,
while N-way retrieval sits **at chance** and the off-target margin is
**negative**. Both of DRAFT-R0's registered metrics saturate together;
they are not independent checks.

DRAFT-R0's collapse watch ("S conditioning, restricted singular values,
read-output cosine spread") does not catch this: a rank-1 collapsed
*value space* with a healthy-looking S is possible, and no trip-wire band
is registered anyway (see M14).

**Required fix.** Pre-register, as PRIMARY and co-equal with continuous
recovery: (i) off-target margin `cos(Sk_i, v_i) − max_{j≠i} cos(Sk_i, v_j)`,
(ii) M-way retrieval accuracy against chance, (iii) pairwise-cosine of the
value-target set itself, with a VOID band. This is not optional — it is
the literal remediation §G3-B26 mandated for this program's instruments,
and the design cites §G3-B26 while omitting its central lesson.

### F4 [FATAL] — A5 is (most likely) a duplicate of A6, so the one contrast that isolates the usage signal may not exist in the design

The mechanism gate is `g(u_i, ‖r_i‖, novelty)` — three inputs. A5 is
specified only as *"A4's scores shuffled between items."* Three readings,
three different experiments:

1. **Shuffle the final gate output `g_i`.** Then top-M of a random
   permutation of N=64 items **is** a uniformly random M-subset — A5 and
   A6 are *the same distribution*. The design would carry two identical
   controls and would never isolate the usage channel.
2. **Shuffle only the `u_i` channel, keep `‖r_i‖`/novelty live.** A5 ≠ A6
   and the A4−A5 contrast does isolate usage — the only informative
   reading. But on this task usefulness is planted independently of
   content, so the `‖r‖` channel carries no signal about the useful set
   either, and A5 collapses back toward A6 empirically.
3. **Transplant A4's trained gate and shuffle at test time only.** Then
   A5 is a broken-model ablation, not a trained control, and any margin is
   uninterpretable (the model never had a chance to route around the
   corrupted input).

Compounding this: DRAFT-R0 never says whether `g` is **learned** or
**hand-specified**. §1 calls it "a per-example learning rate" and frames
the question as "the optimal test-time learning-rate policy" (⇒ learned);
§3 lists A4 flatly as "attention-usage-trace gated" (⇒ possibly a fixed
top-M rule). The A5 protocol is undefined until this is fixed, and so is
the meaning of the headline margin.

Also missing: whether the write-count-matched **top-M selection** is
differentiable. A hard top-M is not. No straight-through / soft-gate /
surrogate is specified.

### F5 [FATAL] — if the gate is learned, it cannot receive a learning signal, and the design cannot tell an instrument-null from a policy-null

Two independent attenuations sit between the gate decision and the loss:

1. **Decay.** The post-eviction query happens ≥16W tokens later. Gradient
   through S is multiplied by `λ^{16W}`:

```
 W=64 -> 16W = 1024 steps
   lam=0.99    half-life    69.3   surviving mass = 3.39e-05
   lam=0.999   half-life   693.1   surviving mass = 3.59e-01
   lam=0.9993  half-life  1024.0   surviving mass = 5.00e-01
```
   Only λ ≥ 0.99932 (at W=64) leaves ≥50% of the signal — and *the same
   factor attenuates the gradient*, not just the read. Any λ ≤ 0.99 gives
   the gate a gradient ~3e-5 of the forward signal.

2. **Truncation.** BPTT length is never specified. If truncation ≤ W (the
   natural choice for an SWA model), the gate receives **exactly zero**
   gradient from post-eviction queries, and A4's selector is untrained.

Under either, the expected outcome is **A4 ≈ A6**. DRAFT-R0 has no probe
that distinguishes "the usage signal doesn't help" (policy null, the
registered NULL branch) from "the gate never learned anything" (instrument
null). It also registers no **signal-strength pre-check** — e.g. AUC of
`u_i` predicting the designated-useful set, measured before the arms run.
Without it, the NULL branch is unpublishable and the compute is wasted.

**Required fix.** Register λ's floor (or freeze λ), register the BPTT
truncation length, register a pre-run AUC(u → useful) gate with a VOID
threshold, and — if the gate is learned — register the surrogate for the
discrete top-M.

### F6 [FATAL] — "all arms matched on state bytes, write count, write mass, FLOPs" is unsatisfiable, and the one implementable matching choice moves a headline arm by 2.2× the decision margin

**The claim is provably false as written.** A1 (SWA-only) writes zero
times — it cannot be write-count or write-mass matched. A2 (dense/
write-all) writes N=64 while A3/A4/A5/A6 write M=16 — they cannot be
simultaneously count- and mass-matched. A7's read is 16 dot products
(≈512 MACs) vs the matrix read's d²=1024 MACs — not FLOP-matched. A8
needs a read head the matrix arm does not have — not param-matched
without changing the matrix arm.

**And the direction of mass-matching decides the verdict.** Realized
Frobenius write mass at η=1 (20 seeds, d=32, N=64):

```
  A3/A4 (write 16)      realized mass =  18.31    cos(useful) = 0.8395
  A2 dense (write 64)   realized mass =  82.36    cos(useful) = 0.1824
  mass ratio A4/A2 = 0.222
```

To match, either scale A2's η down to 0.222 or scale A4's η up 4.5×.

* Scaling A2 **down** to η=0.222: `cos(useful)` moves **0.1824 → 0.5126,
  a +0.33 swing — 2.2× the pre-registered 0.15 decision margin**, from an
  implementation detail DRAFT-R0 does not specify.
* Scaling A4 **up** 4.5× is **mathematically forbidden**: the delta write
  leaves residual `(1−η)(v − Sk)`, so it converges only for `0 < η < 2`.
  Measured collapse past the band:

```
   eta  cos(useful)    ||S||_F
  0.222      0.8378   8.68e-01
  1.000      0.8373   3.66e+00
  1.900      0.6695   7.68e+00
  2.500      0.5155   1.20e+01
  4.500      0.1429   7.90e+01
```

* Additionally, **η ≠ 1 breaks the claim's own framing**: at η=1 the delta
  write is exactly one projection / normalized-GD step ("test-time SGD in
  a linear sub-model", §1). Mass-matching forces η ≠ 1 for at least one
  arm, so at least one arm is no longer doing the thing the claim is
  about.
* **Mass matching is also non-causal.** A total-mass budget over a stream
  requires either look-ahead over future gate values or an unregistered
  running normalizer. The look-ahead form leaks future information into
  the gated arm and not into the others.

**Required fix.** Replace the blanket "all matched on four axes" with an
explicit per-arm invariant table stating, for each arm, which invariants
hold, which are deliberately broken, and in whose favor; pin η per arm
inside the stability band; specify the mass-budget mechanism causally, or
report realized mass ex post with a tolerance instead of enforcing it.

---

## Attack B — ARITHMETIC AND TASK CONSTRUCTION (MAJOR)

### M1 — W is an unbound free symbol
`W` is never assigned a value anywhere in §3, yet the distractor span
(≥16W), the exposure normalization, the eviction schedule, and the λ floor
all depend on it. Worse, **the relationship between W and N=64 is
unspecified**: if W ≥ 64 the whole candidate set co-resides and *nothing
is evicted before the query*, and the experiment measures in-window
attention. Register W, and register `N`, `M`, `W` jointly with the
eviction schedule.

### M2 — λ is unconstrained, per-arm-trainable, and its clock is unspecified
Three separate defects. (a) **No floor**: the retention arithmetic above
shows λ ≥ 2^(−1/16W) is required for 50% survival (0.99865 at W=32,
0.99932 at W=64, 0.99966 at W=128); nothing registers it. (b) **Trainable
per arm** (Tensor Cache's λ_h = σ(θ) is the cited skeleton) makes the
retention horizon itself an arm-dependent variable — A4's "advantage"
could be entirely a larger learned λ. Pin λ shared and frozen, or report
it per arm as a primary covariate. (c) **Decay clock undefined**: per
token or per write event? Under per-event decay, A2 (64 writes) eats 4×
more decay than A4 (16 writes) during the write phase for reasons wholly
unrelated to the gate — at λ=0.99 that is `0.99^64 = 0.526` vs
`0.99^16 = 0.851`, a 38% handicap handed to the control that the
hypothesis needs to beat. §3 risk (v) names the λ/stream-length
interaction but not this.

### M3 — "exposure-normalized" is vacuous under the phase-1 design
In a fixed-W sliding window with hard eviction at age W, **every item is
resident exactly W steps**. Exposure normalization is then division by a
constant, i.e. a no-op that cannot change any ranking. The
exposure-normalization is the *registered novel component* of the claim
(`research/consolidation-policy-novelty-2026-08-11.md`: "No exposure
normalization anywhere" is the stated gap). Phase-1 as specified cannot
test it. Either introduce genuine residency/query-density variance (which
must then itself be de-confounded from position) or drop
"exposure-normalized" from the phase-1 claim and register it as a
phase-2 axis.

### M4 — the usage-independence requirement is self-contradictory and partly non-constructible; two distinct objects are conflated
"in-window usage randomized INDEPENDENTLY of identity, position,
frequency, salience": **usage IS in-window query frequency**, so
independence from frequency is a contradiction. Independence from
salience is not constructible for a *measured attention trace*, because a
trained model allocates attention as a function of content — whatever
content features drive attention will correlate with `u_i` by
construction. The design conflates (a) the planted usefulness label
(which A3 reads), (b) the exogenous in-window query schedule, and
(c) the model's observed attention mass (which A4 reads); these are three
different objects with three different independence properties. Also
unspecified: **which layer's and which head's attention defines `u`** — in
an L×H model that is L·H candidate instruments, and a post-hoc choice is
p-hacking. Register the definition of `u` down to the layer/head reduction
before any run.

### M5 — the task is rigged: predictive validity ρ(usage, future usefulness) = 1 by construction
M=16 "designated-useful" items are, by construction, the high-usage items.
So A3 (oracle) and A4 (usage-gated) read the *same planted label* modulo
estimation noise, and the experiment can only fail for instrument or
optimization reasons. The scientifically interesting question — does
in-window usage predict *future* usefulness when the two are not
identical? — is definitionally excluded. External validity to any real
workload is nil. **Sweep ρ ∈ {1.0, 0.7, 0.4} as a pre-registered axis**;
the ρ=1.0 cell is the sanity check, not the result.

### M6 — the query-family mix is unregistered and arithmetically determines the winner
Three families are named (high-use, low-use, never-used late-surprise)
plus aggregate queries, with **no registered mixing weights**. The
never-used late-surprise family is answerable *only* by A2 (write-all)
and is failed by A3 and A4 by construction. The fraction `f` of the
evaluation drawn from it therefore sets the A4-vs-A2 verdict directly.
Register the weights and **report every metric per family; never
aggregate across families into the decision statistic.**

### M7 — two of the three headline margins are met by selection-count arithmetic, with no usage signal in play
Measured (d=32, N=64, M=16, 20 seeds, mean cosine on the useful set):

```
  A4/A3 (perfect selector)     0.8348
  A5==A6 (random 16 of 64)     0.1842      expected useful hits = M²/N = 4 of 16
  A2 dense (writes all 64)     0.1805      N/d = 2 -> guaranteed over capacity

  A4 − A2      = +0.654        criterion asks for >= +0.15
  A4 − A5/A6   = +0.651        criterion asks for >= +0.15
```

Both criteria clear by 4.3×, driven entirely by (a) N/d = 2 guaranteeing
A2 self-destructs on capacity and (b) a random 16-of-64 selector hitting
only 4 of the 16 useful items in expectation. **A "WIN" on these two
contrasts carries no information about the usage signal.** Only A4−A5
under reading 2 of F4 is informative, and its expected effect size is far
smaller. Re-derive the margins from the arm-specific null, not a single
global 0.15.

### M8 — metric definition, thresholds, and statistics are underspecified for the decision they carry
* "Continuous recovery … scored by cosine+relative-error" — is the
  statistic mean cosine, mean relative error, or the fraction of items
  passing a threshold? If the last, the threshold is unregistered. The
  combination rule (AND / OR / primary+secondary) is unregistered. The two
  are wildly different in stringency: cos 0.90 ⇔ relerr ≈ 0.44 at best
  rescale, while relerr ≤ 0.10 ⇔ cos ≥ 0.995.
* "In-window recall within **1pt** of A1" applies a percentage-point unit
  to a cosine-valued metric.
* **d=32 cosine null is wide**: random-pair cosine has sd = 0.177, and
  `P(cos > 0.15) = 0.204`. One in five purely random reads clears the
  decision margin at item level. (d=16: sd 0.250; d=64: sd 0.125.)
* A "chance reader" floor is not zero: the mean-value reader scores
  cos ≈ 0.14–0.25 at d=32 with no information (F3 table, top row). No
  chance-reader control arm is registered.
* **No CI or test procedure**, despite this program's own precedent
  (§1.40: paired CIs excluding the pre-registered margin). Sampling noise
  alone at M=16, d=32 gives sd 0.044 on a seed mean and SEM 0.026 over 3
  seeds — but seed-to-seed *training* variance dominates that, and this
  program's recorded experience (Task-2 trainability variance, where one
  fresh seed alone cleared the bar) says n=3 is too few for a from-scratch
  tiny model. Register the test, the pairing, and n ≥ 5.

### M9 — A8 (param-matched flat-vector) is either vacuous or a provable strawman
The hard rule bites. If A8's write is `s ← λs + η·vec(r kᵀ)` and its read
is any reshape-equivalent bilinear form, **A8 is literally A4** — reshape
equivalence makes it the same arm. If instead A8 uses a *linear* readout
over the concatenated state and query, `o = U_s s + U_q q`, then
`∂o/∂s` is constant in `q`: the memory's contribution to the output is
**identical for every query**, so A8 has provably zero key-addressed
retrieval capability — a strawman whose failure is architectural, not
informative. Any informative A8 needs an explicitly multiplicative
(bilinear or MLP-with-interaction) read, and then "param-matched" is
strained: the matrix arm's read has **zero** parameters, while a flat read
head `U ∈ R^{32×1056}` is 33,792 parameters. The design also conflates
*state bytes* with *parameters*. Specify A8's write op, read op, and which
quantity is matched, or drop the arm and say why.

### M10 — "aggregate queries no lookup can answer" is false; it is a restriction on the control's decoder, not an information bound
An exact-KV control holding the *same* 16 pairs can compute the
least-squares map `S_ls = V K⁺` **at read time, at zero extra bytes**, and
answer any in-span composition query exactly. Measured (d=32, M=16,
200 novel in-span query keys, 30 seeds):

```
  matrix S (1-pass delta)             aggregate-query cos = 0.8362
  A7 cache + regression readout       aggregate-query cos = 1.0000
  A7 cache + kNN readout (strawman)   aggregate-query cos = 0.3609
```

The claimed separation exists only if A7 is *forbidden* from regressing —
i.e. §1's leg 2 ("lookup vs regression") is an assumption about the
baseline, not a property of the medium. Legs 1 and 2 of the §1 separation
therefore collapse into leg 3 (the bytes-per-item ratio), which F1 shows
is at most 2× and is erased by index coding. Pre-register A7's readout
family explicitly and argue why it is the *strongest* fair form.

### M11 — byte-matching is not pinned to a representation
dtype (fp32/fp16/int8) and coding (verbatim key vs index) change A7's
capacity by 2–4× (16 → 31 → 63 slots at d=32). An unpinned byte budget
means the falsifier's strength is a post-hoc choice. Pin the exact
serialization of both arms, in bytes, in the pre-registration.

---

## Attack C — INSTRUMENT AND CAUSAL BATTERY (MAJOR)

### M12 — the blank-out test is ill-defined for this architecture
"Post-encoding corruption of evicted sources" has **no tensor to corrupt**
in an SWA model: once evicted, the source is out of the window by
construction, and its key/value tensors no longer exist. The test as
written is vacuous — and a vacuous structural check is exactly the failure
mode the hard rules call out ("verify the bottleneck holds with a
gradient-based blank-out test … not a vacuous shape check"). The real
leakage channel is different: information about item *i* propagates
forward through the **residual stream of the in-window tokens that
attended to i while it was resident**, and then hops forward
token-by-token indefinitely; ≥16W distractors do not break that chain.
The battery's **zero-S leg is the test that actually closes this**
(if recall survived via the residual chain, zeroing S would not kill it).
Respecify: delete the "corrupt evicted sources" leg, promote zero-S to the
leakage test, and add the missing quantitative form — recall must drop to
the *chance-reader* floor (M8), not merely "fail".

### M13 — the zero-attention leg is off-manifold and can fail for plumbing reasons
If the memory query `q` is computed from a hidden state that was itself
produced by attention, zeroing attention corrupts the read path as well as
the recent-recall path, and "old survives" fails for reasons unrelated to
the claim. Specify exactly which tensors are zeroed (SWA attention output
only? the whole attention block? at which layers?), and add the positive
control that the memory read is unchanged under the intervention on a
synthetic S.

### M14 — the collapse watch has no trip-wire
"report S conditioning, restricted singular values, read-output cosine
spread" registers *what to print*, not *what invalidates the run*. The
§G3-B26 lesson is precisely that an unbanded watch gets read charitably
after the fact. Register numeric VOID bands (e.g. value-target pairwise
cosine > X ⇒ VOID; read-output pairwise cosine across items > Y ⇒ VOID;
`s1/s2` of S > Z ⇒ flag power-iteration degeneracy) before launch.

### M15 — no writer-capacity pre-check gates the A3 kill switch
A3 can fail from at least five causes: readout family, value-space
collapse, capacity (M/d), key coherence, and η. The kill switch attributes
failure to "the WRITER" unconditionally. A **closed-form, zero-compute
numpy pre-check** — write the M oracle items, measure recovery, coherence,
and the off-target margin — resolves all five before a training run. The
scripts backing this report are that check; running them is free and must
be a pre-launch gate.

### M16 — missing attribution arms
The contribution per the novelty gate is *the usage channel specifically*.
`g` has three inputs. Without single-channel arms — **usage-only**,
**‖r‖-only**, **novelty-only** — an A4 margin cannot be attributed to
usage, and the paper's central claim is unsupported even on a WIN. These
arms are cheap (same skeleton, different gate) and belong in phase-1, not
a follow-on.

---

## minor

* **m1** — `ε` in `r kᵀ/(‖k‖²+ε)` is unpinned; if `ε` is not ≪ ‖k‖² the
  write is a partial step and single-pass recovery degrades further.
* **m2** — `η` is a free parameter that materially changes results even
  inside the stability band (measured: η=0.5 gives 0.8555 vs η=1.0's
  0.8373 at M/d=0.5, because partial steps reduce interference). Pin it
  identically across arms or sweep it identically.
* **m3** — "read before write" makes `r_i` depend on write **order**, so
  arms with different selections traverse different residual sequences;
  order effects should be reported (e.g. shuffled-order replicate).
* **m4** — "A4 within 0.05 of A3" has no stated sidedness. A4 > A3 is
  possible (the oracle's top-M definition need not match the evaluation
  query distribution) and is currently unhandled.
* **m5** — "the pigeonhole-regime cells" are referenced in the success
  criteria but never parameterized anywhere in §3.
* **m6** — the pilot runs on MPS/CPU and the confirm wave on H100. Pin
  dtype and reduction order, and verify pilot↔confirm numerical agreement
  on one shared cell before any cross-comparison of their numbers.
* **m7** — §3 risk (i) says the pilot "may use exact attention (small W)".
  Exact vs approximate attention changes `u` itself, which is the
  instrument. Pin the attention implementation across pilot and confirm,
  not just across arms.

---

## What §3's six "known design risks" MISSED

§3's list covers (i) usage-trace accounting, (ii) usage/salience
independence, (iii) A7 pigeonhole "by construction", (iv) hard eviction,
(v) λ/stream-length, (vi) seed variance. Of the 29 findings above, the
list anticipates parts of five of them (M4 partially, F1 nominally, M12
nominally, M2 partially, M8 partially) and **misses entirely**:

* the value-space-collapse / saturated-instrument mode (**F3**) — despite
  §3 citing §G3-B26 for the collapse watch;
* the A3-vs-A7 mutual exclusivity and the key-coherence dependence
  (**F2**);
* the A5≡A6 degeneracy and the learned-vs-fixed gate ambiguity (**F4**);
* gradient attenuation / BPTT truncation killing the gate's learning
  signal (**F5**);
* the unsatisfiability of the four-way matching claim and the η stability
  band (**F6**);
* that A7's byte capacity depends on its *coding*, not just its dtype
  (**F1** break #2, **M11**);
* that "exposure-normalized" is a no-op under fixed-W residency (**M3**) —
  i.e. the registered novel component is untested;
* that the task plants ρ=1 predictive validity (**M5**);
* that two of three headline margins are arithmetic gimmes (**M7**);
* that A8 is vacuous-or-strawman under reshape equivalence (**M9**);
* that a regression-readout cache defeats the aggregate-query separation
  (**M10**).

Notably, §3 risk (iii) states the requirement correctly — "provably
un-answerable by A7 **by construction** (pigeonhole spec, not vibes)" —
and DRAFT-R0 then supplies exactly the vibes: no spec, and parameters
(d=32, M=16) for which the counting argument runs the *other* way.

---

## Verdict

**BLOCKED.** 6 FATAL / 16 MAJOR / 7 minor.

Do not run phase-1 in any form until F1–F6 are discharged. F1 and F2 in
particular are settled by ~60 seconds of CPU arithmetic that costs nothing
and can be re-run by the adjudicator; if the coordinator disagrees with
any table here, the resolution is to re-execute the scratchpad scripts,
not to weigh prose — per the contradictory-rounds hard rule.

The most economical path forward is not a patch. It is to re-derive the
phase-1 task **from the falsifier backwards**: choose the query family and
metric first such that an index-coded, byte-matched exact-KV store is
information-theoretically defeated, verify that choice in numpy before
writing any model code, and only then fix d, M, N, W, λ, and η. Under the
current parameterization the matrix arm's maximum possible advantage over
a competent A7 is **+0.031** against a **0.15** bar — the experiment is
pre-decided, and the decision is against the lane's own hypothesis.

Two structural notes for the adjudicator, offered without prejudice to the
lane: (a) F2's key-coherence dependence means this lane is coupled to the
NCR spearhead's open ortho-write problem — a fact worth surfacing before
any support-lane compute is committed; (b) F3 is the second appearance of
the §G3-B26 saturated-instrument pattern in a design that *cites*
§G3-B26. A standing checklist item ("does the primary metric have an
information-free floor? measure it with a rank-1 reader before launch")
would have caught it at draft time and is cheap to institutionalize.
