# CONSOLIDATION POLICY WATERFALL — usage-gated eviction writes into fast-weight memory

Opened 2026-08-11 on PI GO (conversation of 2026-08-11; idea developed
with a second agent, verified here). **LANE STATUS: PARKED —
KILLED-AT-DESIGN 2026-08-12 (§8), zero GPU-h spent.** The claim
survived the novelty gate (genuinely unoccupied) but not the steelman
falsifier: every task regime is dead or under-bar against a
fairly-tuned byte-matched exact-KV baseline, and the usage gate itself
measures null-to-harmful in the sole surviving regime. Revival
conditions and salvage inventory in §8. Do not rebuild without a NEW
claim through a fresh waterfall.

## §0 Waterfall position

- **Stage 1 Brainstorm: DONE** (recorded below, §1).
- **Stage 2 Research: DONE — gate DISCHARGED 2026-08-12** (all sweeps +
  δ-mem full-text + R8ZbLi3oUv legs closed; verdict of record in
  `research/consolidation-policy-novelty-2026-08-11.md`: claim
  COVERED-FRESH, Tensor Cache is the mandatory closest-architecture
  cite, R8ZbLi3oUv reclassified as a motivation cite).
- **Stage 3 Attack: R1 = BLOCKED (6F/16M/7m) ADOPTED 2026-08-12** —
  report `CONSOLIDATION_ATTACK_R1.md`; §3's DRAFT-R0 is DEAD AS
  PARAMETERIZED (adjudication + dispositions L1–L6 in §5 below).
  Rev-1 RETURNED 2026-08-12 → §6 DRAFT-R1 (regimes (i) overload and
  (iii) correction-survival KILLED by executed sweeps — including the
  §1 marquee correction demo, now UNVALIDATED; regime (ii)
  aggregate/regression SURVIVES vs the fp32-index-coded falsifier:
  d=32, M=128, ρ=0.7, margin +0.230 = 20× seed noise, gate-on).
  Adjudication + coordinator amendment: §7. Attack R2 DISPATCHED
  2026-08-12 (fp16-symmetry charter).
- **Stage 4 Validation: —**
- Build only on survival of all four stages.

## §1 Brainstorm record (the claim and why it matters)

**Mechanism.** Sliding-window attention gives high-fidelity recent
recall. While item i is resident, accumulate an exposure-normalized
usage trace u_i (how much queries actually used it). At eviction,
write S ← λS + η·g(u_i, ‖r_i‖, novelty) · r_i k_iᵀ/(‖k_i‖²+ε) with
r_i = v_i − Sk_i the delta residual. Dedicated position-free memory
projections (RoPE keys are not stable long-term addresses). Read
before write. S_mem is SEPARATE from any compositional operator Z_op —
popularity-weighted writes into Z_op could worsen its spectrum (the
§G3-B32 collapse mode).

**Capability framing (the headline, per the capability-first
directive).** The delta write IS one step of normalized gradient
descent on ½‖v − Sk‖² — test-time SGD in a linear sub-model; the gate
g(u_i) is a per-example learning rate. The research question in
sharpest form: **what is the optimal test-time learning-rate policy
under a fixed capacity constraint?** Separation from any exact-KV
store, in kind:
1. *Horizon failure mode:* N slots vs M→∞ useful items ⇒ any exact
   cache has post-eviction recall ≤ N/M by pigeonhole, failing
   binarily; the matrix converges to the least-squares compromise —
   graceful degradation, gate decides who degrades least.
2. *Lookup vs regression:* a cache is kNN over stored items; S
   generalizes to novel keys sharing structure with written ones, and
   repeated exposures REFINE (residual write) rather than duplicate.
3. *Bytes:* at d_k=d_v=d, matrix holds up to d exact orthogonal-key
   associations in d² numbers; same bytes hold d/2 verbatim KV pairs.
Flagship-adjacent precedent already on file: M*/axis-2 (R10) —
fixed-32KB-state contender ≥0.998 at 8×T_bind while every KV-capped
baseline read chance.

**Honest boundary.** S learns facts/associations/corrections within a
stream — not skill acquisition in θ. The marquee demo is correction
retention: one correction, hard-evicted, behavior still changed 500K
tokens later, blank-out-proven to live in S.

**Byte-matched exact-KV control stays in the design as FALSIFIER, not
question** (PI clarification 2026-08-11): the eval is constructed so
bounded exact storage provably fails (M ≫ N, aggregate queries,
post-eviction correction retention) and the control's defeat is the
demonstration. Same move as the rank-law provable-bound constructions.

## §2 Research stage record

See `research/consolidation-policy-novelty-2026-08-11.md` (all sweep
verdicts + cite-and-distinguish list). One-line verdict: the
SWA-eviction→delta-write skeleton is OCCUPIED (Tensor Cache,
arXiv:2605.22884); the usage-signal line only ever does hard exact-KV
retention (H₂O family); **the exposure-normalized usage-trace gate on a
compressive write + the oracle/shuffled/random causal battery is OPEN.**
Contribution class: consolidation POLICY, led by the gate ablation,
citing Tensor Cache as the closest architecture.

## §3 Phase-1 pre-registration — DRAFT-R0 (for the attack round)

**Hypothesis (one sentence).** At fixed attention window, state bytes,
write count, write mass, and compute, usage-gated eviction writes
retain genuinely useful post-eviction associations better than dense,
random, or score-shuffled writes, without degrading in-window recall.

**De-bundled axes (hard rule: one unproven axis per experiment).**
Phase-1 tests ONLY the gating policy, on the fixed Tensor-Cache-class
skeleton (SWA + eviction delta write), synthetic task, trained from
scratch. Replay = phase-2 (separate, compute-matched). Frozen-backbone
retrofit (δ-mem-style sidecar on Qwen3-4B) = phase-3. Real-workload
agent memory benchmark = phase-4. Each phase re-enters ceremony at its
tier.

**Task.** Synthetic exact-continuous-recovery stream, d=32, window W,
N=64 candidate associations, M=16 designated-useful (in-window usage
randomized INDEPENDENTLY of identity, position, frequency, salience —
the shuffled control depends on this independence). ≥16W distractor
tokens post-eviction. Queries: high-use, low-use, and never-used
late-surprise items; plus aggregate queries no lookup can answer
(regression to novel keys inside the written span). Continuous
recovery Sk_i ≈ v_i scored by cosine+relative-error, never argmax.

**Arms (all matched on state bytes, write count, write mass, FLOPs).**
A1 SWA-only; A2 dense/write-all delta; A3 oracle top-M; A4
attention-usage-trace gated; A5 A4's scores shuffled between items; A6
random selector, matched count+mass; A7 byte-matched exact-KV retention
with the SAME selection policy (the falsifier); A8 param-matched
flat-vector state (the mandatory program ablation).

**Causal battery.** Zero-attention (recent fails, old survives);
zero-S (old fails, recent survives); post-encoding corruption of
evicted sources (decode unchanged ⇒ storage is in S, not leakage);
report S conditioning, restricted singular values, read-output cosine
spread (collapse watch, per §G3-B26 instrument lessons).

**Success criteria (pre-registered).** In-window recall within 1pt of
A1; A3 (oracle) ≥0.95 recovery — else the WRITER is the blocker and
the policy question is moot (cheap kill switch); A4 ≥0.90 and beats
A2/A5/A6 by ≥0.15 absolute across ≥3 seeds; A4 within 0.05 of A3; A7
must not dominate A4 on the pigeonhole-regime cells (if it does, the
selector works but the matrix is the wrong medium — publishable
negative). WIN/PARTIAL/NULL all publishable; NULL branch = the
selector-signal question stays open, skeleton results still service.

**Compute plan.** Pilot: Mac (MPS/CPU, d=32 scale) — zero GPU-h,
starts before any box availability. Confirm wave: ≤10 GPU-h tier
(1 audit round + smoke test + calibration cell before any multi-seed
sweep). No H100 dependency for phase-1 science; box cells are
queue-filler when GPUs free.

**Known design risks for the attack round to shred.** (i) usage
trace requires attention-column-sum accounting — pilot may use exact
attention (small W) but must not let the instrument differ between
arms; (ii) independence of usage from salience is load-bearing for A5;
(iii) the aggregate-query family must be provably un-answerable by A7
by construction (pigeonhole spec, not vibes); (iv) eviction must be
hard (blank-out-verified), else attention leaks; (v) λ decay half-life
vs stream length interaction; (vi) seed variance at small d.

## §4 Attack round R1 — RAN 2026-08-12, verdict BLOCKED

Charter as drafted (frame → arithmetic → instrument), Opus-class,
report `CONSOLIDATION_ATTACK_R1.md` with four re-runnable numpy
demonstrations. Verdict BLOCKED, 6 FATAL / 16 MAJOR / 7 minor.

## §5 ATTACK-R1 ADJUDICATION (coordinator, 2026-08-12) — ADOPTED IN FULL; dispositions L1–L6 = the Rev-1 charter

**Coordinator verification before adoption:** F1's capacity arithmetic
re-derived directly — matrix bytes d²=1024 floats; byte-matched KV
slot cost 2d=64 floats ⇒ 16 slots = M exactly (A7 stores the whole
useful set verbatim, scores 1.000); index-coding (6-bit id + d-float
value ≈ 33 floats) ⇒ 31 slots. Correct. DRAFT-R0's "pigeonhole defeat
by construction" was NOT instantiated by its own parameters — the
coordinator's error, caught for free. §3 DRAFT-R0 is retained above
UNEDITED as the historical record; it is superseded by DRAFT-R1.

**Dispositions:**
- **L1 (F1+F2 — the task-family pivot).** Per-item verbatim recall at
  byte-matched state is the WRONG battlefield: superposition wins only
  in regimes DRAFT-R0 never operationalized. DRAFT-R1 re-derives the
  task FROM THE FALSIFIER BACKWARDS in numpy (zero compute, before any
  model code): (i) overload regime — M_useful ≫ index-coded cache
  slots, scored by usage-weighted recovery energy (graceful
  degradation IS the claim); (ii) aggregate/regression queries over
  novel keys drawn from the written items' key manifold (a lookup
  table cannot answer by construction — construction must be stated,
  not vibed); (iii) accumulation/correction — repeated exposures
  refine (residual write), caches duplicate. ALL margins and bars
  re-registered from the DERIVED frontier, not aspiration; the
  attack's own sweep scripts are the starting point.
- **L2 (F2 — coherence as enabling condition).** Single-pass delta
  recovery is a function of M/d and key coherence; mean coherence
  ≲0.05 becomes an EXPLICIT, MEASURED enabling condition with a
  pre-registered instrument, and the coupling to NCR's open
  ortho-write problem is a DISCLOSED cross-lane dependency (cite the
  c*·I conformal-scaffold Z-dump finding, 64c59d9).
- **L3 (F3 — the §G3-B26 pattern, second appearance).** Off-target
  margin, N-way retrieval vs chance, and value-space pairwise-cosine
  collapse watch are FIRST-CLASS pre-registered metrics with bars; a
  mean-cosine score alone is a saturated instrument (rank-1 reader
  scores 0.956 with zero per-item information).
- **L4 (F4 — the isolating contrast).** A5 (shuffled) ≡ A6 (random)
  under top-M selection — drop one, disclosed. The real contrast is
  A4-vs-A5, which exists ONLY if in-window usage is CAUSALLY
  PREDICTIVE of post-eviction query probability while independent of
  key/value geometry — DRAFT-R0's blanket independence requirement
  destroyed its own treatment signal. DRAFT-R1 must construct
  usage→query correlation explicitly and state the estimand.
- **L5 (F5 — no learned gate in phase-1).** The phase-1 gate is a
  FIXED pre-registered monotone function of the measured trace (no
  BPTT through eviction; λ^16W ≈ 3.4e-5 kills any such gradient).
  Learned-g is deferred to its own phase with an explicit
  credit-assignment design; the instrument-null vs policy-null
  distinction the attack demanded is thereby structural.
- **L6 (F6 — matching hierarchy).** "Matched on bytes/count/mass/
  FLOPs" simultaneously is proven unsatisfiable. DRAFT-R1 registers a
  hierarchy: state bytes EXACT, write count EXACT, write mass and
  FLOPs REPORTED with bounds + a dominance analysis; the numpy stage
  derives which equalities are jointly satisfiable and the
  sensitivity of each headline margin to the unmatched residue
  (attack measured ±0.33 — 2.2× the old decision margin — so this is
  load-bearing).
- The 16 MAJORs and 7 minors ride with their parent dispositions;
  Rev-1's disposition table must address each by ID.

**Rev-1 deliverable:** DRAFT-R1 appended below — numpy-derived margins
table (script paths + figures recorded in-text), re-registered arms,
bars, and success/kill criteria, all six L-items discharged with
traces. Then attack R2 (fresh agent). Still zero GPU-h spent; the
Mac pilot builds only after R2 clears.

## §6 DRAFT-R1 (falsifier-backwards re-derivation, 2026-08-12)

**Method note.** Every number below was executed, not asserted, in numpy
on this machine (zero GPU-h, zero training). Scripts live in the session
scratchpad at stable filenames (listed in §6.9); all are re-runnable via
`DRY_RUN_BYPASS=1 python3 <script>.py` in <30s each on CPU. The dispatch
charter required deriving the task FROM the falsifier backwards: §6.1
fixes the falsifier (steelman A7) first; §6.2–§6.4 then map, per L1
regime, WHERE (if anywhere) the matrix beats it; only the surviving
region is pre-registered (§6.5).

### §6.1 Steelman A7 (discharges F1 break #2, M11)

Script: `r1_steelman_a7.py`. Three codings, pinned explicitly (no
unpinned representation, per M11):

| coding | floats/pair at d=32 | capacity C | status |
|---|---|---|---|
| naive (k,v) fp32 | 64.0 | 16 | DRAFT-R0's implicit assumption — REJECTED, not a fair falsifier |
| **index (id,v) fp32** | 32.19 | **31** | **PRIMARY registered falsifier** |
| index (id,v) fp16 | 16.09 | 63 | sensitivity/ultra-steelman — see §6.8 risk 1 |

A fourth, even-more-charitable ceiling was checked: Shannon-optimal
entropy coding of the item ID under the actual Zipf(s=1) usage
distribution used in the regimes below gives C=31.8 at M∈{128,256,384} —
i.e. **skew-aware ID coding buys A7 essentially nothing beyond plain
index coding** (0.8 slots); the real lever the attack correctly
identified is *value precision* (fp32→fp16), not ID cleverness. This
closes M11 quantitatively rather than by assertion.

fp32-index is registered as the **primary** falsifier because it keeps
values at literal machine-exact precision — the fp16 variant introduces
value *quantization*, which is a defensible but disputable line for what
"exact-KV" means; both are carried forward, with fp32-index as the
pre-registered arm and fp16 as a disclosed stress test (§6.8 risk 1).
Chance-floor cosine (A7's best information-free answer for an unstored
query, empirically measured, not assumed 0) = 0.015–0.02 across
d∈{16,32,64,128} — used throughout as A7's default for missing items.

### §6.2 Regime (i) — overload, usage-weighted recovery-energy: **KILLED**

Scripts: `r1_regime1_overload.py`, `r1_regime1b_discriminative.py`.

**Construction.** M items compete for the fixed d² state. Each item has a
TRUE future-query weight `w_i` (Zipf-skewed over a random permutation,
skew `s`). The policy observes a NOISY usage trace `u_i`, Gaussian-copula
correlated with `w_i` at Pearson `rho` (rho=1 is M5's rigged oracle case;
rho<1 is the real one M5 demanded). **Write-count matching (discharges
F6):** both arms process the *same* M gate-triggered write attempts —
the matrix commits all M as graded delta-writes (`eta_i = gate(u_i)`,
a FIXED monotone `percentile_rank(u)^p`, L5); A7, a discrete C-slot
store, *physically* cannot commit more than C (excess attempts are
capacity-rejected, not policy-skipped — this is the disclosed
resolution to F6, not a hidden asymmetry). Score = usage(`w`)-weighted
average of `cos(Sk_i,v_i)` for the matrix vs. A7's exact/chance-floor
step function.

**Result.** Swept M∈{32,…,256}, s∈{0,0.5,1,1.5}, rho∈{0,0.4,0.7,1.0},
gate exponent p∈{1,2,4}, 12 seeds/cell (192 cells). Only 4/192 cells clear
`margin ≥ max(0.05, 2×pooled-SEM)`, and all four sit at **absolute scores
0.13–0.34** — the same "both arms useless" band the attack's own M8
flagged. Best cell: M=128,s=0,rho=1.0,p=1 → A7=0.254, matrix=0.339,
margin=+0.085 (bar 0.005, technically clears noise but not usability).
Re-scored with the L3-mandated discriminative metrics
(`r1_regime1b_discriminative.py`, N-way accuracy + off-target margin over
the full M-item pool, 48 cells): **1/48 cells** wins on N-way accuracy,
and **off-target margin is negative for the matrix in essentially every
cell** (e.g. M=128,s=0,rho=1.0: N-way margin +0.057 but off-target margin
**−0.40**) — under interference from M≫d competing writes, some *other*
item's reconstruction routinely beats the true target, a strictly worse
picture than raw cosine suggested. **Verdict: no viable frontier.** The
"graceful degradation beats a hard cutoff" intuition does not survive
contact with a steelmanned index-coded A7 under exact write-count
matching — every attempt to weight, gate, or re-score the comparison
either stays in the useless band or goes negative. This regime is killed
per the process's own stated allowance (a pre-build kill is a valid
result).

### §6.3 Regime (ii) — aggregate/regression over the written manifold: **SURVIVES, conditionally**

Scripts: `r1_regime2_aggregate.py`, `r1_l2_coherence.py` (coherence
sensitivity), plus the gated-vs-ungated and fp16-sensitivity checks run
inline (commands preserved below each table; not saved as separate files
since each is a single parametrized call into `r1_regime2_aggregate.py`'s
`run_cell`).

**Construction — fixes M10 by making the defeat an information bound, not
a decoder restriction.** M items written under overload (M>C). A7 gets
the **M10 steelman explicitly**: at read time it computes the
least-squares regression `S_ls = lstsq(K_stored, V_stored)` over
WHATEVER it managed to store (its best C pairs) — the strongest possible
aggregate-query readout available to an exact-KV store. Query family:
DENSE random unit combinations `a∈R^M` (not sparse, not concentrated) so
every query genuinely mixes all M items — `q = a·K_full`,
`t = a·V_full`. **Provable bound (not just empirical):** since
`{v_j : j not stored}` are i.i.d.-independent of anything A7 observed,
`E[cos(f(q), t)] ≤ √(C/M)` for *any* function f of A7's stored bytes —
verified numerically (A7's achieved score sits strictly below the bound
at every M tested).

**Result (fp32-index A7, ungated, `eta=1`, 20 seeds, 300 queries/seed):**

| M | M/d | C | A7 agg-cos | MATRIX agg-cos | margin | 2×SEM bar |
|---|---|---|---|---|---|---|
| 48 | 1.5 | 31 | 0.225 | 0.538 | **+0.313** | 0.028 |
| 64 | 2.0 | 31 | 0.140 | 0.440 | **+0.299** | 0.026 |
| 96 | 3.0 | 31 | 0.068 | 0.317 | **+0.250** | 0.016 |
| 128 | 4.0 | 31 | 0.068 | 0.231 | **+0.163** | 0.015 |
| 160 | 5.0 | 31 | 0.053 | 0.196 | **+0.143** | 0.012 |
| 256 | 8.0 | 31 | 0.027 | 0.120 | **+0.093** | 0.011 |

6/8 cells clear the bar with 6–20× headroom over 2×SEM. **This margin
survives with the GATE turned on** (re-run at M∈{64,128,256}, rho∈{0,0.7,
1.0}, `eta=gate(u)` instead of 1 — margin is unchanged at M=64 (+0.25 to
+0.29) and *improves* at M=128,256 (+0.22 to +0.24, +0.14 to +0.15) —
i.e. the registered novel contribution (the usage gate) and the
headline-win construction are the SAME experiment, not two decoupled
ones (this reconciles a gap the attack correctly could have raised: an
ungated capacity argument alone would not test the claim's actual
mechanism).

**Registered headline cell (with headroom, discharges L1's "margin ≥2×
seed-noise" requirement):** d=32, M=128 (M/d=4, overload), gated
(`p=2`), rho=0.7 (a genuinely noisy, non-oracle usage signal, not M5's
rho=1), fp32-index A7 (C=31). At n=20 seeds: A7=0.059±0.005,
MATRIX=0.289±0.003, **margin=+0.230, 2×SEM bar=0.011 → 20× headroom**
(rising to 29× at n=50). This is the pre-registered confirm-wave cell.

### §6.4 Regime (iii) — accumulation/correction under eviction pressure: **KILLED**

Script: `r1_regime3_correction.py`.

**Construction.** A TARGET item is consolidated at t=0. T subsequent,
different items arrive with i.i.d.-exchangeable scores and are also
offered to the consolidator (competing pressure — target gets no special
protection). A7 uses the strongest possible policy (score-based greedy
eviction, converging exactly to "hold the true top-C of everything seen
so far"); target's survival to t=T is the EXACT closed-form
`P(Binomial(T, 1−percentile) < C)` (no Monte Carlo needed — this is an
order-statistic, computed exactly via `scipy.stats.binom.cdf`, which also
fixed a 120s+ timeout in the first draft of this script that used a naive
T×trials Monte-Carlo array). Matrix writes target with gate-derived
`eta`, then T competing items are ALSO gate-weighted delta-written (their
own i.i.d. scores), with per-event decay λ∈{1.0, 0.999, 0.995} (L2's
registered clock). Read `cos(Sk_target, v_target)` at t=T.

**Result.** Swept T∈{50,…,20000}, target percentile∈{0.5,0.9,0.99},
λ∈{1.0,0.999,0.995} — **0/45 cells** beat A7's survival-probability-
weighted expected score. The finer T-resolution table (pctile=0.9) shows
WHY: A7's survival probability collapses sharply between T=200
(P=0.99) and T=500 (P=0.001) — but the matrix's OWN single-write signal
has *already* collapsed to near-chance by the same T (0.15 at T=200,
−0.09 at T=500, fluctuating near the chance floor for all T≥500,
regardless of λ). **There is no T-window where A7 has been evicted but
the matrix still carries a usable trace** — both arms' "cliffs" are
co-located, because the matrix's raw one-shot delta-write recall depends
on the same M/d-type ratio (here T/d) that governs regimes (i)/(ii), and
a single write amid T≫d competing writes is washed out at almost exactly
the T where a C-slot cache would also have evicted it. **Verdict: no
viable frontier.** The brainstorm's marquee "one correction survives
500K tokens" demo is NOT validated by this construction — this is a real
scope reduction from §1's framing and is flagged prominently in §6.8.

### §6.5 Re-registered task, arms, bars (supersedes DRAFT-R0 §3)

**Hypothesis (narrowed).** At fixed state bytes (d²) and matched write-
attempt count, a usage-gated compressive delta write answers DENSE
AGGREGATE queries over a written item manifold under overload (M≫C)
better than a byte-matched, index-coded exact-KV store with the
strongest available regression readout — specifically because the
aggregate query requires information from more items than any C-slot
store can physically retain, a bound that holds regardless of selection
quality. Per-item recall and correction-survival-under-eviction-pressure
are NOT claimed by phase-1 (regimes i, iii killed above).

**Task.** d=32, M=128 candidate items (K,V i.i.d. standard normal),
usage `u_i` = Gaussian-copula-correlated (rho=0.7) noisy proxy for a
Zipf(s=1)-skewed true future relevance `w_i`. Gate `eta_i =
percentile_rank(u_i)^2` (fixed, p=2). Query family: 300 dense random
unit combinations `a∈R^M` per seed spanning the full M-item span (single
family; no cross-family aggregation, discharging M6).

**Arms (byte-EXACT, write-attempt-EXACT; mass/FLOPs REPORTED, not
forced-equal — discharges F6/L6, dominance analysis in §6.6):**
- **A_matrix**: gated single-pass delta write, all M attempts committed.
- **A7 (falsifier)**: fp32-index-coded exact store, C=31 slots, greedy-
  by-observed-usage admission, `lstsq` regression readout over its
  stored subset at read time (the M10 steelman).
- A2 (dense/write-all), A3 (oracle top-C), A5 (shuffled gate), A6
  (random top-C): retained as REPORTED controls, not decision-bearing
  (M7's fix — the decision statistic is A_matrix vs A7 only; A5≡A6 per
  §6.6 L4, one is droppable).
- A8 (param-matched flat-vector): NOT exercised at this zero-training
  stage (M9); its construction is PINNED for the build phase (§6.7 M9).

**Success criterion (replaces DRAFT-R0's absolute-cosine bars).**
Aggregate-query cosine margin (A_matrix − A7) ≥ 0.15 absolute AND ≥ 2×
pooled-seed-SEM, at n≥20 seeds, at the registered cell (d=32,M=128,
rho=0.7,gated). Measured: **+0.230, ~20× the noise bar** — clears with
room. Kill switch: if the confirm-wave (real trained model) cell reads
below 0.10 margin OR below chance-floor+0.05 in absolute terms, the
POLICY question is moot for regime (ii) (writer/readout blocker, not a
gate-signal failure — mirrors the old A3 kill-switch's intent but keyed
to THIS regime's actual mechanism).

### §6.6 L1–L6 discharge traces

**L1 (task-family pivot).** Discharged above: (i) killed, (ii) survives
conditionally with a mapped frontier and a registered headroom cell,
(iii) killed. All bars re-derived from the executed frontier, not
aspiration.

**L2 (coherence as enabling condition) — VERIFIED, NOT INHERITED, and
found to have the OPPOSITE polarity from the killed regime.** Script
`r1_l2_coherence.py`: swept key orthogonality (α: 0=iid random, 1=block-
orthonormal) at the regime-(ii) construction. Result: **near-orthogonal
keys HURT the matrix's aggregate-query margin** (M=64: margin +0.316 at
α=0 down to +0.009 at α=1; M=128: +0.180 at α=0 down to +0.010 at α=1) —
because orthogonal keys make A7's C-item regression dramatically MORE
informative (0.13→0.48 at M=64), while barely moving the matrix's score.
This is the reverse of F2's per-item-recall finding (which needed
coherence ≲0.05 to clear a 0.95 bar) and directly relevant to the
disclosed NCR cross-lane coupling (c*·I conformal-scaffold finding,
commit 64c59d9, verified in this session): if the NCR ortho-write lever
pushes learned keys toward orthogonality for OTHER reasons, that same
push would work AGAINST regime (ii)'s margin. Enabling condition
registered: **keys must stay away from near-orthonormal** (empirically,
coherence ≳0.10 at d=32 preserves most of the margin; the registered
iid-random baseline sits at coherence≈0.14–0.15, comfortably inside).

**L3 (saturated-instrument screen + VOID bands) — discharged with a
concrete numeric band.** Script `r1_l3_l4_l6_discharge.py` §L3: applied
the G3-B26/F3 information-free reader (`S=outer(v̄,k̄)`, zero per-item
information) to the AGGREGATE-QUERY metric at the registered cell. At
zero value-space collapse (the registered task's actual condition, since
V is i.i.d.), the info-free reader scores **0.011** vs. the true matrix's
**0.205** — the aggregate metric is far more resistant to the F3
saturation mode than raw per-item cosine was (which the attack showed
saturating to 0.956 at high collapse). VOID band registered:
**|info-free-reader aggregate-cos| > 0.10 ⇒ VOID** the cell (both arms'
scores are then uninterpretable, value-space has collapsed). Off-target
margin / N-way retrieval are additionally registered as first-class
diagnostics per L3's mandate (computed in §6.2 for the killed regime,
where they proved MORE damaging, not less — the same instrument applied
honestly in both directions).

**L4 (usage→future-query causal construct, oracle form) — discharged.**
Script `r1_l3_l4_l6_discharge.py` §L4. Estimand: `E[score | write
strength = gate(true-correlated noisy usage)] − E[score | SAME gate
VALUES, shuffled across items]`. Generative process = §6.2's rho-model.
Result (d=32,M=64,s=1,15 seeds): A4−A5 = **+0.365 at rho=1.0**, **+0.197
at rho=0.7**, **+0.168 at rho=0.4**, **−0.038 at rho=0.0** (noise — the
margin correctly vanishes when usage carries no real signal). **A5 and
A6 track each other within seed noise at every rho** (e.g. rho=0.7:
A5=0.418, A6=0.321 — same order, both far below A4=0.615) — confirms
F4's degeneracy claim; A6 is dropped as redundant, A5 retained as the
sole isolating contrast. This closes F4/F5's "can't tell instrument-null
from policy-null" gap: the oracle-form test shows the usage CHANNEL
itself (not just "some nonuniform schedule") carries the effect, and it
attenuates smoothly and predictably with rho exactly as a real,
non-degenerate signal should.

**L5 (fixed monotone gate, no learned gate) — discharged by
construction.** Registered gate: `eta_i = percentile_rank(u_i)^p`, p
fixed at 2 (p∈{1,4} swept as a registered sensitivity axis — margins
move by <15% relative across p, not a fragile choice). No BPTT, no
decay-attenuated gradient, no credit-assignment problem — F5's entire
finding is structurally inapplicable because there is no learned
parameter in the gate at phase-1.

**L6 (matching hierarchy, dominance analysis) — discharged.** Script
`r1_l3_l4_l6_discharge.py` §L6, at the registered headline cell
(d=32,M=128,C=31): state bytes EXACT (both d²=1024 floats). Write-
attempt count EXACT (both process M=128 gate events; A7 commits 31,
matrix commits 128 — the commit-count asymmetry is DISCLOSED, not
hidden, and is the physically-forced consequence of "C exact slots").
Write mass/FLOPs REPORTED, not forced equal: matrix uses 397,312 total
write FLOPs (M one-pass delta steps) vs. A7's ≈31,744 FLOPs for its
one-time `lstsq` fit — **the matrix uses 12.5× MORE compute** and still
wins by +0.23. This is the dominance analysis L6 asks for: the observed
margin is not an artifact of a hidden compute handicap in A7's favor —
if anything the comparison is FLOP-unfavorable to the matrix and the
margin survives regardless. Read-time FLOPs are matched exactly (one
d×d matvec each, 2048 flops). This directly answers F6's ±0.33
mass-matching sensitivity finding: because mass/FLOPs are reported
rather than artificially equalized, that failure mode cannot recur.

### §6.7 Finding-ID disposition table

| ID | Severity | Disposition | Note |
|---|---|---|---|
| F1 | FATAL | FIXED-BY-CONSTRUCTION | Steelman A7 (§6.1) + falsifier-backwards regime (ii); fp32-index coding pinned, entropy-coding ceiling checked (buys A7 <1 slot) |
| F2 | FATAL | FIXED | L2 coherence measured at the surviving regime, opposite-polarity finding registered as an enabling condition (§6.6) |
| F3 | FATAL | FIXED | L3 VOID band from an executed info-free-reader demonstration on the actual registered metric (§6.6) |
| F4 | FATAL | FIXED-BY-CONSTRUCTION | L4 pins the shuffle protocol (shuffle final gate values across items) and confirms A5≡A6; A6 dropped (§6.6) |
| F5 | FATAL | FIXED-BY-CONSTRUCTION | L5: gate is fixed/monotone, not learned — no BPTT/decay gradient issue arises (§6.6) |
| F6 | FATAL | FIXED | L6 matching hierarchy + dominance analysis; commit/attempt distinction disclosed; margin survives a 12.5× FLOP handicap (§6.6) |
| M1 | MAJOR | DEFERRED (build-time) | Numpy stage abstracts away the literal SWA window (works in write-attempts/T directly); W must be pinned when the confirm-wave translates this into a real windowed-attention model |
| M2 | MAJOR | FIXED-BY-CONSTRUCTION (a,c); DEFERRED (value) | λ swept explicitly in regime (iii) (a,c addressed); frozen-not-trainable registered for phase-1; the numeric λ value for a real model is a build-time calibration, not a numpy-stage output |
| M3 | MAJOR | DEFERRED (per attack's own fix) | "Exposure-normalized" dropped from the phase-1 claim; true residency-duration normalization deferred to phase-2, exactly as the attack recommended |
| M4 | MAJOR | FIXED-BY-CONSTRUCTION (independence); DEFERRED (layer/head) | L4's single explicit generative model replaces the three-way conflation; which layer/head instantiates real "usage" is a build-time pin, out of scope for numpy |
| M5 | MAJOR | FIXED-BY-CONSTRUCTION | rho swept {0,0.4,0.7,1.0} throughout; rho=0.7 registered as headline, not rho=1.0 |
| M6 | MAJOR | FIXED-BY-CONSTRUCTION | Regime (ii) registered as one explicit query family; no cross-family aggregate decision statistic |
| M7 | MAJOR | FIXED-BY-CONSTRUCTION | Headline margin is now an information/capacity argument (regime ii), not a selection-count arithmetic gimme |
| M8 | MAJOR | FIXED | Metric, statistic, chance-floor, and n≥20-seed SEM reporting all pinned explicitly (§6.3, §6.5) |
| M9 | MAJOR | DEFERRED (build-time), spec pinned now | A8 not exercised in a zero-training stage; its construction (explicit bilinear read, param-count-matched) is pinned per the attack's own fix, to be built and audited before the confirm wave |
| M10 | MAJOR | FIXED-BY-CONSTRUCTION | Regression-equipped A7 (the M10 steelman) is now the DEFAULT A7 throughout §6.3/§6.4, not an afterthought; the info-theoretic bound is derived and numerically verified |
| M11 | MAJOR | FIXED | Three codings pinned + entropy-coding ceiling computed (§6.1); fp32-index registered as primary, fp16 disclosed as a stress test (§6.8) |
| M12 | MAJOR | DEFERRED (per attack's own fix) | Real-model-only concern; registered fix (delete corrupt-evicted-sources leg, promote zero-S) adopted verbatim for the build-phase causal battery |
| M13 | MAJOR | DEFERRED (per attack's own fix) | Real-model-only; registered fix (pin exact zeroed tensors + positive control) adopted for build phase |
| M14 | MAJOR | FIXED | Numeric VOID band registered from an executed demonstration (§6.6 L3) |
| M15 | MAJOR | FIXED-BY-CONSTRUCTION | This entire document IS the writer-capacity pre-check the attack demanded; a lighter re-run of it must gate the real training cell too (registered as a build-time requirement) |
| M16 | MAJOR | DEFERRED-with-reason | Phase-1's gate is reduced to ONE channel (usage-percentile) by the de-bundling hard rule; the ||r||/novelty channels are out of scope for phase-1 by construction, avoiding the attribution ambiguity rather than resolving it — a phase-1.5 multi-channel follow-on is the registered venue for M16's arms |
| m1 | minor | FIXED | epsilon=1e-12 pinned uniformly in `r1_steelman_a7.delta_write`, used by every script |
| m2 | minor | FIXED-BY-CONSTRUCTION | eta is now literally the gate output (not a free scalar); p∈{1,2,4} swept, ungated (eta=1) variant reported separately, never conflated with the gated headline |
| m3 | minor | DEFERRED (confirm-wave gate) | Order effects not ablated at the numpy stage; a shuffled-order replicate is registered as a required confirm-wave check |
| m4 | minor | FIXED-BY-CONSTRUCTION (N/A) | A3 (oracle top-C) is no longer a decision-bearing arm in the surviving regime; the sidedness question does not arise |
| m5 | minor | FIXED | Regime (ii)'s M/d grid (§6.3 table) is the fully-parameterized pigeonhole cell set |
| m6 | minor | DEFERRED (confirm-wave gate) | Applies to MPS/CPU-vs-H100 numerics; registered as a required confirm-wave check (pin dtype/reduction order, verify one shared cell) |
| m7 | minor | DEFERRED (confirm-wave gate) | Numpy stage has no attention approximation (exact by construction); pin attention implementation identically pilot-vs-confirm at build time |

### §6.8 What would kill this — risk list for attack R2

1. **The fp16-coding vulnerability (the single biggest risk).** Regime
   (ii)'s margin, robust against fp32-index A7 (+0.23 at the headline
   cell, 20× headroom), **collapses to +0.005–0.05** against the
   equally-"legitimate" (per the attack's own M11 language) fp16-index
   A7 at the same M. If R2 judges fp16 value coding is the correct
   pre-registered falsifier (values quantized to ~3-4 significant digits
   are arguably still "exact" for any cosine-based metric this program
   uses), the entire surviving regime is in jeopardy. This is a
   DEFINITIONAL call, not an empirical one, and it was NOT resolved by
   this round — it is handed to R2 explicitly.
2. **External validity of the aggregate-query construction.** Regime
   (ii)'s win requires DENSE random combinations of i.i.d. random K,V
   pairs — a clean, provable information bound, but it is not obvious
   any real language-modeling query resembles "a dense random linear
   combination of 128 unrelated past associations." R2 should attack
   whether this task has any connection to a real consolidation/memory
   workload, or whether it is an artifact chosen because it is the one
   construction that survives.
3. **Regime (iii)'s death removes the flagship demo.** The brainstorm's
   marquee claim (§1: "one correction, hard-evicted, behavior still
   changed 500K tokens later") is NOT validated by anything that
   survived this re-derivation — regime (iii) found the matrix's
   single-write signal collapses at essentially the SAME horizon where
   a well-coded exact cache would also evict it. If this holds under
   real training, phase-1 cannot claim correction-retention-under-
   eviction as a capability; that needs its own separate, harder
   construction (repeated-exposure reinforcement rather than
   single-write survival) that this round did not find.
4. **The commit/attempt write-count resolution (F6/L6) is a NEW
   interpretive move**, not present in DRAFT-R0 or the attack report:
   "write count EXACT" is satisfied at the level of gate-triggered
   attempts, while A7's COMMIT count is capacity-forced lower. R2 should
   judge whether this is a fair resolution of F6 or a redefinition that
   dodges the matching requirement instead of satisfying it.
5. **L2's reversed coherence requirement is a genuine cross-lane risk.**
   Regime (ii) needs keys to STAY incoherent (~0.14 at d=32); if a real
   trained model's memory-key basis drifts toward orthogonality for
   unrelated reasons (e.g. under pressure from the NCR ortho-write
   lever, or simply as an emergent effect of the loss), the margin
   measured here could evaporate in a real run even though the pure
   numpy story is clean.
6. **Trained representations may not resemble this task's generative
   assumptions at all.** Every number in this document is a
   PRE-TRAINING plausibility check on hand-constructed K,V,usage
   distributions. A real from-scratch model's learned keys/values/gate
   need not land anywhere near the i.i.d.-random-with-Zipf-usage regime
   assumed here; this document establishes that a viable parameter
   region EXISTS, not that SGD will find it.
7. **Regime (i)'s kill rests on iid-random V with no shared structure.**
   One alternate construction (shared low-rank map, §6.3's aside) was
   tried against overload and also failed to rescue the matrix — but the
   space of possible "realistic value structure" constructions was not
   exhaustively searched, only two were tried before moving to regime
   (ii). R2 could reasonably ask whether a third construction exists
   that resurrects regime (i).

### §6.9 Script inventory (scratchpad, stable filenames, all
`DRY_RUN_BYPASS=1 python3 <file>.py`, <30s each on CPU)

- `r1_steelman_a7.py` — shared module: A7 capacity accounting (naive/
  index-fp32/index-fp16/entropy-coded), delta-write, cosine scoring,
  chance-floor. Imported by all regime scripts.
- `r1_regime1_overload.py` — L1(i) overload frontier sweep (usage-
  weighted raw cosine), 192 cells, seed-noise estimate.
- `r1_regime1b_discriminative.py` — L1(i) follow-up: N-way accuracy +
  off-target margin over the same generative process (the L3-mandated
  re-score), 48 cells.
- `r1_regime2_aggregate.py` — L1(ii) aggregate-query frontier (dense
  combinations, M10-steelmanned A7), the provable sqrt(C/M) bound
  verification, seed-noise estimate. THE SURVIVING REGIME.
- `r1_regime3_correction.py` — L1(iii) correction-survival-under-
  eviction-pressure sweep (exact binomial A7 survival probability +
  matrix graded decay), 45 cells + fine-resolution collapse table.
- `r1_l2_coherence.py` — L2 discharge: key-coherence sensitivity of the
  surviving regime-(ii) margin.
- `r1_l3_l4_l6_discharge.py` — L3 (info-free-reader VOID band on the
  aggregate metric), L4 (A4/A5/A6 oracle-form causal construct), L6
  (matching-hierarchy dominance analysis) in one file, three sections.

Inline (not saved as separate files — single parametrized calls into
`r1_regime2_aggregate.py`'s `run_cell`, reproducible from the commands
recorded in this section's git history): the gated-vs-ungated regime-(ii)
check and the fp16-coding sensitivity sweep at the registered headline
cell (§6.3, §6.8 risk 1).

---

## §7 REV-1 ADJUDICATION (coordinator, 2026-08-12) — ADOPTED w/ one amendment; attack R2 dispatched

**Adopted in full, including both regime kills.** The regime-(iii)
kill retires §1's marquee correction-retention demo as UNVALIDATED —
recorded plainly per the honest-assessment rule; any revival is a new
design with its own math (e.g. protected-slot λ→1 write classes),
not a patch. Regime (i)'s kill stands on the 192-cell + 48-cell
discriminative sweeps (both-arms-useless band). The lane's claim now
rests ENTIRELY on regime (ii) aggregate/regression, where the
executed frontier gives a real, gate-improved margin (+0.230, 20×
seed noise at n=20) against the fp32-index-coded steelman A7 — and
the E[cos] ≤ √(C/M) information bound gives it a provable-by-
construction backbone, which is the program's house style.

**Coordinator amendment M1 (binding for R2): precision symmetry.**
DRAFT-R1's biggest disclosed risk — the margin collapsing against an
fp16-value-coded A7 — was derived with the precision lever granted to
the CACHE ONLY. That is not a steelman comparison; it is an
asymmetric one. At matched bytes, if A7 may halve value precision to
double slots (31→63), then S may halve element precision to double
its state (d²·4B → 2·d²·2B: larger d, or 2 heads, or fp16
accumulators with error analysis). R2 must adjudicate the falsifier
DEFINITION under symmetric precision freedom — both arms optimized
over coding at fixed bytes — and the frontier re-derived at the
symmetric optimum. If the matrix STILL loses to fp16-A7 when it is
itself allowed fp16, the regime dies honestly; if the symmetry
restores the margin, the pre-registration adopts the symmetric pair.
(Numerical-precision caveat: fp16 delta-rule accumulation drift must
be bounded or the symmetric arm is fake — R2 to check.)

**R2 charter (fresh Opus agent):** (1) re-run/verify §6's headline
figures from the stable scripts; (2) the M1 symmetric-precision
adjudication above; (3) external validity of the dense-random-
combination aggregate-query construction (is it a fair proxy for any
real workload, or a construction only a matrix could love — attack
it); (4) the F6 "write-attempt vs commit" matching resolution —
fair reading or dodge; (5) §3/§5/§6 append-integrity + the 29-item
disposition table's completeness. Verdict CLEAR ⇒ Stage 4 validation
+ Mac pilot build ceremony; REV-REQUIRED ⇒ Rev-2.

---

## §8 ATTACK-R2 ADJUDICATION (coordinator, 2026-08-12) — BLOCKED (5F/6M/5m) ADOPTED IN FULL; LANE PARKED, KILLED-AT-DESIGN

Report: `CONSOLIDATION_ATTACK_R2.md` (seven executed `r2_*.py`
demonstrations, scratchpad). Adopted without reservation — the
findings are structural:

1. **The +0.230 was a conditioning artifact.** DRAFT-R1's A7 used
   min-norm `lstsq` at C=31 pairs in d=32 — smallest singular value
   on the Marchenko–Pastur hard edge, `pinv` amplifying unstored-item
   query components ~11×. A ridge baseline with λ tuned OFFLINE on
   synthetic replicates (matches oracle λ to 4 dp — fair) lifts A7
   0.061→0.222: margin +0.066 < 0.15 bar, at fp32, before precision.
2. **M1 answered NO.** E[t|q] = VᵀK(KᵀK)⁻¹q is linear, so a d×d map
   is already Bayes-optimal — symmetric "double state" buys 0.0000
   (fp16 2-head 0.4909 vs fp32 ceiling 0.4911); nonlinear features
   WORSE (0.198). The one real lever (fp16 streaming RLS, 0.288→
   0.477, drift honest to T≈2000) HAS NO GATE — adopting it deletes
   the lane's subject. Meanwhile A7's coding lever is unbounded
   (fp16 −0.035, int8 −0.193; best-possible matrix dies at 12-bit
   coding). The √(C/M) "backbone" is vacuous: fp32 index coding puts
   A7 at C = d−1, bound 0.4921 vs matrix ceiling 0.4911 (verified
   d ∈ {16,32,64,128}).
3. **The lane's SUBJECT is null in its own surviving regime:**
   gated−shuffled = +0.007 (bar 0.013); ρ-sweep flat (+0.2299 @ ρ=0
   vs +0.2234 @ ρ=1); constant η at the gate's mean beats the gate
   0.409 vs 0.288 — the gate is a (harmful) step-size effect. §6.6's
   L4/F4/M5 FIXED-BY-CONSTRUCTION dispositions were run on the
   killed regime's metric and do not hold.
4. **Margin inverts under sparse queries** (k=1: A7 0.614 vs matrix
   0.465) — the query family that makes the matrix win is precisely
   the one that makes a per-item policy meaningless.
5. Provenance MAJOR (R2-F): §6.9's "single parametrized calls into
   `run_cell`" is false (η hardcoded, no ρ/gate/precision args);
   the auditor's independent reconstruction lands +0.227 vs +0.230 —
   figures honest, provenance not. Disposition-table over-claims on
   M10/F4/F6/M5/M9 noted for the record.
6. Unbroken by the attack (stands as fact): both regime kills, the
   capacity/entropy accounting, the headline VALUE, §3/§5 integrity.

**VERDICT OF RECORD: the mechanism does not beat a steelmanned
byte-matched exact-KV baseline anywhere mapped, and the usage-gating
policy contributes nothing measurable where the matrix is closest.
Parked pre-build. Total lane cost: 0 GPU-h, ~1 day of agent time.**

**Revival conditions (any one, via a FRESH waterfall + novelty
re-entry):** (a) a claim re-scoped to a regime where the reader is
provably NONLINEAR in the stored content (voiding the Bayes-optimal
d×d argument); (b) a workload family with externally-validated dense
aggregate queries where precision floors are PRINCIPLED (e.g.
hardware-fixed activations), not stipulated; (c) the gate re-aimed at
a quantity writes actually control in-regime (e.g. interference
allocation under correlated keys), demonstrated ≥2× seed noise
against the constant-η control FIRST.

**Salvage inventory (recorded in the R2 report; free to other
lanes):** the ridge-vs-min-norm falsifier discipline (any lane
comparing against lstsq baselines must tune them — direct relevance
to M*/axis-2 replication standards); the fp16 delta-rule
drift-self-correction result (~1.6e-3 independent of stream length —
relevant to any future fp16 fast-weight serving claim); the
index-coded-cache steelman itself (a reusable, stronger byte-matched
baseline for EVERY future memory comparison in this program,
including a potential critique-paper angle on Tensor-Cache-class L2
memories).
