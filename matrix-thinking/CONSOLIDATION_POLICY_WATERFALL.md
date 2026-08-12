# CONSOLIDATION POLICY WATERFALL — usage-gated eviction writes into fast-weight memory

Opened 2026-08-11 on PI GO (conversation of 2026-08-11; idea developed
with a second agent, verified here). **Lane status: SUPPORT LANE under
the one-spearhead doctrine** — the NCR real-LM spearhead
(write-conditioning design round, K-wall characterization) is not
displaced; this lane draws idle/queue-filler compute only, unless the PI
re-ratifies priorities.

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
  Rev-1 dispatched: falsifier-backwards numpy re-derivation at zero
  compute → DRAFT-R1 → attack R2.
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
