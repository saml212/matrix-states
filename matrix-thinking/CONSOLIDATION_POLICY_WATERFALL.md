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
- **Stage 3 Attack: DISPATCHED 2026-08-12** (Opus-class, charter §4) →
  report `CONSOLIDATION_ATTACK_R1.md`, adjudication pending. Target:
  §3's DRAFT-R0 pre-registration.
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

## §4 Attack round — pending Stage-2 gate discharge

Charter (draft): attack the DRAFT-R0 pre-registration frame-first
(is the partition of arms airtight? can any arm silently see the
usage signal? is A7's defeat truly by construction?), then the
arithmetic (bytes/write-mass matching), then the instrument (recovery
metric, collapse watch). Opus-class. Report to
`CONSOLIDATION_ATTACK_R1.md`. Dispatch ONLY after
`research/consolidation-policy-novelty-2026-08-11.md` records the
R8ZbLi3oUv adjudication and this file's §0 flips Stage 2 to DONE.
