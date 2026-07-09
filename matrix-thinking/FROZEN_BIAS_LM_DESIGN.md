# FROZEN-BIAS-LM Design — Does the Constancy-Suffices Fix Improve a Real From-Scratch Small LM?

**Status: RUNG-1 TRAINING + MEASUREMENT COMPLETE (2026-07-06). All 20
mandatory training cells ran; full measurement pipeline (BANDS_PINNED,
retrofit re-evals, fit, cosine diagnostic) complete. Result: FOURTH
OUTCOME, "sim-training divergence" (§1.3) — DESCRIPTIVE TIER ONLY, not a
CONFIRM or REFUTE. See the VERDICT section at the end of this file for
the full honest readout.** No training/model code
written for this document. CPU-only simulation work performed and reported
below (`sim_frozen_bias_direction.py`, round 2; `sim_frozen_bias_training_
mediated.py`, round 3, EXTENDED round 4 — 2 effect families, wider
effect_strength grid — EXTENDED AGAIN round 5, family B's own λ-sweep,
§11b-3) — NO GPU work at design time. Round-2's
own primary bar was KILLED by an independent round-2 attack (FATAL:
auto-pass artifact, §1/§7.1) — round 3 REDESIGNED the primary observable
(Arm 2 vs. a new eval-only Arm 1′ control, co-primary with pre-blend
`k_raw` geometry) rather than patching the old bar. Round 4 found the
round-3 bar's own validating sim used a SYNTHETIC noise floor never
checked against real cross-seed training variance; real archived data
shows the real floor is 6.9×–16.9× larger, so the fixed-threshold primary
bar was DEMOTED to an ESTIMATION-mode readout (§7.1-real).
**Round 5 (independent attack verdict: REVISE-THEN-PROCEED)
PINS the single operative CI formula across every bar in this document
(§7.1-real.1 — `mean_delta ± t(2,0.975)·s_ref/√n`, pinned thresholds
0.0546/0.1064 for openr1/wikitext) resolving a 3-way threshold
inconsistency that existed pre-fix, and DESCOPES the design's committed
scope to RUNG-1-ONLY (§6.2, §8.1) — rung-2 is formally PARKED, its gate
kept on the books but requiring a fresh design review before any launch.**
See the ROUND-5 REVISION LOG (and,
before it, the ROUND-4 and ROUND-3 REVISION LOGs) at the end of this file
for the full finding→fix trace, and the VERDICT section at the very end
for the rung-1 result itself.**

---

## 0. Reading list this design builds on (context, not repeated here)

- `matrix-thinking/KEY_ANCHORING_DESIGN.md` §10.13 (mechanism-tier confirmation
  wave VERDICT, 2026-07-06) and §10.14 (candidate-(e) VERDICT, 2026-07-07,
  CONFIRMED-BY-ABLATION) — the constancy-suffices finding this whole program
  exists to carry into a real LM. Every number below is re-quoted from these
  two sections, verified directly against the file (not recomputed or
  estimated) during this design pass.
- `matrix-thinking/deltanet_rd/key_anchoring.py` — `anchor_blend_gather_scatter`
  (the exact blend `sub_blend = normalize((1-λ)·k_raw + λ·anchor)`),
  `frame_potential_init` / `random_unit_rows_init` (candidate-(e)'s two frozen
  constructions), `LAMBDA_LOG_CADENCE_STEPS`, `ANCHOR_INIT_SEED`.
- `matrix-thinking/deltanet_rd/model_rd.py` (`DeltaNetRDBlock.__init__`,
  read directly this session) — the `anchor_active` / `anchor_lambda_mode`
  / `anchor_table_frozen` / `anchor_table_init_mode` flags and, critically,
  the hard assert `anchor_active REQUIRES geo3_active` (line ~812: "anchoring
  is defined as a modification to geo3's own write path, not a replacement
  for it"). This assert is the single most important fact this design has to
  route around — see §2 below.
- `matrix-thinking/SCALE_TRANSFER_DESIGN.md` §4 (Track B: geo3-in-LM, the two
  registered Wave −1 gate criteria, the hard no-launch routing) and its own
  §5.9/§5.6 (the Wave-C-scale 14M control config, measured throughput, rung
  definitions) — the from-scratch-LM harness and corpora this program reuses,
  and the graveyard this program's design must not re-enter.
- `matrix-thinking/TRACKB_REDESIGN.md` §1, §14.1, §14.6 — restated in full in
  §3 below: what actually killed Track B (two independent barriers), so this
  program's design does not repeat either failure mode by construction.
- `matrix-thinking/deltanet_rd/run_lm_rd_trackc_sweep.py` — the sweep harness
  conventions this program's launcher must inherit: `PROGRAM_SPENT_GPUH`
  budget guard (`budget_guard`, ~line 786), the disk-space gate
  (`disk_space_check`, ~line 847, `DISK_SAFETY_FACTOR=1.5`), calibration-first
  discipline (`derive_timing_constants`, ~line 610), `BATCH_SIZE_BY_RUNG`,
  rung param-count verification (`verify_param_count`).
- `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` — the `DeltaNetLM` /
  `DeltaNetLMMixer` from-scratch-LM harness itself (`d_model=256, d_state=64,
  n_layers=2` at 14,048,896 params, read directly this session).
- `STATE.md` "SESSION HANDOFF (2026-07-06)" and "Hardware" — GPU occupancy
  (GPUs 0–1 running Track C rung-3 until ≈05:00 UTC 2026-07-08; GPUs 2–7
  idle) and the rung-3 2× timing-overrun lesson (§5 below).
- `matrix-thinking/submissions/iclr-2027/NARRATIVE.md` §0/§1 (round 5) — the
  paper arc this demo serves: a rejected entity-alignment mechanism, a
  confirmed constancy-suffices explanation (ablation, in the from-scratch
  synthetic grammar only so far), and now the question this document answers:
  does that architecturally-free fix do anything in a real from-scratch LM?

---

## 1.0 Expected outcome, stated plainly (ROUND-5 fix 5, MINOR-6)

**Under the pinned CI formula (§7.1-real.1) and this design's own tested
CPU-sim proxies, the most likely rung-1 outcome is an ESTIMATION-mode
result whose CI includes zero at λ=0.58** — i.e. the measured
`span_frac(Arm 2) − span_frac(Arm 1′)` delta, run through
`mean_delta ± t(2,0.975)·s_ref/√n`, most likely straddles 0 on at least
one (very plausibly both) of the two corpora, because neither tested proxy
effect family (family A within its trustworthy interpolation window,
family B at its own asymptote) produces a synthetic signal anywhere close
to either corpus's pinned threshold (0.0546 openr1 / 0.1064 wikitext,
§7.1-real.1). **This is not a failure mode to hide — it is a precise,
honest characterization of the transplant-faithful fix's own effect size**
(or absence of one), and is publishable as such: a null/zero-crossing CI,
reported with the real measured delta and its correctly-computed
small-sample interval, is a legitimate, informative result in its own
right, distinct from either a clean CONFIRM or a clean REFUTE. **A
CONFIRM requires a REAL effect larger than anything this design's own CPU
proxies modeled — which the sims can neither rule in nor rule out; that is
precisely why the real measurement (rung-1's actual training runs) is
being run at all, rather than settling this question by simulation
alone.** Rung-2 (§6.2) is, correspondingly, a LOW-PROBABILITY,
wikitext-gated event this wave: it requires not just a clean rung-1
CONFIRM on every condition (§6.2's gate) but ALSO a fresh design review
re-deriving rung-2's own power from rung-1's measured numbers (§6.2's new
PARK clause) — a compound condition this design does not expect to be met
routinely, disclosed here so no reader mistakes rung-1's own committed
budget (§8.1) for an implicit promise that rung-2 follows.

---

## 1. One-sentence hypothesis

**Adding an episode-constant (here: corpus-constant) frozen-random additive
bias to keys at a fixed blend weight λ≈0.58, with zero new trainable
parameters in the bias term itself, measurably produces a training-mediated
key-geometry stabilization signature in a real from-scratch small LM
relative to an architecture- and token-matched baseline with no such bias.**

**ROUND-4 fix 5 (reworded from behavioral to geometric language):** the
pre-round-4 text above said "long-range/multi-hop token-recall behavior" —
behavioral language this design's own registered observables (§4.a/§4.a-i:
`span_frac`, a Tier-2 descriptive/interventional key-Gram-deviation readout;
§9's own recap) have never measured and, per §4.a's own disclosed weakness
and §4.c's explicitly-deferred recall-suite follow-on, do not currently
attempt to measure. No recall task, token-recall metric, or behavioral
eval exists anywhere in this program's registered scope (§4.c is REJECTED
as primary this wave). Restating the hypothesis in the language of what is
actually registered — a geometric stabilization signature in key space,
read via `span_frac` pre- and post-blend (§4.a/§4.a-i) — makes §1
consistent with §4.a/§9 rather than promising a behavioral claim the
design's own instrument cannot support this wave.

**Revision note (ROUND-2, superseded by ROUND-3 below, kept for the
historical trace):** the first attack round's FATAL finding 1 correctly
identified that the primary bar's direction (§7.1, originally: Arm 2's
`span_frac` must fall) was asserted, not derived. The round-2 revision built
a CPU simulation (`deltanet_rd/sim_frozen_bias_direction.py`, reusing the
actual probe/span_frac code, never reimplemented) that appeared to resolve
the direction: span_frac falls under dense per-token blending, provided the
baseline already has non-orthonormal structure in the archived 0.248–0.389
range. **This reading itself has now been KILLED by the round-2 attack
(FATAL finding 1, this round): the "falls" result is a mechanical artifact
of the blend arithmetic, not evidence of learning — see the ROUND-3
REVISION note directly below and §7.1's rewritten derivation.**

**Revision note (ROUND-3, this pass — supersedes the ROUND-2 note above):**
independent round-2 attack found, correctly, that the round-2 primary bar
(Arm 2 post-blend `span_frac` ≥2 SD below Arm 1) is an **AUTO-PASS
ARTIFACT**: the sim's own `crossover_scan` (quoted in full below, §7.1)
shows `dense_arm2_span_frac` confined to a narrow ~0.02–0.09 band across
EVERY tested baseline structure (`shared_frac` 0.0→0.7), while the
baseline's own `span_frac` ranges 0.0→0.35 over the identical sweep — dense
per-token blending toward one of 50,257 near-orthogonal random rows
(`E[pairwise dot]≈1/√64≈0.125`) mechanically PINS the measured population's
`span_frac` near this narrow band regardless of what the pre-blend keys look
like. The bar was measuring blend arithmetic applied once, at eval time, to
a static population — never a training effect. Confirmed independently, this
pass: the same pinning artifact exists for Arm 2′ (global single vector) at
an even more extreme pinned band (~0.61–0.71, since a single fixed direction
is a more severe collinearity injection than 50,257 distinct rows) — meaning
neither Arm2-vs-Arm1 nor Arm2-vs-Arm2′ under the round-2 scheme was ever a
valid discriminator between "the mechanism did something" and "the blend
arithmetic did something," regardless of which arm was on which side.

**The fix is a redesigned primary observable, not a patched bar.** §7.1 now
compares **Arm 2 (trained THROUGH the bias for the full step budget, then
measured) against Arm 1′ (Arm 1's own checkpoint — trained WITHOUT the bias
— with the IDENTICAL blend applied ONLY at eval/measurement time, same
frozen table, same λ, same code path)**. Both sides of this comparison pass
through the exact same mechanical pin by construction; any difference
between them cannot be blend arithmetic (that is now common to both sides)
and must instead come from what training did differently to the pre-blend
key population. This is validated, not assumed: a new CPU sim
(`deltanet_rd/sim_frozen_bias_training_mediated.py`) builds a null case (no
injected training effect — proves the new bar produces an EXACT zero delta,
not merely a small one, when there is nothing to detect) and an injected
case (a synthetic training-mediated stabilization effect at several
strengths — proves the bar's sensitivity has teeth). Full numbers in §7.1.
**The design is NOT killed by this test** — the null case is provably exact
and the injected case is detected at a plausible effect size — but the sim
also surfaces a real, disclosed weakness (the post-blend comparison is
markedly LESS sensitive than measuring pre-blend `k_raw` geometry directly,
and higher λ further suppresses the post-blend signal), which is why
pre-blend `k_raw` geometry is promoted from "strongly recommended
secondary" to a **co-primary** observable (§4.a, §7.1).

### 1.1 What would CONFIRM this (ROUND-3: rewritten around the Arm-2-vs-Arm-1′ primary; ROUND-5 fix 1: CONFIRM criterion restated under the PINNED CI formula, §7.1-real.1 — RUNG-1-ONLY per this round's descope, §6/§8/§9)

- **Arm 2 (trained through the bias) clears the pre-registered bar against
  Arm 1′ (Arm 1's own checkpoint, blended ONLY at eval time, §5/§7.1) by a
  margin outside the cross-seed noise floor**, with the effect direction
  consistent across corpora (or a disclosed, non-cherry-picked split — see
  §7 bars). Because both sides of this comparison share the identical
  mechanical blend, clearing this bar can only reflect a difference in what
  SGD did to the PRE-blend key population under the bias — it cannot be
  blend arithmetic, by construction (§7.1's null-vs-injected sim proof).
  **ROUND-5 fix 1, operative CONFIRM criterion, PINNED formula
  (§7.1-real.1): "clears the bar" now means the measured
  `span_frac(Arm 2) − span_frac(Arm 1′)` delta's own two-sided CI —
  `mean_delta ± t(2,0.975)·s_ref/√n` at n=3, `t(2,0.975)=4.303` — EXCLUDES
  zero, in the mechanism-predicted (negative) direction, on BOTH corpora
  independently. The pinned per-corpus thresholds this CI must clear
  (equivalently, the CI half-width that must not straddle 0):
  **openr1-mix-ext: 0.0546; wikitext-mix-ext: 0.1064** (§7.1-real.1's own
  derivation from this program's real archived per-corpus stds). Per
  §7.1-real's own power analysis, NEITHER tested CPU-sim proxy family
  clears either pinned threshold at any tested effect size — the
  UNDER-POWERED-AGAINST-REAL-NULL verdict stands under this pinned
  formula, meaning the most likely rung-1 outcome is a CI that does NOT
  exclude zero (an estimation-mode null), not a proxy-validated CONFIRM
  (see the plain-language outcome statement, §1.0 below). Rung-1 is this
  design's ENTIRE committed scope this wave — rung-2 is PARKED (§6.2,
  §8.1).**
- **Pre-blend `k_raw` geometry (co-primary, promoted this round — §4.a,
  §7.1) moves in the same direction and is at least as sensitive an
  instrument as the post-blend comparison** — the round-3 sim shows the
  pre-blend comparison is UNIFORMLY MORE sensitive than the post-blend one
  at every tested effect size and every λ in the mini-sweep grid (§7.1's own
  numbers), so a real training-mediated effect should show up here at least
  as clearly as in the post-blend primary. A CONFIRM where post-blend moves
  but pre-blend does not is treated as suspicious, not automatically
  accepted (§1.3).
- The effect is NOT explainable by a position-only or vector-only shortcut
  (§9 attack question 1) — i.e., an ablation that removes the "keyed" (per-token
  identity) character of the bias while preserving its constancy still loses
  to the full construction, OR the design demonstrates by construction that no
  such shortcut is available at the observable's measurement site (closing the
  loophole the standing rule about position-decomposition warns against).
  **Operationally (attack finding 2, §5/§6/§7.1a, RE-ROLLED this round under
  the same Arm-1′-style control): Arm 2′ (trained through the global-vector
  bias) must beat Arm 1″ (Arm 1's checkpoint, global-vector-blended ONLY at
  eval time) by LESS than Arm 2 beats Arm 1′ — i.e. the per-token-keyed
  construction's training-mediated effect must be larger than the
  generic-constant-vector construction's training-mediated effect, measured
  on the SAME artifact-free footing.** If Arm 2′-vs-Arm-1″ matches or
  exceeds Arm 2-vs-Arm-1′, the effect is not about per-token keying — see
  below.
- **λ is not badly mismatched (attack finding 3, §5/§6.1/§8):** the rung-1
  λ∈{0.3, 0.58, 0.8} mini-sweep, read under the SAME Arm-2-vs-Arm-1′ family
  at each λ, does not show a *different* λ producing a materially larger
  effect than 0.58 while 0.58 itself is null. **Disclosed, pre-registered
  expectation from the sim (§7.1): the post-blend bar's sensitivity itself
  falls monotonically as λ rises (a fixed synthetic training effect produced
  a mean detected delta of −0.0099 at λ=0.3, −0.0027 at λ=0.58, −0.00009 at
  λ=0.8, same effect_strength=0.2 throughout) — a null reading at λ=0.8 is
  therefore expected to be LESS informative than a null reading at λ=0.3,
  purely from this sensitivity gradient, independent of whether λ=0.8 is
  mechanistically better or worse. The mini-sweep's readout (§1.2a) is
  interpreted with this gradient in hand, not read as if all three λ cells
  had equal power.**

### 1.2 What would REFUTE this (ROUND-3: rewritten around the Arm-2-vs-Arm-1′ primary)

- Arm 2 does not clear its bar against Arm 1′ at either rung, OR clears it
  only within a margin indistinguishable from the cross-seed noise floor (a
  null result, reported honestly — this program pre-registers that outcome
  as informative, not a failure to hide), **AND pre-blend `k_raw` geometry
  (the co-primary, more sensitive instrument) is ALSO null, AND the
  λ∈{0.3, 0.58, 0.8} mini-sweep (§5/§6.1), read under the same Arm-2-vs-
  Arm-1′ family, is itself flat at every λ — this is a clean REFUTE.** Note
  the AND: because the post-blend bar is the LESS sensitive of the two
  co-primary instruments (§7.1), a REFUTE now additionally requires the
  more sensitive pre-blend instrument to also be null — this is a stricter,
  not laxer, REFUTE condition than the round-2 text, closing exactly the
  gap a weak/rare instrument would otherwise leave (a REFUTE claim resting
  solely on the less sensitive of two available observables would itself be
  a weak claim).
- The effect exists but is reproduced identically (on the SAME Arm-1′-style
  artifact-free footing) by the generic-constant-vector control — i.e. Arm
  2′-vs-Arm-1″'s training-mediated delta matches or exceeds Arm 2-vs-Arm-1′'s
  — this would mean the LM-scale gain is a generic regularization/norm
  effect of training through ANY constant additive bias, not a transplant of
  the constancy-suffices mechanism this program is supposed to be testing.
  **Operationalized as the mandatory Arm-2′-vs-Arm-2 comparison (§5, §7.1a,
  re-rolled this round): Arm 2′'s OWN training-mediated delta (vs. its own
  Arm-1″ control) matching or exceeding Arm 2's training-mediated delta (vs.
  Arm 1′) is this REFUTE reading, not a partial win for either arm.**

### 1.2a λ-mismatch reading (attack finding 3 — a THIRD, distinct outcome, not folded into CONFIRM/REFUTE; ROUND-3: re-rolled under the new primary, sensitivity gradient disclosed; ROUND-5 fix 6: family B's own λ-gradient folded in, previously derived from family A alone)

If Arm 2 (λ=0.58) is null against Arm 1′ at rung-1's primary bar, BUT the λ
mini-sweep shows a *different* grid value (0.3 or 0.8) clearing the
Arm-2-vs-Arm-1′ bar at that λ while 0.58 does not, this is reported as
**"λ-mismatch, mechanism present"** — explicitly NEITHER a clean CONFIRM
(0.58 was pre-registered as primary and did not clear) NOR a clean REFUTE
(the mechanism moved geometry at some λ). This reading is pre-registered now,
before any run, precisely so it cannot be constructed post-hoc to rescue a
null 0.58 result. **Round-3 addition, mandatory to disclose alongside any
λ-mismatch reading:** because the bar's own sensitivity falls as λ rises
(§1.1's sim numbers), a λ=0.3 cell clearing the bar while λ=0.58 does not is
weaker evidence of a genuine λ-mismatch than the reverse (λ=0.8 clearing
while λ=0.3 and λ=0.58 do not would be surprising given the sensitivity
gradient, and correspondingly stronger evidence) — any λ-mismatch report
must state which direction the mismatch runs and note whether it is
consistent with, or contrary to, the sensitivity gradient, not report the
bare pass/fail pattern alone.

**ROUND-5 fix 6 addition: this sensitivity gradient is NOT specific to
family A.** Until this round, the λ-sensitivity-falls-as-λ-rises gradient
above was derived ONLY from family A (`shared_dirs`) — §7.1a-real/§11b
question 3 disclosed this as an open scope gap. Family B
(`anchor_directed`, the mechanistically-motivated family) was swept at the
same λ∈{0.3, 0.58, 0.8} grid this round (§11b-3, resolved) and shows the
SAME QUALITATIVE PATTERN, confirming the gradient is a property of the
blend arithmetic itself, not an artifact of family A's own particular
construction: minimal detectable effect (interpolated crossing of the
`mean_ref+2·s_ref` synthetic bar) is **0.036 (λ=0.3), 0.077 (λ=0.58), 0.494
(λ=0.8)** for family B — versus family A's own 0.177/0.284/0.862 — the
SAME monotonically-worsening-with-λ shape, just uniformly more sensitive
throughout (family B detects real effects at roughly 3–5× smaller
`effect_strength` than family A at every λ in the grid). **A NEW,
family-B-specific disclosure, stronger than anything family A showed
within its own mini-sweep λ range: at λ=0.8, family B's own detected delta
flips POSITIVE starting at the very first tested nonzero `effect_strength`
(0.05: `+0.00003`; every larger `effect_strength` tested is also positive
and growing, e.g. `+0.00275` at `effect_strength=1.0`)** — unlike family
A, whose own sign reversal only appears well past the mini-sweep's own
λ=0.58/0.8 relevance (at `effect_strength≥0.7`, §7.1a-real), family B's
λ=0.8 cell is ALREADY on the wrong side of zero at the smallest tested
effect size. This means a λ=0.8 null (or even a small positive) reading at
rung-1 is closer to being an EXPECTED, low-information outcome under
family B's own proxy than a genuinely surprising one — any λ-mismatch
report involving the λ=0.8 cell must state this explicitly, not treat a
λ=0.8 result at face value.

### 1.3 What is UNINTERPRETABLE (must be flagged, not spun)

- Any result where the val-loss tolerance gate (§7) is violated in the
  direction of the frozen-bias arm being **better** on loss alone without the
  primary observable also moving — a loss-only win could be a generic
  regularization/shrinkage artifact of adding *any* constant vector to keys,
  not evidence for the specific constancy mechanism.
- **(ROUND-3, new) Post-blend Arm-2-vs-Arm-1′ clears its bar while pre-blend
  `k_raw` geometry (the co-primary, MORE sensitive instrument per §7.1) does
  NOT move outside its own noise floor.** Per §1.1, the pre-blend instrument
  is expected to be at least as sensitive as the post-blend one at every
  tested effect size in the validating sim — a post-blend-only "win" inverts
  that expectation and must be flagged as suspicious (possible residual
  artifact this design's sim did not anticipate, or a real but
  instrument-specific effect requiring its own explanation) rather than
  reported as a clean CONFIRM.
- Any result that depends on choosing which corpus's split to report after
  seeing both (the same "requires the same outcome in both corpora, a split
  is reported as INCONCLUSIVE overall" rule Track B's own redesign used,
  `TRACKB_REDESIGN.md` §5.1, reused verbatim here — see §7).
- Any result from a rung whose calibration run (§6) was skipped — per the
  standing calibration-before-sweep rule, no sweep-scale claim is licensed
  without one.
- **(ROUND-3, new — the fourth pre-registered outcome, round-2 Finding 2)
  Sim-training divergence: informative, not REFUTE.** The round-2 attack's
  second FATAL finding was that the sim blends a STATIC population post-hoc,
  while the real Arm 2 trains `k_raw` through the bias for 15k+ steps, and
  SGD could cancel, exploit, or amplify the bias in ways no static sim can
  anticipate. If the real LM comparison behaves in a way inconsistent with
  every sim prediction here (e.g. the pre-blend co-primary moves opposite to
  the post-blend primary, or the λ-sensitivity gradient is violated), this is
  reported as **"sim-training divergence"** — explicitly NOT a REFUTE and NOT
  silently absorbed into a CONFIRM either, but its own disclosed, informative
  outcome. **Cheap diagnostic, pre-registered now:** compute the cosine
  similarity of each trained token's raw key (`k_raw`, pre-blend) against
  that token's own frozen anchor row `B[token_id]`, BEFORE training (at
  init) and AFTER training, for both Arm 2 and Arm 1. A rising cosine in Arm
  2 only (not Arm 1) is direct evidence SGD is actively aligning raw keys
  toward the frozen anchors — i.e. compensating for, rather than ignoring,
  the bias — and should be reported alongside any sim-training divergence
  finding as the first place to look for why the sim's static-population
  proxy diverged from the trained result.

---

## 2. The central design decision: decouple the frozen bias from geo3/Newton-Schulz

This is the single most consequential fact this design had to resolve before
anything else, and it is why this program is NOT "port Track B's geo3-in-LM
plus anchoring and try again."

**The as-coded mechanism (`model_rd.py`) hard-couples the anchor bias to
geo3.** The assert at `model_rd.py` (~line 812) reads: `anchor_active REQUIRES
geo3_active` — "anchoring is defined as a modification to geo3's own write
path, not a replacement for it." The blend itself
(`key_anchoring.anchor_blend_gather_scatter`) operates on `k_eff_raw`, the
pre-Newton-Schulz gathered key for a batch of **K discrete named entities per
episode** (`batch["key_ids"]`, drawn from `pools.train_name_ids` — a
closed, ~107-entity vocabulary specific to the synthetic K-cycle grammar).
There is no notion of "episode," "entity," or "K items per binding" in a
free-running LM token stream.

**Porting the anchor mechanism into an LM by reusing `geo3_active`'s own
insertion site inherits Track B's own graveyard whole.** §3 below restates
exactly what killed Track B: (1) the LM's own learned β is measurably
near-uniform (Gini≈0.099), so there is no natural small write-worthy subset
to gather/orthogonalize/anchor in the first place; (2) even forcing a
hard-selected subset (`TRACKB_REDESIGN.md`'s own build) hit a **duplicate-key
numerical-stability wall** inside Newton-Schulz at production LM scale
(`skip_rate=0.6319` vs. a `≤0.01` bar, with a probative positive control —
196/326 forward calls carrying ≥6 duplicated selected key rows, `max_dup_per_call`
reaching 32/32). Both barriers are properties of the **geo3/Newton-Schulz
write-time orthogonalization step**, not of the anchor-bias blend itself.

**This design's resolution: strip the mechanism down to its own
confirmed-load-bearing core and skip geo3 entirely.** §10.13.4/§10.14's own
mechanistic account (verified directly against `KEY_ANCHORING_DESIGN.md`) is
explicit that constancy in the blend arithmetic — not geo3, not entity
alignment, not Newton-Schulz — is the active ingredient: *"stability arrives
from the blend's arithmetic structure, not from the raw key learning to align
with a fixed target"* (§10.13.4), confirmed by candidate (e) matching or
beating the learned table with geo3 held fixed in both arms (§10.14). Geo3
was present in every synthetic-grammar cell only because that is where the
mechanism was discovered (the anchor blend feeds geo3's own orthogonalization
pass downstream) — the ablation's own logic never required geo3 to be present
for the constancy effect itself.

**Concretely, this program builds a NEW, minimal hook, independent of
`geo3_active`, `hard_select_active`, and `TRACKB_REDESIGN.md`'s entire
selection-mechanism family:**

```
k_biased = normalize((1 - λ) · k_raw + λ · B[token_id])
```

applied directly to every token's key in `DeltaNetLMMixer.forward`
(`lm_pretrain_rd.py`), where `B` is a frozen, never-trained `(vocab_size,
d_state)` table (candidate-(e)'s own `random_unit_rows_init`, same
construction, same RNG convention), and `λ` is fixed at 0.58 (§10.13.4's
own cross-cell mean) with NO learned λ arm as primary (see Arms, §5). This
is applied to **every token position**, not a selected top-K subset — there
is no selection step, no orthogonalization step, and therefore neither of
Track B's two barriers (β-uniformity, duplicate-key NS instability) applies:
this construction never calls Newton-Schulz and never gathers a subset by β
magnitude. It is a strictly simpler intervention than anything Track B or
its redesign attempted.

**This is a genuinely new build, not a config flag.** `_geo3_lm_select_and_
orthogonalize` and the anchor blend as coded both assume a `key_ids` batch
field that free text does not have. The new hook needs: (a) a frozen
`(vocab_size, d_state)` embedding table, built once via
`random_unit_rows_init` (reusing `key_anchoring.py`'s existing function,
`requires_grad_(False)` on the whole table — no `anchor_trained_mask`/
`t_idx` gather needed since every token gets biased, not a masked subset);
(b) one line in the forward pass blending it into the post-conv key before
`chunk_delta_rule`; (c) the fixed-λ scalar (a Python float, not even an
`nn.Parameter` in the primary arm). This is deliberately far smaller than
Track B's own build (no selection logic, no STE, no chunk gather/scatter).

---

## 3. Track B's own graveyard, restated exactly (so this design does not re-enter it)

Verified directly from `TRACKB_REDESIGN.md` and `SCALE_TRANSFER_DESIGN.md` §4:

1. **The original geo3-in-LM Wave −1 gate returned a hard NO-LAUNCH**
   (`EXPERIMENT_LOG.md` entry title, quoted in `TRACKB_REDESIGN.md` §0: *"Wave
   −1 gate measured on all 6 archived Wave C checkpoints — HARD NO-LAUNCH"*).
   The measured numbers (`TRACKB_REDESIGN.md` §1, `wave_neg1_gate.json`):
   top-K_sel β-mass fraction 0.309 (K_sel=16) / 0.569 (K_sel=32) vs. a ≥0.60
   bar (FAIL at both); mean β at non-selected positions 0.387/0.363 vs. a
   ≤0.25 bar (FAIL); non-selected write-mass fraction 0.691/0.431 vs. a ≤0.40
   bar (FAIL). Gini≈0.099 at both K_sel — the LM's own learned β is close to
   uniform, not concentrated on a small write-worthy subset. **Criterion (b)
   failing is registered as a hard no-launch regardless of criterion (a)** —
   Track B's own §4.2 routing table.
2. **The redesign (hard top-K selection with STE) built past that gate and
   hit a second, independent barrier: numerical instability under duplicate
   keys.** The M8 stability smoke (`TRACKB_REDESIGN.md` §14.1) ran 163/2,000
   requested steps before the harness's own skip-rate accounting stopped it:
   `n_skipped_steps=103`, `skip_rate=0.6319018404907976` — **63× over the
   ≤0.01 bar**, with a probative (not vacuous) positive control:
   `n_calls_total=326`, `n_calls_meeting_floor=196` (≥6 duplicated selected
   rows), `max_dup_per_call` reaching 32/32 repeatedly. Cells 3/4 (the
   geo3-active arms) never launched as a result; the interaction/headline bars
   became **uncomputable, not merely unevaluated** (§14.1's own wording).
3. **Even the selectivity-only main effects that DID run (no geo3) returned
   an INCONCLUSIVE headline** (§14.6): val-loss tolerance failed on openr1 for
   all three selectivity arms (hard selectivity itself implicated, not an
   STE-specific artifact, since a soft-top-K comparator failed alongside the
   hard-STE candidate); passed on wikitext for all three. Same-instrument Gram
   deviation showed a real Cell-2-vs-random-control separation on openr1 but
   only a marginal Cell-2-vs-baseline gap; on wikitext the reverse. Per Track
   B's own pre-registered rule ("requires the same outcome in both corpora; a
   split is reported as INCONCLUSIVE"), the combined verdict is INCONCLUSIVE.

**Why this program does not walk into the same trap:** both of Track B's
barriers are specific to inserting **Newton-Schulz orthogonalization over a
selected top-K subset of a chunk**. This program's construction (§2) has no
selection step and no orthogonalization step — every token gets the same
fixed-λ frozen additive bias, applied densely, with no gather/scatter over a
β-nominated subset and no NS solve anywhere in the forward pass. The
β-uniformity finding (barrier 1) is irrelevant here because this design never
asks β to nominate anything. The duplicate-key NS instability (barrier 2) is
irrelevant because there is no NS call. **This program's build risk is
categorically smaller than Track B's, precisely because it discards the part
of the mechanism that Track B's own evidence shows is LM-hostile (selection +
orthogonalization) and keeps only the part the ablation confirmed is
load-bearing (constancy in an additive blend).**

---

## 4. The observable — this is the hard part

The K=32-gain analog in the synthetic grammar was `rec@0.9` at h=4: exact,
continuous, cosine-thresholded recovery of a specific entity's bound value
after 4 compositional hops, measured against a provable `rank≥K` necessity
argument. **No such provable-necessity task exists natively in free-running
text**, and Track B's own experience shows that inventing one on top of an
LM's continuous, unmasked internals is exactly where things go wrong (β
turned out not to concentrate the way the design assumed). Three options,
weighed explicitly:

### 4.a Attractor-probe instrument (from the scale program) — RECOMMENDED PRIMARY (ROUND-3: now paired with a co-primary, §4.a-i)

The already-validated, corpus-matched-pooling key-Gram-deviation /
`span_frac` instrument from `SCALE_TRANSFER_DESIGN.md` (Track C), reused
unmodified. Concretely: `key_gram_deviation_mean` (`‖K_eff^T K_eff − I‖_F`,
pooled per chunk) and its normalized `span_frac` (the cross-scale-comparable
quantity Fig. 9's own regeneration note specifies — verified directly:
`archived4_span_frac` is the convention because raw `gram_deviation_mean` is
NOT comparable across `d_state`). **Why this is the strongest choice:**

- **Already validated at exactly this scale.** The 14M control-cell
  measurement already exists (`SCALE_TRANSFER_DESIGN.md` §5.10 table:
  `14,048,896` params, raw Gram deviation `21.74 ± 5.80`, `span_frac 0.248`,
  12,288 chunk-episodes, `d_model=256/n_layers=2/d_state=64`) — this design's
  baseline arm is a literal re-run of an already-measured, already-instrumented
  configuration, not a new instrument built from scratch.
- **It measures the SAME quantity the constancy-suffices mechanism is
  supposed to affect.** The synthetic-grammar mechanism account (§10.13.4)
  is explicitly about **cross-episode key stability / write-time key
  geometry** — key-Gram deviation is a direct geometric readout of exactly
  that quantity, transplanted with zero new invention.
- **Weakness, disclosed:** it is Tier-2 descriptive/interventional, never
  causal, and it does not by itself demonstrate "the LM got better at
  anything a user would recognize" — a geometry change without a behavioral
  correlate is a weaker demo than a recall task would be. This is why a
  secondary is registered (§4.c-lite, below) rather than relying on 4.a alone.
- **ROUND-3 correction, essential: measured on POST-BLEND keys, this
  instrument is now compared Arm-2-vs-Arm-1′ (§7.1), NOT Arm-2-vs-Arm-1
  directly.** The round-2 attack's FATAL finding showed that comparing
  post-blend Arm 2 against never-blended Arm 1 is an auto-pass artifact —
  the blend arithmetic itself, applied once at eval time to a static
  population, dominates the measured `span_frac` regardless of the baseline
  key population's own structure (§7.1's crossover-scan evidence). Arm 1′
  (Arm 1's own checkpoint, blended ONLY at measurement time, same code path)
  is the artifact-matched reference this instrument must be read against.

### 4.a-i Pre-blend `k_raw` geometry — NEW, promoted to CO-PRIMARY this round (round-2 Finding, "strongly recommended" → mandatory)

**The SAME `span_frac` instrument, applied to the UN-blended keys
(`k_raw`, captured before this program's own bias-blend hook, both arms) —
no blend is present anywhere in the measured population, so no mechanical
pinning artifact is possible by construction.** This was "strongly
recommended" in the round-2 revision brief and is promoted here to a
CO-PRIMARY observable, not merely a nice-to-have secondary, for a concrete,
evidence-backed reason: the round-3 validating sim
(`sim_frozen_bias_training_mediated.py`, §7.1) shows the pre-blend
comparison is UNIFORMLY MORE SENSITIVE than the post-blend Arm-2-vs-Arm-1′
comparison at every tested synthetic effect size and every λ in the
mini-sweep grid — e.g. at a fixed injected effect (`effect_strength=0.2`,
N=3 seeds matching this program's own per-arm seed count), the pre-blend
delta-to-noise ratio is roughly 3× the post-blend one (raw mean
`−0.0121`/std `0.0041` vs. post-blend mean `−0.0027`/std `0.0014`; full
numbers in §7.1). **Direction, pre-registered from the mechanism account
(§10.13.4), not left exploratory:** if training through the bias produces
the constancy-suffices stabilization the mechanism account predicts, Arm
2's OWN `k_raw` (pre-blend) should show LESS non-orthonormal drift relative
to Arm 1's own `k_raw` than would be expected from seed noise alone — i.e.
the SAME one-sided "falls" direction as the post-blend primary, measured on
an artifact-free population. A CONFIRM requires both the post-blend primary
(§7.1) and this co-primary to move in this same direction outside their
respective noise floors (§1.1); a split between them is flagged per §1.3,
not silently resolved in favor of whichever instrument happened to clear
its bar.

### 4.b Held-out perplexity deltas — REJECTED as primary, kept only as a gate

Held-out validation loss is already collected by the harness for free at
every checkpoint. **Why this is weak/confounded as a PRIMARY observable:**
adding any additive constant to every key changes the key distribution's
norm and direction statistics globally; a loss improvement from that alone
is indistinguishable from a generic regularization/shrinkage effect (e.g.,
biasing every key toward a fixed random direction slightly reduces the
effective key-space dimensionality used per token, which can smooth
training dynamics in a way that has nothing to do with cross-episode
stability). Track B's own experience is the direct precedent: its
selectivity arms showed a real geometry change (Cell 2 vs. Cell 1 Gram
deviation) alongside a **loss regression** on one corpus and no reliable
loss change on the other — geometry and loss did not move together, and
loss alone would have given the wrong headline. **Retained role: a
gate, not a claim.** §7 registers a val-loss tolerance (mirroring Track B's
own +5%-relative convention) that the frozen-bias arm must clear to avoid
landing in the "uninterpretable, loss-only" bucket of §1.3 — but crossing
it is necessary, not sufficient, for a positive result.

### 4.c Synthetic multi-hop recall suite injected into eval — REJECTED as primary this wave, registered as a documented follow-on

The strongest observable in principle (it would give something closer to
the synthetic grammar's own `rec@0.9`), but it inherits exactly the trap
this program's own standing instruction calls out: Track B's `TRACKB_REDESIGN.md`
own churn/skip_rate machinery is the direct evidence that bolting a
closed-vocabulary recall task onto a free-running LM's own token/embedding
space is a nontrivial, failure-prone construction (§14.5's positional-
concentration and support-size bands, the entmax support-collapse finding —
median support 18–20 of a nominal 32 — are all instances of a synthetic-task
transplant behaving in ways the design had to discover empirically, not
assume). Building a bespoke multi-hop recall eval for this specific program,
under this budget, repeats that risk for a program whose entire point is to
demonstrate a SIMPLER, lower-risk fix than Track B's. **Not attempted this
wave.** If rung-1 passes its primary+secondary bars (§7) and a follow-on
program is authorized, a hop-recall suite adapted from the synthetic
grammar's own K-cycle construction (transplanted onto the pretrained LM's
embedding table, following `SCALE_TRANSFER_DESIGN.md` §5.5 item 2's own
"frontier-probe transplant" precedent and its own mandatory validation step —
require non-trivial recovery on an unconstrained baseline before trusting any
number) is the natural next step, explicitly out of scope here.

### Recommendation (ROUND-3, rewritten): co-primary = 4.a (post-blend Arm-2-vs-Arm-1′) + 4.a-i (pre-blend `k_raw`), secondary = 4.b (val-loss tolerance, gating not headline)

Pre-registered before any run: the frozen-bias arm's headline claim rests on
BOTH §4.a's post-blend `span_frac` delta (Arm 2 vs. its own artifact-matched
Arm 1′ control) AND §4.a-i's pre-blend `k_raw` `span_frac` delta (Arm 2 vs.
Arm 1, no blend present, no artifact possible), at pre-registered bars (§7)
— not §4.a alone, per the round-2 attack's FATAL finding that a bar built
from post-blend `span_frac` against a never-blended reference auto-passes
regardless of training. Val-loss (§4.b) is a **necessary gate**: if the
frozen-bias arm's val loss regresses beyond the pre-registered tolerance,
the geometry result is reported as a real-but-costly trade-off, explicitly
per §1.3, never spun as an unqualified win.

---

## 5. Arms

All arms share: identical corpus (Wave C's own `reasoning_mix_eot_extended`
and `wikitext103_mix_eot_extended`, the SAME corpora used throughout
`SCALE_TRANSFER_DESIGN.md` Track C, per the standing "use the same dataset
for ALL experiments in a comparison" rule), identical tokenization (GPT-2
BPE, `vocab_size=50257`, held fixed per the standing "hold tokenization fixed
when testing a primary hypothesis" `[LEARN]`), identical architecture
(`d_model=256, d_state=64, n_layers=2, conv_size=4, num_heads=1`), identical
seeds per corpus, identical step budget per rung, identical checkpoint
cadence (every 1,000 steps, matching Wave C's own convention).

### Arm 1 — Baseline LM (no anchor bias)

Exactly Wave C's own architecture and training recipe, re-run fresh at this
program's own step budget (not merely cited from the archive — see §6,
calibration run) so that every arm in this specific comparison comes from
the identical training-harness invocation, avoiding any cross-session
harness-drift risk. `anchor_active=False`, no new code path exercised beyond
what Wave C's own archived cells already used.

### Arm 1′ — Arm 1's own checkpoint, dense per-token blend applied ONLY at eval time (NEW this round — the primary comparator, round-2 Finding 1 fix)

**Not a new training run.** Arm 1′ is Arm 1's own checkpoint (same weights,
same seed, same corpus, same step) with the IDENTICAL post-conv-key blend
Arm 2 uses (`k_biased = normalize((1-λ)·k_raw + λ·B[token_id])`, same frozen
table `B`, same `λ=0.58`, same insertion point, same code path) applied ONLY
at the measurement/eval pass — never during training, never affecting the
gradient Arm 1 itself received. **Purpose: this is now the PRIMARY
comparator for Arm 2 (§7.1), replacing the round-2 Arm-2-vs-Arm-1 comparison
that the round-2 attack killed as an auto-pass artifact.** Because Arm 2 and
Arm 1′ are measured through the EXACT SAME mechanical blend, any post-blend
`span_frac` difference between them cannot be the blend arithmetic itself
(which is now common to both) — it can only reflect a difference in the
PRE-blend key population, i.e. what training did differently when the model
could see/adapt to the bias (Arm 2) versus when it could not (Arm 1). This
is the training-mediated comparison the mechanism claim is actually about.
**Cost: eval-only, near-zero GPU-h** (one forward-hook measurement pass per
existing Arm-1 checkpoint per corpus/seed — no additional training cell;
§8 budget). **Validated, not assumed:** `sim_frozen_bias_training_mediated.py`
(§7.1) proves this comparison produces an EXACT zero delta when there is no
training-mediated difference to detect (the null-case falsifiability proof)
and detects a synthetic training-mediated effect at a plausible strength
(the sensitivity falsifiability proof) — see §7.1 for full numbers.

### Arm 1″ — Arm 1's own checkpoint, global-single-vector blend applied ONLY at eval time (NEW this round — the Arm-2′ comparator, generalizing the Arm-1′ fix)

**Not a new training run**, same discipline as Arm 1′: Arm 1's own
checkpoint, with Arm 2′'s blend (`k_biased = normalize((1-λ)·k_raw +
λ·b_global)`, same derived `b_global`, same `λ`) applied ONLY at eval time.
**Confirmed this round, not assumed:** the same mechanical-pinning artifact
that killed the round-2 Arm-2-vs-Arm-1 bar exists for the global-vector
construction too, and MORE severely — a CPU check this round (§7.1a) shows
the global-blended population pins to a `span_frac` band of ~0.61–0.71
across the full baseline-structure sweep (`shared_frac` 0.0→0.7), an even
narrower and more extreme collinearity artifact than Arm 2's own
~0.02–0.09 band, since blending every key toward ONE single fixed direction
is a more severe rank-1 collapse than blending toward one of 50,257 distinct
rows. **This means the round-2 Arm-2-vs-Arm-2′ comparison was NEVER a valid
discriminator, regardless of which arm was nominally the "control"** — both
sides were dominated by their own target geometry's mechanical pin. Arm 1″
fixes this the same way Arm 1′ fixes the Arm-2 side: Arm 2′ (trained through
the global-vector bias) is now compared against Arm 1″ (never-bias-trained,
blended only at eval), isolating the training-mediated effect from the
identical blend arithmetic. **Cost: eval-only, near-zero GPU-h**, reuses Arm
1's checkpoints exactly as Arm 1′ does (§8 budget).

### Arm 2 — Frozen random anchor table at fixed λ≈0.58 (PRIMARY intervention)

The §2 construction: `B` = a `(50257, 64)` table built once via
`key_anchoring.random_unit_rows_init(n=50257, d=64, seed=ANCHOR_INIT_SEED)`
(the SAME function candidate (e)'s arm 'e' used, extended from the
~107-row trained-entity table to the full 50,257-row GPT-2 vocabulary — every
token gets a frozen random unit-row bias, not just a closed entity set, since
free text has no natural "trained entity" subset), `requires_grad_(False)`
on the whole table, `λ` fixed at **0.58** (§10.13.4's own registered
cross-cell mean value from the mechanism-tier wave — verified: "at an
SGD-preferred interior λ≈0.55-0.60 (measured, this wave, all 7 cells)").
Applied to every token's post-conv key before `chunk_delta_rule`, per token
identity (`B[token_id]`), every forward pass, every position — no
episode/entity/selection machinery. Zero new trainable parameters in the
bias term; the frozen table itself is new state (50,257×64×4 bytes ≈ 12.9MB
of buffers, non-trainable — see §5.1 for the param-matching convention this
implies).

### Arm 2′ — Global single frozen vector, no token lookup (MANDATORY control, promoted from attack-round prose — attack finding 2)

**Promoted from attack-round prose to a mandatory registered arm, both
rungs.** Same frozen table `B` as Arm 2 (SAME construction, SAME seed —
`key_anchoring.random_unit_rows_init(n=50257, d=64, seed=ANCHOR_INIT_SEED)`),
but instead of a per-token lookup `B[token_id]`, every position at every
step is blended toward the SAME single fixed vector: `B`'s own row-mean,
re-normalized to a unit vector (`b_global = normalize(mean_i B[i])`),
applied as `k_biased = normalize((1-λ)·k_raw + λ·b_global)` at the identical
λ=0.58, identical insertion point, identical every-position density as Arm
2. **Zero new trainable parameters** (same convention as Arm 2 — `b_global`
is derived once from the frozen table, itself never trained); total
parameter count is IDENTICAL to Arm 1 (no per-token table needed at all,
since `b_global` is a single `(64,)` vector, not a `(50257,64)` buffer —
disclosed as an even smaller footprint than Arm 2, reported per §5.1's
convention). **Purpose (mandatory, not optional):** isolates whether any
Arm-2 effect is about the bias being *keyed by token identity* at all, or
merely about adding *any* constant vector to every key (the attack round's
Q1 risk). **Licensing condition (§7.1a, §1.1, RE-ROLLED this round under the
Arm-1″ control, round-2 Finding 1's fix generalized):** the transplant claim
("this is a transplant of the constancy-suffices mechanism") is licensed
ONLY if Arm 2's OWN training-mediated delta (vs. its Arm-1′ control) exceeds
Arm 2′'s OWN training-mediated delta (vs. its Arm-1″ control) — NOT "Arm 2
beats Arm 2′" directly, since Arm 2 and Arm 2′ are pinned to structurally
different post-blend bands (Arm 2 ~0.02–0.09, Arm 2′ ~0.61–0.71) that are
not comparable on a shared numeric scale; comparing their OWN
training-mediated deltas (each relative to its own artifact-matched control)
is the only apples-to-apples reading — see §7.1a. This arm is NOT cuttable
under budget pressure (§8's cut order takes Arm 3 first, then the rung-2
optional scope — Arm 2′ is core to both rungs).

### Arm 3 — OPTIONAL learned-λ arm, ONLY if budget allows

Same frozen table as Arm 2, but λ is a single learned scalar
(`sigmoid(raw_param)`, `raw_param` init 0.0 → λ=0.5, mirroring
`model_rd.py`'s own `anchor_lambda_mode="learned"` convention exactly) rather
than fixed at 0.58. Adds exactly 1 trainable parameter. **Purpose:** a cheap
internal check on whether this program's own fixed-λ choice (0.58,
transplanted from the synthetic grammar) is close to what free text's own
SGD dynamics would prefer, without claiming anything about entity alignment
(there are no entities here — this arm tests λ-transfer only, not the
rejected Outcome-C mechanism). **Cut first if budget is tight** (§8 cut
order) — this arm answers a secondary curiosity question, not the
program's primary hypothesis. **Distinct from, and does not substitute for,
the mandatory λ∈{0.3, 0.58, 0.8} fixed-grid mini-sweep below** — Arm 3 asks
"what would SGD itself prefer," the mini-sweep asks "is 0.58 specifically
mismatched," and both are registered because a null Arm 3 (learned λ
converges near 0.58) would not by itself rule out a badly-shaped, non-convex
λ-response surface the fixed grid can still catch.

### Arm 2-λ-mini-sweep — MANDATORY, rung-1 scope only (promoted from optional Arm-3-only coverage — attack finding 3; ROUND-3: readout re-rolled under the Arm-1′ primary, sensitivity gradient disclosed)

**Promoted from "Arm 3 is the only check on λ-transfer, and it is cut
first" to a mandatory rung-1 scope item, run alongside the calibration run
(§6.1, §6.3).** Arm 2's own construction (dense, per-token, frozen table),
at λ∈**{0.3, 0.58, 0.8}** — the design's own registered primary (0.58)
plus two brackets — **1 seed, 1 corpus (openr1-mix-ext)**, matching the
sim's own sensitivity grid (`sim_frozen_bias_direction.py` /
`sim_frozen_bias_training_mediated.py`, §7.1) exactly so the LM-scale sweep
and the CPU-sim sweep are directly comparable. **Purpose:** closes attack
finding 3 (λ≈0.58 was transplanted from a closed ~107-entity synthetic
grammar with a materially different loss landscape; there was no a priori
reason to expect 0.58 to be near-optimal for 50,257-way next-token
prediction). Each mini-sweep cell requires its OWN Arm-1′-style eval-only
control at the same λ (the blend's λ is shared between the Arm-2-analog and
its Arm-1′ control at every grid point — no additional training, per
Arm 1′'s own eval-only cost model).

**Readout, RE-ROLLED this round (the round-2 readout compared raw `span_frac`
values across λ against a fixed seed-noise floor; that comparison inherits
the exact auto-pass artifact this whole revision exists to fix — a mini-sweep
run under the killed bar would be equally uninterpretable). Under the new
scheme, each λ cell is read as its OWN Arm-2-vs-Arm-1′ comparison** (same
family as §7.1's primary, just at a different fixed λ), feeding §1.2/§1.2a's
three-way CONFIRM / REFUTE / λ-mismatch classification: a flat mini-sweep (no
λ's Arm-2-vs-Arm-1′ delta clears its own λ-matched noise floor) supports a
clean REFUTE if λ=0.58's own Arm-2-vs-Arm-1′ comparison is also null; a
mini-sweep where a DIFFERENT λ clears its bar while 0.58 does not is the
λ-mismatch reading, reported as such, never silently re-run at the "better"
λ as if it had been primary all along. **Mandatory disclosure alongside any
mini-sweep readout (§1.1/§1.2a, new this round):** the validating sim shows
the bar's own sensitivity FALLS as λ rises — a fixed synthetic
training-mediated effect (`effect_strength=0.2`) produced a detected
post-blend delta of −0.0099 at λ=0.3, −0.0027 at λ=0.58, and only −0.00009 at
λ=0.8 (§7.1 full numbers) — so a null result at λ=0.8 carries systematically
less evidentiary weight against the mechanism than a null result at λ=0.3,
purely from this sensitivity gradient. Any λ-mismatch or REFUTE reading must
state which λ cells were null and note this gradient, not treat all three λ
cells as equally powered tests.

### 5.1 Param-matching convention (decided and justified)

**A frozen table adds ~zero TRAINABLE parameters but nonzero TOTAL parameter
count (buffers).** Convention adopted, stated explicitly rather than left
implicit: **match on trainable parameter count** (the standard convention for
"does the fix cost anything to train" claims — a frozen buffer never
receives a gradient, never appears in the optimizer state, and never
increases AdamW memory, which is the practically load-bearing cost axis at
LM training scale). Report **both** counts in every result table: Arm 1 and
Arm 2 have IDENTICAL trainable-parameter counts (14,048,896 in both, since
Arm 2's added table is entirely non-trainable); Arm 2's TOTAL parameter count
(trainable + buffers) is higher by exactly `50257 × 64 = 3,216,448` (the
table's own element count) — disclosed as a memory-footprint fact, never
conflated with a capacity/expressivity claim, since a frozen buffer with no
gradient path is architecturally inert with respect to the model's learned
function class except through the one fixed additive term it contributes.
Arm 3 adds exactly 1 trainable parameter beyond Arm 2 (the λ scalar) —
reported separately, not rounded away. **Arm 2′'s trainable-parameter count
is IDENTICAL to Arm 1's (14,048,896) — its TOTAL parameter count is HIGHER
than Arm 1's by only 64 (the single `b_global` buffer), i.e. Arm 2′ is
*smaller* than Arm 2 in total footprint despite sharing the same source
table** (Arm 2′ never materializes the full `(50257,64)` table as a live
buffer at inference/training time — only the one derived vector — though
this design conservatively still counts the full table's construction cost
in Arm 2′'s Wave −1/build accounting, since the same `random_unit_rows_init`
call produces both `B` and `b_global`). **Arm 1′ and Arm 1″ (NEW,
eval-only) add ZERO parameters of any kind, trainable or buffer, beyond
what Arm 1 itself already has at inference time** — they are Arm 1's own
checkpoint with a blend applied transiently during the measurement pass
only, never saved as a distinct set of weights or a distinct checkpoint;
the frozen table `B` (or its derived `b_global`) used at measurement time is
the SAME buffer Arm 2/Arm 2′ already construct, not a new allocation
specific to Arm 1′/1″.

---

## 6. Scale, seeds, and the mandatory calibration run (ROUND-5 fix 3: committed scope is RUNG-1-ONLY this wave, §6.2/§8.1/§8.5.1 — rung-2 is PARKED)

### 6.0 Pre-registered seeds (attack finding 8)

**Literal seed integers, chosen now, verified distinct from every archived
run's own seeds.** A grep of every archived result JSON's own `"seed"` field
across `experiment-runs/` (this session) shows the following integers already
in use somewhere in this codebase's archives: `{0, 1, 2, 3, 4, 7, 10, 11, 12,
20, 21, 22, 30, 31, 32, 42, 60, 61, 62, 70, 71, 72, 900, 1337, 95000}`
(`corpus_fixed_seed(...) + 95_000`-style derived seeds excluded from this
list's "base" reading since they are a different seed namespace,
`lm_attractor_probe_rd.py`'s own eval-sampling convention, not a training
seed). **This program registers its own base training seeds as {0, 1, 2}**
— the SAME 3 integers Track C's own `run_lm_rd_trackc_sweep.py`
(`SEEDS = (0, 1, 2)`) and every archived rung/mixcontrol/wave-1ext cell in
`SCALE_TRANSFER_DESIGN.md` already used. **This is a deliberate, disclosed
choice, not an oversight:** reusing {0, 1, 2} is this codebase's own
established house convention for "N=3 seeds per corpus per arm" (the exact
convention this program's own §6.1 already cites), and per this design's own
§5's "identical seeds per corpus" requirement, matching Track C's own seed
integers means any accidental cross-arm data leakage (e.g. a corpus loader
bug that ignores the seed argument) would be caught by an anomalous
resemblance to ALREADY-ARCHIVED Track C numbers — a property a genuinely
novel seed choice would not have. The λ mini-sweep (§5) uses **seed=0 only**
(1-seed, pre-registered as such, §5's own "1 seed, 1 corpus" scope). Any
seed-count contingency addition (§8.5 headroom) draws the NEXT integer NOT
already in the archived-seed set above, in order: **{3, 4}** are themselves
already archived (48 and 44 occurrences respectively — corrected this round,
round-2 attack Finding 6: re-grepped `"seed":\s*3\b` over
`experiment-runs/` this session, exact command `grep -rho
'"seed":\s*3\b' experiment-runs/ | wc -l` → 48, not the pre-revision text's
43 — likely another program's own seeds either way, the exact count does
not change this program's own seed choice, only the disclosed provenance
number) — the first genuinely new integer in ascending order is **5** —
this program's own seed-contingency draw order is registered now as
**{5, 6, 8, 9}** (skipping every already-archived integer below 13), so a
contingency seed can never accidentally collide with any other program's
archived run.

### 6.1 Rung-1 (14M) — first wave

Reuses the exact architecture Wave C already validated: `d_model=256,
d_state=64, n_layers=2`, `n_params=14,048,896` (verified,
`SCALE_TRANSFER_DESIGN.md` §5.9/§5.10 tables). **N=3 seeds per corpus per
arm** (seeds {0,1,2}, §6.0; matching this program's own established
convention throughout `SCALE_TRANSFER_DESIGN.md` and `KEY_ANCHORING_DESIGN.md`
for a first-wave cell), **2 corpora** (`reasoning_mix_eot_extended`,
`wikitext103_mix_eot_extended` — the SAME extended mixes Track C's own
rung-1/rung-2 cells used, per the same-dataset rule).

**Mandatory manifest — TRAINING runs (unchanged in count from the round-2
revision; Arm 1′/1″ add ZERO training runs, §5, §8):**

| Item | Arms | Seeds | Corpora | Training runs |
|---|---|---|---|---|
| Core 3-arm comparison | 1, 2, 2′ | 3 | 2 | 18 |
| λ mini-sweep (§5) | 2 @ λ∈{0.3,0.8} (λ=0.58 already covered by the core row above) | 1 | 1 (openr1-mix-ext) | 2 |
| **Mandatory training total** | | | | **20** |
| Optional Arm 3 (learned λ, cut first) | 3 | 3 | 2 | +6 |

(20 mandatory training runs + up to 6 optional, UNCHANGED from the round-2
revision — the round-3 redesign adds no new training cells, only new
eval-only measurement passes over the SAME checkpoints, see the manifest
below.)

**Mandatory manifest — EVAL-ONLY measurement passes (NEW this round, §5,
§7.1/§7.1a — near-zero GPU-h, reuses Arm 1's own checkpoints, no new
training):**

| Item | Basis | Seeds | Corpora | λ values | Eval-only passes |
|---|---|---|---|---|---|
| Arm 1′ (Arm-1-checkpoint, dense per-token blend at eval) | Arm 1's 6 checkpoints | 3 | 2 | 0.58 (core) | 6 |
| Arm 1′ @ λ mini-sweep brackets | Arm 1's seed-0/openr1 checkpoint | 1 | 1 | 0.3, 0.8 | 2 |
| Arm 1″ (Arm-1-checkpoint, global-vector blend at eval) | Arm 1's 6 checkpoints | 3 | 2 | 0.58 | 6 |
| Pre-blend `k_raw` co-primary measurement (§4.a-i) | Arm 1 + Arm 2 checkpoints, no blend | 3 | 2 | n/a | 12 (6+6, Arms 1 and 2 both need a raw-key pass) |
| **Mandatory eval-only total** | | | | | **26** |

Every row above is a forward-hook measurement pass over an ALREADY-TRAINED
checkpoint — no gradient step, no optimizer state, no new training wall
time. Cost model and budget accounting in §8 (§8.0-style smoke cost, not
§8.3's per-training-step arithmetic).

### 6.2 Rung-2 (98M) — PARKED this wave (ROUND-5 fix 3: descoped to RUNG-1-ONLY estimation mode; gate definition kept on the books, confirmation only if rung-1 CLEANLY passes AND a fresh design-review clears rung-2 for launch)

**ROUND-5 descope, stated up front: rung-2 is formally PARKED.** This
program's own committed, budgeted, launchable scope THIS WAVE is
**RUNG-1-ONLY** (§8.1). Rung-2's gate definition below is retained
verbatim — the CONDITIONS under which rung-2 would be worth running are
still worth having on record — but satisfying them is no longer sufficient
to launch rung-2 spend by itself; see the new clause immediately below the
gate. This is a direct consequence of §7.1-real.1's pinned-formula power
analysis (round-4/round-5): under the correct statistic, neither tested
CPU-sim proxy family comes close to clearing either corpus's real-floor
threshold, so the CURRENT power arithmetic gives rung-2 no realistic path
to a clean launch — continuing to budget and pre-plan rung-2 as if it were
this wave's second phase would be planning around a launch condition this
design's own evidence does not support yet.

**NEW gate clause (ROUND-5 fix 3, mandatory, supersedes any prior reading
that rung-2 launches automatically once the conditions below are met):
rung-2 spend additionally requires a NEW attack-round pass, run AFTER
rung-1 completes, that RE-DERIVES rung-2's own power arithmetic from
rung-1's OWN MEASURED deltas and floors** — not from this design's current
CPU-sim proxies, which §7.1-real.1 has already shown do not clear the real
floor at this program's own λ=0.58/corpus combination. Concretely: even if
rung-1 clears every condition below (a clean CONFIRM on both corpora,
both instruments, both controls), the launch decision for rung-2 is NOT
"conditions met, launch" — it is "conditions met, THEN commission a fresh
design review that uses rung-1's own real measured effect size and real
measured cross-seed floor (not the pre-launch CPU proxies) to ask whether
rung-2, at 98M params and a materially larger per-run cost (§8.3), would
actually be adequately powered to detect (or fail to detect) a
scale-transfer of whatever rung-1 measured." This closes the gap the
task's own instruction names explicitly: the current power arithmetic
(§7.1-real.1) gives rung-2 no realistic path under today's evidence, and
that fact must be disclosed alongside the gate, not silently left for a
future reader to discover only after rung-1's own results are in.

Reuses Track C's own rung-1 architecture (`d_model=768, n_layers=12,
d_state=64`, `n_params≈97,618,176`, verified `lm_rd_rung_configs.py`
`RUNGS[1]`) — note the deliberate naming collision with Track C's own
"rung 1": **this program's rung-2 IS Track C's rung-1 config.** Flagged
explicitly to avoid the exact kind of cross-document rung-numbering
confusion this codebase's own history has hit before (Track B's rung
terminology vs. Track C's). This program's own rung table:

| This program's rung | Params | Architecture | Reuses |
|---|---|---|---|
| **rung-1 (14M)** | 14,048,896 | `d_model=256, n_layers=2, d_state=64` | Wave C's own config |
| **rung-2 (98M)** | ≈97,618,176 | `d_model=768, n_layers=12, d_state=64` | Track C's own "rung 1" config |

**Rewritten launch gate (attack finding 5 — this REPLACES every "≥1 corpus"
reading anywhere else in this document; §7.1a's headline pass condition is
the single source of truth this gate points to. ROUND-3: conditions 1 and 2
re-rolled under the new Arm-1′/Arm-1″ primary — see §7.1/§7.1a. ROUND-4:
condition 1's own pass criterion updated for §7.1-real's fixed-threshold →
ESTIMATION-mode fallback — a CI-excludes-zero reading, not a
"clears the bar" reading. ROUND-5 fix 1: condition 1 restated with the
SINGLE pinned CI formula, §7.1-real.1 — the formula and both corpus
thresholds are now fully specified, not left as "the real-floor-derived
CI" without a pinned multiplier. ROUND-5 fix 3: this gate's own conditions
are necessary but NO LONGER sufficient for launch — see the new PARK
clause above, which additionally requires a fresh design-review pass after
rung-1 completes):** rung-2 launches **ONLY IF ALL of the
following hold, AND the PARK clause above is separately satisfied**:

1. Rung-1's primary (§7.1, Arm 2 vs. Arm 1′) reads as a **CONFIRM under
   the PINNED CI formula (§7.1-real.1): `mean_delta ± t(2,0.975)·s_ref/√n`
   (`t(2,0.975)=4.303`, n=3) excludes zero, in the mechanism-predicted
   direction, against the PINNED per-corpus thresholds — openr1-mix-ext:
   0.0546; wikitext-mix-ext: 0.1064** — on **BOTH corpora independently**
   (not pooled, not "≥1 corpus" — a split outcome is INCONCLUSIVE, and
   INCONCLUSIVE does **NOT** authorize rung-2 spend, closing attack finding
   5's ambiguity explicitly). **This is a materially higher bar than the
   pre-round-4 fixed-threshold reading it replaces** — per §7.1-real.1,
   NEITHER tested proxy effect family clears either pinned threshold at
   any tested effect size, so a CONFIRM under this reading, if it occurs
   on real launch data, is itself informative that the true
   training-mediated effect exceeds what this design's own CPU proxies
   modeled;
2. **Rung-1's co-primary (§4.a-i, pre-blend `k_raw` geometry) is ALSO hit on
   BOTH corpora** (round-3 addition — the co-primary was promoted precisely
   because it is the more sensitive of the two instruments per the
   validating sim, §7.1; a post-blend-only pass with a null co-primary is a
   §1.3 uninterpretable result, not a licensed rung-2 launch);
3. Arm 2's OWN training-mediated delta (vs. Arm 1′) exceeds Arm 2′'s OWN
   training-mediated delta (vs. Arm 1″) on the mandatory comparison (§7.1a)
   on **BOTH** corpora;
4. No §7's uninterpretable-result exclusion (§1.3) is triggered.

**BINDING CONSTRAINT (ROUND-5 fix 2, mirrored from §7.1-real.1): wikitext
is the binding corpus on condition 1 above.** Its own pinned threshold
(0.1064) is roughly 2× openr1's (0.0546), tracking the real archived
data's own ~2× larger cross-seed std at wikitext — no tested proxy effect
family comes close to clearing the wikitext threshold at any tested
effect size (§7.1-real.1). Condition 1's "both corpora independently"
requirement means wikitext, not openr1, is effectively what determines
whether rung-1 can produce a clean CONFIRM at all; this is disclosed here
so the gate is not misread as "openr1 is the real test, wikitext is a
formality."

**A split on ANY of conditions 1–3 is INCONCLUSIVE overall and does NOT
authorize rung-2.** This closes the exact gap attack finding 5 named: the
prior text said rung-1 must "pass... with no gate violation" but never
stated whether INCONCLUSIVE counts as passing for LAUNCH-AUTHORIZATION
purposes (as opposed to headline-reporting purposes, where §7.1's own rule
already applied). It does not. Only a clean pass on both corpora, on all
three conditions, launches rung-2 spend.

Same arm set as rung-1 (1, 2, 2′) — the λ mini-sweep is **rung-1-only**
(§5's own registered scope; a rung-2 mini-sweep is not budgeted, §8) — N=3
seeds {0,1,2}, 2 corpora = **18 mandatory TRAINING runs** at rung-2 (Arm 3
not repeated at rung-2 — if authorized at all, it is a rung-1-only curiosity
probe per §5, Arm 3). **PLUS the same eval-only measurement passes as
rung-1's manifest (§6.1), scaled to rung-2's own checkpoints** (Arm 1′: 6
passes; Arm 1″: 6 passes; pre-blend `k_raw` co-primary: 12 passes — no λ
mini-sweep passes at rung-2, since the mini-sweep itself is rung-1-only) —
**24 eval-only passes**, near-zero GPU-h, §8.

### 6.3 MANDATORY calibration run — standing [LEARN], non-negotiable

Per this project's own standing rule ("A calibration run... before a big
sweep is mandatory, not optional... catches convergence ceilings... and
confirms a 'bigger model' guess doesn't silently diverge"), **one full Arm-2
training run at rung-1's target step count must complete and be inspected
(loss curve, `span_frac` trajectory, no NaN/divergence, no OOM) BEFORE the
remaining rung-1 cells launch.** This is in addition to, not a substitute
for, the forward/backward/gradient-check smoke (§8) and the Wave −1
instrument-validity smoke (§8.0, new — attack finding 7) — the smokes prove
the code runs and the probe measures the right thing; the calibration run
proves the specific config trains sensibly at scale. Concretely: launch Arm
2, corpus=openr1-mix-ext, seed=0 alone first; inspect its own `span_frac`
trajectory and val-loss curve at the harness's own 1,000-step checkpoint
cadence; only then launch the remaining rung-1 cells (Arm 1's 6 cells, Arm
2's remaining 5 cells, Arm 2′'s 6 cells, the λ mini-sweep's 2 cells — 19
remaining of the 20 mandatory rung-1 cells) as one batch. If rung-2 is
authorized, repeat the identical single-cell-first discipline there before
its own 17 remaining cells launch — this project's rung-3 experience (§8.4)
is the direct cautionary precedent for skipping this at a new scale.

---

## 7. Pre-registered bars — exact numeric thresholds, derivation method, no post-hoc moves

**Sign-convention note, one sentence (closes round-2 Finding 5, applies to
every `±`/`≥`/`≤` in §7.1, §7.1a, §7.2 below):** every bar in this design is
stated as "quantity X must be `k` sample-SDs BELOW (more negative than)
reference quantity Y" — i.e. a bar clearing means `X ≤ Y − k·s_ref`, NEVER
`X ≥ Y + k·s_ref`, so a positive `Δ = X − Y` NEVER clears any bar in this
document, only a sufficiently negative one does (val loss in §7.2 is the
one exception, stated with its own explicit direction: Arm 2's loss must be
`≤ Arm 1's mean + tolerance`, i.e. a bar on how much WORSE loss is allowed
to get, not a "must fall" bar like the geometry bars).

### 7.1 Primary observable (§4.a/§4.a-i) — ROUND-3 REDESIGN: Arm 2 vs. Arm 1′ (post-blend, co-primary with pre-blend `k_raw`), replacing the round-2 bar KILLED as an auto-pass artifact

**What round-2's own revision built, and why the round-2 attack correctly
killed it as FATAL:** the round-2 text derived a DIRECTION for "Arm 2's
`span_frac` must fall ≥2 SD below Arm 1's" via a CPU sim
(`sim_frozen_bias_direction.py`, kept, still used below for its instrument
plumbing) and reported a confident, stable "falls" reading in the regime
matching real archived baselines. **The round-2 attack's FATAL finding is
that this reading is real but says nothing about training:** the sim's own
`crossover_scan` (still on record, `results/frozen_bias_sim/
sim_frozen_bias_direction_results.json`, unedited) shows
`dense_arm2_span_frac` moving only from `0.0218` (`shared_frac=0.0`) to
`0.0936` (`shared_frac=0.7`) — a ~0.07-wide band — while the SAME sweep's
`baseline_span_frac` moves from `-0.00002` to `0.3536` — a ~0.35-wide range,
five times larger. **Dense per-token blending toward one of 50,257
near-orthogonal random rows (pairwise dot ≈ 1/√64 ≈ 0.125 in expectation)
mechanically PINS the measured post-blend population into a narrow band
regardless of what the pre-blend keys looked like.** Whatever `span_frac`
Arm 1's real (never-blended) checkpoint happens to sit at, Arm 2's post-blend
population was ALWAYS going to land near this pinned band — the "falls"
reading was comparing a mechanically-pinned quantity (post-blend Arm 2)
against an unpinned one (never-blended Arm 1), which auto-passes or
auto-fails based on where Arm 1 happens to sit, not on anything Arm 2's
training did. **Confirmed this round for Arm 2′ too** (§7.1a): the
global-single-vector blend pins to an even narrower, more extreme band
(~0.61–0.71) for the same reason — Arm 2-vs-Arm-2′ under the round-2 scheme
was equally an artifact, since BOTH sides were dominated by their own
target's pinned geometry, not by anything a real comparison could
discriminate.

**The redesign: difference out the pin by construction, not by threshold
adjustment.** The pin is a property of the BLEND ARITHMETIC ITSELF, applied
once to whatever population is fed into it. If Arm 2 (trained through the
bias) and Arm 1′ (never trained through the bias, but blended at eval time
with the IDENTICAL code, table, and λ) are BOTH measured through this same
pinning arithmetic, the pin is common to both sides and cancels in the
comparison — any surviving difference must come from a difference in the
PRE-blend key populations the two arms actually trained, which is precisely
what "does training through the bias change the raw key geometry" (the
mechanism claim) asks.

- **Arm 2** = `measure(dense_blend(k_raw_arm2_trained, B, λ))`, where
  `k_raw_arm2_trained` is Arm 2's own trained checkpoint's raw (pre-blend)
  key population.
- **Arm 1′** = `measure(dense_blend(k_raw_arm1_trained, B, λ))`, the
  IDENTICAL `dense_blend` function, IDENTICAL frozen table `B`, IDENTICAL
  λ, applied to Arm 1's own trained (never-bias-exposed) raw key population,
  at EVAL TIME ONLY (§5, Arm 1′). Not a new training run — an eval-only pass
  over Arm 1's existing checkpoints.

**Bar as originally derived this round (ONE-SIDED, `mean_ref ± k·s_ref`,
`k=2`) — HISTORICAL, SUPERSEDED by §7.1-real.1's pinned two-sided
`t(2,0.975)`-corrected formula, kept verbatim below so the derivation's
own direction/mechanism-account logic (which §7.1-real.1 does NOT
re-litigate) stays legible; see §7.1-real.1 for the OPERATIVE formula and
pinned thresholds:** Arm 2's mean pooled post-blend `span_frac`
must be **≥2 sample standard deviations BELOW** Arm 1′'s own freshly-measured
mean post-blend `span_frac`, on **both** corpora independently. Direction
(falls, not "moves") is licensed by the SAME mechanism account as before
(§10.13.4 — constancy stabilizes cross-episode key geometry): if training
through the bias causes SGD to arrange `k_raw` so that its OWN structure
already partially anticipates the fixed blend target (the constancy-suffices
account's own claim), the post-blend result should be MORE, not less,
internally consistent than blending the SAME fixed target onto Arm 1's
raw keys, which never had a chance to adapt. A rise (Arm 2 ABOVE Arm 1′) or
a null result are both pre-registered, non-cherry-pickable outcomes (§1.2).

**Falsifiability, validated with a NEW sim this round
(`deltanet_rd/sim_frozen_bias_training_mediated.py`, CPU-only, imports
`sim_frozen_bias_direction.py`'s own instrument plumbing rather than
duplicating it — same `gram_deviation`/`chunk_key_gram_stats`/
`summarize_gram_records`/`anchors`/`span_frac`/`random_unit_rows_init`
chain, unmodified):**

1. **NULL CASE — must NOT auto-pass.** When there is NO injected
   training-mediated effect (`effect_strength=0.0`: Arm-2-equivalent raw
   keys are constructed to be BIT-IDENTICAL to Arm 1's own raw keys), the
   Arm-2-vs-Arm-1′ post-blend delta is **EXACTLY 0.0 in all 5 tested seeds**
   (`max_abs_post_blend_delta_at_null = 0.0`, `n_seeds_exact_zero_delta =
   5/5`) — checked with an exact equality assertion (`torch.equal`-level,
   zero tolerance), not a "delta is small" numerical stand-in. This is the
   proof the round-2 bar never had: a bar built from two arms passing
   through the SAME mechanical arithmetic cannot auto-pass, because with no
   real difference to detect, there is nothing left for the arithmetic to
   manufacture.
2. **SENSITIVITY — must detect a real effect.** A synthetic training-mediated
   stabilization effect is injected into Arm 2's pre-blend raw keys at
   `effect_strength`∈{0.05, 0.1, 0.2, 0.3} (a convex blend of Arm 1's raw
   keys toward extra fixed directions, independent of `B` so this is not the
   bias itself leaking in — see the script's own `inject_training_effect`).
   Read against a **realistic cross-seed noise floor** (NOT the degenerate
   null cell's own zero std — a genuine `s_ref` derived from measuring
   Arm-2-analog vs. Arm-1′-analog across INDEPENDENT seed pairs at
   `effect_strength=0`, since real Arm 2/Arm 1′ measurements will always
   differ by at least this much sampling noise even under a true null;
   `measure_seed_noise_floor`, post-blend `s_ref=0.002541` at N=5 pairs):
   the post-blend comparison clears the `mean_ref + 2·s_ref` bar starting at
   `effect_strength=0.3` (mean Δ=−0.00557, ≈2.2× the noise floor). **At N=3
   seeds (matching this program's own real per-arm seed count), the same
   qualitative pattern holds** (verified separately this session: post-blend
   mean Δ=−0.00586 at effect_strength=0.3, cross-seed floor comparable in
   magnitude).

**Disclosed limitation the sim ALSO surfaces (this is why §4.a-i's pre-blend
co-primary exists, not an afterthought):** the pre-blend (raw `k_raw`)
comparison is UNIFORMLY MORE SENSITIVE than the post-blend one at every
tested effect size — e.g. at `effect_strength=0.2`ᴺ⁼³, raw mean Δ=−0.01205
vs. post-blend mean Δ=−0.00272 (≈4.4× larger raw effect, before accounting
for each instrument's own noise floor). The post-blend comparison remains
the primary because it is the one that most directly mirrors what the
mechanism account is about (cross-episode key stability AS SEEN THROUGH the
blend the model actually uses, §10.13.4) — but its lower sensitivity means a
null post-blend result is weaker evidence of a true null than a null
pre-blend result would be, which is exactly why §1.1/§1.2's CONFIRM/REFUTE
logic requires BOTH instruments to agree, not either alone.

**λ-sensitivity gradient (disclosed, feeds §1.2a and the mini-sweep
readout, §5):** the SAME fixed synthetic effect (`effect_strength=0.2`) run
at each of the mini-sweep's own λ values shows the post-blend bar's own
sensitivity FALLING as λ rises: mean Δ=−0.0099 (λ=0.3), −0.0027 (λ=0.58),
−0.00009 (λ=0.8) — over a 100× shrinkage from λ=0.3 to λ=0.8. This makes
mechanistic sense (as λ→1, both Arm 2's and Arm 1′'s post-blend populations
converge toward the frozen anchor table itself, `B[token_id]`, regardless of
`k_raw`, mechanically suppressing any detectable pre-blend difference) and
must be disclosed alongside any λ-mismatch or null reading (§1.2a).

Full numbers, all effect strengths/seeds, and the seed-noise-floor
derivation: `matrix-thinking/deltanet_rd/results/frozen_bias_sim/
sim_frozen_bias_training_mediated_results.json`.

### 7.1-real REAL cross-seed noise floor (ROUND-4 fix 1 — supersedes the synthetic `s_ref_post_blend=0.002541` used above as this bar's OPERATIVE reference)

**What round-3 got wrong, stated plainly:** §7.1's own falsifiability proof
above validates that the Arm-2-vs-Arm-1′ comparison HAS sensitivity in
principle, using a synthetic i.i.d.-population-sampling `s_ref_post_blend=
0.002541` (`measure_seed_noise_floor`, Zipf-token synthetic keys, no
training anywhere). **This was never checked against what REAL cross-seed
training variance on this exact instrument, this exact architecture,
actually looks like.** It should have been — the design's own §1.2a/§7.1
already promise `s_ref` is "computed from Arm 1′'s own fresh measurement,"
i.e. real data, once the program launches; the synthetic number was only
ever meant to validate the COMPARISON's sensitivity in principle, not to be
mistaken for, or silently inherited as, the real-launch `s_ref`. A fresh
audit (round-4) checked whether archived data already settles this question
before the program launches, rather than leaving it for launch day.

**Archived data found: YES — real, same-architecture, same-corpus,
cross-seed `span_frac` data exists, and it is NOT within 2× of the
synthetic floor; it is roughly an ORDER OF MAGNITUDE larger.** Two
independent archived families qualify, both at this program's exact target
config (`d_model=256, n_layers=2, d_state=64`, `n_params=14,048,896`,
`all7_span_frac` — the SAME instrument, `chunk_key_gram_stats`/
`summarize_gram_records`/`anchors`/`span_frac`, this program's own §4.a
already reuses unmodified), 3 training seeds {0,1,2} per corpus, sourced
from `experiment-runs/2026-07-06_trajectory_probes/
reference_finals_archived.json` (mirrored byte-identically repo-root vs.
SSD, verified `diff -rq` this session; itself a re-run of the archived
probe instrument, validated to reproduce the original harvest to ≤1e-6,
per that directory's own `TRAJECTORY_SUMMARY.txt`):

| Family | Corpus | seed 0 | seed 1 | seed 2 | mean | sample std (n−1, df=2) | range |
|---|---|---|---|---|---|---|---|
| `mixcontrol` (extended mixes — SCALE_TRANSFER_DESIGN.md §5.10, this program's own §5 corpus choice) | openr1-mix-ext | 0.218102 | 0.259618 | 0.226281 | 0.234667 | **0.021992** | 0.041516 |
| `mixcontrol` | wikitext-mix-ext | 0.322033 | 0.268555 | 0.237326 | 0.275971 | **0.042838** | 0.084707 |
| `wave1_ctrl` (original mixes — SCALE_TRANSFER_DESIGN.md §5.9, NOT this program's own corpus choice, kept here as a second independent check) | openr1-mix | 0.213846 | 0.258157 | 0.214767 | 0.228923 | **0.025321** | 0.044311 |
| `wave1_ctrl` | wikitext-mix | 0.286274 | 0.254522 | 0.283280 | 0.274692 | **0.017532** | 0.031752 |

**Every real per-corpus std (0.0175–0.0428) is 6.9×–16.9× the synthetic
`s_ref_post_blend=0.002541`** — squarely the design's own pre-registered
"real spread >> synthetic floor (order of magnitude)" case, not the "≤~2×"
case. This is confirmed independently in BOTH corpora and BOTH archived
mix families (4 of 4 real per-corpus stds exceed the synthetic floor by
≥6.9×), so this is not a single-cell fluke — verified directly against the
raw JSON this session (recomputed independently from
`reference_finals_archived.json`, not merely quoted from a summary).

### 7.1-real.1 THE PINNED CI FORMULA (ROUND-5 fix 1 — SINGLE operative formula, resolves FATAL-1 + MAJOR-3)

**Three inconsistent thresholds coexisted across this document before this
fix — round-4's own text and round-3-era text derived numbers from THREE
different statistics without ever declaring which one governs:**

1. raw-std, `k=2`: `2·s_ref` (no `√n` division) — 0.043984 (openr1) /
   0.085676 (wikitext). This is the WRONG statistic for a claim about a
   MEAN DIFFERENCE (`mean(Arm2) − mean(Arm1′)`, each mean itself computed
   over n=3 seeds) — it is the spread of a SINGLE observation, not the
   spread of a 3-seed sample mean, and using it overstates how much a
   3-seed mean could plausibly vary by ignoring the `1/√n` averaging
   benefit entirely.
2. SE-scaled, `k=2` (two-sided, no small-n correction): `2·s_ref/√n` —
   0.025394 (openr1) / 0.049465 (wikitext). Correct denominator (`√n`),
   but `k=2` is the LARGE-sample (`z`) multiplier, not the correct
   small-sample multiplier at `n=3` (`df=2`) — at this small n the true
   sampling distribution of a standardized mean has materially fatter
   tails than the standard normal, and `k=2` understates the interval,
   making a CONFIRM easier to obtain than the data supports.
3. **SE-scaled, `t`-corrected (two-sided)** — the statistically honest
   one, PINNED here as the SINGLE operative formula for every CI/threshold
   in this document from this point forward:

   **`mean_delta ± t(n−1, 0.975) · s / √n`, with `n=3` seeds →
   `t(2, 0.975) = 4.303`.**

   This is the standard two-sided 95% confidence interval for a mean
   (or mean-difference) estimated from a small (`n=3`) sample with unknown
   population variance — the `t`-distribution, not the normal, is the
   correct sampling distribution at this `n`, and a two-sided interval is
   the conservative, symmetric convention this document already uses
   everywhere else (§7.1a-real's own `11b` question 2 flagged exactly this
   choice as needing to be pinned, not left implicit).

**PINNED thresholds (this program's own two real per-corpus stds, §7.1-real
above, `n=3`, `s` = each corpus's own real per-seed sample std):**

- **openr1-mix-ext:** `t(2,.975) · s/√n = 4.303 × 0.021992/√3 = 4.303 ×
  0.0126971 = 0.054636` → **0.0546**
- **wikitext-mix-ext:** `4.303 × 0.042838/√3 = 4.303 × 0.0247325 =
  0.106424` → **0.1064**

**Every other CI/threshold number in this document is DELETED and replaced
by these two pinned values, everywhere** (§1.1, §9, §11b-2 — see the
ROUND-5 REVISION LOG for the exact list of edited locations). The raw-std
computation (family 1 above) is deleted outright, not merely superseded in
place — it was never the correct statistic for this claim and is not kept
as an alternate reading.

**§7.1a-real's own MDE/threshold figures below are UNCHANGED by this
fix** — the `sim_frozen_bias_training_mediated.py` internal falsifiability
grid (§7.1a-real) is checked against its OWN synthetic `s_ref_post_blend`
using the sim's own internal `mean_ref + 2·s_ref` convention, which is a
DIFFERENT quantity serving a DIFFERENT purpose (proving the comparison has
teeth in principle, not the operative real-launch bar) — this pin governs
the REAL-floor bar only (§7.1-real, §1.1, §9, §6.2).

**Re-derived minimal detectable effect against the REAL floor and the
PINNED formula, this program's own corpus (`mixcontrol`, extended mixes —
§5's registered corpus choice, not `wave1_ctrl`):**

Read against the extended sensitivity grid (§7.1a-real below,
`effect_strength`∈{0.05,...,1.0}, family A/"shared_dirs"): **NEITHER
pinned threshold is cleared anywhere in the tested grid.** The largest
`shared_dirs`-family post-blend delta at any tested effect size is
`|−0.0055| (effect_strength=0.3)` before the effect reverses sign at
`effect_strength≥0.5` (§7.1a-real's own disclosed non-monotonicity) —
`0.0055` is roughly 10× SMALLER than the pinned openr1 threshold
(`0.0546`), and roughly 19× smaller than the pinned wikitext threshold
(`0.1064`). **Family B ("anchor_directed," the mechanistically-motivated
family added round-4, extended to its own λ-sweep round-5 — §11b-3 below)
saturates around `|Δ|≈0.023–0.026` at large effect sizes — still 2.1×–2.4×
smaller than the pinned openr1 threshold alone**, i.e. even the MORE
sensitive of the two proxy families, pushed to its own asymptote, does not
clear the pinned real-floor bar at this program's own corpus.

**Verdict, restated ONCE with the pinned formula (this task's fix 1): the
post-blend primary bar (§7.1) as registered has NO DEMONSTRATED POWER
against the REAL cross-seed null, evaluated against either tested
synthetic-proxy effect family, at any effect size this sim tested, under
the PINNED `t`-corrected threshold.** This is marked
**UNDER-POWERED-AGAINST-REAL-NULL** here, prominently, now for the correct
statistical reason (a properly small-sample-corrected CI, not merely a
looser or stricter ad hoc multiplier). **This does NOT mean the true
training-mediated effect (if one exists) is too small to matter** — it
means the SPECIFIC proxy families used to validate this bar's sensitivity
never produced a large enough synthetic signal to clear the REAL noise
floor under the statistically correct threshold, and no evidence-based
reason exists to expect the real effect to be larger than what these
proxies produce. Absent a proxy family shown to clear the real floor,
licensing this bar as a binary CONFIRM/REFUTE instrument would be
asserting power the design cannot currently demonstrate.

**Honest disclosure, pre-registration timing (this task's own
instruction, stated once, here, as the canonical record):** under the
REJECTED `k=2` SE-scaled formula (family 2 above, 0.025394 at openr1),
family B's own saturating delta (`|Δ|≈0.0264` at `effect_strength=0.5`,
§7.1a-real) would have **MARGINALLY CLEARED** the openr1 threshold
(`0.0264 ≥ 0.025394`, a margin of ≈0.001). This is recorded here explicitly
so the formula choice is visibly made BEFORE this fact was used to argue
for or against it: the pinned `t`-corrected formula (0.0546) is the
STRICTER of the three candidate families at both corpora, chosen on
statistical-correctness grounds (small-`n` sampling theory, above) BEFORE
re-checking which family family-B's own numbers happened to clear — not
selected because it kills (or saves) this particular result. Under the
pinned formula, family B's 0.0264 does NOT clear (0.0264 < 0.0546) — the
UNDER-POWERED-AGAINST-REAL-NULL verdict stands under the pinned formula,
and would have been marginally contradicted only under the weaker,
statistically-incorrect `k=2` family this fix rejects.

**Pre-registered fallback (adopted now, before launch, per this task's own
instruction, PINNED formula): the post-blend §7.1 primary's headline
readout changes from a binary CONFIRM/REFUTE bar to an ESTIMATION
readout.** Concretely:

- **Report the measured `span_frac(Arm 2) − span_frac(Arm 1′)` delta
  directly, with a confidence interval derived from the REAL per-corpus
  cross-seed std above using the PINNED formula** (`mean_delta ±
  t(2,0.975)·s_ref/√n` at n=3 — the two-sided, small-sample-corrected
  interval, NOT the raw per-seed std used in either rejected family —
  the bar threshold and the CI are the SAME pinned quantity now, not two
  different derivations of `s_ref` as in the pre-fix text).
- **CONFIRM (post-blend instrument) now requires the CI to exclude zero**
  — i.e. the measured delta's own pinned-width interval must not straddle
  0, replacing "clears a fixed threshold derived from a synthetic floor"
  with "the real measured effect is statistically distinguishable from no
  effect, using the real noise floor and the correct small-sample `t`
  statistic." A null/zero-crossing CI is reported honestly as a null
  estimate, not spun.
- **This changes NOTHING about §4.a-i's pre-blend co-primary or §7.1a's
  Arm-2-vs-Arm-2′ control in KIND** — both of those already specify
  `mean_ref ± k·s_ref` derived from FRESH real data at launch time
  (§7.2/§7.3's own blind-pinning discipline already applies) — but BOTH
  now also adopt the SAME pinned `t(2,0.975)` multiplier for consistency
  (one formula, one `k`, everywhere in this document, per this fix's own
  mandate), superseding the flat `k=2` these sections used pre-fix.
- **Mechanical enforcement:** `BANDS_PINNED-FrozenBias.json` (§7.3) is
  extended to carry Arm 1′'s own REAL per-seed post-blend `span_frac` std,
  computed from Arm 1′'s actual eval-only measurement pass (not assumed
  from this archived `mixcontrol` figure, which is a same-config PRIOR
  check only — the launch-time pin still re-derives `s_ref` from THIS
  program's own fresh Arm 1′ measurement, per §7.2's existing discipline,
  extended here to the post-blend bar too, using the PINNED `t(2,0.975)`
  multiplier). The archived `mixcontrol` figures above are the PRE-LAUNCH
  power check that justifies moving to ESTIMATION mode; they are not
  themselves substituted for the launch-time pin.

**BINDING CONSTRAINT (ROUND-5 fix 2, MAJOR-2): wikitext is the binding
corpus on any CONFIRM.** The real archived per-corpus floor is not
symmetric across this program's two corpora — wikitext-mix-ext's own real
cross-seed std (0.042838) is essentially **2× openr1-mix-ext's own**
(0.021992), and the pinned threshold inherits this asymmetry directly
(0.1064 vs. 0.0546, §7.1-real.1). **Because §6.2's own gate (and this
section's own headline pass condition, below) requires BOTH corpora to
independently clear their own pinned threshold, wikitext — not openr1 — is
the corpus that actually constrains whether any CONFIRM is reachable at
all: clearing openr1's looser 0.0546 threshold is a necessary but weaker
condition than clearing wikitext's 0.1064 threshold, and NO tested proxy
effect family, at any tested effect size, comes anywhere near the
WIKITEXT threshold either** (family A's largest monotonic-range delta,
0.0055, is ≈19× smaller than 0.1064; family B's asymptotic delta, ≈0.0264,
is ≈4.0× smaller). Any future reading of this design that treats openr1
as "the" test corpus and wikitext as a formality should be corrected by
this note: wikitext is the harder bar, not a secondary check, and the
UNDER-POWERED-AGAINST-REAL-NULL verdict is, if anything, more clearly true
at wikitext than at openr1.

**Disclosed scope note:** the real archived data is PRE-blend training
variance (Arm 1's own raw-key population across seeds, never exposed to
any frozen-bias blend) — it characterizes how much a real training run's
`span_frac` varies seed-to-seed BEFORE any Arm-1′-style eval-only blend is
applied. The design's own §7.1 falsifiability sim shows the post-blend
comparison is a DAMPED version of the pre-blend signal (λ-sensitivity
gradient, above) — meaning a real post-blend cross-seed floor, once
measured from Arm 1′ directly, could plausibly be SMALLER than this raw
pre-blend floor, not larger. This is disclosed as the one respect in which
this fix's own arithmetic is conservative (i.e., it uses the raw pre-blend
std as a stand-in upper-bound-ish estimate for the eventual real post-blend
std, since no real POST-blend cross-seed data exists yet — Arm 1′ itself
does not exist until Arm 1 trains and is eval-blended) — but conservative
in the direction of UNDER-stating, not over-stating, the bar's power
problem: even a smaller real post-blend floor would still need the tested
proxy effect families to clear it, and neither family's detected deltas
(≤0.0056 for family A within its monotonic range, ≤0.026 for family B) are
large by comparison to ANY of the four measured real stds (0.0175–0.0428),
so the qualitative UNDER-POWERED finding is not sensitive to this
pre-vs-post-blend distinction at the margins observed here.

**Headline pass condition (both instruments, both corpora):** the
post-blend primary (this section) AND the pre-blend co-primary (§4.a-i)
must BOTH clear their own bars on **both** corpora independently (not
pooled) — a split on either instrument or either corpus is reported as
**INCONCLUSIVE**, mirroring `TRACKB_REDESIGN.md` §5.1's own
same-outcome-in-both-corpora rule, extended here to also require
same-outcome-across-both-instruments. **INCONCLUSIVE does NOT authorize
rung-2 launch** (§6.2's rewritten gate).

### 7.1a-real Extended sensitivity grid + second effect family (ROUND-4 fix 2 — resolves 11a attack question 1, re-derives the λ=0.58 minimal detectable effect)

**Two extensions to `sim_frozen_bias_training_mediated.py` this round,
re-run, results regenerated at the same path
(`deltanet_rd/results/frozen_bias_sim/
sim_frozen_bias_training_mediated_results.json`):**

1. **`effect_strength` grid extended past 0.3**, to {0.5, 0.7, 1.0}
   (`INJECTED_EFFECT_STRENGTHS`), so the razor-thin λ=0.58 minimal-detectable-
   effect figure the round-3 attacker computed by extrapolation (≈0.284,
   against a grid edge of 0.3) is now read INSIDE the tested grid, not
   linearly extrapolated past its own last point.
2. **A second injected-effect family, "anchor_directed"**
   (`inject_training_effect_anchor_directed`), per §10.13.4's own
   mechanistic account ("raw keys learn to anticipate the blend target"):
   rather than family A's ("shared_dirs," the original round-3 family)
   convex blend toward `n_shared_dirs=4` directions INDEPENDENT of `B`,
   family B blends `k_raw` PARTIALLY TOWARD `B[token_id]` ITSELF — a more
   direct model of "SGD nudges each token's raw key to anticipate its own
   frozen anchor row," rather than a generic proxy for "some other
   training-induced structure change."

**Both families' detection thresholds, at λ=0.58, the calibrated baseline
(`shared_frac=0.6`), N=5 seeds, against the SYNTHETIC noise floor
(`s_ref_post_blend=0.002541` — this grid's own internal falsifiability
check; §7.1-real above is the analysis against the REAL floor and
supersedes this synthetic-floor reading for launch purposes):**

| Effect family | Smallest `effect_strength` clearing synthetic bar | Behavior at large effect_strength |
|---|---|---|
| A: `shared_dirs` (original, round-3) | 0.3 (mean Δ=−0.00557) | **NON-MONOTONIC, disclosed as new this round:** Δ continues negative at 0.3 (−0.00557) but WEAKENS at 0.5 (−0.00169, back to sub-bar) then **flips sign** at 0.7 (+0.02445) and 1.0 (+0.04133). Mechanistically: at high `effect_strength`, `k_raw_arm2_equiv` collapses toward the 4 shared directions themselves, an even MORE extreme structure than the calibrated baseline's own `shared_frac=0.6` population, overshooting past Arm 1′ and landing on the opposite side. **This means family A's own "detects at ≥0.3" reading is a LOCAL, not global, property of the grid — a fresh attack should not assume monotonicity holds past the originally-tested range**, confirming round-3 attack question 1's suspicion that this family's parametric choice was not independently derived from theory.
| B: `anchor_directed` (NEW, mechanistically motivated, §10.13.4) | **0.1** (mean Δ=−0.00666) — 3× MORE sensitive than family A's own 0.3 threshold | Monotonic and saturating: −0.0136 (0.2), −0.0200 (0.3), −0.0264 (0.5), −0.0251 (0.7), −0.0234 (1.0) — approaches an asymptote around `|Δ|≈0.023–0.026`, never reverses sign in the tested range.

**ROUND-5 fix 4 (MAJOR-4, family A collapse note, verified this round by
direct SVD, not asserted):** at `effect_strength=1.0` family A's own
`k_raw_arm2_equiv` population **collapses to EXACTLY rank
`n_shared_dirs=4` of the ambient `d_state=64`** — verified directly this
round (`torch.linalg.svd` on the flattened `(102400, 64)` population,
`effect_strength=1.0`: singular values `[180.53, 166.02, 157.10, 132.54,
2.9e-05, ...]`, i.e. the 5th singular value is ~7 orders of magnitude
below the top 4, an exact numerical-rank-4 population; at `effect_strength
∈{0.3, 0.5, 0.7}` the same check shows full rank 64, confirming the
collapse is specific to the `effect_strength=1.0` endpoint, where the
family's own construction sets `mixed = (1−1.0)·k_raw + 1.0·extra_component`
— i.e. `k_raw` drops out entirely and the population becomes a pure convex
combination of the 4 fixed `extra_dirs` rows). **This means family A is
NOT a trustworthy instrument beyond roughly `effect_strength≈0.3`: its own
MDE/λ-grid figures (this section, §7.1-real.1) are an
INTERPOLATION-WINDOW-ONLY result** — valid for describing the sim's
behavior strictly between the tested low-`effect_strength` regime and the
point where the population starts moving toward this rank-4 degeneracy,
not a general statement about what "a larger synthetic effect" would show,
since past a certain `effect_strength` family A is not modeling "more of
the same kind of structure change," it is modeling a qualitatively
different, degenerate rank-collapsed population. Family B
(`anchor_directed`) does not have this failure mode in the tested range
(monotonic, no sign reversal, §7.1a-real table above) and is the more
trustworthy of the two families outside family A's own interpolation
window.

**λ=0.58 minimal detectable effect, re-derived from the EXTENDED grid
(linear interpolation between the two grid points straddling the
`mean_ref+2·s_ref` synthetic-floor threshold, reproducible via this
script's own new `interpolate_minimal_detectable_effect` helper — this is
the exact reproducible version of the round-3 attacker's own ~0.284
estimate, not a fresh guess):**

- **Family A (`shared_dirs`): 0.2841** (bracketed between the tested
  `effect_strength=0.2` (Δ=−0.00253, below threshold) and `0.3`
  (Δ=−0.00557, above threshold) points) — this MATCHES the round-3
  attacker's own ≈0.284 estimate almost exactly, confirming that estimate
  was correct, not merely close.
- **Family B (`anchor_directed`): 0.0768** — a MATERIALLY smaller minimal
  detectable effect, since this family's own effect at any given
  `effect_strength` is larger (it pulls directly toward the SAME table the
  post-blend step itself uses, so a given `effect_strength` under family B
  produces a bigger post-blend signature than the same nominal
  `effect_strength` under family A's independent-direction construction).

**λ-grid (family A only, since family B was not run at brackets other than
0.58 this round — disclosed scope, mirrors the round-3 sim's own λ-grid
scope): minimal detectable effect at each mini-sweep λ, family A:**

| λ | Minimal detectable `effect_strength` (family A) |
|---|---|
| 0.3 | **0.1775** |
| 0.58 | **0.2841** |
| 0.8 | **0.8622** |

This confirms and sharpens the round-3 sensitivity-gradient finding (§7.1's
own effect_strength=0.2-only snapshot: Δ=−0.0099/−0.0027/−0.00009 at
λ=0.3/0.58/0.8): the FULL grid shows λ=0.8's own minimal detectable effect
(0.862) is so large it sits almost at the top of the entire tested range —
under family A, λ=0.8 is barely powered to detect ANYTHING within a
plausible effect-size regime, materially reinforcing (not merely echoing)
the round-3 disclosure that a null reading at λ=0.8 carries far less
evidentiary weight than a null reading at λ=0.3.

**§5's λ=0.58 registration, confronted directly (per this task's own
instruction):** λ=0.58 is only marginally-to-moderately powered under
family A against the SYNTHETIC floor (MDE=0.284, i.e. requires a fairly
large synthetic effect before detection) and, per §7.1-real above, is
**NOT demonstrated to be powered at all against the REAL floor** by either
family at any tested effect size. λ=0.3 is a materially better-powered
grid point under family A alone (MDE=0.177, ≈1.6× more sensitive than
0.58) — but this improvement is NOT large enough to change the §7.1-real
verdict, since even λ=0.3's own tested deltas at this sim's own effect
sizes do not approach the REAL per-corpus thresholds (0.0546/0.1064, the
§7.1-real.1 pinned t-corrected values; this sentence originally cited the
since-rejected raw-std 0.044–0.086 figures — corrected at round-5
verification, and the conclusion is unchanged under the pinned, stricter
formula) either;
the λ-grid was not re-run against the real floor this round (disclosed
scope limitation, below). **Decision: λ=0.58 is KEPT as the primary,
registered value, not re-registered to 0.3.** Justification: (1) 0.58 is
this program's own transplant-fidelity anchor — it is §10.13.4's actual
measured SGD-preferred value in the source synthetic grammar, and the
program's stated purpose (§1) is to test whether THAT specific transplant
does anything in a real LM, not to hand-pick whichever λ the CPU proxy
happens to favor; (2) the power problem identified in §7.1-real is not
plausibly closed by moving to λ=0.3 alone (still far from clearing the
real floor in this sim's own tested range); (3) the mandatory λ mini-sweep
(§5, §6.1) still runs λ=0.3 and 0.8 alongside 0.58, so the "is 0.58
mismatched" question remains empirically answered by the real launch
data, not foreclosed by this CPU-only revision pass. **The power tradeoff
is disclosed, not resolved by re-registration** — per §1.2a's own
mandatory-disclosure rule, any λ=0.58 null reading at launch must be
reported alongside this MDE gradient (0.177/0.284/0.862), and, per
§7.1-real's own new ESTIMATION-mode fallback, the post-blend headline at
ANY λ is now read as a CI-excludes-zero estimate, not a fixed-threshold
pass/fail — which somewhat blunts (but does not eliminate) the
practical cost of keeping the less-sensitive λ=0.58 as primary.

**RESOLVED, ROUND-5 fix 6 (was: "Disclosed scope limitation" — family B's
own λ-sensitivity not swept, §11b question 3):** family B
(`anchor_directed`) was swept this round at the SAME λ∈{0.3, 0.58, 0.8}
grid, same discipline as family A's own grid above
(`sim_frozen_bias_training_mediated.py`, re-run, results JSON
regenerated). **Family B's own MDE gradient: 0.036 (λ=0.3), 0.077
(λ=0.58), 0.494 (λ=0.8)** — the SAME qualitative falling-with-λ pattern as
family A (0.177/0.284/0.862), uniformly more sensitive throughout (by a
factor of roughly 3–5×, consistent with family B's own λ=0.58 MDE already
being smaller than family A's, §7.1a-real above). **New disclosure family
A did not have within its own tested mini-sweep range: family B's λ=0.8
cell flips POSITIVE starting at the SMALLEST tested nonzero
`effect_strength` (0.05), not merely at an extreme endpoint** — i.e.
family B's own λ=0.8 proxy is barely informative (worse than merely
low-powered — its very small residual sensitivity points the wrong
direction) across almost its ENTIRE tested effect range, a stronger
version of the same λ=0.8-is-a-weak-null-test disclosure §1.2a already
carries for family A, now folded into §1.2a's own λ-mismatch reading logic.
Full per-λ, per-effect_strength rows: `deltanet_rd/results/frozen_bias_sim/
sim_frozen_bias_training_mediated_results.json`, key
`lambda_grid_family_b_anchor_directed`.

### 7.1a Mandatory Arm 2 vs. Arm 2′ comparison (attack finding 2 — licenses the transplant claim; ROUND-3: RE-ROLLED under the Arm-1′/Arm-1″ scheme, round-2 FATAL finding 1 applies equally here)

**Confirmed this round: the round-2 Arm-2-vs-Arm-2′ bar (Arm 2's `span_frac`
≥2 SD below Arm 2′'s) is the SAME artifact class as the round-2 primary,
not a valid guard against it.** A direct check this round (CPU, reusing
`sim_frozen_bias_direction.py`'s own `build_global_blended_population`
unmodified) shows the global-vector blend pins `span_frac` to a
**~0.61–0.71 band across the full baseline-structure sweep**
(`shared_frac`∈{0.0, 0.2, 0.4, 0.6, 0.7}: global-blended `span_frac` =
0.6092, 0.6159, 0.6441, 0.6916, 0.7114 respectively — compare Arm 2's own
~0.02–0.09 band over the identical sweep). **Arm 2 sitting far below Arm 2′
in the round-2 scheme was therefore GUARANTEED by construction (a per-token
lookup across 50,257 distinct rows is always going to look less collapsed
than every key collapsing onto ONE shared direction), regardless of
training.** The round-2 sim's own "Arm 2 clears the bar comfortably"
finding was real but, like the round-2 primary, uninformative about
training.

**Redesign, generalizing §7.1's fix:** Arm 2′ gets its own artifact-matched
control, **Arm 1″** (§5) — Arm 1's checkpoint, global-vector-blended ONLY at
eval time, same table-derived `b_global`, same λ. The transplant claim is
now licensed by comparing each arm's OWN training-mediated delta against
its OWN control, not by comparing Arm 2 directly against Arm 2′ (which sit
on structurally different, non-comparable pinned bands and were never on a
shared numeric scale to begin with):

- **Arm 2's training-mediated delta** = `span_frac(Arm 2) − span_frac(Arm
  1′)` (§7.1).
- **Arm 2′'s training-mediated delta** = `span_frac(Arm 2′) − span_frac(Arm
  1″)`.

**Bar (ROUND-5 fix 1: now uses the SAME pinned `t(2,0.975)=4.303`
SE-scaled multiplier as §7.1-real.1, superseding the flat `k=2` this
subsection used pre-fix — one formula, one `k`, everywhere in this
document):** Arm 2's training-mediated delta must be **`t(2,0.975)·s_ref/√n`
sample-SEs MORE NEGATIVE** (further below zero, in the direction §7.1's
mechanism account predicts) than Arm 2′'s training-mediated delta, on
**both** corpora independently (same split-is-INCONCLUSIVE rule as §7.1).
**Pre-registered readings:**

- **Arm 2's delta is more negative than Arm 2′'s (this bar clears):** the
  per-token-KEYED character of the bias matters — training through a
  per-token-keyed bias produces a LARGER training-mediated stabilization
  effect than training through a generic constant-vector bias. Necessary
  (not sufficient — §7.1 must also clear) for CONFIRM (§1.1).
- **Arm 2′'s delta matches or exceeds Arm 2's (this bar does not clear, or
  clears in Arm 2′'s favor):** the effect (if any) is a generic
  regularization/norm effect of training through ANY constant additive
  bias, not a transplant of the specific constancy-suffices mechanism —
  REFUTE (§1.2).

**Sim status, disclosed honestly:** the round-2 sim's own
"Arm2-vs-Arm2′-favors-Arm-2" finding does NOT carry over to this redesigned
bar, because that finding was itself computed on the killed artifact
comparison. **No sim in this design directly validates the NEW Arm-2′-vs-
Arm-1″ construction's null-vs-injected behavior with its own dedicated
grid** (the null-exactness proof in §7.1 generalizes by the same
code-path-identity argument — Arm 2′ and Arm 1″ share the identical
`build_global_blended_population` call, so a null training effect produces
an exact-zero delta here too, verified directly this session: at
`effect_strength=0`, `global_blend(k_raw)` applied identically to both sides
gives `span_frac(Arm2′-analog) == span_frac(Arm1″-analog)` exactly,
`0.6915597488222989 == 0.6915597488222989`) — but the FULL sensitivity grid
(detecting a real injected effect at this construction specifically) was not
separately re-run, since the mechanism is identical to §7.1's own proof by
construction (both are instances of "the SAME blend function applied to two
raw-key populations"). This is disclosed as a scope limitation, not silently
assumed clean.

### 7.2 Secondary/gate bar (§4.b, val loss) — tolerance now DERIVED from Arm 1's own seed spread, not imported flat (attack finding 4)

**What the attack round correctly flagged:** the pre-revision text imported
a flat "+5% relative" tolerance from `TRACKB_REDESIGN.md` §5.3 by precedent,
without checking whether this program's own (materially smaller,
better-characterized) architecture has a tighter or looser natural
seed-to-seed val-loss spread — a flat 5% could be looser than warranted,
letting a real regression through.

**Fix: derive the tolerance from Arm 1's own fresh rung-1 seed spread,
computed and blind-pinned BEFORE any Arm 2 inspection** — the exact
`BANDS_PINNED-FrozenBias.json` pattern §7.3 already uses for the primary
bar, extended to cover the val-loss tolerance in the same pin, same
blinding discipline. **Formula:** `tolerance_abs = k · s_ref`, where
`s_ref` is Arm 1's own sample standard deviation of per-seed val loss (n=3,
df=2, same-corpus), and the pass condition is `val_loss(Arm 2) ≤
val_loss(Arm 1)_mean + tolerance_abs` (an ABSOLUTE nats tolerance derived
from measured spread, not a relative-percentage figure imported from a
different architecture's own loss scale). **k chosen: k=2, verified against
this codebase's own house convention, not Track B's.** Track B's own +5%/
+2% figures (`TRACKB_REDESIGN.md` §5.3, §14.3) are FLAT RELATIVE
PERCENTAGES, not a `mean_ref ± k·s_ref` derivation at all — there is no `k`
to inherit from Track B's own convention, because Track B never derived its
tolerance from measured seed spread in the first place (this is itself
worth naming: Track B's own tolerance was ALSO an imported flat figure, the
same class of problem this finding fixes here). The actual `mean_ref ±
k·s_ref` house convention in this codebase is `KEY_ANCHORING_DESIGN.md`
§3.6's `BANDS_PINNED` family (`derive_engaged_bands`,
`key_anchoring.py`: `engaged_k = mean_ref + 2.0 * s_ref`) — **k=2**, reused
here.

**ROUND-5 disclosure (fix 1's "one formula, one verdict, everywhere" scope,
stated precisely): this gate (§7.2) is a one-sided TOLERANCE on how much
WORSE a single arm's loss is allowed to get, not a two-sided CONFIRM/REFUTE
claim about a training-mediated effect size — it is a materially different
statistical object than §7.1-real.1's pinned CI, and this document does
NOT extend the `t(2,0.975)`-corrected small-sample multiplier to it.** The
pinned `t`-formula (§7.1-real.1) governs every bar that makes a
two-sided/CI-style claim about an effect's existence (§7.1's primary,
§7.1a's Arm-2-vs-Arm-2′ control) — both now use `t(2,0.975)·s_ref/√n`, NOT
this section's flat `k=2·s_ref` (no `√n`, no `t`-correction). This section
keeps `k=2·s_ref` (raw per-seed std, not standard-error-scaled) as its own
one-sided tolerance convention, unchanged by fix 1, because a val-loss gate
is answering a different question ("is the cost acceptable," not "is the
effect distinguishable from zero") and borrowing the CI apparatus here
would not make the gate more correct, only more confusing. Any future
attack round that wants a single universal `k`/formula across ALL bars in
this document, gate included, should treat that as a fresh, separate
finding — not silently assumed resolved by this fix.

**Mechanical enforcement:** computed as part of the SAME
`BANDS_PINNED-FrozenBias.json` write (§7.3) — Arm 1's fresh per-seed val
loss (in addition to its `span_frac`) is captured, hash-pinned, and the
derived `tolerance_abs` per corpus is written into the pin BEFORE any Arm 2
result is inspected, under the identical `assert_blind_not_broken` timing
check.

**Role: still a gate, not a headline claim** (unchanged from the
pre-revision text): Arm 2 must clear this on the corpus where its primary
bar (§7.1) is claimed. Failing this gate while passing §7.1 routes the
result to the §1.3 "uninterpretable" bucket (real geometry change,
disclosed unacceptable loss cost) — reported plainly, not treated as a
partial win, per the standing "a geometry win at a real loss cost is
explicitly NOT a success" convention this program borrows directly from
`SCALE_TRANSFER_DESIGN.md` §4.6. Arm 2′ is held to the identical derived
tolerance for the same reason (§7.1a's re-rolled comparison must not be
confounded by a loss-only artifact either). **ROUND-3 note: Arm 1′ and Arm
1″ have NO val loss of their own to gate** — they are eval-only measurement
passes over Arm 1's own checkpoint (§5), which never trains through any
blend, so Arm 1's own already-gated val loss is the only val-loss number on
that side of the comparison; nothing new is added to this gate by their
introduction, and §7.1/§7.1a's re-rolled bars remain the ONLY place Arm 1′/1″
enter this design's pass/fail logic.

### 7.3 Mechanical pin (BANDS_PINNED-FrozenBias, no post-hoc moves — scope expanded to cover §7.1a and §7.2's derived tolerance; ROUND-3: extended to Arm 1′/1″ eval-only measurements)

Following the house `BANDS_PINNED` pattern exactly
(`key_anchoring.py`'s `write_bands_pinned` / `validate_bands_pinned` /
`assert_blind_not_broken`, reused as-is, not reinvented): **Arm 1's fresh
per-seed val loss mean/std AND Arm 1′/Arm 1″'s own freshly-measured
post-blend `span_frac` mean/std (computed by applying the eval-only blend to
Arm 1's checkpoints, §5) AND Arm 1/Arm 2's pre-blend `k_raw` `span_frac`
mean/std (§4.a-i)** — all quantities this round's redesigned bars need as a
reference — are computed and hash-pinned to a `BANDS_PINNED-FrozenBias.json`
file **immediately after Arm 1's 6 training cells complete, validate, AND
the Arm 1′/1″ eval-only passes over those checkpoints complete**, BEFORE any
Arm 2 OR Arm 2′ (trained) result is inspected. **Why Arm 1′/1″ must be
computed before, not after, Arm 2/2′'s own data is inspected, even though
Arm 1′/1″ are themselves derived from Arm 1's already-complete
checkpoints:** the blinding concern this pin exists to close is "was the
REFERENCE value fixed before or after the comparison arm's result was
known" — Arm 1′/1″'s span_frac IS the reference value for §7.1/§7.1a's new
bars, exactly as Arm 1's val loss was the reference for §7.2's, so the same
timing discipline applies unchanged; the fact that Arm 1′/1″ require no NEW
training does not exempt them from the pre-registration timing rule. The
launcher refuses to compute or report Arm 2's pass/fail against §7.1's bar,
Arm 2 vs. Arm 2′ against §7.1a's re-rolled bar, or either arm's pass/fail
against §7.2's derived tolerance, unless this pin's timestamp strictly
precedes the earliest start time across BOTH Arm 2 and Arm 2′'s TRAINING
runs (`assert_blind_not_broken`, reused verbatim, extended to check both
arms' start times, not just Arm 2's — unchanged from round 2). This closes
the same "was the bar set before or after seeing the answer" concern this
codebase's own §3.6 pattern was built to close, applied here to the
redesigned artifact-matched comparisons rather than assumed automatically
inherited.

---

## 8. Budget — REDONE END-TO-END for the new mandatory scope (Arm 2′ both rungs, λ mini-sweep, attack findings 2+3+6; ROUND-3: Arm 1′/1″ eval-only additions confirmed near-zero-cost, total essentially UNCHANGED from round 2 — see §8.0a) — ROUND-5 fix 3: OPERATIVE committed budget is now RUNG-1-ONLY, §8.3.1/§8.5.1; §8.3a/§8.4/§8.5's both-rungs arithmetic is retained as historical reference for a future gated rung-2 review, not this wave's ask

**Read this section top-to-bottom for the full both-rungs history (§8.0
through §8.5), but treat §8.3.1 and §8.5.1 as the two subsections that
actually state this wave's committed spend (≈11.20–14.22 GPU-h,
rung-1-only, 2× contingency). Everything upstream of §8.3.1 (§8.0/§8.0a/
§8.0b/§8.1/§8.2/§8.2a/§8.3) still applies as written — those subsections
already describe rung-1-scoped costs/smokes/contention that are part of
this wave's real ask; only §8.3a/§8.4/§8.5's OWN both-rungs totals are
superseded as the operative number.**

### 8.0 MANDATORY Wave −1 instrument-validity smoke (attack finding 7 — new, non-negotiable)

**Before any real training cell launches (rung-1 or rung-2), run the probe
against an Arm-2-configured (frozen-bias-active) TINY checkpoint and
VERIFY — an executed test with an assertion, not a read — that the
captured key population reflects the post-bias keys, not the pre-bias
keys.** This closes attack finding 7: nothing in the pre-revision design
checked that `lm_attractor_probe_rd.py`'s non-invasive forward hook (which
captures `k_conv1d`'s output — the RAW post-conv key, BEFORE this
program's own new frozen-bias blend, which sec 2/Arm 2 applies AFTER the
conv, before `chunk_delta_rule`) actually observes the post-bias key, not a
pre-bias key that happens to look similar.

**Mechanical check, specified exactly:** build a tiny (`d_model=32,
d_state=64, n_layers=1`) `DeltaNetLM` with the new frozen-bias hook wired in
and active (`anchor_active=True`-equivalent for this program's own new,
non-`geo3`-coupled hook — §2's construction); run one forward pass on a
fixed batch; capture (a) the probe's own hooked output
(`k_conv1d`'s registered forward-hook capture, exactly
`capture_raw_keys`'s own mechanism) and (b) a SEPARATE, independent
side-channel capture of the actual post-blend key tensor the model's
forward pass computes internally (a debug attribute set at the blend site,
e.g. `model.blocks[i].mixer.last_k_biased`, analogous to
`key_anchoring.py`'s own `anchor_last_k_blend_raw` side-channel
convention). **Assertion:** `torch.equal(probe_captured_keys,
model_internal_post_blend_keys)` must hold on every position of a
spot-checked batch — if the probe instead matches the PRE-blend key (i.e.
the hook is mis-positioned relative to this program's own new insertion
point), the assertion fails loudly, and no real launch proceeds until the
hook position is fixed. **This is genuinely new work relative to
`lm_attractor_probe_rd.py`'s own existing smoke** (`--smoke` items [1]–[5]
verify the STATISTIC is correct on hand-built tensors; none of them verify
the hook observes the RIGHT tensor once this program's new frozen-bias code
path exists — that code path does not exist yet in the probe's own smoke
suite, since the probe was written before this design). Cost: <0.1 GPU-h
(a handful of forward passes on a toy model, CPU-eligible in principle but
run on-box for exact hook-registration parity with the real training code
path).

### 8.0b MANDATORY Wave −1 code-path EQUALITY smoke (ROUND-4 fix 3, NEW — closes the "are Arm 2's live-training blend and Arm 1′'s eval-only retrofit blend actually the same arithmetic" gap)

**§8.0 verifies the PROBE observes the post-blend key correctly. This item
verifies something §8.0 does not: that Arm 2's live, training-mode forward
blend and Arm 1′'s eval-time retrofit blend — which §7.1's entire redesign
depends on being byte-identical arithmetic, applied to different raw-key
populations (§7.1's own "both sides pass through the EXACT SAME mechanical
pinning arithmetic" load-bearing claim) — actually produce bit-identical
tensors when applied to the SAME input.** §7.1's null-case sim proof
(`check_null_is_exact`) proves this AT THE SIM-CODE level (both call the
same Python function, `build_dense_blended_population`); it does NOT prove
it at the REAL-model level, where Arm 2's blend runs inside
`DeltaNetLMMixer.forward` under live training-mode execution (dropout
active if any is configured, `RMSNorm`/`ShortConvolution` in train mode)
while Arm 1′'s retrofit blend runs as a separate eval-only forward hook
(model in eval mode). If these two code paths diverge in ANY way not
captured by the sim-level proof — different mode-dependent state, a
different `B` buffer instance, a token-id mapping mismatch — §7.1's entire
"the pin cancels, so any surviving difference is training-mediated"
argument silently breaks, and the redesigned bar would be re-introducing
exactly the artifact class this round-3 revision exists to close, just one
layer deeper (a code-path INEQUALITY artifact instead of a mechanical-pin
artifact).

**Mechanical check, specified exactly (a second executed assertion, added
to the same Wave −1 smoke item as §8.0, not a separate wave):**

1. **Build one Arm-2-configured checkpoint** (the SAME tiny `d_model=32,
   d_state=64, n_layers=1` `DeltaNetLM` §8.0 already specifies, frozen-bias
   hook wired in and active), at a fixed training step, with a fixed random
   seed for any stochastic layer state.
2. **Same checkpoint, same step, same input batch, TWO forward passes:**
   - **Pass A (Arm-2-live):** `model.train()`, run the forward pass through
     the LIVE training-mode blend (the actual `k_biased = normalize((1-λ)·
     k_raw + λ·B[token_id])` call inside `DeltaNetLMMixer.forward`, §2),
     capturing the post-blend key tensor via the SAME internal side-channel
     §8.0 already wires (`model.blocks[i].mixer.last_k_biased`).
   - **Pass B (Arm-1′-retrofit):** `model.eval()`, run the SAME batch
     through Arm 1′'s own eval-only retrofit blend code path (the function
     §5/§7.1 specifies for Arm 1′'s measurement pass, applied here to the
     SAME checkpoint rather than a genuinely-never-bias-trained one, purely
     to isolate the CODE-PATH question from the "different raw-key
     population" question that is the whole point of the real Arm-1′
     comparison), capturing the post-blend key tensor via the identical
     side-channel.
3. **Capture-point specification (closes the "which tensor, exactly"
   ambiguity a vague "compare the keys" instruction would leave open):**
   both passes capture the post-blend key tensor at the SAME point in the
   forward graph — immediately after the blend's own `F.normalize` call,
   BEFORE `chunk_delta_rule` consumes it — at `float32` (this program's own
   training dtype, §5's architecture spec; if the real training harness
   runs mixed precision, this smoke is re-run at that precision too, since
   a dtype-dependent rounding divergence would itself be exactly the kind
   of code-path inequality this item exists to catch).
4. **Same `B` buffer instance, same token-id mapping, explicitly:** both
   passes MUST read from the identical in-memory `B` tensor object (not two
   separately-constructed tables with the same seed — a construction-order
   or dtype-cast difference between two "identical" `random_unit_rows_init`
   calls is exactly the kind of subtle divergence this smoke is meant to
   catch, not assume away) and the identical `token_ids` tensor for the
   fixed input batch.
5. **Mode-state handling, specified exactly (the dropout/RMSNorm-state
   concern named in this task's own instruction):** this program's own
   architecture (§5: `d_model=256, d_state=64, n_layers=2, conv_size=4,
   num_heads=1`, reusing Wave C's own `DeltaNetLM`/`DeltaNetLMMixer`) is
   verified this session to carry NO dropout layer in `DeltaNetLMMixer`'s
   own forward path (Wave C's own architecture has no configured dropout
   probability > 0 anywhere in the reused recipe) — so `model.train()` vs.
   `model.eval()` cannot introduce a stochastic divergence via dropout by
   construction, not merely by assumption; this must be re-verified against
   the actual instantiated model at build time (a config regression that
   silently turns on dropout would invalidate this smoke's own "no
   stochastic divergence possible" premise). `RMSNorm` (if present in the
   mixer's pre-blend path) is deterministic in both train and eval mode
   (no running-statistics buffer, unlike `BatchNorm`) — verified against
   `fla.modules.RMSNorm`'s own implementation convention (LayerNorm-family,
   not batch-statistics-family) — so no train/eval-mode RMSNorm divergence
   is expected either, but this smoke's own assertion is the actual proof,
   not the verification of this paragraph's claims about why it SHOULD
   pass.
6. **Assertion:** `torch.equal(post_blend_keys_pass_A, post_blend_keys_pass_B)`
   must hold across every position of the fixed batch. **If this fails, no
   real launch proceeds** — the failure would mean Arm 1′/Arm 1″'s own
   eval-only retrofit blend is NOT byte-identical to Arm 2's live training
   blend, invalidating §7.1/§7.1a's entire "the pin cancels" argument at
   the real-model level, and the retrofit code path must be fixed (or the
   divergence's source understood and the bar's derivation revisited) before
   any Arm 2/Arm 1′ comparison can be trusted.

**STATUS: OPEN until executed on built code** — same convention as §8.0's
own status line: this is a specification, not yet run (no `DeltaNetLM`
code implementing this program's own frozen-bias hook exists yet). A
future revision or the Wave −1 launch step itself must update this status
line once the assertion has actually been run and passed, not assume it
inherited from this design pass. **Cost: folded into the SAME §8.0 Wave −1
budget line (<0.1 GPU-h total for both assertions)** — this is one
additional forward pass and one additional side-channel capture on the
SAME toy checkpoint §8.0 already builds, not a separate build or a
separate cost line.

### 8.0a NEW this round: Arm 1′/1″ eval-only cost (near-zero — reuses Arm 1's own checkpoints, no new training)

**The round-3 redesign's entire primary-observable fix (§7.1, §7.1a) is
built from measurement passes over checkpoints Arm 1's OWN mandatory
training runs already produce — no new training cell, no new seed, no new
corpus.** Per-pass cost: one forward-hook measurement pass (identical
mechanism to the existing `lm_attractor_probe_rd.py --smoke` / `run_
measurement()` machinery, §8.0, reused unmodified for the eval-only blend
application) over an already-saved checkpoint, batched across the probe's
own `n_windows`-worth of held-out text — the SAME order of magnitude as the
existing Wave −1 instrument-validity smoke's own cost (§8.0, "<0.1 GPU-h... a
handful of forward passes"), scaled up modestly for full-corpus coverage
rather than a toy model. **Rung-1: 26 eval-only passes (§6.1's manifest);
rung-2 (if authorized): 24 eval-only passes (§6.2).** Conservatively
budgeting **0.02 GPU-h/pass** (an order of magnitude above the Wave −1
smoke's own toy-model cost, to cover full-corpus, full-checkpoint-size
measurement at rung-2 scale) gives **≈0.52 GPU-h (rung-1) + ≈0.48 GPU-h
(rung-2) ≈ 1.0 GPU-h total, before contingency** — **≈2.0 GPU-h at the
same 2× contingency multiplier §8.4 applies to everything else** (disclosed
as conservative; these are read-only forward passes with no training
dynamics to diverge, so the 2× rung-3-style "unexplained slowdown" risk
this multiplier exists to cover is far less likely here than for an actual
training cell — kept anyway for uniformity, not because a specific failure
mode is expected). **This is added to, not folded silently into, §8.5's
summary table** — see the updated table below.

### 8.1 Program ceiling: **135 GPU-h** (unchanged) — committed scope THIS WAVE is RUNG-1-ONLY (ROUND-5 fix 3); the ceiling is a headroom cap, not a target

**ROUND-5 descope: the 135 GPU-h ceiling itself is UNCHANGED (chosen at
the prior rounds' own 120–150 GPU-h planning band), but this program's
COMMITTED, LAUNCHABLE scope this wave is now RUNG-1-ONLY** (§6.2's PARK
clause). Rung-2's own arithmetic (formerly §8.3/§8.3a/§8.4/§8.5's largest
line items, ≈107–143 GPU-h of the pre-round-5 totals) is **NOT part of
this program's committed spend** — it remains on the books as a
gated-and-parked follow-on whose own launch requires both §6.2's rewritten
gate AND a fresh design review (§6.2's new clause), not a budgeted line
this design is asking to spend against right now. **Rung-1-only mandatory
scope (§8.3, rebudgeted below): Arm 1, Arm 2, Arm 2′ × 2 corpora × 3 seeds
(18 training runs) + the λ mini-sweep (2 more training runs, 20 total) +
Arm 1′/Arm 1″/pre-blend-`k_raw` eval-only passes (26 passes) + the
mandatory calibration cell (already one of the 20 training runs, called
out for schedule visibility) + the Wave −1 instrument-validity and
code-path-equality smokes (§8.0/§8.0b) — with a 2× contingency multiplier
throughout (§8.4's own rung-3 lesson, unchanged): total ≈11.20 GPU-h
mandatory, ≈14.22 GPU-h including the optional Arm 3 curiosity probe
(§8.5's rewritten table below).** This is a small fraction of the 135
GPU-h ceiling — the ceiling was never rung-1's own natural size, it was
sized for a two-rung program this round no longer commits to running as a
single planned wave. The large remaining headroom (≈121–124 GPU-h) is
NOT silently re-allocated to rung-2 by default; it sits unspent, available
ONLY if rung-2's own future design-review pass (§6.2) authorizes a launch
against it.

### 8.2 GPU availability (verified from STATE.md, this session)

GPUs 0–1 occupied by Track C rung-3 until **≈05:00 UTC 2026-07-08**
(STATE.md's own corrected ETA, "measured 1.416 s/step from live logs"); GPUs
2–7 are idle and unclaimed by any registered queue item (STATE.md: "GPUs
2-7: legitimately idle — the registered experimental queue is EMPTY"). This
program should launch on GPUs 2–7 only, leaving 0–1 untouched until rung-3's
own `ALL_DONE` sentinel + harvest completes, per the standing rule against
launching un-gauntleted work onto a box running unrelated tracked
experiments. Operative grant budget ≈192 GPU-h/day for the uptime-metered
window (STATE.md "Hardware," confirmed); this program's 135 GPU-h ceiling is
a separate, tighter, program-specific cap, not a claim on the full daily
hardware budget.

### 8.2a Contention gate mirror (ROUND-4 fix 4, NEW — mirrors `KEY_ANCHORING_DESIGN.md` §12.2.3's own registered ordering constraint)

**This program shares GPUs 2–7 with the concurrent capacity-cliff
(K-anchoring) wave** (§8.2 above; `KEY_ANCHORING_DESIGN.md` §12.5's own GPU
plan puts both programs on the same 6-GPU allocation). `KEY_ANCHORING_
DESIGN.md` §12.2.3 already registers the ordering constraint from ITS OWN
side and explicitly names this file as the one that must mirror it:

> "This wave launches FIRST and ALONE on its 2 GPUs (per the staged launch
> above — K=38/K=42 first), so its own first-stage realized rate is
> observed under single-program conditions, not compounded contention.
> **The frozen-bias LM program's own calibration cell must not start until
> this wave's first-stage rates (K=38/K=42) are observed within
> bracket** — i.e. this wave's staged first-stage check (above) is also the
> gate for the LM program's launch timing, not merely a fact about this
> wave's own internal sequencing. **Cross-reference requirement, not
> performed here:** this mitigation must be mirrored in
> `FROZEN_BIAS_LM_DESIGN.md` at its own next revision so both design
> documents state the same ordering constraint — that file is owned by
> another agent currently working on it; this note only registers the
> requirement on this wave's side and does not edit that file."
> (`KEY_ANCHORING_DESIGN.md` §12.2.3, Rev 12.2 round-2 fix 5)

**Mirrored here, this program's own operative gate:** this program's
own mandatory calibration cell (§6.3 — the first Arm-2 cell, corpus=
openr1-mix-ext, seed=0) **must NOT launch until the K-anchoring wave's own
Stage-1 cells (K=38 and K=42, 3 seeds each, §12.2.3/§12.5 there) have their
realized `wall_s`/GPU-h rates observed within that wave's own K48-calibrated
bracket** (i.e. the K-anchoring wave's own §12.2.3 abort rule has NOT
triggered on either K=38 or K=42's own completed cells). This closes the
compounding-contention risk both documents independently disclose (each
program's own 2× contingency multiplier is calibrated against a
SINGLE-sibling precedent, §8.4 here / §12.2.3 there; neither is derived
against a scenario where both multi-cell programs run simultaneously) by
the same mechanism the K-anchoring wave's own text proposes: stagger the
two programs' launches so at least the FIRST one's realized rate is
observed under uncontended conditions before the second one's calibration
cell adds load to the shared pool. **Operative check before §6.3's launch:
confirm via the K-anchoring wave's own tracked queue/log state (not
assumed from elapsed wall-clock time alone) that K=38 and K=42's own 6
cells have completed and their realized rate sits within the K48-calibrated
bracket** — if that wave's own §12.2.3 abort rule has fired (a ≥1.5×
overrun on either K), this program's own calibration cell also waits until
that wave's own re-pricing/diagnosis (§12.2.3 items 2–3 there) resolves,
since launching into an already-contended, already-anomalous shared pool
would make this program's OWN calibration reading (§6.3, §8.4's abort
rule) uninterpretable for the same reason.

### 8.3 Per-run arithmetic, from banked rung-1 timing constants — REVISED RUN COUNTS (attack findings 2+3)

**Rung-1 (14M) per-run cost**, derived from the ALREADY-MEASURED Wave-C-scale
control-cell timing (`SCALE_TRANSFER_DESIGN.md` §5.6/§5.9, re-verified this
session, not re-estimated): 6 control runs at 6,103 steps each measured
**≈0.46 GPU-h total** → **≈0.0767 GPU-h/run** at throughput **≈361K
tok/s/GPU** (359–365K across the 6 runs, `steps × batch × seq_len / wall_s`
formula, MFU≈3.1%). This program's rung-1 arms will very likely run at a
**different step count** than Wave C's own 6,103 (that count was chosen for
Track C's own mix-control purpose, not this program's own target token
budget) — the arithmetic below is therefore expressed per-1,000-steps so it
scales cleanly once this program's own calibration run (§6.3) fixes a real
step count:

- Measured rate: 6,103 steps / 0.0767 GPU-h ≈ **79,570 steps/GPU-h** at
  batch=32, seq_len=512, `d_model=256/n_layers=2/d_state=64`.
- At an illustrative target of **20,000 steps/run** (≈3.3× Wave C's own
  6,103-step control length, a round planning number ONLY, UNCHANGED from
  the pre-revision draft — rung-1 was always cheap and stays untouched by
  this revision's cut, §8.3a): ≈20,000 / 79,570 ≈ **0.2514 GPU-h/run**.
- **Rung-1 mandatory manifest, REVISED (§6.1): 20 runs** (core 3-arm
  comparison — Arms 1, 2, 2′ × 2 corpora × 3 seeds = 18 — **+ the λ
  mini-sweep's 2 runs**, λ∈{0.3, 0.8} at seed=0/openr1-mix-ext only; λ=0.58
  already counted in the core-18's own Arm-2 row, not double-counted) ×
  0.2514 ≈ **5.03 GPU-h** (vs. the pre-revision's 12-run, ≈3.0 GPU-h figure
  — the +2.03 GPU-h delta is exactly Arm 2′'s 6 added core runs + the
  mini-sweep's 2 runs, at the same per-run rate). Optional Arm 3 (+6 runs)
  ≈ **+1.51 GPU-h**, unchanged.

**Rung-2 (98M) per-run cost, RETAINED HERE AS REFERENCE ONLY (ROUND-5 fix
3: rung-2 is PARKED, §6.2/§8.1 — this arithmetic is NOT part of this
program's committed rung-1-only spend; kept so a future gated rung-2
launch does not have to re-derive it from scratch)**, derived from Track
C's OWN already-measured rung-1 (=this program's rung-2) timing:
`SCALE_TRANSFER_DESIGN.md`/`EXPERIMENT_LOG.md` — 67,547 steps at
`n_params=97,618,176`, banked calibration **0.7135 s/step** (STATE.md,
"banked 0.7135 s/step calibration"). **Rung-2 manifest, IF authorized by a
future design review (§6.2): 18 runs** (Arms 1, 2, 2′ × 2 corpora × 3
seeds — the +6 delta vs. a 12-run figure is exactly Arm 2′'s added core
runs; no λ mini-sweep at rung-2, §5's own rung-1-only scope).

### 8.3.1 Rung-1-ONLY committed budget (ROUND-5 fix 3 — the actual committed arithmetic this wave; supersedes §8.3a/§8.4/§8.5's prior both-rungs totals as the OPERATIVE numbers)

**This wave's actual spend commitment uses ONLY the rung-1 arithmetic
above.** §8.3a/§8.4/§8.5 immediately below are RETAINED VERBATIM as the
historical record of the round-2/3/4 both-rungs arithmetic (the cut
analysis, the rung-3 contingency lesson, the old summary table) — they are
NOT deleted, since a future rung-2 design-review pass will need exactly
this arithmetic as its own starting point — but they no longer describe
this wave's committed spend. **§8.5.1 (new, after §8.5) is the operative
rung-1-only budget table.** Read §8.3a/§8.4/§8.5 below as "what a two-rung
program would have cost, and the cut that would have been needed if one
were launched" — informative context, not this wave's own ask.

### 8.3a THE CUT (this revision's own budget fix — mandatory scope no longer fits the ceiling at the pre-revision's illustrative step counts) — HISTORICAL, both-rungs arithmetic, retained for a future rung-2 review (see §8.3.1)

**Arithmetic check at the ORIGINAL 20,000-step/run illustrative target for
BOTH rungs, before any cut:** rung-1 mandatory 20 runs × 0.2514 ≈ 5.03
GPU-h (2×: 10.05); rung-2 mandatory 18 runs × (20,000 × 0.7135s/3600) ≈
3.964 GPU-h/run × 18 ≈ 71.35 GPU-h (2×: 142.7). **Total mandatory, 2×
contingency, both rungs: 10.05 + 142.7 ≈ 152.8 GPU-h — already ≈18 GPU-h
OVER the 135 GPU-h ceiling before the optional Arm 3 (+3.0 GPU-h at 2×) is
even added (≈156 GPU-h total).** This is the direct, disclosed consequence
of attack findings 2+3 promoting Arm 2′ (both rungs) and the λ mini-sweep
(rung-1) from optional/prose to mandatory — the pre-revision ceiling
arithmetic (§8.4's own pre-revision 101.0 GPU-h figure) no longer holds.

**Cut order, applied explicitly (per this task's own instruction: "cut
rung-2's optional arms first"):**

1. **Rung-2 has no optional arms in this design** (§5's Arm 3 is
   explicitly rung-1-only; §6.2 registers no rung-2-only optional scope) —
   stated here so the cut order is not silently skipped past. There is
   nothing to cut at step 1.
2. **Rung-1's own optional Arm 3 is the next, and only, arm-level cut
   lever** (already the design's own "cut first if budget is tight"
   item, §5) — cutting it saves 1.51 GPU-h (1×) / 3.02 GPU-h (2×). **Not,
   by itself, enough:** even with Arm 3 cut, the total is still ≈149.8
   GPU-h, ≈15 GPU-h over ceiling. Arm 3 is flagged as CUT-IF-NEEDED, not
   yet sufficient on its own.
3. **The only remaining lever that does not touch a single arm, seed, or
   corpus the attack round made mandatory: reduce rung-2's own illustrative
   PLANNING step-count target** (20,000 → a lower, explicitly disclosed
   round number). This is legitimate, not a stealth methodology cut,
   because §6.3's mandatory calibration run — unchanged by this revision —
   is what actually FIXES the real step count before any real spend
   commits; the 20,000 figure was always labeled a "round planning number
   ONLY" (§8.3, both pre- and post-revision text). **Adjusted rung-2
   planning target: 15,000 steps/run** (a disclosed, round reduction, not
   silently chosen to hit an exact ceiling number). At 15,000 steps: 15,000
   × 0.7135s ≈ 10,703s ≈ **2.973 GPU-h/run** × 18 mandatory runs ≈ **53.51
   GPU-h** (1×), **107.03 GPU-h** (2×).

**No seed count, corpus count, or arm is cut anywhere in this design as a
result of this budget fix** — the entire adjustment is confined to one
illustrative planning number that §6.3's own calibration run supersedes
with a measured value before any real spend commits, exactly the same
role the pre-revision 20,000-step figure already played.

### 8.4 The rung-3 lesson: budget with a 2× contingency multiplier (unchanged rule, re-applied to the revised counts) — extended to rung-2's own calibration cell (attack finding 6) — HISTORICAL, both-rungs arithmetic (see §8.3.1: the 2× contingency RULE itself still applies and is re-derived for rung-1-only in §8.5.1; this section's own totals below describe the parked two-rung plan, not this wave's committed spend)

**STATE.md's own rung-3 entry is the direct, load-bearing precedent:**
Track C's rung-3 measured **1.416 s/step from live logs**, exactly **2×** the
banked **0.7135 s/step** calibration rate that this program's own §8.3
rung-2 arithmetic is built on — STATE.md's own diagnosis: *"root cause of
the 2× gap unknown (calibration cell ran solo; suspect dataloader/CPU
contention between the two concurrent runs)"*. Concretely: rung-3's own
booked estimate was 76.25 GPU-h; actual measured cost was **≈144 GPU-h**,
pushing the whole scale program to ≈334/300 against its own ceiling.

**This program's own rung-1/rung-2 manifests deliberately run MULTIPLE
concurrent cells per rung (up to 3 arms × 2 corpora × 3 seeds = 18
simultaneous-ish cells per rung, sharing a subset of GPUs 2–7)** — exactly
the multiple-runs-sharing-a-box condition STATE.md's own rung-3 entry flags
as the suspected cause of its 2× overrun. **Registered contingency: every
GPU-h figure in §8.3/§8.3a is doubled before being counted against this
program's own 135 GPU-h ceiling.** Rung-1 mandatory training: `5.03 × 2 =
`**10.06 GPU-h**; + optional Arm 3 (cut-if-needed, §8.3a): `1.51 × 2 = 3.02`
GPU-h. Rung-2 mandatory training (at the adjusted 15,000-step planning
target): `53.51 × 2 = `**107.02 GPU-h**. Wave −1 smoke: `0.05 × 2 = 0.10`
GPU-h. **NEW this round (§8.0a): Arm 1′/1″/pre-blend-`k_raw` eval-only
passes, both rungs: `1.00 × 2 = 2.00` GPU-h.** **Total, contingency-adjusted,
both rungs, both mandatory arm sets, WITH the optional Arm 3 kept:** `10.06
+ 3.02 + 107.02 + 0.10 + 2.00 =` **122.20 GPU-h**, inside the 135 GPU-h
ceiling with **≈12.80 GPU-h** of headroom remaining. **Without Arm 3 (cut
first if any further squeeze is needed):** `10.06 + 107.02 + 0.10 + 2.00 =`
**119.18 GPU-h**, **≈15.82 GPU-h** headroom. **(Round-3 reconciliation,
closes round-2 Finding 7: these figures now match §8.5's summary table
exactly, computed from full-precision sub-totals rather than summed from
already-rounded display figures — the pre-revision text's 117.08/120.10
here vs. 117.18/120.20 in §8.5 was itself a stray rounding inconsistency,
independent of this round's own eval-only addition.)**

**Abort rule, extended to rung-2's own calibration cell (attack finding 6 —
the pre-revision text only covered the rung-1→rung-2 gate, not rung-2's OWN
internal calibration-vs-manifest check):** if rung-1's own calibration run
(§6.3) measures a per-step rate close to or worse than 2× its own
banked/estimated rate, rung-2 is NOT launched until the discrepancy is
understood (pre-revision rule, unchanged). **NEW: the identical check
applies to rung-2's OWN mandatory calibration cell, one level down** — if
rung-2's calibration cell (the first Arm-2 cell launched alone, §6.3) itself
measures **≥1.5× its own banked rate** (0.7135 s/step, or ≥1.5× whatever
rate rung-1's own calibration run most recently confirmed, if that differs),
**halt the remaining 17 rung-2 cells and diagnose first** — do not launch
the full rung-2 manifest on an unexplained ≥1.5× miss at its OWN
calibration step, mirroring the rung-1→rung-2 gate's own discipline one
level down rather than only checking the boundary between rungs. Repeating
rung-3's own "run continues because the grant is uptime-metered" reasoning
is NOT license to skip understanding either check; STATE.md's own entry
explicitly flags the rung-3 root cause as "unknown," not resolved, and this
program should not inherit an unresolved anomaly silently at either gate.

### 8.5 Summary table — REDONE end-to-end (ROUND-3: eval-only row added, §8.0a; all sub-totals reconciled to consistent rounding — round-2 Finding 7) — HISTORICAL, both-rungs totals (see §8.3.1: superseded as this wave's OPERATIVE budget by §8.5.1 immediately below; retained as the two-rung reference a future gated rung-2 review would start from)

**Note on the calibration run's own accounting:** exactly as in the
pre-revision text, each rung's mandatory calibration cell (§6.3) IS one of
that rung's own mandatory Arm-2 cells (the first one launched, alone) — it
is not a separate, additional run beyond the mandatory-manifest counts
below. The table calls it out on its own line purely for schedule
visibility (it must complete and be inspected before the REMAINING cells of
that row launch), not as extra spend layered on top.

**Rounding note (closes round-2 Finding 7):** every 2×-contingency figure
below is computed as exactly `2 × (1× figure)` at full precision, then
rounded to 2 decimal places for display — e.g. `5.03 × 2 = 10.06`, not the
round-2 text's `10.05` (a stray rounding-down that the round-2 attack
correctly flagged as an inconsistency, §11/Finding 7). All sub-totals below
are recomputed from the underlying full-precision numbers, not summed from
the already-rounded display figures, so the grand totals reconcile exactly.

| Item | Runs / Passes | GPU-h (1×) | GPU-h (2× contingency-adjusted) |
|---|---|---|---|
| Rung-1 mandatory TRAINING (Arms 1, 2, 2′ × 2 corpora × 3 seeds + λ-mini-sweep ×2) | 20 (incl. 1 calibration cell) | 5.03 | 10.06 |
| Rung-1 optional Arm 3 TRAINING (2 corpora × 3 seeds) — **CUT FIRST if squeezed (§8.3a)** | +6 | 1.51 | 3.02 |
| Rung-2 mandatory TRAINING (Arms 1, 2, 2′ × 2 corpora × 3 seeds, **gated on rung-1 §6.2**, planning target ADJUSTED 20,000→15,000 steps/run, §8.3a) | 18 (incl. 1 calibration cell) | 53.51 | 107.02 |
| Wave −1 instrument-validity smoke (§8.0, attack finding 7) | — | 0.05 | 0.10 |
| **NEW (§8.0a): Arm 1′/1″/pre-blend-k_raw eval-only passes, both rungs** | 26 (rung-1) + 24 (rung-2) = 50 | 1.00 | 2.00 |
| **Total mandatory (both rungs, all training + eval-only + smoke, Arm 3 excluded)** | 38 training + 50 eval-only | 59.59 | **119.18** |
| **Total incl. optional Arm 3** | 44 training + 50 eval-only | 61.10 | **122.20** |
| **Program ceiling** | | | **135** |
| **Headroom remaining at ceiling (Arm 3 kept)** | | | **≈12.80** |
| **Headroom remaining at ceiling (Arm 3 cut)** | | | **≈15.82** |

**This still fits — the round-3 redesign costs ≈2 GPU-h more than round-2's
own total (119.18/122.20 vs. round-2's 117.18/120.20), entirely the new
eval-only measurement passes (§8.0a), and remains well inside the 135 GPU-h
ceiling.** Remaining headroom (≈12.80–15.82 GPU-h depending on whether Arm 3
is kept) is reserved for: seed contingency (add-seeds-not-steps, this
codebase's own standing convention, draw order {5, 6, 8, 9} per §6.0), and
any further re-calibration forced by §8.4's 2×- or 1.5×-anomaly checks
(rung-1→rung-2 gate and rung-2's own internal calibration check,
respectively). **If rung-2's own calibration run (§6.3, §8.4) reveals the
adjusted 15,000-step planning target itself needs revising once a real rate
is measured, that revision happens THEN, against this headroom — not by
silently re-padding the illustrative number here.**

### 8.5.1 RUNG-1-ONLY OPERATIVE BUDGET (ROUND-5 fix 3 — THIS is the committed spend this wave; supersedes §8.5's both-rungs table as the number this design is actually asking for)

**Committed scope: Arm 1, Arm 2, Arm 2′ × 2 corpora × 3 seeds (18 training
runs) + the λ∈{0.3, 0.8} mini-sweep (2 more training runs, 20 total,
1 already counted at λ=0.58 in the core 18) + Arm 1′/Arm 1″/pre-blend-
`k_raw` eval-only passes (26 passes, §6.1's manifest) + the mandatory
calibration cell (one of the 20 training runs, called out for schedule
visibility only, not extra spend) + the Wave −1 instrument-validity smoke
(§8.0) and code-path-equality smoke (§8.0b, folded into the same budget
line) — WITH the same 2× contingency multiplier §8.4's rung-3 lesson
requires, applied identically to rung-1-only figures.** Rung-2 (§6.2) is
PARKED and contributes ZERO to this table — its own arithmetic is §8.3a/
§8.4/§8.5's historical record, not this wave's ask.

| Item | Runs / Passes | GPU-h (1×) | GPU-h (2× contingency-adjusted) |
|---|---|---|---|
| Rung-1 mandatory TRAINING (Arms 1, 2, 2′ × 2 corpora × 3 seeds + λ-mini-sweep ×2) | 20 (incl. 1 calibration cell) | 5.03 | 10.06 |
| Wave −1 instrument-validity + code-path-equality smoke (§8.0, §8.0b) | — | 0.05 | 0.10 |
| Arm 1′/1″/pre-blend-`k_raw` eval-only passes, rung-1 (§8.0a) | 26 | 0.52 | 1.04 |
| **Total mandatory (rung-1-only, Arm 3 excluded)** | 20 training + 26 eval-only | 5.60 | **11.20** |
| Rung-1 optional Arm 3 TRAINING (2 corpora × 3 seeds) — **CUT FIRST if squeezed, unchanged cut-order logic from §8.3a** | +6 | 1.51 | 3.02 |
| **Total incl. optional Arm 3** | 26 training + 26 eval-only | 7.11 | **14.22** |
| **Program ceiling** | | | **135** |
| **Headroom remaining at ceiling (Arm 3 excluded)** | | | **≈123.80** |
| **Headroom remaining at ceiling (Arm 3 included)** | | | **≈120.78** |

**This is the ENTIRE committed ask this wave: ≈11.20–14.22 GPU-h at 2×
contingency, well inside the 135 GPU-h ceiling with substantial headroom
(≈121–124 GPU-h) that this design explicitly does NOT pre-allocate to
rung-2.** The ceiling remains 135 GPU-h (unchanged, §8.1) because a future
rung-2 launch, IF a fresh design review (§6.2's new PARK clause)
authorizes one, would draw against this SAME ceiling using §8.3a/§8.4/
§8.5's historical arithmetic (itself already shown to fit, ≈119–122 GPU-h
for a full two-rung program) — but that is a decision explicitly deferred,
not this wave's spend. **Nothing in rung-1's own manifest, seed count,
corpus count, or arm set is reduced by this descope** — every mandatory
rung-1 item from §6.1 is preserved in full; only rung-2's OWN scope moves
from "planned second phase" to "parked, pending review."

## 9. Pre-registered bars — recap (see §7 for full derivation; ROUND-3 REVISED — round-2 FATAL findings, primary observable redesigned around Arm 1′/1″ eval-only controls; ROUND-4: primary bar's fixed threshold demoted to ESTIMATION mode, §7.1-real; ROUND-5 fix 1: SINGLE pinned CI formula restated here, superseding every prior threshold number in this recap)

- **Primary, POST-BLEND (§7.1/§7.1-real/§7.1-real.1, ESTIMATION MODE,
  PINNED FORMULA):**
  Arm 2 (trained through the bias) vs. Arm 1′'s own fresh measurement (Arm
  1's checkpoint, IDENTICAL blend applied ONLY at eval time — §5), on
  **both** corpora independently, **rung-1 ONLY this wave (§6.2, §8.1 —
  rung-2 is PARKED)**.
  **ROUND-5 fix 1: the SINGLE pinned CI formula governing this bar is
  `mean_delta ± t(2,0.975)·s_ref/√n` (n=3, `t(2,0.975)=4.303`), pinned
  thresholds `0.0546` (openr1-mix-ext) / `0.1064` (wikitext-mix-ext)** —
  this replaces every other threshold number that appeared in earlier
  drafts of this document (a raw-std `k=2` family, 0.044/0.086, and an
  uncorrected-SE `k=2` family, 0.025/0.049, both DELETED, §7.1-real.1). Real
  archived same-config, cross-seed `span_frac` data (`experiment-runs/
  2026-07-06_trajectory_probes/reference_finals_archived.json`, `mixcontrol`/
  `wave1_ctrl` families, 4 corpus-cells, N=3 seeds each) is the source of
  `s_ref` for this pin. NEITHER tested synthetic injected-effect family
  (`shared_dirs` or `anchor_directed`, §7.1a-real) produces a large enough
  proxy delta to clear EITHER pinned threshold at any tested effect
  size — the fixed-threshold bar is UNDER-POWERED-AGAINST-REAL-NULL under
  the pinned formula, disclosed with full arithmetic in §7.1-real.1.
  **Headline: report the measured Arm-2-vs-Arm-1′ delta with the pinned
  CI (measured fresh at launch from Arm 1′ itself, per §7.2/§7.3's
  existing blind-pin discipline); CONFIRM requires this CI to exclude
  zero, in the mechanism-predicted direction, on both corpora.** This
  replaces the round-2 Arm-2-vs-Arm-1 bar (KILLED by round-2 attack as an
  auto-pass mechanical artifact — dense blending pins post-blend
  `span_frac` to a narrow band regardless of baseline structure, §7.1's
  crossover-scan evidence) with a bar that is ALSO now known, honestly, to
  need real launch data to demonstrate power rather than a synthetic
  proxy. Validated with a null-vs-injected CPU sim
  (`sim_frozen_bias_training_mediated.py`, extended round-4 to 2 effect
  families and effect_strength∈{0.05,...,1.0}, extended round-5 to family
  B's own λ-sweep, §11b-3): exact-zero delta at null (5/5 seeds, both
  families), synthetic-floor detection at `effect_strength≥0.3` (family A,
  non-monotonic past this point) / `≥0.1` (family B, monotonic in the
  tested range at λ=0.58 but NOT at λ=0.8, §1.2a) — but neither family
  clears the PINNED REAL threshold at either corpus (§7.1-real.1). **Per
  §1.0's plain-language outcome statement, the most likely rung-1 result
  under this pinned formula is a null (zero-including) CI** — an
  estimation-mode result, not a proxy-validated CONFIRM. Split outcome →
  INCONCLUSIVE, not a partial win, and INCONCLUSIVE does **NOT** authorize
  rung-2 launch (§6.2, itself now gated further by a required fresh design
  review even if rung-1 does clear cleanly).
- **Co-primary, PRE-BLEND (§4.a-i, §7.1):** the SAME
  `span_frac` instrument applied to `k_raw` (pre-blend, no artifact
  possible by construction) — Arm 2 vs. Arm 1, direction derived from the
  same mechanism account. **Both the post-blend primary AND this co-primary
  must clear their own bars, on both corpora, for a CONFIRM** — the sim
  shows the pre-blend instrument is uniformly MORE sensitive than the
  post-blend one, so requiring both closes the gap a single, less-sensitive
  instrument would leave (§1.1/§1.2/§1.3).
- **Mandatory control (§7.1a, PINNED FORMULA, ROUND-5 fix 1):** Arm 2's OWN
  training-mediated delta (vs. its Arm-1′ control) `t(2,0.975)·s_ref/√n`
  sample-SEs MORE NEGATIVE than Arm 2′'s OWN training-mediated delta (vs.
  its NEW Arm-1″ control — Arm 1's checkpoint, global-vector blend applied
  only at eval), on **both** corpora independently — same pinned
  multiplier as the primary bar above, superseding the flat `k=2` this
  bar used pre-round-5. **Confirmed round-3 that the round-2
  Arm-2-vs-Arm-2′ bar was the SAME artifact class** (global-vector blending
  pins `span_frac` to an even more extreme ~0.61–0.71 band, guaranteeing
  Arm 2 "beats" Arm 2′ regardless of training) — this bar compares each
  arm's OWN artifact-matched training-mediated delta instead of the two
  arms' raw post-blend values directly.
- **Gate (§7.2, k=2·s_ref, NOT the pinned t-corrected multiplier — a
  deliberately DIFFERENT statistical object, ROUND-5 disclosure):** Arm 2
  (and Arm 2′) val loss within a DERIVED absolute tolerance (`mean_ref +
  2·s_ref` of Arm 1's own fresh per-seed val loss — a one-sided
  raw-std-scaled tolerance, not a two-sided CI, so it does NOT adopt
  §7.1-real.1's pinned `t(2,0.975)` multiplier; see §7.2's own ROUND-5
  disclosure for why), same corpus, at the rung/corpus where
  the primary bar is claimed. Failing this while passing the primary bar →
  reported as a real geometry-vs-loss trade-off, not a success. Arm 1′/1″
  have no val loss of their own to gate (eval-only reuse of Arm 1, §7.2).
- **λ-mismatch reading (§1.2a, ROUND-5 fix 6: family B's own λ-gradient
  now folded in, previously family-A-only):** a null Arm 2
  (λ=0.58), read under the Arm-2-vs-Arm-1′ family at that λ, alongside a
  non-null mini-sweep cell (λ=0.3 or 0.8, same family) is reported as
  "λ-mismatch, mechanism present" — a third, pre-registered outcome,
  distinct from CONFIRM/REFUTE. **Mandatory disclosure:**
  the bar's own sensitivity falls monotonically as λ rises for BOTH tested
  effect families — family A: a fixed synthetic effect produced a detected
  delta of −0.0099/−0.0027/−0.00009 at λ=0.3/0.58/0.8 (`effect_strength=
  0.2` snapshot); family B (NEW, round-5): MDE of 0.036/0.077/0.494 at
  λ=0.3/0.58/0.8 respectively, the SAME qualitative pattern, uniformly more
  sensitive. **Family B's own λ=0.8 cell is additionally disclosed as
  barely informative: its detected delta flips POSITIVE starting at the
  SMALLEST tested nonzero `effect_strength` (0.05), not merely at an
  extreme endpoint** — any λ-mismatch report must state this gradient
  (both families) and this specific λ=0.8 weakness, not treat all three λ
  cells as equally powered.
- **Fourth outcome, NEW this round (§1.3, round-2 Finding 2):**
  "sim-training divergence — informative, not REFUTE" — if the real LM
  comparison contradicts every sim prediction here (e.g. pre-blend and
  post-blend instruments disagree in direction, or the λ-sensitivity
  gradient is violated), report this explicitly, with the cheap diagnostic
  (cosine of trained `k_raw` against `B[token_id]`, before/after training,
  both arms) as the first thing to check.
- **Mechanical enforcement:** `BANDS_PINNED-FrozenBias.json`, written after
  Arm 1's training completes AND its Arm 1′/1″ eval-only passes complete,
  before Arm 2 OR Arm 2′ (trained) is inspected; timestamp must strictly
  precede the earliest start across BOTH Arm 2 and Arm 2′'s training runs.
  The pin carries Arm 1's per-seed val-loss mean/std (§7.2), Arm 1′/1″'s
  post-blend `span_frac` mean/std (§7.1/§7.1a's new reference), and Arm 1's
  pre-blend `k_raw` `span_frac` mean/std (§4.a-i's co-primary reference).

---

## 10. Standing constraints (inherited, checked off explicitly)

- **Wave −1 instrument-validity smoke (§8.0, attack finding 7). STATUS:
  OPEN — a SPECIFICATION, not yet executed (round-2 attack Finding 4,
  fixed this pass: the pre-revision text's phrasing read as if this smoke
  had already been run; it has not — §8.0 specifies the exact mechanical
  check, the assertion, and the cost, but no code implementing it has been
  written or run yet, and this status must not be conflated with the
  ALREADY-EXECUTED `sim_frozen_bias_direction.py` /
  `sim_frozen_bias_training_mediated.py` CPU sims, which HAVE been run this
  session and whose numbers are quoted directly in §7.1/§7.1a).** Verify,
  with an executed assertion (not a read), that the probe's forward hook
  observes the POST-bias key, not the pre-bias key, on an Arm-2-configured
  tiny checkpoint, before any real training cell launches. This item
  becomes DONE only when it has actually been built and run on the real
  hook, with its assertion passing — a future revision or the Wave −1
  launch step itself must update this status line, not assume it inherited
  from this design pass.
- **Smoke test, including eval batch size** (standing rule: "eval can OOM
  even if training fits"): forward/backward/gradient-finite check on the new
  frozen-bias hook at LM shapes, BOTH training batch size and eval batch
  size, before any real launch — a required Wave −1 item, not assumed clean
  by analogy to `model_rd.py`'s own anchor smoke (that smoke tests the
  synthetic-grammar gather/scatter path, which this program's own hook does
  not use).
- **tmux + supervisor for unattended runs:** every rung's manifest launches
  inside `tmux new-session -d -s <name> "<cmd>"`, wrapped in a self-healing
  `while [ ! -f STOP ]; do <cmd>; sleep 15; done` supervisor loop, resume-safe
  (skip already-completed cells by checking output validity, not existence).
- **try/except per config:** the sweep launcher must not let one seed/corpus
  cell's crash kill the remaining manifest, per the standing sweep-robustness
  rule.
- **Checkpoints to `/data` symlink**, never container disk (standing HF-cache/
  checkpoint rule) — mirrors `run_lm_rd_trackc_sweep.py`'s own
  `/data/lm_rd_trackc_ckpts/` convention exactly, new subdirectory for this
  program.
- **Disk-space gate:** reuse `run_lm_rd_trackc_sweep.py`'s own
  `disk_space_check` (`DISK_SAFETY_FACTOR=1.5`) against a REAL measured
  checkpoint size for this program's own config (never assumed from another
  rung's byte count) before any real launch.
- **Archives: repo (≤25MB) + SSD mirror**, per the hybrid archive policy —
  small result JSONs committed to `experiment-runs/`, any larger
  trajectory/checkpoint dump to the SSD path only.
- **Blinding / writer-gate-readout separation:** the `BANDS_PINNED-FrozenBias`
  pin (§7.3) is the load-bearing blind for this program — the judgment calls
  that could leak are "what counts as Arm 1's own reference val loss,"
  "what counts as Arm 1′/1″'s own reference post-blend span_frac" (ROUND-3:
  the pin's own reference quantity, since Arm 1′/1″ ARE the reference for
  the redesigned primary bars, §7.1/§7.1a), and "what counts as Arm 1's own
  reference pre-blend `k_raw` span_frac" (§4.a-i's co-primary) — and the pin
  mechanically prevents any of these numbers from being touched after Arm 2
  or Arm 2′'s TRAINED data exists. No other judgment call in this design
  requires a separate writer/gate/readout split (unlike
  `KEY_ANCHORING_DESIGN.md`'s own multi-instrument engagement test, which
  this program does not use).
- **Calibration-before-sweep:** §6.3, non-negotiable, one real training run
  at each rung's target config before that rung's remaining manifest
  launches.
- **Hold tokenization fixed:** GPT-2 BPE throughout, unchanged from Wave C /
  Track C, per the standing "never bundle a second axis" rule — this
  program's ONLY varied axis is the presence/absence (and, optionally, the
  λ-mode) of the frozen bias term.

---

## 11. ATTACK-ROUND QUESTIONS — the 5 weakest points (SUPERSEDED — this design's own self-attack, kept verbatim as historical record; the actual independent adversarial attack round covered the same ground plus 3 more findings — see the REVISION LOG at the end of this file for the authoritative finding→fix map)

**Status: every item below is now addressed in the operative sections
(§1, §5, §6, §7, §8) — this section is retained unmodified as the
design's own pre-attack self-critique, not re-edited, so the REVISION LOG's
own finding→fix trace stays legible against what this document said BEFORE
the independent attack. Item 1 → Arm 2′ (§5, §7.1a). Item 2 → the mandatory
λ mini-sweep (§5, §6.1, §8). Item 3 → unchanged, still an open, disclosed
limitation (§4.a, §4.c — no fix was possible or attempted this pass, see
the REVISION LOG's "could not fix" list). Item 4 → §7.2's derived tolerance.
Item 5 → §6.2's rewritten gate.**

1. **Does the observable actually require the anchor mechanism, or can a
   vector/positional shortcut produce the same delta?** This is the
   position-decomposition trap named in this codebase's own standing rules
   ("in any full-attention model, 'hold K items' is trivially satisfiable via
   K positions at rank-1 each"). Here the risk takes a different but related
   shape: adding **any** fixed vector to every key (not specifically a
   per-token-identity-keyed one) could plausibly produce a similar
   `span_frac` shift purely as a generic norm/shrinkage effect, with no
   dependence on the bias being *keyed by token identity* at all. **This
   design does not yet include a control that isolates "keyed by token" from
   "merely constant across the corpus."** A cheap, decisive addition worth
   registering before Wave 1 launches: an Arm 2′ that adds the SAME
   frozen table's **global mean row** (a single fixed vector, no token
   lookup) at the same λ — if Arm 2′ matches Arm 2's `span_frac` delta, the
   effect is not about per-token keying at all, and the "transplant of the
   constancy-suffices mechanism" framing would need to be narrowed to "any
   constant additive bias helps," a materially weaker and less interesting
   claim. This control is currently a documented gap, not a closed question.
2. **Is λ≈0.58 transferable, or purely testbed-specific?** 0.58 was measured
   as an SGD-preferred INTERIOR value in a closed ~107-entity synthetic
   grammar with a very different loss landscape (binding recall, not
   next-token prediction over a 50,257-token vocabulary). There is no a
   priori reason a free-running LM's own optimal blend weight — if the
   analogous "optimal" concept even applies here — should land anywhere
   near 0.58. Arm 3 (optional, §5) is the only check on this, and it is the
   first arm cut under budget pressure (§8) — meaning this program could
   easily ship a null result at a badly-chosen fixed λ and never learn
   whether a different λ would have worked. This is the single biggest risk
   to a false-negative REFUTE verdict.
3. **The primary observable (`span_frac`) has never been shown to correlate
   with any user-legible behavioral improvement in an LM** (per §4.a's own
   disclosed weakness) — a positive result here is a real, Tier-2 geometric
   finding, but it is one step further from "this fix helps a real LM do
   something" than the paper's own headline synthetic-grammar result
   (`rec@0.9`). A skeptical reader could reasonably ask whether a `span_frac`
   win is a demo at all, or merely a geometry measurement with an unclear
   practical payoff. §4.c's rejected recall-suite option is the honest
   answer to "what would make this land harder" — explicitly deferred, not
   solved here.
4. **The +5% val-loss gate (§7.2) is a tolerance, not a target — a design
   that "passes" by landing at +4.9% on both corpora is technically clean
   but could still represent a real, if narrowly-tolerated, cost that a
   less charitable bar (e.g., +2%) would have failed.** The 5% figure was
   imported by precedent (Track B's own number) for comparability, not
   re-derived from this program's own loss variance — if this program's own
   seed-to-seed val-loss variance turns out to be much tighter than Track
   B's LM-scale runs (plausible, since Wave C's own architecture is smaller
   and better-characterized), a fixed 5% tolerance could be looser than it
   should be, letting a real regression through. Worth tightening to a
   seed-variance-derived bar (mirroring §7.1's own `mean_ref ± k·s_ref`
   convention) once Arm 1's fresh rung-1 seed spread is measured, rather
   than keeping the imported flat 5%.
5. **The rung-2 gate ("launch only if rung-1 passes") could itself be gamed
   by a favorable INCONCLUSIVE-vs-fail distinction.** §7.1 registers a
   split-corpus outcome as INCONCLUSIVE, not FAIL — but §6.2's rung-2
   launch condition only says rung-1 must pass "with no gate violation,"
   which does not explicitly state whether INCONCLUSIVE counts as passing
   for the purpose of AUTHORIZING rung-2 spend (as opposed to reporting the
   headline). As written this is ambiguous and should be closed before
   launch: the recommended reading (INCONCLUSIVE does NOT authorize rung-2 —
   only a clean pass on both corpora does) should be stated as an explicit
   rung-2 gate condition, not left to be inferred from §7.1's own
   claim-reporting rule, since a launch-authorization ambiguity is exactly
   the kind of judgment call this codebase's own house discipline
   (pre-registration, no post-hoc moves) exists to close in advance.

---

## 11a. ROUND-3 ATTACK QUESTIONS — the weakest points in THIS revision (fresh self-attack, written after the redesign above; a fresh independent attack round should target these first)

**Status update (ROUND-4): items 1 and 2 below are now RESOLVED by this
round's fixes — kept verbatim, not rewritten, so the finding→fix trace
stays legible; see the ROUND-4 REVISION LOG at the end of this file for
the authoritative map. See §11b for the fresh round-4 self-attack.**

1. **RESOLVED, ROUND-4 fix 2 (§7.1a-real).** Is the "training-mediated
   effect" injection in `sim_frozen_bias_training_mediated.py`'s
   sensitivity proof a REALISTIC proxy for what SGD actually does, or an
   arbitrary convex-blend construction chosen because it was easy to code?
   `inject_training_effect` blends `k_raw` toward `n_shared_dirs=4` extra
   fixed directions, independent of the frozen anchor table `B` — this is
   explicitly NOT a simulation of SGD (the design says so, §7.1), but the
   SPECIFIC parametric family chosen (convex blend toward a small number of
   shared directions) was picked because it mirrors the calibrated-baseline
   construction already used in `sim_frozen_bias_direction.py`, not because
   it was independently derived from any theory of what gradient descent
   does to `k_raw` under a per-token frozen bias. A fresh attack should ask
   whether a qualitatively different injected-effect family (e.g. one that
   pulls `k_raw` PARTIALLY toward `B[token_id]` itself, rather than toward
   unrelated shared directions) would change the sensitivity comparison's
   own qualitative conclusion (pre-blend more sensitive than post-blend) —
   this was not tested as a robustness check on the sensitivity claim
   itself. **Round-4 answer:** a second family (`anchor_directed`, exactly
   this proposed construction) was added and run (§7.1a-real). It is MORE
   sensitive (MDE 0.077 vs. family A's 0.284 at λ=0.58) and monotonic where
   family A is NOT (family A reverses sign past `effect_strength=0.5`,
   disclosed as new this round) — so the qualitative conclusion this
   question worried about (pre-blend more sensitive than post-blend) is
   NOT overturned, but family A's own robustness is now known to be
   narrower than assumed (local, not global, monotonicity).
2. **RESOLVED, ROUND-4 fix 1 (§7.1-real).** The cross-seed noise floor
   (`measure_seed_noise_floor`) uses only 5 seed PAIRS, and its own std is
   itself a single point estimate with no error bar of its own — the
   smallest detected effect size (0.3, at N=5-seed cells; also checked at
   N=3, matching the real per-arm seed count) could shift materially with a
   different noise-floor sample. The real experiment's own noise floor
   (from real seed-to-seed training variance) could be larger OR smaller
   than this synthetic floor predicts, and nothing in this design commits
   to re-deriving the noise floor from REAL seed variance once Arm 1's
   actual training runs exist (unlike §7.2's val-loss tolerance, which
   explicitly IS re-derived from real Arm-1 spread before Arm 2 is
   inspected). Should the primary bar's own `s_ref` (§7.1) be derived from
   Arm 1′'s REAL cross-seed spread (analogous to §7.2's discipline), rather
   than assumed to inherit the sim's synthetic floor? This was flagged as
   "currently unresolved" pending verification that the distinction between
   the synthetic validation floor and the real launch-time floor is
   actually maintained, not conflated. **Round-4 answer:** archived
   same-config real cross-seed `span_frac` data WAS found
   (`experiment-runs/2026-07-06_trajectory_probes/
   reference_finals_archived.json`) and the real floor (std 0.0175–0.0428
   per corpus) is 6.9×–16.9× the synthetic floor — NOT close enough for the
   synthetic number to stand in for the real one, confirming this question's
   own suspicion in the "much larger" direction. §7.1-real demotes the bar
   to ESTIMATION mode as a direct consequence.
3. **The λ-sensitivity gradient (bar sensitivity falls as λ rises) was
   measured at exactly ONE injected effect size (`effect_strength=0.2`) and
   ONE baseline regime (`shared_frac=0.6`) — is this gradient itself robust
   across effect sizes and baseline regimes, or could it reverse or flatten
   under a different proxy construction?** If the gradient is an artifact of
   the specific `effect_strength=0.2`/`shared_frac=0.6` combination rather
   than a general property of the blend arithmetic, the λ-mismatch
   disclosure (§1.2a) could itself be miscalibrated.
4. **Arm 1′/1″'s "near-zero GPU-h" costing (§8.0a) is an ESTIMATE
   (0.02 GPU-h/pass), not a measured figure** — unlike §8.3's per-training-
   step arithmetic, which is grounded in ALREADY-MEASURED Wave-C/Track-C
   timing constants, no real forward-pass-only measurement pass over a real
   14M or 98M checkpoint has been timed yet. If a real measurement pass
   turns out to be materially more expensive than 0.02 GPU-h (e.g. if
   loading a full checkpoint and running enough windows for a stable
   `span_frac` estimate is slower than assumed), the ≈12.80–15.82 GPU-h
   headroom this design relies on could shrink. A fresh attack should ask
   whether this estimate needs its own calibration step (analogous to
   §6.3's mandatory training calibration run) before being trusted at
   rung-2 scale.
5. **The Arm-2′-vs-Arm-1″ null-exactness claim (§7.1a) is verified by
   ANALOGY to §7.1's own proof (same code-path-identity argument), not by
   an independently-run null-vs-injected grid specific to the global-vector
   construction.** The design discloses this as a scope limitation, but a
   fresh attack should ask whether the analogy actually holds in every
   detail — e.g. does `build_global_blended_population`'s use of a SINGLE
   derived vector (rather than a per-token gather) introduce any subtlety
   the per-token case doesn't have, that could break exact-zero-at-null in
   a way the spot-check (one seed, one shared_frac value) didn't catch?
   Running the full `sim_frozen_bias_training_mediated.py`-style grid
   against the global-vector construction, not just a single confirming
   data point, would close this more rigorously.
6. **Both new eval-only arms (1′, 1″) apply the SAME frozen table/λ used
   during Arm 2/Arm 2′'s own training — but do they need to be applied at
   the SAME measurement windows/positions as Arm 2/Arm 2′'s own probe pass
   (matched content, not just matched code), or could a mismatch in WHICH
   held-out windows get measured (e.g. different random subsample per arm)
   introduce its own confound?** The design does not yet specify that Arm
   1′/1″'s eval-only measurement must be run over the EXACT SAME held-out
   window set (same seed, same window indices) as Arm 1's own original
   probe pass and as Arm 2's own probe pass, only that the CODE PATH is
   identical. A fresh attack should ask whether this needs to be pinned
   down explicitly (matched windows, not just matched code) before launch,
   since a window mismatch could reintroduce a different kind of
   apples-to-oranges risk even with the code-path fix in place.

---

## 11b. ROUND-4 ATTACK QUESTIONS — the weakest points in THIS revision (fresh self-attack, written after the round-4 fixes above; kept short per this round's own instruction)

**Status update (ROUND-5): items 2 and 3 below are now RESOLVED — kept
verbatim, not rewritten, so the finding→fix trace stays legible; see the
ROUND-5 REVISION LOG for the authoritative map. See §11c for the fresh
round-5 self-attack / open-questions list.**

1. **The real-floor archived data (§7.1-real) is PRE-blend cross-seed
   variance (Arm 1's own raw keys, never blended), used as a stand-in for
   the POST-blend cross-seed floor the §7.1 bar actually needs.** §7.1-real
   discloses this and argues the substitution is conservative (real
   post-blend variance is plausibly damped relative to pre-blend, per the
   λ-sensitivity gradient), but this argument is itself not directly
   verified — no real Arm-1′ eval-only post-blend cross-seed measurement
   exists yet (Arm 1′ does not exist until Arm 1 trains). A fresh attack
   should ask whether the "damped, so conservative" argument actually holds
   at THIS program's own λ=0.58, or whether some other effect (e.g. the
   post-blend pin's own narrow-band mechanics, §7.1's crossover-scan)
   could make real post-blend cross-seed variance LARGER than pre-blend in
   a way the λ-gradient argument does not anticipate. **Still OPEN —
   round-5 did not attempt this (out of scope for this pass's CPU-only,
   no-real-launch-data revision); carried to §11c.**
2. **RESOLVED, ROUND-5 fix 1 (§7.1-real.1).** The ESTIMATION-mode fallback
   (§7.1-real) still needs its own exact CI formula pre-registered with the
   same rigor the old fixed threshold had — this design specifies
   `mean_ref ± 2·s_ref/√n` at n=3 but does not specify, e.g., whether a
   t-distribution correction is needed at this small n (df=2), or whether
   the CI should be one-sided (matching the bar's own one-sided mechanism
   account) or two-sided (the more conservative, symmetric convention this
   fix's own text uses). **Round-5 answer:** pinned the single operative
   formula — two-sided, `t(2,0.975)=4.303`-corrected,
   `mean_delta ± t(2,0.975)·s_ref/√n` — as the SOLE CI/threshold
   convention for every two-sided effect-existence claim in this document
   (§7.1's primary, §7.1a's Arm-2-vs-Arm-2′ control), superseding both the
   raw-std and the uncorrected-`k=2`-SE families that coexisted before this
   fix. Pinned thresholds: 0.0546 (openr1), 0.1064 (wikitext).
3. **RESOLVED, ROUND-5 fix 6 (§11b-3 / §1.2a).** The `anchor_directed`
   effect family's own λ-sensitivity was not swept (§7.1a-real's own
   disclosed scope limitation) — only family A's λ-grid was extended. If
   family B's sensitivity gradient across λ behaves differently (e.g. does
   not fall as sharply as family A's does), the λ=0.58-vs-0.3 power
   tradeoff this round's fix 2 disclosed could look different under family
   B. **Round-5 answer:** swept family B at λ∈{0.3, 0.58, 0.8} (same grid
   discipline as family A). Family B's own MDE gradient: **0.036 (λ=0.3),
   0.077 (λ=0.58), 0.494 (λ=0.8)** — QUALITATIVELY THE SAME falling-with-λ
   pattern as family A (0.177/0.284/0.862), confirming the λ-mismatch
   disclosure (§1.2a) is not an artifact specific to family A. Family B's
   own λ=0.8 cell additionally reverses sign within the tested
   `effect_strength` grid itself (positive from `effect_strength=0.05`
   onward, not merely at an extreme endpoint like family A's reversal) —
   a NEW, stronger disclosure than family A's own non-monotonicity, folded
   into §1.2a.
4. **The code-path equality smoke (§8.0b, fix 3) is, like §8.0, a
   SPECIFICATION not yet executed** — its own mode-state argument (no
   dropout in this recipe, RMSNorm is deterministic) is asserted from this
   session's reading of the reused architecture, not re-verified against
   the actual instantiated model at build time. A fresh attack should treat
   this as unverified until the assertion actually runs. **Still OPEN —
   carried to §11c.**

---

## 11c. OPEN QUESTIONS (ROUND-5, refreshed — short, per this round's own instruction)

Carried forward, unresolved by this round (see §11a/§11b for the fuller
original phrasing of each):

1. **Real post-blend cross-seed floor is still unmeasured** (§11b
   question 1) — §7.1-real.1's pinned thresholds still rest on PRE-blend
   archived variance as a stand-in; the "damped, so conservative"
   argument is disclosed, not verified, until Arm 1′ actually exists.
2. **§8.0/§8.0b's smokes are specifications, not executions** (§11b
   question 4) — the mode-state (no-dropout, deterministic-RMSNorm)
   argument is asserted from this session's code reading, not from a run.
3. **NEW, ROUND-5: the rung-2 PARK clause's own "fresh design review"
   (§6.2) is itself unspecified beyond "re-derive power from rung-1's
   measured deltas and floors"** — this document does not yet pre-register
   WHAT would make that future review conclude rung-2 is worth launching
   (e.g. a minimum measured effect size, a specific real-floor ratio); a
   fresh attack round, once rung-1 data exists, should pin this down rather
   than leave the future review's own bar undefined.
4. **NEW, ROUND-5: family A's own rank-4 collapse (§7.1a-real, fix 4) was
   verified only at `shared_frac=0.6` (the calibrated baseline) and
   `n_shared_dirs=4`** — whether the SAME interpolation-window-only caveat
   applies at other baseline regimes or a different `n_shared_dirs` was not
   checked this round; disclosed as an unverified generalization.
5. **NEW, ROUND-5: family B's own λ=0.8 early-sign-flip (§1.2a/§11b-3
   resolution) was not mechanistically explained beyond "λ→1 suppresses
   the pre-blend signal"** — WHY family B flips sign at the very smallest
   tested `effect_strength` at λ=0.8 (rather than merely shrinking toward
   zero, as a pure suppression account would predict) is an open question
   a fresh attack could usefully chase.

---

## ROUND-5 REVISION LOG — 2026-07-06, this pass's own fixes (pinned CI formula, rung-1-only descope, family-A collapse note, plain-language outcome, family-B λ-sweep)

| # | Fix (one-line) | Applied | Landed in |
|---|---|---|---|
| 1 | FATAL-1 + MAJOR-3: three inconsistent CI/threshold formulas coexisted (raw-std k=2: 0.044/0.086; SE-scaled k=2: 0.025/0.049; SE-scaled t-corrected: 0.055/0.106, nowhere pinned) | Pinned the SINGLE operative formula everywhere: two-sided `mean_delta ± t(2,0.975)·s_ref/√n` (n=3, `t(2,0.975)=4.303`). Pinned thresholds: openr1-mix-ext 0.0546, wikitext-mix-ext 0.1064. Deleted the raw-std computation outright (wrong statistic for a mean-difference claim). Restated §7.1-real's power verdict once with the pinned formula: UNDER-POWERED-AGAINST-REAL-NULL stands, now for the correct statistical reason. Recorded, honestly and BEFORE-data, that family B's own saturating delta (0.0264) would have marginally cleared the REJECTED k=2 SE formula's openr1 threshold (0.0254) — disclosed so the pinned formula's stricter choice is visibly pre-registered, not selected post-hoc to kill or save the result. Extended the same pinned multiplier to §7.1a's Arm-2-vs-Arm-2′ control for consistency; left §7.2's val-loss gate on its own `k=2·s_ref` (a one-sided tolerance, a different statistical object, disclosed explicitly as intentionally NOT adopting the pin) | §7.1-real.1 (new subsection), §1.1, §9, §11b (item 2, marked RESOLVED), §7.1a (bar restated), §7.2 (disclosure added) |
| 2 | MAJOR-2: wikitext (the real floor is ~2× openr1's) was never named as the binding corpus on any CONFIRM | Added an explicit binding-constraint note: wikitext's pinned threshold (0.1064) is ~2× openr1's (0.0546); no tested proxy clears wikitext under any candidate formula; wikitext, not openr1, is what actually constrains whether a CONFIRM is reachable | §7.1-real.1 (new note), §6.2 (mirrored note) |
| 3 | Descope to RUNG-1-ONLY: the promoted mandatory scope (both rungs) no longer has a realistic CONFIRM path under the pinned-formula power analysis | Rung-2 formally PARKED: gate definition kept on the books (§6.2), but a NEW clause requires a fresh design-review pass re-deriving rung-2's own power from rung-1's MEASURED deltas/floors before any rung-2 launch, disclosed as giving rung-2 no realistic path under TODAY's evidence. Rebudgeted: committed scope is rung-1-only (Arm 1/2/2′ × 2 corpora × 3 seeds + λ mini-sweep + Arm 1′/1″/k_raw eval passes + calibration + smokes), 2× contingency, new operative total ≈11.20 GPU-h mandatory / ≈14.22 GPU-h incl. optional Arm 3. 135 GPU-h ceiling UNCHANGED; §8.3a/§8.4/§8.5's both-rungs arithmetic retained as historical reference for a future gated review, not this wave's ask | §6.2 (PARK clause + rewritten header), §8.1 (rewritten), §8.3.1 (new), §8.5.1 (new operative table), §1.1 (CONFIRM criterion restated rung-1-only) |
| 4 | MAJOR-4: family A collapses to exactly rank `n_shared_dirs=4` of 64 at `effect_strength=1.0`, not previously verified or disclosed | Verified directly by SVD this round (`torch.linalg.svd` on the flattened population: top-4 singular values ~130–180, 5th singular value ~3e-05 — an exact numerical rank-4 population at `effect_strength=1.0`; full rank 64 at 0.3/0.5/0.7). Disclosed: family A is not a trustworthy instrument beyond ~0.3; its MDE figures are interpolation-window-only | §7.1a-real (new note, in the effect-family table) |
| 5 | MINOR-6: no plain-language statement of the most likely rung-1 outcome existed | Added §1.0: under the pinned formula and tested proxies, the most likely rung-1 outcome is an estimation-mode result whose CI includes zero at λ=0.58 — an honest, publishable characterization, not a failure to hide; a CONFIRM requires a real effect larger than any tested proxy; rung-2 is a low-probability, wikitext-gated, review-gated event | §1.0 (new subsection) |
| 6 | §11b-3 (family B's own λ-sensitivity never swept) now consequential given the pinned formula and the rung-1-only descope | Extended `sim_frozen_bias_training_mediated.py`: swept family B (`anchor_directed`) at λ∈{0.3, 0.58, 0.8}, same grid discipline as family A's own λ-grid. Re-ran (CPU-only, local disposable venv), regenerated results JSON. Family B's own MDE gradient: 0.036/0.077/0.494 (λ=0.3/0.58/0.8) — same qualitative falling-with-λ pattern as family A, uniformly more sensitive; NEW disclosure: family B's λ=0.8 cell flips positive starting at the smallest tested nonzero effect_strength (0.05), a stronger non-monotonicity than family A shows within its own mini-sweep λ range. Folded into §1.2a's λ-mismatch reading logic | §11b (item 3, marked RESOLVED), §1.2a (λ-gradient folded in, both families), §7.1a-real (scope-limitation note resolved), `sim_frozen_bias_training_mediated.py` (extended: `lambda_grid_family_b`), results JSON regenerated |
| 7 | — | ROUND-5 revision log + refreshed open-questions list needed | This entry; added §11c (5 short items, 3 new) | §11c, this table |

### What changed in the sim script, concretely (fix 6, for reproducibility)

`matrix-thinking/deltanet_rd/sim_frozen_bias_training_mediated.py`: added
a `lambda_grid_family_b` loop (mirrors the existing `lambda_grid_family_a`
loop exactly, same `sim1.SENSITIVITY_LAMBDAS = [0.3, 0.58, 0.8]`, same
per-λ noise-floor re-derivation and `interpolate_minimal_detectable_effect`
call, `effect_family="anchor_directed"` instead of `"shared_dirs"`);
threaded the new grid into `out`'s JSON (`lambda_grid_family_b_anchor_
directed` key) and the printed summary
(`lambda_grid_family_b_mde_summary`). No changes to `sim_frozen_bias_
direction.py` or to family A's own code path. Re-run this session via the
same local disposable CPU-only venv convention the round-4 log already
documents (`python3 -m venv` + `pip install torch --index-url
https://download.pytorch.org/whl/cpu`); results written to the same path,
`deltanet_rd/results/frozen_bias_sim/
sim_frozen_bias_training_mediated_results.json`. No GPU touched; nothing
committed per the task instruction.

---

## ROUND-4 REVISION LOG — 2026-07-06, this pass's own fixes (real-floor check, extended sensitivity grid, code-path equality smoke, contention-gate mirror, hypothesis reword)

| # | Severity | Finding (one-line) | Fix applied | Landed in |
|---|---|---|---|---|
| 1 | FATAL-class | §7.1's primary bar was validated (falsifiability-checked) against a SYNTHETIC i.i.d.-sampling noise floor (`s_ref_post_blend=0.002541`), never checked against real cross-seed training variance at this program's own architecture/corpus | Searched archives; found real same-config (14M, `d_model=256/n_layers=2/d_state=64`), same-corpus (`mixcontrol` extended mixes + `wave1_ctrl` original mixes), cross-seed {0,1,2} `span_frac` data at `experiment-runs/2026-07-06_trajectory_probes/reference_finals_archived.json`. Real per-corpus std (0.0175–0.0428) is 6.9×–16.9× the synthetic floor — the "real spread >> synthetic floor" case, not "≤2×". Re-derived the real-floor threshold (openr1: 0.0440, wikitext: 0.0857); confirmed NEITHER tested proxy effect family's detected deltas clear it at any tested effect size. Marked the fixed-threshold bar UNDER-POWERED-AGAINST-REAL-NULL with full arithmetic; pre-registered the fallback: primary readout becomes ESTIMATION (measured delta + real-floor CI), CONFIRM requires CI excludes zero | §7.1-real (new subsection), §9 (recap rewritten), §6.2 (condition 1 rewritten) |
| 2 | MAJOR | Sensitivity grid stopped at effect_strength=0.3 (razor-thin margin vs. λ=0.58's own ~0.284 minimal detectable effect); only one injected-effect family (`shared_dirs`, independent of `B`) was tested, not the mechanistically-motivated one (pulling toward `B[token_id]` itself) | Extended `sim_frozen_bias_training_mediated.py`: effect_strength grid to {0.5, 0.7, 1.0}; added `inject_training_effect_anchor_directed` (family B, sec 10.13.4-motivated); re-ran, regenerated results JSON. Found family A is NON-MONOTONIC past 0.3 (sign-reverses at 0.7/1.0, new disclosed limitation); family B is MORE sensitive (MDE 0.077 vs 0.284) and monotonic. λ=0.58's own MDE confirmed at 0.2841 (matches round-3 attacker's ~0.284 almost exactly). λ-grid MDE: 0.3→0.177, 0.58→0.284, 0.8→0.862 (family A). λ=0.58 KEPT as primary (transplant-fidelity justification), power tradeoff disclosed, not resolved by re-registration | §7.1a-real (new subsection), `sim_frozen_bias_training_mediated.py` (extended), results JSON regenerated |
| 3 | MAJOR | No check existed that Arm 2's live training-mode blend and Arm 1′'s eval-only retrofit blend — which §7.1's entire redesign depends on being byte-identical — actually produce identical tensors on real model code, only that the sim's OWN code path is self-consistent | Added §8.0b: a second executed assertion (mandatory Wave −1 smoke, folded into the same item as §8.0) — build one Arm-2-configured checkpoint, run Arm-2-live (train mode) and Arm-1′-retrofit (eval mode) blends on the SAME batch/checkpoint/B-buffer/token-ids, assert `torch.equal` on the post-blend key tensor at a specified capture point/dtype. Mode-state (dropout/RMSNorm) argument stated and flagged for re-verification at build time. STATUS: OPEN until executed, same convention as §8.0 | §8.0b (new subsection) |
| 4 | MINOR | This program's own GPU-sharing risk with the concurrent K-anchoring wave was never mirrored from `KEY_ANCHORING_DESIGN.md` §12.2.3's own registered ordering constraint (that file explicitly named this one as needing the mirror) | Added §8.2a: this program's calibration cell (§6.3) must not launch until the K-anchoring wave's Stage-1 (K=38/K=42) realized rates are observed within bracket, quoting §12.2.3 directly | §8.2a (new subsection) |
| 5 | MINOR | §1's one-sentence hypothesis used behavioral language ("token-recall behavior") the design's own registered observables (`span_frac`, a geometric readout) do not measure | Reworded to "training-mediated key-geometry stabilization signature," consistent with §4.a/§9; added a note explaining the change and why no recall/behavioral claim is licensed this wave | §1 (hypothesis sentence + note) |
| 6 | — | Round-4 revision log + attack questions needed | This entry; added §11b (4 short questions) | §11b, this table |

### What changed in the sim script, concretely (fix 2, for reproducibility)

`matrix-thinking/deltanet_rd/sim_frozen_bias_training_mediated.py`: added
`inject_training_effect_anchor_directed` (family B); extended
`INJECTED_EFFECT_STRENGTHS` to `[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]`;
added `EFFECT_FAMILIES = ["shared_dirs", "anchor_directed"]`; threaded
`effect_family` through `run_pair_cell`/`run_null_vs_injected_grid`; added
`interpolate_minimal_detectable_effect` (reproducible linear-interpolation
MDE, replacing ad hoc extrapolation); `main()` now runs both families at
λ=0.58 plus family A's own λ-grid (0.3/0.58/0.8), and writes all of it to
the same results path
(`deltanet_rd/results/frozen_bias_sim/sim_frozen_bias_training_mediated_results.json`).
No changes to `sim_frozen_bias_direction.py` (imported unmodified, as
before). CPU-only; no GPU touched; run via a local disposable venv
(`python3 -m venv` + `pip install torch --index-url
https://download.pytorch.org/whl/cpu`) since this dev machine's system
Python had no `torch` installed this session — noted here since a fresh
run of this script from a clean shell will hit the same
`ModuleNotFoundError` this session did, and needs the same one-time setup,
not a GPU/remote box.

---

## REVISION LOG — 2026-07-06, independent-attack-round response (KILL-as-written / REVISE-THEN-PROCEED → this pass)

Every attack finding, its fix, and where it landed, so a fresh attack round
can trace every change without re-deriving what moved and why.

| # | Severity | Attack finding (one-line) | Fix applied | Landed in |
|---|---|---|---|---|
| 1 | FATAL | Primary bar's direction (span_frac must FALL) was asserted, never derived; the scale ladder (rising span_frac with capability) and a collinearity argument both cut against the assumed sign | Built `deltanet_rd/sim_frozen_bias_direction.py` (CPU-only), importing the REAL `gram_deviation`/`chunk_key_gram_stats`/`summarize_gram_records`/`anchors`/`span_frac`/`random_unit_rows_init` functions unmodified (via a documented, never-called `fla` import stub — real `fla` absent on the CPU dev box). Ran a 2-regime (naive i.i.d.-random baseline vs. calibrated-to-archived-reference baseline) + λ-sweep + Zipf-skew-sweep + continuous crossover scan. Result: direction depends on baseline non-orthonormality; every REAL archived Arm-1 reference (span_frac 0.248–0.389) sits far above the ≈0.03 crossover point, deep in the "falls" regime, with a stable, low-variance signal (mean Δ=−0.2015, std=0.00395, 5/5 seeds). Bar kept ONE-SIDED (falls), now on evidence, not assumption | §1 (revision note), §7.1 (full derivation + numbers), results JSON at `deltanet_rd/results/frozen_bias_sim/sim_frozen_bias_direction_results.json` |
| 2 | MAJOR | Missing control: no arm isolates "keyed by token identity" from "any constant vector helps" | Promoted Arm 2′ (global single frozen vector, same table's mean row, same λ, no per-token lookup) from attack-round prose to a MANDATORY registered arm at BOTH rungs, with its own budget rows; added §7.1a as a new mandatory comparison bar (Arm 2 must beat Arm 2′) that licenses the transplant claim | §5 (new Arm 2′ subsection), §6.1/§6.2 (manifest counts), §7.1a (new bar), §8.3/§8.5 (budget rows), §1.1/§1.2 (CONFIRM/REFUTE operationalization) |
| 3 | MAJOR | λ≈0.58 was transplanted from a closed synthetic grammar with a different loss landscape; the only check (Arm 3, learned λ) was optional and cut first | Promoted a 1-seed λ∈{0.3, 0.58, 0.8} mini-sweep (1 corpus, openr1-mix-ext) to MANDATORY rung-1 scope, run alongside calibration, budgeted explicitly. Added §1.2a's three-way CONFIRM/REFUTE/λ-mismatch classification | §5 (new mini-sweep subsection), §6.1 (manifest), §1.2/§1.2a (outcome classification), §8.3/§8.5 (budget) |
| 4 | MINOR/MAJOR | The flat +5% relative val-loss tolerance was imported from Track B by precedent, never derived from this program's own (likely tighter) seed spread | Replaced with `mean_ref + k·s_ref` derived from Arm 1's OWN fresh rung-1 per-seed val-loss spread, computed and blind-pinned in `BANDS_PINNED-FrozenBias.json` BEFORE any Arm 2/Arm 2′ inspection (same pattern §7.3 already used for `span_frac`). k=2, explicitly justified: verified Track B's own +5%/+2% figures are flat relative percentages with no `k` to inherit (Track B never derived its own tolerance from measured spread either); k=2 is this codebase's OWN actual `mean_ref ± k·s_ref` house convention, from `KEY_ANCHORING_DESIGN.md` §3.6 / `key_anchoring.derive_engaged_bands` (`engaged_k = mean_ref + 2.0*s_ref`) — reused for consistency, one k across every bar in this document | §7.2 (rewritten derivation), §7.3 (pin scope expanded to include val loss) |
| 5 | MAJOR | Gate contradiction: §6.2 said rung-1 must pass "with no gate violation" but never stated whether INCONCLUSIVE (a split-corpus outcome) counts as passing for LAUNCH-AUTHORIZATION (vs. headline-reporting) purposes | Rewrote §6.2's operative text: rung-2 launches ONLY IF rung-1's primary bar is hit on BOTH corpora AND Arm 2 beats Arm 2′ on BOTH corpora; a split on EITHER condition is INCONCLUSIVE and does NOT authorize rung-2 spend. Removed every "≥1 corpus" reading | §6.2 (rewritten gate, 3 explicit conditions), §7.1 (headline pass condition cross-referenced), §9 (recap) |
| 6 | MINOR | Abort-rule asymmetry: the 2×-anomaly check only covered the rung-1→rung-2 BOUNDARY, not rung-2's own internal calibration-vs-manifest step | Extended §8.4's abort rule: if rung-2's OWN mandatory calibration cell measures ≥1.5× its own banked rate, halt the remaining rung-2 cells and diagnose first — same discipline one level down, not just at the rung boundary | §8.4 (extended abort rule) |
| 7 | MAJOR | Instrument validity never checked: nothing verified the probe's forward hook (which captures `k_conv1d`'s output) actually observes the POST-bias key once this program's new frozen-bias hook exists, rather than a pre-bias key that happens to look similar | Added a MANDATORY Wave −1 smoke item: run the actual probe against an Arm-2-configured tiny checkpoint and assert (executed test, not a read) that the probe-captured key population is bit-identical (`torch.equal`) to a separately-captured internal post-blend side-channel, on a spot-checked batch | §8.0 (new mandatory Wave −1 smoke, exact mechanical check specified), §9/§10 (cross-referenced) |
| 8 | MINOR | Seeds were never pre-registered as literal integers, and no check was made that they're distinct from every archived run's own seeds | Grepped every archived result JSON's `"seed"` field (`experiment-runs/`); registered this program's own seeds as {0,1,2} (a deliberate, disclosed REUSE of Track C's own convention, not a novel draw — reasoning given); registered the seed-contingency draw order as {5,6,8,9} (the first integers NOT already archived below 13, since 3 and 4 are themselves already used 48/44 times respectively) | §6.0 (new subsection, full seed census + reasoning) |
| Budget | — | Redo §8 end-to-end for the new mandatory scope, 2× contingency on everything, confirm ≤135 GPU-h | The promoted mandatory scope (Arm 2′ both rungs + λ mini-sweep) no longer fits at the pre-revision's illustrative 20,000-step/run planning target for BOTH rungs (≈152.8–156 GPU-h at 2×, ≈18–22 GPU-h over ceiling). Cut order applied explicitly: (1) rung-2 has no optional arms to cut — stated, not skipped; (2) rung-1's own optional Arm 3 cut-if-needed (saves 3.02 GPU-h at 2×, not sufficient alone); (3) rung-2's own illustrative PLANNING step-count target reduced 20,000→15,000 (disclosed, not a stealth methodology cut — §6.3's calibration run still fixes the real count). Final total: ≈117.18 GPU-h (Arm 3 cut) to ≈120.20 GPU-h (Arm 3 kept), both under the 135 GPU-h ceiling, with ≈14.8–17.8 GPU-h headroom | §8.1 (ceiling note), §8.3/§8.3a (arithmetic + cut order), §8.4 (contingency), §8.5 (redone summary table) |

### What could NOT be fixed this pass, and why

- **Attack-round self-attack item 3 (§11, unrelated to the 8 independent
  attack findings above — this design's own pre-existing disclosed gap):**
  `span_frac` has never been shown to correlate with any user-legible
  behavioral improvement in a real LM. No fix was attempted this pass — the
  design's own §4.c already registers a hop-recall-suite follow-on as
  explicitly out of scope for this wave, and nothing in the 8 independent
  attack findings asked for this to change. Left as a disclosed, standing
  limitation, not silently dropped.
- **The Arm 2′ vs. Arm 2 comparison's own predicted outcome (§7.1a) rests on
  one particular sim proxy for "what a real trained baseline's raw keys
  look like" (§7.1's `shared_frac` construction) — the sim predicts Arm 2
  beats Arm 2′ comfortably (Arm 2′ actually RAISES `span_frac`, Arm 2 lowers
  it, `delta_arm2_minus_global`≈−0.62 to −0.65), but this is disclosed as a
  sim prediction under a proxy construction, not a guarantee about real
  training data.** This could not be resolved further this pass without
  either running the real GPU comparison (out of scope for a CPU-only
  revision pass) or over-claiming confidence the sim's own proxy-baseline
  caveat does not support. Flagged, not resolved — the mandatory §7.1a
  comparison is what actually settles this on real data.
- **The rung-2 planning-step-count cut (§8.3a) is itself a real, if
  disclosed, reduction in rung-2's own token budget relative to the
  pre-revision plan** (15,000 vs. 20,000 steps/run, both illustrative). This
  was the least-bad lever available given the instruction to preserve every
  arm/seed/corpus the attack round made mandatory — but it is disclosed
  here as a genuine trade-off, not spun as cost-free: if §6.3's own rung-2
  calibration run later shows 15,000 steps under-trains the 98M config
  relative to what a real comparison needs, this headroom question will
  need to be revisited against the ≈14.8–17.8 GPU-h remaining, not silently
  absorbed.

---

## ROUND-3 REVISION LOG — 2026-07-06, independent round-2-attack response (observable REDESIGN, not a patch)

**This is a redesign of the PRIMARY OBSERVABLE, prompted by a FATAL round-2
finding that invalidated the round-2 bar itself, not an incremental fix
list layered on an intact bar.** Every round-2 finding, its fix, and where
it landed, mapped so a fresh (round-3) attack round can trace every change.

| # | Severity | Round-2 finding (one-line) | Fix applied | Landed in |
|---|---|---|---|---|
| 1 | FATAL | AUTO-PASS ARTIFACT: the round-2 primary bar (Arm 2 post-blend `span_frac` ≥2 SD below Arm 1) passes mechanically — dense blending toward 50,257 near-orthogonal rows pins the measured population's `span_frac` to a narrow ~0.02–0.09 band regardless of baseline structure (`crossover_scan`: `dense_arm2_span_frac` moves 0.0218→0.0936 while baseline moves −0.00002→0.3536, a ~5× narrower range). Arm-2-vs-Arm-2′ does NOT guard it — confirmed this round that Arm 2′ pins to an even more extreme ~0.61–0.71 band, the same artifact class, opposite direction. | Redesigned the primary observable around Arm 1′ (Arm 1's own checkpoint, IDENTICAL blend applied ONLY at eval time) as the comparator for Arm 2, and the analogous Arm 1″ for Arm 2′. Both sides of each new comparison share the identical mechanical pin by construction, so any surviving difference is training-mediated. Validated with a NEW CPU sim (`sim_frozen_bias_training_mediated.py`): null case produces EXACT zero delta (5/5 seeds, `torch.equal`-level, not a tolerance check); sensitivity case detects a synthetic training-mediated effect starting at `effect_strength≈0.3` against a realistic cross-seed noise floor (`s_ref=0.002541` at N=5 pairs). Promoted pre-blend `k_raw` geometry (no blend present, no artifact possible) from "strongly recommended" to CO-PRIMARY, since the sim shows it is uniformly MORE sensitive than the post-blend comparison at every tested effect size and λ. | §1 (ROUND-3 revision note), §4.a/§4.a-i (co-primary), §5 (new Arm 1′/Arm 1″), §7.1/§7.1a (redesigned bars), §1.1/§1.2/§1.2a/§1.3 (re-rolled CONFIRM/REFUTE/λ-mismatch/uninterpretable), results JSON at `deltanet_rd/results/frozen_bias_sim/sim_frozen_bias_training_mediated_results.json` |
| 2 | FATAL | SIM-TO-TRAINING GAP: the round-2 sim blends a STATIC population post-hoc; the real Arm 2 trains `k_raw` through the bias for 15k+ steps, and SGD can cancel/exploit/amplify it — no caveat or fallback outcome existed for this in the design. | Registered a FOURTH pre-registered outcome, "sim-training divergence — informative, not REFUTE" (§1.3), triggered if the real LM result contradicts every sim prediction here (direction disagreement between co-primaries, or violation of the λ-sensitivity gradient). Registered its own cheap diagnostic: cosine of trained `k_raw` against `B[token_id]`, before/after training, both arms — a rising cosine in Arm 2 only is direct evidence SGD is compensating for the bias rather than ignoring it. | §1.3 (new fourth outcome + diagnostic), §9 (recap) |
| 3 | MINOR | F7 (Wave −1 probe-validity smoke, §8.0) was worded as if already executed; it is a SPECIFICATION, not yet run on the built hook. | Rewrote §10's Wave −1 smoke bullet to state STATUS: OPEN explicitly, distinguished from the ALREADY-EXECUTED CPU sims (`sim_frozen_bias_direction.py`, `sim_frozen_bias_training_mediated.py`), which HAVE been run this session with numbers on record. | §10 (status line rewritten) |
| 4 | MINOR | Seed census said seed 3 appears 43 times in `experiment-runs/`; re-grepping this session (`grep -rho '"seed":\s*3\b' experiment-runs/ \| wc -l`) gives 48, not 43. | Corrected the count to 48, stated the exact grep command/scope used so the number is independently reproducible, and noted the correction does not change this program's own seed choice ({0,1,2}, unaffected). | §6.0 |
| 5 | MINOR | No single sentence clarified the `±`/`≥`/`≤` sign convention shared across the (now four) bars in §7, risking a misread of which direction "clears" a bar. | Added a one-sentence sign-convention note at the top of §7: every geometry bar clears when `X ≤ Y − k·s_ref` (X must be BELOW/more-negative-than Y), never the reverse; the val-loss gate (§7.2) is the one bar with the opposite-direction "must not exceed" framing, called out explicitly. | §7 (new note before §7.1) |
| 6 | MINOR | Rounding inconsistency across the document: §8.4 quoted 117.08/120.10 GPU-h while §8.5's table quoted 117.18/120.20 for the SAME totals — a stray rounding-down in §8.4 that was never reconciled. | Recomputed every 2×-contingency figure from full-precision sub-totals (not from already-rounded display numbers): `5.03×2=10.06` (not 10.05), `53.51×2=107.02` (not 107.03). §8.4 and §8.5 now both read 119.18/122.20 GPU-h (post-round-3, including the new eval-only line item) and agree to the decimal place. | §8.4, §8.5 (both reconciled) |

### Budget impact of the ROUND-3 redesign itself

**The redesign adds ZERO new training runs** (Arm 1′/1″ reuse Arm 1's own
checkpoints; §5/§6.1/§6.2 training-run counts are byte-identical to
round-2's own 20+18 mandatory training runs). **It adds 50 new eval-only
measurement passes** (26 at rung-1, 24 at rung-2; §6.1/§6.2/§8.0a) at an
estimated **≈2.00 GPU-h at 2× contingency** (§8.0a) — this is the ONLY
budget delta versus round-2's own total. **New total: ≈119.18 GPU-h (Arm 3
cut) to ≈122.20 GPU-h (Arm 3 kept)**, both still well under the 135 GPU-h
ceiling, with ≈12.80–15.82 GPU-h headroom remaining (down from round-2's
≈14.8–17.8 GPU-h, a ≈2 GPU-h reduction entirely attributable to the new
eval-only passes, §8.5).

### What could NOT be fixed this pass, and why (ROUND-3, in addition to the round-2 list above, which still stands unresolved where noted)

- **The design is NOT killed by its own null-vs-injected sim — but the sim's
  own injected-effect construction is itself a proxy, disclosed as such
  (§11a, new attack question 1).** The null case is proven exact by
  construction (not a proxy — it follows directly from code-path identity),
  but the SENSITIVITY claim ("the bar would detect a real training-mediated
  effect if one exists, at a plausible size") rests on one particular
  synthetic-effect family. This is the same category of limitation as
  round-2's own disclosed sim-proxy caveats (§7.1a's old text, retained in
  spirit) — carried forward honestly, not resolved, because resolving it
  further requires either a real training run (out of scope for a CPU-only
  revision pass) or a broader sim sensitivity-family sweep than this pass's
  budget covered (flagged as round-3 attack question 1, §11a).
- **The Arm-2′-vs-Arm-1″ null-exactness claim is verified by analogy to
  Arm-2-vs-Arm-1′'s own proof (same code-path-identity argument), not by an
  independently-run full null-vs-injected grid specific to the
  global-vector construction** (§7.1a, §11a attack question 5). A single
  confirming data point (one seed, one `shared_frac`) was checked and
  matched exactly; the FULL sensitivity grid was not separately re-run for
  this construction, since the design judged the underlying mechanism
  (identical blend function applied to two raw-key populations) to make a
  separate grid redundant — disclosed as a scope limitation a fresh attack
  round should test, not assumed silently clean.
- **The eval-only cost estimate (§8.0a, ≈0.02 GPU-h/pass) is an ESTIMATE,
  not a measured figure**, unlike the training-run arithmetic in §8.3, which
  rests on already-measured Wave-C/Track-C timing constants. Flagged as
  round-3 attack question 4 (§11a) — a real calibration measurement of one
  eval-only pass at rung-2 scale, before the full rung-2 manifest launches,
  would close this, but was not performed this pass (CPU-only revision, no
  GPU access).
- **Every "what could NOT be fixed" item from the round-2 REVISION LOG above
  still stands, unresolved by this pass**, except item 2 (Arm-2′-vs-Arm-2's
  predicted sim outcome), which is superseded by this round's own redesign
  of that comparison (§7.1a) rather than independently resolved — the
  round-2 sim's "Arm 2 beats Arm 2′" finding under the OLD scheme is now
  understood to be an artifact of the old scheme itself, not evidence
  either for or against the mechanism under the new one.

---

## VERDICT — Rung-1 training + measurement complete (2026-07-06)

**All 20 mandatory training cells ran to completion** (18 core cells: 2
corpora × {off, per-token λ=0.58, global-vector λ=0.58} × 3 seeds; 2
mini-sweep cells: per-token at λ∈{0.3, 0.8}, openr1-mix-ext, seed 0 —
§5/§6.1's registered manifest, byte-for-byte), followed by the full
measurement pipeline (46 retrofit re-evals, `BANDS_PINNED-FrozenBias.json`,
`PHASE_D_FULL_REPORT.json`, the cosine diagnostic). Every number below is
quoted from those archived JSONs (`experiment-runs/2026-07-06_frozen_bias_rung1/`,
this repo) and independently re-derived once as a cross-check (see
"Verification," below) — nothing here is estimated or recalled from
memory.

### Headline numbers

**PRIMARY (§7.1-real.1, Arm 2 − Arm 1′, post-blend `span_frac`, pinned
t(2,.975)=4.303 CI):**

| Corpus | mean Δ | 95% CI | Excludes zero? | Sign vs. every sim prediction |
|---|---|---|---|---|
| openr1-mix-ext | **+0.1955** | [0.0937, 0.2973] | Yes | **Opposite** (sims predicted negative) |
| wikitext-mix-ext | **+0.2273** | [0.0926, 0.3621] | Yes | **Opposite** |

**CO-PRIMARY (§4.a-i, Arm 2 − Arm 1′ on pre-blend `k_raw` geometry, same
pinned formula):**

| Corpus | mean Δ | 95% CI | Excludes zero? |
|---|---|---|---|
| openr1-mix-ext | **+0.1097** | [0.0491, 0.1704] | Yes |
| wikitext-mix-ext | **+0.1345** | [0.0070, 0.2621] | Yes |

The co-primary moves in the **same direction** as the primary in both
corpora — this rules out the specific §1.3 "suspicious result" pattern
(post-blend-only win while the more-sensitive pre-blend instrument stays
null) and confirms the effect is training-mediated, not a static
blend-arithmetic artifact of the post-blend measurement alone.

**CONTROL (§7.1a, Arm 2′ − Arm 1″, global-vector bias, identical
artifact-free footing):**

| Corpus | mean Δ | 95% CI | Direction |
|---|---|---|---|
| openr1-mix-ext | **−0.3319** | [−0.6362, −0.0276] | Negative (stabilizing) |
| wikitext-mix-ext | **−0.2308** | [−0.2838, −0.1777] | Negative (stabilizing) |

**§7.1a licensing check FAILS**: the per-token construction's
training-mediated delta (+0.1955/+0.2273) is not more negative than the
global-vector construction's (−0.3319/−0.2308) — in fact it has the
opposite sign entirely. Per §7.1a's own pre-registered reading, this is
literally the REFUTE-adjacent pattern for the "is this about per-token
keying specifically" question (the global-constant control shows a
*larger-magnitude, stabilizing* effect than the per-token arm), while
simultaneously the primary/co-primary pair clears its own CI-excludes-zero
bar in the *positive* direction — the two readings are not
in conflict, they are simply about different questions (per-token vs.
constant, and each vs. its own artifact-matched control), but neither one
is the pre-registered CONFIRM.

**Cosine diagnostic (§1.3's own instrument):** cosine similarity of each
trained `k_raw` against its own frozen anchor row `B[token_id]` is ~0 in
every arm, both at an init proxy and after training (e.g. Arm 2 final:
0.0017 / 0.0007 across the two measured dims; Arm 1 final: −0.0043 /
0.0062) — no evidence SGD is aligning raw keys toward the frozen anchors
in either arm. This rules out "key-anchor alignment" as the mechanism
behind the positive primary delta; whatever is driving Arm 2's `span_frac`
increase, it is not the entity-alignment mechanism the testbed's own
account (`KEY_ANCHORING_DESIGN.md` §10.13.4) proposed.

**Val-loss gate (§7.2):** PASSES in both arms, both corpora (Arm 2 mean
2.1184/4.3426 vs. ceiling 2.1935/4.3828 openr1/wikitext; Arm 2′ mean
2.1526/4.3614, same ceilings) — the positive `span_frac` shift is not
purchased with a capability regression on next-token loss.

**λ mini-sweep (n=1 per cell, openr1-mix-ext, per-token only):** detected
delta +0.038 (λ=0.3), +0.219 (λ=0.58), +0.290 (λ=0.8) — monotonically
increasing with λ, no sign flip anywhere in the tested range. This is the
opposite of the sim's own pre-registered sensitivity-falls-as-λ-rises
gradient (§1.1's family-A/B figures) for the *magnitude of a detected
synthetic effect* — here the *real* detected delta grows with λ, which is
consistent with a real, training-scale effect that gets stronger as more
of the key is replaced by the frozen bias, not with an artifact whose
apparent size is an inverse function of λ alone.

### Classification: FOURTH OUTCOME, "sim-training divergence" (§1.3) — verbatim

§1.3 of this document pre-registers, before any real training cell ran,
exactly this contingency:

> **"(ROUND-3, new — the fourth pre-registered outcome, round-2 Finding 2)
> Sim-training divergence: informative, not REFUTE.** The round-2 attack's
> second FATAL finding was that the sim blends a STATIC population
> post-hoc, while the real Arm 2 trains `k_raw` through the bias for 15k+
> steps, and SGD could cancel, exploit, or amplify the bias in ways no
> static sim can anticipate. If the real LM comparison behaves in a way
> inconsistent with every sim prediction here (e.g. the pre-blend
> co-primary moves opposite to the post-blend primary, or the
> λ-sensitivity gradient is violated), this is reported as **"sim-training
> divergence"** — explicitly NOT a REFUTE and NOT silently absorbed into a
> CONFIRM either, but its own disclosed, informative outcome."

**This is exactly what happened, via the second listed trigger.** The
pre-blend co-primary does NOT move opposite to the post-blend primary
(both are positive, so that specific sub-trigger did not fire) — but the
λ-sensitivity gradient IS violated: every sim family in this design
(§1.1, §1.2a, the ROUND-5 fix-6 family-B extension) predicted the
*post-blend bar's own sensitivity* falls monotonically as λ rises, and
separately predicted the mechanism's effect, where present at all, would
be **negative** (stabilizing) at every tested λ. The real λ-sweep shows a
**positive, monotonically-growing-with-λ** effect — the sign itself is
wrong relative to every sim family's prediction, at every λ, which is a
stronger and more direct divergence than the gradient-shape violation
§1.3 names as its example trigger. **Verdict: NOT a CONFIRM (the sign is
wrong relative to the registered mechanism), NOT a REFUTE (the pre-blend
co-primary is not null — quite the opposite, it moved with a clean,
CI-excluding-zero effect in both corpora) — this is the fourth outcome,
sim-training divergence, exactly as pre-registered.**

### The descriptive-tier caveat — the blind-pin fired, and why

**§7.3's blind-pin (`BANDS_PINNED-FrozenBias.json`) is the load-bearing
mechanism preventing any judgment call in this design from being made
after seeing Arm 2/Arm 2′'s trained data.** Per its own pinned timestamp
(`pinned_at_iso: "2026-07-06T14:27:24Z"`), the pin file was written
**after** all 20 training cells had already completed and their result
JSONs already existed on disk. This means the pin correctly did its job —
it fixed Arm 1′/Arm 1″'s own reference quantities from Arm 1's real,
already-measured data rather than letting them be tuned after seeing
Arm 2 — but it does **not** retroactively grant the blinding property the
design's own §8.6 (writer/gate/readout separation) was built to provide,
which requires the pin to exist **before** the arm being measured against
it has trained. **This wave's own pin was, by construction, a
post-hoc pin on an already-trained wave.** Per the pin's own registered
rule (blinding requires pre-training placement), this result is correctly
placed in the **DESCRIPTIVE TIER**, not the confirmatory tier its CI
math would otherwise suggest — the CI-excludes-zero calculation is
numerically real and independently re-verified (below), but the
*confirmatory license* that would come from a true pre-registered blind
was not available this wave, because the pin was constructed after the
fact rather than staged ahead of the training launch.

**Process finding for future waves, stated plainly so it is not
repeated:** the blind-pin must be written and committed **before** the
training manifest launches, not after training completes and before the
measurement/fit scripts run. This wave's operational sequence (build →
launch all 20 cells → THEN construct the pin from the resulting Arm 1
data) preserved the pin's narrower guarantee (Arm 1′/1″ references are
fixed relative to Arm 2/2′, so no result-dependent tuning of the
*reference* happened) but forfeited its broader one (a reader cannot be
assured the *entire measurement protocol*, including which arms/bars to
report, was fixed before any trained result existed). Any future
frozen-bias-style wave should treat "pin before launch" as a hard
sequencing rule, on par with the standing calibration-before-sweep rule.

### The control contrast — the single most striking number in this wave

**Per-token frozen key bias and global-vector frozen key bias produce
opposite-signed, both-significant, both-training-mediated effects on the
same observable, in the same architecture, at the same λ, on the same
two corpora:**

- Per-token (Arm 2 vs. Arm 1′): **+0.1955 / +0.2273** — training through
  a dense, per-token-identity-keyed frozen bias makes post-blend
  `span_frac` **larger** (MORE collapsed / destabilized — CORRECTED
  2026-07-07, sec 12.0: the original parenthetical here read "more
  spread, less collapsed", which inverts the instrument's direction;
  verified numerically — identical keys give span_frac=1.0, i.i.d.
  random keys ≈0.01, and fit_frozenbias_estimation.py's own
  mechanism_direction="negative" convention agrees) than the
  artifact-matched no-bias control.
- Global-vector (Arm 2′ vs. Arm 1″): **−0.3319 / −0.2308** — training
  through the *same frozen table's single mean row*, added identically to
  every token's key with no per-token lookup at all, makes post-blend
  `span_frac` **smaller** (LESS collapsed / stabilized — CORRECTED
  2026-07-07, same inversion as above) than its own artifact-matched
  no-bias control.

Both deltas exclude zero at 95% CI in both corpora. Neither sim family in
this design predicted this split — every sim family predicted a
stabilizing (negative) direction for the training-mediated effect,
regardless of whether the bias was per-token-keyed or a single constant
vector. **The real result is that the two constructions are not just
different in degree, they are different in sign** — dense per-token
keying and a single constant vector, at matched λ and matched training
budget, pull the same geometric observable in opposite directions. This
is a genuinely surprising, precisely-measured geometry finding on its own
terms, independent of whether it confirms the original transplant
hypothesis (it does not).

### What this does NOT show

- **No capacity or mechanism claim.** The cosine diagnostic rules out
  key-anchor alignment specifically, but does not identify what IS
  driving either the per-token or the global-vector training-mediated
  shift. Nothing in this wave's instrumentation supports a positive
  mechanistic account of either direction.
- **`span_frac` has no established behavioral correlate in an LM** —
  §4.a's own standing disclosure, unchanged by this wave's result. A
  positive (or negative) shift in this observable has never been shown to
  track next-token prediction quality, downstream task performance, or
  any other user-legible capability in this or any prior program. The
  val-loss gate passing means there is no *loss* cost, but that is a
  different claim from "the geometry change is behaviorally meaningful."
- **No licensed rung-2 launch.** §1.3's own uninterpretable-result
  exclusion and the descriptive-tier caveat above both independently
  block treating this as a clean CONFIRM that would authorize further
  spend under §6.2's gate. Rung-2 remains PARKED.
- **No claim that the constancy-suffices mechanism (as validated in the
  synthetic-grammar testbed) is present or absent at LM scale** — the
  fourth-outcome classification exists precisely because the evidence
  does not cleanly support either "yes it transplants" or "no it doesn't."

### Verification performed this pass

Recomputed the primary delta (Arm 2 − Arm 1′, openr1-mix-ext,
post-blend `span_frac`) directly from the raw per-seed values in
`fitinput_arm2_post_blend.json` (`[0.31476, 0.37706, 0.26591]`) and
`fitinput_arm1prime_post_blend.json` (`[0.09594, 0.15754, 0.11774]`),
using the pinned formula (`mean_delta ± t(2,.975=4.303)·s/√3`):
**mean_delta = 0.1955009366341799, CI = [0.0936538066504167,
0.2973480666179431]** — matches `PHASE_D_FULL_REPORT.json` /
`estimation_primary_arm2_vs_arm1prime.json` to full float precision, zero
discrepancy.

**Realized GPU-h.** Summed `wall_s` across all 20 `frozenbias_lm_*.json`
training-cell result files: **18175.744 s = 5.0488 GPU-h** (20/20 cells
present, tight 899–914s band, no crashes or retries). Adding the ~1.6
GPU-h of retrofit/measurement eval work (46 retrofit passes + cosine
diagnostic + fit, all short forward-pass-only jobs) and the ~0.25 GPU-h
calibration-run prior (rung-1 single-cell calibration + smoke suite):
**≈6.90 GPU-h realized for the entire rung-1 wave**, against the
program's own 135 GPU-h ceiling (~14.2 GPU-h committed at 2× contingency
for rung-1 alone) — well under budget in either accounting.

### The honest scientific value of this wave

This wave does not confirm the hoped-for result: **the testbed's
constancy-suffices gain (candidate (e), `KEY_ANCHORING_DESIGN.md` §10.13)
does NOT straightforwardly transplant via a dense per-token frozen key
bias at 14M-parameter LM scale**, stated plainly and without spin. What it
delivers instead is a precisely-measured, controlled, and genuinely
surprising training-mediated geometry effect: a per-token frozen key bias
and a global-constant frozen key bias, trained under otherwise identical
conditions, move the same observable in opposite, both-significant
directions — a result no sim family in this design anticipated, caught
only because the design's own co-primary and control instruments were
built before training, precisely so that a divergence like this one would
be visible and reportable rather than silently absorbed into a single
pass/fail bar. This is a discovery-shaped negative-plus-surprise, not a
transplant confirmation: the original hypothesis did not survive, but the
wave's instrumentation surfaced a real, controlled, reproducible
(3-seed, 2-corpus) geometric phenomenon worth a follow-up mechanism study
in its own right, should the project choose to pursue one.

---

## 12. MECHANISM-WAVE — follow-up design for the sim-training-divergence puzzle (opened 2026-07-07, this session)

### 12.0 Scope, framing, and one verified documentation correction

**This is an explicitly EXPLORATORY/MECHANISTIC follow-up, not a further
confirmatory test of the original transplant hypothesis.** The rung-1
wave's own fourth-outcome classification stands, unchanged, unrevisited
by anything below. Every hypothesis here is being written with knowledge
of rung-1's own measured direction (there is no way to un-know it), so
nothing in §12 can claim the blind-pin discipline §7.3/§10 built for the
primary bars — every reading below is Tier-2 descriptive/exploratory BY
CONSTRUCTION, same tier the VERDICT's own geometry finding already
occupies, never elevated to confirmatory regardless of how clean a number
comes back.

**A verified correction to the VERDICT section's own plain-language gloss
(numbers unaffected, wording only) — load-bearing for how every hypothesis
below is phrased, so stated first, precisely, before anything else:**
the VERDICT section (§ above, lines ~3086 and ~3091 of this file) glosses
the per-token delta `+0.1955/+0.2273` as "span_frac larger (**more
spread, less collapsed**)" and the global-vector delta `−0.3319/−0.2308`
as "span_frac smaller (**more collapsed**)". This pass re-derived
`span_frac` numerically from its own registered formula
(`analyze_probe_wave2.py`'s `anchors`/`span_frac`, reused unmodified
throughout this design) on three synthetic populations (K=16, d=16,
pure-Python, no framework dependency): i.i.d. random unit keys give
`span_frac≈0.012` (near the "random" anchor, by construction); **fully
collapsed keys (all K rows identical) give `span_frac=1.000` exactly**
(the "collapse" anchor, by construction); a population blended 70% toward
one shared direction gives `span_frac≈0.78` (between the two, much closer
to collapse). **This proves the opposite of the VERDICT's own
parentheticals: HIGHER span_frac is MORE collapsed / less differentiated,
LOWER (more negative) span_frac is MORE spread / more differentiated** —
consistent with `fit_frozenbias_estimation.py`'s own hard-coded
`mechanism_direction="negative"` default (a CONFIRM requires the delta to
be NEGATIVE, i.e. the hoped-for constancy-suffices mechanism is expected
to make span_frac FALL) and with §4.a-i's own prose ("Arm 2's OWN k_raw...
should show LESS non-orthonormal drift... i.e. the SAME one-sided
'falls' direction"). **Corrected reading, numbers unchanged:** Arm 2
(per-token, `+0.1955/+0.2273`) — training makes post-blend keys MORE
mutually collinear / MORE collapsed, the OPPOSITE of the hoped-for
mechanism. Arm 2′ (global-vector, `−0.3319/−0.2308`) — training makes
post-blend keys LESS collinear / LESS collapsed, i.e. in the
mechanism-predicted, stabilizing direction. **This matches this task's
own framing exactly** ("per-token bias made write-geometry LESS
stable... global-vector bias made it MORE stable") — it is specifically
the VERDICT section's own two parentheticals that read backwards. Every
hypothesis below uses the corrected direction throughout; the VERDICT
section's own two parentheticals now carry this same fix directly
(correction applied in working tree, to be committed with this design),
consistent with §7.1/§7.1a/§9's own pinned
`mechanism_direction="negative"` convention.

**Pre-check performed this pass (read-only, zero GPU, no training,
no code changed on the box):** `ssh youthful-indigo-turkey` (the same
Brev box rung-1 trained on) confirms (a) all **400** per-checkpoint `.pt`
files are present at `/data/deltanet_rd_frozenbias_ckpts/` — exactly
20 training cells × 20 checkpoints (every 1,000 steps, matching §5's
registered cadence), not merely the final-step checkpoints the existing
retrofit tool consumed; (b) `/data` has 15TB free (2.6TB/18TB used); (c)
`lm_pretrain_rd.py`, `lm_attractor_probe_rd.py`,
`frozen_bias_retrofit_eval_rd.py`, `key_anchoring.py` are present,
unmodified, at `/home/nvidia/chapter2/deltanet_rd`; (d) GPUs 2-7 were idle
(0 MiB used) at check time, GPUs 0/1 were at 100% util (the concurrent
capacity-cliff wave the rung-1 README already discloses). **This
materially de-risks every eval-only item below (the intermediate
checkpoints are the input every Stage-0/Stage-1 item needs) but must be
re-verified immediately before a BUILD agent executes anything** — a
disk-pressure cleanup or a concurrent wave's own launch could have changed
either fact since this check.

### 12.1 Verified numbers this wave's hypotheses must explain (re-cited, not re-derived)

| Quantity | openr1-mix-ext | wikitext-mix-ext | Source |
|---|---|---|---|
| Primary, post-blend `span_frac` Δ (Arm2−Arm1′) | +0.1955 [0.0937,0.2973] | +0.2273 [0.0926,0.3621] | `estimation_primary_arm2_vs_arm1prime.json` |
| Co-primary, pre-blend `k_raw` `span_frac` Δ (Arm2−Arm1) | +0.1097 [0.0491,0.1704] | +0.1345 [0.0070,0.2621] | same file |
| Control, post-blend `span_frac` Δ (Arm2′−Arm1″) | −0.3319 [−0.6362,−0.0276] | −0.2308 [−0.2838,−0.1777] | `PHASE_D_FULL_REPORT.json` |
| Cosine(trained k_raw, own frozen anchor row), Arm2 final | 0.0017 / 0.0007 (2 dims) | — | `cosine_diagnostic_sec1_3.json` |
| Cosine, Arm1 final (never trained through bias) | −0.0043 / 0.0062 | — | same file |
| Val-loss gate | PASS both arms | PASS both arms | `PHASE_D_FULL_REPORT.json` |
| λ-mini-sweep (n=1, per-token, openr1 only) | +0.038 (λ=.3) / +0.219 (λ=.58) / +0.290 (λ=.8) | — | same file |
| Realized GPU-h, entire rung-1 wave | 6.90 / 135 ceiling | | VERDICT §, this file |
| Per-training-cell wall time | 899–914s (≈0.25 GPU-h) | | `frozenbias_lm_*.json` `wall_s` |
| Per-retrofit-eval-pass wall time (avg, all 7 corpora/call) | ≈0.0348 GPU-h (≈125s) | | 1.6 GPU-h / 46 passes |

Nothing in §12 disputes any of these figures — they are the fixed
explanandum. The puzzle, restated with the corrected direction (§12.0):
**why does training through a per-token frozen key bias make keys MORE
collapsed, while training through a global-constant frozen key bias makes
keys LESS collapsed, at matched λ, matched architecture, matched budget,
with no evidence of anchor-cosine alignment in either arm?**

### 12.2 Hypothesis table

| # | Statement | Primary test statistic | New GPU? | Stage | Est. cost |
|---|---|---|---|---|---|
| H1 | Token-identity structure: the per-token bias lets training homogenize raw keys for REPEATED occurrences of the same token within a chunk (their distinguishing job is now redundant with the bias), inflating within-chunk collinearity; the global bias offers no per-token handle to exploit, so no analogous repeat-specific homogenization occurs. | `repeat_excess` (defined §12.4) | Yes (forward-pass only, reuses existing ckpts) | 1 | ≈0.21–0.63 GPU-h |
| H2 | Effective-dimension corroboration: the SAME Gram-spectrum that produces `span_frac` also produces `effective_rank`/`stable_rank` (already archived, unused as a headline). If the collapse story is real, Arm2 should show LOWER effective/stable rank than Arm1′, Arm2′ HIGHER than Arm1″. | `effective_rank_mean`, `stable_rank_mean` deltas, same pinned CI | **No — zero GPU, already-archived JSON fields** | 0 | ≈0 |
| H3 | Optimizer/gradient-flow interaction: the λ-blend's `normalize()` Jacobian, evaluated at a per-token-random vs. a fixed-global bias direction, differentially reshapes the gradient signal reaching `k_raw`, and this difference compounds over training. | Backward-hook gradient-norm/variance telemetry on `k_raw`, short instrumented runs | Yes — new short training | 2 (conditional) | ≈0.15–0.8 GPU-h |
| H4 (new) | Compensatory-constant hypothesis: a GLOBAL constant bias is a translation-like perturbation training can partly absorb via correlated drift in a low-dimensional locus (`k_conv1d`'s own weights, or upstream); a PER-TOKEN bias cannot be absorbed this way (50,257 different, uncorrelated perturbations), so training instead suppresses raw-key diversity to reduce interaction with the un-absorbable per-token noise. | Checkpoint parameter-diff norm/cosine, `k_conv1d` (and block-0 more broadly, per attack Q2) | **No — zero GPU, `torch.load`+diff only** | 0 | ≈0 (CPU/box time only) |
| H5 (new, refines H1) | Frequency-concentration confound: any apparent "token-identity" effect could be driven entirely by a handful of extremely high-frequency tokens (punctuation, "the", sub-word fragments) that recur predictably, rather than a genuine broad-vocabulary identity effect — i.e. H1's signal could really be a small-support, quasi-positional artifact. | Stratify `repeat_excess` by in-sample token frequency band (top-20 vs. long tail) | **No new capture — reuses H1's own captured tensors** | 1 (same pass as H1) | ≈0 marginal |

**Critiques folded into the table, stated plainly:**
- H2 is honestly not independent evidence — `gram_deviation_mean`,
  `effective_rank_mean`, and `stable_rank_mean` are three statistics of
  the SAME per-episode eigenspectrum (`rank_utils.py`'s
  `effective_rank`/`stable_rank`, `model_rd.py`'s `gram_deviation`, all
  computed inside the SAME `chunk_key_gram_stats` call). A "positive" H2
  result is close to a restatement of the primary bar in different units,
  not a new causal claim. Still worth running FIRST because it costs
  literally nothing (§12.3) and is a legitimate robustness triangulation
  (do independent rank estimators agree on direction, not just the raw
  Gram-deviation number) — but it must be reported as corroboration, not
  as discriminating evidence between H1/H3/H4.
- H1's originally-suggested formulation (η², a one-way ANOVA
  variance-explained-by-token-identity share) was REJECTED in favor of
  `repeat_excess` (§12.4): with a 50,257-token vocabulary sampled through
  a ≤512-token window, the overwhelming majority of distinct tokens are
  singletons in any given sample, and singleton-heavy groups make η²
  trivially close to 1 regardless of any real effect (each singleton's
  "between-group" contribution is just its own distance to the grand
  mean, contributing no real group structure) — an uninformative,
  near-tautological statistic at this vocabulary/window-size ratio.
  `repeat_excess` restricts attention to tokens that actually recur
  within the same chunk, directly targeting what H1 claims.

### 12.3 Stage 0 — zero-GPU analyses (run first, no forward pass, no training)

**12.3.1 H2 — effective-rank/stable-rank pinned-CI reharvest.**
`pooled_subset()` (`experiment-runs/2026-07-05_trackc_rung2/analyze_probe_wave2.py`,
already the span_frac pipeline's own dependency) already returns
`effective_rank_mean` and `stable_rank_mean`, n-weighted-pooled across
both layers, for every one of the **48** already-archived `retrofit_*.json`
files (46 counted in an earlier pass of this design plus 2 kraw-mode
λ-mini-sweep control files, `retrofit_kraw_lam030_openr1_s0.json` and
`retrofit_kraw_lam080_openr1_s0.json` — both verified present on disk
alongside the other 46, and equally in scope for this reharvest since
they are the SAME kraw-mode/pre-blend population this subsection's own
pre-blend comparison already touches) — the SAME files
`build_fit_inputs_and_run.py` already parses for `span_frac`. **Spec:**
clone `build_fit_inputs_and_run.py`'s
`extract_group`/`span_frac_from_retrofit_json` pattern into a new
`build_fit_inputs_rankstats.py`, swapping which field of `pooled_subset()`'s
return dict is read (`effective_rank_mean`, `stable_rank_mean` instead of
computing `span_frac` from `gram_deviation_mean`), and feed the resulting
per-seed values into the UNMODIFIED `derive_estimation()`
(`fit_frozenbias_estimation.py`) — same pinned `t(2,.975)=4.303` CI
formula, zero new statistical machinery. Produces 4 new deltas with CIs:
Arm2−Arm1′ (post-blend, both stats), Arm2′−Arm1″ (post-blend, both
stats), plus the pre-blend versions using the already-archived
`retrofit_kraw_arm2_*`/`retrofit_kraw_arm1_*` files. **Pre-registered
reading:** corroboration-consistent-with-collapse-story (never
"CONFIRM" — see the schema requirement below) requires
`effective_rank`/`stable_rank` deltas to move OPPOSITE to `span_frac`'s
own sign convention (higher rank = less collapsed = should FALL for
Arm2, RISE for Arm2′) — i.e. the SAME direction §12.0's corrected
reading predicts, just in rank units instead of Gram-deviation units.
**Cost: 0 GPU-h — pure Python over already-repo-committed JSON files,
runs on the Mac, no SSH needed.**

**Per-comparison `mechanism_direction` pin (MAJOR-3 fix — pinned NOW, by
comparison, never chosen per-call after a result is seen):**
`derive_estimation()`'s `mechanism_direction` argument must be set
explicitly for every call this subsection (and §12.3.2) makes, per the
comparison being estimated, not left at the function's own default:
- **`mechanism_direction="negative"`** for every Arm2-vs-Arm1′ (post-blend)
  and Arm2-vs-Arm1 (pre-blend) comparison — the per-token arm vs. its own
  control. A FALLING rank (`effective_rank`/`stable_rank` delta negative)
  is what corroborates H2/the collapse story for this comparison.
- **`mechanism_direction="positive"`** for every Arm2′-vs-Arm1″
  (post-blend) and Arm2′-vs-Arm1 (pre-blend, §12.3.2) comparison — the
  global-vector arm vs. its own control. A RISING rank (delta positive)
  is what corroborates here — the OPPOSITE sign from the per-token case,
  because §12.0's corrected reading has the two arms moving in opposite
  directions on the same underlying observable.

**Expected-sign sanity anchors (attack-round-executed values, to be
REPRODUCED at Stage 0, not assumed correct by construction):** the
independent attack round that reviewed this design already ran this
exact Arm2-vs-Arm1 pre-blend rank reharvest once, obtaining
**`effective_rank` Δ=−2.09 [−2.73,−1.46] (openr1-mix-ext)** and
**`stable_rank` Δ=−1.75 [−2.26,−1.24] (wikitext-mix-ext)** — both
negative, both CI-excluding-zero, both matching the `mechanism_direction
="negative"` pin above. A Stage 0 run of `build_fit_inputs_rankstats.py`
that reproduces these two figures (same files, same formula) is a strong
implementation-correctness check on the new script; a Stage 0 run that
disagrees with these attack-round-executed numbers should be treated as
a bug in the new script before it is treated as a substantive finding.

**Schema requirement, binding on every §12 JSON artifact — Stage 0, 1,
and 2 alike, not just this subsection (MAJOR-2 fix):**
1. Every JSON file any §12 script writes must carry a top-level
   `"tier": "exploratory-mechanism-wave — NOT a confirmatory bar"` field,
   set unconditionally, so no downstream reader (human or script) can
   mistake a §12 artifact for a §7–§9 confirmatory one from its shape
   alone.
2. `derive_estimation()`'s returned `confirm_direction_consistent` field
   must be renamed (or wrapped under an added key, leaving the original
   in place for traceability) to **`direction_consistent_with_hypothesis`**
   before anything is persisted to disk — the word "confirm" in a
   persisted field name is exactly the kind of shape-level leak that lets
   an exploratory reading get quoted later as if it carried confirmatory
   license, which §12.0 already disclaims everywhere in this section.
3. **`headline_verdict()` (`fit_frozenbias_estimation.py`) must NEVER be
   called anywhere in §12** — not in Stage 0, 1, or 2, not even "just to
   see." That function's entire purpose is to emit a
   CONFIRM/INCONCLUSIVE/ESTIMATION verdict under the blind-pin discipline
   §7.3/§9 built for the primary bars; §12.0 already establishes there is
   no blind pin here and every reading is post-hoc, so a `headline_verdict()`
   call would manufacture a confirmatory-shaped output this wave has no
   license to produce. Any script needing a combined multi-corpus read
   must write its own small, explicitly-exploratory summary function
   instead of importing `headline_verdict()`.

**12.3.2 The missing Arm2′-pre-blend comparison (a gap in the existing
archive, closed for free).** `retrofit_kraw_arm2p_{openr1,wikitext}_s{0,1,2}.json`
(6 files) already exist on disk (verified present, both locally and on
the SSD mirror) but were **never fed into `derive_estimation()`** — the
existing `PHASE_D_FULL_REPORT.json` only computed Arm2-vs-Arm1′ pre/post
and Arm2′-vs-Arm1″ post-blend; the pre-blend Arm2′-vs-Arm1 comparison
(the direct global-arm analogue of the co-primary) was captured but never
summarized. **Spec:** same script as 12.3.1, one more `extract_group("kraw", "arm2p_")`
call, one more `derive_estimation()` call against the already-extracted
`arm1_kraw` — `mechanism_direction="positive"` per 12.3.1's pin (this is
an Arm2′-vs-Arm1 comparison), and subject to the SAME schema requirement
(`"tier"` field, field rename, no `headline_verdict()`) as every other
§12 JSON artifact. **Cost: 0 GPU-h.**

**12.3.3 H4 — checkpoint parameter-diff (compensatory-constant test).**
For a matched seed/corpus triple (recommend seed 0, openr1-mix-ext — same
choice as the λ-mini-sweep, for consistency), `torch.load` (CPU,
`map_location="cpu"`) the step-1000 and step-20000 checkpoints for Arm1,
Arm2, Arm2′ (6 files, ~56MB each = ~336MB, trivial). Define
`ΔW(arm) := W(arm, step20000) − W(Arm1, step1000)` for a chosen parameter
subset (primary: `blocks[i].mixer.k_conv1d`'s own weight/bias; per attack
Q2, widen to the FULL block-0 parameter set as a secondary reading, since
compensation could sit upstream of `k_conv1d`). Report `‖ΔW(Arm2)‖_F` vs.
`‖ΔW(Arm2′)‖_F`, and `cos(ΔW(Arm2′), b_global)` where structurally
meaningful (e.g. if a bias/weight column can be reshaped to `(d_state,)`).
**Pre-registered reading:** H4-consistent if `‖ΔW(Arm2′)‖` shows a more
COHERENT (higher-cosine-to-`b_global`, i.e. lower-rank/more
one-directional) drift pattern than `‖ΔW(Arm2)‖`'s own drift relative to
Arm1's matched drift — NOT merely "which norm is bigger" (a norm
comparison alone is not diagnostic of compensation; directionality is).
**Cost: 0 GPU-h — CPU-only `torch.load` + tensor diff, run on the box
(needs the checkpoint files) or after `scp`-ing the ~336MB down; no CUDA
context ever created.**

**12.3.4 Stage 0.5 — synthetic self-tests for this wave's THREE new
statistics (MAJOR-5b fix; GATE: Stage 1 may not launch until every test
below passes).** Per the standing "run the negative unit test that's
supposed to prove the check has teeth to completion" `[LEARN]`, and
mirroring `fit_frozenbias_estimation.py`'s own `_self_test()` rigor
(hand-picked synthetic inputs, hard-coded expected outputs, hard
assertions, executed to completion, not merely written) — none of
`repeat_excess`, the parameter-diff cosine, or the gradient-flow norm has
had its OWN implementation independently verified before this pass; each
gets one below. All three are pure-Python/CPU, zero GPU, zero dependence
on the real checkpoints.

*Self-test 1 — `repeat_excess` (H1/H5).* Two pinned constructions, both
on the same 12 hand-picked (not RNG-drawn) 3-D vectors, L2-normalized:
`raw = [(1,2,-1), (0.5,-1.5,2), (-2,0.5,1), (1.5,1.5,1.5), (-1,-2,0.5),
(2,-0.5,-1.5), (0,1,-2), (-1.5,0.5,-1), (1,-1,1), (-0.5,2,0.5), (2,2,-0.5),
(-1,0,2)]`.
  - **(a) Uncorrelated-assignment construction:** token ids assigned
    round-robin, uncorrelated with the vector values —
    `['A','B','C','D']*3`. Computed `repeat_excess = -0.238404`
    (`same_tok_sim=-0.252186`, `diff_tok_sim=-0.013781`, n_same=24,
    n_diff=108). **Pass:** the implementation must reproduce
    `-0.238404 ± 1e-4` on this EXACT input — an implementation-
    correctness check (does the formula/pooling match this hand-verified
    reference), not a claim that this value is "the null."
  - **(b) Planted-clustering construction:** SAME 12 vectors, but the 6
    positions forming the 3 closest-cosine pairs (`(0,10)`, `(1,8)`,
    `(2,11)`, cosines 0.924/0.906/0.781) are all merged into ONE shared
    token id; every other position gets a unique singleton id (no other
    repeats). Computed `repeat_excess = +0.034437`
    (`same_tok_sim=-0.030517`, `diff_tok_sim=-0.064954`, n_same=30,
    n_diff=102). **Pass:** the implementation must reproduce
    `+0.034437 ± 1e-4` on this EXACT input, AND this value must exceed
    construction (a)'s value by **≥ +0.2** (planted delta = +0.272841) —
    i.e. the statistic detects a real, planted same-token-clustering
    effect of this registered strength, not just that it computes SOME
    number.

*Self-test 2 — parameter-diff norm/cosine (H4).* Mock 6-dimensional
flattened parameter subset (stand-in for a tiny `k_conv1d` weight/bias
slice), frozen `b_global = (0.3, -0.1, 0.2, 0.4, -0.2, 0.1)`.
  - **(a) Planted-coherent construction:** `ΔW = 2.5 · b_global` exactly
    (a pure scalar multiple). **Pass:** `‖ΔW‖_F` must equal
    `1.479019945774904 ± 1e-9` and `cos(ΔW, b_global)` must equal
    `1.0 ± 1e-6` — the implementation must report perfect coherence when
    the drift IS a scalar multiple of `b_global`, by construction.
  - **(b) Incoherent construction:** `ΔW = (0.4, 0.4, -0.3, -0.1, 0.35, 0.5)`
    (hand-picked, not a multiple of `b_global`). **Pass:** `‖ΔW‖_F` must
    equal `0.8902246907382427 ± 1e-9` and `cos(ΔW, b_global)` must equal
    `-0.0759497473859234 ± 1e-6`, AND `abs(cos) < 0.2` (correctly reports
    LOW coherence for an unrelated drift direction). Construction (a)'s
    cosine must exceed construction (b)'s by **≥ 0.5** — the comparative
    "more coherent than" reading §12.3.3's pre-registered reading actually
    depends on, verified on a case where the right answer is known.

*Self-test 3 — gradient-flow norm telemetry (H3, feeds §12.5's smoke).*
Closed-form linear construction, no autodiff needed to derive the
expected value: mock "k_raw" stand-in `x = (1.0, -2.0, 0.5)` (1×3),
downstream weight `W = [[0.2,-0.3],[0.4,0.1],[-0.5,0.6]]` (3×2), target
`t = (0.1, -0.2)`, `y = x @ W`, `L = 0.5·Σ(y-t)²`. Hand-derived:
`y = (-0.85, -0.2)`, `resid = (-0.95, 0.0)`, `L = 0.45125`,
`dL/dx = resid @ Wᵀ = (-0.19, -0.38, 0.475)`,
`‖dL/dx‖₂ = 0.6372793735874401`. **Pass:** (i) the hook-captured gradient
norm on THIS exact `x`/`W`/`t` triple must equal `0.637279 ± 1e-5`
(verifies the hook observes the correct tensor's gradient, not a stale or
mismatched one); (ii) the hook must fire exactly once per forward/backward
pair (call-count assertion); (iii) the captured value must be finite and
nonzero (true by construction here since `resid ≠ 0`). This self-test is
the exact-value counterpart to — not a replacement for — §12.5's own
loss-trajectory-equality smoke (hook-vs-no-hook `torch.equal` check),
which stays a Stage-2-specific gate on top of this one.

**Gate:** Stage 1 (§12.4) may not launch until all three self-tests above
have been RUN and PASSED — a specification that has not been executed is
not a passed gate, same status-disclosure convention §10 already uses for
§8.0/§8.0b. **Cost: 0 GPU-h, pure Python, no checkpoints/box needed —
runs entirely on the Mac.**

### 12.4 Stage 1 — cheap eval-only GPU passes on already-existing checkpoints (H1, H5) — gated on §12.3.4 (Stage 0.5) passing

**`repeat_excess`, the primary H1 statistic, defined precisely:** within
each `(b, chunk, head)` episode (identical chunking to
`chunk_key_gram_stats`, `chunk_size=64`, `content_mask` excludes EOT
exactly as today), among the `n_valid` content positions, group by token
id via the ALREADY-AVAILABLE `token_ids_cat` (joined by position — the
prior agent's finding, confirmed: `chunk_key_gram_stats`'s own records
carry no token id, but `capture_raw_keys` already returns `token_ids_cat`
aligned to the SAME `(B,T)` axes as the captured key tensor, so the join
is a reshape, not new plumbing). Require ≥1 pair `(i,j)`, `i≠j`,
`tok(i)=tok(j)` (a genuine within-chunk repeat) AND ≥1 pair with
`tok(i)≠tok(j)`, else the episode is excluded (counted, not silently
zeroed, per the standing "don't silently treat undefined as 0"
`[LEARN]`). Compute, on L2-normalized keys (same pre-normalization
`chunk_key_gram_stats` already applies):

```
same_tok_sim = mean over all (i,j), i≠j, tok(i)=tok(j), of cos(k_i, k_j)
diff_tok_sim = mean over all (i,j), tok(i)≠tok(j), of cos(k_i, k_j)
repeat_excess = same_tok_sim − diff_tok_sim
```

computed per episode, pooled (n-weighted mean) exactly the way
`summarize_gram_records` already pools `gram_deviation`. **Pre-registered
reading:** H1-consistent requires `Δrepeat_excess` (Arm2−Arm1, pre-blend
`k_raw` — the artifact-free population, same discipline as the
co-primary) to be POSITIVE (repeats become more self-similar) AND
materially larger than `Δrepeat_excess` (Arm2′−Arm1, pre-blend) — since
the global bias gives no token-varying signal for training to exploit
specifically among repeats, H1 predicts this second delta stays near the
pre-blend co-primary's own established noise floor. **H5 (frequency
confound) reuses the exact same captured `(k_raw, token_ids_cat)` pairs**:
stratify the pooled `repeat_excess` computation by whether `tok(i)` falls
in the top-20 most frequent tokens in THIS sample vs. the rest, report
both strata's own delta separately — if H1's signal is concentrated
entirely in the top-20 stratum, that is disclosed as a small-support/
quasi-positional reading, not a broad-vocabulary identity effect, per
H5's own registered purpose.

**Yield estimate and mechanical check (MINOR-4 fix).** One-line
back-of-envelope: at `chunk_size=64` over a ~50,257-token BPE vocabulary
with a realistically Zipfian frequency profile (a handful of tokens each
carry ≫1/64 of total mass), a within-chunk same-token collision is a
birthday-problem-style near-certainty for any *effective* vocabulary
support well under a few hundred categories — `repeat_excess` is expected
to be a HIGH-yield statistic (most 64-token chunks should contain ≥1
repeat), not a rare-event one; this is an estimate, not a measurement.
**Mechanical yield check, registered minimum n=50 (run automatically at
Stage-1 start, per the standing "don't silently treat undefined as 0"
discipline already invoked above for per-episode exclusion):** the probe
script must report the realized count of INCLUDED (non-excluded)
episodes pooled across the descriptive pass, before any `repeat_excess`
number is trusted; if that pooled count is **< 50**, STOP and report
`repeat_excess` as under-powered/insufficient-yield rather than as a
trusted reading — a registered floor decided now, not a post-hoc call
made after seeing how many episodes happened to qualify.

**Spec (new script, `frozen_bias_token_identity_probe.py`, cloning
`frozen_bias_retrofit_eval_rd.py`'s `load_checkpoint`/`capture_raw_keys`
verbatim — same pod-safety convention this codebase already uses
throughout):** identical sampling discipline to the existing retrofit
tool (`n_windows=32`, `batch_size=16`, `seq_len=512`,
`corpus_fixed_seed(corpus)+95_000` generator seed) so the captured
population is IDENTICAL to what the existing co-primary `kraw` retrofit
already measured — this makes `repeat_excess` and `gram_deviation`
directly joint-interpretable per episode (e.g. a per-chunk correlation
between the two, a free byproduct worth reporting).

**Cost, descriptive-first (n=1, seed 0, both corpora-trained-on, 3 arms
— Arm1/Arm2/Arm2′):** 6 checkpoint-passes × ≈0.0348 GPU-h/pass ≈ **0.21
GPU-h**. **Cost, full 3-seed confirmatory grid (matching this design's own
pinned-CI convention, if the descriptive pass shows a nonzero signal):**
18 checkpoint-passes ≈ **0.63 GPU-h**.

**12.4.1 Trajectory sub-study (optional but cheap, informs Stage 2's
decision rule) — reuses the SAME 400 already-present intermediate
checkpoints, no new capture logic beyond pointing the same script at 5
steps instead of 1.** Purpose: determine WHEN the Arm2-vs-Arm1 /
Arm2′-vs-Arm1 divergence emerges — this directly gates Stage 2's
step-count decision (§12.5) at zero marginal training cost, since these
checkpoints already exist regardless of whether Stage 2 ever launches.
Executed as an explicit numbered procedure (MAJOR-5a fix — promotes the
densification fallback from an attack-round question into a mandatory
step, per attack Q3):

1. Run the identical `repeat_excess` (and, for a free cross-check,
   `gram_deviation`/`span_frac` via the existing `kraw` mode) computation
   at steps {1000, 5000, 10000, 15000, 20000}, 1 seed (0), 1 corpus
   (openr1-mix-ext, matching the mini-sweep's own corpus choice), 3 arms:
   15 checkpoint-passes ≈ **0.52 GPU-h**.
2. Read the resulting 5-point signed-delta trajectory separately for
   each of the two comparisons (Arm2-vs-Arm1, Arm2′-vs-Arm1). Define
   **"ambiguous" MECHANICALLY, not by eyeball**: a trajectory is
   ambiguous if the sign of the delta flips between any two ADJACENT
   sampled checkpoints in the 5-point grid (e.g. step 5000's delta
   positive and step 10000's delta negative, or vice versa, for either
   comparison).
3. **If step 2 finds either comparison's trajectory ambiguous by this
   mechanical definition, densify for free** (all 400 checkpoints already
   exist at 1,000-step spacing — zero new training): compute the SAME
   `repeat_excess`/`gram_deviation`/`span_frac` statistics at the 3
   additional steps {2000, 3000, 4000}, and re-apply step 2's
   adjacent-sign-flip check to the resulting 8-point grid before
   concluding anything is genuinely ambiguous.
4. Feed the (possibly-densified) final trajectory into Stage 2's gate
   (§12.5) — the gate itself is now a single frozen numeric rule, not a
   judgment call made after looking at the shape of the curve.

### 12.5 Stage 2 — new short instrumented training cells (H3), conditional on 12.4.1

**Gate (pinned NOW, per attack MAJOR-4 — a single frozen numeric
criterion, fixed before any BUILD agent runs 12.4.1, replacing the
previously-deferred early/late/ambiguous trichotomy that left the actual
threshold for a BUILD agent to choose after seeing the data):**

Stage-2 full 20,000-step runs are authorized ONLY if the step-5,000
descriptive delta's sign matches the step-20,000 sign AND
|Δ@5000| ≥ 0.5·|Δ@20000|; otherwise truncated 3,000-step runs suffice.

This criterion is evaluated separately for each of 12.4.1's two
comparisons (Arm2-vs-Arm1, Arm2′-vs-Arm1), read off the (possibly
densified, per 12.4.1 step 3) `repeat_excess`/`span_frac` trajectory the
same script already captures, on openr1-mix-ext (12.4.1's only corpus).
If the two comparisons select different branches under this rule, the
full 20,000-step branch governs for both — Stage 2 is never downgraded
to the cheaper branch by one comparison's own weaker signal.

**Spec:** new script `frozen_bias_gradflow_probe.py`, NOT a modification
of `lm_pretrain_rd.py` in place (clone the training loop, per this
codebase's own duplication-over-cross-import convention) — instruments a
backward hook capturing the gradient norm reaching `k_raw` (the
post-conv, pre-blend key tensor `apply_frozen_bias_blend` consumes) at a
fixed cadence (every 100 steps), for Arm1 (off), Arm2 (per-token, λ=0.58),
Arm2′ (global, λ=0.58), 1 seed (0), 1 corpus (openr1-mix-ext) — 3 cells.
**Mandatory Stage-2-specific smoke, before any real cell launches (per
the standing "smoke test every model before training" hard rule, same
discipline as §8.0/§8.0b's Wave −1 smokes for rung-1):** verify on a tiny
CPU-runnable model that (a) the hook fires exactly once per forward pass
per layer, (b) captured gradient norms are finite and nonzero, (c) adding
the hook does not change the loss trajectory relative to an unhooked
control run on the same seed/data (a `torch.equal`-level check on the
first few steps' loss, mirroring §8.0b's own code-path-equality
discipline). **This smoke does not exist yet — a SPECIFICATION only,
same status disclosure convention §10 already uses for §8.0/§8.0b.**

**Cost:** at 3,000 steps, ≈137s/cell × 3 ≈ **0.11 GPU-h** (+ smoke, ≈0
GPU-h, CPU-only). At the full 20,000-step fallback, ≈912s/cell × 3 ≈
**0.76 GPU-h**. No separate calibration run required (§6.3's standing
rule) — these are short DERIVATIVES of an already-calibrated recipe
(identical architecture/lr/corpus/seed convention to rung-1's own
calibrated cell), differing only in step count and the added hook; the
Stage-2-specific smoke above is the load-bearing gate instead.

### 12.6 Budget summary

| Item | GPU-h (1×) | GPU-h (2× contingency) |
|---|---|---|
| Stage 0 (H2 reharvest, missing Arm2′-pre-blend comparison, H4 param-diff) | 0 | 0 |
| Stage 0.5 (synthetic self-tests: `repeat_excess` null+planted, parameter-diff cosine, gradient-flow norm — §12.3.4, gates Stage 1) | 0 | 0 |
| Stage 1 core, descriptive (n=1, 6 passes) | 0.21 | 0.42 |
| Stage 1 core, full confirmatory grid — **12 ADDITIONAL passes** beyond the descriptive pass's own seed-0 data (seeds 1,2 × 3 arms × 2 corpora only; NOT 18 independent passes — seed 0's 6 passes are the SAME 6 already counted in the row above, not re-run and not re-counted here), if the descriptive pass shows signal | 0.42 | 0.84 |
| Stage 1.1 trajectory sub-study (15 passes, includes the conditional 3-step densification from §12.4.1 step 3 — zero marginal cost, already-present checkpoints) | 0.52 | 1.04 |
| Stage 2, conditional, cheap branch (3 cells × 3,000 steps) | 0.11 | 0.22 |
| Stage 2, conditional, expensive branch (3 cells × 20,000 steps) | 0.76 | 1.52 |
| **Total, cheapest path (descriptive-only Stage 1, Stage-2 cheap branch)** | **0.84** | **1.68** |
| **Total, most expensive plausible path (full Stage 1 grid as 12 ADDITIONAL passes + Stage-2 expensive branch)** | **1.91** | **3.82** |

**Corrected arithmetic (MINOR-1 fix):** the previous revision of this
table listed the full confirmatory grid as 18 independent passes (0.63
GPU-h) ADDED on top of the descriptive pass's own 6 passes (0.21 GPU-h),
double-counting seed 0's 6 passes twice (descriptive row + first 6 of the
"18"). The full grid is 3 seeds × 3 arms × 2 corpora = 18 passes TOTAL,
of which 6 (seed 0) are already the descriptive pass; only the
**remaining 12** (seeds 1 and 2) are incremental cost: 12 × ≈0.0348
GPU-h/pass ≈ **0.42 GPU-h** (1×), **0.84 GPU-h** (2×) — not 0.63/1.26.
Most-expensive-path total recomputed: 0.21 (descriptive) + 0.42 (12
additional) + 0.52 (trajectory) + 0.76 (Stage 2 expensive) = **1.91
GPU-h** (1×), **3.82 GPU-h** (2×). The cheapest-path total (0.84/1.68) is
unaffected — it never included the full confirmatory grid.

Against the rung-1 wave's own realized spend (6.90/135 GPU-h) and the
committed rung-1-only budget's own remaining headroom (≈121–124 GPU-h,
§8.5.1), this entire mechanism wave — EVERY item, both branches, worst
case — costs under **3.82 GPU-h at 2× contingency**, i.e. under 3.2% of
the rung-1-only headroom and under 2.9% of the full 135 GPU-h program
ceiling. No case in this design approaches a budget concern; the
constraint on this wave is design/interpretive rigor, not compute.

### 12.7 Standing constraints carried forward (checked, not re-derived)

- Same corpora, same tokenization, same architecture as rung-1 (§5, §10)
  — no new axis introduced anywhere in §12.
- Blind-pin discipline (§7.3) does NOT apply here — §12.0 already
  discloses every hypothesis is post-hoc/exploratory; no new pin is
  claimed or needed, and no result here should be reported with a
  confirmatory CI-excludes-zero framing without that caveat attached.
- tmux+supervisor, try/except per config, `/data` checkpoint convention,
  disk-space gate, repo(≤25MB)+SSD-mirror archive policy — all apply
  unchanged to any Stage-1/Stage-2 script that runs on the box.
- Smoke-before-train (Stage 2 only, §12.5) — mandatory, not yet built.
- Re-verify the 12.0 pre-check (checkpoint presence, disk space, GPU
  contention) immediately before any BUILD agent executes anything —
  it was true at check time this session, not guaranteed true later.

### 12.8 ATTACK-ROUND-1 QUESTIONS — the 5 weakest points in this section (RESOLVED-OR-CARRIED-FORWARD by Rev 12.1 below — kept verbatim as historical record; see the Rev 12.1 REVISION LOG for the finding→fix map, and §12.9 for what's still open after this pass)

1. **Correlational ceiling.** `repeat_excess`, the rank-stat deltas
   (H2), and the parameter-diff (H4) are all OBSERVATIONAL/descriptive —
   none of them intervene on the candidate mechanism directly. Even a
   clean, CI-excluding-zero `repeat_excess` delta establishes correlation
   with the collapse phenomenon, not causation; a true causal test (e.g.
   forcibly perturbing raw-key repeat-similarity and checking whether
   `span_frac` follows) is out of scope this wave and should be named as
   a limitation in any writeup, not silently implied.
2. **H4's parameter-diff locus may be wrong.** The primary reading
   diffs `k_conv1d` only; if the compensatory adaptation this hypothesis
   proposes actually happens upstream (the token embedding table, an
   earlier FFN, or the residual stream feeding `k_conv1d`'s input), a
   `k_conv1d`-only diff would show nothing even if H4's causal story is
   correct elsewhere — a null H4 result at this locus does NOT refute
   the broader compensatory-constant idea, only this specific
   operationalization of it. The widened block-0 reading (§12.3.3) is a
   partial mitigation, not a complete one.
3. **Checkpoint sampling density risk.** §12.4.1's 5-point trajectory
   grid (1000/5000/10000/15000/20000) assumes a roughly monotonic
   divergence; the training loss curve itself (already in the archived
   `trajectory` arrays) drops most steeply in the first ~2,000–3,000
   steps, and a transient early divergence that later reverses could
   alias through a 5,000-step-spaced sample. All 400 checkpoints already
   exist at 1,000-step spacing — if §12.4.1's coarse grid looks
   ambiguous, densify for free (add steps 2000/3000/4000, still zero new
   training) before falling back to Stage 2's own new-training branch.
4. **The Stage-2 gate (§12.5) itself could be mis-specified.** It
   pre-registers "early/late/ambiguous" as a trichotomy but does not
   pre-register a numeric threshold for what counts as "already
   resolved by step ≤5,000" (e.g. CI-excludes-zero on a 1-seed
   descriptive read is not the same evidentiary bar as the confirmatory
   n=3 grid uses elsewhere in this document) — a BUILD agent implementing
   §12.5 should pin an explicit numeric rule (e.g. "the step-5,000
   descriptive delta's sign matches the step-20,000 descriptive delta's
   sign AND exceeds half its magnitude") before running 12.4.1, not
   choose one after seeing the trajectory data, to avoid exactly the
   post-hoc-pin failure mode the rung-1 wave's own descriptive-tier
   caveat already flagged once.
5. **None of §12's new statistics have their own validating synthetic
   self-test yet** — `sim_frozen_bias_training_mediated.py` earned real
   trust for `span_frac` specifically by proving (a) an exact-zero null
   on synthetic data with no training-mediated difference and (b)
   detection of a planted synthetic effect at a disclosed strength,
   BEFORE any real number was trusted. `repeat_excess`, the parameter-diff
   norm/cosine, and the gradient-flow telemetry (H3) have no analogous
   self-test. Per the standing "run the negative unit test that's
   supposed to prove the check has teeth to completion" `[LEARN]`, each
   new statistic should get a cheap synthetic self-test (e.g. verify
   `repeat_excess≈0` on i.i.d. random keys with randomly-assigned token
   ids, and that it detects a planted same-token-clustering effect at a
   known strength) as a mandatory Stage-0.5 gate — not skipped for
   expedience, and not treated as optional polish.

---

## Rev 12.1 REVISION LOG — 2026-07-07, §12 round-1 attack response (REVISE-THEN-PROCEED, every fix below is text-only — no code run, nothing committed)

| # | Severity | Finding (one-line) | Fix applied | Landed in |
|---|---|---|---|---|
| 1 | MAJOR-1 | NARRATIVE.md's own plain-language gloss inverted the same `span_frac` direction §12.0 already caught in THIS file's VERDICT section | Fixed separately, in a different document (`matrix-thinking/submissions/iclr-2027/NARRATIVE.md`) — out of scope for this pass, not touched here | `NARRATIVE.md` (not this file) |
| 2 | MAJOR-2 | §12's JSON artifacts had no schema-level marker distinguishing them from confirmatory-tier output; `derive_estimation()`'s own `confirm_direction_consistent` field name and `headline_verdict()` both carry confirmatory connotations §12.0 explicitly disclaims | Added a mandatory top-level `"tier"` field on every §12 JSON artifact; required renaming/wrapping `confirm_direction_consistent` to `direction_consistent_with_hypothesis` before persisting; explicitly forbade calling `headline_verdict()` anywhere in §12 | §12.3.1 (new "Schema requirement" paragraph), cross-referenced from §12.3.2 |
| 3 | MAJOR-3 | H2's rank-stat reharvest never pinned `mechanism_direction` per comparison explicitly, and had no independently-executed reference numbers to sanity-check the new script against | Pinned `mechanism_direction="negative"` for Arm2-vs-Arm1′/Arm1, `="positive"` for Arm2′-vs-Arm1″/Arm1; added the attack round's own executed sanity-anchor values (openr1 `effective_rank` Δ=−2.09 [−2.73,−1.46]; wikitext `stable_rank` Δ=−1.75 [−2.26,−1.24], Arm2 pre-blend), to be reproduced (not assumed) at Stage 0 | §12.3.1 (new "Per-comparison `mechanism_direction` pin" + "Expected-sign sanity anchors" paragraphs), §12.3.2 (cross-referenced) |
| 4 | MAJOR-4 | §12.5's Stage-2 gate deferred its own numeric threshold to a future BUILD agent ("e.g." language only), risking exactly the post-hoc-pin failure mode §12.0's own descriptive-tier caveat already flagged once | Pinned the frozen criterion IN THE DESIGN TEXT NOW: Stage-2 full 20,000-step runs are authorized ONLY if the step-5,000 descriptive delta's sign matches the step-20,000 sign AND \|Δ@5000\| ≥ 0.5·\|Δ@20000\|; otherwise truncated 3,000-step runs suffice. Applied per-comparison, with an explicit tie-break (disagreement → full branch governs) | §12.5 (Gate rewritten, no "e.g." language) |
| 5 | MAJOR-5 | Two attack-round questions (checkpoint-density risk, no self-test) were raised as OPEN QUESTIONS rather than built into the numbered spec | (a) §12.4.1 rewritten as 4 numbered steps; step 2 defines "ambiguous" mechanically (adjacent-checkpoint sign flip); step 3 promotes the densification fallback (steps 2000/3000/4000, zero GPU) from a suggestion into a mandatory conditional step. (b) New §12.3.4 "Stage 0.5" — 3 pinned synthetic self-tests (`repeat_excess` null+planted, parameter-diff cosine, gradient-flow norm), each with exact hand-computed expected values and hard pass/fail tolerances, mirroring `fit_frozenbias_estimation.py`'s own `_self_test()` rigor; added as a zero-GPU row in §12.6; Stage 1 is now explicitly gated on Stage 0.5 passing | §12.4.1 (rewritten, numbered), §12.3.4 (new section), §12.4 header (gate cross-reference), §12.6 (new budget row) |
| 6 | MINOR-1 | §12.6's "most expensive path" total double-counted seed 0's 6 descriptive-pass checkpoint-passes inside the "18-pass full grid" row | Recomputed: full grid is 12 ADDITIONAL passes (seeds 1,2 only) at ≈0.42/0.84 GPU-h, not 18 passes at 0.63/1.26; most-expensive-path total corrected 2.12→**1.91** (1×), 4.24→**3.82** (2×); cheapest-path total (0.84/1.68) unaffected | §12.6 (table + new "Corrected arithmetic" note, narrative percentages updated) |
| 7 | MINOR-2 | §12.3.1 undercounted the already-archived `retrofit_*.json` population as 46 files; 48 actually exist (verified by directory listing this pass) | Corrected 46→48, identifying the 2 previously-uncounted files by name (`retrofit_kraw_lam030_openr1_s0.json`, `retrofit_kraw_lam080_openr1_s0.json`) | §12.3.1 |
| 8 | MINOR-3 | §12.0 described the VERDICT section's own direction-gloss fix as "no committed edit was made there," which reads as unresolved even though the VERDICT section already carries the corrected wording in this working tree | Reworded to "correction applied in working tree, to be committed with this design" | §12.0 |
| 9 | MINOR-4 | `repeat_excess`'s expected YIELD (fraction of chunks actually containing a within-chunk token repeat) was never estimated or checked, so a low-yield/near-empty result could silently masquerade as "no signal" | Added a one-line Zipfian/birthday-problem back-of-envelope yield estimate (HIGH-yield expected) plus a mechanical yield check at Stage-1 start with a registered minimum n=50 included episodes, pooled | §12.4 (new "Yield estimate and mechanical check" paragraph) |
| — | — | Rev 12.1 revision log + refreshed round-2 questions needed | This entry; §12.8 retitled to mark ROUND-1/historical; added §12.9 (5 short items) | §12.8 (header), this table, §12.9 |

### What could NOT be fixed this pass, and why

- **Attack-round-1 item 1 (correlational ceiling)** — unaddressed by
  design; this is a genuine scope limitation of every Stage-0/Stage-1
  instrument (`repeat_excess`, the rank-stat deltas, the parameter-diff),
  not a bug fixable by rewording. Carried forward to §12.9.
- **Attack-round-1 item 2 (H4's parameter-diff locus may be wrong)** —
  the widened block-0 reading (§12.3.3) remains a partial, not complete,
  mitigation; no design-only fix closes this without new instrumentation
  scoped beyond this pass. Carried forward to §12.9.

### 12.9 ROUND-2 ATTACK QUESTIONS — refreshed, short (post-Rev-12.1)

1. **Correlational ceiling still stands** (carried from attack-round-1
   item 1, unresolved by design) — every Stage-0/Stage-1 statistic here
   remains observational; a clean result establishes correlation with the
   collapse phenomenon, not causation.
2. **H4's parameter-diff locus risk still stands** (carried from
   attack-round-1 item 2) — a null result at `k_conv1d` does not refute
   the broader compensatory-constant idea if the real adaptation sits
   upstream; the widened block-0 reading is a partial mitigation only.
3. **The Stage-2 gate's own `0.5` multiplier is itself an unexamined
   constant.** Rev 12.1 froze the threshold at "\|Δ@5000\| ≥
   0.5·\|Δ@20000\|," which closes the post-hoc-pin risk, but the
   specific value 0.5 (vs. 0.4 or 0.6) was chosen for round-number
   simplicity, not derived from any noise-floor or power argument — a
   fresh attack could ask whether this threshold is itself well-calibrated
   to the trajectory's actual step-to-step noise.
4. **Stage 0.5's self-tests validate arithmetic correctness, not the real
   pipeline's plumbing.** All three synthetic self-tests (§12.3.4) use
   hand-picked mock tensors/vectors, not the actual `capture_raw_keys`/
   `torch.load`/hook-registration code paths the real Stage-1/Stage-2
   scripts will run — passing them proves the FORMULAS are implemented
   correctly, not that the real scripts wire the correct real tensors
   into those formulas (the same gap §8.0b's own code-path-equality smoke
   exists to close for the rung-1 primary bar; §12 has no analogous
   real-code-path check yet, only the formula-level one).
5. **MINOR-4's yield estimate is a back-of-envelope, not a measurement**
   — if the real per-episode exclusion rate turns out far lower than the
   "birthday-problem near-certainty" estimate predicts (e.g. if
   `content_mask`/EOT handling excludes more positions than assumed, or
   `chunk_size=64` chunks are dominated by a single repeated token that
   never pairs with a different one), the registered n=50 floor is the
   only thing standing between a silently-underpowered `repeat_excess`
   and a trusted number — worth a fresh attack checking whether n=50
   itself is adequately conservative given the real pooled sample sizes
   Stage 1's 6-pass descriptive run will actually produce.

### 12.10 STAGE-0 + H4-REAL RESULTS — executed 2026-07-07 (exploratory tier throughout)

**Provenance.** The four Stage-0 scripts (`mech_schema.py`,
`mech_stage05_selftests.py`, `build_fit_inputs_rankstats.py`,
`mech_h4_paramdiff.py`) were built, independently audited
(STAGE0-CLEARED, zero findings), executed, and committed at `d20cbe8`;
raw outputs live in `matrix-thinking/deltanet_rd/results/mech_wave/`.
Every output JSON is stamped `"exploratory-mechanism-wave -- NOT a
confirmatory bar"` by `mech_schema.wrap_exploratory()`. The
correlational ceiling (§12.9 item 1) applies to EVERYTHING below —
these results order hypotheses for Stage 1/2; none is a headline claim.

**12.10.1 Stage 0.5 self-test gate (§12.3.4): PASSED.**
`mech_stage05_selftests_results.json` records `gate_passed: true` — all
three statistics (repeat_excess, parameter-diff norm/cosine,
gradient-flow norm) reproduced their hand-pinned synthetic values within
the registered tolerances, including the planted-effect detections. The
Stage-1 launch gate is OPEN.

**12.10.2 H2 rank reharvest (§12.3.1 + §12.3.2): H2 CORROBORATED
(exploratory).** `build_fit_inputs_rankstats_results.json`; n=48
archived retrofit JSONs verified; the sanity anchor reproduced the
attack-round reference numbers within 0.02. Four-comparison grid,
pinned `t(2,.975)=4.303` CIs, Δ = (arm − baseline), 2 metrics
(effective_rank_mean, stable_rank_mean) × 2 corpora:

- **Comparison 1 — Arm2 vs Arm1′ (post-blend): rank FALLS, 4/4 cells
  CI-exclude-zero.** eff_rank openr1 −5.618 [−8.144, −3.092], wikitext
  −9.142 [−12.836, −5.448]; stable_rank openr1 −3.109 [−5.913, −0.305],
  wikitext −3.961 [−4.516, −3.406].
- **Comparison 2 — Arm2 vs Arm1 (pre-blend): rank FALLS, 3/4
  exclude zero.** eff_rank openr1 −2.094 [−2.731, −1.457]; stable_rank
  openr1 −0.988 [−1.774, −0.201], wikitext −1.750 [−2.260, −1.241].
  wikitext eff_rank −2.375 [−8.016, +3.266] straddles zero — the cell
  §12.3.1 pre-flagged as expected-ambiguous, reported honestly.
- **Comparison 3 — Arm2′ vs Arm1″ (post-blend): rank RISES, 3/4
  exclude zero.** eff_rank wikitext +3.328 [2.238, 4.418]; stable_rank
  openr1 +1.885 [0.524, 3.246], wikitext +1.158 [0.848, 1.467]. openr1
  eff_rank +6.135 [−2.780, +15.050] ambiguous.
- **Comparison 4 — Arm2′ vs Arm1 (pre-blend; the §12.3.2 gap, run for
  the first time here): ALL FOUR cells ambiguous** (every CI straddles
  zero) — an honest non-result; the pre-blend Arm2′ contrast does not
  distinguish at n=3.

Reading: the destabilizing per-token arm's key-space rank falls against
BOTH baselines; the stabilizing global arm's rank rises post-blend —
the direction H2 (rank-collapse) predicts. Pre-blend Arm2′ is
uninformative.

**12.10.3 H4 checkpoint parameter-diff — REAL RUN (§12.3.3):
H4-CONSISTENT at the block-0 `k_conv1d` locus (exploratory).**
Executed on-box 2026-07-07 04:32 UTC, CPU-only, via `mech_h4_chain.sh`
in tmux session `mech_h4` (the on-box copy re-passed `--self-test`
before the real run; chain exits on any failure). Registered cell:
corpus=openr1-mix-ext, seed 0, `ΔW := W(step20000) − W(Arm1,
step1000)`. Raw: `results/mech_wave/mech_h4_paramdiff_results.json`.

- **Norms (reported only — NOT the diagnostic, per the §12.3.3
  pre-registration):** ‖ΔW(k_conv1d)‖_F = 3.7911 (Arm1 natural drift) /
  5.1157 (Arm2) / 4.0941 (Arm2′).
- **Directionality (the pre-registered reading).** cos(ΔW column,
  b_global), d_state=64 columns, chance E|cos| ≈ 0.0997:
  block-0 **Arm2′ mean|cos| = 0.1778, max|cos| = 0.3740, all four
  columns NEGATIVE** (drift anti-aligned with b_global); **Arm2
  0.0816 / 0.1387 (≈ chance)**; Arm1 natural drift 0.1219 / 0.1963.
  Block-1: Arm2′ 0.0310 / 0.0454 — no directional component; the
  effect is block-0-localized.
- **Verdict per the pre-registered reading:** Arm2′'s drift is more
  b_global-coherent than Arm2's (mean |cos| 0.1778 vs 0.0816) →
  H4-CONSISTENT at this locus. The observed SIGN is anti-alignment —
  the stabilized arm's key-conv weights drift AWAY from the frozen
  global-bias direction, a compensatory signature — but sign was not
  pre-registered; noted as observed, not predicted.
- **Secondary block0_full reading:** best-aligned single params sit at
  max|cos| 0.39–0.42 in ALL arms (Arm2 k_proj 0.4245 the highest) — no
  clean cross-arm separation at full-block granularity; the primary
  reading stands on `k_conv1d` only, consistent with §12.9 item 2's
  locus caveat cutting both ways.
- **Shared-baseline sanity check:** step-1000 cross-arm k_conv1d
  Frobenius distances 0.8407 (Arm2 vs Arm1) / 0.8043 (Arm2′ vs Arm1) —
  ~21% of the drift norms, small enough to support the shared
  step-early baseline.
- **Caveats:** single seed, single corpus; 8 columns tested → the
  0.374 max-stat carries a multiple-comparisons discount (naive
  Gaussian approx: per-column two-sided p ≈ 0.003 at |cos| = 0.374,
  ×8 comparisons ≈ 0.02 — suggestive, not confirmatory); and the
  block-0 mean-level signal (0.1778 vs chance 0.0997) at n=4 columns is
  the steadier component of the reading.

**Where this leaves the hypothesis table (§12.2):** H2 corroborated
(rank collapse tracks the destabilization direction in both arms), H4
consistent at one locus (compensatory directional drift in the
STABILIZED arm) — mutually compatible: Arm2's keys collapse toward a
lower-rank configuration, Arm2′'s keys make a coherent low-rank
correction against b_global and keep their rank. H1/H5
(token-clustering / frequency) remain untested pending Stage 1's
forward-pass probe (§12.4); H3 (gradient-flow) remains Stage-2-gated.

### 12.11 STAGE-1 RESULTS — H1 REFUTED, H5 clean, Stage-2 gate resolves to the FULL 20,000-step branch (executed 2026-07-07, exploratory tier)

**Provenance.** `frozen_bias_token_identity_probe.py` +
`mech_stage1_chain.sh` (built this session, self-test through the
probe's OWN code path, independent adversarial audit
CLEARED-WITH-MINORS — zero FATAL/MAJOR, committed `2432e23`). Run
on-box in tmux `mech_stage1`, GPU 2, 2026-07-07 ~05:02–05:04 UTC (chain
run 2; run 1 died at the Stage-0.5 gate re-run because
`mech_stage05_selftests.py` had not been shipped — a deploy-closure
miss, disclosed; both gates PASSED on run 2 before any GPU pass). Raw:
`results/mech_wave/mech_stage1_descriptive_{openr1,wikitext}-mix-ext_s0.json`
+ `mech_stage1_trajectory_openr1-mix-ext_s0.json`; archive
`experiment-runs/2026-07-07_mech_stage1/` (+SSD). Realized cost ≈0.03
GPU-h (log-timestamp bound) vs. the 0.73 estimate — eval passes at
dm256/ds64/L2 are seconds each, the banked per-pass estimate was
conservative ~20×. Yield floor: 512 included / 0 excluded episodes per
arm per corpus in the descriptive pass (1,024 across corpora ≥ the
registered 50); trajectory invocation pooled 7,680 included. Every
number below re-verified from the pulled raw JSONs.

**12.11.1 H1 (token-identity clustering): REFUTED at the pre-registered
reading (1 seed, exploratory).** §12.4 required
`Δrepeat_excess(Arm2−Arm1)` POSITIVE (repeats become MORE self-similar
under the destabilizing per-token bias) AND materially above
`Δ(Arm2′−Arm1)`. Measured (pooled across layers, step 20000, kraw):

| corpus | Arm1 (off) | Arm2 (per-tok) | Arm2′ (global) | Δ(Arm2−Arm1) | Δ(Arm2′−Arm1) |
|---|---|---|---|---|---|
| openr1-mix-ext | 0.38380 | 0.29354 | 0.26353 | **−0.09026** | −0.12028 |
| wikitext-mix-ext | 0.40906 | 0.31900 | 0.44003 | **−0.09005** | +0.03098 |

The required sign FAILS in both corpora — and the Arm2 delta is
strikingly corpus-CONSISTENT (−0.0903 / −0.0901). The per-token bias
makes same-token keys LESS self-similar relative to control. H1's
clustering account of the destabilization is dead at this tier; read
jointly with §12.10.2, the H2 rank collapse is NOT organized around
token identity. (Arm2′ is corpus-INCONSISTENT here — −0.120 on openr1,
+0.031 on wikitext — reported, not interpreted, at n=1 seed.)

**12.11.2 H5 (frequency confound): clean.** The top-20-token stratum
does NOT carry the signal — the rest-stratum repeat_excess is ≥ the
top-20 stratum in every arm on openr1 (e.g. off 0.42191 vs 0.36858;
per_token 0.31710 vs 0.27671) and comparable on wikitext. No
small-support/quasi-positional disclosure is required; the H1 refutation
is broad-vocabulary.

**12.11.3 Trajectory (§12.4.1): mechanically unambiguous, monotone from
step 1000.** repeat_excess deltas (openr1): Arm2−Arm1
−0.0600/−0.0721/−0.0763/−0.0863/−0.0903 at steps
1000/5000/10000/15000/20000; Arm2′−Arm1
−0.0264/−0.0570/−0.0869/−0.1106/−0.1203. Zero adjacent sign flips in
either comparison (`ambiguous: false` both) — no densification needed.
The divergence exists already at step 1000 and deepens smoothly.

**12.11.4 Stage-2 gate (§12.5's frozen rule): FULL 20,000-step branch.**
Arm2: sign(Δ@5000)=sign(Δ@20000) ✓ and |−0.0721| ≥ 0.5·|−0.0903|=0.0451
✓ → full. Arm2′: sign ✓ but |−0.0570| < 0.5·|−0.1203|=0.0601 →
truncated. Per the rule's own clause ("if the two comparisons select
different branches, the full 20,000-step branch governs for both"):
**Stage 2 is authorized at full 20,000 steps** (≈0.76 GPU-h + the
mandatory CPU smoke, §12.5). span_frac cross-check concurs at layer 0
(Arm2 +0.2457@5000 vs half-of +0.3008@20000; monotone, no flips).

**12.11.5 Layer-resolved span_frac observations (free §12.4.1
cross-check — reported, not pre-registered):** (a) layer-0 span_frac
reproduces the rung-1 story in direction: Arm2 +0.30084 vs Arm1 at
step 20000 (collapse), Arm2′ +0.10590. (b) layer-1 is genuinely
dynamic: Arm2's delta sign-flips (+0.117@1000 → −0.051@20000 — an early
transient collapse that REVERSES), and Arm2′'s layer-1 delta RISES
steadily (+0.012 → +0.225). The rung-1 primary measured its stabilization
claim on a different captured population/pooling; this layer-1 pattern
is a real, unexplained wrinkle at n=1 seed — flagged for Stage 2's
instrumented cells, not resolved here.

**Where this leaves §12.2 after Stage 1:** H1 REFUTED, H5 clean (no
confound), H2 corroborated (Stage 0), H4 consistent at one locus
(§12.10.3), H3 now the live open mechanism — Stage 2 (gradient-flow
instrumented cells, full-length per the gate) is the registered next
step.

### 12.12 MECH-WAVE CONCLUSION — STAGE-2 (H3 gradient-flow) RESULTS + FULL WAVE SYNTHESIS — executed 2026-07-07, exploratory tier throughout, WAVE CONCLUDED

**§12.2's H3 row, quoted verbatim (the registered hypothesis; NO sign was
pre-registered, unlike H1/H2):** "H3 | Optimizer/gradient-flow
interaction: the λ-blend's `normalize()` Jacobian, evaluated at a
per-token-random vs. a fixed-global bias direction, differentially
reshapes the gradient signal reaching `k_raw`, and this difference
compounds over training. | Backward-hook gradient-norm/variance
telemetry on `k_raw`, short instrumented runs | Yes — new short training
| 2 (conditional) | ≈0.15–0.8 GPU-h."

**Provenance.** `frozen_bias_gradflow_probe.py` + `mech_stage2_chain.sh`
(§12.5 spec, FULL 20,000-step branch per §12.11.4's frozen gate) were
built, smoke-verified (three-part CPU smoke: hook fires exactly once per
forward/layer on cadence-eligible steps only, captured norms finite and
nonzero, hooked vs. unhooked loss trajectories `torch.equal`-identical),
and independently audited **CLEARED-WITH-MINORS, zero FATAL/MAJOR
blockers**, committed `ffce8bb`. The auditor re-executed the smoke plus
three mutation tests (cadence break, grad-mutating hook — loss
divergence traced to exactly step 3 on a constant flip) and re-derived
the 11-file import closure. **MINOR-1, fixed at that commit:** the smoke
step's device was pinned to `MECH_STAGE2_GPU` — unpinned, it would have
landed on GPU 0 (rung-3's), a collision the pin also turned into a real
kernel exercise rather than a no-op. Run on-box as one combined-JSON
process (all 3 arms trained sequentially in a single invocation), GPU 2,
tmux session exited cleanly by harvest time (`tmux list-sessions` shows
only `trackc3` remaining; GPU 2 independently confirmed idle, 0 MiB / 0%
util at harvest). Raw:
`results/mech_wave/mech_stage2_gradflow_openr1-mix-ext_s0.json`; archive
`experiment-runs/2026-07-07_mech_stage2/` (result JSON + both scripts +
chain log + smoke log + real-run log, ~672KB, all files ≤25MB) + SSD
mirror, `diff -rq` confirmed byte-identical. The two on-box scripts are
ALSO byte-verified zero-drift against this repo's committed copies (md5
`144c8b98c53c55433fe74317eb93c104` /
`381a12da42da21409668aafecf2cb1e1`, both sides match exactly).

**Measured rate vs. estimate — a DISCLOSED CORRECTION to this harvest's
own inherited framing.** Re-derived directly from on-box file
timestamps, not asserted: chain deploy (`mech_stage2_chain.sh` written
to the box) 2026-07-07 06:14:27.301 UTC → `MECH_STAGE2_DONE` sentinel
06:59:02.321 UTC = **2675.0s ≈ 44.58 min ≈ 0.7431 GPU-h** (1 GPU),
corroborated independently by the sum of the 3 cells' own instrumented
`wall_s` fields (879.2546 + 890.9115 + 888.7030 = 2658.8691s = 0.73858
GPU-h — within 16s / <1% of the outer wall-clock bound, the gap being
smoke-test + JSON-write overhead). **This is UNDER §12.5's 0.76 GPU-h
estimate** (≈2–3% under, not over) — the realized per-cell rate
(≈886s/cell) is well-calibrated against the ≈912s/cell estimate, unlike
Stage 1's ~20×-conservative eval-pass estimate. **Flagged prominently
per this task's own re-derive-from-raw-artifacts instruction:** an
assumption carried into this harvest task (a claimed ≈70 min / ≈1.2
GPU-h realized wall, framed as an overrun against the 0.76 estimate) is
**not supported by the on-box timestamps** — the real run finished in
well under 45 minutes and under budget, not over it. Every downstream
ledger number in this section uses the re-derived 0.7431 GPU-h, not the
inherited figure.

**Verification (re-derived from the raw JSON directly, not the box's own
printed summary).** All 3 arms present (`off`, `per_token`, `global`);
`steps_completed=20000`, `n_skipped=0` for all three (no silently
skipped/truncated cell). Cadence-100 grad-norm series: 200 events × 2
layers × 3 arms = **1200 total events**, every one finite and nonzero
(checked programmatically over the full grid, zero exceptions). Cadence
itself verified exact — each layer's recorded steps are exactly
`[100, 200, ..., 20000]`, no gaps or duplicates. Metadata: `seed=0`,
`corpus=openr1-mix-ext`, `lambda=0.58` for `per_token`/`global` (`off`
correctly carries `lam=None`), architecture `d_model=256 / d_state=64 /
n_layers=2` matches rung-1's own registered architecture. `tier` field
present, reads exactly `"exploratory-mechanism-wave -- NOT a
confirmatory bar"`. Loss trajectories (201 points each, steps 1..20000)
all finite; final losses 3.8434 / 3.8322 / 3.8675 (off / per_token /
global) — closely matched, no divergence in any arm.

**H3 per-arm per-layer grad-norm results.** Independently re-derived
from the raw per-step series (own script, not the JSON's pre-computed
summary) — matches the JSON's own `h3_gradnorm_comparison` block to 6
decimal places, a plumbing cross-check on the archived summary, not a
re-quote of it:

| Layer | Arm | mean (n=200) | early mean (≤5000, n=50) | late mean (>15000, n=50) | Δ vs off (all) | Δ vs off (late) | linear trend (Δnorm / 1000 steps) |
|---|---|---|---|---|---|---|---|
| 0 | off | 0.04017 | 0.02824 | 0.04870 | — | — | +0.00136 |
| 0 | per_token | 0.00595 | 0.00768 | 0.00497 | **−0.03421** | −0.04374 | **−0.000176** |
| 0 | global | 0.02190 | 0.01786 | 0.02628 | −0.01826 | −0.02242 | +0.000558 |
| 1 | off | 0.02207 | 0.01143 | 0.03031 | — | — | +0.00126 |
| 1 | per_token | 0.00486 | 0.00404 | 0.00529 | **−0.01721** | −0.02502 | +0.000086 |
| 1 | global | 0.02240 | 0.01347 | 0.02937 | **+0.00034** | −0.00094 | +0.001065 |

Suppression relative to `off` (1 − arm/off), early→late window: layer 0
per_token 72.8%→89.8% (deepens), layer 0 global 36.7%→46.0% (deepens
mildly); layer 1 per_token 64.6%→82.5% (deepens), layer 1 global
**−17.9%→+3.1%** (starts ABOVE `off`, ends within noise of parity — no
persistent suppression at any point in training). Gradient-norm variance
(std) falls over training in every arm/layer (expected under any
converging optimization) and is not itself a differentiator between
arms; the mean-suppression pattern above is the load-bearing signal.

**H3 verdict: H3-CONSISTENT (exploratory tier — no sign was
pre-registered per §12.2's own row; this reads the qualitative SHAPE of
the hypothesis, not confirmation of a predicted direction).** Three
findings jointly support "the λ-blend's `normalize()` Jacobian
differentially reshapes the gradient reaching `k_raw`, compounding over
training," read against the standing correlational ceiling (§12.9 item
1):
1. **Differential magnitude, both layers.** The per-token arm suppresses
   `k_raw`'s gradient norm far more than the global arm at BOTH layers
   (layer 0: 89.8% vs 46.0% suppression by late training; layer 1: 82.5%
   vs ≈0%). This is the gradient-flow analogue of H2's rank-collapse
   finding (per-token collapses, global stabilizes) — here manifesting
   as differential gradient attenuation rather than differential
   Gram-spectrum rank.
2. **Layer-resolved, and the resolution is qualitative, not just a
   magnitude difference.** At layer 0, BOTH blended arms show real
   suppression relative to `off` (unequal, but both present). At layer
   1, the global arm's gradient norm is statistically at parity with —
   briefly above, then within ≈3% of — the unblended baseline
   *throughout training*, i.e. no suppression signature at all, while
   the per-token arm remains substantially suppressed (82.5% by late
   training). The per-token/global divergence is thus MORE starkly
   qualitative at layer 1 (one arm fully recovers to baseline parity,
   the other never does) than at layer 0 (both arms suppressed, to
   different degrees). This is a second, independent mechanistic probe
   this wave finding layer 1 qualitatively distinct from layer 0 —
   distinct in its own particulars from Stage 1's layer-1 span_frac
   sign-flip wrinkle (§12.11.5), but the same broader theme (layer 1
   behaves unlike layer 0 across multiple instruments this wave).
3. **Compounds over training, with the one exception itself
   informative.** Per-token suppression deepens at both layers (layer 0:
   72.8%→89.8%; layer 1: 64.6%→82.5%), and layer 0's linear trend is
   FLAT-TO-NEGATIVE (slope −0.000176/1000 steps — the ONLY negative
   trend in the whole 6-cell grid — against `off`'s own +0.00136
   growth). Global's layer-0 suppression also deepens, more mildly
   (36.7%→46.0%). The one cell where nothing compounds is exactly the
   one cell with nothing to compound: global/layer-1 tracks `off`'s own
   growing trend almost in parallel (slope +0.001065 vs. `off`'s
   +0.00126).

No claim is made beyond this description. H3 remains correlational/
descriptive per §12.9 item 1: it establishes that gradient flow to
`k_raw` IS differentially reshaped, in a layer- and training-stage-
resolved way, tracking the destabilizing (per-token) vs. stabilizing
(global) split established elsewhere in this wave — NOT that this
reshaping is the CAUSE of the collapse/stabilization rather than a
shared downstream consequence of the same underlying training dynamic.

**12.12.1 FULL MECH-WAVE SYNTHESIS — H1 through H5, the mechanism
picture this wave leaves behind.** Read jointly (each hypothesis still
stands on its own single-seed/single-corpus-per-item evidentiary
footing — this is a synthesis, not a re-run at higher n):

- **H1 REFUTED** (§12.11.1): the per-token bias makes same-token keys
  LESS mutually self-similar, not more (Δrepeat_excess = −0.09026
  openr1 / −0.09005 wikitext, corpus-consistent to 3 decimals) — the
  opposite of H1's predicted sign. The rank-collapse this wave otherwise
  documents is NOT organized around token identity.
- **H5 clean** (§12.11.2): no frequency-concentration confound — the
  rest-of-vocabulary stratum carries at least as much repeat-similarity
  signal as the top-20-token stratum in every arm tested. H1's
  refutation is a broad-vocabulary finding, not a small-support artifact.
- **H2 corroborated** (§12.10.2): effective_rank/stable_rank — an
  independent estimator of the SAME Gram-spectrum `span_frac` measures —
  agree in direction: per-token arm's post-blend key-rank FALLS vs. both
  baselines (4/4 CI-excluding-zero cells), global arm's RISES (3/4).
- **H4 consistent at one locus** (§12.10.3): the global (stabilizing)
  arm's block-0 `k_conv1d` weight drift is `b_global`-coherent AND
  anti-aligned (mean|cos|=0.1778 vs. chance 0.0997, all 4 tested columns
  negative) — a low-rank, directionally-organized correction against the
  frozen bias — while the per-token arm's drift sits at chance
  (0.0816). Block-1 shows no such directional component in either arm.
- **H3, this section:** the per-token arm's gradient signal reaching
  `k_raw` is substantially and increasingly suppressed relative to an
  unblended control, at BOTH instrumented layers, while the global arm's
  suppression is present but shallower at layer 0 and effectively ABSENT
  at layer 1 (statistical parity with the unblended control throughout
  training).

**The composite picture (still exploratory/correlational throughout,
never elevated past this tier):** the per-token bias induces a genuine,
BROAD (not token-identity-organized — H1/H5) collapse of the model's
raw key-space rank (H2), and this collapse is accompanied by a
substantial, deepening suppression of the gradient signal that reaches
`k_raw` during training, undiminished across both instrumented layers
(H3) — i.e. the destabilizing arm's own optimization pathway to `k_raw`
is progressively starved of gradient signal at exactly the locus whose
geometry collapses. The global bias, by contrast, is met with a
coherent, low-rank, anti-aligned compensatory adjustment concentrated in
block-0's `k_conv1d` weights (H4) that plausibly explains why ITS
gradient path is only shallowly and locally (layer-0-only) suppressed —
consistent with the compensation being sufficient to largely neutralize
the global bias's distortion of the gradient path beyond block 0, where
the per-token bias (50,257 uncorrelated per-token perturbations, no
single low-rank direction to compensate against) leaves suppression
un-neutralized at both layers, and the rank never recovers. This is a
coherent, mutually-consistent story across four independent instruments
(H1/H5 ruling out the token-identity account, H2 the direct
Gram-spectrum read, H4 the parameter-space compensatory signature, H3
the gradient-flow read) — but it is a STORY assembled from four
observational instruments applied to the SAME two training runs, not
four independent causal tests, and no instrument here manipulates the
candidate mechanism directly.

**What remains open.** The correlational ceiling (§12.9 item 1) stands
for the entire wave, unchanged by Stage 2 — nothing in §12 intervenes on
the candidate mechanism; Stage 2's instrumented cells are the closest
this wave gets to an interventional design (a controlled, matched-
everything-but-bias-mode pair of training runs with direct gradient
telemetry), but they still observe an emergent consequence of training,
not a designed perturbation of the gradient-reshaping mechanism itself.
H4's parameter-diff locus risk (§12.9 item 2 — the real compensatory
adaptation could sit upstream of `k_conv1d`) is untouched by Stage 2.
Single seed (0), single corpus (openr1-mix-ext) throughout Stage 2,
matching §12.5's registered design — no cross-seed or cross-corpus
replication of the gradient-flow finding exists yet.

**Ledger.** Stage 2 realized **0.7431 GPU-h** (log-timestamp bound,
re-derived above). Frozen-bias LM program ledger: 6.9288 (rung-1 6.8988
+ Stage 0 0 + Stage 1 ≈0.03) + 0.7431 = **≈7.672 / 135 GPU-h**. This
entire mechanism wave (Stage 0 + Stage 0.5 + Stage 1 + Stage 2), every
item, cost **≈0.773 GPU-h realized** against §12.6's own worst-case
2× estimate of 3.82 GPU-h (≈20% of the worst-case ceiling, and a small
fraction of the frozen-bias LM program's 135 GPU-h ceiling).

**MECH-WAVE STATUS: CONCLUDED, exploratory tier throughout, per §12.0's
original framing.** H1 REFUTED, H5 clean, H2 corroborated, H4 consistent
at one locus, H3 consistent (no sign pre-registered) — no hypothesis in
§12.2 remains untested; no further stage is registered or gated.

---

## 13. FIX-AT-SCALE WAVE — design Rev 1 (opened 2026-07-09, PI-directed GPU-saturation charter; revised 2026-07-09 per the §13.13 attack-round-1 verdict — see §13.14 for the finding→fix map)

**This is the wave the PI's directive text calls "this program's own §2"** (`STATE.md`
"GPU SATURATION DIRECTIVE (PI, 2026-07-09)": *"FIX-AT-SCALE ... registry =
FROZEN_BIAS_LM_DESIGN §2"*). It is numbered **§13** in this file's own continuous
heading scheme instead, because `§2` above (line 374, "The central design decision:
decouple the frozen bias from geo3/Newton-Schulz") is already load-bearing text from
the original rung-1 design — reusing the literal number would silently overwrite an
existing section header. Disambiguated once, here: **"§2" in the PI directive's own
words = this section, §13, in this file's actual numbering.**

### 13.0 Scope and framing — what this wave is, and what it is NOT

**What it is.** A PI-directed charter (quoted verbatim, `STATE.md` 2026-07-09: *"how
will [we] ensure that all these 8 gpu's are hot for the next few days. I don't want
these sitting idle anymore"*) to close this paper's own biggest disclosed 14M-only
caveat on the frozen-bias fix: `09_discussion_limitations.tex` item 6 states the
**pathology** (write-geometry attractor WORSENING with scale) is now measured
14M→98M→392M→1.31B (span-fraction 0.248→0.344→0.389→0.455, `SCALE_TRANSFER_DESIGN.md`
§5.10 Addendum / Track C Waves 1ext/2/3), but the **fix's own training evidence**
(does the frozen-bias intervention change that trajectory) exists **only at 14M**
(this file's own rung-1 VERDICT, above). This wave trains the fix at **98M and 392M**
to extend that evidence two rungs up the SAME ladder the pathology is already measured
on.

**What it is NOT — the §6.2 gate distinction, stated once, precisely, so no reader
conflates the two.** This file already has a formal mechanism for "launch the next
scale": §6.2's rung-2 (=98M in this file's OWN internal numbering, see §13.1 below)
gate, which requires rung-1 to read as a **CONFIRM** under the pinned CI formula on
both corpora, plus the co-primary, plus the Arm-2-exceeds-Arm-2′ licensing check
(§6.2 conditions 1–3). **Rung-1 did NOT clear that gate** — the VERDICT section
(above) classifies it as the **FOURTH OUTCOME, "sim-training divergence"**: the
primary delta is CI-excludes-zero on both corpora, but its **sign is the opposite**
of every mechanism prediction (per-token training makes `span_frac` WORSE, not
better). §6.2's own text is explicit: *"No licensed rung-2 launch... Rung-2 remains
PARKED."* **This wave does not claim to satisfy that gate — it is authorized on a
DIFFERENT basis entirely** (the PI's saturation directive + the paper's disclosed-caveat
argument, both independent of whether rung-1 was a mechanism CONFIRM), and it is
disclosed here, explicitly, as such: **a scale-extension of the FIX's OWN
literal-transplant training evidence, not a mechanism-confirmation follow-on that
inherited §6.2's gate.** §6.2's gate itself is untouched and remains formally unmet;
a future wave that wants to invoke it still needs the fresh design review §6.2's own
PARK clause requires. This wave's own outcome (§13.6) is pre-registered independent
of that gate's machinery.

### 13.1 Scale disambiguation (a real, previously-flagged collision — read before anything else)

Three different rung-numbering schemes coexist in this codebase and this wave sits at
the exact intersection point §6.2 (line 964) already flagged as a collision risk:

| Label used below | Params (measured) | Architecture | This file's OWN rung # | `lm_rd_rung_configs.py` `RUNGS` # |
|---|---|---|---|---|
| 14M | 14,048,896 | `d_model=256, n_layers=2, d_state=64` | rung-1 | n/a (Wave-C config) |
| **98M** | 97,618,176 | `d_model=768, n_layers=12, d_state=64` | rung-2 (PARKED, §6.2) | **1** |
| **392M** | 391,869,440 | `d_model=1536, n_layers=16, d_state=128` | not assigned a number in this file's OWN scheme (every "rung-3" string in this file refers to Track C's 1.31B rung, not a 392M label) | **2** |
| 1.31B | (self-terminated, ≈84.7% of budget) | `d_model=2560, n_layers=22, d_state=128` | n/a | 3 |

**This wave uses PARAM-COUNT LABELS ONLY (98M / 392M) throughout**, never bare
"rung-N," to avoid resurrecting the exact confusion §6.2 already named once.

### 13.2 One-sentence hypothesis

**The per-token frozen-bias fix (λ=0.58, the literal transplant already trained at
14M as this file's "Arm 2") produces a measurable, CI-distinguishable-from-zero
change in the write-geometry attractor (`span_frac`, Arm-off′-comparator convention,
§4.a/§4.a-i, unmodified) at 98M and 392M, with a val-loss cost that stays inside the
same `k=2·s_ref` gate rung-1 established, and this wave reports the MEASURED SIGN of
that change plainly — WIN if mechanism-predicted (stabilizing, `span_frac` falls),
PARTIAL/NULL otherwise — rather than assuming the 14M direction generalizes.**

**Load-bearing disclosure, stated up front, not buried in the self-attack register
(§13.11 item 1 restates this with the full evidentiary detail):** the 14M evidence
for THIS EXACT construction (per-token, λ=0.58) is **not** a stabilizing result — it
is the opposite (`span_frac` +0.1955/+0.2273, CI-excludes-zero, MORE collapsed than
the artifact-matched control). The hypothesis above is written as a real test of
transfer, not a foregone conclusion; a repeat of the 14M sign at 98M/392M is itself
informative (the sim-training-divergence pattern generalizes, not a 14M-only fluke),
and a REVERSAL toward the mechanism-predicted direction at scale is ALSO informative
(the fix "turns on" only past some capacity threshold). Both are pre-registered,
reportable outcomes — see §13.6.

### 13.3 Arms

**Per scale (98M, 392M), same corpora (`openr1-mix-ext`, `wikitext103_mix_eot_extended`
aka `wikitext-mix-ext` — the SAME extended mixes rung-1 and Track C Waves ≥2 used,
per the standing same-dataset rule), same tokenization (GPT-2 BPE, `vocab_size=50257`),
identical checkpoint cadence convention (every 1,000 steps):**

- **`arm_off`** — baseline, `anchor_active=False`. Exactly the plain architecture
  Track C already trained at both scales (98M: Wave 1ext, ext-mix; 392M: Wave 2) —
  this wave RE-TRAINS it fresh (not cited from the archive), for the same reason
  rung-1's own Arm 1 was re-trained fresh rather than reused: every arm in one
  comparison must come from the identical harness invocation, avoiding cross-session
  harness-drift risk (§5, Arm 1 rationale, unchanged here).
- **`arm_per_token`** — the pinned fix, λ=0.58, `B = key_anchoring.random_unit_rows_init(
  n=50257, d=d_state, seed=ANCHOR_INIT_SEED)`, `k_biased = normalize((1-λ)·k_raw +
  λ·B[token_id])`, applied post-conv-key, every token, every position, every step.
  This is the LITERAL transplant of rung-1's own "Arm 2" at the new scales — same
  construction, same λ, same seed convention, only `d_state` changes (64→64 at 98M,
  no change; 64→128 at 392M, `B` reshaped to `(50257, 128)` accordingly, per
  `lm_rd_rung_configs.py` `RUNGS`).

**MANDATORY, near-zero-cost addition — `arm_off′` eval-only retrofit (NOT a third
training arm, reuses `arm_off`'s own checkpoints, no new training cell):** the SAME
per-token blend (`λ=0.58`, same table `B`) applied to `arm_off`'s checkpoint ONLY at
the measurement pass, never during training — exactly rung-1's `Arm 1′` (§5, above).
**This is not optional.** Comparing `arm_per_token` (post-blend, trained through the
bias) directly against `arm_off` (never blended at all) reintroduces the EXACT
auto-pass artifact the round-2 attack killed as FATAL in this file's own history
(§7.1, "the round-2 attack's FATAL finding"): the blend arithmetic itself, applied
once at eval time to any static key population, dominates measured `span_frac`
regardless of what training did. `arm_off′` is the artifact-matched reference that
makes the comparison mean anything; every one of §13.5/§13.6's bars below is read
against `arm_off′`, never bare `arm_off`.

**Excluded from the PRIMARY (n=3/n=3, CI-gated) manifest this wave, disclosed
as a scope decision, not an oversight (§13.11 item 2 has the full
self-attack; text below is the original Rev 0 rationale, superseded in part
by the Rev 1 addition immediately after it): the global-vector control
(rung-1's "Arm 2′," which was registered "NOT cuttable... core to both
rungs" at 14M) is NOT trained as a full n≥2 arm at 98M or 392M this wave.**
Rung-1's own data shows this is the construction that actually moved in the
mechanism-predicted (stabilizing) direction (−0.3319/−0.2308 CI-excludes-zero
both corpora) — dropping the FULL arm here trades away the §7.1a "is this
about token-identity keying at all" licensing check at scale as a CI-gated
result, purely for budget reasons (a third full arm at either scale would
add another complete training manifest, §13.7). This remains the single
most consequential scope cut in this design and is flagged, not hidden — the
Rev 1 addition below narrows, but does not close, the gap.

**Rev 1 addition (§13.13 BINDING item 1) — `arm_global_probe`, an
exploratory-tier global-vector probe.** n=1 seed (no seed replication —
exploratory/descriptive tier only, mirroring §12's tier convention: every
output artifact carries a `"tier": "exploratory-probe — NOT a confirmatory
bar, n=1"` stamp, the same discipline `mech_schema.wrap_exploratory()`
enforces in §12), run at BOTH scales (98M at the full Track-C-matched
67,547-step budget; 392M at the SAME reduced 20,000-step budget
`arm_per_token` uses, §13.7), BOTH corpora — **4 training cells total**
(98M×2 corpora + 392M×2 corpora, each ×1 seed). Construction: the LITERAL
transplant of rung-1's own Arm 2′ (§ "Arm 2′ — Global single frozen vector,
no token lookup," above) — same frozen table `B`
(`key_anchoring.random_unit_rows_init(n=50257, d=d_state, seed=ANCHOR_INIT_SEED)`),
collapsed to a single fixed vector `b_global = normalize(mean_i B[i])`,
blended at every position/step as `k_biased = normalize((1-λ)·k_raw +
λ·b_global)`, identical λ=0.58, identical insertion point, identical density
to `arm_per_token` — only `d_state` (hence `b_global`'s dimension) changes
per scale, same convention as `arm_per_token`. **Comparator: reuses
`arm_off`'s own already-scheduled checkpoints** (already being trained in
the PRIMARY manifest — zero incremental training cost), re-blended at eval
time with `b_global` instead of the per-token table, mirroring `arm_off′`'s
own eval-only-retrofit convention (this section, above) and rung-1's own
Arm 1″ control. No separate training cell is needed for the comparator, and
no additional eval-only GPU-h line is added beyond the 4 training cells
below — the eval-only pass itself is the same sub-0.02-GPU-h/pass class
§13.7's existing eval-only rows already price, immaterial at n=1×2 scales×2
corpora. **Pre-registered — what this probe informs:** whether the
geometry-STABILIZING construction (the one that actually worked at 14M)
transfers to scale, independent of whatever `arm_per_token`'s own primary
bars (§13.6) read. It is the input to any future "deploy the right arm"
decision, not a claim this wave resolves — see §13.6's Rev 1 note on how
this probe's own reading relates to (and does not gate) the WIN/PARTIAL/NULL
bar, and §13.11 item 2 for the licensing-check gap it narrows without
closing (n=1, no CI, cannot itself license a transplant claim the way a
full n≥2 §7.1a comparison would).

### 13.4 Observables — reused unmodified from §4, no new instrument built

- **Primary + co-primary (§4.a / §4.a-i, `lm_attractor_probe_rd.py`, byte-identical
  to every prior harvest — md5 `3fb0f80028477d0b1cefe468c81b1da4` per the rung-2/
  wave-1ext harvests, re-verify at build time, do not assume):** post-blend
  `span_frac(arm_per_token) − span_frac(arm_off′)` (co-primary: the SAME instrument
  on pre-blend `k_raw`, both arms, no blend anywhere in that population). This is
  the SAME cross-scale-comparable `span_frac` convention Track C's own Fig. 9 Panel A
  already uses across all four ladder points (14M/98M/392M/1.31B) — reusing it here
  is not a new cross-scale-comparability assumption, it is the SAME one the paper's
  own headline scaling claim already rests on.
- **Secondary/gate (§4.b, §7.2 convention, unmodified): val loss**, tolerance
  `= arm_off`'s own fresh per-seed val-loss mean `+ 2·s_ref` (k=2, this codebase's
  own house `mean_ref ± k·s_ref` convention, `key_anchoring.derive_engaged_bands`),
  computed and blind-pinned from THIS wave's own `arm_off` data before `arm_per_token`
  is inspected — identical mechanical discipline to rung-1's `BANDS_PINNED-FrozenBias.json`
  pattern, extended to cover both new scales (§13.10).

**Rung-1's own side-effect bound, cited as the standing evidentiary reference
(the task's own "≤X%" ask):** at 14M, the val-loss gate PASSED in every
arm/corpus with clear margin under the `k=2·s_ref` ceiling — openr1-mix-ext
ceiling 2.1935 vs. realized Arm-2 mean 2.1184 (margin 0.0751 nats, ≈3.4% of the
ceiling value); wikitext-mix-ext ceiling 4.3828 vs. 4.3426 (margin 0.0402, ≈0.9%).
This wave's own gate (§13.10) re-derives fresh bands from `arm_off`'s OWN 98M/392M
seed spread — the 14M numbers above are cited as the standing precedent that the
fix's val-loss cost has, so far, never approached its own gate, not as a number
this wave inherits directly (a flat percentage import across three different
model scales would repeat exactly the mistake attack finding 4 already fixed once,
§7.2).

### 13.5 Instrument comparability — no new confound relative to what §4.a already licenses

`span_frac` is already the cross-scale-comparable convention this program's OWN
scaling headline (`09_discussion_limitations.tex` item 6, Fig. 9 Panel A) is built
on — nothing new is being asked of the instrument here that Track C's own 4-point
ladder didn't already ask of it. What IS new: this wave's `arm_off′`-vs-`arm_per_token`
comparison has never been run at 98M or 392M before (only at 14M) — the instrument's
cross-scale comparability is inherited, but the SPECIFIC comparator pair
(post-blend-vs-eval-only-retrofit) at these two scales is genuinely new data, not a
citation.

### 13.6 Pre-registered outcomes — WIN / PARTIAL / NULL, both independently reportable and both publishable

Per scale, per corpus, read against the SAME pinned-CI-formula discipline §7.1-real.1
already established (`mean_delta ± t(2,0.975)·s_ref/√n`, thresholds re-derived from
THIS wave's own fresh `arm_off′` cross-seed spread at that scale — not inherited from
14M's 0.0546/0.1064, which are architecture-specific to the 14M cell and have no
standing claim on a 98M or 392M noise floor):

- **WIN** — primary AND co-primary BOTH clear their own bar in the
  mechanism-predicted (stabilizing, `span_frac` FALLS) direction, on BOTH corpora,
  val-loss gate passes. This is the outcome that would REVERSE rung-1's own sign —
  itself the single most interesting possible result this wave could produce (a
  scale-dependent mechanism onset), reported with the same "surprising, not spun"
  register the rung-1 VERDICT used for its own control-contrast finding.
- **PARTIAL** — primary and co-primary move in the SAME direction as rung-1's own
  14M reading (destabilizing) on at least one corpus, OR a split between corpora/
  instruments (mirroring §1.3's uninterpretable-result register) — reported as
  "the sim-training-divergence pattern persists/attenuates at scale," feeding
  `09_discussion_limitations.tex` directly (this section already carries the 14M
  caveat; this wave's job is to extend it, not necessarily resolve it).
- **NULL** — CI includes zero on both corpora, both instruments, at both scales —
  the fix's own training-mediated effect (whichever sign) does not replicate outside
  noise at these larger scales. Reported plainly, per the standing "negative results
  are data" rule (CLAUDE.md).

**Every one of these three outcomes is publishable and each feeds a specific,
already-open section of the paper** (07_the_fix's own arc for WIN, 09_discussion
item 6 for PARTIAL/NULL) — no outcome of this wave is a "failed" wave in the
budget sense; the wave's job is measurement, not confirmation of a specific sign.

**Rev 1 note (§13.13 BINDING item 1) — `arm_global_probe` does not gate any
of the three outcomes above.** The WIN/PARTIAL/NULL bar stays exactly as
specified — `arm_per_token` vs. `arm_off′`, CI-gated, n=3/n=3 — unchanged by
the probe's existence. The probe's own reading (§13.3) is reported
descriptively, alongside, at exploratory/descriptive tier (n=1, no CI): it
answers a different, narrower question ("does the CONSTRUCTION KNOWN TO
STABILIZE at 14M still stabilize at scale") than §13.6's own primary
question ("does the DEPLOYED construction's effect transfer/reverse/null
at scale"). A reader must not treat the probe's sign as evidence for or
against any of WIN/PARTIAL/NULL above — that would silently promote an n=1
exploratory reading to gate a CI-pinned bar, exactly the descriptive-tier
conflation §13.10 gate 5 exists to prevent for the primary arms.

### 13.7 Cost table — REALIZED rates, with a load-bearing correction to the pre-registered figure

**Correction, stated first, because it changes every number below and must not be
silently inherited.** The PI directive's own recorded planning figure
(`STATE.md`: *"realized 392M rate: 128.3 GPU-h / 91,552 steps = 0.0014017 GPU-h/step
→ ≈28.03 GPU-h per 20,000-step-equivalent cell"*, sourced from
`HEAD_TO_HEAD_DEMO_DESIGN.md:1604-1605`) **divides the 6-CELL WAVE TOTAL by a
SINGLE cell's own step count — a 6× error, not a per-cell rate.** Verified two
independent ways this pass: (1) `EXPERIMENT_LOG.md:3937-3948` states "6× rung-2
(`dm1536/L16/ds128`)... training measured 128.3 GPU-h" in the same sentence
structure as "6× mixcontrol... 0.46 GPU-h" three clauses later (unambiguously a
6-cell total there too); (2) `SCALE_TRANSFER_DESIGN.md:751` pre-registers, BEFORE
any 392M data existed, "0.836 s/step × 91,552 steps ≈ 21.6 h/**run** → ≈129
GPU-h for **the wave**" — the realized 128.3 matches the ≈129 WAVE estimate almost
exactly, confirming the per-RUN figure is ≈21.4–21.6 GPU-h, not 128.3. **The
correct, realized, twice-cross-checked per-step rate is 0.836 s/step
(≈2.322×10⁻⁴ GPU-h/step), not 0.0014017 GPU-h/step (6× too high).** A second,
independent error was found and must ALSO not be inherited: this file's own §8.3
(line 2084) cites a "banked calibration 0.7135 s/step" for its internal
98M rung — traced this pass to `SCALE_TRANSFER_DESIGN.md:793` /
`EXPERIMENT_LOG.md:5472`, **this is rung-3's (1.31B, batch=16) SOLO calibration
constant**, later found itself to be 1.985× optimistic vs. the measured 1.416
s/step (the exact overrun that caused rung-3's self-termination). **Neither
mislabeled figure is used below.**

**Verified realized rates actually used (plain-baseline architecture, extended-mix
corpora, batch=32 both scales, `BATCH_SIZE_BY_RUNG={1:32,2:32,3:16}`,
`run_lm_rd_trackc_sweep.py:223`):**

| Scale | Source | Cells | Steps/cell | Realized total | Per-cell | Per-step |
|---|---|---|---|---|---|---|
| 98M | Wave 1ext (ext-mix), `EXPERIMENT_LOG.md:4001` | 6 | 67,547 | 26.87 GPU-h | 4.478 GPU-h | 0.236 s/step |
| 392M | Wave 2 (ext-mix), `EXPERIMENT_LOG.md:3937` | 6 | 91,552 | 128.3 GPU-h | 21.38 GPU-h | 0.836 s/step |

**Anchor-blend overhead assumption (flagged, not verified at these scales):** at
14M, all 20 rung-1 cells (off / per-token / global, every λ) fell in a tight
899–914s band regardless of arm (VERDICT, "Verification performed this pass") —
i.e. the frozen-bias blend added no measurable per-step overhead at 14M. This
wave carries that finding forward as the planning assumption for BOTH arms at
98M/392M (the gather-scatter-normalize operation is O(batch×seq), independent of
`d_model`/`n_layers`, so it should become a SMALLER fraction of per-step cost as
scale grows, not larger) — but this is explicitly UNVERIFIED at 98M/392M and is
exactly what the mandatory timing pilot (§13.10) exists to confirm before the full
sweep launches, not to assume.

**Step-budget decision — 98M matches Track C exactly, 392M does NOT, disclosed
explicitly:**

- **98M: full Track-C-matched step count, 67,547 steps/cell.** Affordable (below),
  and gives a DIRECT apples-to-apples read against the already-measured Track-C
  98M baseline (`span_frac=0.344`, Wave 1ext, ext-mix) — no reason to shorten it.
- **392M: reduced to 20,000 steps/cell, NOT the Track-C-matched 91,552.**
  Disclosed deviation from "same step budget as Track C" — matching Track C's
  full 91,552-step 392M budget at even the seed floor (n=2, 8 cells) costs
  `8 × 21.38 = 171.1 GPU-h` at **1× alone**, before contingency, before 98M's own
  cost, before eval-only passes — an order of magnitude larger single-scale
  commitment than any wave this program has run to date (rung-1's own committed
  ask was ≈11–14 GPU-h, §8.5.1). **20,000 steps is not an arbitrary shrink**: it
  is THIS SAME PROGRAM'S OWN precedent (§12.5/§12.12, the mechanism-wave's Stage-2
  H3 gradient-flow probe, "the FULL 20,000-step branch") — a training length
  already shown, in this exact codebase, to produce clean, non-degenerate,
  informative training-mediated geometry/gradient signal (H2/H3/H4's own findings
  are ALL derived from 20,000-step cells at 14M). Using it again here is reuse of
  a validated convention, not an invented number.

**Cost table (2× contingency, §8.4's rung-3-lesson convention, unchanged — see
disclosure below on why this stays at 2× despite these being REALIZED, not
placeholder, base rates):**

| Item | Cells | Steps/cell | GPU-h (1×) | GPU-h (2×) |
|---|---|---|---|---|
| 98M training (`arm_off` + `arm_per_token` × 2 corpora × 3 seeds, incl. 1 calibration cell) | 12 | 67,547 | 53.74 | 107.48 |
| 98M eval-only (`arm_off′` retrofit 6 passes + pre-blend `k_raw` co-primary 12 passes, ESTIMATE scaled from rung-1's realized 0.02 GPU-h/pass × params-ratio ≈6.95×) | 18 passes | n/a | 2.50 | 5.00 |
| 392M training, n=3 (PRIMARY default) | 12 | 20,000 | 55.73 | 111.47 |
| 392M eval-only, n=3 (18 passes, ESTIMATE scaled ×27.9×) | 18 passes | n/a | 10.04 | 20.09 |
| Wave −1 instrument-validity + code-path-equality smoke (both scales, mirrors §8.0/§8.0b) | — | n/a | 0.20 | 0.40 |
| **Total, n=3/n=3 PRIMARY DEFAULT (Rev 0)** | 24 training + 36 eval-only | | **122.21** | **244.44** |
| **`arm_global_probe`, n=1, both scales, both corpora (Rev 1, §13.13 BINDING item 1 — see §13.3)** | 4 | 67,547 (98M) / 20,000 (392M) | **18.30** | **36.60** |
| **Total, n=3/n=3 PRIMARY DEFAULT + Rev 1 probe** | 28 training + 36 eval-only | | **140.51** | **281.04** |
| *392M training, n=2 (cost-relief fallback, §13.3's task-authorized "if cost demands")* | *8* | *20,000* | *37.16* | *74.31* |
| *392M eval-only, n=2 (12 passes)* | *12 passes* | *n/a* | *6.70* | *13.39* |
| *Total, n=3-98M / n=2-392M FALLBACK (Rev 0)* | *20 training + 30 eval-only* | | *100.30* | *200.58* |
| *Total, n=3-98M / n=2-392M FALLBACK + Rev 1 probe* | *24 training + 30 eval-only* | | *118.60* | *237.18* |

**Rev 1 probe row, arithmetic shown (§13.13 BINDING item 1):** 2 cells at
98M's 4.478 GPU-h/cell (openr1 + wikitext, ×1 seed) + 2 cells at 392M's
4.671 GPU-h/cell (openr1 + wikitext, ×1 seed) = `2×4.478 + 2×4.671 = 8.956 +
9.342 = 18.298 ≈ 18.30` GPU-h at 1×, `≈ 36.60` at 2× — both rates taken
directly from this section's own verified per-cell figures above (98M
4.478, 392M-20k 4.671), no new rate assumption. No incremental eval-only
line is added (§13.3's Rev 1 addition explains why: the comparator reuses
`arm_off`'s already-scheduled checkpoints at near-zero eval-only cost,
already priced inside this table's existing eval-only rows' order of
magnitude).

**Margin vs. the 300 GPU-h cap, PRIMARY + Rev 1 probe (the binding total for
this revision):** `300 − 281.04 = 18.96` GPU-h headroom (**≈6.3% of the
cap**) — down from Rev 0's ≈23% margin (see §13.8), but still positive
margin, not a breach. The FALLBACK + Rev 1 probe total (237.18 GPU-h, 2×)
retains a much larger ≈21% margin if the n=2/392M cost-relief option is
invoked instead.

**Why 2× contingency is kept despite realized (not placeholder) base rates:**
rung-3's own history is the direct counter-example to "measured means safe" — its
own SOLO calibration (0.7135 s/step) was itself a real, measured number, and still
undershot the sustained real rate by 1.985× (`EXPERIMENT_LOG.md:5560`), causing a
self-terminated run at 84.7% of budget. The 98M/392M rates used here ARE more
trustworthy than rung-3's ever were (both are FULL, completed, non-timed-out waves,
not single-cell extrapolations) — but the one genuinely new variable at these two
scales (the anchor-blend's own overhead, verified only at 14M) is exactly the kind
of unverified-at-this-scale assumption 2× contingency exists to buffer.

### 13.8 Ledger — reconciling the PI's recorded ~170 GPU-h figure with this wave's own arithmetic

The PI directive's own recorded figure is **"~170 GPU-h"** (`STATE.md`, explicitly
approximate — a pre-design placeholder, the same role §5.6's "±2×" pre-calibration
table played for Track C before its own Wave −1 revised it). Per §13.7's correction,
the arithmetic behind that placeholder inherited the 6× per-step error — this
design's own job is exactly to firm that number up, the same way Track C's Wave −1
superseded its own brief's placeholder. **Reconciliation:**

- The n=3-98M/n=2-392M FALLBACK (≈200.58 GPU-h, 2×) is the closest fit to the
  recorded ~170 figure and is fully costed above as the explicit, task-authorized
  ("n=2 at 392M if cost demands") cost-relief option.
- **This design's own PRIMARY recommendation is n=3/n=3 (≈244.44 GPU-h, 2×)** —
  chosen over the fallback because the PI's own directive text argues for MORE
  spend, not less ("pre-registered seed extensions exercised liberally... grow
  seeds/grids to use the room," the same directive's own language for the sibling
  CAPABILITY STAGE 2 wave), the operative grant rate is ≈192 GPU-h/day (this
  wave's entire 2× committed ask is under 1.3 days of ONE day's supply), and the
  third 392M seed closes exactly the CI-width self-attack item (§13.11 item 3)
  a n=2 design would otherwise carry as an open weakness.

**Proposed ledger: `fix-at-scale`, cap = 300 GPU-h** (headroom-cap convention, not
a target — mirrors rung-1's own 135 GPU-h ceiling against an ≈11–14 GPU-h committed
ask, §8.1). Rev 0 figure, cited for the record: 300 GPU-h gave ≈23% headroom over
the n=3/n=3 primary default alone (≈244.44 GPU-h) and ≈50% over the fallback alone,
while staying a similar order of magnitude to Track C's own 300 GPU-h program
ceiling (`run_lm_rd_trackc_sweep.py:306`, itself already 12.23 GPU-h over-spent
and disclosed as such) — a genuinely NEW, separate ledger, not a draw against
frozen-bias's existing 135 GPU-h ceiling (§8.1) or its ≈123.57 GPU-h of unused
headroom (`STATE.md` LEDGERS: "frozen-bias: 11.43/135 (~123 headroom — funds
the head-to-head)" — that headroom is EARMARKED for the head-to-head demo per
STATE.md's own parenthetical, not free for this wave to silently draw against).

**Rev 1 update (§13.13 BINDING item 1, supersedes the ≈23% figure above as the
binding number for this revision):** with the `arm_global_probe` addition folded
in (§13.7's recomputed table), the PRIMARY-default total rises to **281.04 GPU-h
(2×)**, leaving **18.96 GPU-h headroom (≈6.3% of the 300 GPU-h cap)** — still
positive margin, the cap is not breached, but the room for further scope growth
inside this ledger without a cap increase is now materially smaller than Rev 0's
≈23%. If the n=2/392M FALLBACK is invoked instead, the probe-inclusive total is
237.18 GPU-h (2×), restoring a ≈21% margin. This tightened margin is disclosed
here, not silently absorbed — a future BUILD or harvest pass that finds any of
§13.7's ESTIMATE-tagged eval-only rows running materially over their scaled
projection should re-check against this narrower 6.3% band, not the Rev 0 23%
figure.

**Authorization: the PI's 2026-07-09 GPU-saturation directive** (`STATE.md`,
quoted in full at §13.0) is the recorded authorization for opening this new ledger,
per this project's own "never-self-amend" rule (`STATE.md`: *"RESPONSE (recorded as
the authorization the never-self-amend rule requires)"*) — this design registers
the ledger, it does not self-authorize it.

**Circuit breaker (task-mandated, scoped sanely — NOT the 10× bracket a
pre-calibration placeholder table uses, since that would exceed this wave's entire
proposed cap in one cell):** **1.5× measured/calibrated per-step rate, hard-abort
per cell**, checked live against the cell's OWN calibration constant (not a
program-wide constant), mirroring the capability campaign's own precedent of
designing its own enforceable mechanism rather than inheriting one. **This
threshold is independently validated by rung-3's own failure**: rung-3's real
overrun was **1.985×** its solo calibration — a 1.5× abort would have FIRED before
that run burned its full budget, exactly the failure mode this breaker exists to
catch cheaply instead of expensively. Mechanically: each cell's supervisor script
checks `wall_s_so_far / steps_so_far` against `1.5 × calibrated_per_step_s` every
checkpoint (1,000-step cadence); on breach, the cell writes an `ABORT_1.5X.json`
sentinel with the last-known rate and exits — the sweep's own budget guard
(`budget_guard`, `run_lm_rd_trackc_sweep.py:794`) is a SEPARATE, program-wide check
and stays unchanged; this is a NEW, per-cell, real-time check this wave's own
launcher must add.

### 13.9 GPU allocation — box layout

**Occupancy at design time (this session, verify fresh before any BUILD agent
executes — standard discipline, §12.0's own precedent):** `STATE.md`'s LEDGERS
section (this session's snapshot) reads "GPUs 0-7 all currently idle," but the
LATEST narrative entries (same file, later lines) record the head-to-head rung-1
wave HARD-STOPPED at its calibration-verdict gate (2026-07-09 box time, an
instrument-design failure, unrelated to this wave) with "GPUs pass to the
CAPABILITY campaign in the interim — box stays utilized," and CAPABILITY STAGE 2
is being designed CONCURRENTLY with this wave, by a separate design agent, under
the SAME PI directive (`STATE.md`: "registry = CAPABILITY_SEPARATION_DESIGN §2").
**Net: this design cannot assume a specific live occupancy snapshot; it proposes a
partition and registers the cross-wave ordering discipline this codebase already
has a precedent for (§8.2a, the frozen-bias/key-anchoring contention-gate mirror),
rather than hard-coding a GPU list that will be stale by build time.**

**Proposed partition (all 8 GPUs "in play" per the directive's own text, "GPU 7 no
longer reserved"):**

- **GPUs 0–3: 392M cells.** Single-GPU per cell (see DDP note below) — up to 4
  cells run concurrently, the remaining 2–8 (n=3 or n=2) cells queue behind them
  as slots free, matching Track C's own proven `--gpus N --gpu-offset` sweep
  pattern (`run_lm_rd_trackc_sweep.py` usage docstring) — no new orchestration
  code needed.
- **GPUs 4–6: 98M cells.** Same pattern, up to 3 concurrent.
- **GPU 7: flex slot** — this wave's own calibration/timing-pilot cell runs here
  FIRST, alone, before either bank above launches its full manifest (§13.10);
  after that, GPU 7 is available for whichever concurrent wave (CAPABILITY STAGE
  2, or a contingent third wave, `STATE.md`'s "2×2-at-scale (~168 GPU-h) iff
  tonight's screening splits") needs an interleave slot — this design does not
  claim GPU 7 exclusively, matching the directive's own "small-cell waves
  interleaved" framing.

**DDP — explicitly NOT used, a disclosed deviation from the directive's own
planning language.** `STATE.md`'s directive text says "392M DDP cells" as a
planning sketch, but **zero DDP/`torchrun` code path exists anywhere in
`run_lm_rd_trackc_sweep.py` or `lm_pretrain_rd.py`** (verified this pass, grep
clean) — every realized Track C 392M cell to date ran single-GPU
(`BATCH_SIZE_BY_RUNG[2]=32`, one process per GPU, parallelized ACROSS cells via
GPU assignment, never within one). §13.7's own corrected cost table shows
single-GPU 392M cells are already affordable (21.38 GPU-h/cell at full step count,
4.64 GPU-h/cell at this wave's reduced 20k-step budget) — DDP would only buy
wall-clock speed, not GPU-h savings, at the cost of building and smoke-testing a
genuinely new, unvalidated code path in a harness that has never needed one. **This
design recommends against building DDP for this wave** — single-GPU-per-cell,
more cells in parallel across the GPU bank, achieves the same "keep GPUs hot"
saturation goal with zero new engineering risk. If wall-clock time (not GPU-h) is
later found to be the binding constraint, DDP is a scoped, separate build item —
not bundled into this wave by default.

**Cross-wave ordering (mirrors §8.2a, registered from THIS side since CAPABILITY
STAGE 2's own design doc is owned by a different, concurrently-working agent):**
this wave's own calibration/timing-pilot cell (§13.10) should not launch on a GPU
bank another wave's OWN first-stage calibration is actively contending for — check
live queue/tmux state (`tmux list-sessions`, `nvidia-smi`) immediately before
launch, not assumed from this design pass's own occupancy snapshot. **Cross-reference
requirement, not performed here:** `CAPABILITY_SEPARATION_DESIGN.md` should mirror
this same constraint from its own side, exactly as `KEY_ANCHORING_DESIGN.md` §12.2.3
mirrored into this file's own §8.2a.

### 13.10 Gates

1. **Wave −1 instrument-validity + code-path-equality smoke** (mirrors §8.0/§8.0b,
   re-run at each new scale's real `d_state` — the pre-blend/post-blend code path
   must be re-verified bit-identical at `d_state=64` (98M, unchanged from 14M) AND
   `d_state=128` (392M, genuinely new shape).
2. **Deploy closure**: scripts on-box byte-verified against this repo's committed
   copies (md5, mirroring §12.12's own convention) before any cell launches; GPU
   idle/occupancy re-verified live, not inherited from §13.9's snapshot.
3. **Timing pilot, per scale (NEW gate, not in rung-1's design — that wave never
   needed one, since 14M's timing was already banked from Wave C):** a single,
   short (≤2,000-step) `arm_per_token` cell, run ALONE, before the calibration
   cell — confirms the §13.7 no-overhead-from-the-blend assumption holds at this
   scale (real per-step rate within the 1.5× circuit-breaker band of the plain
   Track-C rate) BEFORE committing to a full-length calibration cell, let alone
   the full manifest. This is the gate that would have caught rung-3's own
   1.985× miss two orders of magnitude cheaper than rung-3's own discovery-by-
   timeout did. **(Rev 1 addition, §13.13 BINDING item 2, near-zero cost):** the
   SAME pilot cell also logs **peak VRAM (GB)** — `torch.cuda.max_memory_allocated()`
   at pilot completion, per GPU, with an `nvidia-smi --query-gpu=memory.used`
   cross-check — for both scales. No separate pass is added; this is a logging-only
   change to the existing pilot cell. This closes a real instrumentation gap: no
   VRAM figure appears anywhere in §13.7–§13.9 today, and §13.9 proposes running up
   to 4 concurrent 392M cells and 3 concurrent 98M cells per GPU bank — per-cell
   VRAM headroom under that concurrent occupancy is exactly the kind of assumption
   this design should measure once, cheaply, at pilot time rather than discover via
   OOM once the full 28-cell manifest (§13.7, Rev 1 total) is already running.
4. **Calibration-first per scale (CLAUDE.md's standing rule, §6.3's own precedent):**
   one full-length `arm_per_token` cell (openr1-mix-ext, seed 0), run alone, THEN
   inspected — loss curve, `span_frac` trajectory, no NaN/divergence, no OOM —
   against BANDS anchored to Track C's own archived val-loss curves at the
   matching config: **98M (Wave 1ext, ext-mix) self-loss openr1 1.290 / wikitext
   3.189; 392M (Wave 2, ext-mix) self-loss openr1 1.135 / wikitext 2.847**
   (`EXPERIMENT_LOG.md:3980-3984`, `4046-4047`) — a calibration cell landing far
   outside these bands (accounting for the shorter 20k-step budget at 392M, which
   should read HIGHER than the full-length 91,552-step reference, not lower) signals
   a real problem before the remaining 23 cells launch, not after.
5. **Blind-pin discipline, sequencing FIXED (the process lesson rung-1's own
   VERDICT names explicitly — "the blind-pin must be written and committed
   BEFORE the training manifest launches, not after"):** `BANDS_PINNED-FrozenBias-
   98M.json` / `-392M.json` are written and committed from `arm_off`'s fresh
   per-seed data IMMEDIATELY after `arm_off`'s cells complete and validate,
   BEFORE `arm_per_token`'s cells are inspected — and, per rung-1's own
   post-mortem, this pin's timestamp must be checked (`assert_blind_not_broken`)
   to strictly precede `arm_per_token`'s own training START times, not merely
   exist before the measurement script runs. Getting this right is what promotes
   this wave's own result out of the DESCRIPTIVE TIER rung-1's result was stuck
   in (§ "The descriptive-tier caveat," above) — the single most avoidable process
   mistake this design can close relative to its own predecessor.
6. **Chain + tmux + supervisor**, per-config `try/except`, resume-safe (skip
   already-completed cells by output validity, not existence) — standing rule,
   unchanged.
7. **Harvest with verify-vs-raws**: every headline delta recomputed once, directly
   from the per-seed raw JSONs, cross-checked against the report to full float
   precision — mirrors the VERDICT section's own "Verification performed this
   pass" convention exactly.

### 13.11 Self-attack register — ≥6 items, weakest points first

1. **[MOST CONSEQUENTIAL] The pinned fix's own 14M evidence shows the WRONG-DIRECTION
   effect for the exact construction this wave commits to re-testing.** `arm_per_token`
   at 14M made `span_frac` WORSE (+0.1955/+0.2273, CI-excludes-zero, both corpora)
   — the opposite of "reduces the write-geometry attractor." This wave's own §13.2
   hypothesis is therefore NOT a safe bet on the 14M data; it is a genuine test of
   whether that sign generalizes, reverses, or attenuates with scale. **Adjudicated,
   pinned:** tested as literally specified (per-token, λ=0.58, unchanged) — testing
   the AS-VALIDATED-AT-14M construction is the honest scale-transfer question the
   paper's caveat actually asks; silently swapping to the global-vector construction
   (which DID stabilize at 14M) would answer a different, easier question while
   still calling it "the fix," which this design explicitly declines to do. The
   excluded global-vector arm (§13.3) is the disclosed, budget-permitting way to ask
   the easier question later, not smuggled into this wave's own headline.
   **Rev 1 update — this adjudication is now SETTLED by an independent attack round,
   not merely self-adjudicated (§13.13 ATTACK ROUND 1 VERDICT, cited here in full):**
   the wrong-direction claim restated above is CONFIRMED TRUE by the round-1
   attack's own raw-table re-verification (per_token Δspan_frac +0.1955/+0.2273,
   both CI-exclude-zero = WORSE; global-vector −0.3319/−0.2308 = stabilizing;
   rung-1 was a FOURTH-OUTCOME descriptive tier, never a clean CONFIRM for either
   arm). The same attack round independently checked whether the h2h §1.2 pin of
   `per_token` as the DEPLOYED arm rests on a misread of these numbers — it does
   not: the pin quotes these exact figures and chooses `per_token` for disclosed
   engineering reasons (Newton-Schulz/β-uniformity stability + a clean val-loss
   gate), not a claim that `per_token` is the geometry-stabilizing construction.
   This wave's primary manifest still tests `per_token` as literally specified,
   unchanged from the paragraph above. One clause above is superseded by §13.3's
   Rev 1 addition: "the excluded global-vector arm... disclosed, budget-permitting
   way to ask the easier question later" — "later" is now THIS wave, via
   `arm_global_probe` at exploratory/descriptive tier (n=1), not deferred further;
   see §13.3 and §13.6's Rev 1 note for what that probe does and does not license.
2. **The global-vector control (Arm 2′) is missing at both new scales** — the
   §7.1a "is this about token identity, or any constant vector" licensing check
   cannot be run this wave. A WIN outcome (§13.6) at 98M/392M would therefore be
   reported as "the per-token construction stabilizes at scale," NOT as "the
   constancy-suffices mechanism transplants at scale" — the two claims are not
   equivalent, and this design does not have the instrumentation to distinguish
   them this wave. Flagged as the natural next follow-on if budget allows.
3. **392M's n=2 fallback (if adopted over the n=3 primary default) has the same CI
   weakness rung-1 itself never had to face** — rung-1 used n=3 at 14M throughout;
   an n=2 392M cell has `df=1`, `t(1,0.975)=12.706` (not `t(2,0.975)=4.303`) if the
   SAME pinned-CI-formula convention is reused verbatim — a materially WIDER
   interval, harder to CI-exclude-zero even with a real effect present. **This
   pushes the design toward the n=3/n=3 primary default (§13.8) rather than the
   fallback** — disclosed here as the actual reason, not merely "more seeds are
   better in general."
4. **λ=0.58 was tuned at 14M (§10.13.4's cross-cell mean from the synthetic-grammar
   mechanism-tier wave) — is re-tuning needed at scale, or is using it as-is the
   honest test?** **Adjudicated, pinned:** used as-is, unchanged. Re-tuning λ per
   scale would conflate "does the FIX (as validated) transfer" with "does SOME
   value of λ work better at this scale" — a different, larger, unbudgeted design
   (a λ-sweep at two new scales, §5's Arm-2-λ-mini-sweep precedent shows even a
   RUNG-1-ONLY 2-point mini-sweep already needed its own registered scope). Using
   λ=0.58 unchanged is the literal transplant this wave is chartered to test, per
   §13.0's own framing — the CLAUDE.md hard rule ("hold the second axis fixed when
   testing a primary hypothesis") applies directly: λ is the second axis here,
   held fixed.
5. **The fix's known interaction with `use_qk_l2norm_in_kernel=True` is untested
   this wave.** This flag is ON in every cell this design proposes (the stock,
   production-matching default, `lm_pretrain_rd.py:984`) — a concurrent,
   independently-chartered wave (`ATTRACTOR-ROBUSTNESS 2×2`, git history
   `55f0cfc`, qk-norm × gating) exists specifically to probe this interaction at
   14M, but its results are not yet available to this design and are NOT folded
   in. If that wave finds a qualitative qk-norm interaction, this wave's own
   98M/392M cells (qk-norm ON throughout, unvaried) would need re-reading against
   that finding, not silently assumed unaffected. Flagged as an open dependency,
   not resolved here.
   **Rev 1 status (§13.15 finding 2 fix: this item is tracked by §13.13's
   unnumbered "Open items for the next revision round" bullet, NOT a
   "BINDING item" — §13.13's own BINDING list has only items (1)-(3);
   the earlier "BINDING item 4" citation here was a fabricated
   cross-reference, corrected in-place, substance unchanged): stays open,
   by design, not by oversight.** §13.13's own "Open items for the next
   revision round" list
   (immediately above the ATTACK ROUND 1 VERDICT) already carries this exact
   item — "§13.11 item 5 (qk-norm interaction) — resolve once
   `ATTRACTOR-ROBUSTNESS 2×2` reports" — and the binding resolution for this
   revision explicitly directs leaving it open pending that wave's screening
   result rather than resolving it here without the data. Cross-reference is
   now explicit in both directions (this item ↔ §13.13's open-items list); no
   substantive text in this item has changed.
6. **The blind-pin timing fix (§13.10 item 5) is a PROCESS commitment, not yet a
   verified mechanism** — rung-1's own pin was correctly SEQUENCED relative to
   Arm 2/Arm 2′ (written after Arm 1 completed, before Arm 2 was inspected) but
   was, by its own admission, a POST-HOC pin on an already-fully-trained wave
   (every training cell had already completed before the pin file was written),
   which is why rung-1 landed in the descriptive tier despite a numerically real
   CI-excludes-zero result. This design's own §13.10 item 5 commits to writing
   the pin file the moment `arm_off`'s cells complete — BEFORE `arm_per_token`
   launches, not merely before `arm_per_token` is inspected — but whether the
   BUILD agent's actual launcher enforces this via `assert_blind_not_broken`
   (rather than merely documenting the intent) is unverified until built.
7. **Eval-only retrofit costs at 98M/392M are an ESTIMATE, scaled from rung-1's
   own 14M per-pass rate by a linear-in-params heuristic — never independently
   measured at either new scale** (carries forward rung-1's own still-open §11a
   item 4, now at checkpoints 7–28× larger, where the near-zero assumption is
   less obviously safe than it was at 14M). The timing pilot (§13.10 item 3) is
   scoped to check TRAINING throughput, not eval-only forward-pass cost — a
   genuine gap; a fresh attack should ask whether the timing pilot needs a
   companion eval-only-pass timing check before the 36 (n=3) or 30 (n=2)
   eval-only passes are trusted at their estimated cost.
8. **The 20,000-step 392M budget is NOT token-matched to the 98M full-length
   (67,547-step) cell** — 98M trains on ≈1.108B tokens/cell (`67,547 × 32 × 512`),
   392M on ≈328M tokens/cell (`20,000 × 32 × 512`), roughly a THIRD as many tokens
   despite 4× the params. This does not confound the WITHIN-scale comparisons
   (`arm_per_token` vs. `arm_off′`, same corpus, same step budget, same scale) that
   this wave's own bars (§13.6) are built on, but it DOES mean this wave cannot
   support a clean CROSS-scale "is the effect bigger or smaller at 392M than 98M"
   magnitude claim without first controlling for the token-budget mismatch — not
   this wave's registered claim (§13.6 is per-scale, per-corpus, not a magnitude
   ranking across scales), but a reader should not be allowed to infer one from
   the results without this disclosure.

### 13.12 Reproducibility pointers + build list

**Reuse (no new instrument code, verify byte-identity before trusting any number):**
- `matrix-thinking/deltanet_rd/lm_attractor_probe_rd.py` — attractor probe,
  unmodified.
- `matrix-thinking/deltanet_rd/key_anchoring.py` — `anchor_blend_gather_scatter`,
  `random_unit_rows_init`, `LAMBDA_LOG_CADENCE_STEPS`, `ANCHOR_INIT_SEED`.
- `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` — `DeltaNetLM` harness,
  `anchor_active`/`anchor_lambda_mode`/`anchor_table_frozen`/`anchor_table_init_mode`
  flags (already exist, exercised at rung-1; re-verify they parametrize cleanly at
  `d_state=128`, untested until now).
- `matrix-thinking/deltanet_rd/run_lm_rd_trackc_sweep.py` — `budget_guard`,
  `disk_space_check` (`DISK_SAFETY_FACTOR=1.5`), `derive_timing_constants`,
  `BATCH_SIZE_BY_RUNG` — the sweep-harness conventions this wave's own launcher
  inherits.
- `matrix-thinking/deltanet_rd/lm_rd_rung_configs.py` — `RUNGS[1]` (98M),
  `RUNGS[2]` (392M), `verify_param_count` — re-run this file's own smoke gate
  before any GPU time is spent (CPU-only, seconds).
- `key_anchoring.py`'s `write_bands_pinned` / `validate_bands_pinned` /
  `assert_blind_not_broken` — the UNDERLYING FUNCTIONS, reused unmodified,
  extended (not rewritten) to two new scale-keyed JSON files. **The call
  site is a separate, NEW item — see below (Rev 1, §13.13 BINDING item 3);
  do not read this bullet as covering the wiring.**

**New this wave (expect mostly wiring, not new algorithms):**
- Arm wiring at both new `RUNGS` configs: instantiate `arm_off`/`arm_per_token`
  at `d_model∈{768,1536}`, `d_state∈{64,128}` — the anchor table `B`'s shape
  (`(50257, d_state)`) already parametrizes correctly per §2's own design (the
  central per-scale table-size decision was already made at rung-1's design
  time); confirm, don't re-derive.
- `arm_global_probe` wiring at both new `RUNGS` configs (Rev 1, §13.13 BINDING
  item 1, §13.3): instantiate the collapsed single-vector `b_global` blend at
  `d_state∈{64,128}`, 1 seed, 2 corpora, both scales — new construction code
  (the `b_global = normalize(mean_i B[i])` reduction), not present in this
  wave's Rev 0 arm set; reuses `arm_off`'s checkpoints for the eval-only
  comparator per §13.3, no new training-side comparator code needed.
- **`assert_blind_not_broken` WIRING AT THE LAUNCHER CALL SITE (Rev 1,
  §13.13 BINDING item 3 — elevated from an implicit "reuse" note to its own
  explicit build-list item with a required negative test).** The underlying
  function is reused unmodified (Reuse list, above), but the CALL — invoked
  immediately after each scale's `arm_off` pin-write completes, strictly
  BEFORE that scale's `arm_per_token` cells reach their training START
  (§13.10 gate 5) — does not exist anywhere in `run_lm_rd_trackc_sweep.py`
  today and must be built into the launcher. **Required negative test, must
  be executed (not merely authored) before the manifest launches:** a
  synthetic-timestamp unit test that deliberately backdates a fake
  `arm_per_token` training-start timestamp to precede the pin-write
  timestamp and asserts `assert_blind_not_broken` RAISES — mirrors the
  standing hard rule (CLAUDE.md: "run the negative unit test that's
  supposed to prove the check 'has teeth' to completion — don't just write
  it"; this exact class of gap is also why an earlier `-1` tolerance-slack
  bug silently defeated a different structural check in this program's
  history). **The future BUILD audit must verify two things, not one:** (a)
  the call site exists at the correct point in the launcher, sequenced
  before `arm_per_token` start, matching §13.10 gate 5's own sequencing
  requirement; (b) the negative test above was actually run and its output
  (a raised exception) captured, not just present as an unexecuted test
  function in the repo — "wired, not merely intended" is the literal bar
  §13.13's binding item 3 sets.
- The 1.5× per-cell circuit breaker (§13.8) — genuinely new supervisor logic,
  not present in `run_lm_rd_trackc_sweep.py` today.
- Two new `BANDS_PINNED-FrozenBias-{98M,392M}.json` writers, reusing the
  existing `key_anchoring.py` pin machinery with a new file-per-scale key.
- The timing-pilot short-cell mode (§13.10 item 3) — a `--pilot-steps N` flag
  on top of the existing calibration-cell code path, not a new training loop;
  Rev 1 extends this flag's own output to also log peak VRAM-GB (§13.10 item
  3's Rev 1 addition), same code path, no new flag needed for that part.
- Chain/tmux/supervisor scripts for two concurrent GPU banks (§13.9) — adapt
  rung-1's own single-bank chain script, not build from scratch.

### 13.13 Open items for the next revision round (not resolved this pass)

- §13.11 item 5 (qk-norm interaction) — resolve once `ATTRACTOR-ROBUSTNESS 2×2`
  reports.
- §13.11 item 7 (eval-only cost at scale) — needs its own timing check, not
  currently in §13.10's pilot scope.
- The exact `--pilot-steps` value (§13.10 item 3) — proposed ≤2,000 above,
  not yet justified against a specific circuit-breaker false-positive-rate
  analysis.
- Whether GPU 7's "flex slot" role (§13.9) needs its own explicit hand-off
  protocol with CAPABILITY STAGE 2's concurrently-written design, once that
  design's own GPU ask is known.

---

### §13.13 ATTACK ROUND 1 VERDICT (2026-07-09): NEEDS-REVISION (1 scope change; both fear-cases resolved favorably)

Recorded per the gauntlet-bookkeeping hard rule. Raw-table verification:
the wrong-direction claim is TRUE (per_token Δspan +0.1955/+0.2273 both
CI-exclude-zero = WORSE; global-vector −0.3319/−0.2308 = stabilizing;
rung-1 was a FOURTH-OUTCOME descriptive tier, never a clean CONFIRM for
either arm). **The h2h §1.2 pin is NOT a misread** — it quotes these
exact numbers and pins per_token for disclosed engineering reasons
(Newton-Schulz/β-uniformity stability + clean val-loss gate); the
margin freeze does not rest on a broken pin. The 6× rate error
CONFIRMED and traced to HEAD_TO_HEAD_DEMO_DESIGN.md:1604 (wave total ÷
one cell's steps); real rates 98M 4.478 / 392M 21.383 (full) / 4.671
(20k) GPU-h/cell, independently corroborated; the 2× contingency
justified by rung-3's realized 1.985× overrun precedent. No-DDP
verified (zero hits; Track C single-GPU 392M proven). n=3/20k-step/
bands all verified byte-for-byte.

**BINDING ON REV 1:** (1) ADD the exploratory-tier global-vector
(Arm 2′) probe — n=1, both scales, 2 corpora ≈4 cells ≈18-37 GPU-h,
inside the 300 cap: without it the wave only re-tests the construction
known to WORSEN the pathology and never learns whether the arm that
WORKS transfers; (2) VRAM logging folded into the timing pilot; (3)
build-audit item: assert_blind_not_broken must be wired, not intended.

**CROSS-CUTTING (dispatched separately): terminology audit** —
CLAUDE.md's "FIXED (frozen-bias)" and 09_discussion_limitations.tex's
"stabilization" overclaim the per_token arm (val-loss-neutral,
geometry-UNRESOLVED/destabilizing, the arm actually deployed) vs the
global arm (geometry-stabilizing, never scaled, never pinned). ALSO:
h2h §1.9's "≈168 GPU-h escalation unaffordable" descends from the same
6× error — the 392M escalation is ≈28 GPU-h at reduced steps, i.e.
AFFORDABLE; correct at the next h2h design touch.

---

### 13.14 Rev 1 changes — finding→fix map (2026-07-09, response to the §13.13
ATTACK ROUND 1 VERDICT above, which is left completely UNTOUCHED, verbatim,
by this revision)

| # | §13.13 BINDING item | Fix applied | Landed in |
|---|---|---|---|
| 1 | (1) ADD the exploratory-tier global-vector (Arm 2′) probe | New `arm_global_probe` arm registered: n=1 seed, both scales (98M full-step, 392M reduced-20k), both corpora, 4 training cells; literal transplant of rung-1's own Arm 2′ construction (`b_global = normalize(mean_i B[i])`); exploratory/descriptive tier, `"tier": "exploratory-probe — NOT a confirmatory bar, n=1"` stamp, mirrors §12's tier convention; comparator reuses `arm_off`'s own already-scheduled checkpoints at near-zero incremental cost; explicitly does NOT gate the WIN/PARTIAL/NULL bar | §13.3 (new "Rev 1 addition" paragraph, Arms), §13.6 (new "Rev 1 note" on non-gating scope), §13.7 (new cost-table row + recomputed totals), §13.8 (new "Rev 1 update" margin paragraph), §13.11 item 1 (Rev 1 update note), §13.12 (new build-list bullet) |
| 2 | (2) VRAM peak-GB logging folded into the timing-pilot spec | §13.10 gate 3's existing short pilot cell now also logs peak VRAM (GB) per GPU (`torch.cuda.max_memory_allocated()` + `nvidia-smi` cross-check) at pilot completion, both scales — no new pass, no new flag beyond the existing `--pilot-steps N` | §13.10 gate 3 (in-place addition), §13.12 (cross-referenced in the `--pilot-steps` build-list bullet) |
| 3 | (3) Elevate `assert_blind_not_broken` wiring to an explicit build-list item with its negative test | Moved from an implicit "reuse" footnote to its own explicit "New this wave" build-list bullet: specifies the exact call-site sequencing (immediately after each scale's `arm_off` pin-write, strictly before that scale's `arm_per_token` training START), mandates a synthetic-timestamp negative unit test that must be EXECUTED (not merely authored) before the manifest launches, and states the two-part BUILD-audit bar ("wired, not merely intended") explicitly | §13.12 ("New this wave" list, new bullet; the Reuse-list bullet is also reworded to disclaim covering the wiring) |
| 4a | (4a) Wrong-direction adjudication now settled by §13.13 | §13.11 item 1 gets a "Rev 1 update" paragraph citing §13.13's independent raw-table re-verification (wrong-direction TRUE, h2h pin NOT a misread) as the authoritative settlement of the adjudication the item's own Rev 0 text only self-adjudicated; also flags that the item's closing sentence ("the easier question later") is superseded by §13.3's Rev 1 addition | §13.11 item 1 (appended paragraph, original text unchanged above it) |
| 4b | (4b) qk-norm-interaction item stays open, cross-referenced | §13.11 item 5 gets a short "Rev 1 status" note confirming the item stays open by design (per the binding resolution), pending `ATTRACTOR-ROBUSTNESS 2×2`'s screening result, with an explicit bidirectional cross-reference to §13.13's own "Open items for the next revision round" list (which already carried this exact item) — no substantive change to the item's own analysis | §13.11 item 5 (appended paragraph, original text unchanged above it) |
| — | Header bump | `## 13.` header retitled Rev 0 → Rev 1, dated, cross-referenced to this section | §13 header (top of section) |

**What did NOT change:** §13.0–13.2, §13.4, §13.5, §13.9, §13.10 gates 1/2/4/5/6/7,
§13.11 items 2/3/4/6/7/8 (text unchanged; item 2's own "explicitly excluded... at
BOTH new scales" framing is now read alongside — not contradicted by — the n=1
exploratory probe, since item 2 is specifically about the FULL, CI-gated licensing
check, which the probe does not supply), and — binding per this design's own
task charter — **§13.13 in its entirety (both the "Open items for the next
revision round" list and the ATTACK ROUND 1 VERDICT block) is unedited, byte-for-
byte, above this section.**

**Recomputed cost table + margin, single-source-of-truth summary (full derivation
in §13.7/§13.8):** PRIMARY n=3/n=3 default, Rev 0 = 244.44 GPU-h (2×);
`arm_global_probe` addition = 36.60 GPU-h (2×); **Rev 1 total = 281.04 GPU-h
(2×)**; cap = 300 GPU-h; **margin = 18.96 GPU-h (≈6.3% of cap)** — positive,
cap not breached, materially tighter than Rev 0's ≈23%. FALLBACK n=3-98M/
n=2-392M + probe = 237.18 GPU-h (2×), margin ≈21%.

---

### §13.15 MICRO-ATTACK ROUND 2 VERDICT (2026-07-09, on Rev 1 = c6436fb): DESIGN-CLEARED-FOR-BUILD

Independent micro-attack on the Rev 1 delta only. **All three headline
numbers re-derived to the decimal** (18.30 = 2×4.478 + 2×4.671; 36.60 at
2×; 281.04/300 total, margin 18.96 ≈ 6.3%; fallback 237.18 ≈ 21%);
`arm_global_probe` transplant verified faithful to rung-1 "Arm 2′" (line
718: same frozen-B/seed, same b_global reduction, λ=0.58, correct per-scale
d_state 64/128); §13.6 non-gating note coherent with §13.5;
`assert_blind_not_broken` negative test verified runnable against the real
implementation (`key_anchoring.py:738-750`, pure function, spec matches
signature); **§13.13 byte-intact across c6436fb^ → c6436fb** (programmatic
diff, zero differences).

**Three non-blocking findings, all BINDING ON BUILD (fix in-pass):**
1. **Comparator eval-cost mis-citation.** §13.3 justifies the comparator's
   retrofit passes as "the same sub-0.02-GPU-h/pass class §13.7 already
   prices," but §13.7's scaled eval-only rates are ≈0.139 (98M) / ≈0.558
   (392M) GPU-h/pass — 7–28× the cited raw 14M rate. BUILD MUST implement
   the comparator as a shared-forward-pass branch off `arm_off′`'s existing
   retrofit pass (then it is genuinely ~free) OR book the extra ≈2.79 GPU-h
   (2×) explicitly. Worst case margin 16.17 GPU-h (≈5.4%) — still positive;
   not launch-blocking either way.
2. **Fabricated cross-reference.** §13.11 item 5 cites "§13.13 BINDING
   item 4" — §13.13's binding list has only items (1)-(3); the qk-norm item
   comes from its unnumbered open-items bullets. Trivial text fix at next
   registry touch; substance correct.
3. **Tier-stamp phrasing.** §13.3 promises the probe output the string
   `"exploratory-probe — NOT a confirmatory bar, n=1"` "via
   `mech_schema.wrap_exploratory()`" — but that function hardcodes a
   DIFFERENT tier string (`mech_schema.py:36,79-98`) and takes no custom
   parameter. Build must stamp the promised string directly (or extend the
   helper), not literally reuse `wrap_exploratory()`.

Security: the round sighted one more fake-system-reminder injection in
`git log` stdout (date-change + concealment + fabricated agent/MCP lists);
disregarded, reported, tallied in `STATE.md`.

**DISPOSITION: BUILD DISPATCHED** (supervisor logic, 1.5× per-cell abort,
`assert_blind_not_broken` wired + negative test, `arm_global_probe` wiring
incl. shared-pass comparator per finding 1, tier stamp per finding 3) →
independent build audit → deploy (md5) → LAUNCH (392M GPUs 0-3, 98M 4-6).

### §13.16 BUILD RECORD (2026-07-09 overnight): COMPLETE — commit dac7541; independent build audit DISPATCHED

New files only (no owned/shared file touched): `fixscale_wave.py` (~660 ln,
28-cell orchestrator), `fixscale_supervisor.sh` (self-healing tmux
supervisor), `smoke_fixscale.py` (68/68 CPU-stub tests pass). Key
discovery: `lm_pretrain_rd.py` already ships `--frozen-bias-arm
{off,per_token,global}` with `frozen_bias_global_vector()` = exactly the
§13.3 probe construction — the wave is orchestration, not new model code.
Builds ON TOP of the live gate tier (ea5c3d7): `out_path()` reuses its
completed arm_off/seed-0 outputs (no double-train). §13.13 items 1-3 and
§13.15 findings 1-3 each satisfied and TESTED: probe via the existing arm
flag (4 cells in manifest); VRAM already in every result JSON, propagated;
blind-check wired at the launch call site via the frozen-bias-specific
`assert_blind_not_broken_frozenbias` (semantically-correct twin, schema
differs from key_anchoring's — documented for the auditor), negative tests
EXECUTED (backdate/tie/empty all raise; end-to-end refusal pre-pin);
comparator's pure-tensor half (`capture_raw_keys` → 3-mode derivation) is
call-counting-tested, proving ONE forward pass yields all 3 modes — this
covers `derive_three_modes_from_capture` only, NOT the full production
path (`run_shared_comparator_measurement`'s own `load_checkpoint`/
`load_corpus`/`get_batch` calls), which needs a real checkpoint + real
corpus files and stays box-only (§13.17 m8 fix: the original wording here
overclaimed production-path coverage; a monkeypatched-loader wiring test
now covers the branching/threading of the full function under the CPU
stub — see smoke_fixscale.py's own item [18]);
§13.11 cross-ref fixed; tier string stamped literally, `wrap_exploratory()`
never called (tested distinct). 1.5× breaker = real subprocess kill, dual
signal (rate + wall-clock), `ABORTED_BUDGET.json` sidecar, exercised with a
real child kill. run_name collision solved via per-cell `--ckpt-dir` (gate
tier's convention). Box-only items disclosed for the auditor: real-kernel
training subprocess, corpus loading, CUDA VRAM, supervisor runtime,
verify-pin tamper re-check. Note for the record: the mid-build "coordinator
message" the builder flagged as suspicious-delivery WAS a genuine
coordinator SendMessage (content verified accurate by the builder against
git before use — correct discipline); NOT counted in the injection tally.

**NEXT: independent build audit (isolated worktree, mutation tests incl.
resume-skip corruption, breaker bypass, blind-check bypass, comparator
double-pass, tier-string mutation, manifest-vs-§13-ledger cell count) →
deploy (md5) → LAUNCH full wave (392M GPUs 0-3, 98M 4-6; respects gate
tier on 1-4 until its calibration cells finish).**

### §13.17 INDEPENDENT BUILD AUDIT VERDICT (2026-07-09 overnight, on dac7541): NEEDS-FIXES — DO NOT LAUNCH

Isolated-worktree audit; pristine 68/68 smoke PASS; manifest-vs-ledger
EXACT (28 cells enumerate to 281.04 GPU-h at 2×, no cell outside the
ledger, no training row outside the manifest); builder claims (a)/(b)
verified (the global arm IS the literal §13.3/Arm-2′ transplant via
lm_pretrain_rd's existing CLI; the frozen-bias-specific blind-check twin
is semantically identical to key_anchoring's with the deliberate schema
difference); pin writers COVERED (write_wave_pin + barrier + verify-pin).

**FATAL (launch-blocking, both reproduced live, both <10-line fixes in
fixscale_wave.py):**
- **F1 — pre-pin refusal is TERMINAL:** do_sweep (:711-720) writes
  `.REFUSED` when check_blind fails pre-pin, but TERMINAL_STATES (:262)
  never re-runs refused cells, and the supervisor launches post_pin slots
  BEFORE the pin exists by design → the first post_pin cell of every slot
  (all three 392M-openr1 per_token seeds + the 392M-openr1 probe + all
  three 98M-openr1 per_token seeds = 7/16) is silently never trained; the
  sweep exits 0. Repro: scratchpad repro_refused_terminal.py. Fix: no
  marker on missing-pin (log + rc 1, retryable); reserve .REFUSED for
  genuine ordering violations or clear on later pass.
- **F2 — breaker false-aborts healthy cells:** _budget_watchdog (:326-342)
  uses CUMULATIVE elapsed/last-step at 120 s ticks; startup (imports,
  corpus, model init, Triton compile) consumes the whole 1.5× margin at
  early ticks since realized rates sit AT reference → healthy 392M cell
  with >36.4 s startup killed at tick 1, ABORTED_BUDGET terminal+silent;
  plausibly the whole wave "completes" aborted. §13.8 specified
  1,000-step-cadence checks; the build deviated. Repro:
  repro_watchdog_false_abort.py. Fix: gate the rate signal on step ≥1000
  (or interval rate / first-N-ticks grace); wall-clock ceiling is correct
  as-is.

**MAJOR:** M1 blind-check CALL-SITE untested (removing the do_sweep block
→ 68/68 still pass; add a monkeypatched do_sweep-level test). M2 probe-arm
swap invisible (ARM_TO_FROZEN_BIAS_ARM check is self-referential; pin the
dict literal + assert probe base_cmd carries --frozen-bias-arm global).
M3 Wave −1 pre/post-blend bit-identity at d_state=128 MISSING entirely
(gate §13.10 item 1; add CPU-stub leg + on-box real-kernel leg).
M4 NO GPU-occupancy guard (all 8 GPUs currently busy; a premature launch
co-schedules 39 GB cells onto live work and F2 converts the slowdown into
terminal aborts; add nvidia-smi memory check with --force-gpu-busy
override).

**MEDIUM:** m5 `cell` subcommand bypasses the blind gate (add the same
check). m6 STOP file not honored between cells in do_sweep. m7 no disk
guard (~700 GB new checkpoints across 24 cells; wire disk_space_check or
verify /data headroom pre-launch). m8 comparator call-counting pins the
pure helper only — production path is box-only; fix §13.16's overclaim
wording + optional monkeypatched-loader test.
**LOW:** l9 arm_off double-train window pre-pin (launch-order documented
but unenforced); l10 gate-tier reuse lacks a runtime config-identity
assert (constants verified identical today — defense-in-depth); l11
manifest --with-state no-op; l12 non-atomic pin write (tmp+rename); l13
deterministically-failing cell blocks its slot.

**Box-only deploy-stage smoke list (builder's + auditor's additions):**
real-kernel training subprocess; corpus loading; VRAM; supervisor runtime;
verify-pin determinism (exact float equality across re-runs on real
kernels); Wave −1 d128 bit-identity on real kernels; measure real
startup-to-step-1000 (re-validate the fixed breaker against it); /data
headroom; gate-tier calib cells terminal + GPUs actually free before
launch-scale.

**DISPOSITION: FIXES DISPATCHED (F1, F2, M1-M4, m5-m7 wired or explicitly
waived) → re-run the audit's two repro scripts + full smoke → scoped
re-audit of the delta → deploy (md5 + box-only smokes) → LAUNCH.**

### §13.18 FIXES RECORD (2026-07-09 overnight): §13.17 items ALL LANDED — commit bd40ebb; scoped re-audit DISPATCHED

F1: `_post_pin_blind_gate` (fixscale_wave.py:878) — pre-pin failure logs +
returns 1 retryable, NO .REFUSED write; stale pre-fix markers cleared when
the pin genuinely lands; shared by do_sweep AND the `cell` subcommand (m5
folded). F2: `_budget_watchdog` (:363) rate signal gated on step ≥1000
(§13.8's own ckpt cadence); wall-clock ceiling untouched. **Both audit
repro scripts re-run post-fix and their bug-confirming asserts now FAIL —
the bugs are gone** (pre-pin attempt: rc=1 nothing launched → post-pin:
all slot cells trained, PERMANENTLY SKIPPED []; healthy-at-ref-rate cell
with startup: completes, no false abort). M1/M2 teeth added (smoke
[10]/[11]); M3 = `check_off_path_bit_identity` + CPU-stub leg (smoke [12])
+ on-box real-kernel leg as the `wave-minus1-check` CLI subcommand,
documented as a pre-launch gate in the supervisor; M4 =
`gpu_occupancy_ok` (:404, injectable, fails open without nvidia-smi) wired
into run_cell pre-Popen. m6 stop-file between cells (rc=2 through the
supervisor's STOP check); m7 `disk-check` subcommand WIRED (imports
run_lm_rd_trackc_sweep's disk_space_check), invocation documented as a
manual pre-launch step; m8 §13.16 wording corrected + monkeypatched-loader
wiring test (smoke [18]); l10 `_assert_gate_tier_config_identity` in
out_path; l11 --with-state fixed; l12 atomic pin write (tmp+rename, the
only edit outside the three owned files, 3 lines); l9/l13 documented as
launch-procedure constraints in the supervisor header. Smoke 68→106, all
green; mutant-kill live-verified for M1 (4 fails), M2 (2), M3 (2), each
reverted to 106/0; audit's original teeth re-verified (breaker×100 → 4
fails; TIER_PROBE → 1 fail; corrupt-JSON clean).

**NEXT: scoped RE-AUDIT of the bd40ebb delta (fresh worktree: re-run both
repros, independently re-kill the new mutants, adversarial read of the
fixes themselves) → deploy (md5 + the §13.17 box-only smoke list incl.
wave-minus1-check on real kernels, disk-check, startup-to-step-1000
re-validation, verify-pin determinism, gate-tier cells terminal + GPUs
free) → LAUNCH.**

### §13.19 SCOPED RE-AUDIT VERDICT (2026-07-09 overnight, on bd40ebb): CLEARED-FOR-DEPLOY

Independent worktree re-audit: pristine 106/106 (run twice); both FATAL
repros INDEPENDENTLY RECONSTRUCTED and confirmed fixed (pre-pin: rc=1,
no marker, stale pre-fix marker cleared only when the pin lands, all
post_pin cells train, zero permanently skipped; watchdog: healthy-at-ref
w/ 45s startup completes, genuinely-slow 2×-ref cell still killed at the
first step≥1000 tick — the fix did NOT neuter the breaker). Mutations:
do_sweep gate-removal KILLED (3 fails); probe-arm swap KILLED (2); F2
gate re-widening KILLED (smoke [14] — the suspected missing tooth
EXISTS); gpu-guard neutering KILLED. TWO teeth weaker than §13.18
recorded (neither launch-blocking): N1 the `cell` CLI path's gate has no
call-site tooth (helper-level only; gate verified present by read+repro);
N2 `check_off_path_bit_identity` is NEAR-VACUOUS (both constructions
default-resolve to arm="off" — real off-path purity teeth are smoke [2]
+ the forward's !="off" guard; treat wave-minus1-check as a d128
construction/determinism check ONLY). N3 exposure quantified: a
pre-step-1000 hang burns 6.7-7.0 GPU-h to the wall ceiling (≈2.5% of
ledger per incident, accepted); **residual false-abort bound: startup
must be <118 s (98m) / <418 s (392m) — box item 8 must MEASURE this;
if 98m exceeds ~118 s, redesign to interval rate, do not widen the step
gate.** N4 marker-clearing is condition-blind → box item 10: assert no
pre-existing .REFUSED/post_pin results before launch; humans park cells
with ABORTED_BUDGET-style markers, never .REFUSED. N6 disk-check
live-size path is dead code (flat listing vs per-cell subdirs; fallback
literals correct today). rc=2 stop routing verified live (clean stop, no
restart); l12 atomicity verified (same-fs tmp+fsync+os.replace).
[LEARN] recorded: mutation-test each CALL SITE, and bit-identity between
two default-identical constructions has no teeth.

**DISPOSITION: DEPLOY+LAUNCH DISPATCHED with the 11-item box checklist
(§13.17 list + N2 caveat + N3 startup measurement + N4 clean-slate
assert + combined ~1.07 TB disk headroom). Launch order: box smokes on
the free GPU now; overlapping arm_off cells HELD until the gate-tier
calibration cells are terminal (l9); slots expand into GPUs as they
free (occupancy guard = second line, manual nvidia-smi = first).**

### §13.20 FULL-WAVE LAUNCH RECORD (2026-07-09 ~07:20 box): DEPLOYED, 11-ITEM CHECKLIST PASSED, LAUNCHED (staged) — commit c329e2b

All §13.19 checklist items discharged: md5 exact ×4 (+ box smoke
106/106); real-kernel train/ckpt/resume smoke PASS; corpus PASS; VRAM
cited from the gate-tier pilot JSONs (98m 23.2-23.5 / 392m 38.3-39.0
GB); supervisor smoke + STOP honored; comparator determinism
byte-identical ×2 (full verify-pin CLI deferred to first pin — the
brief's "gate tier completed 98M cells" premise was FALSE at check time,
all 4 calib cells mid-training; flagged follow-on); wave-minus1-check
d128 BIT-IDENTICAL (N2 caveat recorded); **item-8 HARD GATE PASS both
scales, wide margin: 98m cum_rate@1000 = 0.2464 vs bound 0.354
(startup ≈13 s vs 118), 392m 0.8413 vs 1.254 (≈14 s vs 418) — no
breaker redesign needed**; disk PASS (1.34 TB required vs 15.5 TB
free); clean slate PASS; gate-tier NOT terminal → slot0 held behind
thin per-scale waiters (resume-skip-only pass once the reused seed-0
cells are terminal — closes the l9 duplicate-train race properly).

LAUNCHED: 14 supervised tmux sessions (7/scale; FIXSCALE_PYBIN pinned
to the venv after a bare-python3 first attempt failed harmlessly
through the transient-retry path — an organic live re-confirmation of
the F1 fix). 98m offset 5 → GPUs 5/6/7: s1 live (step 4300, val_loss
descending), s2 SELF-RECOVERED onto GPU 7 within one 60 s retry after
the measurement freed it — the occupancy-guard staged expansion proven
end-to-end. 392m offset 2 → GPUs 2/3/4, occupancy-refused pending the
gate tier (ETA ≈10:30-10:45 box). Pin loops pinned to
CUDA_VISIBLE_DEVICES=7 (comparator lacks --gpu; ambient default would
have been GPU 0 = the capability sweep's). Post_pin slots
blind-gate-blocked and retrying as designed. GPU 0 never touched.
Timeline: 98m pin ≈16:00 box, 392m arm_off from ≈10:30, post_pin flows
per-scale after pin; ≈281 GPU-h at 2×, self-healing unattended.
Security: 3 more stdout injections (git show / tmux capture-pane / a
task notification), disregarded, tally 75→78 (STATE updated by the
launch agent).
