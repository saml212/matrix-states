# ICLR 2027 Narrative Architecture — DeltaNet Rank Recruitment vs. Exactness

**Status: DRAFT — round 7 adds a small, distinct side-chapter (frozen-bias
LM transplant test, rung-1 complete, honestly closed as a
discovery-shaped negative-plus-surprise, not a confirm — see round 7,
below); round 6, capacity cliff LOCATED (a measurement upgrade on
top of round 5's ARC COMPLETE key-anchoring program), is otherwise
unchanged.** Round 6 absorbs the anchoring program's actual, final outcome — not the
Outcome-C mechanism null round 4 left as the ending, but the two waves that
ran *after* it: candidate (e)'s frozen-random-table ablation
(CONFIRMED-BY-ABLATION: constancy suffices) and the K=48 capacity-curve
extension (bar missed 0/3, the curve completes as a cliff). The key-anchoring
program is now formally CLOSED end to end (`KEY_ANCHORING_DESIGN.md`
§10.14, §11.12; `STATE.md`, 2026-07-07). This is the biggest structural
change to the paper's own story since round 3's 2×2 design: what was an
open mechanism question is now a closed, three-tier finding with its own
headline figure. Grounded against `DELTANET_RD_EXACTNESS_DESIGN.md` (§1–8,
§14, §15, §16 incl. the §16.7 dated correction), `DELTANET_REALDATA_DESIGN.md`
(§14–19), `DELTANET_CAUSAL_RANK_DESIGN.md` (§1, §2, §4, §12),
`KEY_ANCHORING_DESIGN.md` (§1–§3, §9/§9.6/§9.7, §10/§10.13/§10.14, §11/§11.12
— the complete arc from mechanism-tier confirmation through the constancy
ablation to the capacity cliff), `KEYANCHOR_REV6_ATTACK.md`,
`KEYANCHOR_REV7_ATTACK.md`, `SCALE_TRANSFER_DESIGN.md` (§2 claim tiers, §3.8,
§4, §5.9, §5.10 + Addendum, §6.8), `TRACKB_REDESIGN.md` §14, `STAGE_G_DESIGN.md`
§15, `STATE.md`'s 2026-07-07 picture, and `EXPERIMENT_LOG.md` (2026-07-01 →
2026-07-07, including every key-anchoring verdict entry). Every number below
is quoted from those sources, several re-verified directly against the
archived run JSONs during this revision, not recomputed or estimated.

Round-4 ending (restated): the anchoring wave had run twice (initial +
confirmatory) and landed on Outcome C — a real, twice-reproduced K=32 h=4
behavioral gain (0.4105 → 0.556–0.665) whose proposed entity-alignment
mechanism the wave's own per-entity engagement instrument could not confirm
(<14% engagement in every leg), with a same-day rescue attempt independently
attacked and rejected. The paper's honest ending, at round 4, was "real
effect, unconfirmed cause, mechanism question open for a future wave." The
scale-transfer ladder had closed its mix-axis confound and become a clean
pure-scale, 3-point claim (0.248→0.344→0.389), with a fourth rung launched
and not yet reported.

**New this round, absorbed from the 2026-07-06/07 cascade — the mechanism
question that round 4 left open is now closed, and the fix's regime is now
bounded by a completed capacity curve:**

(1) **A mechanism-tier confirmation wave (2026-07-06) independently
replicated the behavioral gain on 6 genuinely fresh seeds** (not the
same-seed reproduction round 4's confirm wave used) across two architecture
variants — K=32 h4 0.6141–0.7141 — while reconfirming Outcome C at a
strictly stronger evidentiary bar, immune to every specific objection that
had killed the round-4 rescue attempt (unit-anchor-norm assumption measured
directly at 1.06–1.16, not assumed at 1.0; a formal dip-test rules out a
masked bimodal split; an effect-size floor rules out power-manufactured
engagement). This wave also registered a falsifiable follow-on prediction:
**if cross-episode stability comes from the mere presence of an
episode-constant term in the blend (not from the anchor table's learned
content), a frozen, never-trained random table at the same blend weight
should match candidate (d)'s gains.**

(2) **That prediction was tested and held.** Candidate (e) (2026-07-07): two
frozen, never-trained anchor tables — one pure-random-init, one
frame-potential-init — both at fixed λ=0.58, **both match or slightly exceed
the learned table's own K=32 mean** (frozen arms 0.7274/0.7413 vs. learned
(d)'s 0.6669; no frozen seed falls below (d)'s own minimum). The r_e
negative control passes at the strongest null in the design's history
(median r_e negative, −0.13 to −0.24, for the pure-random arm, which cannot
mechanically align to anything). **This is CONFIRMED-BY-ABLATION: constancy
alone suffices.** The "learned anchoring" framing is superseded — the
deployable fix is a frozen random key-bias at matched blend weight, no
training loop required — while the primary entity-alignment hypothesis
stays Outcome C throughout, unchanged.

(3) **The capacity curve was extended to K=48 (K/d=0.75) and completed as a
cliff, not a smooth decline (2026-07-07).** Candidate (d) misses its
registered bar 0/3 seeds (mean 0.0215 vs. bar 0.0244), fully admissible,
with both of the design's own pre-registered explanatory channels
(drift-space stabilization, closing ~81% of the gap to ceiling;
value-Gram-relief, 0.626× the reference's deviation) measurably active but
insufficient. The three-point curve — ~1.00 (K=16) → ~0.65 (K=32) → ~0.02
(K=48) — falls off a cliff between half and three-quarter capacity while the
theoretical λ=1 ceiling declines only gently (0.97→0.94→0.90) over the same
range. Binding survives everywhere (h=1 guard 1.0000, every cell, every K,
this design's entire history); composition is what collapses.

(4) **The program is formally complete.** Five waves plus a rejected rescore
attempt, ≈55.83/80 GPU-h against its own budget cap, no further waves
scheduled. The paper's own honesty discipline — an adversarial gauntlet
catching its own attempted rescue of a null result before it entered the
record — is now followed, one step later, by the same discipline
registering and then confirming a falsifiable account of *why* the null
mechanism still produced a real effect. The story that reaches the paper is
no longer "a real effect with an open cause" — it is "a real effect, a
rejected wrong explanation, a confirmed right explanation, and a measured
boundary of where the fix works."

Only one program-relevant readout remains outstanding: **Track C rung-3**
(1.31B params), launched on GPUs 0–1, `ALL_DONE` expected imminently. This
is the sole "still landing" item in the whole arc (§9).

**Round 6 (2026-07-06) — the capacity cliff is now LOCATED, not just
bracketed.** Round 5 closed the key-anchoring program's mechanism arc and
left the capacity curve as three points (K/d=0.25/0.5/0.75) with a cliff
known only to lie somewhere in the 0.25-wide `(0.5, 0.75)` interval between
the K=32 and K=48 anchors. This round's arc, start to finish: a
power-analysis simulation (`sim_cliff_power.py`) first showed that the
original binary sigmoid-vs-linear shape test was underpowered at 6 points
(P(ambiguous) 87–96% across plausible cliff widths) and was retracted in
favor of a parameter-estimation deliverable (`x0 = A ± B`, `w ≤ C`); the
localization wave itself then ran 4 new K's — **34, 38, 42, 46**
(K/d = 0.53125/0.59375/0.65625/0.71875) — at 3 seeds each (12 candidate-(d)
cells, 3.18 realized GPU-h against a 23.36 GPU-h worst-case ceiling), giving
a 7-point curve alongside the 3 archived anchors (K=16/32/48). A sigmoid
`h4(x) = L / (1 + exp((x − x0)/w))` fit to all 7 points by
`scipy.optimize.curve_fit`, with a 4,000-trial seed-level parametric
bootstrap CI, lands the transition at **`x0 = 0.5455`** (95% CI
**[0.5385, 0.5513]**, width **0.0127**, 0/4000 degenerate fits) and
**`w = 0.0597`** (CI [0.0557, 0.0642]) — the cliff's midpoint sits just
above K=34 (K/d=0.53125), and its characteristic width spans roughly
K≈31–39 at d=64. This narrows the localization from a 0.25-wide bracket to
a ±0.0064-wide point estimate, a ≈20× tightening, without needing to
resolve the sigmoid-vs-linear shape question the retracted test chased
(KEY_ANCHORING_DESIGN.md §12.4/§12.9). This strengthens the paper's
capacity-law section by converting a qualitative "somewhere between half
and three-quarter capacity" observation into a quantitative, CI-bounded
parameter — the kind of number a reviewer can check a mechanistic theory
against, not just a described trend. **Honest scope, stated plainly:** this
result covers the candidate-(d) arm only (the bare-geo3 reference arm was
cut from this wave's scope as a disclosed budget tradeoff, so
reference-arm behavior across K=34→46 remains unmeasured); it makes no
mechanism claim (it locates WHERE and HOW WIDE the transition is, not WHY
it occurs at this specific K/d ratio — value-crowding remains the design's
named-but-untested candidate explanation); and it is measured at a single
`d_state=64` — whether `x0 ≈ 0.55` (in K/d terms) holds at other `d` is the
program's own registered open question, not something this wave attempted.
New headline figure: `fig_cliff` (`matrix-thinking/submissions/iclr-2027/
figures/make_fig_cliff.py`), which supersedes Fig. 11's 3-point plot as the
paper's capacity-law headline (Fig. 11 itself is unchanged and can move to
the appendix as a simpler 3-point predecessor view, or be dropped, a
camera-ready layout call, not resolved here).

**Round 7 (2026-07-06) — the frozen-bias LM transplant test: a separate,
smaller side-chapter, honestly closed as a discovery-shaped
negative-plus-surprise, not the hoped-for confirm.** This round absorbs
`FROZEN_BIAS_LM_DESIGN.md`'s completed rung-1 wave — a distinct program
from the key-anchoring arc above (its own design doc, own 135 GPU-h
ceiling, own GPU allocation), testing whether the key-anchoring program's
own "constancy suffices" finding (candidate (e), round 5 above: a frozen,
never-trained anchor table matches the learned one) transplants from the
~107-entity synthetic-grammar testbed to a real from-scratch 14M-parameter
DeltaNet LM via a dense per-token frozen key bias blended into `k_raw` at
training time. **Included here because it shares this paper's exact
architecture and instrumentation lineage** (Wave C's own 14M
`d_model=256/d_state=64/n_layers=2` config, the same `span_frac`
key-Gram-deviation observable from `SCALE_TRANSFER_DESIGN.md` §5.10) and
because its own five-round design gauntlet — 2 independent adversarial
KILLs before any GPU was touched — is itself a methods contribution worth
recording alongside the paper's other examples of the same discipline
(§8's `KEYANCHOR_REV6/7_ATTACK.md` gauntlets, above).

**The design gauntlet, briefly:** round 2's own primary bar (Arm 2's
post-blend `span_frac` vs. never-blended Arm 1) was KILLED as an AUTO-PASS
ARTIFACT — dense blending toward 50,257 near-orthogonal rows pins the
measured population's `span_frac` to a narrow band regardless of baseline
structure, an artifact the design's own crossover-scan caught before any
training cell launched. Round 3 redesigned the primary around
artifact-matched controls (Arm 1′/Arm 1″: Arm 1's own checkpoint, blended
ONLY at eval time) so any surviving difference is training-mediated by
construction — **this auto-pass-artifact-and-redesign story is itself a
methods point**: an observable that looks clean under a naive
before/after comparison can be dominated by the measurement arithmetic
itself, and the fix (comparing against an identically-blended, untrained
control) generalizes to any "did training change this geometry" question
built on a fixed post-hoc transform. Round 4's own second independent
KILL found the redesigned bar's validating simulation used a synthetic
noise floor never checked against real cross-seed training variance —
real archived data showed the true floor was 6.9×–16.9× larger, forcing
an **estimation-mode pivot**: the design demoted its own primary from a
fixed-threshold pass/fail bar to a pinned-CI point estimate
(`mean_delta ± t(2,.975)·s/√n`, thresholds derived from the program's own
real noise floors, not an assumed one). Round 5 pinned this single CI
formula across every bar in the document and descoped the program to
rung-1-only, with rung-2 formally parked behind a fresh review.

**The rung-1 result itself (all 20 mandatory cells + full measurement
pipeline complete, verified independently against the raw per-seed
`fitinput_*.json` arrays to full float precision):** the **primary**
(Arm 2 − Arm 1′, post-blend `span_frac`) excludes zero in BOTH corpora —
openr1-mix-ext **+0.1955** [0.0937, 0.2973], wikitext-mix-ext **+0.2273**
[0.0926, 0.3621] — but is **positive**, the opposite direction from every
sim family's prediction (all predicted a stabilizing, negative effect).
The **co-primary** (pre-blend `k_raw` geometry) moves the same direction
(openr1 +0.1097 [0.0491, 0.1704], wikitext +0.1345 [0.0070, 0.2621]),
ruling out the specific "post-blend-only win" suspicious pattern the
design pre-registered a flag for, and confirming the effect is
training-mediated rather than a static blend-arithmetic artifact. This is
exactly `FROZEN_BIAS_LM_DESIGN.md` §1.3's own pre-registered **FOURTH
OUTCOME, "sim-training divergence"** — explicitly neither a CONFIRM (the
sign is wrong relative to the registered mechanism) nor a REFUTE (the
more-sensitive co-primary instrument is not null; it moved cleanly, same
direction as the primary) — reported as its own disclosed, informative
category rather than forced into either bucket.

**The single most striking number in this chapter is the control
contrast.** The same frozen table's per-token-keyed construction and its
global-mean-vector construction, trained identically at matched λ=0.58 on
the same two corpora, move the same observable in **opposite,
both-CI-excluding-zero directions**: per-token training-mediated delta
+0.1955/+0.2273 (destabilizing — the observable spreads out) versus
global-vector training-mediated delta **−0.3319/−0.2308** (stabilizing —
the observable collapses), per §7.1a's mandatory licensing comparison.
Neither sim family in the design anticipated a sign split; both predicted
a uniformly stabilizing direction regardless of per-token keying. The
cosine diagnostic the design pre-registered specifically for this
divergence outcome (trained `k_raw` vs. its own frozen anchor row,
before/after training) reads ~0 in every arm, ruling out key-anchor
alignment as the mechanism behind either direction — leaving the sign
split itself unexplained, an open mechanism question for a future,
separate wave.

**Honest framing, stated as plainly in this narrative as in the design
doc's own verdict section:** this is a discovery-shaped
negative-plus-surprise, not the hoped-for transplant confirm. The
testbed's own constancy-suffices gain does **not** straightforwardly
transplant via a dense per-token frozen key bias at 14M-parameter LM
scale. What the wave delivers instead is a precisely-measured, controlled,
genuinely unpredicted training-mediated geometry effect (the
destabilize-vs-stabilize split above), caught only because the co-primary
and control instruments were built into the design before any training
cell ran. The result sits at **descriptive tier only, not confirmatory** —
the blind-pin (`BANDS_PINNED-FrozenBias.json`) was, by its own timestamp,
written after all 20 training cells had already completed, so while it
correctly fixed the Arm 1′/Arm 1″ reference quantities against
after-the-fact tuning, it forfeits the full pre-registration guarantee a
before-launch pin would have carried — a process lesson the design's own
verdict section names explicitly for future waves (pin before the
training launch, not just before the fit). Realized cost: **≈6.90 GPU-h**
(5.05 training + 1.6 measurement eval + 0.25 calibration), against a
~14.2 GPU-h committed estimate at 2× contingency, itself a small fraction
of this paper's own much larger anchoring-program and scale-program
budgets. No figure or claim from this chapter is proposed for the main
paper's own headline results — it is recorded here as a disclosed,
adjacent side-finding sharing the same architecture and instrumentation
lineage, not as a load-bearing result for §0/§1's own thesis. Full
verdict: `FROZEN_BIAS_LM_DESIGN.md` (VERDICT section, appended
2026-07-06); archive: `experiment-runs/2026-07-06_frozen_bias_rung1/`
(+ SSD mirror).

---

## 0. The one-sentence thesis

**Rank recruitment and rank *exactness* are different achievements: a
fixed-state linear-attention model (DeltaNet) trained end-to-end on real
tokenized language reliably recruits the provably-necessary rank at every
tested capacity K, but the per-binding exactness of what it writes into
that rank decays sharply with K and compounds geometrically (~ε^h) under
composition — a decay traceable to a non-orthonormal write-time key
attractor that every input geometry converges to, and — once directly
tested via a 2×2 existence-proof design cutting the archive along two
independent geometric axes — shown to require TWO independently necessary
and jointly sufficient ingredients (within-episode orthogonality and
cross-episode key stability; neither alone moves held-out composition off
a ~0.005–0.011 floor statistically indistinguishable from the fully-learned
baseline's own 0.009, while the two together saturate to 1.00). A
differentiable, per-episode structural orthogonalization mechanism supplies
the first ingredient alone and already restores near-perfect compositional
recall at the pre-registered minimum-publishable bar (K=16, 3/3 seeds) and
improves the harder K=32 regime ~45× at full evidentiary admissibility —
narrowly missing its pre-registered headline bar for a reason attributed to
the missing second ingredient (trained cross-episode key drift, measured
0.90–0.94, inside the registered HIGH band at both K). **A follow-on
mechanism designed to let SGD supply that second ingredient itself has now
run to completion across four waves, and the finished story is sharper and
more interesting than "SGD supplies it": learned per-entity anchoring lifts
K=32 held-out four-hop recovery from a freshly-measured 0.4105 baseline to
0.61–0.71 (independently replicated on 9 seed-runs across 3 distinct seed
sets), but a purpose-built per-entity engagement instrument shows this gain
is NOT attributable to entities' raw keys learning to align with a
per-entity anchor (<14–41% individual engagement, median alignment below
the design's own partial-credit floor, at every measured leg) — Outcome C,
confirmed at increasing evidentiary strength across two independent
mechanism-tier waves, immune to every specific rescue attempt tried.**
**The actual explanation was then found and confirmed by ablation: a
frozen, never-trained, RANDOM anchor table at the same blend weight
matches or exceeds the learned table's own gain (0.67–0.76 vs. 0.61–0.71),
with a negative-median alignment reading confirming the instrument
correctly detects "nothing to align to."** The mechanism is not learned
entity alignment; it is the mere arithmetic presence of an
episode-constant additive term in the key blend at a large-enough blend
weight — SGD's only real contribution is tuning that one scalar. **This
fix's regime is now bounded, not open-ended: extending the same design to
K=48 (three-quarter capacity) shows the mechanism's two named explanatory
channels are both measurably active yet the fix still misses its own bar
0/3 seeds — the capacity curve is a cliff (~1.00 → ~0.65 → ~0.02 at
K/d=0.25/0.5/0.75), not a smooth decline, and the theoretical ceiling
itself declines only gently over the same range, meaning the shortfall is
a real, structural saturation, not a measurement artifact.** The attractor
this entire story is about is not a small-model curiosity: on a controlled
parameter ladder, closing its own residual data-mix confound at both
tested scales, the write-geometry deviation climbs monotonically on a
single, held-fixed-mix axis — 0.248 (14M) → 0.344 (98M) → 0.389 (392M) — a
**pure-scale** finding, with a fourth (1.31B) rung running as this is
written, the paper's one remaining open readout. Beyond DeltaNet itself, a
companion measurement program installing the same fix mechanism inside a
free-running-text LM (Track B) found it **doubly barred** from that
setting, while a substantially larger version of the same non-orthonormal
write geometry is measurable in deployed production fixed-state language
models, though not yet attributable to the delta-rule mechanism
specifically at this measurement tier (an honest non-finding, not a
confirmation).**

Shorter version for the abstract's first line: *SGD reliably finds the rank
a task requires; it does not reliably find the geometry that rank needs to
compose exactly — that gap has a cause with two necessary parts, a
demonstrated fix for one of them, a named and pre-measured residual for the
other, and a completed test of whether SGD can be made to find that second
part unassisted, whose answer turned out to be "no, but something even
simpler than SGD can": a frozen, untrained noise table does the same job,
and that fix's working regime has now been measured and bounded — it holds
at K/d≤0.5 and collapses at K/d=0.75.*

---

## 1. Abstract (draft)

> Recent work measures the rank of fast-weight linear-attention states in
> pretrained LLMs and finds it bounded by sequence length, but never asks
> whether SGD is *forced* to use that rank, nor whether the resulting
> operator is exact. We close both gaps in a single architecture family.
> First, on a synthetic task with a provable `rank(S_T) ≥ K` lower bound for
> exact recovery, we show a production DeltaNet layer (`chunk_delta_rule`)
> both recruits effective rank exactly K (no inflation: entity-subspace and
> whole-matrix rank agree to four decimal places) and — via a causal
> eval-time truncation staircase — depends on every recruited dimension: a
> razor-sharp cliff at k=K−1→K, reaching `rec@0.9`=1.0000 at every
> composition depth up to h=21 for k≥K. Second, transplanting the identical
> `(d=64, K)` operating point onto real GPT-2-tokenized language, we find
> rank recruitment survives unchanged (94–99% of target K at every cell) but
> per-binding *exactness* collapses into a graded, K-dependent frontier:
> held-out two-hop composition falls from 0.999 (K=8) to 0.26 (K=32), and
> the eval-truncation transition — a single-step cliff synthetically —
> widens to a 12–13-rank window. Third, we attribute this gap causally: a
> controlled embedding-source interpolation shows every tested input
> geometry (learned, frozen-orthonormal, frozen real-embedding,
> Gram-matched) converges to the *same* non-orthonormal write-time key
> attractor, and a surgical orthonormal-key pin (bypassing the learned
> projection entirely) achieves perfect composition (1.00/1.00/1.00 at
> h=1/2/3, K=32) — proving the architecture can represent and use the exact
> solution, and that ordinary SGD does not find it. Fourth, we show two
> pre-registered soft interventions (an orthogonality penalty, ZCA
> whitening) move the attractor only 3–8% and the resulting recovery gain
> shrinks geometrically with hop depth, exactly as an ε^h compounding law
> predicts — a real but categorically insufficient effect. Fifth, we design,
> gate, and demonstrate a differentiable fix: per-episode Newton-Schulz key
> orthogonalization at the write site, launched only after a pre-registered
> drift-based simulator predicted it would clear the bar. The fix lifts
> held-out four-hop recovery at K=16 from a 0.42–0.47 baseline to
> 0.95–1.00 (3/3 seeds admissible against a pre-registered, arm-specific
> criterion, clearing the ≥0.8 minimum-publishable bar) and improves K=32
> four-hop recovery ~45× (0.009 → 0.39–0.50, 3/3 seeds fully admissible),
> narrowly missing the pre-registered ≥0.5 headline bar (mean 0.44, one
> seed at the line) — a miss with a pre-measured cause: trained
> cross-episode key drift (0.904) falls in the pre-registered high-drift
> band (<0.95), the exact "stable-not-just-orthogonal geometry" failure
> mechanism an adversarial design round had named and measured before the
> wave ran, isolating cross-episode key *stability* as the remaining
> bottleneck. Sixth, we sharpen this attribution with a 2×2 existence-proof
> design cutting the same archive along two independent geometric axes
> (orthogonal × stable): neither ingredient alone moves held-out four-hop
> recovery off a 0.005–0.011 floor statistically indistinguishable from the
> fully-learned baseline (0.009), while a hand-built key that is both
> orthogonal and stable saturates to 1.00 at h=4 *and* h=7 — the fix above
> supplies exactly one of two necessary ingredients. Seventh, we test
> whether SGD can supply the second ingredient itself via a learned,
> per-entity cross-episode key anchor, across two mechanism-tier waves at
> increasing evidentiary strength: the behavioral gain is real and
> independently replicated (K=32 held-out four-hop recovery 0.41→0.61–0.71
> across 9 seed-runs, 3 distinct seed sets, 2 architecture variants), but a
> purpose-built per-entity engagement instrument shows fewer than 14–41% of
> entities individually engage at every measured leg — the entity-alignment
> mechanism is rejected (Outcome C), immune to every specific objection an
> independent attack round could raise. Eighth, we then ask *why* a
> rejected mechanism still produces a real, reproducible gain, and answer
> it by ablation, not speculation: a frozen, never-trained anchor table
> (random init or frame-potential init, both tested) at the same blend
> weight matches or exceeds the learned table's own gain (0.67–0.76 vs.
> 0.61–0.71), with a negative-median alignment reading in the random-init
> arm confirming the instrument correctly reports "nothing to align to."
> Constancy in the blend arithmetic — not learned content — is the active
> ingredient; SGD's only demonstrated contribution is tuning the blend
> weight. Ninth, we bound this fix's operating regime by extending the same
> design to K=48 (K/d=0.75): the fix misses its own bar 0/3 seeds (mean
> 0.0215 vs. 0.0244) while remaining fully admissible and while both of its
> own named explanatory channels (drift-space closure, ~81% of the gap to
> ceiling; value-Gram relief, 0.626× the reference) are measurably active —
> the capacity curve completes as a cliff (~1.00→~0.65→~0.02 at
> K/d=0.25/0.5/0.75), not a smooth decline, while the theoretical best-case
> ceiling declines only gently (0.97→0.94→0.90) over the same range;
> binding survives everywhere (h=1 guard 1.0000, every cell, every K) while
> held-out composition is what collapses. Tenth, we show the write-geometry
> attractor motivating this entire study is not a small-model artifact and,
> once a residual data-mix confound is closed at both tested scales (Δ
> −0.004 span at 14M, Δ −0.014 span at 98M, both within noise), worsens
> monotonically on a clean, single-mix, three-point ladder (span-fraction
> 0.248 → 0.344 → 0.389 at 14M/98M/392M) — a **pure-scale** finding, with a
> fourth (1.31B) rung running as this is written; a substantially larger
> version of the same non-orthonormal write geometry is separately
> measurable in production-scale fixed-state language models (RWKV-7 1.5B,
> Falcon-Mamba-7B; 3–6× their own random anchors), though a matched
> no-fixed-state negative control shows the same magnitude, so
> production-scale attribution to the delta-rule write mechanism
> specifically remains open, an honest non-finding reported as such.
> Eleventh, a companion attempt to install the demonstrated fix mechanism
> inside a free-running-text LM is blocked before any interaction claim can
> even be tested: a duplicate-key numerical-stability smoke overshoots its
> bar by 63× with a probative positive control, compounding an earlier gate
> refusal on the same setting; the selectivity-only question that setting
> could still measure, with the fix mechanism inactive, returns an
> inconclusive split verdict across two corpora. We report the full
> pre-registration, attack, and gating trail as an artifact, and discuss
> what remains unexplained: a separate iteration-depth decay (h=21) that
> key orthogonalization does not touch, an unresolved K≈d/2 structural
> boundary now sharpened into a measured capacity cliff, why value-crowding
> (not yet directly tested) is the design's own leading candidate for what
> would need to change to cross K/d=0.75, whether the fix mechanism can be
> safely installed in a free-running-text setting at all, and whether the
> production-scale signature is ever attributable to fixed-state writes
> specifically.

*(Word count target ~300–340 for the actual submission; this draft runs
well over that and is not yet trimmed — the anchoring arc (Seventh through
Ninth) is now a complete, closed sub-story and should compress hard for the
real submission: one or two sentences ("a learned mechanism produces a real
gain we show by ablation is not due to learned content, and a capacity
extension bounds where the fix works") can likely replace most of the
current length, with the full sequence kept in the body. No load-bearing
placeholders remain for anything already decided; every open item above is
explicitly flagged as open (rung-3, value-crowding, Track B transfer,
production-scale attribution), not a placeholder.)*

---

## 2. Figure plan (11 figures, exact data sources — Fig. 11 ADDED this round; Fig. 10 REWRITTEN)

**Fig. 1 — The synthetic-vs-real gap (motivating figure, goes in §1 Intro).**
Two eval-truncation staircase curves (`rec@0.9` vs. truncation rank k) at
the *identical* operating point `d_state=64, K=32`, h=1, side by side:
synthetic (razor cliff, k=31→32: 0.9681→1.0000, ceiling exactly 1.0000) vs.
real tokenized text (graded, steepest single-step jump only +0.066 at
k=19→20, ceiling reached exactly at k=32 but the ceiling itself is
**0.795, not 1.000** — real data never reaches the synthetic ceiling even
at full untruncated rank, a second gap layered on top of the width gap;
state both in the caption). Annotate the window width explicitly: 12–13
ranks below K=32 show non-trivial recovery on real data vs. 1 rank
(k=31 only) on synthetic.
- Synthetic data: `experiment-runs/2026-07-02_deltanet_waves/eval_trunc/`
  (`DELTANET_CAUSAL_RANK_DESIGN.md` §12.8.3 table).
- Real data: `experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA}/`,
  aggregated by `matrix-thinking/deltanet_rd/analyze_eval_truncation_rd.py`
  (`DELTANET_REALDATA_DESIGN.md` §17.3/§17.4 — K=32 uses n=7 seeds pooled
  from wave0_rerun + waveA).

**Fig. 2 — The headline phenomenon: a graded K-axis exactness frontier on
real text.** `rec@0.9` vs. h (1,2,3 in-distribution; 4–7 held-out), one line
per K∈{8,16,24,32}. Shows: K=8 near-exact through several held-out hops
(1.000/0.999/0.978 at h=1/2/3), K=16 partial (0.997/0.90/0.68), K=24
(0.94/0.58/0.27), K=32 collapsed beyond h=1 (0.78/0.26/0.05).
- Data: `experiment-runs/2026-07-03_deltanet_rd_waves/waveA/`
  (`DELTANET_REALDATA_DESIGN.md` §16.2).

**Fig. 3 — Capacity is not the difference: entity-subspace rank recruited
vs. K.** Bar/line chart, measured effective rank as % of target K, at every
tested K (synthetic ≈100%, real 94–99%) — establishes that Fig. 2's decay is
NOT a rank-recruitment failure.
- Data: `experiment-runs/2026-07-03_deltanet_rd_waves/waveA/` (§16.4) +
  `experiment-runs/2026-07-02_deltanet_waves/wave0/` (§12.4, synthetic
  no-inflation reference).

**Fig. 4 — The attribution: every input geometry converges to the same
attractor.** Bar chart of key-Gram deviation (`‖K_eff^T K_eff − I‖_F`) at
K=32 across arms: learned (iii, baseline, 2.71–2.77), frozen orthonormal
(i), frozen real GPT-2 embedding (ii), Gram-matched frozen control (iv) —
all clustering at 2.71–2.92 — versus the surgical pin (i-strong, 0.000).
- Data: `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wave1/`
  (`EXPERIMENT_LOG.md` "Wave 1 ATTRIBUTION VERDICT," 2026-07-04).

**Fig. 5 — The 2×2 existence proof: orthogonality and stability are each
necessary, jointly sufficient.** A 2×2 grid, axes {orthogonal,
not-orthogonal} × {stable (context-free), not-stable (episode-conditional)},
cells populated with `rec@0.9` at h=4 (and h=7 in the caption), K=32, all
values re-verified directly against the archived per-seed JSONs:
  - **stable + orthogonal** (i-strong, hand-built context-free orthonormal
    key, pool-restricted to 32 ≤ d_state — an existence-proof concession,
    stated in the caption, not hidden): h=4 **1.0000**, h=7 **1.0000**, all
    3 seeds.
  - **stable + not-orthogonal** (frozen-embedding arms i/iv — a fixed input
    table, but the trained `W_k` path still de-orthogonalizes it): h=4
    **0.0075–0.0114** (arm i, seed-0 0.0093) and **0.0049–0.0079** (arm iv,
    seed-0 0.0079), h=7 **≈0.0000** both arms, all seeds.
  - **not-stable + orthogonal** (bare geo3, F-geo-3 alone): h=4 **0.44**
    mean (0.39–0.50 admissible range, §16.4), h=7 **0.018** mean.
  - **not-stable + not-orthogonal** (the fully-learned baseline, arm
    iii-β): h=4 **0.009**, h=7 **≈0.0000**.
  Read across the grid: neither axis alone clears the ~0.01 floor
  (stable-alone ≈ baseline, confirming raw input geometry stays causally
  irrelevant on this axis too); orthogonality alone recovers most of the
  gap (44–56×) but plateaus at 0.44; adding stability is not a partial
  improvement over that 0.44, it is a discontinuous jump to 1.00 — the
  interaction, not either main effect, is the figure's point. Caption must
  disclose the i-strong cell's pool restriction explicitly, and must now
  add a forward-pointer to Fig. 10/11: the "jointly sufficient" corner's
  only positive demonstration remains this hand-built pin — the learned
  mechanism built to reach that corner (§9) produced a real gain via a
  DIFFERENT mechanism than hypothesized (see Fig. 10), not a second route
  to the same corner.
- Data: `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wave1/`
  (arms i, iv, iii-β), `experiment-runs/2026-07-03_deltanet_rd_waves/
  exactness/wavegeo3/` (geo3-alone), and `KEY_ANCHORING_DESIGN.md` §2.0's
  own extraction table.

**Fig. 6 — Soft fixes fail, but proportionally, not randomly (the ε^h
compounding law).** Line chart: recovery gain over baseline at h=2/3/4 for
the pooled soft arms (F-geo-1 λ∈{0.1,1.0}, F-geo-2 ZCA) at K=32 — gain
shrinks from +0.07–0.10 (h=2) to +0.04–0.05 (h=3) to +0.005–0.017 (h=4),
against the two pre-registered bars (headline ≥0.5, missed ~20–25×; minimum
publishable ≥0.8, missed) shown as reference lines.
- Data: `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/waveF/`
  (`DELTANET_RD_EXACTNESS_DESIGN.md` §15.1–§15.3).

**Fig. 7 — The fix: F-geo-3's headline demonstration.** Grouped bar chart,
h=4 `rec@0.9` at K=16 and K=32: baseline vs. best soft arm vs. F-geo-3, with
the two pre-registered bars marked as horizontal reference lines (K=16
≥0.8: HIT 3/3 admissible seeds, 0.95–1.00; K=32 ≥0.5: improved ~45× to
0.390/0.416/0.504, all 3 escalation seeds fully admissible — zero fallback
steps, every substitute-stack criterion passed — mean 0.44, bar narrowly
missed with 1/3 seeds at the line). Overlay or annotate the measured
trained cross-episode drift per K (0.94 K=16 vs. 0.9037 K=32 against the
pre-registered 0.95 HIGH-band threshold) so the figure itself carries the
outcome-F attribution of the K=32 residual. The admission-stack difference
drops to a footnote in the caption. Do NOT caption this figure with the
retracted "the simulator overestimated K=32" reading (§16.7's dated
correction) — fed its OWN drift, the simulator is CONSERVATIVE at K=32, and
the outcome-F attribution never rested on the simulator's point estimate.
- Data: `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wavegeo3/`
  (`EXPERIMENT_LOG.md` "F-geo-3 WAVE VERDICT" 2026-07-04 + the completed
  n_iter=20 escalation cells in the same archive directory).

**Fig. 8 (appendix / generality figure) — Wave 2: the phenomenon is not an
artifact of the synthetic probe grammar.** Truncation-damage-vs-k curves
(nats/token), reasoning (OpenR1-Math) vs. narrative (WikiText) corpora, at
k∈{8,16,24,32,48,64}, showing openr1 consistently more truncation-sensitive
at low-to-moderate k, converging to the same noise floor at k*=48 for BOTH
corpora. Caption must carry two named confounds (symbol-density, home- vs.
cross-domain competence), not smoothed over. Offered as descriptive/
interventional support (RD-2 tier, explicitly NOT premise-conditional
causal).
- Data: `experiment-runs/2026-07-04_lm_rd_wave2/{waveC,waveD}/`, analysis
  via `analysis_lm_w2.py` (`DELTANET_REALDATA_DESIGN.md` §19.3).

**Fig. 9 (appendix / generality figure) — The attractor persists with
scale, and a larger version of it exists at production scale,
unattributed.** Two-panel figure. **Panel A (Track C pure-scale ladder):**
pooled chunk-level key-Gram deviation, a clean 3-point, single-(extended)-
mix ladder: 14M (mixcontrol) span-fraction **0.248**, 98M (wave-1ext)
span-fraction **0.344**, 392M (rung-2) span-fraction **0.389**,
monotonically increasing, plotted against each rung's own random/collapse
anchors. **Regeneration note (new this round):** a dense per-seed
trajectory dataset now exists for this panel —
`experiment-runs/2026-07-06_trajectory_probes/trajectories_tidy.{json,csv}`
(540 rows, all 6 seeds × both corpora × all 4 cells at 7–35 checkpoints
each, not just the archived seed-0-only 8–11-point reads) — use
`archived4_span_frac` as the plotted quantity (the cross-scale-comparable
convention; raw `gram_deviation_mean` is NOT comparable across `d_state`).
Validated exactly against the archived harvests (66/66 finals ≤1e-6,
24/24 overlapping trajectory points ≤1e-6) — this is a pure re-plot with a
denser per-seed dataset, not a new claim. A fourth, 1.31B-parameter rung
(rung-3) is running as this is written; harvest is expected imminently and
is NOT included in this figure or its numbers — when it lands, add it as a
fourth point using the same instrument (md5 `3fb0f80028477d0b1cefe468c81b1da4`)
and the same `archived4_span_frac` convention. **Panel B (Track D Phase 1):**
the same chunk=64 Gram-deviation statistic measured (Tier 3,
measurement-only) in three production-scale pretrained checkpoints: RWKV-7
1.5B (43.5–44.0), Falcon-Mamba-7B (49.9–50.2), and a no-fixed-state
negative control, Qwen2.5-1.5B (46.0–48.5) — all three cluster together,
RWKV-7 and the negative control statistically indistinguishable. Caption
must carry the honest verdict: the signature is measurably present and
larger at production scale, but not attributable to the delta-rule family
specifically without a sharper instrument.
- Data: Panel A — `experiment-runs/2026-07-04_trackc_rung1/`,
  `experiment-runs/2026-07-05_trackc_rung2/`,
  `experiment-runs/2026-07-05_wave1ext/` (archived seed-0 harvests) +
  `experiment-runs/2026-07-06_trajectory_probes/` (dense per-seed
  regeneration dataset, this round). Panel B —
  `experiment-runs/2026-07-04_track_d/` (`SCALE_TRANSFER_DESIGN.md` §6.8).

**Fig. 10 (REWRITTEN this round — was "a real gain, an unconfirmed
mechanism"; now "a real gain, a rejected mechanism, a confirmed
explanation") — The anchoring wave's complete arc.** Three-panel figure,
replacing round 4's two-panel draft. **Panel A:** grouped bar chart, K=32
h=4 `rec@0.9` across every wave in the program's history, in chronological
order: fresh bare-geo3 reference (**0.4105**), Wave-1/confirm (same-seed
reproduction, 0.556–0.665), mechanism-tier wave (6 fresh seeds, 2
architecture variants: candidate (d) 0.6141–0.7141, candidate (d′)
0.6661–0.7141), with the ≥0.5 bar drawn as a reference line — visually
establishes the behavioral gain as independently replicated (9 seed-runs,
3 distinct seed sets), not a one-off. **Panel B:** per-cell
`engaged_frac_v3` / median `r_e` scatter (mechanism-tier wave, both K=32
arms, all 6 seeds) against the design's own bands (C: median r_e<0.25;
A_partial/A: higher) — every K=32 cell lands band C (engaged_frac 3.7–41.1%,
median r_e 0.15–0.23), visually showing the engagement instrument's verdict
is not a borderline call. Caption must state the literal assigned outcome
(**C**, both arms, 3/3 seeds each) and note this is a *strictly stronger*
null than round 4's confirm-wave C (unit-anchor-norm measured not assumed;
formal dip-test rules out masked bimodality; effect-size floor rules out
power-manufactured engagement). **Panel C (NEW):** the resolution — grouped
bar chart, K=32 h=4, candidate (d) learned table (mean **0.6669**) vs.
frozen-random arm e (mean **0.7274**) vs. frozen-frame-potential arm e-fp
(mean **0.7413**), with per-seed dots shown, plus a small inset showing
arm e's median r_e is **negative** (−0.13 to −0.24) — the negative-control
reading that confirms the instrument works correctly on a table that
cannot mechanically engage. Caption states the routing plainly: "both
frozen arms match or exceed the learned table ⇒ constancy alone suffices;
the entity-alignment mechanism (Panel B) is rejected, but the behavioral
gain (Panel A) is real and is now explained."
- Data: Panel A — `experiment-runs/2026-07-05_keyanchor_wave/`,
  `experiment-runs/2026-07-05_keyanchor_confirm/`,
  `experiment-runs/2026-07-06_keyanchor_mech/`. Panel B — same
  mechanism-tier wave archive. Panel C —
  `experiment-runs/2026-07-07_keyanchor_e/`. Derivation:
  `KEY_ANCHORING_DESIGN.md` §9/§9.6/§10.13/§10.14.

**Fig. 11 (NEW this round, main-text candidate — THE HEADLINE FIGURE for
this program's arc) — The capacity curve: where the fix works, and where
it stops.** Line chart, x-axis K/d_state ∈ {0.25, 0.5, 0.75} (K=16/32/48 at
d_state=64), two lines: (1) candidate (d)'s measured h=4 `rec@0.9` — ~1.00
(K=16, no-regression check, saturated) → 0.61–0.71 (K=32, HIT its ≥0.5 bar)
→ 0.0215 (K=48, MISS its ≥0.024434 bar, 0/3 seeds) — a visible cliff
between the second and third points, not a smooth decline; (2) the
theoretical λ=1 ceiling computed alongside each point — 0.9745 → 0.9423 →
0.8987 — declining gently and near-linearly over the same range. The gap
between the two lines at K=48 is the figure's point: the mechanism is not
failing because the ceiling collapsed (it's still ~0.90), it is failing
to reach anywhere near a ceiling that is still high. Annotate K=48 with
the two checked explanatory channels and their status: drift-space
closure (post-NS drift 0.887 vs. ceiling 0.899 vs. reference 0.838 — closes
~81% of the gap) and value-Gram relief (0.626× the reference's deviation)
— both real, neither sufficient. Small inset: h=1 in-distribution guard,
flat at 1.0000 across all three K — binding survives, composition
collapses. This is the program's cleanest single-figure summary and should
be considered for main-text placement over Fig. 10's more detailed
three-panel breakdown, which can move to the appendix if main-text space is
tight.
- Data: `experiment-runs/2026-07-06_keyanchor_mech/` (K=16/32),
  `experiment-runs/2026-07-07_keyanchor_k48/` (K=48, both candidate-(d) and
  fresh reference arms, plus `BANDS_PINNED_K48.json` for the ceiling
  values). Derivation: `KEY_ANCHORING_DESIGN.md` §11.4.2/§11.11/§11.12.

**Fig. cliff (NEW, round 6 — supersedes Fig. 11 as the capacity-law
headline) — The capacity cliff, located.** Single-panel figure, x-axis
K/d_state (bottom) with a twin top axis in raw K at d_state=64: the
7-point curve — 3 archived anchors (K16/32/48, grey squares) plus 4 new
points this wave measured (K34/38/42/46, orange circles) — with per-seed
scatter shown around each point's mean, the fitted sigmoid
`h4(x) = L/(1+exp((x−x0)/w))` overlaid as a smooth curve, and a shaded
vertical band marking the 95% bootstrap CI on `x0`
(**[0.5385, 0.5513]**, centered on the fitted **`x0=0.5455`**,
`w=0.0597`). Distinguishes visually, by marker and color, the 3 archived
anchor points from the 4 new localization points, so a reader can see at a
glance which points pre-existed this wave and which are new. This is the
figure a reviewer checks the "candidate-(d) capacity cliff sits at
K/d≈0.55, width≈0.06" claim against directly — it replaces Fig. 11's
3-point line chart as the paper's capacity-law headline (Fig. 11's
simpler 2-line, ceiling-vs-measured framing can move to the appendix as a
predecessor view, or be cut, a camera-ready layout call).
- Data: `experiment-runs/2026-07-06_keyanchor_cliff/` (4 new K's, per-seed
  cell JSONs + `fit_cliff_curve_results.json` for the sigmoid fit and
  bootstrap CI), `experiment-runs/2026-07-06_keyanchor_mech/` (K=32
  archived anchor, seeds {10,11,12}), `experiment-runs/2026-07-07_
  keyanchor_k48/` (K=48 archived anchor, seeds {30,31,32}). Derivation:
  `KEY_ANCHORING_DESIGN.md` §12.4/§12.9. Script:
  `matrix-thinking/submissions/iclr-2027/figures/make_fig_cliff.py`.

---

## 3. Section-by-section argument flow (maps to `main.tex` skeleton)

### §1 Introduction
- Open on the tension: descriptive rank-measurement papers (Nazari & Rusch;
  Sun et al., both Feb 2026) show fast-weight state rank is bounded by
  sequence length in pretrained LLMs, but never show SGD is *forced* to use
  that rank, nor that the resulting operator is *exact*.
- Contribution bullets (claim-tiered from the first sentence):
  1. A provable `rank(S_T) ≥ K` necessity result for exact-continuous
     recovery in a production fast-weight layer (DeltaNet), CONFIRMED
     causally via a razor-cliff eval-truncation staircase on synthetic
     data (CONFIRMED tier — no hedge needed here).
  2. The SAME causal necessity result, CONFIRMED on real tokenized
     language at the identical `(d,K)` operating point (RD-1, CONFIRMED,
     graded-not-razor qualifier attached every time it is stated).
  3. A NEW phenomenon this transfer exposes that the synthetic result
     cannot see: per-binding exactness (not capacity) is the thing that
     degrades with K on real data, and it compounds geometrically with
     composition depth.
  4. A causal attribution of that phenomenon to write-time key geometry
     via a controlled embedding-source intervention, plus an existence
     proof that the exact solution is architecturally reachable.
  5. A demonstrated (not just diagnosed) fix — differentiable per-episode
     key orthogonalization — that clears the pre-registered bar at K=16
     (3/3) and improves the harder K=32 case ~45× at full admissibility,
     with the narrow headline-bar miss itself attributed to a second,
     pre-registered, pre-measured mechanism (cross-episode key drift,
     outcome F) — the causal chain closes with no unexplained residual.
  6. A follow-on mechanism (learned per-entity key anchoring) that tests
     whether SGD can supply that second ingredient itself — run to
     completion across four waves — produces a real, independently
     replicated behavioral gain (K=32 h=4 0.41→0.61–0.71) but REJECTS its
     own hypothesized mechanism (Outcome C, entity alignment not engaged,
     confirmed at increasing evidentiary strength across two mechanism-tier
     waves, immune to a specific rescue attempt that was independently
     attacked and rejected).
  7. The gap's actual explanation, found by a pre-registered ablation, not
     speculation: a frozen, never-trained anchor table matches or exceeds
     the learned table's own gain — cross-episode stability comes from the
     mere arithmetic presence of an episode-constant blend term, not from
     entity-specific learned content. CONFIRMED-BY-ABLATION, the strongest
     evidentiary tier this program's own claim-tier system recognizes for
     a secondary mechanistic account.
  8. A measured boundary of where this fix's regime holds: extending the
     identical design to K/d=0.75 shows the fix fully admissible but
     missing its own bar 0/3 seeds, with both of its own named explanatory
     channels active-but-insufficient — the capacity curve is a cliff
     (~1.00→~0.65→~0.02), not a gradual decline, while the theoretical
     ceiling itself declines only gently over the same range.
  9. The process itself as a minor, explicitly-not-self-congratulatory
     contribution: every wave above was pre-registered, attacked by an
     independent reviewer, and gated on a predicted outcome before any GPU
     time was spent — including, this round, a falsifiable account of a
     mechanism (the constancy-suffices explanation) that was itself
     registered with a stated pass/fail prediction BEFORE the confirming
     data existed, and then confirmed on fresh data under a hash-locked
     instrumentation stack.

### §2 Related Work
See §5 below (full map). Placed early per ICLR convention; light touch,
saves the sharper distinctions (esp. vs. M²RNN and Nichani et al.) for
where they're load-bearing in §4/§5 body text via forward-reference.

### §3 Setup and Theory
- Architecture: vanilla single-head (`H=1`), single-layer DeltaNet,
  `chunk_delta_rule`, standard delta rule
  `S_t=S_{t-1}(I-β_t k_t k_t^T)+β_t v_t k_t^T`.
- The provable bound (§4.1 of `DELTANET_CAUSAL_RANK_DESIGN.md`): under a
  *pinned linear-unbind readout* (`pred = S_T · k_j`, cosine-scored, never
  argmax), exact recovery of K independent bindings requires
  `rank(S_T) ≥ K`. State clean premise and the two escapes closed by
  construction (position-decomposition; multi-head/multi-layer — ReserveMH
  lifts the H=1 qualifier, 6/6 cells, every head at both H∈{2,4}
  independently recruits full rank K=32).
- Short, non-self-congratulatory paragraph on method: every claim below
  was pre-registered, attacked by an independent reviewer round, and — for
  the constructive fix and for the anchoring program's mechanistic
  account — gated on a predicted outcome from a committed instrument or
  simulator before any Wave 1 GPU spend. Concrete instance to cite (not
  just assert): a mini-audit caught a target-INDEX bug in the real-data
  grammar that made the binding task unsatisfiable as originally specced,
  caught by the premise-gate instruments at every one of its 10 seeds and
  correctly never written up as a finding, before the fix.
- **Two "primary K" values, kept distinct on purpose:** `K=16` is the
  *causal-necessity* program's primary cell; `K=32` is the
  *exactness-mechanism/fix/anchoring* program's headline cell, chosen
  because it is the most dramatic, most publication-relevant exactness
  failure, sitting at `K=d_state/2` — and, new this round, `K=48` extends
  this same operating-point logic one step further to `K=3·d_state/4`, the
  point where the same fix's regime is shown to end.

### §4 The Phenomenon: Rank Recruitment Is Reliable, Exactness Is Not
- Synthetic baseline (razor cliff): `DELTANET_CAUSAL_RANK_DESIGN.md` §12.3
  (Wave 0 saturates to `rec@0.9=1.0000` at every tested `(d,K)` cell) +
  §12.4 (no-inflation) + §12.8.3 (the k=31→32 cliff, matching the
  theoretical `(K−h)/K` prediction to 0.001–0.003).
- Real-data transfer (RD-1, CONFIRMED): same architecture, transplanted to
  GPT-2-tokenized language. Headline frontier (Fig. 2). Rank recruitment
  intact (Fig. 3: 94–99% of K). Eval-truncation staircase is real and
  causal but graded: ceiling reached exactly at k=K at every K tested, but
  the ceiling itself is sub-1.0 and falls with K, and the transition below
  ceiling widens with K.
- The depth-amplification law, imported from the from-scratch chapter
  (`TASK_E_FINDINGS.md` §3): raw iteration/composition depth is a sharper
  exactness probe than any single-hop cosine tolerance. This `ε^h`-shaped
  law recurs three more times in the paper: Wave F's proportionality
  reading (§6); the h=21 residual failure of the fix itself (§8); and,
  new to this round, the interpretation of why the capacity curve (§9,
  Fig. 11) is a cliff rather than a smooth decline — held-out four-hop
  composition is a depth-4 probe, and small per-binding write-quality
  losses at K=48 compound geometrically across 4 compositions the way
  they do everywhere else in this paper.
- The `h=21` caveat, stated once, cited everywhere it recurs.
- K=48 rider, restated with its final status: the frontier extends past
  d/2 then breaks. This is no longer only a diagnostic aside — §9/§11
  below turn it into a fully-measured third point on a completed capacity
  curve, with both of the design's own explanatory channels checked
  directly (not speculated) and shown active-but-insufficient. Flag this
  as the paper's one measured (not merely observed) structural boundary
  (K≈d/2 → collapse by K=3d/4).

### §5 Mechanism: Causal Attribution via Embedding-Source Interpolation
- The design idea: hold grammar/model/loss fixed, vary ONLY the embedding
  source feeding `k_eff`. Result: EVERY arm converges to the SAME
  non-orthonormal attractor — raw embedding geometry is causally
  IRRELEVANT; the trained W_k/conv path maps every input geometry onto the
  same write-geometry basin.
- The complementary non-interventional evidence: measured-β closed-form
  crosstalk reconstruction is near-exact.
- The existence proof (i-strong): a non-trainable, hand-built, per-identity
  k_eff pin achieves gram dev 0.000 and rec@0.9 = 1.00/1.00/1.00 at
  h=1/2/3, K=32. The architecture and the delta rule's own algebra CAN
  represent and use the perfect solution. Ordinary training does not find
  it.
- The 2×2 (`KEY_ANCHORING_DESIGN.md` §2.0, Fig. 5): orthogonality alone
  recovers most of the gap (0.44, a 44–56× jump) but plateaus well short
  of 1.0; stability alone buys nothing (0.005–0.011, indistinguishable
  from baseline); only the conjunction saturates. This reframes the
  paper's turning point from a single named cause to a two-part
  necessary-and-sufficient mechanism.
- **§9's full arc now belongs here as the closing move of this section,
  not a forward-pointer to an open question.** The 2×2's "jointly
  sufficient" corner was, at round 3/4, demonstrated only by the hand-built
  i-strong pin. The anchoring program asked whether a learned mechanism
  could reach that corner via SGD, ran it through four completed waves, and
  the answer is now final: **no, not via the hypothesized channel** (Outcome
  C, entity alignment rejected, confirmed at increasing rigor across two
  mechanism-tier waves) — **but the real behavioral gain those waves
  produced has its own confirmed explanation** (constancy in the blend
  arithmetic, not entity-specific learned content, per the candidate-(e)
  ablation), **and that explanation's operating regime is now measured**
  (K/d≤0.5 works, K/d=0.75 does not, per the K=48 capacity-curve
  extension). State this as what it is: the 2×2's positive "jointly
  sufficient" corner still rests on the hand-built pin for a *causal*
  demonstration; the *practical* fix that emerged from testing whether SGD
  could reach that corner does not use the hypothesized mechanism at all,
  works via a different (simpler, cheaper) route, and has a
  now-known-and-bounded regime of applicability. Both facts belong in the
  paper, clearly separated, neither one substituting for the other.

### §6 Soft Fixes Fail: The Attractor Is Robust
- F-geo-1 (L_orth penalty) and F-geo-2 (ZCA whitening): both pre-registered,
  both land in the SAME narrow basin (gram dev 2.51–2.57 at K=32 vs.
  baseline 2.71–2.77 — a 3–8% cleanup) regardless of mechanism or λ.
- Every pre-registered bar MISSED while the h=1 no-sacrifice guard is
  satisfied everywhere.
- The honest read: this is the pre-registered NEGATIVE branch, explicitly
  anticipated in the design, not a failed pilot re-run until something
  worked.

### §7 The Fix: Structural, Differentiable Orthogonalization (F-geo-3)
- Motivation: (1) the exact solution exists and composes perfectly
  (i-strong); (2) SGD does not find it under any soft push tried.
- Design: cubic Newton-Schulz/Björck-Bowie iteration at the k_eff site,
  chosen over Gram-Schmidt (order-dependent, reintroduces write-history
  asymmetry) and QR (order-dependent, backward-unstable near collinear
  input). Order-equivariant, differentiable, sqrt(K) pre-scaling gives an
  unconditional convergence-basin guarantee.
- The gating discipline: a two-round independent attack found a
  load-bearing failure mode the design missed (cross-episode key-identity
  drift), which became a pre-registered Wave −1 gating diagnostic with its
  own committed simulator and launch-read rule — Wave 1 GPU spend was
  authorized only after the simulator predicted the K=16 bar would clear.
- The substitute admission-stack caveat — DISCHARGED: with the K=32
  n_iter=20 escalation cells passing EVERY substitute criterion including
  zero fallback steps (K=16 3/3 from the start), this caveat drops from
  headline asterisk to footnote.
- **The direct follow-on to the outcome-F residual — designed, attacked,
  built, run to completion across four waves, and now CLOSED.** A
  trainable, per-entity anchor table blended into the key before geo3's
  Newton-Schulz pass. The full arc, in the order it actually happened:
  (1) launched twice (initial + confirmatory wave), producing a real,
  seed-stable K=32 h=4 gain (0.4105→0.556–0.665) whose confirmatory-wave
  per-entity engagement readout showed <14% engagement — Outcome C, not
  Outcome A — and whose same-day rescue attempt (a λ-scaled threshold
  re-derivation) was independently attacked and rejected; (2) a
  mechanism-tier wave (2026-07-06) reconfirmed Outcome C at a strictly
  stronger evidentiary bar (6 genuinely fresh seeds, 2 architecture
  variants, unit-anchor-norm measured not assumed, formal dip-test, an
  effect-size floor) and registered a falsifiable account —
  "cross-episode stability comes from the mere presence of an
  episode-constant blend term, not learned alignment" — with a stated,
  pre-registered pass/fail prediction; (3) candidate (e), a frozen
  never-trained anchor table (both random-init and frame-potential-init
  variants), tested that prediction directly and it HELD: both frozen
  arms match or exceed the learned table's own gain (0.67–0.76 vs.
  0.61–0.71), CONFIRMED-BY-ABLATION; (4) a K=48 capacity-curve extension
  bounded the fix's regime: fully admissible, both explanatory channels
  active, the bar still missed 0/3 seeds — the fix works at K/d≤0.5 and
  does not transplant to K/d=0.75. Program complete at ≈55.83/80 GPU-h, no
  further waves scheduled.

### §8 Results: The Fix Demonstration
- The pre-spend gate: predicted K16 h4=1.00, K32 h4=0.77 (later corrected
  to 0.06–0.09, a conservative-not-miscalibrated direction, §16.7 — the
  gate's own registered, decision-bearing cell was K=16 only and is
  unaffected).
- K=16 (minimum-publishable headline): HIT, 3/3 admissible seeds.
- K=32 (primary/headline anomaly cell) — admissibility CLEARED (n_iter=20
  escalation, zero fallback steps), headline bar (≥0.5) NOT met (mean 0.44,
  honest narrow miss), the miss ATTRIBUTED to outcome F (trained
  cross-episode drift 0.9037 in the pre-registered HIGH band).
- The fallback-irrelevance finding: n_iter=20 (zero fallback) and n_iter=12
  (56 fallback steps) are numerically IDENTICAL — rules out solver noise as
  a competing explanation.
- One number that does NOT move under the fix: h=21 literal-depth
  composition still collapses under F-geo-3.
- **The anchoring program's results now belong here as a completed
  sub-section, not a forward-reference — see §7's arc above for the
  narrative and Fig. 10/11 for the figures.** Report, in order: the
  behavioral gain (independently replicated, 9 seed-runs, 3 seed sets);
  the mechanism rejection (Outcome C, both architecture variants, immune
  to the specific objections a rescue attempt raised); the confirmed
  explanation (constancy suffices, by ablation); the measured capacity
  boundary (K/d≤0.5 works, K/d=0.75 does not, with both explanatory
  channels checked and found active-but-insufficient at the miss). This
  closes what round 4 left as this paper's single most open thread.

### §9 Discussion and Limitations
(Full content spec in §6 below — reproduced compactly in-line here too.)

### §10 Conclusion
- Restate the thesis: capacity and exactness are separable, measurable,
  and separately fixable properties of a fast-weight architecture's
  learned state — this paper is the first to show the gap exists on real
  text, name its cause (a two-part necessary-and-sufficient mechanism),
  demonstrate a mechanism-matched fix for the first part, and — new to
  this closing round — fully characterize a completed attempt at the
  second part: a real fix whose actual mechanism differs from its
  hypothesized one, confirmed by ablation, with its working regime
  measured and bounded.
- Close the causal chain explicitly: bar hit at K=16 (3/3); ~45× at K=32
  at full admissibility with the narrow bar miss attributed to outcome F;
  independently corroborated by the 2×2. The named follow-on (learned
  cross-episode key anchoring) ran to completion across four waves: a
  real, independently-replicated K=32 h=4 gain (0.41→0.61–0.71) whose
  hypothesized entity-alignment mechanism is rejected (Outcome C,
  confirmed twice, immune to a rejected rescue attempt), whose actual
  mechanism is confirmed by ablation (a frozen random table matches or
  beats the learned one — constancy, not alignment, is the active
  ingredient), and whose operating regime is now measured (works at
  K/d≤0.5, fails at K/d=0.75, both explanatory channels active-but-
  insufficient at the miss, a cliff not a smooth decline). No loose ends
  on the mechanism — the behavioral result, the rejected hypothesis, the
  confirmed explanation, and the capacity boundary are reported as the
  four separate things they are.
- **Closing note on scale (updated framing, same substance as round 4):**
  the phenomenon this paper studies is not confined to the scale it is
  studied at — a controlled 14M→98M→392M ladder, on a single held-fixed
  data mix, shows the write-geometry deviation climbing monotonically
  (span-fraction 0.248→0.344→0.389), a **pure-scale** finding with a
  1.31B-parameter fourth rung running as this is written — the paper's
  one still-open readout. A substantially larger version of the same
  signature is present in deployed production fixed-state models, not yet
  attributable to the same mechanism. A companion attempt to install the
  demonstrated fix mechanism itself inside a free-running-text setting
  (Track B) is, independently, blocked twice on numerical-stability
  grounds — named honestly as a current boundary, not smoothed over.

### §11 Reproducibility
Pointer to `REPRODUCIBILITY.md` (§4 below), kept in the paper as a compact
paragraph + link, per house convention.

---

## 4. Claim-tier cheat sheet (carry into every section verbatim, do not
paraphrase loosely — this is the list a reviewer-facing tier audit checks
against before submission)

| Claim | Tier | Source |
|---|---|---|
| Synthetic rank-K necessity (razor cliff, k=31→32) | **CONFIRMED** (causal) | `CAUSAL_RANK` §12.8.5 |
| Real-data rank-K necessity (RD-1, graded staircase) | **CONFIRMED**, verbatim: "RD-1's branch (b), causal necessity, is CONFIRMED at all four K tested (8, 16 primary, 24, 32)" — with the mandatory qualifier "graded across a window several ranks wide, not a single-step threshold" | `REALDATA` §17.9/§17.11 |
| Embedding-source interpolation arms (i/ii/iv vs. iii) | **causal, premise-conditional**; bundled (geometry+trainability) unless the i-vs-iv contrast is named, which is geometry-only | `RD_EXACTNESS` §7 |
| i-strong existence proof | **diagnostic tie-breaker**, NOT a standalone causal headline (two-variable by design) | `RD_EXACTNESS` §7 |
| §3's closed-form reconstruction, Test A | **β-idealization adequacy diagnostic only** — no explanatory claim | `RD_EXACTNESS` §7 |
| §3's closed-form reconstruction, Test B | **structural, deterministic reproduction** — strong non-causal explanatory claim, not causal | `RD_EXACTNESS` §7 |
| Wave F soft-arm interventions | **causal, premise-conditional** on the intervention variable; demonstration claim ("fix moves frontier") is **FALSE**, stated plainly | `RD_EXACTNESS` §15.7 |
| F-geo-3 demonstration claim ("fix moves frontier") | **causal, premise-conditional**, TRUE at K=16 (3/3 admissible, bar ≥0.8 HIT); at K=32 the ~45× improvement is at **full evidentiary tier** but the ≥0.5 headline bar is **NOT met** (mean 0.44, 1/3 seeds at the line) — never blur "admissible and large" into "bar met" | `RD_EXACTNESS` §14.10; `EXPERIMENT_LOG` 2026-07-04 + escalation cells |
| K=32 bar-miss attribution (outcome F: stable-not-just-orthogonal geometry) | **pre-registered mechanism verdict** — trained drift 0.9037 < 0.95 + resid≈0 + graded h-decay = the three-part outcome-F signature registered before the wave ran | `RD_EXACTNESS` §14.5/§14.8; escalation cells |
| F-geo-3 attribution claim ("therefore the geometry attribution was right") | **supported inference**, never proven by the demo alone — keep separate from the demonstration claim in every mention | `RD_EXACTNESS` §14.10 |
| F-geo-3 cross-arm seed-count comparability | pre-registered UNVERIFIED-comparability language now discharged to a **footnote**: escalation cells passed every substitute criterion including zero fallbacks | `RD_EXACTNESS` §14.10 |
| Wave 2 (Waves C+D, corpus generality) | verbatim: "**descriptive+interventional (RD-2, §6.3, §14.7) -- NOT premise-conditional causal**"; Q2's own conclusion further hedged as "**consistent with (not proof of)**" | `REALDATA` §6.3/§19, §19.3 |
| 2×2 existence-proof design (arms i/iv as "stable, not orthogonal" cell) | **causal, premise-conditional**, same tier as the embedding-source interpolation arms it reuses; the "both together" cell (i-strong) keeps its existing **diagnostic tie-breaker** tier and its pool-restriction caveat (32≤d_state) | `KEY_ANCHORING_DESIGN.md` §2.0 |
| §16.7 simulator correction | **meta-claim about instrumentation, not about the mechanism**: the retracted "K=32 overestimate" reading is WITHDRAWN; corrected, the simulator is **conservative** at K=32. The outcome-F attribution itself is **UNCHANGED** | `RD_EXACTNESS` §16.7; `experiment-runs/2026-07-04_geo3_simulator_recheck/` |
| Track C rung-1/rung-2/mixcontrol/wave-1ext attractor persistence (14M→98M→392M, matched mix) | **Tier 2 — descriptive + interventional**, a **pure-scale** claim (mix axis closed at both 14M and 98M, both within noise); still geometry-leg-only, 3 points with a 4th (1.31B) rung running | `SCALE_TRANSFER_DESIGN.md` §5.9/§5.10/Addendum |
| Track D production-scale measurement | **Tier 3 — measurement-only, no causal or interventional language.** Explicitly **NOT attributable** to the delta-rule write mechanism at this measurement tier (matched no-fixed-state negative control shows the same magnitude) | `SCALE_TRANSFER_DESIGN.md` §6.8 |
| Track B (geo3-in-LM) Wave −1 gate + measurement waves | **registered hard no-launch** (β-uniformity) **plus a second independent barrier** (duplicate-key numerical instability, 63× its own bar, probative positive control); the selectivity main effects that DID complete are **INCONCLUSIVE** (split verdict across corpora) | `SCALE_TRANSFER_DESIGN.md` §4.2; `TRACKB_REDESIGN.md` §14 |
| Stage-G H_e full manifest (sibling architecture — NOT a DeltaNet finding) | Closed: `h_b_factored_r4` does **not** rescue matrix composition; vector-composes/matrix-cannot inversion **seed-stable at h=3**; a hop-depth-2 phase-transition anomaly (one seed only) left **open, untriaged** | `STAGE_G_DESIGN.md` §15 |
| **Key-anchoring wave, behavioral result (K=32 h=4, all waves)** | **Upgraded this round from DESCRIPTIVE to INDEPENDENTLY-REPLICATED behavioral tier.** Real, reproducible, admissible under items 1–4 across all four waves; 9 total seed-runs across 3 distinct seed sets (Wave-1's 0/1/2, the confirm wave's same-seed reproduction, and the mechanism-tier wave's genuinely fresh 10/11/12 and 20/21/22) and 2 architecture variants — 6 of those 9 are the FIRST genuinely independent draws, upgrading from "reproduced" to "independently replicated." Range across all fresh-seed waves: 0.556–0.7141 vs. a freshly-measured 0.4105 baseline | `KEY_ANCHORING_DESIGN.md` §9.4/§9.6/§10.13.1; `experiment-runs/2026-07-05_keyanchor_wave/`, `.../2026-07-05_keyanchor_confirm/`, `.../2026-07-06_keyanchor_mech/` |
| **Key-anchoring wave, mechanism hypothesis (§1 interaction claim)** | **Outcome C — CONFIRMED at mechanism-tier rigor, a strictly stronger null than the round-4 confirm wave.** Per-entity `engaged_frac_v3` <14–41% and median `r_e` <0.25 at every K=32 leg, both architecture variants (global-λ candidate (d) and per-entity-λ candidate (d′)), immune to the anchor-norm/λ-degeneracy/power-manufactured-engagement objections that forced the round-4 rescue attempt's rejection | `KEY_ANCHORING_DESIGN.md` §10.13.1/§10.13.2/§10.13.5 |
| **Rev-6 λ-scaled engagement-threshold re-derivation** | **REJECTED by an independent attack round; CLOSED.** Not a claim-bearing result | `KEY_ANCHORING_DESIGN.md` §9.7 (CLOSED annotation); `KEYANCHOR_REV6_ATTACK.md` |
| **NEW — Candidate (d′), per-entity-λ architecture variant** | **Inconclusive/mixed** — same band (C) as candidate (d), partial-not-unanimous Spearman(λ_e, r_e) significance (2/3 seeds); closes the "parameter-sharing artifact" explanation in the negative (independent per-entity capacity converges to the same null) without itself demonstrating differential engagement | `KEY_ANCHORING_DESIGN.md` §10.13.2 |
| **NEW — Construction-stabilization account (why (d) works despite Outcome C)** | **CONFIRMED-BY-ABLATION** — a falsifiable, pre-registered prediction ("a frozen never-trained table should match (d)'s gain if constancy alone is the channel") was stated before the confirming data existed and held on fresh data under a hash-locked instrumentation stack. The strongest evidentiary tier this design's claim-tier system recognizes for a secondary mechanistic account | `KEY_ANCHORING_DESIGN.md` §10.13.4 (registration), §10.14 (verdict) |
| **NEW — Candidate (e)/(e-fp), frozen-random and frozen-frame-potential anchor tables** | **CONFIRMED, causal (ablation).** Both frozen, never-trained arms match or slightly exceed candidate (d)'s own learned-table mean (e: 0.7274; e-fp: 0.7413; (d): 0.6669); no frozen seed falls below (d)'s minimum. r_e negative control (median r_e negative for arm e, the pure-random init) passes at the strongest null in the design's history | `KEY_ANCHORING_DESIGN.md` §10.14.1/§10.14.2 |
| **NEW — K=48 capacity-curve extension** | **Informative negative, admissible.** Candidate (d) fully admissible 3/3, item 5 passes cleanly (ruling out "mechanism didn't engage"), both pre-registered explanatory channels (drift-space closure ~81%, value-Gram relief 0.626×) measurably active, bar still missed 0/3 seeds (mean 0.0215 vs. 0.024434). Completes the capacity curve as a cliff (~1.00→~0.65→~0.02 at K/d 0.25/0.5/0.75), not a smooth decline; the theoretical λ=1 ceiling declines gently (0.97→0.94→0.90) over the same range | `KEY_ANCHORING_DESIGN.md` §11.12 |
| **NEW — Key-anchoring program, overall status** | **PROGRAM COMPLETE.** Five waves plus one rejected rescore attempt, ≈55.83/80 GPU-h against its own budget cap. No further waves scheduled; the next open question (value-crowding at K/d=0.75) is named but not yet tested and would require a fresh design | `KEY_ANCHORING_DESIGN.md` (header); `STATE.md` 2026-07-07 |

---

## 5. Related-work map

1. **Nazari & Rusch (arXiv:2602.04852) and Sun et al. (arXiv:2602.02195),
   both Feb 2026.** Descriptive rank measurement of fast-weight states in
   pretrained LLMs, uncontrolled K, no necessity theorem, no causal
   intervention. We are the mirror image: controlled K, a provable LOWER
   bound, and a causal (train-time-then-eval-time) intervention. Cite as
   the paper's primary "descriptive vs. causal" foil.
2. **Nichani, Lee & Bietti (arXiv:2412.06538, ICLR 2025 Spotlight).**
   Closest prior theoretical construction — a rank-m hand-built existence
   proof for associative recall, but under DISCRETE ARGMAX decoding, where
   a rank-1 matrix already recovers ≈d associations. This is the exact
   reason our readout is pinned to exact-continuous cosine recovery, never
   argmax.
3. **Schlag, Irie & Schmidhuber (arXiv:2102.11174, ICML 2021).** Origin of
   "linear-attention state as associative memory."
4. **DeltaNet / Gated DeltaNet production lineage** and **DeltaProduct**
   (arXiv:2502.10297). DeltaProduct's "rank" is the PER-STEP TRANSITION
   matrix, a different object from the STATE's own rank this paper
   measures. Also cite Grazzi et al. (arXiv:2411.12537).
5. **Barnfield et al. (arXiv:2605.05189).** Proves capacity thresholds for
   STATIC rank-constrained associative memories. Our delta lives entirely
   in the recurrent/causal/compositional part.
6. **Based / Arora et al. (arXiv:2402.18668, Thm F.1) and Zoology (Arora
   et al., arXiv:2312.04927).** Based bounds total state SIZE; justify
   explicitly why RANK, not size, is the binding resource under our pinned
   linear-unbind readout.
7. **M²RNN (Mishra, Tan, Stoica, Gonzalez, Dao; arXiv:2603.14360).** The
   single closest prior-art paper. Their own comparison table has a plain
   vector-state GRU matching the matrix model on their composition task,
   Thm 1 is representational/existence, not necessity, and they run NO
   causal rank intervention. **Caveat carried into the writing process:**
   verify all specifics before external citation; abstract-level sources
   only so far.
8. **Superposition / distributed representation literature.** Elhage et
   al. (Toy Models of Superposition); Plate 1995 (HRR); Frady, Kleyko &
   Sommer 2020 (Resonator Networks) — cite briefly to scope that their
   capacity notion is DIFFERENT from the exact-rank framing this paper
   uses.
9. **Prior chapter in this project's own line of work
   (Task D/E, `matrix-thinking/chapter2/`).** The from-scratch, bespoke-
   attention-encoder version of the identical rank-recruitment result,
   including the depth-amplification law this paper imports directly and
   the earlier, now-superseded ProsQA/matrix-CODI negative result
   (workshop paper) that motivated pinning the readout to exact-continuous
   recovery in the first place.
10. **This paper's own live follow-on and scale program, cited for
    completeness, not as external related work.** (a) `KEY_ANCHORING_
    DESIGN.md` is the direct continuation of §7/§8's outcome-F residual —
    now fully CLOSED across five waves. Cite its complete, final outcome:
    a real, independently-replicated K=32 h=4 behavioral gain
    (0.4105→0.61–0.71, 9 seed-runs, 3 seed sets, 2 architecture variants)
    at behavioral tier, with the entity-alignment mechanism hypothesis
    itself resolving to **Outcome C** across two mechanism-tier waves (not
    confirmed, immune to a rejected rescue attempt), the actual mechanism
    resolved by a pre-registered ablation to **constancy-suffices**
    (CONFIRMED-BY-ABLATION — a frozen, never-trained anchor table matches
    or exceeds the learned one), and the fix's operating regime
    **measured and bounded** by a completed capacity-curve extension
    (works at K/d≤0.5, an informative-negative cliff at K/d=0.75, both
    of the design's own explanatory channels checked and found
    active-but-insufficient at the miss). Never round the behavioral gain
    up into a confirmed entity-alignment mechanism, and never let the
    Outcome-C null read as a failure of the fix demonstrated in §7/§8,
    which is a separate, already-evidenced claim resting on outcome F and
    the 2×2. (b) `SCALE_TRANSFER_DESIGN.md`'s Track C (a **pure-scale**
    14M→98M→392M attractor-persistence ladder, Tier 2, mix-axis confound
    closed at both tested scales; a 1.31B rung running, the paper's one
    remaining open readout) and Track D (production-model measurement,
    Tier 3, non-attributable) are this paper's own scale-generality
    program, cited in the Limitations section rather than the results
    body. Track B (installing the SAME fix mechanism inside a
    free-running-text LM) is a third scale-generality axis specifically
    for the fix itself: it is double-barred before any interaction claim
    can be tested, and its own selectivity-only readout is INCONCLUSIVE —
    cited in Limitations, not as a positive result. (c) A sibling result
    from a DIFFERENT architecture family in this same project
    (`matrix-thinking/chapter2/`'s matrix-native model, Stage-G's H_e
    task-swap manifest, CLOSED at all 6 cells) found the OPPOSITE
    separation on a composition-heavy corpus: a vector baseline composes
    in-context while the from-scratch matrix architecture does not, and
    the Kronecker-separable projection fix that recovers most of Stage G's
    per-FLOP quality gap does NOT rescue this compositional gap. This is
    explicitly NOT a DeltaNet finding and must never be cited as if it
    were — noted here only because it is part of the same project's live
    evidence about when structured/compositional state representations do
    and do not compose, and a reviewer who has seen one might ask about
    the other.

---

## 6. Limitations section content

State every item below in the paper's own limitations section, undiluted:

1. **Small scale.** `d_model` in the low hundreds, `d_state=64`, models
   sub-1M–low-single-digit-M parameters. No claim here has been tested at
   a scale where these mechanisms might behave differently.
   **Partial evidence, not a resolution:** a controlled 14M→98M→392M
   scaling ladder (Tier 2, geometry-only, a clean 3-point read on a single
   held-fixed corpus mix) finds the attractor does NOT dissolve with
   scale — it WORSENS monotonically (span-fraction 0.248→0.344→0.389), a
   **pure-scale** claim. A fourth, 1.31B-parameter rung is running as this
   is written, but this item's core claim (untested at production scale)
   stands until that rung reports and a compositional-recovery
   cross-check, not just geometry, exists at scale.
2. **Single architecture family.** Everything is vanilla, single-head,
   single-layer DeltaNet with the standard delta rule. Gated variants,
   DeltaProduct's multi-Householder generalization, and other fast-weight
   families are untested — flag explicitly that the H=1/single-layer
   choice was a deliberate gate-simplicity decision, not evidence the
   phenomenon is H=1-specific (ReserveMH's result that every head
   independently recruits full rank at H∈{2,4} is suggestive but was run
   on the synthetic task).
3. **The h=21 literal-depth decay remains unexplained and unfixed.**
   F-geo-3 demonstrably does not touch it.
4. **The K=32 headline bar is not met, and its named follow-on has now run
   to completion with a fully resolved, if more complex than hoped,
   result.** The fix demonstration is fully admissible at BOTH K, but the
   pre-registered ≥0.5 headline bar at K=32 is narrowly missed: h=4 mean
   0.44, 1/3 seeds at the line. The miss is attributed to outcome F
   (trained drift 0.9037 in the pre-registered HIGH band).
   **`KEY_ANCHORING_DESIGN.md`'s learned anchor-table follow-on, designed
   to close that residual, has now run to completion across four waves.**
   It lifts K=32 h=4 recovery from a freshly-measured 0.4105 baseline to
   0.61–0.71 (independently replicated, 9 seed-runs across 3 seed sets, 2
   architecture variants). But its own per-entity engagement instrument
   reads under 14–41% engagement at every K=32 leg, confirming Outcome C
   ("mechanism not engaged") at increasing evidentiary rigor across two
   waves, immune to a rejected rescue attempt. **The actual explanation
   was then found and confirmed by ablation:** a frozen, never-trained
   anchor table (random or frame-potential init) at the same blend weight
   matches or exceeds the learned table's own gain — constancy in the
   blend arithmetic, not entity-specific learned content, is the active
   ingredient; SGD's only demonstrated role is tuning the blend weight.
   **The fix's operating regime is now measured, not open-ended:** a
   capacity-curve extension to K=48 (K/d=0.75) shows the identical fix
   fully admissible, both of its own named explanatory channels
   measurably active, and the bar still missed 0/3 seeds — the curve is a
   cliff (~1.00→~0.65→~0.02), not a smooth decline. State plainly: the
   behavioral K=32 h=4 gain is real and its cause is now known
   (constancy, confirmed by ablation); the entity-alignment hypothesis
   originally proposed for it is rejected; and the fix's regime is bounded
   at K/d≤0.5. This item, open at round 4, is now closed.
5. **The K≈d/2 structural boundary is now a measured cliff, not merely
   observed.** The K=48 rider originally showed the frontier breaking down
   further with i-strong's own dimensional guard correctly refusing to
   run there. The completed capacity-curve extension (item 4 above) turns
   this into a fully measured three-point curve with both of the design's
   own candidate explanatory channels checked directly and found
   active-but-insufficient at K=48 — the open question is no longer "does
   it break past d/2" (yes, measured) but "why does the break happen so
   sharply, between K/d=0.5 and K/d=0.75, when the theoretical ceiling
   itself declines only gently over the same range" — the design's own
   named, not-yet-tested candidate for this is value-crowding (more
   entities packed into the same fixed value dimensionality reducing
   per-entity value-recovery headroom independent of key-side
   conditioning), which would require a fresh design to test.
6. **Scale-transfer to a full pretrained LLM checkpoint is now PARTIALLY
   measured, and the fix mechanism's transfer to free-running text is
   separately tested and blocked.** Wave 2 (Waves C+D) establishes the
   truncation-damage phenomenon generalizes to naturalistic LM pretraining
   at small scale (descriptive+interventional tier only). Track C/D go
   further: (a) on a controlled 14M→98M→392M parameter ladder, on a single
   held-fixed data mix, the write-geometry attractor does not dissolve —
   it WORSENS monotonically, a **pure-scale** finding, not yet extended to
   compositional recovery at scale, with a 1.31B-parameter fourth rung
   running; (b) a much larger version of the SAME non-orthonormal
   write-geometry signature is measurable directly in production-scale
   pretrained fixed-state models; (c) but a matched no-fixed-state
   negative control shows the SAME magnitude, so (b) is explicitly **NOT**
   attributable to the delta-rule write mechanism specifically at this
   measurement tier; and (d) a companion attempt to install the SAME fix
   mechanism inside a free-running-text LM (Track B) is blocked before any
   interaction claim can be tested — a duplicate-key numerical-stability
   smoke overshoots its bar by 63× with a probative positive control,
   compounding the original β-uniformity gate refusal. Net honest
   statement: the phenomenon persists where it can be controlled for and
   now scales purely with parameter count (a), a phenomenologically
   similar signature exists at production scale (b) but causal
   attribution there remains open and confounded (c), and the fix itself
   has not yet been shown to transfer to a free-running-text setting at
   all (d).
7. **The admission-stack asymmetry, now a footnote-level limitation.**
   Because F-geo-3 makes the standard premise-validity instruments
   tautological, a substitute pre-registered criterion applies to its
   seeds. With every substitute criterion passed at both K, this drops
   from a headline evidentiary asterisk to a footnote — but the paper
   still names both stacks and both realized pass rates wherever geo3 and
   non-geo3 seed counts appear side by side.
8. **The 2×2's "stable" cells lean on existence-tier or bundled arms, and
   this is no longer an open dependency — it is a closed, mixed answer.**
   The 2×2 (Fig. 5) that demonstrates orthogonality and stability are
   jointly necessary uses i-strong (pool-restricted to 32≤d_state) for the
   "both together" cell and frozen-embedding arms for the "stable, not
   orthogonal" cells. The key-anchoring program was built specifically to
   test whether a full-vocabulary, gradient-discovered anchor mechanism
   can reach the "jointly sufficient" regime, and it has now run to
   completion across four waves with a final, closed answer: the wave
   produces a real K=32 h=4 behavioral gain (0.4105→0.61–0.71) but NOT via
   the hypothesized entity-alignment channel (Outcome C, confirmed twice);
   the gain's actual cause is confirmed by ablation to be constancy in
   the blend arithmetic, not learned content; and this alternate
   mechanism's own regime is bounded (works at K/d≤0.5, fails at
   K/d=0.75). So: the 2×2's "jointly sufficient" corner still rests on the
   hand-built i-strong pin for a *causal* demonstration at K=32; the
   *practical*, SGD-adjacent route that was tested to reach that corner
   does not use the hypothesized mechanism, works via a simpler
   (constancy-only) route instead, and that alternate route's own working
   regime is now known. This is a closed, three-part answer, not an open
   dependency — state it as such.
9. **The key-anchoring program is now fully complete, and its complete
   arc is part of this paper's evidence base.** Every claim in §7/§8 about
   "the residual mechanism" (outcome F, the 2×2) still rests on the
   outcome-F attribution and the 2×2, not on the anchoring program's
   numbers, for the causal MECHANISM claim about why F-geo-3 misses its
   K=32 bar. What the anchoring program contributes, now completely: a
   real, independently-replicated K=32 h=4 behavioral improvement; a
   rejected entity-alignment hypothesis (Outcome C, confirmed twice); a
   confirmed alternate explanation (constancy suffices, by ablation); a
   measured capacity boundary (K/d≤0.5 works, K/d=0.75 does not); and a
   demonstrated instance of this project's own gating discipline both
   catching an attempted post-hoc rescue of a null result AND
   pre-registering and then confirming a falsifiable account of a
   surprising positive result. Report all of this, not just the
   flattering parts.

---

## 7. Reviewer-attack pre-mortem (16 hardest reviews + our answers)

1. **"K=32 missed its bar, and your own registered simulator predicted
   0.77 there while you measured 0.44 — isn't the outcome-F 'attribution'
   just a post-hoc story rescuing a failed prediction, and the simulator
   demonstrably miscalibrated?"**
   Answer, three parts. (a) The launch gate was pre-registered on K=16
   ONLY; the K=32 prediction was explicitly registered as non-gating.
   (b) The attribution rests on the three-part outcome-F signature
   pre-registered with numerically pinned bands: resid≈0 + trained drift
   in the HIGH band (measured 0.9037 < 0.95) + graded h-decay steeper at
   higher K. All three hold. (c) The fallback-irrelevance check eliminates
   the one competing within-run account (solver noise).
2. **"Your 'substitute admission stack' for F-geo-3 was invented after
   you saw that the standard one was tautological — isn't that exactly
   the kind of post-hoc criterion-shopping the field is trying to stop?"**
   Answer: the substitute stack was pre-registered BEFORE any Wave 1
   F-geo-3 data existed, specifically in response to an independent
   attacker's finding, and its comparability to the standard stack is
   explicitly flagged UNVERIFIED rather than assumed.
3. **"Is 'graded, not razor' on real data just a fancy way of saying the
   causal effect is weak / noisy?"**
   Answer: no — the eval-truncation ceiling is reached EXACTLY at k=K at
   every tested K; only the WIDTH of the transition below the ceiling
   differs. The causal necessity claim does not rest on cliff sharpness.
4. **"You ran embedding-source interpolation and found geometry is
   causally irrelevant to the ATTRACTOR — but then you spend three more
   sections fixing geometry. Isn't that a contradiction?"**
   Answer: no — "irrelevant" refers to the RAW INPUT embedding geometry;
   the attractor itself IS write-time key geometry, produced by the
   trained W_k/conv path regardless of input. The fix intervenes on the
   WRITE-TIME geometry directly.
5. **"i-strong bypasses the learned projection entirely — isn't that just
   showing a trivial oracle can solve an easy problem?"**
   Answer: i-strong is explicitly scoped as a DIAGNOSTIC TIE-BREAKER,
   never a standalone causal headline. It demonstrates the delta rule's
   OWN algebra can represent and use an exact solution at this (d,K) — an
   existence fact — which licenses treating the gap as an
   optimization-attractor problem.
6. **"Your proportionality/ε^h law is fit to only two soft-arm data
   points at one K — is this really a 'law' or just three numbers going
   down?"**
   Answer: correctly scoped as a READING, not a fitted/validated closed
   form — the qualitative shape is the load-bearing claim, imported from
   and consistent with the independently-derived depth-amplification
   mechanism and the exact `(K−h)/K` match on synthetic data. **Updated
   this round:** the same shape now also appears a fourth time, in the
   K=48 capacity-curve's own explanation — held-out four-hop recovery is
   itself a depth-4 probe, and the same compounding logic that explains
   why soft fixes' gains shrink with h is the same logic offered
   (disclosed as a reading, not a proof) for why small per-binding losses
   at K=48 collapse so sharply under 4-fold composition.
7. **"H=1, single layer — how do you know this isn't a degenerate corner
   case that says nothing about production Gated-DeltaNet configurations?"**
   Answer: stated directly in limitations as an open scope boundary; the
   ReserveMH result is offered as suggestive generality evidence but
   flagged as not yet transferred to the real-data exactness frontier.
8. **"Wave 2's corpus-generality result is a mixed, metric-dependent,
   layer-0-only signal, and Q4 runs BACKWARDS from the 'SGD recruits more
   rank as it learns' intuition your causal chapters built — doesn't that
   undercut the main story?"**
   Answer: no, because Wave 2 is explicitly scoped at a DIFFERENT claim
   tier answering a DIFFERENT question than the controlled causal
   chapters. We report the mixed and counter-intuitive findings in full.
9. **"Isn't 'name the mechanism, then demonstrate the fix' just standard
   ablation-study practice dressed up with pre-registration language?"**
   Answer: partially fair, and we don't oversell it. What we argue is
   specific: the gating discipline is what let a load-bearing failure
   mode get caught BEFORE any Wave 1 GPU spend, via a committed, versioned
   simulator with a pre-registered launch-read threshold. **A second,
   independent instance this round:** the same discipline registered a
   falsifiable account of a *positive* result (constancy suffices) with a
   stated pass/fail prediction BEFORE the confirming data existed, and the
   prediction held — a concretely falsifiable claim about a mechanism,
   not just a post-hoc story fit to already-observed numbers.
10. **"This whole story is six-plus sequential negative-then-positive
    results stacked on top of each other — isn't the paper just narrating
    your own research process rather than reporting a clean finding?"**
    Answer: the sequence IS the finding — the central claim could not have
    been established any other way. We frame the step structure explicitly
    as the proof structure in §1's contribution list, not as a research
    diary.
11. **"Your 2×2's 'stable' cell (i-strong) is a hand-built pin, not
    a learned mechanism — so all you've shown is that a pin composes,
    which nobody doubted. And your own follow-on tested this and landed
    on a rejected mechanism — doesn't that make the whole 2×2 story
    weaker, not stronger?"**
    Answer, updated to the program's final outcome. i-strong is a
    deliberate, disclosed, three-concession existence proof, never
    presented as a learned result. What the 2×2 actually licenses — that
    stability WITHOUT orthogonality buys nothing and orthogonality
    WITHOUT stability buys most-but-not-all of the gap — is fully
    supported by non-pin data; only the "and jointly SUFFICIENT" corner
    needed the pin. The anchoring program tested whether a learned
    mechanism could reach that corner via the hypothesized channel, and
    the answer, now final, is no — but it found something arguably more
    interesting: a real gain via a completely different, much simpler
    mechanism (constancy, not alignment), confirmed by a pre-registered
    ablation, with a measured operating boundary. This does not weaken the
    2×2's causal claim (which never depended on this wave) — it adds a
    second, independent finding about what actually stabilizes
    cross-episode keys in a real trained system, which happens not to be
    what was originally hypothesized.
12. **"You corrected your own simulator (§16.7) after finding a bug that
    flipped your reported K=32 prediction from an overestimate to an
    underestimate — doesn't that undermine every quantitative claim in
    §8/§16, not just the simulator figure?"**
    Answer: no — the retracted number was never used to authorize any
    spend and was never an input to the outcome-F attribution, which
    rests on three directly measured quantities. We report the bug, the
    fix, and the corrected direction in full.
13. **"You wrote a whole subsection trying to re-derive the very metric
    that killed your mechanism claim, and then a LATER subsection
    confirming a DIFFERENT explanation for the same gain — isn't this
    just moving the goalposts until something publishable emerges?"**
    Answer: distinguish the two carefully, because they are opposite moves.
    The first (Rev 6, the λ-scaled threshold re-derivation) was an attempt
    to rescue the ORIGINAL hypothesis by changing the metric — it was
    independently attacked, found to be threshold-shopping, and rejected;
    we report the attempt and its rejection in full, not deleted. The
    second (candidate (e)) did NOT change any metric, threshold, or bar —
    it tested a NEW, explicitly pre-registered hypothesis ("constancy, not
    alignment, is the channel") against the SAME engagement instrument and
    the SAME behavioral bar that rejected the original hypothesis, with a
    stated pass/fail prediction made before the data existed. One is a
    rejected attempt to save a claim by changing the ruler; the other is a
    fresh, falsifiable claim tested against the unchanged ruler and
    confirmed. We name this distinction explicitly in the paper because a
    careless reader could otherwise conflate them.
14. **"Your key-anchoring waves are described as reproducing/replicating
    each other — but the first two used the same seeds. Isn't 'independent
    replication' a stretch?"**
    Answer: no, and we are specific about which waves are which. The
    confirmatory wave (round 4) reused the initial wave's exact seed
    integers — a reproduction/verification check, not a fresh sample, and
    we said so explicitly at the time. The mechanism-tier wave (this
    round) drew six GENUINELY FRESH seeds (10/11/12, 20/21/22) never used
    anywhere in this program before, across two architecture variants,
    and all six cleared the bar. That is what licenses "independently
    replicated" — a materially different, stronger claim than the earlier
    "reproduced," and we mark the transition explicitly rather than let
    the stronger word creep in before it was earned.
15. **"Your candidate (e) result — a random, untrained table performing
    as well as a trained one — sounds almost too clean. Could this just be
    that K=32 h=4 is easy to hit for ANY reasonable anchor construction,
    making the whole ablation uninformative?"**
    Answer: this is exactly why the design registered BOTH a random-init
    and a frame-potential-init frozen arm, and why it built in the r_e
    negative control. If "any anchor works" were the full story, we would
    not expect the pure-random arm's own alignment readout to be
    negative — but it is (median r_e −0.13 to −0.24, the most decisively
    null reading anywhere in this program's history), which is exactly
    what an instrument correctly reporting "nothing to align to" should
    show. The result is not "anchors are magic" — it is "the specific
    hypothesized channel (learned entity alignment) is not what's
    happening, and a much simpler mechanical account (constancy in the
    blend) fully explains the behavioral gain without needing entity-
    specific content at all." We also do not claim this generalizes past
    K=32 or λ=0.58 — both are named as untested by this specific ablation.
16. **"Track B's numerical-instability finding sounds like an engineering
    bug, not a scientific result — why does a skip-rate overshoot belong in
    a paper about rank exactness?"**
    Answer: because it is pre-registered, itself, as a gate on a
    scientific question, and a positive control confirms it is not a
    spurious harness artifact — the duplicate-key regime this fix
    mechanism cannot tolerate is a real property of free-running text, not
    a coding mistake. We report it as a genuine boundary condition on
    where the demonstrated fix mechanism currently applies, the same
    epistemic status as the K≈d/2→d/4 capacity cliff.

---

## 8. Methodology framing (weave-in notes, not a section)

Per the task brief: the adversarial-gauntlet process (design → attack →
verify → build → audit → gated launch) is a contribution, but must be woven
in "without self-congratulation." Concretely:

- **Where it appears:** one paragraph in §3 (Setup), one paragraph in §7
  (The Fix — specifically the launch-gate decision, since that is where
  the process actually changed the outcome), and the Reproducibility
  section/artifact. NOT a standalone "Our Methodology" section.
- **Framing rule:** always pair a process claim with the SPECIFIC finding
  it caught or would have missed. Show, don't narrate.
- **What NOT to claim:** do not claim the process guarantees correctness,
  prevents all errors, or is itself novel as a general ML methodology.
- **A second concrete instance (round 4, unchanged):** an attempted
  post-hoc rescue of the key-anchoring wave's null mechanism result (a
  λ-scaled engagement-threshold re-derivation) was itself put through an
  independent attack round and rejected before it could be reported as a
  positive finding.
- **A third concrete instance, NEW this round, and arguably the sharpest
  one in the paper:** the same discipline was used PROSPECTIVELY, not just
  defensively. After Outcome C was confirmed a second time (mechanism-tier
  wave), the design team did not simply report a null and move on — it
  formulated a specific, falsifiable alternative account (construction-
  stabilization: constancy, not alignment, is the channel), stated exactly
  what result would confirm it and what result would falsify it, registered
  this BEFORE running the confirming experiment, and then ran it. The
  account held. This is a different and, we think, more scientifically
  interesting use of the gating discipline than catching an error: using
  pre-registration to convert "we don't know why this works" into a
  tested, falsifiable, and ultimately confirmed causal claim about
  mechanism — cite this alongside the F-geo-3 cross-episode-drift catch
  and the Rev-6 rejection as the paper's third and final illustration of
  the discipline doing real evidentiary work, not just process theater.

---

## 9. Results still landing (as of 2026-07-07) — updated this round

Per the task brief: list every wave that is currently running or
built-but-not-launched, with its pre-registered decision rule, so the
narrative is submission-ready modulo these specific pending readouts. Every
item that was "still landing" as of round 4 has now reported. **Exactly one
item remains genuinely open.**

1. **Key-anchoring program — FULLY CLOSED, no longer listed here as
   pending.** (Moved out of this section entirely this round; see §4's
   claim-tier table and §6 item 4/8/9 for the complete, final story.) For
   the record: five waves plus one rejected rescore attempt,
   ≈55.83/80 GPU-h against its own budget cap, program formally complete,
   no further waves scheduled. The one genuinely open follow-on this
   program itself named (value-crowding at K/d=0.75, `KEY_ANCHORING_
   DESIGN.md` §11.12) is a NEW, unregistered direction requiring a fresh
   design/attack/build/audit cycle — it is not a continuation of the
   existing design and is not "still landing" in the sense this section
   tracks.
2. **SCALE-TRANSFER Track C rung-1/rung-2/mixcontrol/wave-1ext — all
   HARVESTED and CLOSED, no longer running.** Together they closed the
   residual mix-axis confound at both 14M and 98M and yielded a clean,
   single-(extended)-mix, 3-point ladder: span-fraction 0.248 (14M) →
   0.344 (98M) → 0.389 (392M), monotonically increasing — a **pure-scale**
   claim. Geometry-leg-only; no compositional-recovery cross-check exists
   at any rung. A dense per-seed trajectory dataset for this ladder
   (`experiment-runs/2026-07-06_trajectory_probes/`) was produced this
   period purely for figure regeneration (Fig. 9 Panel A) — validated
   exactly against the archived harvests, no new claims.
3. **SCALE-TRANSFER Track C rung-3 (1.31B params) — THE ONLY ITEM STILL
   RUNNING.** Launched on GPUs 0–1 (token-matched to rung-2 at 1.5B
   tokens/run, per the user-signed-off Rev 2.2 amendment), `ALL_DONE`
   expected imminently (~19:00 UTC 2026-07-06 per the original estimate;
   this narrative is being finalized before that readout lands). Decision
   rule unchanged from §5.7: a fourth-rung reading that continues
   climbing, plateaus, or reverses the span-fraction trend are all
   pre-registered as informative — this is the literal completion of
   §5.7's 3-rung criterion (98M/392M/1.3B), now extended by one more rung
   than originally specified. **This is the single dependency this
   narrative has on a number not yet in hand.** When it lands: update Fig.
   9 Panel A with the fourth point (using the same `archived4_span_frac`
   convention and instrument as the dense trajectory dataset above), the
   §0/§1 thesis language ("a fourth rung running" → the actual reading),
   and §6 item 1/6 and §9 item 3's own text.
4. **Stage-G H_e full 6-cell manifest (sibling architecture — NOT a
   DeltaNet result) — CLOSED, all 6 cells complete.** Matrix baseline,
   matrix `h_b_factored_r4`, and vector baseline all completed 40,000/
   40,000 steps. Verdict: `h_b_factored_r4` does NOT rescue matrix
   composition; the vector-composes/matrix-cannot inversion is
   **seed-stable at hop-depth 3**; an open, untriaged anomaly at
   hop-depth 2 (one seed only, no proposed mechanism) is disclosed, not
   resolved. Cited here only per §5 item 10c's explicit scoping — never as
   a DeltaNet finding.
5. **SCALE-TRANSFER Track B (geo3-in-LM) — CLOSED.** Beyond the original
   β-uniformity no-launch, a redesigned Wave −1 stability smoke measured a
   63× overshoot of its skip-rate bar with a probative positive control —
   a second independent barrier that refused further training GPU-hours
   before they were spent. The selectivity main-effects readout that
   could still run completed 18 cells and reads **INCONCLUSIVE** (a split
   verdict across corpora, per the design's own combined rule). No further
   Track B wave is pending; a hard-zero-β variant remains a registered but
   unscheduled conditional follow-on requiring its own attack round.

---

## 10. Open items before this narrative can be finalized

1. **[RESOLVED, round 3+]** The K=32 `n_iter=20` escalation completed: all
   3 seeds fully admissible, h=4 = 0.504/0.416/0.390 (mean 0.44). Headline
   bar ≥0.5 narrowly MISSED; miss attributed to pre-registered outcome F.
2. **[RESOLVED, round 3+]** Real-data eval-truncation archive path
   confirmed: K=16 uses n=10 pooled seeds, K=32 uses n=7, K=8/24 use n=2
   each — note this seed-count asymmetry in the actual paper's methods/
   appendix.
3. **Re-verify M²RNN (arXiv:2603.14360) citation specifics** before the
   camera-ready bibliography — flagged in the source design doc as
   abstract-level-only verification so far. Still open, unchanged since
   round 3.
4. **[RESOLVED, round 4]** Exact §14 R2-4 gate mechanism and loss
   confirmed (the target-index bug, the anti-collapse NCE loss, the
   10/10 collapse-free rerun) — belongs in §3/§4's setup description.
5. **[RESOLVED, round 5 — this revision's own absorption.]** This round
   absorbed the key-anchoring program's complete closure: the
   mechanism-tier wave's independent replication and strengthened Outcome
   C (§0, §1, §3 §5/§7/§8 updates, §4 claim-tier rows, §6 items 4/8/9, §7
   pre-mortem #9/#11/#13/#14 rewrites, §9), candidate (e)'s
   confirmed-by-ablation constancy-suffices finding (same locations, plus
   Fig. 10 Panel C and pre-mortem #15, new), and the K=48 capacity-curve
   extension completing the fix's operating-regime characterization (§0,
   §1, §4, §6 items 4/5, §7 pre-mortem #6 update, new Fig. 11 as the
   program's headline figure). **Nothing in this narrative currently
   depends on a wave that has not yet reported a number, except rung-3
   (§9 item 3) — the single remaining open readout, stated as such
   everywhere it is mentioned, never presupposed in any earlier section's
   prose.**
6. **Round-5 self-flagged weaknesses, honestly, not by assumption (for
   whoever drafts the camera-ready):**
   (i) **The abstract and §0 thesis are now long and cover a fully-closed,
   five-wave sub-story (the anchoring program) in nearly as much detail as
   the paper's original three-section causal chain (attribution → soft
   fixes → structural fix).** This is the single biggest trim target for
   the actual submission: the anchoring program's four-part arc
   (behavioral gain → rejected mechanism → confirmed alternate mechanism →
   measured capacity boundary) is scientifically complete and interesting,
   but at submission length it risks reading as a second paper bolted onto
   the first. Recommend compressing the abstract's anchoring material
   (currently the Seventh–Ninth sentences) to 2-3 sentences total,
   trusting Fig. 10/11 and §7/§8's body text to carry the detail, and
   moving the four-part arc's full texture to the body/appendix rather
   than the headline claims.
   (ii) **Fig. 11 (the capacity-curve headline figure) and Fig. 10 (the
   three-panel mechanism-arc figure) overlap substantially in what they
   show at K=32.** A camera-ready editorial decision is needed on whether
   both survive as main-text figures (space permitting) or whether Fig.
   10 moves to the appendix with Fig. 11 as the sole main-text figure for
   this program, per Fig. 11's own note in §2 that it "should be
   considered for main-text placement over Fig. 10's more detailed
   breakdown." This narrative does not resolve that call — it is a
   layout/space decision, not a substance one.
   (iii) **The K=48 "value-crowding" candidate explanation is named but
   completely untested** — it is the design's own honest naming of what it
   does NOT yet know, not a finding. Make sure the paper's language around
   it stays at "named candidate, not yet tested" throughout, including in
   the abstract's closing "what remains unexplained" list — it would be
   easy to accidentally upgrade this into something that sounds more
   established than it is during final editing.
   (iv) **This is very likely the final round before rung-3 lands and
   this document needs one more, hopefully small, pass** — everything
   else in the arc that round 3/4 flagged as open is now closed. If rung-3
   reports before the actual submission deadline, only §9 item 3, Fig. 9
   Panel A, and the handful of "fourth rung running" phrases in §0/§1/§6
   need updating; no other section's substance depends on that number.
