# ICLR 2027 Narrative Architecture — DeltaNet Rank Recruitment vs. Exactness

**Status: DRAFT — round 3, ARC EXTENDED to the mechanism's two-ingredient
attribution and to scale. Grounded against `DELTANET_RD_EXACTNESS_DESIGN.md`
(§1–8, §14, §15, §16 incl. the §16.7 dated correction),
`DELTANET_REALDATA_DESIGN.md` (§14–19), `DELTANET_CAUSAL_RANK_DESIGN.md`
(§1, §2, §4, §12), `KEY_ANCHORING_DESIGN.md` (Rev 5, §1–§3),
`SCALE_TRANSFER_DESIGN.md` (§2 claim tiers, §3.8, §4, §5.9, §6.8),
`STATE.md`'s 2026-07-04 picture, and `EXPERIMENT_LOG.md` (2026-07-01 →
2026-07-04, including the Track B/C/D and Stage-G H_e entries dated
2026-07-04). Every number below is quoted from those sources — several
re-verified directly against the archived run JSONs during this revision —
not recomputed or estimated.

Round-2 ending (still current, restated): the F-geo-3 K=32 `n_iter=20`
escalation COMPLETED — all 3 seeds fully admissible (zero fallback steps),
h=4 `rec@0.9` = 0.504/0.416/0.390 (mean 0.44), numerically identical to the
n_iter=12 cells; the ≥0.5 headline bar narrowly MISSED (1/3 seeds at the
line), attributed to pre-registered outcome F (trained cross-episode key
drift 0.9037, inside the HIGH band <0.95).

**New this round, absorbed from the last three days of verdicts:**
(1) a 2×2 existence-proof design (`KEY_ANCHORING_DESIGN.md` §2.0, using two
new frozen-embedding arms plus already-archived data) shows orthogonality
and cross-episode stability are BOTH independently necessary and JOINTLY
sufficient — neither alone moves composition off the ~0.01 baseline,
together the pair saturates to 1.00; (2) a dated correction to the
pre-spend simulator (§16.7) traces the recorded "K=32 overestimate" to an
input-wiring bug (K=16's drift was silently reused for K=32's prediction)
— corrected, the simulator is CONSERVATIVE at K=32, and the outcome-F
attribution itself is unaffected because it never depended on the
simulator; (3) a learned-anchoring mechanism designed to let SGD supply
the missing stability ingredient itself has been designed, attacked across
five revisions, built, and CPU/smoke-verified — GPU launch is pending box
availability, not yet run; (4) the write-geometry attractor motivating
this entire mechanism study demonstrably persists on a controlled
14M→98M-parameter ladder, and a much larger version of the same signature
is measurable in production fixed-state LMs (RWKV-7 1.5B, Falcon-Mamba-7B)
— though a matched no-fixed-state negative control shows the same
magnitude, so production-scale attribution is an honest open non-finding,
not a confirmation. The paper's frame is unchanged at its core and
strengthened at its edges: the fix moves the frontier decisively at K=16
(bar hit 3/3), moves it ~45× at K=32 without clearing the pre-registered
bar, the residual gap has a named, two-part necessary-and-sufficient
mechanism (not just a single attribution), the follow-on that tests
whether SGD can find that mechanism unassisted is live, and the phenomenon
this all rests on is not a small-scale artifact.**

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
differentiable, per-episode structural orthogonalization mechanism
supplies the first ingredient alone and already restores near-perfect
compositional recall at the pre-registered minimum-publishable bar (K=16,
3/3 seeds) and improves the harder K=32 regime ~45× at full evidentiary
admissibility — narrowly missing its pre-registered headline bar for a
reason now doubly confirmed as the missing second ingredient (trained
cross-episode key drift, measured 0.90–0.94, inside the registered HIGH
band at both K), with a follow-on mechanism designed to let SGD supply
that second ingredient itself — attacked across five revisions, built, and
smoke-verified — now gated on a GPU slot, its outcome an open, live
question this narrative frames honestly rather than presupposes. The
attractor this entire story is about is not a small-model curiosity: it
persists, un-dissolved, on a controlled 14M→98M-parameter ladder (with a
392M-parameter rung running and a 1.3B rung queued), and a substantially
larger version of the same non-orthonormal write geometry is measurable in
deployed production fixed-state language models (RWKV-7 1.5B,
Falcon-Mamba-7B) — architectures marketed specifically for long-context,
cache-free inference, where a compositionally-imprecise fixed state is a
live deployment-reliability question, not only an academic one — though at
present that production-scale signature cannot yet be distinguished from
generic trained-LM key anisotropy (a matched no-fixed-state negative
control shows the same magnitude), an honest non-attribution reported as
such, not a confirmation dressed up as one.**

Shorter version for the abstract's first line: *SGD reliably finds the rank
a task requires; it does not reliably find the geometry that rank needs to
compose exactly — that gap has a cause with two necessary parts, a
demonstrated fix for one of them, a named and pre-measured residual for the
other, a live test of whether SGD can be made to find that second part
unassisted, and evidence the underlying attractor is not a toy-scale
artifact.*

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
> bottleneck and its stabilization as the direct follow-on. Sixth, we
> sharpen this attribution with a 2×2 existence-proof design cutting the
> same archive along two independent geometric axes (orthogonal × stable):
> neither ingredient alone moves held-out four-hop recovery off a
> 0.005–0.011 floor statistically indistinguishable from the fully-learned
> baseline (0.009), while a hand-built key that is both orthogonal and
> stable saturates to 1.00 at h=4 *and* h=7 — the fix above supplies
> exactly one of two necessary ingredients. We report a learned-anchoring
> mechanism designed to let SGD supply the second (cross-episode
> stability) itself — attacked across five revisions, built, and
> smoke-verified — as an open, live test pending a GPU launch, not a
> result, and we correct our own pre-registered simulator: an audit found
> the recorded K=32 "overestimate" was an input-wiring bug (K=16's
> measured drift was silently reused for K=32's prediction); corrected,
> the simulator is conservative, not miscalibrated, at K=32, and the
> outcome-F attribution above is unaffected because it never depended on
> the simulator's point prediction. Seventh, we show the write-geometry
> attractor motivating this entire study is not a small-model artifact: it
> persists on a controlled 14M-to-98M-parameter scaling ladder (chunk-level
> key-Gram deviation 21.9 → 27.8, both well above a random anchor of 7.9
> and below full collapse at 63.5), and a substantially larger version of
> the same non-orthonormal write geometry is measurable in production-scale
> fixed-state language models (RWKV-7 1.5B, Falcon-Mamba-7B; 3–6× their own
> random anchors) — but a matched no-fixed-state negative control (a
> standard softmax-attention model) shows the same magnitude, so
> production-scale attribution to the delta-rule write mechanism
> specifically remains open, an honest non-finding reported as such. We
> report the full pre-registration, attack, and gating trail as an
> artifact, and discuss what remains unexplained: a separate iteration-depth
> decay (h=21) that key orthogonalization does not touch, an unresolved
> K≈d/2 structural boundary, whether learned anchoring closes the K=32
> headline bar, and whether the production-scale signature is ever
> attributable to fixed-state writes specifically.

*(Word count target ~300–340; trim during actual drafting — the Sixth/
Seventh additions are the least compressible content in this revision and
are the first candidates to shorten, not cut, since both name genuinely
open questions rather than resolved claims. No load-bearing placeholders
remain for anything already decided; the two new open items above are
explicitly flagged as open, not placeholders.)*

---

## 2. Figure plan (9 figures, exact data sources — Fig. 9 ADDED this round)

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
necessary, jointly sufficient.** REVISED this round from a 2-bar comparison
into the full 2×2 the archive already supports (`KEY_ANCHORING_DESIGN.md`
§2.0). A 2×2 grid, axes {orthogonal, not-orthogonal} × {stable
(context-free), not-stable (episode-conditional)}, cells populated with
`rec@0.9` at h=4 (and h=7 in the caption), K=32, all values re-verified
directly against the archived per-seed JSONs this revision:
  - **stable + orthogonal** (i-strong, hand-built context-free orthonormal
    key, pool-restricted to 32 ≤ d_state — an existence-proof concession,
    stated in the caption, not hidden): h=4 **1.0000**, h=7 **1.0000**, all
    3 seeds.
  - **stable + not-orthogonal** (frozen-embedding arms i/iv — a fixed input
    table, but the trained `W_k` path still de-orthogonalizes it, §16
    "Wave 1 ATTRIBUTION VERDICT"): h=4 **0.0075–0.0114** (arm i,
    seed-0 0.0093) and **0.0049–0.0079** (arm iv, seed-0 0.0079), h=7
    **≈0.0000** both arms, all seeds.
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
  disclose the i-strong cell's pool restriction explicitly (§2.1 of
  `KEY_ANCHORING_DESIGN.md`: this cell is a diagnostic tie-breaker proving
  the mechanism exists, never a claim about what a full-pool trainable
  anchor can reach — that ceiling is a separate, computed number, see the
  "results still landing" section).
- Data: `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wave1/`
  (arms i, iv, iii-β — files `w1_rdx_K32_armi_s{0,1,2}.json`,
  `w1_rdx_K32_armiv_s{0,1,2}_rho0p225.json`, `w1_rdx_K32_armiii-beta_s*.json`,
  h=4/h=7 read directly from each file's final-checkpoint `M3_held_out`
  block this revision), `experiment-runs/2026-07-03_deltanet_rd_waves/
  exactness/wavegeo3/` (geo3-alone), and
  `KEY_ANCHORING_DESIGN.md` §2.0's own extraction table.

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
drops to a footnote in the caption (all substitute criteria passed
including zero fallbacks; both stacks and realized pass rates still named,
per pre-registration). **Corrected-simulator framing (this round, §16.7's
dated correction):** do NOT caption this figure with the retracted "the
simulator overestimated K=32" reading — that number (0.7734) was computed
by feeding K=32's prediction K=16's own measured drift (a shared-scalar
wiring bug in `geo3_drift_diagnostic.py`/`geo3_simulator.py`, GPU-verified
on the box, `experiment-runs/2026-07-04_geo3_simulator_recheck/`). Fed its
OWN drift (0.9037), the simulator predicts 0.06–0.09 — it is CONSERVATIVE
at K=32, undershooting the measured 0.4368 rather than overshooting it. If
the caption mentions the simulator at all, state it this way: the
pre-registered launch gate correctly authorized the wave using K=16's
prediction only (the gate's registered cell), the K=32 prediction was
explicitly non-gating and informational, and the outcome-F attribution
itself never rested on the simulator's point estimate — it rests on the
three measured signatures in §16.6 (resid≈0, drift <0.95, graded steeper
h-decay at higher K), all measured directly, none simulated.
- Data: `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wavegeo3/`
  (`EXPERIMENT_LOG.md` "F-geo-3 WAVE VERDICT" 2026-07-04 + the completed
  n_iter=20 escalation cells in the same archive directory).

**Fig. 8 (appendix / generality figure) — Wave 2: the phenomenon is not an
artifact of the synthetic probe grammar.** Truncation-damage-vs-k curves
(nats/token), reasoning (OpenR1-Math) vs. narrative (WikiText) corpora, at
k∈{8,16,24,32,48,64}, showing openr1 consistently more truncation-sensitive
at low-to-moderate k, converging to the same noise floor at k*=48 for BOTH
corpora: k=8 raw 0.1018±0.0046 (openr1) vs. 0.0885±0.0067 (wikitext); k=16
raw 0.0399±0.0035 vs. 0.0339±0.0026; k=24 raw 0.0145±0.0039 vs.
0.0098±0.0008; by k=32 the gap is negligible (0.0034 vs. 0.0032). Caption
must carry two named confounds, not smooth them over: (1) the gap holds
within every token class at k=8 including plain `word` tokens (0.1267 vs.
0.0961), partially but not fully addressing a symbol-density confound
(no symbol-matched third arm was run); (2) at every non-trivial k,
home-trained models take MORE truncation damage than cross-domain-trained
models on the SAME eval text — the headline gap is entangled with domain
competence, not isolated to text content alone, stated as an open confound
in the figure caption itself. Offered as descriptive/interventional support
(RD-2 tier, explicitly NOT premise-conditional causal, and Q2's own
conclusion is hedged further as "consistent with, not proof of") that
binding-rank sensitivity generalizes past the hand-built probe task to
naturalistic LM pretraining.
- Data: `experiment-runs/2026-07-04_lm_rd_wave2/{waveC,waveD}/`, analysis
  via `analysis_lm_w2.py` (`DELTANET_REALDATA_DESIGN.md` §19.3).

**Fig. 9 (NEW this round, appendix / generality figure) — The attractor
persists with scale, and a larger version of it exists at production
scale, unattributed.** Two-panel figure. **Panel A (Track C rung-1
harvest, `SCALE_TRANSFER_DESIGN.md` §5.9):** pooled chunk-level key-Gram
deviation at two points on a controlled parameter ladder, same
architecture family, same instrument, matched corpora mix: 14M-param
control **21.93 ± 5.90** vs. 98M-param rung-1 **27.82 ± 12.87**, plotted
against a K=64/d=64 random anchor (**7.94**) and full collapse
(**63.50**) as reference lines — the deviation sits well above random and
below collapse at both points and gets *slightly worse, not better*, from
14M→98M (Tier 2, descriptive+interventional per `SCALE_TRANSFER_DESIGN.md`
§2; explicitly a 2-point read, geometry-only, no compositional-recovery
cross-check at this scale yet — caption must state this, not imply a
3-rung trend). **Panel B (Track D Phase 1, `SCALE_TRANSFER_DESIGN.md`
§6.8):** the same chunk=64 Gram-deviation statistic measured (Tier 3,
measurement-only — no causal or interventional language) in three
production-scale pretrained checkpoints: RWKV-7 1.5B (**43.5–44.0**),
Falcon-Mamba-7B (**49.9–50.2**), and a no-fixed-state negative control,
Qwen2.5-1.5B, ordinary softmax attention (**46.0–48.5**) — all three
cluster together, RWKV-7 and the negative control statistically
indistinguishable. Caption must carry the honest verdict, not the
tempting misreading: the geometry Fig. 4/5 attribute to the delta-rule
write mechanism is measurably present and larger at production scale, but
because the no-fixed-state control matches it, the signature is dominated
by generic trained-LM key anisotropy (massive activations, Sun et al.
2024, arXiv:2402.17762) at this measurement tier, and is **NOT**
attributable to the delta-rule family specifically without a sharper
instrument (e.g. an outlier-robust/centered variant, reported alongside
raw in the same source table) — an honest, registered non-finding, offered
as motivation for why the small-scale causal chain (Fig. 4/5/7) matters,
not as evidence the causal chain replicates at scale.
- Data: Panel A — `experiment-runs/2026-07-04_trackc_rung1/`
  (`SCALE_TRANSFER_DESIGN.md` §5.9; `EXPERIMENT_LOG.md` "SCALE-TRANSFER
  Track C Wave 1 (rung-1) harvest," 2026-07-04). Panel B —
  `experiment-runs/2026-07-04_track_d/` (`SCALE_TRANSFER_DESIGN.md` §6.8;
  `EXPERIMENT_LOG.md` "SCALE-TRANSFER Track D Phase 1," 2026-07-04).

---

## 3. Section-by-section argument flow (maps to `main.tex` skeleton)

### §1 Introduction
- Open on the tension: descriptive rank-measurement papers (Nazari & Rusch;
  Sun et al., both Feb 2026) show fast-weight state rank is bounded by
  sequence length in pretrained LLMs, but never show SGD is *forced* to use
  that rank, nor that the resulting operator is *exact*.
- Contribution bullets (claim-tiered from the first sentence, per the
  house rule "overclaiming is the one unforgivable sin"):
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
     (not value geometry, not NCE crowding, not kernel precision) via a
     controlled embedding-source intervention, plus an existence proof
     that the exact solution is architecturally reachable.
  5. A demonstrated (not just diagnosed) fix — differentiable per-episode
     key orthogonalization — that clears the pre-registered bar at K=16
     (3/3) and improves the harder K=32 case ~45× at full admissibility,
     with the narrow headline-bar miss itself attributed to a second,
     pre-registered, pre-measured mechanism (cross-episode key drift,
     outcome F) — the causal chain closes with no unexplained residual.
  6. The process itself as a minor, explicitly-not-self-congratulatory
     contribution: every wave above was pre-registered, attacked by an
     independent reviewer, and gated on a predicted outcome before any GPU
     time was spent — released as an artifact (§ Reproducibility) because
     the negative results (Wave F) are exactly as load-bearing to the
     story as the positive ones.

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
  argmax — the Nichani/Lee/Bietti-motivated design choice), exact recovery
  of K independent bindings requires `rank(S_T) ≥ K`. State clean premise
  (learned β_j ≠ 0 suffices, β=1 not required) and the two escapes closed
  by construction: position-decomposition (single `d×d` state, no P=1
  slotting) and the architecture-specific multi-head/multi-layer
  reincarnations (H=1 primary gate; ReserveMH later lifts the H=1
  qualifier — 6/6 cells, every head at both H∈{2,4} independently recruits
  full rank K=32, no distribution across heads).
- Short, non-self-congratulatory paragraph on method: every claim below
  was pre-registered (hypothesis + success bar stated before data),
  attacked by an independent reviewer round, and — for the constructive
  fix — gated on a predicted outcome from a committed simulator before any
  Wave 1 GPU spend. This discipline is what makes the negative results
  (Wave F) trustworthy as negatives rather than "we didn't try hard
  enough," and is offered as a template, not a boast. Concrete instance to
  cite here (not just assert): a mini-audit caught a target-INDEX bug in
  the real-data grammar (`forward()` gathered the scoring target one hop
  past the queried answer, `π^{h+1}(a)` instead of `π^h(a)`) that made the
  binding task unsatisfiable as originally specced — the resulting
  "1.000 recovery" was a value-collapse artifact (entity-subspace rank
  ≈1.0, value salvage ≈0.000, 0/10 seeds premise-valid), caught by the
  premise-gate instruments at every one of its 10 seeds and correctly
  never written up as a finding, before the fix (§ Phenomenon below).
- **Two "primary K" values, kept distinct on purpose, stated once here so
  it is never ambiguous downstream:** `K=16` is the *causal-necessity*
  program's primary cell (chosen at the §14.7 gate specifically because
  K=32 already showed a real, non-budget sub-exactness plateau at
  0.78–0.80 recovery — an anomaly, not a clean baseline for a force-rank
  ablation); `K=32` is the *exactness-mechanism/fix* program's headline
  cell (chosen because it is exactly that anomaly — the most dramatic,
  most publication-relevant exactness failure, sitting at `K=d_state/2`).
  Both choices are principled and pre-registered; a reviewer who notices
  the two programs cite different "primary" K without this framing will
  read it as an inconsistency rather than the deliberate choice it is.

### §4 The Phenomenon: Rank Recruitment Is Reliable, Exactness Is Not
- Synthetic baseline (razor cliff): `DELTANET_CAUSAL_RANK_DESIGN.md` §12.3
  (Wave 0 saturates to `rec@0.9=1.0000` at every tested `(d,K)` cell, no
  late-transition regime at all) + §12.4 (no-inflation: entity-subspace
  rank = whole-matrix rank = K to four decimals) + §12.8.3 (the k=31→32
  cliff, quantitatively matching the theoretical `(K−h)/K` bimodal
  per-mode-drop prediction to 0.001–0.003 at every hop tested including
  h=21).
- Real-data transfer (RD-1, CONFIRMED): same architecture, same delta rule,
  same provable bound, transplanted to GPT-2-tokenized language via a
  grammar+embedding path (`DELTANET_REALDATA_DESIGN.md` §5). Headline
  frontier (Fig. 2 numbers above). Rank recruitment intact (Fig. 3: 94–99%
  of K). Eval-truncation staircase is real and causal (RD-1 CONFIRMED — the
  same instrument, same verdict direction) but graded: ceiling reached
  exactly at k=K at every K tested — but the *ceiling itself* is
  sub-1.0 and falls with K (1.000 at K=8 → 0.995 at K=16 → 0.938 at K=24
  → 0.795 at K=32, §17.4) — and the transition below ceiling is a window
  several ranks wide (2–3 at K=8, 12–13 at K=32, steepest single-step jump
  shrinking from +0.375 at K=8 to +0.066 at K=32) — state both the
  sub-ceiling fact and the widening-window fact as the chapter's own
  qualifier, never silently dropped.
- **Narrative connective tissue, worth stating explicitly:** the very
  first sign of this phenomenon appears earlier than Wave A — the §14.7
  gate-decision record shows Wave 0's K=32 rerun plateauing at
  `rec@0.9` 0.78–0.80 for the final 21,000 of 25,000 steps (flat, not
  climbing; entity-subspace rank 29.9–30.2, i.e. capacity was NOT the
  issue even then) while K=16 converged cleanly to 0.996–0.999. That
  single anomalous plateau is what motivated designating K=32 the
  exactness-mechanism program's headline cell (§3's "two primary K's"
  note) — the whole `DELTANET_RD_EXACTNESS_DESIGN.md` research line is
  downstream of this one number.
- The depth-amplification law, imported from the from-scratch chapter
  (`TASK_E_FINDINGS.md` §3): raw iteration/composition depth is a sharper
  exactness probe than any single-hop cosine tolerance — a rank-K−1
  operator can pass at shallow hops (`sqrt(1−1/K)` per-dropped-mode
  cosine) and only reveal its deficiency at depth. This is the theoretical
  spine for why "graded frontier at h=1" becomes "collapsed by h=4" — an
  `ε^h`-shaped law, not a separate phenomenon, and it recurs three more
  times in the paper (Wave F's proportionality reading, §6; the K=48 rider;
  the h=21 residual failure of the fix itself, §8).
- The `h=21` caveat, stated once, cited everywhere it recurs: it is a
  depth-decay PROBE (raw iteration count), not an independent held-out-hop
  generalization test, because `21 mod K` collides with in-distribution
  residues at small K — read via `effective_hop`, per the project's own
  standing guidance.
- K=48 rider (frontier extends past d/2, then breaks): key-Gram deviation
  keeps growing (2.7 at K=32 → 4.3 at K=48), h=1 recovery halves
  (0.41–0.44), composition is gone by h≥2; the i-strong pin is correctly
  REFUSED here by its own dimensional guard (96 train+heldout identity
  vectors > d_state=64) — flag this as the paper's one unresolved
  structural boundary (K≈d/2), not smoothed over.

### §5 Mechanism: Causal Attribution via Embedding-Source Interpolation
- The design idea: hold grammar/model/loss fixed, vary ONLY the embedding
  source feeding `k_eff` — learned (baseline), frozen orthonormal, frozen
  real GPT-2 span embeddings, Gram-matched frozen non-orthonormal control —
  a genuine intervention, claim tier "causal, premise-conditional," with
  the bundled-vs-geometry-only contrast distinction stated explicitly
  (i-vs-iv is the geometry-only comparison; frozen-vs-learned bundles
  geometry AND trainable-param count).
- Result: EVERY arm converges to the SAME non-orthonormal attractor
  (gram dev 2.71–2.92 at K=32 across arms i/ii/iv vs. learned 2.71–2.77) —
  raw embedding geometry is causally IRRELEVANT; the trained W_k/conv path
  maps every input geometry onto the same write-geometry basin.
- The complementary non-interventional evidence: measured-β closed-form
  crosstalk reconstruction is near-exact (residual 0.0038–0.0041 vs.
  0.148–0.377 under the wrong-regime β=1 idealization) — "geometry + β"
  is essentially the complete state-formation account, at the (weaker,
  non-causal) explanatory tier §7 assigns it (Test B: structural
  deterministic reproduction, not a causal claim — state this distinction
  explicitly, it is exactly the kind of tier-blur a reviewer will probe).
- The existence proof (i-strong, two-variable by design, diagnostic
  tie-breaker role — NEVER a standalone causal headline, stated as
  written): a non-trainable, hand-built, per-identity k_eff pin bypassing
  W_k/conv achieves gram dev 0.000 and rec@0.9 = 1.00/1.00/1.00 at
  h=1/2/3, K=32 — matching the synthetic harness exactly. The architecture
  and the delta rule's own algebra CAN represent and use the perfect
  solution. Ordinary training does not find it. This is the paper's
  turning point: from "here is a gap" to "here is what would close it."
- **NEW this round — the 2×2 sharpens "here is what would close it" into
  "here are the two things that close it, and neither alone does"**
  (`KEY_ANCHORING_DESIGN.md` §2.0, Fig. 5): reading the archive along both
  the orthogonal/not-orthogonal axis AND a stable/not-stable
  (context-free/episode-conditional) axis at once shows i-strong's perfect
  recovery is not explained by orthogonality alone — two NEW arms that are
  stable (fixed input table) but NOT orthogonal (the trained `W_k` path
  still de-orthogonalizes them, exactly as arms i/ii/iv already showed)
  land at h=4 **0.0075–0.0114** and **0.0049–0.0079**, statistically
  indistinguishable from the fully-learned baseline's 0.009 — stability
  without orthogonality buys nothing. Orthogonality alone (bare geo3, §7
  below) buys most of the gap (0.44, a 44–56× jump) but plateaus well
  short of 1.0. Only the conjunction saturates. This reframes the paper's
  turning point from a single named cause to a two-part necessary-and-
  sufficient mechanism, and directly motivates the anchoring follow-on
  (§7 below, "results still landing" §9) as a test of whether the second
  part is SGD-discoverable at all, not just hand-constructible. **Caveat
  carried at every mention, not just here:** the "both together" cell in
  this 2×2 is i-strong itself, which remains pool-restricted (32 ≤
  d_state) — the 2×2 demonstrates the interaction is real, it does not by
  itself set the ceiling a full-vocabulary trainable mechanism could
  reach; that ceiling is a separately computed number (§9 below).

### §6 Soft Fixes Fail: The Attractor Is Robust
- F-geo-1 (L_orth penalty, λ∈{0.1,1.0}) and F-geo-2 (ZCA whitening): both
  pre-registered, both ran to completion with NO further tuning (anti-
  Goodhart cap honored — the single registered shot, stated explicitly so
  it doesn't read as "we could have tried harder").
- Both land in the SAME narrow basin (gram dev 2.51–2.57 at K=32 vs.
  baseline 2.71–2.77 — a 3–8% cleanup) regardless of mechanism (weak soft
  push, strong soft push, population-level decorrelation) or λ (10× change
  in penalty weight bought nothing — the penalty saturates). This
  "three approach angles, one attractor" reading (Fig. 6 caption) is
  itself evidence, not just a null result.
- Every pre-registered bar MISSED (headline ~20–25× below bar; minimum
  publishable ~1.6–1.7× above achieved) while the h=1 no-sacrifice guard
  is satisfied everywhere — the interventions do what they promise, just
  not enough of it.
- The honest read, stated in the paper's own voice, not hedged: this is
  the pre-registered NEGATIVE branch, explicitly anticipated in the
  design ("a dominant mechanism whose matching fix does NOT move the
  frontier is itself a publishable finding"), not a failed pilot that got
  quietly re-run until something worked.

### §7 The Fix: Structural, Differentiable Orthogonalization (F-geo-3)
- Motivation stated as the two facts that motivate it, not as engineering
  for its own sake: (1) the exact solution exists and composes perfectly
  (i-strong); (2) SGD does not find it under any soft push tried.
- Design: cubic Newton-Schulz/Björck-Bowie iteration at the k_eff site,
  chosen over Gram-Schmidt (order-dependent, rejected on a
  project-specific ground: it would reintroduce exactly the write-history-
  asymmetry mechanism (g) already named as part of the problem) and QR
  (order-dependent for the same underlying reason, and backward-unstable
  near collinear input). Order-equivariant, differentiable through
  ordinary matmuls, sqrt(K) pre-scaling gives an unconditional
  convergence-basin guarantee.
- The gating discipline, stated as method not marketing: a two-round
  independent attack found a load-bearing failure mode the design missed
  (cross-episode key-identity drift — orthogonalizing correctly WITHIN an
  episode does not guarantee the SAME key across episodes of the same
  entity), which became a pre-registered Wave −1 gating diagnostic with
  its own committed simulator and launch-read rule — Wave 1 GPU spend was
  authorized only after the simulator predicted the K=16 bar would clear.
  This is presented as substance (why the reader should trust the launch
  decision), not as a novelty claim about "our process."
- The substitute admission-stack caveat — now largely DISCHARGED, but the
  discharge itself must be shown, not asserted: F-geo-3 makes the standard
  key-Gram/alignment premise-validity instruments tautological by
  construction (orthogonality ≡ true, alignment ≡ 1.0), so a
  DIFFERENT, pre-registered-before-any-data admission criterion applies
  to F-geo-3 seeds (value-side salvage tier + Newton-Schulz convergence
  without eigh fallback + finite loss + a task-performance floor). With
  the K=32 n_iter=20 escalation cells passing EVERY substitute criterion
  including zero fallback steps (and K=16 3/3 from the start), this
  caveat drops from headline asterisk to footnote — but the footnote
  still names both admission stacks and both realized pass rates wherever
  a geo3 seed count appears, per the pre-registration's own requirement.
  The fallback-irrelevance finding (§8 below) additionally shows the one
  criterion that ever disqualified a seed was conservative, not
  explanatory.
- **NEW this round — the direct follow-on to the outcome-F residual is
  itself designed, attacked, and built** (`KEY_ANCHORING_DESIGN.md`), named
  here rather than left as a future-work sentence: a trainable, per-entity
  anchor table (frame-potential-minimized init, hitting the Welch-bound rms
  coherence 0.0796 exactly) blended into the key BEFORE geo3's existing
  Newton-Schulz pass — division of labor by construction (the anchor
  supplies cross-episode stability only; per-episode orthogonality remains
  geo3's job, matching the 2×2's own finding that the two ingredients are
  separable). A computed ceiling, not a hope, bounds what this can achieve:
  with the full 107-entity train pool in R^64, a globally-orthonormal
  table is Welch-bound-impossible, and the best achievable post-NS
  cross-episode drift at K=32 is **~0.9423** (frame-potential init) —
  closing at most ~40% of the distance from the measured 0.9037 to the
  0.95 LOW band, itself close to K=16's own bare-geo3 drift level (0.9416,
  the level at which K=16 cleared its bar with a wide margin). The design
  was attacked across five revisions (closing, in order: 3 FATALs on the
  original Rev 1 construction, including a QR-orthonormal init that
  crashes at n=107>d=64; 5 MAJORs on Rev 2's masked-blend/λ machinery,
  including a 1-ULP renormalization non-no-op and a `torch.where`
  gradient-poisoning footgun on held-out rows; a bounded verification round
  closing every finding with an exact repro). The harness is built and
  CPU/smoke-verified (9/9 Wave −1 smokes pass, including a NaN-injection
  test proving the fixed gather/scatter form is gradient-safe where the
  superseded form was not); GPU launch awaits a free box slot (all 8 GPUs
  were saturated with other waves at build time) — reported honestly as
  built-and-gated, not launched, in "results still landing" (§9).

### §8 Results: The Fix Demonstration
- The pre-spend gate: predicted K16 h4=1.00, K32 h4=0.77 from measured
  cross-episode drift (0.94 K=16, 0.90 K=32) — training stabilizes keys
  relative to the untrained-probe reference, resolving the attack's
  cross-episode-drift risk favorably enough to authorize the launch. The
  gate's own registered cell was **K=16 only** (predicted 1.00, measured
  0.9767 — accurate); K=32's 0.77 prediction was explicitly non-gating and
  informational. **Dated correction (§16.7, this round):** that 0.77
  figure was later found to be computed with K=16's drift silently reused
  for K=32 (a shared-scalar wiring bug); K=32's own drift (0.9037) predicts
  0.06–0.09 instead — the simulator is CONSERVATIVE at K=32, not
  miscalibrated the direction originally reported. State this plainly
  rather than let the retracted number stand uncorrected anywhere it is
  quoted: the gate decision itself does not change (it never used the
  K=32 number to authorize anything), and the outcome-F attribution below
  does not change (it rests on measured drift + measured h=4, never on
  the simulator) — only the now-retired "simulator overestimated K=32"
  reading is withdrawn.
- K=16 (minimum-publishable headline): HIT, 3/3 admissible seeds. h=4
  0.95–1.00 vs. bar ≥0.8, vs. baseline 0.42–0.47. h=7 0.55–0.67 vs.
  baseline 0.07–0.10. h=1 no-sacrifice: 1.0 everywhere.
- K=32 (primary/headline anomaly cell) — **RESOLVED, report in three
  moves, in this order:**
  1. **Admissibility: CLEARED.** The original n_iter=12 cells were 0/3
     admissible solely on the Newton-Schulz fallback criterion (56/20,000
     steps = 0.28% eigh-fallback — small but, per pre-registration,
     disqualifying under item 2 of §14.10's substitute stack). The
     audit-verified n_iter=20 escalation completed: ALL 3 seeds fully
     admissible — ZERO fallback steps, every substitute-stack criterion
     passed. The ~45× improvement (h=4 0.009 → 0.390/0.416/0.504; ID h=2
     1.0 vs. baseline 0.26; h=3 0.87–0.95 vs. 0.05) is claimable at full
     evidentiary tier; the admission-stack difference drops to a footnote
     (both stacks and realized pass rates still named, per
     pre-registration).
  2. **Headline bar (≥0.5): NOT met — honest narrow miss, reported
     exactly.** h=4 = 0.504/0.416/0.390, mean 0.44; 1/3 seeds at the
     line. Never rounded up, never restated as "approximately met."
  3. **The miss is ATTRIBUTED, not unexplained.** Trained cross-episode
     key drift at K=32 measured 0.9037 < 0.95, the pre-registered HIGH
     band (§14.5's pinned D/E/F split) → outcome F: "stable-not-just-
     orthogonal geometry is the bottleneck" — the exact mechanism the
     adversarial attack round (finding F1) named, and whose diagnostic
     bands were pinned, BEFORE the wave ran. Per-identity `v_eff` cannot
     chase an episode-conditional orthogonalized key; within-episode
     orthogonality is delivered (resid≈0 by construction), cross-episode
     key IDENTITY is the residual requirement. This is narratively the
     paper's closing strength: even the shortfall has a named,
     pre-registered, pre-measured cause and an obvious follow-on
     (cross-episode key anchoring/stabilization — e.g. EMA-anchored or
     identity-registered orthogonalization, named as a direction in the
     design's own outcome-F row) → future work, not a loose end.
- **The fallback-irrelevance finding (report alongside, it sharpens the
  attribution):** the n_iter=20 escalation cells (zero fallback steps)
  are numerically IDENTICAL to the n_iter=12 cells (56 fallback steps) —
  h=4 0.504/0.416/0.390 vs. 0.39–0.50. The 56 "corrupted" steps changed
  nothing; the residual gap is drift, not solver noise. This
  simultaneously (a) validates that the pre-registered admission
  criterion was conservative rather than explanatory, and (b) rules out
  solver-health as a competing explanation for the bar miss — the
  outcome-F attribution stands alone.
- One number that does NOT move under the fix, stated without spin: h=21
  literal-depth composition still collapses under F-geo-3 — orthogonality
  fixes write-time interference between bindings, not the separate
  iteration-compounding mechanism the depth-amplification law (§4)
  describes. Name this explicitly as the fix's own scope boundary, not a
  gap in the evidence.

### §9 Discussion and Limitations
(Full content spec in §6 below — reproduced compactly in-line here too, per
the deliverable spec.)

### §10 Conclusion
- Restate the thesis in the paper's own final voice: capacity and
  exactness are separable, measurable, and separately fixable properties
  of a fast-weight architecture's learned state — this paper is the first
  to show the gap exists on real text, name its cause (now sharpened to a
  two-part necessary-and-sufficient mechanism, not a single attractor), and
  demonstrate (not just diagnose) a mechanism-matched, pre-registered fix
  for the first part.
- Close the causal chain explicitly: bar hit at K=16 (3/3); ~45× at K=32
  at full admissibility with the narrow bar miss attributed to a
  pre-measured second mechanism (outcome F, drift 0.9037 in the
  registered HIGH band), independently corroborated by the 2×2's own
  existence-proof design (Fig. 5); the named follow-on (learned
  cross-episode key anchoring) is designed, attacked, and built, its
  launch outcome an explicitly open question, not assumed either way. No
  loose ends on the mechanism — every residual has a name and a number,
  even the ones still running.
- **NEW closing note this round:** state plainly, without overreaching,
  that the phenomenon this paper studies is not confined to the scale it
  is studied at — a controlled 14M→98M ladder shows the same write
  geometry, un-dissolved, and a substantially larger version of the same
  signature is present in deployed production fixed-state models. The
  paper does NOT claim the production-scale signature is caused by the
  same mechanism (a matched negative control shows it is not yet
  distinguishable from generic key anisotropy) — it claims only that the
  question this paper's small-scale causal chain answers is a live one for
  architectures already being deployed for long-context, cache-free
  inference, which is why the mechanism-level answer given here matters
  beyond the scale it was measured at.

### §11 Reproducibility
Pointer to `REPRODUCIBILITY.md` (§4 below), kept in the paper as a compact
paragraph + link, per house convention (`09_reproducibility.tex` in the
existing ICML submission).

---

## 4. Claim-tier cheat sheet (carry into every section verbatim, do not
paraphrase loosely — this is the list a reviewer-facing tier audit checks
against before submission)

| Claim | Tier | Source |
|---|---|---|
| Synthetic rank-K necessity (razor cliff, k=31→32) | **CONFIRMED** (causal) | `CAUSAL_RANK` §12.8.5 |
| Real-data rank-K necessity (RD-1, graded staircase) | **CONFIRMED**, verbatim: "RD-1's branch (b), causal necessity, is CONFIRMED at all four K tested (8, 16 primary, 24, 32)" — with the mandatory qualifier "graded across a window several ranks wide, not a single-step threshold... If a future write-up quotes 'razor cliff at k=K−1→K' for the real-data result, that is not what was measured here" | `REALDATA` §17.9/§17.11 |
| Embedding-source interpolation arms (i/ii/iv vs. iii) | **causal, premise-conditional**; bundled (geometry+trainability) unless the i-vs-iv contrast is named, which is geometry-only, conditional on iv's re-verification gate | `RD_EXACTNESS` §7 |
| i-strong existence proof | **diagnostic tie-breaker**, NOT a standalone causal headline (two-variable by design) | `RD_EXACTNESS` §7 |
| §3's closed-form reconstruction, Test A | **β-idealization adequacy diagnostic only** — no explanatory claim | `RD_EXACTNESS` §7 |
| §3's closed-form reconstruction, Test B | **structural, deterministic reproduction** — strong non-causal explanatory claim, not causal | `RD_EXACTNESS` §7 |
| Wave F soft-arm interventions | **causal, premise-conditional** on the intervention variable; demonstration claim ("fix moves frontier") is **FALSE**, stated plainly | `RD_EXACTNESS` §15.7 |
| F-geo-3 demonstration claim ("fix moves frontier") | **causal, premise-conditional**, TRUE at K=16 (3/3 admissible, bar ≥0.8 HIT); at K=32 the ~45× improvement is at **full evidentiary tier** (n_iter=20 escalation: 3/3 admissible, zero fallback steps, every substitute criterion passed) but the ≥0.5 headline bar is **NOT met** (mean 0.44, 1/3 seeds at the line) — never blur "admissible and large" into "bar met" | `RD_EXACTNESS` §14.10; `EXPERIMENT_LOG` 2026-07-04 + escalation cells |
| K=32 bar-miss attribution (outcome F: stable-not-just-orthogonal geometry) | **pre-registered mechanism verdict** — trained drift 0.9037 < 0.95 (the pinned HIGH band, §14.5) + resid≈0 + graded h-decay = the three-part outcome-F signature registered before the wave ran; the fallback-irrelevance check (n_iter=20 ≡ n_iter=12 numerically) rules out solver noise as a competing account | `RD_EXACTNESS` §14.5/§14.8 (outcome F); escalation cells |
| F-geo-3 attribution claim ("therefore the geometry attribution was right") | **supported inference**, never proven by the demo alone — keep separate from the demonstration claim in every mention | `RD_EXACTNESS` §14.10 |
| F-geo-3 cross-arm seed-count comparability | pre-registered UNVERIFIED-comparability language now discharged to a **footnote**: escalation cells passed every substitute criterion including zero fallbacks; both stacks and realized pass rates still named wherever a geo3 seed count appears | `RD_EXACTNESS` §14.10 |
| Wave 2 (Waves C+D, corpus generality) | verbatim: "**descriptive+interventional (RD-2, §6.3, §14.7) -- NOT premise-conditional causal**"; Q2's own conclusion further hedged as "**consistent with (not proof of)**" | `REALDATA` §6.3/§19, §19.3 |
| **NEW — 2×2 existence-proof design (arms i/iv as "stable, not orthogonal" cell)** | **causal, premise-conditional**, same tier as the embedding-source interpolation arms it reuses; the "both together" cell (i-strong) keeps its existing **diagnostic tie-breaker** tier and its pool-restriction caveat (32≤d_state) — the 2×2 demonstrates the interaction, it does NOT set a full-pool ceiling | `KEY_ANCHORING_DESIGN.md` §2.0; per-seed values re-verified against `w1_rdx_K32_armi_s*.json`/`armiv_s*_rho0p225.json` this revision |
| **NEW — §16.7 simulator correction** | **meta-claim about instrumentation, not about the mechanism**: the retracted "K=32 overestimate (0.7734)" reading is WITHDRAWN (input-wiring bug, K=16's drift fed to K=32's prediction); corrected, the simulator is **conservative** at K=32 (predicts 0.06–0.09 vs. measured 0.4368). The outcome-F attribution itself is **UNCHANGED** — it never depended on the simulator's point estimate, only on the three measured signatures (§16.6) | `RD_EXACTNESS` §16.7 (dated correction); `experiment-runs/2026-07-04_geo3_simulator_recheck/` |
| **NEW — key-anchoring wave (learned cross-episode stability mechanism)** | **not yet a result — design + build only.** No claim-tier applies to an outcome that does not exist yet; the design itself (candidate selection, computed λ=1 ceiling ~0.9423, Welch-bound argument) is **derivation, not measurement** | `KEY_ANCHORING_DESIGN.md` Rev 5, §1–§2; build verified via CPU/smoke tests only, `experiment-runs/` GPU archive does not yet exist for this wave |
| **NEW — Track C rung-1 attractor persistence (14M→98M)** | **Tier 2 — descriptive + interventional** (`SCALE_TRANSFER_DESIGN.md` §2's own convention); explicitly a 2-point read, geometry-only, no compositional cross-check; never state as a 3-rung trend | `SCALE_TRANSFER_DESIGN.md` §5.9; `EXPERIMENT_LOG.md` "Track C Wave 1 (rung-1) harvest," 2026-07-04 |
| **NEW — Track D production-scale measurement** | **Tier 3 — measurement-only, no causal or interventional language.** Explicitly **NOT attributable** to the delta-rule write mechanism at this measurement tier (matched no-fixed-state negative control shows the same magnitude) | `SCALE_TRANSFER_DESIGN.md` §6.8; `EXPERIMENT_LOG.md` "Track D Phase 1," 2026-07-04 |
| **NEW — Track B (geo3-in-LM) Wave −1 gate** | **registered hard no-launch**, not a soft "did not prioritize" — criterion (b) (excluded-position write-mass ≤0.40) measured 0.431, a clean fail on real archived checkpoints, before any training GPU-hour spent | `SCALE_TRANSFER_DESIGN.md` §4.2; `EXPERIMENT_LOG.md` "Track B... HARD NO-LAUNCH," 2026-07-04 |

---

## 5. Related-work map

1. **Nazari & Rusch (arXiv:2602.04852) and Sun et al. (arXiv:2602.02195),
   both Feb 2026.** Descriptive rank measurement of fast-weight states in
   pretrained LLMs, uncontrolled K, no necessity theorem, no causal
   intervention. We are the mirror image: controlled K, a provable LOWER
   bound, and a causal (train-time-then-eval-time) intervention — run on
   the exact architecture family they measure. Cite as the paper's primary
   "descriptive vs. causal" foil, first in the related-work section and
   again in the intro contribution list.
2. **Nichani, Lee & Bietti (arXiv:2412.06538, ICLR 2025 Spotlight).**
   Closest prior theoretical construction — a rank-m hand-built existence
   proof for associative recall, but under DISCRETE ARGMAX decoding, where
   a rank-1 matrix already recovers ≈d associations. This is the exact
   reason our readout is pinned to exact-continuous cosine recovery, never
   argmax — state the distinction explicitly, it is load-bearing for why
   our bound is a genuine necessity result and theirs is not directly
   comparable.
3. **Schlag, Irie & Schmidhuber (arXiv:2102.11174, ICML 2021).** Origin of
   "linear-attention state as associative memory" — DeltaNet/fast-weight-
   programmer lineage, motivating citation for why rank/capacity matters
   once writes exceed ambient dimension.
4. **DeltaNet / Gated DeltaNet production lineage** (the `fla-org` kernel
   this paper's harness uses directly) and **DeltaProduct**
   (arXiv:2502.10297). DeltaProduct's "rank" is the PER-STEP TRANSITION
   matrix `A_t = I − β_t k_t k_t^T` (a different object from the STATE's
   own rank `rank(S_t)` this paper measures) — name this distinction
   explicitly in one sentence; it is the single most likely reviewer
   confusion per the source design doc's own attack-round finding. Also
   cite Grazzi et al. (arXiv:2411.12537, negative-eigenvalue state
   tracking) as the reason composability is gated by eigenvalue
   sign/phase, not rank magnitude alone — relevant context for the h=21
   depth-amplification discussion.
5. **Barnfield et al. (arXiv:2605.05189).** Proves capacity thresholds for
   STATIC rank-constrained associative memories (fixed matrix, no
   recurrence, no training dynamics). Our delta lives entirely in the
   recurrent/causal/compositional part — the training-time force-rank
   ablation on a time-varying, streamed state, plus the held-out-hop
   composition test.
6. **Based / Arora et al. (arXiv:2402.18668, Thm F.1) and Zoology (Arora
   et al., arXiv:2312.04927).** Based bounds total state SIZE
   (architecture-and-decoding-agnostic); justify explicitly why RANK, not
   size, is the binding resource under our pinned linear-unbind readout —
   a strictly sharper, mechanism-specific claim a size bound cannot make
   (our force-rank-k ablation manipulates rank while holding ambient d²
   fixed, which a size-only argument cannot do). Zoology is the standard
   synthetic associative-recall (MQAR) benchmark family our probe grammar
   resembles — cite for task-family lineage, note we vary rank spectra
   where Zoology does not.
7. **M²RNN (Mishra, Tan, Stoica, Gonzalez, Dao; arXiv:2603.14360).** The
   single closest prior-art paper — matrix-valued recurrent state trained
   at seq-len 128, evaluated near-perfectly at 512 on permutation-group
   (S₃) composition, the same "compose beyond the training horizon" shape
   as this paper's held-out-hop test. Their own comparison table has a
   plain vector-state GRU matching the matrix model on that task (so their
   result does not show matrix rank is load-bearing), Thm 1 is
   representational/existence, not necessity, and they run NO causal rank
   intervention. Lead with what their own results already concede, then
   state the three-way mirror (necessity bound + causal intervention +
   controlled-K framing). **Caveat to carry into the writing process, not
   just note here:** the source design doc flags this citation's specifics
   as "verify all specifics before external citation; abstract-level
   sources only so far" — re-verify before the camera-ready bibliography
   is finalized, do not cite claims beyond what has been independently
   checked.
8. **Superposition / distributed representation literature.** Elhage et
   al. (Toy Models of Superposition) as the general framing for why a
   bounded-dimension state might pack more than d independent directions
   under favorable readout assumptions — relevant background for why
   "capacity" and "exactness" are not the same question, and for framing
   why our pinned-readout choice matters (superposition results generally
   assume the SAME favorable readout — sparse/near-orthogonal recovery —
   our provable bound explicitly closes off by requiring EXACT recovery
   under a fixed linear unbind). Also Holographic Reduced Representations
   (Plate 1995) and Resonator Networks (Frady, Kleyko & Sommer 2020) as
   the VSA/HRR capacity tradition — cite briefly to scope that their
   capacity notion (SNR/bundling/codebook factorization) is DIFFERENT from
   the exact-rank framing this paper uses, so a reviewer from that
   tradition doesn't read a false equivalence.
9. **Prior chapter in this project's own line of work
   (Task D/E, `matrix-thinking/chapter2/`).** Not external related work,
   but must be cited as the paper's own immediate lineage: the
   from-scratch, bespoke-attention-encoder version of the identical
   rank-recruitment result (`TASK_D_PREREGISTRATION.md`,
   `TASK_E_FINDINGS.md`), including the depth-amplification law this
   paper imports directly (§3 of `TASK_E_FINDINGS.md`) and the earlier,
   now-superseded ProsQA/matrix-CODI negative result
   (workshop paper, `matrix-thinking/submissions/icml-mi-workshop-2026/`)
   that motivated pinning the readout to exact-continuous recovery in the
   first place. State explicitly that this paper's contribution is
   TRANSFERRING that from-scratch result to a production architecture
   family and, new to this paper, discovering and fixing the exactness gap
   the from-scratch chapter's synthetic construction never exposed.
10. **NEW — this paper's own live follow-on and scale program, cited for
    completeness, not as external related work.** (a) `KEY_ANCHORING_
    DESIGN.md` is the direct continuation of §7/§8's outcome-F residual —
    cite it as the paper's own designed-and-built (not yet run) next step,
    never as a completed result. (b) `SCALE_TRANSFER_DESIGN.md`'s Track C
    (14M→98M attractor persistence, Tier 2) and Track D (production-model
    measurement, Tier 3, non-attributable) are this paper's own scale-
    generality program, cited in the Limitations section (§6) rather than
    the results body, since neither changes the small-scale causal chain's
    tier — they extend its motivation, not its evidence base. (c) A
    sibling result from a DIFFERENT architecture family in this same
    project (`matrix-thinking/chapter2/`'s matrix-native model, Stage-G's
    H_e task-swap calibration, `STATE.md` 2026-07-04) found the OPPOSITE
    separation on a composition-heavy corpus at 40K steps — a vector
    baseline composes in-context (h1/h2/h3 = 1.0/1.0/1.0) while the
    from-scratch matrix architecture does not (1.0/0.027/0.013). This is
    explicitly NOT a DeltaNet finding and must never be cited as if it
    were — it concerns a structurally different (single-Z-bottleneck,
    from-scratch) architecture and a different composition task; it is
    noted here only because it is part of the same project's live
    evidence about when structured/compositional state representations
    do and do not compose, and a reviewer who has seen one might ask about
    the other.

---

## 6. Limitations section content

State every item below in the paper's own limitations section, undiluted:

1. **Small scale.** `d_model` in the low hundreds, `d_state=64`, models
   sub-1M–low-single-digit-M parameters. No claim here has been tested at
   a scale where these mechanisms might behave differently (e.g., where
   the write-time attractor might be easier or harder for SGD to escape
   under a much larger, more overparameterized key-projection path). **NEW
   — partial evidence, not a resolution:** a controlled 14M→98M scaling
   ladder (Tier 2, geometry-only, 2-point read, `SCALE_TRANSFER_DESIGN.md`
   §5.9) finds the attractor does NOT dissolve with scale on this slice
   (chunk-level key-Gram deviation 21.93→27.82, both far above a random
   anchor of 7.94) — a 392M rung is running and a 1.3B rung is queued, but
   this item's core claim (untested at production scale) stands until
   those complete and a compositional-recovery cross-check, not just
   geometry, exists at scale.
2. **Single architecture family.** Everything is vanilla, single-head,
   single-layer DeltaNet with the standard delta rule. Gated variants
   (Gated DeltaNet, the actual production configuration in Qwen3-Next),
   DeltaProduct's multi-Householder generalization, and other fast-weight
   families are untested — flag explicitly that the H=1/single-layer
   choice was a deliberate gate-simplicity decision (closing the
   multi-head/multi-layer escape cleanly), not evidence the phenomenon
   is H=1-specific (ReserveMH's result that every head independently
   recruits full rank at H∈{2,4} is suggestive but was run on the
   synthetic task, not yet on the real-data exactness frontier).
3. **The h=21 literal-depth decay remains unexplained and unfixed.**
   F-geo-3 demonstrably does not touch it. This is presented as a named,
   scoped-out mechanism (iteration-compounding, distinct from write-time
   interference), not folded into the "the fix works" narrative.
4. **The K=32 headline bar is not met.** The fix demonstration is fully
   admissible at BOTH K (K=16 3/3; K=32 3/3 on the n_iter=20 escalation,
   zero fallback steps), but the pre-registered ≥0.5 headline bar at K=32
   is narrowly missed: h=4 = 0.504/0.416/0.390, mean 0.44, 1/3 seeds at
   the line. State the miss exactly; state the attribution (outcome F,
   trained drift 0.9037 in the pre-registered HIGH band — cross-episode
   key stability, not within-episode orthogonality, is the residual
   bottleneck); and state the follow-on (cross-episode key anchoring/
   stabilization, e.g. EMA-anchored or identity-registered
   orthogonalization) as FUTURE WORK with a measured target — a named
   residual, not an open mystery, but still a limitation of what this
   paper demonstrates. **NEW — the follow-on is now designed and built,
   not only named:** `KEY_ANCHORING_DESIGN.md`'s learned anchor-table
   mechanism is attacked (5 revisions), built, and CPU/smoke-verified;
   its own PRE-COMPUTED ceiling (Welch-bound-limited post-NS drift ~0.9423
   at K=32, full 107-entity pool) means even a fully successful launch is
   not guaranteed to clear the ≥0.5 bar with margin — state this ceiling
   alongside the follow-on, not as an afterthought, so a successful launch
   is not read as a foregone conclusion. GPU launch has not yet occurred
   (§9, "results still landing").
5. **The K≈d/2 structural boundary is observed, not explained.** The K=48
   rider shows the frontier breaking down further and i-strong's own
   dimensional guard correctly refusing to run there; no mechanism account
   is offered for why d/2 specifically, beyond dimension-counting
   (train+held-out identity pools both needing ≥K coverage).
6. **Scale-transfer to a full pretrained LLM checkpoint is now PARTIALLY
   measured, not wholly unknown — state the upgraded picture precisely,
   not as a still-open unknown.** Wave 2 (Waves C+D) establishes the
   truncation-damage phenomenon generalizes to naturalistic LM pretraining
   at small scale (descriptive+interventional tier only). **NEW —
   Track C/D (`SCALE_TRANSFER_DESIGN.md`, 2026-07-04) go further and
   change what can honestly be said:** (a) on a controlled 14M→98M
   parameter ladder, the write-geometry attractor does not dissolve
   (Tier 2, 2-point geometry read, §5.9 — a genuine but narrow result,
   not yet extended to compositional recovery at scale); (b) a much
   larger version of the SAME non-orthonormal write-geometry signature is
   measurable directly in production-scale pretrained fixed-state models
   (RWKV-7 1.5B, Falcon-Mamba-7B; Tier 3, measurement-only, §6.8); (c) but
   a matched no-fixed-state negative control (Qwen2.5-1.5B, ordinary
   softmax attention) shows the SAME magnitude, so (b) is explicitly
   **NOT** attributable to the delta-rule write mechanism specifically at
   this measurement tier — a real, load-bearing non-finding, not a soft
   hedge. Net honest statement for the paper: the phenomenon persists
   where it can be controlled for (a), a phenomenologically similar
   signature exists at production scale (b), but causal attribution at
   production scale remains open and is confounded by generic trained-LM
   key anisotropy (c) — do not round (b)+(c) up into "the mechanism
   transfers to production scale," and do not round (a) down into "scale
   transfer is untested," either.
7. **The admission-stack asymmetry, now a footnote-level limitation.**
   Because F-geo-3 makes the standard premise-validity instruments
   tautological, a substitute pre-registered criterion applies to its
   seeds. With every substitute criterion passed at both K (including
   zero fallback steps on the escalation cells), this drops from a
   headline evidentiary asterisk to a footnote — but the paper still
   names both stacks and both realized pass rates wherever geo3 and
   non-geo3 seed counts appear side by side, because the two stacks
   filter different failure modes and their relative selectivity was
   never empirically equated.
8. **NEW — the 2×2's "stable" cells lean on existence-tier or bundled
   arms, not yet a full-pool learned mechanism.** The 2×2 (Fig. 5) that
   demonstrates orthogonality and stability are jointly necessary uses
   i-strong (pool-restricted to 32≤d_state) for the "both together" cell
   and frozen-embedding arms (which bundle fixed geometry with reduced
   trainable parameter count relative to the learned baseline) for the
   "stable, not orthogonal" cells. The interaction the 2×2 demonstrates is
   real and reproducible from already-archived data, but it does not by
   itself establish that a full-vocabulary, gradient-discovered anchor
   mechanism can reach the same regime — that is precisely what the
   (not-yet-launched) key-anchoring wave tests, and its own pre-computed
   ceiling (~0.9423 post-NS drift at K=32, Welch-bound-limited) says even
   success there would not reach i-strong's 1.00.
9. **NEW — the key-anchoring follow-on has not launched.** Designed,
   attacked across five revisions, built, and CPU/smoke-verified, but no
   GPU run exists yet (all 8 GPUs were saturated with other waves at
   build time). Every claim in this paper about "the residual mechanism"
   rests on the outcome-F attribution (§8) and the 2×2 (§5), not on the
   anchoring wave's results — the anchoring wave is future work in the
   literal sense of not having run, not a diagnosed-but-unwritten result.

---

## 7. Reviewer-attack pre-mortem (10 hardest reviews + our answers)

1. **"K=32 missed its bar, and your own registered simulator predicted
   0.77 there while you measured 0.44 — isn't the outcome-F 'attribution'
   just a post-hoc story rescuing a failed prediction, and the simulator
   demonstrably miscalibrated?"**
   Answer, three parts. (a) The launch gate was pre-registered on K=16
   ONLY (predicted 1.00, measured 0.95–1.00 — accurate); the K=32
   prediction was explicitly registered as non-gating precisely because
   whether training shrinks drift was an open empirical question the
   design named in advance. (b) The attribution does not rest on the
   simulator's point prediction — it rests on the three-part outcome-F
   signature pre-registered in §14.8 with numerically pinned bands
   (§14.5): resid≈0 (delivered by construction) + trained drift in the
   HIGH band (measured 0.9037 < 0.95) + graded h-decay steeper at higher
   K. All three hold. Outcome F was written, banded, and its measurement
   protocol pinned BEFORE the wave ran — the dated design revisions show
   it. (c) The fallback-irrelevance check (n_iter=20 with zero fallbacks
   is numerically identical to n_iter=12 with 56) eliminates the one
   competing within-run account (solver noise). What the miss costs us is
   the K=32 headline-bar claim, which we do not make; what it cannot be
   spun into is an unexplained anomaly.
2. **"Your 'substitute admission stack' for F-geo-3 was invented after
   you saw that the standard one was tautological under your own fix —
   isn't that exactly the kind of post-hoc criterion-shopping the field
   is trying to stop?"**
   Answer: the substitute stack was pre-registered BEFORE any Wave 1
   F-geo-3 data existed (the design doc's own dated revision history shows
   this), specifically in response to an independent attacker's finding
   (not the authors' own convenience), and its comparability to the
   standard stack is explicitly flagged UNVERIFIED rather than assumed —
   we show our work rather than assert equivalence.
3. **"Is 'graded, not razor' on real data just a fancy way of saying the
   causal effect is weak / noisy?"**
   Answer: no — the eval-truncation ceiling is reached EXACTLY at k=K at
   every tested K, to the same precision as the synthetic razor cliff; only
   the WIDTH of the transition below the ceiling differs (2–3 ranks at
   K=8 vs. 12–13 at K=32). The causal necessity claim (RD-1 CONFIRMED) does
   not rest on cliff sharpness — it rests on the ceiling location, which is
   exact. We say this explicitly rather than letting "graded" read as a
   hedge on the causal claim itself.
4. **"You ran embedding-source interpolation and found geometry is
   causally irrelevant to the ATTRACTOR — but then you spend three more
   sections fixing geometry. Isn't that a contradiction?"**
   Answer: no — "irrelevant" refers to the RAW INPUT embedding geometry
   (what feeds the projection); the attractor itself IS write-time key
   geometry, produced by the trained W_k/conv path regardless of input.
   The fix (F-geo-3) intervenes on the WRITE-TIME geometry directly, not
   the input — this is precisely why an input-side intervention (arms i/
   ii/iv) couldn't move the needle and a structural, post-projection
   intervention could. State this distinction crisply in §5, it is the
   paper's most easily-misread step.
5. **"i-strong bypasses the learned projection entirely — isn't that just
   showing a trivial oracle can solve an easy problem, not that SGD is
   failing at anything meaningful?"**
   Answer: i-strong is explicitly scoped, per our own claim-tier
   discipline, as a DIAGNOSTIC TIE-BREAKER, never a standalone causal
   headline (two-variable by construction: embedding source AND
   projection bypass, bundled). Its evidentiary role is narrow and stated
   as such: it demonstrates the delta rule's OWN algebra can represent and
   use an exact solution at this (d,K) — an upper-bound/existence fact —
   which is what licenses treating the gap as an optimization-attractor
   problem rather than an architectural-capacity ceiling. We do not use it
   to claim anything about what SGD "should" find.
6. **"Your proportionality/ε^h law is fit to only two soft-arm data
   points (h=2,3,4) at one K — is this really a 'law' or just three
   numbers going down?"**
   Answer: correctly scoped as a READING, not a fitted/validated closed
   form, in the source material, and we keep that scoping in the paper —
   the qualitative shape (shrinking absolute gain at deeper h from a fixed
   per-binding cleanup) is the load-bearing claim, imported from and
   consistent with the independently-derived depth-amplification
   mechanism (§4/Task E) and the exact `(K−h)/K` match on synthetic data —
   not a new curve-fit asserted on three points alone.
7. **"H=1, single layer — how do you know this isn't a degenerate,
   easy-to-analyze corner case that says nothing about the production
   Gated-DeltaNet configurations actually deployed (Qwen3-Next)?"**
   Answer: stated directly in limitations (§9/§6 above) as an open
   scope boundary, not argued away; the ReserveMH result (every head
   independently recruits full rank at H∈{2,4}, no cross-head
   distribution) is offered as suggestive generality evidence but was run
   on the synthetic task and explicitly flagged as not yet transferred to
   the real-data exactness frontier this paper's headline result concerns.
8. **"Wave 2's corpus-generality result (Q1, rank by corpus) is a mixed,
   metric-dependent, layer-0-only signal, and Q4 runs BACKWARDS from the
   'SGD recruits more rank as it learns' intuition your causal chapters
   built — doesn't that undercut the main story?"**
   Answer: no, because Wave 2 is explicitly scoped at a DIFFERENT claim
   tier (descriptive+interventional, RD-2 — never premise-conditional
   causal) answering a DIFFERENT question (does naturalistic LM pretraining
   show a directionally consistent truncation-sensitivity signature) than
   the controlled causal chapters. We report the Q1/Q4 mixed and
   counter-intuitive findings in full rather than only the clean Q2
   result, and read Q4 as a general LM training dynamic finding, not a
   reasoning-specific one — exactly per the source material's own
   instruction. This is presented as honest scope discipline, not spin.
9. **"Isn't 'name the mechanism, then demonstrate the fix' just standard
   ablation-study practice dressed up with pre-registration language?"**
   Answer: partially fair, and we don't oversell it — the contribution
   claim in the intro is explicitly modest ("offered as a template, not a
   boast"). What we argue is specific, not generic: the gating discipline
   is what let a load-bearing failure mode (cross-episode key drift) get
   caught by an independent attacker BEFORE any Wave 1 GPU spend, via a
   committed, versioned simulator with a pre-registered launch-read
   threshold — a concretely falsifiable go/no-go decision, not a vague
   "we were careful" claim. We show the artifact (§ Reproducibility) so
   this claim is checkable, not asserted.
10. **"This whole story is six sequential negative-then-positive results
    stacked on top of each other — isn't the paper just narrating your
    own research process rather than reporting a clean finding?"**
    Answer: the sequence IS the finding — the central claim
    ("capacity and exactness are separable, and only one of them is easy
    for SGD to find on real data") could not have been established any
    other way: it required first confirming capacity was NOT the
    bottleneck (Wave A rank recruitment), then attributing the bottleneck
    to a specific, falsifiable mechanism (attribution + i-strong), then
    testing whether that attribution predicts a working fix under
    increasingly targeted interventions (soft arms fail proportionally,
    structural fix succeeds at the predicted magnitude) — a positive
    result that arrived any faster would not have ruled out the
    alternative mechanisms (c)–(f) the design explicitly kept live until
    each was tested. We frame the six-step structure explicitly as the
    proof structure in §1's contribution list, not as a research diary.
11. **NEW — "Your 2×2's 'stable' cell (i-strong) is a hand-built pin, not
    a learned mechanism — so all you've shown is that a pin composes,
    which nobody doubted. Doesn't the whole 2×2 argument rest on an
    intervention you can't get SGD to do?"**
    Answer: correct as far as it goes, and we say so explicitly (§6 item
    8) — i-strong is a deliberate, disclosed, three-concession existence
    proof (bypasses the learned path, closed pool, pool-restricted to
    ≤d_state), never presented as a learned result. What the 2×2 actually
    licenses is narrower and does not depend on i-strong being learnable:
    it shows that stability WITHOUT orthogonality (two independent,
    non-pin arms, frozen embeddings under the ordinary trained W_k path)
    buys nothing (0.005–0.011, indistinguishable from baseline) — this
    half of the 2×2 uses no pin at all. Combined with orthogonality-
    without-stability (bare geo3, also no pin, plateauing at 0.44), the
    INTERACTION claim (both ingredients necessary) is fully supported by
    non-pin data; only the "and jointly SUFFICIENT" corner currently
    requires the pin to demonstrate. That is exactly the gap the
    key-anchoring wave is built to close — it will show, when it runs,
    whether "jointly sufficient" survives without the pin. We do not
    claim it does yet. The pre-mortem's own honest answer, stated plainly:
    this review is answered fully by the anchoring wave's result EITHER
    WAY it lands — a clear success extends the interaction claim to a
    learned mechanism; a clean failure (mechanically engaged, bars
    missed) still leaves the interaction claim intact and additionally
    tells us the stability ingredient, though real, may not be
    SGD-discoverable at this scale without hand construction, which is
    itself worth knowing and reportable as such.
12. **NEW — "You corrected your own simulator (§16.7) after finding a
    bug that flipped your reported K=32 prediction from an overestimate
    to an underestimate — doesn't that undermine every quantitative claim
    in §8/§16, not just the simulator figure?"**
    Answer: no, and the reason is mechanical, not rhetorical — trace what
    the retracted number was actually used for. The 0.7734 figure was
    NEVER used to authorize any spend (the launch gate's registered,
    decision-bearing cell was K=16 only, predicted 1.00, measured 0.9767
    — unaffected by this bug) and was never an input to the outcome-F
    attribution (§16.6), which rests on three DIRECTLY MEASURED quantities
    — trained cross-episode drift (0.9037, measured on the real
    checkpoint), resid≈0 (measured, by construction), and the graded
    h-decay shape (measured, steeper at K=32 than K=16) — none of which
    the simulator touches. What the correction changes is exactly one
    thing: a secondary, explicitly-labeled "informational, non-gating"
    prediction, now corrected from "the simulator overshot K=32" to "the
    simulator undershot K=32 (predicts 0.06–0.09 against a measured
    0.44)" — if anything, this makes the fix's real-world performance
    look BETTER relative to the (corrected) model's expectation, not
    worse. We report the bug, the fix, and the corrected direction in
    full (§8, §16.7) rather than quietly restate the number, because a
    reviewer checking our arithmetic should find the correction before
    they find the bug themselves.

---

## 8. Methodology framing (weave-in notes, not a section)

Per the task brief: the adversarial-gauntlet process (design → attack →
verify → build → audit → gated launch) is a contribution, but must be woven
in "without self-congratulation." Concretely:

- **Where it appears:** one paragraph in §3 (Setup), one paragraph in §7
  (The Fix — specifically the launch-gate decision, since that is where
  the process actually changed the outcome: a real failure mode was caught
  pre-spend), and the Reproducibility section/artifact. NOT a standalone
  "Our Methodology" section — that would tip into self-congratulation.
- **Framing rule:** always pair a process claim with the SPECIFIC finding
  it caught or would have missed (e.g., "an independent attack round found
  cross-episode key drift, which the original design's outcome table had
  conflated with incomplete orthogonalization" — not "we used a rigorous
  multi-round review process"). Show, don't narrate.
- **What NOT to claim:** do not claim the process guarantees correctness,
  prevents all errors, or is itself novel as a general ML methodology —
  it is offered as "here is what let us trust a negative result enough to
  publish it," nothing more.

---

## 9. Results still landing (as of 2026-07-04) — NEW section this round

Per the task brief: list every wave that is currently running or built-
but-not-launched, with its pre-registered decision rule, so the narrative
is submission-ready modulo these specific pending readouts. None of the
items below are assumed to land any particular way; each has a decision
rule that was fixed BEFORE this section was written, per the project's
own pre-registration discipline.

1. **Key-anchoring wave (`KEY_ANCHORING_DESIGN.md`, this paper's own
   direct follow-on to the outcome-F residual, §7/§8).** Status: designed,
   attacked across 5 revisions, built, CPU/smoke-verified (9/9 Wave −1
   smokes pass); GPU launch pending a free box slot, not yet run. Manifest:
   28 runs (6 reference bare-geo3 arms for seed-variance + probe-vs-final-
   checkpoint calibration, 8+2 Wave −1 probes, 12 Wave 1 cells for the
   primary anchor-table candidate). Decision rule (`KEY_ANCHORING_DESIGN.md`
   §3.5): **Outcome A (CONFIRMED)** — pre-NS drift ≥0.95 AND h=4 ≥0.5 (3/3
   admissible) AND learned λ in [0.2, 0.8] for ≥2/3 seeds AND per-entity
   anchor-input-alignment engagement ≥90%; **Outcome A′ (pin
   rediscovery)** — bars clear but λ lands >0.95 (SGD re-derives the
   fixed-frame regime, existence-tier only); **Outcome B (informative
   negative, re-attribution required)** — drift stabilizes but h=4 still
   misses, splitting further into a coherence-floor-residual read (B1) or
   a value-geometry re-attribution (B2); **Outcome C (mechanism not
   engaged)** — drift or per-entity engagement never clears its own
   mechanical threshold, uninformative about the hypothesis either way.
   The wave's own computed ceiling (post-NS drift ~0.9423 at K=32,
   Welch-bound-limited by the full 107-entity pool) means even Outcome A
   is not guaranteed to look like i-strong's 1.00 — state this ceiling
   alongside any eventual result.
2. **SCALE-TRANSFER Track C rung-2 (392M params).** Status: LAUNCHED
   2026-07-04 06:10 UTC on GPUs 0–5 (6 runs: 3 seeds × 2 extended-mix
   corpora, ≈21.6 GPU-h/run, ≈129.4 GPU-h for this wave), harvest due
   ≈04:00 UTC 2026-07-05. Decision rule (`SCALE_TRANSFER_DESIGN.md` §5.7):
   attractor persistence read against the rung-1 band (control 21.93,
   rung-1 27.82) — a rung-2 reading that continues climbing, plateaus, or
   reverses are all pre-registered as informative; frontier-probe
   transplant and fix-effect-at-scale stay explicitly out of scope for
   this rung (gated behind Track B, which itself hard-no-launched, §9
   below item 5).
3. **SCALE-TRANSFER Track C wave-1ext (rung-1-on-extended-mix
   de-confounder).** Status: LAUNCHED on GPU 6 solo, ≈27 GPU-h, 6 cells
   sequential. Purpose: rung 1 was originally trained on the ORIGINAL
   (non-extended) data mixes while rungs 2–3 use EXTENDED mixes (a
   step-budget remedy, Rev 2.1 amendment) — this wave retrains rung-1's
   exact architecture on the SAME extended mixes at rung-1's own closed
   step count (67,547 steps, recovered from the archived rung-1 JSON, not
   assumed), isolating the scale axis from the data-mix axis per
   CLAUDE.md's hold-the-second-axis-fixed rule. Decision rule: does
   rung-1-on-extended-mix's key-Gram deviation match rung-1-on-original-
   mix (27.82) within noise, or does the mix change alone move the
   number — this determines whether rung-1→rung-2's reported delta is a
   clean scale effect or still a bundled scale+mix effect.
4. **Stage-G H_e full 6-cell manifest (sibling architecture, cited per
   §5 item 10c's explicit scoping — NOT a DeltaNet result).** Status: 2/6
   cells complete (40K-step calibration: matrix h1/h2/h3 =
   1.0/0.027/0.013, vector = 1.0/1.0/1.0 — the inverse of the design's own
   H_e hypothesis, which expected the structured/matrix representation to
   compose at least as well). Remaining 4 cells (matrix seed 1,
   `h_b_factored_r4` seeds 0+1, vector seed 1) running on GPU 7, ≈16h.
   Decision rule (pre-registered, fired already at the 2-cell stage per
   `EXPERIMENT_LOG.md`'s "Stage-G H_e 40K calibration VERDICT"): signal
   cleared at 40K steps → run the full manifest at 40K (not 80K); the
   full manifest decides whether the matrix architecture's zero
   composition is a seed artifact, universal across the factored-Householder
   variant too, or specific to one configuration.
5. **SCALE-TRANSFER Track B (geo3-in-LM) — CLOSED, not pending, listed
   here for completeness since it is easy to mistake for still-running.**
   Already returned a registered HARD NO-LAUNCH (`EXPERIMENT_LOG.md`,
   2026-07-04): criterion (b), excluded-position write-mass ≤0.40,
   measured 0.431 on all 6 archived Wave C checkpoints — a clean fail,
   caught before any training GPU-hour was spent. No further Track B wave
   is pending; the registered conditional follow-on (a hard-zero-β
   variant) requires its own attack round before any build and is not
   scheduled.

---

## 10. Open items before this narrative can be finalized

1. **[RESOLVED]** The K=32 `n_iter=20` escalation completed: all 3 seeds
   fully admissible (zero fallback steps, every substitute-stack criterion
   passed), h=4 = 0.504/0.416/0.390 (mean 0.44) — numerically identical to
   the n_iter=12 cells (fallback-irrelevance). Headline bar ≥0.5 narrowly
   MISSED (1/3 seeds at the line); miss attributed to pre-registered
   outcome F (trained drift 0.9037 < 0.95 HIGH band). The abstract, §0
   thesis, §3 flow (§8 Results), claim-tier table, Fig. 7, limitations,
   and pre-mortem #1 are all updated to this ending — no load-bearing
   placeholders remain in this narrative.
2. **[RESOLVED]** Real-data eval-truncation archive path confirmed:
   `experiment-runs/2026-07-03_deltanet_rd_waves/{wave0_rerun,waveA}/`,
   aggregated by `matrix-thinking/deltanet_rd/analyze_eval_truncation_rd.py`
   (K=16 uses n=10 pooled seeds, K=32 uses n=7, K=8/24 use n=2 each —
   note this seed-count asymmetry in the actual paper's methods/appendix,
   since K=32's headline ceiling number (0.795) rests on the smallest n
   of the four cells reported in Fig. 1/Fig. 2).
3. **Re-verify M²RNN (arXiv:2603.14360) citation specifics** before the
   camera-ready bibliography — flagged in the source design doc as
   abstract-level-only verification so far.
3b. **[RESOLVED, this revision.]** The K=32 `n_iter=20` escalation JSONs
   are now confirmed present and read directly:
   `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wavegeo3/
   wgeo3_rdx_K32_armgeo3_s{0,1,2}_geo3n20.json` all exist alongside the
   `n_iter=12` files, and `EXPERIMENT_LOG.md`'s "F-geo-3 escalation
   VERDICT" entry (2026-07-04) is landed, not "still running." Every
   number this revision quotes from §16 was cross-checked against this
   archive and/or `DELTANET_RD_EXACTNESS_DESIGN.md` §16 directly, per the
   house provenance rule.
3c. **[NEW, this revision — same provenance discipline applied to the new
   material.]** The 2×2's arm-i/arm-iv per-seed h=4 values (0.0093, 0.0079
   quoted at seed 0 in Fig. 5/§5/§0) were re-derived directly from
   `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wave1/
   w1_rdx_K32_armi_s0.json` and `..._armiv_s0_rho0p225.json`'s own
   `checkpoints[-1].M3_held_out` blocks this session — these exact
   per-seed figures are NOT independently stated in
   `KEY_ANCHORING_DESIGN.md` §2.0 (which reports only the pooled range
   0.0049–0.0114 across both arms), so if a future editor cannot
   reproduce them from the JSONs above, flag it rather than assume a
   transcription error. The Track C/D/B numbers in Fig. 9, §6, and §9
   were all read directly from `EXPERIMENT_LOG.md`'s 2026-07-04 entries
   and cross-checked against `SCALE_TRANSFER_DESIGN.md`'s cited
   subsections; none required re-derivation from raw JSONs.
4. **[RESOLVED]** Exact §14 R2-4 gate mechanism and loss confirmed: Wave 0
   originally value-collapsed 10/10 seeds (entity-subspace rank ≈1.0,
   value salvage ≈0.000) — traced by a mini-audit to a target-index bug
   (scoring gathered `π^{h+1}(a)` instead of the queried `π^h(a)`), fixed,
   and paired with a pre-registered anti-collapse loss
   `L_train = L_cos + λ_nce · L_nce` (in-episode InfoNCE, `λ_nce=1.0`,
   `T=0.1`, fixed, not tuned) chosen specifically because collapse is a
   **zero-gradient stationary point** under the NCE term (not merely
   penalized) — the rerun is 10/10 collapse-free. This exact story is
   the concrete example cited in the methodology weave-in (§8 above) and
   belongs in §3/§4's setup description, not left as a footnote — it is
   the paper's second full illustration (after the F-geo-3 cross-episode
   drift catch) of the gating discipline catching a load-bearing bug
   before it produced a spurious headline number.
5. **[NEW, this revision — round-3 absorption status.]** This round
   absorbed: the 2×2 existence-proof design and its per-seed numbers
   (§0, §1, §5, Fig. 5), the §16.7 simulator correction (§1, §7, §8,
   Fig. 7, pre-mortem #12), the key-anchoring wave's build-and-gate status
   (§7, §6 items 4/8/9, §9 item 1), and the SCALE-TRANSFER Track B/C/D
   results (§0, §1, §6 items 1/6, §9 items 2/3/5, Fig. 9, claim-tier
   table). Stage-G H_e (§5 item 10c, §9 item 4) is included per the task
   brief's explicit instruction, scoped honestly as a sibling-architecture
   result, not a DeltaNet finding. **Nothing in this narrative currently
   depends on a wave that has not yet reported a number** — every "still
   landing" item (§9) is stated as pending with its own decision rule, not
   presupposed in any earlier section's prose. The three weakest points
   this revision itself introduces (a reviewer's most likely angle of
   attack, flagged for whoever drafts the camera-ready): (i) the 2×2's
   "jointly sufficient" corner still leans on the pool-restricted i-strong
   pin, not a learned mechanism — pre-mortem #11 answers this but the
   underlying dependency doesn't disappear until the anchoring wave lands;
   (ii) Fig. 9's production-scale panel is an honest non-finding dressed
   in figure form — it strengthens the paper's motivation, not its
   evidence, and a careless read could conflate the two; (iii) the
   thesis/abstract now carry more forward-looking, unresolved material
   (anchoring wave, rung-2/3, Stage-G aside) than round 2's fully-closed
   ending did — trim aggressively toward the resolved K=16/K=32/2×2 core
   for the camera-ready abstract, and keep the pending items to the
   "results still landing" section and limitations, not the headline
   claims.
