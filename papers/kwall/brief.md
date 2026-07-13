# Paper brief — kwall ("the composition wall, four levers, and where it holds")

Stage 0 of the `paper` skill (repo mode). This is a brief + outline +
evidence table, **not the paper itself**: no section is drafted until every
planned numerical claim below has a three-field evidence row (claim | source
§ | raw artifact), per the skill's own gate.

**Subject.** Native Composition Reads (NCR) — a novel fast-weight
architecture that writes relational operators in-context and reads
`o = read(Z^h, q)` (relation chains `Z_{r_n}···Z_{r_1}`) via exact O(log h)
repeated-squaring, no chain-of-thought — hits a scaling wall on its
composition-depth axis K (the number of entities forming the single group
the model must compose over). This brief covers the closed, fully-recorded
attack on that wall: four pre-registered levers, a measured mechanism, a
methods byproduct, and a bound. NCR itself already survived the full
novel-architecture waterfall (brainstorm → attack → Rev 1 → attack → Rev 2
→ zero-defect rule-math verification, `NOVEL_ARCH_WATERFALL.md` §1–§5,
DESIGN-CLEARED-FOR-BUILD-QUEUE) and a first scaling ladder (§9) that
**discovered** the wall this paper is about — that discovery, and
everything after it, is what this paper reports.

**Naming collision, disambiguated up front (source of real confusion in
this repo — must be handled explicitly in §2).** An unrelated, older
"K-wall" already exists in this codebase: `EXPERIMENT_LOG.md` 2026-07-02
"Task E K-wall resolution," a **training-budget** wall in a *different*
task/architecture (Task E matrix-permutation composition, not NCR) that
*was* rescued by more steps alone (K=12: 0/5→3/3 at 80K steps;
`papers/rank-recruitment-ws/brief.md` R6). This paper's K-wall is a
different phenomenon in a different architecture, only rescued by more
steps at K=16 (partially) and never at K=32 — the paper must name the
Task E precedent once, explicitly, and never conflate the two.

## Title candidates (3 — PI chooses; T-BOUND recommended as the drafting
title because the bound is the honest headline)

**T-BOUND (recommended):** *"The K-Axis Closes at 32: Four Levers
Against a Fast-Weight Composition Wall"* — echoes the registry's own
closing language (§11.6: "this CLOSES the K-axis book at K=32") rather
than a looser paraphrase; leads with the disclosed boundary, not the
fix, matching the house convention of naming the paper after what
actually held up (cf. "Three Bounds on a Null").

**T-MECHANISM:** *"Leakage, Not Capacity: A State-Dimension Convention
Fixes — and Bounds — a Composition Wall"* — leads with §11.4a's
measured mechanism (7–14× leakage difference) and the convention fix,
with the bound in the subtitle.

**T-PROCESS:** *"The Map Overrules the Reader: Pre-Registered Verdicts on
a Compositional-Depth Wall"* — leads with the process story (binding
decision maps overriding informal reads twice, an adversarial catch
reversing a lenient gate before launch); mirrors "The Instrument Is the
First Suspect" (`papers/measurement-ws/`, same house, published-adjacent).

## Venue

Workshop-scale (honest negative-with-mechanism + a methods instrument +
a process discipline story) — the same class as `papers/measurement-ws/`
and the reasoning-null-moss paper, not a capability-separation flagship
claim. Re-verified live this session (2026-07-12/13), not merely cited
from `papers/VENUE_MAP.md` (2026-07-10 scout):

- **Recommended: MOSS @ COLM 2026 — late window.** CFP text, re-fetched
  this session: *"We will allow later submissions depending on
  capacity"* (deadline was Jul 3 '26 AoE). Small-Scale Frontier Track,
  4pp max main content, non-archival, no official proceedings, dual
  submission OK; exact scope fit (rigorous small-scale science, honest
  negative-with-mechanism, ≤3B-param testbeds — every NCR cell in this
  program is under 200K params, per the pre-registered `P(d,h)` formula
  in `NOVEL_ARCH_WATERFALL.md` §11/§9.3). Same venue class already used by this program's other honest-negative
  papers (`papers/reasoning-null-moss/`, and the MOSS backup slot in
  `papers/measurement-ws/`). **Action:** send the late-add capacity
  request before drafting, per the standing `VENUE_DECISION.md`
  precedent — do not assume acceptance.
- **Backup: 2nd Workshop on Efficient Reasoning @ COLM 2026 — OPEN,
  deadline Jul 19, 2026 AoE.** Re-verified live this session (page
  states the deadline was extended from Jul 12 to Jul 19 AoE); 4–10pp
  COLM format, double-blind, non-archival, accepts work under review
  elsewhere. Topic fit is weaker than for this program's memory/M*
  paper (that CFP's named topic is KV-cache/memory management; this
  paper is about compositional-depth trainability, adjacent but not
  identical) — flagged, not disqualifying. **Clock risk:** the K-wall
  evidence base closed only hours before this brief; 7 days from
  "today" to a from-scratch 4pp draft is the tightest turnaround of any
  venue considered here.
- **Third option, UNVERIFIABLE THIS SESSION — flagged, not proposed as
  primary:** a NeurIPS 2026 measurement/evaluation/science-of-DL-class
  workshop. `papers/VENUE_MAP.md` recorded the accepted-workshop list as
  not yet posted on 2026-07-10 with notifications due "Jul 11 '26 AoE."
  Re-fetched `neurips.cc/Conferences/2026/Workshops` live this session:
  **still 404** — the list has not landed even though its own
  notification date has passed. Cannot be finalized; re-check before
  committing to it.
- **Format assumption for the outline below: 4.0pp main text** (the
  strictest of the three candidates — MOSS's own hard cap — matching
  `papers/measurement-ws/`'s same choice for the same reason).

## Thesis (one falsifiable sentence)

> In Native Composition Reads, a parameter-free training-time LayerNorm
> crutch (annealed out; eval always reads pure exact matmul) recovers
> convergence a plain raw-matmul recipe cannot, and — combined with a
> corrected ambient-dimension convention (state dimension d = K+1, not
> the field-default d = 2K) — extends genuine far-depth exact
> composition from K=14 to K=24 entities; but the fix is bounded, not
> general: at K=32, every tested (dimension, training-budget) combination
> — 24 distinct, independently re-verified cells (16 across 4 dimension
> ratios at 1× budget, plus 8 more at 2×/4× budget on the best-performing
> ratio) — fails to reach robust convergence, and far-depth recovery reads
> exactly 0.0 in literally every one of them, closing the K-axis under a
> pre-registered stopping rule.

**Falsifier (concrete, would void the bound half of the thesis):** any
single K=32 cell, at any tested d ∈ {K+1, 1.25K, 1.5K, 2K} × budget ∈
{1×, 2×, 4×}, reaching CONVERGED-ROBUST (≥3/4 seeds clearing in-dist
recovered_frac@0.9 ≥ 0.9 and A_eff_rank ≥ 0.9K) with median far-depth
recovery at h*=253 ≥ 0.9. None did, across all 24 distinct cells
independently re-pulled and re-verified this program.

## Background / setup (for §2 of the outline)

**NCR in one paragraph.** A fast-weight state Z (d×d) is written
in-context from relation tokens and read via `o = read(Z^h, q)`,
computed by O(log h) repeated squaring — exact, not an attention-weighted
approximation. K is the size of the single cyclic group of entities the
model must learn to compose over in a given cell; h is composition depth
(number of hops/relation applications); the pre-registered target depth
is `h* = 8K-3` (the m=8 rung of a geometric ladder `m·K-3`).
`failure_front_h` ("front") is the sweep depth beyond which recovered
accuracy falls below the band; front = K-3 is the **trivial** front (no
real composition beyond the shortest train residues). `A_eff_rank` is the
effective rank of the learned entity-cycle operator (bar: ≥0.9K).
`δ` = `phase_resid_max_mean`, the write-quality residual; the theoretical
conservative hold-horizon is `arccos(0.9)/δ = 0.451/δ` hops (worst-case
coherent phase drift), used throughout as a predictive, not gating,
quantity.

**Gates, reused verbatim throughout (pre-registered once, §11, never
re-invented per cell).** GATE 1 (convergence), per K over 4 seeds:
in-dist (h=1,2,3, min) recovered_frac@0.9 ≥ 0.9 AND A_eff_rank ≥ 0.9K →
per-cell CONVERGED / PARTIAL (in-dist ∈ [0.5,0.9)) / DEAD (<0.5);
per-K CONVERGED-ROBUST (≥3/4 CONVERGED) / CONVERGED-PARTIAL (1–2/4) /
TRAINABILITY-DEAD (0/4). GATE 2 (far-depth), scored only on CONVERGED
cells: recovered_frac@0.9 at h* and along the ladder, banded HOLD (≥0.9)
/ DEGRADED (0.5,0.9) / FAIL (≤0.5).

**The origin finding (K0, background not a headline claim).** The plain
raw-matmul recipe shows a sharp, single-seed trainability cliff between
K=14 (CONVERGED: in-dist 1.000, shadow-certified against fp64 at
~5×10⁻⁸, A_eff_rank≈14.0, δ=0.0072) and K=15/K=16 (DEAD: never trained,
loss flat ~0.99→~0.99, rank collapsed to ≈1) — discrete collapse, not
gradual degradation (`NOVEL_ARCH_WATERFALL.md` §9.10).

## Contribution bullets

1. **A trainability fix, replicated.** On the operator-count (R) axis at
   R=3, a parameter-free early-LN crutch (blended in at training only,
   weight annealed 1→0 over the first half; eval always the inherited
   pure-matmul exact read) RECOVERS convergence (in-dist 1.0/1.0/1.0,
   A_eff_rank 7.98–7.99) where three matched baselines (control, warmup,
   curriculum) all FAIL (0.0/0.0/0.0 each), replicated 9/9 seeds with
   0/9 dead (§8.9 n=1 + §8.10 n=9 extension). Transplanting this recipe
   to the K (entity-count) axis is this paper's starting point.
2. **Four pre-registered levers on the K-wall, every one reported.**
   Training budget → **NO-LAW** (Gate-1 convergence improves
   monotonically with budget but the write-residual is non-monotonic in
   3/4 seeds and a converged seed's far-depth front regresses; two of
   three pre-registered anomaly triggers fire, barring any fitted law).
   Anneal-shape → **falsified at K=24** (indistinguishable from
   baseline on every metric), directional-only at K=16 with zero
   far-depth movement. The state-dimension convention (d=K+1 vs the
   field-default d=2K) → **the one lever that works**: moves Gate-1
   convergence from 1/4 to 4/4 at both K=16 and K=24 on 1× budget alone,
   and at K=16 buys genuine far-depth recovery (2/4 seeds clear the 0.9
   HOLD band at h*=125, reading 0.9471/0.9877, and the same 2 seeds reach
   one ladder rung further still) that no d=2K cell ever produced at any
   of 16 tested budget/anneal combinations (rec@h* stayed 0.0000–0.0001
   in every one).
3. **Mechanism measured, not asserted.** Leakage into the
   ambient/cross-term subspace is 7.1× (K=16) and 14.1× (K=24) larger at
   d=2K than d=K+1; the qualitative shape differs too — at d=K+1, 93–98%
   of the (small) leak sits harmlessly in the single spare dimension; at
   d=2K, the (much larger) leak migrates into the cross-terms that touch
   the K-dimensional entity subspace the readout depends on. Independent
   opus audit CLEAN with bit-for-bit numerical replication.
4. **The bound — the honest headline.** The convention fix does not
   generalize past K=24. At K=32, all three tested d-ratios (K+1, 1.25K,
   1.5K) plus the d=2K reference fail to reach CONVERGED-ROBUST at 1×
   budget; front is pinned at the trivial K-3=29 rung and far-depth
   recovery reads 0.0000 in all 16 cells. A dedicated budget-rescue
   probe on the best-performing arm (d=K+1) reaches only 1/4 and 2/4
   fully-CONVERGED at 2×/4× budget (still short of the 3/4 robustness
   bar) while front stays pinned at 29 in all 12 of those cells and
   rec@h* stays 0.0000 in every one — and the same anomaly trigger that
   fired at K=16's 4× probe fires again, pre-empting a verdict label.
   This CLOSES the K-axis book at K=32 under the pre-registered staging
   rule, which formally blocks the next rung (K=48).
5. **A methods byproduct.** Far-depth composition ability is predictable
   from the write's phase-residual δ alone, without running the deep
   test: Spearman ρ(δ, front) = −0.8771, ρ(δ, sweep_min_rec) = −0.8734
   (n=12, K=24@d=25 seed extension) — both exceed the pre-registered
   |ρ|≳0.6 bar, both negative sign as predicted. A cheap, loss-blind
   proxy for a metric that otherwise requires a full depth sweep.
6. **A process contribution.** Every verdict was pre-registered with a
   binding decision map, and the maps repeatedly overrode informal
   reads: a "law-flattens" impression was ruled NO-LAW by the pinned
   anomaly trigger (the recorder's own words: *"this differs from this
   recorder's own pre-harvest informal read... the gate wins per the
   standing tiebreak rule"*), and — in the K=32 budget-rescue probe —
   the same anomaly clause fired and the recorder declined to assign any
   of the map's four labels rather than force a nearest fit. Separately,
   an independent adversarial pass caught an under-specified
   ladder-reopening threshold (satisfiable by any front past the
   trivial rung — inconsistent with the design's own worked example)
   and tightened it to require median front ≥ h* **before any K=32 or
   K=48 GPU cell launched.**

## Per-section page budget (4.0pp main text, MOSS/measurement-ws convention)

| Section | Pages | Purpose |
|---|---|---|
| 1 Introduction | 0.45 | The wall, the four-lever attack, the bound as the headline; contribution bullets |
| 2 Background & setup | 0.45 | NCR in one paragraph; K vs R axes; the gates; the K0 origin finding; the naming-collision disclaimer vs Task E's older K-wall |
| 3 The trainability fix | 0.35 | earlyln RECOVERED at R=3 (n=1) + n=9 replication; the transplant motivation to the K axis |
| 4 Four levers on the K-wall | 1.10 | Budget → NO-LAW; anneal-shape → falsified at K=24; the state-dimension convention → CONFIRM at K=16/K=24 |
| 5 Mechanism | 0.45 | §11.4a leakage magnitude/shape; the D_share flip at K=24 |
| 6 The bound | 0.45 | K=32's full d(K) grid + budget-rescue dissociation probe; CLOSED-AT-THIS-K; WAVE-1b blocked |
| 7 Predictability + process discipline | 0.35 | δ→far-depth Spearman correlation; the pre-registration overrides; the caught-and-tightened gate |
| 8 Related work & limitations | 0.40 | By-name distinctions; scope (K≤24 only; mechanism scoped to K∈{16,24}; anomaly unexplained) |
| **Total** | **4.00** | = the strictest candidate venue's main-text cap |

Appendix (not counted, MOSS "unrestricted supplementary"): full per-seed
tables for every wave, the anomaly trajectory tables (K=16 and K=32),
GPU-h ledgers per wave, md5 manifests.

## Claims-to-evidence-to-figure map

One row per numerical claim. Every row cites a real § in
`matrix-thinking/NOVEL_ARCH_WATERFALL.md` (or the two pre-registration
design docs) and a raw artifact verified to exist on disk this session
(directory listing + md5, not taken on trust from prose).

| Id | Claim (with the number) | Verdict record (§) | Raw artifact (path + md5) | Figure/table |
|---|---|---|---|---|
| K0 | Origin finding: plain-recipe cliff between K=14 (CONVERGED: in-dist 1.000, A_eff_rank≈14.0, δ=0.0072, shadow-certified ~5e-8) and K=15/16 (DEAD: never trained, rank collapsed ≈1) | `NOVEL_ARCH_WATERFALL.md` §9.10 | `experiment-runs/2026-07-11_ncr_wcap_diag/` (md5_manifest.txt; `ncr_ncr_K14_s0.json` md5:8d77d4d34525da69fbb9d5154a06ab49, `ncr_ncr_K15_s0.json` md5:b947c71325338c0ef9ea03060c5a84a4) | fig1 (background panel) |
| K1 | Trainability fix, n=1: earlyln RECOVERED (in-dist 1.0/1.0/1.0, A_eff_rank 7.98–7.99, swap gap +0.5526) vs baseline/warmup/curriculum FAILED (0.0/0.0/0.0 each, swap gap +0.0009/+0.0011/−0.0020 — no relation-selective read in any of the three) | §8.9 | `experiment-runs/2026-07-11_ncr_opbank_recover/` (md5_manifest.txt, 26 files; `ncropbank_recover_baseline.json` md5:dcf9f2283130cb311e32da7f335613fa; script `ncr_opbank_recover.py` md5:6007b092fb7e860757b45a20f233b6d5) | fig2a |
| K2 | n=9 seed replication: 9/9 converged (dead-rate 0/9); far-depth h*=61: 2/9 HOLD (≥0.9), 2/9 DEGRADED, 5/9 FAIL; the 4 seeds with max phase_resid ≤0.0086 are exactly the 4 with far-61 ≥0.615 | §8.10 | `experiment-runs/2026-07-11_ncr_opbank_seedrep/` (md5_manifest.txt, SUMMARY.md, 8 seed dirs + §8.9's own seed 0) | fig2b |
| K3 | K-scaling ladder: earlyln moves convergence K=14→K=15 (4/4 CONVERGED, Gate-2 median 0.9929 → HOLD → **SCALES**); wall re-forms at K=16 (1/4 CONVERGED) and K=24 (0/4, **TRAINABILITY-DEAD**); pooled worst-of = **TRAINABILITY-STILL-LIMITED**; earlyln also drops K=14's converged residual 0.0072→0.0020-0.0042; K=24 (a new rung with no plain-recipe baseline of its own) shows partial operator formation (loss →0.36, A_eff_rank→17.7/24) distinct from §9.10's flat-loss rank-1 collapse pattern, without converging | §11.2 | `experiment-runs/2026-07-11_ncr_earlyln_scale/` (md5_manifest.txt, 16 cell JSONs; K15 s0 md5:705d13beff52ac7de882c83773850e1a, K16 s2 md5:c189231ea176208f3a529b921e67d637, K24 s0 md5:581100720be799b91185ebd882db9d94) | fig3 (K-ladder panel) |
| K4 | Training budget = **NO-LAW**: K16 Gate-1 rate 1/4→3/4→4/4 CONVERGED at 1×/2×/4×, but δ non-monotonic (↓ then ↑) in 3/4 seeds and one CONVERGED seed's front regresses 29→13 between 2×/4×; 2 of 3 pinned anomaly triggers fire; extrapolation barred by the pre-registered rule | §11.3 + §11.4 (Q1) | `experiment-runs/2026-07-12_ncr_nextlever_wave/budget4x/` (4× cells; K16 s0 md5:8245a49b8dde009f41f86ffc4f7ebe21 … s3 md5:31c15859a5f6b64be6d1b966b33f609a) + `experiment-runs/2026-07-12_ncr_earlyln_budget2x/` (2× reference; K16 s0 md5:7627339add95f2fba9032d0a5d1e5687) | fig4a (per-seed 1×→2×→4× trajectory table) |
| K5 | Anneal-shape falsified at K=24: B-24 Gate-1 stays 0/4 CONVERGED (identical to baseline), front pinned at 21=K-3 in all 4 seeds (identical), δ dip ~12% inside the pre-existing seed-noise band (not material); B-16 partial/directional only (Gate-1 1/4→2/4, δ mean −38.5%) with front pinned at 13 in all 4 (zero far-depth movement) | §11.4 (Probe B) | `experiment-runs/2026-07-12_ncr_nextlever_wave/annealshape/` (md5_manifest.txt entries under `annealshape/`, e.g. K16 s0 md5:961881a811ef0af626c2d52274305208) | fig4b |
| K6 | State-dimension convention CONFIRM at both K: Gate-1 4/4 CONVERGED at d=K+1 vs 1/4 (K16@d32) / 0/4 (K24@d48) at d=2K, 1× budget alone; K16@d17 far-depth: 2/4 seeds clear the 0.9 HOLD band at h*=125 (0.9471, 0.9877) and the same 2 reach one ladder rung further (front=253) vs 0.0000–0.0001 in literally every d=2K cell across 16 tested budget/anneal combos; K24@d25 n=12 extension 12/12 CONVERGED but low reliability (sweep_min_rec: 0/12 HOLD, 1/12 DEGRADED, 11/12 FAIL; front≥h* 4/12) | §11.4 (Probe A) + §11.5 (Q2, n=12 extension) | `experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/` (K16 s0-s3 md5:26bcca7ce1b461f9899722f97973b995…69e36a2a331061a41cb90497905296d6; K24 s0-s3 md5:2661201feae0ed8ed6ad7b809f7fd361…437f7ad1e9389d402a1b3884133cf343) + `experiment-runs/2026-07-12_ncr_mappinglaw_wave1/q2_K24_seedext/` (md5_manifest.txt, seeds 4–11) | fig5 (K16 vs K24 vs K32 front/δ scatter) |
| K7 | Mechanism: normalized leakage 7.1× larger at d=2K vs d=K+1 (K16: leak_ratio 1.797 vs 0.254) and 14.1× larger (K24: 3.023 vs 0.215); at d=K+1, D_share 0.929–0.985 (leak sits in the spare dimension); at d=2K, D_share drops to 0.27 (K24)/0.55 (K16) — leak moves into entity-touching cross-terms; independent opus audit CLEAN, bit-for-bit replication (md5 2870ccb1...) | §11.4a | `experiment-runs/2026-07-12_ncr_q3_mechanism/` (`q3_mechanism_results.json` md5:2870ccb13df2ac87cf491cfd2cb62148; `analyze_dratio_blocks.py` md5:b8cde278883ec16c9aab7ce84ccff808) | fig6 (leak_ratio + D_share bars, K16 vs K24) |
| K8 | **The bound**: K=32 full d(K) grid (d=33/40/48/64) all 0/4 fully-CONVERGED at 1× (d=33 "least dead," 3/4 PARTIAL, still short of ROBUST); front pinned at trivial K-3=29 in all 16 cells, rec@h*=0.0000 in all 16 → **CLOSED-AT-THIS-K**, blocking K=48 (WAVE-1b). Budget-rescue probe on d=33: 1/4 (2×) and 2/4 (4×) fully-CONVERGED (still <3/4), front pinned at 29 in all 12 cells (all 3 budgets), rec@h*=0.0000 in all 12; anomaly trigger fires again (δ non-monotonic in 3/4 seeds), pre-empting a verdict label; the sole unblocking outcome (BUDGET-CONVERGES + median front ≥ h*=253) is independently excluded by the raw numbers | §11.5 + §11.6 | `experiment-runs/2026-07-12_ncr_mappinglaw_wave1/` (dratio_K32_d33/, dratio125_K32_d40/, dratio150_K32_d48/, scale_k32_2kref/; md5_manifest.txt) + `experiment-runs/2026-07-12_ncr_k32_budget/` (md5_manifest.txt; budget2x/4x K32 cells + SUMMARY.md md5:eb88c1f4bf5ea564cda7772b4efb3b38) | fig7 (K=32 12-cell + 8-cell grid, all-DEAD/PARTIAL heatmap) |
| K9 | Methods byproduct: far-depth composition predictable from δ alone — Spearman ρ(δ, front) = −0.8771, ρ(δ, sweep_min_rec) = −0.8734 (n=12), both exceed the pre-registered \|ρ\|≳0.6 bar, both negative sign as predicted → **δ-PREDICTABLE** | §11.5 (Q2 covariate result) | `experiment-runs/2026-07-12_ncr_mappinglaw_wave1/q2_K24_seedext/` (same archive as K6's n=12 extension; md5_manifest.txt) | fig8 (δ vs front scatter, n=12, Spearman fit) |

**Process citations (qualitative, no numbers → no evidence row needed,
same convention as `papers/flagship/brief.md`'s and
`papers/measurement-ws/brief.md`'s methodology citations):**

- **P1 — the map overrode an informal read.** §11.4's own words, quoted
  above in contribution bullet 6, citing the pinned anomaly-trigger rule
  in `NCR_NEXT_LEVER_DESIGN.md` §1.7.
- **P2 — the map overrode itself a second time by refusing to answer.**
  §11.6: *"the mechanical §4.6 verdict map is PRE-EMPTED per its own
  pinned rule, no (a)/(b1)/(b2)/(c) label is assigned"* — citing
  `NCR_MAPPING_LAW_DESIGN.md` §4.5/§4.6.
- **P3 — a lenient gate caught and tightened before launch.** An
  adversarial pass on the pre-attack draft of `NCR_MAPPING_LAW_DESIGN.md`
  found the REOPENS verdict-bar, as originally worded ("front moves past
  the trivial rung"), was satisfied by the design's own K24@d25 worked
  example (front=141) even though that example was simultaneously cited
  as the CONVERGES-ONLY (weaker) template — an internal contradiction
  the literal wording could not resolve. DISPOSITION: conceded in full;
  REOPENS rewritten to require median front ≥ the cell's own h* before
  any GPU cell for K≥32 launched (`NCR_MAPPING_LAW_DESIGN.md` §3,
  finding #1).

## Figures to generate

- `fig1` — background panel: the K=14/K=15 origin cliff (loss + A_eff_rank
  vs K, from `experiment-runs/2026-07-11_ncr_wcap_diag/`).
- `fig2` — (a) the 4-arm bar chart (earlyln vs baseline/warmup/curriculum,
  n=1); (b) the n=9 seed-replication far-depth distribution with the
  δ≤0.0086 cutoff marked.
- `fig3` — K-ladder panel: Gate-1 rate and Gate-2 median rec@h* vs K ∈
  {14,15,16,24}, SCALES/TRAINABILITY-STILL-LIMITED labels shown.
- `fig4` — (a) the K16 1×→2×→4× per-seed δ/front trajectory table
  (non-monotonicity visualized); (b) the anneal-shape B-16/B-24 bar
  comparison.
- `fig5` — front/δ scatter across K∈{16,24,32}, colored by d-convention
  (K+1 vs 2K), showing the convention advantage narrowing to nothing at
  K=32.
- `fig6` — leak_ratio and D_share bars, K16 vs K24, K+1 vs 2K (the
  single most mechanistically informative number: D_share 0.929→0.270
  at K=24).
- `fig7` — the K=32 grid heatmap: 4 d-ratios × up to 3 budgets, all
  DEAD/PARTIAL, front pinned at 29 throughout.
- `fig8` — δ vs front scatter (n=12, K=24@d=25), Spearman fit line,
  ρ=−0.877 annotated.

All figures generated from a single versioned `figures/figure-gen.py`
(house convention) that asserts input md5s against this map before
plotting — no figure ships from a number not in the table above.

## Nearest prior work (distinguish by name)

- **TensorLog (arXiv:1605.06523) / Neural-LP (arXiv:1702.08367) / DRUM
  (arXiv:1911.00055):** query-conditioned variable-length products of
  relation matrices — NCR's closest architectural neighbors on the
  composition mechanism itself; this paper's contribution is not the
  mechanism (established prior art, disclosed in the base design) but
  the trainability/scaling behavior of that mechanism as K grows, which
  none of these papers characterize.
- **Guu et al. 2015 (arXiv:1506.01094):** composition error-cascading,
  documented a decade before this program; §11.4a's leakage measurement
  and the δ→far-depth Spearman correlation (K9) are a mechanistic,
  measured instance of exactly this cascading, in a fast-weight matrix
  substrate rather than a symbolic KB-embedding one.
- **MesaNet (arXiv:2506.05233):** the closest recent query-conditioned
  matrix-function reader; distinguished by NCR's in-context (not
  pretrained-then-frozen) operator writing and the exact O(log h)
  repeated-squaring read this paper's mechanism section is about.
- **Looped Transformers (arXiv:2409.15647) / depth-recurrent
  architectures (arXiv:2603.21676):** already establish that some
  current architectures CAN do variable-depth iterated computation —
  this paper is not "no architecture can compose," it is a bounded,
  measured trainability/dimension-convention study of one specific
  exact-composition mechanism.
- **arXiv:2602.03655:** log-depth analysis of matrix powers — the
  theoretical grounding for NCR's O(log h) read; this paper is the
  empirical trainability/scaling counterpart, not a theory paper.
- **This program's own K-wall (`papers/rank-recruitment-ws/`,
  Task E, 2026-07-02):** a DIFFERENT architecture's DIFFERENT wall,
  rescued by training budget ALONE (K=12: 0/5→3/3 at 80K). This paper's
  central negative finding is that budget alone does NOT rescue NCR's
  K-wall (K4: NO-LAW) — the contrast is worth one explicit sentence,
  not a shared "K-wall" umbrella claim.

## What we do NOT claim

- We do **not** claim the K-wall is solved, or that NCR scales past
  K=24 under any tested recipe. The fix is bounded to K≤24; K=32 is a
  closed failure at every d and budget tested (24 distinct K=32 cells:
  16 across 4 dimension ratios at 1× budget, plus 8 more at 2×/4×
  budget on the best-performing ratio — 0.0 far-depth recovery in
  every one).
- We do **not** claim to know WHY K=32 fails at every dimension,
  including the d=K+1 convention that works at K≤24. §11.4a's leakage
  mechanism is explicitly scoped to K∈{16,24} only (its own "Coverage"
  note states this) — no K=32 leakage analysis exists.
- We do **not** claim the training-budget anomaly (δ non-monotonic,
  observed at both K=16's 4× probe and K=32's 2×/4× probe) is
  understood. It has no loss-curve signature (checked once, K=16) and
  is reported as an open, twice-observed phenomenon, not resolved.
- We do **not** claim this result generalizes across the R
  (operator-count) axis. The trainability fix and its n=9 replication
  are at R=3 only; the K-scaling ladder is single-relation (R=1) only.
- We do **not** offer a benchmark comparison against published
  length/depth-generalization baselines. This is a within-architecture
  ablation program (NCR under different training levers/conventions),
  not a leaderboard claim.
- We do **not** re-claim NCR's base survival of the novel-architecture
  waterfall (§1–§5) as a new result of this paper — it is cited as
  precedent/background only.
- We do **not** claim the state-dimension convention (d=K+1) is optimal.
  The s-sweep that would establish the shape of the true d(K) mapping
  law (d ∈ {1.25K, 1.5K} at intermediate points) was run only at K=32,
  where it showed no material advantage over d=2K — untested at K≤24
  beyond the two endpoints already reported.

## Anonymization surface (double-blind assumed per MOSS/Efficient-Reasoning CFPs)

Tokens per the eventual `venue-requirements.md`: `Sam Larson`,
`samlarson16`, `samuellarson`, `learned-representations` (repo name),
`youthful-indigo-turkey` (Brev instance). Design-doc filenames and
internal § anchors must not appear in rendered prose (source comments
exempt).

## Dual output

- [ ] Venue submission (anonymized, 4.0pp, venue TBD per the three
  candidates above — MOSS late-window email must go out before drafting
  starts)
- [ ] Public write-up (pebbleml.com findings page — this program's
  per-finding-publisher convention; consumes the same evidence map)

Neither has been produced yet — this file is Stage 0 only. The evidence
map is freeze-ready: all 10 numbered rows (K0–K9) plus 3 process
citations (P1–P3) verified against raw artifacts this session, 2026-07-12/13.
