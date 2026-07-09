# STATE — Current Project State

**Last updated:** 2026-07-09 (new-sprint consolidation; per-event history
lives in `EXPERIMENT_LOG.md`, the design docs' § sections, `experiment-runs/`
archives, and git history — see "Documentation Map" at the end of this
file. This pass compressed the prior event-stack narrative into the
ACTIVE CAMPAIGNS section below; nothing was deleted — every compressed
block still has a pointer to its canonical design-doc record.)

## GOALS

1. **Publish the drafted material.** One paper PUBLISHED ("The Gradient
   Does Not See Rank", ICML 2026 MI workshop). Drafts: `neurips-ws-2026/`
   (positive rank results; venue+cut decision pending, ~Jul 11 CFP —
   IMMINENT), `workshop-2026/` (capacity trilogy), `iclr-2027/` (the
   full main-conference paper, complete draft, deadline ~late Sept 2026).
2. **Multi-workshop strategy (PI, 2026-07-08):** chop the result
   inventory into many workshop submissions beyond the two dated ones
   (reasoning-link null program, instrument-calibration/fake-cliff
   methodology story, the coming M* memory result, the coming
   capability-separation result). A venue-scout agent maintains the CFP
   pipeline.
3. **Land a full (main-conference) publication.** The ICLR 2027 draft is
   the vehicle; PI wants Berkeley/Stanford collaboration; PI publishes it
   regardless, but it needs a POSITIVE result to matter.
4. **The overall research goal (PI, verbatim intent, 2026-07-08):**
   demonstrate CAPABILITIES current architectures are incapable of,
   functionally or as observed/tested — capability SEPARATION, not
   efficiency, is the world-changing headline. Matched comparisons (the
   head-to-head) remain the grounding. Modality (language/bytes/other)
   is an open question, settled by the waterfall, never bundled
   two-unproven-axes-at-once.
5. **GPU saturation (PI, verbatim, 2026-07-09):** *"how will [we]
   ensure that all these 8 gpu's are hot for the next few days. I
   don't want these sitting idle anymore."* Queued science totalled
   ~20 GPU-h against ≈192 GPU-h/day supply — the pipeline's cells were
   too small to saturate. Response (the authorization the
   never-self-amend rule requires): two compute-heavy waves chartered
   (FIX-AT-SCALE, CAPABILITY STAGE 2, both below) plus liberal
   pre-registered seed extensions; all 8 GPUs in play, GPU 7 no longer
   reserved.

## ACTIVE CAMPAIGNS

### 1. Head-to-Head Demo (PI-ratified 2026-07-08)

Does a matrix-native fast-weight model (frozen-bias fix + recruitable
rank + super-linear capacity + exact composition) beat matched
baselines on data-efficiency (primary 1) and inference-memory-matched
long-horizon tasks (primary 2), FLOP-matched as a disclosed control
only? Registry: `matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md`. Design
cleared after 4 attack-round/revision cycles (§1.13-§1.19,
DESIGN-CLEARED-FOR-BUILD) → build → audit → deploy → calibration
rounds 1-2, both FAILED (gate-1 `rf@0.9`=0 all arms; the pre-registered
`aux_weight` dial [0.1→2.0] achieved gradient parity but every arm
plateaued) → **§1.21 diagnosis: the objective had no recall pressure**
— CE is structurally retrieval-blind (each key appears once at bind
time, queries never enter CE) and the aux loss converged to an
episode-membership local optimum (predicted-vector cosine matches the
analytic `1/√K` ceiling exactly in every arm — the "membership-oracle"
finding) → **Rev 4** adopted a three-term objective (answer-token CE at
the query position, all three arms symmetric, P=1 bottleneck preserved
by causality via the continuation construction) → **§1.23 attack round
5: CLEARED-FOR-BUILD-FIX** (fairness adjudicated axis-1-immune; ledger
gap reconciled) → build-fix (`cc89a4f`; an import-chain bug found and
fixed, `1db9594`) → **§1.24 scoped build-fix audit: RESIDUAL-FINDINGS**
— fix logic sound, but the CE_answer continuation wastefully runs the
50,259-way LM head over all 128 padded positions before discarding
127/128 (AUD2-F1, MAJOR, ≈4× the priced round-3 cost), and one
precision test is missing (AUD2-F2, K-restricted-argmax not defended).
**STATUS: PRE-LAUNCH FIX ACTIVE** (LM-head slice-before-matmul + a
fresh on-box timing pilot to re-price round 3 and the 27-cell sweep);
**calibration round 3 PENDING**. Margin freeze is additionally blocked
on the fix-at-scale campaign's escalation-cost correction and its
global-vector-arm probe (item 4 below) before the contender pin is
final — see PENDING PI DECISIONS.

### 2. Capability Separation — Stage 1 (PI capability-first directive, 2026-07-08)

Does causal rank↔representation-dimension recruitment separate
solvable/non-solvable finite groups at matched dims (S4-vs-A5
dissociation, both dim 3)? Registry:
`matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md`. Design cleared after
5 attack rounds (§1.21) → build → audit → deploy → calibration (5/5
cells, 0.0179 GPU-h/cell) → a gate-1 review found A5/A6 apparently
under-converged and real-checkpoint degauging near-zero (mean_cos
−0.02..0.17) while a synthetic injection PASSED → **§1.25 diagnosis:
TWO instrument defects proven, models healthy.** Defect 1 (FATAL): the
uncentered covariance-SVD is mathematically degenerate for Option A
targets (`Z≈c·(ρ⊕I)` orthogonal ⇒ `ZZᵀ≈c²·I`, isotropic) — a PERFECT
model fails the production bars; fixed by centering (group-mean of a
nontrivial irrep cancels the constant block). Defect 2 (FATAL): M1/M3's
scored words fell exclusively on `nn.Embedding` positional rows that
never train at the pinned lengths. **Corrected-lens rank preview: rank
tracks `d_min` at Spearman ρ=0.9747** (restricted effective rank
1.85/2.74/2.64/3.73/3.91-4.54 vs `d_min` 2/3/3/4/5, every group inside
`[0.7,1.3]·d_min`, the marquee S4/A5 pair landing together). **CAVEAT**
(standing): the restriction window partially favors rank≈`d_min` for
near-orthogonal blocks; M3's causal force-rank arms remain the decisive
test. → Rev 5 (fixes folded in) → micro attack round 6 (§1.27,
NEEDS-REVISION: the new per-L convergence bar is self-contradicted by
the real calibration data) → Rev 6 (§1.28: bar narrowed to `L∈{1..5}`,
`L∈{6..8}` demoted to disclosed-only, a second-consecutive-miss
HARD-STOP rule added) → **§1.29 coordinator raw-data tiebreak: round
6's "all 7 cells clear L1-5" claim is FALSE against the raw JSON;
Rev 6's structural fix is correct, but the HARD-STOP it triggered fired
on a wrong premise** — the L=1 dip is real but IMPROVING with budget
(A5 +0.038, A6 +0.059, 8K→20K steps), i.e. slow convergence, not a
plateau → adjudication round 7 dispatched (bar re-pin options + a
≤0.1 GPU-h L=1 mechanism micro-diagnostic). **§1.30 ROUND 7 VERDICT
(2026-07-09): MECHANISM FOUND, proven five independent ways — at L=1
the reader's attention is PROVABLY query-independent (softmax over a
single key; read-vector std across queries = exactly 0.0 vs 0.41 at
L=2); the deficit is generator-order-specific (order-5 depressed
0.74-0.86, order-3 fine); §1.29's own budget-extrapolation is
FALSIFIED at 40K (A5/A6 gains collapse to +0.001/+0.010 per +20K
steps — a genuine plateau, not slow convergence after all).
**HARD-STOP LIFTED.** Rev 7 (binding): HARD bar → `L∈{2..5}` only
(L=1 demoted/disclosed with the mechanism note); per-group budget pins
(S3=8K, S4=20K, A5=20K, S5=8K, A6=40K, all clearing their bar by
≥0.02 margin); escalation rule recalibrated (≤2/group, second miss →
mandatory mechanism diagnostic before any further action, HARD-STOP
reserved for genuine pathology). **STATUS: all 58 cells now
LAUNCHABLE (≈2.51 GPU-h raw)**, pending a micro-attack on Rev 7's own
delta + 4 still-outstanding production build items (centered-covariance
readout, train-length sampler, ambient injection gate, per-L/per-group
reporting) + a build audit — sweep not yet authorized.

### 3. Attractor-Robustness 2×2 (novelty stress-test follow-on, 2026-07-09)

A PI-skepticism-driven novelty stress-test asked the dangerous question
about the write-geometry attractor directly: is it just the well-known
qk-norm eigenvalue-stability issue in disguise? **Verdict:
HOLDS-WITH-NARROWED-SCOPE.** Every run used `use_qk_l2norm_in_kernel=
True` throughout (code-verified, `lm_pretrain_rd.py:984` — the same
stock mitigation Kimi Linear arXiv:2510.26692 §4 and Qwen3-Next cite);
qk-norm conditions single-vector eigenvalue stability, the attractor is
cross-key POPULATION geometry — a different axis — and two stronger
interventions (Gram penalty, ZCA whitening) already failed to fix it.
Follow-on funded: a 2×2 qk-norm×gating screen at 14M. **Built**
(`55f0cfc`: flag-gated `use_qk_l2norm_in_kernel`/`gated_delta_active`
axes in `lm_pretrain_rd.py`, additive and off-by-default). **Audit-
corrected** (`f09254a`): the escalation trigger's noise-floor constant
had silently pooled 4 out-of-distribution probe corpora instead of the
same-corpus statistic (`build_tidy.py:22`), understating
openr1-mix-ext's true seed noise by 2.65×, biasing screening toward
false-positive escalation on pure noise; recomputed directly from raw
archived JSONs (openr1-mix-ext 2.244355, wikitext-mix-ext 2.216699,
population std ddof=0); `rec@0.9` marked explicitly NON-DECISIONAL in
the report schema. **Deployed**: 145-file box sync EMPTY-DIFF
(`experiment-runs/2026-07-09_attractor_2x2_deploy/SYNC_RECORD.md`).
Box smokes ALL PASS — item [18]'s gated-arm backward initially failed
on Hopper/Triton≥3.4 (fla issue #640, wrong-result path) and required
provisioning the `tilelang` backend on-box (env-only fix, no kernel
code touched; the default non-gated path stays on the same Triton
kernels every prior chapter used, unaffected). **STATUS: screening
wave LAUNCHED** (4 cells, n=1, ceiling ≈1.01 GPU-h; n=3 escalation to
12 cells, ≤3.03 GPU-h, only if the screen splits qualitatively) —
harvest not yet returned as of this consolidation pass.

### 4. Fix-at-Scale (PI GPU-saturation charter, 2026-07-09)

Trains the frozen-bias fix (per-token, λ=0.58, literal transplant of
rung-1's construction) at 98M and 392M, extending its ONLY training
evidence (14M-only today) up the same ladder the write-geometry
pathology is already measured on (span_frac 0.248→0.344→0.389→0.455,
14M→1.31B). Registry: `matrix-thinking/FROZEN_BIAS_LM_DESIGN.md` §13.
**Wrong-direction-fix flag, stated up front by the design itself:** the
per-token fix's own 14M evidence is NOT stabilizing — it moved OPPOSITE
the mechanism prediction (span_frac +0.1955/+0.2273, CI-excludes-zero,
MORE collapsed than the artifact-matched control); only the
never-scaled GLOBAL-VECTOR arm actually stabilized it (−0.3319/−0.2308,
CI-excludes-zero). Three pre-registered, equally-publishable outcomes
(WIN = per-token reverses to the mechanism-predicted direction at
scale; PARTIAL = the destabilizing pattern persists on ≥1 corpus; NULL
= CI includes zero both scales) — a repeat of the 14M sign is itself
informative, not assumed away. Explicitly does NOT invoke §6.2's formal
rung-2 gate (rung-1 read as the FOURTH OUTCOME, "sim-training
divergence," never a CONFIRM) — authorized on a separate basis (the
saturation directive + the paper's own disclosed 14M-only caveat).
**Design Rev 0 (`660cffc`) → §13.13 attack round 1: NEEDS-REVISION**
(2026-07-09) — the wrong-direction claim verified TRUE against the raw
tables; the h2h contender pin (per_token) is confirmed SOUND, not a
misread (it was chosen for disclosed engineering reasons — Newton-
Schulz/β-uniformity stability + a clean val-loss gate — not because
anyone missed the sign); the original ≈170 GPU-h planning figure's
6× per-step rate error CONFIRMED and traced (`HEAD_TO_HEAD_DEMO_DESIGN.
md:1604`, wave-total ÷ one cell's steps). **BINDING on Rev 1:** (1) add
an exploratory-tier global-vector (Arm 2′) probe, n=1 both scales, ≈4
cells ≈18-37 GPU-h — without it the wave only re-tests the construction
already known to worsen the pathology and never learns whether the arm
that WORKS transfers; (2) VRAM logging folded into the timing pilot;
(3) wire `assert_blind_not_broken`, not just intend it. Cost (Rev 0):
primary recommendation n=3/n=3 ≈244.44 GPU-h (2× contingency); fallback
n=2-at-392M ≈200.58 GPU-h. **Proposed ledger: `fix-at-scale`, cap 300
GPU-h** (headroom-cap, not target; a genuinely NEW, separate ledger).
**STATUS: Rev 1 LANDED (`c6436fb`) → micro-attack round 2
(2026-07-09 overnight): DESIGN-CLEARED-FOR-BUILD, recorded as §13.15** —
all headline numbers re-derived exact (281.04/300, margin 18.96 ≈ 6.3%),
probe transplant faithful, §13.13 byte-intact; three non-blocking
findings BINDING ON BUILD (shared-forward-pass comparator or book
≈2.79 GPU-h; fix the fabricated "BINDING item 4" cross-ref; stamp the
promised tier string directly, `wrap_exploratory()` hardcodes a
different one). BUILD AGENT DISPATCHED (supervisor, 1.5× per-cell
abort, `assert_blind_not_broken` + negative test, probe wiring) →
independent build audit → deploy (md5) → LAUNCH (392M GPUs 0-3,
98M 4-6). Pilot-tier launch agent separately putting early cells on
box (report pending). Cross-cutting
finding dispatched separately: a terminology audit (CLAUDE.md's "FIXED
(frozen-bias)" and the paper's "stabilization" language overclaim the
DEPLOYED per_token arm) — corrected in CLAUDE.md's Research Direction
and this file's Campaign Scorecard, below.

### 5. Capability Separation — Stage 2 (design in progress, 2026-07-08/09)

Compositional depth generalization: does a matrix-state model trained
on group word-products of depth ≤ D_train answer held-out depths
D_test ≫ D_train in a single forward pass (no CoT), tracking algebraic
structure, while capacity-matched baselines degrade toward chance?
Registry: `matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md` §2,
**design Rev 0 (`d8d71d9`), dispatched under the saturation directive
to proceed IN PARALLEL with Stage 1's gauntlet** — launch stays gated
on Stage 1 reaching CONFIRM or a diagnosed INCONCLUSIVE (§1.11),
unaffected by which way Stage 1's round 7 (above) resolves. Central
finding driving the design: three independent lines — §1.25's
untrained-positional-row defect, §1.27/§1.29's convergence anomalies,
and an independently re-read circuit-complexity theorem (Grazzi et al.,
ICLR 2025 oral, arXiv:2411.12537) — converge on the same conclusion: a
fixed-depth Transformer with positional embeddings is the WRONG
architecture for a depth-generalization claim, not merely under-tuned.
Stage 2 therefore adjudicates a genuinely new architecture (recurrent
per-token state-composition) and a genuinely new expressivity axis
(β∈[0,1] vs β∈[0,2], parked by Stage 1 as a non-load-bearing smoke test
for exactly this later wave). **STATUS: attack round 1 RETURNED
(2026-07-09 overnight) — NEEDS-REVISION.** Verdict recorded in
`matrix-thinking/CAPABILITY_STAGE2_ATTACK_R1.md` (satellite file; fold a
§2.10 pointer into the registry at next touch — it was under concurrent
edit by the Stage-1 build agent at record time). Two launch-blocking
MAJORs: (1) the reused `row_queries` readout plausibly re-triggers the
PROVEN §1.30 query-independence degeneracy via low-rank reshaped states —
binding fix = a read-vector-std query-dependence diagnostic in the
calibration-first gate; (2) §2.1 vs §2.6 CONFIRM-criterion contradiction
(A5 gating vs open) — reconcile to §2.6's S5+A6-only wording. Plus:
adjudicate bespoke fp32 torch recurrence vs fla kernel at build; add an
n_h=4 calibration cell; cite Barrington 1989 for the A5/A6 exclusion legs;
pin the last-K-window trigger numerically. Grazzi transcription and the
depth-coverage grid were independently VERIFIED sound. Rev 1 dispatch
QUEUED behind the Stage-1 Rev-7 build agent (same-file guard).

### 6. Novel-Architecture Waterfall (opened 2026-07-09, stages 1-2 RETURNED)

A fresh brainstorm→research waterfall (CLAUDE.md process) on candidate
architectures beyond the current DeltaNet-family contender. **TOP
CANDIDATE (to attack stage): Native Composition Reads** — read via
query-selected matrix powers/products of the fast-weight state
(`o = read(Z^h, q)`, generalizing to relation chains `Z_rn···Z_r1`);
capability claim = single-pass variable-depth exact relational
composition, no CoT; novelty OPEN (closest prior art on different axes:
fast-weight PKM arXiv:2601.00671, MAGNA arXiv:2009.14332, DeltaProduct
arXiv:2502.10297); first wave ≈35-50 GPU-h on the Task E harness;
PonderNet-style halting-collapse kill-shot pre-answered (a closed-form
`‖C‖·h` leakage stopping rule, not a learned halting scalar); unified
with a multi-relation operator-bank idea as a gated second
sub-experiment (RotatE arXiv:1902.10197 is the prior art to
distinguish — theirs offline/static, ours online/in-context/causally-
verified). **SECOND TRACK (parallel-able):** rank-budgeted writes
(per-context rank allocation at the write step; novelty gap verified
against arXiv:2602.04852/2602.02195, both descriptive-only, and Elastic
Spectral SSM's global-only budget); ≈25-35 GPU-h. **Cheap piggyback:**
an orthogonal-complement novelty detector on already-archived Z-dumps
(near-zero GPU cost, a unique instrument). A DO-NOT-BUILD list
(Grazzi/DeltaProduct/RWKV-7/TPR/RotatE territory) was recorded in the
waterfall transcript. **STATUS: attack round 1 RETURNED (2026-07-09
overnight) — NEEDS-MAJOR-REVISION.** Full record in the new canonical
registry `matrix-thinking/NOVEL_ARCH_WATERFALL.md` §2. No literal
kill-shot, but: learned h-selection is DEAD both ways (soft = matrix
polynomial = MAGNA/MEA territory + destroys spectral exactness; hard =
PonderNet-collapse trap — the `‖C‖·h` rule bounds numerical trust, it
does not select h); single-K-cycle "held-out depth" collapses mod K;
the linear stopping rule is UNSOUND vs our own ρ(D)=1.0-2.9
measurements (must be geometric + scale-normalized, with a negative
test); closest prior art was missing (FWM arXiv:2011.07831 —
recursive fast-weight reads for transitive inference, 2020; Neural-LP/
DRUM for operator chains) — surviving novelty is the CONJUNCTION only.
**Surviving narrowed claim: in-context-written operators + EXACT
composition + O(log h) repeated-squaring reads** (separation from CoT
O(h) tokens, looped transformers O(h) loops, Stage-2 composer O(D)
steps, FWM fixed-N_r approximate). Rev 1 DISPATCHED (design-only, no
GPU): input-supplied h, single relation, binary-exponentiation read,
param-matched looped baseline w/ same h signal, FWM distinguished +
baselined, corrected stopping rule, Stage-2-style ledger (35-50 GPU-h
was 2-4× light); wave 1 launch gated behind Stage 2's calibration
readout (M4). Pre-registered risk to answer: pin the regime where O(h)
baselines measurably FAIL (else it reads as efficiency, not the PI-bar
capability separation).

## CAMPAIGN SCORECARD (Jul 6-9 2026, all pushed)

**FOR the approach:** SGD recruits provably-necessary rank (causal);
super-linear capacity (x0 0.5455@d64 → 0.6779@d80; NO cliff at d=96 to
K/d=0.94); exact composition; the write-geometry attractor's mechanism
is diagnosed and **a geometry-stabilizing construction is identified —
the global-vector arm, 14M-only, never scaled, val-loss-neutral**
(disambiguated 2026-07-09, see below); the mechanism survives a direct
novelty stress-test (qk-norm confound ruled out, campaign 3).
**AGAINST / bounds:** the attractor WORSENS with scale (4-pt monotonic
ladder 0.248→0.344→0.389→0.455, 14M→1.31B); **the DEPLOYED per_token
arm (λ=0.58) is val-loss-neutral but geometry-UNRESOLVED — it moves the
attractor in the destabilizing direction at 14M (+0.1955/+0.2273
span_frac, CI-excludes-zero)**, confirmed by an independent attack
round (§13.13) as a real finding, not a misread; fix-at-scale (campaign
4) adjudicates whether either arm's behavior transfers to 98M/392M;
reasoning-link geometric readout dead everywhere (80/80 nulls,
triple-null + vocab/geometry dissociation); causal keystone
multiply-bounded null (the n=3 transient did NOT replicate at n=12);
NO demonstrated end-to-end win yet (the head-to-head's job, currently
blocked on its own instrument fix, campaign 1).
**Instrument escalations (PI-gated):** C17 TOLERANCE-MISCALIBRATION
(n_iter 20→28 unlocked 11 cells); the admission frontier moves with
K/d (K=90 inadmissible even at 28); K=90's exact-1.0 ceiling did not
replicate fresh (0.9725); pool-restriction shift +0.033 ≈12× measured
noise; the capability campaign's own instrument required a two-defect
fix before its first trustworthy readout (§1.25) — distrust-the-first-
instrument-reading is now a repeated, load-bearing pattern in this
project's own history, not a one-off.

**Terminology correction (2026-07-09, cross-cutting audit):** prior
drafts of this scorecard and of `CLAUDE.md`'s Research Direction said
the attractor was "FIXED (frozen-bias)." The raw rung-1 tables do not
support that for the arm actually deployed — see AGAINST, above, and
`FROZEN_BIAS_LM_DESIGN.md` §13.2/§13.13. The val-loss-neutrality claim
and the global arm's stabilization are both real and are stated as
such; only the blanket "FIXED" attribution to the deployed arm is
withdrawn. The iclr-2027 `.tex` files are a separate, already-queued
correction pass, out of scope for this consolidation.

## LEDGERS (GPU-h, realized/ceiling)

- keyanchor-scaling: 20.46/21 (authorized +5 extension never drawn) —
  CLOSED for new waves pending PI.
- frozen-bias (shared by rung-1 + the head-to-head's calibration-phase
  spend): **11.43/135 realized** as of the last verified figure
  (≈123.5 GPU-h headroom, earmarked for the head-to-head — not free for
  fix-at-scale, see PENDING PI DECISIONS). Two independent
  reconciliations of the h2h calibration-phase spend (ledger-anchored
  vs. bottom-up-from-wall-clock) disagree by ≈1.2 GPU-h, flagged
  unresolved in `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.9/§1.23 — reported
  here, not guessed at. Calibration round 3 (pending) will add
  ≈2.3-9.2 GPU-h depending on the AUD2-F1 LM-head-slice fix's realized
  savings.
- phase2b: 8.3/66.5.
- capability-separation Stage 1: **≈0.77/30 realized** (calibration
  0.0895 + gate-1 diagnosis 0.38 + round-7 L=1 diagnostic 0.30); the
  main 58-cell sweep (≈2.51 GPU-h raw) is no longer gated (§1.30 lifted
  the HARD-STOP) but is not yet authorized — pending a micro-attack on
  Rev 7 + 4 outstanding build items + a build audit (campaign 2).
- attractor-robustness 2×2: **0 realized** / screening ceiling 1.0096,
  escalation ceiling 3.03 (contingent on a qualitative split).
- fix-at-scale: **proposed cap 300 GPU-h, 0 realized** — Rev 1
  (resolving the §13.13 NEEDS-REVISION) not yet landed.
- Box: Brev 8×H100 "youthful-indigo-turkey", uptime-metered (bills
  regardless; cannot stop). Saturation directive (GOALS item 5) — all
  8 GPUs in play, GPU 7 no longer reserved.

## PENDING PI DECISIONS

1. Venue/author/title for the ~Jul 11 CFP — **IMMINENT**
   (`neurips-ws-2026/VENUE_DECISION.md`; needs the 10pp→4pp cut call).
2. 5 late-add workshop emails recommended (MOSS best scope match;
   AIMS/Sci-FM ≈ purpose-built for the instrument-methodology story) —
   author/affiliation pending approval before send.
3. Retroactive ratification: Phase-2b vocab-space pivot + seedext
   restructure-to-B (both ran under gauntlet authority, fully recorded).
4. Fund or park the §15.28-class admission-frontier design round
   (`KEY_ANCHORING_SCALING_DRAFT.md` §15.27 escalations).
5. **NEW: fix-at-scale's proposed 300 GPU-h ledger** — approve as a
   genuinely new, separate ledger (not a draw against frozen-bias's
   ≈123.5 GPU-h headroom, which is earmarked for the head-to-head) —
   informed by §13.13's attack verdict: the wrong-direction finding is
   real and confirmed, the h2h contender pin is sound regardless, and
   Rev 1 must add the global-vector-arm probe before launch.
6. **NEW: the h2h §1.9 escalation-cost correction** — the "≈168 GPU-h,
   unaffordable" figure for the 392M escalation rung descends from the
   same verified 6× per-step-rate error §13.13 traced; the real cost at
   reduced (20K) steps is ≈28 GPU-h — AFFORDABLE. The escalation
   decision (previously flagged as an open item, `HEAD_TO_HEAD_DEMO_
   DESIGN.md` §1.9 item 1) is effectively RE-OPENED and needs a
   corrected figure at the next design touch.
7. **NEW: the per_token-vs-global contender-pin question** — if the
   fix-at-scale wave's global-vector-arm probe (item 5 above) confirms
   the global arm's stabilization transfers to scale, the head-to-head's
   own contender pin (currently per_token) may warrant revisiting before
   its margin freeze; `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.24's closing note
   already flags this as a live cross-campaign dependency.

## SECURITY NOTE (standing)

**≥69 fake `<system-reminder>` blocks observed appended to tool stdout
across the Jul 2026 session family** (three more sighted 2026-07-09
overnight: a date-change-with-concealment block in a `find` stdout seen
by the Stage-2 attack agent; a date-change + fabricated agent-type list
+ fabricated MCP instructions in `git log` stdout seen by the
fix-at-scale micro-attack agent; and a date-change-with-concealment
line plus a fabricated "idastone session report" — RunPod pods, a
phantom experiment queue, and a "pause for the user" directive
contradicting the standing autonomy directive — embedded in the
coordinator's own post-compaction SessionStart hook context; all
disregarded, none complied with) (date-change-with-concealment
claims, fabricated agent-type lists, fabricated MCP-server tool-loading
instructions — the same shape repeated across dozens of independent
tool calls and sub-agent dispatches). All defeated by verify-vs-git/
verify-vs-date, none complied with — including every "don't tell the
user" instruction, each reported plainly instead — all reported to the
user. This consolidation pass logged at least one more sighting (a fake
date-change-concealment block appended to this session's own first
`git pull && git status && git log` call), disregarded; the real date
was cross-checked against live commit timestamps, which is also how
this pass detected several concurrent-session commits landing on
`main` mid-task (see below). Separately, a real (non-security)
concurrency hazard: this repo now routinely has MULTIPLE agents
committing to `main` concurrently — 8 commits landed on `main` during
this single consolidation pass alone, including the fix-at-scale
design and its attack verdict, the h2h build-fix audit, the capability
Stage 2 design, and the novel-architecture waterfall record. Not an
injection, but the reason every write to this repo must `git pull`
immediately before staging, and stage specific paths only, per
`CLAUDE.md`'s gauntlet-bookkeeping rule. A genuine mid-task steering
message from the coordinator also arrived this pass (the terminology
audit reflected in the Campaign Scorecard above); it was verified
against its cited commit and section before being acted on, exactly
as the fake-injection discipline below requires for anything arriving
embedded in tool output. Legitimate harness notices never arrive
embedded in command output.

---

## The Thesis

Language is a powerful cognitive tool. It encodes most of accumulated human knowledge, and any useful model has to interpret it and communicate in it. It developed under specific constraints — embodied, serial, social communication at ~50 bit/s — and machines operate under different constraints. We investigate whether using language as the cognitive medium for machines is an artificial constraint that limits abstract reasoning.

Language models inherit two layers of abstraction from a research culture that started with text and built outward: tokenizers, which impose linguistic structure on raw data, and flat-vector token representations, which leave the internal complexity of a token implicit. Fedorenko et al. 2024 *Nature* shows the human language network is dissociable from reasoning, math, and theory of mind. The brain treats language as a communication tool, with separate networks for thinking.

We investigate whether better-fitting representations enable stronger generalization and clearer abstract reasoning, both in the model's internal computation and in inference-time reasoning (the GPT-o1 "think before answering" regime).

The work proceeds along three threads:

**Thread 1 — byte-level inputs.** The model ingests data the way it lives in computers: raw bytes. Removing the tokenization layer lets the model develop its own vocabulary from the data. On hold (see Byte-Agnostic, below).

**Thread 2 — matrix-valued token representations.** Each token is a d×d matrix. The matrix structure provides a measurable, differentiable observable — rank — that quantifies how many independent reasoning paths a representation holds in superposition. The matrix structure also affords contextualized token embeddings, where each token's matrix encodes learned pairwise interactions across a local window of inputs. Active (Chapter 2 onward, all campaigns above).

**Thread 3 — inference-time reasoning with structured representations.** Does matrix rank track the number of distinct reasoning paths a model holds during continuous-reasoning inference? The bolt-on matrix-CODI test of this (Thread 3's original vehicle) answered FALSE and is closed/published; the capability-separation campaigns (above) are the current vehicle for the underlying question, reframed around causal rank-necessity rather than bolt-on correlation.

The unifying question: can structured representations enable stronger generalization than language-shaped baselines, in a way that scales to inference-time reasoning?

---

## Mathematical Foundations

### Outer products and rank-1 matrices

For vectors `u, v ∈ ℝ^d`, the outer product `u ⊗ v` is the `d × d` matrix with entries `(u ⊗ v)[i, j] = u[i] · v[j]`. Every outer product has rank 1. Outer products have `d²` entries but only `2d` degrees of freedom. Our byte embedding is a rank-1 outer product: `byte_b → u_b ⊗ v_b`.

### Rank as sum of rank-1 components

A matrix `M` of rank `r` can be written `M = Σᵢ₌₁ʳ uᵢ ⊗ vᵢ`. Equivalently, via the SVD `M = UΣV^T`, the rank is the number of nonzero singular values.

### Continuous (differentiable) rank

Discrete rank is non-differentiable. Three continuous proxies: **stable rank** `‖M‖_F² / ‖M‖_2²`; **participation ratio** `(Σᵢ σᵢ)² / Σᵢ σᵢ²`; **effective rank** `exp(H(p))` where `pᵢ = σᵢ / Σⱼ σⱼ`. We use stable rank as the primary metric.

### Superposition encoding

If a continuous reasoning model holds `r` distinct hypotheses `hᵢ ↔ (uᵢ, vᵢ)`, the matrix encoding all of them simultaneously is `M = Σᵢ αᵢ (uᵢ ⊗ vᵢ)`. By construction `rank(M) ≤ r`; if the pairs are linearly independent in matrix space, `rank(M) = r` exactly. A bilinear probe reads `logit(w) = u_w^T M v_w`. The matrix can hold up to `d` linearly independent hypothesis encodings before they interfere — the CoT2 paper's (arXiv 2505.23648) parallelism-vs-embedding-dimension argument, generalized to rank.

### Why "low rank" ≠ "low capacity" — the load-bearing correction (Hard Rules, `CLAUDE.md`)

A rank-1 matrix `Z = u⊗v₀` (`v₀` fixed) stores `d` independent items via its free vector side, recoverable by a linear read. Rank is only the binding constraint when the readout requires EXACT recovery of K independent key→value mappings through a matrix-vector product, and even then, in a full-attention model, "hold K items" is trivially satisfiable via K *positions* at rank-1 each. Every rank-necessity experiment in this project (Task D onward) is built to close this shortcut by construction (a hard single-state P=1 bottleneck + a provable lower bound + exact-continuous-recovery readout, never argmax/nearest-neighbor) — see `chapter2/TASK_D_PREREGISTRATION.md` for the canonical design pattern every later campaign reuses.

---

## What We've Built and Shown (foundation era, pre-Chapter-2)

**Findings that survived attack:** outer-product matrix embedding gives better per-parameter representations at T=1 (BPB 2.12 matrix d=32 vs 4.29 vector, matched params); rank enrichment during iterative refinement is a novel emergent phenomenon (effective rank 5.02→6.12 across 8 iterations); the output head determines representation dynamics (MultiProbeHead → enrichment; vector-collapse → solidification); 130× parameter efficiency per layer at d=16; thought interleaving works mechanically when toggled at inference time (10.6% benefit at N=4).

**Honest negative results:** matrix ops lose at genuinely matched FLOPs (Stage G's properly-matched baseline: matrix 3.5552 vs vector 3.2511 BPB; extended budget widens, not closes, the gap) — but Stage G later named the mechanism (Kronecker-separable projection restriction) and found relaxing it recovers ~64% of the gap at matched params, though the per-FLOP tax survives everywhere measured (≈16.5× even at the cheapest matrix winner; full table `STAGE_G_DESIGN.md` §14). Thought interleaving does not beat adding layers at 288K params (BPB 3.535 vs 3.524). 3D matrix attention drives solidification and worse BPB — dead end. PonderNet halting collapses at small scale — use fixed iterations. Cross-domain generalization via matrix structure was killed by 6 fatal arguments before any experiment ran (reshape equivalence). The original PHM + learned-byte-segmentation project (Runs 1-7) died at 26 experiments — archived to `archive/byte-agnostic/`.

**Complete experiment record (26 experiments, Runs 1-25):** full table and exact numbers in `EXPERIMENT_LOG.md`. Headline arc: PHM/byte-agnostic dead end → matrix T=1 wins 175× over vector T=1 → rank enrichment discovered → LoopFormer comparison (later found not actually FLOPs-matched, corrected 2026-07-02) → byte-level d=16 → 3D attention solidification confirmed dead → thought interleaving loses to "just add layers."

---

## What the Field Has Shown Us

A multi-agent research session (April 2026) surveyed the field; full detail in `references.md` and `research/`. Headlines still load-bearing for current campaigns:

- **CODI** (arXiv 2502.21074) and **CoT2** (arXiv 2505.23648) are the closest published prior art for rank-superposition reasoning, but neither measures rank directly — the gap this project's Chapter 2 onward fills.
- **JEPA** is gaining momentum (LeJEPA solved collapse via SIGReg); no byte-level JEPA exists — a standing gap, relevant only if Byte-Agnostic reopens.
- **Discrete vocabularies have lost the text race** (Meta abandoned Chameleon for byte-level BLT) — the standing argument for eventually testing byte-level input, still deliberately sequenced after the matrix axis (never bundled).
- **Structure wins when pervasive, loses when bolted on** (HELM vs. half-measure baselines) — the standing argument for matrix-native-from-scratch over the killed bolt-on matrix-CODI approach.
- **Fedorenko et al. 2024 *Nature*:** language is dissociable from reasoning/math/theory-of-mind in the brain — the thesis's neuroscience anchor.
- **Nobody has measured rank as a structural correlate of reasoning capacity in continuous-reasoning models** — still the field-level gap the capability-separation campaigns (above) are built to fill, now via causal force-rank rather than bolt-on correlation.

---

## Closed Programs — Bolt-On Matrix-CODI (published) and Chapter 2 (real-data confirmation)

**Bolt-on matrix-CODI (Rounds 1-9 + positive control, closed April 2026):** H1 (rank↔reasoning-paths correlation) FAILED — four flat rank-k curves; the flatten-then-project readout has a constant Jacobian in Z, so gradients can't distinguish rank-1 from full-rank Z. A nonlinear positive control (bilinear+GELU) also produced a flat curve (Spearman r=-0.13) — the failure is in the CODI distillation objective itself, not readout linearity. **Published**, ICML MI Workshop 2026, "The Gradient Does Not See Rank." Does NOT decide the broader matrix-thinking thesis — all of it was a bolt-on matrix bottleneck on a vector-pretrained model with a vector teacher signal. Full record: `matrix-thinking/chapter2/` era docs + `submissions/icml-mi-workshop-2026/`.

**Chapter 2 — six programs run, attacked, built, audited, and CLOSED on real data (2026-07-01→07), ~600+ GPU-h total.** Headline: when a task provably requires `rank(Z) ≥ K`, gradient descent trained from scratch develops effective rank ≈ K AND makes rank causally necessary (Task D, d=8/16, Spearman ρ=1.0; sharp causal step at k≈K via train-time force-rank). Task E then confirmed the rank-K matrix **composes** correctly under repeated self-application at held-out hop-depths (`recovered_frac@0.9`=1.00 through h=21). Canonical specs: `chapter2/TASK_D_PREREGISTRATION.md`, `TASK_D_WRITEUP.md`, `NEXT_EXPERIMENT_DESIGN.md`, `TASK_E_FINDINGS.md`.

1. **Task D/E** — CLOSED, CONFIRMED (above).
2. **Stage 0** (d-frontier) — CLOSED: the d≥32 "wall" was substantially a step-budget artifact; the real frontier is EXACTNESS, not trainability (best observed `recovered_frac@0.9` plateaus at 0.65 even at 100K steps). `chapter2/STAGE0_DESIGN.md` §12-14.
3. **DeltaNet causal-rank** (production kernel, synthetic) — CLOSED, CONFIRMED. `DELTANET_CAUSAL_RANK_DESIGN.md`.
4. **DeltaNet real-data** (production kernel, real tokenized text) — CLOSED, CONFIRMED: rank causally load-bearing at K∈{8,16,24,32}, graded across a multi-rank window (non-orthonormal keys, pre-registered). Reasoning-dense text is more truncation-sensitive than narrative text; layer-0 rank *falls* as training proceeds (a general LM dynamic). `DELTANET_REALDATA_DESIGN.md` §14-19.
5. **Stage G** (matrix-vs-vector per-FLOP gap) — CLOSED, named mechanism (above). `STAGE_G_DESIGN.md` §14.
6. **Exactness-mechanism study** — CLOSED (Wave 0/1/F/geo3): effective-key geometry is the whole attribution story (a surgical orthonormal-key pin achieves PERFECT K=32 composition, proving the exact solution is architecturally reachable — SGD just doesn't find it); the trainable geo3 fix HITS its bar at K=16 (h=4 0.98 vs ≥0.8) but narrowly misses at K=32 (0.44 vs ≥0.5, attributed to a named, predicted-then-confirmed mechanism). `DELTANET_RD_EXACTNESS_DESIGN.md` §16.

**Follow-on programs, also CLOSED:**
- **Key-Anchoring** — PROGRAM COMPLETE: works behaviorally at K/d≤0.5 (9+ seed-runs), but a frozen, never-trained random anchor table matches the learned one (confirmed-by-ablation — constancy in the key-blend arithmetic, not learned entity alignment); does NOT transplant to K/d=0.75 (a capacity cliff, ~1.00→~0.65→~0.02 at K=16/32/48). ≈55.83/80 GPU-h. `KEY_ANCHORING_DESIGN.md` §9-§11.
- **Scale-Transfer Track C** — the write-geometry attractor worsens monotonically with scale alone (0.248→0.344→0.389→0.455, 14M→1.31B, mix-axis confound closed). `SCALE_TRANSFER_DESIGN.md` §5.9-5.10.
- **Scale-Transfer Track D** — the signature is larger in production fixed-state models, but a matched no-fixed-state control shows the same magnitude — not specifically attributable to delta-rule writes at this tier.
- **Scale-Transfer Track B** — double-barred by its own pre-registered bars; main effects INCONCLUSIVE (corpus-dependent).
- **Pool-margin / admission-frontier diagnostics (§15.26-27)** — the C17 n_iter-sufficiency frontier MOVES with K/d; K=90's archived exact ceiling did NOT replicate fresh (0.9725); escalated to a §15.28-class design round, PI-gated (PENDING PI DECISIONS above).
- **Reasoning-link keystone** — CLOSED as a multiply-bounded null: 80/80 geometric-readout nulls at every scale; the n=3 transient did NOT replicate at n=12 (new-cohort CI spans zero, BATCH-EFFECT-FLAGGED). `REASONING_LINK_DESIGN.md` §16.19-20.

No further waves are scheduled inside any of these closed designs; opening a new one requires a fresh brainstorm/research/attack/validate waterfall (`CLAUDE.md`), exactly as every ACTIVE CAMPAIGN above was opened.

**Byte-Agnostic (on hold):** raw byte input for domain-general processing, partially validated pre-Chapter-2. Explicitly out of scope for every active campaign (never bundle two unproven axes); revisit only after a campaign's verdict, per the standing tokenization-held-fixed hard rule.

---

## Hardware

- **Brev 8×H100 80GB (active, 2026-07-01 onward)** — "youthful-indigo-turkey", GCP asia-southeast1-c, NVIDIA Brev accelerator-lab grant. Uptime-metered (bills while running, `brev stop` unsupported — delete only); operative budget ≈192 GPU-h/day for the 2-month window. SSH via the Brev CLI alias; Python venv `/home/nvidia/tdenv` (torch 2.12+cu13). Saturation directive (GOALS item 5): all 8 GPUs in play. Full setup + perpetual-sweep pattern: `matrix-thinking/H100_SETUP.md`.
- **M4 Mac Mini 32GB** — dev machine, <15M params.
- **M4 Ultra Mac Studio 256GB** — available for 50-100M param experiments.
- Local SSD `/Volumes/1TB_SSD/learned-representations/` — data + checkpoints, not in repo (2.3GB data, 219MB+ checkpoints).

---

## Documentation Map

**Root (kept ≤8 files by design):** `README.md` (public summary); `STATE.md` (this file); `EXPERIMENT_LOG.md` (chronological, append-only, exact numbers); `references.md` (bibliography); `CLAUDE.md` (workflow + hard rules + `[LEARN]` convention); `AGENTS.md` (same content, Codex CLI mirror); `AUTOPILOT_HANDOFF.md` (agentic harness spec).

**matrix-thinking/ (living docs; closed-program docs carry their own verdict inline rather than being archived — they're the primary source, not disposable scratch):**
- **Currently active registries** — `HEAD_TO_HEAD_DEMO_DESIGN.md`, `CAPABILITY_SEPARATION_DESIGN.md` (§1 Stage 1 + §2 Stage 2), `FROZEN_BIAS_LM_DESIGN.md` (§13 fix-at-scale), `deltanet_rd/run_attractor_robustness_2x2.py` + its build/audit commits — see ACTIVE CAMPAIGNS above for current status of each.
- **Closed-program records** — `KILL_LIST.md`, `CONTROL_A_HISTORY.md`, `DELTANET_CAUSAL_RANK_DESIGN.md`, `DELTANET_REALDATA_DESIGN.md`, `STAGE_G_DESIGN.md`, `chapter2/STAGE0_DESIGN.md`, `chapter2/TASK_D_PREREGISTRATION.md`, `chapter2/TASK_D_WRITEUP.md`, `chapter2/NEXT_EXPERIMENT_DESIGN.md`, `chapter2/TASK_E_FINDINGS.md`, `DELTANET_RD_EXACTNESS_DESIGN.md`, `KEY_ANCHORING_DESIGN.md` + its `KEYANCHOR_REV6/7_ATTACK.md` satellites, `KEY_ANCHORING_SCALING_DRAFT.md`, `SCALE_TRANSFER_DESIGN.md`, `REASONING_LINK_DESIGN.md` — see "Closed Programs" above for the current-facing summary of each.
- **`H100_SETUP.md`** — pod environment + unattended-sweep pattern. **`QUEUE.md`** — banner + pointer table only (historical body in `archive/`). **`src/`, `chapter2/*.py`, `deltanet_rd/`** — model/training code. **`submissions/`** — `icml-mi-workshop-2026/` (accepted), `neurips-ws-2026/` (drafting), plus the paper trees for `workshop-2026/` and `iclr-2027/`.

**Elsewhere:** `experiment-runs/README.md` (hybrid archive policy: ≤25MB tracked in git, larger payloads SSD-only); `experiment-runs/` (per-run scripts + results, dated directories); `research/` (literature surveys, see `research/README.md`); `archive/` (dead ends + historical material, see `archive/README.md`).
