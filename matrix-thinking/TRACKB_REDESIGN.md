# TRACKB_REDESIGN — Hard-Selectivity geo3-in-LM: Forcing β's Missing Precondition Before Retesting Orthogonalization on Free Text

> **Rev 1 — 2026-07-04. Design only, no GPU, no box access.** Not yet attack-reviewed —
> per `CLAUDE.md`'s waterfall and this project's own standing "independent attack round
> before build" rule, this document is not build-ready until an adversarial pass runs
> against it (§11 lists the open questions that pass should target first). This document
> **is** `SCALE_TRANSFER_DESIGN.md` §4.2's own registered conditional follow-on —
> quoted verbatim there: *"a hard-zero-beta-at-non-selected-positions variant... is noted
> as the conditional follow-on, **requiring its own attack pass before any build**, since
> it reintroduces exactly the kind of hard-masking machinery LM mode was deliberately
> built without."* Nothing in `SCALE_TRANSFER_DESIGN.md` is modified by this document —
> Track B's own §4 stands as written, its Wave −1 gate verdict (`no_launch_redesign`)
> stands as measured, and this document is the redesign that verdict itself routes to.

---

## 0. Reading list this design builds on (context, not repeated here)

- `matrix-thinking/SCALE_TRANSFER_DESIGN.md` §4 — Track B's full geo3-in-LM design: the
  three failure modes free text poses for F-geo-3 (§4.2), the β-gated top-K construction,
  the two registered Wave −1 gate criteria and their routing table (§4.2, outcome (iii)),
  the insertion point (§4.3), required instrumentation (§4.4), cells/budget (§4.5),
  success/failure criteria (§4.6), claim tier (§4.7), attack-yourself (§4.8).
- `EXPERIMENT_LOG.md`, **"SCALE-TRANSFER Track B (2026-07-04): geo3-in-LM built +
  smoke-clean; Wave −1 gate measured on all 6 archived Wave C checkpoints — HARD
  NO-LAUNCH"** — the actual gate measurement this redesign responds to, plus the
  independent audit round (1 MAJOR + 4 MINOR, all fixed) and the two `[LEARN]` blocks
  (the gate's own algebraic-complement degeneracy; the forced-failure-test-precondition
  lesson) both of which this redesign inherits and must not repeat.
- `matrix-thinking/deltanet_rd/results/lm_rd_geo3/wave_neg1_gate.json` — the raw gate
  measurement this document's §1 restates exactly (read directly this session, not
  taken from prose).
- `matrix-thinking/DELTANET_RD_EXACTNESS_DESIGN.md` §14 (F-geo-3's mechanism: why
  Newton–Schulz, the insertion spec, the stability plan, the i-strong-vs-F-geo-3
  distinction), §16 (F-geo-3 results: K=16 bar HIT, K=32 narrowly missed, outcome-F
  attribution — cross-episode drift, not incomplete orthogonalization, is the residual
  bottleneck), **§16.7 including its dated correction** (a shared-scalar simulator input
  bug that silently mis-attributed a K=32 prediction to the wrong K's measured drift —
  the general lesson, restated in §8 item 5 below, is load-bearing for this redesign if
  it ever builds an analogous predictive gate).
- `matrix-thinking/KEY_ANCHORING_DESIGN.md` Rev 5 §1–§2.2 — the 2×2 (orthogonal ×
  stable) mechanism picture this redesign's own scientific readout (§5) mirrors
  directly: neither ingredient alone matters much, orthogonality alone buys most of the
  gap, but the two **together** produce a discontinuous jump to saturation, not a partial
  additive improvement. §3.6/§8 (R4-1) — the override-stamping precedent this redesign
  reuses (§6) for running a known-failing-premise construction as a deliberate reference
  arm.
- `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` — read in full for this design:
  `DeltaNetLMMixer.__init__`/`forward` (`:419–570`), `_geo3_lm_select_and_orthogonalize`
  (`:206–398`, the existing beta-gated/naive-window gather→orthogonalize→scatter
  construction this redesign's candidates 1/2/4 extend), the `num_heads==1`-only scope
  guard on `geo3_active` (`:474–477`).
- `matrix-thinking/deltanet_rd/run_lm_rd_geo3_sweep.py` — `selection_mode_for_verdict`
  (`:165–169`) hard-raises given `verdict="no_launch_redesign"`; this is a real,
  already-built refusal this redesign's own reference-arm cell (§5) must explicitly
  override, not bypass silently.
- `STATE.md` "Hardware" + `SCALE_TRANSFER_DESIGN.md` §5.6 Rev 2.1/2.2 amendments — the
  program-wide 300 GPU-h ceiling and its current consumption, restated with the
  arithmetic in §7 below.

---

## 1. Why this redesign exists — the gate finding, restated exactly

Track B's Wave −1 gate (`SCALE_TRANSFER_DESIGN.md` §4.2) measured, on all 6 archived Wave
C checkpoints (openr1×3 seeds, wikitext×3 seeds, step 6103, `n_params=14,048,896`, pooled
over 12,288 chunk-episodes, `chunk_size=64`), whether the LM's own learned β nominates a
small write-worthy subset per chunk. It does not:

| Criterion | K_sel=16 | K_sel=32 | Bar | Verdict |
|---|---|---|---|---|
| (a) top-K_sel β-mass fraction | 0.309 | **0.569** | ≥0.60 | FAIL (close miss at K=32) |
| Gini (chunk-level, K_sel-independent by construction) | 0.099 | 0.099 | — (shape check) | near-uniform |
| (b) mean β at non-selected positions | 0.387 | **0.363** | ≤0.25 | FAIL |
| (b) non-selected write-mass fraction | 0.691 | **0.431** | ≤0.40 | FAIL |

(`wave_neg1_gate.json`'s own `pooled` block, read directly this session — not a
recomputation.) A concrete way to see how close to uniform this is: at K_sel=16 (25% of
a 64-position chunk), a perfectly uniform β would capture exactly 25% of chunk mass in
the top 16 positions; the measured capture is **30.9%** — barely above uniform. At
K_sel=32 (uniform prior 50%), measured capture is **56.9%** — again barely above
uniform. Gini≈0.099 (0 = perfectly uniform, 1 = all mass on one position) is the same
number under both K_sel because it summarizes the whole-chunk β distribution shape, not
a K_sel-dependent statistic — it is the single cleanest number for "how far from uniform
is this gate," and the answer is: not very far.

**Verdict recorded:** `gate_verdict: "no_launch_redesign"` — per §4.2's own registered
routing, criterion (b) failing is a **hard no-launch regardless of (a)**. Track B's Wave
1 (β-gated or naive-window) does not launch on the as-designed construction. The
mechanism premise F-geo-3 needs — a small top-K subset captures the bulk of write-mass,
so orthogonalizing it controls state geometry — does not hold on this harness's actual
trained β distribution: 43.1% of chunk write-mass sits at non-selected positions,
writing non-orthogonally into the same shared state, at the very K_sel the primary
construction was designed around.

**A build-time subtlety this redesign inherits and must not re-trip.** The gate script's
own literal implementation makes criteria (a) and (b)'s write-mass sub-check exact
algebraic complements over the same chunk-total denominator (`0.60+0.40=1.00`) —
`EXPERIMENT_LOG.md`'s own `[LEARN]` block on this finding notes the "(a) fails, (b)
passes → naive-window primary" branch is consequently unreachable for *any* real β
distribution, not just empirically rare. §4 below inherits this exact-complement
property for the masking-based candidates and shows it becomes *more* degenerate there,
not less — see §4.1.

---

## 2. Design principles governing every candidate below

Two facts jointly constrain what "properly designed" means here, both inherited directly
rather than re-derived:

1. **Orthogonality alone is not the whole story — KEY_ANCHORING's own 2×2 already showed
   this on the synthetic harness.** `KEY_ANCHORING_DESIGN.md` §2.0's four-cell table
   (stable×orthogonal / stable×not / not-stable×orthogonal / not-stable×not, all K=32,
   `d_state=64`, real archived runs) found: neither ingredient alone matters much
   (stable-only ≈ baseline, ~0.005–0.010 at h=4); orthogonality alone buys most of the
   gap (geo3's 44–56× jump, `DELTANET_RD_EXACTNESS_DESIGN.md` §16); **both together is
   not a partial improvement, it is a discontinuous jump to saturation** (0.44→1.0000).
   Track B's failed gate establishes that the LM harness currently lacks even the
   *orthogonality* precondition's own target (a defined write-set to orthogonalize) — but
   the KEY_ANCHORING lesson says that fixing selectivity alone, without geo3, should
   **not** be expected to move much either. §5's 2×2 factorial is designed specifically
   to check whether this same interaction shape reproduces on free-running LM text, not
   just to confirm selectivity can be forced.
2. **The bolt-on lesson (`SCALE_TRANSFER_DESIGN.md` §6.4) applies here with equal force,
   not just to Track D's pretrained-model graft.** Retrofitting a hard/sparse mechanism
   onto Wave C's *already-trained*, continuous-β checkpoint via continued fine-tuning
   risks the exact failure shape that killed the original matrix-CODI program: downstream
   layers co-adapted to the *old* representation's statistics and the bolt-on could not
   out-compete that co-adaptation. Given Wave C's own architecture trains in **≈0.077
   GPU-h/run** (6,103 steps in 274–278s, `SCALE_TRANSFER_DESIGN.md` §5.6's own measured
   throughput), there is essentially no GPU-cost reason to accept the bolt-on risk to
   save wall-clock. **Default for every candidate below: train from random init at Wave
   C's own architecture/corpus/step-budget, never continue-train from an existing Wave C
   checkpoint.** A continued-training retrofit ablation is named as a possible, explicitly
   NOT-default follow-on in §9, not silently assumed superior for being cheaper.
3. **Scope stays at `num_heads=1`, mirroring geo3_active's own registered scope.**
   `lm_pretrain_rd.py:474–477` hard-refuses `geo3_active` with `num_heads>1` because no
   registered cell exercises the H-folding math at scope. None of the candidates below
   are exercised at `num_heads>1` either — the same discipline applies, for the same
   reason (untested-at-scope, not proven-broken).

---

## 3. Candidate mechanisms

Four candidates, spanning the full range the task asked for: a minimal learned-hard-mask
change (3.1), a differentiable adaptively-sparse alternative (3.2), an architecturally
fixed (non-learned) write schedule that is a genuinely different model (3.3), and a
soft-then-hard annealed compromise this project's own history already predicts will
underperform in its soft half (3.4).

### 3.1 Candidate 1 — Hard top-K β masking with straight-through gradients (PRIMARY)

**Mechanism.** Forward: compute `beta_soft = sigmoid(b_proj(x))` exactly as today
(`lm_pretrain_rd.py:529`); within each `chunk_size=64` window, select the top-`K_sel`
positions **by β magnitude** — literally the same `priority`/`topk_idx` computation
`_geo3_lm_select_and_orthogonalize`'s `beta_topk` branch already implements
(`:292–301`), just reused one call earlier — and construct a binary mask `m∈{0,1}^T`
with exactly `K_sel` ones per chunk (content positions only, EOT/padding hard-excluded
exactly as §4.2 item 3 already specifies). Output: `beta_hard = beta_soft * m` — literal
zero write mass at every non-selected position, satisfying the queued idea's own name
exactly. Backward: a straight-through estimator (Bengio, Léonard & Courville 2013;
the same trick underlying hard-concrete/L0 gating, Louizos, Welling & Kingma 2018) —
`dL/d(beta_soft) := dL/d(beta_hard)`, i.e. gradient flows through the mask as identity,
so `b_proj`'s weights keep receiving a training signal even for positions the current
forward pass zeroed.

**Insertion site.** `DeltaNetLMMixer.forward`, `lm_pretrain_rd.py`, immediately after
`beta = torch.sigmoid(self.b_proj(x))` (current line 529) and before `q,k,v` are
reshaped into heads (line ~531) — the exact site `_geo3_lm_select_and_orthogonalize` is
already called from (per §4.3's own insertion spec). Concretely: a new sibling function
(or an extra return value threaded out of the existing selection logic, since
`topk_idx`/`valid_sel` are already computed there, `:300–301`, and already returned in
`diag`) applies the same top-K selection **to `beta` itself**, which the existing
function currently never touches (it only gathers/orthogonalizes `k`). A new
constructor flag, `hard_select_active: bool` + `hard_select_k_sel: int`, mirrors
`geo3_active`/`geo3_k_sel`'s existing additive-off-by-default pattern exactly.
`hard_select_active` and `geo3_active` are **independently toggleable** — this is the
axis §5's 2×2 varies.

**Cost.** Cheapest of the four. No new orthogonalization, no new solver, no new
autograd primitive (STE's backward is a single elementwise pass-through). Reuses
already-audited selection code verbatim. Estimated build: well under a day. Wave −1
manipulation-check cost: §4.1 below.

**What failure means.** (i) Val loss regresses sharply relative to an unconstrained
control → STE's biased gradient (treating a genuinely discontinuous mask as identity in
the backward) is too crude an approximation for this optimization landscape — a known,
named STE pathology, not a surprise if it occurs. (ii) The selected-position **set**
churns heavily step-to-step (a position flips in/out of top-K as β shifts by noise near
the K_sel-th-ranked boundary) — a distinct, second failure signature from (i), requiring
its own diagnostic (§4.1, §8 item 2), not conflated with ordinary training instability.
(iii) The mask converges to a **content-independent positional artifact** (e.g., always
selects position 0 mod some period, regardless of token identity) — a fake positive on
the gate metrics (Gini rises, non-selected write-mass falls) without buying the
content-selectivity geo3's mechanism actually needs; checked directly by comparing the
selected-set distribution across documents at matched intra-chunk offsets (§8 item 1).

### 3.2 Candidate 2 — Entmax/sparsemax β (adaptively sparse, differentiable exact zeros)

**Mechanism.** A genuinely different differentiable primitive from candidate 1's
hard-then-STE approach: replace the independent, per-position sigmoid with a
**chunk-normalized, adaptively sparse** transform. Compute a raw per-position score
`s = b_proj(x)` exactly as today, but instead of an elementwise sigmoid, apply
**sparsemax** (Martins & Astudillo, ICML 2016 — Euclidean projection onto the simplex,
closed-form, exact zeros for below-threshold entries) or **entmax** (Peters, Niculae &
Martins, ACL 2019 — a tunable-sparsity generalization via bisection) **over the
`chunk_size` axis, per head, per chunk** — i.e., β becomes a distribution over the
chunk's 64 positions that sums to (at most) 1, with exact, input-dependent zeros rather
than a fixed, hand-chosen K. This is a **structural change to the model**, not merely an
added selection step on top of an otherwise-identical architecture: today's independent
sigmoids let a chunk write anywhere from near-zero to near-`T` total β-mass; a
chunk-normalized sparsemax/entmax bounds total per-chunk write mass to ≤1 by
construction — a different write-budget regime, flagged explicitly so any downstream
effect is not silently misattributed to "selectivity" when it may instead be "budget."

**Insertion site.** Replace `self.b_proj` + `torch.sigmoid(...)`
(`lm_pretrain_rd.py:494`, `:529`) with the same linear layer producing raw scores,
followed by a **new primitive** — this codebase has no sparsemax/entmax implementation;
sparsemax's simplex projection is a ~10-line sort-and-threshold with a documented,
analytic backward (no external dependency needed, matching this repo's "reuse audited
code, hand-roll simple closed forms" discipline over taking an unaudited PyPI
dependency); entmax's bisection variant is more code for a tunable sparsity knob, ranked
as a stretch, not the primary sub-choice, for exactly that reason.

**Cost.** Moderate. New differentiable primitive requiring its own Wave −1
gradient-finiteness smoke (mirroring geo3's own §14.6 item 2 convention exactly) —
sparsemax's gradient is **zero outside the support** by construction (a documented,
not-a-bug property, Martins & Astudillo 2016 §3), which could starve `b_proj`'s weights
for non-selected positions of *any* gradient at all — a real, qualitatively different
risk from STE's dense (if biased) gradient in candidate 1, and must be checked, not
assumed benign. Also requires verifying `chunk_delta_rule`'s own derivation only assumes
`β≥0` (compatible) and does not implicitly assume independent per-position β
unconstrained by a chunk-sum budget (unverified — a build-time check, not an assumption).

**What failure means.** (i) The chunk-sum-≤1 constraint could shrink total write mass
well below what the baseline's independent sigmoids produce, causing a val-loss
regression attributable to **underwriting**, not to selectivity per se — the total
per-chunk β-mass (pre- vs post-change) must be reported alongside val loss so this
confound is distinguishable, not assumed away (§8 item 3). (ii) The natural sparse
support could still be too broad if raw scores cluster closely — entmax's `α` knob (or a
temperature on the sparsemax input) would need its own escalation, mirroring geo3's own
`n_iter=12→20` escalation precedent (`DELTANET_RD_EXACTNESS_DESIGN.md` §14.4) — one
pre-registered escalation, hard cap, never open-ended tuning.

### 3.3 Candidate 3 — Hand-specified (non-learned) k-hot write schedule, restoring `model_rd.py`'s own audited buffer/mask convention on free text

**Mechanism.** This candidate does not try to make the continuous β gate sparse at all
— it abandons *learned* write-selection and restores the **synthetic harness's own
already-audited, already-proven convention**: `model_rd.py`'s `DeltaNetRDBlock` reserves
a fixed BUFFER token and an architecturally-zero β at all non-write positions
(`lm_pretrain_rd.py`'s own docstring names this exact machinery as "the one deliberate
subtraction from `model_rd.py`'s audited block per this build's brief"). Candidate 3
re-adds it, applied to free-running text via a **preprocessing step, not a model
change**: insert a reserved write/buffer token every `W` real tokens (a fixed period,
e.g. `W=chunk_size/K_sel`), with β architecturally pinned to zero everywhere else and to
a real, learned value only at reserved positions — an actual `item_pos` analog for real
text, constructed by fiat rather than nominated by a learned quantity. This directly
answers the task brief's "trained-from-scratch small LM with k-hot write gating": the
"k-hot"-ness here is **architectural, not learned** — no STE, no sparsemax, no gradient
approximation for the mask itself, since the mask is fixed before training starts.

**Honesty check, stated plainly.** This candidate does **not** escape Track B's original
design problem (§4.2 item 2: "what is the orthogonalization set for free-running
text?") — it answers it by fiat (a periodic schedule) rather than learning it, trading
the original problem for a different, cruder one: a fixed period has no reason to align
with where content-meaningful "write events" actually occur in real text, and this
candidate cannot claim its selected positions are semantically write-worthy the way
candidates 1/2's β-nominated positions at least attempt to be.

**Insertion site.** A new corpus-preprocessing/tokenization adapter (upstream of
`lm_pretrain_rd.py`'s existing data pipeline, `rebuild_lm_corpora_rd.py`-adjacent, not a
`DeltaNetLMMixer.forward` change) that inserts reserved tokens at a fixed period into the
GPT-2-tokenized stream, plus reinstating `model_rd.py`'s own existing
buffer-token/hard-β-mask machinery (already built and audited there) rather than
`lm_pretrain_rd.py`'s plain sigmoid gate. This is architecturally closer to
`model_rd.py`'s block than to `lm_pretrain_rd.py`'s — a genuinely different model
class, not a flag on the existing one.

**Cost.** Lowest *mechanism*-audit risk of the four (the write-mask/buffer machinery
itself is already fully audited on the synthetic harness) but the highest *scope*
change: a new preprocessing pipeline, a new tokenization-adjacent build item, and its
own from-scratch training manifest (no retrofit path exists or is even attempted here —
this candidate cannot be applied to Wave C's existing tokenized data without
re-tokenizing it). GPU cost is trivial regardless (Wave C's own ≈0.077 GPU-h/run scale).

**What failure means.** (i) If a fixed period simply never aligns with anything a
downstream reader would call a meaningful write event, val loss regresses for a reason
that has nothing to do with orthogonality or stability — a **schedule-design** failure,
not a mechanism failure, and should be reported as such (candidate 3 failing does not
bear on candidates 1/2/4's learned-selectivity approach at all). (ii) If it trains fine
and clears the manipulation check trivially (guaranteed by construction, §4.1), the
informative content is entirely in §5's 2×2 factorial, not in the gate.

### 3.4 Candidate 4 — Auxiliary selectivity loss with a hard-mask schedule (RANKED LAST, ablation-only)

**Mechanism.** Keep β architecturally unchanged (plain, continuous, independent
sigmoid) but add an auxiliary loss term during training that directly penalizes
deviation from the gate criteria themselves — e.g. `L_sel = -Gini(β_chunk)` or a direct
penalty on the non-selected write-mass fraction, λ-scheduled from 0 upward (mirroring
this project's own established λ-warm-in convention, cited from `SCALE_TRANSFER_DESIGN.md`
§6.4 item 1 and from F-geo-1's own `L_orth` penalty, `DELTANET_RD_EXACTNESS_DESIGN.md`
§14.0 item 2). Optionally, snap to a literal hard top-K mask (candidate 1's own
mechanism) in a final phase once the soft loss has concentrated β somewhat.

**Why this is ranked last, not merely listed fourth.** This project has **already run**
a structurally identical experiment on the adjacent axis (key-orthogonality rather than
β-selectivity): F-geo-1's soft `L_orth` penalty, at `λ∈{0.1,1.0}`, saturated at a 3–8%
cleanup of key-Gram deviation — nowhere near i-strong's exact 0.000 ceiling, and
`λ=0.1` vs `1.0` were nearly identical (`DELTANET_RD_EXACTNESS_DESIGN.md` §14.0 item 2).
`KEY_ANCHORING_DESIGN.md` §2.4 independently ranks its own directly analogous soft
regularizer candidate ("Explicit cross-episode drift regularizer `L_anchor`") **"soft
tier, ranked LAST, ablation-only"** for exactly this reason. Candidate 4's soft phase is
the same strategy shape applied to a different quantity (β-concentration instead of
key-orthonormality) with no house precedent suggesting it would behave differently.
Only the hard-snap final phase — mechanically identical to candidate 1 — has a real
chance of reaching a literal-zero bar; the soft phase alone is a registered, before-any-
data prediction of underperformance, not an open question.

**Insertion site.** No architecture change to `DeltaNetLMMixer.forward`; a new loss
term in the training loop reading `beta` (already exposed via the existing
`geo3_last_diag`-style side channel convention, `lm_pretrain_rd.py:489`) alongside the
existing next-token cross-entropy. The optional hard-snap phase reuses candidate 1's
code verbatim.

**Cost.** Cheapest to build incrementally (a loss-term addition, no new inference-time
architecture) but the most expensive to *calibrate* — λ-schedule search is the same
open-ended problem F-geo-1's own λ calibration already explored and found saturating,
unlike candidates 1/2/3 which are architecturally deterministic once `K_sel`/period is
fixed.

**What failure means.** If the soft phase alone, measured against the same gate
criteria, still fails — unsurprising, and should be reported as an independent
confirmation of the F-geo-1 finding at a new target (β-selectivity rather than
key-orthogonality), not as a novel negative result. The only open question this
candidate can answer that candidate 1 alone cannot is whether soft warm-up makes the
*subsequent* hard phase's optimization measurably easier or more stable than a
from-cold hard mask — a legitimate, but secondary, readout (§9).

---

## 4. Measurement plan — the Wave −1 gate as a manipulation check

The task brief is explicit: the gate that failed becomes the manipulation check — a
candidate must move Gini/write-mass past the pre-registered thresholds **or it isn't a
test**. Applying this literally across all four candidates surfaces an asymmetry that
must be registered before any run, not discovered after.

### 4.1 Trivial-by-construction vs. genuinely empirical (the asymmetry)

**Candidates 1 and 3 pass the original numeric bars by tautology, not by discovery.**
Once non-selected β is architecturally forced to exactly zero, `top_k_mass_frac = 1.0`,
`mean_nonselected_beta = 0.0`, and `nonselected_write_mass_frac = 0.0` **identically, for
any input, by construction** — re-running the exact same probabilistic gate script on a
hard-masked model is not a manipulation check in the causal-inference sense at all, it
is a unit-test-shaped assertion dressed as an experiment. Restating §1's inherited
`[LEARN]` finding: the original gate's (a)/(b) are exact algebraic complements over one
denominator; under hard masking this collapses further — both criteria degenerate to a
single trivial yes/no ("is the mask exactly `K_sel`-sparse") carrying zero information
beyond what a construction-level assertion already guarantees.

**The real manipulation check for candidates 1 and 3 is therefore reframed as a
training-viability check, not a geometric-threshold check:** (i) does the model
converge to a reasonable val-loss floor at all under the forced constraint (STE bias for
candidate 1; a schedule-misalignment risk for candidate 3); (ii) is the gradient
everywhere finite (mirroring geo3's own §14.6 item 2 convention); (iii) for candidate 1
specifically, what is the selection-set churn rate (§8 item 2) — a genuinely open,
not-construction-guaranteed quantity.

**Candidates 2 (entmax/sparsemax) and 4's soft phase are where the original bars are
genuinely informative**, because neither guarantees anything about the gate metrics by
construction: sparsemax/entmax's support size is content- and score-dependent, not fixed
to `K_sel`; the soft auxiliary loss (candidate 4) makes no hard guarantee at all before
any hard-snap phase. Re-running the exact original probe script on these two candidates
is a real test of whether the mechanism concentrates β enough — closer in spirit to the
original gate's own intent (measuring an emergent property of a learned mechanism)
than candidates 1/3 allow.

**A registered caveat specific to candidate 2:** the inherited top-`K_sel`-mass-fraction
metric implicitly assumes a fixed `K_sel`, which sparsemax/entmax do not respect — a
genuinely sparse-10 output at K_sel=32 would read as ~100% "top-32 mass captured"
despite having a much narrower true support than K_sel implies. Candidate 2's Wave −1
measurement must **additionally** report the actual support size (count of exactly-zero
entries), not rely on the inherited metric alone (§8 item 5 restates this as a required
attack-round finding, pre-answered here rather than left for the attack round to
discover).

### 4.2 What gets measured, per candidate, and against which bars

| Candidate | Manipulation-check content | Bars |
|---|---|---|
| 1 (hard top-K + STE) | Training viability (val loss finite/reasonable, gradient-finite) + selection-churn rate | No construction-trivial gate re-run reported as an experimental finding; churn rate is the real diagnostic (§8 item 2) |
| 2 (entmax/sparsemax) | Original gate metrics (genuinely empirical here) **plus** actual support size | Original thresholds (≥60% top-K mass / ≤0.25 mean non-selected β / ≤0.40 non-selected write-mass), reported alongside support-size, both descriptive |
| 3 (fixed schedule) | Training viability only (gate metrics are construction-trivial: selected/non-selected split is fixed by the preprocessing step, not learned) | Same training-viability bars as candidate 1 |
| 4 (soft phase) | Original gate metrics — genuinely empirical, predicted (per §3.4) to still fail | Original thresholds; failure here is an expected, informative replication of the F-geo-1 pattern, not a surprise |
| 4 (hard-snap phase) | Same as candidate 1 once snapped | Construction-trivial, same reframing as candidate 1 |

---

## 5. Pre-registered bars for the scientific readout — the 2×2 factorial

Per the task brief and the KEY_ANCHORING precedent (§2, principle 1): the actual
scientific question is not "can selectivity be forced" (§4 already answers that
per-candidate) but **"does forcing selectivity, combined with geo3-style
orthogonalization, improve the LM's memory-fidelity/attractor metrics beyond either
ingredient alone"** — mirroring the exactness program's own 2×2 logic exactly.

### 5.1 The four cells

| | Continuous β (no hard selectivity) | Hard-selective β |
|---|---|---|
| **No geo3** | **Cell 1 (baseline)** — the archived Wave C checkpoint, free (already exists, `n_params=14,048,896`, `d_model=256, d_state=64, n_layers=2`) | **Cell 2 (selectivity-only)** — the surviving candidate(s) from §4, `geo3_active=False` |
| **geo3 applied** | **Cell 3 (geo3-only, reference arm)** — Track B's **original, already-built** construction (`--use-geo3-lm --geo3-selection-mode beta_topk`), the exact construction whose own Wave −1 gate already returned `no_launch_redesign` | **Cell 4 (target)** — the surviving candidate + `geo3_active=True`, geo3 now orthogonalizing a genuinely selective (near-zero-non-selected-mass) write set |

Cell 3 is included **only** as the pre-registered 2×2 reference arm the interaction test
requires, not as a viable standalone candidate — it is already known (§1) to violate its
own mechanism premise, and is expected to underperform accordingly. Running it requires
a **deliberate, logged override** of `run_lm_rd_geo3_sweep.py`'s existing hard refusal
(`selection_mode_for_verdict` raises given `verdict="no_launch_redesign"`,
`:165–169`) — this refusal is a real safety feature against treating a failed-gate
construction as a primary arm, and must not be silently bypassed. Mirroring
`KEY_ANCHORING_DESIGN.md`'s own `--unblind-override`/override-stamping precedent (R4-1,
§8.3): any cell-3 run under this redesign must write an explicit
`gate_override: true` + `gate_override_reason: "reference arm, SCALE_TRANSFER_DESIGN.md §4.2 outcome (iii)"`
+ timestamp field into its own result JSON at assembly time, never post-hoc, so it can
never later be mistaken for a passing-gate primary run by anyone reading the JSON in
isolation.

### 5.2 Metrics — what "memory-fidelity/attractor metrics" means at LM scale

The LM has no natural hop/recovery structure the synthetic grammar has, so the readout
is scoped honestly to what Track B's own instrumentation plan (§4.4) already commits to
building, plus the same-instrument-family numbers Track C's rung-1 probe already
measured at the identical `chunk=64`/`d_state=64` convention:

1. **Per-chunk key-Gram deviation at selected/orthogonalized positions** — reusing
   `rank_utils.py`/`model_rd.py`'s existing Gram-deviation computation, per Track B's own
   required build item (§4.4).
2. **Cross-chunk drift diagnostic for repeated tokens** — the free-text analog of
   F-geo-3's entity-drift diagnostic (§4.4: "how much a token's orthogonalized key
   direction changes across the differently-composed chunks it appears in"), with §4.4's
   own caveat restated: free text's chunk membership is far less stable than a synthetic
   K-cycle's fixed episode membership, so this instrument is expected to be
   noisier/worse-behaved than the synthetic-harness case it is ported from, not assumed
   to transfer cleanly.
3. **Effective/stable rank** of the intervened region (`rank_utils.py`, unmodified).
4. **Bonus, conditional, never required:** a recovery-style metric, only if Track C's
   own frontier-probe transplant (`SCALE_TRANSFER_DESIGN.md` §5.5 item 2) has separately
   validated at rung 1 by the time this redesign runs — an explicit **dependency on a
   different track's own gate**, not something this redesign controls or may assume.

### 5.3 Numeric bars (this design's own proposal — flagged as such, calibrated against archived same-instrument-family numbers, not house-derived)

Cell 1's reference band is **inherited from Track C's own rung-1 probe** (`§5.9` of
`SCALE_TRANSFER_DESIGN.md`), the nearest already-measured same-architecture,
same-`chunk=64` number available, since Track B's own key-Gram logging was still a
"required build item" (§4.4) as of the reading list — not yet run under Track B's own
instrumentation at the time of this writing:

- Cell-1 reference: raw key-Gram deviation **21.93 ± 5.90** (Wave C control, 14M,
  `dm256/L2`, `chunk=64`); random-unit-vector anchor **7.94** (K=64, d=64); full
  collapse **63.50** (same, §5.9/§6.8).
- **Proposed headline bar for cell 4:** key-Gram deviation at selected positions ≤
  **half** of cell-1's reference mean (≤ ≈11.0) **and** markedly closer to the random
  anchor (7.94) than to cell-1's own band — echoing i-strong's 0.000-exactness reading
  convention at the synthetic-harness scale, translated to a real-text-appropriate,
  non-zero target given §4.2 item 3's own free-text degradation risk (forcing apart
  non-distinct real tokens is expected to be lossier than the synthetic harness's clean
  entity separation).
- **Interaction bar (the actual test, mirroring KEY_ANCHORING §2.0's discontinuous-jump
  framing):** `cell1 − cell4 ≥ 1.5 × max(cell1 − cell2, cell1 − cell3)` — cell 4 must
  beat **both** single-ingredient cells by a margin at least 1.5× the better single
  ingredient's own gain over baseline, not merely beat cell 1 alone (which could be
  attributable to either ingredient individually). The 1.5× multiplier is this design's
  own proposed number, explicitly labeled conservative-not-derived, in the same spirit
  `DELTANET_RD_EXACTNESS_DESIGN.md` §14.4 labels `resid_tol=1e-2` — chosen by analogy,
  not fit to data that doesn't yet exist.
- **Val-loss tolerance:** selectivity-only cells (2) held to a **wider, explicitly more
  lenient** +5% relative bar (this design's own proposal — hard write-sparsity is a
  bigger architectural constraint than geo3's orthogonalization-only intervention, so a
  tighter bar here would conflate "selectivity itself costs something" with "the
  combination fails"); the combined cell (4) held to Track B's own original, tighter
  **+2% relative** bar (§4.6), since that is the claim this redesign ultimately wants to
  make comparably to Track B's original ambition.

---

## 6. Budget

**Arithmetic, stated explicitly per the task brief.** `SCALE_TRANSFER_DESIGN.md`'s own
300 GPU-h program ceiling (§7) is registered as consumed as follows: rung-2's Rev 2.1
amendment registers ≈129.4 GPU-h for its wave, **cumulative ≈163/300** at rung-2
*launch* time (§5.6, "cumulative ≈162.5/300" / "≈163"); rung-3's Rev 2.2 amendment,
token-matched to rung-2 at 1.5B tokens/run, prices at ≈76.2 GPU-h for its 2-run wave,
**cumulative ≈266/300** (§5.6 item 4's own arithmetic: "≈266/300 — passes every gate").
`300 − 266 = 34` GPU-h nominal remaining headroom.

**This is a projection, not a harvested actual — flagged, per the task's own
instruction.** Both the rung-2 129.4 GPU-h figure and the rung-3 76.2 GPU-h figure were
**registered before their respective waves' real completion data existed** (§5.6's own
text: "registered BEFORE any rung-3 training data exists"). At the time the documents
this redesign is grounded in were read, `EXPERIMENT_LOG.md` contains **no** rung-2 or
rung-3 *harvest* entry with realized (not projected) costs — a direct grep for
"rung-3"/"rung 3" over `EXPERIMENT_LOG.md` returned nothing. **Before this redesign
launches anything, the current actual cumulative GPU-h spend across the whole
SCALE-TRANSFER program must be re-read from the latest `STATE.md`/`EXPERIMENT_LOG.md`
entries** — rung-2's harvest and rung-3's real run may have already landed with numbers
that differ from the 266/300 figure cited here, in either direction (calibration
surprises have already moved this program's own numbers once, per `SCALE_TRANSFER_DESIGN.md`
§10 item 2's own warning that Track C is "the single largest and least-validated cost
driver").

### 6.1 This redesign's own budget, sized to fit under the ≤25 GPU-h target

| Wave | Purpose | Scope | Est. GPU-h | Gate |
|---|---|---|---|---|
| **−1 (manipulation check)** | Short probe/training runs to get past-init β/score ranking (unlike the original Track B gate, which reused an *already-trained* Wave C checkpoint — none of these candidates have one) for all 4 candidates, plus training-viability/gate-metric measurement per §4 | 4 candidates × ~2,000-step short probes × 1 corpus (openr1) × 1 seed, at Wave C's own measured ≈0.0000123 GPU-h/step (0.077 GPU-h / 6,103 steps) → ≈0.008 GPU-h/probe | **~0.05–0.1** | Training-viability pass (candidates 1/3) or genuine gate-threshold measurement (candidates 2/4) per §4.2's table; ranks candidates for Wave 1 |
| **1 (selectivity-only, full manifest, surviving candidates)** | Full Wave-C-scale training, `geo3_active=False`, establishes val-loss floor + churn/support diagnostics | up to 4 candidates × 2 corpora × 3 seeds = up to 24 runs × 0.077 GPU-h | **~0.9–1.85** | Val loss within §5.3's +5% bar; churn-rate (candidate 1) / support-size (candidate 2) logged; ranks the top 1–2 candidates for Wave 2 |
| **2 (2×2 factorial, top 1–2 ranked candidates only)** | Cell 3 (override-flagged reference arm, run once, shared across candidates) + cell 4 (selectivity+geo3 combined) per surviving candidate | 1 shared cell-3 run (2 corpora × 3 seeds = 6 runs) + up to 2 candidates × cell-4 (2 corpora × 3 seeds = 6 runs each) = 18 runs × 0.077 GPU-h | **~1.4** | §5.3's interaction bar; val loss within the tighter +2% bar for cell 4 |
| **3 (geometry/drift/rank instrumentation, all cells)** | Port §4.4's key-Gram/drift instrumentation; forward-hook probes, no backward pass, mirroring Track C/D's own ~0.08–0.9 GPU-h probe costs | all cells from waves 1–2 | **~1–2** | Descriptive, Tier 2 throughout |
| **Total** | | | **≈3.5–5.5 GPU-h central estimate** | |

**Wide-case (2× multiplier, this project's own standing calibration-uncertainty
convention, e.g. `SCALE_TRANSFER_DESIGN.md`'s own "±2×" pre-calibration framing, §5.6):
≈7–11 GPU-h.** Even doubled again for a genuine worst case (all 4 candidates carried
through the full 2×2 rather than the top-ranked 1–2, plus entmax/sparsemax's
new-primitive engineering risk materializing as extra debug-run spend) the total stays
**well under the 25 GPU-h target**, and comfortably under the ≈34 GPU-h nominal
headroom even before that headroom figure is itself re-verified. This redesign's own
spend is genuinely small relative to the program ceiling; the arithmetic above exists
to make that margin explicit and auditable, not because this redesign is expected to
threaten the ceiling on its own.

---

## 7. Falsification map

| Claim | Falsified by |
|---|---|
| "Hard selectivity is achievable on this harness without destroying trainability" | Any candidate's val loss fails to converge to a reasonable floor at all (not just misses the ±5%/±2% tolerance — an outright divergence or NaN-collapse) under standard skip-step handling |
| "The gate metrics, once forced, are a meaningful manipulation check" | For candidates 1/3: nothing — §4.1 already establishes these are construction-trivial; the manipulation check for these candidates is training-viability, not the gate metric, and no result on the gate metric itself can falsify or confirm anything |
| "Selectivity alone measurably improves memory-fidelity metrics" | Cell 2 does not clear cell 1 by a margin distinguishable from seed noise — consistent with, not contradicting, KEY_ANCHORING's own "stability/selectivity alone buys little" finding (§2) |
| "Orthogonality + selectivity interact super-additively on free text, mirroring the synthetic-harness 2×2" | Cell 4 fails §5.3's interaction bar (`cell1−cell4 < 1.5×max(cell1−cell2, cell1−cell3)`) — the additive, non-interacting outcome is equally reportable and would mean the synthetic-harness interaction does not transfer to free-running real text at this scale |
| "This redesign licenses a claim about pretrained/production delta-rule LMs" | Nothing in this design can establish this regardless of outcome — Track D's own Phase 1 non-attribution finding (§9 of this document) already forecloses it a priori for any from-scratch small-model result |

---

## 8. Five most likely attack-round findings, pre-answered

1. **"The manipulation check is trivially satisfied for candidates 1/3 — is this
   redesign just moving the goalposts until the test can't fail?"** Yes, exactly, for
   the *geometric-threshold* framing — §4.1 states this explicitly and reframes the real
   manipulation check for those two candidates as a training-viability check instead.
   This is not a defect discovered by the attack round; it is disclosed here, before any
   run, as a structural property of what "literal zero" means for a gate built to detect
   *emergent* concentration.
2. **"STE's hard top-K mask will churn — positions near the K_sel-th-ranked boundary
   flip in/out step-to-step, a distinct instability from ordinary optimization noise."**
   Registered as a required, first-class Wave −1/Wave 1 diagnostic (selection-set churn
   rate across consecutive checkpoints), not an afterthought — mirroring this project's
   own "a first-class readout, not an afterthought" convention (`SCALE_TRANSFER_DESIGN.md`
   §4.8 item 4, applied there to the drift diagnostic for exactly this reason).
3. **"Forcing exactly K_sel writes per chunk is also a write-BUDGET change, not just a
   targeting change — any downstream effect could be attributable to the budget shift,
   not the selectivity-orthogonality interaction."** Registered explicitly for candidate
   2 (§3.2, where the chunk-sum-≤1 constraint is a *known* budget change) and applies in
   milder form to candidates 1/3/4 too (total per-chunk β-mass under a hard mask can
   differ from the unconstrained baseline's own total). Total per-chunk β-mass (or an
   equivalent "write energy") must be reported alongside every val-loss number, per this
   project's own hold-the-second-axis-fixed hard rule (`CLAUDE.md`), before attributing
   any regression or improvement to selectivity per se.
4. **"Even trained from scratch, will a future write-up quietly frame these results as
   'fixing' Wave C's pretrained LM, blurring the different-model-tier boundary?"** This
   is exactly what §9 exists to foreclose explicitly, per-candidate, not left implicit.
5. **"The inherited top-K_sel-mass-fraction metric assumes a fixed K_sel that
   entmax/sparsemax does not respect — a genuinely sparse-10 output could misleadingly
   read as ~100% 'top-32 mass captured.'"** Pre-answered in §4.1: candidate 2's
   measurement plan requires reporting actual support size as an additional, required
   statistic, not relying on the inherited metric alone.

**A sixth, process-level lesson carried forward rather than re-earned:**
`DELTANET_RD_EXACTNESS_DESIGN.md` §16.7's dated correction found that a predictive
"launch-read" simulator tool wired a *single* K's measured drift scalar into *both* K=16
and K=32's predictions, silently mis-attributing the K=32 result. If this redesign ever
builds an analogous predictive/manipulation-check tool that takes a per-candidate or
per-K_sel measured statistic as input, that statistic must be threaded per-cell, never
shared across cells via one API call — a concrete, checkable build-time item, not a
vague "be careful" reminder.

---

## 9. Claim tiers

Per `SCALE_TRANSFER_DESIGN.md` §2's house tier system (Tier 1 premise-conditional
causal / Tier 2 descriptive+interventional / Tier 3 measurement-only), inherited
verbatim, extended per-candidate:

- **Candidates 1, 2, 4 — trained from scratch (this design's own default, §2 principle
  2): Tier 2 only.** "This intervention, on this from-scratch-trained small LM, at this
  scale/window/selection definition, changed this measured quantity by this amount, on
  this corpus" — Track B's own §4.7 language, inherited exactly, since these candidates
  are architecturally the *same model class* (`DeltaNetLMMixer`) as Wave C, merely
  retrained with a different β mechanism. **None of these may claim** "we fixed Wave C"
  or "pretrained/production LMs of this family need hard selectivity" — a from-scratch
  run is a different trained instance, not a repair of an existing one, regardless of
  architectural continuity.
- **If a future follow-on instead continue-trains (retrofits) any of candidates 1/2/4
  onto the *existing* Wave C checkpoint** (explicitly NOT this design's default, §2
  principle 2, §9 of this section names it as a possible but non-default follow-on): it
  inherits the Track-D-style bolt-on risk in full, and can claim **at most Tier 2 for
  that one specific checkpoint** — never a general claim, mirroring
  `SCALE_TRANSFER_DESIGN.md` §2's own precedent for Track D's Phase 2 ("would, if it ever
  runs, earn at most Tier 2 for one specific model, never a general claim about the
  delta-rule family").
- **Candidate 3 (fixed periodic write schedule) — different-model tier, explicitly.**
  This is not a different trained instance of the same architecture; it is a different
  architecture (a distinct tokenization/preprocessing pipeline, a non-learned
  write-decision mechanism structurally closer to `model_rd.py`'s synthetic-harness
  convention than to `lm_pretrain_rd.py`'s LM convention). Its scientific value is
  narrower and more purely mechanistic — "does orthogonality+stability jointly help when
  a hard, non-learned write-schedule exists on free text" — and it **must not** be cited
  as evidence about what continuous-β LMs of the Wave-C/`DeltaNetLMMixer` family do or
  need, let alone pretrained ones.
- **Restated, per the task brief's own explicit instruction, against Track D's
  non-attribution finding (`STATE.md`, `SCALE_TRANSFER_DESIGN.md` §6.8):** even if this
  redesign's 2×2 factorial reproduces KEY_ANCHORING's clean discontinuous-interaction
  shape (§2 principle 1) on this project's own from-scratch small models, that result
  licenses **zero** claims about production/pretrained fixed-state LMs. Track D's own
  Phase 1 measurement already found that the write-geometry-attractor signature at
  production scale (RWKV-7 1.5B, Falcon-Mamba-7B) is **not attributable to the
  delta-rule write mechanism specifically** — the registered no-fixed-state negative
  control (Qwen2.5-1.5B, standard softmax attention, no recurrent state of any kind)
  showed the **same magnitude** signature, meaning the effect is dominated by generic
  trained-LM key anisotropy (massive activations, Sun et al. 2024), present regardless
  of architecture. A clean Track-B-family finding, however clean, stays confined to Tier
  2 language about this project's own from-scratch harness; any temptation to generalize
  toward "therefore pretrained delta-rule LMs need hard-selective writes too" is
  explicitly blocked by this already-measured result, cited by name, not merely gestured
  at.

---

## 10. Cut order

**Never cut:** the Wave −1 manipulation check for candidate 1 (cheapest, most direct
answer to Track B's own gate finding, and the necessary precondition-satisfaction check
before anything else in this redesign is worth running).

**Cut, in this order, until back under whatever budget line is live at launch time:**

1. Candidate 2 (entmax/sparsemax) — highest build risk (new autograd primitive, unclear
   chunk-sum-budget interaction, §8 item 3) and the one candidate whose manipulation
   check is most likely to need its own threshold re-derivation (§8 item 5) before it is
   even interpretable; cut first if squeezed.
2. Candidate 4's soft phase — already predicted, on this project's own F-geo-1
   precedent, to underperform (§3.4); its only unique informative content (does soft
   warm-up ease the subsequent hard phase) is a secondary readout, not a headline one.
3. Candidate 3 — highest scope change (new preprocessing pipeline); cut before
   candidate 1's own manifest is trimmed, since candidate 1 is cheaper, more direct, and
   answers the queued idea's own literal framing ("hard-zero-β") most closely.
4. The shared cell-3 (geo3-only) reference arm's second corpus — drop to openr1 only if
   squeezed; the interaction bar (§5.3) is directionally assessable from one corpus,
   though the full claim needs both, mirroring `SCALE_TRANSFER_DESIGN.md` §8 item 5's
   own precedent for a symmetric cut.
5. Candidate 1's own naive-window-style ablation sensitivity check (if one is added
   later) before candidate 1's primary beta-topk arm itself — the primary arm carries
   the entire causal-adjacent claim and is never cut.

---

## 11. Open questions for the attack round

1. Is §5.3's proposed headline bar (cell-4 Gram deviation ≤ half of cell-1's reference
   band, closer to random than to baseline) the right calibration, or should it be
   derived more rigorously from a simulator-style tool analogous to
   `geo3_simulator.py` — and if such a tool is built, does it repeat §16.7's shared-scalar
   mis-wiring risk (§8's sixth, process-level finding)?
2. Is candidate 1's straight-through estimator the right gradient approximation, or
   would a hard-concrete/L0-style relaxation (Louizos, Welling & Kingma 2018, cited by
   name in §3.1 but not adopted as primary here) behave meaningfully better at this
   scale — an open empirical question this design does not resolve by fiat.
3. Is cell 3 (the geo3-only reference arm) worth spending real GPU-h on at all, given
   its own Wave −1 gate already measured its premise fails — or should its role in the
   2×2 be filled by inference from the *existing* gate JSON's already-measured numbers,
   saving the ≈0.46 GPU-h a fresh cell-3 run would cost? This redesign currently assumes
   a fresh run is worth it (a clean same-seed/same-manifest comparison point), but the
   attack round should weigh this explicitly rather than let it pass by default.
4. Does candidate 3's fixed-period write schedule need its own attack pass on the period
   choice itself (`W=chunk_size/K_sel` is asserted here, not derived) before any build,
   mirroring the escalation discipline this project applies to every other free
   hyperparameter?
5. Is the ≈34 GPU-h headroom figure (§6) still accurate — has rung-2's harvest or
   rung-3's actual run landed with numbers that differ materially from the 266/300
   projection this document cites? This must be re-checked against the latest
   `STATE.md`/`EXPERIMENT_LOG.md` before this redesign's own §6 budget is treated as
   final, not merely noted as a caveat and then ignored.

---

## Reproducibility pointers

This design: `matrix-thinking/TRACKB_REDESIGN.md` (Rev 1, this file). Grounding sources,
all read in full or in the cited section this session: `SCALE_TRANSFER_DESIGN.md` §4
(+§2, §6.4, §6.8, §5.6, §5.9, §7); `EXPERIMENT_LOG.md`'s "SCALE-TRANSFER Track B" entry
and its two `[LEARN]` blocks; `matrix-thinking/deltanet_rd/results/lm_rd_geo3/wave_neg1_gate.json`
(raw numbers, read directly); `DELTANET_RD_EXACTNESS_DESIGN.md` §14, §16 (incl. §16.7's
dated correction); `KEY_ANCHORING_DESIGN.md` Rev 5 §1–§2.2, §3.6/R4-1;
`matrix-thinking/deltanet_rd/lm_pretrain_rd.py` (`:1–60`, `:206–398`, `:419–570`);
`matrix-thinking/deltanet_rd/run_lm_rd_geo3_sweep.py` (`:165–169`); `STATE.md`
"Hardware" section. No GPU run, no box access, no push performed in producing this
document.
