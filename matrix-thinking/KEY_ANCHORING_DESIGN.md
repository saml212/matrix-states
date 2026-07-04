# KEY_ANCHORING_DESIGN — From "orthogonal but drifting" to "stable and orthogonal": a follow-on to F-geo-3

> **Rev 1 — 2026-07-03. Design only. No model/training code is written here,
> nothing is launched, no GPU or SSH access was used to produce this
> document.** This is the pre-registered design for the follow-on wave named
> at the end of `DELTANET_RD_EXACTNESS_DESIGN.md` §16.6's outcome-F
> attribution: F-geo-3 (per-episode Newton-Schulz key orthogonalization)
> **HIT** the K=16 minimum-publishable bar 3/3 admissible seeds (`rec@0.9`
> h=4 = 0.9767 mean, bar ≥0.8) but **narrowly MISSED** the K=32 headline bar
> (h=4 = 0.4368 mean, bar ≥0.5, ~0.06 short) at full evidentiary tier (3/3
> admissible, zero fallbacks after the `n_iter=12→20` escalation). The named
> residual bottleneck is **cross-episode key drift**: mean pooled pairwise
> cosine of an entity's own orthogonalized key across independently-resampled
> episode contexts measures **0.9037 at K=32** and **0.9416 at K=16**, both
> below the pre-registered 0.95 HIGH-drift threshold
> (`DELTANET_RD_EXACTNESS_DESIGN.md` §14.5, §16.1, §16.6). Keys are
> well-orthogonalized *within* an episode; the key **frame is not stable
> across episodes**. This design's target is **KEY ANCHORING**: hold the
> already-solved orthogonality fixed (reuse geo3's own machinery unchanged)
> and add cross-episode **stability** on top of it.

**A load-bearing finding surfaced while writing this design, not assumed
going in (flagged per this project's standing transparency norm — see
§2.0):** the *fully* stable-and-orthogonal cell of the relevant 2×2 has
**already been run and archived, at zero further GPU cost.** Arm
(i-strong) — `DELTANET_RD_EXACTNESS_DESIGN.md` §4.4, a hand-built,
per-entity, context-free `k_eff` pin that bypasses `W_k`/conv entirely —
was previously reported only at h=1/2/3 (§14.0, §15.5: "1.00/1.00/1.00").
Extracting its **held-out** hops directly from the already-archived result
JSONs (`experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wave1/
w1_rdx_K32_armi-strong_s{0,1,2}_sp.json`, read this session, not
previously tabulated at these hops in `DELTANET_RD_EXACTNESS_DESIGN.md`
§14–16 or `EXPERIMENT_LOG.md` beyond the qualitative "saturates every
headline hop" note) shows: **`rec@0.9` h=4/5/6/7 = 1.0000 in all 3 seeds,
at every logged checkpoint from step 2000 onward**; even h=21 (the
literal-iteration-depth cell every other arm collapses on, §16.8) climbs
from 0.0 at step 2000 to **0.9987–0.9994 by step 20000**. This is the
existence proof this design's hypothesis needs, already sitting in the
archive. §2.0 below spells out the full, now-quantified 2×2 this implies.

---

## 0. Reading list this design builds on

- `matrix-thinking/DELTANET_RD_EXACTNESS_DESIGN.md` §14 (F-geo-3's full
  mechanism spec — Newton-Schulz orthogonalization, the `bind()`/`readout()`
  threading, the substitute admission stack, the pre-registered drift
  bands), §15 (Wave F soft-arm results — `L_orth`/ZCA both saturate at 3–8%
  Gram cleanup, nowhere near i-strong's 0.000; the informative parallel this
  design's own attack surface (§7) leans on), §16 (F-geo-3 results — the
  K=16 hit / K=32 narrow miss / outcome-F attribution this design extends).
- `matrix-thinking/deltanet_rd/model_rd.py` — read in full; exact insertion
  sites cited by line number throughout §2.
- `matrix-thinking/deltanet_rd/run_deltanet_rd_exactness_sweep.py` — the
  wave harness (`_spec`, `geo3_wave1_manifest`, `gate_geo3_drift`) this
  design's own manifest/gate functions are modeled on.
- `matrix-thinking/deltanet_rd/geo3_simulator.py` and
  `matrix-thinking/deltanet_rd/geo3_drift_diagnostic.py` — the committed
  launch-gate pattern (§14.6/§14.13) this design replicates, and (§4 below)
  extends with a documented, honestly-labeled calibration attempt.
- New numbers produced **this session, CPU-only** (no GPU, no repo files
  modified — a scratch script importing `geo3_simulator.py` under a local
  throwaway venv): the i-strong held-out-hop extraction above; a
  reachability check on the i.i.d.-Gram-deviation ceiling; a naive
  value-Gram-corrected simulator extension and its calibration failure mode;
  and a Pearson correlation between key- and value-Gram deviation across
  archived checkpoints. All four are cited with their exact numbers and
  method in §2.0 and §4 — none are invented, all are reproducible from
  files already in this repo.

---

## 1. Hypothesis

Anchoring the key frame across episodes — holding the already-solved
per-episode orthonormality fixed via geo3's existing Newton-Schulz
machinery, and adding a mechanism that makes an entity's orthogonalized key
**consistent across independently-drawn episode contexts** — lifts K=32
h=4 composition from its measured 0.4368 (§16.4) past the pre-registered
≥0.5 bar, **because** cross-episode key drift (measured 0.9037 at the
trained checkpoint, §16.1/§16.6) is the named residual bottleneck, not a
restatement of the orthogonality problem F-geo-3 already solved.

---

## 2. Intervention candidates

### 2.0 Why these four, against a now-quantified 2×2

Before designing new interventions, the already-archived data answers a
sharper question than "does stability matter": **does it matter more than,
less than, or jointly with orthogonality?** Four cells, all real, all
K=32, d_state=64, from files already in this repo (extracted this session;
Gram-deviation numbers are each run's own final-checkpoint
`key_gram_deviation_mean`, `rec@0.9` is each run's own final-checkpoint
`M3_held_out` field):

| | key-Gram dev (orthogonal?) | key-Gram dev (NOT orthogonal) |
|---|---|---|
| **stable** (context-free key, same regardless of episode) | **i-strong**: dev 0.000, h=4 **1.0000**, h=7 **1.0000** (all 3 seeds, `w1_rdx_K32_armi-strong_s{0,1,2}_sp.json`) | **arms i/iv** (frozen embedding table, but `W_k` still learns to de-orthogonalize, §16 "Wave 1 ATTRIBUTION VERDICT"): dev 2.71–2.87, h=4 **0.0049–0.0114**, h=7 **0.0000** (`w1_rdx_K32_armi_s*.json`, `w1_rdx_K32_armiv_s*_rho0p225.json`) |
| **NOT stable** (episode-conditional key) | **geo3**: dev ~7×10⁻⁷, h=4 **0.4368** [0.3903–0.5045], h=7 **0.0177** (§16.2/§16.4) | **baseline / arm iii-beta**: dev 2.72–2.77, h=4 **0.0073–0.0103**, h=7 **0.0000–0.0001** (`w1_rdx_K32_armiii-beta_s*.json`) |

Read across the table: **neither ingredient alone matters** (stable-only
and baseline are statistically the same, ~0.005–0.010 at h=4 — stability
without orthogonality buys nothing, confirming §16's own "raw embedding
geometry is causally irrelevant" verdict extends to this axis too);
**orthogonality alone buys most of the gap** (geo3's 44–56× jump over
baseline, §16); **both together is not a partial improvement over geo3, it
is a discontinuous jump to saturation** (0.44 → 1.0000, not 0.44 → 0.6 or
0.8). This is why the hypothesis is framed as an *interaction*, not a
second additive lever, and why candidate (a) below is reported as an
existence proof rather than designed as a new experiment — it already ran.

### 2.1 Candidate (a) — Fixed/frozen key frame (existence-proof tier). ALREADY CONFIRMED, zero new cost.

**What it is.** Arm (i-strong) itself, reframed: `k_eff_items[j] :=
u_{entity(j)}`, a persistent, hand-built, per-entity orthonormal lookup
(`u: entity → R^64`, drawn once via QR), shared across every episode and
train/eval alike. **No new code** — `model_rd.py`'s existing
`strong_pin_table` (constructor lines 610–611, 715–725), `bind()`'s
`strong_pin_active` branch (lines 856–870), and `effective_key_window()`'s
matching branch (lines 780–798) are the exact mechanism.

**Cost.** Zero — the data already exists (§0). This wave's own contribution
is the re-analysis in §2.0, not a new run.

**Why it is not itself KEY_ANCHORING's deliverable.** i-strong is a
deliberate **two-variable** exception (`DELTANET_RD_EXACTNESS_DESIGN.md`
§4/§4.4): it bypasses the learned key path entirely, works only for the
closed, hand-built entity pool it was constructed over (no generalization
to identities outside that pool), and requires the identity of the queried
entity be known independent of any learned representation. It proves the
mechanism is real and the upper bound is reachable; it is not a trainable
fix. Its role here is the pre-registered ceiling every trainable candidate
below is read against.

**If it "failed" (it didn't):** it already didn't — this is the strongest
possible confirmation the hypothesis has. Its only open risk is scope: it
does not by itself tell us whether a mechanism that keeps the learned key
path in the loop can get anywhere close.

### 2.2 Candidate (d) — Anchor-first blend with a learned per-entity anchor table (PRIMARY, structural tier)

**Mechanism.** A new, genuinely trainable, backprop-connected per-entity
anchor table `A` (an `nn.Embedding(vocab_size_total, d_state)`, initialized
via the **same QR-orthonormal construction** `strong_pin_table` already
uses at init, lines 715–722 — a trainable *starting point*, not a frozen
pin). At `bind()`'s existing gather site, **before** Newton-Schulz runs,
blend each entity's raw per-episode key toward its own anchor row:

```
k_blend_raw[j] = normalize( (1 - λ) · k_eff_raw[j] + λ · A[key_ids[j]] )
k_eff_items    = geo3_orthogonalize_logged(k_blend_raw, n_iter=..., resid_tol=...)   # UNCHANGED call
```

`λ` is registered as a **single learned scalar** (`sigmoid(raw_param)`,
init 0.5) — SGD decides how much to lean on the stable anchor versus the
raw episode-conditional signal, matching this project's own "structural,
not soft-penalty; let training decide" house preference
(`DELTANET_RD_EXACTNESS_DESIGN.md` §5.5). A **fixed grid**
`λ∈{0.3,0.6,0.9}` (no further tuning) is the pre-registered fallback
diagnostic, run only if the learned-λ trajectory is ambiguous (flat,
oscillating, or pinned at an endpoint with no clear reading).

**Why this order, not orthogonalize-then-blend (candidate (b)'s order):**
blending the RAW keys before the single Newton-Schulz call keeps
`geo3_orthogonalize_logged`'s contract, cost, and fallback semantics
**completely unchanged** — one NS pass, same `resid_tol`, same
fallback-triggered flag meaning. Blending *after* orthogonalization (as in
candidate (b)) breaks the guaranteed `QQ^T=I_K` the blend was just
enforcing, and requires a second NS pass to restore it (§2.3's own cost
line explains why that candidate is ranked second, not first).

**Degenerate limits, stated plainly:** at `λ→0` this is bare geo3
(§16's own measured 0.4368 at K=32). At `λ→1` this becomes (once
`A[key_ids]` is itself well-conditioned) an orthogonalized version of
i-strong's own construction — i.e., the family **interpolates smoothly
between the two already-measured endpoints of §2.0's 2×2**, rather than
betting on an unexplored point.

**Insertion sites (`model_rd.py`):**
- Constructor: add `anchor_active: bool = False`, `anchor_lambda_mode:
  str = "learned"` (or `"fixed"`), `anchor_lambda_fixed: float | None =
  None` to `DeltaNetRDBlock.__init__` (alongside the existing
  `geo3_active`/`geo3_n_iter`/`geo3_resid_tol` args, lines 612–613);
  register the `nn.Embedding` and (if `anchor_lambda_mode=="learned"`) a
  scalar `nn.Parameter`, mirroring the existing `strong_pin_table` buffer
  construction (lines 715–722) but as a **trainable** module, not a
  `register_buffer`. Assert `not (anchor_active and strong_pin_active)`
  (same mutual-exclusivity pattern as line 653–655) and `anchor_active
  implies geo3_active` (anchoring is defined as a modification to geo3's
  own write path, not a replacement for it).
- `bind()`'s existing `geo3_active` branch (lines 871–888): insert the
  blend line between `k_eff_raw = _gather_at(...)` (line 877) and the
  `geo3_orthogonalize_logged(...)` call (line 878) — one new line, the
  orthogonalization call itself is untouched.
- `readout()`/`forward()` (lines 926–990): **byte-identical**, no change —
  the query-side `a_slot` gather from `k_eff_items` (lines 943–953)
  already reads whatever `bind()` produced, regardless of how it got
  there.

**Param/FLOP cost.** `vocab_size_total × d_state ≈ 50,259 × 64 ≈ 3.22M` new
trainable params (~13MB fp32, mirroring the `strong_pin_table` comment's
own arithmetic at lines 713–714) — a real, disclosed **~25% increase** over
the reported 12,899,841-param baseline (§16 header). FLOPs: one extra
gather + blend + renormalize per `bind()` call, then the **same** single
12-iteration Newton-Schulz pass geo3 already pays for — no second
orthogonalization, unlike (b).

**What failure would mean.** If the learned λ stays near 0 (SGD prefers the
raw per-episode signal even though the anchor is available), that is the
informative negative: the raw episode-conditional key carries information
the loss values enough to resist stabilizing — a genuinely different
failure mode from F-geo-1/2's saturation (§7 names the mechanistic reason
this is plausible: this session's own finding that geo3's value-Gram
deviation is 1.3–1.8× **worse** than baseline at both K, §4 below, suggests
value geometry may already be under strain when keys are forced clean; an
anchor that pulls keys even further from their "natural" per-episode
value-co-adapted direction could make this worse, giving SGD a live reason
to resist λ→1).

### 2.3 Candidate (b) — EMA-anchored frame with post-hoc re-orthogonalization (SECONDARY, conditional fallback)

**Mechanism.** A slowly-updating, **stop-gradient** EMA anchor (mirroring
`ZCAWhiten`'s own `_update_stats`/`_recompute_transform` pattern, lines
530–568, momentum convention 0.99, bias-corrected): after geo3's own NS
pass produces `k_eff_items` for this episode, update
`A[key_ids[j]] ← momentum·A[key_ids[j]] + (1-momentum)·k_eff_items[j].detach()`
(no_grad, BN-style — the anchor never receives its own gradient, only
copies the model's own settled output), then blend the *orthogonalized*
key toward the anchor and **re-orthogonalize once more**:
`k_write = geo3_orthogonalize_logged(normalize(k_eff_items + λ(A[key_ids] -
k_eff_items)), ...)`. `A` is a `register_buffer`, not an `nn.Parameter` —
identical no-gradient-into-statistics discipline to `ZCAWhiten` (line 483).

**Why ranked second, not first.** Two costs candidate (d) does not pay:
(1) a **second** Newton-Schulz pass every step (the post-blend
re-orthogonalization) — roughly doubles this mechanism's own FLOP
contribution, though still trivial next to the kernel call itself;
(2) a genuine **circularity risk**, named here rather than discovered at
audit time: the anchor is an EMA of the model's *own* per-episode output,
which is itself a function of `W_k`/conv, which is *still training* while
the anchor accumulates. If `W_k` keeps moving, the anchor chases a moving
target and may never converge to a stable value — a qualitatively
different failure mode from (d)'s (where the anchor is genuinely
backprop-trained toward whatever helps the loss, not merely averaging a
non-stationary quantity). Kept as the **registered fallback** if (d)'s
learned-λ run fails specifically because the trainable anchor table
degenerates (e.g., collapses to a low-rank set under its own gradient,
closed by the same key-side salvage check §3/§7 registers) — a different
diagnosis than "λ stays at 0," and one an EMA-based, gradient-free anchor
would not share.

**Insertion sites.** Same constructor location as (d); the buffer/EMA
logic is a near-verbatim adaptation of `ZCAWhiten`'s existing
`_update_stats`/`_recompute_transform`/warm-up scaffold (lines 515–568) —
this project's own precedent for "EMA statistics, stop-gradient, jitter-
retry-then-clamp on `eigh` if ever needed" is reused wholesale, not
reinvented. `bind()`'s `geo3_active` branch gains the EMA update (a
`torch.no_grad()` block after the existing NS call, line 878) and a
**second** call to `geo3_orthogonalize_logged` on the blended result.

**Cost.** Same cell count as (d) if triggered (K∈{16,32}×3 seeds); per-run
FLOP overhead roughly double (d)'s own — priced separately in §5, not in
the mandatory baseline.

### 2.4 Candidate (c) — Explicit cross-episode drift regularizer `L_anchor` (soft tier, ranked LAST, ablation-only)

**Mechanism.** The same EMA anchor buffer as (b), but used **only** as a
soft loss-side target, never as a hard blend at the write site:
`L_anchor = λ_anchor · Σ_j (1 - cos(k_eff_items[j], stopgrad(A[key_ids[j]])))`,
added alongside the existing `L_cos + λ_nce·L_nce` in `run_deltanet_rd.py`'s
training loop — the direct structural transplant of F-geo-1's own
`L_orth` penalty (`DELTANET_RD_EXACTNESS_DESIGN.md` §5.5), but targeting
cross-episode **stability** instead of within-episode **orthogonality**.
Model behavior at inference is **byte-identical** to bare geo3; only the
training-time loss changes.

**Why ranked last — this is not a hedge, it is a mechanistic prediction,
made before any data exists, exactly matching the project's
falsify-before-you-run norm.** §15 already ran this *exact* playbook for a
different geometric property and it failed in a specific, diagnosed way:
F-geo-1's `L_orth` saturated at 3–8% Gram cleanup regardless of
`λ_orth∈{0.1,1.0}` (a 10× weight change bought "essentially nothing," §15.4)
because **nothing in the primary task loss requires exact orthonormality
to reduce loss** — h=1 recovery is already near-ceiling without it (§15.1),
so the soft penalty is fighting the primary loss's own indifference, not
reinforcing it. The identical structural argument applies to `L_anchor`:
h=1 scores directly against `v_eff` and is **provably drift-immune**
(geo3_simulator's own construction, §14.6 — "targets are v_eff... drift-
immune at one hop"), so nothing in `L_cos`/`L_nce` requires cross-episode
key stability either. A soft `L_anchor` term risks the same saturation
signature (proportional, real, but geometrically shrinking gain per hop,
`~ε^h`, §15.3) for the same underlying reason. It is kept in this design —
run at a single, pre-registered `λ_anchor` value at K∈{16,32}×3 seeds,
BLOCK-2's own reduced-scope template (`DELTANET_RD_EXACTNESS_DESIGN.md`
§14.7) — **as a cheap, always-run comparison point** (does *any* soft
nudge help on top of geo3's already-solved orthogonality, the way F-geo-1
partially helped baseline's raw geometry), not as this wave's primary bet.

**Insertion site.** `run_deltanet_rd.py`'s loss composition (wherever
`L_cos + λ_nce·L_nce` currently assembles, the same site F-geo-1's
`L_orth` addition used, §5.5); requires threading `key_ids` + a
(stop-gradient) anchor buffer through to the loss — no change to
`model_rd.py`'s `bind()`/`readout()` at all.

**Cost.** 6 runs (K∈{16,32}×3 seeds, one `λ_anchor`), same anchor as (d).

---

## 3. Pre-registered bars and claim tiers

**Primary bar (headline demo).** K=32, h=4, `rec@0.9` **≥ 0.5**, at full
evidentiary tier: **3/3 seeds admissible**, **zero fallbacks** (Newton-
Schulz converges without the eigh fallback at every logged checkpoint,
identical semantics to §14.10 item 2 — the blend in (d)/(b) feeds the
*same* `geo3_orthogonalize_logged` call, so "fallback" means exactly what
it already means for geo3). This mirrors §16.4's own table format and bar
exactly; no new bar is invented.

**Secondary (no-regression check, not a fresh bar).** K=16, h=4, **must
stay within −0.02 of geo3's own already-measured K=16 h=4 baseline (0.9767
mean, §16.4)** — the identical −0.02 convention this design doc's parent
already uses for the h=1 no-sacrifice guard (§5.5, §16.4, §16.8), applied
here to the ALREADY-WON K=16 cell instead of h=1. A candidate that "wins"
K=32 by degrading K=16 is not an improvement; this guard makes that a hard
admissibility failure, not an implicit hope.

**Guard (both tiers, inherited unchanged).** h=1, same K, within −0.02 of
geo3's own h=1 baseline (1.0000 at both K, §16.4/§16.8) — unchanged from
§16's own convention.

**Substitute admission stack — inherited from §14.10 verbatim, PLUS two
new items this design registers.** Every KEY_ANCHORING candidate that
touches `bind()` ((d), (b)) still routes its final write-key through
`geo3_orthogonalize_logged`, so the **same tautology** §14.9 item 1/§14.10
found for geo3 applies unchanged: key-Gram deviation ≡ ~0 and the patched
`q_self ≡ k_eff_items` alignment instrument ≡ 1.0, **by construction, for
every candidate, every seed** — both remain non-evidence, logged only.
Items 1–4 of §14.10's substitute stack (value-side salvage ≥0.1;
Newton-Schulz converged without fallback at every checkpoint; finite loss
throughout; task-performance floor — final loss ≥50% improved, h=1 ≥0.5)
apply **unchanged**. Two new items, closing gaps items 1–4 do not cover
for an anchoring (rather than a bare-orthogonalizing) mechanism:

- **Item 5 — THE MANIPULATION CHECK (this design's own gate, not
  inherited).** Cross-episode key drift, measured by the **identical
  statistic** already registered at §14.6/§16.1 (mean pooled pairwise
  cosine of an entity's orthogonalized key across ≥32 independently-
  resampled episode contexts, ≥8 entities, drawn from the train pool,
  measured at the trained final checkpoint on `k_eff_items` — the same
  object geo3's own diagnostic reads, computed by
  `geo3_drift_diagnostic.py`'s existing `measure_drift`, unmodified) **must
  reach ≥0.95 — the same LOW-drift threshold §14.5 already pinned.** **A
  run that clears h=4 ≥ 0.5 without clearing drift ≥ 0.95 does NOT count
  as a test of this design's hypothesis, regardless of the h4 number** —
  it would mean h4 moved for some other reason (§6's falsification map
  names the candidates), and the write-up must say so, not claim
  confirmation. This is the literal operationalization of the task's own
  requirement and is symmetric with the h4 bar: **both must clear
  together** for the run to count as a confirming instance.
- **Item 6 — key-side salvage ratio ≥0.1 post-anchoring (closes attack
  surface #2, §7).** `σ_K/σ_1` computed on the final `k_eff_items` tensor
  the **same way** item 1 already computes it for `v_eff_items` — geo3
  itself never needed this (its key-Gram is forced to ~0 IDENTITY by
  construction, which already implies full rank), but an anchored/blended
  key set is only guaranteed **within-episode** orthonormal by the (still
  intact) final NS pass — nothing yet checks that the anchor table itself
  hasn't collapsed toward a shared, low-rank set of directions across the
  **whole vocabulary**. A degenerate anchor could inflate the drift
  statistic favorably (same direction every time *looks* stable) while
  destroying between-entity distinctiveness; this item catches that
  directly.

**Three-way outcome frame at K=32 (extends §14.8's outcome table to this
design's own gate):**

- **Outcome A — CONFIRMED.** drift ≥0.95 AND h4 ≥0.5, 3/3 admissible under
  items 1–6. KEY_ANCHORING is the winning mechanism; write up per §7's
  claim-tier language.
- **Outcome B — INFORMATIVE NEGATIVE, re-attribution required.** drift
  ≥0.95 but h4 stays <0.5. The manipulation check passes; the behavioral
  bar still misses. This means the outcome-F attribution (cross-episode
  drift is *the* bottleneck) is incomplete — §6 names the leading
  candidate re-attribution (value geometry), grounded in this session's
  own measurements, not speculation.
- **Outcome C — mechanism not engaged.** drift stays <0.95 (still
  HIGH/INDETERMINATE band) regardless of h4. The intervention did not
  achieve its own mechanical goal; not an admissible test of the
  hypothesis either way. Routes to a stronger λ / a different candidate,
  not to reinterpreting the drift-bottleneck theory.
- **Outcome D — reference only.** Candidate (a)'s already-measured
  drift≈1.0 (trivial, context-free) and h4=1.0000 — the ceiling, not a new
  data point this wave produces.

---

## 4. Launch gate

**Primary, registered, unchanged mechanism: reuse `geo3_simulator.py`'s own
`launch_read` exactly, at the existing K=16 gate cell.** No new file, no
new formula — the drift→recovery mapping (§14.6/§14.13: cross-episode
drift enters as the value→key cross-alignment factor `c` in
`tilt_to_cos`, composed through the exact delta-rule recursion in
`delta_rule_exact`) is already committed, CPU-runnable, and already
validated as **directionally reliable and quantitatively accurate at
K=16** (§16.7: predicted 1.0000 vs measured 0.9767, error −0.023). This
design's Wave −1 re-runs it against whatever drift the anchoring
candidate's own 5,000-step probe measures (via a new
`keyanchor_drift_diagnostic.py`, a near-verbatim clone of
`geo3_drift_diagnostic.py` — same `sample_batch_fixed_entity`,
`measure_entity_rows`, `measure_drift`, unmodified — the statistic is a
pure function of whatever `bind()` returns, so nothing about it is
candidate-specific). **Gate: predicted K=16 h=4 `rec@0.9` ≥ 0.8 under the
mean mapping — unchanged from §14.6.**

**An honest, quantified caveat this session's own computation adds at
K=32 — a genuine attempt to close §16.7's named calibration gap, and a
genuine partial failure worth reporting rather than hiding.**

§16.7 already named the gap: the registered simulator tilts *only* the
value representation by the drift cosine against otherwise-idealized keys;
it carries no term for the independently measured value-Gram deviation,
which is **2.7× larger at K=32** (5.9274) than at K=16 (2.1948, §16.2/
§16.3). This session attempted the natural fix — compose an *independent*
value-side Gram perturbation on top of the existing tilt, calibrated by
bisection to the measured `value_gram_deviation_mean` — using the exact
`perturbed_orthonormal` bisection machinery already in `geo3_simulator.py`
(lines 95–113). **It failed in an informative way, twice, in sequence:**

1. **Reachability check (mirrors §4.5/BLOCK-1's own finding, applied to the
   value side for the first time here).** For K i.i.d. uniform unit vectors
   in `R^d`, `E‖G−I‖_F ≈ sqrt(K(K−1)/d)` — at `d_state=64` this ceiling is
   **1.9365 (K=16) and 3.9370 (K=32)**. Both measured
   `value_gram_deviation_mean` figures (2.1948, 5.9274) **exceed their own
   i.i.d. ceiling** — real trained value geometry has genuine
   shared-direction/anisotropic structure, not merely i.i.d. crosstalk,
   exactly mirroring §4.5's identical finding on the key side. A naive
   isotropic-noise perturbation therefore **cannot reach the target at
   all** (verified: bisection saturates at the ceiling, realized
   cross-alignment collapses to ~0.03–0.04 — the correction destroys the
   very drift signal it was supposed to sit on top of).
2. **Fix attempted (BLOCK-1's own shared-direction construction,
   transplanted to the value side) — still miscalibrates, in the opposite
   direction from §16.7's own miss.** Blending toward a single shared
   random direction (bisected on `ρ`, not noise scale) escapes the
   reachability ceiling, but the corrected prediction at K=32 now
   **undershoots**: ~0.001 at the measured drift (0.9037) and only ~0.19
   even at a **perfect** drift fix (`c=1.0`) — the true measured value
   (0.4368, §16.4) falls **between** the uncorrected simulator's overshoot
   (0.7734) and this correction's undershoot, closer to the overshoot side.
3. **Why the correction undershoots — a real, quantified, previously
   unreported fact about this architecture, checked directly against
   already-archived data (no new GPU spend):** across arm (iii-β)'s own
   trajectories (K=16+32 pooled, 50 checkpoint×seed points, files already
   in `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wave1/`),
   key-Gram and value-Gram deviation are **almost perfectly correlated**
   (Pearson r = **0.9933**) under *natural*, unconstrained training
   variation — contradicting the independence this session's naive
   correction assumed. **But** forcing key orthogonality via geo3 does
   **not** leave value geometry at its "naturally correlated" level: geo3's
   own final-checkpoint value-Gram deviation (5.9274 at K=32, 2.1948 at
   K=16) is **1.3–1.8× higher** than the SAME architecture's frozen/
   baseline arms at the identical K (arms i/iv/iii-β: 3.18–3.58 at K=32,
   1.35–1.67 at K=16 — extracted this session from the same archived
   files). Forcing keys clean appears to **release** value geometry to be
   *more* tangled, not neutral — the opposite of what the natural
   correlation would predict under intervention.

**Registered launch-gate decision, given the above (honest, not
gold-plated):** the K=16 gate (predicted `rec@0.9` ≥0.8, mean mapping,
tilt-only, unchanged mechanism) remains the **sole hard pass/fail
launch criterion** — it is cheap, and it is the one cell where the
existing, uncorrected simulator is already verified accurate (§16.7). At
K=32, **both** naive bounds are reported **non-gating**, explicitly
bracketing rather than pinpointing the expected outcome (tilt-only
upper bound; value-Gram-corrected lower bound), with the correlation
finding above stated as the reason **neither** bound should be read as a
tight prediction — the true relationship is *correlated but not
intervention-invariant*, which this session could quantify but not yet
model in closed form. This is the honest generalization of §16.7's own
closing sentence ("a single-scalar tilt is not a complete quantitative
account of the K=32 residual") — this design's attempt to complete it
surfaced *why* it resists completion, which is itself the useful output
of this section, not a decorative caveat.

**One new, real, zero-cost Wave 0 diagnostic this finding motivates
(no simulator changes required):** once KEY_ANCHORING's Wave 1 cells
produce fresh dumps, read `key_gram_deviation` (trivially ~0, non-
evidence) alongside `value_gram_deviation_mean` (already logged, zero
cost) at every checkpoint — if drift improves toward ≥0.95 while
value-Gram deviation *also* moves toward the frozen-arm range (3.2–3.6 at
K=32, 1.35–1.67 at K=16) rather than staying elevated near geo3's own
5.93/2.19, that is direct evidence the anchoring candidate is not just
stabilizing keys but *relieving* whatever pressure pushed geo3's value
geometry to be worse than baseline — a mechanistic bonus finding, read
for free from data the wave produces anyway.

---

## 5. Budget

**A discrepancy, flagged rather than silently resolved.** The task brief
that commissioned this design cited "the 300 GPU-h scale-program ceiling
(~90 spent after rung-2; rung-3 ~110 reserved)" as the budget context to
fit this wave under. That figure and those numbers do **not** appear
anywhere in `DELTANET_RD_EXACTNESS_DESIGN.md` or `EXPERIMENT_LOG.md` —
grepped directly, no match. A 300 GPU-h ceiling **does** exist, but in a
different, unrelated document (`matrix-thinking/SCALE_TRANSFER_DESIGN.md`
line 31, a separate rung-1/2/3 scale-transfer research thread with its own
Track A–D structure), and even there the specific "~90 spent... rung-3
~110 reserved" wording does not appear verbatim. This looks like a
cross-program mix-up rather than a real, load-bearing ceiling for this
wave — **per this project's own no-invented-numbers rule, this design does
not adopt it.** The actually-relevant, documented ceiling is
`DELTANET_RD_EXACTNESS_DESIGN.md` §6's own manifest: total required
baseline ~37.5 GPU-h, expected path ~52.5–55.5, all-conditionals-max ~90
on paper capped at a hard **80 GPU-h**, with **Wave F** (the fix-
demonstration wave F-geo-3 itself belongs to) carrying its own **15
nominal / ≤25 hard-band** reserve, cut-protected last (§6, §5.5). This is
the ceiling KEY_ANCHORING is scoped against below. **This mismatch is
itself flagged as the first item for the attack round (§7).**

**Realized F-geo-3 spend to date (the actual anchor, not the pre-registered
estimate):** ~1.67 GPU-h across all 9 runs — 6 mandatory K∈{16,32}×3
(`n_iter=12`) + 3 escalation K=32 (`n_iter=20`) — per-run average ≈0.19
GPU-h, **substantially cheaper** than the §6/§14.7 pre-registered anchor
(~0.58 GPU-h/run, Wave A's own measured continuity figure) (§16.10). This
design anchors its own per-run estimate on the **realized** number, widened
for new (untested) code paths by the same ×1.3–1.5 factor the harness
itself already applies to unmeasured arms (`run_deltanet_rd_exactness_
sweep.py`'s own `_PER_STEP_S_ANCHOR = 0.15 * 1.3` convention) —
≈0.24–0.28 GPU-h/run.

| Item | Cells | New runs | Est. GPU-h | Notes |
|---|---|---|---|---|
| Wave 0 (free) | i-strong re-analysis (§2.0); reachability/correlation checks (§4) | 0 | **0** | Done this session, zero GPU |
| Wave −1 (blocking smoke) | anchor-table init/blend/backward smoke (~6–8 short probes, NEG1_PROBE_STEPS-class); `keyanchor_drift_diagnostic.py` gating run, K∈{16,32}, 5,000-step probe each | ~8–10 short + 2 probes | ~0.6–1.0 | Mirrors geo3's own Wave −1 discipline (§14.6) exactly |
| Wave 1 — candidate (d), PRIMARY, mandatory | K∈{16,32}×3 seeds×20,000 steps | 6 | ~1.5–1.7 | The headline cells §3's bars are read against |
| Wave 1 — candidate (c), ablation, always-run (cheap) | K∈{16,32}×3 seeds, one `λ_anchor` (BLOCK-2 template) | 6 | ~1.5–1.7 | Comparison point, not the primary bet (§2.4) |
| Seed contingency (finding-5-style, one iteration) | +2 seeds, K=32 headline arm only | ≤2 | ~0.5–0.6 | Same add-seeds-not-steps discipline as §6/§14.10 |
| **Baseline total** | | ~22–24 | **~4.1–5.0** | |
| Candidate (b), CONDITIONAL fallback | K∈{16,32}×3 seeds, only if (d) fails via the diagnosed anchor-collapse mode (§2.3, §6) | ≤6 | ~1.5–1.7 | Reserved, not in the baseline sum |
| **All-conditionals-max** | | ~28–30 | **~5.6–6.7** | |

**Fit against the actual ceiling.** Realized F-geo-3-family spend to date
(~1.67 GPU-h, §16.10) plus this wave's worst case (~6.7 GPU-h) ≈ **8.4
GPU-h total** — comfortably inside Wave F's own 15-nominal/≤25-hard-band
reserve (§5.5/§6), and nowhere close to stressing
`DELTANET_RD_EXACTNESS_DESIGN.md`'s own 80 GPU-h hard cap. If the
(unverified) 300 GPU-h scale-program figure the task brief cited turns out
to be a real, intended, *shared* ceiling across programs once the
mix-up is resolved with the user, this wave's footprint is small enough
(≤7 GPU-h net-new) that it does not change any cut-order decision either
way — flagged so the attack round can settle which ceiling actually
applies without this design silently picking one.

---

## 6. Falsification map

| Candidate | Kills it | 
|---|---|
| (a) i-strong (reference) | N/A — already succeeded; its only open question is scope (closed key-only path bypasses the learned projection), not correctness |
| (d) primary (learned-λ blend) | Learned λ stays near 0 across all 3 seeds (SGD does not choose to lean on the anchor even though it is offered) **or** drift stays <0.95 even at the fixed-grid λ=0.9 cell (the anchor mechanism itself cannot escape geo3's own attractor, mirroring F-geo-1/2's saturation, §15) **or** the anchor table collapses (fails the new item-6 key-salvage check) |
| (b) EMA fallback | Anchor drifts because it chases a non-stationary target (`W_k` still training) — diagnosed directly by comparing the anchor table's own step-to-step movement against `W_k`'s gradient norm; if the anchor never stabilizes even after (d) is ruled out for a *different* reason, this is a distinct, informative failure |
| (c) soft `L_anchor` | Saturates like F-geo-1/2 — gain shrinks geometrically per hop (`~ε^h`), Gram/drift improvement small and λ-insensitive (the exact §15.4 signature, checked directly against this run's own λ sensitivity, none pre-registered beyond the single value per §2.4) |

**If ALL candidates fail — drift moves to ≥0.95 (item 5 passes) but h4
stays below 0.5 for every candidate (Outcome B, §3) — what gets
re-attributed, concretely, not speculatively.** This design's own §4
computation already supplies the leading candidate: geo3's value-Gram
deviation is measurably **worse** than baseline at both K (1.3–1.8×,
§4), and even a hypothetically **perfect** drift fix, once the
independently-measured value-Gram channel is included in the simulator,
predicts h4 capping near ~0.19 at K=32 — well short of 0.5. The
re-attribution this design pre-registers for that outcome: **cross-episode
key drift is real and necessary but not sufficient — value-side geometry
(mechanism (b) in the parent design's own §2 taxonomy, `DELTANET_RD_
EXACTNESS_DESIGN.md`) is a second, independently binding constraint, and
forcing key-orthogonality via any geo3-family mechanism may be actively
making it worse rather than leaving it neutral.** The named next step (not
designed here, flagged as the natural follow-on): a **value-anchoring** or
joint key+value orthogonalization/stabilization candidate — the value-side
analog of this entire design, gated on this outcome actually materializing,
not commissioned pre-emptively.

---

## 7. Attack surface

1. **Shortcut leakage — the anchor table could leak train-only identity
   information into held-out-entity evaluation.** The anchor table is
   indexed by `key_ids`; if its gradient/EMA update ever fires using an
   episode containing a held-out entity, that entity's row would be
   updated using information the "held-out" split is supposed to withhold.
   **Closed by:** gate every anchor update (gradient path for (d), EMA
   update for (b)) to fire **only** for entities in `pools.train_name_ids`
   — mirroring `geo3_drift_diagnostic.py`'s own `sample_batch_fixed_entity`
   convention (line 68/115: drawn from `pools.train_name_ids` exclusively)
   and the existing `C17_heldout_entities` diagnostic's own train/held-out
   discipline. **Required Wave −1 smoke:** assert the anchor table's
   gradient is **exactly zero** at every held-out-entity row after a full
   backward pass on a real batch containing both splits.
2. **Manipulation-check gaming — a collapsed anchor could inflate the
   drift statistic without genuine stability.** If many entities' anchor
   rows converge toward a small number of shared directions, the
   within-entity cross-episode-resample cosine (§14.6's statistic) could
   read as "low drift" while the anchor is simply too collapsed to
   distinguish entities from each other — a degenerate, useless solution
   that would still pass item 5. **Closed by:** the new item 6
   (§3) — key-side salvage ratio `σ_K/σ_1 ≥ 0.1` on `k_eff_items`,
   computed the identical way item 1 already computes it on the value
   side. A collapsed anchor fails this directly; passing both items
   together is required for admissibility.
3. **Seed cherry-picking.** Inherited unchanged from the parent design's
   own finding-5/§14.10 discipline: 3/3 admissible seeds required for the
   full evidentiary tier (§3), add-seeds-not-steps contingency (+2, one
   iteration only, §5), failing arms report at descriptive tier, never
   silently promoted. No new invention — the identical machinery closes
   this here too.
4. **Simulator overfit — trusting a closed-form or simulated prediction
   over the real behavioral bar.** §4 already surfaces this directly: this
   session's own attempt to correct the K=32 calibration miss swung from
   a +0.34 overshoot to a comparably-sized undershoot, and the reason
   (key/value Gram correlation that breaks under intervention) is
   **quantified but not yet modelable in closed form.** **Closed by:**
   neither bound is used as a gate at K=32 (§4) — only K=16's already-
   validated tilt-only prediction gates launch; the real, measured
   `rec@0.9` is the only quantity §3's bars are ever read against.
5. **K=16 regression masking — a candidate could "win" K=32 by quietly
   degrading the already-solved K=16 cell, and a write-up focused on the
   K=32 headline could miss it.** **Closed by:** §3's own numeric
   no-regression guard (K=16 h=4 within −0.02 of geo3's own 0.9767
   baseline) is a **hard admissibility item**, checked and reported at
   every headline comparison, not an implicit assumption — a candidate
   that clears K=32 while breaching this guard is not admitted as an
   improvement regardless of the K=32 number.

**A sixth item, added because this design's own §5 surfaced it rather than
an adversarial reviewer:** the budget-ceiling mismatch (§5) should be the
attack round's first read — if the 300 GPU-h/rung-2/rung-3 figures the
task brief cited turn out to trace to a real document or conversation this
design missed, the attack round should name it; if they are confirmed to
be a cross-program mix-up, that should be stated back to the user
explicitly rather than left ambiguous.

---

## Reproducibility pointers

- This design: `matrix-thinking/KEY_ANCHORING_DESIGN.md` (Rev 1).
- Builds on (read, not modified): `matrix-thinking/DELTANET_RD_EXACTNESS_
  DESIGN.md` §14–16.
- Harness to extend (read in full this session, not modified):
  `matrix-thinking/deltanet_rd/{model_rd.py, run_deltanet_rd_exactness_
  sweep.py, geo3_simulator.py, geo3_drift_diagnostic.py}`.
- Archived data read this session (not modified):
  `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wave1/
  w1_rdx_K32_arm{i,iv,iii-beta,i-strong}_s*.json`,
  `w1_rdx_K16_arm{i,iii-beta}_s*.json` — the §2.0 2×2 table and the §4
  key/value-Gram correlation are both read directly from these files.
- Scratch computation this session (CPU-only, throwaway venv, not part of
  the repo): reachability check, naive value-Gram-corrected simulator
  extension, and the Pearson-correlation readout, all reported in §4 with
  their exact numbers and construction.
- Next: independent attack round (this document is submitted un-audited,
  per the task brief) → build (the §2.2 `model_rd.py` diff, the
  `keyanchor_drift_diagnostic.py` clone, the `run_deltanet_rd_exactness_
  sweep.py` manifest/gate additions) → independent code audit → Wave −1 →
  Wave 1 (candidates (d) and (c) only, mandatory) → assess against §3's
  outcome frame → candidate (b) only if triggered by (d)'s specific
  diagnosed failure mode (§6).
