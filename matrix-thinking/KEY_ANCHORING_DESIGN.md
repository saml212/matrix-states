# KEY_ANCHORING_DESIGN — From "orthogonal but drifting" to "stable and orthogonal": a follow-on to F-geo-3

> **Rev 2 — 2026-07-03, post-attack-round-1.** The independent adversarial
> review of Rev 1 (`matrix-thinking/KEY_ANCHORING_ATTACK_R1.md`) returned
> **NEEDS-REV-2: 3 FATAL (two code-verified), 3 MAJOR, 4 MINOR** — the
> hypothesis and infrastructure reuse survived ("none of these require
> abandoning the hypothesis or the candidate ranking"), but the closures
> that make the headline *interpretable* were broken as written. Every
> finding is addressed in this revision under orchestrator-decided
> resolutions; the finding→change map is **§8**. The load-bearing changes:
> **all evidentiary drift/collapse metrics move to PRE-Newton-Schulz
> quantities** (F1 — Rev 1's item 6 measured a post-NS tensor that is
> tautologically full-rank; the rewritten checks were verified to have
> teeth by a negative-control computation this session, §3.1); the **"λ→1
> ⇒ i-strong" equivalence is RETRACTED and replaced with a computed
> ceiling** (F2 — the cited init crashes at `n=107 > d_state=64` by its own
> assert, and the Welch bound makes a globally-orthonormal 107-row table
> impossible; the anchor table now supplies **cross-episode stability
> only**, per-episode orthogonality remains geo3's job, and the λ=1
> post-NS drift ceiling was **computed this session: 0.9423 at K=32 —
> below the 0.95 LOW band, a fixed packing fact disclosed up front**);
> **numeric λ-trajectory bands are pre-registered** (F3 — interior
> anchoring [0.2, 0.8] is the only headline-eligible outcome;
> λ>0.95 = pin rediscovery, existence tier only); C17 held-out entities
> get an explicit eval-time bypass + a scope limit in the claims section
> (M1); a **CPU-only K=32 construction gate** joins the K=16 simulator
> gate (M2 — prototype already run this session, PASS); a pre-registered
> **first-checkpoint early-stop** ('value-geometry-bound') guards the M3
> risk with thresholds extracted from geo3's own archived trajectories
> (§3.4); minors m1–m4 fixed (misquoted ranges, run-count arithmetic, a
> numeric EMA-circularity threshold for candidate (b)).

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

**A load-bearing finding surfaced while writing Rev 1, not assumed
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
from 0.0 at step 2000 to **0.9988–0.9999 by step 20000** (per-seed finals
0.99939 / 0.99878 / 0.99994 — range corrected at Rev 2 per attack m1).
This is the existence proof this design's hypothesis needs, already
sitting in the archive — **with one Rev 2 caveat that now travels with it
everywhere it is cited (attack F2): i-strong changes TWO variables at
once relative to bare geo3 — context-free stability AND a pool collapsed
to 32 ≤ d_state entities — so its 1.0000 ceiling is NOT the reachable
target for any full-pool (107-entity) trainable anchor; §2.2 computes the
actually-reachable ceiling.** §2.0 below spells out the full,
now-quantified 2×2 this implies.

---

## 0. Reading list this design builds on

- `matrix-thinking/DELTANET_RD_EXACTNESS_DESIGN.md` §14 (F-geo-3's full
  mechanism spec — Newton-Schulz orthogonalization, the `bind()`/`readout()`
  threading, the substitute admission stack, the pre-registered drift
  bands), §15 (Wave F soft-arm results — `L_orth`/ZCA both saturate at 3–8%
  Gram cleanup, nowhere near i-strong's 0.000; the informative parallel this
  design's own attack surface (§7) leans on), §16 (F-geo-3 results — the
  K=16 hit / K=32 narrow miss / outcome-F attribution this design extends).
- `matrix-thinking/KEY_ANCHORING_ATTACK_R1.md` — attack round 1 on Rev 1
  of this document (3 FATAL / 3 MAJOR / 4 MINOR; response map §8).
- `matrix-thinking/deltanet_rd/model_rd.py` — read in full; exact insertion
  sites cited by line number throughout §2.
- `matrix-thinking/deltanet_rd/embed_arms.py` — `_qr_orthonormal_rows`'s
  `assert n <= d` (L81) and `restrict_pools_for_strong_pin`'s 32+32 pool
  restriction (L276, `_I_STRONG_POOL_SEED` L28) — the code facts behind
  attack F2, verified against source at Rev 2.
- `matrix-thinking/deltanet_rd/run_deltanet_rd_exactness_sweep.py` — the
  wave harness (`_spec`, `geo3_wave1_manifest`, `gate_geo3_drift`) this
  design's own manifest/gate functions are modeled on.
- `matrix-thinking/deltanet_rd/geo3_simulator.py` and
  `matrix-thinking/deltanet_rd/geo3_drift_diagnostic.py` — the committed
  launch-gate pattern (§14.6/§14.13) this design replicates, and (§4)
  extends with a documented, honestly-labeled calibration attempt.
- New numbers produced across the Rev 1 and Rev 2 sessions, **CPU-only**
  (no GPU, no repo files modified — scratch scripts importing
  `geo3_simulator.py` under a local throwaway venv): the i-strong
  held-out-hop extraction above; a reachability check on the
  i.i.d.-Gram-deviation ceiling; a naive value-Gram-corrected simulator
  extension and its calibration failure mode; a Pearson correlation
  between key- and value-Gram deviation across archived checkpoints
  (all Rev 1, §4); **and (Rev 2)** the frame-potential anchor-init
  construction with its coherence/conditioning/λ=1-drift-ceiling
  measurements (§2.2), the M2 CPU-gate prototype (§4), the item-6
  negative control (§3.1), and geo3's own first-checkpoint trajectory
  extraction for the M3 early-stop thresholds (§3.4). All are cited with
  their exact numbers and method — none invented, all reproducible from
  files already in this repo.

---

## 1. Hypothesis

Anchoring the key frame across episodes — holding the already-solved
per-episode orthonormality fixed via geo3's existing Newton-Schulz
machinery, and adding a mechanism that makes an entity's key
**consistent across independently-drawn episode contexts** — lifts K=32
h=4 composition from its measured 0.4368 (§16.4) past the pre-registered
≥0.5 bar, **because** cross-episode key drift (measured 0.9037 at the
trained checkpoint, §16.1/§16.6) is the named residual bottleneck, not a
restatement of the orthogonality problem F-geo-3 already solved.

**Scope note (Rev 2 — quantifying what "anchoring" can and cannot deliver
at this operating point).** With the full 107-name train pool and
`d_state=64`, a globally-orthonormal anchor table is impossible (Welch
bound, §2.2), so the maximum stability any full-pool anchoring mechanism
can deliver on the **written** (post-NS) key is bounded by the
subset-dependent NS correction — computed this session at **post-NS drift
≤ 0.9423 at K=32** (frame-potential init, λ=1; §2.2). The hypothesis
therefore stands or falls on whether moving drift from 0.9037 toward
≤0.9423 — closing at most ~40% of the distance to the 0.95 LOW band — is
enough to move h=4 from 0.4368 past 0.5 (a 0.06 miss). For calibration:
0.9423 is almost exactly K=16's bare-geo3 drift (0.9416), the drift level
at which K=16 cleared its own bar with a wide margin — suggestive but not
dispositive, since K=32 has twice the competing directions. This is the
honest, pre-registered version of the bet, stated before any data.

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
| **stable** (context-free key, same regardless of episode) | **i-strong**: dev 0.000, h=4 **1.0000**, h=7 **1.0000** (all 3 seeds, `w1_rdx_K32_armi-strong_s{0,1,2}_sp.json`) — **Rev 2 caveat: also pool-restricted to 32 ≤ d_state, see below** | **arms i/iv** (frozen embedding table, but `W_k` still learns to de-orthogonalize, §16 "Wave 1 ATTRIBUTION VERDICT"): dev 2.71–2.87, h=4 **0.0049–0.0114**, h=7 **0.0000** (`w1_rdx_K32_armi_s*.json`, `w1_rdx_K32_armiv_s*_rho0p225.json`) |
| **NOT stable** (episode-conditional key) | **geo3**: dev ~7×10⁻⁷, h=4 **0.4376** [0.3890–0.5040] (§16.2; the admissible n_iter=20 tier reads 0.4368, §16.3/§16.4), h=7 **0.0177** | **baseline / arm iii-beta**: dev **2.71–2.77** (per-seed finals 2.7108/2.7156/2.7672 — range corrected at Rev 2 per attack m2), h=4 **0.0073–0.0103**, h=7 **0.0000–0.0001** (`w1_rdx_K32_armiii-beta_s*.json`) |

Read across the table: **neither ingredient alone matters** (stable-only
and baseline are statistically the same, ~0.005–0.010 at h=4 — stability
without orthogonality buys nothing, confirming §16's own "raw embedding
geometry is causally irrelevant" verdict extends to this axis too);
**orthogonality alone buys most of the gap** (geo3's 44–56× jump over
baseline, §16); **both together is not a partial improvement over geo3, it
is a discontinuous jump to saturation** (0.44 → 1.0000, not 0.44 → 0.6 or
0.8). This is why the hypothesis is framed as an *interaction*, not a
second additive lever.

**The Rev 2 confound disclosure (attack F2, accepted in full):** the
2×2's "stable × orthogonal" cell is occupied by an arm that ALSO
collapsed the entity pool to 32+32 ≤ d_state
(`embed_arms.py::restrict_pools_for_strong_pin`, L276) — the only regime
in which 32 simultaneously-orthonormal, context-free keys even exist.
The table above therefore demonstrates the *interaction* but does NOT
license reading i-strong's 1.0000 as the reachable ceiling for a
full-pool trainable anchor: with 107 train names in R⁶⁴, the Welch bound
(`max|cos| ≥ sqrt((n−d)/(d(n−1))) = sqrt(43/6784) ≈ 0.0796`) guarantees
residual anchor coherence, and the per-episode NS correction therefore
stays genuinely subset-dependent at every λ. The reachable ceiling is
computed, not assumed, in §2.2 — and every outcome in §3 is read against
*that* ceiling, never against i-strong's.

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
§4/§4.4) — and Rev 2 sharpens the count to **three** structural
concessions: it bypasses the learned key path entirely; it works only for
the closed, hand-built entity pool it was constructed over (no
generalization to identities outside that pool); and (attack F2) that
pool was deliberately restricted to **32 ≤ d_state** entities, which is
the *only* reason zero cross-episode drift is geometrically possible at
all. It proves the mechanism is real and that *some* ceiling above geo3
exists; it does **not** set the reachable target for the full-pool
candidates below — §2.2's computed ceiling does.

**If it "failed" (it didn't):** it already didn't — this is the strongest
possible confirmation the interaction hypothesis has, within its
pool-restricted scope.

### 2.2 Candidate (d) — Anchor-first blend with a learned per-entity anchor table (PRIMARY, structural tier) — REWRITTEN, Rev 2 (attack F2/F3)

**Mechanism, restated with the Rev 2 division of labor (orchestrator
resolution of F2): the anchor table provides CROSS-EPISODE STABILITY
ONLY — each entity's anchor is a fixed direction; coherence between
different entities' anchors is whatever the packing allows. Per-episode
orthogonality remains geo3's job**: the per-episode gather of the ≤K
active (blended) keys goes through the existing Newton-Schulz
orthogonalization exactly as raw keys do now. Stability from the table,
orthogonality from geo3 — matching the verified 2×2 (§2.0), with no
appeal to an impossible globally-orthonormal table.

```
k_blend_raw[j] = normalize( (1 - λ_eff[j]) · k_eff_raw[j] + λ_eff[j] · A[key_ids[j]] )
k_eff_items    = geo3_orthogonalize_logged(k_blend_raw, n_iter=..., resid_tol=...)   # UNCHANGED call
```

`λ_eff[j] = λ · trained_mask[key_ids[j]]` — the per-entity mask is the M1
held-out bypass (§3.3). `λ` is registered as a **single learned scalar**
(`sigmoid(raw_param)`, init 0.5), with a **fixed grid** `λ∈{0.3,0.6,0.9}`
(no further tuning) as the pre-registered fallback diagnostic, run only
if the learned-λ trajectory lands in an ambiguous band under §3.2.

**Anchor-table init — specified exactly, replacing Rev 1's construction,
which is RETRACTED (attack F2, code-verified):** Rev 1 cited "the same
QR-orthonormal construction `strong_pin_table` already uses" — that path
is `embed_arms.py::_qr_orthonormal_rows`, which **asserts `n ≤ d` (L81)
and crashes at construction time** on the actual train pool
(`n_train_names = 107 > d_state = 64`, verified from
`w1_rdx_K32_armiii-beta_s0.json`'s own `pool_report`). The Rev 2 init:

- **Construction: frame-potential minimization.** Start from seeded
  random unit rows (107×64); minimize the frame potential
  `F(X) = Σ_{i≠j} (x_i·x_j)²` by projected gradient descent
  (`grad_i = 4·Σ_{j≠i}(x_i·x_j)x_j`, renormalize rows each step; 4,000
  steps, lr 0.05 — deterministic, seeded, CPU, one-time; frozen
  thereafter **as the init of the trainable table**, not as a frozen
  pin). Computed this session on a prototype draw:
  - **rms coherence 0.0796 — the Welch floor exactly** (frame-potential
    minimizers are tight frames; `σ_64/σ_1 = 1.0000` on the 107×64
    table); max |cos| **0.2832** (an equiangular tight frame at (107,64)
    may not exist, so the max sits above the Welch value; the rms is
    what the bound pins).
  - Random-unit init, for contrast: max |cos| **0.4789**, rms **0.1265**,
    `σ_64/σ_1 = 0.1377` — strictly worse on every measure.
- **Expected episode-level conditioning (32 draws from 107 anchors in
  R⁶⁴), computed on 512 sampled subsets:** pre-NS subset Gram deviation
  mean **2.508** / max **2.661** at K=32 (K=16: 1.231/1.425) — for scale,
  comparable to the learned baseline attractor's own 2.71–2.77, i.e. the
  anchor subset is NOT better-conditioned than raw learned keys in
  Frobenius terms; what it is, is **the same correction every time the
  same subset recurs**, which raw learned keys are not. Newton-Schulz
  converges on every sampled subset with **zero fallbacks at both
  n_iter=12 and n_iter=20** (max post-NS residual ~8×10⁻⁷) — the M2 CPU
  gate (§4, Gate 2) formalizes this as a launch requirement.
- **The computed λ=1 drift ceiling — the number that replaces the
  retracted "λ→1 ⇒ i-strong" claim.** Fixing one anchor row and
  resampling its K−1 co-drawn rows from the other 106 (8 entities × 32
  resamples, the §14.6 sampling spec, NS n_iter=20), the fixed entity's
  **post-NS** output drifts to pooled pairwise cosine:

  | init | K=16 mean / p10 | K=32 mean / p10 |
  |---|---|---|
  | frame-potential | **0.9745** / 0.9640 | **0.9423** / 0.9243 |
  | random-unit | 0.9410 / 0.9155 | **0.8926** / 0.8576 |
  | (reference: geo3's measured trained drift, §16.1) | 0.9416 | 0.9037 |
  | (reference: i-strong, pool ≤ d_state) | 1.0 trivially | 1.0 trivially |

  Three consequences, all load-bearing: (i) **candidate (d) at λ=1
  cannot reach i-strong's ceiling and cannot reach the 0.95 LOW band at
  K=32** — the best achievable post-NS drift with the full pool is
  ~0.9423, a fixed geometric fact (§1's scope note); (ii) **the init
  choice is mandatory, not aesthetic**: with a random-unit table, the
  λ=1 "stability endpoint" (0.8926 at K=32) is actually **worse than
  bare geo3's own trained drift (0.9037)** — a random-init anchor would
  be a drift *regression*; only the frame-potential init clears geo3's
  own level at both K; (iii) the outcome frame (§3.5) reads K=32 success
  against the ~0.9423 ceiling, and §3.1's "mechanically engaged" sanity
  band is pinned from these numbers.

**Insertion sites (`model_rd.py`):**
- Constructor: add `anchor_active: bool = False`, `anchor_lambda_mode:
  str = "learned"` (or `"fixed"`), `anchor_lambda_fixed: float | None =
  None` to `DeltaNetRDBlock.__init__` (alongside the existing
  `geo3_active`/`geo3_n_iter`/`geo3_resid_tol` args, lines 612–613);
  register the `nn.Embedding` (init loaded from the frozen
  frame-potential construction above) and (if learned-λ) a scalar
  `nn.Parameter`; register the boolean `anchor_trained_mask` vocab-length
  buffer (True at `pools.train_name_ids` rows — the M1 bypass). Assert
  `not (anchor_active and strong_pin_active)` (same mutual-exclusivity
  pattern as line 653–655) and `anchor_active implies geo3_active`
  (anchoring is defined as a modification to geo3's own write path, not
  a replacement for it).
- `bind()`'s existing `geo3_active` branch (lines 871–888): insert the
  masked blend between `k_eff_raw = _gather_at(...)` (line 877) and the
  `geo3_orthogonalize_logged(...)` call (line 878); additionally stash
  `self.anchor_last_k_blend_raw = k_blend_raw.detach()` — the
  **pre-NS side channel** §3.1's evidentiary drift measurement reads,
  following the exact side-channel pattern `geo3_last_resid` already
  established (lines 734–742; bind()'s 3/4-tuple return contract stays
  untouched).
- `readout()`/`forward()` (lines 926–990): **byte-identical**, no change —
  the query-side `a_slot` gather from `k_eff_items` (lines 943–953)
  already reads whatever `bind()` produced, regardless of how it got
  there.
- λ-trajectory logging (F3): `sigmoid(raw_param)` recorded into the
  result JSON's trajectory records at every existing `log_every` step —
  the full trajectory is a required field per run, not just the final
  value (§3.2).

**Param/FLOP cost.** `vocab_size_total × d_state ≈ 50,259 × 64 ≈ 3.22M` new
trainable params (~13MB fp32, mirroring the `strong_pin_table` comment's
own arithmetic at lines 713–714) — a real, disclosed **~25% increase** over
the reported 12,899,841-param baseline (§16 header). FLOPs: one extra
gather + blend + renormalize per `bind()` call, then the **same** single
Newton-Schulz pass geo3 already pays for — no second orthogonalization,
unlike (b).

**What failure would mean.** If the learned λ lands in the <0.05 band
(§3.2) across seeds — SGD declines the anchor even though it is offered —
that is the informative negative: the raw episode-conditional key carries
information the loss values enough to resist stabilizing (§7 names the
mechanistic reason this is plausible: geo3's value-Gram deviation is
1.3–1.8× **worse** than baseline at both K, §4 — value geometry may
already be under strain when keys are forced clean, and an anchor pulling
keys further from their value-co-adapted directions could aggravate it).
If λ lands in the >0.95 band, the result is **pin rediscovery** —
existence-tier only, per §3.2's bands, explicitly not this design's
novel deliverable.

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
The M1 `trained_mask` bypass applies identically (held-out rows are never
updated and never blended).

**Why ranked second, not first.** Two costs candidate (d) does not pay:
(1) a **second** Newton-Schulz pass every step; (2) a genuine
**circularity risk**: the anchor is an EMA of the model's *own*
per-episode output, which is a function of `W_k`/conv, which is *still
training* while the anchor accumulates — the anchor may chase a moving
target indefinitely. **Numeric stabilization criterion, pre-registered
(Rev 2, attack m4 — the parent design's "adjectives cannot gate a
mechanism claim" rule, §14.5/§14.13):** the anchor counts as
**stabilized** iff the relative Frobenius change of its train-row block
over a trailing 1,000-step window,
`‖A_t − A_{t−1000}‖_F / ‖A_{t−1000}‖_F`, falls below **1e-3** at or
before **step 10,000** (50% of the run) and stays below it for every
subsequent logged window. A run whose anchor never stabilizes by this
criterion reports at **descriptive tier only** — "still chasing" is a
measured verdict, not an adjective. Kept as the **registered fallback**
if (d) fails specifically via anchor-table degeneration (item 6 collapse,
§3.1) — a different diagnosis than "λ in the <0.05 band," and one a
gradient-free EMA anchor would not share.

**Insertion sites.** Same constructor location as (d); the buffer/EMA
logic adapts `ZCAWhiten`'s existing scaffold (lines 515–568). `bind()`'s
`geo3_active` branch gains the EMA update (a `torch.no_grad()` block
after the existing NS call, line 878) and a second
`geo3_orthogonalize_logged` call on the blended result; the pre-NS side
channel for §3.1's evidentiary drift is the input to that **final** NS
call.

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
made before any data exists.** §15 already ran this *exact* playbook for a
different geometric property and it failed in a specific, diagnosed way:
F-geo-1's `L_orth` saturated at 3–8% Gram cleanup regardless of
`λ_orth∈{0.1,1.0}` (a 10× weight change bought "essentially nothing," §15.4)
because **nothing in the primary task loss requires exact orthonormality
to reduce loss** — h=1 recovery is already near-ceiling without it (§15.1).
The identical structural argument applies to `L_anchor`: h=1 scores
directly against `v_eff` and is **provably drift-immune**
(geo3_simulator's own construction, §14.6), so nothing in `L_cos`/`L_nce`
requires cross-episode key stability either. A soft `L_anchor` term risks
the same saturation signature (proportional, real, but geometrically
shrinking gain per hop, `~ε^h`, §15.3) for the same underlying reason. It
is kept — run at a single, pre-registered `λ_anchor` value at
K∈{16,32}×3 seeds, BLOCK-2's own reduced-scope template
(`DELTANET_RD_EXACTNESS_DESIGN.md` §14.7) — **as a cheap, always-run
comparison point**, not as this wave's primary bet.

**Insertion site.** `run_deltanet_rd.py`'s loss composition (the same
site F-geo-1's `L_orth` addition used, §5.5); requires threading
`key_ids` + the stop-gradient anchor buffer through to the loss — no
change to `model_rd.py`'s `bind()`/`readout()` at all. For §3.1's
evidentiary drift measurement, candidate (c)'s "input to the final NS
call" is simply the raw gathered `k_eff_raw` (there is no blend) — the
regularizer must move THAT tensor's cross-episode stability to claim the
mechanism worked.

**Cost.** 6 runs (K∈{16,32}×3 seeds, one `λ_anchor`), same anchor as (b).

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

### 3.1 Substitute admission stack — §14.10 items 1–4 inherited verbatim, plus two Rev-2-rewritten items (attack F1)

Every KEY_ANCHORING candidate that touches `bind()` ((d), (b)) still
routes its final write-key through `geo3_orthogonalize_logged`, so the
**same tautology** §14.9 item 1/§14.10 found for geo3 applies unchanged:
key-Gram deviation ≡ ~0 and the patched `q_self ≡ k_eff_items` alignment
instrument ≡ 1.0, by construction, for every candidate, every seed —
both remain non-evidence, logged only. **Rev 2 adds (attack F1, accepted
in full): the post-NS `k_eff_items` tensor is likewise excluded from
carrying ANY new evidentiary check this design registers** — Newton-Schulz
forces `σ_K/σ_1 ≈ 1` on its output for *any* input, collapsed or not, so
a post-NS conditioning check is blind by the same mechanism (Rev 1's
item 6 walked into exactly the tautology its own §14.10 citation
describes; the attack's finding is accepted verbatim). **All evidentiary
drift/collapse metrics below are PRE-Newton-Schulz quantities.**

Items 1–4 of §14.10's substitute stack (value-side salvage ≥0.1;
Newton-Schulz converged without fallback at every checkpoint; finite loss
throughout; task-performance floor — final loss ≥50% improved, h=1 ≥0.5)
apply **unchanged**. The two new items:

- **Item 5 — THE MANIPULATION CHECK (REWRITTEN, Rev 2 — now pre-NS).**
  Cross-episode drift of **the input to the final Newton-Schulz call at
  `bind()`'s write site** — `k_blend_raw` for (d), the blended
  second-NS input for (b), the raw gathered `k_eff_raw` for (c) — read
  from the `anchor_last_k_blend_raw` side channel (§2.2), measured by
  the **identical statistic and sampling spec** registered at
  §14.6/§16.1 (mean pooled pairwise cosine, ≥8 train-pool entities, ≥32
  independently-resampled episode contexts each, trained final
  checkpoint; `geo3_drift_diagnostic.py`'s `measure_drift` machinery,
  repointed at the pre-NS tensor). **Bar: ≥ 0.95.** A run that clears
  h=4 ≥ 0.5 without clearing pre-NS drift ≥ 0.95 does **not** count as a
  test of this design's hypothesis, regardless of the h4 number — h4
  moved for some other reason (§6), and the write-up must say so, not
  claim confirmation. Both must clear together for a confirming
  instance. *(Known residual gameability, disclosed: at λ→1 the pre-NS
  blend is trivially the fixed anchor row and this check saturates at
  1.0 — item 6 and the §3.2 λ bands are what keep that from being
  laundered into a headline; §7 item 2 spells out the three-way
  closure.)*
- **Item 6 — anchor-table conditioning (REWRITTEN, Rev 2 — now the raw
  table, per attack F1's required fix).** Computed by SVD directly on
  the anchor table's **train-row block** (107×64, the raw `A` weight —
  never anything downstream of NS): **(6a)** `σ_64/σ_1 ≥ 0.1` (the house
  salvage-ratio convention), AND **(6b)** max pairwise row coherence
  `max_{i≠j}|cos(A_i,A_j)| ≤ 0.5`. Reference points, measured this
  session: the frame-potential init sits at `σ_64/σ_1 = 1.0000`,
  max |cos| = 0.2832 (both pass with wide margin); the Welch floor at
  (107,64) is 0.0796, so 6b's 0.5 cap leaves real training headroom
  while still catching collapse. **Negative control, RUN this session,
  not just written (per the standing run-the-negative-test rule):** a
  deliberately-collapsed table (all 107 rows near 2 shared directions —
  the attack's own scenario) measures `σ_64/σ_1 = 0.0147` (**FAILS 6a**)
  and max |cos| = 0.9263 (**FAILS 6b**), while the *old* Rev-1 post-NS
  check reads `σ_K/σ_1 = 1.0000` across 64 sampled episodes (**PASSES —
  confirmed blind**) and the pre-NS λ=1 drift on the same collapsed
  table reads 1.0000 (**item 5 alone is gameable — confirmed**). The
  rewritten pair has teeth exactly where the old check had none. For
  candidates (b)/(c), whose anchor is an EMA buffer rather than a
  trained table, items 6a/6b apply to that buffer's train-row block
  identically.

**Post-NS drift — retained as a REPORTED SANITY METRIC, explicitly
non-evidentiary (orchestrator resolution of F1).** The post-NS
`k_eff_items` drift (the §16.1 statistic, unchanged) is still logged at
every measurement point — it is the quantity the parent design's §14.5
bands and the outcome-F attribution were defined on, so continuity
requires it — but it carries no admission weight in this design.
Interpretation bands, pinned now from §2.2's computed ceilings (used
ONLY for outcome disambiguation, §3.5): at K=32, **"mechanically
engaged" = post-NS drift ≥ 0.92** (more than half the distance from
geo3's 0.9037 toward the 0.9423 λ=1 ceiling); at K=16, **≥ 0.96**
(geo3's 0.9416 toward the 0.9745 ceiling). Note the ceiling itself is
below the parent design's 0.95 LOW band at K=32 — a fixed packing fact
(§1 scope note), which is precisely why the band is defined relative to
the computed ceiling rather than to the unreachable 0.95.

### 3.2 λ-trajectory bands (NEW, Rev 2 — attack F3, orchestrator-pinned)

The full λ trajectory (`sigmoid(raw_param)` at every logged step) is a
**required** logged field for every learned-λ run. Final λ̄ = mean over
the final 1,000 logged steps, per seed. Bands, pinned before any data:

| Final λ̄ (per seed) | Label | Claim tier |
|---|---|---|
| **[0.2, 0.8]** | **learned interior anchoring** | the novel positive — **headline-eligible** |
| **> 0.95** | **pin rediscovery** | existence tier ONLY — SGD gradient-descends its way to (approximately) the fixed-frame regime; worth one sentence ("SGD finds the pin"), explicitly NOT a novel result and NOT support for §1's interaction hypothesis as framed |
| **< 0.05** | **anchoring not recruited** | negative — triggers the Outcome-B value-geometry follow-on consideration (§6) |
| (0.05, 0.2) or (0.8, 0.95) | **ambiguous** | no headline either way; the fixed-grid λ∈{0.3,0.6,0.9} diagnostic (§2.2) becomes the registered follow-up |

Arm-level label: **≥2/3 seeds in a band** assigns that band; seeds
spread across non-adjacent bands → ambiguous, no headline. The
headline-eligibility requirement composes with §3.1: **Outcome A (§3.5)
requires the interior band** — a bar-clearing run in the >0.95 band
reports as Outcome A′ (pin rediscovery), never as the interaction
headline.

### 3.3 C17 held-out entities (NEW, Rev 2 — attack M1, orchestrator-pinned)

**Eval-time behavior, specified in the design, not left to the build:**
held-out entities **bypass the blend** — `λ_eff = 0` wherever
`key_ids ∉ pools.train_name_ids` (the `anchor_trained_mask` buffer,
§2.2). C17 episodes are drawn entirely from the held-out pool, so under
the bypass a C17 episode routes through `bind()` **identically to bare
geo3** — C17 measures pure-geo3 behavior by construction. **Wave −1
smoke:** on an all-held-out batch with `anchor_active=True`, `bind()`'s
outputs must be **bit-identical** to the same weights with the anchor
disabled (achievable exactly: at λ_eff=0 the blend is
`normalize(1·k_eff_raw + 0) = k_eff_raw`, already unit-norm — the same
tensor).

**Reporting requirement:** `C17_heldout_entities` (already computed at
every checkpoint, zero marginal cost) is a **required, reported
diagnostic** — not a hard gate — for every anchoring cell, at the same
hop depths M2 covers, expected ≈ bare geo3's own C17 under the bypass.

**Scope limit (goes in every claims write-up, per the orchestrator
resolution):** the anchoring fix **as designed makes no claim of
held-out-entity gains.** Anchors exist only for trained entities; a
held-out entity gets bare-geo3 treatment. If Outcome A lands, the honest
claim is "anchoring lifts K=32 composition **for entities with trained
anchors**" — the generalization-to-new-entities question is a named
follow-on, not a deliverable of this wave.

### 3.4 First-checkpoint early-stop — 'value-geometry-bound' (NEW, Rev 2 — attack M3, orchestrator-pinned)

A **budget guard, not a claim.** The harness logs checkpoints every
2,000 steps; the rule fires at the **first** checkpoint (step 2,000),
per arm, per K, using geo3's own archived same-step trajectories as the
matched-stage reference (extracted this session from
`wgeo3_rdx_K32_armgeo3_s{0,1,2}_geo3n20.json` and
`wgeo3_rdx_K16_armgeo3_s{0,1,2}_geo3n12.json`):

| K | geo3 step-2000 `value_gram_deviation_mean` (3 seeds) | geo3 step-2000 h=4 `rec@0.9` (3 seeds) | KILL the arm at step 2000 iff |
|---|---|---|---|
| 32 | 3.3937 / 3.8543 / 3.8991 (max **3.90**) | 0.1011 / 0.1321 / 0.0834 (min **0.0834**) | value-Gram dev > **3.90** AND h4 < **0.0834** |
| 16 | 1.6596 / 1.6985 / 1.6701 (max **1.70**) | 0.7797 / 0.7123 / 0.7504 (min **0.7123**) | value-Gram dev > **1.70** AND h4 < **0.7123** |

Both conditions must hold (an AND, matching the orchestrator's spec):
value geometry degrading past geo3's own realized same-step band *while*
recovery underperforms geo3's own same-step trajectory. A killed arm
logs outcome **'value-geometry-bound'**, its spend stops at ~10% of the
run, and the §6 Outcome-B re-attribution machinery takes over. The
comparison is deliberately step-to-step matched (the same-stage
discipline attack M2 itself demanded), never early-checkpoint-to-final.

### 3.5 Outcome frame at K=32 (REVISED, Rev 2)

- **Outcome A — CONFIRMED (headline).** Pre-NS drift ≥0.95 (item 5) AND
  h4 ≥0.5, 3/3 admissible under items 1–6, AND final λ̄ in the interior
  band [0.2, 0.8] (§3.2) for ≥2/3 seeds. KEY_ANCHORING as an
  *interaction mechanism with the learned key path in the loop* is
  confirmed.
- **Outcome A′ — pin rediscovery (existence tier, NEW).** Bars clear but
  λ̄ > 0.95: SGD re-derives the fixed-frame regime. Confirms
  reachability-by-gradient-descent of a near-pin solution over the full
  107 pool (itself one publishable sentence, given §2.2's ceiling
  computation says even that regime tops out at post-NS drift ~0.9423);
  does NOT support §1's interaction hypothesis as framed, and is never
  written up as the headline.
- **Outcome B — INFORMATIVE NEGATIVE, re-attribution required.** Item 5
  passes (pre-NS drift ≥0.95) but h4 stays <0.5. **Disambiguation step,
  pre-registered (uses the non-evidentiary post-NS sanity bands, §3.1):**
  (B1) if post-NS drift stayed **below 0.92** (not mechanically
  engaged — the stabilized pre-NS input still produces subset-dependent
  NS output at near-geo3 levels), the failure is attributed to the
  **coherence-floor residual** (the F2 packing geometry: stabilizing the
  input cannot stabilize the output past the ~0.9423 ceiling, and the
  realized input→output stability transfer was worse than the ceiling
  computation's idealized estimate) — the follow-on is pool restriction
  or d_state escalation, NOT value geometry; (B2) if post-NS drift
  **reached ≥0.92** (engaged, near-ceiling) and h4 still missed, the
  outcome-F attribution is incomplete and the re-attribution goes to
  **value geometry** (§6) — the direction §4's own pre-existing evidence
  (geo3's 1.3–1.8× elevated value-Gram deviation; the corrected
  simulator's ~0.19 ceiling estimate) already points.
- **Outcome C — mechanism not engaged.** Item 5 fails (pre-NS drift
  <0.95) regardless of h4 — the intervention did not achieve its own
  mechanical goal; not an admissible test of the hypothesis either way.
  Routes to the fixed-grid λ diagnostic or a different candidate, not to
  reinterpreting the drift-bottleneck theory.
- **Outcome D — reference only.** Candidate (a)'s archived
  pool-restricted result — the (multi-concession) ceiling, not a new
  data point this wave produces.

---

## 4. Launch gates

**Gate 1 (unchanged from Rev 1): the registered K=16 simulator gate.**
Reuse `geo3_simulator.py::launch_read` exactly — the drift→recovery
mapping (§14.6/§14.13) is committed, CPU-runnable, and validated as
quantitatively accurate at K=16 (§16.7: predicted 1.0000 vs measured
0.9767, error −0.023). Wave −1 re-runs it against whatever drift the
anchoring candidate's own 5,000-step probe measures (via
`keyanchor_drift_diagnostic.py`, a near-verbatim clone of
`geo3_drift_diagnostic.py` — Rev 2 note: the clone measures **both** the
pre-NS side-channel tensor (§3.1's evidentiary quantity) and the post-NS
output (sanity + this gate's input)). **The launch-read's input stays the
post-NS measured drift** — that is the quantity §16.7's K=16 validation
was calibrated on, and swapping its input would silently invalidate the
one calibration this instrument has earned; this is a prediction
instrument, not an admission check, so F1's pre-NS rule does not bind it
(scope carve-out recorded in §8's F1 row). **Gate: predicted K=16 h=4
`rec@0.9` ≥ 0.8 under the mean mapping — unchanged from §14.6.**

**Honest limitation of Gate 1, conceded at Rev 2 (attack M2):** geo3's
own bare K=16 drift (0.9416) already maps to a passing prediction, so
Gate 1 cannot fail unless the anchor actively damages K=16 — it screens
for harm, not for benefit, and says nothing about K=32.

**Gate 2 (NEW, Rev 2 — attack M2, orchestrator-pinned): CPU-only
pre-spend K=32 construction gate.** Before any Wave 1 spend, verify on
CPU that the λ=1 anchor-blend construction achieves i-strong-level
**per-episode** conditioning through the production NS path, over the
actual pool geometry (32 drawn from 107 in R⁶⁴):

- **Procedure:** on the frozen registered anchor init (the
  frame-potential table, §2.2), sample **≥512** random 32-subsets (and
  16-subsets), run the production `newton_schulz` at the production
  iteration tier (`n_iter=20` for K=32, 12 for K=16 — geo3's own
  realized tiers, §16.3), and apply the §14.10-analog admission read:
  **100% of subsets must converge to residual ≤ `resid_tol=1e-2` with
  zero fallbacks** (item-2 semantics), with the pre-NS subset Gram-dev
  band reported alongside. **If the construction cannot clear admission
  on CPU, the wave does not launch.**
- **Prototype result (this session, on a prototype seed of the init —
  the gate re-runs at build time on the registered frozen table):**
  **PASS** — zero fallbacks in 512/512 subsets at both K and both
  n_iter tiers; max post-NS residual ~8×10⁻⁷; pre-NS subset Gram dev
  2.508 mean / 2.661 max at K=32 (1.231/1.425 at K=16). The gate is
  expected to pass at build time; it exists so that a bad registered
  seed or a build-time construction bug cannot silently reach GPU spend.
- **Adaptation from the orchestrator's phrasing, reported per
  instruction:** the spec mentioned running this "on an archived geo3
  K=32 checkpoint." Two facts make the checkpoint-free form above the
  correct implementation: (i) at λ=1 the blend is
  `normalize(A[key_ids])` — a pure function of the anchor table,
  **checkpoint-independent by construction** — so no trained weights
  enter the λ=1 computation at all; (ii) the local archive carries
  result JSONs only, not model weight checkpoints
  (`experiment-runs/.../wavegeo3/` contents verified — 10 JSONs, no
  weight files), so a checkpoint-loading variant is not locally runnable
  anyway. Nothing evidentiary is lost: interior-λ blends mix the anchor
  subset with geo3's raw keys, and both endpoints' NS-convergence are
  covered (anchors by this gate, raw keys by geo3's own completed wave);
  intermediate blends are covered by Wave −1's standard smoke on
  realistic probe keys.

**A reproduction caveat on the simulator at K=32, surfaced this session
and reported for completeness:** re-running the registered
`simulate_recovery` on CPU at the recorded §16.1 inputs reproduces the
K=16 prediction (0.998 vs the recorded 1.0000) but **not** the K=32
prediction (0.0664 vs the recorded 0.7734, same c=0.9037, same
`gram_resid=1e-2`, seed 0). The h=4 `rec@0.9` statistic at K=32 sits on
a steep cliff in the drift cosine (measured this session on one
CPU/seed: c=0.9037 → 0.066, c=0.9243 → 0.305, c=0.9423 → 0.746), so
platform/RNG-stream differences move it enormously at K=32 while K=16
sits on a plateau. This is one more independent reason K=32 simulator
reads are **non-gating** here (Gate 2, a construction check with no
simulator in the loop, is the K=32 go/no-go) — and a caution against
ever citing a single-seed K=32 simulator number without its platform.

**The Rev 1 value-Gram calibration investigation (retained in full — it
now feeds §3.5's Outcome-B disambiguation and §3.4's early-stop).**
§16.7 named the gap: the registered simulator tilts *only* the value
representation by the drift cosine against otherwise-idealized keys; it
carries no term for the independently measured value-Gram deviation,
which is **2.7× larger at K=32** (5.9274) than at K=16 (2.1948, §16.2/
§16.3). The Rev 1 session attempted the natural fix — an independent
value-side Gram perturbation calibrated by bisection to the measured
`value_gram_deviation_mean` — and it failed in an informative way,
twice:

1. **Reachability (mirrors §4.5/BLOCK-1, applied to the value side for
   the first time).** For K i.i.d. uniform unit vectors in `R^d`,
   `E‖G−I‖_F ≈ sqrt(K(K−1)/d)` — at `d_state=64` the ceiling is
   **1.9365 (K=16) / 3.9370 (K=32)**. Both measured value-Gram figures
   (2.1948, 5.9274) **exceed their own i.i.d. ceiling** — real trained
   value geometry has genuine shared-direction structure. A naive
   isotropic perturbation cannot reach the target at all (verified:
   bisection saturates at the ceiling and destroys the drift signal it
   was meant to sit on top of — realized cross-alignment ~0.03).
2. **The shared-direction fix (BLOCK-1's own construction, transplanted)
   reaches the target but undershoots the measured outcome:** corrected
   K=32 predictions land at ~0.001 at the measured drift and only
   **~0.19 even at a perfect (`c=1.0`) drift fix** — the true measured
   value (0.4368) falls *between* the tilt-only overshoot and this
   correction's undershoot.
3. **Why it undershoots — a real, quantified fact:** across arm
   (iii-β)'s archived trajectories (50 checkpoint×seed points, K=16+32
   pooled), key-Gram and value-Gram deviation are almost perfectly
   correlated (Pearson r = **0.9933**) under natural training
   variation — but forcing keys clean via geo3 does not leave value
   geometry at the correlated level: geo3's final value-Gram deviation
   (5.9274 / 2.1948 at K=32/16) is **1.3–1.8× higher** than the same
   architecture's frozen/baseline arms at identical K (3.18–3.58 /
   1.35–1.67). Forcing keys clean appears to **release** value geometry
   to be *more* tangled. The correlation is real but not
   intervention-invariant — which is why neither naive bound gates
   anything, and why §3.4's early-stop and §3.5's B2 branch both watch
   the value-Gram channel explicitly.

**Zero-cost Wave 0/Wave 1 diagnostic (retained):** read
`value_gram_deviation_mean` (already logged every checkpoint) alongside
drift — if drift improves while value-Gram moves toward the frozen-arm
range (3.2–3.6 at K=32, 1.35–1.67 at K=16) rather than staying near
geo3's elevated 5.93/2.19, the anchoring candidate is not just
stabilizing keys but relieving the value-side strain — a mechanistic
bonus finding, free.

---

## 5. Budget

**Binding ceilings (Rev 2, orchestrator-pinned).** The exactness
program's own **80 GPU-h cap** (`DELTANET_RD_EXACTNESS_DESIGN.md` §6),
and for this wave specifically **≤10 GPU-h nominal / ≤15 with reserve**.
The Rev 1 provenance flag on the task brief's "300 GPU-h scale-program
ceiling" stands resolved: that figure traces to the unrelated
`matrix-thinking/SCALE_TRANSFER_DESIGN.md` (line 31), a conclusion the
attack round independently verified by grep ("this specific self-flagged
concern clears," `KEY_ANCHORING_ATTACK_R1.md` Verdict).

**Per-run anchor.** Realized F-geo-3 spend: ~1.67 GPU-h across all 9 runs
(≈0.19 GPU-h/run, §16.10), substantially under the §6 pre-registered
anchor (~0.58). This design widens the realized figure by the harness's
own ×1.3–1.5 unmeasured-code-path convention
(`run_deltanet_rd_exactness_sweep.py::_PER_STEP_S_ANCHOR = 0.15 * 1.3`)
→ ≈0.24–0.28 GPU-h/run.

| Item | Cells | New runs | Est. GPU-h | Notes |
|---|---|---|---|---|
| Wave 0 (free) | i-strong re-analysis (§2.0); reachability/correlation checks (§4); **Rev 2: anchor-init construction + λ=1 ceiling + item-6 negative control + Gate-2 prototype (§2.2/§3.1/§4)** | 0 | **0** | All done on CPU across the two design sessions |
| Gate 2 (blocking, CPU) | ≥512-subset NS admission check on the frozen registered init, K∈{16,32} | 0 | **0** | §4; prototype PASS this session |
| Wave −1 (blocking smoke) | **8 short smoke probes** (anchor init/blend/backward, held-out-bypass bit-identity, λ-logging, item-5/6 instrument checks — NEG1_PROBE_STEPS-class) **+ 2 drift-diagnostic probe runs** (K∈{16,32}, 5,000 steps each) = **10 runs** (m3 fix: prose and column now agree) | 10 | ~0.7–1.0 | Mirrors geo3's own Wave −1 discipline (§14.6); Gate 1 launch-read reads the K=16 probe |
| Wave 1 — candidate (d), PRIMARY, mandatory | K∈{16,32}×3 seeds×20,000 steps, learned λ | 6 | ~1.5–1.7 | Headline cells; §3.4 early-stop armed |
| Wave 1 — candidate (c), ablation, always-run | K∈{16,32}×3 seeds, one `λ_anchor` (BLOCK-2 template) | 6 | ~1.5–1.7 | Comparison point (§2.4); early-stop armed |
| Fixed-grid λ diagnostic (conditional) | λ∈{0.3,0.6,0.9}, K=32 only ×1 seed each, fired only on an ambiguous §3.2 band | ≤3 | ~0.8 | Registered follow-up, not tuning — grid fixed now |
| Seed contingency (finding-5-style, one iteration) | +2 seeds, K=32 headline arm only | ≤2 | ~0.5–0.6 | Same add-seeds-not-steps discipline as §6/§14.10 |
| **Baseline total (mandatory)** | | 22 | **~3.7–4.4** | |
| **With both conditionals fired** | | ≤27 | **~5.0–5.8** | |
| Candidate (b), CONDITIONAL fallback | K∈{16,32}×3 seeds, only on (d)'s diagnosed anchor-collapse failure (§2.3, §6) | ≤6 | ~1.7–2.0 | Reserved, outside the baseline |
| **All-conditionals-max** | | ≤33 | **~6.7–7.8** | |

**Fit.** Worst case ~7.8 GPU-h — under the orchestrator's ≤10 nominal /
≤15 reserve wave ceiling with margin, and (combined with F-geo-3's
realized ~1.67) nowhere near stressing the program's 80 GPU-h cap or
Wave F's own 15/≤25 reserve (§5.5/§6 of the parent design). The §3.4
early-stop can only reduce these figures (a killed arm stops at ~10% of
its run cost).

---

## 6. Falsification map

| Candidate | Kills it |
|---|---|
| (a) i-strong (reference) | N/A — already succeeded within its pool-restricted, three-concession scope (§2.1); it sets an existence proof, not this wave's target |
| (d) primary (learned-λ blend) | Final λ̄ in the <0.05 band for ≥2/3 seeds (anchoring not recruited, §3.2) **or** pre-NS drift <0.95 even at the fixed-grid λ=0.9 cell (item-5 fail — the blend cannot stabilize its own input, Outcome C) **or** anchor-table collapse (items 6a/6b fail — routes to candidate (b), the gradient-free anchor) **or** the §3.4 early-stop fires ('value-geometry-bound') |
| (b) EMA fallback | Anchor never stabilizes under the pinned numeric criterion (trailing-1,000-step relative Frobenius change ≥1e-3 past step 10,000, §2.3) — measured, not adjectival; distinct from (d)'s failure modes by construction (no gradient into the anchor) |
| (c) soft `L_anchor` | Saturates like F-geo-1/2 — pre-NS drift improvement small, gain shrinking geometrically per hop (`~ε^h`), the exact §15.3/§15.4 signature |

**If ALL candidates fail with the manipulation check passing — pre-NS
drift ≥0.95 (item 5) but h4 <0.5 everywhere (Outcome B) — what gets
re-attributed, via §3.5's pre-registered disambiguation:**

- **B1 (post-NS sanity <0.92 — not mechanically engaged):** the failure
  re-attributes to the **coherence-floor residual** — the F2 packing
  geometry means input stability does not purchase output stability
  past the computed ~0.9423 ceiling, and the realized input→output
  stability transfer was worse than the ceiling computation's idealized
  version. Named follow-on: a **pool-restricted** anchor cell (≤64
  train entities — deliberately trading the full-pool scope for
  geometry, with the i-strong precedent as the limit case) or a
  `d_state=128` rider (the parent design's own §9 K=d/2 boundary
  question, where 107 < 128 would dissolve the packing obstruction
  entirely). Not run inside this wave's budget.
- **B2 (post-NS ≥0.92 — engaged, near-ceiling, still missing):** the
  outcome-F attribution is incomplete — **value-side geometry is a
  second, independently binding constraint** (mechanism (b) in the
  parent design's §2 taxonomy), exactly the direction §4's pre-existing
  evidence points (geo3's 1.3–1.8× elevated value-Gram; the corrected
  simulator's ~0.19 ceiling; the intervention-broken key/value
  correlation). Named follow-on: a **value-anchoring / joint key+value
  stabilization** design — the value-side analog of this entire
  document — gated on B2 actually materializing, not commissioned
  pre-emptively.

---

## 7. Attack surface

1. **Shortcut leakage — the anchor table could leak train-only identity
   information into held-out-entity evaluation.** **Closed by:** the
   `anchor_trained_mask` bypass (§2.2/§3.3) — anchor updates (gradient
   for (d), EMA for (b)) and the blend itself fire **only** for
   `pools.train_name_ids` entities, mirroring
   `geo3_drift_diagnostic.py`'s own train-pool-only convention.
   **Required Wave −1 smokes:** (i) anchor-table gradient exactly zero
   at every held-out row after a mixed-split backward; (ii) the §3.3
   bit-identity check on an all-held-out batch.
2. **Manipulation-check gaming — a collapsed or trivialized anchor could
   inflate the drift statistic without genuine, useful stability.**
   **Closed by a three-part co-firing requirement, each part now
   verified to have teeth (Rev 2, attack F1):** item 5 (pre-NS drift
   ≥0.95) is trivially satisfiable at λ→1 and by a collapsed table
   (measured: collapsed-table pre-NS drift ≡ 1.0000) — alone it proves
   nothing; item 6 (raw-table `σ_64/σ_1 ≥0.1` AND max|cos| ≤0.5) catches
   collapse directly (negative control: 0.0147 / 0.9263 — both FAIL on
   the planted degenerate table, while the old post-NS check read a
   blind 1.0000); the §3.2 λ bands cap the λ→1 route at existence tier
   regardless of every other number. A headline requires all three
   simultaneously; no single instrument is load-bearing alone.
3. **Seed cherry-picking.** Inherited unchanged from the parent design's
   finding-5/§14.10 discipline: 3/3 admissible for the full tier,
   add-seeds-not-steps contingency (+2, one iteration), failing arms at
   descriptive tier. The §3.2 band assignment uses the same ≥2/3
   convention, pinned before data.
4. **Simulator overfit — trusting a simulated prediction over the
   behavioral bar.** **Closed by:** no simulator number gates K=32
   (Gate 2 is a construction check with no simulator in the loop); the
   K=32 platform-sensitivity finding (§4 — CPU rerun does not reproduce
   the recorded K=32 prediction while K=16 reproduces cleanly) is
   disclosed as an instrument limitation; §3's bars are read only
   against measured `rec@0.9`.
5. **K=16 regression masking.** **Closed by:** §3's numeric
   no-regression guard (K=16 h=4 within −0.02 of geo3's 0.9767) is a
   hard admissibility item at every headline comparison — a candidate
   clearing K=32 while breaching it is not admitted as an improvement.
6. **Budget-ceiling provenance (Rev 1 self-flag).** Resolved: the attack
   round's own grep confirmed the 300 GPU-h figure belongs to
   `SCALE_TRANSFER_DESIGN.md`, and the orchestrator has pinned this
   wave's ceilings (≤10 nominal / ≤15 reserve under the program's 80
   cap, §5). Closed.

---

## 8. Rev 2 — attack-round-1 responses (finding → change map)

The independent adversarial review of Rev 1
(`KEY_ANCHORING_ATTACK_R1.md`, 2026-07-03) returned **NEEDS-REV-2: 3
FATAL, 3 MAJOR, 4 MINOR**. Disposition: **every finding accepted;
resolutions for the contested points were decided by the orchestrator
and implemented as directed**, with two scoped technical adaptations
declared inline (F1's launch-read carve-out; M2's checkpoint-free gate
form), each with its reason stated where it is applied.

| # | Finding (condensed) | Change made | Where |
|---|---|---|---|
| F1 | FATAL — Rev 1's item 6 measured post-NS `k_eff_items`, which NS forces to `σ_K/σ_1 ≈ 1` for ANY input — tautologically blind to anchor collapse (the same instrument-tautology class as the parent design's own attack F3) | All evidentiary drift/collapse metrics moved to **pre-NS** quantities per orchestrator resolution: item 5 = cross-episode drift of the input to the final NS call (side-channel `anchor_last_k_blend_raw`, bar ≥0.95); item 6 = raw anchor-table conditioning (`σ_64/σ_1 ≥ 0.1` AND max pairwise \|cos\| ≤ 0.5 on the train-row block). Post-NS drift retained as **reported sanity, explicitly non-evidentiary**, with pinned interpretation bands (engaged ≥0.92 at K=32 / ≥0.96 at K=16, derived from the §2.2 computed ceilings) used only for Outcome-B disambiguation. **Negative control RUN, not just written:** a planted 2-direction collapsed table FAILS both new checks (0.0147; 0.9263) while the old post-NS check reads a blind 1.0000 and the pre-NS λ=1 drift reads a gameable 1.0000 — confirming both the old check's blindness and why items 5+6 must co-fire. **Declared scope carve-out:** the K=16 `launch_read` keeps its post-NS drift input — §16.7's validation of that instrument was calibrated on post-NS inputs, and it is a prediction instrument, not an admission check | §3.1, §3.5, §4 (Gate 1 note), §7 item 2 |
| F2 | FATAL, code-verified — the cited init (`_qr_orthonormal_rows`) asserts `n ≤ d` and crashes at 107 > 64; the Welch bound (max\|cos\| ≥ 0.0796 at n=107, d=64) makes a globally-orthonormal table impossible; "λ→1 ⇒ i-strong" is false — i-strong changes two variables (stability AND pool ≤ d_state) and the reachable λ=1 ceiling was never computed | Per orchestrator resolution: **no globally-orthonormal table attempted** — the anchor supplies cross-episode stability only; per-episode orthogonality stays with geo3's NS (mechanism restated, §2.2). Init specified and computed: **frame-potential-minimized 107×64** (rms coherence = Welch floor 0.0796 exactly, max 0.2832, tight frame `σ_64/σ_1 = 1.0000`); random-unit rejected with measurements (max 0.4789 — and its λ=1 K=32 drift ceiling, 0.8926, is WORSE than bare geo3's 0.9037, making the init choice mandatory). Episode conditioning computed (K=32 subset Gram dev 2.508 mean / 2.661 max; NS zero fallbacks at n_iter 12 and 20 over 512 subsets). **"λ→1 ⇒ i-strong" RETRACTED**, replaced by the computed λ=1 post-NS drift ceiling: **0.9423 / 0.9243 (mean/p10) at K=32**, 0.9745/0.9640 at K=16 — below the 0.95 LOW band at K=32, disclosed as a fixed packing fact in §1's new scope note; §2.0's 2×2 gains the pool-size confound disclosure; every outcome now reads against the computed ceiling, never i-strong's 1.0000 | §1 (scope note), §2.0, §2.1, §2.2, §3.1 (bands), §3.5 (B1) |
| F3 | FATAL — no pre-registered criterion separates "genuine learned interior anchoring" from "λ saturates toward the trivial endpoint"; both reported identically as Outcome A | Orchestrator-pinned λ bands registered (§3.2): final λ̄ (mean over final 1,000 steps, per seed; full trajectory logged per run, required field): **[0.2, 0.8] = learned interior anchoring (headline-eligible)**; **>0.95 = pin rediscovery (existence tier only, one sentence, never the headline)**; **<0.05 = anchoring not recruited (negative → Outcome-B follow-on)**; boundary intervals (0.05–0.2, 0.8–0.95) ambiguous, no headline, fixed-grid λ∈{0.3,0.6,0.9} diagnostic as the registered follow-up. Arm label = ≥2/3 seeds. Outcome A now requires the interior band; **Outcome A′ (pin rediscovery)** added to the outcome frame | §3.2, §3.5, §2.2, §5 (conditional row) |
| M1 | MAJOR — C17 held-out entities: anchor rows for held-out entities are never trained; a λ>0 blend on a C17 batch would inject untrained, identity-arbitrary directions; M2/M3 bars never see the C17 pool | Orchestrator-pinned: **held-out entities bypass the blend** (`λ_eff = 0` via the `anchor_trained_mask` buffer) → C17 measures pure-geo3 behavior by construction, with a bit-identity Wave −1 smoke; `C17_heldout_entities` promoted to a **required, reported diagnostic** (not a hard gate) for every anchoring cell; **scope limit registered in the claims section: the fix as designed claims NO held-out-entity gains** — anchored-entity claims only, generalization-to-new-entities is a named follow-on | §3.3, §2.2 (mask), §7 item 1 |
| M2 | MAJOR — the only hard launch gate (K=16 simulator) passes almost trivially (bare geo3's 0.9416 drift already predicts 1.0); K=32, the actual headline bet, had no go/no-go read | K=16 gate kept, with its screens-for-harm-not-benefit limitation conceded in text; **Gate 2 added per orchestrator resolution: CPU-only pre-spend K=32 construction gate** — ≥512 sampled 32-subsets of the frozen anchor init through production NS at n_iter=20 must clear §14.10-item-2 admission semantics (100% resid ≤ 1e-2, zero fallbacks) or the wave does not launch. **Prototype run this session: PASS** (zero fallbacks 512/512, max resid ~8×10⁻⁷). Declared adaptation: implemented checkpoint-free — at λ=1 the blend is a pure function of the table (checkpoint-independent), and the local archive holds result JSONs, not weight checkpoints. Bonus disclosure feeding the same finding: the K=32 simulator prediction is platform/RNG-sensitive (CPU rerun 0.0664 vs recorded 0.7734 at identical inputs; K=16 reproduces cleanly) — one more reason no simulator number gates K=32 | §4 (Gates 1–2), §7 item 4 |
| M3 | MAJOR — the design's own value-Gram evidence (~0.19 corrected-simulator ceiling; geo3's 1.3–1.8× elevated value-Gram) suggests the bar may be uncapturable regardless, with no mid-run catch | Orchestrator-pinned **first-checkpoint early-stop** registered (§3.4): at step 2,000 (the harness's existing cadence), kill the arm and log **'value-geometry-bound'** iff value-Gram dev exceeds geo3's own realized same-step band max (K=32: >3.90; K=16: >1.70) AND h4 is below geo3's own same-step min (K=32: <0.0834; K=16: <0.7123) — thresholds extracted this session from the archived geo3 trajectories, matched step-to-step (the same-stage discipline M2 itself demanded). A budget guard, not a claim; feeds §3.5's Outcome-B machinery | §3.4, §5 (early-stop note), §6 |
| m1 | MINOR — i-strong h=21 range misquoted (0.9987–0.9994; actual range excludes seed 2's value) | Corrected to **0.9988–0.9999** with per-seed finals quoted (0.99939/0.99878/0.99994) | header block |
| m2 | MINOR — §2.0's iii-beta key-Gram range (2.72–2.77) doesn't match its own cited files | Corrected to **2.71–2.77** with per-seed finals quoted (2.7108/2.7156/2.7672); the geo3 cell's h=4 figure now also cites both tiers explicitly (0.4376 §16.2 / 0.4368 admissible §16.3) to keep every §2.0 number traceable to its exact source | §2.0 |
| m3 | MINOR — Wave −1 prose (~6–8 probes) vs column (~8–10 + 2) mismatch; baseline sum drift | Pinned: **8 short smoke probes + 2 drift-diagnostic probe runs = 10 Wave −1 runs**; table recomputed — mandatory baseline 22 runs / ~3.7–4.4 GPU-h, all-conditionals ≤33 / ~6.7–7.8 GPU-h — internally consistent and within the orchestrator's ≤10/≤15 wave ceiling | §5 |
| m4 | MINOR — candidate (b)'s EMA-circularity risk had no numeric threshold ("adjectives cannot gate a mechanism claim") | Numeric criterion pinned (ranking unchanged, per the finding's own or-clause): anchor **stabilized** iff trailing-1,000-step relative Frobenius change of the train-row block < **1e-3** by step **10,000** and sustained; otherwise 'still chasing' → descriptive tier only | §2.3, §6 |

---

## Reproducibility pointers

- This design: `matrix-thinking/KEY_ANCHORING_DESIGN.md` (**Rev 2**,
  2026-07-03 — Rev 1 same day; attack-round-1 map is §8).
- Attack round: `matrix-thinking/KEY_ANCHORING_ATTACK_R1.md`.
- Builds on (read, not modified): `matrix-thinking/DELTANET_RD_EXACTNESS_
  DESIGN.md` §14–16.
- Harness to extend (read in full, not modified):
  `matrix-thinking/deltanet_rd/{model_rd.py, embed_arms.py,
  run_deltanet_rd_exactness_sweep.py, geo3_simulator.py,
  geo3_drift_diagnostic.py}`.
- Archived data read across both design sessions (not modified):
  `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wave1/
  w1_rdx_K32_arm{i,iv,iii-beta,i-strong}_s*.json`,
  `w1_rdx_K16_arm{i,iii-beta}_s*.json`,
  `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wavegeo3/
  wgeo3_rdx_K{16,32}_armgeo3_s*_geo3n{12,20}.json` — §2.0's 2×2, §3.4's
  early-stop thresholds, and §4's correlation readout all come directly
  from these files.
- Scratch computations (CPU-only, throwaway venv, not part of the repo),
  Rev 1: value-Gram reachability, corrected-simulator calibration,
  key/value-Gram correlation (§4). Rev 2: frame-potential init
  construction + coherence/conditioning/λ=1-ceiling measurements (§2.2),
  Gate-2 prototype (§4), item-6 negative control (§3.1), geo3
  first-checkpoint trajectory extraction (§3.4) — every number quoted
  with its construction, reproducible from repo files +
  `geo3_simulator.py`'s own functions.
- Next: verification round on this Rev 2 (confirm each §8 change closes
  its finding — the parent design's own two-round discipline) → build
  (the §2.2 `model_rd.py` diff incl. the pre-NS side channel and
  `anchor_trained_mask`, the frozen frame-potential init with a
  registered seed, `keyanchor_drift_diagnostic.py`, manifest/gate
  additions to `run_deltanet_rd_exactness_sweep.py`, Gate 2 as a
  committed script) → independent code audit → Gate 2 (CPU) → Wave −1
  (smokes + probes; Gate 1 launch-read) → Wave 1 (candidates (d) and
  (c), early-stop armed) → assess against §3.5's outcome frame →
  candidate (b) / fixed-grid λ / follow-ons only per their registered
  triggers.
