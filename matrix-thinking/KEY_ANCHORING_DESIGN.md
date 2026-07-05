# KEY_ANCHORING_DESIGN — From "orthogonal but drifting" to "stable and orthogonal": a follow-on to F-geo-3

> **Rev 5 — 2026-07-04, post-bounded-verify-round-4. Single-field
> revision.** Round 4 (appended to
> `matrix-thinking/KEY_ANCHORING_ATTACK_R3.md`) confirmed **all five R3
> items CLOSED** — including an executable reproduction of the §2.2
> gather/scatter NaN fix with a contrast run proving the superseded
> `torch.where` form really was gradient-poisoned — with **one blocker**:
> §3.6's `--unblind-override` demotion was committed only to the wave
> summary, while this repo's own audit methodology (and its own
> `lm_pretrain_rd.py::_assemble_result` `claim_tier` precedent, L1090,
> smoke-asserted at L1420) reads individual result JSONs. Rev 5 is
> exactly the prescribed edit and nothing else: the override now stamps
> **every affected anchor-arm's own result JSON at result-assembly
> time** (`claim_tier: "descriptive"`, `unblind_override: true`,
> `unblind_override_at`; non-override runs always carry
> `unblind_override: false` so absence is never ambiguous — §3.6), with
> a new in-process CPU smoke 9 asserting both cases land in the JSON
> (§5; no new GPU probe run, Wave −1 counts and budget unchanged), and
> response-map row §8.4.

> **Rev 4 — 2026-07-04, post-verify-round-3. Expected FINAL revision
> round.** Round 3 (`matrix-thinking/KEY_ANCHORING_ATTACK_R3.md`)
> returned **NEEDS-REV-4: all 9 R2 findings confirmed closed** (the
> Gate-2 regression quadruple, the early-stop archive table, and the
> shared-c root cause each independently reproduced to 4 decimal places
> or to the literal source line; the hypothesis and candidate ranking
> unthreatened for the fourth straight round) — but **3 MAJOR + 2 MINOR
> gaps inside Rev 3's own new machinery**. All five addressed under
> binding orchestrator decisions; map in **§8.3**; everything else is
> FROZEN per instruction. The changes: **the λ oscillation window is
> pinned in LOG-POINT space** (last 5 logged points = 1,000 steps at
> the registered 200-step cadence, harness-asserted at startup so a
> cadence change fires an assertion instead of silently resizing the
> window; §3.2); **the reference-arm blinding becomes a mechanical
> harness gate, not a norm** — `BANDS_PINNED.json` written only after
> all reference arms complete, anchor cells REFUSE to launch without a
> validating file, and the readout script asserts the pin timestamp
> precedes every anchor-arm start time (three registered build
> requirements with failure modes; §3.6); **reference seeds n=2→3 per
> K** (6 reference arms, +1.6–1.7 GPU-h; mandatory baseline ~8.7–9.4
> GPU-h, still ≤10 nominal; band formula and UNRESOLVABLE guard
> recomputed at n=3; §3.6/§5); **the blend construction moves from
> `torch.where` to masked gather/scatter** — arithmetic touches
> trained-entity rows ONLY, so no 0×NaN gradient-poisoning path exists
> in either direction, with a registered NaN-injection unit test
> (§2.2/§5 smoke 4); and **§3.7's metric is renamed
> anchor-INPUT-alignment** (what it is), keeping it as the registered
> `engaged_frac` driver while adding a non-load-bearing per-entity h=1
> behavioral diagnostic so the headline never silently rests on an
> input-side proxy (§3.7).

> **Rev 3 — 2026-07-03, post-attack-round-2.** The round-2 adversarial
> review (`matrix-thinking/KEY_ANCHORING_ATTACK_R2.md`) returned
> **NEEDS-REV-3: all three R1 FATALs confirmed CLOSED** (the verifier
> independently re-derived the frame-potential construction from scratch
> and matched the λ=1 K=32 ceiling at 0.9424 vs. this doc's 0.9423,
> seed-stable 0.940–0.942 across 4 seeds) — but it constructed **real
> counterexamples for 5 new MAJORs**, all addressed in this revision
> under binding orchestrator decisions; the finding→change map is
> **§8.2**. The load-bearing changes: **λ-band assignment gains an
> oscillation exclusion** (final value AND trailing-1,000-step mean in
> the same band AND trailing-window range < 0.1 — the same medicine
> candidate (b) already got at m4; §3.2); **items 6a/6b are promoted
> INTO the launch gate and re-run at every admission checkpoint** —
> closing R2's demonstrated blind spot (a moderately-collapsed table
> sailed through both Rev 2 gates), with that table pinned as a seeded
> REGRESSION CASE the amended gate demonstrably fails (run this session:
> 6a=0.0762 FAIL, 6b=0.5371 FAIL, NS-only leg PASS — §4); the
> **early-stop kill leg is now value-Gram-only at two consecutive
> checkpoints** (h4 recorded but non-load-bearing — its 58% seed spread
> vs. value-Gram's ~15% is disclosed inline; §3.4); the provisional
> 0.92/0.96 "engaged" bands are **superseded by
> reference-arm-derived bands** — the archive check confirmed R2's
> finding (no per-seed drift data exists anywhere in the archives; the
> 0.9037/0.9416 reference points were 5,000-step probe measurements,
> never the true final checkpoint), so **2 bare-geo3 reference seeds per
> K join the manifest (~3.3 GPU-h) and bands are pinned from their
> measured final-checkpoint drift BEFORE any anchor arm is read out**
> (ordering/blinding protocol, §3.6); a **per-entity anchor-alignment
> readout** is pre-registered with pinned engagement bands (≥90%
> headline / 50–90% partial / <50% not-recruited; §3.7); the R2
> simulator discrepancy is **RESOLVED mid-revision by the coordinator's
> GPU re-measurement — it was a BUG, not platform sensitivity**:
> `geo3_drift_diagnostic.py::main()` extracts only K=16's measured
> drift and `launch_read()` applies that one scalar to BOTH K, so the
> recorded K=32 "prediction" (0.7734) was computed at K=16's drift
> (0.9416), never at K=32's own (0.9037) — with correct per-K inputs
> the mapping **underestimates** K=32 recovery (0.06–0.09 vs. measured
> 0.4368), GPU==CPU everywhere, parent §16.7 now carries the dated
> correction (archive:
> `experiment-runs/2026-07-04_geo3_simulator_recheck/`); the §3.4
> caveat box states the true cause, Gate 1 is unaffected (K=16 used its
> own drift), and the per-K drift-threading API fix + unit test is
> registered in THIS wave's build scope (§4); minors fixed (the 42.2%
> arithmetic slip; the C17 bypass rebuilt as a genuinely bit-exact
> select/where path after R2 measured 337/1000 rows differing at 1 ULP
> under the old multiply-by-zero framing; the Wave −1 smoke suite
> itemized, §5).

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
  of this document (3 FATAL / 3 MAJOR / 4 MINOR; response map §8.1).
- `matrix-thinking/KEY_ANCHORING_ATTACK_R2.md` — attack round 2 on Rev 2
  (R1 FATALs verified closed by independent from-scratch reproduction;
  5 new MAJOR / 3 MINOR, several with constructed counterexamples;
  response map §8.2).
- `matrix-thinking/KEY_ANCHORING_ATTACK_R3.md` — verify round 3 on Rev 3
  (all R2 findings confirmed closed, several to 4-decimal reproduction;
  3 MAJOR / 2 MINOR, all inside Rev 3's own new machinery; response map
  §8.3).
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
  extraction for the M3 early-stop thresholds (§3.4); **and (Rev 3)**
  the pinned moderate-collapse REGRESSION CASE run against the amended
  Gate 2 (§4), the archive-extractability check for per-seed drift
  (negative — decision branch for §3.6), geo3's step-4000 trajectory
  extraction (§3.4), and the corrected band arithmetic (§3.1). All are
  cited with their exact numbers and method — none invented, all
  reproducible from files already in this repo.

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
t_idx              = anchor_trained_mask[key_ids].nonzero(as_tuple=True)   # trained positions ONLY
sub_blend          = normalize( (1 - λ) · k_eff_raw[t_idx] + λ · A[key_ids[t_idx]] )
k_blend_raw        = k_eff_raw.clone()          # held-out rows: original values — no arithmetic,
k_blend_raw[t_idx] = sub_blend                  #   and NO graph edge into any held-out anchor row
k_eff_items        = geo3_orthogonalize_logged(k_blend_raw, n_iter=..., resid_tol=...)   # UNCHANGED call
```

The masked gather/scatter is the M1 held-out bypass (§3.3) — **REVISED
at Rev 3 (R2 target 6), REVISED AGAIN at Rev 4 (R3 finding 4,
orchestrator decision 4)**: Rev 2's multiply-by-zero formulation
(`λ_eff = λ·mask`, then re-normalize) claimed the held-out path was
"exactly the same tensor," but R2 measured that re-normalizing an
already-unit-norm fp32 vector is NOT a universal no-op (337/1000 rows
differ by up to 5.96e-8 — 1 ULP). Rev 3's `torch.where` select fixed the
**forward** path (zero arithmetic on the chosen value) but left the
standard backward footgun R3 named: `torch.where` evaluates BOTH
branches, so a non-finite value arising anywhere in the *discarded*
blend branch for a held-out row (e.g. a pathological near-zero vector
through `normalize`, in the same adversarial regime the eigh fallback
exists for) can poison gradients through the 0×NaN pattern even though
the forward select is clean. The Rev 4 gather/scatter form computes the
blend **only on trained-entity rows** — no op in the graph ever touches
a held-out anchor row in either direction (forward OR backward), the
held-out rows of `k_blend_raw` are a bit-exact `clone` of `k_eff_raw`
(so §3.3's strict `torch.equal` smoke remains valid), and the anchor
table's gradient is structurally zero at held-out rows rather than
zero-by-cancellation. **Registered unit test (wired into §5 smoke 4):
inject NaN into every held-out anchor row, run forward+backward on a
mixed-split batch, assert all gradients finite and held-out outputs
bit-equal to pure-geo3.** `λ` is registered as a
**single learned scalar**
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
- λ-trajectory logging (F3; oscillation stats added at Rev 3 per R2
  MAJOR 5; window pinned in LOG-POINT space at Rev 4 per R3 finding 1):
  `sigmoid(raw_param)` recorded into the result JSON's trajectory
  records at every logged step — the harness's confirmed cadence is
  **200 steps** (verified by R3 against a real archived trajectory:
  steps 1, 200, 400, …, 20000), registered as the constant
  `LAMBDA_LOG_CADENCE_STEPS = 200` alongside
  `LAMBDA_WINDOW_LOG_POINTS = 5`, **asserted at harness startup**
  (`assert log_every == LAMBDA_LOG_CADENCE_STEPS`) so a cadence change
  fires loudly instead of silently resizing the window. The full
  trajectory is a required field per run, and the final-window summary
  (final value; mean and max−min range over the **last 5 logged
  points** = 1,000 steps at the registered cadence) is written as its
  own result field so §3.2's oscillation exclusion is
  machine-checkable, not eyeballed.

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

**Interpretation bands — PROVISIONAL at Rev 3, superseded by §3.6's
reference-derived bands before any anchor arm is read out (R2 MAJOR 2 +
MINOR 1, orchestrator decision 4).** Rev 2 pinned "mechanically
engaged" = post-NS drift ≥ 0.92 at K=32 / ≥ 0.96 at K=16, justifying
0.92 as "more than half the distance" from geo3's 0.9037 toward the
0.9423 ceiling — **that arithmetic was wrong** (gap 0.0386, halfway
0.9230; 0.92 is **42.2%** of the distance — R2's correction, verified:
the K=16 figure, 55.9%, was correct but the claim was only made where
it was false). More importantly, R2 established that the 0.9037/0.9416
reference points themselves are **single-seed, 5,000-step-probe
measurements** (`geo3_drift_diagnostic.py` trains a fresh throwaway
model; it never loads the real 20,000-step checkpoints), with **no
seed-level variance data anywhere in the archives** — so no threshold
hung off them can distinguish "engaged" from "baseline drifted up on a
lucky seed." Resolution: the provisional 0.92/0.96 numbers stand ONLY
until §3.6's reference arms complete; the operative bands are then
**derived from measured baseline mean + spread at the true final
checkpoint**, per the pre-registered formula and blinding protocol in
§3.6. No anchor-arm outcome is ever read against the provisional
numbers.

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

**Oscillation exclusion (Rev 3 — R2 MAJOR 5; window PINNED IN LOG-POINT
SPACE at Rev 4 — R3 finding 1, orchestrator decision 1; the same
medicine candidate (b)'s EMA already got at m4).**
A mean can land in-band while the trajectory oscillates through it —
Rev 2 logged the trajectory but excluded nothing. Rev 3's fix wrote the
window as "the trailing 1,000 steps," which R3 showed is ambiguous
against the harness's real logging cadence (200 steps, verified from an
archived trajectory — 101 points over a 20,000-step run): read as
steps it holds 5–6 samples; read as "1,000 *logged* samples" it would
silently consume the entire trajectory including early unsettled
training. Rev 4 pins it: **the window is the LAST 5 LOGGED POINTS**
(= 1,000 steps at the registered `LAMBDA_LOG_CADENCE_STEPS = 200`;
constants and the startup assertion in §2.2's logging bullet — a
cadence change fires the assertion rather than silently changing the
window). Per-seed band assignment requires **all three** of: (i) the
**final λ value** (last logged point) in the band; (ii) the **mean λ̄
over the last 5 logged points** in the same band; (iii) the **range
(max − min) over the last 5 logged points < 0.1**. A seed failing any
of the three lands in the **ambiguous** band regardless of its mean —
a pre-registered exclusion, not a logging note. The three summary
statistics are machine-readable result fields (§2.2), so the exclusion
is checkable without trajectory eyeballing. **Statistical power of a
5-point window, stated honestly (R3's own ask):** this is a
gross-oscillation catch — it detects swings ≥ 0.1 that persist at the
200-step cadence; sub-cadence oscillation is unobservable at any
window size this cadence supports, and a 5-point range is a coarse
statistic by construction. That is the intended scope: the exclusion
exists to stop a *visibly* unstable λ from being banded by its mean,
not to certify fine-grained stability — and any seed it excludes lands
in the ambiguous band, whose registered follow-up (the fixed-grid λ
diagnostic, §2.2) does not depend on trajectory statistics at all.

Arm-level label: **≥2/3 seeds in a band** (each seed banded under the
three-part rule above) assigns that band; seeds spread across
non-adjacent bands → ambiguous, no headline. The headline-eligibility
requirement composes with §3.1: **Outcome A (§3.5) requires the
interior band** — a bar-clearing run in the >0.95 band reports as
Outcome A′ (pin rediscovery), never as the interaction headline.

### 3.3 C17 held-out entities (NEW, Rev 2 — attack M1, orchestrator-pinned)

**Eval-time behavior, specified in the design, not left to the build:**
held-out entities **bypass the blend** — the masked gather/scatter on
the `anchor_trained_mask` buffer routes held-out rows through **zero
arithmetic in either direction** (§2.2, Rev 4 form — the Rev 3
`torch.where` form was superseded per R3 finding 4's backward-path
footgun). C17 episodes are drawn entirely from
the held-out pool, so under the bypass a C17 episode routes through
`bind()` **identically to bare geo3** — C17 measures pure-geo3 behavior
by construction. **Wave −1 smoke (restated honestly at Rev 3 — R2
target 6):** on an all-held-out batch with `anchor_active=True`,
`bind()`'s outputs must be **bit-identical** (strict `torch.equal`) to
the same weights with the anchor disabled. Rev 2's version of this
claim rested on `normalize(1·k_eff_raw + 0·A) = k_eff_raw` being a
floating-point no-op — R2 measured that it is not (**337/1000
already-unit-norm fp32 rows differ by up to 5.96e-8 (1 ULP)** on
re-normalization), which would have made a strict-equality smoke
intermittently fail on correct code. The gather/scatter path (Rev 4
form, §2.2) makes the bypass genuinely bit-exact — held-out rows are a
`clone` of the original tensor values, never multiplied, added, or
re-normalized — so strict equality is the *correct* assertion again,
and a failure of this smoke now indicates a real routing bug, not ULP
noise.

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

### 3.4 Early-stop — 'value-geometry-bound' (Rev 2, attack M3; kill rule REVISED at Rev 3 — R2 target 5, orchestrator decision 3)

A **budget guard, not a claim.** The harness logs checkpoints every
2,000 steps. Reference values, extracted from geo3's own archived
trajectories (`wgeo3_rdx_K32_armgeo3_s{0,1,2}_geo3n20.json`,
`wgeo3_rdx_K16_armgeo3_s{0,1,2}_geo3n12.json` — step-2000 values
verified independently by R2 target 5, "exact match"):

| K | geo3 `value_gram_deviation_mean`, step 2000 → step 4000 (3 seeds) | geo3 h=4 `rec@0.9`, step 2000 (3 seeds) |
|---|---|---|
| 32 | 3.3937→3.8777 / 3.8543→4.8545 / 3.8991→5.2612 | 0.1011 / 0.1321 / 0.0834 |
| 16 | 1.6596→1.5092 / 1.6985→1.6324 / 1.6701→1.5695 | 0.7797 / 0.7123 / 0.7504 |

**The Rev 3 kill rule (replaces Rev 2's single-checkpoint AND):**
**KILL the arm iff `value_gram_deviation_mean` exceeds the threshold
(K=32: > 3.90; K=16: > 1.70) at BOTH of the first two consecutive
checkpoints (step 2,000 AND step 4,000).** Value-Gram is the sole
load-bearing kill leg. **h4 at the same checkpoints stays recorded and
reported but is explicitly NON-load-bearing** — the archived data
itself shows why: at step 2000, K=32, h4 has a **58% relative seed
spread** (0.0834–0.1321) across just 3 known-good geo3 seeds, while
value-Gram spreads only ~15% (3.3937–3.8991) at the same checkpoint.
Rev 2's AND rule conditioned the kill on the *noisier* leg, so an arm
decisively past the value-Gram threshold would survive whenever h4
wobbled above its floor by chance — exactly the false-continue R2
demonstrated is realistic at the observed noise. The two-consecutive-
checkpoint requirement replaces h4's confirmation role with a
*persistence* check on the low-noise signal itself.

**Cost-asymmetry rationale, stated explicitly (R2's own ask):** a
false-kill costs a mandatory headline cell (re-run ≈0.25–0.28 GPU-h +
schedule delay + a manifest deviation to document); a false-continue
costs at most the remaining ~90% of one run (≈0.22–0.25 GPU-h). The
costs are comparable, so the rule is tuned for *signal quality* rather
than either error direction: kill only on the low-noise leg, and only
when it persists across two checkpoints.

**Threshold asymmetry across K, disclosed:** at K=16, geo3's own
value-Gram *falls* from step 2000 to 4000 (all seeds < 1.70 at step
4000), so both legs of the K=16 rule are genuinely geo3-relative. At
K=32, geo3's own value-Gram *rises* (2 of 3 seeds exceed 3.90 by step
4000), so the step-4000 leg at the same 3.90 threshold is nearly
automatic once the step-2000 leg fires — at K=32 the load-bearing
comparison is the step-2000 read (which exceeds geo3's own same-step
max), and the step-4000 leg functions as a transient-spike guard, not
an independent geo3-relative test. Stated so the rule is read for what
it is.

A killed arm logs outcome **'value-geometry-bound'**, its spend stops
at ~20% of the run, and the §6 Outcome-B re-attribution machinery takes
over. The comparison is deliberately step-to-step matched (the
same-stage discipline attack M2 itself demanded), never
early-checkpoint-to-final.

> **CAVEAT BOX (Rev 3 — R2 MAJOR 3, orchestrator decision 6; RESOLVED
> mid-revision by the coordinator's GPU re-measurement): the §16.7
> discrepancy was a BUG, not platform sensitivity.** Root cause,
> confirmed on GPU and verifiable in the code as committed:
> `geo3_drift_diagnostic.py::main()` extracts only **K=16's** measured
> drift (`lr16 = per_k[16]["after_probe"]`) and
> `geo3_simulator.launch_read()` applies that one scalar `c` to **both**
> K in its loop — so the recorded "K=32 prediction 0.7734" (§16.1) was
> actually `simulate_recovery(K=32, c=0.9416)`, computed at **K=16's**
> drift; K=32's own measured drift (0.9037) was never wired in. With
> c=0.9416 the 0.7734 reproduces to the last digit; with K=32's own
> 0.9037 the prediction is **0.06–0.09** — a strong **underestimate** of
> the measured 0.4368. **GPU==CPU everywhere**; R2's CPU numbers were
> correct all along (this design's own steep-cliff sweep — c=0.9037 →
> 0.066, c=0.9423 → 0.746 — is exactly why the wrong `c` produced such a
> large phantom effect at K=32 and none at K=16). R2's TF32 hypothesis
> is disconfirmed by the same measurement. The parent
> `DELTANET_RD_EXACTNESS_DESIGN.md` §16.7 now carries the dated
> correction (made from the GPU measurement, not CPU inference; archive:
> `experiment-runs/2026-07-04_geo3_simulator_recheck/`). Consequences
> here: (i) this section's **kill thresholds are unaffected** — they are
> calibrated purely from geo3's archived training trajectories and never
> touch `simulate_recovery` (R2 verified none of outcome-F's three legs
> do either); (ii) **Gate 1 is unaffected** — K=16 used its own drift
> and reproduces cleanly (§4); (iii) with correct per-K inputs the
> registered mapping is **conservative at K=32** (underestimates
> recovery ~5–7×: predicted 0.06–0.09 vs. geo3's measured 0.4368), so
> any future K=32 simulator gate would be strict/pessimistic — see §4;
> (iv) §6's Outcome-B2 value-geometry re-attribution now rests on the
> simulator-independent evidence ONLY (the measured 1.3–1.8× elevated
> value-Gram deviation and the intervention-broken key/value
> correlation, §4) — the old "§16.7 overestimate" narrative is void, and
> this design's own Rev 1 "~0.19 corrected-simulator ceiling" is
> **retired as evidence** (the measured 0.4368 already exceeds it,
> proving that correction over-pessimistic; §4).

### 3.5 Outcome frame at K=32 (REVISED, Rev 2; engagement + derived-band hooks added Rev 3)

- **Outcome A — CONFIRMED (headline).** Pre-NS drift ≥0.95 (item 5) AND
  h4 ≥0.5, 3/3 admissible under items 1–6, AND per-seed λ in the
  interior band [0.2, 0.8] under §3.2's three-part rule (final value +
  trailing mean + range < 0.1) for ≥2/3 seeds, AND per-entity
  anchor-input-alignment engagement ≥90% (§3.7). KEY_ANCHORING as an
  *interaction mechanism with the learned key path in the loop* is
  confirmed.
- **Outcome A″ — partial anchoring (NEW, Rev 3 — R2 MAJOR 4).** Bars
  and λ band clear but per-entity anchor-input-alignment engagement
  lands in [50%, 90%) — reported as a named partial outcome ("anchoring
  lifts composition for the engaged subset"), **no headline** (§3.7).
- **Outcome A′ — pin rediscovery (existence tier).** Bars clear but λ
  lands in the >0.95 band: SGD re-derives the fixed-frame regime.
  Confirms reachability-by-gradient-descent of a near-pin solution over
  the full 107 pool (itself one publishable sentence, given §2.2's
  ceiling computation says even that regime tops out at post-NS drift
  ~0.9423); does NOT support §1's interaction hypothesis as framed, and
  is never written up as the headline.
- **Outcome B — INFORMATIVE NEGATIVE, re-attribution required.** Item 5
  passes (pre-NS drift ≥0.95) but h4 stays <0.5. **Disambiguation step,
  pre-registered (uses the non-evidentiary post-NS sanity bands — Rev 3:
  the §3.6 REFERENCE-DERIVED bands, never the provisional 0.92/0.96):**
  (B1) if post-NS drift stayed **below the derived engaged threshold**
  (not mechanically engaged — the stabilized pre-NS input still
  produces subset-dependent NS output at baseline-indistinguishable
  levels), the failure is attributed to the **coherence-floor residual**
  (the F2 packing geometry: stabilizing the input cannot stabilize the
  output past the ~0.9423 ceiling, and the realized input→output
  stability transfer was worse than the ceiling computation's idealized
  estimate) — the follow-on is pool restriction or d_state escalation,
  NOT value geometry; (B2) if post-NS drift **cleared the derived
  engaged threshold** and h4 still missed, the outcome-F attribution is
  incomplete and the re-attribution goes to **value geometry** (§6) —
  supported at Rev 3 by the simulator-independent evidence only
  (geo3's 1.3–1.8× elevated value-Gram deviation; the
  intervention-broken key/value correlation, §4), per §3.4's caveat
  box: the §16.7 simulator-overestimate narrative is void (shared-c
  bug, corrected in the parent doc from the coordinator's GPU
  re-measurement), and the Rev 1 "~0.19 corrected-simulator ceiling"
  is retired as evidence.
- **Outcome C — mechanism not engaged.** Item 5 fails (pre-NS drift
  <0.95) regardless of h4, **or (Rev 3, §3.7) per-entity
  anchor-input-alignment engagement <50% regardless of aggregate
  drift** — the intervention did not
  achieve its own mechanical goal; not an admissible test of the
  hypothesis either way. Routes to the fixed-grid λ diagnostic or a
  different candidate, not to reinterpreting the drift-bottleneck
  theory.
- **Outcome D — reference only.** Candidate (a)'s archived
  pool-restricted result — the (multi-concession) ceiling, not a new
  data point this wave produces.

### 3.6 Reference arms and the derived "engaged" bands (Rev 3 — R2 MAJOR 2; n=3 + mechanical blinding gate at Rev 4 — R3 findings 2/3, orchestrator decisions 2/3)

**The archive-extractability check, run first per the decision's own
ordering (this session, CPU, free):** no per-checkpoint or per-seed
drift field exists in any archived geo3 run JSON (checkpoint keys are
`M2_in_distribution`/`M3_held_out`/`C17_heldout_entities`/
`C19_heldout_template`/`fixedref_entity_subspace` only; a full-text
scan of the JSONs for `drift`/`pairwise`/`resample` finds nothing), and
`GEO3_DRIFT_DIAGNOSTIC.json` — the only drift measurement in the
archive — records `sampling_spec.seed = 0`, single seed, 5,000-step
probe. R2's finding is confirmed in full: **the 0.9037/0.9416 reference
points are single-seed probe measurements, never measured on the
actual, fully-trained, bar-clearing geo3 model, and no seed-variance
data exists.** The free branch is closed; the reference arms are
therefore required.

**Reference arms (added to the wave manifest; n=2→3 at Rev 4 — R3
finding 3 / Judgment Call (c), orchestrator decision 3):** **3 fresh
bare-geo3 seeds × K∈{16,32} = 6 runs** (seeds 1, 2, 3 — matching the
×3-seed convention every mandatory cell in §5 already uses; the
existing seed-0 probe value is quoted as a sanity point only, never
pooled into the derivation — it is a different training stage), 20,000
steps, identical config to the archived geo3 cells **plus the
per-checkpoint drift diagnostic active** (both the post-NS §16.1
statistic and, for instrument symmetry with the anchor arms, the
pre-NS raw-key drift; `measure_drift` machinery unchanged). Priced at
**~5.0 GPU-h** (~0.83 GPU-h/run at the instrumented rate the Rev 3
allocation implied — the drift diagnostic's ~2,560 bind() calls per
checkpoint × 10 checkpoints make these materially more expensive than
plain runs; the n=3 upgrade adds ~1.6–1.7 over Rev 3's 4-run row).
These arms close two gaps at once: seed variance, and the
probe-vs-true-final-checkpoint gap — drift is finally measured on real
20,000-step geo3 models.

**Band derivation, pre-registered before any reference data exists
(recomputed at n=3 per decision 3):** per K, over the **3**
reference-arm **final-checkpoint** post-NS drift values:
`engaged_K = mean_ref + 2·s_ref` (sample std, n=3, df=2). Why n=3 and
not Rev 3's n=2 (R3's power analysis, accepted): at n=2 the sampling
distribution of *s* itself has a relative standard error of ≈ 1/√2 ≈
71% — `engaged_K` would be dominated by which two seeds happened to be
drawn (the same "engaged vs. lucky seed" problem R2-M2 was raised to
fix, reappearing inside the fix); at n=3 (df=2) the RSE drops to ≈ 50%,
and this project's archived data already shows ~13–15% relative
seed-to-seed spread on adjacent statistics, making the n=2 estimator's
failure modes realistic rather than hypothetical. Per-seed values and
the range are always reported alongside the derived threshold.
**Degenerate-case guard (trigger recomputed at n=3):** if
`engaged_K = mean_ref + 2·s_ref (n=3) ≥ ceiling_K − 0.005` (ceilings:
0.9423 at K=32, 0.9745 at K=16, §2.2), the post-NS engagement read is
declared **UNRESOLVABLE at that K** — baseline seed noise swallows the
achievable window — and §3.5's B1/B2 disambiguation reports
"indeterminate" at that K (mirroring the parent §14.5's
indeterminate-band discipline). Any UNRESOLVABLE declaration must be
accompanied by the **leave-one-out sensitivity report** (the three
`engaged_K` values computed dropping each seed in turn), so a trigger
driven by a single outlier seed is visible as such rather than read as
a structural floor-vs-ceiling collision — at n=3 this is
disclosure-only (no re-decision rule hangs off it; the guard's verdict
stands as computed).

**Blinding — a MECHANICAL HARNESS GATE, not a stated norm (REWRITTEN at
Rev 4 — R3 finding 2, orchestrator decision 2). Three registered build
requirements, each with its failure mode:**
1. **The writer.** The wave harness writes `BANDS_PINNED.json` —
   containing the derived bands (or UNRESOLVABLE verdicts) per K, the
   per-seed drift inputs, the formula version, **sha256 hashes of all 6
   reference-arm result JSONs**, and a timestamp — **only after every
   reference arm's result JSON validates as complete**
   (`complete == true`, `steps_completed == 20000`, drift fields
   present; the same `is_done`-style validation the sweep harness
   already applies). *Failure mode:* any reference arm incomplete,
   timed out, or missing drift fields → the file is not written → gate
   2 blocks every anchor cell; there is no partial-bands state.
2. **The launcher gate.** Anchor-arm cells **REFUSE to launch** unless
   `BANDS_PINNED.json` exists AND validates — the launcher re-hashes
   the referenced reference-arm JSONs and checks they match the
   recorded hashes, and parses the bands (the same
   loud-refusal-with-explicit-override pattern as
   `gate_geo3_drift`/the wave-2 gate chain). This supersedes Rev 3's
   "anchor arms may train concurrently" allowance: **reference arms
   complete first, bands pin, anchor arms launch after — sequencing is
   enforced by the launcher, not by analyst discipline.** *Failure
   modes:* missing file → refusal with the instruction to run the
   reference arms; hash mismatch (a reference JSON changed after
   pinning) → refusal flagged as a pin-integrity error, never silently
   re-derived. An explicit `--unblind-override` exists (mirroring
   `--accept-gate-override`'s loudly-logged *refusal/override* pattern)
   but its use **automatically demotes every anchor-arm readout in
   that wave to descriptive tier** — the override changes the claim
   tier, it never silently preserves it. **Where the demotion is
   recorded (REVISED at Rev 5 — round-4 judgment call, orchestrator
   prescription): in EVERY affected anchor-arm's own individual result
   JSON, at result-assembly time, IN ADDITION TO the wave summary —
   never the summary alone.** When the launcher is invoked with
   `--unblind-override`, it threads the override stamp down to each
   spawned anchor run, and the runner's result assembly writes three
   top-level fields into that run's result JSON as it is produced (not
   patched post-hoc): `claim_tier: "descriptive"`,
   `unblind_override: true`, and `unblind_override_at: <the override
   timestamp>`. This adopts this repo's own working precedent —
   `lm_pretrain_rd.py::_assemble_result` (L1090) writes `claim_tier`
   into every individual run's result dict, asserted by its own smoke
   (L1420), and the field is confirmed present in every archived
   Track-C result JSON — rather than the `--accept-gate-override`
   pattern, whose track record in this exact harness family is
   empirically zero trace in individual run JSONs (round 4 checked: no
   `gate_bypassed`/`claim_tier` key in any archived exactness result;
   a launch-console/wave-summary artifact is exactly the record this
   project's own file-by-file audit methodology would miss). Runs
   launched WITHOUT the override always carry
   `unblind_override: false` (written at assembly, so the field's
   absence can never be read as evidence of a clean blind — presence
   is mandatory either way); `claim_tier: "descriptive"` is written
   only on the override path, because the demotion is the one tier
   verdict knowable at launch time regardless of anything the run
   later earns (non-override tiers remain readout-time verdicts, per
   the admission stack). §5 smoke 9 asserts both cases.
3. **The readout assertion.** The readout/analysis script asserts
   `BANDS_PINNED.json`'s timestamp **strictly precedes the earliest
   anchor-arm start time** recorded in the anchor result JSONs (each
   run JSON already carries its own timing fields; the build adds an
   explicit `started_at` if absent). *Failure mode:* violation → the
   readout aborts, the wave summary marks the blind as broken, and
   every affected anchor readout reports at descriptive tier only —
   the assertion makes a broken blind a *recorded, tier-demoting
   event*, not a judgment call.

The provisional 0.92/0.96 numbers (§3.1) are **void** the moment the
derived bands exist; no anchor-arm outcome is ever read against them.

### 3.7 Per-entity anchor-INPUT-alignment readout (Rev 3 — R2 MAJOR 4; renamed + behavioral companion added at Rev 4 — R3 finding 5, orchestrator decision 5)

R2 target 4(b) established that nothing in the training-eval pipeline
has per-entity visibility: the headline `rec@0.9` and drift statistics
are pooled over episodes drawn from the full 107-entity train pool, so
a passing aggregate could mask a mechanism engaged for a subset of
entities while the rest behave as bare geo3 — invisible to items 5 and
6 alike (item 6 checks the table's geometry, not per-entity causal
engagement). §3.3 already set the precedent at the train/held-out
boundary; this section extends it **within** the train pool.

- **Required logging: per-entity anchor-input-alignment** (the Rev 4
  name — R3 confirmed the pre-NS blend is the correct object for R2's
  literal ask, and the rename makes the metric say what it is: an
  **input-side** quantity):
  `a_e = mean over ≥8 independent episode resamples of
  cos(pre-NS blended key of entity e, A[e])`, computed for **all 107
  train entities** (the existing `measure_drift`/
  `sample_batch_fixed_entity` machinery, swept over the full pool
  instead of 8 sampled entities — eval-only, no training-path change),
  logged at every admission checkpoint; the claim readout uses the
  **final step**. For candidate (b), `a_e` reads the input to the final
  NS call vs. the EMA row; for candidate (c), the raw gathered
  `k_eff_raw` vs. the EMA row (no blend exists — the regularizer must
  move the raw key itself).
- **Partial-anchoring readout, pre-registered:** `engaged_frac` =
  fraction of train entities with `a_e ≥ 0.9` at the final step.
  `engaged_frac` (input-alignment-based) remains the **registered
  driver** of the bands below.
- **Bands (orchestrator-pinned):** **≥ 90%** engaged → headline-eligible
  (a requirement of Outcome A, §3.5); **[50%, 90%)** → **partial
  anchoring**, a named outcome (Outcome A″), reported in full, no
  headline; **< 50%** → **not recruited, regardless of aggregate
  drift** — routes to Outcome C even if item 5's pooled statistic
  passes (a pooled drift number carried by a minority of
  strongly-anchored entities is exactly the aggregate-masking scenario
  this readout exists to catch).
- **Behavioral companion diagnostic (NEW, Rev 4 — non-load-bearing,
  disclosed as diagnostic-only):** per-entity **h=1 recovery**,
  computed over eval episodes *containing that entity* (restricting the
  already-computed per-item h=1 cosine scores by entity id — a
  bookkeeping change to the existing eval, no new forward passes),
  reported alongside `a_e` for all 107 entities at the final
  checkpoint. **Scope limit, stated per §3.3's own precedent (R3
  finding 5):** `a_e` is an input-side proxy — it measures whether the
  entity's key is pulled toward its anchor at the blend stage, not
  whether NS's cross-row mixing translates that into a stabilized
  *written* key with better per-entity recovery; `M3_held_out` remains
  pooled with no per-entity h=4 breakdown even after this fix. The
  behavioral companion narrows that gap at h=1 only (per-entity h=4
  restricted samples would be too sparse to read at this eval budget)
  and carries **no admission or claim-tier weight** — it exists so a
  headline never silently rests on the input-side proxy alone: a large,
  visible divergence between the `a_e` vector and the per-entity h=1
  vector is reported as an open question in the write-up, not
  adjudicated by any pre-registered rule.
- The full per-entity vectors (both `a_e` and the h=1 companion, not
  just the fraction) are required result fields, so the write-up can
  report *which* entities disengage (e.g. frequency- or
  coherence-correlated patterns) rather than only how many.

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

**Gate 2 (Rev 2 — attack M2; AMENDED at Rev 3 — R2 target 1,
orchestrator decision 2): CPU-only pre-spend K=32 construction gate,
now including the table-conditioning legs.** R2 constructed the exact
counterexample the Rev 2 gates were worried about: a **moderately
collapsed** table (two shared directions, noise=0.30) that already
fails items 6a/6b — yet sailed through **both** Rev 2 pre-spend gates
(Gate 2's NS leg: 0/512 fallbacks; Gate 1: post-NS drift 0.9361 →
predicted 0.9922 ≥ 0.8), because NS convergence and drift-based
prediction are simply not collapse detectors. The amendment closes
this at both ends:

- **Procedure (amended):** on the frozen registered anchor init (the
  frame-potential table, §2.2), the gate now has **three legs, all
  CPU-cheap, all mandatory**: **(G2-a)** item 6a on the raw table —
  `σ_64/σ_1 ≥ 0.1` (SVD of the 107×64 train block); **(G2-b)** item 6b —
  `max_{i≠j}|cos| ≤ 0.5`; **(G2-c)** the NS leg — ≥512 random
  32-subsets (and 16-subsets) through the production `newton_schulz` at
  the production tier (`n_iter=20` for K=32, 12 for K=16, §16.3), 100%
  converging to residual ≤ `resid_tol=1e-2` with zero fallbacks
  (§14.10-item-2 semantics), pre-NS subset Gram-dev band reported.
  **Any leg failing blocks the launch.** **And during training (R2
  target 1(b) — Gate 2 alone can never see SGD-driven collapse): items
  6a/6b are re-run at every admission checkpoint** on the
  then-current table (a 107×64 SVD every 2,000 steps — negligible), so
  a table that *degrades into* the moderate-collapse regime mid-run is
  flagged at the next checkpoint, not discovered post-hoc at final
  admission.
- **Pinned REGRESSION CASE (run this session; the amended gate must
  demonstrably fail it, and does).** Recipe, pinned for bit-level
  reproducibility: two shared unit directions and row assignments drawn
  at torch seed 42; 107 rows perturbed at noise σ=0.30, renormalized
  (R2's own construction; R2's instance — a different seed — measured
  σ_ratio=0.0989, max|cos|=0.5376). Measured on the pinned instance:
  **G2-a: σ_64/σ_1 = 0.0762 → FAIL** (< 0.1); **G2-b: max|cos| =
  0.5371 → FAIL** (> 0.5); **G2-c: 0/512 fallbacks at both K, max
  resid ~7×10⁻⁷ → PASS — i.e. the old NS-only gate passes exactly as R2
  demonstrated, and the amended gate FAILS the table on both new
  legs.** Context points, same session: a severe collapse (σ=0.05)
  fails harder (6a=0.0141, 6b=0.9333, NS still 0/512 — NS is blind at
  every severity); the healthy frame-potential init **passes all three
  legs** (6a=1.0000, 6b=0.2832, 0/512 fallbacks) — the gate
  discriminates, it doesn't just reject; and a **localized** collapse
  (10 of 107 rows onto one direction, σ=0.02, planted on the healthy
  init) passes 6a (0.1787 — aggregate dilution, as R2 predicted) but
  is caught by 6b (0.9830 — a max statistic is structurally immune to
  dilution, R2 target 4(a)'s own reassuring finding, reproduced here
  with the pinned recipe). This regression case (all four tables, the
  seed, and the expected verdicts) ships with the build as a committed
  CPU test.
- **Residual gap, stated plainly (R2's ask):** Gate 1 remains a
  harm-screen only (its collapse sensitivity starts around severe
  collapse — R2 measured post-NS drift 0.8853 → predicted 0.6152 < 0.8
  at σ=0.05 — but not before); the amended Gate 2 catches init-time
  collapse; the per-checkpoint 6a/6b re-runs catch training-time
  collapse at ≤2,000-step latency; and **items 5/6 at final admission
  remain the evidentiary backstop** — the gates and checkpoint re-runs
  bound the *cost* of a collapse (≈ one checkpoint interval of one arm,
  ~0.03 GPU-h, vs. R2's estimate of one full wasted run under Rev 2),
  they do not replace the admission stack.
- **Adaptation from the orchestrator's Rev 2 phrasing, unchanged:** the
  gate is checkpoint-free at init because at λ=1 the blend is
  `normalize(A[key_ids])` — a pure function of the table — and the
  local archive carries result JSONs only, no weight checkpoints.
  Interior-λ blends are covered by Wave −1's standard smoke on
  realistic probe keys; the Rev 3 checkpoint re-runs now also cover the
  trained-table case R2 target 1(b) named.

**The K=32 simulator discrepancy — RESOLVED (Rev 3 supplement; the
Rev 2 "platform sensitivity" reading is retracted).** Rev 2 reported
that a CPU rerun of `simulate_recovery` reproduced the recorded K=16
prediction but not the K=32 one (0.0664 vs. recorded 0.7734 "at the
same c=0.9037") and attributed it to platform/RNG sensitivity; R2
deepened the investigation (seed sweep, 8× batch, fp64 — all ~0.07–0.10)
and hypothesized TF32. The coordinator's GPU re-measurement (archive:
`experiment-runs/2026-07-04_geo3_simulator_recheck/`; parent §16.7 now
carries the dated correction) found the true cause, a **shared-c bug**:
`geo3_drift_diagnostic.py::main()` extracts only K=16's drift and
`launch_read()` applies that one scalar to both K — the recorded 0.7734
was computed at **c=0.9416 (K=16's drift)**, not at K=32's own 0.9037.
With the correct per-K input the K=32 prediction is **0.06–0.09**, and
**GPU==CPU everywhere** — this design's own cliff sweep (c=0.9037 →
0.066, c=0.9423 → 0.746) already contained the explanation: the wrong
`c` sat on the far side of the steep K=32 response cliff, while K=16's
plateau made the same bug invisible there. Three consequences:
- **Gate 1 is unaffected** — K=16's launch-read used K=16's own drift,
  and its §16.7 validation (predicted 1.0000 vs. measured 0.9767)
  stands.
- **With correct inputs the mapping is CONSERVATIVE at K=32**: it
  underestimates measured recovery ~5–7× (predicted 0.06–0.09 vs.
  geo3's realized 0.4368). Any future K=32 simulator gate would
  therefore be a **strict/pessimistic** gate — a PASS would be strong
  go-evidence, but a FAIL would not distinguish "won't work" from "the
  idealized β=1 mapping is simply too pessimistic at K=32" — sized
  accordingly, K=32 simulator reads stay **non-gating** in this wave
  (Gate 2, a construction check with no simulator in the loop, remains
  the K=32 go/no-go).
- **The per-K drift-threading API fix is registered in THIS wave's
  build scope** (orchestrator decision, Rev 3 supplement):
  `geo3_drift_diagnostic.py::main()` and the `launch_read` signature
  move to a per-K drift dict (each K simulated at its OWN measured
  mean/p10), with a **unit test asserting the K=16 and K=32 calls
  receive different `c` values whenever the measured per-K drifts
  differ**. `keyanchor_drift_diagnostic.py` (the clone) inherits the
  fixed API by construction.

**The Rev 1 value-Gram calibration investigation (retained — items 1
and 3 still feed §3.5's Outcome-B2 evidence base; item 2's ceiling
estimate is retired at Rev 3, see its inline note).**
§16.7's original framing named the gap as: the registered simulator tilts *only* the value
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
   **~0.19 even at a perfect (`c=1.0`) drift fix. (REFRAMED, Rev 3
   supplement:** Rev 1 read the true value 0.4368 as falling "between
   the tilt-only overshoot and this correction's undershoot" — the
   "overshoot" side is now void (the 0.7734 was the shared-c bug's
   artifact, §3.4 caveat box); at correct per-K inputs the tilt-only
   mapping ALSO underestimates (0.06–0.09), so the measured 0.4368
   **exceeds every correctly-computed idealized β=1 prediction**, the
   whole CPU mapping family is conservative at K=32, and this item's
   "~0.19 ceiling" is **retired as evidence** — it survives only as a
   demonstration that the value-Gram-corrected mapping is
   over-pessimistic, never as a bound on what the wave can achieve.)
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
| Wave −1 (blocking smoke) | **8 short smoke probes** (itemized below — NEG1_PROBE_STEPS-class) **+ 2 drift-diagnostic probe runs** (K∈{16,32}, 5,000 steps each) = **10 runs** | 10 | ~0.7–1.0 | Mirrors geo3's own Wave −1 discipline (§14.6); Gate 1 launch-read reads the K=16 probe |
| **Reference arms (Rev 3 — §3.6, MANDATORY, first in manifest; n=2→3 at Rev 4 per decision 3)** | bare-geo3, seeds {1,2,3} × K∈{16,32}, 20,000 steps, per-checkpoint drift diagnostic active | 6 | **~5.0** | Bands pinned from their final checkpoints; anchor cells refuse to launch until `BANDS_PINNED.json` validates (§3.6 mechanical gate) |
| Wave 1 — candidate (d), PRIMARY, mandatory | K∈{16,32}×3 seeds×20,000 steps, learned λ | 6 | ~1.5–1.7 | Headline cells; §3.4 early-stop armed; §3.7 per-entity logging active |
| Wave 1 — candidate (c), ablation, always-run | K∈{16,32}×3 seeds, one `λ_anchor` (BLOCK-2 template) | 6 | ~1.5–1.7 | Comparison point (§2.4); early-stop armed |
| Fixed-grid λ diagnostic (conditional) | λ∈{0.3,0.6,0.9}, K=32 only ×1 seed each, fired only on an ambiguous §3.2 band | ≤3 | ~0.8 | Registered follow-up, not tuning — grid fixed now |
| Seed contingency (finding-5-style, one iteration) | +2 seeds, K=32 headline arm only | ≤2 | ~0.5–0.6 | Same add-seeds-not-steps discipline as §6/§14.10 |
| **Baseline total (mandatory, incl. reference arms)** | | 28 | **~8.7–9.4** | |
| **With both conditionals fired** | | ≤33 | **~10.0–10.8** | |
| Candidate (b), CONDITIONAL fallback | K∈{16,32}×3 seeds, only on (d)'s diagnosed anchor-collapse failure (§2.3, §6) | ≤6 | ~1.7–2.0 | Reserved, outside the baseline |
| **All-conditionals-max** | | ≤39 | **~11.7–12.8** | |

**Fit (Rev 4).** Mandatory baseline ~8.7–9.4 GPU-h ≤ the orchestrator's
**≤10 nominal** (the n=3 reference upgrade consumed most of the
headroom, deliberately — R3's Judgment Call (c) priced it and both
ceilings still clear); all-conditionals-max ~11.7–12.8 sits inside the
**≤15 reserve** band (everything past 10 is conditional: the λ-grid,
the seed contingency, and candidate (b), the explicitly-reserved
fallback). Combined with F-geo-3's realized ~1.67 and the exactness
program's measured cumulative spend (R2 target 6, re-verified by R3:
**34.9 GPU-h** summed from all archived `wall_s` fields), the worst
case projects to ~48 GPU-h program-total — nowhere near the 80 GPU-h
cap. The §3.4 early-stop can only reduce these figures (a killed arm
stops at ~20% of its run cost).

**Wave −1 smoke suite, itemized (Rev 3 — R2 m3: a countable list, not
an asserted count, matching the parent §14.6's own discipline):**

1. **Anchor init load + Gate-2 legs at init** — the frozen
   frame-potential table loads with the registered seed; G2-a/b/c pass
   on it; the §4 regression case (all four pinned tables) produces its
   expected verdicts.
2. **Blend forward/backward** — `bind()` with `anchor_active=True`:
   finite loss, finite gradients on every parameter INCLUDING the
   anchor table and the λ raw-param, at a realistic and an adversarial
   (near-duplicate raw keys) input.
3. **Held-out bypass bit-identity** — all-held-out batch: `bind()`
   outputs strictly `torch.equal` to the anchor-disabled path (§3.3,
   valid under the Rev 4 gather/scatter construction).
4. **Held-out gradient isolation (BROADENED at Rev 4 — R3 finding 4,
   orchestrator decision 4)** — mixed-split batch backward, run on the
   SAME adversarial near-duplicate input smoke 2 specifies (not only a
   generic batch), **with NaN injected into every held-out anchor
   row**: assert (i) `torch.isfinite` on ALL parameter gradients — not
   only exact-zero at held-out rows; (ii) anchor-table gradient exactly
   zero at every held-out row; (iii) held-out output rows strictly
   `torch.equal` to pure-geo3 at identical weights despite the planted
   NaNs — the registered unit test proving the gather/scatter
   construction has no 0×NaN path in either direction (§2.2).
5. **λ logging** — trajectory field present at every logged step; the
   startup cadence assertion (`log_every == LAMBDA_LOG_CADENCE_STEPS =
   200`, §3.2) fires on a deliberately mis-set cadence (negative
   control); final-window summary fields (final value / last-5-point
   mean / last-5-point range) present and consistent with the
   trajectory (§3.2).
6. **Item-5 instrument** — the pre-NS side channel
   (`anchor_last_k_blend_raw`) is populated, detached, correct shape,
   and differs from post-NS `k_eff_items` on a generic batch (they are
   different tensors by construction — assert it, don't assume it).
7. **Item-6 checkpoint wiring** — 6a/6b computed at a live checkpoint
   on the current table; the pinned collapsed table substituted in
   place of the real one must FAIL both (negative control wired into
   the harness, not just the design doc).
8. **Per-entity anchor-input-alignment instrument (§3.7)** — the
   full-pool sweep returns 107 values in [−1, 1]; on the healthy init
   at fixed λ=1 the values are ≈1 by construction (the blend IS the
   anchor row); per-checkpoint logging present; the h=1 behavioral
   companion field (Rev 4) present with 107 entries at the final
   checkpoint of the probe run.
9. **Override-demotion stamping (NEW at Rev 5 — round-4 prescription;
   CPU, in-process/dry-context — NOT a GPU probe run, so the "8 GPU
   smoke probes + 2 drift probes = 10 Wave −1 runs" count and the §5
   budget row are unchanged, same class as Gate 2's own CPU checks).**
   Two asserts, mirroring `lm_pretrain_rd.py`'s own in-process
   `claim_tier` smoke (L1420): (i) invoke the launcher gate with
   `--unblind-override` in a dry context (no `BANDS_PINNED.json`
   present, sandbox dir) and assert the override path returns the
   stamp payload (timestamp included) that `build_cmd` threads into
   the spawned run's flags; (ii) invoke the runner's result assembly
   with and without the stamp, write each assembled result to a temp
   JSON, re-load, and assert the fields land: override case →
   `claim_tier == "descriptive"`, `unblind_override == true`,
   `unblind_override_at` present; non-override case →
   `unblind_override == false` present (never absent). A missing
   field in either case is a smoke FAILURE — the audit trail is
   load-bearing, not decorative (§3.6).

---

## 6. Falsification map

| Candidate | Kills it |
|---|---|
| (a) i-strong (reference) | N/A — already succeeded within its pool-restricted, three-concession scope (§2.1); it sets an existence proof, not this wave's target |
| (d) primary (learned-λ blend) | λ in the <0.05 band for ≥2/3 seeds under §3.2's three-part rule (anchoring not recruited) **or** persistent λ oscillation (trailing-window range ≥0.1 → ambiguous band, no headline, fixed-grid diagnostic fires — §3.2, Rev 3) **or** pre-NS drift <0.95 even at the fixed-grid λ=0.9 cell (item-5 fail — the blend cannot stabilize its own input, Outcome C) **or** per-entity engagement <50% regardless of aggregate drift (§3.7, Rev 3 — Outcome C) **or** anchor-table collapse (items 6a/6b fail at a checkpoint or at final admission — routes to candidate (b), the gradient-free anchor) **or** the §3.4 early-stop fires ('value-geometry-bound') |
| (b) EMA fallback | Anchor never stabilizes under the pinned numeric criterion (trailing-1,000-step relative Frobenius change ≥1e-3 past step 10,000, §2.3) — measured, not adjectival; distinct from (d)'s failure modes by construction (no gradient into the anchor) |
| (c) soft `L_anchor` | Saturates like F-geo-1/2 — pre-NS drift improvement small, gain shrinking geometrically per hop (`~ε^h`), the exact §15.3/§15.4 signature |

**If ALL candidates fail with the manipulation check passing — pre-NS
drift ≥0.95 (item 5) but h4 <0.5 everywhere (Outcome B) — what gets
re-attributed, via §3.5's pre-registered disambiguation:**

- **B1 (post-NS sanity below the §3.6-derived engaged threshold — not
  mechanically engaged):** the failure re-attributes to the
  **coherence-floor residual** — the F2 packing geometry means input
  stability does not purchase output stability past the computed
  ~0.9423 ceiling, and the realized input→output stability transfer was
  worse than the ceiling computation's idealized version. Named
  follow-on: a **pool-restricted** anchor cell (≤64 train entities —
  deliberately trading the full-pool scope for geometry, with the
  i-strong precedent as the limit case) or a `d_state=128` rider (the
  parent design's own §9 K=d/2 boundary question, where 107 < 128 would
  dissolve the packing obstruction entirely). Not run inside this
  wave's budget.
- **B2 (post-NS clears the derived engaged threshold — engaged,
  near-ceiling, still missing):** the outcome-F attribution is
  incomplete — **value-side geometry is a second, independently binding
  constraint** (mechanism (b) in the parent design's §2 taxonomy),
  supported by the **simulator-independent evidence only** (geo3's
  1.3–1.8× elevated value-Gram; the intervention-broken key/value
  correlation — Rev 3: the "§16.7 overestimate" narrative is void per
  the shared-c bug correction, and the Rev 1 "~0.19
  corrected-simulator ceiling" is retired as evidence, §3.4 caveat
  box/§4). Named follow-on: a **value-anchoring / joint key+value
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
   **Closed by a three-part co-firing requirement, each part verified to
   have teeth (Rev 2, attack F1) — and hardened at Rev 3 (R2 target 1,
   decision 2):** item 5 (pre-NS drift ≥0.95) is trivially satisfiable
   at λ→1 and by a collapsed table (measured: collapsed-table pre-NS
   drift ≡ 1.0000) — alone it proves nothing; item 6 (raw-table
   `σ_64/σ_1 ≥0.1` AND max|cos| ≤0.5) catches collapse directly
   (negative controls: severe 0.0147/0.9263 and the pinned moderate
   regression case 0.0762/0.5371, both FAIL, while the old post-NS
   check reads a blind 1.0000 on both), **and now fires at three
   times — inside the launch gate (G2-a/b), at every admission
   checkpoint during training, and at final admission** — closing R2's
   demonstrated moderate-collapse blind spot at bounded (~0.03 GPU-h)
   worst-case latency; the §3.2 λ bands (with the Rev 3 oscillation
   exclusion) cap the λ→1 route at existence tier regardless of every
   other number. A headline requires all three simultaneously; no
   single instrument is load-bearing alone.
3. **Seed cherry-picking.** Inherited unchanged from the parent design's
   finding-5/§14.10 discipline: 3/3 admissible for the full tier,
   add-seeds-not-steps contingency (+2, one iteration), failing arms at
   descriptive tier. The §3.2 band assignment uses the same ≥2/3
   convention, pinned before data.
4. **Simulator overfit — trusting a simulated prediction over the
   behavioral bar.** **Closed by:** no simulator number gates K=32
   (Gate 2 is a construction check with no simulator in the loop); the
   K=32 discrepancy is RESOLVED at Rev 3 as the shared-c bug (§3.4
   caveat box/§4 — the recorded 0.7734 was computed at K=16's drift;
   with correct per-K inputs the mapping is conservative at K=32,
   underestimating ~5–7×), the per-K threading fix + unit test is in
   this wave's build scope, and §3's bars are read only against
   measured `rec@0.9`.
5. **K=16 regression masking.** **Closed by:** §3's numeric
   no-regression guard (K=16 h=4 within −0.02 of geo3's 0.9767) is a
   hard admissibility item at every headline comparison — a candidate
   clearing K=32 while breaching it is not admitted as an improvement.
6. **Budget-ceiling provenance (Rev 1 self-flag).** Resolved: the attack
   round's own grep confirmed the 300 GPU-h figure belongs to
   `SCALE_TRANSFER_DESIGN.md`, and the orchestrator has pinned this
   wave's ceilings (≤10 nominal / ≤15 reserve under the program's 80
   cap, §5). Closed.
7. **Aggregate masking — a passing pooled statistic carried by a
   minority of strongly-anchored entities (NEW at Rev 3 — R2 target
   4(b)'s constructed scenario).** **Closed by:** §3.7's per-entity
   anchor-input-alignment readout (with its Rev 4 input-side-proxy
   disclosure and h=1 behavioral companion) — the full 107-entity
   alignment vector is a
   required result field, `engaged_frac < 50%` routes to Outcome C
   regardless of every aggregate number, and [50%, 90%) caps the claim
   at the named partial outcome (A″), never the headline. The
   companion *geometric* version of the same attack (a localized
   anchor-row collapse hiding inside aggregate conditioning stats) is
   independently caught by item 6b's max statistic (measured on the
   pinned localized case: 6a passes at 0.1787, 6b fails at 0.9830 —
   dilution-immune, R2 target 4(a)).

---

## 8. Revision history — finding → change maps

### 8.1 Rev 2 — attack-round-1 responses

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

*(Rev 3 note on two §8.1 rows: the M2 row's closing "platform/RNG-
sensitive" sentence and the M3 row's "~0.19 corrected-simulator
ceiling" justification are superseded — see §8.2 rows R2-M3 and the
§3.4 caveat box; the tables above are preserved as the historical
record of what Rev 2 believed, per this project's revision-map
convention.)*

### 8.2 Rev 3 — attack-round-2 responses

The round-2 adversarial review of Rev 2
(`KEY_ANCHORING_ATTACK_R2.md`, 2026-07-03) returned **NEEDS-REV-3**:
all three R1 FATALs verified CLOSED by independent from-scratch
reproduction (the K=32 λ=1 ceiling matched at 0.9424 vs. 0.9423,
seed-stable), plus **5 new MAJOR and 3 MINOR findings**, several
demonstrated with constructed counterexamples. Disposition: **every
finding accepted; contested points resolved by binding orchestrator
decisions 1–7 and implemented as directed**; one finding (R2-M3, the
simulator discrepancy) was **resolved mid-revision by the
coordinator's own GPU re-measurement**, which disconfirmed both Rev
2's platform-sensitivity reading and R2's TF32 hypothesis and found a
shared-c bug instead — folded in per the supplementary instruction.

| # | Finding (condensed) | Change made | Where |
|---|---|---|---|
| R2-M1 | MAJOR — constructed counterexample: a *moderately* collapsed table (σ_ratio≈0.099, max\|cos\|≈0.538 — already item-6-failing) sails through BOTH Rev 2 pre-spend gates (Gate 2 NS leg 0/512 fallbacks; Gate 1 predicted 0.9922 ≥ 0.8); Gate 2 also never sees the table after SGD starts updating it | Orchestrator decision 2: **items 6a/6b promoted INTO Gate 2** (three legs, all mandatory, any failure blocks launch) **and re-run at every admission checkpoint** during training (107×64 SVD per checkpoint, negligible cost, ≤2,000-step detection latency). **Pinned regression case run this session** (2 dirs, noise=0.30, torch seed 42): G2-a = 0.0762 FAIL, G2-b = 0.5371 FAIL, NS leg 0/512 PASS — the amended gate demonstrably fails the table the old gate passed; healthy init passes all three legs (1.0000/0.2832/0 fallbacks); severe (0.0141/0.9333) and localized (6a 0.1787 pass / 6b 0.9830 fail — max-statistic dilution-immunity, R2 target 4(a)) context cases pinned alongside; ships as a committed CPU test. Residual gap stated plainly: Gate 1 stays a harm-screen; items 5/6 at final admission remain the evidentiary backstop | §4 (Gate 2), §7 item 2, §5 (smoke 1/7) |
| R2-M2 | MAJOR — the 0.9037/0.9416 "trained" reference drifts are single-seed, 5,000-step-probe measurements (`geo3_drift_diagnostic.py` trains a fresh throwaway model, never loads the real checkpoints); no seed-variance data exists anywhere; the "engaged vs. lucky seed" separation cannot be made | Orchestrator decision 4, first branch checked and CLOSED NEGATIVE this session (no drift/pairwise/resample field in any archived run JSON; the diagnostic records `seed=0` only) → second branch implemented: **4 reference arms** (bare-geo3, seeds {1,2} × K∈{16,32}, 20,000 steps, per-checkpoint drift active, ~3.3 GPU-h) join the manifest FIRST; **bands derived from measured final-checkpoint mean + spread** (`engaged_K = mean_ref + 2·s_ref`, degenerate-case guard vs. the computed ceiling → UNRESOLVABLE) and recorded in a `BANDS_PINNED` block **before any anchor-arm JSON is opened**; provisional 0.92/0.96 declared void on derivation; ordering/blinding protocol spelled out | §3.6 (new), §3.1, §3.5 (B1/B2), §5 |
| R2-M3 | MAJOR — the §16.7 "simulator overestimates K=32 (0.7734) because of a missing value-Gram term" narrative rests on a GPU number R2's CPU reruns (fp32+fp64, seeds, 8× batch) consistently contradict | **RESOLVED mid-revision (orchestrator supplement): a BUG, not platform sensitivity** — `main()` extracts only K=16's drift and `launch_read()` applies that one scalar to both K; the 0.7734 was computed at c=0.9416 (K=16's drift), reproduces to the last digit; at K=32's own 0.9037 the prediction is 0.06–0.09 (an UNDERESTIMATE of 0.4368); GPU==CPU; R2's TF32 hypothesis disconfirmed. Parent §16.7 corrected by the coordinator from the GPU measurement (archive: `experiment-runs/2026-07-04_geo3_simulator_recheck/`) — NOT touched by this design. Here: §3.4 caveat box states the true cause; Gate 1 confirmed unaffected (K=16 used its own drift); with correct inputs the mapping is **conservative at K=32** (~5–7× underestimate), so any K=32 simulator gate would be strict/pessimistic — K=32 reads stay non-gating; **the per-K drift-threading API fix + different-c unit test is registered in THIS wave's build scope**; B2's evidence base reduced to simulator-independent measurements and the Rev 1 "~0.19 ceiling" retired as evidence | §3.4 (caveat box), §3.5 (B2), §4, §6 (B2), §7 item 4, Reproducibility (build list) |
| R2-M4 | MAJOR — zero per-entity visibility in the training-eval pipeline: a passing pooled aggregate could mask a mechanism engaged for a subset of entities, invisible to items 5 and 6 alike | Orchestrator decision 5: **per-entity anchor-alignment logging required** (all 107 train entities, ≥8 resamples each, every admission checkpoint; full vector a required result field); **partial-anchoring readout pre-registered**: `engaged_frac` = fraction with `a_e ≥ 0.9` at final step; bands pinned — **≥90% headline-eligible** (a requirement of Outcome A), **[50%, 90%) = Outcome A″ "partial anchoring"** (named, reported, no headline), **<50% = not recruited regardless of aggregate drift** (routes to Outcome C) | §3.7 (new), §3.5 (A/A″/C), §6, §7 item 7, §5 (smoke 8) |
| R2-M5 | MAJOR-leaning-MINOR — λ's claim-tier band gates on the final-1,000-step mean only; an oscillating trajectory whose mean lands in-band would be labeled "learned interior anchoring" (the same gap candidate (b)'s m4 fix closed for its EMA) | Orchestrator decision 1: per-seed band assignment now requires **final value in band AND trailing-1,000-step mean in the same band AND trailing-window range < 0.1**; any failure → ambiguous band, a pre-registered exclusion; the three summary statistics are machine-readable result fields | §3.2, §2.2 (logging bullet), §6 |
| R2-m1 | MINOR — "more than half the distance" claim for the K=32 0.92 threshold is arithmetically wrong (42.2%, not >50%) | Corrected in text (gap 0.0386, halfway 0.9230, 0.92 = 42.2%; the K=16 figure, 55.9%, was correct but unclaimed) — and mooted in practice: the provisional bands are superseded by §3.6's derived bands per decision 4 | §3.1 |
| R2-m2 | MINOR — the C17 bypass "bit-identical" claim is not universally exact: re-normalizing an already-unit-norm fp32 vector perturbs 337/1000 rows by up to 1 ULP; a strict-equality smoke would flake on correct code | Orchestrator decision 7, bit-exact branch chosen: the blend is rebuilt as a **`torch.where` select** — held-out rows pass through with ZERO arithmetic (no multiply-by-zero, no re-normalize), so strict `torch.equal` is again the correct assertion and a smoke failure now means a real routing bug; R2's ULP measurement quoted as the rationale | §2.2 (code block), §3.3, §5 (smoke 3) |
| R2-m3 | MINOR — the Wave −1 smoke suite was prose, not an enumerable list (count of "8" asserted, unlike the parent §14.6's itemized discipline) | Itemized as a numbered 8-item list (init+gate legs / blend fwd-bwd / bypass bit-identity / held-out zero-grad / λ logging / item-5 instrument / item-6 checkpoint wiring incl. the in-harness negative control / per-entity instrument) | §5 |
| R2-t5 | MINOR-to-MAJOR — the early-stop AND rule conditions the kill on its noisier leg (h4: 58% relative seed spread at step 2000 vs. value-Gram's ~15%), enabling the "wobbles above the floor" false-continue; the cost-asymmetry rationale was never stated | Orchestrator decision 3: **value-Gram is the sole load-bearing kill leg, required at TWO consecutive checkpoints (2,000 AND 4,000)**; h4 recorded and reported, explicitly non-load-bearing, the 58%-vs-15% spread disclosed inline; cost-asymmetry rationale stated; the K-asymmetry of the two-checkpoint rule disclosed (geo3's own K=32 value-Gram already exceeds 3.90 at step 4000 in 2/3 seeds — the second leg is a transient-spike guard there, a genuinely geo3-relative test at K=16) | §3.4 |

*(Rev 4 note on three §8.2 rows, mirroring §8.1's own convention: the
R2-M2 row's "4 reference arms / seeds {1,2} / ~3.3 GPU-h" figures are
superseded by §8.3 row R3-3 (6 arms, seeds {1,2,3}, ~5.0 GPU-h); the
R2-M5 row's step-space window phrasing is superseded by §8.3 row R3-1
(log-point space); the R2-m2 row's `torch.where` construction is
superseded by §8.3 row R3-4 (masked gather/scatter). The tables above
are preserved as the historical record of what Rev 3 registered.)*

### 8.3 Rev 4 — verify-round-3 responses

Round 3 (`KEY_ANCHORING_ATTACK_R3.md`, 2026-07-04) returned
**NEEDS-REV-4**: all 9 R2 findings confirmed closed (the Gate-2
regression quadruple reproduced to 4 decimal places from an
independently-reconstructed recipe; all 12 early-stop archive numbers
exact; the shared-c bug verified at the source-line level against the
GPU recheck archive, including `tf32_matmul: false` directly falsifying
R2's TF32 hypothesis; the budget re-derived exactly, 34.90 GPU-h
program spend confirmed) — plus **3 MAJOR and 2 MINOR findings, all
inside Rev 3's own new machinery**. Disposition: **all five accepted
and implemented under binding orchestrator decisions 1–5; everything
else frozen per the same instruction** (in particular, R3's optional
K=32 single-checkpoint-kill efficiency suggestion — adjudicated
"acceptable as disclosed, not required" by R3 itself — and the
localized-case seed-pin provenance note — "not a new finding" per R3 —
were both left as-is). This is the expected final revision round; a
bounded verify pass on only these five items decides
CLEARED-FOR-BUILD.

| # | Finding (condensed) | Change made | Where |
|---|---|---|---|
| R3-1 | MAJOR — the λ oscillation window ("trailing 1,000 steps") is ambiguous against the harness's real 200-step log cadence (verified from an archived trajectory: 101 points/run): read as steps it holds 5–6 samples (low power, unacknowledged); read as "1,000 logged samples" it silently consumes the whole trajectory including early training — and it sits on the headline Outcome-A gate path | Orchestrator decision 1: window pinned in **LOG-POINT space** — the last **5 logged points** (= 1,000 steps at the registered cadence); `LAMBDA_LOG_CADENCE_STEPS = 200` and `LAMBDA_WINDOW_LOG_POINTS = 5` registered as constants; **harness asserts the cadence at startup** (a cadence change fires the assertion instead of silently resizing the window — negative-control smoke added); the 5-point window's power stated honestly (a gross-oscillation catch: swings ≥0.1 persisting at cadence; sub-cadence oscillation unobservable; excluded seeds land in the ambiguous band whose registered follow-up needs no trajectory statistics) | §3.2, §2.2 (logging bullet), §5 (smoke 5) |
| R3-2 | MAJOR — the §3.6 blinding protocol was a stated norm, not a mechanical gate: nothing prevented an operator seeing anchor progress pre-pin, and nothing forced `BANDS_PINNED` to be computed by a process that cannot access anchor results | Orchestrator decision 2 — three registered build requirements with failure modes: **(a)** the harness writes `BANDS_PINNED.json` (derived bands + per-seed inputs + formula version + sha256 hashes of all reference-arm JSONs + timestamp) only after every reference arm validates complete (any incomplete arm → no file → everything blocked); **(b)** anchor cells **REFUSE to launch** without an existing, hash-validating `BANDS_PINNED.json` (gate_geo3_drift's loud-refusal pattern; `--unblind-override` exists but auto-demotes every anchor readout to descriptive tier) — supersedes Rev 3's "may train concurrently" allowance, sequencing now launcher-enforced; **(c)** the readout script asserts the pin timestamp strictly precedes the earliest anchor-arm start time (violation → readout aborts, blind recorded broken, descriptive tier only) | §3.6 (rewritten protocol), §5 (table note) |
| R3-3 | MAJOR — the n=2 band formula is statistically fragile: at n=2 the sample std's own relative standard error is ≈71%, so `engaged_K` is dominated by which two seeds are drawn — the "engaged vs. lucky seed" problem reappearing inside the fix built for it; can both inflate B2 confidence and spuriously trip the UNRESOLVABLE guard | Orchestrator decision 3 (the verifier's own fix, adopted): **3 reference seeds per K** (6 reference arms, +1.6–1.7 GPU-h → reference row ~5.0; mandatory baseline ~8.7–9.4, still ≤10 nominal; all-max ~11.7–12.8, ≤15 reserve); formula recomputed at n=3 (`engaged_K = mean_ref + 2·s_ref`, df=2, s-RSE ≈50%); UNRESOLVABLE guard trigger recomputed on the n=3 statistic, now accompanied by a mandatory **leave-one-out sensitivity report** (disclosure-only) so an outlier-driven trigger is visible as such | §3.6, §5 (budget table + fit) |
| R3-4 | MINOR — `torch.where` evaluates BOTH branches: a non-finite value in the discarded blend branch for a held-out row (adversarial near-zero-vector regime) could poison gradients via 0×NaN even though the forward select is bit-clean; smoke 4 lacked smoke 2's adversarial qualifier | Orchestrator decision 4: blend construction moved to **masked gather/scatter** — arithmetic touches trained rows ONLY, held-out rows are a bit-exact `clone` with no graph edge in either direction (§3.3's strict-equality smoke stays valid); **registered NaN-injection unit test** folded into smoke 4: NaNs planted in every held-out anchor row, forward+backward on smoke 2's adversarial input, assert all gradients finite (`torch.isfinite` broadly, not only exact-zero), held-out gradient exactly zero, held-out outputs `torch.equal` to pure-geo3 | §2.2 (code block), §3.3, §5 (smoke 4) |
| R3-5 | MINOR — §3.7's metric measures input-side alignment, not behavioral per-entity engagement; `M3_held_out` stays pooled, so "input pulled to anchor but NS mixing doesn't stabilize the written key" stays invisible; the correct-object choice (Judgment Call (b)) is affirmed but the scope limit was undisclosed | Orchestrator decision 5: metric **renamed anchor-input-alignment** (kept as the registered `engaged_frac` driver — R3 affirmed it is the correct object for R2's ask); **non-load-bearing behavioral companion added**: per-entity h=1 recovery restricted to eval episodes containing that entity (bookkeeping on existing eval scores, no new forward passes), reported alongside for all 107 entities, explicitly diagnostic-only; scope-limit disclosure added mirroring §3.3's precedent (input-side proxy; per-entity h=4 too sparse at this eval budget; a visible a_e-vs-h1 divergence reports as an open question, never adjudicated by a hidden rule) | §3.7, §3.5 (A/A″/C wording), §5 (smoke 8), §7 item 7 |

### 8.4 Rev 5 — bounded-verify-round-4 response (single field)

Round 4 (appended to `KEY_ANCHORING_ATTACK_R3.md`, 2026-07-04, on
commit `7349f97`) confirmed all five R3 items CLOSED — R3-4 via an
executable reproduction of the gather/scatter NaN test **plus a
contrast run proving the superseded `torch.where` form produced
non-finite gradients** (the fix is the mechanism, not a rename); R3-1
verified against the harness's real `log_every=200` default with a
no-regression check on the pre-existing self-tests; R3-2's mechanical
gate judged "stronger than the literal ask"; the §5 budget re-derived
exactly. **One blocker (NEEDS-REV-5), prescriptive fix, implemented as
exactly one field and nothing else:**

| # | Finding (condensed) | Change made | Where |
|---|---|---|---|
| R4-1 | BLOCKER — the `--unblind-override` demotion was recorded only in "the wave summary"; this repo's own audit methodology reads individual result JSONs file-by-file, its own `lm_pretrain_rd.py::_assemble_result` precedent (L1090, smoke-asserted L1420, field confirmed in every archived Track-C JSON) writes `claim_tier` into every run's own result, and the `--accept-gate-override` pattern §3.6 cited instead has an empirically-confirmed track record of leaving zero trace in individual run JSONs — an overridden run's numbers could later be cited at evidentiary tier by anyone reading the JSON in isolation | Per the orchestrator's prescription: **(1)** §3.6's launcher-gate item now registers that `--unblind-override` threads a stamp to every spawned anchor run, whose result assembly writes `claim_tier: "descriptive"` + `unblind_override: true` + `unblind_override_at: <timestamp>` into that run's own result JSON at assembly time, never post-hoc, in addition to the wave summary; non-override runs always write `unblind_override: false` so field absence is never ambiguous (`claim_tier` itself is written only on the override path — non-override tiers remain readout-time verdicts per the admission stack); **(2)** new §5 smoke 9 (CPU, in-process/dry-context, mirroring `lm_pretrain_rd.py`'s own L1420 claim-tier smoke — not a GPU probe, so the 10-run Wave −1 count and budget row are untouched): dry-context override launch returns the stamp payload, and result assembly with/without the stamp lands the correct fields in a written-and-reloaded JSON, missing field = smoke FAILURE; **(3)** this map row. Everything else frozen | §3.6 (launcher-gate item), §5 (smoke 9), header |

---

## 9. Wave results (2026-07-05) — VERDICT

**The wave ran to completion.** 6 reference arms (bare geo3, fresh seeds
{1,2,3}×K∈{16,32}, `--drift-probe` active) + 6 candidate-(d) cells
(learned-λ blend, seeds {0,1,2}×K∈{16,32}) + 6 candidate-(c) cells (soft
`L_anchor`, same seeds/K) all completed 20,000/20,000 steps, `complete:
true`, zero timeouts — 18 mandatory cells, 10.98 realized GPU-h (vs. the
§5 estimate of ~8.7–9.4 for this subset), plus 8 short GPU probes, the
CPU smoke suite (10/10 items PASS, including the Rev-5 override-demotion
smoke), and Gate 2 (PASS: G2-a σ-ratio 1.0000, G2-b max\|cos\| 0.2842,
G2-c zero fallbacks at both K over 512 subsets; the full pinned
regression quadruple reproduced its expected verdicts to 4 decimals).
`BANDS_PINNED.json` was written and validates; the readout script
(`readout_keyanchor.py`, re-run this session, CPU-only) confirms **BLIND
INTACT** — the pin (`2026-07-04T22:49:31Z`) strictly precedes the
earliest anchor-arm start, 12 runs checked — and **no run carries
`unblind_override: true`** (the override path was never invoked).

### 9.1 λ verdicts (sec 3.2) — all six candidate-(d) seeds land INTERIOR

| K | seed | λ final | λ mean (last 5 logged pts) | range | band |
|---|---|---|---|---|---|
| 16 | 0 | 0.5615 | 0.5615 | 0.0002 | interior |
| 16 | 1 | 0.5448 | 0.5446 | 0.0007 | interior |
| 16 | 2 | 0.5582 | 0.5586 | 0.0008 | interior |
| 32 | 0 | 0.5751 | 0.5746 | 0.0009 | interior |
| 32 | 1 | 0.5682 | 0.5676 | 0.0010 | interior |
| 32 | 2 | 0.5701 | 0.5697 | 0.0008 | interior |

Arm-level label (≥2/3 rule): **interior at both K, 3/3 seeds each** —
headline-eligible band, not pin-rediscovery, not ambiguous. The
oscillation guard passes trivially (ranges 0.0002–0.0010, two orders of
magnitude under the 0.1 exclusion bar) — SGD settles on a tight,
seed-stable ≈0.55–0.58 blend weight at both K, not 0 (not-recruited) and
not >0.95 (pin rediscovery). Candidate (c) has no λ (fixed
`lambda_anchor=1.0`, a loss-side weight, not a blend fraction).

### 9.2 `BANDS_PINNED.json` — derived engaged thresholds (sec 3.6, n=3)

| K | mean_ref | s_ref | engaged_K | ceiling | UNRESOLVABLE? |
|---|---|---|---|---|---|
| 16 | 0.9289 | 0.0076 | 0.9440 | 0.9745 | No |
| 32 | 0.8746 | 0.0059 | 0.8864 | 0.9423 | No |

Both K resolvable (engaged_K sits comfortably below its ceiling minus
0.005); leave-one-out sensitivity is tight (K=16: 0.9364–0.9502; K=32:
0.8795–0.8884) — no single reference seed drives the threshold.

### 9.3 THE GAP — items 5, 6 (final admission), and sec 3.7's `engaged_frac` were never measured on the admitted Wave-1 cells

This is the load-bearing finding of this verdict, surfaced by direct
inspection of the result JSONs and the harness source, not assumed:

- **`keyanchor_wave1_manifest()` (`run_deltanet_rd_exactness_sweep.py`
  L361–380) never passes `drift_probe=True`** to `_spec(...)` for
  candidate (d) or (c) — only `reference_arms_manifest()` (L307–324)
  does. Confirmed by direct JSON inspection: none of the 12 Wave-1
  result JSONs contains a `drift_probe` key anywhere (checked by
  recursive string search), so **item 5's pre-NS blend drift
  (`anchor_last_k_blend_raw`) was never logged**, and no per-checkpoint
  item-6 table conditioning or sec 3.7 per-entity alignment sweep exists
  either — the only place those computations are wired into the running
  code is (a) Gate 2's one-time INIT-only check (`gate2_construction_test.py`,
  which only ever sees the frozen table, never the trained one) and (b)
  `keyanchor_drift_diagnostic.py`, a standalone GPU driver that trains
  its own **separate, fresh 5,000-step probe model** — never the real
  20,000-step admitted arms. The `keyanchor_wave1_manifest()` docstring's
  own claim ("both candidates' per-entity/lambda logging is active by
  construction") is **only true for λ; false for per-entity/drift
  logging** — a real discrepancy between the design's requirement
  (§2.2/§3.1/§4/§3.7: item 5 + item 6 checkpoint re-runs + `engaged_frac`
  are all specified as "at every admission checkpoint") and what the
  build actually wired into the training loop.
- **The one artifact that was supposed to supply this (and Gate 1's
  pre-spend launch-read) crashed on its first invocation and was never
  fixed or re-run.** `logs/04_drift_diag.log`: `keyanchor_drift_diagnostic.py
  --k 16 32` measured the K=16 at-init drift (0.7466 pre-training — not
  informative alone) then crashed immediately when starting the 5,000-step
  probe train: `probe_steps // 5 = 1000` was passed as `log_every`, which
  fails the harness's own `assert_lambda_log_cadence` (registered
  `LAMBDA_LOG_CADENCE_STEPS = 200`, the Rev-4/R3-1 fix from this SAME
  design) — a self-inconsistency inside the wave's own tooling, not an
  external cause. `results/deltanet_rd_exactness/keyanchor_drift/` does
  not exist; no `wave_neg1_drift.json` was ever produced. The
  `keyanchor_chain.sh` header states "any failure stops the chain and
  surfaces" (`&&`-gated), yet `waveref`/`wavekeyanchor` both completed
  hours later — the chain was continued past this failure by hand
  (or via separate direct invocations of the later `--wave ref` /
  `--wave keyanchor-bands` / `--wave keyanchor` stages), not by the
  script as written. Net effect: **Gate 1's pre-registered pre-spend
  gate (predicted K=16 h=4 ≥ 0.8) was silently never computed**, and the
  smoke-8 console output for this same session says so explicitly:
  *"the full through-real-episodes sweep + h=1 behavioral companion
  (keyanchor_drift_diagnostic.py) require model.bind()/CUDA — out of
  this CPU-only smoke's scope; run on GPU as a follow-up (scrutiny
  item)."* That follow-up was never done.

**Consequence for §3.5's outcome map: Outcome A/A″/A′/B/C cannot be
mechanically assigned.** Every branch of the map is keyed on item 5
and/or `engaged_frac`, neither of which exists for the actual admitted
runs — this is not a "fails the bar" result (which would route to
Outcome B or C), it is a **missing measurement**, and per this design's
own §6 rule ("a run that clears h4 ≥ 0.5 without clearing pre-NS drift
≥ 0.95 does NOT count as a test of this design's hypothesis... the
write-up must say so, not claim confirmation"), the honest reporting
tier is **descriptive, pending re-measurement** — not because anything
failed, but because the mechanistic proof the outcome map requires was
never produced by the code that ran.

### 9.4 What IS traceable from the JSONs — the behavioral result

Despite the gap above, every quantity NOT gated behind `drift_probe` is
fully present, 3/3 admissible, zero fallbacks, at both K, for all three
arms (reference/d/c):

**`rec@0.9` at final checkpoint (`M3_held_out`), mean [min, max] over 3 seeds:**

| Arm | K | h=4 | h=5 | h=7 |
|---|---|---|---|---|
| reference (fresh bare geo3) | 16 | 0.9716 [0.9525, 0.9812] | 0.8852 [0.8420, 0.9158] | 0.5776 [0.5475, 0.6049] |
| reference (fresh bare geo3) | 32 | **0.4105** [0.3906, 0.4233] | 0.1478 [0.1469, 0.1485] | 0.0217 [0.0181, 0.0258] |
| candidate (c), soft `L_anchor` | 16 | 0.9987 [0.9985, 0.9988] | 0.9723 [0.9640, 0.9803] | 0.7344 [0.7017, 0.7642] |
| candidate (c), soft `L_anchor` | 32 | 0.4806 [0.3994, 0.5241] | 0.1336 [0.1121, 0.1484] | 0.0068 [0.0064, 0.0073] |
| **candidate (d), learned-λ blend** | 16 | 0.9997 [0.9995, 1.0000] | 0.9901 [0.9886, 0.9918] | 0.8404 [0.8351, 0.8506] |
| **candidate (d), learned-λ blend** | **32** | **0.6132 [0.5590, 0.6647]** | 0.2061 [0.1823, 0.2314] | 0.0120 [0.0112, 0.0126] |

**K=32 headline: candidate (d) clears the ≥0.5 bar in 3/3 seeds** (0.559,
0.615, 0.665 — all comfortably above bar), a **+0.20 absolute / +49%
relative** lift over the freshly-reproduced reference mean (0.4105) and
+0.176 over the prior archived pin-free admissible-tier figure (0.4368,
§16.4) — the fresh reference arms reproduce that old number within seed
noise (0.3906–0.4233 vs. 0.4368, consistent). **Candidate (c) does
NOT reliably clear the bar** (2/3 seeds pass, mean 0.4806 — indistinguishable
from the reference mean given the overlapping seed ranges, s1=0.3994
actually below the fresh reference's own s1), matching the pre-registered
falsification prediction (§2.4/§6: "saturates like F-geo-1/2").

**K=16 no-regression guard (bar: within −0.02 of 0.9767):** both
candidates clear it easily and in fact *improve* on the fresh reference
(0.9716) — (d) 0.9997, (c) 0.9987. **h=1 guard**: `h1_recovered_frac_at_0.9_final
= 1.0000` for every one of the 18 cells (`geo3_admission`). **Admissibility
stack (items 1–4):** `admissible: true`, `ns_converged_no_fallback: true`,
`n_geo3_fallback_train_steps: 0`, `finite_loss_no_divergence: true`,
`task_performance_floor_pass: true` for all 18 cells — full evidentiary
tier, 3/3 clean seeds, zero fallbacks, both K, both candidates, and the
reference arms.

**Early-stop / value-Gram kill rule (sec 3.4) — never triggered, and a
disclosed bonus finding.** `value_gram_deviation_mean` at steps
2000/4000 never exceeds the kill thresholds (K=32: >3.90 at BOTH
checkpoints; K=16: >1.70 at BOTH) for any of the 12 Wave-1 cells — the
closest approach is candidate (c) K=32 s1 (3.60→4.27, but the step-2000
leg alone is under threshold, so the AND-both-checkpoints rule never
fires). More strikingly, **candidate (d)'s FINAL (step-20000) value-Gram
deviation is roughly HALF the fresh reference's own**: K=32 mean 3.85
(3.33–4.48) vs. reference 6.69 (6.47–6.85); K=16 mean 1.24 (1.234–1.239)
vs. reference 2.32 (2.05–2.67) — well below even the "frozen-arm range"
(3.2–3.6 / 1.35–1.67) the design's own §4 diagnostic named as the
target for a value-geometry-relief bonus. This is exactly the
"mechanistic bonus, free" §4 pre-registered watching for — but it also
means **part of h4's gain may be flowing through relieved value
geometry rather than (or in addition to) the cross-episode key
stability the hypothesis is about**, which is precisely why item 5's
absence (§9.3) is not a formality: it is the one measurement that would
distinguish "the key-stability channel moved" from "the anchor
parameters relieved value strain by some other route."

**C17 held-out-entity diagnostic (sec 3.3) — pure-geo3 by construction, confirmed.**
h=1 `rec@0.9` = 1.0000 identically across reference, candidate (d), and
candidate (c) at K=32 (h=2/h=3 sit near chance for all three arms alike,
~0.0–0.024) — the `anchor_trained_mask` bypass is behaving exactly as
designed: held-out entities see bare-geo3 treatment with no leakage from
the anchor mechanism, matching the pre-registered scope limit (no
held-out-entity generalization claim).

### 9.5 Outcome assignment

**Per §3.5's map literally: UNASSIGNABLE — not A, not A′, not A″, not B,
not C.** The behavioral profile (bars clear 3/3, λ interior 6/6, no
regression, no kill, admissible) is the pattern Outcome A's *other* three
legs would produce — but item 5 and `engaged_frac` were never measured on
the admitted runs, and per this design's own transparency rule that
gap cannot be papered over with an inference from adjacent evidence (the
bare-geo3 reference arms' OWN pre-NS raw-key drift, logged incidentally,
is already 0.994–0.998 — suggestively high, but measuring a different
tensor than candidate (d)'s blended pre-NS key, and not a substitute for
it).

**Working claim tier for this wave: DESCRIPTIVE (behavioral) for the
h=4/λ result, NOT YET admissible as a confirmed test of the key-anchoring
interaction hypothesis.** The h=4/λ numbers are real, reproducible,
admissible, and via a large, seed-stable, non-trivial-λ margin — they are
worth reporting and worth the (cheap) follow-up below — but §6's own rule
is explicit that a bar-clearing h4 without a cleared item 5 does not
license a confirmation claim.

**Required follow-up before this wave can be closed at full evidentiary
tier (cheap, no new 20k-step training needed):** (1) fix
`keyanchor_drift_diagnostic.py`'s `log_every` default (`probe_steps // 5`
→ must be registered as a multiple of `LAMBDA_LOG_CADENCE_STEPS`, or the
call should pass `log_every=200` explicitly) and re-run the 2 sequential
K=16/K=32 probes — this recovers Gate 1's verdict (after the fact, for
the record) AND supplies item 5 + `engaged_frac` + the h=1 behavioral
companion on a probe model; (2) for a stronger claim tier than a
probe-model proxy, add `--drift-probe` to a short (~2,000–4,000-step)
confirmatory re-run of one admitted candidate-(d) K=32 seed's exact
config, reading item 5/6/`engaged_frac` directly off training-in-progress
checkpoints of the actual architecture/data path that produced the h4
numbers above. Neither requires new hyperparameter search or a new
design revision — this is instrumentation debt, not a hypothesis
question.

Archive: `experiment-runs/2026-07-05_keyanchor_wave/` (repo, size-capped)
+ SSD mirror at the same relative path under
`/Volumes/1TB_SSD/learned-representations/`. Full verdict + build-gap
narrative: `EXPERIMENT_LOG.md`, "KEY-ANCHORING WAVE VERDICT" (2026-07-05).

### 9.6 Confirmatory wave (2026-07-05) — closes sec 9.3's gap: literal outcome is **C**

Commit `5963616` (same session as this addendum) fixed both root causes
§9.3 identified — `keyanchor_drift_diagnostic.py`'s `log_every` crash and
`keyanchor_wave1_manifest()`'s missing `drift_probe=True` threading — and
added `--wave keyanchor-confirm`: **candidate (d), K=32 seeds {0,1,2} + a
K=16 seed-0 spot check, all with `drift_probe=True` wired directly into
the real 20,000-step training loop** (not a separate probe model — a
stronger design than §9.5 follow-up (2)'s suggested 2,000–4,000-step
probe re-run; follow-up (1)'s fixed-diagnostic probe also ran as this
wave's pre-launch Gate-1 read, see below). All 4 result JSONs
(`experiment-runs/2026-07-05_keyanchor_confirm/wavekeyanchor-confirm/*.json`)
were pulled from the box and inspected directly, field-by-field, for this
addendum — every number below is read from the JSON, not taken from the
launch console. `KEYANCHOR_CONFIRM_CHAIN_DONE` present; wave log: "4
succeeded this session, 0 failed, 0 still pending."

**Pre-launch gates (both closed).** The fixed
`keyanchor_drift_diagnostic.py` probe (Gate 1, §4) now completes:
`predicted_gate_value: 1.0 ≥ 0.8` bar → PASS (`launch: true`) — recovers,
after the fact, the pre-spend check §9.3 found had silently never run.
The launcher's mechanical §3.6 gate re-hashed and validated the
**original wave's own** `waveref/BANDS_PINNED.json` (no new reference
arms were run — the derived bands from §9.2, `engaged_K = 0.9440` at
K=16 / `0.8864` at K=32, are reused as-is): `"sec 3.6 GATE PASSED:
...validates (hashes match every referenced reference-arm result
JSON)."` All 4 result JSONs carry `unblind_override: false` and no
`claim_tier` key (the override path was not invoked — consistent with a
clean, non-demoted blind).

**Per-leg table, verified from `checkpoints[-1]` (final, step 20000) in
each JSON:**

| K | seed | pre-NS drift, item 5 (bar ≥0.95, §3.1) | post-NS drift (sanity; K's `engaged_K` band, §3.6) | item 6a σ₆₄/σ₁ (bar ≥0.1) | item 6b max\|cos\| (bar ≤0.5) | §3.7 `engaged_frac` (a_e≥0.9 frac, bands ≥90%/[50,90%)/<50%) | h4 `rec@0.9`, M3 (bar ≥0.5) | λ final (band) |
|---|---|---|---|---|---|---|---|---|
| 32 | 0 | 0.9912 **PASS** | 0.9188 (≥0.8864 ✓, non-gating) | 0.1072 **PASS** | 0.3815 **PASS** | **0.1308 (13.08%) — <50%** | 0.6654 **PASS** | 0.5751 (interior) |
| 32 | 1 | 0.9918 **PASS** | 0.9130 (≥0.8864 ✓, non-gating) | 0.0706 **FAIL** | 0.3899 PASS | **0.0374 (3.74%) — <50%** | 0.6160 **PASS** | 0.5682 (interior) |
| 32 | 2 | 0.99998 **PASS** | 0.9189 (≥0.8864 ✓, non-gating) | 0.1464 **PASS** | 0.4081 **PASS** | **0.0467 (4.67%) — <50%** | 0.5556 **PASS** | 0.5703 (interior) |
| 16 | 0 (spot check — see scope note below) | 0.99996 **PASS** | 0.9732 (≥0.9440 ✓, non-gating) | 0.2671 **PASS** | 0.3668 **PASS** | **0.1121 (11.21%) — <50%** | 0.9995 (no-regression clean) | 0.5615 (interior) |

Items 1–4 (admissible, `ns_converged_no_fallback`, `finite_loss_no_divergence`,
`task_performance_floor_pass`) are `true` for all 4 legs;
`n_geo3_fallback_train_steps: 0` and `h1_recovered_frac_at_0.9_final: 1.0`
for all 4. **Note on the threshold being compared against item 5:** item
5's own registered bar is the fixed `≥0.95` of §3.1 — all four legs clear
it by a 0.04–0.05 margin regardless of K. The `0.8864`/`0.9440` figures
in the table are the §3.6 **post-NS** `engaged_K` bands (reused,
unchanged, from the original wave's pin), reported here only as
non-gating sanity context (they gate Outcome B1/B2's disambiguation,
which never triggers — h4 already clears its own bar in every leg).

**§3.7 per-entity alignment distribution** (all 107 train entities,
`n_resamples=32` each, min/median/max of `a_e`): K=32 s0
[0.824, 0.876, 0.922]; s1 [0.787, 0.867, 0.923]; s2 [0.788, 0.875,
0.912]; K=16 s0 [0.809, 0.872, 0.919] — **uniform-high and narrow**, not
bimodal (no visible split between an "engaged" cluster near 1.0 and a
"disengaged" cluster near 0). The per-entity h=1 behavioral companion
(§3.7's non-load-bearing diagnostic) is **exactly 1.0000 for all 107
entities, all 4 legs** — the aggregate-masking scenario §3.7 was built to
catch is not visible in behavioral h=1 recovery, only in the input-side
`a_e` metric.

**Outcome routing, applying §3.5 literally (the section is titled
"Outcome frame at K=32" — K=16's spot-check leg is reported as
supplementary no-regression/gap-closing context, per its role in §9.4,
and is not itself assigned a separate Outcome letter):**

1. Item 5 passes decisively, 3/3 (and at the K=16 spot check).
2. h4 passes decisively, 3/3 (and K=16's no-regression bar).
3. λ lands interior, 3/3 (and at K=16) — matches §9.1's own table almost
   exactly (see seed-identity note below).
4. Item 6 (AND of 6a+6b) passes only **2/3** at K=32 — seed 1 fails 6a
   (`0.0706 < 0.1`). Outcome A's own text requires "3/3 admissible under
   items 1–6"; K=32 does not clear that bar even setting aside point 5.
5. **§3.7's `engaged_frac` lands in the `<50%` band in all 4 legs**
   (13.08%/3.74%/4.67% at K=32; 11.21% at K=16) — and §3.7's own text is
   explicit that this **"routes to Outcome C even if item 5's pooled
   statistic passes."**

**Literal assigned outcome: Outcome C — "mechanism not engaged... not an
admissible test of the hypothesis either way. Routes to the fixed-grid λ
diagnostic or a different candidate, not to reinterpreting the
drift-bottleneck theory."** This is unanimous, 3/3 K=32 seeds, and the
K=16 spot check shows the identical pattern. Point 5 alone is
sufficient and overriding per §3.7's own text; point 4 (seed 1's item-6a
miss, 2/3 not 3/3) is an independent, additional reason Outcome A could
not have been reached even absent the engagement failure.

**Claim tier.** This addendum **resolves** §9.5's UNASSIGNABLE verdict —
the gap was a missing measurement, not a failure, and it is no longer
missing. Two separate claims, two separate tiers:
- **The §1 key-anchoring interaction hypothesis:** per Outcome C, **not
  admissible as a test in either direction** — this wave neither
  confirms nor disconfirms it. The aggregate pre-NS drift channel
  genuinely stabilizes (item 5, §9.6's own new evidence) but §3.7's
  per-entity readout — built specifically to catch a pooled statistic
  masking a disengaged majority — shows that stabilization reaching
  ≥0.9 individual alignment for fewer than 14% of entities in every
  measured leg. The behavioral gain cannot, on this wave's own
  instrumentation, be attributed to a majority-entity key-stabilization
  mechanism.
- **The h4 behavioral result** (candidate (d), K=32, `rec@0.9`
  ≈0.41→≈0.61): remains real, reproducible, and admissible under items
  1–4 (4/4 legs clean) — unchanged in *kind* from §9.5's own descriptive
  tier, but changed in *reason*: §9.5 was descriptive because item 5 and
  `engaged_frac` were never measured; this wave measured them, and they
  say "not the proposed mechanism" rather than "unknown." Working tier:
  **DESCRIPTIVE (behavioral)** — a real aggregate/behavioral positive
  with a now-measured mechanistic null underneath it, not a promotion to
  evidentiary/confirmed.

**Verification note — seed identity, not new independent seeds.** The
confirm-wave cells reuse the **same seed integers** as Wave-1's admitted
cells (0/1/2 at K=32, 0 at K=16), trained fully independently from
scratch (not resumed from any Wave-1 checkpoint — `steps_completed:
20000` on a fresh run each). Cross-checked directly against
Wave-1's own archived JSONs
(`experiment-runs/2026-07-05_keyanchor_wave/wavekeyanchor/`): K=32 s0 h4
0.66467→0.66541 and λ_final 0.575117→0.575055; s1 h4 0.61591→0.61603 and
λ_final 0.568165→0.568153; s2 h4 0.55896→0.55560 and λ_final
0.570139→0.570261; K=16 s0 h4 **bit-identical**
(0.99951171875→0.99951171875) and λ_final 0.561516→0.561459. These are
within 0.0001–0.001 of Wave-1's own recorded values per matching seed —
consistent with GPU run-to-run floating-point nondeterminism at a fixed
seed (not a copied file: the numbers are close but not bitwise
identical, and do differ), **not** three newly and independently drawn
seeds. **Correction to the working framing carried into this addendum's
brief:** the confirm wave's h4 numbers should not be read as adding a
second independent 3-seed sample's worth of statistical power to the
h4 seed-robustness claim — that evidence already existed in Wave-1's own
3-seed spread. This wave's evidentiary contribution is specifically the
item-5/item-6/`engaged_frac` measurement on training dynamics that are
functionally the admitted cells, not a fresh replication sample.

**Supporting trajectory observation (descriptive, non-gating).** Reading
`checkpoints[0..9]` for K=32 s0: pre-NS drift is already `0.9589` at
step 2000 (h4 only `0.3641` there) and reaches `0.9836` by step 4000
(h4 `0.6343`, already near its final value) — the pooled pre-NS channel
saturates near its ceiling early and stays roughly flat for the
remaining 16,000 steps while h4 continues to move. This pattern (present
in all 4 legs) is itself a small piece of supporting color for why the
per-entity `engaged_frac` check was necessary in the first place: a
pooled statistic that saturates early and stays flat is exactly the kind
of signal that could mask a minority-only engagement, which is what the
per-entity numbers above show actually happened.

**Disclosures.**
- No per-checkpoint `engaged_frac`/`per_entity_alignment` **trajectory**
  exists — confirmed directly: `per_entity_alignment` and
  `per_entity_h1_companion` keys are present only on the step-20000
  checkpoint dict in all 4 JSONs (absent at steps 2000–18000), while
  item 5 (`drift_probe`) and item 6 (`item6_table_conditioning`) ARE
  present at every one of the 10 checkpoints. This is a final-checkpoint-only
  scoping of the full-pool per-entity sweep specifically (auditor-adjudicated
  at build time, consistent with §3.7's "the claim readout uses the final
  step" — but §3.7's own text also says "logged at every admission
  checkpoint," which the per-entity sweep does not do; item 5/6 do).
- The confirm cells are fresh full 20,000-step runs, not re-probes of the
  originals in the sense of resuming a checkpoint — but see the
  seed-identity note above for what "fresh" does and does not mean here.
  The h4 pattern (0.665/0.616/0.556 here vs. 0.665/0.615/0.559 in Wave-1)
  is a same-seed reproduction check, not a second independent-seed
  replication; say so plainly rather than imply 6 independently-drawn
  data points.

**Disclosed post-hoc observation — non-evidentiary, does NOT re-score
anything above.** The `a_e` distribution's uniform-high, narrow shape
(median ≈0.87, min ≈0.79–0.82 across the four legs) is arithmetically
consistent with the registered blend formula
(`key_anchoring.py` L265–266: `sub_blend = normalize((1-λ)·k_raw +
λ·anchor)`, cosine measured against `anchor`). At the SGD-preferred
λ≈0.57–0.58 (§9.1/§9.6's own tables, seed-stable, interior — not a
degenerate λ→1 pin-rediscovery), and letting `r = cos(k_raw, anchor)`
(NOT itself logged this wave — this is an algebraic back-solve from the
observed `a_e` values and the known λ, not a new measurement):
`cos(blend, anchor) = [(1-λ)r + λ] / sqrt((1-λ)² + λ² + 2λ(1-λ)r)`.
Solving for the `r` needed to cross `a_e ≥ 0.9` at λ=0.57 gives
`r ≳ 0.48`; the observed median `a_e ≈ 0.87` back-solves to `r ≈ 0.33–0.35`
— short of the crossover. As λ→1 the formula collapses to `cos ≡ 1`
regardless of `r` (the same class of triviality as the Rev-2 λ=1
drift-ceiling correction, §3.1) — meaning the registered `a_e ≥ 0.9`
bar is close to only cleanly satisfiable in the λ→1 (pin-rediscovery,
Outcome A′, explicitly "not interesting") regime, and an SGD-preferred
*interior* λ structurally caps `a_e` below 0.9 for any entity whose raw
key hasn't independently reached moderately high alignment with its
anchor on its own. **This motivates, but does not itself execute,** a
Rev 6 threshold re-derivation (below); it is disclosed here strictly as
context for why `engaged_frac` reads low even with h4 and item 5 both
strong, not as a justification for re-scoring this wave's Outcome-C
assignment.

**Rev 6 (REGISTERED STUB — NOT SPECIFIED, NOT SCHEDULED, NO RETROACTIVE
RESCORING AUTHORIZED).** The current §3.7 bar (`a_e ≥ 0.9`, flat,
λ-independent) conflates two different things: whether an entity's raw
key has moved toward its anchor at all (the mechanism's actual claim),
and whether the run's current scalar λ happens to be close enough to 1
that blend arithmetic alone pushes the post-blend cosine over 0.9
regardless of the first. A future revision should either (i) measure
`r = cos(k_eff_raw, anchor)` directly, upstream of the blend, so the
metric no longer depends on λ, or (ii) rescale the 0.9 bar per-run by
the closed-form ceiling at that run's own measured λ (the crossover
surface sketched above). Per this project's own repeated finding that
independent adversarial audit rounds catch different bugs each time,
**any such revision requires its own attack round before any existing or
future wave's numbers are re-scored against it** — this stub registers
the need, it does not adjudicate it, and no wave's Outcome assignment
above is changed by it.

Archive: `experiment-runs/2026-07-05_keyanchor_confirm/` (4 result
JSONs, chain logs, the fixed-diagnostic Gate-1 probe JSON/log,
`keyanchor_confirm_chain.sh`, `smoke_keyanchor_confirm.py`; ~6.9MB, all
files ≤2MB, committed) + SSD mirror at the same relative path under
`/Volumes/1TB_SSD/learned-representations/`. The code that produced
these results (`key_anchoring.py`, `run_deltanet_rd_exactness_sweep.py`,
`run_deltanet_rd.py`, `keyanchor_drift_diagnostic.py`,
`smoke_key_anchoring.py`) was already committed in `5963616` (build-only,
no GPU spend in that commit). Full narrative:
`EXPERIMENT_LOG.md`, "KEY-ANCHORING CONFIRMATORY WAVE VERDICT"
(2026-07-05). `STATE.md` updated.

### 9.7 Rev 6 DRAFT — λ-scaled engagement threshold re-derivation (2026-07-05/06)

**CLOSED 2026-07-05 — RE-SCORE REJECTED BY THE ATTACK ROUND
(`KEYANCHOR_REV6_ATTACK.md`); OUTCOME C STANDS.** Three findings killed it:
(1) the z that preserves the never-challenged flat-0.9 bar's stringency at the
realized λ is ≈3.76–4.03 — BOTH menu options {2,3} are relaxations, so any
outcome-changing pick is threshold-shopping by construction; (2) the "blind
z-pick" procedure was self-defeating — this section's own tables publish each
z's outcome; (3) the unverified `‖anchor‖=1` assumption admits a plausible
alternative (`m≈1.34`, ordinary Adam drift for an unconstrained embedding)
under which the observed a_e median is a chance-level scale artifact. The
attack also surfaced the structural point that a single global-scalar λ cannot
differentially engage entities, so §3.7's bimodality frame was never a test
candidate (d) could fail informatively. Consequence: the mechanism-engagement
question requires a FRESH pre-registered wave (per-entity instrumentation
incl. anchor-row norms, and an engagement metric derived for interior λ before
any data exists) — it cannot be settled by re-scoring this wave's data. The
behavioral h4 result (§9.6) is unaffected: real, reproducible, descriptive
tier. Section retained below as the historical record of the attempt.

**Status: DRAFT. Pending its own attack round (per this project's own
repeated finding — see the Hard Rules entry on multi-round adversarial
audit — that a single self-review is not a substitute). Executes §9.6's
registered stub verbatim (both its listed options, unified below into
one derivation). No Outcome assignment above (§9.5, §9.6) is changed by
this section. No retroactive rescoring is authorized until the attack
round clears this derivation AND the open items in §9.7.5 (the
unverified anchor-norm assumption) and §9.7.6 (the z-choice) resolve.**

#### 9.7.1 The problem, restated precisely

§9.6's post-hoc observation showed `a_e ≥ 0.9` (§3.7) is not a
λ-independent statement: at the SGD-preferred interior λ (≈0.57–0.58,
all four confirm-wave legs), the blend formula requires the entity's
raw key to independently reach `r = cos(k_eff_raw, anchor) ≳ 0.48–0.50`
to cross 0.9 — but at λ→1 the SAME "0.9" number is trivially satisfied
by every entity regardless of `r` (§9.6: "the formula collapses to
`cos ≡ 1` regardless of r"). A flat numeric bar whose implied
`r`-space stringency swings from "demanding" (`r≈0.9` at λ→0) to
"vacuous" (`r`=anything at λ→1) is not measuring the same thing across
different λ regimes, candidates, or K values — it is measuring
"however much `r` this run's own λ happens to require," which is not a
portable definition of "engaged." **The fix is not to find a
different flat number; it is to define engagement in `r`-space (which
does not degenerate with λ) and re-express it in `a_e`-space, per-run,
using that run's own already-logged λ — so the criterion means the
same thing at every λ, not just at whichever λ this project happened
to measure.**

#### 9.7.2 Forward derivation (pure algebra, from the registered formula)

Blend formula, verbatim, `key_anchoring.py` L263–269:

```python
trained_here = anchor_trained_mask[key_ids]
t_idx = trained_here.nonzero(as_tuple=True)
sub_blend = F.normalize(
    (1.0 - lam) * k_eff_raw[t_idx] + lam * anchor_weight[key_ids[t_idx]], dim=-1)
```

`k_eff_raw` is gathered from `k_norm_raw = F.normalize(k_conv, dim=-1)`
(`model_rd.py` L980) — **unit norm, confirmed by reading the line, not
assumed.** Let `k` = the raw key (unit vector), `â` = the anchor row's
unit *direction*, `r = cos(k, â)`, `λ` = the run's current blend
weight. **Assuming `‖anchor_weight[e]‖ = 1`** (flagged and tested for
in §9.7.5 below — this is the one load-bearing assumption in this
derivation), the pre-normalize blend is `v = (1-λ)k + λâ`, and:

```
v·â = (1-λ)r + λ
‖v‖² = (1-λ)² + λ² + 2λ(1-λ)r
a_e(λ, r) ≡ cos(v, â) = [(1-λ)r + λ] / sqrt[(1-λ)² + λ² + 2λ(1-λ)r]
```

(`F.normalize` inside `sub_blend` doesn't change this — cosine is
scale-invariant, so `cos(normalize(v), â) = cos(v, â)` identically;
the code's `measure_full_pool_alignment`, `key_anchoring.py` L673–695,
also calls `F.cosine_similarity` directly, which internally
renormalizes `blended` regardless.)

**Sanity checks (closed-form, matches §9.6's own back-of-envelope
exactly):**
- `a_e(1, r) ≡ 1` for all `r` — the λ→1 degeneracy §9.6 already flagged.
- `a_e(0, r) = r` — the λ→0 limit recovers the raw cosine directly.
- Solving `a_e(λ=0.57, r) = 0.9` for `r` gives `r ≈ 0.483` — matches
  §9.6's `r ≳ 0.48` to 3 decimal places (verified by direct
  quadratic solve in `r`, not re-typing the approximation: `α²r² +
  2αλ(1-a_e²)r + [λ²(1-a_e²) - a_e²α²] = 0`, `α = 1-λ`).
- Inverting the *observed* per-entity `a_e` values back to `r` at each
  cell's own logged `λ_final` reproduces §9.6's `r ≈ 0.33–0.35` median
  claim: K=32 s0/s1/s2 median `r_e` = 0.350 / 0.326 / 0.359, K=16 s0 =
  0.371 (computed CPU-only from the archived JSONs, script below).

**Numeric verification, all 4 confirm-wave cells** (`checkpoints[-1]`,
step 20000, `experiment-runs/2026-07-05_keyanchor_confirm/wavekeyanchor-confirm/*.json`;
`d_state = 64` for all 4 — read from each JSON, not assumed):

| cell | λ_final (own, logged) | null floor `a_e(λ,0)` | crossover `r` @ `a_e`=0.9 | old `engaged_frac` (flat 0.9) |
|---|---|---|---|---|
| K=32 s0 | 0.5751 | 0.8042 | 0.4696 | 0.1308 |
| K=32 s1 | 0.5682 | 0.7961 | 0.4873 | 0.0374 |
| K=32 s2 | 0.5703 | 0.7986 | 0.4820 | 0.0467 |
| K=16 s0 | 0.5615 | 0.7881 | 0.5036 | 0.1121 |

Note the null floor itself: at every measured λ, an entity whose raw
key contributes **zero** alignment (`r=0`) already reads `a_e ≈
0.79–0.80` — only ~0.10–0.11 below the old flat bar. This is the
mathematical fact the compression language in §9.6 was pointing at:
the entire discriminative *range* available to `a_e` between "zero
raw-key contribution" and "old bar" is compressed into a band roughly
0.10 wide, at this λ.

#### 9.7.3 Why a flat `a_e` floor isn't enough — the chance-level re-derivation

`a_e(λ, 0)` is the value produced by an entity whose raw key is
**exactly orthogonal** to its anchor. But "orthogonal" is not the same
statistical object as "not engaged" — with `k` and `â` both unit
vectors in `R^{d_state}`, two vectors that have never interacted at
all (independent, uncorrelated directions — i.e., the null hypothesis
"this entity's key never moved toward its anchor") have `r = cos(k,
â)` distributed with **mean 0 and variance exactly `1/d_state`** (an
exact algebraic fact for a uniformly-random direction against any
fixed reference direction in `R^d` — not an asymptotic approximation:
by symmetry, the `d` squared components of a random unit vector each
have expectation `1/d`, and the cosine against a fixed axis is one
such component). For `d_state = 64` (this run family, confirmed from
each JSON's own `d_state` field): `σ_chance = 1/√64 = 0.125` exactly.

This gives a **λ-independent, portable engagement criterion in
`r`-space**: an entity's raw key is *detectably* pulled toward its
anchor — the actual claim §1/§3.7 exists to test — only if its `r`
sits meaningfully above this chance band, i.e. `r ≥ z·σ_chance =
z/√d_state` for a registered `z`. Re-expressed in `a_e`-space via
§9.7.2's own formula, per run, at that run's own logged `λ_final`:

```
engaged(e)  ⟺  a_e(e) ≥ a_e(λ_final, z/√d_state)     [λ-SCALED THRESHOLD]
```

This is exactly the `a_e ≥ a_e_expected(λ, r=0) + margin` form §9.6's
stub sketched, with the margin **not** a free constant fit to any run's
data — it is `a_e(λ, z/√d_state) − a_e(λ, 0)`, a closed-form function
of `λ` (already logged, per-run) and `d_state` (a registered
architectural constant, fixed since before Wave 1 ran) and `z` (see
§9.7.6 — the one remaining registered choice, deliberately NOT
resolved by this draft). The a_e(λ,0) floor and the z/√d margin are
thus **both** derived purely from (i) the unchanged, already-committed
blend formula and (ii) numbers that exist independent of how any
K=32 cell's engagement happens to score.

**Full table, `z ∈ {1, 2, 3}` (all three reported; none singled out as
"the" answer by this draft — see §9.7.6):**

| cell | λ_final | thresh z=1 (`r`≥0.125) | thresh z=2 (`r`≥0.25) | thresh z=3 (`r`≥0.375) | new `engaged_frac` z=2 | new `engaged_frac` z=3 | old `engaged_frac` |
|---|---|---|---|---|---|---|---|
| K=32 s0 | 0.5751 | 0.8303 | 0.8560 | 0.8812 | **0.7664** | 0.4112 | 0.1308 |
| K=32 s1 | 0.5682 | 0.8236 | 0.8505 | 0.8768 | **0.8224** | 0.2617 | 0.0374 |
| K=32 s2 | 0.5703 | 0.8257 | 0.8522 | 0.8782 | **0.8318** | 0.4393 | 0.0467 |
| K=16 s0 | 0.5615 | 0.8169 | 0.8451 | 0.8725 | **0.8879** | 0.4393 | 0.1121 |

At `z=1` every cell reads 97–99% engaged (non-discriminative — 1σ
above a symmetric null still contains most of the null mass one-sided,
so this bracket is disclosed but not a serious candidate). At `z=2`
all four cells land in **[50%, 90%)** — the §3.5 Outcome A″ band. At
`z=3` all four cells stay **<50%** (26.2–44.0%) — Outcome C stands
unchanged. **The registered z choice is the single remaining degree of
freedom that determines whether this re-derivation changes anything at
all**, which is exactly why §9.7.6 pre-commits to a menu rather than a
pick.

#### 9.7.4 Secondary check: does the `r`-space distribution show the bimodality §3.7 was built to detect?

Inverting each cell's 107 `a_e` values to `r_e` (per §9.7.2's formula,
at that cell's own λ) gives a substantially *wider* spread than `a_e`
itself (e.g. K=32 s0: `r_e` ∈ [0.095, 0.581] vs. `a_e` ∈ [0.824,
0.922] — confirming numerically that the blend arithmetic compresses
the `r`-space range by roughly 4–5× at this λ). A coarse 10-bin
histogram and max-gap check on the sorted `r_e` values (not a formal
dip-test — disclosed as informal, a candidate for a proper Hartigan's
dip statistic in the attack round if more rigor is wanted) shows **no
gap** consistent with two separated clusters in any of the 4 cells —
each is single-peaked, roughly bell-shaped, largest gaps at the
distribution's edges (expected for a unimodal sample, not a signature
of a bimodal split). **This means the aggregate-masking scenario §3.7
was designed to catch — a strongly-engaged minority hiding inside a
disengaged majority — is not what these numbers show, even in the
less-compressed `r`-space.** The picture instead reads as *uniform
partial engagement*: every entity's raw key moved moderately toward
its anchor (median `r_e` ≈ 0.33–0.37), none moved decisively, and none
stayed at exactly chance level. This is a real, disclosed, alternative
mechanistic story to have on record regardless of which `z` the attack
round eventually pins.

#### 9.7.5 Open item — the unverified unit-anchor-norm assumption

§9.7.2's derivation assumes `‖anchor_weight[e]‖ = 1` at read time.
`anchor_table` is initialized at unit norm (`frame_potential_init`,
`key_anchoring.py` L230, `F.normalize(...)`) but is a **plain
`nn.Embedding`** (`model_rd.py` L815) with no renormalization step
found anywhere in `model_rd.py` or `key_anchoring.py` (checked by
grep, not assumed) — its row norms are free to drift under 20,000
steps of unconstrained gradient descent. **This was checked, not
assumed: no checkpoint file exists anywhere to verify the actual norm.**
The local SSD mirror
(`/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-05_keyanchor_confirm/`)
contains only JSONs/logs, no `.pt`. A read-only remote check
(`ssh youthful-indigo-turkey`, this session) found
`/data/deltanet_rd_keyanchor_ckpts/{wavekeyanchor,wavekeyanchor-neg1,waveref}/`
— all three **empty directories**, and **no** corresponding
`.../keyanchor-confirm/` checkpoint directory exists at all. No raw
`anchor_table.weight` tensor is recoverable from any archived
artifact, for any wave. **This derivation's numeric threshold values
(§9.7.3's table) are therefore conditional on an assumption that
cannot currently be checked** — flagged here rather than silently
carried forward. Two ways to close this, priced for the attack round
or a future wave, not this draft: (a) cheapest — add one scalar
(`anchor_table.weight[train_ids].norm(dim=-1).mean()`, and ideally the
per-row vector) to the existing per-checkpoint result-assembly path
(`key_anchoring.py`/`model_rd.py`'s existing checkpoint writer — no new
forward pass, reads a parameter already in memory) in any *future*
confirm-style wave; (b) if norms turn out to deviate materially from
1, the general (non-unit-norm) formula is a two-line extension of
§9.7.2 (`v = (1-λ)k + λmâ`, `m = ‖anchor_weight[e]‖`) — the derivation
*approach* survives, only the numeric constants would move.

#### 9.7.6 The z choice — registered menu, orchestrator decision required

`z` is the one genuinely open parameter in this derivation. `z=2` ("2σ
above chance," the most common textbook convention for "detectably
non-null") and `z=3` (the stricter convention this project's own
rigor culture — see the Hard Rules on multi-round adversarial audit,
exact structural thresholds, and provable rank bounds — would favor)
bracket materially different verdicts (§9.7.3: A″-band at z=2, Outcome
C at z=3, for all 4 cells). **This draft does not pick between them.**
Per this design's own established pattern for exactly this kind of
judgment call (§3.6's n=2→n=3 orchestrator decision, §3.4's kill-rule
target, §3.6's Judgment Call (c)), the choice is registered as an
**orchestrator decision at the attack round**, made by inspecting
*only* the derivation and the z=1/2/3 menu above — **never** by first
computing all three against the K=32 cells and then arguing backward
for whichever one was picked. The order this draft followed (and the
attack round must too, to stay clean): fix the formula and the
`d_state`-derived floor first (§9.7.2), fix the candidate z menu from
a standard, external convention second (§9.7.3, before recomputing
per-cell numbers), THEN compute what each implies (the table) — never
the reverse.

#### 9.7.7 Pre-registered re-score procedure

**Scope — which runs get rescored:** checked directly against the
archive, not assumed. **Only the 4 confirm-wave cells**
(`experiment-runs/2026-07-05_keyanchor_confirm/wavekeyanchor-confirm/
wkeyanchor-confirm_rdx_{K32_armd_s0,K32_armd_s1,K32_armd_s2,K16_armd_s0}
_geo3n{20,20,20,12}_anchor_learned_dprobe.json`) carry a
`per_entity_alignment` key at all. **All 12 Wave-1 cells** — both
`experiment-runs/2026-07-05_keyanchor_wave/{wavekeyanchor,
wavekeyanchor-neg1}/*.json` and the earlier `w-1_rdx_*.json` files (6 +
6 = 12 checked, field-by-field, this session) — **lack
`per_entity_alignment` entirely** (confirmed by direct key inspection
of every file's final checkpoint, not inferred). Re-scoring Wave-1
would require a fresh instrumented rerun (the per-entity sweep did not
exist until the confirm-wave build, `5963616`/`03edb29`), which is a
GPU-spend decision out of scope for this design-only draft — not
something this section authorizes.

**Exact computation, per confirm-wave cell, once `z` is pinned
(§9.7.6):**
1. Read `λ_final = anchor_lambda_summary.final_value` (already logged,
   this run's own value — never re-measured, never chosen to fit).
2. Compute `thresh = a_e(λ_final, z/√64)` per §9.7.2's closed form.
3. `new_engaged_frac = mean(1{a_e[e] ≥ thresh} for e in the 107 train
   entities)`, reading `checkpoints[-1].per_entity_alignment.a_e`
   directly — no resampling, no new forward passes.
4. Route through the **unchanged** §3.5/§3.7 band edges (≥90% / [50%,
   90%) / <50%) — only the per-entity criterion changes, not the
   banding logic, the outcome letters, or any other admission item.

**Anti-post-hoc protections (the whole point of this section):**
- The formula (§9.7.2) is pure algebra from code already committed in
  `5963616`, predating this draft and any awareness of how the
  confirm-wave engagement numbers would land.
- `d_state=64` is a registered architectural constant, fixed since
  before Wave 1, not chosen for this derivation.
- `λ_final` per cell is an already-logged, already-fixed number from a
  run that predates this draft — not remeasured, not selected.
- The `z` menu (§9.7.6) is fixed to standard external conventions
  (2σ/3σ) and explicitly NOT resolved to a single value by this draft
  — deferred to an orchestrator decision made by inspecting the
  derivation, not the per-cell payoff table.
- §9.7.5's open item (unverified anchor-norm assumption) is disclosed,
  not hidden, and is a precondition on treating any numeric threshold
  value as final.
- No Outcome letter anywhere in §9.5/§9.6 changes as a result of this
  section. The re-score executes only after (a) the attack round pins
  `z`, and (b) §9.7.5 is either closed or explicitly waived with
  reasoning recorded.

#### 9.7.8 What each re-score outcome means

**Under z=2 (computed here as a non-binding preview — the registered
re-score happens only after the attack round, per §9.7.7):** all four
cells land in `new_engaged_frac ∈ [50%, 90%)` (76.6% / 82.2% / 83.2% /
88.8%) — the §3.5 **Outcome A″ ("partial anchoring — anchoring lifts
composition for the engaged subset"), reported in full, no headline**.
But Outcome A″ inherits the same "bars clear" requirement as Outcome
A, and §9.6's own table already shows **K=32 seed 1 fails item 6a**
(`σ_64/σ_1 = 0.0706 < 0.1`) — independent of engagement, this caps any
upgrade at **2/3 seeds**, not 3/3. **So even under the more permissive
z=2 preview, the ceiling this wave could reach is Outcome A″ at a 2/3
(not 3/3) seed tier — never Outcome A (the headline tier requires
`engaged_frac ≥ 90%`, and no cell reaches it even at z=2's most
generous reading: K=16's 88.8% is the closest and still short).** The
K=16 spot check is not separately outcome-lettered under §3.5's own
K=32 scope note (unchanged by this draft) — it remains supplementary
replication context, now additionally showing the same A″-band pattern
under z=2.

**Under z=3 (also a non-binding preview):** all four cells stay
`<50%` (26.2–44.0%) — **Outcome C stands, unchanged from §9.6.** The
mechanism story then rests on §9.7.4's disclosed finding: the failure
mode is not "a masked disengaged majority" (no bimodal split found even
in `r`-space) but "uniform, moderate, incomplete engagement across the
whole pool" — a different open question (is the anchor pull
under-powered at this λ for structural reasons, a capacity limit, or
simply a training-length/learning-rate issue not yet explored) worth
its own follow-on, not a re-interpretation of the drift-bottleneck
theory the parent design's Outcome C routing already refuses to grant.

**Either way, the h4 behavioral result's own tier (DESCRIPTIVE,
behavioral — §9.6) is untouched by this section**: nothing here
revisits items 1–5 or the h4 bar itself, only §3.7's engagement
criterion.

#### 9.7.9 Budget

**~0 GPU-h.** This entire section is a re-read of 4 already-local,
already-committed JSONs (`experiment-runs/2026-07-05_keyanchor_confirm/
wavekeyanchor-confirm/*.json`) plus pure CPU arithmetic (closed-form
algebra and one quadratic solve per entity; the full 4-cell, 107-
entity-each verification ran in well under a second). The one
read-only network call this draft made was an `ssh` `find`/`ls` to the
Brev box to check for checkpoint files (§9.7.5) — no training, no box
writes, no GPU touched. If the attack round wants §9.7.5 closed with
real data, the cheapest path (one scalar added to an existing
checkpoint-write call, no new forward pass) is priced at effectively
zero incremental GPU-h on top of any *future* wave that was going to
run anyway — but no such wave is scheduled or requested by this draft.
Rescoring the 12 Wave-1 cells (if ever wanted) is explicitly **out of
scope and unpriced here** — it would need a fresh full 20,000-step
instrumented rerun per cell, priced the same as any other §5-budget
line item, and is not requested or authorized by this section.

#### 9.7.10 Attack surface — 5 pre-answered

1. **"You moved the bar until it passed" (post-hoc threshold-shopping —
   the headline risk).** Answer: the formula is pure algebra from code
   committed in `5963616`, before this draft existed; `d_state=64` and
   each cell's `λ_final` are fixed, pre-existing numbers untouched by
   this section; the `z` menu is pinned to standard external
   conventions and explicitly left unresolved (§9.7.6) rather than
   collapsed to whichever value looks best — and even the more
   permissive registered candidate (z=2) only reaches a **non-headline**
   partial outcome (A″, capped at 2/3 seeds by an unrelated item-6a
   miss), never the headline Outcome A. A threshold actually shopped
   to rescue a null result would aim for the strongest available
   positive; this one tops out well short of that even at its most
   generous reading.
2. **"Isn't picking `z=2` (or any single z) exactly the same sin as
   picking `a_e≥0.9`?"** Answer: no single z is picked by this draft
   (§9.7.6) — both standard brackets (2σ, 3σ) are reported with equal
   prominence, their diverging implications disclosed plainly, and the
   final choice is explicitly deferred to an orchestrator decision at
   the attack round, following the same judgment-call pattern this
   design already uses elsewhere (§3.4, §3.6). The *menu* is
   constrained to widely-recognized conventions, not an open range.
3. **The unverified `‖anchor_weight‖=1` assumption (§9.7.5).** Answer:
   disclosed, not hidden — checked this session (SSD mirror + a
   read-only remote `find`) and confirmed no checkpoint artifact
   exists anywhere to verify it; the numeric thresholds in §9.7.3 are
   explicitly conditional on it, with a two-line extension path if it
   turns out false, and a near-zero-cost logging addition specified
   for any future wave that would close the gap.
4. **Circularity — the K=16 spot check is one of the same 4 scored
   cells, not an independent calibration source.** Answer: correct,
   and this draft does **not** use K=16 to derive any constant in the
   formula (`z`, `d_state`, and the algebraic form are all independent
   of any cell's data) — K=16 is used only as a **replication check**
   (does the same registered rule produce a consistent pattern on an
   independent K), never as the source of a fitted number. No margin
   constant in this derivation is fit to K=16 or K=32 data.
5. **Jensen's-gap bias in the `r`-inversion.** `a_e[e]` is already a
   mean over ≥8 (32 at K=32, per `n_resamples`) independent episode
   resamples, each with a potentially different raw key draw for that
   entity (`measure_full_pool_alignment`, `key_anchoring.py` L688–693:
   the per-resample cosines are averaged, not the raw keys). Inverting
   the *mean* `a_e` through the nonlinear `a_e(λ,r)` map does not
   exactly recover `mean_i(r_i)` (the map is not affine) — the
   recovered `r_e` in §9.7.4 is an **effective** summary, disclosed as
   such, not an exact per-resample average. This does not affect
   §9.7.3's `engaged_frac` computation (which thresholds the logged
   `a_e` values directly, never the inverted `r_e`) — it only
   qualifies §9.7.4's secondary bimodality read, which is already
   labeled informal.

---

## 10. Rev 7.1 — Mechanism-tier confirmation wave (design-only; attacked at Rev 7, revised in place)

**Status: Rev 7.1 — the attack-response revision. Rev 7 (commit
`09eb392`) received its own attack round (`KEYANCHOR_REV7_ATTACK.md`,
verdict NEEDS-REV): 1 FATAL (the §10.3.3 pin artifact was *claimed*
committed but had never been written — a verify-before-claiming
violation at the center of the admissibility argument) + 4 MAJORs (pin
enforcement incomplete; a second Jensen's-gap instance in the null-pool
construction; an aggregate-only null check blind to hub-entity
contamination; magnitude-calibrated bands silently reused for a
discovery-rate quantity). All are addressed in place below;
`rev7_threshold_derive.py` + `REV7_THRESHOLD_PINNED.json` now actually
exist and are committed alongside this revision; the finding→change map
is §10.12. Rev 7.1 remains DRAFT pending a bounded verify round of these
fixes before any GPU is spent — per this project's own repeated finding
(multi-round adversarial audit catches different bugs each round) and
the direct precedent one section up (§9.7/Rev 6, rejected for
self-adjudicating without one). Zero GPU-h spent producing this section
(design + CPU code-read/derivation work only). No Outcome assignment in
§9.5/§9.6/§9.7 is touched, revisited, or rescored by anything below.**

### 10.1 What this wave is answering, and why Rev 6 couldn't

`KEYANCHOR_REV6_ATTACK.md` killed the Rev 6 rescore attempt on three
grounds plus one deeper structural point, and this wave is built as a
direct, one-to-one response to all four — not a second attempt at the
same derivation with nicer numbers:

1. **Blind-pick was already broken by its own document** (attack §2):
   §9.7.3/§9.7.8 published every candidate z's outcome in the same
   file that told the attack round to pick blind. **Fix (§10.3):** the
   engagement threshold this wave uses has **no menu and no choice
   left for anyone to make** — it is a closed-form function of
   `n_train=107` and `d_state=64` (both registered since before Wave 1)
   and a conventional `α=0.05`, with zero dependency on any anchor-arm
   result. **Correction at Rev 7.1 (attack-R7 FATAL, §10.12 F1):** Rev 7
   *described* this artifact as already committed when it had not been
   written — as of Rev 7.1 it genuinely exists
   (`matrix-thinking/deltanet_rd/rev7_threshold_derive.py` +
   `REV7_THRESHOLD_PINNED.json`, committed with this revision, smoke 5's
   data-independence check run and passing), hash-locked before any of
   this wave's data exists — a strictly earlier blind than §3.6's own
   reference-arm gate requires, and now one that is *instantiated*, not
   merely planned.
2. **The offered z-menu excluded the bar-preserving value and both
   options were relaxations** (attack §3): there is no menu in this
   wave to exclude anything from — see (1).
3. **The unverified `‖anchor‖=1` assumption could make the entire
   signal a chance-level scale artifact** (attack §4, `m≈1.34`
   plausible): **fix (§10.2):** the primary engagement statistic
   (`r_e`) is measured via `F.cosine_similarity`, which renormalizes
   both operands internally — the anchor row's norm cancels out of the
   computation by construction, not by assumption. Anchor-row norms
   are additionally logged directly, every checkpoint, every run
   (§10.2.1), so the question is closed either way, not carried forward
   a third time.
4. **The deepest point (attack §5): a single global-scalar `λ` has no
   structural capacity to differentially engage entities — the only
   source of per-entity variation in the old metric was pre-existing,
   incidental key/anchor correlation, not anything the mechanism
   controls.** A metric fix alone (items 1–3) cannot address this — it
   can only measure candidate (d) more honestly, not give it a
   capability it structurally lacks. **Fix (§10.5):** a genuinely new
   arm, candidate (d′), replaces the single scalar with a per-entity
   table `λ_e`, so the question "does anything about this class of
   mechanism ever differentially engage entities" has an arm that can
   actually answer it either way.

**The top pre-registered objection this draft expects — "this is
attempt #3 at the same claim, what makes it admissible where Rev 6
wasn't" — is answered by the four points above taken together, not by
any one of them alone: everything load-bearing here is derived before
any of this wave's data exists (§10.3.3's hash-lock has *zero* data
dependency, stronger than Rev 6's reference-arm-gated blind); the old
confirm-wave's four JSONs are never reopened or rescored (they lack the
`r_e`/anchor-norm fields entirely — re-scoring them is not merely
undesirable, it is not physically possible from data that does not
exist, §10.4); and the metric is redesigned for the regime the
mechanism actually operates in (interior λ, §9.6's own finding) instead
of re-deriving yet another number in `a_e`-space that still has to pass
back through a λ-dependent blend formula.**

#### 10.1.1 Verification this section relied on (code-read, this session, CPU only)

Before designing the null model below, the anchor-table construction
was re-read directly (`model_rd.py` L805–817), not assumed from the
docstring or from §2.2's own summary:

```python
anchor_table_full = torch.zeros(vocab_size_total, d_state)
anchor_table_full[anchor_train_ids] = init_table   # frame-potential-inited, 107 rows only
self.anchor_table = nn.Embedding(vocab_size_total, d_state)
```

**This is a load-bearing correction to a design idea this draft
initially considered and discarded before writing §10.3**: anchor rows
for entities OUTSIDE the 107-name train pool (every held-out entity,
every non-name vocab token) are initialized to the **exact zero
vector**, not a random unit direction. A held-out entity's own cosine
to its own (never-trained, never-touched-by-gradient — confirmed by
the existing NaN-injection smoke, §2.2/§5 smoke 4) anchor row is
therefore **degenerate**: `F.cosine_similarity` against an exact-zero
vector returns exactly `0.0` (its internal `eps`-clamped denominator
makes this well-defined, not `NaN` — verified this session, one-line
CPU check, not assumed from the PyTorch docs), for every held-out
entity, always, regardless of the held-out entity's own raw key. Using
"held-out entity vs. its own anchor row" as an empirical null sample
(an idea this draft considered first) would therefore silently measure
a constant, not a distribution — a real near-miss, caught by reading
the actual init code rather than trusting the docstring, which is
exactly the failure mode the project's "verify before claiming" rule
exists to catch. **§10.3's actual design uses a different, non-degenerate
empirical reference (mismatched real anchor rows, never an entity's own
zero-valued held-out row) for exactly this reason.**

### 10.2 Metric redesign — `r_e` measured directly, never backed out through the blend

**Old design (§3.7, unchanged; Rev 6, rejected):** log `a_e` (post-blend,
λ-dependent), then either threshold it flat (§3.7 original) or invert it
through the nonlinear `a_e(λ,r)` formula to recover `r_e` (Rev 6) —
both routes depend on `λ` and on the `‖anchor‖=1` assumption, and the
inversion route additionally has a disclosed Jensen's-gap issue
(§9.7.10 item 5).

**New design (this wave): log `r_e` directly, upstream of the blend,
per entity, per resample:**

```
r_e = mean over n_resamples of  cos( k_eff_raw[e] (pre-blend, this resample), anchor_table.weight[e] )
```

computed via `F.cosine_similarity` exactly as
`measure_full_pool_alignment` already does for `a_e`
(`key_anchoring.py` L690) — same function, same resampling machinery,
**pointed at the pre-blend tensor instead of the post-blend one**. This
is a one-line change to an already-audited eval-only code path, not new
arithmetic. Two direct consequences:

1. **`r_e` does not depend on `λ` at all** — it is measured before the
   blend touches anything, so there is no λ-degeneracy to correct for
   (the entire problem §9.6/§9.7 existed to fix is structurally absent
   from this metric, not patched around).
2. **`r_e` does not depend on `‖anchor_table.weight[e]‖`** —
   `F.cosine_similarity` normalizes both its arguments internally
   (confirmed by reading `torch.nn.functional.cosine_similarity`'s
   own normalization step, not assumed), so an anchor row at norm 1.34
   (the attack's plausible value) or any other norm produces the
   **identical** `r_e` as a unit-norm row would. The attack's finding 3
   is closed by construction for this metric, not merely disclosed.

`a_e` (post-blend) is **still logged too** — it remains a required
result field for backward comparability with §3.7/§9.6/§9.7's own
tables — but it is demoted to a secondary, diagnostic quantity;
`r_e` is the new registered driver of `engaged_frac`.

#### 10.2.1 Anchor-row norm instrumentation (closes attack finding 3/4 as a disclosed diagnostic, no longer load-bearing)

Even though `r_e` no longer depends on anchor-row norm, the norm itself
is still logged — both because it is now genuinely free (the parameter
is already in memory at every checkpoint write, §9.7.5(a)'s own
costing) and because a large, unexplained norm drift is independently
interesting training-dynamics information the project has never
actually measured:

```
anchor_row_norms = anchor_table.weight[anchor_train_ids].norm(dim=-1)   # (107,) vector
```

logged as **both** the full 107-vector and its mean/min/max, at
**every** admission checkpoint (not final-only — cheap, and it lets the
write-up show a norm-drift trajectory the way λ already has one). This
closes attack finding 3/4 (the `m≈1.34` scenario) empirically and
permanently: whatever the true anchor-row norms turn out to be, they
are now on the record, for the first time in this design's history.

### 10.3 The engagement test — derived before any of this wave's data exists

**Null hypothesis, per trained entity `e`:** `k_eff_raw[e]`'s direction
carries no anchoring-mechanism-attributable pull toward
`anchor_table.weight[e]`'s direction — i.e., `r_e` is no different from
the cosine between `e`'s raw key and **some anchor row it was never
trained to align with**.

#### 10.3.1 Primary test — closed-form null, zero data dependency

For a direction drawn independently of a fixed reference axis in
`R^{d_state}`, `d_state=64`, the exact null distribution of the cosine
is `(1+r)/2 ~ Beta((d_state-1)/2, (d_state-1)/2) = Beta(31.5, 31.5)`
(a standard closed-form fact, not an asymptotic approximation — the
`Var(r) = 1/d_state` identity §9.7.3 already used and the attack round
did **not** fault is the second moment of this same exact
distribution). At `d_state=64` this is close to, but not exactly,
Gaussian — and the test needs to resolve a Bonferroni-level tail
(`α/107 ≈ 0.00047`), where the Gaussian approximation's own error is
largest. **This wave uses the exact Beta CDF for the p-value, not the
Gaussian z-approximation** (`scipy.stats.beta.cdf`; the familiar
`z = r·√64` framing is retained only as a disclosed, secondary,
intuition-level cross-check, never as the number a threshold is pinned
to) — the kind of "use the exact structural threshold, not a
tolerance/approximation copied from elsewhere" discipline this
project's own Hard Rules already require for integer/structural
checks, applied here to a continuous one.

Per-entity one-sided p-value (directional: the hypothesis is "moved
*toward*," not "moved"):
`p_e = 1 - Beta_CDF( (1+r_e)/2 ; 31.5, 31.5 )`.

**Multiple-comparisons correction — pre-registered, no menu:**
Benjamini–Hochberg step-up at `q=0.05` across the **107** simultaneous
per-entity tests (sort `p_(1) ≤ ... ≤ p_(107)`; find the largest `k`
with `p_(k) ≤ (k/107)·0.05`; declare entities `1..k` "engaged"). BH is
the pre-registered **primary** procedure (standard for this class of
many-simultaneous-test problem, adaptive to effect size); the
**Bonferroni family-wise cross-check** is **always reported
alongside**, never substituted in post-hoc.

**Exact-Beta primacy everywhere (REVISED at Rev 7.1 — attack-R7
finding 8/§10.12):** the operative Bonferroni cross-check number is the
**exact-Beta quantile `r ≥ 0.4009`** (equivalent z in chance units:
`0.4009·8 ≈ 3.207`), computed from
`1 − Beta_CDF((1+r)/2; 31.5, 31.5) = 0.05/107` — pinned in
`REV7_THRESHOLD_PINNED.json`. The familiar Gaussian framing
(`z_crit = Φ⁻¹(1−0.05/107) ≈ 3.3095`, `r ≈ 0.4137`) is a **disclosed
approximation only, never the operative number anywhere in §10**. The
gap is quantified, not just named (from the pinned reference table,
independently reproduced by the attack round): the exact Beta's
bounded-support tails are *lighter* than Gaussian and the gap grows
with `r` — the true p is 0.83× the Gaussian p at `r=0.35`, 0.70× at
`r=0.40`, and 0.12× at `r=0.581` (the largest `r_e` the prior wave
back-solved). Practical consequence, stated plainly: the registered
exact test is *more powerful* at the effect sizes already observed
than the Gaussian reference number suggests — this cuts toward
*easier* detection, which is exactly why the effect-size floors in
§10.3.4 exist.

**Dependence assumption, stated rather than borrowed (Rev 7.1 —
attack-R7 finding 13):** plain BH provably controls FDR under
independence or PRDS (Benjamini–Yekutieli 2001); the 107 per-entity
tests share one trained model and are not certifiably independent. So
the **Benjamini–Yekutieli (arbitrary-dependence) discovery count is
always reported alongside BH** (correction factor
`Σ_{i=1}^{107} 1/i ≈ 5.247`, effective `q_BY ≈ 0.00953` — pinned in
the same artifact), in exactly the same always-reported-never-
substituted role as the Bonferroni cross-check. A large BH-vs-BY
discovery gap is disclosed as sensitivity-to-dependence in the
write-up; the registered primary remains BH.

All of these numbers are fully computable **today**, from `n=107` and
`d_state=64` (registered since before Wave 1) and a generic
`α=q=0.05` (not tuned to this problem in any way) — there is nothing
left for a reader, an attack round, or an orchestrator to choose once
this wave's data exists, closing Rev-6-attack findings 1 and 2
completely rather than replacing a 3-item menu with a 1-item one.

#### 10.3.2 Secondary validation — an empirical, in-run null (never a substitute, a disclosed consistency check)

The exact-Beta null in §10.3.1 assumes raw keys behave like
independent random directions relative to a non-corresponding anchor.
This is checkable **directly, from data this wave produces, at zero
extra training cost**, rather than left as an assumption a fourth time.

**Construction — mismatched anchor pairs, not an entity's own held-out
row (§10.1.1 explains why the latter is degenerate). REWRITTEN at
Rev 7.1 (attack-R7 finding 4 — the M2 Jensen recurrence):** Rev 7
specified `C[i,j] = cos(mean-resample raw key of i, anchor[j])` — a
**cosine-of-the-mean** — and asserted its diagonal "reproduces" `r_e`,
which is a **mean-of-cosines** (§10.2, matching
`measure_full_pool_alignment`'s own averaging order,
`key_anchoring.py` L690–691). The attack round correctly showed these
are different statistics whenever raw keys vary across resamples —
which is this design's entire premise — the *same* Jensen's-gap
mechanism §9.7.10 item 5 already caught once in `a_e`-space. **Fix:
the one-matmul optimization is dropped. `C[i,j]` is defined as
MEAN-OF-COSINES, identically to `r_e`:**

```
C[i,j] = mean over n_resamples of cos( k_eff_raw[i] (pre-blend, this resample), anchor_table.weight[j] )
```

— the same per-resample cosine, the same `.mean()`, the same code
path, just swept over all 107 anchor columns instead of only the
matching one. Cost: `107 × 107 × n_resamples ≈ 3.7e5` cosines of
dim-64 vectors at `n_resamples=32` — trivial on CPU (well under a
second), so nothing about the "no new forward pass" claim changes:
the per-resample raw keys are already collected for `r_e` itself.
**Consequence, stated as the attack round demanded: the
diagonal-equivalence `C[e,e] = r_e` is no longer an assertion — it is
an identity, because the diagonal IS the §10.2 statistic computed by
the same function.** The **off-diagonal**, `107×106 = 11,342` entries,
is a large empirical sample of "a real, trained raw key's cosine
against a real, trained anchor it was never pushed toward" — the
actual null hypothesis, measured on the real (possibly non-uniform,
non-idealized) raw-key geometry rather than an idealized random
direction. **Optionally widened, same cost class:** held-out entities'
raw keys (real, non-degenerate — only their *own* anchor row is zero,
per §10.1.1) against every trained anchor row adds ≈`106×107 ≈ 11,342`
more valid null pairs, since a held-out key is guaranteed
(mechanically, via the M1 bypass's zero-gradient property, §2.2) never
to have been pulled toward any anchor, trained or not.

**Pooled decision rule — pre-registered, mechanical, both branches
specified now:** compare the off-diagonal pool's empirical `(mean, SD)`
against the analytic null's `(0, 0.125)`. If `|mean_off-diag| ≤ 0.03125`
(`≤ 0.25·σ_chance`) **and** `SD_off-diag ∈ [0.100, 0.156]` (`±25%` of
`0.125`) — both round, symmetric, pre-committed tolerances, not tuned
after seeing the number — the exact-Beta test (§10.3.1) is confirmed
and **remains the reported primary result**. If either tolerance is
violated, that mismatch is disclosed as a finding in its own right (the
real raw-key/anchor geometry deviates from the idealized model), **and
the primary result switches to the empirical permutation p-value**,
`p_e = (1 + #{(i,j): i≠j, C[i,j] ≥ r_e}) / 11,343` (a standard,
provably-conservative rank-based p-value — now an apples-to-apples rank
since both sides are the same mean-of-cosines statistic), through the
**same** BH step-up at `q=0.05` — never re-litigated as a fresh
judgment call at that point, because the switch rule is registered
here, before any number exists. (Resolution note: at ≈11,342 samples
this empirical test comfortably resolves p-values below the Bonferroni
tail `4.7e-4` [floor ≈`1/11,343 ≈ 8.8e-5`], unlike a ~107-sample
held-out-own-row scheme, which would have floored around `1/108` and
been underpowered in exactly the regime this test needs resolution — a
second, independent reason the mismatched-pair construction was chosen
over the discarded held-out-own-row idea in §10.1.1.)

**Per-entity null-validation layer (NEW at Rev 7.1 — attack-R7
finding 5, the hub-contamination scenario):** the pooled `(mean, SD)`
check alone cannot catch a small minority of anisotropic "hub"
entities whose raw keys correlate with *every* anchor row for reasons
unrelated to anchoring — the attack constructed a 5/107-hub scenario
that shifts the pooled mean by only ~13% of the tolerance band while
contaminating those 5 entities' own engagement declarations 100%.
This is the identical aggregate-masks-a-minority blind spot §3.7 was
built to close at the outcome level, and it is closed here at the
null-validation level with two registered additions, both re-slices of
the same `C` matrix (no new computation class):

1. **Per-entity empirical percentile, reported for all 107 entities:**
   each entity's `r_e = C[e,e]` is ranked within **its own row** of
   106 mismatched-pair cosines —
   `p_emp_e = (1 + #{j≠e: C[e,j] ≥ r_e}) / 107`. Resolution floor
   `1/107 ≈ 0.0093` per entity, so this is a **validation layer, not a
   replacement primary** (it cannot resolve Bonferroni-level tails; BH
   on the §10.3.1 p-values remains primary, unchanged). Its job is
   directional: an entity BH-declared "engaged" whose own-row
   percentile is unremarkable (`p_emp_e > 0.05`) is flagged as a
   **null-model discrepancy** for that entity, reported entity-by-
   entity in the write-up.
2. **Hub-detection diagnostic, registered threshold (AMENDED at Rev 7.1's
   bounded verify, `KEYANCHOR_REV7_ATTACK.md` §16.5 — non-blocking MINOR,
   folded in before build per the coordinator's clearance note): compute
   each entity's own mismatched-row mean `m_e = mean_{j≠e}(C[e,j])`, then
   the across-entity **median** and **MAD** (median absolute deviation)
   of `{m_e}` — `median_rowmeans = median({m_e})`,
   `mad_rowmeans = median(|m_e - median_rowmeans|)`, unscaled. Any entity
   with `m_e > median_rowmeans + 2·mad_rowmeans` is **flagged as a hub**.
   §16.5's own simulation showed `mean + 2·SD` (the original Rev 7.1
   text) has NO formal breakdown-point guarantee and is provably masked
   by a contaminated fraction around 20–25/107 (own sweep: threshold
   crosses the constructed hub effect size 0.35 between n_hub=20 and
   n_hub=25); `median + 2·MAD` has a 50% breakdown point and does not
   share this failure mode, while leaving the ORIGINAL constructed
   5/107-hub scenario (§16.5's own re-check) still correctly flagged.
   `r_min_partial = 2·σ_chance` (§10.3.4) is UNCHANGED by this
   amendment — that constant is a fixed theoretical quantity on the null
   scale, not an empirical dispersion estimate, and was never subject to
   this masking mechanism.** Flagged entities are **reported separately, never silently pooled**:
   their engagement declarations are annotated as
   `hub_flagged: true` in the result JSON and the arm's
   `engaged_frac_v3` is reported **both** with and without flagged
   entities (the with/without pair is a required result field; if the
   two differ enough to change the §10.3.4 band, the band is assigned
   from the **excluding-hubs** figure and the discrepancy is disclosed
   as a named caveat — registered now, before any data exists).

`engaged_frac_v3(arm, seed) = (# entities declared "engaged" by
whichever branch is primary) / 107`, routed through the **NEW §10.3.4
bands — NOT the old §3.5/§3.7 magnitude bands** (REVISED at Rev 7.1 —
attack-R7 finding 6: a BH discovery rate is a different statistical
object from the old `a_e ≥ 0.9` magnitude fraction, and the 90/50 band
edges do not port between them unexamined; §10.3.4 re-derives the
bands for the new quantity and adds the minimum-effect-size floor).

#### 10.3.3 The hash-lock — writer/gate/readout triple (COMPLETED at Rev 7.1 — attack-R7 findings 2 and 3)

**Correction of record (attack-R7 FATAL):** Rev 7's text claimed the
pin artifact was "written and committed as part of this draft" when
neither file existed anywhere in git history — a verify-before-claiming
violation, caught by the attack round in under a minute
(`git show --stat 09eb392`). **As of Rev 7.1 the claim is true:**
`matrix-thinking/deltanet_rd/rev7_threshold_derive.py` (pure Python —
Lentz continued-fraction incomplete beta and an erf-bisection normal
quantile, no scipy/torch, so it runs in any attack-round sandbox) and
`matrix-thinking/deltanet_rd/REV7_THRESHOLD_PINNED.json` are committed
with this revision, and smoke 5's data-independence check has actually
been run and passed (derived block byte-identical between a repo-cwd
run and a fresh empty-sandbox run with no wave data present). Pinned
operative values: exact-Beta Bonferroni `r_crit = 0.4009`
(`z`-equivalent 3.207); Gaussian disclosed-approximation `z = 3.3095` /
`r = 0.4137`; BH `q = 0.05`, `n = 107`, full step-up threshold vector;
BY factor 5.247; `σ_chance = 0.125`; effect-size floors
`r_min_partial = 0.25 = 2σ_chance` (derived) and
`r_min_headline = 0.35` (registered literal, provenance in §10.3.4);
consistency reference `r_at_bh_median_rank = 0.2437`. Every one of
these matches the attack round's own independent pure-Python
re-derivation (`KEYANCHOR_REV7_ATTACK.md` §1) exactly.

Unlike `BANDS_PINNED.json` (§3.6), which requires 6 completed
reference-arm runs before it can be written, **every number in
§10.3.1's primary procedure requires zero data of any kind** — only
`n_train=107` and `d_state=64` (both fixed since before Wave 1) and the
generic constant `α=q=0.05`. The script's `derive()` is a pure function
of those three numbers; the only file it ever reads is its own source
(to hash itself into the pin's provenance block).

**Mechanical enforcement — the full three-part gate, matching §3.6's
and §10.10's own standard (COMPLETED at Rev 7.1 — attack-R7 finding 3
correctly showed Rev 7 had only the writer plus a smoke; "zero data
dependency" is a property of the formula, "tamper-evident" is a
property of the pipeline, and both are required):**

1. **The writer.** `rev7_threshold_derive.py` writes
   `REV7_THRESHOLD_PINNED.json` containing the deterministic `derived`
   block plus a provenance block (its own source sha256, timestamp).
   Committed with this revision. *Failure mode:* the derivation cannot
   silently drift — any edit to the script changes its hash, which
   gate 2 catches.
2. **The launcher gate.** Every anchor-arm cell — candidate (d) AND
   (d′) — **REFUSES to launch** unless: (i) `REV7_THRESHOLD_PINNED.json`
   exists and parses; (ii) `sha256(rev7_threshold_derive.py)` in the
   working tree matches the hash recorded inside the pin; (iii) a live
   re-run of `derive()` reproduces the pin's `derived` block
   byte-identically. Same loud-refusal pattern as §3.6's
   `BANDS_PINNED` gate. *Failure modes:* missing/corrupt pin → refusal
   with the instruction to regenerate and re-commit; script-hash
   mismatch (the script was edited after pinning — even innocuously,
   e.g. during the (d′) build) → refusal flagged as a pin-integrity
   error, never silently re-derived; `derived`-block mismatch on
   re-run → same refusal (catches an environment-dependent numeric
   drift, which the pure-Python implementation exists to preclude).
   An explicit override exists and **automatically demotes every
   affected readout to descriptive tier, stamped into each run's own
   result JSON at assembly time** — verbatim the §3.6/Rev-5
   override-demotion machinery, reused not reinvented.
3. **The readout assertion.** The readout/analysis script **loads its
   BH/Bonferroni/BY constants and effect-size floors FROM the pinned
   file** — never recomputes them inline — and asserts (i) the pin's
   recorded script hash matches the committed script at readout time,
   and (ii) the pin's timestamp strictly precedes the earliest
   anchor-arm `started_at` across the wave's result JSONs (the same
   timestamp assertion §3.6 leg 3 already uses). *Failure mode:*
   violation → the readout aborts, the wave summary marks the pin
   integrity as broken, and every affected readout reports at
   descriptive tier only — a recorded, tier-demoting event, not a
   judgment call.

#### 10.3.4 Bands for a discovery-rate quantity — re-derived, with an effect-size floor (NEW at Rev 7.1 — attack-R7 finding 6)

**Why the old bands cannot port:** the old `engaged_frac` (§3.7) was a
**magnitude** fraction ("what share of entities have `a_e ≥ 0.9`");
`engaged_frac_v3` is a **BH discovery rate** ("what share of entities
have `r_e` statistically distinguishable from zero at controlled
FDR"). For any nonzero true effect, however small, a discovery rate
approaches 100% as measurement precision grows (`n_resamples` up to 32
here) — so a discovery rate alone can manufacture "engagement" out of
raw statistical power, the same species of over-claiming
`KEYANCHOR_REV6_ATTACK.md` §5 flagged for the old metric, reintroduced
through a different door. The fix is a **joint criterion**: band
grades couple the detection-rate leg with a **minimum-effect-size
floor** in `r`-space. Under the global null the expected discovery
rate is ≈`q·(fraction of true nulls) ≈ 0.05` — so both registered
detection legs (50%, 90%) sit an order of magnitude above chance-level
discovery; the magnitude legs then prevent power alone from earning a
band.

**Registered bands (no menu — each constant's derivation stated):**

- **A-grade (headline):** BH discovery rate ≥ **90%** (3/3 seeds, per
  §10.6's aggregation rule) **AND** median `r_e` (over all 107
  entities, not only discoveries) ≥ **0.35**.
- **A″-grade (partial anchoring, named, no headline):** discovery rate
  ≥ **50%** **AND** median `r_e` ≥ **0.25**.
- **C (not engaged):** discovery rate < 50% **OR** median `r_e` <
  0.25.

**Derivations, one per constant:**
- `0.25 = 2·σ_chance = 2/√64` **exactly** — §3.6's own registered
  `mean + 2s` multiplier applied to the null scale; independently, the
  `r` at which the exact-Beta p equals the BH step-up threshold at the
  median rank (`k=54` of 107: `p = (54/107)·0.05`) computes to
  `r ≈ 0.2437` (pinned as a consistency reference) — two separate
  derivations landing at ≈0.25, neither fit to any data.
- `0.35` = the prior confirm-wave's **cross-leg median-of-medians** of
  back-solved `r_e` (0.350/0.326/0.359/0.371, §9.7.2's published
  table → 0.3545), rounded down to the 0.05 grid. Registered meaning:
  **a headline can never be earned by a fresh wave whose typical
  per-entity effect is weaker than the wave that was assigned
  Outcome C** — the floor uses only already-published prior-wave
  summaries (fixed before any Rev-7 data exists; blinding is
  unaffected) and makes the A-grade strictly harder than "reproduce
  the old effect with more statistical power."
- The 50%/90% detection legs are carried from §3.5/§3.7's banding
  *structure* (majority / near-total) but are now explicitly the
  detection HALF of a joint criterion, not the whole criterion — the
  part of the old bands that survives is the tiering shape, not the
  claim that a rate alone suffices.

Both floors, both detection legs, and the consistency reference are
pinned in `REV7_THRESHOLD_PINNED.json` (§10.3.3), inside the same
hash-locked artifact as the test constants themselves.

#### 10.3.5 Pre-registered expectation on prior-like data (Rev 7.1 — attack-R7 finding 7, turned from omission into disclosure)

The attack round ran the check Rev 7 should have run itself:
reconstruct 107-entity `r_e` distributions consistent with the only
published prior-wave summaries (K=32 s0: `r_e ∈ [0.095, 0.581]`,
median ≈0.350, "bell-shaped", §9.7.4) and push them through this exact
registered procedure. **Reproduced independently for this revision
(500 draws each, same machinery as the pinned script):** triangular
reconstruction → mean `engaged_frac_v3` **0.888** (range 0.757–0.963);
uniform reconstruction → **0.735** (0.533–0.888) — matching the attack
round's own 0.889/0.732 within simulation noise. **Expectation stated
now, before any Rev-7 data exists:** if the fresh wave's data resembles
the prior wave's, the *detection* leg lands mid-to-high **A″, brushing
the 90% A-grade edge** — the new bar does **not** doom candidate (d) by
construction, and this is disclosed up front rather than discovered
post-hoc in either direction. On the *magnitude* leg, the same
reconstructions give median `r_e ≈ 0.34` — **above the A″ floor (0.25),
below the A floor (0.35, which is by construction the prior wave's own
median-of-medians)** — so prior-like data lands **A″ overall, not A**:
the headline requires the fresh wave to show a typical effect
*exceeding* anything the prior wave produced, not merely to re-detect
the prior effect with better instruments. The fresh wave decides with
fresh seeds; this paragraph is the registered prior expectation it is
read against, and full transparency here is the attempt-#3 defense in
practice: the machinery's predicted behavior on prior-like data is on
the record before the data that counts exists.

### 10.4 Scope — which runs this wave produces, which it does not touch

**New data only.** This wave trains fresh cells with the new
instrumentation (§10.2/§10.3) wired directly into the training loop
(matching the confirm wave's own upgrade from a separate probe model to
in-loop instrumentation, §9.6). **The 4 confirm-wave JSONs
(`experiment-runs/2026-07-05_keyanchor_confirm/.../wkeyanchor-confirm_*.json`)
and the 12 original Wave-1 JSONs are never reopened, re-read for new
fields, or rescored** — none of them logged `r_e`, the pairwise cosine
matrix, or per-row anchor norms, so rescoring them under §10.3 is not a
policy choice this draft declines to make, it is data that was never
collected and cannot be recovered from an archived scalar (`a_e`)
that already discarded the information §10.3 needs. §9.5's UNASSIGNABLE
verdict, §9.6's Outcome C, and §9.7/`KEYANCHOR_REV6_ATTACK.md`'s
rejection all stand, untouched, as the historical record.

### 10.5 Arms

**Registered manifest strings (Rev 7.1 — attack-R7 finding 11's minor
gap):** wave name **`keyanchor-mech`** (already implied by §10.10's
checkpoint directory `wavekeyanchor-mech`, now explicit); candidate
(d′)'s arm string **`dprime`**. Fresh seeds alone already make
filename/`is_done()` collisions impossible under the harness's naming
scheme (the attack round verified this directly against
`run_deltanet_rd_exactness_sweep.py::_spec/is_done`), but the strings
are pinned anyway, and a "smoke C"-style zero-collision assertion
(§10.9 item 13) mirrors the confirm wave's own discipline rather than
leaving it implicit.

**Candidate (d) re-run — global scalar λ, full new instrumentation,
NEW seeds (not a reproduction of 0/1/2 a third time — §9.6's own
disclosure flagged that reusing seed integers does not add a fresh
statistical sample):**
- K=32, seeds `{10, 11, 12}` — 3 runs, 20,000 steps each.
- K=16, seed `10` — 1 spot-check run (supplementary, per §3.5's own
  K=32 scope note — not separately outcome-lettered, matching §9.6's
  precedent).

**Candidate (d′) — per-entity learned λ (NEW arm, §10.5.1):**
- K=32, seeds `{20, 21, 22}` — 3 runs, 20,000 steps each. K=16 is
  **not** run for this arm (K=16 already saturates `h4≈1.0` at
  baseline — no headroom left to distinguish a mechanism effect there;
  K=32 is the only regime where this design has ever shown daylight
  between candidates, §9.4).

**Reference arms — reuse `BANDS_PINNED.json` if its mechanical
validation still passes (§3.6's own launcher re-hash, unchanged,
exactly as §9.6 already did with zero new reference-arm spend):**
`BANDS_PINNED.json`'s post-NS-drift bands are **not load-bearing for
this wave's primary question** (§10.3's engagement test does not
consult them at all — it is derived independently of any reference-arm
data) — they only feed the same non-gating §3.5 B1/B2 sanity context
§9.6 already reported. **Contingency, registered now, only if the
mechanical hash-check fails** (e.g., a reference JSON was moved,
truncated, or the harness schema changed): launch exactly **2** fresh
reference arms, **1 new seed per K** (seed 4), and fold them into the
existing 3-seed set per K (`n=4`, `df=3`, tighter RSE than the
registered n=3 formula per the same argument §3.6 already used for
n=2→n=3) — never a full 6-arm re-derivation from scratch, since 3 valid
reference seeds per K already exist and are not invalidated by this
wave's own metric change.

**Gate-1-style pre-launch probe for the new architecture only:**
candidate (d′), K=32, 5,000 steps, 1 run — the same fixed-diagnostic
pre-spend check §9.6 already ran for candidate (d) (predicted `h4 ≥
0.8` bar); candidate (d)'s own architecture is unchanged from Wave 1/
the confirm wave, so its Gate 1 read is **not** re-run (would be
re-measuring a gate that already passed twice on identical code).

**REGISTRATION CORRECTION (2026-07-05, first live firing):** the `≥0.8`
bar above was mis-inherited — it is the K=16-era Gate-1 PREDICTION bar
(where K=16 realized ≈0.95+, so 0.8 was a meaningful screen), applied
here to a K=32 measured probe. At K=32 the wave's own registered
success bar is `h4 ≥ 0.5` and candidate (d)'s realized range is
0.556–0.665; demanding a probe of the NEW arm exceed 0.8 requires it to
beat the wave's success criterion by 0.3 — unpassable for any correct
prediction, and internally inconsistent with this section's own bars.
Corrected operative bar: **the wave's registered K=32 bar, 0.5**
(`KEYANCHOR_MECH_GATE1_BAR` in the gate code). The gate fired correctly
at first firing (probe read 0.6261 < 0.8 → refused; fail-closed held),
the mis-calibration was diagnosed from the registered documents, and
the correction is made BEFORE any anchor-arm cell has run — the probe
itself is a harness-validation cell, not an evidentiary one. Probe
verdict under the corrected bar: 0.6261 ≥ 0.5 → PASS (and the reading
matches candidate (d)'s known K=32 range — the harm-screen's purpose,
refusing GPU spend on a doomed arm, is affirmatively served). Four
adversarial rounds reviewed the new threshold machinery and did not
catch this inherited constant; logged as a lesson (K-dependent bars
must be re-derived, never carried across K).

#### 10.5.1 The per-entity-λ decision — why both (a) and (b), not either/or

The attack's own framing offered a choice: (a) redesign the metric
around what CAN vary under a global λ (`r_e`, done — §10.2/§10.3), or
(b) add a per-entity `λ_e` variant so differential engagement is
structurally possible. **This draft does both, and they are not
redundant:**

- (a) alone fixes how honestly candidate (d)'s *existing* mechanism is
  measured — it removes the anchor-norm assumption, the λ-dependence,
  and the menu-laundering risk. But per the attack's own §5, under a
  **single global scalar** λ, `r_e`'s only source of per-entity
  variation is **pre-existing, incidental correlation between two
  independently-trained free parameters** (the raw-key subspace and
  the anchor row) — the intervention itself cannot differentially
  pull one entity harder than another. A clean measurement of that
  quantity is real progress (it answers "is candidate (d)'s r_e
  distribution distinguishable from chance" honestly, for the first
  time) but it **cannot** answer whether §3.7's bimodal
  recruited-vs-not framing is even a coherent target for this class of
  mechanism — it structurally never could differentiate, engaged
  metric or not.
- (b) is the only way to test that question directly: give the
  mechanism **genuine per-entity capacity** and see what SGD does with
  it. Two outcomes are both informative and neither is assumed: real
  differential engagement (a positive, structural finding beyond
  anything Rev 6 could produce even in its best case), or convergence
  to a near-uniform `λ_e` despite full freedom (a **stronger** null
  than anything (a) alone could show — it rules out "parameter sharing
  artifact" as the explanation for uniform low engagement, closing the
  attack's deepest point definitively in either direction).

**Marginal cost of (b): 3 more K=32 runs (~1.0–1.1 GPU-h, §10.7) —
cheap relative to the diagnostic value, and it is kept as a clearly
separate, independently-labeled arm** (§10.6) so nothing about
candidate (d)'s own re-run result depends on what (d′) shows, or vice
versa.

**Architecture — a minimal, audited extension, not a new mechanism:**

```python
# Replaces self.anchor_lambda_raw (a single nn.Parameter) with a per-entity table,
# same masked gather/scatter/held-out-bypass pattern as anchor_table itself (sec 2.2):
self.anchor_lambda_table = nn.Embedding(vocab_size_total, 1)     # init raw=0.0 -> lambda_e=0.5 for all e
lam_e = torch.sigmoid(self.anchor_lambda_table(key_ids[t_idx]).squeeze(-1))   # (len(t_idx),), NOT a scalar
sub_blend = F.normalize((1.0 - lam_e[:, None]) * k_eff_raw[t_idx] + lam_e[:, None] * anchor_weight[key_ids[t_idx]], dim=-1)
```

Held-out rows: **identical bypass to §2.2** — the masked gather/scatter
never reads `anchor_lambda_table` for a held-out row in either
direction, so the existing NaN-injection unit test (§2.2/§5 smoke 4)
extends verbatim (re-run fresh for this architecture, not assumed to
still pass, §10.9 item 9). Init matches candidate (d)'s own
`λ=0.5` starting point exactly, so any divergence in the final `λ_e`
distribution is attributable to training, not to a different starting
condition. Param cost: **+107 scalars** (four vocab-length-embedding
rows' worth of memory, ×1 column instead of ×64 — negligible,
`<1KB`), functionally free next to the existing 3.22M-param anchor
table.

**New checks, registered now:**
- **Per-entity `λ_e` interior-band fraction** — extends §3.2's 3-part
  rule (final value + last-5-point mean + range `<0.1`) **per entity**
  instead of to one shared scalar; reports the fraction of the 107
  entities landing in the `[0.2,0.8]` interior band, mirroring
  `engaged_frac`'s own fraction framing.
- **Spearman rank correlation, final `λ_e` vs. final `r_e`, across the
  107 entities** — two-sided, `α=0.05`, a **single** test (no BH
  correction needed — one test per seed, not 107) asking whether the
  per-entity blend weight tracks per-entity raw-key/anchor alignment in
  either direction (more anchor reliance for poorly-aligned entities,
  or less — sign not pre-specified, either is informative).
- **Hartigan's dip test, formalized** (the attack's §9.7.10 item 5
  explicitly flagged Rev 6's max-gap check as "informal... a candidate
  for a proper Hartigan's dip statistic if more rigor is wanted") — run
  on both the `λ_e` distribution (candidate (d′) only, where bimodality
  is structurally possible) and the `r_e` distribution (both (d) and
  (d′)), `α=0.05` pre-registered, implementation validated against a
  known-bimodal and a known-unimodal synthetic sample **before** it
  touches real data (§10.9 item 11).

### 10.6 Outcome map — routing stated exactly

**Seed-aggregation rule, registered now (a real gap the original §3.7
left unstated, since all 4 confirm-wave legs happened to agree
unanimously and this never had to be resolved — closed here before it
can become a second post-hoc judgment call):** an arm's engagement
band is the band (per §10.3.4's joint detection+magnitude criteria)
that **≥2/3 of its K=32 seeds** land in; Outcome A itself still
requires literal **3/3** per §3.5's own unchanged text (items 1–6 +
λ-interior + the §10.3.4 A-grade criterion, all 3/3) — the 2/3 rule
governs Outcome A″ and Outcome C assignment only, matching the
convention §3.2 already uses for λ-band labeling.

**Candidate (d) re-run — new seeds, new instrumentation, same
mechanism as §9.6 (band column REVISED at Rev 7.1 to the §10.3.4
joint criteria — attack-R7 finding 6):**

| §10.3.4 band (≥2/3 seeds; each = BH rate AND median-`r_e` floor jointly) | Item 5 / item 6 / λ-interior / h4 (3/3, unchanged bars) | Routed Outcome |
|---|---|---|
| A-grade: rate ≥90% AND median `r_e` ≥0.35 (3/3 required) | all pass, 3/3 | **A** — full mechanistic confirmation, headline |
| A″-grade: rate ≥50% AND median `r_e` ≥0.25 | pass | **A″** — partial anchoring, named, no headline |
| C: rate <50% OR median `r_e` <0.25 | pass | **C** — mechanism not engaged (reconfirms §9.6, now immune to the anchor-norm/λ-degeneracy objections that forced Rev 6's rejection AND to the power-manufactured-engagement objection, via the effect-size floor) |
| any band | items 1–6 fail (3/3) | **C** regardless of engagement (unchanged §3.5 rule) |

(If the hub-detection diagnostic (§10.3.2) changes the band between
the with-hubs and excluding-hubs figures, the band is assigned from
the excluding-hubs figure with the discrepancy disclosed — registered
in §10.3.2, restated here so the routing table is self-contained.)

**Candidate (d′) — per-entity λ, its own independent routing, never
merged into (d)'s own Outcome (worse-band row NAMED at Rev 7.1 —
attack-R7 finding 9's minor gap):**

| `engaged_frac_v3(d′)` band vs. (d)'s | Dip test / Spearman (≥2/3 seeds significant) | Routed finding |
|---|---|---|
| higher band than (d) | and/or significant | **A(d′)** — per-entity capacity is used; differential engagement demonstrated (new, positive, structural finding) |
| same band as (d) | not significant | **C′** — structural null: even given genuine per-entity freedom, SGD does not differentiate (a stronger, capacity-controlled negative than (a) alone could produce) |
| **worse band than (d)** | **significant** | **D′ (named at Rev 7.1)** — per-entity capacity actively destabilizes: a real per-entity split exists but training is worse for it — informative structural finding (freedom is used, and used badly), reported in full, distinct from mere ambiguity |
| any other combination | mixed | **Inconclusive/mixed** — reported in full with both pieces of evidence, no forced binary call (mirrors §3.6's own UNRESOLVABLE discipline) |

### 10.7 Budget

**Estimation basis REVISED at Rev 7.1 (attack-R7 finding 12):** the
attack round pulled `wall_s` from every archived 20k-step cell and
found realized per-cell cost spans **0.18–0.77 GPU-h (>4× range)** for
nominally-identical configs — and the *more* instrumented confirm-wave
cells ran *faster* than the less instrumented Wave-1 cells, so the
dominant cost driver is shared-GPU contention/scheduling variance, not
workload. The `×1.3–1.5` unmeasured-code-path convention is the wrong
correction for that (it prices uncounted code, not scheduler noise).
Estimates below are therefore **brackets over the full observed
0.18–0.77 GPU-h per-cell range**, not point estimates:

| Item | Cells | New training runs | Est. GPU-h (bracket) |
|---|---|---|---|
| Wave −1 CPU smoke suite (13 items, §10.9) + `REV7_THRESHOLD_PINNED` derivation (already run) | — | 0 | **0** |
| Gate-1 probe, candidate (d′) only, K=32, 5,000 steps | 1 | 1 | 0.03–0.12 (realized comparable probe: 0.030) |
| Candidate (d) re-run, K=32, seeds {10,11,12}, full instrumentation | 3 | 3 | 0.54–2.31 |
| Candidate (d) K=16 spot check, seed 10 | 1 | 1 | 0.18–0.77 |
| Candidate (d′), per-entity λ, K=32, seeds {20,21,22} | 3 | 3 | 0.54–2.31 |
| **Mandatory baseline** | | **8** | **~1.3–5.5** |
| Reference re-pin contingency (only if `BANDS_PINNED` hash-check fails), 1 seed × K∈{16,32} | ≤2 | ≤2 | 0.36–1.54 |
| Seed contingency (either arm lands in an ambiguous A″ band, +2 seeds, one iteration) | ≤2 | ≤2 | 0.36–1.54 |
| **All-conditionals-max** | | **≤12** | **~2.0–8.6** |

**Registered nominal ceiling for this wave: ≤12 GPU-h** — headroom
above even the all-conditionals worst-case bracket top (~8.6),
reserved for an uninstrumented crash/re-run; the attack round's own
recomputation (worst historical per-cell cost applied to every cell)
independently confirmed the ceiling is not threatened.

**Program arithmetic (states the number, doesn't hide it):** anchoring
program spend so far ≈**51.5/80 GPU-h** against
`DELTANET_RD_EXACTNESS_DESIGN.md`'s own 80 GPU-h cap
(`KEY_ANCHORING_DESIGN.md` §5; consistent with `STATE.md`'s own ≈51/80
figure). This wave's registered ceiling (≤12) brings the worst case to
**≤63.5/80**, leaving **≥16.5 GPU-h reserve** under the exactness
program's cap — comfortable room for either of the other two
DECISION-PENDING options (`STATE.md`: a K=48 F-geo-3 extension, or a
different direction) without threatening the 80 GPU-h ceiling, and no
call on the separate `SCALE_TRANSFER_DESIGN.md` 300 GPU-h program,
which this wave does not touch.

### 10.8 Falsification map

| Result | What it means |
|---|---|
| Candidate (d) clears the §10.3.4 A-grade (BH rate ≥90% AND median `r_e` ≥0.35), 3/3 seeds, items 1–6/λ/h4 all pass | **Outcome A** — the key-anchoring interaction hypothesis (§1) is confirmed for the first time at full mechanistic tier; the behavioral h4 gain is attributable to majority-entity key stabilization at an effect size exceeding anything the prior wave showed |
| Candidate (d) lands A″-grade (rate ≥50% AND median `r_e` ≥0.25), ≥2/3 seeds | **Outcome A″** — partial anchoring, real but incomplete; the §10.3.5 expectation says this is where prior-like data lands, so an A″ here is consistent-with-prior, not a new mechanistic tier; follow-on is understanding *why* engagement is partial (capacity, LR, training length — §9.7.8's own open question, now answerable with real per-entity data instead of an inverted proxy) |
| Candidate (d) lands C (rate <50% OR median `r_e` <0.25), ≥2/3 seeds | **Outcome C reconfirmed** — this time immune to the objections that sank Rev 6 (no menu, no anchor-norm assumption, no λ-degeneracy) AND to power-manufactured engagement (effect-size floor) — a genuinely stronger negative than §9.6's own, not a repeat of it; note the §10.3.5 expectation makes this outcome *informative against the prior wave's own data pattern*, not merely against chance |
| Candidate (d′) shows real differential engagement (band shift and/or significant dip-test/Spearman) | **Outcome A(d′)** — the per-entity-capacity hypothesis is correct: global λ was the limiting factor, not the anchoring idea itself; motivates a full-scale per-entity-λ wave as the new PRIMARY candidate |
| Candidate (d′) converges to near-uniform `λ_e`, no significant differentiation | **Outcome C′** — the uniform/incomplete engagement pattern is structural, not a parameter-sharing artifact; closes attack finding 5 in the negative direction, with real evidence rather than an inference |
| `REV7_THRESHOLD_PINNED.json`'s integrity chain fails at any leg — the data-independence smoke (§10.9 item 5), the launcher's script-hash/`derive()`-re-run gate, or the readout's pin-precedes-anchor-start assertion (§10.3.3) | **Design-invalidating** — the blind this section is built around is broken before (or while) runs launch; this section itself would need to be attacked and revised again, not the anchor-arm data |
| Anchor-row norms (§10.2.1) drift far from 1 in either direction | Disclosed diagnostic only — does **not** invalidate `r_e`/`engaged_frac_v3` (norm-invariant by construction), but is itself a reportable training-dynamics finding never previously measured |

### 10.9 Wave −1 smoke suite (all CPU, all free, itemized per §5's own
countable-list discipline)

1. **`r_e` norm-invariance positive control** — synthetic anchor rows
   scaled by `{1, 5, 0.1}`; assert logged `r_e` is bit-identical across
   all three (proves the norm-invariance claim in the actual code path,
   not just in algebra).
2. **Anchor-row norm logging correctness** — construct a table with a
   known non-unit-norm row; assert the logged norm matches to float
   precision.
3. **Held-out-row zero-init verification** — read
   `anchor_table_full` at non-train rows post-construction, assert
   exactly `torch.zeros`; assert `F.cosine_similarity` against an
   exact-zero row returns exactly `0.0`, never `NaN` (verifies §10.1.1's
   finding, doesn't just cite it).
4. **Mismatched-pair null-pool construction, WITH resample jitter
   (strengthened at Rev 7.1 — attack-R7 finding 4's fix (3): a
   zero-jitter toy example would let a cosine-of-mean implementation
   pass trivially)** — a tiny synthetic 5-entity example whose
   per-resample raw keys deliberately vary (registered jitter σ large
   enough that mean-of-cosines and cosine-of-mean differ by ≫ float
   tolerance); assert (i) the pairwise matrix's diagonal is
   bit-identical to the §10.2 `r_e` code path's own output (the
   identity §10.3.2 now claims by construction), and (ii) the
   off-diagonal values match hand-computed mean-of-cosines — a
   cosine-of-mean implementation MUST fail this smoke.
5. **`REV7_THRESHOLD_PINNED` zero-data-dependency proof (already run
   once for Rev 7.1, wired in as a permanent smoke)** — run
   `rev7_threshold_derive.py` twice: once in a fresh sandbox with no
   wave JSON present, once from the repo root; assert the `derived`
   block is byte-identical both times and the emitted script hash
   matches the pin's recorded hash. First execution: this revision
   (PASS — repo cwd vs. empty sandbox, identical derived blocks).
6. **BH-FDR step-up + BY correctness** — run on a synthetic p-value
   vector with a hand-verified BH cutoff; assert the implementation
   matches; assert the BY discovery count equals BH run at
   `q/Σ(1/i)` (the pinned `by_effective_q`).
7. **Exact-Beta-vs-empirical decision-rule, both branches exercised** —
   feed a synthetic null sample deliberately outside the registered
   `(mean, SD)` tolerance (assert fallback to the empirical branch) and
   one inside it (assert the analytic branch is used) — proves the
   registered rule is reachable both ways, not assumed.
8. **Candidate (d′) forward/backward** — finite loss and finite
   gradients on every parameter including `anchor_lambda_table`, at a
   realistic and an adversarial (near-duplicate raw keys) input
   (mirrors existing smoke 2, new architecture).
9. **Candidate (d′) held-out bypass + NaN-injection isolation** —
   re-run fresh, not assumed inherited: all-held-out batch bit-identity
   to bare geo3, plus NaN planted in every held-out `anchor_lambda_table`
   row, assert finite gradients everywhere and exact-zero gradient at
   held-out rows (mirrors existing smokes 3/4).
10. **Per-entity `λ_e` interior-band-fraction computation** — synthetic
    107-entity trajectory set with a known interior/non-interior split;
    assert the computed fraction matches by hand-count.
11. **Hartigan's dip-test positive control** — a known-bimodal and a
    known-unimodal synthetic sample; assert correct discrimination at
    the registered `α=0.05` before trusting it on real `r_e`/`λ_e` data.
12. **Checkpoint save/reload round-trip** (see §10.10 for why this is
    now mechanical, not aspirational) — train a tiny stub a few steps,
    save `anchor_table` (+ `anchor_lambda_table` for (d′)) state dicts,
    reload, assert exact tensor equality; wave-closure gate refuses to
    mark the wave complete without every cell's checkpoint passing this
    check.
13. **Zero-collision manifest assertion (NEW at Rev 7.1 — mirrors the
    confirm wave's own "smoke C"; attack-R7 finding 11)** — build the
    full `keyanchor-mech` manifest (both arms, all seeds, the (d′)
    `dprime` arm string) and assert every generated `out_path()` is
    distinct from every existing result file across ALL prior waves
    (`wavekeyanchor`, `wavekeyanchor-neg1`, `waveref`,
    `wavekeyanchor-confirm`), and that `is_done()` returns False for
    every fresh cell before launch.

### 10.10 Checkpoints — mandatory, mechanical, closes the "no checkpoint exists anywhere" gap for good

Both prior waves left **zero** checkpoints anywhere (SSD, box, or
git) — the attack round's `m≈1.34` scenario was unverifiable "not
because closing it was expensive, but because nobody wrote the file."
This wave fixes that as a **mechanical harness gate, matching §3.6's
own precedent, not a stated intention a third artifact could silently
skip**:

1. **The writer.** Every cell in this wave writes, at minimum,
   `anchor_table.state_dict()` (and, for candidate (d′),
   `anchor_lambda_table.state_dict()`) to a checkpoint file at the
   final step (27KB negligible; also at every admission checkpoint for
   the anchor-row-norm/λ_e trajectory, still negligible) — to
   `/root/data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-mech/<cell_name>/`
   on the training box, pulled to
   `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-06_keyanchor_mechanism/checkpoints/`
   **immediately after each run completes**, not deferred to end-of-wave.
2. **The gate.** The wave cannot be marked `KEYANCHOR_MECH_CHAIN_DONE`
   (the confirm wave's own completion-marker convention) until a
   readout-time assertion confirms every cell's checkpoint file exists
   **and** round-trip-loads (smoke item 12, extended to real run
   output) — refusal, not a silent skip, exactly the §3.6 pattern.
3. **The readout.** Any future audit of this wave's anchor-row norms or
   `λ_e` values reads the checkpoint directly — no future attack round
   should ever again have to report "no checkpoint exists anywhere" for
   this design.

### 10.11 Attack surface — 5 pre-answered

1. **"This is attempt #3 at the same claim (Wave 1 → confirm wave →
   Rev 6 → this) — what makes it admissible where Rev 6 wasn't?"**
   Answer: everything load-bearing is derived **before** any of this
   wave's data exists, with a stronger (zero-data-dependency, not
   merely reference-arm-gated) blind than §3.6 itself requires — and
   at Rev 7.1 the pin artifact **actually exists, is committed, and
   passed its data-independence smoke** (§10.3.3; the attack round
   caught Rev 7 claiming this before it was true, and the correction
   is on the record in §10.12 F1, not papered over); the prior waves'
   JSONs are never reopened or rescored — this wave's claim rests
   entirely on fresh data collected under the new instrumentation
   (§10.4); the metric is built for the regime the mechanism actually
   operates in (interior λ, no back-solve through a degenerate blend
   formula, no anchor-norm assumption); and the machinery's predicted
   behavior on prior-like data is **disclosed up front** (§10.3.5:
   detection leg lands A″ brushing A, magnitude leg holds it at A″) —
   the fresh wave is read against a stated prior expectation, in
   either direction, not against silence.
2. **"You still get to pick BH vs. Bonferroni vs. BY, or the
   empirical-vs-analytic fallback rule — isn't that the same
   laundering risk as z=2 vs. z=3?"**
   No: BH is registered as the single primary procedure with Bonferroni
   AND BY as always-reported (never substituted) cross-checks — there
   is no post-hoc choice among them, and the operative cross-check
   number is the exact-Beta quantile (0.4009), with the Gaussian
   figure demoted to a disclosed approximation whose gap is quantified
   in the pin itself. The empirical/analytic fallback rule is a
   **mechanical, symmetric, pre-registered tolerance check**
   (§10.3.2) exercised in both directions by a dedicated smoke (item
   7) before any real data is read — the rule, not a reader, decides
   which branch is primary, and the decision criterion itself needs no
   anchor-arm data to evaluate. The one place a discovery-rate
   quantity could still launder a weak result into a strong-sounding
   band — raw statistical power — is closed by the §10.3.4
   effect-size floors, both pinned in the same hash-locked artifact.
3. **"Candidate (d′) is a whole new architecture — doesn't that risk
   new bugs rather than resolving the old question?"** Acknowledged
   and mitigated three ways: it is a one-line indexing change to the
   exact same masked gather/scatter/held-out-bypass code path already
   audited for the scalar-λ case (§2.2), not a new mechanism; it is
   kept as a separate, independently-labeled arm (§10.6) whose result
   never touches candidate (d)'s own; and it gets its own full smoke
   suite (items 8–11) before any GPU spend, per this project's own
   audit-before-run rule.
4. **"Why trust checkpoint-saving this time when it was skipped
   twice?"** It is now a mechanical writer/gate/readout triple
   (§10.10), the same enforced-refusal class as `BANDS_PINNED`'s own
   blind gate (§3.6) — not a stated intention, which is exactly the
   class of thing that was skipped twice before.
5. **"Does closing this wave quietly rescore the confirm wave's
   Outcome C or Rev 6's rejection after the fact?"** No — both stand,
   untouched, as historical record (§10.4). This wave produces its own
   fresh data and its own fresh Outcome assignment (§10.6/§10.8); if it
   also lands in the §10.3.4 C band, that **confirms** (does not
   merely repeat) Outcome C under a metric immune to the specific
   objections that forced Rev 6's rejection — a null that survives a
   harder, better-instrumented test is stronger evidence, not a
   redundant re-check, and it is reported as such either way — and per
   §10.3.5's registered expectation, a C here would additionally be a
   *surprise against the prior wave's own data pattern*, which is
   disclosed now, not explained away later.

### 10.12 Rev 7.1 — attack-round-R7 response map (finding → change)

`KEYANCHOR_REV7_ATTACK.md` verdict: NEEDS-REV — 1 FATAL, 4 MAJOR, plus
moderates/minors. Every numbered requirement from its §15 list is
addressed; same finding→change table discipline as §8.1–§8.4.

| # | Finding (attack-R7) | Change | Where |
|---|---|---|---|
| F1 | **FATAL** — `rev7_threshold_derive.py` / `REV7_THRESHOLD_PINNED.json` claimed "written and committed" but never existed in git history (verify-before-claiming violation at the center of the admissibility defense) | Both artifacts **actually written and committed with this revision** (`matrix-thinking/deltanet_rd/`); pure-Python (Lentz CF incomplete beta + erf-bisection quantile, no scipy/torch — runs in any attack sandbox); pinned values reproduce the attack round's own independent derivation exactly (r_crit_exact 0.4009, z_gauss 3.3095, by_factor 5.247); smoke 5 run for real (derived block byte-identical, repo cwd vs. empty sandbox); §10.3.3's tense corrected and the correction acknowledged in place, not silently rewritten | §10.3.3, §10.1 pt 1, §10.11 item 1; the two new files |
| M1 (R7 §3) | Threshold-pin enforcement was writer+smoke only — no launcher re-hash, no readout assertion ("zero data dependency" ≠ "tamper-evident") | Full three-part gate specified with failure modes, mirroring §3.6/§10.10: launcher refuses anchor-arm launch unless the pin exists, the script's sha256 matches the pin's recorded hash, AND a live `derive()` re-run reproduces the derived block; readout loads constants FROM the pin (never inline recomputation), asserts script-hash match and pin-timestamp < earliest anchor start; override path reuses the §3.6/Rev-5 descriptive-tier demotion stamping verbatim | §10.3.3 (items 2–3) |
| M2 (R7 §4) | Null-pool `C[i,j]` was cosine-of-mean while `r_e` is mean-of-cosines — diagonal equivalence asserted, not derived; second instance of the §9.7.10-item-5 Jensen's-gap mechanism; smoke 4 had no power to catch it | One-matmul optimization dropped; `C[i,j]` redefined as **mean-of-cosines, computed by the same code path as `r_e`** (~3.7e5 dim-64 cosines, trivial CPU) — diagonal equivalence is now an identity, not an assertion; the empirical fallback rank is now apples-to-apples; smoke 4 rebuilt with registered resample jitter so a cosine-of-mean implementation MUST fail it | §10.3.2, §10.9 item 4 |
| M3 (R7 §5) | Pooled `(mean, SD)` null check is blind to a hub-entity minority (constructed 5/107 scenario moves the pooled mean ~13% of tolerance while contaminating those entities' declarations 100%) — the §3.7 aggregate-masking lesson reproduced inside the null-validation layer | Per-entity null layer added: (i) each entity's `r_e` ranked against ITS OWN 106-value mismatched row (empirical percentile, reported for all 107; validation layer, resolution floor 1/107 disclosed — BH primary stands); (ii) registered hub-detection: `m_e > pooled mean + 2·SD(row means)` → flagged, reported separately, never silently pooled; `engaged_frac_v3` reported with AND without flagged entities; band assigned from excluding-hubs figure on divergence, registered now | §10.3.2 (per-entity layer), §10.6 note |
| M4 (R7 §6) | `engaged_frac_v3` (a BH discovery rate) silently inherited the ≥90%/50% bands calibrated for the old magnitude quantity; no minimum-effect-size floor — raw power could manufacture "engagement" | New §10.3.4: joint bands registered, no menu — A-grade = rate ≥90% AND median `r_e` ≥0.35; A″ = rate ≥50% AND median `r_e` ≥0.25; C = below either. Derivations stated per constant: 0.25 = 2σ_chance exactly (§3.6's own 2s multiplier; independently ≈ the BH median-rank crossing, 0.2437, pinned); 0.35 = prior wave's cross-leg median-of-medians (0.3545 → 0.05 grid) — headline requires exceeding the prior (Outcome-C) wave's typical effect, not re-detecting it with more power. Floors pinned in the same hash-locked artifact | §10.3.4, §10.6, §10.8 |
| R7 §7 | The draft never computed what its own registered procedure implies on prior-like data (the same category of omission that sank Rev 6, on the power side) — attack's simulation: BH detection lands 0.73–0.89, brushing A | Turned into a pre-registered expectation statement: simulation reproduced independently for this revision (0.888/0.735 means, 500 draws, vs. attack's 0.889/0.732); disclosed BEFORE any Rev-7 data exists — prior-like data lands **A″** under the new joint bands (detection brushes 90% but median r_e ≈0.34 < the 0.35 A floor); the fresh wave is read against this stated expectation in either direction | §10.3.5 |
| R7 §1/§8 | Exact-Beta-vs-Gaussian tail gap disclosed only qualitatively; Gaussian numbers (z≈3.305/r≈0.413) appeared as the operative cross-check | **Exact-Beta primacy everywhere**: operative Bonferroni cross-check is r ≥ 0.4009 (exact quantile, z-equiv 3.207); Gaussian demoted to disclosed approximation; the gap quantified in the pin's reference table (exact p = 0.83×/0.70×/0.12× Gaussian at r = 0.35/0.40/0.581) with the practical consequence stated (more power at observed effect sizes — cuts toward easier detection, hence the floors) | §10.3.1, §10.3.3, pin artifact |
| R7 §12 | BH's independence/PRDS assumption unexamined | Stated; BY (arbitrary-dependence) discovery count always reported alongside BH (factor 5.247, effective q 0.00953, pinned); BH-vs-BY gap disclosed as dependence sensitivity; smoke 6 extended | §10.3.1, §10.9 item 6 |
| R7 §8 (minor) | (d′) worse-band-plus-significant result indistinguishable from mere ambiguity in the routing table | Named row added: **D′** — per-entity capacity actively destabilizes (freedom used, and used badly) — distinct from Inconclusive/mixed | §10.6 |
| R7 §10 (minor) | Manifest wave-name / (d′) arm-name strings never explicitly pinned | Pinned: wave `keyanchor-mech`, arm `dprime`; zero-collision smoke added mirroring the confirm wave's "smoke C" | §10.5, §10.9 item 13 |
| R7 §11 (minor) | Budget point estimates not grounded in the two most comparable realized numbers; ×1.3–1.5 margin is the wrong correction for contention variance (realized per-cell range 0.18–0.77 GPU-h, >4×) | Budget re-bracketed over the full observed per-cell range (mandatory ~1.3–5.5, all-max ~2.0–8.6); ≤12 GPU-h ceiling unchanged (independently confirmed unthreatened by the attack round's own worst-case recomputation) | §10.7 |

Cleared by the attack round, no change required: Beta(31.5,31.5)
parameterization and the `Var(r)=1/64` identity (re-derived
independently, exact match); candidate (d′) init/parameterization;
the §10.10 checkpoint writer/gate/readout triple; `is_done()`/seed
collision safety; §10.4's no-retroactive-rescoring claim (verified
against the archived JSONs' actual fields).

---

### 10.13 Mechanism-tier wave VERDICT (2026-07-06) — measured, not projected

**Ran to completion.** 7/7 mandatory cells + the candidate-(d′) Gate-1
probe, all `complete: true`, `steps_completed: 20000` (probe: 5000), zero
timeouts, zero failed-then-recovered. `ALL_DONE` present for both
`wavekeyanchor-mech` and `wavekeyanchor-mech-gate1`. Realized cost: **1.500
GPU-h** total (7 mandatory cells 1.439 GPU-h + Gate-1 probe 0.061 GPU-h) —
well inside the §10.7 mandatory-baseline bracket (1.3–5.5) and a small
fraction of the ≤12 GPU-h ceiling; no contingency cells were needed
(`BANDS_PINNED.json` hash-validated clean, no reference re-pin; no seed
contingency triggered). This verdict pass re-ran `readout_rev7.py` fresh
(CPU-only, `youthful-indigo-turkey`, GPUs 0–1 untouched/running rung-3
throughout) and independently re-extracted every number below directly
from the 8 archived JSONs — nothing here is taken from the launch console
or from the orchestrator's own summary without a JSON-level check.

**Blind integrity, mechanically checked, both legs.** `REV7_THRESHOLD_PINNED.json`:
script hash `a746dec7...bc738` matches the working-tree script exactly; a
fresh `derive()` re-run in a **new, empty sandbox** (no repo, no wave
data) reproduces the pinned `derived` block by direct dict equality, not
eyeballing. Pin `generated_at` (2026-07-05T17:40:49Z, epoch
1783273249) strictly precedes the earliest anchor-arm `started_at`
(1783284449, seed-12 K=32) by **≈3.1 hours** — REV7 PIN BLIND INTACT, 7
runs checked. `BANDS_PINNED.json` (the existing §3.6 gate, reused
non-gating) independently re-validated clean. All 8 JSONs (7 wave cells +
gate1 probe): `unblind_override: false`, no `claim_tier` key anywhere —
no tier-demoting event fired. **Checkpoint gate (§10.10): 70 files checked
(7 cells × 10 admission checkpoints), 0 bad** — independently confirmed by
listing `/data/deltanet_rd_keyanchor_ckpts/wavekeyanchor-mech/` directly
on the box (7 cell subdirectories, 70 `.pt` files total) — the first time
in this design's three-wave history a checkpoint has existed anywhere for
this mechanism, closing `KEYANCHOR_REV6_ATTACK.md`'s "no checkpoint exists
anywhere" gap for good.

#### 10.13.1 Per-cell table (every number re-verified against the raw JSON, not the readout console alone)

| K | arm | seed | h4 `rec@0.9` (bar ≥0.5, K=16: no-regression) | item 5 pre-NS drift (bar ≥0.95) | item 6a σ-ratio (bar ≥0.1) | item 6b max\|cos\| (bar ≤0.5) | λ (final / band) | `engaged_frac_v3` w/hubs | median `r_e` | band (w/hubs = wo/hubs here) | n_hub_flagged | primary branch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | d | 10 | **0.9998** (no-regression, clean) | 0.9999 PASS | 0.2503 PASS | 0.3532 PASS | 0.5830 interior | 0.7383 | 0.2934 | **A_partial** | 8 | exact_beta |
| 32 | d | 10 | **0.6741** PASS | 1.0000 PASS | 0.1603 PASS | 0.3723 PASS | 0.6044 interior | 0.2710 | 0.2225 | **C** | 9 | exact_beta |
| 32 | d | 11 | **0.7125** PASS | 1.0000 PASS | 0.1208 PASS | 0.3505 PASS | 0.5825 interior | 0.4112 | 0.2292 | **C** | 3 | exact_beta |
| 32 | d | 12 | **0.6141** PASS | 1.0000 PASS | 0.1731 PASS | 0.3813 PASS | 0.5787 interior | 0.2710 | 0.1949 | **C** | 11 | exact_beta |
| 32 | d′ | 20 | **0.7021** PASS | 1.0000 PASS | 0.2193 PASS | 0.3775 PASS | λ_e: 100% interior | 0.0748 | 0.1517 | **C** | 5 | exact_beta |
| 32 | d′ | 21 | **0.6661** PASS | 1.0000 PASS | 0.1799 PASS | 0.3821 PASS | λ_e: 100% interior | 0.2430 | 0.1896 | **C** | 7 | exact_beta |
| 32 | d′ | 22 | **0.7141** PASS | 1.0000 PASS | 0.2319 PASS | 0.3843 PASS | λ_e: 100% interior | 0.0374 | 0.1462 | **C** | 1 | exact_beta |

Items 1–4 (`admissible`, `ns_converged_no_fallback`, `finite_loss_no_divergence`,
`task_performance_floor_pass`, `h1_recovered_frac_at_0.9_final: 1.0`) are
`true`/clean for all 7 cells — full admissibility, 7/7. Item 6 (6a AND 6b)
passes **3/3** at K=32 for arm (d) this wave (unlike the confirm wave's
2/3 — seed 11's own σ-ratio, 0.1208, clears the 0.1 bar this time; no
seed here repeats the confirm wave's seed-1 miss, consistent with fresh
seeds drawing a different σ-ratio realization, not a fix to the
underlying quantity). Pooled null-check (§10.3.2) passes its
`(mean, SD)` tolerance for all 7 cells (`mean_ok=true, sd_ok=true`,
11,342 pairs each) — the exact-Beta branch is primary everywhere, the
empirical-permutation fallback is never invoked.

**Anchor-row norms (§10.2.1) — the m≈1.34 confound, now measured, not
assumed:** per-cell mean row norm ranges **1.062–1.159** across all 7
cells (grand mean of per-cell means 1.105; per-entity min/max envelope
0.717–1.266, one low outlier at d′ seed 20's minimum, 0.7175). The
plausible chance-level-scale-artifact value the Rev-6 attack round
floated (m≈1.34) is **not realized** — measured norms sit well below it —
but this is now a moot question for `r_e` specifically, which is
norm-invariant by construction (`F.cosine_similarity`); the norm
instrumentation closes the question as a disclosed diagnostic, not a
load-bearing one.

**Candidate (d′) per-entity λ — genuine freedom, used, but not toward
differentiation.** `λ_e` interior-band fraction is **100% (107/107)** at
all 3 seeds — every entity's individually-learned blend weight lands in
`[0.2, 0.8]`, matching candidate (d)'s own global-λ band. Hartigan's dip
test on `λ_e`: **not significant at any seed** (p = 0.184 / 0.930 /
0.848) — no bimodal split in blend weights. Dip test on `r_e`: **not
significant at any of the 7 cells** (p ranges 0.475–0.938) — confirms
§9.7.4's earlier informal finding with a formal statistic: uniform
partial engagement, not a masked bimodal split, at both K and in both
arms. **Spearman(λ_e, r_e), per seed: significant 2/3** (s20 ρ=−0.413,
p=7.5e-6; s21 ρ=−0.282, p=0.0031; s22 ρ=−0.123, p=0.205, not
significant) — where significant, the sign is **negative**: entities
whose raw key aligns *less* with its anchor get pulled *harder* (higher
`λ_e`) by the mechanism, the intuitive compensating direction, not the
reinforcing one. This is a real, if partial (2/3, not 3/3), signal that
the per-entity mechanism is doing *something* non-degenerate with its
freedom — but it does not lift engagement into a higher band at any
seed, and `λ_e`'s own dip test never finds the bimodal split that would
indicate a genuine engaged/disengaged entity split forming.

#### 10.13.2 Arm-level routing (§10.6 tables, applied literally)

**Candidate (d), K=32 (the wave's primary cell — §10.5's registered
scope):** `engaged_frac_v3` with-hubs = 0.271 / 0.411 / 0.271, median
`r_e` = 0.2225 / 0.2292 / 0.1949 — **all three seeds land band C**
(median `r_e` < 0.25 in all three; the rate leg is moot once the
magnitude leg fails, per §10.3.4's joint-AND criterion). 3/3, not merely
≥2/3. Items 1–6 pass 3/3 (this wave, unlike the confirm wave). λ interior
3/3. **Routed Outcome: C — "mechanism not engaged... reconfirms §9.6, now
immune to the anchor-norm/λ-degeneracy objections that forced Rev 6's
rejection AND to the power-manufactured-engagement objection, via the
effect-size floor."** K=16 (supplementary, not separately
outcome-lettered per §3.5's own K=32 scope note): `engaged_frac_v3` =
0.738, median `r_e` = 0.2934 → **A_partial** — clears the rate leg (≥50%)
and the magnitude leg (≥0.25) but not the headline magnitude floor
(0.35) — consistent with K=16 having no discriminating headroom at
h4≈1.0 (§10.5's own reason for not running K=16 for (d′)) and with
`r_e` itself simply being higher at the easier K.

**Candidate (d′), K=32 (independent routing, §10.6):** `engaged_frac_v3`
with-hubs = 0.075 / 0.243 / 0.037, median `r_e` = 0.1517 / 0.1896 /
0.1462 — **all three seeds land band C**, same as (d), and at a
*lower* median `r_e` and lower detection rate than (d)'s own K=32 cells
in every seed-for-seed comparison (band is identical — C vs. C — so
per §10.6's own table this is **not** "higher band than (d)"). Dip-test/
Spearman: significant in 2/3 seeds (Spearman only; the dip test is never
significant for either `λ_e` or `r_e`, any seed). Per §10.6's table this
combination (same band as (d), *some* but not unanimous significance) is
not a clean fit to either the "same band, not significant → C′" row or
the "higher band, and/or significant → A(d′)" row — it is the
**Inconclusive/mixed** catch-all, reported in full per the routing
table's own no-forced-binary-call discipline. **Read plainly: (d′) does
not demonstrate differential engagement strong enough to move the band,
but it is also not a clean structural null — the partial Spearman signal
is real and is not nothing.**

#### 10.13.3 The registered prior expectation (§10.3.5) — FAILED, and why

§10.3.5 registered, before this wave's data existed: prior-like data
(reconstructed from the confirm wave's back-solved `r_e ∈ [0.095, 0.581]`,
median ≈0.350) would land the *detection* leg mid-to-high A″ brushing the
90% A edge, and the *magnitude* leg at median `r_e` ≈0.34 — A″ overall,
not A, but a materially stronger read than what this wave actually
measured. **The fresh, direct measurement: median `r_e` 0.1462–0.2934
across all 7 cells (0.1462–0.2292 at K=32), engaged_frac_v3 3.7%–41.1% at
K=32** — the registered expectation is **not met**; the fresh wave reads
substantially weaker on both legs than the prior-wave-consistent
reconstruction predicted.

**Diagnosis, as instructed:** the registered expectation was built by
*back-solving* `r_e` through the nonlinear `a_e(λ, r)` blend formula
(§9.7.2) from the confirm wave's *post-blend* `a_e` values, at each cell's
own logged λ — not from a direct measurement (`r_e` did not exist as a
directly-logged quantity until this wave's instrumentation). That
back-solve carries exactly the two bias classes this design's own attack
rounds flagged and never fully closed before Rev 7.1: (i) it assumed
`‖anchor_weight[e]‖ = 1` (§9.7.5's own disclosed open item) — this wave's
direct measurement shows per-cell mean anchor-row norms of 1.06–1.16, not
1.0, so the back-solve's implicit unit-norm assumption was measurably
wrong, in the direction that would make the back-solved `r` an
overestimate relative to the true norm-invariant `r_e` (a larger blend
term relative to the raw-key term, at a fixed observed `a_e`, is
consistent with a smaller true `r` than the unit-norm inversion implies);
and (ii) inverting a *mean* `a_e` through a nonlinear map does not recover
the *mean* of the per-resample `r` values (§9.7.10 item 5's own disclosed
Jensen's-gap caveat) — the back-solved "effective" `r_e` is a biased
summary of the true per-resample distribution, not an unbiased estimate
of it, and the direction of that bias was never bounded, only disclosed
as a caveat. Both mechanisms push the same way: the back-solved
`r ≈ 0.33–0.35` was always a construction built to survive exactly the
objections that killed Rev 6, and its failure to predict this wave's
directly-measured, assumption-free `r_e` is the clearest evidence yet
that those two objections were not merely theoretical — they were
quantitatively load-bearing. **The fresh direct measurement supersedes
the back-solved prior expectation; this is disclosed as a failed
prediction, not explained away.**

#### 10.13.4 Synthesis — measured-and-rejected mechanism, decisively positive behavior and stability

**The entity-alignment mechanism framing, as posed by §1 and tested by
§10.3's engagement criterion, is measured-and-rejected at K=32 — literal
Outcome C, both arms, 3/3 seeds each, on the wave's own pre-registered,
attack-hardened, hash-locked test.** This is not a repeat of the confirm
wave's C: it is a **strictly stronger** null, immune to every specific
objection (`‖anchor‖=1`, λ-degeneracy, menu-laundering, power-manufactured
engagement via raw statistical precision) that forced Rev 6's rejection,
and it is now additionally tested under genuine per-entity capacity
(candidate d′) — which converges to the **same** band as the global-scalar
arm, closing the "parameter-sharing artifact" explanation in the
negative, with real per-entity evidence rather than an inference.

**Behavior and stability, independently, are decisively positive and now
at independent-replication strength.** K=32 h4 `rec@0.9`: candidate (d)
fresh seeds {10,11,12} = 0.6741/0.7125/0.6141; candidate (d′) fresh seeds
{20,21,22} = 0.7021/0.6661/0.7141 — **6 of 6 fresh seeds clear the ≥0.5
bar**, at a materially different seed set (not seed-integer reuse, unlike
the confirm wave's own explicitly-disclosed same-seed limitation, §9.6)
and a **third independent architecture variant** (per-entity λ, not a
re-run of the same global-scalar mechanism). Combined with K=16 s10 =
0.9998 (no-regression, clean) and the pre-existing 3 waves' own seed sets
(Wave-1's 0/1/2, the confirm wave's same-seed reproduction, and now
10/11/12 and 20/21/22), the behavioral h4 gain has been observed **9
times total across 3 distinct seed sets and 3 waves** — this wave's
specific contribution is the **first genuinely fresh 6-seed sample**,
upgrading the behavioral claim from "reproduced at the same seed
integers" (confirm wave) to **independent replication** in the ordinary
statistical sense (new, never-before-drawn seeds, same result). Item 5
(pre-NS drift) clears its 0.95 bar by 0.05–0.05 margin in all 7 cells,
and item 6 (both legs) clears 3/3 at K=32 for arm (d) this wave (an
improvement on the confirm wave's 2/3).

**The parsimonious mechanistic account, given both results together: THE
ANCHOR BLEND STABILIZES BY CONSTRUCTION, NOT BY ENTITY ALIGNMENT.** The
blend `sub_blend = normalize((1−λ)·k_raw + λ·anchor)` mixes an
**episode-constant** term (`anchor_table.weight[e]`, fixed across every
resample of entity `e` within a run) with an **episode-varying** term
(`k_eff_raw[e]`, the actual source of cross-episode drift this design
exists to fix). At an SGD-preferred interior λ ≈0.55–0.60 (measured, this
wave, all 7 cells — consistent with every prior wave's own λ range), the
blended key's cross-episode variance is mechanically damped by a factor
related to `(1−λ)²` regardless of whether `k_raw` ever moves toward
`anchor` in direction — **stability arrives from the blend's arithmetic
structure, not from the raw key learning to align with a fixed target.**
This reframes the 2×2 (behavior × mechanism) that has been open since
§9.5: the "missing ingredient" for cross-episode stability was never
entity-level alignment (`r_e` large) — it was the mere **presence** of an
episode-constant additive term in the blend, at a large-enough λ, which
the raw-key term cannot disrupt no matter how weakly it aligns. This also
explains candidate (d′)'s null result without needing any auxiliary
story: if alignment isn't the channel, giving each entity its own λ_e
changes *how much* stability each entity's blend gets, not *whether*
alignment is the source of that stability — so per-entity freedom has
nothing to differentially recruit, and SGD's mild, partial,
non-band-changing use of that freedom (the 2/3 Spearman signal) is
exactly the residual noise this account predicts: some entities'
raw keys are cheaper to leave alone (already closer to chance-orthogonal
to their anchor, hence a slightly higher λ_e is "free" stability with
little raw-key value lost) while others' raw keys carry more real
signal worth preserving (slightly lower λ_e) — a second-order
optimization over the SAME non-alignment-based stability channel, not
evidence of an alignment-based one.

**What this account predicts, and what would falsify it (the decisive
next probe):** if stability is purely a property of blending in *any*
episode-constant term at the right λ, then **a random, FROZEN anchor
table** (never trained, `anchor_lambda_mode` still learned or fixed at a
matched λ) should deliver **similar h4/pre-NS-drift gains** to candidate
(d)'s own learned table — because the mechanism this account proposes
does not require the anchor table to carry any entity-specific
information at all, only to be *present and constant across resamples*.
If a frozen-random table matches (d)'s gains at matched λ, the
construction-stabilization account is confirmed and the "learned" part of
candidate (d)'s anchor table is doing no additional mechanistic work
beyond providing *a* fixed point. If a frozen-random table performs
**worse** than candidate (d) by a wide margin, the account is
**falsified** — some property of the *learned* table beyond mere
constancy (plausibly bulk pool geometry, not per-entity alignment) would
need a different explanation.

**Registered next-probe stub — candidate (e): frozen-random-table
ablation.** K=32, `anchor_lambda_mode` **fixed** at λ ≈0.58 (the
cross-cell mean of this wave's own measured λ, so the comparison is
matched on the one hyperparameter this account claims matters), anchor
table **initialized once via the existing `frame_potential_init` and then
never trained** (`anchor_table.weight.requires_grad_(False)` or
equivalent — a minimal, one-line addition to the existing masked
gather/scatter/held-out-bypass path, not a new mechanism), 2–3 seeds.
Success criterion for the construction-stabilization account: h4 and
pre-NS drift within ordinary seed noise of candidate (d)'s own K=32
range (h4 0.61–0.71, pre-NS drift ≈1.00). Estimated cost: **~1 GPU-h**
(3 cells at the realized ≈0.2–0.35 GPU-h/cell range this design has
repeatedly observed, §10.7/§11's own budget precedent) — cheap relative
to the diagnostic value, and it does not require attacking or revising
any machinery already hash-locked in this section (no new metric, no new
threshold, no new smoke suite beyond a forward/backward check on the
frozen-parameter path). **Has the archived data already spoken to
this?** No — no wave in this design's history has ever run
`anchor_lambda_mode` fixed with the anchor table simultaneously frozen at
init (the closest existing cell, `wavekeyanchor-neg1`'s
`armkeyanchor-d-fixed`, fixes λ but still trains the anchor table); this
probe would be the first direct test and is not retroactively answerable
from any existing JSON.

**Registration amendment (2026-07 K48+e build, pre-launch/pre-data, per
the build's final-audit prescription):** this stub as written carries an
internal tension — its NAME ("frozen-**random**-table") and §10.13.4's
motivating text describe an unoptimized random table, while its init
sentence above literally says `frame_potential_init` — so the wave is
registered with **BOTH arms** (6 cells total, one `--wave keyanchor-e`
dispatch): arm **'e'** (seeds {60,61,62}, frozen **random unit rows** —
`frame_potential_init`'s own pre-optimization starting point) answers
whether mere episode-constancy of *any* fixed table delivers candidate
(d)'s gains (the construction-stabilization account's strongest form),
and arm **'e-fp'** (seeds {70,71,72}, frozen **`frame_potential_init`**
table, this stub's literal init text) answers whether the table's
*optimized bulk geometry* matters beyond constancy even with zero
training — jointly: e-fp≈(d) with e-random collapsing ⇒ bulk geometry is
the carrier; both≈(d) ⇒ constancy alone suffices; both collapsing toward
geo3-alone ⇒ the *learned* table matters beyond both construction
properties. Same fixed λ=0.58, same instrumentation, cost 6 × 0.20–0.35 ≈
1.2–2.1 GPU-h (registered ceiling 2.5).

#### 10.13.5 Claim tier — §10.6/§3.5 applied literally

**The §1 key-anchoring entity-alignment mechanism hypothesis: Outcome C,
confirmed at the mechanism-tier bar this wave was built to provide** — a
genuinely stronger, assumption-free, power-floor-protected negative than
the confirm wave's own C, and immune to every specific objection that
forced Rev 6's rejection. Candidate (d′)'s independent per-entity-capacity
question routes to **Inconclusive/mixed** per §10.6's own table (same
band as (d), partial-not-unanimous Spearman significance) — not a clean
C′, but also not A(d′); reported as such, no forced call.

**The h4/λ/stability behavioral result: upgraded to independent-replication
strength.** Per §10.6's own tier discipline (this section routes claim
tiers, not just Outcome letters) and §3.5's admission stack: 6 fresh
seeds (2 architecture variants) + 1 K=16 spot check all clear their own
pre-registered bars; items 1–6 and λ-interior are clean at every K=32
(d) cell this wave (3/3, an improvement on the confirm wave's 2/3); the
behavioral claim now rests on **9 total seed-runs across 3 independently-drawn
seed sets and 3 waves**, of which this wave supplies the first 6 that are
genuinely fresh draws, not integer reuse. This licenses describing the
h4 gain as **independently replicated**, not merely reproduced — a
meaningfully stronger evidentiary statement than either prior wave could
make, while the *mechanistic attribution* for that gain remains the
construction-stabilization account above: **descriptive/interpretive**,
consistent with real evidence (the account correctly predicts d′'s null
result and is falsifiable at ~1 GPU-h) but not itself confirmed —
confirmation is exactly what candidate (e) is registered to test.

**No spin, stated plainly:** the registered outcome for the
mechanism-as-framed question (§1's entity-alignment hypothesis, tested by
§10.3's engagement criterion) is **C**. It is not A, not A″, not
Inconclusive. Behavioral and stability gains are real and now
independently replicated; the *reason* those gains occur is not entity
alignment, and this wave's own hash-locked, pre-registered machinery is
what makes that a measured conclusion rather than an inference.

---

## 11. Rev K48.1 — Capacity-curve extension (K/d ∈ {0.25, 0.5, 0.75} at
d_state=64) [Rev K48.1 — the attack-response revision; DRAFT, pending a
fresh verify round before CLEARED-FOR-BUILD; zero GPU spent producing
or revising this section; design + CPU code-read/derivation work only,
identical discipline to §10's own Rev-7.1 header]

**Status.** Rev K48 (commit `daa79dc`) received its own independent
4-agent attack round this session (math-reproducer, code-truth-auditor,
budget-schedule-auditor, circularity-bias-hunter — two of the four
independently reproduced every computed number from a fresh CPU-only
venv before trusting it). Verdict: **1 FATAL-leaning + 13 MAJOR + 3
MINOR, zero contested** — every finding accepted and addressed in
place; the full finding→change map is **§11.10**. The load-bearing
fixes: the bar derivation's multiplicative-vs-additive choice and its
"mean, always" statistic are now justified explicitly, closing the
FATAL-leaning "no free choice" gap (§11.2); the K=32 source data used
to derive the K=48 bar is relabeled from a wave that has never run
("Rev 7.1 mech-wave") to its real source (Wave 1, §9.4); §11.4.1's
i-strong-infeasibility mechanism is corrected (a hardcoded `args.K<=32`
CLI guard, not a `48+48>d_state` dimensional impossibility) and
§11.4.3's arm is replaced accordingly (a fixed-λ=1 candidate-(d) probe,
not an asymmetric i-strong pin, which would not have validated the
same mechanism anyway); §11.0's manifest-hardcoding line citations are
corrected; a falsification-map row, a hub-detection checklist item, a
BH/BY K-dependence disclosure, and several budget-arithmetic
corrections and mechanical-gate honesty downgrades round out the rest
(§11.10). This section turns the single K=32 point result (§9/§9.6,
Outcome C at behavioral-descriptive tier; §10's Rev 7.1 mechanism-tier
re-test still DRAFT, pending its own bounded verify round before any
GPU is spent) into a three-point **recovery-vs-memory-load curve**:
K/d_state = 0.25 (K=16, already run), 0.50 (K=32, already run), 0.75
(K=48, this section). User has signed off on the capacity-curve
program; this design still goes through the full attack gauntlet
before build/launch, per this project's own standing multi-round
adversarial-audit rule. **This wave inherits Rev 7.1's instrumentation
wholesale, not as a citation but as literally the same code path**:
direct `r_e` measurement upstream of the blend (§10.2), anchor-row-norm
logging every checkpoint (§10.2.1), the exact-Beta/BH/BY engagement
test with its effect-size floors (§10.3, §10.3.4), the hash-locked
`REV7_THRESHOLD_PINNED.json` writer/gate/readout triple (§10.3.3), and
the mandatory checkpoint-saving writer/gate/readout triple (§10.10).
None of this is re-derived below except where K=48 genuinely changes a
number (the bar, the NS/conditioning legs, the budget bracket) — where
it does not change (the entity pool, the engagement-test constants),
that is verified explicitly, not assumed.

### 11.0 Relationship to Rev 7.1 — what depends on it, what does not

This wave's **mandatory** cells (reference arms, the Gate-1-style
probe, candidate (d)) do **not** depend on Rev 7.1 reaching a verdict —
they inherit its instrumentation as **code**, which already exists and
is committed (`key_anchoring.py`, `rev7_threshold_derive.py`,
`REV7_THRESHOLD_PINNED.json`), independent of whether Rev 7.1's own
K=32 wave has been run, attacked, or verdict-assigned yet. Only
candidate (d′) (§11.1) is conditioned on Rev 7.1's own outcome —
currently an orchestrator sign-off gated on Rev 7.1's own written
verdict, not yet a mechanical gate (honestly downgraded at Rev K48.1
per attack finding budget-schedule-auditor M3; §11.1 item 3, §11.10).

**A design virtue worth stating plainly, not just claiming:** the K=32
program needed four revisions (Wave 1 → confirm wave → Rev 6 → Rev 7.1)
to arrive at mechanism-tier instrumentation, because Wave 1 launched
without it (§9.3's documented gap — `keyanchor_wave1_manifest()` never
threaded `drift_probe=True`, and no per-entity/`r_e` field existed at
all until the mech-wave build). **K=48's candidate (d) launches with
the full Rev-7.1-era instrumentation from its very first cell** — it
cannot repeat §9.3's specific mistake, because the code it calls
already has `r_e`, anchor-norm logging, and the BH engagement test
wired into `run_deltanet_rd.py`'s training loop, not bolted on after
the fact. **Citations corrected at Rev K48.1 (attack finding
code-truth-auditor M2 — 3/5 line numbers were wrong in the original
draft; §11.10):** `run_deltanet_rd_exactness_sweep.py`'s
`reference_arms_manifest` (def L307, `for K in (16, 32)` at L318) and
`keyanchor_wave1_manifest` (def L361, `for K in (16, 32)` at L392 AND
a second such loop at L398, both omitted from the original citation)
each genuinely hardcode the K=16/32 loop; `keyanchor_confirm_manifest`
(def L408) does **not** share that shape — it hardcodes K=32 via `for
s in range(3)` (L463) plus one conditional K=16 append
(`include_k16_spot_check`, L468) — a structurally different pattern,
not a `for K in (16,32)` loop, so the blanket "all hardcode `for K in
(16,32)`" claim is corrected to name only the two functions that
actually have that shape, with the third described accurately. (Lines
217/339/343 — mis-cited in the original draft — belong to
`geo3_wave1_manifest`, def L208, and `keyanchor_wave_neg1_manifest`,
def L327, respectively; NEITHER is one of the three functions this
paragraph is about, and `geo3_wave1_manifest` is the SAME sentence's
own named K48-aware exception, so citing its line as evidence for the
other functions was self-contradictory.) The higher-level conclusion —
none of `reference_arms_manifest`/`keyanchor_wave1_manifest`/
`keyanchor_confirm_manifest` has a K=48 branch in any form, and
`geo3_wave1_manifest(include_k48=True)` is the only existing K=48-aware
manifest function in the file — is independently re-verified (full
grep of "48" across the file) and stands unchanged. **Build scope,
registered now:** this wave requires a new `keyanchor_k48_manifest()`
(mirroring `geo3_wave1_manifest`'s own `include_k48` opt-in pattern,
generalizing the two `for K in (16,32)` functions to accept an explicit
`K` argument, and adding an analogous `K` parameter to
`keyanchor_confirm_manifest`'s own shape) — a build task, not a design
gap; flagged here so the attack round can check the generalization
doesn't silently change K=16/32 behavior (a registered Wave −1 smoke,
§11.9 item 14, asserts byte-identical manifests for K=16/32 before and
after the refactor).

### 11.1 Arms

**Registered manifest name:** `keyanchor-k48` (mirrors §10.5's Rev-7.1
naming-pinning discipline — attack-R7 finding 11). **Seed convention:**
continues the escalating-decade-block pattern this program has used at
every new wave (Wave 1 candidate d/c: 0–2; reference arms: 1–3;
Rev-7.1 candidate-d re-run: 10–12; Rev-7.1 candidate-d′: 20–22) — new,
never-before-used-anywhere blocks are assigned per arm below so no
seed integer's provenance is ever ambiguous across K, even though K
alone already guarantees no filename collision (`is_done()`/`out_path()`
key on K).

1. **Candidate (d), learned-λ, full Rev-7.1 mechanism-tier
   instrumentation, K=48 — MANDATORY, PRIMARY.** Seeds **{30, 31, 32}**,
   3 runs, 20,000 steps each. Architecture: byte-identical to Rev 7.1's
   candidate (d) (`anchor_blend_gather_scatter`, single learned scalar
   λ, frame-potential anchor table init at `ANCHOR_INIT_SEED=20260705`
   — same table, same seed, verified below to be K-independent) **plus**
   `--drift-probe` (item 5's pre-NS blend drift) **and** the Rev-7.1
   `r_e`/anchor-row-norm/BH-engagement logging wired directly into the
   training loop from run 1 (§11.0). No K=16 spot check is registered
   for this arm — K=16 already saturates near h4≈1.0 under bare geo3
   (§9.4's own reference: 0.9716), leaving no headroom to see an
   anchoring effect there, the same reasoning Rev 7.1 gave for skipping
   K=16 on candidate (d′) (§10.5).

2. **Reference arms — bare geo3, K=48, full per-checkpoint drift
   diagnostic — MANDATORY, first in manifest (bands don't exist at
   K=48, so this K needs its own writer/gate/readout triple).** Seeds
   **{1, 2, 3}** (reusing the §3.6 reference-arm seed convention at a
   genuinely new K — not a repeat of any prior (K, seed) cell), 3 runs,
   20,000 steps each, `drift_probe=True` from step 1 (mirrors §3.6's
   own reference-arm spec exactly: post-NS **and** pre-NS pooled
   pairwise-cosine drift, `measure_drift`'s existing machinery,
   unmodified). **Full writer/gate/readout triple, specified below
   (§11.1.1)** since — unlike Rev 7.1's mech wave, which reused K=16/32's
   existing `BANDS_PINNED.json` — no post-NS engagement band exists at
   K=48 in any form.

3. **Candidate (d′), per-entity λ_e, K=48 — CONDITIONAL, cut-eligible.**
   Seeds **{40, 41, 42}**, 3 runs, 20,000 steps each, IF AND ONLY IF
   Rev 7.1's own K=32 wave resolves to **Outcome A(d′)** (real
   differential engagement demonstrated) **or Outcome D′** (per-entity
   capacity used, and used badly — §10.6's routing table) — i.e., IF
   Rev 7.1's own data shows a per-entity λ_e table does *something*
   structurally different from a global scalar at K=32, in either
   direction. **Does NOT fire on Outcome C′** (near-uniform λ_e, no
   significant differentiation — the structural null): if K=32's own
   per-entity mechanism shows no capacity effect, there is no
   registered reason to expect K=48 — a strictly harder packing regime
   (§11.4) — to behave differently, and the cell is cut, not run
   speculatively. **Honestly downgraded at Rev K48.1 (attack finding
   budget-schedule-auditor M3; §11.10):** the original draft called
   this a "mechanical launch precondition... the launcher checks Rev
   7.1's own result JSONs / readout verdict field" — but no script,
   named JSON field, hash-lock, or failure mode was ever specified, and
   per §10.6's own text, Outcome A(d′)/C′/D′/Inconclusive is a
   multi-field routing table (engagement-band comparison AND dip-
   test/Spearman significance, aggregated over ≥2/3 seeds) that every
   actual Outcome assignment in this document (§9.5, §9.6) has so far
   been assigned by **narrative analyst read-off**, never written as a
   single machine-checkable field in any run's JSON. Until a concrete
   `k48_dprime_gate.py`-style script exists that reads Rev 7.1's raw
   per-seed fields and applies §10.6's routing table programmatically
   (mirroring `validate_bands_pinned`'s own pattern, with its own smoke
   test proving both branches reachable) — a real, registered build
   item, not yet built — this is accurately described as an
   **orchestrator sign-off gated on Rev 7.1's own written verdict**,
   not a mechanical gate: the orchestrator reads Rev 7.1's completed
   analysis (once it exists) and decides whether A(d′)/D′ was reached
   before authorizing this cell's manifest entry. Architecture: byte-identical to
   Rev 7.1's own (d′) (`anchor_lambda_table`, one-line indexing change
   over the scalar case, §10.5.1) at K=48.

4. **Gate-1-style pre-launch probe — candidate (d) architecture only,
   K=48, 5,000 steps, 1 run, seed 0 — DISCLOSED, NON-GATING.** Mirrors
   Rev 7.1's own d′ probe (§10.5) and inherits its own scope carve-out:
   the simulator drift→recovery mapping (`geo3_simulator.py::launch_
   read`) is **already established as non-gating at K=32** (§4's own
   "conservative at K=32... K=32 reads stay non-gating" ruling,
   underestimating measured recovery 5–7×) — K=48 is a *harder* packing
   regime than K=32 (§11.4), so the same non-gating status is inherited,
   not re-derived: this probe is run and reported (an early wall-clock/
   loss-sanity check, cheap at 5,000 steps) but **never** a hard go/
   no-go. **Gate 2 (the construction check, §11.3) remains the sole
   go/no-go for this wave**, exactly as it already is at K=32.

5. **Fixed-λ=1 candidate-(d) ceiling-validation probe — OPTIONAL,
   LOWEST PRIORITY, cut-eligible first if budget is tight (REPLACED at
   Rev K48.1 — the original "asymmetric-pool pin arm" is withdrawn,
   §11.4.3).** Seed **50**, 1 run, 20,000 steps, candidate (d)'s own
   architecture with `anchor_lambda_mode="fixed"`, `anchor_lambda_
   fixed=1.0` (extending §2.2's existing fixed-grid diagnostic with a
   ceiling-validation point). See §11.4.3 for why this exists, what it
   costs, and why it is registered as optional rather than mandatory
   (the free, already-computed λ=1 anchor-blend ceiling, §11.4.2,
   already answers the "hard question" the task requires — this arm
   only adds a second, independently-*trained* confirmation of the same
   mechanism at its λ→1 extreme).

**Seed contingency (inherited unchanged, §6/§14.10's add-seeds-not-
steps discipline):** if candidate (d) or (d′)'s λ-band assignment lands
ambiguous at 2/3 (not 3/3) seeds, +2 seeds, one iteration — reserved
block **{33, 34}** for candidate (d), **{43, 44}** for candidate (d′).

#### 11.1.1 K=48 reference-band writer/gate/readout triple (new — bands don't exist at this K)

Mirrors §3.6 (Rev 4) exactly, scoped to the 3 K=48 reference arms only
(not a re-derivation of K=16/32's own `BANDS_PINNED.json`, which is
untouched):

1. **The writer.** The harness writes `BANDS_PINNED_K48.json` —
   containing the derived `engaged_48` threshold (or an UNRESOLVABLE
   verdict), the 3 per-seed final-checkpoint post-NS drift inputs, the
   formula version string (`"sec 3.6 Rev4 (n=3, engaged_K = mean_ref +
   2*s_ref, df=2)"` — the **same, unmodified** `derive_engaged_bands`/
   `write_bands_pinned` functions in `key_anchoring.py`, called with
   `per_k_final_drift={48: [...]}` and `ceiling_by_k={48: 0.8987}` —
   §11.4.2's computed ceiling), sha256 hashes of all 3 reference-arm
   result JSONs, and a timestamp — **only after** all 3 K=48 reference
   arms validate complete (`reference_arm_result_valid`, unmodified,
   §3.6's own validator).
2. **The launcher gate.** Candidate (d) and (d′) cells at K=48 **refuse
   to launch** unless `BANDS_PINNED_K48.json` exists and re-hash-
   validates (`validate_bands_pinned`, unmodified) — the same loud-
   refusal pattern, the same `--unblind-override` demotion-stamping
   fallback (§3.6 Rev 5's fields written into every affected K=48
   anchor-arm's own result JSON at assembly time, verbatim, not
   reinvented).
3. **The readout assertion.** `assert_blind_not_broken` (unmodified)
   checks `BANDS_PINNED_K48.json`'s timestamp strictly precedes the
   earliest K=48 anchor-arm start time.

**Degenerate-case guard, pre-stated (not presumed):** `engaged_48 =
mean_ref + 2·s_ref` (n=3, df=2) is declared UNRESOLVABLE at K=48 iff it
lands `≥ ceiling_48 − 0.005 = 0.8937` (§11.4.2). A rough, low-fidelity
**prior expectation** exists for context only (`DELTANET_RD_EXACTNESS_
DESIGN.md` line 2253–2254: a single-entity, 200-resample attack-round
probe measured K=48 drift 0.80–0.82 — coarser than the 8-entity/32-
resample protocol these reference arms use, and explicitly super-seded
by them the same way the old 0.9037/0.9416 single-seed K=32/16 numbers
were superseded once real reference arms ran, §3.6). If the fresh
K=48 reference arms land anywhere near that coarse prior (≈0.80–0.85),
the gap to `ceiling_48` (0.8987) is comparable in absolute size to
K=32's own resolved gap (0.9423−0.9037=0.0386 vs. K=48's plausible
≈0.05–0.09) — likely resolvable, but this is a **disclosed
expectation, not a presumed result**; the guard is evaluated
mechanically on the real 3-seed data, exactly as §3.6 already requires.
Post-NS engagement bands remain **non-gating sanity context only**
(§3.5 B1/B2 disambiguation), never load-bearing for this wave's own h4
bar (§11.2) or for the Rev-7.1-inherited `r_e`/BH engagement test
(§11.0), which needs no reference-arm data at all (§10.3.3's own
zero-data-dependency property, unaffected by K).

**REV7_THRESHOLD_PINNED.json is K-independent — verified, not assumed
(the task's own explicit check).** `rev7_threshold_derive.py`'s
`derive()` (read this session, `matrix-thinking/deltanet_rd/rev7_
threshold_derive.py` lines 48–50) takes exactly three free inputs:
`N_ENTITIES = 107`, `D_STATE = 64`, `ALPHA = 0.05` — **`K` (the
per-episode entity draw count) does not appear anywhere in the file**.
`N_ENTITIES=107` is the **train-pool vocabulary size**, not a
per-episode draw count — confirmed independently this session by
`grammar_rd.py::build_entity_pools()` (lines 194–253): the pool is
built once from 213 verified names via a **fixed `heldout_frac=0.5`
split** (`n_train_names=107`, `n_heldout_names=106`), with **no
reference to `K` anywhere in the function** — `K` is consumed only
downstream, as an assertion that the pool is *large enough* to draw K
names from (`run_deltanet_rd.py`'s own `pool_report["n_train_names"] <
args.K` guard). Empirically cross-checked against two archived result
JSONs at different K (`wgeo3_rdx_K48_armgeo3_s0_geo3n20.json` and
`wgeo3_rdx_K32_armgeo3_s0_geo3n20.json`): **byte-identical
`pool_report`** in both (`n_train_names: 107, n_heldout_names: 106`
verbatim). **Consequence: `REV7_THRESHOLD_PINNED.json` — the exact-Beta
Bonferroni critical value (r≥0.4009), the BH/BY constants, the
`σ_chance=0.125` null scale, and the A/A″ effect-size floors
(0.35/0.25)** — applies UNCHANGED at K=48. No new pin, no new hash-lock,
no new derivation. The **launcher gate** (§10.3.3, item 2) is reused
verbatim for K=48 candidate (d)/(d′) cells: same script, same recorded
hash, same `derive()` re-run check. This closes the task's explicit
ask cleanly: the entity pool really is K-independent, and the
consequence for the pin is not merely "probably fine" but a verified
zero-change fact. **Scope note added at Rev K48.1 (attack finding
circularity-bias-hunter Finding 5; §11.10, §11.8):** K-independence of
the pin's own *constants* does not by itself guarantee K-independence
of the *dependence structure* the BH-vs-BY primary/cross-check pair is
watching for — K=48 draws a much larger fraction of the 107-entity pool
per episode than K=32 does, which could plausibly widen the BH-vs-BY
discovery gap; see §11.8's own disclosure for the registered read on
this.

### 11.2 Bars — derivation (no free choice left at readout)

**h4 bar at K=48 — the relative-gain derivation, restated exactly, not
a fresh judgment call.** The K=16 (≥0.8) and K=32 (≥0.5) bars in
`DELTANET_RD_EXACTNESS_DESIGN.md` §5.5 were pre-registered constants,
not the output of a stated formula — chosen (per that document's own
Welch-bound-defended language, line 1487) to sit "meaningfully above
baseline, achievable in principle." **This wave cannot inherit that
judgment-call method directly (there is no equivalent prior "what
looks achievable" intuition for K=48, and inventing one now would be
exactly the free choice the task rules out) — so it instead reuses the
one number this program's own K=32 wave already published and can
transplant mechanically: the realized relative gain candidate (d)
delivered over its own fresh bare-geo3 reference, at the one K where
the mechanism has ever shown daylight.**

From already-published, already-fixed numbers — **provenance corrected
at Rev K48.1 (attack finding math-reproducer M1 / circularity-bias-
hunter Finding 2 — the original draft mislabeled this data "Rev 7.1
mech-wave"; §11.10):** these are **Wave 1's** own numbers (§9.4's
table — the wave that "ran to completion," 18 mandatory cells, 10.98
realized GPU-h), **not** any Rev-7.1-labeled data — Rev 7.1 (§10) is
still DRAFT and has never launched a single GPU cell, so it cannot be
the source of a real number. (The confirm-wave's own re-measurement of
the same candidate-d K=32 cells, §9.6, gives a close but numerically
different mean, 0.61235 — disclosed here so a reader chasing either
label finds the right file: this derivation uses Wave 1's numbers
specifically, traced to `experiment-runs/2026-07-05_keyanchor_wave/
waveref/wref_rdx_K32_armgeoref_s{1,2,3}_geo3n20_dprobe.json` and
`.../wavekeyanchor/wkeyanchor_rdx_K32_armd_s{0,1,2}_geo3n20_anchor_
learned.json`):
- K=32 fresh reference (bare geo3, 3 seeds, **Wave 1**, §9.4):
  h4 `rec@0.9` mean = **0.4105**.
- K=32 candidate (d) (**same wave**, §9.4): h4 `rec@0.9` mean =
  **0.6132**.
- Relative-gain factor: `g = 0.6132 / 0.4105 = 1.4938...` (this **is**
  the design's own already-published "+49% relative lift," §9.4 —
  re-derived here to full precision, not re-typed from the rounded
  prose).

**Why MULTIPLICATIVE, not additive, and why the MEAN — the two degrees
of freedom the attack round showed were left implicit (circularity-
bias-hunter Finding 1, FATAL-leaning; §11.10) — pinned explicitly, with
reasons, not asserted away:**
1. **Functional form: multiplicative.** An additive transplant
   (`bar = baseline_K48 + (h4_cand,K32 − h4_ref,K32) = 0.016357 +
   0.2027 ≈ 0.219`) is algebraically equally "parameter-free" by the
   same transplant logic, but it is not well-defined on this metric:
   `rec@0.9` is bounded in `[0,1]`, K=48's baseline (0.0164) sits
   ~25× below K=32's own baseline (0.4105), and an additive gain
   calibrated at the *larger* baseline, applied at a baseline this
   close to the metric's own floor, asks the K=48 point to move by an
   absolute amount that is **more than 13× its own starting value** —
   a demand with no interpretable analog at K=32's own operating point
   (where the *same* additive gain was under half of baseline, not 13×
   it). A relative/multiplicative reading is the only one of the two
   that stays comparably-scaled across a 25× baseline gap; this is the
   registered reason, not merely a label attached after the fact.
2. **Statistic: the mean, always.** This design's own established
   convention — used for every guard and bar in this document without
   exception (the K=16/K=32 h=1 guards, §3/§16.4/§16.8; the K=16
   no-regression check, §3; the early-stop thresholds, §3.4) — reads
   the **mean over 3 seeds**, never a single seed's lower or upper
   bound. `g` above uses Wave 1's own published means on both sides of
   the ratio (0.6132/0.4105), inheriting this pre-existing convention
   rather than introducing a new one; using any other statistic (a
   conservative seed, a pooled quantile) would be the actual free
   choice, and this derivation does not make it.

Applied to K=48's own baseline — the **only** available K=48 reference
point, the archived geo3-alone K=48 arm (3 seeds, `wgeo3_rdx_K48_
armgeo3_s{0,1,2}_geo3n20.json`, held-out h4 = 0.011963/0.019165/
0.017944, mean **0.016357**; disclosed honestly: this baseline is
itself **non-admissible** — `value_salvage_tier_pass: false` at all 3
seeds, item 1 of the substitute admission stack, §16.9 — retained here
only as the best-available anchor point for the bar's *derivation*,
exactly the same epistemic status §10.3.4's own `R_MIN_HEADLINE`
literal already accepts for a prior wave's published summary):

```
bar_K48 = baseline_K48 x g = 0.016357 x 1.49378 = 0.024434
```

(**Arithmetic corrected at Rev K48.1 — attack finding math-reproducer
m1: the original draft's "= 0.024440" was a multiplication slip; full
precision, `0.016357421875 x 1.493788063... = 0.024434`, §11.10.**)

**Registered bar: h4 `rec@0.9` ≥ 0.0244 at K=48** (the computed value
to 4 significant figures; no grid-rounding applied beyond that, since
no established rounding-grid convention exists at this metric's scale).
**Achievability sanity-check, stated plainly, not hidden:** this bar
embeds the *identical* relative-improvement magnitude candidate (d)
already demonstrated once, at the one K/d ratio (0.5) where this
mechanism has ever shown any daylight — it is exactly as hard, in
relative terms, as the bar the K=32 wave already cleared, and no harder
or easier by construction. Its small **absolute** size is an honest,
direct consequence of K=48's own near-total baseline collapse (0.0164),
not an artificially softened target.

**Guard (inherited, unchanged convention): h=1, within −0.02 of geo3-
alone K=48's own h=1 baseline.** geo3-alone K=48 h1 = 1.0000 (mean, all
3 seeds, `M2_in_distribution`) → **guard: h=1 ≥ 0.98.**

**K=16/K=32-style no-regression cross-checks are NOT required at K=48**
(each K is a separately-trained model — the task's own scoping, and
consistent with this program's existing practice: K=16's own bar was
never checked against K=32's result either). **Curve-reporting rule,
registered now:** the eventual paper/write-up figure reports **all
three K points (16, 32, 48)** on one plot (x-axis K/d_state ∈
{0.25, 0.50, 0.75}, y-axis h4 `rec@0.9`), each point **individually
labeled with its own bar and its own claim tier** (K=16: minimum-
publishable, HIT, admissible/confirmed; K=32: headline-demo,
descriptive-behavioral per Outcome C — or Outcome A/A″ if Rev 7.1
clears differently before this wave's own write-up; K=48: this bar,
tier TBD by this wave's own result) — **never pooled, averaged, or
presented as a single aggregate statistic across K.** If K=48's own
geo3-alone baseline (item 2 below) also fails the value-salvage floor
on fresh reference-arm seeds (a real, disclosed possibility, §11.6's
falsification map), the curve's K=48 point is reported with that
caveat attached, not silently smoothed over.

### 11.3 NS settings at K=48

**n_iter=20** (the production tier already established for K=48 by
`run_deltanet_rd_exactness_sweep.py::geo3_wave1_manifest`'s own
`--include-k48` comment: "NS at n_iter=12 lands at resid 0.104 > tol on
realistic near-collinear probes; 20 converges to ~1e-6"), unchanged
from the archived geo3-alone K=48 riders. **Zero-fallback requirement
stands, doubly confirmed this session:**

- **Real trained-run evidence (already archived, re-read this
  session):** all 3 archived geo3-alone K=48 runs (`geo3n20`, 20,000
  steps each) report `ns_converged_no_fallback: true`,
  `n_geo3_fallback_train_steps: 0`, `checkpoint_fallback_seen: false` —
  **0 fallback steps across 60,000 real training steps** (3 seeds ×
  20,000), at production tier, on real (not synthetic) episode-
  conditional keys. This is the single most reassuring fact this
  section can report: the K=48 rider's own non-admissibility is
  entirely a **value-salvage-ratio** failure (item 1, §11.2's own
  disclosure), never an NS-convergence failure (item 2) — the two are
  independent legs of the same admission stack, and only one is broken
  at K=48.
- **Fresh CPU construction-gate check, this session, on the SAME
  registered frame-potential anchor init the anchor-blend mechanism
  will actually use** (`key_anchoring.py::frame_potential_init(107, 64,
  seed=ANCHOR_INIT_SEED=20260705)` — the identical table, identical
  seed, already gating K=16/32; **not** a new table, so this check is
  additive to Gate 2, never a parallel or divergent construction):
  the amended three-leg Gate 2 (§4) is run at K=48 exactly as it
  already runs at K=16/32 (`gate2_construction_check`, extending
  `GATE2_N_ITER_BY_K` from `{16: 12, 32: 20}` to `{16: 12, 32: 20, 48:
  20}` — a one-line dict extension, not new gate logic):

  | Leg | K=48 result | Bar | Verdict |
  |---|---|---|---|
  | G2-a (`σ_64/σ_1`, raw table) | **0.999999642** | ≥0.1 | **PASS** (same table as K=16/32 — this leg is K-independent by construction, verified identical to 4 decimal places against the design's own cited 1.0000) |
  | G2-b (`max\|cos\|`, raw table) | **0.284151** | ≤0.5 | **PASS** (K-independent, same table, matches the design's own cited 0.2842) |
  | G2-c (NS admission, 512 random 48-subsets, `n_iter=20`) | **0/512 fallbacks, max resid 1.02e-6** | 0 fallbacks | **PASS**, with wide margin (K=16: max resid 5.7e-7 on this run; K=32: 6.9e-7 — K=48's own margin is the same order of magnitude, not degraded) |
  | G2-c at `n_iter=12` (context, not the production tier) | 0/512 fallbacks, max resid 1.5e-3 | — | Also passes on THIS table (the frame-potential anchor table's own subsets are better-conditioned than the "realistic near-collinear" *episode* keys the `n_iter=12→20` escalation was originally about, §16.3 — no contradiction: production tier stays 20, matching geo3-alone's own established K=48 setting, since the actual `bind()` NS call orthogonalizes the *blended* key (part raw episode key, part anchor row), not a pure anchor-table subset) |

  **All three legs PASS at K=48**, run this session, CPU-only, on the
  registered init, using the unmodified `gate2_ns_leg`/`raw_table_
  conditioning`/`gate2_construction_check` functions in `key_
  anchoring.py` (no new code, a one-line manifest extension only).
  **Build requirement, registered:** `gate2_construction_test.py`
  (the committed CPU test) extends its `ks=(16, 32)` argument to
  `(16, 32, 48)` — a one-line change, asserted by a new Wave −1 smoke
  (§11.9 item 15) that the K=48 leg is present and passing before any
  GPU launch, mirroring the existing test's own "ships with the build"
  discipline.

- **Episode-level pre-NS subset Gram-deviation, K=48, for context
  (mirrors §2.2's own "episode conditioning" report — 512 sampled
  48-subsets of the 107 anchor rows):** mean **3.781**, max **3.901**
  (compare K=32: mean 2.505 / max 2.676; K=16: mean 1.227 / max 1.425,
  all measured on the identical table, identical sampling protocol,
  this session — scaling monotonically with K, as expected, and
  consistent with the design's own already-published K=16/32 numbers
  to within simulation noise from a different random draw). **Episode-
  level conditioning is achievable at K=48**: NS converges cleanly
  despite the larger pre-correction deviation — Newton-Schulz's own
  job is exactly to close this gap, and it does, at the same
  production tier already used for geo3-alone's own K=48 riders.

### 11.4 The hard question — computed ceiling at K=48

**Pre-answered, per the task's own instruction, before any of this
wave's GPU data exists.**

#### 11.4.1 Why i-strong cannot exist at K=48 as currently built (corrected mechanism at Rev K48.1)

**Corrected at Rev K48.1 (attack findings code-truth-auditor MAJOR-1 /
my own independent code read, this session; §11.10) — the original
draft's causal story was wrong, though its conclusion (no i-strong at
K=48 without a code change) still stands.** Candidate (a) (i-strong,
the context-free per-entity pin, §2.1) is currently infeasible at K=48,
but **not** because a combined 48+48=96 trips a `2K ≤ d_state`
dimensional check. Reading `embed_arms.py`/`run_deltanet_rd.py`
directly: `build_i_strong_pool` (`embed_arms.py` L254) takes
`n_per_side` as a parameter **hardcoded to a default of 32** — neither
call site (`run_deltanet_rd.py` L1028, L1284) ever threads `K` into it,
so the pool is **unconditionally** 32 train + 32 held-out, regardless
of K. The train and held-out blocks are each built by an
**independent** QR draw (`_qr_orthonormal_rows(n_per_side, d_state,
seed)` and `(..., seed+1)`, different seeds) — `build_i_strong_pool`'s
own docstring states plainly that "cross-set orthonormality between
the two 32-sets is **not required**" (train/held-out episodes never
mix names within one K-cycle), and the self-test only checks each
32×32 block's own Gram against `I_32`, never a combined 64×64
cross-block Gram. So the QR assert `n<=d` (`_qr_orthonormal_rows`,
L81, correctly cited) would **not** fire at `n_per_side=48` either
(48≤64 holds for each independent draw) — the code never computes
"48+48=96" anywhere in this path, and "saturating `d_state=64` exactly"
(§2.1's own phrasing, echoed here) describes the pool's *row count*,
not a verified joint 64-row orthonormal frame.

**The actual guard is a separate, hardcoded, unconditional CLI assert:**
`run_deltanet_rd.py` L1282, `assert args.K <= 32, "arm (i-strong)
requires K <= 32 (sec 4.4's reduced 32-train+32-heldout pool)"` — a
fixed sampling-adequacy cap (you cannot draw K distinct names from an
always-32-slot pool), unrelated to any QR dimensional-feasibility
computation. This matches §2.1's own, earlier, more careful framing
("that pool was deliberately restricted to **32 ≤ d_state** entities" —
a fixed constant, not "K ≤ d_state"); the error was introduced only in
this new §11.4.1, which silently over-generalized "32" to "K." The
archived K=48 rider (`EXPERIMENT_LOG.md`, "K=48 rider — frontier
extends past d/2," 2026-07-04 — quote verified verbatim this session)
attributes the refusal to "train+heldout identity vectors 96 >
d_state=64" — the SAME imprecise gloss, inherited from that earlier log
entry rather than introduced fresh here, but repeated in this draft as
newly-verified fact, which it was not. **Net effect: i-strong genuinely
does not run at K=48 today** (the CLI assert fires, unconditionally,
regardless of `n_per_side`), and no archived i-strong K=48 data exists
anywhere (confirmed by exhaustive `grep`/`find` across
`experiment-runs/` this session, and independently by every archived
K=48 JSON's `exactness_config.strong_pin: false`) — but the reason is a
hardcoded convention, not a dimensional impossibility, which matters
directly for §11.4.3 below.

#### 11.4.2 The substitute ceiling — computed this session, zero GPU cost

Exactly as §2.2 already established for K=16/32 (retracting the
original "λ→1 ⇒ i-strong" equivalence and replacing it with a computed
ceiling, precisely BECAUSE i-strong's pool-restriction confound makes
its 1.0000 non-representative of the full-pool mechanism), this wave
computes the **λ=1 post-NS drift ceiling** at K=48 via the identical
method: fix one anchor row (8 sampled train entities), resample its
K−1 co-drawn rows from the other 106, run the full row set through
production-tier Newton-Schulz (`n_iter=20`), measure the fixed entity's
pooled pairwise cosine across 32 independent resamples — **on the
same registered frame-potential table, same construction, no pool
restriction, no bypassed learned-key path** (i-strong's own two
disclosed confounds do not apply to this ceiling — it is measured on
the *actual* full-pool mechanism candidate (d) will use, at λ=1).
**Method verified this session** by reproducing K=16/32's own already-
published numbers before trusting the new K=48 figure:

| K | n_iter | This session's reproduction | Design's own published value (§2.2) |
|---|---|---|---|
| 16 | 12 | mean 0.9744 / p10 0.9642 | 0.9745 / 0.9640 |
| 32 | 20 | mean 0.9412 / p10 0.9228 | 0.9423 / 0.9243 |
| **48** | **20** | **mean 0.8987 / p10 0.8712** | **(none exists — new)** |

(K=16/32 reproductions differ from the design's own cited values only
in the 3rd–4th decimal, consistent with a different random draw of the
8 fixed entities/32 resamples — not a methodology discrepancy; this is
the same validation-before-trust discipline this project's Hard Rules
require.)

**The hard question, answered honestly: at K/d=0.75, even PERFECT
orthogonalization+stability (λ=1, the theoretical best case) tops out
at post-NS drift ≈0.8987 — BELOW bare geo3's own already-measured,
UN-anchored trained drift at the easier K=32 point (0.9037, §16.1).**
This is a materially worse packing regime than K=32's own gap to its
ceiling (K=32: baseline 0.9037 → ceiling 0.9423, a gap of 0.0386;
K=48's own internal gap — baseline still to be measured by this wave's
own reference arms, §11.1.1, but the coarse prior of 0.80–0.82 implies
a gap of roughly 0.08–0.10 to its own 0.8987 ceiling, a comparable-to-
larger absolute window, but starting and ending at a materially lower
*absolute* drift level than K=32 ever operates at). **Three concrete
consequences, stated as falsifiable interpretations, not hedges:**

1. **A positive h4 result at K=48 cannot be attributed to "the anchor
   pushed post-NS drift into K=32-quality territory"** — the ceiling
   itself forbids that reading; K=48's best-case stabilized state is
   still worse-conditioned than K=32's own untreated baseline. If
   candidate (d) clears its bar (§11.2) at K=48, the mechanism must be
   operating through a **different or additional channel** — most
   plausibly the same "free" value-geometry relief Wave 1 itself
   already measured as a disclosed bonus at K=32 (**relabeled at Rev
   K48.1 from "Rev 7.1's own confirm-wave" — that data is Wave 1's,
   §9.4, not Rev 7.1's, §11.10**: candidate (d)'s final value-Gram
   deviation ran roughly HALF the
   fresh reference's own, at both prior K) — this is a **pre-registered
   candidate explanation**, checked directly off the same `value_gram_
   deviation_mean` field this wave already logs at every checkpoint,
   not an ad hoc rescue invented after seeing a K=48 result.
2. **A miss at K=48 is not, by itself, evidence the anchoring mechanism
   is "wrong"** — it may simply mean K=48 sits past the point where
   post-NS drift-space stabilization has enough packing headroom left
   to matter, a structurally different failure mode than Outcome C at
   K=32 (mechanism not engaged) or Outcome B (engaged but insufficient,
   §3.5). The falsification map (§11.6) keeps these distinguishable.
3. **This ceiling is itself a data point for the curve** — plotted
   alongside the three h4 points (§11.2's curve-reporting rule), a
   monotonically-declining λ=1 ceiling (0.9745 → 0.9423 → 0.8987 across
   K/d = 0.25/0.50/0.75) is, on its own, a clean, already-computed,
   zero-cost quantitative characterization of how fast the full-pool
   packing problem hardens with load — independent of whether any
   trainable candidate ever reaches it.

#### 11.4.3 The optional ceiling-validation arm — REPLACED at Rev K48.1 (attack finding circularity-bias-hunter Finding 6; §11.10)

**The original draft's asymmetric-pool i-strong proposal is withdrawn,
for two independent reasons surfaced this round:**

1. **Its own premise was wrong.** It was built on §11.4.1's original
   (now-corrected) belief that a symmetric 48+48 i-strong pool is
   dimensionally impossible at `d_state=64`. §11.4.1's correction shows
   the opposite: each side's QR draw only needs `n_per_side ≤ 64`
   independently (cross-set orthonormality is explicitly not required,
   per `build_i_strong_pool`'s own docstring), so a **symmetric** 48+48
   pool is not actually harder to construct than the asymmetric
   48+16 one — the asymmetric shrink (and the C17-generalization-power
   cost it was paying) was solving a constraint that does not exist in
   the code.
2. **Even fixed, it would not validate §11.4.2's ceiling — a category
   error the attack round named directly.** i-strong (candidate (a))
   bypasses `W_k`/`k_conv1d` and the entire learned key path (§2.1) — it
   is a structurally different mechanism from candidate (d)'s
   learned-blend-then-Newton-Schulz pipeline. Measuring i-strong's own
   drift at K=48 (symmetric or asymmetric pool) would answer "what
   ceiling does a hand-built, context-free lookup reach," not "what
   ceiling does candidate (d)'s own full-pool mechanism reach at
   λ→1" — the exact confound §2.1/§8.1 F2 already forced a retraction
   over once (i-strong's 1.0000 is not the reachable target for the
   full-pool learned mechanism). Proposing an i-strong-family arm as a
   validator for §11.4.2's number would repeat that retracted mistake
   inside the same document.

**Registered replacement: a fixed-λ=1 candidate-(d) probe — the same
mechanism §11.4.2 computed, actually trained.** This is a minimal
extension of the existing fixed-grid λ diagnostic already registered
in §2.2 (`anchor_lambda_mode="fixed"`, the pre-registered fallback
grid `λ∈{0.3,0.6,0.9}`) — adding `λ=1.0` as a **ceiling-validation**
point, K=48, candidate (d)'s own architecture, 1 seed, 20,000 steps
(seed **50**, continuing the escalating-block convention). This
measures the *actual trained* full-pool anchor mechanism at its λ→1
extreme — the identical object §11.4.2's computation approximates —
so it is a genuine empirical check of the computed ceiling, not a
different candidate class. Registered as **optional, cut-eligible,
lowest priority** (unchanged from the withdrawn arm's own priority):
§11.4.2's computed number already answers the "hard question" at zero
cost, so this arm's marginal value is confirmation, not a load-bearing
requirement — priced at 1 run (~0.23–0.97 GPU-h, §11.5), never
load-bearing for any Outcome assignment in §11.6, first cut under
budget or schedule pressure.

### 11.5 Budget

**Per-cell cost basis — scaled from realized `wall_s`, per Rev 7.1's
own revised (bracket-not-point) costing discipline (§10.7), extended
by the K=32→K=48 step-cost ratio extracted from the archive this
session, not assumed:**

| K | Arm | Realized `wall_s` (3 seeds) | Mean GPU-h/cell |
|---|---|---|---|
| 16 | geo3 (uninstrumented) | 621.3 / 622.1 / 613.3 | 0.1719 |
| 32 | geo3 (uninstrumented) | 701.3 / 711.7 / 683.9 | 0.1942 |
| 48 | geo3 (uninstrumented) | 895.7 / 871.7 / 883.4 | 0.2454 |
| 32 | candidate (d), drift-probe instrumented (confirm-wave) | 748.5 / 750.5 / 724.4 | 0.2059 |

**K=48/K=32 step-cost ratio (uninstrumented, the cleanest apples-to-
apples pair): 0.2454 / 0.1942 = 1.264×.** Instrumentation overhead at
K=32 (drift-probe vs. bare) is only ≈+6% (0.2059/0.1942). **Gap
disclosed at Rev K48.1 (attack finding budget-schedule-auditor M2;
§11.10):** this K-invariance-of-instrumentation-overhead assumption is
checked at only **one** (K, instrumented-vs-bare) pair — K=32. A second
pair already exists in the archive at zero marginal cost: the
confirm-wave's K=16 seed-0 spot check ran with identical
`drift_probe=True` instrumentation (§9.6), and its bare-K=16 comparator
is this same table's own K=16 row — but that instrumented K=16 cell's
own `wall_s` has never been pulled and compared. **Registered as a
required, free, pre-launch check (not yet performed): compute the K=16
instrumented-vs-bare overhead ratio from the already-archived
`experiment-runs/2026-07-05_keyanchor_confirm/` JSON before trusting
the K-invariance assumption below** — if the K=16 ratio is close to
K=32's own ≈+6%, that is real support for K-invariance; if it diverges
materially, instrumentation overhead may itself scale with K (e.g. item
6a's `σ_K/σ_1` SVD-adjacent computation on a (K,d) matrix is a
plausible candidate for super-linear K-scaling), and the bracket below
would need to widen accordingly. **K=48-adjusted per-cell bracket
(provisional pending that check): 0.18–0.77 × 1.264 ≈ 0.23–0.97
GPU-h/cell** (applied uniformly to every K=48 training cell below,
mandatory or conditional — the same bracket-not-point discipline, since
the dominant cost driver is contention, not workload, and there is no
reason yet, absent the K=16 check, to expect that to change at K=48).

| Item | Cells | Runs | Est. GPU-h (bracket) |
|---|---|---|---|
| Wave −1 CPU smoke suite (inherited + K=48 additions, §11.9) + Gate 2 K=48 leg (already run, this session) | — | 0 | **0** |
| Gate-1-style pre-launch probe, candidate (d), K=48, 5,000 steps | 1 | 1 | 0.06–0.24 (5,000/20,000 × bracket) |
| Reference arms, bare geo3 + drift-diagnostic, K=48, seeds {1,2,3} | 3 | 3 | 0.69–2.91 |
| Candidate (d), learned-λ, full mechanism instrumentation, K=48, seeds {30,31,32} | 3 | 3 | 0.69–2.91 |
| **Mandatory baseline** | | **7** | **~1.44–6.06** |
| Candidate (d′), CONDITIONAL on Rev 7.1's own K=32 verdict (A(d′) or D′), seeds {40,41,42} | ≤3 | ≤3 | ≤0.69–2.91 |
| Seed contingency (either primary arm lands ambiguous, +2 seeds, one iteration) | ≤2 | ≤2 | ≤0.46–1.94 |
| Fixed-λ=1 ceiling-validation probe, OPTIONAL, lowest cut priority (§11.4.3) | ≤1 | ≤1 | ≤0.23–0.97 |
| **All-conditionals-max** | | **≤13** | **~2.8–11.9** |

(**Lower bound corrected at Rev K48.1 — attack finding math-reproducer
m2: summing this table's own component rows gives 1.44+0.69+0.46+0.23 =
2.82, not the original draft's "2.6"; the upper bound, 11.88≈11.9, was
already correct.**)

**Registered nominal ceiling for this wave: ≤12 GPU-h** — the all-
conditionals-max bracket top (≈11.9) sits just inside it, with only
≈1% headroom (compare Rev 7.1's own ≤12 ceiling against its own
~8.6 bracket top — ≈40% headroom); if realized cost tracks the
bracket's own upper edge across every conditional firing
simultaneously, the ceiling-validation probe (lowest priority, §11.4.3)
is cut first, restoring headroom (≈11.9 − 0.97 ≈ 10.9). **Overshoot
cross-check, added at Rev K48.1 (attack finding budget-schedule-auditor
M4; §11.10):** this program's own realized spend has previously
exceeded its own point-estimates by +17–26% (Wave 1: realized 10.98
vs. estimated 8.7–9.4); applying the SAME rate to this wave's own
~11.9 bracket top (`11.9 × 1.20 ≈ 14.3`) would breach the ≤12 ceiling
by ≈2.3 GPU-h. **Registered mitigation, mechanical, not only a
pre-launch paper check:** after each completed K=48 cell, compare
cumulative realized `wall_s` against a running projection for the
remaining mandatory+conditional cells; if the projection would exceed
12, cut conditionals in the stated priority order (ceiling-validation
probe → seed contingency → candidate (d′)) **before** they launch —
the same per-cell, data-driven-cutoff discipline this design's own
§3.4 early-stop already applies to arm-level kills, extended here to
the wave's own aggregate budget.

**Program arithmetic — states the number, doesn't hide it (the task's
own explicit ask), with its own derivation gap disclosed at Rev K48.1
(attack finding budget-schedule-auditor M1; §11.10).** Anchoring
program spend so far is quoted as **≈51.5/80 GPU-h** (`KEY_ANCHORING_
DESIGN.md` §5, `STATE.md`'s own ≈51/80 figure) — but summing every
component this document itself discloses (§5's own confirmed 34.90 +
F-geo-3's realized ~1.67 = 36.57; Wave 1's realized 10.98, §9 header,
running total 47.55; the confirm-wave's own wall_s, three of whose four
legs are in this section's own table, ≈0.8–0.85, running total
≈48.3–48.4) reaches only **≈48.3–49.4**, leaving a **~2–3 GPU-h gap**
between the asserted 51.5 and what this document's own itemization can
reconstruct. `STATE.md` cites this same §5 as its source, but §5's own
arithmetic only reaches "~48 GPU-h program-total" as a pre-Wave-1
*worst-case projection*, not a post-hoc realized sum — so the 51.5
figure is carried forward from an untraced addition, not re-derived
here. **Registered as a required pre-launch action, not deferred
indefinitely:** re-run whatever script sums "all archived `wall_s`
fields" (§5's own method for its 34.9 figure) and publish the itemized
total the same way, closing this gap before either wave's own spend is
added on top of an unverified base. Pending that reconciliation, this
wave's own worst-case arithmetic is stated provisionally: Rev 7.1's own
registered ceiling for its (still-DRAFT, not-yet-run) K=32 mechanism-
tier wave is **≤12 GPU-h**; this wave's own registered ceiling is
**≤12 GPU-h**; worst case, both waves at their full nominal ceilings,
using the asserted-not-yet-reconciled base: **51.5 + 12 + 12 = 75.5/80,
leaving 4.5 GPU-h reserve** under the exactness program's own 80 GPU-h
cap — comfortable in principle, but tight enough (and now resting on an
unreconciled base number) that it is flagged as a real scheduling risk,
not waved through. **Registered sequencing recommendation (not a hard
gate, since both waves' mandatory cells are otherwise independent,
§11.0): reconcile the 51.5 base figure FIRST; run Rev 7.1's own
mech-wave to completion second, confirming its REALIZED (not
estimated) spend; re-check the arithmetic above before this wave's own
mandatory cells launch** — if the reconciled base or Rev 7.1's realized
spend comes in higher than assumed, this wave's own ≤12 nominal may
need to shrink (cut the conditional/optional rows first, in the stated
priority order) to keep the combined worst case under 80. This does not
touch the separate, uninvolved `SCALE_TRANSFER_DESIGN.md` 300 GPU-h
program.

### 11.6 Falsification map + 5 pre-answered attacks

| Result | What it means |
|---|---|
| Candidate (d) at K=48 clears h4 ≥0.0244 (3/3 seeds), items 1–4 of the substitute admission stack pass, h=1 guard holds | **Behavioral positive at K=48** — the curve's third point lands above its own derived bar; per §11.4.2's ceiling analysis, this CANNOT be attributed to post-NS drift-space stabilization alone (the ceiling forbids it) — the write-up must check the value-Gram-relief channel (§11.4.2 point 1) before claiming any mechanism, exactly as Rev 7.1's own r_e/BH engagement test (inherited wholesale, §11.0) would need to pass before any *mechanistic* (not merely behavioral) claim is made |
| Candidate (d) clears h4 but the K=48 reference arms (fresh, §11.1.1) ALSO fail the value-salvage-ratio floor at 3/3 seeds (replicating the archived rider's own non-admissible finding) | **The whole K=48 point is non-admissible by the SAME item-1 gate that already governs every other K** — reported at descriptive tier only, exactly the epistemic status the archived K=48 rider already carries; not a new failure mode, a confirmed old one |
| Candidate (d) misses h4 <0.0244 | **Informative negative, distinguishable from Outcome C at K=32 (mechanism not engaged) by the ceiling context (§11.4.2):** check whether pre-NS/post-NS drift (item 5, inherited) cleared its own bar before concluding anything about the mechanism itself — a miss with item 5 passing cleanly says "the packing ceiling was reached and wasn't enough," a genuinely different, and arguably more informative, finding than a miss with item 5 also failing (mechanism simply didn't engage, as at K=32) |
| Reference-arm engagement bands land UNRESOLVABLE at K=48 (`engaged_48 ≥ 0.8937`) | Post-NS sanity bands report indeterminate at this K (§3.5's own indeterminate-band discipline) — does **not** block h4 admission or the `r_e`/BH engagement test (both are independent of this guard, §11.1.1) |
| Candidate (d′) fires (Rev 7.1 resolved A(d′) or D′) and shows the SAME pattern at K=48 (differential engagement, or destabilization) | Confirms the per-entity-capacity finding generalizes across the K-axis, not just at K=32 — a materially stronger structural claim than either K alone |
| Candidate (d′) fires but shows a DIFFERENT pattern at K=48 (e.g., differential engagement at K=32 but uniform/near-uniform at K=48, or vice versa) | Reported in full as a K-dependent capacity-utilization finding — neither outcome is assumed; both are informative and neither collapses into the other's routing table (§10.6's own A(d′)/C′/D′/Inconclusive rows, applied per-K, never pooled) |
| `REV7_THRESHOLD_PINNED.json`'s integrity chain fails at K=48 (script-hash mismatch, `derive()` re-run mismatch, or timestamp-precedes-launch violation) | **Design-invalidating for this wave's mechanism-tier claim specifically** — exactly Rev 7.1's own §10.8 row, inherited verbatim; the behavioral h4 result is unaffected (item 5/BH engagement gates the *mechanistic* claim only) |
| **Candidate (d) clears h4 ≥0.0244 (3/3 seeds) AND `value_gram_deviation_mean` does NOT show the K=32-style relief bonus (i.e., stays near geo3-alone's own elevated K=48 level rather than moving toward the frozen-arm range) — NEW row at Rev K48.1 (attack finding circularity-bias-hunter Finding 3; §11.10)** | **Unexplained by any channel this design pre-registers.** §11.4.2 point 1 names value-Gram relief as the *only* pre-registered candidate explanation for a hit that the ceiling itself forbids attributing to drift-space stabilization — this row is what happens if THAT explanation also fails to materialize. Reported explicitly as an **open question in the write-up**, not resolved by inventing a new post-hoc channel at that point; this is the one falsification-map cell the original draft's "three concrete consequences" left unfilled, and per this project's own threshold-shopping discipline (§9.7.10 item 1), no new explanation may be adopted for this cell without its own registered derivation and its own attack round |

**Five pre-answered attacks:**

1. **"The h4 bar (≥0.0244) is so small it's nearly meaningless — any
   noise could clear it."** Answer (§11.2): the bar is not chosen for
   absolute size — it transplants the *identical* relative-gain factor
   (1.494×) candidate (d) already demonstrated once, at the one K
   where this mechanism has ever shown daylight, onto K=48's own
   collapsed baseline. Its smallness in absolute terms is a direct,
   disclosed consequence of how far K=48's baseline has already fallen
   (0.0164), not a softened target. A single-seed noise excursion
   clearing 0.0244 would need to be a >49% relative jump over a
   near-floor baseline across 3/3 seeds — checkable directly against
   the reference arms' own seed spread once they exist, and reported
   as a caveat if the margin turns out to be within reference-arm noise.
2. **"You're building a K=48 wave on top of an unresolved K=32
   mechanism question (Rev 7.1 still DRAFT) — isn't that premature?"**
   Answer (§11.0): the mandatory cells here (reference arms, Gate-1
   probe, candidate d) do not depend on Rev 7.1's verdict at all — they
   inherit its **code**, not its **conclusion**. Only candidate (d′) is
   conditioned on Rev 7.1's own outcome, and that dependency is a
   mechanical launch precondition (checked against Rev 7.1's own result
   JSONs), never a judgment call made in this wave's own favor.
3. **"The archived K=48 baseline (0.0164) that the bar is built on is
   itself non-admissible (fails the value-salvage floor) — isn't the
   whole derivation built on sand?"** Answer (§11.2, §11.6's own
   falsification-map row): disclosed, not hidden — the derivation uses
   it only as the best-available anchor point (the identical epistemic
   status `R_MIN_HEADLINE`'s own prior-wave-summary provenance already
   has, §10.3.4). This wave's own fresh reference arms independently
   re-measure whether K=48 geo3-alone *also* fails value-salvage at
   fresh seeds — if it does (plausible, since all 3 archived seeds
   clustered tightly at 0.071–0.094 against a 0.1 floor, suggesting an
   architectural rather than seed-noise effect), the entire K=48 point
   may be non-admissible by the same gate that already governs every
   other K, and this is registered as an explicit, expected-possible
   outcome (§11.6's own falsification-map row), not discovered
   post-hoc.
4. **"The computed λ=1 ceiling (0.8987) is BELOW K=32's own untreated
   baseline (0.9037) — doesn't this mean the whole wave is known to be
   pointless before any GPU is spent?"** Answer (§11.4.2): no — it
   means a positive result, if one occurs, cannot be attributed to
   post-NS drift-space stabilization reaching "K=32-quality" territory,
   and must instead be attributed to a different, pre-registered
   channel (the value-Gram-relief bonus Rev 7.1's own confirm-wave
   already measured at K=32, checkable directly off an already-logged
   field). The wave is not pointless; its POSSIBLE positive outcomes
   are pre-narrowed to a smaller, more falsifiable set of honest
   explanations, which is the entire purpose of stating the hard
   question before the data exists.
5. **"Why register a per-entity λ_e arm (d′) and an optional pin-ceiling
   arm at all, if both are conditional/optional and might never run?"**
   Answer (§11.1, §11.4.3): registering them now — with their exact
   trigger conditions, seeds, and costs pinned before any data exists —
   is what prevents them from becoming free post-hoc additions later
   (exactly the same discipline `R_MIN_HEADLINE`'s "fixed before any
   Rev-7 data exists" provenance and §3.6's pre-registered band formula
   already rely on). A conditional arm that is never triggered costs
   nothing (§11.5's budget table prices it at its own line, separate
   from the mandatory baseline) and is reported as "not triggered,
   per its own pre-registered condition" — not silently omitted.

### 11.7 What this wave does not re-open

Exactly per §10.4's own precedent: this wave never reopens or rescores
any of the 12 Wave-1 JSONs, the 4 confirm-wave JSONs, or (once it
exists) Rev 7.1's own mech-wave JSONs — none of them carry K=48 data,
and this wave's own K=48 cells are new data, read fresh, on their own
terms. §9.5/§9.6/§9.7/§10's Outcome assignments at K=32 stand,
untouched, as historical record; this section's own K=48 Outcome
assignment (once run) is independent and separately reported on the
capacity-curve figure (§11.2's curve-reporting rule), never blended
into a single cross-K verdict.

### 11.8 Instrumentation inheritance — restated as a checklist (nothing new, verified present)

Per the task's explicit "inherits wholesale" instruction, checked
against `key_anchoring.py`/`rev7_threshold_derive.py` this session, not
assumed:

- Direct `r_e` measurement, pre-blend (`measure_full_pool_alignment`'s
  pre-NS-pointed sibling, §10.2) — same function, same resampling
  machinery, applies unchanged at any K (it operates per-entity, over
  the fixed 107-entity pool, never over K).
- Anchor-row-norm logging, full 107-vector + mean/min/max, every
  admission checkpoint (§10.2.1) — K-independent (reads
  `anchor_table.weight[anchor_train_ids]`, a fixed-size parameter slice
  regardless of K).
- Exact-Beta/BH/BY engagement test via the unmodified `REV7_THRESHOLD_
  PINNED.json` (§11.1.1's own verification: K-independent, confirmed).
- Hash-locked writer/gate/readout triple for the engagement-test pin
  (§10.3.3) — reused verbatim, zero new artifact.
- Checkpoint-saving writer/gate/readout triple (§10.10) — reused
  verbatim; K=48 checkpoints write to `/root/data/deltanet_rd_
  keyanchor_ckpts/wavekeyanchor-k48/<cell_name>/`, pulled to
  `/Volumes/1TB_SSD/learned-representations/experiment-runs/
  2026-07-0X_keyanchor_k48/checkpoints/` immediately after each run,
  gated by the same round-trip-load smoke before the wave can be marked
  complete.
- Per-entity `λ_e` interior-band fraction, Spearman rank correlation
  (`λ_e` vs. `r_e`), and the formalized Hartigan's dip test (§10.5.1) —
  apply unchanged to candidate (d′) at K=48, IF and when that arm fires.
- **Per-entity null-validation / hub-detection layer (§10.3.2) — ADDED
  at Rev K48.1, was missing from the original checklist (attack finding
  circularity-bias-hunter Finding 4; §11.10).** The mismatched-pair null
  construction, the pooled `(mean, SD)` tolerance check, the per-entity
  empirical percentile, AND the hub-detection diagnostic (`m_e >
  pooled_mean + 2·SD(row_means)`) all apply unchanged at K=48 —
  inherited, not re-derived. **Disclosed, not silently carried forward:**
  this rule's `mean+2·SD` statistic is itself non-robust (the Rev-7.1
  bounded-verify round recommended `median+2·MAD` instead, a MINOR that
  was never applied to the doc text) — and K=48's own episode-level
  pre-NS Gram deviation is already measured higher than K=32's (mean
  3.781 vs 2.505, §11.3), meaning hub/anisotropy risk is plausibly
  *worse*, not better, at this K. Inheriting the un-hardened statistic
  here is a live, not merely historical, risk; the write-up should treat
  any K=48 hub-detection read with this in mind, and hardening it (to
  `median+2·MAD`) is registered as an available, cheap fix if the attack
  round on this revision wants it prioritized.
- **BH-vs-BY dependence sensitivity at K=48 — NEW disclosure at Rev
  K48.1 (attack finding circularity-bias-hunter Finding 5; §11.10).**
  §10.3.1's BH-primary/BY-cross-check pair was disclosed as a
  dependence-sensitivity hedge without examining whether the dependence
  structure itself could shift with K. K=48 draws ~45% of the 107-entity
  pool per episode (vs. ~30% at K=32) — plausibly increasing true
  cross-entity statistical dependence (more entities jointly
  orthogonalized via one NS call per episode), exactly the regime where
  BH's independence/PRDS assumption becomes more optimistic relative to
  BY's arbitrary-dependence correction. **Pre-registered now:** a
  materially larger BH-vs-BY discovery-count gap at K=48 than was
  observed at K=32 (once both exist) is read as evidence of this
  specific risk, not merely reported in passing alongside the numbers.

### 11.9 Wave −1 smoke suite — inherited items + K=48-specific additions

Items 1–13 of §10.9 are inherited **unchanged** (they are architecture-
level checks — norm invariance, held-out zero-init, the mismatched-pair
null construction, BH/BY correctness, the empirical fallback rule, (d′)
forward/backward and bypass, the dip-test positive control, checkpoint
round-trip, zero-collision manifest assertion — none of them are
K-specific). Two new items, K=48-specific:

14. **Manifest-refactor non-regression** (§11.0's own build-scope
    flag) — build `reference_arms_manifest()`/`keyanchor_wave1_
    manifest()`/`keyanchor_confirm_manifest()`'s generalized,
    K-parameterized forms; assert the K=16/K=32 manifests they produce
    are **byte-identical** (every `_spec(...)` field) to the pre-
    refactor hardcoded versions, before trusting the new K=48 branch.
15. **Gate 2's K=48 leg wired into the committed test** —
    `gate2_construction_test.py`'s `ks` argument extended to
    `(16, 32, 48)`; assert the K=48 leg reports PASS on the registered
    init (reproducing this session's own 0/512-fallback,
    σ_ratio≈1.0/max\|cos\|≈0.284 result) before any GPU launch.
16. **Zero-collision smoke's wave-list scope, corrected (NEW at Rev
    K48.1 — attack finding budget-schedule-auditor M5; §11.10).** §10.9
    item 13's own text enumerates the waves its zero-collision check
    scans against: `wavekeyanchor`, `wavekeyanchor-neg1`, `waveref`,
    `wavekeyanchor-confirm` — **`wavegeo3` (the archive housing the
    pre-existing K=48 bare-geo3 cells at seeds 0/1/2 this same wave's
    own §11.5 cost-basis table reads `wall_s` from) is never in that
    list.** §11.1's own claim ("K alone already guarantees no filename
    collision") is correct only if the actual `out_path()`/zero-
    collision implementation scans the full results tree dynamically
    rather than a hardcoded enumeration matching item 13's own prose.
    **Required, before any K=48 reference-arm or ceiling-validation
    cell launches:** read the actual zero-collision smoke source (not
    this document's prose) to determine which it is; if hardcoded, add
    `wavegeo3` (and any other wave name the archived K=48 rider used)
    to the enumerated list for the K=48-specific smoke variant before
    trusting the "no collision" claim.

---

### 11.10 Rev K48.1 — attack-round response map (finding → change)

**Independent adversarial review, this session:** a 4-agent attack
team (math-reproducer, code-truth-auditor, budget-schedule-auditor,
circularity-bias-hunter) reviewed the Rev K48 DRAFT (commit `daa79dc`)
against the cited source code, archived JSONs, and this document's own
earlier sections — the same discipline (and, for two of the four
agents, an independent from-scratch CPU-venv reproduction of every
computed number) this document's own KEY_ANCHORING_ATTACK_R1/R2/R3.md
rounds used. **Disposition: every finding accepted and addressed below
or in place at its own cited location; nothing was contested or
deferred.** No Outcome/verdict anywhere in §9/§9.6/§9.7/§10 is touched
by this response — all changes are confined to §11's own new content.

| # | Finding (condensed) | Severity | Change made | Where |
|---|---|---|---|---|
| C1 | The bar derivation's "no free choice left at readout" claim was false: an additive functional-form reading is equally "parameter-free" by the same transplant logic and produces a bar ≈9× larger (0.219 vs 0.024) — an undisclosed degree of freedom (circularity-bias-hunter) | FATAL-leaning | Multiplicative-vs-additive choice justified explicitly (additive is ill-defined at a baseline 25× smaller than the calibrating K=32 point, asking for a >13× absolute move with no K=32 analog); "mean, always" pinned as the registered statistic, inheriting this document's own universal pre-existing convention rather than introducing a new one | §11.2 |
| C2 | K=32 reference/candidate-d numbers used in the bar derivation were mislabeled "Rev 7.1 mech-wave" — that wave has never run any GPU cell; the real source is Wave 1 (§9.4) (math-reproducer M1, circularity-bias-hunter Finding 2, independently) | MAJOR | Relabeled to "Wave 1 (§9.4)" everywhere it appears in §11.2 and §11.4.2; the confirm-wave's own close-but-different re-measurement (0.61235) disclosed alongside so a reader chasing either label finds the right file | §11.2, §11.4.2 |
| C3 | §11.4.1's causal mechanism for "i-strong cannot exist at K=48" was wrong: the code has no `2K≤d_state` check; `n_per_side` is hardcoded to 32 regardless of K, and cross-set (train-vs-held-out) orthonormality is explicitly not required per the code's own docstring — the actual guard is a hardcoded `assert args.K<=32` (code-truth-auditor MAJOR-1, independently found this session before the attack round reported it) | MAJOR | §11.4.1 rewritten with the correct mechanism, file:line citations, and an explicit note that §2.1's own earlier, more careful "32 ≤ d_state" framing was the one this new section over-generalized | §11.4.1 |
| C4 | §11.4.3's asymmetric-pool pin arm (a) rested on the same wrong premise as C3, and (b) would not have validated §11.4.2's ceiling anyway — i-strong bypasses the learned key path entirely, a different candidate class (circularity-bias-hunter Finding 6) | MAJOR | Arm withdrawn; replaced with a fixed-λ=1 candidate-(d) probe (extending §2.2's own existing fixed-grid λ diagnostic) — the same mechanism §11.4.2 computed, actually trained, closing both objections at once | §11.4.3, §11.1 arm 5 |
| C5 | §11.0's "grepped directly" manifest-hardcoding line citations were wrong in 3 of 5 cases; one of the three named functions (`keyanchor_confirm_manifest`) doesn't share the claimed shape at all (code-truth-auditor MAJOR-2) | MAJOR | Citations corrected against the actual function definitions and loop lines; `keyanchor_confirm_manifest`'s different shape (K=32 via `range(3)` + one conditional K=16 append) described accurately instead of folded into the blanket claim; the higher-level conclusion (no K=48 branch in any of the three) re-verified and stands | §11.0 |
| C6 | §11.6's falsification map had no row for "h4 clears its bar AND the value-Gram-relief field does NOT show the K=32-style bonus" — every possible outcome was pre-narrated into a non-refuting story (circularity-bias-hunter Finding 3) | MAJOR | New falsification-map row added, registering this specific combination as leaving the result unexplained by any pre-registered channel and requiring it to be reported as an open question, not resolved by inventing a new explanation at write-up time | §11.6 |
| C7 | §11.8's "nothing new, verified present" checklist omitted the per-entity null-validation/hub-detection layer (§10.3.2) entirely, despite claiming wholesale inheritance of §10.3/§10.3.4; that layer's own non-robust statistic (mean+2SD) is more consequential at K=48's own higher episode-level Gram deviation (circularity-bias-hunter Finding 4) | MAJOR | Hub-detection/per-entity-null-layer item added to the §11.8 checklist explicitly, with the known mean+2SD-vs-median+2MAD open item disclosed and flagged as higher-stakes at K=48 | §11.8 |
| C8 | The BH-vs-BY dependence disclosure (§10.3.1) was never re-examined for K: K=48 draws ~45% of the 107-pool per episode vs. ~30% at K=32, plausibly increasing true cross-entity dependence in exactly the regime where BH becomes optimistic relative to BY (circularity-bias-hunter Finding 5) | MAJOR | Disclosure added: a materially larger BH-vs-BY discovery-count gap at K=48 (once both exist) is pre-registered as evidence of this specific risk, not merely reported in passing | §11.8, §11.1.1 (pointer) |
| C9 | The "51.5/80 GPU-h spent" figure is asserted, not derived; summing this document's own itemized components reaches only ≈48.3–49.4, a ~2–3 GPU-h untraced gap; `STATE.md` cites §5 as the source but §5 never contains "51" (budget-schedule-auditor M1) | MAJOR | Gap disclosed explicitly with the itemized partial sum shown; a required pre-launch reconciliation action registered (re-run the `wall_s`-summing method §5 used for its own 34.9 figure); the 75.5/80 worst-case arithmetic re-stated as provisional pending that reconciliation | §11.5 |
| C10 | The K=48/K=32 cost-scaling ratio (1.264×) is checked at only one instrumented data point (K=32, +6%); the confirm-wave's own K=16 instrumented spot check — already archived, zero marginal cost — was never pulled to test whether instrumentation overhead is genuinely K-invariant (budget-schedule-auditor M2) | MAJOR | Required free pre-launch check registered: compute the K=16 instrumented-vs-bare overhead ratio from the already-archived confirm-wave JSON before trusting the K=48 bracket's implicit K-invariance assumption | §11.5 |
| C11 | Candidate (d′)'s "mechanical launch precondition" on Rev 7.1's verdict named no script, JSON field, hash-lock, or failure mode — well below the bar §3.6/§10.3.3 set for their own gates in this same document (budget-schedule-auditor M3) | MAJOR | Claim honestly downgraded to "orchestrator sign-off gated on Rev 7.1's own written verdict" until a concrete `k48_dprime_gate.py`-style script (reading Rev 7.1's raw fields, applying §10.6's routing table programmatically, with its own reachable-both-ways smoke) is actually built — registered as a real, named build item, not asserted as already mechanical | §11.1 item 3 |
| C12 | The ≤12 GPU-h ceiling has only ~1% headroom over its own bracket top (≈11.9); applying this project's own documented +17–26% historical overshoot rate to that bracket top breaches the ceiling by ≈2.3 GPU-h, and no mid-flight (only pre-launch) cost check exists to catch it (budget-schedule-auditor M4) | MAJOR | Overshoot arithmetic stated explicitly; a mechanical per-cell budget check registered (compare cumulative realized `wall_s` against a running projection after each completed cell; cut conditionals in priority order before they launch if the projection would exceed 12) — the same data-driven-cutoff discipline §3.4's early-stop already applies to arm-level kills, extended to the wave's aggregate budget | §11.5 |
| C13 | The zero-collision smoke's enumerated wave list (§10.9 item 13) never includes `wavegeo3` — the one archive actually relevant to K=48's seed reuse (seeds 0/1/2 already used there); if the check is a hardcoded enumeration rather than a dynamic scan, the one real collision risk this wave introduces is never checked (budget-schedule-auditor M5) | MAJOR | New Wave −1 smoke item 16 registered: verify against the actual implementation (hardcoded vs. dynamic) before K=48 launch; add `wavegeo3` to the enumerated list if hardcoded | §11.9 item 16 |
| c1 | Bar arithmetic slip: `0.016357 × 1.49378` stated as `= 0.024440`, full-precision recompute gives `0.024434` (math-reproducer m1) | MINOR | Corrected to `0.024434`; registered bar (0.0244) unaffected either way | §11.2 |
| c2 | All-conditionals-max bracket's own lower bound (2.6) doesn't sum its own listed component rows (2.82); the upper bound (11.9) was already correct (math-reproducer m2) | MINOR | Corrected to ~2.8; ≤12 ceiling unaffected | §11.5 |
| c3 | "Saturating `d_state=64` exactly" (§11.4.1, echoing §2.1) overstates the i-strong pin table's actual orthonormality — the two 32-blocks are independently drawn and never checked against each other (code-truth-auditor MINOR) | MINOR | Folded into the §11.4.1 rewrite (C3): the corrected text states this is a row-count fact, not a verified joint 64-row orthonormal frame | §11.4.1 |

**Status after this response: Rev K48.1, still DRAFT.** Every finding
from this round is addressed in place; per this project's own repeated
finding that independent adversarial audit rounds catch different bugs
each round, this revision itself now needs a fresh verify pass (a
round-2 check that these fixes are complete and didn't introduce new
errors) before CLEARED-FOR-BUILD — not self-certified by the same
session that made the fixes. No GPU has been spent at any point in
this section's drafting or revision.

### 11.11 External verify round (2026-07-06) — fresh independent reviewer, no code changed

**Scope discipline, stated up front:** this is a *design*-only bounded
verify round — no GPU spent, no code edited (read-only against
`key_anchoring.py`, `embed_arms.py`, `run_deltanet_rd.py`,
`run_deltanet_rd_exactness_sweep.py`, `grammar_rd.py`,
`rev7_threshold_derive.py`, `gate2_construction_test.py`,
`geo3_simulator.py`), independent of, and with no access to, the
internal 4-agent Rev K48.1 attack team's own working notes — only this
document's committed text and the repo's committed/archived state.
Every computed number below was reproduced from a **fresh CPU-only
venv** built in this session (`torch`+`numpy` installed clean, no
inherited state), not copied from the design's own prose. **Verdict:
CLEARED-FOR-BUILD.** No new FATAL or MAJOR found in the substance of
§11.2/§11.4; two fresh, genuinely new MINOR findings surfaced (both
citation/process, not math or mechanism) plus one worthwhile
evidentiary addition; both are addressed below, neither blocks build.

**1. THE BAR — reproduced, both readings, pin confirmed clean.**
Recomputed from the raw archived JSONs (not the doc's rounded prose):
K=32 fresh reference mean h4 `rec@0.9` (Wave 1, `waveref` seeds 1/2/3)
= **0.410482**; candidate (d) mean (Wave 1, `wavekeyanchor` seeds 0/1/2)
= **0.613180**; K=48 baseline mean (`wavegeo3` seeds 0/1/2) =
**0.016357**. Multiplicative bar = `0.016357 × (0.613180/0.410482)` =
**0.024435** — matches the design's own stated 0.024434 to the digit
that matters (tiny residual from which rounded intermediates were
carried). Additive-reading recompute = `0.016357 + (0.613180 −
0.410482)` = **0.219055** — matches the design's own disclosed
rejected-alternative value (0.219) essentially exactly; ratio
additive/multiplicative ≈ **8.97×**, consistent with the doc's own "≈9×"
framing. **The pin is genuinely single-valued at readout**: the
document states the multiplicative reading as the registered bar, gives
the bounded-metric argument for rejecting the additive reading (a
demand for a >13× absolute move at a baseline 25× below the calibrating
K=32 point, with no K=32-side analog), and discloses the rejected
number rather than hiding it. No residual free choice found. **On
meaningfulness (the task's own explicit sanity-judgment ask):** 0.0244
is a small absolute number, but it embeds the identical *relative* lift
(1.494×) already demonstrated once at K=32 — clearing it requires a
>49% relative jump over a near-floor baseline, 3/3 seeds, which is not
a low bar in relative terms merely because the baseline collapsed in
absolute terms. This reasoning is sound and adequately pre-registered
(§11.2's own "achievability sanity-check" paragraph, §11.6 pre-answered
attack 1). **Admissibility trap, checked against fresh archive data the
design itself doesn't cite for this purpose:** the design correctly
routes the "candidate (d) clears h4 but K=48 reference arms also fail
value-salvage" scenario to a disclosed, non-silent descriptive-tier
outcome (§11.6 row 2) — this is the right structural answer. Worth
adding as **quantitative support, not previously surfaced in §11**:
Wave 1's own archived K=32 JSONs show candidate (d)'s
`value_salvage_ratio_final` running **0.208–0.250** (mean ≈0.230)
against the fresh K=32 reference's **0.112–0.132** (mean ≈0.119) — i.e.
candidate (d) very roughly **doubles** the value-salvage ratio over
bare geo3 at K=32, comfortably clearing the 0.1 floor on both arms. If
a comparable relative lift transfers to K=48 (whose bare-geo3 baseline
already sits at 0.071–0.094, tantalizingly close to the 0.1 floor), a
~2× multiplier would plausibly clear admissibility — but K=48 is a
harder packing regime (§11.4.2's own ceiling analysis) and this
transfer is not guaranteed, only suggestively supported by an adjacent
K. Recommend citing this Wave-1 value-salvage comparison explicitly in
the write-up as the one piece of existing evidence bearing on whether
K=48's own admissibility trap is likely to bind — not a new gate, just
a disclosure the design currently omits despite having the data
on-hand.

**2. THE CEILING — reproduced at K=16/32/48, method validated.**
Independently re-implemented the λ=1 ceiling protocol from its prose
description (fix one of 8 sampled anchor rows, resample K−1 co-drawn
rows from the other 106, run full-pool production-tier Newton-Schulz,
measure the fixed row's pooled pairwise cosine across 32 resamples) —
built from scratch against `key_anchoring.py::frame_potential_init` and
`geo3_simulator.py::newton_schulz`, not copied from any existing
ceiling-computation script. Results (fresh random draw, same table,
same seed `ANCHOR_INIT_SEED=20260705`): **K=16 n_iter=12: mean 0.9743 /
p10 0.9647** (design: 0.9745/0.9640); **K=32 n_iter=20: mean 0.9409 /
p10 0.9232** (design: 0.9423/0.9243); **K=48 n_iter=20: mean 0.8974 /
p10 0.8710** (design's own session: 0.8987/0.8712). All three land
within the same 3rd–4th-decimal noise band the design itself discloses
for a different random draw — the ceiling number is reproducible, not
fabricated, and the "below K=32's own untreated baseline" reading
(0.897 < 0.904) holds under independent reproduction. **i-strong
infeasibility mechanism — verified at file:line, with a correction to
the design's own citations (see finding EV-1 below):** confirmed
`embed_arms.py::build_i_strong_pool` (L254) hardcodes `n_per_side: int
= 32` with two independent QR draws (`seed`, `seed+1`) and no
cross-block Gram check, matching the design's mechanism claim exactly;
confirmed the real guard is `run_deltanet_rd.py`'s `assert args.K <=
32` — found at **L1438-1439** in the current repo (not L1282 as the
design's own §11.4.1 cites; see EV-1). The two `build_i_strong_pool`
call sites are at **L1167 and L1440** (not L1028/L1284 as cited). The
substantive mechanism claim is correct; only the line numbers are
stale. The optional λ=1-probe replacement (§11.4.3) is well-reasoned:
i-strong bypasses the learned key path entirely, so no i-strong-family
arm — symmetric or asymmetric — could have validated §11.4.2's
number anyway; a trained λ=1 candidate-(d) probe is the right
substitute, correctly scoped as optional/confirmatory, not load-bearing.

**3. GATE-2 AT K=48 — reproduced exactly, fresh venv, this session.**
Loaded `frame_potential_init(107, 64, seed=20260705)` fresh and ran
`raw_table_conditioning` + `gate2_ns_leg` directly (bypassing
`gate2_construction_check`, which currently `KeyError`s on `ks=(...,
48)` since `GATE2_N_ITER_BY_K` is still `{16: 12, 32: 20}` in the
committed code — confirming the design's own "one-line dict extension,
not yet built" framing is accurate, not aspirational-as-if-already-done).
Results: **G2-a σ-ratio = 0.9999996423721313** (design: 0.999999642,
exact match — the init is fully deterministic under `manual_seed`, so
this is a bit-level reproduction, not independent-noise agreement);
**G2-b max|cos| = 0.2841511368751526** (design: 0.284151, exact match);
**G2-c, K=48, n_iter=20, 512 subsets, seed offset `0+48=48`** (matching
`gate2_construction_check`'s own `seed=seed+K` convention): **0/512
fallbacks, max resid = 1.019e-6** (design: 1.02e-6 — matches almost to
the digit); **n_iter=12 context check: 0/512 fallbacks, max resid =
1.54e-3** (design: 1.5e-3, matches). Episode-level pre-NS Gram
deviation, independently resampled: K=16 mean 1.234/max 1.434, K=32
mean 2.509/max 2.673, K=48 mean 3.783/max 3.901 — design's own cited
values (1.227/1.425, 2.505/2.676, 3.781/3.901) match to within the
disclosed different-random-draw noise. **All three Gate-2 legs
independently confirmed PASS at K=48**, on the registered init, no
code changes required to reproduce.

**4. BUDGET — the ~3 GPU-h gap is real and still open; arithmetic
re-verified.** Traced §5's own text directly: §5 states "the worst case
projects to **~48 GPU-h program-total**" (line 1355) as an explicit
**pre-Wave-1 worst-case projection**, and separately states "**34.9
GPU-h** summed from all archived `wall_s` fields" (line 1354) — §5
never contains the string "51" anywhere. `STATE.md` (line 567) and
§10.7/§11.5 (lines 3250, 4197) all cite "≈51/80" or "≈51.5/80" as if
sourced from §5, but it is not there. Reconstructing the itemized sum
myself: §5's confirmed 34.90 + F-geo-3's realized 1.67 = 36.57; + Wave
1's realized 10.98 (§9 header, independently confirmed against the
"18 mandatory cells... 10.98 realized GPU-h" text) = 47.55; + the
confirm-wave's own `wall_s` (three of its four legs sum to 748.47 +
750.51 + 724.38 = 2223.36s ≈ 0.618 GPU-h, independently pulled and
computed this session from the raw JSONs, slightly under the design's
own "≈0.8–0.85" estimate for this partial leg-set) ≈ **48.1–48.4** —
confirms the design's own disclosed reconstruction (≈48.3–49.4) and the
**~3.1–3.4 GPU-h gap** against the asserted 51.5 remains genuinely
untraced, not resolved by this verify round either (it is correctly
flagged in §11.5/§11.10 C9 as a required pre-launch action, not
silently deferred — appropriately so; I could not independently locate
the missing ~3 GPU-h anywhere in the archive tree, either, in a
targeted search). Program arithmetic recomputed: `51.5 (asserted,
unreconciled) + 12 (this wave's ceiling) + 12 (Rev 7.1's ceiling) =
75.5/80`, leaving 4.5 GPU-h reserve — arithmetic checks out given the
stated (not yet reconciled) base. **K=32→48 cost-scaling validated**:
recomputed `0.2454/0.1942 = 1.2636` (design: 1.264×) directly from the
cited `wall_s` triples; the K=32 instrumentation-overhead check
(`0.2059/0.1942 = 1.060`, design's "+6%") also reproduces exactly.
**This remains an open, correctly-disclosed item, not a newly resolved
one** — recommend performing the registered pre-launch reconciliation
(re-summing `wall_s` program-wide) before this wave's mandatory cells
launch, exactly as §11.5 already requires.

**5. K-INDEPENDENCE of REV7_THRESHOLD_PINNED.json — confirmed.** Read
`rev7_threshold_derive.py` lines 48–50 directly: `N_ENTITIES = 107`,
`D_STATE = 64`, `ALPHA = 0.05` are the only three free inputs, and `K`
does not appear anywhere else in the file (confirmed by full-file
grep, zero hits on a bare `\bK\b` token). Read `grammar_rd.py::
build_entity_pools` (lines 194–253) directly: `heldout_frac: float =
0.5` is a fixed default, the 107/106 split is computed once from a
shuffled 213-name list with no reference to any per-episode draw count,
and no parameter named `K` (or resembling it) appears in the function
signature or body. The design's claim is accurate as stated: the pool
construction is genuinely blind to K, so the pin's constants really do
carry over unchanged.

**6. RESPONSE-MAP INTEGRITY — 4 of 14 findings spot-checked, all
confirmed genuinely fixed at their cited locations** (not merely
claimed in the map): **C4** (asymmetric-pool arm withdrawn, replaced by
the fixed-λ=1 probe) — confirmed at §11.4.3 and §11.1 arm 5, both
containing the "REPLACED at Rev K48.1" / "withdrawn" language cited.
**C7** (hub-detection/per-entity-null layer added to the §11.8
checklist) — confirmed present, with the `mean+2SD`-vs-`median+2MAD`
open item explicitly disclosed. **C11** (candidate (d′)'s launch
precondition honestly downgraded from "mechanical gate" to
"orchestrator sign-off") — confirmed the exact downgraded phrase
appears at §11.0 and §11.1 item 3, matching the map's "Where" column.
**c2** (all-conditionals-max lower bound corrected 2.6→2.8) — confirmed
the table now reads "~2.8–11.9" with the arithmetic shown (1.44+0.69+
0.46+0.23=2.82). All four check out; no evidence of a stale or
unfulfilled map entry in this sample.

**7. Fresh-eyes findings (new this round):**

- **EV-1 (MINOR, new).** §11.4.1's own file:line citations for the
  i-strong guard are themselves stale in the *current* repo state — the
  same class of bug Rev K48.1's own C5 finding already corrected
  elsewhere in this document (§11.0), but §11.4.1's citations were
  apparently never re-checked against the live file at Rev K48.1, and
  have since drifted further. Cited: `run_deltanet_rd.py` L1028, L1282,
  L1284. Actual (this session, `grep -n`): `build_i_strong_pool` call
  sites at **L1167** and **L1440**; the `assert args.K <= 32` line at
  **L1438-1439**. The substantive mechanism/text described is correct
  (verified independently above) — only the absolute line numbers are
  wrong. **Root cause, worth naming plainly:** line-number citations
  in this document are captured by hand at draft time and never
  re-verified by any mechanical check (e.g. a grep-based CI smoke) at
  either drafting or attack-response time, so they silently rot as the
  cited files are edited for unrelated reasons — this is a structural,
  recurring failure mode (this is the *second* citation-drift bug found
  in the same section, after C5's own three wrong citations), not a
  one-off. Recommend, as a cheap process fix: either (a) drop absolute
  line numbers from prose entirely in favor of function/pattern names
  (`grep`-able, drift-proof), or (b) add a committed smoke test that
  greps the cited line numbers against the cited exact substrings and
  fails loudly on drift. Non-blocking for build (the mechanism itself
  is independently re-verified correct in this same round), but flagged
  because it will keep recurring and keep costing an attack-round
  finding each time otherwise.
- **EV-2 (MINOR, new, related to EV-1).** The same +7-line uniform
  offset was found across every one of §11.0's own "corrected" citations
  into `run_deltanet_rd_exactness_sweep.py` (`reference_arms_manifest`
  cited at def-line 307, actually 314; its `for K in (16, 32)` cited at
  318, actually 325; `keyanchor_wave1_manifest` cited at 361, actually
  368, its two loops cited at 392/398, actually 399/405;
  `keyanchor_confirm_manifest` cited at 408, actually 415). The
  uniformity of the +7 offset (six-for-six) indicates a single
  whole-file insertion above these functions after the citations were
  captured, not scattered errors — but it means C5's own "corrected"
  citations are, in the live repo today, wrong again, by a consistent
  amount. Functionally harmless (the higher-level conclusion — no K=48
  branch exists in any of the three functions — was independently
  re-verified by this round via direct `grep`, and stands), but
  reinforces EV-1's root-cause point: these numbers do not survive
  contact with an actively-edited file, and no mechanism catches it.
  Recommend folding into the same fix as EV-1 (name-based citation or a
  drift-detecting smoke), not a separate one.
- **Everything else swept and clean:** `gate2_construction_test.py`
  confirmed still hardcoded to `ks=(16, 32)` at all 4 call sites (matches
  the design's "not yet built" framing); `out_path()`/`is_done()`
  confirmed to key on `spec['name']` within a caller-supplied `out_dir`
  rather than a global dynamic scan, so item 13's "zero-collision" smoke
  genuinely is a hardcoded per-wave-directory enumeration that omits
  `wavegeo3` — C13/item-16's diagnosis and fix are both correct as
  written, independently confirmed by reading `out_path`/`is_done`
  directly (`run_deltanet_rd_exactness_sweep.py` L831-836). h=1 guard
  independently reproduced from the raw K=48 JSONs: 1.0000/1.0000/1.0000
  across all 3 seeds (guard: ≥0.98, cleared with large margin). No
  further FATAL/MAJOR surfaced.

**Verdict: CLEARED-FOR-BUILD.** Every load-bearing number in §11.2/§11.4
was independently reproduced from a fresh CPU venv this session and
matches the design's own claims (bar, both readings; λ=1 ceiling at
K=16/32/48; all three Gate-2 legs at K=48; K-independence of the
Rev-7.1 pin; the ~3 GPU-h budget gap, confirmed still open and correctly
disclosed, not resolved). Response-map integrity holds on the 4-item
sample checked. The two new findings (EV-1, EV-2) are MINOR, process-only
(stale line-number citations, not math/mechanism errors), and do not
block build; folding them into a single "citations rot, stop hand-
tracking them" fix is recommended but not gating. No FATAL or MAJOR
survives this round.

---

## Reproducibility pointers

- This design: `matrix-thinking/KEY_ANCHORING_DESIGN.md` (**Rev 5**,
  2026-07-04 — Rev 1–3 on 2026-07-03, Rev 4 same day; round maps are
  §8.1/§8.2/§8.3/§8.4).
- Attack/verify rounds: `matrix-thinking/KEY_ANCHORING_ATTACK_R1.md`,
  `matrix-thinking/KEY_ANCHORING_ATTACK_R2.md`,
  `matrix-thinking/KEY_ANCHORING_ATTACK_R3.md` (rounds 3 AND 4 — the
  bounded round-4 verify pass is appended to the R3 file).
- Builds on (read, not modified): `matrix-thinking/DELTANET_RD_EXACTNESS_
  DESIGN.md` §14–16 (§16.7 carries the coordinator's dated shared-c
  correction, made from the GPU re-measurement archived at
  `experiment-runs/2026-07-04_geo3_simulator_recheck/` — not from this
  design).
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
  first-checkpoint trajectory extraction (§3.4). Rev 3: the pinned
  moderate/severe/localized collapse tables vs. the amended Gate 2
  (§4), the archive-extractability check for per-seed drift (§3.6),
  geo3 step-4000 trajectory extraction (§3.4), corrected band
  arithmetic (§3.1) — every number quoted with its construction,
  reproducible from repo files + `geo3_simulator.py`'s own functions.
- Next: **coordinator verification of the §8.4 single-field edit
  against the round-4 prescription** (if clean → CLEARED-FOR-BUILD) →
  build: the §2.2 `model_rd.py` diff
  (masked-gather/scatter blend, pre-NS side channel,
  `anchor_trained_mask`, λ logging incl. the registered cadence
  constants + startup assertion and the last-5-point summary fields),
  the frozen frame-potential init with a registered seed,
  `keyanchor_drift_diagnostic.py` (pre-NS + post-NS + full-pool
  per-entity sweep + the h=1 behavioral companion), **the per-K
  drift-threading fix to `geo3_drift_diagnostic.py::main()` +
  `launch_read` (per-K `c` dict; unit test: K=16/K=32 calls receive
  different `c` values whenever the measured per-K drifts differ —
  Rev 3 supplement, in-scope for this wave)**, manifest/gate additions
  to `run_deltanet_rd_exactness_sweep.py` (reference arms FIRST; the
  §3.6 mechanical blinding chain — `BANDS_PINNED.json` writer,
  launcher refusal gate with hash validation, readout timestamp
  assertion, and the Rev 5 override-demotion stamping through
  `build_cmd` into result assembly), the amended Gate 2 + pinned
  regression case + the NaN-injection unit test + smoke 9's
  stamping test as committed CPU tests → independent code
  audit → Gate 2 (CPU) → Wave −1 (smokes 1–8 + probes; Gate 1
  launch-read) → reference arms (§3.6, bands pinned mechanically) →
  Wave 1 (candidates (d) and (c), early-stop armed, per-entity logging
  active) → assess against §3.5's outcome frame → candidate (b) /
  fixed-grid λ / follow-ons only per their registered triggers.
