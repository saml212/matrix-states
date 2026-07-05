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
