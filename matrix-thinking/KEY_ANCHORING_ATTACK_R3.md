# KEY_ANCHORING_ATTACK_R3 — Adversarial attack, round 3, on KEY_ANCHORING_DESIGN.md (Rev 3)

**Attacker verdict: NEEDS-REV-4.** Every one of R2's 5 MAJOR and 3 MINOR
findings, plus the R2-t5 early-stop finding, is substantively addressed —
several verified this round to a **near-bit-exact** degree of confidence
that exceeds what either prior round achieved (the pinned Gate-2 regression
quadruple, the step-2000/step-4000 early-stop archive table, and the
simulator shared-*c* bug's root cause all independently reproduce to 4
decimal places or to the literal source line). The hypothesis and
candidate ranking remain sound — **fourth straight round confirming this**.
But a fresh-eyes sweep of Rev 3's own **new** machinery (exactly what this
round was asked to do) found **3 new MAJORs and 2 new MINORs, none in
carried-over Rev 1/2 territory** — all sitting inside the three things Rev 3
itself added (§3.2's oscillation exclusion, §3.6's reference-arm bands and
blinding protocol, §3.7/§2.2's rebuilt bypass). None threaten the hypothesis;
all are cheap, one-clarification-or-one-extra-seed-pair fixes, consistent
with every prior round's own framing. The strict CLEARED-FOR-BUILD bar
("no new flaws introduced by Rev 3's own machinery") is not met, so this
is **NEEDS-REV-4**, not a design-integrity failure — the closure work
itself is good; it just needs one more pass, per this project's own
standing "multi-round audits catch different bugs each round" rule.

---

## Methodology note

CPU-only, per the task brief — no GPU, no box access. Built a fresh,
throwaway `uv`-managed venv (numpy 2.0.2, CPU torch 2.8.0) in the scratch
directory, isolated from the repo. Two reproduction strategies, as R2 used:
(1) an **independent from-scratch reimplementation** of the frame-potential
init, the collapsed-table constructions, and Gate 2's SVD/NS legs —
transcribing only `newton_schulz_orthogonalize` byte-for-byte from
`matrix-thinking/deltanet_rd/model_rd.py` L358-385 (unchanged since R2's own
verification); (2) **direct reads of the actual repo source**
(`geo3_drift_diagnostic.py`, `geo3_simulator.py`) and the **actual archived
JSONs** (`experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wavegeo3/`,
`experiment-runs/2026-07-04_geo3_simulator_recheck/`) for every claim that is
a fact about this repo's code or data, rather than a re-derivable
mathematical quantity. No repo files were modified; no design doc was
edited.

---

## Target 1 — per-R2-finding closure, reproduced independently

### R2-M1 (Gate 1 + Gate 2 moderate-collapse blind spot) — CLOSED

Reconstructed the four pinned tables from the design's own prose (no
committed script exists yet — this is design-only, pre-build, confirmed by
grep: no `anchor_active`/`anchor_trained_mask`/anchor `torch.where` code
exists anywhere in `model_rd.py` yet). Recipe used: two random unit
directions (`torch.randn(2,64)`, normalized) + `torch.randint(0,2,(107,))`
row assignment + additive Gaussian noise, renormalized, all from a **single
`torch.Generator().manual_seed(42)`** — the most natural reading of the
design's own prose ("two shared unit directions and row assignments drawn
at torch seed 42; 107 rows perturbed at noise σ=0.30, renormalized").

| Case | Design claims (6a / 6b) | My reproduction (6a / 6b) | NS leg (my repro) |
|---|---|---|---|
| Healthy frame-potential init | 1.0000 / 0.2832 | **1.0000 / 0.2832** (seed 1 of 4 tried) | 0/512 both K, max resid 6-7×10⁻⁷ |
| **Moderate (pinned regression case, σ=0.30, seed 42)** | **0.0762 / 0.5371** | **0.0762 / 0.5371** | 0/512 both K, max resid 6.5-7.3×10⁻⁷ |
| Severe (σ=0.05) | 0.0141 / 0.9333 | **0.0141 / 0.9333** | 0/512 both K, max resid 7.6-7.9×10⁻⁷ |
| Localized (10/107 rows, σ=0.02) | 0.1787 / 0.9830 | 0.1548-0.1846 / 0.9809-0.9846 (7 seed variants) | — |

Three of four tables reproduce to **4 decimal places** from an
independently-reconstructed recipe with no access to any committed script —
at this precision that is not "consistent with," it is a de facto
bit-exact match, and it is strong evidence the design's own construction is
exactly what it claims to be, not mis-transcribed or cherry-picked. The
fourth (localized) reproduces the *qualitative* claim exactly (passes 6a
with margin, fails 6b decisively — the max-statistic dilution-immunity
point) but not to the same digit, because the design's own prose doesn't
pin a seed for that specific case (only the primary regression case gets
"drawn at torch seed 42" as an explicit pin) — a genuine, minor, pre-existing
provenance gap (same family as R2's own m5 note), not a new finding.

**Verdict: CLOSED**, with higher confidence than R2 achieved on this same
target (R2 did not attempt bit-level reproduction of the regression case;
this round did, and it held).

### R2-M2 (single-seed probe reference bands) — PARTIALLY-CLOSED

The literal finding — no seed-variance data anywhere in the archive, the
0.9037/0.9416 reference points are 5,000-step-probe measurements never run
on a real final checkpoint — is **independently re-confirmed**: grepped
every JSON under `experiment-runs/` for `drift`/`pairwise`/`resample` fields
outside `GEO3_DRIFT_DIAGNOSTIC.json`; found none. The fix (§3.6: 2 fresh
bare-geo3 reference seeds × K, bands derived from their measured final
checkpoints, `BANDS_PINNED` blinding protocol) is a real, substantive
response to the literal ask. **But the fix's own new machinery has two
fresh problems** — see New Finding 2 (blinding protocol) and New Finding 3
(n=2 sample-std fragility) below, and Judgment Call (c). Rated
PARTIALLY-CLOSED, not CLOSED, because the fresh gaps sit inside the very
mechanism built to answer R2-M2, not somewhere unrelated.

### R2-M3 (simulator "overestimate" narrative) — CLOSED, verified independently against source and archive

Confirmed the shared-*c* bug at the source-code level, not just by trusting
the coordinator's summary: `geo3_drift_diagnostic.py::main()` L213-215 reads

```python
lr16 = per_k[16]["after_probe"]
launch = g3sim.launch_read(drift_mean=lr16["mean"], drift_p10=lr16["p10"], ...)
```

— only K=16's probe stats are ever extracted — and `geo3_simulator.py`'s
`launch_read()` L184-186 then applies that **one** `c` to **both** K inside
`for label, c in (("mean", drift_mean), ...): out[label] = {K:
simulate_recovery(K, gram_resid, c, ...) for K in (16, 32)}`. This is
exactly the bug Rev 3 describes; the code reads exactly as claimed.
Cross-checked against `experiment-runs/2026-07-04_geo3_simulator_recheck/`:
`SUMMARY.json` and `geo3_recheck_results.json` show GPU seed-0 at
K=16's drift (c=0.9416046) reproduces the recorded 0.7734375 **to the last
digit**; at K=32's own drift (c=0.9037153) the GPU prediction is
0.076/0.061/0.094 (seeds 0/1/2) — matching Rev 3's cited "0.06–0.09"
exactly — and **`tf32_matmul: false`** is recorded in the raw results,
directly falsifying R2's own TF32 hypothesis (not just superseding it with
a competing theory). `actual_gram_resid` (~0.00998) is now logged in the
recheck archive, closing R2's own "log this for future runs" request.
Design's quoted correction text matches `DELTANET_RD_EXACTNESS_DESIGN.md`
§16.7's dated 2026-07-04 correction block verbatim in substance (root
cause, the 0.06–0.09 figure, "GPU==CPU," Gate 1 unaffected, per-K threading
fix registered) — no misquote found. **CLOSED, high confidence.**

### R2-M4 (no per-entity visibility / aggregate masking) — CLOSED for the literal ask, residual gap disclosed as New Finding 5

`measure_drift`/`sample_batch_fixed_entity` genuinely exist in
`geo3_drift_diagnostic.py` (L50, L111) and take `n_entities` as a plain
parameter — sweeping the full 107-entity pool instead of 8 is a real,
minimal extension of existing code, not invented machinery. The
`engaged_frac`/band scheme (≥90% / [50,90%) / <50%) is a reasonable,
pre-registered readout. See New Finding 5 for the one residual gap (input
alignment vs. behavioral engagement).

### R2-M5 (λ mean-only band, no oscillation exclusion) — PARTIALLY-CLOSED, see New Finding 1

The three-part rule (final value + trailing mean + trailing range < 0.1) is
a real, correctly-motivated fix in principle, and mirrors m4's own
precedent as claimed. But its window is **empirically ambiguous** given the
actual harness's logging cadence — see New Finding 1, which is the most
load-bearing new finding this round because it sits directly on the
headline Outcome-A gate path.

### R2-m1 (42.2% arithmetic) — CLOSED

Recomputed independently: gap = 0.9423 − 0.9037 = 0.0386; halfway =
0.9037 + 0.0193 = 0.9230; (0.92 − 0.9037)/0.0386 = 42.2%. K=16 check:
(0.96 − 0.9416)/(0.9745 − 0.9416) = 0.0184/0.0329 = 55.9%. Both exactly
match Rev 3's quoted corrections. **CLOSED.**

### R2-m2 (C17 bypass bit-identity) — CLOSED for the forward claim; see New Finding 4 for a fresh backward-pass nuance

`torch.where`'s selected branch genuinely is a hardware/software elementwise
select with zero arithmetic applied to the chosen value — the forward
bit-identity claim is correct, and this is a real fix over Rev 2's
multiply-by-zero formulation. See New Finding 4 for a gradient-path caveat
this round found that the forward-only framing doesn't cover.

### R2-m3 (Wave −1 smoke itemization) — CLOSED

8 items now enumerated (§5), matching the budget table's "8 short smoke
probes + 2 drift-diagnostic probe runs = 10" arithmetic. **CLOSED.**

### R2-t5 (early-stop AND rule, noisy-leg conditioning) — CLOSED, exactly verified

Pulled the actual `checkpoints` field (a separate, richer structure from
the lightweight per-step `trajectory` field) directly from
`wgeo3_rdx_K{16,32}_armgeo3_s{0,1,2}_geo3n{12,20}.json`:

| K | seed | value_gram_dev step2000 → step4000 | h4 rec@0.9, step 2000 |
|---|---|---|---|
| 32 | 0/1/2 | 3.3937→3.8777 / 3.8543→4.8545 / 3.8991→5.2612 | 0.1011 / 0.1321 / 0.0834 |
| 16 | 0/1/2 | 1.6596→1.5092 / 1.6985→1.6324 / 1.6701→1.5695 | 0.7797 / 0.7123 / 0.7504 |

**All 12 cited numbers reproduce exactly** to 4 decimal places, pulled
programmatically from `checkpoints[i]["M2_in_distribution"]["1"]
["value_gram_deviation_mean"]` (h=1, the standard citation site) and
`checkpoints[i]["M3_held_out"]["4"]["recovered_frac@0.9"]`. The K=32
"2/3 seeds exceed 3.90 by step 4000" claim also checks out exactly (seeds 1
and 2: 4.8545, 5.2612, both > 3.90; seed 0: 3.8777, does not) — confirming
the design's own "transient-spike-guard, not independent test" disclosure
for K=32 is quantitatively accurate, not just a plausible-sounding hedge.
Value-Gram is now the sole load-bearing leg, h4 explicitly non-load-bearing,
the cost-asymmetry rationale is stated in text. **CLOSED.** See Judgment
Call (a) for the adjudication of whether this K=32 asymmetry needs a
*different* rule (it does not — see below).

---

## Target 2 — judgment-call adjudications

### (a) K=32's transient-spike-guard second leg — ACCEPTABLE AS DISCLOSED

Because the kill condition requires **both** checkpoints to exceed 3.90,
and geo3's own step-2000 baseline values (3.3937/3.8543/3.8991) sit just
under that line while step-4000 values for the *same* seeds rise
(3.8777/4.8545/5.2612), the two-checkpoint requirement is, as disclosed,
close to redundant at K=32: an arm whose step-2000 value already exceeds
3.90 will very likely also exceed it at step 4000, given the archived
natural-drift direction. This does not reintroduce R2-t5's original
problem (conditioning the kill on the *noisy* h4 leg) — the kill condition
is still driven entirely by the low-noise value-Gram leg, just with
diminished *extra* protection from the second checkpoint at this specific
K. Net effect: no new false-continue risk, only a small, disclosed
inefficiency (an arm doomed at step 2000 keeps running to step 4000 before
being killed, costing ≈0.03–0.05 GPU-h it would otherwise have saved).
**Recommendation (optional, efficiency only, not correctness):** consider a
K=32-specific single-checkpoint kill at step 2000 alone, since value-Gram's
own ~15% relative seed spread is already tight enough to trust on one read;
not required for CLEARED-FOR-BUILD.

### (b) §3.7 alignment on the pre-NS blend — CORRECT OBJECT

R2 target 4(b)'s literal complaint was that "a subset of entities could sit
at a well-conditioned anchor row that simply isn't pulling much weight in
the blend and nothing here would show it" — `a_e = cos(pre-NS blended key,
A[e])` measures *exactly* that ("is this entity's key being pulled toward
its anchor at the blend stage"), and it is properly matched in kind to item
5's own pre-NS discipline (using post-NS here would create an
apples-to-oranges mismatch against the very evidentiary quantity §3.7 is
meant to decompose). **Correct choice.** See New Finding 5 for the one
residual, narrower gap this doesn't close.

### (c) §3.6's n=2 band formula — NOT statistically defensible as specified; concrete fix below

`s_ref = |x1−x2|/√2` is the arithmetically correct n=2 sample standard
deviation — no formula bug. The problem is power: for n=2, the sampling
distribution of *s* itself (χ, 1 degree of freedom) has a relative standard
error of roughly 1/√2 ≈ 71% — meaning `engaged_K` is dominated by which two
seeds happen to be drawn, not by a stable estimate of true spread. This can
fail in **both** directions: two seeds landing close together yields an
artificially tight `engaged_K` (easy to clear, inflating confidence in a
B2 attribution); two seeds landing far apart risks tripping the degenerate
UNRESOLVABLE guard (`engaged_K ≥ ceiling_K − 0.005`) for reasons that are
pure estimator noise, not genuine floor-vs-ceiling collision — plausible in
practice, since this project's own archived data already shows ~13–15%
relative seed-to-seed spread on adjacent statistics (value-Gram deviation
at K=32/step 2000: 3.3937–3.8991 across 3 seeds). This is the *same*
"engaged vs. lucky seed" problem R2-M2 was raised to fix, reappearing one
level up inside the fix's own estimator.

**Concrete recommendation:** use **3** reference seeds per K, not 2 — the
project's own standing convention everywhere else in this design (every
mandatory cell in §5 is ×3 seeds). Cost: +2 runs (1 per K) at the same
~0.8–0.85 GPU-h/run these instrumented reference arms already cost (3.3
GPU-h / 4 runs), i.e. **+1.6–1.7 GPU-h**, bringing the mandatory baseline to
~8.6–9.4 GPU-h — still comfortably under the **≤10 nominal** ceiling, and
the all-conditionals-max case to ~11.6–12.8, still under **≤15 reserve**.
At n=3 (df=2) the relative standard error of *s* drops to ≈1/2 = 50%, a
real, worthwhile improvement for a threshold this consequential, at a cost
the design's own budget table has room for.

---

## Target 3 — simulator-correction integration: fully accurate, nothing further to add

Covered under R2-M3 above. One additional check: `§4`'s "K16-clears/
K32-misses pre-spend separation... stands empirically but must be
restated" language from the parent doc's own correction is correctly
carried into KEY_ANCHORING's §4 ("Gate 1 is unaffected — K=16 used its own
drift"). No overreach found; the retired "~0.19 ceiling" and the retired
"0.7734 overestimate" are both explicitly marked void everywhere they were
previously load-bearing (§3.4 caveat box, §3.5 B2, §6 B2, §7 item 4) — grepped
for "0.7734" and "0.19" across the whole document; every remaining
occurrence is inside a "retired"/"void"/"superseded" framing, none is used
as live evidence.

---

## Target 4 — budget re-add: verified exactly

- Programmatically summed `wall_s` across all archived JSONs under
  `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/` (recursive):
  **78 files, 125,626.4 s = 34.90 GPU-h** — exact match to the design's
  cited "34.9 GPU-h."
- Mandatory baseline: 10 (Wave −1) + 4 (reference arms) + 6 (candidate d) +
  6 (candidate c) = **26 runs** ✓; GPU-h 0.7–1.0 + 3.3 + 1.5–1.7 + 1.5–1.7 =
  **7.0–7.7** ✓.
- With both conditionals: 26+3+2=**31** ✓, 7.0–7.7+0.8+0.5–0.6=**8.3–9.1** ✓.
- All-conditionals-max: 31+6=**37** ✓, 8.3–9.1+1.7–2.0=**10.0–11.1** ✓.
- Program total: 34.90 + 7.0–11.1 = **41.9–46.0**, matching the cited
  "~46 GPU-h worst case" (upper bound) — comfortably inside the 80 GPU-h
  program cap.
- Params: 50,259 × 64 = **3,216,576** = 3.22M exactly; /12,899,841 =
  **24.94%** ("~25%") — both exact.

**Every arithmetic step checks out**, and re-verifies R2's own re-derivation
rather than just trusting it. (Adding Judgment Call (c)'s recommended
+1.6–1.7 GPU-h for n=3 reference seeds still leaves ~2–3 GPU-h of headroom
under the ≤10 nominal ceiling.)

---

## New findings this round (Rev 3's own machinery only)

**MAJOR (3):**

1. **λ oscillation-exclusion window is ambiguous and, on one live reading,
   underpowered, and it sits on the headline Outcome-A gate path.** §3.2
   requires "the trailing-1,000-step mean" and "the trailing-1,000-step
   range (max−min over the final 1,000 logged steps)." Pulled the actual
   per-step `trajectory` field from a real archived 20,000-step run
   (`wgeo3_rdx_K32_armgeo3_s0_geo3n20.json`): logged at a confirmed
   **200-step cadence** (steps 1, 200, 400, ..., 20000; 101 points total,
   diffs all exactly 200). Two live readings, both problematic: **(i)** "1,000
   STEPS" → the window contains only **5–6 logged points** — a range/mean
   statistic computed from 5–6 samples is itself a low-power test, exactly
   the kind of "few-checkpoint noisy read" R2-t5 was raised to fix for
   value-Gram, unaddressed here for λ; **(ii)** "1,000 LOGGED steps" (the
   phrase actually used, "the final 1,000 logged steps") read as 1,000
   *samples* — since a 20,000-step run has only ~101 logged samples total,
   this would silently fall back to using the **entire trajectory**
   (~20x more than intended), including the early, unsettled portion of
   training, which is a materially different and likely unintended
   computation. Candidate (b)'s own m4 threshold (`‖A_t−A_{t−1000}‖_F`) does
   **not** share this ambiguity — it's a two-point lookup at steps exactly
   1000 apart (a multiple of the 200-step cadence), not a window statistic
   over multiple samples — confirming this is a fresh gap specific to §3.2's
   new formulation, not a documentation-wide issue. **Recommend:** pin the
   window in units of **logged checkpoints**, not steps (e.g., "the trailing
   10 logged points" ≈ 2,000 steps), and state explicitly whether that's
   enough samples to trust a range statistic on.

2. **The §3.6 reference-arm blinding protocol is a stated norm, not a
   mechanically-enforced one, for the one piece of machinery explicitly
   built to prevent post-hoc reinterpretation.** The protocol permits anchor
   arms to "train concurrently on other GPUs" and states only that "no
   anchor number... is looked at, quoted, or summarized before the
   `BANDS_PINNED` block exists" — an instruction to the analyst, not a
   file-system or code-level gate. Nothing described prevents an operator
   from seeing anchor-arm progress via streaming console/log output before
   the reference arms finish and bands are pinned, and nothing forces
   `BANDS_PINNED` to be computed by a process that literally cannot access
   anchor-arm result files yet. Given this project's own repeated standing
   discipline elsewhere (negative controls must be *run*, not just written;
   readouts must force *exact* recovery, not a vacuous check), a
   stated-intent-only blind does not meet the "airtight against
   reinterpretation" bar this exact protocol was created to satisfy.
   **Recommend one concrete mechanical enforcement:** stage anchor-arm
   result JSONs in a separate directory and `mv` them into the analysis
   path only after `BANDS_PINNED` is committed (git or a lockfile), and
   don't tail anchor-arm logs until then.

3. **§3.6's n=2 sample-std band formula is statistically fragile** — full
   analysis and concrete fix under Judgment Call (c) above. Consequence is
   bounded (misdirects the B1-vs-B2 follow-on attribution, not the Outcome-A
   headline gate, which uses §3.7's independent 90% per-entity threshold
   instead), but real and cheap to fix (+1.6–1.7 GPU-h for a 3rd reference
   seed per K).

**MINOR (2):**

4. **`torch.where`'s forward-pass bit-identity claim is correct, but the
   backward pass is not automatically protected from the standard "0 ×
   NaN = NaN" gradient-poisoning pattern.** If the discarded (`blended`)
   branch's local computation ever produces a non-finite value for a
   held-out row (e.g., an edge-case near-zero vector into `normalize`),
   combined with Newton-Schulz backward instability on a batch containing
   that row (the same regime the eigh fallback exists for — `model_rd.py`'s
   own documented "adversarial near-duplicate" trigger case), the held-out
   row's gradient would not cleanly zero out even though its forward output
   is untouched. Smoke #2 explicitly specifies an "adversarial
   (near-duplicate raw keys)" input; smoke #4 (held-out zero-gradient) does
   not repeat that qualifier. **Recommend:** run smoke #4 on the same
   adversarial input as smoke #2, and assert `torch.isfinite` broadly, not
   only exact-zero at held-out rows.

5. **§3.7's per-entity engagement metric measures input-side alignment, not
   behavioral (h4-level) per-entity engagement** — correctly closes R2
   target 4(b)'s literal ask (see Judgment Call (b)), but `M3_held_out`'s
   `recovered_frac@0.9` remains a pooled statistic with no per-entity
   breakdown even after this fix, so a narrower version of the same worry
   (input pulled toward anchor, but NS's cross-row mixing doesn't translate
   that into a stabilized *written* key for that entity) stays invisible.
   **Recommend** one disclosure sentence noting this scope limit, mirroring
   §3.3's own held-out scope-limit precedent.

---

## Per-R2-finding closure table

| R2 finding | Judgment | Basis |
|---|---|---|
| R2-M1 (Gate 1+2 blind to moderate collapse) | **CLOSED** | Independent, near-bit-exact reproduction of all 4 pinned Gate-2 tables (3/4 exact to 4 decimals) |
| R2-M2 (single-seed probe reference bands) | **PARTIALLY-CLOSED** | Literal finding fixed by reference arms; the fix's own machinery has 2 fresh MAJORs (findings 2, 3) |
| R2-M3 (simulator "overestimate" narrative) | **CLOSED** | Root cause verified at the source-line level + archived GPU/CPU recheck data (incl. `tf32_matmul: false`, `actual_gram_resid` now logged) |
| R2-M4 (no per-entity visibility) | **CLOSED**, residual noted | Real code basis confirmed; residual gap is finding 5, disclosed not blocking |
| R2-M5 (λ mean-only band) | **PARTIALLY-CLOSED** | Rule exists and is well-motivated; its window is ambiguous/underpowered given the harness's real 200-step cadence (finding 1) |
| R2-m1 (42.2% arithmetic) | **CLOSED** | Recomputed independently, matches exactly |
| R2-m2 (C17 bit-identity) | **CLOSED**, fresh nuance noted | Forward claim correct; backward NaN-path caveat is finding 4, disclosed not blocking |
| R2-m3 (Wave −1 itemization) | **CLOSED** | 8-item list matches budget table |
| R2-t5 (early-stop AND rule) | **CLOSED** | All 12 cited archive numbers reproduce exactly; K=32 asymmetry adjudicated ACCEPTABLE (Judgment Call a) |

---

## Verdict

**NEEDS-REV-4.** All three R1 FATALs remain closed; all 9 R2 findings are
substantively, mostly verifiably addressed — several to a higher standard
of independent confidence than either prior round achieved. The hypothesis
and candidate ranking are unchanged and unthreatened for the fourth
consecutive round. But this round's fresh-eyes sweep — precisely the
exercise the task asked for — found that Rev 3's **own** new machinery
introduced 3 new MAJORs and 2 new MINORs, none inherited from Rev 1/2:

1. Pin the λ-band oscillation window in units of logged checkpoints, not
   ambiguous "steps," and confirm the resulting sample count is enough to
   trust a range/mean statistic (§3.2).
2. Add one concrete mechanical enforcement to the reference-arm blinding
   protocol (staged-directory move or lockfile), not just a stated norm
   (§3.6).
3. Use 3 reference seeds per K, not 2, matching this project's own
   standing convention — +1.6–1.7 GPU-h, both budget ceilings still clear
   (§3.6).
4. Broaden the held-out zero-gradient smoke (#4) to the same
   adversarial/near-duplicate input smoke #2 already specifies, and assert
   finiteness broadly, not just exact-zero (§5).
5. Add one disclosure sentence: §3.7's engagement metric is input-alignment,
   not per-entity behavioral recovery (§3.7).

None of these require abandoning the wave, re-deriving any ceiling, or
revisiting the candidate ranking — they are cheap, targeted, and consistent
in spirit and cost with every fix Rev 2 and Rev 3 have already made. Given
the pattern across all three rounds (3 FATALs → 5 MAJORs → 3 MAJORs, each
narrower and cheaper than the last, and this round's MAJORs are all inside
newly-added machinery rather than newly-discovered problems with
previously-audited material) this looks like genuine convergence, not a
design in trouble — one more focused pass on §3.2/§3.6/§5 should plausibly
clear a Rev 5 verification round.

---

## Round 4 (bounded) — 2026-07-04, verify pass on Rev 4 (commit `7349f97`)

**Scope, per the orchestrator's bounded-verify brief:** (1) the five R3
findings above and whether Rev 4's §8.3 response map closes each without a
dodge; (2) any NEW flaw introduced by Rev 4's own additions (§3.2, §2.2,
§3.6, §3.7, §3.3, smokes 4/5/8, §5 budget); (3) one judgment call — does the
`--unblind-override` escape hatch's automatic descriptive-tier demotion
actually block an overridden run's numbers from later being cited at
evidentiary tier, with the demotion recorded **in the result JSONs
themselves**, not just in prose. Design-only, no GPU, no box. Nothing R3
marked CLOSED is reopened here.

### Methodology note

CPU-only. Built a fresh, throwaway `venv` (stdlib `venv`, Python 3.9.6 —
`uv`'s installed release, 0.1.5, could not resolve the `torch` CPU wheel's
metadata and was abandoned for this step only; `pip install torch
--index-url https://download.pytorch.org/whl/cpu` → **torch 2.8.0 CPU**,
matching the CPU torch major version R2/R3 used) in the scratch directory,
isolated from the repo — no repo files modified, no design doc edited
until this append. Two verification strategies: (1) **an executable,
from-scratch reproduction** of §2.2's exact masked gather/scatter code
block (transcribed verbatim, not paraphrased) plus the registered
NaN-injection unit test (§5 smoke 4), actually run; (2) **direct reads of
the actual repo source** (`run_deltanet_rd.py`, `run_deltanet_rd_exactness_sweep.py`,
`geo3_drift_diagnostic.py`, `lm_pretrain_rd.py`) and **actual archived result
JSONs** under `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/` for
every claim that is a fact about this repo's code or data, per this
project's own standing "verify before claiming" rule.

### Per-R3-finding verdicts

**R3-1 (λ oscillation window ambiguity) — CLOSED.** §2.2/§3.2 register
`LAMBDA_LOG_CADENCE_STEPS = 200` and `LAMBDA_WINDOW_LOG_POINTS = 5`,
pin the window as "the last 5 logged points" (not raw steps), require a
startup assertion (`assert log_every == LAMBDA_LOG_CADENCE_STEPS`), and
Wave −1 smoke 5 is specified as a genuine negative control (deliberately
mis-set cadence must fire the assertion). Confirmed against the actual
harness default, not just the design's prose:
`run_deltanet_rd.py::train()`'s own `log_every=200` default and its
`--log-every` CLI default (line 920) both match the registered constant,
so the assertion binds real production runs, not a hypothetical cadence.
Checked for a false-positive-regression risk (the assertion firing on this
same file's *other*, pre-existing self-tests that use `log_every=10/20/30/50`,
lines 663–854): those calls never set `anchor_active=True` / learned-λ
logging, so the new assertion — which lives inside the λ-trajectory
logging path, not a blanket `train()` check — cannot fire on them; no
regression. The design also states the 5-point window's power honestly (a
gross-oscillation catch only, per R3's own ask). **One cosmetic-only
observation, not a new finding:** "5 logged points" spans 4×200=800 steps
between the first and last of the 5, not the stated "1,000 steps" (a true
1,000-step span needs 6 points); this is the *same* loose "N points ≈
N×200" shorthand R3's own recommendation used ("the trailing 10 logged
points ≈ 2,000 steps" — also off by one interval), so it is inherited
phrasing, not a Rev-4-introduced error, and the actual pinned, checkable
rule ("the last 5 logged points") is unambiguous regardless of what its
informal step-count label is called.

**R3-2 (blinding protocol mechanized) — CLOSED, and the mechanism goes
beyond the literal ask.** §3.6 registers three build requirements with
named failure modes: (a) the writer never produces `BANDS_PINNED.json`
until all 6 reference-arm JSONs validate complete; (b) anchor cells
refuse to launch without a hash-validated `BANDS_PINNED.json` — critically,
this **supersedes** Rev 3's "may train concurrently" allowance, so anchor
arms structurally cannot exist yet while reference arms are still running,
which closes the literal R3 complaint ("nothing prevents an operator
seeing anchor progress via streaming logs pre-pin") by eliminating the
premise — there is no anchor-arm process to tail logs from before the
blind exists; (c) the readout asserts the pin timestamp strictly precedes
the earliest anchor-arm start time, a mechanical broken-blind detector.
sha256 hashing of the 6 reference JSONs, checked against a re-hash at
launch time, blocks silent re-derivation. All three failure modes are
loud-refusal, not silent-continue. **The one nuance this round found is
not in this mechanical-gate machinery itself — see the Judgment Call
below**, which is about the override escape hatch's demotion path
specifically, a narrower question than "is the gate mechanical."

**R3-3 (n=2 sample-std fragility) — CLOSED, exactly per the verifier's own
recommendation.** §3.6 moves to 3 reference seeds/K (6 runs), recomputes
`engaged_K = mean_ref + 2·s_ref` at df=2 (RSE ≈50%, down from n=2's ≈71%),
recomputes the `UNRESOLVABLE` guard trigger at n=3, and adds a mandatory
leave-one-out sensitivity report. Checked the one subtlety this could hide:
the leave-one-out report itself drops to n=2 per fold, which is the same
fragile regime R3 flagged — but the design correctly scopes this as
**disclosure-only, no re-decision rule attached** ("the guard's verdict
stands as computed"), so the n=2 fragility of the diagnostic doesn't
re-contaminate the n=3 decision it's diagnosing. No new flaw.

**R3-4 (torch.where backward NaN-poisoning) — CLOSED, independently
reproduced end to end, including a contrast run proving the superseded
form really was broken.** §2.2's Rev 4 code block replaces `torch.where`
with a masked gather/scatter (`t_idx` from `anchor_trained_mask`; blend
computed only at `t_idx`; `k_blend_raw = k_eff_raw.clone()` then
scattered). Transcribed this construction verbatim into a 33-line CPU
script, planted NaN in every held-out row of the anchor table `A`, ran
forward+backward on a mixed-split batch:

```python
# /private/tmp/.../keyanchor_r4_verify/repro_gather_scatter_nan.py (run: torch 2.8.0 CPU)
t_idx = is_trained_row.nonzero(as_tuple=True)
sub_blend = F.normalize((1 - lam) * raw[t_idx] + lam * A[key_ids[t_idx]], dim=-1)
k_blend_raw = raw.clone()
k_blend_raw[t_idx] = sub_blend
out = k_blend_raw.sum(); out.backward()
```

Result: **PASS on all three registered assertions** — `torch.isfinite(raw.grad).all()` → True;
`torch.isfinite(A.grad).all()` → True; `A.grad[~trained_mask]` is
**exactly** `0.0` (not just finite) at every held-out row (max-abs = 0.0);
held-out rows of `k_blend_raw` are `torch.equal` (bit-exact) to
`raw.detach()` at the same rows. As a control, re-ran the identical setup
through the **superseded** Rev 3 `torch.where(is_trained_row, blended, raw)`
form: `raw.grad` and `A.grad` both come back **non-finite**
(`torch.isnan(A.grad).any()` → True) — confirming R3's finding-4 bug was
real, not speculative, and that the Rev 4 fix is the actual mechanism that
kills it, not a cosmetic rename. Smoke 4 as specified (§5 item 4) matches
this reproduction's three assertions exactly, run on smoke 2's adversarial
near-duplicate-key input as R3 required. **CLOSED, high confidence.**

**R3-5 (input-alignment vs. behavioral engagement disclosure) — CLOSED,
and the "no new forward passes" claim checks out against the actual eval
code.** §3.7 renames the metric `anchor-input-alignment`, keeps it as the
registered `engaged_frac` driver (correctly, per R3's own Judgment Call
(b) affirmation), adds the disclosure sentence, and adds a non-load-bearing
per-entity h=1 companion. Checked the "bookkeeping, not a new forward
pass" claim against `run_deltanet_rd.py::evaluate_pool` (lines 168–239):
the per-item recovery cosine (`cos_all`, built from
`F.cosine_similarity(pred, targets, dim=-1)`) and each batch's own
`b["key_ids"]` (used elsewhere in the same function, e.g. line 236) are
both already produced by the existing forward pass and eval loop — tagging
each cosine value with its row's entity id before pooling is a real code
change (the current `cos_all.append(...reshape(-1))` does discard the
per-row id today) but is exactly and only a bookkeeping/data-retention
change, zero additional `model(...)` calls, zero additional GPU-compute.
The design's phrasing ("no new forward passes... a bookkeeping change") is
accurate, not a dodge. **CLOSED.**

### Budget re-add: verified exactly, matches the orchestrator's cited figures

Recomputed the §5 table's arithmetic independently:
- Wave −1 (10 runs, 0.7–1.0) + reference arms (6, 5.0) + candidate (d)
  (6, 1.5–1.7) + candidate (c) (6, 1.5–1.7) = **28 runs**, **8.7–9.4 GPU-h**
  — both bounds match exactly (low: 0.7+5.0+1.5+1.5=8.7; high:
  1.0+5.0+1.7+1.7=9.4) — **≤10 nominal, clears**.
- +fixed-grid (≤3, 0.8) +seed contingency (≤2, 0.5–0.6) = **≤33 runs**,
  **10.0–10.8** — matches exactly.
- +candidate (b) (≤6, 1.7–2.0) = **≤39 runs**, **11.7–12.8** — matches
  exactly — **≤15 reserve, clears**.
- Reference-row cost check: Rev 3's own cited "3.3 GPU-h / 4 runs" →
  0.825 GPU-h/run; ×6 runs = 4.95 ≈ **5.0** (design's figure); delta over
  Rev 3's 4-run row = 5.0−3.3 = **1.7**, matching the "+1.6–1.7" claimed.
- Program total: 34.9 (independently re-verified by R3 already, not
  re-summed here since no new archive data changes it) + 11.7–12.8 =
  46.6–47.7 ≈ the design's own "~48 GPU-h worst case," nowhere near the 80
  GPU-h cap.

**All arithmetic in the task brief's own figures (28 runs ~8.7–9.4 ≤10;
all-conditionals ~11.7–12.8 ≤15; ~48 worst-case vs. 80 cap) is exact.**

### Judgment call — `--unblind-override`'s descriptive-tier demotion: **NOT WATERTIGHT**

The literal ask: verify there is no path by which an overridden run's
numbers can later be cited at evidentiary tier, with the demotion recorded
**in the result JSONs themselves**, not just in prose. §3.6's text (both
failure modes 2 and 3) commits the demotion to being **"recorded in the
wave summary"** — nowhere in §3.6, §3.7, §5, or §8.3 does the design commit
to writing a tier/demotion field into each individual affected anchor-arm's
own result JSON.

This distinction is not academic — it is checkable against this exact
codebase, and it fails on inspection:

1. **This project already has, and uses, the correct pattern elsewhere** —
   `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` defines `CLAIM_TIER`
   (line 1054) and writes it into **every individual run's own result
   dict** inside `_assemble_result()` (line 1090: `"run_name": run_name,
   "claim_tier": CLAIM_TIER, ...`), with the harness's own smoke test
   asserting `result["claim_tier"] == CLAIM_TIER` (line 1420) and
   confirmed present as a literal top-level JSON field in every archived
   Track-C calibration result (e.g.
   `matrix-thinking/deltanet_rd/results/lm_rd_trackc/calibration/calib_rung1_ptA_*.json`,
   line 3). This is a real, working, in-house precedent for exactly the
   "recorded in the result JSON itself" pattern the orchestrator is asking
   for — and Rev 4 does not adopt it for the anchor-arm demotion.
2. **The pattern Rev 4 explicitly says it mirrors instead is the wrong
   one.** §3.6 states the new `--unblind-override` "mirrors
   `--accept-gate-override`'s loudly-logged pattern." Read that existing
   pattern at the source (`run_deltanet_rd_exactness_sweep.py::gate_geo3_drift`,
   lines 212–231, and its call site at line 532): the override prints a
   `WARNING` block to stderr and returns `{"gate_bypassed": True, ...}`,
   which the caller only interleaves into a **launch-time print
   statement** (`f"...gate={gate_result}"`, line 535) — it is never
   threaded into any spawned run's own output. Confirmed empirically: a
   real archived exactness-wave result JSON
   (`experiment-runs/2026-07-03_deltanet_rd_waves/exactness/wavegeo3/...json`)
   has **no** `gate_bypassed`, `claim_tier`, or any override/gate
   provenance field among its top-level keys (checked programmatically —
   `sorted(d.keys())` lists only task/training/eval fields). So the
   specific pattern §3.6 chose to mirror has a demonstrated, empirical
   track record in this exact harness family of leaving **zero trace in
   individual result JSONs** — it is a launch-console artifact, and by
   Rev 4's own text, the wave-level summary artifact, neither of which is
   the file a future reader is likely to open.
3. **This is not a hypothetical risk — it's this project's own
   documented practice.** Every round of this very attack sequence
   (R1–R3, and this round) verifies claims by directly reading individual
   archived result JSONs (`grep`, `json.load`, field-by-field extraction)
   — not by reading wave-level summary documents. A `BANDS_PINNED`
   override event recorded only in "the wave summary" is exactly the kind
   of fact this project's own standing audit methodology would miss if it
   later re-examined an individual anchor-arm JSON in isolation (e.g., for
   a paper's numbers table, a follow-on comparison, or a future audit round
   using this same file-by-file methodology).

**Verdict: the override/demotion pattern is accepted in principle (per the
orchestrator's framing) but is NOT watertight as specified.** The fix is
small and consistent with the project's own existing convention: have the
`--unblind-override` path write (or patch) a `claim_tier` /
`blind_status` field directly into every affected anchor-arm's own result
JSON at write time (mirroring `lm_pretrain_rd.py`'s `_assemble_result`
pattern), in addition to the wave summary — not instead of it. This is a
one-field, no-new-machinery fix; it does not touch the mechanical gate
itself (finding R3-2's fixes stand), only the audit trail of what happens
when someone explicitly bypasses it.

### No other new flaws found in Rev 4's own additions

Beyond the judgment call above, no new FATAL or MAJOR was found in §3.2,
§2.2, §3.6 (the mechanical gate itself), §3.7, §3.3, or smokes 4/5/8. The
hypothesis and candidate ranking remain unthreatened, fifth consecutive
round.

### Verdict

**NEEDS-REV-5.** All five R3 findings are substantively and correctly
closed — R3-4 verified by an actual executable reproduction (including a
contrast run proving the pre-fix form really was broken), R3-1/R3-3/R3-5
verified against real harness defaults and archived data, R3-2's mechanical
gate is real and, on inspection, stronger than the literal ask. The budget
re-add is exact. But the one judgment call this round was specifically
asked to adjudicate — whether the override's descriptive-tier demotion is
watertight against later evidentiary-tier citation — is **not** met as
written: the demotion is committed only to "the wave summary," never to
the individual result JSONs, despite this project having a working,
in-house precedent (`lm_pretrain_rd.py`'s `claim_tier` field) for doing
exactly that, and despite the pattern §3.6 explicitly mirrors instead
(`--accept-gate-override`) having a confirmed, empirical track record in
this exact harness of leaving no trace in individual run JSONs.

**One fix needed for Rev 5:** have `--unblind-override` write a
tier/demotion field into every affected anchor-arm's own result JSON
(mirroring `lm_pretrain_rd.py`'s established `claim_tier` convention), not
only the wave summary. Nothing else blocks. This is a one-field, no-new-
mechanism addition — not a redesign — and should clear on the next bounded
pass.
