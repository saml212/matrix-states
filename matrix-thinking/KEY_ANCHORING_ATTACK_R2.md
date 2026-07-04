# KEY_ANCHORING_ATTACK_R2 — Adversarial attack, round 2, on KEY_ANCHORING_DESIGN.md (Rev 2)

**Attacker verdict: NEEDS-REV-3.** All three Rev-1 FATALs are genuinely
closed — I independently re-derived the frame-potential construction,
the Welch bound, the λ=1 drift ceilings at both K, and the item-6
negative control from scratch (own numpy implementation, not the
designer's own scripts) and matched Rev 2's numbers to within seed
noise. But this round found **5 new MAJOR findings** and **3 MINOR
findings**, several of them **empirically demonstrated with constructed
counterexamples**, not just argued. None require abandoning the
hypothesis or candidate ranking (same conclusion as R1) — all are
concrete, cheap, one-paragraph-to-one-threshold fixes. The two
priority "declared adaptations" (F1's Gate-1 carve-out, M2's
checkpoint-free Gate 2) are each **legitimate as far as they claim to
go, but I built and ran the exact scenario each was worried about and
found both pre-spend gates share a real, bounded-cost blind spot to
*moderate* anchor-table collapse** that only the post-hoc admission
stack (items 5+6) actually catches.

---

## Methodology note

Per the task brief, all computation here is CPU-only, done in an
isolated scratchpad venv (numpy 2.0.2 + CPU-only torch 2.8.0, no CUDA,
no repo files touched). No GPU box was contacted. Two reproduction
strategies were used deliberately: (1) an **independent from-scratch
numpy reimplementation** of the frame-potential init / Newton-Schulz /
drift-ceiling machinery (transcribing only the one piece of math that
must match exactly — the Newton-Schulz update rule itself, verified
against `model_rd.py` L358-385 and `geo3_simulator.py` L65-78, which are
byte-identical to each other) — this is the right way to adversarially
verify a claimant's numbers, not re-running their own scripts; (2)
**direct execution of the actual repo file** `geo3_simulator.py` for
the platform-sensitivity investigation (target 3), since that is a
question about the *actual* code's behavior, not an independently
derivable fact.

---

## PRIORITY TARGET 1 — the two declared adaptations: dodge check

### 1(a) — F1's Gate-1 carve-out (post-NS input kept for the K=16 launch-read)

**Verdict: legitimate as narrowly stated, but I built the exact
counterexample the task asked me to look for, and it works.**

I constructed a *moderately* collapsed 107×64 anchor table (two shared
directions, noise=0.30 — `sigma64/sigma1=0.0989`, `max|cos|=0.5376`,
which **marginally FAILS both item 6a (<0.1) and item 6b (>0.5)** — i.e.
exactly the kind of table item 6 exists to catch) and fed it through:

- **Gate 2** (CPU construction check, 512 subsets, production n_iter):
  `fallbacks=0/512` at both K=16 (n_iter=12) and K=32 (n_iter=20), max
  residual ~1e-15. **Gate 2 PASSES.**
- **Gate 1's actual mechanism** (post-NS drift → `geo3_simulator.launch_read`,
  reproduced by direct execution of the repo file): this table's
  post-NS drift at K=16 is 0.9361, which maps to a predicted `rec@0.9`
  h=4 of **0.9922** — comfortably clears the ≥0.8 bar. **Gate 1 PASSES.**

So a table that already fails the post-hoc admission criterion (item 6)
sails through *both* pre-spend gates. I also checked the sensitivity
boundary: a *severely* collapsed table (noise=0.05, `sigma_ratio=0.0186`,
`max|cos|=0.9367`) drops post-NS K=16 drift to 0.8853, predicting
`rec@0.9`=0.6152 — **below** the 0.8 bar, so Gate 1 *would* catch severe
collapse. The gate is not blind, but its sensitivity is graded and its
threshold sits well above what moderate (still-item-6-failing) collapse
produces. This directly answers the brief's question: **yes, it lets a
collapsed-anchor arm past the launch gate that pre-NS checks would catch
only after spend** — for the moderate-collapse regime specifically.

**Severity: MAJOR, not FATAL.** Consequence is bounded: item 6 (the
post-hoc admission criterion) *does* still catch this at Wave-1
admission time (§3.1's item 6, both 6a and 6b would fail on this
table — see target 4 below on why 6b in particular is robust), so no
false headline claim could survive to publication; the cost is at most
one wasted mandatory-cell run (~0.25-0.28 GPU-h/run, §5's own anchor)
before the post-hoc check flags it. The design's own text already half-
concedes this for Gate 1 ("screens for harm, not benefit," §4) but does
not extend that concession to the collapse-specific failure mode
tested here.

### 1(b) — M2's checkpoint-free Gate 2

**Verdict: honestly scoped, but the phrase "achieves i-strong-level
per-episode conditioning" invites more confidence than the test
delivers.** Gate 2 tests exactly one thing: does Newton-Schulz converge
without the eigh fallback on subsets of the **frozen, pre-training**
anchor init. It is checkpoint-free by construction (λ=1 is a pure
function of the table) and that specific adaptation is correctly
argued (I confirm the local archive holds no weight checkpoints, only
result JSONs, matching the design's own claim). But Gate 2:

- Never touches the table **after** it becomes a trainable `nn.Embedding`
  and receives 20,000 steps of gradient updates (§2.2's own insertion
  spec). If SGD drives the anchor toward the kind of moderate collapse
  tested in 1(a) *during* training (attack surface #2's own named
  scenario), nothing before or during the run catches it except the
  post-hoc item-6 check — Gate 2 cannot, by definition, see this,
  since it only runs once, pre-spend, on the frozen init.
- Never involves the actual grammar/task data, the actual model forward
  pass, or value-side geometry — it is a pure linear-algebra sanity
  check on one matrix, not a proxy for "the arm will clear §14.10's
  four-item admission stack" as the framing loosely implies.

This is a real, disclosed-nowhere gap, but bounded in the same way as
1(a) (post-hoc items 5/6 remain the actual backstop). **MAJOR.**

---

## PRIORITY TARGET 2 — the 0.9423 ceiling re-derivation

**Independently reproduced. Confirms F2's closure with high confidence.**
All numbers below are from my own from-scratch numpy code (frame
potential via projected gradient descent on the exact formula in §2.2,
Newton-Schulz transcribed from `model_rd.py` L358-385), not the
designer's scripts.

| Quantity | Design claims | My reproduction (seed 0) | Seed range (0-3) |
|---|---|---|---|
| Welch bound, n=107,d=64 | ≈0.0796 | 0.079614 | exact (analytic) |
| Frame-potential rms coherence | 0.0796 | 0.0796 | 0.0796 (all seeds) |
| Frame-potential max\|cos\| | 0.2832 | 0.2805 | 0.28–0.29 |
| Frame-potential σ64/σ1 | 1.0000 | 1.0000 | 1.0000 (all seeds) |
| Random-unit max\|cos\|/rms/σratio | 0.4789/0.1265/0.1377 | 0.4747/0.1269/0.1574 | consistent order |
| Subset Gram dev K=16 mean/max | 1.231/1.425 | 1.236/1.424 | — |
| Subset Gram dev K=32 mean/max | 2.508/2.661 | 2.507/2.667 | — |
| Gate-2 fallbacks (512 subsets, both K) | 0/512, resid~8e-7 | 0/512, resid~1e-15 | — |
| **λ=1 ceiling, frame-potential, K=16 mean/p10** | **0.9745/0.9640** | **0.9742/0.9631** | 0.974–0.975 |
| **λ=1 ceiling, frame-potential, K=32 mean/p10** | **0.9423/0.9243** | **0.9424/0.9224** | 0.940–0.942 |
| λ=1 ceiling, random-unit, K=16 mean/p10 | 0.9410/0.9155 | 0.9403/0.9184 | — |
| λ=1 ceiling, random-unit, K=32 mean/p10 | 0.8926/0.8576 | 0.8943/0.8613 | — |

**Every headline number reproduces within seed noise.** The K=32
frame-potential ceiling (the design's single most load-bearing new
number) is also **seed-stable**: my own reruns at seeds 1-3 give
0.9404-0.9412, not a fragile one-off draw. This is a clean CLOSE of F2.

**Negative control (item 6, §3.1):** qualitatively fully confirmed —
across a wide range of collapse severities (noise 0.05-0.5), the
constructed collapsed table always FAILS at least one of 6a/6b while
the OLD post-NS check always reads a blind 1.0000 and pre-NS λ=1 drift
always reads a trivially-gameable 1.0000. The design's *exact* cited
numbers (σ64/σ1=0.0147, max\|cos\|=0.9263) are not bit-reproducible
from my own construction (I get σ_ratio≈0.020, max\|cos\|≈0.925 at the
closest-matching noise level) — but the design never specifies its own
collapsed-table recipe precisely enough to reproduce bit-exactly, and
the qualitative claim (both checks fail; the old check is blind; item 5
alone is gameable) holds at every noise level I tried. **Minor
provenance gap, not a substantive problem** (see m5 below).

**Bands defensibility (the "42% not 50%" finding).** §3.1 states the
K=32 "mechanically engaged" threshold (post-NS drift ≥0.92) represents
"more than half the distance from geo3's 0.9037 toward the 0.9423
ceiling." I computed this precisely: gap = 0.0386, halfway point =
0.9230. **0.92 is actually 42.2% of the distance, not more than half.**
(The K=16 threshold, 0.96, *is* correctly more than half — 55.9% — but
the design doesn't explicitly make that claim for K=16, only for K=32
where the claim is wrong.) This is a **MINOR** numeric-provenance slip
in the same family as R1's m1/m2 findings — doesn't change any
downstream decision, since 0.92 vs. a "correctly-halfway" 0.923 is a
trivial difference, but the project's own standing rule ("adjectives
cannot gate a mechanism claim," and by extension arithmetic backing a
numeric claim should be checked) is violated here.

**The separation argument the task demanded — DOES NOT EXIST.** I went
looking in the archive for seed-level variance of geo3's baseline
drift (0.9037/0.9416) to test whether "engaged at 0.92-0.94" is
distinguishable from "baseline drifted up on a lucky seed." Finding:
**`GEO3_DRIFT_DIAGNOSTIC.json` is the *only* archived drift measurement
that exists, and it was run at a single seed (seed=0) — there is no
seed-level replication of the drift statistic anywhere in the repo or
the SSD archive** (confirmed by exhaustive `find`/`grep` for any other
drift-diagnostic artifact). Worse: I noticed the "trained checkpoint"
language throughout §14.5/§16.6 of the *parent* design (and inherited
verbatim by KEY_ANCHORING's §1/§2.2/§3.1) actually refers to a
**5,000-step probe-trained throwaway model** (`geo3_drift_diagnostic.py`
L136-176's `run_one_k` trains a *fresh* model for `probe_steps=5000`,
it never loads any of the real 20,000-step Wave-geo3 checkpoints) — the
"0.9037"/"0.9416" reference points that the entire §3.1 band scheme is
anchored to were **never measured on the actual, fully-trained,
bar-clearing geo3 model**, only on a short-probe proxy. This is a
genuine, previously-undisclosed gap: **the design cannot currently
support the very separation argument the task asked for, because
neither seed-variance data nor a true-final-checkpoint measurement of
the reference drift level exists.** **MAJOR** — cheap to close (rerun
`geo3_drift_diagnostic.py` at ≥3 seeds, and optionally against the real
20k-step checkpoints if a loading path is added), but currently absent,
and load-bearing for exactly the "engaged vs. lucky seed" question §3.1
exists to answer.

---

## PRIORITY TARGET 3 — platform sensitivity disclosure

**Confirmed exactly, then investigated further than the design does.**
Direct execution of the actual `geo3_simulator.py` (not a
reimplementation) on CPU at the recorded §16.1 inputs:

```
K=16, c=0.9416046...: rec@0.9 h4 = 0.9980   (design: 0.998   ✓ exact match)
K=32, c=0.9037153...: rec@0.9 h4 = 0.0664   (design: 0.0664  ✓ exact match)
```

I went further than the design's own disclosure to check whether this
is Monte-Carlo noise (the design leaves this open, calling it
"platform/RNG-stream differences"):

- **Seed sweep at K=32, same c, CPU:** 0.0664, 0.0723, 0.0898, 0.0820,
  0.0781, 0.0918 — all tightly clustered in 0.066-0.092.
- **8× the batch size (B=4096):** 0.085, 0.079, 0.080 — no movement
  toward 0.77.
- **fp64 instead of fp32, same CPU:** 0.0977 — still in the same
  regime, ruling out CPU fp32-precision as the CPU-side explanation.

**This rules out sampling noise as the explanation for the CPU-side
number** (CPU consistently and robustly gives ~0.07-0.10 regardless of
seed, batch size, or precision) — the discrepancy is a genuine,
reproducible **CPU-vs-GPU** effect, not scatter. My leading hypothesis
(untestable without a GPU, but concrete and falsifiable): PyTorch
defaults `torch.backends.cuda.matmul.allow_tf32=True` on CUDA, which
silently truncates fp32 matmul precision to ~10-bit mantissa on GPU
only. `simulate_recovery`'s 30-round bisection (`perturbed_orthonormal`)
and the K-step delta-rule recursion (`delta_rule_exact`) are both
matmul-heavy and sit at K=32 exactly where the response curve is
steepest (my own sweep: rec@0.9 h4 goes from 0.05 at c=0.90 to 1.0 at
c=0.97 — a narrow, high-slope window) — small precision perturbations
there could plausibly produce a large swing in the output statistic,
while K=16 (a flatter part of the curve, near-saturated already) would
show no visible effect, exactly the asymmetry the design reports.
**Actionable, cheap test for the build phase:** rerun with
`torch.backends.cuda.matmul.allow_tf32=False` (or
`torch.set_float32_matmul_precision('highest')`) on the actual GPU and
see if the CPU/GPU gap closes.

**Does anything evidentiary depend on the unstable simulator?**
Checked directly: **no.** Outcome-F's three signatures (§16.6) are (1)
key-Gram deviation ~3-8e-7 — measured directly from real training
checkpoints, no simulator; (2) cross-episode drift — measured by
`geo3_drift_diagnostic.py`'s `measure_drift`/`pairwise_drift_stats`,
which never calls `simulate_recovery` at all, only plain cosine
statistics on real model outputs; (3) graded h-decay — real `M3_held_out`
behavioral numbers from the actual 20k-step runs. **None of the three
outcome-F legs touch `simulate_recovery`.** Good news, and worth
stating plainly since the task asked.

**But there IS a real consequence one level removed.** §16.7 (parent
design, inherited by KEY_ANCHORING §3.4/§6) builds a whole narrative —
"the registered simulator *overestimates* K=32 recovery (0.7734
predicted vs. 0.4368 measured) because it's missing a value-Gram
term" — entirely on the GPU-computed 0.7734 number. If that number is
itself TF32-inflated rather than a faithful fp32 expectation (my CPU
reproduction, both fp32 and fp64, consistently lands near 0.07-0.10,
i.e. an *underestimate* relative to 0.4368, not an overestimate), the
entire direction of the §16.7 "missing value-Gram term" story could be
an artifact rather than a genuine finding — and KEY_ANCHORING's own
§3.4 early-stop thresholds and §6 Outcome-B2 branch explicitly inherit
that direction ("the direction §4's own pre-existing evidence... already
points" toward value geometry). **This is a fresh finding this round
(not raised by R1) and it is MAJOR**, not because it invalidates the
early-stop thresholds themselves (those are calibrated directly from
geo3's own archived step-2000 trajectories, verified independently
below in target 5, and don't depend on the simulator at all) but
because the *narrative justification* for why Outcome B2 (value
geometry) is the more plausible failure mode leans on a number I could
not independently validate as computed correctly, and the archive does
not log the original GPU run's `actual_gram_resid` (only the CPU rerun
does), so I cannot rule out a GPU-side bisection convergence bug as an
alternative to the TF32 hypothesis. **Recommend:** log
`actual_gram_resid` from any future GPU-side `launch_read` call, and
add one sentence to §16.7/§6 flagging that the overestimate-direction
claim rests on an unresolved platform artifact.

---

## PRIORITY TARGET 4 — co-firing sufficiency

**(a) Per-entity λ collapse — the premise doesn't apply as feared, but
a related row-collapse vector does, and item 6b is more robust against
it than expected.** First: `λ` is confirmed a **single global scalar**
(§2.2: `sigmoid(raw_param)`, `λ_eff[j] = λ · trained_mask[key_ids[j]]`)
— there is no per-entity continuous λ in this design, only a per-entity
*binary* trained/held-out gate. The task's literal "per-entity λ" attack
doesn't apply. I then tested the adjacent, real question: can a LOCAL
subset of degenerate/near-duplicate anchor ROWS hide inside passing
AGGREGATE item-6 statistics? Constructed: 10 of 107 rows collapsed onto
one shared direction (noise=0.02), rest left at the healthy
frame-potential init. Result: **σ64/σ1 stays at 0.1609 (still PASSES
6a)**, but **max|cos| jumps to 0.9825 (FAILS 6b)**. Because 6b is a
`max` statistic, it is structurally immune to dilution-by-aggregation —
any single near-duplicate pair anywhere in the table trips it,
regardless of how small the affected subset is. This is a genuinely
reassuring, evidence-based finding **in the design's favor**: the
co-firing pair is more robust to localized collapse than a first-glance
read of "aggregate stats could hide local failure" would suggest,
because 6a and 6b are not redundant — 6a catches diffuse/global
degradation, 6b catches any local one. **No new finding here beyond
confirming the design holds up.**

**(b) Partial collapse (50 anchored / 57 not) — a genuine, undisclosed
gap.** Grepped `run_deltanet_rd.py` for any per-entity breakdown in the
actual M2/M3 evaluation path: **none exists.** The only per-entity
instrumentation anywhere in this codebase is `geo3_drift_diagnostic.py`'s
8-entity drift probe (a separate, small, non-representative diagnostic
tool), not the main training-loop eval. This means the headline `rec@0.9`
h=4 and drift statistics are pooled averages over randomly-drawn
episodes from the full 107-entity train pool, with **zero visibility**
into whether a passing aggregate number reflects uniform anchoring
benefit or is concentrated in a subset of entities while the rest
behave like bare geo3. Critically, this failure mode would **not** be
caught by item 6 either — item 6 checks geometric conditioning of the
anchor table, not whether the mechanism is *causally engaged* per
entity; a subset of entities could sit at a well-conditioned anchor row
that simply isn't pulling much weight in the blend and nothing here
would show it. §3.3 already sets a direct precedent for exactly this
kind of caveat at the train/held-out boundary ("the honest claim is
'anchoring lifts K=32 composition for entities with trained anchors'")
but the discipline is not extended to heterogeneity **within** the
train pool. **MAJOR** — cheap fix: log a per-decile (or even just
per-8-entity-sample, reusing the existing drift-diagnostic machinery)
recovery/drift breakdown as a required, non-gating diagnostic,
mirroring C17's own resolution (M1).

**(c) λ oscillating but landing in-band at the final step — confirmed
gap, in the design's own stated terms.** §3.2 pins the claim-tier rule
purely on `λ̄` = mean of the final 1,000 logged steps. The **full
trajectory is logged** (required field, §2.2's last bullet) but **no
numeric criterion excludes an oscillating trajectory whose mean happens
to land in [0.2, 0.8]** from being labeled "learned interior anchoring
/ headline-eligible." This is a direct parallel to candidate (b)'s own
EMA-circularity risk, which Rev 2 *did* fix with a pinned numeric
stability criterion (m4: relative Frobenius change <1e-3 by step
10,000) — the same "adjectives/logging-without-exclusion cannot gate a
mechanism claim" discipline was applied there but not here, for
candidate (d)'s own λ. **MAJOR-leaning-MINOR** (plausible but not the
likeliest SGD failure mode for a single scalar under Adam-style
optimizers over 20,000 steps — no evidence either way exists yet since
Wave 1 hasn't run) — cheap fix: add a variance/oscillation bound (e.g.
std of λ over the final 1,000 steps below some threshold) alongside the
existing mean-based band, exactly mirroring m4's own pattern.

---

## PRIORITY TARGET 5 — early-stop thresholds (M3 fix)

**Numbers: exactly verified, code-verified, not just cited-and-trusted.**
Pulled the actual step-2000 checkpoint fields directly from
`wgeo3_rdx_K{16,32}_armgeo3_s{0,1,2}_geo3n{12,20}.json`:

| K | seed | value_gram_deviation_mean (step 2000) | h4 rec@0.9 (step 2000) |
|---|---|---|---|
| 32 | 0/1/2 | 3.3937 / 3.8543 / 3.8991 | 0.1011 / 0.1321 / 0.0834 |
| 16 | 0/1/2 | 1.6596 / 1.6985 / 1.6701 | 0.7797 / 0.7123 / 0.7504 |

**Exact match to §3.4's table**, including the max/min extraction
(3.90/0.0834 at K=32, 1.70/0.7123 at K=16). **CLOSED, high confidence.**

**Is the AND the right combination rule? Empirically demonstrated
soft spot, bounded cost.** The archived data itself shows h4 at step
2000, K=32 has a **58% relative spread** across just 3 known-good
seeds (0.0834 to 0.1321), while value-Gram dev has only a **~15%
relative spread** (3.3937 to 3.8991) at the same checkpoint — h4 is
the noisier of the two joined conditions by a wide margin, at exactly
the single checkpoint (step 2000, no smoothing/averaging window) the
rule fires on. Under the current AND logic, an arm whose value-Gram is
already decisively past 3.90 (a strong, comparatively low-noise
signal) will **not** be killed if h4 happens to read even slightly
above 0.0834 by chance — exactly the "wobbles above the floor" scenario
the task asked me to check for, and the observed seed-level noise
makes it a real (not merely hypothetical) possibility. **The
consequence is bounded and the design's own budget anchor prices it**:
a false-continue costs at most one arm's remaining ~90% of its run
(~0.25 GPU-h, §5), trivial against the wave's ~10-15 GPU-h ceiling — so
the AND (biased toward not-killing) is a defensible engineering choice
on a cost-asymmetry argument, but **the design never states that
argument** — it just asserts the AND without justifying the choice
over OR or acknowledging the demonstrated noise differential between
its two legs. **MINOR-to-MAJOR; recommend one sentence stating the
cost-asymmetry rationale explicitly**, since every other numeric choice
in this design gets exactly that kind of justification and this one
doesn't.

---

## PRIORITY TARGET 6 — the standard sweep

**Budget arithmetic: verified correct.** Re-derived §5's table from its
own row entries: mandatory 10 (Wave −1) + 6 (candidate d) + 6
(candidate c) = 22 runs ✓, ~0.7-1.0 + 1.5-1.7 + 1.5-1.7 = ~3.7-4.4 GPU-h
✓. With conditionals: +3 (λ-grid) +2 (seed contingency) = 27 runs,
+0.8+0.5-0.6 GPU-h = ~5.0-5.8 ✓. All-conditionals-max: +6 (candidate b)
= 33 runs, +1.7-2.0 = ~6.7-7.8 GPU-h ✓. **Every arithmetic step checks
out**, closing R1's m3.

**Cumulative program spend vs. the 80 GPU-h cap: verified, comfortable
margin.** Summed `wall_s` across all 78 result JSONs with that field
under `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/`
(Wave 0/1/F/geo3 + K48 stretch): **34.9 GPU-h already spent.** Adding
KEY_ANCHORING's worst-case 7.8 GPU-h projects to ~42.7 GPU-h total —
comfortably inside the 80 GPU-h cap, confirming the design's
"nowhere near stressing the cap" claim. **CLOSED.**

**Params/FLOPs: verified.** `50,259 × 64 = 3,216,576 ≈ 3.22M`; against
the 12,899,841-param baseline that's **24.94% ≈ "~25%"** ✓, exactly as
claimed.

**Wave −1 composition (10 runs):** internally consistent with the
budget table (m3 already fixed this), but I note the smoke suite is
described only in prose ("anchor init/blend/backward, held-out-bypass
bit-identity, λ-logging, item-5/6 instrument checks"), never as an
explicit numbered list the way the *parent* design enumerates its own
10 Wave −1 smokes (§14.6 items 1-10). The count of "8" is therefore
asserted, not concretely itemized and countable the way the rest of
this design otherwise holds itself to. **MINOR** (presentation/rigor
consistency, not a correctness issue).

**C17 bypass bit-identity smoke — a genuine floating-point subtlety
found by direct testing.** §3.3 claims the held-out bypass is
"achievable exactly... the SAME tensor" via `normalize(1·k_eff_raw + 0)
= k_eff_raw`. I tested this literally: took 1,000 already-unit-norm
fp32 vectors and re-applied `F.normalize`. **663/1000 were bit-identical,
337/1000 differed by up to 5.96e-8 (1 ULP)** — normalizing an
already-normalized vector is not universally a floating-point no-op,
because `‖x‖` is rarely *exactly* 1.0 in fp32 after an earlier rounding
step, and dividing by a near-but-not-exactly-1 scalar perturbs the
low-order bit for a genuine fraction of rows. **If the actual Wave −1
smoke implements this check with `torch.equal` (strict bit-identity)
rather than a tolerance, it will intermittently FAIL depending on which
entities land in the test batch — not because of a logic bug, but
because the "achievable exactly" claim is a hair too strong.** **MINOR,
easy fix:** use `torch.allclose(atol=1e-6)` for this smoke, or better,
special-case `λ_eff==0` to skip the redundant `normalize` call
entirely rather than relying on its being a no-op.

---

## Per-R1-finding closure judgment

| R1 finding | Judgment | Basis |
|---|---|---|
| **F1** (item-6 tautology) | **CLOSED** | Items 5/6 rewritten to pre-NS quantities; negative control independently re-run by me and confirmed (old check reads blind 1.0000, item 5 alone reads gameable 1.0000, items 6a/6b both fail on a collapsed table). The declared Gate-1 carve-out is a *separate*, disclosed scope decision — see target 1(a) for a fresh, related MAJOR gap it does not fully insulate against. |
| **F2** (λ→1 ≠ i-strong) | **CLOSED**, high confidence | Every cited number (Welch bound, frame-potential coherence stats, λ=1 ceilings at both K/both inits, subset Gram deviations, Gate-2 admission) independently reproduced from scratch within seed noise. Seed-stability of the K=32 headline ceiling (0.940-0.942 across 4 seeds) checked and holds. |
| **F3** (λ-trajectory bands) | **PARTIALLY-CLOSED** | Numeric bands ARE pre-registered (closes the core ask), but no oscillation/variance exclusion rule exists — only the final-1000-step mean gates the claim tier, unlike candidate (b)'s own m4 fix which DID add a stability criterion. See target 4(c). |
| **M1** (C17 bypass) | **CLOSED**, with a minor caveat | Bypass logic is sound and the scope-limit language is appropriately added. Found one floating-point nuance (bit-identity claim not universally exact, target 6) that could cause smoke-test flakiness if implemented too literally — trivial fix, doesn't touch the underlying design decision. |
| **M2** (no real K=32 gate) | **PARTIALLY-CLOSED** | Gate 2 is real, useful, and its own claimed numbers reproduce almost exactly (0/512 fallbacks both K; my own residuals ~1e-15 vs. claimed ~8e-7 — same order of "way below tolerance," difference likely just which residual is reported, not a discrepancy). Platform-sensitivity disclosure is honest and I independently confirmed the exact 0.0664/0.7734 gap, going further to rule out MC noise and propose a concrete TF32 mechanism. BUT: target 1 empirically demonstrates neither Gate 1 nor Gate 2 reliably catches *moderate* anchor-table collapse — that gap is real, demonstrated, and not addressed by Rev 2's response, even though the post-hoc admission stack (items 5/6) does eventually catch it. |
| **M3** (early-stop) | **PARTIALLY-CLOSED** | All thresholds exactly verified against the archive (code-verified, not just cited). The AND-vs-OR choice has a demonstrated (58% vs. 15% relative seed noise) asymmetry between its two legs that the design never discusses or justifies, though the cost of the resulting gap is bounded and small relative to the wave budget. |
| **m1** (h=21 misquote) | **CLOSED** | Verified directly against archived per-seed finals (0.99939/0.99878/0.99994); corrected range [0.9988,0.9999] is exactly right. |
| **m2** (iii-beta key-Gram range) | **CLOSED** | Verified directly against archive: 2.7156/2.7672/2.7108 — corrected range 2.71-2.77 is exactly right. |
| **m3** (Wave −1 arithmetic) | **CLOSED** | Full budget table re-derived and checks out arithmetically at every row (target 6). |
| **m4** (candidate-b EMA threshold) | **CLOSED** | Numeric criterion is pinned (<1e-3 relative Frobenius change by step 10,000); no issue found. Notably, this same discipline is what's *missing* for candidate (d)'s own λ (see F3 above) — an inconsistency between two fixes in the same revision. |

---

## New findings this round, ranked

**MAJOR (5, all closable without redesign):**
1. Gate 1 + Gate 2 jointly fail to catch a *moderate* anchor-table
   collapse before spend (empirically constructed counterexample,
   target 1a/1b) — caught only post-hoc by item 6, at bounded cost.
2. The "engaged" band reference points (geo3's 0.9037/0.9416 "trained"
   drift) are single-seed, 5,000-step-probe measurements with **no
   seed-level variance data anywhere in the repo** — the "engaged vs.
   lucky seed" separation argument the design's own framing invites
   cannot currently be made (target 2).
3. The §16.7 "simulator overestimates because of a missing value-Gram
   term" narrative — which KEY_ANCHORING's §3.4/§6 explicitly
   inherits — rests on a GPU-computed number (0.7734) I could not
   validate; my CPU reproduction (fp32 and fp64, multiple seeds/batch
   sizes) consistently contradicts its direction, and the archive
   doesn't log enough to distinguish a platform-precision artifact from
   a genuine GPU-side bug (target 3).
4. No per-entity/per-subset breakdown exists anywhere in the actual
   training-eval pipeline, so a genuine partial-anchoring outcome (some
   entities functionally anchored, others not) would be invisible to
   every pre-registered check, including item 6 (target 4b).
5. λ's claim-tier band (§3.2) gates on the mean of the final 1,000
   steps only, with no companion variance/oscillation criterion, unlike
   the parallel fix already made for candidate (b)'s EMA (target 4c).

**MINOR (3):**
1. §3.1's "more than half the distance" justification for the K=32
   engaged-band threshold (0.92) is arithmetically wrong — it's 42.2%
   of the distance, not >50% (target 2).
2. The bit-identity claim for the C17 bypass smoke is not universally
   exact at the ULP level in fp32; ~1/3 of rows would differ if the
   smoke uses strict equality (target 6).
3. The Wave −1 smoke suite is described in prose, not itemized as a
   numbered list the way the parent design's own §14.6 is, making the
   claimed count of "8" asserted rather than concretely enumerable
   (target 6).

---

## Verdict

**NEEDS-REV-3.** All three original FATALs are genuinely, verifiably
closed — Rev 2 is a real improvement, and the hypothesis/candidate
ranking remain sound (unchanged from R1's own conclusion). But this
round found 5 new MAJOR gaps, several demonstrated with actual
constructed counterexamples rather than argued abstractly, that a
design holding itself to this project's own "numeric-not-adjective,
run-the-negative-control, multi-round-audits-catch-different-bugs"
standard should close before spend:

1. Add a per-seed drift-diagnostic re-run (or note the gap explicitly
   if not fixed) so the "engaged" bands have a real noise floor to
   read against.
2. Add the missing λ-variance/oscillation criterion, mirroring m4.
3. Add a per-entity/per-subset diagnostic (reuse the existing 8-entity
   drift-probe machinery) as a required, non-gating report field,
   mirroring the C17/M1 precedent.
4. State explicitly (one sentence each) the cost-asymmetry rationale
   for the M3 AND rule and the residual collapse-detection gap in
   Gates 1/2 (items 5/6 remain the real backstop — say so).
5. Fix the "more than half the distance" arithmetic and the bit-identity
   smoke's tolerance.

None of these require abandoning the wave or re-deriving the ceiling —
they are cheap, targeted, and consistent in spirit with the fixes Rev 2
already made for other findings in this same revision.
