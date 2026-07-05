# KEYANCHOR_REV7_ATTACK — adversarial attack on KEY_ANCHORING_DESIGN.md §10 (Rev 7 DRAFT)

**Target:** `matrix-thinking/KEY_ANCHORING_DESIGN.md` §10, commit `09eb392`.
**Mandate:** this is attempt #3 at the key-anchoring mechanism-tier claim (Wave 1 →
confirm wave → Rev 6 [rejected, `KEYANCHOR_REV6_ATTACK.md`] → this). Design-only,
read-only box access, no GPU spend. Every number below is independently computed
(pure-Python reproduction, no `scipy`/`torch` available in this sandbox — the Beta
CDF and normal quantile are hand-implemented; see script below) or verified
directly against the repo's git history, source, and archived JSONs — nothing is
taken from the draft's own tables without a re-derivation.

**Reproduction artifact:** `/private/tmp/claude-501/.../scratchpad/beta_check.py`
(CPU-only, pure Python — incomplete-beta via Lentz's continued fraction, normal
quantile via bisection on `math.erf`). All numbers in §1/§6/§7 below are its output.

---

## 0. Bottom line

**Verdict: NEEDS-REV.** The wave is a genuine, substantive response to
`KEYANCHOR_REV6_ATTACK.md` — the `r_e` metric redesign (§10.2), the closed-form
null (§10.3.1), and candidate (d′) (§10.5) are real engineering, not window
dressing, and none of the findings below show the hypothesis or the general
approach is dead. But the draft has one outright **FATAL** (a specific,
load-bearing, and easily-falsified factual claim about work already done), plus
four **MAJOR** gaps in the new machinery itself — including a second instance of
the exact Jensen's-gap failure mode this project already caught once (§9.7.10
item 5) — and the central "attempt #3 admissibility" argument (§10.1, §10.11
item 1) needs a correction: the draft frames the risk as "does the new bar doom
candidate (d)," but an independent reconstruction of the disclosed summary
statistics suggests the *opposite* risk (a statistically-manufactured "free win")
is at least as likely, and the draft never checked this despite having every
number it needed to.

1. **FATAL — the artifact §10.3.3 says already exists does not exist.**
   `REV7_THRESHOLD_PINNED.json` and `rev7_threshold_derive.py` are claimed
   "written and committed as part of this draft" (§10.3.3). `git show --stat
   09eb392` shows the commit touched exactly one file (`KEY_ANCHORING_DESIGN.md`,
   625 insertions); `git log --all --diff-filter=A -- '*REV7*' '*rev7_threshold*'`
   returns nothing, ever; neither file exists in the working tree. The
   specific thing that is supposed to make this wave's blind "strictly
   stronger... than §3.6's own" (§10.1 point 1, §10.3.3, §10.11 item 1) has not
   been produced. This is a "verify before claiming" violation on the project's
   own standing rule, at the load-bearing center of the draft's central defense.
2. **MAJOR — the threshold-pin's mechanical enforcement is a writer + a smoke
   test, not the writer/gate/readout triple §3.6 established and this SAME
   section's own checkpoint mechanism (§10.10) correctly builds.** No launcher
   re-hash of `rev7_threshold_derive.py` before any anchor-arm cell launches; no
   readout-time assertion that the constants actually used trace back (by hash)
   to the pinned file. §10.10 gets all three parts right for checkpoints; §10.3.3
   only gets one (plus a data-independence smoke) for the threshold pin.
3. **MAJOR — §10.3.2's null-pool diagonal is not provably equal to `r_e`, and
   the discrepancy is the same Jensen's-gap mechanism already flagged once
   (§9.7.10 item 5).** `r_e` (§10.2) is a **mean of per-resample cosines**
   (confirmed against the existing, audited `measure_full_pool_alignment`,
   `key_anchoring.py` L690-691, whose pattern `r_e` is explicitly said to reuse).
   `C[i,j]` (§10.3.2) is described as `cos(mean-resample raw key of entity i,
   anchor_table.weight[j])` — a **cosine of the mean vector** — justified
   explicitly by "one matmul... no new forward pass," which is only true for the
   mean-of-vectors construction. "The diagonal `C[e,e]` reproduces `r_e`" is
   asserted, not derived, and is false in general whenever an entity's raw key
   direction varies resample-to-resample (which is the entire premise of this
   design — cross-episode key drift).
4. **MAJOR — the aggregate (mean, SD)-only tolerance check in §10.3.2 cannot
   catch a minority of anisotropic/"hub" entities**, reproducing inside the new
   null-validation machinery the exact "aggregate can mask a disengaged (or
   here, falsely-engaged) minority" blind spot that motivated building §3.7's
   per-entity readout in the first place (§3 below constructs the scenario).
5. **MAJOR — `engaged_frac_v3` silently inherits the `≥90%/[50,90%)/<50%` bands
   that were calibrated for a magnitude criterion (`a_e ≥ 0.9`), and applies
   them unchanged to a fundamentally different significance-rate quantity
   (BH-FDR discovery rate) with no minimum-effect-size floor.** A discovery
   rate is not a magnitude fraction; the draft never re-derives or re-justifies
   the band edges for the new quantity, and — per an illustrative reconstruction
   below (§7) — this substitution plausibly cuts the *other* way from what the
   "attempt #3, is the bar honest" framing (prompt / §10.1) worried about.
6. **The old-data-vs-new-bar arithmetic, computed both ways (§6/§7 below):**
   the median entity's back-solved `r ≈ 0.33–0.35` does fail the disclosed
   Bonferroni-style cross-check (`r ≥ 0.401–0.413`, both values re-derived
   independently below) — but that is *not* the registered primary criterion.
   Reconstructing plausible 107-entity distributions consistent with the
   *only* numbers on record (K=32 s0: `r_e ∈ [0.095, 0.581]`, median ≈0.350,
   "bell-shaped") and running the actual registered BH-FDR procedure against
   them gives mean `engaged_frac_v3 ≈ 0.73–0.89` (up to 0.97 on individual
   draws) — solidly in the Outcome A″ band and brushing the Outcome A
   threshold, the **opposite** of "doomed." Neither this draft nor Rev 6 ran
   this check; it should have been run before the attack round, using exactly
   the numbers already in §9.7.2's own table — the same category of omission
   that got Rev 6 rejected, now on the power side rather than the threshold
   side.

**What is cleared, for balance:** the Beta(31.5,31.5) parameterization is
exactly right (re-derived from scratch, §1); candidate (d′)'s init/parameterization
is fully specified and the outcome map is complete modulo one coarse catch-all
cell (§8); the checkpoint writer/gate/readout triple (§10.10) is properly built,
in contrast to finding 2 (§9); fresh seeds make `is_done()` collision-safe by
construction regardless of wave-name choice, though the wave-name string itself
is never explicitly pinned (§10); the ≤12 GPU-h ceiling and the 80 GPU-h program
cap are not seriously threatened by anything found here, though the per-cell
point estimates are not well-grounded in the two most comparable realized
numbers (§11).

---

## 1. The null: re-deriving Beta(31.5, 31.5) and the Bonferroni r-threshold from scratch

**Parameterization.** For a uniformly random unit vector in `R^d`, one
coordinate (equivalently, the cosine against any fixed reference axis) has the
classical marginal density `f(x) ∝ (1-x²)^{(d-3)/2}` on `[-1,1]`. Substituting
`u = (1+r)/2` (so `r = 2u-1`, `1-r² = 4u(1-u)`) gives `f(u) ∝ [u(1-u)]^{(d-3)/2}
= u^{α-1}(1-u)^{α-1}` with `α - 1 = (d-3)/2 ⟹ α = (d-1)/2`. At `d_state = 64`:
`α = 63/2 = 31.5` exactly. **`(1+r)/2 ~ Beta(31.5, 31.5)` is exactly right, not
an approximation** — confirmed independently, matching §10.3.1 verbatim.

**Variance cross-check.** `Var(U)` for `Beta(α,α)` is `α²/[(2α)²(2α+1)] =
1/(4(2α+1))`; with `α=(d-1)/2`, `2α+1=d`, so `Var(U)=1/(4d)` and `Var(R) =
4·Var(U) = 1/d`. Computed directly: `1/(4·(2·31.5+1)) · 4 = 0.015625 = 1/64`
exactly — matches the `Var(r)=1/d_state` identity §9.7.3/§10.3.1 both lean on.
This part of the design is airtight.

**Bonferroni z and r, both ways.** Computed via a from-scratch normal-quantile
bisection (not `scipy`) and a from-scratch incomplete-beta (Lentz continued
fraction, not `scipy`):

| quantity | Gaussian approximation (disclosed in §10.3.1) | **exact Beta(31.5,31.5)** (the wave's actual primary machinery) |
|---|---|---|
| tail target | `α/n = 0.05/107 = 4.673e-4` | same |
| `z_crit` / equivalent-z | **3.3095** (doc: "≈3.305" ✓) | 3.2073 |
| `r` threshold | **0.4137** (doc: "≥0.413" ✓) | **0.4009** |

The disclosed Gaussian cross-check numbers are correct (small rounding only).
But the design's own registered **primary** procedure computes p-values from
the **exact** Beta CDF (§10.3.1: "this wave uses the exact Beta CDF for the
p-value, not the Gaussian z-approximation... where the Gaussian approximation's
own error is largest"). I quantified the size of that stated-but-unquantified
error at several `r` values (one-sided p, exact vs. Gaussian):

| r | exact-Beta p | Gaussian-approx p | ratio (exact/Gaussian) |
|---|---|---|---|
| 0.095 | 0.2258 | 0.2236 | 1.01 |
| 0.250 | 0.02230 | 0.02275 | 0.98 |
| 0.330 | 3.631e-3 | 4.145e-3 | 0.88 |
| 0.350 | 2.131e-3 | 2.555e-3 | 0.83 |
| 0.413 | 3.14e-4 | 4.77e-4 | 0.66 |
| 0.480 | 2.60e-5 | 6.15e-5 | 0.42 |
| 0.581 | 1.95e-7 | 1.68e-6 | 0.12 |

**Finding (MODERATE, feeds §6/§7):** the exact Beta distribution has genuinely
lighter tails than Gaussian at this `d` (bounded support does that), and the
gap **grows with `r`** — at `r=0.58` the true p-value is ~8× smaller than the
Gaussian-approx number would suggest. The design discloses the *direction* of
this correctly but never quantifies it or states its practical consequence:
**the actual registered primary test is meaningfully more powerful (easier to
clear) at the r-values candidate (d) has already shown (up to 0.581, §9.7.4)
than the disclosed "z≈3.305/r≥0.413" reference number implies.** This is not
fatal on its own, but it directly feeds finding 6/§7 below.

---

## 2. FATAL — the hash-locked artifact §10.3.3 claims already exists does not exist

§10.3.3, verbatim: *"The output (`REV7_THRESHOLD_PINNED.json`: BH q, Bonferroni
z_crit, both derived constants, a sha256 of the script itself, a timestamp) is
written and committed **as part of this draft, not deferred to Wave −1** —
there is no reason to wait, since nothing in it can change once written."*
This claim, plus "this draft's own attack round is... instructed to run the
derivation script itself (`rev7_threshold_derive.py`, CPU, <1s) and confirm its
output before reading anything else in this section" (§10.3.3), is the load-bearing
basis for §10.1 point 1 and §10.11 item 1's entire defense ("a stronger — zero
data dependency, not merely reference-arm-gated — blind than §3.6 itself
requires").

Checked directly, this session:

```
$ git show --stat 09eb392
    matrix-thinking/KEY_ANCHORING_DESIGN.md | 625 ++++++++++++++++++++++++++++++++
    1 file changed, 625 insertions(+)
$ git log --all --diff-filter=A --name-only | grep -i "rev7\|REV7_THRESHOLD"
    (no output)
$ find . -iname "*REV7*" -o -iname "*rev7_threshold*"
    (no output)
```

**Neither file exists anywhere in git history or the working tree, staged or
unstaged.** Commit `09eb392` touched exactly one file. The attack round cannot
"run the derivation script... before reading anything else in this section"
because the script does not exist. §10.9 item 5 (the smoke that's supposed to
prove the script's output is data-independent) cannot have been run for the
same reason.

**Why this is FATAL, not MINOR:** this is precisely the class of thing the
project's own Hard Rules exist to catch ("verify before claiming... use web
search or research agents"), applied here to the codebase itself, checkable in
under a minute. It is also self-undermining in a specific way: the whole
point of §10.3's redesign, per §10.1, is that "everything load-bearing is
derived **before any of this wave's data exists**... a strictly earlier blind
than §3.6's own reference-arm gate requires." A blind that is *described* as
already-instantiated but is not, in fact, instantiated, is not a blind at all —
it is a plan for one. Until `rev7_threshold_derive.py` and
`REV7_THRESHOLD_PINNED.json` are actually written, hashed, and committed, and
smoke item 5 actually run and shown to pass, **every claim in §10.1/§10.3.3/
§10.11 item 1 about this wave's blind being "stronger than §3.6's own" is
unverified**, not merely "pending" — the specific verification method the
draft names for itself has not been executed.

---

## 3. MAJOR — the threshold-pin's mechanical enforcement is incomplete relative to its own precedent

Rev 4/R3 finding 2 established, at real engineering cost, the rule this
project now cites repeatedly: a blind must be **"a MECHANICAL HARNESS GATE, not
a stated norm"** — three parts, each with a named failure mode: (1) a **writer**
that only commits after validation, (2) a **launcher gate** that re-validates
(re-hashes) before any dependent cell runs and refuses on mismatch, (3) a
**readout assertion** that the values actually used trace back to the pinned
artifact. §3.6 (`BANDS_PINNED.json`) has all three. §10.10 (checkpoints, this
same section) has all three, explicitly modeled on §3.6.

`REV7_THRESHOLD_PINNED.json` (§10.3.3) has only:
- a **writer** (once it's actually created — see §2 above), and
- a **smoke test** (§10.9 item 5) that the *script's output* doesn't depend on
  wave data.

It has **no launcher-refusal gate** that re-hashes `rev7_threshold_derive.py`
(or re-validates `REV7_THRESHOLD_PINNED.json` against a live re-run of it)
before any candidate-(d)/(d′) cell is allowed to launch, and **no readout-time
assertion** that the BH/Bonferroni constants actually consumed by the readout
script trace back, by hash, to the pinned file rather than a live
recomputation that could have silently drifted (e.g., if
`rev7_threshold_derive.py` is edited — even innocuously, during the build of
candidate (d′)'s own code — between the draft's commit and the wave's launch).

**This is a real, if narrower, gap: "zero data dependency" (a property of the
*formula*) is not the same property as "mechanically tamper-evident" (a
property of the *pipeline*), and the draft's own repeated language ("strictly
stronger than §3.6's own blind") conflates the two.** The formula genuinely has
no data dependency — that part of the claim is fine. The *enforcement*
does not yet match §3.6's or even this same section's own §10.10 standard.
**Fix:** add the missing two legs — a launcher re-hash of the script (refuse
to launch any anchor-arm cell if `sha256(rev7_threshold_derive.py)` doesn't
match the hash recorded in `REV7_THRESHOLD_PINNED.json`) and a readout
assertion that the constants used were read from the pinned file, not
recomputed inline.

---

## 4. MAJOR — §10.3.2's null-pool diagonal is not `r_e`: a second instance of the Jensen's-gap failure mode

This is the sharpest, most concrete finding, because it is directly checkable
against already-committed, already-audited code.

**`r_e` (§10.2), read literally:** *"log `r_e` directly, upstream of the
blend, per entity, per resample: `r_e = mean over n_resamples of cos(k_eff_raw[e]
(pre-blend, this resample), anchor_table.weight[e])`... computed via
`F.cosine_similarity` exactly as `measure_full_pool_alignment` already does for
`a_e`."* Reading `measure_full_pool_alignment` directly
(`key_anchoring.py` L672-693): for each entity `e`, it draws `n_resamples`
independent episodes (`sample_batch_fixed_entity`), computes **one cosine per
resample** (`cos = F.cosine_similarity(blended, anchor_row, dim=-1)`, a
length-`n_resamples` vector), and only then takes `.mean()`. This is
unambiguously **mean-of-cosines**.

**`C[i,j]` (§10.3.2), read literally:** *"compute the full `107 × 107` cosine
matrix `C[i,j] = cos(mean-resample raw key of trained entity i,
anchor_table.weight[j])` — one `matmul` on already-available, already-normalized
tensors, no new forward pass."* "Mean-resample raw key" means: average the
`n_resamples` **raw key vectors** for entity `i` first, then take **one**
cosine of that averaged vector against every anchor row — **cosine-of-the-mean**.
The stated cost justification ("one matmul... no new forward pass") is only
true for this construction: computing the true mean-of-cosines for all
`107×107` pairs would require, for each of 107 query entities, comparing each
of its `n_resamples` raw-key draws against all 107 anchor rows individually and
averaging — an `O(107 × n_resamples × 107)` computation, not "one matmul."

**"The diagonal `C[e,e]` reproduces `r_e`" (§10.3.2) is therefore an assertion,
not a derivation, and is false in general** whenever an entity's raw key
direction varies meaningfully across resamples — which is not an edge case
here, it is **the entire premise of this design**: pre-NS cross-episode key
drift is the named residual bottleneck the whole `KEY_ANCHORING_DESIGN.md`
document exists to fix (§1, §16 of the parent doc), and the confirm-wave's own
disclosed pre-NS pooled drift (0.912–0.919 at K=32, §9.6 table) shows this
variability is real and non-trivial, not a rounding-level effect. Mean-of-cosines
and cosine-of-mean-vector provably diverge (by Jensen's inequality, in the same
direction §9.7.10 item 5 already worked out for `a_e`) whenever the averaged
vectors are not perfectly aligned across resamples — which is exactly the
regime this design operates in.

**Why this matters beyond bookkeeping:** `C[e,e]` is not a decorative
byproduct — the decision rule in §10.3.2 (whether the exact-Beta test or the
empirical-permutation fallback is primary) is validated by comparing the
**off-diagonal** pool's statistics against the analytic null, and the same
matrix's diagonal is asserted to be the primary test statistic itself. If the
diagonal actually diverges from `r_e` by a Jensen's-gap amount, then either (a)
the fallback empirical permutation p-value (§10.3.2, "primary" whenever the
tolerance check fails) is being computed by ranking `r_e` (mean-of-cosines)
against an off-diagonal null pool built from a **different** statistical
construction (mean-of-vectors) than the quantity being ranked — an
apples-to-oranges rank comparison — or (b) the implementation, when actually
written, will have to choose silently between the two constructions, and no
attack round can adjudicate that choice from the design text as written,
because the text asserts they're the same object when they aren't.

**Does the registered smoke suite catch this?** §10.9 item 4 ("a tiny synthetic
5-entity example with known geometry; assert the pairwise-cosine matrix's
diagonal/off-diagonal split matches hand-computed values") is the only
candidate. As specified, it does not obviously exercise resample-to-resample
variation — a toy example with zero or trivial per-resample jitter would make
mean-of-cosines and cosine-of-mean coincide trivially, letting an
inconsistent implementation pass the smoke test cleanly. **Fix:** (1) pick one
construction and make §10.2 and §10.3.2 use the *literal same function call*
for the diagonal (even at the cost of the "one matmul" efficiency claim, or by
explicitly computing the diagonal separately from the off-diagonal, which
already breaks the "one matmul for everything" framing but is honest about
it); (2) if cosine-of-mean is kept for the off-diagonal for cost reasons,
quantify and disclose the Jensen's-gap-driven diagonal/`r_e` discrepancy the
same way §9.7.10 item 5 disclosed it for `a_e`, rather than asserting equality;
(3) add resample-jitter to smoke item 4's synthetic example specifically so it
has power to catch a real implementation bug here, not just a trivial one.

---

## 5. MAJOR — does the mismatched-pair null even generalize per-entity? A failure scenario that survives the aggregate check

The prompt's deepest question: is the random-direction analytic null (or its
empirical mismatched-pair proxy) the right null for a **trained** key against
**its own** anchor, given that training is known in this project to correlate
things for reasons unrelated to the mechanism under test (massive-activation
anisotropy, CLAUDE.md's own §6.8 lesson)? §10.3.2's answer is a **pooled**
decision rule: compute the `107×106 = 11,342` off-diagonal pairs, check whether
the pool's `(mean, SD)` sits within `±0.25σ_chance` / `±25%` of the idealized
`(0, 0.125)`; if yes, trust the analytic Beta test; if no, fall back to the
pool's own empirical permutation p-value.

**Constructed failure scenario.** Suppose a small number of trained entities
(say 5 of 107) are "hub" tokens whose raw keys carry a shared, anisotropic
direction (a well-documented phenomenon in trained embeddings) that happens to
correlate at `r≈0.35` with **every** anchor row, not because of anchoring but
because of shared subspace geometry the raw-key path and the anchor table
independently picked up during training. Their contribution to the pooled
off-diagonal mean: `(5×106/11,342)×0.35 ≈ 0.016` — **comfortably under the
0.03125 tolerance**, so the check does **not** trigger, and the exact-Beta/BH
test remains "confirmed" and primary. But those same 5 entities' **diagonal**
value (`r_e`, cosine against **their own** anchor) is contaminated by the exact
same anisotropic effect — they would show `r_e≈0.35` and very plausibly clear
BH significance (per §1's table, `r=0.35` already gives `p≈2.1e-3`, easily
significant under BH at `n=107`), contributing false "engaged" declarations
that have nothing to do with the anchoring mechanism.

**Why the aggregate check misses this by construction:** 5/107 hub entities
move the pooled mean by ~13% of the tolerance band — invisible in a sum over
11,342 pairs — while contaminating those 5 entities' own individual
declarations 100%. **This is the identical "aggregate statistic can mask a
minority pattern" blind spot that motivated building §3.7's whole per-entity
readout in the first place (R2 target 4(b))** — now reproduced one level down,
inside the machinery meant to validate the per-entity test's own null. A
design this careful about avoiding that exact failure mode at the outcome
level should not reintroduce it, unexamined, at the null-validation level.

**Fix:** add a per-row (per-entity) diagnostic alongside the pooled check —
e.g., report each entity's own off-diagonal row mean/SD (its 106 mismatched
cosines) and flag entities whose row deviates from the pool by more than some
registered amount, or run a formal heterogeneity test (Levene's test for equal
variance across rows, or simply eyeballing the distribution of row-means for
outliers) before trusting a uniform null for every one of the 107 per-entity
tests. This does not require new data — it is a re-slice of the same
`107×107` matrix already being computed.

---

## 6. MAJOR — `engaged_frac_v3` silently repurposes magnitude-calibrated bands for a significance-rate quantity

`engaged_frac_v3` (§10.3.2's closing line) = fraction of the 107 entities whose
null is **BH-rejected** — i.e., it is literally an **FDR-controlled discovery
rate**, not (as the original §3.7 `a_e ≥ 0.9` fraction was) a **magnitude**
fraction. These are different statistical objects:

- The old `engaged_frac` (§3.7) asked "is this entity's post-blend alignment
  *large* (≥0.9)?" — a fixed-magnitude bar, insensitive to sample size.
- `engaged_frac_v3` asks "is this entity's `r_e` *statistically distinguishable
  from exactly zero*, at a controlled false-discovery rate?" — a question whose
  answer, for **any** non-zero true effect, however tiny, approaches 100% as
  measurement precision increases (here, `n_resamples` up to 32 per entity).
  BH-FDR bounds the *rate of false positives among rejections*; it puts **no
  floor on the number of true rejections** when most of the population carries
  even a small genuine effect — which is exactly the "uniform, moderate,
  incomplete engagement" story §9.7.4 already found in this exact data
  (no bimodal split; every entity's `r_e` sits somewhat above true chance).

Carrying the **same numeric band edges** (`≥90%` headline / `[50%,90%)`
partial / `<50%` not-recruited — pinned in §3.5/§3.7 for the *old*, magnitude-based
quantity) over unchanged to this fundamentally different quantity, without
re-deriving or even re-justifying them, is a substitution the draft never
examines. It is not automatically wrong — but it means the routing table in
§10.6 could produce **Outcome A** ("full mechanistic confirmation, headline")
from a result that means only "we had enough statistical power to detect that
`r_e` is reliably nonzero for most entities," a categorically weaker and less
interesting claim than "the anchoring mechanism majority-engages entities" —
precisely the same species of over-claiming risk `KEYANCHOR_REV6_ATTACK.md` §5
raised about the *old* metric's inability to test anything for a global-λ
mechanism, now potentially reintroduced through the back door of raw
statistical power rather than through metric degeneracy.

**Fix:** register a minimum-effect-size floor (e.g., in `r`-space, alongside
the significance test — "engaged" requires both BH-significance **and**
`r_e ≥ r_min` for some pre-registered, practically-motivated `r_min`, decided
before data exists, exactly as this project's own rank-vs-capacity Hard Rule
insists elsewhere: "rank is only the binding constraint when..."); and/or
re-derive the `90%/50%` band edges specifically for a discovery-rate
interpretation (what discovery rate would a *practically meaningful* level of
engagement imply, given `n=107` tests and this power?) rather than porting the
old bar's numbers unchanged.

---

## 7. The old-data-vs-new-bar arithmetic, computed both the naive way and the registered way

**Naive comparison (what the prompt's framing suggests, and what a single-test
Bonferroni bar would give):** §9.6/§9.7.2's own disclosed back-solve gives
median `r ≈ 0.33–0.37` across the four confirm-wave legs (K32 s0/s1/s2 =
0.350/0.326/0.359; K16 s0 = 0.371). The Bonferroni-style single-test bar this
wave discloses is `r ≥ 0.4137` (Gaussian approx) or `r ≥ 0.4009` (exact Beta,
§1 above). **Either way, the median entity in the already-collected data would
fail this bar** — confirms the prompt's arithmetic is directionally correct.

**But this is not the registered primary criterion — BH-FDR is (§10.3.1),
explicitly not a fixed r-threshold.** I reconstructed plausible 107-entity
distributions consistent with the *only* numbers actually on record for K=32
s0 (`r_e ∈ [0.095, 0.581]`, median ≈0.350, described as "single-peaked, roughly
bell-shaped, largest gaps at the edges," §9.7.4) and ran the **exact registered
procedure** (per-entity exact-Beta p-value, BH step-up at `q=0.05`, `n=107`)
against them, 500 draws each, two plausible shapes:

| reconstructed shape | mean `engaged_frac_v3` (500 draws) | range |
|---|---|---|
| triangular(0.095, 0.350, 0.581) — "bell-shaped, peak at median" | **0.889** | 0.748 – 0.972 |
| uniform(0.095, 0.581) | **0.732** | 0.551 – 0.869 |

**Both plausible reconstructions land the arm mid-to-high in the Outcome A″
band ([50%,90%)), and the bell-shaped reconstruction sits right at the ≥90%
Outcome A edge on average, with individual draws crossing it.** This is the
**opposite** conclusion from "the new bar dooms candidate (d) by construction."
It is only an illustrative estimate (the true 107-value list is not published,
only summary statistics), and I flag it as such — but it is exactly the check
the draft itself should have run and disclosed before the attack round, using
numbers already sitting in its own §9.7.2 table, at zero additional cost (same
category of omission — publish a self-adjudicated preview vs. never check
what your own machinery implies on the one dataset that resembles what's
coming — that got Rev 6 rejected, now manifesting as an *absence* of a check
rather than a *premature* one).

**Answering the prompt's specific question ("is the new bar honest, or does it
doom candidate (d) by construction, and is that disclosed"):** the new bar is
**not obviously honest in either direction, and the draft does not disclose
which way it leans, because it never computed it.** The single-test/Bonferroni
framing (which the draft correctly does NOT register as primary) would
plausibly doom the median entity; the actual registered primary framing (BH)
plausibly does the opposite. Neither the "doomed by construction" reading nor
a "safely neutral" reading is supported without running this check — which is
cheap, CPU-only, and should be a required part of any rescoring gate.

---

## 8. Candidate (d′) — init/parameterization: mostly cleared

`self.anchor_lambda_table = nn.Embedding(vocab_size_total, 1)`, init
`raw=0.0 → sigmoid(0)=0.5` — matches candidate (d)'s own starting λ exactly, so
any divergence is attributable to training, not initial condition (§10.5.1,
verified against the quoted code snippet — consistent with the existing
`anchor_lambda_raw` pattern, `model_rd.py` L822-825 region). Held-out bypass is
specified as identical to §2.2's masked gather/scatter, re-exercised by a fresh
smoke (§10.9 item 9) rather than assumed inherited — good discipline, matches
this project's own "re-run the negative test, don't just cite it" rule.

**One completeness gap in §10.6's outcome map:** the routing table for (d′)
only names two explicit rows ("higher band than (d)" → A(d′); "same band,
not significant" → C′); a genuinely **worse** band than (d) (with or without
significant differentiation) falls into the catch-all "any other combination →
Inconclusive/mixed." This is an acceptable, disclosed catch-all (consistent
with the document's own "no forced binary call" philosophy elsewhere), but it
is coarse — a worse-band-plus-significant-dip-test result would be quite
informative (per-entity capacity making training *less* stable while still
creating a real split) and is currently indistinguishable, in the routing
table, from a merely-ambiguous result. **Minor fix, not blocking:** give this
specific combination its own named row.

---

## 9. Checkpoints (§10.10) — cleared, and a useful contrast with §3's finding

Unlike §10.3.3's threshold-pin, §10.10 correctly builds all three parts: a
**writer** (state_dict saved at final + every admission checkpoint, pulled to
SSD immediately, not deferred to end-of-wave), a **gate** (wave cannot be
marked done until a readout-time assertion confirms every checkpoint exists and
round-trip-loads — refusal, not silent skip), and a **readout** (future audits
read the checkpoint directly). This is the correct application of the Rev 4/R3
"mechanical harness gate, not a stated norm" lesson, and it directly closes the
`m≈1.34` unverifiable-anchor-norm gap `KEYANCHOR_REV6_ATTACK.md` §4 raised (no
checkpoint existed anywhere for either prior wave). **No finding here** — flagged
specifically to make finding 3 (§3 above) sharper by contrast: this section
knows how to build the three-part gate; it just didn't apply the same standard
to its own new pinned-threshold artifact.

---

## 10. `is_done()` / seed-collision safety — cleared, one documentation gap

Read `run_deltanet_rd_exactness_sweep.py::_spec`/`is_done` directly
(L89-127, L626-685): the result filename is built from
`f"w{wave}_rdx_K{K}_arm{arm}_s{seed}..."` — **wave name is bit 0** — and
`is_done()` additionally cross-checks `K`/`seed`/`steps`/every arm-relevant
config field against the result JSON's own `exactness_config`, never trusting
the filename alone (the same discipline `keyanchor_confirm_manifest()`'s own
docstring reasons through explicitly for ITS seed choice, and verifies with a
dedicated smoke, "smoke C" in the confirm-wave chain log: `out_path()
collisions -- vs original keyanchor: False, vs reference arms: False`).
Because Rev 7's seeds (`{10,11,12}` for (d), `{20,21,22}` for (d′)) have never
been used by **any** prior arm at either K, and (d′) is itself a brand-new arm
name, **no filename or `is_done()` collision is possible regardless of which
wave-name string Rev 7 ends up using** — fresh seeds alone are a sufficient
safety net given this naming scheme.

**One real, if minor, gap:** unlike `keyanchor_confirm_manifest()`, which
explicitly reasons through and names its own wave string to guarantee
non-collision (and backs it with "smoke C"), §10 never explicitly registers
the wave-name string its own manifest function will use (only implied, via
§10.10's checkpoint directory name, `wavekeyanchor-mech`). Given how much
precision this document invests elsewhere in naming exact constants
(`LAMBDA_LOG_CADENCE_STEPS`, `ANCHOR_INIT_SEED`, etc.), this is a small,
easy, and consistent-with-house-style fix: **pin the manifest wave-name string
explicitly (e.g. `"keyanchor-mech"`) and the (d′) arm-name string (e.g.
`"dprime"`), and add a "smoke C"-style zero-collision assertion mirroring the
confirm wave's own**, rather than leaving it implicit.

---

## 11. Budget — recomputed from realized per-cell costs

Pulled `wall_s` directly from every archived candidate-(d)/(c)/reference 20k-step
result JSON (`experiment-runs/2026-07-05_keyanchor_{wave,confirm}/.../*.json`):

| cell class | n | realized GPU-h (mean) | range |
|---|---|---|---|
| Wave-1 candidate (d), K=32, `drift_probe=False` | 3 | 0.581 | 0.573–0.585 |
| Wave-1 candidate (c), K=32, `drift_probe=False` | 3 | 0.751 | 0.742–0.766 |
| Confirm-wave candidate (d), K=32, `drift_probe=True` (full instrumentation) | 3 | **0.206** | 0.201–0.208 |
| Reference arms, K=32, `drift_probe=True` | 3 | 0.585 | 0.566–0.596 |
| Gate-1 probe (candidate d, K=32, 5,000 steps, full per-entity + h=1 sweep) | 1 | **0.030** (109.1s) | — |

**Two findings:**

1. **Realized cost for nominally-identical 20k-step cells spans 0.18–0.77
   GPU-h (a >4× range)**, and counter-intuitively, the **more heavily
   instrumented** confirm-wave cells (`drift_probe=True`, full per-entity
   sweep) ran **faster** than the **less** instrumented Wave-1 cells of the
   same nominal config — the opposite of what instrumentation overhead would
   predict. This strongly suggests the dominant driver of realized cost on
   this box is **shared-GPU contention/scheduling variance**, not the
   workload itself. §10.7's `~0.35 GPU-h/run` point estimate for candidate
   (d)'s re-run (with *even more* instrumentation than confirm-wave's own
   `drift_probe`) sits between these two data points but is not clearly
   derived from either, and the registered `×1.3–1.5` "unmeasured-code-path"
   margin is the wrong correction for contention variance (it is sized to
   cover code the estimate didn't count, not scheduler-level noise, which
   could swing the actual cost either cheaper or ~2.5× more expensive than
   assumed, independent of what the new code does). **Fix:** bracket the
   mandatory-baseline estimate using the observed 0.18–0.77 GPU-h range
   rather than a single point estimate; disclose the bracket rather than one
   number.
2. **This does not threaten the ≤12 GPU-h ceiling.** Even the worst
   historically-observed per-cell cost (0.77 GPU-h) applied to all 8
   mandatory Rev-7 cells gives ~6.2 GPU-h, plus the ≤2 GPU-h all-conditionals
   contingencies, still comfortably under 12. The Gate-1 probe estimate
   (`~0.12` GPU-h in §10.7) is, by contrast, **safely conservative** — the
   actual most-recent, most-comparable measurement (an equivalently
   instrumented 5,000-step K=32 probe, §10.9's own precedent) took only
   **0.030 GPU-h**, a ~4× safety margin in the right direction. The
   `≈51.5/80` program-spend figure cross-checks cleanly against `STATE.md`'s
   own independently-recorded `≈51/80` (line 567-568) — no discrepancy found
   there.

---

## 12. Minor — BH's independence/PRDS assumption is unexamined

Standard Benjamini–Hochberg (1995) provably controls FDR at the nominal level
under independent tests or under positive regression dependency (PRDS,
Benjamini–Yekutieli 2001); under arbitrary dependence, the conservative
Benjamini–Yekutieli correction (an extra `Σ 1/i ≈ ln(107)+0.577 ≈ 5.24×`
factor) is required to guarantee the same bound. The 107 per-entity tests here
are **not** obviously independent — they all derive from one trained model,
sharing `W_k`/conv weights across entities, and entities with correlated raw-key
geometry (shared subword structure, frequency effects — the same anisotropy
concern in §5 above) would induce cross-entity p-value dependence of unknown
sign. The design never states or justifies the dependence assumption plain BH
requires. This is a fairly standard omission in practice and not obviously
severe here, but it is exactly the kind of "use the exact structural
assumption, don't borrow one that happens to be convenient" gap this project's
own Hard Rules ask to be checked elsewhere (the Gaussian-vs-exact-Beta
discipline in §10.3.1 is the right instinct; it should be applied one level up,
to the BH-vs-BY choice, too). **Recommend:** either justify PRDS informally
(e.g., cite that the off-diagonal empirical check in §10.3.2 already probes
this) or report both BH and BY discovery counts side by side, the same
"always-reported cross-check, never substituted" pattern §10.3.1 already uses
for Bonferroni vs. the exact test.

---

## 13. Responses to the draft's own 5 pre-answered items (§10.11)

1. **"Attempt #3 — what makes it admissible?"** Partially undercut: the
   "zero-data-dependency, stronger blind" claim (point (1) of the four-part
   defense) is unverified as written (§2/§3 above — the artifact doesn't
   exist, and even once created its mechanical enforcement is incomplete).
   Points (2)/(3)/(4) of the defense (prior JSONs never rescored; metric
   redesigned for the interior-λ regime) stand up under inspection.
2. **"BH vs. Bonferroni / empirical-vs-analytic fallback — same laundering
   risk as z=2 vs. z=3?"** Correct that there's no post-hoc *choice* between
   BH and Bonferroni (BH primary, Bonferroni always-reported) — but §6/§7
   above show the *substitution of a significance-rate quantity for a
   magnitude quantity*, carried through unchanged band edges, is a laundering
   risk of a different, unexamined kind, and §4 shows the mismatched-pair
   fallback rule's own null-pool construction may not equal the primary
   statistic at all.
3. **"Candidate (d′) — new architecture, new bugs?"** Adequately mitigated
   (§8) — one-line indexing change, own smoke suite, independently labeled
   arm. Cleared, with the minor outcome-map gap noted.
4. **"Why trust checkpoint-saving this time?"** Correctly answered and
   independently verified (§9) — this is the one place in the section that
   fully lives up to its own stated standard.
5. **"Does this quietly rescore Outcome C / Rev 6's rejection?"** No —
   verified: §10.4's claim that the prior JSONs lack `r_e`/pairwise-cosine/
   anchor-norm fields entirely (hence cannot be rescored even in principle)
   checks out; no code exists yet that could have retroactively touched them
   (`key_anchoring.py` has no `r_e`/pairwise/REV7 references as of this
   session). Cleared.

---

## 14. Summary table

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | Beta(31.5,31.5) parameterization, `Var(r)=1/64` identity | — | **Confirmed correct**, re-derived independently |
| 2 | `REV7_THRESHOLD_PINNED.json`/`rev7_threshold_derive.py` claimed committed, don't exist | **FATAL** | Verified via `git show`/`git log`/`find` |
| 3 | Threshold-pin mechanical gate incomplete (writer+smoke only, no launcher-refusal/readout-assertion legs) | **MAJOR** | Contrast with §10.10, which gets it right |
| 4 | `C[e,e]` (mean-of-vectors) asserted to equal `r_e` (mean-of-cosines) — unproven, generally false | **MAJOR** | Verified against `key_anchoring.py` L672-693 |
| 5 | Aggregate null-validation check can't catch a minority of anisotropic/hub entities | **MAJOR** | Constructed failure scenario, passes the tolerance check |
| 6 | `engaged_frac_v3` reuses magnitude-calibrated bands for a significance-rate quantity, no effect-size floor | **MAJOR** | Reasoned + simulated (finding 7) |
| 7 | Old-data-vs-new-bar arithmetic: naive Bonferroni fails median entity; registered BH-FDR plausibly does not | reframes prompt's item 2 | Computed both ways (§1, §7); illustrative BH sim: mean 0.73–0.89 engaged_frac_v3 |
| 8 | Exact-Beta vs. Gaussian tail gap unquantified (up to 8× at r=0.58) | MODERATE | Computed (§1) |
| 9 | Candidate (d′) init/parameterization | cleared | minor outcome-map catch-all gap only |
| 10 | Checkpoints (§10.10) | cleared | properly triple-specified |
| 11 | `is_done()`/seed collision | cleared | robust by construction; wave-name string not explicitly pinned (minor) |
| 12 | Budget vs. realized per-cell costs | MINOR-MODERATE | 0.18–0.77 GPU-h range found; ceiling not threatened |
| 13 | BH independence/PRDS assumption unexamined | MINOR | standard gap, low probable severity |

---

## 15. Final verdict: **NEEDS-REV**

Not DEAD — nothing here shows the hypothesis, the `r_e` redesign, or candidate
(d′) is unsalvageable; several parts (checkpoints, the Beta-null derivation,
(d′)'s architecture) are demonstrably solid. Not CLEARED-FOR-BUILD — one FATAL
(a false "already done" claim at the center of the admissibility argument) and
four MAJOR gaps in the new machinery remain. Required before any GPU is spent:

1. **Actually produce `rev7_threshold_derive.py` and
   `REV7_THRESHOLD_PINNED.json`**, run smoke item 5 for real, and correct
   §10.3.3's tense from "is written and committed" to match reality until it
   is true.
2. **Add the launcher-refusal-gate and readout-assertion legs** for the
   threshold pin, matching §3.6's and this section's own §10.10 standard.
3. **Resolve the `C[i,j]`-vs-`r_e` construction mismatch**: pick one
   (mean-of-cosines vs. cosine-of-mean) for both the diagonal and off-diagonal,
   or explicitly quantify and disclose the Jensen's-gap discrepancy the way
   §9.7.10 item 5 did for `a_e`; strengthen smoke item 4 with resample jitter
   so it has power to catch a real inconsistency.
4. **Add a per-entity/row heterogeneity diagnostic** to §10.3.2's null-validation
   decision rule, not just the pooled `(mean,SD)` check, to close the
   hub/anisotropy minority-masking scenario in §5.
5. **Re-derive or re-justify `engaged_frac_v3`'s band edges** for a
   significance-rate quantity, and register a minimum-effect-size floor
   alongside BH-significance so a large `n_resamples` cannot manufacture
   "engagement" from a trivially small true effect.
6. **Run and disclose the sensitivity check in §7** (what does the registered
   BH-FDR procedure produce on data resembling what's already collected)
   before the attack round adjudicates anything else — using the numbers
   already in §9.7.2's own table, at zero additional cost.
7. **Disclose the exact-Beta-vs-Gaussian gap (§1/§8)** quantitatively, not
   just qualitatively.
8. Minor: pin the manifest wave-name and (d′) arm-name strings explicitly and
   add a zero-collision smoke mirroring the confirm wave's own "smoke C";
   re-bracket the budget using the full 0.18–0.77 GPU-h observed range; state
   or justify the BH independence/PRDS assumption.

None of the above requires new GPU spend — all are CPU-only design/code fixes,
consistent with this project's own "audit before running" discipline.
