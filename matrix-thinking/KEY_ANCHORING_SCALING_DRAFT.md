# KEY_ANCHORING_SCALING_DRAFT — §15 candidate text: cliff-location scaling law across d_state

**Status: DESIGN-ONLY DRAFT, zero GPU/CPU spent building anything, pending its
own attack round(s).** This file is NOT part of `KEY_ANCHORING_DESIGN.md` —
per the task brief, it is appended there (as `## 15.`) only after surviving
attack rounds, exactly as §12/§13/§14 each went through DRAFT →
attack-round(s) → REVISED → CLEARED-FOR-BUILD before any cell ran. Every
numbered subsection below is written to be attack-ready: pre-registered
thresholds, exact citations to the source design doc and to the actual code
(file + line), and a self-attack round (§15.17) at the end. All numeric
"power-check" results in §15.8 are REAL — this session ran the existing,
unmodified `sim_cliff_power.py` machinery (not a hypothetical) — but are
disclosed as an illustrative/pre-registration-quality run pending the wave's
own committed, hash-locked re-run before build (mirrors §12.3's own
provenance-hygiene convention).

---

## 15. Cliff-location scaling law across d_state (design-only; DRAFT, pending
its own attack round — zero GPU spent producing this section)

**PI-DECISION flagged up front, exactly as §12's own header required before
anything else.** `STATE.md` (2026-07-07 pre-compaction snapshot, lines 9–32)
and `KEY_ANCHORING_DESIGN.md` §14.13 both declare the KEY_ANCHORING program's
own internal 80 GPU-h sub-ledger **effectively closed**: "the anchoring
program's own effective close — no further wave is registered against this
ceiling" (§14.13), realized spend **78.2270/80 GPU-h, reserve 1.7730/80**.
This draft's own mandatory-only cost (§15.11/§15.12: ≈21 GPU-h at the
pessimistic 2× bracket) **cannot be funded from the existing reserve** —
launching this wave requires BOTH (a) the same kind of explicit PI reopening
decision §12's own header required ("Which next program should I take through
the gauntlet..."), AND (b) a **NEW sub-ledger allocation** (not an extension
of the exhausted 80 GPU-h one), since that ledger's own remaining 1.773 GPU-h
is smaller than even one mandatory cell's own realized cost at this design's
own scale. **This section is CLEARED-FOR-BUILD on technical grounds only,
once it survives attack; it does not self-authorize either the reopening or
the new ceiling.** Proposed new ceiling and the exact numbers behind it: §15.12.

---

### 15.0 Hypothesis and pre-registered outcomes

**One-sentence hypothesis (falsifiable):** if the d=64 cliff is an absolute
state-capacity (load-ratio) effect — i.e. the critical quantity is how loaded
the recurrent state is, `x = K/d_state`, not any fixed absolute `K` and not
(per §14.12/§14.13's own full exoneration) anchor-table coherence — then the
cliff's midpoint `x0`, measured in the SAME `n_entities=107 > d_state`
("organic-coherence") regime that produced the original d=64 measurement,
should be **approximately INVARIANT** as `d_state` grows from 64 to 80 to 96:
`x0(80) ≈ x0(96) ≈ x0(64) = 0.5455`. A **key-geometry (coherence) account**
predicts the same invariance test result for a different, now-EXCLUDED
reason (§14.12/§14.13 already directly exonerated coherence as a sufficient
driver, at BOTH rank-4 and diffuse structures, up to and beyond d=64's own
realized coherence band) — this wave is a **constructive cross-check**, not
a re-test, of that exoneration: it uses the ORIGINAL, organically-coherent
(non-frozen, non-dosed, learned) table construction, not §14's synthetic
frozen/dosed one, so a result here speaks to whether NATURALLY-ARISING
coherence (as opposed to injected/frozen coherence) behaves differently — see
§15.1 for why this is a genuinely new comparison, not a repeat. A **fixed-K
account** (the cliff occurs at a roughly constant ABSOLUTE `K`, independent of
`d_state`) predicts **DRIFT**: in ratio terms, `x0 = K_crit/d` would shrink as
`d` grows. §13.10/§14.13's own "absolute state capacity" survivor candidate
(`d_state` itself, or `d_state − K` slack, not any ratio) is the account this
wave's own invariance-in-ratio-terms framing is testing FOR — if `x0` drifts
UPWARD with `d` even within this narrow `d∈{64,80,96}` window (before the
regime-change at d=128 where `n_entities < d_state`), that is itself evidence
against a clean ratio-invariant story and narrows the surviving-candidate
space further (§15.0's own outcome 2/3 below).

**Why this is well-posed and not already answered by §13.10.** §13.10 already
showed d=128 (where `n_entities=107 < d_state=128`, table exactly orthogonal
at init) is FLAT at h4=1.0 across the same K/d window that cliffs at d=64 —
but that comparison confounds THREE things at once (state size, optimization
dynamics, and table coherence regime, §13.10's own disclosure) and, per
§14.12/§14.13, the coherence axis specifically has now been ruled out as the
explanation even when isolated. **What has never been measured: whether the
cliff's ratio location holds steady across d_state values WITHIN the SAME
coherence regime as d=64** (`n_entities=107 > d_state`, forced non-zero
coherence by the Welch bound, organically arising from learned-table SGD, not
injected) — d=80 and d=96 are the only two `_SAFE_D_STATE`-feasible-pending-
verification (§15.2) points strictly between 64 and the d=128 regime change,
and both satisfy `107 > d_state` (§15.4), preserving the SAME regime. This is
the constructive complement to §14's exoneration: §14 asked "does INJECTING
coherence into an at-init-ORTHOGONAL table reproduce the cliff" (no); this
wave asks "does the cliff's RATIO location stay put as `d` grows while the
table remains ORGANICALLY NON-ORTHOGONAL the whole way" — a different
question neither §13 nor §14 measured.

**Pre-registered outcomes, exact numeric criteria, stated before any d=80/96
cell runs (mirrors §12.0/§13.0's "no third ambiguous bucket left undefined"
discipline):**

1. **SCALING-LAW CONFIRMED:** neither new-d fit is degenerate (§15.10 item 1's
   reused-verbatim rule) AND both `x0(80)` and `x0(96)`'s own 95% bootstrap CIs
   overlap the pre-registered invariance band (§15.10, centered at 0.5455).
2. **SCALING-LAW REFUTED:** neither fit is degenerate AND at least one of
   `x0(80)`/`x0(96)`'s CI excludes the band. (A sub-case worth naming even
   though it is still formally REFUTED: if exactly ONE of the two excludes
   the band while the other overlaps, this is flagged as **DIRECTIONAL DRIFT,
   NOT YET SATURATED** — invariance holding near d=64 but breaking down by the
   farther new-d point — a qualitatively different, more informative REFUTED
   than a clean two-for-two exclusion.)
3. **AMBIGUOUS (data-quality, not a judgment call):** the REAL degenerate
   fraction (§12.4 item 3's own definition, reused verbatim) exceeds 10% at
   EITHER new-d fit — this is a reliability question about that fit
   specifically, evaluated BEFORE the overlap test is even applied (§15.10
   item 2). Pre-registered follow-up: +2 seeds at the affected d (the
   already-reserved seed-contingency block, §15.3), re-fit, re-check.

**Secondary, pre-registered, NOT load-bearing for the primary call (per the
task's own request and §12.4's "linear form ... disclosed secondary, never
load-bearing" precedent): does the cliff WIDTH `w` scale with `d`?** Report
`w(64)=0.0597` `[0.0557,0.0642]` (§12.9, cited verbatim) alongside `w(80)` and
`w(96)`'s own CIs, side by side, purely descriptively — no CONFIRM/REFUTE bar
attached (§15.10.1).

---

### 15.1 Relationship to §12/§13/§14 — what this wave does and does not reopen

Per §11.7/§12.1's own precedent ("this wave never reopens or rescores any of
the [prior] JSONs"): this wave does not touch, revisit, or rescore any
Outcome assignment in §9/§10/§11/§12/§13/§14. It adds two NEW `d_state`
points (80, 96) to the SAME capacity-cliff research question, using
candidate (d)'s own unchanged architecture (§12.1's own "single learned
scalar λ, frame-potential init at `ANCHOR_INIT_SEED=20260705`" — verified
this session to still be the correct construction call, §15.9).

**The three-way comparison this wave completes, stated explicitly (a genuine
gap, not redundant with any prior wave):**

| Wave | d_state | Table construction | `n_entities` vs `d_state` | Result |
|---|---|---|---|---|
| §12 (cliff) | 64 | organic (learned, frame-potential init) | 107 > 64 (non-orthogonal, Welch-forced) | Cliff LOCATED, x0=0.5455 |
| §13 (dstate) | 128 | organic (learned, frame-potential init) | 107 < 128 (exactly orthogonal at init) | NO cliff, flat h4=1.0 |
| §14 (dose) | 128 (fixed) | FROZEN, synthetically dosed (rank-4 AND diffuse) | 107 < 128 (base), coherence injected on top | NO cliff at ANY dose up to 0.40 — coherence EXONERATED |
| **§15 (this draft)** | **80, 96 (NEW)** | **organic (learned, frame-potential init) — SAME as §12/§13** | **107 > 80, 107 > 96 (non-orthogonal, Welch-forced — SAME regime as §12/d=64)** | **untested** |

No prior wave has measured an intermediate `d_state` while HOLDING the
`n_entities > d_state` (organic-coherence) regime fixed. §13 changed BOTH
`d_state` AND the coherence regime in one jump (64→128 crosses `n=107`);
§14 held `d_state=128` fixed (the OTHER regime) and manipulated coherence
directly, but only ever in the FROZEN/synthetic form. This wave is the
missing cell: same regime as §12, new `d_state`.

**What licenses reusing candidate (d)'s exact architecture, unchanged
(§12.1's own reasoning restated):** §10.14's CONFIRMED-BY-ABLATION verdict
(constancy alone suffices, learned table content contributes nothing
measurable) and §14.12/§14.13's full coherence exoneration together mean the
open question left standing is NOT about the anchor mechanism's own
correctness — it is about capacity scaling. No new candidate arm is proposed
here; this is a pure `d_state`/`K` extension of the same candidate (d)
measurement, exactly as §12 and §13 each were.

---

### 15.2 Kernel-safety gate — d=80/96 feasibility is NOT yet verified (a NEW
risk, absent from §12/§13, which only ever used pre-verified d values)

**This is the single most important NEW finding of this draft: unlike every
prior wave, d=80 and d=96 have never been run through this harness's kernel
at all — §13.1's own "d=32 infeasibility, full verification, not assumption"
discipline applies here with the opposite starting condition (unknown,
not known-crashing) and MUST be run before anything else in this design.**

**Verified directly this session, `model_rd.py` lines 95–118:**

```python
_MIN_KERNEL_T = 128
_SAFE_D_STATE = (64, 128)
```
with the hard `assert d_state in _SAFE_D_STATE` at `DeltaNetRDBlock.__init__`
(lines 735–737). The comment block (lines 95–104) documents the ORIGINAL
measurement that produced this allowlist: a fresh-process-per-probe sweep of
`chunk_delta_rule`'s backward, fla 0.5.1 / torch 2.12.1+cu130 / triton 3.7.1,
F15-LM checkpoint rounds 1–2 (2026-07-02), testing `D ∈ {16, 32, 64, 128}` at
`T ∈ {128, 224, 448}`: **D=16 crashed at T=224 (flaky at T=128); D=32 crashed
at T=128 AND T=224; D=64 and D=128 were clean at every T tested. D=80 and
D=96 were NEVER TESTED — they are not "known safe," they are simply
untested, structurally identical in risk-status to how D=32 looked before
its OWN measurement (which then found it crashes).**

**Additionally verified this session, `run_deltanet_rd.py` line 1278:**
```python
ap.add_argument("--d-state", type=int, default=64, choices=[64, 128], ...)
```
The CLI itself rejects `--d-state 80`/`96` with an argparse error BEFORE
`model_rd.py`'s own assert is ever reached — a second, independent gate that
also needs extending, strictly AFTER (not concurrently with) the kernel
measurement below.

**Required Wave −1 step, registered here as MANDATORY and BLOCKING — nothing
else in this draft can build until this passes:**

1. **Re-run the EXACT F15 measurement protocol** (`f15_lm_checkpoint.py`'s own
   `check_short_t_sweep`/kernel-safety harness, lines 358–491 and the
   fresh-subprocess-per-probe pattern at line ~454, "so a crash cannot poison
   this process's CUDA context") at `D ∈ {80, 96}`, `T ∈ {128, 224, 448}`,
   BOTH forward and backward passes of `chunk_delta_rule` called directly
   (not through the stock `fla.layers.delta_net.DeltaNet` auto-fallback,
   matching how this harness actually calls the kernel, §13's own
   `_MIN_KERNEL_T` padding rationale). **PASS criterion: zero CUDA illegal
   memory access / Triton autotuner crashes at every (D, T) cell, both
   passes, on THIS box's exact build** (fla/torch/triton versions must match
   or be re-verified, since a dependency upgrade since 2026-07-02 could
   itself change the crash matrix — check versions before assuming the
   original matrix still applies even at D=64/128).
2. **Mechanical per-d drop rule, mirroring §13.1's own verdict exactly:** if
   EITHER D=80 or D=96 crashes at any tested T, that `d_state` is DROPPED
   from this wave's scope (not retried, not "probably fine at production
   T" — the measured failure is definitive per §13.1's own precedent), the
   wave's own grid/budget is recomputed for whatever subset survives
   (possibly a single-d wave, degrading the invariance test to "is x0(80)
   [or 96] alone consistent with 0.5455," a materially weaker but still
   reportable result), and this is disclosed plainly, not smoothed over.
3. **Only after BOTH new d's pass (or the surviving subset is settled):**
   extend `_SAFE_D_STATE = (64, 80, 96, 128)` in `model_rd.py` and
   `choices=[64, 80, 96, 128]` in `run_deltanet_rd.py`'s `--d-state`
   argparse (both edits gated on step 1's own PASS, never made speculatively
   ahead of the measurement).

**This gate is NOT hypothetical box-standard due diligence — it is
structurally the same category of risk that already once produced a FAIL
(D=32, D=16) in this exact codebase, on this exact kernel path, at nearby
head-dimension values.** Treating d=80/96 as "obviously fine because they're
between two verified-safe values" would be an unverified analogy of exactly
the kind this project's Hard Rules prohibit ("verify, don't assume by
analogy," §12.2.2/§13.2's own repeated discipline) — head-dimension kernel
crashes in this measured matrix were NOT monotonic in D (D=64 safe, D=32
unsafe, is already a non-monotonic boundary), so interpolating safety between
64 and 128 is not licensed by the existing data.

---

### 15.3 K-grid construction — translating §12's grid to d=80 and d=96

**Reused verbatim from §12.2 (Rev 12.1): the four ratios `{0.53125, 0.59375,
0.65625, 0.71875}`** — the same re-picked, power-simulation-justified point
set (§12.4a) that produced the actual d=64 measurement (§12.9). Per the task's
own instruction, these ratios are re-instantiated at the new `d_state` values
by direct multiplication:

| d_state | ratio × d | Result |
|---|---|---|
| 64 (reference) | 34.0, 38.0, 42.0, 46.0 | integers (as built) |
| 96 | 51.0, 57.0, 63.0, 69.0 | **exact integers** — 96 = 64×1.5, and 34×1.5=51, 38×1.5=57, 42×1.5=63, 46×1.5=69, all exact (verified by direct multiplication this session) — mirrors §13.2's own "exact 2× scaling" finding for d=128, here at 1.5× |
| 80 | 42.5, 47.5, 52.5, 57.5 | **non-integer** — 80 = 64×1.25, and 1.25× a set containing both even and odd members does not stay integer |

**d=96 grid: `K ∈ {51, 57, 63, 69}`, ratios IDENTICAL to d=64's own grid to
machine precision** (51/96 = 0.53125 = 34/64, etc., verified by direct
division). No rounding, no disclosed deviation — the cleanest possible
translation.

**d=80 grid — mechanical, disclosed rounding rule (mandatory since all four
targets land on a half-integer):** round half up, uniformly, at all four
points (a single, stated convention, not a per-point judgment call):
`K ∈ {43, 48, 53, 58}`. Achieved ratios: `43/80=0.53750`, `48/80=0.60000`,
`53/80=0.66250`, `58/80=0.72500` — a **uniform +0.00625 offset** from the
nominal targets (since all four half-integers rounded the same direction),
disclosed exactly, not hidden. The grid retains the SAME exact spacing
(`80 × 0.0625 = 5`, an exact integer step) as the original — only the
starting phase shifts, not the resolution.

**Predicted cliff K at each new d (point estimate only, from `x0=0.5455`,
not a claim, since the whole point of the wave is to test whether this
transfers):** `0.5455 × 80 = 43.64`; `0.5455 × 96 = 52.368`. Both predicted
values fall almost exactly ON the low end of each grid (`K=43` at d=80 is
0.36 away from the raw predicted K=43.64; `K=51` at d=96 is 1.37 away from
the raw predicted K=52.368) — the translated grid brackets the predicted
transition tightly, by construction (since it is the SAME ratio-spaced grid
that bracketed the actual x0=0.5455 at d=64, and the ratios are preserved or
near-preserved).

**A MANDATORY low-K anchor point is added at each new d — NOT §12's own
grid, a NEW addition this draft registers and justifies with a real,
already-run simulation (§15.8), not by analogy:** `K=20` at d=80 (ratio
exactly 0.25, mirroring K=16's own role at d=64) and `K=24` at d=96 (ratio
exactly 0.25). See §15.8 for why this is mandatory rather than optional.

---

### 15.4 Entity-pool and single-K-cycle verification

**Pool size, verified this session (`grammar_rd.py::build_entity_pools`,
lines 194–226):** `n_heldout = round(len(names_v) * 0.5)`; with 213 verified
names, Python's banker's rounding gives `round(106.5) = 106` (confirmed by
direct execution this session: `round(213*0.5) == 106`), so **train pool =
107, heldout pool = 106** — independent of `d_state`/`K` entirely (the
function signature takes no `d_state` argument, confirmed by grep, same
verification §13.1 already performed for the d=32 check).

**Margin check, both new d's, both pools (mirrors §13.2 item 2's own
arithmetic, "worth stating plainly, not just a lookup"):**

| d_state | max K in grid (incl. anchor) | margin vs train pool (107) | margin vs heldout pool (106) |
|---|---|---|---|
| 80 | 58 | 49 | 48 |
| 96 | 69 | 38 | 37 |
| (128, for comparison) | 92 | 15 | 14 |

Both new d's have MORE margin than d=128's own K=92 (which itself already
passed this check, §13.2 item 2) — pool feasibility is not a binding
constraint at either new d. **Note, precisely, why d=128 showed no cliff —
NOT a pool-capacity story:** the reason d=128's own K-grid (up to K=92) never
reached the predicted cliff was NOT that K=92 exceeded the pool (it did not —
92 < 107 with margin, same as this wave's own K's) — it was that
`n_entities=107 < d_state=128`, making the anchor table EXACTLY orthogonal
and removing the (already-exonerated-as-sufficient, but still the only
naturally-occurring source of) coherence entirely (§13.10's own disclosed
comparison axis). **d=80 and d=96 both keep `107 > d_state`** (verified:
107 > 80, 107 > 96), so the table CANNOT be exactly orthogonal at either new
d — this is the SAME regime as d=64, which is exactly what makes the
`x0(80)`/`x0(96)` vs `x0(64)` comparison clean (§15.1's own table) — it is
not confounded by a regime change the way the d=64→128 jump was.

**Single-K-cycle construction, verified this session (`grammar_rd.py` lines
262–274, `_permutation_graph`/`_iterate_permutation`):** builds a random
SINGLE Hamiltonian K-cycle per row (not a general permutation, closing the
cycle-length-periodicity confound per the project's own [LEARN] rule on
permutation-based hop-depth tasks) — fully parameterized by `K` alone, no
`d_state` dependence anywhere in the function. Remains valid, unmodified, at
every new K value in this design (`20, 24, 43, 48, 51, 53, 57, 58, 63, 69`) —
no build task, no re-verification needed beyond this citation.

---

### 15.5 Per-d threshold-pin derivation

**Already fully generalized — NO build task, verified this session,
`rev7_threshold_derive.py` lines 285–311:** `main()`'s own `--d-state` CLI
flag (added at §13.2's own Rev 13.1 fix) takes any int with no `choices`
restriction, threading straight to `derive(d_state=args.d_state)`, which is
`N_ENTITIES=107` (module constant, unaffected) and `d_state` (free argument)
only.

**Pin naming convention, verified this session (`run_deltanet_rd.py` lines
1669–1670):**
```python
_pin_name = ("REV7_THRESHOLD_PINNED.json" if args.d_state == 64
             else f"REV7_THRESHOLD_PINNED_D{args.d_state}.json")
```
so the two new pins are, mechanically, `REV7_THRESHOLD_PINNED_D80.json` and
`REV7_THRESHOLD_PINNED_D96.json` — matching the existing `_D128` convention
exactly, no new code path.

**Exact commands (mirrors §13.2's own d=128 derivation invocation):**
```
python matrix-thinking/deltanet_rd/rev7_threshold_derive.py \
    --d-state 80 --out matrix-thinking/deltanet_rd/REV7_THRESHOLD_PINNED_D80.json
python matrix-thinking/deltanet_rd/rev7_threshold_derive.py \
    --d-state 96 --out matrix-thinking/deltanet_rd/REV7_THRESHOLD_PINNED_D96.json
```

**Derived quantities, computed this session (mirrors §13.2's own by-hand
verification table):**

| d_state | `sigma_chance = 1/sqrt(d)` | `a_shape = (d-1)/2` | `r_min_partial = 2·sigma_chance` | `r_min_headline` |
|---|---|---|---|---|
| 64 | 0.125000 | 31.5 | 0.25000 | 0.35 (registered literal) |
| 80 | 0.111803 | 39.5 | 0.22361 | 0.35 (unchanged) |
| 96 | 0.102062 | 47.5 | 0.20412 | 0.35 (unchanged) |
| 128 | 0.088388 | 63.5 | 0.17678 | 0.35 (unchanged) |

**Required pre-launch check, mechanical, PASS condition is asymmetric
(mirrors §13.2's own "byte-identical would be a FATAL sign" logic):** confirm
each new pin's `derived.inputs.d_state` matches (80 or 96), and that
`sigma_chance`/`r_min_partial`/Beta-shape fields DIFFER from every other
existing pin (64, 128) while `r_min_headline` matches 0.35 exactly across
ALL FOUR pins (the one cross-d invariant, a REGISTERED LITERAL not a function
of `d_state`, §13.2's own disclosure) — a byte-identical `derived` block
against any other pin, or a `r_min_headline` that differs, is a FATAL,
blocking launch.

---

### 15.6 `GATE2_N_ITER_BY_K` — the K=48 collision risk and its fix (a REAL
build-scope finding, not previously flagged in §12/§13 since neither reused
a K value across different d's)

**Verified this session, `key_anchoring.py` lines 65–67:**
```python
GATE2_N_ITER_BY_K = {16: 12, 32: 20, 48: 20,
                     34: 20, 38: 20, 42: 20, 46: 20,
                     68: 20, 76: 20, 84: 20, 92: 20}
```
**This dict is keyed by `K` ALONE — it has no `d_state` axis.** Every prior
extension (§12.2.1 for K=34–46, §13.2 item 3 for K=68–92) added K values that
had never appeared before, so this was never previously a live risk. **This
wave's own d=80 grid coincidentally reuses `K=48`** (§15.3: `{43, 48, 53,
58}`) — **an EXACT collision with the existing d=64 entry.** If the build
naively does `GATE2_N_ITER_BY_K[K]` at d=80 the way every prior wave did at
its own new K's, `K=48` at `d_state=80` would SILENTLY inherit the d=64-only
n_iter=20 verification — a value that has NEVER been checked for
convergence at `d_state=80` specifically, only at `d_state=64`. Per
`key_anchoring.py`'s own comment (lines 82–104), the Newton-Schulz
convergence GEOMETRY is disclosed as regime-dependent (the Welch floor's
sign flips at `n=d`, and its MAGNITUDE differs continuously with `d` even
within the same `n>d` regime) — reusing a value verified at a DIFFERENT
`d_state` without a fresh check is exactly the "by-analogy, not verified"
gap §12.2.2/§13.2 item 3 both flagged and closed for their own new K/d
combinations.

**Required build task, registered here as MANDATORY (not by-analogy):**

1. **Do NOT silently extend the existing flat dict.** Either (a) restructure
   `GATE2_N_ITER_BY_K` to be keyed by `(K, d_state)` tuples (the structurally
   correct fix, closing this collision class permanently for any future
   wave), or (b) — a smaller, scoped fix if restructuring is judged too
   invasive for this wave alone — add a NEW, `d_state`-namespaced dict/lookup
   (`GATE2_N_ITER_BY_K_D80`, `_D96`, mirroring the `_D{d_state}` pin-file
   convention already established) that the d=80/96 code paths consult
   INSTEAD OF the flat dict, never falling through to it.
2. **New keys required regardless of (1)'s resolution:** `{20: 20, 43: 20,
   48: 20, 53: 20, 58: 20}` at d=80 and `{24: 20, 51: 20, 57: 20, 63: 20,
   69: 20}` at d=96 — all at the `n_iter=20` production tier BY ANALOGY
   ONLY (matching every prior wave's own initial choice), pending the
   mandatory Wave −1 sufficiency check below.
3. **Wave −1 n_iter-sufficiency check, mandatory, mirrors
   `keyanchor_cliff_niter_check.py`/`keyanchor_dstate_niter_check.py`
   exactly:** for EVERY one of the 10 new (K, d) pairs above — **including
   `(K=48, d=80)` specifically, even though `K=48` "already has an entry"** —
   compute the pooled pairwise post-Newton-Schulz cosine at
   `n_iter ∈ {12, 16, 20, 24}` and confirm convergence (rel_change 20→24
   below the established <0.5% tolerance) before trusting `n_iter=20`. This
   is the exact discipline that already caught nothing wrong at K=34–46/
   d=64 and at K=68–92/d=128 — it must be RUN, not skipped because "K=48
   already passed once" (it passed once, at a different `d_state`, which is
   precisely the geometry this check exists to verify is NOT silently
   assumed to transfer).
4. **Negative unit test, registered (mirrors §12.2.1's own second Wave −1
   smoke, "assert the fitting script's input list contains ZERO reference-
   arm paths"):** assert that `(K=48, d_state=80)`'s OWN n_iter-sufficiency
   check ran and produced its OWN result artifact, distinct from d=64's
   K=48 check — i.e., a build regression that silently reused the d=64
   result (by key collision) must be mechanically catchable, not merely
   "shouldn't happen."

---

### 15.7 The λ=1 ceiling at the new (K, d) pairs

Per §11.4.2/§12.2.2/§13.2.1's own method (fix one sampled anchor row,
resample K−1 co-drawn rows from the other 106/105, full-pool production-tier
Newton-Schulz, pooled pairwise cosine, same registered frame-potential
table/seed): **required pre-launch action, NOT yet performed in this
drafting session, at all 10 new (K, d) pairs** (§15.3/§15.6's own K-grids).
Per §13.2.1's own disclosed caveat (no monotone-sequence expectation is
stated as confidently as §12.2.2's, since this crosses a genuinely new `d`
each time): **stated expectation only** — since the Welch floor DECREASES
monotonically as `d` increases within the `n>d` regime (verified this
session, §15.9's own table: 0.0796 at d=64 → 0.0564 at d=80 → 0.0329 at
d=96), the ceiling (which tracks the achievable ORTHOGONALITY, hence inversely
tracks the achieved coherence) is expected to INCREASE monotonically
64→80→96 — a plausibility argument from the Welch-floor monotonicity, not a
proven consequence, computed and checked (not assumed) before Gate 2 is
trusted at either new d.

---

### 15.8 The anchor-point gap — `fit_cliff_curve.py`'s `ANCHORED_D_STATES`
and the power-check that makes the low-K anchor point MANDATORY (not merely
optional)

**Verified this session, `fit_cliff_curve.py` line 85:**
```python
ANCHORED_D_STATES = (64,)
```
**Only `d_state=64` gets the free `K=16` (fixed anchor) + `K=32`/`K=48`
(archived, resampled) flanking points folded into its own fit** (lines
280–327: at any `d_state` NOT in `ANCHORED_D_STATES`, `--k-grid` is REQUIRED
and used AS-IS with no anchors — verified directly, `k32_dir`/`k48_dir` must
be OMITTED and raise an assertion error if passed). **This means §12.9's own
"6-point" (in the doc's own count) / actually 7-row curve (K=16,32,34,38,
42,46,48) benefited from THREE flanking points the new d=80/96 fits will
NOT get for free** — d=80 and d=96 have no pre-existing archive at any K,
unlike d=64 which inherited K=16/32/48 from EARLIER waves in the same
program. This is a real, load-bearing asymmetry that a naive "just translate
§12's 4-point grid" reuse would silently under-power relative to d=64's own
CI.

**This session's own power-check, run using the EXISTING, unmodified
`sim_cliff_power.py` machinery — `bootstrap_ci_width(new_ks, n_seeds_new=3,
truth, n_trials=4000, include_anchors=False, d_state=80|96, seed=20260706)`
— `include_anchors=False` and `d_state=<int>` are BOTH pre-existing
parameters of this function (added at §13.3 item 8 for the d=128 power
check, never previously exercised at d=80/96), so this required ZERO code
changes, only a new driver call:**

| Config | mean CI(x0) width | max (worst-truth) CI(x0) width |
|---|---|---|
| d=64, anchored, 4pt (reference — reproduces §12.4a to within RNG-stream drift: published 0.0247/0.0684-ish) | 0.0246 | 0.0684 (gradual_w15) |
| **d=80, NO anchor, bare 4pt `{43,48,53,58}`** | **0.0742** | **0.1942** (gradual_w15) |
| **d=96, NO anchor, bare 4pt `{51,57,63,69}`** | **0.0726** | **0.1910** (gradual_w15) |
| d=80, WITH low-K anchor, 5pt `{20,43,48,53,58}` | 0.0382 | 0.1295 (gradual_w15) |
| d=96, WITH low-K anchor, 5pt `{24,51,57,63,69}` | 0.0381 | 0.1291 (gradual_w15) |

**Reading, mechanical, not a judgment call:** the bare 4-point (no-anchor)
fit's mean CI width (~0.074) is **≈3× wider** than d=64's own anchored fit
(~0.025), and its worst-case (~0.19) approaches the FULL (0.5,0.75)
sub-bracket span (0.25) — several of the 8 simulated truths (sharp_w05,
moderate_w08, gradual_w15, offcenter_x055_w08) already exceed a 0.05 CI-width
trigger (comparable to d=64's own measured cliff width `w=0.0597` — a CI this
wide could not distinguish "exactly at x0=0.5455" from "shifted by a whole
cliff-width"). **Adding ONE low-K anchor point per new d (K=20 at d=80, K=24
at d=96 — ratio 0.25 exactly, mirroring K=16's own role at d=64) roughly
HALVES both the mean and worst-case CI width at negligible extra cost (3
seeds × 2 d's = 6 cells, §15.12).** Given this measured, not projected,
improvement: **the low-K anchor point is registered as MANDATORY, part of
the primary grid (§15.3), not a conditional/cut-eligible escalation** —
this is a stronger, evidence-backed design decision than the conditional
mechanism this draft originally considered, made possible by running the
already-existing power-check machinery rather than assuming its outcome.

**Disclosed residual gap, honestly, not smoothed over:** even WITH the added
anchor point, the new-d fits' worst-case CI width (~0.13) remains roughly
**2× wider** than d=64's own anchored fit (~0.068) — one added low-K point
does not fully equalize to d=64's THREE flanking points (K=16 fixed + K=32/
K=48 archived-and-resampled). This residual gap is priced into the
invariance-band construction (§15.10), not ignored.

**Required Wave −1 step before build (mirrors §12.3's own provenance-hygiene
re-run discipline):** re-run this EXACT power-check as a committed,
hash-locked artifact (`sim_cliff_power_results_scaling.json` or a
purpose-built driver script importing `sim_cliff_power.py`'s own functions
unmodified) at the wave's own registered seed and `n_trials≥4000`, before any
GPU cell launches — this session's numbers above are illustrative/
pre-registration-quality (real, reproducible, but not yet the wave's own
committed artifact).

---

### 15.9 Configs and arms — candidate (d) only, organic (non-frozen) table,
no reference arm

**Architecture: byte-identical to §12/§13's own candidate (d)** —
`anchor_active=True`, `anchor_lambda_mode="learned"` (single learned scalar
λ), `anchor_table_init_mode="frame_potential"`, `ANCHOR_INIT_SEED=20260705`,
`drift_probe=True`, `rev7_engagement=True`. **This wave does NOT use §14's
`anchor_table_frozen=True`/`anchor_table_init_mode="dosed"` path** — the
table is learned (organic), not frozen or synthetically dosed, matching §12/
§13 exactly and deliberately NOT matching §14 (§15.0/§15.1 explain why this
distinction is the point).

**Natural (organic, un-forced) table coherence at construction, VERIFIED
THIS SESSION by actually running `key_anchoring.frame_potential_init(n=107,
d=d, seed=ANCHOR_INIT_SEED)` (not estimated):**

| (n, d) | Welch floor | realized `max\|cos\|` at `ANCHOR_INIT_SEED=20260705` | `sigma_ratio` |
|---|---|---|---|
| 107, 64 | 0.079614 | 0.284151 (§12/§14, cited) | 1.000000 |
| 107, 80 | 0.056427 | **0.200148** (verified this session) | 1.000000 |
| 107, 96 | 0.032878 | **0.097811** (verified this session) | 1.000000 |
| 107, 128 | 0.000000 | 0.000000 (§13.10, cited) | 1.000000 |

The tight-frame property (`sigma_ratio=1.0`) holds at every point (Gate-2's
own G2-a leg would pass trivially, `sigma_ratio_min=0.1`). Natural coherence
DECREASES monotonically as `d` grows within the `n>d` regime, exactly
tracking the Welch floor's own monotonic decrease — this is expected and
disclosed, not a new finding, but now VERIFIED rather than assumed at the
two new d's specifically. **This table is worth noting for interpretation,
not for the primary decision rule:** if the cliff DOES reappear at d=80/96,
it does so at a LOWER organic coherence than d=64's own 0.284 — but since
§14.12/§14.13 already showed coherence (even injected FROZEN, up to 0.40, well
above ANY of these organic values) does not drive the cliff, this comparison
is context, not a live alternative hypothesis being tested by this wave.

**No reference (bare geo3) arm — same disclosed cut as §12.2 item 2/§13.2,
same justification:** this wave's fit reads candidate-(d) h4 values only;
cutting the reference arm loses per-K admissibility-floor context at these
new d's (whether bare geo3's own value-salvage floor also fails here), judged
an acceptable, disclosed, cheap-to-backfill-later cut against this wave's own
tight budget (§15.12), exactly mirroring the prior waves' own reasoning.

**No d′ (per-entity λ) or fixed-λ=1 probe** — same §12.2 item 4 reasoning:
the "hard question" is already answered (§11.12), and the ceiling is cheap to
recompute analytically per (K,d) (§15.7).

**h1 sanity guard (inherited, unchanged, §11.2/§12.4 item 6): h=1 ≥ 0.98 per
(K,d) cell**, checked against candidate (d)'s own `M2_in_distribution['1']`
field — self-contained, no reference-arm comparator needed.

**Degenerate-run exclusion, reused VERBATIM from the `geo3_admission` stack
(§12.6/§14.12's own convention):** every cell must show `admissible: true`,
`ns_converged_no_fallback: true`, `n_geo3_fallback_train_steps: 0`,
`checkpoint_fallback_seen: false`, `finite_loss_no_divergence: true`,
`task_performance_floor_pass: true` — no new fields, no new logic, the same
instrument every prior wave's own admissibility check used.

---

### 15.10 Pre-registered success criteria

**Primary (scaling-law invariance), mechanical, computed identically at each
new d via `fit_cliff_curve.py --d-state {80|96} --k-grid <this wave's own
5-point grid, incl. the mandatory anchor> --n-trials >=4000` (no `--k32-dir`/
`--k48-dir`, since neither 80 nor 96 is in `ANCHORED_D_STATES`, §15.8):**

1. **Fit form, bounds, and bootstrap — reused VERBATIM from §12.4 (Rev
   12.2)**: `h4(x) = L/(1+exp((x-x0)/w))`, `scipy.optimize.curve_fit`, bounds
   `L∈[0.5,1.2]`, `x0∈[0.3,0.9]`, `w∈[0.005,0.5]`; seed-level parametric
   bootstrap, `N_BOOT≥2000` (this wave registers `N_BOOT=4000` to match
   §12.4a's own Rev 12.2 convention), 95% percentile interval. **Reused
   VERBATIM degenerate-fit definition (§12.4 item 3):** a bootstrap replicate
   is degenerate if `curve_fit` fails to converge, OR the fitted `w`/`x0`
   lands within 1% of either bound — counted and reported as a fraction.
2. **Data-quality gate, evaluated FIRST, before any overlap test:** if the
   REAL degenerate fraction at `d=80`'s fit OR `d=96`'s fit exceeds 10%
   (§12.4 item 3's own threshold) → **outcome = AMBIGUOUS** for that d
   (§15.0 outcome 3), reliability caveat attached, pre-registered follow-up
   = fire that d's own reserved +2-seed contingency block (§15.12), re-fit,
   re-check. This is checked independently per d — one d can be AMBIGUOUS
   while the other is cleanly CONFIRMED/REFUTED.
3. **Invariance band, pre-registered, mechanical construction (justified
   from d=64's own measured CI half-width plus the projected new-d CI
   half-width, per the task's own instruction):**
   `m = δ64 + δ_proj`, where `δ64 = 0.0127/2 = 0.00635` (half of §12.9's own
   realized `x0` CI width) and `δ_proj` is the PESSIMISTIC-CORNER projected
   new-d half-width from §15.8's own power-check (the WITH-anchor, 5-point
   config's worst-simulated-truth row, `gradual_w15`): mean of d=80's 0.1295
   and d=96's 0.1291 = 0.1293, half-width **0.0647**. **`m ≈ 0.0710`.**
   **Band = `[0.5455 − 0.0710, 0.5455 + 0.0710] = [0.4745, 0.6165]`, width
   ≈0.142.** This clears the §12.4b-style honesty bar (narrower than the
   (0.5,0.75) 0.25-wide sub-bracket, by a ≈43% margin: `(0.25-0.142)/0.25`)
   — the test is not vacuous even under the pessimistic-corner assumption.
   **Context, not the frozen rule:** using the MEAN (not worst-case)
   with-anchor CI half-width instead (`≈0.019`) would give a much tighter
   band, `[0.520, 0.571]` (width ≈0.051) — disclosed as the "expected,
   plausible" scenario, but the WORST-CASE band above is the one actually
   registered as the frozen decision rule, per this project's own standing
   discipline of pricing anything load-bearing against the pessimistic
   corner (§12.4b's own precedent).
4. **Decision, mechanical:**
   - **SCALING-LAW CONFIRMED** iff neither fit is AMBIGUOUS (item 2) AND
     both `CI(x0,80)` and `CI(x0,96)` overlap `[0.4745, 0.6165]`.
   - **SCALING-LAW REFUTED** iff neither fit is AMBIGUOUS AND at least one
     of the two CIs excludes the band. Sub-case, named but still formally
     REFUTED: exactly one excludes → **DIRECTIONAL DRIFT, NOT YET
     SATURATED** (§15.0 outcome 2's own parenthetical).
   - **AMBIGUOUS** at either d individually per item 2; the WAVE's own
     overall call is AMBIGUOUS if either d's fit is AMBIGUOUS, deferred to
     the pre-registered seed-contingency follow-up before a CONFIRMED/
     REFUTED call is made at all.

**Secondary (cliff-width scaling), pre-registered, NOT load-bearing (§15.0):**
report `w(64)=0.0597 [0.0557,0.0642]`, `w(80)` `[CI]`, `w(96)` `[CI]` side by
side; note the qualitative direction (increasing/decreasing/flat) and whether
the three CIs pairwise overlap — purely descriptive, no bar, mirrors §12.4's
own "linear form... disclosed secondary" precedent exactly.

#### 15.10.1 What this criterion structurally cannot show (disclosed now,
before any data, mirrors §13.0's own unfalsifiability disclosure)

With only three `d_state` points in the FULL `n>d` regime series (64, 80,
96 — d=128 is a different regime, §15.1), and only two of them NEW, this
wave can determine "invariant-in-`K/d`-ratio-terms vs. not" at these
specific three points, but — per §13.0's own already-registered structural
limit — **cannot rule out that some OTHER monotone-in-`d` rescaling would
ALSO read as "invariant" across these same three points.** A genuine
discriminating test between `K/d`, `K/√d`, or any other normalization
requires more `d`-points spread more widely than this wave's own tight
64–96 window affords; this wave answers the NARROWER, still-useful question
of whether the ratio account holds up locally, immediately above d=64, before
the d=128 regime change — not the fully general normalization-identification
question.

---

### 15.11 Cost model — interpolated from real d=64/d=128 measurements (a
BETTER-GROUNDED estimate than §13.4's own first-principles-only derivation,
since two real endpoints now bracket the target range)

**Real per-cell wall-clock, pulled directly from the archived harvests
(§12.9/§13.10), verified this session:**

- d=64, K∈{34,38,42,46}, 12 candidate-(d) cells: individual `wall_s` values
  900.0–985.6s, mean **954.10s = 0.26503 GPU-h/cell** (§12.9's own per-cell
  table, re-averaged this session).
- d=128, K∈{68,76,84,92}, 12 candidate-(d) cells: `wall_s` sum 26326.74s,
  mean **2193.90s = 0.60942 GPU-h/cell** (§13.10's own realized total,
  re-divided this session).

**Log-log power-law interpolation (computed this session):** fitting
`cost(d) = A·d^p` through these two real points gives `p = 1.2013`,
`A = 6.4545`. **Predicted per-cell cost at the new d's:**

| d_state | predicted per-cell `wall_s` | predicted GPU-h/cell | cross-check (linear interpolation in `d`) |
|---|---|---|---|
| 80 | 1247.4s | **0.3465** | 1264.1s = 0.3511 GPU-h (+1.3% vs power-law) |
| 96 | 1552.8s | **0.4313** | 1574.0s = 0.4372 GPU-h (+1.4% vs power-law) |

Both interpolation methods agree within ~1.5%, giving reasonable confidence
in the ~0.35/0.43 GPU-h/cell point estimates — a materially tighter
estimate than §13.4's own zero-data, first-principles-only bracket (which
had no real d-scaling measurement to anchor against at the time it was
written).

**Disclosed limitation, unchanged in kind from §13.5's own note:** these two
real anchor points (d=64, d=128) BOTH used matched-`K/d`-ratio grids, so like
§13.5's own calibration cell, this interpolation conflates `K`-scaling and
`d`-scaling — but since THIS wave's own grid ALSO holds `K/d` fixed at the
same ratios, the conflated (combined) scaling is exactly the relevant
quantity to extrapolate, not a confound to correct for.

**Anchor-point (low-K) cells are cheaper — same within-d sub-linear-in-K
scaling as §12.5's own measured 1.264× ratio over a 1.5× K change** (local
exponent `q ≈ 0.775`, derived from that same ratio): `K=20` at d=80 predicted
**≈0.169 GPU-h/cell**; `K=24` at d=96 predicted **≈0.212 GPU-h/cell**.

---

### 15.12 Budget and program ceiling — PI-DECISION

**Full cost table, 2× contingency applied throughout per house discipline
(mirrors every prior wave's own mandatory multiplier):**

| Item | Cells | GPU-h/cell (point estimate) | Total (1×) | Total (2×) |
|---|---|---|---|---|
| Mandatory grid, d=80 (K=43,48,53,58, 3 seeds) | 12 | 0.3465 | 4.158 | 8.316 |
| Mandatory grid, d=96 (K=51,57,63,69, 3 seeds) | 12 | 0.4313 | 5.176 | 10.352 |
| Mandatory anchor, d=80 (K=20, 3 seeds) | 3 | 0.169 | 0.507 | 1.014 |
| Mandatory anchor, d=96 (K=24, 3 seeds) | 3 | 0.212 | 0.636 | 1.272 |
| **Mandatory total (30 cells)** | **30** | | **10.478** | **20.956** |
| Optional Gate-1 probes, 1 per K-group incl. anchor (5/d × 2d), 0.25× rate | 10 | — | 0.873 | 1.746 |
| Seed contingency, +2/K-group, worst case ALL 10 K-groups fire | ≤20 | — | 6.985 | 13.970 |
| **All-in worst case (60 cells)** | **60** | | **18.336** | **36.672** |

**Mandatory-only, pessimistic 2× bracket: ≈21.0 GPU-h — the load-bearing
go/no-go ceiling number.** All-in worst case (every optional/conditional
cell fires): ≈36.7 GPU-h at 2×.

**KEY_ANCHORING program ledger status, verified against `STATE.md` and
§14.13, 2026-07-07: 78.2270/80 GPU-h spent, reserve 1.7730/80 — the existing
sub-ledger CANNOT fund even the cheapest single mandatory cell of this design
at its own pessimistic bracket, let alone the full mandatory grid.** This
wave requires a **NEW sub-ledger allocation, not an extension of the
exhausted one** — mirroring the framing every prior §12/§13/§14 header used
when opening a new wave against a nearly-exhausted or formally-closed
program.

**Proposed two-tier ceiling request (mirrors §14.4's own Option-1
"mechanical default, stage the ask" pattern):**

- **Tier 1 (mandatory-only): request a NEW ceiling `H_scaling = 21 GPU-h`.**
  Covers the full 30-cell mandatory grid (including the mandatory low-K
  anchor cells, §15.8) at the pessimistic 2× bracket with ≈0.04 GPU-h
  margin. This is the minimum ask that lets the primary invariance test
  (§15.10) run to completion at all.
- **Tier 2 (conditional, PI-gated exactly as §14.4 Option 2): up to an
  ADDITIONAL +16 GPU-h** (rounding up `36.672 − 20.956 ≈ 15.72`) if Gate-1
  probes and/or seed contingency are needed. Per this design's own
  mechanical cut priority (§15.14), these are the FIRST things cut under
  budget pressure — the +16 GPU-h ask is a ceiling, not an expectation.

**Realized-rate expectation (not the go/no-go number, mirrors every prior
wave's own disclosure): mandatory-only at 1× (≈10.5 GPU-h) is the reasonable
single best guess** — this program's own historical realized/ceiling ratios
have ranged 13.6% (§12.9) to 87.0% (§14.13, cumulative both stages), with a
median well under 50%; there is no reason to expect this wave's own realized
cost to land near the pessimistic 2× tail.

**Explicit surplus disclosure, per the task's own instruction NOT to pad the
grid to consume more compute:** the nominal "saturate 6 GPUs (2–7) for 1–2
days" framing implies a nominal target of roughly `6 GPUs × 24–48 hours =
144–288 GPU-h`. **This design's own honest worst-case ask (≈36.7 GPU-h,
all conditionals firing at the pessimistic bracket) is 107–251 GPU-h BELOW
that nominal target — a real surplus, not a design gap.** Wall-clock is
similarly far under budget: 30 mandatory cells across 6 GPUs run in
`ceil(30/6) = 5` parallel rounds at ≈1250–1550s/cell ≈ **2.2 hours**; even
the full 60-cell worst case completes in `ceil(60/6) = 10` rounds ≈ **4.3
hours** — nowhere near 1–2 days. **This draft does NOT recommend padding the
K-grid, seed count, or `d_state` set to consume the nominal target** — the
honest grid is what it is (§15.3's translation is fully mechanical, not
tunable for compute-burning purposes); the surplus GPU-time should go to
other queued work (the mechanism wave's own H4/H1/H5 items, or the
frozen-bias LM program's own largely-unused 6.9/135 GPU-h budget,
`STATE.md`), not to artificially inflating this wave.

---

### 15.13 Staging — calibration-first, mechanical gates

**Stage −1 (Wave −1, zero GPU, CPU-only, BLOCKING — nothing below launches
until this completes):**
1. Kernel-safety measurement, d=80 and d=96 (§15.2) — mechanical drop rule
   if either fails.
2. Extend `_SAFE_D_STATE` / `--d-state` argparse choices, ONLY after (1)
   passes.
3. Derive + byte-verify the two new threshold pins (§15.5).
4. Fix the `GATE2_N_ITER_BY_K` collision + run the n_iter-sufficiency check
   at all 10 new (K,d) pairs, including `(K=48, d=80)` specifically (§15.6).
5. Compute the λ=1 ceiling at all 10 new (K,d) pairs (§15.7).
6. Re-run the committed power-check (§15.8) at the wave's own frozen seed —
   confirms the mandatory anchor point's own justification one more time
   against the FINAL grid before build.
7. Manifest-regression smoke: assert the new wave's manifest-generator
   functions produce byte-identical output to the EXISTING `keyanchor_
   dstate_manifest(d_state=128, Ks=(68,76,84,92))` call when invoked with
   those same arguments (mirrors §12.2.1/§13.3 item 6's own regression
   guarantee).
8. Full `smoke_key_anchoring.py`-style smoke suite, re-run at d=80 and d=96.

**Stage 0 (calibration, mandatory house rule — mirrors §13.5/§14.4b
verbatim): ONE cell per new d, the CHEAPEST K in that d's own grid** — d=80:
`K=43`, seed 1030 (the lowest K in the MAIN 4-point grid; the K=20 anchor
point is cheaper still but is deliberately NOT the calibration cell, since
its own per-step cost profile may differ enough at very low K to be a less
representative calibration point for the grid's own bulk); d=96: `K=51`,
seed 1430. **Blinded readout: `read_wall_s_only(path)` (§13.5's own F9-fix
helper, reused verbatim) reads ONLY `wall_s`; `h4` is quarantined until the
full per-d manifest is generated.** Both calibration cells can run IN
PARALLEL (independent d's, no shared-calibration dependency the way §14.4b's
rank-4/diffuse arms had) — 2 cells, 2 GPUs, in parallel.

**Checkpoint after Stage 0:** compare each d's realized `wall_s` against its
own §15.11 point estimate. **Abort/re-price trigger (mirrors §12.2.3's own
1.5× threshold):** if EITHER calibration cell's realized `wall_s` ≥ 1.5× its
own point-estimate-at-2×-contingency bracket edge (d=80: `1.5 × 0.3465 × 2 ×
3600 = 3742.2s`; d=96: `1.5 × 0.4313 × 2 × 3600 = 4658.1s`) — halt, diagnose
(check `nvidia-smi` for contention, per §12.2.3's own leading-suspect
ordering), re-price the FULL §15.12 budget table before proceeding.

**Stage 1 (remaining mandatory grid): 26 cells** (11 remaining at d=80's
main grid + 3 anchor cells at d=80, 11 remaining at d=96's main grid + 3
anchor cells at d=96 — i.e., everything in the 30-cell mandatory total
except the 2 calibration cells already run in Stage 0), launched across the
available GPUs in parallel batches. Per-cell abort rule and running-
projection cut rule, both carried verbatim from §12.2.3 (§15.14).

**Stage 2 (conditional/optional, cut-eligible per the running-projection
guard): Gate-1 probes (10), then seed contingency (≤20)** — priority order
in §15.14.

**Pre-launch contention check (mirrors §12.2.3's own compounding-contention
note, re-verified rather than assumed current):** `STATE.md`'s 2026-07-07
pre-compaction snapshot reports GPUs 2–7 as idle (no concurrent multi-cell
program currently sharing this allocation, unlike the §12/§14-era frozen-bias
LM concurrency) — **this must be RE-CHECKED at actual launch time, not
assumed still true from this drafting session's own read of `STATE.md`**;
if a concurrent program IS running by launch time, apply the SAME
compounding-contention mitigation §12.2.3 registered (stage this wave alone
first, gate any concurrent program's own calibration on this wave's Stage 0
clearing).

---

### 15.14 Abort/degenerate rules — carried from §12 verbatim

**Per-cell abort (§12.2.3 item 1, restated with this wave's own bracket
numbers):** after ANY completed cell (not only the first), if that cell's
realized `wall_s` ≥ 1.5× its own d's pessimistic-bracket upper edge — halt
all remaining launches immediately. d=80 threshold: 3742.2s (main grid) /
1825.2s (K=20 anchor, using its own 0.169×2×3600×1.5 figure); d=96
threshold: 4658.1s (main grid) / 2289.6s (K=24 anchor).

**Diagnose-before-continuing (§12.2.3 item 2, verbatim):** check `nvidia-smi`
process list and any concurrent program's own GPU assignment before assuming
any other cause.

**Re-price before continuing (§12.2.3 item 3, verbatim):** if contention is
confirmed, wait for it to clear or re-derive the bracket at the contended
rate and re-check against §15.12's budget before launching another cell.

**Running-projection cut rule (§12.2.3, verbatim mechanism, this wave's own
priority order):** after each completed cell (once ≥3 cells have completed),
compare cumulative realized `wall_s` against a running projection using the
REALIZED rate, not the original estimate. If the projection would exceed the
mandatory-only 2× ceiling (§15.12), cut in this order: **(1) seed
contingency → (2) Gate-1 probes not yet launched → (3) halt and report if
even mandatory-only cells are projected to overshoot.**

**Degenerate-run exclusion — reused verbatim, §15.9's own citation.**

**Seed-contingency trigger, mechanical, NEW for this wave (since there is no
per-K h4 "bar" here to be "ambiguous at 2/3" the way §12.2's own trigger
was worded — a genuinely new condition, not a copy-paste):** fires per
(K,d)-group IFF **either** (a) that group's own h1 guard
(`M2_in_distribution['1']`) lands below 0.98 at ANY of its 3 seeds
(training-health concern), **or** (b) that group's own 3-seed h4 range
(max−min) exceeds **0.15** — an empirically-derived threshold, verified
this session against §14.0b's own archived per-K seed ranges at d=64
(K=34: range 0.0951 — the widest observed, near the steepest part of the
curve; K=38: 0.018; K=42: 0.0132; K=46: 0.0013), set at roughly 1.6× the
single widest historical within-K range to give comfortable margin without
being so loose it never fires.

---

### 15.15 Build scope — wave dispatch and manifest generalization

**Verified this session, `run_deltanet_rd_exactness_sweep.py`:**

1. **`_spec()` (lines 89–193) is ALREADY fully generalized** — `d_state=64`
   default parameter, filename bit `f"d{d_state}"` emitted only when
   `d_state != 64` (lines 172–173), threaded into the returned identity dict
   (`"d_state": d_state`, line 190). **No build task.**
2. **`keyanchor_dstate_manifest(d_state: int = KEYANCHOR_DSTATE_D, Ks=
   KEYANCHOR_DSTATE_KS)` (line 1905) is ALREADY parameterized by both
   `d_state` and `Ks`** — the §13 build generalized this function's
   signature even though the wave dispatch itself (module constants
   `KEYANCHOR_DSTATE_D = 128`, `KEYANCHOR_DSTATE_KS = (68,76,84,92)`, lines
   1877–1890) hardcodes a SINGLE d. **Required build task:** add new module
   constants (`KEYANCHOR_SCALING_D80_KS = (20, 43, 48, 53, 58)`,
   `KEYANCHOR_SCALING_D96_KS = (24, 51, 57, 63, 69)`) and two thin wrapper
   calls (`keyanchor_dstate_manifest(d_state=80, Ks=KEYANCHOR_SCALING_
   D80_KS)`, similarly for 96) — reusing the EXISTING generalized function
   body directly, mirroring the task's own framing ("near-clone of the
   keyanchor-cliff wave at new d_state values," structurally closer to
   keyanchor-dstate's own d-parameterization plus keyanchor-cliff's own
   K-grid/fitting-estimator logic).
3. **`build_cmd`'s own `--d-state` passthrough (line 2691) already threads
   any `d_state != 64` value to `run_deltanet_rd.py`'s CLI** — no build task
   beyond §15.2's own argparse `choices` extension (which this passthrough
   depends on).
4. **New `--wave` choice required:** `run_deltanet_rd_exactness_sweep.py`'s
   own `--wave` argparse `choices` list (line ~2809) currently lists
   `[..., "keyanchor-e", "keyanchor-cliff", "keyanchor-dstate",
   "keyanchor-dose"]` — register a new value, `"keyanchor-scaling"`,
   mirroring the existing naming convention.
5. **Seed blocks (fresh, escalating, never-reuse convention — §11.1's own
   stated reasoning: not strictly required for collision-safety anymore
   since `d_state` is now a name/identity bit per item 1 above, but still
   followed for human provenance-tracing):**

| d_state | K | primary seeds | +2 contingency | Gate-1 probe |
|---|---|---|---|---|
| 80 | 20 (anchor) | 1020,1021,1022 | 1023,1024 | 1025 |
| 80 | 43 | 1030,1031,1032 | 1033,1034 | 1035 |
| 80 | 48 | 1130,1131,1132 | 1133,1134 | 1135 |
| 80 | 53 | 1230,1231,1232 | 1233,1234 | 1235 |
| 80 | 58 | 1330,1331,1332 | 1333,1334 | 1335 |
| 96 | 24 (anchor) | 1420,1421,1422 | 1423,1424 | 1425 |
| 96 | 51 | 1430,1431,1432 | 1433,1434 | 1435 |
| 96 | 57 | 1530,1531,1532 | 1533,1534 | 1535 |
| 96 | 63 | 1630,1631,1632 | 1633,1634 | 1635 |
| 96 | 69 | 1730,1731,1732 | 1733,1734 | 1735 |

6. **Manifest-regression smoke (registered, mirrors §12.2.1's own):** assert
   the generalized wrapper calls produce byte-identical output to
   `keyanchor_dstate_manifest(d_state=128, Ks=(68,76,84,92))`'s own EXISTING
   output when called with those same arguments — must PASS before trusting
   the new d=80/96 branches.
7. **`fit_cliff_curve.py`: NO build task** (§15.8/§15.10 already establish
   `--d-state`/`--k-grid` are free CLI parameters that work unmodified for
   any `d_state` outside `ANCHORED_D_STATES`).

**Addendum (2026-07-07, FIX agent, K=20 anchor-cell H_extra hop-collision):**
during the wave-2 launch (`keyanchor_scaling_run2`), the 3 d=80 K=20 anchor
cells (seeds 1020–1022, item 5's own table above) failed deterministically
at config construction, never reaching the training loop. Root cause:
`run_deltanet_rd.py`'s `--h-extra` CLI flag defaults to `[7, 21]`
(unchanged since before this wave); at K=20, `21 % 20 == 1`, which
coincides with `H_train=(1,2,3)`'s own residue set `{1, 2, 3}` at that K.
`grammar_rd.py`'s `DeltaNetRDTaskConfig.__post_init__` periodicity guard
(the verbatim Finding B fix already documented in this file's own §15.4)
correctly refused to construct the config — **the assertion worked exactly
as designed; this is not a bug in the guard**, it is a K=20-specific gap in
the wave's own manifest builder, which never threaded any `H_extra`
override through `_spec()`/`build_cmd()` before this fix (every prior
wave's cells all sit at K where `[7,21]`'s residues `{7, 1}` happen not to
collide with `{1,2,3}` — K=20 is the first K in this program's own history
where `21 % K` lands on a training residue).

Fix (orchestrator design decision, implemented in
`run_deltanet_rd_exactness_sweep.py`): a K-scoped exception,
`KEYANCHOR_SCALING_H_EXTRA_OVERRIDE_BY_K = {20: (7, 25)}`, consulted by
`_keyanchor_scaling_spec()` and threaded through `_spec()`'s new `h_extra`
parameter into `build_cmd()`'s new `--h-extra` passthrough — **ONLY** for
K=20 cells; every other cell (all 27 already-complete cells across both
d=80 and d=96, including d=96's own anchor K=24) keeps the unmodified CLI
default and an byte-identical emitted command line (verified locally,
below). `H_extra=(7, 25)`: residues `7 % 20 == 7` and `25 % 20 == 5`, both
outside `{1, 2, 3}`; 25 (like the original 21) stays `> K`, preserving the
"beyond-K literal-depth extrapolation" role the extra hop plays (sec 15.9's
own candidate-(d) instrumentation reads `H_test + H_extra` at eval time).
`H_test=[4, 5, 6]` (the CLI default) is **UNCHANGED** — h4 is the
fit-relevant hop this wave's own §15.10 success criteria read, and `[4,5,6]`
does not collide with K=20's training residues either way.

Local verification (CPU, `grammar_rd.DeltaNetRDTaskConfig` direct
construction): `K=20, H_extra=(7,25)` — **PASSES** `__post_init__`.
`K=20, H_extra=(7,21)` (the old default) — **still FAILS** with
`"H_test/H_extra hop h=21 has h % K=20 == 1, coinciding with a TRAINING
hop's residue ([1, 2, 3])"` (negative test, confirms the root cause
reproduces and the guard has teeth). `K=24, H_extra=(7,21)` (d=96's own
anchor, unmodified) — **PASSES**, confirming no override was needed there.
`smoke_keyanchor_scaling.py`'s 16-item suite re-run clean after the fix.

This is a disclosed, K-conditional exception, not a silent global change —
no other cell's manifest, filename, or emitted CLI arguments changed.

---

### 15.16 Standing constraints (restated, apply unchanged to this wave)

- Exact thresholds, no numerical-tolerance slack on structural checks (the
  pin byte-diff check, §15.5; the manifest-regression smoke, §15.15 item 6).
- Negative unit tests run to completion, not merely written (§15.6 item 4;
  §15.15 item 6).
- Smoke test every model incl. eval batch before launch (`smoke_key_
  anchoring.py`-style suite, re-run at d=80/96, §15.13 Stage −1 item 8).
- tmux + supervisor launch pattern for unattended runs — this wave's chain
  script (`keyanchor_scaling_chain.sh`, mirroring `keyanchor_dstate_chain.sh`'s
  own structure) launches inside `tmux new-session -d -s <name> "<cmd>"`.
- Sweep script try/except per config (already the existing behavior of
  `run_deltanet_rd_exactness_sweep.py`'s per-cell dispatch loop, inherited
  unchanged).
- Archives to repo (≤25MB) + SSD mirror, both, no exceptions.
- Multiple independent adversarial audit rounds, not one — this section is
  explicitly DRAFT pending its own attack round(s) before CLEARED-FOR-BUILD.

---

### 15.17 ATTACK-ROUND QUESTIONS — ROUND 0 (self-attack, minimum 5, each with
a best-current-answer or an explicit TODO for the real attack rounds)

**Q1. Is the kernel-safety gate (§15.2) actually going to be run before
anything else, or will schedule pressure tempt a "probably fine between two
verified values" shortcut?** This is the single highest-consequence risk in
this draft — if skipped or rushed, a crash mid-wave (or worse, a SILENT
numerical corruption that doesn't crash but produces wrong results) could
invalidate everything downstream. **Current answer:** the gate is written as
explicitly BLOCKING (§15.2, §15.13 Stage −1 item 1) with a mechanical drop
rule if either d fails — but a real attack round should check whether the
build/launch tooling can even physically proceed without this artifact
existing (i.e., should the chain script itself refuse to run if the
kernel-safety result JSON is missing or reports a FAIL) rather than relying
on the design doc's own prose to be followed. **TODO for attack round:**
require an ENFORCED tooling gate (the chain script checks for a PASSING
kernel-safety artifact file before invoking `run_deltanet_rd.py` at all),
not merely a documented prerequisite.

**Q2. Does the pre-registered invariance band (§15.10) actually distinguish
the hypothesis under test from its alternatives, or is it wide enough
(±0.071, ≈0.142 total) to pass almost any plausible outcome?** The band's own
width is dominated by the projected new-d CI half-width (0.065 of the 0.071
total), which is itself a SIMULATED, not measured, quantity, and the
mandatory low-K anchor point only ever reduces it to roughly HALF the
no-anchor number, not down to d=64's own tight 0.0064 half-width. **Current
answer:** the band (±0.071) IS narrower than the full observed (0.5,0.75)
transition bracket (0.25) by a real, checked margin (~43%), so a clean
REFUTED call (a fitted `x0` landing well outside, e.g. near 0.75 or 0.35)
remains possible and would be a real, informative result — but a CONFIRMED
call at this band width could not distinguish "x0 truly frozen at 0.5455"
from "x0 drifted by up to ~0.07 in ratio terms" (≈4.5 raw K at d=80, ≈6.8 raw
K at d=96) — a MEANINGFUL amount of drift that this design could still call
CONFIRMED. **TODO for attack round:** decide whether a narrower, secondary
"tight-invariance" band (e.g., ±0.03) should ALSO be reported (not as the
primary bar, but as a disclosed secondary reading, mirroring the cliff-width
secondary reading already registered) so a reader can see whether the result
clears the strict as well as the loose standard.

**Q3. Is the mandatory low-K anchor addition (K=20/K=24, §15.3/§15.8) itself
a confound — does adding a point far from the transition change what the fit
is actually testing, relative to a "pure" 4-point translation of §12's own
grid?** A skeptical reader could argue this draft is quietly changing the
experimental design (adding data d=64's own wave never needed, since IT had
real archived K=16/32/48 anchors) rather than translating it "verbatim" as
instructed. **Current answer:** the anchor point plays the EXACT SAME role
K=16 played at d=64 (a point far below the transition, expected at or near
ceiling, that primarily constrains `L` and the logistic's left tail, not the
`x0`/`w` region itself) — it is not "new data invented to help the answer
land favorably," it is the closest available substitute for the flanking
information d=64 got for free. But this is a genuine, disclosed departure
from a LITERAL verbatim translation, and the task's own instruction to
"reuse §12's grid construction principle... anything that cannot be reused
verbatim gets its own subsection with a verification step" is exactly why
§15.8 exists as a separate, justified subsection rather than being silently
folded into §15.3. **TODO for attack round:** confirm the anchor point's own
predicted h4 (expected ≈1.0, since K/d=0.25 is far below any plausible x0)
is checked and disclosed as a sanity/admissibility read (mirrors the h1
guard), not silently assumed.

**Q4. Does this wave's own "organic table, n>d regime" framing actually
survive contact with what §14 already showed, or is it a distinction that
sounds meaningful but isn't?** §14.12/§14.13 already tested coherence UP TO
0.40 (well above ANY organic value this wave will encounter — d=80's own
0.200, d=96's own 0.098, both far below 0.40) and found NO cliff, in EITHER
structure. A hostile reviewer could argue: if coherence up to 0.40 (frozen)
doesn't cause a cliff at d=128, why would organic coherence of only 0.20 or
0.098 (at d=80/96) cause one? **Current answer:** this is a real tension the
draft should not paper over. The counter-argument this wave is actually
built on is NOT "maybe a LOWER coherence dose does something a higher frozen
dose didn't" (that would be backwards and is not the claim) — it is that
§14's dosed cells were run at `d_state=128` (the OTHER regime, where the base
table is exactly orthogonal absent injection, and `d_state` itself is 2× the
d=64 cliff's own value), so §14 never actually tested "does an ORGANICALLY
non-orthogonal, LEARNED table at an INTERMEDIATE `d_state` (80 or 96, closer
to 64) reproduce a cliff at the matched `K/d` ratio" — the closest §14 came
was FREEZING a table at exactly a dose but ALSO at `d_state=128`, conflating
"frozen vs. learned" with "d_state=128 vs. intermediate" in a way this wave's
own design (learned table, intermediate d) does not. **This is, honestly,
the weakest part of this draft's own motivating case** — if a reviewer judges
that §14's exoneration is dose-general enough (any coherence level, frozen
or organic, at any `d_state`≥80, is safely below whatever threshold would
matter) to make this wave redundant, that is a legitimate KILL-worthy
finding for the real attack round, not something this self-attack can
resolve on its own. **TODO for attack round: this is the single question
most likely to determine whether this wave is worth running at all** — it
should get the most attention, not a process/statistics fix.

**Q5. Is the KEY_ANCHORING program ledger framing (§15.12: request a NEW
sub-ledger, not an extension) actually the right PI-decision framing, or
should this wave instead be scoped under a completely different program name
(since the parent program is explicitly declared COMPLETE in `STATE.md`)?**
**Current answer:** this draft follows §12's own precedent (reopening a
program the project's own state file marked closed, via an explicit PI
sign-off, not a self-authorization) — but §12 was reopening the SAME program
for one bounded, related wave; this draft is proposing to reopen a program
declared complete TWICE now (once before §12, again after §14.13). A hostile
reviewer might reasonably ask whether continuing to bolt new waves onto a
program that keeps getting re-declared "complete" and then reopened is
itself a process smell — perhaps this cluster of `d_state`-scaling questions
(this wave, plus the not-yet-designed `d_state`-vs-`K` factorial §14.13
itself named as the true remaining open question) deserves its OWN named
program/ledger from the outset, rather than another one-off reopening.
**TODO for attack round:** decide whether this wave should be the FIRST
wave of a newly-named, freshly-budgeted successor program (e.g.
"KEY_ANCHORING_CAPACITY_SCALING") rather than another patch onto the
KEY_ANCHORING ledger specifically.

**Q6 (bonus, process-level).** **Is 3 seeds/K still the right convention at
these new K/d combinations, given §15.8's own power-check shows meaningfully
wider CIs than d=64 even with the mandatory anchor?** §12.4a's own analogous
check found 5 seeds/K bought a further ≈17% CI reduction over 3 seeds/K at
d=64, "not registered since the marginal gain doesn't justify departing from
convention" there — but this wave's own CI is ALREADDY wider than d=64's, so
the marginal value of extra seeds here could be larger, not smaller, than it
was at d=64. **TODO for attack round:** re-run §15.8's own power-check at
4 and 5 seeds/K (both d's, with-anchor config) and check whether the
marginal CI improvement justifies departing from the 3-seed convention here
specifically, rather than assuming §12.4a's own conclusion (reached at a
DIFFERENT, better-anchored configuration) transfers unchanged.

---

## §15.18 ATTACK-ROUND-1 (2026-07-07) — VERDICT: RUN-AFTER-REASONING-LINK (wave PARKED)

Independent adversarial attack (fresh agent, re-derived every load-bearing
number). Verdict: technically competent, mostly arithmetically sound, but
one code-level FATAL falsifying the draft's own "no build task" claim, one
gate weaker than advertised, and the draft's own Q4 already concedes it is
the program's weakest-motivated wave. **Sequencing decision (accepted by
the orchestrator): REASONING-LINK Phase 1 (eval-only, existing checkpoints,
the STATE.md-registered publication keystone) takes the GPU budget first;
this wave's build gaps get fixed in parallel at zero GPU cost; launch is
revisited after REASONING-LINK Phase 1 lands.**

**FATAL-1 (build gap, fails loud):** `keyanchor_dstate_manifest()` /
`_gate1_manifest()` / `_calibration_manifest()` all index module-level
seed dicts hardcoded to K∈{68,76,84,92}
(`run_deltanet_rd_exactness_sweep.py:1888-1895`) — calling them at the new
K grids raises `KeyError: 20` immediately. §15.15 item 2's "already fully
generalized" claim is FALSE; the real build task is threading
`seeds_by_k`/`gate1_seed_by_k` parameters through (or fresh
`keyanchor_scaling_*` manifests). §15.15 item 6's manifest-regression
smoke cannot catch this (it only re-verifies existing Ks).

**FATAL-2 (gate insufficiency — RESOLVED same session):** the first
kernel smoke tested only T=256 — not §15.2 item 1's registered
T∈{128,224,448} protocol — and T=128 (the `_MIN_KERNEL_T` padding floor,
hence the most common real value) is exactly where D=32 historically
crashed. The attacker predicted the d=32 "pass" anomaly was the untested
T. **Re-run executed to the full registered protocol
(`smoke_dstate_kernel.py` superseding version, box GPU 3, artifact
`results/smoke_dstate_kernel_result.json` committed): d=80/96 PASS
forward+backward with finite grads AND finite forward outputs at ALL of
T∈{128,224,448} → gate GENUINELY CLEARED. d=32 FAILS at T=128 with
`cudaErrorIllegalAddress` — the historical crash reproduces exactly where
predicted, confirming the T=256-only pass was a false negative.** (d=32's
T=224/448 rows are cascade artifacts of the poisoned CUDA context, not
independent measurements; candidates ran before d=32 in loop order, so
their cells are clean.) The chain script must mechanically require this
PASSING artifact before launch (Q1 TODO stands).

**MAJOR-3 (build gap):** §15.13 item 8's "re-run smoke_key_anchoring.py
at d=80/96" is not a re-run — that suite has no `--d-state` flag and
hardcodes d∈{64,128} per helper. Needs a wave-specific
`smoke_keyanchor_scaling.py` (per the `smoke_keyanchor_cliff.py` /
`smoke_keyanchor_dstate.py` convention) — a new file never entered in
§15.15's build-scope enumeration.

**MINOR-4:** §15.3 arithmetic slip — K=43 at d=80 is 0.64 (not 0.36)
from raw predicted 43.64. Non-load-bearing.

**Additional latent collision named:** sibling per-wave `_ceiling_by_k`
functions follow the same K-keyed pattern as GATE2_N_ITER_BY_K — §15.7's
ceiling computation needs fresh namespaced storage, same fix class.

**Verified correct by independent re-derivation (no changes needed):**
Welch-floor/σ_chance/a_shape table; cost power-law (p=1.2013, A=6.4545,
0.3465/0.4313 GPU-h/cell); budget table (20.956 GPU-h at 2×, ≈0.04
margin); K-grid translations both d's; pool arithmetic; fit_cliff_curve
generality; pooled_null_check d_state-genericity; GATE2 collision scope
(K=48 the only one); `sim_cliff_power.py` bootstrap re-run reproduced the
cited CI widths (bare 0.0741/0.0724, anchored 0.0385/0.0383). Band-power:
fixed-K rival predicts x0(96)≈0.364 and absolute-slack rival predicts
x0(80)≈0.636/x0(96)≈0.697 — BOTH outside the ±0.071 band, so the
pre-registered criterion is not vacuous against either named rival;
residual weakness (Q2) unchanged: mild drift ≤~0.07 would still read
CONFIRMED.

**Q4 adjudication (existence question):** a law needs ≥3 points and no
prior wave held organic coherence regime constant while varying d — the
wave is not redundant with §14. But §14's exoneration headroom (doses
0.13-0.40 vs organic 0.098-0.200) makes CONFIRM the strongly expected
outcome, and the result does not feed the registered publication keystone.
Hence: park, fix build gaps at zero GPU cost, revisit after
REASONING-LINK Phase 1.
