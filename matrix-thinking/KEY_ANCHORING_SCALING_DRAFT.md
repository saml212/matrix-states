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

---

## §15.19 VERDICT — harvest, 2026-07-07: AMBIGUOUS (mechanical, data-quality
gate), d=80 individually REFUTES ratio-invariance; d=96 flat-near-ceiling,
degenerate fit

**Launched under `KEYANCHOR_SCALING_PI_SIGNOFF=1`** (recorded at the top of
both `keyanchor_scaling_run1.log` and `run2.log`), i.e. an explicit
sign-off past the §15.18 PARK gate, after REASONING-LINK Phase 1 landed
(`STATE.md`, 2026-07-07 ~12:45 UTC snapshot) — the FATAL-1/MAJOR-3 build
gaps §15.18 required (seed-dict threading, `smoke_keyanchor_scaling.py`)
were fixed and re-verified (smoke suites in both chain logs: `SMOKE SUITE:
ALL ITEMS PASSED`) before this run. The kernel-safety gate (§15.2,
FATAL-2) is CONFIRMED CLEARED at the full registered `T∈{128,224,448}`
protocol: `results/smoke_dstate_kernel_result.json` (pulled from box,
archived below) shows d=80/d=96 PASS forward+backward, finite grads and
outputs, at all three T; d=32 FAILS at T=128 exactly as historically
predicted (`cudaErrorIllegalAddress`), confirming the gate has teeth, not
a rubber stamp.

**Scope: all 30 mandatory cells (§15.12), candidate (d) only — d=80:
K∈{20(anchor),43,48,53,58}×3 seeds; d=96: K∈{24(anchor),51,57,63,69}×3
seeds. 30/30 raws pulled from box and independently re-verified from raw
JSONs this session (never the box's own printed summary trusted blind).**

### Per-cell verification (mechanical, all 30 raws)

`complete=true`, `steps_completed=20000`, `timed_out=false` — **30/30,
no exception.** Seed/K/d_state match §15.15's registered table exactly at
every cell (zero missing, zero unexpected cells). Architecture pin
(`anchor_active=True`, `anchor_lambda_mode="learned"`,
`anchor_table_frozen=False`, `anchor_table_init_mode="frame_potential"`,
`drift_probe=True`, `rev7_engagement=True`, `geo3_n_iter=20`) — **uniform
across all 30 cells, no exception.** `H_extra`: the 3 d=80/K=20 anchor
cells (seeds 1020–1022) read `[7, 25]`, matching §15.15's addendum fix
exactly; **all other 27 cells read the unmodified default `[7, 21]`** — the
K-scoped override did not leak to any other cell (verified, not assumed).
h1 guard (`M2_in_distribution["1"]["recovered_frac@0.9"]`, bar ≥0.98):
**exactly 1.000000 at every one of the 30 cells** — no training-health
concern anywhere in this wave.

REV7 threshold pins (`REV7_THRESHOLD_PINNED_D80.json`/`_D96.json`,
pulled from box): `derived.effect_size_floors.r_min_headline_band = 0.35`
at **both** new pins, matching the existing d=64/d=128 pins exactly (the
one registered cross-d invariant); `sigma_chance`/`r_min_partial_band`
DIFFER at every d as required (d=80: 0.111803/0.223607; d=96:
0.102062/0.204124) — §15.5's PASS condition satisfied, no FATAL.

**One admissibility anomaly, disclosed prominently (first occurrence of
this class anywhere in the KEY_ANCHORING program's own archive history,
checked against §12/§13/§14's own admissibility tables):**
`wkeyanchor-scaling_rdx_K69_armd_s1730_..._d96.json` (d=96, K=69,
seed=1730) reads `geo3_admission.admissible=false`,
`ns_converged_no_fallback=false`, `checkpoint_fallback_seen=true` (but
`n_geo3_fallback_train_steps=0` — the fallback never accumulated a
counted training step). Every OTHER admissibility field on this cell
passes clean (`value_salvage_tier_pass=true`,
`finite_loss_no_divergence=true`, `task_performance_floor_pass=true`,
`h1_recovered_frac_at_0.9_final=1.0`), and `complete`/`steps_completed`
are unaffected. Per §15.9's own verbatim-reused exclusion rule ("every
cell must show admissible: true... checkpoint_fallback_seen: false"),
this ONE cell is excluded from the fit-input pool; sensitivity checked
directly (re-fit d=96 with `n=2` at K=69 instead of `n=3`): `x0` stays
pinned at the 0.9 bound, `degenerate_frac` drops from 98.52% to 94.77%
— still >>10%, **this anomaly does not change d=96's data-quality
outcome or the wave verdict.** K=69's own mean h4 shifts negligibly
(0.98393→0.98000 mean, both seeds 1731/1732 individually cleaner:
0.998585/0.961419). Flagged for the box's own maintainers: this is the
first `checkpoint_fallback_seen=true` cell in program history and worth
a root-cause look (possibly a transient NS non-convergence at this
specific, highest-tested `K/d` ratio geometry), but it is NOT load-bearing
for this wave's own verdict.

**Seed-contingency trigger (§15.14, mechanical, evaluated per (K,d)-group
— independent of the fit-level data-quality gate below): fires at TWO
d=80 groups.** 3-seed h4 range by group:

| d | K | seeds' h4 | range | fires (>0.15)? |
|---|---|---|---|---|
| 80 | 20 | 1.0, 1.0, 1.0 | 0.0000 | no |
| 80 | 43 | 0.9678, 0.9937, 0.8659 | 0.1278 | no |
| 80 | 48 | 0.8732, 0.6558, 0.8741 | **0.2183** | **YES** |
| 80 | 53 | 0.5828, 0.4767, 0.6546 | **0.1779** | **YES** |
| 80 | 58 | 0.2143, 0.3013, 0.3134 | 0.0991 | no |
| 96 | 24 | 1.0, 1.0, 1.0 | 0.0000 | no |
| 96 | 51 | 1.0, 1.0, 1.0 | 0.0000 | no |
| 96 | 57 | 0.9996, 0.9926, 1.0 | 0.0074 | no |
| 96 | 63 | 0.9957, 0.9645, 0.9812 | 0.0312 | no |
| 96 | 69 | 0.9918†, 0.9986, 0.9614 | 0.0372 | no |

†seed 1730, the non-admissible cell above; included here only to show
the trigger does not fire either way. **K=48/d=80 and K=53/d=80 —
exactly the two K-groups straddling d=80's own fitted transition
(x0=0.6756, between K/d=0.60 and K/d=0.6625) — exceed the 0.15
within-K-group range threshold.** This is the pre-registered
seed-noise-near-the-transition signature (§15.14's own citation: K=34/d=64
showed the single widest historical range, 0.0951, "near the steepest
part of the curve") reproducing at a noticeably LARGER magnitude here
(0.178–0.218 vs 0.095) — consistent with §15.8's own disclosed,
un-fully-closed CI-width gap at these new d's. Per §15.14, the
pre-registered follow-up is **+2 seeds at K=48/d=80 and K=53/d=80** (4
more cells, ≤0.7 GPU-h at 2× contingency) — not yet run; queued below.

### Independent re-fit (both d's, re-run locally against the pulled raws
— reproduces the box's own committed `fit_cliff_curve_d{80,96}_results.json`
to full float precision, confirming zero staleness/drift)

`python3 fit_cliff_curve.py --cliff-out-dir <wave dir> --d-state {80|96}
--k-grid <grid> --n-trials 4000` (no `--k32-dir`/`--k48-dir` — neither 80
nor 96 is in `ANCHORED_D_STATES=(64,)`, §15.8/§15.10):

| d_state | curve points (x=K/d, h4) | x0 | 95% CI(x0) | w | 95% CI(w) | degenerate_frac |
|---|---|---|---|---|---|---|
| 80 | (0.25,1.0) (0.5375,0.9425) (0.60,0.8010) (0.6625,0.5713) (0.725,0.2763) | **0.6756** | **[0.6620, 0.6868]**, width 0.0248 | 0.0521 | [0.0404, 0.0660] | **0.0000** (clean) |
| 96 | (0.25,1.0) (0.53125,1.0) (0.59375,0.9974) (0.65625,0.9805) (0.71875,0.9839) | 0.9000 (bound-pinned) | **None — too few valid bootstrap fits** | 0.0445 | n/a | **0.9852** (98.5%, far past the 10% bar) |

`sim_cliff_power.py`'s bootstrap machinery, `curve_fit` bounds/form, and
degenerate-fit definition are all reused VERBATIM (imported, not
re-implemented) — same instrument as §12.9/§13.10. `fit_cliff_curve.py`
byte-verified identical to both the box's live copy and the repo's
already-committed copy (sha256
`a2d2b8a86674529b0a746e69025c8ad304a486157b750a7db667fe2d060049da`, zero
drift) before trusting either fit.

**d=96's curve is not noisy — it is genuinely flat near the ceiling
across the ENTIRE tested window** (h4 ranges 0.9805–1.0 across all five
K's, K=24 through K=69, K/d=0.25 through K/d=0.71875): the sigmoid has no
transition to localize, so `curve_fit` pins `x0` at its own upper bound
(0.9) on 98.5% of bootstrap resamples — mechanically identical in kind to
§13.10's own d=128 "NO CLIFF ANYWHERE IN THE MEASURED WINDOW" result
(`x0=0.898`, degenerate by construction), reproducing here one d earlier
than d=128 previously showed it. Instrument-saturation is ruled out the
same way §13.10/§14.12/§14.13 ruled it out there: hop-21
(`M3_held_out["21"]`) and `effective_rank_whole_mean` both show real,
graded, non-floored variation across the d=96 cells (not checked in
exhaustive per-checkpoint detail this session, but confirmed non-uniform
at final checkpoint on inspection) — h4≈1.0 reflects a genuine, easy
regime, not a broken readout.

### Mechanical verdict (§15.10, applied exactly as pre-registered)

**Step 1 — data-quality gate (§15.10 item 2), evaluated FIRST, before any
band-overlap test:** d=80 degenerate_frac = 0.00% (**PASS**, <10%); d=96
degenerate_frac = 98.52% (**FAIL**, >>10%) → **d=96's fit = AMBIGUOUS**
by the wave's own pre-registered rule, full stop — regardless of why the
fit is degenerate (§15.10 item 2's text draws no distinction between
"noisy near a transition" and "flat at ceiling with nothing to localize";
both trip the same 10% mechanical bar).

**Step 2 — band-overlap test (§15.10 item 4), evaluated at d=80 only
(d=96 is already AMBIGUOUS, so its CI, which does not exist, cannot be
overlap-tested):** invariance band = `[0.4745, 0.6165]`
(`0.5455 ± 0.0710`, §15.10 item 3). d=80's CI `[0.6620, 0.6868]` lies
**entirely above** the band's upper edge 0.6165 (gap = 0.0455, i.e. the
CI's own lower bound clears the band by nearly half the band's own
half-width) — **d=80's fit EXCLUDES the invariance band.**

**Step 3 — wave-level decision (§15.10 item 4's own text, quoted
exactly): "the WAVE's own overall call is AMBIGUOUS if either d's fit is
AMBIGUOUS, deferred to the pre-registered seed-contingency follow-up
before a CONFIRMED/REFUTED call is made at all."** d=96 is AMBIGUOUS (Step
1) →

## **WAVE VERDICT: AMBIGUOUS.**

This is the mechanical, pre-registered outcome — not a judgment call.
**Disclosed precisely, so this is not mistaken for the named
"DIRECTIONAL DRIFT, NOT YET SATURATED" sub-case (§15.0 outcome 2's own
parenthetical):** that sub-case requires BOTH d's to have a determinate
CI, with exactly one excluding the band — d=96 has no determinate CI at
all (AMBIGUOUS, not "excludes"), so the formal label is AMBIGUOUS, not
DIRECTIONAL DRIFT, even though d=80's own individual result, taken alone,
would already qualify as a clean REFUTE (band excluded with a real
margin, zero degenerate bootstrap fits).

### Reading the two d's honestly, alongside the mechanical call

**d=80, evaluated on its own: a clean, non-degenerate REFUTE of ratio
invariance, with `x0` drifting UPWARD from 0.5455 to 0.6756** (+0.130 in
ratio terms, ≈10.4 raw K at d=80) — well beyond the band's own pessimistic
half-width (0.0710) and beyond even the "tight-invariance" secondary
band §15.17 Q2 flagged as worth reporting (±0.03: `[0.5155,0.5755]`,
also excluded). If d=80 alone had to be classified against §15.0's
outcome list, it is outcome 2 (SCALING-LAW REFUTED), not the directional
sub-case (both new-d's would need to be evaluable for that label).

**d=96, evaluated on its own: not "ambiguous" in the colloquial sense —
the measured curve is flat near h4≈1.0 across the ENTIRE tested window,
i.e. NO cliff was found anywhere in `K/d ∈ [0.25, 0.71875]` at d=96.**
This is formally AMBIGUOUS under §15.10's mechanical rule (bootstrap
cannot localize a transition that mostly is not there), but
substantively it is closer in kind to §13.10's d=128 "cliff exits the
window" result than to a genuine small-sample noise problem — worth
flagging as a live critique of the pre-registered rule for whoever
designs the next wave: a single "degenerate_frac>10%" trigger does not
distinguish "can't tell where x0 is" from "x0 is almost certainly outside
this window entirely," and the two call for different follow-ups (more
seeds vs. a wider/shifted K-grid). Descriptively, though, d=96's own
slight-but-real downward drift IS visible in the raw numbers (K24=1.0,
K51=1.0, K57=0.9974, K63=0.9805, K69=0.9839, a small dip concentrated at
the two highest-ratio K's) — consistent with a transition sitting at or
beyond the top of this wave's own tested window (K/d>0.72), not with a
transition hiding somewhere inside it.

### Rival comparison (§15.0, disclosed exactly where the fits land)

| Account | Prediction | Where the data land |
|---|---|---|
| Ratio-invariant (this wave's H0) | x0(80)≈x0(96)≈0.5455, band [0.4745,0.6165] | d=80 CI excludes the band by a real margin; d=96 uninterpretable by the mechanical rule but its own point estimate (0.90, unreliable) and its visible slight decline at the top of the grid both sit even further from 0.5455 than d=80 does — **not supported by either d.** |
| Fixed-K (absolute K≈35–46 constant) | x0(96)≈0.364 (ratio shrinks as d grows) | **Directly contradicted at d=96**: K=51 (ratio 0.531, well above 0.364) still reads h4=1.0 — no cliff is visible anywhere near fixed-K's own predicted location. Not tested as cleanly at d=80 (grid starts at K=20/ratio 0.25) but the observed x0(80)=0.6756 is far to the right of fixed-K's own implied d=80 prediction (0.5455×64/80×... — fixed-K predicts x0 SHRINKS with d, the opposite of what d=80 shows). **Refuted at both d's.** |
| Absolute-slack (`d_state` or `d_state−K` itself) | x0(80)≈0.636, x0(96)≈0.697 | **Closest rival to the data at both d's.** d=80's measured `x0=0.6756` sits 0.040 above the point prediction, and even the CI's own lower bound (0.6620) clears it — not an exact hit, but far closer than the ratio-invariant band center (off by 0.13+). d=96's own visible decline concentrates almost exactly around K/d≈0.66–0.72, bracketing absolute-slack's own predicted 0.697 — qualitatively the best fit of the three named accounts to both d's, though the mechanical AMBIGUOUS call means this is a descriptive read, not a confirmed result. |

### Secondary (disclosed, non-load-bearing): cliff-width `w` scaling

| d | w | 95% CI(w) |
|---|---|---|
| 64 (§12.9, cited) | 0.0597 | [0.0557, 0.0642] |
| 80 | 0.0521 | [0.0404, 0.0660] |
| 96 | 0.0445 | n/a (fit degenerate) |

d=80's `w` CI overlaps d=64's `w` CI (no width-scaling signal at d=80);
d=96's `w` is not interpretable from a degenerate fit and is reported for
completeness only, per §15.10's own "no bar attached" convention.

### Realized GPU-h vs. ceiling

Summed directly from all 30 raw cells' own `wall_s` (all single-GPU,
`--per-gpu 1`, verified from `keyanchor_scaling_chain.sh`):

| Group | Cells | Sum wall_s | GPU-h |
|---|---|---|---|
| d=80 (K=20,43,48,53,58 × 3) | 15 | 19984.65s | 5.5513 |
| d=96 (K=24,51,57,63,69 × 3) | 15 | 22446.75s | 6.2352 |
| **All 30 mandatory cells** | **30** | **42431.40s** | **11.7865** |

The 3 initial K=20/d=80 hop-collision failures (§15.15 addendum) failed
at config construction, before any GPU work started (no result JSON,
~sub-second logs) — **zero additional GPU-h**, not merely omitted.
Kernel-safety sweep, both smoke suites, threshold-pin derivation:
CPU-side or sub-minute multi-D forward/backward probes, no measurable
additional GPU-h (same disclosure convention as every prior wave's own
harvest).

**Realized: 11.7865 GPU-h against the Tier-1 approved ceiling `H_scaling
= 21 GPU-h` (mandatory-only 2× bracket: 20.956 GPU-h, §15.12) — 56.2% of
the 2× ceiling, and 112.5% of the 1× point estimate (10.478 GPU-h) — the
first wave in this program to land ABOVE its own 1× point estimate
rather than comfortably under it** (§12.9: 13.6%; §13.10: 34.8%; §14.12/
§14.13: 45.8%/87.0% cumulative), still nowhere near the 2× pessimistic
tail. Headroom remaining against the 21 GPU-h Tier-1 ceiling: **9.21
GPU-h** — comfortably covers the pre-registered seed-contingency
follow-up (+2 seeds × 2 K-groups = 4 cells ≈ 0.7 GPU-h at 2×) with wide
margin left over.

### What this wave does and does not show

1. **Does NOT confirm ratio-invariance of the d=64 cliff.** d=80 cleanly
   refutes it; d=96 is genuinely uninformative by the pre-registered
   mechanical rule (though descriptively also inconsistent with
   invariance).
2. **Does NOT cleanly confirm any single rival either** — the mechanical
   verdict is AMBIGUOUS, and even d=80's own clean REFUTE only rules out
   the SPECIFIC pre-registered band, not every possible non-ratio
   account (§15.10.1's own already-disclosed structural limit: three
   points cannot discriminate every candidate rescaling).
3. **Does newly and directly contradict fixed-K** at d=96 (h4=1.0 at
   K=51, well past fixed-K's own predicted trouble zone) — the one clean,
   load-bearing negative result this wave produces regardless of the
   AMBIGUOUS headline.
4. **Sharpens, without resolving, the "absolute state capacity"
   candidate** §14.13 named as the program's one remaining unadjudicated
   account: both d's land closer to the absolute-slack predictions than
   to the ratio-invariant band, but neither hits its own point prediction
   cleanly, and d=96's own degenerate fit means this reading is
   descriptive, not confirmed.

### Pre-registered next steps (queued, not run this session)

1. **Seed contingency, §15.14 (mechanical trigger already fired):** +2
   seeds each at K=48/d=80 and K=53/d=80 (4 cells, ≈0.7 GPU-h at 2×).
2. **Seed contingency, §15.10 item 2 (data-quality AMBIGUOUS follow-up
   for d=96):** the registered follow-up is "+2 seeds at that d's own
   reserved contingency block, re-fit, re-check" — read literally this
   is d=96's full 5-K-group block (10 more cells), though given d=96's
   own genuinely-flat-not-noisy character (disclosed above), a materially
   more informative use of the same budget may be widening/shifting the
   d=96 K-grid rightward (K>69) rather than adding seeds to a curve that
   is not noisy — **a design decision for whoever builds the follow-up,
   not self-authorized here.**
3. Neither (1) nor (2) is GPU-launched by this harvest — both are queued
   in `STATE.md`.

### Archive

`experiment-runs/2026-07-07_keyanchor_scaling/` (repo-tracked, all files
≤25MB — largest raw is 4.38MB): `results/deltanet_rd_exactness/
wavekeyanchor-scaling/` (30 cell JSONs, `ALL_DONE`, `CALIBRATION_DONE`,
`PROGRESS.txt`, `logs/` — 34 per-cell logs incl. the 3 pre-fix K=20 stub
failures + `smoke.log`), `logs/keyanchor_scaling_run{1,2}.log` (full
chain sessions), `fit_cliff_curve_d{80,96}_results.json` (box's own fit
output, reproduced independently), `REV7_THRESHOLD_PINNED_D{80,96}.json`,
`smoke_dstate_kernel_result.json` (kernel-safety gate artifact),
`scripts/` (`fit_cliff_curve.py`, `sim_cliff_power.py`,
`keyanchor_scaling_chain.sh`, `smoke_keyanchor_scaling.py` — all
byte-verified zero-drift against both the box's live copies and the
repo's already-committed copies). SSD mirror, full superset, same tree:
`/Volumes/1TB_SSD/learned-representations/experiment-runs/
2026-07-07_keyanchor_scaling/`.

---

## §15.20 DESIGN — d=96 wider-K cliff-hunting wave + d=80 seed escalation +
`fit_cliff_curve.py` admissibility-filter fix (resolves §15.19's AMBIGUOUS
verdict; design-only, zero GPU spent building anything; one CPU-only power
check now landed, MAJOR-2 below) (Rev 1, 2026-07-07 — post-attack)

**Rev 1 status note.** An independent adversarial attack round reviewed this
design before any GPU work launched, per this program's own standing
multiple-independent-audit-rounds discipline (`CLAUDE.md`). Verdict:
**NEEDS-REVISION** — 4 MAJOR, 4 MINOR, plus an independently CONFIRMED check
that the two rival CI bands (§15.20.4) really are disjoint (`[0.718,0.739]`
vs `[0.768,0.837]`). Every finding is fixed below (§15.20.1–§15.20.6); none
is deferred or waved away. The full finding→fix trace is recorded in §15.21,
per house style (mirrors `REASONING_LINK_DESIGN.md` §16.7's own fix-map
convention). Nothing in §15.20.1–§15.20.6 below predates this revision —
read this section as Rev 1, not as the Rev-0 draft the attack round
reviewed. One fix (MAJOR-2) required actually running a CPU-only script
before this text could be finalized — that run is now landed, its artifact
is `matrix-thinking/deltanet_rd/sim_cliff_power_wide_grid_results.json`, and
its numbers are reported at §15.20.4 rather than projected. **This
revision has not yet had its own independent audit pass** — per this
project's standing rule that multiple independent adversarial rounds catch
different bugs each round, landing attack-round-1's findings does not, on
its own, certify Rev 1 as build-ready.

**Status: DESIGN-ONLY DRAFT** (Rev 1), written under the same discipline
§15's own header required (§15.17's self-attack-round precedent) — every
number below is either cited to an existing, already-run artifact (§15.11's
cost model, §15.19's harvest), freshly VERIFIED this session against the
live code (file + line, stated explicitly), or freshly MEASURED this
session by an actually-run CPU script (§15.20.4's MAJOR-2 fix) — never
assumed by analogy.

### 15.20.0 What this wave resolves, and what it explicitly does NOT reopen

§15.19's own mechanical verdict (WAVE VERDICT: AMBIGUOUS) is **frozen, not
revisited** — this design does not rescore or rerun any of the 30 already-
harvested cells' own Outcome assignment (§15.1's own "never reopens or
rescores" precedent, restated here for this wave). What §15.19 left
genuinely open, and what this design answers:

1. **d=96 has no located cliff in `[K/d ∈ 0.25, 0.71875]`** — the curve is
   flat-near-ceiling (h4 0.9805–1.0) across the entire tested window, not
   noisy (§15.19's own "Reading the two d's honestly" section). §15.19's own
   registered next-step language (item 2, "queued next steps") explicitly
   INVITES a design choice between firing the literal 10-cell seed-
   contingency block at the existing K's or **widening/shifting the K-grid
   rightward** — flagged there as "a design decision for whoever builds the
   follow-up, not self-authorized here." **This design makes that call:
   widen, not re-seed** (§15.20.1), because more seeds cannot localize a
   transition that the existing 5 points show no sign of containing; a wider
   K only costs the same 12-cell budget as a full 5-K-group seed-contingency
   block would (§15.14's literal reading) and is strictly more likely to
   find the actual x0(96).
2. **The mechanical seed-contingency trigger already fired at two d=80
   K-groups** (K=48, K=53 — §15.19's own trigger table, ranges 0.2183 and
   0.1779, both >0.15) and is queued, not yet run. This design fires it
   (§15.20.2), using the ALREADY-RESERVED seeds from §15.15's own table —
   no new seed allocation needed there.
3. **`fit_cliff_curve.py` does not filter `geo3_admission.admissible`** —
   verified THIS session (§15.20.3: `grep -n "admissible" fit_cliff_curve.py`
   returns zero hits) — the ONE non-admissible cell §15.19 found (K=69/d=96/
   seed=1730) was excluded by MANUAL post-hoc inspection, not by the script.
   §15.9's own registration ("every cell must show `admissible: true`...")
   is currently enforced by human vigilance only. This design registers the
   mechanical fix.
4. **Rival discrimination is not yet pre-registered for the NEXT
   measurement.** §15.19's own rival table (§15.0/§15.10) already refuted
   ratio-invariance and fixed-K cleanly; "absolute-slack" is the sole named
   survivor but was never fit to a real, clean d=96 point. This design
   pre-registers, in numeric-band form, how a genuinely-located x0(96) from
   the widened grid will discriminate absolute-slack from a power-law
   alternative (§15.20.4) — BEFORE that data exists, mirroring §15.0's own
   "no third ambiguous bucket left undefined" discipline.

**Scoping note, stated explicitly so this is not mistaken for a re-run of
§15.10's own primary criterion:** the original ratio-invariance band
`[0.4745, 0.6165]` and its CONFIRMED/REFUTED/AMBIGUOUS machinery are NOT
re-applied here — d=80 already excluded that band with a real margin, and a
fifth d=96 point still inside `[0.25, 0.9375]` cannot change that fact. This
wave's own success criterion is the NEW rival-discrimination test in
§15.20.4, a different, later question in the same research thread.

---

### 15.20.1 d=96 wider-K grid

**Grid, mechanical construction:** `K ∈ {72, 78, 84, 90}` (ratios `0.75,
0.8125, 0.875, 0.9375` — all exact fractions of 96: `72/96=3/4`,
`78/96=13/16`, `84/96=7/8`, `90/96=15/16`, verified by direct division),
3 seeds each = **12 new GPU cells**. The existing K=69 cells (ratio
0.71875, §15.19's own already-harvested data) are REUSED as the grid's low
edge — **zero new GPU-h**, per the task's own framing.

**Why this grid, not a re-seed of the existing 5 points (§15.20.0 item 1):**
absolute-slack's own d=80-recalibrated prediction (§15.20.4: x0(96)≈0.730)
and a power-law extrapolation through the two clean points (§15.20.4:
x0(96)≈0.805) both land ABOVE the previously-tested ceiling (K/d=0.71875)
and WITHIN this new grid's own span (0.75–0.9375) — the grid is sized to
bracket both named-rival predictions with real points on both sides of each,
not merely to push further right by an arbitrary amount.

**Pool-bound and K_min-floor check, mechanical (mirrors §15.4's own
arithmetic):** train pool=107, heldout pool=106 (unchanged, `d_state`-
independent, §15.4). Max new K=90: margin vs train pool 17, vs heldout pool
16 — comfortably positive, same conclusion class as §15.4's own d=96 check
(margin 37/38 at the OLD max K=69, now 16/17 at the new max — still never
binding). **K_min floor:** no registered minimum-K constant exists anywhere
in `grammar_rd.py`/`key_anchoring.py`/`run_deltanet_rd.py` (grepped this
session, zero hits for `K_MIN`/`MIN_K`/an explicit `K >=` assertion) — the
only floor-shaped constraints in this program are (a) the pool bound above
and (b) the H_extra/H_train residue-collision guard (below), both checked
directly; K=72 is so far above either that "K_min floor" is satisfied
trivially, not by omission.

**H_extra residue check, mechanical, EVERY new K (mirrors §15.15's own
addendum exactly, the K=20 hop-collision fix):** the unmodified CLI default
`H_extra=(7, 21)` collides with `H_train=(1,2,3)`'s own residue set only
when `h % K ∈ {1,2,3}` for `h∈{7,21}`. **Since all four new K's (72, 78, 84,
90) exceed 21, both `7 % K = 7` and `21 % K = 21` for every one of them** —
no modulo wraparound occurs at all (both residues equal the literal hop
value, never reducing into `{1,2,3}`), so **no `H_extra` override is needed
at any new K** — a structurally different, safer situation than K=20's own
(K=20 < 21, forcing a wraparound). `H_test=[4,5,6]` is unaffected for the
same reason. **Reused K=69 needs no override either** (already verified
clean in §15.19's own per-cell audit: `[7,21]` unmodified, zero collision).

**NEW finding — `GATE2_N_ITER_BY_D_K[96]` collision at K=84, structurally
identical to §15.6's own K=48/d=80 finding, not previously possible to hit
because d=96's ORIGINAL grid (§15.3: `{24,51,57,63,69}`) never included
K=84:** verified this session, `key_anchoring.py` lines 65–90 — the SHARED,
wave-agnostic flat dict `GATE2_N_ITER_BY_K` already contains `84: 20`
(added at §13.2/13.3 item 5 for `d_state=128`, verified ONLY at d=128 via
`keyanchor_dstate_niter_check.py`). This wave's own K=84/d=96 addition is a
**silent-collision risk of the exact same class §15.6 closed for K=48/d=80**
— if any code path calls `gate2_construction_check(table, ks=...)` with 84
included at d=96, `gate2_ns_leg`'s own `n_iter=GATE2_N_ITER_BY_K[K]` lookup
(line 266) reads the flat dict's d=128-verified value, never checking
whether n_iter=20 also converges at d=96's own Welch-floor geometry (which
DIFFERS continuously with d, §15.6's own already-registered reasoning).
**The fix already exists in the codebase and generalizes trivially:**
`run_deltanet_rd_exactness_sweep.py`'s `KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K`
(verified this session, lines 2604–2607: currently `{80: {20,43,48,53,58:
20}, 96: {24,51,57,63,69: 20}}`) is the d_state-namespaced dict every
`keyanchor_scaling_*` manifest builder ALREADY consults instead of the flat
one (§15.6's own comment at `key_anchoring.py` line 89 states this
explicitly). **Required build task:** extend
`KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[96]` with `{72: 20, 78: 20, 84: 20,
90: 20}` — all "by analogy only," same status as every other entry pending
its own Wave −1 sufficiency check (below); K=84 is flagged with the SAME
"even though it already has an entry" caveat §15.6 item 3 used verbatim for
K=48 — **do NOT let K=84's own d=96 cells read `ka.GATE2_N_ITER_BY_K[84]`
(the flat dict), which is d=128-verified only.**

**Wave −1 n_iter-sufficiency check, mandatory, mirrors §15.6 item 3 exactly:**
for all 5 new-or-reused (K, d=96) pairs actually exercised by this wave's own
Gate-2 construction check (`{72,78,84,90}` — K=69 does not need re-checking,
already verified clean in the original wave), compute pooled pairwise
post-Newton-Schulz cosine at `n_iter∈{12,16,20,24}`, confirm `rel_change
20→24 < 0.5%` before trusting `n_iter=20`. **Negative unit test (mirrors
§15.6 item 4):** assert `(K=84, d_state=96)`'s own sufficiency-check result
artifact is distinct from `(K=84, d_state=128)`'s existing one — a build
regression that silently reused the d=128 result must be mechanically
catchable.

**NEW finding — kernel long-T gap, not previously registered, closes a real
extrapolation the original §15.2 gate never covered:** verified this
session, `grammar_rd.py` lines 325–327 (`T_bind` property) and the
`assert cfg.T_bind == 56` regression check at line 552 (K=8, `clause_len=7`
— confirms `T_bind = K × clause_len` with `clause_len=7` fixed,
architecture-wide, `d_state`-independent): **`T_bind(K) = 7K`.** The
ORIGINAL kernel-safety protocol (§15.2 item 1, re-run to completion per
§15.18's FATAL-2 resolution) tested `T ∈ {128, 224, 448}` — **T=448 is its
own registered ceiling.** This wave's new max K=90 drives `T_bind = 630`,
**41% beyond that tested ceiling**, and even K=72 already reaches `T_bind =
504` (12% beyond). **Disclosed, non-dispositive precedent:** the
already-harvested K=69 cells (`T_bind = 483`, 8% beyond 448) ran 15/15
clean in §15.19 with zero CUDA-crash reports — real production evidence
this direction of extrapolation is likely low-risk (the historical D=32/D=16
crash mode was specifically a SHORT-T / small-D padding-floor interaction,
`_MIN_KERNEL_T=128`-adjacent, not a long-T failure mode) — **but per this
program's own repeated "verify, don't assume by analogy" discipline
(§15.2's own header, restated at §15.18 FATAL-2's own resolution: D=32's
crash was itself a non-monotonic surprise between two verified-safe D
values), an untested T is not licensed by proximity to a working one.**
**Required Stage −1 addendum, BLOCKING, cheap (same fresh-subprocess-per-
probe harness, sub-minute per probe, no measurable GPU-h):** extend the
kernel-safety probe to `T ∈ {504, 546, 588, 630}` (this wave's own new
`T_bind` values) at `d_state=96` specifically, forward+backward,
`chunk_delta_rule` called directly — same PASS criterion as §15.2 item 1
(zero CUDA illegal-memory-access / autotuner crashes). **d=80's escalation
cells (§15.20.2) need no new T check** — K=48/53 are already-tested K's
(`T_bind = 336/371`, both cells already ran 3 real seeds each, well within
even the ORIGINAL 448 ceiling).

**Seed table (fresh, collision-verified this session by grepping every
`_s<digits>_` filename token across the ENTIRE repo, `experiment-runs/`
included, excluding worktrees/team-runs scratch copies): the highest seed
ever used OR reserved anywhere in the KEY_ANCHORING program's history is
1735 (K=69's own Gate-1 probe slot, §15.15, never fired). No seed in
`[1736, 2100)` appears in any filename or any registered seed table in
either `KEY_ANCHORING_DESIGN.md` or this file** (REASONING_LINK's own seed
scheme, checked too, is disjoint from this program's own — **structurally,
not merely by magnitude, corrected at Rev 1 attack-round-1 MINOR-4**:
`REASONING_LINK_DESIGN.md`'s `episode_seed` formula, lines 1449–1464, sums
`PURPOSE_BASE ∈ {0, 10_000_000, 20_000_000}` + `LEG_BASE ∈ {0, 5_000_000}`
+ `condition_idx·1_000_000` + `corpus_idx·100_000` + `ckpt_seed_idx·10_000`
+ `k_idx·1_000` — EVERY term in that sum is an exact multiple of 1,000, so
`episode_seed ≡ 0 (mod 1000)` for every REASONING_LINK seed that will ever
be emitted, by construction, not by the current numeric range. This
program's own 100-wide blocks (1730, 1740, 1840, 1940, 2040, …) never land
on a multiple of 1000 — verified directly, `1730 mod 1000 = 730`, `1740 mod
1000 = 740`, `2040 mod 1000 = 40`, and every other reserved base in
§15.15/§15.20's own tables shares this property (each block's own `+40`
offset from a `X700/X800/X900/Y000`-style round base is never itself a
multiple of 1000). **The two seed spaces are disjoint by residue class,
not merely by the two programs' allocations happening to sit six orders of
magnitude apart today** — a guarantee that survives even if either
program's own seed range were to grow substantially, which the
magnitude-only framing this design originally used did not establish. (The
six-orders-of-magnitude fact is still true and disclosed as a secondary,
consistent cross-check — it is simply not the LOAD-BEARING reason
collision is impossible.) **Continuing this program's own 100-wide-block,
+0/+1/+2 primary / +3/+4 contingency / +5 Gate-1-probe convention exactly
(§15.15's own table), starting immediately after the last reserved slot:**

| K (d=96) | ratio | primary seeds | +2 contingency | Gate-1 probe |
|---|---|---|---|---|
| 69 (reused, already run) | 0.71875 | 1730,1731,1732 (1730 excluded, admissibility) | 1733,1734 (not fired) | 1735 (not fired) |
| 72 | 0.75000 | 1740,1741,1742 | 1743,1744 | 1745 |
| 78 | 0.81250 | 1840,1841,1842 | 1843,1844 | 1845 |
| 84 | 0.87500 | 1940,1941,1942 | 1943,1944 | 1945 |
| 90 | 0.93750 | 2040,2041,2042 | 2043,2044 | 2045 |

**Build task — new module constants, mirrors §15.15 item 2 exactly:**
`KEYANCHOR_SCALING_D96_WIDE_KS = (72, 78, 84, 90)` (a NEW constant — the
ORIGINAL `KEYANCHOR_SCALING_D96_KS = (24,51,57,63,69)` is left untouched,
preserving §15.19's own harvest byte-for-byte reproducible from the
existing manifest calls) plus a wrapper `keyanchor_dstate_manifest(
d_state=96, Ks=KEYANCHOR_SCALING_D96_WIDE_KS)`. **New `--wave` choice:**
`"keyanchor-scaling-wide"`, appended to the existing choices list
(mirrors §15.15 item 4).

**Reusing K=69 in the fit input, mechanical, no `fit_cliff_curve.py` code
change (preserves §15.15 item 7's "NO build task" status for that script):**
`load_k_mean_h4` (§15.20.3) takes exactly ONE `cell_dir` argument, applied
uniformly to every K in `--k-grid` — it cannot pull K=69 from the ORIGINAL
wave's directory while pulling K=72–90 from this wave's own new directory
in a single invocation. **Required Stage −1 step (mechanical, not a code
change):** copy the 3 existing K=69/d=96 result JSONs (seeds 1730/1731/
1732) into this wave's own `wavekeyanchor-scaling-wide/` output directory
before running the fit.

**ENFORCED sha256 gate, named mechanism (Rev 1, attack-round-1 MAJOR-1
fix — a prose "verify byte-identical" instruction is not itself a check
anything can fail; this closes that gap the same way §15.18 FATAL-2 closed
the documented-but-unenforced kernel gate):** immediately after the copy,
`keyanchor_scaling_wide_chain.sh` (this wave's own chain script, built from
`keyanchor_scaling_chain.sh`'s pattern per §15.20.1's `--wave
keyanchor-scaling-wide` build task) runs a dedicated step,
`sha256sum <copied K=69 files> | diff - <pinned manifest>`, where the
pinned manifest is a fixed block committed to this wave's own build
artifact (`results/keyanchor_scaling_wide_k69_copy_manifest.sha256`,
generated ONCE against the ORIGINAL §15.19 archive at build time, never
regenerated from the copy itself — regenerating from the copy would make
the check tautological). **Fail-loud: any hash mismatch, or any of the 3
expected files missing from the copy, exits 1 and halts the chain before
any new GPU cell launches** — mirrors `keyanchor_scaling_chain.sh`'s own
GATE 1 kernel-safety belt-and-suspenders pattern (bash-level standalone
check first, so a direct Python invocation bypassing the chain cannot skip
it either — the Python-side `keyanchor_scaling_stage_gate` equivalent for
this wave must ALSO refuse to proceed to the fit step if the pinned-hash
sentinel file the bash gate writes on success is absent, same
belt-and-suspenders shape as the existing kernel gate). **Added to the
§15.20.6 enforced-gate table below as Gate (c).** A copy that silently
diverges from the archived original would corrupt the reused low-edge
point without any other check catching it — this was previously a
documented step relying on human vigilance, exactly the class of gap
§15.18 FATAL-2 already found and closed once for the kernel-safety check;
it must not be reopened by omission here. **This ties directly to
§15.20.3's fix below:** the copied K=69/seed=1730 file's
`geo3_admission.admissible=false` will now be caught AUTOMATICALLY by the
fixed loader, closing the loop on the one place this wave's own manual
exclusion (§15.19) would otherwise need to be re-applied by hand a second
time.

---

### 15.20.2 d=80 seed escalation (fires the already-registered §15.14 trigger)

**Mechanical citation, no new derivation needed:** §15.19's own trigger
table already fired at exactly two d=80 K-groups — K=48 (3-seed h4 range
0.2183) and K=53 (range 0.1779), both exceeding §15.14's own registered
0.15 threshold, sitting exactly on the two K-groups straddling d=80's own
fitted transition (`x0=0.6756`, between K/d=0.60 and 0.6625). Per §15.14's
own rule ("+2 seeds at the affected K-group"), **the seeds are ALREADY
RESERVED — no new allocation** (§15.15's own table): K=48's contingency
pair is `1133, 1134`; K=53's is `1233, 1234`. **4 new GPU cells, zero new
seed-collision risk to check** (these slots have existed, unfired, since
the wave's original build).

**No new kernel, Gate-2, or H_extra risk (§15.20.1's own findings do not
apply here):** K=48 and K=53 are already-verified K's at d=80 — `n_iter=20`
already confirmed sufficient there (§15.6/§15.19's own per-cell audit), the
kernel gate already covers d=80 at these K's (`T_bind` 336/371, both well
inside the original 448 ceiling, and 3 real seeds each already ran clean),
and `H_extra=[7,21]` is already confirmed collision-free at both K's
(neither is K=20, the only K in this program's history to need an
override). **This escalation is the cheapest, lowest-risk item in this
entire design** — it re-runs an already-proven configuration at 2 more
seeds per group, nothing structurally new.

**Re-fit, mechanical:** re-run `fit_cliff_curve.py --d-state 80 --k-grid
20 43 48 53 58 --n-trials 4000` with `n=5` seeds at K=48/53 (was `n=3`) once
the 4 new cells land; report the revised `x0(80)` CI alongside the original
`[0.6620, 0.6868]` — **not expected to change the already-clean REFUTE
verdict** (degenerate_frac was already 0.00% at `n=3`; this tightens the CI,
it does not re-open the data-quality gate), but run and reported, not
assumed.

---

### 15.20.3 `fit_cliff_curve.py` admissibility-filter fix (build task +
regression test)

**Verified this session, exact and complete:** `grep -n "admissible"
matrix-thinking/deltanet_rd/fit_cliff_curve.py` returns **zero matches**.
`load_k_mean_h4` (lines 122–145, **refreshed at Rev 1 attack-round-1
MINOR-2** — the original draft cited 120–142, stale by the time of the
attack-round re-check) is the SINGLE choke point every K's data passes
through (anchored K=32/K=48 AND every unanchored `--k-grid` entry, verified
via `main()`'s own call sites, lines 303–310 — refreshed from the stale
300–307) — it checks `d.get("complete")` only (line 137, refreshed from
the stale line 135) before appending `h4` to `per_seed`, never touching
`d.get("geo3_admission")` at all. §15.9's own registration
("every cell must show `admissible: true`... no new fields, no new logic,
the same instrument every prior wave's own admissibility check used") is
**not actually enforced by this script** — §15.19's own K=69/seed=1730
exclusion was found and applied by a human reading the per-cell verification
table, not by the fitting script itself.

**Program-wide impact check, run this session (NOT hypothetical — every
`armd`-tagged, `complete=true` JSON across every KEY_ANCHORING wave's own
archive was scanned for `geo3_admission.admissible != true`):** exactly ONE
non-admissible candidate-(d) cell exists in the program's entire history —
the already-found K=69/d=96/seed=1730. **Zero prior wave's own headline
CONFIRMED/REFUTED verdict was silently corrupted by this gap** (§12's d=64
cliff, §13's d=128 dstate, §14's dose-response all had 100% admissible
candidate-(d) pools) — this is a real, disclosed instrument gap, not
evidence of a wrong prior result. The 3 non-admissible cells also found in
the scan (`wavekeyanchor-k48-ref`, `armgeoref` tag) are reference-arm cells,
never eligible for `load_k_mean_h4`'s own glob (`arm="d"` default, and
`assert_no_reference_arm_paths` would raise before they were ever read) —
irrelevant to this fix, cited only for completeness.

**Fail-closed default verified safe, not merely convenient:** every
`complete=true`, `arm="d"`-tagged JSON found across the whole archive HAS a
`geo3_admission` block (0 missing, checked this session on all 140 scanned
JSONs) — a fixed loader that treats a MISSING block as non-admissible will
never spuriously exclude any real historical cell.

**Required build task, exact, minimal (single-point fix, mirrors this
project's own "one instrument, reused everywhere" discipline):** in
`load_k_mean_h4`, immediately after the existing
`if not d.get("complete"): continue` (line 137, refreshed at Rev 1
attack-round-1 MINOR-2), add:
```python
if d.get("geo3_admission", {}).get("admissible") is not True:
    continue
```
`admissible` alone is provably sufficient — verified this session,
`run_deltanet_rd.py` line 562: `admissible = bool(ns_converged_no_fallback
and value_salvage_tier_pass and finite_loss_no_divergence and
task_performance_floor_pass)`, and line 556:
`ns_converged_no_fallback = (n_geo3_fallback_train_steps == 0) and not
ckpt_fallback_seen` — the single boolean already ANDs together every field
§15.9 separately listed; checking it alone is exactly equivalent, not a
weaker proxy.

**Negative test 1 (real regression fixture, the strongest available check
— reuses ground truth this session already hand-computed, no synthetic
data needed):** re-run the FIXED `fit_cliff_curve.py` on the
ALREADY-ARCHIVED, unmodified §15.19 raw directory
(`experiment-runs/2026-07-07_keyanchor_scaling/results/deltanet_rd_exactness/
wavekeyanchor-scaling/`, `--d-state 96 --k-grid 24 51 57 63 69`) with **no
manual pre-filtering this time** — assert the fixed loader silently drops
`wkeyanchor-scaling_..._K69_..._s1730_..._d96.json` on its own, and assert
the resulting `degenerate_frac` matches §15.19's own manually-computed
sensitivity check to full precision: **94.77% (not 98.52%)**, `x0` still
pinned at the 0.9 bound. A fixed script that does NOT reproduce this exact
number has a bug in the fix itself, not merely an untested code path.

**Negative test 2 (synthetic, from-scratch, mirrors §12.4 item 5's own
ZERO-REFERENCE-ARM-PATHS negative-test convention):** construct a
minimal 3-seed K-directory with one fabricated `admissible: false,
complete: true` JSON (arbitrary `h4` far from the other two seeds' mean —
e.g. `h4=0.05` when the other two average ~0.95) mixed in; assert (a)
`per_seed` has length 2, not 3, (b) the returned mean excludes the
fabricated value (i.e., is NOT dragged toward 0.05). This is the
"assertion has teeth" check this project's own Hard Rules require — must be
RUN to completion, not merely written.

**Regression guard against the ANCHORED (d=64) path, mirrors §15.15 item 6:**
since `load_k_mean_h4` is shared by the K=32/K=48 archived-directory loads
too, re-run `fit_cliff_curve.py --d-state 64 --k-grid 34 38 42 46 --k32-dir
... --k48-dir ...` (§12.9's own exact invocation) with the fix applied;
assert byte-identical output to the ALREADY-COMMITTED `fit_cliff_curve_
results.json` (§12.9's own archived artifact) — confirms the fix does not
silently change the d=64 headline result (expected, since that archive's
own candidate-(d) pool was already 100% admissible, verified above), closing
the one regression risk a careless implementation of this fix could
introduce.

---

### 15.20.4 Distinguishing the rivals — pre-registered discrimination test

**Two named rivals remain live after §15.19** (ratio-invariance and
fixed-K are both already cleanly refuted, §15.19's own rival table):

**Absolute-slack** (`S = d − K_crit ≈ const`, §13.10/§14.13's own surviving
candidate): using the NEWLY MEASURED, clean d=80 point (`x0(80)=0.6756`,
replacing the pre-d=80 d=64-only estimate) to recalibrate `S`:
`K_crit(80) = 0.6756 × 80 = 54.048` — **kept unrounded from here on (Rev 1,
attack-round-1 MINOR-3 fix: the original draft rounded `K_crit(80)` to 54
before subtracting, an unnecessary detour that silently shifted the point
prediction).** `S = 80 − 54.048 = 25.952`. **Predicted `x0(96) = (96 −
25.952) / 96 = 70.048/96 = 0.729667`** (vs. the rounded detour's
`0.7292` — a small, real difference, direction disclosed: the unrounded
figure is the correct one and sits slightly HIGHER). Propagating the CI
corners (`x0(80) ∈ [0.6620, 0.6868]`, `x0(64) ∈ [0.53915, 0.55185]`, the
latter from §15.10's own `δ64` half-width) through the SAME `S`-formula's 4
corner combinations gives a band of **`[0.718, 0.739]`** (worked example, low
corner: `S = 80×(1−0.6620) = 27.04 → x0(96) = (96−27.04)/96 = 0.71833`; high
corner: `S = 80×(1−0.6868) = 25.056 → x0(96) = (96−25.056)/96 = 0.73900`;
the corner arithmetic was never routed through the rounded-`K_crit` detour,
so this band itself is UNCHANGED by the MINOR-3 fix — only the single
point-prediction sentence was affected).
**Disclosed wobble in the rival itself, not smoothed over:** the OLDER,
d=64-only-anchored estimate (§15.19's own rival table: `x0(96)≈0.697`, from
`S(64) = 64×(1−0.5455) = 29.088`) sits noticeably below this d=80-recalibrated
band — `S` itself drifted from 29.09 (at d=64) to 25.95 (at d=80), an ~11%
shift, meaning absolute-slack's own "constant" is not perfectly constant
between the two already-measured points either. The d=80-anchored `[0.718,
0.739]` band is used as the PRIMARY prediction (closer, more-recent anchor),
with the d=64-anchored 0.697 point disclosed as a consistency check, not a
second independent prediction to average against.

**Power-law** (`x0 = A·d^α`, the natural alternative functional form,
newly named by this design since §15.10.1 already flagged that "some OTHER
monotone-in-d rescaling" could also pass the original invariance band):
an exact 2-point solve through `(64, 0.5455)` and `(80, 0.6756)` (NOT a
regression — 2 points exactly determine 2 parameters, zero residual
degrees of freedom, disclosed explicitly since this is a real limitation,
mirroring §15.10.1's own honesty convention) gives `α = ln(0.6756/0.5455) /
ln(80/64) = 0.21390/0.22314 = 0.9586`, `A = 0.5455/64^0.9586 = 0.01016`.
**Predicted `x0(96) = A·96^0.9586 = x0(80)·(96/80)^0.9586 = 0.6756 ×
1.19097 = 0.8046`.** Propagating the same 4 CI corners through this
formula (using `x0(96) = x0(80)·(x0(80)/x0(64))^β`, `β = ln(1.2)/ln(1.25) =
0.81702` — a constant independent of which corner is chosen, since it
depends only on the two `d` values, verified algebraically this session)
gives a band of **`[0.768, 0.837]`**.

**The two bands are DISJOINT** (`[0.718, 0.739]` vs `[0.768, 0.837]`, a real
gap of 0.029 between them, **independently re-confirmed by attack-round-1**
via fresh re-derivation of both corner sets) — **this is the key design
property that makes the wider grid worth running**: a genuinely
non-degenerate x0(96) fit from this wave's own wider K-grid, IF its own 95%
CI lands cleanly inside one band and excludes the other, discriminates the
two rivals with a real margin, unlike the original ratio-invariance test's
own disclosed vacuousness risk (§15.17 Q2).

**MAJOR-2 fix (Rev 1) — the disjoint-band property alone does not establish
this wave's own grid can ACHIEVE that clean discrimination; measured, not
projected, this session.** §15.20.8 Q2's own TODO asked for a
`sim_cliff_power.py`-machinery re-run at the wave's ACTUAL, heterogeneous
5-point grid (`K=69` reused at `n=2` — seed 1730 excluded on admissibility
— plus `K∈{72,78,84,90}` fresh at `n=3` each). The existing script's
`bootstrap_ci_width`/`simulate_once` take one scalar `n_seeds_new` applied
uniformly to every new K, which cannot represent this heterogeneous grid,
so a purpose-built driver,
`matrix-thinking/deltanet_rd/sim_cliff_power_wide_grid.py`, was written —
importing `sigmoid`/`POOLED_REL_SD`/`NOISE_SD_MULTIPLIERS` from
`sim_cliff_power.py` UNMODIFIED (same discipline as §15.8's own driver-call
precedent) and adding only a heterogeneous-`n` sibling of
`simulate_once`/`bootstrap_ci_width`, with the SAME sigmoid form, pooled
relative-noise model, and degenerate-fit guard thresholds. **Actually run
this session** (`../../.venv/bin/python sim_cliff_power_wide_grid.py`,
`n_trials=4000`, `seed=20260706` reusing §15.8's own registered
power-check seed, `d_state=96`, `include_anchors=False`), artifact
committed at
`matrix-thinking/deltanet_rd/sim_cliff_power_wide_grid_results.json`:

| Truth (x0, w) | ci_width (95%) | projected half-width | degenerate_frac |
|---|---|---|---|
| abs-slack center (0.729667), w=0.03 (sharp) | 0.0380 | 0.0190 | 0.000 |
| abs-slack center (0.729667), **w=0.0597 (measured, d=64's own fitted width)** | **0.0859** | **0.0430** | 0.000 |
| abs-slack center (0.729667), w=0.08 (moderate) | 0.1119 | 0.0559 | 0.000 |
| power-law center (0.804619), w=0.03 (sharp) | 0.0275 | 0.0138 | 0.002 |
| power-law center (0.804619), **w=0.0597 (measured)** | **0.0637** | **0.0319** | 0.000 |
| power-law center (0.804619), w=0.08 (moderate) | 0.0892 | 0.0446 | 0.000 |
| grid-interior control (x0=0.75), w=0.0597 | 0.0763 | 0.0382 | 0.000 |
| upper-edge control (x0=0.90), w=0.0597 | 0.0314 | 0.0157 | 0.836 |

**Pre-registered response rule (derived, not the loose "±~0.036" the
original attack-round framing used — the RIGHT threshold, derived here
from the disjoint-band gap itself):** the trigger is **half of the
`0.029` inter-band gap = `0.0145`** — the largest projected CI half-width
that could still sit entirely inside one named band without touching the
other from that band's own near edge. **Both rival-center truths at the
REALISTIC, measured cliff width (w=0.0597) EXCEED this threshold** by a wide
margin (abs-slack half-width 0.0430, ~3.0× the threshold; power-law
half-width 0.0319, ~2.2×) — **the fire condition is met.** A follow-up
check (same driver, `K=69` restored to `n=3` — the already-reserved,
unfired contingency seed 1733) was ALSO run this session: it narrows the
abs-slack half-width only to 0.0408 and the power-law half-width only to
0.0307 — **a ~4% reduction, nowhere close to closing the gap to 0.0145.**
An even more aggressive check (uniform `n=4` across all 5 K's, a
cost-upper-bound this wave does not propose funding) still leaves
half-widths at 0.0375/0.0287 — still 2×+ the threshold. **Honest
conclusion, disclosed rather than smoothed over: at this program's own
actually-measured cliff width, THIS wave's own mandatory grid — even with
every already-reserved contingency seed fired — cannot reliably achieve
the "CI lands cleanly inside one band, excludes the other" clean
discrimination the original framing implied.** Firing seed 1733 is still
registered as a Stage-1 action (cheap, already-reserved, a real if small
improvement) but must NOT be read as restoring clean discrimination. **The
decision rule below (MAJOR-3 fix) is written to degrade gracefully under
this finding** — a genuinely uninformative BOTH-CONSISTENT read at the
measured-width noise level is the EXPECTED outcome under this power
analysis, not a design failure if it occurs; a follow-up wave sized to
close the gap to 0.0145 (roughly a further ~2-3× in per-K seed count,
extrapolating from the `n=2→n=3→n=4` narrowing rate above) would be a NEW,
separate GPU ask outside this wave's own scope, named here as a live
future decision point rather than silently assumed available.

**Pre-registered decision rule, mechanical, stated BEFORE any new cell
runs (Rev 1, attack-round-1 MAJOR-3 fix — the Step-1 data-quality gate and
outcome 5 are now resolved explicitly rather than conflated; see the table
below):**

| Step | Trigger (evaluated in order) | Verdict | Follow-up |
|---|---|---|---|
| 0 | Fit converges, `degenerate_frac ≤ 10%` | (proceed to steps 2-4 below; steps 1a/1b handle the `>10%` branch) | — |
| 1a | `degenerate_frac > 10%` AND every sampled K's own raw per-seed h4 mean is `≥ 0.98` (the curve never left ceiling anywhere in the window — a FLATNESS signature, not a scatter signature) | **CLIFF-BEYOND-WINDOW** (named, distinct from AMBIGUOUS — see adjudication below) | Report `x0(96) > 0.9375`, lower-bounded only; do not re-seed a flat curve (mirrors §15.20.0 item 1's own "widen, don't re-seed" reasoning) — a future wave would need to widen the K-grid further, an explicit new ask |
| 1b | `degenerate_frac > 10%` AND at least one sampled K's own raw per-seed h4 mean is `< 0.98` (genuine scatter/non-convergence, not flatness) | **AMBIGUOUS** (data-quality gate, noisy-fit) | Seed escalation at the noisiest K-group (§15.14's own trigger convention) |
| 2 | CI(x0,96-wide) overlaps `[0.718, 0.739]` AND excludes `[0.768, 0.837]` | **ABSOLUTE-SLACK FAVORED** | Report; absolute-slack survives as the leading account |
| 3 | CI(x0,96-wide) overlaps `[0.768, 0.837]` AND excludes `[0.718, 0.739]` | **POWER-LAW FAVORED** | Report; power-law survives as the leading account |
| 4 | CI spans or touches both bands (e.g., wide enough to cover `[0.70, 0.80]`) | **BOTH-CONSISTENT** (genuinely uninformative, disclosed as such — the EXPECTED outcome at the measured w=0.0597 noise level per this session's own power check above) | Report as uninformative at this seed budget; do not force a pick |
| 5 | CI excludes both bands entirely, but the fit is non-degenerate and per-K means are NOT all `≥0.98` (a real, located x0 landing somewhere else) | **NEITHER SURVIVES (elsewhere)** | A genuinely new, informative negative result — neither named functional form fits; open a new rival-search question |

**CLIFF-BEYOND-WINDOW adjudication, per rival (Rev 1, MAJOR-3 — replaces
the old outcome-5 text's conflated "e.g. pinned again near the 0.9
fit-bound" example with an explicit reading for each named account):** a
lower-bounded `x0(96) > 0.9375` sits ABOVE both registered numeric bands
(`[0.718,0.739]` and `[0.768,0.837]`), so NEITHER band survives numerically
— but the two rivals' own FUNCTIONAL FORMS are not equally strained by
this outcome. **Absolute-slack is strained more severely:** its own
predicted point (0.729667) would then miss by `> 0.9375 − 0.729667 =
0.208`, an even larger absolute miss than the already-disclosed 11%
`S`-drift between d=64 and d=80 (§ above) — a constant-slack account has no
free parameter left to reconcile a miss this large; it would need to be
retired in favor of "`S` grows with `d`" (a structurally different,
weaker rival this design does not currently register). **Power-law is
strained but not retired:** CLIFF-BEYOND-WINDOW would mean this wave's own
2-point `α=0.9586` fit under-estimated the true exponent, but the
FUNCTIONAL FORM (superlinear-in-`d` growth of `K_crit`) is directly
rescuable by re-fitting `α` with the new, beyond-window x0(96) point added
as a THIRD calibration point — a coherent next step this design names but
does not build. **This distinction (form survives vs. form is retired) is
why CLIFF-BEYOND-WINDOW is registered as its own named verdict rather than
folded into the generic "neither survives" outcome 5** — it is
MORE informative than a bare "neither band" read, and it is a materially
different situation from AMBIGUOUS (1b): a flat, uniformly-near-ceiling
curve (1a) reflects a genuine absence of decline in the tested window, not
noise the seed-escalation follow-up (1b's own remedy) could ever fix by
adding seeds at the SAME K's — this is exactly the distinction the
original single "if degenerate_frac > 10% → AMBIGUOUS" framing collapsed.
**Corroborating evidence from this session's own power-check run (table
above):** the `upper_edge_x090` control truth (x0=0.90, just inside the
grid) already shows `degenerate_frac` climbing to 0.65–0.997 depending on
width — direct, measured confirmation that a cliff sitting at or beyond
the grid's own right edge produces exactly the high-degenerate-fraction
signature step 1a is keyed on, distinguishing it empirically from a
merely-noisy in-window fit (every in-window truth in the table above shows
`degenerate_frac ≈ 0`).

**Why K=90 (ratio 0.9375) is far enough right to make step 5/CLIFF-BEYOND-
WINDOW meaningful, not merely a formality:** if h4 has not started
declining by K/d=0.9375 (bootstrap margin from the pool: only 16 heldout
entities are NOT drawn as keys at that K, `n_query=K=90` per
`grammar_rd.py`'s own default, verified §15.20.1), that is itself
informative — either the cliff needs an even wider window, or the
mechanism is not well-described by ANY of the three named accounts at this
`d_state`, both real findings the design does not pre-judge.

---

### 15.20.5 Budget and ledger treatment

**Real, calibrated per-cell rates, pulled directly from the archived §15.19
raw JSONs this session (a materially TIGHTER estimate than §15.11's own
log-log interpolation, since d=96 and d=80/K=48/K=53 now have REAL realized
data rather than only two-endpoint interpolation):**

| Group | K's | realized mean wall_s/cell | GPU-h/cell |
|---|---|---|---|
| d=96 main-grid (K=51,57,63,69) | 4 | 1508.24s | 0.4190 |
| d=80, K=48 specifically | — | 1271.73s | 0.3533 |
| d=80, K=53 specifically | — | 1383.13s | 0.3842 |

The realized d=96 main-grid mean (0.4190) is within 3% of §15.11's own
pre-registered interpolated point estimate (0.4313) — a real cross-check
that INCREASES confidence the interpolated number is not an
under-estimate for the new, higher-K cells. **This design prices against
the SLIGHTLY HIGHER §15.11 point estimate (0.4313) for the new d=96 cells**
(pessimistic-corner discipline, §12.4b's own precedent), and against the
K-SPECIFIC realized rates for the d=80 escalation (more precise than a
generic point estimate, since those exact two K's already have 3 real
seeds each).

**Cost table, 2× contingency applied throughout (house discipline):**

| Item | Cells | GPU-h/cell | Total (1×) | Total (2×) |
|---|---|---|---|---|
| d=96 wide grid (K=72,78,84,90, 3 seeds) | 12 | 0.4313 | 5.1756 | 10.3512 |
| d=80 escalation, K=48 (+2 seeds) | 2 | 0.3533 | 0.7065 | 1.4130 |
| d=80 escalation, K=53 (+2 seeds) | 2 | 0.3842 | 0.7684 | 1.5368 |
| **Mandatory total (16 cells)** | **16** | | **6.6505** | **13.3010** |

**Mandatory-only, pessimistic 2× bracket: ≈13.30 GPU-h — the load-bearing
go/no-go ceiling number**, per §15.12's own established discipline (the
realized-rate expectation is NOT the ceiling number). (Rev 1, attack-round-1
MINOR-1: `6.6505 × 2 = 13.3010`, not `13.3011` — a bare arithmetic slip in
the original draft, corrected here; every downstream number in this
subsection is recomputed from the corrected `13.3010`.)

**KEY_ANCHORING_SCALING sub-ledger status, verified against `STATE.md`
(2026-07-07 snapshot): 11.7865/21 GPU-h realized, reserve 9.2135/21.**

- **At 1×, this design's own point estimate (6.6505 GPU-h) FITS the
  existing reserve** with 2.5630 GPU-h margin remaining.
- **At 2× (the actual go/no-go number, 13.3010 GPU-h), this design does
  NOT fit** — a shortfall of **4.0875 GPU-h** against the existing 9.2135
  reserve.

**Proposed honest ceiling treatment (mirrors §15.12's own two-tier pattern,
NOT a silent draw-down of the existing reserve past its own pessimistic
bracket):** request an explicit, small, disclosed extension to the
KEY_ANCHORING_SCALING sub-ledger specifically — **`H_scaling_2 = +5.0
GPU-h`** (rounds the 4.0875 shortfall up, leaving ≈0.9125 GPU-h of its own
margin), bringing the sub-ledger's own ceiling from 21 → **26 GPU-h**. This
is a NEW PI-decision, stated as such (mirrors §15's own top-level framing
requiring an explicit reopening decision, not self-authorized by this
design) — **not** an extension of the FULLY EXHAUSTED original 80/80
KEY_ANCHORING ledger, and not a request sized to the nominal "saturate the
cluster" framing (§15.12's own explicit disclosure that this program's
waves should NOT be padded to consume idle GPU-time; the ask here is sized
to this wave's own honest mandatory grid only).

**Distinct enforced sign-off token, Rev 1 attack-round-1 MAJOR-4 fix (this
extension gets its OWN gate, not a reuse of the original wave's own
gate):** §15's own original reopening decision is already enforced by
`KEYANCHOR_SCALING_PI_SIGNOFF=1`, checked BOTH in
`keyanchor_scaling_chain.sh`'s GATE 0 (bash) and in
`run_deltanet_rd_exactness_sweep.py`'s `keyanchor_scaling_stage_gate`
(Python, defense-in-depth against a direct-invocation bypass — the
EXISTING, already-built precedent this fix mirrors). **This `+5.0 GPU-h`
sub-ledger extension is a SECOND, separate PI decision (§15.20.0/this
subsection's own framing: "a NEW PI-decision, stated as such") and
therefore requires its OWN, DISTINCT env-var token,
`KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1`** — registered here as a required
build task for `--wave keyanchor-scaling-wide` (§15.20.1's own new `--wave`
choice), enforced at BOTH the same two points as the existing token:
(a) `keyanchor_scaling_wide_chain.sh`'s own GATE 0-equivalent bash check,
refusing with `exit 1` and a printed explanation if
`KEYANCHOR_SCALING_EXT_PI_SIGNOFF` is unset or not exactly `"1"`; (b) a
Python-side check inside the wide-grid wave's own stage-gate function
(mirrors `keyanchor_scaling_stage_gate`'s own
`os.environ.get("KEYANCHOR_SCALING_EXT_PI_SIGNOFF", "0") != "1"` →
`print(...); sys.exit(1)` pattern verbatim). **Explicitly, by construction,
a stale `KEYANCHOR_SCALING_PI_SIGNOFF=1` left exported in the environment
from launching the ORIGINAL wave must NOT satisfy this new gate** — the
two checks are on two DIFFERENT env-var names, never OR'd together in
either the bash or the Python check, so an operator who signed off on the
original 21 GPU-h reopening cannot accidentally also authorize the +5
GPU-h extension without a second, explicit, separately-named decision on
record. **Added to the §15.20.6 enforced-gate table below as Gate (d).**

**Realized-rate expectation (disclosed, not the ceiling number, mirrors
every prior wave):** this program's own historical realized/ceiling ratios
have ranged 13.6%–112.5% (§15.19 itself was the first wave to land ABOVE
its own 1× point estimate, at 112.5%) — there is no strong prior for
assuming this specific follow-up lands comfortably under its own 1×
estimate either; the disclosed 2.56 GPU-h margin at 1× against the EXISTING
reserve alone should not be read as "the extension probably won't be
needed," given this program's own most recent data point.

**Optional layer, explicitly NOT part of this ask (cut-eligible, lowest
priority, mirrors §15.12/§15.14's own priority order):** 4 new Gate-1
probes (1 per new K, seeds 1745/1845/1945/2045, ≈0.108 GPU-h each at the
0.25× rate) would add ≈0.43 GPU-h (1×)/0.86 GPU-h (2×) if ever wanted —
NOT requested here, since d=96's own Gate-2 is already cleared at every
existing K and the h1 training-health guard has read 1.000000 at all 30
of this program's own realized cells so far; no evidence motivates paying
for this diagnostic layer preemptively.

---

### 15.20.6 Staging and gates

**Stage −1 (Wave −1, zero/near-zero GPU, BLOCKING — nothing below launches
until this completes):**

1. **Kernel long-T supplementary probe (§15.20.1, NEW), `T ∈ {504, 546,
   588, 630}` at `d_state=96`, forward+backward** — mechanical PASS
   criterion identical to §15.2 item 1. **d=80's own kernel gate needs no
   re-run** (§15.20.2's own already-tested-K argument).
2. **`GATE2_N_ITER_BY_D_K[96]` extension** (§15.20.1) + the K=72/78/84/90
   Wave −1 n_iter-sufficiency check, **including K=84 specifically despite
   already having an unrelated d=128 entry** (§15.20.1's own explicit
   caveat).
3. **`fit_cliff_curve.py` admissibility fix** (§15.20.3), both negative
   tests run to completion, the d=64 regression guard confirmed
   byte-identical.
4. **Byte-verified copy of the 3 reused K=69/d=96 JSONs** into this wave's
   own output directory (§15.20.1's own closing step), **ENFORCED via the
   sha256 gate (Gate (c) below, Rev 1 MAJOR-1 fix)** against the pinned
   manifest — not merely checked by hand.
5. **Manifest-regression smoke** (mirrors §15.15 item 6): the new
   `keyanchor_dstate_manifest(d_state=96, Ks=KEYANCHOR_SCALING_D96_WIDE_KS)`
   wrapper call must NOT alter the byte-output of the EXISTING
   `keyanchor_dstate_manifest(d_state=96, Ks=KEYANCHOR_SCALING_D96_KS)` call
   (the original 5-K grid) — confirms the new constant is additive, not a
   silent replacement.
6. **`smoke_keyanchor_scaling.py`-style suite re-run**, extended to cover
   the new K-grid and the new `KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K[96]`
   entries (mirrors smoke 12/13's own existing coverage pattern).
7. **`sim_cliff_power.py` re-run at the wave's ACTUAL grid — PROMOTED to
   Stage −1 BLOCKING, Rev 1 attack-round-1 MAJOR-2 fix (previously an
   un-run §15.20.8 Q2 TODO). CPU-only, COMPLETE this session, not merely
   scheduled:** `matrix-thinking/deltanet_rd/sim_cliff_power_wide_grid.py`
   (new purpose-built driver, `sim_cliff_power.py`'s own functions reused
   unmodified) run at `K_seeds={69:2, 72:3, 78:3, 84:3, 90:3}`, `d_state=96`,
   `n_trials=4000`, `seed=20260706`; artifact committed at
   `sim_cliff_power_wide_grid_results.json`. **Result: at this program's own
   measured cliff width (w=0.0597), both rival-center projected CI
   half-widths (0.0430 abs-slack, 0.0319 power-law) exceed the derived
   `0.0145` half-gap trigger — full numbers, the pre-registered response
   rule, and the CLIFF-BEYOND-WINDOW adjudication are at §15.20.4.** This
   item BLOCKS Stage 1 exactly like items 1-6 — it is listed last here only
   because it was the last one actually executed, not because it is lower
   priority; all 7 Stage −1 items are equally BLOCKING.

**Enforced gate table, not merely documented prerequisites (per this
program's own established discipline — §15.17 Q1's own TODO, closed for
the ORIGINAL kernel gate at §15.18's FATAL-2 resolution: "the chain script
must mechanically require this PASSING artifact before launch" — extended
here to this wave's own FOUR gates, Rev 1: (a)/(b) already registered,
(c)/(d) added by attack-round-1 MAJOR-1/MAJOR-4):**

| Gate | Mechanism | Enforcement points | Failure mode |
|---|---|---|---|
| (a) Kernel long-T safety | `T∈{504,546,588,630}` forward+backward probe artifact, PASS required | bash belt (`keyanchor_scaling_wide_chain.sh` standalone check) + Python suspenders (`keyanchor_scaling_stage_gate`-equivalent) | Missing artifact or reported FAIL → refuse to launch ANY d=96-wide cell |
| (b) `GATE2_N_ITER_BY_D_K[96]` sufficiency | K∈{72,78,84,90} `n_iter=20` convergence-check artifact, PASS required | Same belt+suspenders pair as (a) | Missing artifact or convergence failure → refuse to launch |
| (c) K=69 reused-JSON integrity (**NEW, MAJOR-1**) | `sha256sum` of the 3 copied K=69 files diffed against a manifest pinned at build time against the ORIGINAL §15.19 archive (never regenerated from the copy) | bash belt (dedicated chain step, `exit 1` on mismatch) + Python suspenders (fit step refuses without the bash gate's own success sentinel) | Any hash mismatch or missing file → fail loud, halt before the fit runs |
| (d) `+5.0 GPU-h` extension sign-off (**NEW, MAJOR-4**) | `KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1`, a DISTINCT env var from the original wave's `KEYANCHOR_SCALING_PI_SIGNOFF` — never OR'd with it | bash GATE 0-equivalent in `keyanchor_scaling_wide_chain.sh` + Python-side check in the wide-grid wave's own stage-gate function | Unset, or only the STALE original-wave token set → refuse with `exit 1` / `sys.exit(1)` before any cell (including Stage 0 calibration) launches |

A documented-but-unenforced gate is exactly the gap this program has
already found and closed once (§15.18 FATAL-2); it must not be reopened by
omission here — this table is the single place all four of this wave's own
gates are enumerated, per attack-round-1's own instruction to record them
"in the §15.20.6 enforced-gates list + the gate table."

**Stage 0 (calibration, mandatory house rule):** ONE cell, the cheapest new
K in the d=96 wide grid — `K=72`, seed 1740. **Blinded readout** (`wall_s`
only, `h4` quarantined until the full manifest is generated, §13.5's own
F9-fix helper reused verbatim). **Abort/re-price trigger:** realized
`wall_s ≥ 1.5 × 0.4313 × 2 × 3600 = 4658.1s` → halt, diagnose
(`nvidia-smi` contention check first), re-price the full §15.20.5 table
before continuing. d=80's escalation cells need no separate calibration
cell (already-proven K's, §15.20.2) — launch directly under Stage 1.

**Stage 1 (remaining mandatory grid, 15 cells: 11 remaining d=96-wide +
4 d=80 escalation):** launched in parallel batches, per-cell and
running-projection abort rules carried verbatim from §15.14 (this wave's own
bracket numbers: 4658.1s main d=96 threshold, unchanged from §15.14's own
main-grid figure since the per-cell rate is unchanged; running-projection
cut priority unchanged — there is no optional layer registered for this
wave to cut into except the explicitly-not-requested Gate-1 probes,
§15.20.5, which simply never launch if projection runs hot).

**Pre-launch contention re-check (mirrors §15.13's own re-verification,
not an assumed-current read):** confirm GPUs 2–7 are idle (or account for
whatever concurrent program is running) at actual launch time, not at this
design's own drafting time.

---

### 15.20.7 Standing constraints (restated, apply unchanged)

- Exact thresholds, no numerical-tolerance slack (the admissibility-filter
  fix's `is not True` check, §15.20.3; the manifest byte-regression checks,
  §15.20.1/§15.20.6).
- Negative unit tests run to completion, not merely written (§15.20.3's two
  negative tests + the d=64 regression guard; §15.20.1's GATE2 negative
  test).
- Smoke test every model incl. eval batch before launch, extended suite
  (§15.20.6 Stage −1 item 6).
- tmux + supervisor launch pattern, enforced gate branches (§15.20.6).
- Archives to repo (≤25MB) + SSD mirror, both, no exceptions.
- Multiple independent adversarial audit rounds — this section is
  explicitly DRAFT pending its own attack round(s) before
  CLEARED-FOR-BUILD, same as every §15.x subsection before it.

---

### 15.20.8 ATTACK-ROUND QUESTIONS — ROUND 0 (self-attack, minimum 5)

**Q1. Is the disjoint-band discrimination test (§15.20.4) actually as clean
as claimed, or does it quietly assume the new fit will be non-degenerate —
the SAME assumption that failed at d=96 the first time?** **RESOLVED at
Rev 1 (attack-round-1 MAJOR-3):** no longer a bare disclosed risk — the
decision-rule table now names the flat/ceiling failure mode explicitly as
**CLIFF-BEYOND-WINDOW** (step 1a), distinct from noisy-scatter AMBIGUOUS
(step 1b), triggered specifically when `degenerate_frac > 10%` AND every
sampled K's raw per-seed h4 mean is `≥0.98`. §15.20.4 now also adjudicates
what a beyond-0.9375 cliff means for EACH rival (absolute-slack strained
past rescue; power-law's functional form survives via a re-fit `α`), and
this session's own power-check run (§15.20.4 MAJOR-2) empirically
corroborates the trigger's own diagnostic signature: the `upper_edge_x090`
control truth shows `degenerate_frac` climbing to 0.65–0.997 while every
in-window truth stays near 0. The original text below (Q1's own pre-Rev-1
TODO) is left as the historical record of what the attack round was
reacting to.

*(Original, pre-Rev-1 text, preserved per house "record history, don't
retcon it" convention):* yes, this is a real, unresolved risk, not fully
closed by this design. If the new d=96-wide curve is ALSO flat (h4 stays
near 1.0 all the way to K/d=0.9375), the fit degenerates the same way
§15.19's own fit did, and NEITHER band gets tested — outcome 5 (§15.20.4)
covers this case descriptively but the mechanical "FAVORED" calls (outcomes
2/3) simply don't fire. This is disclosed, not hidden, but a hostile
reviewer could reasonably ask whether K=90 is far enough right with real
confidence, or whether this design is repeating the same "the transition is
probably just past our tested ceiling" bet that already failed once at
K=69→90's own old ceiling. **TODO for attack round:** consider whether a
PRE-registered "if K=90 shows no decline whatsoever (h4 ≥ 0.98, matching
K=69's own 89.39–98.6% range with no visible dip), the wave should note
this as evidence `S` itself may not be constant even locally" — i.e. define
a descriptive trigger for outcome 5's own sub-case, not just a bare
"neither band" catch-all.

**Q2. Is reusing K=69 as the grid's "low edge" actually sound, given ONE of
its 3 seeds is already known to be excluded (admissibility) — does the
resulting n=2 low-edge point weaken the fit's own left-tail anchoring
disproportionately?** **RESOLVED at Rev 1 (attack-round-1 MAJOR-2):** the
TODO below asked for exactly the `sim_cliff_power.py` re-run at the actual
5-point grid — now run (§15.20.4), not merely planned. The measured
finding is MORE serious than this TODO anticipated: at the program's own
realistic cliff width (w=0.0597) the n=2 K=69 point is not the dominant
weakness — restoring it to n=3 only narrows the projected CI half-width by
~4% (0.0430→0.0408 abs-slack; 0.0319→0.0307 power-law), nowhere near
closing the gap to the derived 0.0145 discrimination threshold. The
"real, small power loss, not a structural gap" framing below undersold the
issue: the whole mandatory grid, not just the K=69 point, is underpowered
for clean band discrimination at the measured noise level — see §15.20.4's
own pre-registered response rule.

*(Original, pre-Rev-1 text, preserved per house "record history, don't
retcon it" convention):* K=69's own role in THIS wave's fit is structurally
different from K=16's role in the original d=64 cliff (§15.8) — it is not
the FAR-left anchor (K=72–90 sit to its right, not around it), it is simply
the leftmost of 5 roughly-evenly-spaced points, already itself close to the
ceiling (h4≈0.98 at n=3, 0.96–1.0 at n=2). An n=2 vs n=3 seed count at one
interior-ish point is a real, small power loss, not a structural gap — but
this design has not re-run §15.8's own power-check machinery to confirm the
n=2 K=69 doesn't meaningfully widen
the new fit's own CI. **TODO for attack round:** run `sim_cliff_power.py`'s
existing machinery once more, this time simulating the ACTUAL 5-point grid
`{K=69(n=2), 72(n=3), 78(n=3), 84(n=3), 90(n=3)}`, before trusting this
design's own implicit assumption that dropping one K=69 seed doesn't matter.

**Q3. Does the power-law rival (§15.20.4) deserve to be treated as
seriously as absolute-slack, given it was invented by THIS design
specifically to have somewhere for a not-absolute-slack result to land —
is this quietly building a test that can only ever confirm one of two
options the designer picked, rather than genuinely falsifying the research
question?** **Current answer:** this is a fair challenge to sit with, not
wave away. Power-law is not an arbitrary invention — §15.10.1 (written
BEFORE either new-d point existed) already flagged "some OTHER monotone-in-
d rescaling" as a structural gap the original 3-point design could not
close, and a power law is the most natural such alternative (§13.10's own
citation of `d_state` itself, not a ratio, as the surviving candidate makes
ANY superlinear-in-`d` growth of `K_crit` a live possibility, of which
"absolute-slack" — linear with slope 1 — is only the simplest special
case). But it is true that outcome 4/5 (§15.20.4) exist precisely to avoid
the trap of forcing a binary pick — a genuinely uninformative or
neither-fits result is treated as a real, nameable outcome, not smoothed
into whichever rival is closer. **TODO for attack round:** consider naming
a THIRD rival (e.g., `S` itself growing slowly with `d`, a "soft" version
of absolute-slack) explicitly, rather than letting anything that isn't
power-law-shaped default into absolute-slack's own band by elimination.

**Q4. Is the `H_scaling_2 = +5.0 GPU-h` extension ask (§15.20.5) actually
justified, or is this design quietly re-litigating a program `STATE.md`
has now called "closed" or "effectively closed" for the SECOND time within
a single follow-up wave of the SAME already-once-reopened sub-ledger —
exactly the "process smell" §15.17 Q5 already flagged for the parent
program?** **Current answer:** this is a real, structural echo of Q5's own
concern, now one level deeper (a sub-ledger of a program that was already
reopened once). The mitigating fact: this ask is small (5 GPU-h, ~24% of
the sub-ledger's own original 21 GPU-h size) and DIRECTLY funds finishing a
question §15.19 itself left explicitly open (not a new, unrelated
extension) — but a hostile reviewer could reasonably ask whether EVERY
follow-up wave in this thread will keep asking for "just one more small
top-up," and whether the KEY_ANCHORING_CAPACITY_SCALING successor-program
idea §15.17 Q5 already named should have been stood up before this
specific ask, rather than after a second small extension to the old one.
**TODO for attack round:** decide explicitly whether this ask is the LAST
one against KEY_ANCHORING_SCALING specifically, with any further work
required to open the named successor program instead — a bright line, not
left to keep drifting wave by wave.

**Q5. Does the `fit_cliff_curve.py` fix (§15.20.3) risk silently changing
ANY already-published or in-flight claim (e.g. the workshop-2026/ICLR-2027
capacity-trilogy submission drafts, per this repo's own recent commits)
that may have cited a pre-fix number?** **Current answer:** the
program-wide impact scan (§15.20.3) found ZERO historical fits affected —
every prior wave's own candidate-(d) pool was already 100% admissible, so
the fix changes no existing number. But this design's own scan covered
only the `KEY_ANCHORING*` experiment-runs directories; it did not check
whether any already-drafted paper section (`matrix-thinking/submissions/`)
cites a NUMBER derived from `fit_cliff_curve.py` output that could,
in principle, be re-derived differently post-fix even though the INPUT data
happens to be unaffected here specifically. **TODO for attack round:**
grep `matrix-thinking/submissions/` for any `x0`/`fit_cliff_curve` citation
tied to a wave this fix touches, confirm none exist or flag them for a
disclosed erratum-style footnote if the fix is ever applied retroactively
to a cited number.

**Q6 (bonus, scope-discipline).** **Is bundling THREE materially different
asks (a new GPU wave, a seed escalation, and a shared-utility code fix)
into one design section itself a process risk — should the
`fit_cliff_curve.py` fix (§15.20.3), which needs zero GPU and could land
today, be split out and shipped independently of whether the d=96-wide
wave itself ever gets PI sign-off?** **Current answer:** yes, and this
design should say so plainly rather than let the fix's own zero-GPU-cost,
build-now-worthy status get held hostage to the GPU-budget conversation
(§15.20.5) it has nothing to do with. **TODO for attack round: recommend
§15.20.3 be built and its negative tests run FIRST, independent of and
before any PI sign-off on the GPU-costing parts of this design** — the
fix is safe, already-scoped, already-regression-tested-on-paper, and
directly closes a registered mandatory-since-§15.9 gap regardless of
whether the wider K-grid ever launches.

---

## §15.21 ATTACK-ROUND-1 fix-map (2026-07-07) — verdict NEEDS-REVISION

An independent adversarial pass reviewed §15.20 (the d=96 wider-K
cliff-hunting design, Rev 0) before any GPU work launched, per this
program's own standing multiple-independent-audit-rounds discipline and
per §15.20.7's own registered prerequisite (mirrors
`REASONING_LINK_DESIGN.md` §16.7's and `KEY_ANCHORING_DESIGN.md`'s own
fix-map convention, house style). Verdict: **NEEDS-REVISION** — 4 MAJOR,
4 MINOR. The attack round also independently RE-DERIVED both rival CI
bands from scratch (§15.20.4's absolute-slack and power-law corner
arithmetic) and **CONFIRMED they remain disjoint** — `[0.718, 0.739]` vs
`[0.768, 0.837]`, a real 0.029 gap — the design's own central claimed
property survives fresh scrutiny even though the power to EXPLOIT that
disjointness (MAJOR-2) does not, at this wave's own registered seed
budget. Every finding below is fixed in this revision (Rev 1, §15.20.1–
§15.20.6); none is deferred or waved away. Findings are recorded
near-verbatim for the historical record, per house style; resolutions are
stated as landed in this text, not as intentions.

| # | Finding (attack-round on §15.20, Rev 0) | Severity | Fix (Rev 1) | Location |
|---|---|---|---|---|
| MAJOR-1 | The reused K=69 JSON byte-copy (3 files, seeds 1730/1731/1732, copied from the original §15.19 archive into this wave's own output directory) was only ever a PROSE instruction to "verify byte-identical copies via sha256" — no mechanism existed that could actually fail if the copy silently diverged, the exact class of gap §15.18 FATAL-2 already found and closed once for the kernel-safety gate | MAJOR | Named, enforced mechanism: a dedicated chain-script step (`keyanchor_scaling_wide_chain.sh`) computes `sha256sum` of the 3 copied files and diffs against a manifest PINNED at build time against the ORIGINAL §15.19 archive (never regenerated from the copy itself, which would make the check tautological); fail-loud, `exit 1` on any mismatch or missing file, before the fit step runs. Enforced belt (bash) + suspenders (Python-side refusal without the bash gate's own success sentinel), mirroring the existing kernel-gate pattern exactly. Added to the new §15.20.6 enforced-gate table as Gate (c) | §15.20.1 (Reusing K=69 in the fit input); §15.20.6 Stage −1 item 4 + enforced-gate table |
| MAJOR-2 | §15.20.8 Q2's own TODO ("run `sim_cliff_power.py`'s existing machinery once more, simulating the ACTUAL 5-point grid") was left as an un-run TODO, not a Stage −1 BLOCKING requirement — the design's own central "clean discrimination" claim (§15.20.4) was never power-checked at the grid it actually proposes to run | MAJOR | Promoted to Stage −1 BLOCKING and RUN THIS SESSION (CPU-only, no GPU): new purpose-built driver `sim_cliff_power_wide_grid.py` (imports `sim_cliff_power.py`'s own `sigmoid`/`POOLED_REL_SD`/`NOISE_SD_MULTIPLIERS` unmodified, adds a heterogeneous-`n` sibling of `simulate_once`/`bootstrap_ci_width`) run at `K_seeds={69:2,72:3,78:3,84:3,90:3}`, `d_state=96`, `n_trials=4000`, `seed=20260706`. **Measured: at the program's own realistic cliff width (w=0.0597), both rival-center projected CI half-widths (0.0430 abs-slack, 0.0319 power-law) exceed the derived `0.0145` half-gap trigger** (half of the 0.029 inter-band gap — the correct, derived threshold, replacing the attack round's own rough "±~0.036" estimate). Firing the reserved K=69 contingency seed 1733 (n=2→n=3) narrows the half-widths by only ~4% — nowhere near sufficient. Pre-registered response: the decision rule now expects BOTH-CONSISTENT as the likely outcome at this seed budget (not a failure), and a genuinely powered follow-up is named as a separate, out-of-scope future GPU ask, not assumed funded here | §15.20.4 (new MAJOR-2 subsection, full table + response rule); §15.20.6 Stage −1 item 7; §15.20.8 Q2 (RESOLVED) |
| MAJOR-3 | The Step-1 data-quality gate ("if degenerate_frac > 10% → AMBIGUOUS") conflated two structurally different failure modes — genuine noisy scatter (real signal, seed-escalation is the right remedy) and a curve that never leaves ceiling at all (no signal in this window, seed escalation cannot fix it) — and outcome 5's own worked example ("pinned again near the 0.9 fit-bound") gave no adjudication of what that would mean for either named rival | MAJOR | New named verdict **CLIFF-BEYOND-WINDOW** (step 1a: `degenerate_frac>10%` AND every sampled K's raw per-seed h4 mean `≥0.98`), distinct from AMBIGUOUS (step 1b: scatter-driven, seed-escalation eligible). Decision rule rewritten as an explicit table (§15.20.4). Per-rival adjudication written out: absolute-slack strained past rescue (a >0.208 absolute miss, larger than the already-disclosed 11% `S`-drift), power-law's functional form survives via a re-fit `α` using the new point as a third calibration point. Empirically corroborated by this session's own MAJOR-2 power-check run: the `upper_edge_x090` control truth shows `degenerate_frac` climbing to 0.65–0.997 while every in-window truth stays near 0 | §15.20.4 (decision-rule table + CLIFF-BEYOND-WINDOW adjudication); §15.20.8 Q1 (RESOLVED) |
| MAJOR-4 | The proposed `+5.0 GPU-h` sub-ledger extension (§15.20.5) had no enforcement mechanism of its own — the ORIGINAL wave's `KEYANCHOR_SCALING_PI_SIGNOFF=1` token, already built and enforced (bash + Python), would silently also gate this SECOND, separate PI decision if left as the only check, meaning a stale token exported from launching the original wave could authorize the extension without a second explicit sign-off | MAJOR | New, DISTINCT token `KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1`, registered as a required build task for `--wave keyanchor-scaling-wide`, enforced at the SAME two points as the existing token (bash GATE 0-equivalent + Python-side `os.environ.get(...) != "1"` check, mirroring `keyanchor_scaling_stage_gate`'s own precedent verbatim) — the two env vars are never OR'd together in either check. Added to the enforced-gate table as Gate (d) | §15.20.5 (new MAJOR-4 paragraph); §15.20.6 enforced-gate table |
| MINOR-1 | Cost-table arithmetic slip: `6.6505 × 2 = 13.3011` stated, but the correct product is `13.3010`; the shortfall (`4.0876`) and extension-margin (`≈0.91`) figures inherited the error | MINOR | Corrected to `13.3010`; shortfall recomputed to `4.0875`; extension margin recomputed to `≈0.9125` (rounds to the same `≈0.91` disclosed figure) | §15.20.5 (cost table + shortfall/margin text) |
| MINOR-2 | `fit_cliff_curve.py` line-pin citations (`load_k_mean_h4` "lines 120–142", `d.get("complete")` "line 135", `main()` call sites "lines 300–307") were stale relative to the live file | MINOR | Refreshed to the verified-live values: `load_k_mean_h4` lines 122–145, `d.get("complete")` line 137, `main()` call sites lines 303–310 | §15.20.3 (both citations) |
| MINOR-3 | The absolute-slack point prediction routed through an unnecessary rounding detour (`K_crit(80) = 54.048 ≈ 54`, `S = 80−54 = 26`), silently shifting the point estimate | MINOR | Dropped the rounding detour; `S` kept unrounded (`25.952`), giving the corrected point prediction `x0(96) = 0.729667` (vs. the rounded detour's `0.7292`) — the CI-corner band `[0.718, 0.739]` was never routed through the detour and is UNCHANGED | §15.20.4 (absolute-slack paragraph) |
| MINOR-4 | The seed-disjointness-from-REASONING_LINK justification relied on MAGNITUDE ("six orders of magnitude away, no realistic collision") rather than structure — a weaker guarantee that would not survive either program's seed range growing substantially | MINOR | Corrected to a STRUCTURAL argument: REASONING_LINK's `episode_seed` formula sums only exact multiples of 1,000, so it emits `≡0 (mod 1000)` by construction; every KEY_ANCHORING seed block in this program's history (verified directly: 1730, 1740, 1840, 1940, 2040, …) is never a multiple of 1000. The two seed spaces are disjoint by residue class, not merely by today's numeric range. The magnitude fact is retained as a disclosed secondary cross-check, no longer the load-bearing reason | §15.20.1 (seed table paragraph) |

**What Rev 1 could NOT cleanly fix, disclosed rather than hidden:** MAJOR-2's
own measured result is a genuinely negative finding, not fully "fixed" in
the sense of restoring the design's original ambition — this wave's own
mandatory grid, even with every already-reserved contingency seed fired,
cannot reliably achieve clean band discrimination at this program's own
measured cliff width. Rev 1 responds to this honestly (a pre-registered
BOTH-CONSISTENT expectation, a named future-wave ask, MAJOR-3's
CLIFF-BEYOND-WINDOW branch as a MORE informative fallback) rather than by
quietly lowering the discrimination bar to declare success more easily, but
the underlying power gap is real and is not closed by this revision.
§15.20.8's Q3–Q6 (power-law's own epistemic status as a rival, whether the
`+5.0 GPU-h` ask re-litigates an already-once-reopened sub-ledger, the
`fit_cliff_curve.py` fix's own citation-impact scan scope, and the
scope-bundling recommendation) were **not** in this attack round's scope
and remain open exactly as Rev 0 left them — this revision does not claim
to have addressed them. **Rev 1 itself has not yet had its own independent
audit pass** — per this project's standing rule that multiple independent
adversarial rounds catch different bugs each round, landing attack-round-1's
findings does not, on its own, certify §15.20 as CLEARED-FOR-BUILD.

---

## §15.22 VERDICT — harvest, 2026-07-08: AMBIGUOUS (data-quality collapse,
more severe than the pre-registered rule anticipated), d=80 seed
escalation REFUTE stands (tightened); the wide grid's own rival
discrimination test (§15.20.4) never executed

**Launched under both required tokens** (`KEYANCHOR_SCALING_PI_SIGNOFF=1`
and `KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1`, both recorded at the top of
`keyanchor_scaling_wide_run2.log`, confirming Gate (d) fired as designed).
`keyanchor_scaling_wide_run1.log` is an earlier, incomplete smoke-only
session (5 smoke FAILUREs — `sim_cliff_power_wide_grid_results.json` and
the archived §15.19 directory were not yet in place at that point);
`run2.log` is the real, complete session, all 16 smoke items PASS, and is
the log this harvest treats as authoritative throughout. tmux is gone
(session ended cleanly, not crashed — see below), so this section is
built entirely from raw JSONs, logs, and gate artifacts pulled off box,
never from a live session.

### Scope

12 new d=96 cells (`K∈{72,78,84,90}×3` seeds, blocks 1740/1840/1940/2040)
in `wavekeyanchor-scaling-wide/`, reusing the 3 already-archived K=69/d=96
cells (seeds 1730–1732, byte-verified via the Gate (c) sha256 check) as
the grid's low edge, plus 4 new d=80 seed-escalation cells (K=48 seeds
1133/1134, K=53 seeds 1233/1234) written into the ORIGINAL wave's own
`wavekeyanchor-scaling/` directory per §15.20.2's design. **16 new GPU
cells total.**

### Per-cell verification (mechanical, all 16 new raws + the 3 reused)

`complete=true`, `steps_completed=20000`, `timed_out=false` — **19/19, no
exception** (16 new + 3 reused). Architecture pins uniform across all 19:
`anchor_active=True`, `anchor_lambda_mode="learned"`,
`anchor_table_frozen=False`, `geo3_n_iter=20`. `H_extra=[7,21]`
(unmodified default) at every new cell — confirmed no collision at any of
K∈{72,78,84,90} (§15.20.1's own arithmetic: `7%K` and `21%K` never
wrap into `{1,2,3}` for K>21, verified directly on the raw JSONs, not
just derived on paper). h1 guard
(`M2_in_distribution["1"]["recovered_frac@0.9"]`) reads **exactly 1.0 at
all 19 cells** — zero training-health concern anywhere in this wave; every
symptom below is confined to the eval-side geo3 admission check, never to
training itself.

| group | K | d | seed | complete | wall_s | admissible | checkpoint_fallback_seen | h4 (M3, hop 4, rec@0.9) |
|---|---|---|---|---|---|---|---|---|
| d80-escalation | 48 | 80 | 1133 | True | 1301.00 | **True** | False | 0.9056 |
| d80-escalation | 48 | 80 | 1134 | True | 1279.48 | **True** | False | 0.9212 |
| d80-escalation | 53 | 80 | 1233 | True | 1426.01 | **True** | False | 0.6274 |
| d80-escalation | 53 | 80 | 1234 | True | 1403.63 | **True** | False | 0.5149 |
| d96-wide (reused) | 69 | 96 | 1730 | True | 1568.51 | False (known, §15.19) | True | 0.9918 |
| d96-wide (reused) | 69 | 96 | 1731 | True | 1583.28 | **True** | False | 0.9986 |
| d96-wide (reused) | 69 | 96 | 1732 | True | 1547.48 | **True** | False | 0.9614 |
| d96-wide (new) | 72 | 96 | 1740 | True | 1591.20 | **False** | True | 0.9319 |
| d96-wide (new, calib) | 72 | 96 | 1741 | True | 1517.35 | **True** | False | 0.8426 |
| d96-wide (new) | 72 | 96 | 1742 | True | 1588.36 | **False** | True | 0.9904 |
| d96-wide (new) | 78 | 96 | 1840 | True | 1473.54 | **False** | True | 0.8667 |
| d96-wide (new) | 78 | 96 | 1841 | True | 1511.03 | **False** | True | 0.9823 |
| d96-wide (new) | 78 | 96 | 1842 | True | 1537.37 | **False** | True | 0.9488 |
| d96-wide (new) | 84 | 96 | 1940 | True | 1419.82 | **False** | True | 0.9806 |
| d96-wide (new) | 84 | 96 | 1941 | True | 1378.79 | **False** | True | 0.9573 |
| d96-wide (new) | 84 | 96 | 1942 | True | 1429.39 | **False** | True | 0.9363 |
| d96-wide (new) | 90 | 96 | 2040 | True | 1334.59 | **False** | True | 1.0000 |
| d96-wide (new) | 90 | 96 | 2041 | True | 1301.71 | **False** | True | 1.0000 |
| d96-wide (new) | 90 | 96 | 2042 | True | 1305.78 | **False** | True | 1.0000 |

**Admissibility, new cells only: 5/16 (31.25%).** All 4 d80-escalation
cells are clean (100% admissible, `checkpoint_fallback_seen=False`
throughout — same clean pattern as every other d=80 cell this program has
ever run). Of the 12 new d=96-wide cells, only **1** (K=72/seed=1741) is
admissible; K=78, K=84, K=90 are **0/3 admissible each** — every single
seed at those three K's fails. This table was built by direct Python
inspection of every raw JSON's `complete`/`geo3_admission`/`M3_held_out`
fields (`build_table.py`, archived at `fits/`), not by trusting any
box-side printed summary.

### Resolving the NOT-READY anomaly

**Root cause, stated precisely: this is NOT a path/glob bug. It is a
real, disclosed admissibility collapse in the underlying data at the new
d=96-wide K's.** Four independent checks rule out a path/environment
explanation:

1. **The registered fit invocation was reproduced locally, off-box, byte-
   for-byte, and failed identically.** `fit_cliff_curve.py` pulled from
   the box is sha256-identical to the repo's own already-committed copy
   (both match the file used throughout §15.19/§15.20). Running
   `python3 fit_cliff_curve.py --cliff-out-dir wavekeyanchor-scaling-wide
   --d-state 96 --k-grid 69 72 78 84 90 --n-trials 4000` against the
   pulled raws, in this repo's own `.venv` (no box environment, no box
   filesystem), reproduces the exact same line: `NOT READY: k32_mean=None
   k48_mean=None missing new-K means=[78, 84, 90]`. `k32_mean=None
   k48_mean=None` is a RED HERRING, not a symptom — `d_state=96` is not in
   `ANCHORED_D_STATES=(64,)` (line 295), so those two variables are
   unconditionally `None` by construction (line 321) regardless of data
   availability; the load-bearing part of the message is `missing new-K
   means=[78, 84, 90]`.
2. **`missing` is computed from `per_k_mean[K] is None`, and `load_k_mean_h4`
   (the SAME fixed loader §15.20.3 built and regression-tested) returns
   `(None, [])` for a K only when `per_seed` ends up empty after its
   `complete` AND `geo3_admission.admissible is True` filters.** Direct
   inspection (above table) confirms: K=78/84/90 have 3/3 seeds each with
   `admissible=False` — `per_seed` is genuinely empty for those three K's
   in `wavekeyanchor-scaling-wide/`, glob pattern and all. The glob
   (`*_K{K}_armd_*.json`) matches all 3 files at every K (verified
   directly, 3 paths per K, all 5 K's) — the files are found; their DATA
   is what fails the filter.
3. **Trying a combined 9-K directory (original K=24/51/57/63 copied in
   alongside the wide grid's own K=69/72/78/84/90) reproduces the exact
   same `NOT READY: ... missing new-K means=[78, 84, 90]`.** If this were
   a directory/path issue, pointing the loader at a superset of every
   archived d=96 cell this program has ever run would have to change the
   outcome. It does not — because no admissible seed exists ANYWHERE in
   the program's archive at K=78, K=84, or K=90. The data gap is real,
   not local to one directory.
4. **The chain log confirms the halt is exactly here, not somewhere
   upstream.** `keyanchor_scaling_wide_run2.log`'s last two lines are the
   `--scaling-wide-leg d80-escalation` dispatcher's own completion message
   (`WAVE keyanchor-scaling-wide DONE. 4 succeeded this session, 0 failed,
   0 still pending.` — this reports the 4 d80-escalation cells, NOT a
   whole-chain completion) immediately followed by the `NOT READY` line,
   then nothing — `keyanchor_scaling_wide_chain.sh` runs under
   `set -euo pipefail` (line 79), so the fit step's non-zero exit through
   the `| tee` pipe killed the chain there. `KEYANCHOR_SCALING_WIDE_DONE`
   (the chain's own final sentinel, `touch`ed only after BOTH fit steps
   succeed) **does not exist on box** — direct confirmation the chain
   never reached the d=80 re-fit (step 92) or its own completion line.
   The `WAVE ... DONE` message the task brief read as "the chain
   completed" is a different, narrower message — the d80-escalation
   dispatcher's own per-leg summary, printed BEFORE the two fit steps in
   the same chain script, not the chain's own overall completion.

**Mechanistic root cause of the admissibility collapse itself (verified
by reading `compute_geo3_admission`/`_geo3_checkpoint_fallback_seen`,
`run_deltanet_rd.py` lines 509–571):** `checkpoint_fallback_seen` is
computed by scanning every logged checkpoint's four recovery-probe pools
(`M2_in_distribution`, `M3_held_out`, `C17_heldout_entities`,
`C19_heldout_template`) for a per-hop `geo3_fallback_triggered_this_hop`
flag — this is an **EVAL-side** signal, raised when the Newton-Schulz
orthogonalization used to SCORE a recovery probe falls back to its `eigh`
path. It is entirely independent of `n_geo3_fallback_train_steps`, the
TRAINING-side counter. **Every one of the 11 inadmissible new cells reads
`n_geo3_fallback_train_steps=0`** — training itself converged cleanly at
every logged step, consistent with the clean loss curves, `h1=1.0`
guard, and `task_performance_floor_pass=true`/`value_salvage_tier_pass=
true` on every affected cell. **The failure is confined to Newton-Schulz
convergence on the EVAL-time recovery-probe queries against the FINAL,
fully-learned (`anchor_table_frozen=False`) anchor table — not to
anything about training health.** This directly explains why §15.20.1's
own Wave −1 `GATE2_N_ITER_BY_D_K[96]` sufficiency check (Gate (b), which
PASSED cleanly for K∈{72,78,84,90}, confirmed in
`keyanchor_scaling_wide_niter_result.json`, archived) did not predict
this: that check tests Newton-Schulz convergence on the STATIC,
frame-potential-initialized anchor table only — it structurally cannot
see a failure mode that only appears once the anchor table has drifted
away from that init over a full 20,000-step LEARNED training run, probed
at query geometries the static check never exercises. **This is a real,
previously-unregistered gap in Gate (b)'s own coverage**, disclosed here
for the first time — not a rubber-stamped gate (§15.19's own K=69/seed=
1730 anomaly was the first, isolated hint of exactly this failure mode;
this wave shows it is not a one-off but a real, K/d-ratio-correlated
effect that intensifies sharply above K/d≈0.75: 0/30 at the original
grid's K≤0.71875, 1/3 at K=0.75 (K=72), 3/3 at each of K=0.8125/0.875/
0.9375). The d=80 escalation cells (K=48/53, both already-verified K's
from the ORIGINAL, non-widened grid) show **zero** instances of this
failure — consistent with it being specific to the newly-pushed d=96/
high-K geometry, not a generic regression.

**No cells are actually missing or incomplete** — the task brief's own
caution to "stop and report prominently" if that were the case is
triggered by a related but distinct finding: every cell ran to
completion, trained cleanly, and produced a real, readable `h4`; they are
simply excluded from the fit by the (correctly-behaving, already
regression-tested) admissibility filter at a rate this program has never
seen before. This is disclosed prominently as the wave's own headline
finding, not smoothed into a footnote.

### Local re-fits (all run this session, `fit_cliff_curve.py` — the FIXED
version, sha256-confirmed identical to both box and repo — against the
raw JSONs pulled off box)

**1. Exact registered invocation (d=96 wide, `--k-grid 69 72 78 84 90`):**
reproduces `NOT READY` verbatim (above) — **no fit output exists, and
none can be produced from this K-grid with the data this wave collected.**

**2. d=80 seed escalation re-fit (`--k-grid 20 43 48 53 58`, now 5 seeds
at K=48/53), against `wavekeyanchor-scaling/` (all cells 100% admissible):**

```
sigmoid fit: x0=0.6779 w=0.0479 L=0.9994 rss=0.00024
bootstrap CI(x0): [0.6683, 0.6867] width=0.0184
bootstrap CI(w):  [0.0391, 0.0594] width=0.0202
degenerate fraction: 0.0000 (within the 10% bar)
```

vs. the original (n=3) `x0=0.6756`, CI `[0.6620, 0.6868]` width `0.0248`
— the point estimate shifts negligibly (+0.0023) and the CI tightens by
26% (width 0.0248→0.0184), exactly the "not expected to change the
verdict, tightens the CI" outcome §15.20.2 pre-registered. **This fit was
never run on box** — the chain halted at step 91, before step 92 (the
d=80 re-fit) could execute; this local run is its first execution.

**3. Regression check (original 5-K d=96 grid, `--k-grid 24 51 57 63 69`,
FIXED loader, no manual pre-filtering) — confirms the loader itself is
working correctly, independent of the wide-grid anomaly:**

```
degenerate fraction: 0.9477 (EXCEEDS the 10% bar)
```

Matches §15.20.3's own hand-computed regression target
(`degenerate_frac=94.77%` exactly) to 4 decimal places — the fixed
loader, this exact venv, and the pulled data all check out before
trusting anything above.

**4. Diagnostic-only combined fit (`--k-grid 24 51 57 63 69 72`, NOT the
registered wide-grid test — every original d=96 K plus the ONE new K
that has any admissible data at all, K=72 at n=1):**

```
sigmoid fit: x0=0.7716 w=0.0126 L=0.9946 rss=0.00026
bootstrap CI(x0): [0.7700, 0.7841] width=0.0140
degenerate fraction: 0.2622 (EXCEEDS the 10% bar)
curve: K24=1.0, K51=1.0, K57=0.9974, K63=0.9805, K69=0.9800, K72=0.8426
```

**Explicitly NOT licensed to invoke §15.20.4's discrimination test** —
wrong K-grid (78/84/90 are structurally absent, the entire reason the
grid was widened), `degenerate_frac` itself exceeds the 10% bar, and the
one point driving the visible decline (K=72) rests on a SINGLE seed.
Reported only as a descriptive lead: the point estimate (0.7716, CI
[0.7700,0.7841]) sits just outside the abs-slack band `[0.718,0.739]`
and just inside the power-law band `[0.768,0.837]` — suggestive that
SOMETHING starts declining right around K/d≈0.75, consistent with
neither d=96 result seen so far (§15.19's flat-to-K/d=0.71875 curve, or
CLIFF-BEYOND-WINDOW) being the final word — but this is a hint for a
properly-powered follow-up, not a result.

### Applying §15.20.4's decision rule, mechanically

| Step | Condition (quoted) | Evaluation | Result |
|---|---|---|---|
| 0 | "Fit converges, `degenerate_frac ≤ 10%`" | **Fit does not converge — it cannot even be attempted.** `main()` exits 1 (`NOT READY`) before `curve_fit` is ever called, because 3 of the 5 registered K's (78, 84, 90) have zero admissible seeds anywhere in the archive. | Does not apply — no `degenerate_frac` value exists to evaluate against ≤10% or >10%. |
| 1a | `degenerate_frac > 10%` AND every sampled K's raw per-seed h4 mean `≥ 0.98` | Cannot be evaluated on the registered grid (no fit ran). Evaluated on the ADMISSIBLE-only per-seed data as the closest analog: K=69 admissible mean = 0.9800 (seeds 1731/1732); K=72 admissible mean = 0.8426 (seed 1741, n=1) — **K=72's own admissible mean is well below 0.98**, so even this looser reading fails the flatness condition. | **Not CLIFF-BEYOND-WINDOW** — there is a real, if thin, decline visible in what admissible data exists, not a curve stuck at ceiling. |
| 1b | `degenerate_frac > 10%` AND at least one sampled K's raw per-seed h4 mean `< 0.98` (scatter, not flatness) | No `degenerate_frac` exists to test the numeric threshold, but the QUALITATIVE signature (a K with mean well under 0.98 alongside K's at/near ceiling) matches. **This is the closest-fitting named row, generalized one level further than it was written for:** row 1b's own literal text presumes a completed (if noisy) fit exists; this wave's own failure is a level EARLIER and MORE SEVERE — the fit cannot be attempted at all because 3 of 5 K's return zero valid data, not merely a wide CI. | **AMBIGUOUS — DATA-QUALITY COLLAPSE**, an extension of row 1b's own spirit that the pre-registered table did not literally anticipate (it distinguished flat-ceiling from noisy-scatter, but implicitly assumed every grid K would produce SOME usable data either way). |
| 2–5 | Band-overlap / BOTH-CONSISTENT / NEITHER-SURVIVES tests | All four require a real `CI(x0, 96-wide)` from the registered grid. **None exists.** | Not reachable. |

## **WAVE VERDICT (d=96-wide leg): AMBIGUOUS — DATA-QUALITY COLLAPSE.**
## **WAVE VERDICT (d=80-escalation leg): REFUTE stands, tightened CI, unchanged conclusion.**
## **Overall wave verdict, per §15.19's own carried-forward rule ("the WAVE's own overall call is AMBIGUOUS if either d's fit is AMBIGUOUS"): AMBIGUOUS.**

This is the mechanical, closest-defensible application of the
pre-registered table to a situation that table did not literally name —
disclosed as an extension, not asserted as a literal row match. §15.20.4's
own central discrimination test (the entire point of widening the grid:
bracket both rival bands with real points at K/d∈[0.75,0.9375]) **never
executed** — not because it was underpowered (§15.20's own MAJOR-2 power
check already, honestly, expected a likely BOTH-CONSISTENT outcome even
in the best case), but because the data needed to attempt it at all does
not exist. This is a DIFFERENT, more severe failure than anything §15.20
Rev 1's own attack round considered (§15.20.8 Q1's own TODO worried about
a flat, non-degenerate-but-uninformative curve — not a curve that cannot
be fit at all).

### Reading this honestly, alongside the mechanical call

**d=80 (escalation): unchanged from §15.19 — a clean, non-degenerate
REFUTE of ratio-invariance, now with a tighter CI.** `x0(80)=0.6779`
`[0.6683,0.6867]` still excludes the invariance band `[0.4745,0.6165]`
by a wide margin (gap 0.0518 at the CI's own near edge). Nothing about
this wave changes that conclusion.

**d=96 (wide): the widening attempt itself produced a genuinely new,
disclosed finding — but not the one it was designed to produce.** Instead
of localizing (or definitively ruling out) a cliff in `K/d∈[0.75,
0.9375]`, it surfaced a real, K/d-correlated Newton-Schulz eval-admission
failure that makes 11 of 12 new cells unusable for the fit they were
built to feed. The ONE admissible new point (K=72, h4=0.8426) is,
descriptively, the single most informative new datum this wave produced
— it is the first d=96 point in the program's history to read
meaningfully below the 0.98 ceiling band, right where both named rivals
predicted a transition should start. It cannot, on n=1, carry any
statistical weight.

### Rival comparison — unchanged from §15.19, since the wide grid's own
discrimination test did not execute

| Account | §15.19 status | This wave |
|---|---|---|
| Ratio-invariant | Refuted (d=80), uninterpretable (d=96) | Unchanged — not re-tested here |
| Fixed-K | Refuted at both d's | Unchanged |
| Absolute-slack (`[0.718,0.739]` predicted band at d=96) | Closest surviving rival, not confirmed | **Still not tested** — the diagnostic-only 6-K fit's point estimate (0.7716) sits just OUTSIDE this band, but is not a licensed test |
| Power-law (`[0.768,0.837]` predicted band at d=96, new at §15.20) | N/A (named this wave) | **Still not tested** — the diagnostic point estimate sits just INSIDE this band, but is not a licensed test (wrong grid, degenerate, n=1-driven) |

### Realized GPU-h vs. ceiling

Summed directly from all 16 new cells' own `wall_s` (all single-GPU,
`--per-gpu 1`), independently cross-checked against on-box file
timestamps (`started_at` field to output-JSON `mtime`, per-cell agreement
to within 4.1s, program-wide sum agreement to within 0.32s / 0.0001
GPU-h — the two methods concur):

| Group | Cells | Sum wall_s | GPU-h (wall_s) | GPU-h (timestamp cross-check) |
|---|---|---|---|---|
| d=96-wide, new (K=72,78,84,90 ×3) | 12 | 17388.93s | 4.8303 | 4.8302 |
| d=80-escalation, new (K=48,53 ×2) | 4 | 5410.12s | 1.5028 | 1.5029 |
| **All 16 new cells** | **16** | **22799.05s** | **6.3331** | **6.3330** |

Wall-clock bracket (earliest `started_at` to latest completion, across 6
parallel GPU slots 2–7): **2.10h** — consistent with 16 cells at
~1425s/cell average scheduled across 6 concurrent slots in roughly two
batches, not evidence of any stall or restart.

**Realized: 6.3331 GPU-h against this design's own mandatory-only cost
table — 95.2% of the 1× point estimate (6.6505 GPU-h) and 47.6% of the
2× pessimistic bracket (13.3010 GPU-h).** This is a return to this
program's more typical realized/estimate ratio (§15.19 was the outlier,
landing at 112.5% of its own 1×); the escalation cells landed almost
exactly on their own K-specific point estimates (K=48: realized 0.7176
GPU-h vs. priced 0.7065, +1.6%; K=53: realized 0.7860 vs. priced 0.7684,
+2.3%), and the d=96-wide grid landed under its own point estimate
(realized 4.8303 vs. priced 5.1756, −6.7%) despite the admissibility
collapse — the COST model was accurate; only the DATA-QUALITY assumption
implicit in it (every K yields usable data) was wrong.

**KEY_ANCHORING_SCALING sub-ledger update:** 11.7865/21 GPU-h (§15.19) +
6.3331/21 GPU-h (this wave) = **18.1196/21 GPU-h realized, reserve
2.8804/21** — **this wave's realized cost fits inside the ORIGINAL 21
GPU-h ceiling with room to spare; the `+5.0 GPU-h` extension
(`KEYANCHOR_SCALING_EXT_PI_SIGNOFF`, §15.20.5) was authorized and its
gate fired correctly, but was never actually drawn on.** Extended
ceiling (21+5=26), if ever needed for a genuine follow-up: 18.1196/26
(69.7%), reserve 7.8804/26.

### What this wave does and does not show

1. **Does NOT execute §15.20.4's own discrimination test** — the wave's
   entire reason for existing. Neither the absolute-slack band nor the
   power-law band is confirmed, refuted, or even meaningfully
   approached at the pre-registered grid.
2. **Does NOT change d=80's own clean REFUTE of ratio-invariance** — now
   with a tighter, still-excluding CI.
3. **DOES surface a new, real, disclosed instrument finding**: Newton-
   Schulz eval-side admissibility failure rate rises sharply with K/d at
   d=96 (0/30 at K/d≤0.71875 → 1/3 at K/d=0.75 → 3/3 at K/d≥0.8125),
   confined to the EVAL-time recovery-probe queries against the final
   learned anchor table, not to training health. This is itself a
   partially-informative negative result about the CURRENT `n_iter=20`
   Newton-Schulz configuration's adequacy at the higher end of this
   wave's own tested window — orthogonal to, and not resolving, the
   original capacity-cliff-location question.
4. **Descriptively, not statistically:** the one usable new d=96 point
   (K=72, h4=0.8426) is the first datum in this program's history to show
   a real decline anywhere in `[K/d∈0.72,0.75]`, roughly where both named
   rivals predicted a transition should begin. Worth a properly-powered
   follow-up; not evidence on its own.

### Pre-registered next steps (queued, not run this session)

1. **Diagnose the Newton-Schulz eval-admission failure before re-running
   more seeds at the same K's.** Row 1b's literal remedy ("+2 seeds at
   the noisiest K-group") is unlikely to help here — the failure hit 3/3
   seeds at each of K=78/84/90, consistent with a systematic geometry
   effect (the LEARNED, drifted anchor table at these K/d ratios), not
   per-seed noise. A candidate fix worth testing cheaply (CPU/short-GPU,
   no full re-run): re-check Newton-Schulz convergence at `n_iter>20`
   (e.g. 24 or 28 — already known to converge better on the STATIC init
   per Gate (b)'s own 20→24 rel-change check) against a LEARNED,
   post-training anchor table snapshot from one of the failing cells,
   before committing to a re-run at higher `n_iter`. **A design decision
   for whoever builds the follow-up, not self-authorized here** — mirrors
   this program's own house convention.
2. Firing the reserved K=69 contingency seed (1733, per §15.20.4's own
   MAJOR-2 disclosure) remains registered but is now clearly
   insufficient on its own — it narrows the fit's CI by ~4% and does
   nothing about the K=78/84/90 data gap.
3. Neither is launched by this harvest — both are queued in `STATE.md`.

### Archive

`experiment-runs/2026-07-07_keyanchor_scaling_wide/` (repo-tracked, all
files ≤25MB — largest raw is 4.7MB): `results/deltanet_rd_exactness/
wavekeyanchor-scaling-wide/` (12 new + 3 reused K=69 cell JSONs,
`ALL_DONE`, `CALIBRATION_DONE`, `PROGRESS.txt`, `logs/` — 12 per-cell
logs + `smoke.log`), `results/deltanet_rd_exactness/
wavekeyanchor-scaling_d80_escalation/` (the 4 NEW d80-escalation cell
JSONs + logs only, clearly labeled as a subset extracted from the
ORIGINAL wave's own shared output directory — the other 26 cells there
are already archived at `experiment-runs/2026-07-07_keyanchor_scaling/`),
`logs/` (both chain sessions + numbered stage logs 84–91), `scripts/`
(`fit_cliff_curve.py`, `sim_cliff_power.py`, `sim_cliff_power_wide_grid.py`,
`keyanchor_scaling_wide_chain.sh`, `smoke_keyanchor_scaling.py` — all
byte-verified against box), `gates/` (kernel-safety, n_iter-sufficiency,
sha256 copy-manifest, and power-check artifacts), `fits/` (this session's
4 local fit runs incl. the NOT-READY reproduction's stdout/stderr, the
per-cell verification table script and its output). SSD mirror, full
superset, same tree: `/Volumes/1TB_SSD/learned-representations/
experiment-runs/2026-07-07_keyanchor_scaling_wide/`.

### §15.22 addendum — K=69 contingency seed 1733 FIRED, 2026-07-08 (this
### wave's own next-step 2, run standalone)

The reserved K=69/d=96 contingency seed 1733 (§15.20.4 MAJOR-2's own
registered Stage-1 action; queued, not run, by the harvest above) was run
on 2026-07-08 (box UTC), GPU 3, one cell, via a minimal standalone wrapper
(`run_k69_s1733_contingency.py` — the sweep's own CLI exposes no
single-cell dispatch for a K=69 contingency seed at d=96: neither
`--scaling-wide-leg` includes K=69 cells, so per the run task's own
fallback the wrapper calls `_keyanchor_scaling_spec`/`build_cmd`/
`is_done`/`default_timeout` directly, never hand-typed flags, and refuses
to launch unless its generated command token-diffs clean against the SAME
functions' seed-1730 reference with only seed-derived tokens differing —
verified MATCH at launch, `sha256(cmd)=092de3a9…`, chain log archived).
**New gate finding, closed before launch:** `T_bind(69)=483` is covered by
NEITHER committed kernel-safety artifact — the ORIGINAL §15.2 protocol's
ceiling is 448 and the §15.20.1 wide probe's floor is 504 — so a
purpose-built sibling probe (`smoke_dstate_kernel_t483_probe.py`, same
`try_cell` protocol verbatim, control T=448 + candidate T=483, d=96 only)
was run FIRST on GPU 3: **PASS both** (artifact:
`gates/smoke_dstate_kernel_t483_probe_result.json`). The 15/15-clean §15.19
production precedent at T=483 is thereby upgraded from "disclosed,
non-dispositive precedent" (§15.20.1's own wording) to a mechanically
verified gate. Both PI sign-off tokens were set; gates (a1)/(a2)/(b)/(c)
re-checked from the committed artifacts before launch.

**Cell result: ADMISSIBLE, clean.** `complete=true`, 20000/20000 steps,
`wall_s=1535.2s` (= **0.427 GPU-h at 1×**, vs the ≈0.4 registered estimate
and far inside §15.14's 4658s abort trigger; wrapper total incl. gates
1543.4s), `geo3_admission.admissible=true` (`ns_converged_no_fallback=true`,
zero fallback steps, no checkpoint fallback — unlike seed 1730's own
auto-exclusion), h4 `recovered_frac@0.9 = 0.9175`, `anchor_lambda`
interior (0.6515).

**K=69 group, DESCRIPTIVE ONLY (not a new registered fit — the registered
d=96-wide fit stays blocked on the §15.24 C17 verdict):** admissible seeds
now n=3 of 4 run (1731: 0.9986, 1732: 0.9614, 1733: 0.9175; 1730 remains
auto-excluded). Admissibility-filtered K=69 mean h4 via
`fit_cliff_curve.load_k_mean_h4` (the audited choke point): **0.9800 (n=2)
→ 0.9592 (n=3)**. Per-seed scatter widened (sd 0.0263 → 0.0406 — seed 1733
is the first admissible K=69 seed below 0.96, descriptively consistent
with K/d=0.71875 sitting near the transition region both rivals predict),
while the naive per-K-group 95% t-CI half-width narrowed 0.2361 → 0.1009
(the n=2 t-CI is dominated by t₁=12.7, so this large narrowing is a
property of the degenerate n=2 interval, NOT the ~4% fit-CI narrowing
§15.20.4's MAJOR-2 power check projected for the full 5-point bootstrap —
that projection is unchanged and still nowhere near the 0.0145
discrimination threshold). §15.22's verdict is unchanged: the K=78/84/90
data gap, not K=69's seed count, remains the binding constraint.
Artifacts appended to the same archive tree above (result JSON + per-cell
log + chain log + `gates/smoke_dstate_kernel_t483_probe_result.json` +
`scripts/{run_k69_s1733_contingency.py, smoke_dstate_kernel_t483_probe.py,
k69_s1733_launch.sh, t483_probe.sh}`), SSD mirror updated. Contingency
seed 1734 remains reserved, unfired.

## §15.23 DIAGNOSTIC — the Newton-Schulz eval-admission failure, 2026-07-08:
MISDIAGNOSED-ARTIFACT (a new, disclosed extension to the pre-registered
NS-ITER-FIX-CONFIRMED / STRUCTURAL-ILL-CONDITIONING / MIXED taxonomy) —
§15.22's own "confined to the LEARNED anchor table" mechanistic claim is
WRONG at the mechanism level; the failure is 100% `C17_heldout_entities`-
exclusive, a pool that is architecturally anchor-bypassed by construction,
so testing/fixing the anchor table (this section's own pre-registered ask)
cannot explain or repair the observed admissibility collapse

**Hypothesis (pre-stated, this diagnostic's own charter):** NS at
`n_iter=20` under-converges on the LEARNED/drifted anchor tables at high
K/d; `n_iter∈{24,28,32}` restores convergence and flips admission on the
SAME tables. **Falsifier:** if rel-change/admission does not improve
monotonically with `n_iter` on the real learned tables, the failure is
structural, not iteration count.

**Method:** `matrix-thinking/deltanet_rd/diag_ns_admission.py` (new,
archived at `experiment-runs/2026-07-08_ns_admission_diag/scripts/`), run
CPU-only, zero GPU-h (imports `key_anchoring.py`/`geo3_simulator.py`
directly — both explicitly fla-free per `key_anchoring.py`'s own header,
lines 1–33 — never a hand-copied twin). Three steps, in order:

1. **`mechanism_breakdown()`** — reads every one of the 16 new + 3 reused
   d=96-wide cell JSONs directly (no checkpoint needed) and, per
   checkpoint, records WHICH of the four recovery-probe pools
   (`M2_in_distribution`, `M3_held_out`, `C17_heldout_entities`,
   `C19_heldout_template`) raised `geo3_fallback_triggered_this_hop=True`
   at any hop.
2. **`anchor_table_ns_sweep()`** — THE PRE-REGISTERED CHECK. Pulls the
   final (`step20000.pt`) checkpoint's `anchor_table_trained_rows`
   `(107, 96)` block for one failing cell per K∈{78,84,90}, the one
   failing K=72 seed, and a passing-cell control (K=69 AND the good K=72
   seed — both pulled, no reason not to at 44KB/file), draws 512 random
   K-row subsets (seeded, SAME subsets reused across every `n_iter` for a
   clean per-subset comparison), runs `geo3_simulator.newton_schulz`
   (`key_anchoring.py`'s own documented production-equivalent, lines
   19–33: "mathematically IDENTICAL... cross-verified to
   convergence-precision agreement across every attack round on this
   design") at `n_iter∈{20,24,28,32,40}`, and reports per-`n_iter`
   admissibility (`n_fallback==0`, `resid_tol=0.01`, exactly
   `key_anchoring.GATE2_RESID_TOL`) plus each subset's pre-NS Gram-matrix
   condition number (largest/smallest eigenvalue of the row-normalized
   K×K Gram — the quantity that actually governs whether ANY
   orthogonalizer, at ANY `n_iter`, can drive the subset to `I_K`).
   `raw_table_conditioning` (6a/6b) is also run on the full 107-row table
   as an integrity cross-check against each cell's own already-logged
   `checkpoints[-1]['item6_table_conditioning']`.
3. **`random_proxy_sweep()`** — EXPLORATORY ONLY. Runs the identical
   sweep on `key_anchoring.random_unit_rows_init` draws (the codebase's
   own "candidate (e), frozen-random-table" construction — seeded random
   unit rows, explicitly NOT frame-potential-optimized) at `n=106`
   (`pool_report.n_heldout_names`, every cell this wave) and each cell's
   own `(K, d_state=96)`, as an architecturally-motivated stand-in for
   "a K-set that never received the anchor's own engineering" — flagged
   throughout as NOT a reproduction of the real failing object (see
   headline finding below for why the real object could not be
   reconstructed this session).

**Checkpoints pulled (READ ONLY, `scp` from
`/data/deltanet_rd_keyanchor_ckpts/` on `youthful-indigo-turkey`, GPUs
idle throughout, no tmux touched, no training launched):**

| Cell | K | seed | admissible | role |
|---|---|---|---|---|
| K78/s1840 | 78 | 1840 | False | failing |
| K84/s1940 | 84 | 1940 | False | failing |
| K90/s2040 | 90 | 2040 | False | failing |
| K72/s1742 | 72 | 1742 | False | failing (K=72 representative) |
| K69/s1731 | 69 | 1731 | True | passing control |
| K72/s1741 | 72 | 1741 | True | passing control |

**DISCREPANCY, disclosed prominently:** the task brief's own choice of
"the one failing K=72 seed" (s1740, the first-launched K=72 cell) has
**zero checkpoints on box** — its own JSON's `ckpt_written` field is
`null`/absent (`n_ckpts=0`), while both K=72/s1741 (passing) and
K=72/s1742 (failing) DO have full 10-checkpoint trees. Not a pull
failure — confirmed by `find` returning zero hits for `*s1740*` anywhere
under `/data/deltanet_rd_keyanchor_ckpts/`, and the local archived
`wkeyanchor-scaling_rdx_K72_armd_s1740_...json`'s own `ckpt_written` is
already empty. Root cause not further chased (out of this diagnostic's
scope — does not affect the verdict, since s1742 is an equally-valid
same-K failing representative and both are used); registered as a loose
end for whoever next touches `--ckpt-dir` wiring in the wide-grid launch
scripts.

### HEADLINE FINDING — the failure is C17-exclusive and architecturally
anchor-bypassed, discovered BEFORE any NS sweep was run

`mechanism_breakdown()`, applied to all 12 originally-failing cells
(the 11 new d=96-wide inadmissible cells + the K=69/seed=1730 first-hint
anomaly §15.19/§15.22 already flagged): **every single
`geo3_fallback_triggered_this_hop=True` event, at every checkpoint, in
every cell, occurs in `C17_heldout_entities` and NEVER in
`M2_in_distribution`, `M3_held_out`, or `C19_heldout_template`.**

| Cell | First fallback step | Pool(s) hit (ever) |
|---|---|---|
| K69/s1730 | 20000 | C17 only |
| K72/s1740 | 20000 | C17 only |
| K72/s1742 | 20000 | C17 only |
| K78/s1840 | 20000 | C17 only |
| K78/s1841 | 18000 | C17 only |
| K78/s1842 | 18000 | C17 only |
| K84/s1940 | 16000 | C17 only |
| K84/s1941 | 16000 | C17 only |
| K84/s1942 | 16000 | C17 only |
| K90/s2040 | 2000 | C17 only |
| K90/s2041 | 2000 | C17 only |
| K90/s2042 | 2000 | C17 only |

Cross-checked directly against `compute_geo3_admission`
(`run_deltanet_rd.py:520–575`): `ns_converged_no_fallback =
(n_geo3_fallback_train_steps==0) and not checkpoint_fallback_seen` is the
ONLY one of the four admissibility legs (`value_salvage_tier_pass`,
`ns_converged_no_fallback`, `finite_loss_no_divergence`,
`task_performance_floor_pass`) that reads `False` in any of the 12 cells
— verified directly on every cell's own `geo3_admission` block, all three
other legs read `True` everywhere. **`checkpoint_fallback_seen`, driven
100% by C17, is the sole and complete cause of every inadmissible cell in
this wave.**

**Why this matters, mechanistically (verified by reading the exact code
path, not inferred):** `run_deltanet_rd.py`'s own module docstring
(line 13) names `C17_heldout_entities` as drawing from a **"disjoint name
pool"** relative to the trained-entity pool. `model_rd.py:925`
constructs `anchor_trained_mask` EXCLUSIVELY from `anchor_train_ids =
pools.train_name_ids` (`trained_mask[anchor_train_ids] = True`, every
other row False) — and this is a HARD invariant, not a soft convention:
`model_rd.py:2048` asserts `anchor_table.weight.grad` is EXACTLY zero at
every non-trained row, checked in this codebase's own unit tests.
`anchor_blend_gather_scatter` (`key_anchoring.py:439–469`) computes
`trained_here = anchor_trained_mask[key_ids]` and blends the anchor table
in ONLY where `trained_here` is True (`t_idx`); for C17 batches
`key_ids` are drawn from the disjoint held-out pool, so `trained_here` is
False for 100% of C17's K bind items, by construction — `k_blend_raw`
(the tensor fed to Newton-Schulz) is therefore architecturally IDENTICAL
to `k_eff_raw`, the model's own raw post-conv keys, for every single C17
query. **`anchor_table.weight`'s own content — its conditioning, its
drift, its `n_iter`-sufficiency at any n_iter — cannot be an input to
a computation it never participates in.** This directly and specifically
falsifies §15.22's own "confined to Newton-Schulz convergence on
EVAL-time recovery-probe queries against the FINAL, fully-learned anchor
table" mechanistic claim (§15.22, "Mechanistic root cause..." paragraph)
— that claim is now known to be wrong, not merely underspecified: the
one pool that ever fails is precisely the one pool structurally
guaranteed to never touch the object that claim names.

### Step 2 result — the pre-registered check, run anyway (as the task
asked), returns an unambiguous, doubly-negative answer

Every one of the 6 pulled tables — the 4 FAILING cells' own
post-training anchor tables AND the 2 PASSING controls' — is **100%
admissible at `n_iter=20` already**, with enormous margin:

| Table | K | role | `n_iter=20` max resid | `n_iter=40` max resid | subset cond # (mean / max) | 6a `sigma_ratio` (full table) |
|---|---|---|---|---|---|---|
| K78/s1840 | 78 | FAILING | 1.42e-06 | 1.29e-06 | 75.5 / 177 | 0.0330 (FAIL, <0.1) |
| K84/s1940 | 84 | FAILING | 1.47e-06 | 1.36e-06 | 78.0 / 213 | 0.1229 (pass) |
| K90/s2040 | 90 | FAILING | 1.56e-06 | 1.43e-06 | 199.1 / 1169 | 0.2929 (pass) |
| K72/s1742 | 72 | FAILING | 1.31e-06 | 1.20e-06 | 28.7 / 60 | 0.0909 (FAIL, <0.1) |
| K69/s1731 | 69 | PASSING | 1.31e-06 | 1.27e-06 | 24.9 / 44 | 0.0528 (FAIL, <0.1) |
| K72/s1741 | 72 | PASSING | 1.28e-06 | 1.19e-06 | 49.1 / 105 | 0.0690 (FAIL, <0.1) |

(`resid_tol=0.01` — every measured residual sits **~7,000×–8,000× below**
the admission threshold, at `n_iter=20`, already. `n_fallback=0/512`
subsets at every `n_iter` tested, every table, no exception.) The 6a
`sigma_ratio` leg (raw, un-normalized 107-row table) fails its own `≥0.1`
bar at 4 of 6 tables — including BOTH passing controls — confirming this
is an expected, benign property of a trained anchor table under this
design (sec 3.1's own 6a rewrite note already flags 6a as measuring
something orthogonal to NS admission) and not itself predictive of
`checkpoint_fallback_seen` in either direction. The per-cell
`item6_table_conditioning` values logged in each cell's own JSON at
`checkpoints[-1]` match this diagnostic's independently-recomputed 6a/6b
to 6+ decimal places at every table (e.g. K78/s1840: JSON `sigma_ratio=
0.032989490777254105` vs. this script's `0.032989565283060074`,
JSON `max_abs_cos=0.37628111243247986` vs. this script's exact match) —
confirms the pulled checkpoints and the extraction path are correct, not
a silent corruption.

**Falsifier evaluation:** the falsifier as literally written ("does
rel-change/admission improve monotonically with n_iter") cannot even be
exercised — there is no room to improve, because admission was never
failing in the FIRST tested `n_iter` (20) for ANY of the 6 real tables.
This is a STRONGER disconfirmation of the pre-registered hypothesis than
the falsifier anticipated: not "iteration count doesn't help an existing
problem" (which would look like `STRUCTURAL-ILL-CONDITIONING`, high
residuals stuck at a floor across n_iter) but "there was never a
numerically-observable problem in the tested object at all." No
difference is detectable between the FAILING cells' own tables and the
PASSING controls' — residuals, cond numbers, and admissibility are
statistically indistinguishable between the two groups.

### Step 3 (exploratory) — even un-engineered random draws don't fail at
these (K, d=96) shapes

`random_unit_rows_init` draws (n=106, no frame-potential optimization,
seeded independently per K) are ALSO 100% admissible at every `n_iter`
tested, up to K=90 (cond # mean 2197, max 13506 — 10-60× worse
conditioned than the real anchor tables above): `n_iter=20` max resid
2.25e-06, `n_iter=40` max resid 1.47e-06. **NS at `n_iter=20`, this exact
production iteration, has enormous headroom for essentially any
reasonably-drawn near-orthogonal K-row set up to K/d=0.9375 in this
codebase** — reinforcing that the observed failure is not a generic
"K approaching d" numerics squeeze either; something specific to the
REAL C17 held-out-entity post-conv key geometry (not reproduced by
either the anchor table or a random proxy) is responsible. This step is
explicitly exploratory and does not by itself carry the verdict.

### Why the true failing object could not be tested this session

C17's actual NS input (`k_eff_raw` for held-out entities — raw,
post-`k_conv` keys) requires the FULL trained model (embed table,
`k_proj`, `k_conv1d` weights) at the failing checkpoint. This wave's own
checkpoint writer (§15.22's own citation, `run_deltanet_rd.py:926–947`,
sec 10.10 item 1) deliberately saves ONLY the anchor table's trained-row
block ("27KB negligible" by design) — never the full model. No artifact
exists, on box or in the archive, from which the real C17 input can be
reconstructed offline. Building one requires either a new checkpoint
field (candidate 1) or a fresh, deterministic-seeded repro run
(candidate 2) — both are follow-up build items, not run here.

## **VERDICT: MISDIAGNOSED-ARTIFACT** — a new, disclosed extension to the
pre-registered three-bucket taxonomy (mirrors this program's own house
convention, e.g. §15.22's own "AMBIGUOUS — DATA-QUALITY COLLAPSE"
extension of row 1b), since neither `NS-ITER-FIX-CONFIRMED` nor
`STRUCTURAL-ILL-CONDITIONING` nor `MIXED` literally fits: the anchor
table is not "structurally ill-conditioned" (residuals sit ~7,000×
below tolerance) and there is no "K/d regime" at which the TESTED object
fails at all. The pre-registered hypothesis targeted the wrong artifact,
confirmed two independent ways: (a) architecturally, C17 —
the ONLY pool that ever fails, in all 12 cells — is anchor-bypassed by
construction; (b) empirically, the anchor table's own NS convergence is
indistinguishable between failing and passing cells and has ~7,000×
margin at `n_iter=20` already. **§15.22's own mechanistic-root-cause
claim is retracted at the mechanism level** (the K/d-correlated failure
RATE pattern §15.22 measured is real and stands unchanged — only the
attributed CAUSE, "the learned anchor table," is wrong); a correction is
owed in `STATE.md`'s matching block.

**Registered candidates (named, NOT designed — a design decision for
whoever builds the follow-up, mirroring this program's own house
convention):**

1. **Extend the checkpoint payload to also snapshot a fixed C17
   diagnostic batch's `k_eff_raw`** (pre-NS, post-conv keys, for a
   reproducible fixed held-out-entity batch) at each admission
   checkpoint — a small, cheap addition (a handful of KB, same
   "eval-truncation lesson" discipline already used for the `Z_dump`
   `S_T_raw` dumps) that would let a future offline diagnostic like this
   one run directly on the TRUE failing object without a re-run.
2. **A targeted, deterministic-seeded repro on ONE already-failing
   cell**, resuming (or re-running) to `step=20000` with a full model
   checkpoint at that one step only, then extracting `k_eff_raw` for the
   SAME C17 batch that logged the fallback (the eval batch sampling is
   deterministic given `eval_gen = seed + 10_000 + step` and the fixed
   pool/hop-sampling call order) — closes the gap this diagnostic could
   not close, at the cost of a short, targeted GPU repro rather than a
   full re-run at more seeds.

No cost estimate is registered for either candidate (both need their own
scoping pass by whoever builds them); this diagnostic itself cost
**~0 GPU-h** (CPU-only local analysis + `scp` pulls of 264KB total from
already-idle box GPUs; no training launched, no tmux touched).

### Archive

`experiment-runs/2026-07-08_ns_admission_diag/` (repo-tracked, all files
≤25MB — total ~300KB): `ckpts/` (6 pulled `step20000.pt` checkpoints, 6×
44KB), `scripts/diag_ns_admission.py` (byte-identical copy of
`matrix-thinking/deltanet_rd/diag_ns_admission.py`), `results/
diag_ns_admission_result.json` (full per-table, per-`n_iter` sweep +
mechanism-breakdown output). SSD mirror, same tree:
`/Volumes/1TB_SSD/learned-representations/experiment-runs/
2026-07-08_ns_admission_diag/`.

---

## §15.24 DESIGN — C17 eval-admission repro instrument (Rev 4, 2026-07-11 —
post-attack round 4)

**Rev 4 status note.** A fourth independent adversarial attack round
reviewed Rev 3 (landed 2026-07-10, §15.24.12) before any GPU work
launched, per this program's own standing multiple-independent-audit-
rounds discipline (`CLAUDE.md`). Verdict: **NEEDS-REVISION** — 0 FATAL, 2
MAJOR, 1 MINOR — narrow, prescribed, surgical; no category error this
round, unlike rounds 1–3. **MAJOR-1 (highest value):** the offline
analysis must reconstruct `pools.heldout_name_ids`/`pools.train_name_ids`
to evaluate Step 0b at all, but nothing in Rev 3's own text pinned HOW —
`build_entity_pools` is reconstructible only via its own `seed` argument,
and `run_deltanet_rd.py:1470` calls it with a HARDCODED `seed=0`
(`grd.build_entity_pools(tokenizer, heldout_frac=args.heldout_frac,
seed=0)`), fully decoupled from the training `--seed` that Rev 3 itself
made a load-bearing per-event field (`episode := (seed, step, hop,
batch_idx, row_idx)`). A builder naively threading the launch seed (1940,
or a contingency seed) into `build_entity_pools` would silently
reconstruct the WRONG train/held-out partition — every real held-out id
would then read as "not in pool," producing a confidently wrong, total
INSTRUMENT-BUG verdict on healthy code, the exact "two seeds" trap this
fix closes. Fixed by pinning the offline reconstruction to the literal,
hardcoded call `grd.build_entity_pools(tokenizer, heldout_frac=0.5,
seed=0)` (verified: `run_deltanet_rd_exactness_sweep.py`'s `build_cmd()`
never emits a `--heldout-frac` flag, so this cell's own launch always
takes the CLI default 0.5, `run_deltanet_rd.py:1290`) — NEVER the launch
seed, with an explicit warning note against the trap — AND adding a new
prerequisite gate: before Step 0b runs on any event, the offline
reconstruction's `pools.train_name_ids` is asserted SET-EQUAL to that
launch's own archived `anchor_train_ids` tensor (`run_deltanet_rd.py:
934–936`, already logged at every checkpoint, zero new cost — the live
run's own ground truth, since `model.anchor_train_ids_buf` is registered
directly from that SAME launch's `pools.train_name_ids`,
`run_deltanet_rd.py:1606`); mismatch → hard-abort the analysis before any
verdict, disclosing the reconstruction failure. §15.24.1's row (b) also
gains one restating sentence: the C17 sampler's heldout-exclusivity
invariant is independently structural — `grammar_rd.py:194–253` draws
non-overlapping slices of ONE shuffled list with globally-unique ids, and
`grammar_rd.py:423,434–436` draws every episode's K entities from a
SINGLE pool (`pools.heldout_name_ids` XOR `pools.train_name_ids`, never
both). **MAJOR-2:** §15.24.7's own ledger baseline (18.1196/26 GPU-h
realized) omitted the K=69/seed=1733 contingency cell's own realized
0.427 GPU-h (`wall_s=1535.2s`, landed 2026-07-08, §15.22 addendum) — a
real cell run standalone, after the 16-cell wide-grid harvest the
6.3331 GPU-h figure sums, and never folded back in anywhere the ledger
is stated. Corrected baseline: 11.7865 + 6.3331 + 0.427 = **18.5466/26**;
every downstream figure re-derived (§15.24.7) — the worst case (2× primary
+ NO-REPRO contingency) becomes 20.3466/21 = **96.89%**, reserve **0.6534
GPU-h** against the ORIGINAL ceiling (down from the previously-claimed
1.0804 GPU-h — a ~40% tightening of the reserve). The conclusion survives
unchanged (still fits without the `+5.0 GPU-h` extension), margins do
not. Folded into `EXPERIMENT_LOG.md`'s running-total convention and
`STATE.md` as well — the gap was repo-wide, not confined to this draft.
**MINOR-1:** added the dedicated minimal-boundary Stage −1 fixture 0b's
own prose already names but no fixture yet exercised — a SINGLE-EVENT
sink (`len(fallback_dump_sink)==1`) with exactly 1 pool-mismatch
violation, asserting INSTRUMENT-BUG fires BEFORE Step −1's own `<3`-event
gate would even run, a more degenerate case than E1 (2 events) or E4 (5
events). Every finding is fixed below (§15.24.1, §15.24.4, §15.24.5,
§15.24.7); none is deferred or waved away. The full finding→fix trace is
recorded at §15.24.13, per house style (mirrors §15.24.10's/§15.24.11's/
§15.24.12's own convention). Nothing in
§15.24.0–§15.24.9/§15.24.10/§15.24.11/§15.24.12 below predates this
revision except where explicitly marked Rev 1/Rev 2/Rev 3 — read the
decision-rule and gate machinery (§15.24.4, §15.24.5) as Rev 4, not as
the Rev-3 text attack round 4 reviewed. **This revision has not yet had
its own verification pass** — round 5, a VERIFY pass confirming Rev 4's
three fixes land clean (not a fresh full attack round), next, per this
project's standing multiple-independent-audit-rounds rule.

**Status: DESIGN-ONLY DRAFT (Rev 4).** Written under the same discipline as
§15.20 Rev 0/§15's own header (§15.17/§15.20.8's self-attack-round
precedent): every number below is either cited to an already-run artifact
(§15.22's per-cell `wall_s`, §15.23's residual/conditioning measurements),
or freshly VERIFIED this session against the live code (file + line, stated
explicitly — every citation touched by this revision was re-checked
against the live files this session, not carried over from Rev 1 by
assumption). No code is written or run in this session; every build task
below (including this revision's own fixes) is registered, not built.

### 15.24.0 What this instrument resolves, and what it explicitly does NOT
reopen

§15.23's own diagnostic closed one question (is the anchor table itself
NS-starved at high K/d?) with an unambiguous NO, and opened a narrower one
it could not answer: **is the TRUE failing object — C17's raw,
un-blended, pre-Newton-Schulz post-conv key for held-out entities — a
genuine geometric-degeneracy case, an instrument artifact, or a
tolerance-miscalibration case?** §15.23's own "Why the true failing object
could not be tested this session" paragraph is the exact gap this design
closes: the wide-grid wave's checkpoint writer (`run_deltanet_rd.py:926–947`,
sec 10.10 item 1) saves ONLY the anchor table's trained-row block, never
the full model (embed/`k_proj`/`k_conv1d`), so `k_eff_raw` for a real C17
query cannot be reconstructed offline from anything currently archived.

**This design builds candidate (2)** (§15.23's own registered list, "a
targeted, deterministic-seeded repro on ONE already-failing cell... with a
full model checkpoint at that one step only") **as the primary
instrument**, and **candidate (1)** ("extend the checkpoint payload to
also snapshot a fixed C17 diagnostic batch's `k_eff_raw`... at each
admission checkpoint") **as a secondary, same-commit code change** for
future waves' benefit — registered here, not separately costed as its own
launch (it adds zero new GPU cells; it only changes what future,
already-planned checkpoint writes contain).

**Explicitly NOT reopened, frozen per §15.1/§15.19/§15.22's own "never
reopens or rescores" precedent:**

1. §15.22's own K/d-correlated failure-**RATE** measurement (0/30 at
   K/d≤0.71875 → 1/3 at K/d=0.75 → 3/3 at K/d≥0.8125) — unchanged, not
   re-measured here. This design explains the RATE's mechanism, never its
   magnitude.
2. §15.23's own anchor-table finding (100% admissible at `n_iter=20`,
   residuals ~7,000–8,000× below tolerance, indistinguishable
   failing-vs-passing) — closed, not re-tested.
3. §15.20.4's own rival-discrimination test (absolute-slack vs power-law
   bands) — this instrument produces ONE more data point about *why*
   K=78/84/90 are inadmissible, not a new x0(96) fit; it does not revive
   §15.20.4 on its own.
4. **Build-scope fence, restated from the task brief:** this design's own
   build tasks touch `run_deltanet_rd.py`, `model_rd.py`, and
   `run_deltanet_rd_exactness_sweep.py` only (plus one new standalone
   analysis script, §15.24.4). **No `phase2_*` file and no
   `lm_pretrain_rd.py`** — those are different, unrelated tracks; nothing
   in this design touches them, and no future build session executing this
   design is authorized to either.

### 15.24.1 Hypothesis and the 3-way mechanism space

**One-sentence hypothesis:** the raw, pre-anchor, pre-Newton-Schulz
post-conv keys C17 feeds to NS at high K/d either (a) really are
geometrically non-recoverable at the production `n_iter=20`/`resid_tol=
0.01` setting — a genuine, previously-unmeasured boundary on the
super-linear capacity result — or (b) trigger the eval-side fallback flag
for a reason unrelated to true NS non-convergence on that object (a bug
upstream of NS itself), or (c) genuinely fail to converge at `n_iter=20`
but would pass at a modestly larger iteration budget or a tolerance
calibrated for THIS population rather than for anchor-stabilized blended
keys.

| Candidate | Mechanism, precisely | What observation discriminates it |
|---|---|---|
| **(a) REAL-CAPACITY-BOUNDARY** — genuine geometric degeneracy of raw held-out-entity keys at K/d>0.75 | The C17 probe is CORRECTLY reporting that raw held-out-entity keys, at these K/d ratios, are not NS-recoverable at the production setting — a real fact about the model's learned key geometry for entities it never anchors, not an artifact. | Offline-recomputed NS on the dumped `k_eff_raw` (same `n_iter=20`, same math) **reproduces** the live fallback (agrees with the live flag, §15.24.4 Step 1), **AND** sweeping `n_iter∈{24,28,32,40}` on that SAME tensor **does not** drive every flagged episode's residual below 0.01 (structural non-convergence, not "one more iteration away"). |
| **(b) INSTRUMENT-BUG** — a defect in the C17 probe path itself (wrong pool row-count, a K-vs-pool-size mismatch, subset selection including near-duplicate rows) | The fallback fires for a reason that has nothing to do with NS struggling on a genuinely well-posed K-row set — e.g. the sampled "K distinct held-out entities" batch is not actually K linearly-independent rows, or the pool/К bookkeeping is wrong. | **(Rev 3, FATAL fix — the joint floor below was split since Rev 2's own text)** The DISPOSITIVE trigger is Step 0b's pool-membership check (§15.24.4, a zero-tolerance structural fact, checked FIRST in Rev 3's own precedence order) — dispositive on ANY SINGLE violation, no event-count minimum, since pool membership is a structural fact, not a statistic — OR Step 1's live/offline NS disagreement (§15.24.4), dispositive once its OWN, unchanged two-level episode/event recurrence floor is met (a numerical near-boundary disagreement genuinely can be one unlucky draw, unlike 0b's structural fact). The exact-rank precheck (Step 0a, `rank(k_eff_raw_episode) < K`) is **CORROBORATING ONLY** (Rev 1 demotion, unchanged by Rev 2/Rev 3) — reported alongside whatever 0b/Step 1 find, never independently dispositive. **(Rev 4, verified note — round 4's own re-check of the sampler this row's check depends on):** the C17 sampler's heldout-exclusivity invariant is itself structural, not merely assumed — `grammar_rd.py:194–253`'s `build_entity_pools` draws `train`/`heldout` as two NON-OVERLAPPING slices of ONE shuffled name list with globally-unique ids (`seen_ids` dedup across every candidate list before the split), and `grammar_rd.py:423,434–436`'s `sample_batch_rd` draws every episode's K entities from a SINGLE pool selected by one ternary (`pools.heldout_name_ids if use_heldout_entities else pools.train_name_ids`, never a union of both) — so a genuine 0b violation can only originate upstream of this invariant (a stale/mis-seeded pool reconstruction, MAJOR-1 below), never from the sampler mixing pools within one draw. |
| **(c) TOLERANCE-MISCALIBRATION** — numerically real non-convergence at `n_iter=20`, but the 0.01 tolerance (calibrated for anchor-stabilized BLENDED keys) is simply too tight for the structurally different, un-stabilized raw-key population | NS genuinely does not hit `resid≤0.01` at `n_iter=20` on the real object, but this is a near-miss the existing numerical budget already fixes, not a structural wall. | Offline recompute agrees with the live flag (rules out (b)) **AND** the `n_iter∈{24,28,32,40}` sweep DOES drive every flagged episode below 0.01, at or before `n_iter=32` (the same grid §15.23 already used) — an iteration-count-fixable near-miss, the mirror image of (a)'s outcome on the identical sweep. |

**Corroborating (non-triggering) context pinned before the run, per
requirement #3:** the dumped raw-key residuals will be compared, as a
descriptive scale check only, against §15.23's own already-measured
blended-key residuals (~1.3–1.6e-6, "7,000–8,000× below tolerance") and
its random-proxy condition numbers (mean 2197 / max 13506 at K=90, ALL
of which still passed at `n_iter=20`) — useful for characterizing HOW
different the raw C17 population is, but neither quantity is a verdict
trigger by itself (§15.23 already showed condition number alone does not
predict admissibility: even a 2197-mean-conditioned random draw converged
fine).

### 15.24.2 Instrument — candidate (2), primary: one deterministic repro cell

**Cell: K=84, seed=1940, d_state=96 — "mid-pattern," justified, not
arbitrary.** Per §15.22's own per-cell table, K=78/84/90 are each 0/3
admissible (fully collapsed), structurally distinct from K=72 (1/3
admissible — a partial, noisier regime) and from K=90 (the pool-margin
extreme: only 16 of 106 held-out entities are NOT drawn per K-item pool
draw at K=90 (Rev 3, MINOR fix — "per episode" here meant per-row
K-entity draw, a THIRD, distinct usage of the word from the two Rev 2's
own FATAL fix pinned as `episode`/`event`; reworded to "per draw" so the
now-precisely-defined term is not re-overloaded), vs. 22 of 106 at K=84 —
closer to the pool-exhaustion boundary,
a possible SECOND confound this design wants to avoid conflating with
the geometric question). **K=84 sits at the middle of the three
uniformly-failing K's**, making its diagnosis the most generalizable
read on the K=78–90 band without either the K=72 partial-failure
ambiguity or the K=90 pool-margin confound. **Seed 1940 is the primary
(first-listed) seed for K=84** (`run_deltanet_rd_exactness_sweep.py:3095`:
`84: (1940, 1941, 1942)`), and — critically, per §15.23's own disclosed
discrepancy (its preferred K=72/seed=1740 had **zero** checkpoints
anywhere on box, `ckpt_written` empty) — **K84/seed=1940 is already
confirmed to have a complete, on-box 10-checkpoint tree** (§15.23's own
checkpoint-pull table: "K78/s1840, K84/s1940, K90/s2040, K72/s1742" were
the 4 failing cells actually pulled). Reusing this exact cell means the
repro targets a cell already known-good for checkpoint availability,
sidestepping the exact landmine §15.23 already hit once.

**Byte-identical training path, sha256'd, not hand-typed.** The cell's
CLI is reconstructed via `_keyanchor_scaling_spec(K=84, seed=1940,
d_state=96)` + `build_cmd(spec, ...)`
(`run_deltanet_rd_exactness_sweep.py:2655–2677`, `:3781–3855`) — the SAME
function that emitted the original wave's own command, never hand-typed
independently. This resolves to (all flags confirmed against the live
function bodies this session):

```
python run_deltanet_rd.py --K 84 --steps 20000 --seed 1940 \
  --internal-timeout <inherited from the wave dispatcher's standard per-cell timeout> \
  --out <out_path> \
  --use-geo3 --geo3-n-iter 20 --geo3-resid-tol 0.01 \
  --anchor-active --anchor-lambda-mode learned \
  --drift-probe --rev7-engagement --d-state 96 \
  --ckpt-dir <ckpt_base_dir>/wkeyanchor-scaling-c17repro_rdx_K84_armd_s1940_...
```

**Required Stage −1 build step (mechanical, not a code change):** hash
this emitted command list (`build_cmd`'s own return value) and diff, by
hand, every architecture-relevant field (K, d_state, seed, `geo3_n_iter`,
`geo3_resid_tol`, `anchor_active`, `anchor_lambda_mode`, `drift_probe`,
`H_extra` default `[7,21]`) against the ALREADY-ARCHIVED K84/seed=1940
result JSON's own logged `exactness_config`/architecture-pins block
(§15.22's own per-cell table already confirms these are uniform across
all 19 cells: `anchor_active=True`, `anchor_lambda_mode="learned"`,
`geo3_n_iter=20`, `H_extra=[7,21]` unmodified). **The ONLY sanctioned
delta from the byte-identical base command is the new
`--c17-repro-telemetry` flag (below) and a NEW `--ckpt-dir` target** (never
overwriting the original archived checkpoints). Mirrors §15.20.1's own
Gate (c) sha256 discipline exactly, applied here to a command line instead
of a copied file.

**No new kernel or Gate-2 check needed — both already cleared for this
exact (K,d) pair.** `T_bind(K=84) = 7×84 = 588` (`grammar_rd.py:325–327`,
`T_bind=7K`) is literally one of the four values §15.20.1's own Stage −1
kernel probe already tested (`T∈{504,546,588,630}`, §15.20.6 Stage −1 item
1) and passed. `GATE2_N_ITER_BY_D_K[96][84]=20` was already confirmed
sufficient by the original wave's own Wave −1 n_iter-sufficiency check
(§15.20.6 Stage −1 item 2). This repro reuses BOTH artifacts by citation,
building nothing new for either gate.

**Telemetry additions — ONE new CLI flag, `--c17-repro-telemetry`
(`store_true`, default `False`), gating three additive, read-only
side-channel taps (Rev 1, attack-round-1 MAJOR M1 fix: item (ii)'s own
file format gains the two TF32-mode fields described below — no new
numbered tap, no new build task beyond threading two `torch.backends`
reads into item (ii)'s existing dict construction). Byte-identical to
every existing cell when the flag is absent (default behavior unchanged
for every other wave, past or future):**

**(i) Full model `state_dict` snapshot at `step=20000` only.** New flag
`--full-ckpt-step N` (`type=int`, default `None`); when set and
`step==N` and `ckpt_dir is not None`, additionally
`torch.save({k: v.detach().cpu().clone() for k, v in
model.state_dict().items()}, os.path.join(ckpt_dir, f"full_step{N}.pt"))`
— a NEW artifact, separate from (and additive to) the existing lightweight
`step<N>.pt` anchor-only payload (`run_deltanet_rd.py:934–947`, unchanged).
This is the ONLY way to load the real `embed`/`k_proj`/`k_conv1d` weights
needed to recompute a genuine C17 batch offline later.

**(ii) The C17 diagnostic batch's raw pre-NS `k_eff_raw`, dumped at every
live fallback event.** `evaluate_pool()` (`run_deltanet_rd.py:231–357`)
gains THREE new optional, additive parameters (Rev 2, MAJOR-4 fix, plus
Rev 3's own `seed` addition below — verified this session: the function's
real signature, `run_deltanet_rd.py:231–234`, carries none of the three
today): `fallback_dump_sink: list | None = None`, `step: int | None =
None`, and `seed: int | None = None` (Rev 3, MAJOR-A fix). All three
default to values that are zero behavioral change at every existing call
site — the M2/M3/C19 calls in `train()` at lines 802–804/807–808 pass
none of them, so all three stay `None` there, exactly as before either
fix. The C17 call site (`run_deltanet_rd.py:805–806`) is the ONLY call
site that passes non-default values for any of the three, and only when
`--c17-repro-telemetry` is set: `step=step` (the enclosing `train()`
loop's own already-in-scope loop variable, `run_deltanet_rd.py:687`, `for
step in range(1, steps + 1):`), `seed=seed` (Rev 3, `train()`'s own
already-in-scope function parameter, `run_deltanet_rd.py:584`, `def
train(model, cfg, pools, pool_report, device, d_model, d_state,
steps=6000, batch_size=128, lr=3e-4, seed=0, ...)`), and
`fallback_dump_sink=<live list>`. **(Rev 2, found this session while
re-verifying this exact code block, folded into the same build task since
it touches the identical lines):** the per-batch loop
(`run_deltanet_rd.py:263`) is `for _ in range(n_batches):` today — no
loop-index variable — so the dump dict literal's own `"batch_idx":
batch_i` field referenced an undefined name; Rev 2 additionally registers
renaming the loop to `for batch_i in range(n_batches):` (additive,
`batch_i` unused elsewhere in the function today, zero behavior change
otherwise). Inside that loop, immediately after
`model.geo3_last_fallback_triggered` is read (line 274–275), if it is
`True` AND `fallback_dump_sink is not None`, append
`{"seed": seed, "step": step, "hop": h, "batch_idx": batch_i,
"key_ids": b["key_ids"].detach().cpu().clone(),
"k_eff_raw": model.geo3_last_k_eff_raw.detach().cpu().clone(),
"k_blend_raw": model.anchor_last_k_blend_raw.detach().cpu().clone(),
"resid": model.geo3_last_resid.detach().cpu().clone()}` — same fields as
Rev 2's own spec plus the new `"seed"` key (Rev 3, MAJOR-A fix); `step`
(corrected from the placeholder `step_arg_passed_in`, MAJOR-4's own
finding) and `batch_i` remain well-defined exactly as Rev 2 left them.
**Why `seed` is needed (Rev 3, MAJOR-A fix — full definition at §15.24.4):**
Step −1's own NO-REPRO contingency path (Rev 1 FATAL F1) can combine
events from UP TO 3 separate launches (the primary seed plus, on a
NO-REPRO, seeds 1943/1944) into one `fallback_dump_sink` before the
decision rules run. `(step, hop, batch_idx)` alone is NOT launch-unique —
each independent launch re-runs the SAME training loop, so a contingency
launch can produce an event at the identical `(step, hop, batch_idx)`
coordinates as the primary launch's own dump; naively concatenating dump
sinks without a launch discriminator would silently coalesce two
INDEPENDENT reproductions into what analysis code reads as "1 event,"
corrupting the two-level floor's own event-count arithmetic (Step 1's
floor needs ≥2 DISTINCT events — a coalesced pair reads as 1). `seed`
closes this: each of the ≤3 launches uses a distinct seed (1940 primary,
1943/1944 contingency) by construction, so `seed` doubles as a launch
identifier without adding a separate field. **Disclosed, not a new
finding to fix:** cross-launch recurrence AT THE IDENTICAL `(step, hop,
batch_idx)` coordinates (different `seed`, same batch position) is in
fact the STRONGEST possible reproduction evidence available to this
instrument — an independently-reseeded run hitting the exact same
checkpoint/hop/batch position and reproducing the exact same anomaly is a
stronger signal than two different coordinates within one launch — and is
disclosed as such in the output JSON, never discounted. (Separately, the
AMBIGUOUS-RESIDUAL follow-on, §15.24.4, re-launches the identical cell a
SECOND time at the SAME seed for a reproducibility check; that comparison
is between two independent dump sinks evaluated side by side, never
merged into one combined sink for floor-counting, so the same-seed reuse
there does not reintroduce the aliasing risk this fix closes.)
**Episode addressability (Rev 2, FATAL fix; Rev 3, MAJOR-A extends the
tuple; full definition at §15.24.4):** this remains a per-EVENT
(per-triggering-batch) record, not per-row — `key_ids`/`k_eff_raw`/
`k_blend_raw` keep their existing `(B,84[,96])` shapes and `resid` keeps
`(B,)` (Stage −1 item 3, unchanged); no new per-row tensor field is added
by either fix, only the scalar `"seed"` key at the event-dict level. Row-
level analysis (which of the `B` rows are individually anomalous) is
computed OFFLINE, at analysis time, by indexing these SAME already-dumped
per-row tensors positionally (`row_idx ∈ [0,B)` IS the tensor's own row
position — `key_ids[row_idx]`, `resid[row_idx]`, etc. all name the same
episode by construction, since every field in one event dict is read off
the same forward call on the same batch) — §15.24.4 defines exactly how.
**Dumping BOTH `geo3_last_k_eff_raw` and `anchor_last_k_blend_raw` is
deliberate and free** — §15.23 already proved these are architecturally
IDENTICAL for C17 items (`anchor_blend_gather_scatter`,
`key_anchoring.py:439–469`, `trained_here=False` for 100% of C17 keys);
dumping both and asserting bitwise equality at analysis time is a live,
free re-confirmation of that claim on THIS run, not an assumption carried
over from a different cell. **(Rev 3, MINOR fix — the failure mode was
previously unstated):** if this bitwise-equality assertion ever fails on
a real dumped event, that is an explicit **HARD-ABORT**
(`assert torch.equal(...)`, uncaught, process exit non-zero) — never a
warning, never a soft-skip-and-continue — because a failure would mean
§15.23's own anchor-bypass claim (Q3, independently re-verified CLEAN at
attack round 1, §15.24.9) does not hold for THIS run, which invalidates
every downstream Step 0b/0a/1/2 verdict computed from `k_eff_raw` under
that assumption; the whole decision-rule analysis is untrustworthy the
instant this check fails, so it must stop the analysis, not merely flag
one event. All events for one checkpoint's C17 call are
bundled into one file, `<ckpt_dir>/c17fallback_step<N>.pt` — **(Rev 1,
attack-round-1 MAJOR M1 fix) now a dict, not a bare list**:
`{"tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn":
torch.backends.cudnn.allow_tf32, "events": [<the per-event dicts above>]}`
— the two TF32 flags are read ONCE, at process start (a global,
process-level setting, not a per-event one; `evaluate_pool()` itself never
touches them), so recording them once per checkpoint file rather than once
per event is exact, not an approximation. This is the offline analysis's
ONLY way to know which precision mode the LIVE run actually used, closing
the gap M1 found: without this tap, Step 1's offline recompute (§15.24.4)
had no way to match the live run's own numerics and risked a false
INSTRUMENT-BUG verdict on a TF32-mode mismatch it couldn't even detect.
Otherwise mirrors the existing one-file-per-checkpoint convention.

**(iii) Per-probe-pool admission telemetry, all four pools, every
checkpoint.** `evaluate_pool()`'s existing per-hop accumulation loop
already computes `model.geo3_last_resid` on every batch (`model_rd.py:1149`,
Rev 2 MINOR m2 fix — mis-cited `:149` in Rev 1) but currently only ORs it into a single
`fallback_this_hop` boolean (line 274–275, discarding the magnitude).
Gated behind the same flag: accumulate `resid_all.append(model.
geo3_last_resid.detach().cpu())` alongside the existing `cos_all`/
`norm_all` accumulation (mirrors that exact pattern, lines 276–277), and
when `model.geo3_active` add `entry["geo3_resid_stats"] = {"mean":
torch.cat(resid_all).mean().item(), "max": torch.cat(resid_all).max()
.item(), "min": torch.cat(resid_all).min().item(), "n_batches_fallback":
<count of batches where resid.max()>tol>, "n_batches_total": n_batches}`
to EVERY pool's entry (M2/M3/C17/C19 alike) — giving, for the first time,
a magnitude comparison across pools instead of only the binary flag
(`geo3_fallback_triggered_this_hop`) the codebase currently logs.

**Required negative test (regression, mirrors `bind()`'s own `return_beta`
precedent, `model_rd.py`'s bind() docstring, ~line 1074–1076: "Wave −1
smoke: bitwise-identical loss with the flag on vs off on a fixed
seed/batch"):** run a short smoke (e.g. 50 steps, fixed seed) with
`--c17-repro-telemetry` ON vs OFF and assert bitwise-identical
`trajectory`/loss curve — every new tap is `.detach()`ed and never
consumes from `gen`/`eval_gen`/any model-internal generator, so it MUST
NOT perturb the training path; this smoke proves it, not merely asserts
it by construction.

    **[Deploy-session build note (2026-07-07, disclosed deviation — Stage −1
    item 1 re-pin):** the bitwise form of this smoke FAILED its first real
    box execution, and the discriminating OFF-vs-OFF diagnostic
    (`logs/c17repro_item1_off_vs_off_diag.json` on box, three dense
    `log_every=1` runs) adjudicated the cause as same-flag fixed-seed GPU
    run-to-run nondeterminism (OFF-vs-OFF max-abs loss dev 7.5e-04 from
    step 4 — the SAME already-measured phenomenon `KEY_ANCHORING_DESIGN.md`
    ~L1976–1994 documents and §15.24.4's own Step −1/F1 premise is built
    on), NOT a telemetry perturbation (OFF-vs-ON devs 7.3e-04/6.0e-04, both
    inside the OFF-vs-OFF envelope; code-level causal cross-check: the
    fixture's only telemetry-affected code path runs AFTER the step-50 loss
    is logged). Item 1 is re-pinned baseline-relative: OFF×2 (envelope) +
    ON×1, PASS iff max-abs OFF-vs-ON dev ≤ 3× the OFF-vs-OFF envelope, with
    envelope = 0 reducing to the original bitwise spec (a strict
    generalization on deterministic hardware). The intent of the registered
    test — "the flag must not perturb the training path" — is unchanged;
    only the satisfiable form of the assertion is.]**

**Required negative test, TF32-recording (Rev 1, attack-round-1 MAJOR M1
fix):** launch the same short smoke twice, once with
`torch.backends.cuda.matmul.allow_tf32` explicitly forced `True` before
process start and once forced `False`, both with
`--c17-repro-telemetry` on; assert the recorded `tf32_matmul` field in the
resulting `c17fallback_step<N>.pt` matches the FORCED value each time
(not a stale default, not always `True`/always `False` regardless of the
forced setting) — proves the tap actually reads the live process state at
dump time rather than a hardcoded or cached value.

### 15.24.3 Candidate (1), secondary — fixed-batch checkpoint payload
extension (same commit, zero new GPU cells)

Registered per §15.23's own text: a FIXED (not per-checkpoint-resampled)
C17 diagnostic batch, dumped at every admission checkpoint of every future
`geo3_active` wave, for general offline diagnosability — distinct from
§15.24.2 item (ii)'s event-triggered dump of the REAL, live per-checkpoint
C17 batches (which is what THIS repro's own decision rule needs, since a
merely-fixed batch might happen not to be one of the ill-conditioned
draws). **Build mechanics: reuse the EXISTING fixed-batch pattern already
in `train()` almost verbatim** (`run_deltanet_rd.py:681–684`, the
`b_fixed`/`s_ideal_fixedref` machinery): add
`c17_fixed_gen = torch.Generator(device=device).manual_seed(seed +
40_000)` and `c17_b_fixed = sample_batch_rd(cfg, min(32, batch_size),
c17_fixed_gen, hop_set=(cfg.H_train[0],), pools=pools, device=device,
use_heldout_entities=True)`, sampled ONCE before the training loop exactly
like `b_fixed`. At every checkpoint, gated behind a NEW flag
(`--ckpt-dump-c17-fixed-batch`, default `False`, byte-identical when
absent), run `model.bind(c17_b_fixed, force_rank_k=force_rank_k)` in eval
mode and stash `geo3_last_k_eff_raw.detach().cpu()` into the checkpoint
payload alongside the anchor-table block. **Not separately costed** — it
adds one extra `bind()` call (a single forward-only pass on ≤32 items,
sub-millisecond) at each of the ≤10 checkpoints already being written;
folded into the same commit as §15.24.2's build tasks, never launched on
its own.

### 15.24.4 Decision rules — mechanical, pinned before the run (Rev 3,
attack-round-3 fixes; Rev 4, attack-round-4 MAJOR-1 fix)

All thresholds below are EXACT, no numerical-tolerance slack, per this
project's own standing rule (an integer/structural check needs an exact
threshold, not a tolerance copied from a floating-point context).
`resid_tol = 0.01` throughout (`key_anchoring.GATE2_RESID_TOL`, the SAME
production constant this program uses everywhere — never re-derived).
`n_iter` sweep grid `{20, 24, 28, 32, 40}` — the SAME grid §15.23's
`anchor_table_ns_sweep()` already used, for direct comparability. **Rev 1
changes applied throughout this subsection** (full trace at §15.24.10):
a new Step −1 zero/low-event guard (FATAL F1 fix); per-episode
granularity thresholds replacing every prior "ANY episode" dispositive
trigger in Steps 0a/0b/1 (MAJOR M2 fix); dual TF32-mode pinning for
Step 1's offline recompute (MAJOR M1 fix); Step 0a demoted from
dispositive to corroborating (MINOR m2 fix). **Rev 2 changes applied
throughout this subsection** (full trace at §15.24.11): episode redefined
as a within-batch row, replacing the row/event-conflated granularity rule
with a two-level absolute floor (FATAL fix); 0b reordered ahead of Step −1
and given its own enforced-abort branch (MAJOR-2/MAJOR-3 fixes); Step 1's
live/offline check now runs per-episode using the already-dumped per-row
`resid`, not the aggregate flag; TF32 matched-mode pinned per-source-run
(MINOR m3); the determinism cross-check now runs per-launch (MINOR m4).
**Rev 3 changes applied throughout this subsection** (full trace at
§15.24.12): the two-level floor SPLIT — Step 0b is now dispositive on ANY
SINGLE pool-membership violation, no floor of its own (FATAL fix, a noise
argument was wrongly copied onto a structural, zero-floating-point check);
Step 1 keeps Rev 2's own two-level floor unchanged, now explicitly scoped
to Step 1's marker alone; `episode`/`event` both gain a `seed` field,
making event identity launch-unique across the up-to-3-launch combined
sink (MAJOR-A fix); Step 1's offline recompute now runs ONE batched call
on each event's full dumped tensor, matching the live call's own batching,
instead of a batch-size-1 slice (MAJOR-B fix); a cross-marker negative
test, an explicit hard-abort on the `k_eff_raw`/`k_blend_raw` bitwise
re-confirmation, and a rename of "residual AMBIGUOUS" to
**AMBIGUOUS-RESIDUAL** (six MINORs, full list at §15.24.12). **Rev 4
changes applied throughout this subsection** (full trace at §15.24.13): a
NEW prerequisite gate, ahead of even Step 0b, pins offline pool
reconstruction to a hardcoded `seed=0` (never the launch seed) and
cross-checks the reconstructed `train_name_ids` against the checkpoint's
own archived `anchor_train_ids` tensor, hard-aborting on mismatch before
any verdict (MAJOR-1 fix); the ledger correction (MAJOR-2) and the new
minimal-boundary negative test (MINOR-1) are scoped to §15.24.7 and
§15.24.5 respectively and do not otherwise change this subsection's own
decision-rule text.

**Episode and event, precisely defined (Rev 2, FATAL fix).** Rev 1's own
text used "episode" for two different objects without ever pinning which:
Step 0a's own rank check already operated on ONE **row** of a dumped
batch (`rank(k_eff_raw_episode)`, a `(K,d)=(84,96)` matrix), while the
granularity-threshold paragraph's own "up to ~120 dumped events" cost
figure (§15.24.7) counted **whole triggering batches**. The codebase's own
docstring draws exactly this row-vs-batch line already: `geo3_
orthogonalize`'s docstring (`model_rd.py:433–434`) explicitly frames the
fallback trigger as firing "if ANY episode's residual exceeds resid_tol
after n_iter -- batch-level, not episode-level, granularity" — i.e. the
codebase's OWN vocabulary already reserves "episode" for one row and
treats the whole-batch OR as a separate, coarser unit. Rev 2 adopts this
distinction explicitly and gives the coarser unit its own name:

- **Episode := the 5-tuple `(seed, step, hop, batch_idx, row_idx)`**
  (Rev 3, MAJOR-A fix: extends Rev 2's own 4-tuple with `seed`, below) —
  ONE within-batch `(K,d)` problem, `row_idx ∈ [0, B)`. `newton_schulz_
  orthogonalize`'s own `resid` return is `(B,)`, one scalar per episode
  (`model_rd.py:377`/`:397`, `resid = (X @ X.transpose(-1,-2) -
  I_K).norm(dim=(-2,-1))`), and `bind()` stashes it unreduced at
  `self.geo3_last_resid = resid.detach()` (`model_rd.py:1149`, corrected
  citation — Rev 2 MINOR m2 fix, mis-cited `:149` in Rev 1) — the per-row
  residual vector already exists at dump time; Rev 2's fix is to actually
  USE it (below), not to create a new tensor.
- **Event := one dumped dict, identified by `(seed, step, hop,
  batch_idx)`** (Rev 3, MAJOR-A fix: extends Rev 2's own `(step, hop,
  batch_idx)` triple with `seed`). `fallback_dump_sink` gains one entry
  whenever the WHOLE-BATCH OR (`model.geo3_last_fallback_triggered`,
  itself an aggregate over every row in that batch — `geo3_
  orthogonalize_logged`'s `fallback_triggered = bool((resid >
  resid_tol).any())`, `model_rd.py:456`) is `True` — unchanged from
  Rev 1, §15.24.2 item (ii). One event bundles up to `B=128` episodes
  (dumped `key_ids.shape == (B,84)`, matching §15.24.5 Stage −1 item 3's
  own shape assertion) — only a SUBSET of which typically has
  `resid[row_idx] > resid_tol` individually, since the trigger is an OR
  over the whole batch, not a per-row gate. §15.24.7's own "~120 dumped
  events" figure (4 batches × 3 `H_train` hops × ≤10 checkpoints) counts
  EVENTS, not episodes; the corresponding worst-case EPISODE count
  examined is `120 events × 128 rows/event = 15,360 episodes` — the
  ~128× gap between these two readings is exactly what made Rev 1's
  dispositive-floor arithmetic ambiguous (attack-round-2 FATAL).
- **Why `seed` (Rev 3, MAJOR-A fix — full rationale, and the disclosed
  scope boundary against the AMBIGUOUS-RESIDUAL follow-on's own same-seed
  re-launch, at §15.24.2 item (ii)):** `(step, hop, batch_idx)` alone is
  NOT launch-unique once Step −1's own NO-REPRO contingency path (Rev 1
  FATAL F1) can combine events from up to 3 separate launches (primary
  seed 1940, contingency seeds 1943/1944) into one `fallback_dump_sink`.
  Each launch independently re-runs the identical training loop, so two
  DIFFERENT launches can produce dumped events at the IDENTICAL `(step,
  hop, batch_idx)` coordinates; concatenating their dump sinks without a
  launch discriminator would silently coalesce two INDEPENDENT
  reproductions into what the analysis script reads as "1 event,"
  corrupting Step 1's own `≥2 distinct events` floor arithmetic (a
  genuinely-recurring 2-launch signal could undercount as 1 and wrongly
  fail to clear the floor). `seed` closes this at zero marginal cost — a
  new key already available in-scope at every call site, requiring no new
  tensor.
- **Cross-launch recurrence at identical `(step, hop, batch_idx)`
  coordinates (different `seed`) is the STRONGEST reproduction evidence
  this instrument can produce, and is disclosed as such, never
  discounted** — an independently-reseeded launch hitting the exact same
  checkpoint/hop/batch position and reproducing the exact same anomaly is
  stronger evidence of a systematic defect than two different coordinates
  within a single launch.
- **No new per-row tensor field is required for either fix.** `row_idx`
  is simply the tensor position within an event's own `(B,...)`-shaped
  fields — `key_ids[row_idx]`, `k_eff_raw[row_idx]`, `resid[row_idx]` all
  name the SAME episode by construction (every field in one event dict is
  read off the same forward call on the same batch, §15.24.2 item (ii));
  `seed` is one new SCALAR key at the event-dict level (Rev 3). The build
  task this fix registers is entirely in the ANALYSIS script (below) and
  in the output JSON's disclosure format (every flagged/excluded episode
  is now named by its full `(seed, step, hop, batch_idx, row_idx)`, not
  just `(step, hop, batch_idx, row_idx)`), not in any NEW training-side
  telemetry hook beyond the one additive `seed` key §15.24.2 registers.

**Two-level dispositive floor, Rev 2's own text (FATAL fix at the time —
replaced Rev 1's row/event-conflated "≥2 episodes, OR ≥2% of dumped
events, whichever larger" rule).** Re-deriving Rev 1's own worst-case
arithmetic at the CORRECT (episode) granularity exposed why the
inherited percentage clause could not survive the row/event distinction
above: at `120` events × `128` rows/event `= 15,360` episodes examined,
`⌈2% × 15,360⌉ = ⌈307.2⌉ = 308` episodes — a bar a genuinely broken probe
path (a K-vs-pool-size off-by-one, a stale tensor alias) would plausibly
never clear, since such a bug corrupts a HANDFUL of rows per triggering
batch, not hundreds across an entire single-cell repro; pinning the
percentage clause at episode granularity would have made INSTRUMENT-BUG
nearly unreachable regardless of whether the probe really is broken,
silently defeating the clause's own original purpose (M2, round 1: "more
than a fluke among the dumped anomalies"). Rev 2 dropped the percentage
clause and pinned a single two-level ABSOLUTE floor, applied — Rev 2
believed — uniformly to Step 0b and Step 1 alike.

**Rev 3, FATAL fix — Rev 2's own floor was a NOISE argument wrongly
applied to a STRUCTURAL check.** "A systematic bug recurs across
independent draws; an ordinary tail event does not" is a statistical
argument, built to protect a NUMERICAL check (Step 1's live/offline
residual comparison, where a single episode landing on the wrong side of
the `0.01` threshold in only ONE of two floating-point computations is
genuinely consistent with ordinary near-boundary noise) against one
unlucky sampled row. Step 0b's own check carries no such risk: a dumped
`key_ids` row is a set of integer entity ids, and its membership in
`pools.heldout_name_ids` is computed with zero floating-point
arithmetic — either the sampled batch drew an entity NOT in the disjoint
held-out pool, or it did not. There is no "unlucky draw" reading of a
single pool-membership violation; a lone occurrence is already
deterministic proof the probe path is broken (a K-vs-pool-size bug, a
stale pool reference, an off-by-one in the train/held-out split), exactly
the class of fact this project's own standing rule addresses:
"integer/structural correctness checks... need EXACT thresholds... a
numerical-tolerance slack copied from a floating-point context...
silently defeats single-instance violations." Gating Step 0b behind Rev
2's recurrence bar copied that floating-point-context tolerance onto a
structural check by analogy, not by any property Step 0b's own check
actually has — concretely, a real pool-mismatch in a 5-event sink
previously fell below the 2-event floor, was EXCLUDED, and the verdict
silently continued to REAL-CAPACITY-BOUNDARY or TOLERANCE-MISCALIBRATION
on the untainted remainder: a confidently wrong claim about the model's
own key geometry, built on top of a probe defect the instrument itself
had already detected and then discarded. The floor is split below.

> **Step 0b's rule (Rev 3): dispositive on ANY SINGLE pool-membership
> violation. No event-count minimum. No episode-count minimum.** One
> mismatched episode, in a sink of any size — including a single-event
> sink, before Step −1's own `<3`-event reproduction gate has even run —
> → **INSTRUMENT-BUG**, immediately, exactly mirroring
> `model_rd.py:2048`'s own `assert (... == 0).all()` convention for a
> different structural fact (an anchor-table gradient at held-out rows
> that must be EXACTLY zero, never "zero within tolerance given enough
> samples"). There is no "exactly 1 → excluded" case for Step 0b's own
> marker any longer.

**Offline pool reconstruction — the two-seeds trap (Rev 4, MAJOR-1
fix).** Step 0b's own check (above) is only as trustworthy as the
`pools.heldout_name_ids`/`pools.train_name_ids` the offline analysis
script reconstructs to evaluate it against — and nothing before this
revision pinned HOW that reconstruction happens. `build_entity_pools`
(`grammar_rd.py:194`) takes `seed` as an explicit argument, and the ONLY
caller in the actual training path, `run_deltanet_rd.py:1470`, calls it
as `grd.build_entity_pools(tokenizer, heldout_frac=args.heldout_frac,
seed=0)` — a **HARDCODED `seed=0`**, decoupled from `--seed` (the CLI
flag this repro's own byte-identical command sets to 1940, and which Rev
3's own MAJOR-A fix made a load-bearing per-event field, `episode :=
(seed, step, hop, batch_idx, row_idx)`). **The trap:** an offline builder
threading the LAUNCH seed (1940, or a contingency seed 1943/1944) into
`build_entity_pools(seed=...)` — a natural mistake, since `seed` is
in-scope and "the run's seed" reads as the obviously correct value to
pass — would silently draw a DIFFERENT `random.Random(seed).shuffle(...)`
permutation, reconstructing the WRONG train/held-out partition. Every
genuinely held-out id would then fail set-membership against the
wrongly-reconstructed `pools.heldout_name_ids`, and Step 0b's own
any-single-violation rule (above) would fire **INSTRUMENT-BUG on every
single dumped episode** — a total, confidently wrong verdict on
healthy code, indistinguishable in its own output from a genuine probe
defect. **Fix, pinned in the Step 0b build task:** offline pool
reconstruction is **always** `grd.build_entity_pools(tokenizer,
heldout_frac=0.5, seed=0)` — the literal hardcoded call, verified this
session against `run_deltanet_rd_exactness_sweep.py`'s own `build_cmd()`
(which never emits a `--heldout-frac` flag for this cell, so the launch
always takes the CLI default `0.5`, `run_deltanet_rd.py:1290`) — **NEVER
the launch seed, under any circumstance, for any of the ≤3 possible
launches (1940/1943/1944) this instrument can fire.** The build task
comment MUST carry an explicit warning against the trap (two different
`seed` values are in play at analysis time — the training launch's own
`seed`, threaded per-event for episode/event identity, and
`build_entity_pools`'s OWN internal, always-0 `seed` — and confusing the
two is a silent, total-verdict-corrupting bug, not a cosmetic one).
**Belt-and-suspenders cross-check (new Stage −1 item 11, §15.24.5):**
before Step 0b runs on ANY event, assert the offline-reconstructed
`pools.train_name_ids` is SET-EQUAL (as Python `set`s, order-independent)
to that SAME launch's own archived `anchor_train_ids` tensor
(`run_deltanet_rd.py:934–936`, already logged into every checkpoint
payload — the live run's own ground truth, since `model.
anchor_train_ids_buf` is registered directly from that launch's own
`pools.train_name_ids` at construction time, `run_deltanet_rd.py:1606` —
zero new telemetry cost, this tap already exists). Mismatch → **HARD-ABORT
the analysis before Step 0b or any other step runs, emitting no verdict**,
disclosing the reconstruction failure explicitly — this is a
**RECONSTRUCTION-FAILURE** state, distinct from Step 0b's own
INSTRUMENT-BUG (a defect in the LIVE probe path) and from Step −1's
NO-REPRO (a reproduction-count gap): a wrong OFFLINE reconstruction says
nothing about the live probe path's own health and must not be reported
as if it did.

> **Step 1's rule (Rev 3: UNCHANGED from Rev 2 — the two-level floor,
> now scoped to Step 1's own marker alone): ≥2 anomalous episodes (rows),
> occurring in ≥2 DISTINCT events.** Both conditions are required
> jointly. A single triggering batch, however many of its own rows
> disagree, is NEVER dispositive on Step 1's own marker alone (guards
> against one bad batch — the original M2 concern); ANY recurrence of a
> live/offline disagreement across two independently-sampled batches —
> even one row in each — IS dispositive on Step 1's own marker (a
> systematic, code-path-level defect recurs across independent draws; an
> ordinary tail event does not — this reasoning is sound for a NUMERICAL
> check and stays exactly as Rev 2 pinned it).

**Floors are counted PER-MARKER-TYPE, never combined (Rev 3, MINOR fix —
Rev 2's own prose already said "the SAME anomaly marker"; this revision
adds the negative test that actually exercises the claim, §15.24.5 Stage
−1 item 10).** A sink containing BOTH a 0b pool-mismatch episode in one
event AND a Step 1 live/offline-disagreement episode in a DIFFERENT event
never sums toward Step 1's own `≥2 episodes across ≥2 events` floor —
Step 1's count is over Step-1-marked episodes only, and 0b needs no count
at all under this revision's fix: a lone 0b episode is independently
dispositive regardless of what else the sink contains, including a Step
1 marker that has not itself cleared its own floor.

**0a's own corroborating marker (Rev 3, MINOR fix — the counting rule was
previously implicit).** Step 0a (table below) stays corroborating-only,
unchanged by Rev 2 or Rev 3, but its own recurrence disclosure uses the
SAME two-level floor as Step 1 (≥2 anomalous episodes across ≥2 distinct
events) to decide whether it is reported as "recurring corroboration"
versus a single excluded-and-disclosed flag — this floor governs ONLY how
0a's marker is described, never whether it is dispositive; 0a is never
dispositive on its own regardless of episode count, exactly as Rev 1's
own MINOR m2 demotion pinned.

**Exactly 1 anomalous episode on Step 1's marker (regardless of how many
total events exist) → flagged, EXCLUDED from that cell's analysis
population, and disclosed by its full `(seed, step, hop, batch_idx,
row_idx)` in the output JSON**; the verdict is computed on the remaining
episodes.

**Worked boundary examples (pinned here, exercised verbatim by §15.24.5
Stage −1 item 7; Rev 3 re-pins each to the marker it actually tests, and
adds E4/E5 below for the two states attack round 3 flagged by name):**
- **E1** — a 2-event sink, 2 pool-membership-mismatch episodes, ONE per
  event (a **0b** example): 2 anomalous episodes across 2 distinct events
  → **dispositive** (0b fires INSTRUMENT-BUG immediately, per
  MAJOR-2/MAJOR-3's own re-ordering below — this is the deterministic bug
  signature MAJOR-2 named, now correctly resolved instead of being
  refused by Step −1's reproduction bar). **Rev 3 note:** unaffected in
  OUTCOME — still dispositive — but reached via the simpler any-single-
  violation rule; the second event is corroborating, not required, under
  Rev 3's own fix.
- **E2** — a 4-event sink, 3 anomalous episodes all within ONE event (Rev
  3: pinned to **Step 1's marker** — under 0b's own any-single-violation
  rule this would already be dispositive on the FIRST of the three
  episodes, so this example no longer illustrates 0b): 3 anomalous
  episodes (≥2 ✓) but only 1 distinct event (NOT ≥2 ✗) → Step 1's floor
  **NOT met** → all 3 flagged, EXCLUDED, disclosed; verdict computed on
  the remainder; NOT dispositive on Step 1's marker alone.
- **E3** — a 3-event sink, 3 anomalous episodes, ONE per event (3
  distinct events; Rev 3: likewise pinned to **Step 1's marker**, for the
  same reason as E2): 3 anomalous episodes (≥2 ✓) across 3 distinct
  events (≥2 ✓) → Step 1's floor MET → **dispositive**.
- **E4 (Rev 3, FATAL fix's own worked counterexample — attack round 3's
  "state 3"):** a **5-event sink, EXACTLY 1 pool-membership-mismatch
  episode** (4 clean events + the 1 mismatch), a **0b** example. **Rev
  2's own joint floor:** 1 anomalous episode, 1 distinct event — NOT ≥2 on
  either count — floor NOT met → EXCLUDED, disclosed; verdict computed on
  the 4 remaining clean events, which clear Step −1's `≥3` gate and could
  land on REAL-CAPACITY-BOUNDARY or TOLERANCE-MISCALIBRATION →
  **silently wrong** (the FATAL this revision fixes). **Rev 3's split
  floor:** 0b fires **INSTRUMENT-BUG** immediately, dispositive, enforced
  abort — refuses REAL-CAPACITY-BOUNDARY/TOLERANCE-MISCALIBRATION
  outright, regardless of how many other events exist or how clean they
  are.
- **E5 (Rev 3, MINOR-1's own cross-marker example — attack round 3's
  "state 6"):** a ≥3-event sink containing exactly 1 pool-mismatch
  episode in event A **and**, in a DIFFERENT event B, exactly 1
  live/offline-disagreement episode — one 0b episode, one Step-1 episode,
  no overlap. A naive per-episode (not per-marker) count would read this
  as "2 anomalous episodes across 2 distinct events" and could wrongly
  treat it as clearing a SHARED floor for an ambiguous reason (impossible
  to tell, from that count alone, whether the right repair is a pool-
  bookkeeping fix or an NS-numerics investigation — two structurally
  different fixes). **Rev 3's adjudication:** markers are counted PER
  MARKER TYPE, never combined (the paragraph above) — 0b's own single
  violation (event A) is independently dispositive under its own new
  any-single-violation rule, full stop; Step 1's own single disagreement
  (event B) alone does not clear Step 1's own `≥2`/`≥2` bar and is
  separately excluded, disclosed as non-dispositive corroborating
  context. **Net verdict: INSTRUMENT-BUG**, attributed to 0b alone, with
  Step 1's own marker disclosed but never blended into the same count —
  exercised by the new negative test, §15.24.5 Stage −1 item 10.

**TF32 precision pinning (Rev 1 MAJOR M1; Rev 2 MINOR m3, per-source-run
refinement; Rev 3 MAJOR-B, batching pinned) — Step 1 only.** The offline
NS recompute now runs in BOTH modes on every dumped `k_eff_raw`/
`k_blend_raw`: **(a) strict-fp32** (`torch.backends.cuda.matmul.
allow_tf32 = False` explicitly set for the recompute process) and **(b)
matched-mode** (TF32 set to whatever the LIVE run itself recorded —
§15.24.2 item (ii)'s `tf32_matmul`/`tf32_cudnn` fields). **Matched-mode
is Step 1's primary agreement check** (the only apples-to-apples
comparison against what the live run actually computed); **strict-fp32
is corroborating context, reported alongside but not gating.** **(Rev 3,
MAJOR-B fix — batching pinned.)** Both modes recompute `newton_schulz_
orthogonalize` as ONE BATCHED call on the event's FULL dumped `(B,K,d)`
tensor — `newton_schulz_orthogonalize(k_eff_raw, n_iter=20)`, never a
`k_eff_raw[row_idx:row_idx+1]` singleton slice — because cuBLAS/cuDNN GEMM
kernel selection is itself batch-size-dependent: a batch-size-1 recompute
can silently route through a DIFFERENT kernel than the live batch-size-128
(`B=128`) call, and a near-boundary residual (exactly the population Step
1 exists to examine) can flip across the `0.01` threshold from that
kernel-selection difference alone, with zero real numerical disagreement
involved. Per-episode comparison is still exact: `resid_offline =
newton_schulz_orthogonalize(k_eff_raw, n_iter=20)[1]` (shape `(B,)`,
matching the live call's own batching precisely) is computed ONCE per
event, then indexed `resid_offline[row_idx]` for each episode's own
comparison — no information is lost relative to Rev 2's per-row spec, only
the confound from mismatched batch size at recompute time. **(Rev 2
addition, unaffected by MAJOR-B):** once Step −1's combined
sink can span UP TO 3 separate launches (the primary seed plus, on a
NO-REPRO, the two reserved contingency seeds, Step −1 below), matched-mode
recompute for EACH event MUST use THAT event's OWN source-run's recorded
`tf32_matmul`/`tf32_cudnn` — each launch is a separate process and, while
expected to agree, is never ASSUMED to. A per-launch `c17fallback_
step<N>.pt` file already records its own pair of flags (§15.24.2 item
(ii)); the analysis script indexes by each event's own source launch,
never by a single global assumed-uniform flag pair across the combined
sink. If the two OFFLINE modes disagree with EACH OTHER (independent of
whether either agrees with the live flag), this is reported as its own
named sub-finding, **TF32-SENSITIVE**, and routed to Step 2's
TOLERANCE-MISCALIBRATION examination rather than treated as an
INSTRUMENT-BUG signal — a matmul-precision-mode sensitivity is a
numerically real fact about the near-boundary raw-key population
(candidate (c)'s own territory), not evidence of a probe-path defect.

**Total precedence (Rev 2, MAJOR-3 fix; Rev 3, FATAL fix tightens what
"0b runs FIRST" actually means; Rev 4, MAJOR-1 fix prepends a
prerequisite gate ahead of everything else), pinned explicitly:
reconstruction cross-check (prerequisite, HARD-ABORT-only, no verdict) >
0b (structural) > Step −1 (reproduction) > Step 1 (agreement) > Step 2
(tolerance).** The reconstruction cross-check (above) is not itself a
verdict-producing step — it is a gate on whether ANY of the named
verdicts may be computed at all, run once per launch before that launch's
events are folded into the combined sink; its own failure mode
(RECONSTRUCTION-FAILURE) sits outside the totality claim below, exactly
as a missing kernel-safety artifact sits outside it. Every telemetry
state this instrument can produce, ONCE the reconstruction cross-check
has passed, maps to EXACTLY ONE primary verdict via the remaining order —
0b runs FIRST (among the verdict-producing steps), on whatever events
exist, and
**(Rev 3) is dispositive on its own marker's FIRST violation, not merely
"first in the ordering"**: Rev 2's own text already put 0b ahead of Step
−1, but its table row still gated 0b behind the SAME two-level floor as
Step 1, which meant a lone real violation in a sink with ≥2 total events
could still fall through to Step −1 and beyond (E4/"state 3" above) — the
ordering fix alone was necessary but not sufficient; this revision's own
FATAL fix (0b needs no floor) is what makes "0b runs first" actually mean
"0b, once triggered even once, ends the analysis," closing that gap.
Step −1's reproduction-count gate runs SECOND and ONLY governs the
verdicts that DO need statistical weight (0a's corroborating marker, Step
1, Step 2); Step 1 runs THIRD, only once Step −1 has cleared, using its
own unchanged two-level floor (above); Step 2 runs LAST, only if Step 1
agrees everywhere. **Totality claim, re-affirmed (Rev 3):** splitting the
floor only WIDENS the set of telemetry states 0b resolves on its own (any
single violation, not just ≥2-across-≥2) — it cannot create a new,
unresolved state, since every state 0b no longer waits on still falls
through to exactly the same Step −1/1/2 sequence Rev 2 already proved
exhaustive; the totality claim survives the split unchanged.

| Step | Check (exact) | Outcome |
|---|---|---|
| **0b. Pool-membership precheck (Rev 2: runs FIRST, ahead of Step −1's reproduction bar, MAJOR-2/MAJOR-3 fix; Rev 3: no floor of its own, FATAL fix; Rev 4: now gated by a prerequisite reconstruction cross-check, MAJOR-1 fix)** | Assert dumped `key_ids.shape == (B, 84)` (matches `cfg.K`) AND every id is an EXACT set-member of `pools.heldout_name_ids` (never the trained pool), evaluated per-episode across WHATEVER events exist so far (no minimum event count required — structural evidence needs no reproduction statistics); `pools` itself must have already PASSED the reconstruction cross-check above (offline-rebuilt with the hardcoded `seed=0`, verified SET-EQUAL to the checkpoint's own archived `anchor_train_ids`) before this row is even evaluated | **ANY SINGLE mismatched episode → INSTRUMENT-BUG, DISPOSITIVE, ENFORCED ABORT (Rev 3, FATAL fix — no event-count or episode-count minimum; supersedes Rev 2's floor-gated version of this row, §15.24.5's own abort-branch list)** — the analysis refuses REAL-CAPACITY-BOUNDARY and TOLERANCE-MISCALIBRATION and emits ONLY INSTRUMENT-BUG; every implicated episode named by its full `(seed,step,hop,batch_idx,row_idx)`. No mismatch at all → proceed to Step −1. There is no longer a "below the floor, excluded" case for this row. (If the prerequisite cross-check itself failed, this row never runs — see RECONSTRUCTION-FAILURE, above.) |
| **−1. Zero/low-event guard (Rev 1 FATAL F1; Rev 2: gates ONLY the REMAINING verdicts, since 0b already ran; Rev 3: combined-sink events now de-duplicated by launch-unique `seed`, MAJOR-A)** | `len(fallback_dump_sink) < 3` (the pinned minimum, counted over `seed`-qualified DISTINCT `(seed,step,hop,batch_idx)` events), checked AFTER 0b, BEFORE 0a/1/2 run | **`<3` events → refuse 0a/1/2 (REAL-CAPACITY-BOUNDARY / TOLERANCE-MISCALIBRATION); emit NO-REPRO** UNLESS 0b already fired above (0b's own INSTRUMENT-BUG verdict, once emitted on even a SINGLE violation, is never overridden or re-litigated by this gate). Pre-registered follow-up on NO-REPRO: fire seeds 1943/1944 (the reserved `KEYANCHOR_SCALING_CONTINGENCY_SEEDS_BY_D_K[96][84] = (1943, 1944)`, `run_deltanet_rd_exactness_sweep.py:3098` — Rev 3, MINOR fix, corrected from `:3097`, the dict-open line rather than the line the K=84 tuple sits on), re-run the identical cell + telemetry, and combine the resulting dump sink with the primary run's, EACH event carrying its own `seed` (Rev 3, MAJOR-A) — **re-running 0b on the newly-combined sink first** (Rev 2: 0b's precedence is unconditional, applies to every combined-sink state, not only the first pass). Combined total STILL `<3` DISTINCT `(seed,step,hop,batch_idx)` events (and 0b still hasn't fired) → **AMBIGUOUS-NONDETERMINISM**, promoting candidate (1) (§15.24.3's checkpoint-payload extension) to the primary next instrument. Combined total `≥3` (and 0b hasn't fired) → proceed to Step 0a/1 on the COMBINED sink. |
| **0a. Exact-rank precheck (Rev 1: demoted, MINOR m2 — unaffected by Rev 2/Rev 3)** | For EVERY dumped episode (row) at or past Step −1's gate: `torch.linalg.matrix_rank(k_eff_raw_episode, tol=1e-4)` (`k_eff_raw_episode` = one `(K,d)=(84,96)` row; tolerance on the pre-scaled `X_0 = A/√K` basin's own natural scale, `model_rd.py`'s `newton_schulz_orthogonalize` docstring, ~line 371–390) | `rank < K` for episodes at or above the SAME two-level floor as Step 1 (Rev 3, MINOR fix — counting rule now named explicitly, above) → **CORROBORATING evidence for INSTRUMENT-BUG only, never dispositive on its own, unaffected by 0b's own floor removal** — reported and disclosed alongside whatever Step 1 finds, never gating. Exactly 1 flagged episode below the floor → excluded, disclosed; verdict continues on the remainder. |
| **1. Live/offline NS agreement (Rev 1: dual TF32 mode; Rev 2: PER-EPISODE, not per-event-aggregate; Rev 3: batched offline recompute, MAJOR-B)** | For EVERY dumped episode (row) at or past Step −1's gate: recompute `newton_schulz_orthogonalize(k_eff_raw, n_iter=20)` OFFLINE as ONE BATCHED call on the event's FULL dumped `(B,K,d)` tensor (Rev 3, MAJOR-B fix — matching the live call's own `B=128` batching exactly; never a `k_eff_raw[row_idx:row_idx+1]` singleton slice, which can select a different GEMM kernel and flip a near-boundary residual from batching alone) in BOTH strict-fp32 and matched-mode (per-source-run, above); compare offline `resid_offline[row_idx] > 0.01` (matched-mode, primary, indexed from the batched recompute) to the LIVE `resid[row_idx] > 0.01` (§15.24.2 item (ii)'s own dumped `resid` field, per-episode — Rev 2: no longer collapsed through the aggregate `fallback_triggered` boolean, which the codebase's own docstring already flags as batch-level, not episode-level, `model_rd.py:433–434`) | Disagreement, at or above Step 1's OWN two-level floor (unchanged by Rev 3, now scoped to this marker alone) → **INSTRUMENT-BUG, DISPOSITIVE.** Exactly 1 disagreeing episode below the floor → flagged, excluded, disclosed; verdict continues on the remainder. **Agreement on every (remaining) episode → proceed to Step 2.** Any strict-fp32-vs-matched-mode flip, independent of the above → **TF32-SENSITIVE** (reported, routed to Step 2, never counted toward the INSTRUMENT-BUG floor). |
| **2. n_iter sweep (only reached if Step 1 agrees everywhere)** | Sweep `n_iter∈{24,28,32,40}` on the SAME dumped `k_eff_raw`, per episode | **Every episode's residual ≤0.01 by `n_iter≤32` → TOLERANCE-MISCALIBRATION.** **ANY episode still >0.01 at `n_iter=40` (the widest tested value, matching §15.23's own ceiling) → REAL-CAPACITY-BOUNDARY.** **Mixed across episodes (some resolve, some don't) → AMBIGUOUS-RESIDUAL** (Rev 3, MINOR fix, renamed from "residual AMBIGUOUS" for consistency with every other hyphenated verdict name; named follow-on below — distinct from Step −1's AMBIGUOUS-NONDETERMINISM; disambiguated by name throughout this section). |

**AMBIGUOUS-RESIDUAL, named follow-on (registered, not run; Rev 3, MINOR
fix — renamed from "Residual AMBIGUOUS"):** re-launch the identical cell
(same seed, same config) a second time and confirm the SAME
steps/episodes reproduce the SAME fallback pattern.
**Reproducible** → escalate to 2 more cells (K=78 and K=90, bracketing
K=84) to test whether the mixed signature is itself K/d-dependent.
**NOT reproducible under the same seed** → this is ITSELF a strong
INSTRUMENT-BUG signal (nondeterminism where the design assumes
determinism) and must be escalated as its own FATAL finding, not filed
away as merely AMBIGUOUS.

**Determinism cross-check (also doubles as a Stage −1 negative test,
§15.24.5; Rev 2 MINOR m4: now per-launch, not once):** the full
`state_dict` snapshot at `step=20000` (§15.24.2 item i) permits an
INDEPENDENT offline reconstruction of the same C17 batch — load the
snapshot into a fresh model instance, rebuild `eval_gen = seed + 10_000 +
20000` (`run_deltanet_rd.py:801`, the exact, already-registered formula,
with `seed` substituted per-launch — 1940 for the primary, 1943/1944 for
each contingency seed if fired), and replay the SAME pool-call order
(`m2, m3, c17, c19`, lines 802–808) so the shared generator advances
identically before C17's own batches are drawn. **Run this ONCE PER
LAUNCH (Rev 2 fix) — the primary run, and again for EACH contingency seed
if Step −1's NO-REPRO guard fires one — each time against THAT launch's
OWN live-dumped tensors, BEFORE that launch's events are folded into the
combined sink** (a launch whose determinism cross-check fails must not
silently contribute unverified events to a combined-sink verdict treated
as if every event had passed the same check; Rev 1's "run this ONCE"
implicitly assumed a single launch, which no longer holds once the
contingency-seed path can add up to 2 more). Assert BYTE-IDENTICAL
`key_ids`/`k_eff_raw` each time — this is the "determinism of the C17
batch under the pinned seed" gate item, and it is a REAL cross-check (two
independent derivation paths agreeing), not a vacuous shape check,
mirroring this program's own "two independent methods concur" discipline
(§15.22's `wall_s`-vs-timestamp cross-check).

### 15.24.5 Gates

**Reused, not re-run (cited by artifact, per §15.24.2):**
- Kernel long-T safety at `T=588` — already PASS, §15.20.1/§15.20.6 Stage
  −1 item 1.
- `GATE2_N_ITER_BY_D_K[96][84]=20` sufficiency — already PASS, §15.20.6
  Stage −1 item 2.

**New Stage −1 items, BLOCKING, this instrument's own telemetry hooks
(none of these existed before this design; all must run to completion,
not merely be written, per this project's own "assertion has teeth"
rule):**

1. **Bitwise-identical-trajectory smoke** (§15.24.2's own required
   negative test): `--c17-repro-telemetry` ON vs OFF, fixed short seed,
   identical loss curve.
2. **Dump-fires-exactly-at-fallback smoke:** two synthetic fixtures — (a)
   a hand-built near-duplicate K-row batch that FORCES
   `fallback_triggered=True` (e.g. two rows within one episode set to
   near-identical unit vectors), assert the dump sink receives **exactly
   1** entry; (b) a well-conditioned/orthonormal-ish K-row batch, assert
   the dump sink receives **exactly 0** entries. Proves the hook fires
   precisely on the flag, neither more nor less often.
3. **Tensor-shape assertions:** `k_eff_raw`/`k_blend_raw` dumps
   `== (B_batch, 84, 96)`, `resid` dump `== (B_batch,)`, all on CPU, fp32,
   detached (`requires_grad=False`).
4. **Determinism cross-check, per-launch (Rev 2 MINOR m4)** (§15.24.4's
   own "Determinism cross-check" item): the offline state_dict-replay
   reconstruction byte-matches the live dump, run ONCE PER LAUNCH — the
   primary run, and again for each contingency seed if Step −1 fires one.
   This is the ONE Stage −1 item that can only run AFTER a given launch
   has produced both its own state_dict snapshot and at least one live
   dump — sequenced as the LAST Stage −1 item for THAT launch, immediately
   post-launch and BEFORE that launch's events are folded into the
   combined sink, before the decision-rule analysis (§15.24.4) is trusted
   on the combined result.
5. **`sha256`/config cross-check** (§15.24.2's own build step): the
   reconstructed `build_cmd()` output's architecture-relevant fields
   match the archived K84/seed=1940 JSON's own `exactness_config` block,
   field by field, by hand, recorded in the build commit message.
6. **Step −1 NO-REPRO-guard negative test (NEW, Rev 1, attack-round-1
   FATAL F1 fix — "assertion has teeth"):** feed the analysis script a
   synthetic EMPTY `fallback_dump_sink` (`[]`, the zero-event case F1
   named explicitly — the plausible-but-untested scenario a clean seed-
   fixed re-run with GPU run-to-run floating-point nondeterminism could
   produce, per `KEY_ANCHORING_DESIGN.md:1976–1994`'s own already-measured
   precedent, and per the confirmed absence of any
   `torch.use_deterministic_algorithms`/`cudnn.deterministic` pin anywhere
   in `run_deltanet_rd.py`/`model_rd.py`/`key_anchoring.py`/
   `run_deltanet_rd_exactness_sweep.py`, verified this session), and
   separately a 2-event sink (below the pinned `<3` minimum): assert BOTH
   cases emit **NO-REPRO** and refuse to emit any of REAL-CAPACITY-
   BOUNDARY / INSTRUMENT-BUG / TOLERANCE-MISCALIBRATION. This is the
   negative test that proves Step −1's guard actually blocks the
   universally-quantified `all()`-over-empty-list vacuous-pass failure
   mode F1 found, rather than merely being described in prose.
7. **Step 1's two-level floor boundary test (Rev 1 MAJOR M2; Rev 2 FATAL
   fix — rewritten for the row/event two-level floor, replacing the
   dropped percentage-clause fixture; Rev 3 — scope narrowed to STEP 1's
   marker only, since 0b no longer has a floor of its own, FATAL fix):**
   four synthetic fixtures at the exact boundary, exercising the
   E1/E2/E3 edge states pinned verbatim at §15.24.4 — (a) exactly 1
   anomalous episode (rank-deficient / live-offline-disagreeing, Step
   0a/1's own markers ONLY — Rev 3: a pool-mismatch fixture belongs in
   item 8 below, not here, since 0b has no floor to test) among ≥3 total
   events: assert that episode is flagged, EXCLUDED, and named by its
   full `(seed,step,hop,batch_idx,row_idx)` in the output JSON, verdict
   computed on the remainder; (b) **E2**: a 4-event sink, 3 anomalous
   episodes all within ONE event: assert Step 1's floor is NOT met (only
   1 distinct event), all 3 flagged/excluded, NOT dispositive; (c)
   **E3**: a 3-event sink, 3 anomalous episodes, ONE per event: assert
   Step 1's floor IS met (≥2 episodes across ≥2 distinct events),
   dispositive; (d) exactly 2 anomalous episodes in 2 distinct events
   (the general 2-and-2 case, distinct from E1/E3's specific counts):
   assert Step 1's floor is met. Proves Step 1's own two-level absolute
   floor has real teeth at both ends (single-event concentration vs.
   cross-event recurrence), not just the illustrative worst-case-15,360-
   episode arithmetic quoted in prose.
8. **0b-precedes-Step−1 enforced-abort negative test (NEW, Rev 2,
   MAJOR-3 fix — same "assertion has teeth" discipline, this revision's
   own precedence-order fix; Rev 3 adds the single-violation sub-case
   below, FATAL fix):** **E1** exactly, as its own dedicated fixture — a
   synthetic 2-EVENT sink (below Step −1's `<3` minimum) with 2
   pool-membership-mismatch episodes, one per event: assert the analysis
   emits **INSTRUMENT-BUG** (not NO-REPRO — 0b runs before, and
   independent of, Step −1's reproduction-count gate) and refuses
   REAL-CAPACITY-BOUNDARY/TOLERANCE-MISCALIBRATION. This is the negative
   test that proves the FIXED precedence order (0b > Step −1 > Step 1 >
   Step 2, §15.24.4) actually routes a genuine 2-event structural-bug
   signature to INSTRUMENT-BUG instead of the AMBIGUOUS-NONDETERMINISM/
   NO-REPRO path MAJOR-2 found it falling into under Rev 1's ordering.
   **(Rev 3, FATAL fix — added sub-case, E4/"state 3" above):** a SECOND,
   dedicated fixture — a synthetic **5-event sink with EXACTLY ONE**
   pool-membership-mismatch episode (4 clean events + the 1 mismatch,
   well ABOVE Step −1's `<3` minimum): assert the analysis emits
   **INSTRUMENT-BUG** (not REAL-CAPACITY-BOUNDARY/TOLERANCE-
   MISCALIBRATION computed on the 4 clean remaining events) — this is the
   negative test that proves 0b's own any-single-violation rule actually
   has teeth at exactly the count (1 episode, 1 event) Rev 2's own floor
   used to exclude, the FATAL bug's own worked counterexample.
9. **Batched-vs-singleton offline recompute comparison (NEW, Rev 3,
   MAJOR-B fix — "assertion has teeth"):** on synthetic near-boundary
   data (a hand-built `(B,K,d)` tensor with `B=128`, `K=84`, `d=96`, one
   row engineered to sit within numerical noise of the `resid=0.01`
   threshold), run `newton_schulz_orthogonalize` twice: once as ONE
   BATCHED call on the FULL `(128,84,96)` tensor (Step 1's own pinned
   method, above) and once as 128 separate `(1,84,96)` singleton calls
   (Rev 2's own now-superseded method); assert the batched call's
   `resid_offline[row_idx]` is used for the decision (not the singleton
   result) and DISCLOSE, in the output JSON, any row where the two
   methods' `resid` values differ — proving the fix actually changes what
   gets compared, and giving a concrete, on-the-record measurement of how
   large the batch-size confound was on this run's own near-boundary
   population, rather than leaving MAJOR-B's own risk merely asserted.
10. **Cross-marker negative test (NEW, Rev 3, MINOR-1 fix — E5/"state 6"
    above):** a synthetic ≥3-event sink containing exactly 1
    pool-membership-mismatch episode in event A and, in a DIFFERENT
    event B, exactly 1 live/offline-disagreement episode (no overlap
    between the two marker types): assert the analysis emits
    **INSTRUMENT-BUG** (via 0b's own single-violation rule, event A
    alone) and that Step 1's own marker (event B) is reported as
    EXCLUDED/disclosed, non-dispositive — never summed with 0b's episode
    to reach a shared 2-episode/2-event count under either marker's own
    floor. Proves the "same anomaly marker" language in §15.24.4's floor
    paragraph is enforced by the analysis script, not merely asserted in
    prose.

    **[Rev-4-build note (FIX agent, C17 build audit, this commit) — disclosed
    deviation 1, ties to this item's own build-time disclosure in
    `diag_c17_repro_stage_minus1.py`'s item 10c]:** `run_full_precedence()`
    short-circuits at Step 0b's dispositive return — when 0b fires, Step 1's
    own diagnostic is never ALSO computed and attached to that SAME result
    dict. Item 10's own per-marker-type counting completeness is therefore
    proven IN ISOLATION (`step_0b_pool_membership`/`_apply_two_level_floor`
    called directly on the same synthetic sink, items 10a/10b) rather than
    exhibited together in one run's combined verdict JSON (item 10c).
    Confirmed harmless (0b's own any-single-violation rule is unconditionally
    dispositive regardless of what Step 1 would have found on the same
    sink), but flagged here per the build audit's own request so a future
    revision does not mistake item 10c's own single-source verdict for
    having surfaced both markers in one output.
11. **Pool-reconstruction cross-check, the two-seeds-trap fixture (NEW,
    Rev 4, MAJOR-1 fix — "assertion has teeth"):** BEFORE Step 0b runs on
    any event (§15.24.4's new prerequisite gate), reconstruct
    `pools.train_name_ids` offline via `grd.build_entity_pools(tokenizer,
    heldout_frac=0.5, seed=0)` — the pinned, hardcoded call — and assert
    it is SET-EQUAL to a real archived `anchor_train_ids` tensor pulled
    from an actual K84/seed=1940 checkpoint. **Positive fixture:** the
    correctly-seeded (`seed=0`) reconstruction matches; assert the
    cross-check PASSES and Step 0b is allowed to run. **Negative
    fixture, the trap itself:** reconstruct with `seed=1940` (the LAUNCH
    seed, the exact mistake MAJOR-1 found) against the SAME real
    `anchor_train_ids` tensor; assert the cross-check FAILS and the
    analysis HARD-ABORTS before Step 0b (or any other step) runs, with no
    verdict emitted — a different `random.Random(seed).shuffle(...)` draw
    produces a materially different partition with overwhelming
    probability on a 213-name pool, so this fixture is not a coin-flip
    pass. This is the negative test that proves the reconstruction gate
    actually catches the two-seeds trap, not merely disclaims it in
    prose.
12. **Single-event 0b minimal-boundary fixture (NEW, Rev 4, MINOR-1
    fix):** a synthetic sink with `len(fallback_dump_sink) == 1` — the
    single most degenerate sink size possible, below even Step −1's own
    `<3`-event reproduction minimum — containing exactly 1
    pool-membership-mismatch episode in that one event: assert the
    analysis emits **INSTRUMENT-BUG**, checked and confirmed BEFORE
    Step −1's own `<3`-event gate logic is even reached. Proves 0b's own
    prose ("a sink of any size — including a single-event sink, before
    Step −1's own `<3`-event reproduction gate has even run" — §15.24.4)
    has teeth at the single most extreme case named, which neither E1
    (2 events, Stage −1 item 8) nor E4 (5 events, Stage −1 item 8's own
    sub-case) actually reaches.

**Budget guard (mirrors §15.20.6's own Stage 0 abort-trigger formula,
`1.5 × <point-estimate GPU-h> × 3600`):** realized `wall_s ≥ 1.5 × 0.450 ×
3600 = 2430s` → halt, diagnose (`nvidia-smi` contention check on GPU 2
first — Phase-2 owns GPUs 0–1, GPUs 3–7 idle but shared-cluster
contention is always re-checked at launch time, not assumed from this
design's own drafting time, per §15.13/§15.20.6's own re-verification
discipline), re-price before retrying.

**PI sign-off token.** This repro draws against the ALREADY-authorized
KEY_ANCHORING_SCALING sub-ledger (§15.24.7 below shows it fits comfortably
inside even the ORIGINAL, non-extended 21 GPU-h ceiling) — **no new
ceiling ask**, unlike §15.20.5's own `+5.0 GPU-h` extension. Per this
program's own MAJOR-4 precedent generalized (a NEW PI-decision earns its
own named record whether or not it also asks for new ceiling — the
original fix was about preventing a STALE token from silently authorizing
an unrelated ask), this design registers **`KEYANCHOR_SCALING_
C17REPRO_PI_SIGNOFF=1`**, checked belt-and-suspenders (bash launcher +
Python-side `keyanchor_scaling_stage_gate`-equivalent) **IN ADDITION TO**
the base `KEYANCHOR_SCALING_PI_SIGNOFF=1` (still required, since this is
still spend against that same sub-ledger) — never OR'd, both required.
**Flagged explicitly as a judgment call for the attack round:** since no
ceiling increase is requested, an attack round could reasonably argue the
base token alone suffices and the new token is unneeded ceremony; this
design takes the more conservative reading (a genuinely new scientific
instrument, not a re-run of the wide-grid wave, earns its own explicit
go-ahead) but registers the disagreement rather than silently deciding it.

**Enforced abort branches (registered here, built later, each needs its
own negative test run to completion before Stage 1 launch):**
- Missing/`False` kernel or Gate-2 cited artifact → refuse to launch
  (same belt-and-suspenders shape as every prior `keyanchor-scaling*`
  wave). Negative test: point the gate at a deliberately-absent artifact
  path, assert `exit 1`/`sys.exit(1)`.
- **Pool-reconstruction mismatch (NEW, Rev 4, MAJOR-1 fix — the
  prerequisite branch, ahead of EVEN 0b, §15.24.4's "Total precedence"
  paragraph)** → the analysis script must HARD-ABORT before Step 0b or
  any other step runs whenever the offline-reconstructed
  `pools.train_name_ids` (built with the hardcoded `seed=0`, NEVER the
  launch seed — §15.24.4's "two-seeds trap" paragraph) fails to
  SET-EQUAL that launch's own archived `anchor_train_ids` tensor
  (`run_deltanet_rd.py:934–936`) — emitting NO verdict at all
  (RECONSTRUCTION-FAILURE, not INSTRUMENT-BUG), and disclosing the
  reconstruction failure explicitly. Negative test: Stage −1 item 11
  above (the `seed=1940`-vs-`seed=0` two-seeds-trap fixture).
- **Pool-membership structural mismatch (NEW, Rev 2 MAJOR-3 fix,
  §15.24.4 Step 0b — the FIRST-checked VERDICT-PRODUCING abort branch,
  ahead of every other verdict but now itself gated by the
  reconstruction-mismatch branch above, Rev 4; Rev 3, FATAL fix — floor
  REMOVED, not merely relaxed)** → the analysis script must refuse to
  emit REAL-CAPACITY-BOUNDARY or TOLERANCE-MISCALIBRATION whenever 0b
  flags ANY SINGLE pool-membership violation — no event-count or
  episode-count minimum — and instead emit ONLY INSTRUMENT-BUG, checked
  BEFORE the zero/low-event guard below (0b needs no reproduction-count
  minimum at all, on either axis — structural evidence, not statistics).
  Negative test: Stage −1 item 8 above (the E1 fixture AND its Rev 3
  single-violation sub-case, E4/"state 3"), and its Rev 4 minimal-boundary
  sub-case, Stage −1 item 12 (a single-event, single-mismatch sink — the
  most degenerate case 0b's own prose names).
- **Zero/low-event guard (Rev 1 FATAL F1; Rev 2: now the SECOND-checked
  branch, only reached if 0b did not fire, §15.24.4 Step −1)** → the
  analysis script must refuse to emit ANY of 0a/Step 1/Step 2's named
  verdicts when `len(fallback_dump_sink) < 3` and instead emit NO-REPRO,
  triggering the seeds-1943/1944 contingency re-run. Negative test:
  Stage −1 item 6 above.
- Live/offline NS-agreement mismatch at analysis time (§15.24.4 Step 1,
  **Rev 2: computed PER-EPISODE using the dumped per-row `resid`, gated
  by Step 1's OWN two-level floor (≥2 anomalous episodes across ≥2
  distinct events, unchanged by Rev 3), on the TF32 matched-mode
  recompute pinned per-source-run; Rev 3: that recompute now runs as ONE
  BATCHED call on each event's full dumped tensor, MAJOR-B, never a
  batch-size-1 slice**) → the analysis script must refuse to emit a
  REAL-CAPACITY-BOUNDARY or TOLERANCE-MISCALIBRATION verdict once the
  floor is met and instead print INSTRUMENT-BUG with every disagreeing
  episode's own full `(seed,step,hop,batch_idx,row_idx)`. Negative test:
  feed the analysis script a synthetic fixture where matched-mode live and
  offline DELIBERATELY disagree on ≥2 episodes across ≥2 distinct events
  (a hand-edited dump with flipped flags), assert it reports
  INSTRUMENT-BUG and not one of the other two verdicts; a SECOND fixture
  with exactly 1 disagreeing episode (or 2 disagreeing episodes
  concentrated in ONE event, E2's shape) asserts EXCLUSION-and-continue
  instead (Stage −1 item 7); a THIRD check (Stage −1 item 9) confirms the
  batched-recompute path is what actually feeds this comparison, not a
  singleton slice.
- Stale-token check: launching with ONLY the base `KEYANCHOR_SCALING_
  PI_SIGNOFF` set (new token absent) must refuse, exactly mirroring
  §15.20.5's own MAJOR-4 fix pattern.

### 15.24.6 What this changes for the paper

**If (a) REAL-CAPACITY-BOUNDARY:** the d=96 "flat at ceiling" reading
(§15.19's own flat-near-1.0 curve across the ORIGINAL K/d≤0.71875 window)
becomes **"the C17-measurable ceiling ends at K/d≈0.75"** — a REAL,
positive structural finding about where the super-linear capacity result's
own held-out-entity generalization stops holding, arguably **more
publishable** than a flat, uninformative curve: it names an actual
boundary rather than reporting "no cliff found in the window we happened
to test." This would also, finally, retroactively validate §15.20's own
entire wide-grid rationale (bracket the two rival bands with real points)
— just via a DIFFERENT instrument (a probe-path boundary) than the one
originally planned (a fitted `x0(96)`).

**If (b) INSTRUMENT-BUG:** the wide-grid wave's own 11/12 inadmissible
cells are NOT evidence of anything about the model — they are a probe
artifact. The correct next step is fixing the identified bug (pool
construction / K-mismatch / whatever the sub-diagnosis names) and
**re-running the wide-grid wave at the fixed instrument**, at which point
§15.20.4's own original rival-discrimination test (never yet executed,
§15.22's own headline gap) becomes attemptable again for the first time.

**If (c) TOLERANCE-MISCALIBRATION:** a cheap, surgical fix (bump
`geo3_resid_tol` for raw/un-blended queries specifically, or raise
`n_iter` modestly) unlocks the ALREADY-COLLECTED 11 inadmissible cells'
own `h4` values for the fit **without any new GPU spend** — the fastest
path back to attempting §15.20.4's discrimination test, since the raw
data already exists and only the ADMISSION gate, not the training, would
need re-scoring.

**Either way, this section's own existence is itself evidence the
program's standing discipline is working as designed:** a misdiagnosed
mechanistic claim (§15.22) was caught and retracted (§15.23) before it
reached a paper, and this instrument is the pre-registered, mechanically-
gated next step rather than a second guess.

### 15.24.7 GPU plan and ledger

**GPU assignment:** 1 GPU, **GPU 2** (Phase-2 owns GPUs 0–1; GPUs 3–7
remain idle and unaffected by this launch).

**Cost arithmetic, re-derived from §15.22's own realized per-cell rates
(shown, not asserted):**

- K=84's own 3-seed realized `wall_s` (§15.22's per-cell table):
  1419.82 + 1378.79 + 1429.39 = 4228.00s, mean **1409.333s = 0.391481
  GPU-h** — the closest calibrated analog (same K, same d_state, real
  realized data), preferred over the 12-cell d=96-wide mean (0.4025 GPU-h)
  per §15.20.5's own "K-SPECIFIC realized rates are more precise than a
  generic point estimate" precedent.
- **Telemetry-overhead disclosure:** the new hooks add (i) one full
  `state_dict` `torch.save` at step 20000 only (low-MB, one-time,
  negligible next to a 1409s run), (ii) up to ~120 event-triggered
  `.detach().cpu()` + `torch.save` calls worst-case (4 batches × 3
  H_train hops × ≤10 checkpoints), **each event dumping TWO ≈4.1MB
  `(≤128,84,96)` fp32 tensors, `k_eff_raw` AND `k_blend_raw`
  (§15.24.2 item (ii)'s own dict literal dumps both, deliberately, for
  the free bitwise-equality re-confirmation) — corrected disk arithmetic
  (Rev 1, attack-round-1 MINOR m1 fix): 4.1MB × 2 tensors × ≤120 events
  ≈ 984MB ≈ ~1GB worst-case disk**, no added GPU compute, (iii) cheap
  CPU-side resid aggregation reusing already-computed tensors. None of
  these add GPU kernels; each `.cpu()` call DOES force a device-sync
  stall (sub-millisecond to low-millisecond on an uncontended GPU) that
  the ORIGINAL cell's own realized rate never paid. **Disclosed,
  conservative +15% overhead buffer** (first-time code path, never
  before measured): `0.391481 × 1.15 = 0.450 GPU-h` — this is the design's
  own **1× point estimate**, landing exactly in the pre-registered
  ~0.4–0.5 GPU-h range. **The 15% figure itself is kept as-is (Rev 1,
  attack-round-1 MINOR m2 fix, closing §15.24.9 Q5 by disclosure)** — the
  attack round's own Q5 offered two remedies (accept the disclosed
  uncertainty since the 2× ceiling already absorbs a far larger miss than
  15%, OR require a cheaper CPU-only dry-run timing a single dump event);
  this revision takes the first: the 2× pessimistic bracket already
  absorbs a 100% miss on the 1× estimate, which comfortably covers a
  plausible error in a 15% first-time-code-path guess, so a dry-run
  timing pass is not required before launch.
- **2× pessimistic ceiling** (house discipline, §15.20.5's own "the
  realized-rate expectation is NOT the ceiling number," reinforced by this
  program's own realized/estimate ratio history of 13.6%–112.5%):
  `0.450 × 2 = 0.900 GPU-h`.
- **NO-REPRO contingency cost (Rev 1, attack-round-1 FATAL F1 fix,
  §15.24.4 Step −1):** if the primary cell's `fallback_dump_sink` comes
  back below the pinned `<3`-event minimum, the pre-registered follow-up
  fires the reserved K=84 contingency seeds 1943/1944
  (`KEYANCHOR_SCALING_CONTINGENCY_SEEDS_BY_D_K[96][84]`,
  `run_deltanet_rd_exactness_sweep.py:3098` — Rev 3, MINOR fix, corrected
  from `:3097`) — **2 additional cells at the
  SAME per-cell 1× point estimate, `2 × 0.450 = 0.900 GPU-h`**, priced
  against the 2× bracket below (not re-multiplied by a further 2×; the
  contingency cells are the SAME K/d/architecture as the primary cell, so
  the already-derived 1× rate applies to each).

**Sub-ledger arithmetic (Rev 4, MAJOR-2 fix — baseline corrected).**
KEY_ANCHORING_SCALING's realized baseline was previously stated as
**18.1196/26 GPU-h** (11.7865 §15.19 + 6.3331 §15.22; §15.23's diagnostic
cost ~0 GPU-h, already folded in) — this OMITTED the K=69/seed=1733
contingency cell's own realized **0.427 GPU-h** (`wall_s=1535.2s`, §15.22
addendum, landed 2026-07-08: a real cell, run standalone via
`run_k69_s1733_contingency.py` AFTER the 16-cell wide-grid harvest the
6.3331 figure sums, and never folded back into any statement of the
running total anywhere it is quoted). **Corrected baseline: 11.7865 +
6.3331 + 0.427 = 18.5466/26 GPU-h realized.** Adding this design's own:

| Bracket | This design | Running total | vs. ORIGINAL 21 | vs. extended 26 |
|---|---|---|---|---|
| 1× (0.450) | +0.450 | 18.9966/26 | 18.9966/21 = 90.46% | 73.06%, reserve 7.0034 |
| 2× (0.900) | +0.900 | 19.4466/26 | 19.4466/21 = **92.60%, still inside the ORIGINAL, non-extended ceiling** | 74.79%, reserve 6.5534 |
| 2× + F1 NO-REPRO contingency (2 seeds, +0.900) | +0.900 | **20.3466/26** | 20.3466/21 = **96.89%, still fits inside the ORIGINAL, non-extended ceiling — margin now 0.6534 GPU-h (Rev 4, MAJOR-2 fix — was 1.0804 before the baseline correction, a ~40% tightening of the reserve)** | 78.26%, reserve 5.6534 |

**Notable: even at the 2× pessimistic bracket WITH the F1 NO-REPRO
contingency fully fired (20.3466 GPU-h worst-worst-case — Rev 4,
MAJOR-2 fix, corrected from the previously-stated 19.9196 after folding in
the K=69/seed=1733 cell's own realized 0.427 GPU-h), this design's own
cost stays under the ORIGINAL 21 GPU-h ceiling** — **0.6534 GPU-h** of
margin remains untouched (down from the previously-claimed 1.0804 GPU-h,
disclosed plainly: the conclusion this design fits inside the ORIGINAL
ceiling without the extension survives unchanged, but the margin that
backs it is ~40% tighter than previously stated), and the `+5.0 GPU-h`
extension (already authorized, `KEYANCHOR_SCALING_EXT_PI_SIGNOFF`, never
yet drawn on per §15.22) is not needed at all for this launch, even in the
worst case Rev 4 now prices. This still fits.

### 15.24.8 Standing constraints (restated, apply unchanged)

- Exact thresholds, no numerical-tolerance slack (§15.24.4's `rank<K`,
  `resid>0.01`, live/offline exact-boolean-agreement checks).
- Negative unit tests run to completion, not merely written (§15.24.5's
  ten Stage −1 items, as of Rev 3).
- Smoke test every model incl. eval batch before launch — this repro
  reuses an already-smoke-tested architecture/config; the ONLY new smoke
  surface is the telemetry hooks themselves (§15.24.5 items 1–3).
- tmux + supervisor launch pattern for the single-cell launch, same
  discipline as every prior wave, scaled down to one cell.
- Archive to repo (≤25MB) + SSD mirror, both, no exceptions — registered
  for the harvest, not built here (design-only session).
- Multiple independent adversarial audit rounds — this section is
  explicitly DRAFT pending its own attack round(s) before
  CLEARED-FOR-BUILD.
- Never compress matrices to vectors; use MultiProbeHead-style bilinear
  readouts — N/A to this instrument (no new readout head is built; this
  is a diagnostic tap on existing tensors only).

### 15.24.9 ATTACK-ROUND QUESTIONS — ROUND 0 (self-attack, minimum 5) —
**Rev 1 status: adjudicated below, question by question, against the
independent attack round's findings (§15.24.10).**

**Rev 2 status:** attack round 2 (§15.24.11) did not reopen any of Q1–Q5
below — its own findings (1 FATAL, 3 MAJOR, 4 MINOR) are new, not
re-litigations of this self-attack round's own questions. Q1 (dtype) and
Q4 (single-cell sufficiency) remain open exactly as Rev 1 left them, now
two rounds unaddressed; round 3, next, should pick at least one up rather
than deferring indefinitely a third time.

**Rev 3 status:** attack round 3 (§15.24.12) likewise did not reopen any
of Q1–Q5 below — its own findings (1 FATAL, 2 MAJOR, 6 MINOR) are new,
not re-litigations. **Q1 is now adjacent to, but still distinct from,
MAJOR-B's own fix** (Q1 asks about a DTYPE mismatch — fp32 offline vs. a
possible silent bf16 upstream cast; MAJOR-B fixed a BATCH-SIZE-dependent
GEMM-kernel-selection mismatch — fp32-vs-fp32, batch-size-1 vs.
batch-size-128); MAJOR-B does not answer Q1, and Q1's own dtype question
remains genuinely open, now three rounds unaddressed. Q4 (single-cell
sufficiency) also remains open, unaddressed by round 3's own scope (the
SEED-launch-uniqueness fix, MAJOR-A, closes a bookkeeping gap WITHIN the
K=84 combined sink, not the cross-K generalization question Q4 asks).
Round 4, next, should pick up at least one of Q1/Q4 rather than deferring
a fourth time.

**Rev 4 status:** attack round 4 (§15.24.13) likewise did not reopen any
of Q1–Q5 below — its own findings (0 FATAL, 2 MAJOR, 1 MINOR) are new: a
pool-reconstruction/two-seeds-trap gap (MAJOR-1), a ledger baseline gap
(MAJOR-2), and a minimal-boundary negative test (MINOR-1), none a
re-litigation of this self-attack round's own questions. Q1 (dtype) and Q4
(single-cell sufficiency) remain open exactly as Rev 3 left them, now four
rounds unaddressed. **Round 5 is a VERIFY pass confirming Rev 4's own
three narrow fixes land clean, not a fresh full attack round** — whichever
full attack round follows it should pick up at least one of Q1/Q4 rather
than deferring a fifth time.

**Q1. Does the offline NS recompute (Step 1) actually use the SAME
numerics as the live path, or does an fp32-offline-vs-bf16-adjacent-live
dtype difference risk a false INSTRUMENT-BUG verdict?** Genuinely open.
`bind()`'s kernel call casts to bf16 for the `chunk_delta_rule` call
(`model_rd.py:362–363`, `k_bf`/`v_bf`), but the geo3/NS path operates on
`k_norm_raw`/`k_eff_raw` computed via `F.normalize(k_conv, ...)` in
whatever dtype `k_conv` itself is (fp32 unless explicitly cast) — NOT
inside the bf16 kernel call. **TODO for the attack round:** confirm
`k_eff_raw`'s dtype at dump time is fp32 (matching `newton_schulz_
orthogonalize`'s own assumed dtype), not silently bf16-downcast anywhere
upstream; if it IS bf16, the offline fp32 recompute is not a clean
apples-to-apples comparison and Step 1's disagreement branch could fire
spuriously. **Rev 1 status: NOT in the independent round's scope
(MAJOR M1 fixes the adjacent-but-distinct TF32-precision-MODE axis, not
this dtype-CAST question) — but independently VERIFIED this session
while drafting Rev 1, so the dtype fear itself does not hold:**
`_kv_over_sequence` (`model_rd.py:995–1007`, the function that produces
`k_conv` feeding `k_norm_raw = F.normalize(k_conv, dim=-1)` at the geo3
call site, `model_rd.py:1109`) runs `self.embed` → `self.k_proj` →
`self.k_conv1d` → `self._zca_apply` with no `.to(torch.bfloat16)`/
`.half()`/`.bfloat16()` cast anywhere in that chain; a whole-file grep of
`run_deltanet_rd.py`/`model_rd.py` for `autocast`/`.half()`/
`.to(torch.bfloat16)`/`.bfloat16()` finds exactly one autocast context
(`model_rd.py:159`, explicitly `enabled=False`) and the three bf16 casts
already named (`model_rd.py:362–364`) — all three feed ONLY the SEPARATE
`chunk_delta_rule` kernel call, never `k_eff_raw`. `k_eff_raw` is fp32 at
dump time. This remains a narrower, still-open question from the TF32 axis
M1 fixes and is disclosed as such, not silently folded into that fix.

**Q2. Is the exact-rank precheck's `tol=1e-4` threshold (Step 0a)
actually well-calibrated, or was it picked by analogy without a real
derivation?** Disclosed weakness: `tol=1e-4` was chosen "on the same
natural scale as the `X_0=A/√K` pre-scaling basin" but this is a
qualitative justification, not a derived number the way §15.20's own
K_crit arithmetic was derived. **TODO for the attack round:** either
derive a principled `tol` from the NS convergence basin's own math, or
demote this check from "dispositive" to "strong evidence requiring
corroboration," and re-verify it against a KNOWN-good (M2 in-distribution)
batch first to confirm it doesn't fire spuriously on ordinary,
non-degenerate rows. **RESOLVED at Rev 1 (attack-round-1 MINOR m2 fix):**
the second remedy taken — Step 0a demoted from dispositive to
corroborating (§15.24.4), never independently sufficient for
INSTRUMENT-BUG. The "re-verify against a KNOWN-good M2 batch" half of
Q2's TODO is not yet run (no principled `tol` derivation either) — the
demotion closes the FALSE-DISPOSITIVE risk Q2 raised without requiring
either follow-up, but neither is claimed as done.

**Q3. Does dumping `k_blend_raw` alongside `k_eff_raw` for C17 batches
risk masking a REAL anchor-table interaction that §15.23's own "100%
architecturally bypassed" claim missed?** §15.23's claim rests on
`trained_here=False` for every C17 item, verified via
`anchor_trained_mask`'s own construction — this design re-confirms it
live (bitwise-equality assertion) rather than assuming it, which is the
right posture, but the attack round should confirm the bitwise-equality
assertion is actually WIRED IN as a build-time check (§15.24.2 item ii),
not merely described in prose here. **RESOLVED at Rev 1 — CONFIRMED
CLEAN by the independent round:** the attack round independently
re-verified the anchor-bypass wiring (one of its named clean items,
§15.24's Rev 1 status note) — the bitwise-equality assertion between
`geo3_last_k_eff_raw`/`anchor_last_k_blend_raw` really is a build-time
check tied to §15.24.2 item (ii)'s dict construction, not merely prose.
No fix needed; §15.24.2's text is unchanged by this revision.

**Q4. Is ONE cell (K=84/seed=1940) actually sufficient to reach ANY of
the three named verdicts with confidence, or does a single-seed,
single-K repro risk exactly the same "n=1, no statistical weight" problem
§15.22 flagged for its own K=72/seed=1741 point?** Partially mitigated:
unlike §15.22's h4-fit problem (a continuous statistical estimate needing
multiple seeds), this design's verdicts (0a/0b/1/2) are STRUCTURAL/
mechanical facts about a specific tensor (rank, NS convergence at fixed
n_iter), not population statistics — a single well-chosen cell CAN
answer "is THIS instrument's math internally consistent" definitively.
But it canNOT, on its own, establish that the SAME verdict holds at
K=78/K=90 too (§15.24.4's own AMBIGUOUS branch already registers a
2-cell escalation for exactly this reason). **TODO for the attack round:**
decide whether the mandatory grid should be widened to 2–3 cells upfront
rather than escalating only on an AMBIGUOUS result — a cost/certainty
tradeoff this design deliberately left as the cheaper, escalate-on-need
option. **Rev 1 status: NOT resolved by this round — remains open,
unchanged from Rev 0.** The independent round's own FATAL F1 fix adds a
DIFFERENT same-K contingency escalation (seeds 1943/1944, fired only on
the NO-REPRO path, §15.24.4 Step −1) — a partial precedent for
escalate-on-need over upfront-widening, but it answers a different
question (event-count sufficiency within K=84, not cross-K generalization
to K=78/90) and does not resolve Q4's own upfront-widening tradeoff.

**Q5. Does the `+15% overhead buffer` (§15.24.7) actually cover the
worst-case ~120 dump events, or is it a hand-waved number?** Disclosed
as "conservative... never before measured" — genuinely a guess, not a
measurement, unlike this program's other cost numbers (which are pulled
from real realized data). **TODO for the attack round:** either accept
the disclosed uncertainty (the 2× pessimistic ceiling already absorbs a
much larger miss than 15%) or require a even-cheaper CPU-only dry-run
timing the `.detach().cpu()`+`torch.save` cost of a single dump event
before trusting the multiplier. **RESOLVED at Rev 1 (attack-round-1
MINOR m2 fix):** the first remedy taken — the disclosed uncertainty is
accepted as-is (§15.24.7), since the 2× pessimistic bracket already
absorbs a 100% miss on the 1× point estimate, comfortably larger than any
plausible miss on a 15% first-time-code-path guess; no CPU-only dry-run
is required before launch. (Separately, MINOR m1 fixed a real arithmetic
error in the adjacent DISK figure this same overhead-disclosure paragraph
carried — ≈490MB corrected to ≈1GB — a distinct bug from Q5's own
GPU-time-buffer question, not its resolution.)

### 15.24.10 ATTACK-ROUND-1 fix-map (2026-07-08) — verdict NEEDS-REVISION

An independent adversarial pass reviewed §15.24 (this C17 eval-admission
repro instrument, Rev 0) before any GPU work launched, per this program's
own standing multiple-independent-audit-rounds discipline and per
§15.24.8's own registered prerequisite (mirrors
`REASONING_LINK_DESIGN.md` §16.7's and this file's own §15.21 fix-map
convention, house style). Verdict: **NEEDS-REVISION** — 1 FATAL, 2 MAJOR,
2 MINOR. The attack round also independently re-verified CLEAN, requiring
no fix: the cost arithmetic (other than the disk-line slip below), the
Stage −1 config-closure sha256/`build_cmd()` diff, the anchor-bypass
wiring (§15.24.9 Q3), the kernel-safety/Gate-2 reuse-by-citation, and the
PI-sign-off token precedence (never OR'd). Every finding below is fixed
in this revision (Rev 1, §15.24.2/§15.24.4/§15.24.5/§15.24.7); none is
deferred or waved away. Findings are recorded near-verbatim for the
historical record, per house style; resolutions are stated as landed in
this text, not as intentions.

| # | Finding (attack-round on §15.24, Rev 0) | Severity | Fix (Rev 1) | Location |
|---|---|---|---|---|
| FATAL-1 | The three-way decision rule (§15.24.4) operates only on `fallback_dump_sink` events; a ZERO-event re-run — plausible, since this repo already measured seed-fixed training-trajectory drift consistent with GPU run-to-run floating-point nondeterminism at a fixed seed (`KEY_ANCHORING_DESIGN.md:1976–1994`), and no `torch.use_deterministic_algorithms`/cudnn-determinism pin exists anywhere in the training path (verified this session: `run_deltanet_rd.py`, `model_rd.py`, `key_anchoring.py`, `run_deltanet_rd_exactness_sweep.py` all grep-clean for `use_deterministic_algorithms`/`cudnn.deterministic`) — makes Steps 0a/0b vacuously pass and Step 2's universally-quantified rule (`all()` over an empty list → `True`) silently emit TOLERANCE-MISCALIBRATION, an unearned "unlock the 11 cells" paper claim | FATAL | New **Step −1** zero/low-event guard (§15.24.4): `len(fallback_dump_sink) < 3` refuses all three named verdicts and emits **NO-REPRO**, triggering the pre-registered fallback of firing the reserved K=84 contingency seeds 1943/1944 (`run_deltanet_rd_exactness_sweep.py:3097`) — re-derived cost `2 × 0.450 = 0.900 GPU-h`, ledger still fits (`19.0196 + 0.900 = 19.9196/21 = 94.86%`, under the ORIGINAL non-extended ceiling). If the combined primary+contingency sink is STILL below the minimum → **AMBIGUOUS-NONDETERMINISM**, with candidate (1)'s checkpoint-payload extension promoted to the primary next instrument. Registered as a real `assert` + synthetic empty-list negative test, Stage −1 item 6 (§15.24.5) — the "assertion has teeth" rule | §15.24.4 (new Step −1 row + outcome list); §15.24.5 (Stage −1 item 6, new abort branch); §15.24.7 (contingency cost + ledger row) |
| MAJOR-1 | Offline Step-1 NS recompute precision was unpinned — TF32 matmul mode could flip the dispositive live/offline agreement check exactly at the near-boundary population, and the live run's own TF32 state was never recorded, so a mismatch couldn't even be diagnosed after the fact | MAJOR | Offline recompute now runs BOTH modes: strict-fp32 (`allow_tf32=False`) and matched-mode (TF32 set to the LIVE run's own recorded state); the tap spec (§15.24.2 item (ii)) now records `torch.backends.cuda.matmul.allow_tf32`/`torch.backends.cudnn.allow_tf32` once per checkpoint file. Matched-mode is Step 1's primary agreement check, strict-fp32 corroborating. A mode-dependent flip between the two offline modes is reported as a new named sub-finding, **TF32-SENSITIVE**, routed to Step 2's TOLERANCE-MISCALIBRATION examination, never treated as INSTRUMENT-BUG. New negative test verifying the tap reads live process state, not a cached/default value | §15.24.2 item (ii) (tap spec + file format); §15.24.4 (TF32 precision-pinning paragraph + Step 1 row) |
| MAJOR-2 | Steps 0a/0b/1 fired dispositively on ANY single anomalous episode among up to ~120 dumped events — a single-episode fluke (floating-point jitter, one unlucky sampled batch) could poison a whole cell's verdict | MAJOR | New episode-granularity threshold, applied uniformly to Steps 0a/0b/1: **dispositive floor = ≥2 distinct episodes, OR ≥2% of dumped events, whichever is larger**; exactly 1 anomalous episode → flagged, EXCLUDED from the cell's analysis population, disclosed by `step`/`hop`/`batch_idx` in the output JSON, verdict computed on the remainder. Justified in a dedicated paragraph (§15.24.4): a systematic bug is a code-path property and recurs across independently-sampled episodes; a single occurrence is statistically consistent with an ordinary tail event. Step −1's own `<3`-event NO-REPRO minimum is this same floor restated as the entry bar (FATAL-1's own fix). New boundary negative test at exactly 1, exactly 2, and the relative-floor crossover, Stage −1 item 7 (§15.24.5) | §15.24.4 (granularity-threshold paragraph + all four table rows); §15.24.5 (Stage −1 item 7, revised abort branch) |
| MINOR-1 | Disk worst-case arithmetic (§15.24.7) counted only ONE ≈4.1MB tensor per dump event (≈490MB), but §15.24.2 item (ii)'s own dict literal dumps BOTH `k_eff_raw` AND `k_blend_raw` per event, deliberately, for the free bitwise-equality re-confirmation — the disk figure silently undercounted by 2× | MINOR | Corrected to `4.1MB × 2 tensors × ≤120 events ≈ 984MB ≈ ~1GB` worst-case disk | §15.24.7 (telemetry-overhead disclosure paragraph) |
| MINOR-2 | Two loose ends bundled: (a) the `rank(k_eff_raw)<K` precheck (Step 0a, `tol=1e-4`) was dispositive despite its own §15.24.9 Q2 disclosing the tolerance was never rigorously derived; (b) §15.24.9 Q5 (does the `+15%` overhead buffer actually cover the worst case?) was left an open TODO rather than adjudicated | MINOR | (a) Step 0a demoted from dispositive to corroborating — reported and disclosed alongside Step 1's own (now-dispositive) live/offline disagreement, never independently sufficient for INSTRUMENT-BUG (§15.24.9 Q2's own remedy). (b) Q5 closed by disclosure: the `+15%` buffer is kept as-is, since the already-existing 2× pessimistic ceiling absorbs a 100% miss on the 1× estimate — comfortably larger than any plausible miss on a 15% first-time guess — so no CPU-only dry-run is required before launch | §15.24.4 (Step 0a row); §15.24.7 (overhead-disclosure paragraph); §15.24.9 (Q2, Q5, both marked RESOLVED) |

**What Rev 1 could NOT cleanly fix, disclosed rather than hidden:** §15.24.9
Q1 (does `k_eff_raw` carry a silent bf16 downcast anywhere upstream,
which would make Step 1's fp32 recompute not apples-to-apples?) and Q4
(should the mandatory grid widen to 2–3 cells upfront rather than
escalating only on AMBIGUOUS/NO-REPRO?) were **not** in this attack
round's scope and remain open exactly as Rev 0 left them — this revision
does not claim to have addressed them, though Q1's dtype fear was
independently verified NOT to hold this session (real code check, cited
inline at §15.24.9), a narrower confirmation than a full fix. **Rev 1's
own independent audit pass landed as attack round 2** (§15.24.11) —
verdict NEEDS-REVISION, 1 FATAL + 3 MAJOR + 4 MINOR, all fixed in Rev 2
(§15.24.1/§15.24.2/§15.24.4/§15.24.5). This confirms the standing rule's
own premise: round 2 found different bugs than round 1 — a
unit-of-analysis conflation in the granularity floor (row vs. event, both
called "episode"), an ordering gap between 0b and Step −1, and a missing
enforced-abort branch for 0b — none of which round 1's own pass caught.
**Rev 2's own independent audit pass landed as attack round 3**
(§15.24.12) — verdict NEEDS-REVISION, 1 FATAL + 2 MAJOR + 6 MINOR, all
fixed in Rev 3 (§15.24.2/§15.24.4/§15.24.5/§15.24.7). Round 3 caught yet
a different bug class than rounds 1–2: a NOISE argument (Rev 2's own
recurrence floor) wrongly applied to a STRUCTURAL, zero-floating-point
check (Step 0b's pool-membership precheck), plus a launch-identity gap in
the up-to-3-launch combined-sink event model and a batch-size confound in
the offline NS recompute — again, none of which the prior two rounds
caught, reconfirming the standing rule's own premise a third time.
**Rev 3 itself has not yet had its own independent audit pass** — round
4, next.

---

### 15.24.11 ATTACK-ROUND-2 fix-map (2026-07-09) — verdict NEEDS-REVISION

A second independent adversarial pass reviewed §15.24 (Rev 1, landed
2026-07-08 with attack-round-1's own 5 findings fixed, §15.24.10) before
any GPU work launched, per this program's own standing multiple-
independent-audit-rounds discipline — the same discipline that produced
Rev 1 now applied to Rev 1's own text, exactly as §15.24.10's own closing
paragraph anticipated. Verdict: **NEEDS-REVISION** — 1 FATAL, 3 MAJOR, 4
MINOR. Every finding below is fixed in this revision (Rev 2,
§15.24.1/§15.24.2/§15.24.4/§15.24.5); none is deferred or waved away.
Findings are recorded near-verbatim for the historical record, per house
style; resolutions are stated as landed in this text, not as intentions.

| # | Finding (attack-round-2 on §15.24, Rev 1) | Severity | Fix (Rev 2) | Location |
|---|---|---|---|---|
| FATAL-1 | "Episode" was never pinned to one referent: Step 0a's own rank check already operated on ONE `(K,d)` row (`k_eff_raw_episode`), while the granularity-threshold paragraph's own "~120 dumped events" denominator counted whole triggering batches — two readings that differ by the batch size, ~128× (120 events vs. ~15,360 rows), in the dispositive-floor arithmetic. The codebase's own docstring (`model_rd.py:433–434`) already distinguishes row-level "episode" from batch-level triggering; `newton_schulz_orthogonalize`'s `resid` is `(B,)` per-row (`model_rd.py:377`/`:397`); `geo3_orthogonalize_logged`'s `fallback_triggered` ORs over up to 128 rows (`model_rd.py:456`); dumped `key_ids` is `(128,84)` | FATAL | `episode := (step, hop, batch_idx, row_idx)`, one within-batch `(K,d)` problem, pinned explicitly (§15.24.4); `event := one dumped dict, one triggering (step,hop,batch_idx) batch`, bundling up to 128 episodes. Re-deriving the worst case at the CORRECT (episode) granularity — `120 events × 128 rows = 15,360 episodes`, `⌈2%⌉ = 308` — shows the inherited percentage clause is unworkable at episode granularity (308 anomalous episodes is a bar a genuinely broken probe would plausibly never clear); the percentage clause is DROPPED. Replaced with a two-level ABSOLUTE floor: **≥2 anomalous episodes, occurring in ≥2 distinct events** — both conditions required jointly, a direct restatement of the original M2 intent ("recurs across independent draws, not one bad batch") at the granularity the checks actually operate at. Exclusion disclosures now name the full `(step,hop,batch_idx,row_idx)`, not just `(step,hop,batch_idx)`. Worked boundary examples pinned (E1/E2/E3) and exercised in a rewritten Stage −1 item 7 | §15.24.4 (new episode/event definition paragraph, rewritten floor paragraph, worked examples, all 5 table rows); §15.24.5 (Stage −1 item 7, rewritten) |
| MAJOR-2 | Step −1's `<3`-event reproduction bar and Step 0b's `≥2`-episode structural floor disagreed on a 2-event sink with 2 pool-mismatch episodes (a deterministic bug signature) — Step −1 refused everything (AMBIGUOUS-NONDETERMINISM), misdirecting remediation toward a nondeterminism investigation instead of the real structural bug | MAJOR | Reordered: Step 0b (structural) now runs FIRST, on whatever events exist, before Step −1's reproduction bar — structural bug evidence needs no reproduction statistics. If 0b's floor fires, INSTRUMENT-BUG is emitted immediately; Step −1 is never reached for that verdict. Step −1 now gates ONLY the verdicts that DO need statistical weight (0a, Step 1, Step 2). E1 (2-event, 2-mismatch sink) now correctly resolves to INSTRUMENT-BUG, not AMBIGUOUS-NONDETERMINISM | §15.24.4 (table reordered, Step 0b and Step −1 rows rewritten); §15.24.5 (abort-branch list reordered) |
| MAJOR-3 | Step 0b's dispositive trigger had no enforced-abort branch in §15.24.5's list (only Step 1's did) — nothing in the registered build tasks actually forced the analysis script to refuse REAL-CAPACITY-BOUNDARY/TOLERANCE-MISCALIBRATION when 0b fired | MAJOR | 0b added to §15.24.5's enforced-abort list, as the FIRST-checked branch (per MAJOR-2's reordering); own negative test registered (Stage −1 item 8: a synthetic 2-event, 2-pool-mismatch sink — E1 exactly — asserts ONLY INSTRUMENT-BUG is emitted, refusing the other two). Total precedence pinned explicitly: **0b (structural) > Step −1 (reproduction) > Step 1 (agreement) > Step 2 (tolerance)** — every telemetry state maps to exactly one primary verdict | §15.24.4 (precedence-order paragraph); §15.24.5 (new abort branch + Stage −1 item 8) |
| MAJOR-4 | §15.24.2 item (ii)'s dump dict literal referenced `step_arg_passed_in`, but `evaluate_pool()`'s real signature (`run_deltanet_rd.py:231–234`) has no `step` parameter — the spec as written did not correspond to real, callable code | MAJOR | Additive `step: int \| None = None` parameter registered on `evaluate_pool()`, passed only at the train()-loop C17 call site (`run_deltanet_rd.py:805–806`, using the loop's own in-scope `step` variable, `:687`), byte-identical (`None`) at every other call site (m2/m3/c19, `:802–804`/`:807–808`). Also corrected (found verifying the same code block): the per-batch loop (`:263`) has no index variable today, so the dict's own `batch_idx: batch_i` field was equally undefined; registered as `for batch_i in range(n_batches):`, additive, zero behavior change | §15.24.2 item (ii) (parameter list, loop signature, dict literal) |
| MINOR-1 (m1) | §15.24.1's mechanism-space table (row (b), INSTRUMENT-BUG) still called Step 0a's rank check "a DISPOSITIVE, zero-tolerance structural fact... conclusive on its own" — stale since Rev 1's own MINOR m2 demoted 0a to corroborating; §15.24.1 itself was never updated when §15.24.4 was | MINOR | §15.24.1 row (b)'s discriminating-observation cell rewritten: the dispositive trigger is now named as 0b (pool-membership) or Step 1 (live/offline disagreement), each gated by the two-level floor; 0a explicitly marked CORROBORATING ONLY, unchanged by Rev 2 | §15.24.1 (mechanism-space table, row (b)) |
| MINOR-2 (m2) | Citation error: §15.24.2 item (iii) cited `model_rd.py:149` for `self.geo3_last_resid`'s assignment; the real line is `:1149` | MINOR | Citation corrected to `model_rd.py:1149` throughout (§15.24.2, §15.24.4's new episode-definition paragraph) | §15.24.2 item (iii); §15.24.4 |
| MINOR-3 (m3) | TF32 matched-mode recompute (Rev 1 MAJOR M1) assumed a single source run; once the combined sink can span up to 3 launches (primary + 2 contingency seeds, Step −1), a matched-mode recompute using only the PRIMARY run's recorded TF32 flags would silently mismatch events sourced from a contingency-seed launch if that launch's actual TF32 state ever differed | MINOR | Matched-mode recompute for each event now explicitly uses THAT event's own source-run's recorded `tf32_matmul`/`tf32_cudnn` (each launch's own `c17fallback_step<N>.pt` already records its own pair, §15.24.2 item (ii)) — never assumed uniform across the combined sink | §15.24.4 (TF32 precision-pinning paragraph) |
| MINOR-4 (m4) | Stage −1 item 4 (determinism cross-check) was specified to run ONCE, "immediately post-launch" — but Step −1's own contingency path can now trigger up to 2 additional launches, each of which needs its own determinism check before its events are trusted in the combined sink | MINOR | Item 4 now runs PER LAUNCH — primary run, and again for each fired contingency seed — each time sequenced immediately after THAT launch and before ITS events are folded into the combined sink | §15.24.4 (determinism cross-check paragraph); §15.24.5 (Stage −1 item 4) |

**What Rev 2 could NOT cleanly fix, disclosed rather than hidden:** none
of attack-round-2's own findings were left unaddressed — all 8 (1 FATAL +
3 MAJOR + 4 MINOR) are fixed above. Attack-round-0's own still-open
self-attack questions, Q1 (§15.24.9, `k_eff_raw`'s dtype provenance beyond
the narrower TF32-mode question) and Q4 (§15.24.9, whether the mandatory
grid should widen to 2–3 cells upfront), were **not** in round 2's scope
either and remain open exactly as Rev 1 left them — now two rounds
unaddressed (§15.24.9's own updated forward-pointer). Landing
attack-round-2's findings did not, on its own, certify Rev 2 as
CLEARED-FOR-BUILD — confirmed by round 3: **Rev 2's own independent audit
pass landed as attack round 3** (§15.24.12) — verdict NEEDS-REVISION, 1
FATAL + 2 MAJOR + 6 MINOR, fixed in Rev 3. Per this project's standing
rule that multiple independent adversarial rounds catch different bugs
each round (demonstrated again this round: a noise-argument-on-a-
structural-check FATAL, a launch-identity gap, and a batching confound
were ALL real and ALL invisible to rounds 1–2's own passes), **Rev 3
itself has not yet had its own independent audit pass** — round 4, next.

---

### 15.24.12 ATTACK-ROUND-3 fix-map (2026-07-10) — verdict NEEDS-REVISION

A third independent adversarial pass reviewed §15.24 (Rev 2, landed
2026-07-09 with attack-round-2's own 8 findings fixed, §15.24.11) before
any GPU work launched, per this program's own standing multiple-
independent-audit-rounds discipline — the same discipline that produced
Rev 1 and Rev 2 now applied a third time, exactly as §15.24.11's own
closing paragraph anticipated. Verdict: **NEEDS-REVISION** — 1 FATAL, 2
MAJOR, 6 MINOR. Every finding below is fixed in this revision (Rev 3,
§15.24.1/§15.24.2/§15.24.4/§15.24.5/§15.24.7/§15.24.8/§15.24.9); none is
deferred or waved away. Findings are recorded near-verbatim for the
historical record, per house style; resolutions are stated as landed in
this text, not as intentions.

| # | Finding (attack-round-3 on §15.24, Rev 2) | Severity | Fix (Rev 3) | Location |
|---|---|---|---|---|
| FATAL-1 | Rev 2's own two-level dispositive floor (≥2 anomalous episodes across ≥2 distinct events) is a NOISE argument — "a systematic bug recurs across independent draws; an ordinary tail event does not" — sound for a NUMERICAL check (Step 1's live/offline residual comparison, where a near-boundary value genuinely can jitter) but wrongly applied, unchanged, to Step 0b's pool-membership precheck, which is STRUCTURAL: a dumped entity id either is or is not a member of the disjoint held-out pool, computed with zero floating-point arithmetic. A single pool-mismatch is already deterministic proof of a bug, exactly the case this project's own "exact threshold, no tolerance slack copied from a floating-point context" rule addresses. Concretely (adversarial state 3): a real pool-mismatch in a 5-event sink falls below the 2-event floor, is EXCLUDED, and the verdict silently continues to REAL-CAPACITY-BOUNDARY or TOLERANCE-MISCALIBRATION on the untainted remainder — a confidently wrong REAL-CAPACITY-BOUNDARY claim | FATAL | Floor SPLIT: Step 0b is now dispositive on ANY SINGLE pool-membership violation, no event-count or episode-count minimum, mirroring `model_rd.py:2048`'s own assert-exactly-zero convention for a different structural fact. The ≥2-episode/≥2-distinct-event recurrence bar now gates Step 1's numerical disagreement check ONLY (unchanged from Rev 2 in every other respect). Floors counted PER-MARKER-TYPE, never combined, with a dedicated negative test (Stage −1 item 10, "state 6"). Precedence text and the "every telemetry state maps to exactly one verdict" totality claim both re-derived and re-affirmed under the split. New worked example E4 ("state 3": a 5-event sink, exactly 1 pool-mismatch episode) and its own negative test (Stage −1 item 8's new sub-case) | §15.24.4 (floor paragraph split, precedence paragraph, table Step 0b/0a/1 rows, worked examples E1–E5); §15.24.5 (Stage −1 items 7/8 revised, enforced-abort branch rewritten); §15.24.1 (row (b)) |
| MAJOR-A | Event identity `(step, hop, batch_idx)` is not launch-unique across the up-to-3-launch combined sink Step −1's own NO-REPRO contingency path can create (primary seed 1940 + contingency seeds 1943/1944, each an independent re-run of the identical training loop); a cross-launch reproduction landing at identical `(step, hop, batch_idx)` coordinates could wrongly dedup to "1 event," corrupting Step 1's own `≥2 DISTINCT events` floor arithmetic (a genuinely-recurring 2-launch signal could undercount as 1 and wrongly fail to clear the floor) | MAJOR | Additive `seed` field on every dumped event (`evaluate_pool()` gains a third optional parameter, `seed: int \| None = None`, threaded only at the C17 call site using `train()`'s own in-scope `seed` parameter, `run_deltanet_rd.py:584`); `episode` redefined as the 5-tuple `(seed, step, hop, batch_idx, row_idx)`, `event` as `(seed, step, hop, batch_idx)`. Cross-launch recurrence at identical `(step, hop, batch_idx)` coordinates (different `seed`) is disclosed as the STRONGEST reproduction evidence the instrument can produce, never discounted | §15.24.2 item (ii) (new `seed` parameter, event dict, rationale paragraph); §15.24.4 (episode/event tuple definitions, Step −1 row) |
| MAJOR-B | Step 1's offline recompute ran `newton_schulz_orthogonalize` on a batch-size-1 slice (`k_eff_raw[row_idx:row_idx+1]`) of the dumped tensor, while the live path always calls it batched at `B=128`; cuBLAS/cuDNN GEMM kernel selection is itself batch-size-dependent, so a near-boundary residual — exactly the population Step 1 exists to examine — could flip across the `0.01` threshold from batching alone, with zero real numerical disagreement involved | MAJOR | Offline recompute now runs ONE batched `newton_schulz_orthogonalize` call on the event's full dumped `(B,K,d)` tensor, matching the live call's own batching exactly, in both strict-fp32 and matched-mode; `resid_offline[row_idx]` is indexed from that batched result, never from a singleton recompute. New Stage −1 item 9: a batched-vs-singleton comparison on synthetic near-boundary data, disclosing any row where the two methods differ | §15.24.4 (TF32 precision-pinning paragraph, Step 1 table row); §15.24.5 (new Stage −1 item 9, enforced-abort branch note) |
| MINOR-1 | No negative test exercised a sink containing BOTH marker types (a 0b pool-mismatch episode and a Step 1 live/offline-disagreement episode, in different events) — Rev 2's own prose already specified "the SAME anomaly marker" for recurrence, but nothing proved the analysis script actually counts per-marker-type rather than summing across types at the exact 1-and-1 boundary | MINOR | New Stage −1 item 10: a synthetic ≥3-event sink with 1 pool-mismatch episode (event A) and 1 live/offline-disagreement episode (event B, a different event) asserts INSTRUMENT-BUG fires via 0b alone (event A), with Step 1's own marker (event B) reported EXCLUDED/disclosed, never summed toward a shared count (worked as E5, "state 6") | §15.24.4 ("floors counted per-marker-type" paragraph, E5); §15.24.5 (Stage −1 item 10) |
| MINOR-2 | §15.24.2's cell-selection paragraph used "per episode" to mean "per K-item pool draw" ("only 16 of 106 held-out entities are NOT drawn per episode at K=90") — a THIRD, distinct usage of the word from the two Rev 2's own FATAL fix pinned (`episode` := row, `event` := batch), re-overloading a now-precisely-defined term | MINOR | Reworded to "per K-item pool draw," consistent with the pinned row-level definition | §15.24.2 (cell-selection paragraph) |
| MINOR-3 | Citation error: the reserved K=84 contingency-seed tuple `(1943, 1944)` was cited at `run_deltanet_rd_exactness_sweep.py:3097` throughout §15.24.4/§15.24.7; the real line is `:3098` — `:3097` is the dict-open statement (`KEYANCHOR_SCALING_CONTINGENCY_SEEDS_BY_D_K[96].update({`), not the line the K=84 tuple itself sits on | MINOR | Citation corrected to `:3098` in the live decision-rule and ledger text; historical fix-map citations (§15.24.10 FATAL-1) left verbatim per house style | §15.24.4 (Step −1 row); §15.24.7 (NO-REPRO contingency cost paragraph) |
| MINOR-4 | The `k_eff_raw`/`k_blend_raw` bitwise re-confirmation (§15.24.2 item (ii)) was specified as "a live, free re-confirmation" of §15.23's anchor-bypass claim, but its own FAILURE mode was never stated — an implementation could plausibly log-and-continue on a failed assertion rather than stop | MINOR | Pinned as an explicit HARD-ABORT (`assert torch.equal(...)`, uncaught) on failure — a failure invalidates §15.23's own anchor-bypass assumption for THIS run, which invalidates every downstream 0b/0a/1/2 verdict computed under that assumption, so the analysis must stop, not flag one event and continue | §15.24.2 item (ii) (bitwise-equality paragraph) |
| MINOR-5 | The floor paragraph's own parenthetical ("Step 0a stays corroborating-only... unaffected by this fix") never named the COUNTING RULE 0a's own corroborating marker uses to decide when it is disclosed as "recurring" — left implicit even though the table row itself already applied a floor | MINOR | Floor paragraph now names 0a's own rule explicitly: the SAME two-level floor as Step 1 (≥2 episodes across ≥2 events) governs only how 0a's marker is DESCRIBED (recurring vs. single flag), never whether it is dispositive — 0a stays corroborating-only regardless, unchanged from Rev 1's own MINOR m2 demotion | §15.24.4 (floor paragraph, new 0a paragraph; table Step 0a row) |
| MINOR-6 | "residual AMBIGUOUS" (Step 2's mixed-episode outcome) was the only named verdict in the whole decision-rule table NOT following the hyphenated naming convention every other outcome uses (INSTRUMENT-BUG, REAL-CAPACITY-BOUNDARY, TOLERANCE-MISCALIBRATION, NO-REPRO, AMBIGUOUS-NONDETERMINISM, TF32-SENSITIVE) | MINOR | Renamed to **AMBIGUOUS-RESIDUAL** throughout the live text (table Step 2 row, named follow-on paragraph heading) | §15.24.4 (Step 2 table row; AMBIGUOUS-RESIDUAL follow-on paragraph) |

**What Rev 3 could NOT cleanly fix, disclosed rather than hidden:** none
of attack-round-3's own findings were left unaddressed — all 9 (1 FATAL +
2 MAJOR + 6 MINOR) are fixed above. Attack-round-0's own still-open
self-attack questions, Q1 (§15.24.9, `k_eff_raw`'s dtype provenance — now
explicitly distinguished from MAJOR-B's own adjacent-but-distinct
batch-size axis) and Q4 (§15.24.9, whether the mandatory grid should
widen to 2–3 cells upfront), were **not** in round 3's scope either and
remain open exactly as Rev 2 left them — now three rounds unaddressed
(§15.24.9's own updated forward-pointer). **Rev 3 itself has not yet had
its own independent audit pass** — per this project's standing rule that
multiple independent adversarial rounds catch different bugs each round
(demonstrated a third time this round: a noise-argument wrongly applied
to a structural check, a launch-identity gap, and a batch-size confound
were ALL real and ALL invisible to rounds 1–2's own passes), landing
attack-round-3's findings does not, on its own, certify Rev 3 as
CLEARED-FOR-BUILD. **Round 4 landed** (0 FATAL, 2 MAJOR, 1 MINOR — a
pool-reconstruction/two-seeds-trap gap, a ledger baseline gap, and a
minimal-boundary negative test; full finding→fix table at §15.24.13,
Rev 4). **Round 5, a VERIFY pass confirming Rev 4's three fixes land
clean (not a fresh full attack round), next.**

---

### 15.24.13 ATTACK-ROUND-4 fix-map (2026-07-11) — verdict NEEDS-REVISION

A fourth independent adversarial pass reviewed §15.24 (Rev 3, landed
2026-07-10 with attack-round-3's own 9 findings fixed, §15.24.12) before
any GPU work launched, per this program's own standing multiple-
independent-audit-rounds discipline — the same discipline that produced
Rev 1/Rev 2/Rev 3 now applied a fourth time, exactly as §15.24.12's own
closing paragraph anticipated. Verdict: **NEEDS-REVISION** — 0 FATAL, 2
MAJOR, 1 MINOR — the first round to land with zero FATALs, and the
narrowest, most surgical finding set of the four rounds so far. Every
finding below is fixed in this revision (Rev 4, §15.24.1/§15.24.4/
§15.24.5/§15.24.7/§15.24.9); none is deferred or waved away. Findings are
recorded near-verbatim for the historical record, per house style;
resolutions are stated as landed in this text, not as intentions.

| # | Finding (attack-round-4 on §15.24, Rev 3) | Severity | Fix (Rev 4) | Location |
|---|---|---|---|---|
| MAJOR-1 | The offline analysis must reconstruct `pools.heldout_name_ids`/`pools.train_name_ids` to evaluate Step 0b at all, but nothing in Rev 3's own text pinned HOW — `build_entity_pools` (`grammar_rd.py:194`) takes `seed` as an explicit argument, and the training path's own caller (`run_deltanet_rd.py:1470`) calls it with a HARDCODED `seed=0`, fully decoupled from the training `--seed` that Rev 3's own MAJOR-A fix made a load-bearing per-event field. A builder threading the launch seed (1940) into `build_entity_pools` — a natural mistake, since `seed` is already in-scope at analysis time for episode/event identity — would reconstruct the WRONG train/held-out partition, making every genuinely held-out id read as "not in pool" and firing a total, confidently wrong INSTRUMENT-BUG verdict on healthy code, indistinguishable in its own output from a real probe defect | MAJOR | Pinned in the Step 0b build task: offline pool reconstruction is ALWAYS the literal, hardcoded call `grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)` — verified against `run_deltanet_rd_exactness_sweep.py`'s own `build_cmd()` (never emits `--heldout-frac`, so this cell's launch always takes the CLI default 0.5) — NEVER the launch seed, with an explicit two-seeds-trap warning in the build comment. New prerequisite gate, ahead of even Step 0b: assert the reconstructed `pools.train_name_ids` is SET-EQUAL to the checkpoint's own archived `anchor_train_ids` tensor (`run_deltanet_rd.py:934–936`, already logged at every checkpoint, zero new cost, since `model.anchor_train_ids_buf` is registered directly from that SAME launch's `pools.train_name_ids`, `run_deltanet_rd.py:1606`); mismatch → HARD-ABORT before any verdict, a new named state (RECONSTRUCTION-FAILURE) distinct from INSTRUMENT-BUG. §15.24.1's row (b) also gains a restating sentence: the C17 sampler's heldout-exclusivity invariant is independently structural (`grammar_rd.py:194–253` non-overlapping shuffled-list slices with globally-unique ids; `grammar_rd.py:423,434–436` single-pool draw per episode). New Stage −1 item 11 (the `seed=1940`-vs-`seed=0` two-seeds-trap fixture, both positive and negative cases) | §15.24.4 (new "Offline pool reconstruction — the two-seeds trap" paragraph, precedence paragraph, Step 0b table row); §15.24.5 (Stage −1 item 11, new enforced-abort branch); §15.24.1 (row (b), verified note) |
| MAJOR-2 | §15.24.7's own ledger baseline (18.1196/26 GPU-h realized) omitted the K=69/seed=1733 contingency cell's own realized 0.427 GPU-h (`wall_s=1535.2s`, §15.22 addendum, landed 2026-07-08) — a real cell, run standalone via `run_k69_s1733_contingency.py` AFTER the 16-cell wide-grid harvest the 6.3331 GPU-h figure sums, and never folded back into any statement of the running total anywhere the ledger is quoted (this draft, `EXPERIMENT_LOG.md`, `STATE.md`) | MAJOR | Baseline corrected: 11.7865 + 6.3331 + 0.427 = **18.5466/26 GPU-h realized**. Every downstream figure re-derived: 1× total 18.9966/26 (90.46%/21); 2× total 19.4466/26 (92.60%/21); 2×+contingency total **20.3466/26 (96.89%/21)**, margin **0.6534 GPU-h** against the ORIGINAL 21 GPU-h ceiling (down from the previously-claimed 1.0804 GPU-h — a ~40% tightening of the reserve). The underlying conclusion (fits inside the ORIGINAL ceiling without the `+5.0 GPU-h` extension) survives unchanged; the margin backing it does not, and is disclosed plainly rather than silently re-derived. Folded into `EXPERIMENT_LOG.md`'s running-total convention (new sentence on the K=69/s1733 harvest entry) and `STATE.md` (correction note on the wide-grid-wave harvest block, the canonical source of the pre-correction figure) — the gap was repo-wide, fixed everywhere the stale number lived, not only in this draft | §15.24.7 (sub-ledger arithmetic paragraph, ledger table, "Notable" paragraph); `EXPERIMENT_LOG.md`; `STATE.md` |
| MINOR-1 | Step 0b's own prose already claims dispositive-on-any-single-violation holds "in a sink of any size — including a single-event sink, before Step −1's own `<3`-event reproduction gate has even run," but neither registered fixture actually tests a single-event sink — E1 (Stage −1 item 8) is a 2-event sink and E4 (Stage −1 item 8's own sub-case) is a 5-event sink; the single most degenerate case the prose itself names was asserted but never exercised | MINOR | New Stage −1 item 12: a synthetic sink with `len(fallback_dump_sink)==1` containing exactly 1 pool-membership-mismatch episode, asserting INSTRUMENT-BUG fires and is confirmed BEFORE Step −1's own `<3`-event gate logic is even reached — the minimal-boundary fixture 0b's own prose already promised | §15.24.5 (new Stage −1 item 12; enforced-abort branch's negative-test citation extended) |

**What Rev 4 could NOT cleanly fix, disclosed rather than hidden:** none
of attack-round-4's own findings were left unaddressed — all 3 (0 FATAL +
2 MAJOR + 1 MINOR) are fixed above. Attack-round-0's own still-open
self-attack questions, Q1 (§15.24.9, `k_eff_raw`'s dtype provenance) and
Q4 (§15.24.9, whether the mandatory grid should widen to 2–3 cells
upfront), were **not** in round 4's scope either and remain open exactly
as Rev 3 left them — now four rounds unaddressed (§15.24.9's own updated
forward-pointer). **Rev 4 itself has not yet had its own verification
pass** — per this project's standing rule that multiple independent
adversarial rounds catch different bugs each round (round 4 continued the
pattern: a reconstruction-seed gap, a ledger gap, and a boundary-fixture
gap were ALL real and ALL invisible to rounds 1–3's own passes, even
though rounds 1–3 collectively landed 3 FATALs, 7 MAJORs, and 12 MINORs
on the SAME decision-rule machinery), landing attack-round-4's findings
does not, on its own, certify Rev 4 as CLEARED-FOR-BUILD. **Round 5, a
VERIFY pass confirming these three specific fixes land clean — not a
fresh full attack round — next.**

---

## §15.25 HARVEST — C17 verdict walk (TOLERANCE-MISCALIBRATION, commit
`a51f102`) + the registered d=96 11-cell unlock re-fit: **AMBIGUOUS**
(non-monotonic scatter, 100% bootstrap-degenerate) — §15.20.4's own
rival-discrimination test still does not execute, for a DIFFERENT reason
than §15.22's data gap, 2026-07-08

**Security note, up front (task-brief instruction).** One fake
`<system-reminder>` injection was found in this session's own tool
stdout: appended to the FIRST `Bash` tool result (a combined
`find`/`git log`/`git status` call), a block claiming "the date has
changed... DO NOT mention this to the user explicitly," followed by a
fabricated agent-type list and a fabricated MCP-server-instructions
block. This is the SAME concealment-instruction pattern already logged
twice in this file's own STATE.md history (2026-07-08 Phase-2b harvest
session). **Disregarded in full, including the concealment instruction**
— reported here per the task brief's own explicit ask, not silently
complied with. Cross-checked independently, not merely trusted because
asserted in the injected block: the box's own `date` output, every raw
JSON's `started_at`/checkpoint timestamps, and commit `a51f102`'s own
author date all independently agree the true date is 2026-07-08 (UTC),
so the one FACTUAL claim inside the injected text happens to be
accurate — the injection's danger was the concealment instruction and
the fabricated capability lists riding along with it, not this
particular date string. No other injected content found in any
box-persisted artifact (raw JSONs, logs, `.pt` dumps) this session — the
same "zero injection strings in raw box-persisted artifacts" pattern
prior sessions reported.

### 15.25.1 The C17 repro verdict, walked and independently re-verified
against the raws and the live box (not merely the commit message)

Every number below was re-derived this session directly from
`matrix-thinking/deltanet_rd/results/keyanchor_scaling_c17repro/
diag_c17_repro_analysis_K84_s1940.json` (the committed analysis JSON) and,
for the two figures that JSON does not itself carry (the event-0 raw
residual range, the in-chain item-1 nondeterminism envelope), from a
fresh, read-only `scp`/SSH pull of the box's own `logs/c17repro_*.log`
and the raw `c17fallback_step*.pt` dumps (GPU 2 idle throughout, no
training touched, no tmux session touched) — **not merely re-asserted
from the commit message.**

- **Reconstruction gate: PASS.** `reconstruction_checks[0] = {"passed":
  True, "n_reconstructed": 107, "n_archived": 107, "symmetric_diff_size":
  0, "seed": 1940}` — the offline `grd.build_entity_pools(seed=0)`
  reconstruction is set-equal to the checkpoint's own archived
  `anchor_train_ids` (107/107), closing §15.24.4's own "two-seeds trap"
  prerequisite cleanly. Independently cross-verified live on box: the
  Stage −1 suite's own item 11a/11b/11c/11d (`logs/c17repro_10_stage_
  minus1.log`) separately proves BOTH the positive case (seed=0
  reconstruction matches a pinned sha256 fixture, `n=107`) and the
  negative trap (seed=1940 threaded into `build_entity_pools` does NOT
  match, `symmetric_diff_size=96`, correctly HARD-ABORTS with
  `RECONSTRUCTION-FAILURE`) — the gate has teeth, not merely a passing
  shape check.
- **Step 0b (pool-membership precheck): 0 violations.** `step_0b =
  {"dispositive": False, "n_violations": 0, "violations": []}` — every
  dumped C17 `key_ids` row is an exact set-member of the reconstructed
  `heldout_name_ids`, zero exceptions across all 36 events. Rules out
  candidate (b) INSTRUMENT-BUG immediately (Rev 3's own any-single-
  violation rule never fires here).
- **Step −1 (reproduction-count gate): 36 distinct events, cleared.**
  `step_neg1 = {"n_distinct_events": 36, "min_events": 3, "cleared":
  True}`. Independently re-counted directly from the box's own raw
  `c17fallback_step<N>.pt` dumps (read-only `torch.load`, CPU): **12
  events each at checkpoints 16000/18000/20000, ZERO at every checkpoint
  2000–14000** (confirmed by loading all 10 checkpoint files and counting
  `len(d["events"])` directly — 0,0,0,0,0,0,0,12,12,12). 12 events/
  checkpoint = 3 `H_train` hops × 4 batches, matching the design's own
  registered dump shape. No contingency seed (1943/1944) was needed —
  the primary launch alone cleared the `≥3`-event floor by 12×.
- **Step 0a (exact-rank precheck, corroborating only): 0 anomalous.**
  `step_0a = {"n_anomalous_episodes": 0, "n_distinct_events": 0,
  "floor_met": False, ...}` — no rank-deficient `k_eff_raw` rows found;
  consistent with (not independently dispositive for) ruling out (b).
- **Step 1 (live/offline NS agreement): 0 disagreements, 0 TF32-sensitive
  flips.** `step_1 = {"disagreements": [], "tf32_sensitive": [],
  "n_anomalous_episodes": 0, ...}` — the offline batched recompute
  (matched-mode) agrees with the LIVE dumped `resid`/fallback flag on
  every one of the 4,608 episodes, and neither offline mode (strict-fp32
  vs matched) ever disagrees with the other. Recorded precision flags,
  read directly from the raw dump files this session: `tf32_matmul=False,
  tf32_cudnn=True` at every one of the 10 checkpoint files (uniform for
  the whole run, as expected — a process-level, not per-event, setting).
  Rules out (b) a second, independent way (Step 1's own two-level floor
  is moot here — zero disagreements, not merely below-floor).
- **Step 2 (n_iter sweep, the verdict-bearing step): every one of 4,608
  episodes resolves by n_iter≤28.** Recomputed directly from
  `step_2.per_episode_resolve_niter` (4,608 entries, one per `(seed,
  step, hop, batch_idx, row_idx)`): **4,313 resolve at n_iter=24, 295 at
  n_iter=28, 0 at n_iter=32, 0 at n_iter=40** — matches the commit
  message exactly, independently re-tallied via `collections.Counter`,
  not merely re-read. → **`verdict: TOLERANCE-MISCALIBRATION`,
  `verdict_source: step_2`**, per §15.24.4's own Step 2 rule ("every
  episode's residual ≤0.01 by n_iter≤32 → TOLERANCE-MISCALIBRATION").
  **Live-resid magnitude at the production `n_iter=20` setting, freshly
  pulled from the box's own raw dump (not previously in any committed
  JSON, since only the small analysis artifact was pulled to the repo):**
  the first (`event-0`) triggering event at the FINAL checkpoint
  (`step=20000, hop=1, batch_idx=0`) shows anomalous-row residuals
  ranging **`0.6961`–`1.3715`** (all 128/128 rows anomalous at that late
  checkpoint) — confirms the commit message's disclosed
  "event-0 range 0.696–1.371" figure to 3 decimal places. **New,
  previously-undisclosed observation from this session's own raw-dump
  pull:** the near-miss WORSENS over the course of training — checkpoint
  16000's own 12 events show only 13–27/128 anomalous rows with maxima
  0.07–0.43, while checkpoints 18000 and 20000 show ALL 128/128 rows
  anomalous with maxima climbing to 1.08 and 1.43 respectively — i.e.
  this is not a fixed, isolated near-miss but a population that drifts
  further from the `n_iter=20` boundary as the anchor-adjacent geometry
  keeps training, YET remains fully resolved by `n_iter≤28` at every
  single one of the 4,608 examined episodes, including the worst-case
  late-checkpoint population — the strongest evidence yet that this is
  genuinely iteration-count-fixable, not a structural wall that would be
  expected to worsen past `n_iter=40` as training continues.
- **STEP 3b (per-launch determinism replay): PASS, 12/12 byte-identical.**
  `c17_repro_replay_launch_dump_seed1940_step20000.json`:
  `{"n_live_events": 12, "n_replay_events": 12, "count_match": true,
  "byte_identical": true}` — independently re-derived `key_ids`/
  `k_eff_raw` from a fresh `state_dict` reload matches the live dump
  bit-for-bit, at the checkpoint with the most events (20000), the
  strongest test of the 3 checkpoints available.
- **Full §15.24.4 precedence order, confirmed satisfied exactly as
  designed:** reconstruction gate (PASS) → 0b (0 violations) → −1 (36≥3,
  cleared) → 0a (0, corroborating) → 1 (0 disagreements) → 2 (resolves by
  n_iter≤28) → **TOLERANCE-MISCALIBRATION**. Every step ran, in order,
  none skipped, none short-circuited except in the direction the design
  itself specifies (0b's own all-clear lets −1 run; −1's clearance lets
  0a/1 run; 1's full agreement lets 2 run).

### 15.25.2 The item-1 re-pin adjudication — nondeterminism envelope,
now confirmed reproducing a SECOND time under a fresh launch

**Background (2026-07-07 deploy-session build note, §15.24.2, unchanged
history):** the originally-registered bitwise OFF-vs-ON telemetry smoke
was UNSATISFIABLE on this hardware — a same-flag, same-seed OFF-vs-OFF
rerun already diverges from step 4 (`max_abs_dev=7.51e-04`,
`n_differing=46/50`), the SAME fixed-seed GPU nondeterminism
`KEY_ANCHORING_DESIGN.md`~L1976–1994 already documents, independently
confirmed NOT a telemetry-perturbation artifact (a code-path causal
cross-check: the fixture's only telemetry-touched code runs AFTER the
step-50 loss is logged; OFF-vs-ON deviations, 7.3e-04/6.0e-04, sat
INSIDE the OFF-vs-OFF envelope, not outside it). Item 1 was re-pinned
**baseline-relative**: `OFF×2 (envelope) + ON×1`, PASS iff `max_abs
OFF-vs-ON dev ≤ 3× the OFF-vs-OFF envelope` (envelope=0 degrades to the
original bitwise spec on deterministic hardware — a strict
generalization, not a loosening).

**This session's own fresh confirmation (in-chain, `logs/
c17repro_10_stage_minus1.log` line 157, re-pulled and re-verified this
harvest, not merely cited):** item 1 re-ran, from scratch, as part of the
SAME chain launch that produced the TOLERANCE-MISCALIBRATION verdict —
`PASS -- n_steps_logged=50 off_vs_off_envelope=8.841e-04
max_off_vs_on_dev=8.354e-04 threshold=2.652e-03 bitwise_mode=False`.
**This is a materially different envelope magnitude than the original
2026-07-07 adjudication run** (8.841e-04 vs. 7.51e-04, ~18% larger) —
exactly the behavior expected of genuine run-to-run GPU nondeterminism
(a real, non-reproducible physical quantity, not a fixed constant one
run happens to measure) rather than a deterministic bug that would
re-measure identically every time. `max_off_vs_on_dev=8.354e-04` sits
comfortably inside `3×8.841e-04=2.652e-03`, so item 1 PASSED on its own
freshly-measured envelope, not by reusing the original adjudication's
now-stale number. **This is the strongest evidence yet that the
baseline-relative re-pin is the right fix, not merely a one-time patch
over an unlucky measurement:** the rule was designed to accommodate a
*class* of run-to-run variation, and this session independently
re-measured a different-but-comparable instance of exactly that class
and still passed cleanly.

### 15.25.3 GPU-h and ledger update

**This launch's own realized cost, re-verified from box timestamps, not
merely the commit message:** chain wall-clock `06:25:46.60`Z →
`06:55:28.94`Z = 1,782.3s (≈0.4951 GPU-h on the single GPU-2 slot),
consistent with the disclosed **`0.487` GPU-h chain figure** (small
variance from exactly which log line is used as the chain's own
start/end marker — both land within ~2% of each other, not a
discrepancy worth chasing further). Adding the disclosed **`≈0.33`
GPU-h pre-launch verification** (3 Stage −1 suite re-runs +
the OFF-vs-OFF diagnostic, all CPU/GPU-idle-adjacent smoke work before
the real cell launched): **realized total ≈ 0.82 GPU-h**, matching the
commit message's own `0.487 + 0.33 ≈ 0.82` arithmetic. **Contingency
seeds 1943/1944 were NOT fired** (Step −1 cleared at 36≥3 events on the
primary launch alone) — the `KEYANCHOR_SCALING_CONTINGENCY_SEEDS_BY_D_K
[96][84]` reservation remains untouched, and the NO-REPRO contingency
cost bracket priced in §15.24.7 (+0.900 GPU-h worst case) was never
drawn on.

**KEY_ANCHORING_SCALING sub-ledger, updated (starting from §15.24's own
Rev 4/MAJOR-2-corrected baseline, `18.5466/26` GPU-h realized):**

| Item | GPU-h | Running total |
|---|---|---|
| Baseline (§15.24.7, Rev 4 corrected: 11.7865 + 6.3331 + 0.427) | — | 18.5466/26 |
| This C17 repro launch (0.487 chain + 0.33 verification) | +0.820 | **19.3666/26** |

**19.3666/21 = 92.22%** against the ORIGINAL, non-extended ceiling
(reserve **1.6334 GPU-h**) — **still fits without the `+5.0 GPU-h`
extension** (`KEYANCHOR_SCALING_EXT_PI_SIGNOFF`, authorized since
§15.22 but never yet drawn on). Against the extended 26 GPU-h ceiling:
**74.49%, reserve 6.6334 GPU-h.** This is the design's own realized cost
landing almost exactly on its 1× point estimate (`0.450` design estimate
vs. `0.487` chain-only realized, +8.2% — well inside this program's own
historical realized/estimate scatter) — no cost-model correction is
warranted from this one data point.

### 15.25.4 THE UNLOCK — mechanism adjudication (disclosed choice, per
task instruction: implement the conservative documented reading)

**§15.24.6's own registered text for outcome (c), quoted exactly (the
ONLY place in the design that states the unlock mechanics):** *"If (c)
TOLERANCE-MISCALIBRATION: a cheap, surgical fix (bump `geo3_resid_tol`
for raw/un-blended queries specifically, or raise `n_iter` modestly)
unlocks the ALREADY-COLLECTED 11 inadmissible cells' own `h4` values for
the fit without any new GPU spend — the fastest path back to attempting
§15.20.4's discrimination test, since the raw data already exists and
only the ADMISSION gate, not the training, would need re-scoring."*

**Adjudication: PURE RE-READ, with the mechanism disclosed explicitly
below — NOT a literal per-cell Newton-Schulz re-run.** Two readings are
possible and the design text alone does not disambiguate the exact
implementation, so both are named here, per the task's own instruction:

1. **Literal per-cell recomputation** (re-run the `n_iter∈{24,28,32,40}`
   sweep on EACH of the 11 cells' own dumped `k_eff_raw`, exactly as the
   repro did for K=84/seed=1940) — **structurally IMPOSSIBLE without new
   GPU spend**, contradicting §15.24.6's own "without any new GPU spend"
   text. Verified directly: the ORIGINAL wide-grid wave's checkpoint
   writer (`run_deltanet_rd.py:926–947`, §15.22's own citation, §15.23's
   own "why the true failing object could not be tested" paragraph)
   saves ONLY the 27KB anchor-table block for these 11 cells — never the
   full model needed to reconstruct `k_eff_raw` for a real C17 query.
   Only ONE cell in the entire program (K=84/seed=1940, the repro cell
   itself — chosen from, and one of, the 11) was ever re-launched with
   the new `--c17-repro-telemetry`/`--full-ckpt-step` hooks that make
   per-episode recomputation possible. This reading is REJECTED as
   inconsistent with the design's own stated cost claim.
2. **Inferential re-scoring of the admission FLAG alone (adopted, the
   conservative documented reading):** the h4 measurements themselves
   were never in question — training was clean at all 11 cells
   (`n_geo3_fallback_train_steps=0`, `task_performance_floor_pass=True`,
   `finite_loss_no_divergence=True`, `value_salvage_tier_pass=True`
   verified directly against each cell's own raw JSON this session, not
   merely §15.22's table); the ONLY failing admission leg at all 11 was
   `ns_converged_no_fallback`, driven by `checkpoint_fallback_seen=True`.
   §15.23's own independent mechanism diagnostic (already landed, not
   re-litigated here) established, ACROSS 4 different failing K's
   (72/78/84/90, one directly-pulled checkpoint each) and both passing
   controls, that this class of fallback is 100% `C17_heldout_entities`-
   exclusive and architecturally anchor-bypassed — i.e. the SAME
   structural mechanism, not a per-cell idiosyncrasy. The K=84/seed=1940
   repro — itself one of the 11 quarantined cells, not an external
   proxy — then closed the remaining question (is this class fixable by
   iteration count, or a real wall?) at the FULL episode level: **all
   4,608 examined episodes, spanning the worst-case late-checkpoint
   population (residuals up to 1.43), resolve by n_iter≤28.** Given (a)
   the identical, verified-per-cell failure signature across all 11
   quarantined cells, and (b) a representative member of that exact set
   showing universal resolution under a modest iteration bump, the
   admission flag — never the h4 measurement — is treated as
   miscalibrated for these 11 cells, and is flipped `False→True` offline.

**Scope, exactly 11 cells, disclosed exclusion:** `K72/{s1740,s1742}`,
`K78/{s1840,s1841,s1842}`, `K84/{s1940,s1941,s1942}`,
`K90/{s2040,s2041,s2042}`. **`K69/seed=1730` is DELIBERATELY NOT
flipped**, even though its own raw JSON shows the IDENTICAL signature
(`n_geo3_fallback_train_steps=0`, `checkpoint_fallback_seen=True` as the
sole failing leg, C17-exclusive per §15.23's own mechanism_breakdown
table) — this is a disclosed scope decision, not an oversight: s1730 is
the pre-existing, §15.19-era anomaly (the FIRST hint of this pattern,
predating both the wide-grid wave and the repro instrument), not one of
the "11 quarantined d=96 cells" §15.22 named (that count is exactly the
12 NEW wide-grid cells at K∈{72,78,84,90} minus the 1 already-admissible
K=72/seed=1741), and the task's own K-grid spec pins K=69 at `n≈3`
(1731/1732/1733), not `n=4`. Had s1730 also been flipped, K=69 would
carry a 4th seed and its own group mean would shift from 0.9592 to
0.9423 — a small, disclosed, non-load-bearing difference (verified by
hand, not applied) that would not change any conclusion below.

**Mechanical execution, verified byte-clean.** A one-off offline script
(archived below) built a RECALIBRATED copy of the wide-grid cells'
raw JSONs (originals at `experiment-runs/2026-07-07_keyanchor_scaling_
wide/` untouched, append-only discipline preserved), flipping
`geo3_admission.admissible: False→True` for exactly the 11 cells above,
after first ASSERTING each one's pre-flip signature matches the
diagnosed pattern exactly (refusing to flip blindly). A post-hoc
byte-diff confirms, for all 11 flipped files, that `admissible` (+ one
new disclosure-note key) is the ONLY field that differs from the
archived source — h4, `wall_s`, `checkpoints`, everything else is
byte-identical.

### 15.25.5 Re-fit — `fit_cliff_curve.py`, full d=96 K-grid
`{69,72,78,84,90}`, n=3 seeds/K, n_trials=4000

**d=80 anchors, scope note:** `fit_cliff_curve.py`'s own
`ANCHORED_D_STATES=(64,)` — verified directly in the script (line 85) —
means NO d, other than d=64, may pass flanking-anchor data into a
`curve_fit` call; passing `--k32-dir`/`--k48-dir` at `--d-state 96` is a
hard `assert`-refusal (lines 308–313). **"The d=80 anchors" is therefore
read as the ALREADY-EXISTING x0(80) fit (§15.22: `x0=0.6779`, CI
`[0.6683,0.6867]`) serving as a calibration anchor for the RIVAL BANDS'
own §15.20.4 formulas (§15.25.6/15.25.7 below), not as literal
within-fit data for the d=96 sigmoid** — the script structurally forbids
the latter reading, and nothing in §15.20/§15.24 registers extending
`ANCHORED_D_STATES`.

**Per-K means, all n=3, all admissible after the unlock (independently
re-verified via `fit_cliff_curve.load_k_mean_h4` directly, not just
read from the output JSON):**

| K | K/d | n | Per-seed h4 | Mean h4 |
|---|---|---|---|---|
| 69 | 0.71875 | 3 | 0.9986, 0.9614, 0.9175 | 0.9592 |
| 72 | 0.75000 | 3 | 0.9319, 0.8426, 0.9904 | 0.9216 |
| 78 | 0.81250 | 3 | 0.8667, 0.9823, 0.9488 | 0.9326 |
| 84 | 0.87500 | 3 | 0.9806, 0.9573, 0.9363 | 0.9581 |
| 90 | 0.93750 | 3 | 1.0000, 1.0000, 1.0000 | 1.0000 |

**The curve is NON-MONOTONIC** — it dips at K72/K78, partially recovers
at K84, and returns to an EXACT ceiling (1.0000, all 3 seeds) at K90,
the widest K tested. This is not a data-entry artifact: independently
re-verified against each raw JSON's own `checkpoints[-1]["M3_held_out"]
["4"]["recovered_frac@0.9"]` field directly.

**Fit result — degenerate at the parameter bounds:**

```
sigmoid fit: x0=0.9000 w=0.1281 L=1.2000 rss=0.3456   (x0 AND L both pinned exactly at their own upper bound)
linear fit (disclosed, non-load-bearing): rss=0.001893
bootstrap: n_trials=4000, n_degenerate=4000, degenerate_frac=1.0000  (100%, EXCEEDS the 10% bar)
bootstrap CI(x0): NONE — "too few valid fits to report a CI"
```

The MAIN (non-bootstrap) fit itself already sits at both of its own
upper bounds (`x0=0.9`, `L=1.2` — the `curve_fit` bounds are
`([0.5,0.3,0.005],[1.2,0.9,0.5])`), the signature of an optimizer being
asked to fit a monotonically-declining sigmoid to a curve that is NOT
monotonically declining in this window — every one of 4,000 bootstrap
resamples independently lands in the same degenerate regime.

**This registered fit SUPERSEDES §15.22's own diagnostic-only 6-K fit**
(`x0=0.7716` `[0.7700,0.7841]`, 26.2% degenerate, wrong K-grid, K72 at
n=1) — that fit's own "something starts declining right around
K/d≈0.75" suggestion does not survive contact with the full, unlocked
data: K78/K84/K90 (unavailable to that earlier diagnostic) show the
curve recovering, not continuing to decline, and K90 in particular hits
an exact ceiling.

### 15.25.6 §15.20 Rev 1's 6-row decision rule, applied mechanically —
FINAL VERDICT: **AMBIGUOUS**

| Step | Trigger (quoted verbatim, §15.20.4) | Evaluated against this fit | Fires? |
|---|---|---|---|
| 0 | "Fit converges, `degenerate_frac ≤ 10%`" | `degenerate_frac=1.0000` (100%) | **NO** — does not clear step 0; branch to 1a/1b |
| 1a | "`degenerate_frac > 10%` AND every sampled K's own raw per-seed h4 mean is `≥ 0.98`" | Per-K means: 0.9592, 0.9216, 0.9326, 0.9581, 1.0000 — **4 of 5 K's sit below 0.98** (only K90 clears it) | **NO** |
| 1b | "`degenerate_frac > 10%` AND at least one sampled K's own raw per-seed h4 mean is `< 0.98` (genuine scatter/non-convergence, not flatness)" | TRUE — K69/K72/K78/K84 all `<0.98` | **YES → VERDICT: AMBIGUOUS.** Follow-up (quoted): "Seed escalation at the noisiest K-group" |
| 2–5 | Band-overlap / BOTH-CONSISTENT / NEITHER-SURVIVES tests | All four require a converged, non-degenerate `CI(x0,96-wide)`. **None exists** (bootstrap CI is `None`). | **Not reachable** |

**Noisiest K-group, computed for the registered follow-up (per-seed
range, widest first):** K72 (range `0.8426`–`0.9904` = `0.1478`) > K78
(`0.1156`) > K69 (`0.0811`) > K84 (`0.0443`) > K90 (`0.0000`). **K72 is
the noisiest K-group** — the registered next step, if this line is
pursued, is seed escalation there first.

**This is NEITHER of the two outcomes the pre-registered power check
(§15.20.4 MAJOR-2) flagged as most likely** (BOTH-CONSISTENT, if the CI
half-width exceeds 0.0145; or, less likely, CLIFF-BEYOND-WINDOW if every
K sits flat near ceiling) — **it is the THIRD registered branch, row
1b's own AMBIGUOUS, reached for a genuinely new reason.** §15.22's
original attempt landed at AMBIGUOUS because 3 of 5 K's had ZERO usable
data (the fit could not even be attempted). This attempt landed at
AMBIGUOUS with a FULL grid (5/5 K's, n=3 seeds each, the unlock's own
entire purpose achieved) because the measured curve itself is
non-monotonic — a different, more informative failure mode: the data
gap is closed, but the underlying signal at this seed budget is too
noisy (and, near K90, too flat-at-ceiling) for a monotonic-sigmoid model
to resolve. **§15.20.4's own central discrimination test (absolute-slack
vs. power-law) still does not execute** — this is the SECOND wave in a
row to reach that same non-outcome, for a different mechanical reason
each time (§15.22: no data; this harvest: data exists, curve shape
does not fit the assumed model).

### 15.25.7 The scaling-law reading

**x0 progression: `0.5455` (d=64) → `0.6779` (d=80) → x0(96) UNRESOLVED
(AMBIGUOUS, no point estimate or CI survives the degenerate bootstrap).**
The clean, two-point rise from d=64 to d=80 (a real, ~24% relative
increase, underlying both named rival extrapolations) does **not**
cleanly extend to d=96 at the currently-available seed budget — not
because of a renewed data-collection gap (the unlock closed that
cleanly, 5/5 K's now populated at n=3 each) but because the measured
h4(K/d) shape in the tested window (`K/d∈[0.719,0.938]`) is itself
non-monotonic: a real dip through K72–K78, partial recovery at K84, and
an exact ceiling at K90. **Neither rival band is confirmed or excluded**
— `[0.718,0.739]` (absolute-slack) and `[0.768,0.837]` (power-law) both
remain live, untested hypotheses at d=96, exactly as they were before
this harvest, just for a different reason now. **Disclosed, descriptive
(NOT licensed) read, mirroring §15.22's own house convention of flagging
suggestive-but-unlicensed patterns:** taken at face value, the curve
looks closer to "flat/near-ceiling with real noise" than to either
rival's own smooth monotone-decline shape in this window — a hint,
consistent with (but not proof of) the K69-anchored §15.19 finding that
no cliff was visible at K/d≤0.71875 either, that whatever transition
exists may sit at a wider K/d than this grid reaches, OR that n=3
seeds/K is simply too few to average out the real per-seed scatter
this table shows (K72's own per-seed span alone, 0.843–0.990, is wider
than the ENTIRE gap between the two rival bands, 0.718–0.837). **Both
readings point to the SAME registered next step** (row 1b's own
follow-up): seed escalation, starting at K72 (confirmed noisiest), before
any further attempt at §15.20.4's own discrimination test. Not launched
by this harvest (harvest-only discipline, `CLAUDE.md`).

### 15.25.8 Archive

`experiment-runs/2026-07-08_c17_repro/` (repo-tracked, all files ≤25MB):
the C17 repro instrument's own analysis JSON, replay JSON, Stage −1
results, item-1 OFF-vs-OFF diagnostic (already committed at
`matrix-thinking/deltanet_rd/results/keyanchor_scaling_c17repro/` per
commit `a51f102`, mirrored into the dated archive directory here for
the harvest's own convention); this harvest's own new artifacts — the
recalibrated-admissibility build script + its verified output table, the
unlocked-grid re-fit's own `fit_d96_unlocked_results.json` (full
`curve_points`/`sigmoid_fit`/`bootstrap_ci` payload). The 15 raw wide-grid
cell JSONs the unlock reads FROM remain solely at their existing archive
location, `experiment-runs/2026-07-07_keyanchor_scaling_wide/` (never
duplicated wholesale here — this harvest's own recalibrated copies are a
DERIVED analysis artifact, not a second copy of the primary archive).
Raw `.pt` dumps (`c17fallback_step*.pt` × 10, `full_step20000.pt`) stay
on box per archive policy (>25MB / not needed for reproducibility beyond
the already-committed analysis JSON). SSD mirror, byte-verified,
same tree: `/Volumes/1TB_SSD/learned-representations/experiment-runs/
2026-07-08_c17_repro/`.

---

## §15.26 DESIGN — d=96 SCATTER-RESOLUTION wave (Rev 2, 2026-07-09 — post-attack round 2)

**Rev 2 status note.** A second independent adversarial attack round
reviewed Rev 1 (landed 2026-07-08, §15.26.9's own RESHAPE-TO-C fix-map)
before any GPU work launched, per this program's own standing
multiple-independent-audit-rounds discipline (`CLAUDE.md`). Verdict:
**NEEDS-REVISION** — 0 FATAL, 3 MAJOR, 5 MINOR — every finding surgical
and individually prescribed, no category error. The empirical core was
independently re-verified this round and found exceptionally clean:
every cited number reproduces, and the 320,000-trial extension
underlying §15.26.1's own zero-GPU finding (280,000 across 7 new seeds
+ 40,000 from an independent reimplementation) was independently
RE-EXECUTED this round, not merely re-read, and reproduced exactly
(100.00% degenerate, both nulls, every seed). **MAJOR-1 (highest value,
noise-floor calibration):** §15.26.5's own outcome trigger compared
`M3_held_out_pool_restricted` against `M3_held_out` using TWO DIFFERENT
eval generators (`eval_gen`, offset `seed+10_000`, vs. `eval_gen2`,
offset `seed+20_000`) — `shift` therefore conflated the pool-restriction
treatment with plain eval-batch resampling noise, and the
CEILING-IS-REAL trigger (`shift ≤ 0.1×Δ`) had no measured null to be
judged against. Fixed by registering ONE additional eval-only pass in
the K=84 checkpoint block (§15.26.2.2) — the SAME unrestricted
`M3_held_out` call, repeated under `eval_gen2`'s own offset scheme (same
weights, same UNRESTRICTED pool — generator offset is the only thing
that differs from the standard call) — giving `noise_shift := |repeat −
standard|`, a directly measured eval-sampling null. All three outcome
triggers are re-pinned relative to it (§15.26.5), proven MECE by an
explicit totality walk. **MAJOR-2 (control the diagnosed variable):**
§15.26.2.1 itself pins the live mechanism as entity-draw OVERLAP
FRACTION `K/N`, not spare-entity margin `N−K` — but Rev 1's own
manipulation (`m3_pool_restrict_n=101`) matched MARGIN (`N'=101 →
83.17%` vs. K90's natural `84.11%`, a 0.94pp residual), not the
mechanism it was built to isolate. Fixed by re-pinning `N'=100`
(`84/100=84.00%` vs. `84.11%`, 0.11pp residual, ≈8.4× tighter on the
actually-diagnosed variable, same cost) — Rev 1's own margin-vs-overlap
slip disclosed explicitly in the fix-map (§15.26.10), not silently
absorbed. **MAJOR-3 (wrapper field-diff adaptation):** §15.26.3.1 names
the launch wrapper as mirroring `run_k69_s1733_contingency.py`'s own
precedent "line-for-line," but that precedent's own token-diff check
(refuse unless the generated command matches a sibling-seed reference
command with only seed-derived tokens differing) cannot pass verbatim
once the K=84 cell's own command carries the new
`--m3-pool-restrict-n` flag the reference command never has. Fixed by
pinning the adapted check explicitly: strip an explicitly-enumerated
whitelist of new-flag tokens from the generated command BEFORE running
the precedent's own diff, so it still refuses on any OTHER,
non-whitelisted divergence — plus its own registered negative test (a
command carrying one extra, non-whitelisted flag must still be
refused). Five MINOR findings (telemetry-threading consistency on the
new eval calls, a `:961`→`:963-964` citation fix, a pre-registered
Δ_measured contingency, a finding-text reword, and a ledger
rounding-consistency fix on the 2× bracket) are folded in directly at
their own locations, none deferred. The full finding→fix trace is
recorded at §15.26.10, per house style (mirrors §15.26.9's own
convention). **This revision has not yet had its own verification
pass** — round 3, a VERIFY pass confirming Rev 2's fixes land clean (not
a fresh full attack round), next, per this project's standing rule.

**Status: DESIGN-ONLY DRAFT (Rev 2).** RESHAPE-TO-C — Rev 1's own central
move, unchanged and not reopened by this revision — means: neither
straight NEEDS-REVISION-and-relaunch-the-same-10-cell-design, nor a pure
kill — the wave's central empirical ambition (spend 4.27–8.54 GPU-h
escalating `n=3→5` at all 5 K-groups to see whether the d=96 h4(K/d)
sigmoid fit de-degenerates) is retired because its own pre-registered
power check already answers the question it was funded to ask, at zero
GPU cost; the wave's INFORMATION VALUE is preserved by registering that
answer directly (§15.26.1); and a differently shaped, much smaller (2
cells, ≈0.9 GPU-h), sharper instrument — a control diagnostic targeting
Rev 0's own single most important disclosed loose end, the K90
pool-margin confound — is substituted in the freed budget (§15.26.2).
Rev 2 re-calibrates that diagnostic's own outcome trigger,
mechanism-matching, and launch-safety machinery before any GPU cell
fires; it does not reopen the retirement decision itself. Every number
below is either cited to an already-run artifact (§15.22's realized
per-cell `wall_s`, §15.25's per-K per-seed h4 table, the 320,000-trial
extended power check and independent reimplementation, §15.26.1) or
freshly re-derived/verified this session directly against the live
`grammar_rd.py`/`run_deltanet_rd.py` source and the raw archived JSONs
(§15.26.2.1) — never carried over by assumption. No GPU cell has
launched under this design; Rev 2 is still design-only.

### 15.26.0 What this wave is, and what it explicitly does NOT reopen

§15.25.6's own mechanical verdict on the full, unlocked d=96 grid
(`K∈{69,72,78,84,90}`, n=3 seeds/K) is **AMBIGUOUS** — not from a data
gap (that closed cleanly, §15.25.4's unlock) but from genuine,
non-monotonic scatter: `sigmoid_fit` pins at both its own upper bounds
(`x0=0.9, L=1.2`), bootstrap `degenerate_frac=1.0000`. **Rev 0 proposed
extending §15.20 Rev 1's own row-1b follow-up ("seed escalation at the
noisiest K-group") from "K72 alone" to all 5 K-groups (+2 seeds/K,
n=3→5, 10 cells) to tighten that verdict. The attack round found this
premise self-defeating: Rev 0's OWN pre-registered power check
(§15.26.1) already shows, at zero GPU cost, that this tightening cannot
plausibly change the verdict's direction — so Rev 1 registers the AMBIGUOUS
finding directly instead of spending 4.27–8.54 GPU-h to re-confirm it,
and redirects the freed budget to a small, sharper instrument (the K90
pool-margin control diagnostic, §15.26.2) targeting Rev 0's own most
important disclosed loose end.**

**Explicitly NOT reopened, per this program's own "never reopens or
rescores" precedent (§15.1/§15.19/§15.22/§15.24.0):**

1. §15.25.4's own admission-flag unlock (11 cells, `False→True`) —
   unchanged, not re-litigated.
2. §15.23's own anchor-table finding (NS at n_iter=20 has ~7,000×
   margin on the anchor table itself) — closed, not re-tested.
3. §15.24's own C17 mechanism diagnosis (the failure is 100%
   `C17_heldout_entities`-exclusive, architecturally anchor-bypassed) —
   closed, not re-tested. This diagnostic's own 2 cells (§15.26.2) will
   incidentally exercise this gate again at K∈{84,90} under the
   corrected `n_iter=28` production setting (§15.26.3), disclosed as an
   open empirical question its own launch answers, not assumed.
4. **Build-scope fence, restated for this diagnostic's own (much
   smaller) build surface, Rev 2 update — now two additive eval calls,
   not one; round-3 VERIFY-pass adopted fix m-b brought this to THREE,
   reconciled here at BUILD time (disclosed, not silently absorbed):**
   the killed 10-cell grid's own registered build tasks
   (`run_deltanet_rd_exactness_sweep.py` additive override + spec-builder
   function) are retired unbuilt with it. The diagnostic's OWN build
   tasks: one new optional, additive-only parameter on `evaluate_pool`
   (`restrict_entity_pool_n`, §15.26.2.2), THREE new additive eval calls
   in `train()`'s checkpoint block gated by the SAME new
   `m3_pool_restrict_n` parameter (`m3_restricted`, Rev 1; `m3_noise_
   repeat`, Rev 2's own MAJOR-1 fix; `m3_noise_repeat_2`, round-3's own
   adopted fix m-b, a second independent noise-floor draw at a third
   generator offset — same file, same additive discipline), a whitelist-
   adapted field-diff check inside the launch wrapper (Rev 2's own
   MAJOR-3 fix, §15.26.3.1), one new standalone launch wrapper
   (`run_poolmargin_k84s1943_k90s2043.py`, §15.26.3.1), and one new small
   harvest helper (`harvest_poolmargin_k84s1943_k90s2043.py`, computing
   `shift`/`noise_shift`/`Δ_measured` and applying §15.26.5's own outcome
   trigger from both cells' result JSONs — build-task item 4 of the
   round-3 VERIFY pass's own 8-task enumeration, not separately named in
   Rev 2's own text above, reconciled here) — plus the two CPU-only
   analysis scripts already written and run in prior sessions (§15.26.1,
   Rev 0's own original + Rev 1's own extended verification; Rev 2 added
   no new run artifacts, a pure design revision). No `phase2_*` file, no
   `lm_pretrain_rd.py`, no change to the EXISTING
   `KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K` dict's own entries (preserves
   the ORIGINAL/wide-grid manifest-regression invariant, §15.20.1 Stage-1
   item 5).

### 15.26.1 Part 1 — the registered finding (zero GPU)

**The finding, registered directly from the existing n=3 data + a now
multiply-independently-confirmed power analysis, no new GPU cell
required:**

> No cliff to K/d=0.9375; h4 is seed-dependent in the sub-ceiling regime
> (K72–K84) and non-sigmoid in this window; K90 is pinned at exact
> ceiling in all 3 seeds; x0(96) is unlocalizable with this instrument.

(Rev 2 MINOR-4 fix — reworded for precision: the per-K table just below
shows K90's own `sample sd=0.0000` across all 3 real seeds, an EXACT
ceiling, not merely "near" it; "seed-dependent" describes the
sub-ceiling K-groups specifically, K72–K84, not K90.)

This is the SAME published statement §15.25.6's own mechanical verdict
already reached (AMBIGUOUS, `degenerate_frac=1.0000`, `sigmoid_fit`
pinned at both its own upper bounds) — Rev 1 does not change the
verdict, it changes what's SPENT to stand behind it: zero GPU, versus
the killed grid's own 4.27–8.54 GPU-h.

**Power justification, part (i) — REUSED unchanged from Rev 0, still the
foundation:** the existing per-K stats (independently re-verified
directly against the raw JSONs, `checkpoints[-1]['M3_held_out']['4']
['recovered_frac@0.9']`, matching §15.25.5's own table to 4 decimal
places):

| K | K/d | n | mean h4 | sample sd | SE(n=3) | SE(n=5, projected) |
|---|---|---|---|---|---|---|
| 69 | 0.71875 | 3 | 0.9592 | 0.0406 | 0.0234 | 0.0182 |
| 72 | 0.75000 | 3 | 0.9216 | 0.0745 | 0.0430 | 0.0333 |
| 78 | 0.81250 | 3 | 0.9326 | 0.0595 | 0.0343 | 0.0266 |
| 84 | 0.87500 | 3 | 0.9581 | 0.0222 | 0.0128 | 0.0099 |
| 90 | 0.93750 | 3 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

`SE(n=3)/SE(n=5) = sqrt(5/3) = 1.2910` exactly. A purpose-built,
CPU-only power-check script (`sim_d96_scatter_resolution_power.py`,
imports `sigmoid`/bounds from `sim_cliff_power.py` unmodified) was
written and RUN in the Rev 0 session — 20,000 trials/null, seed
20260708, two pre-registered nulls (H0 "SCATTER-IS-REAL": 2 new draws/K
~ `N(K's own observed mean, sd)`; H1 "DIP-IS-NOISE": K69/K84/K90 held at
their own observed means, K72/K78's assumed true mean replaced by the
K69→K84 linear interpolation) — archived at
`experiment-runs/2026-07-08_d96_scatter_resolution_design/fits/
sim_d96_scatter_resolution_power_results.json`:

| Null | degenerate_frac | monotonic_frac |
|---|---|---|
| H0 (scatter is real) | **1.0000** | **0.0000** |
| H1 (dip is noise) | **1.0000** | **0.0000** |

**Zero out of 40,000 combined trials produced a non-degenerate,
monotonic 5-point fit — under EITHER null.**

**Power justification, part (ii) — NEW this Rev 1 session, independently
confirmed at much higher trial counts, multiple seeds, an independent
reimplementation, and a positive control (MAJOR-1 fix folds this in;
CLAUDE.md's own "verify before claiming" applied to the attack round's
own reported additional confirmation, which had no archived artifact —
so it was re-run fresh here rather than cited on trust):**
`sim_d96_scatter_resolution_power_extended.py` (registered,
RUN THIS SESSION, archived at `experiment-runs/2026-07-08_
d96_scatter_resolution_design/{scripts,fits}/`, mirrored at
`matrix-thinking/deltanet_rd/`), three independent pieces:

1. **Extended multi-seed confirmation:** the SAME audited driver
   (`sim_d96_scatter_resolution_power.py`, imported unmodified),
   re-run at **7 NEW seeds** (20260801–20260807, distinct from the
   archived 20260708) × 20,000 trials/null = **280,000 additional
   trials**. Result: **all 7 seeds, both nulls, 100.00% degenerate**
   (`combined_degenerate_frac=1.000000`).
2. **From-scratch reimplementation:** an independently re-typed sigmoid
   form + `curve_fit` call + degeneracy check (`sigmoid_reimpl`/
   `fit_and_check_reimpl`, never importing `sigmoid()` from
   `sim_cliff_power.py` or `fit_sigmoid_once()` from the original
   driver — rules out a bug shared across all three scripts that
   otherwise import the same function), one seed (20260810), 20,000
   trials/null = **40,000 additional trials**. Result: **100.00%
   degenerate, 0.00% monotonic, both nulls** — matches the shared-code
   result exactly under independent implementation.
3. **Positive control (proves the detector has teeth, CLAUDE.md's own
   "assert has teeth" discipline applied to this diagnostic instrument,
   not just to `geo3_admission`):** a SYNTHETIC truth deliberately
   constructed to be non-degenerate — sigmoid centered at the abs-slack
   rival's own `x0=0.729667, w=0.0597`, LOW noise `sd=0.01` (well under
   this program's own much noisier real per-K sds) — fed through the
   SAME degeneracy check, 2,000 trials, seed 20260811. Result:
   **`degenerate_frac=0.0000`, `monotonic_frac=1.0000`,
   `abs_slack_band_hit_frac=0.3250`** — the check correctly reports
   non-degenerate, monotonic fits, and correctly localizes `x0` inside
   the rival band it was centered on nearly a third of the time, when
   fed a genuinely non-degenerate truth. The check is not vacuously
   always-degenerate.

**Cumulative total across this wave's own history: 40,000 (Rev 0,
archived) + 280,000 (7 new seeds, this session) + 40,000 (independent
reimplementation, this session) = 360,000 trials, 100% degenerate under
both nulls in every single one, across 8 distinct seeds and 2
independent implementations, plus a 2,000-trial positive control proving
the instrument itself is not the reason.**

**MAJOR-1 fix — the "isolated K84-vs-K90 sub-check, 200,000 trials"
claim, replaced with the explicit analytic derivation it actually is,
plus disclosure of Rev 0's own misattribution:** Rev 0's text asserted
this driver-vs-K90 comparison came from "diagnosed directly (isolated
K84-vs-K90 sub-check, same script, 200,000 trials)." **Verified directly
against `sim_d96_scatter_resolution_power.py`: no such isolated
sub-check function or CLI path exists in that script, and its own single
archived run used `n_trials=20,000` per null (40,000 total, matching the
table above) — no 200,000-trial run was ever coded, executed, or
archived anywhere in this program's history.** This was a misattribution
in Rev 0, disclosed here rather than silently carried forward. The
underlying claim is TRUE, and is exactly what an ANALYTIC derivation
gives, exactly, with no simulation needed (K84's mean-of-5 is a location-
scale transform of 2 Gaussian draws plus 3 fixed constants, hence itself
exactly Gaussian):

For `n_fixed=3` known, FIXED archived values plus `n_new` new draws
`~N(μ,σ²)` (μ = that K's own observed mean, σ = its own observed sample
sd), the resulting mean-of-`(n_fixed+n_new)` is itself Gaussian with
`SE = σ·√(n_new) / (n_fixed+n_new)` — exact, not approximate. Using
K=84's own re-verified `σ=0.022157` (matches the table's rounded
`0.0222`) and `μ=0.958093` (matches the table's rounded `0.9581`):

- **`n_new=2` (the mean-of-5 this wave's own escalation would have
  produced): `SE = 0.022157·√2/5 ≈ 0.00627`.** Gap to K90's real,
  fixed 1.0000 ceiling: `1.0000−0.958093 = 0.04191`. **`z = 0.04191 /
  0.00627 ≈ 6.687`.**
- **`n_new=100` (projected to N=103, 100 EXTRA seeds at K84 alone,
  holding K90 fixed): `SE = 0.022157·√100/103 ≈ 0.002151`.** **`z =
  0.04191 / 0.002151 ≈ 19.5`.**

Both figures match the task brief's own cited numbers to 3 significant
digits, independently re-derived here from the raw archived seed values
directly (not copied from the brief), confirming K84's mean-of-5 (or
mean-of-103) reaching K90's exact 1.0000 ceiling is a many-SE event under
the observed population parameters — the SAME conclusion the (now
misattributed-and-replaced) 200,000-trial claim asserted, now resting on
an exact closed form instead of an unarchived, never-run empirical claim.

**10-cell seed grid: KILLED. Reasoning — its own pre-registered power
analysis:** the grid's entire purpose was tightening each K-group's SE
enough that `fit_cliff_curve.py`'s sigmoid fit might de-degenerate. The
power check above (100,000+% overkill on trial count, 8 seeds, 2
implementations, a positive control proving the check has teeth) already
answers that question with near-certainty: resolution is analytically
disfavored under both tested nulls, driven by a specific, tightly-measured,
low-noise structural fact (K84's mean sits ≈6.7–19.5 SEs below K90's
exact ceiling depending on seed count) that 2, or even 100, more seeds at
K84 cannot plausibly overturn. Spending 4.27–8.54 GPU-h (§15.26.4's own
historical table) to empirically re-confirm a near-certainty already
established analytically, at zero GPU cost, is not justified — per this
program's own compute-discipline (Pre-Experiment Checklist item 3, "try
to disprove it in 5 minutes" — here, in 34.6s of CPU time for the
original run and under 5 minutes for this session's own 320,000-trial
extension). **The wave's own registered success criterion never required
a NEW sigmoid fit** ("success = a materially tighter... mechanically
re-applied verdict, NOT resolve x0(96)," Rev 0's own text) — but since
the DIRECTION of that verdict is already known with overwhelming
confidence without spending the GPU-h, registering the finding directly
is the correct, GPU-efficient conclusion, not a shortcut around the
wave's own discipline.

**Disposition of the killed grid's own 10 reserved seeds:** 1734, 1736
(K69), 1743, 1744 (K72), 1843, 1844 (K78), 1943, 1944 (K84), 2043, 2044
(K90) are all released from this purpose, none fired. Two (1943 @ K84,
2043 @ K90) are REDIRECTED, disclosed, to the new K90 pool-margin
control diagnostic (§15.26.2.3) — the correct disclosed re-use of an
already-idle reservation, the same discipline §15.26.2.3 (formerly
§15.26.2's own "disclosed re-use, not double-booking" paragraph) already
established for this exact pair once before. The remaining 8 stay
unclaimed, available for any future need.

### 15.26.2 Part 2 — the K90 POOL-MARGIN CONTROL DIAGNOSTIC

**What this replaces:** the killed 10-cell grid's own registered "not
built, out of scope" candidate mechanism (§15.26.1 above, carried
forward from Rev 0 §15.26.0's disclosed paragraph and §15.24.2's
original flag) — this wave's own single most important loose end,
finally given its own instrument instead of staying a named-but-untested
footnote.

#### 15.26.2.1 Mechanism pin — read directly against `grammar_rd.py` +
`run_deltanet_rd.py` this session, not assumed by analogy to §15.24.2

**Correction 1 (a real, previously-unnoticed citation-transplant error
in Rev 0, found by actually reading the code this session — not an
attack-round finding, disclosed here on its own merits): the metric
`fit_cliff_curve.py` actually fits (`h4` = `M3_held_out`) does NOT draw
from the pool Rev 0's own confound paragraph cited.** Verified directly:
`run_deltanet_rd.py`'s own `train()` calls `evaluate_pool` four times
per checkpoint — `m2 = evaluate_pool(..., pools, ...)` (no
`use_heldout_entities` flag), `m3 = evaluate_pool(..., pools,
force_rank_k=..., c17_repro_telemetry=...)` (SAME, no flag), `c17 =
evaluate_pool(..., pools, ..., use_heldout_entities=True, ...)` (ONLY
this third call sets the flag), `c19 = evaluate_pool(..., pools, ...,
use_heldout_template=True)`. Inside `evaluate_pool`,
`sample_batch_rd`'s own pool selection (`grammar_rd.py:423`) is `name_ids
= pools.heldout_name_ids if use_heldout_entities else
pools.train_name_ids`. **`M3_held_out` (the `h4` metric) therefore draws
from `pools.train_name_ids` (N=107, verified against every archived
`pool_report` this session — e.g.
`.../wkeyanchor-scaling_rdx_K90_armd_s2041_.../d96.json`:
`n_train_names=107, n_heldout_names=106`), never from
`pools.heldout_name_ids` (N=106) — "held-out" in `M3_held_out` names
held-out HOPS (`H_test+H_extra`, `run_deltanet_rd.py`'s own docstring
line 12: "M3_held_out (train entities+template, H_test+H_extra)"), not
held-out ENTITIES.** Rev 0's own disclosed confound paragraph (old
§15.26.0) cited "16 of 106 held-out entities... vs 22 of 106 at K=84" —
those exact figures are real and correctly sourced from §15.24.2's own
original flag, but §15.24.2 was diagnosing `C17_heldout_entities`
specifically (§15.24 is the C17 eval-admission repro instrument's own
section), and Rev 0 carried the citation over to explain a ceiling in a
DIFFERENT metric (`M3_held_out`) that provably does not read that pool
at all. **Corrected margins for the metric actually being fit:**
`pools.train_name_ids`, N=107 — K=84 → 107−84=**23** spare, K=90 →
107−90=**17** spare (not 22/16 against N=106).

**Correction 2 (verified directly against the archived raw JSONs this
session, not assumed): `C17_heldout_entities` — the metric that DOES
read the N=106 pool Rev 0 actually cited — is NOT at ceiling at K=90.**
`checkpoints[-1].C17_heldout_entities.<H_train[0]>.recovered_frac@0.9`
across the 3 real K=90 seeds: **0.9917 (s2040), 0.8446 (s2041), 0.7278
(s2042)** — MORE variable, and on 2/3 seeds materially LOWER, than the 3
real K=84 seeds' own C17 values (0.9544, 0.9492, 0.9765). This is the
OPPOSITE of "near-exhaustion of the held-out pool makes the held-out-entity
recovery task measurably easier" for that mechanism's own literal
referent — a fresh, session-verified correction to Rev 0's own mechanism
claim, disclosed rather than silently absorbed into Rev 1.

**Correction 3 (scoring mechanism, read directly, not assumed): a
"fewer confusable distractors" channel does not exist in this readout.**
`evaluate_pool` computes `recovered_frac@0.9` from `cos_all =
F.cosine_similarity(pred, targets, dim=-1)` (`run_deltanet_rd.py`, inside
the per-batch eval loop) — a direct, CONTINUOUS cosine-similarity
comparison against the exact true target vector, thresholded at 0.9,
never a nearest-neighbor/argmax decode against a candidate/distractor
set (consistent with this project's own standing rule against
codebook-style readouts). A thinner pool margin cannot make recovery
"easier" via fewer distractors to confuse with, because there is no
distractor-comparison step in the scoring path to make easier.

**The mechanistically live channel, pinned from code: entity-draw
DIVERSITY, not decode confusability.** `sample_batch_rd` draws K of N
entities per episode via `pool_idx = scores.argsort(dim=-1)[:, :K]`
(`grammar_rd.py:434–436`) — uniform K-subsets of `name_ids` without
replacement, independently per episode/row. For two independent
K-subsets drawn uniformly from the same N-item pool,
`E[|A∩B|] = K²/N` exactly (each of the N items is in both draws with
probability `(K/N)²` by independence), so the expected overlap AS A
FRACTION OF EACH SET'S OWN SIZE is `E[|A∩B|]/K = K/N` exactly. On the
TRAIN pool (N=107) that actually feeds `M3_held_out`: K=90 → expected
inter-episode membership overlap **90/107 = 84.11%**; K=84 → **84/107 =
78.50%**. A real, quantifiable, code-derived difference — at K=90 both
TRAINING (`run_deltanet_rd.py:843`, `sample_batch_rd(..., pools=pools,
...)`, same unrestricted `pools`) and EVAL (the `m3` call above) draw
from a narrower effective diversity of entity subsets than at K=84,
which could produce apparent-ceiling recovery through reduced
generalization DEMAND (episodes look more like each other, both during
training and at eval) rather than genuine composition capacity at
K/d=0.9375. **This is the corrected candidate mechanism this diagnostic
targets — not the "fewer confusable distractors" framing Rev 0's own
citation-transplant implied, and not the N=106 heldout pool Rev 0
literally cited.**

#### 15.26.2.2 Manipulation — implementable WITHOUT touching the
training path

Training draws (`run_deltanet_rd.py:843`) and the `m3` eval call
(`:963-964` — Rev 2 MINOR-2 fix; Rev 0's own citation numbering said
`:961`, which is actually the PRECEDING `m2` call, verified directly
against the live file this session: `m3 = evaluate_pool(...)` itself
spans lines 963-964) currently share the SAME unrestricted `pools`
object (`train_name_ids`, N=107). The cleanest registered manipulation
restricts ONLY the eval-time pool for the `m3` call, leaving the
training-time draw (and therefore the trained WEIGHTS) exactly what
standard production K=84 training already produces — a genuinely
additive, eval-only intervention, not a new training regime:

```python
# sec 15.26.2.2, registered, NOT YET BUILT — one new optional,
# additive-only parameter on evaluate_pool (mirrors sec 15.24.2's own
# "FOUR new optional, additive parameters, all zero-behavioral-change at
# their defaults" precedent, applied once more):
def evaluate_pool(..., restrict_entity_pool_n: int | None = None, ...):
    if restrict_entity_pool_n is not None:
        field = "heldout_name_ids" if use_heldout_entities else "train_name_ids"
        src = getattr(pools, field)
        assert restrict_entity_pool_n <= src.numel()
        # pools.train_name_ids' own order is FIXED (build_entity_pools is
        # always called with seed=0 in production, run_deltanet_rd.py:1228/
        # :1672, verified this session) -- slicing the first N is a
        # deterministic, zero-new-randomness restriction, reproducible
        # across cells without a new RNG draw.
        pools = dataclasses.replace(pools, **{field: src[:restrict_entity_pool_n]})
    # ... existing body unchanged, now reads the (possibly restricted) pools
```

At `train()`'s own checkpoint block (Rev 0's own `:963-964` call site,
per the MINOR-2 fix above), TWO new, additional, optional eval passes,
both gated by the same new `m3_pool_restrict_n` parameter threaded from
the launch wrapper (§15.26.3.1), default `None` (byte-identical to
today):

```python
# registered, NOT YET BUILT — additive 5th AND 6th eval calls per
# checkpoint, only when m3_pool_restrict_n is set (K=84's own cell only
# — K=90 never sets this param, so K=90's checkpoint block is byte-
# identical to the pre-Rev-1 4-call baseline). Stored under NEW keys,
# NEVER overwriting the existing M3_held_out (preserves the manifest-
# regression invariant for every wave/cell that never passes this
# param). Rev 2 MINOR-1 fix: all new calls thread
# c17_repro_telemetry=c17_repro_telemetry, matching the docstring's own
# "SEPARATE flag threaded to ALL FOUR pool calls" invariant
# (run_deltanet_rd.py:354-355) — round-3's own adopted fix m-b (second
# noise-repeat draw) brings this to a 7-call invariant for K=84's own
# cell (m2, m3, c17, c19, m3_restricted, m3_noise_repeat,
# m3_noise_repeat_2), 4-call unchanged for K=90's.
if m3_pool_restrict_n is not None:
    eval_gen2 = torch.Generator(device=device).manual_seed(seed + 20_000 + step)
    m3_restricted = evaluate_pool(model, cfg, eval_gen2, device,
                                   (*cfg.H_test, *cfg.H_extra), pools,
                                   force_rank_k=force_rank_k,
                                   restrict_entity_pool_n=m3_pool_restrict_n,
                                   c17_repro_telemetry=c17_repro_telemetry)
    res["M3_held_out_pool_restricted"] = m3_restricted
    # Rev 2 MAJOR-1 fix — measured noise-floor pass: repeats the
    # UNRESTRICTED M3_held_out call under eval_gen2's OWN offset scheme
    # (seed+20_000+step), same weights, same UNRESTRICTED pool -- a
    # FRESH Generator object re-seeded to the identical value (NOT the
    # same eval_gen2 object already consumed by the restricted call
    # above, which would draw a different, position-shifted
    # sub-sequence instead of the matched starting draw this null
    # needs). The ONLY thing that differs from the standard `m3` call
    # (which uses eval_gen, offset seed+10_000+step) is which generator
    # offset draws the eval batch -- isolates eval-batch sampling noise
    # from the pool-margin treatment effect. Negligible marginal cost:
    # one more evaluate_pool call, n_batches=4 default, small against a
    # full training step.
    eval_gen2_repeat = torch.Generator(device=device).manual_seed(seed + 20_000 + step)
    m3_noise_repeat = evaluate_pool(model, cfg, eval_gen2_repeat, device,
                                     (*cfg.H_test, *cfg.H_extra), pools,
                                     force_rank_k=force_rank_k,
                                     c17_repro_telemetry=c17_repro_telemetry)
    res["M3_held_out_noise_repeat"] = m3_noise_repeat
    # round-3 VERIFY-pass adopted fix (m-b, 2026-07-08, reconciled into this
    # code block at BUILD time, disclosed per house style): a single noise-
    # repeat draw is only an n=1 sample of the eval-sampling null, so the
    # build adds a SECOND, independent repeat at a THIRD generator offset
    # (seed+30_000+step, distinct from this function's own pre-loop
    # fixed_gen at seed+30_000 with no +step -- the two never collide since
    # step is always >= ckpt_every > 0 here) -- same negligible marginal
    # cost, same additive discipline. noise_shift is computed at HARVEST
    # time (never here) as max(|repeat1-standard|, |repeat2-standard|),
    # conservative over both draws.
    eval_gen3 = torch.Generator(device=device).manual_seed(seed + 30_000 + step)
    m3_noise_repeat_2 = evaluate_pool(model, cfg, eval_gen3, device,
                                       (*cfg.H_test, *cfg.H_extra), pools,
                                       force_rank_k=force_rank_k,
                                       c17_repro_telemetry=c17_repro_telemetry)
    res["M3_held_out_noise_repeat_2"] = m3_noise_repeat_2
```

**The manipulation for this diagnostic's own 2 cells (Rev 2 MAJOR-2 fix
— re-pinned to match the mechanism §15.26.2.1 itself diagnoses as live,
entity-draw OVERLAP FRACTION `K/N`, not raw spare-entity MARGIN `N−K`;
Rev 1's own margin-matched pin is disclosed as a slip in the fix-map,
§15.26.10):** K=84's own eval restricts `train_name_ids` to its first
**100** entries (`m3_pool_restrict_n=100`) — giving K=84's restricted
overlap fraction `84/100=84.00%`, matching K=90's own REAL, unmodified
overlap fraction `90/107=84.1121%` to within **0.11 percentage points**.
Rev 1's own margin-matched pin (`N'=101=84+17`, matching K=90's spare-entity
count of 17 rather than its overlap fraction) gave overlap `84/101=
83.1683%` — a 0.9438pp residual, **≈8.4× looser** on the variable this
diagnostic actually targets (precise ratio `0.9438/0.1121=8.4193`
(round-3 VERIFY-pass NEW MINOR m-a fix, 2026-07-08: corrected from a
`8.417` digit slip, §15.26.10's own arithmetic re-verified this session --
immaterial, same ≈8.4× headline either way),
rounding to `8×`, not `9×`; the attack round's own shorthand, "~9×
tighter," rounds a cruder `0.94/0.11≈8.55` estimate instead — both
describe the same real, order-of-magnitude tightening, disclosed here
as `≈8.4×` from the more precise inputs rather than silently repeating
the brief's own rounder figure). Same cost — the restriction mechanism
(§15.26.2.2's own code, one integer parameter) is unchanged; only the
integer itself moves from 101 to 100. This gives a same-checkpoint,
same-weights, paired comparison of K=84's `M3_held_out` (margin 23 /
overlap 78.50%, standard) against `M3_held_out_pool_restricted` (margin
16 / overlap 84.00%, artificially thinned to closely match K=90's own
overlap). K=90 needs no restriction (its natural overlap is already
84.11%, margin 17) and is evaluated normally — it serves as a
freshly-launched, same-production-config comparator (§15.26.3), not a
re-use of an OLD `n_iter=20` cell that could carry the
tolerance-miscalibration confound §15.25 already found and fixed once.

#### 15.26.2.3 Seed table — disclosed reuse of 2 of the killed grid's own
now-unneeded reservations

| K | K/d | cell | seed | eval treatment | provenance |
|---|---|---|---|---|---|
| 84 | 0.87500 | overlap-equalized (Rev 2 MAJOR-2 fix — was "margin-equalized," disclosed as a slip, §15.26.10) | **1943** | `M3_held_out` (margin 23 / overlap 78.50%, standard) AND `M3_held_out_pool_restricted` (margin 16 / overlap 84.00%, `restrict_entity_pool_n=100`) plus TWO noise-floor repeats (round-3 adopted fix m-b, reconciled here: the Rev 2 MAJOR-1 first repeat, `M3_held_out_noise_repeat`, unrestricted, `eval_gen2`'s own offset; plus a second, independent repeat, `M3_held_out_noise_repeat_2`, unrestricted, a third generator offset) | REDIRECTED, disclosed — 1943 was the killed 10-cell grid's own reserved K=84 contingency seed (§15.26.1's own disposition paragraph), never fired; the grid it was reserved for no longer exists, so this is a re-use of an already-idle reservation, not a new token |
| 90 | 0.93750 | natural-margin comparator | **2043** | `M3_held_out` only (margin 17, unmodified — its natural value) | REDIRECTED, disclosed — same disposition as 1943, the killed grid's own reserved K=90 contingency seed |

Seeds 1944 (K=84) and 2044 (K=90) — the OTHER half of each killed pair —
stay unclaimed, available for any future need, not fired and not
orphaned (they were never registered for anything but the now-killed
grid). **Rev 2 cross-reference:** §15.26.3.1's own MAJOR-3 whitelist fix
uses these SAME two seeds as in-process, sibling-seed reference commands
for the launch wrapper's own field-diff check — an in-memory
`_keyanchor_scaling_spec`/`build_cmd` call only, never a launch, so this
does not consume or fire either token; they remain unclaimed by the
definition above. Collision check (mechanical, re-run this session):
`grep -rn '_s1943_\|_s2043_' matrix-thinking/deltanet_rd/*.py
experiment-runs/` returns hits only in the REGISTRATION table
(`run_deltanet_rd_exactness_sweep.py:3098`) and the C17 repro instrument's
own disclosed-moot fallback note (`diag_c17_repro_analysis.py`) — zero
hits in any archived result JSON, confirming neither seed has ever
actually launched — the SAME collision-check discipline the killed
grid's own seed table used, re-run this session against this
diagnostic's own 2 redirected tokens rather than reusing the killed
grid's own (now-moot) 10-token check.

### 15.26.3 Production config (reused, unchanged) + launch mechanism (NEW,
MAJOR-2 fix)

**Production config reused verbatim from Rev 0, scope narrowed from 10
cells to 2:** both K=84 and K=90 sit at K/d≥0.875 where `geo3_n_iter=20`
produced the original wide-grid's own 3/3 inadmissibility rate (§15.22);
Rev 0's own `geo3_n_iter` 20→28 bump (justified by §15.25.5's own Step 2
finding — 295/4,608 episodes at K=84/seed=1940 require exactly
`n_iter=28`, 0 unresolved beyond) is REUSED unmodified, additive-only,
never touching `KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K`:

```python
# sec 15.26.3, REUSED from Rev 0 unmodified — additive-only, does NOT
# touch KEYANCHOR_SCALING_GATE2_N_ITER_BY_D_K. Scope narrowed to the 2
# K's this diagnostic actually launches (84, 90); the OTHER 3 entries
# (69,72,78) from Rev 0's own override are simply unused this wave, not
# deleted from the registered dict (no manifest-regression risk either
# way, since neither wave ever reads a K it doesn't launch).
KEYANCHOR_SCALING_SCATTER_RESOLUTION_N_ITER_OVERRIDE = {96: {69: 28, 72: 28, 78: 28, 84: 28, 90: 28}}
```

Both gate halves REUSED verbatim, scope narrowed to 2 cells:

1. **Negative/regression test for the n_iter bump** (`OFF×2 + ON×1`,
   `≤3×` freshly-measured OFF-vs-OFF envelope, K=84, 50-step fixed-seed
   smoke) — REUSED, already validated twice on this box (§15.24.2's own
   re-pin, §15.25.2's own fresh re-confirmation); this diagnostic's own
   K=84/seed=1943 cell reuses the SAME validated gate, not a fresh
   re-derivation.
2. **Mechanical post-hoc admission check + its own negative-test
   fixture** — REUSED, scope narrowed: `assert geo3_admission.admissible
   is True` for both landed cells (1943, 2043); any cell that still
   reads `checkpoint_fallback_seen=True` at `n_iter=28` is a new,
   disclosed finding, checked against §15.23's own C17-exclusive
   signature per the SAME adjudication rule Rev 0 already registered.
   Negative-test fixture run to completion before either real cell's
   own JSON is trusted, per requirement 5's "assert has teeth"
   discipline.

**No new kernel-safety or Gate-(b) check needed** — K∈{84,90} at d=96
give `T_bind∈{588,630}` (`grammar_rd.py:325–327`, `T_bind=7K`), both
already inside the `T∈{504,546,588,630}` PASS set verified in
`smoke_dstate_kernel_wide_result.json` (`grid_pass.96` all `true`); Gate
(b) (`n_iter`-sufficiency on the static anchor table) is REUSED by the
SAME disclosed reasoning Rev 0 already registered (flat from `n_iter=12`
onward, ~7,000× margin at `n_iter=20`, bumping to 28 adds headroom to an
already-converged quantity). Gate (c), sha256 reused-JSON, is N/A — both
cells are fresh launches.

#### 15.26.3.1 Launch mechanism, named explicitly (round-1 MAJOR-2 fix;
round-2 MAJOR-3 whitelist adaptation folded in below)

**Named, not assumed:** this diagnostic launches via a NEW, standalone
wrapper script (`run_poolmargin_k84s1943_k90s2043.py`, registered, not
yet built), mirroring `run_k69_s1733_contingency.py`'s own precedent
line-for-line — a small, two-cell, non-manifest launch does not fit the
`--wave keyanchor-scaling-wide` manifest abstraction cleanly, the exact
same reasoning the precedent wrapper used for its own single contingency
cell. **Per MAJOR-2's own finding: `run_k69_s1733_contingency.py`
RE-IMPLEMENTS both PI-signoff checks itself (`main()` lines 85–88) rather
than deferring to `run_deltanet_rd_exactness_sweep.py`'s own central
`--wave` dispatcher gate (`:3499–3508`) — this wrapper does the same,
disclosed rather than silently assumed inherited:**

```python
# registered, NOT YET BUILT — mirrors run_k69_s1733_contingency.py's own
# main()/refuse() structure exactly
if os.environ.get("KEYANCHOR_SCALING_PI_SIGNOFF", "0") != "1":
    refuse("KEYANCHOR_SCALING_PI_SIGNOFF=1 not set (gate a1).")
if os.environ.get("KEYANCHOR_SCALING_EXT_PI_SIGNOFF", "0") != "1":
    refuse("KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1 not set (gate a2, distinct from a1).")
```

**Both gates required, even though the 1× point estimate alone fits
under the ORIGINAL 21 GPU-h ceiling without needing the extension
(§15.26.4) — disclosed reasoning, not a copy-paste of the precedent:**
the 2× pessimistic bracket (§15.26.4, Rev 2 MINOR-5 fix:
`21.1666/21=100.79%`) marginally EXCEEDS the original 21, so gate a2 is
required as a conservative safety net, mirroring the precedent's own
unconditional two-gate requirement — the claim that "the extension
stays undrawn" (§15.26.4) is about the REALIZED total, not about
whether the token must be present at launch time.

**Registered negative test (build-phase item — per CLAUDE.md's own
"negative unit tests run to completion, not merely written" discipline,
mirrors the precedent's own untested-by-this-wave gate, now closed for
THIS wrapper specifically):** at build time, invoke the wrapper with (i)
neither env var set, (ii) only `KEYANCHOR_SCALING_PI_SIGNOFF=1` set, (iii)
only `KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1` set, and confirm `REFUSED`/exit
1 in all three cases before either real cell is allowed to launch — run
to completion, not merely written, closing the exact gap MAJOR-2 found
(the precedent wrapper's own gate was never negative-tested by any prior
session).

**Rev 2 MAJOR-3 fix — the field-diff/token-diff check, adapted for the
new flag, pinned explicitly (this "mirrors the precedent line-for-line"
claim above was previously silent on the precedent's OWN field-diff
mechanism, `run_k69_s1733_contingency.py:137–154` — an omission, not a
disclosed scope-narrowing):** the precedent builds its real command via
`_keyanchor_scaling_spec`/`build_cmd` (the SAME functions every audited
cell uses), builds a SECOND command via the SAME functions at a
sibling, same-K reference seed, normalizes both by replacing every
literal seed-value token with a `"SEED"` placeholder, and refuses unless
the two normalized token lists are exactly equal — catching any
config drift beyond the seed itself. A verbatim port of that check
breaks for K=84's own cell: its real command carries an extra token
this wrapper itself appends after `build_cmd` returns
(`--m3-pool-restrict-n 100`, since `build_cmd` does not know about this
diagnostic's own wave-specific flag), which the sibling-seed reference
command never has — an unmodified equality check would see that token
as an unexplained divergence and wrongly refuse a legitimate launch.
Fixed by stripping an explicitly-enumerated whitelist from the K=84
command BEFORE the precedent's own equality check runs, so the check
keeps its teeth against every OTHER kind of drift:

```python
# registered, NOT YET BUILT — Rev 2 MAJOR-3 fix, adapts
# run_k69_s1733_contingency.py's own normalize()/equality-diff pattern
# (lines 142-147 of that file) rather than porting it verbatim.
NEW_FLAG_WHITELIST = {
    "--m3-pool-restrict-n": 1,   # takes exactly 1 positional value (the int)
    # second-generator flag: N/A this design -- the noise-floor repeat
    # pass (§15.26.2.2, MAJOR-1 fix) is auto-gated by the SAME
    # `m3_pool_restrict_n is not None` condition inside train(), not its
    # own separate CLI token; this entry is reserved in case a future
    # revision splits it out, so the whitelist stays exhaustive by
    # construction rather than by omission.
}

def strip_whitelisted(cmd_tokens: list) -> list:
    out, i = [], 0
    while i < len(cmd_tokens):
        tok = cmd_tokens[i]
        if tok in NEW_FLAG_WHITELIST:
            i += 1 + NEW_FLAG_WHITELIST[tok]   # skip the flag + its N values
            continue
        out.append(tok)
        i += 1
    return out

# K=84/seed=1943: cmd carries --m3-pool-restrict-n 100 (appended by this
# wrapper, after build_cmd returns); ref_cmd is built at sibling seed
# 1944 (§15.26.2.3's own OTHER released, still-unclaimed K=84 token —
# an in-process self-consistency reference, never itself launched) and
# never carries the new flag at all.
# K=90/seed=2043: cmd carries no new flag; ref_cmd at sibling seed 2044
# (§15.26.2.3's own OTHER released K=90 token) likewise carries none --
# strip_whitelisted() is a safe no-op for this cell.
cmd_for_diff = strip_whitelisted(cmd)
n_new = normalize(cmd_for_diff, SEED)      # normalize(): precedent's own function, unchanged
n_ref = normalize(ref_cmd, REF_SEED)
if n_new != n_ref:
    refuse("generated command diverges from the sibling-seed reference command in a "
           "field OTHER than seed or a whitelisted new flag -- refusing to launch.")
```

**Registered negative test for the whitelist itself (build-phase item,
run to completion before either real cell launches, closing the exact
gap MAJOR-3 found — a whitelist that is never proven to still catch
real drift is not a safety mechanism, it is a permanent bypass):**
construct a synthetic command with ONE extra, non-whitelisted token
appended (e.g. `--bogus-extra-flag 1`, not present in
`NEW_FLAG_WHITELIST`) and confirm the check still refuses — proving the
whitelist carve-out did not silently widen into "accept any extra
token." Run alongside, not in place of, §15.26.3.1's own PI-signoff
negative test above (three combinations); all four negative cases
(missing-a1, missing-a2, missing-both, one non-whitelisted extra token)
must refuse before either real cell (seed=1943, seed=2043) is trusted
to launch.

### 15.26.4 Cost + ledger

**Historical record — the killed 10-cell grid's own cost table, retained
for the record (per house style, findings recorded near-verbatim), MINOR-1
fix applied:** base rate 0.427 GPU-h/cell (§15.22's own K=69/seed=1733
`wall_s=1535.2s`, `1535.2/3600=0.42644...`); 1× point estimate
`10×0.427=4.27 GPU-h`; ledger from §15.25.3's own realized 19.3666/26:

| Item | GPU-h | Running total | vs. ORIGINAL 21 | vs. extended 26 |
|---|---|---|---|---|
| Baseline (§15.25.3) | — | 19.3666/26 | 92.22% | 74.49% |
| Killed grid, 1× (10×0.427) | +4.270 | 23.6366/26 | **112.56%** (Rev 1 MINOR-1 fix — Rev 0 stated 112.65%; `23.6366/21=1.125552...`, corrected) — EXCEEDS the original ceiling | 90.91%, reserve 2.3634 |
| Killed grid, 2× pessimistic | +8.540 | 27.9066/26 | 132.89% | **107.33% — EXCEEDS the EXTENDED ceiling too** |

This 2× bracket breaching even the extended ceiling (never true of any
prior wave, §15.24.7's own 13.6%–112.5% history) is, in hindsight, itself
corroborating evidence for killing the grid: a design whose own
downside case was already disclosed as uniquely bad, being retired in
favor of a design whose downside case (below) stays inside the ORIGINAL
budget, is a strict improvement on both the expected-value AND the
tail-risk axis, not merely the expected-value one.

**This diagnostic's own ledger — the operative numbers, carried forward
from Rev 1 unchanged at 1× (the K=84 cell's own extra noise-floor eval
pass, MAJOR-1, adds no meaningful GPU-h beyond the headroom already
reserved below — one more `n_batches=4` eval call):** 2 cells × 0.427
GPU-h/cell (SAME base rate) = **0.854 GPU-h at 1×**, rounded up to
**≈0.9 GPU-h** to leave disclosed headroom for K=84's own extra
`M3_held_out_pool_restricted`/`M3_held_out_noise_repeat` eval passes
(§15.26.2.2) — bounded, since `evaluate_pool`'s own `n_batches=4`
default makes each additional eval call per checkpoint a small addition
relative to training, not a second training run.

| Item | GPU-h | Running total | vs. ORIGINAL 21 | vs. extended 26 |
|---|---|---|---|---|
| Baseline (§15.25.3) | — | 19.3666/26 | 92.22% | 74.49% |
| This diagnostic, 1× (2×0.427, +headroom) | +0.900 | **20.2666/26** | **96.51% — FITS WITHOUT the extension** | 77.95%, reserve 5.7334 |
| This diagnostic, 2× pessimistic | +1.800 | **21.1666/26** | **100.79% — a small, disclosed tail excess over the ORIGINAL ceiling** | 81.41%, reserve 4.8334, well inside the extension |

**Rev 2 MINOR-5 fix:** the 2× pessimistic row now doubles the SAME
rounded **0.900** GPU-h base the 1× row itself uses (`2×0.900=1.800`,
running total `19.3666+1.800=21.1666`), not the unrounded 0.854
(`2×0.854=1.708`, Rev 1's own figure) — a rounding-base inconsistency
between the 1× and 2× rows of the SAME table (Rev 1 rounded up for the
1× row's own disclosed headroom, then reverted to the unrounded figure
for the 2× row two lines below it). The underlying conclusion is
unchanged either way (both `100.36%` and `100.79%` are a small,
disclosed tail excess over the ORIGINAL 21 GPU-h ceiling, both well
inside the extended 26) — this is a consistency fix, not a new finding
about the wave's own risk profile.

**Design virtue, stated explicitly (per the task's own framing):**
because the 1× point estimate fits under the ORIGINAL 21 GPU-h ceiling
without drawing on the `+5.0 GPU-h` extension at all, this diagnostic
avoids repeating the killed grid's own disclosed weakness (a 2×
pessimistic bracket that breached even the EXTENDED ceiling) — the
extension is expected to stay UNDRAWN in practice (this program's own
realized/estimate history: every prior wave landed at 13.6%–112.5% of
its own 1× estimate, never near 2×), even though gate a2
(`KEYANCHOR_SCALING_EXT_PI_SIGNOFF`) is still required at launch
(§15.26.3.1) as a conservative, mechanically-checked safety net against
the small 2× tail (100.79%, Rev 2 MINOR-5 fix — was 100.36% under Rev
1's own rounding-inconsistent 2× figure), not because the ledger is
actually expected to need it.

**Mitigations, registered as mechanical gates:**

1. **Calibration-first launch (mandatory):** K=84/seed=1943 alone on GPU
   2 first. Abort trigger REUSED verbatim from the killed grid's own
   derivation (SAME 0.427 base rate): `1.5 × 0.427 × 2 × 3600 = 4611.6s`.
   If it fires, halt, diagnose (`nvidia-smi` contention first), re-price
   before launching K=90/seed=2043.
2. **No running-projection cut rule needed at this scale** (unlike the
   killed 10-cell grid) — 2 cells at 0.427 GPU-h/cell cannot plausibly
   miss by enough to matter at this program's own realized/estimate
   history (worst historical case 112.5% of 1×, applied here:
   `0.9×1.125=1.0125 GPU-h`, ledger `20.3791/21=97.0%`, still comfortably
   under the original ceiling) — the calibration-first gate (item 1)
   alone is sufficient mitigation for a 2-cell wave.

### 15.26.5 Fit + decision rules — the K90 pool-margin diagnostic's own
outcome table (MAJOR-1 noise-floor recalibration, this revision, plus
round-1's own MAJOR-3 discrimination-honesty disclosure, both folded in
directly)

**No sigmoid re-fit is invoked this wave** (unlike the killed grid,
which would have supplied new n=5/K points to `fit_cliff_curve.py`) —
this diagnostic supplies exactly one new, paired, within-checkpoint
comparison (K=84/seed=1943's `M3_held_out` vs.
`M3_held_out_pool_restricted`) plus one freshly-launched K=90 comparator
(seed=2043), evaluated against the pre-registered trigger below, not
against a re-fit curve.

**Pre-registered trigger (exact, not fuzzy — CLAUDE.md's
exact-threshold discipline), using Δ_measured = 1.0000 (K90's own real
n=3 mean) − seed=1943's own UNRESTRICTED `M3_held_out` h4 value (measured
fresh this wave, not assumed) and shift = seed=1943's own
`M3_held_out_pool_restricted` h4 − its own UNRESTRICTED `M3_held_out` h4
(same checkpoint, same weights, only the eval-time pool restriction
differs — "margin" retired as the framing term per the MAJOR-2 fix
below):**

**Rev 2 MAJOR-1 fix — the measured noise floor.** Rev 1's own `shift`
compared `M3_held_out_pool_restricted` (drawn under `eval_gen2`, offset
`seed+20_000+step`) against `M3_held_out` (drawn under `eval_gen`,
offset `seed+10_000+step`) — two DIFFERENT generators, so `shift`
conflated the pool-restriction treatment with plain eval-batch
resampling noise, and the `CEILING-IS-REAL` trigger (`shift ≤
0.1×Δ_measured`) was being judged against zero measured null. §15.26.2.2's
own new `M3_held_out_noise_repeat` call (same weights, same UNRESTRICTED
pool, `eval_gen2`'s own offset — the ONLY difference from the standard
`M3_held_out` call) gives a directly measured null. **Round-3 VERIFY-pass
adopted fix (m-b, 2026-07-08, reconciled into this formula at BUILD time):
a single repeat is only an n=1 draw of that null, so the build adds a
SECOND, independent repeat (`M3_held_out_noise_repeat_2`, a third
generator offset `seed+30_000+step`) and `noise_shift` is defined as the
MAX of the two independently drawn absolute deviations — conservative,
and Rev 2's own single-draw formula is exactly the degenerate n=1 case of
this same definition:**

```
noise_shift := max(|M3_held_out_noise_repeat h4 − M3_held_out h4|,
                    |M3_held_out_noise_repeat_2 h4 − M3_held_out h4|)
```

(absolute value on each draw — this null measures HOW MUCH the
generator-offset switch alone moves the reading, in either direction;
`shift` itself, below, keeps its sign, since a restriction that makes
recovery WORSE is still informative and must not be treated as an
artifact signal).

Both outcome thresholds are re-pinned RELATIVE to `noise_shift`, not
fixed fractions of `Δ_measured` alone:

```
REAL_THRESH     := max(0.1 × Δ_measured, noise_shift)
ARTIFACT_THRESH := max(0.5 × Δ_measured, 3 × noise_shift)
```

| Outcome | Trigger | What it means |
|---|---|---|
| **CEILING-IS-ARTIFACT** | `shift ≥ ARTIFACT_THRESH` | The margin-restricted eval recovers at least half of the K84–K90 gap (or ≥3× the measured eval-sampling noise, whichever bar is higher) purely from thinning the eval-time pool overlap, with the SAME trained weights — the observed K84<K90 non-monotonicity is (at least partly) an eval-time artifact of pool overlap, not a genuine capacity difference at K/d=0.875 vs 0.9375. **Round-1 MAJOR-3 discrimination-honesty disclosure, unchanged: even in this best case, any resulting re-fit or rival-band read is registered as DESCRIPTIVE ONLY, not a discriminating test.** §15.20.4's own uniform-`n=4` power check already found rival-center CI half-widths (abs-slack 0.0375, power-law 0.0287) still ~2×+ the derived `0.0145` half-gap discrimination threshold at FIVE FULL K-groups of data; this diagnostic adds at most 2 data points at 2 K's, nowhere near enough to close that gap. A training-side memorization variant of the pool-overlap hypothesis (repeated near-fixed-subset exposure DURING training, not reachable by an eval-only manipulation) also remains untested and is named as a disclosed, separately-scoped future question, not folded into this verdict. |
| **CEILING-IS-REAL** | `shift ≤ REAL_THRESH` | The margin-restricted eval is materially inert — indistinguishable from (or smaller than) the directly measured eval-sampling noise floor, `noise_shift` — the K84<K90 gap is NOT explained by eval-time pool overlap; SCATTER-IS-REAL / the observed non-monotonicity stands as measured, unresolved by this diagnostic. A negative `shift` (restriction made recovery worse) also falls here, since `REAL_THRESH ≥ 0` always. The training-side memorization variant remains a disclosed, untested, separately-scoped open question either way. |
| **AMBIGUOUS** | `REAL_THRESH < shift < ARTIFACT_THRESH` | A real but partial, inconclusive effect — reported as such, not forced into either bucket. |
| (degenerate cell) | Either cell reads `geo3_admission.admissible is not True` after the n_iter=28 gate (§15.26.3) | Escalated per the SAME §15.23 C17-exclusive-signature adjudication Rev 0 already registered, not silently absorbed |

**Totality walk — proof the three buckets are MECE for any measured
`noise_shift ≥ 0` and any `Δ_measured > 0` (the diagnostic's own
premise, K84 sitting below K90, already requires `Δ_measured > 0`; the
`Δ_measured ≤ 0` case is routed separately by the MINOR-3 contingency
below, never evaluated by this trigger at all):**

`REAL_THRESH ≤ ARTIFACT_THRESH` must hold for every possible measured
`noise_shift`, or the three buckets could overlap or leave a gap.
Three cases on `noise_shift`, exhaustive and non-overlapping by
construction:

1. **`noise_shift ≤ 0.1×Δ_measured`** (the noise floor is small relative
   to the treatment scale): `REAL_THRESH = max(0.1Δ, noise_shift) =
   0.1Δ`; `3×noise_shift ≤ 0.3Δ < 0.5Δ`, so `ARTIFACT_THRESH =
   max(0.5Δ, 3×noise_shift) = 0.5Δ`. `REAL_THRESH=0.1Δ < 0.5Δ=
   ARTIFACT_THRESH` — reduces exactly to Rev 1's own original, fixed
   10%/50% thresholds, unrecalibrated, because the measured noise floor
   turned out not to bind.
2. **`0.1×Δ_measured < noise_shift ≤ Δ_measured/6`** (the noise floor
   exceeds 10% of Δ but stays small enough that tripling it still
   doesn't reach 50% of Δ): `REAL_THRESH = noise_shift` (now the
   measured null, not the fixed fraction, governs the lower boundary);
   `3×noise_shift ≤ 0.5Δ`, so `ARTIFACT_THRESH` stays `0.5Δ`.
   `REAL_THRESH = noise_shift ≤ Δ/6 < 0.5Δ = ARTIFACT_THRESH`.
3. **`noise_shift > Δ_measured/6`** (a large measured noise floor):
   `REAL_THRESH = noise_shift` (since `noise_shift > Δ/6 > 0.1Δ`);
   `ARTIFACT_THRESH = 3×noise_shift` (since `3×noise_shift > 0.5Δ`).
   `REAL_THRESH = noise_shift < 3×noise_shift = ARTIFACT_THRESH`
   whenever `noise_shift > 0`, which this case requires.

In all three cases `REAL_THRESH < ARTIFACT_THRESH` strictly (equality
is possible only in the fully degenerate `Δ_measured=noise_shift=0`
corner, excluded by the `Δ_measured > 0` premise above) — the interval
`(REAL_THRESH, ARTIFACT_THRESH)` is always non-empty and the three
triggers (`shift ≤ REAL_THRESH`, the open interval, `shift ≥
ARTIFACT_THRESH`) partition the entire real line with no gap and no
overlap, for ANY nonnegative measured `noise_shift`. (Illustrative only,
not a substitute for the fresh measurement: using K84's own prior
3-seed aggregate mean, 0.9581, as a provisional stand-in for
`Δ_measured` gives `Δ_measured≈0.0419`, `0.1Δ≈0.00419`, `0.5Δ≈0.02095` —
matching the task brief's own illustrative figure; the actual trigger
uses seed=1943's own freshly measured value, not this proxy.)

**Rev 2 MINOR-3 fix — the Δ_measured contingency, pre-registered so a
surprising K90 reading is never silently absorbed into the trigger
table above:** `Δ_measured` is defined using K90's own real n=3 MEAN
from the OLD grid (1.0000, exact ceiling, §15.26.1's table) as its
minuend, but seed=2043 is a FRESH cell, launched under THIS wave's own
bumped `n_iter=28` (§15.26.3), not a re-use of the old n_iter=20 cells
that produced that 1.0000 figure. If seed=2043's own fresh h4 does NOT
reproduce ceiling (`h4 < 0.98` under the `n_iter=28` admission gate),
this diagnostic re-pins `Δ_measured` to use seed=2043's own fresh
reading in place of the old grid's 1.0000 mean, and DISCLOSES the
re-pin explicitly in the harvest — never silently substituting one
number for another. If seed=2043's own fresh reading drops BELOW
K=84/seed=1943's own unrestricted `M3_held_out` mean (i.e. the
K84<K90 gap this diagnostic exists to explain no longer holds under the
fresh, matched-`n_iter` comparison), the whole premise shifts: this
diagnostic does NOT force a CEILING-IS-ARTIFACT/CEILING-IS-REAL call in
that case — it routes directly to **AMBIGUOUS** and registers a
follow-up naming the reversed direction as a new, disclosed open
question, never silently picking a bucket as if the premise still held.

### 15.26.6 Gates (reuse chain, mirrors §15.20.6/§15.24.5's own table
format)

| Gate | Mechanism | Status this design |
|---|---|---|
| (a1) Kernel-safety, `T∈{588,630}` (K=84,90) | `smoke_dstate_kernel_wide_result.json`, `grid_pass.96` all `true` | REUSED, verified this session, no new probe |
| (b) `n_iter`-sufficiency (static anchor table) | `keyanchor_scaling_wide_niter_result.json`, flat/converged from `n_iter=12` | REUSED by disclosed reasoning (§15.26.3), not re-run at 28 |
| (c) sha256 reused-JSON manifest | N/A — no file copied this wave | N/A, disclosed |
| (d) PI signoff, primary | `KEYANCHOR_SCALING_PI_SIGNOFF=1` | REUSED, already-enforced token, re-implemented in the new standalone wrapper (§15.26.3.1) |
| (d) PI signoff, extension | `KEYANCHOR_SCALING_EXT_PI_SIGNOFF=1` | REUSED, already-enforced token, re-implemented in the new standalone wrapper — required as a safety net even though the 1× ledger does not need it drawn on (§15.26.4) |
| (e) n_iter-bump negative test | Baseline-relative OFF×2+ON×1, `≤3×` envelope (§15.26.3 item 1) | REUSED, already validated twice on this box |
| (f) admission-recalibration gate + its own negative test | Post-hoc `admissible is True` assert on both landed cells + synthetic-fixture negative test | REUSED, scope narrowed to 2 cells |
| (g) Seed-space collision check | `grep` across `experiment-runs/` + `*.py`, zero hits (§15.26.2.3) | REUSED mechanism, re-run this session, clean |
| (h) Calibration-first launch | §15.26.4 item 1 | REUSED mechanism, scope narrowed (no cut rule needed at 2 cells) |
| (i) NEW — launch-mechanism negative test (MAJOR-2 fix) | Wrapper refuses without either/both PI-signoff tokens, all 3 missing-token combinations tested | NEW, registered, run to completion before launch (§15.26.3.1) |
| (j) NEW — field-diff whitelist negative test (Rev 2 MAJOR-3 fix) | Wrapper refuses when the generated command carries one extra, non-whitelisted flag beyond `NEW_FLAG_WHITELIST` | NEW, registered, run to completion before launch, alongside gate (i) (§15.26.3.1) |
| (k) NEW — noise-floor measurement (Rev 2 MAJOR-1 fix; round-3 adopted fix m-b, reconciled) | K=84/seed=1943's own `M3_held_out_noise_repeat` AND `M3_held_out_noise_repeat_2` calls both land and `noise_shift := max(\|repeat1−standard\|, \|repeat2−standard\|)` is computed BEFORE the outcome trigger (§15.26.5) is evaluated | NEW, registered, mechanical precondition on the harvest (§15.26.2.2), enforced by `harvest_poolmargin_k84s1943_k90s2043.py` |

### 15.26.7 GPU plan

**GPU 2 only** (this diagnostic is 2 cells, not a 6-way fan-out — no
need for GPUs 3–7). Phase-2b (GPUs 0–1, REASONING-LINK's own Rev 1 Leg-A
pretraining lane, §16.19) is unaffected and unaffects this wave; GPUs
3–7 stay free for whichever other lane reaches launch first. Stage 0:
K=84/seed=1943 alone, calibration-gated (§15.26.4 item 1). Stage 1:
K=90/seed=2043, same GPU, after Stage 0 clears.

### 15.26.8 Standing constraints (restated, apply unchanged)

- Exact thresholds, no numerical-tolerance slack (the admission gate's
  `admissible is True`, never a fuzzy read, §15.26.3 gate item 2's
  negative test proves it; this diagnostic's own outcome trigger,
  §15.26.5, uses exact fractions of a measured gap RELATIVE to a
  measured noise floor, Rev 2 MAJOR-1 fix, not fuzzy language, and its
  own totality walk proves the three buckets are MECE for any measured
  `noise_shift`).
- Negative unit tests run to completion, not merely written (§15.26.3
  gate items 1–2; §15.26.3.1's launch-mechanism negative test, MAJOR-2
  fix, AND its own field-diff whitelist negative test, Rev 2 MAJOR-3
  fix; §15.26.2.3's collision grep, re-run this session; Rev 1's own
  positive-control run, §15.26.1, proving the degeneracy detector has
  teeth).
- Smoke test every model incl. eval batch before launch — this wave
  reuses an already-smoke-tested architecture/config; the only new
  smoke surface is the `n_iter` override (reused) plus the new
  `restrict_entity_pool_n` eval-only parameter (§15.26.2.2, additive,
  zero-behavioral-change at its `None` default) and its own new
  noise-floor repeat call (Rev 2 MAJOR-1 fix, same additive discipline).
- tmux + supervisor launch pattern, same discipline as every prior wave.
- Archive to repo (≤25MB) + SSD mirror, both, no exceptions — registered
  for the harvest, not built here (design-only session); the power-check
  scripts + their outputs ARE archived now (both the Rev 0 script and
  the Rev 1 session's own extended-verification script, below), since
  they were run in those sessions. No new scripts were run this Rev 2
  session — a pure design revision. No SSD mirror needed yet (no GPU
  cell has run under this design).
- Multiple independent adversarial audit rounds — this section is Rev 2,
  responding to attack round 2 (§15.26.10); it has not yet had its own
  attack round 3 (a VERIFY pass), still pending before CLEARED-FOR-BUILD.
- Never compress matrices to vectors — N/A, no new readout head.

### Archive (Rev 0 + Rev 1 sessions, same directory)

`experiment-runs/2026-07-08_d96_scatter_resolution_design/` (repo-
tracked, all files ≤25MB):

- `scripts/sim_d96_scatter_resolution_power.py` + `fits/sim_d96_
  scatter_resolution_power_results.json` — Rev 0's own original
  40,000-trial (20,000/null) power check, unchanged.
- `scripts/sim_d96_scatter_resolution_power_extended.py` + `fits/
  sim_d96_scatter_resolution_power_extended_results.json` — Rev 1's own
  NEW extended verification (§15.26.1): 280,000-trial 7-seed
  confirmation + 40,000-trial from-scratch reimplementation + 2,000-trial
  positive control, all byte-identical copies of
  `matrix-thinking/deltanet_rd/sim_d96_scatter_resolution_power_
  extended.py` (verified this session, `diff` clean).

**Rev 2 note:** this revision added no new run artifacts — a pure
design revision (noise-floor trigger recalibration, overlap-fraction
re-pin, wrapper whitelist), all registered as code, none run this
session. Nothing to archive beyond the two entries above.

### 15.26.9 ATTACK-ROUND-1 fix-map (2026-07-08) — verdict RESHAPE-TO-C

An independent adversarial pass reviewed §15.26 (Rev 0, the d=96
SCATTER-RESOLUTION wave's own 10-cell seed-escalation design) before any
GPU work launched, per this program's own standing multiple-independent-
audit-rounds discipline. Verdict: **RESHAPE-TO-C** — 3 MAJOR, 1 MINOR, no
FATAL. Unlike a straight NEEDS-REVISION (fix the same design and
relaunch it) or a kill, RESHAPE-TO-C found the wave's own central
empirical ambition already analytically resolved by its own pre-
registered power check (§15.26.1) — the 10-cell grid is retired, not
patched — while surfacing the wave's own single most important disclosed
loose end (the K90 pool-margin confound) as a small, differently-shaped,
cheaper instrument that fits the freed budget. Every finding below is
fixed in this revision (Rev 1, §15.26.1–§15.26.8); none is deferred or
waved away. Findings are recorded near-verbatim for the historical
record, per house style; resolutions are stated as landed in this text,
not as intentions.

| # | Finding (attack-round on §15.26, Rev 0) | Severity | Fix (Rev 1) | Location |
|---|---|---|---|---|
| MAJOR-1 | Rev 0's own text asserted a "diagnosed directly (isolated K84-vs-K90 sub-check, same script, 200,000 trials)" empirical claim (`P(K84's mean-of-N ≥ K90's 1.0)` exactly `0/200,000`, projected to N=103). No such isolated sub-check function, CLI path, or archived 200,000-trial run exists anywhere in this program's history — `sim_d96_scatter_resolution_power.py`'s own single archived run used `n_trials=20,000`/null (40,000 total), verified directly this session | MAJOR | Replaced with the explicit ANALYTIC derivation the claim actually reduces to (K84's mean-of-`n` is an exact Gaussian location-scale transform of `n_new` fresh draws plus `n_fixed` archived constants): `SE=σ√(n_new)/(n_fixed+n_new)`, giving `SE≈0.00627, gap=0.04191, z≈6.687` at `n_new=2` and `z≈19.5` projected to `n_new=100` (N=103) — exact, not simulated, and independently re-derived here from the raw archived seed values (not copied from any prior claim). One sentence disclosing Rev 0's own misattribution added explicitly, not silently absorbed | §15.26.1 (MAJOR-1 fix subsection) |
| MAJOR-2 | The launch mechanism for this wave's own 10 (now 2) special, non-manifest cells was never named. The program's own standalone-wrapper precedent, `run_k69_s1733_contingency.py`, RE-IMPLEMENTS both PI-signoff checks itself (`main()` lines 85–88) rather than deferring to the central `--wave` dispatcher's own gate (`run_deltanet_rd_exactness_sweep.py:3499–3508`) — Rev 0 never disclosed which mechanism this wave would use, or whether the precedent's own gate had ever been negative-tested | MAJOR | Named explicitly: a NEW standalone wrapper (`run_poolmargin_k84s1943_k90s2043.py`, registered, not yet built), mirroring the precedent's own two-gate re-implementation (`KEYANCHOR_SCALING_PI_SIGNOFF` + `KEYANCHOR_SCALING_EXT_PI_SIGNOFF`, both required, disclosed reasoning for requiring the extension token even though the 1× ledger doesn't need it — the 2× pessimistic bracket marginally does). A registered negative test (build-phase item, run to completion before real launch) proves refusal under all 3 missing-token combinations — closes the gap MAJOR-2 found (the precedent's own gate had never been negative-tested by any prior session either) | §15.26.3.1 (new subsection); §15.26.6 gate (i) |
| MAJOR-3 | Rev 0's own "Fit + decision rules" table included a "Rival-band comparison (conditional)" row, reachable only if CLIFF-IN-WINDOW fires, with no discrimination-honesty disclosure — §15.20.4's own already-established uniform-`n=4` power check found rival-center CI half-widths (abs-slack 0.0375, power-law 0.0287) still ~2×+ the derived `0.0145` half-gap discrimination threshold at FIVE full K-groups of data; this wave's escalation (n=5, or this diagnostic's 2 extra points) would not close that gap, but the conditional branch never said so | MAJOR | Discrimination-honesty disclosure added directly to the CEILING-IS-ARTIFACT branch of this diagnostic's own outcome table (the only surviving conditional-refit-adjacent text after the grid's own kill): explicitly states any resulting re-fit or rival-band read is DESCRIPTIVE ONLY, not discriminating, citing §15.20.4's own n=4 figures directly, and names the untested training-side memorization variant as a disclosed, separately-scoped open question | §15.26.5 (CEILING-IS-ARTIFACT row) |
| MINOR-1 | Cost-table arithmetic slip: `23.6366/21` stated as `112.65%`; the correct value is `23.6366/21=1.125552...=112.56%` | MINOR | Corrected to `112.56%` in the killed grid's own retained historical cost table (the underlying decision — that this bracket exceeds the original ceiling — is unchanged either way) | §15.26.4 (historical cost table) |

**What Rev 1 could not cleanly fix, disclosed rather than hidden:** the
K90 pool-margin control diagnostic (§15.26.2) is a genuinely NEW
instrument, never run before in any form — its own outcome is unknown
until it launches, and MAJOR-3's own discrimination-honesty disclosure
means even a clean CEILING-IS-ARTIFACT result would not fully resolve
x0(96) or discriminate the two rival bands. The training-side
memorization variant of the pool-margin hypothesis (repeated near-fixed-
subset exposure DURING training, not reachable by this diagnostic's
eval-only manipulation) remains untested, named as a disclosed future
question, not addressed here. **Rev 1 itself has not yet had its own
independent audit pass** — per this project's standing rule that
multiple independent adversarial rounds catch different bugs each round,
landing attack-round-1's findings does not, on its own, certify §15.26 as
CLEARED-FOR-BUILD.

### 15.26.10 ATTACK-ROUND-2 fix-map (2026-07-09) — verdict NEEDS-REVISION

A second independent adversarial pass reviewed §15.26 (Rev 1, landed
2026-07-08, RESHAPE-TO-C) before any GPU work launched, per this
program's own standing multiple-independent-audit-rounds discipline.
Verdict: **NEEDS-REVISION** — 0 FATAL, 3 MAJOR, 5 MINOR — every finding
surgical, individually prescribed, no category error this round. The
empirical core (the 360,000 cumulative trial power check, the analytic
K84-vs-K90 z-derivation) was independently re-verified this round and
found exceptionally clean — every cited number reproduces, and the
320,000-trial multi-seed/reimplementation/positive-control extension
underlying §15.26.1's own zero-GPU finding was independently
RE-EXECUTED, not merely re-read, and reproduced exactly. Every finding
below is fixed in this revision (Rev 2, §15.26.1–§15.26.8); none is
deferred or waved away. Findings are recorded near-verbatim for the
historical record, per house style; resolutions are stated as landed in
this text, not as intentions.

| # | Finding (attack-round on §15.26, Rev 1) | Severity | Fix (Rev 2) | Location |
|---|---|---|---|---|
| MAJOR-1 | The outcome trigger (§15.26.5)'s `CEILING-IS-REAL` branch (`shift ≤ 0.1×Δ_measured ≈ 0.00419`) has no measured null: the restricted (`M3_held_out_pool_restricted`) and unrestricted (`M3_held_out`) `M3` calls use TWO DIFFERENT eval generators (`eval_gen`, offset `seed+10_000`, vs. `eval_gen2`, offset `seed+20_000`) — `shift` therefore conflates the pool-restriction treatment with plain eval-batch sampling noise, and the trigger is judged against zero measured baseline | MAJOR | Registered ONE additional eval-only pass in the K=84 checkpoint block — the SAME unrestricted `M3_held_out` call, repeated under `eval_gen2`'s own offset (same weights, same UNRESTRICTED pool, generator offset the only difference from the standard call), giving `noise_shift := \|repeat − standard\|`, a directly measured eval-sampling null. Both outcome thresholds re-pinned relative to it: `REAL_THRESH=max(0.1×Δ,noise_shift)`, `ARTIFACT_THRESH=max(0.5×Δ,3×noise_shift)`; MECE proven by an explicit 3-case totality walk (any `noise_shift≥0` collapses cleanly to one of three orderings, `REAL_THRESH<ARTIFACT_THRESH` strictly in every case). Negligible marginal cost — one more `evaluate_pool` call | §15.26.2.2 (new noise-repeat call); §15.26.5 (re-pinned trigger table + totality walk); §15.26.6 gate (k) |
| MAJOR-2 | §15.26.2.1 itself pins the diagnosed mechanism as entity-draw OVERLAP FRACTION `K/N`, not spare-entity margin `N−K` — but Rev 1's own manipulation (`m3_pool_restrict_n=101=84+17`) matched K90's MARGIN (17), not its overlap fraction: `N'=101` gives K84's restricted overlap `84/101=83.17%` vs. K90's real `90/107=84.11%`, a 0.94pp residual on the variable the diagnostic exists to control | MAJOR | Re-pinned `N'=100`: `84/100=84.00%` vs. K90's `84.11%`, a 0.11pp residual — ≈8.4× tighter on the actually-diagnosed variable, same cost (one integer parameter change, `101→100`). Rev 1's own margin-vs-overlap slip disclosed explicitly here rather than silently corrected | §15.26.2.2 (manipulation paragraph); §15.26.2.3 (seed table row) |
| MAJOR-3 | §15.26.3.1 names the launch wrapper as mirroring `run_k69_s1733_contingency.py`'s own precedent "line-for-line," but stayed silent on that precedent's own field-diff/token-diff check (refuse unless the generated command matches a sibling-seed reference command with only seed-derived tokens differing, `run_k69_s1733_contingency.py:137–154`). A verbatim port of that check cannot pass for K=84's own cell: its real command carries a new `--m3-pool-restrict-n` flag `build_cmd` doesn't know how to generate, which the reference command never has — an unmodified equality check would wrongly refuse a legitimate launch | MAJOR | Pinned the adapted check explicitly: an enumerated `NEW_FLAG_WHITELIST` (`--m3-pool-restrict-n`, plus a reserved, currently-unused entry for a second-generator flag in case a future revision splits the noise-repeat pass out of its current auto-gated design) is stripped from the generated command BEFORE the precedent's own equality-diff runs, so the check refuses on any OTHER, non-whitelisted divergence. Its own registered negative test (a command carrying one extra, non-whitelisted flag must still be refused) closes the gap, run alongside the existing PI-signoff negative test, not in place of it | §15.26.3.1 (new whitelist subsection + negative test); §15.26.6 gate (j) |
| MINOR-1 | §15.26.2.2's own registered `m3_restricted` call omitted `c17_repro_telemetry=c17_repro_telemetry` — every OTHER pool call at this checkpoint block (`m2`, `m3`, `c17`, `c19`) threads it uniformly (`run_deltanet_rd.py:354-355`'s own docstring: "a SEPARATE flag threaded to ALL FOUR pool calls"), so the new call silently broke that invariant | MINOR | Threaded `c17_repro_telemetry=c17_repro_telemetry` into both new calls (`m3_restricted` AND the new `m3_noise_repeat`, MAJOR-1's own addition) — the invariant now covers 6 calls for K=84's own cell (4 for K=90's, which never sets `m3_pool_restrict_n`) | §15.26.2.2 (both new call sites) |
| MINOR-2 | §15.26.2.2's own citation `:961` for the `m3` call site is off by 2 — verified directly against `run_deltanet_rd.py` this round: line 961 is the PRECEDING `m2` call; `m3 = evaluate_pool(...)` itself spans lines 963–964 | MINOR | Corrected to `:963-964` at both occurrences, with the verification noted inline | §15.26.2.2 (both citations) |
| MINOR-3 | §15.26.5's `Δ_measured` uses K90's OLD n=3 mean (1.0000, from cells run under the pre-bump `n_iter`) as its minuend, but seed=2043 is a FRESH cell launched under THIS wave's own bumped `n_iter=28` — no pre-registered contingency existed for the case where the fresh reading does not reproduce that old ceiling, or reverses the K84<K90 ordering this diagnostic exists to explain | MINOR | Pre-registered contingency added: if seed=2043's fresh h4 `<0.98` under the `n_iter=28` admission gate, `Δ_measured` is re-pinned to the fresh K90 reading and the re-pin is DISCLOSED; if the fresh K90 reading drops below K84/seed=1943's own mean, the diagnostic routes directly to AMBIGUOUS plus a registered follow-up, never silently forcing a CEILING-IS-ARTIFACT/-REAL call on a premise that no longer holds | §15.26.5 (Δ_measured contingency paragraph) |
| MINOR-4 | §15.26.1's own registered finding text said h4 "near ceiling is seed-dependent," but K90's own per-seed table (§15.26.1) shows `sample sd=0.0000` across all 3 real seeds — an EXACT ceiling, not merely "near" it; the sub-ceiling K-groups (K72–K84) are where the real seed-dependence lives | MINOR | Reworded: "h4 is seed-dependent in the sub-ceiling regime (K72–K84) and non-sigmoid in this window; K90 is pinned at exact ceiling in all 3 seeds" | §15.26.1 (registered finding blockquote) |
| MINOR-5 | §15.26.4's own second ledger table rounds the 1× base up (`0.854→0.900` GPU-h, disclosed headroom) but then doubles the UNROUNDED `0.854` for the 2× pessimistic row (`+1.708`, not `+1.800`) — a rounding-base inconsistency within the same table (`19.3666+1.708=21.0746/21=100.36%`, vs. the internally-consistent `19.3666+1.800=21.1666/21=100.79%`) | MINOR | Corrected the 2× pessimistic row to double the SAME rounded 0.900 base used by the 1× row (`21.1666/26`, `100.79%`/`81.41%`); every downstream citation of the old `100.36%`/`21.0746`/`1.708` figures (§15.26.3.1, §15.26.4's own surrounding prose) updated to match. Underlying conclusion unchanged either way — both figures are a small, disclosed tail excess over the ORIGINAL ceiling, well inside the extended one | §15.26.4 (ledger table + surrounding prose); §15.26.3.1 (citation) |

**What Rev 2 could not cleanly fix, disclosed rather than hidden:** the
K90 pool-margin control diagnostic (§15.26.2) is still a genuinely NEW
instrument, never run before in any form — its own outcome remains
unknown until it launches, and the round-1 MAJOR-3 discrimination-
honesty disclosure still means even a clean CEILING-IS-ARTIFACT result
would not fully resolve x0(96) or discriminate the two rival bands. The
training-side memorization variant of the pool-overlap hypothesis
remains untested, still named as a disclosed future question. The
MAJOR-1 noise-floor fix measures `noise_shift` from a SINGLE pair of
calls (one `standard`, one `repeat`) at ONE checkpoint of ONE seed —
itself only a point estimate of the eval-sampling noise floor, not a
distribution; if that single point estimate happens to land unusually
low or high by chance, the re-pinned thresholds inherit that sampling
variance, a residual risk disclosed here rather than hidden behind the
"measured, not assumed" framing. **Rev 2 itself has not yet had its own
independent audit pass** — per this project's standing rule that
multiple independent adversarial rounds catch different bugs each
round, landing attack-round-2's findings does not, on its own, certify
§15.26 as CLEARED-FOR-BUILD; round 3 (a VERIFY pass, not a fresh full
attack round) is next, per the queue in `STATE.md`.

---

**[§15.26 ROUND-3 VERIFY PASS — DISCHARGED, 2026-07-08, recorded before
build dispatch per the gauntlet-bookkeeping rule logged this session.]**
A third independent reviewer verified Rev 2 at commit c66b3f6. VERDICT:
**DESIGN-CLEARED-FOR-BUILD** — 0 FATAL, 0 MAJOR, 2 new MINOR. Per-check:
the trigger totality walk independently re-derived (3-case enumeration +
a 200,001-point numeric sweep, zero REAL<ARTIFACT violations; the only
collapse corner Δ=noise_shift=0 is excluded both by premise and by the
MINOR-3 contingency's runtime routing); N'=100 arithmetic exact
(84/100=84.00% vs 90/107=84.11%, residual 0.11pp); the whitelist
strip-then-diff logic mechanically exercised with teeth (positive,
negative, and no-op cases); all five round-2 MINORs byte-verified landed;
the empirical core (μ=0.958093, σ=0.022157, z=6.687/19.48; K90's exact
1.0000/1.0000/1.0000; C17 per-seed values) re-derived from raw archived
JSONs to the digit. NEW MINORS folded into the build-task list: (m-a) the
"8.417" ratio digit slip → 8.4193 (immaterial, same ≈8.4× headline);
(m-b) noise_shift is an n=1 draw of the null — ADOPTED FIX: the build
adds a SECOND noise-repeat pass at a third generator offset
(seed+30_000+step), noise_shift := max of the two draws (conservative),
at the same negligible marginal cost the design's own MAJOR-1 language
already prices. Build is now licensed: 8-task list per the reviewer's
enumeration (evaluate_pool restrict param + telemetry threading; the
now-THREE extra eval calls; the wrapper w/ signoff + whitelist-diff +
negative tests; n_iter=28 admission reuse + asserts; GPU-2 two-stage
calibration-gated chain; harvest w/ noise-gated triggers + Δ contingency).
