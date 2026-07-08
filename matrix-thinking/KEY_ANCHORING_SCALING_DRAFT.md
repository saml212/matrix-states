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

## §15.24 DESIGN — C17 eval-admission repro instrument (Rev 0, 2026-07-08 —
pre-attack)

**Status: DESIGN-ONLY DRAFT (Rev 0).** Written under the same discipline as
§15.20 Rev 0/§15's own header (§15.17/§15.20.8's self-attack-round
precedent): every number below is either cited to an already-run artifact
(§15.22's per-cell `wall_s`, §15.23's residual/conditioning measurements),
or freshly VERIFIED this session against the live code (file + line, stated
explicitly). **This section has NOT yet had its own independent attack
round** — per this project's standing multiple-independent-audit-rounds
rule, it is written to survive one, not asserted as already build-ready.
No code is written or run in this session; every build task below is
registered, not built.

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
| **(b) INSTRUMENT-BUG** — a defect in the C17 probe path itself (wrong pool row-count, a K-vs-pool-size mismatch, subset selection including near-duplicate rows) | The fallback fires for a reason that has nothing to do with NS struggling on a genuinely well-posed K-row set — e.g. the sampled "K distinct held-out entities" batch is not actually K linearly-independent rows, or the pool/К bookkeeping is wrong. | EITHER the exact-rank precheck finds `rank(k_eff_raw_episode) < K` for some episode (§15.24.4 Step 0a, a DISPOSITIVE, zero-tolerance structural fact, independent of NS entirely), OR the offline NS recompute at `n_iter=20` **disagrees** with the live flag (§15.24.4 Step 0/1) — either is conclusive on its own. |
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
extreme: only 16 of 106 held-out entities are NOT drawn per episode at
K=90, vs. 22 of 106 at K=84 — closer to the pool-exhaustion boundary,
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
side-channel taps. Byte-identical to every existing cell when the flag is
absent (default behavior unchanged for every other wave, past or
future):**

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
gains one new optional parameter, `fallback_dump_sink: list | None = None`
(default `None` = zero behavioral change at every existing call site —
the M2/M3/C19 calls in `train()` at lines 802–804/807–808 pass nothing).
The C17 call site (line 805–806) passes a live list when
`--c17-repro-telemetry` is set. Inside the per-batch loop
(`run_deltanet_rd.py:263–277`), immediately after
`model.geo3_last_fallback_triggered` is read (line 274–275), if it is
`True` AND `fallback_dump_sink is not None`, append
`{"step": step_arg_passed_in, "hop": h, "batch_idx": batch_i,
"key_ids": b["key_ids"].detach().cpu().clone(),
"k_eff_raw": model.geo3_last_k_eff_raw.detach().cpu().clone(),
"k_blend_raw": model.anchor_last_k_blend_raw.detach().cpu().clone(),
"resid": model.geo3_last_resid.detach().cpu().clone()}`.
**Dumping BOTH `geo3_last_k_eff_raw` and `anchor_last_k_blend_raw` is
deliberate and free** — §15.23 already proved these are architecturally
IDENTICAL for C17 items (`anchor_blend_gather_scatter`,
`key_anchoring.py:439–469`, `trained_here=False` for 100% of C17 keys);
dumping both and asserting bitwise equality at analysis time is a live,
free re-confirmation of that claim on THIS run, not an assumption carried
over from a different cell. All events for one checkpoint's C17 call are
bundled into one file, `<ckpt_dir>/c17fallback_step<N>.pt` (a list),
mirroring the existing one-file-per-checkpoint convention.

**(iii) Per-probe-pool admission telemetry, all four pools, every
checkpoint.** `evaluate_pool()`'s existing per-hop accumulation loop
already computes `model.geo3_last_resid` on every batch (line 149,
`model_rd.py`) but currently only ORs it into a single
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

### 15.24.4 Decision rules — mechanical, pinned before the run

All thresholds below are EXACT, no numerical-tolerance slack, per this
project's own standing rule (an integer/structural check needs an exact
threshold, not a tolerance copied from a floating-point context).
`resid_tol = 0.01` throughout (`key_anchoring.GATE2_RESID_TOL`, the SAME
production constant this program uses everywhere — never re-derived).
`n_iter` sweep grid `{20, 24, 28, 32, 40}` — the SAME grid §15.23's
`anchor_table_ns_sweep()` already used, for direct comparability.

| Step | Check (exact) | Outcome |
|---|---|---|
| **0a. Exact-rank precheck** | For EVERY dumped episode: `torch.linalg.matrix_rank(k_eff_raw_episode, tol=1e-4)` (tolerance on the pre-scaled `X_0 = A/√K` basin's own natural scale, `model_rd.py`'s `newton_schulz_orthogonalize` docstring, ~line 371–390). If `rank < K` for ANY episode | **INSTRUMENT-BUG, DISPOSITIVE** — an exact structural fact (near-duplicate/degenerate held-out rows), independent of anything NS-related. Verdict fires immediately; later steps still run for diagnosis but cannot overturn this. |
| **0b. Pool-membership precheck** | Assert dumped `key_ids.shape == (B, 84)` (matches `cfg.K`) AND every id is an EXACT set-member of `pools.heldout_name_ids` (never the trained pool) | Mismatch on EITHER → **INSTRUMENT-BUG** (wrong pool row-count / K-vs-pool-size mismatch, the task's own named example) |
| **1. Live/offline NS agreement** | Recompute `newton_schulz_orthogonalize(k_eff_raw_dump, n_iter=20)` OFFLINE (fp32, same math as the live bf16-adjacent path — the live/offline dtype difference is itself part of what this check tests) for every dumped event; compare `(resid > 0.01).any()` to the live `geo3_last_fallback_triggered` flag that was `True` at that exact event | **Disagreement on ANY event → INSTRUMENT-BUG** (something in the eval-time computation, e.g. a dtype mismatch or a stale/aliased tensor reference, differs from a straightforward re-application of the documented NS math). **Agreement on EVERY event → proceed to Step 2.** |
| **2. n_iter sweep (only reached if Step 1 agrees everywhere)** | Sweep `n_iter∈{24,28,32,40}` on the SAME dumped `k_eff_raw`, per event | **Every event's residual ≤0.01 by `n_iter≤32` → TOLERANCE-MISCALIBRATION.** **ANY event still >0.01 at `n_iter=40` (the widest tested value, matching §15.23's own ceiling) → REAL-CAPACITY-BOUNDARY.** **Mixed across events (some resolve, some don't) → residual AMBIGUOUS** (named follow-on below). |

**Residual AMBIGUOUS, named follow-on (registered, not run):** re-launch
the identical cell (same seed, same config) a second time and confirm the
SAME steps/episodes reproduce the SAME fallback pattern.
**Reproducible** → escalate to 2 more cells (K=78 and K=90, bracketing
K=84) to test whether the mixed signature is itself K/d-dependent.
**NOT reproducible under the same seed** → this is ITSELF a strong
INSTRUMENT-BUG signal (nondeterminism where the design assumes
determinism) and must be escalated as its own FATAL finding, not filed
away as merely AMBIGUOUS.

**Determinism cross-check (also doubles as a Stage −1 negative test,
§15.24.5):** the full `state_dict` snapshot at `step=20000`
(§15.24.2 item i) permits an INDEPENDENT offline reconstruction of the
same C17 batch — load the snapshot into a fresh model instance, rebuild
`eval_gen = seed + 10_000 + 20000` (`run_deltanet_rd.py:801`, the exact,
already-registered formula), and replay the SAME pool-call order
(`m2, m3, c17, c19`, lines 802–808) so the shared generator advances
identically before C17's own batches are drawn. Run this ONCE against the
live-dumped tensors from §15.24.2 item (ii) and assert BYTE-IDENTICAL
`key_ids`/`k_eff_raw` — this is the "determinism of the C17 batch under
the pinned seed" gate item, and it is a REAL cross-check (two independent
derivation paths agreeing), not a vacuous shape check, mirroring this
program's own "two independent methods concur" discipline (§15.22's
`wall_s`-vs-timestamp cross-check).

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
4. **Determinism cross-check** (§15.24.4's own "Determinism of the C17
   batch" item): the offline state_dict-replay reconstruction byte-matches
   the live dump. This is the ONE Stage −1 item that can only run AFTER
   the repro cell itself has produced both the state_dict snapshot and at
   least one live dump — sequenced as the LAST Stage −1 item, immediately
   post-launch, before the decision-rule analysis (§15.24.4) is trusted.
5. **`sha256`/config cross-check** (§15.24.2's own build step): the
   reconstructed `build_cmd()` output's architecture-relevant fields
   match the archived K84/seed=1940 JSON's own `exactness_config` block,
   field by field, by hand, recorded in the build commit message.

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
- Live/offline NS-agreement mismatch at analysis time (§15.24.4 Step 1)
  → the analysis script must refuse to emit a REAL-CAPACITY-BOUNDARY or
  TOLERANCE-MISCALIBRATION verdict and instead print INSTRUMENT-BUG with
  the disagreeing event's own step/hop/batch index. Negative test: feed
  the analysis script a synthetic fixture where live and offline
  DELIBERATELY disagree (a hand-edited dump with a flipped flag), assert
  it reports INSTRUMENT-BUG and not one of the other two verdicts.
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
  H_train hops × ≤10 checkpoints, each a ≈4.1MB `(≤128,84,96)` fp32
  tensor — ≈490MB worst-case disk, no added GPU compute), (iii) cheap
  CPU-side resid aggregation reusing already-computed tensors. None of
  these add GPU kernels; each `.cpu()` call DOES force a device-sync
  stall (sub-millisecond to low-millisecond on an uncontended GPU) that
  the ORIGINAL cell's own realized rate never paid. **Disclosed,
  conservative +15% overhead buffer** (first-time code path, never
  before measured): `0.391481 × 1.15 = 0.450 GPU-h` — this is the design's
  own **1× point estimate**, landing exactly in the pre-registered
  ~0.4–0.5 GPU-h range.
- **2× pessimistic ceiling** (house discipline, §15.20.5's own "the
  realized-rate expectation is NOT the ceiling number," reinforced by this
  program's own realized/estimate ratio history of 13.6%–112.5%):
  `0.450 × 2 = 0.900 GPU-h`.

**Sub-ledger arithmetic:** KEY_ANCHORING_SCALING currently **18.1196/26
GPU-h realized** (11.7865 §15.19 + 6.3331 §15.22; §15.23's diagnostic cost
~0 GPU-h, already folded in). Adding this design's own:

| Bracket | This design | Running total | vs. ORIGINAL 21 | vs. extended 26 |
|---|---|---|---|---|
| 1× (0.450) | +0.450 | 18.5696/26 | 18.5696/21 = 88.4% | 71.4%, reserve 7.4304 |
| 2× (0.900) | +0.900 | 19.0196/26 | 19.0196/21 = **90.6%, still inside the ORIGINAL, non-extended ceiling** | 73.2%, reserve 6.9804 |

**Notable: even at the 2× pessimistic bracket, this design's own worst
case (19.0196 GPU-h) stays under the ORIGINAL 21 GPU-h ceiling** —
1.9804 GPU-h of margin remains untouched, and the `+5.0 GPU-h` extension
(already authorized, `KEYANCHOR_SCALING_EXT_PI_SIGNOFF`, never yet drawn
on per §15.22) is not needed at all for this launch. This fits.

### 15.24.8 Standing constraints (restated, apply unchanged)

- Exact thresholds, no numerical-tolerance slack (§15.24.4's `rank<K`,
  `resid>0.01`, live/offline exact-boolean-agreement checks).
- Negative unit tests run to completion, not merely written (§15.24.5's
  five Stage −1 items).
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

### 15.24.9 ATTACK-ROUND QUESTIONS — ROUND 0 (self-attack, minimum 5)

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
spuriously.

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
non-degenerate rows.

**Q3. Does dumping `k_blend_raw` alongside `k_eff_raw` for C17 batches
risk masking a REAL anchor-table interaction that §15.23's own "100%
architecturally bypassed" claim missed?** §15.23's claim rests on
`trained_here=False` for every C17 item, verified via
`anchor_trained_mask`'s own construction — this design re-confirms it
live (bitwise-equality assertion) rather than assuming it, which is the
right posture, but the attack round should confirm the bitwise-equality
assertion is actually WIRED IN as a build-time check (§15.24.2 item ii),
not merely described in prose here.

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
option.

**Q5. Does the `+15% overhead buffer` (§15.24.7) actually cover the
worst-case ~120 dump events, or is it a hand-waved number?** Disclosed
as "conservative... never before measured" — genuinely a guess, not a
measurement, unlike this program's other cost numbers (which are pulled
from real realized data). **TODO for the attack round:** either accept
the disclosed uncertainty (the 2× pessimistic ceiling already absorbs a
much larger miss than 15%) or require a even-cheaper CPU-only dry-run
timing the `.detach().cpu()`+`torch.save` cost of a single dump event
before trusting the multiplier.

---
