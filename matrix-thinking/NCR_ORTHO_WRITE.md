# NCR ORTHOGONAL / SPECTRAL-NORMALIZED WRITE — PRE-REGISTRATION

**Opened 2026-07-16 UTC, RECORD-FIRST: this document is committed BEFORE any
code that could be tuned to an outcome (CLEAN_BYPASS: pre-registration doc, no
code). The build (`matrix-thinking/ncr/ncr_ortho_write.py`) lands in a SEPARATE,
later commit. This wave does NOT run the GPU experiment — a separate independent
audit + a pre-launch audit + a runner follow. This doc extends
`NOVEL_ARCH_WATERFALL.md` §11 (the K-axis wall) and §11.4a (the leakage/
non-normality mechanism); it is the design of record for the orthogonal-write
lever.**

Every quantitative choice below is grounded in the CPU-only diagnosis on the
ARCHIVED K=32/d=33 z-dumps (`experiment-runs/2026-07-12_ncr_k32_budget/`), whose
numbers are reproduced verbatim in §5 and were produced by the three diagnosis
scripts (`k32_spectral_diag.py`, `k32_counterfactual.py`,
`k32_nonnormal_frontier.py`) reusing the audited `ncr_spectral` /
`analyze_zdump` machinery.

---

## §1 HYPOTHESIS + MECHANISM

**The wall (established, not hypothesized).** §11.5/§11.6 CLOSED the K-axis at
K=32 on the tight-spare (d=K+1=33) recipe: even at 4× budget (320K steps),
in-distribution convergence plateaus below the 3/4 CONVERGED-ROBUST bar and,
decisively, **far-depth composition is a clean unbroken zero** — the failure
front is pinned at the trivial K−3 rung (h=29) and `rec@h*` reads 0.0000 in
every one of the 12 cells across 1×/2×/4×. §11.6 explicitly left "why" OPEN.

**The mechanism (this wave's hypothesis, diagnosis-supported).** The write is
trained with a **spectrum-blind cosine loss at h≤3**. Cosine at shallow depth
constrains only the *direction* of `Z q, Z²q, Z³q` — it is invariant to the
operator's conditioning and to its departure from normality. SGD therefore has
no gradient pressure to make the written entity operator `A` *normal* (let alone
orthogonal). The diagnosis measures exactly this: at K=32/d=33 the trained
entity operator is **non-normal** (Henrici departure-from-normality 0.055–0.063)
and **ill-conditioned** (σ_max/σ_min ≈ 320–550; the weakest entity eigenmode has
|λ|/c\* as low as 0.12–0.28). Under `h`-fold application the weakest eigenmode
decays geometrically as `(|λ|_min/c*)^h`, which is ≈0 by h≈6 — the far-depth
read annihilates it, and cosine to the true (effective-depth) target collapses.
This is a **write-conditioning failure**, not a capacity failure: `A_eff_rank`
sits at 30.2–31.2 (≈K) in every cell — the K modes are all present, just
spectrally mis-shaped.

**The fix (this wave's lever).** Constrain the write's entity operator to be
**orthogonal by construction** (all singular values equal, every eigenvalue on
the unit circle). An orthogonal operator is norm-preserving at every depth:
`Z^h` never decays or amplifies any mode, so far-depth composition holds up to
floating-point precision. In-silico preview (§5): projecting the *already-
trained, free-write* K=32 operator to its nearest orthogonal matrix (polar
factor) extends the working depth from h≈6 to **h≈27 (2× budget) / h≈51 (4×
budget)** and lifts `rec@h=20` from ~0.06 to **0.82–0.98** and `rec@h=40` to
**0.73–0.94** on the better seeds — with NO retraining. The hypothesis is that
*training* the constraint (so the model learns to write an orthogonal operator
directly, rather than a free one polar-projected post hoc) recovers realistic-
depth composition at K=32.

**Falsifiable prediction.** An orthogonal-write NCR model at K=32/d=33 will
recover far-depth composition at a realistic depth (§3's re-registered h\*)
where the pinned free-write baseline is DEAD (`rec@h* ≈ 0`, §11.6). The
mechanistic signature must move WITH the behavioral result: the trained
orthogonal write's departure-from-normality → ~0, cond# → ~1, min|λ|/c\* → ~1.

---

## §2 THE FIX — EXACT PARAMETRIZATION

**Primary lever: differentiable Newton–Schulz polar projection of the written
Z-operator.** For the encoder's raw output `Z ∈ R^{d×d}` (per batch item), the
write fed to the composition/read is the orthogonal polar factor
`Q = Z (ZᵀZ)^(−1/2)`, computed WITHOUT any eig/SVD (SVD backward is numerically
unstable when singular values are near-degenerate — and the ideal here has K
singular values degenerate at 1) via the classical Newton–Schulz iteration:

1. **Spectral-norm pre-scale (required for convergence).** `X₀ = Z / σ̂`, where
   `σ̂` is a spectral-norm estimate from a fixed-seed, DETACHED power iteration
   (`n_power = 12` steps; detached so the scale is a read-invariant no-op and
   the gradient through `X₀` is direction-preserving; **fixed seed so `encode()`
   is a deterministic pure function of `Z`** — load-bearing for the bit-identical
   z-dump/blank-out eval checks). Pre-scaling puts every singular value in
   `(0, 1]` ⊂ `(0, √3)`, the `(1.5, −0.5)` iteration's convergence basin. A
   *near-orthogonal* Z (all σ equal, the converged regime) is pre-scaled to
   σ=1 and is a fixed point, so the constraint is **near-free at convergence** —
   the reason 3–5 iterations suffice there even though an ill-conditioned early-
   training transient needs more (slow, never divergent).
2. **Newton–Schulz iteration (`n_iter` steps, default set by §6 smoke):**
   `X ← 1.5·X − 0.5·X (XᵀX)`. Quadratically convergent to the orthogonal polar
   factor near the fixed point. Batched matmuls only — fully autograd-safe.
3. **Output `Q = X`** (c=1; the scale-managed read is scale-invariant, so a free
   scalar is a no-op — we emit the pure orthogonal factor). Target check:
   `‖QᵀQ − I‖_F` small.

**Where it attaches.** A new `NCROrthoWriteModel(NCREarlyLNModel)` overrides ONLY
`encode()`: `encode = newton_schulz_polar(self.encoder(keys, values))` when the
`orthogonal` flag is set. `forward()` (the §11 earlyln inter-hop LN blend),
`eval_read()`, `arm='ncr'` are INHERITED unchanged, so every audited `run_ncr`
instrument (z_dump, `ncr_spectral` deep probe, Axis-C lock, trust screen,
`blank_out_check`, `eval_cell`) runs against the ORTHOGONALIZED Z verbatim. With
the flag OFF, `encode()` is bit-identical to `NCREarlyLNModel` (the pinned free-
write baseline) — the two arms differ by exactly the projection, isolating the
lever.

**Fallbacks named (if the smoke shows NS won't stably orthogonalize, STOP and
switch):** (a) Cayley map `Q = (I−W)(I+W)^(−1)` on a skew-symmetric `W`; (b)
skew-symmetric matrix-exponential `Q = expm(W−Wᵀ)`. Both parametrize the
orthogonal group directly. NS-polar is primary because it constrains the raw
encoder output (not a re-parametrized weight) and reuses the whole audited write
path unchanged.

---

## §3 RE-REGISTERED h\* (REALISTIC DEPTH) — pinned before any run

The synthetic §11 grid h\* for K=32 is 253 (`8K−3`). We **do NOT chase 253** —
the diagnosis (§5) shows even a perfectly polar-orthogonalized K=32 operator
recovers only ~0.14–0.35 at h=253 (fp accumulation + residual write-imperfection
over 253 physical applications). We re-register a **realistic** far depth that is
still ≥10× the training depth and far beyond anything a single forward pass of a
transformer can do.

**Depth semantics (single-relation task).** Depth `h` = number of PHYSICAL
operator applications (raw `h`). Under the single Hamiltonian K-cycle the target
is the effective-depth permutation `π^(h mod K)`; the CLAIM residue `h mod K`
must be *novel* (∉ {0,1,2,3} — the mod-K hygiene `ncr_task` already enforces).
The far-depth stress is that the operator is applied `h` times PHYSICALLY and
must still land on the effective target — exactly what non-normality breaks.

- **Training depth:** `h ∈ {1,2,3}` (unchanged; backprop through ≤3 naive
  matmuls, §3.1).
- **Realistic eval ladder (physical depths, all novel residues mod 32):**
  `{5, 12, 20, 29, 40, 61}`. `29` and `61` are the audited `GRIDS[32]` ladder
  rungs (residue 29, Axis-C-locked by the verbatim `eval_cell` pipeline);
  `{5,12,20,40}` are added via a thin realistic-ladder helper reusing the
  audited `binexp_read` + `sample_eval_batch` (NO `GRIDS` edit). `40` has
  residue 8 (novel); `61` has residue 29.
- **PRIMARY re-registered h\* = 40** (physical 40 applications ≈ **13×** the
  training depth, residue 8 novel, inside the task's suggested 32–48 band).
  Justification: the diagnosis polar frontier at K=32 reaches h≈27 (2×) / h≈51
  (4×) physical applications, and `rec@h=40` ≈ 0.73–0.94 on the better seeds
  *from a post-hoc projection of a free-write Z*; a model that TRAINS the
  constraint should write a cleaner orthogonal operator than the post-hoc polar
  of a non-orthogonal one, so h=40 is an appropriately ambitious-but-supported
  realistic target (calibrated to the ~70–75% odds in §7).
- **Nearer checkpoint h = 29** (physical ≈10× training depth) and **stretch
  h = 61** (physical ≈20×) are reported; both are audited-ladder rungs.

---

## §4 PRE-REGISTERED WIN / PARTIAL / NULL / FAIL — quantitative, before results

**Gate-0 precondition (both arms, per cell): in-distribution convergence.**
`min over h∈{1,2,3} of recovered_frac@0.9 ≥ 0.9` AND `mean A_eff_rank ≥ 0.9·K`
(the §11 Gate-1 bar, verbatim). A cell failing Gate-0 is DEAD and cannot be
scored on far depth.

**recovered threshold:** `recovered_frac@0.9` = fraction of query items whose
read-cosine to the true target exceeds 0.9 (the standing @0.9 bar). "Cleared"
at a depth = median over the cell's Gate-0-passing (converged) seeds ≥ 0.9.

### Part A — the K=32 orthogonal-write cell (primary)

- **WIN** (the realistic-depth crack): ortho-write **median rec@0.9 at h\*=40
  ≥ 0.9** across its converged seeds, AND the pinned free-write baseline reads
  **< 0.5 at h=40** (measured; §11.6 has free-write front=29, `rec@h*`=0 — the
  baseline is DEAD at every depth ≥29). The mechanistic signature must
  corroborate: ortho-write departure-from-normality ≤ 0.02, cond# ≤ ~2,
  min|λ|/c\* ≥ 0.9.
- **PARTIAL** (wall cracked, shallower): ortho-write median rec@0.9 ≥ 0.9 at
  h∈{20 OR 29} but < 0.9 at h=40 — a genuine far-depth extension (≥3× the
  free-write frontier of ~6) short of the full h\*=40 target.
- **NULL** (no far-depth gain): ortho-write median rec@0.9 < 0.9 at every
  depth ≥ 12 (no meaningful improvement over the free-write baseline beyond
  seed noise), while Gate-0 still passes (the write trains but the orthogonality
  constraint did not buy far depth — the mechanism is wrong or incomplete).
- **FAIL** (constraint breaks trainability): ortho-write Gate-0 DEAD
  (in-dist < 0.5) in ≥3/4 seeds — the projection is too rigid to train through.
  (If FAIL, the pre-registered next move is a fallback parametrization §2 or a
  softer spectral-penalty variant, not more budget.)

### Part B — the STRUCTURED-OPERATOR discriminator cell (see §4b)

- **WIN**: ortho-bank median rec@0.9 at **L\*=32** (32 DISTINCT-operator
  compositions) ≥ 0.9 AND free-bank < 0.5 at L=32.
- **PARTIAL / NULL / FAIL** defined identically to Part A with L (path length)
  in place of h and L∈{12,20} the PARTIAL checkpoints.

**Pinned baselines (free-write / earlyln at the same K, d=K+1):**
- K=32/d=33 free-write: `experiment-runs/2026-07-12_ncr_k32_budget/`
  `budget4x_earlyln_K32_s{0..3}.json` (4× budget; front=29, `rec@h*`=0.0000,
  `A_eff_rank`≈31 — §11.6 Table 2). Re-analyzed on the identical realistic
  ladder from the archived z-dumps (CPU, free).
- K=24/d=25 free-write: `experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/`
  `earlyln_K24_s{0..3}.json` (§11.4 Table 2).
- The runner supports `--arm free` so a coordinator MAY re-run the free-write
  arm fresh for a fully in-run-matched baseline (adds ~12 GPU-h); the default
  ~12-GPU-h plan pins the archived baseline because its far-depth death is
  triply established (§11.5/§11.6 behavioral + §11.4a mechanism + §5 frontier).

---

## §4b THE STRUCTURED-OPERATOR DISCRIMINATOR (Part B) — mod-K-trap-safe by construction

**Why a discriminator.** A skeptic can argue Part A's far-depth recovery is an
artifact of the single K-cycle's *structure* (one fixed permutation whose powers
are periodic; residue-stratification handles claim-eligibility but the recovered
object is still ONE operator). Part B removes every periodicity handle: it
composes a PATH of **distinct** operators, so **depth = number of compositions**
with no `h mod cycle_length` collapse possible.

**Task (precise).** One episode writes a bank of **R distinct** random-orthogonal
operators `{R_1,…,R_R}` over a shared orthonormal K-entity pool (reusing the
audited `ncr_opbank_task.generate_bank_episode` / `BankBindingEncoder` write →
`Z_bank ∈ R^{B×R×d×d}`; each `R_r` an independent Hamiltonian K-cycle, `R=4`).
A query specifies a **path** `(o_1,…,o_L)` of operator indices; the target is
`R_{o_L} ∘ … ∘ R_{o_1}` applied to the query key, computed EXACTLY by integer
index iteration (no floating point). Depth = path length `L`.

**Mod-K-trap safety (three guards, by construction, not by assumption):**
1. **Distinct operators per hop.** The composed permutation is a *product of
   different K-cycles*, NOT a power of one matrix — there is no single
   `cycle_length` to reduce modulo. This is the CLAUDE.md-rule-mandated
   "distinct operators per hop" branch, chosen over "stratify by effective
   distance."
2. **No consecutive repeats.** Paths are drawn with `o_{t+1} ≠ o_t` (a repeat
   is the only within-path route toward a single-operator power) — enforced by
   construction and asserted in the batch sampler.
3. **Fixed-point exclusion.** Query/path pairs whose composite permutation fixes
   the start (`σ(a)=a`, an exact integer check) are excluded — the identity
   shortcut is removed, mirroring `sample_eval_batch_axis_b`'s own m1 fix.
   Depth is reported RAW (L physical compositions); the target is the exact
   composite (no reduction anywhere).

**Depths.** Train `L∈{1,2,3}`; eval `L∈{5,8,12,16,20,24,32,40}`. **L\*=32** (32
genuinely-distinct compositions — no periodicity — ≈11× training depth).

**Read (continuous, no argmax/codebook — CLAUDE.md hard rule).** The chain read
applies the selected (orthogonalized) operator slices in sequence, one matvec
per hop with per-step L2 renorm (the `loop_read` scale-management, cosine-
invariant); `binexp` does not apply (the composite is not a power of one
matrix). Scored by continuous `recovery_cosine`.

**Prediction (why ~80%).** A product of L orthogonal matrices is EXACTLY
orthogonal for every L — the ortho-write bank should hold to very deep L (limited
only by fp precision and write-orthogonality quality), while the free-write
bank's non-normal products decay every weak mode. The mechanism transfers
cleanly and, if anything, more sharply than Part A (no residual-eigenphase
question — orthogonality of a product is exact).

**Bottleneck (P=1).** The chain read reads ONLY `Z_bank` (the written state),
never raw inputs — verified by a blank-out check (corrupt raw inputs post-write,
confirm the read is bit-identical and grad w.r.t. raw inputs is exactly zero).

---

## §5 IN-SILICO PREVIEW — reproduced diagnosis numbers (grounds every §3/§4 choice)

CPU-only, on the ARCHIVED K=32/d=33 z-dumps; reproduced this session
(`.venv/bin/python`, numpy 2.0.2) from `k32_nonnormal_frontier.py` /
`k32_counterfactual.py` / `k32_spectral_diag.py` (audited `ncr_spectral` /
`analyze_zdump` reuse). "front@.9" = deepest physical h holding cos≥0.9;
"polar" = the same learned Z projected to its nearest orthogonal matrix (the
in-silico stand-in for the trained constraint), NO retraining.

| group (per-seed mean) | depart-normality | cond# | front asis@.9 | front polar@.9 | polar rec@h=20 | polar rec@h=40 |
|---|---|---|---|---|---|---|
| K14@d16 (healthy)     | 0.005  | 1.0    | 179.6 | 109.0 | 1.000 | 0.999 |
| K16@d32 2K (dead)     | 0.129  | 3.6    | 3.4   | 36.6  | 0.976 | 0.919 |
| K24@d48 2K (dead)     | 0.261  | 2951.5 | 0.0   | 0.2   | 0.157 | 0.049 |
| **K32@d33 2× (wall)** | 0.063  | 395.3  | 4.6   | 26.9  | (per-seed below) | |
| **K32@d33 4× (wall)** | 0.055  | 320.8  | 5.9   | **51.4** | (per-seed below) | |

Per-K32-seed counterfactual (`k32_counterfactual.py`): as-is `rec@h*(253)`
0.056–0.076 → polar 0.274–0.354; **polar `rec@h=20` 0.823–0.979, polar
`rec@h=40` 0.728–0.942**. Spectral diagnosis (`k32_spectral_diag.py`), K32@d33
4×: `c*` 0.75–2.49, **min |λ|/c\* 0.125–0.284** (the annihilated weak mode),
phase-resid-max 0.19–0.57, A_cond 55–551.

**Reading.** The free write is non-normal and ill-conditioned; its far-depth
frontier dies at h≈5–6 physical applications. Orthogonalizing the SAME learned
operator (no retraining) pushes the frontier to h≈27 (2×) / h≈51 (4×) and lifts
mid-realistic-depth recovery to 0.73–0.98. K24@d48 (2K) is a cautionary contrast
— cond#≈2951 is so severe that even post-hoc polar barely helps; this is the 2K
convention, NOT the d=K+1 tight-spare regime this wave runs. K24@d25 (K+1) is
the far healthier baseline this wave actually uses.

---

## §6 CONFIG + COST

- **Primary (Part A):** ortho-write arm, `K∈{24,32}`, `d=K+1` (25, 33; tight-
  spare, the §11.5/§11.6 regime), `n=4` seeds, **4× budget** (320K steps, the
  §11.6 4× convention). 8 cells × ~1.5 GPU-h ≈ **12 GPU-h**. Baseline pinned from
  archive (§4). Encoder `h=64`, earlyln LN-blend recipe inherited verbatim.
- **Discriminator (Part B):** {free-bank, ortho-bank} × `K=32`, `d=33`, `R=4`,
  `n=4` seeds, 4× budget. 8 cells × ~1 GPU-h ≈ **8 GPU-h** (both arms fresh — no
  archived bank baseline).
- **Total ≈ 20 GPU-h.** Params ≤ ~185K/cell (single-relation ortho ≈175K; bank
  R=4 ≈182K — actual counts asserted in the build self-test). **Memory-trivial,
  co-resident-safe** (sub-200K params, kilobytes/example — §9.3's "never the
  constraint" applies).
- **Discipline (blind-fit / record-before-read, the R0 pattern):** WIN/PARTIAL/
  NULL/FAIL bands (§4) are frozen by THIS commit before any GPU runs; no band is
  moved after seeing results. The runner emits per-cell `recovered@h` (Part A) /
  `recovered@L` (Part B) PLUS the mechanistic spectral diagnostics
  (departure-from-normality, cond#, min|λ|/c\*, phase-resid) so the assessment is
  mechanistic — **the runner emits NO verdict**; a separate blind-assess step
  applies the §4 map.
- **Not run by this wave:** the GPU experiment. A separate independent code audit
  → a pre-launch (resource/placement) audit → a runner follow. This wave
  pre-registers (this doc) and BUILDS + SMOKES the code only.

---

## §7 PRE-REGISTERED ODDS (for an honest assessment)

- **~70–75%** the orthogonal write cracks realistic-depth (h≈20–40) composition
  at K=32 (Part A WIN or high-PARTIAL). Basis: the mechanism is measured and the
  in-silico polar preview already achieves it post-hoc without retraining;
  residual risk is (a) the constraint slowing/blocking Gate-0 convergence
  (→FAIL) or (b) trained orthogonality not being tight enough for h=40 over 40
  physical applications (→PARTIAL at h≈20–29).
- **~80%** the structured-operator discriminator (Part B) shows the ortho-bank
  recovering deep distinct-operator paths where the free-bank is dead — higher
  than Part A because a product of orthogonals is EXACTLY orthogonal (no residual
  eigenphase question).
- **~20–25%** the synthetic h=253 depth is recovered (explicitly NOT the target;
  reported only if it happens).

---

## §8 STATUS

- [x] §1–§7 pre-registered (this commit, record-first, before any tunable code).
- [ ] BUILD: `matrix-thinking/ncr/ncr_ortho_write.py` (separate commit) — NS
      polar projection, `NCROrthoWriteModel` (free/ortho flag), the structured-
      operator discriminator, the free/ortho runner, CPU self-test kill-proofs.
- [ ] SMOKE (CPU / tiny forward): NS polar differentiable + `QᵀQ≈I` + grads flow.
- [ ] Independent code audit (fresh agent).
- [ ] Pre-launch (resource/placement) audit.
- [ ] Runner: ~20 GPU-h, 4× budget, then blind assess against §4.
