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

---

## § CEILING AMENDMENT (v2 re-launch) — recorded 2026-07-16 UTC, BEFORE the re-run

**What changed (compute-budget GUARD only, NOT science).** The runner's
`--ceiling-gpuh` runaway guard is raised **3.0 → 6.0 h**. A first v2 runner
launched Part A (16 primary cells) on a drained GPU under a 3.0 h wall-clock
ceiling and correctly HALTED Part B (8 discriminator cells) because the
discriminator cells MEASURED ~4.24 h > 3.0 h → they would have aborted mid-run.
The 3.0 h ceiling was priced **~2× optimistic** against the actual on-box rate.

**Measured completion time for the FROZEN 320K-step cells (unchanged science):**
- Primary (ortho / free) single-relation cell: **~2.8 h** to complete 320K steps.
- Discriminator (ortho-bank / free-bank, R=4) cell: **~4.24 h** to complete 320K
  steps.

The 6.0 h ceiling sits comfortably above BOTH measured rates and remains a pure
runaway/hang guard — it aborts only a genuinely stuck run, never a healthy one.

**Nothing scientific moves.** 320K steps (4× budget), h\*=40, the realistic eval
ladder {5,12,20,29,40,61}, the NS-polar orthogonal-write parametrization
(`n_power=12`, `n_iter`/anneal), R=4 bank, and the frozen §4 WIN/PARTIAL/NULL/FAIL
bands are ALL UNCHANGED. `--ceiling-gpuh` is not a science parameter; it does not
enter any recovery/verdict computation. The runner still emits NO verdict; the
blind assess step still applies the §4 map verbatim.

**Compute correction.** The §6 pre-registration estimated **~20 GPU-h** total. The
MEASURED rate implies ~77 GPU-h for the full 24 cells (16 primary × ~2.8 h ≈ 45
GPU-h + 8 disc × ~4.24 h ≈ 34 GPU-h ≈ **~77 GPU-h**, wall-clock reduced by
distributing across several drained GPUs). The estimate was ~2× low per cell and
the discriminator arm heavier than priced; the science bar is unaffected.

**Discipline.** This amendment is recorded and committed BEFORE the v2 re-launch
(record-verdict-first). Any completed Part A cells from the first v2 run are
resume-safe and retained (the re-launch skips COMPLETED cells).

---

## §9 VERDICT OF RECORD (blind assess, 2026-07-17)

Assessed blind against the frozen §4 / §4b bands ONLY, from the raw per-cell
JSONs. (§7 ODDS and §8 STATUS occupy 7/8; this is the next free number.)

### §9.0 Integrity check — CLEAN

- **Frozen bands intact.** `git show cffc209:…NCR_ORTHO_WRITE.md` diffed against
  the working copy: the ONLY difference is the appended `§ CEILING AMENDMENT`
  (commit 62a6fb6, a recorded compute-guard amendment, not tampering). §4/§4b are
  byte-identical to cffc209 — the frozen WIN/PARTIAL/NULL/FAIL bands govern
  verbatim.
- **Run script md5 MATCH.** On-box `/home/nvidia/ncr/ncr_ortho_write.py` md5 =
  `83b5d7bd273e9e83698fed27a9f2ef45` == committed `3086dfa` == working copy ==
  the pinned prefix `83b5d7bd…`. No drift.
- **Logs clean.** 0 `Traceback`, 0 `CUDA out of memory`, 0 `ABORTED-BUDGET` across
  all `run*.log`/`finisher*.log` (the only "ceiling" hits are parameter echoes in
  cell headers). All 24 cells `train.status = COMPLETED` at 320000 steps; all 24
  `blank_out.passed = True` (P=1 bottleneck holds: read is bit-identical under
  raw-input corruption, grad w.r.t. raw inputs exactly zero); all 16 Part-A cells
  carry `axis_c_lock_sha256`. No malformed or missing-field cells. Per-cell wall:
  Part-A free ~2.35 h, Part-A ortho ~3.3 h, Part-B ~3.4–4.3 h — all under the
  amended 6.0 h ceiling.
- **Archives (both, md5-verified spot check):**
  `experiment-runs/2026-07-16_ncr_ortho_write/` (in-repo, 5.7 MB, all files
  <25 MB cap) and
  `/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-16_ncr_ortho_write/`.
  40 JSONs (24 primary + 16 axis_c_lock) + 7 logs + the run script.

### §9.1 PART A — VERDICT: **FAIL** (constraint breaks trainability)

Scored arm = **ortho-write at K=32** (the §4 primary). Gate-0 precondition:
`min_{h∈{1,2,3}} recovered_frac@0.9 ≥ 0.9` AND `mean A_eff_rank ≥ 0.9·32 = 28.8`.
`recovered_frac@0.9` for in-dist h∈{1,2,3} read from `eval.points[h].reads.binexp`;
far depth from `realistic_ladder[h].recovered_frac@0.9`; spectral from
`spectral.mean`.

Per-cell raw metrics (all 16 Part-A cells):

| cell | ind h1 | ind h2 | ind h3 | A_eff_rank | loss min | loss final | rec@40 | rec@20 | rec@29 | rec@61 | depNrm | A_cond | min\|λ\|/c* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| free_K24_s0 | 1.000 | 1.000 | 1.000 | 24.00 | 0.0001 | 0.0001 | 1.000 | 1.000 | 1.000 | 1.000 | 0.005 | 1.0 | 1.00 |
| free_K24_s1 | 1.000 | 1.000 | 1.000 | 24.00 | 0.0001 | 0.0001 | 1.000 | 1.000 | 1.000 | 1.000 | 0.004 | 1.0 | 1.00 |
| free_K24_s2 | 1.000 | 1.000 | 1.000 | 24.00 | 0.0002 | 0.0004 | 1.000 | 1.000 | 1.000 | 1.000 | 0.008 | 1.1 | 1.00 |
| free_K24_s3 | 1.000 | 1.000 | 1.000 | 24.00 | 0.0003 | 0.0004 | 0.999 | 1.000 | 1.000 | 0.999 | 0.009 | 1.1 | 0.99 |
| free_K32_s0 | 0.988 | 0.952 | 0.875 | 31.09 | 0.0383 | 0.0395 | 0.000 | 0.000 | 0.000 | 0.000 | 0.058 | 118.7 | 0.32 |
| free_K32_s1 | 0.991 | 0.963 | 0.912 | 31.05 | 0.0312 | 0.0312 | 0.000 | 0.001 | 0.000 | 0.000 | 0.047 | 551.1 | 0.12 |
| free_K32_s2 | 0.991 | 0.963 | 0.912 | 31.09 | 0.0324 | 0.0328 | 0.000 | 0.001 | 0.000 | 0.000 | 0.054 | 54.9 | 0.28 |
| free_K32_s3 | 0.989 | 0.958 | 0.896 | 31.05 | 0.0333 | 0.0353 | 0.000 | 0.000 | 0.000 | 0.000 | 0.060 | 558.6 | 0.31 |
| **ortho_K24_s0** | 0.000 | 0.000 | 0.000 | 13.53 | 0.0652 | 1.0011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.685 | 365.5 | 3.15 |
| **ortho_K24_s1** | 0.000 | 0.000 | 0.000 | 14.37 | 0.0656 | 0.9987 | 0.000 | 0.000 | 0.000 | 0.000 | 0.463 | 334.2 | 22.26 |
| **ortho_K24_s2** | 0.000 | 0.000 | 0.000 | 16.39 | 0.0642 | 1.0049 | 0.000 | 0.000 | 0.000 | 0.000 | 0.341 | 161.7 | 1.57 |
| **ortho_K24_s3** | 0.000 | 0.000 | 0.000 | 17.96 | 0.0656 | 1.0011 | 0.000 | 0.000 | 0.000 | 0.000 | 0.302 | 153.8 | 2.85 |
| **ortho_K32_s0** | 0.000 | 0.000 | 0.000 | 27.30 | 0.9087 | 0.9965 | 0.000 | 0.000 | 0.000 | 0.000 | 0.170 | 420.8 | 24.84 |
| **ortho_K32_s1** | 0.000 | 0.000 | 0.000 | 20.54 | 0.3222 | 1.0023 | 0.000 | 0.000 | 0.000 | 0.000 | 0.422 | 1774.5 | 3.32 |
| **ortho_K32_s2** | 0.000 | 0.000 | 0.000 | 20.29 | 0.4743 | 1.0027 | 0.000 | 0.000 | 0.000 | 0.000 | 0.442 | 420.7 | 7.32 |
| **ortho_K32_s3** | 0.000 | 0.000 | 0.000 | 17.61 | 0.0632 | 1.0006 | 0.000 | 0.000 | 0.000 | 0.000 | 0.442 | 4386.5 | 0.86 |

**Band arithmetic (ortho K=32, the scored cell):**
- Gate-0 in-dist recovered_frac@0.9 = **0.000** at h=1,2,3 for **all 4/4 seeds**
  (< 0.9 bar; also < the 0.5 FAIL threshold). Rank leg also fails: A_eff_rank
  17.6–27.3 < 28.8. Gate-0 is DEAD in 4/4 seeds.
- **FAIL band** ("ortho-write Gate-0 DEAD, in-dist < 0.5, in ≥3/4 seeds — the
  projection is too rigid to train through"): satisfied at **4/4 ≥ 3/4 → FAIL.**
- WIN/PARTIAL/NULL are all unreachable: median far-depth recovered_frac@0.9 = 0.000
  at every ladder rung {5,12,20,29,40,61}; the WIN mechanistic signature
  (depart-normality ≤ 0.02, cond ≤ ~2, min|λ|/c* ≥ 0.9) is contradicted
  (ortho depart 0.17–0.69, cond 154–4387) — the untrained model's write operator
  is degenerate, not orthogonal. (NULL is additionally blocked because it requires
  Gate-0 to PASS, which it does not.)

**Loss signature (the FAIL mechanism, observed).** The ortho cells' loss
transiently *dips* then *collapses back to ~1.0*: K32 min-loss 0.063–0.909, K24
min-loss ~0.065, but every final loss ≈ 1.0 (random). The optimization briefly
engages the task then destabilizes under the orthogonal constraint — the
"too rigid to train through" mechanism §4's FAIL band names, not a flat
never-started curve.

**Baseline behaves exactly as pre-registered — isolating the constraint.** The
free-write arm trained cleanly in the SAME harness: free_K32 converges in-dist
(≈0.99/0.95/0.88 at h1/2/3) and dies at far depth (recovered_frac@0.9 = 0.000 at
every h ≥ 20, front ≈ 5–6), and its spectral diagnosis (depart 0.047–0.060, cond
54.9–558.6, min|λ|/c* 0.12–0.32) reproduces the §5 K32 wall diagnosis (0.055–0.063
/ 320–395 / 0.125–0.284) that motivated this wave. Because free trains and ortho
does not in the identical pipeline, the failure localizes to the NS-polar
constraint, not the task/eval/harness.

**Per §4, the pre-registered next move on FAIL is a fallback parametrization
(§2 Cayley map or skew-symmetric matrix-exponential) or a softer spectral-penalty
variant — NOT more budget.**

**Secondary observation (K24, not scored by §4 — primary is pinned at K=32).**
The ortho K24 cells fail identically (4/4 Gate-0 dead, same dip-then-collapse).
Notably the free K24 baseline recovers ALL far depths (h=5…61 ≈ 1.0) and is
spectrally healthy (depart ~0.006, cond ~1.0, A_eff_rank = 24.0 = full K): the
far-depth wall is a K=32 phenomenon at this tight-spare (d=K+1) recipe; K24/d25
free-write does not exhibit it, consistent with §5's framing of K24 as the
healthier baseline.

### §9.2 PART B — VERDICT: **FAIL** (ortho-bank Gate-0 dead 4/4), compounded by a dead baseline

Scored per §4b with path-length L in place of h; in-dist from `in_dist[L]`,
depth from `chain_ladder[L].recovered_frac@0.9`.

| cell | ind L1 | ind L2 | ind L3 | loss min | loss final | L20 | L32 | L40 | bank ortho_err | bank cond |
|---|---|---|---|---|---|---|---|---|---|---|
| disc_free_s0 | 0.000 | 0.000 | 0.000 | 0.9936 | 0.9989 | 0.000 | 0.000 | 0.000 | 32.4 | 6.6e12 |
| disc_free_s1 | 0.000 | 0.000 | 0.000 | 0.9943 | 1.0021 | 0.000 | 0.000 | 0.000 | 27.8 | 6.6e12 |
| disc_free_s2 | 0.412 | 0.026 | 0.001 | 0.2074 | 0.2103 | 0.000 | 0.000 | 0.000 | 3.15 | 212.6 |
| disc_free_s3 | 0.000 | 0.000 | 0.000 | 0.9949 | 0.9962 | 0.000 | 0.000 | 0.000 | 30.0 | 9.9e6 |
| **disc_ortho_s0** | 0.000 | 0.000 | 0.000 | 0.9928 | 1.0009 | 0.000 | 0.000 | 0.000 | 0.176 | 1.86 |
| **disc_ortho_s1** | 0.000 | 0.000 | 0.000 | 0.9946 | 1.0016 | 0.000 | 0.000 | 0.000 | 0.151 | 2.43 |
| **disc_ortho_s2** | 0.000 | 0.000 | 0.000 | 0.9942 | 1.0021 | 0.000 | 0.000 | 0.000 | 0.107 | 1.19 |
| **disc_ortho_s3** | 0.000 | 0.000 | 0.000 | 0.9936 | 0.9998 | 0.000 | 0.000 | 0.000 | 0.290 | 2.00 |

**Band arithmetic (ortho-bank):** in-dist recovered_frac@0.9 = **0.000** at
L=1,2,3 for **all 4/4 seeds** (loss flat at ~1.0 throughout); chain_ladder = 0.000
at every L. Gate-0 DEAD in 4/4 → **FAIL** (≥3/4). WIN (median rec@0.9 at L*=32 ≥ 0.9
AND free-bank < 0.5 at L=32) and PARTIAL (L∈{12,20}) are unreachable; NULL is
blocked (requires Gate-0 to pass).

**Compounded null — the discriminator is uninformative.** The free-bank BASELINE
also failed Gate-0: s0/s1/s3 in-dist 0.000 (loss ~1.0), s2 partial (in-dist
L1=0.412 < 0.5, loss min 0.207) → 4/4 free-bank seeds < 0.5. Neither arm learned
even the in-distribution (L=1,2,3) task at the Part-B config (R=4 bank, 320K
steps), so there is NO trained free-bank baseline to discriminate against. Part B
does not test the ortho-vs-free far-depth contrast it was designed for — the whole
Part-B task failed to train, over and above the ortho-constraint FAIL. `bank_
orthogonality` confirms neither model is orthogonal in the trained state (free
degenerate, cond up to 6.6e12; ortho ortho_err 0.11–0.29 — the write is not driven
to Q^T Q = I because the encoder output is degenerate/untrained).

### §9.3 Spectral diagnostics summary

- **Free K24 (healthy/normal):** depart-normality 0.004–0.009, cond 1.0–1.1,
  min|λ|/c* 0.99–1.00, A_eff_rank 24.0 (= full K). Far-depth recovery ≈ 1.0 at all
  ladder rungs.
- **Free K32 (the wall, reproduced):** depart 0.047–0.060, cond 54.9–558.6,
  min|λ|/c* 0.12–0.32, A_eff_rank ≈ 31. Non-normal + ill-conditioned; far-depth
  dies at h≈5–6. Matches §5's independent CPU diagnosis of the archived z-dumps —
  the mechanism the wave targeted is real and replicates.
- **Ortho K24/K32 (untrained/degenerate):** depart 0.17–0.69, cond 154–4387,
  A_eff_rank collapsed to 13.5–27.3 (below K). No trained orthogonal operator was
  produced in any ortho cell; the spectral fields reflect degenerate encoder
  output, not a normal/orthogonal write.

### §9.4 Anomalies / caveats (foregrounded)

1. **FAIL is behaviorally the pre-registered mechanism, but a code bug cannot be
   excluded from behavioral data alone.** The transient loss-dip-then-collapse
   argues the optimization engages (favoring the rigid-constraint reading), and
   the free arm training cleanly in the same pipeline localizes the failure to the
   NS-polar path — but before committing the §2 fallback, an independent code
   re-audit of `newton_schulz_polar` / `NCROrthoWriteModel.encode` under the full
   320K-step optimizer trajectory is warranted (the CPU smoke passed step-0
   grad-norm 0.5–0.9 yet training still destabilized).
2. **Part B is a compounded null.** Its free-bank baseline never trained, so Part B
   yields NO discriminator signal. The R=4 distinct-op-bank task likely needs its
   own calibration (it never cleared in-dist at 320K steps in either arm) before it
   can serve as the mod-K-trap-safe discriminator §4b intends. This is separate
   from, and does not strengthen, the Part-A ortho FAIL.
3. **Baseline-side Gate-0 nit (does not affect any verdict).** Free K32 marginally
   misses the strict Gate-0 ≥0.9 bar at h=3 (0.875 s0, 0.896 s3; 0.912 s1/s2). The
   free arm is used only for the WIN "free-write < 0.5 at h=40" comparison, which
   holds decisively (0.000 all seeds); no verdict depends on this.

### §9.5 Verdicts of record

- **Part A: FAIL** — ortho-write Gate-0 dead in 4/4 K=32 seeds (in-dist
  recovered_frac@0.9 = 0.000; loss dips then collapses to ~1.0); the orthogonal
  constraint is too rigid to train through at this parametrization. Free baseline
  reproduces the pre-registered wall (converges in-dist, dies at far depth). No
  far-depth crack. Pre-registered next move: §2 fallback (Cayley / skew-exp) or
  softer spectral penalty, not more budget.
- **Part B: FAIL** — ortho-bank Gate-0 dead in 4/4 seeds; additionally the
  free-bank baseline is dead (4/4 < 0.5), so the discriminator is uninformative
  (compounded task-level null).

**Archive paths.** `experiment-runs/2026-07-16_ncr_ortho_write/` (in-repo) and
`/Volumes/1TB_SSD/learned-representations/experiment-runs/2026-07-16_ncr_ortho_write/`
(SSD mirror). Raws: 24 primary JSONs + 16 axis_c_lock + run/finisher logs + the
pinned run script `ncr_ortho_write.py` (md5 83b5d7bd…).

---

## §10 POST-FAIL CODE RE-AUDIT (2026-07-17)

Independent, read-only code auditor. The §9 behavioral verdict is FAIL; this
section answers the one question the verdict cannot: **is the FAIL a MECHANISM
(the constraint as parametrized is genuinely not stably trainable) or a BUG (an
implementation defect a fix would reverse)?** Method: static analysis of the
pinned script (`ncr_ortho_write.py`, md5 `83b5d7bd273e9e83698fed27a9f2ef45` —
byte-identical to the canonical `matrix-thinking/ncr/ncr_ortho_write.py` and the
on-box run) + four CPU repros importing the EXACT pinned `newton_schulz_polar`
and the real training modules (`.venv` torch 2.8.0 / numpy 2.0.2). Repro
scripts and full logs archived in scratchpad; the load-bearing numbers are
reproduced inline below.

### §10.0 Verdict (up front): **(C) MECHANISM-CONFIRMED — the code is faithful; no bug.**

`newton_schulz_polar` is a correct NS polar iteration; the pre-scale is
computed per-forward from the live matrix and is essentially exact; the wiring,
dtype (pure fp32, no autocast), and gradient path are all correct. The FAIL is
a genuine **optimization pathology of the hard NS-polar parametrization**, not a
defect. Specifically: in the tight-spare `d=K+1` regime the encoder's write is
driven into an **ill-conditioning runaway** (an unconstrained/scale-free
near-singular direction), where (i) 40 NS iterations no longer orthogonalize the
output and (ii) the polar-projection backward Jacobian **explodes ~1/σ_min**;
grad-clipping to norm 1.0 then converts the explosion into task-destroying unit
steps. At the target scale K=24/32 this ill-conditioned basin is **absorbing**
(the §9 loss dips then collapses to ~1.0 permanently). Nuance that keeps this
honest: the orthogonal target itself is NOT infeasible — at K=8 the same ortho
arm escapes the trap and converges (loss 0.0017), and the free-write K24 arm
reaches a naturally near-orthogonal operator (cond 1.0) on its own. So the
recommendation is NOT "abandon orthogonality" but "change the parametrization so
the near-singular trap cannot form" — the pre-registered §2 fallback.

### §10.1 Checklist item 1 — NS convergence-domain / basin violation (the prime suspect): **REFUTED.**

The cubic NS map on singular values is `σ ← 1.5σ − 0.5σ³`, convergent to 1 only
for `σ ∈ (0, √3)`. The code pre-scales `X₀ = Z/σ̂` with `σ̂` a DETACHED,
deterministic ones-init power iteration (`n_power=12`) — computed **per-forward
from the live matrix**, not stale. Repro 1a (d=33, imposed cond 1…1e6): the
power-iter estimate is essentially exact, `est/true ∈ [0.9998, 1.0000]`, so the
post-scale true σ_max is `≤ 1.0002` in every case — **never near √3=1.732**. NS
cannot diverge from a bad pre-scale here; the basin-violation hypothesis is dead.
Repro 1b confirms the forward: with the pre-scale, NS at n_iter=40 orthogonalizes
to `‖QᵀQ−I‖≈0.0000, cond(Q)=1.00` for input cond up to **1e6**, only degrading at
cond ≥1e7 (err 0.31) / 1e8 (err 1.89) — i.e. NS is correct and robust across the
entire realistic range.

### §10.2 Item 2 — gradient path through 40 iterations: flows correctly, but EXPLODES as σ_min→0.

There is no detach/straight-through/no_grad on the projection itself (only the
scalar σ̂ pre-scale is detached, which merely fixes the gradient's scale and
preserves direction). Gradient reaches the pre-projection encoder params. BUT
Repro 2 (d=33, one tiny singular value, loss=`sum(Q)`): `‖dL/dZ‖_F ≈ 24` for
σ_min ≥ 1e-6, then **`4.6e7` at σ_min=1e-8** — the polar-Jacobian `1/(σ_i+σ_j)`
blowup. Crucially this value is *finite*, so the runner's `isfinite` guard does
NOT skip it (matches the archived `n_skipped_steps = 0–2`): the explosion is
**clipped to norm 1.0 and applied** as a unit step in a task-irrelevant
(near-singular) direction. This is the destabilizer.

### §10.3 Item 3 — numerical precision: clean (not the cause).

The NS loop casts bf16/fp16 inputs up to fp32 (`X = Z if Z.dtype in
(float32,float64) else Z.float()`) and the training loop (`train_earlyln_cell`)
runs with NO autocast — the encoder emits fp32, NS runs fp32, `orig_dtype` is
fp32. 40 iterations in fp32 is fine; precision is not the failure. (The
`Xn.to(orig_dtype)` round-trip would matter only under bf16 training, which does
not occur here.)

### §10.4 Item 4 — wiring: correct. Projection is per-write, pre-storage; eval reads the projected operator.

`encode()` applies `newton_schulz_polar` to the encoder output and returns it;
`z_dump`, `binexp_read`, `eval_cell`, `realistic_ladder_eval` and the spectral
diagnostics all consume that projected `Z`. Confirmed empirically from the
archived z-dumps: ortho cells have stored **σ_max = 1.000 exactly** (an NS fixed
point — proof the projection ran on the stored operator), free cells have σ_max
3–6 (unprojected). No init-only / wrong-tensor / post-read degeneracy. The
flag-off path is byte-identical to `NCREarlyLNModel` (free arm σ_max≠1 confirms).

### §10.5 Item 5 — loss interaction: none. No auxiliary/regularizer fights the projection.

The only loss is `cosine_loss` on the projected read; `orthogonality_error` is
used solely in diagnostics/self-test, never in training. The constraint is
enforced by projection, not penalty, so there is nothing to fight. (This is
itself relevant to the fix — see §10.8.)

### §10.6 Item 6 — why the step-0 CPU smoke passed: it could not see a training-drift-only pathology.

Repro 2b: on a random-Gaussian d=33 (what the encoder emits at init) the raw
σ_min sits ~1e-2 (cond 50–500), NS orthogonalizes to err 0.0000, and grad norm
is a benign 4–70. Everything the step-0 smoke checks (forward, backward,
finite grad, `QᵀQ≈I`, ortho-no-worse-than-free at 4 steps) is TRUE at init. The
failure is a **drift-only** phenomenon: it requires SGD to first push a singular
value below ~1e-7, which the smoke's 0–4 steps never reach. The §5 in-silico
polar preview is blind for the same reason — it polar-projects an *already
well-conditioned converged* free operator ONCE (K32 free is cond~300, K24 free
cond~1.0), never training THROUGH the projection while the encoder drifts singular.

### §10.7 The mechanism, measured on the archived run and reproduced locally

**(a) The stored ortho operator is itself non-orthogonal / rank-deficient — NS's
forward did not converge on it.** Direct `‖QᵀQ−I‖` and SVD of the archived
`z_dump.Z` (the projected operator):

| cell (per-example range) | σ_max(Q) | σ_min(Q) | cond(Q) | eff_rank | scaled ‖QᵀQ−I‖ |
|---|---|---|---|---|---|
| ortho_K32_s0 | 1.000 | 0.017–0.035 | 28–59 | 27.7–29.1 | 3.5–4.0 |
| ortho_K32_s3 | 1.000 | 0.0017–0.0041 | 244–592 | 17.7–18.4 | 10.4–10.7 |
| ortho_K24_s0 | 1.000 | 0.0014–0.0053 | 189–706 | 13.8–14.3 | 11.7–12.9 |
| free_K32_s0 (wall) | ~3.0 | ~0.01 | 248–466 | 31.9 | ~1.3 |
| free_K24_s0 (healthy) | ~6.2 | ~6.0 | **1.0** | **25.0** | 0.08 |

The ortho operator has σ_max pinned to 1 (top mode is an NS fixed point) but a
**collapsed bottom mode** (σ_min down to 1.4e-3), so it is NOT orthogonal and
eff_rank fell below K. Back-solving NS growth (`σ≈1.5×/iter` while small): a
stored σ_min≈2.4e-3 after 40 iters implies the *raw encoder* σ_min was ≈**1e-9**
relative to σ_max, i.e. **encoder cond ≈1e8** — three-plus orders past the init
cond≈8500 that n_iter=40 was calibrated for (Repro 1c: n_iter=40 orthogonalizes
a rank-deficient d=33 down to σ_min≈1e-6, FAILS at 1e-8 → err≈1.0, needs 80).
By contrast free_K24 reached cond 1.0 / full rank with NO projection — proof the
orthogonal target is a natural, trainable solution to this task.

**(b) Local reproduction (K=8, d=9, identical pipeline, 4000 CPU steps).** The
free arm converges normally (loss→0.014, raw-Z cond→1.7). The ortho arm engages
(loss dips to ~0.17) but is **trapped for ~2000 steps in the ill-conditioning
runaway** — raw-Z cond driven to **1e5–9.5e7**, σ_min touching **1.4e-7** (step
1600), σ_max ballooning 5→13.3 — precisely the near-singular regime where NS's
forward fails and its backward explodes. Then at step ~3000 it *escaped*
(cond 9.5e7→75) and converged (loss 0.0017). **K=8 shows the trap is metastable
and escapable at small scale; K=24/32 over 320K steps shows it is absorbing** —
the §9 dip-then-permanent-collapse is the same trap that K8 happened to climb out
of but the target-scale cells did not. Note `n_skipped_steps = 0` throughout in
both the local repro and the archived cells: the failure is clipped-explosion
destabilization, NOT NaN/Inf.

**Why the encoder drifts singular in the first place.** The cosine read is
scale-invariant and the pre-scale σ̂ is detached, so the loss exerts **zero
pressure on σ_max** (it drifts up freely — observed 5→13) and, in the tight-spare
`d=K+1` geometry, **zero pressure on the (d−K)=1 spare direction's magnitude**
(the read only queries the K entity keys). That unconstrained spare σ random-walks
downward; once it crosses ~1e-7 the polar backward explodes, clipping injects a
task-irrelevant unit step, and conditioning worsens further — a positive-feedback
runaway with a near-singular absorbing state. This is intrinsic to **hard-polar-
projecting the full d×d matrix when the task only constrains a K<d subspace and
the scale is free**; it is not a coding error.

### §10.8 Recommendation (grounded in the mechanism)

Per §4, the pre-registered next move on FAIL is a §2 fallback or a softer
spectral penalty — NOT more budget. The audit sharpens the choice:

1. **PRIMARY — skew-symmetric group parametrization (§2 fallback: Cayley
   `Q=(I−W)(I+W)⁻¹` or matrix-exp `Q=expm(W−Wᵀ)` on a learned skew `W`).** This
   structurally **eliminates the failure mechanism**: it never inverts a
   near-singular matrix (Cayley inverts `I+W`, always well-conditioned since W
   skew ⇒ eigenvalues `1±iθ`, |·|≥1), orthogonality is EXACT at every step (not
   approached by an iteration that can stall), and there is no free scale or
   unconstrained spare direction to run away. The free_K24 result (a fully
   orthogonal d×d operator is naturally reached and recovers all depths) and the
   K8 ortho convergence both show the *target* is trainable — only the *hard-NS
   route to it* is trapped. Do NOT bump `n_iter` (it worsens the backward
   explosion) and do NOT just raise budget (the state is absorbing at scale).

2. **Cheap confirmatory test BEFORE committing the fallback (≤0.5 GPU-h, or CPU):**
   re-run ONE ortho K=24 cell with NS-polar plus a **raw-Z conditioning
   regularizer** — a σ_min floor / `Z ← Z + εI` damped polar, or a small penalty
   on `σ_max` growth and `σ_min` collapse of the raw encoder output. If that
   single change rescues Gate-0 convergence, it *confirms* the ill-conditioning
   trap is the cause (and is itself a viable, minimal-change fix — the "softer
   spectral" branch of §2). This is the decisive mechanism-vs-artifact check and
   costs almost nothing.

**Bottom line for the coordinator:** the FAIL is real and correctly recorded; the
code is faithful (no bug to fix that would reverse it); proceed to the §2
skew-symmetric (Cayley / matrix-exp) parametrization, optionally gated by the
one-cell damped-polar confirmatory test above. The K=32 far-depth wall itself
(the free arm) is untouched by this — it remains the genuine phenomenon the next
parametrization must crack.
