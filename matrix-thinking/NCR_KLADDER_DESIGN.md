# NCR ORTHO-WRITE K-LADDER — SCALE-UP DESIGN

**REV-2 — CLEAR-FOR-CONDITIONAL-BUILD (§A3 round-3 verdict, 2026-07-16;
gauntlet: draft → attack 1 [BUILD-BLOCKED, FATAL confirmed] → Rev 1 →
attack 2 [REVISE, 4 MAJOR] → Rev 2 → round-3 verification [CLEAR, 7/7
faithful, arithmetic exact, 2 MINOR seams fixed at freeze — §A3-ADJ]).
Execution remains DOUBLE-GATED per §9: the ortho-write verdict
(WIN/PARTIAL) AND Stage-0. Dated 2026-07-16 (see §R2/§R1 changelogs).** This document is written BLIND to the live run in
`experiment-runs/2026-07-16_ncr_ortho_write/` / any `results_ortho_write`
path — no such path was read while drafting this design OR either
revision. This is a CONDITIONAL design: it does not authorize any GPU
spend by itself. It executes ONLY if the running orthogonal-write
pre-registration (`NCR_ORTHO_WRITE.md` §4) returns **WIN** or the
pre-registered **PARTIAL** band for Part A. See §9 for the exact branch.
Everything costed below is a plan, not a launch. **§1–§9 below are REVISED
TWICE (Rev 1 post-ATTACK ROUND 1 §A1; Rev 2 post-ATTACK ROUND 2 §A2); §A1,
§A1-ADJUDICATION, §R1, §A2, and §A2-ADJUDICATION are all left UNTOUCHED as
the historical record.**

**Reused, unmodified inputs (grounding every number below):**
- `NCR_ORTHO_WRITE.md` §2 (Newton–Schulz polar write, `NS_ITER_DEFAULT=40`,
  `NS_POWER_DEFAULT=12`), §3 (h\* re-registration method), §4 (WIN/PARTIAL/
  NULL/FAIL bands, Gate-0), §6 (measured rates), the CEILING AMENDMENT
  (2.8 h primary / 4.24 h discriminator, 320K steps, measured on-box).
- `NOVEL_ARCH_WATERFALL.md:3899-3901` — the pinned param/FLOP formulas
  `P(d,h)=40h²+4dh+46h+d`, `F(K,d,h)=76Kh²+4dh²+12K²h+4Kdh+4d²h` — and its
  own documented finding that FLOP-ratio scaling was **empirically refuted**
  at small K (4 measured cells flat within noise despite 1.688×–2.102× FLOP
  spread — kernel-launch/small-batch overhead-bound, not compute-bound).
- `ncr_task.py` `_gen_grid(K)`: `ladder_residue=K-3`, `ladder = m·K-3` for
  `m∈{1,2,4,8,16,32,64,128}`, synthetic `h_star=8K-3` — already defined for
  every `K∈{20,32,48,64,96,128,192,256}`, reused verbatim below (no new
  grid-generation code needed).
- `ncr_ortho_write.py`: `NS_ITER_DEFAULT=40`, `NS_POWER_DEFAULT=12`,
  `TRAIN_BATCH=256` (`run_ncr.py:78`), `d=K+1` tight-spare convention,
  encoder hidden CURRENTLY `h=64` fixed across K (`GRID_SHAPES[K]["h"]`,
  `ncr_earlyln_scale.py:75`) — **this describes the fixed input AS THE CODE
  EXISTS TODAY; §2 below (Rev 1) requires the build to change this to
  `h(K)=2K` before any K>32 cell can pass Gate-0 (§A1.1) — not yet
  implemented, a build-time requirement this design licenses, not a
  reused-verbatim input.**

---

## §1 HYPOTHESIS (one sentence)

If the Newton–Schulz orthogonal write cracks realistic-depth composition at
K∈{24,32} (WIN or PARTIAL), the SAME write mechanism — trained fresh at each
larger K∈{48,64,96,128} under the untested `d=K+1` AND `h(K)=2K`
extrapolations (§2's h-CAPACITY fix, corrected Rev 2/A2.1 — `h` cannot move
the rank ceiling past `d`, it only gives the encoder more capacity to try to
FILL `d`'s cap) — will keep producing a near-orthogonal, well-conditioned
entity operator whose recovery at a K-relative PHYSICAL depth (`h*=K+8`,
held at a fixed low EFFECTIVE relational depth by construction — residue 8;
a second novel probe at the coprime residue K−1, §3/A1.5, is measured and
reported at every cell but does NOT gate this claim, A2.4/Rev 2) stays
≥0.9, i.e., the crack is a property of the orthogonal-write mechanism's
PHYSICAL-APPLICATION fidelity and not an artifact special to K≈32; the
ladder's job is to find out whether that holds flat, decays gracefully, or
hits a new cliff. This is a SINGLE-ARM (ortho-only) trainability + fidelity
claim at K>32 — the free-write comparison is measured only up to K=32; it is
not re-measured or re-gated at larger K (§4, §6, Rev 1 / A1.2).

---

## §2 GRID + PER-K STATE DIMENSION + FLOPs/MEMORY

**Grid justification.** The task-specified `K∈{48,64,96,128}` is KEPT. Ratio
between successive rungs is 1.5×/1.33×/1.5×/1.33× — a moderate geometric
step, not a pure power-of-2 jump (32→64→128→256 was considered and
rejected: a 256 jump has zero validated calibration point anywhere near it,
and a 4-point ladder needs resolution near the boundary where uncertainty is
highest, not one giant untested leap). Four points is the minimum for a
crude power-law/decay fit with one point to spare for a sanity check.

**d convention: `d=K+1` (tight-spare), UNCHANGED — carried by extrapolation,
not validation.** This is the single biggest structural assumption in this
design (flagged again in §5). Precedent: tight-spare `d=K+1` reached 4/4
Gate-1 CONVERGED at 1× budget at BOTH K=16 and K=24
(`NCR_NEXT_LEVER_DESIGN.md` §2.1), and the currently-running wave extends it
to K=32. **Nothing has validated it past K=32.** A sharper way to see the
risk: the convention holds the ABSOLUTE headroom fixed at exactly one
dimension (`d-K=1`) while the RELATIVE headroom `1/d` shrinks monotonically
— 5.9% at K=16, 4.0% at K=24, 3.0% at K=32, and by this ladder: **2.0% at
K=48, 1.5% at K=64, 1.0% at K=96, 0.78% at K=128.** If trainability depends
on relative (not absolute) spare capacity, this ladder could fail Gate-0 for
a reason that has nothing to do with orthogonality — the calibration cell
(§5) exists partly to catch exactly this confound. **Pre-registered fallback, reframed as a DIAGNOSTIC per A1.9 (Rev 1), not a
presumed rescue:** if Gate-0 fails at the calibration K under `d=K+1`,
retest that SAME K once at `d=1.25K` to ask "does MORE headroom help" — the
answer is NOT assumed yes. `NCR_ORTHO_WRITE.md` §5 shows the opposite
convention (2K headroom) was CATASTROPHICALLY worse at K=24
(`cond#≈2951`, "polar barely helps") than the K+1 tight-spare ("far
healthier") — moving toward more headroom could just as easily worsen
conditioning as fix it; this retest is diagnostic evidence-gathering, not a
fix with presumed success. **Priced (new in Rev 1; the pre-fix design left
this un-repriced):** at K=128 (the design's own named example, `d=1.25·128=
160`, `h(K)=2K=256` per §2's h-fix below, UNCHANGED by this fallback since
`h` is a function of K only, not d): `F(128,160,256)=776,994,816`,
`NS(160)=656,588,800` (≈1.907× `NS(129)`, confirming the attack's own
estimate), `F+NS=1,433,583,616`, ratio-vs-K32 **81.27×**, single-seed cost
**≈227.6h at the primary rate (2.8h × 81.27)** — LARGER than the entire
floor deliverable (§4's 125.4h). This fallback, if triggered, is therefore
explicitly OUT-OF-BAND: it cannot be absorbed inside the 150 GPU-h
committed-sweep cap and must be scoped as its own PI-gated follow-on
decision, exactly like Part B at K=128 (§5) — not a silent in-budget retry.

**h(K) convention: `h(K)=2K` (encoder hidden scales with K) — THE A1.1 FIX,
replacing the FIXED `h=64` that made Gate-0 structurally unpassable at
K≥64.** Evidence (confirmed against the raw code, coordinator's tiebreak
read, §A1-ADJUDICATION): `BindingEncoder.row_out = nn.Linear(h, d)`
(`chapter2/model_v4.py:52`) produces every row of `Z` as the image of a
SHARED `h→d` linear map, so `rank(Z) ≤ h+1` (the `+1` is the bias)
REGARDLESS of `d` or `K`. Gate-0 requires `mean A_eff_rank ≥ 0.9·K`
(`ncr_earlyln_scale.py:322`), and `A_eff_rank ≤ rank(Z)`, so a FIXED `h=64`
caps the achievable gate at `0.9·K ≤ 65 ⟺ K ≤ 72.2` — structurally
unpassable at K=96 (bar 86.4 > ceiling 65) and K=128 (bar 115.2 > ceiling
65), and razor-thin at K=64 (bar 57.6 vs ceiling 65, 88.6% fill required).
`h(K)` MUST scale with K so the ceiling `h(K)+1` clears the bar `0.9K` with
real margin, not by luck of a soft-trained encoder happening to nearly-fill
a hard cap.

**Choice: `h(K)=2K`, the coordinator's steer — RE-JUSTIFIED on
encoder-CAPACITY grounds (A2.1 fix, Rev 2; the Rev-1 justification below
this point was WRONG about WHICH constraint `h` fixes, and is retracted).**
`Z` is `d×d` (`chapter2/model_v4.py:63`), so `rank(Z) ≤ d` UNCONDITIONALLY
— on top of A1.1's `rank(Z) ≤ h+1` bound, the BINDING ceiling is always
`min(d, h+1)`. Since `d=K+1` is untouched by the h-fix, `min(K+1, h+1) =
K+1` for every `h ≥ K`, so **`h=K` and `h=2K` have IDENTICAL rank
ceilings** — Rev 1's rejection of `h=K` ("reproduces the razor-thin
problem forever, `h=2K` fixes it") was comparing a ceiling `h` cannot
move past `d` either way; that comparison is false and is struck. The
correct reason to prefer `h=2K` is narrower: **`h` itself must not
BIND.** Under the old fixed `h=64`, `h` WAS the active constraint at
K≥64 (`h+1=65 < d`, A1.1's actual FATAL). Under `h=K`, `h+1=K+1=d`
EXACTLY — `h` no longer binds below `d`, but it ties `d` with ZERO
headroom of its own, offering the encoder no representational slack.
Under `h=2K`, `h+1=2K+1` sits comfortably above `d=K+1` (roughly 2×) —
`h` is nowhere near its own constraint, freeing the encoder's spare
capacity (wider hidden layers, more attention/FFN parameters in
`BindingEncoder`, per the params table below) to be spent on the ONE
constraint that actually matters: FILLING `d`'s cap (the real content of
the relative-headroom risk two paragraphs up, and of the fill table
below — these are the SAME question, not two). `h=2K` is also the ratio
already realized, unmodified, at the LAST VALIDATED rung (K=32:
`GRID_SHAPES[32] = h=64 = 2·32`, `ncr_earlyln_scale.py:75`) — extending it
to K∈{48,64,96,128} adds exactly ONE new extrapolation dimension (whether
the ratio itself holds past K=32), not two. A steeper choice (`h=8K`,
matching the K=8-era ratio noted in §A1-ADJUDICATION) is still rejected,
for a reason that DOES survive this correction: it buys no additional
ceiling headroom either (still `min(d,h+1)=d`), while multiplying the
already-large `40h²` param/FLOP cost 16× relative to `h=2K`, with no
evidence that extra capacity is *needed* — the tight-spare precedent
(below) already shows SGD fills the `d`-cap well at a much narrower
`h`-to-`d` ratio than 8×.

**Fill table (A2.1 fix, Rev 2 — REPLACES the retracted "new ceiling
`2K+1`, comfortable ≈0.445–0.448 margin" table. The TRUE binding ceiling
is `min(d,h+1)=d=K+1` at every rung under `h(K)=2K`, not `2K+1`):**

| K | binding ceiling `min(d,h+1)=d` | bar `0.9K` | TRUE fill `0.9K/(K+1)` |
|---|---|---|---|
| 32 (ref, validated) | 33 | 28.8 | **0.873** |
| 48 | 49 | 43.2 | **0.882** |
| 64 | 65 | 57.6 | **0.886** |
| 96 | 97 | 86.4 | **0.891** |
| 128 | 129 | 115.2 | **0.893** |

This is the SAME regime the pre-Rev-2 text elsewhere called "razor-thin"
(its own old-`h=64` K=64 row read 0.886, in those words) — honesty
requires calling K=32's own validated fill (0.873) and every extrapolated
rung (0.882-0.893) by the same name, not the "comfortable 0.445 margin"
computed against a `2K+1` ceiling `h` cannot actually deliver past `d`.
**Gate-0 is still structurally POSSIBLE at every rung** — `K+1 > 0.9K`
always, so the ceiling strictly exceeds the bar, unlike the pre-Rev-1
impossibility at K≥96 — but "possible" and "comfortable" are different
claims, and only the first is true of `2K+1`; the true ceiling `K+1`
supports neither "impossible" (Rev 0) nor "comfortable" (Rev 1), just
"tight and precedented" (below).

**Why `h(K)=2K` is still the right choice — empirical grounding, not just
structural possibility.** The tight-spare (`d=K+1`) precedent shows SGD
fills this exact ~0.87–0.89 requirement with real margin. Verified
directly against the archived per-cell JSONs (not the summary prose) for
the wave that actually ran `d=K+1` at K=16/24
(`NCR_NEXT_LEVER_DESIGN.md` §2.1's Probe-A design; executed results in
`EXPERIMENT_LOG.md`'s 2026-07-12 entry, raw cells at
`experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/*.json`,
`deep_probe.A_eff_rank`, mean over each seed's 4 eval examples): K=16
(`d=17`, bar 14.4) reaches mean `A_eff_rank` 15.9993-15.9996 across all 4
seeds → fill **0.9411-0.9412**; K=24 (`d=25`, bar 21.6) reaches
23.924-23.998 across all 4 seeds → fill **0.9570-0.9599**. Both clear the
~0.87-0.89 requirement this ladder's rungs will face with a real
~0.05-0.09 fill-fraction margin — not the false 0.445 Rev 1 claimed, but
positive and precedented. (Round-2's own "~0.93" citation for this
precedent read a DIFFERENT wave's row — the `d=2K` K=16 budget2x cell at
`NCR_NEXT_LEVER_DESIGN.md:62`, `AER 15.88`, divided by 17 instead of that
cell's own `d=32` — the numbers above are pulled from the actual `d=K+1`
Probe-A cells and are the correct ones to cite.) K=32's own validated
fill (0.873, table above) is the LOWEST of the five — the ladder's
extrapolated fills (0.882-0.893) sit slightly ABOVE the one point that's
actually measured at `d=K+1`, mild positive evidence, not proof, since no
cell has yet trained `h(K)=2K` jointly with `d=K+1` at K>32 — Stage-0
(§5) still owns that validation job, reframed below as ONE risk, not two.

**Honesty on validation status (required, §A1-ADJUDICATION; corrected
A2.1, Rev 2):** `h(K)=2K` at K∈{48,64,96,128} is ITSELF an extrapolation
of a ratio validated at exactly ONE point (K=32) — nothing has trained an
encoder with `h=96/128/192/256` and confirmed SGD actually fills the
`d=K+1` cap to anywhere near the 0.87-0.89 fraction the tight-spare
precedent achieves at K≤24. The fill table above proves the constraint is
no longer STRUCTURALLY blocking (Gate-0 is mathematically passable at
every rung); it does NOT prove trainability, and — per A2.1 — it is not a
"generous ceiling" question: `h` cannot raise the `d`-cap past `d` itself,
so the open question is purely whether the encoder, given `h=2K`'s extra
representational capacity (its actual and only role here), FILLS `d`'s
cap at these larger `K`/`d` — the SAME question as the relative-headroom
risk two paragraphs up, not a separate one. Rev 1 counted these as TWO
risks (relative-headroom convergence AND "h(K)-sufficiency"); Rev 2
collapses them into ONE, since both manifest identically (a
rank-deficient write, `orthogonality_error` reading order-unity) and
neither is separable from the other by Gate-0 alone. The Stage-0
calibration cell (§5) still inherits this job, now correctly scoped as
THREE risks (`d`-cap fillability, NS-convergence quality, realized
wall-clock), not four — the fourth was never distinct, only a mislabeled
restatement of the first.

**Params/FLOPs — closed form, `h=h(K)=2K` (Rev 1; REPLACES the fixed
`h=64` that produced A1.1), no new formula invented (same pinned formulas,
new `h` substitution):**

`P(d,h) = 40h² + 4dh + 46h + d`
`F(K,d,h) = 76Kh² + 4dh² + 12K²h + 4Kdh + 4d²h`

Plus the ortho-write-SPECIFIC term, UNCHANGED by the h-fix (depends only on
`d`, per `newton_schulz_polar`/`orthogonality_error` operating on the
`d×d` write, never on `h`): `NS(d) = 160d³ + 48d²`. Verified against the
K=32/d=33 cell (`h(32)=2·32=64`, identical to the old fixed value, so this
reference row is UNCHANGED by Rev 1): `NS(33)=5,802,192`; `P(33,64)=175,265`
— matches the pre-registration's own disclosed "≈175K params" for the
K=32 ortho cell almost exactly.

| K | d=K+1 | h(K)=2K | K/d | 1/d (rel. headroom) | P(d,h) params | F(K,d,h) | NS(d) | F+NS | ratio vs K=32 |
|---|---|---|---|---|---|---|---|---|---|
| 32 (ref) | 33 | 64 | 0.9697 | 3.0% | 175,265 | 11,837,696 | 5,802,192 | 17,639,888 | 1.000 |
| 48 | 49 | 96 | 0.9796 | 2.0% | 391,921 | 39,905,664 | 18,939,088 | 58,844,752 | 3.336 |
| 64 | 65 | 128 | 0.9846 | 1.5% | 694,593 | 94,536,192 | 44,142,800 | 138,678,992 | 7.862 |
| 96 | 97 | 192 | 0.9897 | 1.0% | 1,557,985 | 318,874,368 | 146,479,312 | 465,353,680 | 26.381 |
| 128 | 129 | 256 | 0.9922 | 0.78% | 2,765,441 | 755,631,104 | 344,269,008 | 1,099,900,112 | 62.353 |

**Two findings worth flagging, REVISED from the pre-fix design (both
FLIPPED by the `h(K)=2K` fix — the direct, disclosed cost of closing
A1.1, not a free lunch).** (1) Params are **NO LONGER trivial in relative
terms** — the pre-fix design's "+14%, never the constraint" claim is FALSE
under the fix: params now grow `175,265 → 2,765,441`, **+1,478% (~15.8×)**,
because the dominant `40h²` term is now quadratic in K (`h=2K`). In
ABSOLUTE terms this remains small by systems standards (2.8M params at the
top rung is nowhere near a memory or compute bottleneck on an 80GB H100) —
"trivial" now means "trivially fits," not "trivially flat." (2) The
Newton–Schulz term NO LONGER overtakes the base task's own FLOPs anywhere
in the ladder — the pre-fix design's "NS dominates by K=128, 5.5× F"
finding is FALSE under the fix: because `h=2K` makes `F(K,d,h)`'s
dominant `76Kh²` term ALSO cubic in K (`76K·(2K)²=304K³`), `NS(d)` (cubic
in `d≈K`) and `F` now grow at comparable rates — the `NS/F` ratio is
**0.475 (K48) → 0.467 (K64) → 0.459 (K96) → 0.456 (K128)**, roughly flat
and mildly DECLINING, the opposite direction from the pre-fix finding.
**This K-ladder is, computationally, now closer to a joint
base-task-and-NS cubic-cost ladder** — both terms grow together because
`h(K)=2K` was the price of A1.1, not because `n_iter` changed (still
fixed at 40, unchanged, still a legitimate future amendment trigger if
Stage-0 shows insufficient NS convergence at large `d` — see §5, §7).

**Memory (order-of-magnitude, `TRAIN_BATCH=256`).** The NS backward driver
is UNCHANGED by the h-fix (`NS(d)` depends only on `d`, and `d=K+1` is
untouched by Rev 1) — autograd must retain ≈2 `d×d` tensors per iteration
for `n_iter=40` iterations (the un-checkpointed default) ≈ `320·d²`
bytes/example, reusing the exact same table as the pre-fix design:

| K | NS backward mem/example | NS backward mem/batch (256) |
|---|---|---|
| 32 (ref) | ~0.35 MB | ~89 MB |
| 48 | ~0.77 MB | ~197 MB |
| 64 | ~1.35 MB | ~346 MB |
| 96 | ~3.01 MB | ~772 MB |
| 128 | ~5.33 MB | ~1.36 GB |

At K=32 this is a small slice of the PI-reported 1.7GB total. By K=128 it
could be a **comparable-or-larger slice** (~1.4GB vs. the ~1.6-1.7GB
"everything else" baseline) — a real, K-growing memory driver, still
trivial relative to an 80GB H100, but no longer negligible the way the base
model's memory always has been.

**NEW (Rev 1): the h-fix makes base-model activation memory an open
question too, previously dismissed as "kilobyte/example-trivial" when
`h=64` was fixed.** With `h(K)=2K`, the encoder's own attention buffers
(the `row_queries`/`reader` multihead-attention scores, `(B, n_heads, d, K)`
shape, `n_heads=4`) and the `TransformerEncoder`'s self-attention
(`(B, n_heads, K, K)`) both scale with `h` and `K` together. Back-of-
envelope (forward-tensor byte count only, NOT a measured number): at
K=128 (`d=129, h=256`) these two buffers are order **~65-135 MB per BATCH
of 256** (not per-example) — dwarfed by the NS driver's ~1.36 GB at the
same K (roughly 5-10% of it), but no longer literally negligible the way
it was pre-fix. This estimate is EXPLICITLY UNVERIFIED — confirm with real
`nvidia-smi`/DCGM numbers at the Stage-0 calibration cell (§5) alongside
the NS estimate, don't trust either blind. Params+optimizer(Adam, 2×)
memory stays trivial in ABSOLUTE terms even at the new param count (2.77M
params × 4 bytes × 3 ≈ 33 MB total, fixed, not per-example or per-batch) —
this part of the original "never the constraint" claim still holds, just
not the params-FLOP-relative framing (see the findings above).

---

## §3 h\* SELECTION PER K — explicit mod-K arithmetic (avoiding the collapse)

Generalizes `NCR_ORTHO_WRITE.md` §3's method exactly rather than inventing a
new one. At K=32 that section picked `h*=40=K+8` (residue 8, novel,
∉{0,1,2,3}), and reused the audited `GRIDS[32]` ladder's own first two rungs
(`m=1`→29=K-3, `m=2`→61=2K-3, both residue `K-3`) as the "nearer" and
"stretch" checkpoints. The SAME schema generalizes cleanly to every K in
this ladder because `2K-3 ≡ K-3 (mod K)` for any K — the ladder's own two
lowest rungs are always on the SAME residue class, and `K+8` is always a
DIFFERENT, independently novel residue (`(K+8) mod K = 8`, and `8∉{0,1,2,3}`
for every K≥48). Shallow sanity points `{5,12,20}` are kept as absolute
(not K-scaled) depths — they were never K-scaled at K=32 either (`5,12,20 <
32`), and for K≥48 they remain trivially `<K`, so their residue equals their
raw value, trivially novel.

| K | near = K-3 | h\* = K+8 (primary) | stretch = 2K-3 | **NEW: coprime probe = 2K-1 (Rev 1)** | synthetic h_star=8K-3 (disclaimed, NOT chased) |
|---|---|---|---|---|---|
| 48 | 45 (mod 48 = 45) | **56** (mod 48 = 8) | 93 (mod 48 = 45) | 95 (mod 48 = 47) | 381 |
| 64 | 61 (mod 64 = 61) | **72** (mod 64 = 8) | 125 (mod 64 = 61) | 127 (mod 64 = 63) | 509 |
| 96 | 93 (mod 96 = 93) | **104** (mod 96 = 8) | 189 (mod 96 = 93) | 191 (mod 96 = 95) | 765 |
| 128 | 125 (mod 128 = 125) | **136** (mod 128 = 8) | 253 (mod 128 = 125) | 255 (mod 128 = 127) | 1021 |

**A1.5 fix — the coprime probe (new, Rev 1): closes the residue-8-
monoculture gap.** Every OTHER checkpoint in this ladder sits on one of two
residue classes: `{K-3, 2K-3}` (both ≡ `-3 mod K`) and `K+8` (≡ `8 mod K`,
`gcd(8,K)=8` — EIGHT sub-cycles at every rung, identical special structure
across the whole ladder, per A1.5's aggravator finding). A COPRIME residue
(`gcd=1`, a single full K-length cycle, no sub-cycle degeneracy) was
entirely absent. The new probe, physical depth `2K-1`, residue `K-1`, is
coprime to K for EVERY K (`gcd(K-1,K)=1` always, consecutive integers) —
verified in the table above for all four rungs — and is numerically
distinct from every existing checkpoint at every K (`2K-1 ∉ {K-3, K+8,
2K-3}` for K≥48, since the pairwise differences `K+2`, `K-9`, `2` are all
nonzero). It does NOT close the "effective relational depth stays
fixed/low" finding (A1.5's primary point — see the reworded claim in
§1/§6a below) — `K-1 ≡ -1 mod K` is still a shallow effective depth, same
character as the residue-8 primary — but it DOES rule out a residue-8-
specific numerical artifact (a coincidence tied to `gcd=8` specifically)
masquerading as a general result: if BOTH the `gcd=8` primary and the
`gcd=1` coprime probe clear their bars, the result is not an artifact of
one particular sub-cycle count. **MANDATORY-REPORTED, NON-GATING (A2.4
fix, Rev 2 — corrects this bullet's own pre-Rev-2 "gated into WIN(K)"
close, which turned out to make WIN(K) internally unsatisfiable: `2K-1`
is deeper than the primary `h*=K+8`, so gating it demands a strictly
tighter `min|λ|` bar than the one pinned for the primary, one the K=32
anchor itself does not clear — §6's WIN(K) band has the full
arithmetic).** Still measured and disclosed at every cell, still able to
positively rule out a `gcd=8`-specific artifact if it clears, just no
longer able to veto an otherwise-clean primary-bar WIN if it doesn't.

Per-K realistic ladder: `{5, 12, 20, K-3, K+8, 2K-1, 2K-3}` (the coprime
probe `2K-1` added, Rev 1). All residues verified ∉{0,1,2,3} for all four K
(mechanical check, not assumed): `5, 12, 20` are literal values `<K`, always
novel; `K-3` and `2K-3` share residue `K-3` which is `≥45` for K≥48, never
0-3; `K+8` has residue `8`, never 0-3; `2K-1` has residue `K-1`, `≥47` for
K≥48, never 0-3. Training-depth multiple of the primary target grows with K
(`h*/3`): 18.7× (K48), 24× (K64), 34.7× (K96), 45.3× (K128) — all
comfortably past the "≥10× training depth" bar the K=32 pre-registration
used, growing MORE ambitious (not less) as K increases, since `h*=K+8`
scales with K while training depth stays pinned at 3.

**Part B (discriminator) needs no new arithmetic.** Depth there is
compositions-of-DISTINCT-operators (`L`), never a power of one matrix, so
`L mod anything` collapse cannot occur BY CONSTRUCTION (§4b's three guards).
`L∈{1,2,3}` train / `{5,8,12,16,20,24,32,40}` eval / `L*=32` stays IDENTICAL
across every K in this ladder — nothing to re-derive.

---

## §4 SEEDS + CELL GRID + GPU-h PRICING

**Measured base rates (the ONLY pricing inputs, per instruction — from
`NCR_ORTHO_WRITE.md`'s CEILING AMENDMENT, K=32, 320K steps, on-box):**
primary (single-relation) cell **2.8 h**; discriminator (R=4 bank) cell
**4.24 h** (the measured WORST case).

**Cost-scaling assumption, stated explicitly AS AN ASSUMPTION (not fact):**
project cost with the `(F+NS)` FLOP ratio from §2 — i.e., assume
COMPUTE-BOUND scaling from the K=32 measured rate. This is deliberately the
CONSERVATIVE (over-pricing) choice for planning/ceiling purposes: the
project's own established finding at small K (`NOVEL_ARCH_WATERFALL.md`
§9.10/§11) is that these cells are actually OVERHEAD-BOUND — measured rates
were FLAT within noise despite up to 2.1× FLOP spread. That precedent argues
the TRUE cost at K=48/64 will likely track well BELOW the FLOP-ratio
estimate. But the FLOP spread in THIS ladder reaches **62×** (not 2×; Rev 1: even
LARGER than the pre-fix design's 23×, since `h(K)=2K` — the A1.1 fix, §2 —
makes BOTH `F` and `NS` grow together, not just `NS`), driven by
cubic-in-K growth in both terms (§2) — a regime the small-K precedent
never tested, and where compute-bound behavior becoming real is entirely
plausible. Given the last ceiling was mispriced 2× LOW by trusting a
similar per-cell extrapolation (the 3.0h→6.0h amendment), this design
prices the CEILING off the pessimistic assumption and gates actual spend
behind a calibration cell (§5) that measures the truth before commit.

**Nominal (FLOP-ratio-scaled) worst-case per-cell hours (Rev 1: recomputed
with `h(K)=2K`, §2 — LARGER than the pre-fix table at every K, the direct
cost of closing A1.1):**

| K | primary (×2.8h) | disc (×4.24h) |
|---|---|---|
| 48 | 9.34 h | 14.14 h |
| 64 | 22.01 h | 33.33 h |
| 96 | 73.87 h | 111.85 h |
| 128 | 174.59 h | 264.38 h |

**Cell grid (this design's scope, cost-trimmed by construction, not by
seed-count alone).** Part A (single-relation, ortho arm ONLY) runs at **all
four K, n=4 seeds each** — the primary trainability + physical-depth-
fidelity scaling-law axis this design exists to answer. **A1.2 fix (Rev
1): the free-write arm is NOT re-run at any new K in this grid — stated
now as a DELIBERATE SCOPE CHOICE, not baseline reuse.** The pre-fix design
treated the K≤32 free-write DEATH as a "reusable pinned baseline" inside
the WIN(K) gate at K>32; that was wrong (§A1.2) — no free-write z-dump
exists past K=32, so there is nothing to "reuse," only something to
extrapolate and mislabel as measured. Rev 1 does NOT add fresh free-write
cells either: even ONE extra n=1 free-write cell at the cheapest rung
(K=48) costs ≈9.4h (the F-only-scaled rate, no NS term for the free arm),
more than the ≈0.6h of headroom the worst-case-satisfying trim order
leaves under the 150h cap (below) — so the correct fix is REFRAMING, not
re-measuring: **every WIN(K)/PARTIAL(K)/NULL(K) band at K>32 (§6, Rev 1)
is now an explicit ONE-ARM (ortho-only) trainability + physical-depth-
fidelity claim.** The free-write comparison is cited ONLY as directional
context, labeled "extrapolated from the K≤32 measured death, NOT measured
at this K" — it never gates a WIN/PARTIAL/NULL verdict past K=32. Part B
(discriminator, ortho-bank ONLY, same one-arm logic) runs ONLY at the two
ENDPOINTS, **K=48 and K=128, n=4 seeds each** — enough to
check the mod-K-trap-safe compositional result generalizes to a near point
and a far point without paying for the full 4-K sweep on both arms.
Contingency: if K=48's ortho-bank cell breaks the expected pattern, the two
skipped middle K's (64, 96) for Part B become a pre-registered follow-on,
not silently dropped forever.

**Nominal (worst-case, FLOP-scaled) total for this grid (Rev 1: recomputed
with `h(K)=2K`, §2 — the direct, disclosed cost of closing A1.1; ~2.55× the
pre-fix design's sticker price):**

| arm | K48 | K64 | K96 | K128 | subtotal |
|---|---|---|---|---|---|
| A (n=4) | 37.36 h | 88.05 h | 295.46 h | 698.35 h | 1,119.23 h |
| B (n=4, endpoints only) | 56.58 h | — | — | 1,057.51 h | 1,114.08 h |
| **total** | | | | | **≈2,233.3 h** |

This number is a SICKER-PRICE upper bound, not a plan — K=128 alone (A+B,
1,755.9 h) is 79% of it. It is roughly 2.55× the pre-fix design's 875.5h
sticker price — larger, not smaller, because `h(K)=2K` (the A1.1 fix)
makes BOTH `F` and `NS` grow, not just `NS`; this strengthens, not
weakens, the case for §5/§7's calibration gating: if the compute-bound
assumption is even roughly right, K=128 could eat this entire wave's
budget several times over. **Planning cap for the COMMITTED sweep
(post-calibration): 150 GPU-h**, chosen as roughly 2× the prior
ortho-write wave's actual spend (≈77 GPU-h, 24 cells) to cover the
ladder's added scope — an adjustable planning number, not an external
constraint.

**Pinned trim order (Rev 1, A1.6 fix — re-derived to ACTUALLY reach the
150 GPU-h cap under the new, larger worst-case pricing above; the pre-fix
trim order provably bottomed out at 163.4h, still over cap — this version
is verified below to land under it):**
1. Drop Part B entirely (both K=48 and K=128) — worst case 1,114.08h, the
   single largest cost block; already provisionally deferred at K=128
   (§5) and now extended to K=48 too under this cap-enforcement branch
   (the calibration-corrected, likely-cheaper regime may restore it —
   see the contingency note above; this drop applies only when the
   worst-case pricing must be honored).
2. Drop Part A at K=128 entirely (worst case 698.35h at n=4; even n=1 is
   174.59h, which alone exceeds the ≈24.6h of headroom left after the
   floor — no partial trim reaches feasibility, so the sweep cell is
   dropped entirely). The Stage-0 calibration cell (§5) is NOT this cell
   — Stage-0 is a SEPARATE, independently budget-bounded item (≤24h by
   its own abort trigger, §7) that still runs regardless of this trim,
   because it is the mandatory gate, not a sweep commitment.
3. Drop Part A at K=96 entirely (worst case 295.46h at n=4; even n=1 is
   73.87h, still exceeding the remaining headroom after Stage-0 + floor).
4. **Floor, never trimmed:** Part A at K=48 and K=64, n=4 = **125.41h**,
   IN-CAP. **125.41h ≤ 150h cap, a real 24.59h margin** (A2.3 fix, Rev 2
   — REPLACES the pre-Rev-2 "≤149.41h ≤ 150h cap, 0.59h margin" framing,
   which double-counted Stage-0 INSIDE the cap in this one sentence while
   the surrounding text, §5, and the §R1 changelog all stated it OUTSIDE
   — see the single cap-accounting table below, now the ONLY place this
   number is computed). Stage-0 (**≤24h** single-seed nominal, up to
   **≤48h** two-seed contingent worst case, §5) is priced and tracked
   SEPARATELY, out-of-cap, alongside the `d=1.25K` fallback and the
   per-K rate-probes — never added into the 150h figure anywhere else in
   this document. This is the wave's minimum viable, cap-VERIFIED
   deliverable under the pessimistic (compute-bound) worst case.

**THE cap-accounting table (A2.3 fix, Rev 2 — the ONE place this document
computes cap vs. exposure; every other in-cap/out-of-cap mention in §4/§5
points here rather than re-deriving its own number):**

| Component | Convention | Worst-case hours |
|---|---|---|
| Committed sweep (floor: K48+K64 Part A, n=4) | **IN-CAP** | 125.41 h |
| Committed-sweep cap | — | ≤150 h |
| **Committed-sweep margin** | — | **24.59 h** |
| Stage-0 calibration, completed-run branch (1-2 seeds, §5 Phase 0b) | OUT-OF-CAP | ≤48 h |
| Stage-0 calibration, ABORTED-ON-COST branch (§5 Phase 0a — a cheaper alternative outcome of the SAME cell, not additive with the row above) | OUT-OF-CAP | ≈0.15-1.1 h |
| `d=1.25K` diagnostic fallback (CONFIRMED FAIL only, §2/§5, PI-gated) | OUT-OF-CAP | ≈227.6 h |
| Per-K 500-step rate-probes (CONFIRMED FAIL branch only) | OUT-OF-CAP | ≈0.05 h |
| **Worst-case out-of-cap exposure** (Stage-0 completed-run branch + fallback + probes; excludes the cheaper ABORTED-ON-COST alternative, see below) | — | **≈275.65 h** |
| **WORST-CASE PROGRAM TOTAL** (committed cap + out-of-cap exposure) | — | **≈150 + 275.65 ≈ 425.65 h** |

**Read this honestly, both numbers.** (1) The committed-sweep spend is
**125.41h in-cap, 24.59h margin under the 150h cap** — under the
worst-case pricing model, that cap buys almost nothing past the two
nearest rungs; K=96 and K=128 are entirely priced out of the committed
sweep if the compute-bound assumption holds. This is a much harder cut
than the pre-fix trim order implied (which incorrectly suggested n=1
confirmatory cells at K=96/128 were affordable — A1.6 caught this). (2)
The disclosed worst-case PROGRAM total, if every contingent branch fires
along its most expensive path, is **≈425.65h** — dominated by the
`d=1.25K` fallback (≈227.6h), itself PI-gated and not authorized by
default. The ABORTED-ON-COST row is strictly CHEAPER than the
completed-run row it is compared against (≈2h vs ≤48h); it is listed for
completeness, not summed into the worst-case total, since the two are
mutually exclusive outcomes of the SAME Stage-0 cell and the disclosed
bound must use the more expensive of the two, not both. The escape hatch
for the committed-sweep number is exactly what §5 already builds in:
Stage-0's REALIZED rate (not this nominal ceiling) reprices the grid
before any commitment past the floor, and the project's own established
precedent (`NOVEL_ARCH_WATERFALL.md` §9.10/§11: measured rates FLAT
within noise despite up to 2.1× FLOP spread at small K) argues the true
cost likely lands well under this sticker price — but that argument now
has to survive a 62× FLOP spread (not 2×), a regime it has never been
tested against, so it is not something to bank the K=96/K=128 rungs on
without Stage-0's confirmation.

---

## §5 CALIBRATION-CELL-FIRST PLAN (mandatory, before any sweep cell launches)

CLAUDE.md's calibration rule exists for exactly two failure modes this
ladder is at real risk of: (a) a convergence ceiling — Gate-0 silently
capping below 0.9 at larger K under an unvalidated `d=K+1`, now correctly
understood as a `d`-cap FILLABILITY question (A2.1, Rev 2 — `h(K)=2K`
supplies encoder CAPACITY toward this, not a raised ceiling; §2's fill
table: 0.882-0.893 fill required at K=48-128, vs 0.873 validated at K=32
— real, precedented, but thin margin, not the "comfortable" one Rev 1
claimed), and (b) a "bigger model" guess that diverges or blows the
wall-clock budget before a sweep's compute is committed (§4's ≈2,233h
sticker-price risk — bigger than the pre-A1.1-fix 875.5h because closing
A1.1 makes `F` and `NS` grow together, §2). A THIRD, ortho-write-specific
risk belongs here too: **NS convergence quality is untested past d=33.**
`n_iter=40` was calibrated to fully orthogonalize a "measured cond#≈8500
at random init" case at the K=24/32 build; condition numbers of random
d×d matrices grow with d, so `n_iter=40` might not fully orthogonalize
the K=128/d=129 write — a failure mode invisible to Gate-0 (training
could still converge) but fatal to the far-depth claim (an imperfectly
orthogonal operator still decays weak modes, just more slowly). **Rev 1
counted a FOURTH risk ("does h(K)=2K let SGD fill the rank ceiling")
separately from (a); Rev 2 retracts that split (A2.1 fix):** `h` cannot
move the ceiling past `d` either way, so "h(K)-capacity-sufficiency" and
"relative-headroom convergence" are the SAME risk, read off the SAME two
instruments (Gate-0 + `orthogonality_error`) below — not a
Gate-0-alone-separable fourth one.

**Stage 0 (mandatory, blocks everything else): ONE calibration cell,
K=128 (the top target config, deliberately the hardest, not the
cheapest), Part A ortho arm, gated by a TWO-PHASE check — an early cost
gate, THEN (only if it clears) a full run to convergence or its own
health-based abort — BEFORE any other cell in this ladder launches,
packed or not.**

**Phase 0a — early cost gate (A2.2 fix, Rev 2 — makes Stage-0 explicitly
USE §7's own early-step-rate projection trigger, already pinned there,
and adds the decision branch §7 never specified for Stage-0 itself).**
§7 already pins the mechanism generally: after the first 2,000 of
320,000 steps (<1% of budget), extrapolate total wall-clock linearly
from the measured per-step rate,
`projected_total_h = (elapsed_h_at_step_2000 / 2000) × 320000`, and abort
if it exceeds **24h — Stage-0's own pinned, flat threshold** (distinct
from the `packed_ceiling`-based trigger sweep cells use, §7). Rev 1
defined this trigger but described Stage-0's OWN deliverables (Phase 0b
below) as though a full run to convergence were the only outcome, and
never stated what happens when the trigger fires FOR STAGE-0
SPECIFICALLY — the gap A2.2 found: under the compute-bound outcome
Stage-0 exists to test (nominal 174.59h, §4), a flat "wait 24h" reading
would burn the entire budget having completed only ~0.6% of training,
delivering the rate (deliverable 1) and nothing else, with no branch to
act on it. Evaluating the SAME 24h number as an early PROJECTION at step
2,000 (not a waiting period) catches this in ≈0.15-1.1h instead of a
full day, and feeds a real decision branch:
- **`projected_total_h ≤ 24h`** → proceed to Phase 0b (full run). Itself
  informative: K=128 is tracking closer to the overhead-bound small-K
  precedent than the worst-case FLOP-ratio estimate — deliverable 1
  (realized rate) is already trending favorable.
- **`projected_total_h > 24h` → ABORTED-ON-COST (NEW, first-class
  Stage-0 outcome, A2.2 fix).** Stage-0 terminates immediately (having
  spent ≈0.15-1.1h, not 24h). This IS deliverable 1, answered in the
  compute-bound direction: K=128 (and, by the §2 ratio table, K=96 too)
  are confirmed too expensive to calibrate directly, let alone sweep.
  **Decision:** K=96 and K=128 are declared priced out of the committed
  sweep (consistent with — and CONFIRMING, not contradicting — §4's own
  worst-case trim order, which already drops both under the
  compute-bound assumption). Proceed straight to the floor (K=48 then
  K=64, §4) via the SAME staged escalation as the PASS branch below,
  K48-first. **Stage-0's two remaining deliverables (Gate-0 pass/fail —
  which IS `d`-cap fillability under `h(K)=2K`, one risk not two, per
  §A2.1's merge — and orthogonality-at-convergence) are
  REASSIGNED to the FIRST FLOOR CELL, K=48** — stated explicitly: K=48's
  own Gate-0 + `orthogonality_error` readout becomes the gate for
  advancing to K=64 ONLY (the staged-escalation rule already required
  K=48's real numbers before committing K=64; this branch does not
  weaken that, it removes K=128 as the source of those numbers). **The
  ladder's upper half (K=96, K=128) is OUT OF THIS WAVE'S SCOPE under
  this branch and needs its OWN future calibration gate** (a fresh
  Stage-0-equivalent, priced and scheduled separately, PI-gated like the
  `d=1.25K` fallback below) before any future wave revisits them —
  nothing in a floor-only sweep validates `d`-cap-fillability or
  NS-convergence at d≥97.

**Phase 0b (full run, reached only if Phase 0a clears) answers THREE
things (Rev 2: not four — A2.1's retraction folds the old 4th risk into
the 1st, below):**
1. **Real wall-clock** → computes a realized-vs-predicted ratio against
   the FLOP-scaled **174.59h estimate (§4)**, which corrects the pricing
   for K=48/64/96 too (interpolated, not re-derived from scratch — same
   qualitative regime assumed unless Stage 0 says otherwise).
2. **Gate-0 pass/fail = `d`-cap fillability, ONE risk not two (A2.1
   fix)** → catches whether the encoder, given `h(K)=2K`'s extra
   capacity, actually fills the razor-thin `d=K+1` cap (0.893 fill
   required at K=128, §2) — a Gate-0 failure at K=128 can no longer be
   blamed on the old fixed-h rank cap (A1.1 is closed by construction:
   the ceiling is structurally clearable), narrowing the diagnosis to
   genuine SGD-trainability / capacity-fillability questions, which
   Gate-0 alone cannot further decompose (§2's honesty note).
3. **Orthogonality quality at convergence, EXACT tolerance (A1.4 fix):**
   measure `orthogonality_error` (`ncr_ortho_write.py:144-151`, the
   scale-normalized `‖QᵀQ/scale − I‖_F`, `scale` = mean diagonal of
   `QᵀQ` ≈ mean `σ²`) at the FINAL checkpoint, `.max()` over the eval
   batch (not `.mean()` — a single badly-orthogonalized item is a real
   problem for ITS far-depth read even if the batch mean looks fine,
   mirroring the code's own `t2`/`t4` self-test convention,
   `ncr_ortho_write.py:719,747`, A2.7 fix). **Pinned bands (not vibes):**
   - `≤ 1e-2` → HEALTHY, matches the code's own `t2`/`t4` self-test bound
     for "tightly orthogonalized" (the pre-existing, already-audited
     criterion — reused, not invented).
   - `1e-2 < value ≲ 1` (order-of-magnitude, not machine precision) →
     MARGINAL — plausibly an `n_iter=40` insufficiency at this `d`
     (candidate fix: raise `n_iter`, a science-parameter change requiring
     its own pre-registration amendment, not a silent bump).
   - `≳ 1` (order unity or above; the coordinator's confirmed A1.1 estimate
     for a rank-deficient write is **≈11**) → CATASTROPHIC / rank-
     deficiency signature, NOT an iteration-count problem — a zero
     singular value is a fixed point of the NS map `x←1.5x−0.5x³` (§A1.1),
     so raising `n_iter` CANNOT fix this band; it means the encoder
     failed to fill the `d`-cap at these larger `K`/`d` (should not occur
     under `h=2K`'s capacity margin, §2, but Stage-0 is exactly the check
     that confirms it).
   - **ABORT trigger:** `orthogonality_error.max() > 1e-2` at the final
     checkpoint is the exact, frozen abort condition (replaces "materially
     non-zero"). Which remediation applies is read off the band above.
   - **Forced-fail negative test (A1.4's teeth requirement, to add to the
     CPU self-test suite BEFORE Stage-0 launches, mirroring the existing
     `t1`-`t5` kill-proofs in `ncr_ortho_write.py`):** construct a
     SYNTHETIC `Z` with a KNOWN rank deficiency matching A1.1's exact
     failure mode — e.g. a `d=129` matrix built as `Z = U @ V` with
     `U:(129,65), V:(65,129)` (rank ≤65, the OLD broken `h=64` ceiling,
     deliberately reused as the adversarial case since it's a REAL failure
     mode this design closed, not a strawman) — run it through
     `newton_schulz_polar(Z, n_iter=40, n_power=12)` and assert
     `orthogonality_error(Q).max() > 1e-2` (expect ≈11, matching the
     coordinator's hand-derived estimate) EVEN AFTER full `n_iter=40`.
     This proves the abort trigger fires on the exact pathology A1.1
     identified, closing the loop between the two findings, and that
     raising `n_iter` inside this same negative test does NOT drive the
     error below `1e-2` (confirming the "NOT an iteration-count problem"
     band is real, not asserted).

**Decision rule after Phase 0b (full run only — Phase 0a's ABORTED-ON-COST
branch has its own, separate decision, already stated above):**
- **PASS** (Gate-0 holds, orthogonality tight, wall-clock finite and
  reported): re-price the full grid using the realized rate (not the FLOP
  ratio), apply §4's trim order against the 150h cap using REAL numbers,
  launch the committed sweep K=48→64→96→128 in that order (nearest-first,
  not all at once — see next bullet). **A single-seed PASS is provisional
  (A2.6 fix, Rev 2): the same seed-variance precedent that motivates the
  2nd-seed-on-FAIL rule below implies a single seed can also LUCKILY
  pass. Treat it as PASS for planning purposes, but K=48's own n=4 Gate-0
  result — the very next thing that runs, per the staged escalation
  immediately below — is what actually CONFIRMS it before K=64/96/128 see
  further commitment. The staged escalation is the containment for a
  false PASS, not a second calibration seed.**
- **STAGED escalation, not a flat launch.** Even after PASS, commit K=48
  first (cheapest, closest to the validated K=32 regime), confirm its own
  real rate and Gate-0 result, THEN advance to K=64, etc. — each rung's
  real numbers refine the next rung's price before it is committed. This
  costs a little wall-clock in sequencing but removes the single biggest
  risk in this design (a blind flat commit against a possibly-wrong cost
  model), AND (Rev 2) is the same mechanism that catches a false Stage-0
  PASS or absorbs an ABORTED-ON-COST reassignment — one staging rule
  serves both roles.
- **APPARENT FAIL (single seed) → confirm before acting (A1.10 fix):** if
  seed 0's Gate-0 is dead OR `orthogonality_error` is not tight (and
  Phase 0a cleared for that seed — an ABORTED-ON-COST seed never reaches
  a Gate-0 verdict, so it cannot be an APPARENT FAIL either), do NOT
  immediately declare FAIL and redesign. The project's own seed-variance
  precedent (`STATE.md` §1.40: "one fresh seed cleared the bar" after
  prior seeds hadn't) is explicit that single-seed trainability reads can
  flip. Run ONE additional calibration seed (same K=128 cell, seed 1) —
  contingent cost, only spent if seed 0 fails AND Phase 0a cleared for it,
  ≤24h more (bounded by the same Phase-0a-style projection), i.e.
  worst-case Stage-0 totals ≤48h across both seeds, NOT drawn from the
  150h committed-sweep cap (out-of-cap, per §4's unified cap table).
  **Only declare CONFIRMED FAIL if BOTH seeds fail** (Gate-0 dead or
  orthogonality not tight in 2/2). If seed 1 PASSES where seed 0 failed,
  treat the cell as a PASS for branching purposes (subject to the same
  A2.6 provisional-PASS caveat above) but flag the seed-sensitivity
  explicitly in the writeup — do not silently discard the failed seed.
- **CONFIRMED FAIL (2/2 seeds)**: STOP. Do not launch K=96/K=128 under
  `d=K+1`. Escalate per §2's pre-registered fallback (retest K=128 once at
  `d=1.25K`) — an explicit DIAGNOSTIC, not a presumed rescue, and PRICED
  at ≈227.6h single-seed worst case (§2, A1.9) — itself PI-gated,
  out-of-band from the 150h cap (§4's unified cap table), not a silent
  in-budget retry — before concluding anything about the ladder's upper
  end. K=48/K=64 may still be launched independently (they are closer to
  the validated regime and the calibration failure may be K=128-specific,
  e.g., the shrinking relative-headroom / capacity-fillability effect) —
  but ONLY after their own cheap rate-probe confirms sane per-cell cost
  (a 500-step probe per K, mirroring `NCR_NEXT_LEVER_DESIGN.md`'s own
  Phase-0a pattern, ≈0.05 GPU-h aggregate, negligible).
- **Part B (disc) at K=128 is explicitly DEFERRED, not committed, pending
  Stage 0's realized-vs-predicted ratio (or its ABORTED-ON-COST outcome,
  which answers the same question conservatively).** If Stage 0 shows
  costs tracking close to the FLOP ratio (genuinely compute-bound —
  whether confirmed via full-run PASS/FAIL or via an early
  ABORTED-ON-COST), Part B at K=128 (nominal **264.38h/cell** — heaviest
  single cell type in this design) is very likely out of this wave's
  budget and should be scoped as a separate, PI-gated follow-on rather
  than folded in here.

---

## §6 DRAFT WIN/NULL BANDS PER K + THE RECOVERY-vs-K SCALING-LAW READOUT

**Per-K bands (Rev 1: A1.2 one-arm reframe + A1.3 per-K mechanistic
thresholds + A1.7 wording fix; Rev 2: A2.4 resolves the A1.3/A1.5
coprime-gate collision, A2.5 fixes an arithmetic slip).** Gate-0
precondition unchanged and K-general: `min over h∈{1,2,3} rec@0.9 ≥0.9` AND
`mean A_eff_rank ≥0.9K` (measured against the `min(d,h+1)=d=K+1`-binding
ceiling, §2/A2.1). The bands below are GENERALIZED from
`NCR_ORTHO_WRITE.md` §4 Part A's K=32 bands (A1.7: not "mirrored exactly"
— the checkpoint set and the free-write clause both differ from the
frozen K=32 text, spelled out below), by K's own re-registered depths
from §3:

- **WIN(K), STILL A ONE-ARM CLAIM (A1.2 fix, unchanged by Rev 2):**
  median rec@0.9 at `h*=K+8` ≥0.9 across Gate-0-passing seeds. **GATES ON
  THE PRIMARY ONLY (A2.4 fix, Rev 2 — corrects Rev 1's dual-residue gate,
  which was internally unsatisfiable, evidence below).** The coprime
  probe `h=2K-1` (§3, residue `K-1`, A1.5) is **MANDATORY-REPORTED,
  NON-GATING**: every WIN(K)-eligible cell still measures and discloses
  rec@0.9 at `2K-1` for every seed, but its outcome does not gate
  WIN/PARTIAL/NULL — the same "reported, never chased" convention already
  applied to the synthetic `h_star=8K-3` checkpoint (§3's own disclaimed
  column). **Why gating was wrong:** `2K-1 > K+8` for all K≥10 — the
  coprime probe is strictly DEEPER than the primary, not a lateral
  cross-check — so a gate on it demands a `min|λ|` bar tighter than the
  one pinned for the primary. An operator that exactly meets the
  primary's pinned bar (table below) provably FAILS rec@0.9 at `2K-1` at
  every rung in this ladder (weak-mode residual 3.7e-4–7.9e-4 vs the
  0.0148 floor, off by 19-40×, reported table below), and the K=32 anchor
  itself does not clear a `2K-1` gate (in-silico front ≈51.4 < 63 =
  2·32-1, `NCR_ORTHO_WRITE.md` §5) — so gating WIN(K) on the coprime probe
  would make the band unsatisfiable even by the cell this ladder is meant
  to extend (A2.4). Rev 2 takes the coordinator's default
  (§A2-ADJUDICATION (d)) rather than re-pinning a deeper
  `floor^(1/(2K-1))` bar and keeping it gating — that alternative remains
  available but requires showing the K=32 anchor clears it first, which
  it does not. The coprime probe still does real work as a REPORTED
  cross-check: if it independently clears rec@0.9 at some rung, that
  rules out a `gcd=8`-specific numerical artifact at that rung (worth
  noting positively, per A1.5's original intent); if it does not clear
  (the mechanistically EXPECTED outcome, per the math above), that is
  itself reported, not treated as a defect or allowed to veto an
  otherwise-clean primary result. **The free-write comparison is REMOVED
  as a gating clause.** It is reported as directional context only,
  explicitly labeled: "free-write is measured DEAD at K≤32
  (`NCR_ORTHO_WRITE.md` §4); NOT measured at K>32 in this design;
  assumed-but-unverified to remain dead by extrapolation." A K>32 WIN is
  therefore a claim about the ortho arm's OWN trainability +
  physical-depth-fidelity survival, not a fresh ortho-vs-free separation
  at that K — the separation claim stays anchored at K≤32 where it is
  actually measured. **Mechanistic corroboration, NOW PER-K (A1.3 fix,
  replaces the flat K-independent constants):** departure-from-normality
  ≤0.02 and cond#≤~2 stay FLAT (properties of the operator itself —
  preconditions for the eigen-based decay formula below to be a valid
  predictor at ANY h, not quantities that compound with depth); **`min|λ|/c*
  ≥ 0.9^(40/h*(K))`, DERIVED from band arithmetic, not asserted, and GATES
  ONLY the primary `h*(K)` (A2.4, Rev 2):** the K=32 bar (`min|λ|/c*≥0.9`
  at `h*=40`) implies the weakest mode is allowed to decay to a residual
  amplitude `floor = 0.9^40 ≈ 0.0148` (per `NCR_ORTHO_WRITE.md` §1's own
  decay formula, `(min|λ|/c*)^h`) — Rev 1 pins THIS residual floor, not
  the exponent base, as the K-invariant quantity (honoring CLAUDE.md's
  instrument-relative rule: "*the n_iter-sufficiency frontier MOVES with
  K/d … never carry an admission profile derived at one K/d to another
  without re-validating*"), and re-derives the per-K exponent-base bar so
  the SAME final residual is required at every K:

  | K | h\*(K) | min\|λ\|/c\* bar (GATES WIN) | (bar)^h\*(K) check |
  |---|---|---|---|
  | 32 (ref) | 40 | 0.9000 | 0.01478 |
  | 48 | 56 | 0.9275 | 0.01478 |
  | 64 | 72 | 0.9431 | 0.01478 |
  | 96 | 104 | 0.9603 | 0.01478 |
  | 128 | 136 | 0.9695 | 0.01478 |

  The bar TIGHTENS toward 1 as K grows (more physical applications means
  each one must leak less to hold the same final residual) — the correct
  direction (a threshold flat at 0.9 would, per A1.3, leave the weak mode
  at `0.9^136 ≈ 5.98·10⁻⁷` (≈6·10⁻⁷) at K=128 — A2.5 fix, Rev 2, corrects
  Round 1's own `≈4·10⁻⁷`; order-correct, purely illustrative, the
  vacuity point is unaffected — satisfied by operators far too imperfect
  to plausibly WIN — vacuous corroboration).

  **Coprime probe's implied bar, REPORTED ONLY, NOT GATING (A2.4 fix, Rev
  2 — what `min|λ|` WOULD need to be to also clear rec@0.9 at `2K-1`,
  computed the same way, for context in the writeup, never for a
  pass/fail decision):**

  | K | primary bar (GATES) | `min\|λ\|` needed at coprime `2K-1` (REPORTED ONLY) | residual at `2K-1` if operator only meets the primary bar |
  |---|---|---|---|
  | 48 | 0.9275 | 0.9566 | 7.9e-4 (≪0.0148 — expected to fail) |
  | 64 | 0.9431 | 0.9674 | 5.9e-4 (≪0.0148 — expected to fail) |
  | 96 | 0.9603 | 0.9782 | 4.4e-4 (≪0.0148 — expected to fail) |
  | 128 | 0.9695 | 0.9836 | 3.7e-4 (≪0.0148 — expected to fail) |
- **PARTIAL(K):** median rec@0.9 ≥0.9 at the `K-3`/`2K-3` checkpoints (A1.7:
  this GENERALIZES the frozen K=32 PARTIAL band `{20 OR 29=K-3}` — it drops
  the absolute shallow-20 checkpoint and substitutes the deeper `2K-3`, a
  deliberate choice for the K-ladder's own audited rungs, NOT a literal
  mirror) but <0.9 at `h*=K+8`.
- **NULL(K):** rec@0.9 <0.9 at every depth ≥ `K-3`, Gate-0 still passes —
  the write trains but buys no far-depth gain at this K.
- **FAIL(K):** Gate-0 DEAD in ≥3/4 seeds — the constraint (or the untested
  `d=K+1`/`h(K)=2K` convention at this K) breaks trainability.

**The scaling-law readout — three pre-registered shapes, decided before
data, not after (A1.5 fix: reworded from "compositional depth
generalization" to what this ladder actually measures):**
- **(a) FLAT-HOLD:** WIN at all four K (gated on the primary `h*=K+8`
  alone, A2.4 fix, Rev 2). Publishable as "the orthogonal write's
  realistic PHYSICAL-APPLICATION-FIDELITY ceiling is K-general" — NOT
  "compositional depth generalization" (A1.5: every checkpoint's
  EFFECTIVE relational depth, `h mod K`, stays fixed and low — `8` for
  the primary; the ladder measures whether many PHYSICAL applications of
  a K-relative operator preserve a fixed shallow target, not whether
  deeper RELATIONAL composition generalizes). **The coprime probe
  (`2K-1`, residue `K-1`) is reported alongside but is NOT expected to
  also clear (A2.4): its implied `min|λ|` requirement is strictly tighter
  than the primary's pinned bar at every rung (table above), so a
  primary-only FLAT-HOLD with a non-clearing coprime probe is the
  MATH-CONSISTENT expected pattern, not a partial or weaker result — only
  a SURPRISE clearance of the coprime probe adds the stronger "across two
  independent residue classes" claim, and that should be reported as a
  bonus finding if it happens, not assumed going in.** Still the
  strongest possible primary-only outcome, no evidence of a new wall
  inside this ladder's range.
- **(b) GRACEFUL DECAY:** WIN/PARTIAL at K=48/64, degrading toward
  NULL by K=96/128, with `rec@h*(K)` roughly monotonically falling. Report
  the fitted trend (simple monotonic fit against K or log K — four points is
  enough to characterize direction and rough rate, not enough for a tight
  power-law exponent) and the point estimate for where `rec@h*` crosses back
  under 0.9 — i.e., report a NEW, higher K-wall location, not just "it got
  worse."
- **(c) CLIFF:** WIN holds at K=48 (and maybe 64), then a sharp drop to
  NULL/FAIL at a specific rung. Cross-check against the mechanistic
  diagnostics: if the cliff co-occurs with `orthogonality_error` going
  non-tight (NS not fully converging at that d, §5's third risk), it is an
  NS-iteration-count problem (plausibly fixable); if Gate-0 itself fails
  with clean orthogonality, it is a genuine SGD-trainability boundary
  (harder, matches the free-write K-axis wall's own character).

All three outcomes are informative and none require a follow-up experiment
to be reportable — this is itself a discipline point (mirrors this
project's own "negative results are data" rule).

---

## §7 ABORT CRITERIA + CEILING PRICING (headroom from the measured WORST case)

**Two DISTINCT thresholds, not one** — the amendment this design makes over
the prior ceiling-mispricing incident (3.0h→6.0h, §CEILING AMENDMENT):
a **budget-planning ceiling** (how much to reserve) is not the same number
as a **runaway-abort trigger** (when to kill a hung/stuck cell), and
conflating them is exactly what under-priced the last one.

**Budget-planning ceiling per cell**, priced from the measured WORST case
(4.24h, the discriminator rate — used as the base even for Part A cells'
ceiling, per instruction to price off worst-case not mean) scaled by the
§2 FLOP ratio (Rev 1: ratios recomputed under `h(K)=2K` — LARGER than the
pre-fix ratios, the direct cost of closing A1.1, §2), with a 2× margin
(tighter margins were tried before and under-priced; this design
deliberately doubles the prior 1.4-1.5× margin given the ladder reaches
untested K):

`ceiling(K) = 2 × ratio(K) × 4.24 h`

| K | ceiling (solo) |
|---|---|
| 48 | 28.3 h |
| 64 | 66.7 h |
| 96 | 223.7 h |
| 128 | 528.8 h |

A 529h SOLO ceiling for one cell is not a real operating threshold — it is
a worst-case sticker price, ~2.7× the pre-fix design's 195.8h (the same
factor the A1.1 fix costs throughout this document, §2, §4). **The actual
runaway-abort trigger is an early-step-rate projection, not the ceiling
above:** after the first 2,000 of 320,000 steps (<1% of budget, mirrors
`NCR_NEXT_LEVER_DESIGN.md`'s own 500-step rate-probe discipline, cost
negligible), extrapolate total wall-clock; if the extrapolation exceeds
**24h for the Stage-0 calibration cell** (independent, flat number — NOT
derived from the ratio-scaling above; now a SMALLER fraction of the
nominal K=128 ceiling than before, `24/528.8≈4.5%` vs the pre-fix
`24/195.8≈12.3%` — even MORE conservative in relative terms, so no change
to the 24h number itself is warranted) **or 1.5× that cell's own
`packed_ceiling(K,N)` (A1.8 fix, Rev 1 — replaces `ceiling(K)`; `N` is
whatever packing factor the cell is ACTUALLY running under, §8; for the
Rev-1 default `N=1` this equals `ceiling(K)` exactly, so the fix only
bites once/if a rung is upgraded to `N=2` packing)** for any sweep cell,
ABORT immediately rather than waiting for the full ceiling to elapse. This
turns "is this cell healthy" into a decision made in minutes, not days.
**For the Stage-0 cell specifically, firing the flat-24h leg of this
trigger is a first-class outcome with its own decision branch
(ABORTED-ON-COST, §5 Phase 0a, A2.2 fix, Rev 2) — not just a kill
signal; §5 is the authority on what happens next for Stage-0, this
section only pins the mechanism.**

**Gate-0 abort (mirrors `NCR_ORTHO_WRITE.md` §4 Part A FAIL clause
verbatim):** if a K-rung's Gate-0 fails in ≥3/4 seeds, STOP that rung
immediately — do not spend the remaining seed budget chasing it with more
compute. This is a design question (§2's fallback, or a `n_iter` amendment
per §5), never a "try again with more steps" question.

**Packed-cell ceiling correction (A1.8 fix folded directly into the trigger
sentence above, Rev 1):** when N cells share a GPU, each cell's OWN
wall-clock lengthens by the contention factor (§8 estimates ~1.3× at N=2);
the per-cell abort trigger now references `packed_ceiling(K,N)` at the
SOURCE (not a separate correction paragraph a literal reader could miss,
A1.8's exact complaint) — a healthy packed cell can no longer be killed for
looking "slow" relative to a solo baseline it was never running under.

---

## §8 SATURATION-PACKING PLAN (PI directive 2026-07-16)

**Rejected baseline:** current cells draw ~44-59% SM and ~1.7GB on an
80GB H100 — both badly under-saturated, MEASURED AT THE VALIDATED `K≤32,
h=64` config. **Rev 1 re-derivation (required — larger `h` changes the
SM-utilization/memory picture, per the coordinator's steer): this measured
profile is no longer a safe anchor for ANY rung in this ladder.** Under the
pre-fix design, K=48/64 were argued to sit "near" this baseline because `h`
was held at the SAME fixed 64 as the validated regime; under `h(K)=2K`
(§2, the A1.1 fix), EVERY rung in the ladder now runs a BIGGER encoder than
anything measured (`h=96` at K=48 is already 1.5× the validated `h=64`, up
to `h=256`, 4×, at K=128) — the split this design previously drew between
"K=48/64 near-baseline, pack N=2" and "K=96/128 NS-dominated, default N=1"
was predicated on K=48/64 resembling the SMALL validated model; that
predicate is now FALSE at every rung. Compounding the memory picture: §2's
h-fix ALSO makes the base task's own FLOPs (`F(K,d,h)`) grow roughly
cubically in K now (via `76Kh²=304K³`), not just `NS(d)` — so the
"K96/128 is NS-dominated, hence higher SM%" argument for defaulting to
N=1 there is now ALSO true, independently, of the BASE task at every rung,
not a K96/128-specific phenomenon.

**Rev 1 default: N=1 (no packing) at ALL FOUR K in this ladder, pending
per-rung confirmation — no K-split by default.** This replaces the pre-fix
K48/64-vs-K96/128 split with a single conservative default, because the
premise that justified packing K48/64 (resembling the measured small-model
profile) no longer holds anywhere in this h-fixed ladder. **Upgrade path,
per rung, not assumed:** the ALREADY-SPECIFIED first-batch mini-calibration
(watch realized `nvidia-smi`/DCGM SM% and per-cell step-rate for the first
~5,000 steps of a rung's FIRST cell) now governs whether THAT rung gets
upgraded to `N=2` packing — if a rung's real solo SM% reads comfortably
below ~50% (i.e., the bigger `h` did NOT translate into materially higher
utilization, consistent with the small-K overhead-bound precedent
persisting even at bigger `h`), extend `N=2` (contention-factor
assumption unchanged, ~1.3×, to be confirmed empirically per rung) to that
rung's REMAINING cells. Stage-0's K=128 calibration cell (§5) reports the
FIRST real SM% data point under `h(K)=2K` and should be read as informative
for the whole ladder's packing prior, even though it is priced/gated
separately from the sweep.

**Contention-factor assumption (unchanged from pre-fix, stated explicitly,
to be confirmed empirically): each packed cell (where `N=2` is confirmed)
runs ~1.3× slower in wall-clock than solo**, giving a net GPU-throughput
gain of `2/1.3 ≈ 1.54×` cells-per-GPU-hour — a real saturation win WHERE
packing is confirmed safe; under the Rev 1 default (`N=1` everywhere until
confirmed), this factor does not apply anywhere at launch.

**Re-priced ceiling under packing:** `packed_ceiling(K, N) = ceiling(K) ×
contention_factor(N)`. At the Rev 1 default `N=1`: `packed_ceiling(K,1) =
ceiling(K)` exactly (§7's table, unchanged). PROSPECTIVE `N=2` values
(apply only once a rung is confirmed and upgraded, Rev 1 recomputed under
`h(K)=2K`): K=48 `28.3h×1.3=36.8h`; K=64 `66.7h×1.3=86.7h`; K=96
`223.7h×1.3=290.8h`; K=128 `528.8h×1.3=687.4h` — none in effect at launch,
listed for completeness once/if any rung is upgraded.

---

## §9 BRANCH LOGIC — WIN vs PARTIAL of the running ortho-write experiment

This design executes ONLY under one of the two conditions below; the
`NCR_ORTHO_WRITE.md` §4 FAIL/NULL bands do not license any part of this
document.

**Under WIN (Part A: `rec@0.9` at `h*=40` ≥0.9 at K=32, mechanistically
corroborated):** run the FULL K-ladder as designed above — `h*=K+8` at every
rung, the complete §4 cell grid (subject to §5's calibration gate and §4's
trim order). The ambitious per-K target (`K+8`, proportionally deeper as K
grows) is licensed because K=32 already cleared the equivalent ask.

**Under PARTIAL (Part A: `rec@0.9`≥0.9 only at `h∈{20 OR 29}`, i.e. short of
`h*=40`, "wall cracked, shallower"):** the K-32 result did NOT clear the
`K+8`-equivalent target, so committing every rung to the SAME (proportionally
harder, since `h*=K+8` grows with K) ambitious ask is premature. Two
changes from the WIN branch: (1) **re-register the per-K primary target one
notch down** — use the `K-3` checkpoint (the ladder's own audited `m=1`
rung) as the PARTIAL-branch `h*`, not `K+8`; report `K+8` as a stretch
checkpoint instead of the primary bar. (2) **shrink the committed grid to
K=48 only, n=4 seeds, Part A alone, no Part B, no Stage-0 calibration cell
needed at this reduced scope** (a single K=48 cell's Rev-1-priced cost
under `h(K)=2K`, **37.36h for n=4** — up from the pre-fix design's 23.9h,
the direct cost of the A1.1 fix, §2/§4 — is still comfortably inside the
150h cap without any grid commitment past it) — the question under
PARTIAL is narrower ("does even the SHALLOW crack survive ANY scaling past
K=32"), and answering it cheaply before spending on K=64/96/128 is the
correct sequencing.

**PARTIAL-branch band mapping (A1.11 fix, Rev 1 — was previously left
implicit; §6's bands are written for the `K+8` primary and do NOT
automatically transfer to the moved `K-3` bar):**
- **WIN (K=48, PARTIAL-branch):** median rec@0.9 at the re-registered
  primary `h=K-3=45` ≥0.9 across Gate-0-passing seeds; mechanistic
  corroboration departure-from-normality ≤0.02, cond#≤~2 (flat, unchanged
  rationale, §6), `min|λ|/c* ≥ 0.9^(40/45) ≈ 0.9106` (§6's per-K formula,
  Rev 1, evaluated at the MOVED bar `h=45` rather than `h=56`). `h=K+8=56`
  is reported as a stretch checkpoint only, not gating (per item (1)
  above); the coprime probe `2K-1=95` is likewise mandatory-reported,
  non-gating here too (§6/A2.4, Rev 2 — same convention, no PARTIAL-branch
  exception).
- **PARTIAL (K=48, PARTIAL-branch):** median rec@0.9 ≥0.9 at one of the
  shallow sanity points `{5,12,20}` but <0.9 at `K-3=45` — cracked
  shallower than even the PARTIAL-branch's own re-registered bar.
- **NULL (K=48, PARTIAL-branch):** rec@0.9 <0.9 at every depth ≥12, Gate-0
  still passes.
- **FAIL (K=48, PARTIAL-branch):** Gate-0 DEAD in ≥3/4 seeds (unchanged
  rule).

If K=48 clears its WIN bar (`K-3` primary), escalate to the WIN branch's
full ladder as a follow-on; if it doesn't, the K-ladder line closes here,
cheaply.

**Under NULL or FAIL:** this document does not execute. No cells launch.

---

## §A1 ATTACK ROUND 1 (2026-07-16, pre-build, independent)

Adversarial pre-build review. Every §2/§3/§4/§7/§8 number was recomputed from
the pinned formulas; every load-bearing citation was checked against
`NCR_ORTHO_WRITE.md`, `NOVEL_ARCH_WATERFALL.md:3899-3901`, `ncr_task.py`,
`ncr_ortho_write.py`, and `chapter2/model_v4.py`. Arithmetic is CLEAN
throughout (see "verified clean" at the end). The killers are structural.

### FATAL

**A1.1 — FATAL. The `h=64` encoder-hidden cap makes Gate-0 STRUCTURALLY
UNPASSABLE at K=96 and K=128 (and razor-thin at K=64). The write cannot
reach rank K.**
- Defective claim (§2): "*`h=64` fixed (encoder hidden unchanged across K…)*"
  and finding (1): "*Params stay trivial end to end (175K→200K, +14%) — the
  'never the constraint' precedent holds for PARAMS.*" The design treats
  `h=64` as a param-count question ONLY and never asks what it does to the
  **write rank**.
- Evidence: the write is produced by `BindingEncoder`
  (`chapter2/model_v4.py:52-63`): `self.row_out = nn.Linear(h, d)`, then
  `Z = self.row_out(q)` where `q` is `(B, d, h)`. Every one of the `d` rows of
  `Z` is the image of an `h`-vector under ONE shared `h→d` linear map, so
  **`rank(Z) ≤ h + 1 = 65`** for all K (the `+1` is the bias). Gate-0 requires
  `mean A_eff_rank ≥ 0.9·K` (design §6; enforced verbatim at
  `ncr_earlyln_scale.py:322`, `AEFF_RANK_FRAC_BAR*K`), and `A_eff_rank` is
  `effective_rank(A)` of the entity operator `A` derived from `Z`
  (`chapter2/analyze_zdump.py:166`, `= exp(entropy(σ))≤ rank`), so
  `A_eff_rank ≤ rank(Z) ≤ 65`. Then:
  - K=48 (d=49): cap `min(d,h)+1≈49`, bar `0.9·48=43.2` → feasible (d<h; no cap).
  - K=64 (d=65): cap `≈65`, bar `0.9·64=57.6` → RAZOR-THIN (needs eff-rank
    57.6 against a hard structural ceiling of 65; a soft-trained encoder rarely
    fills 89% of its rank ceiling).
  - K=96 (d=97): cap `≈65`, bar `0.9·96=86.4` → **IMPOSSIBLE (86.4 > 65)**.
  - K=128 (d=129): cap `≈65`, bar `0.9·128=115.2` → **IMPOSSIBLE (115.2 > 65)**.
- Why the precedent is silent: the validated regime (K=16/24/32) has `d ≤ 33 <
  h=64`, so the `h` cap NEVER binds there (`A_eff_rank≈31` at K=32 sits under
  `d=33`, not under `h`). The ladder is the FIRST time `d` crosses 64
  (K≥64) — exactly where the cap bites — and it does so with `h` held fixed.
  "Carried by extrapolation" (§2) is thus doubly unvalidated: the design flags
  the relative-headroom axis but MISSES the write-rank-expressivity axis.
- The design's own safety nets misdiagnose it: (a) the Stage-0 calibration cell
  is at K=128 (§5) — it will fail Gate-0 for THIS reason, and the design would
  misattribute it to relative-headroom or NS-convergence. (b) The §2 fallback
  (retest at `d=1.25K` = 160 at K=128) makes it strictly WORSE — rank still
  capped at 65, bar still 115.2, in an even larger ambient space. (c) The §5
  `orthogonality_error` abort WOULD fire (NS-polar of a rank-65 `Z` in d=129 is
  a partial isometry; scale-normalized `‖QᵀQ−I‖_F ≈ 11`, not machine
  precision), but the pre-registered remediation ("raise `n_iter`") CANNOT fix
  it: a zero singular value is a fixed point of the NS map `x←1.5x−0.5x³`, so no
  number of iterations orthogonalizes a rank-deficient write.
- Minimal fix: `h` (encoder hidden) MUST scale with K so `h ≳ d = K+1` (e.g.
  `h = max(64, K+ margin)`), which is what actually lets `row_out` write rank-K.
  This is NOT a patch: the `40h²` param term then grows as K² (invalidating the
  §2 "params trivial" finding and the entire §2/§4/§7/§8 cost model), and the
  FLOP/NS tables must be recomputed. BUILD-BLOCKED until `h(K)` is redesigned
  and every downstream table redone. (Alternatively cap the ladder at K≤48 —
  the only rung whose `d<h` keeps the write full-rank in the validated regime —
  but that abandons the scaling-law premise.)

### MAJOR

**A1.2 — MAJOR. The free-write "<0.5 at same depth" WIN(K) clause is never
measured at any new K, and the cited precedent is mischaracterized.**
- Defective text (§4): "*the free-write baseline is NOT re-run at each new K; it
  is already established DEAD at K=24/32 …, §4's own pinned baseline-reuse
  precedent*", carried into §6 WIN(K): "*free-write baseline (extrapolated-dead
  …, not freshly re-measured) stays <0.5 at the same depth*".
- Evidence: `NCR_ORTHO_WRITE.md` §4 pins the K=32 free clause as **measured**
  ("*reads < 0.5 at h=40 (measured; … re-analyzed on the identical realistic
  ladder from the archived z-dumps, CPU, free)*", §4 + the "Pinned baselines"
  block). That precedent REUSES AN EXISTING MEASUREMENT (a K=32 z-dump that
  physically exists). No free-write z-dump exists at K∈{48,64,96,128} — there is
  nothing to re-analyze — so the design is not "reusing a measurement," it is
  EXTRAPOLATING across K and calling it reuse. The WIN(K) band therefore
  contains a gate clause that can never be checked, so WIN(K) collapses to a
  single-arm ortho claim — which directly undercuts the PI's capability-
  SEPARATION headline (a separation claim needs both arms AT that K).
- Minimal fix: measure a free-write anchor at ≥1 new K (minimum: K=48, the floor
  cell; a fresh free run is cheap relative to the wave), OR restate WIN(K) as an
  explicit single-arm claim with the free comparison labeled
  "extrapolated-from-K≤32, NOT measured at this K." Do not pre-register an
  unverifiable clause inside a WIN band.

**A1.3 — MAJOR. The mechanistic corroboration thresholds are asserted
K-independent, but their far-depth meaning compounds over `h*=K+8` and goes
vacuous at large K — contra the CLAUDE.md instrument-relative rule.**
- Defective text (§6 WIN(K)): "*mechanistic corroboration unchanged
  (departure-from-normality ≤0.02, cond#≤~2, min|λ|/c\*≥0.9) — these thresholds
  … are not themselves K-dependent*", with §6 also claiming the bands "*mirror
  `NCR_ORTHO_WRITE.md` §4 Part A exactly*".
- Evidence: the far-depth recovery these numbers corroborate decays as
  `(min|λ|/c*)^h` (`NCR_ORTHO_WRITE.md` §1). The SAME threshold `min|λ|/c*=0.9`
  leaves the weak mode at `0.9^40 = 0.015` at K=32 (h*=40) but `0.9^136 = 4·10⁻⁷`
  at K=128 (h*=136) — total annihilation. So a threshold that is meaningful
  corroboration at K=32 is satisfied by operators far too imperfect to win at
  K=128; the corroboration has NO discriminating power at large K. CLAUDE.md's
  hard rule is explicit: "*the … n_iter-sufficiency frontier MOVES with K/d …
  Never carry an admission profile derived at one K/d to another without
  re-validating.*" Asserting flat K-independence violates it.
- Minimal fix: tie corroboration tightness to `h*` (require
  `(min|λ|/c*)^(K+8) ≥` a fixed fidelity, likewise for cond#/departure), OR
  explicitly demote the three numbers to non-binding sanity checks and state
  the behavioral `rec@0.9` is the SOLE gate. Either way, drop the false
  "not K-dependent" and "mirror exactly" wording.

**A1.4 — MAJOR. The Stage-0 orthogonality abort trigger is a pre-registered
decision gate with no pinned threshold (adjudication-by-vibes).**
- Defective text (§5): "*Target: `‖QᵀQ-I‖_F` at machine-precision-ish … a
  **materially non-zero** residual at K=128 is an explicit ABORT-and-redesign
  trigger*".
- Evidence: "machine-precision-ish" and "materially non-zero" are not numbers.
  `orthogonality_error` is a scale-NORMALIZED Frobenius residual
  (`ncr_ortho_write.py:144-151`) that is never exactly zero and (per A1.1) reads
  ≈11 under rank-deficiency and O(1e-2) under mild NS under-convergence — utterly
  different regimes that "materially non-zero" cannot distinguish. Pre-
  registration discipline (the doc's own record-first framing; the frozen §4
  bands) requires a frozen number for any abort gate.
- Minimal fix: pin an exact numeric tolerance (e.g. scale-normalized
  `orthogonality_error ≤ 1e-2` per the code's own self-test t4 bound, or the
  realized K=32 ortho value once available) AND a forced-fail negative test, per
  the CLAUDE.md "run the negative unit test that proves the check has teeth"
  rule.

**A1.5 — MAJOR. Residue-8 monoculture: the primary bar has FIXED effective
relational depth at every K, so §6's "K-general depth composition" readout
overstates what the ladder measures.**
- Defective framing (§1 / §6a): "*one full relational cycle plus a novel
  residue*" and "*the orthogonal write's realistic-depth composition ceiling is
  K-general*".
- Evidence: every primary `h*=K+8` has residue 8 (design §3), so the effective
  target is `π^8` at EVERY K; and for a faithfully-learned orthogonal operator
  `Z^K≈I`, so `Z^(K+8)≈Z^8` — the "+K" contributes nothing but a physical-
  application stress. The full realistic ladder's forward effective depths are
  `{5,8,12,20}` (fixed, K-independent) plus `π^{-3}` (near=K-3 AND stretch=2K-3
  both `≡ -3 mod K`). No ladder point probes a forward effective depth that
  GROWS with K. So the "scaling law" measures orthogonality-survival over `K+8`
  physical applications with EFFECTIVE relational depth pinned at ≤20 — not
  deeper compositional generalization. (Aggravator: residue 8 has identical
  special structure `gcd(8,K)=8 ⇒ 8 sub-cycles` at every ladder K, so a
  residue-8-specific effect would masquerade identically across all four rungs.)
- Minimal fix: add a novel primary checkpoint whose effective (mod-K) depth
  GROWS with K and whose gcd-with-K varies (e.g. a residue coprime to K giving a
  single long cycle), OR restate the §1/§6 readout precisely as
  "physical-application-fidelity scaling," dropping "compositional depth
  generalization" at the primary. (Note: this monoculture is inherited from the
  frozen K=32 pre-reg, so the fix is claim-scoping, not a build blocker on its
  own — but the ladder AMPLIFIES a single-K confound into a four-point "law.")

**A1.6 — MAJOR. The pinned trim order provably cannot reach the 150 GPU-h cap
in the compute-bound worst case it is built for.**
- Defective claim (§4): "*Pinned trim order if the calibration-corrected
  projection still exceeds the cap*" (steps 1–5), against "*Planning cap … 150
  GPU-h*".
- Evidence (recomputed on the design's own §4 nominal table): applying ALL four
  trims — drop B@K128 (−391.6), K96 A n4→n1 (−90.4), drop B@K48 (−36.2), K128 A
  n4→n1 (−194.0) — leaves `K48 A 23.9 + K64 A 44.7 + K96 A n1 30.13 + K128 A n1
  64.65 = 163.4 h`, still **> 150 h**. Step 5 is the floor (K48+K64 only), so
  K96/K128 A at n1 are neither trimmable-to-zero nor floor — the trim order
  bottoms out at 163.4 h and cannot satisfy its own cap. (The floor deliverable,
  68.6 h, IS protected — that part is clean.)
- Minimal fix: add explicit "drop K96 A entirely / drop K128 A entirely" steps
  before the floor, OR raise the cap to ≥165 h, OR state plainly that the cap
  binds only under the overhead-bound (calibration-corrected) regime and the
  compute-bound worst case is PI-gated, not trim-order-satisfiable. (Real-world
  stakes are low — uptime-metered grant, cap is "adjustable" — but the stated
  mechanism is internally inconsistent.)

### MINOR

**A1.7 — MINOR. §6 claims PARTIAL(K) "mirror[s] §4 Part A exactly," but the
checkpoint set differs.** Frozen K=32 PARTIAL (`NCR_ORTHO_WRITE.md` §4) is
"`h∈{20 OR 29=K-3}`"; design §6 PARTIAL(K) is "`K-3 / 2K-3`" — it drops the
shallow-20 checkpoint and substitutes the deeper 2K-3. A silent re-derivation of
a frozen band. Fix: say "generalized (20→2K-3)," not "exactly."

**A1.8 — MINOR. §7 runaway-abort formula uses the SOLO ceiling for packed
cells.** §7 states the trigger as "1.5× that cell's own `ceiling(K)`" while the
same section's packed-correction paragraph and §8 require the CONTENDED
projection — a literal reader kills healthy packed cells. Fix: write
"1.5× `packed_ceiling(K,N)`" for packed cells.

**A1.9 — MINOR. The `d=1.25K` fallback is under-motivated by, and mildly in
tension with, the only d-variation evidence in the cited source.** It is
well-posed (`K≤d` holds at d=60/80/120/160, task defined — attack-surface-2's
narrow question passes), BUT `NCR_ORTHO_WRITE.md` §5 shows the MORE-headroom
(2K) convention was catastrophically ill-conditioned (K24@d48 cond#≈2951,
"polar barely helps") vs the K+1 tight-spare being "far healthier." Moving
toward 2K to rescue trainability is thus a DIAGNOSTIC ("does headroom matter"),
not a presumed rescue, and its own cost is un-repriced (`NS(160)≈656M` at K=128,
~1.9× `NS(129)`). Fix: reframe as a diagnostic and cite the §5 cautionary point.

**A1.10 — MINOR. Single-seed Stage-0 drives a branch/redesign decision despite
documented high trainability seed-variance.** §5's FAIL clause (Gate-0 dead at
K=128 → don't launch upper ladder / go to `d=1.25K`) fires on ONE seed, but the
project's own record (`CLAUDE.md` / STATE §1.40: "one fresh seed cleared the
bar") warns trainability is seed-variance-heavy. Fix: require a 2nd calibration
seed before the FAIL→redesign branch fires (cheap; one extra cell).

**A1.11 — MINOR. §9 PARTIAL branch and §6 bands are not fully reconciled.** §9
moves the primary to `K-3` under PARTIAL, but §6's WIN(K)/PARTIAL(K)/NULL(K)
bands are written only for the `K+8` primary; the branch does not restate the
full band set at the moved bar. Fix: give the PARTIAL-branch band mapping
explicitly.

### Verified CLEAN (attack surfaces exercised, no defect found)
- **All §2 tables** (params `166784+257d`; FLOP `311296K+16384d+768K²+256Kd+256d²`
  at h=64; `NS=160d³+48d²`; F+NS; ratio-vs-K32; `1/d`; K/d; memory `320·d²`/ex ×256):
  every cell recomputed EXACT, including `NS(33)=5,802,192`, `P(33,64)=175,265`.
- **§3 mod-K residue table**: all residues recomputed EXACT (near=K-3, h*≡8,
  stretch=2K-3, synthetic=8K-3); all `∉{0,1,2,3}` verified; `h*/3` multiples
  (18.7×/24×/34.7×/45.3×) EXACT. Matches `ncr_task._gen_grid` (ladder_residue=K-3,
  ladder=m·K-3, h_star=8K-3) and GRIDS is pre-defined for all four K.
- **§4 pricing**: per-cell (5.98/9.06/11.18/16.93/30.13/45.63/64.65/97.90 h),
  subtotals (A 447.7 / B 427.8 / total 875.5), K128=650.2h=74% — all EXACT.
- **§7 ceiling** (`2·ratio·4.24`: 18.1/33.9/91.3/195.8) and **§8 packed**
  (×1.3: 23.5/44.1; `2/1.3=1.54×`) — EXACT.
- **Attack-surface-8 (disc-rate ceiling distorting trim order)**: does NOT hold.
  The trim order (§4) prices A at 2.8h and B at 4.24h correctly; the disc-rate
  +2× conservatism lives ONLY in the §7 sticker-ceiling, which does not feed the
  trim order. (Separately flawed via A1.6, but not by double-counting.)
- **Code-claim checks**: `d=K+1` (`ncr_ortho_write.py:286`, overrides
  GRID_SHAPES d=2K), `h=64` (`:287`, `GRID_SHAPES[K]["h"]`), `NS_ITER_DEFAULT=40`
  / `NS_POWER_DEFAULT=12` (`:82-83`), `orthogonality_error` instrumented
  (`:144`) — all match the design's "reused inputs."

### VERDICT: **BUILD-BLOCKED** (A1.1 is fatal: K=96/K=128 Gate-0 is
unpassable because the fixed `h=64` write caps `rank(Z)≤65 < 0.9·K`; the
calibration cell, the `d=1.25K` fallback, and the `n_iter` remediation all fail
to address it). Re-registration required before any build: scale `h(K)` and
redo every §2/§4/§7/§8 table, then clear A1.2–A1.6.

---

## §A1-ADJUDICATION (coordinator, 2026-07-16 — recorded before dispatching Rev 1)

**A1.1 FATAL: CONFIRMED against the raw code by the coordinator directly**
(per the conflicting-claims/tiebreak rule — verified, not taken on the
attacker's word): `chapter2/model_v4.py:52` `row_out = nn.Linear(h, d)` with
`Z = row_out(q)`, `q:(B,d,h)` ⇒ `Z = QW + 1bᵀ` ⇒ `rank(Z) ≤ h+1 = 65`;
`ncr_earlyln_scale.py:75` `GRID_SHAPES` pins `h=64` for ALL K∈{14..128};
`ncr_earlyln_scale.py:322` gate requires `aer_mean ≥ 0.9·K` = 86.4 (K=96) /
115.2 (K=128) — both `> 65`, structurally unpassable. Context: `h=64 = 8K`
at the original K=8; the h/K ratio silently shrank as K grew and the
validated regime (K≤32, bar 28.8) never touched the ceiling.

**Disposition:** BUILD-BLOCKED verdict ACCEPTED. Rev 1 dispatched with:
(a) h must scale with K — coordinator steer: `h=2K` preserves the ratio at
the LAST VALIDATED rung (K=32, h=64=2K), making the rank ceiling `2K+1 ≥
0.9K` trivially and minimizing extrapolation distance (h=8K matches only the
K=8 rung and 4×-overshoots the validated ratio); Rev 1 may argue an
alternative but must justify it against this default and re-derive EVERY
§2/§4/§7/§8 cost/memory table for the chosen h(K);
(b) A1.2–A1.6 each addressed with an exact, testable change (A1.2: measured
free-write confirmatory cells n=1 per new K, or an explicit one-arm
reframing of WIN(K) — no extrapolated baseline inside a WIN clause;
A1.3: mechanistic thresholds re-derived per-K from band arithmetic, e.g.
`min|λ| ≥ floor^(1/h*(K))`, honoring the instrument-relative rule;
A1.4: exact orthogonality_error tolerance + a forced-fail negative test;
A1.5: honest reframe of effective-depth claims + at least one added
non-residue-8 novel probe depth per K; A1.6: trim order re-derived to
actually reach the 150h cap);
(c) minors A1.7–A1.11 folded in. Rev 1 output → fresh independent ATTACK
ROUND 2 (multi-round rule) before any build authorization. The design
remains CONDITIONAL on the ortho-write verdict regardless.

---

## §R1 REVISION 1 (2026-07-16)

Rev 1, dispatched per §A1-ADJUDICATION's binding disposition. Every A1.x
finding addressed below with an exact section reference; §A1 and
§A1-ADJUDICATION themselves are UNCHANGED (historical record). This design
now carries status **DRAFT-STAGE-1-REV-1 (POST-ATTACK-1, PRE-ATTACK-2)**.

| Finding | Disposition | Where fixed |
|---|---|---|
| **A1.1 (FATAL)** — fixed `h=64` makes Gate-0 unpassable at K=96/128, razor-thin at K=64 | Adopted the coordinator's default `h(K)=2K` (no competing alternative argued — justified against it instead: `h=K` reproduces the razor-thin problem forever, `h=8K` overshoots 4× with no evidence). Rank ceiling `2K+1` now clears the `0.9K` bar with a flat ~0.445-0.448 margin at every rung (vs. impossible/razor-thin before, matching K=32's own 0.443 margin). Every downstream table re-derived from the pinned formulas with `h=h(K)` substituted; validation status stated honestly (h(K)=2K is itself unvalidated past K=32, Stage-0 inherits the job). | §2 (new h(K) subsection + margin table + re-derived P/F/NS/memory table), §4 (all pricing tables + trim order), §5 (Stage-0 now validates h(K) as a 4th risk), §7 (ceiling table), §8 (packing re-derivation) |
| **A1.2** — extrapolated free-write baseline inside a WIN clause | Chose option (b): WIN(K)/PARTIAL(K)/NULL(K) at K>32 reframed as explicit ONE-ARM (ortho-only) claims; free-write cited only as unmeasured directional context, never gating. No fresh free-write cells added — priced: even n=1 at K=48 costs ≈9.4h, more than the ≈0.6h of post-trim headroom under the cap. | §4 (cell-grid paragraph), §6 (WIN(K) band), §1 (hypothesis reworded to state the one-arm scope) |
| **A1.3** — flat K-independent mechanistic thresholds go vacuous at large K | `min\|λ\|/c*` bar re-derived per-K from `bar = floor^(1/h*(K))`, `floor = 0.9^40 ≈ 0.01478` pinned from the K=32 validated bar so the SAME final residual amplitude is required at every K (table in §6, also applied to the PARTIAL-branch's moved bar in §9); departure-from-normality/cond# stay flat (operator-shape preconditions, not depth-compounding quantities) — reasoning made explicit, citing CLAUDE.md's instrument-relative rule. | §6 (WIN(K) band), §9 (PARTIAL-branch band mapping) |
| **A1.4** — orthogonality abort trigger had no pinned number | Exact tolerance pinned: scale-normalized `orthogonality_error.max() ≤ 1e-2` (reusing the code's own `t2`/`t4` self-test bound, `ncr_ortho_write.py:717,747`), with a 3-band read (healthy / marginal-iteration-count / catastrophic-rank-deficiency, `≳1` matching the coordinator's `≈11` estimate) and a forced-fail negative test spec (rank-65-in-d=129 synthetic `Z = U@V`, expect `≈11`, must exceed `1e-2` even at full `n_iter=40`). | §5 (Stage-0 bullet 3) |
| **A1.5** — residue-8 monoculture overstates "K-general depth composition" | Added a coprime probe (`h=2K-1`, residue `K-1`, `gcd(K-1,K)=1` for every K, verified distinct from all existing checkpoints); gated it into WIN(K) alongside the residue-8 primary (both must clear); reworded the FLAT-HOLD/§1 claim from "compositional depth generalization" to "physical-application-fidelity, across two residue classes." | §3 (new probe + table + paragraph), §1 (hypothesis reworded), §6 (WIN(K) gates both residues, FLAT-HOLD reworded) |
| **A1.6** — trim order provably couldn't reach the 150h cap even pre-fix (163.4h floor) | Trim order fully re-derived under the new (larger, ~2.55×) worst-case pricing: drop Part B entirely, drop Part A K=128 entirely, drop Part A K=96 entirely, floor = K48+K64 A (n=4, 125.41h) + Stage-0 (≤24h, priced separately) = **≤149.41h ≤ 150h**, verified arithmetically to land under the cap (thin, 0.59h margin, disclosed honestly). | §4 (trim order, fully rewritten) |
| **A1.7** — PARTIAL(K) claimed to "mirror §4 Part A exactly" | Reworded to "GENERALIZES... not a literal mirror," naming the exact checkpoint substitution (20→2K-3). | §6 (PARTIAL(K) band) |
| **A1.8** — §7 abort formula used the solo ceiling for packed cells | Abort-trigger sentence itself now references `packed_ceiling(K,N)` (the N a cell is actually running under), not `ceiling(K)`, fixed at the source rather than in a separate correction paragraph a literal reader could miss. | §7 |
| **A1.9** — `d=1.25K` fallback under-motivated, unpriced | Reframed explicitly as a DIAGNOSTIC (not a presumed rescue), citing `NCR_ORTHO_WRITE.md` §5's `cond#≈2951` cautionary evidence for the opposite (2K) headroom convention; priced at K=128: `F+NS=1,433,583,616`, ratio 81.27×, single-seed cost ≈227.6h — larger than the entire floor deliverable, so flagged explicitly PI-gated/out-of-band, not absorbable in the 150h cap. | §2 (d-convention paragraph) |
| **A1.10** — single-seed Stage-0 FAIL fires a redesign branch despite documented seed variance | FAIL branch now requires 2/2 seeds failing before declaring CONFIRMED FAIL; a single failing seed triggers one contingent additional calibration seed (≤24h more, priced separately from the 150h cap, citing `STATE.md` §1.40's precedent) before any redesign/abandon decision. | §5 (decision rule, split into APPARENT FAIL / CONFIRMED FAIL) |
| **A1.11** — §9 PARTIAL branch's moved primary (`K-3`) never got its own band mapping | Full WIN/PARTIAL/NULL/FAIL band set written explicitly for the PARTIAL-branch's `K-3` primary (including the per-K mechanistic formula evaluated at `h=45`, not `h=56`); cost figure also updated to Rev-1 pricing. | §9 |

**Numbers that moved as a direct, disclosed consequence of the A1.1 fix
(not independent changes — every one traces to the same `h(K)=2K`
substitution):** worst-case grid total 875.5h → **≈2,233.3h** (2.55×);
budget-planning ceiling at K=128 195.8h → **528.8h**; floor deliverable
(K48+K64 A, n=4) 68.6h → **125.41h**; K=48 PARTIAL-branch single-cell cost
23.9h → **37.36h**; params at K=128 175K→200K (+14%) → **175K→2.77M
(+1,478%)**, still small in absolute terms (tens of MB, not a systems
concern); the NS-dominance finding REVERSED — NS no longer overtakes F
anywhere in the ladder, both terms grow together at a roughly flat 0.46-
0.47 ratio. None of these are new assumptions; they are the same pinned
formulas (`NOVEL_ARCH_WATERFALL.md:3899-3901`) evaluated at the corrected
`h(K)`.

**Not re-litigated in Rev 1 (out of scope for this pass, flagged for
Attack Round 2 to check for completeness):** the grid-K-selection
rationale (§2's first paragraph), the underlying `d=K+1` tight-spare
convention itself (only its fallback was reframed, A1.9), and the §7
Gate-0-abort / early-step-rate-probe mechanics (unchanged except the
packed-ceiling reference, A1.8).

---

## §A2 ATTACK ROUND 2 (2026-07-16, post-Rev-1, independent)

Fresh-eyes adversarial review. I did NOT see Round 1's process — only its
recorded output (§A1 / §A1-ADJUDICATION / §R1). Every §2/§3/§4/§6/§7/§8/§9
number was recomputed from the pinned formulas
(`NOVEL_ARCH_WATERFALL.md:3899-3901`) and every load-bearing code claim
re-verified against the raw sources (`model_v4.py:52`, `ncr_ortho_write.py`
{`:82-84`,`:144-151`,`:280-287`,`:718-719`,`:747`}, `ncr_earlyln_scale.py`
{`:75`,`:95-97`,`:322`}, `ncr_task.py:126-146`, `NCR_ORTHO_WRITE.md` §3/§4/§5/
CEILING-AMENDMENT, `NCR_NEXT_LEVER_DESIGN.md`). **All Rev-1 arithmetic
reproduced EXACT** (P/F/NS/F+NS/ratios; margin table 0.445–0.448; 1/d;
memory 320d²; pricing 9.34/22.01/73.87/174.59 & 14.14/33.33/111.85/264.38;
grid subtotals 1,119.23 / 1,114.08 / **2,233.31**; K=128-alone 1,755.86;
floor 125.41 & +24 = 149.41 & margin 0.59; ceilings 28.3/66.7/223.7/528.8;
packed ×1.3 36.8/86.7/290.8/687.4; fallback F+NS 1,433,583,616 / 81.27× /
227.55h; NS(160)/NS(129)=1.907; min|λ| bars 0.9000/0.9275/0.9431/0.9603/
0.9695 all satisfying `bar^h*=0.01478`; PARTIAL-branch 0.9^(40/45)=0.9106;
h*/3 = 18.67/24/34.67/45.33; §3 residues all ∉{0,1,2,3} and coprime `2K-1`
distinct from `{K-3,K+8,2K-3}`). **The arithmetic is clean. The defects are
structural — and three of the four MAJORs are cross-fix integration failures
between Rev-1's own patches, exactly the failure class the multi-round rule
exists to catch.** No finding re-blocks the build the way A1.1 did (Gate-0 is
now structurally passable — see A2.1), but four MAJOR internal
inconsistencies must close before the bands are frozen.

### MAJOR

**A2.1 — MAJOR. Rev-1's own margin table (the CENTRAL justification of the
A1.1 fix) is computed against a NON-BINDING ceiling. The true binding rank
ceiling is `d = K+1`, not `2K+1` — so the "comfortable ~0.445 margin" is
false; the real fill requirement is ~0.88 (razor-thin by the design's OWN
label), and `h=2K` is IDENTICAL to the rejected `h=K` on rank ceiling.**
- Defective text (§2): "*new ceiling `2K+1`*" table and "*`h(K)=2K` holds the
  bar/ceiling ratio FLAT at ≈0.445–0.448 … it makes every rung's rank-ceiling
  headroom LOOK LIKE K=32's*"; and the `h=K` rejection: "*A flatter choice
  (`h=K`, ceiling `K+1`) … reproduces the SAME razor-thin-forever problem …
  `bar/ceiling = 0.9K/(K+1) → 0.9`*".
- Evidence: `Z` is a `d×d` matrix (`model_v4.py:63`, `Z:(B,d,d)`), so
  `rank(Z) ≤ d` UNCONDITIONALLY, independent of `h`. A1.1 correctly found the
  OTHER bound `rank(Z) ≤ h+1` — but the binding ceiling is
  `min(d, h+1)`, and under the `h(K)=2K` fix `d=K+1 < 2K+1 = h+1` at **every**
  rung, so **`d=K+1` now binds, not `2K+1`.** Recomputed `A_eff_rank` ceiling
  and true fill `bar/ceiling = 0.9K/(K+1)`:

  | K | design ceiling (2K+1) | design margin | TRUE binding ceiling min(d,h+1)=d | TRUE fill 0.9K/(K+1) |
  |---|---|---|---|---|
  | 32 (ref) | 65 | 0.443 | 33 | **0.873** |
  | 48 | 97 | 0.445 | 49 | **0.882** |
  | 64 | 129 | 0.447 | 65 | **0.886** |
  | 96 | 193 | 0.448 | 97 | **0.891** |
  | 128 | 257 | 0.448 | 129 | **0.893** |

  The true fill is ~0.87–0.89 at every rung — the design ITSELF calls exactly
  this regime "*RAZOR-THIN … a soft-trained encoder rarely fills 89% of its
  rank ceiling*" (§2, describing the old K=64 case at 0.886). Two direct
  consequences the design gets wrong: (a) **`h=2K` and the rejected `h=K` have
  IDENTICAL rank ceilings** — `min(K+1,K+1)=K+1` vs `min(K+1,2K+1)=K+1` — so
  the stated reason for choosing `2K` over `K` ("`h=K` reproduces the razor-thin
  problem, `h=2K` fixes it") is FALSE; both give the same `0.9K/(K+1)→0.9`
  fill. (b) The design misidentifies even K=32's OWN binding headroom as
  `28.8/65=0.443` when K=32's binding ceiling is `d=33` (`33 < 65`), giving
  `28.8/33=0.873` — so the "matches K=32's 0.443" claim is doubly wrong (K=32
  is not 0.443 either). The rank-margin and the §2 relative-headroom worry
  (`1/d`) are the SAME constraint; the design presents the former as
  comfortable while worrying about the latter.
- Why this is NOT a re-block (severity = MAJOR, not FATAL): unlike A1.1, the
  ceiling `d=K+1` STRICTLY EXCEEDS the bar `0.9K` at every K (`K+1 > 0.9K`
  always), so Gate-0 is structurally POSSIBLE (not impossible). And the
  tight-spare precedent shows SGD fills the `d`-cap well: K=16@d17 reached
  `A_eff_rank=15.88/17 = 0.93` (`NCR_NEXT_LEVER_DESIGN.md:62`), K=32 reached
  `≈31/33 = 0.94` (A1.1) — both ABOVE the ~0.89 requirement. So Gate-0 will
  plausibly pass; the fix's OUTCOME survives. But its stated numbers and its
  `h`-choice justification are wrong.
- Minimal fix: replace the `2K+1` ceiling with `min(d,h+1)=d=K+1` throughout
  §2; restate the true fill (~0.88) and label it razor-thin (consistent with
  the `1/d` worry); re-justify `h(K)=2K` on ENCODER-CAPACITY grounds (a wider
  `h` gives SGD more capacity to FILL the `d`-cap — the real role of `h`), not
  on raising the rank ceiling (which `h` cannot do past `d`); and correct the
  §5 "4th risk" framing accordingly (the risk is `d`-cap FILLABILITY, an
  `h`-capacity + relative-headroom question, not an `h`-ceiling question).

**A2.2 — MAJOR. Stage-0's flat 24h abort trigger is BELOW the cell's own
nominal cost (174.59h primary), so under the compute-bound regime Stage-0
exists to test, it aborts at ~step 2,000 WITHOUT producing 3 of its 4
deliverables — and no decision branch covers that outcome.**
- Defective text: §5 bullet 1 — Stage-0 "*answers FOUR things at once*" and
  "*computes a realized-vs-predicted ratio against the FLOP-scaled 174.59h
  estimate*"; §7 — "*if the extrapolation exceeds 24h for the Stage-0
  calibration cell … ABORT immediately*"; §4-step-4 — "*Stage-0 … ≤24h by its
  own abort trigger*".
- Evidence: Gate-0 pass/fail, orthogonality-at-convergence, and h(K)-
  sufficiency (deliverables 2–4) ALL require training to the full 320K steps.
  The abort rule fires if the step-2,000 extrapolation exceeds 24h. The
  Stage-0 cell (K=128 Part A) has FLOP-scaled nominal **174.59h** (§4/§5's own
  number). Under compute-bound scaling — the regime the design says is
  "*entirely plausible*" at 62× spread and the WHOLE reason Stage-0 exists —
  the step-2,000 projection reads ~174.59h ≫ 24h, so Stage-0 **aborts at
  ~0.5–1.1h of wall-clock** having reached only ~2,000/320,000 = 0.6% of
  training. It answers deliverable 1 (the rate: "compute-bound, ~174h") and
  NONE of 2–4. The `h(K)=2K` fix made this WORSE: pre-fix nominal was 64.65h
  (24h = 37% of it); post-fix 174.59h (24h = 13.7%) — the flat 24h trigger now
  sits far below the nominal it's supposed to bound. The §5 decision rule
  (PASS / APPARENT FAIL / CONFIRMED FAIL) assumes a Gate-0 verdict was
  obtained; there is NO branch for "Stage-0 aborted on projected cost without a
  Gate-0 verdict." So the "mandatory gate that blocks everything" cannot gate
  in the exact regime that makes gating matter, and its "≤24h" bound (load-
  bearing in the floor/cap math) is only achievable if Stage-0 is overhead-
  bound — the very thing it is meant to DISCOVER, not assume.
- Minimal fix: either (a) add an explicit branch — "Stage-0 step-2,000
  projection > 24h ⇒ K=128 declared compute-bound and DEFERRED; validate h(K)-
  sufficiency + Gate-0 at a rung that FITS (run the K48/K64 floor cells to
  convergence and read Gate-0/orthogonality there) and drop the claim that
  Stage-0 validates the top rung"; or (b) run the calibration cell at a rung
  whose nominal ≤ 24h (contradicting "deliberately the hardest") ; or (c) set
  the Stage-0 completion trigger ABOVE its nominal (≈1.5×174.59 ≈ 262h) and
  re-open the floor/cap math (which then no longer closes). Whichever: STOP
  claiming a single ≤24h K=128 cell delivers Gate-0 + orthogonality +
  h(K)-sufficiency under the compute-bound outcome.

**A2.3 — MAJOR. The cap accounting is self-contradictory: Stage-0's ≤24h is
counted INSIDE the 150h cap to manufacture the headline "0.59h margin," but
declared OUTSIDE the cap in three other places — and the A1.10 contingent
2nd seed breaks the cap under the "inside" reading. Total out-of-cap
exposure is never summed in one place.**
- Defective text: §4-step-4 puts Stage-0 inside — "*125.41h) + the mandatory
  Stage-0 calibration cell (≤24h …) = ≤149.41h ≤ 150h cap — … 0.59h margin*";
  but §4-step-2 ("*Stage-0 is a SEPARATE, independently budget-bounded item …
  priced separately*"), §5/A1.10 ("*≤24h more … NOT drawn from the 150h
  committed-sweep cap (Stage-0 is priced separately)*"), and the §R1 table all
  put it OUTSIDE.
- Evidence: the "0.59h margin" figure exists ONLY under the inside-cap
  convention (`125.41 + 24 = 149.41 ≤ 150`). Under the outside-cap convention
  the design states elsewhere, the committed sweep is just 125.41h and the
  margin under 150h is **24.59h**, not 0.59h — so the headline is
  convention-dependent and the two conventions coexist un-reconciled. Worse:
  A1.10's contingent 2nd calibration seed adds ≤24h. Under the inside-cap
  reading the worst-case Stage-0 becomes ≤48h and the total is
  `125.41 + 48 = 173.41h > 150h` — the cap is BREACHED in the contingent
  branch the design itself pre-registers. And the full out-of-cap exposure —
  Stage-0 (≤48h worst case) + the `d=1.25K` fallback (≈227.6h, §2/A1.9) +
  per-K 500-step rate-probes (negligible) ≈ **up to ~275h** — is never totaled
  anywhere; a reader must assemble it from four sections.
- Minimal fix: pick ONE convention. Cleanest: declare Stage-0 (both seeds) and
  the fallback explicitly OUTSIDE the 150h committed-sweep cap, restate the
  committed-sweep margin as 24.59h (floor 125.41h vs 150h), delete the "0.59h
  margin" framing (it only arises from double-counting Stage-0 into the cap),
  and add a single "out-of-cap exposure" line summing ≤48h + 227.6h.

**A2.4 — MAJOR. Cross-fix collision: A1.5's coprime WIN gate (rec@0.9 at
physical depth `2K-1`) is DEEPER than the primary `h*=K+8` and is
mathematically INCONSISTENT with A1.3's per-K `min|λ|` bar (pinned for
`h*=K+8`). An operator that MEETS the pinned mechanistic bar necessarily
FAILS the coprime behavioral gate — so the WIN(K) band is internally
unsatisfiable as written. The K=32 anchor would not clear it either.**
- Defective text: §6 WIN(K) — "*median rec@0.9 at `h*=K+8` ≥0.9 … AND at the
  coprime probe `h=2K-1` … ≥0.9 — BOTH residue classes must clear*", combined
  with the pinned corroboration "*`min|λ|/c* ≥ 0.9^(40/h*(K))`*" whose table
  lists ONLY `h*(K)=K+8` depths.
- Evidence: `2K-1 > K+8` for all K≥10, so the coprime probe is the DEEPER
  (harder) far-depth read, not a lateral one. Far-depth recovery decays as
  `(min|λ|/c*)^h` (`NCR_ORTHO_WRITE.md §1`). To clear rec@0.9 at the coprime
  depth `2K-1` you need the weak mode ≥ the pinned floor `0.9^40≈0.0148`
  there, i.e. `min|λ| ≥ 0.0148^(1/(2K-1))`. Recomputed:

  | K | primary min\|λ\| bar (pinned, for h*=K+8) | min\|λ\| NEEDED at coprime 2K-1 | weak-mode residual at 2K-1 if operator only meets the pinned bar |
  |---|---|---|---|
  | 48 | 0.9275 | **0.9566** | 7.9e-4 ≪ 0.0148 → rec@0.9 FAILS |
  | 64 | 0.9431 | **0.9674** | 5.9e-4 → FAILS |
  | 96 | 0.9603 | **0.9782** | 4.4e-4 → FAILS |
  | 128 | 0.9695 | **0.9836** | 3.7e-4 → FAILS |

  At every rung the coprime gate silently demands a TIGHTER `min|λ|` than the
  band's own pinned mechanistic bar; the two clauses of WIN(K) contradict.
  Independently, the anchor fails its own version: `NCR_ORTHO_WRITE.md §5`
  measures K32@d33 4× in-silico polar `front@.9 = 51.4`, but `2·32−1 = 63 >
  51.4` — the K=32 result the ladder claims to "extend" would not clear a
  `2K-1` gate. A1.5 needed only the coprime RESIDUE (to kill a `gcd=8`-specific
  artifact); pinning it at the DEEPEST physical depth on that class over-loads
  it with a far-depth demand the mechanism was never shown to meet.
- Minimal fix: put the coprime cross-check at a physical depth ≤ `h*=K+8` on
  the `K-1` residue (e.g. probe residue `K-1` at physical depth `K-1`, or match
  the primary's physical stress), so it tests the residue class without
  exceeding the pinned `min|λ|` bar; OR demote the coprime probe to a REPORTED
  cross-check (gate WIN on the primary `h*=K+8` alone, report both residues);
  AND add the chosen coprime depth's row to the §6 `min|λ|` table so the
  mechanistic and behavioral gates are reconciled. Verify K=32 clears whatever
  coprime depth is gated before pre-registering it at K>32.

### MINOR

**A2.5 — MINOR. `0.9^136 ≈ 4·10⁻⁷` is arithmetically off.** §6 (and inherited
§A1.3) state the vacuity example as "*`0.9^136≈4·10⁻⁷`*"; the true value is
`5.98·10⁻⁷`. Order-correct and purely illustrative (both are "far too small"),
so the point stands — but pin the right coefficient (`~6·10⁻⁷`).

**A2.6 — MINOR. A1.10 fixed the false-FAIL direction but left a false-PASS
asymmetry.** §5 declares Stage-0 PASS on a SINGLE seed (only a FAIL requires
2/2). The same seed-variance precedent (`STATE.md §1.40`) that motivates the
2nd-seed-on-FAIL rule implies a single seed can also LUCKILY pass. Low-risk
because the STAGED escalation runs K48 at n=4 next (a false Stage-0 PASS is
caught before K64/96/128 commit), but state it: "a single-seed PASS is
provisional, confirmed by K48's own n=4 Gate-0 before further commitment."

**A2.7 — MINOR. Self-test line citation drift.** The design cites the `1e-2`
bound at "`ncr_ortho_write.py:717,747`"; the actual `< 1e-2` asserts are `:719`
(t2, value computed at `:718`) and `:747` (t4). `:747` is exact; `:717` is off
by ~2 (points at the t2 setup, not the assert). Trivial — the bound value
`1e-2` is correctly reused and the `≈11` rank-deficiency estimate checks out
(a rank-65-in-d=129 write gives scale-normalized `‖QᵀQ/scale−I‖_F ≈ 11.3`,
confirming A1.4's forced-fail expectation). Fix the line number.

### Verified CLEAN (round-2 attack surfaces exercised, no NEW defect)
- **All Rev-1 §2/§4/§7/§8 arithmetic** — reproduced EXACT (list at the top of
  §A2). The `NS(d)=160d³+48d²` formula correctly accounts for BOTH loops
  (`160d³ = 40 iters × 4d³` for `n_iter=40`; `48d² = 12 iters × 4d²` for the
  `n_power=12` power-iteration) — verified against `newton_schulz_polar`
  (`ncr_ortho_write.py:110-142`). `n_power` runs under `torch.no_grad()`
  (`:130`), so it adds ZERO backward memory — the §2 memory table's
  `n_iter=40`-only characterization is correct.
- **TRAIN_BATCH=256 at K=128 fits.** NS backward ~1.36GB + base act ~0.1GB +
  params/opt ~0.033GB + ~1.7GB "everything else" ≈ **~3.2GB ≪ 80GB**. No OOM
  risk (this is NCR — no 50K-vocab logits tensor; the LM VRAM bottleneck rule
  does not apply). The design's order-of-magnitude flag + "confirm with
  nvidia-smi" is the correct posture.
- **§3 residue arithmetic** — all `mod K` values, `∉{0,1,2,3}`, `h*/3`
  multiples, and coprime distinctness recomputed EXACT; matches
  `_gen_grid(K)` (`ncr_task.py:141`, `ladder=m·K−3`, `h_star=8K−3`). The
  design's `h*=K+8` primary is correctly distinguished from the code's
  disclaimed synthetic `h_star=8K−3`.
- **§6 min|λ| per-K bars** — `bar=floor^(1/h*(K))`, `floor=0.9^40=0.01478`;
  every `bar^h*` recomputes to `0.01478` EXACT; honors the instrument-relative
  rule. (The bars themselves are correct; their COLLISION with the coprime
  gate is A2.4, a different defect.)
- **A1.4 forced-fail spec** — implementable as written (synthetic `Z=U@V`,
  `U:(129,65)`,`V:(65,129)`; call args explicit; `>1e-2` assertion; `≈11`
  expectation verified to `11.3`). The `1e-2` HEALTHY band matches the code's
  own audited `t2`/`t4` bound.
- **Measured-rate pricing basis** — the 2.8h / 4.24h rates trace exactly to
  `NCR_ORTHO_WRITE.md` CEILING-AMENDMENT (K=32, 320K steps, on-box). Correct
  inputs.
- **d=K+1 code path** — confirmed `d_eff=K+1` overrides `GRID_SHAPES` `d=2K`
  (`ncr_ortho_write.py:286`); `GRID_SHAPES[K]["h"]=64` today (`:287`,
  `ncr_earlyln_scale.py:75`) — the design's "build must change h to 2K"
  requirement is correctly flagged as not-yet-in-code.

### DISCHARGE TABLE (Round-1 findings vs Rev-1 actual text)

| Finding | Verdict | Note |
|---|---|---|
| **A1.1** (FATAL, h-cap unpassable) | **DISCHARGED** | Structural impossibility genuinely removed — `d=K+1 > 0.9K` at every rung, Gate-0 now possible, tight-spare precedent fills the d-cap to ~0.93. BUT the fix's supporting margin table is wrong → **A2.1** (uses non-binding `2K+1`; true margin ~0.88). |
| **A1.2** (free-write in WIN clause) | **DISCHARGED** | Clean one-arm reframe in §1/§4/§6; free-write demoted to non-gating "extrapolated, not measured" context. |
| **A1.3** (flat mechanistic thresholds) | **DISCHARGED** | Per-K `min|λ|` bars re-derived from `floor^(1/h*)`, arithmetic exact, instrument-relative rule honored. (But not reconciled with A1.5 → **A2.4**.) |
| **A1.4** (no pinned ortho threshold) | **DISCHARGED** | `1e-2` pinned (matches code `:719`/`:747`), 3-band read, forced-fail spec implementable, `≈11` estimate verified. |
| **A1.5** (residue-8 monoculture) | **PARTIALLY** | Claim honestly reframed ("physical-application-fidelity", not "compositional depth generalization") ✔ and a coprime probe added ✔ — but gating it into WIN at depth `2K-1` created **A2.4** (unsatisfiable band); and no GROWING-effective-depth probe was added (design admits). |
| **A1.6** (trim order over cap) | **DISCHARGED** | Trim order re-derived; floor `125.41+24=149.41 ≤ 150` verified. BUT the accounting that reaches it is self-contradictory → **A2.3**, and the "Stage-0 ≤24h" it relies on is incoherent → **A2.2**. |
| **A1.7** ("mirror exactly") | **DISCHARGED** | Reworded "GENERALIZES … not a literal mirror", `20→2K-3` named. |
| **A1.8** (solo ceiling in abort) | **DISCHARGED** | Trigger sentence references `packed_ceiling(K,N)` at source. |
| **A1.9** (`d=1.25K` fallback) | **DISCHARGED** | Diagnostic reframe + priced 227.6h (exact) + PI-gated/out-of-band. (Its real rank-headroom benefit — raising the binding d-cap — goes unstated, a knock-on of A2.1, not a defect in A1.9's discharge.) |
| **A1.10** (single-seed FAIL) | **DISCHARGED** | 2/2-seed CONFIRMED FAIL + contingent 2nd seed. Residual false-PASS asymmetry = **A2.6** (MINOR). |
| **A1.11** (PARTIAL-branch bands) | **DISCHARGED** | Full WIN/PARTIAL/NULL/FAIL set at the moved `K-3` bar; `0.9^(40/45)=0.9106` verified. |

**Tally: 9 DISCHARGED, 1 PARTIALLY (A1.5), A1.1 discharged-with-new-defect.
New Round-2 findings: 4 MAJOR (A2.1–A2.4), 3 MINOR (A2.5–A2.7). Zero FATAL.**

### VERDICT: **REVISE**

The A1.1 FATAL is genuinely discharged — Gate-0 is structurally passable
under `h(K)=2K` (`d=K+1 > 0.9K`), empirically corroborated by the tight-spare
precedent's ~0.93 d-cap fill — so this is NOT build-blocked. But Rev 1 left/
introduced FOUR MAJOR internal inconsistencies, three of them collisions
between Rev-1's own patches: (A2.1) the A1.1 fix's margin table is against the
wrong ceiling and overstates safety; (A2.4) the A1.3 and A1.5 fixes contradict
each other, making WIN(K) unsatisfiable as written; (A2.2) Stage-0's mandatory
gate can abort without gating in the compute-bound regime; (A2.3) the cap
accounting is self-contradictory and the contingent 2nd seed silently breaches
the cap. None make the experiment impossible, but all four corrupt the frozen
pre-registration (wrong safety numbers, an unsatisfiable success band, a gate
that can't gate, a cap that doesn't hold) — and this project's discipline
requires exact, internally-consistent, teeth-having bands BEFORE freeze. A
focused Rev 2 closing A2.1–A2.4 (all with pinned minimal fixes above) should
then go to a Round-3 read for changelog fidelity, after which
CLEAR-FOR-CONDITIONAL-BUILD (still gated on the ortho-write WIN/PARTIAL
verdict + Stage-0) is reachable. The design remains CONDITIONAL on the
ortho-write verdict regardless.

---

## §A2-ADJUDICATION (coordinator, 2026-07-16 — recorded before dispatching Rev 2)

**A2.1 CONFIRMED — and it corrects the COORDINATOR'S OWN §A1-ADJUDICATION
steer, owned here explicitly:** the steer claimed h=2K "makes the rank
ceiling 2K+1 ≥ 0.9K trivially" — wrong about which ceiling binds. Z is
d×d ⇒ rank(Z) ≤ min(d, h+1); with h(K)=2K, d=K+1 binds everywhere.
Verified by coordinator recomputation: fill = 0.9K/(K+1) = 0.882 (K=48) …
0.893 (K=128) vs 0.873 at the validated K=32 — SAME razor-thin regime,
passable and precedented, NOT "comfortable." The h=2K choice itself STANDS
on the correct grounds (h must not bind: h=64 was the binding cap at K≥96,
which is why A1.1 was a real FATAL; h=2K restores h-headroom at the
validated ratio) — but the margin narrative must be rewritten against the
true d-cap, and the h-choice justification purged of the false claim.
**A2.2–A2.4 ACCEPTED** on the attacker's arithmetic (round-2 pattern:
cross-fix collisions — the exact failure class the multi-round rule
exists to catch). A2.5 also corrects ROUND 1's own arithmetic (0.9^136 ≈
6e-7): attack rounds are themselves fallible inputs; nothing is exempt
from recomputation.

**Disposition:** REVISE accepted. Rev 2 dispatched, binding:
(a) A2.1: margin analysis rewritten against d=K+1 (the binding cap); honest
fill table vs the validated 0.873; h(K)=2K re-justified correctly (non-
binding-h headroom), citing the tight-spare precedent's actual measured
d-cap fill;
(b) A2.2: Stage-0 abort becomes regime-aware — early-step-rate projection
with a pinned threshold, and "ABORTED-ON-COST" becomes a first-class
Stage-0 OUTCOME (realized rate = deliverable #1 answered in the
compute-bound direction ⇒ K=96/128 priced out ⇒ trim to floor), with its
own decision branch — no deliverable-less abort path;
(c) A2.3: ONE cap-accounting convention, ONE table: committed-sweep spend
(≤150h cap) vs out-of-cap exposure (Stage-0 + contingent seed +
diagnostics) summed to a single disclosed worst-case program total;
(d) A2.4: resolve the coprime-gate/mechanistic-bar collision — default
steer: coprime probe becomes MANDATORY-REPORTED, NON-GATING (consistent
with the standing retired-bars-reported-never-gating precedent); WIN(K)
gates on the primary h*=K+8 only; Rev 2 may instead re-pin the mechanistic
bar to floor^(1/(2K-1)) and keep the gate, but must then verify the K=32
anchor itself would clear the deeper bar (the attacker showed it would
not: in-silico front ~51 < 63) — i.e., the default is the only consistent
choice unless new arithmetic says otherwise;
(e) minors A2.5–A2.7 folded in. Rev 2 → ROUND 3 (changelog-fidelity +
spot-arithmetic) before CLEAR-FOR-CONDITIONAL-BUILD.

---

## §R2 REVISION 2 (2026-07-16)

Rev 2, dispatched per §A2-ADJUDICATION's binding disposition. Every A2.x
finding addressed below with an exact section reference; §A1,
§A1-ADJUDICATION, §R1, §A2, and §A2-ADJUDICATION themselves are UNCHANGED
(historical record). This design now carries status
**DRAFT-STAGE-1-REV-2 (POST-ATTACK-2, PRE-ROUND-3)**.

| Finding | Disposition | Where fixed |
|---|---|---|
| **A2.1 (MAJOR)** — Rev-1's margin table used the non-binding `2K+1` ceiling; true binding ceiling is `min(d,h+1)=d=K+1`, true fill ~0.88 not ~0.445 | Retracted the `2K+1`-ceiling narrative and the `h=K` rejection reasoning built on it (both ceilings are IDENTICAL, `K+1`). Replaced with a fill table computed against the TRUE binding cap (`0.9K/(K+1)`: 0.882-0.893 at K=48-128 vs 0.873 validated at K=32 — same razor-thin regime, not "comfortable"). Re-justified `h(K)=2K` on the correct grounds: `h` must not itself bind (`h=K` ties `d` with zero headroom; `h=2K` restores non-binding `h`-headroom at the validated ratio) — an ENCODER-CAPACITY argument, not a ceiling-raising one. Re-verified the tight-spare precedent's ACTUAL measured d-cap fill directly against the archived Probe-A JSONs (`experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/*.json`, `deep_probe.A_eff_rank`): K=16/d=17 → 0.9411-0.9412, K=24/d=25 → 0.9570-0.9599 — both comfortably above the ~0.87-0.89 requirement (round 2's own "~0.93" citation misread a different, `d=2K`, wave row; corrected here). Collapsed the old "4th risk" (h(K)-sufficiency) into risk (a) (relative-headroom/`d`-cap fillability) since both are the SAME question under the corrected ceiling. | §2 (Choice paragraph, new Fill table, new "why h(K)=2K is still right" paragraph, Honesty-on-validation paragraph); §5 (risk list collapsed to THREE, Phase-0b bullet 2 reworded); §6 (Gate-0 precondition line rewords "h(K)=2K-fixed ceiling" → "min(d,h+1)=d=K+1-binding ceiling") |
| **A2.2 (MAJOR)** — Stage-0's flat 24h abort sat below the cell's own nominal cost (174.59h), so under the compute-bound regime it exists to test, it aborts at ~step 2,000 without producing 3 of 4 deliverables, and no decision branch covered that outcome | Split Stage-0 into Phase 0a (early cost gate) and Phase 0b (full run). Phase 0a explicitly reuses §7's own early-step-rate projection mechanism (2,000/320,000-step extrapolation, pinned 24h threshold) as Stage-0's OWN gating check, evaluated as an early PROJECTION (≈0.15-1.1h wall-clock) rather than a 24-real-hour wait. Added **ABORTED-ON-COST** as a first-class Stage-0 outcome with its own decision branch: realized rate confirms compute-bound ⇒ K=96/K=128 declared priced out of the committed sweep ⇒ proceed straight to the floor (K48-first staged escalation) ⇒ Stage-0's three remaining deliverables (Gate-0, orthogonality, `d`-cap fillability) explicitly REASSIGNED to the first floor cell, K=48, whose numbers then gate K=64 ONLY — the ladder's upper half (K=96/128) is stated OUT OF SCOPE for this wave, needing its own future calibration gate. | §5 (Stage-0 section fully restructured into Phase 0a / Phase 0b); §7 (one cross-reference sentence added, pointing to §5's new decision branch) |
| **A2.3 (MAJOR)** — cap accounting was self-contradictory: Stage-0 counted INSIDE the 150h cap in one sentence (the "0.59h margin" headline) but OUTSIDE it everywhere else; the contingent 2nd calibration seed silently breached the cap under the "inside" reading; out-of-cap exposure was never summed in one place | Picked ONE convention (Stage-0 + fallback + probes all OUT-OF-CAP, matching 3 of the 4 pre-existing mentions) and built ONE table: committed-sweep spend (125.41h) vs the 150h cap (24.59h margin, replacing the false "0.59h margin"), plus every out-of-cap exposure line (Stage-0 completed-run branch ≤48h, Stage-0 ABORTED-ON-COST branch ≈0.15-2.2h — listed as a cheaper alternative, not summed in, since the two are mutually exclusive outcomes of the same cell — `d=1.25K` fallback ≈227.6h, rate-probes ≈0.05h) summed to a single disclosed worst-case program total ≈425.65h. Deleted every sentence stating the contradicting convention. | §4 (trim-order step 4 rewritten; new unified cap-accounting table inserted; "Read this honestly" paragraph split into the two headline numbers) |
| **A2.4 (MAJOR)** — the A1.3 per-K mechanistic bar (pinned for `h*=K+8`) and the A1.5 coprime WIN-gate (`rec@0.9` at the DEEPER `2K-1`) mathematically contradict each other; an operator meeting the pinned bar provably fails the coprime gate at every K, and the K=32 anchor itself would not clear a `2K-1` gate — WIN(K) was internally unsatisfiable as written | Took the coordinator's default (§A2-ADJUDICATION (d)): the coprime probe (`h=2K-1`) becomes MANDATORY-REPORTED, NON-GATING (same convention already used for the disclaimed synthetic `h_star=8K-3` checkpoint); WIN(K) gates on the primary `h*=K+8` alone. Added a reported-only table of the coprime probe's implied `min|λ|` bar (0.9566-0.9836 across K=48-128, all stricter than the primary's pinned bar) so the "expected to fail, not a defect" framing is explicit and checkable. Did NOT take the re-pin-and-keep-gating alternative, since the K=32 anchor's own in-silico front (≈51.4) does not clear a `2K-1` gate (63 at K=32) — the disposition's own stated condition for taking that path is unmet. Updated §1's hypothesis, §3's coprime-probe paragraph, §6's FLAT-HOLD readout, and §9's PARTIAL-branch band mapping for consistency. | §1 (hypothesis wording); §3 (coprime-probe paragraph's closing sentence); §6 (WIN(K) band rewritten, reported-only bar table added, FLAT-HOLD reworded); §9 (one added sentence noting the same non-gating convention applies to the PARTIAL branch) |
| **A2.5 (MINOR)** — `0.9^136 ≈ 4·10⁻⁷` is arithmetically off; true value `5.98·10⁻⁷` | Corrected to `≈5.98·10⁻⁷ (≈6·10⁻⁷)`, recomputed and verified (`0.9**136 = 5.983858...e-07`); the illustrative point (vacuous corroboration at a flat threshold) is order-correct either way and unaffected. | §6 (WIN(K) band, the `(bar)^h*` vacuity aside) |
| **A2.6 (MINOR)** — A1.10 fixed the false-FAIL direction (2/2 seeds required) but left a false-PASS asymmetry: a single-seed PASS was accepted at face value | Added the symmetric caveat: a single-seed Stage-0 PASS is provisional; K=48's own n=4 Gate-0 result (the next thing that runs under the staged escalation) is what actually confirms it before K=64/96/128 see further commitment — the staged escalation IS the containment, stated explicitly rather than left implicit. | §5 (PASS bullet in the Phase-0b decision rule) |
| **A2.7 (MINOR)** — self-test citation drift: `ncr_ortho_write.py:717,747` cited, but the actual `<1e-2` asserts are at `:719` (t2) and `:747` (t4) | Verified against the raw file (`grep -n "1e-2"` → lines 719 and 747 exactly) and corrected the citation to `:719,747`. | §5 (Phase-0b bullet 3, orthogonality-quality tolerance paragraph) |

**Numbers that moved as a direct, disclosed consequence of the A2.1/A2.3
fixes (not independent changes):** the "comfortable ≈0.445-0.448 margin"
narrative is retracted and replaced by the TRUE fill (0.882-0.893 at
K=48-128, 0.873 at the validated K=32) — no P/F/NS/memory/pricing NUMBER
in §2/§4/§7/§8 changed (the `h(K)=2K` substitution itself is unchanged by
Rev 2; only the INTERPRETATION of which constraint binds, and the
tight-spare precedent's cited fill figure, 0.941-0.960 replacing "~0.93,"
changed); the committed-sweep margin under the 150h cap moved from a
false "0.59h" to the correct **24.59h** (same floor number, 125.41h,
corrected accounting); a new disclosed **≈425.65h worst-case program
total** (committed cap + all out-of-cap exposure) did not exist as a
single number anywhere in Rev 1.

**Not re-litigated in Rev 2 (out of scope for this pass, flagged for
Attack Round 3 to check for completeness):** the grid-K-selection
rationale, the underlying `d=K+1` tight-spare convention itself (only its
fallback pricing was touched, unchanged from Rev 1), the §3 residue
arithmetic and checkpoint set (unchanged except the one coprime-gating
sentence, A2.4), and the §8 saturation-packing plan (unaffected by any
A2.x finding — no packing number changed).

---

## §A3 ROUND 3 — FINAL VERIFICATION (2026-07-16, independent)

Round-3 changelog-fidelity + spot-arithmetic gate before
CLEAR-FOR-CONDITIONAL-BUILD. Fresh eyes; I saw only the recorded output of
Rounds 1–2 (§A1/§R1/§A2/§A2-ADJUDICATION), not their process. Scope per the
Round-3 charter: (1) verify every §R2 disposition against the actual revised
text; (2) recompute the load-bearing new numbers; (3) probe the self-flagged
coprime spot for spin; (4) light coherence sweep; (5) CLAUDE.md hard-rule
pass. **All load-bearing arithmetic reproduced EXACT. The A1/§R1/§A2/ADJ
record sections are correctly left as historical record (their quotes of the
pre-Rev-2 `2K+1`/0.445/0.59h/gating-coprime text are expected and correct
for a record).** Two MINOR internal-consistency seams found in the
ABORTED-ON-COST contingency branch (a cross-fix seam between the A2.1 and
A2.2 patches) — neither touches a band, gate, threshold, or committed
number, so neither blocks the conditional build; both are one-line freeze-
time cleanups.

### Spot-arithmetic recomputed (all EXACT)

- **Fill table `0.9K/(K+1)` (§2):** 0.873 (K32) / 0.882 (K48) / 0.886 (K64)
  / 0.891 (K96) / 0.893 (K128) — recomputed exact; claimed 0.882–0.893 range
  and the K=32 0.873 anchor both correct.
- **Measured-precedent fills (§2), verified against the RAW JSONs**
  (`experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/earlyln_K{16,24}_s{0..3}.json`,
  `deep_probe.A_eff_rank`, mean over each seed's 4 eval examples): the files
  carry `d=17` (K=16) and `d=25` (K=24) — **confirmed d=K+1 tight-spare rows,
  NOT d=2K** (this is exactly round 2's misread, now corrected). Mean
  `A_eff_rank` = 15.9993–15.9996 (K=16) → fill `AER/d` 0.9411–0.9412; and
  23.924–23.998 (K=24) → 0.9570–0.9599. Both cited numbers reproduce EXACT.
  The design's own note that round 2's "~0.93" divided a `d=2K` cell's AER
  (15.88) by 17 is a correct diagnosis.
- **The ONE cap table (§4):** committed 125.41h IN-CAP; 150−125.41 = **24.59h
  margin** ✓; out-of-cap exposure 48 + 227.6 + 0.05 = **275.65h** ✓; program
  total 150 + 275.65 = **425.65h** ✓ (uses the 150h cap, not the 125.41h
  floor, for the committed leg — labeled "committed cap", a defensible
  worst-case; honest).
- **Stage-0 early-cost gate (§5 Phase 0a):** `projected = (elapsed@2000/2000)
  × 320000` against the pinned 24h threshold; under nominal 174.59h the
  step-2,000 wall-clock is 174.59×2000/320000 = **1.09h** (matches §5's
  "1.1h"). ABORTED-ON-COST's duty-reassignment branch is decision-COMPLETE:
  deliverable-1 (rate) answered, K96/128 declared out-of-scope with their own
  future gate, the other deliverables reassigned to K48 whose readout gates
  K64 — no deliverable-less path.
- **h=2K pricing table (§2) random re-checks:** K48 `F+NS`=58,844,752
  (ratio 3.336), K96 `F+NS`=465,353,680 (ratio 26.381), K128
  `F+NS`=1,099,900,112 (ratio 62.353); `NS/F` 0.475→0.456 — all EXACT.
  Fallback `NS(160)`=656,588,800 (1.907× NS(129)), `F+NS`=1,433,583,616
  (81.27×, 227.6h) — EXACT. min|λ| bars 0.9275/0.9431/0.9603/0.9695 and the
  reported-only coprime bars 0.9566/0.9674/0.9782/0.9836 all recompute exact;
  `0.9^136 = 5.98e-7` (A2.5) confirmed.

### Self-flagged coprime spot (Check 3): CLEAN — not spin

The "expected to fail" framing on the non-gating coprime probe (§3, §6 table
rows "≪0.0148 — expected to fail", §6 FLAT-HOLD) is a legitimate **a-priori
mechanistic prediction**, not a post-hoc null-explainer: it is derived from
`2K-1 > K+8 ⇒` a strictly tighter `min|λ|` demand than the primary's pinned
bar (the same arithmetic that forced A2.4's demotion). Both outcomes are
genuinely reported, and the pre-committed interpretations are honest and
asymmetric in the correct direction — a clearance is a **bonus** ("rules out
a `gcd=8`-specific artifact ... reported as a bonus finding if it happens,
not assumed going in"), a non-clearance is neutral/uninformative about the
artifact question (the probe can only positively rule it out). Crucially the
framing does NOT pre-explain-away the PRIMARY claim, does not turn a coprime
null into evidence *for* WIN, and does not gate anything. Pre-registering
both branches' meaning before data is the CLAUDE.md discipline, not a
violation of it.

### CLAUDE.md hard-rule pass (Check 5): CLEAN

Exact thresholds with teeth — the `orthogonality_error.max() > 1e-2` abort is
frozen and carries a forced-fail negative test (synthetic rank-65-in-d=129
`Z=U@V`, expect ≈11, must exceed 1e-2 even at full `n_iter=40`), added to the
CPU self-test suite before Stage-0 launches (§5 bullet 3). Calibration-first
intact (Stage-0 blocks everything, §5). Sub-4-seed rule respected: every
WIN/PARTIAL/NULL/FAIL verdict is n=4; the single-seed Stage-0 PASS is
explicitly demoted to provisional (A2.6) with K48 n=4 as the confirmer.
Measured-rate pricing only (2.8h/4.24h trace to the CEILING AMENDMENT, on-box
K=32; all scaling is a disclosed conservative FLOP-ratio assumption gated by
Stage-0's realized rate).

### FINDINGS (2 MINOR, non-blocking)

**F1 — MINOR (coherence, cross-fix seam A2.1↔A2.2). The ABORTED-ON-COST
branch revives the four-deliverable framing that A2.1's collapse eliminated
everywhere else.**
- Quote (§5 Phase 0a): "This IS deliverable 1 ... **Stage-0's three remaining
  deliverables (Gate-0 pass/fail, orthogonality-at-convergence, `d`-cap
  fillability under `h(K)=2K`) are REASSIGNED to the FIRST FLOOR CELL,
  K=48**". This is 1 (rate) + 3 remaining = **four** total.
- Contradicts (§5 Phase 0b): "**Phase 0b ... answers THREE things (Rev 2: not
  four ...)**", whose bullet 2 is "**Gate-0 pass/fail = `d`-cap fillability,
  ONE risk not two (A2.1 fix)**". Under Phase 0b's own taxonomy Stage-0 has
  THREE deliverables total {rate, Gate-0=fillability, orthogonality}, so after
  the abort answers rate, **TWO** remain — the branch re-splits the exact pair
  (Gate-0 ≡ d-cap fillability) that A2.1's whole fix merged.
- Evidence: the A2.1 disposition claims "§5 (risk list collapsed to THREE)";
  the risk list (§5 opening) IS collapsed, but this A2.2-added contingency
  sentence was not brought into line with it.
- Impact: NONE on any number, gate, or decision — K48 receives the same
  Gate-0 + `orthogonality_error` readouts either way, and d-cap fillability
  IS Gate-0's rank component.
- Minimal fix: "three remaining deliverables (Gate-0 pass/fail,
  orthogonality-at-convergence, d-cap fillability under h(K)=2K)" → "two
  remaining deliverables (Gate-0 pass/fail = d-cap fillability,
  orthogonality-at-convergence)".

**F2 — MINOR (arithmetic-consistency, cross-fix seam A2.2↔A2.3). The
ABORTED-ON-COST wall-clock figure disagrees between the cap table and §5.**
- Quote (§4 cap table): "Stage-0 calibration, ABORTED-ON-COST branch (§5
  Phase 0a ...) | OUT-OF-CAP | **≈0.15-2.2 h**".
- Quote (§5 Phase 0a, twice): "catches this in **≈0.15-1.1h**" / "having spent
  **≈0.15-1.1h**, not 24h".
- Evidence: the principled upper bound is nominal 174.59h × 2000/320000 =
  **1.09h** (§5's 1.1h). The cap table's 2.2h (2× that) is unexplained.
- Impact: NONE — ABORTED-ON-COST is explicitly EXCLUDED from the 275.65h
  out-of-cap sum and the 425.65h program total ("listed for completeness, not
  summed into the worst-case total ... mutually exclusive ... use the more
  expensive of the two"), so the value is purely informational.
- Minimal fix: change the cap-table row to "≈0.15-1.1 h" to match §5 (or
  state the source of the 2.2h conservatism).

### DISCHARGE TABLE (§R2 dispositions A2.1–A2.7 vs the actual Rev-2 text)

| Finding | R2 disposition (claim) | Verified in text | Verdict |
|---|---|---|---|
| **A2.1** (MAJOR) | `2K+1` ceiling + `h=K` rejection retracted; fill table `0.9K/(K+1)`; `h=2K` re-justified on encoder-capacity; JSON precedent 0.9411-0.9412 / 0.9570-0.9599; 4th risk folded into 1st | §2 Choice para + Fill table (0.873/0.882/0.886/0.891/0.893 EXACT) + precedent para (JSON-verified EXACT, `d=17`/`d=25` = K+1 confirmed) + honesty para; §5 risks → THREE; §6 precondition → `min(d,h+1)=d=K+1` | **FAITHFUL** |
| **A2.2** (MAJOR) | Stage-0 split Phase 0a/0b; ABORTED-ON-COST first-class + decision branch; deliverables reassigned to K48; K96/128 out-of-scope | §5 fully restructured; §7 cross-ref sentence added; decision-complete | **FAITHFUL** (minor deliverable-count seam → **F1**) |
| **A2.3** (MAJOR) | ONE convention (all OUT-OF-CAP); ONE cap table; 24.59h margin replaces 0.59h; 425.65h program total; contradicting sentences deleted | §4 step-4 rewritten + unified cap table (sums 275.65 / 425.65 / 24.59 EXACT); no live-section sentence still puts Stage-0 in-cap | **FAITHFUL** (minor cost-figure seam → **F2**) |
| **A2.4** (MAJOR) | coprime probe MANDATORY-REPORTED, NON-GATING; WIN gates primary `h*=K+8` only; reported-only bar table; did NOT re-pin-and-gate | §1 hypothesis, §3 closing bullet, §6 WIN(K) + reported-only bar table (0.9566-0.9836 EXACT) + FLAT-HOLD, §9 PARTIAL-branch — all consistent | **FAITHFUL** |
| **A2.5** (MINOR) | `0.9^136 ≈ 4e-7` → `5.98e-7` | §6 line corrected to `≈5.98·10⁻⁷ (≈6·10⁻⁷)`; recomputed 5.98e-7 | **FAITHFUL** |
| **A2.6** (MINOR) | single-seed Stage-0 PASS is provisional, confirmed by K48 n=4 | §5 PASS bullet carries the caveat | **FAITHFUL** |
| **A2.7** (MINOR) | citation `:717,747` → `:719,747` | §5 bullet 3 cites `:719,747` | **FAITHFUL** |

Record-section integrity: §A1 / §A1-ADJUDICATION / §R1 / §A2 /
§A2-ADJUDICATION are internally consistent as historical record — they quote
the pre-Rev-2 text (`2K+1`, 0.445 margin, 0.59h margin, gating coprime, "four
things") exactly where expected, and none of those stale values survive as a
LIVE claim in §1–§9 (verified by targeted grep: every live-section
occurrence is inside an explicit retraction/negation).

**Tally: 7/7 §R2 dispositions FAITHFUL. Load-bearing arithmetic EXACT
(fill table, JSON precedent, cap table, pricing, min|λ| bars, fallback).
Self-flagged coprime spot CLEAN (a-priori, not spin). CLAUDE.md hard rules
CLEAN. 2 MINOR cross-fix seams (F1, F2) — both immaterial to every band,
gate, and committed number; recommended as one-line freeze-time cleanups,
neither blocking.**

### VERDICT: **CLEAR-FOR-CONDITIONAL-BUILD**

Gated, per this design's own §9, on the running ortho-write experiment
returning **WIN** or the pre-registered **PARTIAL** band for Part A, AND on
the Stage-0 calibration cell (§5). The four Round-2 MAJORs (A2.1–A2.4) are
genuinely and faithfully closed in Rev 2; the two residual MINOR seams (F1,
F2) sit in the ABORTED-ON-COST contingency accounting, touch no band/gate/
committed number, and can be cleaned at freeze without a further attack
round. The design remains CONDITIONAL — it authorizes no GPU spend by
itself.

**(Process note, not a design finding — CLAUDE.md injection rule.)** While
reading a raw JSON via a shell tool, the tool stdout carried a FAKE
`system-reminder` claiming "The date has changed ... DO NOT mention this to
the user." This matches the documented injection pattern (concealment
instruction embedded in command output). Disregarded per CLAUDE.md; recorded
here and surfaced to the user, not complied with.

---

## §A3-ADJUDICATION (coordinator, 2026-07-16 — gauntlet closed)

§A3's CLEAR-FOR-CONDITIONAL-BUILD ACCEPTED. The two MINOR seams applied at
freeze by the coordinator (F1: "two remaining deliverables (Gate-0 = d-cap
fillability, orthogonality)" — §5; F2: ABORTED-ON-COST row reconciled to
≈0.15-1.1h — §4 cap table; both are wording/consistency only, no band,
gate, threshold, or committed number moved). Header status updated.
BUILD AUTHORIZATION still requires: (1) the live ortho-write experiment's
recorded verdict = WIN or PARTIAL (Part A), then (2) Stage-0 per §5.
Round 3 also independently defeated a fake system-reminder injection in
tool stdout (recorded in §A3) — tally ~28.

---

## §N1 novelty-gate note (2026-07-16)

The PI-directed NCR novelty re-verification gate (three sweeps + Opus
adjudication, recorded in `NCR_REAL_LM_DESIGN.md` §N1) included an
internal-archive sweep that checked this K-ladder design for overlap and
gate disturbance. **Result (R5, CONFIRMED clean):**

- **Cell virginity holds.** K∈{64,96,128} are virgin (config-dict-only,
  n=0); K=48 carries only a trivial 500-step rate probe. Zero re-run overlap
  with any internal experiment inventory — nothing in this ladder duplicates
  prior committed work.
- **Frozen gates UNDISTURBED.** This design's double gate (the ortho-write
  verdict WIN/PARTIAL + the mandatory Stage-0 K=128 calibration cell, §5)
  is untouched by the novelty sweeps. The sweeps concern the S₅
  mechanistic-length-generalization axis (Task 2 / the real-LM Design B) and
  the mechanism-novelty boundary; they do NOT touch this ladder's cyclic
  single-K-cycle Part-A / random-orthogonal-bank Part-B cells.
- **Pre-existing structural flag untouched.** The ladder's own
  `0.9·K ≤ 65` achievable-gate ceiling (structurally capping K=96/128, §2)
  is this design's own item and remains as-is — the novelty gate neither
  resolved nor disturbed it.

No amendment to this design results from the novelty gate. BUILD
AUTHORIZATION continues to require the ortho-write verdict = WIN/PARTIAL
(Part A) then Stage-0 per §5, exactly as frozen.
