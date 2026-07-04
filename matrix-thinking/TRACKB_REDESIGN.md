# TRACKB_REDESIGN — Hard-Selectivity geo3-in-LM: Forcing β's Missing Precondition Before Retesting Orthogonalization on Free Text

> **Rev 3 — 2026-07-04, post-attack-round-2.** Round 2 (`TRACKB_REDESIGN_ATTACK_R2.md`,
> fresh agent, against commit `f814b99`) returned **NEEDS-REV-3**: round 1's F1 is
> fully CLOSED (the MC anchor math independently verified against the closed forms),
> F2/M1/M7/M8 partially closed, and **8 new MAJORs + 2 MINORs, all in Rev 2's own
> additions**. Every finding is addressed per the orchestrator's binding decisions;
> the round-2 finding→change map is **§13** (§12, round 1's map, is frozen).
> Load-bearing changes: **NEW-1** — the clamp-shortfall PARTIAL trigger is now
> symmetric (one numeric rule, every masking cell, not candidate 2 alone); **NEW-2 /
> NEW-3** — Cell 2R's random selection is registered as resampled per chunk instance
> per step (killing the frozen-mask artifact inside the control itself), and its
> decision rule is re-registered as seed-range non-overlap with an explicit
> INCONCLUSIVE outcome and an honest low-power acknowledgment (α ≈ 0.10 per corpus
> under the exchangeability null); **NEW-4** — the churn ceiling becomes the max of
> TWO nulls (unmasked pilot + the gated run's own early window), calibration
> limitation acknowledged; **NEW-5** — the positional-artifact check, which Rev 2
> cited into a broken cross-reference with no spec behind it, is fully specified and
> folded into `BANDS_PINNED-TrackB`; **NEW-6** — the comparator's anneal is pinned to
> τ→0 by 10% of steps so it isolates the gradient estimator, not an annealing
> curriculum; **NEW-7** — the NaN-smoke slice is re-specified from the code's actual
> failure mechanism (duplicate keys among SELECTED positions) with a positive-control
> requirement so the smoke cannot pass vacuously; **NEW-8** — the budget source's
> internal ~26 GPU-h inconsistency, caught by this attack, was fixed upstream in
> `SCALE_TRANSFER_DESIGN.md` (committed `13eb71c`); §6 quotes the corrected text —
> headroom after rung-3 = 300 − 266.4 ≈ **33.6 GPU-h**. **§9 (claim tiers) and every
> item round 2 marked CLOSED remain frozen.** This is intended as the final full
> revision; remaining findings are bounded. **Design only, no GPU, no box access.**
>
> **Rev 2 — 2026-07-04, post-attack-round-1.** The independent adversarial review of
> Rev 1 (`TRACKB_REDESIGN_ATTACK_R1.md`) returned **NEEDS-REV-2**: 2 FATAL, 8 MAJOR,
> 3 MINOR. Every finding is addressed in this revision, per the orchestrator's binding
> decisions; the finding→change map is **§12** (placed at the end so §1–§11 read as the
> current design, not a diff). The load-bearing changes: **F1** — Rev 1's §5.3 headline
> bar compared two different instruments over two different K-populations; Rev 2
> registers ONE instrument for every factorial cell (the §4.4 selected-key instrument
> at K_sel), with the random/collapse anchors recomputed AT K_sel at Wave −1 before any
> bar is stated — Track C's full-K=64 numbers (21.93 / 7.94 / 63.50) are banned from
> bar usage; **F2** — the write-budget confound is now controlled, not just reported:
> every masking candidate renormalizes selected β to a pinned baseline per-chunk total,
> and the factorial gains a required budget-matched random-selection control cell (2R)
> with a registered decision rule; **M1** — churn-rate and support-size get
> pre-registered bars via the KEY_ANCHORING Rev 5 §3.6 `BANDS_PINNED` house pattern
> (derivation rules stated now, numbers pinned at Wave −1 readout, launcher-gated);
> **M4** — Rev 1 contained a **fabricated quote** (a "cumulative ≈162.5/300" string
> presented in quotation marks as sourced from `SCALE_TRANSFER_DESIGN.md` §5.6, where
> it does not exist) — a serious citation-discipline violation, acknowledged in §12 and
> corrected here with every quoted string grep-verified against its cited source;
> **M5** — the Cell-3 override is re-targeted at the *actual* refusal mechanism
> (`_refuse_if_no_launch` / `sys.exit(3)`, not `selection_mode_for_verdict`); **M6** —
> Cell 4's composition rule is pinned (single selection source, `hard_select_k_sel ==
> geo3_k_sel`); **M7** — a temperature-annealed soft-top-K comparator cell is added so
> a candidate-1 negative separates STE bias from selectivity's own cost; **M8** — a
> geo3-LM full-training stability smoke on the tabular-risk corpus slice is added to
> Wave −1, gating Cells 3/4. **§9 (claim tiers) is FROZEN from Rev 1** — the one attack
> surface that came back clean. **Design only, no GPU, no box access.** This document
> remains `SCALE_TRANSFER_DESIGN.md` §4.2's own registered conditional follow-on —
> quoted verbatim there: *"a hard-zero-β-at-non-selected-positions variant... is noted
> as the conditional follow-on, **requiring its own attack pass before any build**, since
> it reintroduces exactly the kind of hard-masking machinery LM mode was deliberately
> built without."* Nothing in `SCALE_TRANSFER_DESIGN.md` is modified by this document —
> Track B's own §4 stands as written, its Wave −1 gate verdict (`no_launch_redesign`)
> stands as measured, and this document is the redesign that verdict itself routes to.

---

## 0. Reading list this design builds on (context, not repeated here)

- `matrix-thinking/SCALE_TRANSFER_DESIGN.md` §4 — Track B's full geo3-in-LM design: the
  three failure modes free text poses for F-geo-3 (§4.2), the β-gated top-K construction,
  the two registered Wave −1 gate criteria and their routing table (§4.2, outcome (iii)),
  the insertion point (§4.3), required instrumentation (§4.4), cells/budget (§4.5),
  success/failure criteria (§4.6), claim tier (§4.7), attack-yourself (§4.8).
- `EXPERIMENT_LOG.md`, **"SCALE-TRANSFER Track B (2026-07-04): geo3-in-LM built +
  smoke-clean; Wave −1 gate measured on all 6 archived Wave C checkpoints — HARD
  NO-LAUNCH"** — the actual gate measurement this redesign responds to, plus the
  independent audit round (1 MAJOR + 4 MINOR, all fixed) and the two `[LEARN]` blocks
  (the gate's own algebraic-complement degeneracy; the forced-failure-test-precondition
  lesson) both of which this redesign inherits and must not repeat.
- `matrix-thinking/deltanet_rd/results/lm_rd_geo3/wave_neg1_gate.json` — the raw gate
  measurement this document's §1 restates exactly (read directly this session, not
  taken from prose).
- `matrix-thinking/DELTANET_RD_EXACTNESS_DESIGN.md` §14 (F-geo-3's mechanism: why
  Newton–Schulz, the insertion spec, the stability plan, the i-strong-vs-F-geo-3
  distinction), §16 (F-geo-3 results: K=16 bar HIT, K=32 narrowly missed, outcome-F
  attribution — cross-episode drift, not incomplete orthogonalization, is the residual
  bottleneck), **§16.7 including its dated correction** (a shared-scalar simulator input
  bug that silently mis-attributed a K=32 prediction to the wrong K's measured drift —
  the general lesson, restated in §8 item 5 below, is load-bearing for this redesign if
  it ever builds an analogous predictive gate).
- `matrix-thinking/KEY_ANCHORING_DESIGN.md` Rev 5 §1–§2.2 — the 2×2 (orthogonal ×
  stable) mechanism picture this redesign's own scientific readout (§5) mirrors
  directly: neither ingredient alone matters much, orthogonality alone buys most of the
  gap, but the two **together** produce a discontinuous jump to saturation, not a partial
  additive improvement. §3.6/§8 (R4-1) — the override-stamping precedent this redesign
  reuses (§6) for running a known-failing-premise construction as a deliberate reference
  arm.
- `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` — read in full for this design:
  `DeltaNetLMMixer.__init__`/`forward` (`:419–570`), `_geo3_lm_select_and_orthogonalize`
  (`:206–398`, the existing beta-gated/naive-window gather→orthogonalize→scatter
  construction this redesign's candidates 1/2/4 extend), the `num_heads==1`-only scope
  guard on `geo3_active` (`:474–477`).
- `matrix-thinking/deltanet_rd/run_lm_rd_geo3_sweep.py` — **`_refuse_if_no_launch`
  (`:172–183`)** is the actual production hard refusal: given
  `verdict="no_launch_redesign"` it prints a loud diagnostic and calls `sys.exit(3)`,
  invoked in `main()` at `:600` immediately after `load_gate_verdict` (`:598`), before
  any manifest is built. This is the function this redesign's reference-arm cell (§5)
  must explicitly, stampedly override — not bypass silently. (`selection_mode_for_verdict`,
  `:165–169`, which Rev 1 mis-cited as the refusal, is caller-protected dead-code
  defense: its own assert comment reads "no_launch_redesign must be refused by the
  CALLER before this function is ever reached," and it is only called from
  `waveB1_manifest`, `:200`/`:307` — code `main()` never reaches when the verdict is
  no-launch. Corrected per attack finding M5.) Also load-bearing from this file:
  `_PER_STEP_S_PLACEHOLDER_GEO3 = 0.15` (`:94–104`) — the registered 3× geo3-active
  planning placeholder every geo3-active cell in §6 is priced at; and
  `GEO3_N_ITER_BY_K_SEL = {16: 12, 32: 20}` (`:84`) — the §1.1-registered per-K
  Newton–Schulz iteration counts.
- `matrix-thinking/KEY_ANCHORING_DESIGN.md` Rev 5 §3.6 — the **`BANDS_PINNED`**
  pattern (writer / launcher-gate / readout-assertion, with sha256 hashes of the
  reference JSONs and a timestamp-precedes-launch check) that §4.3's pinned bars and
  §5.3's pinned reference band reuse by name.
- `matrix-thinking/TRACKB_REDESIGN_ATTACK_R1.md` — attack round 1 on Rev 1 of this
  document (2 FATAL, 8 MAJOR, 3 MINOR); §12's response map addresses every finding.
- `matrix-thinking/TRACKB_REDESIGN_ATTACK_R2.md` — attack round 2 on Rev 2 (fresh
  agent; F1 verified CLOSED, 8 new MAJOR + 2 MINOR in Rev 2's own additions, plus
  adjudications of §11's open questions); §13's response map addresses every finding.
- `STATE.md` "Hardware" + `SCALE_TRANSFER_DESIGN.md` §5.6 Rev 2.1/2.2 amendments — the
  program-wide 300 GPU-h ceiling and its current consumption, restated with the
  arithmetic in §7 below.

---

## 1. Why this redesign exists — the gate finding, restated exactly

Track B's Wave −1 gate (`SCALE_TRANSFER_DESIGN.md` §4.2) measured, on all 6 archived Wave
C checkpoints (openr1×3 seeds, wikitext×3 seeds, step 6103, `n_params=14,048,896`, pooled
over 12,288 chunk-episodes, `chunk_size=64`), whether the LM's own learned β nominates a
small write-worthy subset per chunk. It does not:

| Criterion | K_sel=16 | K_sel=32 | Bar | Verdict |
|---|---|---|---|---|
| (a) top-K_sel β-mass fraction | 0.309 | **0.569** | ≥0.60 | FAIL (close miss at K=32) |
| Gini (chunk-level, K_sel-independent by construction) | 0.099 | 0.099 | — (shape check) | near-uniform |
| (b) mean β at non-selected positions | 0.387 | **0.363** | ≤0.25 | FAIL |
| (b) non-selected write-mass fraction | 0.691 | **0.431** | ≤0.40 | FAIL |

(`wave_neg1_gate.json`'s own `pooled` block, read directly this session — not a
recomputation.) A concrete way to see how close to uniform this is: at K_sel=16 (25% of
a 64-position chunk), a perfectly uniform β would capture exactly 25% of chunk mass in
the top 16 positions; the measured capture is **30.9%** — barely above uniform. At
K_sel=32 (uniform prior 50%), measured capture is **56.9%** — again barely above
uniform. Gini≈0.099 (0 = perfectly uniform, 1 = all mass on one position) is the same
number under both K_sel because it summarizes the whole-chunk β distribution shape, not
a K_sel-dependent statistic — it is the single cleanest number for "how far from uniform
is this gate," and the answer is: not very far.

**Verdict recorded:** `gate_verdict: "no_launch_redesign"` — per §4.2's own registered
routing, criterion (b) failing is a **hard no-launch regardless of (a)**. Track B's Wave
1 (β-gated or naive-window) does not launch on the as-designed construction. The
mechanism premise F-geo-3 needs — a small top-K subset captures the bulk of write-mass,
so orthogonalizing it controls state geometry — does not hold on this harness's actual
trained β distribution: 43.1% of chunk write-mass sits at non-selected positions,
writing non-orthogonally into the same shared state, at the very K_sel the primary
construction was designed around.

**A build-time subtlety this redesign inherits and must not re-trip.** The gate script's
own literal implementation makes criteria (a) and (b)'s write-mass sub-check exact
algebraic complements over the same chunk-total denominator (`0.60+0.40=1.00`) —
`EXPERIMENT_LOG.md`'s own `[LEARN]` block on this finding notes the "(a) fails, (b)
passes → naive-window primary" branch is consequently unreachable for *any* real β
distribution, not just empirically rare. §4 below inherits this exact-complement
property for the masking-based candidates and shows it becomes *more* degenerate there,
not less — see §4.1.

---

## 2. Design principles governing every candidate below

Two facts jointly constrain what "properly designed" means here, both inherited directly
rather than re-derived:

1. **Orthogonality alone is not the whole story — KEY_ANCHORING's own 2×2 already showed
   this on the synthetic harness.** `KEY_ANCHORING_DESIGN.md` §2.0's four-cell table
   (stable×orthogonal / stable×not / not-stable×orthogonal / not-stable×not, all K=32,
   `d_state=64`, real archived runs) found: neither ingredient alone matters much
   (stable-only ≈ baseline, ~0.005–0.010 at h=4); orthogonality alone buys most of the
   gap (geo3's 44–56× jump, `DELTANET_RD_EXACTNESS_DESIGN.md` §16); **both together is
   not a partial improvement, it is a discontinuous jump to saturation** (0.44→1.0000).
   Track B's failed gate establishes that the LM harness currently lacks even the
   *orthogonality* precondition's own target (a defined write-set to orthogonalize) — but
   the KEY_ANCHORING lesson says that fixing selectivity alone, without geo3, should
   **not** be expected to move much either. §5's 2×2 factorial is designed specifically
   to check whether this same interaction shape reproduces on free-running LM text, not
   just to confirm selectivity can be forced.
2. **The bolt-on lesson (`SCALE_TRANSFER_DESIGN.md` §6.4) applies here with equal force,
   not just to Track D's pretrained-model graft.** Retrofitting a hard/sparse mechanism
   onto Wave C's *already-trained*, continuous-β checkpoint via continued fine-tuning
   risks the exact failure shape that killed the original matrix-CODI program: downstream
   layers co-adapted to the *old* representation's statistics and the bolt-on could not
   out-compete that co-adaptation. Given Wave C's own architecture trains in **≈0.077
   GPU-h/run** (6,103 steps in 274–278s, `SCALE_TRANSFER_DESIGN.md` §5.6's own measured
   throughput), there is essentially no GPU-cost reason to accept the bolt-on risk to
   save wall-clock. **Default for every candidate below: train from random init at Wave
   C's own architecture/corpus/step-budget, never continue-train from an existing Wave C
   checkpoint.** A continued-training retrofit ablation is named as a possible, explicitly
   NOT-default follow-on in §9, not silently assumed superior for being cheaper.
3. **Scope stays at `num_heads=1`, mirroring geo3_active's own registered scope.**
   `lm_pretrain_rd.py:474–477` hard-refuses `geo3_active` with `num_heads>1` because no
   registered cell exercises the H-folding math at scope. None of the candidates below
   are exercised at `num_heads>1` either — the same discipline applies, for the same
   reason (untested-at-scope, not proven-broken).
4. **Write-budget renormalization is REGISTERED and mandatory for every masking
   candidate (NEW, Rev 2 — F2; orchestrator-pinned, not optional).** Masking is a
   write-*budget* change as well as a write-*targeting* change: zeroing 32–48 of 64
   positions removes 43–69% of the baseline's total per-chunk β-mass (§1's own gate
   numbers), and any downstream geometry/loss effect would otherwise be confounded
   between "targeted better" and "wrote less." The registered rule: **after masking,
   the selected positions' β values are rescaled by a per-chunk, per-head scalar so
   total per-chunk β-mass equals `B_pinned` — the unconstrained baseline's own mean
   per-chunk total write mass** — with each rescaled β clamped at 1.0 (the sigmoid
   range `chunk_delta_rule`'s stock convention assumes) and any clamp-induced mass
   shortfall logged per chunk, never silent. `B_pinned` is pinned at Wave −1 from the
   Cell-1 same-instrument re-measurement (§5.3) into the `BANDS_PINNED-TrackB` file
   (§4.3); the planning approximation, derivable from the archived gate JSON's own
   fields (non-selected mass at K_sel=32 = 32 × 0.3625 ≈ 11.60; total =
   11.60 / 0.4308 ≈ **26.9** of a 64-position chunk maximum), implies a mean selected
   β ≈ 0.84 at K_sel=32 — feasible under the ≤1.0 clamp on average, with tail-chunk
   clamping expected and logged. Applies to candidates 1 and 3, candidate 4's
   hard-snap phase, and the §5.1 comparator/control cells; candidate 2's
   sum-to-≤1 sparsemax output is rescaled by the same rule but **structurally cannot
   always reach `B_pinned` under the clamp at small support**.

   **The PARTIAL trigger is SYMMETRIC — one numeric rule, every masking cell
   (REVISED, Rev 3 — NEW-1, orchestrator-pinned; Rev 2 gave candidate 2 a decision
   rule and everyone else logging-only, reintroducing the "reporting is not a
   control" pattern F2 was raised to kill).** Registered rule, identical for
   candidates 1/2/3, candidate 4's hard-snap phase, the comparator, and Cells
   2R/4R: per chunk, `shortfall_c = max(0, B_pinned − Σ clamped β) / B_pinned`; the
   cell is declared **BUDGET-PARTIAL** in every readout that compares it to a
   fully-matched cell if `median(shortfall_c) > 0.10` **or**
   `frac(shortfall_c > 0.10) > 0.25`. The 0.10 chunk-level threshold is Rev 2's own
   candidate-2 rule made universal; the 0.25 exceedance fraction is
   proposed-conservative (same labeling convention as the §5.3 multipliers). Why
   this cannot be pre-derived and must be measured: the attack's own arithmetic
   (reproduced from the gate JSON: pre-renorm mean selected β ≈ 15.30/32 ≈ 0.478,
   mean scale factor ≈ 0.84/0.478 ≈ 1.76×, leaving only ~16 pp of clamp headroom on
   an *average* chunk) shows tail chunks will plausibly clamp for every masking
   candidate, but `wave_neg1_gate.json` records only pooled means and one Gini —
   no per-chunk dispersion — so no source can currently say how often. The Wave −1
   Cell-1 re-measurement (§5.3) therefore additionally records the **per-chunk**
   total-mass and selected-subset β distributions (same forward passes, zero extra
   GPU) and `BANDS_PINNED-TrackB` carries the resulting predicted-shortfall
   distribution as the record every cell's realized shortfall is later read
   against. Residual honesty caveat, unchanged: matched *total* mass over fewer
   positions still means fewer, stronger writes — that redistribution IS the
   selectivity axis under test, not a confound to remove (concurred by attack round
   2's own adjudication of §11 item 6: a mass-conservation identity, not a control
   gap).

---

## 3. Candidate mechanisms

Four candidates, spanning the full range the task asked for: a minimal learned-hard-mask
change (3.1), a differentiable adaptively-sparse alternative (3.2), an architecturally
fixed (non-learned) write schedule that is a genuinely different model (3.3), and a
soft-then-hard annealed compromise this project's own history already predicts will
underperform in its soft half (3.4).

### 3.1 Candidate 1 — Hard top-K β masking with straight-through gradients (PRIMARY)

**Mechanism.** Forward: compute `beta_soft = sigmoid(b_proj(x))` exactly as today
(`lm_pretrain_rd.py:529`); within each `chunk_size=64` window, select the top-`K_sel`
positions **by β magnitude** — literally the same `priority`/`topk_idx` computation
`_geo3_lm_select_and_orthogonalize`'s `beta_topk` branch already implements
(`:292–301`), just reused one call earlier — and construct a binary mask `m∈{0,1}^T`
with exactly `K_sel` ones per chunk (content positions only, EOT/padding hard-excluded
exactly as §4.2 item 3 already specifies). Output: `beta_hard = beta_soft * m` — literal
zero write mass at every non-selected position, satisfying the queued idea's own name
exactly. Backward: a straight-through estimator (Bengio, Léonard & Courville 2013;
the same trick underlying hard-concrete/L0 gating, Louizos, Welling & Kingma 2018) —
`dL/d(beta_soft) := dL/d(beta_hard)`, i.e. gradient flows through the mask as identity,
so `b_proj`'s weights keep receiving a training signal even for positions the current
forward pass zeroed.

**Insertion site (CORRECTED, Rev 2 — N2; Rev 1 misstated the codebase here).**
`DeltaNetLMMixer.forward`, `lm_pretrain_rd.py`, immediately after
`beta = torch.sigmoid(self.b_proj(x))` (`:529`). Rev 1 called this "the exact site
`_geo3_lm_select_and_orthogonalize` is already called from" — wrong: the existing geo3
call sits **later**, at `:539–551`, *after* the q/k/v head reshape (`:531–533`),
operating on the already-reshaped `(B,T,H,head_dim)` k tensor per that function's own
shape contract (`B, T, H, d = k_raw.shape`). The mask, by contrast, applies to `beta`
(shape `(B,T,H)`, unaffected by the q/k/v reshape) and therefore sits at `:529–531`,
upstream of both the reshape and the geo3 call — the *selection logic* is shared with
`_geo3_lm_select_and_orthogonalize`'s `beta_topk` branch (`:292–301`), the *call site*
is not. Concretely: a new sibling function applies the same top-K selection **to
`beta` itself**, which the existing function never touches (it only
gathers/orthogonalizes `k`), followed by the §2 principle-4 renormalization to
`B_pinned`. A new constructor flag, `hard_select_active: bool` +
`hard_select_k_sel: int`, mirrors `geo3_active`/`geo3_k_sel`'s existing
additive-off-by-default pattern exactly. `hard_select_active` and `geo3_active` are
**independently toggleable** — this is the axis §5's 2×2 varies — with Cell 4's
composition rule pinned in §5.1 (single selection source, `hard_select_k_sel ==
geo3_k_sel`).

**STE-bias isolation comparator (Rev 2 — M7; SCHEDULE RE-PINNED, Rev 3 — NEW-6,
orchestrator decision).** A temperature-annealed **soft-top-K comparator** runs
alongside candidate 1 in Wave 1: same top-K selection, but instead of a hard mask +
STE, non-selected β is multiplied by a decay factor `τ(t)` annealed `1→0` linearly
over **the first 10% of steps, then exactly 0 for the remaining 90%** — with the same
`B_pinned` renormalization applied at every `τ`. No gradient estimator is involved:
non-selected positions retain an exact (shrinking) gradient path while `τ>0`. Rev 2's
80%-of-steps anneal was itself a confound (attack NEW-6): a comparator that trains
soft for most of the run and hardens only at the end is candidate 4's
soft-warmup-curriculum idea in disguise, so its success over candidate 1 could not
distinguish "STE was the problem" from "the curriculum was the fix." At the 10% pin,
both arms are hard-masked (candidate 1 exactly; the comparator numerically —
non-selected β multiplied by exact float `0.0`) for ≥90% of training, so the dominant
difference between them is the gradient estimator at non-selected positions during
that shared hard regime: STE's dense biased pass-through vs. the comparator's honest
zero. **Registered residual confound, one sentence:** the first 10% of steps still
differs between the arms (soft window vs. hard-from-init), so the attribution is
"STE bias, up to an early-training-window difference," not an exact isolation. The
τ=0-endpoint forward pass is bit-equivalent to candidate 1's mask (exact-zero float
multiply) — a true but **non-load-bearing** claim: no registered check depends on
bit-level equality, and if any future check ever does, it is registered now as a
numeric bound (max-abs forward diff ≤ 1e-6 on a fixed probe batch) rather than
bitwise identity. Decision rule, unchanged in shape: candidate 1 misses its val-loss
tolerance and the comparator does not → attributed to STE gradient bias (with the
one-sentence caveat above); both miss together → hard selectivity itself is
implicated. Chosen over the alternative (pre-registering candidate-1 negatives as
uninterpretable and promoting candidate 2 to co-primary) because the comparator costs
≈0.5 GPU-h (6 non-geo3 runs) against ample headroom, whereas the alternative would
leave the redesign's PRIMARY candidate structurally unable to produce an
interpretable negative — a bad trade at this budget. Interim discipline until the
comparator reads out: every candidate-1 result is labeled **"hard-masked-with-STE,"**
never bare "hard selectivity."

**Cost.** Cheapest of the four. No new orthogonalization, no new solver, no new
autograd primitive (STE's backward is a single elementwise pass-through). Reuses
already-audited selection code verbatim. Estimated build: well under a day. Wave −1
manipulation-check cost: §4.1 below.

**What failure means.** (i) Val loss regresses sharply relative to an unconstrained
control → STE's biased gradient (treating a genuinely discontinuous mask as identity in
the backward) is too crude an approximation for this optimization landscape — a known,
named STE pathology, not a surprise if it occurs. (ii) The selected-position **set**
churns heavily step-to-step (a position flips in/out of top-K as β shifts by noise near
the K_sel-th-ranked boundary) — a distinct, second failure signature from (i), requiring
its own diagnostic (§4.1, §8 item 2), not conflated with ordinary training instability.
(iii) The mask converges to a **content-independent positional artifact** (e.g., always
selects position 0 mod some period, regardless of token identity) — a fake positive on
the gate metrics (Gini rises, non-selected write-mass falls) without buying the
content-selectivity geo3's mechanism actually needs. This failure mode is the more
dangerous sibling of churn (a frozen positional collapse shows near-ZERO churn and
sails under the churn ceiling), and it now has a fully specified, pinned bar: §4.3's
positional-concentration statistic (Rev 3 — NEW-5; Rev 2's citation here pointed at
an unrelated passage and no spec existed behind it).

### 3.2 Candidate 2 — Entmax/sparsemax β (adaptively sparse, differentiable exact zeros)

**Mechanism.** A genuinely different differentiable primitive from candidate 1's
hard-then-STE approach: replace the independent, per-position sigmoid with a
**chunk-normalized, adaptively sparse** transform. Compute a raw per-position score
`s = b_proj(x)` exactly as today, but instead of an elementwise sigmoid, apply
**sparsemax** (Martins & Astudillo, ICML 2016 — Euclidean projection onto the simplex,
closed-form, exact zeros for below-threshold entries) or **entmax** (Peters, Niculae &
Martins, ACL 2019 — a tunable-sparsity generalization via bisection) **over the
`chunk_size` axis, per head, per chunk** — i.e., β becomes a distribution over the
chunk's 64 positions that sums to (at most) 1, with exact, input-dependent zeros rather
than a fixed, hand-chosen K. This is a **structural change to the model**, not merely an
added selection step on top of an otherwise-identical architecture: today's independent
sigmoids let a chunk write anywhere from near-zero to near-`T` total β-mass; a
chunk-normalized sparsemax/entmax bounds total per-chunk write mass to ≤1 by
construction — a different write-budget regime, flagged explicitly so any downstream
effect is not silently misattributed to "selectivity" when it may instead be "budget."

**Insertion site.** Replace `self.b_proj` + `torch.sigmoid(...)`
(`lm_pretrain_rd.py:494`, `:529`) with the same linear layer producing raw scores,
followed by a **new primitive** — this codebase has no sparsemax/entmax implementation;
sparsemax's simplex projection is a ~10-line sort-and-threshold with a documented,
analytic backward (no external dependency needed, matching this repo's "reuse audited
code, hand-roll simple closed forms" discipline over taking an unaudited PyPI
dependency); entmax's bisection variant is more code for a tunable sparsity knob, ranked
as a stretch, not the primary sub-choice, for exactly that reason.

**Cost.** Moderate. New differentiable primitive requiring its own Wave −1
gradient-finiteness smoke (mirroring geo3's own §14.6 item 2 convention exactly) —
sparsemax's gradient is **zero outside the support** by construction (a documented,
not-a-bug property, Martins & Astudillo 2016 §3), which could starve `b_proj`'s weights
for non-selected positions of *any* gradient at all — a real, qualitatively different
risk from STE's dense (if biased) gradient in candidate 1, and must be checked, not
assumed benign. Also requires verifying `chunk_delta_rule`'s own derivation only assumes
`β≥0` (compatible) and does not implicitly assume independent per-position β
unconstrained by a chunk-sum budget (unverified — a build-time check, not an assumption).

**What failure means.** (i) The chunk-sum-≤1 constraint shrinks total write mass well
below the baseline's independent sigmoids — **now controlled, not just reported (Rev
2 — F2):** candidate 2's output is rescaled toward `B_pinned` per §2 principle 4, with
the registered PARTIAL-match declaration if clamping prevents full budget matching at
small support; a residual val-loss regression under a declared-PARTIAL budget match is
reported as jointly attributable (underwriting + selectivity), never silently assigned
to either. (ii) The natural sparse support could still be too broad if raw scores
cluster closely — entmax's `α` knob (or a temperature on the sparsemax input) would
need its own escalation, mirroring geo3's own `n_iter=12→20` escalation precedent
(`DELTANET_RD_EXACTNESS_DESIGN.md` §14.4) — one pre-registered escalation, hard cap,
never open-ended tuning; the pass/fail bars for support size itself are pinned via
§4.3's `BANDS_PINNED-TrackB` machinery, not left as report-only.

### 3.3 Candidate 3 — Hand-specified (non-learned) k-hot write schedule, restoring `model_rd.py`'s own audited buffer/mask convention on free text

**Mechanism.** This candidate does not try to make the continuous β gate sparse at all
— it abandons *learned* write-selection and restores the **synthetic harness's own
already-audited, already-proven convention**: `model_rd.py`'s `DeltaNetRDBlock` reserves
a fixed BUFFER token and an architecturally-zero β at all non-write positions
(`lm_pretrain_rd.py`'s own docstring names this exact machinery as "the one deliberate
subtraction from `model_rd.py`'s audited block per this build's brief"). Candidate 3
re-adds it, applied to free-running text via a **preprocessing step, not a model
change**: insert a reserved write/buffer token every `W` real tokens (a fixed period,
e.g. `W=chunk_size/K_sel`), with β architecturally pinned to zero everywhere else and to
a real, learned value only at reserved positions — an actual `item_pos` analog for real
text, constructed by fiat rather than nominated by a learned quantity. This directly
answers the task brief's "trained-from-scratch small LM with k-hot write gating": the
"k-hot"-ness here is **architectural, not learned** — no STE, no sparsemax, no gradient
approximation for the mask itself, since the mask is fixed before training starts.

**Honesty check, stated plainly.** This candidate does **not** escape Track B's original
design problem (§4.2 item 2: "what is the orthogonalization set for free-running
text?") — it answers it by fiat (a periodic schedule) rather than learning it, trading
the original problem for a different, cruder one: a fixed period has no reason to align
with where content-meaningful "write events" actually occur in real text, and this
candidate cannot claim its selected positions are semantically write-worthy the way
candidates 1/2's β-nominated positions at least attempt to be.

**Insertion site.** A new corpus-preprocessing/tokenization adapter (upstream of
`lm_pretrain_rd.py`'s existing data pipeline, `rebuild_lm_corpora_rd.py`-adjacent, not a
`DeltaNetLMMixer.forward` change) that inserts reserved tokens at a fixed period into the
GPT-2-tokenized stream, plus reinstating `model_rd.py`'s own existing
buffer-token/hard-β-mask machinery (already built and audited there) rather than
`lm_pretrain_rd.py`'s plain sigmoid gate. This is architecturally closer to
`model_rd.py`'s block than to `lm_pretrain_rd.py`'s — a genuinely different model
class, not a flag on the existing one.

**Cost.** Lowest *mechanism*-audit risk of the four (the write-mask/buffer machinery
itself is already fully audited on the synthetic harness) but the highest *scope*
change: a new preprocessing pipeline, a new tokenization-adjacent build item, and its
own from-scratch training manifest (no retrofit path exists or is even attempted here —
this candidate cannot be applied to Wave C's existing tokenized data without
re-tokenizing it). GPU cost is trivial regardless (Wave C's own ≈0.077 GPU-h/run scale).

**What failure means.** (i) If a fixed period simply never aligns with anything a
downstream reader would call a meaningful write event, val loss regresses for a reason
that has nothing to do with orthogonality or stability — a **schedule-design** failure,
not a mechanism failure, and should be reported as such (candidate 3 failing does not
bear on candidates 1/2/4's learned-selectivity approach at all). (ii) If it trains fine
and clears the manipulation check trivially (guaranteed by construction, §4.1), the
informative content is entirely in §5's 2×2 factorial, not in the gate.

### 3.4 Candidate 4 — Auxiliary selectivity loss with a hard-mask schedule (RANKED LAST, ablation-only)

**Mechanism.** Keep β architecturally unchanged (plain, continuous, independent
sigmoid) but add an auxiliary loss term during training that directly penalizes
deviation from the gate criteria themselves — e.g. `L_sel = -Gini(β_chunk)` or a direct
penalty on the non-selected write-mass fraction, λ-scheduled from 0 upward (mirroring
this project's own established λ-warm-in convention, cited from `SCALE_TRANSFER_DESIGN.md`
§6.4 item 1 and from F-geo-1's own `L_orth` penalty, `DELTANET_RD_EXACTNESS_DESIGN.md`
§14.0 item 2). Optionally, snap to a literal hard top-K mask (candidate 1's own
mechanism) in a final phase once the soft loss has concentrated β somewhat.

**Why this is ranked last, not merely listed fourth.** This project has **already run**
a structurally identical experiment on the adjacent axis (key-orthogonality rather than
β-selectivity): F-geo-1's soft `L_orth` penalty, at `λ∈{0.1,1.0}`, saturated at a 3–8%
cleanup of key-Gram deviation — nowhere near i-strong's exact 0.000 ceiling, and
`λ=0.1` vs `1.0` were nearly identical (`DELTANET_RD_EXACTNESS_DESIGN.md` §14.0 item 2).
`KEY_ANCHORING_DESIGN.md` §2.4 independently ranks its own directly analogous soft
regularizer candidate ("Explicit cross-episode drift regularizer `L_anchor`") **"soft
tier, ranked LAST, ablation-only"** for exactly this reason. Candidate 4's soft phase is
the same strategy shape applied to a different quantity (β-concentration instead of
key-orthonormality) with no house precedent suggesting it would behave differently.
Only the hard-snap final phase — mechanically identical to candidate 1 — has a real
chance of reaching a literal-zero bar; the soft phase alone is a registered, before-any-
data prediction of underperformance, not an open question.

**Insertion site.** No architecture change to `DeltaNetLMMixer.forward`; a new loss
term in the training loop reading `beta` (already exposed via the existing
`geo3_last_diag`-style side channel convention, `lm_pretrain_rd.py:489`) alongside the
existing next-token cross-entropy. The optional hard-snap phase reuses candidate 1's
code verbatim.

**Cost.** Cheapest to build incrementally (a loss-term addition, no new inference-time
architecture) but the most expensive to *calibrate* — λ-schedule search is the same
open-ended problem F-geo-1's own λ calibration already explored and found saturating,
unlike candidates 1/2/3 which are architecturally deterministic once `K_sel`/period is
fixed.

**What failure means.** If the soft phase alone, measured against the same gate
criteria, still fails — unsurprising, and should be reported as an independent
confirmation of the F-geo-1 finding at a new target (β-selectivity rather than
key-orthogonality), not as a novel negative result. The only open question this
candidate can answer that candidate 1 alone cannot is whether soft warm-up makes the
*subsequent* hard phase's optimization measurably easier or more stable than a
from-cold hard mask — a legitimate, but secondary, readout (§9).

---

## 4. Measurement plan — the Wave −1 gate as a manipulation check

The task brief is explicit: the gate that failed becomes the manipulation check — a
candidate must move Gini/write-mass past the pre-registered thresholds **or it isn't a
test**. Applying this literally across all four candidates surfaces an asymmetry that
must be registered before any run, not discovered after.

### 4.1 Trivial-by-construction vs. genuinely empirical (the asymmetry)

**Candidates 1 and 3 pass the original numeric bars by tautology, not by discovery.**
Once non-selected β is architecturally forced to exactly zero, `top_k_mass_frac = 1.0`,
`mean_nonselected_beta = 0.0`, and `nonselected_write_mass_frac = 0.0` **identically, for
any input, by construction** — re-running the exact same probabilistic gate script on a
hard-masked model is not a manipulation check in the causal-inference sense at all, it
is a unit-test-shaped assertion dressed as an experiment. Restating §1's inherited
`[LEARN]` finding: the original gate's (a)/(b) are exact algebraic complements over one
denominator; under hard masking this collapses further — both criteria degenerate to a
single trivial yes/no ("is the mask exactly `K_sel`-sparse") carrying zero information
beyond what a construction-level assertion already guarantees.

**The real manipulation check for candidates 1 and 3 is therefore reframed as a
training-viability check, not a geometric-threshold check:** (i) does the model
converge to a reasonable val-loss floor at all under the forced constraint (STE bias for
candidate 1; a schedule-misalignment risk for candidate 3); (ii) is the gradient
everywhere finite (mirroring geo3's own §14.6 item 2 convention); (iii) for candidate 1
specifically, the selection-set churn rate — a genuinely open,
not-construction-guaranteed quantity, now gated by a pre-registered numeric bar pinned
via §4.3's `BANDS_PINNED-TrackB` machinery (Rev 2 — M1; Rev 1 registered this metric
with no pass/fail rule, exactly the degenerate-model gap the attack round named).

**Candidates 2 (entmax/sparsemax) and 4's soft phase are where the original bars are
genuinely informative**, because neither guarantees anything about the gate metrics by
construction: sparsemax/entmax's support size is content- and score-dependent, not fixed
to `K_sel`; the soft auxiliary loss (candidate 4) makes no hard guarantee at all before
any hard-snap phase. Re-running the exact original probe script on these two candidates
is a real test of whether the mechanism concentrates β enough — closer in spirit to the
original gate's own intent (measuring an emergent property of a learned mechanism)
than candidates 1/3 allow.

**A registered caveat specific to candidate 2:** the inherited top-`K_sel`-mass-fraction
metric implicitly assumes a fixed `K_sel`, which sparsemax/entmax do not respect — a
genuinely sparse-10 output at K_sel=32 would read as ~100% "top-32 mass captured"
despite having a much narrower true support than K_sel implies. Candidate 2's Wave −1
measurement must **additionally** report the actual support size (count of exactly-zero
entries), not rely on the inherited metric alone — and, per Rev 2 (attack M1; §8 item
5's disposition), support size is not merely reported but gated against §4.3's pinned
band.

### 4.2 What gets measured, per candidate, and against which bars

| Candidate | Manipulation-check content | Bars |
|---|---|---|
| 1 (hard top-K + STE) | Training viability (val loss finite/reasonable, gradient-finite) + selection-churn rate | No construction-trivial gate re-run reported as an experimental finding; churn rate is the real diagnostic, gated by the §4.3 pinned churn ceiling |
| 2 (entmax/sparsemax) | Original gate metrics (genuinely empirical here) **plus** actual support size | Original thresholds (≥60% top-K mass / ≤0.25 mean non-selected β / ≤0.40 non-selected write-mass) reported descriptively; support size gated by the §4.3 pinned band [floor pinned at Wave −1, ceiling = K_sel a priori] |
| 3 (fixed schedule) | Training viability only (gate metrics are construction-trivial: selected/non-selected split is fixed by the preprocessing step, not learned) | Same training-viability bars as candidate 1 (no churn bar — the schedule cannot churn by construction) |
| 4 (soft phase) | Original gate metrics — genuinely empirical, predicted (per §3.4) to still fail | Original thresholds; failure here is an expected, informative replication of the F-geo-1 pattern, not a surprise |
| 4 (hard-snap phase) | Same as candidate 1 once snapped | Construction-trivial, same reframing as candidate 1; inherits candidate 1's pinned churn ceiling |

### 4.3 Pinned bars for churn-rate and support-size — the `BANDS_PINNED` pattern (NEW, Rev 2 — M1, orchestrator-pinned)

Rev 1 registered churn-rate and support-size as required *measurements* with no
pass/fail rule — the attack round correctly identified this as the degenerate-model
hole ("selective by construction, useless in function" could pass every registered
gate). Rev 2 closes it by reusing `KEY_ANCHORING_DESIGN.md` Rev 5 §3.6's
**`BANDS_PINNED`** house convention wholesale: derivation rules stated NOW (before any
data), numbers pinned at Wave −1 readout (before Wave 1 launches), enforced by the
launcher, with the blind mechanically verified.

**Metric definitions (registered now).**
- **Churn rate (candidate 1 + candidate 4's hard-snap phase):** at each 200-step log
  point (the trajectory-logging convention's own resolution), the mean over
  (chunk, head) episodes of `|S_t \ S_{t−1}| / K_sel`, where `S_t` is the selected
  position set for a fixed, corpus-seeded probe batch (same batch every log point, so
  churn measures the mechanism, not data variation).
- **Support size (candidate 2):** per (chunk, head), the count of strictly-positive β
  entries; summarized as the median and p10 over the same fixed probe batch.
- **Positional concentration (candidates 1/2 + candidate 4's hard-snap phase — NEW,
  Rev 3, NEW-5):** over the fixed probe batch, the marginal selection frequency of
  each intra-chunk offset `o ∈ {0..63}` (`f_o` = fraction of (chunk, head) episodes
  selecting offset `o`); the statistic is the **total-variation distance between the
  normalized offset marginal `p_o = f_o / Σf` and uniform(1/64)**. Content-driven
  selection across many documents keeps the offset marginal near-uniform; a
  content-independent positional artifact concentrates it. Measured at the 50% and
  final checkpoints of every gated run.

**Derivation rules (registered now; numbers pinned at Wave −1 readout).**
- **Churn ceiling — the MAX of TWO nulls (REVISED, Rev 3 — NEW-4,
  orchestrator-pinned).** Null A (retained from Rev 2): `mean_ref + 2·s_ref` over the
  **unmasked, Cell-1-architecture reference pilot's** implicit top-K_sel-by-β churn
  at its last 5 log points (steps 1,200–2,000, past init transients) — non-circular
  (the reference is the soft ranking the mask thresholds), but measured on a
  categorically different training regime. Null B (NEW): `mean + 2·s` over the
  **gated run's own churn at the log points inside its first 10% of steps** (for a
  6,103-step run: pairs at steps 200/400/600) — same mechanism, same regime, but an
  early-training window that includes init transients. Registered ceiling for a
  given run = **max(pinned Null A, that run's own Null B)**. Registered calibration
  limitation, one sentence (orchestrator-required): both nulls are approximations —
  Null A cannot see mask-specific dynamics (STE entrenchment could suppress churn,
  legitimate selection-sharpening could raise it, in either direction), and Null B
  overweights init transients — so the ceiling is a coarse degeneracy screen, not a
  calibrated bound, and a breach routes to diagnosis, never to silent acceptance or
  rejection. Wave-1 candidate-1 cells whose steady-state churn
  (checkpoint-resolution, past the first 10%) exceeds the ceiling are flagged
  **selection-degenerate**: excluded from Cell-4 inheritance, reported descriptively.
- **Support band:** ceiling = `K_sel` **a priori** (structural — candidate 2 must
  reach at least the sparsity regime the matched-K_sel factorial requires; if its
  pilot median exceeds K_sel, the §3.2 registered single temperature escalation fires,
  hard cap, then the candidate is cut if still over). Floor pinned at Wave −1 readout:
  `support_floor = max(4, p10_pilot)` — the pilot's own 10th-percentile support at its
  final log point, floored at 4 (below 4 selected writes per chunk, geo3
  orthogonalization is near-trivial and the matched-K_sel comparison is void). Wave
  1's full runs must keep median support inside `[support_floor, K_sel]`; a drop below
  the pinned floor at any checkpoint past 20% of training = **degenerate-collapse
  flag**, cell excluded from the factorial, reported descriptively. This is the
  KEY_ANCHORING pattern's exact shape: the pilot is the reference arm, the pinned
  numbers gate the later full-scale cells against drift from what the pilot
  established.
- **Positional-concentration ceiling (NEW, Rev 3 — NEW-5):**
  `pos_ceiling = mean_ref + 2·s_ref` over the unmasked reference pilot's own
  TV-from-uniform at its last 5 log points — the same reference-null shape as churn
  Null A, and the right null for the same reason: the baseline's β already has some
  organic positional structure (conv-boundary effects), and the bar catches **excess**
  concentration beyond it, not all structure. A gated run whose final-checkpoint TV
  exceeds `pos_ceiling` is flagged **positionally degenerate**: excluded from Cell-4
  inheritance, reported descriptively. Registered limitation, one sentence: a
  TV-from-uniform statistic cannot distinguish "content-driven but positionally
  biased" from "purely positional" — a run flagged here gets the finer document-
  dependence diagnostic (selected-set overlap across documents at matched offsets)
  as follow-up analysis, not as a registered bar. Together with churn this closes
  both halves of the degenerate-selection risk: a frozen positional collapse shows
  near-zero churn but maximal positional concentration; an unstable selector shows
  the reverse — each bar catches the failure mode the other is blind to.

**Mechanics, reused from §3.6 by name, all three parts:** (1) **the writer** — Wave
−1's harness writes `BANDS_PINNED-TrackB.json` (pinned churn Null A, support floor,
positional-concentration ceiling, `B_pinned` + the per-chunk predicted-shortfall
distribution from §2 principle 4, the recomputed K_sel anchors from §5.3, per-pilot
input values, sha256 hashes of the pilot result JSONs, timestamp) only after every
pilot validates as complete; (2) **the launcher gate** — Wave 1/2 cells REFUSE to
launch unless the file exists, validates, and re-hashes clean; (3) **the readout
assertion** — the analysis script asserts the pin timestamp strictly precedes the
earliest Wave-1 start time, else every affected readout demotes to descriptive tier.
No provisional numbers exist to void: Rev 2 deliberately states no placeholder churn
or support values at all — the derivation rules above are the only registered
objects until Wave −1 writes the file.

---

## 5. Pre-registered bars for the scientific readout — the 2×2 factorial

Per the task brief and the KEY_ANCHORING precedent (§2, principle 1): the actual
scientific question is not "can selectivity be forced" (§4 already answers that
per-candidate) but **"does forcing selectivity, combined with geo3-style
orthogonalization, improve the LM's memory-fidelity/attractor metrics beyond either
ingredient alone"** — mirroring the exactness program's own 2×2 logic exactly.

### 5.1 The four cells

| | Continuous β (no hard selectivity) | Hard-selective β |
|---|---|---|
| **No geo3** | **Cell 1 (baseline)** — the archived Wave C checkpoints (already exist, `n_params=14,048,896`, `d_model=256, d_state=64, n_layers=2`), **re-measured at Wave −1 with the §5.2 unified instrument** (F1 — Track C's whole-chunk number is no longer this cell's citable value) | **Cell 2 (selectivity-only)** — the surviving candidate(s) from §4, `geo3_active=False`, β renormalized to `B_pinned` (§2 principle 4). **Cell 2R (budget-matched random control, REQUIRED — Rev 2, F2; cadence PINNED, Rev 3 — NEW-2):** a β-blind *random* K_sel-subset selection, **resampled independently for every (chunk instance, training step)** from an RNG stream keyed to the run seed and step counter — never a fixed function of intra-chunk position — renormalized to the same `B_pinned`: same count, same total mass, zero targeting information |
| **geo3 applied** | **Cell 3 (geo3-only, reference arm)** — Track B's **original, already-built** construction (`--use-geo3-lm`, `beta_topk`), the exact construction whose own Wave −1 gate returned `no_launch_redesign` | **Cell 4 (target)** — the surviving candidate + `geo3_active=True`, geo3 orthogonalizing the mask's own write set under the composition rule below |

**Why per-step resampling is the registered cadence, and what the alternative would
have been (Rev 3 — NEW-2):** a fixed random draw of "which K_sel-of-64 offsets"
applied identically to every chunk would silently degenerate Cell 2R into a fixed
positional schedule — structurally candidate 3's non-learned mask, and exactly the
§3.1(iii) frozen-mask artifact the round-1 record already worries about — corrupting
the one comparison (2 vs 2R) the F2 control exists to make. Per-instance resampling
is the only cadence that delivers the cell's own stated property ("zero targeting
information"). A **fixed-mask variant** (one frozen random offset draw for the whole
run) is named as a separate, cut-eligible **diagnostic** cell — informative about the
pure positional-structure effect, NEVER usable as the budget-matched control.

**Cell 2R's registered decision rule (REVISED, Rev 3 — NEW-3, orchestrator-pinned;
Rev 2's "matches within seed noise" had no operationalization and borrowed
KEY_ANCHORING's n=3 headcount without its one-sample-vs-band power analysis).**
Evaluated per corpus on the primary metric (§5.2's same-instrument Gram deviation),
per-seed values, n=3 per arm:
- Cell 2's seed range (min..max) and Cell 2R's seed range **disjoint, Cell 2
  better** → β-informed targeting matters; "β-informed selectivity" language
  licensed.
- Ranges **disjoint, Cell 2R equal-or-better** → the gain is write concentration;
  every §5.3 readout downgrades "β-informed selectivity" to "write concentration,"
  and Cell 4's interaction claim inherits the downgrade.
- Ranges **overlap → INCONCLUSIVE**: reported as such, **no downgrade and no
  targeting claim** — readouts use the registered neutral phrasing "selective
  writing (targeting-vs-concentration unresolved at n=3)."
The headline verdict requires the same outcome in both corpora; a split is reported
as INCONCLUSIVE overall. **Low-power acknowledgment, registered rather than
implied:** under the exchangeability null, complete range separation at n=3-per-arm
has probability 2/C(6,3) = **0.10 per corpus** — a lenient α, and the test's power
against small real differences is poor; INCONCLUSIVE is the *expected* outcome for
small effects and is an honest report, not a failure. This comparison is a crude
screen, not a hypothesis test with a power target — the KEY_ANCHORING n=3 precedent
was calibrated for a one-sample-vs-pinned-band read, not this two-sample contrast,
and Rev 3 does not pretend otherwise.

A **Cell 4R** (random selection + geo3, budget-matched — the geo3-side targeting
control, same per-step resampling cadence as 2R) is registered as a RESERVE cell,
run only if Cell 4 clears the interaction bar, to test whether targeting matters
once orthogonalization is present; priced in §6.1's wide case, cut-eligible.

**Cell 4 composition rule (PINNED, Rev 2 — M6, orchestrator decision).**
`hard_select_k_sel == geo3_k_sel` is **REQUIRED** (constructor-asserted), and in Cell
4 geo3's own `beta_topk` selection is **REPLACED by the mask's selection** — the
mask's selected indices are threaded directly into
`_geo3_lm_select_and_orthogonalize` (a new forced-selection argument), which then
performs only its gather→orthogonalize→scatter role. One selection source, no
double-selection. Justification: at equal K the post-mask `beta_topk` re-ranking
would reproduce the mask's set anyway (non-selected β is exactly zero, selected β is
positive, so topk recovers the mask deterministically up to exact-zero ties) — but
relying on that argument leaves the equality implicit and tie-handling
implementation-defined, and any `geo3_k_sel > hard_select_k_sel` configuration would
force geo3 to orthogonalize functionally inert zero-β positions (orthogonalization
budget spent on writes that contribute nothing to the state). Making the single
source structural closes both cases by construction and makes §5.3's "interaction"
unambiguous: Cell 4 tests *mask-selection + orthogonalization of exactly the masked
write set* — selectivity supplies the target, geo3 supplies the geometry, neither
re-decides the other's job. **EOT/padding validity threading (NEW, Rev 3 —
MINOR-2):** `_geo3_lm_select_and_orthogonalize`'s current validity mask
(`valid_sel = topk_val > neg_inf/2`, `lm_pretrain_rd.py:301`) is derived from its own
internal `topk` output and has nothing to compare against once the forced-selection
argument bypasses that `topk` — under forcing, validity is instead derived by
**gathering `content_mask` at the forced indices**, feeding the same downstream
degenerate-episode machinery (zero-row handling, masked-identity residual,
fallback denial) unchanged. Registered build smoke: a forced index pointing at an
EOT position must come out invalid and take the existing no-op scatter path — the
same negative-case discipline the function's own smoke [8] items already apply to
the topk path.

Cell 3 is included **only** as the pre-registered 2×2 reference arm the interaction
test requires, not as a viable standalone candidate — it is already known (§1) to
violate its own mechanism premise, and is expected to underperform accordingly.
**Override mechanics (CORRECTED, Rev 2 — M5; Rev 1 targeted the wrong function).**
Running Cell 3 requires a deliberate, logged override of the **actual** production
refusal: `run_lm_rd_geo3_sweep.py::_refuse_if_no_launch` (`:172–183`), which
`main()` calls at `:600` — immediately after `load_gate_verdict` (`:598`) and before
any manifest is built — and which exits the process (`sys.exit(3)`) on
`no_launch_redesign`. (Rev 1 cited `selection_mode_for_verdict` (`:165–169`); that
function's assert is caller-protected dead-code defense — its own comment reads
"no_launch_redesign must be refused by the CALLER before this function is ever
reached" — and it is only invoked from `waveB1_manifest` (`:200`/`:307`), code the
no-launch path never reaches. Patching it would not clear the real gate.) The
registered override: a new explicit flag (e.g. `--accept-no-launch-reference-arm`)
that (i) does NOT skip `load_gate_verdict`'s config cross-validation (`:144–161` —
the chunk-size/K_sel/gate-cell checks stay live), (ii) converts
`_refuse_if_no_launch`'s exit into a loud, logged proceed for the registered
reference-arm manifest ONLY, and (iii) threads a stamp into every spawned Cell-3
run's result JSON **at assembly time, never post-hoc** — `gate_override: true`,
`gate_override_reason: "reference arm, SCALE_TRANSFER_DESIGN.md §4.2 outcome (iii)"`,
`gate_override_at: <timestamp>`, `claim_tier: "descriptive"` — mirroring
`KEY_ANCHORING_DESIGN.md` Rev 5 §3.6 item 2 / R4-1's `claim_tier`/`unblind_override`
stamping precedent exactly (which itself adopts `lm_pretrain_rd.py::_assemble_result`'s
working per-run-JSON convention), so no one reading a Cell-3 JSON in isolation can
mistake it for a passing-gate primary run. Non-override runs write
`gate_override: false` at assembly, so field absence is never ambiguous.

**Cells 3 and 4 are additionally gated on the Wave −1 geo3-LM stability smoke
(Rev 2 — M8; SLICE RE-SPECIFIED FROM THE ACTUAL FAILURE MECHANISM, Rev 3 — NEW-7,
orchestrator-pinned).** The code-documented NaN path (`lm_pretrain_rd.py:349–353`)
requires **≥~6 exactly-duplicated rows among a chunk's β-SELECTED top-K keys** —
not merely repetition somewhere in the window. Keys duplicate exactly when their
conv contexts do: `k` at position `t` is a function of tokens `t−3..t` through
`k_conv1d` (`conv_size=4`), so at layer 0 (where upstream features are pure token
functions), **identical token 4-grams within one chunk produce exactly identical
key rows**. Rev 2's slice criterion (window-level 4-gram repetition fraction) was
only a proxy: a window can rank high while its repeated spans never land among the
positions the evolving learned β selects — a smoke on that slice could pass without
ever exercising the failure regime (false reassurance on exactly the gate that
protects Cells 3/4). The Rev 3 construction, from the mechanism itself:
1. **Slice selection:** rank openr1 windows by `n_dup_max` = the largest count of
   positions *within a single 64-token chunk* sharing one identical token 4-gram;
   the stress slice = windows containing a chunk with `n_dup_max ≥ 8` (comfortably
   past the audit-measured ≥~6 onset), selected deterministically under the
   corpus-fixed-seed convention.
2. **Positive-control requirement (the smoke must prove it reached the regime, per
   the house forced-failure lesson — a passing smoke whose failure precondition
   never held is vacuous):** the smoke logs, per forward call, the count of
   exactly-duplicated rows **within the selected top-K set specifically** (a
   caller-side check on the gathered rows via the diag's existing `_topk_idx`).
   Registered floor: ≥25 forward calls during the smoke with ≥6 duplicated selected
   rows. If the floor is met, the smoke's pass/fail bars apply; **if β never
   selects the duplicated positions and the floor is missed, the smoke is declared
   NON-PROBATIVE (not "passed")** and the registered fallback fires: re-run the
   smoke with selection FORCED onto the duplicated positions via the M6
   forced-selection argument (which Cell 4 needs built anyway), guaranteeing the
   eigh-fallback/duplicate-Gram path is exercised.
3. **Pass bars, unchanged from Rev 2:** zero non-finite losses at completion AND
   skip-rate (the isfinite-grad guard's own counter) ≤1% — evaluated only on a
   probative smoke.
Any NaN or skip-rate breach = a **registered stability finding**, recorded in the
wave summary, and Cells 3/4 do not launch until a fix wave with its own independent
audit addresses or bounds it — this is the first-ever multi-thousand-step LM-mode
geo3 training anywhere in the project (all prior evidence is synthetic-harness
training, forward-only LM probes, or short smokes), and the risk its own code
comments flag "for Wave 1 monitoring" must be probed before, not during, the
factorial's spend.

### 5.2 Metrics — what "memory-fidelity/attractor metrics" means at LM scale

**ONE instrument for every factorial cell (REGISTERED, Rev 2 — F1,
orchestrator-pinned).** Every bar-carrying Gram-deviation number in this design —
Cells 1, 2, 2R, 3, 4, and 4R alike — is produced by the **same** instrument: Track
B's §4.4 selected-key logger, computed over exactly `K_sel` keys per (chunk, head)
episode. For cells with a selection mechanism, the measured set is that cell's own
selected/orthogonalized set; for Cell 1 (no selection mechanism exists), the
instrument applies the identical `beta_topk` top-K_sel selection **read-only at
measurement time** to the baseline's own β — so every cell's statistic is a
K_sel×K_sel Gram deviation over positions nominated by the same rule family. Gram
deviation over K=64 keys and over K_sel=32 keys are **not the same statistic** (the
Frobenius deviation from `I_K` has no K-invariance), which is exactly the
instrument-mismatch Rev 1's bars silently committed — no cell may cite Track C's
whole-chunk K=64 numbers for any bar (they survive below only as explicitly
non-citable context). The LM has no natural hop/recovery structure the synthetic
grammar has, so the readout is scoped honestly to:

1. **Per-chunk key-Gram deviation over the K_sel selected/orthogonalized keys** (the
   unified instrument above) — reusing `rank_utils.py`/`model_rd.py`'s existing
   Gram-deviation computation, per Track B's own required build item (§4.4). This is
   the only instrument §5.3's bars read.
2. **Cross-chunk drift diagnostic for repeated tokens** — the free-text analog of
   F-geo-3's entity-drift diagnostic (§4.4: "how much a token's orthogonalized key
   direction changes across the differently-composed chunks it appears in"), with §4.4's
   own caveat restated: free text's chunk membership is far less stable than a synthetic
   K-cycle's fixed episode membership, so this instrument is expected to be
   noisier/worse-behaved than the synthetic-harness case it is ported from, not assumed
   to transfer cleanly.
3. **Effective/stable rank** of the intervened region (`rank_utils.py`, unmodified).
4. **Bonus, conditional, never required:** a recovery-style metric, only if Track C's
   own frontier-probe transplant (`SCALE_TRANSFER_DESIGN.md` §5.5 item 2) has separately
   validated at rung 1 by the time this redesign runs — an explicit **dependency on a
   different track's own gate**, not something this redesign controls or may assume.

### 5.3 Numeric bars (REWRITTEN, Rev 2 — F1: bars are now registered as derivation rules over same-instrument, same-K_sel quantities pinned at Wave −1, never over cross-instrument numbers)

**K_sel is PINNED: `K_sel = 32`, single value, for the entire Wave 1 / 2×2 manifest
(Rev 2 — N1; reason (ii) restated honestly, Rev 3 — R2 MINOR-1).** Four reasons, in
order: (i) the failed gate's own registered gate cell is K_sel=32
(`wave_neg1_gate.json`'s `gate_k_sel: 32`; the sweep's own config cross-validation
asserts `gate_k_sel == max(K_SELS)`); (ii) **two readings of the gate data exist and
they point in opposite directions** — by *absolute proximity to the registered 0.60
bar*, K=32 came closest (0.569 vs 0.60); by *relative excess over each K's own
uniform floor*, K=16 has the stronger discriminative signal (30.9% vs a 25% floor =
+23.7% relative, against K=32's 56.9% vs 50% = +13.8% relative). Rev 2 stated only
the first reading, which ties the K-choice to the same absolute bar that produced
the failing verdict — a mildly circular flavor. The pin survives either reading
because it never rested on "where is β's organic signal strongest": every masking
candidate *replaces* β's organic concentration with a forced selection, so the
relative-signal reading bears on the baseline's β, not on the intervention — the
pin rests on comparability and reference density, i.e. reasons (i), (iii), (iv),
which are untouched by the tension; (iii) K=32 is the F-geo-3 synthetic program's
own primary-anomaly/headline cell (`DELTANET_RD_EXACTNESS_DESIGN.md` §16.4), so the
synthetic-side interaction evidence this factorial mirrors is densest there; (iv) it
halves the run count vs the {16,32} grid. **K_sel=16 is the FIRST registered
follow-on axis** — upgraded from Rev 2's "not silently dropped" per the attack's own
recommendation, precisely because the relative-signal reading favors it.
Consequence, inherited from the sweep's own registered constants: every geo3-active
cell runs at `geo3_n_iter = 20` (`GEO3_N_ITER_BY_K_SEL = {16: 12, 32: 20}`,
`run_lm_rd_geo3_sweep.py:84` — the §1.1-registered escalation; a uniform n_iter=12
would replicate the known-non-admissible K=32 config).

**Reference quantities — what gets pinned at Wave −1, before any bar is evaluated
(all via the §4.3 `BANDS_PINNED-TrackB` writer):**

- **`anchor_random_32` and `anchor_collapse_32`:** the random-unit-vector and
  full-collapse Gram-deviation anchors **recomputed at (K=32, d=64)** by Monte-Carlo
  (CPU, free), cross-checked against the closed forms `E‖G−I‖_F ≈ √(K(K−1)/d)` and
  `√(K(K−1))` (`SCALE_TRANSFER_DESIGN.md` §6.8's own formulas) — planning values
  ≈3.94 and ≈31.50 respectively, **not citable until the MC recomputation lands**.
  Track C's K=64 anchors (7.94 / 63.50) and its whole-chunk Cell-1 band (21.93 ±
  5.90, §5.9) are **context only, banned from every bar** — Rev 1's use of them was
  the F1 FATAL (a Cell-4 reading of ≈9 at K_sel-scope would have read "close to
  random" against the K=64 anchor 7.94 while actually sitting ~2.3× above the true
  K=32 random floor).
- **`cell1_ref_32`:** Cell 1's same-instrument reference band — the §5.2 unified
  instrument run over the 6 archived Wave C checkpoints (the same checkpoints the
  original gate measured; forward-only, ≈0.1 GPU-h), reported as mean ± std over
  pooled (chunk, head) episodes per corpus.
- **`B_pinned`:** §2 principle 4's baseline per-chunk total write mass, from the same
  re-measurement pass.

**Bars (registered as rules now; numeric values substitute mechanically at Wave −1
readout, no re-negotiation):**

- **Headline bar for Cell 4:** same-instrument Gram deviation ≤ `0.5 × cell1_ref_32`
  **and** closer to `anchor_random_32` than to `cell1_ref_32` (i.e.
  `cell4 < (cell1_ref_32 + anchor_random_32) / 2`) — echoing i-strong's
  0.000-exactness reading convention, translated to a real-text-appropriate non-zero
  target given §4.2 item 3's own free-text degradation risk (forcing apart
  non-distinct real tokens is expected to be lossier than the synthetic harness's
  clean entity separation). The 0.5 multiplier is this design's own proposed number,
  explicitly labeled conservative-not-derived, in the same spirit
  `DELTANET_RD_EXACTNESS_DESIGN.md` §14.4 labels `resid_tol=1e-2`.
- **Interaction bar (the actual test, mirroring KEY_ANCHORING §2.0's
  discontinuous-jump framing):** `cell1 − cell4 ≥ 1.5 × max(cell1 − cell2,
  cell1 − cell3)`, every term the same instrument at K_sel=32 — Cell 4 must beat
  **both** single-ingredient cells by a margin at least 1.5× the better single
  ingredient's own gain over baseline, not merely beat Cell 1 alone. The 1.5×
  multiplier: proposed, conservative-not-derived, as above. **Language rule
  (REVISED, Rev 3 — NEW-3):** the wording of this bar's outcome follows §5.1's
  three-way range-non-overlap verdict — "β-informed selectivity" only on a
  disjoint-Cell-2-better read, "write concentration" on a disjoint-2R-equal-or-better
  read, and the neutral "selective writing (targeting-vs-concentration unresolved at
  n=3)" on the INCONCLUSIVE read.
- **Val-loss tolerance:** selectivity-only cells (2, 2R) held to a **wider,
  explicitly more lenient** +5% relative bar (this design's own proposal — hard
  write-sparsity is a bigger architectural constraint than geo3's
  orthogonalization-only intervention, so a tighter bar would conflate "selectivity
  itself costs something" with "the combination fails"); the combined cell (4) held
  to Track B's own original, tighter **+2% relative** bar (§4.6), since that is the
  claim this redesign ultimately wants to make comparably to Track B's original
  ambition. Both tolerances are relative to Cell 1's own val loss at matched corpus.

---

## 6. Budget

**Arithmetic, stated explicitly per the task brief (QUOTES CORRECTED, Rev 2 — M4;
SOURCE RECONCILED UPSTREAM, Rev 3 — NEW-8; every quoted string below is
grep-verified verbatim in its named source this session, post-fix).**
`SCALE_TRANSFER_DESIGN.md`'s own 300 GPU-h program ceiling (§7) is registered as
consumed as follows: rung-2's Rev 2.1 amendment registers, verbatim, "≈129 GPU-h for
the wave (cumulative ≈163 of the §7 300 GPU-h ceiling)" (§5.6 item 1); the
finer-grained figures — 129.4 GPU-h, cumulative 162.5/300 — appear in **`STATE.md`'s
rung-2 launch entry** (the bullet headed "SCALE-TRANSFER Track C Wave 2 (rung-2,
392M) LAUNCHED"), *not* in `SCALE_TRANSFER_DESIGN.md`, and Rev 1 of this document
wrongly presented them inside quotation marks attributed to the design doc
(acknowledged in §12). Rung-3's Rev 2.2 amendment, token-matched to rung-2 at 1.5B
tokens/run, prices at ≈76.2 GPU-h for its 2-run wave. **Attack round 2 (NEW-8)
caught an internal ~26 GPU-h inconsistency in that amendment's own paragraph** (its
rejected-3B option's "≈316" cumulative implied a 163.5 base while its accepted
option's ≈266 implied a 189.8 base — the stale figure predated the `--wave 1ext`
≈27 GPU-h addition to the committed base); **the source has since been corrected
upstream** (commit `13eb71c`), and now reads, verbatim: "cumulative ≈343 on the
190.22 GPU-h committed base" for the rejected option and "cumulative ≈266/300 —
passes every gate" for the accepted one — mutually consistent on the same 190.22
base (190.22 + 152.5 ≈ 343; 190.22 + 76.2 ≈ 266.4). **Nominal remaining headroom
after rung-3: 300 − 266.4 ≈ 33.6 GPU-h.**

**This is a projection, not a harvested actual — flagged, per the task's own
instruction.** Both the rung-2 129.4 GPU-h figure and the rung-3 76.2 GPU-h figure were
**registered before their respective waves' real completion data existed** (§5.6's own
text: "registered BEFORE any rung-3 training data exists"). At the time the documents
this redesign is grounded in were read, `EXPERIMENT_LOG.md` contains **no** rung-2 or
rung-3 *harvest* entry with realized (not projected) costs — a direct grep for
"rung-3"/"rung 3" over `EXPERIMENT_LOG.md` returned nothing. **Before this redesign
launches anything, the current actual cumulative GPU-h spend across the whole
SCALE-TRANSFER program must be re-read from the latest `STATE.md`/`EXPERIMENT_LOG.md`
entries** — rung-2's harvest and rung-3's real run may have already landed with numbers
that differ from the ≈266.4/300 figure cited here, in either direction (calibration
surprises have already moved this program's own numbers once, per `SCALE_TRANSFER_DESIGN.md`
§10 item 2's own warning that Track C is "the single largest and least-validated cost
driver").

### 6.1 This redesign's own budget, sized to fit under the ≤25 GPU-h target (RE-PRICED, Rev 2 — M2, M3, N3)

**Two pricing constants, both corrected/registered (Rev 2):** non-geo3 runs at Wave
C's measured ≈0.077 GPU-h/run (≈0.0000126 GPU-h/step = 0.077 / 6,103 — Rev 1 printed
"≈0.008 GPU-h/probe" for a 2,000-step probe against these same inputs, a 3× arithmetic
error; the correct product is 2,000 × 0.0000126 ≈ **0.025 GPU-h/probe**). **Every
geo3-active run is priced at the codebase's own registered 3× planning placeholder**
(`run_lm_rd_geo3_sweep.py:94–104`: `_PER_STEP_S_PLACEHOLDER_GEO3 = 0.15` s/step vs
the ~0.05 s/step non-geo3 baseline, "NOT yet measured on-box for the geo3-active
path") — per full 6,103-step run: 0.15 × 6,103 + 7 ckpts × 15 s ≈ 1,020 s ≈
**0.28 GPU-h/run**, superseded by Wave −1's own measured timing before Wave 2
launches, per the calibration-first discipline the placeholder's own comment cites.

| Wave | Purpose | Scope | Est. GPU-h | Gate |
|---|---|---|---|---|
| **−1 (manipulation check + pins)** | (a) Short training probes to get past-init β/score behavior for all 4 candidates (none has a trained checkpoint, unlike the original gate) + a baseline reference pilot for the §4.3 churn derivation; (b) the M8 geo3-LM stability smoke on the tabular-risk slice (§5.1); (c) Cell-1 same-instrument re-measurement on the 6 archived Wave C checkpoints + MC anchor recomputation at K_sel=32 (§5.3); writes `BANDS_PINNED-TrackB.json` | (a) 5 probes × 2,000 steps × 0.025 ≈ 0.13; (b) 2,000 geo3-active steps at the 3× placeholder ≈ 0.10; (c) forward-only ≈ 0.10 + CPU | **~0.3–0.5** | §4.2's per-candidate checks; stability smoke bars (§5.1) gate Cells 3/4; `BANDS_PINNED-TrackB` written or Wave 1 refuses |
| **1 (selectivity-only, non-geo3)** | Full Wave-C-scale training, `geo3_active=False`: surviving candidates + the M7 STE comparator + the F2 Cell-2R budget-matched random control | (2–4 candidates × 2 corpora × 3 seeds = 12–24) + comparator (6) + Cell 2R (6) = 24–36 runs × 0.077 | **~1.8–2.8** | Val loss within §5.3's +5% bar; churn/support against the §4.3 pinned bars; ranks the top 1–2 candidates for Wave 2 |
| **2 (2×2 factorial, geo3-active cells, top 1–2 candidates)** | Cell 3 (override-stamped reference arm, run once, shared) + Cell 4 per surviving candidate, all at `K_sel=32`, `geo3_n_iter=20` | Cell 3 (2 corpora × 3 seeds = 6) + Cell 4 (6–12) = 12–18 runs × 0.28 (3× placeholder) | **~3.4–5.0** | §5.3's interaction bar; val loss within the tighter +2% bar for Cell 4; M8 stability gate cleared first |
| **3 (geometry/drift/rank instrumentation, all cells)** | §4.4's key-Gram/drift instrumentation; forward-hook probes, no backward pass, mirroring Track C/D's own ~0.08–0.9 GPU-h probe costs | all cells from waves 1–2 | **~1–2** | Descriptive, Tier 2 throughout |
| **Total** | | | **≈6.5–10.3 GPU-h (rows sum exactly: 0.3+1.8+3.4+1.0 = 6.5 low; 0.5+2.8+5.0+2.0 = 10.3 high; central ≈8)** | |

**Wide-case (2× multiplier, this project's own standing calibration-uncertainty
convention, e.g. `SCALE_TRANSFER_DESIGN.md`'s own "±2×" pre-calibration framing,
§5.6): ≈13–20.6 GPU-h.** Adding the reserve Cell 4R (§5.1: 6 geo3-active runs ≈ 1.7
GPU-h, run only if Cell 4 clears its bar) brings the absolute worst case to ≈22.3
GPU-h — **under the 25 GPU-h target, and under the corrected ≈33.6 GPU-h nominal
headroom with ≈11 GPU-h to spare at the worst case (Rev 3 — NEW-8). Margin stated
honestly:** at a 10% rung-3 cost overrun (266.4 + 7.6 ≈ 274), the post-Track-B slack
shrinks to ≈4 GPU-h at this worst case — thin, which is why the §11-item-5 rule
stands: re-derive the actual cumulative from real harvest entries (not either
projected figure) immediately before Wave 1 launches. The arithmetic above exists to
make the margin explicit and auditable; Rev 1's version of this section failed that
stated purpose twice (M2's 3× per-probe error, M3's omission of the geo3 pricing
placeholder its own reading list's file registers), and Rev 2's inherited a
since-fixed inconsistency in the source itself (NEW-8) — all corrected, with Wave
−1's measured timing registered as the authority that supersedes every placeholder
before Wave 2 spends.

---

## 7. Falsification map

| Claim | Falsified by |
|---|---|
| "Hard selectivity is achievable on this harness without destroying trainability" | Both candidate 1 AND its M7 soft-top-K comparator fail to converge to a reasonable floor (divergence/NaN-collapse under standard skip-step handling) — the comparator's presence makes this claim's negative attributable to selectivity itself, not to STE (Rev 2 — M7); candidate 1 failing *alone* falsifies only "hard-masked-with-STE is trainable" |
| "The gate metrics, once forced, are a meaningful manipulation check" | For candidates 1/3: nothing — §4.1 already establishes these are construction-trivial; the manipulation check for these candidates is training-viability plus the §4.3 pinned churn bar, and no result on the gate metric itself can falsify or confirm anything |
| "Candidate 1's selection is functional, not degenerate" | Full-run churn exceeds the §4.3 pinned ceiling (selection-set instability), or the §3.1(iii) positional-artifact check fires — either flags the cell selection-degenerate, excluded from Cell-4 inheritance (Rev 2 — M1) |
| "Selectivity alone measurably improves memory-fidelity metrics" | Cell 2 does not clear Cell 1 by a margin distinguishable from seed noise — consistent with, not contradicting, KEY_ANCHORING's own "stability/selectivity alone buys little" finding (§2). **Attribution split (Rev 2 — F2; rule revised Rev 3 — NEW-3):** §5.1's three-way range-non-overlap verdict decides the language — disjoint-2R-equal-or-better downgrades to "write concentration"; overlap = INCONCLUSIVE, no downgrade and no targeting claim |
| "Orthogonality + selectivity interact super-additively on free text, mirroring the synthetic-harness 2×2" | Cell 4 fails §5.3's interaction bar (`cell1−cell4 < 1.5×max(cell1−cell2, cell1−cell3)`, all terms same-instrument at K_sel=32) — the additive, non-interacting outcome is equally reportable and would mean the synthetic-harness interaction does not transfer to free-running real text at this scale |
| "This redesign licenses a claim about pretrained/production delta-rule LMs" | Nothing in this design can establish this regardless of outcome — Track D's own Phase 1 non-attribution finding (§9 of this document) already forecloses it a priori for any from-scratch small-model result |

---

## 8. The Rev-1 "pre-answered" attack predictions — dispositions after the real attack round (Rev 2)

Rev 1 pre-answered five predicted attack findings. The real attack round
(`TRACKB_REDESIGN_ATTACK_R1.md`) validated the *predictions* but found three of the
five *pre-answers* insufficient — in each case the pre-answer disclosed a risk
without controlling or gating it, and the attack correctly held that **disclosure is
not a plan**. Recorded per house transparency norms rather than silently rewritten:

1. **"The manipulation check is trivially satisfied for candidates 1/3."** Predicted
   and pre-answered adequately (§4.1's reframing stands); the attack extended it with
   M1 — the reframed check itself had no numeric gate. Disposition: closed by §4.3's
   pinned bars.
2. **"STE's hard top-K mask will churn."** Predicted; pre-answer insufficient
   (registered as a diagnostic with no pass/fail rule — attack M1's exact point).
   Disposition: closed by §4.3's pinned churn ceiling, derived from the baseline
   reference pilot via the `mean + 2·s` house formula, launcher-gated.
3. **"Forcing K_sel writes is also a write-BUDGET change."** Predicted; pre-answer
   insufficient (a reporting requirement is not a control — attack F2's exact point).
   Disposition: closed by §2 principle 4's mandatory `B_pinned` renormalization plus
   §5.1's Cell 2R budget-matched random control with its registered
   language-downgrade decision rule.
4. **"Will a write-up quietly blur the different-model-tier boundary?"** Predicted and
   pre-answered adequately — the attack round checked §9 explicitly ("the one attack
   surface that came back clean") and found no violation. §9 is frozen from Rev 1.
5. **"The top-K_sel-mass metric assumes a fixed K_sel that entmax/sparsemax does not
   respect."** Predicted; pre-answer insufficient (support size was required
   *reporting*, with no floor — attack M1 again). Disposition: closed by §4.3's
   pinned support band (a-priori ceiling K_sel, Wave-−1-pinned floor).

**The sixth, process-level lesson carried forward rather than re-earned:**
`DELTANET_RD_EXACTNESS_DESIGN.md` §16.7's dated correction found that a predictive
"launch-read" simulator tool wired a *single* K's measured drift scalar into *both* K=16
and K=32's predictions, silently mis-attributing the K=32 result. If this redesign ever
builds an analogous predictive/manipulation-check tool that takes a per-candidate or
per-K_sel measured statistic as input, that statistic must be threaded per-cell, never
shared across cells via one API call — a concrete, checkable build-time item, not a
vague "be careful" reminder. (With K_sel pinned to the single value 32 (§5.3), the
per-K risk surface shrinks but the per-candidate/per-cell version of the same bug
remains live — `BANDS_PINNED-TrackB`'s per-cell fields must be read per-cell.)

---

## 9. Claim tiers

Per `SCALE_TRANSFER_DESIGN.md` §2's house tier system (Tier 1 premise-conditional
causal / Tier 2 descriptive+interventional / Tier 3 measurement-only), inherited
verbatim, extended per-candidate:

- **Candidates 1, 2, 4 — trained from scratch (this design's own default, §2 principle
  2): Tier 2 only.** "This intervention, on this from-scratch-trained small LM, at this
  scale/window/selection definition, changed this measured quantity by this amount, on
  this corpus" — Track B's own §4.7 language, inherited exactly, since these candidates
  are architecturally the *same model class* (`DeltaNetLMMixer`) as Wave C, merely
  retrained with a different β mechanism. **None of these may claim** "we fixed Wave C"
  or "pretrained/production LMs of this family need hard selectivity" — a from-scratch
  run is a different trained instance, not a repair of an existing one, regardless of
  architectural continuity.
- **If a future follow-on instead continue-trains (retrofits) any of candidates 1/2/4
  onto the *existing* Wave C checkpoint** (explicitly NOT this design's default, §2
  principle 2, §9 of this section names it as a possible but non-default follow-on): it
  inherits the Track-D-style bolt-on risk in full, and can claim **at most Tier 2 for
  that one specific checkpoint** — never a general claim, mirroring
  `SCALE_TRANSFER_DESIGN.md` §2's own precedent for Track D's Phase 2 ("would, if it ever
  runs, earn at most Tier 2 for one specific model, never a general claim about the
  delta-rule family").
- **Candidate 3 (fixed periodic write schedule) — different-model tier, explicitly.**
  This is not a different trained instance of the same architecture; it is a different
  architecture (a distinct tokenization/preprocessing pipeline, a non-learned
  write-decision mechanism structurally closer to `model_rd.py`'s synthetic-harness
  convention than to `lm_pretrain_rd.py`'s LM convention). Its scientific value is
  narrower and more purely mechanistic — "does orthogonality+stability jointly help when
  a hard, non-learned write-schedule exists on free text" — and it **must not** be cited
  as evidence about what continuous-β LMs of the Wave-C/`DeltaNetLMMixer` family do or
  need, let alone pretrained ones.
- **Restated, per the task brief's own explicit instruction, against Track D's
  non-attribution finding (`STATE.md`, `SCALE_TRANSFER_DESIGN.md` §6.8):** even if this
  redesign's 2×2 factorial reproduces KEY_ANCHORING's clean discontinuous-interaction
  shape (§2 principle 1) on this project's own from-scratch small models, that result
  licenses **zero** claims about production/pretrained fixed-state LMs. Track D's own
  Phase 1 measurement already found that the write-geometry-attractor signature at
  production scale (RWKV-7 1.5B, Falcon-Mamba-7B) is **not attributable to the
  delta-rule write mechanism specifically** — the registered no-fixed-state negative
  control (Qwen2.5-1.5B, standard softmax attention, no recurrent state of any kind)
  showed the **same magnitude** signature, meaning the effect is dominated by generic
  trained-LM key anisotropy (massive activations, Sun et al. 2024), present regardless
  of architecture. A clean Track-B-family finding, however clean, stays confined to Tier
  2 language about this project's own from-scratch harness; any temptation to generalize
  toward "therefore pretrained delta-rule LMs need hard-selective writes too" is
  explicitly blocked by this already-measured result, cited by name, not merely gestured
  at.

---

## 10. Cut order

**Never cut:** the Wave −1 manipulation check for candidate 1 (cheapest, most direct
answer to Track B's own gate finding, and the necessary precondition-satisfaction check
before anything else in this redesign is worth running); the Wave −1 geo3-LM stability
smoke (Rev 2 — M8: it gates Cells 3/4 and costs ≈0.1 GPU-h); **Cell 2R** (Rev 2 — F2,
orchestrator-pinned as required: it carries the factorial's budget-vs-targeting
attribution, exactly the never-cut role `SCALE_TRANSFER_DESIGN.md` §8 assigns its own
MAJOR-5 mix-control cell); **Cell 3, at minimum one corpus × 3 seeds (PROMOTED, Rev
3 — attack round 2's adjudication of §11 item 3, adopted):** the §5.3 interaction
bar's `cell1 − cell3` term is not computable at all without a real Cell 3 — the
archived gate JSON measured a different quantity (β-shape on a non-orthogonalized
checkpoint, forward-only) and cannot substitute — and Cell 3 doubles as the first
full-scale geo3-active LM training run the M8 stability gate exists to de-risk.

**Cut, in this order, until back under whatever budget line is live at launch time:**

1. The fixed-mask 2R diagnostic variant (§5.1 — NEW-2's named diagnostic, never the
   control) — pure bonus information about positional structure; first to go.
2. Cell 4R (the reserve geo3-side targeting control, §5.1) — already conditional on
   Cell 4 clearing its bar; cutting it costs nothing committed.
3. Candidate 2 (entmax/sparsemax) — highest build risk (new autograd primitive, the
   §3.2 chunk-sum-budget structural change, the only candidate whose budget match can
   be PARTIAL by construction); cut first among candidates if squeezed.
4. Candidate 4's soft phase — already predicted, on this project's own F-geo-1
   precedent, to underperform (§3.4); its only unique informative content (does soft
   warm-up ease the subsequent hard phase) is a secondary readout, not a headline one
   (and NEW-6's re-pinned comparator no longer overlaps it — the comparator hardens
   at 10% of steps precisely so it is NOT a curriculum arm).
5. Candidate 3 — highest scope change (new preprocessing pipeline); cut before
   candidate 1's own manifest is trimmed, since candidate 1 is cheaper, more direct, and
   answers the queued idea's own literal framing ("hard-zero-β") most closely. If it
   survives to Wave 2, its period `W` gets a cheap 2–3-point sensitivity check, not a
   full attack pass (attack round 2's adjudication of §11 item 4, adopted).
6. The M7 STE comparator — cutting it **automatically re-triggers M7's alternative
   branch** (registered here, not renegotiated later): every candidate-1 negative
   becomes uninterpretable w.r.t. STE-vs-architecture, candidate-1 claims stay
   permanently scoped to "hard-masked-with-STE," and candidate 2 is promoted to
   co-primary — the comparator is cheap (≈0.5 GPU-h) precisely so this branch never
   has to fire on budget grounds.
7. The shared Cell-3 (geo3-only) reference arm's **second corpus** — drop to openr1
   only if squeezed (the cell itself is never-cut, above); the interaction bar (§5.3)
   is directionally assessable from one corpus, though the full claim needs both,
   mirroring `SCALE_TRANSFER_DESIGN.md` §8 item 5's own precedent for a symmetric cut.
8. Candidate 1's own naive-window-style ablation sensitivity check (if one is added
   later) before candidate 1's primary beta-topk arm itself — the primary arm carries
   the entire causal-adjacent claim and is never cut.

---

## 11. Open questions — status after attack round 2's adjudications (Rev 3)

Round 2 explicitly adjudicated every Rev-2 open question (`TRACKB_REDESIGN_ATTACK_R2.md`
Part 3 item 7); dispositions recorded here, remaining opens marked:

1. **(RESOLVED — adjudicated NO.)** The 0.5×/1.5× multipliers stay
   proposed-not-derived: a simulator-derivation tool would duplicate the §16.7
   shared-scalar hazard for marginal calibration benefit at this Tier-2 budget scale.
   Registered consequence: bar outcomes are treated as provisional until replicated.
2. **(STILL OPEN, sharpened.)** The hard-concrete/L0 third mechanism stays a
   conditional follow-on — but per the adjudication, NEW-6 weakened the comparator's
   attribution power (even at the Rev-3 10% pin, an early-window difference
   remains), so the promotion trigger is widened: it fires if candidate 1 and the
   comparator disagree, OR if the comparator's verdict is contested at readout on
   early-window grounds.
3. **(RESOLVED — adjudicated YES, and stronger than Rev 2's framing.)** Cell 3 is
   worth the ≈1.7 GPU-h and is now **never-cut at one corpus** (§10): the gate JSON
   measured a different quantity and cannot supply the interaction bar's
   `cell1 − cell3` term nor a protocol-matched val loss, and Cell 3 doubles as the
   M8 stability gate's full-scale de-risk target.
4. **(RESOLVED — adjudicated NO, disproportionate.)** Candidate 3's period `W` gets
   a cheap 2–3-point sensitivity check, gated on the candidate surviving to Wave 2
   (§10 item 5) — not a standalone attack pass.
5. **(PARTIALLY RESOLVED — the source-inconsistency half is fixed upstream, NEW-8/
   commit `13eb71c`; the harvest half stands.)** The corrected headroom is ≈33.6
   GPU-h, but it remains a projection: no rung-2/rung-3 *harvest* entry exists in
   `EXPERIMENT_LOG.md` (re-confirmed by round 2). The standing rule: re-derive the
   actual cumulative from real harvest entries — never from either projected
   figure — immediately before Wave 1 launches. With worst-case ≈22.3 GPU-h against
   ≈33.6 (or ≈26 under a 10% rung-3 overrun), this re-check is load-bearing.
6. **(RESOLVED — adjudicated CONCUR.)** Per-write magnitude is inseparable from
   selectivity at fixed `B_pinned` and fixed `K_sel` (a mass-conservation identity,
   not a control gap); Cell 2R already isolates the orthogonal targeted-vs-random
   question. No further control cell.
7. **(NEW, Rev 3 — for the build phase.)** The §4.3 positional-concentration bar
   uses TV-from-uniform against an unmasked-pilot null; its registered limitation
   (cannot distinguish content-driven-but-positionally-biased from purely
   positional) routes flagged runs to a document-dependence follow-up analysis. Is
   that follow-up's method (selected-set overlap across documents at matched
   offsets) worth pre-specifying to a numeric rule too, or is analysis-tier freedom
   acceptable for a non-gating diagnostic? Rev 3 takes the latter position.

---

## 12. Rev 2 finding→change map (attack round 1, 2026-07-04)

Independent adversarial review of Rev 1 (`TRACKB_REDESIGN_ATTACK_R1.md`, against
commit `52d834f`): **NEEDS-REV-2** — 2 FATAL, 8 MAJOR, 3 MINOR. Every finding
addressed per the orchestrator's binding decisions. (The orchestrator's dispatch used
a shifted numbering — its F1/F2 match the attack's; its M3→M10 map to the attack's
M1→M8; this table uses the attack document's own IDs.)

**Citation-integrity acknowledgment (M4), stated plainly per house transparency
norms:** Rev 1 presented "cumulative ≈162.5/300" inside quotation marks attributed to
`SCALE_TRANSFER_DESIGN.md` §5.6, where that string does not exist (verified by the
attacker's full-file grep and re-verified this session), and stated "129.4" with
false precision against the source's own "≈129." The numbers themselves exist — in
`STATE.md`'s rung-2 launch entry — but attributing them, quoted, to a document that
does not contain them is a **fabricated quote**, exactly the failure `CLAUDE.md`'s
"verify before claiming" rule exists to catch, and it is a serious discipline
violation regardless of the numbers being real somewhere. Rev 2's rule, applied
throughout: every quoted string is literally grep-verifiable in its cited source, and
paraphrases are never placed inside quotation marks.

| Finding | Severity | What was wrong | Fix | Where |
|---|---|---|---|---|
| F1 | FATAL | §5.3's headline bar compared Track C's whole-chunk K=64 instrument (21.93 / 7.94 / 63.50) against Cells 2–4's K_sel-scoped §4.4 instrument — never shown comparable; anchors never recomputed at K_sel | ONE instrument for every cell (orchestrator-pinned): the §4.4 selected-key instrument at K_sel, applied read-only to Cell 1 too; random/collapse anchors recomputed by MC at (K=32, d=64) at Wave −1, cross-checked vs §6.8's closed forms; K=64 numbers banned from bars (context only); all references pinned via `BANDS_PINNED-TrackB` before any bar is evaluated | §5.2, §5.3, §6.1 Wave −1 row |
| F2 | FATAL | The self-flagged write-budget confound was disclosed but never controlled — a reporting requirement is not a control; Cells 2/4 would have carried systematically less write mass than Cells 1/3 across the exact axis the interaction bar computes | Mandatory `B_pinned` renormalization for ALL masking candidates (orchestrator-pinned, registered rule with ≤1.0 clamp + logged shortfall; candidate 2's structural PARTIAL-match case registered); REQUIRED budget-matched random-selection control Cell 2R with a registered language-downgrade decision rule; reserve Cell 4R named | §2 principle 4, §3.2, §5.1, §5.3, §7, §10 |
| M1 | MAJOR | Churn-rate and support-size had no pass/fail bars — the degenerate-model case ("selective by construction, useless in function") could pass every registered gate | New §4.3: metric definitions + derivation rules registered now (churn ceiling = `mean_ref + 2·s_ref` over the baseline reference pilot's implicit top-K_sel churn; support band = [Wave-−1-pinned p10 floor ≥4, a-priori ceiling K_sel]); numbers pinned at Wave −1 readout via the KEY_ANCHORING Rev 5 §3.6 `BANDS_PINNED` pattern (writer/launcher-gate/readout-assertion, all three parts) | §4.1–§4.3, §7 |
| M2 | MAJOR | Wave −1 per-probe cost internally inconsistent by ~3× (stated ≈0.008 GPU-h/probe; the row's own inputs give 2,000 × 0.0000126 ≈ 0.025) | Corrected to ≈0.025 GPU-h/probe; the arithmetic error is named in §6.1 rather than silently fixed | §6.1 |
| M3 | MAJOR | Every geo3-active wave priced at the non-geo3 0.077 GPU-h/run rate, ignoring `run_lm_rd_geo3_sweep.py:94–104`'s own registered 3× planning placeholder (`_PER_STEP_S_PLACEHOLDER_GEO3 = 0.15`) | All geo3-active runs re-priced at ≈0.28 GPU-h/run (orchestrator-pinned: placeholder governs until Wave −1 measures); Wave 2 → ≈3.4–5.0; the placeholder's supersession-by-measurement is registered | §6.1, §11 item 3 |
| M4 | MAJOR | Fabricated quote + false precision (see the acknowledgment block above) | Quote removed (orchestrator-pinned); §6 now quotes only grep-verified strings ("≈129 GPU-h for the wave (cumulative ≈163 of the §7 300 GPU-h ceiling)"; "cumulative ≈266/300 — passes every gate") with the 129.4/162.5 figures correctly attributed to `STATE.md`, unquoted | §6, §12 preamble |
| M5 | MAJOR | The Cell-3 override targeted `selection_mode_for_verdict` (`:165–169`) — unreachable dead-code defense whose own comment says the CALLER must refuse first; the real refusal is `_refuse_if_no_launch` (`:172–183`, `sys.exit(3)`, called in `main()` at `:600`) | Override re-targeted at `_refuse_if_no_launch` (orchestrator-pinned) with full mechanics: a new explicit flag, config cross-validation (`:144–161`) stays live, and per-run assembly-time stamping (`gate_override`/`gate_override_reason`/`gate_override_at`/`claim_tier: "descriptive"`, plus `gate_override: false` on non-override runs) mirroring KEY_ANCHORING Rev 5 §3.6 item 2/R4-1 | §0, §5.1 |
| M6 | MAJOR | Cell 4's composition rule (`hard_select_k_sel` vs `geo3_k_sel`) unspecified; both resolutions had unanalyzed pathologies (redundant re-selection; orthogonalizing inert zero-β positions) | Pinned (orchestrator decision): `hard_select_k_sel == geo3_k_sel` required, and geo3's own selection is REPLACED by the mask's (forced-selection argument; single selection source) — justification paragraph addresses both of the attack's cases | §3.1, §5.1 |
| M7 | MAJOR | A moderate candidate-1 val-loss regression was consistent with both "STE gradient bias" and "hard selectivity costs" — no registered cell separated them | Temperature-annealed soft-top-K comparator cell added (orchestrator's option A, chosen over uninterpretability+co-primary because it costs ≈0.5 GPU-h against ample headroom; the alternative branch is registered as the automatic consequence if the comparator is ever cut, §10 item 5); interim "hard-masked-with-STE" claim-labeling until it reads out | §3.1, §6.1, §7, §10, §11 item 2 |
| M8 | MAJOR | Cell 3 = the first-ever multi-thousand-step LM-mode geo3 training, with a code-documented NaN risk on repetitive text (`lm_pretrain_rd.py:349–353`) — Rev 1 discussed Cell 3 only as budget/bureaucracy | Wave −1 geo3-LM stability smoke added (orchestrator-pinned): 2,000-step `geo3_active=True` run on the top-decile-repetition openr1 slice; bars = zero non-finite losses AND skip-rate ≤1%; breach = registered stability finding that gates Cells 3/4 behind an audited fix wave | §5.1, §6.1, §10 |
| N1 | MINOR | K_sel never pinned for the Wave 1/2×2 manifest; §6.1's run counts carried no K_sel factor | `K_sel = 32` pinned, single value, four stated reasons; `geo3_n_iter=20` consequence registered; K_sel=16 a named follow-on | §5.3 |
| N2 | MINOR | §3.1 called the β-mask site "the exact site" of the existing geo3 call; the geo3 call actually sits after the q/k/v reshape (`:539–551`), on the reshaped tensor | Corrected: shared *selection logic*, different *call site*; β's shape is reshape-unaffected, so the mask sits at `:529–531` upstream of both | §3.1 |
| N3 | MINOR | §6.1's stated total (≈3.5–5.5) did not sum its own rows (3.35–5.35), before the M2/M3 corrections even applied | Re-totaled with the corrected pricing and new cells; the total row now states its own row-sum arithmetic explicitly (6.5 / 10.3) | §6.1 |

Frozen per the orchestrator: **§9 (claim tiers)** — the attack round checked it
explicitly for quiet pretrained-LM claim leakage and found none; not a word of it is
changed in Rev 2.

---

## 13. Rev 3 finding→change map (attack round 2, 2026-07-04)

Independent adversarial review of Rev 2 (`TRACKB_REDESIGN_ATTACK_R2.md`, fresh agent,
against commit `f814b99`): **NEEDS-REV-3** — round 1's F1 verified CLOSED (MC anchor
math independently reproduced), 8 new MAJOR + 2 MINOR, all in Rev 2's own additions.
Every finding addressed per the orchestrator's binding decisions. Items round 2
marked CLOSED (R1's M2/M3/M4/M5/N2/N3, the override stamping, the BANDS_PINNED
mechanics) are **frozen** — nothing in Rev 3 touches them.

| Finding | Severity | What was wrong (in Rev 2's own additions) | Fix | Where |
|---|---|---|---|---|
| NEW-1 | MAJOR | Clamp-shortfall handling was asymmetric: candidate 2 got a numeric PARTIAL rule, candidates 1/3/4 got logging only — "reporting is not a control" reintroduced for a narrower cell set; the attack's own arithmetic (mean scale 1.76×, ~16 pp clamp headroom on an average chunk) shows the gap is live, and the gate JSON's pooled means cannot bound it | ONE symmetric rule, every masking cell (orchestrator-pinned): BUDGET-PARTIAL if `median(shortfall_c) > 0.10` or `frac(shortfall_c > 0.10) > 0.25`; per-chunk dispersion measured at Wave −1 (same forward passes, zero GPU) and pinned into `BANDS_PINNED-TrackB` | §2 principle 4 |
| NEW-2 | MAJOR | Cell 2R's randomization cadence was unspecified — a fixed offset draw would degenerate the control into a positional schedule, corrupting the exact comparison it exists for | Cadence pinned (orchestrator decision): resampled per (chunk instance, training step), RNG keyed to run seed + step, never a fixed function of intra-chunk position; a fixed-mask variant registered as a separate cut-eligible DIAGNOSTIC, never the control | §5.1, §10 item 1 |
| NEW-3 | MAJOR | "Matches within seed noise" had no operationalization; the n=3 headcount borrowed KEY_ANCHORING's convention without its (one-sample) power analysis, for a two-sample comparison needing more | Decision rule re-registered (orchestrator decision): per-corpus seed-range non-overlap, three-way verdict incl. an explicit INCONCLUSIVE (no downgrade, no claim, neutral registered phrasing); low power acknowledged numerically (range separation has α ≈ 0.10/corpus under exchangeability; a crude screen, not a powered test) | §5.1, §5.3, §7 |
| NEW-4 | MAJOR | The churn ceiling's "non-circular" defense proved non-triviality, not calibration — an unmasked pilot cannot see mask-specific dynamics (STE entrenchment could suppress churn; legitimate sharpening could raise it) | Ceiling = max of TWO nulls (orchestrator decision): the unmasked-pilot null (kept) and a same-model null (the gated run's own first-10%-of-steps churn, mean+2s); one-sentence calibration limitation registered — a coarse degeneracy screen, breaches route to diagnosis | §4.3 |
| NEW-5 | MAJOR | The positional-artifact check had no spec at all behind a broken citation ("§8 item 1" points to an unrelated passage) — and it guards the MORE dangerous degenerate case (frozen collapse = near-zero churn, invisible to the churn bar) | Fully specified (orchestrator decision): TV distance of the intra-chunk offset-selection marginal from uniform, fixed probe batch, 50%/final checkpoints, ceiling = unmasked-pilot `mean+2s`, folded into `BANDS_PINNED-TrackB`; flag = positionally degenerate, excluded from Cell-4 inheritance; the TV statistic's own limitation registered; citation fixed | §3.1(iii), §4.3 |
| NEW-6 | MAJOR | The comparator's 80%-soft anneal was candidate 4's soft-warmup curriculum in disguise — its success could not distinguish "STE was the problem" from "the curriculum was the fix" | Anneal re-pinned (orchestrator decision): τ→0 by 10% of steps, both arms hard for ≥90% of training, isolating the gradient estimator; residual early-window confound registered in one sentence; bit-equivalence kept but explicitly non-load-bearing, with a numeric fallback bound (max-abs diff ≤1e-6) registered should any future check depend on it | §3.1, §10 item 4, §11 item 2 |
| NEW-7 | MAJOR | The NaN-smoke slice was selected by a text-level proxy (window 4-gram repetition) that doesn't guarantee duplicated keys land among β-SELECTED positions — the smoke could pass without exercising the failure regime | Slice re-specified from the code's own mechanism (orchestrator decision): rank by within-chunk identical-4-gram count (`n_dup_max ≥ 8`, layer-0 conv-context duplication ⇒ identical key rows); positive-control floor (≥25 forward calls with ≥6 duplicated SELECTED rows, logged via `_topk_idx`) or the smoke is NON-PROBATIVE; registered fallback forces selection onto duplicated positions via the M6 argument | §5.1 |
| NEW-8 | MAJOR | The budget source's own Rev 2.2 paragraph carried a ~26 GPU-h internal inconsistency (≈316 on a stale 163.5 base vs ≈266 on the 190.22 base); Rev 2 quoted it accurately but did not reconcile | Fixed UPSTREAM by the orchestrator (`SCALE_TRANSFER_DESIGN.md`, commit `13eb71c`: 316→343 on the 190.22 base, correction note names this attack); §6 re-quotes the corrected text; headroom = 300 − 266.4 ≈ 33.6 GPU-h; worst-case margin stated honestly incl. the 10%-overrun scenario (≈4 GPU-h slack) | §6, §6.1, §11 item 5 |
| MINOR-1 | MINOR | K_sel=32 reason (ii) used only the absolute-proximity reading; the relative-excess-over-uniform reading inverts it (K=16: +23.7% rel. vs K=32: +13.8% rel.) — a mildly circular tie to the bar that produced the failing verdict | Both readings stated; why the pin survives either (it rests on comparability/reference density — reasons i/iii/iv — not on where β's organic signal is strongest, which the intervention replaces anyway); K_sel=16 upgraded to the FIRST registered follow-on axis | §5.3 |
| MINOR-2 | MINOR | The forced-selection argument bypasses the internal `topk` that the EOT-validity mask (`valid_sel`, `:301`) is derived from — validity threading unspecified | Registered: validity under forcing = `content_mask` gathered at forced indices, feeding the existing degenerate-episode machinery unchanged; build smoke registered (forced EOT index → invalid → no-op scatter) | §5.1 |

Adjudications of §11's open questions (Part 3 item 7 of the attack) are adopted and
recorded in §11's rewritten status list; the two that changed the manifest are Cell
3's promotion to never-cut (§10) and candidate 3's period getting a sensitivity
check instead of an attack pass (§10 item 5).

---

## Reproducibility pointers

This design: `matrix-thinking/TRACKB_REDESIGN.md` (**Rev 3**, this file; Rev 1 at
commit `52d834f`, Rev 2 at `f814b99`). Attack rounds:
`matrix-thinking/TRACKB_REDESIGN_ATTACK_R1.md` and
`matrix-thinking/TRACKB_REDESIGN_ATTACK_R2.md` (both read in full). Grounding
sources, all read in full or in the cited section this session:
`SCALE_TRANSFER_DESIGN.md` §4 (+§2, §6.4, §6.8, §5.6 incl. the Rev 2.1/2.2
amendments — the latter as corrected at commit `13eb71c`, §5.9, §7);
`EXPERIMENT_LOG.md`'s "SCALE-TRANSFER Track B" entry and its two `[LEARN]` blocks;
`matrix-thinking/deltanet_rd/results/lm_rd_geo3/wave_neg1_gate.json` (raw numbers,
read directly, incl. `gate_k_sel: 32`); `DELTANET_RD_EXACTNESS_DESIGN.md` §14, §16
(incl. §16.7's dated correction); `KEY_ANCHORING_DESIGN.md` Rev 5 §1–§2.2, §3.6
(the `BANDS_PINNED` writer/launcher-gate/readout-assertion machinery, read in full),
§8.3/R4-1; `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` (`:1–60`, `:206–398`,
`:419–570`, incl. the `:349–353` NaN-risk comment and the `:301` `valid_sel`
derivation); `matrix-thinking/deltanet_rd/run_lm_rd_geo3_sweep.py` (`:61–74`
constants, `:84` `GEO3_N_ITER_BY_K_SEL`, `:94–104` the 3× placeholder, `:131–183`
gate loading + `_refuse_if_no_launch`, `:598–600` the `main()` call sites);
`STATE.md` "Hardware" section + the rung-2 launch entry. No GPU run, no box access,
no push performed in producing this document.
