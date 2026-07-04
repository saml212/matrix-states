# TRACKB_REDESIGN_ATTACK_R1 — Adversarial round 1 on `TRACKB_REDESIGN.md` (Rev 1, commit `52d834f`)

**Scope.** Design-only attack. No GPU, no box, no pushes. Every number below was
re-derived or grepped against the cited source, not taken from the target
document's own prose. File:line citations point at the actual repo state as of
this session.

**Verdict: NEEDS-REV-2.** Two FATAL findings both attack the document's single
most important deliverable — §5.3's 2×2 interaction bar, the only headline
scientific claim this redesign exists to make — and neither is a nitpick: one
shows the reference numbers on both sides of the comparison were never
demonstrated to be on the same scale, the other shows the comparison's own
self-diagnosed confound (write-budget) is disclosed but never controlled.
Eight MAJOR findings compound this (two independent budget-arithmetic errors,
a fabricated quote, a wrong-function citation for the one safety mechanism
this design explicitly says "must not be silently bypassed," an unspecified
Cell-4 composition rule, an unclosed STE-vs-selectivity confound, two
manipulation-check metrics with no pass/fail bar, and an untested-at-scope
stability risk for the reference arm). None of these show the underlying
hypothesis is unsound — all are closeable with a Rev 2. Do not build against
Rev 1 as written.

---

## FATAL

### F1 — The §5.3 headline bar compares two different instruments over two different K-populations, never shown comparable

**Where:** `TRACKB_REDESIGN.md` §5.3 (lines ~453–470), citing
`SCALE_TRANSFER_DESIGN.md` §5.9 (lines 883–893) for Cell 1's reference numbers
and §4.4 (line 513) for what Cells 2–4 will actually measure.

**The flaw.** Cell 1's reference band — "raw key-Gram deviation **21.93 ± 5.90**
(Wave C control...)," random-unit-vector anchor **7.94**, full collapse **63.50**
— is quoted verbatim from `SCALE_TRANSFER_DESIGN.md` line 890/892/893. I verified
these three numbers appear exactly as cited. But read the source's own framing
at line 884–886: this number comes from `lm_attractor_probe_rd.py`, computed at
**"chunk_size=64 (= d_state, single head ⇒ one 64-key episode per chunk)"** —
i.e. Gram deviation over **all 64 positions** in a chunk, because Track C's
control model has no selection mechanism at all. The random anchor (7.94) and
collapse anchor (63.50) are explicitly labeled **"(K=64, d=64)"** — computed at
K=64, nowhere else.

Cells 2–4, by contrast, will be measured with **Track B's own §4.4 instrument**
— quoted verbatim from `SCALE_TRANSFER_DESIGN.md` line 513: `key_gram_deviation
_mean`-style logging **"computed over each chunk's selected `K_sel` keys"** —
i.e. over 16 or 32 positions, not 64. TRACKB_REDESIGN.md §5.3 itself concedes
this instrument "was still a 'required build item' (§4.4) as of the reading
list — not yet run under Track B's own instrumentation at the time of this
writing" — but never draws the conclusion that follows: **Gram deviation over
K=64 keys and Gram deviation over K=16/32 keys are not the same statistic.**
Nothing in either document establishes that this metric is K-invariant (a
Frobenius-norm deviation from `I_K` over a K×K matrix has no obvious reason to
be); the random/collapse anchors, which the headline bar explicitly leans on
("markedly closer to the random anchor (7.94) than to cell-1's own band"), were
never recomputed at K_sel=16 or 32.

**Concrete failure scenario.** Cell 4 measures key-Gram deviation ≈9 at
K_sel=16 (selected subset only). The design reads this against the K=64 anchor
7.94 and declares "markedly closer to random" — passing the headline bar. But
if the true K=16 random-anchor value is, say, 3.5 (plausible — fewer
off-diagonal terms in a smaller Gram matrix), 9 is actually **2.6× the K=16
random floor**, not "markedly closer to random" at all. The bar's pass/fail
read is an artifact of comparing against the wrong K's anchor, not a real
signal.

**Fix that would close it.** Before §5.3's bars are used for anything: (a)
build Track B's own §4.4 instrument and use it for the *baseline (Cell 1)* too
— i.e. re-measure Cell 1 at the K_sel-selected-subset granularity, not reuse
Track C's whole-chunk number; (b) recompute the random-unit-vector and
full-collapse anchors at K=16 and K=32 specifically (a zero-GPU, closed-form or
Monte-Carlo computation, cheap); (c) only then set numeric bars. Until this is
done, §5.3's proposed bar is not a comparison at all.

### F2 — The self-flagged write-budget confound is disclosed, never controlled — including in the one test it would corrupt

**Where:** `TRACKB_REDESIGN.md` §3.2 (lines 237–241), §8 item 3 (lines 564–572),
§5.3 (lines 471–478).

**The flaw.** §3.2 explicitly names the confound for candidate 2: chunk-sum-≤1
sparsemax bounds total write mass, so "a val-loss regression attributable to
**underwriting**, not to selectivity per se" must be distinguished — and
commits to reporting total per-chunk β-mass. §8 item 3 extends the disclosure:
"applies in milder form to candidates 1/3/4 too... **Total per-chunk β-mass...
must be reported alongside every val-loss number**." That is the entire
control registered — a reporting requirement, not an experimental control.
Hard top-K masking (candidates 1/3, and therefore Cell 4, which is candidate
1-or-3 + geo3) *architecturally* zeroes ~50–75% of chunk β-mass relative to the
continuous-β baseline (per the gate's own numbers restated in §1: baseline
non-selected write-mass is 43.1–69.1% of chunk total at K_sel=32/16). Cell 3
(geo3-only) keeps the full continuous β. So Cells 2 and 4 will, by
construction, have systematically less total write-budget than Cells 1 and 3 —
and this is exactly the axis §5.3's interaction bar (`cell1−cell4 ≥
1.5×max(cell1−cell2, cell1−cell3)`) computes across.

**Concrete failure scenario.** Cell 4 clears the interaction bar. Is this
"orthogonality + selectivity interact super-additively," the claim §5.3 exists
to test — or is it "writing to fewer positions with less total mass produces
geometrically cleaner (lower-rank, less-collided) states, independent of
whether those positions are *targeted* well"? A pure write-budget reduction
(e.g. randomly zeroing the same *fraction* of positions, ungated by β) could
plausibly produce the same Gram-deviation improvement, and nothing in the
current design distinguishes the two. This is precisely the "does the design
control for write-budget... or confound 'selectivity helps' with 'writing
less'" question the task brief raised, and the design's own attack-yourself
(§8 item 3) states the risk but stops one step short of closing it.

**Fix that would close it.** Either (a) renormalize each hard-masked
candidate's selected β so total per-chunk mass matches the *matched* baseline
cell's own measured total (a cheap, deterministic rescale), or (b) add a
budget-matched negative control cell (random, β-blind subset selection at the
*same total write-mass* as the candidate, not just the same K_sel count — note
this is a different control from candidate 1's own `naive_window` mode, which
matches K_sel but not necessarily total mass since it doesn't rescale). Report
both the confound-controlled and raw numbers.

---

## MAJOR

### M1 — Churn-rate and support-size have no pre-registered pass/fail bar (the degenerate-model risk attack surface #1 asked about)

**Where:** §4.1 (lines 363–369, 380–387), §8 item 2 (lines 558–563).

Val-loss tolerances get exact numbers (+5% / +2%, §5.3). The Gram-deviation
bars get exact numbers (≤11.0, 1.5× multiplier, §5.3). But the two metrics
§4.1 explicitly substitutes for the now-tautological gate bars — candidate 1's
**selection-set churn rate** and candidate 2's **actual support size** — are
registered only as "a genuinely open, not-construction-guaranteed quantity" to
be *measured* (§4.1) or *reported* (§8 item 5), with no stated numeric
threshold anywhere in the document for what value would count as a problem.
A model with 40% of the top-K set flipping every checkpoint, or a sparsemax
support that never drops below K_sel−2, would be "reported" and then — by
what rule? — accepted or rejected into Wave 1. This is exactly attack surface
#1's degenerate case: a mechanism that is "selective by construction, useless
in function" can currently pass every registered gate, because the two
metrics designed to catch it have no gate. **Fix:** pre-register a churn-rate
ceiling (e.g., mean fraction of the selected set that differs from the
previous logged checkpoint) and a support-size floor for candidate 2, with the
same rigor already applied to val-loss and Gram-deviation.

### M2 — Wave −1 per-probe cost is internally inconsistent by ~3×

**Where:** §6.1, Wave −1 row (line 518).

The row states: "at Wave C's own measured ≈0.0000123 GPU-h/step (0.077 GPU-h /
6,103 steps) → **≈0.008 GPU-h/probe**" for "~2,000-step short probes." Redoing
the arithmetic the row itself specifies: `2,000 × 0.0000123 ≈ 0.0246 GPU-h`,
not 0.008 — a **3.07× discrepancy**. This isn't a rounding slip: 0.008 vs the
stated inputs' own product of 0.025 is a full order-of-magnitude-adjacent
error. It also breaks internal consistency with the row's own aggregate
estimate: 4 probes × 0.008 = 0.032 GPU-h, which is **below** the row's stated
floor of "~0.05–0.1"; 4 probes × the *correct* 0.025 = 0.098, which lands at
the row's own ceiling. The aggregate estimate is only reconcilable with the
correct per-probe number, meaning the "≈0.008 GPU-h/probe" figure printed in
the document is simply wrong and should read ≈0.025.

### M3 — Every geo3-active wave is priced at the non-geo3 rate, ignoring the codebase's own 3× placeholder

**Where:** §6.1, Waves 2/3 (lines 520–521), vs.
`run_lm_rd_geo3_sweep.py:94–104` (a file this same design's own §0 reading list
cites for a different purpose, lines 53–56).

Wave 2 prices "18 runs × 0.077 GPU-h" ≈ 1.4 GPU-h — but every one of those 18
runs is geo3-active (cell 3 is geo3-only, cell 4 is geo3+selectivity). The
0.077 GPU-h/run figure is the *non-geo3* baseline
(`SCALE_TRANSFER_DESIGN.md` §5.6: "≈361K tok/s/GPU... 6103 × 32 × 512 /
~274–278s"). `run_lm_rd_geo3_sweep.py` — a file in this design's own build
scope — registers its own timing caveat at lines 94–104: geo3-in-LM per-step
cost is "**conservatively 3x the non-geo3 baseline as a PLANNING placeholder**"
(`_PER_STEP_S_PLACEHOLDER_GEO3 = 0.15` vs. the ~0.05s/step non-geo3 baseline),
"NOT yet measured on-box for the geo3-active path." Applying that placeholder
to Wave 2's 18 geo3-active runs gives ≈4.2 GPU-h, not 1.4 — a **3× understatement
on that row alone**, pushing the "≈3.5–5.5 GPU-h central estimate" to roughly
6.3–8.3 GPU-h. This still fits comfortably under the ≤25 GPU-h target and the
≈34 GPU-h headroom, so it is not budget-threatening, but the design's own
stated purpose for §6 ("make that margin explicit and auditable") is
undermined by silently omitting a cost factor its own cited source file
flags.

### M4 — A quoted figure in §6 does not exist in the source it cites; an adjacent figure is stated with fabricated precision

**Where:** §6 (line ~493), citing `SCALE_TRANSFER_DESIGN.md` §5.6.

§6 states: *"rung-2's Rev 2.1 amendment registers ≈**129.4** GPU-h for its
wave, cumulative ≈163/300 at rung-2 launch time (§5.6, **"cumulative ≈162.5/300"**
/ "≈163")."* I grepped `SCALE_TRANSFER_DESIGN.md` in full for `162.5` and for
`129.4` (case-sensitive, whole file): **neither string appears anywhere in the
document.** The source (line 751–752) states, verbatim: *"≈129 GPU-h for the
wave (cumulative ≈163 of the §7 300 GPU-h ceiling)"* — no decimal digit on
129, and no "162.5" anywhere, quoted or otherwise. TRACKB_REDESIGN.md presents
"cumulative ≈162.5/300" **inside quotation marks as if directly sourced**,
which is a citation-integrity failure the CLAUDE.md "verify before claiming"
rule exists precisely to catch — this is a fabricated quote attributed to a
real section of a real document. The "129.4" figure is a related but distinct
problem: false precision not present in, and not derivable from, the cited
source's own "≈129."

### M5 — The override target cites the wrong function; the actual hard-refusal gate is never named

**Where:** `TRACKB_REDESIGN.md` §0 (lines 53–56) and §5.1 (lines 419–422),
citing `run_lm_rd_geo3_sweep.py:165–169`.

Both sections name `selection_mode_for_verdict` (lines 165–169) as "a real,
already-built refusal this redesign's own reference-arm cell (§5) must
explicitly override, not bypass silently," and again as "the existing hard
refusal (`selection_mode_for_verdict` raises given `verdict="no_launch_
redesign"`)." I read the actual file. `selection_mode_for_verdict` (lines
165–169) is:

```python
def selection_mode_for_verdict(verdict: str) -> str:
    assert verdict in ("beta_gated_primary", "naive_window_primary"), (
        f"selection_mode_for_verdict called with verdict={verdict!r} -- no_launch_redesign must "
        f"be refused by the CALLER before this function is ever reached")
    return "beta_topk" if verdict == "beta_gated_primary" else "naive_window"
```

Its own docstring-comment says the quiet part explicitly: *"no_launch_redesign
must be refused by the CALLER before this function is ever reached."* The
actual, production hard refusal is `_refuse_if_no_launch` (lines 172–183),
which prints a loud diagnostic and calls `sys.exit(3)`. Tracing `main()`:
`load_gate_verdict()` is called at line 598, `_refuse_if_no_launch(verdict,
gate_json_path)` immediately after at line 600 — **before** any manifest is
built. `selection_mode_for_verdict` is only ever called later, inside
`waveB1_manifest` (lines 200, 307) — code that `main()` never reaches when
the verdict is `no_launch_redesign`, precisely because `_refuse_if_no_launch`
already exited. **`selection_mode_for_verdict`'s assert is unreachable
dead-code protection, not the refusal mechanism.** Anyone implementing Rev 1's
override exactly as specified would patch or bypass the wrong function and
would not actually clear the real gate (`sys.exit(3)` in `_refuse_if_no_launch`),
or would patch the real one without registering it as required by this
document. Given the design's own framing — "this refusal is a real safety
feature against treating a failed-gate construction as a primary arm, and must
not be silently bypassed" — misidentifying which function *is* that safety
feature is a build-blocking error in the one place this design is most
careful to say it's being careful.

### M6 — Cell 4's composition rule (how `hard_select_k_sel` and `geo3_k_sel` interact) is never specified

**Where:** §3.1 (lines 177–180: "`hard_select_active` and `geo3_active` are
independently toggleable"), §5.1 (Cell 4 description, lines 413–414).

Candidate 1 introduces `hard_select_k_sel` as a separate hyperparameter from
geo3's existing `geo3_k_sel`. §5.1 describes Cell 4 as "the surviving
candidate + `geo3_active=True`, geo3 now orthogonalizing a genuinely selective
(near-zero-non-selected-mass) write set" — implying geo3 orthogonalizes
*within* the already-hard-masked write set. But nowhere does the document
state whether `hard_select_k_sel == geo3_k_sel` is required for Cell 4, nor
does it analyze either resulting case:

- **If equal:** after candidate 1's mask, `beta` is exactly zero everywhere
  except `hard_select_k_sel` positions. `_geo3_lm_select_and_orthogonalize`'s
  own `beta_topk` selection (`lm_pretrain_rd.py:292–301`) then ranks by that
  same `beta` and picks its own top-`geo3_k_sel`. If the two K's match, geo3's
  "selection" is forced, by construction, to reproduce candidate 1's mask
  exactly (only `K_sel` entries are non-`neg_inf`-ranked) — geo3 contributes
  zero new selection information; Cell 4 is candidate-1's mask with geo3's
  orthogonalizer bolted on, not an independent second selection layered on
  top, which changes what the "interaction" in §5.3 is actually testing.
- **If `geo3_k_sel > hard_select_k_sel`:** geo3 is forced to include some
  exactly-zero-`beta` (functionally inert — the write contributes nothing to
  the state regardless of key direction) positions in its top-K, spending
  orthogonalization budget on positions whose orthogonalization has zero
  effect on the actual model output.

Neither case, nor which is intended, is discussed. **Fix:** state the
required relationship explicitly and, if equal (the natural reading), note
that Cell 4's geo3 step is then redundant selection-wise and re-scope what
"the interaction" measures accordingly.

### M7 — STE-bias and selectivity-cost are not separably testable for candidate 1

**Where:** §3.1 "What failure means" (lines 187–199), §7 falsification map row
1 (line 541), §11 item 2 (lines 679–682).

The falsification map registers exactly one distinction: outright
divergence/NaN (uninterpretable, "STE pathology") vs. a bounded regression
past the ±5%/±2% tolerance (interpreted, per §3.1(i), as "STE's biased
gradient... too crude an approximation" *or* left implicitly as evidence about
selectivity's own cost — the document does not say which). A moderate,
converged-but-tolerance-missing regression is consistent with **both** "hard
selectivity architecturally hurts the LM objective" (the scientifically
interesting negative result the whole redesign is partly designed to detect)
**and** "STE is simply a poor gradient estimator for this loss landscape,
independent of whether hard selectivity itself is bad" — and nothing in the
registered metrics distinguishes them. §11 item 2 names the adjacent question
("would hard-concrete/L0... behave meaningfully better") as open and
explicitly "not resolved by fiat" — an honest disclosure, but disclosure is
not a plan: no cell, ablation, or fallback is registered to close this before
Cell 4's headline interaction claim (which inherits candidate 1 if it's the
surviving primary) is drawn. **Fix:** either commit to running the
hard-concrete/L0 alternative as a cheap secondary check whenever candidate 1
misses its val-loss tolerance, or explicitly narrow every claim candidate 1
can support to "hard-masked-with-STE," never "hard selectivity," until that
ambiguity is closed.

### M8 — Cell 3 is an untested-at-scope stability risk the design discusses only as a budget/bureaucracy question

**Where:** §5.1 (lines 413–428), §10 (line 663), §11 item 3 (lines 683–688),
vs. `lm_pretrain_rd.py:349–353`.

Cell 3 (geo3-only reference arm) requires the **first-ever full,
multi-thousand-step training run** of `geo3_active=True` in LM mode. Every
piece of existing evidence for this mechanism is either (a) synthetic-harness
training (`model_rd.py`, a structurally different model with masked β and a
buffer token), or (b) LM-mode forward-pass-only Wave −1 gate probes on an
*already-trained-without-geo3* Wave C checkpoint, or (c) short gradient-
finiteness smoke tests. The LM-mode selection code itself carries a documented,
unresolved risk comment (`lm_pretrain_rd.py:349–353`): *"a FULLY-VALID episode
whose selected keys contain >=~6 exactly-duplicated rows (identical
conv-context 4-grams, e.g. heavily tabular/repetitive text) also yields
coincident Gram eigenvalues in the fallback and can NaN the same way... flagged
for Wave 1 monitoring, not assumed away."* OpenR1-Math (LaTeX, repeated symbols,
templated proof structure) is exactly the "heavily tabular/repetitive text"
regime this comment warns about. TRACKB_REDESIGN.md's own discussion of Cell 3
(§5.1's override mechanics, §10's cut order, §11 item 3's "is a fresh run
worth the ≈0.46 GPU-h") is entirely about bureaucracy and cost-vs-information
tradeoff — it never surfaces this specific, code-documented numerical
stability risk, and registers no Wave −1 stability smoke for Cell 3 analogous
to the ones required for candidates 1–4 (§4.1). **Fix:** add a short (few-
hundred-step) full-training stability probe for Cell 3 specifically, checking
skip-rate from the isfinite-grad guard, before committing Wave-2 GPU-h to the
full 6-run manifest.

---

## MINOR

### N1 — K_sel is never pinned for the Wave 1 / 2×2 manifest

The original Track B design tested `K_sel ∈ {16, 32}` explicitly
(`SCALE_TRANSFER_DESIGN.md` §4.5). TRACKB_REDESIGN.md discusses `K_sel` only
as an abstract per-candidate flag (`hard_select_k_sel`, `geo3_k_sel`) and its
§6.1 budget rows never state how many `K_sel` values are actually run for Wave
1/2 — "up to 4 candidates × 2 corpora × 3 seeds" has no `K_sel` factor in it
at all. If only one `K_sel` is intended, say so and justify which; if both,
the run-count and GPU-h estimates in §6.1 need a ×2 factor that isn't there.

### N2 — §3.1's insertion-site description misstates where the reused code actually sits

§3.1 states the new hard-mask logic goes "immediately after `beta =
torch.sigmoid(self.b_proj(x))` (current line 529) and before `q,k,v` are
reshaped into heads (line ~531) — the exact site `_geo3_lm_select_and_
orthogonalize` is already called from." Reading `lm_pretrain_rd.py:526–551`:
`q, k, v` are reshaped at lines 531–533, and the *existing* geo3 call happens
afterward, at line 539–551, operating on the already-reshaped `(B,T,H,head_dim)`
tensor (required by `_geo3_lm_select_and_orthogonalize`'s own shape contract,
`B, T, H, d = k_raw.shape`). It is not "the exact site" of the reshape
boundary claimed. This doesn't break candidate 1's own mechanism (β's shape
is unaffected by the q/k/v reshape), but it is an inaccurate claim about the
codebase in a document that otherwise cites line numbers as load-bearing
evidence.

### N3 — §6.1's stated central estimate doesn't sum its own table

Summing §6.1's own row ranges: low ends `0.05+0.9+1.4+1 = 3.35`, high ends
`0.1+1.85+1.4+2 = 5.35`. The document states "≈3.5–5.5 GPU-h central
estimate" — close but not exact, and (per M3 above) already an understatement
before the geo3 pricing correction is even applied.

---

## Claim-tier discipline (attack surface #6) — checked, no violation found

I read every candidate section (§3.1–3.4), the motivation sections (§1, §2),
and §9 specifically hunting for a sentence that quietly licenses a claim about
pretrained/production delta-rule LMs. I did not find one. §9 explicitly names
Track D's non-attribution finding by name and forecloses generalization even
in the case of a clean interaction result; the falsification map (§7, last
row) repeats this; §8 item 4 pre-answers the attack round's likely question on
exactly this axis. This is the one attack surface that came back clean —
worth stating plainly rather than manufacturing a finding to fill the
category.

---

## Required changes for Rev 2 (summary)

1. Rebuild Cell 1's reference band using Track B's own §4.4 (K_sel-scoped)
   instrument once built — do not reuse Track C's K=64 whole-chunk number.
2. Recompute the random-anchor/collapse-anchor reference points at K_sel=16
   and 32 specifically; do not reuse the K=64 values (7.94 / 63.50).
3. Add a real write-budget control (renormalization or a budget-matched
   negative-control cell) to the §5.3 interaction test, not just a reporting
   requirement.
4. Pre-register exact numeric pass/fail bars for churn-rate (candidate 1) and
   support-size (candidate 2).
5. Fix the Wave −1 per-probe arithmetic (≈0.025 GPU-h/probe, not 0.008) and
   re-total.
6. Apply `run_lm_rd_geo3_sweep.py`'s own 3× geo3-active pricing placeholder to
   every geo3-active wave/cell and re-total the budget.
7. Remove the fabricated "cumulative ≈162.5/300" quote and correct
   "129.4"→"≈129" to match the actual source text.
8. Re-target the Cell-3 override at `_refuse_if_no_launch` (not
   `selection_mode_for_verdict`) and describe the override mechanics against
   that function's actual code (lines 172–183, 598–600).
9. Specify the required relationship between `hard_select_k_sel` and
   `geo3_k_sel` for Cell 4 and address the redundant-selection /
   zero-β-orthogonalization cases that follow from each choice.
10. Register a plan (or an explicit claim-scope narrowing) to separate STE
    gradient bias from hard-selectivity's own cost before drawing any
    candidate-1-inherited claim in Cell 4.
11. Add a short full-training stability probe for Cell 3 targeting the
    code-documented near-duplicate-row NaN risk on real repetitive text.
12. Pin a concrete `K_sel` value or grid for the Wave 1/2×2 manifest.
13. Fix §3.1's insertion-site description (existing geo3 call is after the
    q/k/v reshape, not before).

---

## Reproducibility pointers

Read in full or in the cited section this session, no GPU/box access:
`matrix-thinking/TRACKB_REDESIGN.md` (target, all 712 lines);
`matrix-thinking/SCALE_TRANSFER_DESIGN.md` §4 (382–599), §5.6 (715–805), §5.9
(860–950), §7 (1305–1320); `matrix-thinking/DELTANET_RD_EXACTNESS_DESIGN.md`
§14 (1550–2269, incl. §14.6 gate spec 2121–2267), §16.6–16.7 (2951–3046);
`matrix-thinking/KEY_ANCHORING_DESIGN.md` header (1–15), §1–§2.0 (231–328),
§3.6 (931–990), §8.3 (1600–1627); `matrix-thinking/deltanet_rd/lm_pretrain_rd.py`
(1–60, 206–398, 419–570); `matrix-thinking/deltanet_rd/run_lm_rd_geo3_sweep.py`
(1–200). Grep-verified claims: `162.5` and `129.4` absent from
`SCALE_TRANSFER_DESIGN.md` (zero hits, full-file search); `21.93`, `7.94`,
`63.50` present and accurate (line 890/892/893); KEY_ANCHORING 2×2 numbers
(44–56×, 0.44→1.0000) accurate against §2.0's own table.
