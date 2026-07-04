# TRACKB_REDESIGN_ATTACK_R2 — Adversarial round 2 on `TRACKB_REDESIGN.md` (Rev 2, commit `f814b99`)

**Scope.** Design-only attack, fresh agent, no memory of round 1's authorship. No GPU,
no box, no pushes. CPU/closed-form reproduction where cheap (Monte-Carlo closed forms,
budget arithmetic, line-number/quote grepping against the actual repo state this
session). Every number below was independently re-derived or grepped, never taken from
either Rev 2's or round 1's own prose at face value — round 1 caught a fabricated quote,
so quote integrity (and, this round, *arithmetic consistency between adjacent quoted
numbers*) is a standing suspicion, not a formality.

**Verdict: NEEDS-REV-3.** Round 1's two FATALs are genuinely, substantively addressed —
F1's single-instrument-at-K_sel fix and F2's mandatory renormalization + Cell 2R control
are real mechanical closures, not disclosures dressed up as controls, and every
code-level citation Rev 2 makes to close M2–M6 was independently verified byte-for-byte
against the actual files (line numbers, function bodies, assert messages all match
exactly). But round 2 hunting Rev 2's own *new* additions surfaces seven new MAJOR
findings, all in mechanisms Rev 2 itself introduced to close round 1's findings: the F2
renormalization control has an asymmetric shortfall-handling gap and an unresolved
randomization-cadence ambiguity in its own new control cell; the M1 churn-ceiling's
"non-circular" defense doesn't establish it's *calibrated*, and the sibling
positional-artifact check it was meant to complement turns out to have no actual
specification at all (a broken internal citation masks this); the M7 comparator's new
temperature-annealing curriculum reintroduces a confound (soft-warmup benefit vs. STE
bias) that the document itself, elsewhere, already treats as a *separate* axis; the M8
stability smoke's slice-construction criterion doesn't verifiably target the failure
mode it exists to catch; and the cited source document's own budget arithmetic
(`SCALE_TRANSFER_DESIGN.md`'s Rev 2.2 amendment) contains an internal ~26–27 GPU-h
inconsistency between two adjacent sentences that Rev 2 quotes accurately but does not
reconcile. None of these is FATAL — none shows the redesign's core premise or headline
test is unsound — but each is closeable only with real design work, not a wording fix,
so this is not yet buildable as written.

---

## Part 1 — Closure table (round-1 findings against Rev 2)

| # | Severity (R1) | Disposition | Verdict |
|---|---|---|---|
| F1 | FATAL | Single §4.4 instrument now used for every cell incl. Cell 1's read-only re-probe; anchors recomputed at K_sel; K=64 numbers banned from bars | **CLOSED** |
| F2 | FATAL | Mandatory `B_pinned` renorm + required Cell 2R budget-matched random control + downgrade rule | **PARTIALLY CLOSED** — real mechanism, two new gaps (below) |
| M1 | MAJOR | `BANDS_PINNED-TrackB` (all 3 parts, writer/launcher-gate/readout-assertion) + derivation rules | **MOSTLY CLOSED** — mechanism is real; churn-ceiling calibration and the sibling positional-artifact check both have new gaps (below) |
| M2 | MAJOR | Per-probe cost corrected to ≈0.025 GPU-h | **CLOSED** — verified: 0.077/6103 = 1.26e-5, ×2000 = 0.0252 |
| M3 | MAJOR | geo3-active runs re-priced at 3× placeholder (≈0.28 GPU-h/run) | **CLOSED** — verified: 0.15×6103 + 7×15 = 1020.45s = 0.2835h |
| M4 | MAJOR | Fabricated quote removed; 129.4/162.5 now correctly (unquoted) attributed to `STATE.md` | **CLOSED** on its own terms — but see new BUDGET finding below (the *source* document's own adjacent cumulative figures don't reconcile) |
| M5 | MAJOR | Override re-targeted at `_refuse_if_no_launch` (:172–183, `sys.exit(3)`, called at :600 after `load_gate_verdict` at :598) | **CLOSED** — verified byte-for-byte against `run_lm_rd_geo3_sweep.py` |
| M6 | MAJOR | Cell 4 composition pinned: `hard_select_k_sel == geo3_k_sel`, single selection source via a new forced-selection argument | **CLOSED**, one MINOR build-time gap (below) |
| M7 | MAJOR | Temperature-annealed soft-top-K comparator added with a registered decision rule | **PARTIALLY CLOSED** — comparator's own curriculum introduces a new confound (below) |
| M8 | MAJOR | Wave −1 geo3-LM stability smoke on a tabular-risk corpus slice, gating Cells 3/4 | **MOSTLY CLOSED** — mechanism (isfinite-grad/skip_rate) verified real in code; slice-construction has a targeting gap (below) |
| N1 | MINOR | `K_sel = 32` pinned, 4 reasons stated | **CLOSED**, with a sharpened caveat on reason (ii) (below) |
| N2 | MINOR | Insertion-site description corrected (mask sits upstream of reshape+geo3 call) | **CLOSED** — verified byte-for-byte against `lm_pretrain_rd.py:526–551` |
| N3 | MINOR | §6.1 re-totaled, states its own row-sum arithmetic | **CLOSED** — verified: 0.3+1.8+3.4+1.0=6.5; 0.5+2.8+5.0+2.0=10.3; ×2 → 13–20.6; +1.7 → 22.3 |

---

## Part 2 — New findings

### NEW-1 (MAJOR) — F2's shortfall handling is asymmetric: only candidate 2 gets a decision rule, candidates 1/3/4 get disclosure only

§2 principle 4 registers a single clamp-shortfall rule with teeth for candidate 2 only:
*"if median shortfall exceeds 10% of `B_pinned`, candidate 2's budget-match is declared
PARTIAL in every readout."* Candidates 1/3/4 and the comparator/control cells get
*"any clamp-induced mass shortfall logged per chunk, never silent"* — logging, not a
threshold or a downgrade rule. Recomputing Rev 2's own planning numbers shows this gap
is not academic: non-selected mass at K_sel=32 = 32×0.3625 ≈ 11.60; total chunk mass =
11.60/0.4308 ≈ 26.9; **pre-renorm** selected-only mass = 26.9−11.60 = 15.30 over 32
positions → pre-renorm mean selected β ≈ 0.478. Post-renorm target (all mass
concentrated on the 32 selected positions) = 26.9/32 ≈ **0.84** — matching Rev 2's own
stated figure. The required per-chunk scale factor is therefore ≈0.84/0.478 ≈ **1.76×**
on average, leaving only **~16% headroom** (1.0 − 0.84) before an *average* chunk's
rescaled β hits the clamp. Given 12,288 pooled chunk-episodes necessarily span
above- and below-average organic concentration, a non-trivial fraction of chunks will
plausibly need some rescaled β > 1.0 for candidates 1/3/4 too — yet only candidate 2 has
a registered consequence when that happens. This reintroduces, for a narrower
cell-subset than before, exactly the "a reporting requirement is not a control" pattern
F2 was raised to kill. **A precise numeric bound cannot be derived from the cited
artifacts**: `wave_neg1_gate.json` reports only pooled means and one Gini per K_sel, no
percentile/variance data on the per-chunk selected-subset's own dispersion — so neither
Rev 2 nor this attack can currently say *how much* shortfall to expect for candidates
1/3/4, only that the headroom margin is tight enough (16%) to worry about.
**Fix:** extend candidate 2's median-shortfall-vs-10%-of-`B_pinned` rule (or an
equivalent) to every masking candidate, not just candidate 2; if budget allows, mine the
already-archived `wave_neg1_gate.json` per-checkpoint/per-layer breakdown for a cheap,
zero-GPU estimate of selected-subset dispersion before Wave 1 commits.

### NEW-2 (MAJOR) — Cell 2R's randomization cadence is unspecified; the frozen-mask risk it should prevent can reappear inside the fix

§5.1 specifies Cell 2R as *"a β-blind, per-chunk *random* K_sel-subset selection
(content positions only, corpus-seeded)."* This does not say whether the random subset
is **redrawn independently for every chunk instance** encountered during training
(required for the control's own stated property, "zero targeting information"), or
**fixed by a seed keyed to intra-chunk position** (e.g., a single draw of "which
K_sel-of-64 relative offsets are selected" applied identically to every chunk in the
corpus). Under the second reading, Cell 2R silently degenerates into a **fixed
positional schedule** — structurally the same failure shape as candidate 3's
non-learned periodic mask, and exactly the kind of "selective by construction" artifact
§3.1(iii) already worries about for candidate 1. If Cell 2R's randomness is fixed rather
than resampled, its Gram-deviation/val-loss numbers could reflect "some positions in a
chunk systematically differ from others" (a real, if crude, structural effect having
nothing to do with random budget-matching) rather than genuinely "zero targeting
information" — corrupting the very comparison (`Cell 2R vs Cell 2` within seed noise)
F2's downgrade rule depends on. **Fix:** state explicitly that the random subset is
redrawn per (chunk instance, training step) from a corpus-and-step-independent RNG
stream, not derived from a fixed function of intra-chunk position.

### NEW-3 (MAJOR) — Cell 2R's decision rule has n=3 not n=1 (contra a plausible reading of the design), but no registered statistical test or power target

§6.1 prices Cell 2R at exactly 6 runs in Wave 1 ("comparator (6) + Cell 2R (6)"),
matching the design's own standing "2 corpora × 3 seeds" convention used everywhere else
(Cell 3's identical 6-run pricing is explicitly 2×3 at §11 item 3) — so Cell 2R is run at
**n=3 seeds**, not n=1, and the decision rule ("if Cell 2R's improvement over Cell 1
matches Cell 2's within seed noise") is at least *evaluable* rather than undecidable.
But nothing registers **how** "matches within seed noise" is operationalized (a
threshold on the difference of means relative to pooled seed SD? an explicit
equivalence-test margin?) or whether n=3 per arm has adequate power for it. This matters
because the reference precedent Rev 2 leans on throughout (`KEY_ANCHORING_DESIGN.md`
§3.6) upgraded from n=2 to n=3 specifically because n=2's sample-SD relative standard
error (~71%) was judged unacceptable, and even at n=3 its own document calls the ~50% RSE
merely "realistic rather than hypothetical," not solved — and that precedent was
calibrated for a **one-sample-vs-fixed-band** comparison (an arm's n=3 mean against a
pre-derived ceiling), not the **two-sample** comparison (Cell 2R's n=3 vs. Cell 2's n=3)
Rev 2 actually needs, which generically requires *more* replication for equivalent power,
not the same headcount. Rev 2 imports the n=3 headcount convention without importing (or
re-deriving) the power justification behind it. **Fix:** register an explicit
equivalence margin (e.g., "matches within seed noise" ≡ |Δ| < some multiple of the
pooled seed SD) and state, even approximately, whether n=3-per-arm supports detecting the
effect size the design cares about — or explicitly flag the comparison as
descriptive-only if it doesn't.

### NEW-4 (MAJOR) — The churn-ceiling's "non-circular by construction" defense establishes non-triviality, not calibration

§4.3 derives `churn_ceiling_Ksel = mean_ref + 2·s_ref` from **an unmasked,
Cell-1-architecture reference pilot's** implicit top-K_sel-by-β churn, arguing this is
"non-circular by construction: the reference is the soft β ranking the mask is built
FROM." That argument is true but proves a weaker property than the document treats it
as proving: it shows the ceiling isn't rigged to trivially pass (candidate 1 isn't
measured against its own churn), but it does **not** show the unmasked pilot's churn is
a *representative null* for a model trained the entire run under a hard mask + STE. Two
concrete, plausible-either-direction mechanisms the reference can't see: (a)
**self-reinforcing entrenchment** — once a position is selected, STE's dense gradient
keeps favoring it (candidate 1's β_soft never sees an honest "would this position have
helped if selected" signal at non-selected positions, since the STE backward treats the
mask as identity everywhere), which could *suppress* churn well below an unmasked
pilot's organic level, for reasons having nothing to do with "the mask injecting
instability"; (b) **selection-sharpening** — a well-functioning selector may
legitimately develop *more* rank churn early in training as it learns to discriminate
positions more sharply than an unmasked model ever needs to, which the ceiling would
misread as instability. Either direction means `mean_ref + 2·s_ref`, measured on a
categorically different training regime, could be a poorly calibrated bound in either
direction. **Fix:** at minimum, register this as an acknowledged approximation (not
"non-circular" ⇒ solved); if budget allows, cross-check the pinned ceiling against
candidate 1's own **early-training** churn (its first 5 log points, mirroring the
reference pilot's own window) as a same-mechanism sanity check before trusting the
full-run ceiling.

### NEW-5 (MAJOR) — The positional-artifact check (§3.1 iii) has no actual specification; its only citation points to the wrong place

§3.1(iii) names the *other* half of the degenerate-model risk M1 was raised to close —
a mask that "converges to a content-independent positional artifact... checked directly
by comparing the selected-set distribution across documents at matched intra-chunk
offsets (**§8 item 1**)." Grepped the document in full: `§8 item 1` is the *only* place
this check is cited, and §8 item 1 (in "The Rev-1 pre-answered attack predictions"
section) is *"The manipulation check is trivially satisfied for candidates 1/3"* —
disposed of by pointing to §4.3's pinned bars, with **no content whatsoever** about
comparing selected-set distributions across documents at matched offsets. This is a
broken cross-reference, and behind it there is genuinely no method spec anywhere in the
document: no statistic, no threshold, no registered bar for the positional-artifact
check — only the one descriptive sentence at §3.1(iii) asserting it "is checked
directly." Unlike churn (which got a full `BANDS_PINNED` treatment this round), the
positional-artifact failure mode — arguably the *more* dangerous of the two, since a
frozen positional collapse would show near-**zero** churn and would sail under the M1
ceiling while still being exactly the "selective by construction, useless in function"
case M1 exists to catch — has no registered defense at all. **Fix:** either specify the
check for real (a concrete statistic — e.g., mutual information or a chi-square test
between selected-position-within-chunk and document identity, over a fixed probe
batch — with a numeric bar, mirroring §4.3's treatment of churn/support) or explicitly
fold it into the same `BANDS_PINNED-TrackB` pinning pass, and fix the citation.

### NEW-6 (MAJOR) — The M7 comparator's soft-then-hard curriculum confounds "STE bias" with "benefit of an annealed schedule," undermining its own decision rule

The comparator's registered curriculum is "non-selected β multiplied by τ(t), linear
1→0 over the first 80% of steps, then exactly 0." At τ≈1 (most of training start), the
comparator trains essentially as **Cell 1 with no selectivity at all** (β at
non-selected positions ≈ full β_soft, i.e. an unmasked model); it only becomes
forward-equivalent to candidate 1's hard mask in the **final 20%** of training.
Candidate 1, by contrast, is hard-masked **from step 0** for the entire run. If the
comparator succeeds where candidate 1 fails, the registered decision rule attributes
this to "STE gradient bias" — but an equally consistent explanation is "a long soft
warm-up before hardening trains better than an abrupt hard mask from init," with no
reference to gradient estimators at all. This is not a hypothetical alternative reading:
it is **exactly candidate 4's own soft-warmup-then-hard-snap idea** (§3.4: "snap to a
literal hard top-K mask... in a final phase once the soft loss has concentrated β
somewhat"), which the document elsewhere explicitly treats as a *distinct*,
lower-priority mechanism from candidate 1's pure hard-mask-with-STE. The M7 comparator
as specified silently reintroduces that same curriculum axis into what is presented as
a clean "same forward, unbiased gradient" isolation of STE — so a comparator success
cannot cleanly distinguish "STE was the problem" from "the curriculum was the fix,"
weakening the very claim (§7's falsification-map row 1, §10 item 5's cut-order
reasoning, §11 item 2's framing) that depends on this isolation being clean.
(Separately, and not a flaw: the "bit-equivalent endpoint" claim is achievable exactly
as stated — multiplying a finite β by an exact `0.0` float always yields exact `0.0` in
IEEE 754, so the τ=0 endpoint forward pass genuinely is bit-identical to candidate 1's
mask, given matching selected-position handling. Nothing in the registered mechanics
actually checks or depends on this bit-equivalence operationally, though, so the
precision of the claim is accurate but decorative — not itself a problem, just not the
place where this comparator's real issue lives.) **Fix:** a clean STE isolation needs
the ONSET of hard selection held constant with candidate 1 (hard from step 0), varying
only the gradient estimator (true, vanishing-but-present gradient at non-selected
positions vs. STE's dense, biased pass-through) — not an annealed curriculum. If the
annealed comparator is kept for its own (real, but different) scientific value, register
it as answering "does an annealed curriculum help," a secondary question already
occupied by candidate 4, and add a genuinely-isolating hard-from-init/true-gradient
variant as the actual STE control before drawing the attribution the falsification map
currently licenses.

### NEW-7 (MAJOR) — The M8 stability-smoke slice is selected by a text-level proxy that doesn't guarantee it stresses the actual NaN mechanism

The tabular-risk slice is "the top decile of openr1 windows ranked by repeated
conv-context 4-gram fraction" — confirmed via direct code read
(`lm_pretrain_rd.py:349–353`) that the real risk requires **≥~6 exactly-duplicated rows
among a chunk's β-*selected* top-K keys** specifically, not merely duplication
somewhere in the window. A window can score high on raw text-level 4-gram repetition
without its repeated spans landing among the specific positions β happens to select —
selection is driven by the (untrained-at-slice-construction-time, evolving) learned β,
not by which tokens are textually repeated. The slice-ranking criterion is therefore
only a proxy for the actual precondition the risk depends on, and the gap between them
is not measured or bounded anywhere in the document. If the proxy under-selects for
the true risk, the Wave −1 smoke could pass cleanly (zero non-finite losses, skip-rate
≤1%) while never actually exercising the near-duplicate-selected-keys regime — false
reassurance on exactly the gate that's supposed to be Cells 3/4's safety check before
committing real training spend. **Fix:** validate the proxy once, cheaply (post-hoc, on
the smoke run itself): log the count of near-duplicate rows **within the selected
top-K set specifically** (not just window-level repetition) alongside the existing
skip-rate/finite-loss bars, so the smoke's own output can confirm or refute whether the
slice actually reached the regime it targets.

### NEW-8 (MAJOR) — `SCALE_TRANSFER_DESIGN.md`'s own Rev 2.2 budget arithmetic has an internal ~26–27 GPU-h inconsistency that Rev 2 quotes accurately but does not reconcile

Recomputing §6's cited source paragraph directly (`SCALE_TRANSFER_DESIGN.md:790–805`):
the rejected 3B-tokens/run option prices at ≈152.5 GPU-h with **"cumulative ≈316"**; the
accepted 1.5B-tokens/run option prices at ≈76.2 GPU-h with **"cumulative ≈266/300."**
Both per-run/per-wave prices are internally consistent with each other (152.5/2 = 76.25
≈ 76.2, exactly the halving expected from halving the token target). But the *implied
baseline* each cumulative figure is added to is not: 316 − 152.5 = **163.5**; 266 − 76.2
= **189.8** — a gap of **26.3 GPU-h** between two sentences in the same paragraph,
pricing the same rung-3 decision against what should be the identical pre-rung-3
baseline. 163.5 matches the rung-2 registration's own stated cumulative (≈163,
`SCALE_TRANSFER_DESIGN.md:751–752`) almost exactly; 189.8 does not match anything
directly stated, but is suspiciously close to the same amendment's own "`--wave 1ext`...
projected cost ≈27.0 GPU-h... **BUILT, GATED, NOT YET LAUNCHED**" figure two paragraphs
earlier — circumstantially consistent with that not-yet-run wave's projected cost having
been folded into one cumulative figure but not its sibling. This is a defect in the
**cited source document**, not a misquotation by `TRACKB_REDESIGN.md` — every string
Rev 2 quotes from this passage is grep-verified present, exactly as R1's discipline
demands — but Rev 2's own headroom claim ("`300 − 266 = 34` GPU-h... nominal headroom")
inherits an unreconciled source figure without flagging that its sibling figure implies
~26–27 GPU-h more headroom is actually available (i.e., the true post-rung-3 cumulative
is more likely ≈240 than ≈266). This *helps* Track B's margin if true, so it is not a
budget-safety risk in the dangerous direction — but it means §11 item 5's "is 34 GPU-h
still accurate" question is under-scoped: the answer isn't just "pending rung-2/3
harvest data" (confirmed absent from `EXPERIMENT_LOG.md` — no `rung-2`/`rung-3` harvest
entry exists yet, only the rung-1 harvest and the rung-2 launch entry), it's also
"pending reconciliation of two disagreeing *planning* figures in the same source
paragraph." **What a 10% rung-3 overrun does to the two readings:** at the stated
baseline (266 + 7.62 = 273.6/300, an 26.4 GPU-h margin before Track B's own spend); at
the internally-consistent baseline (163.5 + 76.2×1.1 = 247.4/300, a 52.6 GPU-h margin).
Track B's own worst case (22.3 GPU-h) fits under either reading, but the actual slack
left *after* Track B ranges from ~4 GPU-h to ~30 GPU-h depending on which figure is
right — not a rounding-scale difference. **Fix:** flag this specific inconsistency to
whoever owns `SCALE_TRANSFER_DESIGN.md` for reconciliation before either number is
treated as authoritative; re-derive current actual cumulative spend from
`EXPERIMENT_LOG.md`'s real harvest entries (not either projected cumulative figure)
immediately before Track B's Wave 1 launches.

### MINOR-1 — N1's K_sel=32 justification (ii) is in tension with a different, equally valid read of the same gate data

Reason (ii) for pinning K_sel=32 is "K_sel=32 is where the original gate came closest to
passing (0.569 vs 0.60)." Read as **excess over each K's own uniform floor** instead of
absolute distance to a fixed bar, the picture inverts: K=16's measured top-K mass
(30.9%) exceeds its uniform floor (25%) by a **relative 23.7%**; K=32's (56.9% vs 50%)
exceeds its floor by only a **relative 13.8%**. By this read, K=16 is where β's
discriminative signal is comparatively *stronger* relative to chance, not K=32. Reason
(ii) therefore ties the K-choice to the same criterion (absolute proximity to the
originally-registered 0.60 bar) that produced the *failing* verdict in the first place —
a mildly circular flavor specific to that one reason (reasons (i), (iii), (iv) are
untouched by this critique and remain independently sufficient to keep K=32 as the
primary pin). **Fix:** either drop reason (ii) or restate it honestly as "closest to an
arbitrary absolute bar, not necessarily the K with the strongest relative signal," and
treat the already-registered K=16 follow-on as higher-priority than "not silently
dropped" implies.

### MINOR-2 — M6's forced-selection argument doesn't yet specify EOT/padding validity threading

`_geo3_lm_select_and_orthogonalize`'s current implementation (verified,
`lm_pretrain_rd.py:292–301`) derives its EOT/padding-exclusion validity mask
(`valid_sel = topk_val > neg_inf/2`) from comparing its own internal `torch.topk`
output against `neg_inf` — a check that has nothing to compare against once Cell 4's
planned "forced-selection argument" bypasses that internal `topk` call and supplies
candidate 1's own indices directly. This is a small, clearly buildable gap (validity
can instead be derived by gathering `content_mask` at the forced indices) but is not
addressed in Rev 2's text and would need to be resolved during implementation, not
assumed away.

---

## Part 3 — Priority-check answers (condensed)

**1. F1 / MC anchors / checkpoint compatibility.** Closed-form check: at K=32, d=64,
`√(K(K−1)/d) = √(992/64) = √15.5 = 3.937 ≈ 3.94`; `√(K(K−1)) = √992 = 31.497 ≈ 31.50` —
both match Rev 2's stated planning values exactly; the same formula reproduces §6.8's
own K=16 anchors (1.94 at d=64) exactly, so the closed form is trustworthy as a
cross-check. Checkpoint-format compatibility for Cell 1's read-only re-probe is
low-risk: these same 6 Wave C checkpoints were **already** successfully loaded and
forward-probed by the existing (already-run) `wave_neg1_gate.json`; `DeltaNetLMMixer`'s
geo3_* constructor args all carry defaults (verified, `:439–443`), so
`DeltaNetLM(**ckpt["config"])` reconstruction from a pre-geo3 config dict will not
fail. (Note for the record: "Cell 1" is unambiguously the 14M-param Wave C LM
checkpoints, `n_params=14,048,896`, not Track C's 98M rung-1 scaling-ladder
checkpoints — Rev 2's own text is clear on this; flagging only because the task
framing's "rung-1-era" phrasing could be misread as the latter.)

**2. F2 clamp/renorm.** See NEW-1 (shortfall asymmetry, ~16% average headroom before
clamp, no numeric bound derivable from the cited gate JSON) and NEW-2/NEW-3 (Cell 2R
randomization-cadence ambiguity; n=3 not n=1, but no registered statistical test).

**3. BANDS_PINNED reuse.** Genuinely reused, not name-dropped — all three parts
(writer/launcher-gate/readout-assertion) are registered with concrete failure modes
that match `KEY_ANCHORING_DESIGN.md` §3.6's own mechanics part-for-part (completeness
validation + sha256 hashing at write time; re-hash + refusal at launch time; timestamp-
precedes-launch assertion at readout time). See NEW-4 (churn-ceiling calibration doubt)
and NEW-5 (the sibling positional-artifact check has no spec and a broken citation).

**4. M7 comparator / "bit-equivalent."** Bit-equivalence at τ→0 is achievable and
correctly claimed (exact-zero float multiply); nothing load-bears on the word "bit."
The real issue is the curriculum confound — see NEW-6.

**5. New-flaw sweep.** Covered above: Cell 2R (NEW-2), override stamping (verified
CLOSED — assembly-time, both-path fields, matches the KEY_ANCHORING convention exactly,
confirmed against `lm_pretrain_rd.py::_assemble_result` at line 1086 and its smoke
assertion at line 1421), NaN smoke slice (NEW-7), K_sel=32 pin (MINOR-1).

**6. Budget.** §6.1 row-sums verified exact (see closure table, N3). Wide-case and
+Cell4R totals verified exact (13–20.6; 22.3). The ≈34 GPU-h headroom claim inherits an
unreconciled ~26–27 GPU-h gap in its own source document — see NEW-8, including the
10%-overrun scenario worked both ways (26.4 GPU-h margin at face value; 52.6 GPU-h
margin at the internally-consistent reading).

**7. §11 adjudications.**
- Item 1 (0.5×/1.5× multipliers — derive via a simulator?): **No.** A predictive
  multiplier-derivation tool duplicates the exact hazard (§16.7's shared-scalar
  mis-wiring) that already bit this project once, for marginal calibration benefit at
  this Tier-2, exploratory budget scale. Keep as proposed-not-derived; treat bar
  outcomes as provisional until replicated.
- Item 2 (hard-concrete/L0 third mechanism): **Correctly scoped as a conditional
  follow-on.** Building it now would be premature engineering against a disagreement
  pattern (candidate 1 vs. comparator) that may never materialize — but given NEW-6, the
  comparator's own attribution power is weaker than assumed, so this follow-on may need
  to be promoted sooner than "if they disagree" implies.
- Item 3 (Cell 3 worth it vs. inference from the existing gate JSON?): **Yes, and
  more than Rev 2's own tentative framing suggests.** The existing gate JSON measured a
  fundamentally different quantity — β-distribution shape on a **non-orthogonalized**
  checkpoint, forward-pass only, no training — and cannot supply a val-loss number under
  the interaction bar's own training protocol, nor can it substitute for Cell 3's
  separate role as the **first full-scale geo3-active LM training run** the M8 stability
  gate exists to de-risk. The §5.3 interaction bar's `max(cell1−cell2, cell1−cell3)` term
  is not computable at all without a real Cell 3. Recommend promoting Cell 3 to
  never-cut alongside Cell 2R, not leaving it as an open question.
- Item 4 (candidate 3's period `W` needs its own attack pass?): **No — disproportionate.**
  Candidate 3 is already cut-order-4 and self-described as answering the
  orthogonalization-set question "by fiat." A full attack pass on `W` before any build is
  more process than a lowest-priority candidate warrants; a cheap 2–3-point sensitivity
  check, gated only if candidate 3 survives to Wave 2, is enough.
- Item 5 (is ≈34 GPU-h headroom still accurate?): **Load-bearing, and more so than
  stated** — not just because rung-2/3 harvest data doesn't exist yet (confirmed: no
  `rung-2`/`rung-3` harvest entry in `EXPERIMENT_LOG.md`), but because the *source's own
  planning figures* disagree with each other by ~26–27 GPU-h (NEW-8). Re-derive from
  real harvest entries before Wave 1 launches; don't trust either projected cumulative.
- Item 6 (per-write magnitude a third axis?): **Concur with Rev 2 — inseparable by
  definition, not a control gap.** Total mass = count × mean, so at fixed `B_pinned` and
  fixed `K_sel`, count and per-write magnitude cannot vary independently; it's a
  mass-conservation identity, not an experimental choice. Cell 2R already isolates the
  interesting orthogonal question (targeted vs. random selection at matched mass and
  matched redistribution) — no further control cell is needed for this axis.

---

## Reproducibility pointers

Read in full or in the cited section this session: `TRACKB_REDESIGN_ATTACK_R1.md` (all
426 lines); `TRACKB_REDESIGN.md` Rev 2, commit `f814b99` (all 1056 lines, both halves);
`SCALE_TRANSFER_DESIGN.md` §4 (382–599), §5.6 incl. Rev 2.1/2.2 (715–806), §5.9
(860–1023), §6.8 (1161–1305), §7 (1305–1322); `KEY_ANCHORING_DESIGN.md` §2.0 (261–288),
§3.6 (931–1062); `matrix-thinking/deltanet_rd/results/lm_rd_geo3/wave_neg1_gate.json`
(read in full — `pooled.32.top_k_mass_frac_mean=0.5691615343093872`,
`gini_mean=0.09925717115402222`,
`nonselected_write_mass_frac_mean=0.4308384656906128`, matching the task's cited
0.569/0.099/0.431 exactly); `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` (1–60,
206–398, 419–570 read in full); `matrix-thinking/deltanet_rd/run_lm_rd_geo3_sweep.py`
(1–320, 585–610 read in full, incl. exact line-number verification of
`selection_mode_for_verdict` at :165–169/:200/:307, `_refuse_if_no_launch` at :172–183,
`load_gate_verdict`/`_refuse_if_no_launch` calls at :598/:600, `GEO3_N_ITER_BY_K_SEL` at
:84, `_PER_STEP_S_PLACEHOLDER_GEO3` at :103); `STATE.md:495–520` (verified "129.4 GPU-h,
cumulative 162.5/300" present exactly as Rev 2 attributes it); `EXPERIMENT_LOG.md`
(grepped for `rung-2`/`rung-3`/`harvest` — confirmed no rung-2/rung-3 harvest entry
exists yet, only the rung-1 harvest and, in `STATE.md`, the rung-2 launch entry). No GPU
run, no box access, no push performed in producing this document.

---

## Round 3 (bounded) — 2026-07-04, fresh verifier against Rev 3 (commit `8ab089d`)

**Scope, per dispatch.** Bounded verify pass only: (a) is each of this round's 10
findings (NEW-1..8, MINOR-1..2) genuinely closed by Rev 3's *implementation*, not just
its prose; (b) new flaws in Rev 3's own additions; (c) grep-verify every quoted string
against its cited source; (d) the §13 adjudications (Cell 3 never-cut, per-write
magnitude mass-conservation identity). Items this round or round 1 already marked
CLOSED are **not reopened**. Design-only — no GPU, no box, no push; all checks below
are CPU arithmetic/closed-form reproduction or direct file reads against the actual
repo state this session.

### Part A — Closure table (round-2 findings against Rev 3)

| # | Verdict | Basis |
|---|---|---|
| NEW-1 | **CLOSED** | §2 principle 4's REVISED paragraph makes the PARTIAL trigger one rule (`median(shortfall_c) > 0.10` **or** `frac(shortfall_c > 0.10) > 0.25`) applied identically to candidates 1/2/3, candidate 4's hard-snap phase, the comparator, and Cells 2R/4R — the asymmetry (candidate 2 alone had teeth) is gone. The per-chunk dispersion measurement is folded into the existing zero-cost Wave −1 Cell-1 forward pass (§5.3) and pinned into `BANDS_PINNED-TrackB`, so the rule is enforceable, not aspirational. Re-checked the cited arithmetic independently: 32×0.3625=11.60; 11.60/0.4308=26.93≈26.9; pre-renorm mean selected β=(26.9−11.60)/32=15.30/32=0.478; scale factor 0.84/0.478=1.757≈1.76×; headroom 1−0.84=0.16 — all reproduce exactly as quoted. |
| NEW-2 | **CLOSED** | §5.1 now states the cadence explicitly: "resampled independently for every (chunk instance, training step) from an RNG stream keyed to the run seed and step counter — never a fixed function of intra-chunk position," with the rejected fixed-offset-draw alternative renamed and demoted to a named, cut-eligible **diagnostic**, explicitly barred from ever substituting as the control. This closes the specific failure mode NEW-2 named (silent degeneration into a positional schedule) by construction. One residual, sub-MAJOR ambiguity noted below (new-flaw sweep). |
| NEW-3 | **CLOSED** | §5.1 replaces the unoperationalized "matches within seed noise" with a concrete, evaluable three-way rule (disjoint-Cell-2-better / disjoint-2R-equal-or-better / overlap→INCONCLUSIVE), removes the false precedent-borrowing (no longer silently importing KEY_ANCHORING's one-sample power analysis for a two-sample comparison), and registers the low-power acknowledgment numerically. The α figure is independently re-derived below (Part C) and confirmed correct. |
| NEW-4 | **CLOSED** | §4.3's churn ceiling is now `max(Null A, Null B)`: Null A retained (unmasked pilot, unchanged), Null B added (the gated run's own first-10%-of-steps churn, same mechanism/regime). This directly answers both of NEW-4's named mechanisms (STE entrenchment suppressing churn; legitimate sharpening raising it) by adding a same-run, same-mechanism reference, and the residual limitation (Null B overweights init transients; still a coarse screen, not a calibrated bound) is registered honestly rather than papered over. Not a full statistical calibration — the doc doesn't claim it is. |
| NEW-5 | **CLOSED** | §4.3 gains a full "Positional concentration" subsection: TV-distance of the intra-chunk offset-selection marginal from uniform(1/64), measured on the fixed probe batch at 50%/final checkpoints, ceiling = unmasked-pilot `mean_ref+2·s_ref` (same null shape as churn Null A, for the same non-circularity reason), folded into `BANDS_PINNED-TrackB`. The broken cross-reference is fixed: §3.1(iii) now cites "§4.3's positional-concentration statistic" directly instead of the stale "§8 item 1." Grep-confirmed §8 item 1 no longer carries this citation burden (checked below). |
| NEW-6 | **CLOSED** | The comparator's anneal is re-pinned to τ: 1→0 linear over the first 10% of steps, then exactly 0 for the remaining 90% — both arms hard for ≥90% of training, so the shared regime isolates the gradient estimator rather than an annealing curriculum. The residual (first-10% divergence between the arms) is registered in one sentence rather than hidden, and §11 item 2's promotion trigger for the hard-concrete/L0 follow-on is widened to account for it — a real, not cosmetic, response to the round's own critique. Bit-equivalence is kept but explicitly marked non-load-bearing with a numeric fallback bound, exactly matching the round's own disposition of that non-issue. |
| NEW-7 | **CLOSED** | Slice selection is rebuilt from the code's actual mechanism (`n_dup_max` = largest within-chunk identical-4-gram count, threshold ≥8, "comfortably past" the code-documented ≥~6 onset at `lm_pretrain_rd.py:349–353` — verified, see Part D) rather than the text-level proxy NEW-7 attacked. The positive-control floor (≥25 forward calls with ≥6 duplicated rows **within the selected top-K set**, logged via `_topk_idx`) makes the smoke provably non-vacuous: if unmet, the smoke is declared NON-PROBATIVE (not silently "passed") and a forced-selection fallback (reusing the M6 argument) guarantees the regime is exercised. This directly answers "the smoke could pass without ever exercising the failure regime." |
| NEW-8 | **CLOSED, verified independently at the source** | Confirmed via `git show 13eb71c -- matrix-thinking/SCALE_TRANSFER_DESIGN.md`: the correction is real, landed in the cited commit, and the diff is exactly what Rev 3 describes (316→343 on the 190.22 GPU-h base, with an explicit in-source note "caught by TRACKB_REDESIGN_ATTACK_R2.md"). Re-verified `190.22` itself is not invented for this fix: `git show c68a3a3` shows it as a real, pre-existing `PROGRAM_SPENT_GPUH` update (163.22→190.22, +27.00 for the real `--wave 1ext` launch) recorded hours before the Rev-3/source-fix commit — i.e., Rev 3 is citing an actual prior commit's number, not backfilling one to make the arithmetic close. 190.22+152.5≈343 and 190.22+76.2≈266.4 both check out exactly. Headroom 300−266.4=33.6 reproduces exactly. |
| MINOR-1 | **CLOSED** | §5.3 now states both readings (absolute-proximity vs. relative-excess-over-uniform) explicitly, explains why the pin survives on reasons (i)/(iii)/(iv) rather than (ii), and upgrades K_sel=16 from "not silently dropped" to "the FIRST registered follow-on axis" — matches the round's own suggested fix almost verbatim. |
| MINOR-2 | **CLOSED** | §5.1's Cell-4 composition-rule paragraph now specifies EOT/padding validity threading under forcing exactly as the round's own fix suggested: `content_mask` gathered at the forced indices (verified buildable — `content_mask`/`content_c` is already computed and available at `lm_pretrain_rd.py:289`, before the internal `topk` call the forced path bypasses), feeding the existing degenerate-episode machinery unchanged, plus a registered negative-case build smoke (forced EOT index → invalid → no-op scatter). |

**All 10 round-2 findings are genuinely closed by real mechanism** — pinned numeric
rules, registered derivation procedures, or upstream source fixes — not by wording
alone. None of the closures merely relabels a disclosure as a control (the specific
failure pattern this whole three-round history keeps hunting for); each adds an actual
gate, formula, or fallback with teeth.

### Part B — New-flaw sweep on Rev 3's own additions

No new FATAL or MAJOR found. Two sub-MAJOR observations, both worth a one-line note
for the build phase but neither blocking:

1. **Cell 2R's RNG cadence is specified for training but not explicitly for
   measurement-time re-invocation.** §5.1 pins resampling to "every (chunk instance,
   **training** step)" keyed to "the run seed and step counter." The §5.2/§5.3
   Gram-deviation instrument, however, is a separate forward-hook probe pass (§6.1
   Wave 3: "forward-hook probes, no backward pass") run against checkpoints, not
   necessarily inside the training loop's own step counter. If the eval-time probe
   pass re-invokes Cell 2R's random-selection mechanism with a *fixed* or
   *checkpoint-identical* seed/step value across repeated measurement passes (rather
   than continuing to draw fresh per-episode randomness), the measured Gram deviation
   would be computed over one frozen random subset per checkpoint — not literally the
   "fixed positional schedule" failure NEW-2 named (the subset would still be content-
   position-random, not periodic), but a milder version of the same concern: a single
   arbitrarily-frozen random draw at measurement time adds sampling noise to Cell 2R's
   own reported number that isn't present in Cell 2's (whose selection is a
   deterministic function of the trained β at that checkpoint). This is very likely a
   non-issue in practice — the natural implementation is to let the same RNG stream
   that drives training continue into eval forward calls, which would already satisfy
   NEW-2's intent — but the design text does not say so explicitly, and given this
   document's own three-round history of exactly this class of cadence ambiguity
   causing real MAJORs, it is worth one explicit sentence at build time: *"the eval-time
   forward-hook probe draws its Cell 2R subset from the same continuing RNG stream as
   training, never a fixed draw pinned to the checkpoint."* Rated sub-MAJOR, not MAJOR:
   unlike NEW-2's original finding, no plausible reading of the current text
   re-introduces a *positional* artifact, and the (unlikely) failure mode this would
   produce is added noise, not a systematic bias in either cell's favor.
2. **The "candidate 3" / "Cell 3" naming collision is a documentation-risk, not a
   design flaw, once traced carefully.** §2 principle 4's REVISED paragraph says the
   symmetric shortfall rule applies to "candidates 1/2/3" — on first read this could be
   misread as including the 2×2 factorial's **Cell 3** (the geo3-only reference arm,
   which is deliberately *not* renormalized, since it exists specifically to replicate
   the original, uncontrolled construction whose gate already measured the 43.1%
   non-selected write-mass). Traced against §3's own numbering, "candidate 3" here
   unambiguously means §3.3's fixed periodic write-schedule *mechanism*, which is a
   masking candidate and does need renormalization — the sentence is correct as
   written. Confirmed no instance in the document where this collision produces an
   actually wrong instruction (checked every "candidate 3" and "Cell 3" occurrence);
   flagging only because a build-time skim risks misapplying renormalization to Cell 3.
   Not required to fix before build; worth a one-line disambiguating footnote if this
   document gets a Rev 4 for any other reason.

Everything else scrutinized under the task's explicit call-outs (the §2 principle 4
shortfall rule's numeric thresholds, the dual churn null's calibration honesty, the
TV-distance positional check's specification completeness, the τ→0-by-10% comparator's
residual-confound disclosure, the duplicate-key slice's mechanism grounding and
positive-control floor, the corrected budget quotes) held up under independent
re-derivation — see Parts A, C, D.

### Part C — α-math check: n=3-vs-n=3 complete range separation under exchangeability

Rev 3 claims (§5.1): *"under the exchangeability null, complete range separation at
n=3-per-arm has probability 2/C(6,3) = 0.10 per corpus."* Independently re-derived
from first principles, not taken on faith:

Under the null that group membership carries no information (the two arms are
exchangeable), condition on the 6 realized values and treat every assignment of 3-of-6
to "arm A" (the remaining 3 to "arm B") as equally likely — `C(6,3) = 6!/(3!·3!) =
720/36 = 20` equally likely assignments, assuming no ties. **Complete range
separation** (every value in one arm strictly below every value in the other) occurs
in exactly two of these 20 assignments: arm A gets the 3 smallest values (arm A
entirely below arm B), or arm A gets the 3 largest values (arm A entirely above arm
B) — there is exactly one assignment realizing each direction. So
`P(complete separation) = 2/20 = 0.10`, exactly as claimed. This is the standard
two-sample rank/Mann–Whitney extreme-statistic result for `n1=n2=3` (the `U=0` and
`U=9` tail probabilities of the exact null distribution, each `1/C(6,3)`), independently
reproducible by brute-force enumeration of all 20 assignments (verified by hand: only
`{1,2,3}` and `{4,5,6}` as arm-A's rank-set give full separation among all
`C(6,3)=20` 3-subsets of `{1..6}`). **The math is correct**, and the framing around it
is honest: this is a coarse, low-power screen (correctly labeled as such), not a
powered hypothesis test, and INCONCLUSIVE is registered as the *expected*, non-failure
outcome for small true effects — which is the right characterization of a test whose
own best-case Type-II error rate at small effect sizes is necessarily high at n=3.

### Part D — Quote verification (this round's new/re-quoted strings)

Every quoted string below was grepped against the live file this session (not taken
from either Rev 3's or this round's own prose at face value, per the standing
citation-integrity suspicion this document's own history established):

| Quote (as it appears in Rev 3) | Cited source | Result |
|---|---|---|
| "≈129 GPU-h for the wave (cumulative ≈163 of the §7 300 GPU-h ceiling)" | `SCALE_TRANSFER_DESIGN.md` §5.6 item 1 | **VERIFIED**, byte-for-byte, `SCALE_TRANSFER_DESIGN.md:751–752` |
| "cumulative ≈343 on the 190.22 GPU-h committed base" | `SCALE_TRANSFER_DESIGN.md` §5.6 (Rev 2.2, post-fix) | **VERIFIED**, `SCALE_TRANSFER_DESIGN.md:795` |
| "cumulative ≈266/300 — passes every gate" | same | **VERIFIED**, `SCALE_TRANSFER_DESIGN.md:800` |
| "registered BEFORE any rung-3 training data exists" | same, amendment header | **VERIFIED**, `SCALE_TRANSFER_DESIGN.md:790` |
| "caught by TRACKB_REDESIGN_ATTACK_R2.md" (the correction note's own self-citation) | `SCALE_TRANSFER_DESIGN.md:797` | **VERIFIED** — present in the live source, confirming the upstream fix genuinely names this attack chain, not paraphrased |
| `gate_verdict: "no_launch_redesign"` | `wave_neg1_gate.json` | **VERIFIED**, `deltanet_rd/results/lm_rd_geo3/wave_neg1_gate.json:53` (`"gate_verdict": "no_launch_redesign"`) |
| "no_launch_redesign must be refused by the CALLER before this function is ever reached" | `run_lm_rd_geo3_sweep.py::selection_mode_for_verdict` | **VERIFIED**, `run_lm_rd_geo3_sweep.py:167–168` (frozen from round 2, re-checked this round, unchanged) |
| `_refuse_if_no_launch` mechanics (`:172–183`, `sys.exit(3)`) called at `:600` after `load_gate_verdict` at `:598` | `run_lm_rd_geo3_sweep.py` | **VERIFIED** — read the function and `main()` directly this session; line numbers and control flow match exactly |
| NaN mechanism description ("a FULLY-VALID episode whose selected keys contain >=~6 exactly-duplicated rows... flagged for Wave 1 monitoring, not assumed away") | `lm_pretrain_rd.py:349–353` | **VERIFIED**, read directly this session; the "~6" threshold and "flagged for Wave 1 monitoring" phrase are both present verbatim (line-wrapped across 353–354) |
| `valid_sel = topk_val > (neg_inf / 2)` at `:301` | `lm_pretrain_rd.py` | **VERIFIED** present at that exact line (Rev 3's backtick rendering drops the parens/whitespace — a formatting simplification, not a misquote, consistent with round 2's own identical rendering of this line) |
| `conv_size=4` / `k_conv1d` short-conv mechanism underlying NEW-7's "identical token 4-grams ⇒ identical key rows" claim | `lm_pretrain_rd.py` | **VERIFIED** — `k_conv1d = ShortConvolution(..., kernel_size=conv_size, ...)` with `conv_size: int = 4` confirmed at the constructor; the 4-gram/kernel-4 correspondence is accurate |
| KEY_ANCHORING precedent characterization ("one-sample-vs-fixed-band," `engaged_K = mean_ref + 2·s_ref` at n=3, ~71%/~50% RSE) | `KEY_ANCHORING_DESIGN.md` §3.6 | **VERIFIED**, `KEY_ANCHORING_DESIGN.md:965–977` — confirms this is indeed a one-arm-vs-band derivation, supporting NEW-3's original critique that Rev 2 borrowed a headcount without its power justification |
| `190.22` GPU-h committed-base figure, independently traced to its origin (not re-derived by Rev 3 or this attack chain) | `git show c68a3a3` (`run_lm_rd_trackc_sweep.py`) | **VERIFIED** — a real, dated, pre-existing commit (`163.22→190.22`, `+27.00` for the actual `--wave 1ext` launch), landed before the source-fix commit `13eb71c` |

**No fabricated or misattributed quote found in Rev 3's own additions.** Every string
presented in quotation marks (or backtick code-literal form) is grep-verifiable in its
cited source, continuing the clean record round 2 established for Rev 2 — the citation
discipline that Rev 1's fabricated quote (M4) originally violated has held for two
consecutive revisions now.

### Part E — §13 adjudications

- **Cell 3 promoted to never-cut at one corpus.** Checked §10 and §11 item 3 together:
  §10's never-cut list states "Cell 3, at minimum one corpus × 3 seeds," and cut-order
  item 7 separately allows dropping Cell 3's **second** corpus under budget pressure
  while explicitly noting "the cell itself is never-cut, above." These two statements
  are consistent, not contradictory — the never-cut guarantee covers the minimum viable
  Cell 3 (one corpus), and only the second-corpus enhancement is cut-eligible. Matches
  the round's own adjudication text exactly ("promoting Cell 3 to never-cut alongside
  Cell 2R, not leaving it as an open question").
- **Per-write magnitude inseparability via mass-conservation.** Checked the identity
  directly: total per-chunk write mass `M = count × mean` is an algebraic tautology: if
  `count` (= `K_sel`, pinned) and `M` (= `B_pinned`, pinned) are both held fixed, then
  `mean = M / count` is *determined*, not free — there is no way to vary per-write
  magnitude independently of count at fixed total mass and fixed count. The identity
  holds exactly; this is correctly not treated as a control gap, and Cell 2R (which
  varies *which* positions are selected, not `count` or `M`) is the right orthogonal
  cell for the question that remains open (targeted vs. random selection at matched
  mass/count).

---

**Round 3 (bounded) verdict: CLEARED-FOR-BUILD.**

- NEW-1..8, MINOR-1, MINOR-2 — all **CLOSED** by real mechanism (Part A); none reopened.
- No new FATAL/MAJOR in Rev 3's own additions. Two sub-MAJOR notes for the build phase,
  neither blocking: (1) Cell 2R's RNG cadence should be stated explicitly to also cover
  eval-time forward-hook probe passes, not just the training loop; (2) the
  "candidate 3"/"Cell 3" naming collision in §2 principle 4 is correct as written but
  worth a disambiguating footnote if the document is touched again.
- α ≈ 0.10 math independently re-derived and confirmed exact (`2/C(6,3) = 0.10`, Part C).
- Every quoted string checked this round (budget quotes, code citations, the
  KEY_ANCHORING precedent characterization, the 190.22 GPU-h base's own provenance)
  is grep-verified against its live source — no fabrication, no misattribution
  (Part D).
- §13's two named adjudications both check out exactly as stated (Part E).

No GPU run, no box access, no push performed in producing this section.
