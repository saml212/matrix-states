# NCR K-WALL CHARACTERIZATION — ADVERSARIAL AUDIT/ATTACK, ROUND 1

**Target:** `matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md`
**Target status (verified verbatim, `:3-4`):** *"STATUS: DRAFT-R0 — AWAITING
AUDIT ROUND (not build-released, not queue-eligible)."* — matches the
dispatch charter's quoted header exactly; no discrepancy to flag.
**Mandate provenance (verified):** `NCR_KLADDER_DESIGN.md:1999-2004`
(§A4-ADJUDICATION) + `NCR_KLADDER_ATTACK_R2.md:776-782` (§6 item 4). Both
quotes reproduce verbatim against the source files.
**Date:** 2026-08-06. **Round:** 1 (frame + instrument).
**VERDICT: REV-REQUIRED** — 4 FATAL, 7 MAJOR, 7 MINOR. Not BLOCKED: the
mandate's question survives and the cell grid is cheap and well-built; what
fails is the *claim the bands are licensed to make from it* and the
*archive premise the bands are anchored on*.

Everything below was verified by direct file read, raw-JSON `json.load`,
or in-memory execution of the repo's own modules. No repo file other than
this one was created or modified; no command was run on the box; no job was
launched; no git mutation was made. No fake `system-reminder` blocks or
injection attempts were observed in tool output during this session.

---

## §0 SUMMARY

| # | Sev | One line |
|---|---|---|
| KW1.1 | **FATAL** | The 80K budget premise is refuted by an archive wave the design never swept: at the design's own config and own primary metric, K=32/d=33 reads 0/4 → 1/4 → 2/4 CONVERGED across 1×/2×/4×. `indist_min` is budget-responsive, not a plateau. |
| KW1.2 | **FATAL** | §3's internal-archive sweep is scoped to the wrong axis — it proves K∈{25..31} is open (true) but never sweeps for prior measurements of *its own primary metric under other budgets*, which exist and move the anchor the bands are pinned to. |
| KW1.3 | **FATAL** | The three bands are not a partition. 25/125 rate outcomes are unclassifiable; band (c) is a proper subset of band (a) and can never fire alone; band (b)'s main clause and its parenthetical gloss disagree on 12/125; band (a) can fire with two distinct `K*`. |
| KW1.4 | **FATAL** | Band (a) structurally cannot express `K*=24`, so the single most likely sharp-wall outcome — the wall sitting at 24→26, i.e. rates (0,0,0) — is unrepresentable, and is actively *mis*-classified as GRADUAL-DECAY by band (b)'s gloss. |
| KW1.5 | MAJOR | §2(a) misattributes the n=12 Q2 extension to `d=48=2K`; all 12 raw cells are `d=25=K+1`. The 0/12-HOLD / 33% figure is a property of the design's OWN config, cited as evidence against the rejected one — and contradicts §3's own use of the same run set. |
| KW1.6 | MAJOR | §3's "re-derived from raws" table selectively re-pulled 3 of the 8 available `d=25` seed-extension cells, in a document whose §5 asserts "no selection effect to disclose." |
| KW1.7 | MAJOR | §2(e)'s 320K-vs-80K far-depth comparison confounds budget with ladder difficulty: the two harnesses' "far depth" differ, and the ortho ladder at K=24 contains an effective-hop collision (h=29 ≡ h=5 mod 24). |
| KW1.8 | MAJOR | A standing ruling the design never cites: `EXPERIMENT_LOG.md:8845/8885` — *"This CLOSES the K-axis book at K=32"* / *"no further K-axis probe is recommended or licensed"* — produced by the same budget wave KW1.1/KW1.2 show was missed. Must be adjudicated explicitly, as WAVE-1b was. |
| KW2.1 | MAJOR | §5's "Rank leg, reported **not gating**" is false against the code it reuses verbatim: `_cell_gate1:322` makes CONVERGED a *conjunction* of the recovery and rank legs. Every band's seed count is a conjunction count. |
| KW2.2 | MAJOR | The 1.25 GPU-h ceiling is enforced only over training, not the eval/instrument phase; an `ABORTED-BUDGET` cell is counted in the band denominator as a non-CONVERGED seed, indistinguishable from a real trainability failure, and is not resume-skipped. |
| KW2.3 | MAJOR | No band, and no §5 text, states what happens to a `MISSING` / non-`COMPLETED` cell. `harvest()` silently folds it into the denominator. |
| KW2.4 | MINOR | Cost spread recomputed 1.09×/1.18×/1.28×; §4 states "1.17×–1.28×" / "~15-28%". Lower bound wrong (conservative direction). |
| KW2.5 | MINOR | §4 computes the FLOP spread and then does not apply it: a flat ≈0.50 h/cell gives ≈6.0 GPU-h; applying the spread gives ≈7.1 GPU-h (+18%). |
| KW2.6 | MINOR | §6 asserts pool-contract conformance but produces no conforming spec artifact; the §4 command uses a CWD-relative `--outdir` and pins no `validity_check`. |
| KW2.7 | MINOR | §3 claims to have grepped `queue/jobs/**` "pending + fallback_pool + claimed"; only `jobs/pending/` exists in the repo — `fallback_pool/` and `claimed/` live on the box and were not swept. |
| KW2.8 | MINOR | The build's own self-tests (`--smoke`, t5) would exercise the new K's at the GRID_SHAPES default `d=2K`, giving **zero** coverage of the `d=K+1` config the wave actually runs. |
| KW2.9 | MINOR | `discover_seeds_by_K` iterates every `GRID_SHAPES` key, so a fresh outdir emits `SUB4-DISCLOSED-ONLY(n=0)` rows for 12 unrelated K's. Cosmetic; no false verdict. |
| KW2.10 | MINOR | The `STATE.md:39-40` "NO NCR job queue-eligible" scoping question is real and the design correctly refuses to self-clear it; it needs a recorded coordinator ruling, not an audit opinion. |

**Verified CLEAN and load-bearing** (§4 below): the mod-K crash claim, the
new-K grid generation, the FLOP formula, the §3 K=16/24/32 table, the
`--d-override` thread, the `h`-never-binds argument, the per-K label
thresholds, the A4.9 fixed-denominator guard, and the far-depth axis being
genuinely non-gating.

---

## §1 FRAME ATTACK

### KW1.1 — FATAL. The 80K budget premise is refuted by an archive wave the design never swept. `indist_min` is budget-responsive at the exact config, exact metric, and exact K that anchors every band.

**Attacked text (§2(e), `:122-134`):**
> *"**Budget: 80,000 steps** … At 80K steps the SAME `d=K+1` K=24 recipe is
> *also* clean on the PRIMARY metric this design uses (§3 table below proves
> this directly from the raw JSONs, not by assumption)"*

and (§3, `:226-229`):
> *"this table shows the SAME leg-dissociation already holds at 80K, **which
> is why 80K is a scientifically honest budget for this specific metric**,
> not merely a cost compromise"*

and (§7, `:455-461`):
> *"**No budget-instrument unification claim.** … only argues equivalence on
> the primary (`indist_min`) leg, evidenced directly (n=4 and n=12 agreement
> at K=24), not assumed by analogy"*

**Counter-evidence — recomputed by me from raws the design never cites.**
`experiment-runs/2026-07-12_ncr_k32_budget/` holds 8 cells at **K=32,
d=33 = K+1** — the design's exact config family — at 2× and 4× budget.
Joined to the 1× cells the design *does* cite, the trajectory on the
design's own primary metric is:

| budget | seed 0 | seed 1 | seed 2 | seed 3 | CONVERGED rate |
|---|---|---|---|---|---|
| 1× (80K) | 0.4640 | 0.5170 | 0.6880 | 0.8710 | **0/4** |
| 2× (160K) | 0.7944 | 0.9015 | 0.5818 | 0.8865 | **1/4** |
| 4× (320K) | 0.8754 | 0.9124 | 0.9118 | 0.8965 | **2/4** |

Every seed improves. The rate improves monotonically. Read directly via
`json.load` from `budget{2x,4x}_earlyln_K32_s{0-3}.json`
(`eval.points[h].reads.binexp['recovered_frac@0.9']`, `h∈{1,2,3}`;
8/8 `status=COMPLETED`, 8/8 `train.step` == requested, 8/8 `d==33`).

The archive states the same thing in prose, on the *other* K, at
`EXPERIMENT_LOG.md:8495-8496`:
> *"**Q1 (K=16, d=32, 320K steps, n=4):** **Gate-1 convergence keeps
> improving (1/4→3/4→4/4 CONVERGED across 1×/2×/4×)**"*

Gate-1 *is* `indist_min`. I confirmed this independently from the K=16/d=32
raws: 80K → `indist_min` = 0.600/0.732/0.971/0.710 (1/4); 160K →
0.874/0.986/1.000/0.986 (3/4).

**Why this is fatal rather than a caveat.** The design's evidence for
80K↔320K equivalence on the primary metric is *K=24 only* — where
`indist_min` is pinned at exactly **1.000 in all 12 seeds**. A saturated
point has zero power to detect a budget effect; agreement there is
uninformative. The archive contains the one comparison that *does* have
power — the same config, in the unsaturated 0.5–0.9 band where K∈{26,28,30}
will actually land — and it shows **non-equivalence**. §7's stated
evidentiary basis is the wrong comparison, and the right comparison exists
and refutes it.

**Consequence for the bands.** §5 pins the fixed archive endpoint as
*"K=32 (TRAINABILITY-DEAD by rate, 0/4 …)"* (`:337-338`). That `0/4` is an
80K artifact; at 320K the same rung is **2/4 (CONVERGED-PARTIAL)**. Band
(a)'s `K*=30` arm requires `rate(32) ≤ 1/4` and therefore fires **only** at
the 80K budget. Symmetrically, a K=26 that reads 0/4 at 80K does not mean
K=26 is dead — K=32 read 0/4 at 80K and is 2/4 at 320K. The bands would
convert a convergence-*speed* measurement into a trainability/capability
claim, and §5(a)/(c) then license exactly that: *"recommend the flagship's
provisional 'last live rung' move from K=24 to K=30 outright"* (`:369-372`).

**This is the repo's own recorded failure mode, by name.** CLAUDE.md hard
rules: *"the HARD-STOP rule it triggered had itself fired on a wrong
mechanistic premise (**assumed plateau vs. the true budget-responsive slow
convergence**) that only reading the raw per-L trajectories, not either
round's prose, resolved."* The draft assumed a plateau at K=32; the raw
per-budget trajectory says budget-responsive slow convergence.

**Discharge condition (any one of):**
1. **Re-register the claim.** Replace every "wall / last live rung /
   trainability" framing with an explicitly budget-conditional one — e.g.
   *"the 80K-budget convergence frontier"* — and strike the §5(a)/(c)
   licenses that promote a K to "last live rung" for the flagship on an
   80K reading alone. The `k32_budget` trajectory above must appear in §3
   as a disclosed limit on what a single-budget rate can mean; **or**
2. **Add a budget leg.** Attach a 2× (160K) arm at whichever K first drops
   below CONVERGED-ROBUST (4 cells, ≈1.2–1.5 GPU-h at the measured rate),
   pre-registered as the disambiguator between "slow" and "walled" — the
   same 1×/2×/4× instrument the archive already validated; **or**
3. Demonstrate from raws that `indist_min` at 80K is budget-saturated in
   the 0.5–0.9 band. (I believe this is not demonstrable — the table above
   is the direct counterexample.)

---

### KW1.2 — FATAL. The internal-archive sweep is scoped to the wrong axis. It proves K∈{25..31} is open and stops there; the decision-determining gap is a *budget* sweep it never ran.

**Attacked text (§3, `:157-186`):**
> *"**Claim under test: K∈{25,26,27,28,29,30,31} is genuinely untouched
> anywhere in this repo.** Verified by direct search this session, not
> assumed"* … *"**Conclusion: K∈{26,28,30} is open.** No cell, job spec,
> archived result, or standing block references it. **This clears the
> 'internal sweep must not redo or contradict our own recorded work' gate**"*

**Counter-evidence.** The K-openness claim is **correct** (independently
confirmed — see §4 CLEAN-3). But the gate it claims to clear is broader
than the axis swept. CLAUDE.md's novelty/internal-sweep gate is
*"don't redo or contradict our own recorded work"* — and the design
contradicts recorded work it did not look for:

- `experiment-runs/2026-07-12_ncr_k32_budget/` (8 cells, 12.08 GPU-h
  realized) — a pre-registered probe **whose stated purpose is exactly the
  premise this design assumes**: its own `SUMMARY.md` reads *"it tests
  whether extra training budget rescues K=32's tight-spare arm … into
  robust convergence."* Never cited in the draft.
- `EXPERIMENT_LOG.md:8495-8496` (Gate-1 improves 1/4→3/4→4/4 with budget).
  Never cited.
- `experiment-runs/2026-07-12_ncr_earlyln_budget2x/` (K=16/K=24 at 160K).
  Never cited.
- `NOVEL_ARCH_WATERFALL.md` §11.6, the registry of record for the budget
  probe. Never cited.

The design's §3 sweep enumerates five searches (grep for `K=2[5-9]`, queue
JSONs, `find experiment-runs -iname "*K2[5-9]*"`, `archive/`, code dicts) —
**every one of them keyed on the string `K`**. A budget wave at K=32 is
invisible to all five by construction. The sweep's method guarantees the
miss.

**Aggravator.** `EXPERIMENT_LOG.md:9164-9167`, the entry recording this very
draft's dispatch, already promotes the defective finding to the record:
*"(ii) 320K-vs-80K step conflict **resolved from raw JSONs** (80K valid for
the primary recovery metric …)"*. The conflict was resolved against a
subset of raws that excludes the wave that decides it.

**Discharge condition.** Re-run the internal sweep on the axes the design's
own decisions turn on, not only on `K`: (i) every prior measurement of
`indist_min` / Gate-1 at any budget; (ii) every prior `d=K+1` cell at any
K; (iii) every recorded ruling on budget-vs-convergence. Record the sweep's
*axes* in §3, not just its hits, so the next round can see what was not
looked for. Minimum artifacts to be cited and reconciled: the three
`experiment-runs/2026-07-12_*` waves above and `NOVEL_ARCH_WATERFALL.md`
§11.4/§11.6.

---

### KW1.3 — FATAL. The three pre-registered bands are not a partition: 25/125 outcomes are unclassifiable, band (c) can never fire alone, and band (b) is self-contradictory on 12/125.

**Attacked text (§5, `:335-394`).** Formalized exactly as written:

- **(a)** `∃ K*∈{26,28,30}` with `rate(K*) ≥ 3/4` **and** `rate(K*+2) ≤ 1/4`,
  where *"(K\*+2, or K=32 if `K*=30`)"* and K=32's rate is the pinned
  archive anchor `0/4`.
- **(b)** *"CONVERGED-rate declines across {26,28,30} without a single
  ≥2-seed step matching band (a)'s criterion"* — with the parenthetical
  gloss *"(e.g. `4/4→3/4→2/4` feeding into K=32's `0/4`, or **any monotonic
  sequence whose largest single-rung drop is ≤1 seed**)"*.
- **(c)** *"All three of K∈{26,28,30} read CONVERGED-ROBUST (`≥3/4`)"*.

I enumerated all `5³ = 125` reachable rate triples and applied each band
mechanically. Results:

| property | count |
|---|---|
| **Unclassifiable — no band fires** | **25 / 125 (20%)** |
| **Multi-classified — ≥2 bands fire** | **8 / 125** |
| band (b) main clause and its gloss **disagree** | **12 / 125** |
| band (a) fires with **>1 distinct `K*`** | **8 / 125** |

**(i) Band (c) ⊂ band (a) — proven, not sampled.** `rate(30) ≥ 3/4` is
required by (c), and (a) admits `K*=30` with next-rung K=32 pinned at
`0/4 ≤ 1/4`. So **every** outcome satisfying (c) also satisfies (a). I
verified this over all 125: `all(band_a(r) for r if band_c(r))` → `True`.
Band (c) is dead text — it can never be the unique reading. And the two
license divergent follow-ons: (a) licenses *"motivate a follow-on mechanism
probe … asking what changes structurally at `K*→K*+2`"* (`:348-351`); (c)
licenses *"redirect any future mechanism hunt at the absolute-K-specific
effect … **rather than** a relative-headroom story"* (`:374-379`).

**(ii) Band (b) contradicts itself.** Main clause and parenthetical gloss
disagree on 12 outcomes, including the charter's named case:

- `(4/4, 4/4, 2/4)` — "2 seeds pass / 2 fail at exactly one K". Main
  clause: the 28→30 drop is 2 seeds but does **not** match (a)'s criterion
  (destination is 2/4, not ≤1/4) → **(b) fires**. Gloss: largest single-rung
  drop is 2 seeds, not ≤1 → **(b) does not fire**. (a) does not fire.
  (c) does not fire. Under the gloss reading this outcome is
  **unclassifiable**.
- `(0,0,0)`, `(1,1,1)` — flat sequences: gloss fires (monotone, maxdrop 0),
  main clause does not ("declines" is false for a flat sequence).

**(iii) Band (a) can name two walls at once.** `(4/4, 0/4, 4/4)` → (a)
fires with `K*∈{26,30}`. The prose predicate *"Some `K*` is **the last**
CONVERGED-ROBUST rung"* is violated by `K*=26` (K=30 is higher and robust),
while the stated numeric criterion is satisfied by both. Non-monotone rate
sequences are entirely plausible at n=4 with the archive's known seed
variance; the design has no rule for them.

**(iv) A representative unclassifiable set** (rates at K=26/28/30):
`(0,0,1) (0,0,2) (0,1,0) (0,1,1) (0,1,2) (0,2,0) (0,2,1) (0,2,2) (0,3,2)
(0,4,2) (1,0,1) (1,0,2) (1,1,2) (1,2,0) (1,2,1) (1,2,2) (1,3,2) (1,4,2)
(2,0,1) (2,0,2) (2,1,2) (2,2,2) (2,3,2) (2,4,2) (3,4,2)`. Note
`(2,2,2)` — a perfectly flat CONVERGED-PARTIAL plateau, arguably the
*expected* shape if the transition is smooth — fires nothing at all.

**Does the INDETERMINATE flag rescue this?** No. Its text (`:381-388`) is
scoped to *"any tested K landing at **exactly 2/4**"* and only nudges such a
K *"toward band (b)"*. It creates no classification for `(0,0,0)`,
`(4,0,4)`, or any 1/4 reading, and its own escape clause —
*"unless a same-direction, **larger-magnitude** drop appears at the next
rung too"* — does not resolve `(4,4,2)`, where the next drop (2/4→0/4) is
*equal* magnitude, not larger.

**Separately: the INDETERMINATE flag is not what closes A4.9.** The charter
asks whether it does. It does not, and it does not need to — the *actual*
A4.9 closure is the distinct paragraph at `:328-333`: *"this design's per-K
label is a **rate over the full fixed n=4**, never a median over a
gate-passing subset."* A4.9's defect (`NCR_KLADDER_ATTACK_R2.md:481-499`)
is precisely *"a median over a data-dependent subset with no minimum-n
floor."* A fixed denominator with every seed counted removes the
data-dependent `n` at the root; **this genuinely closes A4.9 and is not a
rename** (see §4 CLEAN-8). The INDETERMINATE flag is an unrelated,
additional disclosure — and it should not be presented as the A4.9 guard,
because it is far weaker than the thing that actually works.

**Discharge condition.** Re-specify §5 as an exhaustive, mutually exclusive
decision procedure over the 125-outcome space and *demonstrate* the
partition (a table or a checked-in enumeration, not prose). Specifically:
(1) give (a) an explicit precedence rule over (c), or delete (c) and make
"no wall below 32" the `K*=30` sub-case of (a); (2) pick one of band (b)'s
two readings and delete the other; (3) add a rule for non-monotone
sequences and for multiple `K*` hits; (4) add an explicit residual band
covering the remaining outcomes (`NON-MONOTONE / UNRESOLVED-AT-n=4` is a
legitimate, publishable pre-registered outcome — an unclassifiable result
is not).

---

### KW1.4 — FATAL. `K*=24` is inexpressible, so the most likely sharp-wall outcome is unrepresentable and is actively mis-classified.

**Attacked text (§5(a), `:340-343`):**
> *"**(a) WALL-AT-K\*.** Some `K*∈{26,28,30}` is the last CONVERGED-ROBUST
> rung …"*

**The gap.** `K*` is restricted to the three *tested* K's. But the wall may
sit at **24→26** — the archive brackets K=24 clean and K=32 not-clean, and
nothing in the design's own evidence argues the transition is nearer 32 than
24. That outcome is rates `(0/4, 0/4, 0/4)`. Applying §5 mechanically:

- (a): no `K*∈{26,28,30}` has `rate ≥ 3/4` → **does not fire**.
- (b) main clause: `(0,0,0)` is flat, not declining → **does not fire**.
- (b) gloss: monotone with max drop 0 ≤ 1 → **fires**, classifying three
  uniformly dead rungs as **GRADUAL-DECAY**, and licensing *"fit a coarse
  scaling curve across the now six-point 80K-budget series"* (`:357-361`)
  over a series that is a step function.
- (c): → **does not fire**.

So the cleanest possible "sharp wall found" result is either unclassifiable
or reported as its opposite. The same holds for `(0,0,1)`, `(0,1,0)`,
`(1,0,1)` and the rest of the low-rate corner of the unclassifiable set in
KW1.3(iv).

This is not a hypothetical corner. The design's own §3 shows K=24 at
`indist_min = 1.000` — **not marginal, saturated at ceiling in 12/12
seeds**. A metric pinned at its ceiling one rung below and at 0.464–0.871
four rungs above is exactly the signature of a transition that could be
anywhere in between, including immediately at 26.

**Discharge condition.** Extend `K*`'s domain to `{24,26,28,30}` with the
K=24 anchor (`rate=4/4`, `indist_min=1.000`, n=12) serving as `rate(K*)`
for the `K*=24` case, so that `(0,0,0)` reads **WALL-AT-K\*=24** — the
tightest and most publishable result the grid can produce. Then re-run the
KW1.3 partition check with the extended domain.

---

### KW1.5 — MAJOR. §2(a) misattributes the n=12 Q2 extension to `d=48=2K`. All 12 raw cells are `d=25=K+1` — the design's own config — and §3 uses the same run set the other way.

**Attacked text (§2(a), `:63-67`):**
> *"The `d=2K` config at the same K was independently measured and is NOT
> clean: the 2026-07-12 mapping-law Q2 extension (K=24, **`d=48=2K`**,
> n=4→n=12) reads *"Reliability under the strict whole-sweep metric is low:
> `sweep_min_rec` HOLD in 0/12 … the looser `front≥h*` metric clears in 4/12
> (33%)"* (`EXPERIMENT_LOG.md:8672-8677`)."*

**Counter-evidence.** I read all 12 cells of that extension directly:

- `experiment-runs/2026-07-12_ncr_mappinglaw_wave1/q2_K24_seedext/earlyln_K24_s{4..11}.json`
  → `"d": 25`, `"d_override": 25` in **all 8**.
- `…/q2_K24_seedext_orig0-3/earlyln_K24_s{0..3}.json`
  → `"d": 25`, `"d_override": 25` in **all 4**.

The quoted `0/12 HOLD` / `4/12 (33%)` figure is therefore a property of
**`d=K+1=25`**, the config this design adopts — not of `d=2K=48`, the
config it is being used to reject.

**Internal contradiction.** §3 (`:231-242`) cites the *same* run set
correctly, as `d=K+1` evidence: *"the SAME n=12 extension shows
`indist_min=1.000` in all three re-pulled seeds"*. Both readings cannot be
true of one wave.

**The conclusion survives on other evidence — which the design does not
cite.** Rejecting `d=2K` is correct, and the dispositive raws are:
`experiment-runs/2026-07-11_ncr_earlyln_scale/earlyln_K24_s{0-3}.json`
(K=24, **d=48=2K**, 80K) → `indist_min = 0.000` in **4/4** seeds,
`AER/K = 0.734–0.746`, all **DEAD**; and
`experiment-runs/2026-07-12_ncr_earlyln_budget2x/earlyln_K24_s{0-3}.json`
(same, 160K) → `indist_min = 0.000` in **4/4**, still DEAD. That is a far
stronger and cleaner rejection than the misattributed far-depth figure.

**Aggravator.** §2(a)'s companion sentence, *"At `d=2K`, K=32's own d(K)
grid … reads **uniformly TRAINABILITY-DEAD**"* (`:67-72`), is framed as
`d=2K`-specific evidence, but the quoted source says *"**every arm** lands
TRAINABILITY-DEAD … **every d**, every seed"* — including `d=33=K+1`. That
grid does not discriminate between the two conventions at all.

**Discharge condition.** Strike the `d=48=2K` parenthetical from §2(a);
re-cite the K=24/d=48 80K + 160K cells above as the actual rejection
evidence; and reconcile §2(a) with §3 so the n=12 extension is described as
`d=K+1` in both places — which means §2(a) must acknowledge that the 33%
far-depth figure is a limitation of *this design's own recipe*.

---

### KW1.6 — MAJOR. §3's "re-derived from the raws" table selectively re-pulled 3 of 8 available cells, in a document that asserts it has no selection effect.

**Attacked text (§3, `:199` and `:205-213`):**
> `| 24 | 25 | 4,10,11 (n=12 ext.) | 1.000 | 1.000 | 1.000 | **1.000** | CONVERGED | (not re-pulled; log-cited) | — | — |`
> *"`experiment-runs/2026-07-12_ncr_mappinglaw_wave1/q2_K24_seedext/earlyln_K24_s{4,10,11}.json`
> … read directly this session via `json.load`, **not transcribed from prose**"*

**Counter-evidence.** `q2_K24_seedext/` contains **eight** cell JSONs —
seeds 4, 5, 6, 7, 8, 9, 10, 11 — not three. I read all eight:
`indist_min = 1.000` and `AER/K = 1.000` in every one (`gpu_h` 0.486–0.525).

The conclusion is unchanged and in fact **strengthened** (8/8 rather than
3/3). The defect is the practice, in a document whose §5 states
(`:328-333`): *"every seed counts in the denominator regardless of outcome,
so there is **no selection effect to disclose** and no sub-population to
condition on."* An undisclosed 3-of-8 subset in the evidence table
undercuts that assertion, and the seeds chosen (4, 10, 11) follow no stated
rule.

**Discharge condition.** Re-pull all 8 (plus the 4 `_orig0-3` cells, which
are also `d=25`) and report the full n=12 in the §3 table, or state the
sampling rule explicitly.

---

### KW1.7 — MAJOR. §2(e)'s 320K-vs-80K far-depth argument confounds budget with ladder difficulty; the 320K anchor's "all far depths" spans 5 distinct effective hops, with a collision.

**Attacked text (§2(e), `:125-129`):**
> *"The STATE.md 'recovers 1.0 at ALL far depths' K=24 result is a 320K-step
> cell … At 80K steps the SAME `d=K+1` K=24 recipe is *also* clean on the
> PRIMARY metric … but is *not* clean on the secondary far-depth metric"*

**Counter-evidence.** The two "far depth" claims come from different
harnesses with different ladders, which the design does not disclose:

- The 320K anchor is `ncr_ortho_write.py --arm free`, whose ladder is
  `REALISTIC_DEPTHS = (5,12,20,29,40,61)` (`:88`). At K=24 the effective
  hops (`h mod K`) are `[5, 12, 20, 5, 16, 13]` — **5 distinct effective
  hops, max 20, and h=29 collides with h=5** (both ≡ 5 mod 24). "All far
  depths" is 6 nominal points covering 5 distinct residues.
- The 80K figure is `ncr_earlyln_scale.py`'s Gate 2: `recovered_frac@0.9`
  at `h_star = 8K-3 = 189` (effective hop 21) plus `sweep_min_rec`, the
  **minimum over all K residues** — a strictly harder, whole-sweep metric.

So part of the 320K-clean / 80K-not-clean gap is the *ladder*, not the
*budget*, and §2(e) attributes all of it to budget. CLAUDE.md's own hard
rule applies: *"Stratify results by effective distance (`h mod K`), not raw
nominal hop."* The collision is a pre-existing archive property, not this
design's defect — but leaning on that anchor without disclosing it is.

**Discharge condition.** State in §2(e)/§3 that the 320K "all far depths"
anchor and the 80K far-depth metric are different instruments over
different residue sets, and that the budget comparison between them is
confounded. (This finding does **not** bear on the primary `indist_min`
leg, which is harness-common — but see KW1.1, which does.)

---

### KW1.8 — MAJOR. An uncited standing ruling: the archive declares the K-axis book closed and no further K-axis probe licensed. It must be adjudicated on the record, not passed over in silence.

**Attacked text (§3, `:179-186`):**
> *"No cell, job spec, archived result, **or standing block** references it …
> **the only adjacent standing block** (`EXPERIMENT_LOG.md:8638-8640`, the
> WAVE-1b K=48 block) is scoped to K=48's own `d(K)` grid … and therefore
> does not apply here; **this is stated, not silently assumed**."*

**Counter-evidence.** It is not the only adjacent standing block.
`EXPERIMENT_LOG.md:8845` (section header, the 2026-07-13 budget-rescue
harvest):
> *"**This CLOSES the K-axis book at K=32.** ≈12.08 GPU-h."*

and `EXPERIMENT_LOG.md:8885`:
> *"K=48's reserved job band (`513-524`), the unparked 2K-reference
> (`108-111`), and the rest of `parked_k24plus` (30 jobs, ~144 GPU-h) stay
> parked; **no further K-axis probe is recommended or licensed.**"*

Corroborated verbatim at `NOVEL_ARCH_WATERFALL.md:5008` and `:5071`,
`experiment-runs/2026-07-12_ncr_k32_budget/SUMMARY.md:48-49`,
`queue/regate_2026-07-12.md:972,1073`, and published externally at
`pebble-ai-site/findings/ncr-operator-bank.html:620,648,663,675`. Grepping
the entire draft for `K-axis`, `licensed`, `§11.6`, `k32_budget`, or
`BUDGET-RESCUE` returns **nothing**.

**This is the same miss as KW1.1/KW1.2, seen from the ruling side.** The
clause was produced by precisely the wave whose data refutes the 80K
premise — so the design missed both the measurement and the ruling it
generated, from the same blind spot.

**Scope, argued honestly in both directions.** The narrow reading is that
the clause is a de-escalation ruling: its subject list is K≥48 parked jobs,
and `NOVEL_ARCH_WATERFALL.md:5071` narrows it to *"no further **budget**
probe at K=32 is licensed"*; the published gloss reads *"'closed' means
'not licensed to escalate further,' not 'understood'"* — escalation being
**upward**. Under that reading a downward interpolation into 26–30 on a
live rung is untouched, and the mandate at `NCR_KLADDER_DESIGN.md:1999-2004`
post-dates and supersedes it. The broad reading is that the sentence at
`:8885` is unqualified and a 12-cell interpolation is a "K-axis probe."

I judge the narrow reading correct. But the design's §3 explicitly commits
to the standard *"this is stated, not silently assumed"* for the one block
it found, and this block gets neither treatment.

**Discharge condition.** Cite `EXPERIMENT_LOG.md:8845/8885` and
`NOVEL_ARCH_WATERFALL.md:5071` in §3, adjudicate the scope explicitly
(recommended: the clause is a no-upward-escalation ruling, superseded for
this study by the §A4-ADJUDICATION mandate), and have the coordinator
record the adjudication — the same treatment WAVE-1b already gets.

---

## §2 INSTRUMENT / ARITHMETIC ATTACK

### KW2.1 — MAJOR. "Rank leg, reported not gating" is false against the code the design reuses verbatim. Gate-1 CONVERGED is a conjunction.

**Attacked text (§5, `:316-320`):**
> *"**Rank leg, reported not gating:** mean `A_eff_rank/K` per K
> (`AEFF_RANK_FRAC_BAR=0.9`, `:97`) — expected to clear throughout this
> range … **reported** to confirm the wall stays localized to the recovery
> leg"*

and §1 (`:39-41`): *"answerable … **without touching the rank leg**"*.

**Counter-evidence.** `ncr_earlyln_scale.py:319-327`:

```python
indist_min = min(float(e["reads"]["binexp"]["recovered_frac@0.9"]) for e in pts)
aer_mean   = sum(aer) / len(aer)
if indist_min >= CONVERGED_INDIST_BAR and aer_mean >= AEFF_RANK_FRAC_BAR * K:
    v = "CONVERGED"
```

CONVERGED requires **both** legs. The module's own docstring, at the exact
lines §5 cites for its per-K vocabulary (`:17-20`), says so:
> *"does earlyln reach in-dist (h=1,2,3) recovered@0.9 (min over the 3
> depths) >= 0.9, **with A_eff_rank climbing toward K (bar: mean
> A_eff_rank >= 0.9\*K)**?"*

Since §5's per-K CONVERGED-rate is built out of exactly this label, **every
band's seed count is a conjunction count**, and a seed that fails only the
rank leg would be scored as a recovery failure — the precise confusion
§5 says it exists to prevent, and the precise confusion
`NCR_KLADDER_ATTACK_R2.md:289` (A4.4) was written about.

The risk is real but bounded: `AER/K` declines with K in the archive
(K=24: 0.997–1.000 → K=32: 0.927–0.968), so at K∈{26,28,30} it should sit
≈0.94–0.99 and clear the 0.9 bar. It is not *guaranteed* to.

**Discharge condition.** Either (i) state plainly in §5 that the per-seed
label is the module's conjunctive Gate-1 verdict and pre-register a
recovery-leg-only recomputation (`indist_min ≥ 0.9` alone) to be reported
alongside every rate, with a rule for what happens if the two disagree; or
(ii) pre-register a rank-leg-only sub-label and require that any
non-CONVERGED seed be attributed to a specific leg before it enters a band.

---

### KW2.2 — MAJOR. The 1.25 GPU-h ceiling is enforced only over training, and an aborted cell silently deflates the band rate.

**Attacked text (§6, `:412-415`):**
> *"**Own cost ceiling:** every cell carries `--ceiling-gpuh 1.25` (§4),
> **enforced by the runner's own existing ceiling mechanism**"*

and §4 (`:288-299`): *"Hard ceiling (abort trigger) | 1.25 h | **15.0
GPU-h** … **Total ≤15 GPU-h, exactly the mandate's cap**"*.

**Counter-evidence — three distinct defects in one mechanism.**

1. **Scope.** `ceiling_s` is checked only inside `train_earlyln_cell`
   (`:198-201`), and only at `step % log_every == 0` (every 500 steps).
   The entire post-train instrument sequence — `z_dump` → `deep_probe` →
   Axis-C lock → `trust_screen` → `blank_out_check` → `eval_cell` over
   41/43/45 eval points at `EVAL_BATCHES=8 × EVAL_BATCH_SIZE=256` — runs
   **after** the ceiling return path, with no budget check. Worst-case
   per-cell wall clock is `ceiling + eval`, so "≤15 GPU-h" is not a hard
   bound.
2. **Band contamination.** On abort, `run_earlyln_cell` writes
   `status="ABORTED-BUDGET"` and returns. `harvest()` (`:383-386`) then
   records `cells[seed] = dict(status=...)` with **no `gate1` key**, so the
   seed counts in `n_seeds` (the denominator, via `discover_seeds_by_K`
   globbing the file that now exists) but can never count as CONVERGED. A
   cell that hit a wall-clock ceiling is therefore **indistinguishable in
   the rate from a cell that genuinely failed to train** — and the rate is
   what every band reads.
3. **Resume loop.** The skip-if-COMPLETED guard (`:243-248`) skips only
   `status == "COMPLETED"`. An `ABORTED-BUDGET` cell is re-run from scratch
   on the next supervisor pass and re-aborts — burning the ceiling
   repeatedly under the repo's standard
   `while [ ! -f STOP ]; do <cmd>; sleep 15; done` pattern.

**Why this is live, not theoretical.** The repo's saturation-packing
doctrine (CLAUDE.md Operating Doctrine: *"small cells packed N-per-GPU with
contention-priced ceilings"*) is exactly the regime that inflates per-cell
wall clock. At 2.5× nominal headroom, a 3-cells-per-GPU pack can reach the
ceiling.

**Discharge condition.** (i) Pre-register in §5 that any cell with
`status != "COMPLETED"` **voids that K's rate** and forces a re-run rather
than counting as a failed seed — the band arithmetic must never see an
aborted cell; (ii) either raise the ceiling to cover eval or state that the
15.0 GPU-h figure is training-only and give the eval-inclusive worst case;
(iii) pin a re-run/backoff rule so an aborted cell is not looped.

---

### KW2.3 — MAJOR. No band states what a MISSING or non-COMPLETED cell does to the rate.

Related to KW2.2 but independent: §5's entire decision procedure is defined
over `rate = (#seeds CONVERGED)/4` with no statement of what happens if
fewer than 4 cells complete. `harvest()`'s `gate_eligible = n_seeds >= 4`
guard (`:397-400`) protects against a *trimmed* seed list, but
`discover_seeds_by_K` derives `n_seeds` from **files on disk**, so a cell
that ran and aborted counts toward `n_seeds` while a cell that never
launched does not — two different failure modes, two different
denominators, neither disclosed.

**Discharge condition.** Pre-register the completeness precondition
explicitly: bands are read **only** on a K with 4/4 `status=="COMPLETED"`
cells; anything else is `INCOMPLETE-AT-K`, re-run, not classified.

---

### KW2.4 — MINOR. FLOP spread lower bound is wrong.

**Attacked text (§4, `:276-279`):** *"`F(26,27,64)=9,421,568`,
`F(28,29,64)=10,216,704`, `F(30,31,64)=11,022,080` — a mild **1.17×–1.28×**
spread over the measured K=24 rate"* and *"the ~15-28% FLOP growth"*.

**Recomputation** with the design's own formula
`F(K,d,h)=76Kh²+4dh²+12K²h+4Kdh+4d²h`, `h=64`:

| K | d | F | F / F(24,25,64) |
|---|---|---|---|
| 24 | 25 | 8,636,672 | 1.000 |
| 26 | 27 | **9,421,568** ✓ | **1.091** |
| 28 | 29 | **10,216,704** ✓ | **1.183** |
| 30 | 31 | **11,022,080** ✓ | **1.276** |

All three absolute values reproduce **exactly**. The spread is
**1.09×–1.28×**, not 1.17×–1.28×; "~15-28%" should be "~9-28%". Error is in
the conservative direction (overstates cost), so no budget risk.
`P(d,h)=40h²+4dh+46h+d` also reproduces and matches the self-test's own
assertion at `ncr_earlyln_scale.py:585-586`.

**Discharge:** correct the two numbers.

---

### KW2.5 — MINOR. §4 computes the spread and then does not apply it.

**Attacked text (§4, `:279-291`):** *"**Nominal estimate: ≈0.50 GPU-h/cell**
… | Nominal (empirically grounded) | ≈0.50 h | **≈6.0 GPU-h** |"*

A flat 0.50 h is applied to all three K after the 1.09–1.28× spread has
just been derived. Applying it to the measured K=24 rate (0.442–0.525,
mean ≈0.49) gives ≈0.53 / 0.58 / 0.63 h per cell at K=26/28/30 →
`4 × (0.53+0.58+0.63) ≈ **7.1 GPU-h**`, ~18% above the stated ≈6.0.
Immaterial against the 15.0 cap; internally inconsistent as "empirically
grounded" pricing. The ceiling arithmetic `12 × 1.25 = 15.0` is exact ✓.

**Discharge:** restate the nominal as ≈7 GPU-h, or state that 0.50 is a
deliberate round-number stand-in.

---

### KW2.6 — MINOR. §6 asserts pool conformance but ships no conforming spec; the command is CWD-relative and pins no validity check.

**Attacked text (§6, `:398-419`)** claims conformance to the
`idle_fallback_daemon.sh` pool contract (verified verbatim at `:10-18` of
that script). The contract itself is satisfied *in substance* — the cells
genuinely are independent, flat, and per-cell-ceilinged (see §4 CLEAN-9).
What is missing is the artifact:

The design's §4 gives a bare command. The pool's own reference spec,
`queue/jobs/pending/108_laneA_main_K48_s0.json`, carries eight fields:
`id`, `lane`, `hypothesis`, `cmd`, `gpu_h_estimate`, `output_dir`,
`validity_check`, `notes` — with an **absolute** interpreter and working
directory (`cd /home/nvidia/ncr && /home/nvidia/tdenv/bin/python3 …`) and a
`validity_check` that asserts `status=='COMPLETED'`,
`train.step == 80000`, `'eval' in d`, `blank_out.passed is True`. The
design's `--outdir results_kwall_characterization` is **CWD-relative** and
would resolve unpredictably under daemon promotion.

Positive note: choosing a **new** outdir is correct and avoids the
`cell_id(K,seed) = earlyln_K{K}_s{seed}` filename collision between a
`d=K+1` and a `d=2K` cell at the same `(K, seed)` — the exact collision the
07-12 wave had to work around (`EXPERIMENT_LOG.md:8452`).

**Discharge:** produce the 12 job JSONs in the job-108 format with absolute
paths, per-cell `gpu_h_estimate`, and a `validity_check` that additionally
asserts `d == K+1` and `d_override == K+1` (so a mis-flagged cell cannot
silently harvest as a `d=2K` cell).

---

### KW2.7 — MINOR. The queue sweep covered a directory set that does not exist locally.

**Attacked text (§3, `:169-171`):** *"`grep` over every
`matrix-thinking/queue/jobs/**/*.json` (**pending + fallback_pool +
claimed, all lanes**) … zero hits."*

`matrix-thinking/queue/jobs/` contains **only** `pending/`. There is no
`fallback_pool/` or `claimed/` in the repo — they live on the box under
`~/queue/` (per `idle_fallback_daemon.sh:28-30`: `QROOT="$HOME/queue"`,
`POOL="$QROOT/fallback_pool"`). The *result* is correct — I independently
grepped every job spec in the repo for `"K": 2[5-9]|3[01]` and
`--K 2[5-9]|3[01]` and got **zero hits** — but the sweep cannot have
covered the live pool, and a K∈{26,28,30} spec sitting on the box would
have been missed.

**Two further method gaps in the same §3 sweep**, neither of which changes
the (correct) conclusion:

- `:162-168` claims *"the only two textual hits"* are
  `EXPERIMENT_LOG.md:8569` and `NCR_MAPPING_LAW_DESIGN.md:471`. A loose
  search finds more — `EXPERIMENT_LOG.md:4690` (*"roughly K ≈ 31-39 at
  d=64"*, a KEY-ANCHORING sigmoid transition **width**, not an NCR cell),
  `EXPERIMENT_LOG.md:1860` (*"18K→25K"*, training steps), the DeltaNet
  lowercase-`k=28/30/31` truncation-rank family
  (`DELTANET_CAUSAL_RANK_DESIGN.md:1455,1522,1546,…`), and
  `stageg/task_he.py:25,115` (*"K <= 26"*, an alphabet-size bound). All
  adjudicate as non-NCR, so the conclusion stands — but "only two hits" is
  understated, and the documented pattern `K=2[5-9]` cannot have surfaced
  `1.25K=30` (no word boundary before `K`), so that hit was found by some
  route other than the one §3 describes.
- `:171-173` cites `find experiment-runs -iname "*K2[5-9]*"` → zero
  directories. Filename-only `find` is **not sufficient**, and this
  program supplies its own counterexample: **K=20 has real results** (4/4
  DEAD, d=40, ≈2.03 GPU-h, `NOVEL_ARCH_WATERFALL.md:4325-4348`;
  `EXPERIMENT_LOG.md:8468-8478`) and **zero matching filenames anywhere**,
  because those raws were never archived off the box. The load-bearing
  check is the JSON-**content** grep, which §3's five bullets do not list.

**Discharge:** re-scope the sentence to what was actually searched, list
the content-level greps as the load-bearing ones, and have the pre-launch
red-team check `~/queue/{pending,claimed,fallback_pool}` on the box.

---

### KW2.8 — MINOR. The build's own self-tests would give zero coverage of the config the wave runs.

**Attacked text (§2 Build note, `:136-153`):** adding
`GRID_SHAPES[26]=dict(d=52,h=64)` etc. and `GRIDS[K]=_gen_grid(K)`.

The additive-only discipline is correct and the `d=2K` values are required
by the module's own convention assert (`ncr_earlyln_scale.py:564`:
`assert GRID_SHAPES[_K] == dict(d=2*_K, h=64)`) — good catch by the draft.
But two consequences are undisclosed:

- `--smoke` (`:883-889`) and self-test **t5** (`:584-590`) both iterate
  `for K in GRID_SHAPES` and run a full end-to-end micro cell per K. Adding
  3 keys adds 3 cells to every smoke run, **each at the GRID_SHAPES default
  `d=2K` (52/56/60)** — *not* at the `d=K+1` the wave uses. The build's own
  tests would therefore exercise a config the wave never runs and never
  touch the config it does.
- t4b's convention assert loops over a **hardcoded** K list
  `(20,32,48,64,96,128,192,256)`, so the new keys are not auto-covered by
  it either.

The only `--d-override` coverage in the suite is **t10**, pinned to
K=16/d=17 (`:725-780`).

**Discharge:** add a `d=K+1` micro-cell smoke at one of the new K's (the
t10 pattern, `d_override=K+1`, `steps=4`, CPU) and extend t4b's K list.
This is cheap and it is the difference between a tested and an untested
build.

---

### KW2.9 — MINOR. Harvest emits n=0 rows for 12 unrelated K's.

`discover_seeds_by_K` (`:562-570`) iterates every `GRID_SHAPES` key. Run
against a fresh `results_kwall_characterization/`, K∈{14,15,16,20,24,32,
48,64,96,128,192,256} all yield `seeds=()` → `n_seeds=0` →
`gate_eligible=False` → `SUB4-DISCLOSED-ONLY(n=0)`. Noise only; **no false
verdict** (the guard correctly refuses to gate an empty rung). Worth one
sentence in §5 so a harvest reader is not misled.

---

### KW2.10 — MINOR. The `STATE.md:39-40` scoping question needs a recorded ruling, not an audit opinion.

**Attacked text (§6, `:421-428`):** *"The `STATE.md:39-40` 'NO NCR job
queue-eligible' restriction (2026-07-30) is scoped to the in-LM
write-conditioning claim pivot … **If the audit round disagrees with this
reading it should say so explicitly rather than silently defer.**"*

Saying so explicitly, as invited. There is a genuine tension:
`NCR_KLADDER_ATTACK_R2.md:762-765` (§6 item 2) cites the same sentence as a
**standing** restriction that *"bites before the first promotion"* for NCR
pool cells generally, and that report was adopted (`STATE.md:4-6`:
*"coordinator-verified vs raws and adopted"*).

Against that: the same STATE.md tick that adopted it **also** routes this
document explicitly — `STATE.md:11-14`: *"Fallback-runway successor:
K∈{26,28,30} recovery-leg wall characterization on live K=24 — draft agent
in flight (`NCR_KWALL_CHARACTERIZATION_DESIGN.md`, **DRAFT-R0 → audit →
adjudication → build → pool**)."* The coordinator has already ruled that
this line of work may reach the pool.

So the design's reading is **substantively right**, but it is not the
design's ruling to make. **Discharge:** record the scoping ruling in
STATE.md / EXPERIMENT_LOG at adjudication time, before pool insertion —
not inside the design document.

---

## §3 WHAT I COULD NOT ATTACK — VERIFIED CLEAN

These are load-bearing and correct. Several are unusually good work and
should survive any revision intact.

**CLEAN-1 — §2(c)'s mod-K crash claim is exactly right at all three K, and
I verified it by execution, not by reading.** `REALISTIC_DEPTHS =
(5,12,20,29,40,61)` (`ncr_ortho_write.py:88`), asserted novel via
`r = h % K; assert r not in (0,1,2,3)` (`:249`):

| K | residues of (5,12,20,29,40,61) | assert fires on |
|---|---|---|
| 24 | 5, 12, 20, 5, 16, 13 | — (passes) |
| **26** | 5, 12, 20, **3**, 14, 9 | **h=29, r=3** |
| **28** | 5, 12, 20, **1**, 12, 5 | **h=29, r=1** |
| **30** | 5, 12, 20, 29, 10, **1** | **h=61, r=1** |
| 32 | 5, 12, 20, 29, 8, 29 | — (passes) |

Matches the draft's three claims verbatim. This is the design's own
original code-read and it is correct.

**CLEAN-2 — the new-K grids are sound; no mod-K trap is introduced.** I
imported `ncr_task` and executed the proposed additive extension in memory
(no file modified):

| K | ladder | h\* | residue | sweep | eval_points | claim-eligible |
|---|---|---|---|---|---|---|
| 26 | 23,49,101,205,413,829,1661,3325 | 205 | 23 | 183..208 (26) | 41 | 8 |
| 28 | 25,53,109,221,445,893,1789,3581 | 221 | 25 | 197..224 (28) | 43 | 8 |
| 30 | 27,57,117,237,477,957,1917,3837 | 237 | 27 | 211..240 (30) | 45 | 8 |

All invariants pass: every ladder point ≡ `K-3`; `h_star % K == K-3`; sweep
covers every residue exactly once; `residue_label(K-3, K) == "novel"`;
`residue_label(h_star+3, K) == "identity"`. `nt.claim_config(K, d=K+1)`
constructs cleanly (inherited `te.TaskEConfig.__post_init__` periodicity
assert and `K <= d` assert both run and pass), and `nt.eval_points(K,
d=K+1)` passes its own exhaustive-labeling teeth (`ncr_task.py:267-273`).
**§7's "no new far-depth residue arithmetic" non-goal is honored and the
result is correct.**

**CLEAN-3 — K∈{25..31} is genuinely open. This is the design's central
premise and it survives a hostile independent re-sweep.** Confirmed two
ways. (i) By me: `GRIDS`/`GRID_SHAPES` in both `ncr_task.py` and
`ncr_earlyln_scale.py` define K ∈ {8,12,14,15,16,20,24,32,48,64,96,128,
192,256} — no key in 25..31; a regex sweep of every job spec in the repo
for `"K": 2[5-9]|3[01]` / `--K 2[5-9]|3[01]` returns zero hits. (ii) By a
separate exhaustive sweep dispatched for this audit, covering
`EXPERIMENT_LOG.md`, `STATE.md`, `KILL_LIST.md`, all `NCR_*.md`,
`NOVEL_ARCH_WATERFALL.md`, `archive/`, all 122 local run dirs, **the SSD
superset at `/Volumes/1TB_SSD/learned-representations/` (verified mounted,
4,334 files, traversal control-tested)**, all 368 queue specs,
`generate_jobs.py`, all 28 `ncr/*.py`, and **git history via `git log -S`
across all refs**:

- `"K": 2[5-9]|3[01]` over every JSON in local **and** SSD
  `experiment-runs/` → **0 files**.
- Queue filename K tokens: `K16 K20 K24 K32 K48 K64 K96 K128 K192 K256`.
  SSD run-dir K tokens: `K16 K24 K32 K48 K69 K72 K78 K84 K90` (69–90 are
  PARAM_AXIS/keyanchor, non-NCR). Nothing in 25–31.
- `K=26`/`K=28` appear in **exactly one commit ever** (`af8f1cf`, the
  DRAFT-R0 itself). `K=27`/`K=29`/`K=31` appear in **zero commits ever**.
- Bonus: NCR `d` values on record are `{16,17,25,32,33,40,48,64,96}` — so
  the proposed `--d-override {27,29,31}` is also virgin territory.
- `KILL_LIST.md` is entirely Matrix-CODI content (last modified
  2026-04-17) and carries **no NCR content and no K constraint** — the
  design's implicit reliance on it being clear is correct.

Both false positives the draft names (`EXPERIMENT_LOG.md:8569`,
`NCR_MAPPING_LAW_DESIGN.md:471`) adjudicate exactly as the draft says: the
first is garbled prose for *d=25=K+1 at K=24* — confirmed against
`ncr/analyze_dratio_blocks.py:87-96`, which defines exactly four blocks
(K=16 d=17, K=16 d=32, K=24 d=25, K=24 d=48), and against
`q3_mechanism_results.json`, which contains only `"K": 16` and `"K": 24`.
(Subject to KW2.7's caveats about the on-box pool and the sweep's method.)

**CLEAN-4 — §3's K=16/24/32 table reproduces exactly from the raws.** I
recomputed all of it independently (`eval.points[h].reads.binexp
['recovered_frac@0.9']` for `h∈{1,2,3}`, `deep_probe.A_eff_rank`, `gpu_h`).
Every h1/h2/h3, every `indist_min`, every `AER/K`, every `gpu_h` matches to
the digit, and all cells are confirmed `train.step == 80000`,
`status == COMPLETED`, `d_override == d`. The draft's finer disclosure is
also correct and is an improvement on the log: **K=32 seed 0 is DEAD
(0.464 < 0.5), not PARTIAL — 0/4 CONVERGED, 3/4 PARTIAL, 1/4 DEAD**,
against `EXPERIMENT_LOG.md:8642-8644`'s coarser "3/4 PARTIAL". Correcting
the record against the raws is exactly right.

**CLEAN-5 — the runner supports the exact config the cells need.** Verified
against `ncr_earlyln_scale.py:850-912`: `--cell`, `--K`, `--seed`,
`--steps`, `--ceiling-gpuh`, `--outdir`, `--stop-file`, `--d-override` all
exist with the stated semantics. `d_eff = d_override if d_override is not
None else d_default` (`:236`) threads to **model construction**
(`NCREarlyLNModel(d=d_eff, h=h_eff)`, `:253`), to `nt.claim_config(K,
d=d_eff)` (`:251`), to `nt.eval_points(K, d=d_eff)` (`:283`), and to
`rn.eval_cell(..., d=d_eff)` (`:308`); both `d` and `d_default` are
recorded in the JSON (`:250`). `h` comes from `GRID_SHAPES[K]["h"]` and is
**64** in every proposed entry — unchanged. `--K`'s
`choices=sorted(GRID_SHAPES)` does require the additive build, exactly as
the draft states.

**CLEAN-6 — §2(d)'s "h never binds" argument holds.** Binding rank ceiling
`min(d, h+1)`; for K∈{26,28,30}, `d = K+1 ∈ {27,29,31}` and `h+1 = 65`, so
`min(d,65) = d` in every cell. No `h(K)` change is licensed or needed, and
§7's non-goal is correct.

**CLEAN-7 — the per-K label thresholds match `harvest()` exactly.** Design
§5: ROBUST ≥3/4, PARTIAL 1–2/4, DEAD 0/4. Code (`:397-406`):
`rate >= 0.75` → `CONVERGED-ROBUST`; `n_converged >= 1` →
`CONVERGED-PARTIAL`; else `TRAINABILITY-DEAD`. No new threshold invented,
as claimed.

**CLEAN-8 — the A4.9 guard is real, not a rename.** A4.9's defect
(`NCR_KLADDER_ATTACK_R2.md:481-499`) is *"a median over a data-dependent
subset with no minimum-n floor."* §5's fixed-denominator rate over n=4,
with every seed counted regardless of outcome, removes the data-dependent
`n` at its root — there is no conditioning population and no median. **This
genuinely closes A4.9.** (See KW1.3 for the separate point that the
INDETERMINATE flag, which the charter asked about, is *not* what closes it
and should not be presented as such.)

**CLEAN-9 — the far-depth axis is genuinely non-gating everywhere; A4.8's
33% does not leak in.** Confirmed by code path, not by prose: every band
reads the per-K CONVERGED-rate, which comes from `_cell_gate1`
(`indist_min` + `A_eff_rank`). `_cell_gate2` — the `h_star` /
`failure_front_h` / `sweep_min_rec` machinery where the ~33% per-seed rate
lives — is computed only for CONVERGED cells and is referenced by **no**
band. §3's refusal to borrow A4.8's coin-flip caveat for a metric it was
never measured on is correct, and §7's "reported only" scoping holds.

**CLEAN-10 — pool-contract substance.** The 12 cells are genuinely
independent (no cell's launch depends on another's result), the bands are
read at harvest over the complete set rather than as a pre-launch gate, and
each cell carries its own `--ceiling-gpuh`. This is a real improvement over
the SPENT K-ladder design that A4.12 killed. Only the *spec artifact* is
missing (KW2.6) and the ceiling's scope is narrower than claimed (KW2.2).

---

## §4 VERDICT

**REV-REQUIRED.** 4 FATAL, 7 MAJOR, 7 MINOR.

**Forcing findings (all four FATALs must be discharged before build):**

- **KW1.1** — the 80K premise. Either re-register the claim as
  budget-conditional (and strike the "last live rung" licenses), or add a
  2× budget leg as a pre-registered disambiguator. As written, the study
  would measure convergence speed and report it as a trainability wall.
- **KW1.2** — the archive sweep must be re-run on the budget axis, and §3
  must record the *axes* swept, not only the hits.
- **KW1.3** — §5 must be re-specified as a demonstrated partition over the
  125-outcome space, with band (c) subsumed or given precedence, one of
  band (b)'s two readings deleted, non-monotone and multi-`K*` rules added,
  and a residual `UNRESOLVED-AT-n=4` band.
- **KW1.4** — `K*`'s domain must extend to include 24, or `(0,0,0)` — a
  likely and highly publishable outcome — is unreportable.

**Not BLOCKED, and this matters.** The mandate's question is sound and
remains open: K∈{25..31} is genuinely unmeasured (CLEAN-3), the cells are
correctly constructed (CLEAN-2, CLEAN-5), the config choice is right even
though its stated justification is partly miscited (KW1.5), the cost is
small and the arithmetic is close (KW2.4/2.5), and the far-depth noise
problem is correctly quarantined (CLEAN-9). A Rev-1 that fixes the frame
(KW1.1/KW1.2) and re-specifies the bands (KW1.3/KW1.4) is a good, cheap,
GPU-hot experiment.

**Recommended Rev-1 shape**, offered without prejudice to the coordinator's
adjudication: keep the 12 cells exactly as designed; extend `K*`'s domain to
`{24,26,28,30}`; rebuild §5 as an explicit partition; re-register the
headline as *"the 80K-budget convergence frontier between K=24 and K=32"*;
and add a **conditional** 4-cell 160K arm at the first sub-ROBUST rung
(≈1.2–1.5 GPU-h at the measured rate, keeping the total inside the
mandate's cap) as the pre-registered speed-vs-wall disambiguator. That last
addition is what converts KW1.1 from a fatal confound into the design's
strongest result.

**Ceremony note carried forward** (`EXPERIMENT_LOG.md:9173-9176`): the 15
GPU-h worst case sits at the 10–50 tier boundary, so a pre-launch
resource/placement red-team is required at build time in addition to this
audit round. KW2.2 (contention → ceiling → silent rate deflation) should be
handed to that red-team explicitly.

---

*Round 1 attack, 2026-08-06. Read-only pass; the only repo file created or
modified is this one. No command was run on the box, no job was launched, no
git mutation was made. Claims verified by direct file read, raw-JSON
`json.load` of 40 archived cells, in-memory execution of `ncr_task`/`task_e`
against the proposed K extension, and exhaustive enumeration of the §5 band
logic over all 125 reachable rate outcomes.*
