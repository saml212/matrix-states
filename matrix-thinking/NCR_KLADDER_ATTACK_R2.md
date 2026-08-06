# NCR K-LADDER — ADVERSARIAL ATTACK, ROUND 4 (filed as `NCR_KLADDER_ATTACK_R2.md`)

**Target:** `matrix-thinking/NCR_KLADDER_DESIGN.md`
**Target's own status header:** `REV-2 — CLEAR-FOR-CONDITIONAL-BUILD` (§A3
round-3 verdict, dated 2026-07-16; gauntlet draft → A1 → R1 → A2 → R2 → A3,
plus a post-freeze `§N1` novelty note added at commit `366b9fd`).
**This pass dated:** 2026-08-06 (21 days after the design was frozen).
**Scope:** read-only. Nothing else in the repo was modified; no command was
run on the box; no job was launched.

### VERDICT: **BLOCKED**

Not `REV-REQUIRED`. No revision of *this* document can license a build,
because the experiment it gates on has already run and returned **FAIL**,
and because the cells it is now being asked to replace are a *different
experiment* than the one it specifies. The correct disposition is to retire
this document to historical record and open a new design for whatever the
fallback ladder actually is.

**Findings: 4 FATAL, 10 MAJOR, 6 MINOR (20 total).**

---

## §0 TWO PREMISE CORRECTIONS BEFORE THE FINDINGS

Both are stated up front because they change what this round could
legitimately do.

**(0a) The dispatch brief's premise is stale.** The brief describes the
target as carrying status `DRAFT-STAGE-1-REV-1 (POST-ATTACK-1,
PRE-ATTACK-2)` and asks for "a full adversarial round-2 pass on the CURRENT
(Rev 1) text," attacking claims such as "the re-derived Gate-0 margins
(~0.445 claimed at every rung)."

That is not the current text. The design is at **REV-2**, has already passed
an independent attack round 2 (§A2: 0 FATAL, 4 MAJOR) *and* a round-3
verification (§A3: 7/7 dispositions faithful, `CLEAR-FOR-CONDITIONAL-BUILD`),
and the `~0.445` margin claim was **retracted by Rev 2** (finding A2.1) and
replaced with the `0.9K/(K+1)` fill table. The string
`DRAFT-STAGE-1-REV-1 (POST-ATTACK-1, PRE-ATTACK-2)` still appears in the
file — at line 1342, inside the `§R1` changelog, which the header explicitly
preserves as historical record. Git confirms the order:
`8cee2c5` (Rev 1) → `51ea69f` (§A2) → `c8f303a` (Rev 2) → `6c235d8` (§A3
CLEAR) → `366b9fd` (§N1).

This pass therefore attacks **Rev 2**, and is chronologically the **fourth**
adversarial round.

**(0b) Finding IDs use the `A4.x` namespace, not `A2.x`.** The brief asked
for `A2.1, A2.2, ...`. Those IDs are already occupied in the repo by the
recorded round-2 attack (`§A2`, findings A2.1–A2.7, each with an adjudication
and a discharge row). Minting a second, contradictory `A2.1` would corrupt
the discharge bookkeeping that this project's own gate rules depend on. IDs
below are `A4.1`–`A4.20`.

---

## §1 WHAT ROUND 4 CONFIRMS AS CLEAN

Stated first so the FATALs are not read as a general indictment. I
recomputed, independently, from the pinned formulas
(`NOVEL_ARCH_WATERFALL.md:3899-3901`) —
`P(d,h)=40h²+4dh+46h+d`, `F(K,d,h)=76Kh²+4dh²+12K²h+4Kdh+4d²h`,
`NS(d)=160d³+48d²`:

| K | d=K+1 | h=2K | P | F | NS | F+NS | ratio vs K32 | NS/F |
|---|---|---|---|---|---|---|---|---|
| 32 | 33 | 64 | 175,265 | 11,837,696 | 5,802,192 | 17,639,888 | 1.000 | 0.490 |
| 48 | 49 | 96 | 391,921 | 39,905,664 | 18,939,088 | 58,844,752 | 3.336 | 0.475 |
| 64 | 65 | 128 | 694,593 | 94,536,192 | 44,142,800 | 138,678,992 | 7.862 | 0.467 |
| 96 | 97 | 192 | 1,557,985 | 318,874,368 | 146,479,312 | 465,353,680 | 26.381 | 0.459 |
| 128 | 129 | 256 | 2,765,441 | 755,631,104 | 344,269,008 | 1,099,900,112 | 62.353 | 0.456 |

**Every §2 number reproduces exactly.** So do: the fill table
`0.9K/(K+1)` = 0.873/0.882/0.886/0.891/0.893; per-cell pricing
9.34/22.01/73.87/174.59 h and 14.14/33.33/111.85/264.38 h; grid subtotals
1,119.22 / 1,114.09 / **2,233.31** h; floor 125.41 h; cap margin 24.59 h;
out-of-cap 275.65 h; program total 425.65 h; ceilings
28.29/66.67/223.71/528.75 h; the `d=1.25K` fallback (F+NS = 1,433,583,616,
81.27×, 227.55 h; NS(160)/NS(129) = 1.907); `min|λ|` bars
0.9275/0.9431/0.9603/0.9695 (each `bar^h*` = 0.01478); coprime bars
0.9566/0.9674/0.9782/0.9836; the memory table `320d²` (0.35/0.77/1.35/3.01/
5.33 MB per example); `0.9^136 = 5.98e-7`; and every §3 residue.

**Code claims verified against raw source.** `row_out = nn.Linear(h, d)` with
`Z = row_out(q)`, `q:(B,d,h)` ⇒ `rank(Z) ≤ min(d, h+1)`
(`chapter2/model_v4.py:52,63`) — A1.1's bound and A2.1's binding-ceiling
correction are both right. `d_eff = K + 1` hardcoded at
`ncr_ortho_write.py:286`, `h_eff = els.GRID_SHAPES[K]["h"]` at `:287`.
Gate-0 at `ncr_earlyln_scale.py:322` with `AEFF_RANK_FRAC_BAR = 0.9`
(`:97`) and `CONVERGED_INDIST_BAR = 0.9` (`:95`).

**Raw-archive precedent verified.** I re-read
`experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/earlyln_K{16,24}_s{0..3}.json`
directly. They do carry `d=17` / `d=25` (genuine `d=K+1` rows, not `d=2K`),
`h=64`, and per-seed mean `deep_probe.A_eff_rank` of 15.9992–15.9996 (K=16)
and 23.924–23.998 (K=24). Rev 2's correction of round 2's "~0.93" misread was
itself correct.

**Three cited sources spot-checked.** (i) The `P`/`F` formulas are at
`NOVEL_ARCH_WATERFALL.md`'s §9.3-corrected block as cited — faithful.
(ii) "tight-spare `d=K+1` reached 4/4 Gate-1 CONVERGED at BOTH K=16 and
K=24 (`NCR_NEXT_LEVER_DESIGN.md` §2.1)" — confirmed verbatim in
`EXPERIMENT_LOG.md:8488` ("Probe A's tight-spare d=K+1 CONFIRMS at both K=16
and K=24 (Gate-1 4/4 CONVERGED, was 1/4 and 0/4 at d=2K)"). **Faithful as
quoted — but materially incomplete; see A4.6.** (iii) The 2.8 h / 4.24 h
pricing base traces exactly to `NCR_ORTHO_WRITE.md` § CEILING AMENDMENT
(K=32, 320K steps, measured on-box) — faithful.

**The arithmetic is not where this design breaks.** Rounds 1–3 verified the
numbers thoroughly and correctly. Round 4's findings are entirely about
premises, provenance, and scope — the layers under the arithmetic.

---

## §2 FATAL

### A4.1 — FATAL. The design's own execution gate has already resolved NEGATIVE. The ortho-write verdict of record is FAIL, and §9 says this document does not execute under FAIL.

**Attacked text (§9, verbatim):**
> "**Under NULL or FAIL:** this document does not execute. No cells launch."

and the header:
> "This is a CONDITIONAL design ... It executes ONLY if the running
> orthogonal-write pre-registration (`NCR_ORTHO_WRITE.md` §4) returns
> **WIN** or the pre-registered **PARTIAL** band for Part A."

**Counter-evidence.** The run completed and was adjudicated three weeks ago.
`STATE.md:91-93`:
> "the 24-cell ortho-write run COMPLETED and its **verdict of record is
> FAIL** (both Part A + Part B, `NCR_ORTHO_WRITE.md` §9): the
> Newton–Schulz-polar orthogonal write is Gate-0 dead 4/4 seeds at K=24 and
> K=32 — the pre-registered 'too rigid to train through' mechanism."

`NCR_ORTHO_WRITE.md` §9.1 carries the per-cell raws: all 16 Part-A cells,
ortho arm, in-dist `recovered_frac@0.9` = **0.000** at h=1,2,3 in 4/4 seeds
at both K; `A_eff_rank` collapsed to 13.53–17.96 (K=24) and 17.61–27.30
(K=32); every far-depth ladder rung 0.000; loss dips then returns to ≈1.0
(random) in every cell. §9.2: Part B ortho-bank likewise Gate-0 dead 4/4,
compounded by a dead free-bank baseline. §10 re-audit: **mechanism confirmed,
no bug** — a `d=K+1` ill-conditioning trap (scale-invariant loss puts no
pressure on the spare direction → σ_min collapses below the NS working range
→ polar backward explodes ~1/σ_min).

Note what §10 localizes the failure to: **the `d=K+1` tight-spare convention
itself**, which this design carries forward unchanged and calls "the single
biggest structural assumption in this design."

The pre-registered next move was also already taken and also failed: the
expm/Cayley fallback's Stage-0 damped-polar smoke gate returned
"**§B8 verdict = FAIL / never-engaged**" (`STATE.md:103-109`).

**Consequence.** The double gate is discharged in the negative. `§A3`'s
`CLEAR-FOR-CONDITIONAL-BUILD` was correct *at the time it was written* —
its condition simply never came true. The document is not "cleared and
waiting"; it is **spent**. Every downstream artifact (§4 grid, §5 Stage-0,
§6 bands, §7 ceilings, §8 packing, §9 branches) prices and gates a mechanism
now measured dead at both rungs below the ladder's floor.

**What would discharge it.** Nothing, within this document. A fallback
ladder needs a NEW pre-registration naming a mechanism that is actually
alive, with its own novelty-gate re-entry (a mechanism change is a claim
pivot under the standing 2026-07-16 doctrine). This document should be
marked `SPENT — GATE RESOLVED FAIL 2026-07-17` and moved to historical
record.

---

### A4.2 — FATAL. Scope mismatch: the design specifies a different experiment from the parked cells it is being asked to replace. K=192/256 appear in no table and were explicitly rejected.

**Attacked plan (`STATE.md:7-11`):**
> "parked K48–K256 cells ruled NOT pool-eligible (fixed-h=64, attack-1 FATAL
> relics); K-ladder gauntlet ATTACK ROUND 2 dispatched ... — on CLEAR:
> rebuild cells at h(K)=2K → audit → pool = the PI's vital fallback ladder."

**Counter-evidence.** I read the parked specs. They are not this design's
cells on any axis. `queue/jobs/pending/108_laneA_main_K48_s0.json`:

```
"hypothesis": "Does the earlyln recipe ... let K=48 (d=96, h=64) converge and
 compose exactly at far depth, extending S11's own K=14/15/16/24 result up
 the K axis, per the task's K-ladder charter?"
"cmd": "... ncr_earlyln_scale.py --cell --K 48 --seed 0 --steps 80000 ..."
"gpu_h_estimate": 1.154
```

Side by side:

| axis | parked relics | `NCR_KLADDER_DESIGN.md` |
|---|---|---|
| write mechanism | earlyln **free write** | **NS-polar orthogonal** write |
| `d` | `2K` (`GRID_SHAPES`, 96 at K=48) | `K+1` (49 at K=48) |
| `h` | 64 fixed | `2K` |
| steps | 80,000 | 320,000 |
| cost/cell | ~1.154 GPU-h | 9.34–174.59 GPU-h |
| K set | 48, 64, 96, 128, **192, 256** | 48, 64, 96, 128 |
| gating experiment | — | ortho-write verdict (FAILED) |

`GRID_SHAPES` (`ncr_earlyln_scale.py:75-93`) pins `d=2K, h=64` for every
K∈{48…256}, with its own comment: *"Condition-A proportional-headroom
convention (d=2K, h=64) applied verbatim."* The design's `d=K+1` comes from
a different file's hardcode (`ncr_ortho_write.py:286`), which overrides
`GRID_SHAPES`' `d` while still reading its `h`.

And the design **explicitly refuses** the top of the parked range (§2):
> "32→64→128→256 was considered and rejected: a 256 jump has zero validated
> calibration point anywhere near it."

For reference, K=192/256 under the design's own pinned formulas and `h=2K`:
ratio-vs-K32 **209.8×** and **496.7×** → 587 h and 1,391 h per *single*
primary cell at the 2.8 h base. Both are off every table in the document and
each exceeds the entire 150 h committed-sweep cap several times over.

**Consequence.** "Rebuild the cells at h(K)=2K" cannot be executed against
this design. It licenses neither the mechanism, nor the `d`, nor the step
count, nor the cost model, nor the top two rungs of the parked cells. Whatever
gets built would be governed by no design document at all — and the
`h(K)=2K` conclusion itself would be transplanted across a mechanism change,
a `d`-convention change, and a 4× budget change, on the strength of an
analysis performed under none of them.

Note the transplant is not even conclusion-preserving. Under the relics'
`d=2K` with `h=2K`, the binding ceiling is `min(2K, 2K+1) = 2K` and the
required fill is `0.9K/2K = 0.45` — i.e. Rev **1**'s retracted "comfortable
0.445" figure is the *correct* one for `d=2K`, and Rev 2's "razor-thin 0.88"
is correct only for `d=K+1`. The two revisions are each right about a
different config, and the rebuild plan does not say which config it is
building.

**What would discharge it.** A new design that (a) names one mechanism and
one `(d, h, steps)` convention explicitly, (b) covers the actual K range to
be built, and (c) re-derives cost/memory/gates under that convention.

---

### A4.3 — FATAL. "The LAST VALIDATED rung (K=32)" is measured FALSE in this repo's own EXPERIMENT_LOG — recorded four days *before* the design was written, and never cited by it.

**Attacked text (§2, three places):**
> "`h=2K` is also the ratio already realized, unmodified, at the LAST
> VALIDATED rung (K=32 ...)"

> fill table row: "| 32 (ref, **validated**) | 33 | 28.8 | **0.873** |"

> "Precedent: tight-spare `d=K+1` reached 4/4 Gate-1 CONVERGED at 1× budget
> at BOTH K=16 and K=24 ... and **the currently-running wave extends it to
> K=32.**"

**Counter-evidence — `EXPERIMENT_LOG.md:8622-8640`, dated 2026-07-12,
i.e. four days BEFORE this design was drafted:**

> "**K=32 full d(K) grid (d∈{33,40,48,64}, n=4 seeds each, 1× budget): every
> arm lands TRAINABILITY-DEAD (0/4 fully Gate-1 CONVERGED).** `front` is
> pinned at the trivial K−3=29 rung in all 16 cells, every d, every seed —
> zero far-depth signal anywhere. ... **CLOSED-AT-THIS-K**"

> "Nuance disclosed ...: d=K+1=33 is the qualitatively 'least dead' arm (3/4
> seeds PARTIAL, best seed **0.871** in-dist recovery vs the 0.9 Gate-1
> bar ...)"

So `d=K+1` at K=32 had *already been run at n=4* and had already failed
Gate-1, with the best seed 0.871 against a 0.9 bar. The design's "the
currently-running wave extends it to K=32" describes the question as open
when the repo had already closed it.

**Independently reconfirmed** by the very wave the design was blind to —
`NCR_ORTHO_WRITE.md` §9.1's free arm at K=32/d=33, in-dist h=3 reading
0.875 / 0.912 / 0.912 / 0.896 (**2 of 4 seeds below the 0.9 Gate-0 bar**),
`rec@0.9` = **0.000** at every far rung {5,12,20,29,40,61} in 4/4 seeds,
cond 54.9–558.6, `min|λ|/c*` 0.12–0.32.

The only rung that is actually validated is **K=24** (free arm: `A_eff_rank`
24.00 = full K, cond 1.0–1.1, `rec` ≈ 1.000 at all depths through h=61, 4/4
seeds).

**Consequence.** Every extrapolation in §2 is anchored on a point that does
not hold. The h-ratio justification ("extending it to K∈{48,64,96,128} adds
exactly ONE new extrapolation dimension") is anchored at K=32; the fill
table's reference row is labelled "validated" at K=32; §5's calibration
framing treats K=32 as the known-good baseline. The true extrapolation
distance from the last live rung (K=24) to the ladder's floor (K=48) is
**2×**, across a wall the repo has measured twice and not crossed once.

**What would discharge it.** Nothing short of a live K=32 result under
whatever convention the rebuild picks. Absent that, the anchor is K=24 and
the design must say so, price the 2× jump, and stop calling K=32 validated.

---

### A4.4 — FATAL. Every Rev-1/Rev-2 fix targets the Gate-0 leg that already PASSES. The leg that actually kills Gate-0 at K=32 is untouched by `h(K)=2K` and unanalyzed anywhere in the document.

**Structural fact.** Gate-0 is a **conjunction of two legs**
(`ncr_earlyln_scale.py:319-322`):

```python
indist_min = min(recovered_frac@0.9 for h in {1,2,3})
aer_mean   = mean(deep_probe.A_eff_rank)
if indist_min >= 0.9 and aer_mean >= 0.9*K:  v = "CONVERGED"
```

Leg 1 = in-distribution recovery. Leg 2 = effective rank.

**Attacked text.** The entire A1.1 → A2.1 → Rev-2 arc is about **leg 2 only**:
A1.1's FATAL (`rank(Z) ≤ h+1 = 65`), the `h(K)=2K` fix, the binding-ceiling
correction to `min(d,h+1)=d`, the fill table, §5's Stage-0 deliverable 2
("Gate-0 pass/fail = `d`-cap fillability, ONE risk not two"), and §2's
honesty paragraph ("the open question is purely whether the encoder ... FILLS
`d`'s cap"). §5 states it outright:

> "a Gate-0 failure at K=128 can no longer be blamed on the old fixed-h rank
> cap ... narrowing the diagnosis to genuine SGD-trainability /
> capacity-fillability questions"

**Counter-evidence.** At the last rung with data, **leg 2 passes and leg 1
fails**. `EXPERIMENT_LOG.md:8641-8647` on K=32/d=33:

> "3/4 seeds PARTIAL, best seed 0.871 in-dist recovery vs the 0.9 Gate-1
> bar, **`A_eff_rank` already clearing 0.9×32 in all 4 seeds**"

The ortho-write free arm agrees: `A_eff_rank` 31.05–31.09 ≫ 28.8 (leg 2
comfortably passed) while in-dist h=3 reads 0.875–0.912 (leg 1 marginal/failing).

**Consequence.** The design's central fix is aimed at the wrong constraint.
`h(K)=2K` raises encoder capacity toward a rank cap that the measured data
says was never the binding problem at K=32. Meanwhile:

- Leg 1 (in-dist recovery ≥ 0.9) has **no** capacity analysis, **no**
  scaling model, **no** fill table, and **no** Stage-0 instrument beyond a
  pass/fail readout in §5.
- §5's three named risks are `d`-cap fillability, NS-convergence quality,
  and wall-clock. **Recovery-leg degradation with K — the one failure mode
  actually observed — is not on the list.**
- §6's `FAIL(K)` band ("Gate-0 DEAD in ≥3/4 seeds") therefore fires without
  the design being able to say which leg fired it, in the exact regime where
  the two legs are known to dissociate.

This is precisely the "hidden scaling interaction" pattern round 1's FATAL
exemplified, and rounds 1–3 all missed it for a structural reason worth
recording: **round 1 framed the problem as a rank problem, and rounds 2 and 3
inherited that frame.** Round 2 recomputed the ceiling and round 3 verified
the recomputation; neither stepped back to ask whether rank was the binding
leg at all. Multi-round attack does not automatically escape a frame that
round 1 set.

**What would discharge it.** A Stage-0 that instruments **both** legs
separately with pre-registered per-leg bands; a stated model for how in-dist
recovery is expected to scale with K; and an explicit acknowledgement that
the K=32 evidence points at leg 1, not leg 2.

---

## §3 MAJOR

### A4.5 — MAJOR. A standing, pre-registered repo BLOCK on generating K=48 cells is never cited or discharged — and the internal-archive novelty sweep declared the ladder clean without touching it.

**Counter-evidence (`EXPERIMENT_LOG.md:8636-8640`):**
> "**WAVE-1b (K=48's own d(K) grid, jobs reserved at `513`-`524` in the
> design's own semantic numbering) is thereby BLOCKED per the design's
> pre-registered staging rule — not generated, not launched.**"

The trigger for that block is exactly the premise the K-ladder needs:
K=32 reached CONVERGED-ROBUST at **no** `d`, so the staging rule refused to
license K=48.

**Honest scope.** The block names WAVE-1b, the K=48 *d-grid* under
`NCR_MAPPING_LAW_DESIGN.md`. The K-ladder proposes one `d` per K, so this is
adverse precedent rather than a literal prohibition on the same cells — which
is why this is MAJOR and not FATAL. But the *reasoning* transfers intact, and
CLAUDE.md's gate is explicit that the internal sweep exists to ensure we
"don't redo or contradict our own recorded work."

**Aggravator.** §N1's internal-archive sweep reports "Cell virginity holds …
Zero re-run overlap with any internal experiment inventory" — true as a
*cell-inventory* check, and useless as a *precedent* check. It searched for
duplicated cells, not for recorded rulings against the direction. The
2026-07-12 mapping-law harvest is the single most on-point entry in the log
for this design and appears nowhere in it.

**What would discharge it.** Cite the WAVE-1b block, state whether the new
design's K=48 cells fall inside or outside it, and if outside, say why the
staging rule's premise does not apply.

---

### A4.6 — MAJOR. §2's precedent citation reproduces the licensing half of its source and omits the bounding correction recorded in the same log entry.

**Attacked text (§2):**
> "Precedent: tight-spare `d=K+1` reached 4/4 Gate-1 CONVERGED at 1× budget
> at BOTH K=16 and K=24 (`NCR_NEXT_LEVER_DESIGN.md` §2.1)"

**Verification.** The quoted claim is TRUE (`EXPERIMENT_LOG.md:8488`,
`:8509-8511`). But the same log carries an explicit, binding scope correction
on that exact conclusion (`:8654-8662`):

> "**Scope correction this forces on §11.4 (stated plainly).** §11.4's own
> conclusion — that the d=K+1 tight-spare convention is 'implicated' as the
> fix for the K-wall — was drawn from exactly two K's (16, 24), both ≤24.
> This wave shows the SAME convention fails Gate-1-robustness at K=32.
> §11.4's convention-confound conclusion is therefore **bounded to K≤24, not
> a general d(K) law**; an absolute-K component (or an unidentified
> K-dependent factor) is back in play for K≥32."

The design uses the K≤24 result to license a `d=K+1` extrapolation to
K∈{48,64,96,128} — the precise inference the log had already forbidden.

**What would discharge it.** Quote the scope correction alongside the
precedent, and re-argue the extrapolation against it rather than around it.

---

### A4.7 — MAJOR. The Rev-2 fill story — the centerpiece of the A2.1 fix — contains a logic inversion and a number mislabeled as "validated" that was never measured.

**Attacked text (§2):**
> "K=32's own **validated** fill (0.873, table above) is the LOWEST of the
> five — the ladder's extrapolated fills (0.882-0.893) sit slightly **ABOVE**
> the one point that's actually measured at `d=K+1`, **mild positive
> evidence**, not proof"

repeated in §5:
> "§2's fill table: 0.882-0.893 fill **required** at K=48-128, vs **0.873
> validated at K=32**"

**Two defects.**

1. **0.873 is not a measurement.** It is `0.9 × 32 / 33` — the *required*
   fill, a pure arithmetic consequence of the bar and the ceiling. The fill
   table's own column header says so: "TRUE fill `0.9K/(K+1)`". No K=32
   `A_eff_rank/d` measurement appears anywhere in the design. (One exists:
   ≈31.07/33 ≈ 0.94 from `NCR_ORTHO_WRITE.md` §9.3's free arm — which would
   have supported the point honestly. It is not cited.)

2. **The comparison runs backwards.** Comparing *requirements* to
   *requirements*: the ladder's rungs require 0.882–0.893 where K=32 requires
   0.873. A **higher** required fill is a **harder** bar. Calling that "mild
   positive evidence" inverts the sign. Under the alternative reading — that
   "the one point actually measured at `d=K+1`" means the K=16/24 archive
   rows — the sentence is simply false in the other direction: 0.882–0.893
   sit *below* the measured 0.941/0.957, not above.

**Materiality.** The surrounding paragraph's other claim (measured
0.941–0.960 at K=16/24 vs a 0.87–0.89 requirement, "a real ~0.05-0.09
fill-fraction margin") is arithmetically sound and I verified it against the
raws. But this sentence is the one §5 quotes when it certifies the risk as
"real, precedented, but thin" — the certification rests on a
number that was never measured, described as validated.

**What would discharge it.** Delete "validated" from the 0.873 row, cite the
real K=32 measured fill, and either drop the requirement-vs-requirement
comparison or state its true (adverse) direction.

---

### A4.8 — MAJOR. No statistical power analysis anywhere — and the repo's own n=12 data on the *healthiest* rung implies the n=4 verdict is close to a coin flip.

**Attacked design (§4, §6).** n=4 seeds per cell; `WIN(K)` = "median
`rec@0.9` at `h*=K+8` ≥ 0.9 across Gate-0-passing seeds". No CI, no test
statistic, no power calculation, no multiplicity control.

**Counter-evidence (`EXPERIMENT_LOG.md:8663-8681`, K=24 — the one rung that
works — extended to n=12 specifically to characterize seed variance):**
> "Front: 4/12 reach `front=189`(=h*), 7/12 plateau at `front=93`, 1/12 stays
> at the trivial `front=21` ... Reliability under the strict whole-sweep
> metric is low: `sweep_min_rec` HOLD in **0/12**, DEGRADED in 1/12, FAIL in
> **11/12**; the looser `front≥h*` metric clears in **4/12 (33%)**"

At a 33% per-seed success rate, "median of 4 ≥ bar" (needing ≥2 of 4)
clears about **40%** of the time. The instrument returns a near-coin-flip on
the rung that *works*. §6's three-shape readout would then be fit to four
such coin flips.

**Aggravator.** The design's own §5 cites `STATE.md` §1.40's seed-variance
precedent to justify a second Stage-0 seed on FAIL, and A2.6 extended the
symmetry to PASS. So seed variance is acknowledged as decisive for the
*calibration* cell and ignored for every *sweep* verdict.

**What would discharge it.** A power calculation against the measured
per-seed rate, a pre-registered n sufficient to separate WIN from NULL at
that rate, and a pre-registered uncertainty statement on the scaling-law fit.

---

### A4.9 — MAJOR. The verdict is a median over a data-dependent subset with no minimum-n floor, leaving a live gap where a WIN can be declared on two seeds.

**Attacked text (§6):**
> "**WIN(K)** ... median rec@0.9 at `h*=K+8` ≥0.9 **across Gate-0-passing
> seeds**"
> "**FAIL(K):** Gate-0 DEAD in ≥3/4 seeds"

**The gap.** At exactly 2 of 4 seeds Gate-0-passing, FAIL does not fire
(2 < 3), and WIN is adjudicated on a median of **n=2** — the mean of two
values, one of which may be exactly the "LUCKILY pass" seed that A2.6 was
written to guard against. Conditioning the population on Gate-0 also selects
for the better-conditioned seeds, biasing the far-depth median upward by
construction. Neither effect is disclosed.

**What would discharge it.** Pin a minimum number of Gate-0-passing seeds
below which the cell is INDETERMINATE (not WIN, not FAIL), and disclose the
conditioning as a selection effect.

---

### A4.10 — MAJOR. The only deliverable the cap guarantees cannot answer the design's own pre-registered readout. Two of the three §6 shapes are unadjudicable on it, and the third is impossible by definition.

**Attacked text.** §2 sets the requirement:
> "Four points is the **minimum** for a crude power-law/decay fit with one
> point to spare for a sanity check."

§4's trim order delivers two:
> "**Floor, never trimmed:** Part A at K=48 and K=64, n=4 = **125.41h**,
> IN-CAP ... This is the wave's **minimum viable**, cap-VERIFIED
> deliverable under the pessimistic (compute-bound) worst case."

And §5's ABORTED-ON-COST branch reaches the same two points by an
independent route:
> "K=96 and K=128 are declared priced out of the committed sweep ... **The
> ladder's upper half (K=96, K=128) is OUT OF THIS WAVE'S SCOPE under this
> branch**"

**Against §6's three shapes.** (a) FLAT-HOLD = "WIN at all four K" —
**impossible** with two rungs. (b) GRACEFUL DECAY = "WIN/PARTIAL at K=48/64,
degrading toward NULL by K=96/128" — the degradation is defined on the two
rungs that were dropped. (c) CLIFF = "WIN holds at K=48 (and maybe 64), then a
sharp drop at a specific rung" — the drop rung is unobserved. **None of the
three pre-registered outcomes can be adjudicated on the guaranteed
deliverable.**

Both routes to two points are *likely*, not exotic: the trim order fires
under the compute-bound worst case the design says is "entirely plausible"
at 62× FLOP spread, and ABORTED-ON-COST fires on the same premise.

**Consequence.** "Minimum viable deliverable" is false by the design's own
§2 criterion. A wave whose most likely outcome is a two-point ladder is a
wave whose most likely outcome is unreportable against its own bands.

**What would discharge it.** Either protect three rungs in the floor, or
pre-register what a two-point result means (and stop calling four the
minimum).

---

### A4.11 — MAJOR. The K-invariant residual floor is an assumption presented as a derivation, and CLAUDE.md's instrument-relative rule is cited in support of the move it forbids.

**Attacked text (§6):**
> "**`min|λ|/c* ≥ 0.9^(40/h*(K))`, DERIVED from band arithmetic, not
> asserted** ... Rev 1 pins THIS residual floor, not the exponent base, as
> the K-invariant quantity (**honoring CLAUDE.md's instrument-relative
> rule**...)"

**Counter-argument.** The arithmetic *is* exact (I verified every bar; each
`bar^h* = 0.01478`). But the arithmetic is downstream of a free choice. The
floor `0.9^40 ≈ 0.0148` is back-computed from the K=32 bar; deciding that
*residual amplitude* — rather than the exponent base, or `rec@0.9` itself, or
`min|λ|` — is the quantity held invariant across K is an **assertion**, and
it is the only load-bearing content in the derivation.

The cited rule says the opposite of what it is used for. CLAUDE.md:

> "the C17/geo3 n_iter-sufficiency frontier MOVES with K/d ... **Never carry
> an admission profile derived at one K/d to another without
> re-validating.**"

The rule mandates *re-validation at the new K/d*. The design instead selects
an invariant and extrapolates it, and schedules **no re-validation at any
rung** — Stage-0 measures rate, Gate-0, and orthogonality, never the
floor-invariance premise. Worse, the anchor K=32 that the floor is derived
from is the rung A4.3 shows is not validated.

**What would discharge it.** State the invariance as an assumption, and
schedule a rung at which it is checked (e.g. measure `min|λ|` and `rec` at
two depths on the same operator and confirm the decay model before using it
as a gate).

---

### A4.12 — MAJOR. The autonomous fallback pool structurally cannot honor the gates this design pins, and the plan routes design-gated cells straight into it.

**Attacked plan (`STATE.md:10-11`):** "on CLEAR: rebuild cells at h(K)=2K →
audit → **pool** = the PI's vital fallback ladder."

**Counter-evidence — `queue/idle_fallback_daemon.sh`:** on 3 h of confirmed
all-GPU idle with `pending/` and `claimed/` empty, it promotes `WAVE=8`
specs from `fallback_pool/` in **filename order**, unconditionally:

```bash
wave=$(ls "$POOL" 2>/dev/null | sort | head -n "$WAVE")
for f in $wave; do mv "$POOL/$f" "$QROOT/pending/$f" ...
```

The daemon has no concept of Stage-0, staged escalation, `packed_ceiling`,
or the §7 abort trigger. Against the design's own mandates:

| Design mandate | Pool behavior |
|---|---|
| §5: "Stage 0 (mandatory, **blocks everything else**) ... BEFORE any other cell in this ladder launches, packed or not" | promotes on an idle timer; no gate |
| §5: "commit K=48 first ... confirm its own real rate and Gate-0 result, THEN advance to K=64" | promotes 8 at once, filename order |
| §7: abort at `1.5 × packed_ceiling(K,N)` | not implemented anywhere in the launcher |
| §4 trim order: K=96/K=128 dropped | promotes whatever is in the pool |

The daemon's own header states the intended contract — "ONLY audited,
queue-eligible specs may ever be placed in the pool — the pool is the runway,
**the ceremony gate stays upstream of it**" — which is exactly the problem:
this design's gates are *sequential and data-dependent* (Stage-0's realized
rate reprices the grid; K=48's result gates K=64), and a
promote-on-idle pool can only enforce gates that are static and upstream.

**Second, independent conflict.** `STATE.md:24-25` carries a standing
restriction from the current tick: "Next NCR lever = in-LM write-conditioning
= CLAIM PIVOT → novelty gate re-entry before any build; **NO NCR job
queue-eligible.**" K-ladder cells are NCR jobs. No discharge of that
restriction for these cells is recorded anywhere.

**What would discharge it.** Either make the pooled cells gate-free by
construction (a flat, independently-interpretable set with per-cell ceilings
and no sequential dependency), or keep them out of the pool and launch them
under coordinator control. Plus an explicit, recorded scoping of the "NO NCR
job queue-eligible" rule.

---

### A4.13 — MAJOR. The `h(K)=2K` build change has un-scoped cross-lane blast radius and would violate a recorded additive-only discipline.

**Attacked text (header, "Reused, unmodified inputs"):**
> "encoder hidden CURRENTLY `h=64` fixed across K (`GRID_SHAPES[K]["h"]`,
> `ncr_earlyln_scale.py:75`) — ... §2 below (Rev 1) requires **the build to
> change this** to `h(K)=2K` ... a build-time requirement this design
> licenses"

**The gap.** It never says *where*. The value is consumed at
`ncr_ortho_write.py:287` as `h_eff = els.GRID_SHAPES[K]["h"]` — a table
**shared** with the laneA and laneC earlyln cells (39 parked specs; every
`ncr_earlyln_scale.py --cell --K ...` invocation). Editing `GRID_SHAPES` in
place — the natural reading of "change this," since that is the location the
design cites — silently rewrites the configuration of every earlyln cell at
those K.

That table carries its own recorded discipline (`ncr_earlyln_scale.py:80-84`):
> "Queue-system K-ladder extension (2026-07-11 ...): **additive only**,
> K=14/15/16/24 above byte-identical."

An in-place `h` mutation is not additive. It would also desynchronize
archived result JSONs (which record `h`) from the code that claims to have
produced them, against CLAUDE.md's "save the exact script that was run
alongside experiment results."

**What would discharge it.** Name the exact edit site and mechanism —
preferably a new `--h` override threaded like the existing `d_eff` override,
leaving `GRID_SHAPES` byte-identical — and state the blast radius explicitly.

---

### A4.14 — MAJOR. §N1 is live, post-freeze, un-attacked text that asserts a claim Rev 1 retracted.

**Attacked text (§N1, the last live section of the file):**
> "**Pre-existing structural flag untouched.** The ladder's own `0.9·K ≤ 65`
> achievable-gate ceiling (structurally capping K=96/128, §2) is this
> design's own item and **remains as-is** — the novelty gate neither
> resolved nor disturbed it."

**Counter-evidence.** That ceiling is A1.1's FATAL. Rev 1 closed it by
adopting `h(K)=2K`; the live §2 contains no `0.9K ≤ 65` ceiling, and
§2 states the opposite: "**Gate-0 is still structurally POSSIBLE at every
rung** ... unlike the pre-Rev-1 impossibility at K≥96."

**How it survived.** §A3 verified that "none of those stale values survive as
a LIVE claim in **§1–§9**" — a grep scoped to §1–§9. §N1 was appended
*after* the round-3 freeze (commit `366b9fd`, subsequent to `6c235d8`), so it
never faced any verification round. A reader reaching the end of the document
is told the K=96/128 structural cap is still in force.

**What would discharge it.** Correct or strike the bullet; and extend the
freeze-time staleness grep past §9 to cover post-freeze appendices.

---

## §4 MINOR

**A4.15 — MINOR. FLOP-spread citation misstates its own source.** The header
cites `NOVEL_ARCH_WATERFALL.md`'s refutation as "4 measured cells flat within
noise despite **1.688×–2.102×** FLOP spread." That file's table lists four
cells at ratios 1.688 / 1.804 / 2.102 / **3.305**. The spread is 1.688–3.305.
The error understates the design's *own supporting* evidence (a wider verified
flat range would strengthen the overhead-bound argument), so it is
conservative in effect — but the number does not match the source. Fix: quote
1.688×–3.305×.

**A4.16 — MINOR. Base-model activation memory understated ~5–10×.** §2
estimates the encoder's buffers at "order **~65-135 MB** per BATCH of 256"
at K=128, "dwarfed by the NS driver's ~1.36 GB (roughly 5-10% of it)". The
`TransformerEncoder` runs `n_layers=3` (`model_v4.py:42-47`): self-attention
scores alone are `256×4×128×128×4 B ≈ 67 MB` **per layer** ⇒ ~201 MB, the
reader MHA `(B,4,d,K)` adds ~68 MB, and the FFN activations `(B,K,4h)` add
~134 MB per layer. Realistic total ~0.7–1.5 GB — **comparable to** the NS
driver, not 5–10% of it. The design correctly flags the estimate as
UNVERIFIED and the conclusion (fits easily in 80 GB) is unaffected, so this
is MINOR — but the number feeds §8's `N=2` packing decision, where it is
multiplied.

**A4.17 — MINOR. The evidence base mixes training budgets without
normalization.** The fill precedent (K=16/24) is from **80K-step** nextlever
cells; the pricing base (2.8 h/4.24 h) and the K=32 anchor are **320K-step**
ortho cells; the parked relics are **80K-step**. The budget is
outcome-changing: K=24 far-depth reads 4/12 on `front≥h*` at 80K
(`EXPERIMENT_LOG.md:8666`) but 4/4 at `rec ≈ 1.000` at 320K
(`NCR_ORTHO_WRITE.md` §9.1). No cross-budget normalization appears anywhere.

**A4.18 — MINOR. The 1.3× contention factor has no measurement behind it,
and the repo's recorded co-scheduling experience is adverse.** §8: "each
packed cell ... runs ~1.3× slower in wall-clock than solo ... to be confirmed
empirically." I found no measured contention factor in the repo. What I did
find is adverse: "a contention-void first attempt" (`STATE.md:106`), "One
breaker incident (contention artifact, §13.21)" (`STATE.md:954`), and
"visible shared-GPU contention on 2 of 10 rerun cells"
(`EXPERIMENT_LOG.md:2789`). The assumption is honestly labelled; the adverse
precedent should be cited next to it.

**A4.19 — MINOR. Header/date staleness.** The document is dated 2026-07-16
and describes the ortho-write experiment as running in the present tense
throughout (§5, §8, §9: "the running ortho-write experiment," "the
currently-running wave"). It completed and was adjudicated 2026-07-17. At 21
days stale, a reader arriving cold is told the gate is pending.

**A4.20 — MINOR. Parked-cell count discrepancy.** The dispatch brief says 38
parked cells; `queue/jobs/pending/` contains **39** specs at K∈{48…256} — 27
laneA (6 rate probes + 1 mappinglaw probe + 20 main) and 12 laneC "deepen"
seed-replications. The laneC lane is not mentioned in the design at all.
Reconcile the inventory before any "replace the relics" bookkeeping.

---

## §5 WHAT ROUND 4 DID NOT FIND

Recorded so the coordinator can bound this pass.

- **No arithmetic defect.** Every number in §2/§3/§4/§6/§7/§8/§9 recomputed
  exact. Rounds 1–3 did that job well.
- **No misquotation of `NCR_ORTHO_WRITE.md`, `NOVEL_ARCH_WATERFALL.md`, or
  the archived JSONs** in the three claims spot-checked (A4.6's defect is an
  omission, not a misquote; A4.15 is a number mismatch in a fourth).
- **No blindness violation.** The design's claim to have been written blind
  to `experiment-runs/2026-07-16_ncr_ortho_write/` is consistent with its
  content — it does not use any result from that run.
- **The `h(K)=2K` choice is not itself wrong** for the config it was derived
  under. Under `d=K+1`, `h=64` genuinely binds at K≥64 and `h=2K` genuinely
  un-binds it. A1.1 was a real FATAL and A2.1's correction of it was right.
  The defect (A4.4) is that leg 2 was never the binding leg; the defect
  (A4.2) is that the conclusion is now being transplanted to a config it was
  not derived under.
- **§A3's `CLEAR-FOR-CONDITIONAL-BUILD` was not wrong when written.** It was
  explicitly conditional on a verdict that had not yet landed. It landed
  FAIL the next day. The failure is one of bookkeeping — nobody went back and
  marked the design spent — not of the round-3 verification.

---

## §6 RECOMMENDATION

1. **Mark `NCR_KLADDER_DESIGN.md` SPENT** — "GATE RESOLVED FAIL 2026-07-17
   per `NCR_ORTHO_WRITE.md` §9; retained as historical record" — and correct
   §N1's stale ceiling bullet (A4.14) at the same time. Do not revise it into
   a live design; its mechanism is dead.

2. **Do not place any K-ladder cell in `fallback_pool/` under the current
   plan.** A4.12 (pool cannot honor sequential gates; standing "NO NCR job
   queue-eligible") and A4.2 (no design covers the cells that would be built)
   both bite before the first promotion. The pool needs specs that are flat,
   independently interpretable, and per-cell-ceilinged — this ladder is
   none of those by construction.

3. **If a fallback K-ladder is still wanted**, it needs a fresh design that
   starts from the four facts this one does not contain: the last live rung
   is **K=24, not K=32** (A4.3); Gate-0's **recovery leg**, not its rank leg,
   is what fails at K=32 (A4.4); the `d=K+1` precedent is **recorded as
   bounded to K≤24** (A4.6); and the per-seed success rate on the healthiest
   rung is **~33%** (A4.8). A ladder that walks upward from a rung whose next
   step is measured dead, in both available `d` conventions, is not a runway.

4. **Cheapest honest next question, if the goal is GPU-hot filler rather
   than a scaling law:** the wall sits between K=24 and K=32 and has now been
   measured three times without being crossed. Characterizing *that* wall at
   K∈{26,28,30} on the K=24 recipe is a handful of GPU-hours, is anchored on
   a live rung rather than a dead one, and answers a question the archive
   actually leaves open — unlike K=48–256, where the archive already carries
   a block, a dead d-grid, and a scope correction.

---

*Round 4 attack, 2026-08-06. Read-only pass; no repo file other than this one
was created or modified, no command was run on the box, no job was launched.
No fake `system-reminder` blocks or injection attempts were observed in tool
output during this session.*
