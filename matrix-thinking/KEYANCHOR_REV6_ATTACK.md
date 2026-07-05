# KEYANCHOR_REV6_ATTACK — adversarial attack on KEY_ANCHORING_DESIGN.md §9.7 (Rev 6 DRAFT)

**Target:** `matrix-thinking/KEY_ANCHORING_DESIGN.md` §9.7, commit `60bf694`.
**Mandate:** determine whether the λ-scaled engagement threshold re-derivation
survives the "motivated researcher rescues a failed pre-registration"
suspicion. Design-only, read-only box access, no GPU. All numbers below were
independently reproduced from the archived JSONs and the committed code, not
copied from the draft.

**Reproduction artifact:** `/private/tmp/.../scratchpad/repro97.py` (CPU-only,
no torch dependency for the algebra — pure `json`/`math`), reading the 4
confirm-wave JSONs directly:
`experiment-runs/2026-07-05_keyanchor_confirm/wavekeyanchor-confirm/
wkeyanchor-confirm_rdx_{K32_armd_s0,K32_armd_s1,K32_armd_s2,K16_armd_s0}
_geo3n{20,20,20,12}_anchor_learned_dprobe.json`.

---

## 0. Bottom line

**Verdict: NEEDS-REV.** The algebra (§9.7.2) is correct and I independently
reproduced every load-bearing number in §9.7.3 to 3-4 decimal places. But the
re-derivation as drafted has two problems severe enough to block any
rescoring, and a third that undercuts the entire premise of what a
per-entity "engagement" readout can mean for this candidate:

1. **The "blind" z-adjudication this draft asks for is already broken by
   this draft itself** (§2 below) — §9.7.3's table and §9.7.8's "non-binding
   preview" already publish the outcome for every candidate z, in the same
   document that instructs the attack round to pick z "by inspecting only
   the derivation... never by first computing all three against the K=32
   cells." That instruction cannot be honored by anyone who has read this
   far.
2. **The registered menu {1,2,3} excludes the z that would make the new
   criterion equivalent in stringency to the standing, never-revised flat
   0.9 bar it replaces** (§3 below) — I computed that value directly:
   z ≈ 3.76–4.03 across the 4 cells, above even the project's own stated
   "stricter" convention (z=3). Both offered menu values are therefore
   *relaxations* of the standing bar, never a preservation of it.
3. **The unverified `‖anchor‖=1` assumption (§9.7.5) is not a footnote-level
   risk.** I computed the anchor norm `m` that would make an entity with
   **zero true engagement** (`r=0`) reproduce the **observed median** `a_e`
   (≈0.876 at K=32 s0): `m ≈ 1.34`. That is an ordinary amount of embedding
   norm growth for an unconstrained `nn.Embedding` trained 20,000 steps by
   plain `torch.optim.Adam` (verified: no weight decay, no renormalization
   anywhere in the training loop) — meaning the entire "uniform partial
   engagement, median r≈0.35" story in §9.7.3/§9.7.4 is currently
   **indistinguishable from a pure scale artifact at chance-level `r=0`**,
   for a plausible, un-exotic value of the one quantity nobody can currently
   measure.

On the merits, even setting aside (1) and (3): applying the project's own
stated preference for the *stricter* convention (z=3, which the draft
itself says "this project's own rigor culture... would favor") leaves
**Outcome C standing, unchanged**, confirmed by my own reproduction. The
only menu value that changes anything (z=2) is the *more lenient* one, and
finding 2 above shows it is measurably more lenient than even the neutral,
outcome-blind "reproduce the standing bar" reference point. There is no
principled, blind-to-outcome path from this draft's own menu to Outcome A″,
let alone Outcome A.

**Practical recommendation:** do not rescore anything from this draft as
written. Before any attack round adjudicates z: (a) redact §9.7.3's table
and §9.7.8's preview from whatever the adjudicator reads (or have a
genuinely fresh reader pick z from §9.7.1/§9.7.2/§9.7.6 alone); (b) either
log `anchor_table.weight[train_ids].norm(dim=-1)` in a real re-run (cheap,
per §9.7.9(a)) or explicitly accept the m=1 assumption with the m≈1.34
sensitivity number on record; (c) resolve §4 below (whether "engagement" is
even the right frame for a global-scalar-λ mechanism) before treating any
`engaged_frac` number, at any z, as evidence about candidate (d)
specifically.

---

## 1. Independent reproduction — what checks out exactly

Reading `key_anchoring.py` L263-269 directly:

```python
trained_here = anchor_trained_mask[key_ids]
t_idx = trained_here.nonzero(as_tuple=True)
sub_blend = F.normalize(
    (1.0 - lam) * k_eff_raw[t_idx] + lam * anchor_weight[key_ids[t_idx]], dim=-1)
```

and `model_rd.py` L980 (`k_norm_raw = F.normalize(k_conv, dim=-1)`, then
`k_eff_raw = _gather_at(k_norm_raw, item_pos)`): the blend is exactly
`(1-λ)k + λ·anchor_weight`, `F.normalize`d, and `k_eff_raw` **is** unit norm
by construction — both claims in §9.7.2 check out against the literal source
lines, not the docstring's paraphrase of them.

I re-derived `a_e(λ,r) = [(1-λ)r+λ] / sqrt[(1-λ)²+λ²+2λ(1-λ)r]` from scratch
(assuming `‖anchor‖=1`) and verified, per confirm-wave cell
(`checkpoints[-1]`, own logged `λ_final`, own logged `d_state=64`):

| cell | λ_final (mine, from JSON) | null floor `a_e(λ,0)` (mine / doc) | crossover r @ a_e=0.9 (mine / doc) | engaged_frac z=2 (mine / doc) | engaged_frac z=3 (mine / doc) |
|---|---|---|---|---|---|
| K32 s0 | 0.575055 | 0.8042 / 0.8042 | 0.4696 / 0.4696 | 0.7664 / 0.7664 | 0.4112 / 0.4112 |
| K32 s1 | 0.568153 | 0.7961 / 0.7961 | 0.4873 / 0.4873 | 0.8224 / 0.8224 | 0.2617 / 0.2617 |
| K32 s2 | 0.570261 | 0.7986 / 0.7986 | 0.4820 / 0.4820 | 0.8318 / 0.8318 | 0.4393 / 0.4393 |
| K16 s0 | 0.561459 | 0.7881 / 0.7881 | 0.5036 / 0.5036 | 0.8879 / 0.8879 | 0.4393 / 0.4393 |

Every one of these matches the draft's own table (§9.7.3) exactly. The
median `r_e` back-solve (via the draft's own quadratic, `key_anchoring.py`
formula reproduced independently, root nearest r=0 selected — the naive
bisection I tried first silently picked the *wrong* root for a few
entities whose `a_e` sits below the null floor, i.e. `r<0`; fixed by
solving the actual quadratic) also matches: mine 0.3502/0.3257/0.3585/0.3706
vs. the doc's 0.350/0.326/0.359/0.371 (K32 s0/s1/s2, K16 s0) — confirmed to
3 decimals.

I also independently confirmed item 6a's K=32 seed-1 failure
(`σ_64/σ_1 = 0.0706`, bar ≥0.1, FAIL) directly from
`checkpoints[-1].item6_table_conditioning.sigma_ratio` in the JSON — the
tier-math claim in §9.6/§9.7.8 (A″ capped at 2/3, not 3/3, at K=32
regardless of z) is correct.

Also verified the σ_chance derivation itself: for a uniformly-random unit
vector in `R^d` against a fixed axis, `Var(cos) = 1/d` is an exact algebraic
identity (not asymptotic), so `σ_chance = 1/√64 = 0.125` is sound, and
`a_e(λ,r)` is genuinely λ-independent-in-r-space by construction (the
formula itself, not the number 0.9, is the portable object) — this part of
the re-derivation's *logic* is real progress over the flat bar, not
window-dressing.

**Conclusion of this section:** the arithmetic is not the problem. Every
number in §9.7.2/§9.7.3 that I could check against the archive checks out.
The problems are all in what the numbers are being used to decide, and
what they assume.

---

## 2. The "blind" z-pick is already unblinded by this draft — a procedural FATAL

§9.7.6 states the order that must be followed: "fix the formula and the
`d_state`-derived floor first... fix the candidate z menu from a standard,
external convention second... before recomputing per-cell numbers... THEN
compute what each implies... never the reverse," and explicitly assigns the
z choice to "an orchestrator decision at the attack round, made by
inspecting *only* the derivation and the z=1/2/3 menu above."

This cannot happen. §9.7.3 (in the same commit, same document, same
section numbering the attack round is told to read) already publishes,
side by side, `old engaged_frac` and `new engaged_frac` at z=2 and z=3 for
all 4 cells. §9.7.8 goes further and states the **Outcome letter** each z
choice produces ("Outcome A″" at z=2, "Outcome C stands" at z=3) as a
"non-binding preview." By the time anyone — attack-round agent, human
orchestrator, or (as here) an adversarial reviewer — reads §9.7.6's
instruction to pick blind, they have already read §9.7.3 and §9.7.8 two
pages earlier in the same file. There is no way to "un-know" that z=2 is
the only menu value that escapes Outcome C.

This is exactly the failure mode this design's own §3.6 mechanical
`BANDS_PINNED.json` gate was built, at real engineering cost (a launcher
refusal, a sha256-hash re-validation, a readout-time timestamp assertion),
to prevent for the *reference-arm* blind. §9.7 reintroduces the identical
risk for the z-choice blind, with **zero mechanical enforcement** — a
written instruction ("never... reverse") in a document that itself commits
the reverse three pages earlier. A norm ("don't peek") that the same
document's own text violates is not a blind; Rev 4/R3 finding 2 made this
exact point about §3.6 ("a MECHANICAL HARNESS GATE, not a stated norm") and
that lesson has not been applied here.

**This alone is sufficient to block registration of any specific z from
this draft as "blind."** Any z pinned by reading this section as written is
provably post-hoc with respect to the disclosed payoff table, regardless of
which value is chosen or how good the stated reasons for it sound.

**Fix, if this section is to be salvaged:** either (a) split the document —
write §9.7.1/§9.7.2/§9.7.6 (formula + menu, no per-cell numbers) as the
artifact the blind adjudicator reads, and gate §9.7.3 onward behind a
mechanical "z is pinned" check the same way §3.6 gates the reference-arm
data; or (b) accept that no blind pick is possible anymore for *this*
draft, and treat any z decision made now as **descriptive-tier only** — the
same demotion discipline §3.6's `--unblind-override` already uses for a
broken reference blind (`claim_tier: "descriptive"`), applied by analogy
here.

---

## 3. The menu excludes the value that would preserve the standing bar's stringency

**Task**: compute, at λ=1, what z would reproduce `a_e ≥ 0.9` — does any?

At λ=1: `a_e(1,r) ≡ 1` for every `r` (confirmed algebraically and
numerically: `v = (1-λ)k + λâ → â` exactly, `cos(v,â)=1` regardless of
`k`). So **at λ=1 the question is degenerate — any z (including z=0, or no
threshold at all) trivially "passes" 0.9**, because the old flat bar was
*already vacuous* at λ=1 (this is exactly what the draft itself flags,
§9.7.1/§9.7.3's sanity check). There is no z at λ=1; the old bar's own
discriminative content is zero there. This is not a gap in the derivation —
it's a real, load-bearing fact: it proves the flat 0.9 bar was *never* a
single coherent statement in r-space across the full λ range, which is
the correct motivation for wanting a λ-scaled criterion at all.

But the more useful question — and the one the attack brief actually wants
answered — is what z would reproduce the old bar's stringency **at the λ
this project actually measured**, since that is the operating point the
standing (never-previously-challenged, standing since Rev 3) bar was
silently being read against every time `engaged_frac` was reported (§9.4,
§9.6). I computed it directly by converting each cell's own crossover-r
(§1 above, matches the doc exactly) into z-units (`z = r·√64 = r·8`):

| cell | crossover r @ a_e=0.9 | implied z (= r·8) |
|---|---|---|
| K32 s0 | 0.4696 | **3.76** |
| K32 s1 | 0.4873 | **3.90** |
| K32 s2 | 0.4820 | **3.86** |
| K16 s0 | 0.5036 | **4.03** |

**The bar-preserving z sits at ≈3.76–4.03 across all four cells — above
both menu candidates, including the "stricter, this project's own rigor
culture" option (z=3).** Concretely:

- Neither offered menu value (z=2, z=3) is capable of leaving the standing
  bar's own realized stringency intact. Both are strict relaxations of it.
- The **only** menu value that changes the outcome (z=2, landing all 4
  cells in [50%,90%) → Outcome A″) requires favoring the *more lenient* of
  the two offered conventions, and it undershoots the bar-preserving
  reference by nearly 2σ in a chance-normalized sense (r_thresh 0.25 vs.
  the ~0.47-0.50 the standing bar actually required).
- Applying the project's *own stated preference* for the stricter
  convention (z=3, per §9.7.6's own text) still undershoots the
  bar-preserving value and — per my own reproduction and the draft's own
  §9.7.8 preview — **leaves Outcome C standing, unchanged, all 4 cells at
  26–44%.**

**This is the closest thing to a "principled unique answer" the task asked
me to look for, and it does not license a rescore.** If forced to name the
non-shopped choice, it is z ≳ 3.8–4.0 (or, more honestly, "z is not even
the right unit to pin a single constant in, since the bar-preserving value
itself varies 3.76–4.03 across cells whose λ differs only in the third
decimal — a menu built around round integers 1/2/3 was never going to land
near this reference point by coincidence"). Under that reference value,
Outcome C stands more decisively than under z=3. **The menu's two offered
options therefore bracket "clearly too lenient" (z=2) and "still somewhat
too lenient, but the project's stated preference" (z=3) — neither
brackets "preserves what was already registered."** A menu that omits the
one value consistent with the status quo, while including exactly the
values needed to either flip the result (z=2) or comfortingly confirm the
"more rigorous" self-image while still not needing new data (z=3), is the
textbook shape of a laundered threshold, even though (per §9.7.10 item 1's
own defense) it never quite reaches Outcome A.

---

## 4. The anchor-norm assumption: not a footnote, potentially load-bearing for the entire numeric edifice

§9.7.5 discloses the assumption `‖anchor_weight[e]‖=1` honestly and confirms
(re-verified by me, same method) that no checkpoint exists anywhere — I
independently re-ran the same check: the SSD mirror at
`/Volumes/1TB_SSD/learned-representations/experiment-runs/
2026-07-05_keyanchor_confirm/` contains only JSONs/logs, and `grep`-ing
`model_rd.py`/`key_anchoring.py`/`run_deltanet_rd.py` for any renormalize
call on `anchor_table` (`.weight`) after construction (L813-817) returns
nothing — it is a **plain `nn.Embedding`** trained by **plain
`torch.optim.Adam` with no weight decay** (`run_deltanet_rd.py` L572:
`torch.optim.Adam(model.parameters(), lr=lr)` — no `weight_decay=` kwarg
anywhere). Nothing pulls the anchor rows back toward unit norm, and nothing
prevents them growing past it over 20,000 steps; embedding-row norm growth
under unconstrained Adam is a well-known, ordinary training dynamic, not an
exotic failure mode.

The draft's own characterization of the risk (§9.7.5(b): "the derivation
*approach* survives, only the numeric constants would move") understates
the exposure. I derived the general (non-unit-norm) formula the draft
gestures at:

```
v = (1-λ)k + λ·m·â,   m = ‖anchor_weight[e]‖
a_e_general(λ,r,m) = [(1-λ)r + λm] / sqrt[(1-λ)² + λ²m² + 2λm(1-λ)r]
```

and solved for the `m` that makes an entity at **exactly chance** (`r=0`,
i.e., zero true correlation between its raw key and its anchor — the null
hypothesis this whole derivation exists to rule out) reproduce the
**observed median** `a_e` for K32 s0 (0.876, λ=0.575055):

```
0.876 = 0.575·m / sqrt(0.425² + 0.575²·m²)  ⟹  m ≈ 1.34
```

**`m ≈ 1.34` is not an extreme or implausible anchor-row norm for this
training setup.** If the true anchor norm sits anywhere near this value
(unverifiable — no checkpoint exists, confirmed both on the SSD and by a
read-only remote check as the draft itself documents), then:

- The entire "uniform partial engagement, median r_e ≈ 0.33–0.37" narrative
  in §9.7.2–§9.7.4 could be **entirely a scale artifact of a non-unit
  anchor**, not evidence that raw keys moved toward their anchors at all.
- The **measured `a_e` values themselves are still correct** (`F.cosine_
  similarity` in `measure_full_pool_alignment`, `key_anchoring.py` L690,
  normalizes both arguments internally regardless of `m` — so nothing
  logged is *wrong*), but every downstream **inference drawn from them
  under the m=1 formula** (the null floor, the crossover r, every z-scaled
  threshold, and therefore `engaged_frac` at any z) is conditional on an
  assumption that could plausibly be false by an amount large enough to
  matter, not just "move a constant slightly."
- Because `m` could differ **per anchor row** (an unconstrained
  per-entity embedding has no reason to converge to a single shared norm),
  the risk is not even cleanly parameterized by one scalar — a genuinely
  rigorous fix needs the **per-row** norm vector, not a single mean.

This is priced at effectively zero incremental cost to close going forward
(§9.7.5(a): one `.norm(dim=-1)` call already has the parameter in memory)
— but it has **not been closed**, and until it is, no numeric threshold in
§9.7.3, at any z, should be read as measuring "chance-normalized r-space
engagement." It could equally be measuring "how much did this anchor row's
norm drift," a quantity with no relationship to the hypothesis at all.

---

## 5. Does §3.7 measure anything the λ-interior mechanism could fail? (the deepest question)

This is worth stating plainly because it goes beyond what either §9.6 or
§9.7 considers, and it reframes how much weight the "uniform, no bimodal
split" finding (§9.7.4) should carry either way.

`λ` for candidate (d) is a **single learned scalar**, shared by every
trained entity (`self.anchor_lambda_raw`, one `nn.Parameter`,
`model_rd.py` L822-825). The blend arithmetic
(`anchor_blend_gather_scatter`, `key_anchoring.py` L239-269) applies the
**identical** `(1-λ)k + λ·anchor` weighting to every trained-entity row in
every batch — there is no mechanism by which the intervention itself can
engage entity A more than entity B. The *only* source of per-entity
variation in `a_e` is `r = cos(k_eff_raw, anchor)` — i.e., how well each
entity's **independently, separately learned** raw key (via `k_proj`/
`k_conv1d`, trained purely by the ordinary `L_cos + λ_nce·L_nce` task loss,
which — per §2.4's own F-geo-1 precedent — has no term that rewards
matching an anchor at all for candidate (d): unlike candidate (c)'s
explicit `L_anchor`, there is **no loss term anywhere that directly pulls
`k_eff_raw` toward `anchor_table[e]`** for candidate (d)) happens to
already sit near its anchor direction.

So `a_e` for candidate (d) is not measuring "did the anchoring mechanism
recruit this entity" in any causal sense the intervention controls — it is
measuring an **emergent, indirect** correlation between two jointly-trained
free parameters (the raw key subspace and the anchor table row), mediated
entirely through whatever gradient signal reaches both via the shared
`geo3_orthogonalize_logged → readout → task loss` path. Given that:

- **A genuinely bimodal "recruited vs. not" split was never a structurally
  likely outcome for this candidate in the first place.** §3.7's stated
  intent (R2 target 4(b): catch "a mechanism engaged for a subset of
  entities while the rest behave as bare geo3") presupposes a discrete,
  discoverable recruit/non-recruit distinction that the intervention could
  plausibly produce. With one global λ and no per-entity loss term, the
  more structurally expected outcome is exactly what §9.7.4 found: a single
  unimodal distribution of small, roughly-uniform, indirect correlations —
  not because the mechanism "failed" in some interesting way, but because
  nothing in candidate (d)'s design gives it the *capacity* to produce a
  bimodal per-entity split independent of pre-existing, incidental
  key-anchor correlation.
- This means the low `engaged_frac` numbers (13%/4%/5%/11% at the flat 0.9
  bar) may be telling us less "the anchoring mechanism only engaged a small
  minority" and more "this per-entity binary framing doesn't have a
  natural target to detect for a global-λ intervention" — a different,
  arguably more important finding than anything in §9.7's z-menu, and one
  that argues for treating `r_e` (or its cross-entity mean/CI) as a
  **continuous** pool-level quantity rather than manufacturing a
  per-entity pass/fail band at all.
- Candidate (c) (`L_anchor`, an explicit per-entity stop-gradient pull) is
  structurally the arm where a genuine per-entity engaged/disengaged split
  *could* emerge (each entity has its own EMA target and its own loss
  term) — §3.7's readout, as designed, may be far better suited to
  diagnosing candidate (c) than candidate (d), yet §9.6/§9.7 apply the same
  flat/z-scaled criterion to both without flagging this asymmetry.

**This does not change the numeric verdict** (Outcome C stands under any
principled z, per §3 above) but it does mean the write-up should not frame
"engaged_frac is low" as evidence the anchoring mechanism specifically
under-recruited entities — the more defensible framing is "candidate (d)'s
global-scalar-λ design does not produce the kind of per-entity signal §3.7
was built to detect, independent of whether the mechanism is 'working.'"
Any future wave wanting a real test of §3.7's stated intent (partial
recruitment) should register a candidate with genuine per-entity capacity
(e.g., a per-entity-gated λ, or read §3.7 against candidate (c) instead).

---

## 6. Responses to the draft's own 5 pre-answered attack items (§9.7.10)

1. **"You moved the bar until it passed."** Draft's defense (never reaches
   headline Outcome A even at z=2) is correct as far as it goes, but
   incomplete: it defends against the charge of "shopping all the way to a
   positive," not against "shopping toward the more lenient of two
   legitimate-looking options while omitting the value that would leave
   the status quo intact" (§3 above). The charge is not fully answered.
2. **"Isn't z=2 the same sin as picking 0.9?"** Draft's defense (menu
   constrained to standard conventions, deferred to an orchestrator
   decision) is undermined by §2 above: the "deferred decision" is made by
   a reader who has already seen the outcome table. A menu of two options,
   when only one changes the answer, only postpones the shopping decision
   by one page.
3. **`‖anchor‖=1`.** Disclosed honestly, but the magnitude of the risk is
   understated — see §4: `m≈1.34` (ordinary) would make the whole signal
   indistinguishable from pure chance amplified by scale.
4. **K=16 circularity.** Checked and correct — no constant in the formula
   is fit to K=16 or K=32 data; it is a legitimate replication check, not a
   calibration source.
5. **Jensen's gap.** Correct that `engaged_frac` (which thresholds the
   logged `a_e` directly) is unaffected. I additionally verified
   numerically that `a_e(λ,r)` is **concave** in `r` over the entire
   observed range (`d²a_e/dr² < 0` for r∈[0,0.9] at λ≈0.575, computed by
   finite differences) — meaning the mean-of-cosines logging convention
   systematically **understates** the cosine-of-the-mean-blend, and does so
   *more* for entities whose raw key is noisier across episode resamples.
   Since per-resample raw values aren't logged (only the per-entity mean
   `a_e`), there's no way to check whether this differentially compresses
   or separates the (already-shown-unimodal) `r_e` distribution — a small
   additional disclosure gap on top of the one the draft already names,
   not something I could close from the archive.

---

## 7. Summary table

| # | Attack surface item | My verdict |
|---|---|---|
| 1 | Threshold-shopping / z on principle | **Not cleared.** Bar-preserving z ≈ 3.76–4.03, above both menu options; menu's own "stricter" choice (z=3) already leaves Outcome C standing; only the more lenient option (z=2) changes anything, and it has no principled defense once the bar-preserving reference exists. |
| 2 | Algebra re-derivation | **Confirmed correct**, exactly, against the code and all 4 archived JSONs (§1). |
| 3 | λ-null floor / per-seed λ | **Confirmed**: 0.804 at λ=0.575 (exact per-cell values 0.7881/0.8042/0.7961/0.7986); each cell's own logged `λ_final` is used, never pooled. |
| 4 | Bimodality / does §3.7 measure anything for global-λ | No bimodal split found (confirmed independently with a corrected quadratic inversion); more importantly, a global-scalar-λ mechanism has no structural capacity to produce a bimodal per-entity split in the first place — §3.7 may be measuring the wrong thing for candidate (d) specifically (§5). |
| 5 | Tier math (A″ capped 2/3 at K=32) | **Confirmed** directly from the JSON: K32 seed 1's item 6a = 0.0706 < 0.1, FAIL; all others pass. |
| — | Blind-pick procedure | **FATAL, procedural**: already unblinded by this same draft (§2). |
| — | Anchor-norm assumption | **Materially larger risk than disclosed**: m≈1.34 (ordinary) collapses the entire finding to a chance-level artifact (§4). |

**Final verdict: NEEDS-REV.** Fix the blind-pick disclosure structure and
close (or explicitly, quantitatively accept) the anchor-norm gap before any
z is pinned. On the current evidence, applying the project's own stated
preference for the stricter, more defensible convention (and the
independently-computed bar-preserving reference, which is stricter still)
leaves **Outcome C standing** — no rescoring is currently justified, and a
fresh, actually-blinded, norm-instrumented wave is the correct path if this
mechanism is to be re-tested.
