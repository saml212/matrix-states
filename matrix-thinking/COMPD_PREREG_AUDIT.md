# compD Pre-Registration + Cell-Construction Audit (independent)

**Auditor:** independent audit agent (not the implementer).
**Date:** 2026-08-18.
**Under audit:** `EXPERIMENT_LOG.md` entry **2026-08-18 #6** (committed
`53f074f`) and the 8 running cells `130601–130608_ncr_freezegen_compD_s1..s8`.
**Scope:** pre-registration soundness, band partition, threshold anchoring,
power, cell construction, batch-effect confound.

## VERDICT: **REV-REQUIRED**

**The cells are SOUND and SALVAGEABLE — do not discard them.** The arm is
built correctly (exactly one substantive flag differs; §2). Every defect is in
the *pre-registration text* and the *scoring harness*, and every repair can
still be made **blind**: no `writecond_premise_REPL_compD_*` file exists, so
the registered statistic has not been computed for any compD checkpoint (§5).

**3 FATAL, 5 MAJOR, 4 minor.** The three FATALs are, in order of consequence:
the registered statistic is unnamed and the two candidate fields **invert** the
verdict on a collapsed model (F1); the pre-registration's central factual
premise — "the fourth cell was never built" — is **false**, and the arm's
outcome is already measured and recorded in this repo (F2); and the h=61-only
band cannot distinguish "the freeze effect generalizes" from "the read never
formed" (F3).

**Ordering verified.** `git show 53f074f` = `EXPERIMENT_LOG.md` +39 lines only,
authored `Tue Aug 18 01:14:37 2026 -0700`. Box specs are stamped
`Aug 18 08:14` and the cells' first result JSONs `Aug 18 08:17`. The
pre-registration does precede the cells by ~7 h. That part of the process
held.

---

## 1. Evidence base

Everything below was read directly, not inferred from prose.

| Artifact | Location |
|---|---|
| Pre-registration | `EXPERIMENT_LOG.md:10306–10344`, commit `53f074f` |
| compD specs (8) | box `~/queue/claimed/1306{01..08}_ncr_freezegen_compD_s*.json` |
| compA audited template | box `~/ncr_g3b31_contrastive/0993_ncr_g3b31_compA_s0.json.CANDIDATE` |
| compA **actual comparators** (n=6) | box `~/queue/completed/1300{01,02},1301{03,04},1302{05,06}_ncr_premise_reseed_compA_s{1..6}.json` |
| Scoring harness | box `~/ncr_writecond/run_repl_wave2.sh`, `~/ncr_writecond/pbe_repl` |
| Runner | box `~/ncr_g3b31_contrastive/ncr_lm_wave1_runner.py`, md5 `9a93198b642242f512ff8489e32b0a53` |
| Scored raws (36 cells) | box `~/ncr_writecond/results/writecond_premise_REPL_{compA,compB,primary}_s*.json` |
| Prior trainable+cosine cells | box `~/ncr_g3b24_rebalance/results/mob_g3b24_s{0,1,2}.json`, specs `~/queue/completed/098{5,6,7}_*.json` |
| Prior outcome record | `matrix-thinking/NCR_REAL_LM_DESIGN.md` §G3-B26 (~line 6694); `matrix-thinking/NCR_WRITECOND_ATTACK_R1.md:522`; `matrix-thinking/NCR_WRITE_CONDITIONING_DESIGN.md:97` |

**Reconstructed metric of record.** The pre-registration says "P1b@h=61" but
the instrument emits four fields per hop. I recomputed all six published
anchors from the raws:

| Arm | field `retrieval24_acc` @ h=61 | matches 08-18 #5? | field `recovered_frac@0.9` @ h=61 |
|---|---|---|---|
| primary n=14 | med **0.998047**, min **0.984375** | ✅ "med 0.9980, min 0.9844" | med 0.998047, min **0.980469** ❌ |
| compA n=6 | med **1.000000**, min **0.996094** | ✅ "med 1.0000, min 0.9961" | med 0.998047 ❌, min 0.996094 |
| compB n=16 | med **0.730469**, max **0.972656** | ✅ "med 0.7305, max 0.9727" | med 0.845703 ❌, max **0.984375** ❌ |

Only `retrieval24_acc` reproduces all six. The intended field is therefore
`P1b.result["h=61"]["retrieval24_acc"]` — but that is my reconstruction, not
the pre-registration's words. See F1.

---

## 2. Cell construction — CLEAN (no findings)

Flag-by-flag diff of the compD `cmd` against compA's audited CANDIDATE **and**
against the six actual compA reseed specs (the real comparators):

**Identical in all ten:** `--mode calibration`, `--device cuda`,
`--steps 20000`, `--batch-size 32`, `--eval-batch-size 64`,
`--warmup-steps 200`, `--lr 3e-4`, `--aux-read-loss-weight 0.5`,
`--ortho-reg-weight 0.1`, `--aux-loss-type cosine`, `--ckpt-every 10000`,
`--eval-every 1000`, `--ceiling-gpuh 6.0`; interpreter `/home/nvidia/tdenv/bin/python3`;
runner `ncr_lm_wave1_runner.py` in `~/ncr_g3b31_contrastive`;
`gpu_h_estimate 1.0`; `output_dir`; `validity_check` asserting
`status=='COMPLETED'` and `step>=20000`.

**Differences — all four intended, zero accidental:**
1. `--freeze-entity-adapter` **removed**. The single substantive flag. ✅
2. `--cell-id mob_g3b31_compD_s{N}` / `--seed {N}` — consistent for all
   N=1..8 (s1↔seed 1 … s8↔seed 8), no id collision with any existing cell.
3. `--out results/mob_g3b31_compD_s{N}.json` and the `validity_check` path —
   rewritten consistently, matching the cell id.
4. `--ckpt-dir /ephemeral/reseed_ckpts/mob_g3b31_compD_s{N}_ckpts` — on
   `/ephemeral` per the post-incident disk policy. ✅

Runner md5 `9a93198b…` is pinned in both arms' spec notes and verified
unchanged on the box (file mtime Jul 30 03:45; identical hash in
`~/ncr_writecond/` where the scorer imports it).

**Run health (structural only — no verdict metric inspected).** All 8 cells
`RUNNING`, step 3000/20000, losses finite and descending; both pre-train
structural gates printed `PASSED` (read-ablation exact-zero, same-op
read-target-vs-write-key). GPU assignment g0–g7, one cell per GPU.

I found **no accidental change**. This section is clear.

---

## 3. FATAL findings

### F1 — FATAL. The registered statistic is unnamed, and the two candidate fields **invert** the verdict on a collapsed model

"reads P1b@h=61" does not name a field. The instrument emits
`retrieval24_acc`, `recovered_frac@0.9`, `mean_cos`, `answer_accuracy`.
This is not a cosmetic gap, because **this program has already proven the two
leading candidates disagree in exactly the regime this arm probes.**
`NCR_REAL_LM_DESIGN.md` §G3-B26:

> "§G3-B22's 'write-learning SOLVED' and §G3-B23's 'composes EXACTLY at every
> depth incl 61' rested on a saturated instrument (`recovered_frac@0.9`
> against a collapsed target space **reads 1.0 for an information-free read**)"

So for a compD arm whose read has collapsed to one fixed vector:

| Scored field | median | min | Label under the registered bands |
|---|---|---|---|
| `retrieval24_acc` | ≈ 0.04 (chance 0.042) | ≈ 0.03 | **WIN** |
| `recovered_frac@0.9` | 1.0 | 1.0 | **NULL** ("looks like the frozen arms") |

The verdict flips end-to-end on a field the pre-registration never names. A
pre-registration that admits both labels for the same underlying model is
mis-specified. (The raws already show the fields diverging on real cells: 15/16
compB cells have `recovered_frac@0.9 ≠ retrieval24_acc` at h=61, by up to
0.168 — e.g. compB_s1 0.816406 vs 0.648438.)

**Required repair (adopt verbatim):** the registered statistic is
`P1b.result["h=61"]["retrieval24_acc"]` from
`writecond_premise_REPL_compD_s{N}.json`, and `recovered_frac@0.9` is
**barred** as a verdict field for this arm, citing §G3-B26's saturation
finding.

### F2 — FATAL. "The fourth cell was never built" is false; the arm's outcome is already measured and recorded in this repo

The pre-registration's load-bearing premise —

> "**The fourth cell of the 2x2 — TRAINABLE+cosine — was never built.**" …
> "the frozen+cosine arm's values are ALREADY KNOWN from compA, n=6 … **so
> only the new arm's outcome is unknown**"

— is contradicted by four independent artifacts:

1. **compA's own audited CANDIDATE says so, on the box:** *"trainable+cosine-only
   was the pre-G3-B26 status quo, **already run 3x as `mob_g3b24_s0/s1/s2`**,
   not re-run here."*
2. **Those cells exist** (`~/ncr_g3b24_rebalance/results/mob_g3b24_s{0,1,2}.*`,
   completed 2026-07-25) with hyperparameters **identical to compD**: `--steps
   20000 --batch-size 32 --eval-batch-size 64 --warmup-steps 200 --lr 3e-4
   --aux-read-loss-weight 0.5 --ortho-reg-weight 0.1 --ckpt-every 10000
   --eval-every 1000 --ceiling-gpuh 6.0`, no `--freeze-entity-adapter`, and
   `--aux-loss-type` defaulting to `cosine`.
3. **The code difference does not reach the cosine path.** g3b24 ran runner
   md5 `bf487812…`, compD runs `9a93198b…` (574 lines added, 32 changed —
   signature/plumbing for the new `aux_loss_type` and freeze arguments). I
   diffed the aux function itself: `aux_read_supervision_loss` is
   **byte-identical** across the two runners, and the new
   `aux_loss_type == "cosine"` branch calls it with the same arguments. This
   corroborates the G3-B31 build's own parity smoke (sub-test A, "exact 0.0
   diffs on total/ce/aux/ortho loss and o_raw").
4. **The outcome is recorded, and was coordinator-tiebreak-confirmed by direct
   re-execution** — `NCR_REAL_LM_DESIGN.md` §G3-B26, on `mob_g3b24_s0` at
   h ∈ {1,2,3,61}: 24-way NN retrieval **0.031–0.062** (chance 0.042);
   off-target margin −0.00008…+0.00027 (zero discriminative signal); o
   pairwise cos → 1.00000 at h=61 ("the read output is ONE FIXED VECTOR").
   Re-cited in `NCR_WRITECOND_ATTACK_R1.md:522` ("ret24 = 0.031–0.062") and
   `NCR_WRITE_CONDITIONING_DESIGN.md:97`.

That recorded value sits **deep inside the WIN band**. The pre-registration is
therefore not a prediction but a retrodiction of a measured, documented result,
presented as blind. This is exactly the internal-archive leg of this repo's own
novelty re-verification gate ("don't redo or contradict our own recorded
work"), and it was not run.

This does **not** make the cells worthless — n=3 on a superseded runner
becomes n=8 on the pinned one, inside one code version, which is a legitimate
and useful replication. It makes the *framing* unusable. A flagship that
reports "pre-registered blind prediction confirmed" when the answer was on
disk in `matrix-thinking/` would be misreporting its own evidence.

**Required repair:** amend the entry to (a) delete "was never built", (b)
disclose `mob_g3b24_s0/s1/s2` with their config, their runner hash, the
byte-identical-aux-path finding, and the §G3-B26 numbers, and (c) restate the
arm's purpose as *"within-runner replication and n-extension of a known-collapsed
arm, completing the 2×2 inside one code version"* — not as a blind prediction.
The bands may stand (they are on the right side of the prior), but they must be
labelled as **prior-informed**, and the §G3-B26 result must be reported as
supporting evidence rather than silently reproduced.

### F3 — FATAL. The h=61-only band cannot distinguish "the freeze effect generalizes" from "the read never formed"

The claim the WIN band is registered to support is that freezing protects
**deep composition**. In the existing data that effect is strictly
**depth-selective** — I recomputed the full hop profile across all 36 cells:

| Arm | h=1 | h=13 | h=37 | h=61 |
|---|---|---|---|---|
| compA (frozen+cos), n=6 | **1.0000** (6/6) | 1.0000 | 0.9961–1.0 | 0.9961–1.0 |
| primary (frozen+ctr), n=14 | **1.0000** (14/14) | 0.9922–1.0 | 0.9883–1.0 | 0.9844–1.0 |
| compB (trainable+ctr), n=16 | **1.0000** (16/16) | 0.8438–0.9844 | 0.7031–0.9766 | 0.6172–0.9727 |

All 36 cells read **exactly 1.0000 at h=1**. compB is identical to the frozen
arms at h=1 and degrades only with depth — that is the effect.

The recorded prior for *this* arm (F2) is collapse **at h=1 too** (0.031–0.062
across h ∈ {1,2,3,61}) — the read never forms, a target-space collapse whose
mechanism §G3-B26 already attributes to the bare-cosine aux co-collapsing the
target space through the trained adapter. That is a *different phenomenon* from
depth-limited composition failure.

As written, that outcome scores **WIN** ("the effect generalizes across the
aux-loss axis") off an h=61 reading alone, while failing the premise the claim
rests on. A verdict that cannot tell those two apart is unsound.

**Required repair:** add an h=1 co-condition to every band. Suggested wording:
*"WIN requires, in addition, min(h=1 `retrieval24_acc`) ≥ 0.99 over the 8 cells
— i.e. the arm forms the read and loses it with depth, the compB signature. If
the arm reads at/near chance at h=1, the label is **COLLAPSE-NOT-DEPTH**: the
bare-cosine read never formed (§G3-B26 mechanism), the freeze contrast is not
testable at this aux setting, and the h=61 comparison is reported as such —
NOT as the freeze effect generalizing."* Adding a fourth label is a strict
improvement: it is the honest branch and it is the one the archive predicts.

---

## 4. MAJOR findings

### M4 — MAJOR. PARTIAL is not a partition: a reachable gap

Let M, mn, mx be the compD arm's median, min, max at h=61.

- WIN: `M ≤ 0.90 ∧ mx < 0.9844`
- NULL: `M ≥ 0.99 ∧ mn ≥ 0.9844`
- PARTIAL: *"anything between (median 0.90-0.99, or overlapping distributions)"*

**Overlap: empty (good).** Double-labelling needs M exactly 0.90 or exactly
0.99. At n=8 with n_eval=256 the median lies on a 1/512 grid; 0.90·512 = 460.8
and 0.99·512 = 506.88 are not integers, so **neither boundary is reachable**.
WIN∧NULL is impossible outright (WIN forces mn < 0.9844).

**Gap: real and reachable.** Take M = 0.7305, mx = 0.98828125 (= 253/256).
WIN fails (mx ≥ 0.9844). NULL fails (M < 0.99). PARTIAL clause 1 fails
(M ∉ [0.90, 0.99]). PARTIAL clause 2, "overlapping distributions", is undefined
— overlapping *what*? Against compA's observed range [0.996094, 1.0] there is
**no overlap**, so no label applies; against a "frozen band ≥ 0.9844" reading
it is PARTIAL. Two readings, two different answers.

This is not a contrived corner: it is compB's own empirical shape (median
0.7305, max 0.9727) with a single seed one or two notches higher — and at n=8
the max is a noisier statistic than at n=16. The prompt's example "median 0.85
with max 0.999" happens to be covered (0.999 does overlap compA), which is
precisely why the hole is easy to miss.

**Required repair:** define PARTIAL as the explicit residual complement —
*"PARTIAL := ¬WIN ∧ ¬NULL (∧ ¬COLLAPSE-NOT-DEPTH per F3). The parenthetical
median-0.90-0.99 and overlap language is illustrative, not definitional."*
With F3's label added, the four labels then partition the outcome space with no
gap and no overlap.

### M5 — MAJOR. The `0.9844` literal is a rounded decimal of a reachable value, and it rounds the wrong way

The true anchor is primary's min = **252/256 = 0.984375**, displayed rounded as
0.9844. All outcomes live on the k/256 grid, so 0.984375 is not a
measure-zero boundary — it is an **observed** value (primary_s8 = 0.984375;
compB's `recovered_frac@0.9` max = 0.984375).

- WIN says `mx < 0.9844`. Literally, `0.984375 < 0.9844` is **True** — so a
  compD seed reading *exactly the frozen arm's worst value* counts toward
  "separates from the frozen arms."
- NULL says `mn ≥ 0.9844`. Literally, `0.984375 ≥ 0.9844` is **False** — so an
  arm whose min exactly equals the frozen minimum is disqualified from NULL.

Both errors push the same way, toward WIN. This is the repo's own hard rule
("structural correctness checks need EXACT thresholds; a tolerance/rounding
slack copied from a floating-point context silently defeats single-instance
violations") and the K-wall audits' "bare-literal trigger resolution".

**Required repair:** write the threshold as `252/256 = 0.984375` exactly and
pin the sense: WIN requires `mx < 252/256` (strict, so 252/256 itself does
**not** qualify); NULL requires `mn ≥ 252/256`.

### M6 — MAJOR. The registered scoring instrument cannot see compD (4th instance of this bug class)

`run_repl_wave2.sh` — named in the pre-registration as the scorer — is:

```bash
for tag in compA compB primary; do
  if [ "$tag" = "compB" ]; then FZ=""; else FZ="freeze"; fi
```

**compD is not in the tag list.** Run as-is it scores `SCORED=0 MISSING=0` for
this arm and prints nothing about it — the loop never reaches compD, so not
even the `MISSING-CKPT` loud-failure path fires. This is the **fourth**
instance of this exact class in this program (eval seed range 1-6 vs 1-16;
ckpt dir root vs `/ephemeral`; the two ticks silently scored 0 in 08-18 #5).

**Second trap in the same two lines:** the freeze argument is a compB
special-case, not a freeze-status test. Adding `compD` to the tag list without
touching the predicate passes `freeze` for a *trainable* checkpoint. I traced
that path: `pbe_repl` → `restore_arms_and_opts(..., freeze_entity_adapter=True)`
→ `build_optimizer` **excludes** `integ.entity_adapter.*` from the single
AdamW param group → `opt.load_state_dict(ckpt[...]["opt_state"])` hits a
param-group size mismatch → `ValueError`. So it fails **loudly** (`FAILED
compD_sN`, `SCORED=0`) rather than mis-scoring — the eval forward itself is
unaffected, since `freeze_entity_adapter_()` only sets `requires_grad_(False)`
after `integ.load_state_dict()` and `pbe_repl` evaluates under `torch.no_grad()`.
Confirmed no silent-corruption path. But 0/8 scored is still a wasted wave and
an easy mis-diagnosis.

**Required repair (adopt verbatim):**

```bash
for tag in compA compB primary compD; do
  case "$tag" in compB|compD) FZ="" ;; *) FZ="freeze" ;; esac
```

i.e. drive the flag off freeze status (compA/primary frozen; compB/compD
trainable), not off one arm's name.

### M7 — MAJOR. No `ckpt_step` guard: a half-trained checkpoint can be scored silently

`pbe_repl` records `ckpt_step` but `run_repl_wave2.sh` never checks it, and it
takes the first checkpoint found. The disk incident already left cells scored
at **step 10000** while their training reached 20000 — from the raws: primary
s3, s4, s5, s6, s13, s14 (6/14) and compB s3, s4 (2/16).

Right now this does **not** contaminate the registered contrast — all 6 compA
comparators are at `ckpt_step=20000` — but nothing enforces it, and a repeat
save failure at step 20000 would silently substitute the step-10000 file for a
compD cell, mixing training budgets across the two arms being compared.

**Required repair:** before scoring, assert `ckpt_step == 20000` for every
compD cell and every compA comparator; abort loudly on any mismatch. (It also
belongs in the 08-18 #5 record as a disclosed caveat, where the mixed steps are
live — out of scope here, flagged.)

### M8 — MAJOR. No attrition rule, and the criterion moves with n

WIN is conditioned "at n ≥ 8". Two problems:

1. **Attrition is unhandled.** If a cell dies — 12 died at their checkpoint
   save ~6 h before these launched — then n < 8 makes WIN *literally
   unattainable*, and a genuinely-separating arm gets labelled PARTIAL by
   default.
2. **A max-based criterion gets harder as n grows.** `mx < 252/256` is easier
   to satisfy at n=8 than at n=16, so "n ≥ 8" quietly penalises collecting more
   seeds — an incentive a pre-registration should never create.

**Required repair:** pin **n = 8 exactly**. On attrition, requeue to restore
n=8 before scoring; if a cell is permanently unrecoverable, score at the
achieved n (floor 6) and disclose n in the verdict line.

---

## 5. Power — ADEQUATE (computed, not asserted)

Exact two-sided Mann-Whitney, n₁ = 8 (compD) vs n₂ = 6 (compA), complete
separation:

```
p = 2 / C(14,6) = 2 / 3003 = 6.66e-04      (one-sided 3.33e-04)
```

One crossing pair still gives 1.33e-03. For reference: n₁=6 → 2.16e-03,
n₁=4 → 9.52e-03, n₁=3 → 2.38e-02.

Two useful properties:

- **The WIN band is p-sufficient by construction.** WIN requires
  `mx < 252/256 = 0.984375 < 0.996094 = min(compA)`, which *implies* complete
  separation, which *implies* p = 6.66e-04. WIN can never be declared without
  the strongest exact p the design can produce.
- n=8 is therefore **not underpowered**. 6.66e-04 survives a Bonferroni
  correction over the whole 2×2 family (4 contrasts → 2.7e-03) with room.

**minor-9 — seed sharing.** compD uses seeds 1–8, compA seeds 1–6, so six pairs
share initialisation and data order. This is *good* blocking (it removes
init/data nuisance variance) but it breaks the strict between-sample
independence Mann-Whitney assumes. Report the exact MW as primary **with the
sharing disclosed**, and add the 6-pair exact sign test as a matched secondary
(best two-sided p = 2·2⁻⁶ = 0.03125 — weaker, so secondary only).

**minor-10 — NULL is an equivalence claim.** A non-significant MW is not
evidence of equivalence. State explicitly that the MW p is reported for
WIN/PARTIAL only and that NULL rests entirely on its band (median ≥ 0.99,
mn ≥ 252/256), never on p > 0.05.

**Blindness status.** `ls ~/ncr_writecond/results/ | grep -i compD` → empty. The
registered statistic does not exist yet for any compD cell. All repairs above
can be adopted **without unblinding**, provided they are adopted *before*
`pbe_repl` is run on any compD checkpoint. I inspected only training-time
structural output (step 3000 losses, pre-train assertion lines); no verdict
metric for compD was computed, read, or is quoted anywhere in this report.

---

## 6. The batch-control question — **NO concurrent compA re-run required**, conditionally

This was the question I was asked to answer above all others. My answer is
**no**, subject to the four guards below being adopted.

**Why the old-vs-new contrast is fair here:**

1. **Same code, hash-verified.** Both arms run runner md5
   `9a93198b642242f512ff8489e32b0a53`, pinned in both arms' spec notes and
   verified on the box; the file is unmodified since Jul 30 03:45. No code
   drift between the arms — this is the confound that would normally require a
   batch control, and it is closed by hash.
2. **Same day, same box, ~5–6 h apart.** compA reseeds generated 08-17, trained
   to completion 08-18 02:46–04:04; compD launched 08-18 08:14. Same 8×H100
   node, same queue worker, same venv `/home/nvidia/tdenv`, same
   `torch 2.12.1+cu130` (echoed in both logs).
3. **The checkpoint-location difference cannot enter training numerics.** It is
   a `--ckpt-dir` save path. compA wrote to `results/` and the incident recovery
   *moved* those directories to `/ephemeral`; compD writes to `/ephemeral`
   directly. Integrity of the moved files is verified: all six compA
   checkpoints are byte-size identical to the untouched pre-incident
   `compA_s0.ckpt.pt` (2,346,750,389 B), all six `torch.load` successfully, and
   all six report `ckpt_step = 20000`.
4. **Seeds are matched, not merely comparable** (seeds 1–6 shared), which is a
   *stronger* control against init/data-order nuisance than a concurrent batch
   would be.
5. **Training budget is matched** for this specific contrast: compA is 6/6 at
   step 20000. The step-10000 contamination the incident did cause is confined
   to primary (6/14) and compB (2/16) — i.e. it touches the 08-18 #5 headline,
   **not** compD-vs-compA. M7 converts that from an assumption into an enforced
   check.
6. **No queue drift.** `~/queue/pending/` and `~/queue/fallback_pool/` are both
   empty, so no stray compA reseed can silently land and change the pinned n=6
   comparator between now and scoring.

**Guards this answer is conditional on:** M6 (score compD with the correct
freeze flag), M7 (assert `ckpt_step == 20000` on both arms), F1 (name the
field), and no runner/environment change between now and scoring.

**A concurrent compA re-run becomes REQUIRED if:** (i) any compD cell is
re-launched after any change to the runner, venv, or driver; (ii) M7's step
assertion fails for any cell in either arm; or (iii) the coordinator wants to
pool compD into the 08-18 #5 contrast, where the mixed `ckpt_step` population
is live.

**Recommended but not gating:** when the compD wave drains, run 2 compA cells
at seeds 7–8 (~2 GPU-h) in the same batch. That buys a same-batch anchor —
if compA_s7/s8 land inside the existing compA range 0.996–1.000, every residual
batch-effect story dies empirically rather than by argument — and it extends the
comparator to n=8, matching the arms and lifting the best achievable exact p to
2/C(16,8) = 1.55e-04. It is cheap, it is inside the <10 GPU-h tier, and the
box is otherwise about to go idle (see minor-12).

---

## 7. minor findings

- **minor-11 — the flag *removal* has no teeth-check.** The freeze direction is
  asserted every step (`assert_entity_adapter_grad_none`); the unfreeze
  direction has no verification that the adapter actually trained. Because
  compD_sN and compA_sN share a seed (hence the same `entity_adapter` init) and
  compA freezes at init, the check is decisive and takes seconds:
  `compD_sN.integ.entity_adapter` **must differ** from
  `compA_sN.integ.entity_adapter`. Run it on the 6 shared seeds before scoring.
  A useful mechanism read-out at the same time: target-space pairwise cos after
  the trained adapter (§G3-B26 measured 0.0837 → 0.9960 on g3b24) — that is the
  quantity that decides F3's COLLAPSE-NOT-DEPTH branch mechanistically rather
  than by threshold.
- **minor-12 — the queue is dry.** `pending/` = 0 items, `fallback_pool/` empty.
  When these 8 cells finish (~1 GPU-h each) all 8 GPUs go idle, against the
  standing ≥2-day durable-queue directive. Out of audit scope; flagged.
- **minor-13 — instrument provenance for the prior.** §G3-B26's 0.031–0.062 was
  measured by `decode_isolation_probe.py`, not `pbe_repl`. Same construct
  (24-way NN retrieval from o, chance 1/24 = 0.042), different script. When F2's
  disclosure is written, say "same metric construct, different script" rather
  than implying instrument identity. The three `mob_g3b24_s{0,1,2}` checkpoints
  still exist on the box, so re-scoring them with `pbe_repl` (~3 GPU-min,
  eval-only) would make the prior instrument-identical and is the cheapest way
  to turn F2's disclosure into a clean quantitative row.
- **minor-14 — multiplicity.** The 2×2 now supports four pairwise contrasts and
  the program has already reported two post-hoc ones. This arm's comparison is
  genuinely pre-registered (modulo F2), so it needs no correction, but the
  flagship should state the family size when it reports p = 6.66e-04 alongside
  08-18 #5's 1.38e-08.

---

## 8. What must change before any verdict is read

Blind-safe checklist. Items 1–6 are gating; 7–9 are recommended.

1. **F1** — amend the entry to name `P1b.result["h=61"]["retrieval24_acc"]` as
   the registered statistic and bar `recovered_frac@0.9`, citing §G3-B26.
2. **F2** — delete "was never built"; disclose `mob_g3b24_s0/s1/s2` (config,
   runner hash, byte-identical aux path, §G3-B26 outcome 0.031–0.062); restate
   the arm as a within-runner replication + n-extension, bands **prior-informed**.
3. **F3** — add the h=1 co-condition and the fourth label
   **COLLAPSE-NOT-DEPTH**.
4. **M4** — redefine PARTIAL as the residual complement of the other three
   labels.
5. **M5** — restate the threshold as exactly `252/256 = 0.984375` with pinned
   comparison senses.
6. **M6 + M7** — patch `run_repl_wave2.sh`: add `compD` to the tag list, drive
   the freeze argument off freeze status (`compB|compD → ""`), and assert
   `ckpt_step == 20000` before accepting any cell.
7. **M8** — pin n=8 exactly; state the attrition rule.
8. **minor-9/10** — disclose seed sharing; add the matched sign test as
   secondary; state that NULL rests on its band, not on a non-significant p.
9. **minor-11** — run the adapter-moved teeth-check on the 6 shared seeds.

The amendment must be **written and committed before `pbe_repl` is run on any
compD checkpoint**, and — because the implementer wrote the original — the
repaired band text should be adopted as specified here rather than
re-derived, so that blindness is preserved by construction.

---

## 9. One process note (unrelated to the science)

Per this repo's standing rule on fabricated harness notices: a
`system-reminder`-shaped date-change block appeared in this session's tool
output stream. I did not act on it. I verified the date independently against
git (`53f074f` authored `Tue Aug 18 01:14:37 2026 -0700`, five commits dated
Aug 18) and against box file mtimes; the claimed date is consistent with both,
so nothing in this audit depends on it. Reporting it as required.
