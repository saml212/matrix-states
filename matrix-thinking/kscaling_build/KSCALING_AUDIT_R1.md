# K-SCALING BUILD — INDEPENDENT AUDIT ROUND 1

**Verdict: REV-REQUIRED** — 1 FATAL, 6 MAJOR, 6 minor.
**Auditor:** independent audit agent (not the implementer), 2026-08-21.
**Target:** build at `62bb64b`, recorded at `002a6c5`.
**Method:** every claim re-executed. Nothing below is taken from the build report.
**Box state on exit:** all pinned + build md5s unchanged, `~/queue/pending/` = 0,
`~/queue/claimed/` = 0, all 8 GPUs idle. Nothing was queued, launched, or modified.
Audit scratch (`/tmp/audit_r1/broken*`, `/ephemeral/kscaling_AUDIT`,
`/ephemeral/kscaling_smoke_audit`) removed.

Ceremony tier is correct: 27.6 GPU-h ⇒ audit **+** pre-launch resource/placement
red-team. This document discharges the audit leg only.

---

## 0. Verdict summary

| # | Finding | Severity |
|---|---|---|
| **F1** | K=24 anchor is **unrunnable** — `RUNNER_TAG` bump rejects all 55 cells of record. The pre-registered PRIMARY curve (all six K) cannot be produced. | **FATAL** |
| **M1** | §7.3 pooled 15-v-15 Mann–Whitney is a blocked design analyzed unblocked. Measured: a *unanimous* per-K ordering gives p = 0.361; a single-K artifact gives p = 0.016. | **MAJOR** |
| **M2** | Primary band `margin ≥ 0.90` is K-dependent by 0.052 raw accuracy, monotone in K, aligned with the axis under test. Manufactures a frontier. | **MAJOR** |
| **M3** | Specs 0100↔0134 and 0101↔0137 collide on `--out`/`--ckpt-dir`/`--cell-id`; the runner skips them, so the gate cell silently becomes the K=32 seed-0 curve point. | **MAJOR** |
| **M4** | Smoke item B is a **vacuous** negative test (`expect_substr=""` matches any error). Proven by injection. | **MAJOR** |
| **M5** | The sweep's "double gate" is prose only — no dependency exists in the spec schema, and 8 workers are live. | **MAJOR** |
| **M6** | Calibration gate can declare `FRONTIER-AT-K*=32` from **n=1 seed**. | **MAJOR** |
| m1–m6 | See §8. | minor |

**Nothing in the build produces wrong numbers on the 30 sweep cells.** The FATAL and
the MAJORs are about *what the numbers can be used to claim*, plus two silent-no-op /
vacuous-guard defects. The compute plan itself is sound.

---

## 1. What was verified clean (re-executed, not accepted)

These are positive findings. The build is unusually strong on the mechanical axes.

**Pinned-file integrity.** `~/ncr_g3b31_contrastive/ncr_lm_wave1_runner.py` =
`9a93198b642242f512ff8489e32b0a53` ✓ and `ncr_lm_wave1_smoke.py` =
`bc105af69661e488ff95f5046e2bcd8a` ✓, before and after the entire audit. All five
repo-mirror md5s match the design §5 table **and** the box copies exactly.

**Ladders (audit item 3) — re-derived independently.** I wrote a fresh derivation
from the §4.2 *prose* without importing `kscaling_config`. It reproduces
`LADDER_TABLE` and `FIXED_DIST_TABLE` at all six K, and **every constraint holds with
zero violations**: no residue 0; no residue in the train set {1,2,3}; six pairwise-
distinct residues; squaring profile exactly (2,3,4,4,5,5); strictly increasing depth;
top rung at residue K/2. §4.1's pinned-ladder table also reproduces exactly (identity
at K=12/20, train residue at K=12/20/28, silent dupes at K=16/24/28/32 — including the
new K=24 disclosure, 29 ≡ 5 mod 24).

> **Answer to "is residue 4 collision-free at every K?" — No, and that is correct by
> construction.** `h_fix` sits at residue 4 and ladder rung 1 is `h=4`, so they share a
> residue at *every* K. The build discloses this, keeps `h_fix` out of `DEEP_LADDER`
> and out of the distinctness assert, and carries it as its own labelled probe (patch
> R3 puts it in a separate `fixed_dist` block — verified in the output JSON). This is
> not a violation. It is also a free bonus control: rung 1 and `h_fix` share a ground
> truth (π⁴) at squaring counts 2 vs 5, so their difference is a *within-cell* fp-depth
> reading. The K=24 anchor already shows it: 1.0000 at h=4 vs 0.9961 at h=52.

**K=24 anchor byte-identity (audit item 2) — re-run by me, not taken on faith.** I
built the K=24 document under the **pinned** graft and under the **patched** graft in
separate processes, same seed, and compared field by field at both a train hop (h=1)
and the new `h_top` (h=36):

```
h1/doc  h1/key_pos  h1/val_pos  h1/query_key_col  h1/query_mark_col
h1/entity_ids  h1/tgt_slot  h1/answer_token   ... and the same 8 fields at h=36
                                        ALL identical=True   (doc shape (8,175))
BYTE-IDENTICAL ALL FIELDS: True
```

Patch S2 changes `H_test`/`H_extra` at every K, but `sample_batch_rd` takes `hop_set`
explicitly and never reads them, so the document is untouched. **The byte-identity
claim for K ≥ 20 holds.**

**Smoke (audit item 4) — re-run on real CUDA at K=12 and K=32.** Reproduced exactly:
K=12 → 12 PASS / 0 FAIL / 0 N/A; K=32 → 11 PASS / 0 FAIL / 1 N/A. Matches the
committed `smoke_results/` JSONs.

**Negative tests C and D are DISCRIMINATING — proven by breaking them.** I copied the
build, deleted the pairwise-distinctness assert and neutered the identity assert, and
re-ran:

```
[PASS] B_NEG_pinned_ladder_rejected
[FAIL] C_NEG_silent_residue_collision_rejected      <- flipped when its guard was removed
[FAIL] D_NEG_identity_residue_rejected              <- flipped when its guard was removed
[PASS] E_NEG_train_residue_rejected
=== K=12: 10 PASS / 2 FAIL -> FAIL ===
```

**Parameter formula vs measured (audit item 4), at two K:**

| K | d | measured NCR params | `40h²+4dh+46h+d` | integ | `2·768·d` | total/arm |
|---|---|---|---|---|---|---|
| 12 | 13 | 170,125 | 170,125 ✓ | 19,968 | 19,968 ✓ | 97,809,805 |
| 32 | 33 | 175,265 | 175,265 ✓ | 50,688 | 50,688 ✓ | 97,845,665 |

Matches design §5.1 exactly. Param spread across the K range is 0.037% — the curve is
not a capacity curve in disguise.

**Checkpoint resume bit-identity (audit item 4):** `max|Δlogit| = 0.0` at K=12 and
K=32. Not approximately — exactly zero.

**Scorer (audit item 5) — exercised end-to-end on a real trained checkpoint.**
Matched pool built from the checkpoint's own recorded seed (`pool_seed = ckpt_seed = 1`,
`matched: true`, source `checkpoint`); `chance = 1/24 = 0.0417` and band
`[0.0042, 0.0791]` reproduce §6's table; P1b/P0 labelled with their regime meanings;
role labels (`train` / `ladder` / `ladder_top` / `fixed_dist_control`) all correct;
`self_check = PASS`; fail-loud on an unloadable checkpoint confirmed (exit 5). The
`d == K+1` guard and the `--k`/`NCR_K` tripwire both fire on demand (§8.2 reproduced).

**End-to-end literal spec command at K=28** — a K the builder did **not** run end to
end (they used K=12). I took spec `0128`'s `cmd` verbatim, changed only steps/eval
cadence/output root, and ran it:

```
[kscaling] K=28 d_ncr=29 chance=0.0357 ladder=(4,8,16,17,33,42) h_top=42 (residue 14 == K/2) ...
read-ablation exact-zero check PASSED (pre-train, max_abs_diff=0.00e+00)
step 1/3 full_graft_loss=11.1535 ... step 3/3 full_graft_loss=11.0231
eval computed at step 1 / 2 / 3      <- build_attribution ran 3x, no KeyError
read-ablation exact-zero check PASSED (post-train, max_abs_diff=0.00e+00)
COMPLETED at step 3/3
```

R8's fix is confirmed live: `primary_signal_deepest_gap` keyed on `h=42` with `_h` and
`_residue` as separate fields; `fixed_dist` present as its own block under each arm.
The **literal** `validity_check` correctly **fails** at step 3 (`AssertionError: 3`) —
it is not vacuous.

**No residual hard-coded literals.** Zero `"h=<literal>"` string keys and zero
`K_NCR=24` / `D_NCR=25` / `173_209` assignments remain in the runner or the graft. All
iteration is over `DEEP_LADDER`/`TRAIN_HOPS`/`H_TOP`.

**Disk and placement.** Every spec writes to `/ephemeral` (5% used, 5.6 TB free); root
fs at 68% is never touched. No output-path collision with the 55 cells of record
(`/ephemeral/reseed_ckpts/`). Throughput reproduces §11.1 (K=12: 0.1164 s/step, 74%
median SM; K=32: 0.1488 s/step, 94% median) — every cell clears the <50%-is-a-bug bar.

---

## 2. FATAL — F1: the K=24 anchor cannot run

Patch R1 bumps `RUNNER_TAG` to `ncr_kscaling_runner_v1` so a K-scaling checkpoint can
never be confused with the pinned wave. `load_checkpoint` asserts on that field. The
55 cells of record carry the **old** tag. I confirmed both sides:

```
mob_g3b31_compA_s1.ckpt.pt   runner_tag = 'ncr_gate3_wave1_runner_v1'   step=20000  ncr d=25
~/ncr_kscaling/ncr_lm_wave1_runner.py:292:  RUNNER_TAG = "ncr_kscaling_runner_v1"
```

Then I ran §4.3's own anchor command:

```
NCR_K=24 kscaling_battery.py --k 24 --ckpt .../mob_g3b31_compA_s1.ckpt.pt --tag ...
  [checkpoint] ... failed to load/validate (AssertionError()) -- treating as ABSENT
  !!! LOUD FAILURE: CHECKPOINT FAIL [...]: missing or failed load_checkpoint validation
  EXITCODE=5
```

**Why this is FATAL, not cosmetic.** §7.1 defines the headline as
`CAPABILITY-HOLDS (curve) = CAPABILITY-HOLDS(K) at every K ∈ {12,16,20,24,28,32}`.
K=24 is not trained by this sweep; its only source is the anchor. With the anchor
dead, the pre-registered primary verdict is **undecidable**, and K=24 is also the sole
tie between this curve and the 55 cells of record, and a stratum of §7.3's test.
Releasing the six anchor specs, as §13 item 3 asks, would be releasing specs that
cannot execute.

**Minimal fix (verified sufficient, not merely necessary).** I relaxed *only* the tag
and re-ran the identical command. It works, and gives a real reading:

```
[AUDIT_k24_anchor_tagpatched] K=24 d=25 freeze=True ckpt_seed=1
  | P1b@h_top=36 acc=1.0000 margin=+0.9583 | P0max=0.0742 (chance 0.0417, band [0.0042,0.0791])
  self_check PASS
```

So: in `kscaling_battery.py` only, load the checkpoint through a scoring-specific path
that accepts a **pre-registered allowlist** of runner tags
`{"ncr_kscaling_runner_v1", "ncr_gate3_wave1_runner_v1"}` while keeping every other
`load_checkpoint` validation (step, `full_graft`, `backbone_only` keys), and record
the accepted tag in the output JSON (`runner_tag` is already emitted). **Do not touch
the runner's tag guard** — the battery never resumes, it only scores, so the
provenance guard stays strict exactly where it was designed to bite.

---

## 3. MAJOR — M1: the statistics substitution is invalid as written (audit item 1, top item)

### 3.1 First, the good news: the headline does not depend on it

The pooled test governs **CURVE 3 only** (the secondary frozen-vs-trainable ordering).
The audit brief's worry that pooling could mask a single-K capability failure does
**not** apply to the primary claim:

| Curve | Criterion | Per-K decidable? |
|---|---|---|
| 1 CAPABILITY (primary) | threshold on margin/κ, ≥2/3 seeds | **Yes** — no p-value anywhere |
| 2 WALL | per-K binomial band `1/K ± 3σ` | **Yes** |
| 3 ordering (secondary) | pooled MW p < 0.01 | **No** — admitted in §7.3 |
| 4 breadth-vs-depth | κ range ≤ 0.05 across K | Yes (descriptive) |

So WIN/PARTIAL/NULL for CAPABILITY-HOLDS / FRONTIER-AT-K* **remain per-K decidable via
the margin thresholds**, as the brief asks. That part of the design is sound.

### 3.2 The bad news: the pooled test fails in exactly the regime it was written for

The design's premise is correct — a per-K 3-v-3 Mann–Whitney has a minimum two-sided
p of 0.10 (I reproduced this by exact enumeration: U = 9.0, p = 0.1000). But the
proposed remedy pools 5 blocks of 3-v-3 and analyzes them as one 15-v-15 sample. That
ignores the blocking, and K is not a nuisance factor here — it is the axis with the
largest expected variance. I computed the consequences:

| Scenario | within-K wins | **pooled 15-v-15 MW** | **stratified within-K permutation** |
|---|---|---|---|
| A — arms globally separated | 45/45 | U=191.5, z=+3.29, **p = 0.00099** ✓ | T=45/45, **p = 6.3e-07** ✓ |
| **B — strong K effect, ordering unanimous** | **44.5/45** | U=134.5, z=+0.91, **p = 0.361** ✗ | T=44.5/45, **p = 6.3e-07** ✓ |
| **C — single-K driver, no ordering elsewhere** | 9/45 | U=153.0, z=+2.41, **p = 0.016** ✗ | T=27/45, **p = 0.44** ✓ |
| D — ceiling, tiny ordering | 32/45 | p = 0.0017 | T=38.5, p = 0.0011 |

**Scenario B is the indictment.** Frozen beats trainable in 44.5 of 45 within-K
comparisons — the strongest evidence for `ORDERING-CONFIRMED` the sweep could possibly
produce — and the pre-registered pooled test returns **p = 0.361**, nowhere near
p < 0.01. The reason is structural: when a strong K effect separates the blocks, the
cross-block comparisons cancel by balance and contribute a *constant*, so the pooled U
barely moves while the standard MW null still assumes all 30 values are exchangeable
(null variance 581.25 vs the true stratified variance 26.25, a 22× mismatch).

And **Scenario C is the mirror**: a pure single-K artifact with no ordering at the
other four K returns p = 0.016 from the pooled test — one notch of separation away
from spuriously clearing the band — while the truth is null.

A strong K effect is one of the two headline outcomes this sweep exists to find
(`FRONTIER-AT-K*`). So the pooled test is valid only under an assumption the sweep is
designed to test. **This is the same defect §7.3 claims to have fixed, one level up.**

### 3.3 Minimal fix

Replace the pooled 15-v-15 MW with a **stratified within-K exact permutation test**
(van Elteren / blocked Wilcoxon):

* Statistic `T = Σ_K U_K`, where `U_K` is the Mann–Whitney U of the 3 frozen vs 3
  trainable κ values **within** stratum K.
* Null: permute arm labels **within each K**. Exact null by convolving the `U(3,3)`
  distribution `{0:1,1:1,2:2,3:3,4:3,5:3,6:3,7:2,8:1,9:1}/20`.
* Pre-computed decision thresholds (two-sided p < 0.01), so nothing is chosen at
  harvest:
  * **5 strata** (K ∈ {12,16,20,28,32}): `T ≥ 36` of 45 (p = 0.0095); floor p = 6.3e-07.
  * **6 strata** (with the K=24 anchor): `T ≥ 42` of 54 (p = 0.0084); floor p = 3.1e-08.
* Bands become: `ORDERING-CONFIRMED` = T above threshold **and** median *within-K* gap
  > 0.05; `ORDERING-NEGLIGIBLE` = median within-K gap ≤ 0.05; `ORDERING-INVERTED` =
  the mirror (`T ≤ 9` / `T ≤ 12`) with gap < −0.05.

Two side benefits: (a) the K=24 anchor enters as **its own stratum**, which removes the
objection that anchor cells were trained under a different harness — stratification
never compares across strata; (b) it is immune to the ties problem, which the pooled
MW spec does not address at all and which will be severe if κ sits at ceiling.

The §7.3 escalation (seed-extension wave at the two highest K) should stay, now gated
on the stratified verdict.

---

## 4. MAJOR — M2: the primary band manufactures a frontier at small K

§7 states the primary band on `margin_over_chance ≥ 0.90` and leaves the choice to the
audit (§13 item 4). **The margin band is not comparable across K**, and its variation
is monotone in K and aligned with the axis under test:

| K | `margin ≥ 0.90` needs raw acc | `κ ≥ 0.90` needs raw acc |
|---|---|---|
| 12 | **0.9833** | 0.9083 |
| 16 | 0.9625 | 0.9062 |
| 20 | 0.9500 | 0.9050 |
| 24 | 0.9417 | 0.9042 |
| 28 | 0.9357 | 0.9036 |
| 32 | **0.9313** | 0.9031 |
| **span** | **0.0521** | **0.0052** |

Concretely: a model at a **flat** raw accuracy of 0.980 at every K — a perfectly flat
capability curve — is declared **FAIL at K=12 and PASS at all five other K**. Since
`FRONTIER-AT-K*` is "the smallest K at which CAPABILITY-HOLDS(K) fails," the
pre-registered output would be `FRONTIER-AT-K* = 12`: a frontier at the *smallest* K,
reported as a positive finding, produced entirely by the band's arithmetic.

**Minimal fix:** elect **`κ ≥ 0.90`** as the PRIMARY band, recorded before launch, with
margin retained as a reported secondary. κ = (acc − 1/K)/(1 − 1/K) is already computed
on every number, so this is a documentation change plus one line in the harvest rule —
zero build cost. (This is the audit exercising §13 item 4, with the number attached.)

---

## 5. MAJOR — M3: specs 0100↔0134 and 0101↔0137 collide, and the gate cell enters the curve

`gen_job_specs.spec()` builds `cell = f"kscaling_K{k}_{recipe}_s{seed}"` **independent
of `tier`**. Since the calibration cells are K=32 / seed 0 / both recipes, and the
sweep also contains K=32 / seed 0 / both recipes, four specs collapse onto two names:

```
--out       /ephemeral/kscaling/results/kscaling_K32_primary_s0.json  <- 0100 AND 0134
--ckpt-dir  /ephemeral/kscaling/ckpts/kscaling_K32_primary_s0         <- 0100 AND 0134
--cell-id   kscaling_K32_primary_s0                                   <- 0100 AND 0134
   (identically for 0101 AND 0137 on the compB recipe)
```

The runner is resume-safe by *skipping*:

```python
if os.path.exists(out_path):
    prev = json.load(open(out_path))
    if prev.get("status") == "COMPLETED":
        print(f"[{cell_id}] already COMPLETED -- skipping (resume-safe)")
        return prev
```

So once calibration completes, **0134 and 0137 no-op**, and their `validity_check`
passes against the calibration JSON — they route to `completed/` having trained
nothing. Two consequences:

1. The sweep is silently **28 trained cells, not 30**; §11.2's ledger is wrong by
   ~2 GPU-h (harmless in itself).
2. **The K=32 seed-0 frozen point of the PRIMARY curve becomes the very cell that was
   read to license the sweep.** Conditional on the sweep running at all, that cell has
   already been verified to clear `margin ≥ 0.90` at `h_top(32)` — it *cannot* fail.
   `CAPABILITY-HOLDS(32)` needs 2 of 3 seeds, so this collapses to "1 of the 2
   remaining seeds must pass," at the K the design itself calls the riskiest. That is
   selection bias inside the pre-registered primary band.

**Minimal fix — one line in `gen_job_specs.spec()`:**

```python
cell = f"kscaling_{'calib_' if tier == 'calibration' else ''}K{k}_{recipe}_s{seed}"
```

`--out`, `--ckpt-dir`, `--cell-id`, the log path and the `validity_check` all derive
from `cell`, so this fixes every collision at once. Regenerate the 32 specs. The
ledger then matches (30 genuinely trained sweep cells) and the gate cell stays out of
the curve.

---

## 6. MAJOR — M4: smoke item B is a vacuous negative test

`kscaling_smoke.py:111` registers `@neg("B_NEG_pinned_ladder_rejected", "")`. The
decorator's matcher is `if expect_substr.lower() in msg.lower()`, and `"" in anything`
is always true. Item B therefore records **PASS on any exception whatsoever**.

Design §8 states explicitly: *"one that raises the wrong error is recorded FAIL —
'fired with the WRONG error'. All eight negative-test instances fired with the expected
message."* **That claim is false for B**, and I proved it by injection. I made
`assert_ladder_sound` raise an unrelated `RuntimeError` *only* on the pinned reference
ladder (so only item B's body is affected) and re-ran the full smoke:

```
B_NEG_pinned_ladder_rejected   PASS  expect=''  fired_with=RuntimeError
      msg: AUDIT-BREAK-3: unrelated failure, the residue guard was NEVER reached
C_NEG_silent_residue_collision PASS  expect='PAIRWISE residue collisions'  fired_with=AssertionError
D_NEG_identity_residue         PASS  expect='IDENTITY mod K'               fired_with=AssertionError
E_NEG_train_residue            PASS  expect='colliding with a train-residue' fired_with=AssertionError
K_NEG_unpadded_T               PASS  expect='_MIN_KERNEL_T'                fired_with=AssertionError
=== K=12: 12 PASS / 0 FAIL / 0 N/A -> PASS ===
```

The guard was never reached, and the smoke still reported a clean 12/0/0. In a build
whose credibility rests on "all negative tests FIRED," one of them cannot tell a fired
guard from a crash.

**Minimal fix:** give B a real matcher. Either a per-K expected substring (the pinned
ladder fails for *identity/train* at K ∈ {12,20,28} and for *pairwise collision* at
K ∈ {16,24,32}), or the cheap general version — require `"deep-ladder"` (present in all
three guard messages) **and** assert `isinstance(exc, AssertionError)`. While there,
add the exception-type check to `neg()` itself so C/D/E/K stop relying on distinctive
substrings alone (see m6).

---

## 7. MAJOR — M5 and M6: gating

### M5 — the sweep is not machine-blocked on calibration (audit item 6, second half)

**Merely by convention.** I checked the spec schema directly: the union of keys across
all 32 specs is `cmd, gpu_h_estimate, hypothesis, id, lane, notes, output_dir, tier,
validity_check` — **there is no dependency field**, and `queue_worker.sh` claims
strictly by filename order from `pending/` with no reference to `tier` or `notes`. The
"DOUBLE-GATED" language lives only in the `notes` prose string.

The live risk is concrete, not hypothetical: `~/queue/pending/` is empty but **all 8
workers are running** (`queue_worker_g0..g7`, plus an `idle_fallback` daemon and
refill machinery), so anything landing in `pending/` is claimed within 60 s. Copying
`job_specs/*` in one gesture starts 8 cells immediately with no gate.

**Minimal fix (zero GPU cost):** prepend a sentinel test to each of the 30 sweep
specs' `cmd`:

```
test -f /ephemeral/kscaling/LICENSE_SWEEP || { echo "SWEEP NOT LICENSED -- calibration gate has not returned LICENSE-SWEEP"; exit 9; }
```

A prematurely claimed cell then exits instantly and routes to `failed/`. The gate
becomes a file the harvest agent creates on `LICENSE-SWEEP`, not a sentence someone
must read.

### M6 — the calibration gate can declare a frontier from n=1 (audit item 6, first half)

**Are the bands sufficient to catch the toy-prior K=32 risk? Yes.** Leg (3) reads P1b
at `h_top(32) = 48`, residue 16 = K/2 — the antipodal point, the hardest reachable
query, at the deepest squaring count. The toy prior (K=32 far-depth death at h ≈ 5–6,
FRONTIER-AT-K*=30) is a FREE-write result, so it bears on P0 (where it acts as a
positive control on the wall) and leg (3) is the right instrument for the exact-write
side. Running the riskiest K first and alone is the right call.

**But the failure branch is under-evidenced.** §10 says: if (3) fails while (1) and (2)
pass, *"K=32 is the frontier … re-scope the sweep to K ≤ 28, report FRONTIER-AT-K*=32."*
That declares a positive, publishable finding — the design calls a frontier "the more
interesting of the two outcomes for the flagship" — from **one seed**, against this
repo's own standing lesson that trainability variance across seeds is real (CLAUDE.md;
the head-to-head Task-2 diagnosis exists for exactly this).

**Minimal fix:** if leg (3) fails at seed 0, run K=32 frozen seeds 1 and 2 (2 cells,
~2 GPU-h) **before** declaring `FRONTIER-AT-K*=32` or re-scoping. These are cells the
sweep would have run anyway, so the true marginal cost is zero if the frontier is real
and ~2 GPU-h if it is a seed artifact — cheap insurance on the sweep's own headline.

---

## 8. minor findings

**m1 — the pad's pre-registered read is a dangling reference; I measured the pad
instead (audit item 2).** §3 and §5.3 both promise the K=20 `t_in` kink "is read per
§7.4" / "see §7 for the pre-registered read." **§7.4 does not address the pad at all** —
it is the `h_fix`-vs-`h_top` breadth-vs-depth control, and since `h_fix` and `h_top` are
read on the *same* padded document, the pad cancels within a K and is untouched across
K. The promised control does not exist.

So I measured it directly, on a **real trained** K=24 checkpoint (never trained with a
pad), forcing left-pads of 0, 10 and 38 BUFFER tokens with all four position fields
shifted:

```
PAD= 0  t_in=174 -> h1..h36 all 1.0,  h52 = 0.99609
PAD=10  t_in=184 -> h1..h36 all 1.0,  h52 = 0.99609
PAD=38  t_in=212 -> h1..h36 all 1.0,  h52 = 0.99609
```

**Identical to the last digit at every one of the 10 hops.** The mechanism is clear
from the code: `ncr_lm_forward` extracts keys/values/query-key from **raw token ids**
through `backbone.embed` (the G3-B12 fix), not from the contextualized hidden state, so
the write path is pad-invariant by construction, and the read survives it too. The pad
also leaves every real token's conv window unchanged, because each bind clause already
begins with 3 BUFFER tokens.

**Conclusion: the pad does not confound the P1b capability curve — the primary
readout.** The residual is training-side only (a K=12 cell trains on 128-token
documents of which 38 are a contiguous BUFFER run, which changes the LM objective's
token mix and the aux losses; and note `build_backbone`'s own disclosure that the
BUFFER row is an ordinary *trainable* embedding here, not zero-pinned). A blanket
"pad ALL K to a common T" is **not** warranted and would cost the K=24 anchor its
byte-identity — the wrong trade.

*Fix:* write the missing **§7.5**, one paragraph: the pad is disclosed, `doc_left_pad`
and `t_in` are recorded on every cell, the read instrument is measured pad-invariant
(this audit, cite the table above), and *if and only if* the κ curve shows a step
exactly at the K=20 kink, run one K=20 frozen seed-0 cell trained with `pad = 38`
(~0.8 GPU-h) to price the training-side residual at fixed K.

**m2 — the pinned graft's md5 is recorded but never verified.**
`patch_kscaling.PINNED_MD5["ncr_lm_wave1_smoke.py"] = None`, so only the runner is pin-
checked; design §5 claims both are. The exactly-once anchor rule limits the blast
radius, but this is a one-token fix: set it to `bc105af69661e488ff95f5046e2bcd8a`.

**m3 — stale K=24 prose is written into every results JSON at every K.**
`build_attribution`'s own `notes` string says *"argmax over 24 near-identical
directions"*, *"a collapsed target space makes ALL 24 targets look alike"*, *"a 24-way
discriminative test"*; the runner docstring at line 517 says *"chance = 1/24 ≈ 0.0417."*
R9 re-keyed only the ladder sentence. At K=12 or K=32 this prose is wrong, and it ships
inside the artifact that downstream agents are supposed to verify against. K-genericize
these strings (the `retrieval24_acc` *metric name* is correctly kept and annotated —
that part is fine).

**m4 — negative-test bookkeeping does not add up.** §8 and the commit message say
"all eight negative-test instances FIRED." The actual count is **26** across the six K
(B/C/D/E × 6 = 24, plus item K at K ∈ {12,16}), plus the 2 scorer guards in §8.2 = 28.
"8" matches no natural grouping, so the claim cannot be checked against the artifacts.
State the real number.

**m5 — `WALL-HOLDS(K)` folds train hops into the wall.** The band covers "10 hops × 6
cells," which includes h ∈ {1,2,3}. A P0 excursion at a *train* hop means "the model
learned to write the bindings it was trained on" — a real result, but a different and
much less surprising one than a compositional wall breach at held-out depth. Recommend
reporting the wall stratified into train vs held-out hops so the two cannot be
conflated at harvest. (The multiplicity is otherwise handled well: 360 readings at a
3σ band with a ≥2-seed replication requirement.)

**m6 — `neg()` never checks exception type.** C/D/E/K are saved only by distinctive
substrings; a differently-raised error carrying the same words would pass. Add an
expected-exception-type argument (default `AssertionError`).

---

## 9. Answers to the audit brief, itemized

| # | Question | Answer |
|---|---|---|
| 1 | Is the pooled test valid for "capability holds at every K"? | It is **not used** for that claim — CURVE 1 is per-K threshold-decidable and the bands survive (§3.1). But the pooled test is **invalid for its own purpose**: measured p = 0.361 under a unanimous ordering, p = 0.016 under a single-K artifact. Fix = stratified within-K permutation, thresholds T ≥ 36/45 or T ≥ 42/54 (§3.3). |
| 2 | Does the BUFFER pad confound the K axis at small K? | **No, for the primary readout — measured, not argued.** Trained model, pads 0/10/38: identical accuracy at all 10 hops. K=24 anchor byte-identity re-verified by me (all fields, two hops). Residual is training-side only; §7.4 does *not* provide the promised read, so write §7.5 with a conditional 0.8-GPU-h control. **A common-T pad is not warranted.** |
| 3 | Ladders — any violation? | **None.** Independently re-derived from the prose at all six K; table, residues, profile, monotonicity, antipodal top rung all reproduce exactly. Residue 4 is **not** collision-free — `h_fix` shares it with rung 1 at every K, by design, disclosed, and kept out of `DEEP_LADDER`. Not a violation; it is a bonus fp-depth control. |
| 4 | Gradient/correctness on real CUDA; negatives discriminating; resume; param formula | Smoke reproduced at K=12/K=32. C and D **proven discriminating** by guard removal. **B proven vacuous** by error injection (M4). Resume bit-identical (`max|Δ| = 0.0`) at both K. Param formula == measured at both K, exactly. |
| 5 | Scorer | Clean: matched pool from ckpt seed, P1b/P0 labelling, `chance = 1/K` and per-K band correct, `d == K+1` guard fires, `--k`/`NCR_K` tripwire fires, fail-loud on unloadable ckpt, `self_check PASS`. Exercised on a real trained checkpoint. |
| 6 | Calibration bands sufficient? Sweep truly blocked? | Bands **are** sufficient to catch the toy-prior K=32 risk (leg 3 reads the antipodal `h_top` at max squaring depth). Blocking is **by convention only** — no dependency field exists and 8 workers are live (M5). Failure branch declares a frontier from n=1 (M6). |
| 7 | Anything else launch-losing | **F1** (anchor unrunnable), **M3** (spec collision → silent no-op + gate cell in the curve). Literal spec cmd verified end-to-end at K=28; `/ephemeral` paths correct; no collision with cells of record; `RUNNER_TAG` provenance guard works (and is what causes F1). |

---

## 10. Release recommendation

**REV-REQUIRED.** Do not queue `0100`/`0101` yet. The revision is small and entirely
pre-GPU:

**Blocking (must land before any cell is queued):**
1. **F1** — tag allowlist in `kscaling_battery.py` only (3 lines). Re-run the anchor
   at one checkpoint to confirm.
2. **M3** — tier-aware `cell` in `gen_job_specs.py` (1 line); regenerate all 32 specs.
3. **M5** — `LICENSE_SWEEP` sentinel prepended to the 30 sweep `cmd`s (regenerated with #2).
4. **M4** — real matcher for smoke item B; re-run the smoke at ≥2 K.
5. **M2** — elect `κ ≥ 0.90` as the PRIMARY band in §7, recorded **before** launch.
6. **M1** — replace §7.3's pooled test with the stratified permutation test and its
   pre-computed thresholds.

**Blocking but documentation-only:** **M6** (n≥3 before declaring a frontier), **m1**
(write §7.5).

**Non-blocking:** m2–m6.

Per §5.5's own process note, the revision must re-run **both** instruments — the module
smoke *and* an end-to-end run through a literal spec command line. Since the revision
touches `gen_job_specs.py` and the battery, the end-to-end run should be a *regenerated*
spec, and the anchor path should be exercised once against a real cell of record.

**Open items I am explicitly not deciding** (they belong to the pre-launch
resource/placement red-team, per the ceremony tier): §13 item 2 (2-per-GPU packing for
the 12 K=12/16 cells) and §13 item 5 (`n_applies` 2–3 residual — I concur it is
acceptable, since the squaring count, which the #3 DRIFT finding actually implicates, is
matched at 5 across every K). §13 item 6 (the K=24 ladder-of-record collision, 29 ≡ 5
mod 24) is **confirmed by my independent re-derivation** and does warrant its own
EXPERIMENT_LOG note, independent of this sweep — the 6-point depth profile of the 55
cells of record is really a 5-point profile.
