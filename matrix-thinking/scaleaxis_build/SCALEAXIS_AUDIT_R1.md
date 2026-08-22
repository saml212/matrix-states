# SCALE-AXIS BUILD AUDIT R1 — independent, by re-execution

**Verdict: PASS-LAUNCH-RELEASED, with conditions.**
**0 FATAL / 5 MAJOR / 7 minor / 2 process.** No defect invalidates a
measurement the build took, and no defect changes what would run. Every
condition below is a text correction or a one-to-two-line code edit, costs
**zero GPU-hours**, and lands naturally before Stage A0.3 executes.

**Auditor:** independent agent, 2026-08-22. **Build under audit:** `100648c`
(harvest `b9061c9`). **Design of record:** `NCR_SCALE_AXIS_DESIGN.md`
DRAFT-R2 (`2aca91d`), accepted at `331f8d7`. **Box:** `youthful-indigo-turkey`,
`brev-ukptqsu65`, torch 2.12.1+cu130, 8×H100 80GB.

Per EXPERIMENT_LOG 2026-08-22 **#15**'s ceremony ruling this round also
re-verifies DRAFT-R2's own design deltas. **It found one there** (MAJOR-4),
which is the KSCALING precedent holding: a build audit catches design defects.

Nothing in the build was modified. Nothing was queued. This file is the only
repo write, and it is uncommitted.

---

## 0. What was re-executed (not read — run)

| # | action | result |
|---|---|---|
| 1 | Recomputed all 44 `BOX_TREE_MD5.txt` entries on the box **and** in the repo mirror | **44/44 identical**, both sides |
| 2 | md5'd every pinned upstream source; `git status` on all pinned paths | **untouched**; graft `bc105af6…`, battery of record `5735c788…` |
| 3 | Re-ran `scaleaxis_gates.py` at **K=16** against the deployed tree | **20 PASS / 0 FAIL**, `tree_md5` mismatches **0** |
| 4 | Re-ran `scaleaxis_gates.py` at **K=40** | **20 PASS / 0 FAIL**, mismatches **0** |
| 5 | Re-ran `rdelta_aggregate.py` against the archived six-rung JSONs | output **byte-identical** to the build's `RDELTA.json` |
| 6 | Ran the **98M half of A0.3** myself (2 solo `phase0-timing` probes, K=24/K=40) | **the pinned ±10% cross-check FAILS at 46.5%** — MAJOR-1 |
| 7 | Diffed `run_phase0_timing` across all three runner generations | 98M-vs-392M **byte-identical** (FATAL-1's fix proven, not inferred) |
| 8 | Exercised `a0_rules.py`'s never-run code path end-to-end | INCOMPLETE guard fires; P1/P2/P3/P4 arithmetic verified live |
| 9 | Re-derived §3.4's param table from first principles; compared to measured | **exact at all four K**, two vocabs, two rungs |
| 10 | Read `queue_worker.sh`'s claim predicate and checked it against `kappa_reader --reserve-gib` | **both** reservation legs fire |

---

## 1. MAJOR findings

### MAJOR-1 — The pinned ±10% A0.3 cross-check FAILS at 46.5%, and the design's "1.5500× standing instrument note" is falsified by measurement

I ran the two 98M `phase0-timing` probes the design's FATAL-1 fix requires
(`/ephemeral/scaleaxis/audit/a0/`, ≈0.005 GPU-h):

| probe | `mean_s_per_step_both_arms_combined` | `probe_wall_clock_s` |
|---|---|---|
| fresh 98M K=24 (`ncr_kscaling_runner_v1`) | **0.123463** | 7.559 |
| fresh 98M K=40 | **0.176222** | 10.725 |
| **archived pin** (§4.0, `ncr_gate3_wave1_runner_v1`, 2026-07-17) | **0.230755** | 18.454 |

Deviation vs the pin: **−46.50%**, i.e. **4.65× the pinned ±10% tolerance**.
Same host (`brev-ukptqsu65`), same torch (`2.12.1+cu130`), same K, batch 32,
`doc_len` 174, warmup 10, probe 60.

**I diffed the instrument to rule out a code change.** `run_phase0_timing`'s
timed region is *functionally identical* between the archived gate3 runner and
the kscaling runner — the sole difference is one added output field
(`kscaling=KS.provenance(...)`) **outside** the timed block. The per-arm
`torch.cuda.synchronize()` the design blames is present in **both**. So no code
change explains a 1.87× gap: **the archived `0.23075` is a stale box-state
snapshot, not a property of the instrument.**

**Consequences.**

1. The design's FATAL-1 *mechanism* narrative — *"per-arm synchronize inflates
   phase0 by 1.5500× over realized"* — is **wrong as a general claim**. The same
   code today reads **0.829×** realized at K=24 and **0.866×** at K=40. The sign
   is **inverted**: the probe *under*-reads.
2. `a0_rules.py:123-134` will emit `instrument_note_phase0_inflation = 1.5500`
   as fact, in the A0 record, contradicted by the A0 record's own probes.
3. §4.0 pins 1.5500 "as a standing instrument note **for every future wave**
   that prices from `phase0-timing`." That note would mislead the next wave.

**FATAL-1's FIX is nonetheless SOUND — and I verified it more strongly than the
design argued it.** The design reasons that the inflation "appears identically in
numerator and denominator." I checked the stronger fact: `run_phase0_timing` is
**byte-identical** between the 98M (kscaling) and 392M (scaleaxis) runners
(`grab()`-diffed, `len 5880 == 5880`, `a == b` True). R's bias cancels *by
construction*, whatever its absolute level.

**The gate behaves correctly.** I exercised it: `within: false`, verdict
`"INSTRUMENT DRIFT — reported before A0.5 is applied; does NOT block, because R
is a ratio of two FRESH probes."` The design pre-registered exactly this. The
risk is not that it blocks — it is that a coordinator reading a **46.5%
cross-check failure** immediately before committing ~99 GPU-h has no way to know
it is a stale baseline rather than a live fault. **Pre-recording that is the
entire point of this audit.**

> **Fix (text + 1 line, no re-run):** replace the 1.5500 note with the measured
> β table in MAJOR-2; mark the archived `0.23075` pin **STALE — superseded by
> the 2026-08-22 audit probes**, and either re-baseline `ARCHIVED_98M_K24_PHASE0`
> to `0.123463` or drop the cross-check to a reported-only diagnostic.

### MAJOR-2 — R's cancellation assumes a probe bias that my data shows is NOT operating-point-invariant, and the bias direction is unsafe for the cost-out gate

Define `β(K) := phase0(98M,K) / realized(98M,K)` (§8.2's realized row):

| K | phase0 (measured today) | realized (§8.2) | **β** |
|---|---|---|---|
| 24 | 0.123463 | 0.14888 | **0.8293** |
| 40 | 0.176222 | 0.20357 | **0.8657** |

β **rises 4.4%** across a 1.64× `t_in` increase — consistent with the workload
being launch-bound: more compute per kernel shrinks the relative synchronize
penalty. Corroborating, the probe exaggerates K-dependence: fresh
`K40/K24 = 1.4273` vs realized `1.3673`.

`R = ρ_realized × (β₃₉₂ / β₉₈)`. **The design assumes that ratio is 1, with no
measurement.** But 98M→392M is a **4× increase in per-kernel work** — a far
larger move along the *same axis* that already moved β by 4.4%. If
`β₃₉₂ > β₉₈`, then **R under-reads `ρ_realized`, biasing Rule P1 toward
NOMINAL** — the unsafe direction for a gate whose only job is to abort an
over-budget wave. A true `ρ ≈ 4.8` (COST-OUT) could read `R ≈ 4.0` (NOMINAL).

> **Fix (zero GPU-h):** record β₉₈(24)=0.8293 and β₉₈(40)=0.8657 in the A0
> record; state Rule P1's R as an estimate of `ρ_realized` **conditional on
> β₃₉₂ = β₉₈**; and add a **post-hoc reconciliation** — when the **first**
> calibration cell completes, compute `realized(392M,24) / realized(98M,24)` and
> compare it to `R(24)`; **if they differ by >15%, re-enter Rule P1 on the
> realized ratio before Stage B queues.** The first calibration cell produces
> this number anyway, so the check is free and converts an untested assumption
> into a measured one.

### MAJOR-3 — Both `--ceiling-gpuh` paths implement §3.6's FALLBACK and never its primary rule; the budget backstop ships ~2.9× looser than pinned

§3.6 breaker 1 pins `ceiling = 1.5 × (per-cell projection at the **MEASURED
CONTENDED rate R₈**)`, with `3.795 × solo` as the fallback **only** *"if `R₈`
cannot be measured."* Both build paths take the fallback unconditionally:

* `a0_rules.py:200-201` — `contended_ceiling_gpuh = 1.5 × 3.3 × gpu_h_98M × R`
  = **4.95 × solo**, substituting the runner's projection constant
  `CONTENDED_MULTIPLIER = 3.3` for `R₈` — **in the same file that measures `R₈`
  twenty lines earlier** (Rule P2).
* `gen_scaleaxis_specs.py:106-112` — measured branch =
  `1.5 × contended_gpuh_for_target_steps` = `1.5 × 3.3 × phase0-solo`, the same
  substitution on a different base.

Verified live (synthetic R=3.8, R₈=1.12): a0_rules emits K=16 ceiling
**15.084 GPU-h**. The design's primary rule gives `1.5 × 1.12 × 3.047 =
**5.12**` — **2.95× looser**. It is also **32% looser than the spec's own
PROJECTED placeholder** (11.412), so the A0.5 "re-price" *weakens* the breaker
instead of tightening it.

Direction is **LOOSE, not tight**, so MAJOR-1(b)'s "a breaker that fires on
every cell" failure is *not* reintroduced, and B6's rate watcher remains the
fast breaker (≈0.1% of a cell). The cost is that a pathological cell can burn
≈4.95× its projection (~15 GPU-h at K=24) before the backstop fires.

> **Fix (one line each):** use the measured `R₈` where it exists; keep `3.3`
> only on the `R₈`-absent path, and label which was used.

### MAJOR-4 — Design §4.4's P4 instrument identification is INVERTED; the build silently corrected it *(the #15 design-delta mandate)*

§4.4 states: *"The eval-pass leg is `:796` … `:663` is the train-only forward/
backward figure."* Verified against the **pinned** graft
(`kscaling_build/patched/ncr_lm_wave1_smoke.py`):

* **`:663`** sits inside **`smoke_3_backbone_eval_batch`** — backbone-only,
  `no_grad`, B=32/T=512. An *eval* leg.
* **`:796`** sits inside **`smoke_7_full_graft_train_step`** — a *training* step.

The characterization is **exactly backwards**, so §4.4's rationale (*"the #6
correction — training-only peaks understate by ≈1.3 GB — is precisely the gap
between them"*) rests on a false premise. This is the **third recurrence of the
FATAL-3 class** ("a rule keyed to a number the elected instrument does not
produce in the form the rule assumes"), and it survived all five design rounds.

**The build did the right thing in substance.** `scaleaxis_gates.py:403` sets
`P4_reading_gb = max(peak_796, peak_with_eval)`, where `peak_with_eval` is its
own **production-shaped two-arm train + `eval_both_arms`** peak — strictly more
conservative than the design's instrument. **RATIFY the number; correct the
design text.** Measured P4: **17.094 / 18.943 / 21.266 / 23.460 GB** at
K=16/24/32/40, all far below the 40 GB gate (and below §8.1's projected 21-28 GB
band at three of four K — a conservative projection miss worth recording).

Two related defects reach the published record:

* `peak_gb_line_1056_co_residency` is assigned `peak_796` — the field **asserts
  a `:1056` co-residency measurement that was never taken**. Mislabeled; unused
  by P4.
* `eval_pass_delta_gb = 0.0` is an artifact: `reset_peak_memory_stats()` is
  **not** called between `peak_train_only` (`:393`) and `peak_with_eval`
  (`:398`), so the second read is `max(train, eval)` and the delta can only ever
  be ≥0, reading exactly 0 whenever eval does not exceed the training peak.
  That is *correct for a peak* — but **EXPERIMENT_LOG #16's claim "eval adds
  0 GB at 392M — the #6 +1.3 GB correction does not reproduce at scale"
  over-reads the instrument.** Correct wording: *"the eval pass does not raise
  the peak above the training peak at 392M."*

### MAJOR-5 — Silent cross-scale record poisoning: both patched scorers default `--outdir` to the **98M** tree, and no aggregator discriminates on scale

* `patched/kscaling_battery.py:99` and `patched/depthext_eval.py:172` both carry
  `--outdir default=os.path.expanduser("~/ncr_kscaling/results")` — a directory
  that today holds **103 records of record** (all 98M) plus `results_depthext6/`
  (48). The patch never re-pointed it.
* The patch **added** a `scale` field to both scorers' output
  (`kscaling_battery.py:245`, `depthext_eval.py:296`) — and **no consumer reads
  it.** `rdelta_aggregate.load()` keys `cells[(K, recipe)][seed][n_squarings]`
  from `d["K"]`, `d["freeze_entity_adapter"]`, `d["ckpt_seed"]`, with **no scale
  key**; `reproduction_check` keys the same three. A 392M record landing in that
  directory **collides on-key with its 98M twin**, and whichever path sorts later
  silently wins — then reads as a 98M number in Rule R-δ *and* in the
  exact-reproduction cross-check.
* **B5 does not close this.** B5 guards the *checkpoint input*; this is the
  *output destination*.

**Currently LATENT, not executing:** `kappa_reader.score()` always passes
`--outdir` explicitly (its own `--outdir` has no default), and no job spec
invokes a scorer. But **§4.6 Stage C is a manual, unscripted harvest path** —
precisely where a bare invocation happens.

> **Fix (2 lines):** point both defaults at `~/ncr_scaleaxis/results` (or make
> `--outdir` required, as `kappa_reader` already does), **and** have
> `rdelta_aggregate.load()` assert the `scale` field the patch already writes.
> That closes it at the write end and the read end, and turns an added-but-
> unchecked field into a real guard.

---

## 2. Minor findings

**m1 — The gates' own `tree_md5` receipt did not match the deployed tree. CLOSED by my re-execution.**
All four `gates_K*.json` record `gen_scaleaxis_specs.py = a56072d1…`; the
deployed and mirrored file is `e984f5ee…`. mtimes: the generator was edited at
**00:56:36**, *after* gates_K16/24/32 (00:55:44-57) and 35 s after gates_K40
(00:56:01) — and the 24 specs were generated at 00:56:36 by the **un-gated**
version. So B1 and B2's spec-cmd check never saw the shipped generator. I re-ran
the gates at K=16 and K=40 against the deployed tree: **20/20 PASS,
`tree_md5` mismatches 0**, and the B1 hit set is **identical (82 hits)** with the
six `gen_scaleaxis_specs.py` hits shifted by exactly **+8 lines** — an additive
edit that changed no literal B1 detects. Gap closed; record the receipt.

**m2 — B1's pattern set omits two literals §3.2 explicitly enumerates.**
§3.2 requires grepping bare `768`, `64`, `12`, `50257`, `50259`, `98_000_000`.
`scaleaxis_gates.py:137-148` has **no pattern for bare `64` or bare `12`** (only
`d_state\s*=\s*\d+` / `n_layers\s*=\s*\d+`), and none for `SQUARING_PROFILE` —
which is **item 23**, present in the hand-written disposition table but
unreachable by the sweep that claims to *"re-derive the set from the DEPLOYED
tree."* Three dispositions also key on `gen_job_specs.py`, a file not in the
swept tree. An independent exhaustive sweep I commissioned found the residual
bare-`64` sites (`depthext_eval.py:297` `KS.provenance(64, …)`;
`kscaling_config.py:227` `h_enc: int = 64`) and confirmed both **INVARIANT** by
§3.3's code proof (`ENC_H = 64`, backbone-independent). **No 25th live constant
was found** — but B1 is weaker than its table implies, and its own note ("does
not prove a twenty-fifth absent") is the honest reading. Items 22/23/24 are
confirmed real and correctly ported.

**m3 — `overall.gate` ignores the SM-utilisation bug check.**
`a0_rules.py:230-233` computes `gate` from **P1 and P2 only**. A sustained <50%
reading sets `sm_util_verdict = "SUSTAINED <50% IS A BUG — diagnose before ANY
cell queues"` while `gate` still reads `"A0 CLEARS — Stage A may be queued"` —
two contradictory statements in one record, on the launch-decision surface.
Measured util is 97-100% at all four K, so it does not bind. *(P4's memory leg is
correctly excluded: §4.4 says ≥40 GB is explicitly not a blocker.)*

**m4 — P4 reads one K, not the max.** `run_stage_a0.sh:84` hardcodes
`--gates gates_K24.json` (18.943 GB). Memory grows with `t_in`; K=40 is the
largest at 23.460 GB. Both are far below 40, so nothing changes — but the gate
should read the max over the four K.

**m5 — Rule P3's `re_priced_gpu_h` is a step function labeled as an interpolation.**
`a0_rules.py:198-199` assigns `R[24]` to K∈{16,24} and `R[40]` to K∈{32,40},
while its own `action` string says K=32/K=16 *"are INTERPOLATED IN t_in and must
be FLAGGED as interpolations."* Both assignments **over**-price (true
interpolation would put K=16 below `R(24)` and K=32 at the K=24/K=40 midpoint),
so the ledger is safe by ≈3.6 GPU-h. The label is wrong.

**m6 — Rule R-δ's reachability count uses `>=` where §6.2's verdict is strict `>`.**
`rdelta_aggregate.py:171`: `reach = sum(1 for h,_ in H if h >= dstar)`.
SCALE-IMPROVES requires `Δ_scale > δ_depth`, and the max attainable `Δ_scale` is
exactly `H_c`, so a cell with `H_c == δ_depth` is **strictly unreachable**.
Because `floor_to` rounds down to a 0.005 grid, an `H[q-1]` landing *on* the grid
would inflate the count by one and break the rule's stated "≥6/8" guarantee.
**Does not bind here** — `H[2] = 0.096154`, `δ* = 0.095`, and all six counted
cells clear strictly — but the grid values are attainable (at K=16, headroom is
`(256−m)/240`, a multiple of 0.005 whenever `256−m ≡ 0 (mod 6)`). One-line fix.

**m7 — `run_stage_a0.sh` hygiene.** `probe()` echoes `exit=$?` and continues
(`set -u` only, no `set -e`); the `rules` verb does not check `a0_rules`' exit
code. Fail-closure *is* recovered at A0.5 — I exercised it: missing probes ⇒
`"INCOMPLETE — missing solo probes: ['392m_K24','392m_K40']"` with the note *"R
… CANNOT be formed from three"* and `return 3`. More substantively, **A0.4
launches 8 concurrent probes on all 8 GPUs to measure `R₈` — the number that
gates the whole ledger — without recording an `nvidia-smi
--query-compute-apps` snapshot**, while `idle_fallback_daemon.sh` (PIDs
525731/525732) and its minutely `watchdog_idle_daemons.sh` cron are live. Benign
today (pool dry; I verified all 8 GPUs at 0%, 0 MiB, `~/queue/pending` empty),
but exclusivity around `R₈` should be **recorded, not assumed**. Also
`tmux new-session -d -s a0c_g$g "bash $0 …"` relies on a relative `$0` resolving
inside tmux's inherited cwd — invoke with an absolute path.

---

## 3. Process findings

**p1 — The 10-item deviation list is NOT recorded in the repo.**
EXPERIMENT_LOG #16 states *"Deviations 10 items, all disclosed and accepted by
coordinator"*, but no enumerated list exists in `matrix-thinking/scaleaxis_build/`,
in the `100648c` commit body, or in the log. I reconstructed and adjudicated the
ones reachable from artifacts (§4 below), but **an unrecorded list cannot be
independently ratified**, and the standing house rule is that a round's
disclosures are recorded in the repo *before* the dependent stage proceeds.

**p2 — "Stage A first and alone" is not enforced by the mechanism the build cites.**
`gen_scaleaxis_specs.py:24-26` argues the `0190-0195` numbering means *"a sweep
spec must never be claimable while a calibration spec is pending."* Verified
against `queue_worker.sh:119`: claiming is `for f in $(ls "$PENDING" | sort)` —
filename order **orders** claims, it does not **block** them. With 8 live workers
and all 24 staged, workers 7 and 8 claim `0200`/`0201` in the same poll cycle
(and the sextet-first head destroys the longest-first makespan: 10.842 h vs
10.194 h). The real enforcement is the design's own §7.2(D): **stage only
`0190-0195` into `pending/` until the `LICENSE_SWEEP_SCALEAXIS` sentinel fires.**
Make it an explicit launch-checklist line.

Related, already true and worth stating: queue-ineligibility is **location-only**.
`queue_worker.sh` never reads `queue_eligible`, `notes`, or `tier`; the 24 specs
are unclaimable solely because they live in `~/ncr_scaleaxis/job_specs/`, outside
the queue tree. That is sufficient today, but 8 workers are live with no
`STOP`/`PAUSE` sentinel and all 8 GPUs free.

---

## 4. The 10 disclosed deviations — adjudication

The list is unrecorded (p1). These are the deviations reachable from artifacts:

| # | deviation | ruling |
|---|---|---|
| module naming | scaleaxis tree keeps `kscaling_config.py` rather than renaming | **RATIFY.** Renaming turns a one-dict port into a ~60-site rewrite, against §3's framing. Trees are self-contained via `sys.path.insert`; `config_module_tree` is recorded in provenance so a reader never infers which copy produced a number. |
| **#3** legacy smoke item 11 disabled | `smoke_11_ablation_flags_construct` now dies under B2's assert | **RATIFY WITH A NOTE.** B2's constructor assert **is** a strict superset for the production path — it fires on any non-`(linear, add)` construction, wired at **three** levels (spec: runner exposes no `--adapter` at all; runner startup `:1920-1926`; constructor `:351-362`). **But "disabled" was implemented as a crash, not a skip:** `ncr_lm_wave1_smoke.py:1082` calls item 11 unconditionally in `main()`, and it builds `adapter="mlp"` (`:1015`), so the graft's own documented standalone entry point now dies with an uncaught `AssertionError` *before items 1-10 run*. DEAD relative to every production path (runner, battery, depthext, gates and `kscaling_smoke.py` all import the module and never call `main()`) — it cannot touch a record — but it removes a debugging entry point at the worst moment. **One-line skip guard recommended.** |
| **#5** P0@15sq labeling | P0 at 15 squarings marked not-comparable | **RATIFY.** `depthext6_driver.py:96-99` writes into every record: *"the archived four-rung wave ran P0 at 11 squarings, so this P0 reading is NOT comparable to it — P1b is what Rule R-δ reads."* Verified present in the emitted JSONs (single P0 hop, `h=32804`, `n_squarings=15`). |
| **#6** post-hoc note edit | one field edited after the fact | **RATIFY — NO NUMBER WAS TOUCHED.** `depthext6_driver.py:112` records `only_field_modified_post_hoc = "matched.P0.note (a label, not a number)"`. Independently confirmed: my re-run of `rdelta_aggregate.py` reproduced **all 192 archived P1b accuracies exactly** and produced a **byte-identical `RDELTA.json`**. No number any verdict reads was altered. |
| ceilings PROJECTED | specs carry `3.795 × projected solo`, not a re-priced ceiling | **RATIFY AS A CANDIDATE STATE, with a launch condition.** All 24 self-flag it in **two** fields (`scaleaxis.ceiling_provenance` and the `notes` tail): `"PROJECTED-NOT-LAUNCH-READY: 3.795 x projected solo …; MUST be re-derived from Stage A0 before queueing"`. Correct for a candidate build — but §4.5 requires the **re-priced** ceiling, so **the 24 as written are not §4.5-compliant and must be regenerated with `--ceilings-from` after A0** (see C4, and MAJOR-3 for the formula). |
| mixed order declined | §8.3's 9.02 h ELECT-or-DECLINE | **RATIFY the decline** (longest-first at 10.19 h is pinned and reproduces — see §5). Note `--elect-mixed-order` (`:377-380`) only *prints* a declined note; the docstring at `:22` describes it as implementing the order. Cosmetic. |
| A0.3/A0.4 unrun | deliberately deferred | **RATIFY.** Rule P1 is a launch decision, not a build artifact. `/ephemeral/scaleaxis/a0` verified empty. |
| `SQUARING_PROFILE` via driver | item 23 set from a driver, never by editing the wrapper | **RATIFY.** Correct discipline; but see m2 — B1's sweep cannot surface it. |
| B1 found 3 beyond 21 | items 22/23/24 | **RATIFY, all three real.** Item 22 (`kscaling_battery.py`'s `provenance(64, 768)`) is the serious one — it would have recorded **half** the true integ param count in every 392M record; now `provenance(R.H_NCR, RUNG1_BACKBONE["d_model"])`. Items 23 and 24 (the graft's duplicate `_MIN_KERNEL_T`) confirmed and ported. |
| P4 instrument | build measured its own two-arm with-eval peak | **RATIFY the number; the design text is what is wrong** — see MAJOR-4. |

---

## 5. Ratifications — positive receipts, all re-executed

**Provenance.** Repo mirror md5-identical to the box, **44/44**, recomputed
independently both sides. Pinned upstream untouched: graft `bc105af6…` (§3.6's
stated value), `kscaling_build/patched/*` = box `src/*`, battery of record
`5735c788…`; `git status` clean on `experiment-runs/2026-07-17_ncr_gate3_wave1/`,
`matrix-thinking/kscaling_build/`, `matrix-thinking/deltanet_rd/`.

**Gates, re-run by me against the deployed tree.** K=16: **20 PASS / 0 FAIL**.
K=40: **20 PASS / 0 FAIL**. `tree_md5` mismatches **0** in both. Bit-identical on
every deterministic field, including `B3.measured_total_per_arm` and A0.2's
`grad_norm = 38.35804161332175`.

**A0.2 — the first-ever `MIN_KERNEL_T` measurement at `d_state=128`.** Floor
≤128 **HOLDS**; K=16's `t_in = 128` clears at **zero margin, as designed**;
`backward_finite`; and `A02_NEG_below_floor_crashes` (T−1) **FIRED**. Re-fired by
me at K=16.

**B6/B7 forced-fail negatives, re-fired by me.** B6: (i) doubled `elapsed` trips
at step 175, nominal log does not; (ii) blind filter raises on an eval-metric
token with *the line withheld*; (iii) `.json` input refused. B7: truncated ⇒
`UNREADABLE (detected, NOT scored)`; zero-byte ⇒ detected; off-cadence 17321 ⇒
labeled, with `--required-step` set to *the step actually read* so the battery's
`SKIP` is **structurally impossible**; missed window ⇒
`missing_trajectory_points: [15000]` with branch-(B)'s fallback named; plus a
positive control (intact ckpt reads step 15000). Constants design-exact:
`BREAKER_MULT = 1.5`, `CONSECUTIVE = 2`, `LOG_EVERY = 25`; `rate = elapsed/step`
as §3.6 pins.

**The `neg()` harness has teeth.** PASS requires the body to **raise** *and* the
message to contain the expected substring; a clean return is
`FAIL — NEGATIVE TEST DID NOT FIRE, the guard is vacuous`; a wrong error is
`FAIL — fired with the WRONG error`.

**GPU reservation verified against `queue_worker.sh`'s actual predicate, by exact
code path.** The worker **skips** iff `napps > 0 || mem >= 2048` (MiB).
`kappa_reader --reserve-gib` defaults to **2.5 GiB = 2560 MiB ≥ 2048** ✓ **and**
the reader is itself a compute app ⇒ `napps > 0` ✓. **Both** belt-and-braces legs
fire; the design's description of the predicate is accurate.

**Rule R-δ is MECHANICAL, and I reproduced it byte-identically.** My independent
re-run of `rdelta_aggregate.py` against the archived six-rung JSONs produced a
record **identical to `RDELTA.json` in every field**:

* `δ*(11) = floor₀.₀₀₅(0.03629) = 0.035 < 0.05` ⇒ **s=11 REJECTED** — the
  design's own teeth receipt reproduces exactly.
* `δ*(13) = floor₀.₀₀₅(0.096154) = 0.095 ≥ 0.05` ⇒ admissible.
* Shallowest admissible ⇒ **s\* = 13, δ_depth = 0.095**. No judgment is exercised
  anywhere on the path; `quantile_idx = 3 if n == 8` matches §4.6.1's explicit
  enumeration (the prose formula `ceil(n/4)` yields 2 at n=8, which §4.6.1 itself
  corrects to 3 — the build implements the enumeration, correctly).
* **Reproduction 192/192 EXACT**, under a **strict zero tolerance**
  (`abs(new−old) > 0`) — the check has teeth.
* The s=11 headrooms reproduce **§5.5's pinned table in all 8 cells** to 4 dp
  (0.024457 / 0.033333 / 0.036290 / 0.040064 / 0.058333 / 0.069293 / 0.076613 /
  0.124199).

**Ordering extension — EXTENDS, no retraction.** My re-run: **T₈ = 68.0 @13sq**
and **67.0 @15sq** vs the 53/72 bar. At s=11 it returns **T₈ = 61.5/72**,
reproducing **#8's published verdict of record exactly**, and **T_W4 = 30.5** with
`U = (6.5, 9.0, 6.0, 9.0)` — **§2.1's CURVE 3 row exactly**, including the 5/7/9
rows (21.0 / 30.0 / 34.0). The instrument is reference-matched before it runs.

**6/8 at floor, 2/4 frozen — CONFIRMED; the rule needs NO amendment.** Reachable
at s\*=13: K40_frozen (0.0962), K24_frozen (0.0978), K16/K24/K32/K40_trainable.
**Unreachable: K16_frozen (0.0417), K32_frozen (0.0806).** 6/8 is exactly the
rule's guaranteed minimum, so the rule is satisfied as written. **But §5.5's
motivating sentence — *"from 0-of-4-frozen to 3-of-4-frozen. That is the
repair."* — is realized as 2-of-4-frozen** and must be restated before harvest
(condition C6). See §6 for the one boundary ambiguity I did find (m6).

**The §5.5 conservative-linear lower bound is violated in exactly 1/8 cells at
s=13** (K16_frozen: projected ≥0.0624, measured 0.0417) and **0/8 at s=15** —
matching #16's disclosure. **I checked the counterfactual: had K16_frozen met its
projection, `δ*` would still be 0.095** (the 3rd-smallest is unchanged at
0.096154). **The projection error did not move the election.**

**Params: measured == §3.4's formula exactly at all four K**, confirmed against
my own from-first-principles re-derivation (rung-2/50259 backbone = 391,872,512;
totals 392,095,889 / 392,122,521 / 392,149,153 / 392,175,785; ratios 4.0085 /
4.0081 / 4.0078 / 4.0075). Both K=16 and K=40 re-measured by me on real CUDA.

**Checkpoint resume bit-identity:** `I_checkpoint_resume_bit_identical` **PASS**
at K=16, alongside `F_param_counts_exact` and
`K_NEG_unpadded_T_crashes_kernel_floor`.

**All 24 specs are clean** on every content axis: three-way `NCR_K` / `--k` /
`--scale 392m` consistency (runner enforces `--scale` `required=True`,
`choices=("98m","392m")`); the full §3.5 recipe; `--freeze-entity-adapter` as the
**single** recipe discriminator (12 primary / 12 compB); the correct per-K param
literal in three places each and **no 98M param count anywhere**; ladders,
`h_fix`, `t_in`, pad and wall bands matching §2.1/§3.5; `--ckpt-every 5000` on
0190-0195 and 10000 on the other 18; **no `--adapter`/`--read-inject` in any
spec** — and the runner exposes no such flag, so B2 is unfalsifiable at spec
level. `ls | sort` yields calib → K40 → K32 → K16, and **the generator asserts
that order itself** (`:353-355`). Simulated Stage-B makespan **10.195 h** vs the
design's pinned **10.194 h**. **Zero 4-digit id reuse** across the box queue (496
completed, 38 parked, 0 pending) and every repo spec directory. The specs
regenerate **byte-identically** from the deployed generator.

**`validity_check` has real teeth** (verified against the runner's emitting
sites): the param clause compares a generation-time literal against a **measured
`nn.Module` count** (`runner.py:1293/1308`), so a mis-resolved backbone fails
loudly; Gate-0 (`h[-1][1] < h[0][1]`), finiteness, `status`, `step`,
`runner_tag`, and all seven `kscaling` provenance keys are emitted and
checkable. `len(loss_history) >= 100` is defense-in-depth only (real cells log
801) — exactly as the design says.

**`--ceilings-from` carries a proper cross-scale guard**
(`gen_scaleaxis_specs.py:323`: skips any record whose `kscaling.scale != SCALE`,
then asserts non-empty).

**Nothing was queued, and A0 is genuinely unrun.** `/ephemeral/scaleaxis/a0`
empty; `~/queue/pending` empty; `/ephemeral` 5.5 TB free.

---

## 6. On the boundary, as the brief asked

Reachability sits **exactly at the rule's floor** (6/8). I looked specifically
for ambiguity there, since boundary ambiguity discovered at harvest is how
verdicts get contested.

* **The rule itself needs no amendment.** `δ*` is the 3rd-smallest headroom
  rounded down; cells 3..8 therefore clear it, so ≥6/8 is guaranteed by
  construction and 6/8 is compliant, not marginal.
* **One latent ambiguity exists (m6):** reachability is counted with `>=` while
  §6.2's SCALE-IMPROVES is strict `>`. It does **not** bind on this data — no
  cell sits at `δ_depth`, and all six counted cells clear strictly — but it
  would break the ≥6/8 guarantee if `H[q−1]` ever landed on the 0.005 grid, which
  is attainable at these `n = 256` quantizations. Fix it now, in one line, rather
  than argue it at harvest.
* **A second, larger disclosure gap:** Rule R-δ step 2 justifies the 0.05 floor
  with *"the **median** within-cell seed range **at 11 squarings** is 0.0344, so
  0.05 clears typical noise with 45% margin."* That is computed at s=11, but the
  rule **elects** s\*=13. Measured at the elected depth: the **median seed range
  is 0.0844** (2.46× the s=11 value) and the max is 0.3302, so **δ_depth = 0.095
  clears the noise at s\*=13 by only 1.13× — 13% margin, not 45%.** Three of the
  six reachable cells have a 98M within-cell seed range **larger than the
  equivalence margin itself** (K16_trainable 0.2458, K24_trainable 0.3302,
  K40_frozen 0.0962). This does not violate any rule and does not block launch —
  §5.2 already routes around it (*"the rank test is primary at depth and the
  magnitude band is secondary"*) — but the *median* leg of that disclosure no
  longer holds comfortably at the elected depth and must be restated in the same
  sentence that states δ_depth.

---

## 7. Free pre-registration, from measurements I took

Not a defect — a prediction, recorded before A0.3 runs so it cannot be chosen
afterwards.

B8's production-shaped two-arm probe (batch 32, both arms, 12 timed steps —
**a third instrument, not `phase0-timing`**) reads, against §8.2's realized 98M row:

| K | 392M s/step | 98M realized | implied ratio |
|---|---|---|---|
| 16 | 0.3791 | 0.14434 | 2.63 |
| 24 | 0.4788 | 0.14888 | 3.22 |
| 32 | 0.6462 | 0.17249 | 3.75 |
| 40 | 0.7909 | 0.20357 | 3.89 |

`R(40)/R(24)` implied = **1.208 > Rule P3's 1.15 bar**. **Pre-registered
expectation: Rule P3 will FIRE and the ledger must be re-derived per K.** Under
`a0_rules`' current step assignment that yields ≈89.3 GPU-h headline; under true
`t_in` interpolation ≈85.3 — both inside the 87-101 envelope, with `R_max ≈ 3.89
≤ 4.0` ⇒ **P1 NOMINAL**. **These are not R** (different instruments on the two
sides) and must not be substituted for A0.3; they are a prior to check A0.3
against.

---

## 8. Conditions

### Before Stage A0.3 executes — text and ≤2-line code edits, zero GPU-h

* **C1 (MAJOR-1).** Mark the archived `0.23075` cross-check pin **STALE**;
  replace §4.0's and `a0_rules.py:123-134`'s **1.5500× instrument note** with the
  measured β table. State plainly, *in the A0 record*, that the 46.5%
  cross-check failure is a stale-baseline artifact and not a live fault, with the
  byte-identity of `run_phase0_timing` across the two runners as the receipt that
  R is unaffected.
* **C2 (MAJOR-2).** Record β₉₈(24)=0.8293 / β₉₈(40)=0.8657; state Rule P1's R as
  conditional on `β₃₉₂ = β₉₈`; **add the post-hoc reconciliation** on the first
  calibration cell (>15% divergence ⇒ re-enter P1 on the realized ratio before
  Stage B queues).
* **C3 (MAJOR-3).** Use the **measured `R₈`** in `a0_rules.py`'s
  `contended_ceiling_gpuh`; keep 3.3 only on the `R₈`-absent path and label which
  was used.
* **C4 (MAJOR-5).** Re-point both scorers' `--outdir` default to
  `~/ncr_scaleaxis/results` (or make it required); make `rdelta_aggregate.load()`
  assert the `scale` field. Then **re-sync the mirror and re-run the gates at one
  K** — ≈4 minutes, as demonstrated.
* **C5 (m3, m4, m6, m7).** Add the SM-util leg to `overall.gate`; make P4 read
  the max over the four K; make R-δ's reachability strict; absolutize `$0` in
  `run_stage_a0.sh` and snapshot `nvidia-smi --query-compute-apps` before and
  after A0.4.
* **C6 (p1).** **Write the 10 deviations into the repo** — `scaleaxis_build/` or
  EXPERIMENT_LOG — before the dependent stage proceeds. Include §4's rulings and
  §5.5's restatement (**2-of-4 frozen, not 3-of-4**) plus §6's δ_depth-vs-noise
  restatement at s\*=13.
* **C7 (MAJOR-4).** Correct §4.4's `:663`/`:796` characterization; re-point P4 at
  the instrument the build actually built; fix `peak_gb_line_1056_co_residency`;
  and correct EXPERIMENT_LOG #16's "eval adds 0 GB" wording to "the eval pass
  does not raise the peak."

### Before Stage B queues

* **C8 (p2).** Stage **only** `0190-0195` into `~/queue/pending/`. Do not stage
  `0200-0217` until the `LICENSE_SWEEP_SCALEAXIS` sentinel drops. Filename order
  does not gate; staging does.
* **C9 (deviation: ceilings).** Regenerate all 24 specs with `--ceilings-from`
  after A0, so each carries the **re-priced** `--ceiling-gpuh` §4.5 requires. The
  24 as written are candidate-state only.
* **C10 (§8.3.1).** Discharge the daemon-park procedure and the reader-GPU
  reservation as **enumerated** pre-launch checks, verified by a fresh
  `nvidia-smi --query-compute-apps` read. `idle_fallback_daemon.sh` and its
  minutely cron are **live right now**.
* **C11 (m2, optional).** Add bare-`64`, bare-`12` and `SQUARING_PROFILE`
  patterns to B1 and re-run the sweep, so the disposition table and the sweep
  cover the same set. My independent exhaustive hunt found no 25th **live**
  constant, so this is completeness hygiene, not a suspected miss.

---

## 9. Why PASS and not REV-REQUIRED

No finding invalidates a measurement this build took. Every number I re-executed
reproduced — most of them **bit-identically**: the param table at two K, A0.2's
`grad_norm` to 17 digits, `RDELTA.json` field-for-field, 192/192 archived
accuracies under a zero tolerance, #8's `T = 61.5/72`, §2.1's `U`-vector, and
40/40 gate items across two K against the deployed tree.

The five MAJORs are: two **design-text** corrections that change no number
(MAJOR-1's rationale, MAJOR-4's instrument label); one **loose-not-tight**
breaker (MAJOR-3 — it cannot cause a spurious abort, and B6 remains the fast
breaker); one **latent, non-executing** foot-gun (MAJOR-5); and one **free
procedural addition** (MAJOR-2). Two of them — MAJOR-1 and MAJOR-4 — are defects
the *design* carried through five rounds, which is the ceremony working as
intended, not the build failing.

Stage A0 has not run. It is a ~0.5 GPU-h gate that is itself the natural place to
land C1-C5, and it is hard-gated ahead of every training cell. Holding a closing
grant for edits that cost minutes and no GPU time would spend the one resource
that cannot be recovered.

**PASS-LAUNCH-RELEASED**, conditional on C1-C7 before A0.3 and C8-C10 before
Stage B.
