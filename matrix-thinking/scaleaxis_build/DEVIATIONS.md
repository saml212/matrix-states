# 392M SCALE AXIS — BUILD DEVIATIONS, ADJUDICATED

**Condition C6** of `SCALEAXIS_AUDIT_R1.md` §8 (audit commit `bb86a9f`;
adjudication EXPERIMENT_LOG 2026-08-22 **#17**). Process finding **p1** was that
the 10-item deviation list existed only in the build agent's final message and
in EXPERIMENT_LOG #16's *count* — **not** in the repo — and *"an unrecorded list
cannot be independently ratified."* This file is that record, written **before**
Stage A0.3 executes.

Design of record: `NCR_SCALE_AXIS_DESIGN.md` DRAFT-R2 (commit `2aca91d`).
Build: commits `100648c` / `b9061c9`. Rulings below are the **audit's**, quoted
in substance from its §4; the ratifications in §5 were all re-executed, not read.

---

## 1. The ten deviations and their rulings

| # | Deviation | Ruling (audit §4) |
|---|---|---|
| **D1** | **Path naming follows the DOC, not the build brief.** The brief said `matrix-thinking/scale_axis_build/`, `~/ncr_scale_axis/`, `/ephemeral/scale_axis/`; the design (§3.6, §8.1) says `scaleaxis_build/`, `~/ncr_scaleaxis/`, `/ephemeral/scaleaxis/`. | **RATIFY.** The brief's own precedence rule is "where the doc and this brief differ, the doc wins." |
| **D2** | **The scaleaxis tree keeps the module name `kscaling_config.py`** rather than renaming it `scaleaxis_config.py`. | **RATIFY.** Renaming turns a one-dict port into a ~60-site rewrite, against §3's framing. Trees are self-contained via `sys.path.insert(0, dirname(__file__))`; `config_module_tree` is recorded in provenance so a reader never has to infer which copy produced a number. |
| **D3** | **B2's assert disables the graft's legacy smoke item 11** (`mlp`/`mlp_logits` construction check). | **RATIFY WITH A NOTE.** B2 is a strict superset for the production path, wired at three levels (the runner exposes no `--adapter` at all; runner startup; the `NCRIntegration` constructor). **But "disabled" was implemented as a CRASH, not a skip** — `main()` called item 11 unconditionally, so the graft's own standalone entry point died with an uncaught `AssertionError` *before items 1-10 ran*. DEAD relative to every production path (runner, battery, depthext, gates and `kscaling_smoke.py` all import the module and never call `main()`), so no record could be touched — but it removed a debugging entry point at the worst moment. **Audit recommended a one-line skip guard; APPLIED as patch `G3_item11_skip_not_crash`** (see §3 below). |
| **D4** | **The 98M re-score ran all six rungs {5,7,9,11,13,15}**, not only {13,15} as the brief asked. | **RATIFY.** §4.6 specifies six; running the four archived rungs bought the exact-reproduction cross-check (**192/192 EXACT under a strict zero tolerance**). |
| **D5** | **P0 in the six-rung records is at 15 squarings**, so it is NOT comparable to the archived four-rung wave's P0@11sq. | **RATIFY.** `depthext6_driver.py` writes the non-comparability into **every** record; verified present in the emitted JSONs (single P0 hop, `h=32804`, `n_squarings=15`). P1b — what Rule R-δ reads — is unaffected. |
| **D6** | **One cosmetic field edited post-hoc** — `matched.P0.note`, which hard-codes "11 squarings" and would otherwise have been false at six rungs. | **RATIFY — NO NUMBER WAS TOUCHED.** The driver records `only_field_modified_post_hoc`. Independently confirmed: the audit's re-run of `rdelta_aggregate.py` reproduced **all 192 archived P1b accuracies exactly** and produced a **byte-identical `RDELTA.json`**. |
| **D7** | **Specs carry a PROJECTED `--ceiling-gpuh`** (`3.795 × projected solo`), not a re-priced one. | **RATIFY AS A CANDIDATE STATE, WITH A LAUNCH CONDITION.** All 24 self-flag it in two fields. But §4.5 requires the **re-priced** ceiling, so **the 24 as written are not §4.5-compliant and must be regenerated with `--ceilings-from` after A0** — condition **C9**, coordinator-owned. See also §2's MAJOR-3/C3: the *formula* the measured branch uses has been corrected. |
| **D8** | **§8.3's 9.02 h mixed spec order DECLINED**; longest-first (10.19 h) pinned. | **RATIFY the decline.** Simulated Stage-B makespan **10.195 h** vs the design's pinned **10.194 h**. Cosmetic residue: `--elect-mixed-order` only *prints* a declined note while the docstring describes it as implementing the order. |
| **D9** | **Stage A0.3/A0.4 built but deliberately UNRUN.** | **RATIFY.** Rule P1 is a launch decision, not a build artifact; `/ephemeral/scaleaxis/a0` verified empty. |
| **D10** | **B1 found three size-bearing constants beyond the design's 21** — items **22** (`provenance`'s hard-coded `d_model`, at two sites), **23** (`depthext_eval.SQUARING_PROFILE`), **24** (the graft's duplicate `_MIN_KERNEL_T`). | **RATIFY, all three real.** Item 22 is the serious one: `kscaling_battery.py`'s `provenance(64, 768)` **would have recorded half the true integ param count in every 392M record**. Now `provenance(R.H_NCR, RUNG1_BACKBONE["d_model"])`. |

Two further deviations the audit adjudicated that are not on the numbered list:

* **`SQUARING_PROFILE` set from a driver, never by editing the wrapper** —
  **RATIFY**, correct discipline. (But see m2/C11: B1's sweep could not surface
  it; the patterns are now added.)
* **P4 read the build's own two-arm with-eval peak rather than the design's
  named lines** — **RATIFY THE NUMBER; the design text is what is wrong.**
  See §2, MAJOR-4.

---

## 2. What the audit found, and what it changed (C1-C7, C11)

Five MAJORs, **0 FATAL**. Two are **design-text** corrections that change no
number; one is a **loose-not-tight** breaker; one a **latent, non-executing**
foot-gun; one a **free procedural addition**. Two of them — MAJOR-1 and
MAJOR-4 — are defects the *design* carried through five rounds.

* **MAJOR-1 / C1 — the archived `0.23075` cross-check pin is STALE, and the
  design's "1.5500× standing instrument note" is FALSIFIED.** Fresh 98M
  `phase0-timing` on the same box, same torch, same K/batch/`doc_len`: **0.123463
  (K=24)** and **0.176222 (K=40)** — **−46.50%** against the pin, 4.65× the
  pinned ±10% tolerance. The instrument was ruled out as the cause: the timed
  region is functionally identical between the two runners and the per-arm
  `synchronize()` is in **both**. **The sign is inverted** — the probe
  *under*-reads realized (β = 0.8293 / 0.8657), it does not inflate by 1.55×.
  A large A0.3 cross-check deviation is therefore a **stale-baseline artifact,
  not a live fault**, and that is now stated in the A0 record itself.
  **FATAL-1's fix survives on a stronger receipt than the design argued:**
  `run_phase0_timing` is **byte-identical** between the 98M and 392M runners
  (len 5880 == 5880, a == b True), so R's probe bias cancels **by
  construction**, whatever its absolute level.
* **MAJOR-2 / C2 — R's cancellation assumes an unmeasured invariance, and the
  bias direction is unsafe.** β rises **4.4%** across a 1.64× `t_in` increase, so
  `R = ρ_realized × (β₃₉₂/β₉₈)` with `β₃₉₂/β₉₈` **assumed** to be 1 across a **4×**
  move on the same axis. If `β₃₉₂ > β₉₈`, R **under**-reads and biases Rule P1
  toward NOMINAL — the unsafe direction for a cost-out gate. A true `ρ ≈ 4.8`
  could read `R ≈ 4.0`. **Recorded:** β₉₈(24)=0.8293, β₉₈(40)=0.8657, R stated as
  conditional, plus a **mandatory post-hoc reconciliation** on the **first**
  calibration cell — `>15%` divergence between `R(24)` and the realized ratio
  ⇒ **re-enter Rule P1 before Stage B queues.** Free: that cell produces the
  number anyway.
* **MAJOR-3 / C3 — both `--ceiling-gpuh` paths implemented §3.6's FALLBACK and
  never its primary rule.** Each substituted the runner's projection constant
  `CONTENDED_MULTIPLIER = 3.3` for the **measured `R₈`** — in `a0_rules.py`, in
  the same file that measures `R₈` twenty lines earlier — shipping a backstop
  **≈2.9× looser than pinned** and **32% looser than the spec's own PROJECTED
  placeholder**, so the A0.5 "re-price" *weakened* the breaker. Direction is
  **loose, not tight**, so MAJOR-1(b)'s "fires on every cell" failure is not
  reintroduced and B6 remains the fast breaker. **Both paths now use the
  measured `R₈` when it exists and label which basis was used.**
* **MAJOR-4 / C7 — design §4.4's P4 instrument identification is INVERTED.**
  `:663` sits in `smoke_3_backbone_eval_batch` (backbone-only, `no_grad`, an
  **eval** leg); `:796` sits in `smoke_7_full_graft_train_step` (a **training**
  step). §4.4 says the opposite, so its rationale rests on a false premise. This
  is the **third recurrence of the FATAL-3 class**. **The build did the right
  thing in substance** — P4 reads `max(peak_796, peak_with_eval)` where
  `peak_with_eval` is its own production-shaped two-arm train + `eval_both_arms`
  peak, strictly more conservative than either named line. Two related record
  defects also fixed: `peak_gb_line_1056_co_residency` **asserted a measurement
  never taken** (graft `:1056` merely re-stores smoke_7's own peak) and is
  **deleted**; `eval_pass_delta_gb` is renamed `eval_pass_raises_peak_by_gb`
  with its semantics stated, because `reset_peak_memory_stats()` is deliberately
  not called between the reads, so the quantity is `max(train,eval) − train ≥ 0`
  by construction. **The correct wording is "the eval pass does not raise the
  peak above the training peak at 392M"** — *not* "eval adds 0 GB" and *not*
  "#6's +1.3 GB does not reproduce at scale."
* **MAJOR-5 / C4 — silent cross-scale record poisoning.** Both patched scorers
  defaulted `--outdir` to `~/ncr_kscaling/results` — the **98M** tree, holding
  103 records of record plus `results_depthext6/` (48). The patch **added** a
  `scale` field to both scorers and **no consumer read it**, so a 392M record
  landing there **collides on-key** with its 98M twin and whichever path sorts
  later silently wins — then reads as a 98M number in Rule R-δ *and* in the
  exact-reproduction cross-check. **B5 does not close this: B5 guards the
  checkpoint INPUT; this is the output DESTINATION.** Latent, not executing
  (`kappa_reader` always passes `--outdir`; no spec invokes a scorer) — but
  §4.6's Stage C is a manual, unscripted harvest path. **Closed at both ends:**
  both defaults re-pointed to `~/ncr_scaleaxis/results` (patch `S4`), and
  `rdelta_aggregate.load()` now asserts the `scale` field, refuses a mixture,
  and admits unlabelled legacy records **only** for `98m` and **counted**.

Minors applied: **m3** (the SM-utilisation bug check now enters `overall.gate`,
which previously could read "A0 CLEARS" beside "SUSTAINED <50% IS A BUG"),
**m4** (P4 reads the **max over the four K**, not a hardcoded K=24), **m6**
(reachability counted with strict `>`, matching §6.2's strict SCALE-IMPROVES),
**m7** (`$0` absolutized before A0.4; `nvidia-smi --query-compute-apps`
snapshots **before and after** A0.4, since it measures `R₈` on all 8 GPUs while
`idle_fallback_daemon.sh` and its minutely cron are live; `a0_rules`' exit code
propagated), **m2/C11** (bare-`64`, bare-`12` and `SQUARING_PROFILE` patterns
added to B1 so the disposition table and the sweep cover the same set; the three
`gen_job_specs.py` dispositions are now explicitly marked as *not in the swept
tree*).

**m5 is recorded, not silenced:** Rule P3's `re_priced_gpu_h` is a **step
assignment** (`R(24)` for K≤24, `R(40)` for K≥32), not the `t_in` interpolation
its own action string claimed. Both assignments **over**-price, so the ledger is
safe by ≈3.6 GPU-h; **the label was wrong and is corrected**, the arithmetic is
not touched.

**m1 is closed with a receipt:** the gates' `tree_md5` recorded a stale
`gen_scaleaxis_specs.py` hash (the generator was edited 35 s after the last gate
run). The audit re-ran the gates at K=16 and K=40 against the deployed tree —
**20/20 PASS, `tree_md5` mismatches 0**, B1 hit set **identical (82 hits)** with
the six generator hits shifted by exactly **+8 lines**, an additive edit that
changed no literal B1 detects.

---

## 3. Two changes beyond the literal C-list, flagged for trivial reversal

Both are audit-identified, ≤1-line-equivalent, and disclosed here rather than
folded silently into a condition:

1. **`G3_item11_skip_not_crash`** — the one-line skip guard the audit's §4 D3
   ruling explicitly *recommends* ("One-line skip guard recommended"). It turns
   the graft's standalone-entry-point crash into a labelled SKIP that says out
   loud it is **not** a pass of the `mlp` path. Revert by deleting the `G3`
   entry from `patch_scaleaxis.py`'s `GRAFT_PATCHES`.
2. **Rule P3's `re_priced_basis` label** (m5, above). Text only; no arithmetic
   changed. m5 was deliberately excluded from C5, so this is an addition —
   made because a knowingly-wrong label on the launch-decision surface is
   exactly what C6 exists to prevent.

---

## 4. Two restatements the audit requires before harvest

### 4.1 §5.5's repair is realized as **2-of-4 frozen**, not 3-of-4

§5.5's motivating sentence reads: *"So SCALE-IMPROVES goes from 1-of-8 reachable
to 7-of-8, and from **0-of-4-frozen to 3-of-4-frozen**. That is the repair."*
That was a **conservative-linear projection**. **Measured**, at the elected
`s* = 13` with `δ_depth = 0.095`:

| | reachable at `s*=13` (`H_c > δ_depth`) |
|---|---|
| **Reachable (6)** | K40_frozen (0.0962), K24_frozen (0.0978), K16/K24/K32/K40_trainable |
| **UNREACHABLE (2)** | **K16_frozen (0.0417)**, **K32_frozen (0.0806)** |

So the realized repair is **6-of-8 and 2-of-4-frozen** — 6/8 is **exactly the
rule's guaranteed minimum**, so Rule R-δ is satisfied **as written** and needs
no amendment, but the *magnitude* of the repair on the frozen arm is one third
smaller than §5.5 advertised. **The measured δ\*(13) = 0.095 is also materially
above the projected ≈0.060.** The conservative-linear lower bound is violated in
exactly **1 of 8 cells** at s=13 (K16_frozen: projected ≥0.0624, measured
0.0417) and 0/8 at s=15. The audit checked the counterfactual: **had K16_frozen
met its projection, `δ*` would still be 0.095** (the 3rd-smallest is unchanged at
0.096154) — **the projection error did not move the election.**

### 4.2 δ_depth's noise margin at `s*=13` is **13%, not 45%**

Rule R-δ step 2 justifies its 0.05 floor with *"the **median** within-cell seed
range **at 11 squarings** is 0.0344, so 0.05 clears typical noise with 45%
margin."* That is computed at s=11, but the rule **elects s\*=13**. Measured at
the elected depth: the **median seed range is 0.0844** (2.46× the s=11 value) and
the max is 0.3302, so **δ_depth = 0.095 clears the noise at `s*=13` by only
1.13× — a 13% margin, not 45%.** Three of the six reachable cells have a 98M
within-cell seed range **larger than the equivalence margin itself**
(K16_trainable 0.2458, K24_trainable 0.3302, K40_frozen 0.0962).

This violates no rule and does not block launch — §5.2 already routes around it
(*"the rank test is primary at depth and the magnitude band is secondary"*) — but
**the median leg of that disclosure no longer holds comfortably at the elected
depth, and must be restated in the same sentence that states δ_depth.**

---

## 5. Conditions still open (coordinator-owned, before Stage B)

* **C8** — stage **only** `0190-0195` into `~/queue/pending/`. Filename order
  **orders** claims, it does not **block** them (`queue_worker.sh:119` is
  `for f in $(ls "$PENDING" | sort)`), so with all 24 staged and 8 live workers,
  two workers claim `0200`/`0201` in the same poll cycle. Staging is the
  enforcement; the design's own §7.2(D) sentinel is the rule.
* **C9** — regenerate all 24 specs with `--ceilings-from` after A0 (D7).
* **C10** — discharge §8.3.1's daemon-park procedure and the reader-GPU
  reservation as **enumerated** pre-launch checks, verified by a fresh
  `nvidia-smi --query-compute-apps` read.
* **EXPERIMENT_LOG #16's "eval adds 0 GB" wording** — correct to *"the eval pass
  does not raise the peak"* (C7's log leg; retained by the coordinator).
