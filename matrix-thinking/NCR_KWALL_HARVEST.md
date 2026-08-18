# NCR K-WALL CHARACTERIZATION — HARVEST / VERDICT OF RECORD

**Run:** `kwall_char_orchestrator`, job spec pinned at build commit
`d918074`, launched 2026-08-14T00:19:01Z on GPU 5, finished
2026-08-14T05:37:58Z. `run_status=COMPLETE`, `VALIDITY_CHECK PASS`,
12/12 primary cells `COMPLETED` on first attempt, realized **5.3155
GPU-h** against a declared 15.50 h ceiling.

**Design of record (the authority on every band, rule, and threshold
used below):** `matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md`
(STATUS: RELEASED, audit round 11 CLEAR). Classification executed with
the committed implementation `matrix-thinking/kwall_build/kwall_lib/classify.py`,
not re-derived by hand.

**VERDICT (one line).** `(r26, r28, r30) = (4, 4, 3)` →
**`FRONTIER-AT-K*=30`**, no `[NON-MONOTONE]` tag; the conditional
160K trigger FIRED **unanimously at `K_trig=32`**, i.e. the $0
archive branch — **`CONFIRMED-WALL-AT-160K`**, zero new cells
launched. The 80K CONVERGED-ROBUST frontier is *not* crossed anywhere
in K∈{26,28,30}; this is the design's own "best possible primary
outcome" (§4).

---

## §0 PROVENANCE AND INDEPENDENCE

Everything below was recomputed from the raw per-cell JSONs, not read
off `orchestrator_report.json`. Sources:

| Artifact | Location |
|---|---|
| 12 canonical cell JSONs + 12 `.axis_c_lock.json` | box `~/ncr/results_kwall_characterization/` (copied read-only to scratch) |
| Per-attempt copies | box `.../K{26,28,30}_s{0..3}_attempt1/` |
| Ledger / report | `.../ORCHESTRATOR_LEDGER.json`, `.../orchestrator_report.json` |
| Run log | box `~/queue/logs/kwall_char_orchestrator.log` (1975 lines) |
| Job spec | box `~/queue/completed/kwall_char_orchestrator.json` |
| Gate-1 definition | box `~/ncr/ncr_earlyln_scale.py:317-329` (`_cell_gate1`), bars `:95`/`:97` |
| Front definition | box `~/ncr/run_ncr.py:508-524` (`eval_cell`) |
| Lock hashing | box `~/ncr/ncr_spectral.py:183-218` |
| K=24 anchor (n=12, 80K, d=25) | `experiment-runs/2026-07-12_ncr_mappinglaw_wave1/q2_K24_seedext{,_orig0-3}/` |
| K=32 budget table | `experiment-runs/2026-07-12_ncr_k32_budget/`, `.../mappinglaw_wave1/dratio_K32_d33/` |

Nothing on the box was modified. No STATE.md / EXPERIMENT_LOG.md edit,
no commit.

**Independent re-derivations performed (all agree with the run's own
report):** per-seed Gate-1 conjunction for all 12 cells; the
`failure_front_h` selection rule re-executed from `eval.points`;
`classify()` / `classify_with_interval_logic()` / `trigger()` on the
recomputed triple; `validity_check_core` re-run locally against the
copied raws (PASS) **plus three negative controls that all correctly
FAIL** (inflated `realized`, bogus band label, `conditional.launched`
flipped true) — the check has teeth.

---

## §1 PER-CELL TABLE (every cell, every registered metric)

Config, identical in all 12 cells: earlyln free-write, `d = K+1`
(`--d-override` 27/29/31), encoder `h=64`, batch 256, `anneal_frac`
0.5, 80,000 steps, `runner_tag=ncr_earlyln_scale_v1`,
torch 2.12.1+cu130, host `brev-ukptqsu65`, 0 skipped steps everywhere,
`train.status=COMPLETED` everywhere.

| cell | K | d | params | final loss | `indist_min` | `aer_mean` (bar 0.9K) | AER/K | **Gate-1** | `failure_front_h` | m=(h+3)/K | revivals | `sweep_min_recovered` | reducer flagged | blank-out | agree ok/None/bad (max diff) | rule-trusted h | phase resid mean (max) | GPU-h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `earlyln_K26_s0` | 26 | 27 | 173,723 | 8.674e-4 | 1.0000 | 25.9947 (23.4) | 0.9998 | **CONVERGED** | 101 | 4 | 0 | 0.0000 | False | PASS | 10/31/0 (2.6e-05) | 1,2,3,4 | 0.00756 (0.01068) | 0.4430 |
| `earlyln_K26_s1` | 26 | 27 | 173,723 | 1.123e-3 | 1.0000 | 25.9926 (23.4) | 0.9997 | **CONVERGED** | 101 | 4 | 0 | 0.0000 | False | PASS | 10/31/0 (1.7e-04) | 1,2,3,4 | 0.01114 (0.01605) | 0.4330 |
| `earlyln_K26_s2` | 26 | 27 | 173,723 | 1.546e-4 | 1.0000 | 25.9990 (23.4) | 1.0000 | **CONVERGED** | 205 | 8 | 0 | 0.1082 | False | PASS | 10/31/0 (2.2e-05) | 1–7 | 0.00387 (0.00442) | 0.4405 |
| `earlyln_K26_s3` | 26 | 27 | 173,723 | 1.592e-4 | 1.0000 | 25.9988 (23.4) | 1.0000 | **CONVERGED** | 205 | 8 | 0 | 0.2428 | False | PASS | 10/31/0 (2.7e-05) | 1–7 | 0.00401 (0.00427) | 0.4302 |
| `earlyln_K28_s0` | 28 | 29 | 174,237 | 3.001e-4 | 1.0000 | 27.9979 (25.2) | 0.9999 | **CONVERGED** | 109 | 4 | 0 | 0.0000 | False | PASS | 10/33/0 (8.2e-05) | 1–5 | 0.00733 (0.00770) | 0.4386 |
| `earlyln_K28_s1` | 28 | 29 | 174,237 | 1.795e-3 | 1.0000 | 27.9900 (25.2) | 0.9996 | **CONVERGED** | 109 | 4 | 0 | 0.0000 | False | PASS | 10/33/0 (2.4e-05) | 1,2,3 | 0.00885 (0.01231) | 0.4339 |
| `earlyln_K28_s2` | 28 | 29 | 174,237 | 6.235e-3 | 1.0000 | 27.9553 (25.2) | 0.9984 | **CONVERGED** | 25 | 1 | 0 | 0.0000 | False | PASS | 10/33/0 (2.4e-05) | 1 | 0.01892 (0.02431) | 0.4268 |
| `earlyln_K28_s3` | 28 | 29 | 174,237 | 1.354e-3 | 1.0000 | 27.9888 (25.2) | 0.9996 | **CONVERGED** | 53 | 2 | 0 | 0.0000 | False | PASS | 10/33/0 (5.7e-05) | 1,2,3 | 0.00840 (0.00901) | 0.4579 |
| `earlyln_K30_s0` | 30 | 31 | 174,751 | 2.988e-4 | 1.0000 | 29.9957 (27.0) | 0.9999 | **CONVERGED** | 117 | 4 | 0 | 0.0000 | False | PASS | 10/35/0 (8.4e-05) | 1–5 | 0.00609 (0.00784) | 0.4269 |
| `earlyln_K30_s1` | 30 | 31 | 174,751 | 6.528e-2 | **0.6947** | 28.3487 (27.0) | 0.9450 | **PARTIAL** | 27 | 1 | 0 | 0.0000 | False | PASS | 10/35/0 (6.0e-05) | **none** | **0.37493 (0.67705)** | 0.4304 |
| `earlyln_K30_s2` | 30 | 31 | 174,751 | 2.282e-3 | 1.0000 | 29.9796 (27.0) | 0.9993 | **CONVERGED** | 57 | 2 | 0 | 0.0000 | False | PASS | 10/35/0 (2.9e-05) | 1,2 | 0.01266 (0.01567) | 0.4321 |
| `earlyln_K30_s3` | 30 | 31 | 174,751 | 4.379e-3 | 1.0000 | 29.9555 (27.0) | 0.9985 | **CONVERGED** | 57 | 2 | 0 | 0.0000 | False | PASS | 10/35/0 (3.3e-05) | 1 | 0.01453 (0.01757) | 0.5141 |

Notes on the columns, so nothing is read wrong:

- **Gate-1** is the runner's own conjunction (§5, D4):
  `indist_min = min(recovered_frac@0.9)` over `h∈{1,2,3}` **AND**
  `aer_mean ≥ 0.9·K`. **The single non-CONVERGED cell (`K30_s1`) fails
  the RECOVERY leg alone** (0.6947 < 0.9); its rank leg passes
  (28.3487 ≥ 27.0). No cell anywhere fails on the rank leg — AER/K
  ≥ 0.9450 in 12/12, ≥ 0.9984 in the other 11.
- **`failure_front_h`** is the *Gate-2 / secondary far-depth* metric:
  the smallest in-window ladder rung whose primary (binexp) read drops
  below `recovered_frac@0.9 = 0.9`. The ladder is K-parameterized
  (`_gen_grid`, rungs at `h = m·K − 3`, `m ∈ {1,2,4,8,…}`), so the
  raw `h` values are **not** on a common grid across K. The `m` column
  is the apples-to-apples index (number of full K-cycles survived);
  every rung has the same effective hop `h mod K = K−3`, so the front
  measures a *compounding-error horizon at a fixed relational hop*,
  not a new relational task.
- **blank-out**: `bit_identical`, `grad_exactly_zero`,
  `write_path_alive` and `passed` are all `True` in **12/12** cells
  (the runner asserts this before eval; independently re-read here).
- **agreement checks**: `binexp` vs `loop` max item diff, checked only
  at `h ≤ AGREE_H_MAX = 125` (bar 5e-4). 10 checked / 0 failures per
  cell; the `None` rows are h>125, where the check is not defined.
  Worst diff anywhere: 1.67e-04 (`K26_s1`), inside the bar.
- **reducer_signature**: `no_decay_front=False` and `flagged=False` in
  12/12 (a reducer signature would need `sweep_min ≥ 0.9` *and* no
  front; neither holds anywhere).
- **`post_front_revivals` is empty in 12/12** — no cell recovers at a
  rung deeper than its own front.

### §1.1 Ladder recovery curves (the raw evidence behind every front)

`recovered_frac@0.9` at each ladder rung (`h: value`); the front is the
first rung < 0.9.

| cell | ladder (h : recovered_frac@0.9) |
|---|---|
| `earlyln_K26_s0` | 23:1.000 49:0.999 **101:0.161** 205:0.000 413:0.000 829:0.000 1661:0.000 3325:0.000 |
| `earlyln_K26_s1` | 23:1.000 49:0.956 **101:0.011** 205:0.000 413:0.000 829:0.000 1661:0.000 3325:0.000 |
| `earlyln_K26_s2` | 23:1.000 49:1.000 101:0.995 **205:0.118** 413:0.000 829:0.000 1661:0.000 3325:0.000 |
| `earlyln_K26_s3` | 23:1.000 49:1.000 101:1.000 **205:0.265** 413:0.000 829:0.000 1661:0.000 3325:0.000 |
| `earlyln_K28_s0` | 25:1.000 53:1.000 **109:0.004** 221:0.000 445:0.000 893:0.000 1789:0.000 3581:0.000 |
| `earlyln_K28_s1` | 25:1.000 53:0.929 **109:0.003** 221:0.000 445:0.000 893:0.000 1789:0.000 3581:0.000 |
| `earlyln_K28_s2` | **25:0.857** 53:0.001 109:0.000 221:0.000 445:0.000 893:0.000 1789:0.000 3581:0.000 |
| `earlyln_K28_s3` | 25:1.000 **53:0.883** 109:0.001 221:0.000 445:0.000 893:0.000 1789:0.000 3581:0.000 |
| `earlyln_K30_s0` | 27:1.000 57:1.000 **117:0.673** 237:0.000 477:0.000 957:0.000 1917:0.000 3837:0.000 |
| `earlyln_K30_s1` | **27:0.000** 57:0.000 117:0.000 237:0.000 477:0.000 957:0.000 1917:0.000 3837:0.000 |
| `earlyln_K30_s2` | 27:1.000 **57:0.177** 117:0.000 237:0.000 477:0.000 957:0.000 1917:0.000 3837:0.000 |
| `earlyln_K30_s3` | 27:0.917 **57:0.001** 117:0.000 237:0.000 477:0.000 957:0.000 1917:0.000 3837:0.000 |

Residue-sweep windows (the source of `sweep_min_recovered`): K=26
h∈[183,208] (n=26), K=28 h∈[197,224] (n=28), K=30 h∈[211,240] (n=30) —
one full residue class per K, as `_gen_grid` specifies.

### §1.2 Trust screen (τ = 0.2, identical in all 12 cells)

`trust_screen.per_h[h] = {T, rule_trusted}`; `rule_trusted ⇔ T ≤ τ`.
The a-priori spectral rule certifies only a short prefix of the h grid,
and that prefix **shrinks with K**:

| cell | # rule-trusted h | trust horizon (largest trusted h) | T(h=1) | # T=inf | front rule-trusted? |
|---|---|---|---|---|---|
| K26_s0 | 4 | 4 | 0.0347 | 0 | No |
| K26_s1 | 4 | 4 | 0.0338 | 0 | No |
| K26_s2 | 7 | 7 | 0.0125 | 0 | No |
| K26_s3 | 7 | 7 | 0.0148 | 0 | No |
| K28_s0 | 5 | 5 | 0.0277 | 0 | No |
| K28_s1 | 3 | 3 | 0.0463 | 0 | No |
| K28_s2 | 1 | 1 | 0.1003 | 1 | No |
| K28_s3 | 3 | 3 | 0.0386 | 0 | No |
| K30_s0 | 5 | 5 | 0.0248 | 0 | No |
| K30_s1 | **0** | — | **606.6** | **35** | No |
| K30_s2 | 2 | 2 | 0.0631 | 1 | No |
| K30_s3 | 1 | 1 | 0.1003 | 1 | No |

**The front is outside the rule-trusted region in 12/12 cells.** Its
numerical validity rests on the fp64 shadow check, not the a-priori
rule: every ladder point in every cell carries
`trust_label = SHADOW-VERIFIED`, `|shadow_delta| ≈ 5e-8` vs the
`SHADOW_BAR = 5e-3`, and `numeric_divergent_shadow = False` in
**0 of 516 eval points across all 12 cells**. There are **zero
`UNTRUSTED` points anywhere**. So the far-depth numbers are numerically sound but
*rule-uncertified* — exactly the a-priori-screen-only status the NCR
program already recorded for this rule. Gate-1's own h∈{1,2,3} are
rule-trusted in 8/12 cells and SHADOW-VERIFIED in the remaining 4
(`K28_s2` h=2,3; `K30_s1` all three; `K30_s2` h=3; `K30_s3` h=2,3).

### §1.3 Deep probe

| cell | `A_eff_rank` (4 probe examples) | `aer_mean` | `phase_resid_max_per_example` | `phase_resid_max_mean` | `c_star_per_example` | `scale_corrected_residual` |
|---|---|---|---|---|---|---|
| `earlyln_K26_s0` | 25.995 25.994 25.995 25.995 | 25.9947 | 0.0056 0.0072 0.0107 0.0068 | 0.00756 | 1.670 1.695 1.695 1.687 | 0.0272 0.0272 0.0284 0.0254 |
| `earlyln_K26_s1` | 25.992 25.993 25.992 25.993 | 25.9926 | 0.0081 0.0114 0.0160 0.0090 | 0.01114 | 1.778 1.769 1.795 1.744 | 0.0334 0.0320 0.0346 0.0309 |
| `earlyln_K26_s2` | 25.999 25.999 25.999 25.999 | 25.9990 | 0.0033 0.0037 0.0041 0.0044 | 0.00387 | 1.303 1.277 1.314 1.301 | 0.0114 0.0119 0.0123 0.0120 |
| `earlyln_K26_s3` | 25.999 25.999 25.999 25.999 | 25.9988 | 0.0042 0.0043 0.0035 0.0041 | 0.00401 | 1.372 1.384 1.386 1.387 | 0.0116 0.0140 0.0129 0.0127 |
| `earlyln_K28_s0` | 27.998 27.998 27.998 27.998 | 27.9979 | 0.0073 0.0076 0.0067 0.0077 | 0.00733 | 1.035 1.038 1.046 1.033 | 0.0165 0.0165 0.0165 0.0168 |
| `earlyln_K28_s1` | 27.990 27.990 27.990 27.990 | 27.9900 | 0.0119 0.0056 0.0056 0.0123 | 0.00885 | 1.796 1.793 1.803 1.797 | 0.0369 0.0366 0.0368 0.0375 |
| `earlyln_K28_s2` | 27.947 27.962 27.955 27.958 | 27.9553 | 0.0189 0.0243 0.0194 0.0131 | 0.01892 | 1.826 1.834 1.823 1.827 | 0.0811 0.0705 0.0773 0.0735 |
| `earlyln_K28_s3` | 27.991 27.988 27.988 27.988 | 27.9888 | 0.0090 0.0085 0.0084 0.0076 | 0.00840 | 1.648 1.602 1.673 1.649 | 0.0351 0.0391 0.0382 0.0409 |
| `earlyln_K30_s0` | 29.995 29.996 29.996 29.996 | 29.9957 | 0.0078 0.0052 0.0062 0.0051 | 0.00609 | 1.080 1.061 1.068 1.076 | 0.0223 0.0206 0.0201 0.0220 |
| `earlyln_K30_s1` | **28.373 28.210 28.632 28.180** | **28.3487** | **0.6771 0.1917 0.2469 0.3841** | **0.37493** | 1.811 1.789 1.921 1.820 | **0.2933 0.2872 0.2901 0.2890** |
| `earlyln_K30_s2` | 29.977 29.979 29.980 29.983 | 29.9796 | 0.0157 0.0124 0.0102 0.0123 | 0.01266 | 1.862 1.871 1.868 1.861 | 0.0554 0.0518 0.0484 0.0459 |
| `earlyln_K30_s3` | 29.954 29.955 29.956 29.956 | 29.9555 | 0.0168 0.0099 0.0138 0.0176 | 0.01453 | 1.749 1.691 1.760 1.754 | 0.0715 0.0717 0.0723 0.0731 |

`A_eff_rank` is within 0.045 of the full K in 11/12 cells (0.9984 ≤
AER/K ≤ 1.0000); `K30_s1` is the sole outlier at 28.35/30 = 0.9450,
which still clears the 0.9 bar. **The rank leg never binds in this
study.** `phase_resid_max_mean` separates `K30_s1` (0.3749) from every
other cell (0.0039–0.0189) by 20×.

---

## §2 BAND CLASSIFICATION — the design's own rules, executed

**Inputs, recomputed from raws (never from the report):**

| K | n_completed | seeds CONVERGED | rate `r_K` | resolution state |
|---|---|---|---|---|
| 24 | (archive, n=12) | 12/12 | **fixed `r24=4`** (ROBUST) | `EXACT`, by design constant `R24_FIXED` |
| 26 | 4/4 | 4 | **4** | `EXACT` |
| 28 | 4/4 | 4 | **4** | `EXACT` |
| 30 | 4/4 | 3 | **3** | `EXACT` |
| 32 | (archive, 80K) | 0/4 | **fixed `r32=0`** (NOT ROBUST) | `EXACT`, by design constant `R32_FIXED` |

Every K has 4/4 `status=="COMPLETED"` canonical cells, so §4's D5 rule
permits reading a rate at all three K's; no K is `INCOMPLETE-AT-K`, no
interval logic is engaged, `interval_resolved_Ks = []`.

The archive anchors were re-verified from the raw JSONs this harvest:
K=24 n=12 reads `indist_min = 1.000` and AER/K ∈ [0.9968, 0.9999] in
12/12 (⇒ `r24=4`); K=32 at 80K/`d=33` reads `indist_min ∈
[0.4643, 0.8711]`, AER/K ∈ [0.9269, 0.9679], 0/4 CONVERGED (⇒
`r32=0`). Both match the design's §3 to the digit.

**Execution** (`kwall_lib.classify`, whose import-time self-checks — the
125-outcome partition and the 1000-vector G5-gated trigger split —
both passed):

```
classify(4, 4, 3)                      -> ('FRONTIER-AT-K*=30', False)
classify_with_interval_logic(4, 4, 3)  -> ('FRONTIER-AT-K*=30', False, None, [], None)
```

**Why, rule by rule (§5's ordered six-rule procedure):**

1. `ROBUST(r30) = (3 ≥ 3) = True` **and** `r32 = 0 ≤ 1` → **rule (1)
   fires: `FRONTIER-AT-K*=30`.** Rules 2–6 are never reached (first
   match wins; the procedure is a partition by construction).
2. `[NON-MONOTONE]` tag: the ROBUST sequence
   `[r24, r26, r28, r30, r32] = [T, T, T, T, F]` is monotone
   `True…True False…False` ⇒ **tag = False**. No internal dip.

**Per-K reading of the single band label.** §5 is explicit that
`classify()` returns exactly ONE study-level label over the full
triple; there is no per-K band object. The per-K content of that label
is:

| K | rate | ROBUST? | what the band says about this K |
|---|---|---|---|
| 26 | 4/4 | Yes | inside the frontier; 80K-only read, **not** budget-qualified (§7's "only ONE K gets budget-qualified") |
| 28 | 4/4 | Yes | inside the frontier; 80K-only read, not budget-qualified |
| 30 | 3/4 | Yes (3 ≥ 3) | **the frontier sits here** — the largest K measured CONVERGED-ROBUST at 80K |
| 32 | 0/4 (archive) | No | the first non-ROBUST rung; §5 rule (1) subsumes the old "NO-WALL-BELOW-32" band into exactly this reading |

**Fragility disclosure (not part of the verdict, stated so the reader
can price it).** The band hangs on one seed. Re-running `classify` at
neighbouring triples: `(4,4,4) → FRONTIER-AT-K*=30`; `(4,3,3) →
FRONTIER-AT-K*=30`; `(4,4,2) → GRADUAL-DECAY`; `(4,4,1) →
FRONTIER-AT-K*=28`. So had a *second* K=30 seed failed, the study would
have read `GRADUAL-DECAY` and the trigger would have dispatched a PAID
4-cell 160K arm at K=30. The verdict is one seed away from a different
label and a ~1.7 GPU-h different run. That is a property of an n=4
pre-registration, disclosed, not a defect in the execution.

---

## §3 CONDITIONAL 160K ARM — TRIGGER **FIRED**, $0 BRANCH

```
trigger(4, 4, 3)                 -> (K_trig=32, resolution='unanimous', detail=None, diag=None)
trigger_raw_scan_blocked(4,4,3)  -> False      # the raw K-scan decided on its own
trigger_candidate_set(4,4,3)     -> None       # unanimous, so no candidate set to disclose
```

**Derivation (§4's F2 scan).** Scan K = 26, 28, 30, 32 in order for the
smallest K with rate `< 3`: `r26=4` (no), `r28=4` (no), `r30=3` (no —
3 is not `< 3`), `r32=0` (**yes**) ⇒ `K_trig = 32`. One branch only (no
`AMBIGUOUS` K), hence `unanimous`, no tie-break. G5's precondition then
requires the whole-study band to be DECIDED before dispatch — it is
(`FRONTIER-AT-K*=30`, not `INCOMPLETE-AT-K`), so the trigger is not
forced to `TRIGGER-UNRESOLVED`.

**What firing at 32 means (§4 bullet 2, §5's "At `K_trig=32`"):** *no
new cells are launched.* The disambiguation is read off the already
archived K=32 budget table at the **matched 160K row only**. Verified
directly from the raw JSONs this harvest:

| K=32 budget | seeds CONVERGED | `indist_min` per seed |
|---|---|---|
| 80K (1×) | **0/4** | 0.4643, 0.5170, 0.6875, 0.8711 |
| 160K (2×) | **1/4** | 0.7944, **0.9015**, 0.5818, 0.8865 |
| 320K (4×) | 2/4 (context only) | 0.8754, **0.9124**, **0.9118**, 0.8965 |

160K rate = 1/4 ≤ 1/4 ⇒ **`conditional.qualifier_band =
CONFIRMED-WALL-AT-160K`**, at $0 incremental GPU-h. Per §5's own gloss
correction (KW3.6) this reads "K=32 does not clear CONVERGED-ROBUST at
160K," never "does not improve at all." The 320K rise to 2/4 is
disclosed as archive **context only** and was not used to decide any
band (§5's E3 fix).

**Report-shape check.** `conditional.launched = false`,
`conditional.per_seed = []`, and the conditional outdir
`~/ncr/results_kwall_characterization_160k` **does not exist on the box**
— zero conditional cells, exactly as the $0 branch requires.
`validity_check`'s universal assertion 7 clause (b)
(`launched is False and K_trig == 32`) is what licenses a
non-null `qualifier_band` here; my local re-run confirms it fires, and
flipping `launched` to `true` correctly produces the U7 failure.

**Implication for follow-on work.** The 160K disambiguator was spent at
K=32 (for free) and therefore **at no K in {26,28,30}**. Combined with
§7's "only ONE K gets budget-qualified," this study says *nothing*
about whether K=26/28/30 would behave differently at 160K — the
speed-vs-wall question is answered only at K=32, where the answer is
"still walled at matched budget." Any claim that K=30 is the last live
rung is an **80K-budget** claim.

---

## §4 SCIENTIFIC VERDICT — does the wall move with K?

### §4.1 The PRIMARY (recovery) leg: **no, not within K ≤ 30**

The design's recovery leg is Gate-1's `indist_min` at h∈{1,2,3}. It is
**1.0000 in 11 of 12 cells** and 0.6947 in one (`K30_s1`). Across
K∈{26,28,30} the CONVERGED rate is 4/4, 4/4, 3/4 — a total of **1
recovery failure in 12 cells**, and it sits at the largest K. The
frontier has not been crossed anywhere in the window this study opened.
Against the anchors: K=24 is 12/12, K=32 is 0/4 at the same budget. So
the 80K frontier is now localized to the interval **(30, 32]** rather
than the pre-study **(24, 32]** — a genuine 4× narrowing, which is
exactly what this filler wave was chartered to produce.

The one failure is not mysterious: `K30_s1` has a final training loss
of 6.5e-2 (vs 1.5e-4–6.2e-3 elsewhere, a 10–400× outlier), a phase
residual of 0.375 (20× every other cell), `T(h=1) = 607` vs
0.012–0.100 elsewhere, and zero rule-trusted h. It is a
badly-converged seed, not a subtly-different one.

### §4.2 The SECONDARY (far-depth) leg: the front DOES fall, but the evidence is weak-to-moderate and non-monotone at the low end

**Scope warning, stated first.** `failure_front_h` is the
**Gate-2/secondary far-depth** metric. The design (a) never registers
it as a band determinant, (b) records it as a *confounded, noisy*
instrument (§3's KW1.7/D6, and the n=4→n=12 extension's "33% on the
looser metric, 0/12 on the strict one"), and (c) lists "no new
far-depth residue arithmetic" and "no claim of a flagship-level
capability result" as non-goals (§7). **Everything in this subsection
is post-hoc analysis, not a pre-registered outcome, and is reported as
such.**

Fronts in ladder-index units (`m = (h+3)/K`, the number of full
K-cycles survived at fixed effective hop K−3), with the K=24 archive
(n=12, same harness, same budget, same `d=K+1`) as the anchor:

| K | n | m per seed (sorted) | median m | median raw h | ~87.5% distribution-free CI for the median (= [min, max] at n=4) | mean log₂m |
|---|---|---|---|---|---|---|
| 24 (archive) | 12 | 1, 4,4,4,4,4,4,4, 8,8,8,8 | 4 | 93 | [4, 8] (= [x₍₄₎, x₍₉₎], 85.4% coverage; [x₍₃₎, x₍₁₀₎] = [4,8] at 96.1%) | 2.167 |
| 26 | 4 | 4, 4, 8, 8 | 6 | 153 | **[4, 8]** | 2.500 |
| 28 | 4 | 1, 2, 4, 4 | 3 | 81 | **[1, 4]** | 1.250 |
| 30 | 4 | 1, 2, 2, 4 | 2 | 57 | **[1, 4]** | 1.000 |
| 32 (archive) | 4+4 | 1 in 8/8 cells at 160K **and** 320K | 1 | 29 | [1, 1] | 0.000 |

**Say the overlap plainly: the per-seed ranges overlap heavily.**
K=26's interval [4,8] and K=28's/K=30's [1,4] touch at m=4, and m=4
occurs in *every* group including K=24. No adjacent pair of K's is
separated at α=0.05 by any test I ran:

| comparison | statistic | exact two-sided permutation p |
|---|---|---|
| K26 vs K28 (mean log₂m) | +1.250 | 0.171 |
| K26 vs K30 (mean log₂m) | +1.500 | 0.086 |
| K28 vs K30 (mean log₂m) | +0.250 | **1.000** |
| K24 vs K26 (mean log₂m) | −0.333 | 0.712 |
| K24 vs K30 (mean log₂m) | +1.167 | 0.056 |
| K26 vs K28+K30 pooled | +1.375 | 0.026 |
| monotone trend, K∈{26,28,30}, n=12 | Spearman ρ(K, log₂m) = −0.667 | 0.025 (2×10⁵ MC) |
| monotone trend, K∈{24,…,30}, n=24 | Spearman ρ = −0.486 | 0.018 (2×10⁵ MC) |

A finer, less quantized statistic — **depth-area**, the sum of
`recovered_frac@0.9` over the 8 ladder rungs (max 8, one unit per
doubling; also post-hoc) — tells the same story with the same strength:

| K | depth-area per seed | mean | exact p vs next |
|---|---|---|---|
| 24 (n=12) | 0.642–3.872 | 2.454 | vs K26: 0.701 |
| 26 | 1.967, 2.161, 3.113, 3.264 | 2.626 | vs K28: 0.057 |
| 28 | 0.858, 1.884, 1.933, 2.004 | 1.670 | vs K30: 0.457 |
| 30 | 0.000, 0.918, 1.177, 2.673 | 1.192 | — |
| pooled K≤26 (n=16) vs K≥28 (n=8) | — | 2.497 vs 1.431 | **0.005** |

**Three findings that constrain how strongly this can be stated:**

1. **The K=24 anchor breaks monotonicity at the low end.** K=24 (n=12)
   has median m=4 and mean log₂m = 2.167; K=26 (n=4) is *higher*
   (median 6, mean 2.500), p=0.71 for the difference. So over
   K=24→26→28→30 the medians go 4 → 6 → 3 → 2. **There is no monotone
   law here that the data supports.** What the data supports is a
   *level shift* between {24, 26} and {28, 30}, not a slope.
2. **Fixed-K seed noise is enormous — and partially, but not fully,
   explains the spread.** At the *single* K=24, twelve seeds span the
   entire ladder range m∈[1,8]. Enumerating all C(12,4)=495 four-seed
   subsets of that pool: 40.6% have median ≥ 6 (K=26's observed value),
   so **K=26's front distribution is statistically indistinguishable
   from seed noise at K=24.** But 0 of 495 subsets reach a median ≤ 3,
   so K=28's and K=30's medians lie outside anything the K=24 seed pool
   can produce at n=4. (This is a resampling bound on within-K
   variation, not an independent test — the subsets are not
   independent.) A cleaner binary version: "front collapses before 3
   full K-cycles" (m ≤ 2) occurs in 1/16 cells at K≤26 and 5/8 at
   K≥28, Fisher exact two-sided **p = 0.0069**.
3. **The front is not independent noise — it tracks how well the seed
   converged.** Across the 12 cells, log₂m correlates almost perfectly
   with every convergence-quality instrument in the record:

   | correlate | Spearman ρ with log₂(m_front) | MC p |
   |---|---|---|
   | # rule-trusted h (trust horizon) | **+0.938** | <1e-4 |
   | log₁₀ T(h=1) (operator conditioning) | **−0.924** | 1e-4 |
   | log₁₀ final training loss | **−0.924** | 1e-4 |
   | `phase_resid_max_mean` | **−0.895** | 2e-4 |

   And K itself predicts those quality measures only weakly:
   ρ(K, #trusted h) = −0.626 (p=0.037); ρ(K, log₁₀ loss) = +0.591
   (p=0.053). **Mechanistically, the chain is K → (weakly) worse
   operator conditioning at a fixed 80K budget → (strongly) shallower
   compounding-error horizon.** The front is a conditioning readout,
   not a separate capability boundary.

**Verdict on §4's question.** The *recovery-leg* wall does **not** move
within K∈{26,28,30}: it stays above K=30 at 80K, with 1 failure in 12
cells. The *far-depth front* does fall — pooled K≤26 vs K≥28 separates
at p≈0.005, and the trend statistic clears 0.05 — but **no adjacent-K
comparison is significant, K=28 vs K=30 is a dead heat (p=1.0), the
K=24 anchor is below K=26 so the sequence is not monotone, and the
whole effect is confounded with a convergence-quality gradient that the
same 80K budget produces.** State it as: *at a fixed 80K budget the
far-depth horizon degrades between K≤26 and K≥28, on a secondary
instrument the design pre-registered as noisy, with n=4 per K and
heavily overlapping per-seed ranges.* Do not state it as a law, a
scaling exponent, or a per-K wall.

**One counter-consideration worth carrying forward:** at K=32 the front
is pinned at m=1 in 8/8 cells at **both** 160K and 320K, while Gate-1's
own rate rises 0→1→2 over 1×/2×/4×. So at K=32 the front is
budget-*insensitive* while recovery is budget-*sensitive*. If that
holds at K=28/30, the shallow fronts there are not merely "needs more
steps." This is a hypothesis the present data cannot test — nothing in
this study measures a front at any budget other than 80K.

---

## §5 INTEGRITY

**Ledger and charging (design §4's ORCHESTRATOR CONTRACT + §R7's J6
bound).**

| check | result |
|---|---|
| `realized_gpu_h_final` | 5.315480022761556 |
| Σ `attempts[].elapsed_h` | 5.315480022761556 — **exact match** (universal assertion 3) |
| attempts | 12, all `arm=primary`, all `attempt_n=1`, all `COMPLETED`; 0 retries, 0 `GATE-REFUSED`, 0 `PERSISTENTLY-ABORTED`, `open_attempt=null` |
| `ceiling_charged` | **false in 12/12**; `charged_vs_measured.ceiling_charged_gpu_h = 0`, fraction 0.0 (independently recomputed, matches) |
| vs hard gate 15.00 / declared pool 15.50 | 5.3155, **10.18 h of headroom** |
| vs the honest worst-case spend bound **15.3737** (§R7 J6: `R_N + 12τ + 32s`) | 5.3155 ≤ 15.3737, headroom 10.06 h — the bound is never approached |
| `d_override == K+1` in every row | true (universal assertion 4) |
| smoke K26/K28/K30 | PASS/PASS/PASS (ran 00:17–00:18Z, before the orchestrator log opens at 00:19Z; smoke outdir exists on the box with all three K subdirs) |
| GPU-h / wall-clock | 5.3155 / 5.3158 = **0.99993** — strictly sequential, one subprocess in flight, no idle gap; the invariant the whole worst-case bound depends on |

**Charging direction is conservative, as designed.** Ledger
`elapsed_h` exceeds each cell's self-reported `gpu_h` by at most
0.00083 h (3.0 s) — the orchestrator's `dispatch_ts` timer captures
subprocess startup that the cell's own timer does not. Total cell
`gpu_h` = 5.3074 vs ledger 5.3155. Every row over-charges slightly;
none under-charges. The observed per-row leak (≤0.00083 h) sits well
inside the design's derived startup allowance `s = 0.0053 h`, and since
all 12 rows were live-folded (no crash reconstruction), the `32·s`
reconstruction term of the bound was never drawn on at all.

**Axis-C lock hashes — consistent, and consistent in the right sense.**
The lock is **per-cell**, not a shared constant: `write_axis_c_lock`
hashes `{cell_id, K, locked_at_utc, mean_predicted_curve,
per_example_curves, phase_resid_max_per_example, phase_resid_max_mean,
c_star_per_example}`, so 12 distinct hashes are the *correct* outcome
and identical hashes would indicate a bug. Verified for all 12 cells:

- each lock file's stored `lock_sha256` equals the SHA-256 recomputed
  over its own content with that field removed — **12/12 verify**
  (the same check `verify_axis_c_lock` performs, re-run here
  independently);
- each lock's hash equals the `axis_c_lock_sha256` recorded in its
  cell JSON — **12/12 match**;
- each lock's `cell_id` and `K` match its cell's — **12/12**;
- all 12 hashes are pairwise distinct, as they must be.

**Canonical-path integrity.** For all 12 cells the canonical top-level
JSON is **byte-identical (md5)** to its `K{K}_s{s}_attempt1/` copy —
the copy-then-fold atomic write produced no divergence.

**`validity_check` re-run independently:** PASS, with three negative
controls all correctly failing (see §0).

**Two provenance gaps, disclosed:**

1. **Per-cell `git_commit` is `"UNKNOWN"` in all 12 JSONs.** The
   orchestrator's on-box `git` call failed (`fatal: not a git
   repository`, log line 4) because `~/ncr` is not a git checkout. The
   commit recorded in `orchestrator_report.json`
   (`d918074bfd99ca2979231924028612f96c670cf6`) came from the job
   spec's `--git-commit` flag — i.e. it is a **declared**, not an
   on-box-verified, provenance stamp. `d918074` does exist in this repo
   ("K-wall build Rev-1 … build audit R2 dispatched", 2026-08-12) and
   nothing under `matrix-thinking/kwall_build/` has changed since
   except the addition of `BUILD_AUDIT_R2.md`, so the stamp is
   consistent — but it is an unverified assertion, not a hash of what
   ran. The same gap exists in the K=24 archive cells
   (`git_commit: "UNKNOWN"`), so it is pre-existing, not new.
2. **The three micro-smokes ran before the job log opens** and their
   PASS verdicts are recorded only as three strings in the report; the
   smoke output directories exist on the box but their contents were
   not re-verified in this harvest.

**Cross-run comparability of the K=24 anchor used in §4.2:** same
`runner_tag` (`ncr_earlyln_scale_v1`), same host, same torch build,
same batch 256, same `anneal_frac` 0.5, same 80,000 steps, same
`d=K+1`, and the same K-parameterized `_gen_grid` ladder (K=24 rungs
21/45/93/189… at the same m indices). The design itself already treats
this run set as its own §3 anchor for `indist_min`. Comparable.

---

## §6 WHAT THIS LICENSES — AND WHAT IT DOES NOT

**Licensed:**

- **"At 80K steps, `d=K+1` earlyln free-write is CONVERGED-ROBUST
  through K=30 and fails at K=32."** The 80K frontier is now localized
  to (30, 32], down from (24, 32]. Cite as `FRONTIER-AT-K*=30`, n=4 per
  K, rate 4/4, 4/4, 3/4, against archive anchors 12/12 at K=24 and 0/4
  at K=32.
- **"K=32 is still walled at matched 160K budget"** (1/4 CONVERGED) —
  free, from the archive, on the same footing as a paid cell would have
  been.
- **"The recovery and rank legs dissociate; the rank leg never binds
  below K=32."** AER/K ≥ 0.9450 in 12/12 here, and the sole Gate-1
  failure is recovery-only.
- **"The single-state bottleneck held"**: blank-out PASS 12/12
  (bit-identical, exactly-zero gradient, write path alive).
- **Numerical soundness of every read used**: 0 divergent-shadow
  points, 0 UNTRUSTED points, agreement max 1.7e-4 vs a 5e-4 bar, no
  post-front revivals anywhere.
- **A methodological note worth publishing on its own:** the Gate-2
  far-depth front is ~0.9-correlated with training loss, phase residual,
  and the trust horizon — i.e. it is a convergence/conditioning readout,
  not an independent capability axis. That is a useful instrument
  finding for anyone tempted to read fronts as capability walls.

**NOT licensed:**

- **Any monotone "wall moves with K" law.** K=24's own n=12 front
  distribution sits *below* K=26's, no adjacent-K pair separates at
  0.05, and K=28 vs K=30 is p=1.0. The honest statement is a level
  shift between K≤26 and K≥28 on a secondary, pre-registered-noisy
  instrument.
- **Any budget-general claim at K∈{26,28,30}.** Per §7 only ONE K gets
  budget-qualified, and the trigger spent it on K=32. K=26/28/30 are
  **80K-only** reads. "K=30 is the last live rung" must always carry
  "at 80K steps."
- **Any flagship capability claim.** §7 is explicit: this is a
  trainability-characterization filler wave; its outputs feed the
  flagship's "last live rung" bookkeeping at most, never a headline.
- **Any claim about K between 30 and 32, or above 32.** Not measured;
  §7 bars new K≥32 cells.
- **Any claim that the K=30 result is robust to seeds.** It is 3/4, one
  seed from `GRADUAL-DECAY` (see §2's fragility disclosure), and the
  failing seed is a poorly-converged outlier rather than a
  cleanly-walled one.

**Recommended next step for the K-ladder, if any (not a decision — that
is the coordinator's):** the highest-information cheap follow-on is
*not* more K's. It is either (a) more seeds at K=30 to resolve whether
3/4 is the true rate (the whole band hinges on it, and n=4 cannot tell
3/4 from 2/4 or 4/4), or (b) a 160K arm at K=30 to test whether its
sub-unit front and its one failed seed are budget-limited — which the
K=32 evidence (front budget-insensitive, recovery budget-sensitive)
predicts would move recovery but not the front. Either is ~1.7 GPU-h at
this cell cost. Both are outside this design's charter and would need
their own pre-registration.

---

*Harvest performed 2026-08-17 from the raw artifacts named in §0. This
document is the only repo file written. No box state was modified, no
STATE.md / EXPERIMENT_LOG.md edit, no commit.*
