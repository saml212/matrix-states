# NCR K-WALL CHARACTERIZATION — FOCUSED AUDIT ROUND 3 (E1–E6 DISCHARGE VERIFICATION)

**VERDICT: REV-REQUIRED — 3 FATAL / 3 MAJOR / 5 MINOR.**

**Per-disposition:** E1 **NOT-DISCHARGED** · E2 **DISCHARGED** ·
E3 **DISCHARGED** · E4 **PARTIAL** · E5 **DISCHARGED** ·
E6 **PARTIAL**.

**Integrity: PASS.** Header verified as expected
(`DRAFT-R2 — POST-AUDIT-2, AWAITING FOCUSED AUDIT ROUND 3`, `:3-4`) —
no mismatch to report. `§A1-ADJUDICATION` + `§R1` + `§A2-ADJUDICATION`
(old `:857-1033` → new `:1195-1371`, 177 lines) are **byte-identical**
to `HEAD~1` (md5 `9d1c064229e9295d525824db07b85240` both sides). The
six-rule classification text is likewise byte-identical (md5
`492da18360416b38d82704c0af0b27bc`), confirming E5's "rule text
UNCHANGED" claim.

Scope note: this is a FOCUSED round. Rounds 1–2 hold as settled for
material Rev 2 did not touch. Everything below is either an E1–E6
disposition check, or a defect introduced/left standing by Rev 2's own
changes.

**The verified core is clean and I could not break it.** Every
load-bearing number in §3/§4/§5 that I recomputed from raws reproduced
exactly (see §7). The three FATALs are all in one place: the **E1
cumulative-cap machinery**, which is the mechanism Rev 2 introduced to
fix KW3.1's broken 15.00h bound. It does not hold, and its true
worst case (≈28.8–39.8 GPU-h) is **worse than the 30.00h defect it was
created to close**.

---

## §0 SUMMARY TABLE

| # | Sev | One line |
|---|---|---|
| KW4.1 | **FATAL** | `ABORTED-BUDGET` cells burn full-ceiling GPU-h but are invisible to E1's gate on **two** independent grounds (status filter *and* a missing `gpu_h` field), and a retry overwrites the first attempt's `elapsed_s`. True worst case 28.80h (primary only) / **39.80h** (with conditional arm) against a claimed ≤15.20h. |
| KW4.2 | **FATAL** | The induction's premise is false: `realized_before_last_batch` is a COMPLETED-only read, not the true cumulative spend. Batch-atomicity closes only the within-batch hole. Concrete **abort-free** counterexample reaching **20.92h**, driven by the trigger rule's own most-natural reading. |
| KW4.3 | **FATAL** | E1 has no owner in the repo's actual dispatch path (`queue_worker.sh` / `idle_fallback_daemon.sh` carry zero budget state and have no "batch"), and a cumulative cap is by construction an **intra-wave dependency** — which §6's own cited pool contract forbids. §6 asserts pool-eligibility *and* E1 simultaneously; they are mutually exclusive. |
| KW4.4 | **MAJOR** | `ceiling(cell)` denotes two different numbers: the gate charges `max(2×nominal,1.0h)` (1.0211/1.1073/1.1946) while the runner enforces the shared CLI `1.20`/`2.32`. Under-charge up to **1.14h**, so the idealized bound is ≈16.34h, not 15.2016h. |
| KW4.5 | **MAJOR** | KW3.4's deadlock is **reinstated at the trigger**: §4's precondition ("defers … until it resolves") is UNCHANGED from Rev 1 and a `PERSISTENTLY-ABORTED` cell is now TERMINAL, so it never resolves. Worse, 11 configurations exist where interval logic DECIDES the band but `K_trig` remains ambiguous between two K's — a ~4.3–4.6 GPU-h arm with no rule to place it. |
| KW4.6 | **MAJOR** | The KW2.8/KW3.13 accepted-risk rests on a **false internal cross-reference**: the `d=K+1` micro-smoke it twice claims is "recorded above"/"this design specifies" exists nowhere in the living body — only in `§R1`, which Rev 2 itself declares frozen and non-operative. The one smoke gate specific to this design's config family is unowned. |
| KW4.7 | MINOR | Interval logic frequently cannot decide, undisclosed: at `r_known=2` bands differ in 64%/68%/**100%** of configurations (K=26/28/30); two singly-incomplete K's decide in only 45–54%. |
| KW4.8 | MINOR | "`INCOMPLETE-AT-K` **for that K**" is a category error — the six-rule procedure returns one global label for the triple; there is no per-K band. The study-level outcome is left implicit and sits outside the §5 Σ=125 partition. |
| KW4.9 | MINOR | "actual TRAINING time is bounded above by its ceiling" is false: the `elapsed > ceiling_s` test fires only at `log_every=500` boundaries, so training can overshoot by up to one interval. Unpriced in the induction. |
| KW4.10 | MINOR | `1.206×` is a truncation of the true `1.2069` (rounds to `1.207`) — the error direction flatters the margin. Independently recomputed over the same 97 cells / 24 groups. |
| KW4.11 | MINOR | One 2-line diff hunk falls **outside** every section `§R2`'s "Where fixed" column claims: the KW3.7 `STATE.md:11-13→:24-26` fix at `:19-20` is in the pre-`§1` mandate preamble, attributed to "§1". |

---

## §1 E1 — **NOT-DISCHARGED**

### The rule as written (quoted exactly, `:551-570`)

> "before dispatching ANY cell-launch command — a first attempt on a
> not-yet-attempted cell, OR a retry of an `ABORTED-BUDGET` cell (E4,
> below) — the launcher (the same resume-safe supervisor-loop process
> that already gates on `status=="COMPLETED"` for skip-vs-resume, per
> the repo's on-box queue directive) computes `realized_gpu_h = Σ
> gpu_h` read fresh from EVERY cell JSON currently on disk under
> `results_kwall_characterization*/` **whose `status` is `COMPLETED`**
> … and applies, in order:
> 1. **HARD PROGRAM GATE.** If `realized_gpu_h + ceiling(cell) >
>    15.00`, the launch is refused outright … For a batch of cells
>    dispatched together …, the check is applied to the WHOLE batch
>    atomically — `realized_gpu_h + Σ ceiling(cell_i in batch) ≤
>    15.00` — never per-cell in isolation, so simultaneous launches
>    cannot jointly overshoot a bound only checked one cell at a time.
> 2. **RETRY GATE (subordinate to 1).** If `realized_gpu_h ≥ 12.00`,
>    no RETRY is dispatched…"

---

### KW4.1 — **FATAL.** Aborted spend is invisible to the gate on two independent grounds; the bound breaks worse than the defect E1 was built to fix.

**The design's own text on aborted spend:** there is none. I searched
the whole file. E1 prices only `status=="COMPLETED"` JSONs and the
induction paragraph prices only "EVAL overhead" as the unpriced
quantity:

> "the only quantity the admission check does NOT price is EVAL
> overhead" (`:578-579`)

That is false. Code evidence, `matrix-thinking/ncr/ncr_earlyln_scale.py`:

```
199:    return dict(status="ABORTED-BUDGET", step=step, elapsed_s=elapsed, ...)
...
262:    if tr["status"] != "COMPLETED":
263:        rec["status"] = tr["status"]
264:        rec["elapsed_s"] = time.time() - t0
265:        rn.atomic_write_json(out_path, rec)
266:        return rec
...
304:    rec["gpu_h"] = rec["elapsed_s"] / 3600.0 if device == "cuda" else 0.0
```

Line 304 — the **only** assignment of `gpu_h` in the file — is on the
COMPLETED path, *after* the `:262-266` early return. So an
`ABORTED-BUDGET` cell JSON:

1. has `status != "COMPLETED"` → **excluded by E1's status filter**, and
2. has **no `gpu_h` key at all** → `Σ gpu_h` cannot see it even if the
   filter were removed.

Third compounding defect: on retry, `run_earlyln_cell` re-runs and
overwrites the same `out_path` (`:240-243` skip only on `COMPLETED`,
so `os.path.exists` is True but the record is replaced). The first
attempt's `elapsed_s` is **destroyed**, so the aborted spend is not
merely unread — it is not durably recorded.

**Worst case, recomputed including aborted spend.** Ceilings actually
enforced are the CLI values `1.20`h (primary) / `2.32`h (conditional).

*Case A — pure abort.* All 12 primary cells abort, each is retried once
(E1's retry gate reads `realized_gpu_h = 0.00 < 12.00`, so every retry
is admitted), each aborts again → `PERSISTENTLY-ABORTED`:

```
12 cells × 2 attempts × 1.20 h = 28.80 GPU-h,  realized_gpu_h ≡ 0.00 at every gate check
```

The hard gate evaluates `0.00 + 1.20 = 1.20 ≤ 15.00` on all 24
dispatches. Neither gate ever fires.

*Case B — with the conditional arm.* K=26 completes 4/4 at nominal
(`4 × 0.5105 = 2.042`h, the only visible spend); K=28 and K=30's 8
cells each abort twice (`16 × 1.20 = 19.20`h invisible); the trigger
fires at `K_trig=26` and its 4 conditional cells each abort twice
(`8 × 2.32 = 18.56`h invisible). Gate checks: `2.042 + 1.20 ≤ 15.00` ✓
and `2.042 + 4×2.3121 = 11.29 ≤ 15.00` ✓ throughout.

```
realized_actual = 2.042 + 19.20 + 18.56 = 39.80 GPU-h
```

**39.80h vs. the claimed 15.2016h — 2.62×; and 2.65× the mandate's
15.00 cap.** This is strictly worse than KW3.1's 30.00h finding.
E1 replaced a count-capped bound that broke at 2× with a
"budget-capped" bound that breaks at 2.65×, while asserting it is
"bounded by construction (budget-capped, not count-capped)"
(`:594`) and "this time actually TRUE" (`:593`).

**Discharge condition.** (a) Define `realized_gpu_h` over **every**
attempt record regardless of status, using `elapsed_s/3600` (present on
both paths) rather than `gpu_h`; **and** (b) specify the build-stage
code change that makes aborted spend durable — either the runner sets
`gpu_h` on the abort path too, or attempts are appended to a ledger
rather than overwriting `out_path`. Prose alone cannot close this;
the overwrite at `:240-266` destroys the data the rule needs.

---

### KW4.2 — **FATAL.** The induction's premise is false; an abort-free counterexample reaches 20.92h.

**Quote (`:573-590`):**

> "By induction on admission order: the hard gate (1) never admits a
> cell or batch unless `realized_gpu_h`-so-far + that admission's own
> ceiling(s) stays `≤15.00` … So for the LAST batch admitted before the
> gate closes: `realized_before_last_batch + Σ ceiling(last batch) ≤
> 15.00`…"

`realized_before_last_batch` is defined by E1 as the sum over
**COMPLETED** JSONs at admission time. The induction silently treats it
as the true cumulative spend so far. Those are equal only if every
previously admitted cell has already finished and written. Nothing in
the design requires dispatch to be serialized behind completion — and
the repo's actual dispatcher does the opposite (KW4.3).

The batch-atomic clause is necessary but not sufficient. It closes the
*within-batch* hole. It says nothing about two dispatch events
separated in time but both before any completion; each passes
independently on a stale `realized_gpu_h`. So the design's own
sentence —

> "so simultaneous launches cannot jointly overshoot a bound only
> checked one cell at a time" (`:563-564`)

— is true only for launches inside one declared batch, and the design
never defines a batch boundary (§6's red-team item (iii) defers the
packing density to a later round, `:1082-1084`, which means the bound
depends on a number this design does not yet have).

**Abort-free counterexample, using the design's own trigger rule.**

The trigger (`:453-457`) reads:

> "Let `K_trig` = the smallest K in the ordered list `(26, 28, 30, 32)`
> whose primary 80K rate is NOT CONVERGED-ROBUST (`rate<3/4`),
> **evaluated only over K's with 4/4 `status=="COMPLETED"` primary
> cells**"

Under the natural reading of "evaluated only over K's with 4/4
COMPLETED" (restrict the evaluation to those K's), K=26 alone being
4/4 with `rate<3/4` fixes `K_trig=26` immediately — no rule requires
waiting for K=28/K=30. Then:

| step | gate reads | gate arithmetic | admitted? | true spend so far (at ceiling) |
|---|---|---|---|---|
| K=26 ×4 dispatched, complete at nominal | 0.000 | `0 + 4×1.0211 ≤ 15` | ✓ | 2.042 |
| K=28+K=30 ×8 dispatched (still training) | 2.042 | `2.042 + 8×1.151 ≈ 11.25 ≤ 15` | ✓ | 2.042 + 9.60 |
| conditional K=26 ×4 dispatched | **2.042** (8 in-flight cells invisible) | `2.042 + 4×2.3121 = 11.29 ≤ 15` | ✓ | 2.042 + 9.60 + 9.28 |

```
realized_actual = 2.042 + 8×1.20 + 4×2.32 = 20.92 GPU-h    (zero aborts, zero retries)
```

The design contains a *second*, conflicting reading of the same rule
one bullet later — "An `INCOMPLETE-AT-K` rung (§5) defers the trigger
decision until it resolves" (`:467-469`) — under which the conditional
arm waits for all three K's and the primary-only in-flight hole stays
at `12×1.20 + 0.15 = 14.55`h, inside the bound by luck rather than
derivation. **The design does not say which reading governs**, and one
of them breaks the bound by 5.72h with no adversarial behaviour at
all.

**Discharge condition.** Either (a) charge in-flight cells' ceilings
against the cap (a reservation ledger written at dispatch, released at
completion), or (b) require serialization behind completion, and in
either case fix the trigger-rule ambiguity by stating one reading.
Re-derive the induction with `realized_before_last_batch` replaced by
`reserved + realized`.

---

### KW4.4 — **MAJOR.** `ceiling(cell)` means two different numbers; the idealized bound is ≈16.34h, not 15.2016h.

**Quote (`:555-559`):**

> "no cell, first attempt or retry, is ever dispatched once its OWN
> training-phase ceiling (**E2's restored `max(2×nominal,1.0h)`
> value**) would push the projected total past the 15.00 hard cap"

But the value the runner actually enforces as `ceiling_s` is the CLI
argument, and §4's own command blocks pass **shared** values:
`--ceiling-gpuh 1.20` (all three primary K's) and `2.32`
(conditional) — with the design explaining precisely why they are
shared:

> "**1.20h is used as one shared value across the 3-K command** because
> it is `≥2×nominal` for EVERY K in the batch" (`:444-447`)

So the gate charges 1.0211 / 1.1073 / 1.1946 while the runner permits
1.20. The induction step "each admitted cell's actual TRAINING time is
bounded above by its ceiling" is true of the *enforced* 1.20, not of
the *charged* per-K value. Under-charge:

| | charged | enforced | Δ/cell | Δ × cells |
|---|---|---|---|---|
| K=26 | 1.0211 | 1.20 | 0.1789 | 0.7156 (×4) |
| K=28 | 1.1073 | 1.20 | 0.0927 | 0.3708 (×4) |
| K=30 | 1.1946 | 1.20 | 0.0054 | 0.0216 (×4) |
| conditional (K=30 worst) | 2.3121 | 2.32 | 0.0079 | 0.0316 (×4) |

If the last admitted batch is all 12 primary cells: charge 13.292h,
true max 14.400h → **1.108h** of unpriced spend. Adding the
conditional case, the idealized bound is

```
15.00 + 1.14 + 16×0.0126 = 16.34 GPU-h   (not the stated 15.2016)
```

**Discharge condition.** One word: define `ceiling(cell)` as the value
actually passed to `--ceiling-gpuh` (1.20 / 2.32) and re-derive the
bound, **or** pass per-K ceilings on the command line so charged =
enforced. Either fix is a one-line change, but as written the headline
number is not derived.

---

### KW4.3 — **FATAL.** E1 has no owner in the real dispatch path, and contradicts §6's own pool-eligibility claim.

**Quote (`:552-554`):**

> "the launcher (**the same resume-safe supervisor-loop process that
> already gates on `status=="COMPLETED"` for skip-vs-resume**, per the
> repo's on-box queue directive)"

No such process exists. The `status=="COMPLETED"` skip is inside the
**harness**, `run_earlyln_cell` at `ncr_earlyln_scale.py:240-245` — a
per-cell resume guard with no cross-cell state and no knowledge of any
other cell. The design has attributed a launcher-level capability to a
function-level check.

The repo's actual dispatchers, both read this round:

- `matrix-thinking/queue/queue_worker.sh` — N independent per-GPU
  workers, each atomically `mv`-claiming one spec at a time from
  `pending/` in filename order (`:117-136`). `grep` for
  `gpu_h|ceiling|budget|cumul` over the file returns **nothing**
  budget-related. There is no shared state between workers and no
  "batch" for E1's batch-atomic check to apply to.
- `matrix-thinking/queue/idle_fallback_daemon.sh` — promotes `WAVE=8`
  specs at a time from `fallback_pool/` in filename order (`:7`, `:33`).
  Same `grep` returns exactly one line — the contract text itself.

`grep -rl 'realized_gpu_h'` over all `.sh`/`.py` in the repo: zero
hits. E1 is prose with no implementation site named.

**The structural contradiction.** §6 quotes and then claims conformance
to the pool contract (`idle_fallback_daemon.sh:10-16`):

> "the pool holds ONLY flat specs — each fully audited +
> queue-eligible, independently runnable in any order, carrying its own
> cost ceiling, **with NO intra-wave dependencies**, stage gates, or
> staged-escalation semantics (filename-order promotion cannot honor
> them)."

and then asserts:

> "the PRIMARY 12-cell grid is the flat, pool-eligible spec (unchanged
> from R0)" (`:1038-1039`)
>
> "the PROGRAM-level bound is now E1's cumulative-realized-GPU-h check
> … — this is what makes ≤15.20h GPU-h … an actual bound rather than an
> aspirational sum" (`:1052-1057`)

A cumulative program cap **is** an intra-wave dependency: cell N's
admission depends on cells 1..N−1's realized spend. The two bullets in
the same §6 list are mutually exclusive. Either the grid is flat and
pool-eligible (in which case E1 is never executed and the bound is the
per-cell ceiling sum, `12×1.20 + 4×2.32 = 23.68`h, or 47.36h with
retries), or E1 governs and the grid must leave the pool for a chained
one-shot launcher — the `idle_launch_jacobian.sh` pattern the contract
itself names for work that needs sequencing.

**Discharge condition.** Pick one, in the living body:
(a) **Drop E1**, bound by per-cell ceilings, disclose the honest
number (23.68h nominal / 47.36h with the E4 retry), and re-check the
ceremony tier; or
(b) **Keep E1**, withdraw the pool-eligibility claim in §6, and name
the chained one-shot launcher as the dispatcher — specifying the
reservation ledger KW4.1/KW4.2 require. §6's red-team item (i)
("verify E1's cumulative-cap check … against whatever launcher script
the build stage actually produces", `:1077-1080`) is not a substitute:
it defers verification of a mechanism whose *existence* is the open
question.

---

### KW4.9 — MINOR. "bounded above by its ceiling" is not strictly true.

`ncr_earlyln_scale.py:191` wraps the ceiling test inside the logging
block:

```
191:        if step % log_every == 0 or step == 1:
...
198:            if elapsed > ceiling_s:
199:                return dict(status="ABORTED-BUDGET", ...)
```

with `log_every: int = 500` (`:170`). Training can therefore exceed
`ceiling_s` by up to one 500-step interval (≈11s at archive throughput
— `0.0031`h — and proportionally more under exactly the contention the
ceiling exists to survive). Immaterial in magnitude; the induction
nonetheless claims a strict bound. Discharge: add the term or soften
the claim.

---

## §2 E2 — **DISCHARGED**

Every sub-check the round mandate names passes, recomputed this round
by direct execution.

**Restored ceilings vs. `≥2×nominal` and the `1.0h` floor:**

| K | 80K nominal | 2×nominal | `max(2×nom,1.0)` | design | ≥2× ✓ | ≥1.0h ✓ |
|---|---|---|---|---|---|---|
| 26 | 0.51053 | 1.02106 | 1.02106 | 1.0211 | ✓ | ✓ |
| 28 | 0.55362 | 1.10723 | 1.10723 | 1.1073 | ✓ | ✓ |
| 30 | 0.59725 | 1.19450 | 1.19450 | 1.1946 | ✓ | ✓ |
| 26 (160K) | 0.98812 | 1.97624 | 1.97624 | 1.9764 | ✓ | ✓ |
| 28 (160K) | 1.07153 | 2.14306 | 2.14306 | 2.1432 | ✓ | ✓ |
| 30 (160K) | 1.15608 | 2.31215 | 2.31215 | 2.3121 | ✓ | ✓ |

Inputs re-derived, not taken from prose: `F(K,d,64)` executed directly
→ 8,636,672 / 9,421,568 / 10,216,704 / 11,022,080 / 11,837,696
(matches §4's table to the digit); K=24 measured mean
`gpu_h = 0.4680` from
`experiment-runs/2026-07-12_ncr_nextlever_wave/dratio/earlyln_K24_s{0-3}.json`
(0.4977/0.4717/0.4421/0.4606, all `status=COMPLETED`, `d=25`,
`step=80000`).

**Shared CLI values dominate every per-K ceiling they must:**
`1.20 ≥ max(1.0211, 1.1073, 1.1946) = 1.1946` ✓ and
`2.32 ≥ max(1.9764, 2.1432, 2.3121) = 2.3121` ✓. Both correct, with
the smallest margin (K=30 conditional, 0.0079h) still positive.

**KW3.11's conservative-ratio switch verified.** The three archive 2×/1×
ratios recomputed from raws: K16 `0.82490/0.42620 = 1.9354`; K24/d48
`0.93655/0.50406 = 1.8582`; K32 `1.05105/0.56880 = 1.8478`. The design
applies the **maximum** (1.9355, K16) as claimed — the conservative
choice, correctly justified since the figure now feeds a safety bound.

**The false claim is gone.** `grep '1.06'` returns three hits, all
correct: `:631-633` quotes the sentence *as deleted* ("Rev 1's
supporting sentence … is **deleted — it was false**"), and the other two
are the `§A2`/`§R2` finding rows. No live assertion of "within 1.06×"
survives.

**The 1.206× citation is now used** (`:636-637`) and is
independently correct — see KW4.10 for the one-digit quibble.

*(KW4.4's charged-vs-enforced mismatch is filed under E1, not E2: E2's
own instruction — restore the job-108 convention — was executed
correctly. It is E1's induction that consumes the wrong number.)*

---

### KW4.10 — MINOR. `1.206×` should be `1.207×`.

Recomputed over the identical scope the design cites ("all 97 completed
cells / 24 groups"), grouping by (archive dir, K, d, steps) and taking
`max(gpu_h)/mean(gpu_h)`: **24 groups, 97 completed cells** — both
figures reproduce exactly. The maximum ratio is
`1.2685/1.05105 = 1.2069` (K=32, d=33, 160K, seed 3), which rounds to
**1.207**, not 1.206. Headroom `2.00/1.2069 = 1.657×` (design: "≈1.66×",
fine). The truncation slightly *flatters* the safety margin. Runners-up
for the record: 1.1078 (K24/d48, 160K), 1.0918 (K14), 1.0862 (K16, 320K).

---

## §3 E3 — **DISCHARGED**

**The K=32 160K rate, recomputed from raws this round.** Loaded all 4
JSONs in `experiment-runs/2026-07-12_ncr_k32_budget/budget2x_*`,
computing `indist_min = min(recovered_frac@0.9)` over
`eval.points[h∈{1,2,3}].reads.binexp` restricted to
`component=="train_support"` (the runner's own `_cell_gate1` predicate,
`:317-329`) and `aer_mean = mean(deep_probe.A_eff_rank)`:

| seed | status | step | d | `indist_min` | `AER/K` | `≥0.9 AND ≥0.9K`? |
|---|---|---|---|---|---|---|
| 0 | COMPLETED | 160000 | 33 | 0.7944 | 0.9536 | ✗ |
| 1 | COMPLETED | 160000 | 33 | **0.9015** | 0.9748 | **✓** |
| 2 | COMPLETED | 160000 | 33 | 0.5818 | 0.9424 | ✗ |
| 3 | COMPLETED | 160000 | 33 | 0.8865 | 0.9689 | ✗ |

**Rate = 1/4, only seed 1 clears at `indist_min=0.9015`** — matching the
design's `:972-973` claim to the digit.

**Which band the design's rule assigns it.** §5's three conditional
bands are `≤1/4 → CONFIRMED-WALL-AT-160K`, `exactly 2/4 →
PARTIAL-IMPROVEMENT-AT-160K`, `≥3/4 → SLOW-CONVERGENCE-AT-160K` — a
total partition of `{0,1,2,3,4}/4` with no gap and no overlap.
`1/4 ≤ 1/4` → **`CONFIRMED-WALL-AT-160K`**, exactly as the design
states at `:973`. KW3.3's inverted label is genuinely fixed.

**No reachable branch outputs a 320K-dependent label.** All three band
names end in `-AT-160K`. `grep '/320K'` returns exactly three hits, none
of them a live band: `:966` (the E3 fix sentence declaring the suffix
deleted), `:1401` (the `§R2` finding row), `:1428` (the
numbers-that-moved list). The 320K datum appears only under an explicit
"Disclosed separately, as archive context ONLY, never as a band
determinant" heading (`:975-982`).

**Matched-budget is matched on the recipe too** (checked beyond the
mandate, since "matched budget" is worthless if the config differs).
All four archived K=32 160K cells carry `K=32, d=33, d_override=33,
h=64, anneal_frac=0.5, runner_tag=ncr_earlyln_scale_v1, step=160000` —
identical in every axis to what the paid arm's command
(`--K K_trig --d-override K_trig+1 --steps 160000`, defaults elsewhere)
would produce. The `$0` branch is genuinely on the same footing.

**K_trig=32 is reachable**, so the branch is not vacuous: K=32's
archive rate is fixed at 0/4 < 3/4, so if all of {26,28,30} come back
ROBUST, rule (1) fires FRONTIER-AT-K*=30 and the trigger lands on 32.

Fully discharged.

---

## §4 E4 — **PARTIAL**

The **band side** of E4 is delivered exactly as the disposition
specified: retry bounded at 1, `PERSISTENTLY-ABORTED` terminal,
denominator fixed at 4 (A4.9 guard intact), interval logic over both
candidate values, `harvest()`'s `n_completed`-from-`status` patch named
against the real defect (`discover_seeds_by_K`'s glob at `:351-371` →
`gate_eligible = n_seeds >= 4` at `:403` — verified by direct code read,
the design's diagnosis is correct). The **trigger side** is not.

I implemented the rule verbatim and swept it.

### (a) One K incomplete, `r_known = 2` — the ROBUST boundary

| incomplete K | configurations where the two candidates give DIFFERENT bands |
|---|---|
| K=26 | 16/25 = **64%** |
| K=28 | 17/25 = **68%** |
| K=30 | 25/25 = **100%** |

Bands differ overwhelmingly, so `INCOMPLETE-AT-K` is the modal
result — expected, since `r_known=2 → {2,3}` straddles `ROBUST(r):=r≥3`.
Examples: `(2,0,0)→GRADUAL-DECAY` vs `(3,0,0)→FRONTIER-AT-K*=26`;
`(0,2,0)→FRONTIER-AT-K*=24` vs `(0,3,0)→FRONTIER-AT-K*=28 [NON-MONOTONE]`.
At K=30 the rule *never* decides. See KW4.7.

### (b) One K incomplete, `r_known = 3`

Differs in only 1/25 (K=26), 1/25 (K=28), 0/25 (K=30) — the rule
decides ≈96–100% of the time. The residual cases come from rule 5's use
of raw values: `(3,4,2)→NON-MONOTONE-UNRESOLVED` vs
`(4,4,2)→GRADUAL-DECAY`. Working as intended.

### (c) Two K's each with one incomplete cell (2²=4 candidates)

| pair | decides | INCOMPLETE-AT-K |
|---|---|---|
| (26,28) | 43/80 = 54% | 37/80 |
| (26,30) | 38/80 = 48% | 42/80 |
| (28,30) | 36/80 = 45% | 44/80 |

The cross-product composes correctly and terminates. No logic defect.

### (d) Incomplete cell at the K that determines the trigger — **DEADLOCK + non-composition**

### KW4.5 — **MAJOR.**

**Quote (§4, `:467-469`) — UNCHANGED from Rev 1** (confirmed: the §4
diff hunks start at new-side lines 428/432/437+15/475/480+6/…, leaving
new `:452-474` untouched):

> "Precondition (D5): `K_trig` is read only from a K with 4/4 COMPLETED
> cells. An `INCOMPLETE-AT-K` rung (§5) defers the trigger decision
> **until it resolves**."

E4 makes `PERSISTENTLY-ABORTED` **terminal**:

> "that seed becomes **`PERSISTENTLY-ABORTED` — a TERMINAL state, never
> retried again, regardless of remaining budget.**" (`:675-677`)

A K with one terminal-aborted cell can therefore **never** reach 4/4
COMPLETED. "Defers until it resolves" never terminates. **This is
verbatim the deadlock KW3.4 identified** — closed on the band side by
interval logic, left standing on the trigger side because §4's
precondition bullet was never revised to match. §R2's KW3.4 row claims
"no deadlock" (`:1402`); that is true of the classification and false
of the trigger.

**Second, sharper defect: interval logic does not compose with the
trigger even when it succeeds.** I enumerated every single-incomplete
configuration and compared `{bands}` against `{K_trig}`. **11
configurations yield ONE band across both candidates but TWO different
`K_trig` values.** Examples:

| incomplete K | `r_known` | candidates | band (both) | `K_trig` |
|---|---|---|---|---|
| 26 | 2 | (2,2,0) / (3,2,0) | GRADUAL-DECAY | **{26, 28}** |
| 26 | 2 | (2,2,2) / (3,2,2) | GRADUAL-DECAY | **{26, 28}** |
| 26 | 2 | (2,0,3) / (3,0,3) | FRONTIER-AT-K\*=30 [NON-MONOTONE] | **{26, 28}** |

In these cases E4's rule says **DECIDE** — and hands the trigger an
input it cannot resolve. `K_trig=26` vs `K_trig=28` is the difference
between two different 4-cell 160K arms (≈3.95h vs ≈4.29h nominal, up to
9.25h at ceiling) at two different K's. No rule in the design places it.

And in the flat case, interval logic and the trigger simply disagree:
K=26 with `r_known=2`, K=28=4, K=30=4 → band `FRONTIER-AT-K*=30
[NON-MONOTONE]` (r26=2) vs `FRONTIER-AT-K*=30` (r26=3) — different
bands, so `INCOMPLETE-AT-K` — while `K_trig` would be 26 vs 32, i.e.
"launch 4 paid cells" vs "$0 archive reuse."

**Discharge condition.** Extend the interval rule to the trigger
explicitly: evaluate `K_trig` over the *same* candidate set; if all
candidates agree, launch at that `K_trig` with the interval-resolution
disclosed; if they disagree, **no conditional arm is launched** and the
disagreement is reported. And replace "defers … until it resolves" with
a rule that terminates against a TERMINAL state.

---

### KW4.7 — MINOR. The decide-rate is material and undisclosed.

The design presents interval logic as the fix that removes the
deadlock, but never states how often it actually decides. At the
boundary that matters — `r_known=2`, precisely the case a sub-ROBUST
rung produces — it fails to decide in 64%/68%/**100%** of surrounding
configurations. A single terminal abort at K=30 with `r_known=2`
**guarantees** `INCOMPLETE-AT-K`, i.e. an unclassifiable study, since
the design has no further recourse (the retry is exhausted, the state
is terminal). Two singly-incomplete K's decide only 45–54% of the time.

Not a logic error — the rule behaves exactly as written — but a
reliability property a reader of §4/§5 would reasonably assume is
better than it is. Discharge: one disclosure sentence with these
figures, or a policy that spends a third attempt at the ROBUST boundary
specifically (budget permitting).

---

### KW4.8 — MINOR. "INCOMPLETE-AT-K **for that K**" is a category error.

**Quote (`:699-702`, `:713-715`):**

> "**Different bands ⇒ `INCOMPLETE-AT-K` for that K.**"
> "otherwise `INCOMPLETE-AT-K` for the affected K's, both/all candidate
> bands disclosed."

The six-rule procedure is a function of the whole triple
`(r26,r28,r30)` returning **one** label. There is no per-K band, so
"`INCOMPLETE-AT-K` for that K" names an object the procedure cannot
produce. What the *study* reports in that case is left implicit.

Relatedly, §5's totality claim — "a **total, ordered decision
procedure** … no case unhandled" (`:836-837`) and "Σ=125/125" — is over
complete triples only; the E4 states sit outside it. The partition is
still exhaustive over its own domain (verified, §5 below), but the
design's "every outcome is classified exactly once" framing does not
cover the outcomes E4 introduces.

Discharge: state that `INCOMPLETE-AT-K` is a **study-level** verdict
orthogonal to the 125-outcome partition, carrying the affected K(s) as
a disclosure field.

---

## §5 E5 — **DISCHARGED**

I implemented `classify(r26,r28,r30)` from the design's §5 text alone —
the six ordered rules at `:839-844` plus the ROBUST-sequence
monotonicity tag at `:846-851`, with `r24=4`, `r32=0`,
`ROBUST(r):=r≥3` — and ran it over all 5³=125 outcomes without looking
at the printed table first. **My execution reproduces the design's
regenerated table row-for-row:**

| Band | design | my execution | match |
|---|---|---|---|
| FRONTIER-AT-K\*=24 | 18 | 18 | ✓ |
| FRONTIER-AT-K\*=24 [NON-MONOTONE] | 4 | 4 | ✓ |
| FRONTIER-AT-K\*=26 | 12 | 12 | ✓ |
| FRONTIER-AT-K\*=28 | 8 | 8 | ✓ |
| FRONTIER-AT-K\*=28 [NON-MONOTONE] | 12 | 12 | ✓ |
| FRONTIER-AT-K\*=30 | 8 | 8 | ✓ |
| FRONTIER-AT-K\*=30 [NON-MONOTONE] | 42 | 42 | ✓ |
| GRADUAL-DECAY | 15 | 15 | ✓ |
| NON-MONOTONE-UNRESOLVED | 4 | 4 | ✓ |
| NON-MONOTONE-UNRESOLVED [NON-MONOTONE] | 2 | 2 | ✓ |
| **Total / rows** | **125 / 10** | **125 / 10** | ✓ |

**Member lists match too**, not just counts: `NON-MONOTONE-UNRESOLVED`
= `{(2,0,1),(2,0,2),(2,1,2),(3,4,2)}` and its `[NON-MONOTONE]` sibling
= `{(2,3,2),(2,4,2)}` — exactly as `:900-902` states.

**All 10 representative rows reproduce**, including every case a prior
round named:

`(0,0,0)`→K\*=24 ✓ · `(0,0,1)`→K\*=24 ✓ · `(4,3,0)`→K\*=28 ✓ ·
`(0,4,0)`→K\*=28 [NM] ✓ · `(4,4,2)`→GRADUAL-DECAY ✓ ·
`(2,2,2)`→GRADUAL-DECAY ✓ · `(4,0,4)`→K\*=30 [NM] ✓ ·
`(4,4,4)`→K\*=30 ✓ · **`(2,4,2)`→NON-MONOTONE-UNRESOLVED
[NON-MONOTONE]** ✓ (KW3.5's mislabeled row, now correct) ·
`(3,2,1)`→GRADUAL-DECAY ✓.

**And the rule text really is untouched**, as `§R2` claims: `:839-853`
is byte-identical to `HEAD~1`'s `:622-636`
(md5 `492da18360416b38d82704c0af0b27bc` both sides). E5 was discharged
by re-executing the rules, not by hand-editing counts, exactly as the
disposition required. Nothing left open.

*Observation, not a finding (rule text is out of scope for this focused
round and rounds 1–2 settled it):* the rule set can declare
`FRONTIER-AT-K*=24` while K=28 is ROBUST — e.g. `(0,3,2)` → rules 1–3
all fail on their `≤1` conjunct, rule 4 fires on `r26=0≤1`, tag applied.
The `[NON-MONOTONE]` tag does disclose it, which is what the tag is
for. Flagging only so a future non-focused round does not mistake it
for new.

---

## §6 E6 — **PARTIAL**

**Coverage: complete.** The `§R2` table (`:1397-1419`) carries **21
rows**: KW3.1–KW3.16 (all 16), plus the four round-1 PARTIALs
(KW1.3, KW2.2, KW2.3, KW2.8) and the declined KW2.9. Cross-checked
against `NCR_KWALL_ATTACK_R2.md`'s own §0 summary and §1 discharge
table — every finding round 2 raised or left open has a row. No silent
leftovers. E6's literal instruction is met.

**But two problems.**

*(1) Several rows overstate.* KW3.1 and KW3.4 are both marked
"**DISCHARGED.**" This round finds KW3.1's replacement mechanism broken
in three independent ways (KW4.1/4.2/4.3, all FATAL) and KW3.4's
deadlock still live in the trigger path (KW4.5). A close-out table
whose verdicts do not survive the next round is exactly what E6 said
"the next audit can attack."

*(2) One of the two ACCEPTED-RISKs is refuted.* Adjudicated
individually below, per the round mandate.

---

### KW2.8 / KW3.13 — smoke-extension deferral: **REFUTED**

### KW4.6 — **MAJOR.**

**Quote (`:765-776`):**

> "R1's discharge asked for two things: a `d=K+1` micro-smoke
> instruction (**recorded above, in the `validity_check`/job-spec
> instructions**) AND an extension of `t4b`'s own K-list — the second
> half was silently dropped in Rev 1. **Accepted as risk, one
> sentence:** … the build stage **must still run the `d=K+1`
> micro-smoke this design specifies** before release, which is the
> smoke test that actually matters for this design's own config
> family."

The accepted-risk is conditioned twice on a micro-smoke that the
living body does not contain.

The job-spec paragraph it points at (`:745-759`) specifies a job-108
8-field template and a `validity_check` asserting
`d == K+1 and d_override == K+1`. A `validity_check` is a **harvest-time
assertion on a COMPLETED production cell** — it runs after a cell has
consumed its full budget. It is not a smoke test in any sense, and it
cannot catch the failure a smoke test catches (a config that crashes or
mis-shapes on the first forward/backward pass).

`grep -n 'micro-smoke\|t10\|t4b\|--smoke'` over the whole file returns
the actual instruction at exactly one place — `:1276`, inside `§R1`'s
table:

> "flagged here as a build-stage instruction: add a `d=K+1` micro-cell
> smoke (the `t10` pattern) at one new K before build-release"

— i.e. inside the section Rev 2 itself declares frozen and
superseded ("`§R1` is historical and frozen by house convention",
`:1412`). The design therefore asserts the instruction is "recorded
above" (it is not) and that "this design specifies" it (only its
frozen revision log does), while the one operative smoke gate for its
own `d=K+1` config family has no owner in any live section.

This matters against CLAUDE.md's hard rule — *"Smoke test every model
(forward pass, backward pass, gradient check) before training"* — and
against this program's own recorded lesson that CPU-stub/default-path
self-tests are not coverage of the production path.

The *harmlessness* half of the justification I **accept**: three extra
`--smoke` cells at `GRID_SHAPES`' default `d=2K` are cheap and are
coverage of an already-rejected convention, not a claim about
`d=K+1`. (Minor unpriced consequence: smoke spend writes outside
`results_kwall_characterization*/` and so falls outside E1's cap
entirely — negligible in magnitude, noted only because E1 claims to
bound "the WHOLE program.") It is the *deferral target* that does not
exist.

**Discharge condition.** State the `d=K+1` micro-smoke as an explicit
build-release gate in §4 or §6 — one new K, ~500 steps, the `t10`
pattern, run on real CUDA before any production cell — with a pass
criterion. Do not discharge it by reference to `§R1`.

---

### KW3.14 / KW3.15 — frozen-historical cosmetics: **ACCEPTED**

Both are accepted-risk on the same ground:

> "`§R1` is historical and frozen by house convention — its
> mischaracterization is not rewritten. Instead, KW2.9 itself is now
> DISCHARGED for real this revision … One-sentence justification:
> fixing the underlying gap supersedes editing a frozen historical
> claim about whether the gap existed." (`:1412`)

I accept both, on four independently checked grounds:

1. **The convention is real and self-consistent.** `§R1:1250-1255` and
   `§R2:1381-1385` both state it and both scope it identically:
   historical sections are unchanged "EXCEPT where a disposition
   explicitly required rewriting." E6 required a close-out **row**, not
   an edit to `§R1`. The convention is applied, not invented to dodge.
2. **The underlying gap is genuinely closed in the living body.**
   KW2.9's missing sentence now exists at `:1001-1009` — a
   `SUB4-DISCLOSED-ONLY(n=0)` reader-warning paragraph in §5. That is
   what round 1 actually asked for. Verified by direct read.
3. **Both are confirmed cosmetic by the attacking round itself.**
   `NCR_KWALL_ATTACK_R2.md:59` on KW3.15: "(Its substance is reconciled
   through `EXPERIMENT_LOG.md:8495-8496` instead, so the conclusion
   stands.)" I re-read that citation: it records the K=16 1×/2×/4×
   Gate-1 progression 1/4→3/4→4/4, which is the substance §R1's KW1.2
   row claimed §11.4 carried. Conclusion unaffected.
4. **Neither false statement is load-bearing anywhere downstream.**
   Both live in a revision-log table describing what a *previous* round
   said, with the correction visible one section below in `§R2`.

Editing a frozen revision log to correct a claim about a prior round's
wording — while the substantive gap it concerns is fixed in the live
text — would trade a documented, visible correction for an invisible
retcon. Accepting is the right call. **No finding.**

---

## §7 INTEGRITY — **PASS**

**Header:** `:3-4` reads
`**STATUS: DRAFT-R2 — POST-AUDIT-2, AWAITING FOCUSED AUDIT ROUND 3 (not
build-released, not queue-eligible).**` — matches the expected string.
No mismatch.

**Frozen sections byte-identical.**
`git show HEAD~1:…| sed -n '857,1033p' | md5` =
`sed -n '1195,1371p' <new> | md5` = `9d1c064229e9295d525824db07b85240`
(177 lines each). `§A1-ADJUDICATION`, `§R1`, and `§A2-ADJUDICATION` are
unmodified.

**Hunk-to-section map.** `git diff HEAD~1 HEAD -U0` gives 30 hunks,
`568 insertions / 123 deletions`. Mapped against the new file's section
boundaries (§1:28, §2:65, §3:199, §4:421, §5:780, §6:1021, §7:1107,
§A1:1195, §R1:1247, §A2:1327, §R2:1375):

| region | hunks (new-side starts) | claimed by §R2? |
|---|---|---|
| title/status | 3 | ✓ (status header) |
| **pre-§1 mandate** | **19** | ✗ — see KW4.11 |
| §2 | 79 | ✓ (§2(a), KW3.7) |
| §3 | 319, 403 | ✓ (KW3.8, KW3.10) |
| §4 | 428, 432, 437, 475, 480, 518, 531, 588, 592, 653, 761 | ✓ |
| §5 | 805, 867, 890, 894, 908, 920, 926, 963 | ✓ |
| §6 | 1045, 1061, 1070, 1096 | ✓ |
| §7 | 1153 | ✓ |
| §R2 (new) | 1372 | ✓ |

No hunk touches `§A1`/`§R1`/`§A2`. The last hunk (`@@ -1033,0
+1372,107 @@`) is a pure append of `§R2` past the old EOF.

### KW4.11 — MINOR. One hunk lands outside every claimed section.

```
@@ -19 +19,2 @@
-the archive actually leaves open."* `STATE.md:11-13` records this
+the archive actually leaves open."* `STATE.md:24-26` (line renumbered
+this revision, content unchanged — KW3.7) records this
```

This is the KW3.7 line-number fix, whose "Where fixed" column reads
"§1, §2(a), §6 (two citations)" (`:1405`). Line 19 is in the **mandate
preamble**, above `## §1` at line 28. Same fix, same class, correctly
applied — the section attribution is just off by one block. Trivial;
recorded because the round mandate asks that every change fall in a
claimed section.

**All three STATE.md renumbers verified against the live file:**
`:24-26` = "…successor: K∈{26,28,30} recovery-leg wall characterization
on live K=24 — draft agent in flight
(`NCR_KWALL_CHARACTERIZATION_DESIGN.md`, DRAFT-R0 → audit →" ✓;
`:53` = "novelty gate re-entry before any build; NO NCR job
queue-eligible." ✓; `:128` = "finding: **free-write K=24 / d=25
recovers 1.0 at ALL far depths, cond" ✓.

---

## §8 WHAT I COULD NOT BREAK — VERIFIED CLEAN THIS ROUND

Everything below was recomputed from raw artifacts this round and
reproduced the design's stated values. This is the third consecutive
round the scientific core has survived; the defects are confined to
the E1 ops layer and the E4 trigger.

- **§3's K=32 budget table, all 12 cells.** `indist_min` 0.4643/0.5170/
  0.6875/0.8711 (1×), 0.7944/0.9015/0.5818/0.8865 (2×), 0.8754/0.9124/
  0.9118/0.8965 (4×) — exact to 4 decimals. Conjunctive-gate rates
  **0/4 → 1/4 → 2/4** confirmed. 12/12 `status=COMPLETED`, `d==33`,
  `train.step` matching the requested budget exactly.
- **AER/K at K=32/1× = 0.9269–0.9679** (KW3.10's correction) — exact:
  0.9269/0.9287/0.9440/0.9679.
- **The n=12 K=24 table, all 8 seed-extension cells.** `indist_min=1.000`
  in 8/8; `AER/K` = 0.9997/0.9999/0.9999/0.9999/0.9995/1.0000/0.9996/
  0.9997; `gpu_h` = 0.519/0.525/0.491/0.489/0.520/0.486/0.503/0.503 —
  every value matches the printed table.
- **FLOP formula and ratios.** `F(K,d,64)` executed: 8,636,672 /
  9,421,568 / 10,216,704 / 11,022,080 / 11,837,696; spread
  **1.091/1.183/1.276** — KW2.4's "1.09×–1.28×" correction confirmed.
- **K=24 nominal 0.4680** from four raw `gpu_h` values.
- **All three 2×/1× archive ratios** (K16 1.9354, K24/d48 1.8582,
  K32 1.8478) and the design's use of the **maximum** — KW3.11's
  conservative switch verified.
- **Eval overhead, KW3.9.** In the design's exact stated scope
  (K∈{16,24,32}, 80K/160K): **n=64**, percentage range **0.35%–1.58%**,
  max absolute **45.5s = 0.01264 GPU-h** at K=32/2×/seed3
  (train 4521.0s, total 4566.5s). Every figure reproduces, including
  the n. Archive-wide n=97 also confirmed.
- **1.206× / 97 cells / 24 groups** — group structure and cell count
  reproduce exactly (one-digit rounding, KW4.10).
- **E5's 125-outcome table** — row-for-row and member-for-member (§5).
- **The `harvest()` file-glob-vs-status defect E4 diagnoses is real.**
  `discover_seeds_by_K` (`:351-371`) globs filenames;
  `gate_eligible = n_seeds >= 4` (`:403`) therefore reads True for a K
  containing an `ABORTED-BUDGET` cell. The design's location and
  proposed `n_completed` patch are both correct.
- **Every code line-citation in the design is accurate**, spot-checked:
  `:198-201` (ABORTED-BUDGET return, actual 199), `:243-245`
  (COMPLETED skip, actual 243), `:303-304` (`gpu_h`, actual 304),
  `:317-329` (`_cell_gate1`), `:380-406` (`harvest`).
- **§2(c)'s mod-K crash arithmetic** re-spot-checked: `29%26=3` ∈
  forbidden `{0,1,2,3}`, `29%28=1`, `61%30=1` — all three new K's would
  crash `ncr_ortho_write.py`'s assert. Unchanged and still correct.

---

## §9 VERDICT

**REV-REQUIRED.** Forcing findings: **KW4.1, KW4.2, KW4.3** (FATAL),
**KW4.4, KW4.5, KW4.6** (MAJOR).

Not BLOCKED: the science is verified clean for the third round running,
the partition is provably exact, E2/E3/E5 are genuinely discharged, and
every FATAL has a concrete, bounded fix. But the design cannot go to
build in this state — E1 is the mechanism its entire cost claim rests
on, and it is (i) blind to the spend it most needs to see, (ii) derived
from a false premise, and (iii) unimplementable in the dispatch path
this design routes itself into.

**Recommended shape for Rev 3** (offered, not binding):

1. **Decide the E1 architecture question first** (KW4.3), because
   KW4.1/KW4.2/KW4.4 are downstream of it.
   - *Option A — drop E1.* Bound by per-cell ceilings alone; the honest
     unconditional worst case is `12×1.20 + 4×2.32 = 23.68`h, or
     **47.36h** with E4's one retry per cell. Both sit inside CLAUDE.md's
     10–50 GPU-h tier, so the ceremony tier is unchanged and the
     pool-eligibility claim survives intact. Costs only a bigger, honest
     number.
   - *Option B — keep E1.* Withdraw §6's pool-eligibility claim, name a
     chained one-shot launcher (`idle_launch_jacobian.sh` pattern), and
     specify a **reservation ledger**: charge each cell's ceiling at
     dispatch, release on completion, and price aborted attempts from
     `elapsed_s`. Then re-derive the induction over
     `reserved + realized`.
   Option A is cheaper, more honest, and keeps the flat-spec property
   the pool contract and A4.12 both prize. Option B buys a tighter
   number at the cost of real launcher engineering and its own build
   audit.
2. **Fix the trigger** (KW4.5): extend interval logic to `K_trig`
   (agree ⇒ launch; disagree ⇒ no conditional arm, disclosed), and
   replace "defers until it resolves" with a terminating rule. Pick one
   reading of "evaluated only over K's with 4/4 COMPLETED" and say so
   (KW4.2 depends on which).
3. **Relocate the `d=K+1` micro-smoke** into the living body as a
   build-release gate with a pass criterion (KW4.6).
4. **Disclose the interval-logic decide-rate** and clarify
   `INCOMPLETE-AT-K`'s study-level status (KW4.7, KW4.8) — two
   sentences.
5. If Option A is chosen, KW4.4/KW4.9 dissolve; if Option B, fix
   `ceiling(cell)` to the CLI value and add the `log_every` overshoot
   term.

Rounds 1–2 remain settled for untouched material. A Rev 3 round should
be focused on the E1 architecture decision and the trigger rule only —
the partition (E5) and the pricing inputs (E2/E3) are now verified by
two independent executions each and should not be re-litigated.

---

*Focused audit round 3, 2026-08-06. Written from direct reads of
`matrix-thinking/NCR_KWALL_CHARACTERIZATION_DESIGN.md` (1479 lines,
full), `NCR_KWALL_ATTACK_R2.md`, `matrix-thinking/ncr/ncr_earlyln_scale.py`
(lines 165-266, 290-410), `matrix-thinking/queue/idle_fallback_daemon.sh`,
`matrix-thinking/queue/queue_worker.sh`, `STATE.md`, and fresh
`json.load` recomputation of every load-bearing number from
`experiment-runs/2026-07-12_ncr_k32_budget/` (8 cells),
`experiment-runs/2026-07-12_ncr_mappinglaw_wave1/` (dratio_K32_d33 +
q2_K24_seedext, 12 cells), `experiment-runs/2026-07-12_ncr_nextlever_wave/`,
`experiment-runs/2026-07-11_ncr_earlyln_scale/`,
`experiment-runs/2026-07-12_ncr_earlyln_budget2x/`, and an archive-wide
scan of all 97 completed `earlyln_K*` cells. The six-rule partition and
the E4 interval-logic rule were both implemented from the design's text
and executed independently before comparison. `git diff HEAD~1 HEAD`
and md5 comparison used for the integrity check. This file is the only
repo file created or modified; no command was run on the box; no job was
launched; no git mutation was made.*
