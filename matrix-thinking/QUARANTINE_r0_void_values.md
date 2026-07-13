# QUARANTINE — R0 VOID recall-gap outcome values

## ⚠ DO NOT READ unless explicitly authorized, AND you are not the agent
## tasked with pinning `PARAM_AXIS_SCALING_DESIGN.md` §9.1 blind.

Reading any value on this page before pinning §9.1 (the recall-gap
normalization) contaminates you for that decision — exactly as it
contaminated the three agents/coordinator turns before this quarantine
landed (`PARAM_AXIS_SCALING_DESIGN.md` §9.7's ledger; §9.1's handoff
protocol). If you are not that pinner and have no other explicit
authorization to be here, stop reading now and go back to the redacted
original locations, which carry everything a builder needs stated
qualitatively.

**Status of the underlying instrument: VOID / RETRACTED.**
`lm_recall_gap_probe_rd.py` (FIX-B) and `param_axis_r0_driver.py` have a
load-bearing FATAL bug (shared-tensor mass-ablation: a large fraction of
context tokens corrupted per forward pass instead of exactly one) that
reintroduces the exact parametric-memorization confound the metric exists
to remove. **No COUPLED/DECOUPLED/FLOOR verdict was ever computed, is
licensed, or may be inferred from anything on this page.** These values
are preserved here — not deleted, this is quarantine, not destruction —
purely as a provenance record, and are walled off because reading them
defeats a specific downstream blind pre-registration step (§9.1).

**Why this file exists.** `matrix-thinking/queue/regate_2026-07-12.md`
§10.2/§10.3 interleaved these per-rung outcome VALUES with the
methodological findings that explain them, in the same paragraphs and
sentences — so any agent required to read the methodological record for
context got contaminated as a side effect, with no way to read one
without the other. This file extracts every such value; the original
locations (listed below) now carry qualitative restatements only that
preserve 100% of what a builder needs and nothing about direction.

## Pointer back to source

| Value moved from | Commit | Section / lines (as originally committed) |
|---|---|---|
| `matrix-thinking/queue/regate_2026-07-12.md` | `05de661` | §10.2 (FATAL-1), §10.3 (F-4, F-3, M-11) |
| `matrix-thinking/deltanet_rd/lm_recall_gap_probe_rd.py` | `05de661` | module header (DO-NOT-USE banner, original lines 4-51) + 2 inline code comments (near line 299, near line 491) |
| `matrix-thinking/PARAM_AXIS_SCALING_DESIGN.md` | `d0e2798` | §9.2 (two bug-magnitude figures, in the placebo-arm rationale and the "kills F-4" paragraph) |

---

## 1. From `regate_2026-07-12.md` §10.2 — FATAL-1, verbatim as committed at `05de661`

> `run_ar_hit_gap_eval` clones **one** `window_ablated` tensor per batch,
> corrupts **every** candidate's antecedent in it, and runs **one** forward
> pass. Measured corruption density: **12.6% of context tokens (openr1;
> mean 64.6 distinct corrupted positions per 512-token row), 6.0%
> (wikitext)**. So `acc_ablated` measures *"accuracy on a context where an
> eighth of the tokens are random garbage"*, **not** *"accuracy with this
> one antecedent removed"* — and the "gap" collapses back toward the **raw
> AR-hit slice**, which is precisely the parametric-memorization confound
> §7-F3 named as the attacker's kill shot. The metric's *formula* is right;
> the *implementation* is not surgical.
>
> **The false-DECOUPLED is REALIZED, not hypothetical — in the wave's own
> numbers:**
> - **wikitext is self-contradictory.** T2 (one-shot in-context copy of a
>   planted bigram at distance 20) reads `acc_intact = 0.0000` at **14M,
>   98M AND 392M** — these checkpoints have *zero* demonstrable in-context
>   copy ability. The same checkpoints report an "in-context recall gap" of
>   **0.0454 → 0.1719 → 0.1882, rising monotonically with params.** A model
>   with no copy capability cannot have a 0.19 in-context-recall gap. The
>   gap is reading context damage.
> - **the verdict flips on an unpinned normalization.** On openr1, raw gap
>   (0.169/0.238/0.248/0.244) rises-then-plateaus → **DECOUPLED-leaning**.
>   The same JSON normalized by the model's accuracy on ordinary non-AR
>   tokens (0.455/0.499/0.480/0.477) is **flat-to-declining 98M→1.31B** →
>   **COUPLED-leaning.** The design pinned the metric but never pinned
>   whether it is normalized for general capability. **Two admissible
>   choices, opposite headlines, same data.** Disqualifying on its own.

## 2. From `regate_2026-07-12.md` §10.3 — F-4, F-3, M-11, verbatim as committed at `05de661`

> - **F-4 — the headline rung was measured against a different candidate
>   population.** The eval batch size was conflated with the
>   token-arithmetic batch size, so the three batch-32 rungs hit a
>   4000/batch candidate cap (128,000 = 32×4000 exactly; true uncapped count
>   156,908 → **18.4% of candidates dropped**, systematically the late rows)
>   while the **batch-16 1.31B rung never hit it (uncapped)**. The in-code
>   claim that the cap "does not differentially bias the cross-rung
>   comparison" is **FALSE for 1.31B**. Also: the shipped code (cap=20000)
>   **cannot reproduce** the R0 JSON (computed at 4000).
> - **F-3 — Wave −1 quantifies the artifact rather than validating the
>   instrument.** In the *passing* v2 run the shuffled-control gap is
>   **0.0865 on n=5,873 ⇒ ~13σ from zero, and 44% of the real gap.** It
>   "passed" only because T1's bar is an arbitrary *absolute* 0.10. The
>   v1→v2 change (seq_len 256→512) **halves corruption density** — which is
>   *why* the shuffled gap fell. The build's "small-sample artifact" reading
>   was **wrong**; this is mechanistic confirmation of FATAL-1.
> - **M-11 — a pre-registered gate was made easier after it failed.** T2 was
>   moved from distance 350 to distance **20** and its bar cut from absolute
>   >0.9 to >100×chance — contra §7-F8's explicit instruction to
>   **strengthen** it. Recorded as a finding against this build, not
>   defended. (The underlying observation — fast-weight one-shot copy is
>   genuinely weak, and *rises with scale*: 14M 0.03 → 1.31B 0.375 — may be
>   a real, publishable finding, but it must be **re-pinned**, not adjusted
>   in place.)

## 3. From `lm_recall_gap_probe_rd.py` module header — DO-NOT-USE banner, verbatim as committed at `05de661` (original lines 4-51)

```
################ DO NOT USE FOR A VERDICT -- INSTRUMENT IS VOID ################
#
# STATUS (2026-07-12): **VOID / DEFECTIVE.** Committed as the artifact of
# record for the build+audit chain, NOT as a working instrument. An
# independent opus audit found 4 FATALs; the coordinator independently
# re-verified the load-bearing one against this code and the raw data.
# Full record: matrix-thinking/queue/regate_2026-07-12.md S10.
#
# THE DEFECT (FATAL-1): `run_ar_hit_gap_eval` clones ONE `window_ablated`
# tensor per batch and corrupts EVERY candidate's antecedent position in
# it, then runs ONE forward pass (see the loop below, and the comment at
# its head). So a candidate's "ablated" read is taken from a context in
# which ~12.6% of tokens (openr1; 6.0% wikitext -- MEASURED) have been
# replaced with random garbage, not one in which its own single antecedent
# was removed. `acc_ablated` therefore measures GENERIC CONTEXT DAMAGE, and
# the reported "gap" collapses back toward the RAW AR-hit slice -- which is
# precisely the parametric-memorization confound FIX-B exists to remove
# (PARAM_AXIS_SCALING_DESIGN.md S7 F3, "the attacker's kill shot"). The
# metric's FORMULA is right; this IMPLEMENTATION reintroduces the confound.
#
# The false-DECOUPLED is not hypothetical -- it is REALIZED in this
# instrument's own output (/tmp/r0_ar_hit_full.json, RETRACTED): wikitext
# reads a "recall gap" of 0.19 rising with scale at rungs where T2 says the
# model has EXACTLY ZERO in-context copy ability (acc_intact = 0.0000).
#
# BEFORE ANY RE-USE, all of:
#   1. ONE corrupted antecedent per forward pass (batch so each row carries
#      exactly one ablation).
#   2. ADD A PLACEBO-ABLATION ARM -- corrupt a MATCHED COUNT of random
#      NON-antecedent positions at matched distances, and report the
#      difference-in-differences. Without this the gap is NOT IDENTIFIED.
#      (`_shuffle_rows`/T1 is NOT a substitute: it preserves the token
#      multiset and manufactures fresh random adjacencies that genuinely
#      repeat, so its "null" contains real in-context repeats by
#      construction -- its shuffled gap reads 13-sigma-nonzero.)
#   3. PRE-REGISTER the normalization (raw gap vs gap/acc_baseline_nonAR)
#      BEFORE reading. The two admissible choices give OPPOSITE verdicts on
#      the same JSON (raw -> DECOUPLED-leaning; normalized -> COUPLED-
#      leaning). This build's author has now SEEN both, and is therefore
#      contaminated for this choice -- it must be pinned by someone who
#      has not, or by the PI.
#   4. Decouple the EVAL batch size from the token-arithmetic batch size
#      (their conflation in param_axis_r0_driver.py made the 1.31B rung the
#      only UNCAPPED cell -- it was compared against three CAPPED ones).
#   5. Re-pin T2 explicitly (it was weakened here, contra S7 F8's explicit
#      instruction to STRENGTHEN it -- disclosed, and itself a finding).
#
################################################################################
```

## 4. From `lm_recall_gap_probe_rd.py` inline code comments, verbatim as committed at `05de661`

Near original line 299 (inside the FATAL-1 loop):
```
            # !!!!!! FATAL-1 LIVES HERE (see the module header) !!!!!!
            # ONE cloned tensor + ONE forward pass for ALL candidates in the
            # batch => every candidate's antecedent is corrupted SIMULTANEOUSLY
            # (~12.6% of context tokens on openr1, measured). Each candidate's
            # "ablated" read is therefore taken from a mass-corrupted context,
```

Near original line 491 (T2 pass-bar rationale):
```
        # T2 pass bar is CHANCE-RELATIVE (chance=1/50257=1.99e-5), not an absolute
        # transformer-calibrated 90% bar -- v1 of this instrument used a >0.9 absolute
        # bar and every rung "failed" T2; debugging (see report) found the DeltaNet/
        # fast-weight family's one-shot copy accuracy is genuinely low in absolute
        # terms (a real, literature-consistent finding, Arora et al./Zoology), NOT an
        # instrument bug -- confirmed by (a) acc_ablated reading exactly at floor
        # every time (the ablation mechanism has teeth), and (b) acc_intact rising
        # with model scale (14M 0.03 -> 1.31B 0.375 at the same distance) rather than
        # being flatly zero everywhere. "Reads high" is therefore: >=100x chance AND
        # the ablated arm is not (i.e. ablation genuinely destroys the signal).
```

## 5. From `PARAM_AXIS_SCALING_DESIGN.md` §9.2, verbatim as committed at `d0e2798`

Placebo-arm rationale paragraph:
> Without it, the gap is **not identified**: `acc_intact − acc_ablated` confounds
> *"this specific antecedent was removed"* with *"a token somewhere upstream was
> replaced with garbage."* FATAL-1 is the extreme case of that confound (12.6% of
> the context corrupted at once), but the confound exists **even at one corrupted
> token**...

"This also kills F-4" paragraph:
> **This also kills F-4.** The candidate cap is now **per-row (`C_max = 8`), fixed,
> and rung-invariant** — never a per-*batch* cap, which is what silently made the
> batch-16 1.31B rung the only uncapped cell while the three batch-32 rungs dropped
> 18.4% of their candidates. The eval batch size is **decoupled** from the
> token-arithmetic batch size and from candidate selection entirely.

---

**End of quarantined values.** Everything above is retracted, void, and
carries no verdict. Do not read further into this repository's history,
`/tmp` artifacts, or `experiment-runs/` for the same numbers in an attempt
to reconstruct a verdict — none exists; R0 is VOID by the design's own
gate (`PARAM_AXIS_SCALING_DESIGN.md` §9.5's VOID row: "§9.1 is still
unpinned" is itself a VOID condition).
