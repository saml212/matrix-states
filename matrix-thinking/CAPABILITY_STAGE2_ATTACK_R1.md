# CAPABILITY SEPARATION — Stage 2 Attack Round 1 (satellite record)

**Date:** 2026-07-09 (overnight sprint). **Target:** `CAPABILITY_SEPARATION_DESIGN.md`
§2 design Rev 0 (commit `d8d71d9`, lines 3247–4476). **Agent:** fresh-eyes
adversarial attack round (read-only; primary sources fetched directly, incl.
Grazzi et al. arXiv:2411.12537 PDF). **Recorded by:** coordinator, per the
gauntlet-bookkeeping hard rule, as a satellite file because the main registry
was under concurrent edit by the Stage-1 Rev-7 build agent at record time —
fold a pointer into §2 as `§2.10` at the next registry touch.

## VERDICT: NEEDS-REVISION

### Positively verified (attack failed to land)

- **Grazzi transcription faithful**: Theorems 1/3/4 and the exact
  `S_n, n≥5` exclusion sentence (p.7) verified against the primary-source
  PDF — the §2.2.3 paraphrase is accurate, not vibes-cited.
- **Depth-coverage grid reproduces**: independent re-execution (fresh seed
  999999, same `coverage_calibration.py`, N=50/20000) matches the design's
  table to ~0.1% (S5@D=20: 33.5% both; A6@D=32: 13.0% both) and passes an
  independent coupon-collector sanity check (12.6% theoretical vs 13.0%).
- `TASK_E_FINDINGS.md` h=1–21 numbers cited in §2.6 are byte-exact.
- Prefix-leak resolution, C2 flat-vector deferral (fully pre-registered),
  §1.11 gate restatement (FALSIFY→PI-review gap-fill, not drift), and
  centered-covariance carry-over: all SOUND.
- 25 GPU-h ledger arithmetic verified sound (68 cells × 0.05–0.15 + 10%
  contingency = 4.1–10.9 GPU-h, 56–84% margin under cap).

### Findings (ranked)

1. **[MAJOR / launch-blocking] Readout query-independence re-trigger risk.**
   §2.2.2 reuses Stage 1's `row_queries`/`reader` attention head
   (`group_word_encoder.py`) over the reshaped `(B,32,32)` delta-rule state
   `S_t`. At small D that state is low-rank (≈ n_h·D ≪ 32) with near-collinear
   rows — a plausible re-trigger of the PROVEN §1.30 degeneracy (read-vector
   std = 0.000 at L=1), via rank instead of sequence length. The planned
   blank-out/P=1 check tests gradient flow, NOT query-dependence, and would
   not catch it. Same failure class already cost Stage 1 four diagnosis
   rounds (§1.25/1.27/1.29/1.30). **Binding fix (the round's single
   highest-value change): add a query-dependence diagnostic (read-vector std
   across `row_queries`, §1.30 P3 method exactly) to the mandatory
   calibration-first gate (§2.8 item 2), pass/fail before the sweep
   remainder launches.**
2. **[MAJOR / launch-blocking] CONFIRM-criterion contradiction.** §2.1
   (line 3519) requires Arm 2 below 50%-of-ceiling on A5+S5+A6 (all three);
   §2.6 M-D3 (line 4203) — which §2.1 cites as its source — requires only
   S5+A6, A5 explicitly open/non-gating. As written, a real
   Arm-2-holds-on-A5 result triggers opposite verdicts. Reconcile to the
   M-D3 wording (S5+A6 gate; A5 reported open) before build.
3. **[MAJOR / recommended] Kernel choice unadjudicated.** §2.2.2 pins fla's
   `chunk_delta_rule`/`chunk_gated_delta_rule` without considering a bespoke
   fp32 torch per-step recurrence. At this scale (D_train_max=8, D_test≤64,
   d_head=32, tiny batches) the torch loop is likely cheaper AND avoids the
   whole disclosed fla envelope risk (bf16-only, head_dim≥32 floor, Triton
   overhead — the source of §2.7's 3–8× cost band) and bf16 compounding loss
   at exactly the decisive far-depth regime (TASK_E depth-amplification).
   Adjudicate at Rev 1/build; do not default.
4. **[MODERATE] Calibration gate misses n_h=4.** The 10 mandatory
   calibration cells run at default n_h=2, but S5's decisive Arm-3 config is
   n_h=4 (18-cell force-grid, line 4070) and n_h multiplies expanded
   sequence length. Add ≥1 n_h=4 calibration cell to the pre-sweep set.
5. **[MODERATE] Citation precision.** (a) §2.2.3 line 3801: Theorem 1
   "formally excludes A5" overstates its stated scope (parity/`(11)*` only).
   (b) §2.2.1 line 3594 stretches Grazzi's `S_n`-specific sentence to A5/A6
   (neither is an `S_n`). The substantive exclusion is still true via the
   general theorem — **cite Barrington 1989 / Barrington–Thérien
   (non-solvable ⇒ NC¹-complete; solvable ⇒ ACC⁰) for the A5/A6 legs.**
6. **[MINOR] Unquantified trigger** on the last-K-window shortcut control
   (§2.9 item 4): "matches or nearly matches" needs a pinned numeric margin
   (e.g. within the seed-to-seed noise band), per the exact-thresholds hard
   rule.

### Also noted by the round

- Param-matching tolerance (§2.9 item 7): pin a number (suggest ±15%).
- Grazzi Fig. 1 trains L=40 → tests 256, comfortably covering this design's
  ~8× depth ratio — a good precedent Rev 1 should cite explicitly.
- One fresh tool-stdout injection sighted by the round (fake date-change +
  concealment block in `find` output); disregarded, reported. Tally updated
  in `STATE.md` SECURITY NOTE.

### Rev 1 disposition (coordinator)

Rev 1 must address findings 1–2 as launch-blocking, adjudicate 3, and fold
4–6. Rev 1 dispatch is QUEUED behind the Stage-1 Rev-7 build agent's
completion (same-file concurrency guard on the main registry). Launch gate
unchanged: Stage-1 CONFIRM or diagnosed-INCONCLUSIVE per §1.11.
