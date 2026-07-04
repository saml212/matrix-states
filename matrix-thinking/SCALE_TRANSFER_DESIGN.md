# SCALE-TRANSFER Design — Does the Write-Geometry Attractor and Its Fix Survive Scale, and Does Either Transfer Off This Project's Own Code?

> **Rev 2 — 2026-07-04, post-attack-round.** An independent adversarial review of
> Rev 1 returned **NEEDS-REVISION (no FATALs)**: 5 MAJOR, 5 MINOR. Every finding is
> addressed in this revision; the finding→change map is **§13** (placed at the end so
> §1–§12 read as the current design, not a diff). The load-bearing changes: **MAJOR-1**
> — Rev 1's §1.1 "correction" was itself wrong on both of its claims (the "45×" figure
> DOES exist, in `DELTANET_RD_EXACTNESS_DESIGN.md` §16's own title, and the K=32
> admissibility failure was FIXED same-day by the `geo3_n_iter=20` escalation, 0/3 →
> 3/3 admissible, §16.3–16.4) — §1.1 rewritten, Track A's manifest now carries the
> escalation cells (`geo3n20`) as the PRIMARY admissible K=32 comparison; **MAJOR-2** —
> Track B's β-gated construction cannot claim to close the write-position problem
> (non-selected positions still write; β is never masked in LM mode) — a second,
> decision-bearing Wave −1 gate criterion (excluded-position write-mass, thresholds
> registered now) added, claim scope downgraded until measured; **MAJOR-3** — Rev 1
> conflated `chunk_size=64` (a kernel sequence-tiling constant) with `d_state=64` (the
> real Newton–Schulz capacity ceiling) — corrected, with the window rule at Track C's
> `d_state=128` rungs registered; **MAJOR-4** — the brief's "438K tok/s/GPU" figure is
> untraceable in the repo; replaced with the archive-derived **≈361K tok/s/GPU**
> (recomputed from `waveC/*.json`, formula in §5.6) at MFU ≈3.1%; **MAJOR-5** — the
> rung-1 augmented-mix control cell promoted from "if budget allows" to REQUIRED and
> non-cuttable (CLAUDE.md's hold-the-second-axis-fixed hard rule), rung-2/3 headlines
> scoped as scale+data-mix joint claims until it reads out. Two additional fixes were
> applied during the review window itself (a §7 budget-arithmetic error and a
> Falcon-Mamba license misattribution) — verified landed by the reviewer, §13.

**Design only, no training/model code written here.** Track B/C/D build may start on
this revision. Track A has no build phase (pure analysis of already-archived JSON) and
can start immediately.

**Program ceiling: ~300 GPU-h before mandatory human sign-off.** This document's
central-estimate manifest totals **≈130–153 GPU-h** across all four tracks (§7,
depending on whether Track D's conditional graft phase is later authorized), with a
**wide-case ceiling of ≈216 GPU-h** if every track lands at the top of its estimated
range — still **≥84 GPU-h under the 300 GPU-h ceiling** even in that scenario,
deliberately, because Track C's cost is the
least-validated number in this document (§5.8 item 3) and a bad calibration surprise
should not by itself blow through a human-authorized budget.

---

## 0. Reading list this design builds on (context, not repeated here)

- `STATE.md` — "Chapter 2 — STATUS" and "Path Forward" sections: the five closed
  2026-07-01→04 programs, the exactness-mechanism follow-on (living), the campaign
  ledger and hardware/budget facts this design inherits directly (2-month uptime-metered
  Brev window, ~192 GPU-h/day ceiling on the *hardware*; this document's 300 GPU-h figure
  is a separate, tighter, human-set ceiling on *this specific program*, not the full
  hardware budget).
- `matrix-thinking/DELTANET_RD_EXACTNESS_DESIGN.md` — the exactness-mechanism study.
  §2 (mechanism hypotheses a–g), §3 (the closed-form crosstalk reconstruction — the
  instrument this design's Track A analysis reuses conceptually, not literally), §14
  (F-geo-3: the Newton–Schulz per-episode key-orthogonalization mechanism this design's
  Track B ports to free-running text), §14.5–14.6 (cross-episode drift, finding F1),
  **§16 (the F-geo-3 results section — Wave geo3 + the `geo3_n_iter=20` escalation;
  the primary source for every K=16/K=32 number this design inherits, §1.1)**.
- `EXPERIMENT_LOG.md`, **"F-geo-3 WAVE VERDICT (2026-07-04)"** and **"F-geo-3
  escalation VERDICT"** (the latter supersedes the former's K=32 admissibility
  status — Rev 1 of this document read only the first entry and §1.1 was wrong as a
  result; see §13, MAJOR-1).
- `matrix-thinking/DELTANET_REALDATA_DESIGN.md` §19 — the LM Wave 2 (Waves C+D)
  instrument set: real-corpus DeltaNet-LM pretrain + inference-time rank-truncation
  grid. This design's Track B and Track C both extend this exact instrument set rather
  than inventing a new one.
- `matrix-thinking/deltanet_rd/lm_pretrain_rd.py` — the audited LM harness Track B/C
  build on. Read in full for this design (architecture, data pipeline, key/query/value
  path, geo3 hook status — none exists yet).
- `matrix-thinking/H100_SETUP.md` — hardware access, the perpetual-sweep pattern, the
  calibration-first discipline this design inherits and applies to every new scale
  point (§9).

---

## 1. Program framing

Five closed programs plus one living follow-on (`DELTANET_RD_EXACTNESS_DESIGN.md`)
established, at a single fixed small scale (13–14M-param DeltaNet, `d_state=64`):
(1) rank is causally load-bearing on real tokenized text (`DELTANET_REALDATA_DESIGN.md`
§17.11); (2) the trained operator is graded/sub-exact rather than razor-cliff-exact,
unlike the hand-built-orthonormal-key synthetic harness; (3) the sub-exactness is
attributable, mostly, to write-time key geometry (`DELTANET_RD_EXACTNESS_DESIGN.md`
Wave 1 ATTRIBUTION VERDICT, `EXPERIMENT_LOG.md` 2026-07-04); (4) a differentiable fix
(F-geo-3, per-episode Newton–Schulz key-orthogonalization) demonstrably moves the
frontier — K=16 hits its minimum-publishable bar 3/3, and K=32, admissible 3/3 after
the `geo3_n_iter=20` escalation, improves ~43–56× over baseline while narrowly missing
its headline bar on the mean (§1.1).

None of this has been tested outside one fixed scale, one from-scratch synthetic-grammar
harness, and this project's own code. **SCALE-TRANSFER asks three separable questions,
one per remaining axis:**

1. **Sample efficiency** (Track A, zero GPU): given what's already archived, how much
   *faster* does the fixed thing get fixed, in training steps, not just how much better
   is the final ceiling?
2. **Scale transfer** (Tracks B, C): does the mechanism (write-geometry attractor) and
   its fix (geo3) survive the move from a closed synthetic grammar to free-running real
   text (Track B), and from 13M to 1.3B parameters (Track C)?
3. **Generalizability** (Track D): does anything resembling this mechanism exist in a
   pretrained, fixed-recurrent-state model this project did not build — i.e., is this a
   property of the delta-rule family generally, or an artifact of this project's own
   harness?

### 1.1 The F-geo-3 K=32 record, stated correctly (REWRITTEN, Rev 2 — MAJOR-1; Rev 1's version of this section was itself wrong on both of its claims)

The authoritative source is `DELTANET_RD_EXACTNESS_DESIGN.md` **§16** ("F-geo-3
results (Wave geo3 + escalation)") plus `EXPERIMENT_LOG.md`'s **"F-geo-3 escalation
VERDICT"** entry. The correct record:

- **K=16 (`geo3_n_iter=12`): minimum-publishable bar HIT, 3/3 admissible.** h=4
  `rec@0.9` mean 0.9767 [0.9525–0.9969] vs. a bar of ≥0.8 and a learned-arm baseline
  of 0.419–0.465 (§16.4).
- **K=32: admissible 3/3 after the `geo3_n_iter=20` escalation** (§16.3 — the
  original `n_iter=12` wave's 0/3 admissibility failure was a Newton–Schulz eigh
  fallback triggering on 56/11/374 of 20,000 steps per seed; at `n_iter=20` the
  fallback count is zero on all 3 seeds, and the behavioral numbers are essentially
  unchanged, §16.5). The admissible result: **h=4 `rec@0.9` mean 0.4368
  [0.3903–0.5045] vs. baseline 0.009 — a ~43–56× improvement (mean ≈48×, consistent
  with §16's own "45×" title figure) — but the headline bar (≥0.5 on the mean) is
  narrowly MISSED (0.4368, ~0.06 short; seed 0 individually clears it at 0.5045)**,
  with the residual attributed to the pre-registered outcome F (stable-not-just-
  orthogonal geometry — cross-episode drift), not to a broken fix (§16.4, §16.6).

**What Rev 1 of this document got wrong, recorded per house transparency norms:** it
read only the earlier "F-geo-3 WAVE VERDICT" log entry, concluded "K=32 was never
admitted" (stale — the same-day escalation fixed admissibility), and asserted "no
document contains a 45× figure" (false — §16's title). Both errors are corrected
throughout this revision. Consequences for this design: the K=32 geo3 cells are
**admissible evidence** (via the `geo3n20` runs), Track A's manifest carries them as
the primary K=32 comparison (§3.5), and the correct citation discipline is: cite the
**escalation** (`geo3_n_iter=20`) numbers for any admissible claim; the `n_iter=12`
K=32 numbers remain descriptive-only per §16.2's own admission-gate discipline. The
headline framing that travels with any K=32 citation: **"admissible, ~48× over
baseline, narrowly missed its pre-registered headline bar (0.4368 vs. ≥0.5 mean)."**

---

## 2. Claim tiers (defined once; every track cites one)

Adopting `DELTANET_REALDATA_DESIGN.md` §19's convention directly, extended:

- **Tier 1 — Premise-conditional causal.** Only earned by a provable `rank≥K` bound
  plus a train-time force-rank ablation on a closed, controlled task (the synthetic
  grammar's own results). **Nothing built in this program earns Tier 1** — it is
  inherited, cited, never re-derived here.
- **Tier 2 — Descriptive + interventional.** "The trained state's measured geometry
  differs under condition X" and "an intervention changed a measured quantity by Y, at
  this scale, on this corpus." Track B and Track C's core readouts live here.
- **Tier 3 — Measurement-only.** No causal or interventional language at all — "this
  quantity, measured in this pretrained model, has this value." Track D's Phase 1
  (measurement) lives here; Track D's Phase 2 (graft) would, if it ever runs, earn at
  most Tier 2 for one specific model, never a general claim about the delta-rule family.

No track in this document is authorized to publish language stronger than its tier.

---

## 3. TRACK A — Steps-to-criterion sample efficiency from archived checkpoints (zero GPU, run first)

### 3.1 Hypothesis

Archived per-checkpoint trajectories in `experiment-runs/2026-07-03_deltanet_rd_waves/exactness/{wave1,wavegeo3}`
already contain everything needed to compute a **seed-resolved, hop-resolved
steps-to-criterion curve** comparing the geo3 arm against its matched learned baseline
(arm iii-β) at K=16 and K=32 — at zero new GPU-hours — replacing whatever informal
"100 steps vs. 0.23" observation motivated this track with an actual, sourced number.

### 3.2 What's already in hand — verified this session, not assumed

Every JSON in `exactness/{wave1,waveF,wavegeo3,k48rider}` (82 files, 122MB — recounted
Rev 2, MINOR-1) logs two sequences per run:

- `trajectory`: loss-only (`loss`, `loss_cos`, `loss_nce`, `eff_rank_trainbatch`,
  `skip_rate_so_far`), **every ~200 steps** (101 points over a 20,000-step run). No
  recovery metric at this resolution.
- `checkpoints`: the full recovery/rank diagnostic (`recovered_frac@0.9/0.95/0.99`,
  `mean_cos`, `effective_rank_whole_mean`, `key_gram_deviation_mean`, broken out by
  `M2_in_distribution` h=1–3, `M3_held_out` h=4–7 and h=21, `C17_heldout_entities`,
  `C19_heldout_template`), **every 2,000 steps** (10 checkpoints per 20,000-step run).

Checkpoint step grids are **identical across arms**: `[2000, 4000, ..., 20000]` for
`wave1` (baseline, arm names `armi`/`armii`/`armiii-beta`/`armiv`/`armi-strong`) and
for `wavegeo3` — which contains **both** the original wave
(`wgeo3_rdx_K{16,32}_armgeo3_s{0,1,2}_geo3n12.json`) **and the `geo3_n_iter=20`
escalation cells (`wgeo3_rdx_K32_armgeo3_s{0,1,2}_geo3n20.json`) — the admissible K=32
runs Rev 1 of this document missed entirely (Rev 2, MAJOR-1)**. The direct
baseline-vs-geo3 pairs are **arm `iii-beta` vs. arm `geo3`**: at K=16, `geo3n12` (3/3
admissible); at K=32, **`geo3n20` as primary** (3/3 admissible per
`DELTANET_RD_EXACTNESS_DESIGN.md` §16.3) with `geo3n12` retained as a
descriptive-only secondary (its behavioral numbers are near-identical per §16.5, which
is itself a finding worth reproducing in the steps-to-criterion frame).
`n_params=12,899,841` confirmed in every file's `n_params` field.

**Resolution ceiling, stated plainly: 2,000 steps.** The finer-grained `trajectory`
array has no recovery metric, only loss and a *trainbatch* effective-rank proxy — usable
as a secondary, lower-confidence early-training signal (§3.4), never as a substitute for
the recovery-threshold criterion.

### 3.3 The "100-steps-vs-0.23" teaser — status: not found, do not imply otherwise

An exhaustive grep of `EXPERIMENT_LOG.md`, `STATE.md`, every `matrix-thinking/*.md`
design doc, and `pebble-ai-site/` for "100 steps," "0.23," "steps-to-criterion," and
"sample efficiency" (verified this session, two independent passes) **found no passage
combining these numbers.** The checkpoint grid does not even contain a step-100 sample
(the first checkpoint is step 2,000). **This design does not treat the teaser as a real,
citable prior result.** Track A's deliverable is framed as an independent,
first-principles steps-to-criterion analysis that happens to address the same question
the teaser gestured at — not a validation of a number that cannot currently be traced to
any archived data.

### 3.4 Method

For each (K, seed, hop h) cell, walk the 10 checkpoints in order and record the first
step at which `recovered_frac@0.9` crosses each of two pre-registered thresholds
(`≥0.5`, `≥0.8` — chosen to bracket Wave F's own "minimum publishable" bar of 0.8 used
for the K=16 verdict). Report per-seed crossing steps (not just a pooled mean — small-n
seed spread has already bitten this program once, `DELTANET_RD_EXACTNESS_DESIGN.md`
finding 5). If a threshold is never crossed within 20,000 steps, record `>20000`, not a
missing value. Secondary, lower-confidence readout: the `trajectory` array's
`eff_rank_trainbatch` and `loss_cos`, plotted at their native ~200-step resolution,
labeled explicitly as a proxy that does not carry the recovery-threshold claim.

Primary hops: **h=1** (in-distribution, the no-sacrifice guard) and **h=4** (the
held-out hop every prior verdict in this program has used as its headline). Secondary:
h=2, h=3, h=7, h=21 (depth-amplification check), C17/C19 generalization controls — full
table, not just the headline hops, since the analysis is free.

### 3.5 Cells + budget

| Cell | Source files | Seeds | Cost |
|---|---|---|---|
| K=16 baseline (arm iii-β) | `exactness/w1_iiibeta_K16_s{0,1}.json` + `wavegeo3` baseline reference | 2 | 0 GPU-h (analysis only) |
| K=16 geo3 (admissible 3/3) | `exactness/wavegeo3/wgeo3_rdx_K16_armgeo3_s{0,1,2}_geo3n12.json` | 3 | 0 |
| K=32 baseline (arm iii-β) | `exactness/w1_iiibeta_K32_s{0,1,2}.json` | 3 | 0 |
| **K=32 geo3, PRIMARY (escalation, admissible 3/3)** | `exactness/wavegeo3/wgeo3_rdx_K32_armgeo3_s{0,1,2}_geo3n20.json` | 3 | 0 |
| K=32 geo3, secondary (`n_iter=12`, descriptive only — non-admissible per §16.2's gate) | `exactness/wavegeo3/wgeo3_rdx_K32_armgeo3_s{0,1,2}_geo3n12.json` | 3 | 0 |
| **Total** | | | **0 GPU-h; ~1–2 analyst-hours to write and run the extraction script** |

The `n12`-vs-`n20` K=32 pair also yields a free bonus readout: whether the extra
Newton–Schulz iterations change the *steps-to-criterion* profile at all
(`DELTANET_RD_EXACTNESS_DESIGN.md` §16.5 already showed the endpoint behavioral
numbers are near-identical; whether the trajectories match too is unmeasured and
free to check here).

### 3.6 Success/failure criteria

Not a CONFIRM/FALSIFY gate — a **descriptive extraction** with two possible honest
outcomes, both useful: (a) geo3 crosses the recovery thresholds at an earlier checkpoint
than baseline, at matched K and hop → geo3 helps sample efficiency, not just ceiling,
and this becomes a real, sourced number to replace the teaser; (b) geo3's advantage is
entirely a final-ceiling effect (crosses at the same or a later checkpoint than
baseline, or baseline never crosses at all so no comparison exists) → geo3 is a
capacity/ceiling fix, not a speed fix — equally worth reporting, and it would directly
contradict a "faster too" framing if anyone has been assuming one.

### 3.7 Attack-yourself

1. The motivating "100 steps vs 0.23" number is untraceable (§3.3) — report the
   extraction as new work, not verification of a prior claim.
2. 2,000-step resolution cannot distinguish "by step 100" from "by step 1,999." State
   the ceiling in every output; never round a crossing step down to imply finer
   resolution than the data supports.
3. geo3 archives exist only at K=16 and K=32 — no K=8 or K=48 geo3 run exists. Do not
   extrapolate a K-trend from two points.
4. n=2–3 seeds per cell is small. Report per-seed numbers, not just a pooled mean,
   exactly as `DELTANET_RD_EXACTNESS_DESIGN.md`'s own finding 5 already requires
   elsewhere in this program.
5. **(REWRITTEN, Rev 2 — MAJOR-1.)** The K=32 admissibility picture is
   two-tiered: the `geo3n20` escalation cells are 3/3 admissible (the primary
   comparison); the original `geo3n12` cells are 0/3 admissible and stay
   descriptive-only per §16.2's own admission-gate discipline. Track A's output
   must tag every K=32 number with which tier it came from — presenting an
   `n_iter=12` trajectory as if it carried the escalation's admissibility would
   repeat, in miniature, exactly the citation error Rev 1 of this document made.

### 3.8 Results (run 2026-07-03, zero GPU, archive at commit `fc3ded1`)

**Script:** `matrix-thinking/deltanet_rd/analyze_sample_efficiency.py`. **Full output:**
`experiment-runs/2026-07-04_track_a/track_a_sample_efficiency.json` (336 aggregated
(arm×K×hop×threshold) cells, every one seed-resolved) and
`track_a_summary.md` (human-readable). 23 archived run files parsed (all with
`n_params=12,899,841`, verified by assertion, not assumed). Thresholds 0.5/0.8 are
§3.4's own pre-registration; **0.9 is ADDED for this run** (labeled as such in every
output row) since it was explicitly requested and is a legitimate third, stricter
criterion — never presented as part of the original pre-registration.

**Headline finding, stated first because it reframes the rest: within the archived
step budget (20,000 steps) and the 2,000-step checkpoint resolution, only 2 of the
336 analyzed cells yield a genuine, non-censored steps-to-criterion RATIO** (both
seeds crossed in both arms, so an exact interpolated-step comparison exists) —
**both are secondary C19 (held-out-template) controls, not the primary h=1/h=2/h=4
hops**: K=16, C19 h=2 @ threshold 0.8 (baseline 13,321.7 vs geo3 8,709.9 interpolated
steps, **1.53×**), and K=16, C19 h=3 @ threshold 0.5 (baseline 12,831.1 vs geo3
8,953.8, **1.43×**). Every other cell in the table is either **resolution-bound**
(both arms already at/above threshold at the very first checkpoint, step ≤2,000 —
the grid cannot see anything faster) or **ceiling-bound** (the baseline's own
asymptotic capacity sits below the tested threshold, so it never crosses within the
full 20,000-step run — there is no baseline crossing to race against). **This
archive cannot, at this resolution, distinguish "geo3 also gets there faster" from
"geo3 gets somewhere baseline structurally cannot reach at all"** — the two genuinely
measured ratios above (1.4–1.5×) are modest, which leans the honest reading toward
§3.6's outcome (b) — **geo3 reads primarily as a capacity/ceiling fix, not a
demonstrated speed fix** — but this is an absence-of-evidence limitation of the
2,000-step grid, not proof against a real early-training speed effect (a large speed
edge inside the first 2,000 steps would look identical, in this data, to a pure
ceiling effect). The "100-steps-vs-0.23" teaser (§3.3) remains untraceable; nothing
below verifies it — this is new analysis addressing the same question.

**Headline table — geo3 (primary tier per K) vs baseline, h=1/h=2/h=4, all
thresholds** (K=48 and arm i-strong are bonus, out-of-manifest cells, §3.8.3):

| K | hop | thr. | prereg? | baseline steps | geo3 steps | ratio | read |
|---|---|---|---|---|---|---|---|
| 16 | h=1 | 0.5/0.8/0.9 | — | ≤2000 (left-cens., all seeds) | ≤2000 (left-cens., all seeds) | n/a | both saturate before the grid can see anything — resolution-bound |
| 16 | h=2 | 0.5 | yes | ≤2000 (left-cens.) | ≤2000 (left-cens.) | n/a | resolution-bound |
| 16 | h=2 | 0.8/0.9 | yes/ADDED | mixed (1 of 2 seeds crossed) | ≤2000 (left-cens.) | n/a | seeds disagree within baseline — no pooled number reported, see per-seed |
| 16 | h=4 | 0.5 | yes | **>20000 (right-cens., all seeds)** | ≤2000 (left-cens.) | **≥10×** | baseline's own ceiling (0.419–0.465, §1.1) never reaches 0.5 — ceiling-bound lower bound only |
| 16 | h=4 | 0.8 | yes | >20000 (right-cens.) | 2453.0 | **≥8.2×** | baseline never crosses; geo3's real step is known, baseline's isn't — lower bound |
| 16 | h=4 | 0.9 | ADDED | >20000 (right-cens.) | 3355.0 | **≥6.0×** | same pattern, stricter threshold |
| 32 | h=1 | 0.5 | yes | ≤2000 (left-cens.) | ≤2000 (left-cens.) | n/a | resolution-bound |
| 32 | h=1 | 0.8/0.9 | yes/ADDED | >20000 (right-cens.) | ≤2000 (left-cens.) | **≥10×** | baseline plateaus ~0.77–0.79 (§1.1), never reaches 0.8/0.9 — ceiling-bound |
| 32 | h=2 | 0.5/0.8/0.9 | — | >20000 (right-cens.) | ≤2000 (left-cens.) | **≥10×** | baseline plateaus ~0.22–0.27 at h=2 — ceiling-bound at every tested threshold |
| 32 | h=4 (PRIMARY) | 0.5 | yes | >20000 (right-cens.) | mixed (1 of 3 crossed) | n/a | matches §1.1 exactly: only seed 0 (0.5045) individually clears the 0.5 bar |
| 32 | h=4 (PRIMARY) | 0.8/0.9 | yes/ADDED | >20000 (right-cens.) | >20000 (right-cens.) | n/a | **mutual ceiling failure** — geo3's own K=32 mean (0.4368) sits below both thresholds too |

All "≥N×" figures are **crude, resolution-imposed lower bounds** (computed as
20,000/2,000 = 10× when both arms are censored at their respective boundaries, or
20,000/geo3's real interpolated step when only geo3's step is known) — **not
measured ratios**. They are reported because they are still informative (they
rule out geo3 being *slower*), but the true ratio could be much larger or, at the
resolution this grid can see, unknowable.

**3.8.1 — K=32 PRIMARY (`geo3n20`) is confirmed as the only admissible K=32 geo3
comparison; `geo3n12` stays descriptive-only, matching §3.7 item 5 exactly.** Every
`geo3n20` file's own `geo3_admission.admissible` field (read directly from the JSON,
not inferred from the filename) is `True` for all 3 K=32 seeds and all 3 K=16 seeds;
every `geo3n12` K=32 file's own field is `False` for all 3 seeds (cause:
`ns_converged_no_fallback=False`, 56/11/374 fallback steps — matches
`DELTANET_RD_EXACTNESS_DESIGN.md` §16.3 exactly). K=16 has only one tier
(`geo3n12`, admissible 3/3) — there was never an n_iter=20 escalation needed at K=16.

**3.8.2 — Free bonus readout: do the `n_iter=12` and `n_iter=20` K=32 trajectories
match, not just their endpoints?** Yes, where measurable: at h=4, threshold 0.5 (the
only headline cell where both tiers had a defined interpolated mean), `n12` crosses
at 16,448.8 steps vs `n20` at 16,358.7 — a 90.1-step delta, well within the 2,000-step
checkpoint resolution. This extends `DELTANET_RD_EXACTNESS_DESIGN.md` §16.5's
endpoint-only finding ("behavioral numbers do not move") to the trajectory shape
itself, for the one cell where the comparison is measurable at all.

**3.8.3 — Bonus arms, explicitly outside §3.5's pre-registered manifest.** Two extra
comparisons were run because their archived dumps exist and were requested for this
pass; neither is part of the headline geo3-vs-baseline comparison above:

- **Arm i-strong (`strong_pin=True`, K=32 only, 3 seeds).** Saturates *every* headline
  hop (h=1, h=2, h=4) by the first checkpoint in every seed (all left-censored) — a
  substantially stronger baseline variant than arm iii-β, included here for context
  only, not as a geo3 comparator.
- **K=48 (baseline `k48_learned_s{0,1,2}.json` + geo3 `wgeo3_rdx_K48_armgeo3_..._geo3n20.json`,
  3 seeds each).** §3.7 item 3 correctly noted no K=48 geo3 run existed at design
  time; one landed live in the archive during this analysis session (commit
  `fc3ded1`, "geo3: K=48 stretch results", already logged in `EXPERIMENT_LOG.md`'s
  "F-geo-3 K=48 stretch" entry). Included as a **bonus third K-point, not a
  pre-registered manifest extension** — attack item 3's "do not extrapolate a
  K-trend from two points" still applies to the pre-registered K=16/K=32 pair; three
  points is still too few for a trend claim. K=48 geo3 is **non-admissible 0/3**,
  but for a **different reason than K=32's original `n_iter=12` failure**:
  `value_salvage_tier_pass=False` (not an unconverged Newton–Schulz fallback —
  `ns_converged_no_fallback=True` at K=48, i.e. the orthogonalization itself
  converges cleanly; a different admission criterion fails instead, consistent with
  the K=48 stretch's own note that K=48 sits past the `d/2` boundary).

**3.8.4 — Caveats (restated from §3.7, now with concrete numbers behind them).**
Resolution is 2,000 steps; every `interpolated_step` is a linear estimate between
the two bracketing checkpoints, explicitly labeled as such in the JSON, never
presented as a finer-grained measurement. n=2 (K=16 baseline) or n=3 seeds per cell
throughout; every pooled number in this section has its full per-seed breakdown in
`track_a_sample_efficiency.json`'s `aggregated[].per_seed` — cells where seeds
disagreed in censoring status are reported as `mixed` with no pooled ratio computed
at all, rather than averaging incomparable left/right/exact statuses together.

---

## 4. TRACK B — geo3-in-LM (installing per-episode key orthogonalization in a free-running-text harness)

### 4.1 Hypothesis

A free-running-text adaptation of F-geo-3 (§4.2 below) measurably reduces key-Gram
deviation and/or inference-time rank-truncation damage on real GPT-2-tokenized text
relative to the unmodified Wave C baseline (`DELTANET_REALDATA_DESIGN.md` §19), without
a net validation-loss regression beyond a small pre-registered tolerance.

### 4.2 THE key design problem — what is the orthogonalization set for free-running text?

F-geo-3, as built (`model_rd.py`'s `bind()`, `newton_schulz_orthogonalize`), depends on
three properties of the synthetic grammar that **do not hold on real text**, verified by
direct code read this session:

1. **A small, known K fixed at batch-construction time.** The synthetic harness samples
   exactly K entities per episode by construction. Real text has no natural K — a
   document can run to thousands of tokens, far past `d_state=64`, which is also
   Newton–Schulz's own hard capacity ceiling (`K` mutually orthonormal directions cannot
   exceed `d_state`).
2. **A discrete, sparse write-position set (`item_pos`) distinguishable from inert
   positions.** The synthetic harness's β is architecturally zero at all non-write
   positions. `lm_pretrain_rd.py`'s β (`DeltaNetLMMixer.forward`, computed from
   post-norm `x`, `torch.sigmoid(self.b_proj(x))`) is a **continuous, never-masked**
   gate — every one of up to 512 positions per window is a real write with a real
   learned β. There is no `item_pos` to gather.
3. **K guaranteed-distinct entity identities.** Orthogonalizing genuinely distinct
   synthetic entities is a targeted correction. Orthogonalizing an arbitrary window of
   real tokens forces apart items that are frequently *not* distinct identities at all —
   repeated function words, subword fragments of the same word, punctuation — which is a
   substantively different, more lossy intervention than the one F-geo-3 was validated
   on.

**Naive ports that this design rejects, and why (attack-yourself performed in advance,
not after building):**

- *Per-document orthogonalization via `doc_offsets`* — rejected outright. OpenR1-Math
  documents average ~500–700 tokens, WikiText-103 articles run to thousands; both vastly
  exceed `d_state=64`, violating the capacity ceiling before any other consideration.
- *Unconditional fixed-window orthogonalization (orthogonalize every token in a
  window)* — the "naive" baseline. Not rejected outright — **kept as a cheap comparison
  arm** (§4.5) specifically to test whether the position-selection refinement below
  matters, since a priori it might not.

**The proposed primary construction: a chunk-local, β-gated top-K subset.**

1. **Window = `chunk_delta_rule`'s own internal `chunk_size=64` boundary** — chosen as
   a convenient, kernel-aligned tiling boundary, not an arbitrary new hyperparameter.
   **(CORRECTED, Rev 2 — MAJOR-3.)** Rev 1 claimed this window "saturates Newton–
   Schulz's capacity ceiling by construction" — that conflated two unrelated
   constants: `chunk_size=64` is a **sequence-tiling** constant of the kernel
   (independent of state size, per `model_rd.py`'s own comments), while `d_state` is
   the actual capacity ceiling (at most `d_state` mutually orthonormal key
   directions exist). They are merely coincidentally equal in the 13–14M harness
   (`d_state=64`). The corrected, registered rule: **the binding constraint is
   `K_sel ≤ d_state`; the window stays chunk-aligned at 64 for kernel convenience,
   and `K_sel ≤ min(window, d_state)`.** At this track's `d_state=64` the two
   coincide and nothing changes numerically; at Track C's `d_state=128` rungs the
   rule is registered in §5.5 item 3 (window stays 64; `K_sel ≤ 64` there too — a
   64-token window cannot supply more than 64 candidate positions regardless of
   state headroom).
2. **Within each chunk, select the top-`K_sel` positions by β magnitude** (`K_sel` a
   hyperparameter, tested at 16 and 32 — the two K values with F-geo-3 synthetic
   reference results: K=16 admissible and bar-hitting; K=32 admissible via the
   escalation, ~48× over baseline, narrowly missed its headline bar on the mean,
   §1.1) as the orthogonalization set, leaving all other positions in the chunk
   untouched. This is the free-text analog of `item_pos`: instead of a
   hand-specified write mask, the model's own learned β nominates which positions are
   "write-worthy." **(SCOPE CORRECTED, Rev 2 — MAJOR-2.)** Rev 1 claimed this
   "closes failure mode 2." It does not, and the claim is withdrawn: in LM mode β is
   a **continuous, never-masked** sigmoid (`lm_pretrain_rd.py`'s own docstring: "a
   PLAIN LEARNED sigmoid gate at every position... NO mask") — so the ~32–48
   non-selected positions per chunk **still perform real, non-orthogonalized writes
   into the same shared state**, unlike the synthetic harness where non-write
   positions have architecturally-zero β. The honest statement: this construction
   orthogonalizes the *nominated* writes while leaving a background of unnominated
   write-interference whose magnitude is **unmeasured** — measuring it is Wave −1
   gate criterion (b) below, and until that measurement exists, Track B's mechanism
   premise is provisional, not established.
3. **EOT and any padding positions are hard-excluded from selection** (never
   orthogonalized, never counted toward `K_sel`), mirroring F-geo-2's established
   content-position-only convention for buffer handling.

**Mandatory pre-check — now TWO registered gate criteria, both decision-bearing
(REVISED, Rev 2 — MAJOR-2):** load an **already-archived Wave C checkpoint**
(`experiment-runs/2026-07-04_lm_rd_wave2/waveC/`) and run a forward-pass-only probe
(no training) measuring, per 64-token chunk:

- **(a) β-concentration:** the fraction of per-chunk β-mass captured by the top-`K_sel`
  positions (plus a Gini coefficient as a shape check). **Registered threshold: the
  top-32 set must capture ≥60% of chunk β-mass** (i.e., meaningfully above the 50%
  a uniform β would give at `K_sel=32` of 64) for the β-gated construction to be
  meaningfully different from a random subsample.
- **(b) excluded-position write-mass (NEW, Rev 2):** the mean β at **non-selected**
  positions, and the complement of (a) — the fraction of total chunk write-mass the
  orthogonalization never touches. **Registered threshold: mean β at non-selected
  positions ≤0.25 AND non-selected write-mass ≤40% of chunk total.** If non-selected
  write-mass exceeds this, the mechanism premise fails at the root — orthogonalizing
  a minority of the write-mass cannot be expected to control state geometry — and
  **Track B routes to redesign, not launch**: the registered outcomes are (i) both
  pass → β-gated primary proceeds; (ii) (a) fails, (b) passes → fall back to the
  naive-fixed-window arm as primary; (iii) **(b) fails → no Wave 1 launch on this
  construction at all** — a hard-zero-β-at-non-selected-positions variant (option
  (a) of the attack round's framing — architecturally masking unnominated writes,
  which changes the LM's expressivity and is a genuinely different model) is noted
  as the conditional follow-on, **requiring its own attack pass before any build**,
  since it reintroduces exactly the kind of hard-masking machinery LM mode was
  deliberately built without.

Cost: <1 GPU-h (a handful of forward passes on an existing checkpoint). Decided and
recorded in the wave summary **before** Wave 1 proper launches. This Wave −1 is
genuinely decision-bearing — it can kill the track's primary construction, not just
tune it.

### 4.3 Insertion point (code-level)

`DeltaNetLMMixer.forward` (`lm_pretrain_rd.py:203–249`), after `k, _ = self.k_conv1d(...)`
and `beta = torch.sigmoid(self.b_proj(x))`, before the `chunk_delta_rule(...)` call:
gather each chunk's top-`K_sel`-by-β keys (already L2-normalized — the harness applies
`use_qk_l2norm_in_kernel=True` **inside** the kernel rather than externally, unlike
`model_rd.py`'s `bind()`; the ported code must pre-normalize the selected keys itself
before calling `newton_schulz_orthogonalize`, then feed the orthogonalized keys back
through the same in-kernel-normalization path — a plumbing mismatch here is exactly the
kind of subtle scaling bug this project's "Wave −1 smoke" convention exists to catch,
required before any real run, §4.5). Scatter the orthogonalized keys back at their
original positions; leave every non-selected position's key untouched.

### 4.4 Required new instrumentation

`lm_pretrain_rd.py` currently has **no** `key_gram_deviation` or recovery-style
machinery (confirmed by direct code read — its own docstring disclaims this). Porting
`key_gram_deviation_mean`-style logging (computed over each chunk's selected `K_sel`
keys, at eval checkpoints) is a required, small build item, reusing `rank_utils.py`'s
existing rank functions and the Gram-deviation computation already implemented in
`model_rd.py`. A drift diagnostic (adapted from `geo3_drift_diagnostic.py`, measuring
how much a token's orthogonalized key direction changes across the differently-composed
chunks it appears in across training) is also required — free text's chunk membership
for any given token is far less stable than a synthetic episode's fixed K-cycle, and
this project's own drift finding on the synthetic side
(`DELTANET_RD_EXACTNESS_DESIGN.md` §16.1's gate record: training shrinks drift
0.47→0.94 at K=16 but both trained levels still land in the pre-registered HIGH band,
and §16's own K=32 residual is attributed to exactly this outcome-F drift) was
measured on a much better-behaved case than free text will be.

### 4.5 Cells + budget

Same corpora, seeds, step budget, and eval protocol as Wave C/D
(`d_model=256, d_state=64, n_layers=2`, 6,103 steps, checkpoints every 1,000 steps) —
**deliberately identical to the existing baseline so the comparison is clean**, per this
project's own "use the same dataset for ALL experiments in a comparison" rule.

| Wave | Purpose | Scope | Est. GPU-h | Gate |
|---|---|---|---|---|
| **−1 (calibration, blocking, decision-bearing)** | The two registered gate criteria of §4.2 — (a) β-concentration ≥60% top-32 chunk mass, (b) excluded-position write-mass: mean non-selected β ≤0.25 AND non-selected write-mass ≤40% — measured on archived Wave C checkpoints; plus Wave −1 smoke of the orthogonalize→re-normalize→kernel plumbing on hand-built keys (shape/scale assertions, no training) | forward-pass only + unit test | <1 | (a)+(b) pass → β-gated primary; (a) fails, (b) passes → naive-window primary; **(b) fails → NO Wave 1 launch, track routes to redesign** (§4.2's registered outcome iii) — recorded before Wave 1 launches |
| **1 (primary)** | β-gated geo3-in-LM training, `K_sel∈{16,32}` × 2 corpora (openr1, wikitext) × 3 seeds | 12 runs at ~6,103 steps each | ~10–14 | Val loss within pre-registered tolerance of baseline at ≥1 `K_sel`; geometry/truncation readouts computed |
| **2 (ablation, cut first if squeezed)** | Naive-fixed-window arm, same `K_sel` grid, 1 corpus (openr1) × 3 seeds only | 6 runs | ~3–5 | Comparison against Wave 1's β-gated arm — tests whether the selection refinement matters at all |
| **3 (eval-only)** | Wave-D-style inference-time truncation grid on Wave 1's trained checkpoints | truncation grid × held-out eval, no backward pass | ~2–4 | Descriptive+interventional result recorded, Tier 2 |
| **Total** | | | **≈16–24 GPU-h** (sum reconciled Rev 2, MINOR-2) | Slightly above the brief's ~10–20 GPU-h target at the top of the range; Wave 2 is the first cut if the total needs trimming |

### 4.6 Success/failure criteria

**Success (Tier 2 language only, per §2):** at ≥1 `K_sel`, key-Gram deviation at
intervened chunks drops meaningfully below Wave C's archived 1.26–2.77 band, val loss
stays within the pre-registered tolerance (proposed: +2% relative to the matched Wave C
baseline cell), and truncation-damage curves shift favorably at low k. **A geometry win
at a real loss cost is explicitly NOT a success** — report it as a genuine trade-off, not
a partial win.

**Failure, all honestly informative:** (a) the β-concentration gate (§4.2 criterion
(a)) fails, falling back to the naive-window arm as primary — a real finding about β's
role in this harness, not a bug; (b) **the excluded-position write-mass gate (§4.2
criterion (b)) fails — no Wave 1 launch at all**; the mechanism premise (that a
top-`K_sel` subset controls enough of the write-mass to matter) is dead on this
harness, and the finding is reported as such — the hard-zero-β variant becomes a
conditional follow-on gated behind its own attack pass (§4.2); (c) geometry improves
but val loss regresses beyond tolerance — evidence that Newton–Schulz
orthogonalization actively fights the LM objective on real text, exactly the risk
named in §4.2's rejected-constructions discussion; (d) neither geometry nor loss
moves meaningfully — the intervention doesn't reach real text at this scale/window
definition, motivating a different episode definition as a documented follow-on, not
a silent retry.

### 4.7 Claim tier

**Tier 2 throughout.** Track B may never claim "real language models need rank-K
structure" from an intervention's effect on loss or geometry — only "this intervention,
at this window/selection definition, changed this measured quantity by this amount, on
this corpus, at this scale," matching `DELTANET_REALDATA_DESIGN.md` §19's own precedent
exactly.

### 4.8 Attack-yourself

1. **(SHARPENED, Rev 2 — MAJOR-2.)** The β-gated top-K construction is a plausible
   guess whose central premise — that the selected subset controls most of the
   write-mass — is structurally not guaranteed in LM mode (β writes at *every*
   position, unmasked; the ~32–48 non-selected positions per chunk keep writing
   non-orthogonally into the same state). §4.2's Wave −1 gate criterion (b) exists
   precisely to measure this before any training spend, and its failure branch is a
   registered no-launch, not a fallback. Until that measurement exists, every Track
   B claim about the mechanism is provisional.
2. Forcing near-duplicate real tokens apart is not obviously beneficial for language
   modeling even if it improves a narrow Gram-deviation statistic — §4.6's val-loss
   tolerance gate exists specifically to prevent metric-shopping around this risk.
3. The kernel's internal L2-normalization (`use_qk_l2norm_in_kernel=True`) sits at a
   different point in the pipeline than `model_rd.py`'s external normalization —
   plumbing this correctly is a real, non-cosmetic build risk, and it must be
   smoke-tested with hand-built keys and explicit shape/scale assertions before any real
   training run, not discovered mid-sweep.
4. Cross-chunk drift on real text (a token's chunk membership, and hence its
   orthogonalized neighbors, changes far more across training/context than a synthetic
   K-cycle's fixed episode membership) is unmeasured and could be worse-behaved than the
   synthetic case's "resolved favorably" finding — §4.4's drift diagnostic is a
   first-class readout, not an afterthought, precisely because this risk is real and
   otherwise invisible.
5. Any headline here is Tier 2 at best (§4.7) — restated here because it is the single
   most likely place a write-up could accidentally over-claim.

---

## 5. TRACK C — The scaling ladder (100M / 400M / 1.3B params)

### 5.1 Hypothesis

The non-orthonormal write-geometry attractor and its sub-exact composition frontier
(measured at 13–14M params in Wave C/exactness) **persist** — do not spontaneously
resolve via scale alone — as DeltaNet-LM parameter count increases through
~100M → ~400M → ~1.3B, on real-text training budgets that are intentionally
sub-Chinchilla but scaled with model size.

This is explicitly a **persistence-vs-dissolution** question, not a directional bet —
both outcomes are pre-registered as informative (§5.6).

### 5.2 Required harness change (build-time, before any rung runs)

**`lm_pretrain_rd.py` currently hard-asserts `n_layers ∈ {1, 2}`** (`:281–282`,
confirmed by direct code read). Every rung in this ladder needs `n_layers` in the
12–22 range (§5.3). Relaxing this assert is real engineering work, not a config-flag
change — it requires verifying nothing else in the architecture implicitly assumed
`n_layers≤2` (residual/positional scaling, checkpoint-loading shapes), and a full
smoke test (forward pass, backward pass, gradient check) at the new depth, per
`CLAUDE.md`'s standing rule, **before** the first real rung run.

### 5.3 Rung configs — approximate, must be verified empirically before launch

Using the per-layer cost decomposition confirmed by direct code read
(`vocab×d_model` tied embedding/head + `n_layers×(8·d_model² + 4·d_model·d_state)`,
the `8·d_model²` term being the FFN's 4× expansion and `4·d_model·d_state` the four
`d_model×d_state`-scale projections — **q/k/v (`d_model→d_state`) plus `o_proj`
(`d_state→d_model`); β's `b_proj` is `d_model→num_heads`, negligible** (corrected
label, Rev 2 — MINOR-3; the arithmetic was already right, the term naming was not)):

| Rung | d_model | n_layers | d_state | Approx. params | Target |
|---|---|---|---|---|---|
| 1 | 768 | 12 | 64 | ≈98M | ~100M |
| 2 | 1536 | 16 | 128 | ≈392M | ~400M |
| 3 | 2560 | 22 | 128 | ≈1.31B | ~1.3B |

**This table is a planning approximation, not a build spec.** The actual per-layer
formula may omit norm/conv overhead and may not capture the exact shape of the β
projection (possibly per-head, `lm_pretrain_rd.py` carries an `H` dimension whose
exact sizing wasn't fully resolved in this design pass). **Every rung's config must be
verified against the harness's own printed/asserted parameter count (the same
convention already used for the 13–14M cell, bounds `[10M,16M]`) before any GPU time is
spent on it** — this is a Wave −1 (§5.5) blocking item, not assumed correct from this
table.

### 5.4 Data — what exists, what's named as augmentation, and why

**What's already on the box (per `rebuild_lm_corpora_rd.py`, verified this session):**
OpenR1-Math EOT-separated, 43.7M train tokens (the full `open-r1/OpenR1-Math-220k`
train split, bit-exact re-tokenized with document boundaries — this is already the
**entire** available corpus, not a slice; there is no "more OpenR1" to draw on).
WikiText-103 EOT-separated, ~103M tokens.

**Named augmentation (verified this session via WebSearch, not assumed from training
data):**

- **OpenWebMath** (`open-web-math/open-web-math`, HuggingFace) — 6.3M documents, 14.7B
  tokens, **ODC-By 1.0 license** (research use unrestricted; CommonCrawl ToU applies).
  Reasoning-side augmentation once OpenR1-Math's fixed 43.7M-token budget is exhausted
  by the epoch cap below.
- **FineWeb-Edu** (`HuggingFaceFW/fineweb-edu`, HuggingFace) — 1.3T tokens, **ODC-By 1.0
  license**. Narrative-side augmentation/substitute for WikiText-103 at the two larger
  rungs, where WikiText-103's 103M tokens alone would force far more repetition than the
  reasoning side gets from OpenWebMath.

Both are HF-hosted and streamable on the existing box (HF datasets tooling already used
for `open-r1/OpenR1-Math-220k` and `Salesforce/wikitext`) — no new access/licensing
blocker.

**Epoch discipline:** cap any single source's repetition at ≤5 physical epochs (a
standard heuristic against memorization-driven metric contamination at this data-to-param
ratio); once a rung's token budget (§5.5) exceeds that cap for OpenR1/WikiText alone,
the remainder is drawn from OpenWebMath/FineWeb-Edu.

**Named, not resolved here:** even with augmentation, the proposed budgets (§5.5) remain
far below Chinchilla-optimal for the two larger rungs (1.3B wants ≈26B tokens per
standard scaling laws; this design proposes ≈3B). This is acceptable for a **mechanism**
question (does the geometric attractor appear/persist) but not for any capability or
perplexity claim — stated explicitly here so it cannot be silently forgotten at
write-up time, mirroring the exact discipline Wave C already applied to its own
6,103-step, 13M-param run.

### 5.5 The three readouts

1. **Attractor persistence (primary, all 3 rungs):** key-Gram deviation, effective/stable
   rank, and (if the frontier-probe transplant, item 2, validates) held-out-hop recovery
   — the same instrument set as Wave C/exactness, applied unmodified at each rung.
2. **Frontier-probe transplant (rungs 1 and 3 only — cut order drops rung 2 first,
   §8):** splice the from-scratch synthetic grammar's K-cycle probe task onto the
   *pretrained* LM's own embedding table and backbone (freeze or light-adapt), and
   measure `recovered_frac@0.9` the same way the exactness program measured it on a
   from-scratch model. **This is a genuinely untested transplant** — the pretrained
   embedding table's structure comes from real-text next-token prediction, not the
   K-cycle binding objective, and there is no existing evidence its single-token
   "entity" rows behave anything like the synthetic grammar's clean, purpose-built
   vocabulary. **Mandatory validation step:** run the transplant at rung 1 first, and
   require it to reach non-trivial recovery on an *unconstrained* (no orthogonality
   requirement, no geo3) baseline before trusting any frontier number at rung 3 — a null
   result here could be a measurement artifact, not a finding, and must be distinguished
   before being reported as either.
3. **Fix-effect at scale (rungs 1 and 3 only, gated on Track B validating — see
   Sequencing, §11):** apply Track B's validated geo3-in-LM construction at the smallest
   and largest rungs, skipping rung 2 (cut order, §8), to see whether the fix's effect
   (§4.1) itself holds, grows, or shrinks with scale. **Window rule at `d_state=128`
   rungs, registered now (Rev 2 — MAJOR-3):** the orthogonalization window stays
   chunk-aligned at the kernel's `chunk_size=64` (a sequence-tiling constant that does
   not scale with `d_state`), and `K_sel ≤ min(window, d_state) = 64` — the extra state
   headroom at `d_state=128` is deliberately NOT exploited by widening the window in
   this program (a wider window would be a new, unvalidated episode definition, not a
   transplant of Track B's validated one; if the headroom question matters it is a
   documented follow-on, not a silent change bundled into the scale axis).

### 5.6 Cells + budget — pre-calibration, wide error bars, revise after Wave −1

**Throughput is genuinely uncertain and stated as such. (NUMBER CORRECTED, Rev 2 —
MAJOR-4.)** The brief's "438K tok/s/GPU" figure is untraceable in the repo; the
archive-derived measurement, recomputed directly from the Wave C result JSONs
(`experiment-runs/2026-07-04_lm_rd_wave2/waveC/wC_lm_*.json`, formula:
`steps_completed × batch_size × seq_len / wall_s` = `6103 × 32 × 512 / ~274–278 s`),
is **≈361K tok/s/GPU (359–365K across the 6 runs)** at `n_params=14,048,896` (the LM
harness's own count — note this is the ~14M LM model, not the synthetic harness's
12.9M). That corresponds to MFU ≈3.1% of H100 peak bf16 (`6·N·throughput / 990
TFLOPS`) — meaning the small model is bound by per-step/kernel-launch overhead, not
compute, and naively extrapolating a compute-bound model from an overhead-bound
reference point is exactly the kind of unvalidated extrapolation this project's own
calibration-first rule exists to catch. The estimates below use a 25% MFU
compute-bound assumption for planning purposes only and are explicitly **superseded
by Wave −1's measured numbers** before Wave 1's manifest is finalized. (The build
phase must recompute the reference throughput from the archived JSONs with the
formula above rather than citing this paragraph's rounding.)

| Wave | Purpose | Scope | Est. GPU-h (pre-calibration, ±2×) | Gate |
|---|---|---|---|---|
| **−1 (blocking, all 3 rungs)** | Harness `n_layers` assert relaxation + smoke test (§5.2); per-rung parameter-count verification against §5.3's table; per-rung throughput calibration (short run, measure real tok/s before committing full budget); **re-verify OpenWebMath/FineWeb-Edu licenses + HF availability at build time** (both read ODC-By 1.0 as of this design's own WebSearch pass, but the check is re-run, not inherited — MINOR-5) | 3 short calibration runs | ~3–6 | Smoke passes; measured param counts within 15% of target (else adjust config); measured throughput recorded and used to re-price Waves 1–3 below |
| **1 (rung 1, ~100M) + REQUIRED small-model mix control** | Attractor persistence + frontier-probe transplant validation, 2 corpora (openr1-mix, wikitext/finewebedu-mix) × 3 seeds; **plus the REQUIRED data-mix control cell (Rev 2 — MAJOR-5): the Wave-C-scale 13–14M architecture retrained on the SAME augmented mixes, 2 mixes × 3 seeds — at Wave C's measured ~4.6 min/run this costs well under 1 GPU-h and isolates the data-mix axis from the scale axis per CLAUDE.md's hold-the-second-axis-fixed hard rule** | 6 rung-1 runs + 6 control runs | ~2–5 | Geometry/rank readouts logged; frontier-probe transplant reaches non-trivial recovery on the unconstrained baseline (blocking for rung 3's transplant); control cell's geometry read against archived Wave C (isolates mix effect) |
| **2 (rung 2, ~400M)** | Attractor persistence only, 2 corpora × 1 seed | 2 runs, ~1.5B tokens/run | ~6–16 | Geometry/rank readouts logged |
| **3 (rung 3, ~1.3B)** | Attractor persistence + frontier-probe transplant, 2 corpora × 1 seed | 2 runs, ~3B tokens/run | ~40–90 | Geometry/rank readouts logged; transplant only trusted if Wave 1's validation passed |
| **4 (fix-effect, gated on Track B)** | geo3-in-LM applied at rungs 1 and 3 only, 2 corpora × 1 seed each | 4 runs | ~15–35 | Fix-effect-vs-scale comparison recorded |
| **Total** | | | **≈66–152 GPU-h** (central estimate ≈100–115) | Within the brief's ~100–200 GPU-h; **Wave −1's measured throughput is the actual pricing authority, not this table** |

### 5.7 Success/failure criteria — both directions pre-registered as informative

**"Persists" (falsification-of-dissolution) criterion, stated before any data exists:**
if key-Gram deviation and held-out-hop composition sub-exactness do **not**
monotonically improve (deviation shrinking, recovery rising) across the 3 rungs, holding
data/eval protocol fixed, the attractor is scale-invariant (or worsens) — this
strengthens the case that a fix matters for real deployed-scale models, and is the
headline this track is named for. **Claim scoping (Rev 2 — MAJOR-5): until the
required small-model same-mix control cell (§5.6 Wave 1) reads out, every rung-2/3
headline in either direction is stated as a joint scale+data-mix result; the pure
scale attribution is only earned once the control isolates the mix axis.**

**"Dissolves" (the falsifying outcome for the persistence hypothesis) criterion:** if
gram deviation shrinks and composition becomes more exact monotonically across rungs,
the attractor is a small-model artifact scale alone resolves — equally reportable, and
it would mean the exactness-mechanism program's fix (geo3) is a small-model patch, not
something larger pretrained models need. **This is not a failure of the track; it is the
other pre-registered answer to the same question.**

The only true track-level failure mode is **an uninterpretable result**: e.g., the
frontier-probe transplant validation fails at rung 1 (§5.5 item 2's gate) and no
alternative recovery instrument is substituted, leaving only geometry readouts with no
compositional-recovery cross-check.

### 5.8 Attack-yourself

1. §5.3's param-count table is an approximation from a partially-verified per-layer
   formula — real counts will differ, possibly by 10–30%. Verified against the
   harness's own count before any GPU spend (Wave −1, blocking), never assumed from this
   table.
2. The `n_layers` assert relaxation (§5.2) is real, unaudited engineering work with its
   own smoke-test obligation — this design does not treat it as a trivial config change.
3. Throughput estimates rest on a single, non-compute-bound reference point (§5.6) —
   the entire manifest's cost is provisional pending Wave −1's measurement, stated
   explicitly rather than presented with false precision.
4. Even augmented, all three rungs are far below Chinchilla-optimal token budgets — no
   capability/perplexity claim is licensed by this track's checkpoints, stated in every
   write-up, not just here.
5. The frontier-probe transplant (§5.5 item 2) is untested at any scale in this project
   — a null result could be a measurement artifact (pretrained real-text embeddings
   don't behave like the synthetic grammar's clean vocabulary), not a finding. The rung-1
   validation gate exists specifically to catch this before it contaminates the rung-3
   headline.
6. **(UPGRADED from disclosure to required control, Rev 2 — MAJOR-5.)** Corpus mixing
   (OpenR1 + OpenWebMath; WikiText + FineWeb-Edu) changes domain composition relative
   to Wave C's clean two-corpus contrast — bundling it with the scale change violates
   CLAUDE.md's hold-the-second-axis-fixed hard rule if left uncontrolled. The
   small-model same-mix control cell is now **REQUIRED, part of Wave 1, and excluded
   from the cut order** (§5.6, §8) — it costs <1 GPU-h at Wave C's measured
   throughput. Until it reads out, **every rung-2/3 headline is scoped as a joint
   scale+data-mix claim, not a scale claim** — the write-up language is registered
   now, not negotiated later.

---

## 6. TRACK D — Measurement-first probing of a pretrained fixed-state model (conditional graft NOT authorized by this document)

### 6.1 Hypotheses (two, explicitly separated — only the first is authorized to run)

**H-measure (authorized):** at least one open, pretrained, fixed-recurrent-state LLM
exhibits a measurable, non-orthonormal write-geometry signature — via a Gram-deviation
/ rank diagnostic adapted to its specific state-update math — analogous to what this
project measured in its own from-scratch DeltaNet-LM.

**H-graft (NOT authorized — stated for completeness, gated per §6.4):** a single-layer,
λ-warmed-in geo3-style graft improves an appropriately adapted exactness/recall
diagnostic on a pretrained fixed-state model without collapsing its downstream
language-modeling quality. **This document pre-registers the mitigation candidates and
the gate criterion for H-graft; it does not authorize any graft run.**

### 6.2 Candidate model evaluation (verified this session via WebSearch, sources cited)

| | RWKV-7 "Goose" | Falcon-Mamba-7B | RecurrentGemma |
|---|---|---|---|
| Largest available | 13.3B / 7.2B raw `.pth` (`BlinkDL/rwkv7-g1`); **2.9B HF-native** (`RWKV/RWKV7-Goose-World3-2.9B-HF`) | 7B native (`tiiuae/falcon-mamba-7b`) | 9B (`google/recurrentgemma-9b`) |
| License | Apache 2.0 | Falcon Mamba 7B TII License v1.0 (Apache-2.0-derived; **corrected during this design's own verification pass** — royalty-free with no revenue threshold; an earlier draft of this table incorrectly carried the *Falcon-180B* license's $1M/yr royalty clause over to Falcon-Mamba-7B, which does not have one, confirmed by direct fetch of the license page) | Gemma license — gated access, Prohibited-Use Policy, not fully permissive |
| State exposure | `flash-linear-attention`'s `fla/layers/rwkv7.py` exposes WKV recurrent state via cache API | HF `transformers`/`mamba_ssm` expose per-layer `ssm_states`/`conv_states` | Reference repo's `RecurrentBlockCache` exposes `rg_lru_state` + `conv1d_state` |
| Architecture match to delta rule | **Closest** — RWKV-7 explicitly generalizes the delta rule (arXiv:2503.14456: "inspired by DeltaNet," same outer-product write structure `S_t=S_{t-1}(I−diag(w_t)k_tk_t^T)+v_tk_t^T`) | Diagonal-gated input-dependent SSM — `B_t`/write vectors exist but the analogy requires an interpretive choice, not a direct port | **Poor fit** — RG-LRU is a per-channel diagonal linear recurrence with no outer-product key-value write at all; no matrix state for a Gram diagnostic to measure |

**Decision: RWKV-7 (2.9B, HF-native) as primary — closest architecture match, lowest
engineering risk. Falcon-Mamba-7B as secondary — true 7B scale, clean state-exposure
path, but requires validating the `B_t`-as-key interpretive choice against the actual
`mamba_ssm` tensor shapes (not just the paper's equations) before trusting any number.
RecurrentGemma dropped from the primary track** — no outer-product write structure
exists to measure; include only as an explicit negative control if budget allows, never
in a public write-up given the license (§6.7 item 4).

**Sources:** [RWKV-7 paper, arXiv:2503.14456](https://arxiv.org/abs/2503.14456);
[BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1);
[fla RWKV7 layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/rwkv7.py);
[tiiuae/falcon-mamba-7b](https://huggingface.co/tiiuae/falcon-mamba-7b);
[Falcon Mamba license blog](https://huggingface.co/blog/falconmamba);
[google/recurrentgemma-9b](https://huggingface.co/google/recurrentgemma-9b).

### 6.3 Measurement protocol (Phase 1, authorized)

1. Load each candidate frozen (no fine-tuning). Run forward passes over a real-text eval
   set (reuse this project's own held-out OpenR1-Math/WikiText windows for continuity
   with Track B/C's instruments, plus a general-prose sample so the measurement isn't
   confined to this project's own corpora).
2. Extract per-layer state-write vectors at each token position: RWKV-7's `k_t` directly
   (architecturally identical role to this project's own keys); Falcon-Mamba's `B_t` (or
   `Δ_t B_t`) as the interpretive choice, validated against actual `mamba_ssm` forward-pass
   tensor shapes before use, not assumed from the paper's equations alone (§6.7 item 2).
3. Compute key/write-vector Gram deviation and state effective/stable rank using
   `rank_utils.py`'s existing functions unmodified (framework-agnostic — operates on
   any state tensor).
4. **Power/variance check before trusting any per-layer estimate:** a 7B model has far
   more layers/heads than this project's own 1–2-layer harness; the eval-window count
   needed for a stable per-layer Gram estimate must be determined empirically (e.g., a
   bootstrap stability check across window-count subsamples), not assumed to transfer
   from the small-model eval-window count.

### 6.4 The graft-and-finetune phase — conditional, gated, requires its own attack round

**Not authorized by this document.** Gated on Phase 1 showing: (a) a meaningful
non-orthonormal write-geometry attractor exists in the pretrained model, analogous to
this project's own finding; and (b) a plausible single-layer insertion point exists
without an obvious path to destroying pretrained capability.

**The bolt-on-lesson attack, pre-named because it is not hypothetical — it is a repeat
of a confirmed dead end in this project's own history.** The original matrix-CODI
program failed specifically because a matrix bottleneck was bolted onto a
vector-pretrained model with a vector-teacher signal — the readout co-adapted to the
*old* representation's statistics, and the bolt-on could not out-compete that
co-adaptation (`STATE.md`, "The Narrowed Hypothesis — STATUS"). Grafting a geo3-style
orthogonalization into a model whose downstream layers were trained end-to-end on the
*original*, non-orthonormal state statistics is the same failure shape: the graft may
improve a narrow geometry/recall diagnostic while degrading everything downstream that
learned to read the old distribution. **This must be evaluated with at least the same
skepticism that killed matrix-CODI, arguably more — here the graft target is a model
this project did not train and cannot as easily re-audit end-to-end.**

**Mitigation candidates (named now, not invented after a failed graft):**

1. **Blend-coefficient warm-in**, `λ: 0→1` over a brief fine-tune, so downstream layers
   have a gradual, not instantaneous, distribution shift to adapt to.
2. **Single-layer grafts** — intervene at only the one layer showing the largest
   measured Gram deviation (Phase 1's output), bounding the blast radius rather than
   modifying every layer at once.
3. **Incremental recall measurement** — evaluate recall/downstream quality at each step
   of the λ warm-in, with a pre-registered stop/back-off rule if a cliff appears, rather
   than committing to λ=1 blind.

**Hard gate, stated explicitly:** before any graft run, an independent adversarial
attack round must review the *specific* graft design (which layer, which λ schedule,
which base checkpoint, which recall metric, what the stop/back-off rule actually is) —
this document pre-registers the candidate mitigations and the go/no-go criterion; it
does not itself authorize the graft phase.

### 6.5 Cells + budget

| Phase | Purpose | Scope | Est. GPU-h | Gate |
|---|---|---|---|---|
| **1 (measurement, authorized)** | Load RWKV-7 2.9B + Falcon-Mamba-7B frozen; forward-pass state extraction over eval windows; Gram/rank diagnostics; power/variance check | inference only, no backward pass | ~5–10 | H-measure result recorded (Tier 3); go/no-go for Phase 2 decided here |
| **2 (graft, CONDITIONAL, not authorized)** | Single-layer geo3-style graft + λ warm-in fine-tune on whichever candidate Phase 1 flags | reserved, cut-protected last | ~15–30 (reserved) | Requires its own independent attack round (§6.4) before any run — this document does not clear that gate |
| **Total** | | | **~20–40 GPU-h ceiling** (Phase 1 ~5–10 spent regardless; Phase 2's ~15–30 stays reserved, unspent, unless separately authorized) | |

### 6.6 Success/failure criteria

**Phase 1 success:** a clean, reproducible Tier-3 measurement exists, regardless of
which direction it points. A null result (pretrained models show near-orthonormal write
geometry, no attractor) is exactly as valuable as a positive one here — it would mean
this project's finding is specific to training from scratch on a rank-forcing objective,
not a general property of delta-rule-family training, which is itself a real, reportable
boundary condition.

**Phase 1 failure:** the state-exposure path turns out not to give access to genuinely
comparable per-token write vectors (e.g., Falcon-Mamba's `B_t` analogy doesn't survive
contact with the actual tensor shapes, §6.7 item 2) — in which case Phase 1 narrows to
RWKV-7 only, reported as such, not silently dropped.

### 6.7 Attack-yourself

1. The bolt-on-lesson parallel (§6.4) is not a hypothetical risk — it is a repeat of a
   confirmed dead-end pattern in this project's own history, and the graft phase must
   not be greenlit on literature plausibility alone.
2. Falcon-Mamba's `B_t`-as-key interpretation is an unvalidated analogy — a code-level
   audit of the actual forward pass, not just the SSM equations, is required before any
   Falcon-Mamba geometry number is trusted; the analogy could turn out to be looser than
   hoped (state per channel-block, not a clean per-token key set).
3. A modest eval-window count calibrated on this project's own 1–2-layer harness may be
   statistically insufficient for a many-layer 7B model — §6.3 item 4's power check
   exists specifically because this has not been checked.
4. RecurrentGemma's Gemma license (gated, Prohibited-Use Policy) makes it unsuitable
   for any public write-up even as a negative control unless separately reviewed; scope
   any RecurrentGemma results as strictly internal.
5. Falcon-Mamba's own license (royalty-free, no revenue threshold — corrected in §6.2
   above; an earlier pass of this document wrongly imported the *Falcon-180B* license's
   $1M/yr royalty clause) still carries an Acceptable Use Policy and attribution
   requirements for derivatives; any future graft-and-redistribution intent needs
   separate legal review regardless — flagged now, not at publication time.

### 6.8 Phase 1 results (run 2026-07-04, GPU 6 only, ≈0.9 GPU-h — under the §6.5 estimate)

**Instrument:** `matrix-thinking/deltanet_rd/lm_attractor_probe_trackd.py` (new).
Non-invasive forward hooks on real `nn.Linear` submodules; no model code edited, no
gradients anywhere; Tier 3 language throughout per §2. Smoke gate: 9 items including
positive (orthonormal→0), negative (collapsed→√(K(K−1))), a centered-variant
discriminating control, health-gate positive/negative controls, multi-head
vectorization equivalence, and a duplicate-window regression test. Independent audit
(same session, separate agent): NO FATALs; 2 MAJOR both fixed and regression-tested
(duplicate-window visibility under the repeat fallback; per-head Python-loop SVD cost
vectorized). Full output: `experiment-runs/2026-07-04_track_d/attractor_probe_trackd.json`
(+ `trackd_summary.txt`, `run.log`, and the exact script copy, per house archive rule;
mirrored to SSD).

**Model picked — documented deviation from §6.2's literal primary.** §6.2 named RWKV-7
**2.9B**-HF as primary. That exact checkpoint is **broken in this environment**
(fla 0.5.1 + transformers 5.12.1): ~20/32 layers' token-shift parameters
(`x_r/x_w/x_k/x_v/x_a/x_g`) load as MISSING (the checkpoint ships a fused `x_x` key
that current `fla` does not consume), the "freshly initialized" replacements come up
NaN/Inf (bf16: 133 NaN + 5 Inf params; fp32: 55 NaN), and logits are NaN in both
dtypes. Confirmed **not** a systemic incompatibility: the 0.4B and 1.5B RWKV-7
checkpoints load clean (zero NaN/Inf, finite logits) in the same environment — this is
a stale conversion in that one HF repo. **Substitution: RWKV-7 1.5B
(`RWKV/RWKV7-Goose-World3-1.5B-HF`) as the RWKV-7 point**, with Falcon-Mamba-7B
(§6.2's secondary) supplying the true 7B-scale reading. A mandatory health gate
(param-level NaN/Inf scan + forward-pass logits-finiteness check) is now built into
the probe and refuses corrupted checkpoints — added because of this incident; it
would otherwise have silently poisoned every number.

**§12 Q4 resolved (minimal registered choice, recorded here):** negative control =
**Qwen2.5-1.5B** (Apache-2.0, native `transformers`, standard GQA softmax attention —
no fixed-size recurrent state of any kind), scale-matched to the actual RWKV-7 point
measured. Probe point: pre-RoPE `k_proj` output (2 KV heads × head_dim 128).

**State-update equations AS FOUND in the shipped code (not the papers):**

- RWKV-7 (`fla/layers/rwkv7.py` → `chunk_rwkv7` → `chunk_dplr_delta_rule`):
  `S_t = S_{t-1} @ (diag(w_t) + a_t⊗b_t) + v_t⊗k_t`, where `k = k_proj(xk)` is the
  **raw, unnormalized** value-write key (further mixed by the `k_a` gate) and
  `kk = L2norm_perhead(k * k_k)` is a **separately gated, L2-normalized** tensor
  forming the erase/decay pair `a_t = −kk`, `b_t = kk·sigmoid(a_lora(xa))`. **Not
  textbook DeltaNet**: write key and erase key are *different tensors*, and the decay
  is a learned diagonal `diag(w_t)`, not identity. The probe measures **`kk`** (the
  L2-normalized erase side — the convention match to our own probe), reconstructed
  outside the model from the `k_proj` hook plus the model's own `k_k` parameter.
- Falcon-Mamba (`transformers/models/falcon_mamba/modeling_falcon_mamba.py`):
  `h_t[c] = exp(dt_t[c]·A[c])·h_{t−1}[c] + dt_t[c]·B_t·u_t[c]` per channel — a
  diagonal-gated SSM with **no erase term** (not a delta rule; §6.2's caveat
  confirmed at code level). Write vector `B_t = rms_forward(split(x_proj(x))[1])`,
  dim **16** (`state_size`), shared across all channels; `rms_forward` is
  Falcon-Mamba's own unlearned RMS-normalization of B/C (absent in vanilla Mamba).
  The probe measures **`B_t`** post-`rms_forward`.

**The massive-activation confound — found empirically before trusting any number.**
Direct per-channel inspection of all three families found a dominant, largely
input-agnostic outlier channel (3–35× the median channel magnitude; mean pairwise
cosine 0.43–0.9998 across positions) — the known "massive activations" phenomenon
(Sun et al. 2024, arXiv:2402.17762). It alone drives raw Gram-deviation most of the
way to the collapse ceiling in **all three families, including the no-fixed-state
negative control** — so the raw statistic cannot by itself answer H-measure. The
probe therefore reports **both** conventions on every episode: `raw` (as registered,
§6.3 item 3) and `centered` (per-episode per-channel mean subtraction — removes
exactly the shared/constant component; smoke item [2b] proves it discriminates).

**Reference anchors** (how to read a Gram-deviation number): for K i.i.d. random unit
vectors in ℝ^d, E‖G−I‖_F ≈ √(K(K−1)/d); full collapse = √(K(K−1)). At K=16:
random ≈ 1.94 (d=64), 3.87 (d=16), 1.37 (d=128); collapse = 15.49. Our own 14M band
(0.6–4.4 over K=8–48, d_state=64) sits **at or below the random anchor** — the 14M
attractor is "non-orthonormal but no worse than random."

**Results (pooled across layers; per-layer detail in the archive JSON/summary).**
n_windows=16/corpus (halved from the CLI default after a timing pilot — SVD cost, not
forward cost, dominates; still 393K episodes per (RWKV-7, corpus, chunk=16) cell),
seq_len=512, both corpora, chunk sizes 16 and 64, bf16 models, fp32 statistics:

| Model | corpus | chunk | RAW gd (layer range) | CENTERED gd | random / collapse anchor |
|---|---|---|---|---|---|
| RWKV-7 1.5B (kk, d=64) | openr1 | 16 | **10.98** [8.3–12.8] | 5.56 | 1.94 / 15.49 |
| | wikitext | 16 | **10.84** [8.1–13.3] | 4.89 | 1.94 / 15.49 |
| | openr1 | 64 | 44.02 [34–52] | 21.14 | 7.94 / 63.50 |
| | wikitext | 64 | 43.46 [33–54] | 18.22 | 7.94 / 63.50 |
| Falcon-Mamba-7B (B_t, d=16) | openr1 | 16 | **12.63** [5.5–15.1] | 7.26 | 3.87 / 15.49 |
| | wikitext | 16 | **12.47** [5.0–14.7] | 7.11 | 3.87 / 15.49 |
| | openr1 | 64 | 50.22 [20–62] | 28.30 | 15.87 / 63.50 |
| | wikitext | 64 | 49.87 [18–60] | 27.53 | 15.87 / 63.50 |
| Qwen2.5-1.5B control (pre-RoPE k, d=128) | openr1 | 16 | **12.32** [10.9–15.5] | 5.93 | 1.37 / 15.49 |
| | wikitext | 16 | **11.68** [10.1–15.4] | 4.56 | 1.37 / 15.49 |
| | openr1 | 64 | 48.53 [41–63] | 22.24 | 5.61 / 63.50 |
| | wikitext | 64 | 46.03 [38–63] | 15.97 | 5.61 / 63.50 |

All health gates clean (0 NaN/Inf params, finite logits, 3/3 models); no window
duplication (repeat fallback never fired, 0 duplicate pairs, all cells).

**Reading (Tier 3, two findings, both registered-informative per §6.6):**

1. **H-measure: a strongly non-orthonormal write-geometry signature EXISTS in
   production fixed-state models — and it is far more extreme than our own 14M
   attractor.** RWKV-7's per-chunk erase keys at K=16/d=64 (the cell directly
   comparable to our own probe's geometry) sit at raw gd ≈ 10.8–11.0 — ≈5.6× the
   random anchor and ≈70% of the way to full collapse, vs our 14M band (0.6–4.4)
   which never exceeds random. Even after centering removes the shared
   massive-activation direction, RWKV-7 remains ≈2.5–2.9× random. Rising modestly
   with depth (L0 8.3 → L23 12.8, openr1). Falcon-Mamba's B_t is similarly aligned
   (≈3.2× its random anchor at its capacity-matched chunk=16), with a handful of
   notably-less-aligned outlier layers (L17: 5.5, L21: 6.5, L34: 7.7).
2. **The registered negative control shows the SAME-magnitude signature** (raw ≈
   11.7–12.3 at chunk=16 — ≈9× its d=128 random anchor; centered 4.6–5.9,
   overlapping RWKV-7's centered range). **Therefore this measurement cannot
   attribute the signature to the fixed-state/delta-rule write mechanism** — what it
   measures at production scale is dominated by the generic representation
   anisotropy of trained LMs (massive-activation channel + anisotropic key
   distributions), which is present regardless of whether a fixed-size state exists.
   Under §6.6's own framing this is exactly the calibration Q4 asked for, and it is
   decision-bearing: the honest Tier-3 statement is "the geometry geo3 targets
   (non-orthonormal write directions) is present and larger at 1.5–7B production
   scale, but it is NOT identifiable as a delta-rule-family attractor with this
   instrument — a discriminating instrument would need to control for generic key
   anisotropy (matched-architecture softmax controls per layer, token-content dedup,
   or write-path-specific interventions)."

**Phase 2 (graft) go/no-go input:** finding 2 weakens the mechanistic premise for a
geo3-style graft — the measured non-orthonormality at scale is not shown to be a
delta-rule-specific pathology, and §6.4's bolt-on-lesson risk stands unmitigated.
Nothing here authorizes Phase 2 (per §6.4 it needs its own attack round); this result
is evidence AGAINST prioritizing it.

**Caveats (probe-convention differences vs the 14M band, stated for any future
comparison):** (i) our band was measured on K *distinct-entity* episodes; Track D
chunks are arbitrary contiguous text — repeated tokens/subwords legitimately share
key directions, biasing deviation upward; (ii) RWKV-7's `kk` is the erase-side key;
the raw value-write key `k` is a different tensor, not measured separately; (iii)
Falcon-Mamba's d=16 makes orthonormality impossible past K=16 by construction — its
chunk=64 numbers are structurally inflated, the chunk=16 (capacity-matched) column is
the fair one; (iv) the Qwen control is pre-RoPE — post-RoPE keys would differ;
(v) `fla` prints its own "potentially buggy RWKV implementation" warning — the probe
reads `k_proj`/`k_k` directly so kernel-level bugs would not corrupt these
statistics, but the warning is on the record; (vi) §6.3 item 4's bootstrap
power/variance check was NOT run — per-layer means rest on 1.1K–16K scored episodes
per (layer, corpus, chunk) cell for the 24–64-layer models, large but not
bootstrap-verified; (vii) centering removes exactly ONE shared direction per episode
— higher-rank shared structure survives and still inflates `centered`.

---

## 7. Program-wide manifest — budget across all four tracks

| Track | Central estimate | Range | Authorized to run without further sign-off? |
|---|---|---|---|
| A — steps-to-criterion | 0 GPU-h | 0 | Yes — zero GPU, no gate |
| B — geo3-in-LM | ~18 GPU-h | 16–24 | Yes, pending its own Wave −1 calibration (§4.5) — whose criterion-(b) failure branch is a registered no-launch (§4.2) |
| C — scaling ladder (incl. required mix-control cell) | ~105 GPU-h | 66–152 | Wave −1 (calibration) yes; Waves 1–4 pending Wave −1's re-pricing (§5.6) |
| D — 7B measurement | ~7.5 GPU-h (Phase 1 only) | 5–10 | Phase 1 yes; Phase 2 (graft, ~15–30) explicitly NOT authorized (§6.4) |
| **Total (Phase-1-only D)** | **≈130 GPU-h** (130.5, sums reconciled Rev 2 — MINOR-4) | **87–186** | |
| **Total (if D's graft phase is later separately authorized)** | **≈153 GPU-h** | **102–216** | Graft phase requires its own attack round first |

Both totals land under the **300 GPU-h ceiling**, with a **≥84 GPU-h buffer** even in
the wide-case (top-of-range on every track) scenario, and a **≥147 GPU-h buffer** at the
central estimate — sized deliberately given Track C carries the least-validated cost
estimate in this document (§5.8 item 3).

---

## 8. Cut order

**Never cut:** Track A (already zero-GPU); Track B's primary arm + its Wave −1
calibration probe (carries the track's entire causal-adjacent claim, mirroring
`DELTANET_RD_EXACTNESS_DESIGN.md`'s own "B-probe is never cut" rule); Track C's rung 1
(cheapest, most informative, the calibration anchor for the other two rungs) **including
the required small-model same-mix control cell (Rev 2 — MAJOR-5; it costs <1 GPU-h and
carries the two-axis discipline for the whole ladder)**; Track D's
Phase 1 (cheap, no-regret, and its result is what determines whether Phase 2 is even
worth a future attack round).

**Cut, in this order, until back under budget:**

1. Track D's Phase 2 (graft-and-finetune) — highest-risk, least-validated, already
   gated behind its own unrun attack round; cutting it costs nothing this program has
   already committed to.
2. Track C's fix-effect sub-wave at rung 3 (Wave 4's rung-3 half) — keep rung 3's
   baseline geometry measurement (the core scaling question) but drop the geo3 transplant
   there first.
3. Track C's frontier-probe transplant at rung 2 — already scoped to rungs 1 and 3 only
   (§5.5); if further squeezed, this item is a reminder that rung 2 never had a transplant
   to begin with, not a new cut.
4. Track B's naive-fixed-window ablation arm (Wave 2) — keep only the primary β-gated
   arm and the eval-only truncation grid.
5. Track C rung 3's second corpus — drop to openr1-mix only (the corpus where the
   original truncation-sensitivity effect, `DELTANET_REALDATA_DESIGN.md` §19.3, was
   actually measured).
6. Track C rung 2 entirely (attractor-persistence-only, 2 runs) — the least informative
   single wave once rungs 1 and 3 exist, since the ladder's headline question (does the
   attractor persist) is answerable, if less finely resolved, from the two endpoints
   alone.

---

## 9. Calibration-first discipline (consolidated statement)

Every track above with any non-zero GPU cost has a blocking Wave −1 calibration step
before its main manifest is priced or launched — not a suggestion, a hard gate, per this
project's own repeatedly-relearned lesson (`STAGE0_DESIGN.md` §12,
`TASK_E_FINDINGS.md` §10's "three-budget-artifacts" finding, and this document's own
§5.6 admission that the single available throughput reference point is
non-compute-bound and therefore a weak extrapolation anchor). No track's Wave 1+ manifest
is authorized to launch at this document's estimated GPU-hour figures without first
recording its Wave −1 measured numbers in the wave's own summary file.

---

## 10. Attack-yourself (program-level, cross-cutting)

1. **Novelty check not yet performed for this specific combination.** Every prior
   program in this project's campaign ran a research/novelty pass before build
   (`CLAUDE.md`'s waterfall). This design has not yet had an explicit literature check
   for "does the write-geometry/rank-exactness attractor persist across model scale in
   delta-rule-family LMs" — the synthetic/causal-rank work's own novelty checks covered a
   narrower question. This must happen before Track C/D's build phase, not be assumed
   clear by inheritance.
2. **Track C is the single largest and least-validated cost driver** (§5.8 item 3) — if
   Wave −1 calibration reveals throughput far below this design's 25%-MFU assumption
   (e.g. 10% MFU), Track C alone could approach or exceed the entire program's 300 GPU-h
   ceiling on its own. The cut order (§8) is deliberately rung-3-second-to-cut, not
   rung-1-first, because rung 1 is by far the cheapest and most informative spend.
3. **The 300 GPU-h ceiling is a hard stop for this document, not a target.** This
   manifest is built to land with a ~50% margin under central estimates specifically so
   that one bad calibration surprise does not consume the entire authorized budget before
   a human gets to review real results and re-authorize further spend.
4. **(RESOLVED, Rev 2.)** The independent attack round required by `CLAUDE.md`'s
   standing rule has now run: NEEDS-REVISION, 5 MAJOR + 5 MINOR, no FATALs — all
   addressed in this revision (finding→change map: §13). Track B/C/D build may start
   on Rev 2; any further substantive design change (e.g., Track B's registered
   no-launch branch firing and triggering a redesign) re-enters the attack cycle,
   per house convention.

---

## 11. Sequencing — what this unlocks, and the one real dependency

Track A has no dependencies and should run immediately — it costs nothing and its
result (does geo3 show a real sample-efficiency edge, not just a ceiling edge) is
informative context for how much urgency to put behind Track B.

Track C's **attractor-persistence** readout (§5.5 item 1) has no dependency on Track B
and can run in parallel once its own Wave −1 calibration clears. Track C's
**fix-effect** readout (§5.5 item 3, Wave 4) is **hard-gated on Track B validating** —
it applies Track B's specific geo3-in-LM construction, so it cannot run before Track B's
own Wave 1 has a validated, non-fallback construction to transplant.

Track D's measurement phase has no dependency on A/B/C and can run in parallel on
different hardware/nodes if convenient, but its write-up is more informative once A/B/C's
own-project results exist to compare against — sequencing the *synthesis*, not the
*measurement*, after the other three tracks land is the recommended (not required)
order.

---

## 12. Open questions for the attack round (explicit, not resolved by fiat)

1. Is the β-gated top-K construction (§4.2) actually the best available adaptation of
   F-geo-3 to free text, or is there a more principled alternative (e.g. novelty/surprise
   of the key vector itself, rather than β) that an attack round should force onto the
   table?
2. Does §5.3's rung-config table survive contact with the harness's actual parameter
   count, or does the β-projection's true shape (possibly per-head) push the configs
   meaningfully off target?
3. Is a single held frontier-probe-transplant validation at rung 1 (§5.5 item 2) a
   strong enough gate for trusting the rung-3 transplant number, or does this need a
   second, independent validation method?
4. Should Track D's Phase 1 include a genuine negative control beyond RecurrentGemma
   (e.g., a standard softmax-attention Transformer of comparable scale, which has no
   fixed recurrent state at all) to calibrate what "no attractor" would even look like
   under this measurement protocol?
5. Is the 25%-MFU assumption in §5.6 defensible at all given the only measured reference
   point is non-compute-bound, or should Wave −1's calibration be widened (more rungs
   tested at short budget) before any Wave 1+ number is trusted?

*(Status note, Rev 2: the attack round spoke to Q1 — its MAJOR-2 forced the
excluded-position write-mass gate onto exactly the premise Q1 questioned, and the
registered no-launch branch is now the mechanism by which a better construction would
be forced onto the table. Q2–Q5 remain open for their respective Wave −1
calibrations.)*

---

## 13. Rev 2 finding→change map (attack round, 2026-07-04)

Independent adversarial review of Rev 1: **NEEDS-REVISION** (no FATALs), 5 MAJOR,
5 MINOR. Every finding verified against sources before the fix was applied — the
verification results are recorded inline where each fix landed.

| Finding | Severity | What was wrong | Fix | Where |
|---|---|---|---|---|
| MAJOR-1 | MAJOR | Rev 1's §1.1 "correction" was wrong on both its claims: "no document contains 45×" is false (`DELTANET_RD_EXACTNESS_DESIGN.md` §16's title contains it) and "K=32 was never admitted" was stale (the same-day `geo3_n_iter=20` escalation fixed admissibility 0/3 → 3/3 with zero fallback steps and near-identical behavioral numbers, §16.3–16.5; `EXPERIMENT_LOG.md` "F-geo-3 escalation VERDICT"; `STATE.md`). Rev 1 read only the earlier log entry and never saw the `geo3n20` files sitting in the same archive directory. | §1.1 rewritten from §16 as the authoritative source; §0's reading list repointed; §1 item (4) corrected; Track A §3.2 and §3.5 now carry the `geo3n20` cells as the PRIMARY admissible K=32 comparison (with a free `n12`-vs-`n20` trajectory-comparison bonus readout); §3.7 item 5 rewritten as a two-tier tagging rule; Track B §4.2's "near-admitted" framing replaced with "admissible, ~48×, narrowly missed its headline bar (0.4368 vs. ≥0.5 mean)" | §0, §1, §1.1, §3.2, §3.5, §3.7 |
| MAJOR-2 | MAJOR | The β-gated top-K construction's "closes failure mode 2" claim fails structurally: LM-mode β is a continuous, never-masked sigmoid (verified in `lm_pretrain_rd.py`'s own docstring and `b_proj` path), so the ~32–48 non-selected positions per chunk still write non-orthogonally into the shared state; Rev 1's Wave −1 probe (β-concentration only) could not catch this | Claim withdrawn explicitly in §4.2 item 2; a second, decision-bearing Wave −1 gate criterion registered NOW — (b) excluded-position write-mass: mean non-selected β ≤0.25 AND non-selected write-mass ≤40% of chunk total — with a hard no-launch branch (route to redesign) if it fails; the hard-zero-β variant noted as a conditional follow-on requiring its own attack pass; §4.6 failure list and §4.8 item 1 rewritten; Track B claims provisional until measured | §4.2, §4.5, §4.6, §4.8 |
| MAJOR-3 | MAJOR | Rev 1 conflated `chunk_size=64` (kernel sequence-tiling constant, `d_state`-independent) with `d_state=64` (the real Newton–Schulz capacity ceiling) — coincidentally equal at 13–14M, silently breaking at Track C's `d_state=128` rungs | §4.2 item 1 corrected: binding constraint is `K_sel ≤ d_state`, window is a tiling convenience; registered rule for `d_state=128` rungs added to §5.5 item 3 (window stays 64, `K_sel ≤ min(window, d_state) = 64`, headroom deliberately unexploited) | §4.2, §5.5 |
| MAJOR-4 | MAJOR | The brief's "438K tok/s/GPU" is untraceable in the repo; archive-derived measurement is ≈361K tok/s/GPU (359–365K over 6 Wave C runs, formula `steps × batch × seq_len / wall_s`), MFU ≈3.1%, at the LM harness's `n_params=14,048,896` (not the synthetic harness's 12.9M) | §5.6 rewritten with the measured number, the recomputation formula, the correct param count, and an instruction to recompute at build time rather than cite the rounding | §5.6 |
| MAJOR-5 | MAJOR | Track C bundled scale + corpus-mix changes at rungs 2–3 with the confound only disclosed — violating CLAUDE.md's hold-the-second-axis-fixed hard rule | Small-model same-mix control cell promoted to REQUIRED, folded into Wave 1 (costs <1 GPU-h at Wave C's measured ~4.6 min/run), excluded from the cut order; rung-2/3 headline claims scoped as joint scale+data-mix until the control reads out, registered in §5.7 | §5.6, §5.7, §5.8 item 6, §8 |
| MINOR-1 | MINOR | Archive count off by one (81 → 82 files; 117MB → 122MB, and the count must include the `geo3n20` files) | Recounted | §3.2 |
| MINOR-2 | MINOR | Track B's §4.5 total (16–23) didn't match its own rows' sum (16–24) | Reconciled | §4.5 |
| MINOR-3 | MINOR | The per-layer formula's `4·d_model·d_state` term was labeled "q/k/v/β"; the fourth term is `o_proj` (`d_state→d_model`); `b_proj` is `d_model→num_heads`, negligible. Arithmetic unaffected | Label corrected | §5.3 |
| MINOR-4 | MINOR | §7's central total (≈136) didn't match its own rows' sum (130.5) | Reconciled (≈130 / ≈153; ranges 87–186 / 102–216); header updated | header, §7 |
| MINOR-5 | MINOR | OpenWebMath/FineWeb-Edu licenses verified only in this design session; no build-time re-verification step | One-line re-verification item added to Track C's Wave −1 row | §5.6 |
| (pre-review fixes) | — | Two errors were caught by the designer's own QA pass during the review window and fixed before the attack report landed: a §7 budget-arithmetic error, and a Falcon-Mamba license misattribution (the $1M/yr royalty clause belongs to Falcon-180B; Falcon-Mamba-7B's TII License v1.0 is royalty-free — verified by direct license-page fetch) | Verified landed by the reviewer; excluded from the findings list | §6.2, §6.7, §7 |
