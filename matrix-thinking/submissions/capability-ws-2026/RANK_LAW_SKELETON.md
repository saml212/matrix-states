# DRAFT-SKELETON — Capability-Separation Workshop Paper: The Rank Law

**Status: DRAFT-SKELETON (2026-07-09). Venue and framing are PI
decisions — nothing here is venue-formatted or submitted.** This is the
pre-assembled summary of the capability-separation Stage-1 rank-law
trilogy, written so a 4pp workshop draft can be built from it without
re-deriving a single number. Every number is quoted from the verdict
records in `matrix-thinking/CAPABILITY_SEPARATION_DESIGN.md` §1.33
(sweep harvest: M1 CONFIRM + marquee DECLARE + D-AMB diagnosis), §1.36
(M3 fix-wave harvest: CAUSAL-CONFIRM), and §1.36a (S3 seed extension:
S3 CONFIRMED), and the matching `EXPERIMENT_LOG.md` 2026-07-09 harvest
entries. Archives: `experiment-runs/2026-07-09_capability_sweep_harvest/`,
`experiment-runs/2026-07-09_m3fix_harvest/`,
`experiment-runs/2026-07-09_m3fix_s3ext/` (all SSD-mirrored). Realized
campaign cost ≈5.11/30 GPU-h. Distinct arc from the three existing
sibling papers (`workshop-2026/` = capacity trilogy; `iclr-2027/` =
write-geometry attractor + F-geo-3; `neurips-ws-2026/` = Task D/E
rank-recruitment) — no scope overlap.

## The claim (one sentence)

Trained end-to-end on group-composition state tracking, a DeltaNet
fast-weight state recruits effective rank equal to the task group's
minimal faithful representation dimension `d_min` — and `d_min` rank is
causally NECESSARY (recovery is exactly 0.000 one rank below) and
SUFFICIENT (recovery returns at `d_min`) — with the recruited rank set
by representation DIMENSION, not by group solvability.

## Paragraph 1 — the correlation leg (M1, observational)

Across five groups spanning the solvable/non-solvable divide, the
unconstrained arm's restricted effective rank tracks `d_min` essentially
perfectly: S3 1.877±0.060 (`d_min`=2), S4 2.852±0.054 (3), A5
2.832±0.062 (3), S5 3.591±0.069 (4), A6 4.736±0.023 (5). All 19
unconstrained cells sit inside the pre-registered [0.7, 1.3]·`d_min`
band per-seed (19/19 in-band). Spearman ρ = **0.9747**, which is the
exact tie-capped MAXIMUM achievable for this design (S4/A5 share
`d_min`=3, capping ρ below 1 by construction; exact-null
P(ρ≥0.8) = 6.67%) — perfect family ordering, with the tied pair landing
together. The L≥2 robustness split is near-identical to the full-sample
read for every group (max |Δ| = 0.041), so the decisional metric is not
contaminated group-selectively. (Corroborating-only tier by
pre-registration; the causal weight is carried by Paragraph 3.)

## Paragraph 2 — the marquee equivalence (dimension, not solvability)

The designed head-to-head is S4 (solvable) vs A5 (non-solvable, the
smallest non-solvable group), matched at `d_min` = 3. If recruited rank
were set by solvability/complexity class, the pair should separate; if
by representation dimension, they should be equivalent. Welch TOST on
restricted effective rank (n=5 vs n=5, pre-registered margin ±0.5
rank-units): diff **+0.0194** rank-units, se 0.0368, df 7.83,
t1 = 13.06 / t2 = 14.12 vs t_crit = 1.865 → **DECLARE equivalence**,
decisively (both one-sided tests pass by ~7× the critical value; no n=7
escalation needed). The marquee pair lands TOGETHER on dimension, not
apart on solvability.

## Paragraph 3 — the causal razor (necessity and sufficiency)

Force-rank arms cap the state's effective rank at
k ∈ {`d_min`−1, `d_min`, `d_min`+1} against a rank-`d_min` target.
On the pre-registered decisional readout (C1: full-Q Procrustes
crosscheck, `crosscheck_recovered_frac_90`), recovery is a **step
function at `d_min`**, in all 5 groups:

| Group | `d_min` | anchor (unconstr.) | k=`d_min`−1 | k=`d_min` | k=`d_min`+1 |
|---|---|---|---|---|---|
| S3 | 2 | 0.550 | 0.000 | 0.450 (seed-mean 0.5625, see below) | 0.550 |
| S4 | 3 | 0.650 | 0.000 | **0.800** | 0.950 |
| A5 | 3 | 0.700 | 0.000 | **0.700** | 0.750 |
| S5 | 4 | 0.500 | 0.000 | **0.600** | 0.550 |
| A6 | 5 | 0.650 | 0.000 | **0.650** | 0.700 |

Necessity: k=`d_min`−1 reads **exactly 0.000 at all 5 groups** — and,
where seed-extended (S3, 4 independent seeds), **exactly 0.000 in every
one of the 4 seeds** — zero noise on the necessity leg. Sufficiency:
k=`d_min` recovers past the pre-registered 0.9×anchor bar at S4/A5/S5/A6
(0.800 vs bar 0.585; 0.700 vs 0.630; 0.600 vs 0.450; 0.650 vs 0.585),
and k=`d_min`+1 recovers everywhere including S3. S3's decisive cell
(0.450 vs its 0.495 bar) fell inside the pre-stated ±0.05 marginality
trigger and was routed to a 3-seed extension BEFORE being quoted:
seed-mean (all 4 seeds) = **0.5625 ≥ the fixed 0.495 bar** (per-seed
k=`d_min`: 0.450/0.550/0.600/0.650) → S3 CONFIRMED at seed-mean
(§1.36a). Overall verdict: **CAUSAL-CONFIRM, 5/5 groups, including both
marquee members** (§1.36 + §1.36a).

## Methods paragraph — instrument integrity (the honest-instrument story)

Three episodes carry the methodology narrative, each caught by the
design's own pre-registered machinery rather than by luck:

1. **The eye-padding tax, caught by harvest (D-AMB).** The first
   58-cell sweep's force-rank target was silently `rho ⊕ I₂`
   (`groups.py` padded the target with `eye(d_state)`), making a rank-k
   arm's best achievable direct cosine exactly √(k/d_state) — and
   37/39 force-rank cells sat within 0.07 of that ceiling (mean
   |Δ| = 0.028): the arms had trained TO their rank-constrained optimum
   and spent their rank on the constant identity block first, so no
   capped arm ever delivered effective rho-rank ≥ `d_min`. The harvest
   established this five independent ways and the sweep verdict was
   registered **INCONCLUSIVE — not spun** (§1.33); a zero-padded fix
   wave (~1.42 GPU-h) then purchased the causal leg properly (§1.36).

2. **The C1 metric pin, pre-registered and load-bearing.** The
   decisional readout was pinned to the full-Q Procrustes crosscheck
   BEFORE the fix-wave harvest, after an oracle injection proved the
   scale-only primary metric basis-brittle on flawless models. The pin
   mattered: at S4 k=4 the primary read mean_cos −0.019 while the
   crosscheck read 0.966. No conclusion reads the primary.

3. **The S3 marginality trigger, honored.** The pre-stated ±0.05
   trigger fired on S3's decisive cell (|Δ| = 0.045) and routed a seed
   extension before S3 could be quoted as a confirm group; the
   decisional bar was the FIXED pre-registered literal (0.495), not a
   self-referential recompute from the extension's own noisier anchors
   (disclosed: per-seed, k=`d_min` clears its own seed's 0.9×anchor bar
   in only 2/4 seeds — anchor noise on S3's known-noisiest profile is
   exactly why the fixed bar is decisional).

## Open items for the draft (PI decisions)

- Venue + framing (capability-first headline per the 2026-07-08 PI
  directive vs a methods-forward framing).
- Whether Stage 2 (compositional depth generalization, designed, gated
  on Stage 1's readout) folds in or stays a separate paper.
- Title, author block, style file — same PENDING-USER state as the
  sibling workshop papers.
