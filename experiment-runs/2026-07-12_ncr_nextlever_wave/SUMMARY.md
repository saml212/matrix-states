# NCR next-lever probe wave — 2026-07-12

Design: `matrix-thinking/NCR_NEXT_LEVER_DESIGN.md` (a8e848d). Recorded:
`matrix-thinking/NOVEL_ARCH_WATERFALL.md` §11.4. Queue jobs 060-063
(Q1 4x), 066-073 (Probe A), 074-081 (Probe B) — 20/20 COMPLETED on
`youthful-indigo-turkey`. Jobs 064-065 (conditional 8x recon) never
deployed (§1.7 did not fire — see §11.4's NO-LAW verdict).

## Contents

- `budget4x/` — Q1: K=16, d=32, 4x steps (320,000), seeds 0-3
  (`results_earlyln_budget4x/` on the box). 4 cell JSONs + 4
  `.axis_c_lock.json` siblings.
- `dratio/` — Probe A: K=16 d=17 and K=24 d=25 (tight-spare, d=K+1),
  1x steps (80,000), seeds 0-3 each (`results_earlyln_dratio/`). 8 cell
  JSONs + 8 lock siblings.
- `annealshape/` — Probe B: K=16 d=32 and K=24 d=48, anneal_frac=0.75,
  80,000 steps, seeds 0-3 each (`results_earlyln_annealshape/`). 8 cell
  JSONs + 8 lock siblings.

20 cell JSONs + 20 axis_c_lock JSONs = 40 files, 5.7M.

## Verdicts (full detail in the registry §11.4)

- **Q1 (K=16, 4x budget): NO-LAW.** delta non-monotonic in budget for
  3/4 seeds (2x->4x got worse), and a 2x-CONVERGED seed's failure front
  regressed 29->13 at 4x. Per §1.7's pinned map: do not extrapolate,
  escalate. The conditional 8x recon (jobs 064-065) is MOOT — its only
  firing condition (LAW-HOLDS-CROSSING-IN-REACH) was not met, and
  separately the realized wave-cap headroom (6.39 GPU-h) falls short of
  the 6.60 needed even under a counterfactual LAW-HOLDS reading.
- **Probe A (d=K+1 tight-spare): CONFIRM at both K=16 and K=24.**
  Gate-1 rate 4/4 CONVERGED at both d=17 (K=16, vs 1/4 at d=32) and
  d=25 (K=24, vs 0/4 at d=48). Falsifies Story S1 (Mechanism-1 sign)
  and the pure absolute-K-cliff story at both K; confirms Story S2
  (spare-fraction/convention-confound). K=16@d17 also reaches strong
  far-depth recovery (0.80-0.99 in 2/4 seeds at h*=125) that no d=32
  cell at any budget ever reached; K=24@d25's far-depth is weak and
  highly seed-variable (fronts 21-189, sweep_min <=0.051 in all 4).
- **Probe B (anneal_frac=0.75): B-16 CONFIRMED (partial), B-24
  FALSIFIED.** B-16: Gate-1 1/4->2/4 CONVERGED, delta mean -38.5%, but
  failure front stays flat at 13 in all 4 seeds (no far-depth gain,
  unlike the 2x-budget cell). B-24: all gates indistinguishable from
  the frac=0.5 baseline (still 0/4 CONVERGED, front pinned at 21,
  delta shift within pre-existing seed noise).

## Verification

All 20 cells' `axis_c_lock_sha256` field matches its `.axis_c_lock.json`
sibling's own `lock_sha256` byte-for-byte (20/20). All 20:
`status=COMPLETED`, `blank_out/passed=True`,
`eval/reducer_signature/flagged=False`. `git_commit=UNKNOWN` on every
cell — pre-existing disclosed cosmetic box artifact (no `.git` on box,
registry `:1380`), not new to this wave.

## GPU-h ledger

Q1 4x: 6.6089. Probe A: 3.4388. Probe B: 3.5617. **Wave total: 13.6094
GPU-h** (nominal mandatory: 14.05, 96.9% realized).

See `md5_manifest.txt` for per-file checksums.
