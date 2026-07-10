# 2026-07-10 fix-at-scale harvest — FROZEN_BIAS_LM_DESIGN.md §13, verdict §13.22

**VERDICT (pre-registered §13.6, mechanical): PARTIAL at both scales.**
The deployed per_token frozen-bias arm (λ=0.58) repeats its destabilizing
14M sign at 98M (both corpora) and 392M (wikitext; openr1 null), reversing
nowhere; the global-vector probe's 14M stabilization does not transfer
(−0.058/−0.034 at 98M; −0.012/+0.019 sign-flip at 392M; n=1 exploratory).
Val-loss neutrality passes all 8 gates. **No tested frozen-bias
construction stabilizes the write-geometry attractor at scale.**

## Contents (all pulled 2026-07-10 from youthful-indigo-turkey, md5 local==box)

- `train/` — 24 primary training result JSONs + logs + rate-watch CSVs
  (arm_off seeds 1-2, arm_per_token seeds 0-2, arm_global_probe seed 0;
  2 scales × 2 corpora), plus the §13.21 superseded partial + ABORTED
  marker for the resumed 98m cell (provenance).
- `calib/` — the 4 gate-tier arm_off seed-0 cells (reused by the manifest
  as its arm_off/s0 cells, per §13.16's out_path design).
- `pins/` — `BANDS_PINNED-FrozenBias-{98M,392M}.json`: arm_off's own
  val-loss tolerance + arm1prime (off′ per-token retrofit) / arm1double
  (off″ global retrofit) / arm1_kraw (pre-blend) span_frac bands, written
  blind (pin timestamps strictly precede first post_pin launches: +2 s at
  98M, ~3 min at 392M). verify-pin tamper re-check at harvest: VALID.
- `pilots/` — §13.10 gate-3 timing/VRAM pilots (timing tier only, excluded
  from every analysis).
- `measure/` — the 16 harvest-run shared-forward-pass comparator outputs
  (12 per_token + 4 probe final checkpoints, each on its own corpus) + 4
  probe-report JSONs. Driver: `fixscale_harvest_measure.sh` (this dir);
  log: `measure_harvest.log`.
- `analyze_fixscale_harvest.py` → `fixscale_harvest_verdict.json`
  (md5 f2f0aae84908c0db0a42b13c76a85158) — the §13.6 verdict recomputed
  directly from raws (pins + comparators + train JSONs).
- `md5_manifest.txt` — full local manifest, verified against box md5sums.

## Headline table (Δ = arm mean − arm_off-retrofit mean from this scale's own pin; CI = ±t(2,.975)·s_ref/√3)

| Scale | Corpus | PRIMARY (post-blend) | CO-PRIMARY (pre-blend k_raw) | Reading |
|---|---|---|---|---|
| 98M | openr1 | +0.1133 [+0.0543,+0.1723] | +0.0796 [+0.0411,+0.1180] | destabilizing |
| 98M | wikitext | +0.1011 [+0.0541,+0.1482] | +0.0606 [+0.0119,+0.1094] | destabilizing |
| 392M | openr1 | +0.0065 [−0.0356,+0.0486] | +0.0037 [−0.0406,+0.0480] | null |
| 392M | wikitext | +0.0189 [+0.0112,+0.0266] | +0.0140 [+0.0053,+0.0228] | destabilizing |

14M reference (rung-1): +0.1955/+0.2273. Cross-scale attenuation is
descriptive only — 392M ran the reduced 20,000-step budget (token-
confounded vs 98M's 67,547; §13.11 item 8).

Realized cost ≈130.2 GPU-h vs the 281.04 committed (2×) / 300 cap.
Checkpoints (~TB) remain on the box (`/data/fixscale_ckpts/`) — not
archived here (>25MB rule).
