# Track A — steps-to-criterion sample efficiency (zero GPU)

Generated: 2026-07-03T21:08:22.600260+00:00  
Repo commit at analysis time: `fc3ded1de4b9f14e2038bd5a26cb996c8cc845ec`  
Archive: `experiment-runs/2026-07-03_deltanet_rd_waves/exactness`  

**Resolution ceiling: checkpoints are spaced every 2,000 steps.** `interpolated_step` linearly interpolates between the two bracketing checkpoints and is explicitly an estimate, never a finer-grained measurement. `checkpoint_step` is the raw, non-interpolated first checkpoint at which the threshold is met -- the true resolution-honest number. Left-censored = already above threshold at step 2000 (no earlier data exists). Right-censored = never crosses within 20,000 steps.

Thresholds [0.5, 0.8] are pre-registered (SCALE_TRANSFER_DESIGN.md §3.4); threshold(s) [0.9] are ADDED for this run per its explicit instructions, not part of the original pre-registration.

**The '100-steps-vs-0.23' teaser remains untraceable (§3.3).** Everything below is NEW analysis, not verification of a prior number.

## Headline: geo3 (primary tier) vs baseline, h=1/h=2/h=4, all thresholds

| K | hop | threshold | prereg? | baseline steps (interp, mean) | geo3 steps (interp, mean) | ratio | note |
|---|---|---|---|---|---|---|---|
| 16 | h=1 | 0.5 | yes | <=2000 (left-censored, all seeds) | <=2000 (left-censored, all seeds) | n/a | both arms already at/above threshold at the first checkpoint (step<=2000) in every seed -- no speed advantage measurable at this resolution, ceiling-only comparison possible |
| 16 | h=1 | 0.8 | yes | <=2000 (left-censored, all seeds) | <=2000 (left-censored, all seeds) | n/a | both arms already at/above threshold at the first checkpoint (step<=2000) in every seed -- no speed advantage measurable at this resolution, ceiling-only comparison possible |
| 16 | h=1 | 0.9 | ADDED | <=2000 (left-censored, all seeds) | <=2000 (left-censored, all seeds) | n/a | both arms already at/above threshold at the first checkpoint (step<=2000) in every seed -- no speed advantage measurable at this resolution, ceiling-only comparison possible |
| 16 | h=2 | 0.5 | yes | <=2000 (left-censored, all seeds) | <=2000 (left-censored, all seeds) | n/a | both arms already at/above threshold at the first checkpoint (step<=2000) in every seed -- no speed advantage measurable at this resolution, ceiling-only comparison possible |
| 16 | h=2 | 0.8 | yes | mixed (1 crossed / 1 left-cens / 0 right-cens of 2) | <=2000 (left-censored, all seeds) | n/a | seeds disagree in censoring status within at least one arm (some crossed, some left/right-censored) -- no single honest step number exists at this n; see per_seed for the full breakdown rather than trusting a pooled ratio |
| 16 | h=2 | 0.9 | ADDED | mixed (1 crossed / 0 left-cens / 1 right-cens of 2) | <=2000 (left-censored, all seeds) | n/a | seeds disagree in censoring status within at least one arm (some crossed, some left/right-censored) -- no single honest step number exists at this n; see per_seed for the full breakdown rather than trusting a pooled ratio |
| 16 | h=4 | 0.5 | yes | >20000 (right-censored, all seeds) | <=2000 (left-censored, all seeds) | 10.0 | baseline never crosses within 20,000 steps in any seed; geo3 is already >= threshold by its first checkpoint (step<=2000, true crossing could be earlier). Speedup is AT LEAST 10.0x (both bounds conservative -- true ratio is likely larger) |
| 16 | h=4 | 0.8 | yes | >20000 (right-censored, all seeds) | 2453.0 | 8.2 | baseline never crosses within 20,000 steps in any seed; geo3 crosses at interpolated step 2453.0. Speedup is AT LEAST 8.2x (true baseline step is right-censored, ratio is a lower bound, not exact) |
| 16 | h=4 | 0.9 | ADDED | >20000 (right-censored, all seeds) | 3355.0 | 6.0 | baseline never crosses within 20,000 steps in any seed; geo3 crosses at interpolated step 3355.0. Speedup is AT LEAST 6.0x (true baseline step is right-censored, ratio is a lower bound, not exact) |
| 32 | h=1 | 0.5 | yes | <=2000 (left-censored, all seeds) | <=2000 (left-censored, all seeds) | n/a | both arms already at/above threshold at the first checkpoint (step<=2000) in every seed -- no speed advantage measurable at this resolution, ceiling-only comparison possible |
| 32 | h=1 | 0.8 | yes | >20000 (right-censored, all seeds) | <=2000 (left-censored, all seeds) | 10.0 | baseline never crosses within 20,000 steps in any seed; geo3 is already >= threshold by its first checkpoint (step<=2000, true crossing could be earlier). Speedup is AT LEAST 10.0x (both bounds conservative -- true ratio is likely larger) |
| 32 | h=1 | 0.9 | ADDED | >20000 (right-censored, all seeds) | <=2000 (left-censored, all seeds) | 10.0 | baseline never crosses within 20,000 steps in any seed; geo3 is already >= threshold by its first checkpoint (step<=2000, true crossing could be earlier). Speedup is AT LEAST 10.0x (both bounds conservative -- true ratio is likely larger) |
| 32 | h=2 | 0.5 | yes | >20000 (right-censored, all seeds) | <=2000 (left-censored, all seeds) | 10.0 | baseline never crosses within 20,000 steps in any seed; geo3 is already >= threshold by its first checkpoint (step<=2000, true crossing could be earlier). Speedup is AT LEAST 10.0x (both bounds conservative -- true ratio is likely larger) |
| 32 | h=2 | 0.8 | yes | >20000 (right-censored, all seeds) | <=2000 (left-censored, all seeds) | 10.0 | baseline never crosses within 20,000 steps in any seed; geo3 is already >= threshold by its first checkpoint (step<=2000, true crossing could be earlier). Speedup is AT LEAST 10.0x (both bounds conservative -- true ratio is likely larger) |
| 32 | h=2 | 0.9 | ADDED | >20000 (right-censored, all seeds) | <=2000 (left-censored, all seeds) | 10.0 | baseline never crosses within 20,000 steps in any seed; geo3 is already >= threshold by its first checkpoint (step<=2000, true crossing could be earlier). Speedup is AT LEAST 10.0x (both bounds conservative -- true ratio is likely larger) |
| 32 | h=4 | 0.5 | yes | >20000 (right-censored, all seeds) | mixed (1 crossed / 0 left-cens / 2 right-cens of 3) | n/a | seeds disagree in censoring status within at least one arm (some crossed, some left/right-censored) -- no single honest step number exists at this n; see per_seed for the full breakdown rather than trusting a pooled ratio |
| 32 | h=4 | 0.8 | yes | >20000 (right-censored, all seeds) | >20000 (right-censored, all seeds) | n/a | geo3 never crosses this threshold within 20,000 steps in any seed -- this is a ceiling failure for geo3 at this threshold, not a sample-efficiency question |
| 32 | h=4 | 0.9 | ADDED | >20000 (right-censored, all seeds) | >20000 (right-censored, all seeds) | n/a | geo3 never crosses this threshold within 20,000 steps in any seed -- this is a ceiling failure for geo3 at this threshold, not a sample-efficiency question |
| 48 | h=1 | 0.5 | yes | >20000 (right-censored, all seeds) | <=2000 (left-censored, all seeds) | 10.0 | baseline never crosses within 20,000 steps in any seed; geo3 is already >= threshold by its first checkpoint (step<=2000, true crossing could be earlier). Speedup is AT LEAST 10.0x (both bounds conservative -- true ratio is likely larger) |
| 48 | h=1 | 0.8 | yes | >20000 (right-censored, all seeds) | <=2000 (left-censored, all seeds) | 10.0 | baseline never crosses within 20,000 steps in any seed; geo3 is already >= threshold by its first checkpoint (step<=2000, true crossing could be earlier). Speedup is AT LEAST 10.0x (both bounds conservative -- true ratio is likely larger) |
| 48 | h=1 | 0.9 | ADDED | >20000 (right-censored, all seeds) | <=2000 (left-censored, all seeds) | 10.0 | baseline never crosses within 20,000 steps in any seed; geo3 is already >= threshold by its first checkpoint (step<=2000, true crossing could be earlier). Speedup is AT LEAST 10.0x (both bounds conservative -- true ratio is likely larger) |
| 48 | h=2 | 0.5 | yes | >20000 (right-censored, all seeds) | <=2000 (left-censored, all seeds) | 10.0 | baseline never crosses within 20,000 steps in any seed; geo3 is already >= threshold by its first checkpoint (step<=2000, true crossing could be earlier). Speedup is AT LEAST 10.0x (both bounds conservative -- true ratio is likely larger) |
| 48 | h=2 | 0.8 | yes | >20000 (right-censored, all seeds) | <=2000 (left-censored, all seeds) | 10.0 | baseline never crosses within 20,000 steps in any seed; geo3 is already >= threshold by its first checkpoint (step<=2000, true crossing could be earlier). Speedup is AT LEAST 10.0x (both bounds conservative -- true ratio is likely larger) |
| 48 | h=2 | 0.9 | ADDED | >20000 (right-censored, all seeds) | <=2000 (left-censored, all seeds) | 10.0 | baseline never crosses within 20,000 steps in any seed; geo3 is already >= threshold by its first checkpoint (step<=2000, true crossing could be earlier). Speedup is AT LEAST 10.0x (both bounds conservative -- true ratio is likely larger) |
| 48 | h=4 | 0.5 | yes | >20000 (right-censored, all seeds) | >20000 (right-censored, all seeds) | n/a | geo3 never crosses this threshold within 20,000 steps in any seed -- this is a ceiling failure for geo3 at this threshold, not a sample-efficiency question |
| 48 | h=4 | 0.8 | yes | >20000 (right-censored, all seeds) | >20000 (right-censored, all seeds) | n/a | geo3 never crosses this threshold within 20,000 steps in any seed -- this is a ceiling failure for geo3 at this threshold, not a sample-efficiency question |
| 48 | h=4 | 0.9 | ADDED | >20000 (right-censored, all seeds) | >20000 (right-censored, all seeds) | n/a | geo3 never crosses this threshold within 20,000 steps in any seed -- this is a ceiling failure for geo3 at this threshold, not a sample-efficiency question |

## Per-seed detail (headline hops, threshold=0.9)

- **learned (arm iii-beta), K=16, tier=baseline, M2_in_distribution:h1**: s0=left_censored, s1=left_censored
- **learned (arm iii-beta), K=16, tier=baseline, M2_in_distribution:h2**: s0=5843.5, s1=right_censored
- **learned (arm iii-beta), K=16, tier=baseline, M3_held_out:h4**: s0=right_censored, s1=right_censored
- **geo3 (geo3n12), K=16, tier=admissible-primary, M2_in_distribution:h1**: s0=left_censored, s1=left_censored, s2=left_censored
- **geo3 (geo3n12), K=16, tier=admissible-primary, M2_in_distribution:h2**: s0=left_censored, s1=left_censored, s2=left_censored
- **geo3 (geo3n12), K=16, tier=admissible-primary, M3_held_out:h4**: s0=3179.9, s1=3494.5, s2=3390.6
- **learned (arm iii-beta), K=32, tier=baseline, M2_in_distribution:h1**: s0=right_censored, s1=right_censored, s2=right_censored
- **learned (arm iii-beta), K=32, tier=baseline, M2_in_distribution:h2**: s0=right_censored, s1=right_censored, s2=right_censored
- **learned (arm iii-beta), K=32, tier=baseline, M3_held_out:h4**: s0=right_censored, s1=right_censored, s2=right_censored
- **geo3 (geo3n20), K=32, tier=admissible-primary, M2_in_distribution:h1**: s0=left_censored, s1=left_censored, s2=left_censored
- **geo3 (geo3n20), K=32, tier=admissible-primary, M2_in_distribution:h2**: s0=left_censored, s1=left_censored, s2=left_censored
- **geo3 (geo3n20), K=32, tier=admissible-primary, M3_held_out:h4**: s0=right_censored, s1=right_censored, s2=right_censored
- **geo3 (geo3n12), K=32, tier=descriptive-only, M2_in_distribution:h1**: s0=left_censored, s1=left_censored, s2=left_censored
- **geo3 (geo3n12), K=32, tier=descriptive-only, M2_in_distribution:h2**: s0=left_censored, s1=left_censored, s2=left_censored
- **geo3 (geo3n12), K=32, tier=descriptive-only, M3_held_out:h4**: s0=right_censored, s1=right_censored, s2=right_censored
- **arm i-strong (strong_pin=True), K=32, tier=baseline-variant, M2_in_distribution:h1**: s0=left_censored, s1=left_censored, s2=left_censored
- **arm i-strong (strong_pin=True), K=32, tier=baseline-variant, M2_in_distribution:h2**: s0=left_censored, s1=left_censored, s2=left_censored
- **arm i-strong (strong_pin=True), K=32, tier=baseline-variant, M3_held_out:h4**: s0=left_censored, s1=left_censored, s2=left_censored
- **learned (arm iii-beta equivalent, K=48 rider), K=48, tier=baseline, M2_in_distribution:h1**: s0=right_censored, s1=right_censored, s2=right_censored
- **learned (arm iii-beta equivalent, K=48 rider), K=48, tier=baseline, M2_in_distribution:h2**: s0=right_censored, s1=right_censored, s2=right_censored
- **learned (arm iii-beta equivalent, K=48 rider), K=48, tier=baseline, M3_held_out:h4**: s0=right_censored, s1=right_censored, s2=right_censored
- **geo3 (geo3n20), K=48, tier=non-admissible, M2_in_distribution:h1**: s0=left_censored, s1=left_censored, s2=left_censored
- **geo3 (geo3n20), K=48, tier=non-admissible, M2_in_distribution:h2**: s0=left_censored, s1=left_censored, s2=left_censored
- **geo3 (geo3n20), K=48, tier=non-admissible, M3_held_out:h4**: s0=right_censored, s1=right_censored, s2=right_censored

## n_iter=12 vs n_iter=20 trajectory-match bonus readout (K=32 geo3)

| hop | threshold | n12 mean step | n20 mean step | |delta| | within one checkpoint (2000 steps)? |
|---|---|---|---|---|---|
| h=1 | 0.5 | None | None | None | n/a (neither arm has a seed that crossed via interpolation at this cell -- both fully left- or right-censored, see per_seed) |
| h=1 | 0.8 | None | None | None | n/a (neither arm has a seed that crossed via interpolation at this cell -- both fully left- or right-censored, see per_seed) |
| h=1 | 0.9 | None | None | None | n/a (neither arm has a seed that crossed via interpolation at this cell -- both fully left- or right-censored, see per_seed) |
| h=2 | 0.5 | None | None | None | n/a (neither arm has a seed that crossed via interpolation at this cell -- both fully left- or right-censored, see per_seed) |
| h=2 | 0.8 | None | None | None | n/a (neither arm has a seed that crossed via interpolation at this cell -- both fully left- or right-censored, see per_seed) |
| h=2 | 0.9 | None | None | None | n/a (neither arm has a seed that crossed via interpolation at this cell -- both fully left- or right-censored, see per_seed) |
| h=4 | 0.5 | 16448.8 | 16358.7 | 90.1 | yes |
| h=4 | 0.8 | None | None | None | n/a (neither arm has a seed that crossed via interpolation at this cell -- both fully left- or right-censored, see per_seed) |
| h=4 | 0.9 | None | None | None | n/a (neither arm has a seed that crossed via interpolation at this cell -- both fully left- or right-censored, see per_seed) |

## Admissibility tags (two-tier K=32, bonus K=48)

- K=16 seed=0 tier=admissible-primary admissible=True (ns_converged_no_fallback=True, value_salvage_tier_pass=True) in_manifest=True
- K=16 seed=1 tier=admissible-primary admissible=True (ns_converged_no_fallback=True, value_salvage_tier_pass=True) in_manifest=True
- K=16 seed=2 tier=admissible-primary admissible=True (ns_converged_no_fallback=True, value_salvage_tier_pass=True) in_manifest=True
- K=32 seed=0 tier=admissible-primary admissible=True (ns_converged_no_fallback=True, value_salvage_tier_pass=True) in_manifest=True
- K=32 seed=1 tier=admissible-primary admissible=True (ns_converged_no_fallback=True, value_salvage_tier_pass=True) in_manifest=True
- K=32 seed=2 tier=admissible-primary admissible=True (ns_converged_no_fallback=True, value_salvage_tier_pass=True) in_manifest=True
- K=32 seed=0 tier=descriptive-only admissible=False (ns_converged_no_fallback=False, value_salvage_tier_pass=True) in_manifest=True
- K=32 seed=1 tier=descriptive-only admissible=False (ns_converged_no_fallback=False, value_salvage_tier_pass=True) in_manifest=True
- K=32 seed=2 tier=descriptive-only admissible=False (ns_converged_no_fallback=False, value_salvage_tier_pass=True) in_manifest=True
- K=48 seed=0 tier=non-admissible admissible=False (ns_converged_no_fallback=True, value_salvage_tier_pass=False) in_manifest=False
- K=48 seed=1 tier=non-admissible admissible=False (ns_converged_no_fallback=True, value_salvage_tier_pass=False) in_manifest=False
- K=48 seed=2 tier=non-admissible admissible=False (ns_converged_no_fallback=True, value_salvage_tier_pass=False) in_manifest=False
