# Publishability Assessment

Research from agent on 2026-03-24. Honest critical review.

## Current Status: MARGINAL — needs more experiments to be publishable

## Findings Ranked by Novelty

| Finding | Novelty | Evidence | Current Venue | With More Work |
|---------|---------|----------|---------------|----------------|
| PHM → nilpotent (first report) | **HIGH** | Medium | Workshop | Main conference |
| Emergent domain differentiation | Med-High | **WEAK** | Workshop | ICLR/NeurIPS |
| Algebraic reg preserves structure | **HIGH** | Strong | Workshop | Main conference |
| Nilpotent beats quaternion | **HIGH** | Strong | Combined paper | Combined paper |
| Forced segmentation helps | Low | Medium | Supporting | Supporting |

## CRITICAL Missing Experiments (paper will be rejected without these)

1. **Fixed-stride vs learned segmentation** — Is the BPC improvement from LEARNING
   boundaries or just from shorter sequences? This is the #1 priority.

2. **Random/untrained segmenter control** — Does domain differentiation come from
   learning or from raw byte statistics? Must rule out null hypothesis.

3. **PHM vs standard linear vs plain low-rank at matched params** — Does PHM
   structure help at all, or would any factorization work?

4. **Boundary position visualization** — Where do boundaries land? Word boundaries?
   This is the compelling evidence for finding 2.

5. **Scale to 50-100M+ params** — 10.8M is toy scale. Reviewers will dismiss.

## Strongest Paper Framing

"What Do Learned Segmenters and Structured Layers Actually Learn?"
- An empirical analysis paper, not a new SOTA system
- Contribution 1: Emergent domain differentiation in learned segmentation
- Contribution 2: First analysis of PHM learned factors (→ nilpotent)
- Contribution 3: Algebraic regularization technique + finding that nilpotent beats quaternion
- Contribution 4: Architectural boundary enforcement reliably prevents collapse

## Path to Publication

**Workshop (1-2 weeks):** Fixed-stride ablation + random control + boundary viz
**Short paper (2-3 weeks):** Above + PHM vs low-rank + audio domain
**Main conference (4-6 weeks):** Above + scale to 50M+ + comparison to BLT/MBLM baselines
