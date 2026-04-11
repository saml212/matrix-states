# "The Structure Learning Tax" — Research Summary

## The thesis (novel in its strong form):
Learned representational structure degenerates at small-to-medium scale.
Imposed structure wins. Scale may resolve this — BLT works at 8B params.

## Key framing:
"The Structure Learning Tax: Why Imposed Inductive Biases Outperform
Learned Representational Structure Below Critical Scale"

## The thesis is NOVEL because:
- Nobody has argued this across algebra + segmentation + geometry simultaneously
- Geometric Deep Learning (Bronstein 2021) argues imposed = efficient, not imposed = necessary
- The Bitter Lesson (Sutton 2019) argues the OPPOSITE — computation beats structure
- Our evidence spans 3 domains (PHM nilpotent, seg collapse, rank dynamics)

## Would be CONTROVERSIAL because it goes against:
- The bitter lesson (dominant narrative)
- End-to-end learning trend
- BLT's results at scale
- "More learning = better" sentiment

## The critical weakness:
SCALE. BLT works at 8B. Our results are at 2M-13M. The thesis may only
hold below a critical scale threshold. Frame this as a FEATURE not a bug:
"There exists a phase transition in scale for structure learning."

## Key citations to engage:
1. Bronstein et al. 2021 — Geometric Deep Learning
2. Sutton 2019 — The Bitter Lesson (counter-argument)
3. Battaglia et al. 2018 — Relational inductive biases
4. BLT (Meta 2024) — Scale-dependent counter-example
5. Zela et al. 2020 — DARTS degeneration
