# Thought Interleaving Research (April 2026)

## Key Finding: Scale Matters Most
No paper demonstrates latent reasoning working below ~69M params. Our model is 288K.
We are 240-430× below the smallest known working scale.

## What Makes Thoughts Work vs Fail

### CoCoMix (Meta 2025): SAE supervision appears essential
- Interleaving WITHOUT concept prediction: "limited benefit"
- Both together: 2.4% PPL improvement at 69M, 2.8% at 1.38B
- The "21.5% sample efficiency" = need 21.5% less data, NOT 21.5% better PPL
- Inserts at 50% depth (layer 4 of 8)

### Pause Tokens (ICLR 2024): Task and scale dependent
- 1B model: SQuAD +18 points, HellaSwag -0.1 (hurt)
- 130M model: SQuAD gains vanish
- Optimal number of pauses is task-dependent (10 for GSM8k, 50 for SQuAD)
- During pretraining: pauses at only 10% of positions (not 100% like us)

### ThoughtBubbles (Stanford 2025): Score-weighted masking is the key
- Works from scratch with only LM loss
- Score-weighted attention: useful forks get strong gradients, useless get pruned
- 7% PPL improvement at 772M
- Budget: ~N=1 fork per position (matches our N=1)

## Optimal Configuration
- **N=1 or N=2** (COCONUT sees diminishing returns at c=3)
- **Insert at 50% depth** (layer 12 of 24, not layer 8)
- **MultiProbeHead output** (proven to drive rank enrichment)
- **Score-weighted masking** (prevents thoughts from being ignored)

## The Depth Delusion (Jan 2026)
- Critical depth at W=16: ~7 layers
- Our 24 layers is 3.6× over. 48 layers is 7.2× over.
- Gradient starvation kills early layers at extreme depth-to-width ratios
- Width should grow 2.8× faster than depth

## Predictions for Our Sweep
1. Config A (MultiProbeHead N=1): MOST PROMISING — matches our proven findings
2. Config B (N=2): Second best — COCONUT sweet spot
3. Config E (48-layer baseline): Useful control but likely past optimal depth
4. Config D (48 layers + thoughts): Gradient starvation + thoughts unlikely to help
5. Config C (N=4): Too aggressive, attention cost dominates

## What We Should Add Next
1. ThoughtBubbles-style score weighting (gradient highway for useful thoughts)
2. Move insert_layer to 50% depth
3. Scale up to 10M+ params (closer to known working scale)
