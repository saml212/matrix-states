---
target: x.com
canonical: https://pebbleml.com/findings/output-head-dynamics.html
status: ready_for_review
constraints:
  - 4 tweets. This is a tighter follow-on to the rank-enrichment X thread.
  - Post 5-7 days after the rank-enrichment thread so timelines see them as a series.
  - Quote-RT or reply with link to the rank-enrichment thread to re-up engagement.
---

# Tweet 1 (hook)

following up on the rank-enrichment thread —

across THREE matrix-valued transformer configurations sharing the same backbone, the SIGN of the rank trajectory tracks the output mechanism, not the backbone.

same matrix-thinking layers, same data, same compute. opposite internal dynamics.

🧵

# Tweet 2 (the comparison)

· Frobenius attn + MultiProbeHead (32 bilinear probes): rank rises 5.05 → 6.13. T=8 BPB 1.67.
· Frobenius attn + vector-collapse head: rank drops.
· 3D matrix-product attention: rank drops 2.75 → 2.66. T=8 BPB 2.46 (29% worse than the Frobenius-attn baseline at the same config).

# Tweet 3 (the implication)

mechanistic-interpretability work that uses rank as a depth-monitoring tool reads partly the backbone and partly the head it's attached to.

if the head can flip the SIGN of rank trajectories, "depth drives collapse" is a statement about a training incentive, not an architectural inevitability.

# Tweet 4 (link)

honest caveats: single-seed runs, cross-run comparison unmatched on corpus + step count + attention mechanism, no causal rank-projection ablation yet.

within-run trajectory direction is the cleaner signal.

full note: https://pebbleml.com/findings/output-head-dynamics.html
