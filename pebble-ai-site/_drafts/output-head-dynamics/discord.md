---
target: research-oriented Discord servers (#interpretability, #papers)
canonical: https://pebbleml.com/findings/output-head-dynamics.html
status: ready_for_review
note: Optional. The rank-enrichment Discord paste already covers most of this content. Use this if a separate channel asks for follow-up after the rank-enrichment share.
---

# Long form (only if conversation called for follow-up)

follow-on to the rank-enrichment finding I shared earlier:

across three configurations sharing the Matrix Thinker backbone, the sign of the rank trajectory tracks the output mechanism rather than the backbone.

· MultiProbeHead (32 bilinear probes): 5.05 → 6.13 monotonic rise. T=8 BPB 1.67.
· vector-collapse head: drops (direction only logged for that run).
· 3D matrix-product attention: 2.75 → 2.66 drop. T=8 BPB 2.46, 29% worse than the Frobenius-attn baseline at the same config.

caveats: single-seed each. cross-run comparison is unmatched on training corpus, step count, attention mechanism. within-run trajectory direction is the cleaner signal. no causal rank-projection ablation yet — that's the experiment that would convert this from observational to mechanistic.

main implication for interp work: rank-as-a-depth-monitoring-tool reads partly the head, not just the backbone. comparisons across heads may be reading different things.

paper: https://pebbleml.com/findings/output-head-dynamics.html

# Skip Reddit for this finding

The Reddit angle largely overlaps with the rank-enrichment Reddit post. Posting both within 2 weeks risks redundancy + cross-post fatigue. Recommend: skip a separate Reddit post for output-head-dynamics. The rank-enrichment Reddit comments thread is the right place to drop a "follow-on note" reply linking here, after that thread has cooled.
