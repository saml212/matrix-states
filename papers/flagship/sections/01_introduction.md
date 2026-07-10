# 1 Introduction

Linear-attention and fast-weight sequence models replace the growing
key-value cache of a transformer with a fixed-size matrix state,
written by an outer-product rule and read by a matrix-vector product.
The delta-rule family (DeltaNet and its gated descendants) is the
current standard bearer: at each step the state is updated as
$S_t = S_{t-1}(I - \beta_t k_t k_t^{\top}) + \beta_t v_t k_t^{\top}$,
a rank-one correction that overwrites the value previously associated
with the incoming key. The literature evaluates these models almost
exclusively on one axis: quality per unit of compute or memory relative
to attention. The state itself is treated as an implementation detail.

We take the opposite view. A $d \times d$ state is not a cheaper cache;
it is a representational medium with its own capacity structure, its own
causal role, and its own failure modes, none of which appear on a
perplexity leaderboard. This paper measures all three, under
pre-registered verdict criteria, across three experimental programs that
share one object of study: what the matrix state stores.

Our thesis, scoped per leg to the architecture family that carries it:
matrix-valued state representations are a genuine representational
medium. In a matrix-state encoder family trained on group word problems,
stochastic gradient descent recruits exactly the task-minimal state rank,
and forcing one rank fewer destroys the task while restoring that rank
restores it. In a two-layer delta-rule model, the first layer's matrix
state causally carries an episodic-recall capability that
parameter-matched vector-state and compute-matched transformer baselines
lack at equal training budget, and the stored content is linearly legible
only after downstream nonlinear processing. And in delta-rule language
models, the same write mechanism drives a population-geometry pathology
that worsens monotonically across a two-decade parameter ladder and
survives the community's stock mitigation.

Concretely, we contribute:

1. **A rank law, causally closed** (Section 3). Across five permutation
   groups whose minimal faithful representation dimensions $d_{\min}$
   span 2 to 5, recruited effective rank tracks $d_{\min}$ at Spearman
   $\rho = 0.9747$ <!-- evidence: R1 -->, the maximum the family's tie
   structure permits, with 19 of 19 unconstrained cells inside the
   pre-registered band <!-- evidence: R1 -->. A designed pair, $S_4$
   versus $A_5$, shares $d_{\min}=3$ but differs in solvability; an
   equivalence test declares their recruited ranks equivalent
   <!-- evidence: R2 -->, so rank follows dimension, not solvability.
   Train-time rank forcing closes the causal loop: recovery is 0.000 at
   $k = d_{\min}-1$ in all five groups and in all four seeds of the
   extension group, and returns at $k = d_{\min}$ in five of five
   <!-- evidence: R3 -->.
2. **A capability separation at matched budget** (Section 4). On
   single-pass episodic recall, the delta-rule contender reads accuracy
   0.99951 <!-- evidence: R4 --> against 0.03394 for its vector-state
   ablation and 0.02832 for a compute-matched transformer
   <!-- evidence: R4 -->; paired confidence intervals exclude the
   pre-registered 0.30 margin at more than three times its width
   <!-- evidence: R4 -->. Zeroing the first layer's state collapses
   recall to chance while zeroing the second layer's changes nothing
   <!-- evidence: R5 -->, and no state-level linear probe reads the
   content that the model's own forward pass decodes
   <!-- evidence: R5 -->. The capability is also memory-stable: accuracy
   0.998 or higher out to 1798 tokens on a fixed 32,768-byte state,
   while a transformer whose cache is capped at 1 to 32 times that
   budget reads chance everywhere <!-- evidence: R10 -->.
3. **A pathology that scale makes worse and the stock fix does not
   remove** (Section 5). The write keys of delta-rule language models
   drift toward a collapsed population geometry; the span fraction of
   that drift climbs 0.248 to 0.455 across a 14M-to-1.31B ladder at
   held-fixed data mixes <!-- evidence: R6 -->. Removing qk
   normalization changes nothing (0.05$\sigma$ at $n{=}3$)
   <!-- evidence: R7 -->, so the attractor is not an artifact of the
   standard stabilizer, and a frozen-key-bias mitigation that is
   loss-neutral at every scale fails to transfer its 14M geometric
   benefit to 98M or 392M <!-- evidence: R8 -->.
4. **A mechanism scaffold** (Appendix A). In the encoder family the
   trained state decomposes as $Z \approx c^{*} I + $ a rank-$(K{-}1)$
   task correction at 0.5 to 2.9 percent residual, and deviation from
   this scaffold is a loss-blind health signal <!-- evidence: R9 -->;
   the corresponding channel is empty by construction in delta-rule
   states, which bounds how far the scaffold generalizes.

Every numerical claim in this paper traces to a pre-registered verdict
record and an archived, checksummed raw artifact; the figures regenerate
from those artifacts under checksum assertion. Where an instrument broke
during the program, the paper reports the break, the diagnosis, and the
repaired reading rather than the broken one; Section 2 summarizes this
discipline and Section 7 discusses its scope limits.
