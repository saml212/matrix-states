# 7 Discussion and Limitations

**Scale scope of the capability legs.** The rank law and the recall
separation are small-model, synthetic-task results by design: the
encoder family trains in minutes and the head-to-head contender is a
14M-parameter model on a controlled binding task. The strength of that
regime is causal closure (train-time rank forcing, state zeroing,
frozen margins); its limit is external validity. Nothing here shows the
recall separation surviving at language-model scale or on natural
data, and the matched-budget caveat bounds every transformer
comparison: a differently scaled or differently trained transformer
requires its own matched re-run before any extrapolation. The specific
unresolved measurement is a learning-rate search for the transformer
arm on the recall task itself: the frozen protocol's three-point grid
was run on the language-modeling control task only, so the chance-level
recall reading has not been separated from an optimization failure. The
resolving experiment is a re-run of the transformer arm under the
identical frozen protocol at a grid of at least four learning rates
spanning $10^{-4}$ to $3 \times 10^{-3}$, three seeds, 20,000 matched
steps, with training curves reported. A tuned transformer that still
reads below the demonstration bar would strengthen the separation to a
two-baseline result; a tuned transformer that clears it would confine
the separation to the vector-state comparison, which is the comparison
the frozen registration designates as the verdict carrier in either
case.

**One stress point is a locate-only null.** At the disclosed
above-capacity stress load ($K/d = 0.75$), the completed three-arm
table reads chance in every arm: contender 0.0189, ablation 0.0195,
transformer 0.0218, against chance 0.0208 <!-- evidence: R11 -->. The
table is locate-only by pre-registration, not a verdict-grade cell; no
capability claim is made at this load, and it bounds the single-hop
recall separation to loads below this stress point at this training
budget.

**Scope of the pathology leg.** The ladder result is a geometry claim,
not (yet) a capability claim: span fraction worsens monotonically with
scale, but this paper does not tie the drift to a downstream
performance cost, and validation loss is neutral in every mitigation
cell measured <!-- evidence: R8 -->. The tie between the geometric
drift and any behavioral cost is the outstanding scientific question,
and the 392M token-budget confound (Section 5.3) caps what the
transfer wave can say across scales.

**Multi-hop composition is open.** The two-hop task is
verdict-INDETERMINATE, with partial recall in three of nine pooled
contender seeds, none generalizing to held-out hop depths (Section
4.5), and a separate compositional-depth program in this research line
is at the design-and-calibration stage with a pre-registered lens
question resolved but no model verdict; we claim neither. Depth
generalization is future work, stated as such.

**Instrument validity as a recurring risk.** Three times in this
program (Sections 2.5, 3.3, 4.3), the first instrument pointed at a
true phenomenon read false. In each case the repair was cheaper than
the conclusion it prevented: a zero-pad flag, a tap relocation, a
pre-registered crosscheck. Representational claims about matrix states
need analytic anchors, negative controls, and pre-registered
crosschecks, because both false negatives (a linear probe at the wrong
tap) and false positives (an identity block sold as recovery) arise
from instrument choice alone at this scale.
