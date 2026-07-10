# 1. Introduction

Linear-attention and delta-rule language models compress an entire context
into a fixed-size matrix state [Schlag et al. 2021; Yang et al. 2024]. What
that state can hold, and how its contents are arranged, is measurable: prior
work has characterized recall capacity as a function of state size
[Arora et al. 2023; Arora et al. 2024], and a companion research program
measured a family of geometric observables of the written state (the span
fraction occupied by key writes, the effective rank of the key population)
across scales and under targeted interventions. Those geometric measurements
carried an unexamined promise: that the geometry of the write matters for
something a user would recognize, such as composing multi-hop relations
presented in context. This paper reports the program that tested the promise.

The test produced a null. Nulls of this kind are rarely published, and when
they are, a reader has two standing worries: that the instrument never worked
(so the null is vacuous), and that the analysis had freedom to declare a null
after seeing the data (so the null is unfalsifiable). We address both worries
structurally, by reporting a null with three pre-registered bounds, each of
which names the concrete observation that would have overturned it:

1. **An instrument bound.** A state-space composition readout, deployed three
   structurally different ways (zero-shot marker prompts, zero-shot natural
   language, and checkpoints trained on the probe task itself), returns a
   recovered fraction of exactly zero at 366 of 366 readings
   <!-- evidence: C1-phase1 --> on checkpoints from 14M to 1.31B parameters.
   The readout's own validity gates, registered before any data existed,
   fail at every cell; the registered routing therefore reports
   probe-invalid, never an unlicensed refutation of the underlying
   hypothesis. A single nonzero reading or a single gate pass would have
   voided this bound.
2. **A power bound.** A vocabulary-space behavioral contrast, whose
   three-seed detection floor of roughly 1.5 to 1.7 loss units was derived
   and registered before the data were unblinded
   <!-- evidence: C5-power -->, leaves three of four (corpus, arm) contrasts
   unresolved, and its single determinate signal is a transient deviation in
   the harm direction that dissolves by the end of training
   <!-- evidence: C6-transient -->. A determinate effect above the floor
   would have voided this bound.
3. **A replication bound.** A targeted twelve-seed extension of that
   transient was stopped by its own pre-registered batch-effect gate, which
   detected a 4.47-fold variance mismatch between the archived and new seed
   cohorts against a pinned cutoff of 4.0 <!-- evidence: C7-replication -->.
   The new cohort's own interval spans zero. The registered routing reports
   the gate failure rather than a silently pooled verdict; a clean pooled
   interval excluding zero would have voided this bound.

The contribution is the bounded null itself, together with the measurement
discipline that kept it interpretable: validity gates with enforcing code
paths, mechanical outcome routing fixed before unblinding, and a replication
gate that was allowed to refuse. All experiments fit comfortably inside the
small-scale regime this venue studies; the complete program consumed
approximately 9.5 GPU-hours <!-- evidence: C9-compute -->.
