# 2. Models, Task, and Instruments

**Checkpoints.** All experiments probe DeltaNet-family language models
[Yang et al. 2024] pretrained on two corpus mixes: a reasoning-dense mix
built on OpenR1-Math and an encyclopedic mix built on WikiText-103
[Merity et al. 2017]. Two grids share these checkpoints. The *intervention
grid* holds scale fixed (14M parameters) and varies a frozen-key-bias
intervention on the fast-weight write path in three arms: a control arm, a
per-token variant, and a global variant, matched on validation loss across
arms. The *scale ladder* holds the architecture family fixed and spans four
rungs, 14M, 98M, 392M, and 1.31B parameters (fast-weight state dimension 64,
64, 128, 128; depth 2, 12, 16, 22) <!-- evidence: C0-ladder -->. The
motivating observable, the span fraction of the state occupied by key
writes, moves monotonically along this ladder in the companion measurements;
the question here is whether anything behavioral moves with it.

**Task.** Each probe episode binds K entities into a single directed cycle
with BIND statements ("A points to B"), then queries the entity h hops
around the cycle from a start entity, for h from 1 to 4. A full K-cycle,
rather than a random permutation, prevents held-out hop depths from
collapsing modulo short cycle lengths. Load is set by K relative to the
state dimension: K in {20, 32} on the intervention grid and near-capacity
K values (32 or 64) per rung on the ladder, placed at and below a
separately located capacity operating point.

**Instrument 1: a geometric readout.** For a layer's terminal fast-weight
state S_T and an effective query vector q_eff, the readout predicts the
h-hop target as S_T applied h times to q_eff and scores exact continuous
recovery: the fraction of queries whose prediction reaches cosine at least
0.9 with the target value vector. Exact continuous recovery, rather than
argmax over candidates, follows the associative-memory analysis of
Nichani et al. [2025], under which argmax decoding can silently succeed
without the state carrying the claimed structure. Registered with the
readout, before any data, were validity gates: an h=1 sanity floor (recovery
of one-hop bindings must exceed a shuffled-pairing null and an absolute
floor of 0.10) and two alignment premises, (iii) a query projection of an
entity must align with the same entity's key projection above a
shuffled-entity null, and (iv) key and value projections of the same token
must align above the same null. A registered routing rule fixes the
interpretation: gate failure routes the outcome to probe-invalid, never to
refutation.

**Instrument 2: a behavioral contrast.** A familiarization phase trains
checkpoints briefly (5,000 steps) on the bind/query task mixed into the
pretraining corpus, and the contrast measures the in-context query loss
L_q (a vocabulary-space cross-entropy through the language-model head,
deliberately different machinery from Instrument 1) on held-out episodes,
as a difference between control and intervention arms at five checkpoints.
A six-outcome trajectory classification (persistent, transient,
late-emergent, converged-equivalent, unresolved, and a
familiarization-null exclusion) was fixed before unblinding, together with
a differential condition requiring the effect at high load (K=32) to be
determinate while the low-load contrast (K=20) is not.

**Discipline and cost.** Every wave ran gate-first: self-tests, then
validity gates, then the grid, with verdicts routed by the registered rules
rather than by post-hoc judgment. The registrations, gate definitions, and
routing rules are reproduced in the supplementary material. The complete
program, all six waves, consumed approximately 9.5 GPU-hours on at most two
GPUs at a time <!-- evidence: C9-compute -->, within the small-scale regime
this venue studies.
