# 5. The Replication Gate Refuses the Pool

A transient with n=3 evidentiary weight invites one obvious follow-up:
more seeds. Wave 6 is that follow-up, run as a targeted extension of the
single anchor cell (encyclopedic corpus, per-token arm, K=32, checkpoint
2,500) rather than a blanket rerun, with three integrity mechanisms
registered before launch. First, nine fresh pretraining seeds were trained
from scratch for both arms (18 new cells), since the three archived seeds
exhaust the existing checkpoint pool. Second, an archived-values loader
guarantees the original three seeds are never re-scored live; a runtime
guard turns any such call into a crash, and the archived deltas appear
byte-identical in the new harvest <!-- evidence: C7-replication -->. Third,
a batch-effect gate must pass before the two cohorts may be pooled: the
cohort means must agree within twice the pooled standard error, and the
cohort variance ratio must stay below a pinned 4.0.

**The gate fired.** At the anchor cell the archived cohort's between-seed
standard deviation (1.154) is 2.1 times the new cohort's (0.546), a
variance ratio of 4.47, twelve percent past the pinned cutoff
<!-- evidence: C7-replication -->; the means agree comfortably (shift
0.364 against a 1.382 threshold). The registered routing for a gate
failure is explicit: report both cohorts separately, produce no
decision-grade pooled interval, and classify the wave outcome as
batch-effect-flagged rather than confirmed or refuted. Figure 3 (right)
shows the picture. The archived cohort reproduces its original interval,
[-0.624, -0.376], by construction. The new nine-seed cohort's own interval
is [-0.506, +0.357], spanning zero <!-- evidence: C7-replication -->. The
naive twelve-seed pool, shown only as a diagnostic, would read
[-0.509, +0.147], also spanning zero, and would have classified the
transient as refuted <!-- evidence: C7-replication -->; the gate exists
precisely because pooling a high-variance three-seed cohort with a
lower-variance nine-seed cohort without a variance check assumes exactly
what a replication is supposed to test.

**What stands.** The transient is neither confirmed nor refuted. It stands
at its original three-seed weight, no more, and the direction of the
evidence is not spun: across the full extension tables, 7 of 10 primary
and 7 of 10 held-out (checkpoint, K) pairs are computable, and every
computable pooled interval contains zero <!-- evidence: C7-replication -->.
The variance mismatch itself is not established as a systematic
between-wave shift; its direction flips across checkpoints, which is most
consistent with ordinary small-n variance imprecision in the archived
cohort compounding with per-checkpoint training stochasticity. The
methodological point survives independently of the transient's fate: a
replication wave whose own integrity gate can refuse produces a
well-defined result even when it cannot produce a verdict.
