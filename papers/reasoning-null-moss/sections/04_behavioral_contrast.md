# 4. The Behavioral Contrast Is Power-Bounded

Wave 5 promotes the vocabulary-space contrast to the primary instrument:
both intervention arms and the control arm are familiarized under identical
recipes (18 cells), and the per-checkpoint arm effect is the difference in
held-out-episode query loss, control minus arm, so that positive values
mean the intervention helps.

**The registered power floor.** Before unblinding, the detection floor at
three seeds was derived from the control cells' own between-seed spread:
a sigma proxy of 0.43 to 0.48 loss units yields a 95-percent interval
half-width of roughly 1.5 to 1.7 loss units for a two-arm contrast
<!-- evidence: C5-power -->. For calibration, the entire 5,000-step
familiarization effect is itself about 1.69 loss units
<!-- evidence: C5-power -->: at three seeds this instrument can detect only
an arm effect about as large as everything the model learns during the
whole phase. The registration states the consequence in advance: an
unresolved outcome is the single most likely result for any modest effect,
and must be reported as a measurement bound, not as evidence of no effect.

**Results.** Three of the four (corpus, arm) contrasts classify as
unresolved <!-- evidence: C5-power -->: no per-checkpoint interval that
excludes zero survives the registered differential condition, and the
terminal effects of both arms fail to clear noise. The fourth contrast
(encyclopedic corpus, per-token arm) classifies as transient: at checkpoint
2,500 the K=32 effect is determinate at -0.500 with interval
[-0.624, -0.376] while the K=20 contrast is not (mean -0.252, interval
[-0.920, +0.416]), and by checkpoint 5,000 the effect dissolves back to
indeterminate (mean -0.795, interval [-2.513, +0.923])
<!-- evidence: C6-transient --> (Figure 3, left). A held-out-hop secondary
readout fires at the same cell and checkpoint, in the same direction
<!-- evidence: C6-transient -->, consistent with one coherent mid-training
deviation rather than two independent statistical accidents.

**Sign discipline.** The sign is negative: the intervention arm's loss sits
*above* control, so the one determinate signal points in the harm
direction, and every determinate high-load reading anywhere in the primary
table is negative <!-- evidence: C6-transient -->. Two misreadings were
pre-registered as forbidden and are worth naming because each is tempting.
Reading the transient as "the intervention matters" ignores the outcome
classification: transient effects were registered in advance as
training-dynamics artifacts, not durable capability differences. Reading
the three unresolved contrasts as "the intervention does not matter"
ignores the power floor: effects up to an order of magnitude larger than
plausible ones would have been invisible. What the wave delivers is a
bound, an effect above 1.5 to 1.7 loss units in either direction is
excluded, plus one real, bounded, harm-direction transient.
