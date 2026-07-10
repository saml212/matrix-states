# 3. The Geometric Readout Never Fires

**Zero-shot, marker template (wave 1).** The full pre-registered grid, 60
intervention-grid cells (three arms, two corpora, three seeds, two loads,
plus blend-off surgery variants for the two intervention arms) and 18
ladder cells (three rungs, two corpora, three seeds), returns a recovered
fraction of exactly 0.0 at every one of the 312 (cell, hop) readings
<!-- evidence: C1-phase1 -->. The registered validity gates fail everywhere:
the h=1 sanity floor fails both its null-relative and absolute conditions,
and premises (iii) and (iv) pass at 0 of 78 cells each
<!-- evidence: C1-phase1 -->. Figure 1 shows the failure structure: premise
medians track their shuffled nulls closely (premise (iii) medians span
[-0.33, 0.76] across cells <!-- evidence: C1-phase1 -->) but never cross
them. The mean cosine between prediction and target stays centered near
zero at every cell, so a 0.9 threshold is more than ten standard deviations
away; the registered routing therefore reports probe-invalid, and no
refutation of the geometry hypothesis is licensed.

**Scale (wave 2).** The 1.31B rung, run against the ladder's final saved
checkpoint, reproduces the failure in kind: 8 of 8 readings at 0.0 and both
premise gates failing at both corpora <!-- evidence: C2-rung3 -->. The
ladder series is 80 of 80 zero readings across 14M, 98M, 392M, and 1.31B
<!-- evidence: C2-rung3 -->. Whatever prevents this readout from firing is
not a small-model artifact.

**Zero-shot, natural language (wave 3).** Two natural-language template
families, a gift-verb family with the out-of-distribution query marker
dropped entirely and a succession-verb family an order of magnitude more
frequent in the encyclopedic corpus, fail identically: 16 of 16 readings at
0.0 and the h=1 validity gate false at all 4 gate cells
<!-- evidence: C3-phase1b -->. Four structurally different cells failing the
same way is the most informative null this wave could produce: the failure
is not attributable to prompt surface form.

**Task-familiarized (wave 4).** The strongest-case test trains control-arm
checkpoints on the bind/query task itself for 5,000 steps, then applies the
same gate at five checkpoints per cell. The gate fails at 30 of 30
(cell, checkpoint) readings, with 0 of 512 queries recovered at every
reading <!-- evidence: C4-dissoc -->. Meanwhile the vocabulary-space query
loss falls by 21.8 to 46.4 percent (mean 35.9 percent) in 6 of 6 cells
<!-- evidence: C4-dissoc -->: the models measurably learn the task in
vocabulary space while the geometric readout stays at floor (Figure 2).
This dissociation was built into the design, since the training objective
and the gate deliberately share no machinery; it indicates the readout
construct itself, not the absence of task learning, is what fails. The
fall stops short of a registered 50-percent pin in all six cells, so the
registered adjudication reports partial task learning rather than the
stronger claim.

**Gates with enforcing code paths.** One process finding is reported at the
same evidentiary level as the results. Wave 1's launch chain computed its
own abort gate but contained no code path that acted on it, and the grid
ran to completion, spending roughly 0.29 GPU-hours past a gate that had
already declared the probe invalid <!-- evidence: C8-teeth -->. Every
subsequent chain enforced its gates at the process boundary; waves 3 and 4
each refused a full-grid launch mechanically on a real failing gate,
writing a refusal sentinel and exiting <!-- evidence: C8-teeth -->. A
registered gate is a claim about behavior; only an enforcing code path
makes it one about the system.

**Table 1** summarizes the six waves.
