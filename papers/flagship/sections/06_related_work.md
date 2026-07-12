# 6 Related Work

**Associative capacity of fast-weight states.** Nichani, Lee, and
Bietti (2024) prove that under argmax decoding a rank-one matrix can
recover on the order of $d$ associations, which is why no result in
this paper derives a rank or capacity claim from argmax recall: the
rank law of Section 3 forces exact continuous recovery through a
Procrustes-aligned cosine readout, and every argmax recall number in
Section 4 carries their caveat explicitly. Their analysis is the reason
the causal razor of Section 3 refuses argmax decoding anywhere a rank
claim depends on it.

**Associative recall in transformers.** Multi-query associative recall
is a regime where softmax attention is documented as strong: Arora et
al. (2023) introduce the MQAR benchmark and show attention solves it
where sub-quadratic models struggle, Jelassi et al. (2024) report the
same advantage for copying over state-space models, and even two-layer
from-scratch transformers reliably develop induction circuitry
(Olsson et al., 2022). Our transformer baseline's chance-level reading
therefore runs against the expectation this literature sets, and we do
not interpret it as an architectural inability. The leading candidate
explanation was optimization: the arm trained at the shared default
learning rate with no architecture-specific search on the recall task,
and its training loss is near flat across the run (Section 4.1). A
subsequent four-point learning-rate search on the recall task itself
rules this out: the best-optimizing rate reads recall furthest below
chance, so a better-optimized language-modeling objective does not
surface recall on this arm (Section 7 reports the full grid). The
separation verdict accordingly rests on the vector-state ablation
comparison alone, with the transformer now disclosed as a second
failing baseline whose chance-level reading survives an explicit
learning-rate search; Section 7 reports the completed measurement and
its optimization-recall dissociation.

**Rank measurement in linear attention.** Two contemporaneous studies
(arXiv:2602.04852; arXiv:2602.02195) measure the rank of
linear-attention states descriptively, on frozen checkpoints. Section 3
differs in both directions of inference: the force-rank arms intervene
at train time, and the necessity/sufficiency step at $d_{\min}$ is a
causal result a descriptive measurement cannot license.

**Key normalization and stability.** The qk-normalization line (Kimi
Linear, Section 4 of arXiv:2510.26692; Qwen3-Next) conditions
single-vector eigenvalue stability of the state update. The attractor
of Section 5 is a different object, cross-key population geometry. It
was measured with qk normalization active throughout, and removing the
normalization is a within-noise null at $n{=}3$ <!-- evidence: R7 -->.

**DeltaNet and gated variants.** DeltaNet and Gated DeltaNet (Yang et
al.) supply the architecture substrate for the capability and pathology
legs (Sections 4 and 5) and benchmark it for quality and throughput.
This paper asks what the states represent: the rank law of Section 3 is
established on a separate matrix-state encoder family with no
delta-rule write, while the causal recall capability and the
write-induced geometry at scale are established on the delta-rule
substrate itself. The gating
axis reappears here only as a measured factor in Section 5.2, where it
trends toward amplifying the attractor without confirming.

**Prior negative result in this program.** A published companion study
("The Gradient Does Not See Rank," ICML 2026 MI workshop) showed that
bolting a matrix state onto a pretrained backbone through a
flatten-then-project readout leaves the gradient rank-blind. The
present paper is the from-scratch counterpart: when the state is native
and the readout preserves structure, SGD does see rank (Section 3), and
the resulting medium supports a capability its matched ablations lack
(Section 4).
