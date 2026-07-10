# 6 Related Work

**Associative capacity of fast-weight states.** Nichani, Lee, and
Bietti (2025) prove that under argmax decoding a rank-one matrix can
recover on the order of $d$ associations, which is why no result in
this paper derives a rank or capacity claim from argmax recall: the
rank law of Section 3 forces exact continuous recovery through a
Procrustes-aligned cosine readout, and every argmax recall number in
Section 4 carries their caveat explicitly. Their analysis is the reason
our causal razor is decode-proof rather than decode-laundered.

**Rank measurement in linear attention.** Two contemporaneous studies
(arXiv:2602.04852; arXiv:2602.02195) measure the rank of
linear-attention states descriptively, on frozen checkpoints. Section 3
differs in both directions of inference: the force-rank arms intervene
at train time, and the necessity/sufficiency step at $d_{\min}$ is a
causal result a descriptive measurement cannot license.

**Key normalization and stability.** The qk-normalization line (Kimi
Linear, Section 4 of arXiv:2510.26692; Qwen3-Next) conditions
single-vector eigenvalue stability of the state update. The attractor
of Section 5 is a different object: cross-key population geometry. It
was measured with qk normalization active throughout, and removing the
normalization is a within-noise null at $n{=}3$ <!-- evidence: R7 -->,
so the two phenomena are empirically as well as conceptually distinct.

**DeltaNet and gated variants.** DeltaNet and Gated DeltaNet (Yang et
al.) supply the architecture substrate and benchmark it for quality and
throughput. This paper holds the substrate fixed and asks what the
states represent: their recruited dimensionality, their causal role in
a capability, and the geometry their writes induce at scale. The gating
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
