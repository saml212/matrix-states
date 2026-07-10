# Appendix A The $c^{*} I$ Complement Scaffold

A mechanism-level observation, architecture-conditional and offered as
a candidate rather than a claim of the main thesis. In the converged
matrix-state encoder family (the Section 3 substrate's sibling trained
on parallel key-value binding), the trained state decomposes as
$Z \approx c^{*} I_d + \Delta$, where $\Delta$ is a rank-$(K{-}1)$ task
correction: the whole-state law holds at 0.5 to 2.9 percent Frobenius
residual, with per-example identity alignment $\tau \ge 0.9997$ and
effective rank of $Z - c^{*} I$ within 0.3 of the $K{-}1$ target at
both tested loads <!-- evidence: R9 -->. The scaffold is emergent: the
architecture has no identity path or output gain that parameterizes it.

Two properties make it useful and bound it. First, deviation from the
scaffold is a loss-blind health signal: the Procrustes residual of the
complement correlates with held-out task cosine at Spearman
$\rho = -0.973$ across all eleven archived runs <!-- evidence: R9 -->,
read entirely from a subspace the loss never constrains, so it can
grade convergence without labels. Second, the scaffold does not exist
where Sections 4 and 5 live: in delta-rule states the orthogonal
complement channel is numerically empty (complement energy fraction at
most $3.2 \times 10^{-12}$ across 12 of 12 production-family runs
<!-- evidence: R9 -->), because outer-product writes confine the state
to the span of observed keys and values by construction. The scaffold
is therefore an encoder-family instrument and a delta-rule
impossibility, which is itself a datum: the two families that share the
rank law of Section 3 and the capability of Section 4 do not share
their complement structure, and mechanism stories that depend on an
identity scaffold cannot port across that boundary.

**Figure A1 caption.** The complement scaffold is architecture-
conditional. Left: complement Procrustes residual per archived Task-E
encoder run (sorted; log scale). Converged runs sit at 0.003 to 0.018,
two orders of magnitude below the Gaussian-null band (0.53 to 0.54,
shaded); the high-residual runs are the archive's non-converged
members, which is the separation behind the loss-blind health signal
(Spearman $\rho = -0.973$ against held-out task cosine, all eleven
runs). Right: complement energy fraction $f_D$ by architecture family
(log scale): the encoder family carries measurable complement energy,
while delta-rule production states read at most $3.2 \times 10^{-12}$,
numerically empty, because outer-product writes confine the state to
the span of observed keys and values.
