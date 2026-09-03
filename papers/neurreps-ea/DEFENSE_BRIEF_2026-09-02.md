# Rank-2 defense brief — the causal rank law (2026-09-02)

For Sam to own at a whiteboard. Every number is from `brief.md` rows
N1–N14, re-read today from the md5-pinned raw JSONs in
`experiment-runs/2026-07-09_capability_sweep_harvest/`,
`experiment-runs/2026-07-09_m3fix_harvest/`, and
`experiment-runs/2026-07-09_m3fix_s3ext/`. The agent attacks; Sam
answers. Holes get logged at the bottom. Merged paper = `neurreps-ea`
(Candidate A title) absorbing `unireps-ea`'s equivalence section and
instrument appendix; target arXiv Mon Sep 8.

## The absolute claim (lead with this, per the headline rule)

A Transformer-encoded single-matrix-state model, trained end-to-end on
the word problem of five finite groups (S3, S4, A5, S5, A6) under a hard
one-state bottleneck with a fixed decoder, recruits a state whose
effective rank equals the group's minimal faithful real representation
dimension d_min = 2, 3, 3, 4, 5, with two spare dimensions available in
every case (d_state = d_min + 2). Capping the state's rank at d_min − 1
during training drives exact recovery to 0.000 in every group and every
seed; restoring rank d_min brings recovery back above the pre-registered
0.9×anchor bar in all five groups. The solvable/non-solvable pair
matched at d_min = 3 (S4 vs A5) is statistically equivalent in
recruited rank under a pre-registered equivalence test.

Numbers: restricted effective rank 1.877 / 2.852 / 2.832 / 3.591 / 4.736
(S3/S4/A5/S5/A6, n = 3/5/5/3/3), all 19 seeds in [0.7, 1.3]·d_min;
Spearman ρ = 0.9747 (tie-capped maximum; exact null P(ρ ≥ 0.8) = 8/120).
TOST S4 − A5: Δ = +0.019, se 0.037, t = 13.06/14.12 vs t_crit 1.865 at
margin ±0.5. Razor (crosscheck rec@0.9, one seed per cell): k = d_min − 1
→ 0.000 ×5; k = d_min → 0.450* / 0.800 / 0.700 / 0.600 / 0.650 vs bars
0.495 / 0.585 / 0.630 / 0.450 / 0.585; *S3 by four-seed mean 0.5625.

## Held fixed / grew / falsifier (the three sentences)

- HELD FIXED: one architecture, one training recipe per group (step
  budgets pinned by convergence bars before any decisional cell ran),
  the P=1 single-state bottleneck (decoder reads only Z; blank-out
  verified), a fixed readout with no learned weights, exact continuous
  recovery (cosine ≥ 0.9 on held-out words after fitting only a scale
  and an orthogonal gauge on a disjoint split), d_state = d_min + 2.
- GREW: d_min from 2 to 5 across five groups straddling the solvable
  divide. Recruited rank tracked it; the razor step moved with it.
- FALSIFIER (pre-registered before the fix wave ran): any k = d_min − 1
  cell reaching ≥ 0.9× the anchor; any group whose k = d_min cell fails
  to return; any seed outside the band; the marquee pair separating by
  more than 0.5 rank-units. None fired. The one marginal cell (S3 at
  k = d_min, |Δ| = 0.045) fell inside a ±0.05 trigger that was written
  down before the data and routed to a pre-specified 3-seed extension.

## Controls Sam must be able to name unprompted

1. The necessity floor is GEOMETRY, not a discovery. Against a target
   with d_min tied unit singular values, any rank-(d_min − 1) matrix has
   cosine ≤ √((d_min−1)/d_min) ≤ 0.894 < 0.9 (von Neumann + Cauchy–
   Schwarz; unireps Appendix B derives it). So 0.000 at d_min − 1 is
   forced. The empirical content below d_min is that the capped cells
   reach 76–95% (mean 88%) of that ceiling: they trained to the optimum
   the cap allows, they did not collapse. Say this before the reviewer
   does.
2. The causal weight is SUFFICIENCY: nothing guarantees a rank-exactly-
   d_min solution is reachable by SGD, and it is, in 5/5 groups (4
   outright, S3 by seed-mean under the pre-stated trigger).
3. The eye-padding tax (D-AMB). The first 58-cell sweep nulled: its
   force-rank target was ρ_G ⊕ I₂ (rank d_min + 2), the constant identity
   block is the cheapest loss reduction, so every capped arm spent 2 of
   its k on it and never tested the confirm direction. Diagnosed from
   raws (39 cells within 0.028 of √(k/d_state), the rank-k optimum),
   registered INCONCLUSIVE, fixed with the zero-padded target, and a
   deliberately eye-padded corroboration arm reproduces the failure on
   demand at k = d_min + 1.
4. Centering is load-bearing: an uncentered lens scores a flawless
   synthetic model 0.705; the centered production lens scores 0.9996 on
   identical data.
5. The S3 per-seed story, in full: k = d_min clears its own seed's bar in
   2/4 seeds (+0.010, +0.110; misses −0.045, −0.120) because the anchor
   itself ranges 0.550–0.800. The pre-registered comparison is the
   four-seed mean (0.5625) against the bar FIXED before the extension
   ran (0.495, from seed 0's anchor). Recomputing the bar from the
   four-seed anchor mean (0.574) puts the seed-mean 0.011 below it. Both
   numbers are in the paper. Sam says: "S3 confirms by the pre-registered
   rule and is marginal by the self-referential one; the law is 4/5
   clean without it, including both marquee members."
6. Soft convergence at the shortest pinned budget: all 8 S3/S5 razor
   cells miss the 0.92 gate1a validation bar (S3 0.900–0.914, S5
   0.876–0.879). Disclosed in the appendix; necessity unaffected by
   construction; S3/S5 sufficiency reads "directionally consistent".
7. The observational band's upper half is non-binding: restricted
   effective rank is measured inside the top-d_min subspace of the
   centered covariance, so it cannot exceed d_min by construction. Only
   the lower half (≥ 0.7·d_min) is a real test. Means land at
   0.895–0.951·d_min. That is why the observational leg was
   pre-registered as corroborating and the razor carries the claim.

## NEW today, from the same raw files (not yet in either draft)

The raw JSONs carry `whole_matrix_effective_rank`, the entropy effective
rank of the full d_state × d_state state with no subspace lens. It
answers control 7's objection directly:

| arm | target rank | whole-matrix eff. rank, unconstrained | reads |
|---|---|---|---|
| observational (ρ ⊕ I₂) | d_min + 2 = 4/5/5/6/7 | 3.77 / 4.84 / 4.78 / 5.51 / 6.72 | recruits the target's rank |
| razor anchor (ρ ⊕ 0) | d_min = 2/3/3/4/5 | 1.74* / 2.95 / 2.88 / 3.55 / 4.73 | recruits d_min, leaves 2 dims idle |

*S3 four seeds: 1.74, 1.85, 1.89, 1.82. Single seed elsewhere.

So with two spare dimensions available and nothing in the lens to hide
them, the unconstrained zero-padded model recruits d_min and no more.
This is the cleanest instrument-independent statement of "the rank the
task demands" in the whole campaign and it is one appendix table plus
one sentence in §5. Sam decides whether it goes in (recommend yes; it
is descriptive, single-seed outside S3, and computed from files already
md5-pinned in figure_gen.py).

8. ESTIMATOR DEPENDENCE (found 2026-09-02, not in the paper). The
   pre-registered instrument is the entropy effective rank. The raw JSONs
   also carry `restricted_stable_rank` (Frobenius² over spectral²), the
   estimator Nazari et al. use. Under stable rank the same unconstrained
   states read 1.451 / 2.156 / 2.078 / 2.362 / 3.340 against d_min
   2/3/3/4/5, i.e. 0.59–0.73·d_min, OUTSIDE the [0.7,1.3] band in every
   group; the ordering survives (Spearman of group means 0.975, the same
   tie-capped maximum). Sam says: "the ordinal law (rank tracks d_min) is
   estimator-robust; the point estimate `equals d_min` is a property of
   the entropy estimator on a near-flat spectrum with one weak direction,
   which stable rank penalizes; the causal razor caps ALGEBRAIC rank and
   does not depend on the estimator at all." Decide whether to disclose
   this in the instrument appendix (recommend yes, one sentence with the
   five numbers).

## Hostile-reviewer questions (agent asks these; log the answers)

Q1. Your necessity result is a tautology: the metric's own geometry
forbids recovery below d_min. Where is the empirical necessity?
(Answer: control 1 verbatim; then "the causal weight is sufficiency";
then the 76–95%-of-ceiling number.)

Q2. Restricted effective rank cannot exceed d_min by construction. Your
"rank converges to d_min" is half instrument. (Answer: control 7, then
the NEW whole-matrix table: on the zero-padded target the full state
lands at d_min with two dimensions to spare.)

Q3. One seed per razor cell. That is not a causal law, that is five
anecdotes. (Answer: necessity is zero-noise by construction and
unanimous in the 4 S3 seeds; sufficiency is 5/5 with the S3 extension;
the observational leg has 19 seeds and the TOST has n = 5 per group.
Concede: single-seed sufficiency cells are the paper's weakest link and
the limitations section says so.)

Q4. S3 fails its own bar in 2 of 4 seeds and fails the anchor-mean bar
by 0.011. You picked the comparison that passes. (Answer: control 5.
The fixed-literal rule was written before the extension ran, precisely
to avoid judging new data against a bar computed from that same noisy
data. Both readings are printed.)

Q5. Why is S4 at k = d_min (0.800) ABOVE its own unconstrained anchor
(0.650)? A cap should not help. (Answer: N14; S5 too, 0.60 vs 0.50.
Hypothesis, stated as unconfirmed in the appendix: the capped arm skips
the ambient dimensions the zero-padded anchor must still learn to null
in the same step budget. Do not claim more.)

Q6. Isn't this just Nichani, Lee & Bietti's rank-m associative memory?
(Answer: their bound is under argmax decoding, which a rank-1 state
defeats by storing ≈ d items; exact continuous recovery closes that
loophole; and they construct, we measure what SGD recruits and then
intervene on it.)

Q7. Merrill/Grazzi/Siems already sort word problems by solvability.
What does representation dimension add? (Answer: they characterize
what an architecture CAN express; we measure what a trained state DOES
recruit, and it sorts by d_min, not by the NC¹ divide: S4 and A5 are
equivalent to within 0.019 rank-units at matched d_min.)

Q8. d_state = d_min + 2 is a rigged margin. What happens at d_min + 10?
(Answer: not tested; state it. The claim is scoped to d_state = d_min +
2. The binding-task companion has the wider grids.)

Q9. Sub-1M-parameter synthetic models. Why should anyone care?
(Answer: the claim is about what gradient descent buys under a hard
matrix bottleneck, stated where it can be checked exactly; the recall
paper is the same bottleneck at 14M on a task a matched transformer
cannot do. Do not overreach into LLM claims.)

Q10. Your first sweep nulled and you re-ran with a different target
until it worked. (Answer: control 3. The re-run was a registered fix of
a diagnosed instrument defect, the diagnosis is reproducible from the
39 raw cells, the verdict was recorded INCONCLUSIVE not spun, and the
zero-padded prediction was written down before the fix wave ran.)

Q11. The two "papers" (neurreps/unireps) were the same result with two
headlines. Which is the claim? (Answer: one paper now. Headline = the
causal rank law; the equivalence result is its second leg, not a
separate finding.)

Q12. Would stable rank give the same d_min? (Answer: control 8. No for
the point estimate, yes for the ordering; the razor is estimator-free.)

Q13. He et al. (2026) and Shutman et al. (2025) already prove GD finds
low-rank, irrep-structured solutions on group composition. What is new?
(Answer: those are two-layer MLPs on the binary operation g1⋆g2 with
rank-one Fourier coefficients per irrep; no sequential state, no
faithful-dimension target, no intervention. This paper measures a
recurrent matrix state against d_min and caps it.)

## Holes logged (fill during the pass)

- [ ]
