# Rank-law paper scrub log

Base commit: `cbc2874c4962f3d103f14ca6605ff4b747aac7a1`

Scope: `papers/neurreps-ea/arxiv-v2/`, the two mapped figure-generator
functions, their canonical generated PDFs, and this log. The evidence table,
bibliography, raw artifacts, protected source hashes, data loading, and
statistical computations remain unchanged.

Baseline build: Tectonic 0.16.9 completed with 11 pages, 2 figures, 8 tables,
25 unique labels, no missing references, and no unresolved citations.

## Prose changes

### `arxiv-v2/main.tex:64`

- Pattern: sentence longer than 35 words carrying the companion result and
  this paper's extension.
- Original sentence:

  ```tex
  A companion paper \citep{larson2026companion} establishes the analogous recruitment and causal necessity pattern on a $K$-pair associative-binding testbed, where exact recovery provably requires state rank at least $K$; this paper inherits that instrument and extends the rank law from a scalar capacity bound to a representation-theoretic one.
  ```

- New sentence(s):

  ```tex
  A companion paper \citep{larson2026companion} establishes the analogous recruitment and causal necessity pattern on a $K$-pair associative-binding testbed, where exact recovery provably requires state rank at least $K$. This paper inherits that instrument and extends the rank law from a scalar capacity bound to a representation-theoretic one.
  ```

### `arxiv-v2/main.tex:70`

- Pattern: the parenthetical “both marquee members included” restated the
  explicit group list in promotional language.
- Original sentence:

  ```tex
  On group-composition state tracking over five finite groups spanning the solvable/non-solvable divide, the recruited rank equals the group's minimal faithful real representation dimension $\dmin$ (Spearman $\rho = 0.9747$, the design's tie-capped maximum), the dimension-matched solvable/non-solvable pair $S_4$/$A_5$ is statistically equivalent under a pre-registered test, and a pre-registered force-rank razor is exact in both directions: one rank below $\dmin$, recovery is capped by the target's tied unit spectrum at $\sqrt{(\dmin{-}1)/\dmin} \le 0.894$, below the $0.9$ threshold in every group by construction, with observed cells at 76--95\% (mean 88\%) of that ceiling; at $\dmin$, not guaranteed a priori, recovery clears the pre-registered anchor-relative bar at four seeds per group in four of the five groups ($S_3$, $S_4$, $A_5$, $A_6$, both marquee members included), while $S_5$ fails that rule at $n = 4$ (seed-mean 0.4125 against its fixed 0.450 bar) and clears only the self-referential one.
  ```

- New sentence(s):

  ```tex
  On group-composition state tracking over five finite groups spanning the solvable/non-solvable divide, the recruited rank equals the group's minimal faithful real representation dimension $\dmin$ (Spearman $\rho = 0.9747$, the design's tie-capped maximum), the dimension-matched solvable/non-solvable pair $S_4$/$A_5$ is statistically equivalent under a pre-registered test, and a pre-registered force-rank razor is exact in both directions: one rank below $\dmin$, recovery is capped by the target's tied unit spectrum at $\sqrt{(\dmin{-}1)/\dmin} \le 0.894$, below the $0.9$ threshold in every group by construction, with observed cells at 76--95\% (mean 88\%) of that ceiling; at $\dmin$, not guaranteed a priori, recovery clears the pre-registered anchor-relative bar at four seeds per group in four of the five groups ($S_3$, $S_4$, $A_5$, $A_6$), while $S_5$ fails that rule at $n = 4$ (seed-mean 0.4125 against its fixed 0.450 bar) and clears only the self-referential one.
  ```

### `arxiv-v2/sections/01_intro.tex:6`

- Pattern: sentence longer than 35 words carrying both the literature
  characterization and the open question.
- Original sentence:

  ```tex
  Prior work treats this budget descriptively \citep{nazari2026rank, sun2026staterank} or through hand-built constructions \citep{nichani2025factual}; neither answers what a geometric account of learned representations needs answered: when a task's algebra fixes a minimal representational dimension, does gradient descent recruit exactly that dimension, and is it causally load-bearing?
  ```

- New sentence(s):

  ```tex
  Prior work treats this budget descriptively \citep{nazari2026rank, sun2026staterank} or through hand-built constructions \citep{nichani2025factual}. Neither line of work answers what a geometric account of learned representations needs answered: when a task's algebra fixes a minimal representational dimension, does gradient descent recruit exactly that dimension, and is it causally load-bearing?
  ```

### `arxiv-v2/sections/01_intro.tex:14`

- Pattern: sentence longer than 35 words carrying two independent
  experimental constraints.
- Original sentence:

  ```tex
  We answer both halves affirmatively under one discipline: a hard single-state bottleneck (the decoder reads only one matrix $Z$, verified by a gradient blank-out test) and exact-continuous-recovery scoring (cosine against the true continuous target, never argmax over a codebook, which would let a rank-1 state recover on the order of $d$ associations \citep{nichani2025factual}).
  ```

- New sentence(s):

  ```tex
  We answer both halves affirmatively under one discipline: a hard single-state bottleneck and exact-continuous-recovery scoring. The decoder reads only one matrix $Z$, as verified by a gradient blank-out test. This score uses cosine against the true continuous target, never argmax over a codebook, which would let a rank-1 state recover on the order of $d$ associations \citep{nichani2025factual}.
  ```

### `arxiv-v2/sections/01_intro.tex:21`

- Pattern: “The headline is” announced the claim instead of stating its role.
- Original sentence:

  ```tex
  The headline is group-composition state tracking, where representation theory supplies the ground truth: across five finite groups spanning the solvable/non-solvable divide, the minimal faithful real representation dimension $\dmin$ predicts the recruited rank almost perfectly (Section~\ref{sec:observed}), a matched-dimension solvable/non-solvable pair lands statistically equivalent (Section~\ref{sec:equivalence}),
  % <!-- evidence: N4, N5 -->
  and a pre-registered force-rank razor (Section~\ref{sec:razor}) is exact in the necessity direction and confirms sufficiency in four of the five groups: $\dmin{-}1$ is analytically incapable of exact recovery under the $0.9$ threshold ($\sqrt{(\dmin{-}1)/\dmin} \le 0.894$), and $\dmin$ suffices at four seeds per group in $S_3$, $S_4$, $A_5$, $A_6$, which is not guaranteed a priori, while $S_5$ fails the pre-registered seed-mean rule at $n = 4$.
  ```

- New sentence:

  ```tex
  The central test is group-composition state tracking, where representation theory supplies the ground truth: across five finite groups spanning the solvable/non-solvable divide, the minimal faithful real representation dimension $\dmin$ predicts the recruited rank almost perfectly (Section~\ref{sec:observed}), a matched-dimension solvable/non-solvable pair lands statistically equivalent (Section~\ref{sec:equivalence}),
  % <!-- evidence: N4, N5 -->
  and a pre-registered force-rank razor (Section~\ref{sec:razor}) is exact in the necessity direction and confirms sufficiency in four of the five groups: $\dmin{-}1$ is analytically incapable of exact recovery under the $0.9$ threshold ($\sqrt{(\dmin{-}1)/\dmin} \le 0.894$), and $\dmin$ suffices at four seeds per group in $S_3$, $S_4$, $A_5$, $A_6$, which is not guaranteed a priori, while $S_5$ fails the pre-registered seed-mean rule at $n = 4$.
  ```

### `arxiv-v2/sections/02_setup.tex:4`

- Pattern: sentence longer than 35 words carrying the episode, state,
  prediction, and scoring definitions.
- Original sentence:

  ```tex
  In the binding task, each episode presents $K$ key--value pairs, the encoder writes one state $Z \in \R^{d \times d}$, prediction is the literal unbind $Z k_j$, and $\recninety$ is the fraction of queries with cosine to the true value above 0.9.
  ```

- New sentence(s):

  ```tex
  In the binding task, each episode presents $K$ key--value pairs. The encoder writes one state $Z \in \R^{d \times d}$. Prediction is the literal unbind $Z k_j$. $\recninety$ is the fraction of queries with cosine to the true value above 0.9.
  ```

### `arxiv-v2/sections/02_setup.tex:7`

- Pattern: sentence longer than 35 words carrying the task identity, sampling
  procedure, and target.
- Original sentence:

  ```tex
  The group task is the word problem of a finite group $G$: a word $w = g_{i_1} \cdots g_{i_L}$, drawn as a random walk on $G$'s Cayley graph ($L \sim \mathrm{U}\{1,\dots,8\}$), must map to $\rho_G(g_{i_1} \cdots g_{i_L})$ under a pinned orthogonal reference representation of dimension $\dmin(G)$, the minimal faithful real representation dimension.
  ```

- New sentence(s):

  ```tex
  The group task is the word problem of a finite group $G$. A word $w = g_{i_1} \cdots g_{i_L}$ is drawn as a random walk on $G$'s Cayley graph ($L \sim \mathrm{U}\{1,\dots,8\}$). It must map to $\rho_G(g_{i_1} \cdots g_{i_L})$ under a pinned orthogonal reference representation of dimension $\dmin(G)$, the minimal faithful real representation dimension.
  ```

### `arxiv-v2/sections/02_setup.tex:13`

- Pattern: sentence longer than 35 words carrying the group dimensions,
  solvability assignments, and complexity result.
- Original sentence:

  ```tex
  The groups are $S_3, S_4, A_5, S_5, A_6$ with $\dmin = 2, 3, 3, 4, 5$ (Appendix~\ref{app:groups}); the first two are solvable, the rest are not, and non-solvable word problems are $\mathrm{NC}^1$-complete \citep{barrington1989,barringtontherien1988}.
  ```

- New sentence(s):

  ```tex
  The groups are $S_3, S_4, A_5, S_5, A_6$ with $\dmin = 2, 3, 3, 4, 5$ (Appendix~\ref{app:groups}). The first two are solvable, and the rest are not. Non-solvable word problems are $\mathrm{NC}^1$-complete \citep{barrington1989,barringtontherien1988}.
  ```

### `arxiv-v2/sections/02_setup.tex:17`

- Pattern: one sentence combined the distinction between axes with the
  designed comparison pair.
- Original sentence:

  ```tex
  $\dmin$ is a different, representation-theoretic axis; $S_4$/$A_5$, matched at $\dmin = 3$ with opposite solvability, are the designed dissociation pair.
  ```

- New sentence(s):

  ```tex
  $\dmin$ is a different, representation-theoretic axis. The design uses $S_4$/$A_5$, matched at $\dmin = 3$ with opposite solvability, as the dissociation pair.
  ```

### `arxiv-v2/sections/02_setup.tex:24`

- Pattern: sentence longer than 35 words carrying the target construction,
  loss, readout constraint, and decoder constraint.
- Original sentence:

  ```tex
  The target embeds the reference block-diagonally ($\rho_G(\cdot) \oplus I_2$ in the observational arm, $\rho_G(\cdot) \oplus 0$ in the causal-razor arm; Appendix~\ref{app:instrument}); the loss is cosine distance, the readout is fixed with no learned weights that could launder rank, and the decoder reads only $Z$ (blank-out verified).
  ```

- New sentence(s):

  ```tex
  The target embeds the reference block-diagonally ($\rho_G(\cdot) \oplus I_2$ in the observational arm, $\rho_G(\cdot) \oplus 0$ in the causal-razor arm; Appendix~\ref{app:instrument}). The loss is cosine distance. The fixed readout has no learned weights that could launder rank, and the decoder reads only $Z$ (blank-out verified).
  ```

### `arxiv-v2/sections/02_setup.tex:29`

- Pattern: passive voice hid what fixed the step budgets.
- Original sentence:

  ```tex
  Per-group step budgets (8k--40k, recorded in the design document's pre-registration) were pinned by convergence bars before any decisional cell ran.
  ```

- New sentence:

  ```tex
  Convergence bars pinned the per-group step budgets (8k--40k) before any decisional cell ran, as recorded in the design document's pre-registration.
  ```

### `arxiv-v2/sections/02_setup.tex:36`

- Pattern: sentence longer than 35 words carrying three independent
  instrument operations.
- Original sentence:

  ```tex
  We estimate the model's own dominant $\dmin$-subspace $U$ from the SVD of the \emph{centered} covariance of $Z(w)$ over held-out words, measure \textbf{restricted effective rank} (entropy effective rank of $U^{\top} Z(w) U$), and score \textbf{degauged recovery}: cosine after fitting $(c, Q)$ on a fitting split, evaluated on a disjoint split, reported as $\recninety$.
  ```

- New sentence(s):

  ```tex
  We estimate the model's own dominant $\dmin$-subspace $U$ from the SVD of the \emph{centered} covariance of $Z(w)$ over held-out words. We measure \textbf{restricted effective rank} as the entropy effective rank of $U^{\top} Z(w) U$. We score \textbf{degauged recovery} as cosine after fitting $(c, Q)$ on a fitting split and evaluating on a disjoint split, reported as $\recninety$.
  ```

### `arxiv-v2/sections/02_setup.tex:42`

- Pattern: sentence longer than 35 words carrying both the centering
  requirement and the gauge-invariance consequence.
- Original sentence:

  ```tex
  Centering is load-bearing (Appendix~\ref{app:instrument}); rank is invariant under the fitted gauge, so a rank-deficient state cannot be degauged into a full-rank one.
  ```

- New sentence(s):

  ```tex
  Centering is load-bearing (Appendix~\ref{app:instrument}). Rank is invariant under the fitted gauge, so a rank-deficient state cannot be degauged into a full-rank one.
  ```

### `arxiv-v2/sections/02_setup.tex:45`

- Pattern: passive voice hid the pre-registration that fixed the decisional
  readout.
- Original sentence:

  ```tex
  For force-rank cells the decisional readout was pinned in advance to the conservative full-$Q$ Procrustes variant (\emph{crosscheck} $\recninety$).
  ```

- New sentence:

  ```tex
  For force-rank cells, the pre-registration pinned the conservative full-$Q$ Procrustes variant (\emph{crosscheck} $\recninety$) as the decisional readout.
  ```

### `arxiv-v2/sections/03_binding.tex:10`

- Pattern: sentence longer than 35 words carrying three companion-paper
  findings and this paper's scope.
- Original sentence:

  ```tex
  A companion paper \citep{larson2026companion} establishes on this binding testbed that gradient descent recruits effective rank tracking $K$, that the recruited rank is causally necessary (a train-time force-rank step at the provable bound), and that the trained operator composes exactly under repeated self-application, a composition-stability probe periodicity-equivalent to depth 5 under the single 8-cycle target \citep{liu2023shortcuts} (Appendix~\ref{app:period}); this paper inherits that instrument and carries only the group-composition program.
  ```

- New sentence(s):

  ```tex
  On this binding testbed, a companion paper \citep{larson2026companion} establishes that gradient descent recruits effective rank tracking $K$. The same paper shows that the recruited rank is causally necessary through a train-time force-rank step at the provable bound. It also shows that the trained operator composes exactly under repeated self-application. The composition-stability probe is periodicity-equivalent to depth 5 under the single 8-cycle target \citep{liu2023shortcuts} (Appendix~\ref{app:period}). This paper inherits that instrument and carries only the group-composition program.
  ```

### `arxiv-v2/sections/04_ranklaw_observed.tex:7`

- Patterns: caption encodings were compressed into parentheticals, and the
  seed-offset display was not defined.
- Original sentence:

  ```tex
  Each point is one seed's restricted effective rank on the group word problem vs.\ its group's $\dmin$ (filled: solvable; open: non-solvable), with the pre-registered $[0.7, 1.3] \cdot \dmin$ band and the identity line; all 19 seeds are in band, and $S_4$/$A_5$ coincide at $\dmin = 3$.
  ```

- New sentence:

  ```tex
  Each point is one seed's restricted effective rank on the group word problem vs.\ its group's $\dmin$. Filled markers denote solvable groups; open markers denote non-solvable groups. Small horizontal offsets separate seeds at the same categorical value. Shading marks the pre-registered $[0.7, 1.3] \cdot \dmin$ band, and the dashed line marks the identity. All 19 seeds are in band, and $S_4$/$A_5$ coincide at $\dmin = 3$.
  ```

### `arxiv-v2/sections/04_ranklaw_observed.tex:14`

- Pattern: caption fragment; the reference-line encodings were incomplete.
- Original sentence:

  ```tex
  Inset: per-seed $S_4 - \overline{A_5}$ rank differences at matched $\dmin = 3$ against the pre-registered $\pm 0.5$ rank-unit equivalence margin (shaded); the vertical line is the observed mean difference, $+0.019$ (Section~\ref{sec:equivalence}).
  ```

- New sentence(s):

  ```tex
  The lower panel plots per-seed $S_4 - \overline{A_5}$ rank differences at matched $\dmin = 3$ against the pre-registered $\pm 0.5$ rank-unit equivalence margin. Shading marks the margin. The vertical lines mark zero difference and the observed mean difference, $+0.019$ (Section~\ref{sec:equivalence}).
  ```

### `arxiv-v2/sections/04_ranklaw_observed.tex:22`

- Pattern: “Figure ... gives” named the display before stating the result.
- Original sentence:

  ```tex
  Figure~\ref{fig:tracking} gives the observational leg: per-group mean restricted effective rank lands within 4.9--10.2\% of $\dmin$ at every group, and all 19 seeds sit inside the pre-registered $[0.7, 1.3] \cdot \dmin$ band (means and per-seed values in Appendix~\ref{app:m1}).
  ```

- New sentence:

  ```tex
  Per-group mean restricted effective rank lands within 4.9--10.2\% of $\dmin$ at every group, and all 19 seeds sit inside the pre-registered $[0.7, 1.3] \cdot \dmin$ band (Figure~\ref{fig:tracking}; means and per-seed values in Appendix~\ref{app:m1}).
  ```

### `arxiv-v2/sections/04b_equivalence.tex:4`

- Patterns: passive voice; promotional “decisive”; sentence longer than 35
  words carrying the design and theoretical axis.
- Original sentence:

  ```tex
  The pair $S_4$/$A_5$ was designed into the family as the decisive contrast: the same $\dmin = 3$, opposite sides of the solvability divide, the axis on which expressivity theory separates state-tracking architectures \citep{merrill2024illusion, grazzi2025negative, siems2025deltaproduct}.
  ```

- New sentence(s):

  ```tex
  The design uses $S_4$/$A_5$ as its contrast. The groups share $\dmin = 3$ but lie on opposite sides of the solvability divide, the axis on which expressivity theory separates state-tracking architectures \citep{merrill2024illusion, grazzi2025negative, siems2025deltaproduct}.
  ```

### `arxiv-v2/sections/04b_equivalence.tex:10`

- Pattern: sentence longer than 35 words carrying two precursor findings,
  their interpretation, this paper's equivalence test, and the later
  intervention.
- Original sentence:

  ```tex
  The precursor observation is in \citet{siems2025deltaproduct} (\S5.2): $S_4$ and $A_5$ both extrapolate with two Householder reflections and keys of size 3, which they attribute to the groups' embedding in $\mathrm{SO}(3)$, and a PCA of the trained keys is three-dimensional; the present test turns that single-architecture reading into a pre-registered equivalence claim on state rank across the five-group family, and Section~\ref{sec:razor} makes the dimension causal.
  ```

- New sentence(s):

  ```tex
  \citet{siems2025deltaproduct} provide the precursor observation (\S5.2): $S_4$ and $A_5$ both extrapolate with two Householder reflections and keys of size 3. They attribute this to the groups' embedding in $\mathrm{SO}(3)$, and their PCA of the trained keys is three-dimensional. The present test turns that single-architecture reading into a pre-registered equivalence claim on state rank across the five-group family. Section~\ref{sec:razor} makes the dimension causal.
  ```

### `arxiv-v2/sections/04b_equivalence.tex:17`

- Patterns: sentence longer than 35 words carrying the rationale, test,
  sample size, power simulation, and interpretation; passive voice hid the
  pre-registration.
- Original sentence:

  ```tex
  Because the interesting outcome is a null, the comparison was pre-registered as an equivalence test rather than a difference test: Welch two-one-sided tests on restricted effective rank at margin $\pm 0.5$ rank-units (half the spacing of the $\dmin$ ladder), $n = 5$ seeds per group, with a pre-run power simulation, fixed in the design record before the sweep ran, confirming the test reliably \emph{rejects} equivalence at a true gap of 1.0 rank-units, so a real class effect of one ladder step could not have hidden inside the margin.
  ```

- New sentence(s):

  ```tex
  Because the interesting outcome is a null, the pre-registration specifies an equivalence test rather than a difference test. The comparison uses Welch two-one-sided tests on restricted effective rank at margin $\pm 0.5$ rank-units (half the spacing of the $\dmin$ ladder), with $n = 5$ seeds per group. The design record fixed a pre-run power simulation before the sweep ran. It confirms that the test reliably \emph{rejects} equivalence at a true gap of 1.0 rank-units, so a real class effect of one ladder step could not have hidden inside the margin.
  ```

### `arxiv-v2/sections/04b_equivalence.tex:27`

- Patterns: passive voice and promotional “decisively.”
- Original sentence:

  ```tex
  The observed difference is $+0.019$ rank-units (se 0.037, df 7.8), and both one-sided tests pass at roughly seven times the critical value ($t = 13.06$ and $14.12$ against $t_{\mathrm{crit}} = 1.865$): equivalence is declared decisively.
  ```

- New sentence(s):

  ```tex
  The observed difference is $+0.019$ rank-units (se 0.037, df 7.8), and both one-sided tests pass at roughly seven times the critical value ($t = 13.06$ and $14.12$ against $t_{\mathrm{crit}} = 1.865$): the two one-sided tests declare equivalence.
  ```

### `arxiv-v2/sections/04b_equivalence.tex:33`

- Pattern: passive voice hid the algebra as the actor.
- Original sentence:

  ```tex
  Convergence here is governed by what the algebra demands, not by how hard the class is to compute.
  ```

- New sentence:

  ```tex
  What the algebra demands governs convergence here, not how hard the class is to compute.
  ```

### `arxiv-v2/sections/04b_equivalence.tex:31`

- Pattern: the panel reference became stale after the figure redesign.
- Original sentence:

  ```tex
  The two groups' seed clusters are visually coincident in Figure~\ref{fig:tracking} (inset).
  ```

- New sentence:

  ```tex
  The two groups' seed clusters are visually coincident in Figure~\ref{fig:tracking} (lower panel).
  ```

### `arxiv-v2/sections/05_causal_razor.tex:4`

- Patterns: promotional “decisive” and passive experimental action.
- Original sentence:

  ```tex
  The decisive test intervenes on rank: the encoder's output is spectrally truncated to rank $k \in \{\dmin{-}1, \dmin, \dmin{+}1\}$ throughout training, against the zero-padded target $\rho_G(\cdot) \oplus 0$ of rank exactly $\dmin$, alongside an unconstrained anchor from the same family.
  ```

- New sentence:

  ```tex
  The causal test intervenes on rank by spectrally truncating the encoder's output to rank $k \in \{\dmin{-}1, \dmin, \dmin{+}1\}$ throughout training, against the zero-padded target $\rho_G(\cdot) \oplus 0$ of rank exactly $\dmin$, alongside an unconstrained anchor from the same family.
  ```

### `arxiv-v2/sections/05_causal_razor.tex:8`

- Patterns: “Pre-registered reading:” was a fragment that announced the
  interpretation; one sentence combined the integrity control and live
  prediction.
- Original sentence:

  ```tex
  Pre-registered reading: below $\dmin$, a sound readout pins recovery to zero by geometry, making that arm an integrity control whose registered trigger (any recovery there) would indicate an instrument leak; the live prediction is that $k \geq \dmin$ recovers past $0.9\times$ the anchor's crosscheck $\recninety$.
  ```

- New sentence(s):

  ```tex
  The pre-registration treats the below-$\dmin$ arm as an integrity control: a sound readout pins recovery to zero by geometry, and its registered trigger is any recovery there, which would indicate an instrument leak. The live prediction is that $k \geq \dmin$ recovers past $0.9\times$ the anchor's crosscheck $\recninety$.
  ```

### `arxiv-v2/sections/05_causal_razor.tex:66`

- Pattern: “Table ... and Figure ... show” named the displays instead of
  stating the readout used.
- Original sentence:

  ```tex
  Table~\ref{tab:razor} and Figure~\ref{fig:razor} show the result on the pre-pinned crosscheck readout.
  ```

- New sentence:

  ```tex
  The result in Table~\ref{tab:razor} and Figure~\ref{fig:razor} uses the pre-pinned crosscheck readout.
  ```

### `arxiv-v2/sections/05_causal_razor.tex:79`

- Patterns: sentence longer than 35 words carrying replication and comparison
  procedure; passive experimental action.
- Original sentence:

  ```tex
  Every group was run at four seeds, and the pre-registered rule compares the seed-mean of $k{=}\dmin$ against the bar fixed from seed 0 before the further seeds ran, never against a bar recomputed from the seeds being judged.
  ```

- New sentence(s):

  ```tex
  We ran every group at four seeds. The pre-registered rule compares the seed-mean of $k{=}\dmin$ against the bar fixed from seed 0 before the further seeds ran, never against a bar recomputed from the seeds being judged.
  ```

### `arxiv-v2/sections/05_causal_razor.tex:99`

- Pattern: promotional “marquee” obscured which pair the sentence meant.
- Original sentence:

  ```tex
  Both marquee groups confirm at $\dmin$, meeting the causal criterion.
  ```

- New sentence:

  ```tex
  Both groups in the dimension-matched pair confirm at $\dmin$, meeting the causal criterion.
  ```

### `arxiv-v2/sections/06_related.tex:4`

- Pattern: sentence longer than 35 words carrying four literature findings
  and the paper-level contrast.
- Original sentence:

  ```tex
  \citet{chughtai2023universality} reverse-engineers group multiplication via representation theory, \citet{stander2024cosets} and \citet{wu2025unified} dispute and refine that account (coset circuits; verified per-argument equivariance), \citet{he2026spectral} prove that two-layer networks on $g_1 \star g_2$ learn spectral representations with a rank-one alignment of Fourier coefficients, and \citet{shutman2025words} frame word learning as a low-rank 3-tensor; all four study the binary operation, not a sequential state, and none relates a learned dimension to $\dmin$.
  ```

- New sentence(s):

  ```tex
  \citet{chughtai2023universality} reverse-engineers group multiplication via representation theory. \citet{stander2024cosets} and \citet{wu2025unified} dispute and refine that account (coset circuits; verified per-argument equivariance). \citet{he2026spectral} prove that two-layer networks on $g_1 \star g_2$ learn spectral representations with a rank-one alignment of Fourier coefficients. \citet{shutman2025words} frame word learning as a low-rank 3-tensor. All four study the binary operation, not a sequential state, and none relates a learned dimension to $\dmin$.
  ```

### `arxiv-v2/sections/06_related.tex:12`

- Pattern: sentence longer than 35 words carrying two literature findings
  and their common omission.
- Original sentence:

  ```tex
  On sequential composition, \citet{marchetti2026sequential} show recurrent networks acquire irreps one at a time with width scaling in the irrep dimensions on cyclic and dihedral groups, and \citet{zhang2026recurrence} find on looped transformers that $\mathrm{NC}^1$-completeness costs nothing while group order does; neither measures state rank.
  ```

- New sentence(s):

  ```tex
  On sequential composition, \citet{marchetti2026sequential} show recurrent networks acquire irreps one at a time with width scaling in the irrep dimensions on cyclic and dihedral groups. \citet{zhang2026recurrence} find on looped transformers that $\mathrm{NC}^1$-completeness costs nothing while group order does. Neither measures state rank.
  ```

### `arxiv-v2/sections/06_related.tex:22`

- Pattern: sentence longer than 35 words carrying four literature
  comparisons and the estimator conclusion.
- Original sentence:

  ```tex
  \citet{nazari2026rank} and \citet{sun2026staterank} measure state-rank dynamics in pretrained linear-attention models, observationally, with no task algebra fixing a required dimension, and read low rank as under-used capacity where this paper reads it as the rank the task demands; \citet{parnichkun2025ess} apply an entropy effective rank to sequence-model memory as a utilization measure, and \citet{truong2026spectral} track spectral-entropy collapse of representation covariances on group tasks, so the estimator here is standard, not new; \citet{mishra2026m2rnn} train matrix-state RNNs on $S_3$ composition without a rank intervention.
  ```

- New sentence(s):

  ```tex
  \citet{nazari2026rank} and \citet{sun2026staterank} measure state-rank dynamics in pretrained linear-attention models observationally, with no task algebra fixing a required dimension. They read low rank as under-used capacity, whereas this paper reads it as the rank the task demands. \citet{parnichkun2025ess} apply an entropy effective rank to sequence-model memory as a utilization measure. \citet{truong2026spectral} track spectral-entropy collapse of representation covariances on group tasks. The estimator here is therefore standard, not new. \citet{mishra2026m2rnn} train matrix-state RNNs on $S_3$ composition without a rank intervention.
  ```

### `arxiv-v2/sections/06_related.tex:32`

- Patterns: sentence longer than 35 words carrying architecture expressivity,
  benchmark scope, and this paper's comparison; promotional “marquee.”
- Original sentence:

  ```tex
  \citet{grazzi2025negative}, \citet{siems2025deltaproduct}, \citet{merrill2024illusion}, \citet{shakerinava2026diagonal}, and \citet{nowak2026algebraic} characterize which word problems recurrent architectures can express, sorting them by solvability and by the depth a subnormal series demands, an axis orthogonal to the one measured here (dimension governs state rank; series length governs depth), and \citet{deletang2023chomsky} benchmark sequence architectures across the formal-language hierarchy; the marquee equivalence instead sorts by representation dimension.
  ```

- New sentence(s):

  ```tex
  \citet{grazzi2025negative}, \citet{siems2025deltaproduct}, \citet{merrill2024illusion}, \citet{shakerinava2026diagonal}, and \citet{nowak2026algebraic} characterize which word problems recurrent architectures can express. They sort the problems by solvability and by the depth a subnormal series demands, an axis orthogonal to the one measured here (dimension governs state rank; series length governs depth). \citet{deletang2023chomsky} benchmark sequence architectures across the formal-language hierarchy. The equivalence test instead sorts by representation dimension.
  ```

### `arxiv-v2/sections/08_appendix.tex:20`

- Pattern: caption fragment.
- Original sentence:

  ```tex
  The five task groups.
  ```

- New sentence:

  ```tex
  The table lists the five task groups.
  ```

### `arxiv-v2/sections/08_appendix.tex:23`

- Pattern: promotional “marquee.”
- Original sentence:

  ```tex
  $S_4$/$A_5$ form the marquee pair: matched $\dmin$, opposite solvability.
  ```

- New sentence:

  ```tex
  $S_4$/$A_5$ form the comparison pair: matched $\dmin$, opposite solvability.
  ```

### `arxiv-v2/sections/08_appendix.tex:44`

- Pattern: caption fragment.
- Original sentence:

  ```tex
  Restricted effective rank of the unconstrained trained state (mean $\pm$ sd over seeds) against each group's $\dmin$, with the pre-registered $[0.7, 1.3] \cdot \dmin$ band.
  ```

- New sentence:

  ```tex
  The table reports restricted effective rank of the unconstrained trained state (mean $\pm$ sd over seeds) against each group's $\dmin$, with the pre-registered $[0.7, 1.3] \cdot \dmin$ band.
  ```

### `arxiv-v2/sections/08_appendix.tex:53`

- Pattern: sentence fragment.
- Original sentence:

  ```tex
  Per-group deviation of the mean from $\dmin$ (Table~\ref{tab:m1}): $S_3$ 6.1\%, $S_4$ 4.9\%, $A_5$ 5.6\%, $S_5$ 10.2\%, $A_6$ 5.3\%.
  ```

- New sentence:

  ```tex
  The per-group deviations of the mean from $\dmin$ (Table~\ref{tab:m1}) are $S_3$ 6.1\%, $S_4$ 4.9\%, $A_5$ 5.6\%, $S_5$ 10.2\%, and $A_6$ 5.3\%.
  ```

### `arxiv-v2/sections/08_appendix.tex:83`

- Pattern: caption fragment.
- Original sentence:

  ```tex
  Whole-matrix effective rank of the unconstrained trained state: the entropy effective rank of the full $\dstate \times \dstate$ state with no subspace lens, so unlike the restricted quantity of Table~\ref{tab:m1} it can range up to $\dstate = \dmin + 2$.
  ```

- New sentence:

  ```tex
  For the unconstrained trained state, whole-matrix effective rank is the entropy effective rank of the full $\dstate \times \dstate$ state with no subspace lens, so unlike the restricted quantity of Table~\ref{tab:m1} it can range up to $\dstate = \dmin + 2$.
  ```

### `arxiv-v2/sections/08_appendix.tex:87`

- Pattern: caption fragment.
- Original sentence:

  ```tex
  Observational arm: mean $\pm$ sd over seeds against the eye-padded target of rank $\dmin + 2$, which the state recruits ($3.77/4.84/4.78/5.51/6.72$ against $\dstate = 4/5/5/6/7$).
  ```

- New sentence:

  ```tex
  The observational arm reports mean $\pm$ sd over seeds against the eye-padded target of rank $\dmin + 2$, which the state recruits ($3.77/4.84/4.78/5.51/6.72$ against $\dstate = 4/5/5/6/7$).
  ```

### `arxiv-v2/sections/08_appendix.tex:89`

- Pattern: caption fragment.
- Original sentence:

  ```tex
  Zero-padded anchor: the causal wave's unconstrained arm against the rank-$\dmin$ target, mean $\pm$ sd over four seeds in every group.
  ```

- New sentence:

  ```tex
  The zero-padded anchor is the causal wave's unconstrained arm against the rank-$\dmin$ target, reported as mean $\pm$ sd over four seeds in every group.
  ```

### `arxiv-v2/sections/08_appendix.tex:113`

- Patterns: caption fragment and “for completeness.”
- Original sentence:

  ```tex
  Whole-matrix effective rank of every zero-padded razor cell (seed 0; $S_3$ mean $\pm$ sd over its four seeds), for completeness alongside Table~\ref{tab:wmrank}.
  ```

- New sentence:

  ```tex
  The table reports whole-matrix effective rank of every zero-padded razor cell (seed 0; $S_3$ mean $\pm$ sd over its four seeds), alongside Table~\ref{tab:wmrank}.
  ```

### `arxiv-v2/sections/08_appendix.tex:165`

- Pattern: passive experimental verification.
- Original sentence:

  ```tex
  The zero-padded grid's 30 cells were configuration-verified one-by-one against a manifest re-derived independently from the design record (steps, padding mode, and force-rank value per cell; zero skipped steps).
  ```

- New sentence:

  ```tex
  A manifest re-derived independently from the design record verified the zero-padded grid's 30 cells one-by-one (steps, padding mode, and force-rank value per cell; zero skipped steps).
  ```

### `arxiv-v2/sections/08_appendix.tex:218`

- Pattern: the banned word “robust” appeared outside a defined robustness
  check and overstated the qualified ordinal result.
- Original sentence:

  ```tex
  The ordinal law is therefore estimator-robust; the point estimate ``equals $\dmin$'' is a property of the entropy estimator on a near-flat spectrum with one weak direction, which stable rank penalizes; and the causal razor caps algebraic rank and is estimator-free.
  ```

- New sentence:

  ```tex
  The ordinal law is therefore unchanged across estimators; the point estimate ``equals $\dmin$'' is a property of the entropy estimator on a near-flat spectrum with one weak direction, which stable rank penalizes; and the causal razor caps algebraic rank and is estimator-free.
  ```

### `arxiv-v2/sections/08_appendix.tex:224`

- Pattern: one sentence combined the default metric, cross-check, and
  calibration rationale.
- Original sentence:

  ```tex
  \textbf{Primary and crosscheck degauging.} The design record's default pipeline treats scale-only degauging ($\hat{Q} = I$) as the primary metric, with the fitted-$(c, Q)$ Procrustes score retained as a robustness cross-check, after calibration on unconstrained checkpoints found $\hat{Q} \approx I$ empirically.
  ```

- New sentence(s):

  ```tex
  \textbf{Primary and crosscheck degauging.} The design record's default pipeline treats scale-only degauging ($\hat{Q} = I$) as the primary metric and retains the fitted-$(c, Q)$ Procrustes score as a robustness cross-check. This choice followed calibration on unconstrained checkpoints, which found $\hat{Q} \approx I$ empirically.
  ```

### `arxiv-v2/sections/08_appendix.tex:229`

- Pattern: sentence longer than 35 words carrying the broken equivalence,
  cause, pre-registered choice, cross-reference scope, and exclusion of the
  scale-only score.
- Original sentence:

  ```tex
  Under the force-rank grid's zero-padded target this equivalence breaks: the informative block's degenerate singular spectrum makes the scale-only fit's basis arbitrary, so the pre-registration for the causal wave specifically pins the fitted-$(c, Q)$ score, denoted crosscheck $\recninety$ throughout Section~\ref{sec:razor} and Table~\ref{tab:razor} and Figure~\ref{fig:razor}, as decisional; the scale-only score informs no reported conclusion.
  ```

- New sentence(s):

  ```tex
  Under the force-rank grid's zero-padded target, this equivalence breaks. The informative block's degenerate singular spectrum makes the scale-only fit's basis arbitrary. The pre-registration for the causal wave therefore pins the fitted-$(c, Q)$ score, denoted crosscheck $\recninety$ throughout Section~\ref{sec:razor}, Table~\ref{tab:razor}, and Figure~\ref{fig:razor}, as decisional. The scale-only score informs no reported conclusion.
  ```

### `arxiv-v2/sections/08_appendix.tex:335`

- Pattern: caption fragment.
- Original sentence:

  ```tex
  Convergence health (gate1a, min validation cosine over $L \in [2, 5]$) for every $S_3$/$S_5$ razor cell at the shortest pinned step budget.
  ```

- New sentence:

  ```tex
  The table reports convergence health (gate1a, min validation cosine over $L \in [2, 5]$) for every $S_3$/$S_5$ razor cell at the shortest pinned step budget.
  ```

### `arxiv-v2/sections/08_appendix.tex:254`

- Pattern: caption fragment.
- Original sentence:

  ```tex
  $S_3$ four-seed causal detail (crosscheck $\recninety$) behind the seed-mean confirmation in Section~\ref{sec:razor}; the own bar is $0.9 \times$ each seed's anchor.
  ```

- New sentence:

  ```tex
  The table reports the $S_3$ four-seed causal detail (crosscheck $\recninety$) behind the seed-mean confirmation in Section~\ref{sec:razor}; the own bar is $0.9 \times$ each seed's anchor.
  ```

### `arxiv-v2/sections/08_appendix.tex:304`

- Pattern: caption fragment.
- Original sentence:

  ```tex
  Four-seed causal detail (crosscheck $\recninety$) for the four groups extended under a pre-registration recorded before launch, on the same build, manifest, step pins, and decisional readout as seed 0.
  ```

- New sentence:

  ```tex
  The table reports four-seed causal detail (crosscheck $\recninety$) for the four groups extended under a pre-registration recorded before launch, on the same build, manifest, step pins, and decisional readout as seed 0.
  ```

### `arxiv-v2/sections/08_appendix.tex:316`

- Pattern: caption fragment.
- Original sentence:

  ```tex
  Own-bar clears: $S_4$ 4/4, $A_5$ 4/4, $S_5$ 2/4, $A_6$ 4/4.
  ```

- New sentence:

  ```tex
  The own-bar clears are $S_4$ 4/4, $A_5$ 4/4, $S_5$ 2/4, $A_6$ 4/4.
  ```

### `arxiv-v2/sections/05_causal_razor.tex:17`

- Pattern: figure-caption encodings left the group threshold and marker fills
  undefined outside the plot, and one sentence combined the purpose, values,
  line encodings, and marker encodings.
- Original sentence:

  ```tex
  Exact recovery is a step function at $\dmin$: per group, crosscheck $\recninety$ on held-out words at force-rank $k \in \{\dmin{-}1, \dmin, \dmin{+}1\}$, vs.\ the unconstrained anchor (dashed) and the $0.9\times$anchor bar (dotted).
  ```

- New sentence:

  ```tex
  Exact recovery is a step function at $\dmin$. For each group, the plot reports crosscheck $\recninety$ on held-out words at force-rank $k \in \{\dmin{-}1, \dmin, \dmin{+}1\}$. Dashed lines mark the unconstrained anchor, dotted lines mark the $0.9\times$anchor bar, and gray vertical lines mark the group threshold. Filled markers denote solvable groups; open markers denote non-solvable groups.
  ```

### `arxiv-v2/sections/05_causal_razor.tex:21`

- Pattern: the thin $S_3$ seed lines gained visible markers in the redesigned
  figure, so the caption's encoding was stale.
- Original sentence:

  ```tex
  $S_3$ overlays its four seeds (thin lines; bold mean); its below-$\dmin$ zero is unanimous.
  ```

- New sentence:

  ```tex
  $S_3$ overlays its four seeds (thin marked lines; bold mean); its below-$\dmin$ zero is unanimous.
  ```

### `arxiv-v2/sections/05_causal_razor.tex:22`

- Pattern: passive voice hid what the plot and table report.
- Original sentence:

  ```tex
  The other four groups are drawn at seed 0; their four-seed sufficiency verdicts are in Table~\ref{tab:razor}, where $S_5$ fails at $n = 4$.
  ```

- New sentence:

  ```tex
  The plot draws the other four groups at seed 0; Table~\ref{tab:razor} gives their four-seed sufficiency verdicts, and $S_5$ fails at $n = 4$.
  ```

### `arxiv-v2/sections/05_causal_razor.tex:43`

- Pattern: caption fragment.
- Original sentence:

  ```tex
  The causal razor on the zero-padded rank-$\dmin$ target (crosscheck $\recninety$ by force-rank $k$).
  ```

- New sentence:

  ```tex
  The table reports the causal razor on the zero-padded rank-$\dmin$ target (crosscheck $\recninety$ by force-rank $k$).
  ```

### `arxiv-v2/sections/05_causal_razor.tex:45`

- Pattern: a long caption sentence combined the source seed, bar origin, and
  boldface encoding; passive phrasing hid what fixes the bar.
- Original sentence:

  ```tex
  The anchor, $k$, and bar columns are seed 0, one seed per cell; the bar is $0.9\times$ the seed-0 anchor, fixed before any further seed ran, and bold there marks a seed-0 cell that clears it.
  ```

- New sentence(s):

  ```tex
  The anchor, $k$, and bar columns use seed 0, one seed per cell. The seed-0 anchor fixes the $0.9\times$ bar before any further seed ran. Bold marks a seed-0 cell that clears it.
  ```

### `arxiv-v2/sections/05_causal_razor.tex:51`

- Pattern: caption fragment.
- Original sentence:

  ```tex
  Exactly 0.000 one rank below $\dmin$ in all four seeds of every group (the forced ceiling is $\le 0.894$).
  ```

- New sentence:

  ```tex
  Recovery is exactly 0.000 one rank below $\dmin$ in all four seeds of every group (the forced ceiling is $\le 0.894$).
  ```

### `arxiv-v2/sections/05_causal_razor.tex:53`

- Pattern: caption fragment.
- Original sentence:

  ```tex
  Per-seed own-bar clears ($0.9\times$ each seed's own anchor; disclosed, not decisional): $S_3$ 2/4, $S_4$ 4/4, $A_5$ 4/4, $S_5$ 2/4, $A_6$ 4/4.
  ```

- New sentence:

  ```tex
  The per-seed own-bar clears ($0.9\times$ each seed's own anchor; disclosed, not decisional) are $S_3$ 2/4, $S_4$ 4/4, $A_5$ 4/4, $S_5$ 2/4, $A_6$ 4/4.
  ```

## Needs author

Entries here were left unchanged because a claim-preserving rewrite was not
mechanically certain.

### `arxiv-v2/main.tex:70`

- The abstract's empirical sentence carries four independent claims, with
  N4, N5, and N12 comments embedded between clauses and N1/N3/N16 after the
  sentence. Splitting it would change which sentence each comment follows, so
  only the promotional parenthetical was removed and the sentence structure
  remains unchanged.

### `arxiv-v2/sections/01_intro.tex:24`

- The phrase “almost perfectly” remains unchanged. Replacing its qualitative
  degree without importing the later numerical correlation could strengthen
  or soften the claim.

### `arxiv-v2/sections/01_intro.tex:21`

- “The headline is” was replaced locally, but the sentence still combines the
  observational, equivalence, necessity, and sufficiency claims. The N4/N5
  comment divides its clauses and the N1/N12/N16 comment follows the sentence;
  splitting it would change the evidence attachment, so the sentence remains
  multi-claim.

### `arxiv-v2/sections/04_ranklaw_observed.tex:40`

- “The dimension-matched pair $S_4$/$A_5$ is tested directly in
  Section~\ref{sec:equivalence}” is a navigation-only restatement. Deletion is
  cleaner than a synonym-level rewrite, but the scrub instructions authorize
  sentence rewrites rather than claim deletion, so it remains unchanged.

### `arxiv-v2/sections/04_ranklaw_observed.tex:28`

- The correlation sentence combines the observed coefficient, tie cap,
  permutation null, and pre-registered interpretation. Two N4 comments divide
  its clauses. Splitting it would change which sentence each comment follows,
  so it remains unchanged.

### `arxiv-v2/sections/05_causal_razor.tex:67`

- The necessity sentence carries the analytical ceiling, the threshold
  consequence, and the observed fraction of that ceiling. One
  `% <!-- evidence: N12 -->` line follows the full sentence, so it remains
  unchanged rather than redistributing the evidence tag.

### `arxiv-v2/sections/05_causal_razor.tex:82`

- The sentence carries the $S_3$, $S_4$, $A_5$, $A_6$, and $S_5$ outcomes,
  with `% <!-- evidence: N3 -->` embedded after the $S_3$ clause and
  `% <!-- evidence: N16 -->` following the final clause. Splitting it anywhere
  else would create a numerical sentence without its original evidence
  attachment, so it remains unchanged.

### `arxiv-v2/sections/05_causal_razor.tex:55`

- The starred caption sentence combines the $S_3$ marginality trigger and
  first extension with the separate extension of four groups. The caption has
  one evidence comment after its final sentence, so splitting this numerical
  sentence would change that attachment; it remains unchanged.

### `arxiv-v2/sections/05_causal_razor.tex:100`

- “Table ... adds” announces the display, and the sentence combines a
  whole-matrix measurement with its interpretation. One
  `% <!-- evidence: N15, N17 -->` line follows the sentence, so it remains
  unchanged rather than separating the numerical claim from its evidence.

### `arxiv-v2/sections/08_appendix.tex:46`

- The sentence combines the in-band result, Spearman result, and matched-pair
  difference. Splitting it would leave the one `% <!-- evidence: N4 -->` line
  detached from some numerical claims, so it remains unchanged.

### `arxiv-v2/sections/08_appendix.tex:57`

- The sentence combines the $S_4$/$S_5$ numerical inversion with an
  explanation carrying both “plausibly” and “unconfirmed hypothesis.”
  Splitting it would detach `% <!-- evidence: N14 -->` from the numerical
  observation, while removing either hedge would alter uncertainty. It remains
  unchanged.

### `arxiv-v2/sections/08_appendix.tex:124`

- The centering sentence combines the uncentered-covariance defect and the
  production correction. The paragraph's one `% <!-- evidence: N11 -->` line
  follows the later numerical sentence; increasing the sentence count would
  further separate the mechanism from that evidence tag, so it remains
  unchanged.

### `arxiv-v2/sections/08_appendix.tex:128`

- “Flawless synthetic model” is promotional, but replacing “flawless” with
  “exact” could assert a stronger construction property not stated here. The
  sentence remains unchanged.

### `arxiv-v2/sections/08_appendix.tex:132`

- The length-robustness sentence combines the split definition, the reason for
  excluding $L = 1$, and the measured effects. Splitting it would leave the
  one `% <!-- evidence: N10 -->` line detached from numerical claims. The
  defined term and sentence remain unchanged.

### `arxiv-v2/sections/08_appendix.tex:144`

- The ambient-identity-tax sentence carries the mechanism, residual rank
  budget, interpretation, and first raw-artifact signature. One
  `% <!-- evidence: N7 -->` line covers its numerical and mechanistic clauses;
  splitting it would detach the tag, so it remains unchanged.

### `arxiv-v2/sections/08_appendix.tex:155`

- The sentence combines independent raw-artifact signatures (ii) and (iii),
  covered by one `% <!-- evidence: N1 -->` line. It remains unchanged rather
  than guessing how the evidence tag should be distributed.

### `arxiv-v2/sections/08_appendix.tex:170`

- The observational-band paragraph contains a multi-claim sentence about the
  lens, restricted state shape, and estimator range, followed by one
  `% <!-- evidence: N4 -->` line for the paragraph. Splitting it would detach
  that tag from mathematical claims, so it remains unchanged.

### `arxiv-v2/sections/08_appendix.tex:193`

- The necessity sentence combines the analytical interpretation with the
  empirical optimization evidence. The paragraph's one
  `% <!-- evidence: N1, N2, N12 (U3) -->` line follows a later sentence;
  increasing the sentence count would further separate these numerical claims
  from that tag, so it remains unchanged.

### `arxiv-v2/sections/08_appendix.tex:210`

- The first estimator-dependence sentence carries the estimator definition,
  five estimates, band comparison, and ordering result. The paragraph's one
  `% <!-- evidence: N18 -->` line follows the next sentence; increasing the
  sentence count would further separate the numerical claims from that tag,
  so this sentence remains unchanged.

### `arxiv-v2/sections/08_appendix.tex:217`

- “Estimator-robust” was replaced locally, but the sentence still combines the
  ordinal result, the entropy-estimator point estimate, and the estimator-free
  causal razor. The paragraph's one `% <!-- evidence: N18 -->` line follows
  this sentence; splitting it would change the evidence attachment, so the
  three clauses remain together.

### `arxiv-v2/sections/08_appendix.tex:253`

- The $S_3$ per-seed caption combines the four-seed zero result, individual
  clear counts, fixed-bar decision rule, and a counterfactual recomputed bar.
  One `% <!-- evidence: N2, N3 -->` line covers all of these numerical claims,
  so the caption remains unchanged rather than guessing how to distribute the
  evidence tag.

### `arxiv-v2/sections/08_appendix.tex:303`

- The four-group per-seed caption combines 16 necessity cells, four fixed-bar
  group verdicts, a self-referential-bar comparison, and own-bar clear counts.
  One `% <!-- evidence: N16 -->` line covers the caption, so its long sentences
  remain unchanged rather than detaching that tag from any numerical claim.

### `arxiv-v2/sections/07_limitations.tex:3`

- The sentence combines the four-seed status, $S_5$ failure, convergence
  condition, anchors, and limitation hedge. Splitting it would leave the one
  `% <!-- evidence: N16 -->` line detached from one of its numerical claims;
  the sentence remains unchanged.

### `arxiv-v2/sections/08_appendix.tex:161`

- “The sweep verdict was registered” is passive, but the source does not name
  the actor who registered it. Supplying one would add provenance not present
  in the manuscript, so the sentence remains unchanged.

### Author-level result provenance

- Base commit `cbc2874` records a later $S_5$ 20k rerun that confirms at four
  seeds, while this paper consistently reports the earlier pre-registered 8k
  failure and narrows sufficiency to four of five groups. This scrub does not
  alter that claim. The author should confirm that the later-budget follow-up
  remains intentionally outside the paper's decisional result.

## Figures

### `fig1_convergence.pdf`

#### Purpose test

- Before, without the caption: restricted effective rank clusters near
  $d_{\min}$ for all five groups, while the matched $S_4$/$A_5$ differences
  appear to fall inside the shaded equivalence region.
- Caption and referencing-text purpose: recruited rank follows representation
  dimension rather than solvability; all 19 seeds lie in the pre-registered
  band, and the matched $S_4$/$A_5$ comparison lands inside the
  $\pm 0.5$ rank-unit equivalence margin with mean difference $+0.019$.
- Before verdict: the main panel passed, but the equivalence evidence was not
  readable at the figure's 0.58-text-width placement, so the full figure
  failed the purpose test at final size.
- After, without the caption: all five group clusters track the identity
  within the shaded band, and every vertically separated $S_4$ seed
  difference lies inside the explicitly labelled equivalence region near the
  zero and mean reference lines.
- After verdict: pass. The plot, caption, and referencing paragraph state the
  same observational and matched-dimension claims.

#### Readability audit

- The 7 pt main tick and legend type scaled to about 6.2 pt in the paper; the
  6 pt inset type scaled to about 5.3 pt.
- The low-right inset was too small to read at print size. Its five seed marks
  shared one y-coordinate and could merge.
- The zero-difference and mean $+0.019$ lines nearly coincided without direct
  labels, and the inset had no quantity-and-unit axis label.
- Seeds within a group shared one x-coordinate, so close values could merge.
- The identity line and band were defined in the legend, but the inset's
  reference meanings depended on its compressed title and the caption.

#### Generator edit and verification

- Script: `papers/unireps-ea/figures/figure_gen.py`, confined to
  `fig1_convergence`. The edit replaces the inset with a full-width lower
  panel, separates its five seed marks vertically, uses presentation-only
  horizontal dodge for coincident main-panel seeds, labels quantities and
  units, labels all lower-panel references in the plot, uses a two-column
  shared main legend, and saves at the paper's exact 5.5-inch width with a
  minimum explicit type size of 7.25 pt.
- The LaTeX placement changed from `0.58\textwidth` to `\textwidth`; this is a
  permitted presentation resize, not a manuscript number or result change.
- `git diff --unified=0` is confined to the drawing function. `SOURCE_MD5`,
  `_load`, `_ranks`, raw paths, loaded fields, and the mean-difference
  expression are byte-identical to `origin/main`.
- Before and after both plot 24 data marks in 10 data artists: 19 rank values
  in five group scatters and five matched-difference values in five one-point
  seed traces. Both retain two shaded bands and three reference lines (the
  main identity, lower zero, and lower observed mean). No point, seed, group,
  reference, band, or annotation value was dropped or aggregated, and no
  quantitative axis range was tightened to hide a point.
- Regeneration command:

  ```sh
  /Users/samuellarson/Experiments/learned-representations/.venv/bin/python papers/unireps-ea/figures/figure_gen.py --out <temporary-directory> --repo .
  ```

- The committed 200 dpi pair is
  `arxiv-v2/figures/scrub-verification/fig1_convergence-before.png` and
  `arxiv-v2/figures/scrub-verification/fig1_convergence-after.png` (721x603
  before; 1100x710 after).

### `fig1_razor_step.pdf`

#### Purpose test

- Before, without the caption: each displayed trajectory is zero below its
  group's $d_{\min}$ and rises at $d_{\min}$, with $S_3$ visibly carrying
  multiple trajectories.
- Caption and referencing-text purpose: the four-seed $S_3$ overlay and the
  seed-0 traces for the other four groups show the causal step; the table,
  rather than the four single-seed panels, carries the four-seed sufficiency
  verdict in which $S_5$ fails at $n = 4$.
- Before verdict: fail. The four non-$S_3$ panel titles did not disclose that
  they showed only seed 0, so the picture appeared broader than the stated
  four-seed verdict.
- After, without the caption: the $S_3$ panel explicitly shows four marked
  seed traces plus their mean, each other panel explicitly says “seed 0
  shown,” and all five panels show zero below the labelled $d_{\min}$
  threshold followed by recovery at the threshold.
- After verdict: pass. The figure now limits its visible claim to the plotted
  seed scope and points to the table for the four-seed group verdicts.

#### Readability audit

- The 7 pt tick and legend type scaled to about 5.6 pt at the paper's
  text-width placement.
- Five panels in one shallow row compressed titles, ticks, and S3's four thin
  seed traces; those raw seed traces had no markers and merged visually.
- The four non-$S_3$ titles omitted their seed-0 scope.
- The gray vertical $d_{\min}$ reference had no key, and the legend did not
  define the filled/open solvability encoding.
- The legend occupied the last data panel, competing with the $A_6$ marks and
  reference lines.

#### Generator edit and verification

- Script: `papers/neurreps-ea/figures/figure_gen.py`, confined to
  `fig1_razor_step`. The edit uses a 2-by-3 layout, places one shared key in
  the unused panel, adds markers to every S3 seed trace, preserves filled
  markers for solvable groups and open markers for non-solvable groups,
  labels seed scope in every title, and labels both axes with quantities and
  units. The PDF is saved at the paper's exact 5.5-inch width with a minimum
  explicit type size of 7.25 pt.
- `git diff --unified=0` is confined to plotting and layout calls inside
  `fig1_razor_step`. `SOURCE_MD5`, `_load`, `_razor_cell`, raw paths, loaded
  fields, `arms`, per-seed values, mean computation, anchor computation, and
  bar computation are byte-identical to `origin/main`.
- Before and after both plot 27 data points in nine data series: four $S_3$
  seed traces, one $S_3$ mean, and one seed-0 trace for each of the other four
  groups. Both retain 15 reference lines: five anchors, five anchor-relative
  bars, and five $d_{\min}$ thresholds. The legend has two empty reference
  proxies before and five presentation-only proxies after; the three added
  proxies define the threshold and filled/open marker encodings and carry no
  data. No point, seed, series, group, reference, or value was dropped,
  aggregated, smoothed, clipped, or reordered, and no quantitative axis range
  was tightened to hide a point.
- Regeneration command:

  ```sh
  /Users/samuellarson/Experiments/learned-representations/.venv/bin/python papers/neurreps-ea/figures/figure_gen.py --out <temporary-directory> --repo .
  ```

- The committed 200 dpi pair is
  `arxiv-v2/figures/scrub-verification/fig1_razor_step-before.png` and
  `arxiv-v2/figures/scrub-verification/fig1_razor_step-after.png` (1376x321
  before; 1100x650 after).

### Paper-level figure check

- Tectonic 0.16.9 rebuilt the 11-page paper with 2 figures and 8 tables, no
  undefined references, and no unresolved citations.
- Pages 3 and 4 were rendered from the rebuilt paper at 200 dpi and inspected
  at final placement. Both figures are legible, their legends and annotations
  do not cover data, all displayed values remain visible within the retained
  quantitative ranges, and the plot/caption/text purposes agree.

## Follow-up aesthetic refinements

### Figure 1 label clearance

- A final reader review found that several direct labels sat on the identity
  or grid lines even though they did not cover data. Inside
  `papers/unireps-ea/figures/figure_gen.py::fig1_convergence`, the five group
  labels now occupy clear space above or below their markers. Both panel grids
  are suppressed so they cannot cross the legend or lower-panel annotation;
  the quantitative reference lines remain unchanged.
- The lower-panel band label is shortened to
  `$\pm 0.5$ equivalence margin (shaded)` and placed wholly to the right of
  the zero and observed-mean lines. The value and meaning are unchanged.
- The edit changes only annotation positions, annotation wording, and grid
  presentation. The figure still has 24 data marks in 10 data artists, two
  bands, and three reference lines; no datum, statistic, axis limit, or
  reference value changed. The 200 dpi `fig1_convergence-after.png` was
  regenerated from the revised PDF.

### Equal table-caption spacing

- The NeurIPS table wrapper assumed captions above their tabular bodies. With
  this paper's below-table captions, it produced a 0 pt body-to-caption gap and
  a 7 pt after-caption gap. A table-only override in `arxiv-v2/main.tex` now
  uses the venue style's 7 pt caption gap on both sides.
- Float placement, `\intextsep`, and `\textfloatsep` are unchanged. All eight
  tables retain the same body, caption text, numbers, labels, and evidence
  comments. At 200 dpi, pages 4, 7, 8, 10, and 11 were checked after the
  spacing change; no table, caption, or following text collides or clips.
- Tectonic 0.16.9 still produces an 11-page paper with 2 figures and 8 tables,
  no undefined references, and no unresolved citations.
