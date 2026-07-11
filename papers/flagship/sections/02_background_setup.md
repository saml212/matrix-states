# 2 Background and Experimental Setup

## 2.1 Delta-Rule Fast-Weight States

A delta-rule layer maintains a matrix state $S \in \mathbb{R}^{d \times d}$
updated per token as
$S_t = S_{t-1}(I - \beta_t k_t k_t^{\top}) + \beta_t v_t k_t^{\top}$,
with keys $k_t$, values $v_t$, and write strength $\beta_t$ produced from
the token stream; reads are matrix-vector products $S_t q_t$. The update
first erases the value currently bound to $k_t$, then writes $v_t$ at
$k_t$: an associative store with in-context writes. Because outer-product
writes confine $S$ to the span of the observed keys and values, the state's
effective dimensionality and the geometry of the key population are both
measurable objects, and both are studied here.

## 2.2 Model Families

Three model families appear, one per result leg. (i) The *matrix-state
encoder* family (Section 3): a transformer encoder that emits a single
$d_{state} \times d_{state}$ matrix $Z$ per input word under a hard
single-state bottleneck; the decoder reads only $Z$, never the raw
inputs, so the task load must reside within the one matrix. (ii) The
*two-layer delta-rule contender* (Section 4): a DeltaNet-family language
model with $d_{model}=256$, two blocks, $d_{state}=64$
<!-- evidence: R0 -->, and its two matched baselines, described in
Section 4.1. (iii) *Delta-rule language models* at scale (Section 5):
the same architecture family trained as causal LMs at 14M, 98M, 392M,
and 1.31B parameters <!-- evidence: R0 --> on held-fixed data mixes.

## 2.3 Tasks

**Group word problems** (Section 3). A word $w = g_{i_1} \cdots g_{i_L}$
is drawn over a symmetric generating set of a permutation group $G$; the
target is the image of the composed element under a fixed faithful
orthogonal representation. Five groups are used: $S_3$, $S_4$, $A_5$,
$S_5$, $A_6$, with minimal faithful real representation dimensions
$d_{\min} = 2, 3, 3, 4, 5$ <!-- evidence: R1 -->. The pair
$S_4$/$A_5$ is the designed dissociation control: equal $d_{\min}$,
opposite solvability. Train words have length $L \le 8$; held-out
evaluation words are fresh draws.

**Episodic recall** (Section 4). Each episode binds $K{=}32$
key-entity/value-entity pairs (real GPT-2 token identities,
injectivity-guarded) over a 224-token bind phase, then queries one key
<!-- evidence: R0 -->. The metric of record, $\mathrm{acc}_A$, is
episode-restricted 32-way top-1 accuracy at the answer position, read
through each model's own LM-head route; chance is $1/32 = 0.03125$ and
the pre-registered demonstration bar is three times chance, 0.09375
<!-- evidence: R0 -->. Every $\mathrm{acc}_A$ number in this paper
carries the same caveat: recall here means episode-restricted top-1
retrieval under argmax decoding, and under argmax a rank-one state can
support on the order of $d$ associations (Nichani et al., 2024); no
rank or continuous-capacity claim is made from $\mathrm{acc}_A$.

**Language modeling at scale** (Section 5). Causal LMs are trained on
two fixed data mixes (a reasoning mix built on OpenR1-Math and a
narrative mix built on WikiText-103, each with a fixed augmentation
source), held identical across the parameter ladder so that scale is
the only moving axis.

## 2.4 Instruments

**Restricted effective rank** (Section 3). States $Z(w)$ for held-out
words are centered, the dominant $d_{\min}$-dimensional subspace $U$ is
taken from the SVD of the centered second-moment matrix, and effective
rank (the entropy-based Roy-Vetterli measure) is computed on
$U^{\top} Z(w) U$; fitting and evaluation use disjoint word splits.
Centering is load-bearing: without it, an ambient identity block
masquerades as signal.

**Force-rank arms** (Section 3). At train time, $Z$ is projected to rank
$k$ by SVD truncation inside the forward pass, for
$k \in \{d_{\min}-1, d_{\min}, d_{\min}+1\}$; recovery is measured as
the fraction of held-out words whose Procrustes-aligned cosine to the
target exceeds 0.9 ($\mathrm{xrec90}$).

**State zeroing and probe taps** (Section 4). Causal attribution zeroes
one layer's cached state before the query pass and re-measures
$\mathrm{acc}_A$; representational legibility is measured by ridge
probes at four taps (three state-level, one post-block pre-LM-head),
reported as the recovered fraction at cosine 0.9 ($\mathrm{rf@0.9}$)
against a shuffled control.

**Span fraction** (Section 5). For a population of $K$ normalized write
keys $E$, the raw instrument is the key-Gram deviation
$\lVert E^{\top}E - I \rVert_F$; span fraction rescales it between two
analytic anchors, the expectation for $K$ i.i.d. random unit vectors
(value 0) and full collapse onto one direction (value 1)
<!-- evidence: R0 -->, making readings comparable across state widths.
Higher is closer to collapse.

**Confidence intervals.** Paired confidence intervals in this paper
are Student-$t$ intervals on $n{=}3$ per-seed paired differences (two
degrees of freedom, $t_{.975} = 4.303$) <!-- evidence: R4 -->, except
the nine-seed diagnosis pool of Section 4.5, stated there. Per-seed
values are tabulated in Appendix B so any alternative construction can
be applied; the verdicts that rest on these intervals clear their
margins by multiples of the interval width, and the one
threshold-adjacent reading claimed as a trend (Section 5.2) is reported
as unconfirmed.

## 2.5 Instrument Validity as a First-Class Method

Each leg of this program survived at least one
instrument-defect-found-and-fixed round; only repaired readings appear
below. The rank program's first causal sweep was voided by a
target-padding defect that let rank-capped models buy cosine from an
ambient identity block; the diagnosis, fix, and re-run are described
in Section 3.3. The recall program lost three consecutive probe rounds
to a wrong-layer, wrong-route linear instrument before the repaired
tap of Section 4.3 read what the model's own forward pass had decoded
all along. A third case came from a separate composition program (not
claimed here): a primary lens read zero on converged models while a
pre-registered crosscheck lens read one, the contradiction was
resolved by a recorded tiebreak against the raw artifacts, and the
affected endpoint is excluded from this paper entirely. Every verdict-bearing instrument here therefore carries an analytic
anchor or a negative control, with a pre-registered crosscheck wherever
a verdict hangs on a single metric.
