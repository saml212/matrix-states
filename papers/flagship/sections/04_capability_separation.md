# 4 Capability Separation

This section reports a head-to-head comparison designed, margined, and
frozen before its confirmatory sweep ran: a two-layer delta-rule model
against a parameter-matched vector-state ablation and a compute-matched
transformer, on single-pass episodic recall.

## 4.1 Arms, Matching, and Frozen Margins

The contender is a two-block DeltaNet-family LM ($d_{model}=256$,
$d_{state}=64$, 14,048,896 parameters <!-- evidence: R0 -->). The
ablation keeps the embedding table, output head, and feed-forward
blocks identical and replaces only the fast-weight mixer with an
elementwise-gated vector recurrence at the same $d_{state}$
(14,048,384 parameters, a 0.004 percent difference
<!-- evidence: R0 -->); it is constructed as a recurrence, not a
reshape, so matrix structure and not merely parameter count is the
manipulated variable. The transformer baseline is a two-layer pre-norm
decoder with rotary position embeddings and the contender's own MLP
class, FLOP-matched to the contender within 5 percent
<!-- evidence: R0 -->; its learning rate was selected on a calibration
cell from a three-point grid and frozen at $10^{-3}$ before the sweep
<!-- evidence: R0 -->. All arms train 20,000 steps at matched tokens
with seeds identical across arms per cell <!-- evidence: R0 -->.
Verdict tiers, margins, and the primary task were frozen and checksummed
before the confirmatory sweep launched (a WIN requires the paired
per-seed difference's 95 percent confidence interval to exclude 0.30);
the sweep's evaluation applied the audited round protocol verbatim to
its 18 checkpoints with checksum-pinned loading and a fixed evaluation
episode set <!-- evidence: R4 -->.

## 4.2 The Verdict at $n{=}3$

Table 1 (Figure 3, left) gives the per-seed metric of record.

| arm | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| contender | 0.99951 | 1.00000 | 0.99902 | 0.99951 |
| vector-state ablation | 0.03223 | 0.03271 | 0.03687 | 0.03394 |
| transformer | 0.02710 | 0.02930 | 0.02856 | 0.02832 |

<!-- evidence: R4 (whole table) -->

Every contender seed exceeds the demonstration bar by a factor of at
least 10.7; neither baseline clears it in any seed
<!-- evidence: R4 -->. The paired contender-minus-ablation difference is
0.96558 with 95 percent confidence interval (0.95822, 0.97293), and the
contender-minus-transformer difference is 0.97119 with interval
(0.96855, 0.97383); both intervals exclude the frozen 0.30 margin, the
first by more than three times the margin at its floor
<!-- evidence: R4 -->. The pre-registered extension trigger (any seed
below bar, or any interval straddling the margin) did not fire
<!-- evidence: R4 -->.

Two caveats are part of the claim. First, the Nichani caveat stated in
Section 2.3: recall means episode-restricted top-1 retrieval under
argmax decoding, under which a rank-one state can support on the order
of $d$ associations; nothing here is a rank or continuous-capacity
claim. Second, the matched-budget caveat: the transformer read holds
only at this matched parameter count, token budget, and training
compute. Because the transformer sits below its own demonstration bar,
it is recorded as a degenerate-baseline datum, and the separation
verdict is carried by the ablation comparison with the transformer's
non-competitiveness disclosed alongside <!-- evidence: R4 -->.

## 4.3 The Capability Is Fast-Weight-Resident and Nonlinearly Stored

Causal attribution by state zeroing localizes the capability. On the
frozen round checkpoint, zeroing the first block's state before the
query collapses contender recall from 0.9990 to 0.0286, at chance,
while zeroing the second block's state leaves it at 0.9990, unchanged
<!-- evidence: R5 -->. The sweep replicates this on all fresh seeds:
first-state zeroing reads 0.0339, 0.0012, and 0.0002 against the 0.09375
bar, and the hard-stop criterion is clean in 12 of 12 recurrent cells
<!-- evidence: R4 -->. The binding lives in $S_0$; $S_1$ is causally
inert for this task.

Where the content becomes readable is a separate question with an
instructive answer. Ridge probes at three state-level taps, including
directly on the causally load-bearing $S_0$, recover nothing at the
strict threshold ($\mathrm{rf@0.9} = 0.0$ at every state-level tap, with
shuffled-control gaps of at most 0.063) <!-- evidence: R5 -->. The same
probe at the post-block-1, pre-LM-head hidden state reads
$\mathrm{rf@0.9} = 0.674$ with mean cosine 0.894 <!-- evidence: R5 -->.
The ablation's pre-LM-head tap, its own positive control, fails
($\mathrm{rf@0.9} = 0.0$, cosine 0.119) <!-- evidence: R5 -->: its
geometry differs in kind, not merely in strength. The stored binding is
therefore real but linearly illegible at the state; it becomes linearly
decodable only after the downstream block's nonlinear processing. Three
earlier probe rounds that read uniform zeros were a wrong-layer,
wrong-route instrument, and we note that a state-level linear probe
would have falsified this capability that the model's own forward pass
decodes at 0.9995.

## 4.4 Constant-Memory Recall Horizon

The contender's two states occupy $2 \times 64 \times 64$ float32
entries, 32,768 bytes, independent of context length
<!-- evidence: R0 -->. Holding those bytes fixed, recall does not decay
with distance: at horizons of 2, 4, and 8 times the bind phase (454,
902, and 1798 tokens), every seed reads $\mathrm{acc}_A \ge 0.998$
<!-- evidence: R10 -->. The pre-registered memory-matched comparison
caps the transformer's KV cache at $M$ times the contender's state
bytes with sink-plus-FIFO eviction, $M \in \{1, 2, 4, 8, 16, 32\}$;
every capped read at the decision horizon lies between 0.020 and 0.033,
at or below chance, as does the uncapped read <!-- evidence: R10 -->.
Every per-$M$ paired-gap confidence interval has a floor of at least
0.958 against the 0.20 crossover margin <!-- evidence: R10 -->. Because
the uncapped transformer itself fails the demonstration bar, the
pre-registered degenerate-baseline clause fires: the verdict of record
is that the baseline is non-competitive at matched parameters and
tokens, not a strongest-possible-baseline result, and no
crossover-point value is certified <!-- evidence: R10 -->. The two
informative readings are the contender's own flat horizon table and the
answer to the forced-locality question: capping never helps the
transformer <!-- evidence: R10 -->.

## 4.5 Scope Disclosure: the Multi-Hop Task

A second task in the frozen registry, two-hop compositional recall, is
verdict-INDETERMINATE and is not claimed. Its data are disclosed
because they constrain interpretation: one of three contender seeds
clears the demonstration bar (0.33447; the other two and all baseline
seeds read chance), the paired interval straddles the margin
<!-- evidence: R4 -->, the bar-clearing seed's partial recall does not
generalize to held-out hop depths (0.0112) <!-- evidence: R4 --> and
collapses to 0.010 at every extended horizon <!-- evidence: R10 -->.
The single bar-clearing seed shows the failure mode is at least partly
trainability variance rather than a hard capability boundary; a
pre-registered diagnosis round is in flight and nothing beyond
single-hop recall is claimed here.

**Figure 3 caption.** Left: episodic recall $\mathrm{acc}_A$ per seed
and arm at matched parameters, tokens, and compute ($K{=}32$ bindings;
chance 0.03125, dashed; demonstration bar 0.09375, dotted). The
contender reads 0.99902 to 1.00000; both baselines sit at chance in
every seed; paired intervals exclude the frozen 0.30 margin. Right:
recall versus context horizon on a fixed 32,768-byte contender state
(454 to 1798 tokens, all seeds at or above 0.998) against the
KV-capped transformer grid ($M \in \{1..32\}$ times the contender's
state bytes, all reads 0.020 to 0.033). Recall means episode-restricted
argmax retrieval (Nichani caveat, Section 2.3); the transformer is a
degenerate-baseline datum at this matched budget.

**Figure 4 caption.** Storage localization and legibility for the
contender and its vector-state ablation. Left: $\mathrm{acc}_A$ with
both states intact, first state zeroed, and second state zeroed;
first-state zeroing collapses the contender to chance (0.9990 to
0.0286) while second-state zeroing changes nothing. Right: linear-probe
recovered fraction at cosine 0.9 for three state-level taps and the
pre-LM-head tap; only the contender's pre-LM-head tap reads above zero
(0.674), so the binding stored in $S_0$ is linearly legible only after
downstream nonlinear processing.
