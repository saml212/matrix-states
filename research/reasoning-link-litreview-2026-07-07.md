# REASONING-LINK literature validation (research agent pass, 2026-07-07)

Verdict: **GO-WITH-REFRAME.** No outright scoop found. Novelty rests on the
specific combination (K-cycle hop-recovery probe transplanted onto
already-trained checkpoints + controlled 14M→1.31B fixed-state ladder +
param-matched full-attention control + capacity-law framing + causal
intervention angle), NOT on "a capacity cliff exists" or "scale ladder with
attention control" as bare facts. Confidence markers from the research agent:
[VERIFIED-READ] = fetched actual paper text/PDF; [SEARCH-SNIPPET] = search
summaries only; [LOW-CONFIDENCE-FETCH] = inconsistent across fetch passes.

## The three papers the design MUST cite and defend against

1. **Variational Linear Attention (VLA), arXiv:2605.11196 [VERIFIED-READ ×2,
   minor cross-fetch numeric inconsistency]** — nearest empirical neighbor to
   the cliff claim. Shows a genuinely sharp transition in MQAR recall at
   n_pairs/d_h≈1 with their stabilized architecture, and DeltaNet (their
   baseline) collapsing sharply around ratio ~0.25–0.5 (exact numbers
   inconsistent across fetches — treat as approximate). Their stabilization
   (Kalman/variational, Sherman-Morrison update, write-direction normalized so
   recurrence Jacobian spectral norm = 1) pushes the cliff outward to ratio≈1,
   62.3% accuracy exactly at n=d_h. **They frame capacity as tied to key
   linear independence / state-matrix rank — a GEOMETRIC account in direct
   tension with our §14 coherence-exoneration result (frozen doses to 0.40
   flat at h4=1.0, both structures).** Engage this tension head-on: their
   task is single-hop MQAR recall, ours is relational binding + multi-hop
   composition with exact-continuous readout; our dose-response was a direct
   causal manipulation of coherence at matched load. Do not let a reviewer
   discover the tension first.

2. **"Attention Retrieves, MLP Memorizes" (Frozen-QK), arXiv:2506.01115
   [VERIFIED-READ]** — nearest neighbor for the frozen-key intervention.
   Freezes the ENTIRE Q/K projections at init on standard softmax
   transformers; induction heads still form; but has a REAL loss cost
   (Wikitext PPL 3.07 vs 2.78, ~10%). Distinguish on three axes: (a) additive
   frozen bias blended at tuned λ, not full-freeze; (b) loss-NEUTRAL
   (rung-1's own CI result); (c) recurrent-state target, not softmax.

3. **Okpekpe & Orvieto, "Revisiting Associative Recall in Modern Recurrent
   Models", arXiv:2508.19029 [VERIFIED-READ, PDF pp.1-6]** — the credible
   RIVAL CAUSAL ACCOUNT: "Transformers differ from SSMs not in expressive
   power but in their optimization dynamics." Mamba/Hyena MQAR success is
   gated by a narrow LR window; with fine LR grids, Mamba solves MQAR at
   sequence lengths >> hidden size. **A reviewer can claim any
   capacity-cliff-shaped result is an optimization artifact.** Our rebuttal
   is the causal chain: force-rank + eval-time SVD truncation land the
   ceiling exactly at k=K (post-hoc structural manipulation, not a training
   hyperparameter) — the design/paper must make this rebuttal EXPLICITLY.

## Scoop risk — ADJUDICATED 2026-07-07 (orchestrator follow-up pass)

- **"From Recall To Reasoning: Understanding the Role of Associative Memory
  in Hybrid Architectures" (OpenReview R8ZbLi3oUv).** Full read still
  blocked (OpenReview bot-check defeats both the forum page and api2
  endpoint; no arXiv mirror found in two targeted searches). But richer
  search snippets recovered the design: trains HYBRID attention+recurrent
  models (Mamba/GLA/Gated-DeltaNet classes) at 150M/500M on a unified
  math-reasoning curriculum, sweeps ATTENTION-LAYER COUNT, applies
  majority-voting test-time scaling, categorizes errors via LLM-as-a-Judge
  into 8 math-education-inspired types, identifies in-context associative
  recall as a key error factor. **Adjudication: thematically convergent,
  methodologically distant from Leg B** — no load/capacity axis, no causal
  or representation-level intervention, no hop-depth generalization, no
  probe-based state readout, correlational benchmark-level evidence.
  DOWNGRADED from scoop risk to CONVERGENT-MOTIVATION CITATION ("recall
  bottlenecks reasoning in recurrent-family models" — independent support
  for REASONING-LINK's premise). Residual uncertainty disclosed: this is
  snippet-level, not a full read; re-attempt the full read before the
  paper's related-work section is finalized (not build-blocking).
- Adjacent papers surfaced in the same pass, for the related-work sweep:
  arXiv:2510.26912 (Understanding and Enhancing Mamba-Transformer Hybrids
  for Memory Recall and Language Modeling), arXiv:2604.21454 (Reasoning
  Primitives in Hybrid and Non-Hybrid LLMs — state-tracking and recall;
  re-fetch, earlier fetch was low-confidence), arXiv:2507.06457 (A
  Systematic Analysis of Hybrid Linear Attention).

## Other anchors to fold into related work

- **Zoology (arXiv:2312.04927) [VERIFIED-READ]** — MQAR origin; fixed-size
  recurrent state as bottleneck; no cliff/ratio framing, no rank spectra.
  Already in references.md.
- **Based (arXiv:2402.18668) [VERIFIED-READ]** — Theorem 3.1: any recurrent
  model needs Ω(N) bits of state for MQAR ("within and across architecture
  classes"). Also ALREADY runs a 360M/1.3B ladder with param-matched
  Transformer++ control — the ladder+control PATTERN is theirs; cite as
  methodology precedent.
- **Jelassi et al., "Repeat After Me" (arXiv:2402.01032) [VERIFIED-READ]** —
  GSSM copying capped by fixed state (proven + empirical).
- **Sanford, Hsu & Telgarsky (arXiv:2402.09268) [SEARCH-SNIPPET ×2]** —
  k-hop induction-head task; Θ(log k) depth. Anchor citation for the
  hop-task design.
- Hop-depth generalization is the standard 2026 paradigm on
  standard/looped transformers (arXiv:2601.21214, 2604.07822, 2603.21676,
  2605.26789 [all SEARCH-SNIPPET]) — **nobody found running it on
  DeltaNet/Mamba-family fixed-state models; that combination is open.**
- **Hewitt & Liang 2019 [canonical]** — probe control tasks; our
  exact-continuous-recovery discipline already exceeds it; cite as the
  field-standard weaker control. MDL probing (Voita & Titov) as the
  alternative a reviewer might request. arXiv:2606.02907 [SEARCH-SNIPPET]
  (probes detect task format) as a known-pitfall citation.
- Trained key-stabilization neighborhood (all trained, none frozen):
  Preconditioned DeltaNet (2604.21100), Gated DeltaNet-2 (2605.22791),
  VLA's write normalization, QK-norm/MuonClip lineage [SEARCH-SNIPPET].
- Flagged for a closer look pre-finalization: "Scaling Linear Attention
  with Sparse State Expansion" (arXiv:2507.16577) [SEARCH-SNIPPET];
  "Reasoning Primitives in Hybrid and Non-Hybrid LLMs" (arXiv:2604.21454)
  [LOW-CONFIDENCE-FETCH — re-fetch before citing anything from it].

## Novelty table (research agent's ratings)

| Claim | Rating | Nearest neighbor |
|---|---|---|
| (a) capacity-law-predicted load-dependent separation of intervention arms | MEDIUM | VLA + Okpekpe&Orvieto rival account |
| (b) in-context composition tracking controlled scale ladder w/ attention control | MEDIUM-HIGH | Based (pattern); R8ZbLi3oUv (unverified scoop risk) |
| (c) frozen random key-bias as capability intervention | MEDIUM | Frozen-QK (three-axis distinction) |
