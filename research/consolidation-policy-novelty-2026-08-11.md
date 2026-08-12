# Novelty gate — usage-gated eviction-time consolidation (opened 2026-08-11)

**Gate status: DISCHARGED 2026-08-12 (coordinator adjudication).** All
three sweeps + both verification legs closed. Verdict of record: the
claim is COVERED-FRESH — the exposure-normalized usage-trace gate on a
compressive delta write, plus the oracle/shuffled/random causal battery,
is unoccupied as of 2026-08-12; the surrounding skeleton is occupied
(Tensor Cache) and carries mandatory cite-and-distinguish obligations
(list below). Attack stage released.

**LANE OUTCOME (2026-08-12, same day): PARKED — KILLED-AT-DESIGN.**
The claim was novel but the mechanism failed the design gauntlet
(attack R2 BLOCKED, `matrix-thinking/CONSOLIDATION_POLICY_WATERFALL.md`
§8): no regime beats a steelmanned (ridge-tuned, index-coded)
byte-matched exact-KV baseline, and the usage gate measures
null-to-harmful vs a constant-η control. Novelty ≠ merit — this memo
stays as the do-not-redo record and cite inventory; revival conditions
in registry §8. Zero GPU-h spent.

## Candidate claim (verbatim registration)

> While a KV pair resides in a sliding attention window, accumulate a
> per-item, exposure-normalized attention-usage trace; at eviction, use
> that trace to gate/scale a residual delta-rule write of the item into a
> fixed-size associative fast-weight matrix (S_mem, separate from any
> compositional operator), optionally allocating replay budget — causally
> validated against oracle / shuffled / random-selector controls at
> matched state bytes, write count, and compute.

Origin: PI conversation 2026-08-11 (idea developed with a second agent,
brought here for verification). PI GO given 2026-08-11. Registry:
`matrix-thinking/CONSOLIDATION_POLICY_WATERFALL.md`.

## Sweep 1 — citation verification (agent, 2026-08-11): ALL VERIFIED

No fabrications among the 9 works cited in the source conversation:
Gated DeltaNet hybrids (arXiv:2412.06464, ICLR 2025); Qwen3-Next-80B-A3B
(3:1 GDN:attention MoE, 512 experts/10 active, HF + Model Studio); δ-mem
(arXiv:2605.12357); Tensor Cache (arXiv:2605.22884); "Do Language Models
Need Sleep? Offline Recurrence for Improved Online Inference"
(arXiv:2605.26099); GDWM (arXiv:2601.12906 — consolidates via
LoRA-adapter gradient-step allocation, NOT a fast-weight matrix); H₂O
(arXiv:2306.14048, NeurIPS 2023); Priming / AWS Hybrid Model Factory
(arXiv:2605.08301, <0.5% token-budget hybrid conversion, vLLM plugin);
vLLM hybrid KV-cache manager (docs + PyTorch blog).

## Sweep 2 — external by-mechanism (agent, 2026-08-11): PARTIALLY-OCCUPIED, residue open

- **Most dangerous prior art: Tensor Cache (arXiv:2605.22884, 21 May
  2026).** Near-identical skeleton — SWA L1, eviction-triggered
  outer-product/delta-rule write into fixed L2 fast-weight matrix,
  linear-attention read. Its write gate is a FIXED learned per-head
  scalar (λ_h=σ(θ_h^λ), η_h=σ(θ_h^η)), identical for every evicted item;
  no per-item usage signal; ablations are architectural (outer-vs-delta,
  chunk size), never signal-source controls. Must be cited and
  distinguished up front; the gate ablation battery is the contribution.
- Usage-signal line (H₂O cumulative attention; Compressive Transformer
  arXiv:1911.05507 "most-used" sort; SirLLM entropy): signal exists but
  only ever drives HARD exact-KV retain/discard — never a scaled
  compressive write. No exposure normalization anywhere.
- Titans (arXiv:2501.00663): surprise/gradient-gated, every-token, no
  eviction structure. ATLAS (arXiv:2505.23735): windowed Omega-rule
  test-time optimization, no per-item usage gate. CAMELoT
  (arXiv:2402.13449): novelty/recency consolidation. Infini-attention
  (arXiv:2404.07143): uniform every-token compressive write.
- **No paper found combines usage-trace gating with a compressive
  delta-rule write, exposure-normalizes the trace, or runs
  oracle/shuffled/random causal controls on the gating signal.**

## Sweep 3 — internal archive (agent, 2026-08-11)

- `HEAD_TO_HEAD_DEMO_DESIGN.md` §1.9 items 5/7: attention-score-based
  eviction + cap-trained baseline were REGISTERED as the axis-2 fairness
  follow-on and descoped for COST (+0.76/+7.6 GPU-h), never killed. This
  proposal is a smarter version of that shelved item — no dead direction
  revived; KILL_LIST has zero overlap.
- NOT the NCR claim pivot: §G3-B32's queued lever is write-conditioning
  of the compositional operator (ortho/conformal-scaffold family); this
  is a consolidation policy for a separate S_mem. Different lever;
  competes only for GPU-hours/coordinator bandwidth.
- Flagship thesis T1 is orthogonal; nearest evidence row is R10 (M*
  fixed-32KB-state separation, already filed).
- Hard-rule deltas required vs. the source conversation's design:
  de-bundle 3–4 unproven axes (gate policy / delta mechanism / replay /
  frozen-backbone retrofit); add the mandatory param-matched flat-vector
  ablation; gradient blank-out test for real eviction (not a shape
  check); exact continuous recovery throughout (already respected).

## δ-mem full-text leg (agent, 2026-08-12): GAP CONFIRMED

Full PDF + repo read. Gate is β_t = σ(W_β x_t + b) — learned,
per-dimension, computed from the CURRENT hidden state only, at write
time (per-token TSW / per-segment SSW / multi-state MSW). No sliding
window, no eviction event, no residency-accumulated usage signal, no
oracle/shuffled controls. Backbone frozen; only the 0.12–0.48% sidecar
trains. δ-mem occupies the residual-delta-write scaffold and the
frozen-retrofit recipe, NOT the usage-gated part.

## Sleep-paper identity resolution (same agent): TWO distinct papers

- arXiv:2605.26099 — "Do Language Models Need Sleep? Offline Recurrence
  for Improved Online Inference" (Lee, McLeish, Goldstein, Fanti;
  CMU/UMD). Offline recurrent passes distill accumulated context into
  SSM fast weights, then KV cache cleared.
- arXiv:2606.03979 — "Language Models Need Sleep: Learning to
  Self-Modify and Consolidate Memories" (Behrouz, Hashemi, Javanmard,
  Mirrokni; Google/Cornell). Distillation ("Knowledge Seeding") + RL
  synthetic-curriculum "Dreaming" into long-term parameters.
Cite both; they bracket the replay/consolidation neighborhood.

## Cite-and-distinguish obligations (running list)

Tensor Cache (the skeleton); H₂O + Compressive Transformer + SirLLM
(usage signal, hard retention only); δ-mem (delta sidecar, content-only
gate); GDWM (utility→LoRA gradient budget); both sleep papers (periodic
replay); Titans/ATLAS (surprise/optimization-gated test-time
memorization); Infini-attention (uniform compressive write); Gated
DeltaNet + Qwen3-Next (hybrid architecture is commercial prior art);
OpenReview R8ZbLi3oUv (pending read).

## R8ZbLi3oUv leg (agent, 2026-08-12): SCOOP RISK NONE — adjudicated DISCHARGED

OpenReview's bot-check blocked every direct route (API v1/v2, PDF
endpoint, reader-proxy, Wayback; no arXiv/DBLP/Scholar mirror exists).
Recovered instead: the verbatim abstract (berenslab/iclr-dataset
iclr26v1.parquet) + ALL THREE full official reviews (HF
3Liz22/iclr2026_real_reviews, paper_id 16337). The paper is a
DIAGNOSTIC study: 150M from-scratch Mamba/GLA/GDN hybrids swept over
attention proportion on math reasoning, with an 8-category LLM-judge
error taxonomy identifying associative-recall failure as the primary
attention-free error mode. Scores 6/4/2 (borderline, pre-decision). No
write gating, no eviction consolidation, no usage signals, no causal
gate ablations, no compositional/repeated-squaring reads — in the
abstract or ANY review. Verdict: scoop risk NONE to the consolidation
claim AND none to NCR. Coordinator adjudication: three independent,
detailed, non-contradicting reviews with zero trace of either concept =
sufficient to discharge; residual (raw PDF unread) noted — escalation
path if ever needed is a manual OpenReview login. Reclassified as a
MOTIVATION cite (attention-free architectures fail at associative
recall ⇒ the S_mem gap is real).
