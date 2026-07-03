# Empirical-Skeptic Verdict — Rank-Aware Gauntlet v2

## Summary

`conditional_go` — only if the construction-validation step (rank-1-constrained training on ProsQA-MULTI-2) passes. Otherwise paper §6 collapses to an appendix note.

## Key empirical risks

1. **Entropy reward is a near-cousin of §5's failed SVD-augmented readout.** Both expose σ(Z) to the optimizer through `torch.linalg.svdvals`; SVD-aug let CE *use* σ and CE chose not to. Entropy *forces* σ uniform, but a rank-1 functional solution with σ_2..σ_d as orthogonal noise satisfies BOTH losses jointly. Without a precise mechanism story for why entropy escapes the §5.3 "rank-1 active subspace" trap, we will write the §5 result with a different parameter.

2. **Lambda window may not exist.** Documented LLM-pipeline phenomenon (arXiv:2510.10959): entropy reg is highly coefficient-sensitive. H is bounded [0, log 16 ≈ 2.77] so explosion is bounded, but collapse to the rank-1 attractor isn't. Three λ values without commitment to a falsification rule = unfalsifiable experiment.

3. **ProsQA-MULTI is rank-k by ASSERTION, not CONSTRUCTION.** The bilinear-head argument requires that question text alone does not enumerate targets. ProsQA's "which leaf has property P?" generator likely lets W_down memorize P→{a1..ak} mappings, with rank-1 Z encoding "which question." Diagnostic D4 (train rank-1-constrained matrix-CODI on ProsQA-MULTI-2 — if accuracy matches unconstrained, the construction is invalid) is non-negotiable and must run BEFORE the 3x3.

4. **Mode-collapse blind spot.** If entropy inflates effective rank but the model routes everything through σ_1, the curve stays flat and we can't tell whether the rank gain is functional. Diagnostics D1 (per-direction zero), D2 (per-direction probe), D3 (Jacobian erank from §5.3) must be baked into the eval before any claim.

5. **Compute is under-budgeted by ~2x.** Realistic estimate 100-150 H100h (~$150-225 spot) vs. stated $80, accounting for SVD backward overhead, dataset debugging, crash recovery, and triple λ sweep.

6. **10-day deadline is feasible only with triage.** Cut nuclear-norm column and ProsQA-MULTI-4 row. Critical path: T1 construction-validation (day 1) → T2 CE on MULTI-2 (days 2-3) → T3 entropy on MULTI-2 (days 3-4) → T4 entropy on ProsQA-1 control (day 5). ~25 H100h, ~$40, leaves 3 days for writeup.

## Recommendation

Run T1 (rank-1-constrained training on ProsQA-MULTI-2) FIRST as a kill-switch. Commit to falsification rule on lambda. Drop nuclear-norm experiments. Bake D1-D4 diagnostics into eval scripts before any cell launches.

AGENT_DONE

```json
{"key_findings": ["Entropy reward is mechanistically adjacent to §5's failed SVD-augmented readout; need precise story for why it escapes the rank-1-active-subspace trap before burning compute", "Lambda window may not exist; commit to a pre-experiment falsification rule (curve flat at λ where Acc within 2pp of baseline = NULL)", "ProsQA-MULTI is rank-k by ASSERTION not construction; rank-1-constrained baseline (D4) must run first as a kill-switch", "Mode-collapse risk: entropy can inflate rank without making it functional; per-direction ablation diagnostics (D1-D3) required before any positive claim", "Compute under-budgeted ~2x ($80 stated vs $150-225 realistic); 3x3 should be cut to 2x2 (drop nuclear norm column, drop k=4 row)", "Prior negatives (arXiv:2510.10959 entropy coefficient sensitivity, arXiv:2512.11816 RL-on-latent-CoT degrades GSM8K) put the experiment on the wrong side of the prior — need explicit defense"], "recommendation": "Run T1 rank-1-constrained validation on ProsQA-MULTI-2 as day-1 kill-switch. If it passes, run minimal 2x2 (CE/entropy × MULTI-2/ProsQA-1) with D1-D4 diagnostics baked in. Cut nuclear-norm column. Commit to lambda falsification rule pre-experiment.", "go_no_go": "conditional_go"}
```
