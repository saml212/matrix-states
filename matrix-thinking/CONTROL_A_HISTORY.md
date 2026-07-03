> **STATUS: CLOSED (2026-07-04 consolidation).** Control A was a null-baseline
> experiment: a forward-only, propagating fake-Z rank-k ablation on **vanilla**
> (non-matrix) GPT-2 SFT ProsQA checkpoints, designed to test whether ProsQA
> itself is rank-1-solvable (attack A16 on the workshop paper) — i.e. whether
> the matrix-CODI rank-blindness finding could be explained by the *task*
> being trivial rather than the *gradient* being rank-blind. The v1 design was
> killed by two fatal protocol-asymmetry attacks before any GPU ran; the v2
> propagating-intervention redesign passed audit (PASS-WITH-FIXES: 3 P0 / 7 P1
> / 7 P2, all addressed — 10 FIXED + 2 PRESERVED) and ran on 2026-04-28.
> **Result: ambiguous, not flat.** Pooled Spearman r=0.0718 across k∈{1,2,4,8,16}
> (pooled full-sequence accuracy 78.87–79.07%, essentially constant), but the
> **randomized-h sensitivity-floor control** (a purely random ablation) landed
> inside the same narrow band as the real ablation (79.6/79.2/78.2% vs.
> unablated 80.0/78.8/78.0% per seed) — the probe itself has too little
> sensitivity at this scale to distinguish "rank doesn't matter" from "this
> instrument can't detect it." This result was run but never logged in
> `EXPERIMENT_LOG.md` or cited in the accepted paper's discussion section
> (which states "we do not have one at this scale" about exactly this
> alternative-explanation question) — a genuine documentation gap, closed by
> this file and a belated `EXPERIMENT_LOG.md` catch-up entry (both added
> 2026-07-04). This file consolidates six design/audit/attack documents
> (`CONTROL_A_RESEARCH_REPORT.md`, `CONTROL_A_EXECUTION_BRIEF.md`,
> `CONTROL_A_ATTACK_REPORT.md`, `CONTROL_A_IMPLEMENTATION_NOTES.md`,
> `CONTROL_A_AUDIT_REPORT.md`, `CONTROL_A_FIX_RECEIPT.md`), which have been
> moved to `archive/matrix-thinking-2026-04/`. Raw results:
> `experiment-runs/2026-04-28_control_a/control_a/` (`SUMMARY.txt`,
> `rank_projection_ablation_vanilla_sft.json`, `_script_used.py`, `train.log`).

# Control A — History

## What it was

**Experiment:** forward-only rank-k projection ablation on vanilla GPT-2 SFT
(ProsQA) to test whether ProsQA is rank-1-solvable (attack A16). Reused Round
4 vanilla SFT checkpoints (seeds 1337, 42, 7). Owner: Sam Larson. Target
deadline: ICML MI Workshop 2026, 2026-05-08. Originally spec'd as
`matrix-thinking/QUEUE.md` PRIORITY 0.

The question Control A answers: the workshop paper's headline finding is that
matrix-CODI's rank-k truncation has no effect on accuracy (rank-blindness).
One alternative explanation is that ProsQA itself only needs rank 1 to solve
— in which case rank-blindness would be a fact about the *task*, not the
*gradient*. Control A tests this directly by running the same rank-k
ablation, but on a **vanilla vector GPT-2** model fine-tuned on the same
task, with no matrix bottleneck at all — a k-truncation of an ordinary
hidden-state projection. If accuracy is flat across k here too, that's
evidence ProsQA doesn't require much rank regardless of representation; if it
degrades with k, the workshop paper's matrix-specific rank-blindness claim is
better supported by contrast.

## Timeline

1. **Research Report** (`CONTROL_A_RESEARCH_REPORT.md`) — scoped the
   experiment, method, and success criteria for the implementer agent.
2. **Execution Brief** (`CONTROL_A_EXECUTION_BRIEF.md`) — full agent handoff:
   owner, deadline, H100 access, roadmap through submission.
3. **v1 design killed pre-GPU.** **Attack Report**
   (`CONTROL_A_ATTACK_REPORT.md`) verdict: *"do not run Control A as
   currently designed. The two fatal attacks are protocol-asymmetry problems
   that make a flat-vs-flat outcome uninterpretable. They can be fixed, but
   they MUST be fixed before H100 time is spent."* Two fatal protocol
   asymmetries between the matrix and vector ablation paths.
4. **v2 redesign.** **Implementation Notes**
   (`CONTROL_A_IMPLEMENTATION_NOTES.md`) — a **propagating** fake-Z rank-k
   ablation (`matrix-thinking/scripts/run_control_a.py`), fixing both fatal
   issues found by the attack round.
5. **Audit.** **Audit Report** (`CONTROL_A_AUDIT_REPORT.md`) verdict:
   *"PASS-WITH-FIXES. The core intervention logic, SVD truncation, and eval
   [...]"* — 3 P0 / 7 P1 / 7 P2 findings.
6. **Fix Receipt** (`CONTROL_A_FIX_RECEIPT.md`): *"All audit and attack items
   are accounted for: 10 FIXED + 2 PRESERVED [...]"* — cleared to run.

## Result (2026-04-28, previously undocumented outside the raw JSON)

From `experiment-runs/2026-04-28_control_a/control_a/SUMMARY.txt` (variant
16×16, intervene at GPT-2 block 6/12, seeds {1337, 42, 7}, k∈{1,2,4,8,16},
n=500 examples):

- Pooled Spearman r = 0.0718 (p=nan); decision rule labeled this
  **"ambiguous"** (binomial-aware rule; the simpler first-pass rule alone
  would have called it "flat").
- Pooled full-sequence accuracy by k: k=1 78.93%, k=2 79.00%, k=4 78.87%,
  k=8 78.93%, k=16 79.07% — essentially constant.
- **Sensitivity-floor control (randomized-h, i.e. ablating with a random
  rather than task-relevant direction):** 79.6% / 79.2% / 78.2% per seed —
  landing inside the same narrow band as both the real ablation and the
  unablated baseline (80.0% / 78.8% / 78.0% per seed). This is the key
  finding: the ablation protocol cannot be distinguished from a random
  perturbation at this scale, so "accuracy is flat across k" cannot be read
  as "ProsQA is rank-1-solvable" — the instrument doesn't have the
  sensitivity to support that inference either way.
- Per-seed r was noisy and inconsistent in sign (seed 1337: r=0.3162; seed
  42: r=−0.1581; seed 7: r=0.7071), consistent with a near-null, low-power
  measurement rather than a real effect.

**Read:** Control A is inconclusive by its own pre-registered decision rule,
not a clean confirmation of either alternative. It does not overturn the
workshop paper's matrix-specific rank-blindness finding, but it also doesn't
independently rule out task-rank-1-solvability as a contributing factor — the
paper's own discussion section already hedges this ("we do not have one at
this scale"), which turns out to be consistent with what this run actually
found, just not explicitly linked to it until now.
