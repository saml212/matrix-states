# Synthesizer Report: Rank-Aware Gauntlet v2

## The 3x3 Restated with Falsifiable Predictions

| | std-CE | +entropy H(σ̃) | +nuclear ‖Z‖_* |
|---|---|---|---|
| **ProsQA-1** (rank-1-solvable) | FLAT — paper §3–5, already done | predict: **FLAT** (entropy can't create rank demand where task has none) | predict: FLAT |
| **MULTI-2** (rank ≥ 2 by construction) | predict: **FLAT/floor** (objective rank-blind regardless of task) | predict: **BENDS at k=2** — headline result | predict: weak bend, σ_1-only (wrong functional) |
| **MULTI-4** (rank ≥ 4 by construction) | predict: floor effect, low accuracy | predict: BENDS with sharper inflection at k=4 | predict: weak bend |

Each cell has a unique prediction; the three conditions form an interlocking argument: task × objective both required.

## Minimal Sufficient Subset (5 cells, 1 free)

Full 3x3 = 9 cells × 3 seeds ≈ $80 spot. The complete story fits in 5 cells ≈ $36:

1. **(ProsQA-1, std-CE)** — FREE. Paper baseline. Anchor.
2. **(MULTI-2, std-CE)** — Task-alone baseline. CONFIRM: flat → task doesn't fix rank-blindness. FALSIFY: bends → CE training is already rank-sensitive on multi-answer tasks; story collapses.
3. **(MULTI-2, +entropy)** — Headline. CONFIRM: Pearson r > 0.5 on rank-k curve, statistically significant accuracy degradation at k=1. FALSIFY: flat → entropy reward doesn't induce rank use even on rank-demanding tasks.
4. **(ProsQA-1, +entropy)** — Null control. CONFIRM: flat → clean 2×2 factorial. FALSIFY: bends → entropy introduces rank structure spuriously (weaker story but still publishable as "entropy sensitive to task").
5. **(MULTI-2, +nuclear)** — Mechanism comparison. CONFIRM: weaker bend than entropy (or flat) → nuclear is the wrong functional, entropy is the right one. FALSIFY: nuclear matches entropy → undermines entropy novelty.

**Dropped:** All MULTI-4 cells (confirmatory, not decisive). **Dropped:** (ProsQA-1, +nuclear) (irrelevant to narrative).

**Gating:** Run cells 2–3 first. If (MULTI-2, std-CE) bends unexpectedly, stop and reassess before committing remaining cells.

## Critical Attack: The 'Rank-k by Construction' Argument Has a Gap

Report A claims k disjoint ProsQA answers require k orthogonal directions in Z, making rank-1 truncation provably fail. **The gap:** "disjoint" in the DAG means separate leaf nodes — it does not guarantee the model learns orthogonal representations. If the k answers are semantically correlated or the optimizer finds weight-tied projections, all k bilinear heads could project onto the same dominant singular direction.

**Mandatory mitigation (essentially free):** After training (MULTI-2, std-CE), log the cosine similarity between bilinear probe pairs (u₁,u₂) and (v₁,v₂). If near-zero, the bound is empirically satisfied. If not, the "by construction" claim is false and must be softened to "empirically demanding" rather than "formally requiring."

## Ideator Check (to be updated as posts arrive)

No ideator posts yet. Will monitor and evaluate each against the minimal subset.

## Swap Recommendations

None yet — watching for ideator proposals that might be sharper than cells 2–5 above. Candidate upgrade criteria: (a) formally tighter rank-demand argument, (b) cheaper to run, (c) closes a different §7 gap.

---

AGENT_DONE

```json
{
  "key_findings": [
    "Full 3x3 tells a complete story but 5 cells (1 free) are sufficient — saves ~$44 spot",
    "The critical 'rank-k by construction' argument has a logical gap: disjoint DAG answers ≠ orthogonal representations; post-hoc cosine similarity logging is mandatory",
    "MULTI-4 is confirmatory only — gate on MULTI-2 results before committing",
    "Nuclear norm is the wrong functional for rank spread; useful only as contrast to entropy",
    "Headline falsification risk: if (MULTI-2, std-CE) bends, the story collapses — run this cell first"
  ],
  "recommendation": "Run minimal 5-cell subset in order: baseline (MULTI-2, std-CE) first, then headline (MULTI-2, +entropy), null control (ProsQA-1, +entropy), mechanism comparison (MULTI-2, +nuclear). Add cosine similarity between bilinear probe pairs as mandatory logging stat. Gate MULTI-4 on MULTI-2 results.",
  "minimal_subset_cells": [
    "task=ProsQA-1, loss=std-CE (FREE, already done)",
    "task=MULTI-2, loss=std-CE",
    "task=MULTI-2, loss=+entropy",
    "task=ProsQA-1, loss=+entropy",
    "task=MULTI-2, loss=+nuclear"
  ],
  "swap_recommendations": [
    "No swaps yet — monitoring ideator posts. Will upgrade if ideator proposes a formally tighter rank-demand argument or a cheaper path to the same falsification."
  ]
}
```
