# Methodological Skeptic — Verdict

## Summary

The proposed 3×3 (ProsQA-MULTI × {std, entropy, nuclear}) is **not rigorous as
specified**. The headline rank-k-by-orthogonality argument has no contact with
the actual model, and the proposed design contains no cell that falsifies the
most obvious alternative explanation. **Status: requires_revision.**

## Fatal-class objections

1. **Bilinear head is fictional.** The script's readout is
   `h_out = LN(w_down(flatten(Z)))` followed by a full GPT-2 transformer pass,
   repeated 6×, with the LM head reading `h_answer ∈ R^768`, never Z. The
   "u_iᵀ Z v_i needs k orthogonal directions" argument applies to a head that
   does not exist. Fix: either add a real bilinear answer head trained jointly,
   or drop the orthogonality framing and claim only correlational evidence.

2. **Position-decomposition substitutes for rank-decomposition.** Six rank-1
   Z_p positions provide six independent routing channels through 768-dim
   feedback hidden states. Rank-k truncation per-Z_p does not rule this out:
   the task can succeed with all Z_p rank-1 by encoding target_k at position
   k. The 3×3 contains no cell that distinguishes "rank-blind objective" from
   "task is per-position-rank-1-solvable across 6 positions".

3. **Embedding-tying.** The k targets are GPT-2 wte vectors with non-trivial
   pairwise cosine and shared BPE subwords. Required pre-experiment check:
   compute Gram of the target subset and verify rank-k structure is a real
   constraint on this vocab — not a generic claim.

## Lesser-but-real objections

4. SVD backward at uniform σ blows up via 1/(σ_i² − σ_j²); kills (D) without a
   custom autograd. svdvals (B's path) is fine.
5. §5.5 negative control does not bridge to ProsQA-MULTI inferentially.
6. Nuclear-norm column has predicted-null cells — wastes ~1/3 of compute.
7. (E) wd-vs-no-wd at n=3 seeds is underpowered for the observed seed spread.

## Required fix (strongest single intervention)

**Replace the nuclear-norm column with a per-position-rank-1 training penalty
on each Z_p**, and **add a measured target-embedding orthogonality precheck**
before any ProsQA-MULTI run. The penalty + precheck together turn the design
from a correlational sweep into a causal test of the orthogonality argument.
Without these, ProsQA-MULTI succeeding is consistent with the position-routing
explanation and the paper's headline does not follow.

## Recommendation

**plan_status: requires_revision.** (B) entropy reward survives standalone as
a smaller, defensible result. (A) ProsQA-MULTI as currently framed should not
be the headline until the bilinear-head and position-decomposition objections
are addressed structurally.

AGENT_DONE

```json
{"key_findings": ["Bilinear-head u_iᵀ Z v_i is fictional — the actual readout is LN(w_down(flatten(Z))) → GPT-2 layers → LM head on h ∈ R^768; the orthogonality argument has no contact with the model", "Position-decomposition trivially substitutes for rank-decomposition: 6 rank-1 Z_p positions route any k≤6 targets via 6×768-dim feedback channels; per-position rank-k truncation does not constrain this", "GPT-2 vocab targets are not orthogonal; rank-k lower bound presumes target embeddings span k-D and must be MEASURED on the actual ProsQA-MULTI vocab subset before claiming", "SVD backward at uniform σ is singular; kills (D) without custom autograd; svdvals path (B) is the only stable variant", "§5.5 negative control does not inferentially bridge to ProsQA-MULTI", "Nuclear-norm column has predicted-null cells — drop it for a per-position-rank-1 forcing column that actually falsifies position-decomposition", "(E) wd sweep at n=3 seeds is underpowered relative to observed seed spread (~5 SD vs MDE ~12)"], "recommendation": "Replace nuclear-norm column with a per-position-rank-1 training penalty AND run a target-embedding-Gram precheck before ProsQA-MULTI. Without these the design cannot distinguish 'rank-blind loss' from 'position-decomposable task' or 'embedding-tied targets'.", "plan_status": "requires_revision"}
```
