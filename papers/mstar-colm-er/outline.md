# Paper outline — Constant-memory recall (mstar-colm-er)

Venue limit chosen: 7.0-7.25 written main-text pages (workshop band 4-10pp,
refs excluded). Section files live in `sections/` as `.tex`, assembled by
`bundle/main.tex`.

## Section plan

| # | Section | Pages | Claims (ids) | Figures | Notes |
|---|---|---|---|---|---|
| 1 | Introduction | 1.0 | C1, C3, C4 (headline restatement only) | — | KV growth vs fixed state; contribution bullets |
| 2 | Experimental setup | 1.5 | C9, C10 | setup table | task, arms, matching, metric + Nichani caveat, pre-registration discipline |
| 3 | Recall separation at matched budget | 1.0 | C1, C2 | Table 1 | per-seed acc, paired CIs vs frozen 0.30 margin |
| 4 | Fixed state vs KV cache at long horizon | 1.25 | C3, C4 | Fig 1 (+Table 2 appendix) | horizon flatness; capped walk; degenerate-baseline clause verbatim |
| 5 | Where the recall lives | 0.75 | C5, C6, C6b | Fig 2 (+Table 3 appendix) | S0 causal necessity; S1 inert; nonlinear storage |
| 6 | Scope: what does not separate | 0.75 | C7, C8 | Table 4 | task2 seed-variance; K48 chance row |
| 7 | Related work | 0.5 | — | — | five named neighbors from the brief |
| 8 | Limitations | 0.25 | (restates C7/C8 scope) | — | scale, synthetic task, argmax, one arch family |
| 9 | Conclusion | 0.25 | — | — | one paragraph |
|   | **Total** | **7.25** | | | 10pp hard cap leaves float slack |

## Outline sanity checks

- [x] Page budgets sum inside the venue band (7.25 written, cap 10).
- [x] Every claim id (C1-C10, C6b) appears in exactly one body section
      (C1/C3/C4 restated in §1 as headline, owned by §3/§4).
- [x] Both figures placed; appendix tables named.
- [x] Related work distinguishes five neighbors by name (brief).
- [x] No section carries a claim without a complete evidence row.

## Per-section beat sheet

### 1. Introduction
- Long-horizon inference is memory-bound: KV cache grows O(T); the coming
  constraint is HBM, not FLOPs.
- Alternative: fast-weight models carry a fixed-size state; the open question
  is whether such a state actually retains task-critical bindings a matched
  attention model would hold in its cache.
- Our testbed: episode-restricted K-way recall at matched params/tokens;
  headline (C1 numbers), horizon flatness (C3), no cap rescues the baseline
  (C4, verdict-of-record phrasing), causal localization (C5).
- Contribution bullets = brief's four.

### 2. Experimental setup
- Task 1: MQAR-style bind->query episodes, K=32 entity pool, GPT-2 token IDs,
  single hop, T_bind=224, n=4096 eval queries on a pinned eval seed set.
- Metric: acc_A = episode-restricted K-way top-1 at the answer position via
  each arm's own LM head; chance 1/K; demonstration bar 3x chance;
  Nichani caveat stated at definition site.
- Three arms + matching table (C9, C10); training identical (steps, lr,
  optimizer, curriculum, seeds paired across arms).
- Pre-registration: margins/tiers frozen before the sweep; degenerate-baseline
  clause quoted; n=3 seeds verdict-grade paired CIs.

### 3. Recall separation at matched budget
- Table 1 with per-seed acc_A (C1) and both paired CIs (C2).
- Every contender seed >= 10.7x the demonstration bar; neither baseline ever
  clears it; CI floor 0.958 vs margin 0.30.
- Transformer non-competitiveness framed exactly: matched-budget caveat.

### 4. Fixed state vs KV cache at long horizon
- Fig 1: acc_A vs context length; contender flat >=0.998 at 454/902/1798
  tokens, fixed 32,768 bytes (C3).
- Capped transformer at every M x 32,768-byte budget: chance (C4); capping
  does not help (forced locality answered).
- The degenerate-baseline clause: uncapped baseline below bar => no
  memory-multiplier claim; verdict of record sentence verbatim.

### 5. Where the recall lives
- Fig 2: S0-zeroed collapse / S1-zeroed unchanged per seed (C5).
- No linear read on raw state decodes; pre-LM-head hidden does, contender
  only (C6, appendix Table 3); per-seed legibility variance (C6b) as honest
  instrument note.
- Interpretation: bindings stored nonlinearly in block-0's state; the model's
  own forward is the decoder.

### 6. Scope: what does not separate
- task2 multi-hop: Table 4, 9 seeds/arm; 3/9 vs 0/9; pooled CI
  non-decision-grade with the batch-effect flag stated; adjudicated
  trainability/seed-variance (C7); horizon collapse of the partial seed.
- K48 row: all three arms chance at K/d=0.75 (C8) — separation is
  load-bounded.

### 7. Related work
- Fast weights lineage; DeltaNet; MQAR (opposite-headline contrast);
  StreamingLLM as instrument; KV-compression as different question; Nichani
  caveat source.

### 8. Limitations
- 14M-class scale, synthetic task, argmax decoding, single fast-weight
  family, matched-budget caveat, K/d ceiling.

### 9. Conclusion
- One paragraph; no new numbers.
