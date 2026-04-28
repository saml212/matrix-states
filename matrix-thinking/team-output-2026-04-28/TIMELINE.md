# Pragmatist Timeline — ICML MI Workshop, May 8 2026

## Hard Constraints Recap
- 10 days to deadline (today Apr 28)
- Spot H100 @ $1.50/h; no idle
- Each run: ~1–3 H100h
- Subagent cascade overhead: ~½–1 day per new code path (idea→research→attack→defend→script→audit→run)
- ProsQA-MULTI dataset gen: CPU-only, zero GPU cost, parallelizable with any H100 work

---

## Budget Plans

### $50 Plan — "Minimal Proof"
- 2 conditions only: ProsQA-MULTI-2 baseline (std loss) + entropy reward
- 2 × 3 seeds × 2h avg = 12 H100h = **$18 GPU**
- ~4h smoke/debug overhead = **$6**
- Buffer for 1 re-run: **$9**
- **Total: ~$33 GPU + $17 cascade agent costs**
- **Delivers:** Shows "task matters" — entropy bends curve on MULTI-2 but not MULTI-1. Two-condition comparison.
- **Risk:** Reviewer says "cherry-picked single task size." Weak without MULTI-4 or nuclear-norm contrast.

### $100 Plan — "Core Paper" (RECOMMENDED)
- 4 conditions: MULTI-2 + MULTI-4 × {std loss, entropy reward}
- 4 × 3 seeds × 2h = 24 H100h = **$36 GPU**
- ProsQA-1 entropy control (already-have-result check): 3 seeds × 1.5h = **$7**
- Implicit bias wd=0 sweep: 3 seeds × 1h = **$5**
- Overhead + 1 full set of re-runs: **$15**
- **Total: ~$63 GPU + $37 for cascades/agent overhead**
- **Delivers:** 3-condition factorial. The complete story: task needs rank → objective must push rank → neither holds without the other. Defensible to workshop reviewers.

### $200 Plan — "Full 3×3"
- Full 3×3 matrix (A+B+C): 9 cells × 3 seeds × 2h = 54 H100h = **$81**
- Nuclear norm comparison, implicit bias, appendix: ~**$30**
- Overhead + re-runs: **$30**
- **Total: ~$141 GPU + $59 buffer**
- **Delivers:** Best version. Nuclear norm as contrast sharpens the mechanism claim.
- **Risk:** Nuclear norm (C) is scientifically the weakest novelty claim; adds 2 days of cascade work. Only worth it if core story lands cleanly by Day 6.

---

## Critical Path (Day-by-Day)

| Day | Date | GPU | CPU/Agent Work |
|-----|------|-----|----------------|
| 1 | Apr 28 | Off or rerun Round-3 repro | **ProsQA-MULTI DAG mod** — 150 lines, no GPU. Cascade: attack→defend→build→audit |
| 2 | Apr 29 | Smoke test MULTI-2 (1–2h, $3) | Entropy reward code cascade |
| 3 | Apr 30 | **MULTI-2 baseline, 3 seeds** (6h, $9) | Entropy reward audit finalize |
| 4 | May 1  | **MULTI-2 entropy, 3 seeds** (6h, $9) | MULTI-4 cascade, analyze Day-3 results |
| 5 | May 2  | **MULTI-4 both conditions** (12h, $18) | Begin §4 draft; MULTI-1 entropy control |
| 6 | May 3  | Re-runs if needed; wd=0 sweep ($5) | **DECISION DAY**: does entropy bend the curve? |
| 7 | May 4  | GPU off unless nuclear norm added | Write figures, §5 update, §7 revision |
| 8 | May 5  | Off | Full paper draft |
| 9 | May 6  | Off | Polish, proofread |
| 10 | May 7 | Off | Submit buffer day |
| — | May 8  | — | **DEADLINE** |

---

## Parallelizable Work
- **ProsQA-MULTI gen (CPU):** start today, zero GPU cost. Runs while H100 is on anything.
- **Entropy reward code:** can be written (agent cascade) while MULTI-2 baseline trains.
- **Paper writing:** §5.5 (Control A, just committed) is done. §4 additions start Day 5 regardless.

---

## Bare Minimum in 5 Days (by May 3)
- MULTI-2: std loss (flat) + entropy reward (bends) — 6 runs total, ~12h GPU = **$18**
- Either outcome is publishable: "entropy works on a provably rank-2 task" OR "entropy also fails, which is the stronger negative result"
- **5-day cost: ~$25 GPU. 5-day deliverable: central finding either way.**

---

## Risk-Adjusted Expected Value

| Risk | P | Notes |
|------|---|-------|
| ProsQA-MULTI rank-1 loophole (model finds rank-1 solution by embedding tying) | 30% | Most critical scientific risk. Verify lower bound rigorously before running. |
| Entropy reward fails to bend curve even on MULTI-2 | 25% | Publishable as strong negative; paper's thesis is reinforced not broken |
| Subagent cascade blows timeline (>2 days for one code path) | 20% | Mitigate by starting MULTI gen today, not waiting on cascade completion |
| Both experiments flat (no differentiation) | 10% | Weakest outcome. Paper would need reframe around "no objective fixes rank blindness" |

**P(clean positive result) ≈ 55%; P(publishable negative) ≈ 35%; P(total wash) ≈ 10%**

Expected value at $100 spend: strong. Workshop papers accept well-designed negative results.

---

## Recommendation
**Run the $100 plan.** Start ProsQA-MULTI CPU gen TODAY (no GPU needed). Kick the entropy reward cascade TODAY in parallel. First GPU checkpoint is Day 3. Decision point is Day 6 — if core story is clean, add nuclear norm; if not, go to paper immediately with 4-condition result.

**Do not start nuclear norm (C) until Day 6 check-in. It is the least novel, most time-consuming experiment, and adds complexity to a paper that doesn't need it yet.**

**Skip GRPO (G) and symmetric extension (H) entirely — already NO-GO.**

---

AGENT_DONE
