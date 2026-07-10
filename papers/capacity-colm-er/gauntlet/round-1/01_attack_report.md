# Attack report — round 1 (fresh-context subagent, verify-vs-raws pass)

Role: `paper` skill gauntlet stage 1 (`prompts/attack-agent.md`). Draft
state: commit 1996e8d. The attacker independently recomputed C1-C13/C16
from the archived raw JSONs (md5-verified) and reported that every
headline number reproduces to the reported precision; the attacks below
are scope/characterization findings, not provenance findings.

Coordinator verification note: A1's central factual claims were
independently re-verified by the coordinator against
`matrix-thinking/deltanet_rd/run_deltanet_rd.py` (line 1043 is the ONLY
`use_heldout_entities=True` call, the C17 diagnostic),
`grammar_rd.py:423` (M3 draws entities from `train_name_ids`), and a
d=128 cell raw (`C17_heldout_entities` h1/h2/h3 rec@0.9 = 1.0/0.0/0.0 vs
`M2` 1.0/1.0/1.0) before the defense was dispatched. CONFIRMED.

## A1 (CRITICAL, claim-scope / metric mischaracterization)

`02_testbed.tex` asserts h4 is graded on "held-out entities and
bindings." FALSE per the training/eval code: `M3_held_out` (the source
of every h4 number) draws entities from the SAME pool used in training
(`pools.train_name_ids`); the only held-out axis is HOP DEPTH
(H_test+H_extra = 4/5/6/7/21 vs H_train = 1/2/3). The codebase's true
held-out-entity diagnostic (`C17_heldout_entities`, disjoint pool) is
archived in every cell backing C1-C11 and shows recovered_frac@0.9 = 0.0
at hop >= 2 in every sampled cell (12/12 at d=128, 4/4 at d=64). The
paper's "compositional generalization, not a memorized pair" framing
oversells the metric; the strongest counter-evidence sits unreported in
the paper's own evidence artifacts.

Defuse paths offered: (a) rescope every "held-out entities" statement to
held-out hop depth over the trained entity pool and disclose C17's
collapse in Limitations; or (b) report C17 alongside h4 and rescope the
contribution (materially weaker headline).

## A2 (SERIOUS, statistical / claim-scope)

Table 1's d=96 row prints precise-looking bounds (K* >= 90,
>= 2.50 bindings/KiB) built on the K=90 h4=1.0000x3 reading that the
program's own fresh-seed diagnostic (§15.27, brief row C14) found does
NOT replicate (fresh h4 = 0.9725, and the cell stays inadmissible even
at the recalibrated gate). The prose caveat exists but the table gives
the row CI-like visual confidence. Defuse: footnote the d=96 row as
unreplicated-at-fresh-seed / drop the numeric bindings/KiB bound.

## A3 (SERIOUS, missing disclosure in byte accounting)

The "byte honesty" arithmetic counts only the 4d^2 state bytes and
omits the learned per-entity anchor table (107 x d_state fp32: ~27.4 KiB
at d=64, larger than the 16 KiB state; ~53.5 KiB at d=128), which the
paper itself describes as capacity-relevant architecture. Defuse:
disclose the table's O(n*d) footprint and justify amortized-architecture
treatment, or fold it into bindings/KiB.

## A4 (SERIOUS, statistical presentation)

The two-point exponent d^1.97 [1.86, 2.09] is arithmetic-correct
(recomputed) but is fully determined by two points; no goodness-of-fit
exists and only degenerate d=96 could discriminate forms. The
CI-exclusion framing (excludes d^1, consistent with d^2) is the sound
claim and should lead; the point exponent should be demoted to a labeled
two-point interpolation, including in the abstract.

## A5 (SERIOUS/MINOR, reproducibility narrative)

"11 cells flipped; no cell re-ran" omits the 13th examined cell
(K=69/seed=1730), deliberately NOT flipped, disclosed in the design
record as out of the declared 11-cell scope. One-sentence disclosure
defuses.

## A6 (SERIOUS, venue framing)

The paper measures associative-memory capacity, not reasoning workloads;
the venue bridge rests on one CFP bullet. Defuse: one concrete paragraph
connecting the frontier to a reasoning-workload memory budget.

## A7 (MINOR-SERIOUS, missing citations)

Missing directly-adjacent fixed/compressive-memory work: Titans
(arXiv:2501.00663), Infini-attention (arXiv:2404.07143), Gated Delta
Networks (arXiv:2412.06464), Based (arXiv:2402.18668). IDs recalled from
the attacker's training knowledge; MUST be arXiv-API-verified before
citing (same discipline as citation-verification.json).

## Attacks considered and rejected by the attacker

- Fabricated/unverifiable numbers: none found; all headline numbers
  reproduce from md5-verified raws.
- Selective degenerate-fit disclosure: rule applied consistently in all
  three degenerate cases.
- d=32 exclusion: real kernel crash, verified in
  `smoke_dstate_kernel_result.json`; disclosure accurate.
- fig2 zip pairing bug: traced, correct (convoluted but sound).
- 0.2842 vs 0.2832 coherence discrepancy: two different objects
  (dry-run estimate vs launched table); the paper cites the correct one.
