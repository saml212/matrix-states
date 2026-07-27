# Novelty Re-Verification Gate — 2026-07-27 (triple sweep, full paper program)

**Invocation:** PI concern, live session 2026-07-27 ("I'm really becoming
worried that there is nothing new here and we didn't do our novelty check.
send the sonnet 5 subagents to check"). Executed per the 2026-07-16 gate
doctrine: two independent external sweeps (by-task + by-mechanism angles,
separate Sonnet agents, web-verified citations only — every named paper
fetch-verified during the run) + one internal-archive sweep, adjudicated by
the coordinator (this memo). Scope: the full active paper program P1–P6 +
the NCR spearhead — the first program-wide execution of the gate (prior
executions were per-claim: see internal table below).

**Headline verdict: NO CLAIM SCOOPED. Five cite-and-distinguish
obligations issued (three are new finds). Two coverage holes found by the
internal sweep (P4, P5 had no recorded check at all); P4's is closed by
this gate's external sweeps, P5's + one P1 sub-claim are under a targeted
follow-up sweep dispatched today (results to be appended here).**

## 1. External sweep A — by-task (24 queries, all candidates fetch-verified)

| Claim | Verdict | Key prior art |
|---|---|---|
| C1 rank recruitment (erank tracks K 1:1) | NOVEL-AS-CLAIMED (med) | arXiv:2602.04852, 2602.02195 — 2026 linear-attention state-rank papers; diagnostic/pruning use, no rank-vs-K law, no causal claim. Already cited in P1's brief (nazari2026rank/sun2026staterank). |
| C2 training-time spectral-projection causal control | NOVEL-AS-CLAIMED (med-low) | Nothing adjacent found (6+ query variants). |
| C3 rank-law (d_min, causal razor) | NOVEL-AS-CLAIMED (med) | arXiv:2503.02854 (state-tracking mechanisms, no d_min); arXiv:2606.07254 (non-Abelian falsifier benchmark, no rank/d_min). Open item resolved in §4 below. |
| C4 O(log h) composition | **PARTIAL-OVERLAP** (mechanism novel med-high) | **arXiv:2505.23683** — same asymptotic + task shape via O(log k) STACKED DEPTH; ours is state-internal squaring at constant depth. Differentiation paragraph mandatory. (Note: §G3-B14 had already independently found + discharged this — convergence confirms both sweeps.) |
| C5 super-linear capacity d^1.97 | **PARTIAL-OVERLAP** (med) | **arXiv:2505.12960 / Nat. Commun. 2026** — memristor/Hopfield superlinear capacity, N^1.74 CONTINUOUS recall. Different substrate (energy-based hardware vs SGD fast weights), different exponent. Cite + differentiate mandatory. |

## 2. External sweep B — by-mechanism (33 queries, 10 fetch-verifications)

| Mechanism | Verdict | Key prior art |
|---|---|---|
| M1 erank-of-state instrument | PARTIAL-OVERLAP (med) | Same two 2026 papers as C1 (independent reconvergence — good coverage signal). Distinguish causal-tracking use from their diagnostic use. |
| M2 training-time rank projection | NOVEL-AS-CLAIMED (med) | arXiv:2510.02823 = in-training SSM compression for EFFICIENCY, not causal necessity — cite as adjacent. |
| M3 repeated-squaring reads over in-context-written operators | **NOVEL-AS-CLAIMED (high)** | Searched via algorithmic-reasoning / FWP / automata-shortcut (2210.10749) / parallel-scan angles; nothing does state-internal squaring reads. The NCR core mechanism is clean. |
| M4 orthogonality-regularized fast-weight writes | **PARTIAL-OVERLAP** (med) | **arXiv:2607.19390** (Jul 2026) — READ-time Newton-Schulz orthogonalization of mLSTM memory, argued to be a REMOVABLE TRAINING SCAFFOLD, not a real memory improvement. Ours is a write-side training penalty; §G3-B23's h=61 exact-composition probe is evidence their scaffold story doesn't cover. Cite in kwall + NCR; §G3-B24's read must show composition survives, not just recall. |
| M5 P=1 bottleneck + blank-out verification | NOVEL-AS-CLAIMED (low-med) | Novel as a named methodology; generic causal-probing literature adjacent. |
| M6 argmax-loophole closure | **PARTIAL-OVERLAP (med-high)** | **arXiv:2605.05189** (May 2026, Nichani/Lee lineage) — formalizes argmax-vs-exact capacity distinction as scaling theorem (d²≍n log n vs d²≍n). Strongest dilution found. Ours is a task-CONSTRUCTION rule for rank lower bounds, theirs a capacity theorem. Cite prominently in P1/P4 methodology. |

## 3. Internal-archive sweep — coverage of recorded verdicts

13 recorded novelty checks found (2026-07-03 → 07-24; full table in the
sweep output, key rows): task-d-novelty-july2026.md (07-03) covers P1/P2/P3
cores; ncr_separation_grounding.md + §N1/§G3-B14 (07-15→18) cover NCR
(the model case — including the Yau 2506.10918 near-scoop that RESTRUCTURED
the claim, and 2505.23683's prior discharge); ortho_write_grounding.md §4
(07-15) covers P6/kwall (MuonSSM 2606.30461 narrowing on record).

**Coverage adjudication (updated by this gate):**

| Paper claim | Before today | After today |
|---|---|---|
| P1 rank-recruitment core | COVERED-STALE (07-03, pre-dates final framing) | **COVERED-FRESH** (C1+C2+M1+M2) |
| P1 invariant-subspace mechanism sub-claim (brief R5) | UNCOVERED (no standalone check ever) | follow-up sweep IN FLIGHT (Claim B) |
| P2/P3 rank-law trilogy | COVERED-STALE (07-03 check pre-dates 07-09 TOST/razor pinning) | **COVERED-FRESH** (C3 + §4 below) |
| P4 capacity d^1.97 | **UNCOVERED** (zero memo, zero design-doc section) | **COVERED-FRESH** (C5, with obligation) |
| P5 M* bytes-matched head-to-head | **UNCOVERED** (zero memo, zero design-doc section) | follow-up sweep IN FLIGHT (Claim A) |
| P6 kwall | COVERED-FRESH (07-15/16) | unchanged + M4 obligation added |
| NCR spearhead | COVERED-FRESH (07-16/18 chain) | unchanged; M3 high-confidence reinforces |

## 4. Coordinator verification of sweep A's open item

Sweep A could not verify OpenReview id=Tz8Li6G2xU ("Discovering Group
Structures via Unitary Representation Learning", bot-check page). Resolved
by coordinator via arXiv API today: the lineage paper is **Huh,
"Discovering Abstract Symbolic Relations by Learning Unitary Group
Representations", arXiv:2402.17002** (fetch-verified). Abstract-level
finding: matrix embeddings of symbols, implicit bias toward discovering
unitary group representations, and framing rhetoric directly adjacent to
ours ("matrix representations as a compelling alternative to traditional
vector embeddings"). BUT: static symbolic-operation completion — no
sequential state tracking, no rank-vs-K measurement, no d_min necessity or
causal step function. C3's NOVEL-AS-CLAIMED verdict stands. **Obligation:
cite + distinguish in P2/P3 (and flagship framing sections) — the
rhetorical overlap makes omission look like concealment.** Caveat on
record: verified at abstract level only (PDF not read).

## 5. Consolidated obligations (punch list for paper-editing sessions)

1. **Flagship + NCR:** differentiation paragraph vs arXiv:2505.23683 —
   depth-based O(log k) (prior art) vs state-internal constant-depth
   squaring (ours). [C4]
2. **P4 capacity:** cite + differentiate arXiv:2505.12960/Nat.Commun.
   (N^1.74, energy-based hardware) vs our d^1.97 SGD-trained fast weights;
   also cite 2605.05189. [C5, M6]
3. **kwall + NCR ortho sections:** cite + engage arXiv:2607.19390
   (read-time NS-scaffold skeptic result); frame §G3-B23/B24 composition
   evidence against its removable-scaffold interpretation. [M4]
4. **P1 + P4 methodology:** cite 2605.05189 prominently as the
   formalization of the argmax-vs-exact distinction; distinguish
   task-construction from capacity-theorem framing. [M6]
5. **P2/P3 + framing sections:** cite + distinguish Huh 2402.17002 (static
   completion vs sequential tracking; discovery vs causal dimension law);
   also cite 2606.07254 and 2503.02854 as task-family neighbors. [C3, §4]

## 6. Residual holes + fixes applied with this memo

- P5 Claim A + P1-R5 Claim B: targeted external sweep dispatched
  2026-07-27, results to be APPENDED here before either paper's next
  gauntlet round proceeds.
- `research/README.md` index was stale (missing ncr_separation_grounding.md,
  ortho_write_grounding.md) — fixed in this commit; this memo indexed.
- Paper gauntlets contain no novelty stage (novelty coverage lives in
  research/ memos + design registries, per doctrine). Noted for the paper
  skill's next revision: gauntlet stage 0 should VERIFY a fresh memo
  exists rather than assume it.
