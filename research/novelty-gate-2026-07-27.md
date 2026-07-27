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

**Headline verdict (final, gate CLOSED same day): NO CLAIM SCOOPED.
Seven obligations issued (§5 + §7): five cite-and-distinguish, one
mandatory REFRAMING (P5 — the bytes-matched comparison axis is Zoology/
Based prior art; our contribution is the counter-consensus result
direction, see §7), one instrument-not-concept positioning (P1-R5). Two
coverage holes found by the internal sweep (P4, P5 had no recorded check
at all) — both closed today. Every program claim now COVERED-FRESH.**

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
| P1 invariant-subspace mechanism sub-claim (brief R5) | UNCOVERED (no standalone check ever) | **COVERED-FRESH** (§7 Claim B) |
| P2/P3 rank-law trilogy | COVERED-STALE (07-03 check pre-dates 07-09 TOST/razor pinning) | **COVERED-FRESH** (C3 + §4 below) |
| P4 capacity d^1.97 | **UNCOVERED** (zero memo, zero design-doc section) | **COVERED-FRESH** (C5, with obligation) |
| P5 M* bytes-matched head-to-head | **UNCOVERED** (zero memo, zero design-doc section) | **COVERED-FRESH with REFRAMING OBLIGATION** (§7 Claim A) |
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
  2026-07-27 — COMPLETED same day, adjudicated in §7 below. Gate now
  closed program-wide: every claim COVERED-FRESH.
- `research/README.md` index was stale (missing ncr_separation_grounding.md,
  ortho_write_grounding.md) — fixed in this commit; this memo indexed.
- Paper gauntlets contain no novelty stage (novelty coverage lives in
  research/ memos + design registries, per doctrine). Noted for the paper
  skill's next revision: gauntlet stage 0 should VERIFY a fresh memo
  exists rather than assume it.

## 7. Follow-up sweep results (same day, 25 queries, fetch-verified) — ADJUDICATED

### Claim A — P5/M* bytes-matched head-to-head: PARTIAL-OVERLAP (mod-high), with a MANDATORY REFRAMING

The comparison DESIGN is prior art, 2023-2024: **Zoology (arXiv:2312.04927)
literally plots recall accuracy vs. state size in BYTES with capped-window
attention among the baselines, and Based (arXiv:2402.18668, ICML 2024)
names the "recall-memory tradeoff" frontier.** P5 must NOT present
bytes-matched comparison as a novel methodology. What survives as the
contribution, and how the paper must be reframed:
(a) the fast-weight matrix contender (vs their Mamba/H3/Based instances),
(b) the clean matched-PAIR head-to-head (vs their multi-point frontier),
(c) most importantly the RESULT DIRECTION: Based's verified abstract says
fixed-state models "struggle at recall" — our fast-weight WIN at matched
bytes runs COUNTER to the Zoology/Based-era consensus. That reversal is
the headline; the paper must cite both and explicitly reconcile (what
about the architecture/task/training makes the direction flip).
Also on record: HOLA (2607.02303) and 2607.10441 as July-2026 neighbors in
the same problem space, different comparison axes.

### Claim B — P1's invariant-subspace / restricted-operator-erank mechanism: NOVEL-AS-CLAIMED (moderate)

No paper combines (i) trained fast-weight/recurrent STATE matrix, (ii) a
learned composition-relevant subspace within it, (iii) restricted-operator
effective rank vs. a closed-form ideal as the diagnostic. Adjacent
territory to position against (verified): 2602.22719 (SSM activation-
subspace bottlenecks — activations, not state), 2606.02993 (provable
spectral/irrep account for MLP group composition — theory, feedforward),
2602.09783 (invariant subspaces in transformer OV circuits — explicitly
not recurrent state), plus the neuroscience shared-subspace line (PubMed
41299181, cross-domain analogy only). Obligation: present the MEASUREMENT
as the contribution, not "invariant subspaces" as a concept.

### Obligations added to §5 punch list

6. **P5/M*:** reframe as counter-consensus result on the ESTABLISHED
   Zoology/Based recall-memory axis (cite 2312.04927 + 2402.18668 +
   2402.01032); kill any "novel comparison design" language; add the
   reconciliation paragraph explaining the direction flip. [Claim A]
7. **P1 §mechanism:** position restricted-erank measurement against
   2602.22719 / 2606.02993 / 2602.09783; claim the instrument, not the
   concept. [Claim B]

**GATE CLOSED 2026-07-27: all program claims COVERED-FRESH; verdict of
record = this memo; 7 obligations total; no scoop.**
