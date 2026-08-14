# Novelty gate — NCR write-time spectral conditioning (claim pivot, 2026-08-13)

**Gate status: DISCHARGED-WITH-OBLIGATIONS 2026-08-13 (coordinator
adjudication), scoped to the surviving mechanism family.** Triple sweep
complete same-day as DRAFT-R0. The claim's task-level frame is OPEN; the
mechanism territory is ACTIVE — two named nearest-neighbors are
mandatory cite-and-distinguish anchors; the surviving wedge (restricted
conformality / identity-anchored penalties on a written fast-weight
state) is unoccupied. NOTE: attack R1 (same day) BLOCKED DRAFT-R0 and
its D3 disposition replaces mechanisms (b)+(c) with an entity-block
conformality penalty ‖AᵀA − (tr/K)I‖² — this stays INSIDE the swept-open
wedge (soft identity/conformal anchoring at write time: no external
occupant found); Rev-1 must record this mapping and any further
mechanism change re-enters the gate.

## By-task sweep (agent, 2026-08-13): OPEN

No work combines: fast-weight matrix as in-context-written OPERATOR +
O(log h) repeated-squaring reads + deep composition in a real LM +
write-time spectral conditioning. Closest three, with distinctions:
- **DeltaProduct (arXiv:2502.10297) — MOST DANGEROUS.** Householder-
  product transition matrices (provably norm ≤1) improve state-tracking
  in the exact DeltaNet lineage. Distinction to state EARLY: their
  orthogonality governs token-by-token state EVOLUTION; ours conditions
  a written operator INSTANCE self-composed h times at query time. No
  repeated-squaring read, no s1/s2 target, no depth≫layers experiment.
- Sanford/Hsu/Telgarsky (2402.09268) + Wang et al. (2505.23683):
  Θ(log k) k-hop composition via transformer LAYER depth (MPC argument)
  — different mechanism class; already on the 07-27 cite list.
- MeSH (2510.07739): spectral collapse under repeated recursive
  application in looped transformers — independent corroboration of the
  collapse phenomenon; their fix is architectural, not spectral.
Also ruled non-occupying: Gated DeltaNet-2 (2605.22791), Preconditioned
DeltaNet (2604.21100), Grazzi et al. eigenvalue-range (ICLR 2025),
POET/POET-X, classic uRNN line, ATLAS (2505.23735 — k-hop eval NOT
present in the paper; misleading search snippet).

## By-mechanism sweep (agent, 2026-08-13): PARTIALLY-OCCUPIED; (b)/(c) wedges OPEN

- **MuonSSM (arXiv:2606.30461, ICML 2026 ORAL) — MOST DANGEROUS.**
  Newton–Schulz orthogonalization applied to the rank-1 fast-weight
  write INSIDE the forward recurrence; framing near-verbatim
  ("spectral anisotropy… spectrally conditioned updates").
  Differentiators: NS-polynomial approximation vs our family;
  motivation = long-sequence training stability vs our O(log h)
  composition-READ capability; no repeated-squaring read, no h-depth
  retrieval evaluation. MANDATORY cite.
- Closest-5 recorded: MuonSSM; DeltaProduct; Preconditioned DeltaNet
  (diagonal eigenvalue bound on write keys); Variational Linear
  Attention (2605.11196 — unit-spectral-norm update Jacobian via
  normalization); Gated DeltaNet-2 (eigenvalue-range gates).
- **OPEN wedges (verbatim from the sweep):** soft anchoring toward c·I
  for a written state — no external occupant; differentiable
  condition-number/restricted-isometry penalty on the written state —
  no external occupant. D3's entity-block conformality penalty lives
  here.
- Exact-orthogonal write-time parametrization (mechanism (a) as
  drafted) is CROWDED (MuonSSM + DeltaProduct) — one of three
  convergent grounds for cutting (a) from wave-1 (see adjudication).

## Internal-archive sweep (agent, 2026-08-13): CLEAN with one load-bearing omission

- Mechanism (a) is NOT a literal dead-direction violation (expm never
  run; it was §10.8's own recommended alternative to the FAILED
  NS-polar) — but the draft omitted the PI-ratified 2026-07-17 §N2
  ruling DEMOTING the whole expm/Cayley track to idle-filler-only.
  Third convergent ground for cutting (a) from wave-1.
- All draft citations verified against sources (§G3-B32 epigraph
  verbatim incl. TPC/o_pc/retrieval figures; c*·I Z-dump numbers exact
  — with the finding's ARCHITECTURE-CONDITIONALITY caveat (empty in
  DeltaNet-family states, fD ≤ 3e-12) to be carried forward
  explicitly; KILL_LIST/NOVEL_ARCH/archive: zero conflicting entries).
- Repo-hygiene flag: EXPERIMENT_LOG has no entries 2026-07-13→07-29
  (ortho-write arc lives only in STATE + design docs). Backfill
  candidate, non-blocking.

## Standing cite-and-distinguish obligations

MuonSSM (2606.30461); DeltaProduct (2502.10297); Preconditioned
DeltaNet (2604.21100); Variational Linear Attention (2605.11196);
Gated DeltaNet-2 (2605.22791); MeSH (2510.07739, corroborating
collapse); Sanford 2402.09268 + Wang 2505.23683 (log-depth
composition); RWKV-7 / Grazzi eigenvalue-range line (diagnostic
precedent); uRNN/scoRNN/expRNN lineage (technique precedent, wrong
object); the 07-27 program-wide gate's obligations carry forward.

## CORRECTION (coordinator, 2026-08-13, post-attack-R2)

Attack R2 (F5) found the "open wedge" claim above INCOMPLETE: the
wedge is OCCUPIED INTERNALLY — the pinned wave-1 runner
(`ncr_lm_wave1_runner.py`) already trains `0.1·‖ZᵀZ−I‖²/d²` on Z in
all three §G3-B31 baselines, so a spectral-conditioning penalty on
the written state is prior art IN OUR OWN ARCHIVE, and the §G3-B32
collapse occurred WITH it active. Externally, R2 adds a functional
identity between the DRAFT-R1 penalty and matrix Rényi-2 entropy
(+ two further counts, see `NCR_WRITECOND_ATTACK_R2.md` F5). Any
future mechanism claim in this family must treat "conditioning
strength/form" as evidenced-against by default and clear BOTH this
internal precedent and the Rényi-2 identity. The by-task OPEN verdict
and the cite-and-distinguish list stand unchanged.

## Re-entry conditions

Any mechanism change beyond D3's entity-block conformality penalty, or
any headline reframe after the premise cell (attack R1 F1) resolves,
re-enters this gate per the 07-16 doctrine.
