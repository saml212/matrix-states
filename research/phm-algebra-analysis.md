# PHM Algebra Analysis Methods

Research from agent on 2026-03-24. Key methods for analyzing learned Kronecker factors.

## Most Discriminative Invariants (priority order)

1. **Squared trace signature** `[tr(A_i^2)]` — uniquely distinguishes:
   - Quaternion: [-4, -4, -4] (plus identity +4)
   - Split-quaternion: [-4, +4, +4]
   - Tessarine: [-4, -4, +4]
   - Dual numbers: [0, 0, 0]

2. **Killing form rank** — 3 for quaternion/split-quat, 0 for commutative

3. **Anti-commutativity of generators** — Clifford vs non-Clifford

4. **Nilpotent element count** — dual numbers

5. **Determinant pattern** — division algebras have all non-zero dets

## Key Reference Algebras (n=4)

| Algebra | Clifford | i² | j² | k² | Commutative |
|---------|----------|-----|-----|-----|-------------|
| Quaternion | Cl(0,2) | -1 | -1 | -1 | No |
| Split-quat | Cl(1,1) | -1 | +1 | +1 | No |
| Tessarine | — | -1 | -1 | +1 | Yes |
| Dual numbers | — | 0 | 0 | 0 | Yes |

## Implementation

Full code for `extract_structure_constants()`, `classify_learned_algebra()`,
`detect_clifford_signature()`, and visualization in the research agent output.
Key: compare sorted squared traces + Killing rank + commutativity for basis-independent classification.
