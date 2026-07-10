# Paper outline — UniReps 2026 EA ("dimension, not solvability")

## Section plan

| # | Section | Pages | Claims (ids) | Figures | Notes |
|---|---|---|---|---|---|
| 1 | Introduction | 0.70 | (previews U1, U2, U3) | — | the convergence question; groups as probes with algebraic ground truth; contributions |
| 2 | Task, Model, and Instrument | 0.85 | — | — | word problem; encoder; P=1; restricted eff. rank; degauged recovery; pre-registration |
| 3 | Convergence to the Algebraic Minimum | 0.90 | U1, U5 | fig1, Table 1 | M1 table; ρ tie-cap + exact null; band; robustness split footnote |
| 4 | Equivalence at Matched Dimension | 0.60 | U2 | fig1 inset | TOST design + power; DECLARE; what separation would have meant |
| 5 | The Causal Check | 0.55 | U3, U4, U6 | fig2, Table 2 | razor table; S3 extension; D-AMB in brief |
| 6 | Related Work | 0.25 | — | — | five by-name distinctions |
| 7 | Limitations | 0.15 | — | — | scale; n; disclosures |
|   | **Total** | **4.0** | | | 4pp + refs (venue limit flagged in brief) |

## Outline sanity checks

- [x] Budgets sum to 4.0 (excl. references).
- [x] Every claim id U1–U6 appears in exactly one section (U1/U5 → §3; U2 → §4; U3/U4/U6 → §5).
- [x] Both figures placed (fig1 → §3–4, fig2 → §5).
- [x] Related work distinguishes nearest neighbors BY NAME.
- [x] No section carries a claim with a missing/pending evidence row.

## Per-section beat sheet

### 1. Introduction
- UniReps question: when do differently structured systems learn the same representation? Here the "systems" axis is the task algebra itself: five groups, two computational classes, one architecture.
- Representation theory supplies an exact prediction for what convergence should look like: d_min, the minimal faithful real representation dimension.
- Preview: rank converges to d_min (ρ=0.9747, the design maximum); the matched-dimension solvable/non-solvable pair is TOST-equivalent; a causal force-rank razor confirms d_min is load-bearing.
- Contribution bullets.

### 2. Task, Model, and Instrument
- Word problem on Cayley-graph random walks; target ρ_G(product) ⊕ I, fixed non-learned readout; groups table.
- Encoder → single matrix state (P=1, blank-out verified); d_state = d_min+2.
- Restricted effective rank (centered covariance, own dominant subspace, entropy rank); degauged recovery; fit/eval split; pre-registered bands/margins/falsifiers.

### 3. Convergence to the Algebraic Minimum
- Table 1 + fig1: 19 seeds, five groups, [0.7,1.3]·d_min band, means within 4–8% of d_min.
- ρ=0.9747 honesty paragraph: tie-capped maximum, exact-null 8/120, corroborating tier.
- Footnote: L≥2 robustness split near-identical (U5).

### 4. Equivalence at Matched Dimension
- Why S4 vs A5 is the designed head-to-head (solvable vs smallest non-solvable, same d_min).
- TOST: margin ±0.5 rank-units, n=5/5, pre-run power simulation; result |Δ|=0.019, t 13.06/14.12 vs 1.865 → DECLARE.
- Interpretation: convergence is governed by representation dimension, not the complexity divide (Merrill et al.'s axis).

### 5. The Causal Check
- Razor grid + Table 2/fig2: 0.000 below d_min everywhere (all 4 S3 seeds); recovery returns at d_min (bars); S3 fixed-literal bar + seed-mean.
- D-AMB paragraph (U6): the tax, the ceiling match, INCONCLUSIVE registered, fix wave landed the prediction — measurement lesson for convergence claims.

### 6. Related Work
- Five by-name distinctions from the brief.

### 7. Limitations
- Sub-1M synthetic; d_state window; n=1 razor cells except S3; soft-convergence disclosures; the sibling EA and full-paper trajectory.
