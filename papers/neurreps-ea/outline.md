# Paper outline — NeurReps 2026 EA ("the causal rank law")

## Section plan

| # | Section | Pages | Claims (ids) | Figures | Notes |
|---|---|---|---|---|---|
| 1 | Introduction | 0.65 | (previews N1, N4, N6) | — | rank as the geometric quantity; state the law; contributions |
| 2 | Tasks, Models, and the Rank Instrument | 0.85 | — | — | binding task; group word task; encoder; P=1; exact continuous recovery; restricted eff. rank; degauging + C1 pin |
| 3 | Provable and Recruited Rank on Associative Binding | 0.50 | N6 | — | Fact 1; recruited rank tracks K; d=8,K=4 causal step |
| 4 | The Rank Law on Group Composition, Observed | 0.50 | N4, N5 | fig2 | M1 table; ρ tie-cap + exact null; marquee TOST |
| 5 | The Causal Razor | 1.00 | N1, N2, N3, N7, N8 | fig1, Table 2 | decisional table; S3 extension; D-AMB tax + fix wave |
| 6 | Related Work | 0.30 | — | — | five by-name distinctions from the brief |
| 7 | Limitations and Outlook | 0.20 | — | — | scale; n=1 fix-wave cells; soft convergence; depth generalization next |
|   | **Total** | **4.0** | | | 4pp + refs (venue limit flagged in brief) |

## Outline sanity checks

- [x] Budgets sum to 4.0 (excl. references).
- [x] Every claim id N1–N8 appears in exactly one section (N4/N5 → §4; N6 → §3; N1/N2/N3/N7/N8 → §5).
- [x] Both figures placed (fig2 → §4, fig1 → §5).
- [x] Related work distinguishes nearest neighbors BY NAME (five, from brief).
- [x] No section carries a claim with a missing/pending evidence row.

## Per-section beat sheet

### 1. Introduction
- Matrix/fast-weight states make rank the natural geometric budget; prior work bounds it descriptively or constructs capacity by hand.
- The question: does SGD recruit the rank the task's algebra demands, and is that rank causal?
- Two testbeds, one discipline (exact continuous recovery, hard single-state bottleneck): binding (rank ≥ K provable) and group words (rank = d_min, representation theory supplies the ground truth).
- Contribution bullets (from brief), causal razor first.

### 2. Tasks, Models, and the Rank Instrument
- Binding task in two sentences (K bindings → single Z → literal unbind, cosine bar, never argmax; why argmax is a loophole: Nichani et al.).
- Group word task: symmetric-generator random walks on the Cayley graph; target ρ_G(product) ⊕ I (fixed, non-learned readout); groups table (d_min, solvability, |G|).
- Encoder: Transformer-encoder row-reader → Z ∈ R^{d_state×d_state}, d_state = d_min+2; P=1 bottleneck verified by gradient blank-out.
- Instrument: restricted effective rank (centered covariance → own dominant d_min-subspace → entropy effective rank); degauged recovery rec@0.9; the pre-registered decisional-metric pin (full-Q Procrustes crosscheck) for force-rank cells.
- Pre-registration: bands, grids, falsifiers fixed before data.

### 3. Provable and Recruited Rank on Associative Binding
- Fact 1: exact recovery of K independent bindings forces rank(Z) ≥ K (three-line argument).
- SGD recruits it: d=16 effective rank 8.20 @ K=8, 15.08 @ K=16 (N6).
- Causal: d=8, K=4 — rank ≤3 → ≤0.0004, rank 4 → 0.940 (N6).
- One sentence on composition (Z^h exact to depth 21) as the operator-quality signal; one sentence pointing the reader to the full draft for Task E / frontier.

### 4. The Rank Law on Group Composition, Observed
- M1 table (five groups, means ± sd, band) — N4.
- ρ = 0.9747 with honesty: tie-capped max; exact-null P(ρ≥0.8)=8/120; corroborating tier by pre-registration.
- Marquee TOST — N5: design (margin, n=5, power), result (7× critical), meaning (dimension, not solvability).

### 5. The Causal Razor
- The grid: k ∈ {d_min−1, d_min, d_min+1} spectral truncation at train time; falsifier stated.
- The decisional table (N1) + fig1; necessity exactly 0.000 (N2); sufficiency vs 0.9×anchor bars.
- S3: marginality trigger fired, seed extension, fixed-literal bar, seed-mean 0.5625 (N3); disclosed per-seed secondary read.
- The D-AMB story (N7): eye-padded target = rho ⊕ I, √(k/d_state) ceiling match (mean |Δ|=0.028/39 cells), INCONCLUSIVE registered, zero-pad fix wave landed the pre-registered step; config-integrity note (N8).

### 6. Related Work
- The five by-name distinctions from the brief (one sentence each).

### 7. Limitations and Outlook
- Sub-1M-parameter synthetic testbeds; d_state = d_min+2 window; fix-wave cells n=1 except S3 (n=4); soft-convergence disclosures at the 8K-step groups; depth generalization (Stage 2) is designed and gated on this readout.
