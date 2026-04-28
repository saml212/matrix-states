# Paper Writer Agent Brief — ICML MI Workshop 2026

Brief for the agent that will write the final paper when the experimental queue
has drained. Do NOT dispatch this agent until all queued experiments are done
and logged in EXPERIMENT_LOG.md.

## Target venue

**ICML 2026 Mechanistic Interpretability Workshop**
- CfP: https://mechinterpworkshop.com/cfp/
- OpenReview: https://openreview.net/group?id=ICML.cc/2026/Workshop/Mech_Interp
- **Deadline: May 8, 2026 AOE**
- **Double-blind** — strip all names, GitHub handles, HF usernames, pebbleml.com links
- **Non-archival** — NeurIPS double submission allowed later
- **Format: ICML 2026 LaTeX template, 8-page long paper (excluding references)**, unlimited appendix
- Fallback template: https://icml.cc/Conferences/2026/CallForPapers (main-conf style file)
- Reciprocal reviewing: one co-author must sign up for 3 reviews. Budget 6 hours.

## Dual output required

The agent must produce **both**:

### Output 1: Website subpage

- File: `pebble-ai-site/findings/matrix-codi-rank-blindness-paper.html`
- Matches the existing findings-page style in
  `pebble-ai-site/findings/matrix-codi-rank-blindness.html`
- Uses editorial "we" (research-paper convention), NOT first-person "I"
- Full graphs and charts: rank-k curves (inline SVG, one per training regime),
  linear-probe AUC bar chart, depth-sweep line chart, scale-sweep bar chart,
  Jacobian heatmap comparing flatten vs SVD-augmented variants
- Reproducibility section at the bottom with GitHub links to:
  `matrix-thinking/scripts/run_matrix_codi.py`,
  `matrix-thinking/scripts/probe_z.py`, `EXPERIMENT_LOG.md`,
  `matrix-thinking/KILL_LIST.md`, `matrix-thinking/BILINEAR_READOUT_PATCH_PLAN.md`
- Do NOT include the word "audit" anywhere
- Do NOT brag about experiment counts
- Do NOT mention compute budget in dollars or H100-hours
- Do NOT say "we ran 8×H100 pods" — this project uses 1×H100
- Do NOT mention "self-funded"

### Output 2: ICML LaTeX submission

- Directory: `matrix-thinking/submissions/icml-mi-workshop-2026/`
- Files: `main.tex`, `sections/*.tex`, `figures/*.pdf`, `bibliography.bib`,
  `icml2026.sty` (download from ICML 2026 CFP)
- Anonymized: no author names, no institution, no `\acknowledgments`
- Target 8 pages excluding references
- Appendix: unlimited, include all experiment SUMMARY.txt files as raw tables
- Build with `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- Final PDF: `matrix-thinking/submissions/icml-mi-workshop-2026/main.pdf`
- Anonymous code link: Use `https://anonymous.4open.science/` — do NOT include
  the actual GitHub URL in the submission

## Paper thesis

The gradient cannot see rank through a flatten-then-project readout. Matrix-CODI
cannot learn rank-based reasoning structure because
`w_down(Z.flatten())` is linear in Z, the Jacobian is constant, and the gradient
is therefore insensitive to the rank of Z. We demonstrate this with multiple
flat rank-k curves across training regimes, a linear probe gap showing matrix-CODI
encodes reasoning state worse than vanilla GPT-2, and a positive control where
swapping the readout for a nonlinear-in-Z variant (Bilinear+GELU, SVD-augmented,
or quadratic) produces a non-flat rank-k curve. The structural claim is
falsifiable and confirmed.

## Title

"The Gradient Does Not See Rank: A Structural Explanation for the Illusion of
Superposition in Latent Chain-of-Thought"

## Section plan (target 8 pages)

1. **Introduction** (¾ page). CODI / COCONUT baseline. Illusion of
   Superposition (Rizvi-Martel et al.) observed the phenomenon. We give the
   mechanism. Contribution bullets.
2. **Background** (¾ page). CODI distillation recipe. ProsQA reasoning task.
   Rank-k projection ablation as a probe for structured latents.
3. **The Flatten-Then-Project Readout Is Rank-Blind** (2 pages).
   3.1 Observation: four flat rank-k curves across training regimes and scales.
   3.2 Theoretical argument: Jacobian of `w_down ∘ flatten` is constant in Z.
       Gradient in expectation sees only the second-moment of Z, not its rank.
       Formal statement + short proof.
   3.3 Linear probe corroboration: matrix-CODI Z encodes reasoning state at
       AUC 0.673 vs vanilla GPT-2 hidden at AUC 0.846.
4. **Depth and Scale Do Not Rescue Matrix-CODI** (1½ pages).
   4.1 Depth sweep: R8 n=6/16/32/64 on ProsQA. Iterative latent refinement
       alone is also worse than plain SFT.
   4.2 Scale sweep: R6 gpt2-medium, R9 gpt2-large. Scale actively hurts
       vanilla SFT on ProsQA — we discuss why (dataset size) and show matrix
       doesn't catch up either.
5. **Positive Control: Nonlinear Readouts Bend the Curve** (1½ pages).
   5.1 Variant A: MultiProbeHead bilinear. Still flat — confirms reparametrization
       is not the fix.
   5.2 Variant B: Bilinear+GELU. Rank curve bends. (Or: stays flat, discuss.)
   5.3 Variant C: SVD-augmented. Rank curve bends and probe AUC climbs.
   5.4 Variant D: Quadratic (second moment). Rank curve bends.
   5.5 Mini plot: side-by-side rank-k curves for all five readouts.
6. **Related Work** (½ page). CODI, COCONUT, Illusion of Superposition,
   Reasoning by Superposition, SIM-CoT, Dynamics Within Latent CoT.
   **Critical:** distinguish from SIM-CoT (ICLR 2026) — they diagnose as
   insufficient supervision, we diagnose as rank-blind Jacobian. Positive
   control adjudicates.
7. **Discussion and Limitations** (½ page). ProsQA only. One seed per
   positive-control variant (time-bounded). Future work pointers.
8. **Conclusion** (¼ page).

## Figures to generate

- `fig1_rank_curves.pdf` — four flat curves from Rounds 1/2/3/6 on one axis
- `fig2_probe_auc.pdf` — bar chart, matrix vs vanilla linear probe AUC
- `fig3_depth_sweep.pdf` — line chart, R8 n=6/16/32/64 best accuracy
- `fig4_scale_sweep.pdf` — bar chart, gpt2-small/medium/large best accuracy
- `fig5_positive_control.pdf` — side-by-side rank curves, all 5 readout variants
- `fig6_jacobian_diagram.pdf` — schematic of linear vs nonlinear readout Jacobian

## Data sources for the agent

All committed to the repo:
- `EXPERIMENT_LOG.md` — chronological record of every experiment
- `experiment-runs/2026-04-13_*` — SUMMARY.txt files for each round
- `matrix-thinking/KILL_LIST.md` — Lessons 1–4 at the bottom
- `matrix-thinking/ILLUSION_RECIPE.md` — relationship to Rizvi-Martel et al.
- `archive/old-docs/WORKSHOP_PAPER_OUTLINE.md` — earlier draft outline (archived)
- `pebble-ai-site/findings/matrix-codi-rank-blindness.html` — existing website
  version with baseline rank-curve SVG

## Citation targets (verified by research agent)

- CODI: Shen et al. 2025 (EMNLP 2025 main)
- COCONUT: Hao et al. 2024
- Illusion of Superposition: Rizvi-Martel et al. arXiv 2604.06374
- Reasoning by Superposition: Lin/Zhu et al. arXiv 2505.12514 (NeurIPS 2025)
- SIM-CoT: Shen et al. arXiv 2509.20317 (ICLR 2026)
- Dynamics Within Latent CoT: arXiv 2602.08783
- **NEW (Feb 2026) — rank-of-trained-hidden-states, MUST engage:**
  - Nazari & Rusch, "The Key to State Reduction in Linear Attention: A Rank-based
    Perspective", arXiv 2602.04852. Measures effective rank of linear attention
    hidden states; proposes post-training rank pruning of K/Q matrices.
  - Anonymous, "State Rank Dynamics in Linear Attention LLMs", arXiv 2602.02195.
    Finds "State Rank Stratification" — heads bifurcate into persistent low-rank
    and high-rank groups during pretraining.

## Positioning vs Nazari & Rusch and the State Rank Dynamics paper (CRITICAL)

Both papers measure rank in **linear attention hidden states** (fast-weight-style
d×d memory inside the attention layer). They frame rank as a descriptive structural
invariant of trained states. They do not diagnose a mechanism; they observe.

Our work differs on two axes reviewers will check:

1. **Different object of study.** We measure rank in **latent thought matrices**
   (explicit per-position matrix tokens in a CoT-style reasoning model), not
   linear-attention fast-weight memories.
2. **Different claim type.** We make a mechanism claim (the flatten-then-project
   readout has a constant Jacobian in Z, so gradient cannot shape rank). They
   make a descriptive claim. The four positive-control readouts (Bilinear,
   Bilinear+GELU, SVD-augmented, Quadratic) are specifically designed to falsify
   our mechanism claim — they break the constant-Jacobian property. A descriptive
   paper cannot make that claim.

Contribution bullet in intro must be **tightened accordingly**. We are NOT the
first to empirically measure rank in trained transformer states — those two
papers do that for linear-attention states. We ARE the first to diagnose
rank-blindness as a READOUT LINEARITY property, with a falsifiable mechanism
test via nonlinear-in-Z readouts.

Required: read both PDFs before final draft; incorporate any metric definitions
they use that are compatible with ours (effective rank, rank truncation protocol).

## Acceptance criteria

- Both outputs produced
- LaTeX compiles cleanly to PDF
- PDF is 8 pages (excluding references), anonymous
- Website page opens in a browser, all SVGs render, no broken links
- Every numerical claim in the paper has a matching row in EXPERIMENT_LOG.md
- Related work section distinguishes this work from SIM-CoT explicitly
