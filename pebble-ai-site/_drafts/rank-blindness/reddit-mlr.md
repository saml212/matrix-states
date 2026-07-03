---
target: r/MachineLearning
flair: "[R] Research"
title: "[R] Matrix-valued latent CoT is rank-blind: four flat rank-k curves and four positive-control readouts that should have bent them"
canonical: https://pebbleml.com/findings/matrix-codi-rank-blindness.html
status: ready_for_review
constraints:
  - DO NOT auto-post. Reddit detects AI-flavored self-promo and kills accounts.
  - Post manually from Sam's account, ideally Tue/Wed 9am PT for max scrape window before news.smol.ai's daily roundup.
  - First reply (within 30 min) should be the GitHub repro link as a comment, not a top-level post element.
---

# Title

[R] Matrix-valued latent CoT is rank-blind: four flat rank-k curves and four positive-control readouts that should have bent them

# Body

I've been running a matrix-valued variant of CODI — each latent reasoning step is a 16×16 matrix Z instead of a vector — to test whether *rank* of Z behaves as a measure of how many parallel reasoning paths a continuous-reasoning model holds in superposition. The 2025 theoretical literature (Reasoning by Superposition, CoT2) makes that prediction; rank-k SVD truncation of Z is the natural empirical probe.

It doesn't work as a probe. Across four training conditions on ProsQA and GSM8K-Aug, the accuracy-vs-k curves are flat to within 0.6pp. A three-seed replication has accuracy at 81.51 ± 1.2pp while the final effective rank of Z varies 3× across seeds — same loss, same data, same accuracy, ranks {4, 12, 12.9}. The training objective has no signal that rewards any particular rank.

The mechanism: the default flatten-then-project readout φ(Z) = W·vec(Z) has a Jacobian that's constant in Z, so the gradient with respect to Z cannot depend on Z's singular structure. To check whether *any* nonlinear-in-Z readout fixes this, I trained four positive-control variants with everything else held identical: bilinear reparam, bilinear+GELU, SVD-augmented (singular values explicitly fed through an MLP via torch.linalg.svdvals), and quadratic (second moment ZZᵀ). All four rank-k curves stay flat. Spearman p-values 0.63, 0.14, 0.82, 0.46. Readout linearity is sufficient but not necessary — the rank-indifferent gradients come from the full chain rule under the matrix-bottleneck training objective, not just the readout.

A linear probe on the trained Z (1536 features, concat across 6 latent positions) predicts the ProsQA target class at AUC 0.673. Vanilla pretrained GPT-2 — never trained on ProsQA — does it at AUC 0.846 from 768 features. Pre-registered threshold for a positive result was 0.896.

Implication: rank-k ablation shouldn't be used as a superposition probe in CODI-trained matrix latents. The ablation can't reject rank-as-a-functional-property because the objective never put information there to begin with. The result sits parallel to the Illusion of Superposition behavioral finding (Rizvi-Martel et al. 2026, arXiv:2604.06374) — they show the latents can be removed without much accuracy loss; this gives a structural reason matrix-rank latents fail to encode the property the theory predicts.

ICML 2026 MI workshop submission. Full paper, figures, code, eval JSONs:

- Paper: https://pebbleml.com/findings/matrix-codi-rank-blindness.html
- Code + eval data: https://github.com/saml212/learned-representations

Happy to discuss the refined hypothesis (effectively rank-1 active subspace in the trained Jacobian) or the dimension-matched probe deferred to camera-ready.
