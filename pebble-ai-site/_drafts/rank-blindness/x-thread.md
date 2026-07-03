---
target: x.com (Twitter)
canonical: https://pebbleml.com/findings/matrix-codi-rank-blindness.html
status: ready_for_review
constraints:
  - 6 tweets. Hook (#1) must stand alone — readers may see only that.
  - Last tweet is the canonical link.
  - Before posting, tag 2–3 researchers from researcher-outreach-list.md whose work is directly relevant (latent CoT / superposition / matrix state). A quote-tweet from any tracked X account is what gets a thread into news.smol.ai's Twitter Recap.
  - Do NOT auto-post. Sam clicks publish.
---

# Tweet 1 (hook — must stand alone)

new result: matrix-valued latent CoT is rank-blind.

I trained a matrix-CODI model with a 16×16 matrix Z at each latent reasoning step. truncated Z to rank k before the readout. accuracy-vs-k curves are flat across four training conditions and three seeds.

🧵

# Tweet 2 (the result)

three seeds, same loss, same data:

· seed 42  → rank 4.0,  81.25% on ProsQA
· seed 7   → rank 12,   82.81%
· seed 1337 → rank 12.9, 80.47%

accuracy 81.51 ± 1.2pp. final effective rank varies 3×. the loss does not reward any particular rank.

# Tweet 3 (the mechanism)

why: the default flatten-then-project readout φ(Z) = W·vec(Z) has Jacobian ∂φ/∂Z = W[:,ij], constant in Z.

the gradient w.r.t. Z is a vector contracted with a constant tensor. it cannot depend on Z's singular structure. the optimizer has no signal that rewards rank.

# Tweet 4 (the falsification attempt that failed)

natural prediction: a nonlinear-in-Z readout should bend the curve.

I trained 4 positive-control variants — bilinear, bilinear+GELU, SVD-augmented (singular values fed through an MLP), quadratic in ZZᵀ.

all 4 rank-k curves still flat. Spearman p ∈ {0.63, 0.14, 0.82, 0.46}.

# Tweet 5 (what is in Z if not rank)

linear probe on Z (1536 feats, 6 positions): AUC 0.673 at predicting the ProsQA target class.

vanilla pretrained GPT-2, never fine-tuned on ProsQA: AUC 0.846 from 768 feats.

the trained matrix bottleneck encodes less target-predictive information than the raw backbone.

# Tweet 6 (link out)

implication: rank-k ablation should not be used as a superposition probe in CODI-trained matrix latents.

full writeup, all figures, code, eval JSONs:
https://pebbleml.com/findings/matrix-codi-rank-blindness.html

ICML 2026 MI workshop submission. comments + holes welcome.
