---
target: ad-hoc Discord servers (research/papers channels)
canonical: https://pebbleml.com/findings/matrix-codi-rank-blindness.html
status: ready_for_review
servers_to_consider:
  - EleutherAI (#interpretability, #papers)
  - Latent Space (the news.smol.ai community)
  - CUDA MODE (lower fit; only if speed/kernel angle ever applies)
  - Nous Research (#general; weights-oriented community, lower fit until weights ship)
constraints:
  - Manual paste only. Do not attempt to automate Discord.
  - One server at a time. Read the room first; if the channel is on a different topic, wait.
  - Lead with the question, not the brag.
---

# Long form (for #papers channels that want context)

new finding from a small lab — would value structural critique:

I trained a matrix-CODI model (CODI fork with a 16×16 matrix Z at each latent reasoning step) to test whether the rank of Z behaves as a measure of parallel reasoning paths under the superposition hypothesis. across 4 training conditions and 3 seeds, rank-k SVD truncation of Z gives flat accuracy curves. mechanism: the flatten-then-project readout's Jacobian is constant in Z, so gradients can't depend on Z's singular structure. four positive-control readouts nonlinear in Z (bilinear+GELU, SVD-augmented, quadratic) all also produce flat curves — readout linearity is sufficient but not necessary.

linear probe: trained Z (1536 feats) hits AUC 0.673 on ProsQA target class. vanilla pretrained GPT-2 (768 feats, never fine-tuned) hits 0.846.

ICML 2026 MI workshop submission. interested in counterexamples or refinements of the active-subspace hypothesis (which is what I'm chasing for camera-ready).

paper + code + eval JSONs: https://pebbleml.com/findings/matrix-codi-rank-blindness.html

# Short form (for #links / #share / general channels)

new ICML MI workshop submission — matrix-valued latent CoT is rank-blind. four flat rank-k curves, three seeds at the same accuracy with 3× rank spread, four positive-control readouts that should have bent the curve and didn't.

https://pebbleml.com/findings/matrix-codi-rank-blindness.html
