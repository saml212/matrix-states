---
target: linkedin.com (Sam's personal feed)
canonical: https://pebbleml.com/findings/matrix-codi-rank-blindness.html
status: ready_for_review
audience: lab leads, recruiters, grant reviewers
tone: implications-first; what this means, not just what was done
constraints:
  - Do NOT auto-post.
  - 1 image attached: figure 1 (three-seed scatter) or figure 2 (rank-k curves). Sam picks.
---

A new finding from Pebble, published as an ICML 2026 MI workshop submission:

**The training objective for matrix-valued continuous chain-of-thought is rank-indifferent.**

Continuous reasoning models (COCONUT, CODI) compress chain-of-thought into latent vectors. The 2025 theoretical literature predicts those latents hold multiple reasoning paths in superposition. Matrix-valued latents look like a way to make that measurable — a matrix has *rank*, and rank bounds the number of independent directions a representation uses.

The natural test is to truncate the latent matrix Z to rank k before the readout and see if accuracy drops when the task needs more than k paths. Across four training conditions and three seeds, the accuracy-vs-k curves are flat. Same loss, same data, three seeds: ProsQA accuracy at 81.51 ± 1.2 percentage points while the final effective rank of Z varies 3× across seeds. The optimizer has no signal that rewards any particular rank.

The mechanism is structural. The default readout flattens Z and projects — a Jacobian constant in Z, which means the gradient with respect to Z cannot depend on Z's singular structure. To check whether any nonlinear-in-Z readout fixes this, I trained four positive controls — bilinear, bilinear+GELU, SVD-augmented, quadratic. All four rank-k curves stayed flat. Readout linearity is sufficient but not necessary: rank-indifferent gradients come from the full chain rule under the matrix-bottleneck training objective, not just the readout.

**What this means for the field:**

1. Rank-k ablation should not be used as a superposition probe in CODI-trained matrix latents. The probe cannot reject rank-as-a-functional-property because the training objective never put information there to begin with.

2. This is structurally consistent with the recent *Illusion of Superposition* behavioral result (Rizvi-Martel et al. 2026): they show fine-tuned latent reasoners do not need their latents on ProsQA. We give one architecture-level reason matrix-valued variants of those models cannot encode what the theory predicts they should.

3. The next experiment removes the flatten readout entirely — a fully matrix-native transformer trained from scratch on a task that provably requires K independent scalars of state. If matrix structure is functional, that is where it will show. If it isn't, knowing that is also useful.

Negative results are data. Pebble publishes them with the same care as positive results.

Full paper, figures, code, eval data:
https://pebbleml.com/findings/matrix-codi-rank-blindness.html

---

*Pebble is an independent research lab. The research log is maintained by an autonomous agent under my supervision; all claims are verified against experiments run on real hardware.*
