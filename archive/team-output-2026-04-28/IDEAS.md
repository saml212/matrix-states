# Ideator: 6 directions OUTSIDE the proposed 3x3

The proposed 3x3 (task × loss) extends the paper's existing frame. Below are 6
directions that would either (a) close the paper's own promised but unrun
experiment cheaper, (b) test a substrate the paper explicitly does not test
(SAE / vocab-column), or (c) attribute the seed-spread to a specific channel.
All can use existing matrix-CODI checkpoints unless noted.

## 1. Jacobian effective-rank measurement (paper §5.3's own unrun experiment)

The paper *literally promises* this as camera-ready: "compute J(Z)=∂φ/∂vec(Z)
at test inputs for each positive-control checkpoint and report erank(J)
averaged over test examples. Predicts ≈1; ≥4 falsifies." Nobody has done it.
Just measurement on existing 5 checkpoints (flatten, bilinear, bilinear+GELU,
svd-aug, quadratic), no retraining. **Novelty:** turns the §5.3 candidate
hypothesis from speculation into a measured result. **Falsifiable:** erank(J)≥4
on ANY positive control kills the refined hypothesis and reopens mechanism.
**Cost: ~$3 (one H100h, single measurement script).** Highest information per
dollar of any direction here.

## 2. Sparse autoencoder on matrix-CODI Z (Wang 2025 substrate test)

Train an SAE (e.g. JumpReLU, l1-penalized) on the flattened Z latents extracted
from the Round 3 γ=0 checkpoint at all 6 latent positions over 17,886 ProsQA
examples. **Novelty:** no prior work has trained an SAE on continuous CoT
latents — this directly tests Wang 2025's alternative-substrate hypothesis
(superposition lives in feature/vocab-column basis, not SVD basis) on Matrix-CODI
specifically. If the SAE recovers crisp ProsQA-relevant features (e.g.
target-class indicator features, distractor features) where rank-k truncation
recovered nothing, the paper's framing flips: rank is the wrong observable, not
the loss. **Falsifiable:** if SAE features have no task-relevant interpretation
above the random-h baseline, Wang's substrate is also dead in matrix-CODI.
**Cost: ~$20 (extract Z, train SAE, probe SAE features for target-class).**

## 3. Adam-vs-SGD-vs-wd=0 ablation as direct seed-spread channel test

The paper's mystery: same loss, same data, same arch, three seeds at effective
rank {4, 12, 13}. The 3x3 doesn't address this at all. Ran 3 seeds × {AdamW
wd=0.01 (current), AdamW wd=0, SGD+momentum wd=0.01}. **Novelty:** attributes
the seed-spread to a specific implicit-bias channel (vs vague "implicit bias").
"Rich and the Simple" (NeurIPS 2025) predicts SGD tightens to ~rank 4. Kobayashi
2024 (already cited) predicts wd=0 → high rank. **Falsifiable:** if all three
optimizers reproduce the {low, mid, high} spread, implicit-bias-from-optimizer
is NOT the mechanism — pushes attention to data-dependent trajectory chaos.
**Cost: 9 runs × ~1.5 H100h ≈ $25 spot.** Standalone workshop note plus paper
appendix.

## 4. Factored matrix-CODI: born rank-2

Reparametrize the bottleneck so $Z = U V^\top$ with $U, V \in \mathbb{R}^{d
\times r}$ for $r \in \{1, 2, 4\}$ — Z is born rank-r. Train at r=2 on ProsQA.
**Novelty:** factored continuous-CoT bottlenecks have not been tested. Forces
the question "if the model is GIVEN rank ≥ 2 capacity at no extra cost, does
ProsQA train rank-1-truncation-vulnerable solutions?" If r=2 model still
flat-truncates at k=1, the rank-1-solvable-task alternative explanation
(paper §7) becomes much harder to deny. If accuracy *drops* at k=1 only when
r=2, the paper's claim "objective doesn't reward rank" is sharpened, since now
the architecture doesn't reward rank either, but Z still uses both directions.
**Falsifiable:** clear: r=2 trained model's rank-1 truncation drop > 1pp ⇒
rejects strong rank-blindness. **Cost: 4 configs × 3 seeds × 1.5 H100h ≈ $30.**

## 5. Learned-projector rank-k retention (Li & Janson optimal ablation)

The paper cites Li & Janson 2024: "zero/resample ablations overestimate
component importance vs optimal ablation." Apply: instead of top-k SVD
truncation of Z, learn a rank-k projector $P_k$ that maximally preserves the
unablated logits over 500 train examples, then evaluate on test. **Novelty:**
none of the rank-k literature has done learned-projector retention on
continuous-CoT latents. If the learned projector at k=1 also recovers full
accuracy, the rank-k probe is genuinely uninformative (the paper's claim).
If a learned k=2 projector dramatically beats the natural top-2, the natural
SVD axes were just the wrong basis — the model uses 2 directions but they're
not the dominant SVs. This directly addresses the paper's citation of Li &
Janson and could be a new diagnostic tool. **Cost: ~$5, no retraining.**

## 6. Rank-k on Rizvi-Martel's released COCONUT 96.6% checkpoint

The paper itself flags this in §6: "rank-k ablation on Rizvi-Martel's released
checkpoint is a natural camera-ready experiment that would test whether the
same rank-blindness holds at the 96.6% ceiling." Their COCONUT uses *vector*
latents — reshape 768 → 24×32 fake matrix to compute SVD. **Novelty:** moves
the mechanism claim out of matrix-CODI specifically and into latent-CoT
generally. If their 96.6% model also has flat rank-k, the paper's framing
broadens substantially. If it bends, the matrix-CODI finding is architecture-
specific. **Falsifiable; one-day experiment.** **Cost: ~$5 (one H100h on their
checkpoint).**

---

## Recommendation

**The 3x3 is fine but it's safe.** It extends the paper's existing frame (loss
function variants × task variants) and produces a 9-cell factorial paper
update. It's $90 well spent, but it does NOT make a more novel paper.

**The single highest-leverage direction is #1 (Jacobian erank measurement).**
The paper *itself* writes "we have not yet run this" and predicts the answer.
~$3 compute on existing checkpoints. If erank(J)≈1, the paper has a complete
refined mechanism for the camera-ready and the §5.3 hedge becomes a
result-grade contribution. If erank(J)≥4, the paper's candidate hypothesis is
falsified and a new mechanism is needed *before* submission, not after — that
is the kind of result you want to know now, not after camera-ready review.
This single measurement is the difference between a paper with an open
question and a paper with a complete falsifiable mechanism story. **Do this
first regardless of whether the 3x3 runs.**

If forced to pick one *new direction* to replace part of the 3x3: **#2 (SAE on
Z)** would make a more novel paper. The 3x3 is rank-axis variation; SAE asks
"is there ANY structure in Z, even though rank says no?" That's a different
paper and could connect Matrix-CODI to mainstream interpretability.

Compute math: #1 ($3) + #5 ($5) + #2 ($20) = **$28 total** for three new
results that the 3x3 ($90) does not produce. Skip #3, #4, #6 for v1.

AGENT_DONE
```json
{"key_findings": ["Jacobian erank measurement (paper's own unrun §5.3 experiment)", "SAE on matrix-CODI Z latents (Wang 2025 substrate test)", "Adam-vs-SGD seed-spread channel ablation", "Factored born-rank-r matrix-CODI", "Learned-projector rank-k retention (Li & Janson optimal ablation)", "Rank-k on Rizvi-Martel released COCONUT 96.6% checkpoint"], "recommendation": "Run #1 (Jacobian effective-rank measurement) regardless. ~$3, 1 H100h, no retraining. The paper's §5.3 candidate refined mechanism is currently unmeasured speculation; this turns it into a measured result and is the single highest-information-per-dollar action available. The 3x3 is a safe extension of the existing frame; #1+#5+#2 ($28) produce three NEW results the 3x3 does not — including testing Wang 2025's alternative substrate hypothesis directly.", "top_idea_compute_dollars": 3}
```
