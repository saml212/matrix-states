---
target: huggingface.co/datasets/pebble-ml/matrix-codi-rank-blindness
canonical: https://pebbleml.com/findings/matrix-codi-rank-blindness.html
status: draft (no model weights to ship for this finding — this is a DATASET card for eval JSONs, not a model card)
license: cc-by-4.0
---

# Matrix-CODI rank-blindness eval data

> **Canonical paper:** [pebbleml.com/findings/matrix-codi-rank-blindness.html](https://pebbleml.com/findings/matrix-codi-rank-blindness.html)
>
> *This research log is maintained by an autonomous agent under Sam Larson's supervision. All claims are verified against experiments run on real hardware.*

This dataset contains the raw evaluation outputs from the ICML 2026 MI workshop submission *"The gradient does not see rank."* It includes rank-k SVD truncation ablation curves for five readout variants of a matrix-CODI model, three-seed accuracy/rank measurements, and the linear probe AUC raw outputs.

## Why a dataset, not a model

The finding is a structural negative result — a rank-blindness diagnosis of the matrix-CODI training objective. Releasing trained checkpoints would invite users to deploy a model that the paper itself argues should not be used in this configuration. Releasing the eval JSONs lets reviewers independently verify the curves without needing the original H100 setup.

If you want to reproduce the *training* runs from scratch, all configs and the training script are in the parent code repository: https://github.com/saml212/learned-representations.

## Files

- `rank_evals/r1_gsm8k_aug.json` — Run 1, GSM8K-Aug, γ=1.0, thinker on. Rank-k ∈ {1, 2, 4, 8, 16}.
- `rank_evals/r2_prosqa_g10.json` — Run 2, ProsQA, γ=1.0, thinker on.
- `rank_evals/r3a_prosqa_g00.json` — Run 3a, ProsQA, γ=0.0, thinker on.
- `rank_evals/r3b_prosqa_g00_no_thinker.json` — Run 3b, ProsQA, γ=0.0, thinker off.
- `rank_evals/seed_replication/{seed_42,seed_7,seed_1337}.json` — Three-seed flatten-readout replication (Figure 1).
- `rank_evals/positive_controls/{bilinear,bilinear_gelu,svd_augmented,quadratic}.json` — Four positive-control readouts (Figure 2, Table 2).
- `linear_probe/probe_outputs.json` — 5-fold CV ProsQA target prediction AUC for matrix Z, vanilla GPT-2 hidden, random GPT-2 hidden.

Each JSON contains: model config, hyperparameters, evaluation seeds, per-k accuracy, effective rank trajectory, Spearman r and p, evaluation set fingerprint.

## How to load

```python
from datasets import load_dataset
ds = load_dataset("pebble-ml/matrix-codi-rank-blindness")
```

## Reproducing the figures

```python
import json, numpy as np
with open("positive_controls/bilinear_gelu.json") as f:
    run = json.load(f)
ks = run["rank_k_values"]
acc = run["accuracy_per_k"]  # list of percentages
# Spearman: scipy.stats.spearmanr(ks, acc)
```

## Citation

```bibtex
@misc{larson2026rankblind,
  title  = {The gradient does not see rank: a structural explanation for the Illusion of Superposition in latent chain-of-thought},
  author = {Sam Larson},
  year   = {2026},
  url    = {https://pebbleml.com/findings/matrix-codi-rank-blindness.html},
  note   = {ICML 2026 MI Workshop submission}
}
```

## License

CC-BY-4.0. Use anywhere with attribution.

## Contact

sam@pebbleml.com
