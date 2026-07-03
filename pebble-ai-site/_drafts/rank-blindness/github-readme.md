# The gradient does not see rank

A structural explanation for the Illusion of Superposition in latent chain-of-thought.

> **Canonical:** [https://pebbleml.com/findings/matrix-codi-rank-blindness.html](https://pebbleml.com/findings/matrix-codi-rank-blindness.html)

ICML 2026 Mechanistic Interpretability workshop submission. Author: Sam Larson (pebble, San Francisco).

## Result

Across four training conditions of a matrix-CODI model (CODI fork with a 16×16 matrix bottleneck on each latent reasoning step), the rank-k SVD truncation ablation is flat to within 0.6pp. Three seeds of the same configuration land at effective ranks {4, 12, 12.9} while reaching matching accuracy (81.51 ± 1.2pp on ProsQA). Four positive-control readouts nonlinear in Z — bilinear+GELU, SVD-augmented, quadratic — all also produce flat curves. A linear probe on the trained Z underperforms a vanilla pretrained GPT-2 hidden state at predicting the ProsQA target class (AUC 0.673 vs 0.846).

The default flatten-then-project readout's Jacobian is constant in Z, which forbids the optimizer from rewarding any particular rank. The full chain rule under the matrix-bottleneck training objective produces rank-indifferent gradients even when the readout is in-principle nonlinear in Z.

## Files in this folder

- `paper.md` — markdown mirror of the canonical HTML.
- `repro/run_rank_eval.py` — minimal single-GPU repro script. Loads a checkpoint and reproduces the rank-k ablation curve. ~15 minutes on a T4 or M1 Mac (CPU mode supported).
- `repro/probe_z.py` — linear probe replication. Reproduces Figure 5 numbers in ~5 minutes.
- `data/rank_evals/` — raw rank-k evaluation JSONs for the four runs in Table 1 and the five readouts in Table 2.
- `figures/` — SVG and PDF of the four figures.

## Reproduce

The full training pipeline lives in the parent research repo (training takes ~6 H100-hours per run). The repro scripts in this folder load published checkpoints and reproduce the *evaluation* — the ablation curve and the linear probe — in under 1 GPU-hour total, including on a laptop.

```
git clone https://github.com/pebble-ml/matrix-codi-rank-blindness
cd matrix-codi-rank-blindness/repro
pip install -r requirements.txt
python run_rank_eval.py --readout flatten --device cuda    # or --device cpu
python probe_z.py --checkpoint ../checkpoints/r3a-flatten/
```

> **NOTE FOR HANDOFF:** Sam — the `repro/` scripts in this folder are SKELETONS only. They have not been run yet. Per `CLAUDE.md` rule "Do not generate scripts that claim to reproduce experiments you haven't actually verified run", these need to be smoke-tested against a published checkpoint before this README is published. See `SHIP_RUNBOOK.md` step 4.

## Cite

```bibtex
@misc{larson2026rankblind,
  title  = {The gradient does not see rank: a structural explanation for the Illusion of Superposition in latent chain-of-thought},
  author = {Sam Larson},
  year   = {2026},
  url    = {https://pebbleml.com/findings/matrix-codi-rank-blindness.html},
  note   = {ICML 2026 MI Workshop submission}
}
```

## Full training code

The training script that produced the four runs in Table 1 and the five readouts in Table 2 is in the main research repository:

- Training: [`matrix-thinking/scripts/run_matrix_codi.py`](https://github.com/saml212/learned-representations/blob/main/matrix-thinking/scripts/run_matrix_codi.py) — all five readouts selectable via `--readout`.
- Probe: [`matrix-thinking/scripts/probe_z.py`](https://github.com/saml212/learned-representations/blob/main/matrix-thinking/scripts/probe_z.py).
- Run logs: [`experiment-runs/2026-04-17_round_pc/`](https://github.com/saml212/learned-representations/tree/main/experiment-runs/2026-04-17_round_pc).

## License

Code: MIT. Eval data: CC-BY-4.0. Paper: CC-BY-4.0.

## Contact

sam@pebbleml.com — happy to discuss counterexamples, the refined active-subspace hypothesis (camera-ready work), or related results.
