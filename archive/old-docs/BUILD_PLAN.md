# Build Plan: H100 Experiment Queue

## Work Loop (for each variant)
1. Check this doc for next unchecked variant
2. Write H100 version of the script (FA3, DDP, torch.compile, bfloat16)
3. Write Mac Mini version (MPS or CPU, smaller params, fewer steps)
4. Send audit agent to review the code changes for that variant
5. Run Mac Mini version with minimal params until it works
6. Verify output data is correct and readable
7. Check off the variant, move to next

## Variant Queue

### Experiment 1: Base architecture comparison
- [x] 1a. Matrix V3 (Kronecker K=4, mat_dim=16, 6 layers) — VERIFIED on Mac Mini
- [x] 1b. Standard vector transformer (matched params) — VERIFIED on Mac Mini
  NOTE: param matching needed. Vector d=128 4-layer = 13.8M, Matrix d=8 2-layer = 1.2M.
  For H100: find d_model that gives same params as Matrix V3 at mat_dim=16 6-layer.
- [x] 1c. Matrix V2 (flatten→linear, current code) — VERIFIED on Mac Mini

### Experiment 2: Matrix-native operation sweep
- [ ] 2a. Kronecker K=4 projections
- [x] 2b. Householder K=4 reflections — VERIFIED on Mac Mini
- [x] 2c. RowThenColumn with nonlinearity (gelu(A@M)@B) — VERIFIED. Best mini PPL so far (20164 vs Kronecker 36765)
- [x] 2d. Bilinear A@M@B (simplest) — VERIFIED. PPL 42881.

### Experiment 3: Activation function sweep
- [ ] 3a. SwiGLU (current standard)
- [x] 3b. GELU (previous default) — VERIFIED. PPL 33544 (beats SwiGLU Kronecker 36765)
- [x] 3c. No activation (pure multiplicative nonlinearity) — VERIFIED. PPL 39912.

## Mini Test Results Summary (200 steps, ~1.2M params each, WikiText-103 tokenized)
```
MATRIX-NATIVE OPERATIONS (all ~1.2M params):
2c. RowThenColumn gelu(A@M)@B:   20,164 PPL  ← BEST matrix-native
4c. Iterative shared ×8:         31,088 PPL  ← BEST iterative
4b. Iterative shared ×4:         31,994 PPL
3b. GELU bottleneck:             33,544 PPL
1a. Kronecker K=2 + SwiGLU:      36,765 PPL
3c. No activation:               39,912 PPL
2d. Bilinear A@M@B:              42,882 PPL
2b. Householder K=2:             44,138 PPL

DIFFERENT PARAM COUNTS (not directly comparable):
1c. V2 flatten (4.1M params):     1,664 PPL
1b. Vector (13.8M params):        1,281 PPL
```

### Experiment 4: Iterative refinement
- [ ] 4a. 6 unique layers (no sharing, current)
- [x] 4b. 1 shared layer × 4 iterations — VERIFIED. PPL 31994 (beats non-iterative 36765)
- [x] 4c. 1 shared layer × 8 iterations — VERIFIED. PPL 31088 (beats 4-iter 31994)

## Script Structure (same for all variants)
```
h100_scripts/
  exp1a_matrix_v3.py          # H100 version (FA3, DDP, compile)
  exp1a_matrix_v3_mini.py     # Mac Mini version (MPS, small)
  exp1b_vector_baseline.py
  exp1b_vector_baseline_mini.py
  ...
  common.py                   # Shared: data loading, logging, eval
  run_all.sh                  # Launches all H100 experiments in sequence
```

## Data
- WikiText-103 tokenized with HuggingFace BPE tokenizer
- Need to set up before first script runs
- Store tokenized data on SSD

## Output Format (same for all experiments)
```json
{
  "experiment": "1a_matrix_v3",
  "params": 50000000,
  "best_val_ppl": 35.1,
  "best_val_loss": 3.56,
  "training_time_s": 1800,
  "steps": 50000,
  "config": { ... },
  "training_curve": [
    {"step": 1000, "train_loss": 4.2, "val_loss": 4.0, "val_ppl": 54.6},
    ...
  ]
}
```
