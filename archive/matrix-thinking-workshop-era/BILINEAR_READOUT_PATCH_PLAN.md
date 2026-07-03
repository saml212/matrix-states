# Bilinear Readout Patch Plan (Positive Control for Paper)

Four readout variants to test the Jacobian/linearity thesis falsifiably.
None of this code exists yet. This document describes the patch, not the
implementation.

## File to patch

`matrix-thinking/scripts/run_matrix_codi.py` — local copy exists (2,192 lines).
After patching, sync to pod at `/workspace/pebble/round3_gamma0/scripts/`.

## Current readout (lines 274–277 of MatrixBottleneck.forward)

```python
flat_out = Z_out.reshape(B, d * d)      # (B, d²)
h_out = self.w_down(flat_out)           # Linear(d², D)
h_out = self.out_norm(h_out)            # LayerNorm(D)
```

Linear in Z. Jacobian ∂h_out/∂Z is constant (equal to `w_down.weight` reshaped).
Gradient ∂L/∂Z is insensitive to any rank structure of Z in expectation.

## Variant A — MultiProbeHead-style bilinear (CLAUDE.md rule #17)

Reparametrization of the current readout using K bilinear probes. Still linear
in Z. Expected outcome: **flat rank curve.** Serves as the reparametrization
control — proves the problem is not solved by "making it matrix-native" alone.

```python
# __init__ additions
self.U = nn.Parameter(torch.randn(K, d) * (1 / math.sqrt(d)))
self.V = nn.Parameter(torch.randn(K, d) * (1 / math.sqrt(d)))
self.out = nn.Linear(K, hidden_dim, bias=True)

# forward
MV = torch.einsum('bij,kj->bik', Z_out, self.V)    # (B, d, K)
probes = torch.einsum('ki,bik->bk', self.U, MV)    # (B, K), scalar u_k^T Z v_k
h_out = self.out(probes)                           # (B, D)
h_out = self.out_norm(h_out)
```

K = d² = 256 gives full expressiveness (same function class as flatten+Linear).
Run K=d²=256 and K=d=16 to separate expressiveness from parametrization.

## Variant B — Bilinear + GELU + Linear (nonlinear in Z)

Adds one nonlinearity between probes and output. Minimal deviation from Variant
A that breaks linearity in Z. Direct test of the Jacobian argument. Expected
outcome: **rank curve bends** if the theory is right.

```python
# __init__ same as Variant A
# forward
MV = torch.einsum('bij,kj->bik', Z_out, self.V)
probes = torch.einsum('ki,bik->bk', self.U, MV)
h_out = self.out(F.gelu(probes))        # <-- nonlinearity
h_out = self.out_norm(h_out)
```

## Variant C — SVD-augmented readout (explicit rank features)

Concatenates the flatten path with singular values projected through an MLP.
Most direct way to put rank information into the gradient path. SVD in fp32
outside autocast (same as existing truncate_to_rank). Expected outcome: **rank
curve bends and linear probe AUC jumps.**

```python
# __init__ additions
self.sigma_proj = nn.Sequential(
    nn.Linear(d, 4 * d),
    nn.GELU(),
    nn.Linear(4 * d, hidden_dim),
)

# forward
with torch.autocast("cuda", enabled=False):
    sigma = torch.linalg.svdvals(Z_out.float()).to(Z_out.dtype)   # (B, d)
flat_out = Z_out.reshape(B, d * d)
h_out = self.w_down(flat_out) + self.sigma_proj(sigma)
h_out = self.out_norm(h_out)
```

## Variant D — Quadratic (second-moment) readout

Feeds Z Zᵀ and Zᵀ Z into a linear layer. Quadratic in Z so linearity is broken
at the readout entry. Z Zᵀ has eigenvalues σ², so rank is encoded in the
spectrum of the input. Expected outcome: **rank curve bends.**

```python
# __init__ additions
self.w_down_quad = nn.Linear(2 * d * d, hidden_dim, bias=True)

# forward
ZZt = torch.einsum('bij,bkj->bik', Z_out, Z_out)      # (B, d, d)
ZtZ = torch.einsum('bji,bjk->bik', Z_out, Z_out)      # (B, d, d)
quad = torch.cat([ZZt.reshape(B, d*d), ZtZ.reshape(B, d*d)], dim=-1)
h_out = self.w_down_quad(quad)
h_out = self.out_norm(h_out)
```

## CLI flag

Add `--readout {flatten,bilinear,bilinear_gelu,svd_aug,quadratic}`, default
`flatten` (preserves existing behavior). Select variant in `MatrixBottleneck.__init__`.

## Init stability

All new parameters must be zero-init or small-normal (std=0.02) for the
residual to start near zero, matching the existing `w_down` init pattern.
`sigma_proj` final Linear: zero-init. `out` in Variants A/B: zero-init.
`w_down_quad` in Variant D: zero-init.

## Experiment matrix (4 runs)

Each variant runs the Round 3 gamma=0 config — same dataset (ProsQA), same
seed, same epochs, same everything except the readout. After training, run the
rank-k projection ablation at inference to produce the rank-k curve plot.

| Run | Variant       | Expected rank curve | Tests           |
|-----|---------------|---------------------|-----------------|
| C1  | flatten       | flat (baseline)     | (already have)  |
| C2  | bilinear      | flat                | reparametrization |
| C3  | bilinear_gelu | bends               | Jacobian/linearity |
| C4  | svd_aug       | bends               | explicit rank   |
| C5  | quadratic     | bends               | second-moment   |

C1 is already on disk from Round 3. C2–C5 are new.

## Linear probe post-hoc

Re-run the probe_z.py linear probe (matrix-CODI AUC 0.673 vs vanilla 0.846)
on each new checkpoint. If C3/C4/C5 produce non-flat rank curves AND probe
AUC climbs toward 0.846, the paper's story is: diagnosed + fixed.

## Deployment sequence (after user approves code)

1. Patch local `matrix-thinking/scripts/run_matrix_codi.py` with all variants.
2. Smoke test CPU forward+backward on each variant (5 steps, batch=2).
3. scp the patched script to pod `/workspace/pebble/round3_gamma0/scripts/`.
4. Append C2–C5 to `/workspace/pebble/queue.txt`.
5. Master queue picks them up after current R6/R7/R8/R9 queue drains.

## Not in scope for this patch

- Dataset changes (ProsQA only for positive control, to match C1)
- Base model changes (gpt2-small only)
- Thinker network modifications
- New LR schedules
