# Zero-Param Byte Output — Design Audit

## The Idea
At d=16 with byte vocab (256 = 16²), each matrix entry M[i,j] is the logit for byte (16i + j).
257 total output params: 256 bias (unigram prior) + 1 temperature.

## Key Findings

### Temperature is Essential
Matrix magnitudes vary with rank and training. A learned temperature `tau = exp(log_tau)` controls
softmax sharpness. Without it, the model can't balance between sharp (overconfident) and uniform
(underconfident) predictions as representations evolve.

### Bias Matrix Gives Unigram Prior
The 16×16 bias adds a per-byte prior probability. Without it, the model must learn to produce
the byte frequency distribution purely through the thinking layers. The bias handles the easy
part (common bytes like space, 'e', 't' are always likely) and the thinking layers handle context.

### Gradient Flow is Fine
Without an output projection, `dL/dM[i,j] = p[i*16+j] - y[i*16+j]`. This is rank-2 (target
correction + softmax distribution). The gradient norm is O(1) — well behaved. The temperature
controls gradient magnitude: `dL/dM[i,j] = (p - y) / tau`.

### The Hex Mapping is Decent
byte b → (b//16, b%16) gives linguistically coherent structure:
- Digits share row 3
- Uppercase letters in rows 4-5
- Lowercase in rows 6-7
- Case-shifting is a row operation (a=row 6, A=row 4, same column)
- Row sums = P(high nibble) = P(byte class)

### vs Weight Tying
Weight tying: `logit_w = u_w^T M v_w` — bilinear similarity, constrained by embedding structure.
Our approach: `logit_w = M[w//16, w%16]` — reads fixed cell, unconstrained.
Ours has HIGHER expressiveness (256 free logits) but LOWER regularization (no embedding coupling).
The bias partially compensates for the missing regularization.

### Multi-Byte Extension (Future)
A single 16×16 matrix could encode P(nibble_high_byte1, nibble_high_byte2). With two additional
16-vectors for low nibbles: P(b1,b2) = M[h1,h2] * v1[l1] * v2[l2]. Multi-byte from 288 values.

## Implementation
```python
class MatrixByteOutput(nn.Module):
    def __init__(self, d):
        self.bias = nn.Parameter(torch.zeros(d, d))     # 256 params
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # 1 param
    def forward(self, M):
        return (M + self.bias).reshape(B, L, d*d) * self.log_temp.exp()
```

Use flat cross-entropy. Consider row-level auxiliary loss as ablation.
