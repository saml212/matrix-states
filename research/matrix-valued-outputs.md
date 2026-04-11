# Matrix-Valued Outputs for Byte-Level Models

Research from agent on 2026-03-24.

## The Gap We Can Fill

Nobody has combined **algebraically structured (Kronecker/PHM) output heads** with
**multi-byte prediction** at the byte level. The landscape:

| Approach | Joint Distribution? | Algebraic Structure? | Byte-level? |
|----------|-------------------|---------------------|-------------|
| Standard NTP | N/A | No | Yes |
| Independent MTP (Meta) | No (marginals) | No | Yes |
| CP Decomposition MTP | Partially | No | No |
| Prob Circuits MTP | Yes | No (generic) | Yes |
| **PHM-MTP (proposed)** | **Yes** | **Yes** | **Yes** |

## Best Implementation: PHM Multi-Byte Head

Replace `output_head = nn.Linear(d_model, 256)` with a PHM head that predicts K bytes jointly:

```python
class PHMMultiByteHead(nn.Module):
    def __init__(self, d_model, K=8, n=4):
        self.A = nn.Parameter(torch.empty(n, K, K))      # inter-byte dependencies
        self.S = nn.Parameter(torch.empty(n, 256, d//n))  # byte probabilities

    def forward(self, h):
        W = sum(kron(A[i], S[i]) for i in range(n))  # (K*256, d_model)
        return (h @ W.t()).view(B, L, K, 256)
```

The A matrices learn **what dependencies exist between predicted byte positions**.
The S matrices learn **what bytes are likely at each position**.
Kronecker structure couples them without full K*256*d_model parameters.

## Why This Matters for Multi-Modal

Byte-level dependencies differ by domain:
- **Text**: UTF-8 multi-byte chars have deterministic inter-byte dependencies
- **Images**: spatial pixel correlations (strong local, weak long-range)
- **Audio**: temporal sample correlations (very strong short-range)

The PHM output structure should learn different A matrices for different dependency patterns
— and we can measure whether it does.

## Paper Title Direction
"Algebraically Structured Multi-Byte Prediction: Kronecker Product Output Heads for
Byte-Level Language Models"
