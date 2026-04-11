# Cross-Domain Transfer Measurement

Research from agent on 2026-03-24.

## Protocol (from bGPT, HighMMT, Perceiver IO)

Train N+1 models:
1. M_multi: interleaved text+images (our Run 5)
2. M_text: text-only control, same architecture, matched compute
3. M_image: image-only control, same architecture, matched compute

**Transfer coefficient:** `Delta_D = BPB_single(D) - BPB_multi(D)`
- Delta > 0 = positive transfer (multi-modal helps)
- Delta < 0 = interference

**Meaningful thresholds:** >0.02-0.05 BPB (~1-2% relative) is noteworthy.

## Per-Domain Evaluation

Maintain separate val sets per domain. Evaluate both domains independently.
The model sees no domain labels during training but we know domains at eval time.

## Key Findings from Literature

- bGPT: positive transfer between audio↔images, negative between text↔other
- HighMMT: +0.3% to +2.4% from multi-source pretraining
- bGPT attributes text interference to "distinct byte-level organizational patterns"
- **Our PHM layers could be the wild card** — if Kronecker structure captures
  domain-general features, we might see positive transfer where bGPT didn't

## Additional Diagnostics

1. **Domain classification probe**: linear classifier on frozen representations → domain
   - ~50% accuracy = domain-invariant representations
   - ~100% = domain-specific

2. **Gradient cosine similarity**: cos_sim between text and image gradients
   - Positive = domains reinforce each other
   - Negative = domains conflict (PCGrad)

3. **CKA**: compare representation geometry across domains per layer
