---
target: x.com
canonical: https://pebbleml.com/findings/parameter-efficiency.html
status: ready_for_review
constraints:
  - 5 tweets. Hook stands alone.
  - Lead with the gap AND the caveat together — without the FLOP caveat in tweet 1 or 2 the post reads dishonestly.
---

# Tweet 1 (hook)

at d=16, a matrix-native RowThenCol projection silu(A·M)·B uses 512 parameters.

the flattened-vector equivalent (d²×d² dense Linear) uses 65,536. ratio 128×.

honest version of the gap: only 8× per-step FLOPs, not 128×. why ↓

🧵

# Tweet 2 (the FLOP math)

a flattened Linear spends each parameter once per token (2 FMAs / scalar weight).

a matrix-native projection uses each parameter d times — A·M is an inner product against every column of M. at d=16, 16× reuse.

cancels one factor of 16. the other factor of 16 survives as a real FLOP cut.

# Tweet 3 (what it buys you at small scale)

at d=32 with a 50K BPE vocab, an 8-layer Matrix Thinker:

· embed tables: ~3.2M params (63%)
· 8 thinking layers: ~0.2M (4%)
· output head: ~1.6M (31%)

iterative refinement reuses the 8 layers T=8 times per fwd pass. 64 effective layer-applications at 8 layers of params.

# Tweet 4 (where it matters most)

at byte-level d=16 the embedding shrinks 100× (256 bytes vs 50K BPE) and thinking layers claim a much larger share. Run 19 = 218K total params, 33% in thinking layers, 12 layers deep.

structural efficiency where params are scarce and compute isn't.

# Tweet 5 (caveats + link)

what this is NOT: a claim matrix ops are better at matched compute. at matched FLOPs, matrix loses to vector by a wide margin (Run 14: Matrix Thinker BPB 1.67 vs LoopFormer 0.87 at 653K TFLOPs).

structurally efficient ≠ universally better.

full note: https://pebbleml.com/findings/parameter-efficiency.html
