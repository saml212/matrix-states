---
target: research-oriented Discord servers (#research, #efficient-ml, #papers)
canonical: https://pebbleml.com/findings/parameter-efficiency.html
status: ready_for_review
---

# Long form

structural observation from a matrix-valued transformer architecture I've been running:

at d=16, a RowThenCol bilinear projection silu(A·M)·B uses 2·d² = 512 params. the flattened d² × d² Linear equivalent uses 65,536 params. param ratio 128×.

honest FLOP picture: a flattened Linear spends each parameter once per token (2 FMAs / scalar weight). a matrix-native projection uses each parameter d times during the sandwich (A·M is an inner product against every column of M). at d=16, 16× reuse cancels one factor of 16. real FLOP gap is 8×, not 128×. realized speedup on H100 not yet measured; memory bandwidth and kernel launch overhead expected to eat much of the 8×.

what it lets you do: in an 8-layer Matrix Thinker at d=32 / 50K BPE, the 8 thinking layers are 4% of the model — iterative refinement (T=8) reuses them for 64 effective layer-applications at 8 layers of params. at byte-level d=16, embedding shrinks 100× and thinking layers claim 33% (Run 19: 218K total params, 12 layers).

what it isn't: a claim matrix ops are better at matched compute. at matched FLOPs matrix loses to vector by a wide margin (Matrix Thinker 1.67 BPB vs LoopFormer 0.87 at 653K TFLOPs). param-efficient ≠ universally better.

closest established cousin: PHM (Zhang et al. 2021). same family of "express a linear layer as a sum of Kroneckers" param savings. interested in pointers to other relevant prior work.

paper: https://pebbleml.com/findings/parameter-efficiency.html

# Short form

128× fewer params per layer for matrix-native bilinear projections vs flattened Linear at d=16. honest FLOP gap is 8×, not 128× (parameter reuse during the sandwich op).

https://pebbleml.com/findings/parameter-efficiency.html
