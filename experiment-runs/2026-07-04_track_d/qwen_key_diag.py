import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:0"
MODEL_ID = "Qwen/Qwen2.5-1.5B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
tok = AutoTokenizer.from_pretrained(MODEL_ID)

text = ("The mitochondria is the powerhouse of the cell. Photosynthesis converts light energy into "
        "chemical energy. Quantum mechanics describes the behavior of subatomic particles. The "
        "French Revolution began in 1789 and reshaped European politics. Machine learning models "
        "learn patterns from data through optimization. " * 3)
ids = tok(text, return_tensors="pt").input_ids[:, :128].to(device)
print("ids shape:", ids.shape)

captured = {}
def mk_hook(i):
    def hook(m, inp, out):
        captured[i] = out.detach()
    return hook
handles = [model.model.layers[i].self_attn.k_proj.register_forward_hook(mk_hook(i)) for i in [0, 13, 27]]
with torch.no_grad():
    out = model(ids, use_cache=False)
for h in handles:
    h.remove()

num_kv_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // model.config.num_attention_heads

for layer_idx in [0, 13, 27]:
    k = captured[layer_idx][0].float()  # (T, D)
    T, D = k.shape
    k_h = k.view(T, num_kv_heads, head_dim)
    for h in range(num_kv_heads):
        v = k_h[:, h, :]  # (T, head_dim)
        # per-channel stats: is one channel dominating the norm? (massive-activation check)
        chan_meanabs = v.abs().mean(dim=0)  # (head_dim,)
        top_chan = chan_meanabs.argmax().item()
        top_val = chan_meanabs[top_chan].item()
        median_val = chan_meanabs.median().item()
        vn = F.normalize(v, dim=-1)
        cos = vn @ vn.T
        offdiag = cos - torch.eye(T, device=cos.device)
        mean_offdiag_cos = (cos.sum() - T) / (T * (T - 1))
        print(f"layer {layer_idx} head {h}: top_channel={top_chan} top_meanabs={top_val:.3f} "
              f"median_meanabs={median_val:.3f} ratio={top_val/max(median_val,1e-8):.1f}x  "
              f"mean_offdiag_cos={mean_offdiag_cos.item():.4f}  "
              f"gram_dev={torch.linalg.norm(cos - torch.eye(T, device=cos.device)).item():.3f}")

# Also: what does gram deviation look like on a RANDOM (untrained-style) unit-normalized set of same shape,
# for reference (n=128, d=head_dim)
torch.manual_seed(0)
rand_v = torch.randn(128, head_dim)
rand_vn = F.normalize(rand_v, dim=-1)
rand_cos = rand_vn @ rand_vn.T
print(f"\nreference: n=128 GENUINELY RANDOM unit vectors in d={head_dim}: "
      f"gram_dev={torch.linalg.norm(rand_cos - torch.eye(128)).item():.3f} "
      f"(expected order sqrt(n(n-1))/sqrt(d) = {(128*127)**0.5 / (head_dim**0.5):.3f})")
