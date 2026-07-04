import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda:0"
text = ("The mitochondria is the powerhouse of the cell. Photosynthesis converts light energy into "
        "chemical energy. Quantum mechanics describes the behavior of subatomic particles. The "
        "French Revolution began in 1789 and reshaped European politics. Machine learning models "
        "learn patterns from data through optimization. " * 3)


def report(name, v):
    # v: (T, d)
    T, d = v.shape
    chan_meanabs = v.abs().mean(dim=0)
    top_chan = chan_meanabs.argmax().item()
    top_val = chan_meanabs[top_chan].item()
    median_val = chan_meanabs.median().item()
    vn = F.normalize(v, dim=-1)
    cos = vn @ vn.T
    mean_offdiag_cos = (cos.sum() - T) / (T * (T - 1))
    gd = torch.linalg.norm(cos - torch.eye(T, device=cos.device)).item()
    print(f"{name}: top_chan={top_chan} top_meanabs={top_val:.3f} median={median_val:.3f} "
          f"ratio={top_val/max(median_val,1e-8):.1f}x mean_offdiag_cos={mean_offdiag_cos.item():.4f} gram_dev={gd:.3f}")


print("=== RWKV-7 1.5B ===")
mid = "RWKV/RWKV7-Goose-World3-1.5B-HF"
model = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()
tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
ids = tok(text, return_tensors="pt").input_ids[:, :128].to(device)
captured = {}
def mk_hook(i):
    def hook(m, inp, out):
        captured[i] = out.detach()
    return hook
layers_to_check = [0, model.config.num_hidden_layers // 2, model.config.num_hidden_layers - 1]
handles = [model.model.layers[i].attn.k_proj.register_forward_hook(mk_hook(i)) for i in layers_to_check]
with torch.no_grad():
    model(ids, use_cache=False)
for h in handles:
    h.remove()
for i in layers_to_check:
    attn = model.model.layers[i].attn
    k_raw = captured[i][0].float()  # (T, hidden)
    kk = F.normalize((k_raw * attn.k_k.float()).view(-1, attn.num_heads, attn.head_dim), dim=-1)
    report(f"layer {i} head0 kk (post model L2-norm)", kk[:, 0, :])
    # also raw k_proj pre-model-normalization, per head, for comparison
    k_raw_h = k_raw.view(-1, attn.num_heads, attn.head_dim)
    report(f"layer {i} head0 raw k_proj (pre model norm)", k_raw_h[:, 0, :])
del model
torch.cuda.empty_cache()

print("\n=== Falcon-Mamba-7B ===")
mid = "tiiuae/falcon-mamba-7b"
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(device).eval()
tok = AutoTokenizer.from_pretrained(mid)
ids = tok(text, return_tensors="pt").input_ids[:, :128].to(device)
captured = {}
layers_to_check = [0, model.config.num_hidden_layers // 2, model.config.num_hidden_layers - 1]
handles = [model.backbone.layers[i].mixer.x_proj.register_forward_hook(mk_hook(i)) for i in layers_to_check]
with torch.no_grad():
    model(ids, use_cache=False)
for h in handles:
    h.remove()
for i in layers_to_check:
    mixer = model.backbone.layers[i].mixer
    ssm_params = captured[i][0].float()
    tsr, ss = mixer.time_step_rank, mixer.ssm_state_size
    _, B_raw, _C = torch.split(ssm_params, [tsr, ss, ss], dim=-1)
    eps = getattr(mixer, "rms_eps", 1e-6)
    var = B_raw.pow(2).mean(-1, keepdim=True)
    B_t = B_raw * torch.rsqrt(var + eps)
    report(f"layer {i} B_t (post model rms_forward)", B_t)
    report(f"layer {i} raw B (pre rms_forward)", B_raw)
