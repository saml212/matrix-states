#!/usr/bin/env python3
"""C1 PRE-FLIGHT: confirm falcon-mamba-7b loads + forwards through the DEPLOYED
HFLogitsWrapper and produces FINITE logits of the right shape, on the slow
(non-fused) mamba path (mamba_ssm/causal_conv1d absent from tdenv). C1 has never
run in four rounds; this de-risks the ~12 GPU-h gate before launch. NO gate,
NO bridge, NO training touched -- a single tiny forward."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t2a_reference_driver_v2_rd as driver
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
repo = "tiiuae/falcon-mamba-7b"
print(f"[preflight] loading {repo} bf16 (slow non-fused path)...", flush=True)
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
hf = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16).to(DEV).eval()
realized = next(hf.parameters()).dtype
vocab = getattr(hf.config, "vocab_size", None) or len(tok)
eos = driver.resolve_witness_eos_id("C1_falconmamba", tok)
print(f"[preflight] loaded: dtype={realized} vocab_size={vocab} eos_id={eos} "
      f"n_params={sum(p.numel() for p in hf.parameters()):,}", flush=True)
model = driver.HFLogitsWrapper(hf).to(DEV).eval()
# a small induction-shaped batch: repeat a token to see if it copies (sanity, not a gate)
x = torch.randint(100, 5000, (2, 64), device=DEV)
x[:, 40] = x[:, 10]                       # a repeats at 10 and 40
with torch.no_grad():
    lg = model(x)                          # through the deployed wrapper (fp32 upcast + finite check)
print(f"[preflight] logits shape={tuple(lg.shape)} dtype={lg.dtype} "
      f"finite={bool(torch.isfinite(lg).all())}", flush=True)
pred_at_40 = int(lg[0, 40].argmax()); tok_after_first_a = int(x[0, 11])
print(f"[preflight] argmax@pos40={pred_at_40}  token_after_first_a(pos11)={tok_after_first_a}  "
      f"induction_hit={pred_at_40 == tok_after_first_a}", flush=True)
assert lg.shape == (2, 64, vocab) and bool(torch.isfinite(lg).all()) and realized == torch.bfloat16
print("[preflight] C1 PATH OK: falcon-mamba loads, forwards, finite logits, correct shape.", flush=True)
