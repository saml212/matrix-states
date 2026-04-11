"""
Experiment 1b: Standard Vector Transformer Baseline.

Pre-norm transformer with SwiGLU FFN. Matched parameter count to Matrix V3.
This is what we're trying to beat.

H100 version: FA3 attention, DDP, torch.compile, bfloat16.
"""

import sys
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import TrainConfig, train


# ─── Standard Transformer ─────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLUFFN(nn.Module):
    """Standard SwiGLU feed-forward: the activation used by every major LLM."""

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = int(d_model * 8 / 3)  # Llama convention
            d_ff = ((d_ff + 63) // 64) * 64  # Round to multiple of 64

        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_value = nn.Linear(d_model, d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_out(self.dropout(F.silu(self.w_gate(x)) * self.w_value(x)))


class Attention(nn.Module):
    """Standard multi-head causal attention."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape
        H = self.n_heads
        hd = self.head_dim

        Q = self.q_proj(x).reshape(B, L, H, hd).transpose(1, 2)
        K = self.k_proj(x).reshape(B, L, H, hd).transpose(1, 2)
        V = self.v_proj(x).reshape(B, L, H, hd).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hd)
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_dropout(self.o_proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with SwiGLU FFN."""

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VectorTransformer(nn.Module):
    """Standard vector transformer for language modeling."""

    def __init__(self, d_model=256, n_layers=6, n_heads=4, d_ff=None,
                 max_len=512, vocab_size=50257, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)

        # SwiGLU output head (matching matrix model's two-step output)
        self.head_gate = nn.Linear(d_model, d_model, bias=False)
        self.head_value = nn.Linear(d_model, d_model, bias=False)
        self.head_out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, L = x.shape
        h = self.embed(x) + self.pos(torch.arange(L, device=x.device).unsqueeze(0))

        for layer in self.layers:
            h = layer(h)

        h = self.final_norm(h)
        logits = self.head_out(F.silu(self.head_gate(h)) * self.head_value(h))
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Configs ───────────────────────────────────────────────────

def get_h100_config():
    return TrainConfig(
        data_dir="/data/wikitext103_tokenized",
        seq_len=512,
        batch_size=64,
        max_steps=50000,
        lr=3e-4,
        warmup_steps=1000,
        dropout=0.1,
        eval_interval=1000,
        log_interval=100,
        save_dir="/results",
        experiment_name="exp1b_vector_baseline",
        device="cuda",
        compile_model=True,
        use_bfloat16=True,
    )


def get_mini_config():
    return TrainConfig(
        data_dir="/Volumes/1TB_SSD/learned-representations/data/wikitext103_tokenized",
        seq_len=128,
        batch_size=8,
        max_steps=200,
        lr=1e-3,
        warmup_steps=20,
        dropout=0.1,
        eval_interval=50,
        log_interval=25,
        save_dir="/Volumes/1TB_SSD/learned-representations/results/h100_test",
        experiment_name="exp1b_vector_baseline_mini",
        device="mps",
        compile_model=False,
        use_bfloat16=False,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true")
    args = parser.parse_args()

    if args.mini:
        config = get_mini_config()
        # Small model — match param count roughly to exp1a mini
        model = VectorTransformer(d_model=128, n_layers=4, n_heads=4,
                                  max_len=256, vocab_size=50257, dropout=0.1)
    else:
        config = get_h100_config()
        model = VectorTransformer(d_model=512, n_layers=6, n_heads=8,
                                  max_len=1024, vocab_size=50257, dropout=0.1)

    print(f"Model: {model.count_parameters():,} params")
    train(model, config)
