#!/usr/bin/env python3
"""
Experiment 3: Scaled Iterative Matrix Thinker
==============================================
Changes from Exp 2:
1. Wider thinking layers — 4x expansion factor so thinking has real capacity
2. Bigger mat_dim (64 vs 32)
3. More data (~500M tokens: reasoning + FineWeb-Edu)
4. GPU SVD (no CPU transfer), full val set eval, batch_size=128 for eval
5. Multi-GPU ready (DDP wrapper for future 8×H100)
6. 16 DataLoader workers (208 CPU threads available)

Key question: with more capacity in the thinking layers and harder data,
does the thinking benefit (9.8% at Exp 2 step 1500) grow further?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import math, time, json, os, sys, shutil
from pathlib import Path
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    # Model
    "mat_dim": 64,
    "n_thinking_layers": 12,
    "n_heads": 32,             # head_dim=2, SDPA head_dim=2*64=128 (safe for flash attention)
    "expansion_factor": 4,     # NEW: thinking layer inner dim = 4 * mat_dim
    "n_iterations": 8,
    "max_len": 2048,
    "dropout": 0.1,
    # Training
    "batch_size": 32,
    "seq_len": 512,
    "max_steps": 20_000,
    "lr": 3e-4,
    "warmup_steps": 1000,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    # Eval
    "eval_interval": 1000,
    "eval_batch_size": 128,
    "eval_iterations": [1, 2, 4, 8],
    # Logging
    "log_interval": 50,
    # Paths
    "data_dir": "/root/data/large",
    "results_dir": "/root/results/exp3_scaled",
    # Hardware
    "num_workers": 16,         # 208 CPU threads available
}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class MatrixRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, d))
        self.eps = eps

    def forward(self, M):
        rms = torch.sqrt(M.pow(2).mean(dim=(-2, -1), keepdim=True) + self.eps)
        return M / rms * self.weight


class RowThenColProjection(nn.Module):
    """silu(A @ M) @ B — nonlinearity between left and right multiply."""
    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
        self.B = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))

    def forward(self, M):
        return torch.einsum('...ij,jk->...ik',
                            F.silu(torch.einsum('ij,...jk->...ik', self.A, M)),
                            self.B)


class FrobeniusAttention(nn.Module):
    """Multi-head attention with Frobenius inner product. Uses SDPA (flash backend)."""
    def __init__(self, d, n_heads=32, dropout=0.1):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        assert d % n_heads == 0

        self.norm = MatrixRMSNorm(d)
        self.q_proj = RowThenColProjection(d)
        self.k_proj = RowThenColProjection(d)
        self.v_proj = RowThenColProjection(d)
        self.o_proj = RowThenColProjection(d)
        self.dropout_p = dropout

    def forward(self, M):
        B, L, d, _ = M.shape
        H, hd = self.n_heads, self.head_dim

        M_n = self.norm(M)
        Q = self.q_proj(M_n)
        K = self.k_proj(M_n)
        V = self.v_proj(M_n)

        # (B, L, d, d) -> (B, L, H, hd, d) -> (B, H, L, hd*d)
        Q = Q.reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4).reshape(B, H, L, hd * d)
        K = K.reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4).reshape(B, H, L, hd * d)
        V = V.reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4).reshape(B, H, L, hd * d)

        drop_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=drop_p)

        out = out.reshape(B, H, L, hd, d).permute(0, 2, 1, 3, 4).reshape(B, L, d, d)
        return M + self.o_proj(out)


class ExpandedMultiplicativeLayer(nn.Module):
    """
    Per-position: (I + Δ) · M · (I + Γ) + v · kᵀ

    NEW: Δ and Γ are computed through an expanded bottleneck:
    M → expand to K parallel RowThenCol paths → sum → scale
    This gives the thinking layers real parameter capacity.
    """
    def __init__(self, d, expansion_factor=4, dropout=0.1):
        super().__init__()
        K = expansion_factor
        self.norm = MatrixRMSNorm(d)

        # Expanded SwiGLU for delta: K parallel gate*value paths
        self.delta_gates = nn.ModuleList([RowThenColProjection(d) for _ in range(K)])
        self.delta_values = nn.ModuleList([RowThenColProjection(d) for _ in range(K)])
        # Expanded SwiGLU for gamma
        self.gamma_gates = nn.ModuleList([RowThenColProjection(d) for _ in range(K)])
        self.gamma_values = nn.ModuleList([RowThenColProjection(d) for _ in range(K)])

        # Additive write
        self.key_col = nn.Parameter(torch.randn(d, 1) * 0.02)
        self.val_col = nn.Parameter(torch.randn(d, 1) * 0.02)

        # Gates
        self.gate_mult_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_mult_bias = nn.Parameter(torch.tensor(-2.0))
        self.gate_write_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_write_bias = nn.Parameter(torch.tensor(-2.0))

        self.scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('I', torch.eye(d))

    def forward(self, M):
        M_n = self.norm(M)
        scale = self.scale.clamp(0.01, 0.5)

        # Sum of K SwiGLU paths for delta and gamma
        delta = sum(F.silu(g(M_n)) * v(M_n) for g, v in zip(self.delta_gates, self.delta_values)) * scale
        gamma = sum(F.silu(g(M_n)) * v(M_n) for g, v in zip(self.gamma_gates, self.gamma_values)) * scale

        M_mult = torch.matmul(torch.matmul(self.I + delta, M_n), self.I + gamma)

        k = torch.matmul(M_n, self.key_col).squeeze(-1)
        v = torch.matmul(M_n, self.val_col).squeeze(-1)
        M_write = torch.einsum('...i,...j->...ij', v, k)

        g_m = torch.sigmoid((self.gate_mult_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_mult_bias)
        g_w = torch.sigmoid((self.gate_write_W * M_n).sum(dim=(-2, -1), keepdim=True) + self.gate_write_bias)

        return M + self.dropout(g_m * (M_mult - M_n) + g_w * M_write)


class ThinkingBlock(nn.Module):
    def __init__(self, d, n_heads=32, expansion_factor=4, dropout=0.1):
        super().__init__()
        self.attn = FrobeniusAttention(d, n_heads, dropout)
        self.think = ExpandedMultiplicativeLayer(d, expansion_factor, dropout)

    def forward(self, M):
        return self.think(self.attn(M))


class IterativeMatrixThinker(nn.Module):
    """
    Scaled iterative matrix thinker.

    Changes from Exp 2:
    - Expanded thinking layers (4x more capacity per layer)
    - Larger mat_dim (64 vs 32)
    - measure_rank runs on GPU
    """
    def __init__(self, mat_dim=64, n_thinking_layers=12, n_heads=32,
                 expansion_factor=4, max_len=2048, vocab_size=50257, dropout=0.1):
        super().__init__()
        self.mat_dim = mat_dim
        d = mat_dim

        self.embed_u = nn.Embedding(vocab_size, d)
        self.embed_v = nn.Embedding(vocab_size, d)
        self.pos_u = nn.Embedding(max_len, d)
        self.pos_v = nn.Embedding(max_len, d)

        self.thinking_layers = nn.ModuleList([
            ThinkingBlock(d, n_heads, expansion_factor, dropout)
            for _ in range(n_thinking_layers)
        ])

        self.final_norm = MatrixRMSNorm(d)
        self.collapse_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.output_head = nn.Linear(d, vocab_size, bias=False)

    def embed(self, token_ids):
        B, L = token_ids.shape
        u = self.embed_u(token_ids)
        v = self.embed_v(token_ids)
        M = torch.einsum('...i,...j->...ij', u, v)
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        pu = self.pos_u(pos)
        pv = self.pos_v(pos)
        return M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1

    def _one_iteration(self, M):
        for layer in self.thinking_layers:
            if self.training:
                # Checkpoint per LAYER (not per iteration) — only 1 layer's
                # activations in memory at a time during backward recomputation
                M = torch.utils.checkpoint.checkpoint(layer, M, use_reentrant=False)
            else:
                M = layer(M)
        return M

    def measure_rank(self, M):
        """Effective rank via SVD — runs on GPU, no CPU transfer."""
        with torch.no_grad():
            M_flat = M.detach().float().reshape(-1, M.shape[-2], M.shape[-1])
            S = torch.linalg.svdvals(M_flat).clamp(min=1e-10)
            S_norm = S / S.sum(dim=-1, keepdim=True)
            entropy = -(S_norm * S_norm.log()).sum(dim=-1)
            return entropy.exp().reshape(M.shape[:-2])

    def forward(self, token_ids, n_iterations=1, measure_ranks=False):
        M = self.embed(token_ids)

        rank_per_iter = []
        for t in range(n_iterations):
            M = self._one_iteration(M)
            if measure_ranks and (not self.training or t == 0 or t == n_iterations - 1):
                rank_per_iter.append(self.measure_rank(M).mean().item())

        M_n = self.final_norm(M)
        vec = (self.collapse_W * M_n).sum(dim=-1)
        logits = self.output_head(vec)

        return logits, {'rank_per_iteration': rank_per_iter}

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, path, seq_len):
        self.data = torch.load(path, weights_only=True)
        self.seq_len = seq_len
        self.n = len(self.data) // seq_len - 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        c = self.data[s:s + self.seq_len + 1]
        return c[:-1], c[1:]


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.file = open(log_path, 'w')

    def log(self, msg):
        ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self.file.write(line + '\n')
        self.file.flush()

    def close(self):
        self.file.close()


# ═══════════════════════════════════════════════════════════════════════════════
# EVAL
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, val_ds, vocab_size, device, n_iterations, eval_batch_size=128):
    """Full eval over entire val set. GPU SVD, large batches."""
    model.eval()
    eval_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False,
        drop_last=True, num_workers=4, pin_memory=True
    )
    total_loss = 0
    total_tokens = 0
    all_ranks = []

    with torch.no_grad():
        for vx, vy in eval_loader:
            vx, vy = vx.to(device), vy.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, info = model(vx, n_iterations=n_iterations, measure_ranks=True)
            n_tok = vy.numel()
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), vy.reshape(-1))
            total_loss += loss.item() * n_tok
            total_tokens += n_tok
            all_ranks.append(info['rank_per_iteration'])

    avg_loss = total_loss / max(total_tokens, 1)
    avg_ppl = math.exp(min(avg_loss, 20))

    n_ranks = len(all_ranks[0])
    avg_rank_profile = [sum(r[i] for r in all_ranks) / len(all_ranks) for i in range(n_ranks)]

    model.train()
    return avg_loss, avg_ppl, avg_rank_profile


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    cfg = CONFIG
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save this script
    shutil.copy2(Path(__file__).resolve(), results_dir / "script.py")

    logger = Logger(results_dir / "train.log")
    logger.log(f"Experiment 3: Scaled Iterative Matrix Thinker")
    logger.log(f"Config: {json.dumps(cfg, indent=2)}")

    # Data
    meta = json.load(open(os.path.join(cfg["data_dir"], "meta.json")))
    vocab_size = meta["vocab_size"]
    logger.log(f"Vocab: {vocab_size}")
    logger.log(f"Train tokens: {meta.get('train_tokens', 'unknown'):,}")
    logger.log(f"Val tokens: {meta.get('val_tokens', 'unknown'):,}")
    logger.log(f"Sources: {meta.get('sources', 'unknown')}")

    train_ds = TokenDataset(os.path.join(cfg["data_dir"], "train.pt"), cfg["seq_len"])
    val_ds = TokenDataset(os.path.join(cfg["data_dir"], "val.pt"), cfg["seq_len"])

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        drop_last=True, num_workers=cfg["num_workers"], pin_memory=True,
        persistent_workers=True,
    )

    # Model
    device = torch.device("cuda")
    model = IterativeMatrixThinker(
        mat_dim=cfg["mat_dim"],
        n_thinking_layers=cfg["n_thinking_layers"],
        n_heads=cfg["n_heads"],
        expansion_factor=cfg["expansion_factor"],
        max_len=cfg["max_len"],
        vocab_size=vocab_size,
        dropout=cfg["dropout"],
    ).to(device)

    n_params = model.count_params()
    # Break down param distribution
    embed_params = sum(p.numel() for n, p in model.named_parameters() if 'embed' in n or 'pos' in n)
    think_params = sum(p.numel() for n, p in model.named_parameters() if 'thinking' in n)
    output_params = sum(p.numel() for n, p in model.named_parameters() if 'output' in n or 'collapse' in n or 'final' in n)

    logger.log(f"Model params: {n_params:,}")
    logger.log(f"  Embedding: {embed_params:,} ({embed_params/n_params*100:.0f}%)")
    logger.log(f"  Thinking:  {think_params:,} ({think_params/n_params*100:.0f}%)")
    logger.log(f"  Output:    {output_params:,} ({output_params/n_params*100:.0f}%)")
    logger.log(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.log(f"Training: {cfg['max_steps']} steps, T={cfg['n_iterations']}, "
               f"seq={cfg['seq_len']}, batch={cfg['batch_size']}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], betas=(0.9, 0.98)
    )

    def lr_lambda(step):
        if step < cfg["warmup_steps"]:
            return step / cfg["warmup_steps"]
        progress = (step - cfg["warmup_steps"]) / max(cfg["max_steps"] - cfg["warmup_steps"], 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    step = 0
    best_val_loss = float("inf")
    start_time = time.time()
    training_curve = []
    data_iter = iter(train_loader)
    model.train()

    logger.log(f"--- Training starts ---")

    while step < cfg["max_steps"]:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        do_log = (step + 1) % cfg["log_interval"] == 0

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, info = model(x, n_iterations=cfg["n_iterations"],
                                 measure_ranks=do_log)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        step += 1

        if step % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            ppl = math.exp(min(loss.item(), 20))
            ranks = info['rank_per_iteration']
            rank_str = ", ".join(f"{r:.2f}" for r in ranks) if ranks else "n/a"
            tokens_per_sec = step * cfg["batch_size"] * cfg["seq_len"] / elapsed
            logger.log(
                f"Step {step:6d} | Loss {loss.item():.4f} | PPL {ppl:8.1f} | "
                f"GN {grad_norm:.3f} | LR {scheduler.get_last_lr()[0]:.2e} | "
                f"Rank [{rank_str}] | {tokens_per_sec:.0f} tok/s | {elapsed:.0f}s"
            )

        # Eval
        if step % cfg["eval_interval"] == 0:
            logger.log(f"--- Evaluating at step {step} ---")
            eval_results = {}
            for T in cfg["eval_iterations"]:
                vl, vp, vr = evaluate(model, val_ds, vocab_size, device, T,
                                      eval_batch_size=cfg["eval_batch_size"])
                rank_str = ", ".join(f"{r:.2f}" for r in vr)
                marker = ""
                if T == cfg["n_iterations"] and vl < best_val_loss:
                    best_val_loss = vl
                    torch.save(model.state_dict(), results_dir / "best.pt")
                    marker = " *BEST*"
                logger.log(f"  T={T:2d}: Val Loss {vl:.4f} | Val PPL {vp:8.1f} | Rank [{rank_str}]{marker}")
                eval_results[f"T{T}"] = {"val_loss": vl, "val_ppl": vp, "rank_profile": vr}

            training_curve.append({
                "step": step,
                "train_loss": loss.item(),
                "evals": eval_results,
            })

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    elapsed = time.time() - start_time
    logger.log(f"\n{'='*70}")
    logger.log(f"Training complete: {step} steps in {elapsed/60:.1f} minutes")

    model.load_state_dict(torch.load(results_dir / "best.pt", weights_only=True))
    logger.log(f"\nFinal evaluation (best checkpoint):")
    final_evals = {}
    for T in cfg["eval_iterations"]:
        vl, vp, vr = evaluate(model, val_ds, vocab_size, device, T,
                              eval_batch_size=cfg["eval_batch_size"])
        rank_str = ", ".join(f"{r:.2f}" for r in vr)
        logger.log(f"  T={T:2d}: Val Loss {vl:.4f} | Val PPL {vp:8.1f} | Rank [{rank_str}]")
        final_evals[f"T{T}"] = {"val_loss": vl, "val_ppl": vp, "rank_profile": vr}

    ppls = {T: final_evals[f"T{T}"]["val_ppl"] for T in cfg["eval_iterations"]}
    thinking_helps = ppls[cfg["eval_iterations"][-1]] < ppls[cfg["eval_iterations"][0]]

    summary_lines = [
        "", "=" * 70, "  EXPERIMENT 3 SUMMARY", "=" * 70, "",
        f"  Model: Scaled Iterative Matrix Thinker",
        f"  Params: {n_params:,} (Thinking: {think_params:,} = {think_params/n_params*100:.0f}%)",
        f"  mat_dim={cfg['mat_dim']}, {cfg['n_thinking_layers']} layers, "
        f"{cfg['n_heads']} heads, expansion={cfg['expansion_factor']}x, T={cfg['n_iterations']}",
        f"  Data: {meta.get('train_tokens', '?'):,} train tokens",
        f"  Training: {cfg['max_steps']} steps, seq={cfg['seq_len']}, batch={cfg['batch_size']}",
        f"  Time: {elapsed/60:.1f} minutes on {torch.cuda.get_device_name(0)}",
        "",
        "  RESULTS (best checkpoint, full val set):",
        "  " + "-" * 50,
    ]
    for T in cfg["eval_iterations"]:
        e = final_evals[f"T{T}"]
        bar = "#" * min(int(e['val_ppl'] / 5), 40)
        summary_lines.append(f"  T={T:2d}: PPL {e['val_ppl']:8.1f}  {bar}")

    t1, tn = cfg["eval_iterations"][0], cfg["eval_iterations"][-1]
    delta_ppl = ppls[t1] - ppls[tn]
    delta_pct = delta_ppl / ppls[t1] * 100
    summary_lines += [
        "",
        f"  MORE THINKING HELPS: {'YES' if thinking_helps else 'NO'}",
        f"    T={t1} PPL: {ppls[t1]:.1f}",
        f"    T={tn} PPL: {ppls[tn]:.1f}",
        f"    Delta: {delta_ppl:.1f} PPL ({delta_pct:.1f}%)",
        "",
    ]

    r_last = final_evals[f"T{tn}"]["rank_profile"]
    if len(r_last) >= 2:
        summary_lines += [
            f"  RANK PROFILE (T={tn}):",
            f"    First iter: {r_last[0]:.3f}",
            f"    Last iter:  {r_last[-1]:.3f}",
            f"    Solidifies: {'YES' if r_last[-1] < r_last[0] else 'NO'}",
            "",
        ]

    summary_lines += ["=" * 70, ""]
    summary = "\n".join(summary_lines)
    logger.log(summary)

    with open(results_dir / "SUMMARY.txt", 'w') as f:
        f.write(summary)

    results = {
        "experiment": "exp3_scaled_iterative_matrix_thinker",
        "date": datetime.now(timezone.utc).isoformat(),
        "params": n_params,
        "param_breakdown": {"embedding": embed_params, "thinking": think_params, "output": output_params},
        "config": cfg,
        "data": meta,
        "gpu": torch.cuda.get_device_name(0),
        "training_time_minutes": elapsed / 60,
        "best_val_loss": best_val_loss,
        "final_evals": final_evals,
        "thinking_helps": thinking_helps,
        "thinking_benefit_pct": delta_pct,
        "training_curve": training_curve,
    }
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=float)

    logger.log(f"Saved to {results_dir}")
    logger.close()


if __name__ == "__main__":
    run()
