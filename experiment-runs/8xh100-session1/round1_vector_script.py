#!/usr/bin/env python3
"""
Round 1: Matrix Thinker vs Vector Thinker — Head to Head
=========================================================
Both models: ~5M params, 8 layers, T=8 iterations, same data, same training.
The ONLY difference: matrix (32×32) vs vector (48-dim flat).
DDP across 4 GPUs each. Run simultaneously on 8-GPU pod.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run_round1.py --model matrix
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 run_round1.py --model vector
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math, time, json, os, sys, shutil, argparse
from pathlib import Path
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

BASE_CONFIG = {
    "n_iterations": 8,
    "max_len": 2048,
    "dropout": 0.1,
    "batch_size_per_gpu": 32,
    "seq_len": 512,
    "max_steps": 3000,
    "lr": 3e-4,
    "warmup_steps": 300,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "eval_interval": 500,
    "eval_batch_size": 128,
    "eval_iterations": [1, 2, 4, 8],
    "log_interval": 50,
    "data_dir": "/toy_story_slam/data_quick",
    "num_workers": 8,
}


# ═══════════════════════════════════════════════════════════════════════════════
# MATRIX MODEL (~5.15M params)
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
    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
        self.B = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
    def forward(self, M):
        return torch.einsum('...ij,jk->...ik',
                            F.silu(torch.einsum('ij,...jk->...ik', self.A, M)), self.B)


class MatrixFrobeniusAttention(nn.Module):
    def __init__(self, d, n_heads=8, dropout=0.1):
        super().__init__()
        self.d, self.n_heads, self.head_dim = d, n_heads, d // n_heads
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
        Q = self.q_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4).reshape(B, H, L, hd * d)
        K = self.k_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4).reshape(B, H, L, hd * d)
        V = self.v_proj(M_n).reshape(B, L, H, hd, d).permute(0, 2, 1, 3, 4).reshape(B, H, L, hd * d)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True,
                                              dropout_p=self.dropout_p if self.training else 0.0)
        out = out.reshape(B, H, L, hd, d).permute(0, 2, 1, 3, 4).reshape(B, L, d, d)
        return M + self.o_proj(out)


class MatrixMultiplicativeLayer(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.norm = MatrixRMSNorm(d)
        self.delta_gate = RowThenColProjection(d)
        self.delta_value = RowThenColProjection(d)
        self.delta_up = RowThenColProjection(d)
        self.gamma_gate = RowThenColProjection(d)
        self.gamma_value = RowThenColProjection(d)
        self.gamma_up = RowThenColProjection(d)
        self.key_col = nn.Parameter(torch.randn(d, 1) * 0.02)
        self.val_col = nn.Parameter(torch.randn(d, 1) * 0.02)
        self.gate_mult_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_mult_bias = nn.Parameter(torch.tensor(-2.0))
        self.gate_write_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.gate_write_bias = nn.Parameter(torch.tensor(-2.0))
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('I', torch.eye(d))

    def forward(self, M):
        M_n = self.norm(M)
        s = self.scale.clamp(0.01, 0.5)
        delta = self.delta_up(F.silu(self.delta_gate(M_n)) * self.delta_value(M_n)) * s
        gamma = self.gamma_up(F.silu(self.gamma_gate(M_n)) * self.gamma_value(M_n)) * s
        M_mult = torch.matmul(torch.matmul(self.I + delta, M_n), self.I + gamma)
        k = torch.matmul(M_n, self.key_col).squeeze(-1)
        v = torch.matmul(M_n, self.val_col).squeeze(-1)
        M_write = torch.einsum('...i,...j->...ij', v, k)
        g_m = torch.sigmoid((self.gate_mult_W * M_n).sum(dim=(-2,-1), keepdim=True) + self.gate_mult_bias)
        g_w = torch.sigmoid((self.gate_write_W * M_n).sum(dim=(-2,-1), keepdim=True) + self.gate_write_bias)
        return M + self.dropout(g_m * (M_mult - M_n) + g_w * M_write)


class MatrixThinkingBlock(nn.Module):
    def __init__(self, d, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = MatrixFrobeniusAttention(d, n_heads, dropout)
        self.think = MatrixMultiplicativeLayer(d, dropout)
    def forward(self, M):
        return self.think(self.attn(M))


class MatrixThinker(nn.Module):
    def __init__(self, mat_dim=32, n_layers=8, n_heads=8, max_len=2048,
                 vocab_size=50257, dropout=0.1):
        super().__init__()
        self.mat_dim = mat_dim
        d = mat_dim
        self.embed_u = nn.Embedding(vocab_size, d)
        self.embed_v = nn.Embedding(vocab_size, d)
        self.pos_u = nn.Embedding(max_len, d)
        self.pos_v = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([MatrixThinkingBlock(d, n_heads, dropout) for _ in range(n_layers)])
        self.final_norm = MatrixRMSNorm(d)
        self.collapse_W = nn.Parameter(torch.randn(d, d) * 0.02)
        self.output_head = nn.Linear(d, vocab_size, bias=False)

    def _one_iteration(self, M):
        for layer in self.layers:
            if self.training:
                M = torch.utils.checkpoint.checkpoint(layer, M, use_reentrant=False)
            else:
                M = layer(M)
        return M

    def forward(self, token_ids, n_iterations=1, measure_ranks=False):
        B, L = token_ids.shape
        d = self.mat_dim
        u, v = self.embed_u(token_ids), self.embed_v(token_ids)
        M = torch.einsum('...i,...j->...ij', u, v)
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        pu, pv = self.pos_u(pos), self.pos_v(pos)
        M = M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1

        ranks = []
        for t in range(n_iterations):
            M = self._one_iteration(M)
            if measure_ranks:
                with torch.no_grad():
                    Mf = M.detach().float().reshape(-1, d, d)
                    if Mf.shape[0] > 512:
                        Mf = Mf[torch.randperm(Mf.shape[0], device=Mf.device)[:512]]
                    S = torch.linalg.svdvals(Mf).clamp(min=1e-10)
                    Sn = S / S.sum(-1, keepdim=True)
                    ranks.append((-(Sn * Sn.log()).sum(-1)).exp().mean().item())

        M_n = self.final_norm(M)
        vec = (self.collapse_W * M_n).sum(dim=-1)
        logits = self.output_head(vec)
        return logits, {'ranks': ranks, 'model_type': 'matrix'}

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# VECTOR MODEL (~5M params, matched)
# ═══════════════════════════════════════════════════════════════════════════════

class VectorThinkingBlock(nn.Module):
    """Standard pre-norm transformer block."""
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, h):
        h_n = self.norm1(h)
        L = h.shape[1]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(L, device=h.device)
        attn_out, _ = self.attn(h_n, h_n, h_n, attn_mask=mask)
        h = h + attn_out
        h = h + self.ffn(self.norm2(h))
        return h


class VectorThinker(nn.Module):
    """Standard transformer with iterative refinement (shared layers applied T times)."""
    def __init__(self, d_model=48, n_layers=8, n_heads=4, max_len=2048,
                 vocab_size=50257, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([VectorThinkingBlock(d_model, n_heads, dropout)
                                      for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)

    def _one_iteration(self, h):
        for layer in self.layers:
            if self.training:
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        return h

    def forward(self, token_ids, n_iterations=1, measure_ranks=False):
        B, L = token_ids.shape
        h = self.embed(token_ids) + self.pos(torch.arange(L, device=token_ids.device)).unsqueeze(0)
        for t in range(n_iterations):
            h = self._one_iteration(h)
        h = self.norm(h)
        logits = self.output(h)
        return logits, {'ranks': [], 'model_type': 'vector'}

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
    def __len__(self): return self.n
    def __getitem__(self, i):
        s = i * self.seq_len
        c = self.data[s:s + self.seq_len + 1]
        return c[:-1], c[1:]


# ═══════════════════════════════════════════════════════════════════════════════
# EVAL
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, val_ds, vocab_size, device, n_iterations, eval_batch_size=128):
    model.eval()
    loader = torch.utils.data.DataLoader(val_ds, batch_size=eval_batch_size,
                                          shuffle=False, drop_last=True, num_workers=4)
    total_loss, total_tokens = 0, 0
    all_ranks = []
    with torch.no_grad():
        for vx, vy in loader:
            vx, vy = vx.to(device), vy.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, info = model(vx, n_iterations=n_iterations, measure_ranks=True)
            n_tok = vy.numel()
            total_loss += F.cross_entropy(logits.reshape(-1, vocab_size), vy.reshape(-1)).item() * n_tok
            total_tokens += n_tok
            if info['ranks']:
                all_ranks.append(info['ranks'])
    avg_loss = total_loss / max(total_tokens, 1)
    avg_ppl = math.exp(min(avg_loss, 20))
    avg_ranks = []
    if all_ranks:
        n_r = len(all_ranks[0])
        avg_ranks = [sum(r[i] for r in all_ranks) / len(all_ranks) for i in range(n_r)]
    model.train()
    return avg_loss, avg_ppl, avg_ranks


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["matrix", "vector"], required=True)
    args = parser.parse_args()
    cfg = BASE_CONFIG.copy()
    cfg["model_type"] = args.model
    cfg["results_dir"] = f"/toy_story_slam/results/round1_{args.model}"

    # DDP setup
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    results_dir = Path(cfg["results_dir"])
    if is_main:
        results_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve(), results_dir / "script.py")

    # Logger (rank 0 only)
    class Logger:
        def __init__(self, path):
            self.f = open(path, 'w') if is_main else None
        def log(self, msg):
            if not is_main: return
            ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
            line = f"[{ts}] {msg}"
            print(line, flush=True)
            if self.f: self.f.write(line + '\n'); self.f.flush()
        def close(self):
            if self.f: self.f.close()

    dist.barrier()
    logger = Logger(results_dir / "train.log")
    logger.log(f"Round 1: {args.model.upper()} thinker | GPUs: {world_size}")
    logger.log(f"Config: {json.dumps(cfg, indent=2)}")

    # Data
    meta = json.load(open(os.path.join(cfg["data_dir"], "meta.json")))
    vocab_size = meta["vocab_size"]
    logger.log(f"Data: {meta.get('train_tokens', '?'):,} train, {meta.get('val_tokens', '?'):,} val tokens")

    train_ds = TokenDataset(os.path.join(cfg["data_dir"], "train.pt"), cfg["seq_len"])
    val_ds = TokenDataset(os.path.join(cfg["data_dir"], "val.pt"), cfg["seq_len"])
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size_per_gpu"], sampler=sampler,
        drop_last=True, num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=True)

    # Model
    if args.model == "matrix":
        raw_model = MatrixThinker(mat_dim=32, n_layers=8, n_heads=8,
                                   max_len=cfg["max_len"], vocab_size=vocab_size,
                                   dropout=cfg["dropout"]).to(device)
    else:
        raw_model = VectorThinker(d_model=48, n_layers=8, n_heads=4,
                                   max_len=cfg["max_len"], vocab_size=vocab_size,
                                   dropout=cfg["dropout"]).to(device)

    n_params = raw_model.count_params()
    logger.log(f"Model: {args.model} | Params: {n_params:,}")
    logger.log(f"Effective batch: {cfg['batch_size_per_gpu'] * world_size} (per_gpu={cfg['batch_size_per_gpu']} x {world_size})")

    model = DDP(raw_model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                   weight_decay=cfg["weight_decay"], betas=(0.9, 0.98))
    def lr_lambda(step):
        if step < cfg["warmup_steps"]: return step / cfg["warmup_steps"]
        p = (step - cfg["warmup_steps"]) / max(cfg["max_steps"] - cfg["warmup_steps"], 1)
        return 0.5 * (1 + math.cos(math.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step, best_val_loss = 0, float("inf")
    start_time = time.time()
    training_curve = []
    data_iter = iter(train_loader)
    epoch = 0
    model.train()
    logger.log("--- Training starts ---")

    while step < cfg["max_steps"]:
        try:
            x, y = next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        do_log = (step + 1) % cfg["log_interval"] == 0

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, info = model(x, n_iterations=cfg["n_iterations"], measure_ranks=do_log)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()
        scheduler.step()
        step += 1

        if step % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            ppl = math.exp(min(loss.item(), 20))
            tps = step * cfg["batch_size_per_gpu"] * world_size * cfg["seq_len"] / elapsed
            ranks = info.get('ranks', [])
            rank_str = ", ".join(f"{r:.2f}" for r in ranks) if ranks else "-"
            logger.log(f"Step {step:5d} | Loss {loss.item():.4f} | PPL {ppl:8.1f} | "
                       f"GN {grad_norm:.3f} | LR {scheduler.get_last_lr()[0]:.2e} | "
                       f"Rank [{rank_str}] | {tps:.0f} tok/s | {elapsed:.0f}s")

        if step % cfg["eval_interval"] == 0:
            if is_main:
                logger.log(f"--- Eval step {step} ---")
                eval_results = {}
                for T in cfg["eval_iterations"]:
                    vl, vp, vr = evaluate(raw_model, val_ds, vocab_size, device, T, cfg["eval_batch_size"])
                    rank_str = ", ".join(f"{r:.2f}" for r in vr) if vr else "-"
                    marker = ""
                    if T == cfg["n_iterations"] and vl < best_val_loss:
                        best_val_loss = vl
                        torch.save(raw_model.state_dict(), results_dir / "best.pt")
                        marker = " *BEST*"
                    logger.log(f"  T={T:2d}: PPL {vp:8.1f} | Rank [{rank_str}]{marker}")
                    eval_results[f"T{T}"] = {"val_loss": vl, "val_ppl": vp, "ranks": vr}
                training_curve.append({"step": step, "train_loss": loss.item(), "evals": eval_results})
            dist.barrier()

    # Final summary (rank 0)
    elapsed = time.time() - start_time
    if is_main:
        logger.log(f"\n{'='*70}")
        logger.log(f"DONE: {args.model} thinker | {step} steps | {elapsed/60:.1f} min")

        raw_model.load_state_dict(torch.load(results_dir / "best.pt", weights_only=True))
        final = {}
        for T in cfg["eval_iterations"]:
            vl, vp, vr = evaluate(raw_model, val_ds, vocab_size, device, T, cfg["eval_batch_size"])
            rank_str = ", ".join(f"{r:.2f}" for r in vr) if vr else "-"
            logger.log(f"  T={T:2d}: PPL {vp:8.1f} | Rank [{rank_str}]")
            final[f"T{T}"] = {"val_loss": vl, "val_ppl": vp, "ranks": vr}

        ppls = {T: final[f"T{T}"]["val_ppl"] for T in cfg["eval_iterations"]}
        t1, tn = cfg["eval_iterations"][0], cfg["eval_iterations"][-1]
        delta = ppls[t1] - ppls[tn]
        pct = delta / ppls[t1] * 100 if ppls[t1] > 0 else 0

        summary = "\n".join([
            "", "=" * 70,
            f"  {args.model.upper()} THINKER — ROUND 1 RESULTS",
            "=" * 70,
            f"  Params: {n_params:,} | {elapsed/60:.1f} min on {world_size}x H100",
            f"  Data: {meta.get('train_tokens', '?'):,} tokens | Steps: {step}",
            f"  Effective batch: {cfg['batch_size_per_gpu'] * world_size}",
            "",
            *[f"  T={T:2d}: PPL {ppls[T]:8.1f}" for T in cfg["eval_iterations"]],
            "",
            f"  Thinking benefit: {delta:.1f} PPL ({pct:.1f}%)",
            "=" * 70, ""
        ])
        logger.log(summary)
        with open(results_dir / "SUMMARY.txt", 'w') as f: f.write(summary)
        json.dump({"experiment": f"round1_{args.model}", "params": n_params,
                    "config": cfg, "final_evals": final, "training_curve": training_curve,
                    "thinking_benefit_pct": pct, "time_min": elapsed/60},
                   open(results_dir / "results.json", 'w'), indent=2, default=float)
        logger.log(f"Saved to {results_dir}")
        logger.close()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
