#!/usr/bin/env python3
"""
PARAM-MATCHED ABLATION: Matrix ops vs flat vector ops
=====================================================
Same outer-product embedding. Same total params (~2.55M). Same data. Same training.
ONLY DIFFERENCE: thinking layers treat representation as 16x16 matrix vs 256-dim vector.

This is THE experiment that determines whether matrix operations help or just the embedding.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math, time, json, os, shutil, argparse
from pathlib import Path
import datetime as dt_module
from datetime import datetime, timezone

BASE_CONFIG = {
    "mat_dim": 16, "n_layers": 12, "n_heads": 4, "n_iterations": 8,
    "max_len": 2048, "dropout": 0.1, "batch_size_per_gpu": 96, "seq_len": 512,
    "max_steps": 3000, "lr": 3e-4, "warmup_steps": 300, "weight_decay": 0.01,
    "grad_clip": 1.0, "eval_interval": 500, "eval_batch_size": 96,
    "eval_iterations": [1, 2, 4, 8], "log_interval": 50,
    "data_dir": "/toy_story_slam/data", "fallback_data_dir": "/toy_story_slam/data_quick",
    "num_workers": 8,
}

# ═══════════════════════════════════════════════════════════════════════════
# SHARED: Outer-product embedding (identical for both models)
# ═══════════════════════════════════════════════════════════════════════════

class OuterProductEmbed(nn.Module):
    def __init__(self, d, vocab_size, max_len):
        super().__init__()
        self.d = d
        self.embed_u = nn.Embedding(vocab_size, d)
        self.embed_v = nn.Embedding(vocab_size, d)
        self.pos_u = nn.Embedding(max_len, d)
        self.pos_v = nn.Embedding(max_len, d)
    def forward(self, token_ids):
        B, L = token_ids.shape
        u, v = self.embed_u(token_ids), self.embed_v(token_ids)
        M = torch.einsum('...i,...j->...ij', u, v)
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        pu, pv = self.pos_u(pos), self.pos_v(pos)
        return M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1

# ═══════════════════════════════════════════════════════════════════════════
# MODEL A: Matrix thinking (the architecture we've been testing)
# ═══════════════════════════════════════════════════════════════════════════

class MatrixRMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d, d)); self.eps = eps
    def forward(self, M):
        return M / torch.sqrt(M.pow(2).mean(dim=(-2,-1), keepdim=True) + self.eps) * self.weight

class RowThenColProjection(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
        self.B = nn.Parameter(torch.eye(d) + 0.02 * torch.randn(d, d))
    def forward(self, M):
        return torch.einsum('...ij,jk->...ik', F.silu(torch.einsum('ij,...jk->...ik', self.A, M)), self.B)

class MatrixAttention(nn.Module):
    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        self.d, self.n_heads, self.head_dim = d, n_heads, d // n_heads
        self.norm = MatrixRMSNorm(d)
        self.q_proj = RowThenColProjection(d); self.k_proj = RowThenColProjection(d)
        self.v_proj = RowThenColProjection(d); self.o_proj = RowThenColProjection(d)
        self.dropout_p = dropout
    def forward(self, M):
        B, L, d, _ = M.shape; H, hd = self.n_heads, self.head_dim
        M_n = self.norm(M)
        Q = self.q_proj(M_n).reshape(B,L,H,hd,d).permute(0,2,1,3,4).reshape(B,H,L,hd*d)
        K = self.k_proj(M_n).reshape(B,L,H,hd,d).permute(0,2,1,3,4).reshape(B,H,L,hd*d)
        V = self.v_proj(M_n).reshape(B,L,H,hd,d).permute(0,2,1,3,4).reshape(B,H,L,hd*d)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0)
        return M + self.o_proj(out.reshape(B,H,L,hd,d).permute(0,2,1,3,4).reshape(B,L,d,d))

class MatrixThinkingLayer(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.norm = MatrixRMSNorm(d)
        self.delta_gate = RowThenColProjection(d); self.delta_value = RowThenColProjection(d)
        self.delta_up = RowThenColProjection(d)
        self.gamma_gate = RowThenColProjection(d); self.gamma_value = RowThenColProjection(d)
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
        M_n = self.norm(M); s = self.scale.clamp(0.01, 0.5)
        delta = self.delta_up(F.silu(self.delta_gate(M_n)) * self.delta_value(M_n)) * s
        gamma = self.gamma_up(F.silu(self.gamma_gate(M_n)) * self.gamma_value(M_n)) * s
        M_mult = torch.matmul(torch.matmul(self.I + delta, M_n), self.I + gamma)
        k = torch.matmul(M_n, self.key_col).squeeze(-1)
        v = torch.matmul(M_n, self.val_col).squeeze(-1)
        M_write = torch.einsum('...i,...j->...ij', v, k)
        g_m = torch.sigmoid((self.gate_mult_W * M_n).sum(dim=(-2,-1), keepdim=True) + self.gate_mult_bias)
        g_w = torch.sigmoid((self.gate_write_W * M_n).sum(dim=(-2,-1), keepdim=True) + self.gate_write_bias)
        return M + self.dropout(g_m * (M_mult - M_n) + g_w * M_write)

class MatrixBlock(nn.Module):
    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = MatrixAttention(d, n_heads, dropout)
        self.think = MatrixThinkingLayer(d, dropout)
    def forward(self, M): return self.think(self.attn(M))

class MultiProbeHead(nn.Module):
    def __init__(self, d, vocab_size, n_probes=None):
        super().__init__()
        K = n_probes or d
        self.U = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))
        self.V = nn.Parameter(torch.randn(K, d) * (1.0 / math.sqrt(d)))
        self.out = nn.Linear(K, vocab_size, bias=False)
    def forward(self, M):
        MV = torch.einsum('blij, kj -> blik', M, self.V)
        return self.out(torch.einsum('ki, blik -> blk', self.U, MV))

class MatrixModel(nn.Module):
    def __init__(self, d=16, n_layers=12, n_heads=4, max_len=2048, vocab_size=50257, dropout=0.1):
        super().__init__()
        self.d = d
        self.embed = OuterProductEmbed(d, vocab_size, max_len)
        self.layers = nn.ModuleList([MatrixBlock(d, n_heads, dropout) for _ in range(n_layers)])
        self.final_norm = MatrixRMSNorm(d)
        self.output_head = MultiProbeHead(d, vocab_size, d)
    def _one_iter(self, M):
        for layer in self.layers:
            if self.training: M = torch.utils.checkpoint.checkpoint(layer, M, use_reentrant=False)
            else: M = layer(M)
        return M
    def forward(self, token_ids, n_iterations=1, **kwargs):
        M = self.embed(token_ids)
        for t in range(n_iterations): M = self._one_iter(M)
        return self.output_head(self.final_norm(M)), {}
    def count_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ═══════════════════════════════════════════════════════════════════════════
# MODEL B: Flat vector ops (same embed, flatten, standard transformer)
# Param-matched: we size the FFN so total params ≈ matrix model
# ═══════════════════════════════════════════════════════════════════════════

class FlatBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.SiLU(), nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
    def forward(self, h):
        h_n = self.norm1(h)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(h.shape[1], device=h.device)
        attn_out, _ = self.attn(h_n, h_n, h_n, attn_mask=mask)
        h = h + attn_out
        return h + self.ffn(self.norm2(h))

class FlatModel(nn.Module):
    """Same outer-product embed, flatten to d²-dim vector, standard transformer.
    Reshape back to d×d for MultiProbeHead output (same head as matrix model).
    FFN dim=32 to keep total params ≈ matrix model."""
    def __init__(self, d=16, n_layers=12, n_heads=4, ff_dim=32,
                 max_len=2048, vocab_size=50257, dropout=0.1):
        super().__init__()
        self.d = d
        d_model = d * d  # 256
        self.d_model = d_model
        self.embed = OuterProductEmbed(d, vocab_size, max_len)
        self.layers = nn.ModuleList([FlatBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.output_head = MultiProbeHead(d, vocab_size, d)  # SAME output as matrix model
    def _one_iter(self, h):
        for layer in self.layers:
            if self.training: h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else: h = layer(h)
        return h
    def forward(self, token_ids, n_iterations=1, **kwargs):
        B, L = token_ids.shape
        M = self.embed(token_ids)
        h = M.reshape(B, L, self.d_model)  # FLATTEN to 256-dim
        for t in range(n_iterations): h = self._one_iter(h)
        h = self.norm(h)
        M_out = h.reshape(B, L, self.d, self.d)  # RESHAPE back for output
        return self.output_head(M_out), {}
    def count_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ═══════════════════════════════════════════════════════════════════════════
# DATA + EVAL + TRAINING
# ═══════════════════════════════════════════════════════════════════════════

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, path, seq_len):
        self.data = torch.load(path, weights_only=True); self.seq_len = seq_len
        self.n = len(self.data) // seq_len - 1
    def __len__(self): return self.n
    def __getitem__(self, i):
        s = i * self.seq_len; c = self.data[s:s+self.seq_len+1]; return c[:-1], c[1:]

def evaluate(model, val_ds, vocab_size, device, n_iterations, eval_batch_size=96, max_eval_batches=50):
    model.eval()
    loader = torch.utils.data.DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, drop_last=True, num_workers=4)
    total_loss, total_tokens, n_batches = 0, 0, 0
    with torch.no_grad():
        for vx, vy in loader:
            if n_batches >= max_eval_batches: break
            n_batches += 1; vx, vy = vx.to(device), vy.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(vx, n_iterations=n_iterations)
            n_tok = vy.numel()
            total_loss += F.cross_entropy(logits.reshape(-1, vocab_size), vy.reshape(-1)).item() * n_tok
            total_tokens += n_tok
    avg_loss = total_loss / max(total_tokens, 1)
    model.train()
    return avg_loss, math.exp(min(avg_loss, 20)), avg_loss / math.log(2) / 3.7

def run_model(model_type, raw_model, cfg, train_ds, val_ds, vocab_size, device, local_rank, world_size, is_main):
    results_dir = Path(cfg["results_dir"])
    if is_main: results_dir.mkdir(parents=True, exist_ok=True); shutil.copy2(Path(__file__).resolve(), results_dir / "script.py")

    class Logger:
        def __init__(self, path): self.f = open(path, 'w') if is_main else None
        def log(self, msg):
            if not is_main: return
            line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"; print(line, flush=True)
            if self.f: self.f.write(line + '\n'); self.f.flush()
        def close(self):
            if self.f: self.f.close()

    logger = Logger(results_dir / "train.log")
    n_params = raw_model.count_params()
    logger.log(f"Model: {model_type} | Params: {n_params:,} | GPUs: {world_size}")

    model = DDP(raw_model, device_ids=[local_rank])
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size_per_gpu"], sampler=sampler, drop_last=True, num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.98))
    def lr_lambda(step):
        if step < cfg["warmup_steps"]: return step / cfg["warmup_steps"]
        return 0.5 * (1 + math.cos(math.pi * (step - cfg["warmup_steps"]) / max(cfg["max_steps"] - cfg["warmup_steps"], 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step, best_val_loss, start_time = 0, float("inf"), time.time()
    training_curve, data_iter, epoch = [], iter(train_loader), 0
    model.train(); logger.log("--- Training starts ---")

    while step < cfg["max_steps"]:
        try: x, y = next(data_iter)
        except StopIteration: epoch += 1; sampler.set_epoch(epoch); data_iter = iter(train_loader); x, y = next(data_iter)
        x, y = x.to(device), y.to(device); optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x, n_iterations=cfg["n_iterations"])
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step(); scheduler.step(); step += 1
        if step % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time; bpb = loss.item() / math.log(2) / 3.7
            tps = step * cfg["batch_size_per_gpu"] * world_size * cfg["seq_len"] / elapsed
            logger.log(f"Step {step:5d} | BPB {bpb:.3f} | PPL {math.exp(min(loss.item(),20)):8.1f} | GN {grad_norm:.3f} | {tps:.0f} tok/s | {elapsed:.0f}s")
        if step % cfg["eval_interval"] == 0:
            if is_main:
                logger.log(f"--- Eval step {step} ---"); eval_results = {}
                for T in cfg["eval_iterations"]:
                    vl, vp, vbpb = evaluate(raw_model, val_ds, vocab_size, device, T, cfg["eval_batch_size"])
                    marker = ""
                    if T == cfg["n_iterations"] and vl < best_val_loss:
                        best_val_loss = vl; torch.save(raw_model.state_dict(), results_dir / "best.pt"); marker = " *BEST*"
                    logger.log(f"  T={T:2d}: BPB {vbpb:.3f}{marker}")
                    eval_results[f"T{T}"] = {"val_loss": vl, "val_ppl": vp, "val_bpb": vbpb}
                training_curve.append({"step": step, "train_loss": loss.item(), "evals": eval_results})
            dist.barrier()

    elapsed = time.time() - start_time
    if is_main:
        logger.log(f"\n{'='*70}"); logger.log(f"DONE: {model_type} | {step} steps | {elapsed/60:.1f} min")
        raw_model.load_state_dict(torch.load(results_dir / "best.pt", weights_only=True))
        final = {}
        for T in cfg["eval_iterations"]:
            vl, vp, vbpb = evaluate(raw_model, val_ds, vocab_size, device, T, cfg["eval_batch_size"])
            logger.log(f"  T={T:2d}: BPB {vbpb:.3f}")
            final[f"T{T}"] = {"val_loss": vl, "val_ppl": vp, "val_bpb": vbpb}
        bpbs = {T: final[f"T{T}"]["val_bpb"] for T in cfg["eval_iterations"]}
        t1, tn = cfg["eval_iterations"][0], cfg["eval_iterations"][-1]
        summary = "\n".join([
            "", "="*70, f"  PARAM-MATCHED ABLATION: {model_type.upper()}", "="*70,
            f"  Params: {n_params:,}", f"  Time: {elapsed/60:.1f} min on {world_size}x H100", "",
            *[f"  T={T:2d}: BPB {bpbs[T]:.3f}" for T in cfg["eval_iterations"]], "",
            f"  Thinking: BPB {bpbs[t1]:.3f} -> {bpbs[tn]:.3f}",
            "="*70, ""])
        logger.log(summary)
        with open(results_dir / "SUMMARY.txt", 'w') as f: f.write(summary)
        json.dump({"experiment": f"ablation_{model_type}", "params": n_params, "model_type": model_type,
                    "config": cfg, "final_evals": final, "training_curve": training_curve,
                    "time_min": elapsed/60}, open(results_dir / "results.json", 'w'), indent=2, default=float)
        logger.close()
    return final

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["matrix", "flat"], required=True)
    args = parser.parse_args()

    cfg = BASE_CONFIG.copy()
    cfg["results_dir"] = f"/toy_story_slam/results/ablation_matched_{args.model}"

    dist.init_process_group("nccl", timeout=dt_module.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"]); world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank); device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    data_dir = cfg["data_dir"]
    if not os.path.exists(os.path.join(data_dir, "meta.json")): data_dir = cfg["fallback_data_dir"]
    meta = json.load(open(os.path.join(data_dir, "meta.json"))); vocab_size = meta["vocab_size"]
    train_ds = TokenDataset(os.path.join(data_dir, "train.pt"), cfg["seq_len"])
    val_ds = TokenDataset(os.path.join(data_dir, "val.pt"), cfg["seq_len"])

    dist.barrier()

    if args.model == "matrix":
        raw_model = MatrixModel(d=16, n_layers=12, n_heads=4, max_len=cfg["max_len"],
                                 vocab_size=vocab_size, dropout=cfg["dropout"]).to(device)
    else:
        # Size FFN to match matrix model params
        # Matrix model: ~2.55M params. Embedding is shared (~1.67M).
        # Matrix thinking layers: ~74K. Output head: ~805K.
        # Flat model needs: embed(1.67M) + layers(?) + output(50257*256=12.9M) — TOO MANY output params
        # Use tied output to match: output.weight = embed_u.weight... but dims don't match
        # Instead: use small FFN so total params ≈ matrix model
        # With ff_dim=32: each layer has 256*4*4(attn) + 256*2(LN) + 256*32+32*256(FFN) = ~21K
        # 12 layers * 21K = 252K. Plus embed 1.67M + output 256*50257=12.9M = 14.8M. Way too much.
        # The output Linear(256, 50257) is 12.9M params alone — impossible to match at 2.5M total.
        # Solution: tied output. output.weight = embed projecting from d²=256 back through embed tables.
        # Or: use MultiProbeHead same as matrix model for fair comparison.
        # FAIREST: both use MultiProbeHead. Flat model reshapes 256-dim back to 16x16 for output.
        raw_model = FlatModel(d=16, n_layers=12, n_heads=4, ff_dim=32,
                               max_len=cfg["max_len"], vocab_size=vocab_size, dropout=cfg["dropout"]).to(device)

    if is_main:
        n_p = raw_model.count_params()
        print(f"\n{'='*60}\n  {args.model.upper()} MODEL: {n_p:,} params\n{'='*60}\n", flush=True)

    run_model(args.model, raw_model, cfg, train_ds, val_ds, vocab_size, device, local_rank, world_size, is_main)
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__": run()
