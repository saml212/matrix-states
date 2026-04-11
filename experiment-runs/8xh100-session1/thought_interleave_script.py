#!/usr/bin/env python3
"""
CoCoMix-Style Matrix Thought Interleaving
==========================================
Insert N matrix thought tokens per byte at layer K. Process bytes + thoughts
together through remaining layers. Loss only on byte positions.

Proven approach: CoCoMix (Meta 2025) achieved 21.5% sample efficiency gain.

Start with N=1 thought per byte. Baseline: same model with N=0 (no thoughts).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math, time, json, os, shutil, argparse
from pathlib import Path
import datetime as dt_module
from datetime import datetime, timezone

CONFIG = {
    "mat_dim": 16, "n_layers": 24, "n_heads": 4,
    "insert_layer": 8,       # insert thoughts after layer 8 (of 24)
    "n_thoughts": 1,         # thoughts per byte (start with 1)
    "max_len": 4096, "dropout": 0.1,
    "batch_size_per_gpu": 128, "seq_len": 512, "max_steps": 3000,
    "lr": 3e-4, "warmup_steps": 300, "weight_decay": 0.01, "grad_clip": 1.0,
    "eval_interval": 500, "eval_batch_size": 128,
    "log_interval": 50, "data_dir": "/toy_story_slam/data_bytes",
    "num_workers": 8,
}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS
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

class MatrixFrobeniusAttention(nn.Module):
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

class MultiplicativeThinkingLayer(nn.Module):
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

class ThinkingBlock(nn.Module):
    def __init__(self, d, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = MatrixFrobeniusAttention(d, n_heads, dropout)
        self.think = MultiplicativeThinkingLayer(d, dropout)
    def forward(self, M): return self.think(self.attn(M))

# ═══════════════════════════════════════════════════════════════════════════
# THOUGHT INTERLEAVING MODEL
# ═══════════════════════════════════════════════════════════════════════════

class ThoughtInterleavingModel(nn.Module):
    """
    CoCoMix-style: process bytes through layers 1-K, generate matrix thoughts
    from hidden states, interleave bytes+thoughts, process through layers K+1-N.
    Loss only on byte positions.
    """
    def __init__(self, mat_dim=16, n_layers=24, n_heads=4, insert_layer=8,
                 n_thoughts=1, max_len=4096, vocab_size=256, dropout=0.1):
        super().__init__()
        self.mat_dim = mat_dim; d = mat_dim
        self.n_thoughts = n_thoughts
        self.insert_layer = insert_layer

        # Byte embedding
        self.embed_u = nn.Embedding(vocab_size, d)
        self.embed_v = nn.Embedding(vocab_size, d)
        self.pos_u = nn.Embedding(max_len, d)
        self.pos_v = nn.Embedding(max_len, d)

        # All layers
        self.layers = nn.ModuleList([ThinkingBlock(d, n_heads, dropout) for _ in range(n_layers)])

        # Thought generation: project hidden state → thought matrix
        self.thought_proj = RowThenColProjection(d)
        self.thought_pos = nn.Embedding(n_thoughts, d * d)  # position within thought group

        # Output
        self.final_norm = MatrixRMSNorm(d)
        self.output_bias = nn.Parameter(torch.zeros(d, d))
        self.log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, token_ids, use_thoughts=True):
        B, L = token_ids.shape; d = self.mat_dim

        # Embed bytes as matrices
        u, v = self.embed_u(token_ids), self.embed_v(token_ids)
        M = torch.einsum('...i,...j->...ij', u, v)
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        pu, pv = self.pos_u(pos), self.pos_v(pos)
        M = M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1

        # Run pre-insert layers
        for layer in self.layers[:self.insert_layer]:
            if self.training:
                M = torch.utils.checkpoint.checkpoint(layer, M, use_reentrant=False)
            else:
                M = layer(M)

        if use_thoughts and self.n_thoughts > 0:
            # Generate thought matrices from hidden states
            thoughts = self.thought_proj(M)  # (B, L, d, d) — one thought seed per byte

            # Add thought position embedding
            if self.n_thoughts == 1:
                t_pos = self.thought_pos(torch.tensor(0, device=M.device))
                thoughts = thoughts + t_pos.reshape(1, 1, d, d) * 0.1
                # Interleave: [byte₀, thought₀, byte₁, thought₁, ...]
                interleaved = torch.stack([M, thoughts], dim=2)  # (B, L, 2, d, d)
                interleaved = interleaved.reshape(B, L * 2, d, d)
            else:
                # Multiple thoughts per byte
                thought_list = []
                for t in range(self.n_thoughts):
                    t_pos = self.thought_pos(torch.tensor(t, device=M.device))
                    thought_t = thoughts + t_pos.reshape(1, 1, d, d) * 0.1
                    thought_list.append(thought_t)
                # Interleave: [byte₀, t₀₀, t₀₁, ..., byte₁, t₁₀, t₁₁, ...]
                parts = []
                for i in range(self.n_thoughts):
                    if i == 0:
                        parts.append(M.unsqueeze(2))  # byte position
                    parts.append(thought_list[i].unsqueeze(2))
                interleaved = torch.cat(parts, dim=2)  # (B, L, 1+N, d, d)
                interleaved = interleaved.reshape(B, L * (1 + self.n_thoughts), d, d)

            # Run post-insert layers on interleaved sequence
            for layer in self.layers[self.insert_layer:]:
                if self.training:
                    interleaved = torch.utils.checkpoint.checkpoint(layer, interleaved, use_reentrant=False)
                else:
                    interleaved = layer(interleaved)

            # Extract byte positions (every 1+N positions, starting at 0)
            stride = 1 + self.n_thoughts
            byte_indices = torch.arange(L, device=M.device) * stride
            M_out = interleaved[:, byte_indices]  # (B, L, d, d)

            # Also extract thought positions for rank analysis
            thought_matrices = interleaved[:, byte_indices + 1] if self.n_thoughts >= 1 else None
        else:
            # No thoughts — just run remaining layers on bytes only
            for layer in self.layers[self.insert_layer:]:
                if self.training:
                    M = torch.utils.checkpoint.checkpoint(layer, M, use_reentrant=False)
                else:
                    M = layer(M)
            M_out = M
            thought_matrices = None

        # Output: matrix IS the byte prediction
        M_normed = self.final_norm(M_out)
        logits = (M_normed + self.output_bias).reshape(B, L, d * d) * self.log_temp.exp()

        # Measure ranks
        ranks = {}
        with torch.no_grad():
            def measure_rank(tensor, name):
                Mf = tensor.detach().float().reshape(-1, d, d)
                if Mf.shape[0] > 512: Mf = Mf[torch.randperm(Mf.shape[0], device=Mf.device)[:512]]
                S = torch.linalg.svdvals(Mf).clamp(min=1e-10)
                Sn = S / S.sum(-1, keepdim=True)
                ranks[name] = (-(Sn * Sn.log()).sum(-1)).exp().mean().item()
            measure_rank(M_out, 'byte_rank')
            if thought_matrices is not None:
                measure_rank(thought_matrices, 'thought_rank')

        return logits, ranks

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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

def evaluate(model, val_ds, device, use_thoughts=True, max_batches=50, batch_size=128):
    model.eval()
    loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    total_loss, total_tokens, n_batches = 0, 0, 0
    all_ranks = []
    with torch.no_grad():
        for vx, vy in loader:
            if n_batches >= max_batches: break
            n_batches += 1; vx, vy = vx.to(device), vy.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, ranks = model(vx, use_thoughts=use_thoughts)
            total_loss += F.cross_entropy(logits.reshape(-1, 256), vy.reshape(-1)).item() * vy.numel()
            total_tokens += vy.numel()
            if ranks: all_ranks.append(ranks)
    avg_loss = total_loss / max(total_tokens, 1)
    avg_bpb = avg_loss / math.log(2)
    avg_ranks = {}
    if all_ranks:
        for key in all_ranks[0]:
            avg_ranks[key] = sum(r[key] for r in all_ranks) / len(all_ranks)
    model.train()
    return avg_loss, avg_bpb, avg_ranks

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-thoughts", action="store_true", help="Run baseline without thoughts")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    use_thoughts = not args.no_thoughts
    exp_name = "thought_interleave" if use_thoughts else "no_thought_baseline"
    cfg["results_dir"] = f"/toy_story_slam/results/{exp_name}"

    dist.init_process_group("nccl", timeout=dt_module.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"]); world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank); device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

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

    dist.barrier(); logger = Logger(results_dir / "train.log")
    logger.log(f"{'THOUGHT INTERLEAVING' if use_thoughts else 'NO-THOUGHT BASELINE'} | GPUs: {world_size}")
    logger.log(f"Thoughts: N={cfg['n_thoughts'] if use_thoughts else 0}, insert_layer={cfg['insert_layer']}")

    meta = json.load(open(os.path.join(cfg["data_dir"], "meta.json")))
    vocab_size = meta["vocab_size"]
    logger.log(f"Data: {meta.get('train_tokens', '?'):,} bytes")

    train_ds = TokenDataset(os.path.join(cfg["data_dir"], "train.pt"), cfg["seq_len"])
    val_ds = TokenDataset(os.path.join(cfg["data_dir"], "val.pt"), cfg["seq_len"])
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size_per_gpu"], sampler=sampler, drop_last=True, num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=True)

    raw_model = ThoughtInterleavingModel(
        mat_dim=cfg["mat_dim"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        insert_layer=cfg["insert_layer"], n_thoughts=cfg["n_thoughts"] if use_thoughts else 0,
        max_len=cfg["max_len"], vocab_size=vocab_size, dropout=cfg["dropout"]
    ).to(device)

    n_params = raw_model.count_params()
    logger.log(f"Params: {n_params:,}")

    model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=True)
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
        do_log = (step + 1) % cfg["log_interval"] == 0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, ranks = model(x, use_thoughts=use_thoughts)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step(); scheduler.step(); step += 1
        if step % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time; bpb = loss.item() / math.log(2)
            tps = step * cfg["batch_size_per_gpu"] * world_size * cfg["seq_len"] / elapsed
            rank_str = " | ".join(f"{k}={v:.2f}" for k,v in ranks.items()) if ranks else "-"
            temp = raw_model.log_temp.exp().item()
            logger.log(f"Step {step:5d} | BPB {bpb:.3f} | GN {grad_norm:.3f} | Temp {temp:.3f} | {rank_str} | {tps:.0f} tok/s | {elapsed:.0f}s")
        if step % cfg["eval_interval"] == 0:
            if is_main:
                logger.log(f"--- Eval step {step} ---")
                # Eval WITH thoughts
                vl, vbpb, vranks = evaluate(raw_model, val_ds, device, use_thoughts=use_thoughts, batch_size=cfg["eval_batch_size"])
                rank_str = " | ".join(f"{k}={v:.2f}" for k,v in vranks.items()) if vranks else "-"
                marker = ""
                if vl < best_val_loss:
                    best_val_loss = vl; torch.save(raw_model.state_dict(), results_dir / "best.pt"); marker = " *BEST*"
                logger.log(f"  BPB {vbpb:.3f} | {rank_str}{marker}")
                # Also eval WITHOUT thoughts for comparison
                if use_thoughts:
                    _, no_t_bpb, _ = evaluate(raw_model, val_ds, device, use_thoughts=False, batch_size=cfg["eval_batch_size"])
                    logger.log(f"  (no thoughts: BPB {no_t_bpb:.3f})")
                training_curve.append({"step": step, "loss": loss.item(), "bpb": vbpb,
                                       "ranks": vranks, "no_thought_bpb": no_t_bpb if use_thoughts else vbpb})
            dist.barrier()

    elapsed = time.time() - start_time
    if is_main:
        logger.log(f"\n{'='*70}")
        logger.log(f"DONE: {exp_name} | {step} steps | {elapsed/60:.1f} min")
        raw_model.load_state_dict(torch.load(results_dir / "best.pt", weights_only=True))
        vl, vbpb, vranks = evaluate(raw_model, val_ds, device, use_thoughts=use_thoughts, batch_size=cfg["eval_batch_size"])
        no_t_bpb = None
        if use_thoughts:
            _, no_t_bpb, _ = evaluate(raw_model, val_ds, device, use_thoughts=False, batch_size=cfg["eval_batch_size"])

        summary = "\n".join([
            "", "="*70,
            f"  {'THOUGHT INTERLEAVING' if use_thoughts else 'NO-THOUGHT BASELINE'}",
            "="*70,
            f"  Params: {n_params:,} | {elapsed/60:.1f} min on {world_size}x H100",
            f"  Thoughts: N={cfg['n_thoughts'] if use_thoughts else 0}, insert at layer {cfg['insert_layer']} of {cfg['n_layers']}",
            f"  BPB: {vbpb:.3f}",
            f"  Byte rank: {vranks.get('byte_rank', 'N/A'):.2f}" if 'byte_rank' in vranks else "",
            f"  Thought rank: {vranks.get('thought_rank', 'N/A'):.2f}" if 'thought_rank' in vranks else "",
            f"  Without thoughts: BPB {no_t_bpb:.3f}" if no_t_bpb else "",
            f"  Thought benefit: {((no_t_bpb - vbpb) / no_t_bpb * 100):.1f}%" if no_t_bpb and no_t_bpb != vbpb else "",
            "="*70, ""])
        logger.log(summary)
        with open(results_dir / "SUMMARY.txt", 'w') as f: f.write(summary)
        json.dump({"experiment": exp_name, "params": n_params, "config": cfg,
                    "bpb": vbpb, "no_thought_bpb": no_t_bpb, "ranks": vranks,
                    "training_curve": training_curve, "time_min": elapsed/60},
                   open(results_dir / "results.json", 'w'), indent=2, default=float)
        logger.close()
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__": run()
