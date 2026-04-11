#!/usr/bin/env python3
"""
Full Sweep: All thought variants in one script
===============================================
Runs sequentially: each config trains, evals, saves, then next config starts.
Configs:
  A. MultiProbeHead output + N=1 thoughts (tests if output head matters for rank enrichment)
  B. Zero-param output + N=2 thoughts (more thinking slots)
  C. Zero-param output + N=4 thoughts (even more thinking)
  D. Scaled up: 48 layers, ~500K params, N=1 thoughts (more capacity)
  E. Baseline for D: 48 layers, ~500K params, N=0 thoughts
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math, time, json, os, shutil
from pathlib import Path
import datetime as dt_module
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════════
# MODEL (same as thought_interleave but with MultiProbeHead option)
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

class MatrixByteOutput(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(d, d))
        self.log_temp = nn.Parameter(torch.tensor(0.0))
    def forward(self, M):
        B, L, d, _ = M.shape
        return (M + self.bias).reshape(B, L, d * d) * self.log_temp.exp()

class SweepModel(nn.Module):
    def __init__(self, mat_dim=16, n_layers=24, n_heads=4, insert_layer=8,
                 n_thoughts=1, output_type='zero_param', max_len=4096, vocab_size=256, dropout=0.1):
        super().__init__()
        self.mat_dim = mat_dim; d = mat_dim
        self.n_thoughts = n_thoughts
        self.insert_layer = insert_layer
        self.output_type = output_type
        self.embed_u = nn.Embedding(vocab_size, d); self.embed_v = nn.Embedding(vocab_size, d)
        self.pos_u = nn.Embedding(max_len, d); self.pos_v = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([ThinkingBlock(d, n_heads, dropout) for _ in range(n_layers)])
        if n_thoughts > 0:
            self.thought_proj = RowThenColProjection(d)
            self.thought_pos = nn.Embedding(n_thoughts, d * d)
        self.final_norm = MatrixRMSNorm(d)
        if output_type == 'multiprobe':
            self.output_head = MultiProbeHead(d, vocab_size, d)
        else:
            self.output_head = MatrixByteOutput(d)

    def forward(self, token_ids, use_thoughts=True):
        B, L = token_ids.shape; d = self.mat_dim
        u, v = self.embed_u(token_ids), self.embed_v(token_ids)
        M = torch.einsum('...i,...j->...ij', u, v)
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        pu, pv = self.pos_u(pos), self.pos_v(pos)
        M = M + torch.einsum('...i,...j->...ij', pu, pv) * 0.1

        for layer in self.layers[:self.insert_layer]:
            if self.training: M = torch.utils.checkpoint.checkpoint(layer, M, use_reentrant=False)
            else: M = layer(M)

        if use_thoughts and self.n_thoughts > 0:
            thoughts = self.thought_proj(M)
            parts = [M.unsqueeze(2)]
            for t in range(self.n_thoughts):
                t_pos = self.thought_pos(torch.tensor(t, device=M.device))
                parts.append((thoughts + t_pos.reshape(1, 1, d, d) * 0.1).unsqueeze(2))
            interleaved = torch.cat(parts, dim=2).reshape(B, L * (1 + self.n_thoughts), d, d)
            for layer in self.layers[self.insert_layer:]:
                if self.training: interleaved = torch.utils.checkpoint.checkpoint(layer, interleaved, use_reentrant=False)
                else: interleaved = layer(interleaved)
            stride = 1 + self.n_thoughts
            M_out = interleaved[:, torch.arange(L, device=M.device) * stride]
            # Get thought matrices for rank measurement
            thought_mats = interleaved[:, torch.arange(L, device=M.device) * stride + 1] if self.n_thoughts >= 1 else None
        else:
            for layer in self.layers[self.insert_layer:]:
                if self.training: M = torch.utils.checkpoint.checkpoint(layer, M, use_reentrant=False)
                else: M = layer(M)
            M_out = M; thought_mats = None

        logits = self.output_head(self.final_norm(M_out))
        ranks = {}
        with torch.no_grad():
            def mr(tensor, name):
                Mf = tensor.detach().float().reshape(-1, d, d)
                if Mf.shape[0] > 512: Mf = Mf[torch.randperm(Mf.shape[0], device=Mf.device)[:512]]
                S = torch.linalg.svdvals(Mf).clamp(min=1e-10); Sn = S / S.sum(-1, keepdim=True)
                ranks[name] = (-(Sn * Sn.log()).sum(-1)).exp().mean().item()
            mr(M_out, 'byte_rank')
            if thought_mats is not None: mr(thought_mats, 'thought_rank')
        return logits, ranks

    def count_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ═══════════════════════════════════════════════════════════════════════════
# DATA + EVAL
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
    avg_loss = total_loss / max(total_tokens, 1); avg_bpb = avg_loss / math.log(2)
    avg_ranks = {}
    if all_ranks:
        for k in all_ranks[0]: avg_ranks[k] = sum(r[k] for r in all_ranks) / len(all_ranks)
    model.train()
    return avg_bpb, avg_ranks

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_config(cfg, device, local_rank, world_size, is_main, train_ds, val_ds):
    results_dir = Path(cfg["results_dir"])
    if is_main: results_dir.mkdir(parents=True, exist_ok=True)

    class Logger:
        def __init__(self, path): self.f = open(path, 'w') if is_main else None
        def log(self, msg):
            if not is_main: return
            line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"; print(line, flush=True)
            if self.f: self.f.write(line + '\n'); self.f.flush()
        def close(self):
            if self.f: self.f.close()

    logger = Logger(results_dir / "train.log")
    logger.log(f"Config: {cfg['name']} | GPUs: {world_size}")

    raw_model = SweepModel(
        mat_dim=cfg["mat_dim"], n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
        insert_layer=cfg["insert_layer"], n_thoughts=cfg["n_thoughts"],
        output_type=cfg["output_type"], max_len=4096, vocab_size=256, dropout=0.1
    ).to(device)
    logger.log(f"Params: {raw_model.count_params():,} | N={cfg['n_thoughts']} thoughts | output={cfg['output_type']}")

    model = DDP(raw_model, device_ids=[local_rank], find_unused_parameters=True)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler, drop_last=True, num_workers=8, pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.98))
    def lr_lambda(step):
        if step < 300: return step / 300
        return 0.5 * (1 + math.cos(math.pi * (step - 300) / max(cfg["max_steps"] - 300, 1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step, best_val_loss, start_time = 0, float("inf"), time.time()
    data_iter, epoch = iter(train_loader), 0
    model.train()
    logger.log("--- Training ---")

    while step < cfg["max_steps"]:
        try: x, y = next(data_iter)
        except StopIteration: epoch += 1; sampler.set_epoch(epoch); data_iter = iter(train_loader); x, y = next(data_iter)
        x, y = x.to(device), y.to(device); optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, ranks = model(x, use_thoughts=cfg["n_thoughts"] > 0)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step(); step += 1
        if step % 50 == 0:
            elapsed = time.time() - start_time; bpb = loss.item() / math.log(2)
            tps = step * cfg["batch_size"] * world_size * 512 / elapsed
            rank_str = " | ".join(f"{k}={v:.2f}" for k,v in ranks.items()) if ranks else "-"
            logger.log(f"Step {step:5d} | BPB {bpb:.3f} | {rank_str} | {tps:.0f} tok/s | {elapsed:.0f}s")
        if step % 500 == 0 and is_main:
            bpb_t, ranks_t = evaluate(raw_model, val_ds, device, use_thoughts=cfg["n_thoughts"] > 0)
            bpb_no, _ = evaluate(raw_model, val_ds, device, use_thoughts=False) if cfg["n_thoughts"] > 0 else (bpb_t, {})
            rank_str = " | ".join(f"{k}={v:.2f}" for k,v in ranks_t.items())
            logger.log(f"  Eval: BPB {bpb_t:.3f} | {rank_str} | no-thought: {bpb_no:.3f}")
            if bpb_t < best_val_loss: best_val_loss = bpb_t; torch.save(raw_model.state_dict(), results_dir / "best.pt")
            dist.barrier()
        elif step % 500 == 0:
            dist.barrier()

    elapsed = time.time() - start_time
    if is_main:
        raw_model.load_state_dict(torch.load(results_dir / "best.pt", weights_only=True))
        bpb_t, ranks_t = evaluate(raw_model, val_ds, device, use_thoughts=cfg["n_thoughts"] > 0)
        bpb_no, _ = evaluate(raw_model, val_ds, device, use_thoughts=False) if cfg["n_thoughts"] > 0 else (bpb_t, {})
        summary = f"""
{'='*70}
  {cfg['name']}
{'='*70}
  Params: {raw_model.count_params():,} | {elapsed/60:.1f} min
  N={cfg['n_thoughts']} thoughts | output={cfg['output_type']} | {cfg['n_layers']} layers
  BPB: {bpb_t:.3f} | no-thought: {bpb_no:.3f}
  Ranks: {ranks_t}
  Benefit: {(bpb_no - bpb_t) / bpb_no * 100:.1f}%
{'='*70}
"""
        logger.log(summary)
        with open(results_dir / "SUMMARY.txt", 'w') as f: f.write(summary)
        json.dump({"name": cfg["name"], "params": raw_model.count_params(), "bpb": bpb_t,
                    "no_thought_bpb": bpb_no, "ranks": ranks_t, "time_min": elapsed/60, "config": cfg},
                   open(results_dir / "results.json", 'w'), indent=2, default=float)
    logger.close()
    # Clean up model for next config
    del model, raw_model, optimizer, scheduler
    torch.cuda.empty_cache()

def run():
    dist.init_process_group("nccl", timeout=dt_module.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"]); world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank); device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    if is_main: shutil.copy2(Path(__file__).resolve(), Path("/toy_story_slam/results/sweep_script.py"))

    data_dir = "/toy_story_slam/data_bytes"
    train_ds = TokenDataset(os.path.join(data_dir, "train.pt"), 512)
    val_ds = TokenDataset(os.path.join(data_dir, "val.pt"), 512)

    configs = [
        {"name": "A_multiprobe_N1", "n_layers": 24, "n_thoughts": 1, "output_type": "multiprobe",
         "insert_layer": 8, "mat_dim": 16, "n_heads": 4, "batch_size": 128, "max_steps": 3000,
         "results_dir": "/toy_story_slam/results/sweep_A_multiprobe_N1"},

        {"name": "B_zeroparam_N2", "n_layers": 24, "n_thoughts": 2, "output_type": "zero_param",
         "insert_layer": 8, "mat_dim": 16, "n_heads": 4, "batch_size": 96, "max_steps": 3000,
         "results_dir": "/toy_story_slam/results/sweep_B_zeroparam_N2"},

        {"name": "C_zeroparam_N4", "n_layers": 24, "n_thoughts": 4, "output_type": "zero_param",
         "insert_layer": 8, "mat_dim": 16, "n_heads": 4, "batch_size": 64, "max_steps": 3000,
         "results_dir": "/toy_story_slam/results/sweep_C_zeroparam_N4"},

        {"name": "D_scaled_N1", "n_layers": 48, "n_thoughts": 1, "output_type": "zero_param",
         "insert_layer": 16, "mat_dim": 16, "n_heads": 4, "batch_size": 96, "max_steps": 3000,
         "results_dir": "/toy_story_slam/results/sweep_D_scaled_N1"},

        {"name": "E_scaled_baseline", "n_layers": 48, "n_thoughts": 0, "output_type": "zero_param",
         "insert_layer": 16, "mat_dim": 16, "n_heads": 4, "batch_size": 128, "max_steps": 3000,
         "results_dir": "/toy_story_slam/results/sweep_E_scaled_baseline"},
    ]

    for i, cfg in enumerate(configs):
        dist.barrier()
        if is_main: print(f"\n{'#'*70}\n  SWEEP {i+1}/{len(configs)}: {cfg['name']}\n{'#'*70}\n", flush=True)
        train_config(cfg, device, local_rank, world_size, is_main, train_ds, val_ds)

    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__": run()
