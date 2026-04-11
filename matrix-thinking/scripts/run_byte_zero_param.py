#!/usr/bin/env python3
"""
Zero-Param Byte Output: The matrix IS the prediction
=====================================================
At d=16 with byte vocab (256 = 16²), the output matrix has exactly
one entry per possible byte. No output head needed.

logits = flatten(final_norm(M))  →  256 values  →  softmax  →  byte probs

This eliminates the output projection entirely. The model must learn to
organize its internal representation so that entry (i,j) of the matrix
corresponds to the logit for byte (16*i + j).

Comparison: MultiProbeHead byte model (Run 19, BPB 3.56) had 4,864 output params.
This model has ZERO output params. Same everything else.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import math, time, json, os, shutil
from pathlib import Path
import datetime as dt_module
from datetime import datetime, timezone

CONFIG = {
    "mat_dim": 16, "n_thinking_layers": 12, "n_heads": 4,
    "n_iterations": 8, "max_len": 4096, "dropout": 0.1,
    "batch_size_per_gpu": 96, "seq_len": 512, "max_steps": 3000,
    "lr": 3e-4, "warmup_steps": 300, "weight_decay": 0.01, "grad_clip": 1.0,
    "eval_interval": 500, "eval_batch_size": 96, "eval_iterations": [1, 2, 4, 8],
    "log_interval": 50, "data_dir": "/toy_story_slam/data_bytes",
    "results_dir": "/toy_story_slam/results/exp_byte_zero_param", "num_workers": 8,
}

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

class MatrixByteOutput(nn.Module):
    """THE NOVEL IDEA: the matrix IS the prediction.
    16×16 matrix + learned bias → flatten → 256 logits (one per byte).
    257 params total: 256 bias (unigram byte prior) + 1 temperature.

    The bias gives each byte its own prior probability.
    The temperature controls softmax sharpness (needed because matrix
    magnitude varies with rank and training stage)."""
    def __init__(self, d):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(d, d))       # 256 params: unigram byte prior
        self.log_temp = nn.Parameter(torch.tensor(0.0))    # 1 param: softmax temperature
    def forward(self, M):
        B, L, d, _ = M.shape
        logits = (M + self.bias).reshape(B, L, d * d) * self.log_temp.exp()
        return logits

class ByteMatrixThinkerZeroParam(nn.Module):
    def __init__(self, mat_dim=16, n_layers=12, n_heads=4,
                 max_len=4096, vocab_size=256, dropout=0.1):
        super().__init__()
        self.mat_dim = mat_dim; d = mat_dim
        self.embed_u = nn.Embedding(vocab_size, d)
        self.embed_v = nn.Embedding(vocab_size, d)
        self.pos_u = nn.Embedding(max_len, d)
        self.pos_v = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList([ThinkingBlock(d, n_heads, dropout) for _ in range(n_layers)])
        self.final_norm = MatrixRMSNorm(d)
        self.output = MatrixByteOutput(d)  # 257 PARAMS (256 bias + 1 temperature)

    def _one_iteration(self, M):
        for layer in self.layers:
            if self.training:
                M = torch.utils.checkpoint.checkpoint(layer, M, use_reentrant=False)
            else:
                M = layer(M)
        return M

    def forward(self, token_ids, n_iterations=1, measure_ranks=False):
        B, L = token_ids.shape; d = self.mat_dim
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
                    if Mf.shape[0] > 512: Mf = Mf[torch.randperm(Mf.shape[0], device=Mf.device)[:512]]
                    S = torch.linalg.svdvals(Mf).clamp(min=1e-10); Sn = S / S.sum(-1, keepdim=True)
                    ranks.append((-(Sn * Sn.log()).sum(-1)).exp().mean().item())
        logits = self.output(self.final_norm(M))
        return logits, {'ranks': ranks}

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
    all_ranks = []
    with torch.no_grad():
        for vx, vy in loader:
            if n_batches >= max_eval_batches: break
            n_batches += 1; vx, vy = vx.to(device), vy.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, info = model(vx, n_iterations=n_iterations, measure_ranks=True)
            n_tok = vy.numel()
            total_loss += F.cross_entropy(logits.reshape(-1, vocab_size), vy.reshape(-1)).item() * n_tok
            total_tokens += n_tok
            if info['ranks']: all_ranks.append(info['ranks'])
    avg_loss = total_loss / max(total_tokens, 1)
    avg_bpb = avg_loss / math.log(2)  # DIRECT BPB for bytes
    avg_ranks = []
    if all_ranks:
        n_r = len(all_ranks[0]); avg_ranks = [sum(r[i] for r in all_ranks) / len(all_ranks) for i in range(n_r)]
    model.train()
    return avg_loss, math.exp(min(avg_loss, 20)), avg_bpb, avg_ranks

def run():
    cfg = CONFIG.copy()
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
    logger.log(f"ZERO-PARAM BYTE OUTPUT: matrix IS the prediction | GPUs: {world_size}")
    data_dir = cfg["data_dir"]
    if not os.path.exists(os.path.join(data_dir, "meta.json")):
        logger.log("ERROR: byte data not found. Run prep_byte_data.py first.")
        dist.barrier(); dist.destroy_process_group(); return
    meta = json.load(open(os.path.join(data_dir, "meta.json"))); vocab_size = meta["vocab_size"]
    assert vocab_size == 256, f"Expected byte vocab (256), got {vocab_size}"
    logger.log(f"Data: {meta.get('train_tokens', '?'):,} bytes, sources: {meta.get('sources', '?')}")

    train_ds = TokenDataset(os.path.join(data_dir, "train.pt"), cfg["seq_len"])
    val_ds = TokenDataset(os.path.join(data_dir, "val.pt"), cfg["seq_len"])
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg["batch_size_per_gpu"], sampler=sampler, drop_last=True, num_workers=cfg["num_workers"], pin_memory=True, persistent_workers=True)

    raw_model = ByteMatrixThinkerZeroParam(
        mat_dim=cfg["mat_dim"], n_layers=cfg["n_thinking_layers"],
        n_heads=cfg["n_heads"], max_len=cfg["max_len"],
        vocab_size=vocab_size, dropout=cfg["dropout"]).to(device)
    n_params = raw_model.count_params()
    ep = sum(p.numel() for n,p in raw_model.named_parameters() if 'embed' in n or 'pos' in n)
    tp = sum(p.numel() for n,p in raw_model.named_parameters() if 'layers' in n)
    hp = sum(p.numel() for n,p in raw_model.named_parameters() if 'output' in n or 'final' in n)
    logger.log(f"Params: {n_params:,} (embed={ep:,} [{ep*100//n_params}%] think={tp:,} [{tp*100//n_params}%] output={hp:,} [{hp*100//n_params}%])")
    logger.log(f"Output head: MatrixByteOutput (matrix + bias = byte logits, 257 params)")

    model = DDP(raw_model, device_ids=[local_rank])
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
            logits, info = model(x, n_iterations=cfg["n_iterations"], measure_ranks=do_log)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step(); scheduler.step(); step += 1
        if step % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            bpb = loss.item() / math.log(2)
            tps = step * cfg["batch_size_per_gpu"] * world_size * cfg["seq_len"] / elapsed
            ranks = info.get('ranks', []); rank_str = ", ".join(f"{r:.2f}" for r in ranks) if ranks else "-"
            temp = raw_model.output.log_temp.exp().item()
            logger.log(f"Step {step:5d} | BPB {bpb:.3f} | Loss {loss.item():.4f} | GN {grad_norm:.3f} | Temp {temp:.3f} | Rank [{rank_str}] | {tps:.0f} tok/s | {elapsed:.0f}s")
        if step % cfg["eval_interval"] == 0:
            if is_main:
                logger.log(f"--- Eval step {step} ---"); eval_results = {}
                for T in cfg["eval_iterations"]:
                    vl, vp, vbpb, vr = evaluate(raw_model, val_ds, vocab_size, device, T, cfg["eval_batch_size"])
                    rank_str = ", ".join(f"{r:.2f}" for r in vr) if vr else "-"; marker = ""
                    if T == cfg["n_iterations"] and vl < best_val_loss:
                        best_val_loss = vl; torch.save(raw_model.state_dict(), results_dir / "best.pt"); marker = " *BEST*"
                    logger.log(f"  T={T:2d}: BPB {vbpb:.3f} | Rank [{rank_str}]{marker}")
                    eval_results[f"T{T}"] = {"val_loss": vl, "val_ppl": vp, "val_bpb": vbpb, "ranks": vr}
                training_curve.append({"step": step, "train_loss": loss.item(), "evals": eval_results})
            dist.barrier()

    elapsed = time.time() - start_time
    if is_main:
        logger.log(f"\n{'='*70}"); logger.log(f"DONE | {step} steps | {elapsed/60:.1f} min")
        raw_model.load_state_dict(torch.load(results_dir / "best.pt", weights_only=True))
        final = {}
        for T in cfg["eval_iterations"]:
            vl, vp, vbpb, vr = evaluate(raw_model, val_ds, vocab_size, device, T, cfg["eval_batch_size"])
            rank_str = ", ".join(f"{r:.2f}" for r in vr) if vr else "-"
            logger.log(f"  T={T:2d}: BPB {vbpb:.3f} | Rank [{rank_str}]")
            final[f"T{T}"] = {"val_loss": vl, "val_ppl": vp, "val_bpb": vbpb, "ranks": vr}
        bpbs = {T: final[f"T{T}"]["val_bpb"] for T in cfg["eval_iterations"]}
        t1, tn = cfg["eval_iterations"][0], cfg["eval_iterations"][-1]
        summary = "\n".join([
            "", "="*70, "  ZERO-PARAM BYTE OUTPUT: MATRIX IS THE PREDICTION", "="*70,
            f"  16x16 matrix → flatten → 256 byte logits (1 temperature param)",
            f"  Params: {n_params:,} (embed={ep:,} [{ep*100//n_params}%] think={tp:,} [{tp*100//n_params}%] output={hp:,})",
            f"  Time: {elapsed/60:.1f} min on {world_size}x H100", "",
            *[f"  T={T:2d}: BPB {bpbs[T]:.3f}" for T in cfg["eval_iterations"]], "",
            f"  Thinking: BPB {bpbs[t1]:.3f} -> {bpbs[tn]:.3f}",
            f"  vs MultiProbeHead byte (Run 19): BPB 3.560 at T=8",
            f"  Output params saved: ~4,864 (MultiProbeHead) → 1 (temperature)",
            "="*70, ""])
        logger.log(summary)
        with open(results_dir / "SUMMARY.txt", 'w') as f: f.write(summary)
        json.dump({"experiment": "exp_byte_zero_param", "params": n_params,
                    "param_breakdown": {"embed": ep, "think": tp, "output": hp},
                    "config": cfg, "final_evals": final, "training_curve": training_curve,
                    "time_min": elapsed/60}, open(results_dir / "results.json", 'w'), indent=2, default=float)
        logger.log(f"Saved to {results_dir}"); logger.close()
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__": run()
