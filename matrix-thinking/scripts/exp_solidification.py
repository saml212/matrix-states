"""
THE SOLIDIFICATION EXPERIMENT.

Question: When trained on reasoning data, do matrix thoughts crystallize?
Does the rank of generated thought matrices decrease over iterations,
indicating that abstract high-dimensional thinking converges into
a concrete, expressible answer?

This experiment trains the autoregressive matrix thinker on math reasoning
data (OpenR1-Math) and measures:
1. Rank profile over thinking iterations (does it decrease?)
2. Whether harder problems cause more thinking steps
3. Whether the halting distribution shifts during training
4. The shape of the rank curve: does it peak then fall? Monotone decrease?
"""

import sys
import os
import math
import json
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import TrainConfig, get_dataloaders, get_device
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from matrix_thinker import AutoregressiveMatrixThinker


def train_and_measure_solidification(model, config):
    device = get_device(config)
    model = model.to(device)

    meta = json.load(open(os.path.join(config.data_dir, "meta.json")))
    vocab_size = meta["vocab_size"]
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f" SOLIDIFICATION EXPERIMENT")
    print(f" {n_params:,} params | {device} | max_thoughts={model.max_thoughts}")
    print(f" Data: {meta.get('source', 'unknown')}")
    print(f"{'='*60}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                   weight_decay=config.weight_decay, betas=(0.9, 0.98))

    def lr_lambda(step):
        if step < config.warmup_steps: return step / config.warmup_steps
        progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    train_loader, val_loader = get_dataloaders(config)
    save_dir = os.path.join(config.save_dir, config.experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    step = 0
    best_val_loss = float("inf")
    start_time = time.time()
    solidification_log = []
    data_iter = iter(train_loader)

    model.train()
    while step < config.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits, info = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1

        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            ppl = math.exp(min(loss.item(), 20))
            ranks = info['mean_ranks_per_step']
            # Does rank decrease? Measure the slope
            if len(ranks) >= 2:
                rank_slope = (ranks[-1] - ranks[0]) / len(ranks)
                solidifying = "YES" if rank_slope < -0.05 else "no"
            else:
                rank_slope = 0
                solidifying = "?"

            print(f"[{elapsed:7.0f}s] Step {step:5d} | Loss {loss.item():.4f} | PPL {ppl:.1f} "
                  f"| Think {info['expected_steps']:.1f} steps "
                  f"| Ranks {[f'{r:.2f}' for r in ranks]} "
                  f"| Solidifying: {solidifying}")

        if step % config.eval_interval == 0:
            model.eval()
            val_loss = 0; val_n = 0; val_ranks = []; val_steps_list = []
            with torch.no_grad():
                for vx, vy in val_loader:
                    if val_n >= config.eval_batches: break
                    vx, vy = vx.to(device), vy.to(device)
                    vlogits, vinfo = model(vx)
                    val_loss += F.cross_entropy(vlogits.reshape(-1, vocab_size), vy.reshape(-1)).item()
                    val_ranks.append(vinfo['mean_ranks_per_step'])
                    val_steps_list.append(vinfo['expected_steps'])
                    val_n += 1

            val_loss /= max(val_n, 1)
            val_ppl = math.exp(min(val_loss, 20))

            # Average rank profile across validation batches
            avg_ranks = [sum(r[i] for r in val_ranks) / len(val_ranks)
                         for i in range(len(val_ranks[0]))]
            avg_steps = sum(val_steps_list) / len(val_steps_list)

            marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(save_dir, "best.pt"))
                marker = " *"

            solidification_entry = {
                "step": step,
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "avg_rank_profile": avg_ranks,
                "avg_thinking_steps": avg_steps,
                "halt_distribution": info['halt_distribution'],
                "rank_decreasing": avg_ranks[-1] < avg_ranks[0] if len(avg_ranks) >= 2 else None,
                "rank_drop": avg_ranks[0] - avg_ranks[-1] if len(avg_ranks) >= 2 else 0,
            }
            solidification_log.append(solidification_entry)

            print(f"\n  Val Loss {val_loss:.4f} | Val PPL {val_ppl:.1f}{marker}")
            print(f"  Avg thinking steps: {avg_steps:.1f}")
            print(f"  Rank profile: {[f'{r:.2f}' for r in avg_ranks]}")
            print(f"  Rank drop (first→last): {solidification_entry['rank_drop']:.3f}")
            print(f"  Thoughts solidifying: {'YES' if solidification_entry['rank_decreasing'] else 'NO'}")
            print()

            model.train()

    # Save results
    elapsed = time.time() - start_time
    results = {
        "experiment": config.experiment_name,
        "params": n_params,
        "best_val_loss": best_val_loss,
        "best_val_ppl": math.exp(min(best_val_loss, 20)),
        "training_time_s": elapsed,
        "solidification_log": solidification_log,
        "final_verdict": {
            "thoughts_solidify": solidification_log[-1]["rank_decreasing"] if solidification_log else None,
            "rank_drop": solidification_log[-1]["rank_drop"] if solidification_log else 0,
            "avg_thinking_steps": solidification_log[-1]["avg_thinking_steps"] if solidification_log else 0,
        }
    }

    with open(os.path.join(save_dir, "solidification_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n{'='*60}")
    print(f" SOLIDIFICATION VERDICT")
    print(f"{'='*60}")
    if solidification_log:
        last = solidification_log[-1]
        print(f" Thoughts solidify: {'YES' if last['rank_decreasing'] else 'NO'}")
        print(f" Rank drop: {last['rank_drop']:.3f}")
        print(f" Avg thinking steps: {last['avg_thinking_steps']:.1f}")
        print(f" Rank profile: {[f'{r:.2f}' for r in last['avg_rank_profile']]}")
    print(f" Saved to: {save_dir}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--max-thoughts", type=int, default=200)
    args = parser.parse_args()

    if args.mini:
        config = TrainConfig(
            data_dir="/Volumes/1TB_SSD/learned-representations/data/reasoning",
            seq_len=128, batch_size=4, max_steps=500, lr=1e-3, warmup_steps=50,
            dropout=0.1, eval_interval=100, log_interval=25,
            save_dir="/Volumes/1TB_SSD/learned-representations/results/h100_test",
            experiment_name="solidification_mini",
            device="mps", compile_model=False, use_bfloat16=False,
        )
        model = AutoregressiveMatrixThinker(
            mat_dim=8, n_thinking_layers=1,
            max_thoughts=min(args.max_thoughts, 6),  # cap for Mac Mini speed
            n_heads=2, max_len=256, vocab_size=50257, dropout=0.1
        )
    else:
        config = TrainConfig(
            data_dir="/data/reasoning",
            seq_len=512, batch_size=16, max_steps=20000, lr=3e-4, warmup_steps=500,
            dropout=0.1, eval_interval=500, log_interval=50,
            save_dir="/results",
            experiment_name="solidification",
            device="cuda", compile_model=False, use_bfloat16=True,
        )
        model = AutoregressiveMatrixThinker(
            mat_dim=16, n_thinking_layers=2,
            max_thoughts=args.max_thoughts,
            n_heads=4, max_len=1024, vocab_size=50257, dropout=0.1
        )

    print(f"Model: {model.count_parameters():,} params")
    print(f"Max thoughts: {model.max_thoughts}")
    train_and_measure_solidification(model, config)
