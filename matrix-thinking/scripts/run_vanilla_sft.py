#!/usr/bin/env python3
"""Round 4 vanilla SFT for ProsQA — reconstruction (saves checkpoints).

The original `mode=pure_sft` script lived only on the dead pod and never saved
checkpoints (CONTROL_A_RESEARCH_REPORT.md predicted this). This rebuild matches
Round 4's logged config: GPT-2 small + 3 special tokens (resized to vocab 50260),
AdamW(lr=1e-4, betas=0.9/0.98, wd=0.01), batch=16, 25 epochs, cosine warmup=50.

Prompt format (matches matrix-CODI's answer_prefix=" The answer is:"):
    "{question}\n The answer is: {answer}<|endoftext|>"

Labels: -100 on "{question}\n" prefix; loss is computed on
" The answer is: {answer}<|endoftext|>". This trains the model to also produce
the answer prefix, which makes greedy decoding stop reliably.
"""
import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, GPT2LMHeadModel, get_cosine_schedule_with_warmup

CFG = {
    "base_model": "gpt2",
    "lr": 1e-4,
    "warmup_steps": 50,
    "weight_decay": 0.01,
    "betas": (0.9, 0.98),
    "grad_clip": 1.0,
    "batch_size": 16,
    "epochs": 25,
    "max_q_len": 640,
    "max_ans_len": 32,
    "max_total_len": 768,
    "eval_batch_size": 16,
    "log_interval": 50,
    "answer_prefix": " The answer is:",
}


class ProsQAVanillaDataset(Dataset):
    """Vanilla SFT: question -> " The answer is: <full answer sentence><|endoftext|>"."""

    def __init__(self, split, path, tokenizer, cfg):
        with open(path) as f:
            rows = json.load(f)
        self.items = []
        self.tokenizer = tokenizer
        eos = tokenizer.eos_token_id
        prefix = cfg["answer_prefix"]
        n_total = len(rows)
        n_dropped = 0
        for row in rows:
            q = str(row["question"]).strip()
            a = str(row["answer"]).strip()
            q_text = q + "\n"
            tail_text = f"{prefix} {a}"
            q_ids = tokenizer.encode(q_text, add_special_tokens=False)
            tail_ids = tokenizer.encode(tail_text, add_special_tokens=False) + [eos]
            if len(q_ids) > cfg["max_q_len"]:
                q_ids = q_ids[-cfg["max_q_len"]:]
            if len(tail_ids) > cfg["max_ans_len"] + 16:
                n_dropped += 1
                continue
            input_ids = q_ids + tail_ids
            if len(input_ids) > cfg["max_total_len"]:
                n_dropped += 1
                continue
            labels = [-100] * len(q_ids) + tail_ids
            self.items.append({
                "input_ids": input_ids,
                "labels": labels,
                "q_ids": q_ids,
                "answer_text": a,
                "question": q,
            })
        print(
            f"[ProsQAVanillaDataset {split} pure_sft] kept {len(self.items)}/{n_total} "
            f"(dropped len={n_dropped})",
            flush=True,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = L - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def first_sentence_final_class(s):
    """Mirror prosqa_answer_match's first_sentence_final_class logic from
    matrix-thinking/scripts/run_matrix_codi.py."""
    s = s.strip()
    for sep in [".", "?", "!", "\n"]:
        i = s.find(sep)
        if i >= 0:
            s = s[:i]
            break
    s = s.strip().lower()
    toks = re.split(r"\s+", s)
    return toks[-1] if toks and toks[-1] else ""


def prosqa_answer_match(pred_text, gold_text):
    return first_sentence_final_class(pred_text) == first_sentence_final_class(gold_text)


@torch.no_grad()
def evaluate(model, val_ds, tokenizer, cfg, device, max_examples=None):
    model.eval()
    correct = 0
    total = 0
    n = len(val_ds) if max_examples is None else min(len(val_ds), max_examples)
    for i in range(n):
        ex = val_ds[i]
        # Generate from question + " The answer is:" so the model only fills in
        # the entity. This matches Control A's eval-time prompt.
        prompt_ids = ex["q_ids"] + tokenizer.encode(
            cfg["answer_prefix"], add_special_tokens=False
        )
        input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg["max_ans_len"],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0, len(prompt_ids):]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if prosqa_answer_match(gen_text, ex["answer_text"]):
            correct += 1
        total += 1
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--train-path", default="/workspace/pebble/round3_gamma0/data/prosqa_train.json"
    )
    ap.add_argument(
        "--val-path", default="/workspace/pebble/round3_gamma0/data/prosqa_test.json"
    )
    ap.add_argument("--output-dir", default="/workspace/pebble/round4_vanilla_sft")
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--batch-size", type=int, default=CFG["batch_size"])
    ap.add_argument(
        "--eval-every-epoch",
        action="store_true",
        default=True,
        help="Eval at end of every epoch (matches Round 4 logs).",
    )
    ap.add_argument("--smoke-test", action="store_true",
                    help="1 epoch on 32 train / 8 val examples; smaller model warmup.")
    args = ap.parse_args()

    cfg = dict(CFG)
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["prosqa_train_path"] = args.train_path
    cfg["prosqa_val_path"] = args.val_path

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"pure_sft_seed{seed}.log"
    log_f = open(log_path, "a", buffering=1)

    def log(msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<bot>", "<eot>", "<latent>"]}
    )

    log(f"=== Vanilla SFT (mode=pure_sft, seed={seed}) ===")
    model = GPT2LMHeadModel.from_pretrained(cfg["base_model"])
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    log(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    log(f"Tokenizer len: {len(tokenizer)}")

    train_ds = ProsQAVanillaDataset("train", cfg["prosqa_train_path"], tokenizer, cfg)
    val_ds = ProsQAVanillaDataset("val", cfg["prosqa_val_path"], tokenizer, cfg)

    if args.smoke_test:
        train_ds.items = train_ds.items[:32]
        val_ds.items = val_ds.items[:8]
        cfg["epochs"] = 1
    log(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=2,
        drop_last=False,
    )

    steps_per_epoch = (len(train_ds) + cfg["batch_size"] - 1) // cfg["batch_size"]
    total_steps = steps_per_epoch * cfg["epochs"]
    log(f"Total steps: {total_steps} (epochs {cfg['epochs']} x steps/epoch {steps_per_epoch})")

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=cfg["betas"],
        weight_decay=cfg["weight_decay"],
    )
    sched = get_cosine_schedule_with_warmup(optim, cfg["warmup_steps"], total_steps)

    best_acc = 0.0
    final_acc = 0.0
    training_curve = []
    t0 = time.time()
    step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for batch in train_loader:
            step += 1
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attn = batch["attention_mask"].to(device)
            out_t = model(input_ids=input_ids, labels=labels, attention_mask=attn)
            loss = out_t.loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optim.step()
            sched.step()
            if step % cfg["log_interval"] == 0:
                log(
                    f"step {step}/{total_steps} | loss {loss.item():.4f} | "
                    f"lr {sched.get_last_lr()[0]:.2e} | {int(time.time() - t0)}s"
                )
        max_eval = 64 if args.smoke_test else None
        acc = evaluate(model, val_ds, tokenizer, cfg, device, max_examples=max_eval)
        training_curve.append({"epoch": epoch, "step": step, "accuracy": acc})
        log(f"Epoch {epoch}: prosqa accuracy {acc * 100:.2f}%")
        final_acc = acc
        if acc > best_acc:
            best_acc = acc
            ckpt = {
                "model": model.state_dict(),
                "cfg": cfg,
                "seed": seed,
                "tokenizer_len": len(tokenizer),
                "epoch": epoch,
                "step": step,
                "accuracy": acc,
            }
            torch.save(ckpt, out / f"pure_sft_seed{seed}.pt")
            log(f"  -> saved checkpoint @ {acc * 100:.2f}%")

    wall_min = (time.time() - t0) / 60.0
    results = {
        "mode": "pure_sft",
        "seed": seed,
        "best_accuracy": best_acc,
        "final_accuracy": final_acc,
        "training_curve": training_curve,
        "wall_time_min": wall_min,
        "config": cfg,
    }
    with open(out / f"pure_sft_seed{seed}_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(out / f"pure_sft_seed{seed}_SUMMARY.txt", "w") as f:
        f.write(
            f"seed={seed} best={best_acc * 100:.2f}% final={final_acc * 100:.2f}% "
            f"wall={wall_min:.1f}min\n"
        )
    log(f"DONE seed={seed} best={best_acc * 100:.2f}% wall={wall_min:.1f}min")
    log_f.close()


if __name__ == "__main__":
    main()
