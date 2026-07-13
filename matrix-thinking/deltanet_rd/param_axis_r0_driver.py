#!/usr/bin/env python3
"""R0 driver: runs the AR-hit ablation-gap instrument across the 0.328B-token
common-token-count slice for all 4 rungs (openr1-mix-ext) and the 3-rung
supplementary cross-check (wikitext-mix-ext, no 1.31B wikitext checkpoint
exists). Also re-runs the T3 span_frac reproduction check via the existing
lm_attractor_probe_rd.py, unmodified. Writes one aggregated JSON.
"""
import json
import os
import sys
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/nvidia/chapter2/deltanet_rd")

import torch  # noqa: E402
from lm_recall_gap_probe_rd import (  # noqa: E402
    load_checkpoint, build_bigram_mode_table, run_ar_hit_gap_eval,
    run_t2_positive_control, resolve_token_matched_checkpoint, VOCAB_SIZE,
)
from lm_pretrain_rd import load_corpus, corpus_fixed_seed, DEFAULT_DATA_DIR  # noqa: E402

TARGET_TOKENS = 327_680_000  # 14M's own final checkpoint (20000*32*512)
DEVICE = "cuda"
N_WINDOWS = 1024

SPECS = [
    ("14M", "openr1-mix-ext", "/data/deltanet_rd_frozenbias_ckpts/frozenbias_lm_per_token_lam0p58_openr1-mix-ext_dm256_ds64_L2_s0", 32, "lmC_openr1-mix-ext_dm256_ds64_L2_s0"),
    ("14M", "wikitext-mix-ext", "/data/deltanet_rd_frozenbias_ckpts/frozenbias_lm_per_token_lam0p58_wikitext-mix-ext_dm256_ds64_L2_s0", 32, "lmC_wikitext-mix-ext_dm256_ds64_L2_s0"),
    ("98M", "openr1-mix-ext", "/data/fixscale_ckpts/train/fixscale_train_arm_per_token_98m_openr1-mix-ext_s0", 32, "lmC_openr1-mix-ext_dm768_ds64_L12_s0"),
    ("98M", "wikitext-mix-ext", "/data/fixscale_ckpts/train/fixscale_train_arm_per_token_98m_wikitext-mix-ext_s0", 32, "lmC_wikitext-mix-ext_dm768_ds64_L12_s0"),
    ("392M", "openr1-mix-ext", "/data/fixscale_ckpts/train/fixscale_train_arm_per_token_392m_openr1-mix-ext_s0", 32, "lmC_openr1-mix-ext_dm1536_ds128_L16_s0"),
    ("392M", "wikitext-mix-ext", "/data/fixscale_ckpts/train/fixscale_train_arm_per_token_392m_wikitext-mix-ext_s0", 32, "lmC_wikitext-mix-ext_dm1536_ds128_L16_s0"),
    ("1.31B", "openr1-mix-ext", "/data/queue_1p31b_ckpts/queue_1p31b_arm_per_token_openr1-mix-ext_s0", 16, "lmC_openr1-mix-ext_dm2560_ds128_L22_s0"),
]

results = {"target_tokens": TARGET_TOKENS, "n_windows": N_WINDOWS, "cells": []}
corpus_cache = {}

for rung, corpus, ckpt_dir, batch_size, prefix in SPECS:
    t_cell0 = time.time()
    slice_info = resolve_token_matched_checkpoint(ckpt_dir, batch_size, 512, TARGET_TOKENS, prefix)
    assert slice_info["miss_tokens"] == 0, f"{rung}/{corpus}: slice miss {slice_info}"
    path = slice_info["path"]
    print(f"=== {rung} / {corpus} : step {slice_info['chosen_step']} ===", flush=True)

    model, ckpt = load_checkpoint(path, DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    if corpus not in corpus_cache:
        train_tokens, val_tokens, meta, _, _ = load_corpus(DEFAULT_DATA_DIR, corpus, DEVICE)
        mode_next = build_bigram_mode_table(train_tokens, VOCAB_SIZE, DEVICE)
        corpus_cache[corpus] = (train_tokens, val_tokens, mode_next)
    train_tokens, val_tokens, mode_next = corpus_cache[corpus]

    seed = corpus_fixed_seed(corpus) + 424242
    real = run_ar_hit_gap_eval(model, val_tokens, batch_size, 512, N_WINDOWS, DEVICE,
                                mode_next, seed, min_sep=2, shuffle_positions=False)
    shuf = run_ar_hit_gap_eval(model, val_tokens, batch_size, 512, N_WINDOWS, DEVICE,
                                mode_next, seed, min_sep=2, shuffle_positions=True)
    t2 = run_t2_positive_control(model, val_tokens, train_tokens, batch_size, 512, DEVICE,
                                  seed, n_batches=max(2, N_WINDOWS // batch_size // 8),
                                  j0=50, k0=70)

    cell = {
        "rung": rung, "corpus": corpus, "n_params": n_params,
        "ckpt_step": ckpt.get("step"), "ckpt_seed": ckpt.get("seed"),
        "frozen_bias_arm": ckpt["config"].get("frozen_bias_arm"),
        "path": path, "slice_info": slice_info,
        "real": real, "t1_shuffled_control": shuf, "t2_positive_control": t2,
        "t1_pass": (real["gap"] is not None and shuf["gap"] is not None
                    and abs(shuf["gap"]) < 0.10
                    and (shuf["n_candidates"] < 0.25 * real["n_candidates"]
                         or (shuf["acc_intact"] is not None and shuf["acc_baseline_nonAR"] is not None
                             and abs(shuf["acc_intact"] - shuf["acc_baseline_nonAR"]) < 0.10))),
        "t2_pass": (t2["acc_intact"] > 100.0 / VOCAB_SIZE) and (t2["acc_ablated"] < t2["acc_intact"]),
        "elapsed_s": time.time() - t_cell0,
    }
    results["cells"].append(cell)
    print(json.dumps({k: v for k, v in cell.items() if k not in ("slice_info",)}, indent=2), flush=True)

    del model, ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with open("/tmp/r0_ar_hit_full.json", "w") as f:
    json.dump(results, f, indent=2)
print("WROTE /tmp/r0_ar_hit_full.json")
