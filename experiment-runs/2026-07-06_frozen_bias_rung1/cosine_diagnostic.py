"""sec 1.3's cheap diagnostic for a sim-training-divergence finding: cosine similarity of each
trained token's raw key (k_raw, pre-blend) against that token's own frozen anchor row B[token_id],
BEFORE training (at init) and AFTER training, for both Arm 2 (trained through the bias) and Arm 1
(never bias-trained). A rising cosine in Arm 2 only would indicate SGD is actively aligning raw
keys toward the frozen anchors (compensating for the bias), rather than ignoring it -- the first
thing to check when the primary/co-primary point the OPPOSITE direction from every sim's
prediction (this session's actual finding, both corpora, both instruments).

Run ON THE BOX, CUDA required (chunk_delta_rule has no CPU path for the forward pass we need to
capture k_raw from a real checkpoint).
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lm_pretrain_rd import (CORPUS_DIRS, DEFAULT_DATA_DIR, DeltaNetLM, build_frozen_bias_table,
                             corpus_fixed_seed, get_batch, load_corpus)
from key_anchoring import ANCHOR_INIT_SEED
from frozen_bias_retrofit_eval_rd import load_checkpoint, capture_raw_keys

device = "cuda"
CORPUS = "openr1-mix-ext"
SEED = 0
N_WINDOWS = 32
BATCH_SIZE = 16
SEQ_LEN = 512
CHUNK_SIZE_UNUSED = 64  # not needed here, no gram stats

CKDIR = "/data/deltanet_rd_frozenbias_ckpts"
ARM1_CKPT = f"{CKDIR}/frozenbias_lm_off_lam0p00_{CORPUS}_dm256_ds64_L2_s{SEED}/lmC_{CORPUS}_dm256_ds64_L2_s{SEED}_step20000.pt"
ARM2_CKPT = f"{CKDIR}/frozenbias_lm_per_token_lam0p58_{CORPUS}_dm256_ds64_L2_s{SEED}/lmC_{CORPUS}_dm256_ds64_L2_s{SEED}_step20000.pt"
ARM1_CKPT_STEP0 = None  # will look for an early checkpoint below as an "at-init" proxy


def get_val_batches(corpus_name):
    _, val_tokens, meta, _, val_offs = load_corpus(DEFAULT_DATA_DIR, corpus_name, device)
    gen = torch.Generator(device=device).manual_seed(corpus_fixed_seed(corpus_name) + 95_000)
    n_batches = max(1, -(-N_WINDOWS // BATCH_SIZE))
    return [get_batch(val_tokens, BATCH_SIZE, SEQ_LEN, gen)[0] for _ in range(n_batches)]


def cosine_to_anchor(model, table, batches):
    """Returns mean cosine(k_raw[pos], table[token_id[pos]]) across all layers/positions."""
    keys_by_layer, token_ids_cat = capture_raw_keys(model, batches, device)
    num_heads = model.blocks[0].mixer.num_heads
    d_state = model.blocks[0].mixer.d_state
    head_dim = d_state // num_heads
    per_layer_cos = {}
    for layer_idx, k_raw in keys_by_layer.items():
        B, T, D = k_raw.shape
        anchor_rows = table[token_ids_cat.reshape(-1)].to(k_raw.dtype)  # (B*T, D)
        k_flat = k_raw.reshape(-1, D)
        cos = F.cosine_similarity(k_flat, anchor_rows, dim=-1)
        per_layer_cos[layer_idx] = cos.mean().item()
    return per_layer_cos


if __name__ == "__main__":
    table = build_frozen_bias_table(50257, 64, seed=ANCHOR_INIT_SEED).to(device)

    batches = get_val_batches(CORPUS)

    print("=== Arm 1 (never trained through bias) final checkpoint: k_raw vs anchor cosine ===")
    model1, ckpt1 = load_checkpoint(ARM1_CKPT, device)
    cos1_final = cosine_to_anchor(model1, table, batches)
    print(json.dumps(cos1_final, indent=2))
    del model1, ckpt1
    torch.cuda.empty_cache()

    print("\n=== Arm 2 (trained through bias, lam=0.58) final checkpoint: k_raw vs anchor cosine ===")
    model2, ckpt2 = load_checkpoint(ARM2_CKPT, device)
    cos2_final = cosine_to_anchor(model2, table, batches)
    print(json.dumps(cos2_final, indent=2))
    del model2, ckpt2
    torch.cuda.empty_cache()

    # "At init" proxy: an untrained model with the SAME architecture/seed convention as training
    # uses (step ~0 is not checkpointed; use a freshly constructed model with the SAME init seed
    # convention as the harness, disclosed as an approximation, not the literal step-0 checkpoint
    # of either specific run, since none was saved).
    print("\n=== 'At init' proxy (freshly constructed, untrained model, same arch): k_raw vs anchor cosine ===")
    # Reuse Arm 1's own config for architecture fidelity.
    model1b, ckpt1b = load_checkpoint(ARM1_CKPT, device)
    cfg = ckpt1b["config"]
    del model1b, ckpt1b
    torch.manual_seed(SEED)
    model_init = DeltaNetLM(**cfg).to(device)
    model_init.eval()
    cos_init = cosine_to_anchor(model_init, table, batches)
    print(json.dumps(cos_init, indent=2))

    result = {
        "corpus": CORPUS, "seed": SEED,
        "cosine_at_init_proxy": cos_init,
        "cosine_arm1_final_never_trained_through_bias": cos1_final,
        "cosine_arm2_final_trained_through_bias": cos2_final,
        "note": "cosine_at_init_proxy uses a freshly constructed model (same arch/seed convention), "
                "NOT the literal step-0 checkpoint of either run (none was saved) -- disclosed "
                "approximation, sec 1.3's own diagnostic.",
    }
    with open("results/frozen_bias_lm/cosine_diagnostic_sec1_3.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote results/frozen_bias_lm/cosine_diagnostic_sec1_3.json")
