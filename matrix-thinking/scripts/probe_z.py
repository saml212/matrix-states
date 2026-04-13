#!/usr/bin/env python3
"""
Linear Probe Interpretability on Z (Round 5).

Reuses Round 3 Run 1 checkpoint (matrix CODI gamma=0, thinking ON, ProsQA, best 78.91%).

PRE-REGISTERED HYPOTHESIS:
  Z encodes interpretable reasoning content iff a linear probe on Z's flattened
  representation can recover ProsQA target/neg_target labels with AUC > 0.65,
  AND the AUC must EXCEED both control baselines by >= 5 points:
    Control A: Linear probe on raw GPT-2 (untrained) hidden state at the same prompt
    Control B: Linear probe on randomly initialized GPT-2 hidden state

PRE-REGISTERED FALSIFICATION:
  If matrix-CODI Z probe AUC <= max(control AUC + 5pp), Z does not encode
  reasoning content beyond what the question text alone provides. Negative result.

OUTPUT: probe_results.json with AUC + per-fold std for each condition,
        per latent position, with 5-fold cross-validation.
"""
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/workspace/pebble/round3_gamma0/scripts")
from run_matrix_codi import (
    CodiModel, ProsQADataset, CONFIG,
    GPT2TokenizerFast, GPT2LMHeadModel,
)

OUT_DIR = Path("/workspace/pebble/round5_probe/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = open(OUT_DIR / "probe.log", "w")


def log(msg):
    line = "[" + time.strftime("%H:%M:%S") + "] " + str(msg)
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


CKPT = "/workspace/pebble/round3_gamma0/results/run1_gamma0_think_on/best_run_b_matrix.pt"
log("Loading matrix-CODI checkpoint: " + CKPT)

cfg = dict(CONFIG)
cfg["dataset"] = "prosqa"
cfg["prosqa_train_path"] = "/workspace/pebble/round3_gamma0/data/prosqa_train.json"
cfg["prosqa_val_path"] = "/workspace/pebble/round3_gamma0/data/prosqa_test.json"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

device = torch.device("cpu")  # CPU mode

model_matrix = CodiModel(
    base_model_name="gpt2",
    n_latents=6,
    use_matrix_bottleneck=True,
    mat_dim=16,
    use_thinking_iter=True,
).to(device)
model_matrix.add_special_tokens(tokenizer)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
if "model" in sd:
    sd = sd["model"]
clean = {k.replace("module.", ""): v for k, v in sd.items()}
model_matrix.load_state_dict(clean, strict=True)
model_matrix.eval()
log("Loaded matrix-CODI model")

model_vanilla = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model_vanilla.eval()
log("Loaded vanilla GPT-2 (untrained)")

from transformers import GPT2Config
gpt2_config = GPT2Config.from_pretrained("gpt2")
model_random = GPT2LMHeadModel(gpt2_config).to(device)
model_random.eval()
log("Built random GPT-2")

val_ds = ProsQADataset("val", tokenizer, cfg, model_matrix.special_token_ids)
log("ProsQA test: " + str(len(val_ds)) + " problems")

with open(cfg["prosqa_val_path"]) as f:
    raw_test = json.load(f)
log("Loaded " + str(len(raw_test)) + " raw test rows")

# FIX: build question→raw lookup so feature/label alignment survives any drops.
raw_by_question = {r["question"].strip(): r for r in raw_test}
log("Built question lookup: " + str(len(raw_by_question)) + " unique questions")

features_matrix = []
features_vanilla = []
features_random = []
labels_target_sym = []  # symbolic name (shared across problems), not local index
labels_neg_sym = []
labels_chain_depth = []
labels_correct_pair = []  # binary: (target_sym, neg_sym) — alphabetical order, 0 if target is alphabetically first, 1 otherwise. Just a sanity-check label.

log("Extracting features...")
n_processed = 0
t0 = time.time()

with torch.no_grad():
    for i, item in enumerate(val_ds.items):
        if i >= 500:
            break
        if i % 25 == 0:
            elapsed = int(time.time() - t0)
            log("  problem " + str(i) + "/" + str(len(val_ds.items)) + " (" + str(elapsed) + "s)")

        # FIX: lookup raw entry by question string, not by index.
        raw = raw_by_question.get(item["question"])
        if raw is None:
            continue
        target_idx = raw.get("target")
        neg_idx = raw.get("neg_target")
        idx_to_sym = raw.get("idx_to_symbol", [])
        if target_idx is None or neg_idx is None or not idx_to_sym:
            continue
        if target_idx >= len(idx_to_sym) or neg_idx >= len(idx_to_sym):
            continue

        # FIX: use symbolic names (shared across problems), not per-problem local indices.
        target_sym = idx_to_sym[target_idx]
        neg_sym = idx_to_sym[neg_idx]
        chain_depth = len(raw.get("steps", []))

        q_ids = torch.tensor([item["q_ids"]], dtype=torch.long, device=device)
        q_mask = torch.ones_like(q_ids)
        tail_ids = torch.tensor([item["tail_ids"]], dtype=torch.long, device=device)
        tail_mask = torch.ones_like(tail_ids)

        out = model_matrix.student_forward(q_ids, q_mask, tail_ids, tail_mask, save_Z=True)
        Z_list = out["Z_list"]
        z_feat = torch.stack([z.squeeze(0).flatten() for z in Z_list], dim=0).cpu().float().numpy()
        features_matrix.append(z_feat)

        prompt_ids = item["q_ids"] + tokenizer.encode(" The answer is:", add_special_tokens=False)
        prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        prompt_mask = torch.ones_like(prompt_t)

        for model, feat_list in [(model_vanilla, features_vanilla), (model_random, features_random)]:
            v_out = model(prompt_t, attention_mask=prompt_mask, output_hidden_states=True)
            last_hidden = v_out.hidden_states[-1][:, -1, :].cpu().float().numpy()
            feat_list.append(last_hidden[0])
            del v_out

        labels_target_sym.append(target_sym)
        labels_neg_sym.append(neg_sym)
        labels_chain_depth.append(chain_depth)
        # Stable binary label: is target_sym alphabetically before neg_sym?
        labels_correct_pair.append(1 if target_sym < neg_sym else 0)
        n_processed += 1

log("Processed " + str(n_processed) + " problems in " + str(int(time.time() - t0)) + "s")

features_matrix = np.array(features_matrix)
features_vanilla = np.array(features_vanilla)
features_random = np.array(features_random)
labels_target_sym_arr = np.array(labels_target_sym)
labels_correct_pair = np.array(labels_correct_pair)
chain_depths = np.array(labels_chain_depth)

# FIX: encode symbolic targets as global integer labels (shared across problems).
from sklearn.preprocessing import LabelEncoder
target_encoder = LabelEncoder()
labels_target_global = target_encoder.fit_transform(labels_target_sym_arr)

log("matrix shape: " + str(features_matrix.shape))
log("vanilla shape: " + str(features_vanilla.shape))
log("random shape: " + str(features_random.shape))
log("unique target symbols: " + str(len(target_encoder.classes_)) + " classes")
log("target symbol distribution (top 10): " + str(sorted([(s, int(c)) for s, c in zip(*np.unique(labels_target_sym_arr, return_counts=True))], key=lambda x: -x[1])[:10]))
log("depth distribution: " + str({int(d): int((chain_depths == d).sum()) for d in np.unique(chain_depths)}))
log("binary pair label balance: " + str(int(labels_correct_pair.sum())) + " / " + str(len(labels_correct_pair)))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def fit_probe_cv(X, y, n_folds=5):
    """5-fold CV linear probe. Adapts n_folds down if a class has too few samples."""
    if len(np.unique(y)) < 2:
        return None, None
    from collections import Counter
    counts = Counter(y.tolist())
    min_count = min(counts.values())
    actual_folds = min(n_folds, min_count)
    if actual_folds < 2:
        return None, None
    aucs = []
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    try:
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", solver="lbfgs")
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[test_idx])
            try:
                if proba.shape[1] == 2:
                    auc = roc_auc_score(y[test_idx], proba[:, 1])
                else:
                    auc = roc_auc_score(y[test_idx], proba, multi_class="ovr", average="macro")
                aucs.append(auc)
            except ValueError:
                continue
    except ValueError:
        return None, None
    if not aucs:
        return None, None
    return float(np.mean(aucs)), float(np.std(aucs))


log("=" * 60)
log("PROBE A: target SYMBOL classification (multi-class with shared symbol vocab)")
log("=" * 60)

results = {"target_symbol_classification": {}, "binary_pair_classification": {}}

for pos in range(6):
    auc, std = fit_probe_cv(features_matrix[:, pos, :], labels_target_global)
    msg = "  matrix Z[" + str(pos) + "]: " + (f"AUC={auc:.3f}+/-{std:.3f}" if auc is not None else "invalid")
    log(msg)
    results["target_symbol_classification"]["matrix_Z" + str(pos)] = {"auc": auc, "std": std}

all_z = features_matrix.reshape(features_matrix.shape[0], -1)
auc, std = fit_probe_cv(all_z, labels_target_global)
log("  matrix Z[all-concat]: " + (f"AUC={auc:.3f}+/-{std:.3f}" if auc is not None else "invalid"))
results["target_symbol_classification"]["matrix_Zall"] = {"auc": auc, "std": std}

auc, std = fit_probe_cv(features_vanilla, labels_target_global)
log("  vanilla GPT-2 hidden: " + (f"AUC={auc:.3f}+/-{std:.3f}" if auc is not None else "invalid"))
results["target_symbol_classification"]["vanilla_gpt2"] = {"auc": auc, "std": std}

auc, std = fit_probe_cv(features_random, labels_target_global)
log("  random GPT-2 hidden: " + (f"AUC={auc:.3f}+/-{std:.3f}" if auc is not None else "invalid"))
results["target_symbol_classification"]["random_gpt2"] = {"auc": auc, "std": std}

log("=" * 60)
log("PROBE B: binary alphabetical pair-order (chance = 0.5)")
log("=" * 60)
for pos in range(6):
    auc, std = fit_probe_cv(features_matrix[:, pos, :], labels_correct_pair)
    msg = "  matrix Z[" + str(pos) + "]: " + (f"AUC={auc:.3f}+/-{std:.3f}" if auc is not None else "invalid")
    log(msg)
    results["binary_pair_classification"]["matrix_Z" + str(pos)] = {"auc": auc, "std": std}

auc, std = fit_probe_cv(all_z, labels_correct_pair)
log("  matrix Z[all-concat]: " + (f"AUC={auc:.3f}+/-{std:.3f}" if auc is not None else "invalid"))
results["binary_pair_classification"]["matrix_Zall"] = {"auc": auc, "std": std}
auc, std = fit_probe_cv(features_vanilla, labels_correct_pair)
log("  vanilla GPT-2 hidden: " + (f"AUC={auc:.3f}+/-{std:.3f}" if auc is not None else "invalid"))
results["binary_pair_classification"]["vanilla_gpt2"] = {"auc": auc, "std": std}
auc, std = fit_probe_cv(features_random, labels_correct_pair)
log("  random GPT-2 hidden: " + (f"AUC={auc:.3f}+/-{std:.3f}" if auc is not None else "invalid"))
results["binary_pair_classification"]["random_gpt2"] = {"auc": auc, "std": std}

log("=" * 60)
log("PER-DEPTH analysis on matrix Z[all-concat] (target symbol)")
log("=" * 60)
results["per_depth_matrix"] = {}
for depth in sorted(set(chain_depths.tolist())):
    mask = chain_depths == depth
    if mask.sum() < 10:
        continue
    auc, std = fit_probe_cv(all_z[mask], labels_target_global[mask])
    log("  depth=" + str(int(depth)) + " (n=" + str(int(mask.sum())) + "): " + (f"AUC={auc:.3f}+/-{std:.3f}" if auc is not None else "invalid"))
    results["per_depth_matrix"][int(depth)] = {"n": int(mask.sum()), "auc": auc, "std": std}

log("=" * 60)
log("PRE-REGISTERED DECISION")
log("=" * 60)
matrix_auc = results["target_symbol_classification"]["matrix_Zall"]["auc"] or 0.0
vanilla_auc = results["target_symbol_classification"]["vanilla_gpt2"]["auc"] or 0.0
random_auc = results["target_symbol_classification"]["random_gpt2"]["auc"] or 0.0

log("Matrix Z (all positions): " + f"{matrix_auc:.3f}")
log("Vanilla GPT-2 hidden:     " + f"{vanilla_auc:.3f}")
log("Random GPT-2 hidden:      " + f"{random_auc:.3f}")

threshold = max(vanilla_auc, random_auc) + 0.05
verdict = "POSITIVE" if matrix_auc > threshold else "NULL"
log("Threshold (max control + 0.05): " + f"{threshold:.3f}")
log("VERDICT: " + verdict)

results["pre_registered_verdict"] = {
    "matrix_auc": matrix_auc,
    "vanilla_auc": vanilla_auc,
    "random_auc": random_auc,
    "threshold": threshold,
    "verdict": verdict,
}

with open(OUT_DIR / "probe_results.json", "w") as f:
    json.dump(results, f, indent=2)

log("Saved: " + str(OUT_DIR / "probe_results.json"))
log("DONE")
LOG.close()
