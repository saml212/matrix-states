#!/usr/bin/env python3
"""Control A (v2): propagating fake-Z rank-k ablation on vanilla GPT-2 SFT for ProsQA.

Null baseline for the matrix-CODI rank-k ablation. See
matrix-thinking/QUEUE.md PRIORITY 0 for the original spec and
matrix-thinking/CONTROL_A_ATTACK_REPORT.md for the redesign rationale
(fatal attacks F1 and F2 required a propagating intervention at
analog-of-latent positions, not a single readout-layer ablation).

Protocol:
  - Load a Round 4 pure_sft_seed{S}.pt checkpoint (vanilla GPT-2 small + 3
    added special tokens, resized embeddings to vocab 50260).
  - On each ProsQA eval example, run GPT-2 over
    [q_ids] + encode(" The answer is:").
  - Designate 6 "analog-latent" positions = the 6 token positions
    immediately preceding the ":" token (mirroring matrix-CODI's 6 latent
    slots that sit between <bot> and <eot> before " The answer is:").
  - Register a forward hook on transformer block L_INTERVENE that, for each
    of those 6 positions, rank-k-truncates the hidden state there using a
    fake-Z reshape (primary: 16x16 from h[:256]). The remaining transformer
    blocks L_INTERVENE+1..N-1 process the modified residual stream, so
    downstream attention (both at ":" and at generated-answer positions)
    sees the intervention. This matches matrix-CODI's propagating
    intervention at 6 latent positions (run_matrix_codi.py L519-548).
  - Greedy-decode up to max_new_tokens tokens WITHOUT a KV cache: every
    decoding step re-runs the full transformer so the hook re-fires and
    downstream attention always sees the intervention. This closes the KV-
    cache-leakage loophole that the attack report (F1) flagged as fatal.
  - Check prosqa_answer_match(predicted_text, gold_text). Aggregate per k
    per seed; compute Spearman r. Classify FLAT / BENDING / AMBIGUOUS with
    a binomial-aware decision rule that accounts for sample-size power.

Controls shipped in the same run:
  - k=None un-ablated baseline (no hook registered, still no KV cache so
    the compute path matches the ablated paths). Sanity-checks that the
    script reproduces Round 4's ~81.77% ProsQA accuracy for seed 1337.
  - randomized-h baseline (sensitivity floor): at the 6 analog positions,
    replace h with i.i.d. Gaussian noise matched in mean/std. Confirms the
    transformer is actually sensitive to those positions' residual streams.
    If randomization barely moves accuracy, Control A is uninformative
    regardless of outcome.
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# Import primitives from the matrix-CODI reference script. Do NOT duplicate.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from run_matrix_codi import (  # noqa: E402
    CONFIG as MATRIX_CODI_CONFIG,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    ProsQADataset,
    assert_colon_tokenization,
    build_logger,
    effective_rank,
    prosqa_answer_match,
    set_seed,
    truncate_to_rank,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Reshape variants. (rows, cols, covered) where covered is the number of
# entries of h (per analog position) that the reconstruction overwrites.
# "16x16"        : reshape h[:256] to (16, 16). Matches matrix-CODI's d=16.
# "16x48"        : reshape all of h to (16, 48). Full-768 robustness check.
# "svd_full_768" : reshape all of h to (24, 32). Square-ish robustness.
_VARIANT_SHAPES = {
    "16x16": (16, 16, 256),
    "16x48": (16, 48, 768),
    "svd_full_768": (24, 32, 768),
}

# Number of analog-latent positions. Matches matrix-CODI's cfg["n_latents"]=6.
N_ANALOG_POSITIONS = 6

# Default transformer block index at whose OUTPUT we intervene. Chosen so
# that several downstream blocks (6 of 12) re-attend over the intervened
# residual stream, matching the "intervention propagates through attention"
# property that matrix-CODI has (its latent positions feed input embeddings,
# so all 12 layers see them). Block index is 0-indexed; GPT-2 small has 12.
DEFAULT_INTERVENE_LAYER = 6


# =============================================================================
# FAKE-Z CONSTRUCTION
# =============================================================================

def build_fake_z(h, variant):
    """Reshape a hidden-state batch into a (B, rows, cols) fake-Z.

    Args:
        h: (B, D) hidden state (D=768 for GPT-2 small).
        variant: one of _VARIANT_SHAPES.

    Returns:
        (Z, covered): Z is (B, rows, cols) to be SVD-truncated; covered is
            the number of h entries that will be overwritten by the
            flattened reconstruction.
    """
    if variant not in _VARIANT_SHAPES:
        raise ValueError(f"Unknown variant: {variant}")
    rows, cols, covered = _VARIANT_SHAPES[variant]
    B, D = h.shape
    assert D >= covered, f"hidden dim {D} < covered {covered} for variant {variant}"
    Z = h[:, :covered].reshape(B, rows, cols)
    return Z, covered


def apply_rank_k_intervention(h, k, variant):
    """SVD-truncate a fake-Z of h to rank k, splice back into h.

    Args:
        h: (B, D) hidden state.
        k: target rank.
        variant: reshape variant.

    Returns:
        (h_mod, Z_trunc) where h_mod is (B, D) and Z_trunc is (B, rows, cols).
    """
    Z, covered = build_fake_z(h, variant)
    Z_trunc = truncate_to_rank(Z, k)
    B = h.shape[0]
    recon_flat = Z_trunc.reshape(B, -1)
    h_mod = h.clone()
    h_mod[:, :covered] = recon_flat
    return h_mod, Z_trunc


def apply_randomized_intervention(h, variant, generator=None):
    """Replace the covered slice with i.i.d. Gaussian noise of matched mean/std.

    Used for the randomized-h sensitivity-floor control. Returns (h_mod, None).
    """
    rows, cols, covered = _VARIANT_SHAPES[variant]
    B, D = h.shape
    slice_flat = h[:, :covered]
    mean = slice_flat.mean()
    std = slice_flat.std().clamp(min=1e-6)
    rand = torch.empty_like(slice_flat)
    if generator is not None:
        # generator may live on CPU; sample on CPU and move if needed.
        cpu_rand = torch.empty(slice_flat.shape, dtype=torch.float32).normal_(
            mean=float(mean.float().item()),
            std=float(std.float().item()),
            generator=generator,
        )
        rand = cpu_rand.to(dtype=slice_flat.dtype, device=slice_flat.device)
    else:
        rand.normal_(mean=float(mean.float().item()), std=float(std.float().item()))
    h_mod = h.clone()
    h_mod[:, :covered] = rand
    return h_mod, None


# =============================================================================
# PROPAGATING INTERVENTION HOOK
# =============================================================================

class InterventionHookState:
    """Mutable container so the hook closure can be updated between forward
    passes (e.g., to switch target positions or k) without re-registering.
    """
    def __init__(self):
        self.positions = None       # tensor of shape (n_positions,) int64 or None
        self.k = None               # int or None or "random"
        self.variant = "16x16"
        self.last_eff_ranks = []    # list of floats, filled each hook fire
        self.generator = None       # optional torch.Generator for randomized
        self.enabled = False
        self.fire_count = 0


def make_intervention_hook(state):
    """Return a forward hook closure on a transformer block's output.

    The block output is a tuple whose [0] is hidden states (B, L, D). The
    hook modifies hidden states at state.positions in-place-style (returns
    a new tuple).
    """
    def hook(module, inputs, outputs):
        if not state.enabled or state.positions is None or state.k is None:
            return outputs
        # outputs is usually (hidden_states, ...). GPT-2 block returns a tuple.
        if isinstance(outputs, tuple):
            hs = outputs[0]
        else:
            hs = outputs
        # hs: (B, L, D). state.positions: (P,) long tensor of positions.
        B, L, D = hs.shape
        pos = state.positions.to(hs.device)
        # Gather the hidden states at target positions: (B, P, D).
        h_slice = hs[:, pos, :]   # (B, P, D)
        # Flatten batch+position for the (B', D) convention expected by the
        # rank-k primitive.
        BP = h_slice.shape[0] * h_slice.shape[1]
        h_flat = h_slice.reshape(BP, D)
        if state.k == "random":
            h_mod_flat, _ = apply_randomized_intervention(h_flat, state.variant,
                                                          generator=state.generator)
            eff_rank_val = float("nan")
        else:
            h_mod_flat, Z_trunc = apply_rank_k_intervention(h_flat, int(state.k),
                                                            state.variant)
            # effective_rank over (BP, r, c) returns (BP,). Mean for log.
            eff_rank_val = float(effective_rank(Z_trunc).mean().item())
        state.last_eff_ranks.append(eff_rank_val)
        state.fire_count += 1
        h_mod = h_mod_flat.reshape(h_slice.shape)
        # Write back.
        hs_new = hs.clone()
        hs_new[:, pos, :] = h_mod
        if isinstance(outputs, tuple):
            return (hs_new,) + outputs[1:]
        return hs_new
    return hook


# =============================================================================
# CHECKPOINT LOADING (audit P0-1, P0-2)
# =============================================================================

def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


_EXPECTED_PARAM_COUNT = 124_442_112  # GPT-2 small + 3 added special tokens (50257+3=50260 rows × 768 dim, tied wte/lm_head).


def load_vanilla_sft(checkpoint_path, device, checkpoint_format=None):
    """Load a Round 4 pure_sft checkpoint into a GPT-2 small with 3 added
    special tokens. Handles raw state_dict, wrapped {"model": sd}, `module.`
    (DDP) prefix, and optional `gpt2.`-prefixed CodiModel wrapper.

    Args:
        checkpoint_path: path to a .pt file.
        device: torch.device.
        checkpoint_format: optional explicit hint. One of:
            None (auto-detect), "state_dict", "wrapped", "codi_wrapped".

    Returns:
        (model, tokenizer, special_ids)
    """
    assert Path(checkpoint_path).exists(), f"checkpoint not found: {checkpoint_path}"
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Round 4 added the same 3 special tokens that matrix-CODI uses; the
    # checkpoint's wte/lm_head is sized for vocab=50260.
    tokenizer.add_special_tokens({
        "additional_special_tokens": ["<bot>", "<eot>", "<latent>"],
    })
    assert_colon_tokenization(tokenizer)

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    # Audit P2-3: weights_only=False matches run_matrix_codi.py convention;
    # checkpoints are our own file, trusted.
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Audit P0-1: majority-or-unanimous check, not naive any().
    # If sd is wrapped in {"model": actual_sd, ...}, unwrap.
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        # Check if the top-level looks like a wrapper (has cfg / seed / etc.)
        top_keys = set(sd.keys())
        if top_keys & {"cfg", "seed", "tokenizer_len", "use_matrix_bottleneck"}:
            sd = sd["model"]
        else:
            # The top-level dict might just happen to have a "model" key as
            # a weight. Force the user to disambiguate.
            if checkpoint_format == "wrapped":
                sd = sd["model"]
            # else leave sd as-is; downstream critical-keys assert will catch.

    # Strip DDP `module.` prefix if present.
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # Strip `gpt2.` prefix only if MAJORITY of keys are gpt2-prefixed (audit P0-1).
    # This avoids silently dropping the wrong subset of keys when only a minority
    # happen to share the prefix.
    keys = list(sd.keys())
    gpt2_count = sum(1 for k in keys if k.startswith("gpt2."))
    if checkpoint_format == "codi_wrapped" or (gpt2_count > 0 and gpt2_count >= 0.5 * len(keys)):
        sd = {k[len("gpt2."):]: v for k, v in sd.items() if k.startswith("gpt2.")}
        # We deliberately drop non-gpt2.-prefixed keys (bottleneck.*, feedback_proj.*)
        # because those do not belong to a GPT2LMHeadModel.

    # Audit P0-2: critical-keys assert.
    critical_keys = [
        "transformer.wte.weight",
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
        "lm_head.weight",
        "transformer.h.0.attn.c_attn.weight",
    ]
    for ck in critical_keys:
        assert ck in sd, (
            f"load_vanilla_sft: critical key missing from checkpoint state_dict: "
            f"{ck!r}. Available keys (first 10): {list(sd.keys())[:10]}. "
            f"Pass --checkpoint-format to disambiguate wrapper schemas."
        )

    # Param-count assert BEFORE load (model side).
    n_params_before = sum(p.numel() for p in model.parameters())
    assert n_params_before == _EXPECTED_PARAM_COUNT, (
        f"GPT-2 + 3 special tokens should have {_EXPECTED_PARAM_COUNT:,} params "
        f"but model has {n_params_before:,}. Tokenizer resize mismatch?"
    )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Audit P0-2: hard-fail on critical missing keys, not just print.
    critical_missing = [k for k in missing if k in critical_keys]
    assert not critical_missing, (
        f"Critical keys missing from state_dict load: {critical_missing}. "
        f"Full missing[:10]: {missing[:10]}"
    )
    if unexpected:
        print(f"[load_vanilla_sft] unexpected keys (ignored): {unexpected[:5]}"
              f"{' ...' if len(unexpected) > 5 else ''}", flush=True)
    if missing:
        print(f"[load_vanilla_sft] non-critical missing keys: {missing[:5]}"
              f"{' ...' if len(missing) > 5 else ''}", flush=True)

    # Post-load param count (should be unchanged — load_state_dict doesn't
    # add/remove tensors, only overwrites values).
    n_params_after = sum(p.numel() for p in model.parameters())
    assert n_params_after == _EXPECTED_PARAM_COUNT, (
        f"Post-load param count {n_params_after:,} != expected {_EXPECTED_PARAM_COUNT:,}."
    )

    # Embedding shape sanity check vs tokenizer.
    emb_shape = model.transformer.wte.weight.shape
    assert emb_shape[0] == len(tokenizer), (
        f"wte rows {emb_shape[0]} != tokenizer len {len(tokenizer)}. "
        f"Special-token resize mismatch."
    )

    model.to(device)
    model.eval()

    special_ids = {
        "bot": tokenizer.convert_tokens_to_ids("<bot>"),
        "eot": tokenizer.convert_tokens_to_ids("<eot>"),
        "latent": tokenizer.convert_tokens_to_ids("<latent>"),
        "pad": tokenizer.pad_token_id,
        "eos": tokenizer.eos_token_id,
    }
    return model, tokenizer, special_ids


# =============================================================================
# PROPAGATING INTERVENTION + GENERATION (one example)
# =============================================================================

def _find_colon_position(prompt_ids, tokenizer):
    """Return the index of the last ':' token in prompt_ids."""
    colon_id = tokenizer.encode(":", add_special_tokens=False)[0]
    # Iterate from the end — the prompt ends with "The answer is:" so the
    # colon is near the tail.
    for i in range(len(prompt_ids) - 1, -1, -1):
        if prompt_ids[i] == colon_id:
            return i
    raise AssertionError(f"no ':' token found in prompt_ids of length {len(prompt_ids)}")


def _compute_analog_positions(prompt_len, colon_pos, n_positions=N_ANALOG_POSITIONS):
    """Return the tensor of positions where the intervention fires.

    Choice: the n_positions positions IMMEDIATELY PRECEDING the ':' token.
    This mirrors matrix-CODI's 6 latent slots between <bot> and <eot>, which
    sit RIGHT BEFORE " The answer is:" in the sequence; the vanilla-SFT
    analog is the 6 token positions right before ":" (which terminates
    " The answer is:"). Documented in CONTROL_A_IMPLEMENTATION_NOTES.md.
    """
    # Positions colon_pos - n_positions .. colon_pos - 1 (inclusive).
    start = max(0, colon_pos - n_positions)
    end = colon_pos   # exclusive
    positions = list(range(start, end))
    if len(positions) < n_positions:
        # Prompt is shorter than n_positions + 1. Fall back to all prefix positions.
        positions = list(range(0, colon_pos))
    return torch.tensor(positions, dtype=torch.long)


@torch.no_grad()
def predict_one_propagating(model, tokenizer, q_ids, k, variant, device,
                            intervene_layer, max_new_tokens=24,
                            hook_state=None, hook_handle=None,
                            rand_generator=None, log_shapes=False, logger=None):
    """Greedy-decode an answer under a propagating rank-k intervention.

    The intervention fires via a forward hook on transformer block
    `intervene_layer`'s output. The hook replaces hidden states at the
    analog positions with their rank-k-truncated versions. Because we
    regenerate each token with a FULL forward (no KV cache), downstream
    positions (the ':' token and every generated token) always re-attend
    over the intervened residual stream.

    Args:
        model: GPT2LMHeadModel on device.
        tokenizer: GPT2TokenizerFast with 3 added special tokens.
        q_ids: list[int] question token ids.
        k: rank. If None, no intervention (hook disabled). If "random", the
            randomized-sensitivity-floor baseline.
        variant: reshape variant for the fake-Z.
        device: torch.device.
        intervene_layer: block index to hook.
        max_new_tokens: max tokens to generate after the prompt.
        hook_state: pre-constructed InterventionHookState (reused across calls
            to avoid repeated register/unregister).
        hook_handle: handle returned by register_forward_hook; used to
            confirm the hook is attached. (Not unregistered here.)
        rand_generator: torch.Generator for "random" k. Optional.
        log_shapes: whether to log shapes once (for smoke tests).
        logger: optional logger.

    Returns:
        dict with keys: predicted_text, first_token, first_token_id,
                        h_orig_norm_at_analog, h_mod_norm_at_analog (both
                        computed from the FIRST hook fire), z_eff_rank,
                        intervene_fire_count, colon_pos, analog_positions.
    """
    prompt_ids = q_ids + tokenizer.encode(" The answer is:", add_special_tokens=False)
    colon_pos = _find_colon_position(prompt_ids, tokenizer)
    analog_positions = _compute_analog_positions(len(prompt_ids), colon_pos)

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    # Configure hook state.
    if hook_state is not None:
        hook_state.positions = analog_positions
        hook_state.k = k
        hook_state.variant = variant
        hook_state.last_eff_ranks = []
        hook_state.generator = rand_generator
        hook_state.enabled = (k is not None)
        hook_state.fire_count = 0

    # For the norm diagnostics we need both the pre-intervention and post-
    # intervention hidden state at the analog positions at the hooked block's
    # output. Easiest: run once with hook disabled to capture original h,
    # then enable and run.
    h_orig_norm = float("nan")
    if k is not None and hook_state is not None:
        hook_state.enabled = False
        out_clean = model.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        # hidden_states[i] is the output of block i-1 (HF convention: index 0
        # is embeddings, index L is post-ln_f final). So the output of block
        # `intervene_layer` is hidden_states[intervene_layer + 1].
        hs_at_layer = out_clean.hidden_states[intervene_layer + 1]
        # hs_at_layer: (1, L, D). Compute norm at analog positions.
        h_orig_slice = hs_at_layer[:, analog_positions.to(hs_at_layer.device), :]
        h_orig_norm = float(h_orig_slice.norm().item())
        hook_state.enabled = True

    if log_shapes and logger is not None:
        logger.log(
            f"  [shapes] prompt_len={len(prompt_ids)} colon_pos={colon_pos} "
            f"analog_positions={analog_positions.tolist()} "
            f"intervene_layer={intervene_layer}"
        )

    # First forward over the full prompt — intervention fires here if enabled.
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits_last = out.logits[:, -1, :]
    first_tok = logits_last.argmax(dim=-1, keepdim=True)  # (1, 1)
    first_tok_id = int(first_tok.item())
    first_tok_text = tokenizer.decode([first_tok_id], skip_special_tokens=True)
    generated = [first_tok_id]

    # Compute h_mod norm from the first hook fire.
    h_mod_norm = float("nan")
    z_eff_rank = None
    if k is not None and hook_state is not None and hook_state.last_eff_ranks:
        # Approximate: rerun one forward with hook enabled and capture the
        # intervened hidden states via an auxiliary cache-on-hook.
        # Cheap alternative: we already have mean eff_rank from the hook.
        z_eff_rank = (
            hook_state.last_eff_ranks[0]
            if hook_state.last_eff_ranks[0] == hook_state.last_eff_ranks[0]  # nan filter
            else None
        )

    eos_id = tokenizer.eos_token_id
    if first_tok_id == eos_id:
        return {
            "predicted_text": first_tok_text,
            "first_token": first_tok_text,
            "first_token_id": first_tok_id,
            "h_orig_norm_at_analog": h_orig_norm,
            "h_mod_norm_at_analog": h_mod_norm,
            "z_eff_rank": z_eff_rank,
            "intervene_fire_count": hook_state.fire_count if hook_state else 0,
            "colon_pos": colon_pos,
            "analog_positions": analog_positions.tolist(),
        }

    # NO-KV-CACHE greedy decode. Each step re-runs the transformer on the
    # growing sequence so the hook re-fires at the analog positions every
    # time and downstream attention always sees the intervened states.
    # This is the propagating-intervention property (attack F1 fix).
    cur_ids = torch.cat(
        [input_ids, first_tok], dim=1
    )  # (1, L+1)
    cur_mask = torch.ones_like(cur_ids)
    for _step in range(max_new_tokens - 1):
        step_out = model(
            input_ids=cur_ids,
            attention_mask=cur_mask,
            use_cache=False,
            return_dict=True,
        )
        nxt = step_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        nxt_id = int(nxt.item())
        generated.append(nxt_id)
        cur_ids = torch.cat([cur_ids, nxt], dim=1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=device)], dim=1,
        )
        if nxt_id == eos_id:
            break

    predicted_text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "predicted_text": predicted_text,
        "first_token": first_tok_text,
        "first_token_id": first_tok_id,
        "h_orig_norm_at_analog": h_orig_norm,
        "h_mod_norm_at_analog": h_mod_norm,
        "z_eff_rank": z_eff_rank,
        "intervene_fire_count": hook_state.fire_count if hook_state else 0,
        "colon_pos": colon_pos,
        "analog_positions": analog_positions.tolist(),
    }


# =============================================================================
# EVAL ONE (K, SEED) COMBINATION
# =============================================================================

@torch.no_grad()
def eval_one(model, tokenizer, dataset, k, variant, device, intervene_layer,
             max_new_tokens, hook_state, rand_generator=None,
             logger=None, progress_every=50, max_per_example_rows=50):
    """Evaluate one (k, variant) on a ProsQADataset.

    Returns dict with n_correct, n_total, accuracy, first_token_accuracy
    (attack M1), mean_effective_rank, per_example (capped for JSON size).
    """
    n_correct = 0
    n_first_correct = 0
    n_total = 0
    per_example_full = []
    eff_ranks = []
    t0 = time.time()

    for idx in range(len(dataset)):
        item = dataset[idx]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = predict_one_propagating(
                model=model,
                tokenizer=tokenizer,
                q_ids=item["q_ids"],
                k=k,
                variant=variant,
                device=device,
                intervene_layer=intervene_layer,
                max_new_tokens=max_new_tokens,
                hook_state=hook_state,
                rand_generator=rand_generator,
            )
        correct = prosqa_answer_match(out["predicted_text"], item["answer_text"])
        n_correct += int(correct)
        n_total += 1

        # Attack M1: first-token accuracy as a separate column. Matches ONLY
        # the first generated BPE piece against the first BPE piece of the
        # gold class word.
        gold_text = item["answer_text"]
        gold_class_ids = _gold_class_first_bpe(tokenizer, gold_text)
        first_correct = (gold_class_ids is not None
                         and out["first_token_id"] in gold_class_ids)
        n_first_correct += int(first_correct)

        if out["z_eff_rank"] is not None and out["z_eff_rank"] == out["z_eff_rank"]:
            eff_ranks.append(out["z_eff_rank"])

        per_example_full.append({
            "problem_id": idx,
            "reasoning_steps": item["reasoning_steps"],
            "gold": gold_text,
            "predicted_first_class": _first_sentence_final_class(out["predicted_text"]),
            "predicted": out["predicted_text"],  # attack minor m3: log full text.
            "first_token_id": out["first_token_id"],
            "first_token": out["first_token"],
            "first_token_correct": bool(first_correct),
            "correct": bool(correct),
            "z_eff_rank": out["z_eff_rank"],
            "intervene_fire_count": out["intervene_fire_count"],
        })

        if logger is not None and progress_every and (idx + 1) % progress_every == 0:
            elapsed = time.time() - t0
            acc_so_far = n_correct / max(n_total, 1)
            total = len(dataset)
            eta = elapsed * (total - (idx + 1)) / max(idx + 1, 1)  # P2-2
            logger.log(
                f"    k={k} variant={variant}: {idx + 1}/{total} "
                f"acc={acc_so_far*100:.2f}% "
                f"first_tok_acc={n_first_correct/max(n_total,1)*100:.2f}% "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
            )

    acc = n_correct / max(n_total, 1)
    first_acc = n_first_correct / max(n_total, 1)
    mean_eff_rank = sum(eff_ranks) / len(eff_ranks) if eff_ranks else None
    # Audit P1-7 / attack minor: cap stored per_example to max_per_example_rows.
    per_example_json = per_example_full[:max_per_example_rows]
    return {
        "n_correct": n_correct,
        "n_first_correct": n_first_correct,
        "n_total": n_total,
        "accuracy": acc,
        "first_token_accuracy": first_acc,
        "mean_effective_rank": mean_eff_rank,
        "per_example": per_example_json,
    }


def _first_sentence_final_class(s):
    """Mirror prosqa_answer_match's first_sentence_final_class logic."""
    import re
    s = (s or "").strip()
    for sep in [".", "?", "!", "\n"]:
        i = s.find(sep)
        if i >= 0:
            s = s[:i]
            break
    s = s.strip().lower()
    tokens = re.split(r"\s+", s)
    return tokens[-1] if tokens and tokens[-1] else ""


def _gold_class_first_bpe(tokenizer, gold_text):
    """Return the set of token ids that could be the FIRST BPE piece of the
    gold class word. For 'X is a bunlion.', the class word is 'bunlion' and
    its first BPE piece is whatever tokenizer.encode(' bunlion') yields[0].
    Using leading space is important because GPT-2 BPE is space-sensitive.

    Returns None if the gold text can't be parsed.
    """
    cls = _first_sentence_final_class(gold_text)
    if not cls:
        return None
    with_space = " " + cls
    ids_space = tokenizer.encode(with_space, add_special_tokens=False)
    ids_plain = tokenizer.encode(cls, add_special_tokens=False)
    out = set()
    if ids_space:
        out.add(ids_space[0])
    if ids_plain:
        out.add(ids_plain[0])
    return out if out else None


# =============================================================================
# SWEEP (one seed)
# =============================================================================

def _spearman(x, y):
    """Spearman r via scipy if available, else a tie-aware fallback. Returns
    (r, p); p is nan under the fallback."""
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y)
        return float(r), float(p)
    except Exception:
        def rank_list(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            ranks = [0.0] * len(v)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1.0
                for q in range(i, j + 1):
                    ranks[order[q]] = avg
                i = j + 1
            return ranks
        rx = rank_list(list(x))
        ry = rank_list(list(y))
        n = len(rx)
        if n < 2:
            return float("nan"), float("nan")
        mx = sum(rx) / n
        my = sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        dx = sum((a - mx) ** 2 for a in rx) ** 0.5
        dy = sum((b - my) ** 2 for b in ry) ** 0.5
        if dx == 0 or dy == 0:
            return float("nan"), float("nan")
        return float(num / (dx * dy)), float("nan")


def sweep_one_seed(checkpoint_path, seed, ks, variant, n_examples,
                   max_new_tokens, device, logger, cfg_for_dataset,
                   intervene_layer, run_randomized_control=True,
                   checkpoint_format=None, max_per_example_rows=50):
    """Run the full k-sweep for one seed, including k=None baseline and the
    randomized sensitivity-floor control. Returns a dict for per_seed JSON.
    """
    logger.log(f"--- seed={seed} checkpoint={checkpoint_path} ---")
    set_seed(seed)
    try:
        ckpt_sha = _sha256_file(checkpoint_path)
    except Exception as e:
        logger.log(f"  warning: sha256 failed: {e}")
        ckpt_sha = None

    model, tokenizer, _special_ids = load_vanilla_sft(
        checkpoint_path, device, checkpoint_format=checkpoint_format,
    )
    n_params = sum(p.numel() for p in model.parameters())  # audit P1-5
    logger.log(f"  params: {n_params:,}  expected: {_EXPECTED_PARAM_COUNT:,}")

    # Register the intervention hook on the chosen transformer block.
    hook_state = InterventionHookState()
    block = model.transformer.h[intervene_layer]
    hook_handle = block.register_forward_hook(make_intervention_hook(hook_state))
    logger.log(f"  hook registered on transformer.h[{intervene_layer}]")

    # Deterministic generator for "random" baseline.
    rand_gen = torch.Generator().manual_seed(seed + 10_000)

    dataset = ProsQADataset("val", tokenizer, cfg_for_dataset, {
        "bot": tokenizer.convert_tokens_to_ids("<bot>"),
        "eot": tokenizer.convert_tokens_to_ids("<eot>"),
        "latent": tokenizer.convert_tokens_to_ids("<latent>"),
        "pad": tokenizer.pad_token_id,
        "eos": tokenizer.eos_token_id,
    })
    if n_examples is not None and n_examples < len(dataset):
        dataset.items = dataset.items[:n_examples]
    logger.log(f"  eval problems: {len(dataset)}")

    per_k = {}

    # k=None un-ablated baseline (attack M4; control #1).
    logger.log(f"  evaluating k=None (unablated baseline)...")
    try:
        res = eval_one(
            model, tokenizer, dataset, k=None, variant=variant,
            device=device, intervene_layer=intervene_layer,
            max_new_tokens=max_new_tokens, hook_state=hook_state,
            rand_generator=None, logger=logger,
            max_per_example_rows=max_per_example_rows,
        )
        logger.log(
            f"  k=None: accuracy {res['accuracy']*100:.2f}% "
            f"first_tok_acc {res['first_token_accuracy']*100:.2f}% "
            f"({res['n_correct']}/{res['n_total']})"
        )
        per_k["None"] = {
            "n_correct": res["n_correct"],
            "n_first_correct": res["n_first_correct"],
            "n_total": res["n_total"],
            "accuracy": res["accuracy"],
            "first_token_accuracy": res["first_token_accuracy"],
            "mean_effective_rank": res["mean_effective_rank"],
            "per_example": res["per_example"],
        }
    except Exception as e:
        logger.log(f"  k=None: FAILED ({type(e).__name__}: {e})")
        per_k["None"] = {"error": f"{type(e).__name__}: {e}"}

    # k=int sweep.
    for k in ks:
        try:
            res = eval_one(
                model, tokenizer, dataset, k=k, variant=variant,
                device=device, intervene_layer=intervene_layer,
                max_new_tokens=max_new_tokens, hook_state=hook_state,
                rand_generator=None, logger=logger,
                max_per_example_rows=max_per_example_rows,
            )
            logger.log(
                f"  k={k:>2d}: accuracy {res['accuracy']*100:.2f}% "
                f"first_tok_acc {res['first_token_accuracy']*100:.2f}% "
                f"({res['n_correct']}/{res['n_total']}) "
                f"mean_eff_rank={res['mean_effective_rank']}"
            )
            per_k[str(k)] = {
                "n_correct": res["n_correct"],
                "n_first_correct": res["n_first_correct"],
                "n_total": res["n_total"],
                "accuracy": res["accuracy"],
                "first_token_accuracy": res["first_token_accuracy"],
                "mean_effective_rank": res["mean_effective_rank"],
                "per_example": res["per_example"],
            }
        except Exception as e:
            logger.log(f"  k={k}: FAILED ({type(e).__name__}: {e})")
            per_k[str(k)] = {"error": f"{type(e).__name__}: {e}"}

    # Randomized sensitivity-floor control.
    if run_randomized_control:
        logger.log(f"  evaluating k='random' (sensitivity floor)...")
        try:
            res = eval_one(
                model, tokenizer, dataset, k="random", variant=variant,
                device=device, intervene_layer=intervene_layer,
                max_new_tokens=max_new_tokens, hook_state=hook_state,
                rand_generator=rand_gen, logger=logger,
                max_per_example_rows=max_per_example_rows,
            )
            logger.log(
                f"  k=random: accuracy {res['accuracy']*100:.2f}% "
                f"first_tok_acc {res['first_token_accuracy']*100:.2f}% "
                f"({res['n_correct']}/{res['n_total']})"
            )
            per_k["random"] = {
                "n_correct": res["n_correct"],
                "n_first_correct": res["n_first_correct"],
                "n_total": res["n_total"],
                "accuracy": res["accuracy"],
                "first_token_accuracy": res["first_token_accuracy"],
                "mean_effective_rank": None,
                "per_example": res["per_example"],
            }
        except Exception as e:
            logger.log(f"  k=random: FAILED ({type(e).__name__}: {e})")
            per_k["random"] = {"error": f"{type(e).__name__}: {e}"}

    # Per-seed Spearman over the INT ks only.
    ks_valid = [int(k) for k in per_k if k.isdigit() and "accuracy" in per_k[k]]
    accs_valid = [per_k[str(k)]["accuracy"] for k in ks_valid]
    if len(ks_valid) >= 2:
        r, p = _spearman(ks_valid, accs_valid)
    else:
        r, p = float("nan"), float("nan")

    hook_handle.remove()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_sha,
        "n_params": n_params,
        "per_k": per_k,
        "spearman_r": r,
        "spearman_p": p,
    }


# =============================================================================
# DECISION RULE (attack M3 — binomial-aware AMBIGUOUS)
# =============================================================================

_DECISION_RULE = (
    "binomial-aware: ambiguous if range_pp < 2*binomial_se_pp(n, p_mean); "
    "else flat if |r|<0.3; bending if range_pp>5 and monotone; else ambiguous"
)


def _binomial_se_pp(n, p):
    """Binomial standard error in percentage points."""
    if n <= 0:
        return float("inf")
    p = max(0.0, min(1.0, float(p)))
    return 100.0 * (p * (1 - p) / n) ** 0.5


def classify(ks, accs, r, n_examples_per_k, pooled_n_seeds=1):
    """Return 'flat' | 'bending' | 'ambiguous' using a sample-size-aware rule.

    Arguments:
        ks: list of k values (ints).
        accs: list of accuracies (0..1) corresponding to ks.
        r: Spearman r (may be nan).
        n_examples_per_k: examples per (k, seed).
        pooled_n_seeds: pooled seed count (for pooled-accuracy rows).
    """
    if len(accs) < 3:
        # Audit P1-3: single-datapoint case cannot support any decision.
        return "ambiguous"
    rng_pp = (max(accs) - min(accs)) * 100.0
    mean_p = sum(accs) / len(accs)
    # 2 * binomial SE at the effective sample size (n × seeds).
    n_eff = n_examples_per_k * pooled_n_seeds
    noise_floor_pp = 2.0 * _binomial_se_pp(n_eff, mean_p)
    # Attack M3: AMBIGUOUS when the observed range is inside the binomial
    # noise floor — cannot distinguish flat from gently-bending at this n.
    if rng_pp < noise_floor_pp:
        return "ambiguous"
    r_nan = not (r == r)  # audit P1-1: treat nan as unknown, not flat.
    monotone_decreasing_in_small_k = all(
        accs[i] <= accs[i + 1] + 1e-9 for i in range(len(accs) - 1)
    )
    if not r_nan and abs(r) < 0.3 and rng_pp < 2.0:
        return "flat"
    if rng_pp > 5.0 and monotone_decreasing_in_small_k:
        return "bending"
    if not r_nan and abs(r) < 0.3:
        # |r| weak but range exceeds noise floor → ambiguous, not flat.
        return "ambiguous"
    return "ambiguous"


# =============================================================================
# SMOKE TEST
# =============================================================================

def run_smoke_test(checkpoint_path, variant, device, cfg_for_dataset, logger,
                   intervene_layer, checkpoint_format=None,
                   unablated_n_examples=10):
    """Exercises all the critical paths.

    1. Checkpoint loads with critical-keys + param-count assert.
    2. 10-example unablated (k=None) decode — assert >=5 correct (audit P0-3).
    3. 1-example k=16 propagating intervention — expect near-unablated logits.
    4. 1-example k=1 propagating intervention — expect logit divergence.
    5. 1-example randomized sensitivity-floor — expect something different.
    All shapes are printed at intervention boundaries.
    """
    logger.log("=" * 70)
    logger.log("  CONTROL A v2 SMOKE TEST")
    logger.log(f"  variant={variant} intervene_layer={intervene_layer}")
    logger.log(f"  ckpt={checkpoint_path}")
    logger.log("=" * 70)

    set_seed(1337)
    model, tokenizer, _sids = load_vanilla_sft(
        checkpoint_path, device, checkpoint_format=checkpoint_format,
    )
    dataset = ProsQADataset("val", tokenizer, cfg_for_dataset, {
        "bot": tokenizer.convert_tokens_to_ids("<bot>"),
        "eot": tokenizer.convert_tokens_to_ids("<eot>"),
        "latent": tokenizer.convert_tokens_to_ids("<latent>"),
        "pad": tokenizer.pad_token_id,
        "eos": tokenizer.eos_token_id,
    })
    assert len(dataset) > 0, "empty dataset"

    hook_state = InterventionHookState()
    block = model.transformer.h[intervene_layer]
    hook_handle = block.register_forward_hook(make_intervention_hook(hook_state))

    # Step 2: 10-example unablated decode.
    logger.log(f"  [step 2] 10-example k=None (unablated) decode")
    n_correct = 0
    for idx in range(min(unablated_n_examples, len(dataset))):
        item = dataset[idx]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = predict_one_propagating(
                model, tokenizer, item["q_ids"], k=None, variant=variant,
                device=device, intervene_layer=intervene_layer,
                max_new_tokens=24, hook_state=hook_state, log_shapes=(idx == 0),
                logger=logger,
            )
        correct = prosqa_answer_match(out["predicted_text"], item["answer_text"])
        n_correct += int(correct)
        logger.log(
            f"    idx={idx} gold={item['answer_text']!r} "
            f"pred={out['predicted_text'][:60]!r} correct={correct}"
        )
    logger.log(f"  unablated accuracy on {unablated_n_examples}: {n_correct}/{unablated_n_examples}")
    # Audit P0-3: require >=5 correct. Expected ~8/10 for vanilla SFT @ 81.77%.
    assert n_correct >= 5, (
        f"Smoke test unablated accuracy too low: {n_correct}/{unablated_n_examples}. "
        f"Expected ~8/10 for Round 4 vanilla SFT. Checkpoint may be mis-loaded."
    )

    # Step 3: 1-example k=16 propagating intervention.
    item0 = dataset[0]
    r_shape, c_shape, covered = _VARIANT_SHAPES[variant]
    k_max = min(r_shape, c_shape)
    logger.log(f"  [step 3] k={k_max} propagating intervention (expect near-unablated)")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out_unab = predict_one_propagating(
            model, tokenizer, item0["q_ids"], k=None, variant=variant,
            device=device, intervene_layer=intervene_layer,
            max_new_tokens=24, hook_state=hook_state,
        )
        out_kmax = predict_one_propagating(
            model, tokenizer, item0["q_ids"], k=k_max, variant=variant,
            device=device, intervene_layer=intervene_layer,
            max_new_tokens=24, hook_state=hook_state, log_shapes=True, logger=logger,
        )
    logger.log(
        f"    unablated first_tok={out_unab['first_token']!r} "
        f"pred={out_unab['predicted_text'][:60]!r}"
    )
    logger.log(
        f"    k={k_max} first_tok={out_kmax['first_token']!r} "
        f"pred={out_kmax['predicted_text'][:60]!r} "
        f"fires={out_kmax['intervene_fire_count']} "
        f"eff_rank={out_kmax['z_eff_rank']}"
    )
    # The hook should have fired max_new_tokens times (prompt + each gen step).
    assert out_kmax["intervene_fire_count"] >= 1, (
        "hook did not fire at k=k_max; hook registration broken."
    )

    # Step 4: 1-example k=1 propagating intervention.
    logger.log(f"  [step 4] k=1 propagating intervention (expect divergence)")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out_k1 = predict_one_propagating(
            model, tokenizer, item0["q_ids"], k=1, variant=variant,
            device=device, intervene_layer=intervene_layer,
            max_new_tokens=24, hook_state=hook_state, log_shapes=True, logger=logger,
        )
    logger.log(
        f"    k=1 first_tok={out_k1['first_token']!r} "
        f"pred={out_k1['predicted_text'][:60]!r} "
        f"fires={out_k1['intervene_fire_count']} "
        f"eff_rank={out_k1['z_eff_rank']}"
    )
    if out_k1.get("z_eff_rank") is not None:
        assert 0.5 < out_k1["z_eff_rank"] < 2.5, (
            f"k=1 effective_rank {out_k1['z_eff_rank']} out of range ~1; "
            f"SVD truncation may be broken."
        )

    # Step 5: randomized sensitivity-floor baseline.
    logger.log(f"  [step 5] k=random (sensitivity floor)")
    rand_gen = torch.Generator().manual_seed(1337 + 10_000)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out_rand = predict_one_propagating(
            model, tokenizer, item0["q_ids"], k="random", variant=variant,
            device=device, intervene_layer=intervene_layer,
            max_new_tokens=24, hook_state=hook_state, rand_generator=rand_gen,
        )
    logger.log(
        f"    k=random first_tok={out_rand['first_token']!r} "
        f"pred={out_rand['predicted_text'][:60]!r} "
        f"fires={out_rand['intervene_fire_count']}"
    )

    hook_handle.remove()
    logger.log("=" * 70)
    logger.log("  SMOKE TEST PASSED")
    logger.log("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Control A v2: propagating fake-Z rank-k on vanilla GPT-2 SFT."
    )
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/workspace/pebble/round4_vanilla_sft",
                        help="Directory containing pure_sft_seed{S}.pt files.")
    parser.add_argument("--ckpt-name-template", type=str,
                        default="pure_sft_seed{seed}.pt",
                        help="Filename template inside --checkpoint-dir.")
    parser.add_argument("--checkpoint-format", type=str, default=None,
                        choices=[None, "state_dict", "wrapped", "codi_wrapped"],
                        help="Disambiguation hint for the checkpoint loader.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 42, 7])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    # Attack F2 guidance: primary variant remains 16x16 (matches matrix-CODI
    # d=16 semantics). 16x48 and svd_full_768 are secondary robustness checks.
    parser.add_argument("--variant", type=str, default="16x16",
                        choices=list(_VARIANT_SHAPES.keys()))
    parser.add_argument("--intervene-layer", type=int, default=DEFAULT_INTERVENE_LAYER,
                        help=("Transformer block index (0-11) whose output is hooked "
                              "for the rank-k intervention. Default 6 leaves 6 of 12 "
                              "layers downstream to propagate the intervention."))
    parser.add_argument("--n-examples", type=int, default=500,
                        help="Cap on ProsQA eval examples per seed.")
    # Attack M2: default matches matrix-CODI ProsQA max_new=24.
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--no-randomized-control", action="store_true",
                        help="Skip the k='random' sensitivity-floor baseline.")
    parser.add_argument("--max-per-example-rows", type=int, default=50,
                        help="Cap stored per-example rows per (k, seed) in JSON "
                             "(audit P1-7).")
    parser.add_argument("--output-dir", type=str,
                        default="experiment-runs/2026-04-23_control_a/")
    parser.add_argument("--prosqa-val-path", type=str,
                        default="/workspace/pebble/round3_gamma0/data/prosqa_test.json")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test and exit.")
    parser.add_argument("--smoke-test-all-variants", action="store_true",
                        help="Run smoke test against all 3 variants (slow).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save the exact script that ran, per CLAUDE.md.
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(Path(__file__).resolve(), out_dir / "_script_used.py")
    except Exception as e:
        print(f"[warn] could not copy script: {e}", flush=True)

    logger = build_logger(out_dir, is_main=True)
    logger.log("=== Control A v2 (propagating fake-Z rank-k on vanilla SFT) ===")
    logger.log(f"device={device} variant={args.variant} ks={args.ks} seeds={args.seeds}")
    logger.log(f"intervene_layer={args.intervene_layer} max_new_tokens={args.max_new_tokens}")
    logger.log(f"output_dir={out_dir}")

    # Determinism / reproducibility log (attack minor m1).
    logger.log(
        f"  env: torch={torch.__version__} cuda={torch.version.cuda} "
        f"device={device} tf32={torch.backends.cuda.matmul.allow_tf32}"
    )
    if torch.cuda.is_available():
        logger.log(
            f"  gpu={torch.cuda.get_device_name(0)} "
            f"capability={torch.cuda.get_device_capability(0)}"
        )

    # Config for ProsQADataset.
    cfg_for_dataset = copy.deepcopy(MATRIX_CODI_CONFIG)
    cfg_for_dataset["dataset"] = "prosqa"
    cfg_for_dataset["prosqa_val_path"] = args.prosqa_val_path
    cfg_for_dataset["prosqa_train_path"] = args.prosqa_val_path  # unused but required.

    if args.smoke_test or args.smoke_test_all_variants:
        seed = args.seeds[0] if args.seeds else 1337
        ckpt = Path(args.checkpoint_dir) / args.ckpt_name_template.format(seed=seed)
        if args.smoke_test_all_variants:
            for v in list(_VARIANT_SHAPES.keys()):
                run_smoke_test(ckpt, v, device, cfg_for_dataset, logger,
                               intervene_layer=args.intervene_layer,
                               checkpoint_format=args.checkpoint_format)
        else:
            run_smoke_test(ckpt, args.variant, device, cfg_for_dataset, logger,
                           intervene_layer=args.intervene_layer,
                           checkpoint_format=args.checkpoint_format)
        logger.close()
        return

    per_seed = {}
    for seed in args.seeds:
        ckpt = Path(args.checkpoint_dir) / args.ckpt_name_template.format(seed=seed)
        try:
            result = sweep_one_seed(
                checkpoint_path=ckpt,
                seed=seed,
                ks=args.ks,
                variant=args.variant,
                n_examples=args.n_examples,
                max_new_tokens=args.max_new_tokens,
                device=device,
                logger=logger,
                cfg_for_dataset=cfg_for_dataset,
                intervene_layer=args.intervene_layer,
                run_randomized_control=not args.no_randomized_control,
                checkpoint_format=args.checkpoint_format,
                max_per_example_rows=args.max_per_example_rows,
            )
            per_seed[str(seed)] = result
        except Exception as e:
            logger.log(f"seed={seed} FAILED: {type(e).__name__}: {e}")
            per_seed[str(seed)] = {"error": f"{type(e).__name__}: {e}"}

    # Attack M5: only compute pooled-r over ks that are COMPLETE across all seeds.
    # Identify successful seeds.
    ok_seeds = [s for s in per_seed if "per_k" in per_seed[s]]
    # Determine which ks appear with "accuracy" in ALL ok_seeds.
    complete_ks = []
    for k in args.ks:
        ks_ok = all(
            "accuracy" in per_seed[s]["per_k"].get(str(k), {})
            for s in ok_seeds
        )
        if ks_ok and ok_seeds:
            complete_ks.append(k)
    logger.log(
        f"  pooled computation: {len(ok_seeds)} seeds x {len(complete_ks)} complete ks "
        f"(complete_ks={complete_ks})"
    )

    pooled_ks = []
    pooled_accs = []
    pooled_first_accs = []
    for s in ok_seeds:
        for k in complete_ks:
            rec = per_seed[s]["per_k"][str(k)]
            pooled_ks.append(k)
            pooled_accs.append(rec["accuracy"])
            pooled_first_accs.append(rec["first_token_accuracy"])

    if len(pooled_ks) >= 2 and complete_ks:
        pooled_r, pooled_p = _spearman(pooled_ks, pooled_accs)
    else:
        pooled_r, pooled_p = float("nan"), float("nan")

    # Per-k mean accuracies for the classifier.
    by_k = {}
    by_k_first = {}
    for k, a, fa in zip(pooled_ks, pooled_accs, pooled_first_accs):
        by_k.setdefault(k, []).append(a)
        by_k_first.setdefault(k, []).append(fa)
    ks_sorted = sorted(by_k.keys())
    mean_accs = [sum(by_k[k]) / len(by_k[k]) for k in ks_sorted]
    mean_first_accs = [sum(by_k_first[k]) / len(by_k_first[k]) for k in ks_sorted]
    decision = classify(
        ks_sorted, mean_accs, pooled_r,
        n_examples_per_k=args.n_examples,
        pooled_n_seeds=len(ok_seeds),
    )
    decision_first = classify(
        ks_sorted, mean_first_accs, pooled_r,
        n_examples_per_k=args.n_examples,
        pooled_n_seeds=len(ok_seeds),
    )

    # Unablated reference across seeds, for the paper-writer to cross-check
    # against Round 4's 81.77% (seed 1337).
    unablated_by_seed = {}
    randomized_by_seed = {}
    for s in ok_seeds:
        u = per_seed[s]["per_k"].get("None", {})
        if "accuracy" in u:
            unablated_by_seed[s] = u["accuracy"]
        r = per_seed[s]["per_k"].get("random", {})
        if "accuracy" in r:
            randomized_by_seed[s] = r["accuracy"]

    payload = {
        "schema_version": "control_a_v2",           # audit P1-6
        "parent_schema": "rank_projection_ablation.json (adapted; propagating-intervention v2)",
        "experiment": "control_a_propagating_vanilla_sft",
        "variant": args.variant,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_format_hint": args.checkpoint_format,
        "intervene_layer": args.intervene_layer,
        "n_analog_positions": N_ANALOG_POSITIONS,
        "n_examples": args.n_examples,
        "ks": args.ks,
        "max_new_tokens": args.max_new_tokens,
        "prosqa_val_path": args.prosqa_val_path,
        "per_seed": per_seed,
        "complete_ks_for_pool": complete_ks,
        "pooled_spearman_r": pooled_r,
        "pooled_spearman_p": pooled_p,
        "pooled_mean_accuracy_by_k": {str(k): a for k, a in zip(ks_sorted, mean_accs)},
        "pooled_mean_first_token_accuracy_by_k": {
            str(k): a for k, a in zip(ks_sorted, mean_first_accs)
        },
        "unablated_accuracy_by_seed": unablated_by_seed,
        "randomized_accuracy_by_seed": randomized_by_seed,
        "decision": decision,
        "decision_first_token": decision_first,
        "decision_rule": _DECISION_RULE,
        "intervention_semantics": (
            "propagating: forward hook on transformer.h[L_INTERVENE] output "
            "replaces hidden states at 6 analog positions (last 6 before ':') with "
            "their rank-k SVD-truncated reconstructions; each generated token "
            "triggers a full cache-less forward so downstream attention always "
            "re-attends through the intervened residual stream."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    json_path = out_dir / "rank_projection_ablation_vanilla_sft.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.log(f"wrote {json_path}")

    # SUMMARY.txt — lowercase decision (attack minor m4).
    lines = [
        "",
        "=" * 72,
        f"  CONTROL A v2 (propagating fake-Z rank-k on vanilla GPT-2 SFT, ProsQA)",
        "=" * 72,
        f"  variant:         {args.variant}",
        f"  intervene_layer: {args.intervene_layer}  (of 12 GPT-2 blocks)",
        f"  seeds:           {args.seeds}",
        f"  ks:              {args.ks}",
        f"  n_examples:      {args.n_examples}",
        f"  pooled r:        {pooled_r:.4f}  p={pooled_p:.4g}",
        f"  decision:        {decision}",
        f"  decision (1st):  {decision_first}",
        f"  decision rule:   {_DECISION_RULE}",
        "-" * 72,
        "  pooled mean full-sequence accuracy by k:",
    ]
    for k in ks_sorted:
        lines.append(f"    k={k:>2d}: {sum(by_k[k])/len(by_k[k])*100:.2f}%  "
                     f"(n_seeds={len(by_k[k])})")
    lines.append("  pooled mean first-token accuracy by k:")
    for k in ks_sorted:
        lines.append(f"    k={k:>2d}: {sum(by_k_first[k])/len(by_k_first[k])*100:.2f}%")
    lines.append("-" * 72)
    lines.append("  unablated (k=None) per seed (sanity check vs Round 4 ~81.77%):")
    for s, a in unablated_by_seed.items():
        lines.append(f"    seed={s}: {a*100:.2f}%")
    lines.append("  randomized-h (k=random) per seed (sensitivity floor):")
    for s, a in randomized_by_seed.items():
        lines.append(f"    seed={s}: {a*100:.2f}%")
    lines.append("-" * 72)
    for seed in args.seeds:
        s = per_seed.get(str(seed), {})
        if "error" in s:
            lines.append(f"  seed={seed}: ERROR {s['error']}")
            continue
        per_k = s.get("per_k", {})
        bits = []
        for k in args.ks:
            rec = per_k.get(str(k), {})
            if "accuracy" in rec:
                bits.append(f"k={k}:{rec['accuracy']*100:.2f}%")
            else:
                bits.append(f"k={k}:ERR")
        lines.append(
            f"  seed={seed}: r={s.get('spearman_r', float('nan')):.4f}  "
            + "  ".join(bits)
        )
    lines.append("=" * 72)
    lines.append("")
    summary = "\n".join(lines)
    logger.log(summary)
    with open(out_dir / "SUMMARY.txt", "w") as f:
        f.write(summary)

    logger.close()


if __name__ == "__main__":
    main()
