"""lm_attractor_probe_trackd.py -- SCALE_TRANSFER_DESIGN.md sec 6 (TRACK D,
Phase 1 ONLY -- H-measure, authorized; H-graft is NOT authorized and nothing
in this file grafts, fine-tunes, or writes gradients into any of these models).

THE QUESTION (sec 6.1): does the non-orthonormal write-geometry attractor this
project measured in its OWN from-scratch DeltaNet-LM (13-14M params, real text,
`DELTANET_RD_EXACTNESS_DESIGN.md` Wave 1 ATTRIBUTION VERDICT; K=8-48 band
0.6-4.4) exist in a fixed-recurrent-state LLM someone else trained, at ~7B
scale? Tier 3 throughout (sec 2): measurement only, no causal language.

Three models, adapted from the SAME reusable instrument
(`chunk_key_gram_stats`/`summarize_gram_records`, copied verbatim from
`lm_attractor_probe_rd.py` -- pod-safe self-contained convention, same as
`rank_utils.py`'s own copy-not-import rule -- DO NOT re-derive, fix upstream
first):

  1. RWKV-7 "Goose" -- PRIMARY per sec 6.2's decision (closest architecture
     match to our own DeltaNet: a generalized delta rule, arXiv:2503.14456).
     **Size note, decided this session, documented not silently substituted:**
     sec 6.2 named the 2.9B HF-native checkpoint as the primary pick ("largest
     available... 2.9B HF-native"). That EXACT checkpoint
     (`RWKV/RWKV7-Goose-World3-2.9B-HF`) was tried first and is BROKEN in this
     box's environment (fla==0.5.1 + transformers==5.12.1): `from_pretrained`
     reports `x_r/x_w/x_k/x_v/x_a/x_g` (the six token-shift mixing coefficients,
     `fla/layers/rwkv7.py`'s own `self.x_r`, etc.) as MISSING and a spurious
     `x_x` key as UNEXPECTED for roughly 20 of its 32 layers, in BOTH bf16 and
     fp32 -- and the resulting "freshly initialized" replacements for those
     MISSING params are themselves NaN/Inf (133 NaN + 5 Inf params in bf16, 55
     NaN in fp32; logits NaN in both dtypes), most likely because transformers'
     modern meta-device "fast init" path never re-invokes this custom
     architecture's own `_initialize_weights` hook for keys it silently drops.
     Confirmed NOT a systemic fla/transformers incompatibility: the SAME
     environment loads `RWKV/RWKV7-Goose-World2.9-0.4B-HF` (24 layers) and
     `RWKV/RWKV7-Goose-World3-1.5B-HF` (24 layers) with ZERO NaN/Inf params,
     zero MISSING/UNEXPECTED keys, and finite logits -- i.e. this is specific
     to the 2.9B checkpoint's own upload/conversion, not this project's
     environment or the RWKV-7 architecture. **Decision: RWKV-7 1.5B
     (`RWKV/RWKV7-Goose-World3-1.5B-HF`) substitutes as the RWKV-7 measurement
     point** -- a documented downgrade from the design's literal 2.9B pick,
     not a silent one; Falcon-Mamba-7B below supplies the genuine 7B-scale
     reading the design also wanted.
     State-update equation AS FOUND in `fla/layers/rwkv7.py` (read this
     session, NOT assumed from the paper): per-head recurrence via
     `fla.ops.rwkv7.chunk_rwkv7` -> `fla.ops.generalized_delta_rule.
     chunk_dplr_delta_rule` (DPLR = "diagonal plus low-rank" delta rule):
         S_t = S_{t-1} @ (diag(w_t) + a_t (x) b_t) + v_t (x) k_t
     where (module names/lines as shipped): `k = self.k_proj(xk)` (plain
     `nn.Linear`, NOT normalized -- this is the tensor `v_t (x) k_t` writes
     with); `kk = normalize(rearrange(k * self.k_k, ...), dim=-1)` (the SAME
     raw `k_proj` output, times a learned per-channel gate `self.k_k`, THEN
     L2-normalized per head -- this is the tensor that forms the erase/decay
     correction `a_t=-kk, b_t=kk*a_gate`). **This is not textbook DeltaNet**:
     vanilla DeltaNet writes and erases with the SAME normalized key
     (`S_t=S_{t-1}(I-beta*k*k^T)+beta*v*k^T`); RWKV-7 splits these into two
     different tensors (raw-gated `k` for the value write, separately
     L2-normalized `kk` for the decay/erase low-rank correction) and replaces
     the identity decay with a learned diagonal `diag(w_t)`. This probe
     measures **`kk`** (the L2-normalized one) as the write-geometry analog,
     matching this project's own convention ("Gram deviation ... applied to
     the already L2-normalized key side", `model_rd.py`) -- reconstructed
     OUTSIDE the model from a forward hook on `attn.k_proj` (a real
     `nn.Linear`, never monkeypatched) plus the model's own `k_k` parameter,
     never assumed equal to the paper's bare equations.

  2. Falcon-Mamba-7B -- SECONDARY per sec 6.2, true 7B scale
     (`tiiuae/falcon-mamba-7b`, native `transformers.FalconMambaForCausalLM`,
     no external package / no trust_remote_code -- far lower version-skew
     risk than RWKV-7's community `fla` package, confirmed empirically this
     session: loads with ZERO NaN/Inf params on the first try). State-update
     equation AS FOUND in `transformers/models/falcon_mamba/
     modeling_falcon_mamba.py` (a diagonal-gated SSM, NOT a delta rule at
     all -- confirmed, not assumed): per-channel recurrence
         h_t[c] = exp(dt_t[c] * A[c]) * h_{t-1}[c] + dt_t[c] * B_t * u_t[c]
     (`discrete_A = exp(dt[...,None] * A)`, `deltaB_u = discrete_B *
     hidden_states[...,None]`, `ssm_state = discrete_A*ssm_state + deltaB_u`)
     with a SHARED (non-per-channel) write vector `B_t` of dim
     `state_size=16` per token position, and Falcon-Mamba's own distinguishing
     addition over vanilla Mamba: `B = rms_forward(B, variance_epsilon=
     self.rms_eps)` -- an unlearned (no weight) RMS-normalization of `B`
     (and `C`) the base Mamba paper does not have (`rms_forward`'s own
     docstring: "simple RMSNorm with no learnable weights"). `B_t` (post
     `rms_forward`, pre-discretization-by-`dt`) is this probe's key/write
     analog, reconstructed from a forward hook on `mixer.x_proj` (a real
     `nn.Linear` whose raw output is split into `(time_step, B, C)`) plus the
     mixer's own `rms_forward`, copied verbatim from the shipped source (not
     re-derived). **Capacity caveat, load-bearing:** `state_size=16` is far
     smaller than this project's own `d_state=64` (or RWKV-7's `head_dim=64`)
     -- at most 16 mutually orthonormal 16-dim directions can EVER exist, so
     any chunk/window wider than 16 tokens is STRUCTURALLY forced to a
     nonzero Gram deviation regardless of learned geometry. This probe reports
     BOTH the cross-model default window (64, for comparability) AND a
     capacity-matched window (16, for a fair reading) -- see `--chunk-sizes`.

  3. Qwen2.5-1.5B -- REGISTERED NEGATIVE CONTROL, resolving
     `SCALE_TRANSFER_DESIGN.md` sec 12 Q4 (left OPEN/deferred by Rev 2, not
     silently dropped): "a standard softmax-attention Transformer of
     comparable scale, which has no fixed recurrent state at all[,] to
     calibrate what 'no attractor' would even look like under this
     measurement protocol." **Minimal registered choice, decided and recorded
     HERE, this session:** `Qwen/Qwen2.5-1.5B` -- Apache-2.0, fully open,
     native `transformers.Qwen2ForCausalLM` (zero version-skew risk), and
     "comparable scale" is read as comparable to THIS session's actual RWKV-7
     measurement point (1.5B, per the size substitution above), which is the
     model this negative control is meant to calibrate against most directly.
     Standard grouped-query attention (`num_attention_heads=12,
     num_key_value_heads=2, head_dim=128`) -- genuinely unbounded context via
     a growing KV cache, no fixed-size recurrent state of any kind, so under
     H-measure's own framing (sec 6.6) a near-orthonormal (or at least a
     structurally DIFFERENT) reading here is the expected null. Measured
     quantity: `self_attn.k_proj`'s raw output (a forward hook on a real
     `nn.Linear`, pre-RoPE). **Documented limitation:** RoPE rotates each
     position's key by a position-dependent orthogonal transform AFTER
     `k_proj`; this probe deliberately measures the PRE-RoPE "content" key
     (the direct analog of RWKV-7's raw `k_proj` output and Falcon-Mamba's
     pre-discretization `B_t` -- i.e. the model's own LEARNED key geometry,
     not a positional-encoding artifact) and does not claim anything about
     the POST-RoPE keys actually consumed by attention, which would look
     different (RoPE is norm-preserving but not identity, so Gram deviation
     computed after it would differ from what is reported here).

Non-invasive throughout: every captured tensor comes from a `register_
forward_hook` on a real, already-existing `nn.Module` (`k_proj` / `x_proj`),
identical in spirit to `lm_attractor_probe_rd.py`'s own hook on `k_conv1d`.
No model code is edited, no gradients are computed anywhere in this file
(`torch.no_grad()` throughout), no fine-tuning, no graft (sec 6.4 NOT
authorized, NOT attempted).

Usage:
  python lm_attractor_probe_trackd.py --smoke
  python lm_attractor_probe_trackd.py --check-real rwkv7        # Wave-(-1)-style health gate only
  python lm_attractor_probe_trackd.py --models rwkv7 falcon_mamba qwen_control \\
      --out results/lm_rd_trackd/attractor_probe_trackd.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports

from rank_utils import effective_rank, stable_rank

DEFAULT_HF_HOME = "/data/hf_cache"
DEFAULT_OUT_DIR = "results/lm_rd_trackd"

MIN_VALID_FOR_GRAM = 2   # copied convention from lm_attractor_probe_rd.py: a single
                          # item's own Gram-vs-identity deviation is trivially 0 by
                          # construction -- excluded (gram_deviation=None), never
                          # silently averaged as 0.

REFERENCE_BAND_14M = {
    "source": "DELTANET_RD_EXACTNESS_DESIGN.md Wave 1 ATTRIBUTION VERDICT / "
              "SCALE_TRANSFER_DESIGN.md sec 1 (item 3); STATE.md 'Chapter 2 -- STATUS'",
    "n_params": 12_899_841,
    "K_range": "8-48",
    "gram_deviation_range": [0.6, 4.4],
    "note": "this project's OWN from-scratch DeltaNet-LM, real text, K=8-48 episode "
             "sizes -- NOT directly numerically comparable to the pretrained models "
             "below (different d_state/head_dim, different corpora, different probe "
             "convention in places, see per-model caveats) but the qualitative "
             "reference point Track D's H-measure asks about.",
}


# ---------------------------------------------------------------------------
# gram_deviation -- copied VERBATIM from model_rd.py (pod-safe convention, see
# rank_utils.py's own header for the precedent: copy with attribution, do not
# cross-import a heavy project-specific module just for one 5-line function).
# ---------------------------------------------------------------------------

def gram_deviation(eff: torch.Tensor) -> torch.Tensor:
    """model_rd.py's C16 instrument: ||Eff^T Eff - I||_F per batch row.
    eff: (B, K, d)."""
    gram = eff @ eff.transpose(-1, -2)
    n = eff.shape[1]
    eye = torch.eye(n, device=eff.device, dtype=gram.dtype).expand_as(gram)
    return (gram - eye).norm(dim=(-2, -1))


# ---------------------------------------------------------------------------
# chunk_key_gram_stats / summarize_gram_records -- copied VERBATIM from
# lm_attractor_probe_rd.py (already smoke-tested there; reused unmodified so
# that test coverage transfers) with ONE generalization: chunk_size no longer
# needs to divide T exactly -- Track D's models don't share Track B/C's
# chunk_delta_rule tiling constant, so a trailing partial chunk is kept
# (never silently dropped) rather than asserted against.
# ---------------------------------------------------------------------------

def chunk_key_gram_stats(k_raw: torch.Tensor, content_mask: torch.Tensor, num_heads: int,
                          chunk_size: int, min_valid: int = MIN_VALID_FOR_GRAM) -> list[dict]:
    """k_raw: (B,T,D) RAW captured tensor. content_mask: (B,T) bool, True=
    eligible (non-EOS/pad).

    Computes TWO variants per episode, added THIS session after an empirical
    check (qwen_key_diag.py / outlier_check_rwkv_mamba.py, run against all
    three real model families before trusting any number) found a dominant,
    near-input-agnostic "massive activation" channel (Sun et al. 2024,
    arXiv:2402.17762, and its 2025-26 follow-ons -- verified via WebSearch
    this session, not assumed) in EVERY family measured here (Qwen's raw
    k_proj, RWKV-7's kk, Falcon-Mamba's B_t alike -- 3-35x the median
    per-channel magnitude, mean pairwise cosine 0.43-0.9998), which by itself
    drives raw Gram-deviation close to the theoretical fully-collapsed
    ceiling regardless of any genuine write-geometry structure. This is a
    GENERIC, already-published property of trained networks, not something
    specific to fixed-recurrent-state write geometry -- and critically, the
    registered negative control (sec 12 Q4, no fixed state at all) shows the
    SAME pattern, meaning raw Gram-deviation ALONE cannot discriminate
    H-measure's actual question. Two variants are reported for every episode
    so neither reading is silently preferred:
      'raw'      -- L2-normalize, then Gram-deviation (as originally
                    specified, sec 6.3 item 3; kept for full transparency).
      'centered' -- subtract the PER-CHANNEL MEAN ACROSS THE EPISODE (the
                    standard, minimal fix for a shared/input-agnostic outlier
                    direction: centering removes exactly a component that is
                    constant across positions, which is what a 'largely
                    input-agnostic' massive-activation channel is), THEN
                    L2-normalize and Gram-deviate. This is the more
                    informative reading for H-measure's actual question (does
                    a write-geometry attractor exist BEYOND the generic
                    outlier-channel effect); 'raw' is kept alongside it, not
                    replaced, for transparency and reproducibility.
    Both variants use the SAME uniform self-normalization convention across
    all three model families (idempotent where the model already normalizes,
    e.g. RWKV-7's kk) -- matches model_rd.py's/lm_attractor_probe_rd.py's own
    "applied to the already L2-normalized key side" convention for 'raw'.

    Returns one record per (batch-row, chunk, head) episode."""
    B, T, D = k_raw.shape
    assert D % num_heads == 0, f"D={D} not divisible by num_heads={num_heads}"
    n_chunks = -(-T // chunk_size)   # ceil -- trailing partial chunk kept, not dropped
    head_dim = D // num_heads

    # PERFORMANCE NOTE (audit finding MAJOR-2, this session): the original version of this
    # function looped over heads in Python, one tiny SVD call at a time -- empirically
    # measured (RWKV-7 1.5B, num_heads~32) to be the dominant cost, far more than the
    # forward pass itself. Vectorized below: all heads of a given (batch-row, chunk) are
    # stacked into ONE leading batch dimension and passed to gram_deviation/effective_rank/
    # stable_rank in a SINGLE batched call (these already accept an arbitrary leading batch
    # dim via torch.linalg.svdvals) instead of `num_heads` separate Python-loop calls.
    # Per-episode SEMANTICS are unchanged -- this is a speed-only refactor, verified against
    # the original per-head loop in this session's audit re-check (see smoke item [1]-[5],
    # unchanged, plus a dedicated multi-head equivalence check, item [5b]).
    records = []
    for b in range(B):
        for c in range(n_chunks):
            lo, hi = c * chunk_size, min((c + 1) * chunk_size, T)
            mask_slice = content_mask[b, lo:hi]
            valid_idx = mask_slice.nonzero(as_tuple=True)[0]
            n_valid = int(valid_idx.numel())
            if n_valid < min_valid:
                for h in range(num_heads):
                    records.append({"b": b, "chunk": c, "head": h, "n_valid": n_valid,
                                     "gram_deviation": None, "effective_rank": None, "stable_rank": None,
                                     "gram_deviation_centered": None, "effective_rank_centered": None,
                                     "stable_rank_centered": None, "top_channel_ratio": None})
                continue
            k_chunk = k_raw[b, lo:hi].view(hi - lo, num_heads, head_dim)
            keys = k_chunk[valid_idx, :, :].permute(1, 0, 2)              # (num_heads, n_valid, head_dim)

            # 'raw' variant -- ALL heads in one batched call
            keys_n = F.normalize(keys, dim=-1, eps=1e-8)
            gd = gram_deviation(keys_n)                                   # (num_heads,)
            er = effective_rank(keys_n)
            sr = stable_rank(keys_n)

            # 'centered' variant -- remove the per-channel mean across THIS episode first
            keys_c = keys - keys.mean(dim=1, keepdim=True)
            keys_cn = F.normalize(keys_c, dim=-1, eps=1e-8)
            gd_c = gram_deviation(keys_cn)
            er_c = effective_rank(keys_cn)
            sr_c = stable_rank(keys_cn)

            # massive-activation signature, reported directly (not just asserted)
            chan_meanabs = keys.abs().mean(dim=1)                         # (num_heads, head_dim)
            top_ratio = chan_meanabs.max(dim=-1).values / chan_meanabs.median(dim=-1).values.clamp(min=1e-8)

            gd, er, sr = gd.tolist(), er.tolist(), sr.tolist()
            gd_c, er_c, sr_c = gd_c.tolist(), er_c.tolist(), sr_c.tolist()
            top_ratio = top_ratio.tolist()
            for h in range(num_heads):
                records.append({"b": b, "chunk": c, "head": h, "n_valid": n_valid,
                                 "gram_deviation": gd[h], "effective_rank": er[h], "stable_rank": sr[h],
                                 "gram_deviation_centered": gd_c[h], "effective_rank_centered": er_c[h],
                                 "stable_rank_centered": sr_c[h], "top_channel_ratio": top_ratio[h]})
    return records


def summarize_gram_records(records: list[dict]) -> dict:
    valid = [r for r in records if r["gram_deviation"] is not None]
    n_excluded = len(records) - len(valid)
    if not valid:
        return {"n_episodes": len(records), "n_excluded_below_min_valid": n_excluded,
                "n_scored": 0, "note": "every episode had n_valid < MIN_VALID_FOR_GRAM"}
    gd = torch.tensor([r["gram_deviation"] for r in valid])
    er = torch.tensor([r["effective_rank"] for r in valid])
    sr = torch.tensor([r["stable_rank"] for r in valid])
    gd_c = torch.tensor([r["gram_deviation_centered"] for r in valid])
    er_c = torch.tensor([r["effective_rank_centered"] for r in valid])
    sr_c = torch.tensor([r["stable_rank_centered"] for r in valid])
    top_ratio = torch.tensor([r["top_channel_ratio"] for r in valid])
    return {
        "n_episodes": len(records), "n_excluded_below_min_valid": n_excluded, "n_scored": len(valid),
        "gram_deviation_mean": gd.mean().item(), "gram_deviation_std": gd.std(unbiased=False).item(),
        "gram_deviation_min": gd.min().item(), "gram_deviation_max": gd.max().item(),
        "effective_rank_mean": er.mean().item(), "stable_rank_mean": sr.mean().item(),
        "gram_deviation_centered_mean": gd_c.mean().item(),
        "gram_deviation_centered_std": gd_c.std(unbiased=False).item(),
        "gram_deviation_centered_min": gd_c.min().item(), "gram_deviation_centered_max": gd_c.max().item(),
        "effective_rank_centered_mean": er_c.mean().item(), "stable_rank_centered_mean": sr_c.mean().item(),
        "top_channel_ratio_mean": top_ratio.mean().item(), "top_channel_ratio_max": top_ratio.max().item(),
    }


# ---------------------------------------------------------------------------
# Health gate -- MANDATORY before trusting any real checkpoint's numbers.
# Added THIS session after RWKV-7 2.9B silently loaded with 133 NaN/5 Inf
# params (see module docstring). Never skipped for a non-smoke run.
# ---------------------------------------------------------------------------

def health_gate(model: torch.nn.Module, model_id: str) -> dict:
    nan_params = [n for n, p in model.named_parameters() if torch.isnan(p).any()]
    inf_params = [n for n, p in model.named_parameters() if torch.isinf(p).any()]
    n_total = sum(1 for _ in model.named_parameters())
    report = {"model_id": model_id, "n_params_named": n_total,
              "n_nan_params": len(nan_params), "n_inf_params": len(inf_params),
              "nan_param_names_sample": nan_params[:10], "inf_param_names_sample": inf_params[:10]}
    if nan_params or inf_params:
        raise RuntimeError(
            f"HEALTH GATE FAILED for {model_id}: {len(nan_params)} NaN params, "
            f"{len(inf_params)} Inf params (of {n_total}). Sample NaN: {nan_params[:5]}. "
            f"Sample Inf: {inf_params[:5]}. Refusing to measure a corrupted checkpoint -- "
            f"see module docstring for the RWKV-7 2.9B precedent this gate exists to catch.")
    return report


def logits_finite_gate(logits: torch.Tensor, model_id: str) -> None:
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        raise RuntimeError(f"HEALTH GATE FAILED for {model_id}: forward-pass logits contain "
                            f"NaN/Inf on a real text batch (params passed the param-level gate, "
                            f"but the forward pass itself produced non-finite output).")


# ---------------------------------------------------------------------------
# Corpus loading -- reuses this project's own two corpora (continuity with
# Track B/C's instrument set, sec 0's reading list) but RE-TOKENIZES with each
# target model's OWN tokenizer (the pre-tokenized GPT-2 .pt files under
# /data/deltanet_rd_data are meaningless to RWKV-7/Falcon-Mamba/Qwen's own
# vocabularies -- this is NOT a silent divergence, tokenization is genuinely
# model-specific and there is no way around re-tokenizing raw text here).
# ---------------------------------------------------------------------------

def _sample_openr1_text(n_docs: int, seed: int) -> str:
    from datasets import load_dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default")["train"]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_docs].tolist()
    docs = [ds[i]["problem"] + "\n" + ds[i]["solution"] for i in idx]
    return docs


def _sample_wikitext_text(n_docs: int, seed: int) -> list[str]:
    from datasets import load_dataset
    rows = list(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")["validation"]["text"])
    # simple non-empty-run segmentation: group consecutive non-empty rows into
    # pseudo-articles (does not need rebuild_lm_corpora_rd.py's exact heading
    # heuristic -- this probe only needs SOME natural document boundaries to
    # insert EOS separators at, not a validated article count).
    docs, cur = [], []
    for r in rows:
        if r.strip() == "":
            if cur:
                docs.append("".join(cur))
                cur = []
        else:
            cur.append(r)
    if cur:
        docs.append("".join(cur))
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(docs), generator=g)[:n_docs].tolist()
    return [docs[i] for i in idx]


def _windows_from_stream(stream: torch.Tensor, n_windows: int, seq_len: int, seed: int,
                          eos_id: int | None) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Core windowing logic, factored out of build_windows so it is directly unit-testable
    without a real tokenizer/dataset (audit finding, this session: the repeat-fallback
    branch had NO smoke coverage). Returns (windows, content_mask, info) -- see
    build_windows's docstring for what `info` reports and why."""
    stream_numel_pre_repeat = stream.numel()
    need = n_windows * seq_len
    reps = 1
    if stream.numel() < need:
        # repeat the stream (documented, not silently truncating n_windows) --
        # cheap measurement corpora at n_docs~60 are usually already long enough;
        # this only fires for small smoke-scale configs. Reported below, not silent.
        reps = -(-need // stream.numel())
        stream = stream.repeat(reps)

    g = torch.Generator().manual_seed(seed + 17)
    max_start = stream.numel() - seq_len
    starts = torch.randint(0, max_start + 1, (n_windows,), generator=g)
    windows = torch.stack([stream[s:s + seq_len] for s in starts.tolist()])
    content_mask = torch.ones_like(windows, dtype=torch.bool)
    if eos_id is not None:
        content_mask &= (windows != eos_id)

    # direct empirical duplicate-window check (audit MAJOR-1) -- not just the theoretical
    # periodicity argument: compare every pair of realized windows for exact equality.
    n_dup_pairs = 0
    for i in range(n_windows):
        for j in range(i + 1, n_windows):
            if torch.equal(windows[i], windows[j]):
                n_dup_pairs += 1

    info = {
        "stream_numel_pre_repeat": stream_numel_pre_repeat, "repeat_fired": reps > 1,
        "reps": reps, "n_windows": n_windows, "seq_len": seq_len,
        "n_duplicate_window_pairs": n_dup_pairs,
    }
    return windows, content_mask, info


def build_windows(tokenizer, corpus_name: str, n_windows: int, seq_len: int, seed: int,
                   n_docs: int = 60) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Returns (token_ids: (n_windows, seq_len) int64, content_mask: (n_windows, seq_len) bool,
    info: dict -- diagnostic fields added per audit finding MAJOR-1 (independent audit,
    this session): the repeat-fallback in `_windows_from_stream` makes the stream exactly
    periodic, which COULD silently produce duplicate windows (deflating effective sample
    size with zero visible signal). `info` always reports whether the fallback fired AND
    the actual number of exact-duplicate window pairs found by direct comparison (not just
    the theoretical periodicity argument) -- so a silent trigger is visible in the output
    JSON rather than assumed-away."""
    if corpus_name == "openr1":
        docs = _sample_openr1_text(n_docs, seed)
    elif corpus_name == "wikitext":
        docs = _sample_wikitext_text(n_docs, seed)
    else:
        raise ValueError(f"unknown corpus_name={corpus_name!r}")

    eos_id = tokenizer.eos_token_id
    used_pad_fallback = False
    if eos_id is None:
        eos_id = tokenizer.pad_token_id  # fallback -- reported in `info`, not silent
        used_pad_fallback = True
    sep_ids = [eos_id] if eos_id is not None else []

    all_ids: list[int] = []
    for doc in docs:
        ids = tokenizer(doc, add_special_tokens=False)["input_ids"]
        all_ids.extend(ids)
        all_ids.extend(sep_ids)
    stream = torch.tensor(all_ids, dtype=torch.int64)

    windows, content_mask, info = _windows_from_stream(stream, n_windows, seq_len, seed, eos_id)
    info["n_docs_sampled"] = n_docs
    info["used_pad_token_fallback_for_eos"] = used_pad_fallback
    return windows, content_mask, info


# ---------------------------------------------------------------------------
# Per-family adapters
# ---------------------------------------------------------------------------

class FamilyAdapter:
    """Base contract: load() returns (model, tokenizer); register_hooks(model)
    returns (handles, captured_dict_factory); reconstruct(model, layer_idx,
    captured) returns the (B,T,D)-shaped 'write key' tensor for that layer plus
    its num_heads for chunk_key_gram_stats."""
    name: str
    model_id: str
    trust_remote_code: bool = False

    def load(self, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code, dtype=torch.bfloat16)
        model = model.to(device).eval()
        tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        return model, tok

    def num_layers(self, model) -> int:
        raise NotImplementedError

    def register_hook(self, model, layer_idx: int, captured: dict):
        raise NotImplementedError

    def reconstruct(self, model, layer_idx: int, captured: dict) -> tuple[torch.Tensor, int]:
        """Returns (write_key_tensor (B,T,D), num_heads)."""
        raise NotImplementedError

    def architecture_note(self) -> dict:
        raise NotImplementedError


class RWKV7Adapter(FamilyAdapter):
    name = "rwkv7"
    model_id = "RWKV/RWKV7-Goose-World3-1.5B-HF"
    trust_remote_code = True

    def num_layers(self, model) -> int:
        return len(model.model.layers)

    def register_hook(self, model, layer_idx: int, captured: dict):
        attn = model.model.layers[layer_idx].attn
        def hook(module, inp, out):
            captured[layer_idx] = out.detach()
        return attn.k_proj.register_forward_hook(hook)

    def reconstruct(self, model, layer_idx: int, captured: dict):
        attn = model.model.layers[layer_idx].attn
        k_raw = captured[layer_idx]                      # (B,T,hidden_size)
        head_dim, num_heads = attn.head_dim, attn.num_heads
        kk = F.normalize((k_raw * attn.k_k).view(*k_raw.shape[:2], num_heads, head_dim),
                          dim=-1, p=2.0)                  # EXACT model formula, fla/layers/rwkv7.py
        B, T = k_raw.shape[:2]
        return kk.reshape(B, T, num_heads * head_dim), num_heads

    def architecture_note(self) -> dict:
        return {
            "family": "rwkv7", "role": "PRIMARY (size-substituted, documented)",
            "hf_repo": self.model_id,
            "hook_point": "model.model.layers[i].attn.k_proj (nn.Linear, real submodule)",
            "reconstruction": "kk = L2normalize((k_proj_out * attn.k_k).view(B,T,H,head_dim), dim=-1) "
                                "-- verbatim from fla/layers/rwkv7.py RWKV7Attention.forward, "
                                "non-fuse_norm branch (config.fuse_norm=False confirmed for this ckpt)",
            "state_update_equation": "S_t = S_{t-1} @ (diag(w_t) + a_t(x)b_t) + v_t(x)k_t "
                                      "[a_t=-kk, b_t=kk*sigmoid(a_lora(...))] -- via "
                                      "fla.ops.generalized_delta_rule.chunk_dplr_delta_rule",
            "deviation_from_textbook_deltanet": "write key (raw k_proj*a-gate) and erase key (kk, "
                                                  "L2-normalized) are DIFFERENT tensors here, and decay "
                                                  "is a learned diag(w_t), not identity -- textbook "
                                                  "DeltaNet uses the SAME normalized key for both and "
                                                  "an identity decay.",
        }


class FalconMambaAdapter(FamilyAdapter):
    name = "falcon_mamba"
    model_id = "tiiuae/falcon-mamba-7b"
    trust_remote_code = False

    def num_layers(self, model) -> int:
        return len(model.backbone.layers)

    def register_hook(self, model, layer_idx: int, captured: dict):
        mixer = model.backbone.layers[layer_idx].mixer
        def hook(module, inp, out):
            captured[layer_idx] = out.detach()
        return mixer.x_proj.register_forward_hook(hook)

    def reconstruct(self, model, layer_idx: int, captured: dict):
        mixer = model.backbone.layers[layer_idx].mixer
        ssm_params = captured[layer_idx]                 # (B,T, time_step_rank+2*state_size)
        tsr, ss = mixer.time_step_rank, mixer.ssm_state_size
        _, B_raw, _C = torch.split(ssm_params, [tsr, ss, ss], dim=-1)
        eps = getattr(mixer, "rms_eps", 1e-6)
        # rms_forward, copied verbatim from modeling_falcon_mamba.py (no learnable weight)
        B32 = B_raw.to(torch.float32)
        variance = B32.pow(2).mean(-1, keepdim=True)
        B_t = (B32 * torch.rsqrt(variance + eps)).to(B_raw.dtype)
        return B_t, 1   # shared across channels -- no head dimension, num_heads=1

    def architecture_note(self) -> dict:
        return {
            "family": "falcon_mamba", "role": "SECONDARY (true 7B scale)",
            "hf_repo": self.model_id,
            "hook_point": "model.backbone.layers[i].mixer.x_proj (nn.Linear, real submodule)",
            "reconstruction": "B_t = rms_forward(split(x_proj_out)[1], eps=mixer.rms_eps) -- verbatim "
                                "from transformers/models/falcon_mamba/modeling_falcon_mamba.py",
            "state_update_equation": "h_t[c] = exp(dt_t[c]*A[c])*h_{t-1}[c] + dt_t[c]*B_t*u_t[c] "
                                      "(per-channel diagonal-gated SSM, NOT a delta rule)",
            "deviation_from_textbook_deltanet": "no erase term at all (pure additive/decayed write, "
                                                  "no rank-1 subtraction); B_t is SHARED across all "
                                                  "intermediate_size channels at a position (no head "
                                                  "dim); Falcon-Mamba adds an unlearned RMS-normalization "
                                                  "of B/C that vanilla Mamba does not have.",
            "capacity_caveat": f"state_size={16} is far below this project's own d_state=64 -- "
                                 "windows >16 are structurally non-orthonormalizable regardless of "
                                 "learned geometry; report both chunk_size=64 (comparability) and "
                                 "chunk_size=16 (capacity-matched) explicitly.",
        }


class Qwen2NegControlAdapter(FamilyAdapter):
    name = "qwen_control"
    model_id = "Qwen/Qwen2.5-1.5B"
    trust_remote_code = False

    def num_layers(self, model) -> int:
        return len(model.model.layers)

    def register_hook(self, model, layer_idx: int, captured: dict):
        attn = model.model.layers[layer_idx].self_attn
        def hook(module, inp, out):
            captured[layer_idx] = out.detach()
        return attn.k_proj.register_forward_hook(hook)

    def reconstruct(self, model, layer_idx: int, captured: dict):
        attn = model.model.layers[layer_idx].self_attn
        k_raw = captured[layer_idx]                       # (B,T, num_kv_heads*head_dim), PRE-RoPE
        num_heads = attn.config.num_key_value_heads if hasattr(attn, "config") else \
            model.config.num_key_value_heads
        return k_raw, num_heads

    def architecture_note(self) -> dict:
        return {
            "family": "qwen_control", "role": "REGISTERED NEGATIVE CONTROL (resolves sec 12 Q4)",
            "hf_repo": self.model_id,
            "hook_point": "model.model.layers[i].self_attn.k_proj (nn.Linear, real submodule), PRE-RoPE",
            "reconstruction": "raw k_proj output, reshaped to (B,T,num_key_value_heads,head_dim); "
                                "NOT rotated by RoPE (documented limitation, see module docstring)",
            "state_update_equation": "NONE -- standard softmax attention over a growing KV cache, "
                                      "no fixed-size recurrent state of any kind. This IS the point: "
                                      "calibrates what this instrument reads on a model architecturally "
                                      "incapable of having a write-geometry attractor in the sense "
                                      "measured for the other two families.",
            "deviation_from_textbook_deltanet": "not applicable -- not a delta-rule-family model.",
        }


ADAPTERS = {a.name: a for a in [RWKV7Adapter(), FalconMambaAdapter(), Qwen2NegControlAdapter()]}


# ---------------------------------------------------------------------------
# Per-model measurement
# ---------------------------------------------------------------------------

def measure_model(adapter: FamilyAdapter, device: str, corpora: list[str], chunk_sizes: list[int],
                   n_windows: int, seq_len: int, seed: int) -> dict:
    t0 = time.time()
    model, tok = adapter.load(device)
    health = health_gate(model, adapter.model_id)
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = adapter.num_layers(model)

    per_corpus = {}
    for corpus_name in corpora:
        windows, content_mask, window_info = build_windows(tok, corpus_name, n_windows, seq_len, seed)
        if window_info["repeat_fired"] or window_info["n_duplicate_window_pairs"] > 0:
            print(f"  WARNING [{corpus_name}]: window_info={window_info} -- repeat fallback fired "
                  f"and/or duplicate windows found, effective sample size may be smaller than "
                  f"n_windows={n_windows} (see audit MAJOR-1)")
        windows, content_mask = windows.to(device), content_mask.to(device)

        captured = {}
        handles = [adapter.register_hook(model, i, captured) for i in range(n_layers)]
        try:
            with torch.no_grad():
                out = model(windows, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        logits_finite_gate(out.logits, adapter.model_id)

        per_chunk_size = {}
        for cs in chunk_sizes:
            per_layer = {}
            for layer_idx in range(n_layers):
                write_key, num_heads = adapter.reconstruct(model, layer_idx, captured)
                records = chunk_key_gram_stats(write_key.float(), content_mask, num_heads, cs)
                per_layer[layer_idx] = summarize_gram_records(records)
            # pooled-across-layers summary for this (corpus, chunk_size) -- both variants
            # (raw, centered -- see chunk_key_gram_stats docstring for why centered is the
            # more informative reading once the massive-activation confound is controlled)
            all_gd = [v["gram_deviation_mean"] for v in per_layer.values() if v.get("n_scored", 0) > 0]
            all_gd_c = [v["gram_deviation_centered_mean"] for v in per_layer.values() if v.get("n_scored", 0) > 0]
            all_top_ratio = [v["top_channel_ratio_mean"] for v in per_layer.values() if v.get("n_scored", 0) > 0]
            per_chunk_size[cs] = {
                "per_layer": per_layer,
                "pooled_across_layers_gram_deviation_mean":
                    (sum(all_gd) / len(all_gd)) if all_gd else None,
                "pooled_across_layers_gram_deviation_centered_mean":
                    (sum(all_gd_c) / len(all_gd_c)) if all_gd_c else None,
                "pooled_across_layers_top_channel_ratio_mean":
                    (sum(all_top_ratio) / len(all_top_ratio)) if all_top_ratio else None,
                "n_layers_scored": len(all_gd), "n_layers_total": n_layers,
            }
        per_corpus[corpus_name] = {"per_chunk_size": per_chunk_size, "n_windows": n_windows,
                                     "seq_len": seq_len, "window_info": window_info}
        del captured
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "architecture": adapter.architecture_note(),
        "health_gate": health, "n_params": n_params, "n_layers": n_layers,
        "per_corpus": per_corpus, "wall_s": time.time() - t0,
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_all(model_names: list[str], device: str, corpora: list[str], chunk_sizes: list[int],
            n_windows: int, seq_len: int, seed: int) -> dict:
    models_out = {}
    for name in model_names:
        adapter = ADAPTERS[name]
        print(f"\n{'='*70}\nMEASURING {name} ({adapter.model_id})\n{'='*70}")
        try:
            models_out[name] = measure_model(adapter, device, corpora, chunk_sizes, n_windows, seq_len, seed)
            print(f"  OK -- wall_s={models_out[name]['wall_s']:.1f}")
        except Exception as e:
            models_out[name] = {"FAILED": True, "error": str(e), "architecture": adapter.architecture_note()}
            print(f"  FAILED: {e}")
    return {
        "design_ref": "SCALE_TRANSFER_DESIGN.md sec 6 (TRACK D, Phase 1, H-measure, Tier 3)",
        "reference_band_14m_project": REFERENCE_BAND_14M,
        "chunk_sizes": chunk_sizes, "n_windows": n_windows, "seq_len": seq_len,
        "corpora": corpora, "seed": seed,
        "measurement_convention_note": (
            "Every write-key tensor is L2-normalized by THIS probe before Gram-deviation, uniformly "
            "across all three families, regardless of whether the model itself already normalizes it "
            "(idempotent where it does). Content mask excludes the tokenizer's own EOS id (or pad id "
            "fallback) at document-boundary positions, never silently averaged in."),
        "models": models_out,
    }


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------

def smoke():
    print("=" * 60 + "\n  LM_ATTRACTOR_PROBE_TRACKD SMOKE GATE\n" + "=" * 60)

    print("\n[1] chunk_key_gram_stats POSITIVE control: perfectly orthonormal 4-key episode")
    k1 = torch.eye(4).view(1, 4, 4)
    content1 = torch.ones(1, 4, dtype=torch.bool)
    recs1 = chunk_key_gram_stats(k1, content1, num_heads=1, chunk_size=4)
    assert len(recs1) == 1 and recs1[0]["n_valid"] == 4
    assert abs(recs1[0]["gram_deviation"]) < 1e-5, recs1
    assert abs(recs1[0]["effective_rank"] - 4.0) < 1e-3, recs1
    print(f"  gram_deviation={recs1[0]['gram_deviation']:.6f} (expected ~0) -- OK")

    print("\n[2] chunk_key_gram_stats NEGATIVE control: rank-collapsed (4 identical keys)")
    k2 = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).expand(1, 4, 4).contiguous()
    content2 = torch.ones(1, 4, dtype=torch.bool)
    recs2 = chunk_key_gram_stats(k2, content2, num_heads=1, chunk_size=4)
    assert abs(recs2[0]["gram_deviation"] - math.sqrt(12)) < 1e-4, recs2
    print(f"  gram_deviation={recs2[0]['gram_deviation']:.6f} (expected {math.sqrt(12):.6f}) -- OK, "
          f"metric has teeth")

    print("\n[2b] 'centered' variant control: a MASSIVE-ACTIVATION-style episode (one dominant "
          "shared/constant channel + small orthogonal per-item noise) must show HIGH raw deviation "
          "but LOW centered deviation -- this is the exact confound found empirically this session "
          "on all three real model families (see chunk_key_gram_stats docstring) and the whole "
          "reason the centered variant exists; if this check doesn't discriminate, centering is a "
          "no-op and the fix is fake")
    torch.manual_seed(0)
    n_items, d = 16, 8
    shared = torch.zeros(d); shared[0] = 50.0             # one dominant, CONSTANT-across-items channel
    noise = 0.02 * torch.randn(n_items, d)                 # small but genuinely distinct per-item signal, ALL items
    k2b = (shared.unsqueeze(0) + noise).unsqueeze(0)       # (1, n_items, d)
    content2b = torch.ones(1, n_items, dtype=torch.bool)
    recs2b = chunk_key_gram_stats(k2b, content2b, num_heads=1, chunk_size=n_items)
    r2b = recs2b[0]
    assert r2b["gram_deviation"] > 5.0, r2b            # raw: dominated by the shared channel, near-collapsed
    assert r2b["effective_rank"] < 2.0, r2b            # raw: looks rank-1 (the shared channel), structure invisible
    assert r2b["effective_rank_centered"] > d * 0.7, r2b   # centered: per-item structure recovered, near full rank
    assert r2b["gram_deviation_centered"] < r2b["gram_deviation"], r2b
    print(f"  raw: gram_deviation={r2b['gram_deviation']:.4f}, effective_rank={r2b['effective_rank']:.4f} "
          f"(pinned near 1 -- shared channel hides the real per-item structure) vs "
          f"centered: gram_deviation={r2b['gram_deviation_centered']:.4f}, "
          f"effective_rank={r2b['effective_rank_centered']:.4f} (recovers most of d={d} -- OK, "
          f"centering has teeth); top_channel_ratio={r2b['top_channel_ratio']:.1f}x (should be large)")
    assert r2b["top_channel_ratio"] > 20, r2b

    print("\n[3] EOS-exclusion correctness (renamed from EOT for Track D's own convention, "
          "same test as lm_attractor_probe_rd.py item 3)")
    k3 = torch.eye(4)
    k3o = torch.cat([k3, torch.tensor([[999.0, -999.0, 0.0, 0.0]])], dim=0).view(1, 5, 4)
    content3 = torch.tensor([[True, True, True, True, False]])
    recs3 = chunk_key_gram_stats(k3o, content3, num_heads=1, chunk_size=5)
    assert recs3[0]["n_valid"] == 4
    assert abs(recs3[0]["gram_deviation"] - recs1[0]["gram_deviation"]) < 1e-5
    print(f"  EOS position correctly dropped, gram_deviation matches no-EOS case -- OK")

    print("\n[4] MIN_VALID_FOR_GRAM guard")
    k4 = torch.eye(4)[:1].view(1, 1, 4)
    content4 = torch.ones(1, 1, dtype=torch.bool)
    recs4 = chunk_key_gram_stats(k4, content4, num_heads=1, chunk_size=1)
    assert recs4[0]["gram_deviation"] is None
    summ4 = summarize_gram_records(recs4)
    assert summ4["n_scored"] == 0 and summ4["n_excluded_below_min_valid"] == 1
    print(f"  n_valid=1 -> gram_deviation=None, correctly excluded -- OK")

    print("\n[5] NON-DIVISIBLE T/chunk_size (Track D's own generalization vs lm_attractor_probe_rd.py's "
          "assert-divisible): T=10, chunk_size=4 -> chunks [0:4],[4:8],[8:10] (trailing partial kept)")
    k5 = torch.randn(1, 10, 4)   # shape only matters here (chunking/masking logic), not the values
    content5 = torch.ones(1, 10, dtype=torch.bool)
    recs5 = chunk_key_gram_stats(k5, content5, num_heads=1, chunk_size=4)
    chunks_seen = sorted(set(r["chunk"] for r in recs5))
    assert chunks_seen == [0, 1, 2], chunks_seen
    n_valid_per_chunk = {r["chunk"]: r["n_valid"] for r in recs5}
    assert n_valid_per_chunk == {0: 4, 1: 4, 2: 2}, n_valid_per_chunk
    print(f"  chunks={chunks_seen}, n_valid_per_chunk={n_valid_per_chunk} (expected trailing chunk "
          f"n_valid=2, T=10 not divisible by 4) -- OK")

    print("\n[5b] MULTI-HEAD VECTORIZATION regression check (audit MAJOR-2 speed fix, this session): "
          "num_heads=3, each head given a DIFFERENT hand-built episode (identity/collapsed/random) -- "
          "the vectorized batched-SVD path must give the SAME per-head numbers as an independent, "
          "unvectorized reference computed head-by-head in this test, with NO cross-head leakage")
    torch.manual_seed(42)
    hd = 5
    h0 = torch.eye(hd)                                       # orthonormal
    h1 = torch.tensor([1.0, 0, 0, 0, 0]).unsqueeze(0).expand(hd, hd).contiguous()  # collapsed
    h2 = torch.randn(hd, hd)                                  # generic
    k5b = torch.stack([h0, h1, h2], dim=1).unsqueeze(0)       # (1, T=hd, num_heads=3, head_dim=hd)
    k5b = k5b.reshape(1, hd, 3 * hd)                          # (B=1, T=hd, D=15) -- heads contiguous per position
    content5b = torch.ones(1, hd, dtype=torch.bool)
    recs5b = chunk_key_gram_stats(k5b, content5b, num_heads=3, chunk_size=hd)
    by_head = {r["head"]: r for r in recs5b}
    # independent reference: compute head 0 and head 1's raw gram_deviation by hand, the same
    # way the ORIGINAL (pre-vectorization) per-head loop did it
    ref0 = gram_deviation(F.normalize(h0, dim=-1).unsqueeze(0)).item()
    ref1 = gram_deviation(F.normalize(h1, dim=-1).unsqueeze(0)).item()
    assert abs(by_head[0]["gram_deviation"] - ref0) < 1e-4, (by_head[0], ref0)
    assert abs(by_head[1]["gram_deviation"] - ref1) < 1e-4, (by_head[1], ref1)
    assert by_head[0]["gram_deviation"] < 1e-4, by_head[0]                 # head 0 orthonormal ~0
    assert abs(by_head[1]["gram_deviation"] - math.sqrt(hd * (hd - 1))) < 1e-3, by_head[1]  # head 1 fully collapsed
    assert by_head[0]["gram_deviation"] != by_head[2]["gram_deviation"], "heads leaked into each other"
    print(f"  head0 (orthonormal) gram_deviation={by_head[0]['gram_deviation']:.6f} (ref {ref0:.6f}), "
          f"head1 (collapsed) gram_deviation={by_head[1]['gram_deviation']:.6f} (ref {ref1:.6f}, expected "
          f"{math.sqrt(hd*(hd-1)):.6f}) -- OK, vectorized batched path matches the unvectorized reference "
          f"exactly, no cross-head leakage")

    print("\n[6] health_gate POSITIVE (clean model) and NEGATIVE (NaN param) controls")
    lin = torch.nn.Linear(4, 4)
    health_gate(lin, "clean-dummy")
    print("  clean model passes health_gate -- OK")
    with torch.no_grad():
        lin.weight[0, 0] = float("nan")
    try:
        health_gate(lin, "nan-dummy")
        raise AssertionError("health_gate should have raised on a NaN param")
    except RuntimeError as e:
        assert "HEALTH GATE FAILED" in str(e)
        print(f"  NaN-corrupted model correctly raises RuntimeError -- OK ({e})")

    print("\n[7] _windows_from_stream plumbing smoke (no network needed -- this is the real "
          "windowing/repeat/dup-detection logic build_windows calls, exercised directly)")
    # 7a: stream already long enough -- repeat must NOT fire, no duplicates expected (tiny
    # random test data, collision probability negligible)
    torch.manual_seed(0)
    long_stream = torch.randint(0, 1000, (10_000,))
    w7a, mask7a, info7a = _windows_from_stream(long_stream, n_windows=8, seq_len=64, seed=0, eos_id=999)
    assert w7a.shape == (8, 64) and mask7a.shape == (8, 64)
    assert info7a["repeat_fired"] is False and info7a["reps"] == 1, info7a
    assert info7a["n_duplicate_window_pairs"] == 0, info7a
    print(f"  long stream (n={long_stream.numel()}): repeat_fired={info7a['repeat_fired']}, "
          f"n_duplicate_window_pairs={info7a['n_duplicate_window_pairs']} -- OK")

    # 7b: AUDIT MAJOR-1 regression test -- stream deliberately too short, repeat fallback
    # MUST fire and MUST be reported (not silent). n_windows=25 > period=20 makes a
    # start-residue collision (mod period) a PIGEONHOLE CERTAINTY, not a chance event --
    # any two starts sharing the same residue mod 20 give byte-identical windows, since
    # the repeated stream is exactly periodic with period 20 (true regardless of how
    # seq_len relates to the period).
    short_stream = torch.arange(20)   # period-20 stream, need=25*64=1600 >> 20 -- heavy repeat
    w7b, mask7b, info7b = _windows_from_stream(short_stream, n_windows=25, seq_len=64, seed=0, eos_id=None)
    assert info7b["repeat_fired"] is True and info7b["reps"] > 1, info7b
    assert info7b["stream_numel_pre_repeat"] == 20, info7b
    assert info7b["n_duplicate_window_pairs"] > 0, (
        "audit MAJOR-1 regression: a heavily-repeated short stream should produce detectable "
        "duplicate windows, but n_duplicate_window_pairs=0", info7b)
    print(f"  short stream (n={short_stream.numel()}, heavy repeat forced): repeat_fired="
          f"{info7b['repeat_fired']}, reps={info7b['reps']}, n_duplicate_window_pairs="
          f"{info7b['n_duplicate_window_pairs']} -- OK, the fallback is detected and visible, "
          f"not silent (this is exactly the scenario the independent audit flagged)")

    print("\n" + "=" * 60 + "\n  ALL LM_ATTRACTOR_PROBE_TRACKD SMOKE CHECKS PASSED\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-real", type=str, default=None, choices=list(ADAPTERS.keys()),
                     help="load one real model, run ONLY the health gate + a tiny forward pass, then exit")
    ap.add_argument("--models", type=str, nargs="+", default=list(ADAPTERS.keys()), choices=list(ADAPTERS.keys()))
    ap.add_argument("--corpora", type=str, nargs="+", default=["openr1", "wikitext"])
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=95_000)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)

    if args.smoke:
        smoke()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.check_real:
        adapter = ADAPTERS[args.check_real]
        model, tok = adapter.load(device)
        report = health_gate(model, adapter.model_id)
        ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model(ids, use_cache=False)
        logits_finite_gate(out.logits, adapter.model_id)
        print(json.dumps(report, indent=2))
        print(f"\n{args.check_real}: health gate + forward-pass finiteness check PASSED")
        return

    assert device == "cuda", "Track D real measurement requires CUDA"
    result = run_all(args.models, device, args.corpora, args.chunk_sizes, args.n_windows,
                      args.seq_len, args.seed)

    summary = {k: v for k, v in result.items()}
    print("\n" + "=" * 70)
    print("TRACK D PROBE SUMMARY (models section elided, see --out for full detail):")
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=2))
    for name, m in result["models"].items():
        if m.get("FAILED"):
            print(f"  {name}: FAILED -- {m['error']}")
        else:
            print(f"  {name}: n_params={m['n_params']}, n_layers={m['n_layers']}, "
                  f"wall_s={m['wall_s']:.1f}")
    print("=" * 70)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
