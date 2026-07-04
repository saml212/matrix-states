"""wave_neg1_trackb.py -- TRACKB_REDESIGN.md Rev 3's Wave -1: MC anchor
recomputation (sec 5.3), the sec 4.4 selected-key logger's read-only
re-probe path for Cell 1 (sec 5.2/5.3's same-instrument re-measurement),
the duplicate-key NaN-stability stress-slice builder + positive-control
counting logic (sec 5.1, Rev 3 NEW-7), all BUILT and CPU-verifiable here.

HARD CONSTRAINT (this build session): no GPU runs. The genuine Wave -1
GPU-side work this file's functions are FOR (re-measuring the 6 archived
Wave C checkpoints; running the 2,000-step geo3_active=True stability
smoke) is NOT executed here -- only the MC anchor computation (pure CPU
math, run for real below) and the mechanism/plumbing logic (stress-slice
selection, hook registration, positive-control counting) are exercised,
against synthetic data, via test_trackb_smokes.py.
"""
from __future__ import annotations

import json
import os
import time

import torch

import hard_selectivity_rd as hs
from lm_pretrain_rd import _geo3_lm_select_and_orthogonalize, corpus_fixed_seed

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# sec 5.3 -- MC anchor recomputation at (K_sel=32, d_state=64). Pure CPU
# math, free, run NOW (this is the task brief's own explicit instruction:
# "planning values ~=3.94/~=31.50 ... to be superseded by the MC").
# ---------------------------------------------------------------------------

def run_mc_anchor_recomputation(out_path: str, K: int = 32, d: int = 64, n_samples: int = 500_000,
                                 seed: int = 0) -> dict:
    """Runs hard_selectivity_rd.mc_recompute_anchors and writes the result
    (with a wall-clock timestamp) to out_path. Deterministic given seed --
    re-running reproduces the same anchor_random/anchor_collapse values."""
    t0 = time.time()
    result = hs.mc_recompute_anchors(K, d, n_samples=n_samples, seed=seed)
    result["wall_s"] = time.time() - t0
    result["computed_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# sec 5.2/5.3 -- the sec 4.4 selected-key instrument, applied READ-ONLY to
# Cell 1 (the baseline, no selection mechanism). ONE instrument for every
# cell (Rev 2 F1): this is the SAME beta_topk selection
# _geo3_lm_select_and_orthogonalize already implements, called with
# forced_topk_idx=None and selection_mode="beta_topk" -- the k_out it
# returns is simply DISCARDED (Cell 1 never trains with it; only the diag
# Gram-deviation/mass statistics are used).
# ---------------------------------------------------------------------------

def selected_key_instrument_readonly(k_raw: torch.Tensor, beta: torch.Tensor, content_mask: torch.Tensor,
                                      chunk_size: int, k_sel: int, n_iter: int, resid_tol: float) -> dict:
    """Cell 1's own same-instrument reading (sec 5.2): applies the beta_topk
    selection read-only to a baseline (non-geo3) checkpoint's own
    (k_raw, beta), returning per-episode Gram deviation (diag['resid_*'])
    PLUS the per-chunk total write mass (hard_selectivity_rd.
    per_chunk_total_mass) -- the B_pinned ingredient, from the SAME forward
    pass (zero extra GPU cost, sec 4.3's own registered efficiency note)."""
    _, diag = _geo3_lm_select_and_orthogonalize(
        k_raw, beta, content_mask, chunk_size, k_sel, n_iter, resid_tol, selection_mode="beta_topk")
    mass = hs.per_chunk_total_mass(beta, chunk_size)
    return {
        "resid_mean": diag["resid_mean"], "resid_max": diag["resid_max"], "resid_min": diag["resid_min"],
        "frac_valid_selections": diag["frac_valid_selections"],
        "per_chunk_total_mass": mass,
        "support_size": hs.support_size(beta, chunk_size),
    }


def register_kbeta_capture_hooks(model) -> tuple[list, dict]:
    """The 'eval-time forward-hook probe' mechanism build-phase note (1)
    and sec 5.2's Cell-1 read-only re-probe both depend on: registers a
    forward hook on EVERY block's mixer.k_conv1d (captures k_raw, the
    PRE-kernel-normalization conv output _geo3_lm_select_and_orthogonalize's
    own docstring specifies) and mixer.b_proj (captures the raw pre-sigmoid
    beta score) -- non-invasive, works on ANY DeltaNetLM instance (geo3-
    active or not, hard-select-active or not) without modifying
    lm_pretrain_rd.py further. Returns (handles, captured) where
    captured[layer_idx] = {"k_raw": tensor, "beta": tensor (post-sigmoid,
    applied here since b_proj's own forward hook only sees the PRE-sigmoid
    linear output)}. Caller is responsible for calling handle.remove() on
    every handle when done (mirrors nn.Module hook hygiene)."""
    captured: dict = {}
    handles = []
    for layer_idx, blk in enumerate(model.blocks):
        captured[layer_idx] = {}

        def make_k_hook(idx):
            def hook(module, inputs, output):
                k_raw = output[0] if isinstance(output, tuple) else output
                captured[idx]["k_raw"] = k_raw
            return hook

        def make_beta_hook(idx):
            def hook(module, inputs, output):
                captured[idx]["beta"] = torch.sigmoid(output)
            return hook

        handles.append(blk.mixer.k_conv1d.register_forward_hook(make_k_hook(layer_idx)))
        handles.append(blk.mixer.b_proj.register_forward_hook(make_beta_hook(layer_idx)))
    return handles, captured


def cell1_readout_from_captured(captured: dict, token_ids: torch.Tensor, chunk_size: int, k_sel: int,
                                 n_iter: int, resid_tol: float, excluded_token_ids: tuple) -> dict:
    """Given a `captured` dict from register_kbeta_capture_hooks (populated
    by ONE real forward pass -- on GPU for a real checkpoint, requires
    CUDA, NOT run in this build session), builds content_mask from
    token_ids and applies selected_key_instrument_readonly per layer.
    Reshapes each layer's captured (B,T,d_model)-shaped k_raw into
    (B,T,H,head_dim) is the CALLER's job if num_heads>1 -- every registered
    Track B cell is num_heads=1 (sec 2 principle 3), so this function
    assumes k_raw's last dim IS head_dim (H=1) and reshapes accordingly."""
    B, T = token_ids.shape
    content_mask = torch.ones(B, T, dtype=torch.bool, device=token_ids.device)
    for tid in excluded_token_ids:
        content_mask = content_mask & (token_ids != tid)
    per_layer = {}
    for layer_idx, d in captured.items():
        k_raw = d["k_raw"].unsqueeze(2)     # (B,T,1,head_dim) -- H=1 assumption, stated above
        beta = d["beta"]                     # (B,T,1) if num_heads==1 (b_proj outputs num_heads channels)
        per_layer[layer_idx] = selected_key_instrument_readonly(
            k_raw, beta, content_mask, chunk_size, k_sel, n_iter, resid_tol)
    return per_layer


# ---------------------------------------------------------------------------
# sec 5.1 (Rev 3 NEW-7) -- the duplicate-key NaN-stability stress slice,
# built from the CODE's own actual failure mechanism (lm_pretrain_rd.py
# :349-353's documented residual risk): identical token 4-grams within one
# chunk produce EXACTLY identical layer-0 conv-context keys (conv_size=4),
# and >=~6 such duplicated rows among the beta-SELECTED top-K positions can
# NaN the eigh fallback's backward.
# ---------------------------------------------------------------------------

def chunk_n_dup_max(token_ids: torch.Tensor, chunk_size: int, gram_n: int = 4,
                     vocab_bits: int = 17) -> torch.Tensor:
    """Per chunk, the largest count of positions sharing one identical
    token gram_n-gram (the gram ENDING at that position, matching
    k_conv1d's own causal ShortConvolution context -- key at position t is
    a function of tokens t-conv_size+1..t). token_ids: (B,T) int64.

    Gram packing (AUDIT FIX, independent audit 2026-07-04 -- M2): the first
    build packed all gram_n tokens into ONE int64 word (positional
    base-2**vocab_bits encoding), which at the default gram_n=4 x
    vocab_bits=17 = 68 bits OVERFLOWS int64's 63 value bits -- torch int64
    arithmetic wraps silently, so two DIFFERENT grams whose packed values
    differ by a multiple of 2^64 read as duplicates (concretely: first-gram
    tokens 0 vs 8192 with identical trailing tokens collide, 8192*2^51 =
    2^64 == 0 mod 2^64 -- the regression test in test_trackb_smokes.py
    constructs exactly this pair). Fixed by packing across TWO int64 words
    (tokens split half-and-half: <=2x17=34 bits per word at gram_n=4,
    always in range), compared jointly via torch.unique(dim=0) over
    (word0, word1) rows -- exact and collision-free for any gram_n with
    ceil(gram_n/2)*vocab_bits <= 62, asserted below rather than assumed.

    Positions before a full gram_n-length context (t < gram_n-1 in the
    FULL sequence) are excluded from the comparison (conservative: never
    manufactures a duplicate out of two different zero-padded/short
    prefixes). Returns (B, n_chunks) int64."""
    B, T = token_ids.shape
    assert T % chunk_size == 0, f"T={T} not a multiple of chunk_size={chunk_size}"
    assert (1 << vocab_bits) > int(token_ids.max().item()), (
        f"vocab_bits={vocab_bits} (base {1 << vocab_bits}) too small for max token id "
        f"{int(token_ids.max().item())} -- would silently corrupt the exact-gram packing")
    half = (gram_n + 1) // 2
    assert half * vocab_bits <= 62, (
        f"gram_n={gram_n} x vocab_bits={vocab_bits}: {half} tokens/word x {vocab_bits} bits = "
        f"{half * vocab_bits} bits > 62 -- the two-word packing would itself overflow; lower "
        f"vocab_bits or widen the word count deliberately (M2's own failure mode, do not wrap).")
    n_chunks = T // chunk_size
    grams = token_ids.unfold(1, gram_n, 1).to(torch.int64)          # (B, T-gram_n+1, gram_n)
    base = 1 << vocab_bits
    word0 = torch.zeros(B, grams.shape[1], dtype=torch.int64)
    word1 = torch.zeros(B, grams.shape[1], dtype=torch.int64)
    for g in range(half):
        word0 = word0 * base + grams[:, :, g]
    for g in range(half, gram_n):
        word1 = word1 * base + grams[:, :, g]
    packed = torch.stack([word0, word1], dim=-1)                     # (B, n_positions, 2)
    end_pos = torch.arange(gram_n - 1, gram_n - 1 + packed.shape[1])
    chunk_of = end_pos // chunk_size                                  # (n_positions,)
    n_dup_max = torch.zeros(B, n_chunks, dtype=torch.int64)
    for b in range(B):
        for c in range(n_chunks):
            sel = packed[b, chunk_of == c]                            # (n_in_chunk, 2)
            if sel.numel() == 0:
                continue
            _, counts = torch.unique(sel, dim=0, return_counts=True)
            n_dup_max[b, c] = int(counts.max().item())
    return n_dup_max


def select_duplicate_key_stress_windows(token_ids: torch.Tensor, doc_offsets: torch.Tensor, seq_len: int,
                                         chunk_size: int, n_dup_min: int = 8, gram_n: int = 4,
                                         corpus_name: str = "openr1") -> dict:
    """Rev 3 NEW-7 slice selection: rank document-aligned windows by
    n_dup_max (the largest within-chunk identical-4-gram count anywhere in
    the window), select those with n_dup_max >= n_dup_min ('comfortably
    past the audit-measured >=~6 onset'), deterministic under the
    corpus-fixed-seed convention (corpus_fixed_seed, imported from
    lm_pretrain_rd -- SAME function every other eval/rank sampler in this
    codebase uses, not re-derived). token_ids/doc_offsets: full corpus
    tensors (load_corpus's own output shape). Returns a dict with the
    selected window start indices, their n_dup_max values, and a digest."""
    n = token_ids.numel()
    eligible = doc_offsets[doc_offsets + seq_len + 1 <= n]
    assert eligible.numel() > 0, "no eligible document-aligned windows at this seq_len"
    offs = torch.arange(seq_len)
    idx = eligible.unsqueeze(1) + offs.unsqueeze(0)          # (n_eligible, seq_len)
    windows = token_ids[idx]
    per_chunk = chunk_n_dup_max(windows, chunk_size, gram_n=gram_n)     # (n_eligible, n_chunks)
    window_n_dup_max = per_chunk.max(dim=1).values                      # (n_eligible,)
    keep = window_n_dup_max >= n_dup_min
    selected_starts = eligible[keep]
    selected_vals = window_n_dup_max[keep]
    # deterministic ordering: sort by (start index) so the slice is reproducible independent of
    # torch's own nonzero/boolean-mask iteration order across versions
    order = torch.argsort(selected_starts)
    selected_starts = selected_starts[order]
    selected_vals = selected_vals[order]
    return {
        "corpus": corpus_name, "seq_len": seq_len, "chunk_size": chunk_size, "gram_n": gram_n,
        "n_dup_min": n_dup_min, "n_eligible_windows": int(eligible.numel()),
        "n_selected_windows": int(selected_starts.numel()),
        "selected_starts": selected_starts.tolist(), "selected_n_dup_max": selected_vals.tolist(),
        "corpus_fixed_seed_used": corpus_fixed_seed(corpus_name),
    }


def count_max_duplicate_group_in_selected(k_selected: torch.Tensor) -> int:
    """The Rev 3 NEW-7 positive-control quantity: given (n_ep, k_sel, d) the
    GATHERED (selected) raw key rows for ONE forward call, returns the
    largest count of EXACTLY-duplicated rows within any single episode's
    own k_sel selected rows -- what the '>=6 duplicated SELECTED rows'
    floor (per forward call) is measured against. Exact byte-equality
    (duplicate KEYS arise from bit-identical upstream conv-context
    computation on identical token 4-grams, not merely close floats)."""
    n_ep, k_sel, d = k_selected.shape
    max_dup = 0
    for e in range(n_ep):
        rows = k_selected[e].detach().cpu().numpy()
        seen: dict = {}
        for i in range(k_sel):
            key = rows[i].tobytes()
            seen[key] = seen.get(key, 0) + 1
        if seen:
            max_dup = max(max_dup, max(seen.values()))
    return max_dup


class NanStabilityProbeCounter:
    """The Rev 3 NEW-7 positive-control floor, accumulated across a
    training run's forward calls: '>=25 forward calls during the smoke
    with >=6 duplicated selected rows' -- else the smoke is NON-PROBATIVE
    (not 'passed'), per the house forced-failure-test-precondition lesson
    (EXPERIMENT_LOG.md's own [LEARN] block, cited in TRACKB_REDESIGN.md's
    reading list). This class is the counting/logging harness; a real
    GPU-side stability smoke calls `.observe(k_selected)` once per forward
    call. NOT invoked against real training in this build session (no GPU
    runs) -- exercised here only against synthetic per-call data."""

    def __init__(self, floor_n_calls: int = 25, floor_n_dup: int = 6):
        self.floor_n_calls = floor_n_calls
        self.floor_n_dup = floor_n_dup
        self.n_calls_meeting_floor = 0
        self.n_calls_total = 0
        self.max_dup_per_call: list[int] = []

    def observe(self, k_selected: torch.Tensor) -> int:
        max_dup = count_max_duplicate_group_in_selected(k_selected)
        self.n_calls_total += 1
        self.max_dup_per_call.append(max_dup)
        if max_dup >= self.floor_n_dup:
            self.n_calls_meeting_floor += 1
        return max_dup

    def is_probative(self) -> bool:
        return self.n_calls_meeting_floor >= self.floor_n_calls

    def summary(self) -> dict:
        return {
            "n_calls_total": self.n_calls_total,
            "n_calls_meeting_floor": self.n_calls_meeting_floor,
            "floor_n_calls": self.floor_n_calls, "floor_n_dup": self.floor_n_dup,
            "probative": self.is_probative(),
            "max_dup_per_call": self.max_dup_per_call,
            "verdict": ("PROBATIVE" if self.is_probative() else
                        "NON-PROBATIVE (floor not met -- re-run with selection FORCED onto the "
                        "duplicated positions via the M6 forced_topk_idx argument, Rev 3's "
                        "registered fallback)"),
        }


# ---------------------------------------------------------------------------
# AUDIT FIX (independent audit 2026-07-04, F1's dispatch half): the stress-
# slice MATERIALIZER -- turns select_duplicate_key_stress_windows' selected
# windows into a real, load_corpus-compatible corpus directory, so the
# stability smoke's build_cmd names an actual trainable corpus
# ("openr1-stress" in lm_pretrain_rd.CORPUS_DIRS) instead of a manifest
# annotation with nothing behind it. CPU-only, but requires the real source
# corpus under --data-dir -- built this session, NOT run (box /data access
# is a launch-time step).
# ---------------------------------------------------------------------------

def materialize_stress_slice_corpus(data_dir: str, out_subdir: str = "reasoning_stress_eot",
                                     src_corpus: str = "openr1", seq_len: int = 512,
                                     chunk_size: int = 64, n_dup_min: int = 8, gram_n: int = 4,
                                     min_train_windows: int = 64, min_val_windows: int = 8) -> dict:
    """Writes <data_dir>/<out_subdir>/ in load_corpus's exact format
    (meta.json + {split}.pt + {split}_doc_offsets.pt): each selected stress
    window becomes one 'document' (window tokens + one appended EOT --
    load_corpus's own doc-boundary invariant requires the token before
    every non-first doc start to be EOT; appending one EOT per window
    preserves the window's own duplication content unchanged).

    REGISTERED APPROXIMATION (stated, not silent): training windows are
    sampled at RANDOM offsets over this slice corpus (get_batch), so chunk
    boundaries during training do NOT always align with the doc-aligned
    tiling the n_dup_max ranking used -- a duplicated span can straddle two
    chunks and contribute fewer per-chunk duplicates than its nominal
    n_dup_max. This is exactly why the smoke's PROBATIVENESS is measured by
    the NanStabilityProbeCounter's realized floor (>=25 calls with >=6
    duplicated SELECTED rows), never assumed from the slice's nominal
    statistics -- if random tiling dilutes the regime below the floor, the
    smoke declares NON-PROBATIVE and the M6 forced-selection fallback
    fires (TRACKB_REDESIGN.md sec 5.1, Rev 3 NEW-7)."""
    from lm_pretrain_rd import load_corpus, EOT_TOKEN_ID
    train, val, meta, train_offs, val_offs = load_corpus(data_dir, src_corpus, "cpu")

    def _build_split(tokens, offs, min_windows, split_name):
        slice_info = select_duplicate_key_stress_windows(
            tokens, offs, seq_len=seq_len, chunk_size=chunk_size, n_dup_min=n_dup_min,
            gram_n=gram_n, corpus_name=src_corpus)
        n_sel = slice_info["n_selected_windows"]
        assert n_sel >= min_windows, (
            f"{split_name}: only {n_sel} stress windows at n_dup_min={n_dup_min} (need >= "
            f"{min_windows}) -- lower n_dup_min DELIBERATELY (it must stay >= the audit-measured "
            f"~6 onset) or accept a smaller smoke, never silently pad with non-stress windows")
        pieces, doc_offsets, cursor = [], [], 0
        eot = torch.tensor([EOT_TOKEN_ID], dtype=tokens.dtype)
        for start in slice_info["selected_starts"]:
            doc_offsets.append(cursor)
            w = tokens[start:start + seq_len]
            pieces.append(w)
            pieces.append(eot)
            cursor += seq_len + 1
        return torch.cat(pieces), torch.tensor(doc_offsets, dtype=torch.int64), slice_info

    train_out, train_offs_out, train_info = _build_split(train, train_offs, min_train_windows, "train")
    val_out, val_offs_out, val_info = _build_split(val, val_offs, min_val_windows, "val")

    out_dir = os.path.join(data_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(train_out, os.path.join(out_dir, "train.pt"))
    torch.save(val_out, os.path.join(out_dir, "val.pt"))
    torch.save(train_offs_out, os.path.join(out_dir, "train_doc_offsets.pt"))
    torch.save(val_offs_out, os.path.join(out_dir, "val_doc_offsets.pt"))
    out_meta = {
        "vocab_size": meta["vocab_size"], "tokenizer": meta["tokenizer"], "eot_separated": True,
        "eot_token_id": EOT_TOKEN_ID,
        "source": f"duplicate-key stress slice of {src_corpus} (TRACKB_REDESIGN.md sec 5.1, Rev 3 "
                  f"NEW-7; wave_neg1_trackb.materialize_stress_slice_corpus)",
        "slice_params": {"seq_len": seq_len, "chunk_size": chunk_size, "n_dup_min": n_dup_min,
                          "gram_n": gram_n},
        "train_tokens": int(train_out.numel()), "train_docs": int(train_offs_out.numel()),
        "val_tokens": int(val_out.numel()), "val_docs": int(val_offs_out.numel()),
        "train_slice_selection": {k: v for k, v in train_info.items() if k != "selected_starts"},
        "val_slice_selection": {k: v for k, v in val_info.items() if k != "selected_starts"},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(out_meta, f, indent=2)
    return out_meta


# ---------------------------------------------------------------------------
# AUDIT FIX (independent audit 2026-07-04, F1's Wave-3 half): the re-probe
# CLI -- the real command run_trackb_wave.py's cell1probe/Wave-3 manifests
# dispatch. Loads a checkpoint, registers the capture hooks, runs ONE
# doc-aligned forward pass over the val split (corpus-fixed-seeded, FIX-1
# convention), applies the sec 4.4 selected-key instrument read-only, and
# writes a validity-checkable result JSON. GPU REQUIRED (chunk_delta_rule
# has no CPU path) -- built this session, NOT run.
# ---------------------------------------------------------------------------

def run_checkpoint_probe(checkpoint_path: str, data_dir: str, corpus: str, n_docs: int,
                          seq_len: int, chunk_size: int, k_sel: int, n_iter: int,
                          resid_tol: float, out_path: str) -> dict:
    from lm_pretrain_rd import DeltaNetLM, load_corpus, EOT_TOKEN_ID
    assert torch.cuda.is_available(), "run_checkpoint_probe requires CUDA (chunk_delta_rule has no CPU path)"
    device = "cuda"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DeltaNetLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ckpt_step = int(ckpt["step"])

    _, val, meta, _, val_offs = load_corpus(data_dir, corpus, device)
    gen = torch.Generator(device=device).manual_seed(corpus_fixed_seed(corpus) + 60_000)
    n = val.numel()
    eligible = val_offs[val_offs + seq_len + 1 <= n]
    assert eligible.numel() >= n_docs, f"only {eligible.numel()} eligible val docs for n_docs={n_docs}"
    pick = torch.randint(0, eligible.numel(), (n_docs,), generator=gen, device=val_offs.device)
    starts = eligible[pick]
    offs = torch.arange(seq_len, device=val.device)
    batch = val[starts.unsqueeze(1) + offs.unsqueeze(0)]

    handles, captured = register_kbeta_capture_hooks(model)
    try:
        with torch.no_grad():
            # build-phase note (1), load-bearing: pass the CHECKPOINT'S OWN recorded step, so a
            # hard_select_active model's step-dependent mechanism CONTINUES its training-time RNG
            # stream (DeltaNetLMMixer.forward's own docstring contract).
            _ = model(batch, step=ckpt_step)
        per_layer = cell1_readout_from_captured(
            captured, batch, chunk_size, k_sel, n_iter, resid_tol,
            excluded_token_ids=(EOT_TOKEN_ID,))
    finally:
        for h in handles:
            h.remove()

    result = {"complete": True, "checkpoint": checkpoint_path, "checkpoint_step": ckpt_step,
              "corpus": corpus, "n_docs": n_docs, "seq_len": seq_len, "chunk_size": chunk_size,
              "k_sel": k_sel, "n_iter": n_iter, "resid_tol": resid_tol, "per_layer": {}}
    for li, r in per_layer.items():
        mass = r["per_chunk_total_mass"].reshape(-1).double()
        supp = r["support_size"].reshape(-1).double()
        result["per_layer"][str(li)] = {
            "resid_mean": r["resid_mean"], "resid_max": r["resid_max"], "resid_min": r["resid_min"],
            "frac_valid_selections": r["frac_valid_selections"],
            "per_chunk_total_mass": {"mean": mass.mean().item(),
                                      "std": mass.std(unbiased=True).item() if mass.numel() > 1 else 0.0,
                                      "p10": torch.quantile(mass, 0.10).item(),
                                      "p90": torch.quantile(mass, 0.90).item(),
                                      "n": int(mass.numel())},
            "support": {"median": supp.median().item(),
                         "p10": torch.quantile(supp, 0.10).item()},
        }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_mutually_exclusive_group(required=True)
    sub.add_argument("--probe-checkpoint", type=str, default=None,
                      help="read-only sec 4.4 selected-key re-probe of ONE checkpoint (Cell 1's "
                           "archived Wave C checkpoints / Wave 3's instrumentation pass). GPU "
                           "required.")
    sub.add_argument("--build-stress-slice", action="store_true",
                      help="materialize the duplicate-key stress-slice corpus (Rev 3 NEW-7) under "
                           "--data-dir/reasoning_stress_eot. CPU, needs the real source corpus.")
    sub.add_argument("--mc-anchors", action="store_true",
                      help="run the sec 5.3 MC anchor recomputation (CPU, free) and write --out.")
    ap.add_argument("--data-dir", default="/data/deltanet_rd_data")
    ap.add_argument("--corpus", default="openr1")
    ap.add_argument("--n-docs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--k-sel", type=int, default=32)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--resid-tol", type=float, default=1e-2)
    ap.add_argument("--n-dup-min", type=int, default=8)
    ap.add_argument("--mc-samples", type=int, default=500_000)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.probe_checkpoint:
        assert args.out, "--out is required for --probe-checkpoint"
        result = run_checkpoint_probe(args.probe_checkpoint, args.data_dir, args.corpus,
                                       args.n_docs, args.seq_len, args.chunk_size, args.k_sel,
                                       args.n_iter, args.resid_tol, args.out)
        print(json.dumps({k: v for k, v in result.items() if k != "per_layer"}, indent=2))
        print(f"wrote {args.out}")
    elif args.build_stress_slice:
        out_meta = materialize_stress_slice_corpus(args.data_dir, src_corpus=args.corpus,
                                                    seq_len=args.seq_len, chunk_size=args.chunk_size,
                                                    n_dup_min=args.n_dup_min)
        print(json.dumps(out_meta, indent=2))
    else:
        assert args.out, "--out is required for --mc-anchors"
        result = run_mc_anchor_recomputation(args.out, K=args.k_sel, d=64, n_samples=args.mc_samples)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
