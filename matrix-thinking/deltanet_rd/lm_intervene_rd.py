"""lm_intervene_rd.py -- Wave 2 (RD-2) inference-time rank-truncation
intervention. See DELTANET_REALDATA_DESIGN.md section 6.2 (the
interventional addition: reasoning-damage vs. fluency-damage curves) and
MAJOR-5's pre-registered fallback (section 6.1): the third, symbol-density-
matched STEM-prose corpus arm is DEFERRED (sourcing cost not paid this
build phase) -- this script implements the pre-registered fallback instead
of sourcing a new corpus: **frequency-normalized damage metrics** (per-token
truncation damage stratified by token-frequency band and symbol-vs-word
class), reported honestly as the fallback it is, never claimed equivalent
to the matched-corpus arm.

**Claim tier (section 14.7, section 6.3 -- restated, not allowed to drift):
this script's output is DESCRIPTIVE + INTERVENTIONAL. It is inference-time
causal (truncating the state and observing a loss delta on the SAME
checkpoint IS a real intervention), but it is NOT the premise-conditional
causal tier (Wave 1 / RD-1's C16 bound-necessity proof). No premise
machinery (Gram deviation, salvage ratio, alignment cosine) applies here --
this script does not import or compute any of it.**

Method (section 6.2, this build's brief): for a sampled window of
`ctx_len + cont_len + 1` tokens from a corpus's val split,
  1. run the model over the first `ctx_len` tokens (zero initial state,
     matching how the model was actually trained) to get each layer's
     final recurrent state S -- the REAL, UNTRUNCATED state the trained
     model would use at inference time;
  2. for each k in a truncation grid (plus a `k=None` untouched baseline):
     truncate EVERY layer's S to rank k (rank_utils' best rank-k Frobenius
     approximation, layout-agnostic -- see lm_pretrain_rd.DeltaNetLMMixer's
     forward() docstring for why no [K,V]/[V,K] reconciliation is needed
     here, unlike model_rd.py's readout path);
  3. continue the model on the next `cont_len` tokens using the
     (possibly-truncated) state as `initial_states`, scoring PER-TOKEN
     cross-entropy loss on those `cont_len` tokens against the untouched
     (k=None) baseline's per-token loss on the SAME tokens -- the delta
     IS the truncation damage at that k, one number per scored token.

Both continuation legs (context and scored continuation) individually
satisfy `>= _MIN_KERNEL_T` (a hard chunk_delta_rule requirement, not a
choice) -- validated at CLI-parse time.

AUDIT FIXES (independent audit round 1, 2026-07-03): FIX-1 -- eval windows
are seeded from corpus_fixed_seed(corpus), never a training seed (--seed
is a documented no-op for sampling); per-corpus window digests +
boundary-contamination stats (FIX-3, via the rebuilt EOT-separated
corpora's doc_offsets) are logged in the result JSON.

Run the smoke gate FIRST: python lm_intervene_rd.py --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports

from grammar_rd import load_gpt2_tokenizer
from lm_pretrain_rd import (CLAIM_TIER, CORPUS_DIRS, DEFAULT_DATA_DIR, OTHER_CORPUS,
                             DeltaNetLM, _MIN_KERNEL_T, boundary_stats, corpus_fixed_seed,
                             load_corpus, set_and_log_tf32, window_digest)
from model_rd import TRUNC_IMPLS, TruncationError, _LinAlgError, assert_rank_le

DEFAULT_K_GRID = (8, 16, 24, 32, 48, 64)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: str) -> tuple[DeltaNetLM, dict]:
    ckpt = torch.load(path, map_location=device)
    model = DeltaNetLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Frequency-band / symbol-vs-word stratification (MAJOR-5's pre-registered
# fallback, in place of the deferred symbol-density-matched third corpus)
# ---------------------------------------------------------------------------

def build_token_frequency_bins(vocab_size: int, *token_tensors: torch.Tensor, n_bins: int = 4) -> torch.Tensor:
    """bin_of_id[token_id] -> band index in [0, n_bins), 0 = MOST frequent
    quartile of TYPES (by rank, over the union of `token_tensors`' counts),
    n_bins-1 = least frequent (including unseen-in-reference-corpus ids).
    Rank-based (equal TYPE count per band), the standard NLP "frequency
    band" convention -- not equal-MASS bins (which would collapse almost
    the entire vocabulary into one high-frequency band under a
    Zipfian distribution)."""
    counts = torch.zeros(vocab_size, dtype=torch.int64)
    for t in token_tensors:
        counts += torch.bincount(t.detach().cpu(), minlength=vocab_size)[:vocab_size]
    order = torch.argsort(counts, descending=True)
    bin_of_id = torch.empty(vocab_size, dtype=torch.int64)
    bin_size = vocab_size // n_bins
    for b in range(n_bins):
        lo = b * bin_size
        hi = vocab_size if b == n_bins - 1 else (b + 1) * bin_size
        bin_of_id[order[lo:hi]] = b
    return bin_of_id, counts


_SYM_CLASSES = ("word", "symbol", "other")


def build_symbol_word_class(tokenizer, vocab_size: int) -> torch.Tensor:
    """cls_of_id[token_id] -> index into _SYM_CLASSES. Simple, HONESTLY
    coarse heuristic (this IS the fallback, not a linguistic classifier):
    a GPT-2 byte-level BPE piece is "word" if it contains at least one
    alphabetic character after stripping the leading-space marker, "symbol"
    if it's non-empty and contains none, "other" for the empty/whitespace-
    only edge case. Uses convert_ids_to_tokens (a vocab lookup, NOT
    decode-per-id) for speed across the full 50,257-token vocab."""
    toks = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    cls_of_id = torch.empty(vocab_size, dtype=torch.int64)
    for i, tok in enumerate(toks):
        stripped = tok.replace("Ġ", "").replace("Ċ", "")   # GPT-2's leading-space / newline markers
        if stripped == "":
            cls_of_id[i] = _SYM_CLASSES.index("other")
        elif any(c.isalpha() for c in stripped):
            cls_of_id[i] = _SYM_CLASSES.index("word")
        else:
            cls_of_id[i] = _SYM_CLASSES.index("symbol")
    return cls_of_id


def stratify_damage(token_ids_flat: torch.Tensor, damage_flat: torch.Tensor,
                     bin_of_id: torch.Tensor, cls_of_id: torch.Tensor, n_bins: int) -> dict:
    """Per-token truncation damage stratified by frequency band and
    symbol-vs-word class (MAJOR-5's fallback, literally). Also reports a
    "balanced mean" (macro-average across NON-EMPTY frequency bands, equal
    weight per band) alongside the raw token-count-weighted mean -- the
    actual "normalized" half of "frequency-normalized": the raw mean is
    dominated by high-frequency tokens under a Zipfian distribution, the
    balanced mean corrects for that skew.

    token_ids_flat, damage_flat: 1D tensors of equal length (the SCORED
    continuation target tokens and their per-token loss deltas)."""
    device = damage_flat.device
    bin_of_id = bin_of_id.to(device)
    cls_of_id = cls_of_id.to(device)
    bands = bin_of_id[token_ids_flat]
    classes = cls_of_id[token_ids_flat]

    band_means, band_counts = {}, {}
    for b in range(n_bins):
        mask = bands == b
        band_counts[b] = int(mask.sum().item())
        band_means[b] = damage_flat[mask].mean().item() if mask.any() else None
    finite_band_means = [v for v in band_means.values() if v is not None]
    balanced_mean = (sum(finite_band_means) / len(finite_band_means)) if finite_band_means else None

    class_means, class_counts = {}, {}
    for ci, name in enumerate(_SYM_CLASSES):
        mask = classes == ci
        class_counts[name] = int(mask.sum().item())
        class_means[name] = damage_flat[mask].mean().item() if mask.any() else None

    return {
        "raw_mean": damage_flat.mean().item(),
        "n_tokens_scored": int(damage_flat.numel()),
        "by_frequency_band": {str(b): band_means[b] for b in range(n_bins)},
        "by_frequency_band_counts": {str(b): band_counts[b] for b in range(n_bins)},
        "frequency_balanced_mean": balanced_mean,
        "by_symbol_word_class": class_means,
        "by_symbol_word_class_counts": class_counts,
    }


# ---------------------------------------------------------------------------
# Windows + the core two-phase truncation intervention
# ---------------------------------------------------------------------------

def sample_windows(tokens: torch.Tensor, n_windows: int, ctx_len: int, cont_len: int,
                    generator: torch.Generator, return_starts: bool = False):
    """(n_windows, ctx_len + cont_len + 1) contiguous windows; with
    return_starts=True also returns the start indices (for window_digest /
    boundary_stats -- AUDIT FIX-1/FIX-3)."""
    W = ctx_len + cont_len + 1
    n = tokens.numel()
    assert n > W, f"corpus too small ({n} tokens) for ctx_len+cont_len+1={W}"
    starts = torch.randint(0, n - W, (n_windows,), generator=generator, device=tokens.device)
    offs = torch.arange(W, device=tokens.device)
    idx = starts.unsqueeze(1) + offs.unsqueeze(0)
    windows = tokens[idx]
    return (windows, starts) if return_starts else windows


@torch.no_grad()
def truncation_damage(model: DeltaNetLM, windows: torch.Tensor, ctx_len: int, cont_len: int,
                       k: int | None, trunc_impl: str = "eigh") -> torch.Tensor:
    """windows: (B, ctx_len+cont_len+1). k=None -> baseline (state passed
    through UNTOUCHED). k=int -> every layer's context-final state is
    truncated to rank k before the continuation pass. Returns per-token
    continuation loss, (B, cont_len) -- NOT yet reduced to a mean (the
    stratification functions above need the raw per-token values)."""
    model.eval()
    ctx = windows[:, :ctx_len]
    cont_in = windows[:, ctx_len:ctx_len + cont_len]
    cont_tgt = windows[:, ctx_len + 1:ctx_len + cont_len + 1]

    _, ctx_states = model(ctx, initial_states=None, return_states=True)

    if k is not None:
        trunc_fn = TRUNC_IMPLS[trunc_impl]
        truncated = []
        for S in ctx_states:
            try:
                Sk = trunc_fn(S, k)
            except _LinAlgError:
                eye = torch.eye(S.shape[-1], device=S.device, dtype=S.dtype)
                try:
                    Sk = trunc_fn(S + 1e-6 * eye, k)
                except _LinAlgError as e2:
                    raise TruncationError(
                        f"{trunc_impl} truncation at k={k} failed to converge even after the "
                        f"one-shot +1e-6 diagonal-jitter retry") from e2
            assert_rank_le(Sk, k)
            truncated.append(Sk)
        init_states = truncated
    else:
        init_states = ctx_states

    logits, _ = model(cont_in, initial_states=init_states, return_states=True)
    per_token_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), cont_tgt.reshape(-1),
                                      reduction="none").reshape(cont_tgt.shape)
    return per_token_loss, cont_tgt


def run_intervention(model: DeltaNetLM, corpus_data: dict, k_grid, ctx_len: int, cont_len: int,
                      n_windows: int, batch_size: int, bin_of_id: torch.Tensor, cls_of_id: torch.Tensor,
                      n_bins: int, device: str, trunc_impl: str = "eigh") -> dict:
    """corpus_data: {corpus_name: (token_tensor, doc_offsets)} (typically
    the reasoning val split AND the wikitext val split -- section 6.2's two
    arms). For each corpus, samples n_windows windows (processed in batches
    of batch_size -- house VRAM rule: eval batches capped independently),
    computes the k=None baseline once, then for every k in k_grid computes
    per-token damage = loss(k) - loss(None) on the SAME windows/tokens, and
    stratifies it.

    AUDIT FIX-1: the window sampler is seeded from
    corpus_fixed_seed(corpus_name) -- a pure function of the corpus,
    independent of any training seed or CLI --seed -- so every checkpoint
    (from ANY training seed) is measured on the IDENTICAL window set; the
    per-corpus window digest is logged for cross-run byte-identity
    verification. AUDIT FIX-3: per-corpus boundary-contamination stats of
    the sampled windows (context+continuation span) are logged alongside.

    Returns {corpus: {"windows": {digest+boundary stats},
                       "by_k": {str(k): stratify_damage(...)}}}."""
    report = {}
    for corpus_name, (tokens, doc_offsets) in corpus_data.items():
        gen = torch.Generator(device=device).manual_seed(corpus_fixed_seed(corpus_name))
        windows, starts = sample_windows(tokens, n_windows, ctx_len, cont_len, gen, return_starts=True)
        win_info = {"window_digest": window_digest(starts),
                     "window_seed_policy": "corpus_fixed (AUDIT FIX-1: crc32(corpus), independent of --seed)",
                     "boundary_full_window": boundary_stats(starts, ctx_len + cont_len + 1, doc_offsets),
                     # the SCORED span only (continuation): contamination the damage metric actually sees
                     "boundary_scored_continuation": boundary_stats(starts + ctx_len, cont_len + 1, doc_offsets)}
        n_skipped = {k: 0 for k in (None,) + tuple(k_grid)}

        base_losses, base_tgts = [], []
        for i in range(0, n_windows, batch_size):
            wb = windows[i:i + batch_size]
            loss_b, tgt_b = truncation_damage(model, wb, ctx_len, cont_len, k=None, trunc_impl=trunc_impl)
            base_losses.append(loss_b)
            base_tgts.append(tgt_b)
        base_loss = torch.cat(base_losses, dim=0)          # (n_windows, cont_len)
        base_tgt = torch.cat(base_tgts, dim=0)

        by_k = {}
        for k in k_grid:
            # Track exact window INDICES that survived (not just a count) --
            # a TruncationError can skip a MIDDLE batch while a later batch
            # succeeds, so the surviving k_loss rows do not correspond to a
            # simple base_loss[:n_scored] prefix slice. Gathering by the
            # real kept indices is the correctness-relevant fix (a prefix
            # slice would silently pair a window's truncated loss with a
            # DIFFERENT window's baseline loss whenever a non-final batch
            # is skipped).
            k_losses, kept_idx = [], []
            for i in range(0, n_windows, batch_size):
                wb = windows[i:i + batch_size]
                try:
                    loss_b, _ = truncation_damage(model, wb, ctx_len, cont_len, k=k, trunc_impl=trunc_impl)
                except TruncationError:
                    n_skipped[k] += wb.shape[0]
                    continue
                k_losses.append(loss_b)
                kept_idx.extend(range(i, i + wb.shape[0]))
            if not k_losses:
                by_k[str(k)] = {"eval_failed_all_batches": True, "n_skipped": n_skipped[k]}
                continue
            k_loss = torch.cat(k_losses, dim=0)
            kept_idx_t = torch.tensor(kept_idx, device=device, dtype=torch.long)
            matched_base_loss = base_loss[kept_idx_t]
            matched_base_tgt = base_tgt[kept_idx_t]
            damage = (k_loss - matched_base_loss).reshape(-1)
            tgt_flat = matched_base_tgt.reshape(-1)
            entry = stratify_damage(tgt_flat, damage, bin_of_id, cls_of_id, n_bins)
            entry["n_windows_skipped"] = n_skipped[k]
            entry["baseline_mean_loss"] = matched_base_loss.mean().item()
            entry["truncated_mean_loss"] = k_loss.mean().item()
            by_k[str(k)] = entry
        report[corpus_name] = {"windows": win_info, "by_k": by_k}
    return report


# ---------------------------------------------------------------------------
# Smoke gate. REQUIRES CUDA. Two items unique to this script (per the task
# brief's smoke-suite list): "truncation-intervention no-op at full rank"
# and "damage metric sanity on a synthetic case".
# ---------------------------------------------------------------------------

def smoke(device: str):
    print("=" * 60 + "\n  LM_INTERVENE_RD SMOKE GATE\n" + "=" * 60)
    assert device == "cuda", "chunk_delta_rule has no CPU path"
    torch.manual_seed(0)

    print("\n[1] damage metric sanity on a SYNTHETIC case (pure arithmetic, no model): "
          "hand-constructed token ids / damage values with KNOWN band/class membership, "
          "DELIBERATELY UNEVEN band counts so raw_mean != frequency_balanced_mean (a symmetric "
          "case would let the two formulas coincide by accident and not actually discriminate)")
    # vocab of 8 ids, n_bins=2: ids {0,1,2,3} most frequent (band 0), {4,5,6,7} least (band 1)
    bin_of_id = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
    # ids 0,4 = "word"; 1,5 = "symbol"; 2,3,6,7 = "other" (deliberately uneven so counts differ)
    cls_of_id = torch.tensor([0, 1, 2, 2, 0, 1, 2, 2], dtype=torch.int64)   # word,symbol,other,other,...
    # 4 tokens land in band 0 (ids 0,0,0,1), only 2 in band 1 (ids 4,5) -- uneven on purpose
    token_ids = torch.tensor([0, 0, 0, 1, 4, 5], device=device)
    damage = torch.tensor([1.0, 1.0, 1.0, 1.0, 10.0, 10.0], device=device)   # band0 mean=1.0, band1 mean=10.0
    entry = stratify_damage(token_ids, damage, bin_of_id, cls_of_id, n_bins=2)
    assert abs(entry["by_frequency_band"]["0"] - 1.0) < 1e-6, entry
    assert abs(entry["by_frequency_band"]["1"] - 10.0) < 1e-6, entry
    expected_raw_mean = (1.0 + 1.0 + 1.0 + 1.0 + 10.0 + 10.0) / 6   # = 4.0, token-count-weighted
    expected_balanced_mean = (1.0 + 10.0) / 2                        # = 5.5, equal-weight-per-band
    assert abs(entry["raw_mean"] - expected_raw_mean) < 1e-6, entry
    assert abs(entry["frequency_balanced_mean"] - expected_balanced_mean) < 1e-6, entry
    assert abs(expected_raw_mean - expected_balanced_mean) > 1.0, \
        "test construction bug: raw_mean and balanced_mean coincide, this case doesn't discriminate"
    # word ids present: 0 (x3, damage 1,1,1) and 4 (x1, damage 10) -> mean = (1+1+1+10)/4 = 3.25
    assert abs(entry["by_symbol_word_class"]["word"] - 3.25) < 1e-6, entry
    assert entry["by_symbol_word_class_counts"]["symbol"] == 2   # ids 1,5
    print(f"  band means {entry['by_frequency_band']} (raw_mean={entry['raw_mean']:.3f} != "
          f"frequency_balanced_mean={entry['frequency_balanced_mean']:.3f}, as required) and "
          f"symbol/word means {entry['by_symbol_word_class']} match hand-computed expectations exactly")

    print("\n[2] build_token_frequency_bins / build_symbol_word_class run on the REAL GPT-2 "
          "vocab and produce sane, finite, full-coverage output")
    tokenizer = load_gpt2_tokenizer()
    torch.manual_seed(2)
    fake_corpus = torch.randint(0, 50257, (50_000,))
    bins, counts = build_token_frequency_bins(50257, fake_corpus, n_bins=4)
    assert bins.shape == (50257,) and bins.min() >= 0 and bins.max() <= 3
    cls = build_symbol_word_class(tokenizer, 50257)
    assert cls.shape == (50257,)
    # positive control: a known word-piece and a known digit-piece land as expected
    the_id = tokenizer.encode(" the")[0]
    digit_id = tokenizer.encode("7")[0] if len(tokenizer.encode("7")) == 1 else None
    assert cls[the_id].item() == _SYM_CLASSES.index("word"), f"' the' misclassified: {tokenizer.convert_ids_to_tokens([the_id])}"
    if digit_id is not None:
        assert cls[digit_id].item() == _SYM_CLASSES.index("symbol"), \
            f"'7' misclassified: {tokenizer.convert_ids_to_tokens([digit_id])}"
    print(f"  4-bin frequency table + symbol/word class built over the real 50,257-token vocab "
          f"(positive controls: ' the'->word, '7'->symbol)")

    print("\n[3] forward/backward-free intervention smoke at LM shapes (tiny model, synthetic "
          "corpus): truncation_damage runs at several k, per-token loss finite")
    V = 500
    model = DeltaNetLM(V, d_model=64, d_state=64, n_layers=1, conv_size=4).to(device)
    model.eval()
    gen = torch.Generator(device=device).manual_seed(3)
    corpus = torch.randint(0, V, (50_000,), device=device)
    ctx_len, cont_len = _MIN_KERNEL_T, _MIN_KERNEL_T
    windows = sample_windows(corpus, 6, ctx_len, cont_len, gen)
    base_loss, base_tgt = truncation_damage(model, windows, ctx_len, cont_len, k=None)
    assert base_loss.shape == (6, cont_len)
    assert torch.isfinite(base_loss).all()
    k4_loss, _ = truncation_damage(model, windows, ctx_len, cont_len, k=4)
    assert torch.isfinite(k4_loss).all()
    print(f"  baseline mean loss {base_loss.mean().item():.4f}, k=4 mean loss {k4_loss.mean().item():.4f} "
          f"(both finite, as expected at an untrained model)")

    print("\n[4] TRUNCATION-INTERVENTION NO-OP AT FULL RANK: k=d_state must leave the "
          "continuation loss (and the underlying state) numerically unchanged vs the k=None "
          "baseline -- S always has rank <= d_state, so a rank-d_state truncation is the "
          "identity operator up to floating-point reconstruction error")
    d_state = 64
    _, ctx_states = model(windows[:, :ctx_len], initial_states=None, return_states=True)
    S = ctx_states[0]
    from rank_utils import truncate_to_rank
    S_full = truncate_to_rank(S, d_state)
    state_diff = (S - S_full).abs().max().item()
    assert state_diff < 1e-2, f"STATE-LEVEL no-op check failed: max abs diff {state_diff:.4f} at k=d_state"
    full_loss, _ = truncation_damage(model, windows, ctx_len, cont_len, k=d_state)
    loss_diff = (full_loss - base_loss).abs().mean().item()
    assert loss_diff < 0.05, f"END-TO-END no-op check failed: mean abs loss diff {loss_diff:.4f} at k=d_state"
    print(f"  state-level max|S - truncate(S,d_state)| = {state_diff:.2e}; "
          f"end-to-end mean|loss(k=d_state) - loss(None)| = {loss_diff:.4f} (both near-zero, as required)")

    print("\n[5] partial truncation (k < d_state) genuinely CHANGES the state/loss -- sensitivity "
          "control confirming item [4] isn't vacuous (a no-op check with no way to fail proves "
          "nothing; this shows the harness CAN detect a real difference)")
    k_small_diff = (k4_loss - base_loss).abs().mean().item()
    assert k_small_diff > loss_diff, (
        f"k=4 truncation damage ({k_small_diff:.4f}) is not larger than the k=d_state no-op "
        f"residual ({loss_diff:.4f}) -- the no-op check in [4] would be vacuous"
    )
    print(f"  k=4 mean|loss delta| = {k_small_diff:.4f}  >  k=d_state residual = {loss_diff:.4f} "
          f"(the no-op check has teeth)")

    print("\n" + "=" * 60 + "\n  ALL LM_INTERVENE_RD SMOKE CHECKS PASSED\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--checkpoint", type=str, default=None, help="path to a lm_pretrain_rd.py torch.save checkpoint")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--k-grid", type=int, nargs="+", default=list(DEFAULT_K_GRID))
    ap.add_argument("--ctx-len", type=int, default=384)
    ap.add_argument("--cont-len", type=int, default=128)
    ap.add_argument("--n-eval-windows", type=int, default=32,
                     help="capped window count per corpus (house VRAM rule; eval-only, no backward pass)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--freq-bins", type=int, default=4)
    ap.add_argument("--trunc-impl", choices=sorted(TRUNC_IMPLS), default="eigh")
    ap.add_argument("--seed", type=int, default=0,
                     help="AUDIT FIX-1: retained for CLI compatibility but deliberately UNUSED by "
                          "the window sampler -- eval windows are seeded from "
                          "corpus_fixed_seed(corpus) so every checkpoint (from any training seed) "
                          "is measured on the identical window set.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        smoke(device)
        return

    assert device == "cuda", "lm_intervene_rd requires CUDA (chunk_delta_rule has no CPU path)"
    assert args.checkpoint is not None, "--checkpoint is required for a real (non-smoke) run"
    assert args.ctx_len >= _MIN_KERNEL_T, f"--ctx-len={args.ctx_len} < _MIN_KERNEL_T={_MIN_KERNEL_T}"
    assert args.cont_len >= _MIN_KERNEL_T, f"--cont-len={args.cont_len} < _MIN_KERNEL_T={_MIN_KERNEL_T}"

    t0 = time.time()
    tf32_state = set_and_log_tf32()            # audit non-blocking item: explicit + logged
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print(f"loaded checkpoint {args.checkpoint} (step={ckpt.get('step')}, corpus={ckpt.get('corpus')}, "
          f"config={ckpt.get('config')})  tf32={tf32_state}", flush=True)

    corpora = sorted(CORPUS_DIRS)
    tokenizer = load_gpt2_tokenizer()
    corpus_val = {}
    train_tensors = []
    for name in corpora:
        train_t, val_t, meta, _, val_offs = load_corpus(args.data_dir, name, device)
        corpus_val[name] = (val_t, val_offs)
        train_tensors.append(train_t)

    vocab_size = model.vocab_size
    bin_of_id, freq_counts = build_token_frequency_bins(vocab_size, *train_tensors, n_bins=args.freq_bins)
    cls_of_id = build_symbol_word_class(tokenizer, vocab_size)

    report = run_intervention(model, corpus_val, args.k_grid, args.ctx_len, args.cont_len,
                               args.n_eval_windows, args.batch_size, bin_of_id, cls_of_id,
                               args.freq_bins, device, trunc_impl=args.trunc_impl)

    result = {
        "claim_tier": CLAIM_TIER,
        "fallback_note": ("MAJOR-5's pre-registered fallback (DELTANET_REALDATA_DESIGN.md section 6.1): "
                           "the symbol-density-matched third-corpus arm was DEFERRED this build phase; "
                           "these frequency-band / symbol-vs-word-stratified damage metrics are the "
                           "pre-registered substitute, reported as the fallback it is -- NOT claimed "
                           "equivalent to a genuine matched-corpus contrast."),
        "checkpoint": args.checkpoint, "checkpoint_step": ckpt.get("step"),
        "checkpoint_train_corpus": ckpt.get("corpus"), "model_config": model.config(),
        "k_grid": args.k_grid, "ctx_len": args.ctx_len, "cont_len": args.cont_len,
        "n_eval_windows": args.n_eval_windows, "freq_bins": args.freq_bins,
        "trunc_impl": args.trunc_impl,
        "window_seed_policy": "corpus_fixed (AUDIT FIX-1: crc32(corpus), --seed does NOT affect windows)",
        "tf32": tf32_state,
        "damage_by_corpus_and_k": report,
        "wall_s": time.time() - t0,
        "complete": True,
    }
    summary = {k: v for k, v in result.items() if k != "damage_by_corpus_and_k"}
    print("\nRESULT SUMMARY:", json.dumps(summary, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
