"""lm_geo3_wave_neg1_gate.py -- Track B / geo3-in-LM Wave -1 MEASUREMENT
tooling (SCALE_TRANSFER_DESIGN.md sec 4.2/4.5). Runs BEFORE any training
wave, on ALREADY-ARCHIVED checkpoints (Wave C's, per sec 4.2's "load an
already-archived Wave C checkpoint... and run a forward-pass-only probe (no
training)") -- no backward pass, no gradient, no training loop anywhere in
this file.

Measures, per 64-token chunk (chunk_size configurable, default matches
sec 4.2 item 1's kernel-aligned window), TWO REGISTERED gate criteria, both
decision-bearing (sec 4.2, Rev 2 -- MAJOR-2):

  (a) beta-concentration: the fraction of per-chunk beta-mass captured by
      the top-K_sel positions (K_sel=32, sec 4.2's literal "top-32" text),
      plus a Gini coefficient as a shape check.
      REGISTERED THRESHOLD: top-32 mass frac >= 0.60.
  (b) excluded-position write-mass: the mean beta at NON-selected positions,
      AND "the complement of (a)" (sec 4.2's own words) -- the fraction of
      total chunk write-mass the orthogonalization never touches.
      REGISTERED THRESHOLD: mean non-selected beta <= 0.25 AND
      non-selected write-mass <= 0.40 of chunk total.

Registered routing (sec 4.2, literal): (i) both pass -> "beta_gated_primary"
(the beta-gated top-K construction proceeds as Wave 1's primary arm);
(ii) (a) fails, (b) passes -> "naive_window_primary" (fall back to the
naive-fixed-window arm, sec 4.5's Wave 2, AS Wave 1's primary); (iii) (b)
fails (REGARDLESS of (a)) -> "no_launch_redesign" -- a HARD no-launch. The
launcher (run_lm_rd_geo3_sweep.py) reads this script's own JSON output's
top-level "gate_verdict" field and REFUSES to launch Wave 1 when it is
"no_launch_redesign" -- this script's own CLI additionally exits non-zero
(exit code 3) on that verdict so a shell-level `&&`-chained launch also
fails closed without needing to parse JSON.

DOCUMENTED MEASUREMENT-CONVENTION NOTE (this build's own finding, confirmed
and SHARPENED by the independent audit's own re-derivation, round 2 NIT-7):
this script computes (b)'s write-mass sub-criterion as the EXACT complement
of (a)'s concentration fraction over the SAME chunk-total beta-mass
denominator -- the most literal reading of sec 4.2's own "(b) ... and the
complement of (a)" framing. Under that reading, b_write_pass <=> a_pass
IDENTICALLY (the thresholds 0.60 and 0.40 sum to exactly 1.00), so outcome
(ii) is ALGEBRAICALLY UNREACHABLE by construction -- for ANY beta
distribution whatsoever, not merely empirically rare (an earlier draft of
this note said "measure-zero in practice"; the audit's re-derivation showed
the stronger statement holds). The routing LOGIC itself
(gate_verdict_from_bools) is written and tested independently of this
measurement convention (it takes a_pass/b_pass as independent booleans),
so outcome (ii) remains a correctly-routed decision-logic branch -- it is
the MEASUREMENT convention, inherited from the design doc's own threshold
choice, that forecloses it. Reported prominently in every gate JSON's own
"measurement_convention_note" field, not silently absorbed; the design
owner should confirm whether the 3-way branch was ever meant to be
genuinely reachable (if so, criteria (a) and (b) need different
denominators/populations, a design change outside this build's authority).

Usage:
  python lm_geo3_wave_neg1_gate.py --smoke
  python lm_geo3_wave_neg1_gate.py --checkpoints <path1.pt> [<path2.pt> ...] \\
      --data-dir /data/deltanet_rd_data --out results/geo3_lm/wave_neg1_gate.json
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

from lm_pretrain_rd import (CORPUS_DIRS, EOT_TOKEN_ID, DEFAULT_DATA_DIR, DeltaNetLM,
                             GEO3_LM_CHUNK_SIZE_DEFAULT, corpus_fixed_seed, get_batch,
                             gini_coefficient, load_corpus, set_and_log_tf32)

DEFAULT_K_SELS = (16, 32)
GATE_K_SEL = 32                            # sec 4.2's literal "top-32" gate criterion
CONCENTRATION_THRESHOLD = 0.60
NONSELECTED_BETA_THRESHOLD = 0.25
NONSELECTED_WRITE_MASS_THRESHOLD = 0.40


# ---------------------------------------------------------------------------
# Measurement: per-chunk beta statistics
# ---------------------------------------------------------------------------

def beta_chunk_stats(beta: torch.Tensor, content_mask: torch.Tensor, chunk_size: int,
                      k_sel: int) -> dict:
    """beta: (B,T,H) post-sigmoid gate values. content_mask: (B,T) bool,
    True = eligible for top-K_sel selection (mirrors
    lm_pretrain_rd._geo3_lm_select_and_orthogonalize's own selection pool
    EXACTLY -- EOT/excluded positions are never selection candidates, but
    DO still count toward the chunk's total beta-mass denominator, since
    their write is real and unmasked in LM mode, sec 4.2's own point).

    Returns four RAW (not yet reduced) per-chunk-episode tensors, each
    (B,n_chunks,H):
      - top_k_mass_frac: fraction of the chunk's TOTAL beta-mass (ALL
        positions, including excluded ones) captured by the top-k_sel
        eligible positions.
      - gini: Gini coefficient of the FULL chunk beta distribution.
      - mean_nonselected_beta: mean beta over every position NOT in the
        selected set (excluded positions AND any eligible position that
        didn't make the cut).
      - nonselected_write_mass_frac: 1 - top_k_mass_frac (sec 4.2's own
        "the complement of (a)" framing, implemented literally)."""
    B, T, H = beta.shape
    assert T % chunk_size == 0, f"T={T} not a multiple of chunk_size={chunk_size}"
    n_chunks = T // chunk_size
    beta_c = beta.view(B, n_chunks, chunk_size, H)
    content_c = content_mask.view(B, n_chunks, chunk_size, 1).expand(B, n_chunks, chunk_size, H)

    total_mass = beta_c.sum(dim=2)                                        # (B,n_chunks,H)
    neg_inf = torch.finfo(beta_c.dtype).min
    k_eff = min(k_sel, chunk_size)
    priority = torch.where(content_c, beta_c, torch.full_like(beta_c, neg_inf))
    topk_val, topk_idx = torch.topk(priority, k_eff, dim=2)               # (B,n_chunks,k_eff,H)
    valid = topk_val > (neg_inf / 2)

    topk_mass = torch.where(valid, topk_val, torch.zeros_like(topk_val)).sum(dim=2)  # (B,n_chunks,H)
    safe_total = total_mass.clamp(min=1e-12)
    top_k_mass_frac = topk_mass / safe_total
    nonselected_write_mass_frac = 1.0 - top_k_mass_frac

    selected_mask = torch.zeros(B, n_chunks, chunk_size, H, dtype=torch.bool, device=beta.device)
    selected_mask.scatter_(2, topk_idx, valid)
    nonselected_mask = ~selected_mask                                     # includes EOT/excluded positions
    nonselected_count = nonselected_mask.float().sum(dim=2).clamp(min=1.0)
    mean_nonselected_beta = (beta_c * nonselected_mask).sum(dim=2) / nonselected_count

    gini = gini_coefficient(beta_c.permute(0, 1, 3, 2))                   # (B,n_chunks,H)

    return {"top_k_mass_frac": top_k_mass_frac, "gini": gini,
            "mean_nonselected_beta": mean_nonselected_beta,
            "nonselected_write_mass_frac": nonselected_write_mass_frac}


# ---------------------------------------------------------------------------
# Decision logic (pure, independently testable -- see this file's module
# docstring for why it is deliberately decoupled from the measurement
# convention above)
# ---------------------------------------------------------------------------

def gate_verdict_from_bools(a_pass: bool, b_pass: bool) -> str:
    """sec 4.2's registered outcome routing, literally: (i) both pass ->
    beta_gated_primary; (ii) a fails, b passes -> naive_window_primary;
    (iii) b fails (REGARDLESS of a) -> no_launch_redesign."""
    if not b_pass:
        return "no_launch_redesign"
    return "beta_gated_primary" if a_pass else "naive_window_primary"


def evaluate_gate(top_k_mass_frac_mean: float, mean_nonselected_beta_mean: float,
                   nonselected_write_mass_frac_mean: float,
                   concentration_threshold: float = CONCENTRATION_THRESHOLD,
                   nonselected_beta_threshold: float = NONSELECTED_BETA_THRESHOLD,
                   nonselected_write_mass_threshold: float = NONSELECTED_WRITE_MASS_THRESHOLD) -> dict:
    a_pass = top_k_mass_frac_mean >= concentration_threshold
    b_write_pass = nonselected_write_mass_frac_mean <= nonselected_write_mass_threshold
    b_beta_pass = mean_nonselected_beta_mean <= nonselected_beta_threshold
    b_pass = b_write_pass and b_beta_pass
    verdict = gate_verdict_from_bools(a_pass, b_pass)
    return {
        "a_pass": a_pass, "b_pass": b_pass, "b_write_pass": b_write_pass, "b_beta_pass": b_beta_pass,
        "verdict": verdict,
        "measured": {"top_k_mass_frac_mean": top_k_mass_frac_mean,
                     "mean_nonselected_beta_mean": mean_nonselected_beta_mean,
                     "nonselected_write_mass_frac_mean": nonselected_write_mass_frac_mean},
        "thresholds": {"concentration_threshold": concentration_threshold,
                       "nonselected_beta_threshold": nonselected_beta_threshold,
                       "nonselected_write_mass_threshold": nonselected_write_mass_threshold},
    }


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors lm_intervene_rd.load_checkpoint, duplicated
# here per this codebase's own pod-safety/self-containment convention rather
# than cross-imported)
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: str) -> tuple[DeltaNetLM, dict]:
    ckpt = torch.load(path, map_location=device)
    model = DeltaNetLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# beta capture: a non-invasive forward hook on each layer's b_proj -- NO
# model code changes required for this measurement (works on ANY archived
# checkpoint, including ones trained with geo3_active=False, i.e. every
# Wave C checkpoint that exists today -- the whole point of Wave -1).
# ---------------------------------------------------------------------------

def capture_betas(model: DeltaNetLM, batches: list, device: str) -> tuple[dict, torch.Tensor]:
    """batches: list of (B,T) int64 token_id tensors. Returns
    (betas_by_layer: {layer_idx: (sum_B,T,H) tensor}, token_ids_cat:
    (sum_B,T))."""
    captured = {i: [] for i in range(len(model.blocks))}

    def make_hook(i):
        def hook(module, inp, out):
            captured[i].append(torch.sigmoid(out.detach()))
        return hook

    handles = [blk.mixer.b_proj.register_forward_hook(make_hook(i))
               for i, blk in enumerate(model.blocks)]
    try:
        with torch.no_grad():
            for x in batches:
                _ = model(x)
    finally:
        for h in handles:
            h.remove()
    betas_by_layer = {i: torch.cat(v, dim=0) for i, v in captured.items()}
    token_ids_cat = torch.cat(batches, dim=0)
    return betas_by_layer, token_ids_cat


# ---------------------------------------------------------------------------
# Top-level measurement across checkpoints x corpora x layers x k_sels
# ---------------------------------------------------------------------------

def run_measurement(checkpoint_paths: list, data_dir: str, k_sels, chunk_size: int,
                     n_windows: int, batch_size: int, seq_len: int, device: str) -> dict:
    per_checkpoint = {}
    # pooled[k_sel] accumulates RAW per-episode tensors across EVERY
    # (checkpoint, corpus, layer) cell -- the gate's actual decision input
    # (see this file's module docstring: pooling across seeds/corpora/layers
    # is deliberate, not an oversight -- avoids small-n seed-spread noise
    # driving a launch/no-launch call, matching this project's own
    # "report per-seed AND pooled" convention).
    pooled = {k: {"top_k_mass_frac": [], "gini": [], "mean_nonselected_beta": [],
                  "nonselected_write_mass_frac": []} for k in k_sels}

    for ckpt_path in checkpoint_paths:
        model, ckpt = load_checkpoint(ckpt_path, device)
        assert model.d_model and model.n_layers, "loaded checkpoint has a degenerate config"
        per_corpus = {}
        for corpus_name in sorted(CORPUS_DIRS):
            _, val_tokens, meta, _, val_offs = load_corpus(data_dir, corpus_name, device)
            gen = torch.Generator(device=device).manual_seed(corpus_fixed_seed(corpus_name) + 90_000)
            n_batches = max(1, -(-n_windows // batch_size))               # ceil
            batches = [get_batch(val_tokens, batch_size, seq_len, gen)[0] for _ in range(n_batches)]
            betas_by_layer, token_ids_cat = capture_betas(model, batches, device)
            content_mask = (token_ids_cat != EOT_TOKEN_ID)

            per_layer = {}
            for layer_idx, beta_cat in betas_by_layer.items():
                per_k = {}
                for k_sel in k_sels:
                    stats = beta_chunk_stats(beta_cat, content_mask, chunk_size, k_sel)
                    per_k[k_sel] = {
                        "top_k_mass_frac_mean": stats["top_k_mass_frac"].mean().item(),
                        "gini_mean": stats["gini"].mean().item(),
                        "mean_nonselected_beta_mean": stats["mean_nonselected_beta"].mean().item(),
                        "nonselected_write_mass_frac_mean": stats["nonselected_write_mass_frac"].mean().item(),
                        "n_chunk_episodes": int(stats["top_k_mass_frac"].numel()),
                    }
                    for key in ("top_k_mass_frac", "gini", "mean_nonselected_beta",
                                "nonselected_write_mass_frac"):
                        pooled[k_sel][key].append(stats[key].reshape(-1))
                per_layer[layer_idx] = per_k
            per_corpus[corpus_name] = {"per_layer": per_layer,
                                        "n_windows_sampled": int(token_ids_cat.shape[0])}
        per_checkpoint[ckpt_path] = {"corpus_trained_on": ckpt.get("corpus"),
                                      "seed": ckpt.get("seed"), "step": ckpt.get("step"),
                                      "per_corpus": per_corpus}
        # AUDIT ROUND-2 NIT-6: ckpt's model_state_dict is a SECOND full copy of every parameter --
        # free both explicitly + release the CUDA cache between checkpoints, so this same script
        # stays safe if reused at Track C's 100M-1.3B scales (the design doc's own named
        # future-reuse target), not just today's ~14M checkpoints.
        del model, ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pooled_summary = {}
    for k_sel in k_sels:
        cat = {key: torch.cat(v) for key, v in pooled[k_sel].items() if v}
        pooled_summary[k_sel] = {
            "top_k_mass_frac_mean": cat["top_k_mass_frac"].mean().item(),
            "gini_mean": cat["gini"].mean().item(),
            "mean_nonselected_beta_mean": cat["mean_nonselected_beta"].mean().item(),
            "nonselected_write_mass_frac_mean": cat["nonselected_write_mass_frac"].mean().item(),
            "n_pooled_chunk_episodes": int(cat["top_k_mass_frac"].numel()),
        }

    assert GATE_K_SEL in pooled_summary, \
        f"GATE_K_SEL={GATE_K_SEL} was not among the measured k_sels {sorted(pooled_summary)} -- " \
        f"main() should have refused this at CLI-parse time (AUDIT ROUND-2 MINOR-3)"
    gate = evaluate_gate(pooled_summary[GATE_K_SEL]["top_k_mass_frac_mean"],
                          pooled_summary[GATE_K_SEL]["mean_nonselected_beta_mean"],
                          pooled_summary[GATE_K_SEL]["nonselected_write_mass_frac_mean"])

    return {
        "design_ref": "SCALE_TRANSFER_DESIGN.md sec 4.2 (Wave -1 gate) / sec 4.5",
        "gate_k_sel": GATE_K_SEL,
        "k_sels_measured": list(k_sels),
        "chunk_size": chunk_size, "n_windows_per_corpus_per_checkpoint": n_windows,
        "seq_len": seq_len, "checkpoints": checkpoint_paths,
        "measurement_convention_note": (
            "nonselected_write_mass_frac is computed as the EXACT complement of top_k_mass_frac "
            "over the SAME chunk-total beta-mass denominator (sec 4.2's own 'the complement of (a)' "
            "framing, implemented literally). Under this convention, b_write_pass <=> a_pass "
            "IDENTICALLY (the thresholds 0.60+0.40 sum to exactly 1.00) -- outcome (ii) "
            "'naive_window_primary' is therefore ALGEBRAICALLY UNREACHABLE by construction, for any "
            "beta distribution whatsoever, not merely empirically rare (wording strengthened per "
            "the independent audit's own re-derivation, round 2 NIT-7). The decision logic "
            "(gate_verdict_from_bools) is independently correct and tested for all three outcomes; "
            "flagged for the design owner to confirm whether the 3-way branch was ever meant to be "
            "genuinely reachable -- see this file's module docstring."),
        "pooled": pooled_summary,
        "gate": gate,
        "gate_verdict": gate["verdict"],                     # top-level field per the build brief
        "per_checkpoint": per_checkpoint,
    }


# ---------------------------------------------------------------------------
# Smoke gate. Two independent halves per the build brief: (A) beta_chunk_stats
# on HAND-COMPUTED synthetic beta distributions with KNOWN exact expected
# fractions (positive AND negative cases, run to completion); (B)
# gate_verdict_from_bools exercised at all three registered outcomes
# directly (decoupled from the measurement convention, per this file's
# module docstring).
# ---------------------------------------------------------------------------

def smoke(device: str):
    print("=" * 60 + "\n  LM_GEO3_WAVE_NEG1_GATE SMOKE GATE\n" + "=" * 60)

    print("\n[1] beta_chunk_stats on a HAND-COMPUTED chunk (pure arithmetic, no model): "
          "chunk_size=64, k_sel=32, top-32 positions = 0.90, bottom-32 = 0.05, no EOT exclusions "
          "-- exact expected top_k_mass_frac / mean_nonselected_beta / write_mass computed by hand")
    B1, T1, H1 = 1, 64, 1
    beta1 = torch.cat([torch.full((32,), 0.90), torch.full((32,), 0.05)]).view(1, T1, 1)
    content1 = torch.ones(B1, T1, dtype=torch.bool)
    stats1 = beta_chunk_stats(beta1, content1, chunk_size=64, k_sel=32)
    top_mass_exp = 32 * 0.90
    total_exp = 32 * 0.90 + 32 * 0.05
    frac_exp = top_mass_exp / total_exp                            # = 28.8/30.4 = 0.947368...
    assert abs(stats1["top_k_mass_frac"].item() - frac_exp) < 1e-5, stats1
    assert abs(stats1["mean_nonselected_beta"].item() - 0.05) < 1e-5, stats1
    assert abs(stats1["nonselected_write_mass_frac"].item() - (1 - frac_exp)) < 1e-5, stats1
    print(f"  top_k_mass_frac={stats1['top_k_mass_frac'].item():.6f} (expected {frac_exp:.6f}), "
          f"mean_nonselected_beta={stats1['mean_nonselected_beta'].item():.4f} (expected 0.0500), "
          f"nonselected_write_mass_frac={stats1['nonselected_write_mass_frac'].item():.6f} "
          f"(expected {1 - frac_exp:.6f}) -- EXACT match")

    print("\n[2] beta_chunk_stats EOT-exclusion: an EOT position with the HIGHEST beta in the "
          "chunk must be excluded from top-32 selection (never counted toward topk_mass) but "
          "STILL counts toward total_mass and toward 'non-selected' -- hand-verified")
    beta2 = torch.cat([torch.full((32,), 0.90), torch.full((32,), 0.05)]).view(1, T1, 1)
    beta2[0, 0, 0] = 0.99                                            # highest beta in the chunk
    content2 = torch.ones(B1, T1, dtype=torch.bool)
    content2[0, 0] = False                                           # position 0 is EOT-excluded
    stats2 = beta_chunk_stats(beta2, content2, chunk_size=64, k_sel=32)
    # top-32 CONTENT positions are now positions 1..32 (31 at 0.90, plus position 32 at 0.05
    # can't beat 0.90 -- so top-32 content = 31x0.90 + 1x0.90? there are only 31 remaining 0.90s
    # (position 0 excluded) plus 33 positions at 0.05 remain in the "bottom" pool of 63 content
    # positions; top-32 by content beta = the 31 remaining 0.90s + the single largest 0.05 = 0.05
    top_mass_exp2 = 31 * 0.90 + 1 * 0.05
    total_exp2 = 0.99 + 32 * 0.90 + 32 * 0.05 - 0.90                  # total INCLUDES the EOT's 0.99
    # (total_mass = sum of ALL 64 positions = original 30.4 - the position-0 value (0.90, now
    # overwritten to 0.99) = 30.4 - 0.90 + 0.99)
    total_exp2 = (32 * 0.90 + 32 * 0.05) - 0.90 + 0.99
    frac_exp2 = top_mass_exp2 / total_exp2
    assert abs(stats2["top_k_mass_frac"].item() - frac_exp2) < 1e-4, \
        (stats2["top_k_mass_frac"].item(), frac_exp2)
    print(f"  EOT (beta=0.99, chunk max) excluded from selection: top_k_mass_frac="
          f"{stats2['top_k_mass_frac'].item():.6f} (expected {frac_exp2:.6f}, computed from the "
          f"NEXT-best 32 CONTENT positions, not the EOT) -- EXACT match; EOT's own high beta still "
          f"counts in the denominator (total_exp2={total_exp2:.4f}), inflating "
          f"nonselected_write_mass_frac as sec 4.2's own risk framing intends")

    print("\n[3] evaluate_gate CASE 1 (BOTH PASS -> beta_gated_primary): the item-[1] distribution "
          "(top32_frac=0.9474>=0.60, mean_nonselected=0.05<=0.25, write_mass=0.0526<=0.40)")
    gate1 = evaluate_gate(frac_exp, 0.05, 1 - frac_exp)
    assert gate1["a_pass"] is True and gate1["b_pass"] is True
    assert gate1["verdict"] == "beta_gated_primary", gate1
    print(f"  verdict={gate1['verdict']} (as hand-computed)")

    print("\n[4] evaluate_gate CASE 2 -- NEGATIVE CASE, run to completion (uniform chunk: top32_frac"
          "=0.50<0.60 -> (a) FAILS; write_mass=0.50>0.40 -> (b)'s write sub-check ALSO fails "
          "-> (b) FAILS -> no_launch_redesign, the HARD no-launch branch)")
    gate2 = evaluate_gate(top_k_mass_frac_mean=0.50, mean_nonselected_beta_mean=0.10,
                           nonselected_write_mass_frac_mean=0.50)
    assert gate2["a_pass"] is False, gate2
    assert gate2["b_write_pass"] is False, gate2
    assert gate2["b_pass"] is False, gate2
    assert gate2["verdict"] == "no_launch_redesign", gate2
    print(f"  verdict={gate2['verdict']} (a_pass={gate2['a_pass']}, b_pass={gate2['b_pass']}) -- "
          f"NEGATIVE CASE CONFIRMED FAILING AS EXPECTED (this IS the intended outcome, not a bug)")

    print("\n[5] evaluate_gate CASE 3 -- a SECOND, DIFFERENT negative case: (a) passes AND (b)'s "
          "write-mass sub-check passes, but (b)'s MEAN-beta sub-check independently fails "
          "(top32=0.40 each, bottom32=0.26 each -> top_frac=0.6061>=0.60 passes; write_mass="
          "0.3939<=0.40 passes; mean_nonselected_beta=0.26>0.25 FAILS) -> b FAILS overall -> "
          "no_launch_redesign via a DIFFERENT failure mode than case 2 (proves the mean-beta "
          "sub-criterion has independent teeth, not just the write-mass one)")
    top_mass3 = 32 * 0.40
    bot_mass3 = 32 * 0.26
    frac3 = top_mass3 / (top_mass3 + bot_mass3)
    gate3 = evaluate_gate(frac3, 0.26, 1 - frac3)
    assert gate3["a_pass"] is True, gate3
    assert gate3["b_write_pass"] is True, gate3
    assert gate3["b_beta_pass"] is False, gate3
    assert gate3["b_pass"] is False, gate3
    assert gate3["verdict"] == "no_launch_redesign", gate3
    print(f"  top_k_mass_frac={frac3:.4f} (a_pass={gate3['a_pass']}), b_write_pass="
          f"{gate3['b_write_pass']}, b_beta_pass={gate3['b_beta_pass']} (FAILS as expected) -> "
          f"verdict={gate3['verdict']} -- SECOND NEGATIVE CASE CONFIRMED, distinct failure mode")

    print("\n[6] gate_verdict_from_bools exercised DIRECTLY at all three registered outcomes "
          "(decoupled from the measurement convention, per this file's module docstring -- proves "
          "the DECISION logic supports outcome (ii) even though case [3]/[4]'s real measurement "
          "convention structurally cannot reach it)")
    assert gate_verdict_from_bools(True, True) == "beta_gated_primary"
    assert gate_verdict_from_bools(False, True) == "naive_window_primary"
    assert gate_verdict_from_bools(True, False) == "no_launch_redesign"
    assert gate_verdict_from_bools(False, False) == "no_launch_redesign"
    print("  all four (a_pass,b_pass) combinations route to the correct registered outcome, "
          "including (False,True)->naive_window_primary (ALGEBRAICALLY UNREACHABLE under the real "
          "measurement convention, per this file's own documented note, but a correctly-routed "
          "decision-logic branch)")

    print("\n[7] run_measurement end-to-end on a TINY (but REAL-vocab-sized) SYNTHETIC (untrained) "
          "model + synthetic EOT-separated corpus -- shape/finiteness/JSON-serializability smoke, "
          "NOT a claim about real beta behavior (an untrained model's beta is near-random; this "
          "item exists only to prove the plumbing -- checkpoint load, corpus load (which HARD-"
          "ASSERTS the literal EOT_TOKEN_ID=50256, sec 4.2/load_corpus's own contract -- hence the "
          "REAL vocab_size=50257 here, not a tiny synthetic one, so the literal EOT id is "
          "embeddable), hook capture, pooling, JSON assembly -- runs end to end without crashing)")
    import tempfile
    V7 = 50257                                                       # REAL GPT-2 vocab, matches
                                                                       # load_corpus's own literal
                                                                       # EOT_TOKEN_ID=50256 assertion
    torch.manual_seed(0)
    model7 = DeltaNetLM(V7, d_model=32, d_state=64, n_layers=1, conv_size=4).to(device)
    tmpdir = tempfile.mkdtemp(prefix="geo3_gate_smoke_")
    ckpt_path7 = os.path.join(tmpdir, "fake_ckpt.pt")
    torch.save({"step": 0, "model_state_dict": model7.state_dict(), "config": model7.config(),
                "corpus": "openr1", "seed": 0, "run_name": "smoke"}, ckpt_path7)

    data_dir7 = os.path.join(tmpdir, "data")
    for name, dirname in CORPUS_DIRS.items():
        d = os.path.join(data_dir7, dirname)
        os.makedirs(d, exist_ok=True)
        n_tok = 20_000
        toks = torch.randint(0, V7, (n_tok,), dtype=torch.int64)
        offsets = [0]
        pos = 0
        while True:
            pos += 200
            if pos >= n_tok - 1:
                break
            toks[pos] = EOT_TOKEN_ID                                  # literal, matches load_corpus's assert
            offsets.append(pos + 1)
        offsets_t = torch.tensor(offsets, dtype=torch.int64)
        torch.save(toks, os.path.join(d, "train.pt"))
        torch.save(toks, os.path.join(d, "val.pt"))
        torch.save(offsets_t, os.path.join(d, "train_doc_offsets.pt"))
        torch.save(offsets_t, os.path.join(d, "val_doc_offsets.pt"))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"vocab_size": V7, "tokenizer": "gpt2", "eot_separated": True}, f)

    result7 = run_measurement([ckpt_path7], data_dir7, k_sels=(16, 32), chunk_size=64,
                               n_windows=8, batch_size=4, seq_len=128, device=device)
    assert result7["gate_verdict"] in ("beta_gated_primary", "naive_window_primary", "no_launch_redesign")
    json.dumps(result7)                                              # must be JSON-serializable
    print(f"  end-to-end smoke run completed: gate_verdict={result7['gate_verdict']!r} "
          f"(untrained model, NOT a real measurement -- plumbing check only), "
          f"n_pooled_chunk_episodes[k=32]={result7['pooled'][32]['n_pooled_chunk_episodes']}, "
          f"JSON-serializable")

    print("\n" + "=" * 60 + "\n  ALL LM_GEO3_WAVE_NEG1_GATE SMOKE CHECKS PASSED\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--checkpoints", type=str, nargs="+", default=None,
                     help="one or more Wave C (or Track B) checkpoint .pt paths -- REQUIRED for a "
                          "real (non-smoke) run. Pooling across multiple checkpoints (e.g. all 3 "
                          "seeds x 2 corpora) is the intended usage (reduces small-n seed-spread "
                          "noise in the gate decision; per-checkpoint numbers are still reported).")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--k-sels", type=int, nargs="+", default=list(DEFAULT_K_SELS))
    ap.add_argument("--chunk-size", type=int, default=GEO3_LM_CHUNK_SIZE_DEFAULT)
    ap.add_argument("--n-windows", type=int, default=64, help="eval windows sampled PER CORPUS PER CHECKPOINT")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512, help="must match the checkpoints' own training seq_len (Wave C: 512)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        smoke(device)
        return

    assert device == "cuda", "lm_geo3_wave_neg1_gate requires CUDA (chunk_delta_rule has no CPU path)"
    assert args.checkpoints, "--checkpoints is required for a real (non-smoke) run"
    assert args.seq_len % args.chunk_size == 0, \
        f"--seq-len={args.seq_len} must be a multiple of --chunk-size={args.chunk_size}"
    # AUDIT ROUND-2 MINOR-3: fail at CLI-parse time, not with a KeyError AFTER every checkpoint's
    # forward passes have already been spent -- the registered decision quantity is GATE_K_SEL's cell.
    assert GATE_K_SEL in args.k_sels, (
        f"--k-sels {args.k_sels} omits GATE_K_SEL={GATE_K_SEL} -- sec 4.2's registered gate "
        f"criterion (literal 'top-{GATE_K_SEL}') cannot be evaluated without it."
    )

    t0 = time.time()
    tf32_state = set_and_log_tf32()
    result = run_measurement(args.checkpoints, args.data_dir, args.k_sels, args.chunk_size,
                              args.n_windows, args.batch_size, args.seq_len, device)
    result["tf32"] = tf32_state
    result["wall_s"] = time.time() - t0

    summary = {k: v for k, v in result.items() if k != "per_checkpoint"}
    print("\n" + "=" * 70)
    print("WAVE -1 GATE RESULT SUMMARY:", json.dumps(summary, indent=2))
    print("=" * 70)
    print(f"\nGATE VERDICT: {result['gate_verdict']}")
    if result["gate_verdict"] == "no_launch_redesign":
        print("  *** HARD NO-LAUNCH *** -- criterion (b) failed. Track B's Wave 1 must NOT be "
              "launched on the beta-gated (or naive-window) construction as currently specified. "
              "route to redesign per sec 4.2's registered outcome (iii).")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.out}")

    if result["gate_verdict"] == "no_launch_redesign":
        sys.exit(3)                    # shell-level fail-closed for &&-chained launchers


if __name__ == "__main__":
    main()
