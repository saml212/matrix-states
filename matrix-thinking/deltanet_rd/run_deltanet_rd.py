"""run_deltanet_rd.py -- DeltaNet-RD (real-data) training entry point. See
DELTANET_REALDATA_DESIGN.md (frozen, revision 2.1), especially section 5
(Wave 1: the adjacency-constrained grammar, C16/C17/C19), section 4.3/R2-3
(the custom production-kernel block), section 8 (operational-harness
requirements).

Trains ONE (K, force_rank_k, seed, steps) cell and writes a single result
JSON with Stage-0/run_deltanet.py-style instrumentation:
  trajectory   dense (<=200-step default) loss / skip-rate / grad-norm log
  checkpoints  <=2000-step default full eval on FOUR pools per hop-group:
               M2_in_distribution (train entities+template, H_train),
               M3_held_out (train entities+template, H_test+H_extra),
               C17_heldout_entities (disjoint name pool, H_train),
               C19_heldout_template (disjoint relation-verb pool, H_train)
               -- each entry carries recovery stats, whole/entity-subspace
               rank, AND the C16 premise diagnostics (key/value Gram
               deviation, salvage ratio, bind->query alignment cosine),
               computed on the held-out pools too (R2-8: classifies a C17/C19
               failure as premise vs. competence, not left ambiguous).

Both written INCREMENTALLY to --out after every checkpoint, "complete":
false on every incremental dump; ONLY the final post-training write in
main() carries "complete": true (run_deltanet_rd_sweep.py's is_done()
requires this sentinel -- crash-safety, mirrors run_deltanet.py exactly).

--save-z S_T dumps are ON BY DEFAULT (use --no-save-z to disable) -- the
project's own "eval-truncation lesson" (dumps must exist, not be
reconstructable only from a re-run) applies here from the start rather than
being discovered the hard way a second time.

Run the smoke gate FIRST (seconds-to-low-minutes; REQUIRES CUDA + network for
the GPT-2 tokenizer download/cache on first run): python run_deltanet_rd.py --smoke
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
import embed_arms
import grammar_rd as grd
import key_anchoring as ka
import model_rd as mrd
from grammar_rd import DeltaNetRDTaskConfig, EntityPools, sample_batch_rd, self_query_tokens
from model_rd import (DeltaNetRDBlock, TruncationError, assert_rank_le, entity_subspace_rank,
                       gram_deviation, orth_penalty, pin_buffer_row_, pin_rows_, salvage_ratio,
                       target_clause_index, zero_buffer_grad_, zero_rows_grad_)
from rank_utils import effective_rank, stable_rank

TAUS = (0.9, 0.95, 0.99)
# C16's pre-registered numeric rules (DELTANET_REALDATA_DESIGN.md section
# 5.2, R2-1/R2-2, plus the 2026-07-02 audit-round-1 MAJOR-1 value-side
# addendum below). The tau/threshold values are re-registerable ONLY at the
# Wave 0 -> Wave A gate (R2-4) -- any change is a documented deviation
# recorded in Wave 0's summary, never silent post-hoc tuning.
C16_TAU_UNCONSTRAINED = 0.03
C16_SALVAGE_RATIO = 0.1
C16_ALIGNMENT_COS = 0.9
# VALUE-side premise gate (audit round-1 MAJOR-1; dated spec addendum in
# DELTANET_REALDATA_DESIGN.md section 5.2): the frozen spec NAMED premise
# (ii) ("{v_eff_j} linearly independent ... checked with the same
# instrument applied to the value side") but its Rule subsection gated
# premise classification on the KEY side only. The audit constructed a
# collapsed-v_proj state where value_gram_deviation fired (5.78 -> 9.96)
# while both premise flags stayed green, and the FATAL-0 degenerate run was
# a live instance. Symmetric rule, same numeric anchors as the key side (no
# independent value-side anchor exists yet -- Wave 0 measures the real band
# and the R2-4 gate is where these get re-registered if miscalibrated):
# unconstrained-arm tau on the row-normalized value Gram, salvage tier
# sigma_K/sigma_1 on the row-normalized value matrix as the minimal
# linear-independence premise.
C16_TAU_VALUE_UNCONSTRAINED = 0.03
C16_VALUE_SALVAGE_RATIO = 0.1
# Section 14.4's pre-registered anti-collapse training objective
# (2026-07-03, at the R2-4 gate, mini-audit-verified with the target-index
# FATAL fix): L_train = L_cos + NCE_LAMBDA * L_nce, with L_nce the InfoNCE
# term over IN-EPISODE negatives (the same episode's K clauses' v_eff),
# temperature NCE_T, logits in fp32, NO stop-gradient on the target side.
# FIXED, pre-registered constants -- the lambda in {0.3, 3.0} sensitivity
# pair exists ONLY for a marginal section-14.6 gate outcome, never as a
# tuning loop. TRAINING loss only: eval scoring stays absolute cosine
# (section 14.3's Nichani discipline; the no-eval-leak smoke assert below
# enforces it).
NCE_LAMBDA = 1.0
NCE_T = 0.1


class AnchorEMA:
    """KEY_ANCHORING_DESIGN.md sec 2.3/sec 2.4 -- candidate (c)'s stop-
    gradient EMA anchor buffer (ALSO the substrate candidate (b), if ever
    triggered, would reuse -- sec 2.3's own "same anchor as (b)" note).
    Insertion site is run_deltanet_rd.py's loss composition ONLY (sec 2.4:
    "no change to model_rd.py's bind()/readout() at all") -- this class is
    deliberately NOT a model_rd.py nn.Module; it lives entirely in the
    training loop, mirroring ZCAWhiten's own no-grad EMA discipline
    (momentum 0.99) but keyed PER-ENTITY (vocab-indexed) rather than a
    single population statistic, since different entities are updated at
    different frequencies across episodes.

    NO BIAS CORRECTION -- a deliberate, documented divergence from
    ZCAWhiten's bias-corrected convention (2026-07-04 audit fix, MAJOR):
    update() below uses a SET-ON-FIRST-OBSERVATION convention (a row's
    first observed value is copied in directly, never blended against the
    zero init), so there is no zero-init shrinkage for a bias correction
    to undo. Applying the standard `x / (1 - momentum^n)` correction on
    top of this convention would be numerically WRONG, not just redundant:
    at n=1 it divides an already-unbiased value by (1 - 0.99) = 0.01, a
    ~100x blow-up. The per-row `counts` buffer exists ONLY to drive the
    set-vs-blend branch in update() (and as a logged diagnostic) -- do NOT
    reintroduce a bias_corrected() accessor when building candidate (b)
    on this substrate."""

    def __init__(self, vocab_size_total: int, d_state: int, device, momentum: float = 0.99):
        self.momentum = momentum
        self.table = torch.zeros(vocab_size_total, d_state, device=device)
        self.counts = torch.zeros(vocab_size_total, device=device)

    @torch.no_grad()
    def update(self, key_ids: torch.Tensor, k_eff_items: torch.Tensor) -> None:
        """key_ids: (B,K) int64. k_eff_items: (B,K,d) -- the POST-NS
        orthogonalized keys geo3 already produced this step (sec 2.3:
        "after geo3's own NS pass produces k_eff_items for this episode").
        stop-gradient by construction (torch.no_grad(), .detach())."""
        flat_ids = key_ids.reshape(-1)
        flat_vals = k_eff_items.reshape(-1, k_eff_items.shape[-1]).detach()
        for eid in torch.unique(flat_ids).tolist():
            rows = flat_vals[flat_ids == eid]
            new_val = rows.mean(dim=0)
            old_val = self.table[eid]
            n = self.counts[eid].item()
            blended = old_val * self.momentum + new_val * (1.0 - self.momentum) if n > 0 else new_val
            self.table[eid] = blended
            self.counts[eid] = n + 1

    def loss_anchor(self, key_ids: torch.Tensor, k_eff_items: torch.Tensor) -> torch.Tensor:
        """sec 2.4's L_anchor = sum_j (1 - cos(k_eff_items[j], stopgrad(A[key_ids[j]])));
        mean over the batch (matching cosine_loss's own per-row-then-mean
        convention). Rows with zero prior updates (a cold anchor) contribute
        a well-defined but uninformative term (cos against an all-zero
        row -> 0, i.e. maximal loss) -- acceptable since the EMA warms up
        within the first few hundred steps for any entity that recurs at
        all (K draws per episode over a 107-name pool)."""
        B, K, d = k_eff_items.shape
        anchor_rows = self.table[key_ids].detach()             # (B,K,d), stop-gradient by construction
        cos = F.cosine_similarity(k_eff_items, anchor_rows, dim=-1)
        return (1.0 - cos).mean()


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


def nce_loss(pred: torch.Tensor, v_eff_items: torch.Tensor,
              tgt_clause: torch.Tensor) -> torch.Tensor:
    """Section 14.4's L_nce, exactly as pre-registered: for query j,
    logits_{j,i} = cos(pred_j, v_eff_i) / T over the SAME episode's K
    clauses (in-episode negatives only), true class = tgt(j) (the answer
    entity's clause -- target_clause_index, sharing the mini-audit's
    FATAL-fixed site with forward()'s cosine targets), cross-entropy,
    logits in fp32, NO stop-gradient on either side. Under exact value
    collapse all K candidates are identical, logits are uniform, and
    L_nce = log K -- see section 14.4's audit-corrected characterization
    (a non-attracting stationary point; escape is the coupled path via
    L_cos along binding, conditional on the target-index fix).
    pred: (B,Q,d); v_eff_items: (B,K,d); tgt_clause: (B,Q) int64."""
    pred_n = F.normalize(pred.float(), dim=-1)
    v_n = F.normalize(v_eff_items.float(), dim=-1)
    logits = torch.einsum("bqd,bkd->bqk", pred_n, v_n) / NCE_T     # fp32 logits
    K = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, K), tgt_clause.reshape(-1))


def nce_loss_fixed_m(pred: torch.Tensor, v_eff_items: torch.Tensor, tgt_clause: torch.Tensor,
                       m: int, gen: torch.Generator) -> torch.Tensor:
    """F-nce (DELTANET_RD_EXACTNESS_DESIGN.md sec 5.5): the crowding-
    normalized contrastive term -- negatives subsampled to a FIXED count
    m per query REGARDLESS of K (uniform over the K-1 in-episode
    non-targets), temperature NCE_T unchanged, chance floor log(m+1) for
    EVERY K (K-invariant discrimination difficulty by construction --
    the structural fix, house preference). Training-only (sec 14.3's
    standing rule -- nothing here touches eval scoring; this function is
    never imported by evaluate_pool).

    Sampling is driven by the SAME training generator `gen` sample_batch_rd
    already consumes from (deterministic given --seed, no separate RNG
    stream to desync from a resumed/reproduced run).

    pred: (B,Q,d); v_eff_items: (B,K,d); tgt_clause: (B,Q) int64."""
    B, Q, d = pred.shape
    K = v_eff_items.shape[1]
    assert m <= K - 1, f"F-nce m={m} must be <= K-1={K - 1} (not enough in-episode negatives)"
    pred_n = F.normalize(pred.float(), dim=-1)
    v_n = F.normalize(v_eff_items.float(), dim=-1)
    # per (b,q): draw m negatives from {0..K-1} \ {tgt_clause[b,q]} WITHOUT
    # replacement, via an argsort-of-random-scores trick (same
    # without-replacement pattern grammar_rd.py's own entity draw uses) --
    # the target's score is forced to +inf so it can never be drawn as a
    # negative, then the smallest m scores among the remaining K-1 win.
    rand_scores = torch.rand(B, Q, K, generator=gen, device=pred.device)
    is_tgt = torch.zeros(B, Q, K, dtype=torch.bool, device=pred.device)
    is_tgt.scatter_(-1, tgt_clause.unsqueeze(-1), True)
    rand_scores = rand_scores.masked_fill(is_tgt, float("inf"))
    neg_idx = rand_scores.argsort(dim=-1)[..., :m]                          # (B,Q,m)
    cand_idx = torch.cat([tgt_clause.unsqueeze(-1), neg_idx], dim=-1)        # (B,Q,m+1), col 0 = target
    v_cand = torch.gather(v_n.unsqueeze(1).expand(B, Q, K, d), 2,
                           cand_idx.unsqueeze(-1).expand(B, Q, m + 1, d))     # (B,Q,m+1,d)
    logits = torch.einsum("bqd,bqcd->bqc", pred_n, v_cand) / NCE_T           # (B,Q,m+1), fp32
    labels = torch.zeros(B, Q, dtype=torch.int64, device=pred.device)        # target always at col 0
    return F.cross_entropy(logits.reshape(-1, m + 1), labels.reshape(-1))


def _recovery_stats(cos: torch.Tensor, prefix: str = "") -> dict:
    d = {
        f"{prefix}mean_cos": cos.mean().item(),
        f"{prefix}cos_p10": torch.quantile(cos, 0.10).item(),
        f"{prefix}cos_p50": torch.quantile(cos, 0.50).item(),
        f"{prefix}cos_p90": torch.quantile(cos, 0.90).item(),
    }
    for tau in TAUS:
        d[f"{prefix}recovered_frac@{tau}"] = (cos > tau).float().mean().item()
    return d


@torch.no_grad()
def evaluate_pool(model: DeltaNetRDBlock, cfg: DeltaNetRDTaskConfig, gen, device,
                   hops_to_eval, pools: EntityPools, force_rank_k=None,
                   n_batches=4, batch_size=128,
                   use_heldout_entities=False, use_heldout_template=False) -> dict:
    """Per-hop evaluation on ONE pool combination (train/C17/C19 x H_train/
    H_test+H_extra -- run_deltanet_rd.py's train() calls this 4x per
    checkpoint, section 5.3/5.4). Every entry carries C16's premise
    diagnostics computed on THIS pool (R2-8).

    F-geo-3 (DELTANET_RD_EXACTNESS_DESIGN.md sec 14.9 item 1, load-bearing
    correctness fix -- NOT cosmetic): the C16 self-query alignment
    diagnostic below computes q_self via self_query_tokens ->
    effective_key_window -- the RAW, pre-orthogonalization path -- for
    every existing arm. Left unpatched under geo3_active, q_self would
    differ from k_eff_items (orthogonalized) by exactly the
    orthogonalization correction, on EVERY batch, poisoning the finding-5
    alignment gate even when the fix works perfectly. Sec 14.2's patch:
    route geo3_active through q_self = k_eff_items directly (the slot-order
    identity self_query_tokens already guarantees -- verified explicitly,
    model_rd.py's own [model 16] item 4). Per sec 14.10, this makes the
    alignment instrument TAUTOLOGICALLY 1.0 under geo3_active -- logged,
    but EXCLUDED from premise-evidence (see the substitute admission stack
    in train()/_assemble_result)."""
    model.eval()
    per_hop = {}
    for h in hops_to_eval:
        cos_all, norm_all = [], []
        er_whole, sr_whole, er_ent, sr_ent = [], [], [], []
        key_gd, val_gd, salv, val_salv = [], [], [], []
        align_mean, align_min, align_valid = [], [], []
        n_eval_batches_skipped = 0
        fallback_this_hop = False    # sec 14.4/14.10 item 2: reset PER HOP, OR'd over its own batches only
        for _ in range(n_batches):
            b = sample_batch_rd(cfg, batch_size, gen, hop_set=(h,), pools=pools, device=device,
                                 use_heldout_entities=use_heldout_entities,
                                 use_heldout_template=use_heldout_template, assert_injective=True)
            try:
                pred, targets, S_T, k_eff_items, v_eff_items = model(b, force_rank_k=force_rank_k)
            except TruncationError:
                # deviation #6 robustness rule: a truncation non-convergence
                # skips THIS eval batch (counted below), never kills the run
                n_eval_batches_skipped += 1
                continue
            if model.geo3_active and model.geo3_last_fallback_triggered:
                fallback_this_hop = True
            cos_all.append(F.cosine_similarity(pred, targets, dim=-1).reshape(-1))
            norm_all.append(pred.norm(dim=-1).reshape(-1))

            er_whole.append(effective_rank(S_T))
            sr_whole.append(stable_rank(S_T))
            s_ideal = model.effective_ideal_S(k_eff_items, v_eff_items)
            er_e, sr_e = entity_subspace_rank(S_T, s_ideal, cfg.K)
            er_ent.append(er_e)
            sr_ent.append(sr_e)

            key_gd.append(gram_deviation(k_eff_items))
            v_norm_items = F.normalize(v_eff_items, dim=-1)
            val_gd.append(gram_deviation(v_norm_items))
            salv.append(salvage_ratio(k_eff_items))
            val_salv.append(salvage_ratio(v_norm_items))          # MAJOR-1: premise (ii) instrument

            if model.geo3_active:
                # sec 14.2/14.9 item 1's patch: q_self IS k_eff_items by construction (slot j <->
                # row j, self_query_tokens' own slot order) -- NEVER independently recomputed via
                # effective_key_window, which cannot see the per-episode joint orthogonalization.
                q_self = k_eff_items
            else:
                sq = self_query_tokens(cfg, pools, b["key_ids"], b["rel_id"])
                Bq, Kq, Lq = sq.shape
                q_self = model.effective_key_window(sq.reshape(Bq * Kq, Lq)).reshape(Bq, Kq, -1)
            cos_align = F.cosine_similarity(k_eff_items, q_self, dim=-1)      # (B,K), premise (iii)
            align_mean.append(cos_align.mean(dim=-1))
            align_min.append(cos_align.min(dim=-1).values)
            align_valid.append((cos_align >= C16_ALIGNMENT_COS).float().mean(dim=-1))

        if not cos_all:
            # every batch's truncation failed -- record the failure
            # explicitly instead of crashing on an empty cat; downstream
            # premise classification reads missing diagnostics as invalid
            per_hop[h] = {"h": h, "effective_hop": h % cfg.K,
                          "eval_failed_all_batches": True,
                          "n_eval_batches_skipped": n_eval_batches_skipped}
            continue
        cos = torch.cat(cos_all)
        entry = {"h": h, "effective_hop": h % cfg.K,
                 "n_eval_batches_skipped": n_eval_batches_skipped}
        entry.update(_recovery_stats(cos))
        entry["pred_norm_mean"] = torch.cat(norm_all).mean().item()               # F3 diagnostic
        entry["effective_rank_whole_mean"] = torch.cat(er_whole).mean().item()
        entry["stable_rank_whole_mean"] = torch.cat(sr_whole).mean().item()
        entry["entity_subspace_effective_rank_mean"] = torch.cat(er_ent).mean().item()   # PRIMARY
        entry["entity_subspace_stable_rank_mean"] = torch.cat(sr_ent).mean().item()
        entry["key_gram_deviation_mean"] = torch.cat(key_gd).mean().item()        # C16 premise (i)
        entry["value_gram_deviation_mean"] = torch.cat(val_gd).mean().item()      # C16 premise (ii)
        entry["salvage_ratio_mean"] = torch.cat(salv).mean().item()               # R2-1(ii), key side
        entry["value_salvage_ratio_mean"] = torch.cat(val_salv).mean().item()     # MAJOR-1, value side
        entry["alignment_cos_mean"] = torch.cat(align_mean).mean().item()         # C16 premise (iii)
        entry["alignment_cos_min"] = torch.cat(align_min).min().item()
        entry["alignment_valid_frac"] = torch.cat(align_valid).mean().item()
        # Per-side premise flags. Classification per checkpoint is
        # arm-specific (R2-1) and applied downstream (the sweep's
        # aggregate() applies the correct arm's rule -- unconstrained vs
        # causal k>=K vs causal k<K, section 5.2 + the MAJOR-1/MAJOR-2
        # audit addenda); these booleans are the per-side building blocks
        # AND a convenience readout, recorded on BOTH the key and (MAJOR-1)
        # value sides symmetrically.
        entry["premise_valid_unconstrained_tau"] = entry["key_gram_deviation_mean"] < C16_TAU_UNCONSTRAINED
        entry["premise_valid_salvage_tier"] = entry["salvage_ratio_mean"] >= C16_SALVAGE_RATIO
        entry["premise_valid_value_unconstrained_tau"] = \
            entry["value_gram_deviation_mean"] < C16_TAU_VALUE_UNCONSTRAINED
        entry["premise_valid_value_salvage_tier"] = \
            entry["value_salvage_ratio_mean"] >= C16_VALUE_SALVAGE_RATIO
        # R2-2's frozen rule is PER-ITEM ("cos(k_eff_bind_j, q_eff_query_j)
        # >= 0.9 for ALL K items") -- the gate is the MIN over every item
        # in the eval batches, never the mean (round-2 audit
        # pre-registration-fidelity fix); alignment_cos_mean stays logged
        # as a diagnostic only.
        entry["alignment_valid"] = entry["alignment_cos_min"] >= C16_ALIGNMENT_COS
        if model.geo3_active:
            # sec 14.10 items 1/3: the key-Gram and alignment diagnostics are BOTH
            # architecturally forced (~0 / ~1.0) under geo3_active -- logged, but
            # EXPLICITLY excluded from premise-evidence (never used to gate anything).
            entry["geo3_diagnostics_are_non_evidence"] = True
            entry["geo3_fallback_triggered_this_hop"] = fallback_this_hop
        per_hop[h] = entry
    model.train()
    return per_hop


def _dump(path, obj):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"  (incremental write failed, non-fatal: {e!r})", flush=True)


def _assemble_result(cfg, steps, seed, force_rank_k, trunc_impl, pool_report, d_model, d_state,
                      trajectory, checkpoints, n_skipped, t0, timed_out, steps_completed, complete,
                      exactness_config=None, geo3_admission=None, anchor_lambda_summary=None,
                      anchor_lambda_e_summary=None, ckpt_written=None, unblind_override_stamp=None):
    loss_config = {"objective": "L_cos + lambda*L_nce (section 14.4)",
                   "nce_lambda": NCE_LAMBDA, "nce_temperature": NCE_T,
                   "negatives": "in_episode_K_clauses", "stop_grad": False,
                   "logits_dtype": "fp32", "target_index": "pi_inverse_of_tgt_slot (FATAL-fixed)"}
    ec = exactness_config or {}
    if ec.get("lambda_orth", 0.0):
        loss_config["lambda_orth"] = ec["lambda_orth"]        # F-geo-1, sec 5.5
    if ec.get("fnce_m") is not None:
        loss_config["negatives"] = f"fixed_m={ec['fnce_m']}_uniform_over_K-1_nontargets"   # F-nce, sec 5.5
        loss_config["fnce_m"] = ec["fnce_m"]
    if ec.get("lambda_anchor", 0.0):
        loss_config["lambda_anchor"] = ec["lambda_anchor"]    # KEY_ANCHORING sec 2.4, candidate (c)
    result = {
        "K": cfg.K, "conv_size": cfg.conv_size, "d_model": d_model, "d_state": d_state,
        "H_train": list(cfg.H_train), "H_test": list(cfg.H_test), "H_extra": list(cfg.H_extra),
        "force_rank_k": force_rank_k, "trunc_impl": trunc_impl, "seed": seed, "steps": steps,
        "steps_completed": steps_completed,
        "complete": complete,                       # crash-safety sentinel
        "n_skipped_steps": n_skipped,
        "skip_rate": (n_skipped / steps_completed) if steps_completed > 0 else 0.0,
        "pool_report": pool_report,                  # tokenizer/pool build-time verification (section 5.2)
        "c16_thresholds": {"tau_unconstrained": C16_TAU_UNCONSTRAINED,
                            "salvage_ratio": C16_SALVAGE_RATIO, "alignment_cos": C16_ALIGNMENT_COS,
                            # MAJOR-1 (audit round 1, 2026-07-02): value-side premise gate
                            "tau_value_unconstrained": C16_TAU_VALUE_UNCONSTRAINED,
                            "value_salvage_ratio": C16_VALUE_SALVAGE_RATIO},
        # section 14.4's pre-registered training objective (recorded per run
        # so a lambda/T deviation can never be silent) + the mini-audit
        # target-index FATAL fix marker (2026-07-03); DELTANET_RD_EXACTNESS_
        # DESIGN.md's F-geo-1/F-nce additions (if active) fold in above.
        "loss_config": loss_config,
        # DELTANET_RD_EXACTNESS_DESIGN.md sec 4/5.5's arm/instrument identity
        # -- ALWAYS recorded (even at every-default "off"), so
        # run_deltanet_rd_sweep.py's is_done() can cross-check it without
        # guessing at a missing-field default (same discipline as the
        # existing trunc_impl identity check).
        "exactness_config": {
            "embed_source": ec.get("embed_source", "learned"),
            "gram_alpha": ec.get("gram_alpha"), "gram_rho": ec.get("gram_rho"),
            "strong_pin": bool(ec.get("strong_pin", False)),
            "lambda_orth": ec.get("lambda_orth", 0.0),
            "use_zca": bool(ec.get("use_zca", False)),
            "fnce_m": ec.get("fnce_m"),
            # F-geo-3, sec 14: ALWAYS recorded (default off), same discipline as every
            # other arm-identity field above.
            "geo3_active": bool(ec.get("geo3_active", False)),
            "geo3_n_iter": ec.get("geo3_n_iter"),
            "geo3_resid_tol": ec.get("geo3_resid_tol"),
            # KEY_ANCHORING_DESIGN.md sec 2.2/2.4: ALWAYS recorded (default off), same
            # always-recorded-even-at-off discipline as every field above.
            "anchor_active": bool(ec.get("anchor_active", False)),
            "anchor_lambda_mode": ec.get("anchor_lambda_mode"),
            "anchor_lambda_fixed": ec.get("anchor_lambda_fixed"),
            "lambda_anchor": ec.get("lambda_anchor", 0.0),
            # KEY_ANCHORING_DESIGN.md sec 3.6 (2026-07-06 keyanchor-confirm
            # build): ALWAYS recorded (default off), same always-recorded-
            # even-at-off discipline as every field above. Closes a real
            # gap: _spec()'s own docstring (run_deltanet_rd_exactness_
            # sweep.py) already claimed drift_probe is "re-derived from the
            # result JSON's own exactness_config" for is_done()'s identity
            # check, but this field was never actually written here --
            # is_done() relied on filename encoding alone (the `_dprobe`
            # name bit) for drift_probe-driven distinctness.
            "drift_probe": bool(ec.get("drift_probe", False)),
            # KEY_ANCHORING_DESIGN.md sec 10 (Rev 7.1, 2026-07-06 keyanchor-
            # mech build): ALWAYS recorded (default off), same discipline.
            "rev7_engagement": bool(ec.get("rev7_engagement", False)),
            # KEY_ANCHORING_DESIGN.md sec 10.13 (candidate (e), 2026-07 K48+e
            # build): ALWAYS recorded (default off/frame_potential), same
            # always-recorded-even-at-off discipline as every field above --
            # is_done() cross-checks both so a candidate-(e) cell can never
            # silently resume-match a candidate-(d) cell that happens to
            # share every OTHER field.
            "anchor_table_frozen": bool(ec.get("anchor_table_frozen", False)),
            "anchor_table_init_mode": ec.get("anchor_table_init_mode", "frame_potential"),
            # KEY_ANCHORING_DESIGN.md sec 14.1b item 5 (Rev 14.3, coherence
            # dose-response wave): ALWAYS recorded (default None), same
            # always-recorded-even-at-off discipline as every field above --
            # run_deltanet_rd_exactness_sweep.py's is_done() cross-checks
            # dose_target/dose_structure/subspace_seed against these exact
            # keys (never the top level, unlike d_state -- see is_done()'s
            # own comment on the two fields' different write sites).
            "dose_target": ec.get("dose_target"),
            "dose_structure": ec.get("dose_structure"),
            "subspace_seed": ec.get("subspace_seed"),
            "achieved_max_cos": ec.get("achieved_max_cos"),
        },
        "trajectory": trajectory, "checkpoints": checkpoints,
        # KEY_ANCHORING_DESIGN.md sec 3.6 item (c) (2026-07-04 audit fix,
        # MAJOR): the absolute epoch timestamp at run start -- the field
        # key_anchoring.assert_blind_not_broken / readout_keyanchor.py
        # consume to assert the BANDS_PINNED pin-timestamp strictly
        # precedes every anchor-arm start. Written for EVERY run (not only
        # anchor arms), same always-recorded discipline as wall_s.
        "started_at": t0,
        "wall_s": time.time() - t0, "timed_out": timed_out,
    }
    if checkpoints:
        result["final_step"] = checkpoints[-1]["step"]
    if geo3_admission is not None:
        # DELTANET_RD_EXACTNESS_DESIGN.md sec 14.10's SUBSTITUTE per-seed admission
        # stack (the standard finding-5 alignment/key-Gram gate is tautological under
        # geo3_active -- sec 14.9 item 5/14.10). LOGGED FIELDS, never asserts -- the
        # sweep/analysis layer applies these as a read, not this script.
        result["geo3_admission"] = geo3_admission
    if anchor_lambda_summary is not None:
        # KEY_ANCHORING_DESIGN.md sec 3.2/sec 2.2's REQUIRED per-seed summary
        # fields (final value / trailing-window mean / trailing-window range
        # / band), computed from the SAME lambda values already logged into
        # `trajectory` above -- a machine-readable field, not eyeballing.
        result["anchor_lambda_summary"] = anchor_lambda_summary
    if anchor_lambda_e_summary is not None:
        # KEY_ANCHORING_DESIGN.md sec 10.5.1, candidate (d'): the per-entity
        # interior-band-fraction summary (extends anchor_lambda_summary's
        # own discipline to a 107-entity table instead of one scalar).
        result["anchor_lambda_e_summary"] = anchor_lambda_e_summary
    if ckpt_written:
        # KEY_ANCHORING_DESIGN.md sec 10.10 item 1: the list of checkpoint
        # files this run actually wrote -- the readout-time gate (sec 10.10
        # item 2) reads THIS field, never re-globs the checkpoint directory,
        # so a partial/crashed run's incomplete checkpoint set is visible
        # from the result JSON alone.
        result["ckpt_written"] = ckpt_written
    # KEY_ANCHORING_DESIGN.md sec 3.6 Rev 5 (R4-1 fix): unblind_override is
    # ALWAYS written (True or False) so its absence can never be read as
    # evidence of a clean blind; claim_tier is written ONLY on the override
    # path (the one tier verdict knowable at launch time regardless of
    # anything the run later earns -- non-override tiers stay readout-time
    # verdicts per the admission stack). Mirrors lm_pretrain_rd.py's own
    # claim_tier / _assemble_result precedent (L1090, smoke-asserted L1420).
    result.update(ka.assemble_claim_tier_fields(unblind_override_stamp))
    return result


def _geo3_checkpoint_fallback_seen(checkpoints: list) -> bool:
    """Scans every checkpoint's 4 pools' per-hop entries for the fallback
    flag evaluate_pool() logs (sec 14.4/14.10 item 2, EVAL side)."""
    for ckpt in checkpoints:
        for pool_name in ("M2_in_distribution", "M3_held_out", "C17_heldout_entities", "C19_heldout_template"):
            for entry in ckpt.get(pool_name, {}).values():
                if entry.get("geo3_fallback_triggered_this_hop"):
                    return True
    return False


def compute_geo3_admission(cfg, trajectory: list, checkpoints: list, n_geo3_fallback_train_steps: int) -> dict:
    """DELTANET_RD_EXACTNESS_DESIGN.md sec 14.10 -- the SUBSTITUTE per-seed
    admission criterion for geo3_active runs (the standard finding-5
    alignment/key-Gram gate is tautological under this arm -- sec 14.9 item
    5). Returns LOGGED FIELDS, never asserts here -- the sweep/analysis
    layer reads "admissible" as a filter, this function only computes it.

    1. Value-side salvage tier: sigma_K/sigma_1 >= 0.1 on v_eff_items at the
       FINAL checkpoint's h=H_train[0] M2 (in-distribution) entry -- the one
       base-stack instrument F-geo-3 leaves fully independent (v_conv is
       untouched by the orthogonalization).
    2. Newton-Schulz converged WITHOUT the eigh fallback at EVERY logged
       checkpoint AND every training step (both sides tracked: training via
       n_geo3_fallback_train_steps, eval via the per-checkpoint flag
       evaluate_pool() logs).
    3. Finite loss throughout, no divergence.
    4. Task-performance floor: final train loss improves >=50% over the
       step-0 (pre-first-update) loss, AND h=1 in-distribution rec@0.9 >=
       0.5 at the final checkpoint -- an ADMISSIBILITY FLOOR, never a
       success bar (the learned baseline's own h=1 is 0.78 at K=32).
    """
    h1 = cfg.H_train[0]
    step0_loss = trajectory[0]["loss"] if trajectory else None
    final_loss = trajectory[-1]["loss"] if trajectory else None
    pct_improvement = None
    if step0_loss is not None and final_loss is not None and step0_loss != 0:
        pct_improvement = (step0_loss - final_loss) / abs(step0_loss)

    final_ckpt = checkpoints[-1] if checkpoints else None
    h1_entry = (final_ckpt or {}).get("M2_in_distribution", {}).get(h1, {})
    h1_rec_at_09 = h1_entry.get("recovered_frac@0.9")
    value_salvage_final = h1_entry.get("value_salvage_ratio_mean")

    finite_loss_no_divergence = bool(trajectory) and all(
        t["loss"] == t["loss"] and abs(t["loss"]) < 1e6 for t in trajectory)   # != NaN, not exploded
    ckpt_fallback_seen = _geo3_checkpoint_fallback_seen(checkpoints)
    ns_converged_no_fallback = (n_geo3_fallback_train_steps == 0) and not ckpt_fallback_seen
    value_salvage_tier_pass = value_salvage_final is not None and value_salvage_final >= C16_VALUE_SALVAGE_RATIO
    task_performance_floor_pass = (
        pct_improvement is not None and pct_improvement >= 0.5
        and h1_rec_at_09 is not None and h1_rec_at_09 >= 0.5)

    admissible = bool(ns_converged_no_fallback and value_salvage_tier_pass
                       and finite_loss_no_divergence and task_performance_floor_pass)
    return {
        "value_salvage_tier_pass": value_salvage_tier_pass,
        "value_salvage_ratio_final": value_salvage_final,
        "ns_converged_no_fallback": ns_converged_no_fallback,
        "n_geo3_fallback_train_steps": n_geo3_fallback_train_steps,
        "checkpoint_fallback_seen": ckpt_fallback_seen,
        "finite_loss_no_divergence": finite_loss_no_divergence,
        "task_performance_floor_pass": task_performance_floor_pass,
        "step0_loss": step0_loss, "final_loss": final_loss, "pct_improvement": pct_improvement,
        "h1_recovered_frac_at_0.9_final": h1_rec_at_09,
        "admissible": admissible,
        "note": "sec 14.10 substitute admission stack -- comparability to the standard finding-5 "
                "gate is UNVERIFIED (an empirical question with no data until Wave -1/Wave 1 "
                "pass rates exist); NEVER present a geo3 admissible-seed count as interchangeable "
                "with a learned-arm finding-5-clean-seed count without stating both gates' realized "
                "pass rates (sec 14.10's comparability paragraph).",
    }


def train(model, cfg, pools, pool_report, device, d_model, d_state, steps=6000, batch_size=128,
          lr=3e-4, seed=0, force_rank_k=None, log_every=200, ckpt_every=2000, out_path=None,
          timeout_s=None, lambda_orth=0.0, fnce_m=None, exactness_config=None,
          lambda_anchor: float = 0.0, unblind_override_at: float | None = None,
          drift_probe_active: bool = False, drift_probe_n_entities: int = 8,
          drift_probe_n_resamples: int = 32, rev7_pin_derived: dict | None = None,
          ckpt_dir: str | None = None):
    """lambda_anchor (KEY_ANCHORING_DESIGN.md sec 2.4, candidate (c)): soft
    cross-episode drift regularizer weight, > 0 only for candidate (c)
    cells (requires model.geo3_active; model.anchor_active stays False --
    "no change to model_rd.py's bind()/readout() at all", sec 2.4). Model
    behavior at inference is BYTE-IDENTICAL to bare geo3; only this
    training-time loss term differs.

    drift_probe_active (sec 3.6): the reference arms' own per-checkpoint
    drift instrumentation ("identical config to the archived geo3 cells
    PLUS the per-checkpoint drift diagnostic active") -- both post-NS and
    pre-NS pooled drift, logged into each checkpoint's own `drift_probe`
    field (`key_anchoring.reference_arm_result_valid`'s own expected
    shape). Uses `key_anchoring.measure_drift` (fla-free, imported above as
    `ka`) -- no circular import (this module is never imported BACK by
    key_anchoring.py).

    rev7_pin_derived (KEY_ANCHORING_DESIGN.md sec 10, Rev 7.1): the
    ALREADY-VALIDATED `derived` block of REV7_THRESHOLD_PINNED.json (the
    caller -- main(), via `--rev7-engagement` -- validates the pin ONCE
    with `ka.validate_rev7_pin()` and threads the block down here; train()
    itself never re-validates or re-derives). When given (and
    drift_probe_active and model.anchor_active), the FINAL checkpoint
    additionally computes sec 10.2/10.3's r_e + null-pool engagement
    measurement (`ka.measure_r_e_and_null_pool`) -- independent of, and in
    ADDITION to, the existing sec 3.7 `per_entity_alignment` (a_e) below
    (a_e is demoted to secondary/diagnostic per sec 10.2; r_e is the new
    registered driver of `engaged_frac_v3`). None (default) reproduces the
    pre-Rev-7.1 behavior exactly -- byte-identical for every existing wave.

    ckpt_dir (sec 10.10's checkpoint writer): when given (and
    model.anchor_active), the anchor table's TRAINED-ROW block (+ the
    per-entity lambda table's trained-row block for candidate (d'), or the
    scalar anchor_lambda_raw for candidate (d)) is torch.save'd to
    `<ckpt_dir>/step<N>.pt` at every eval checkpoint AND the final step --
    a mechanical, unconditional writer (sec 10.10 item 1), never a stated
    intention. None (default) = no checkpoint writing, byte-identical to
    every existing wave."""
    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trunc_impl = getattr(model, "trunc_impl", "eigh")
    model.train()

    if model.anchor_active and model.anchor_lambda_mode in ("learned", "learned_per_entity"):
        # sec 3.2/R3 finding 1's startup assertion: fires loudly on a
        # mis-set cadence rather than silently resizing the lambda-band
        # trailing window. Applies to candidate (d') too (sec 10.5.1) --
        # the per-entity trajectory uses the SAME registered cadence.
        ka.assert_lambda_log_cadence(log_every)

    if rev7_pin_derived is not None:
        assert model.anchor_active and drift_probe_active, \
            "rev7_pin_derived requires anchor_active=True and drift_probe_active=True (sec 10.2/10.3 " \
            "engagement measurement is defined only for anchor-arm cells with the drift probe active)"

    ckpt_written = []
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)

    if lambda_anchor:
        assert model.geo3_active, \
            "lambda_anchor > 0 (candidate (c)) requires model.geo3_active=True -- the regularizer " \
            "targets the SAME k_eff_items geo3's own NS pass produces (sec 2.4)"
        assert not model.anchor_active, \
            "lambda_anchor > 0 (candidate (c)) and anchor_active (candidate (d)) are mutually " \
            "exclusive interventions -- run them as separate cells, never combined (sec 2.4 vs 2.2)"
        anchor_ema = AnchorEMA(model.vocab_size_total, model.d_state, device)
    else:
        anchor_ema = None

    trajectory, checkpoints = [], []
    # sec 10.5.1, candidate (d'): the per-entity lambda_e trajectory, kept
    # SEPARATE from the scalar `trajectory` list above (candidate (d)'s
    # schema stays byte-identical) -- one {"step":.., "lambda_e": {eid: v}}
    # dict per LOGGED cadence point, same LAMBDA_LOG_CADENCE_STEPS cadence.
    lambda_e_trajectory = []
    n_skipped = 0
    timed_out = False
    steps_completed = 0
    n_geo3_fallback_train_steps = 0   # sec 14.4/14.10 item 2, TRAINING side (eval side: per-checkpoint)

    # MAJOR-3 (audit round 1, 2026-07-02): FIXED-REFERENCE entity-subspace
    # rank. The primary metric's reference subspace (s_ideal built from the
    # model's CURRENT k/v_eff) moves with training -- under an intervention
    # (force-rank) the reference itself shifts, entangling "the state's
    # rank changed" with "the measuring frame changed". ADDITIONAL logged
    # metric (never replacing the primary): a fixed probe batch, with the
    # s_ideal reference SNAPSHOTTED at the first checkpoint (first-post-
    # warmup, per the audit's prescription) and reused unchanged at every
    # later checkpoint. Dated caveat recorded in
    # DELTANET_REALDATA_DESIGN.md section 12.
    fixed_gen = torch.Generator(device=device).manual_seed(seed + 30_000)
    b_fixed = sample_batch_rd(cfg, min(32, batch_size), fixed_gen, hop_set=(cfg.H_train[0],),
                               pools=pools, device=device)
    s_ideal_fixedref = None
    fixedref_snapshot_step = None

    for step in range(1, steps + 1):
        steps_completed = step
        b = sample_batch_rd(cfg, batch_size, gen, hop_set=cfg.H_train, pools=pools, device=device,
                             assert_injective=(step == 1))
        # A TruncationError (truncation non-convergence surviving bind()'s
        # one-shot jitter retry -- Wave -1's eigh forward-crash mode) skips
        # THIS step with the same n_skipped accounting as a non-finite
        # gradient; it must never kill the run (deviation #6 / the standing
        # harness-robustness rule). Checkpoint cadence is preserved (the
        # ckpt block below still runs on a skipped step).
        step_failed = False
        try:
            pred, targets, S_T, k_eff_items, v_eff_items = model(b, force_rank_k=force_rank_k)
            if model.geo3_active and model.geo3_last_fallback_triggered:
                n_geo3_fallback_train_steps += 1
            # Section 14.4's pre-registered objective: L_cos + lambda*L_nce
            # (TRAINING only -- eval scoring stays absolute cosine, section
            # 14.3). tgt_clause comes from the SAME FATAL-fixed site as
            # forward()'s targets (target_clause_index).
            l_cos = cosine_loss(pred, targets)
            tgt_clause = target_clause_index(b["succ"], b["tgt_slot"])
            # F-nce (sec 5.5): fixed-m crowding-normalized negatives when
            # requested, else the UNCHANGED full-K-way L_nce -- selecting
            # between them is the ONLY place this design's F-nce flag
            # touches train(); fnce_m=None (default) reproduces the
            # pre-extension code path exactly.
            if fnce_m is not None:
                l_nce = nce_loss_fixed_m(pred, v_eff_items, tgt_clause, fnce_m, gen)
            else:
                l_nce = nce_loss(pred, v_eff_items, tgt_clause)
            loss = l_cos + NCE_LAMBDA * l_nce
            l_orth = None
            if lambda_orth:
                # F-geo-1 (sec 5.5): per-episode orthogonality penalty on
                # k_eff_items, the EXACT tensor C16's gram_deviation already
                # gathers. TRAINING-loss-only (never touches eval scoring).
                l_orth = orth_penalty(k_eff_items).mean()
                loss = loss + lambda_orth * l_orth
            l_anchor = None
            if lambda_anchor and anchor_ema is not None:
                # KEY_ANCHORING_DESIGN.md sec 2.4, candidate (c): soft
                # cross-episode drift regularizer against the STOP-GRADIENT
                # EMA anchor's CURRENT (pre-this-step) value -- update the
                # EMA only AFTER the loss term reads it, so the regularizer
                # targets a settled population value, not this step's own
                # just-computed output.
                l_anchor = anchor_ema.loss_anchor(b["key_ids"], k_eff_items)
                loss = loss + lambda_anchor * l_anchor
                anchor_ema.update(b["key_ids"], k_eff_items)
            opt.zero_grad()
            loss.backward()
        except TruncationError:
            n_skipped += 1
            step_failed = True
        if not step_failed:
            finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            if finite:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                zero_buffer_grad_(model.embed, model.buffer_id)   # R2-3: buffer row excluded from the optimizer
                if model.frozen_row_ids is not None:
                    # arms (i)/(ii)/(iv), sec 4.2/4.3/4.5: frozen identity
                    # rows excluded from the optimizer, same discipline as
                    # the buffer row (generalized pin_rows_/zero_rows_grad_,
                    # model_rd.py).
                    zero_rows_grad_(model.embed, model.frozen_row_ids)
                opt.step()
                pin_buffer_row_(model.embed, model.buffer_id)      # re-assert exact zero every step
                if model.frozen_row_ids is not None:
                    pin_rows_(model.embed, model.frozen_row_ids, model.frozen_row_values)
            else:
                n_skipped += 1

        if not step_failed and (step % log_every == 0 or step == 1):
            with torch.no_grad():
                er = effective_rank(S_T).mean().item()
            traj_entry = {"step": step, "loss": loss.item(), "loss_cos": l_cos.item(),
                          "loss_nce": l_nce.item(), "eff_rank_trainbatch": er,
                          "skip_rate_so_far": n_skipped / step}
            if l_orth is not None:
                traj_entry["loss_orth"] = l_orth.item()
            if l_anchor is not None:
                traj_entry["loss_anchor"] = l_anchor.item()
            if model.anchor_active and model.anchor_lambda_mode == "learned":
                # KEY_ANCHORING_DESIGN.md sec 2.2's REQUIRED logged field --
                # the full trajectory, at the registered LAMBDA_LOG_CADENCE_STEPS
                # cadence (asserted at this function's own startup, above).
                traj_entry["lambda"] = float(model.anchor_lambda().item())
            elif model.anchor_active and model.anchor_lambda_mode == "learned_per_entity":
                # sec 10.5.1, candidate (d'): the per-entity analogue --
                # logged into the SEPARATE lambda_e_trajectory list (not
                # traj_entry/trajectory) since it is a 107-entity dict, not
                # a scalar. Same cadence, same startup assertion.
                with torch.no_grad():
                    lam_e_now = torch.sigmoid(
                        model.anchor_lambda_table.weight[model.anchor_train_ids_buf].squeeze(-1))
                lambda_e_trajectory.append({
                    "step": step,
                    "lambda_e": {eid: v for eid, v in
                                 zip(model.anchor_train_ids_buf.tolist(), lam_e_now.tolist())}})
            trajectory.append(traj_entry)
            extra = f"  [skip_rate {n_skipped/step:.4%}]" if n_skipped else ""
            orth_str = f" orth {l_orth.item():.4f}" if l_orth is not None else ""
            anchor_str = f" anchor {l_anchor.item():.4f}" if l_anchor is not None else ""
            lambda_str = f" lambda {traj_entry['lambda']:.4f}" if "lambda" in traj_entry else ""
            print(f"  step {step:6d}  loss {loss.item():.4f} (cos {l_cos.item():.4f} "
                  f"nce {l_nce.item():.4f}{orth_str}{anchor_str}{lambda_str})  eff_rank {er:.3f}{extra}",
                  flush=True)

        if step % ckpt_every == 0 or step == steps:
            with torch.no_grad():
                buf_ok = torch.equal(model.embed.weight[model.buffer_id],
                                      torch.zeros(d_model, device=device))
            assert buf_ok, "R2-3 VIOLATED: buffer embedding row is not exactly zero post-optimizer-step"

            eval_gen = torch.Generator(device=device).manual_seed(seed + 10_000 + step)
            m2 = evaluate_pool(model, cfg, eval_gen, device, cfg.H_train, pools, force_rank_k=force_rank_k)
            m3 = evaluate_pool(model, cfg, eval_gen, device, (*cfg.H_test, *cfg.H_extra), pools,
                                force_rank_k=force_rank_k)
            c17 = evaluate_pool(model, cfg, eval_gen, device, cfg.H_train, pools,
                                 force_rank_k=force_rank_k, use_heldout_entities=True)
            c19 = evaluate_pool(model, cfg, eval_gen, device, cfg.H_train, pools,
                                 force_rank_k=force_rank_k, use_heldout_template=True)

            # MAJOR-3: fixed-reference entity-subspace rank on the fixed
            # probe batch; reference snapshotted at the FIRST checkpoint.
            model.eval()
            with torch.no_grad():
                S_fixed, k_eff_fx, v_eff_fx = model.bind(b_fixed, force_rank_k=force_rank_k)
                if s_ideal_fixedref is None:
                    s_ideal_fixedref = model.effective_ideal_S(k_eff_fx, v_eff_fx).detach()
                    fixedref_snapshot_step = step
                er_fx, sr_fx = entity_subspace_rank(S_fixed, s_ideal_fixedref, cfg.K)
            model.train()
            fixedref = {"snapshot_step": fixedref_snapshot_step,
                        "entity_subspace_effective_rank_mean": er_fx.mean().item(),
                        "entity_subspace_stable_rank_mean": sr_fx.mean().item()}

            res = {"step": step, "M2_in_distribution": m2, "M3_held_out": m3,
                   "C17_heldout_entities": c17, "C19_heldout_template": c19,
                   "fixedref_entity_subspace": fixedref}
            if drift_probe_active:
                # sec 3.6's reference-arm per-checkpoint drift instrumentation
                # ("identical config to the archived geo3 cells PLUS the
                # per-checkpoint drift diagnostic active"). pre_ns_attr picks
                # the arm-appropriate side channel: candidate (d)'s blended
                # pre-NS key, or bare geo3/(c)'s raw pre-NS key.
                pre_ns_attr = ("anchor_last_k_blend_raw" if model.anchor_active
                               else "geo3_last_k_eff_raw" if model.geo3_active else None)
                drift_gen = torch.Generator(device=device).manual_seed(seed + 50_000 + step)
                dp = ka.measure_drift(model, cfg, pools, drift_probe_n_entities,
                                        drift_probe_n_resamples, drift_gen, device,
                                        pre_ns_attr=pre_ns_attr)
                res["drift_probe"] = {"post_ns": dp["post_ns"], "pre_ns": dp.get("pre_ns")}
                if model.anchor_active:
                    # KEY_ANCHORING_DESIGN.md sec 3.1 item 6 (2026-07-06
                    # keyanchor-confirm build -- closes sec 9.3's documented
                    # gap: item 6 was previously computed ONLY at Gate 2's
                    # one-time INIT check, never re-run on the TRAINED table
                    # as sec 3.1/sec 4 require "at every admission
                    # checkpoint"). Cheap: pure SVD/cosine on the anchor
                    # table's (n_train, d_state) train-row block, NO forward
                    # passes -- computed at EVERY checkpoint, unlike sec
                    # 3.7's alignment sweep below.
                    res["item6_table_conditioning"] = ka.raw_table_conditioning(
                        model.anchor_table.weight[model.anchor_train_ids_buf].detach())
                    # sec 10.2.1: anchor-row norms, EVERY checkpoint (not
                    # final-only) -- free (the parameter is already in
                    # memory), and a norm-drift trajectory the project has
                    # never previously measured.
                    with torch.no_grad():
                        row_norms = model.anchor_table.weight[model.anchor_train_ids_buf].norm(dim=-1)
                    res["anchor_row_norms"] = {
                        "per_entity": {eid: v for eid, v in
                                       zip(model.anchor_train_ids_buf.tolist(), row_norms.tolist())},
                        "mean": row_norms.mean().item(), "min": row_norms.min().item(),
                        "max": row_norms.max().item()}
                    if step == steps:
                        # sec 3.7's per-entity anchor-INPUT-alignment sweep
                        # (engaged_frac) + the h=1 behavioral companion --
                        # the OTHER half of sec 9.3's gap (previously wired
                        # ONLY into keyanchor_drift_diagnostic.py's separate
                        # 5,000-step probe model, never the real trained
                        # arms). FULL 107-entity pool, ~13x more bind() calls
                        # than the 8-entity item-5 sweep above -- a
                        # DISCLOSED scoping decision (build-report scrutiny
                        # item) to run this ONLY at the FINAL checkpoint,
                        # not every one, keeping the keyanchor-confirm
                        # wave's cost near its priced ~0.28 GPU-h/cell;
                        # sec 3.7's own text says "the claim readout uses
                        # the final step" for a_e, so the headline-relevant
                        # value is unaffected by this scoping choice.
                        align_gen = torch.Generator(device=device).manual_seed(seed + 60_000 + step)
                        alignment = ka.measure_full_pool_alignment(
                            model, cfg, pools, drift_probe_n_resamples, align_gen, device)
                        res["per_entity_alignment"] = alignment
                        h1_gen = torch.Generator(device=device).manual_seed(seed + 70_000 + step)
                        res["per_entity_h1_companion"] = ka.measure_h1_behavioral_companion(
                            model, cfg, pools, h1_gen, device)
                        if rev7_pin_derived is not None:
                            # KEY_ANCHORING_DESIGN.md sec 10.2/10.3 (Rev 7.1):
                            # r_e measured DIRECTLY on the PRE-BLEND raw key
                            # (never backed out through the lambda-dependent
                            # blend, sec 10.2's own redesign) + the null-pool
                            # C matrix (same mean-of-cosines code path, M2's
                            # fix) + BH-FDR engagement, all constants loaded
                            # from the pin (never recomputed inline here).
                            r7_gen = torch.Generator(device=device).manual_seed(seed + 80_000 + step)
                            res["r_e_engagement"] = ka.measure_r_e_and_null_pool(
                                model, cfg, pools, drift_probe_n_resamples, r7_gen, device,
                                rev7_pin_derived)
                            if model.anchor_lambda_mode == "learned_per_entity" and lambda_e_trajectory:
                                # sec 10.5.1's Spearman + Hartigan checks --
                                # final lambda_e vs final r_e, both keyed by
                                # entity id (the SAME 107 trained ids on
                                # both sides).
                                final_lambda_e = lambda_e_trajectory[-1]["lambda_e"]
                                final_r_e = res["r_e_engagement"]["r_e"]
                                res["lambda_e_vs_r_e_spearman"] = ka.spearman_rank_corr(
                                    final_lambda_e, final_r_e)
                                res["lambda_e_dip_test"] = ka.hartigan_dip_test(
                                    list(final_lambda_e.values()), seed=seed + 1)
                                res["r_e_dip_test"] = ka.hartigan_dip_test(
                                    list(final_r_e.values()), seed=seed + 2)
                            elif model.anchor_lambda_mode != "learned_per_entity":
                                # sec 10.5.1: the r_e dip test also runs for
                                # candidate (d) (both (d) and (d') get it).
                                res["r_e_dip_test"] = ka.hartigan_dip_test(
                                    list(res["r_e_engagement"]["r_e"].values()), seed=seed + 2)
                # 2026-07-04 audit fix (MINOR): ka.measure_entity_rows sets
                # model.eval() and never restores -- functionally inert THIS
                # wave (no dropout/BN, ZCA off on every keyanchor cell) but
                # it would silently leave candidate (b)'s future
                # ZCA-composed cells training in eval mode from the first
                # checkpoint on. Restore explicitly here, mirroring
                # evaluate_pool()'s own model.train() restore convention.
                # (Covers the sec 3.7 alignment/h1-companion calls above too
                # -- both also flip to eval() internally and never restore.)
                model.train()
            checkpoints.append(res)
            if ckpt_dir is not None and model.anchor_active:
                # sec 10.10 item 1 (the checkpoint WRITER): the anchor
                # table's TRAINED-ROW block ONLY (held-out/other rows are
                # architecturally zero and never trained, sec 2.2 -- saving
                # them would be dead weight, not a fidelity loss; this
                # matches the design's own "27KB negligible" sizing, which
                # is exactly n_train*d_state*4 bytes = 107*64*4 = 27,392 B
                # for the (107,64) block, NOT the full (vocab,64) table).
                ckpt_payload = {
                    "step": step,
                    "anchor_train_ids": model.anchor_train_ids_buf.detach().cpu(),
                    "anchor_table_trained_rows":
                        model.anchor_table.weight[model.anchor_train_ids_buf].detach().cpu(),
                }
                if model.anchor_lambda_mode == "learned":
                    ckpt_payload["anchor_lambda_raw"] = model.anchor_lambda_raw.detach().cpu()
                elif model.anchor_lambda_mode == "learned_per_entity":
                    ckpt_payload["anchor_lambda_table_trained_rows"] = (
                        model.anchor_lambda_table.weight[model.anchor_train_ids_buf].detach().cpu())
                ckpt_path = os.path.join(ckpt_dir, f"step{step}.pt")
                torch.save(ckpt_payload, ckpt_path)
                ckpt_written.append(ckpt_path)
            geo3_admission = (compute_geo3_admission(cfg, trajectory, checkpoints, n_geo3_fallback_train_steps)
                               if model.geo3_active else None)
            partial = _assemble_result(cfg, steps, seed, force_rank_k, trunc_impl, pool_report,
                                        d_model, d_state, trajectory, checkpoints, n_skipped, t0,
                                        timed_out=False, steps_completed=steps_completed, complete=False,
                                        exactness_config=exactness_config, geo3_admission=geo3_admission,
                                        ckpt_written=ckpt_written,
                                        unblind_override_stamp=(ka.override_stamp_payload(unblind_override_at)
                                                                 if unblind_override_at is not None else None))
            _dump(out_path, partial)
            primary_h1 = m2[cfg.H_train[0]]
            nan = float("nan")   # entries may be the all-batches-skipped variant (TruncationError)
            print(f"  [checkpoint step {step}] h={cfg.H_train[0]}: "
                  f"recovered_frac@0.9={primary_h1.get('recovered_frac@0.9', nan):.3f}  "
                  f"entity_subspace_eff_rank={primary_h1.get('entity_subspace_effective_rank_mean', nan):.3f}  "
                  f"key_gram_dev={primary_h1.get('key_gram_deviation_mean', nan):.4f}  "
                  f"alignment_cos={primary_h1.get('alignment_cos_mean', nan):.4f}  "
                  f"C17_recovered@0.9={c17[cfg.H_train[0]].get('recovered_frac@0.9', nan):.3f}  "
                  f"C19_recovered@0.9={c19[cfg.H_train[0]].get('recovered_frac@0.9', nan):.3f}", flush=True)

        if timeout_s is not None and time.time() - t0 > timeout_s:
            print(f"  internal timeout ({timeout_s}s) reached at step {step}; stopping early", flush=True)
            timed_out = True
            break

    geo3_admission_final = (compute_geo3_admission(cfg, trajectory, checkpoints, n_geo3_fallback_train_steps)
                             if model.geo3_active else None)
    anchor_lambda_summary = None
    if model.anchor_active and model.anchor_lambda_mode == "learned":
        lambda_traj = [{"step": t["step"], "lambda": t["lambda"]} for t in trajectory if "lambda" in t]
        if lambda_traj:
            anchor_lambda_summary = ka.lambda_window_summary(lambda_traj)
    anchor_lambda_e_summary = None
    if model.anchor_active and model.anchor_lambda_mode == "learned_per_entity" and lambda_e_trajectory:
        # sec 10.5.1: the per-entity interior-band-fraction, extending sec
        # 3.2's 3-part rule per entity instead of to one shared scalar.
        per_entity_series: dict = {}
        for pt in lambda_e_trajectory:
            for eid, v in pt["lambda_e"].items():
                per_entity_series.setdefault(eid, []).append(v)
        anchor_lambda_e_summary = ka.interior_band_fraction_per_entity(per_entity_series)
    unblind_stamp = ka.override_stamp_payload(unblind_override_at) if unblind_override_at is not None else None
    return _assemble_result(cfg, steps, seed, force_rank_k, trunc_impl, pool_report, d_model, d_state,
                             trajectory, checkpoints, n_skipped, t0, timed_out, steps_completed,
                             complete=(not timed_out and steps_completed >= steps),
                             exactness_config=exactness_config, geo3_admission=geo3_admission_final,
                             anchor_lambda_summary=anchor_lambda_summary,
                             anchor_lambda_e_summary=anchor_lambda_e_summary,
                             ckpt_written=ckpt_written, unblind_override_stamp=unblind_stamp)


# ---------------------------------------------------------------------------
# Smoke gate (run FIRST on cluster; tiny scale, no real training). REQUIRES
# CUDA (chunk_delta_rule has no CPU path) and network access on first run
# (GPT-2 tokenizer download; cached thereafter).
# ---------------------------------------------------------------------------

def smoke(device):
    print("=" * 60 + "\n  DELTANET-RD SMOKE GATE\n" + "=" * 60)
    assert device == "cuda", "DeltaNet-RD requires CUDA (chunk_delta_rule has no CPU path)"

    print("\n[1] grammar_rd self-test (tokenizer pool verification, template rendering, "
          "beta-mask/buffer exactness, injectivity, C17/C19 pool routing)")
    grd._self_test()

    print("\n[2] model_rd self-test (forward/backward, buffer pinning, no-W_q param surface, "
          "C15 + negative test, blank-out, R2-8 leak check, entity_subspace_rank, C16 diagnostics)")
    mrd._self_test()

    print("\n[2b] NO-EVAL-LEAK assert (section 14.3/14.5's Nichani discipline, mini-audit "
          "prescription): the eval scoring path (evaluate_pool + _recovery_stats) computes "
          "ABSOLUTE COSINE only -- no softmax/argmax/cross-entropy/NCE over candidates may "
          "appear anywhere in it; the contrastive term is a TRAINING-only object. Positive "
          "control: the same scan MUST find the NCE machinery in train()'s source (proving "
          "the scan detects what it looks for).")
    import inspect
    import re
    eval_src = inspect.getsource(evaluate_pool) + inspect.getsource(_recovery_stats)
    # word-boundary tokens (plain substring search would false-positive on
    # e.g. "convenience" containing "nce")
    forbidden = (r"\bnce\w*", r"\bsoftmax\b", r"\bargmax\b", r"\bcross_entropy\b", r"\blogits\b")
    leaked = [pat for pat in forbidden if re.search(pat, eval_src)]
    assert not leaked, f"EVAL LEAK: forbidden contrastive/decision tokens in the eval path: {leaked}"
    train_src = inspect.getsource(train) + inspect.getsource(nce_loss)
    assert re.search(r"\bnce_loss\b", train_src) and re.search(r"\bcross_entropy\b", train_src), \
        "no-eval-leak scan is VACUOUS: it cannot find the NCE machinery even in train()'s own source"
    print("  eval path clean of {nce*, softmax, argmax, cross_entropy, logits}; positive control "
          "finds them in train()/nce_loss (scan has teeth)")

    print("\n[3] end-to-end mini training loop (a handful of REAL SGD steps) stays finite "
          "through held-out hops AND C17/C19 pools")
    tokenizer = grd.load_gpt2_tokenizer()
    pools, pool_report = grd.build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)
    pools = pools.to(device)
    cfg = DeltaNetRDTaskConfig(K=8, conv_size=4, H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    model = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                             buffer_id=pools.buffer_id).to(device)
    result = train(model, cfg, pools, pool_report, device, d_model=64, d_state=64,
                    steps=20, batch_size=16, log_every=10, ckpt_every=20)
    assert result["steps_completed"] == 20 and result["complete"] is True
    assert result["checkpoints"], "no checkpoint was written"
    final = result["checkpoints"][-1]
    for pool_name in ("M2_in_distribution", "M3_held_out", "C17_heldout_entities", "C19_heldout_template"):
        for h, entry in final[pool_name].items():
            assert entry["mean_cos"] == entry["mean_cos"], f"NaN mean_cos at {pool_name} h={h}"
            assert abs(entry["pred_norm_mean"]) < 1e6, f"exploded pred norm at {pool_name} h={h}"
            assert entry["key_gram_deviation_mean"] == entry["key_gram_deviation_mean"], \
                f"NaN key_gram_deviation at {pool_name} h={h}"
    print(f"  20-step mini run: complete={result['complete']}, skip_rate={result['skip_rate']:.4%}, "
          f"all 4 pools' per-hop stats finite (H_train={cfg.H_train}, H_test={cfg.H_test})")

    print("\n[4] force_rank_k end-to-end, BOTH --trunc-impl variants (deviation #6's fallback): "
          "mini run with rank forced trains without crashing, entity-subspace rank constrained, "
          "trunc_impl recorded in the result JSON")
    for impl in ("eigh", "svd_lowrank"):
        model_fr = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64,
                                    conv_size=cfg.conv_size, buffer_id=pools.buffer_id,
                                    trunc_impl=impl).to(device)
        result_fr = train(model_fr, cfg, pools, pool_report, device, d_model=64, d_state=64,
                           steps=20, batch_size=16, log_every=10, ckpt_every=20, force_rank_k=4)
        assert result_fr["complete"] is True
        assert result_fr["trunc_impl"] == impl, \
            f"result JSON records trunc_impl={result_fr['trunc_impl']!r}, expected {impl!r}"
        final_fr = result_fr["checkpoints"][-1]["M2_in_distribution"][cfg.H_train[0]]
        assert final_fr["entity_subspace_effective_rank_mean"] <= 4 + 1e-2, \
            f"[{impl}] force_rank_k=4 did not constrain the entity-subspace rank as expected"
        print(f"  [{impl}] force_rank_k=4 mini run: complete={result_fr['complete']}, "
              f"skip_rate={result_fr['skip_rate']:.4%}, "
              f"entity_subspace_eff_rank={final_fr['entity_subspace_effective_rank_mean']:.3f} <= 4")

    print("\n[5] beta-dump Wave -1 smoke (DELTANET_RD_EXACTNESS_DESIGN.md sec 3.2 build item): "
          "training loss BITWISE IDENTICAL with vs without the post-training beta/succ/tgt_slot "
          "Z_dump call present (that call lives strictly AFTER train() returns, under no_grad, "
          "on an INDEPENDENTLY-seeded probe batch). PRIMARY assertion: the model's OWN "
          "parameters are BITWISE IDENTICAL before vs after the dump call, on the SAME model "
          "instance (exact -- no_grad + zero in-place ops make this the direct, unconfounded "
          "test of 'does the dump call touch the compute graph'). SECONDARY, diagnostic-only: "
          "two independent, sequentially-launched training runs WITHIN THIS SAME PROCESS at the "
          "same seed are compared and printed -- NOT asserted bitwise-equal, because repeated "
          "same-config/same-seed training on this box's stack carries a genuine last-few-ULP "
          "noise floor (Triton kernel-level float non-associativity, unrelated to this build's "
          "changes -- measured CROSS-process on the PRE-EDIT, unmodified run_deltanet_rd.py/"
          "model_rd.py run twice at a fixed seed/config: the two pre-edit runs diverge from "
          "EACH OTHER by a comparable magnitude, step 1 identical then growing; the in-process "
          "repeat below exhibits the same divergence class); asserting exact equality here "
          "would be testing GPU determinism the codebase never guaranteed, not this build's "
          "own correctness.")
    torch.manual_seed(99)
    model_a = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                               buffer_id=pools.buffer_id).to(device)
    result_a = train(model_a, cfg, pools, pool_report, device, d_model=64, d_state=64,
                      steps=100, batch_size=16, seed=42, log_every=20, ckpt_every=100)
    state_before = {k: v.detach().clone() for k, v in model_a.state_dict().items()}
    model_a.eval()
    gz = torch.Generator(device=device).manual_seed(42 + 20_000)
    bz = sample_batch_rd(cfg, 4, gz, hop_set=(1,), pools=pools, device=device)
    with torch.no_grad():
        _ = model_a.bind(bz, force_rank_k=None, return_beta=True)   # "with the dump field"
    state_after = model_a.state_dict()
    mismatched = [k for k in state_before if not torch.equal(state_before[k], state_after[k])]
    assert not mismatched, \
        f"the beta/succ/tgt_slot Z_dump call MUTATED model parameters/buffers: {mismatched}"
    print(f"  PRIMARY: all {len(state_before)} parameter/buffer tensors BITWISE IDENTICAL before "
          f"vs after the post-training beta/succ/tgt_slot Z_dump call (same model instance)")

    torch.manual_seed(99)
    model_b = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                               buffer_id=pools.buffer_id).to(device)
    result_b = train(model_b, cfg, pools, pool_report, device, d_model=64, d_state=64,
                      steps=100, batch_size=16, seed=42, log_every=20, ckpt_every=100)
    losses_a = [t["loss"] for t in result_a["trajectory"]]
    losses_b = [t["loss"] for t in result_b["trajectory"]]
    max_abs_diff = max(abs(a - b) for a, b in zip(losses_a, losses_b))
    print(f"  SECONDARY (diagnostic only, not asserted): two independent SAME-PROCESS training "
          f"runs at seed=42, {len(losses_a)} logged steps, max abs loss diff = {max_abs_diff:.2e} "
          f"(this run's GPU-kernel noise floor -- {losses_a} vs {losses_b})")

    print("\n[6] embed_arms self-test (offline/CPU-only portion: used_identity_ids, arm (i) "
          "orthonormality + feasibility guard, arm (iv) reachability/reproducibility, arm "
          "(i-strong) pool construction)")
    embed_arms._self_test()

    print("\n[7] arm (ii) frozen_gpt2_span: build the frozen table + Wave -1 span-projection "
          "ISOMETRY smoke (max pairwise-cosine discrepancy raw-768 vs projected-d_model < 1e-5, "
          "sec 4.3/6 -- REQUIRES network + GPT-2 model weights on first run). d_model=256 here "
          "(NOT the usual d_model=64 'keep it light' smoke convention) -- sec 4.2's own "
          "feasibility bound (n_identities <= d_model, ~251 <= d_model) makes arms (i)/(ii)/(iv) "
          "STRUCTURALLY INFEASIBLE below d_model=251; this is the real Wave-1 d_model, not padding.")
    ids_ii, vals_ii, span_meta = embed_arms.build_frozen_gpt2_span(pools, d_model=256)
    max_discrepancy = embed_arms.span_projection_isometry_check(ids_ii, vals_ii)
    assert max_discrepancy < 1e-5, \
        f"arm (ii) span-projection isometry FAILED: max pairwise-cosine discrepancy " \
        f"{max_discrepancy:.2e} >= 1e-5"
    print(f"  span_rank={span_meta['span_rank']} (of {span_meta['n_identities']} identities, "
          f"wte_dim={span_meta['wte_dim']}); max pairwise-cosine discrepancy = {max_discrepancy:.2e} "
          f"(< 1e-5 required)")

    print("\n[8] embed-source arms + F-geo/F-nce instruments: short end-to-end mini training runs "
          "(K=8, 30 steps each), each stays finite, arm-specific invariants hold, EVERYTHING here "
          "is ADDITIVE and OFF by default -- the default (no-flag) path exercised in [3]/[4] above "
          "is untouched by any of this. [8a]/[8b]/[8c] use d_model=256 for the same feasibility "
          "reason as [7]; [8d]-[8g] (i-strong/lambda_orth/zca/fnce) do not touch embed_source and "
          "stay at the usual d_model=64 smoke weight")

    print("  [8a] arm (i) frozen_orthonormal: rows exactly pinned pre- AND post-training")
    ids_i, vals_i = embed_arms.build_frozen_orthonormal(pools, d_model=256)
    m_i = DeltaNetRDBlock(pools.vocab_size_total, d_model=256, d_state=64, conv_size=cfg.conv_size,
                           buffer_id=pools.buffer_id, embed_source="frozen_orthonormal",
                           frozen_row_ids=ids_i.to(device), frozen_row_values=vals_i.to(device)).to(device)
    r_i = train(m_i, cfg, pools, pool_report, device, d_model=256, d_state=64, steps=30, batch_size=16,
                log_every=30, ckpt_every=30, exactness_config={"embed_source": "frozen_orthonormal"})
    assert r_i["complete"] is True
    assert torch.allclose(m_i.embed.weight[ids_i.to(device)], vals_i.to(device), atol=1e-5), \
        "arm (i) frozen rows drifted after training"
    print(f"    30-step mini run complete={r_i['complete']}; frozen rows bit-exact post-training")

    print("  [8b] arm (iv) frozen_gram_matched: builds + trains without crashing, rows pinned")
    ids_iv, vals_iv = embed_arms.build_frozen_gram_matched(pools, d_model=256, alpha=1.0, rho=0.3)
    m_iv = DeltaNetRDBlock(pools.vocab_size_total, d_model=256, d_state=64, conv_size=cfg.conv_size,
                            buffer_id=pools.buffer_id, embed_source="frozen_gram_matched",
                            frozen_row_ids=ids_iv.to(device), frozen_row_values=vals_iv.to(device)).to(device)
    r_iv = train(m_iv, cfg, pools, pool_report, device, d_model=256, d_state=64, steps=30, batch_size=16,
                 log_every=30, ckpt_every=30,
                 exactness_config={"embed_source": "frozen_gram_matched", "gram_alpha": 1.0, "gram_rho": 0.3})
    assert r_iv["complete"] is True
    assert torch.allclose(m_iv.embed.weight[ids_iv.to(device)], vals_iv.to(device), atol=1e-5)
    print(f"    30-step mini run complete={r_iv['complete']}; frozen rows bit-exact post-training")

    print("  [8c] arm (ii) frozen_gpt2_span: trains without crashing, rows pinned")
    m_ii = DeltaNetRDBlock(pools.vocab_size_total, d_model=256, d_state=64, conv_size=cfg.conv_size,
                            buffer_id=pools.buffer_id, embed_source="frozen_gpt2_span",
                            frozen_row_ids=ids_ii.to(device), frozen_row_values=vals_ii.to(device)).to(device)
    r_ii = train(m_ii, cfg, pools, pool_report, device, d_model=256, d_state=64, steps=30, batch_size=16,
                 log_every=30, ckpt_every=30, exactness_config={"embed_source": "frozen_gpt2_span"})
    assert r_ii["complete"] is True
    print(f"    30-step mini run complete={r_ii['complete']}")

    print("  [8d] arm (i-strong): restricted 32+32 pool + surgical pin trains without crashing")
    train32, heldout32, pin_ids, pin_vals = embed_arms.build_i_strong_pool(pools, d_state=64)
    strong_pools = embed_arms.restrict_pools_for_strong_pin(pools, train32, heldout32).to(device)
    m_sp = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                            buffer_id=pools.buffer_id, strong_pin_ids=pin_ids.to(device),
                            strong_pin_values=pin_vals.to(device)).to(device)
    r_sp = train(m_sp, cfg, strong_pools, pool_report, device, d_model=64, d_state=64, steps=30,
                 batch_size=16, log_every=30, ckpt_every=30, exactness_config={"strong_pin": True})
    assert r_sp["complete"] is True
    print(f"    30-step mini run complete={r_sp['complete']} on the restricted 32+32 pool")

    print("  [8e] F-geo-1 (lambda_orth): trains without crashing, loss_orth logged and finite")
    m_orth = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                              buffer_id=pools.buffer_id).to(device)
    r_orth = train(m_orth, cfg, pools, pool_report, device, d_model=64, d_state=64, steps=30,
                    batch_size=16, log_every=10, ckpt_every=30, lambda_orth=1.0,
                    exactness_config={"lambda_orth": 1.0})
    assert r_orth["complete"] is True
    assert all("loss_orth" in t and t["loss_orth"] == t["loss_orth"] for t in r_orth["trajectory"]), \
        "lambda_orth>0 but loss_orth missing/NaN in the trajectory log"
    print(f"    30-step mini run complete={r_orth['complete']}; loss_orth logged and finite at "
          f"every trajectory point")

    print("  [8f] F-geo-2 (use_zca): trains without crashing")
    m_zca = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                             buffer_id=pools.buffer_id, use_zca=True).to(device)
    r_zca = train(m_zca, cfg, pools, pool_report, device, d_model=64, d_state=64, steps=30,
                   batch_size=16, log_every=30, ckpt_every=30, exactness_config={"use_zca": True})
    assert r_zca["complete"] is True
    print(f"    30-step mini run complete={r_zca['complete']} (ZCA module active but still in "
          f"identity warm-up at this step count -- model_rd.py's own [model 14] carries the "
          f"module's dedicated post-warmup numerics smoke)")

    print("  [8g] F-nce (fnce_m): trains without crashing at K=8 (m=3 <= K-1=7)")
    m_fnce = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                              buffer_id=pools.buffer_id).to(device)
    r_fnce = train(m_fnce, cfg, pools, pool_report, device, d_model=64, d_state=64, steps=30,
                    batch_size=16, log_every=30, ckpt_every=30, fnce_m=3,
                    exactness_config={"fnce_m": 3})
    assert r_fnce["complete"] is True
    print(f"    30-step mini run complete={r_fnce['complete']}")

    print("\n[9] F-geo-3 (DELTANET_RD_EXACTNESS_DESIGN.md sec 14): end-to-end mini training run "
          "with --use-geo3-equivalent construction, the PATCHED C16 self-query diagnostic "
          "(sec 14.9 item 1 -- confirms alignment_cos_mean is EXACTLY 1.0, not just close, proving "
          "the tautology-by-construction is the CORRECT tautology, not a leftover bug), the "
          "architecturally-forced near-zero key-Gram deviation, and the sec 14.10 substitute "
          "admission stack's fields (logged, sane, finite)")
    m_g3s = DeltaNetRDBlock(pools.vocab_size_total, d_model=64, d_state=64, conv_size=cfg.conv_size,
                             buffer_id=pools.buffer_id, geo3_active=True, geo3_n_iter=12,
                             geo3_resid_tol=1e-2).to(device)
    r_g3 = train(m_g3s, cfg, pools, pool_report, device, d_model=64, d_state=64, steps=100,
                 batch_size=16, log_every=50, ckpt_every=50,
                 exactness_config={"geo3_active": True, "geo3_n_iter": 12, "geo3_resid_tol": 1e-2})
    assert r_g3["complete"] is True
    assert r_g3["exactness_config"]["geo3_active"] is True
    assert r_g3["exactness_config"]["geo3_n_iter"] == 12
    final_g3 = r_g3["checkpoints"][-1]
    for pool_name in ("M2_in_distribution", "M3_held_out", "C17_heldout_entities", "C19_heldout_template"):
        for h, entry in final_g3[pool_name].items():
            assert entry.get("geo3_diagnostics_are_non_evidence") is True, \
                f"geo3 run's {pool_name} h={h} entry missing the non-evidence label"
            align = entry["alignment_cos_mean"]
            assert abs(align - 1.0) < 1e-5, \
                f"PATCHED alignment_cos_mean is NOT ~1.0 at {pool_name} h={h} (got {align}) -- " \
                f"the q_self=k_eff_items patch is not actually wired through evaluate_pool"
            key_gd = entry["key_gram_deviation_mean"]
            assert key_gd < 0.05, \
                f"key_gram_deviation_mean={key_gd:.4f} at {pool_name} h={h} is NOT near-zero -- " \
                f"F-geo-3's structural orthonormality guarantee is not holding"
    print(f"  100-step mini run complete={r_g3['complete']}; alignment_cos_mean EXACTLY ~1.0 and "
          f"key_gram_deviation_mean near-zero at every pool/hop (the architecturally-forced, "
          f"non-evidence diagnostics behave as designed, sec 14.9 item 1 / sec 14.10)")

    admission = r_g3.get("geo3_admission")
    assert admission is not None, "geo3_active run did not attach a geo3_admission block"
    for key in ("value_salvage_tier_pass", "ns_converged_no_fallback", "finite_loss_no_divergence",
                "task_performance_floor_pass", "admissible"):
        assert isinstance(admission[key], bool), f"geo3_admission[{key!r}] is not a bool: {admission[key]!r}"
    assert admission["step0_loss"] == admission["step0_loss"], "geo3_admission step0_loss is NaN"
    assert admission["final_loss"] == admission["final_loss"], "geo3_admission final_loss is NaN"
    print(f"  sec 14.10 substitute admission stack (LOGGED, not asserted as a pass/fail gate here): "
          f"{json.dumps(admission, indent=2)}")

    print("\n" + "=" * 60 + "\n  ALL DELTANET-RD SMOKE CHECKS PASSED\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--K", type=int, default=32, help="bindings per sample (== pool draw size)")
    ap.add_argument("--conv-size", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-state", type=int, default=64, choices=[64, 128],
                     help="DeltaNet state dim (single head, C11). ONLY the measured-safe "
                          "head dims are accepted -- chunk_delta_rule's backward crashes at "
                          "d_state<64 on this box's build (model_rd._SAFE_D_STATE, F15-LM "
                          "2026-07-02). 64 is the Wave-1 primary (section 4.1 continuity).")
    ap.add_argument("--h-train", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--h-test", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--h-extra", type=int, nargs="+", default=[7, 21])
    ap.add_argument("--heldout-frac", type=float, default=0.5,
                     help="entity name pool train/heldout split fraction (C17); see grammar_rd.py's "
                          "build_entity_pools docstring for why 0.5 is the build-time default")
    ap.add_argument("--force-rank-k", type=int, default=None)
    ap.add_argument("--trunc-impl", choices=["eigh", "svd_lowrank"], default="eigh",
                     help="rank-truncation implementation for the force-rank arm. 'eigh' = the "
                          "design-default rank_utils.truncate_to_rank; 'svd_lowrank' = the "
                          "randomized-SVD stability fallback ported from model_dn.py. Build "
                          "deviation #6's pre-registered re-add condition FIRED at Wave -1 "
                          "(2026-07-02): all 3 force-rank cells crashed with a FORWARD "
                          "linalg.eigh non-convergence on real-embedding states. No effect when "
                          "--force-rank-k is not given. Recorded in the result JSON; the sweep "
                          "suffixes non-default impls into run names (no resume collisions).")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--internal-timeout", type=float, default=None)
    ap.add_argument("--no-save-z", dest="save_z", action="store_false", default=True,
                     help="S_T dumps are ON BY DEFAULT for every run (the project's own "
                          "eval-truncation lesson: dumps must exist, not be reconstructable only "
                          "from a re-run). Pass --no-save-z to disable for a throwaway/dry run.")
    # -- DELTANET_RD_EXACTNESS_DESIGN.md sec 4/5.5 extensions. ALL default
    # to the pre-extension behavior (embed_source="learned", no strong-pin,
    # lambda_orth=0, no ZCA, no F-nce) -- byte-identical default path,
    # regression-checked against the pre-edit code (see this repo's build
    # notes for the exact seed/loss comparison).
    ap.add_argument("--embed-source", choices=["learned", "frozen_orthonormal", "frozen_gpt2_span",
                                                "frozen_gram_matched"], default="learned",
                     help="sec 4: which arm's embedding-table construction to use for the "
                          "identity rows (names/rel-verbs/period). 'learned' (default) is the "
                          "UNCHANGED arm (iii) baseline -- every other value freezes those rows "
                          "via model_rd.py's generalized pin_rows_ mechanism (embed_arms.py).")
    ap.add_argument("--gram-alpha", type=float, default=1.0,
                     help="arm (iv) frozen_gram_matched's alpha (sec 4.5). This build fixes "
                          "alpha=1.0 (the family's pure-Gaussian/max-disorder endpoint the "
                          "reachability calc characterizes) and calibrates rho alone -- see "
                          "calibrate_arm_iv.py.")
    ap.add_argument("--gram-rho", type=float, default=0.3,
                     help="arm (iv) frozen_gram_matched's rho (sec 4.5). 0.3 is a build-time "
                          "placeholder (design's own session number); the CALIBRATED per-K value "
                          "comes from calibrate_arm_iv.py's closed-loop probe, not hardcoded here.")
    ap.add_argument("--strong-pin", action="store_true",
                     help="arm (i-strong), sec 4.4: surgical k_eff pin bypassing W_k/conv at "
                          "write+query positions, on a reduced 32-train+32-heldout entity pool. "
                          "REQUIRES K <= 32. Combines with --embed-source (design exercises it "
                          "at the default 'learned' only).")
    ap.add_argument("--lambda-orth", type=float, default=0.0,
                     help="F-geo-1 (sec 5.5): orthogonality penalty weight, "
                          "lambda_orth * ||K_eff^T K_eff - I_K||_F^2 added to the training loss. "
                          "0.0 (default) = OFF.")
    ap.add_argument("--use-zca", action="store_true",
                     help="F-geo-2 (sec 5.5/BLOCK-3): ZCA whitening on the key path. OFF by default.")
    ap.add_argument("--fnce-m", type=int, default=None,
                     help="F-nce (sec 5.5): fixed-m crowding-normalized InfoNCE (m negatives per "
                          "query, uniform over the K-1 in-episode non-targets, REQUIRES m<=K-1). "
                          "None (default) = OFF, the unchanged full-K-way L_nce.")
    ap.add_argument("--use-geo3", action="store_true",
                     help="F-geo-3 (sec 14): per-episode differentiable Newton-Schulz "
                          "orthogonalization of the K keys actually written into S_T. OFF by "
                          "default. MUTUALLY EXCLUSIVE with --strong-pin (model_rd.py asserts this "
                          "at construction). Wave 1 launch is GATED on the sec 14.6 drift diagnostic's "
                          "launch_read result -- see run_deltanet_rd_exactness_sweep.py's geo3 wave.")
    ap.add_argument("--geo3-n-iter", type=int, default=12,
                     help="F-geo-3's Newton-Schulz iteration count (sec 14.4: a single "
                          "pre-registered default, escalated at most once to 20, never searched). "
                          "No effect when --use-geo3 is not given.")
    ap.add_argument("--geo3-resid-tol", type=float, default=1e-2,
                     help="F-geo-3's per-episode Gram-residual fallback trigger (sec 14.4: "
                          "~100-300x tighter than the behavioral bars require, chosen by analogy "
                          "to i-strong's exactness). No effect when --use-geo3 is not given.")
    # -- KEY_ANCHORING_DESIGN.md sec 2.2/2.4 extensions. ALL default to OFF
    # (anchor_active=False, lambda_anchor=0.0) -- byte-identical default
    # path, same discipline as every DELTANET_RD_EXACTNESS_DESIGN.md sec
    # 4/5.5 extension above.
    ap.add_argument("--anchor-active", action="store_true",
                     help="KEY_ANCHORING sec 2.2, candidate (d), PRIMARY: trainable per-entity "
                          "anchor table + masked gather/scatter blend, inserted PRE-Newton-Schulz "
                          "at the existing geo3 site. REQUIRES --use-geo3. Mutually exclusive with "
                          "--strong-pin (model_rd.py asserts this) and with --lambda-anchor "
                          "(candidate (c) is a separate cell, sec 2.4).")
    ap.add_argument("--anchor-lambda-mode", choices=["learned", "fixed", "learned_per_entity"],
                     default="learned",
                     help="'learned' (default): single learned scalar, sigmoid(raw_param), init 0.5. "
                          "'fixed': a pre-registered grid value (sec 2.2's fallback diagnostic, "
                          "e.g. one of {0.3,0.6,0.9}) -- REQUIRES --anchor-lambda-fixed. "
                          "'learned_per_entity' (KEY_ANCHORING sec 10.5.1, candidate (d')): a "
                          "per-entity table, same sigmoid/init-0.5 convention. No effect "
                          "when --anchor-active is not given.")
    ap.add_argument("--anchor-lambda-fixed", type=float, default=None,
                     help="the fixed-grid lambda value when --anchor-lambda-mode=fixed.")
    ap.add_argument("--anchor-table-frozen", action="store_true",
                     help="KEY_ANCHORING_DESIGN.md sec 10.13, candidate (e) ('frozen-random-table "
                          "ablation'): anchor_table.weight.requires_grad_(False) immediately after "
                          "construction -- the trained-row block never receives a gradient (held-out "
                          "rows already never do, sec 3.3). No effect when --anchor-active is not "
                          "given. Composes with any --anchor-lambda-mode; the wave registers this "
                          "with --anchor-lambda-mode=fixed (matched-lambda comparison to candidate (d)).")
    ap.add_argument("--anchor-table-init-mode", choices=["frame_potential", "random_unit_rows", "dosed"],
                     default="frame_potential",
                     help="'frame_potential' (default): candidate (d)'s own tight-frame-minimized "
                          "init, unchanged. 'random_unit_rows' (KEY_ANCHORING_DESIGN.md sec 10.13, "
                          "candidate (e)): seeded random unit rows with NO frame-potential descent "
                          "(key_anchoring.random_unit_rows_init) -- matches sec 10.13's own "
                          "'frozen-random-table' name and its motivating text, not a trained-but-"
                          "frozen tight frame. 'dosed' (KEY_ANCHORING_DESIGN.md sec 14.1b item 2, "
                          "the coherence dose-response wave): the trained-row block is built here, "
                          "at CLI level, via key_anchoring.build_dose_table/calibrate_dose and "
                          "handed to the model constructor via anchor_table_override -- REQUIRES "
                          "--anchor-dose-target and --anchor-dose-structure. No effect when "
                          "--anchor-active is not given.")
    ap.add_argument("--anchor-dose-target", type=float, default=None,
                     help="KEY_ANCHORING_DESIGN.md sec 14.1b item 3, coherence dose-response wave: "
                          "the target max|cos| (e.g. 0.130/0.284/0.40) key_anchoring.calibrate_dose "
                          "calibrates 't' against. REQUIRED when --anchor-table-init-mode=dosed.")
    ap.add_argument("--anchor-dose-structure", choices=["rank4", "diffuse"], default=None,
                     help="KEY_ANCHORING_DESIGN.md sec 14.1's Rev-14.3 dose-dial structures: "
                          "'rank4' -> subspace_rank=4 (concentrated); 'diffuse' -> subspace_rank="
                          "key_anchoring.DIFFUSE_SUBSPACE_RANK (48 -- the maximum scanned rank "
                          "whose ceiling still clears 0.42; subspace_rank=d_state is a "
                          "mathematically degenerate no-op dial and is NEVER used here, sec 14.1's "
                          "boxed correction). REQUIRED when --anchor-table-init-mode=dosed.")
    ap.add_argument("--anchor-subspace-seed", type=int, default=None,
                     help="KEY_ANCHORING_DESIGN.md sec 14.3: the RNG seed the dose dial's random "
                          "subspace is drawn from. Defaults to key_anchoring.ANCHOR_INIT_SEED "
                          "(20260705) when --anchor-table-init-mode=dosed and this flag is omitted "
                          "-- the registered subspace seed for the mandatory dose grid (one fixed "
                          "subspace per structure across all doses, sec 14.3).")
    ap.add_argument("--lambda-anchor", type=float, default=0.0,
                     help="KEY_ANCHORING sec 2.4, candidate (c): soft cross-episode drift "
                          "regularizer weight (L_anchor added to the training loss only; model "
                          "behavior at inference is byte-identical to bare geo3). REQUIRES "
                          "--use-geo3. Mutually exclusive with --anchor-active. 0.0 (default) = OFF.")
    ap.add_argument("--drift-probe", action="store_true",
                     help="KEY_ANCHORING sec 3.6: activate the per-checkpoint drift-diagnostic "
                          "probe (post-NS + pre-NS pooled cross-episode drift, logged into every "
                          "checkpoint's own 'drift_probe' field) -- REQUIRED for the reference arms "
                          "(sec 3.6's band-derivation input); harmless but unnecessary elsewhere.")
    ap.add_argument("--drift-probe-n-entities", type=int, default=8,
                     help="sec 14.6's pinned minimum (>=8) -- only increase, never decrease.")
    ap.add_argument("--drift-probe-n-resamples", type=int, default=32,
                     help="sec 14.6's pinned minimum (>=32) -- only increase, never decrease.")
    ap.add_argument("--unblind-override-at", type=float, default=None,
                     help="KEY_ANCHORING sec 3.6 Rev 5: threaded by the launcher's "
                          "--unblind-override path (run_deltanet_rd_exactness_sweep.py) -- the "
                          "override timestamp. When given, this run's OWN result JSON is stamped "
                          "claim_tier='descriptive' + unblind_override=True + this timestamp at "
                          "assembly time (never post-hoc). Never set this by hand for an ordinary run.")
    ap.add_argument("--rev7-engagement", action="store_true",
                     help="KEY_ANCHORING sec 10 (Rev 7.1): activate the r_e + null-pool BH-FDR "
                          "engagement measurement at the final checkpoint (ka.measure_r_e_and_null_pool). "
                          "REQUIRES --anchor-active and --drift-probe. Validates "
                          "REV7_THRESHOLD_PINNED.json via ka.validate_rev7_pin() at startup -- "
                          "REFUSES (loud error, no training) if the pin is missing, hash-mismatched, "
                          "or fails its own live re-derivation (sec 10.3.3's per-run defense-in-depth "
                          "copy of the sweep-level launcher gate).")
    ap.add_argument("--ckpt-dir", type=str, default=None,
                     help="KEY_ANCHORING sec 10.10: directory to write the anchor table's "
                          "TRAINED-ROW checkpoint (+ the per-entity lambda table for candidate (d'), "
                          "or the scalar anchor_lambda_raw for candidate (d)) at every eval checkpoint "
                          "AND the final step. No effect when --anchor-active is not given. None "
                          "(default) = no checkpoint writing.")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.smoke:
        smoke(device)
        return

    assert device == "cuda", "DeltaNet-RD requires CUDA for real training (chunk_delta_rule has no CPU path)"
    if args.ckpt_every > 2000:
        print(f"WARNING: --ckpt-every={args.ckpt_every} > 2000: violates section 8's build requirement.",
              flush=True)

    tokenizer = grd.load_gpt2_tokenizer()
    pools, pool_report = grd.build_entity_pools(tokenizer, heldout_frac=args.heldout_frac, seed=0)
    pools = pools.to(device)
    if pool_report["n_train_names"] < args.K or pool_report["n_heldout_names"] < args.K:
        print(f"WARNING: K={args.K} exceeds one of the entity pools "
              f"(train={pool_report['n_train_names']}, heldout={pool_report['n_heldout_names']}) "
              f"-- sample_batch_rd will assert and fail.", flush=True)

    # -- sec 4/4.4: embedding-source arm + arm (i-strong) construction.
    # Byte-identical no-op when every new flag is at its default (embed_source
    # ="learned", strong_pin=False) -- frozen_row_ids/strong_pin_ids stay
    # None, DeltaNetRDBlock's constructor takes its original code path.
    frozen_row_ids = frozen_row_values = None
    embed_meta = {}
    if args.embed_source == "frozen_orthonormal":
        frozen_row_ids, frozen_row_values = embed_arms.build_frozen_orthonormal(pools, d_model=args.d_model)
    elif args.embed_source == "frozen_gpt2_span":
        frozen_row_ids, frozen_row_values, embed_meta = embed_arms.build_frozen_gpt2_span(
            pools, d_model=args.d_model)
    elif args.embed_source == "frozen_gram_matched":
        frozen_row_ids, frozen_row_values = embed_arms.build_frozen_gram_matched(
            pools, d_model=args.d_model, alpha=args.gram_alpha, rho=args.gram_rho)
        embed_meta = {"gram_alpha": args.gram_alpha, "gram_rho": args.gram_rho}
    if frozen_row_ids is not None:
        frozen_row_ids = frozen_row_ids.to(device)
        frozen_row_values = frozen_row_values.to(device)

    strong_pin_ids = strong_pin_values = None
    if args.strong_pin:
        assert args.K <= 32, \
            f"arm (i-strong) requires K <= 32 (sec 4.4's reduced 32-train+32-heldout pool), got K={args.K}"
        train32, heldout32, strong_pin_ids, strong_pin_values = embed_arms.build_i_strong_pool(
            pools, d_state=args.d_state)
        pools = embed_arms.restrict_pools_for_strong_pin(pools, train32, heldout32).to(device)
        strong_pin_ids = strong_pin_ids.to(device)
        strong_pin_values = strong_pin_values.to(device)

    if args.fnce_m is not None:
        assert args.fnce_m <= args.K - 1, \
            f"--fnce-m={args.fnce_m} must be <= K-1={args.K - 1} (not enough in-episode negatives)"

    if args.use_geo3:
        assert not args.strong_pin, \
            "--use-geo3 and --strong-pin are MUTUALLY EXCLUSIVE (sec 14.2) -- model_rd.py's own " \
            "constructor assert would also catch this, checked here for a clearer CLI-level error"

    if args.anchor_active:
        assert args.use_geo3, \
            "--anchor-active REQUIRES --use-geo3 (KEY_ANCHORING sec 2.2: anchoring is a " \
            "modification to geo3's own write path)"
        assert not args.lambda_anchor, \
            "--anchor-active (candidate (d)) and --lambda-anchor (candidate (c)) are MUTUALLY " \
            "EXCLUSIVE -- separate cells, sec 2.2 vs sec 2.4"
        if args.anchor_lambda_mode == "fixed":
            assert args.anchor_lambda_fixed is not None, \
                "--anchor-lambda-mode=fixed requires --anchor-lambda-fixed"
    if args.anchor_table_frozen or args.anchor_table_init_mode != "frame_potential":
        assert args.anchor_active, \
            "--anchor-table-frozen / --anchor-table-init-mode require --anchor-active " \
            "(KEY_ANCHORING sec 10.13: candidate (e) is a modification of candidate (d)'s own path)"
    if args.anchor_table_init_mode == "dosed":
        # KEY_ANCHORING_DESIGN.md sec 14.0's F1 fix: every dosed cell in
        # this wave is FROZEN by construction (the whole point is a
        # coherence dose HELD CONSTANT across training, not one SGD is
        # free to drift away from) -- refuse rather than silently allow a
        # dosed-but-trainable cell that would reintroduce the exact drift
        # confound sec 14.0 exists to close.
        assert args.anchor_table_frozen, \
            "--anchor-table-init-mode=dosed REQUIRES --anchor-table-frozen (KEY_ANCHORING_DESIGN.md " \
            "sec 14.0's F1 fix: every dosed cell must be frozen, closing the drift confound by " \
            "construction)"
        assert args.anchor_dose_target is not None and args.anchor_dose_structure is not None, \
            "--anchor-table-init-mode=dosed requires BOTH --anchor-dose-target and " \
            "--anchor-dose-structure (KEY_ANCHORING_DESIGN.md sec 14.1b item 3)"
    if args.rev7_engagement:
        assert args.anchor_active and args.drift_probe, \
            "--rev7-engagement REQUIRES --anchor-active and --drift-probe (KEY_ANCHORING sec 10.2/10.3)"
    if args.lambda_anchor:
        assert args.use_geo3, \
            "--lambda-anchor > 0 (candidate (c)) REQUIRES --use-geo3 (KEY_ANCHORING sec 2.4: the " \
            "regularizer targets geo3's own k_eff_items)"

    cfg = DeltaNetRDTaskConfig(K=args.K, conv_size=args.conv_size, H_train=tuple(args.h_train),
                                H_test=tuple(args.h_test), H_extra=tuple(args.h_extra))

    anchor_table_override = None
    achieved_max_cos = None
    dose_subspace_seed = None
    if args.anchor_table_init_mode == "dosed":
        # KEY_ANCHORING_DESIGN.md sec 14.1b item 2/item 3: build the dosed
        # table HERE (CLI level, CPU, one-time) via the SAME key_anchoring.
        # build_dose_table/calibrate_dose functions every other dosed cell
        # in this wave calls -- never a hand-copied twin. n_train is
        # pools.train_name_ids.numel() (matches the frame_potential/
        # random_unit_rows branches' own n_train derivation inside
        # model_rd.py's constructor).
        dose_subspace_seed = (args.anchor_subspace_seed if args.anchor_subspace_seed is not None
                               else ka.ANCHOR_INIT_SEED)
        subspace_rank = 4 if args.anchor_dose_structure == "rank4" else ka.DIFFUSE_SUBSPACE_RANK
        n_train = pools.train_name_ids.numel()
        healthy_table = ka.frame_potential_init(n_train, args.d_state, seed=ka.ANCHOR_INIT_SEED)
        cal = ka.calibrate_dose(healthy_table, args.anchor_dose_target, subspace_rank, dose_subspace_seed)
        anchor_table_override = ka.build_dose_table(healthy_table, cal["t"], subspace_rank,
                                                       dose_subspace_seed)
        achieved_max_cos = cal["achieved"]
        # sec 14.3's Gate-2-re-semantics dose-verification assertion, run
        # at CELL START (not deferred to a later checkpoint read) --
        # "achieved vs target +/-10%" per Gate-2-as-dose-verification. No
        # assertion needed at the 0.000 control (never routed through this
        # branch at all -- dose_target=0.0 would divide by zero below, and
        # the 0.000 control cells reuse sec 13.10's own already-measured,
        # non-dosed cells per sec 14.2's dose grid, never this CLI path).
        assert args.anchor_dose_target > 0.0, \
            "--anchor-table-init-mode=dosed with --anchor-dose-target=0.0 is a construction error " \
            "(sec 14.2: the 0.000 control reuses sec 13.10's own already-measured cells, never this " \
            "dosed-construction path)"
        rel_err = abs(achieved_max_cos - args.anchor_dose_target) / args.anchor_dose_target
        assert rel_err <= 0.10, (
            f"DOSE VERIFICATION FAILED (sec 14.3): target={args.anchor_dose_target}, "
            f"achieved={achieved_max_cos:.6f}, rel_err={rel_err:.4f} > 0.10 -- this is a "
            f"CONSTRUCTION BUG (calibrate_dose failed to converge within tolerance), not a "
            f"training dynamic; refusing to launch this cell.")
        print(f"sec 14.3 DOSE-VERIFY PASSED: structure={args.anchor_dose_structure} "
              f"subspace_rank={subspace_rank} subspace_seed={dose_subspace_seed} "
              f"target={args.anchor_dose_target} achieved={achieved_max_cos:.6f} "
              f"rel_err={rel_err:.4f} (<=0.10) t={cal['t']:.5f} converged={cal['converged']}",
              flush=True)

    model = DeltaNetRDBlock(pools.vocab_size_total, d_model=args.d_model, d_state=args.d_state,
                             conv_size=cfg.conv_size, buffer_id=pools.buffer_id,
                             trunc_impl=args.trunc_impl, embed_source=args.embed_source,
                             frozen_row_ids=frozen_row_ids, frozen_row_values=frozen_row_values,
                             strong_pin_ids=strong_pin_ids, strong_pin_values=strong_pin_values,
                             use_zca=args.use_zca, geo3_active=args.use_geo3,
                             geo3_n_iter=args.geo3_n_iter, geo3_resid_tol=args.geo3_resid_tol,
                             anchor_active=args.anchor_active, anchor_lambda_mode=args.anchor_lambda_mode,
                             anchor_lambda_fixed=args.anchor_lambda_fixed,
                             anchor_train_ids=(pools.train_name_ids if args.anchor_active else None),
                             anchor_table_frozen=args.anchor_table_frozen,
                             anchor_table_init_mode=args.anchor_table_init_mode,
                             anchor_table_override=anchor_table_override,
                             ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"K={args.K} conv_size={args.conv_size} T_bind={cfg.T_bind} d_model={args.d_model} "
          f"d_state={args.d_state} H_train={cfg.H_train} H_test={cfg.H_test} H_extra={cfg.H_extra} "
          f"force_rank_k={args.force_rank_k} trunc_impl={args.trunc_impl} "
          f"embed_source={args.embed_source} strong_pin={args.strong_pin} "
          f"lambda_orth={args.lambda_orth} use_zca={args.use_zca} fnce_m={args.fnce_m} "
          f"use_geo3={args.use_geo3} geo3_n_iter={args.geo3_n_iter} geo3_resid_tol={args.geo3_resid_tol} "
          f"anchor_active={args.anchor_active} anchor_lambda_mode={args.anchor_lambda_mode} "
          f"anchor_lambda_fixed={args.anchor_lambda_fixed} lambda_anchor={args.lambda_anchor} "
          f"drift_probe={args.drift_probe} "
          f"anchor_table_frozen={args.anchor_table_frozen} "
          f"anchor_table_init_mode={args.anchor_table_init_mode} "
          f"anchor_dose_target={args.anchor_dose_target} "
          f"anchor_dose_structure={args.anchor_dose_structure} "
          f"anchor_subspace_seed={dose_subspace_seed} "
          f"params={n_params} device={device}", flush=True)

    exactness_config = {
        "embed_source": args.embed_source,
        "gram_alpha": args.gram_alpha if args.embed_source == "frozen_gram_matched" else None,
        "gram_rho": args.gram_rho if args.embed_source == "frozen_gram_matched" else None,
        "geo3_active": args.use_geo3,
        "geo3_n_iter": args.geo3_n_iter if args.use_geo3 else None,
        "geo3_resid_tol": args.geo3_resid_tol if args.use_geo3 else None,
        "strong_pin": args.strong_pin, "lambda_orth": args.lambda_orth,
        "use_zca": args.use_zca, "fnce_m": args.fnce_m,
        "anchor_active": args.anchor_active,
        "anchor_lambda_mode": args.anchor_lambda_mode if args.anchor_active else None,
        "anchor_lambda_fixed": args.anchor_lambda_fixed if args.anchor_active else None,
        "lambda_anchor": args.lambda_anchor,
        "drift_probe": args.drift_probe,
        "rev7_engagement": args.rev7_engagement,
        "anchor_table_frozen": args.anchor_table_frozen if args.anchor_active else False,
        "anchor_table_init_mode": args.anchor_table_init_mode if args.anchor_active else "frame_potential",
        # KEY_ANCHORING_DESIGN.md sec 14.1b item 5 (Rev 14.3, coherence
        # dose-response wave): ALWAYS recorded (default None), same
        # always-recorded-even-at-off discipline as every field above --
        # is_done() (run_deltanet_rd_exactness_sweep.py) cross-checks all
        # three identity fields so a dosed cell can never silently
        # resume-match a different dose sharing the same K/seed.
        # achieved_max_cos is the ACTUAL measured value at construction
        # time (from calibrate_dose's own return, not re-derived) so the
        # per-checkpoint item6_table_conditioning.max_abs_cos trajectory
        # (sec 14.0b's read rule) can be cross-checked against the
        # construction-time target directly from the archived JSON.
        "dose_target": args.anchor_dose_target if args.anchor_table_init_mode == "dosed" else None,
        "dose_structure": args.anchor_dose_structure if args.anchor_table_init_mode == "dosed" else None,
        "subspace_seed": dose_subspace_seed if args.anchor_table_init_mode == "dosed" else None,
        "achieved_max_cos": achieved_max_cos if args.anchor_table_init_mode == "dosed" else None,
    }
    rev7_pin_derived = None
    if args.rev7_engagement:
        # sec 10.3.3 leg (ii), per-run defense-in-depth copy of the sweep-
        # level launcher gate: REFUSE before any training happens if the
        # pin is missing, hash-mismatched, or fails its own live
        # re-derivation.
        # sec 13: the pin is PER-d_state -- a d=128 cell must validate (and
        # thread downstream) the D128 pin, never the d=64 default. First
        # d=128 calibration launch failed here: the default pin's script
        # hash was stale AND its d=64 inputs would have silently supplied
        # wrong thresholds to a d=128 run.
        _here = os.path.dirname(os.path.abspath(__file__))
        _pin_name = ("REV7_THRESHOLD_PINNED.json" if args.d_state == 64
                     else f"REV7_THRESHOLD_PINNED_D{args.d_state}.json")
        pin_doc = ka.validate_rev7_pin(pin_path=os.path.join(_here, _pin_name))
        assert pin_doc is not None, (
            f"--rev7-engagement requires a VALID {_pin_name} (sec 10.3.3): the pin "
            "is missing, its recorded script_sha256 does not match the working tree's "
            "rev7_threshold_derive.py, or a live derive() re-run does not reproduce the pin's own "
            "derived block byte-identically. Regenerate with `python rev7_threshold_derive.py "
            "[--d-state N]` and re-commit before launching any Rev-7 anchor-arm cell.")
        assert int(pin_doc["derived"]["inputs"]["d_state"]) == int(args.d_state), (
            f"pin/run d_state mismatch: pin={pin_doc['derived']['inputs']['d_state']} "
            f"run={args.d_state} -- wrong pin file for this cell")
        rev7_pin_derived = pin_doc["derived"]
    result = train(model, cfg, pools, pool_report, device, d_model=args.d_model, d_state=args.d_state,
                    steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                    force_rank_k=args.force_rank_k, log_every=args.log_every, ckpt_every=args.ckpt_every,
                    out_path=args.out, timeout_s=args.internal_timeout, lambda_orth=args.lambda_orth,
                    fnce_m=args.fnce_m, exactness_config=exactness_config,
                    lambda_anchor=args.lambda_anchor, unblind_override_at=args.unblind_override_at,
                    drift_probe_active=args.drift_probe, drift_probe_n_entities=args.drift_probe_n_entities,
                    drift_probe_n_resamples=args.drift_probe_n_resamples,
                    rev7_pin_derived=rev7_pin_derived, ckpt_dir=args.ckpt_dir)
    result["n_params"] = n_params
    if embed_meta:
        result["embed_meta"] = embed_meta

    if args.save_z:
        model.eval()
        gz = torch.Generator(device=device).manual_seed(args.seed + 20_000)
        bz = sample_batch_rd(cfg, 4, gz, hop_set=(1,), pools=pools, device=device)
        with torch.no_grad():
            S_raw, k_eff_z, v_eff_z, beta_z = model.bind(bz, force_rank_k=None, return_beta=True)
            s_ideal_z = model.effective_ideal_S(k_eff_z, v_eff_z)
            dump = {
                "S_T_raw": S_raw.detach().cpu().tolist(),
                "s_ideal_effective": s_ideal_z.detach().cpu().tolist(),
                "k_eff_items": k_eff_z.detach().cpu().tolist(),
                "v_eff_items": v_eff_z.detach().cpu().tolist(),
                # DELTANET_RD_EXACTNESS_DESIGN.md sec 3.2's beta-dump +
                # sec 3.3's succ/tgt_slot dump -- both additive, both
                # gathered/carried EXACTLY like k_eff_items/v_eff_items
                # (beta_items) or lifted verbatim from the SAME sampled
                # batch (succ/tgt_slot -- already computed for free by
                # sample_batch_rd, zero extra compute). Both auto-detected
                # by analyze_exactness_w0.py's load_run() when present.
                "beta_items": beta_z.detach().cpu().tolist(),
                "succ": bz["succ"].detach().cpu().tolist(),
                "tgt_slot": bz["tgt_slot"].detach().cpu().tolist(),
                "note": "4 eval examples, hop_set=(1,), seed+20000 generator. S_T_raw is UNFORCED "
                        "(force_rank_k=None) regardless of this run's own --force-rank-k, so post-hoc "
                        "truncation at ANY k can be re-derived without a re-run (the eval-truncation "
                        "lesson). s_ideal_effective/k_eff_items/v_eff_items built from THIS model's "
                        "post-conv effective keys/values (section 3.6-equivalent) -- project S_T_raw "
                        "onto s_ideal_effective's own SVD subspace for post-hoc entity-subspace-"
                        "restricted spectral analysis at any truncation level. beta_items (B,K): beta "
                        "gathered at item_pos, exactly like k_eff_items (sec 3.2). succ (B,K)/tgt_slot "
                        "(B,Q): the K-cycle bijection and per-query target slot from this SAME sampled "
                        "batch -- enables analyze_exactness_w0.py's multi-hop Test B extension "
                        "(sec 3.3) without a re-run.",
            }
            if args.force_rank_k is not None:
                S_forced, _, _ = model.bind(bz, force_rank_k=args.force_rank_k)
                dump["S_T_forced"] = S_forced.detach().cpu().tolist()
                dump["force_rank_k"] = args.force_rank_k
        result["Z_dump"] = dump

    summary = {k: v for k, v in result.items() if k not in ("trajectory", "checkpoints", "Z_dump")}
    print("\nRESULT SUMMARY:", json.dumps(summary, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
