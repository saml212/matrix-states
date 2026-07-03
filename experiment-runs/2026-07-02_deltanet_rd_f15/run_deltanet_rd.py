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
import grammar_rd as grd
import model_rd as mrd
from grammar_rd import DeltaNetRDTaskConfig, EntityPools, sample_batch_rd, self_query_tokens
from model_rd import (DeltaNetRDBlock, TruncationError, assert_rank_le, entity_subspace_rank,
                       gram_deviation, pin_buffer_row_, salvage_ratio, target_clause_index,
                       zero_buffer_grad_)
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
    diagnostics computed on THIS pool (R2-8)."""
    model.eval()
    per_hop = {}
    for h in hops_to_eval:
        cos_all, norm_all = [], []
        er_whole, sr_whole, er_ent, sr_ent = [], [], [], []
        key_gd, val_gd, salv, val_salv = [], [], [], []
        align_mean, align_min, align_valid = [], [], []
        n_eval_batches_skipped = 0
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
                      trajectory, checkpoints, n_skipped, t0, timed_out, steps_completed, complete):
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
        # target-index FATAL fix marker (2026-07-03)
        "loss_config": {"objective": "L_cos + lambda*L_nce (section 14.4)",
                         "nce_lambda": NCE_LAMBDA, "nce_temperature": NCE_T,
                         "negatives": "in_episode_K_clauses", "stop_grad": False,
                         "logits_dtype": "fp32", "target_index": "pi_inverse_of_tgt_slot (FATAL-fixed)"},
        "trajectory": trajectory, "checkpoints": checkpoints,
        "wall_s": time.time() - t0, "timed_out": timed_out,
    }
    if checkpoints:
        result["final_step"] = checkpoints[-1]["step"]
    return result


def train(model, cfg, pools, pool_report, device, d_model, d_state, steps=6000, batch_size=128,
          lr=3e-4, seed=0, force_rank_k=None, log_every=200, ckpt_every=2000, out_path=None,
          timeout_s=None):
    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trunc_impl = getattr(model, "trunc_impl", "eigh")
    model.train()

    trajectory, checkpoints = [], []
    n_skipped = 0
    timed_out = False
    steps_completed = 0

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
            # Section 14.4's pre-registered objective: L_cos + lambda*L_nce
            # (TRAINING only -- eval scoring stays absolute cosine, section
            # 14.3). tgt_clause comes from the SAME FATAL-fixed site as
            # forward()'s targets (target_clause_index).
            l_cos = cosine_loss(pred, targets)
            tgt_clause = target_clause_index(b["succ"], b["tgt_slot"])
            l_nce = nce_loss(pred, v_eff_items, tgt_clause)
            loss = l_cos + NCE_LAMBDA * l_nce
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
                opt.step()
                pin_buffer_row_(model.embed, model.buffer_id)      # re-assert exact zero every step
            else:
                n_skipped += 1

        if not step_failed and (step % log_every == 0 or step == 1):
            with torch.no_grad():
                er = effective_rank(S_T).mean().item()
            trajectory.append({"step": step, "loss": loss.item(), "loss_cos": l_cos.item(),
                                "loss_nce": l_nce.item(), "eff_rank_trainbatch": er,
                                "skip_rate_so_far": n_skipped / step})
            extra = f"  [skip_rate {n_skipped/step:.4%}]" if n_skipped else ""
            print(f"  step {step:6d}  loss {loss.item():.4f} (cos {l_cos.item():.4f} "
                  f"nce {l_nce.item():.4f})  eff_rank {er:.3f}{extra}", flush=True)

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
            checkpoints.append(res)
            partial = _assemble_result(cfg, steps, seed, force_rank_k, trunc_impl, pool_report,
                                        d_model, d_state, trajectory, checkpoints, n_skipped, t0,
                                        timed_out=False, steps_completed=steps_completed, complete=False)
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

    return _assemble_result(cfg, steps, seed, force_rank_k, trunc_impl, pool_report, d_model, d_state,
                             trajectory, checkpoints, n_skipped, t0, timed_out, steps_completed,
                             complete=(not timed_out and steps_completed >= steps))


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

    cfg = DeltaNetRDTaskConfig(K=args.K, conv_size=args.conv_size, H_train=tuple(args.h_train),
                                H_test=tuple(args.h_test), H_extra=tuple(args.h_extra))
    model = DeltaNetRDBlock(pools.vocab_size_total, d_model=args.d_model, d_state=args.d_state,
                             conv_size=cfg.conv_size, buffer_id=pools.buffer_id,
                             trunc_impl=args.trunc_impl).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"K={args.K} conv_size={args.conv_size} T_bind={cfg.T_bind} d_model={args.d_model} "
          f"d_state={args.d_state} H_train={cfg.H_train} H_test={cfg.H_test} H_extra={cfg.H_extra} "
          f"force_rank_k={args.force_rank_k} trunc_impl={args.trunc_impl} params={n_params} "
          f"device={device}", flush=True)

    result = train(model, cfg, pools, pool_report, device, d_model=args.d_model, d_state=args.d_state,
                    steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                    force_rank_k=args.force_rank_k, log_every=args.log_every, ckpt_every=args.ckpt_every,
                    out_path=args.out, timeout_s=args.internal_timeout)
    result["n_params"] = n_params

    if args.save_z:
        model.eval()
        gz = torch.Generator(device=device).manual_seed(args.seed + 20_000)
        bz = sample_batch_rd(cfg, 4, gz, hop_set=(1,), pools=pools, device=device)
        with torch.no_grad():
            S_raw, k_eff_z, v_eff_z = model.bind(bz, force_rank_k=None)
            s_ideal_z = model.effective_ideal_S(k_eff_z, v_eff_z)
            dump = {
                "S_T_raw": S_raw.detach().cpu().tolist(),
                "s_ideal_effective": s_ideal_z.detach().cpu().tolist(),
                "k_eff_items": k_eff_z.detach().cpu().tolist(),
                "v_eff_items": v_eff_z.detach().cpu().tolist(),
                "note": "4 eval examples, hop_set=(1,), seed+20000 generator. S_T_raw is UNFORCED "
                        "(force_rank_k=None) regardless of this run's own --force-rank-k, so post-hoc "
                        "truncation at ANY k can be re-derived without a re-run (the eval-truncation "
                        "lesson). s_ideal_effective/k_eff_items/v_eff_items built from THIS model's "
                        "post-conv effective keys/values (section 3.6-equivalent) -- project S_T_raw "
                        "onto s_ideal_effective's own SVD subspace for post-hoc entity-subspace-"
                        "restricted spectral analysis at any truncation level.",
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
