"""run_deltanet.py -- DeltaNet causal-rank training entry point. See
DELTANET_CAUSAL_RANK_DESIGN.md (frozen, revision 2.1), especially sections
3.5 (two-kernel split), 4 (the provable bound), 5 (compositional transfer),
6.2-6.3 (controls + primary metric), 6.5 (operational-harness build
requirements, F17).

Trains ONE (d, K, force_rank_k, seed, steps) cell and writes a single result
JSON with Stage-0-style instrumentation (F17):
  trajectory   dense (<=200-step default) loss / skip-rate / grad-norm log
  checkpoints  <=2000-step default full eval: PRIMARY = entity-subspace-
               restricted rank of S_T (design section 3.6/6.3); SECONDARY =
               whole-state effective/stable rank; per-hop recovered_frac@tau
               (H_train/H_test/H_extra, section 5.4); per-step skip RATE
               (F14, not a raw count); the effective-key Gram deviation
               diagnostic (F4); per-hop prediction norms (F3, cosine is
               blind to beta-driven magnitude decay under composition).
Both written INCREMENTALLY to --out after every checkpoint, "complete":
false on every incremental dump; ONLY the final post-training write in
main() carries "complete": true -- run_deltanet_sweep.is_done() requires
this sentinel (crash-safety, mirrors run_stage0.py's FATAL-1 fix exactly).

Run the smoke gate FIRST (seconds): python run_deltanet.py --smoke
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
import task_dn as tdn
import deltanet_core as dc
import model_dn as mdn
from model_dn import (
    DeltaNetGateModel, entity_subspace_rank, assert_rank_le,
    DeltaNetMultiHeadGateModel, per_head_entity_subspace_rank, assert_rank_le_per_head,
)
from rank_utils import truncate_to_rank, effective_rank, stable_rank

# Build-time interpretive decision (F11 / Reserve-wave multi-head probe,
# flagged for audit): per-head state dim == d (total capacity scales with
# H), NOT the standard d/H transformer split -- see
# DeltaNetMultiHeadGateModel's docstring in model_dn.py for the full
# reasoning. Echoed into every H>1 run's result JSON via this constant so a
# reader never has to cross-reference the model file to know which
# convention produced a given result.
MULTIHEAD_DHEAD_CONVENTION = "full_d_per_head (capacity scales with H, not the standard d/H split)"

TAUS = (0.9, 0.95, 0.99)


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


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


def _gram_deviation(k_eff: torch.Tensor) -> float:
    """F4 diagnostic: ||K_eff^T K_eff - I||_F on the effective (post-W_k)
    BIND-item keys -- separates rank-collapse failures from
    keys-drifted-from-orthonormality noise. k_eff: (B, K, d)."""
    gram = k_eff @ k_eff.transpose(-1, -2)                # (B, K, K)
    Keff = k_eff.shape[1]
    eye = torch.eye(Keff, device=k_eff.device).expand_as(gram)
    return (gram - eye).norm(dim=(-2, -1)).mean().item()


@torch.no_grad()
def evaluate_at_hops(model: DeltaNetGateModel, cfg: tdn.DeltaNetTaskConfig, gen, device,
                      hops_to_eval, force_rank_k=None, n_batches=4, batch_size=128):
    """Task-E-style per-hop evaluation (run_task_e.py::evaluate_at_hops,
    adapted): SEPARATE stats at each hop depth, on the SAME batches also
    scored against the architecture-native ideal (C7-equivalent, built from
    THIS model's current effective keys, section 3.6)."""
    model.eval()
    per_hop = {}
    for h in hops_to_eval:
        cos_all, cos_ideal_all, norm_all = [], [], []
        er_whole, sr_whole, er_ent, sr_ent, gram_dev = [], [], [], [], []
        for _ in range(n_batches):
            # Task E convention (task_e.py docstring): train() disables the
            # per-batch injectivity check after step 1 (sync-barrier cost
            # over tens of thousands of steps), but EVAL keeps it on.
            b = tdn.sample_batch(cfg, batch_size, gen, hop_set=(h,), device=device,
                                  assert_injective=True)
            pred, S_T = model(b, force_rank_k=force_rank_k)
            cos_all.append(F.cosine_similarity(pred, b["targets"], dim=-1))
            norm_all.append(pred.norm(dim=-1).reshape(-1))

            s_ideal_eff = model.effective_ideal_S(b["keys"], b["values"])
            pred_ideal = dc.apply_state_power(s_ideal_eff, model.effective_key(b["query_keys"]),
                                               b["hops"])
            cos_ideal_all.append(F.cosine_similarity(pred_ideal, b["targets"], dim=-1))

            er_whole.append(effective_rank(S_T))
            sr_whole.append(stable_rank(S_T))
            er_e, sr_e = entity_subspace_rank(S_T, s_ideal_eff, cfg.K)
            er_ent.append(er_e)
            sr_ent.append(sr_e)
            k_eff_items = model.effective_key(b["keys"])
            gram_dev.append(_gram_deviation(k_eff_items))

        cos = torch.cat([c.reshape(-1) for c in cos_all])
        cos_ideal = torch.cat([c.reshape(-1) for c in cos_ideal_all])
        entry = {"h": h, "effective_hop": h % cfg.K}
        entry.update(_recovery_stats(cos))
        entry.update(_recovery_stats(cos_ideal, prefix="ideal_"))
        entry["pred_norm_mean"] = torch.cat(norm_all).mean().item()          # F3 diagnostic
        entry["effective_rank_whole_mean"] = torch.cat(er_whole).mean().item()
        entry["stable_rank_whole_mean"] = torch.cat(sr_whole).mean().item()
        entry["entity_subspace_effective_rank_mean"] = torch.cat(er_ent).mean().item()  # PRIMARY
        entry["entity_subspace_stable_rank_mean"] = torch.cat(sr_ent).mean().item()
        entry["key_gram_deviation_mean"] = sum(gram_dev) / len(gram_dev)     # F4 diagnostic
        per_hop[h] = entry
    model.train()
    return per_hop


@torch.no_grad()
def evaluate_at_hops_mh(model: DeltaNetMultiHeadGateModel, cfg: tdn.DeltaNetTaskConfig, gen, device,
                         hops_to_eval, force_rank_k=None, n_batches=4, batch_size=128):
    """Multi-head sibling of evaluate_at_hops (F11 / Reserve-wave probe;
    ADDITIVE, does not touch evaluate_at_hops or the H=1 path it serves).
    Same per-hop structure and overall (SUMMED-readout) recovery stats as
    the H=1 function, PLUS the per-head entity-subspace-restricted rank
    (the PRIMARY F11 metric) and its cross-head SUM (the direct measurement
    of the design's joint bound Sigma_head rank(S_T^(head)) >= K)."""
    H = model.H
    model.eval()
    per_hop = {}
    for h in hops_to_eval:
        cos_all, norm_all = [], []
        er_whole_heads = [[] for _ in range(H)]
        sr_whole_heads = [[] for _ in range(H)]
        er_ent_heads = [[] for _ in range(H)]
        sr_ent_heads = [[] for _ in range(H)]
        gram_dev_heads = [[] for _ in range(H)]
        joint_rank_sum = []
        for _ in range(n_batches):
            b = tdn.sample_batch(cfg, batch_size, gen, hop_set=(h,), device=device,
                                  assert_injective=True)
            pred, S_T = model(b, force_rank_k=force_rank_k)              # S_T: (B,H,d,d)
            cos_all.append(F.cosine_similarity(pred, b["targets"], dim=-1))
            norm_all.append(pred.norm(dim=-1).reshape(-1))

            s_ideal_heads = model.effective_ideal_S(b["keys"], b["values"])   # (B,H,d,d)
            ers, srs = per_head_entity_subspace_rank(S_T, s_ideal_heads, cfg.K)
            per_example_sum = torch.zeros(S_T.shape[0], device=device)
            k_eff_items = model.effective_key(b["keys"])                  # (B,K,H,d)
            for hh in range(H):
                er_whole_heads[hh].append(effective_rank(S_T[:, hh]))
                sr_whole_heads[hh].append(stable_rank(S_T[:, hh]))
                er_ent_heads[hh].append(ers[hh])
                sr_ent_heads[hh].append(srs[hh])
                per_example_sum = per_example_sum + ers[hh]
                gram_dev_heads[hh].append(_gram_deviation(k_eff_items[:, :, hh, :]))
            joint_rank_sum.append(per_example_sum)

        cos = torch.cat([c.reshape(-1) for c in cos_all])
        entry = {"h": h, "effective_hop": h % cfg.K}
        entry.update(_recovery_stats(cos))
        entry["pred_norm_mean"] = torch.cat(norm_all).mean().item()
        entry["per_head_entity_subspace_effective_rank_mean"] = [
            torch.cat(er_ent_heads[hh]).mean().item() for hh in range(H)]
        entry["per_head_entity_subspace_stable_rank_mean"] = [
            torch.cat(sr_ent_heads[hh]).mean().item() for hh in range(H)]
        entry["per_head_effective_rank_whole_mean"] = [
            torch.cat(er_whole_heads[hh]).mean().item() for hh in range(H)]
        entry["per_head_stable_rank_whole_mean"] = [
            torch.cat(sr_whole_heads[hh]).mean().item() for hh in range(H)]
        entry["per_head_key_gram_deviation_mean"] = [
            sum(gram_dev_heads[hh]) / len(gram_dev_heads[hh]) for hh in range(H)]
        # PRIMARY F11 metric: the joint bound Sigma_head rank(S_T^(head)) --
        # summed PER EXAMPLE first (matching the design's own "Sigma_head"
        # phrasing), then averaged over the eval batch.
        entry["joint_entity_subspace_effective_rank_sum_mean"] = torch.cat(joint_rank_sum).mean().item()
        per_hop[h] = entry
    model.train()
    return per_hop


def train_mh(model: DeltaNetMultiHeadGateModel, cfg, device, H, steps=6000, batch_size=128, lr=3e-4,
             seed=0, force_rank_k=None, log_every=200, ckpt_every=2000, out_path=None, timeout_s=None):
    """Multi-head sibling of train() (F11 / Reserve-wave probe; ADDITIVE --
    does not modify train() or anything the already-completed H=1
    Wave -1/0/A/Bprobe results depend on). Identical step logic to train(),
    dispatched to evaluate_at_hops_mh at checkpoints; result assembly
    reuses _assemble_result verbatim (H=1-agnostic) with an "H" field and
    the per-head convention note added afterward."""
    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trunc_impl = getattr(model, "trunc_impl", "eigh")
    model.train()

    trajectory, checkpoints = [], []
    n_skipped = 0
    timed_out = False
    steps_completed = 0

    for step in range(1, steps + 1):
        steps_completed = step
        b = tdn.sample_batch(cfg, batch_size, gen, hop_set=cfg.H_train, device=device,
                              assert_injective=(step == 1))
        pred, S_T = model(b, force_rank_k=force_rank_k)
        loss = cosine_loss(pred, b["targets"])
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                er_heads = [effective_rank(S_T[:, hh]).mean().item() for hh in range(H)]
            trajectory.append({"step": step, "loss": loss.item(),
                                "per_head_eff_rank_trainbatch": er_heads,
                                "skip_rate_so_far": n_skipped / step})
            extra = f"  [skip_rate {n_skipped/step:.4%}]" if n_skipped else ""
            print(f"  step {step:6d}  loss {loss.item():.4f}  per_head_eff_rank {er_heads}{extra}", flush=True)

        if step % ckpt_every == 0 or step == steps:
            eval_gen = torch.Generator(device=device).manual_seed(seed + 10_000 + step)
            m2 = evaluate_at_hops_mh(model, cfg, eval_gen, device, hops_to_eval=cfg.H_train,
                                      force_rank_k=force_rank_k)
            m3 = evaluate_at_hops_mh(model, cfg, eval_gen, device,
                                      hops_to_eval=(*cfg.H_test, *cfg.H_extra),
                                      force_rank_k=force_rank_k)
            res = {"step": step, "M2_in_distribution": m2, "M3_held_out": m3}
            checkpoints.append(res)
            partial = _assemble_result(cfg, steps, seed, force_rank_k, trunc_impl,
                                        trajectory, checkpoints,
                                        n_skipped, t0, timed_out=False,
                                        steps_completed=steps_completed, complete=False)
            partial["H"] = H
            partial["multihead_dhead_convention"] = MULTIHEAD_DHEAD_CONVENTION
            _dump(out_path, partial)
            primary_h1 = m2[cfg.H_train[0]]
            print(f"  [checkpoint step {step}] h={cfg.H_train[0]}: "
                  f"recovered_frac@0.9={primary_h1['recovered_frac@0.9']:.3f}  "
                  f"joint_entity_subspace_rank_sum="
                  f"{primary_h1['joint_entity_subspace_effective_rank_sum_mean']:.3f} "
                  f"(K={cfg.K})  per_head={primary_h1['per_head_entity_subspace_effective_rank_mean']}",
                  flush=True)

        if timeout_s is not None and time.time() - t0 > timeout_s:
            print(f"  internal timeout ({timeout_s}s) reached at step {step}; stopping early", flush=True)
            timed_out = True
            break

    result = _assemble_result(cfg, steps, seed, force_rank_k, trunc_impl, trajectory, checkpoints,
                               n_skipped, t0, timed_out, steps_completed,
                               complete=(not timed_out and steps_completed >= steps))
    result["H"] = H
    result["multihead_dhead_convention"] = MULTIHEAD_DHEAD_CONVENTION
    return result


def _dump(path, obj):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"  (incremental write failed, non-fatal: {e!r})", flush=True)


def _assemble_result(cfg, steps, seed, force_rank_k, trunc_impl, trajectory, checkpoints,
                      n_skipped, t0, timed_out, steps_completed, complete):
    result = {
        "K": cfg.K, "d": cfg.d, "conv_size": cfg.conv_size, "orthogonal": cfg.orthogonal,
        "H_train": list(cfg.H_train), "H_test": list(cfg.H_test), "H_extra": list(cfg.H_extra),
        "force_rank_k": force_rank_k, "trunc_impl": trunc_impl, "seed": seed, "steps": steps,
        "steps_completed": steps_completed,
        "complete": complete,                       # crash-safety sentinel (F17)
        "n_skipped_steps": n_skipped,
        "skip_rate": (n_skipped / steps_completed) if steps_completed > 0 else 0.0,   # F14: RATE
        "trajectory": trajectory, "checkpoints": checkpoints,
        "wall_s": time.time() - t0, "timed_out": timed_out,
    }
    if checkpoints:
        result["final_step"] = checkpoints[-1]["step"]
    return result


def train(model, cfg, device, steps=6000, batch_size=128, lr=3e-4, seed=0,
          force_rank_k=None, log_every=200, ckpt_every=2000, out_path=None,
          timeout_s=None):
    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trunc_impl = getattr(model, "trunc_impl", "eigh")   # recorded per-run in the result JSON
    model.train()

    trajectory, checkpoints = [], []
    n_skipped = 0
    timed_out = False
    steps_completed = 0

    for step in range(1, steps + 1):
        steps_completed = step
        # C6 injectivity is structural (identical for every batch of this
        # cfg) -- assert on step 1 only, matching run_task_e.py's convention
        # (avoids a GPU->CPU sync barrier on every one of ~10-40K steps).
        b = tdn.sample_batch(cfg, batch_size, gen, hop_set=cfg.H_train, device=device,
                              assert_injective=(step == 1))
        pred, S_T = model(b, force_rank_k=force_rank_k)
        loss = cosine_loss(pred, b["targets"])
        opt.zero_grad()
        loss.backward()
        # Gradient hygiene (Task D/E's policy, reused verbatim): skip a
        # non-finite step rather than kill the run. n_skipped -> skip_rate
        # (F14) is the reported quantity, not the raw count.
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                er = effective_rank(S_T).mean().item()
            trajectory.append({"step": step, "loss": loss.item(),
                                "eff_rank_trainbatch": er,
                                "skip_rate_so_far": n_skipped / step})
            extra = f"  [skip_rate {n_skipped/step:.4%}]" if n_skipped else ""
            print(f"  step {step:6d}  loss {loss.item():.4f}  eff_rank {er:.3f}{extra}", flush=True)

        if step % ckpt_every == 0 or step == steps:
            eval_gen = torch.Generator(device=device).manual_seed(seed + 10_000 + step)
            m2 = evaluate_at_hops(model, cfg, eval_gen, device, hops_to_eval=cfg.H_train,
                                   force_rank_k=force_rank_k)
            m3 = evaluate_at_hops(model, cfg, eval_gen, device,
                                   hops_to_eval=(*cfg.H_test, *cfg.H_extra),
                                   force_rank_k=force_rank_k)
            res = {"step": step, "M2_in_distribution": m2, "M3_held_out": m3}
            checkpoints.append(res)
            partial = _assemble_result(cfg, steps, seed, force_rank_k, trunc_impl,
                                        trajectory, checkpoints,
                                        n_skipped, t0, timed_out=False,
                                        steps_completed=steps_completed, complete=False)
            _dump(out_path, partial)
            primary_h1 = m2[cfg.H_train[0]]
            print(f"  [checkpoint step {step}] h={cfg.H_train[0]}: "
                  f"recovered_frac@0.9={primary_h1['recovered_frac@0.9']:.3f}  "
                  f"entity_subspace_eff_rank={primary_h1['entity_subspace_effective_rank_mean']:.3f}  "
                  f"whole_eff_rank={primary_h1['effective_rank_whole_mean']:.3f}", flush=True)

        if timeout_s is not None and time.time() - t0 > timeout_s:
            print(f"  internal timeout ({timeout_s}s) reached at step {step}; stopping early", flush=True)
            timed_out = True
            break

    return _assemble_result(cfg, steps, seed, force_rank_k, trunc_impl, trajectory, checkpoints,
                             n_skipped, t0, timed_out, steps_completed,
                             complete=(not timed_out and steps_completed >= steps))


# ---------------------------------------------------------------------------
# Smoke gate (run FIRST on cluster; no real training)
# ---------------------------------------------------------------------------

def smoke(device):
    print("=" * 60 + "\n  DELTANET CAUSAL-RANK SMOKE GATE\n" + "=" * 60)

    print("\n[1] task_dn self-test (streamed BIND/QUERY grammar, zero-buffer embeddings, "
          "beta-mask exactness, injectivity guard + its negative test, s_ideal composition)")
    tdn._self_test()

    print("\n[2] deltanet_core self-test (architecture-native ideal via the ACTUAL "
          "recurrence code path, round-trip, gradcheck through BOTH kernel hooks + "
          "negative tests, apply_state_power)")
    dc._self_test()

    print("\n[3] rank_utils sanity + degenerate-spectrum NaN/Inf-free backward "
          "(reused verbatim from chapter2/rank_utils.py)")
    Z = torch.randn(4, 16, 16, device=device)
    for k in (1, 2, 4, 8):
        er = effective_rank(truncate_to_rank(Z, k)).mean().item()
        assert er <= k + 1e-2, f"truncate_to_rank({k}) gave eff rank {er:.3f} > {k}"
    torch.manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(4, 4, device=device))
    V, _ = torch.linalg.qr(torch.randn(4, 4, device=device))
    Zdeg = (U @ torch.diag(torch.tensor([3., 1., 1., 0.], device=device)) @ V.T
            ).unsqueeze(0).requires_grad_(True)
    truncate_to_rank(Zdeg, 2).sum().backward()
    assert Zdeg.grad is not None and not torch.isnan(Zdeg.grad).any() \
        and not torch.isinf(Zdeg.grad).any(), "NaN/Inf grad on degenerate spectrum"
    print("  OK")

    print("\n[4] model_dn self-test (forward/backward, C15 + negative test, "
          "C_composition-purity, two-sided blank-out incl. sensitivity control, "
          "model-level round trip, entity_subspace_rank identity-regime sanity)")
    mdn._self_test()

    print("\n[5] end-to-end mini training loop (a handful of REAL SGD steps, not just init) "
          "stays finite through held-out AND far-out (H_extra) hops")
    cfg = tdn.DeltaNetTaskConfig(d=16, K=8, conv_size=4,
                                  H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    model = DeltaNetGateModel(d=16).to(device)
    result = train(model, cfg, device, steps=20, batch_size=32, log_every=10, ckpt_every=20)
    assert result["steps_completed"] == 20 and result["complete"] is True
    assert result["checkpoints"], "no checkpoint was written"
    final = result["checkpoints"][-1]
    for h_group in (final["M2_in_distribution"], final["M3_held_out"]):
        for h, entry in h_group.items():
            assert entry["mean_cos"] == entry["mean_cos"], f"NaN mean_cos at h={h}"    # NaN != NaN
            assert abs(entry["pred_norm_mean"]) < 1e6, f"exploded pred norm at h={h}"
    print(f"  20-step mini run: complete={result['complete']}, "
          f"skip_rate={result['skip_rate']:.4%}, all per-hop stats finite "
          f"(H_train={cfg.H_train}, H_test={cfg.H_test}, H_extra={cfg.H_extra})")

    print("\n[6] force_rank_k end-to-end, BOTH --trunc-impl variants (audit FINDING 1): "
          "mini run with rank forced trains without crashing, entity-subspace rank "
          "constrained, trunc_impl recorded in the result JSON")
    for impl in ("eigh", "svd_lowrank"):
        model_fr = DeltaNetGateModel(d=16, trunc_impl=impl).to(device)
        result_fr = train(model_fr, cfg, device, steps=20, batch_size=32, log_every=10,
                           ckpt_every=20, force_rank_k=4)
        assert result_fr["complete"] is True
        assert result_fr["trunc_impl"] == impl, \
            f"result JSON records trunc_impl={result_fr['trunc_impl']!r}, expected {impl!r}"
        final_fr = result_fr["checkpoints"][-1]["M2_in_distribution"][cfg.H_train[0]]
        assert final_fr["entity_subspace_effective_rank_mean"] <= 4 + 1e-2, \
            f"[{impl}] force_rank_k=4 did not constrain the entity-subspace rank as expected"
        print(f"  [{impl}] 20-step force_rank_k=4 mini run: complete={result_fr['complete']}, "
              f"skip_rate={result_fr['skip_rate']:.4%}, "
              f"entity_subspace_eff_rank={final_fr['entity_subspace_effective_rank_mean']:.3f} <= 4")

    print("\n[7] F11 / Reserve-wave multi-head (H=2) end-to-end mini training loop: "
          "REAL SGD steps stay finite, per-head + joint-bound metrics land in the "
          "checkpoint JSON, complete sentinel + H field are set correctly")
    for H in (2, 4):
        model_mh = DeltaNetMultiHeadGateModel(d=16, H=H).to(device)
        result_mh = train_mh(model_mh, cfg, device, H=H, steps=20, batch_size=32,
                              log_every=10, ckpt_every=20)
        assert result_mh["steps_completed"] == 20 and result_mh["complete"] is True
        assert result_mh["H"] == H
        assert result_mh["checkpoints"], "no checkpoint was written (multi-head)"
        final_mh = result_mh["checkpoints"][-1]
        for h_group in (final_mh["M2_in_distribution"], final_mh["M3_held_out"]):
            for h, entry in h_group.items():
                assert entry["mean_cos"] == entry["mean_cos"], f"NaN mean_cos at h={h} (H={H})"
                assert len(entry["per_head_entity_subspace_effective_rank_mean"]) == H
                js = entry["joint_entity_subspace_effective_rank_sum_mean"]
                assert js == js and js >= 0, f"bad joint rank sum at h={h} (H={H}): {js}"
        print(f"  H={H}: 20-step mini run complete={result_mh['complete']}, "
              f"skip_rate={result_mh['skip_rate']:.4%}, per-head + joint-bound metrics finite")

    print("\n" + "=" * 60 + "\n  ALL DELTANET SMOKE CHECKS PASSED\n" + "=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--K", type=int, default=16,
                     help="bindings per sample. KNOWN LIMITATION at this default (K=16, "
                          "audit round-1 MINOR-4): H_extra's h=21 has 21 %% 16 == 5, which "
                          "collides with H_test hop h=5's residue -- the config guard "
                          "(mirroring task_e.py's) only rejects collisions with TRAIN "
                          "residues, so h=21 loads as a legal but NON-NOVEL probe at K=16: "
                          "it re-measures effective composition distance 5 (already covered "
                          "by h=5) at 4x the raw iteration count. Read K=16 h=21 results "
                          "via the reported effective_hop field as depth-amplification "
                          "data (more compositions at the same residue), NEVER as a new "
                          "held-out residue.")
    ap.add_argument("--conv-size", type=int, default=4)
    ap.add_argument("--h-train", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--h-test", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--h-extra", type=int, nargs="+", default=[7, 21])
    ap.add_argument("--orthogonal", dest="orthogonal", action="store_true", default=True)
    ap.add_argument("--gaussian", dest="orthogonal", action="store_false")
    ap.add_argument("--H", type=int, default=1,
                     help="number of DeltaNet heads (F11 / Reserve-wave multi-head probe, "
                          "DELTANET_CAUSAL_RANK_DESIGN.md section 6.4/8). H=1 (default) is "
                          "the UNCHANGED primary gate (DeltaNetGateModel, C11) -- byte-for-"
                          "byte identical code path to every already-completed Wave -1/0/A/"
                          "Bprobe result. H>1 dispatches to DeltaNetMultiHeadGateModel/"
                          "train_mh (model_dn.py's docstring: per-head state dim == d, "
                          "capacity scales with H -- see MULTIHEAD_DHEAD_CONVENTION above).")
    ap.add_argument("--force-rank-k", type=int, default=None,
                     help="H=1: force_rank_k for the primary gate. H>1: applied UNIFORMLY to "
                          "every head (per-head-list force-rank bounds are supported at the "
                          "train_mh()/model.bind() Python API level -- see model_dn.py's "
                          "DeltaNetMultiHeadGateModel.bind docstring -- but not exposed as a "
                          "separate CLI flag; this build's Reserve-wave launch manifest is "
                          "unconstrained-only, force_rank_k=None, at every H).")
    ap.add_argument("--trunc-impl", choices=["eigh", "svd_lowrank"], default="eigh",
                     help="rank-truncation implementation for the force-rank arm (audit "
                          "round-1 FINDING 1). 'eigh' = the design-default "
                          "rank_utils.truncate_to_rank; 'svd_lowrank' = the randomized-SVD "
                          "stability fallback (see model_dn.truncate_to_rank_svd_lowrank's "
                          "docstring). No effect when --force-rank-k is not given. Recorded "
                          "in the result JSON; the Wave -1 / F13 decision selects which "
                          "impl downstream waves use.")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--internal-timeout", type=float, default=None)
    ap.add_argument("--save-z", action="store_true",
                     help="embed a small post-training S_T + s_ideal dump in the output "
                          "JSON (a handful of eval examples at h=1) for post-hoc "
                          "entity-subspace-restricted spectral analysis, mirroring "
                          "run_task_e.py's --save-z convention")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.smoke:
        smoke(device)
        return

    if args.ckpt_every > 2000:
        print(f"WARNING: --ckpt-every={args.ckpt_every} > 2000: violates the design's "
              f"section 6.5 build requirement.", flush=True)
    if args.K > args.d:
        print(f"WARNING: K={args.K} > d={args.d}: no exact solution (lossy K>d regime).", flush=True)

    if args.H < 1:
        print(f"ERROR: --H={args.H} must be >= 1.", file=sys.stderr)
        sys.exit(1)

    cfg = tdn.DeltaNetTaskConfig(d=args.d, K=args.K, conv_size=args.conv_size,
                                  H_train=tuple(args.h_train), H_test=tuple(args.h_test),
                                  H_extra=tuple(args.h_extra), orthogonal=args.orthogonal)
    if args.H == 1:
        # UNCHANGED primary-gate path -- byte-for-byte identical to every
        # already-completed Wave -1/0/A/Bprobe run (F11's H=1 default).
        model = DeltaNetGateModel(d=args.d, trunc_impl=args.trunc_impl).to(device)
    else:
        model = mdn.DeltaNetMultiHeadGateModel(d=args.d, H=args.H, trunc_impl=args.trunc_impl).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"H={args.H} d={args.d} K={args.K} conv_size={args.conv_size} T_bind={cfg.T_bind} "
          f"H_train={cfg.H_train} H_test={cfg.H_test} H_extra={cfg.H_extra} "
          f"orthogonal={args.orthogonal} force_rank_k={args.force_rank_k} "
          f"trunc_impl={args.trunc_impl} params={n_params} device={device}"
          + (f" multihead_dhead_convention={MULTIHEAD_DHEAD_CONVENTION}" if args.H > 1 else ""),
          flush=True)

    if args.H == 1:
        result = train(model, cfg, device, steps=args.steps, batch_size=args.batch_size,
                        lr=args.lr, seed=args.seed, force_rank_k=args.force_rank_k,
                        log_every=args.log_every, ckpt_every=args.ckpt_every,
                        out_path=args.out, timeout_s=args.internal_timeout)
    else:
        result = train_mh(model, cfg, device, H=args.H, steps=args.steps, batch_size=args.batch_size,
                           lr=args.lr, seed=args.seed, force_rank_k=args.force_rank_k,
                           log_every=args.log_every, ckpt_every=args.ckpt_every,
                           out_path=args.out, timeout_s=args.internal_timeout)
    result["n_params"] = n_params

    if args.save_z:
        model.eval()
        gz = torch.Generator(device=device).manual_seed(args.seed + 20_000)
        bz = tdn.sample_batch(cfg, 4, gz, hop_set=(1,), device=device)
        with torch.no_grad():
            _, S_dump = model(bz, force_rank_k=args.force_rank_k)
            s_ideal_dump = model.effective_ideal_S(bz["keys"], bz["values"])
        result["Z_dump"] = {
            "S_T": S_dump.detach().cpu().tolist(),
            "s_ideal_effective": s_ideal_dump.detach().cpu().tolist(),
            "note": ("4 eval examples, hop_set=(1,), seed+20000 generator; s_ideal_effective "
                     "built from THIS model's post-W_k effective keys (design section 3.6), "
                     "not raw generator keys -- project S_T onto s_ideal_effective's own SVD "
                     "subspace for post-hoc entity-subspace-restricted analysis."
                     + (" H>1: S_T/s_ideal_effective carry an explicit HEAD axis "
                        "(shape (...,H,d,d)) -- per-head eval-time truncation "
                        "(rank_utils.truncate_to_rank applied to S_T[...,h,:,:] "
                        "independently per head) can be reconstructed post-hoc from this "
                        "dump exactly as analyze_eval_truncation.py does for H=1, looped "
                        "over the head axis." if args.H > 1 else "")),
        }

    summary = {k: v for k, v in result.items() if k not in ("trajectory", "checkpoints")}
    print("\nRESULT SUMMARY:", json.dumps(summary, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
