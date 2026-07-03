"""run_stageg_he.py -- Stage G Wave C (H_e) task-swap training entry point.
See STAGE_G_DESIGN.md section 4 item 9 ("Task-swap control on a
composition-heavy corpus") and section 6.4 item 3 (Wave C, gated -- the
budget line this build's manifest instantiates).

Reuses the Stage G models VERBATIM (models.py's MatrixModelSpec /
VectorModelSpec -- the exact matrix-vs-vector pair Wave A/B diagnose, no
new architecture code) with a NEW task (task_he.py's discrete K-cycle
composition corpus) and a NEW, bespoke training loop (this file) -- Stage
G's own train_stageg.py/data.py are NOT reused because they are wired
around a STATIC byte-CORPUS-FILE with a single contiguous train/val split
(data.py's ByteDataset), which cannot express "train hops come from
H_train, eval hops come from a DIFFERENT held-out set" -- exactly the
methodology H_e's Wave C needs (task_dn.py/task_e.py's own on-the-fly
per-batch hop_set convention, reused here). Deliberately mirrors those two
files' conventions (fresh-batch-per-step, per-hop evaluate function,
complete-sentinel result JSON) rather than reinventing them.

Score = COMPOSITION RECOVERY (argmax-decoded exact-match accuracy at each
query's answer-byte position), NOT BPB (per the task brief) -- BPB is still
logged (the model also predicts the R/Q line format, so BPB is a real,
informative number) but is explicitly secondary; composition_accuracy is
the metric the manifest's 6 cells exist to compare. NOTE h=1 is a LOOKUP
metric, not composition (the copy leak -- task_he.py module docstring,
audit FIX-A); read h>=2 for composition.

Audit-round fixes carried in this file:
  FIX-B  every result JSON carries a `capacity_confound` block (n_params,
         matrix-baseline params at the identical config, ratio) + a
         launch-time printed warning when the ratio leaves the +/-5% band
         -- h_b_factored_r4 runs at ~2.69x the matrix baseline
         (STAGE_G_DESIGN.md section 14.4's capacity-control lesson: this
         exact cell's Wave-B win partially collapsed against a
         capacity-matched vector control).
  FIX-C  --answer-loss-weight W (default 5.0): cross-entropy upweighted at
         the answer-byte targets (see lm_loss). W=1.0 recovers the
         pre-FIX-C objective BITWISE (verified in the smoke gate).

Run the smoke gate FIRST (seconds on CPU): python run_stageg_he.py --smoke
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # pod-safe imports
from models import MatrixModelSpec, VectorModelSpec
from flops import analytic_flops_for_cell
import task_he as the

LOG2 = math.log(2)


def _autocast_ctx(device):
    """Verbatim copy of train_stageg.py's _autocast_ctx (bf16 on CUDA,
    no-op elsewhere) -- reproduced here rather than imported so this file
    stays independently runnable per this codebase's self-contained-script
    convention (task_dn.py/rank_utils.py's own stated reasoning)."""
    if isinstance(device, torch.device):
        device = device.type
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def build_model(family: str, variant: str, mat_dim: int, max_len: int, vocab_size: int,
                 n_iterations: int):
    """Verbatim logic from train_stageg.py::build_model (matched-params
    defaults UNCHANGED: n_embd=80 for the vector family, the measured param
    match to the d=32 matrix baseline -- audit MAJOR-2, models.py's
    VectorReferenceModel docstring) -- so this task reuses the ALREADY-
    VERIFIED param parity between MatrixModelSpec/VectorModelSpec defaults
    instead of re-deriving a new match for a smaller max_len/vocab."""
    if family == "matrix":
        return MatrixModelSpec(mat_dim=mat_dim, n_heads=8, n_iterations=n_iterations,
                                max_len=max_len, vocab_size=vocab_size, variant=variant).build()
    elif family == "vector":
        return VectorModelSpec(n_embd=80, n_head=4, n_layer=2, n_loops=n_iterations,
                                intermediate_dim=128, max_len=max_len, vocab_size=vocab_size,
                                variant=variant).build()
    raise ValueError(family)


def _fwd_kwargs(family: str, n_iterations: int) -> dict:
    return {"n_iterations": n_iterations} if family == "matrix" else {"n_loops": n_iterations}


MATRIX_BASELINE_CAPACITY_BAND = 0.05   # FIX-B: warn when |ratio - 1| exceeds this


def capacity_confound_block(family: str, n_params: int, mat_dim: int, max_len: int,
                             vocab_size: int, n_iterations: int) -> dict:
    """FIX-B (audit): params + ratio vs the matrix-all baseline at the
    IDENTICAL config, recorded in every result JSON. The reference count is
    computed by instantiating the actual MatrixModelSpec baseline (CPU,
    freed immediately) rather than hardcoding 290,328 -- so the ratio stays
    correct if mat_dim/vocab/max_len ever change. Motivation:
    STAGE_G_DESIGN.md section 14.4 -- h_b_factored_r4 carries ~2.69x the
    baseline's params, and that exact cell's Wave-B "win" partially
    collapsed against a capacity-matched vector control; any H_e win by r4
    must be read with the same confound in view."""
    ref = MatrixModelSpec(mat_dim=mat_dim, n_heads=8, n_iterations=n_iterations,
                           max_len=max_len, vocab_size=vocab_size, variant="baseline").build()
    baseline_params = ref.count_params()
    del ref
    ratio = n_params / baseline_params
    return {
        "n_params": n_params,
        "matrix_baseline_params_same_config": baseline_params,
        "ratio_vs_matrix_baseline": ratio,
        "outside_5pct_band": abs(ratio - 1.0) > MATRIX_BASELINE_CAPACITY_BAND,
    }


def lm_loss(logits, target_ids, pred_pos, answer_loss_weight: float):
    """FIX-C (audit): answer-position loss upweighting. Only Q_per_doc
    answer bytes out of doc_len positions (8/96 ~= 8.3% at the manifest
    config) carry the COMPOSITIONAL signal; the rest is R/Q/digit structure
    the calibration runs showed is learned fast (BPB 8.2 -> 1.6) while
    composition stayed at chance. Upweighting the answer positions shifts
    gradient budget onto the signal without changing the task.

    Returns (loss_for_backward, unweighted_mean_ce):
      answer_loss_weight == 1.0  ->  loss is the EXACT pre-FIX-C objective,
        BITWISE (the identical F.cross_entropy(..., reduction='mean')
        expression, same op order -- verified with torch.equal in the smoke
        gate, not allclose); unweighted_mean_ce is None (loss IS already
        the unweighted CE).
      answer_loss_weight != 1.0  ->  loss = sum(w_i * ce_i) / sum(w_i),
        w_i = W at each query's pred_pos target (the answer byte), 1.0
        elsewhere; unweighted_mean_ce = ce.mean() is also returned so the
        training log keeps a comparable plain-LM BPB next to the weighted
        training loss.
    """
    vocab_size = logits.shape[-1]
    lm_logits = logits[:, :-1, :]
    if answer_loss_weight == 1.0:
        # EXACT old objective -- deliberately NOT folded into the weighted
        # path below (a sum(ce*w)/sum(w) re-derivation of the mean is only
        # allclose-equal, not bitwise-equal; the audit's W=1.0 bar is bitwise).
        return F.cross_entropy(lm_logits.reshape(-1, vocab_size), target_ids.reshape(-1)), None
    ce = F.cross_entropy(lm_logits.reshape(-1, vocab_size), target_ids.reshape(-1),
                          reduction="none")                        # (B*(L-1),)
    B, Lm1 = target_ids.shape
    weights = torch.ones(B, Lm1, device=logits.device, dtype=ce.dtype)
    weights.scatter_(1, pred_pos, float(answer_loss_weight))         # W at answer-byte targets
    weights = weights.reshape(-1)
    loss = (ce * weights).sum() / weights.sum()
    return loss, ce.mean()


@torch.no_grad()
def evaluate_composition(model, cfg: the.HeTaskConfig, family: str, hops_to_eval, gen, device,
                          n_iterations: int, vocab_size: int, n_batches: int = 8, batch_size: int = 64):
    """Composition-recovery accuracy per hop (task_dn.py::evaluate_at_hops
    convention, adapted): for each h, draw n_batches FRESH documents with
    EVERY query forced to that exact hop (hop_set=(h,)), forward once
    (teacher-forced), gather logits at each query's pred_pos, argmax, and
    score against the true answer byte. Also reports whole-document BPB
    (secondary, per the task brief) and the chance baseline (1/K) for
    context."""
    model.eval()
    per_hop = {}
    for h in hops_to_eval:
        correct = 0
        total = 0
        loss_sum, loss_tok = 0.0, 0
        for _ in range(n_batches):
            b = the.sample_doc_batch(cfg, batch_size, gen, hop_set=(h,), device=device)
            with _autocast_ctx(device):
                logits, _info = model(b["input_ids"], **_fwd_kwargs(family, n_iterations))
                lm_logits = logits[:, :-1, :].reshape(-1, vocab_size)
                lm_targets = b["target_ids"].reshape(-1)
                loss_sum += F.cross_entropy(lm_logits, lm_targets, reduction="sum").item()
                loss_tok += lm_targets.numel()

            B, Q = b["pred_pos"].shape
            pred_logits = torch.gather(
                logits, 1, b["pred_pos"].unsqueeze(-1).expand(B, Q, vocab_size))   # (B,Q,vocab)
            pred_byte = pred_logits.argmax(dim=-1)                                  # (B,Q)
            true_byte = the.ENTITY_BASE + b["answer_id"]
            correct += (pred_byte == true_byte).sum().item()
            total += B * Q

        acc = correct / max(total, 1)
        chance = cfg.chance
        per_hop[h] = {
            "h": h, "effective_hop": h % cfg.K,
            "n": total, "composition_accuracy": acc, "chance": chance,
            "chance_adjusted_acc": (acc - chance) / max(1.0 - chance, 1e-9),
            "bpb": (loss_sum / max(loss_tok, 1)) / LOG2,
        }
    model.train()
    return per_hop


class _OracleOneHotModel(torch.nn.Module):
    """Smoke-gate-ONLY positive control: outputs a one-hot spike at the
    TRUE next byte. Under teacher forcing, input_ids[:, t+1] literally IS
    the label for position t (present in the SAME sequence) -- so this
    'oracle' scores ~1.0 by construction, regardless of any real model's
    quality. This is a PIPELINE-correctness check (pred_pos indexing,
    gather alignment, argmax-vs-answer-byte comparison), NOT a claim about
    what a real model can do. Deliberately NOT the same check as "untrained
    model lands near chance=1/K" -- that check is WRONG for this task: an
    untrained model's argmax ranges over the full 256-byte vocab (R/Q/\\n/
    digits/unused bytes included), not just the K valid entity bytes, so
    'near 1/K' is not actually the correct expectation and would either
    spuriously fail or (worse) silently pass a broken pipeline at ~0."""

    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, token_ids, **kwargs):
        B, L = token_ids.shape
        nxt = torch.cat([token_ids[:, 1:], token_ids[:, -1:]], dim=1)   # position t's true next byte
        logits = torch.full((B, L, self.vocab_size), -20.0, device=token_ids.device)
        logits.scatter_(-1, nxt.unsqueeze(-1), 20.0)
        return logits, {"ranks": []}


class _OracleWrongModel(torch.nn.Module):
    """Smoke-gate-ONLY negative control: ALWAYS predicts one FIXED byte
    that is never a valid entity byte, at every position -- confirms
    composition_accuracy is NOT trivially 1.0 regardless of the model (the
    positive oracle's ~1.0 score above has teeth, not a pipeline bug that
    always reports success)."""

    def __init__(self, vocab_size, wrong_byte):
        super().__init__()
        self.vocab_size = vocab_size
        self.wrong_byte = wrong_byte

    def forward(self, token_ids, **kwargs):
        B, L = token_ids.shape
        logits = torch.full((B, L, self.vocab_size), -20.0, device=token_ids.device)
        logits[..., self.wrong_byte] = 20.0
        return logits, {"ranks": []}


def _dump(path, obj):
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"  (incremental write failed, non-fatal: {e!r})", flush=True)


def _assemble(family, variant, mat_dim, vocab_size, n_iterations, steps, seed, K, Q_per_doc,
              H_train, H_test, H_extra, n_params, breakdown, trajectory, checkpoints, t0,
              timed_out, steps_completed, complete, seq_len,
              answer_loss_weight, capacity_confound):
    flops = analytic_flops_for_cell(family, variant, mat_dim, seq_len, n_iterations, vocab_size)
    result = {
        "H_e_task": True,             # identity sentinel (task brief: "add ... to is_done identity")
        "family": family, "variant": variant, "mat_dim": mat_dim, "vocab_size": vocab_size,
        "n_iterations": n_iterations, "steps": steps, "seed": seed, "seq_len": seq_len,
        "K": K, "Q_per_doc": Q_per_doc, "H_train": list(H_train), "H_test": list(H_test),
        "H_extra": list(H_extra), "chance": 1.0 / K,
        "answer_loss_weight": answer_loss_weight,       # FIX-C: part of the run's identity (is_done cross-checks it)
        "capacity_confound": capacity_confound,          # FIX-B: params + ratio vs matrix baseline
        "n_params": n_params, "param_breakdown": breakdown,
        "analytic_flops_per_token": flops,
        "steps_completed": steps_completed, "complete": complete, "timed_out": timed_out,
        "trajectory": trajectory, "checkpoints": checkpoints, "wall_s": time.time() - t0,
    }
    if checkpoints:
        final = checkpoints[-1]
        result["final_step"] = final["step"]
        result["final_composition"] = final["composition"]
    return result


def train(family, variant, mat_dim, max_len, vocab_size, n_iterations, steps, seed, K=12,
          Q_per_doc=8, H_train=(1, 2, 3), H_test=(4, 5), H_extra=(7,), batch_size=64, lr=3e-4,
          warmup_steps=None, weight_decay=0.01, grad_clip=1.0, log_every=100, ckpt_every=500,
          out_path=None, device="cpu", internal_timeout=None,
          eval_n_batches=8, eval_batch_size=64, answer_loss_weight=5.0):
    t0 = time.time()
    torch.manual_seed(seed)
    device = torch.device(device)

    cfg = the.HeTaskConfig(K=K, Q_per_doc=Q_per_doc, H_train=tuple(H_train),
                            H_test=tuple(H_test), H_extra=tuple(H_extra))
    seq_len = cfg.doc_len
    assert seq_len <= max_len, f"doc_len={seq_len} exceeds --max-len={max_len} (position table too small)"
    assert answer_loss_weight > 0, f"answer_loss_weight must be > 0, got {answer_loss_weight}"

    model = build_model(family, variant, mat_dim, max_len, vocab_size, n_iterations).to(device)
    n_params = model.count_params()
    breakdown = model.param_breakdown()
    capacity_confound = capacity_confound_block(family, n_params, mat_dim, max_len,
                                                 vocab_size, n_iterations)   # FIX-B
    print(f"H_e task: family={family} variant={variant} K={K} Q_per_doc={Q_per_doc} "
          f"doc_len={seq_len} H_train={cfg.H_train} H_test={cfg.H_test} H_extra={cfg.H_extra} "
          f"chance={cfg.chance:.4f} params={n_params:,} breakdown={breakdown} "
          f"answer_loss_weight={answer_loss_weight} device={device}", flush=True)
    if capacity_confound["outside_5pct_band"]:
        print(f"  CAPACITY-CONFOUND WARNING (FIX-B, STAGE_G_DESIGN.md section 14.4): this cell "
              f"has {n_params:,} params = "
              f"{capacity_confound['ratio_vs_matrix_baseline']:.2f}x the matrix baseline "
              f"({capacity_confound['matrix_baseline_params_same_config']:,}) at the same "
              f"config -- outside the +/-5% band. Any win/loss vs the baselines confounds "
              f"the component swap with raw capacity; section 14.4 showed this exact "
              f"confound partially collapsed h_b_factored_r4's Wave-B win.", flush=True)

    warmup_steps = warmup_steps if warmup_steps is not None else max(1, steps // 10)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    gen = torch.Generator(device=device).manual_seed(seed)
    trajectory, checkpoints = [], []
    step, steps_completed, timed_out = 0, 0, False
    model.train()

    while step < steps:
        b = the.sample_doc_batch(cfg, batch_size, gen, hop_set=cfg.H_train, device=device,
                                  assert_bijection=(step == 0))   # structural guarantee -- check once (perf)
        opt.zero_grad()
        with _autocast_ctx(device):
            logits, _info = model(b["input_ids"], **_fwd_kwargs(family, n_iterations))
            # FIX-C: weighted objective for backward; unweighted mean CE
            # (None when W==1.0, where loss IS the plain CE) kept for a
            # cross-run-comparable BPB log.
            loss, unweighted_ce = lm_loss(logits, b["target_ids"], b["pred_pos"],
                                           answer_loss_weight)
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if finite:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        else:
            grad_norm = float("nan")
        sched.step()
        step += 1
        steps_completed = step

        if step % log_every == 0 or step == 1:
            plain_ce = loss.item() if unweighted_ce is None else unweighted_ce.item()
            bpb = plain_ce / LOG2         # BPB is always the UNWEIGHTED number (comparable across W)
            entry = {"step": step, "loss": plain_ce, "bpb": bpb,
                      "grad_norm": float(grad_norm), "lr": sched.get_last_lr()[0]}
            if unweighted_ce is not None:
                entry["loss_weighted"] = loss.item()    # the actual training objective at W != 1
            trajectory.append(entry)
            wtag = f"  loss_w {loss.item():.4f}" if unweighted_ce is not None else ""
            print(f"  step {step:6d}  loss {plain_ce:.4f}  bpb {bpb:.4f}{wtag}  "
                  f"gn {float(grad_norm):.3f}", flush=True)

        if step % ckpt_every == 0 or step == steps:
            eval_gen = torch.Generator(device=device).manual_seed(seed + 10_000 + step)
            comp_train = evaluate_composition(model, cfg, family, cfg.H_train, eval_gen, device,
                                               n_iterations, vocab_size, eval_n_batches, eval_batch_size)
            comp_held = evaluate_composition(model, cfg, family, (*cfg.H_test, *cfg.H_extra),
                                              eval_gen, device, n_iterations, vocab_size,
                                              eval_n_batches, eval_batch_size)
            ckpt = {"step": step, "composition": {"in_distribution": comp_train, "held_out": comp_held}}
            checkpoints.append(ckpt)
            partial = _assemble(family, variant, mat_dim, vocab_size, n_iterations, steps, seed,
                                 K, Q_per_doc, H_train, H_test, H_extra, n_params, breakdown,
                                 trajectory, checkpoints, t0, timed_out=False,
                                 steps_completed=steps_completed, complete=False, seq_len=seq_len,
                                 answer_loss_weight=answer_loss_weight,
                                 capacity_confound=capacity_confound)
            _dump(out_path, partial)
            # FIX-A print convention: h=1 is a LOOKUP metric (copy leak,
            # task_he.py docstring); the composition headline is h>=2.
            comp_hops = [h for h in cfg.H_train if h >= 2]
            lookup_part = (f"h=1(lookup)={comp_train[1]['composition_accuracy']:.3f}  "
                            if 1 in comp_train else "")
            comp_summary = "  ".join(f"h={h}:{comp_train[h]['composition_accuracy']:.3f}"
                                      for h in comp_hops)
            held_summary = "  ".join(f"h={h}:{comp_held[h]['composition_accuracy']:.3f}"
                                      for h in comp_held)
            print(f"  [checkpoint step {step}] {lookup_part}in-dist composition (h>=2): "
                  f"{comp_summary} (chance={cfg.chance:.3f})  held-out: {held_summary}", flush=True)

        if internal_timeout is not None and time.time() - t0 > internal_timeout:
            print(f"  internal timeout ({internal_timeout}s) reached at step {step}; stopping early",
                  flush=True)
            timed_out = True
            break

    return _assemble(family, variant, mat_dim, vocab_size, n_iterations, steps, seed, K, Q_per_doc,
                      H_train, H_test, H_extra, n_params, breakdown, trajectory, checkpoints, t0,
                      timed_out, steps_completed, complete=(not timed_out and steps_completed >= steps),
                      seq_len=seq_len, answer_loss_weight=answer_loss_weight,
                      capacity_confound=capacity_confound)


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------

def smoke(device="cpu"):
    print(f"{'=' * 60}\n  STAGE G H_e (TASK-SWAP) SMOKE (device={device})\n{'=' * 60}")

    print("\n[1] task_he self-test (grammar, byte-vocab range, parse-and-recompute "
          "round-trip, periodicity guard, bijection guard)")
    the._self_test()

    print("\n[2] tiny end-to-end training + eval for the 3 planned manifest cells "
          "(matrix baseline, matrix h_b_factored_r4, vector baseline)")
    tiny_cfg = dict(mat_dim=8, max_len=128, vocab_size=256, n_iterations=2, steps=6, seed=0,
                     K=6, Q_per_doc=4, H_train=(1, 2), H_test=(3,), H_extra=(),
                     batch_size=4, log_every=3, ckpt_every=3, device=device)
    for family, variant in (("matrix", "baseline"), ("matrix", "h_b_factored_r4"),
                             ("vector", "baseline")):
        res = train(family, variant, **tiny_cfg)
        assert res["complete"] is True, f"{family}/{variant}: did not complete"
        assert res["steps_completed"] == 6
        assert len(res["checkpoints"]) >= 1
        assert res["H_e_task"] is True
        # FIX-C: answer_loss_weight recorded (train() default 5.0); at W!=1
        # the trajectory carries BOTH the weighted objective and plain CE.
        assert res["answer_loss_weight"] == 5.0
        assert all("loss_weighted" in t for t in res["trajectory"]), \
            f"{family}/{variant}: trajectory missing loss_weighted at W=5.0"
        # FIX-B: capacity_confound block present and correct per cell.
        cc = res["capacity_confound"]
        assert set(cc) == {"n_params", "matrix_baseline_params_same_config",
                            "ratio_vs_matrix_baseline", "outside_5pct_band"}
        if (family, variant) == ("matrix", "baseline"):
            assert cc["ratio_vs_matrix_baseline"] == 1.0 and not cc["outside_5pct_band"], \
                f"matrix baseline must self-ratio to exactly 1.0, got {cc}"
        if variant == "h_b_factored_r4":
            assert cc["ratio_vs_matrix_baseline"] > 1.5 and cc["outside_5pct_band"], \
                f"r4 capacity confound not flagged: {cc}"
        final = res["final_composition"]
        for grp in ("in_distribution", "held_out"):
            for h, entry in final[grp].items():
                acc = entry["composition_accuracy"]
                assert acc == acc, f"{family}/{variant}: NaN composition_accuracy at h={h}"
                assert 0.0 <= acc <= 1.0
        print(f"  {family}/{variant}: OK, n_params={res['n_params']:,} "
              f"(x{cc['ratio_vs_matrix_baseline']:.2f} baseline"
              f"{', CONFOUND FLAGGED' if cc['outside_5pct_band'] else ''}), "
              f"final in-dist h={tiny_cfg['H_train'][0]} "
              f"acc={final['in_distribution'][tiny_cfg['H_train'][0]]['composition_accuracy']:.3f} "
              f"(chance={1.0/tiny_cfg['K']:.3f})")

    print("\n[3] eval-pipeline oracle positive/negative controls (pred_pos indexing, gather "
          "alignment, argmax-vs-answer-byte comparison) -- a stronger, deterministic "
          "replacement for 'check an untrained model's accuracy', which is NOT a reliable "
          "near-chance signal here (an untrained net's argmax ranges over the full 256-byte "
          "vocab, not just the K valid entity bytes)")
    torch.manual_seed(1)
    cfg_c = the.HeTaskConfig(K=10, Q_per_doc=8, H_train=(1, 2, 3), H_test=(4, 5), H_extra=(7,))
    gen_c = torch.Generator(device=device).manual_seed(2)
    oracle = _OracleOneHotModel(vocab_size=256).to(device)
    comp_oracle = evaluate_composition(oracle, cfg_c, "matrix", (*cfg_c.H_train, *cfg_c.H_test),
                                        gen_c, device, n_iterations=2, vocab_size=256,
                                        n_batches=4, batch_size=16)
    for h, entry in comp_oracle.items():
        assert entry["composition_accuracy"] > 0.999, \
            f"oracle (reads the true next byte straight from the input) should score ~1.0 " \
            f"at h={h}, got {entry['composition_accuracy']:.4f} -- pred_pos/gather/argmax " \
            f"alignment is likely broken"
    print(f"  positive control (oracle): composition_accuracy > 0.999 at every hop "
          f"{list(comp_oracle.keys())} (h=1 acc={comp_oracle[cfg_c.H_train[0]]['composition_accuracy']:.4f})")

    wrong_byte = 200   # not in [ENTITY_BASE, ENTITY_BASE+K) for any K<=26, and not R/Q/NL/digit
    assert not (the.ENTITY_BASE <= wrong_byte < the.ENTITY_BASE + 26)
    oracle_wrong = _OracleWrongModel(vocab_size=256, wrong_byte=wrong_byte).to(device)
    comp_wrong = evaluate_composition(oracle_wrong, cfg_c, "matrix", cfg_c.H_train, gen_c, device,
                                       n_iterations=2, vocab_size=256, n_batches=4, batch_size=16)
    acc_wrong = comp_wrong[cfg_c.H_train[0]]["composition_accuracy"]
    assert acc_wrong == 0.0, \
        f"negative control (always predicts a non-entity byte) should score EXACTLY 0.0, " \
        f"got {acc_wrong:.4f} -- composition_accuracy is not discriminating (no teeth)"
    print(f"  negative control (always-wrong-byte oracle): composition_accuracy = "
          f"{acc_wrong:.4f} (expect exactly 0.0) -- the metric has teeth")

    print("\n[4] FIX-C loss equivalence + weighting checks on a FIXED batch: (a) BITWISE "
          "W=1.0 == the pre-FIX-C objective (torch.equal, not allclose); (b) an "
          "independently re-derived weighted mean matches lm_loss at W=5.0; (c) W=5.0 "
          "responds to answer-position error while W=1.0's sensitivity is diluted "
          "(directional check that the weighting actually targets pred_pos)")
    torch.manual_seed(3)
    cfg_w = the.HeTaskConfig(K=8, Q_per_doc=6, H_train=(1, 2, 3), H_test=(4, 5), H_extra=())
    gen_w = torch.Generator(device=device).manual_seed(4)
    bw = the.sample_doc_batch(cfg_w, 8, gen_w, hop_set=cfg_w.H_train, device=device)
    Bw, Lw = bw["input_ids"].shape
    logits_w = torch.randn(Bw, Lw, 256, device=device)

    # (a) bitwise: lm_loss at W=1.0 vs the literal pre-FIX-C expression.
    old_obj = F.cross_entropy(logits_w[:, :-1, :].reshape(-1, 256), bw["target_ids"].reshape(-1))
    new_obj, unw = lm_loss(logits_w, bw["target_ids"], bw["pred_pos"], answer_loss_weight=1.0)
    assert unw is None
    assert torch.equal(new_obj, old_obj), \
        f"W=1.0 is NOT bitwise-identical to the old objective: {new_obj.item()!r} vs {old_obj.item()!r}"
    print(f"  (a) torch.equal(lm_loss(W=1.0), pre-FIX-C objective) = True (loss={old_obj.item():.6f})")

    # (b) independent re-derivation of the weighted mean at W=5.0.
    W = 5.0
    loss_w, unw_w = lm_loss(logits_w, bw["target_ids"], bw["pred_pos"], answer_loss_weight=W)
    ce_flat = F.cross_entropy(logits_w[:, :-1, :].reshape(-1, 256), bw["target_ids"].reshape(-1),
                               reduction="none").reshape(Bw, Lw - 1)
    num, den = 0.0, 0.0
    for row in range(Bw):
        ppos = set(bw["pred_pos"][row].tolist())
        for pos in range(Lw - 1):
            w_i = W if pos in ppos else 1.0
            num += w_i * ce_flat[row, pos].item()
            den += w_i
    manual = num / den
    assert abs(loss_w.item() - manual) < 1e-4, \
        f"weighted loss {loss_w.item():.6f} != independent re-derivation {manual:.6f}"
    assert abs(unw_w.item() - ce_flat.mean().item()) < 1e-5
    print(f"  (b) lm_loss(W=5.0)={loss_w.item():.6f} matches the scalar-loop re-derivation "
          f"({manual:.6f}); returned unweighted CE matches ce.mean()")

    # (c) directional: corrupting ONLY the answer-position logits moves the
    # W=5 loss by ~(5/avg_w) x the W=1 movement (up to weight-mass norm).
    logits_bad = logits_w.clone()
    idx = bw["pred_pos"].unsqueeze(-1).expand(Bw, cfg_w.Q_per_doc, 256)
    logits_bad.scatter_(1, idx, torch.zeros(Bw, cfg_w.Q_per_doc, 256, device=device))  # zero answer logits
    l1_ref, _ = lm_loss(logits_w, bw["target_ids"], bw["pred_pos"], 1.0)
    l1_bad, _ = lm_loss(logits_bad, bw["target_ids"], bw["pred_pos"], 1.0)
    l5_ref, _ = lm_loss(logits_w, bw["target_ids"], bw["pred_pos"], W)
    l5_bad, _ = lm_loss(logits_bad, bw["target_ids"], bw["pred_pos"], W)
    d1 = (l1_bad - l1_ref).item()
    d5 = (l5_bad - l5_ref).item()
    assert abs(d5) > abs(d1) * 1.5, \
        f"W=5.0 should amplify answer-position perturbations vs W=1.0: d5={d5:.6f} d1={d1:.6f}"
    print(f"  (c) answer-position perturbation moves W=5.0 loss {abs(d5/max(abs(d1),1e-12)):.2f}x "
          f"more than W=1.0 loss (d5={d5:.5f}, d1={d1:.5f}) -- weighting targets pred_pos")

    print(f"\nSMOKE PASSED (device={device})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--family", choices=["matrix", "vector"], default="matrix")
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--mat-dim", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--n-iterations", type=int, default=8)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--Q-per-doc", type=int, default=8)
    ap.add_argument("--h-train", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--h-test", type=int, nargs="+", default=[4, 5])
    ap.add_argument("--h-extra", type=int, nargs="+", default=[7])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--eval-n-batches", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--answer-loss-weight", type=float, default=5.0,
                     help="FIX-C (audit): cross-entropy weight at each query's answer-byte "
                          "target position (see lm_loss). 1.0 recovers the pre-FIX-C plain "
                          "LM objective BITWISE (smoke-verified with torch.equal). Recorded "
                          "in the result JSON and part of the run's is_done identity in "
                          "run_stageg_he_sweep.py.")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--internal-timeout", type=float, default=None)
    args = ap.parse_args()

    if args.smoke:
        smoke(device=args.device)
        return

    result = train(args.family, args.variant, args.mat_dim, args.max_len, args.vocab_size,
                    args.n_iterations, args.steps, args.seed, K=args.K, Q_per_doc=args.Q_per_doc,
                    H_train=tuple(args.h_train), H_test=tuple(args.h_test),
                    H_extra=tuple(args.h_extra), batch_size=args.batch_size, lr=args.lr,
                    warmup_steps=args.warmup_steps, log_every=args.log_every,
                    ckpt_every=args.ckpt_every, out_path=args.out, device=args.device,
                    internal_timeout=args.internal_timeout, eval_n_batches=args.eval_n_batches,
                    eval_batch_size=args.eval_batch_size,
                    answer_loss_weight=args.answer_loss_weight)

    summary = {k: v for k, v in result.items() if k not in ("trajectory", "checkpoints")}
    print("\nRESULT SUMMARY:", json.dumps(summary, indent=2), flush=True)
    if args.out:
        _dump(args.out, result)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
