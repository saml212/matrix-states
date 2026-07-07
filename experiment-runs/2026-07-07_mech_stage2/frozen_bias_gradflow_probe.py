"""frozen_bias_gradflow_probe.py -- FROZEN_BIAS_LM_DESIGN.md sec 12.5's Stage
2 (H3): new, SHORT, backward-hook-instrumented training cells measuring the
gradient norm/variance reaching `k_raw` (the post-conv, pre-blend key tensor
`apply_frozen_bias_blend` consumes, lm_pretrain_rd.py:838/855-857) for Arm 1
(off), Arm 2 (per_token, lambda=0.58), Arm 2' (global, lambda=0.58) -- 1 seed
(0), 1 corpus (openr1-mix-ext), FULL 20,000 steps per cell.

**H3, sec 12.2's hypothesis table, quoted verbatim (the hypothesis this
script's own telemetry is the PRIMARY test statistic for):**

    H3 | Optimizer/gradient-flow interaction: the lambda-blend's normalize()
    Jacobian, evaluated at a per-token-random vs. a fixed-global bias
    direction, differentially reshapes the gradient signal reaching k_raw,
    and this difference compounds over training. | Backward-hook
    gradient-norm/variance telemetry on k_raw, short instrumented runs. |
    Yes -- new short training | Stage 2 (conditional) | ~0.15-0.8 GPU-h

**Stage-2 gate result, sec 12.11.4, quoted (this build targets the branch it
selected):**

    Stage-2 gate (sec 12.5's frozen rule): FULL 20,000-step branch. Arm2:
    sign(delta@5000)=sign(delta@20000) [pass] and |-0.0721| >=
    0.5*|-0.0903|=0.0451 [pass] -> full. Arm2': sign [pass] but |-0.0570| <
    0.5*|-0.1203|=0.0601 -> truncated. Per the rule's own clause ("if the two
    comparisons select different branches, the full 20,000-step branch
    governs for both"): Stage 2 is authorized at full 20,000 steps (~0.76
    GPU-h + the mandatory CPU smoke, sec 12.5).

This script therefore runs every cell at args.steps=20000 by default -- NOT
the cheaper 3,000-step branch sec 12.5 also specifies (that branch was NOT
selected by the gate above; --steps is exposed as a CLI override only for
ad-hoc re-runs, never the registered default).

**sec 12.9 item 4's formula-vs-plumbing gap, which this build closes:**
Stage 0.5's synthetic self-tests (mech_stage05_selftests.py) validate
FORMULAS on hand-picked mock tensors, not the real capture_raw_keys/
torch.load/hook-registration code paths. This script's own smoke (--smoke,
below) instead pushes the REAL, unmodified `DeltaNetLM`/`DeltaNetLMBlock`/
`DeltaNetLMMixer` classes (imported directly from lm_pretrain_rd.py, never
reimplemented) through the SAME `register_kraw_gradnorm_hooks()` function
this script's real Stage-2 cells call -- one function object, not a smoke-
only stand-in copy, exercised end-to-end (forward -> backward -> hook
firing) on a tiny CPU-safe configuration.

**Design: NOT a modification of lm_pretrain_rd.py in place -- the training
LOOP is cloned here (per this codebase's own duplication-over-cross-import
convention, restated in every sibling sweep/probe script's docstring:
frozen_bias_retrofit_eval_rd.py, frozen_bias_lm_sweep.py, mech_h4_paramdiff.py
all clone rather than modify shared orchestration code). The MODEL and its
shared primitives are reused UNMODIFIED, imported directly:
`DeltaNetLM`/`DeltaNetLMBlock`/`DeltaNetLMMixer`, `apply_frozen_bias_blend`,
`load_corpus`, `get_batch`, `get_lr`, `set_and_log_tf32`, `CORPUS_DIRS`,
`DEFAULT_DATA_DIR`, `FROZEN_BIAS_ARM_MODES`, `FROZEN_BIAS_LAMBDA_PRIMARY`,
`_MIN_KERNEL_T` -- exactly the same convention frozen_bias_retrofit_eval_rd.py
already uses for its own eval-only tooling (see its own import block).

**How the hook is inserted WITHOUT touching DeltaNetLMMixer.forward's source:**
`k_raw` (lm_pretrain_rd.py's local variable `k` right after
`k, _ = self.k_conv1d(self.k_proj(x))`, line ~838, BEFORE the
`if self.frozen_bias_arm != "off":` blend guard at line ~854) is EXACTLY the
output of the mixer's own `k_conv1d` submodule -- for ALL THREE arms
uniformly (off/per_token/global), since k_conv1d's own computation never
depends on frozen_bias_arm. `register_kraw_gradnorm_hooks()` below installs
ONE `register_forward_hook` per layer's `mixer.k_conv1d` (a real nn.Module
already on the model, no source edit needed); inside that forward-hook
callback, on cadence-eligible steps only, it calls `register_hook` on the
conv's own output tensor -- the standard, non-invasive PyTorch idiom for
observing an intermediate gradient during `.backward()`, strictly READ-ONLY
(the closure returns None, never a replacement gradient). This closes the
Arm-1 ("off") gap a naive `apply_frozen_bias_blend`-wrapping monkeypatch
would have missed: DeltaNetLMMixer.forward only CALLS
`apply_frozen_bias_blend` when `frozen_bias_arm != "off"` (line 854) -- Arm
1's own k_raw gradient would never reach a wrapper hung off that function.
Hooking `k_conv1d`'s output instead captures the identical tensor for every
arm, unconditionally, by construction.

**CPU-safe reduced path (fla is NOT installed on the local dev machine --
`import fla` fails; the real `chunk_delta_rule` Triton kernel "has no CPU
path", lm_pretrain_rd.py's own smoke() assert). This script installs a fla
stub ADAPTED from smoke_frozen_bias_lm.py's/smoke_frozen_bias_wave_neg1.py's
own stub (same `_StubShortConvolution`/`_StubRMSNorm` -- real forward/
backward through q/k/v_conv1d + RMSNorm, CPU-safe stand-ins for those two
pieces, unchanged) -- real forward/backward through q/k/v_conv1d + RMSNorm.
**One deliberate, disclosed difference from that stub's own
`_stub_chunk_delta_rule`:** the sibling files' version computes `o` from
`v`/`beta` ALONE (`o = v * sigmoid(beta)`) -- it never reads `k` in computing
its output, which is fine for THEIR smoke items (shape/finiteness/table
checks) but WRONG for this script's own purpose: a `k` that never
functionally influences the loss gets NO gradient at all, so
`register_hook` on it would silently never fire (found empirically, this
build session -- the first draft of this stub reused the sibling's version
verbatim and every smoke item depending on nonzero telemetry failed
silently until this was traced to the stub, not the hook code). This
file's own `_stub_chunk_delta_rule` therefore adds a cheap, genuinely
`k`-dependent term (`o = v * sigmoid(beta) * sigmoid((q*k).sum(-1))`) so
gradient actually flows back through `k_raw` -- the minimum change needed
to make the stub exercise what this smoke needs to test, while keeping
`_StubShortConvolution`/`_StubRMSNorm` (the pieces the hook itself attaches
to) identical to the sibling files' own versions. On the H100 box, where
the real `fla` package IS installed, `import fla` succeeds and NO stub is
installed -- `DeltaNetLM` there is wired to the real Triton kernel,
unchanged, which of course has a real k-dependency throughout. This means
the REAL cells (non-smoke) always require CUDA (mirrors lm_pretrain_rd.py's
own `assert device == "cuda"`), while `--smoke` runs the identical hook-
registration function against the real model class, minus only the
GPU-only kernel, and completes on CPU.**

**Deliberate build-time scope cut (flagged for audit scrutiny): this script
does NOT save model checkpoints for its 3 cells.** No downstream sec 12 step
registers reusing a Stage-2 checkpoint (Stage 2's own deliverable, sec 12.5,
is the grad-norm telemetry JSON, not a new checkpoint set) -- omitting
`torch.save` keeps these cells cheap and simple. Easy to add `--ckpt-dir`
support later if a follow-on step needs it.

**Design choice: all 3 arms run SEQUENTIALLY inside ONE process/invocation**
(not 3 separate `python ...` launches like frozen_bias_lm_sweep.py's rung-1
cells) -- sec 12.5 explicitly allows either ("Output: one JSON per cell (or
one combined)"). One process amortizes Python/CUDA/model-import startup
overhead across all 3 cells instead of paying it 3x, so real wall time should
UNDERSHOOT the ~0.76 GPU-h estimate below, never exceed it on that account.

**Cost derivation (mech_stage2_chain.sh's own header re-derives/discloses
this too):** sec 12.5 states ~912s/cell x 3 ~= 0.76 GPU-h at the full
20,000-step branch (~0.0456 s/step). This build could not reach the box (no
SSH here) to read the real rung-1 frozen-bias calibration.json (it is not
committed to this repo -- only small archived JSONs are, per the repo's
size-capped archive policy). As a CPU-only corroboration, this repo DOES
carry a two-point calibration at the IDENTICAL architecture
(dm256/ds64/L2, openr1-mix corpus): results/lm_rd_trackc/calibration/
calib_control_ptA_lm_openr1-mix_dm256_ds64_L2_s0.json (100 steps, 6.0068s)
and .../calib_control_ptB_..._s0.json (300 steps, 14.7468s) ->
(14.7468-6.0068)/(300-100) = 0.0437 s/step -- within 5% of sec 12.5's own
0.0456 s/step figure, corroborating (not replacing) the design doc's own
pinned cost numbers, which this script's header/chain reproduce verbatim.

Run the smoke gate FIRST: python frozen_bias_gradflow_probe.py --smoke
Real cells (GPU, requires the real fla package + rebuilt corpora on /data):
  python frozen_bias_gradflow_probe.py --corpus openr1-mix-ext --seed 0 \\
      --steps 20000 --arms off,per_token,global \\
      --out results/mech_wave/mech_stage2_gradflow_openr1-mix-ext_s0.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # pod-safe imports, matches every sibling script in this dir

# ---------------------------------------------------------------------------
# fla stub -- installed ONLY if the real package is not importable (dev
# machine). SAME functionally-real stub smoke_frozen_bias_lm.py /
# smoke_frozen_bias_wave_neg1.py already install (cloned again here, a third
# instance, per this codebase's own duplication-over-cross-import
# convention) -- real forward/backward through q/k/v_conv1d + RMSNorm, a
# CPU-safe stand-in ONLY for the GPU-only chunk_delta_rule kernel math.
# MUST run BEFORE `from lm_pretrain_rd import ...` below (that import
# statement executes lm_pretrain_rd.py's own top-level `from fla... import`
# lines immediately).
# ---------------------------------------------------------------------------

def _ensure_fla_stub() -> bool:
    try:
        import fla  # noqa: F401
        return False
    except ImportError:
        pass

    class _StubShortConvolution(nn.Module):
        def __init__(self, hidden_size: int, kernel_size: int = 4, bias: bool = False,
                     activation: str | None = "silu"):
            super().__init__()
            self.activation = activation
            self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, groups=hidden_size,
                                   padding=kernel_size - 1, bias=bias)

        def forward(self, x: torch.Tensor, cache=None):
            B, T, D = x.shape
            out = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)
            if self.activation == "silu":
                out = F.silu(out)
            return out, None

    class _StubRMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(var + self.eps) * self.weight

    def _stub_chunk_delta_rule(q, k, v, beta, initial_state=None, output_final_state=True,
                                use_qk_l2norm_in_kernel=True):
        """Deliberately DIFFERENT from smoke_frozen_bias_lm.py's own stub of the same name:
        that version computes `o` from v/beta alone, so `k` never influences the loss and a
        gradient-flow hook on k_raw would never fire (this build's own empirical finding --
        see the module docstring). Adds the minimum k-dependent term
        (`sigmoid((q*k).sum(-1))`, a cheap per-position scalar gate) so gradient genuinely
        flows back through k_raw -- everything else (q/k L2-norm, v*sigmoid(beta) gating,
        final_state shape) matches the sibling stub."""
        B, T, H, Dh = q.shape
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        qk_gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        o = v * torch.sigmoid(beta).unsqueeze(-1) * qk_gate
        final_state = torch.zeros(B, H, Dh, Dh, dtype=q.dtype, device=q.device)
        return o, final_state

    fla_mod = types.ModuleType("fla")
    fla_modules = types.ModuleType("fla.modules")
    fla_ops = types.ModuleType("fla.ops")
    fla_ops_delta_rule = types.ModuleType("fla.ops.delta_rule")
    fla_modules.ShortConvolution = _StubShortConvolution
    fla_modules.RMSNorm = _StubRMSNorm
    fla_ops_delta_rule.chunk_delta_rule = _stub_chunk_delta_rule
    fla_mod.modules = fla_modules
    fla_mod.ops = fla_ops
    fla_ops.delta_rule = fla_ops_delta_rule
    sys.modules["fla"] = fla_mod
    sys.modules["fla.modules"] = fla_modules
    sys.modules["fla.ops"] = fla_ops
    sys.modules["fla.ops.delta_rule"] = fla_ops_delta_rule
    return True


_STUB_INSTALLED = _ensure_fla_stub()

from lm_pretrain_rd import (                                               # noqa: E402
    CORPUS_DIRS, DEFAULT_DATA_DIR, FROZEN_BIAS_ARM_MODES, FROZEN_BIAS_LAMBDA_PRIMARY,
    _MIN_KERNEL_T, DeltaNetLM, get_batch, get_lr, load_corpus, set_and_log_tf32,
)
from mech_schema import wrap_exploratory                                    # noqa: E402

# ---------------------------------------------------------------------------
# H3 / gate text, pinned exactly once here (module docstring quotes the same
# text at length; these short constants are what gets stamped into every
# output JSON so a reader never has to cross-reference the design doc).
# ---------------------------------------------------------------------------
H3_HYPOTHESIS_VERBATIM = (
    "H3 | Optimizer/gradient-flow interaction: the lambda-blend's normalize() "
    "Jacobian, evaluated at a per-token-random vs. a fixed-global bias "
    "direction, differentially reshapes the gradient signal reaching k_raw, "
    "and this difference compounds over training. | Backward-hook "
    "gradient-norm/variance telemetry on k_raw, short instrumented runs."
)
GATE_RESULT_SEC_12_11_4_VERBATIM = (
    "Stage-2 gate (sec 12.5's frozen rule): FULL 20,000-step branch. Arm2: "
    "sign(delta@5000)=sign(delta@20000) [pass] and |-0.0721| >= "
    "0.5*|-0.0903|=0.0451 [pass] -> full. Arm2': sign [pass] but |-0.0570| < "
    "0.5*|-0.1203|=0.0601 -> truncated. Per the rule's own clause (\"if the "
    "two comparisons select different branches, the full 20,000-step branch "
    "governs for both\"): Stage 2 is authorized at full 20,000 steps "
    "(~0.76 GPU-h + the mandatory CPU smoke, sec 12.5)."
)

# ---------------------------------------------------------------------------
# Architecture -- IDENTICAL to rung-1's own calibrated cell (sec 12.5:
# "identical architecture/lr/corpus/seed convention to rung-1's own
# calibrated cell, differing only in step count and the added hook").
# Values cloned from frozen_bias_lm_sweep.py's own RUNG1_CFG/SEQ_LEN/
# BATCH_SIZE -- that file remains the source of truth; a future drift there
# should be reconciled here deliberately, not silently.
# ---------------------------------------------------------------------------
RUNG1_CFG = {"d_model": 256, "d_state": 64, "n_layers": 2, "conv_size": 4, "num_heads": 1}
SEQ_LEN = 512
BATCH_SIZE = 32
GRADFLOW_CADENCE_DEFAULT = 100   # sec 12.5: "at a fixed cadence (every 100 steps)"
ARMS_DEFAULT = ("off", "per_token", "global")


# ---------------------------------------------------------------------------
# THE hook. One function, used identically by the real Stage-2 cells
# (run_cell, below) and the CPU smoke (run_smoke, below) -- never a
# smoke-only stand-in copy (sec 12.9 item 4's own gap, closed by
# construction: same Python function object in both places).
# ---------------------------------------------------------------------------

def register_kraw_gradnorm_hooks(model: DeltaNetLM, telemetry: dict, cadence: int,
                                  step_box: dict) -> list:
    """Registers ONE `register_forward_hook` per layer's `mixer.k_conv1d` --
    the real, unmodified submodule whose output IS `k_raw` (lm_pretrain_rd.py
    line ~838, `k, _ = self.k_conv1d(self.k_proj(x))`), for EVERY
    frozen_bias_arm value uniformly (k_conv1d's own computation never reads
    frozen_bias_arm). On cadence-eligible steps only (`step_box["step"] %
    cadence == 0`), the forward-hook callback calls `register_hook` on the
    conv's own output tensor -- captures the REAL gradient reaching that
    exact point during the upcoming `.backward()`, strictly READ-ONLY (the
    closure returns None, never replaces the gradient -- verified by
    smoke item [3]'s torch.equal-level loss-trajectory check).

    telemetry: dict[int, list[dict]], mutated in place -- telemetry[layer_idx]
    accumulates {"step": int, "grad_norm": float} records in step order.
    step_box: a one-key mutable dict {"step": int} the CALLER updates every
    training-loop iteration BEFORE the forward pass -- required because
    `register_forward_hook`'s own callback signature (module, inputs,
    output) carries no step argument, and re-registering a fresh hook every
    step would defeat item [1]'s "registered ONCE" assertion.

    Returns the list of registered hook handles (len == model.n_layers) --
    the caller must call `.remove()` on each when done (both run_cell and
    run_smoke do this explicitly, never relying on garbage collection)."""
    handles = []
    for layer_idx, block in enumerate(model.blocks):
        def _fwd_hook(module, inputs, output, _layer_idx=layer_idx):
            step = step_box["step"]
            if cadence <= 0 or step % cadence != 0:
                return
            k_raw = output[0] if isinstance(output, tuple) else output

            def _bwd_hook(grad, _layer_idx=_layer_idx, _step=step):
                gn = float(grad.detach().float().norm().item())
                telemetry.setdefault(_layer_idx, []).append({"step": _step, "grad_norm": gn})
                return None   # strictly observational -- NEVER replace the real gradient

            k_raw.register_hook(_bwd_hook)

        handles.append(block.mixer.k_conv1d.register_forward_hook(_fwd_hook))
    assert len(handles) == len(model.blocks), (
        f"expected one hook handle per layer ({len(model.blocks)}), got {len(handles)}")
    return handles


def summarize_layer_series(records: list, total_steps: int) -> dict:
    """mean/median/std over ALL captured points, plus an early-vs-late
    windowed split (sec 12.5: "summary stats ... early-vs-late windows").
    sec 12.5 does not pin an exact window boundary -- this build's own
    operationalization, flagged for audit scrutiny (mirrors this codebase's
    own convention of flagging open interpretive calls, e.g. model_rd.py's
    geo3 naive_window docstring): early = step <= total_steps//4, late =
    step > 3*total_steps//4 (the same quartile split sec 12.4.1's own
    5-point trajectory grid brackets, generalized to this cadence-dense
    series)."""
    if not records:
        return {"n": 0, "all": {"n": 0, "mean": None, "median": None, "std": None},
                "early_window": {"n": 0, "mean": None, "median": None, "std": None},
                "late_window": {"n": 0, "mean": None, "median": None, "std": None},
                "early_window_step_cutoff": total_steps // 4,
                "late_window_step_cutoff": (3 * total_steps) // 4}
    steps = [r["step"] for r in records]
    norms = [r["grad_norm"] for r in records]
    early_cut = total_steps // 4
    late_cut = (3 * total_steps) // 4

    def _stats(xs: list) -> dict:
        if not xs:
            return {"n": 0, "mean": None, "median": None, "std": None}
        t = torch.tensor(xs, dtype=torch.float64)
        return {"n": len(xs), "mean": t.mean().item(), "median": t.median().item(),
                "std": (t.std(unbiased=False).item() if len(xs) > 1 else 0.0)}

    early = [n for s, n in zip(steps, norms) if s <= early_cut]
    late = [n for s, n in zip(steps, norms) if s > late_cut]
    return {"n": len(records), "all": _stats(norms), "early_window": _stats(early),
            "late_window": _stats(late), "early_window_step_cutoff": early_cut,
            "late_window_step_cutoff": late_cut}


def h3_gradnorm_comparison(arms_result: dict, n_layers: int) -> dict:
    """The pre-registered H3 comparison, per-layer: Arm2-vs-Arm1 (per_token
    minus off) and Arm2'-vs-Arm1 (global minus off) grad-norm deltas, at the
    all/early/late windows. H3's own sec 12.2 text registers a MECHANISM
    ("differentially reshapes the gradient signal ... compounds over
    training"), not a predicted SIGN -- unlike H1/H2's explicit "Arm2 lower /
    Arm2' higher" direction. No `direction_consistent_with_hypothesis` field
    is manufactured here for a prediction sec 12.2 never registered (that
    field only ever appears where derive_estimation() itself produces one --
    mech_schema.wrap_exploratory renames it if present, it does not invent
    it). These deltas are reported DESCRIPTIVELY, correlationally (sec 12.9
    item 1's standing ceiling)."""
    per_layer = {}
    for i in range(n_layers):
        li = str(i)
        off_s = arms_result["off"]["grad_norm_summary_by_layer"][li]
        pt_s = arms_result["per_token"]["grad_norm_summary_by_layer"][li]
        gl_s = arms_result["global"]["grad_norm_summary_by_layer"][li]

        def _delta(a: dict, b: dict, window: str):
            av = a.get(window, {}).get("mean")
            bv = b.get(window, {}).get("mean")
            return (av - bv) if (av is not None and bv is not None) else None

        per_layer[li] = {
            "per_token_minus_off": {"all": _delta(pt_s, off_s, "all"),
                                     "early": _delta(pt_s, off_s, "early_window"),
                                     "late": _delta(pt_s, off_s, "late_window")},
            "global_minus_off": {"all": _delta(gl_s, off_s, "all"),
                                  "early": _delta(gl_s, off_s, "early_window"),
                                  "late": _delta(gl_s, off_s, "late_window")},
        }
    return {
        "note": ("H3 registers a MECHANISM, not a predicted sign (unlike H1/H2) -- these "
                 "deltas are descriptive/correlational (sec 12.9 item 1), no headline verdict."),
        "per_layer": per_layer,
    }


# ---------------------------------------------------------------------------
# Cloned training loop (sec 12.5: "clone the training loop"). Does NOT call
# lm_pretrain_rd.train() -- that function's own eval/rank-stat/checkpoint/
# geo3/hard-select machinery is Stage 1's concern (already-existing
# checkpoints, forward-pass only), not Stage 2's (short NEW instrumented
# training, telemetry-only deliverable). Reuses the shared, unmodified
# primitives (get_batch, get_lr, load_corpus, set_and_log_tf32) exactly as
# frozen_bias_retrofit_eval_rd.py's own import block already establishes is
# this codebase's convention for THOSE functions specifically (only the
# orchestration loop itself is cloned, never the model or its shared utils).
# ---------------------------------------------------------------------------

def run_cell(arm: str, args: argparse.Namespace, train_tokens: torch.Tensor, vocab_size: int,
             device: str) -> dict:
    assert arm in FROZEN_BIAS_ARM_MODES, f"unknown arm {arm!r}, expected one of {FROZEN_BIAS_ARM_MODES}"
    torch.manual_seed(args.seed)
    model = DeltaNetLM(vocab_size, d_model=args.d_model, d_state=args.d_state,
                        n_layers=args.n_layers, conv_size=args.conv_size, num_heads=args.num_heads,
                        frozen_bias_arm=arm, frozen_bias_lambda=args.lam,
                        frozen_bias_vocab_size=vocab_size).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    telemetry: dict = {i: [] for i in range(args.n_layers)}
    step_box = {"step": 0}
    handles = register_kraw_gradnorm_hooks(model, telemetry, args.gradflow_cadence, step_box)

    gen = torch.Generator(device=device).manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    t0 = time.time()
    trajectory = []
    n_skipped = 0
    for step in range(1, args.steps + 1):
        step_box["step"] = step
        lr = get_lr(step, args.lr, args.warmup_steps, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_tokens, args.batch_size, args.seq_len, gen)
        logits = model(x, step=step)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1

        if step % args.log_every == 0 or step == 1:
            trajectory.append({"step": step, "loss": loss.item(), "lr": lr, "grad_finite": finite})
            print(f"  [{arm}] step {step:6d}  loss {loss.item():.4f}  lr {lr:.2e}"
                  f"{'  [NON-FINITE GRAD, SKIPPED]' if not finite else ''}", flush=True)

    for h in handles:
        h.remove()
    wall_s = time.time() - t0

    layer_summary = {str(i): summarize_layer_series(telemetry[i], args.steps)
                      for i in range(args.n_layers)}
    return {
        "arm": arm, "lam": args.lam if arm != "off" else None, "n_params": n_params,
        "steps": args.steps, "steps_completed": args.steps, "n_skipped": n_skipped,
        "wall_s": wall_s, "final_loss": (trajectory[-1]["loss"] if trajectory else None),
        "loss_trajectory": trajectory,
        "grad_norm_by_layer_raw": {str(i): telemetry[i] for i in range(args.n_layers)},
        "grad_norm_summary_by_layer": layer_summary,
    }


def run_all_cells(args: argparse.Namespace) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", (
        "frozen_bias_gradflow_probe's REAL cells require CUDA (chunk_delta_rule has no CPU "
        "path -- lm_pretrain_rd.py's own smoke() assert restates this exactly). Use --smoke "
        "for the CPU-runnable gate instead.")
    tf32_state = set_and_log_tf32()
    train_tokens, _, meta, _, _ = load_corpus(args.data_dir, args.corpus, device)
    vocab_size = meta["vocab_size"]

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        assert a in FROZEN_BIAS_ARM_MODES, f"unknown arm {a!r} in --arms, expected one of {FROZEN_BIAS_ARM_MODES}"

    results = {}
    for arm in arms:
        print(f"\n{'=' * 70}\n  CELL: arm={arm}  corpus={args.corpus}  seed={args.seed}  "
              f"steps={args.steps}\n{'=' * 70}", flush=True)
        results[arm] = run_cell(arm, args, train_tokens, vocab_size, device)

    payload = {
        "design_ref": "FROZEN_BIAS_LM_DESIGN.md sec 12.5 (Stage 2, H3)",
        "h3_hypothesis_verbatim": H3_HYPOTHESIS_VERBATIM,
        "gate_result_sec_12_11_4_verbatim": GATE_RESULT_SEC_12_11_4_VERBATIM,
        "corpus": args.corpus, "seed": args.seed, "steps": args.steps,
        "gradflow_cadence": args.gradflow_cadence,
        "architecture": {"d_model": args.d_model, "d_state": args.d_state, "n_layers": args.n_layers,
                          "conv_size": args.conv_size, "num_heads": args.num_heads,
                          "seq_len": args.seq_len, "batch_size": args.batch_size},
        "lr": args.lr, "warmup_steps": args.warmup_steps, "weight_decay": args.weight_decay,
        "lambda": args.lam, "tf32_state": tf32_state,
        "arms": results,
    }
    if all(a in results for a in ("off", "per_token", "global")):
        payload["h3_gradnorm_comparison"] = h3_gradnorm_comparison(results, args.n_layers)
    return wrap_exploratory(payload)


# ---------------------------------------------------------------------------
# MANDATORY smoke (sec 12.5's three-part spec). CPU-runnable: uses the fla
# stub (installed above) so it completes without the real Triton kernel/GPU.
# ---------------------------------------------------------------------------

_SMOKE_VOCAB, _SMOKE_D_MODEL, _SMOKE_D_STATE, _SMOKE_N_LAYERS = 200, 64, 64, 2
_SMOKE_SEQ_LEN, _SMOKE_BATCH = _MIN_KERNEL_T, 4
_SMOKE_INIT_SEED, _SMOKE_DATA_SEED = 1234, 999


def _tiny_model(arm: str) -> DeltaNetLM:
    return DeltaNetLM(_SMOKE_VOCAB, d_model=_SMOKE_D_MODEL, d_state=_SMOKE_D_STATE,
                       n_layers=_SMOKE_N_LAYERS, conv_size=4, frozen_bias_arm=arm,
                       frozen_bias_lambda=FROZEN_BIAS_LAMBDA_PRIMARY,
                       frozen_bias_vocab_size=_SMOKE_VOCAB)


def run_smoke() -> bool:
    print("=" * 70)
    print("  FROZEN_BIAS_GRADFLOW_PROBE SMOKE GATE (sec 12.5)")
    print("=" * 70)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  fla_stub_installed={_STUB_INSTALLED}", flush=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    ok = True

    # -----------------------------------------------------------------
    # [1]+[2]: hook mechanics on arm="off" -- the hardest case, since
    # DeltaNetLMMixer.forward NEVER calls apply_frozen_bias_blend for this
    # arm (line ~854's guard). If the hook only worked by wrapping that
    # function, arm="off" would silently capture NOTHING. Hooking
    # k_conv1d's own output instead must work identically here.
    # -----------------------------------------------------------------
    print("\n[1] hook fires exactly once per forward pass per layer, ONLY on cadence-eligible "
          "steps (arm='off' -- the case a naive apply_frozen_bias_blend-wrapping monkeypatch "
          "would silently miss entirely)")
    torch.manual_seed(_SMOKE_INIT_SEED)
    model_hooked = _tiny_model("off").to(device)
    torch.manual_seed(_SMOKE_INIT_SEED)
    model_control = _tiny_model("off").to(device)
    for (n1, p1), (n2, p2) in zip(model_hooked.named_parameters(), model_control.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), (
            f"smoke setup bug: parameter {n1!r} differs between hooked/control models at init -- "
            f"the loss-trajectory-equality check below would be meaningless without identical init")

    telemetry: dict = {i: [] for i in range(_SMOKE_N_LAYERS)}
    step_box = {"step": 0}
    cadence = 2
    handles = register_kraw_gradnorm_hooks(model_hooked, telemetry, cadence, step_box)
    assert len(handles) == _SMOKE_N_LAYERS, f"expected {_SMOKE_N_LAYERS} handles, got {len(handles)}"

    n_smoke_steps = 6
    tokens = torch.randint(0, _SMOKE_VOCAB, (10_000,), device=device,
                            generator=torch.Generator(device=device).manual_seed(7))
    gen_hooked = torch.Generator(device=device).manual_seed(_SMOKE_DATA_SEED)
    gen_control = torch.Generator(device=device).manual_seed(_SMOKE_DATA_SEED)
    opt_hooked = torch.optim.AdamW(model_hooked.parameters(), lr=3e-4)
    opt_control = torch.optim.AdamW(model_control.parameters(), lr=3e-4)

    losses_hooked, losses_control = [], []
    for step in range(1, n_smoke_steps + 1):
        step_box["step"] = step
        x, y = get_batch(tokens, _SMOKE_BATCH, _SMOKE_SEQ_LEN, gen_hooked)
        x2, y2 = get_batch(tokens, _SMOKE_BATCH, _SMOKE_SEQ_LEN, gen_control)
        assert torch.equal(x, x2) and torch.equal(y, y2), (
            "smoke setup bug: hooked/control batch samplers diverged despite identical seeds")

        logits_h = model_hooked(x, step=step)
        loss_h = F.cross_entropy(logits_h.reshape(-1, _SMOKE_VOCAB), y.reshape(-1))
        opt_hooked.zero_grad()
        loss_h.backward()
        opt_hooked.step()
        losses_hooked.append(loss_h.item())

        logits_c = model_control(x2, step=step)
        loss_c = F.cross_entropy(logits_c.reshape(-1, _SMOKE_VOCAB), y2.reshape(-1))
        opt_control.zero_grad()
        loss_c.backward()
        opt_control.step()
        losses_control.append(loss_c.item())

    for h in handles:
        h.remove()

    expected_cadence_steps = [s for s in range(1, n_smoke_steps + 1) if s % cadence == 0]
    ok_1 = True
    for layer_idx in range(_SMOKE_N_LAYERS):
        recorded_steps = [rec["step"] for rec in telemetry[layer_idx]]
        this_ok = recorded_steps == expected_cadence_steps
        n_dupes = len(recorded_steps) - len(set(recorded_steps))
        print(f"  layer {layer_idx}: recorded_steps={recorded_steps}  "
              f"expected={expected_cadence_steps}  duplicates={n_dupes}  PASS={this_ok}")
        ok_1 = ok_1 and this_ok and (n_dupes == 0)
    print(f"[1] PASS={ok_1}")
    ok = ok and ok_1

    print("\n[2] captured grad norms finite and nonzero")
    ok_2 = True
    for layer_idx in range(_SMOKE_N_LAYERS):
        norms = [rec["grad_norm"] for rec in telemetry[layer_idx]]
        this_ok = len(norms) > 0 and all(math.isfinite(g) and g > 0.0 for g in norms)
        print(f"  layer {layer_idx}: grad_norms={norms}  PASS={this_ok}")
        ok_2 = ok_2 and this_ok
    print(f"[2] PASS={ok_2}")
    ok = ok and ok_2

    print("\n[3] adding the hook does NOT change the loss trajectory vs an unhooked control "
          "(torch.equal-level check, same seed/init/data)")
    t_hooked = torch.tensor(losses_hooked, dtype=torch.float64)
    t_control = torch.tensor(losses_control, dtype=torch.float64)
    ok_3 = bool(torch.equal(t_hooked, t_control))
    print(f"  losses_hooked ={losses_hooked}")
    print(f"  losses_control={losses_control}")
    print(f"[3] PASS={ok_3}")
    ok = ok and ok_3

    # -----------------------------------------------------------------
    # [4] generality: the SAME register_kraw_gradnorm_hooks() function also
    # captures finite/nonzero telemetry for arm="per_token" and arm="global"
    # -- the two arms where apply_frozen_bias_blend IS actually invoked, so
    # this closes the loop on all 3 registered Stage-2 cells' arms.
    # -----------------------------------------------------------------
    print("\n[4] hook mechanism generalizes to arm='per_token' and arm='global' "
          "(the two arms where apply_frozen_bias_blend actually runs)")
    ok_4 = True
    for arm in ("per_token", "global"):
        torch.manual_seed(_SMOKE_INIT_SEED)
        m = _tiny_model(arm).to(device)
        tel: dict = {i: [] for i in range(_SMOKE_N_LAYERS)}
        sb = {"step": 0}
        hs = register_kraw_gradnorm_hooks(m, tel, 1, sb)
        gg = torch.Generator(device=device).manual_seed(_SMOKE_DATA_SEED)
        for step in (1, 2):
            sb["step"] = step
            xx, yy = get_batch(tokens, _SMOKE_BATCH, _SMOKE_SEQ_LEN, gg)
            ll = F.cross_entropy(m(xx, step=step).reshape(-1, _SMOKE_VOCAB), yy.reshape(-1))
            m.zero_grad()
            ll.backward()
        for h in hs:
            h.remove()
        this_ok = all(
            len(tel[i]) == 2 and all(math.isfinite(r["grad_norm"]) and r["grad_norm"] > 0.0
                                       for r in tel[i])
            for i in range(_SMOKE_N_LAYERS)
        )
        print(f"  arm={arm}: per-layer records={ {i: tel[i] for i in range(_SMOKE_N_LAYERS)} }  "
              f"PASS={this_ok}")
        ok_4 = ok_4 and this_ok
    print(f"[4] PASS={ok_4}")
    ok = ok and ok_4

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                     help="sec 12.5's mandatory Stage-2 smoke -- CPU-runnable, run this before "
                          "any real cell.")
    ap.add_argument("--corpus", choices=sorted(CORPUS_DIRS), default="openr1-mix-ext")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20000,
                     help="sec 12.11.4's gate resolved to the FULL 20,000-step branch -- this is "
                          "the registered default, NOT the cheaper 3,000-step branch (also "
                          "specified in sec 12.5 but not selected).")
    ap.add_argument("--arms", type=str, default=",".join(ARMS_DEFAULT),
                     help="comma-separated subset of {off,per_token,global}.")
    ap.add_argument("--lam", type=float, default=FROZEN_BIAS_LAMBDA_PRIMARY)
    ap.add_argument("--gradflow-cadence", type=int, default=GRADFLOW_CADENCE_DEFAULT)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--d-model", type=int, default=RUNG1_CFG["d_model"])
    ap.add_argument("--d-state", type=int, default=RUNG1_CFG["d_state"], choices=[64, 128])
    ap.add_argument("--n-layers", type=int, default=RUNG1_CFG["n_layers"])
    ap.add_argument("--conv-size", type=int, default=RUNG1_CFG["conv_size"])
    ap.add_argument("--num-heads", type=int, default=RUNG1_CFG["num_heads"])
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        ok = run_smoke()
        print("\n" + "=" * 70)
        print("FROZEN_BIAS_GRADFLOW_PROBE SMOKE: " + ("ALL PASSED" if ok else "FAILURES PRESENT"))
        print("=" * 70)
        sys.exit(0 if ok else 1)

    assert args.seq_len >= _MIN_KERNEL_T, f"--seq-len={args.seq_len} < _MIN_KERNEL_T={_MIN_KERNEL_T}"
    result = run_all_cells(args)
    summary = {k: v for k, v in result.items() if k != "arms"}
    print("\nRESULT SUMMARY:", json.dumps(summary, indent=2, default=str), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
