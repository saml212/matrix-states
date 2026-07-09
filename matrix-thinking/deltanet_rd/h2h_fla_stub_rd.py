"""h2h_fla_stub_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md build-stage shared CPU
stub installer. Every H2H module that (transitively, via `lm_pretrain_rd`/
`model_rd`) touches `fla` calls `ensure_fla_stub()` FIRST, exactly once per
process, before importing anything from those files -- mirrors
`reasoning_link_probe.py`'s own `_ensure_fla_stub` (lines 108-205), reusing
its "k-ROUTING FIXED" stub content byte-for-byte (the one good stub in this
codebase per that file's own comment: `o` genuinely depends on `k` via a
cheap `sigmoid((q*k).sum(-1))` gate, unlike the sibling smoke files' plain
`o = v * sigmoid(beta)`-only stub, which has no k-dependency at all).
Factored into its OWN module here (rather than each new H2H file hand-
copying the ~70-line stub, which is how the existing codebase does it) so
this build's several new fla-touching files (`ablation_mixer_rd.py`,
`transformer_baseline_rd.py` via its `FFN` import, `probe_head_rd.py`,
`verify_match_gate.py`) share ONE stub implementation -- a real DRY win the
CLAUDE.md-mandated `clean` audit rewards, and it keeps every H2H CPU smoke
test byte-identical in what "the stub" means.

Gate: `REASONING_LINK_FORCE_CPU_STUB=1` (the SAME env var name
`reasoning_link_probe.py` already uses -- one process-wide toggle across the
whole deltanet_rd test suite, not a new H2H-specific name) forces the stub
even when real `fla` is importable (the on-box CPU-only-suite override that
file's own comment documents: real fla exists on the H100 box, but Triton
kernels reject CPU tensors, so CPU-only suites must force the stub there
too). Absent that override, this module auto-falls-back to the stub only
when `import fla` genuinely fails (the plain dev-box case, no env var
needed) -- matching `smoke_frozen_bias_lm.py`'s simpler unconditional
try/except convention for suites that never run under real fla at all.

MUST be called before `from lm_pretrain_rd import ...` / `from model_rd
import ...` -- those files execute their own top-level `from fla...import`
lines immediately on import."""
from __future__ import annotations

import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

_STUB_MARKER = "_H2H_CPU_STUB"   # mirrors reasoning_link_probe.py's own _REASONING_LINK_CPU_STUB
                                  # marker-attribute pattern (MINOR-3 fix): import-success/failure
                                  # alone cannot distinguish "real fla" from "a stub some earlier
                                  # execution in this process already installed."


def ensure_fla_stub() -> bool:
    """Installs a CPU-safe stand-in for `fla.modules.{ShortConvolution,RMSNorm}` and
    `fla.ops.delta_rule.chunk_delta_rule` into `sys.modules`, UNLESS real `fla` is importable and
    `REASONING_LINK_FORCE_CPU_STUB` is not set to "1". Returns True iff (this call OR an earlier
    call in this process) installed the stub, False iff real fla is in use. Idempotent: calling
    this more than once (e.g. from several H2H modules each importing this file) never re-installs
    or double-wraps -- the second call's `import fla` finds either the real package or the
    already-installed stub and returns the correct answer either way."""
    if os.environ.get("REASONING_LINK_FORCE_CPU_STUB", "0") != "1":
        try:
            import fla  # noqa: F401
            return bool(getattr(fla, _STUB_MARKER, False))
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
        """CPU-safe stand-in for the CUDA-only Triton kernel (confirmed CUDA-only this build:
        `chunk_delta_rule` has no CPU path). `o` genuinely depends on `k` (a cheap per-position
        gate) so any hook-based check expecting a causal/functional link to `k` is exercised
        meaningfully. `final_state` is a deliberately zero matrix -- box-only-verifiable items
        depending on a NONZERO, trained `S_T` cannot be given meaningful numbers by this stub;
        every H2H smoke item needing a real nonzero S_T is disclosed as box-only in this build's
        report (mirrors reasoning_link_probe.py's own disclosed stub limitation verbatim)."""
        B, T, H, Dh = q.shape
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        qk_gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        o = v * torch.sigmoid(beta).unsqueeze(-1) * qk_gate
        final_state = torch.zeros(B, H, Dh, Dh, dtype=q.dtype, device=q.device)
        return o, final_state

    def _stub_chunk_gated_delta_rule(q, k, v, g, beta, initial_state=None,
                                     output_final_state=True,
                                     use_qk_l2norm_in_kernel=True,
                                     use_beta_sigmoid_in_kernel=False):
        """CPU-safe stand-in for fla 0.5.1's chunk_gated_delta_rule (CUDA-only), mirroring
        _stub_chunk_delta_rule's disclosed limitations verbatim. `g` is the log-space decay
        gate (logsigmoid convention per the pinned wheel); the stub applies exp(g) as a cheap
        multiplicative dependence so gradient/plumbing checks exercise the g path. final_state
        is deliberately zero -- value-level claims are box-only, same as the ungated stub."""
        B, T, H, Dh = q.shape
        if use_qk_l2norm_in_kernel:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        b = torch.sigmoid(beta) if use_beta_sigmoid_in_kernel else beta
        qk_gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        o = v * b.unsqueeze(-1) * qk_gate * torch.exp(g).unsqueeze(-1)
        final_state = torch.zeros(B, H, Dh, Dh, dtype=q.dtype, device=q.device)
        return o, final_state

    fla_mod = types.ModuleType("fla")
    fla_modules = types.ModuleType("fla.modules")
    fla_ops = types.ModuleType("fla.ops")
    fla_ops_delta_rule = types.ModuleType("fla.ops.delta_rule")
    fla_ops_gated_delta_rule = types.ModuleType("fla.ops.gated_delta_rule")
    fla_modules.ShortConvolution = _StubShortConvolution
    fla_modules.RMSNorm = _StubRMSNorm
    fla_ops_delta_rule.chunk_delta_rule = _stub_chunk_delta_rule
    fla_ops_gated_delta_rule.chunk_gated_delta_rule = _stub_chunk_gated_delta_rule
    fla_mod.modules = fla_modules
    fla_mod.ops = fla_ops
    fla_ops.delta_rule = fla_ops_delta_rule
    fla_ops.gated_delta_rule = fla_ops_gated_delta_rule
    setattr(fla_mod, _STUB_MARKER, True)
    sys.modules["fla"] = fla_mod
    sys.modules["fla.modules"] = fla_modules
    sys.modules["fla.ops"] = fla_ops
    sys.modules["fla.ops.delta_rule"] = fla_ops_delta_rule
    sys.modules["fla.ops.gated_delta_rule"] = fla_ops_gated_delta_rule
    return True
