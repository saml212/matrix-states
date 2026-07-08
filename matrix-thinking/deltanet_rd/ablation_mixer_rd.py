"""ablation_mixer_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3(a): the
CLAUDE.md-mandated param-matched flat-VECTOR-state ablation, and axis 1's
baseline. Identical embedding table, output head, and FFN blocks to the
contender (`lm_pretrain_rd.DeltaNetLM`); the ONLY change is the fast-weight
mixer. Per the CLAUDE.md reshape-equivalence rule ("any d^2-dim vector can
be reshaped to a d x d matrix and vice versa; structure only matters if
OPERATIONS preserve it"), this does NOT reshape the contender's matrix
operator into a vector shape (that would silently preserve the structure it
is supposed to remove) -- it replaces the multiplicative outer-product state
UPDATE itself with an additive/gated one:

    s_t = s_{t-1} (.) g_t + v_t         s_t in R^{d_state}     (Hadamard gate)

`d_state` is PINNED EQUAL to the contender's own `d_state` (sec 1.3(a)'s
own deliberate Rev-1 choice, never left to fall out of width-matching).
Native-read parity (F1): the ablation gets its OWN `q_proj`/`q_conv1d`,
constructed IDENTICALLY to the contender's (same shapes/init/activation),
never entering the state-update recurrence -- read-only, mirroring the
P=1 hard-bottleneck discipline `CLAUDE.md` requires. The ablation's own
native per-token read, analogous to the contender's kernel-internal
`o_t = read(S_t, q_t)`, is the vector-state elementwise analogue:
`o_t = s_t (.) q_t` (Hadamard product -- the vector recurrence has no
off-diagonal structure to contract a matvec against).

Build decisions this file makes that the design doc leaves open (disclosed
here, for the independent audit to scrutinize):
  - `v_t` is derived through the SAME `v_proj + v_conv1d` short-causal-conv
    preprocessing pattern the contender uses for its own v-pathway (removes
    an asymmetry axis the design doc's own "remove asymmetry wherever there
    is a free choice" logic already applies elsewhere, e.g. the shared FFN
    class / pre-norm convention, sec 1.3(b)).
  - `g_t = sigmoid(g_proj(x_t))`, a PLAIN linear-then-sigmoid gate with NO
    conv -- mirrors the contender's own `beta = sigmoid(b_proj(a))`
    construction (the closest true analogue: both are per-position
    CONTROL/gating signals, not content signals like k/v/q, so no-conv
    keeps the two architectures' gating pathways structurally parallel).
  - The recurrence is computed via a plain SEQUENTIAL Python-level loop
    over T (not a chunked/vectorized parallel scan). This is a deliberate
    correctness-first, easy-to-audit choice: a chunked log-domain scan is
    the standard efficient implementation for a diagonal linear recurrence,
    but its numerically stable form requires bounding intra-chunk decay
    depth (a real engineering task in its own right, flagged here as an
    explicit, NOT-built-this-wave follow-on) -- sequential accumulation has
    ZERO numerical-stability risk (every step is one bounded
    elementwise-multiply-add, `g_t` in (0,1) via sigmoid), at the cost of
    O(T) Python-level steps per forward call. Axis 1 (this arm's own use)
    only ever evaluates sequences around `T_bind + query_len` at the
    primary load K=32 (~230 tokens, `grammar_rd.DeltaNetRDTaskConfig`) --
    the ablation is NEVER evaluated at axis 2's long horizons (H2/H4/H8,
    up to 1798 tokens; axis 2 compares contender vs. Transformer only,
    sec 1.4.2), so this tradeoff is scoped to where it is actually cheap.

Parameter-matching arithmetic (verified by `verify_match_gate.py`, sec
1.7 gate 6, not assumed here): at rung-1 (d_model=256, d_state=64,
conv_size=4), this mixer's natural construction below (q_proj+v_proj+
o_proj+g_proj+q_conv1d+v_conv1d+o_norm) already lands within <0.01% of the
contender's own mixer param count (66,112 vs. 66,624 -- a 512-param gap,
negligible against the ~14M-param whole model) with NO extra width knob
needed. `gate_extra_width` below exists as a documented, currently-inert
(default 0) escape hatch for a FUTURE rung where the natural gap might
exceed the sec 1.7 <=1% tolerance -- sec 1.3(a)'s own instruction that any
matching slack be "absorbed by the mixer's OTHER weights (the gate
projection producing g_t), never by d_state itself."

Run the smoke gate: python ablation_mixer_rd.py --smoke
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

from h2h_fla_stub_rd import ensure_fla_stub

_STUB_INSTALLED = ensure_fla_stub()
from fla.modules import RMSNorm, ShortConvolution  # noqa: E402


def _diagonal_linear_recurrence(v: torch.Tensor, g: torch.Tensor,
                                 initial_state: torch.Tensor | None = None) -> torch.Tensor:
    """s_t = g_t (.) s_{t-1} + v_t, s_t in R^{D}, computed by a plain
    sequential Python-level loop over T (see module docstring for the
    correctness-vs-efficiency tradeoff this makes explicitly). v,g:
    (B,T,D). initial_state: None (s_0 implicitly the zero vector -- the
    ONLY mode training ever uses, mirrors DeltaNetLM's own convention) or
    (B,D). Returns s: (B,T,D), every position's post-update state (the
    LAST position, s[:,-1,:], is this mixer's own `final_state`)."""
    B, T, D = v.shape
    s_prev = (torch.zeros(B, D, dtype=v.dtype, device=v.device)
              if initial_state is None else initial_state)
    outs = []
    for t in range(T):
        s_prev = g[:, t, :] * s_prev + v[:, t, :]
        outs.append(s_prev)
    return torch.stack(outs, dim=1)


class AblationLMMixer(nn.Module):
    """Flat-vector-state gated-linear mixing sublayer -- axis 1's baseline
    arm's ONLY architectural difference from the contender (sec 1.3(a))."""

    def __init__(self, d_model: int, d_state: int, conv_size: int = 4,
                 gate_extra_width: int = 0):
        super().__init__()
        assert gate_extra_width >= 0, f"gate_extra_width={gate_extra_width} must be >= 0"
        self.d_model = d_model
        self.d_state = d_state
        self.conv_size = conv_size

        # Native-read parity (F1): IDENTICAL construction to the contender's own q_proj/q_conv1d
        # (lm_pretrain_rd.DeltaNetLMMixer.__init__, same shapes/bias/activation).
        self.q_proj = nn.Linear(d_model, d_state, bias=False)
        self.q_conv1d = ShortConvolution(hidden_size=d_state, kernel_size=conv_size, bias=False,
                                          activation="silu")

        self.v_proj = nn.Linear(d_model, d_state, bias=False)
        self.v_conv1d = ShortConvolution(hidden_size=d_state, kernel_size=conv_size, bias=False,
                                          activation="silu")

        self.g_proj = nn.Linear(d_model, d_state, bias=False)
        # Inert (no-op) matching-slack knob, see module docstring -- when 0 (default), adds
        # nothing; when > 0, an EXTRA linear pair widens the gate pathway's own param count without
        # touching d_state, exactly the sec 1.3(a)-mandated absorption point.
        self.gate_extra_width = gate_extra_width
        if gate_extra_width > 0:
            self.g_extra_in = nn.Linear(d_model, gate_extra_width, bias=False)
            self.g_extra_out = nn.Linear(gate_extra_width, d_state, bias=False)
        else:
            self.g_extra_in = None
            self.g_extra_out = None

        self.o_norm = RMSNorm(d_state, eps=1e-5)
        self.o_proj = nn.Linear(d_state, d_model, bias=False)

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor | None = None,
                token_ids: torch.Tensor | None = None, step: int | None = None):
        """Signature mirrors DeltaNetLMMixer.forward's own
        (x, initial_state, token_ids, step) contract for drop-in symmetry
        in the 27-cell sweep runner -- token_ids/step are accepted and
        unused (this arm has no EOT-exclusion/continuation machinery)."""
        del token_ids, step
        B, T, _ = x.shape
        v, _ = self.v_conv1d(self.v_proj(x))
        gate_logit = self.g_proj(x)
        if self.gate_extra_width > 0:
            gate_logit = gate_logit + self.g_extra_out(F.silu(self.g_extra_in(x)))
        g = torch.sigmoid(gate_logit)
        q, _ = self.q_conv1d(self.q_proj(x))     # read-only: NEVER feeds the recurrence below

        s = _diagonal_linear_recurrence(v, g, initial_state)   # (B,T,D)
        o = s * q                                                # Hadamard native read
        o = self.o_norm(o)
        o = self.o_proj(o)
        final_state = s[:, -1, :]
        return o, final_state


class AblationLMBlock(nn.Module):
    """Pre-norm mixing sublayer + pre-norm FFN sublayer -- structurally
    identical to DeltaNetLMBlock (same RMSNorm pre-norm convention, same
    FFN class instance, reused verbatim, not reimplemented)."""

    def __init__(self, d_model: int, d_state: int, conv_size: int = 4, ffn_mult: int = 4,
                 gate_extra_width: int = 0):
        super().__init__()
        from lm_pretrain_rd import FFN  # noqa: local import, after ensure_fla_stub()
        self.norm1 = RMSNorm(d_model, eps=1e-5)
        self.mixer = AblationLMMixer(d_model, d_state, conv_size=conv_size,
                                      gate_extra_width=gate_extra_width)
        self.norm2 = RMSNorm(d_model, eps=1e-5)
        self.ffn = FFN(d_model, mult=ffn_mult)

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor | None = None,
                token_ids: torch.Tensor | None = None, step: int | None = None):
        o, final_state = self.mixer(self.norm1(x), initial_state=initial_state,
                                     token_ids=token_ids, step=step)
        x = x + o
        x = x + self.ffn(self.norm2(x))
        return x, final_state


class AblationLM(nn.Module):
    """Top-level model -- mirrors DeltaNetLM's own embed/blocks/norm_f/
    tied-head structure and forward(token_ids, initial_states, return_states)
    contract EXACTLY (same call signature, same return shape), so the
    27-cell sweep runner and the shared probe head can treat this arm as a
    drop-in for the contender wherever the API is concerned -- only the
    per-layer mixer differs."""

    def __init__(self, vocab_size: int, d_model: int = 256, d_state: int = 64, n_layers: int = 2,
                 conv_size: int = 4, ffn_mult: int = 4, gate_extra_width: int = 0):
        super().__init__()
        assert n_layers >= 1, f"n_layers={n_layers} must be >= 1"
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)   # AUDIT FIX-2 convention, contender-identical
        self.blocks = nn.ModuleList([
            AblationLMBlock(d_model, d_state, conv_size=conv_size, ffn_mult=ffn_mult,
                             gate_extra_width=gate_extra_width)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model, eps=1e-5)

    def forward(self, token_ids: torch.Tensor, initial_states: list | None = None,
                return_states: bool = False, step: int | None = None):
        B, T = token_ids.shape
        x = self.embed(token_ids)
        if initial_states is None:
            initial_states = [None] * self.n_layers
        assert len(initial_states) == self.n_layers, (
            f"initial_states must have one entry per layer ({self.n_layers}), "
            f"got {len(initial_states)}")
        final_states = []
        for blk, s0 in zip(self.blocks, initial_states):
            x, s_final = blk(x, initial_state=s0, token_ids=token_ids, step=step)
            final_states.append(s_final)
        x = self.norm_f(x)
        logits = F.linear(x, self.embed.weight)
        return (logits, final_states) if return_states else logits


def count_mixer_params(d_model: int, d_state: int, conv_size: int = 4,
                        gate_extra_width: int = 0) -> int:
    """Standalone (no model instantiation) param-count formula for ONE
    AblationLMMixer -- verify_match_gate.py's Pass 2 (independent
    re-derivation) re-derives this from scratch rather than importing it;
    kept here as Pass 1's own quick-check helper."""
    n = 4 * d_model * d_state          # q_proj, v_proj, o_proj, g_proj (each d_model<->d_state)
    n += 2 * d_state * conv_size        # q_conv1d, v_conv1d (depthwise, no bias)
    n += d_state                        # o_norm weight
    if gate_extra_width > 0:
        n += d_model * gate_extra_width + gate_extra_width * d_state
    return n


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def smoke_1_forward_backward_grad_finite():
    torch.manual_seed(101)
    m = AblationLM(500, d_model=32, d_state=16, n_layers=2, conv_size=4)
    x = torch.randint(0, 500, (3, 20))
    y = torch.randint(0, 500, (3, 20))
    logits = m(x)
    finite_fwd = torch.isfinite(logits).all().item()
    loss = F.cross_entropy(logits.reshape(-1, 500), y.reshape(-1))
    loss.backward()
    n_none = [n for n, p in m.named_parameters() if p.grad is None]
    finite_bwd = not n_none and all(torch.isfinite(p.grad).all() for p in m.parameters())
    ok = bool(finite_fwd) and bool(finite_bwd) and logits.shape == (3, 20, 500)
    _report("smoke 1: forward/backward/grad-finite (tiny CPU shapes)", ok,
            f"logits.shape={tuple(logits.shape)} finite_fwd={finite_fwd} finite_bwd={finite_bwd} "
            f"no_grad_params={n_none}")


def smoke_2_recurrence_matches_hand_reference():
    """Independent, hand-rolled reference of _diagonal_linear_recurrence
    over a tiny (B=2,T=5,D=3) case, computed with a DIFFERENT (nested-list,
    no torch broadcasting tricks) implementation -- proves the vectorized
    batch/channel broadcasting inside the sequential loop is not silently
    scrambling axes."""
    torch.manual_seed(7)
    B, T, D = 2, 5, 3
    v = torch.randn(B, T, D)
    g = torch.sigmoid(torch.randn(B, T, D))
    s = _diagonal_linear_recurrence(v, g)

    ref = torch.zeros(B, T, D)
    for b in range(B):
        for d in range(D):
            prev = 0.0
            for t in range(T):
                prev = g[b, t, d].item() * prev + v[b, t, d].item()
                ref[b, t, d] = prev
    ok = torch.allclose(s, ref, atol=1e-6)
    _report("smoke 2: _diagonal_linear_recurrence matches an independent hand-rolled reference",
            ok, f"max_abs_diff={(s - ref).abs().max().item():.2e}")


def smoke_3_q_never_touches_state_bottleneck():
    """Blank-out/bottleneck check (CLAUDE.md's own required discipline for
    a P=1 hard bottleneck): perturbing q AFTER the state s has been
    computed must change the mixer's OUTPUT (q is read-only, still used in
    the Hadamard read) but the RECURRENT STATE itself (s, hence
    final_state) must be COMPLETELY UNCHANGED by q -- proves q never enters
    the state-update recurrence, mechanically, not by code inspection
    alone."""
    torch.manual_seed(3)
    m = AblationLMMixer(d_model=16, d_state=8, conv_size=4)
    x = torch.randn(2, 12, 16)
    x_perturbed_q_path = x.clone()
    # Perturb x only where it feeds q_proj's OWN weight response by scaling the whole input --
    # simplest structural probe: monkey-patch q_proj's weight and confirm final_state is bit-
    # identical (q_proj's output ONLY reaches the Hadamard read, never v/g/the recurrence).
    with torch.no_grad():
        o1, s1 = m(x)
        m.q_proj.weight.mul_(37.0)     # violent, unmistakable perturbation of q's own pathway
        o2, s2 = m(x)
    state_unchanged = torch.equal(s1, s2)
    output_changed = not torch.allclose(o1, o2)
    ok = state_unchanged and output_changed
    _report("smoke 3: q-blank-out bottleneck check (perturbing q_proj changes output, NEVER "
            "final_state)", ok, f"state_unchanged={state_unchanged} output_changed={output_changed}")


def smoke_4_param_count_matches_formula():
    d_model, d_state, conv_size = 256, 64, 4
    mixer = AblationLMMixer(d_model, d_state, conv_size=conv_size)
    real = sum(p.numel() for p in mixer.parameters())
    formula = count_mixer_params(d_model, d_state, conv_size)
    ok = real == formula
    _report("smoke 4: count_mixer_params formula matches a REAL instantiated mixer's numel() sum",
            ok, f"real={real} formula={formula}")


def smoke_5_initial_state_threading():
    """A nonzero initial_state must actually influence the FIRST output
    position (s_0 = g_0*initial_state + v_0, not silently dropped)."""
    torch.manual_seed(9)
    m = AblationLMMixer(d_model=8, d_state=4, conv_size=4)
    x = torch.randn(1, 6, 8)
    with torch.no_grad():
        _, s_zero = m(x, initial_state=None)
        _, s_nonzero = m(x, initial_state=torch.ones(1, 4) * 5.0)
    ok = not torch.equal(s_zero, s_nonzero)
    _report("smoke 5: nonzero initial_state changes final_state (threaded through, not dropped)",
            ok)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.parse_args()
    print("=" * 70)
    print("ablation_mixer_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.3(a) smoke suite")
    print(f"fla stub installed (real fla absent or forced): {_STUB_INSTALLED}")
    print("=" * 70)
    smoke_1_forward_backward_grad_finite()
    smoke_2_recurrence_matches_hand_reference()
    smoke_3_q_never_touches_state_bottleneck()
    smoke_4_param_count_matches_formula()
    smoke_5_initial_state_threading()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
