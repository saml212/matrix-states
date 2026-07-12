"""NCR wave-1 arms + reads (NOVEL_ARCH_WATERFALL.md S3.1/S3.3, Rev 2).

Arms of record (S3.3), all torch fp32, all trained at h in {1,2,3} with the
identical raw-integer h signal, all scored by the same continuous cosine
readout (no argmax/codebook anywhere -- CLAUDE.md hard rule):

  NCRModel       contender: BindingEncoder (chapter2/model_v4, verbatim
                 import) writes Z in context; TRAIN read = <=3 naive matmuls
                 (MatrixCompositionModel.compose, verbatim import); EVAL read
                 = scale-managed binary exponentiation `binexp_read` (the
                 claimed configuration) + `loop_read` (the disclosed
                 cost-control arm / fp32 cross-check, per-step renorm, MA5).
  FWMReadModel   comparison of record #2: SAME write mechanism (own
                 BindingEncoder weights), read = h-fold recursive one-hop
                 LN reads x <- LN(Z x) (Schlag et al. arXiv:2011.07831,
                 cited AND distinguished: FWM fixes N_r=3; here h is the
                 input-supplied loop count). O(h) sequential.
  LoopedVecModel comparison of record #1: param-matched iterated VECTOR map
                 (mi6 pinned family): weight-tied pre-LN residual two-layer
                 GELU MLP  x_{t+1} = x_t + W2 GELU(W1 LN(x_t) + b1) + b2  on
                 the d=16 state; no attention in the loop, no gating, no
                 per-step weights; linear decode; h consumed ONLY as the
                 loop count. Episode enters via the encoder-produced x_0
                 (same transformer family as the NCR encoder + the query as
                 an extra token) -- the pinned family gives the step map no
                 episode conditioning beyond the state itself; whatever
                 convergence profile that buys is a Phase-0 READOUT, not a
                 build-time family swap (mi6: "no post-hoc family swaps").
  C_MLP          Task E's inherited MLPShortcutModel, VERBATIM import --
                 DISCLOSED WEAK CONTROL ONLY (one-hot(h) is its inherited
                 design and exactly why it cannot extrapolate; the S3.1
                 raw-integer-h pin governs the comparisons of record, and
                 C_MLP is pinned as never-the-comparison-of-record).

PARAM MATCH (S3.3): +-15% tolerance vs the NCR arm, computed exactly at
build time (`param_report` + `assert_param_match`). LoopedVec hidden width
529 is derived, not tuned: NCR total = 170,896; LoopedVec total =
153,424 + 33*H  =>  H = 529 gives 170,881 (delta 15 params, 0.009%).

READS -- the scale-management pins (S3.1 + S3.3 MA5), all cosine-invariant
because every renormalization is a strictly positive per-item scalar (a
positive scalar can never move a cosine; verified numerically in the S4
attack round at h=21/61/125 incl. non-power-of-2 partial products, and
negative-tested here in ncr_selftest.py with a deliberately non-scalar
mutation):

  binexp_read : square-and-multiply. The squared BASE is Frobenius-
                renormalized after EVERY squaring (per batch item) AND the
                running partial product (kept as the vector B_k ... q) is
                L2-renormalized after every multiply; the two log-scales are
                tracked separately (S3.1 scale row, Rev 2). ceil(log2 h)
                squarings + <=ceil(log2 h)+1 vector applies.
  loop_read   : literal h-fold matvec with per-step L2 renorm of the running
                iterate + log-scale (MA5: unrenormalized fp32 overflows at
                h ~= 85 for the measured worst c* = 2.843).

Both are dtype-generic (fp32 eval / fp64 shadow, S3.1 precision instrument).

Self-contained apart from chapter2 imports (model_v4/model_e/task_d).
"""
from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
if CHAPTER2 not in sys.path:
    sys.path.insert(0, CHAPTER2)

import task_d as td                                   # noqa: E402
from model_v4 import BindingEncoder                   # noqa: E402 (verbatim)
from model_e import MatrixCompositionModel, MLPShortcutModel  # noqa: E402

D_PIN = 16
ENC_H, ENC_LAYERS, ENC_HEADS, ENC_REFINE = 64, 3, 4, 1   # Task E's own config
LOOPEDVEC_HIDDEN = 529                                   # derived above, not tuned
PARAM_TOL = 0.15                                         # S3.3 pin
_EPS = 1e-30


# ---------------------------------------------------------------------------
# Reads (pure functions; no learned parameters -- composition purity)
# ---------------------------------------------------------------------------

def _renorm_vec(v: torch.Tensor):
    """L2-renormalize per (batch, query) item; return (v', log-norm)."""
    n = v.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    return v / n, torch.log(n.squeeze(-1))


def _renorm_mat(M: torch.Tensor):
    """Frobenius-renormalize per batch item; return (M', log-norm)."""
    n = M.reshape(M.shape[0], -1).norm(dim=-1).clamp(min=_EPS)
    return M / n.view(-1, 1, 1), torch.log(n)


def binexp_read(Z: torch.Tensor, q: torch.Tensor, h: int):
    """o proportional to Z^h q (exactly, up to a strictly positive per-item
    scalar). Z: (B,d,d); q: (B,Q,d); h: int >= 1 (raw, never reduced).
    Returns dict(o, log_scale_base (B,), log_scale_prod (B,Q),
    n_squarings, n_applies)."""
    assert isinstance(h, int) and h >= 1, h
    base = Z
    v = q
    log_base = torch.zeros(Z.shape[0], device=Z.device, dtype=Z.dtype)
    log_prod = torch.zeros(q.shape[0], q.shape[1], device=q.device, dtype=q.dtype)
    n_sq = n_ap = 0
    hh = h
    while hh > 0:
        if hh & 1:
            v = torch.einsum("bij,bqj->bqi", base, v)
            v, ln = _renorm_vec(v)
            log_prod = log_prod + ln     # base's own scale tracked separately below
            n_ap += 1
        hh >>= 1
        if hh > 0:
            base = torch.einsum("bij,bjk->bik", base, base)
            base, lb = _renorm_mat(base)
            log_base = log_base + lb
            n_sq += 1
    return dict(o=v, log_scale_base=log_base, log_scale_prod=log_prod,
                n_squarings=n_sq, n_applies=n_ap)


def loop_read(Z: torch.Tensor, q: torch.Tensor, h: int):
    """Literal h-fold matvec with per-step L2 renorm of the running iterate
    (MA5). Same positive-scalar cosine-invariance argument as binexp_read."""
    assert isinstance(h, int) and h >= 1, h
    v = q
    log_prod = torch.zeros(q.shape[0], q.shape[1], device=q.device, dtype=q.dtype)
    for _ in range(h):
        v = torch.einsum("bij,bqj->bqi", Z, v)
        v, ln = _renorm_vec(v)
        log_prod = log_prod + ln
    return dict(o=v, log_scale_prod=log_prod, n_applies=h)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

class NCRModel(nn.Module):
    """Contender. Encoder verbatim (BindingEncoder); train read = Task E's
    own <=3-naive-matmul compose (verbatim static method, zero params --
    composition purity holds by the same signature pin run_task_e's smoke
    asserts); eval reads are the module-level pure functions above."""

    arm = "ncr"
    deviating_read = False   # direct matvec read -- S3.1 M6 pin

    def __init__(self, d: int = D_PIN, h: int = ENC_H):
        """`h` (S9.7 write-capacity diagnostic): encoder hidden width,
        defaults to ENC_H=64 so every existing call site (Condition A /
        the K=8/K=12 anchor) is unchanged. Condition B (S9.2) passes
        h=8K explicitly -- BindingEncoder already accepts arbitrary h via
        its own constructor signature (S9.5 prereq #3), no change needed
        there."""
        super().__init__()
        self.d = d
        self.encoder = BindingEncoder(d, h, ENC_LAYERS, ENC_HEADS, ENC_REFINE)

    def encode(self, keys, values):
        return self.encoder(keys, values)

    def forward(self, batch: dict):
        """TRAIN path only (h <= 3, backprop through naive matmuls -- S3.1)."""
        Z = self.encode(batch["keys"], batch["values"])
        pred = MatrixCompositionModel.compose(Z, batch["query_keys"], batch["hops"])
        return pred, Z

    @staticmethod
    def eval_read(Z, query_keys, h: int, kind: str = "binexp"):
        if kind == "binexp":
            return binexp_read(Z, query_keys, h)
        if kind == "loop":
            return loop_read(Z, query_keys, h)
        raise ValueError(kind)


class FWMReadModel(nn.Module):
    """Comparison of record #2. Same written Z (own encoder weights); read =
    h-fold recursive one-hop LN reads, x <- LN(Z x). The LN affine (32
    params) is the read's only learned part and is weight-tied across hops.
    O(h) sequential; deviating read => read-vector-std diagnostic at its
    calibration gate (S3.1/S3.3)."""

    arm = "fwm"
    deviating_read = True

    def __init__(self, d: int = D_PIN):
        super().__init__()
        self.d = d
        self.encoder = BindingEncoder(d, ENC_H, ENC_LAYERS, ENC_HEADS, ENC_REFINE)
        self.read_ln = nn.LayerNorm(d)

    def encode(self, keys, values):
        return self.encoder(keys, values)

    def read(self, Z, query_keys, hops):
        """Per-query loop count via masking, mirroring compose()'s h-purity
        convention (each query gets EXACTLY its own h steps)."""
        cur = query_keys
        result = torch.where(hops.unsqueeze(-1) == 0, cur, torch.zeros_like(cur))
        max_h = int(hops.max().item()) if hops.numel() > 0 else 0
        for t in range(1, max_h + 1):
            cur = self.read_ln(torch.einsum("bij,bqj->bqi", Z, cur))
            result = torch.where((hops == t).unsqueeze(-1), cur, result)
        return result

    def read_fixed_h(self, Z, query_keys, h: int):
        """Uniform-depth eval read (no masking loop over max_h ranges)."""
        cur = query_keys
        for _ in range(h):
            cur = self.read_ln(torch.einsum("bij,bqj->bqi", Z, cur))
        return cur

    def forward(self, batch: dict):
        Z = self.encode(batch["keys"], batch["values"])
        pred = self.read(Z, batch["query_keys"], batch["hops"])
        return pred, Z


class LoopedVecModel(nn.Module):
    """Comparison of record #1 (mi6 pinned family; docstring at module top).
    Encoder: the same transformer family as BindingEncoder's trunk (in_proj
    over [key;value] binding tokens + [query;0] query tokens, 3 pre-LN
    layers, d_model 64) -> per-query state head -> x_0 in R^16. Loop: the
    pinned weight-tied step map, h times. Decode: linear d->d."""

    arm = "loopedvec"
    deviating_read = True

    def __init__(self, d: int = D_PIN, hidden: int = LOOPEDVEC_HIDDEN):
        super().__init__()
        self.d = d
        self.in_proj = nn.Linear(2 * d, ENC_H)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=ENC_H, nhead=ENC_HEADS, dim_feedforward=4 * ENC_H,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=ENC_LAYERS,
                                             enable_nested_tensor=False)
        self.state_head = nn.Linear(ENC_H, d)
        # the pinned step map: x + W2 GELU(W1 LN(x) + b1) + b2
        self.step_ln = nn.LayerNorm(d)
        self.step_w1 = nn.Linear(d, hidden)     # W1, b1
        self.step_w2 = nn.Linear(hidden, d)     # W2, b2
        self.act = nn.GELU()
        self.decode = nn.Linear(d, d)

    def encode(self, keys, values, query_keys):
        """Episode + query -> per-query initial state x_0 (B, Q, d)."""
        B, K, d = keys.shape
        Q = query_keys.shape[1]
        bind_tok = self.in_proj(torch.cat([keys, values], dim=-1))          # (B,K,H)
        q_tok = self.in_proj(torch.cat([query_keys,
                                        torch.zeros_like(query_keys)], dim=-1))  # (B,Q,H)
        mem = self.encoder(torch.cat([bind_tok, q_tok], dim=1))             # (B,K+Q,H)
        return self.state_head(mem[:, K:, :])                               # (B,Q,d)

    def step(self, x):
        return x + self.step_w2(self.act(self.step_w1(self.step_ln(x))))

    def iterate(self, x0, hops):
        """h consumed ONLY as the loop count (S3.1); per-query h via masking."""
        cur = x0
        result = torch.where(hops.unsqueeze(-1) == 0, cur, torch.zeros_like(cur))
        max_h = int(hops.max().item()) if hops.numel() > 0 else 0
        for t in range(1, max_h + 1):
            cur = self.step(cur)
            result = torch.where((hops == t).unsqueeze(-1), cur, result)
        return result

    def iterate_fixed_h(self, x0, h: int):
        cur = x0
        for _ in range(h):
            cur = self.step(cur)
        return cur

    def forward(self, batch: dict):
        x0 = self.encode(batch["keys"], batch["values"], batch["query_keys"])
        xh = self.iterate(x0, batch["hops"])
        return self.decode(xh), x0


class CMLPModel(MLPShortcutModel):
    """Task E's inherited C_MLP (disclosed weak control) carrying the arm
    protocol attributes every runner path reads (`model.arm`). The subclass
    adds ZERO behavior -- forward/one-hot/encoder are the inherited class
    verbatim. Without this, `model.arm` raises AttributeError via
    nn.Module.__getattr__ -- the wave-1 cmlp-cell eval crash (§7e): the CPU
    suite's end-to-end micro cell ran only the ncr arm, so the gap stayed
    invisible until the box. Regression teeth now in ncr_selftest t12/t13."""
    arm = "cmlp"
    deviating_read = True   # an MLP read is definitionally not the direct matvec


def make_cmlp(d: int = D_PIN) -> CMLPModel:
    return CMLPModel(d, ENC_H, ENC_LAYERS, ENC_HEADS, ENC_REFINE,
                     h_train_max=max((1, 2, 3)), h_train=(1, 2, 3))


ARM_BUILDERS = {
    "ncr": NCRModel,
    "fwm": FWMReadModel,
    "loopedvec": LoopedVecModel,
    "cmlp": make_cmlp,
}
TRAINED_ARMS = ("ncr", "loopedvec", "fwm")     # comparisons of record + contender
ALL_ARMS = ("ncr", "loopedvec", "fwm", "cmlp")


def build_arm(arm: str, d: int = D_PIN, h: int | None = None):
    """S9.7 write-capacity diagnostic dispatcher: constructs any arm at an
    arbitrary ambient dimension `d`; `h` (encoder hidden width) applies to
    the `ncr` arm only -- the model-scale axis (Condition A/B) is an
    NCR-write-specific question (S9.1), comparison arms stay at their own
    default ENC_H=64 (unmatched-by-design for this diagnostic's NCR-only
    real-training cells; still constructible for the per-arm micro-test
    gate, S9.5 hard-rule bakein)."""
    if arm == "ncr":
        return NCRModel(d=d, h=(h if h is not None else ENC_H))
    if arm == "fwm":
        return FWMReadModel(d=d)
    if arm == "loopedvec":
        return LoopedVecModel(d=d)
    if arm == "cmlp":
        return make_cmlp(d=d)
    raise ValueError(arm)


# ---------------------------------------------------------------------------
# Param matching (S3.3: +-15%, computed exactly at build time)
# ---------------------------------------------------------------------------

def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def param_report() -> dict:
    ncr = n_params(NCRModel())
    rep = {"ncr": ncr}
    for arm in ("fwm", "loopedvec", "cmlp"):
        n = n_params(ARM_BUILDERS[arm]())
        rep[arm] = n
        rep[f"{arm}_ratio_vs_ncr"] = n / ncr
        rep[f"{arm}_in_band"] = abs(n / ncr - 1.0) <= PARAM_TOL
    return rep


def assert_param_match():
    """Comparisons of record must sit inside +-15% of the NCR arm. C_MLP is
    the disclosed weak control -- reported, never gated (S3.3)."""
    rep = param_report()
    for arm in ("fwm", "loopedvec"):
        assert rep[f"{arm}_in_band"], (
            f"{arm} params {rep[arm]} vs ncr {rep['ncr']} outside +-15%")
    return rep


def recovery_cosine(pred, target):
    return td.recovery_cosine(pred, target)


def _self_test():
    torch.manual_seed(0)
    rep = assert_param_match()
    print(f"  params: ncr={rep['ncr']} fwm={rep['fwm']} "
          f"loopedvec={rep['loopedvec']} (ratios "
          f"{rep['fwm_ratio_vs_ncr']:.4f}/{rep['loopedvec_ratio_vs_ncr']:.4f}) "
          f"cmlp={rep['cmlp']} (weak control, unmatched by design)")
    # binexp == loop == literal power on a random Z, fp64 reference
    B, Q, d = 4, 8, D_PIN
    Z = torch.randn(B, d, d, dtype=torch.float64) * 0.4
    q = torch.randn(B, Q, d, dtype=torch.float64)
    for h in (1, 2, 3, 5, 13, 21, 61, 125):
        ref = q
        for _ in range(h):
            ref = torch.einsum("bij,bqj->bqi", Z, ref)
        ref = ref / ref.norm(dim=-1, keepdim=True).clamp(min=_EPS)
        for kind in ("binexp", "loop"):
            o = NCRModel.eval_read(Z, q, h, kind)["o"]
            cos = (o * ref).sum(-1)
            assert torch.all(cos > 1 - 1e-9), (kind, h, cos.min().item())
    print("  binexp_read/loop_read agree with the literal fp64 power at h<=125 (cos>1-1e-9)")
    # far-h finiteness under renorm at a measured-worst-case scale (c*=2.843)
    Zbig = torch.eye(d).unsqueeze(0) * 2.843
    qb = torch.randn(1, 2, d)
    for h in (125, 1021):
        for kind in ("binexp", "loop"):
            o = NCRModel.eval_read(Zbig, qb, h, kind)["o"]
            assert torch.isfinite(o).all(), (kind, h)
    print("  scale-managed reads stay finite at h=1021 with c*=2.843 (MA5 overflow regime)")
    print("ncr_models self-test PASSED")


if __name__ == "__main__":
    _self_test()
