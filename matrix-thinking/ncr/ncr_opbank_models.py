"""NCR operator-bank arms -- NOVEL_ARCH_WATERFALL.md S8.1.2/S8.1.4.

BankBindingEncoder extends model_v4.BindingEncoder (chapter2, verbatim
trunk) with: (a) a learned per-relation embedding added to each binding
token post-in_proj (the relation tag), (b) R independent row-query sets
replacing the single-relation (d,h) set, with the reader/row_norm/row_out
weights SHARED across relations (deliberately: forces R different learned
query vectors to pull different info out of the SAME mem via the SAME
attention weights -- confirmed by the S8.2 attack as "good design, not a
flaw"). forward(...) -> Z_bank: (B,R,d,d).

Arms mirror ncr_models.py's family exactly (mi6 pinned baselines, ±15%
param match), extended with the relation axis:
  NCRBankModel       contender; train read = <=3 naive matmuls per block
                     (Axis R-BANK: 1 block; Axis B-CHAIN: 2 fixed blocks);
                     eval reads reuse ncr_models.binexp_read/loop_read
                     UNMODIFIED on a Z_bank[:, r] slice (relation-agnostic
                     once Z_r is extracted -- zero new read code, S8.1.3).
  FWMBankModel       same write, read = h-fold recursive LN read on the
                     SELECTED Z_r (or Z_r2 after a prior Z_r1 block).
  LoopedVecBankModel param-matched iterated vector map (mi6 pinned step-map
                     family, unchanged); x0 per (r,query) from the shared
                     R*K-token context.
  CMLPBankModel      disclosed weak control, one-hot(h) (x) one-hot(r)
                     extension of the inherited MLPShortcutModel.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
if CHAPTER2 not in sys.path:
    sys.path.insert(0, CHAPTER2)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model_e import MLPShortcutModel           # noqa: E402
import ncr_models as nm                        # noqa: E402 (binexp_read/loop_read reuse)

D_PIN = 16
R_PIN = 3
ENC_H, ENC_LAYERS, ENC_HEADS, ENC_REFINE = 64, 3, 4, 1
LOOPEDVEC_HIDDEN_BANK = 529    # start from the single-relation derived width; re-derive at
                               # build time via param_report_bank if drift exceeds tolerance
PARAM_TOL = 0.15


class BankBindingEncoder(nn.Module):
    """R relations written from ONE shared context. See module docstring."""

    def __init__(self, d: int = D_PIN, R: int = R_PIN, h: int = ENC_H,
                n_layers: int = ENC_LAYERS, n_heads: int = ENC_HEADS,
                n_refine: int = ENC_REFINE):
        super().__init__()
        self.d, self.R, self.h, self.n_refine = d, R, h, n_refine
        self.in_proj = nn.Linear(2 * d, h)
        self.rel_embed = nn.Parameter(torch.randn(R, h) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=n_heads, dim_feedforward=4 * h,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.row_queries = nn.Parameter(torch.randn(R, d, h) * 0.02)
        self.reader = nn.MultiheadAttention(h, n_heads, batch_first=True, dropout=0.0)
        self.row_norm = nn.LayerNorm(h)
        self.row_out = nn.Linear(h, d)

    def forward(self, keys: torch.Tensor, values: torch.Tensor,
               rel_ids: torch.Tensor) -> torch.Tensor:
        """keys, values: (B, R*K, d); rel_ids: (B, R*K) long in [0,R) ->
        Z_bank: (B, R, d, d)."""
        B, RK, _ = keys.shape
        tok = self.in_proj(torch.cat([keys, values], dim=-1))          # (B,RK,h)
        tok = tok + self.rel_embed[rel_ids]                             # relation tag
        mem = self.encoder(tok)                                        # (B,RK,h)
        mem_r = mem.unsqueeze(1).expand(B, self.R, RK, self.h).reshape(B * self.R, RK, self.h)
        q = self.row_queries.unsqueeze(0).expand(B, self.R, self.d, self.h).reshape(
            B * self.R, self.d, self.h)
        for _ in range(self.n_refine):
            read, _ = self.reader(q, mem_r, mem_r, need_weights=False)
            q = self.row_norm(q + read)
        Z = self.row_out(q).reshape(B, self.R, self.d, self.d)
        return Z


def _select_Z(Z_bank: torch.Tensor, r) -> torch.Tensor:
    """Z_bank: (B,R,d,d). Three forms of r, three output shapes:
      int            -> (B,d,d)     whole-batch single relation (eval)
      (B,)  long     -> (B,d,d)     one relation per batch item (swap ablation)
      (B,Q) long     -> (B,Q,d,d)   one relation PER QUERY (train, mixed r1/r2)
    """
    B, R, d, _ = Z_bank.shape
    if isinstance(r, int):
        return Z_bank[:, r]
    if r.dim() == 1:
        idx = r.view(B, 1, 1, 1).expand(B, 1, d, d)
        return torch.gather(Z_bank, 1, idx).squeeze(1)
    Q = r.shape[1]
    Zexp = Z_bank.unsqueeze(1).expand(B, Q, R, d, d)
    idx = r.view(B, Q, 1, 1, 1).expand(B, Q, 1, d, d)
    return torch.gather(Zexp, 2, idx).squeeze(2)


class NCRBankModel(nn.Module):
    arm = "ncr-bank"
    deviating_read = False

    def __init__(self, d: int = D_PIN, R: int = R_PIN):
        super().__init__()
        self.d, self.R = d, R
        self.encoder = BankBindingEncoder(d, R)

    def encode(self, keys, values, rel_ids):
        return self.encoder(keys, values, rel_ids)

    def forward(self, batch: dict):
        """TRAIN path: h1<=3 naive matmuls (block 1, per-query relation r1)
        then, IFF is_chain (h2>0), h2<=3 more (block 2, per-query relation
        r2) -- purity-preserving per-query masking mirrors
        MatrixCompositionModel.compose's convention. r1/r2 are (B,Q)."""
        Z = self.encode(batch["keys"], batch["values"], batch["rel_ids"])   # (B,R,d,d)
        v = batch["query_keys"]
        Zr1 = _select_Z(Z, batch["r1"])       # (B,Q,d,d)
        max_h1 = int(batch["h1"].max().item())
        cur = v
        for t in range(1, max_h1 + 1):
            stepped = torch.einsum("bqij,bqj->bqi", Zr1, cur)
            cur = torch.where((batch["h1"] >= t).unsqueeze(-1), stepped, cur)
        mid = cur
        Zr2 = _select_Z(Z, batch["r2"])       # (B,Q,d,d)
        max_h2 = int(batch["h2"].max().item())
        cur2 = mid
        for t in range(1, max_h2 + 1):
            stepped = torch.einsum("bqij,bqj->bqi", Zr2, cur2)
            cur2 = torch.where((batch["h2"] >= t).unsqueeze(-1), stepped, cur2)
        pred = cur2
        return pred, Z

    @staticmethod
    def eval_read(Z_bank, query_keys, r: int, h: int, kind: str = "binexp"):
        Zr = _select_Z(Z_bank, r)
        return nm.NCRModel.eval_read(Zr, query_keys, h, kind)

    @staticmethod
    def eval_read_chain(Z_bank, query_keys, r1: int, h1: int, r2: int, h2: int,
                        kind: str = "binexp"):
        mid = NCRBankModel.eval_read(Z_bank, query_keys, r1, h1, kind)["o"]
        return NCRBankModel.eval_read(Z_bank, mid, r2, h2, kind)


class FWMBankModel(nn.Module):
    arm = "fwm-bank"
    deviating_read = True

    def __init__(self, d: int = D_PIN, R: int = R_PIN):
        super().__init__()
        self.d, self.R = d, R
        self.encoder = BankBindingEncoder(d, R)
        self.read_ln = nn.LayerNorm(d)

    def encode(self, keys, values, rel_ids):
        return self.encoder(keys, values, rel_ids)

    def read_fixed_h(self, Z_bank, query_keys, r: int, h: int):
        cur = query_keys
        Zr = _select_Z(Z_bank, r)
        for _ in range(h):
            cur = self.read_ln(torch.einsum("bij,bqj->bqi", Zr, cur))
        return cur

    def forward(self, batch: dict):
        Z = self.encode(batch["keys"], batch["values"], batch["rel_ids"])
        v = batch["query_keys"]
        Zr1 = _select_Z(Z, batch["r1"])       # (B,Q,d,d)
        max_h1 = int(batch["h1"].max().item())
        cur = v
        for t in range(1, max_h1 + 1):
            stepped = self.read_ln(torch.einsum("bqij,bqj->bqi", Zr1, cur))
            cur = torch.where((batch["h1"] >= t).unsqueeze(-1), stepped, cur)
        mid = cur
        Zr2 = _select_Z(Z, batch["r2"])       # (B,Q,d,d)
        max_h2 = int(batch["h2"].max().item())
        cur2 = mid
        for t in range(1, max_h2 + 1):
            stepped = self.read_ln(torch.einsum("bqij,bqj->bqi", Zr2, cur2))
            cur2 = torch.where((batch["h2"] >= t).unsqueeze(-1), stepped, cur2)
        return cur2, Z


class LoopedVecBankModel(nn.Module):
    arm = "loopedvec-bank"
    deviating_read = True

    def __init__(self, d: int = D_PIN, R: int = R_PIN, hidden: int = LOOPEDVEC_HIDDEN_BANK):
        super().__init__()
        self.d, self.R = d, R
        self.in_proj = nn.Linear(2 * d, ENC_H)
        self.rel_embed = nn.Parameter(torch.randn(R, ENC_H) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=ENC_H, nhead=ENC_HEADS, dim_feedforward=4 * ENC_H,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=ENC_LAYERS,
                                             enable_nested_tensor=False)
        self.state_head = nn.Linear(ENC_H, d)
        self.step_ln = nn.LayerNorm(d)
        self.step_w1 = nn.Linear(d, hidden)
        self.step_w2 = nn.Linear(hidden, d)
        self.act = nn.GELU()
        self.decode = nn.Linear(d, d)

    def encode(self, keys, values, rel_ids, query_keys):
        B, RK, d = keys.shape
        Q = query_keys.shape[1]
        bind_tok = self.in_proj(torch.cat([keys, values], dim=-1)) + self.rel_embed[rel_ids]
        q_tok = self.in_proj(torch.cat([query_keys, torch.zeros_like(query_keys)], dim=-1))
        mem = self.encoder(torch.cat([bind_tok, q_tok], dim=1))
        return self.state_head(mem[:, RK:, :])

    def step(self, x):
        return x + self.step_w2(self.act(self.step_w1(self.step_ln(x))))

    def forward(self, batch: dict):
        x0 = self.encode(batch["keys"], batch["values"], batch["rel_ids"], batch["query_keys"])
        max_h1 = int(batch["h1"].max().item())
        cur = x0
        for t in range(1, max_h1 + 1):
            cur = torch.where((batch["h1"] >= t).unsqueeze(-1), self.step(cur), cur)
        mid = cur
        max_h2 = int(batch["h2"].max().item())
        cur2 = mid
        for t in range(1, max_h2 + 1):
            cur2 = torch.where((batch["h2"] >= t).unsqueeze(-1), self.step(cur2), cur2)
        return self.decode(cur2), x0

    def iterate_fixed_h(self, x0, h: int):
        cur = x0
        for _ in range(h):
            cur = self.step(cur)
        return cur


class CMLPBankModel(MLPShortcutModel):
    """Disclosed weak control -- one-hot(h) inherited, relation-agnostic by
    construction (never the comparison of record, S3.3/S8.1.4 convention)."""
    arm = "cmlp-bank"
    deviating_read = True

    def forward(self, batch: dict):
        # ignores rel_ids/r1/r2 entirely (architecturally can't use them) --
        # scored on Axis R-BANK only via h as its one-hot signal, disclosed.
        # Train batches (sample_train_batch) carry h1/h2; eval batches
        # (sample_eval_batch_axis_r) carry hops directly -- accept either.
        if "hops" in batch:
            return super().forward(batch)
        b2 = dict(batch)
        b2["hops"] = batch["h1"]
        return super().forward(b2)


def make_cmlp_bank(d: int = D_PIN) -> CMLPBankModel:
    return CMLPBankModel(d, ENC_H, ENC_LAYERS, ENC_HEADS, ENC_REFINE,
                         h_train_max=3, h_train=(1, 2, 3))


ARM_BUILDERS = {
    "ncr-bank": NCRBankModel,
    "fwm-bank": FWMBankModel,
    "loopedvec-bank": LoopedVecBankModel,
    "cmlp-bank": make_cmlp_bank,
}
TRAINED_ARMS = ("ncr-bank", "loopedvec-bank", "fwm-bank")
ALL_ARMS = ("ncr-bank", "loopedvec-bank", "fwm-bank", "cmlp-bank")


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def param_report() -> dict:
    ncr = n_params(NCRBankModel())
    rep = {"ncr-bank": ncr}
    for arm in ("fwm-bank", "loopedvec-bank", "cmlp-bank"):
        n = n_params(ARM_BUILDERS[arm]())
        rep[arm] = n
        rep[f"{arm}_ratio_vs_ncr"] = n / ncr
        rep[f"{arm}_in_band"] = abs(n / ncr - 1.0) <= PARAM_TOL
    return rep


def assert_param_match():
    rep = param_report()
    for arm in ("fwm-bank", "loopedvec-bank"):
        assert rep[f"{arm}_in_band"], (
            f"{arm} params {rep[arm]} vs ncr-bank {rep['ncr-bank']} outside +-15%: {rep}")
    return rep


def _self_test():
    torch.manual_seed(0)
    rep = assert_param_match()
    print(f"  params: ncr-bank={rep['ncr-bank']} fwm-bank={rep['fwm-bank']} "
         f"loopedvec-bank={rep['loopedvec-bank']} (ratios "
         f"{rep['fwm-bank_ratio_vs_ncr']:.4f}/{rep['loopedvec-bank_ratio_vs_ncr']:.4f}) "
         f"cmlp-bank={rep['cmlp-bank']} (weak control, unmatched by design)")
    d, R = D_PIN, R_PIN
    B, Q = 4, 8
    enc = BankBindingEncoder(d, R)
    keys = torch.randn(B, R * 8, d)
    values = torch.randn(B, R * 8, d)
    rel_ids = torch.arange(R).repeat_interleave(8).unsqueeze(0).expand(B, -1)
    Z = enc(keys, values, rel_ids)
    assert Z.shape == (B, R, d, d), Z.shape
    # closed-form check: binexp/loop reads on a Z_bank slice agree with the
    # literal fp64 power (S8.1.5's "shape-only delta" over ncr_models' own
    # self-test) -- for EVERY r, not just r=0.
    Zbig = torch.randn(B, R, d, d, dtype=torch.float64) * 0.4
    q = torch.randn(B, Q, d, dtype=torch.float64)
    for r in range(R):
        ref = q
        for _ in range(13):
            ref = torch.einsum("bij,bqj->bqi", Zbig[:, r], ref)
        ref = ref / ref.norm(dim=-1, keepdim=True).clamp(min=1e-30)
        for kind in ("binexp", "loop"):
            o = NCRBankModel.eval_read(Zbig, q, r, 13, kind)["o"]
            cos = (o * ref).sum(-1)
            assert torch.all(cos > 1 - 1e-9), (r, kind, cos.min().item())
    print("  binexp_read/loop_read agree with the literal fp64 power on EVERY "
         "bank relation (r=0..R-1) at h=13")
    print("ncr_opbank_models self-test PASSED")


if __name__ == "__main__":
    _self_test()
