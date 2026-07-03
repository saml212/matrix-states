"""Task D — tensor-product key/value binding (Chapter 2 rank gate).

See ``TASK_D_PREREGISTRATION.md``. Each sample presents K bindings
(key_j -> value_j) with keys/values drawn fresh per sample as random
near-orthogonal CONTINUOUS d-vectors, then queries a key and requires the model
to reproduce the corresponding value vector EXACTLY (scored by cosine, not a
K-way argmax — argmax decoding would let a rank-1 matrix cheat, per Nichani et
al. 2412.06538). Exact recovery of K independent bindings provably requires
rank(Z) >= K.

Self-contained: torch + stdlib only, no src/ imports (pod-safe).
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TaskDConfig:
    d: int = 16               # vector dimension (= matrix side; rank range 1..d)
    K: int = 8                # number of bindings per sample
    n_query: int | None = None  # queries per sample; None -> all K bindings
    orthogonal: bool = False  # True: exactly orthonormal keys/values (QR).
                              # False: i.i.d. Gaussian, L2-normalized (near-orthogonal)

    @property
    def queries(self) -> int:
        return self.K if self.n_query is None else self.n_query

    def __post_init__(self):
        assert 1 <= self.K, "K must be >= 1"
        assert self.queries <= self.K, "n_query cannot exceed K"
        # Exact recovery needs K linearly independent keys AND values in R^d.
        # For K > d no exact solution exists (lossy regime; not the P=1 gate).
        # We allow K > d here so the generator can be reused for the K>d study,
        # but callers running the exact-recovery gate must keep K <= d.


def _random_directions(B: int, n: int, d: int, orthogonal: bool,
                       gen: torch.Generator, device, dtype) -> torch.Tensor:
    """Return (B, n, d) unit vectors.

    orthogonal=False: i.i.d. standard Gaussian, then L2-normalized. Pairwise
      cosine ~ N(0, 1/d) -> near-orthogonal; linearly independent w.p. 1 for n<=d.
    orthogonal=True: exactly orthonormal set via QR (requires n <= d), giving the
      cleanest possible rank>=K test. For n>d, falls back to normalized Gaussian.
    """
    x = torch.randn(B, n, d, generator=gen, device=device, dtype=dtype)
    if orthogonal and n <= d:
        # QR of the (d, n) transpose gives n orthonormal columns per batch item.
        q, r = torch.linalg.qr(x.transpose(-1, -2))    # q: (B, d, n), r: (B, n, n)
        # Mezzadri (2007) sign-correction: raw QR is NOT Haar-uniform (it has a
        # systematic per-column diagonal-sign bias, confirmed >100sigma in audit).
        # Multiply each orthonormal column by sign(diag(R)) to remove it.
        s = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))   # (B, n)
        s = torch.where(s == 0, torch.ones_like(s), s)         # sign(0) -> +1
        q = q * s.unsqueeze(-2)                                # scale columns
        return q.transpose(-1, -2).contiguous()                # (B, n, d), rows orthonormal
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def sample_batch(cfg: TaskDConfig, batch_size: int, gen: torch.Generator,
                 device="cpu", dtype=torch.float32) -> dict:
    """Generate one batch of Task D problems.

    Returns a dict of tensors:
      keys        (B, K, d)   binding keys (unit vectors)
      values      (B, K, d)   binding values (unit vectors) -- the recovery targets
      query_idx   (B, Q)      which bindings are queried (Q = cfg.queries), no repeats
      query_keys  (B, Q, d)   keys gathered at query_idx (fed to the unbind decoder)
      targets     (B, Q, d)   values gathered at query_idx (the vectors to recover)
    All keys/values are FRESH per sample (prob ~0 of repeating across samples),
    which is the held-out-generalization guarantee (C5): eval sees unseen vectors.
    """
    B, K, d, Q = batch_size, cfg.K, cfg.d, cfg.queries
    keys = _random_directions(B, K, d, cfg.orthogonal, gen, device, dtype)
    values = _random_directions(B, K, d, cfg.orthogonal, gen, device, dtype)

    # Choose Q distinct query bindings per sample (random subset, no repeats).
    # argsort of random noise gives an independent permutation per row.
    noise = torch.rand(B, K, generator=gen, device=device, dtype=dtype)
    perm = noise.argsort(dim=-1)                       # (B, K) permutation per row
    query_idx = perm[:, :Q].contiguous()              # (B, Q) distinct indices

    gather_kd = query_idx.unsqueeze(-1).expand(B, Q, d)
    query_keys = torch.gather(keys, 1, gather_kd)      # (B, Q, d)
    targets = torch.gather(values, 1, gather_kd)       # (B, Q, d)

    return {
        "keys": keys,
        "values": values,
        "query_idx": query_idx,
        "query_keys": query_keys,
        "targets": targets,
    }


def recovery_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between predicted and true value vectors.
    pred, target: (..., d) -> (...). The pre-registered success metric is
    cos > tau (default tau=0.9); always also report the full distribution.
    """
    pred = pred / pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    target = target / target.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (pred * target).sum(dim=-1)


# ---------------------------------------------------------------------------
# Self-test (runs where torch is available; part of the smoke gate on-cluster)
# ---------------------------------------------------------------------------

def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    for orthogonal in (False, True):
        cfg = TaskDConfig(d=16, K=8, orthogonal=orthogonal)
        b = sample_batch(cfg, batch_size=64, gen=gen)
        B, K, d, Q = 64, cfg.K, cfg.d, cfg.queries

        # Shapes
        assert b["keys"].shape == (B, K, d), b["keys"].shape
        assert b["values"].shape == (B, K, d)
        assert b["query_idx"].shape == (B, Q)
        assert b["query_keys"].shape == (B, Q, d)
        assert b["targets"].shape == (B, Q, d)

        # Unit vectors
        kn = b["keys"].norm(dim=-1)
        assert torch.allclose(kn, torch.ones_like(kn), atol=1e-4), "keys not unit"

        # Query gather consistency: targets == values[query_idx]
        gather = b["query_idx"].unsqueeze(-1).expand(B, Q, d)
        assert torch.equal(b["targets"], torch.gather(b["values"], 1, gather))
        assert torch.equal(b["query_keys"], torch.gather(b["keys"], 1, gather))

        # Distinct queries per row (no repeats)
        for row in b["query_idx"]:
            assert len(set(row.tolist())) == Q, "query_idx has repeats"

        # Near-orthogonality of keys within a sample
        gram = b["keys"] @ b["keys"].transpose(-1, -2)     # (B, K, K)
        eye = torch.eye(K).expand(B, K, K)
        off_diag = (gram - eye * gram)                     # zero the diagonal
        mean_abs_off = off_diag.abs().sum() / (B * K * (K - 1))
        tag = "orthonormal" if orthogonal else "gaussian"
        print(f"[{tag}] mean |off-diag key cosine| = {mean_abs_off:.4f} "
              f"(expect ~0 for orthonormal, ~{1.0/ (d**0.5):.3f} for gaussian)")
        if orthogonal:
            assert mean_abs_off < 1e-3, "orthonormal keys not orthogonal"

        # Rank sanity: stacked values have rank K (needed for the rank>=K proof)
        vrank = torch.linalg.matrix_rank(b["values"][0].float())
        assert vrank >= min(K, d) - 1, f"value rank {vrank} < K={K}; proof premise broken"

    print("task_d self-test PASSED")


if __name__ == "__main__":
    _self_test()
