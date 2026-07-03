"""task_dn.py -- DeltaNet causal-rank task generator (BIND/QUERY grammar,
STREAMED sequence form). See DELTANET_CAUSAL_RANK_DESIGN.md sections 4-5
(frozen design, revision 2.1).

Adapts chapter2/task_e.py's permutation-graph generator (single Hamiltonian
K-cycle, injective-by-construction, periodicity-guarded H_test/H_extra) from
Task D/E's non-causal SET encoding into DeltaNet's STREAMED, per-position
grammar: "BIND k_1 v_1 ... BIND k_K v_K" streamed as K literal SEQUENCE
positions (design section 4.1 -- "not a set, as in Task D/E; this is the
actual difference a recurrent architecture buys"), each binding as ONE
token position, separated by >= conv_size-1 zero-embedding buffer tokens
(C14) between items and at the BIND/QUERY boundary. Buffer embeddings are
pinned to EXACT ZERO (NEW-2), not learned -- so the primary harness's
two-kernel split (BIND-only call vs. a continuous call) round-trips exactly,
and so buffers never accidentally carry information the delta rule could
write into the state.

Self-contained (task_d.py's ``_random_directions`` / Mezzadri sign-
correction and task_e.py's ``_permutation_graph`` / ``_iterate_permutation``
/ ``_assert_injective`` are reproduced here verbatim, with attribution,
rather than cross-directory imported) -- matches this codebase's established
pod-safety convention (see rank_utils.py's own docstring for the same
reasoning and its origin, the 2026-04-29 rank_aware_v1 import-fragility
incident).

**Scope decision (build-time, flag for audit):** only the PERMUTATION /
single-Hamiltonian-K-cycle variant is implemented. Design section 4.1's
grammar is defined over a single K-cycle entity pool (mirroring task_e.py's
"permutation" variant exactly); task_e.py's secondary "chain" variant is not
required by the design's primary gate and is out of scope here (matching
the design's own primary/secondary split, section 6.1). The task brief's
"K-cycle + KV-binding variants" is satisfied by ONE generator: h=1 queries
ARE the plain KV-binding (Task-D-equivalent) recall task, and h>1 queries
are the compositional (Task-E-equivalent) extension -- both live on the
same K-cycle graph, exactly as in task_e.py's own H_train=(1,2,3) convention
(h=1 is already a training hop, not a separate code path).

Grammar per sample (T_bind = K * (1 + buf_len), buf_len = max(1,
conv_size-1)):

  positions:  [BIND_1, buf..., BIND_2, buf..., ..., BIND_K, buf...]
  BIND_t raw embedding   = concat(key_t, value_t)  in R^{2d}
  buffer raw embedding   = EXACT ZERO in R^{2d}                  (NEW-2, C14)
  beta_mask[t] = 1.0 on BIND item positions ONLY, 0.0 on every buffer
    position (C9/NEW-3) -- a 0/1 tensor only; the model applies it to its
    OWN learned beta-logit (so the learned gate is still genuinely exercised
    on BIND positions, not architecturally forced to any particular value
    there -- only forced to exactly 0 on non-BIND positions).

The QUERY phase is NOT part of the streamed sequence in the primary
two-kernel-split harness (design section 5.2's revision-2 note: "the
primary bespoke harness satisfies this freeze by construction -- the
BIND-only kernel call simply ends at the phase boundary, and no QUERY token
ever enters the recurrence at all"). Queries are returned as
(query_keys, hops, targets), exactly task_e.py's own contract, for the
model's EXTERNAL pinned readout ``pred(a,h) = S_T^h @ key_a`` (section 5.4).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Reproduced verbatim from task_d.py / task_e.py (attribution in each
# docstring below) -- see module docstring for why this is a copy, not an
# import.
# ---------------------------------------------------------------------------


def _random_directions(B: int, n: int, d: int, orthogonal: bool,
                        gen: torch.Generator, device, dtype) -> torch.Tensor:
    """Verbatim from task_d.py::_random_directions (Mezzadri 2007 sign
    correction included -- raw QR is NOT Haar-uniform without it, confirmed
    >100sigma in the original Task D audit)."""
    x = torch.randn(B, n, d, generator=gen, device=device, dtype=dtype)
    if orthogonal and n <= d:
        q, r = torch.linalg.qr(x.transpose(-1, -2))
        s = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
        s = torch.where(s == 0, torch.ones_like(s), s)
        q = q * s.unsqueeze(-2)
        return q.transpose(-1, -2).contiguous()
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def _permutation_graph(B: int, K: int, gen: torch.Generator, device, dtype) -> torch.Tensor:
    """Verbatim (renamed args identical) from task_e.py::_permutation_graph:
    a random SINGLE Hamiltonian K-cycle per row, NOT a general random
    permutation -- closes the cycle-length-periodicity confound
    (gauntlet/AUDIT_task_e_validity.md Finding B). succ[b,i] = pi(i), a
    bijection on {0..K-1}."""
    order = torch.rand(B, K, generator=gen, device=device, dtype=dtype).argsort(dim=-1)
    next_order = torch.roll(order, shifts=-1, dims=-1)
    succ = torch.empty(B, K, dtype=torch.int64, device=device)
    succ.scatter_(1, order, next_order)
    return succ


def _iterate_permutation(succ: torch.Tensor, a_idx: torch.Tensor,
                          hops: torch.Tensor) -> torch.Tensor:
    """Verbatim from task_e.py::_iterate_permutation: pi^h(a) by exact index
    iteration -- the numerically EXACT ground-truth label."""
    cur = a_idx.clone()
    result = torch.where(hops == 0, cur, torch.full_like(cur, -1))
    max_h = int(hops.max().item()) if hops.numel() > 0 else 0
    for t in range(1, max_h + 1):
        cur = torch.gather(succ, 1, cur)
        result = torch.where(hops == t, cur, result)
    assert (result >= 0).all(), "a query's hop depth exceeded the iteration bound"
    return result


def _assert_injective(values: torch.Tensor) -> None:
    """Verbatim from task_e.py::_assert_injective (C6, load-bearing): EXACT
    rank threshold, no "-1" slack (a single accidental merge drops rank by
    exactly 1; a tolerant threshold cannot detect it -- the exact failure
    mode this guard exists to catch, per the standing house rule on
    structural/integer checks)."""
    K = values.shape[1]
    K_eff = min(K, values.shape[-1])
    vrank = torch.linalg.matrix_rank(values[0].float())
    assert vrank >= K_eff, \
        f"generated values have rank {vrank} < K_eff={K_eff} (K={K}); the in-degree<=1 " \
        f"injectivity guarantee is violated -- the rank(S_T)>=K proof premise is broken"


def ideal_S(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """The classical, ZERO-TRAINING minimum-norm operator S_ideal = sum_i
    value_i key_i^T (task_e.py::ideal_Z, renamed for the streamed-state
    context). Architecture-native for DeltaNet specifically (design section
    3.6's proposition): with beta_t=1 and exactly orthonormal keys, the
    delta rule's OWN native recurrence reaches exactly this S from S_0=0,
    zero training -- checked mechanically in run_deltanet.py's smoke gate,
    not just asserted here. keys, values: (B, K, d) -> (B, d, d)."""
    return torch.einsum("bki,bkj->bij", values, keys)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaNetTaskConfig:
    d: int = 64                    # vector dim == state dim (d x d), design section 6.1
    K: int = 16                    # bindings per sample (== pool size; single K-cycle)
    conv_size: int = 4             # fla-org default (F15-verified); drives buffer length
    n_query: int | None = None     # query slots per sample; None -> K
    H_train: tuple[int, ...] = (1, 2, 3)
    H_test: tuple[int, ...] = (4, 5, 6)
    H_extra: tuple[int, ...] = (7, 21)
    orthogonal: bool = True        # GATE DEFAULT (Task D/E audit finding; C7/section-3.6 exactness)

    @property
    def buf_len(self) -> int:
        return max(1, self.conv_size - 1)

    @property
    def T_bind(self) -> int:
        return self.K * (1 + self.buf_len)

    @property
    def queries(self) -> int:
        return self.K if self.n_query is None else self.n_query

    def __post_init__(self):
        assert self.d >= 1 and self.K >= 1
        assert self.K <= self.d, f"K={self.K} must be <= d={self.d} for an orthonormal pool"
        assert self.conv_size >= 1
        assert len(self.H_train) >= 1 and len(self.H_test) >= 1
        assert min(self.H_train) >= 1, "hop depth must be >= 1"
        assert not (set(self.H_train) & set(self.H_test)), \
            "H_train and H_test must be disjoint -- the held-out-hop test requires it"
        if self.H_extra:
            assert not (set(self.H_train) & set(self.H_extra)) and \
                   not (set(self.H_test) & set(self.H_extra)), \
                   "H_extra must not overlap H_train/H_test"
        # Periodicity guard -- verbatim logic from task_e.py's
        # TaskEConfig.__post_init__ (Finding B fix): with a SINGLE K-cycle,
        # pi^h is periodic ONLY with period K, so a held-out hop whose
        # h % K lands on 0 (identity) or a training hop's residue is
        # secretly trivial or in-distribution, not genuinely held-out.
        train_residues = {h % self.K for h in self.H_train}
        for h in (*self.H_test, *self.H_extra):
            r = h % self.K
            assert r != 0, \
                f"H_test/H_extra hop h={h} is a multiple of K={self.K}: pi^{h} is the " \
                f"IDENTITY -- this probe is confounded (measures nothing), not held-out."
            assert r not in train_residues, \
                f"H_test/H_extra hop h={h} has h % K={self.K} == {r}, coinciding with a " \
                f"TRAINING hop's residue ({sorted(train_residues)}) -- secretly in-distribution."


# ---------------------------------------------------------------------------
# Batch sampler
# ---------------------------------------------------------------------------

def sample_batch(cfg: DeltaNetTaskConfig, batch_size: int, gen: torch.Generator,
                  hop_set: tuple, device="cpu", dtype=torch.float32,
                  assert_injective: bool = True) -> dict:
    """Generate one batch of DeltaNet causal-rank problems (streamed form).

    hop_set: candidate hop depths for THIS call's queries (task_e.py
      convention: pass cfg.H_train for training batches, a single-element
      tuple for a fixed-hop eval curve).

    Returns:
      bind_embed    (B, T_bind, 2d)  raw per-position embeddings; BIND item
                                      positions carry concat(key,value),
                                      buffer positions are EXACT ZERO (NEW-2)
      beta_mask     (B, T_bind)      1.0 on BIND item positions, 0.0 elsewhere
                                      (hard mask -- C9/NEW-3; multiply into
                                      the model's OWN learned beta-logit,
                                      never architecturally force the value
                                      the gate takes ON a BIND position)
      item_pos      (B, K) int64     sequence position of each BIND item t
                                      (0-indexed into bind_embed's T axis)
      keys, values  (B, K, d)        the K raw bindings (reference; also fed
                                      through bind_embed above)
      query_keys    (B, Q, d)        query start-entity vectors (raw)
      hops          (B, Q) int64     hop depth per query, drawn from hop_set
      targets       (B, Q, d)        ground truth pi^h(a), exact index
                                      composition (numerically exact label)
      s_ideal       (B, d, d)        classical minimum-norm reference S
                                      built from the RAW (keys, values) --
                                      section 3.6's C7-equivalent; the
                                      MODEL separately builds an
                                      architecture-native ideal from its own
                                      EFFECTIVE (post-W_k) keys where needed
    """
    B, K, d = batch_size, cfg.K, cfg.d
    Q = cfg.queries
    buf_len, T_bind = cfg.buf_len, cfg.T_bind

    pool = _random_directions(B, K, d, cfg.orthogonal, gen, device, dtype)
    succ = _permutation_graph(B, K, gen, device, dtype)
    keys = pool
    values = torch.gather(pool, 1, succ.unsqueeze(-1).expand(B, K, d))
    if assert_injective:
        _assert_injective(values)

    hops_pool = torch.tensor(hop_set, device=device, dtype=torch.int64)
    hops = hops_pool[torch.randint(0, len(hop_set), (B, Q), generator=gen, device=device)]
    a_idx = torch.randint(0, K, (B, Q), generator=gen, device=device)
    tgt_idx = _iterate_permutation(succ, a_idx, hops)
    query_keys = torch.gather(pool, 1, a_idx.unsqueeze(-1).expand(B, Q, d))
    targets = torch.gather(pool, 1, tgt_idx.unsqueeze(-1).expand(B, Q, d))

    # --- streamed layout: [BIND, buf..., BIND, buf..., ..., BIND, buf...] ---
    bind_embed = torch.zeros(B, T_bind, 2 * d, device=device, dtype=dtype)
    beta_mask = torch.zeros(B, T_bind, device=device, dtype=dtype)
    item_pos = (torch.arange(K, device=device) * (1 + buf_len)).unsqueeze(0).expand(B, K)
    item_pos = item_pos.contiguous()

    idx = item_pos.unsqueeze(-1).expand(B, K, 2 * d)
    bind_embed.scatter_(1, idx, torch.cat([keys, values], dim=-1))
    beta_mask.scatter_(1, item_pos, torch.ones(B, K, device=device, dtype=dtype))

    s_ideal = ideal_S(keys, values)

    return {
        "bind_embed": bind_embed, "beta_mask": beta_mask, "item_pos": item_pos,
        "keys": keys, "values": values,
        "query_keys": query_keys, "hops": hops, "targets": targets,
        "s_ideal": s_ideal,
    }


# ---------------------------------------------------------------------------
# Self-test (part of the smoke gate; run_deltanet.py --smoke calls this)
# ---------------------------------------------------------------------------

def _test_injectivity_guard_detects_merge() -> None:
    """Same discipline as task_e.py's own negative unit test: construct a
    deliberately non-injective value set and confirm the guard raises."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    d, K = 16, 8
    pool = _random_directions(1, K, d, True, gen, "cpu", torch.float32)
    values = pool.clone()
    values[:, 1, :] = values[:, 0, :]
    raised = False
    try:
        _assert_injective(values)
    except AssertionError:
        raised = True
    assert raised, "injectivity guard FAILED to detect a merged (non-injective) value set"
    print("  injectivity guard correctly REJECTS a merged (non-injective) edge set")


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    cfg = DeltaNetTaskConfig(d=16, K=8, conv_size=4,
                              H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    assert cfg.buf_len == 3
    assert cfg.T_bind == 8 * 4

    for _ in range(5):
        b = sample_batch(cfg, 64, gen, hop_set=cfg.H_train)
        B, K, d, Q = 64, cfg.K, cfg.d, cfg.queries
        assert b["bind_embed"].shape == (B, cfg.T_bind, 2 * d)
        assert b["beta_mask"].shape == (B, cfg.T_bind)
        assert b["keys"].shape == (B, K, d)
        assert b["values"].shape == (B, K, d)
        assert b["query_keys"].shape == (B, Q, d)
        assert b["hops"].shape == (B, Q)
        assert b["targets"].shape == (B, Q, d)
        assert b["s_ideal"].shape == (B, d, d)
        assert set(torch.unique(b["hops"]).tolist()) <= set(cfg.H_train)

        # beta_mask is EXACTLY 1 on K positions per row, 0 elsewhere.
        assert torch.equal(b["beta_mask"].sum(dim=-1), torch.full((B,), float(K)))
        assert set(b["beta_mask"].unique().tolist()) <= {0.0, 1.0}

        # buffer positions carry the EXACT-ZERO embedding (NEW-2).
        is_item = torch.zeros(B, cfg.T_bind, dtype=torch.bool)
        is_item.scatter_(1, b["item_pos"], True)
        buf_embed = b["bind_embed"][~is_item]
        assert torch.equal(buf_embed, torch.zeros_like(buf_embed)), \
            "buffer-token embeddings must be EXACT zero (NEW-2), not merely small"

        # item embedding reconstructs (key, value) exactly.
        gathered = torch.gather(b["bind_embed"], 1, b["item_pos"].unsqueeze(-1).expand(B, K, 2 * d))
        assert torch.equal(gathered, torch.cat([b["keys"], b["values"]], dim=-1))

        for row in range(B):
            vr = torch.linalg.matrix_rank(b["values"][row].float())
            assert vr >= K, f"row {row}: value rank {vr} < K={K}"

    # s_ideal composes EXACTLY (cosine ~= 1) at arbitrary h (K-cycle never dead-ends).
    for h in (*cfg.H_train, *cfg.H_test, *cfg.H_extra, 20):
        bh = sample_batch(cfg, 32, gen, hop_set=(h,))
        cur = bh["query_keys"]
        for _ in range(h):
            cur = torch.einsum("bij,bqj->bqi", bh["s_ideal"], cur)
        pred_n = cur / cur.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        tgt_n = bh["targets"] / bh["targets"].norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos = (pred_n * tgt_n).sum(dim=-1)
        assert cos.mean().item() > 1 - 1e-4, \
            f"s_ideal composition failed at h={h}: mean cos {cos.mean():.6f}"
    print("  shapes, beta_mask exactness, zero-buffer embeddings, injectivity, and "
          "s_ideal exact composition all OK")

    _test_injectivity_guard_detects_merge()

    # Buffer-length / conv_size sensitivity: conv_size=1 -> buf_len clamped to 1 (never 0).
    cfg2 = DeltaNetTaskConfig(d=8, K=4, conv_size=1, H_train=(1,), H_test=(2,), H_extra=())
    assert cfg2.buf_len == 1, "buf_len must be clamped to >= 1 even at conv_size=1"

    print("task_dn self-test PASSED")


if __name__ == "__main__":
    _self_test()
