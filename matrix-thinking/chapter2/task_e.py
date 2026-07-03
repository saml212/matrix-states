"""Task E -- compositional multi-hop relational recall (Chapter 2
reasoning-transfer gate). See NEXT_EXPERIMENT_DESIGN.md.

Each episode presents K one-hop bindings (key_i -> value_i = key_pi(i)) drawn
from a fresh orthonormal entity pool, exactly like Task D's grammar, and then
poses multi-hop queries `(a, h)` requiring pred = pi^h(a) -- literal iterated
application of the relation, not a single lookup. Two variants of the edge
function pi, BOTH required by the design spec:

  - "permutation" (PRIMARY): pi is a SINGLE Hamiltonian K-cycle over the
    K-subset (NOT a general random permutation -- a general permutation can
    decompose into several independent, generally-short cycles, which makes
    pi^h periodic with an entity-dependent, often-short period and silently
    collapses many nominal "held-out-hop" queries into in-distribution or
    even trivial-identity ones; see gauntlet/AUDIT_task_e_validity.md Finding
    B). With a single K-cycle, pi^h is periodic ONLY with period K (never
    shorter), so held-out depth queries at arbitrarily large h are always
    well-formed AND non-periodic-confounded by construction, as long as
    H_test/H_extra avoid h % K == 0 and h % K in the training residues --
    enforced by TaskEConfig.__post_init__, not just assumed. This closes the
    MNNS/B-4 "frontier collapses at unspecified depth" trap outright.
  - "chain" (secondary/robustness): pi is a genuinely partial injection into a
    larger pool (K < N); unbound entities are absorbing terminals. Chains are
    finite, and query validity (chain length >= h) is CHECKED and ASSERTED at
    generation time, never assumed.

**Injectivity (C6, load-bearing).** The one-hop rank(Z) >= K proof
(TASK_D_PREREGISTRATION.md section 3) requires linearly independent keys AND
values. Keys are the K pool entities (orthonormal by construction). Values are
{key_pi(i)}, independent ONLY because pi is injective (in-degree <= 1) -- the
exact "K edges != K rank constraints" miscounting trap that killed MNNS
(KILL_LIST.md item 1). Both generators below are injective BY CONSTRUCTION
(a genuine permutation, and a subgraph of a single Hamiltonian path,
respectively -- neither can produce a merge), and this is additionally
verified at runtime via `_assert_injective` (a rank check on the stacked
values) plus a dedicated NEGATIVE unit test in `_self_test` that deliberately
constructs a merge and confirms the check detects it.

Self-contained: torch + stdlib only, no src/ imports (pod-safe). Reuses
task_d.py's `_random_directions` and `recovery_cosine` verbatim (same-directory
sibling import, matching run_task_d.py's own import of task_d).

**Audit fixes applied 2026-07-01** (gauntlet/AUDIT_task_e_correctness.md,
gauntlet/AUDIT_task_e_validity.md): (1) `_assert_injective`'s rank threshold
tightened from a one-off-tolerant `>= K_eff - 1` to an EXACT `>= K_eff` -- the
old slack could not detect a single-pair merge, the exact failure mode C6
exists to catch, and the codebase's own negative unit test proved it. (2) the
permutation variant's `pi` is now a single Hamiltonian K-cycle, not a general
permutation, closing the cycle-length-periodicity confound on M3_E (Finding
B) -- see the updated docstring below and TaskEConfig.__post_init__'s new
periodicity guard.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

import task_d as td


@dataclass(frozen=True)
class TaskEConfig:
    d: int = 16                 # vector dimension (= matrix side)
    K: int = 8                  # number of one-hop bindings (edges) per sample
    variant: str = "permutation"  # "permutation" | "chain"
    N: int | None = None        # entity-pool size; None -> K (permutation) or K+4 (chain)
    n_query: int | None = None  # (a, h) query slots per sample; None -> K
    H_train: tuple[int, ...] = (1, 2, 3)
    H_test: tuple[int, ...] = (4, 5, 6)
    # Further-out graceful-degradation probe. (7, 21) -- NOT (8, 10) -- chosen
    # so that h % K avoids {0} U {h % K : h in H_train} for every K in the
    # planned Stage-2 sweep {8, 12, 16} (K=4 dropped entirely, see
    # __post_init__ and gauntlet/AUDIT_task_e_validity.md Finding B). The old
    # default (8, 10) was self-defeating at the K=8 primary operating point:
    # 8 % 8 == 0 (identity) and 10 % 8 == 2 (== a training hop's residue).
    H_extra: tuple[int, ...] = (7, 21)
    orthogonal: bool = True     # GATE DEFAULT (Task D audit: Gaussian smears the knee)

    @property
    def pool_size(self) -> int:
        if self.N is not None:
            return self.N
        return self.K if self.variant == "permutation" else self.K + 4

    @property
    def queries(self) -> int:
        return self.K if self.n_query is None else self.n_query

    @property
    def H_max(self) -> int:
        vals = list(self.H_train) + list(self.H_test) + list(self.H_extra)
        return max(vals)

    def __post_init__(self):
        assert self.variant in ("permutation", "chain"), self.variant
        assert self.d >= 1 and self.K >= 1
        assert len(self.H_train) >= 1 and len(self.H_test) >= 1
        assert min(self.H_train) >= 1, "hop depth must be >= 1"
        assert not (set(self.H_train) & set(self.H_test)), \
            "H_train and H_test must be disjoint -- M3_E is a HELD-OUT-hop test"
        if self.H_extra:
            assert not (set(self.H_train) & set(self.H_extra)) and \
                  not (set(self.H_test) & set(self.H_extra)), \
                  "H_extra must not overlap H_train/H_test"
        N = self.pool_size
        assert N <= self.d, f"pool size N={N} must be <= d={self.d} for an orthonormal pool"
        if self.variant == "permutation":
            assert N == self.K, \
                "permutation variant requires N == K (every pool entity is bound; cycles only)"
            # Periodicity guard (validity audit Finding B). pi is now built as
            # a SINGLE Hamiltonian K-cycle (_permutation_graph), so pi^h is
            # periodic ONLY with period K -- pi^h(a) == pi^(h mod K)(a) for
            # EVERY entity (they all share the one cycle). A held-out hop
            # whose h mod K lands on 0 (identity) or on a TRAINING hop's
            # residue is therefore secretly trivial or in-distribution, not
            # genuinely held-out -- exactly the confound that made the old
            # general-permutation generator's M3_E uninterpretable. Assert it
            # away at config-construction time rather than discovering it via
            # a contaminated sweep result.
            train_residues = {h % self.K for h in self.H_train}
            for h in (*self.H_test, *self.H_extra):
                r = h % self.K
                assert r != 0, \
                    f"H_test/H_extra hop h={h} is a multiple of K={self.K}: with a " \
                    f"single K-cycle, pi^{h} is the IDENTITY -- this probe is " \
                    f"confounded (measures nothing), not held-out composition. Pick " \
                    f"a different h (see gauntlet/AUDIT_task_e_validity.md Finding B)."
                assert r not in train_residues, \
                    f"H_test/H_extra hop h={h} has h % K={self.K} == {r}, which " \
                    f"coincides with a TRAINING hop's residue " \
                    f"({sorted(train_residues)}) -- with a single K-cycle this query " \
                    f"is secretly IN-DISTRIBUTION, not held-out. Pick a different h " \
                    f"(see gauntlet/AUDIT_task_e_validity.md Finding B)."
        else:
            assert self.K < N, \
                "chain variant requires K < N: a GENUINELY partial injection with absorbing sinks"
            H_max = self.H_max
            assert N - 1 >= H_max, \
                f"pool too small: need N-1={N - 1} >= H_max={H_max} (a Hamiltonian path over " \
                f"N entities has only N-1 possible edges)"
            assert self.K >= H_max, \
                f"K={self.K} must be >= H_max={H_max}: a chain of length H_max needs >= H_max edges"


# ---------------------------------------------------------------------------
# Edge-function generators (both injective by construction; see module docstring)
# ---------------------------------------------------------------------------

def _permutation_graph(B: int, K: int, gen: torch.Generator, device, dtype) -> torch.Tensor:
    """Random SINGLE Hamiltonian K-cycle per row -- NOT a general random
    permutation. succ[b, i] = pi(i), a BIJECTION on {0..K-1}, injective by
    construction (a bijection cannot merge two sources onto the same target),
    AND guaranteed to be exactly ONE cycle of length K (never several smaller
    disjoint cycles).

    A general uniform-random permutation (the previous implementation: a
    single argsort-of-noise treated directly as succ) typically decomposes
    into MULTIPLE independent cycles of varying, often short, length. Because
    pi^h is periodic with the period of the cycle an entity belongs to, that
    makes pi^h(a) secretly collapse to pi^(h mod ell)(a) for an
    entity-dependent, often-small ell -- silently turning many nominal
    "held-out hop" queries into in-distribution or trivial-identity queries
    (gauntlet/AUDIT_task_e_validity.md Finding B, empirically measured: 100%
    collapse at K=8, h=8; 100% collapse at every held-out hop for K=4).

    Forcing a SINGLE K-cycle removes that escape hatch entirely: every entity
    shares the same cycle, so pi^h is periodic ONLY with period K, uniformly.
    TaskEConfig.__post_init__ then only has to guard against ONE thing --
    h % K landing on 0 or a training residue -- instead of an unbounded
    number of entity-dependent short cycles.

    Construction: sample a uniformly random ordering `order` of {0..K-1} (the
    same argsort-of-noise trick as before), then chain
    order[0] -> order[1] -> ... -> order[K-1] -> order[0]:
    succ[b, order[b, i]] = order[b, (i + 1) % K].
    """
    order = torch.rand(B, K, generator=gen, device=device, dtype=dtype).argsort(dim=-1)
    next_order = torch.roll(order, shifts=-1, dims=-1)   # order[i] -> order[i+1], wrapping
    succ = torch.empty(B, K, dtype=torch.int64, device=device)
    succ.scatter_(1, order, next_order)
    return succ


def _iterate_permutation(succ: torch.Tensor, a_idx: torch.Tensor,
                         hops: torch.Tensor) -> torch.Tensor:
    """pi^h(a) by exact index iteration -- the numerically EXACT ground-truth
    label (no floating-point matmul involved). succ: (B, K); a_idx, hops: (B, Q)."""
    cur = a_idx.clone()
    result = torch.where(hops == 0, cur, torch.full_like(cur, -1))
    max_h = int(hops.max().item()) if hops.numel() > 0 else 0
    for t in range(1, max_h + 1):
        cur = torch.gather(succ, 1, cur)      # advance ALL queries one hop in lockstep
        result = torch.where(hops == t, cur, result)
    assert (result >= 0).all(), "a query's hop depth exceeded the iteration bound"
    return result


def _chain_graph(B: int, N: int, K: int, H_max: int, gen: torch.Generator,
                 device, dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Disjoint-union-of-simple-paths functional graph: a random topological
    order over N pool positions, with exactly K of the N-1 possible "forward"
    edges (position i -> i+1) active. Because every edge respects the
    topological order, the graph is ACYCLIC (chains are finite) and injective
    by construction (targets = distinct sources + 1, hence distinct).

    Guarantees, by construction (not by hope), that a run of length >= H_max
    exists: H_max of the K active edges are forced into one contiguous block
    at a random offset, so every required hop depth up to H_max is always
    satisfiable -- the "checked and asserted at generation time, not assumed"
    fix the MNNS post-mortem demands (NEXT_EXPERIMENT_DESIGN.md section 2).

    Returns:
      order    (B, N)    int64  random permutation of pool positions (topological order)
      active   (B, N-1)  bool   True if edge order[i]->order[i+1] is present (exactly K True/row)
      run_len  (B, N)    int64  longest valid hop count starting at topological position i
    """
    order = torch.rand(B, N, generator=gen, device=device, dtype=dtype).argsort(dim=-1)

    start = torch.randint(0, N - H_max, (B,), generator=gen, device=device)  # start+H_max <= N-1
    pos = torch.arange(N - 1, device=device).unsqueeze(0).expand(B, N - 1)
    forced = (pos >= start.unsqueeze(1)) & (pos < (start + H_max).unsqueeze(1))   # (B, N-1)

    extra_needed = K - H_max
    assert extra_needed >= 0
    if extra_needed > 0:
        noise = torch.rand(B, N - 1, generator=gen, device=device, dtype=dtype)
        noise = noise.masked_fill(forced, float("inf"))     # never re-pick a forced slot
        extra_rank = noise.argsort(dim=-1).argsort(dim=-1)   # 0-indexed rank per row
        extra = (extra_rank < extra_needed) & ~forced
    else:
        extra = torch.zeros_like(forced)
    active = forced | extra                                  # exactly K True per row

    run_len = torch.zeros(B, N, dtype=torch.int64, device=device)
    zeros_b = torch.zeros(B, dtype=torch.int64, device=device)
    for i in range(N - 2, -1, -1):
        run_len[:, i] = torch.where(active[:, i], run_len[:, i + 1] + 1, zeros_b)
    return order, active, run_len


def _chain_edge_positions(active: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the K active (source) positions per row, in ascending original
    order (stable-sort trick), plus their target positions (source + 1)."""
    order_by_active = torch.argsort((~active).to(torch.float32), dim=-1, stable=True)
    src_pos = order_by_active[:, :K]
    tgt_pos = src_pos + 1
    return src_pos, tgt_pos


def _sample_valid_start(run_len: torch.Tensor, hops: torch.Tensor, gen: torch.Generator,
                        device, dtype) -> torch.Tensor:
    """For each (row, query), pick a start position i uniformly among
    {i : run_len[b,i] >= hops[b,q]}. Existence for every sampled hop is
    guaranteed by `_chain_graph`'s forced contiguous run -- ASSERTED here
    (not just assumed), the exact discipline the MNNS post-mortem requires."""
    B, N = run_len.shape
    valid = run_len.unsqueeze(1) >= hops.unsqueeze(-1)     # (B, Q, N)
    assert valid.any(dim=-1).all(), "no valid start position exists for a sampled hop depth"
    noise = torch.rand(B, hops.shape[1], N, generator=gen, device=device, dtype=dtype)
    noise = noise.masked_fill(~valid, -1.0)
    return noise.argmax(dim=-1)                             # (B, Q)


# ---------------------------------------------------------------------------
# Idealized reference operator (C7)
# ---------------------------------------------------------------------------

def ideal_Z(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """C7: the classical, ZERO-TRAINING minimum-norm operator
    Z_ideal = sum_i value_i key_i^T, computed analytically from the SAME
    (key, value) bindings the encoder is trained on. Because keys are exactly
    orthonormal and pi is injective, Z_ideal @ key_i = value_i EXACTLY, and
    therefore Z_ideal^h @ key_a = value at hop h EXACTLY for every valid h --
    classical linear-associative-memory chaining (Kohonen 1972/73; Anderson
    1972), not claimed as novel (NEXT_EXPERIMENT_DESIGN.md section 3). Under
    `orthogonal=False` (Gaussian) keys this is only approximately exact --
    the exactness claim is specific to the orthonormal gate default.
    keys, values: (B, K, d) -> Z_ideal: (B, d, d).
    """
    return torch.einsum("bki,bkj->bij", values, keys)


def _iterate_matvec(Z: torch.Tensor, v: torch.Tensor, h: int) -> torch.Tensor:
    """Reference iterated matmul pred[b,q] = Z[b]^h @ v[b,q], used ONLY by
    `_self_test` to confirm Z_ideal composes exactly. Deliberately an
    INDEPENDENT re-implementation (not imported from model_e.py) so a self-test
    bug in the model's own readout can't silently pass; keeps task_e.py free of
    model imports (generators shouldn't depend on models). v is (B, Q, d) --
    the per-query batch of start vectors (Q = cfg.queries, not 1), so the
    einsum must carry the query axis (bqj->bqi), matching compose()."""
    cur = v
    for _ in range(h):
        cur = torch.einsum("bij,bqj->bqi", Z, cur)
    return cur


def _assert_injective(values: torch.Tensor) -> None:
    """C6 (load-bearing): confirm the generated value set is linearly
    independent (rank == K), the necessary condition for rank(Z) >= K to carry
    over to Task E. Checks ONE row per call (cheap; O(K^3)) because injectivity
    here is a STRUCTURAL guarantee of the graph construction, not a
    probabilistic one -- if it fails for one row the construction is wrong for
    EVERY row and must be caught immediately. `_self_test` additionally checks
    every row of several full batches, and `_test_injectivity_guard_detects_merge`
    proves this check has actual discriminating power (not a vacuous pass)."""
    K = values.shape[1]
    K_eff = min(K, values.shape[-1])
    vrank = torch.linalg.matrix_rank(values[0].float())
    # EXACT threshold, no "-1" slack: a single accidental merge (two sources
    # -> one target) drops rank by exactly 1, so a "-1" tolerance can never
    # detect the single most realistic failure mode this guard exists to
    # catch (gauntlet/AUDIT_task_e_correctness.md F1,
    # gauntlet/AUDIT_task_e_validity.md Finding A -- both independently
    # verified this via a from-scratch rank computation). Genuine, non-merged
    # batches have vrank == K_eff exactly (orthonormal-QR-constructed keys),
    # so this is safe, not just tighter.
    assert vrank >= K_eff, \
        f"generated values have rank {vrank} < K_eff={K_eff} (K={K}); the in-degree<=1 " \
        f"injectivity guarantee is violated -- the rank(Z)>=K proof premise is broken"


# ---------------------------------------------------------------------------
# Batch sampler
# ---------------------------------------------------------------------------

def sample_batch(cfg: TaskEConfig, batch_size: int, gen: torch.Generator,
                 hop_set: tuple, device="cpu", dtype=torch.float32,
                 assert_injective: bool = True) -> dict:
    """Generate one batch of Task E multi-hop relational-recall problems.

    hop_set: candidate hop depths for THIS call's queries -- each query
      independently samples h ~ Uniform(hop_set) (matches
      NEXT_EXPERIMENT_DESIGN.md section 2's "h ~ Uniform{1..H_train}"). Pass a
      single-element tuple, e.g. (5,), to force every query in the batch to
      that exact hop depth (used to build the per-hop M3_E held-out curve);
      pass a multi-element tuple (e.g. cfg.H_train) for mixed-hop TRAINING
      batches.

    assert_injective: run the C6 value-rank injectivity guard on this batch.
      Injectivity is a STRUCTURAL property of the graph construction (not
      probabilistic), so it holds identically for every batch of a given
      config -- the guard only needs to fire once per config, plus
      exhaustively in the smoke gate. `train()` therefore checks the FIRST
      batch (assert_injective=True) and disables it thereafter to avoid a
      GPU->CPU sync barrier (an O(K^3) matrix_rank + a Python assert) on every
      one of ~20-30K training steps. Eval and the self-test keep it on.

    Returns a dict of tensors:
      keys, values  (B, K, d)  the K one-hop bindings (encoder input --
                                UNCHANGED from Task D's grammar; the encoder
                                never sees h or the query, C_composition-purity)
      query_keys    (B, Q, d)  query start-entity vectors
      hops          (B, Q)     int64 hop depth per query, drawn from hop_set
      targets       (B, Q, d)  ground truth pi^h(a), via exact index
                                composition (numerically exact label)
      z_ideal       (B, d, d)  the classical minimum-norm reference operator (C7)
    """
    B, K, d = batch_size, cfg.K, cfg.d
    Q = cfg.queries
    N = cfg.pool_size
    pool = td._random_directions(B, N, d, cfg.orthogonal, gen, device, dtype)

    hops_pool = torch.tensor(hop_set, device=device, dtype=torch.int64)
    hops = hops_pool[torch.randint(0, len(hop_set), (B, Q), generator=gen, device=device)]

    if cfg.variant == "permutation":
        succ = _permutation_graph(B, K, gen, device, dtype)      # pi: {0..K-1} -> {0..K-1}
        keys = pool[:, :K, :]
        values = torch.gather(pool, 1, succ.unsqueeze(-1).expand(B, K, d))
        if assert_injective:
            _assert_injective(values)

        a_idx = torch.randint(0, K, (B, Q), generator=gen, device=device)   # always valid (cycles)
        tgt_idx = _iterate_permutation(succ, a_idx, hops)
        query_keys = torch.gather(pool, 1, a_idx.unsqueeze(-1).expand(B, Q, d))
        targets = torch.gather(pool, 1, tgt_idx.unsqueeze(-1).expand(B, Q, d))

    elif cfg.variant == "chain":
        H_max = cfg.H_max
        order, active, run_len = _chain_graph(B, N, K, H_max, gen, device, dtype)
        src_pos, tgt_pos = _chain_edge_positions(active, K)
        gather_src = torch.gather(order, 1, src_pos)
        gather_tgt = torch.gather(order, 1, tgt_pos)
        keys = torch.gather(pool, 1, gather_src.unsqueeze(-1).expand(B, K, d))
        values = torch.gather(pool, 1, gather_tgt.unsqueeze(-1).expand(B, K, d))
        if assert_injective:
            _assert_injective(values)

        chosen_pos = _sample_valid_start(run_len, hops, gen, device, dtype)   # (B, Q)
        assert torch.gather(run_len, 1, chosen_pos).ge(hops).all(), \
            "generated a query whose chain is shorter than its own sampled hop depth"
        tgt_chain_pos = chosen_pos + hops
        query_keys = torch.gather(
            pool, 1, torch.gather(order, 1, chosen_pos).unsqueeze(-1).expand(B, Q, d))
        targets = torch.gather(
            pool, 1, torch.gather(order, 1, tgt_chain_pos).unsqueeze(-1).expand(B, Q, d))
    else:
        raise ValueError(f"unknown variant {cfg.variant}")

    z_ideal = ideal_Z(keys, values)
    return {"keys": keys, "values": values, "query_keys": query_keys,
            "hops": hops, "targets": targets, "z_ideal": z_ideal}


# ---------------------------------------------------------------------------
# Self-test (runs where torch is available; part of the smoke gate on-cluster)
# ---------------------------------------------------------------------------

def _test_injectivity_guard_detects_merge() -> None:
    """Construct a deliberately NON-injective (merging) value set -- two
    distinct keys mapping to the SAME value -- and confirm `_assert_injective`
    raises. This is the concrete unit test NEXT_EXPERIMENT_DESIGN.md section 2
    and section 11 require: proof the injectivity guard is actually CHECKING
    (not assuming) and has discriminating power -- the MNNS/B-4 miscounting
    trap, closed with a test that would fail if the guard were vacuous."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    d, K = 16, 8
    pool = td._random_directions(1, K, d, True, gen, "cpu", torch.float32)
    values = pool.clone()
    values[:, 1, :] = values[:, 0, :]          # force a merge: keys 0 and 1 -> same value
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

    # ---- Permutation variant (primary) ----
    cfg_p = TaskEConfig(d=16, K=8, variant="permutation",
                        H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    for _ in range(5):
        b = sample_batch(cfg_p, 64, gen, hop_set=cfg_p.H_train)
        B, K, d, Q = 64, cfg_p.K, cfg_p.d, cfg_p.queries
        assert b["keys"].shape == (B, K, d)
        assert b["values"].shape == (B, K, d)
        assert b["query_keys"].shape == (B, Q, d)
        assert b["hops"].shape == (B, Q)
        assert b["targets"].shape == (B, Q, d)
        assert b["z_ideal"].shape == (B, d, d)
        assert set(torch.unique(b["hops"]).tolist()) <= set(cfg_p.H_train)

        for row in range(B):     # self-test can afford the full-batch O(K^3) rank check
            vr = torch.linalg.matrix_rank(b["values"][row].float())
            assert vr >= K, f"row {row}: value rank {vr} < K={K}"     # EXACT, no slack (F1)

    # Z_ideal composes EXACTLY (cosine ~= 1) for every hop, including held-out
    # and far-out ones -- permutations never dead-end, so this must hold at
    # ARBITRARY h, not just h <= H_max.
    for h in (*cfg_p.H_train, *cfg_p.H_test, *cfg_p.H_extra, 20):
        bh = sample_batch(cfg_p, 32, gen, hop_set=(h,))
        pred = _iterate_matvec(bh["z_ideal"], bh["query_keys"], h)
        cos = td.recovery_cosine(pred, bh["targets"])
        assert cos.mean().item() > 1 - 1e-4, \
            f"permutation Z_ideal composition failed at h={h}: mean cos {cos.mean():.6f}"
    print("  [permutation] shapes, injectivity, and Z_ideal exact composition all OK")

    # ---- Chain-into-sink variant (secondary/robustness) ----
    cfg_c = TaskEConfig(d=24, K=12, variant="chain", N=16,
                        H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(8, 10))
    for _ in range(5):
        b = sample_batch(cfg_c, 64, gen, hop_set=cfg_c.H_train)
        B, K, d, Q = 64, cfg_c.K, cfg_c.d, cfg_c.queries
        assert b["keys"].shape == (B, K, d)
        assert b["values"].shape == (B, K, d)
        for row in range(B):
            vr = torch.linalg.matrix_rank(b["values"][row].float())
            assert vr >= K, f"row {row}: chain value rank {vr} < K={K}"       # EXACT, no slack (F1)

    for h in (*cfg_c.H_train, *cfg_c.H_test, *cfg_c.H_extra):
        bh = sample_batch(cfg_c, 32, gen, hop_set=(h,))
        pred = _iterate_matvec(bh["z_ideal"], bh["query_keys"], h)
        cos = td.recovery_cosine(pred, bh["targets"])
        assert cos.mean().item() > 1 - 1e-4, \
            f"chain Z_ideal composition failed at h={h}: mean cos {cos.mean():.6f}"
    print("  [chain] shapes, injectivity, and Z_ideal exact composition all OK "
          "(query validity checked+asserted at generation time, not assumed)")

    # ---- Injectivity guard has TEETH ----
    _test_injectivity_guard_detects_merge()

    print("task_e self-test PASSED")


if __name__ == "__main__":
    _self_test()
