"""NCR operator-bank task module -- NOVEL_ARCH_WATERFALL.md S8.1/S8.3/S8.3a.

R=3 independent relations (each a SINGLE Hamiltonian K-cycle, task_e's own
generator reused verbatim per relation) sharing ONE K=8 orthonormal entity
pool in d=16. One episode presents all R*K=24 bindings together, each tagged
with a relation id -- the "written from one shared context" design that
makes cross-relation interference a genuine empirical question (S8.1.1).

Two query axes:
  Axis R-BANK  (headline): single-block (r, h) -> pred = pi_r^h(a).
  Axis B-CHAIN (exploratory, non-headline, B FIXED at 2): (r1,h1,r2,h2) ->
               pred = pi_r2^h2(pi_r1^h1(a)), r1 != r2 required (r1==r2 is a
               disclosed internal consistency check only, never scored).

Held-out depth reuses ncr_task.GRIDS[8] verbatim, per relation -- r is an
orthogonal selection axis, not a depth-periodicity axis (S8.1.3), so the
EXISTING task_e.TaskEConfig.__post_init__ mod-K guard applies unmodified,
called once per relation.

Self-contained apart from chapter2 (task_d, task_e) and ncr_task imports.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
if CHAPTER2 not in sys.path:
    sys.path.insert(0, CHAPTER2)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import task_d as td            # noqa: E402
import task_e as te            # noqa: E402 (mod-K guard + single-cycle generator, verbatim)
import ncr_task as nt          # noqa: E402 (pinned GRIDS[8], residue_label)

D_PIN = 16
K_PIN = 8
R_PIN = 3                       # S8.1.1: bank size, fixed for this wave
H_TRAIN = (1, 2, 3)
GRID8 = nt.GRIDS[8]


@dataclass(frozen=True)
class BankConfig:
    d: int = D_PIN
    K: int = K_PIN
    R: int = R_PIN
    orthogonal: bool = True

    def __post_init__(self):
        assert self.R >= 2, "a bank needs >= 2 relations (R=1 is the single-relation program)"
        assert self.K <= self.d, (self.K, self.d)
        # per-relation reuse of task_e's own mod-K guard: construct one
        # TaskEConfig per relation and let its __post_init__ run verbatim --
        # if it doesn't raise, the guard has accepted GRID8's residues for
        # this K (identical for every relation, since they share K).
        te.TaskEConfig(d=self.d, K=self.K, variant="permutation",
                       H_train=H_TRAIN, H_test=GRID8["ladder"][:3],
                       H_extra=(GRID8["h_star"],), orthogonal=self.orthogonal)


def _relations_distinct(succ: torch.Tensor) -> bool:
    """succ: (B, R, K). True iff every pair of relations differs somewhere,
    in EVERY row -- non-raising checker (used by the retry loop below)."""
    B, R, K = succ.shape
    for r1 in range(R):
        for r2 in range(r1 + 1, R):
            if bool((succ[:, r1, :] == succ[:, r2, :]).all(dim=-1).any()):
                return False
    return True


def assert_relation_distinct(succ: torch.Tensor) -> None:
    """Raising version of `_relations_distinct`, with a diagnostic message.
    An exact structural check (no tolerance), mirroring
    task_e._assert_injective's exact-threshold convention. Negative-tested
    in ncr_opbank_selftest.py t1 (a deliberately duplicated relation must be
    caught) -- called DIRECTLY there on a constructed duplicate, bypassing
    the retry loop below (a genuine duplicate must never resolve by retry)."""
    B, R, K = succ.shape
    for r1 in range(R):
        for r2 in range(r1 + 1, R):
            same = (succ[:, r1, :] == succ[:, r2, :]).all(dim=-1)  # (B,)
            assert not bool(same.any()), (
                f"relation {r1} and {r2} are IDENTICAL for {int(same.sum())} episode(s) "
                f"in this batch -- the bank is not actually R independent relations")


def _relation_graphs(B: int, cfg: BankConfig, gen: torch.Generator,
                     device, dtype, max_retries: int = 20) -> torch.Tensor:
    """R independent Hamiltonian K-cycles, one per relation, over the SAME
    pool positions. succ: (B, R, K).

    Distinctness is CHECKED, not assumed (S8.1.1) -- but the ORIGINAL design
    text's "astronomically small" collision-probability claim was WRONG,
    caught by the real-CUDA build smoke (a case the CPU smoke happened not
    to hit): for K=8 there are (K-1)!=5040 distinct Hamiltonian-cycle
    functions, so P(>=1 collision among R=3 iid draws) = 1 - 5040*5039*5038
    / 5040**3 ~= 5.95e-4 PER EPISODE ROW -- small per draw, but a single
    Phase-0 cell's full pipeline (train steps + z_dump + blank_out + swap
    ablation + rvstd + eval grid) draws thousands of independent rows across
    dozens of batches, so the EXPECTED number of collisions per full cell
    run is ~0.5-1 (i.e. likely, not astronomical) -- exactly what surfaced
    on the box. Fix: retry the WHOLE-BATCH draw (cheap; P(any collision in
    a B=64 batch) ~= 3.7%, so >=2 retries needed only ~0.1% of the time) up
    to max_retries times before raising -- the checked invariant survives
    (a PERSISTENT duplicate, e.g. a real generator bug, still raises via
    `assert_relation_distinct` on exhaustion), only the treatment of a rare,
    expected, non-adversarial collision changes from a hard crash to a
    transparent resample."""
    for attempt in range(max_retries):
        succ = torch.stack(
            [te._permutation_graph(B, cfg.K, gen, device, dtype) for _ in range(cfg.R)],
            dim=1)  # (B, R, K)
        if _relations_distinct(succ):
            return succ
    assert_relation_distinct(succ)  # exhausted retries -- raise with the diagnostic message
    raise AssertionError("unreachable")  # pragma: no cover


def generate_bank_episode(B: int, cfg: BankConfig, gen: torch.Generator,
                          device="cpu", dtype=torch.float32) -> dict:
    """One shared entity pool (N=K, d) + R independent relation graphs.
    Returns dict(pool, succ (B,R,K), keys (B,R*K,d), values (B,R*K,d),
    rel_ids (B,R*K) int64 in [0,R)) -- the encoder's write-time input."""
    N = cfg.K
    pool = td._random_directions(B, N, cfg.d, cfg.orthogonal, gen, device, dtype)
    succ = _relation_graphs(B, cfg, gen, device, dtype)               # (B,R,K)
    keys_per_r = pool[:, :cfg.K, :].unsqueeze(1).expand(B, cfg.R, cfg.K, cfg.d)
    values_per_r = torch.gather(
        pool.unsqueeze(1).expand(B, cfg.R, N, cfg.d), 2,
        succ.unsqueeze(-1).expand(B, cfg.R, cfg.K, cfg.d))
    for r in range(cfg.R):
        te._assert_injective(values_per_r[:, r])
    keys = keys_per_r.reshape(B, cfg.R * cfg.K, cfg.d)
    values = values_per_r.reshape(B, cfg.R * cfg.K, cfg.d)
    rel_ids = torch.arange(cfg.R, device=device).repeat_interleave(cfg.K).unsqueeze(0).expand(B, -1)
    return dict(pool=pool, succ=succ, keys=keys, values=values, rel_ids=rel_ids)


def _target_idx(succ_r: torch.Tensor, a_idx: torch.Tensor, hops: torch.Tensor) -> torch.Tensor:
    """Exact index iteration for ONE relation's succ (B,K), a_idx/hops (B,Q)."""
    return te._iterate_permutation(succ_r, a_idx, hops)


def sample_train_batch(cfg: BankConfig, batch_size: int, gen: torch.Generator,
                       device="cpu", axis_b_frac: float = 0.5) -> dict:
    """Mixed train batch: axis_b_frac of queries are 2-block (r1!=r2), the
    rest single-block (Axis R-BANK). h1,h2,h (single-block) ~ Uniform(H_TRAIN)
    -- S3.1's raw-integer-h pin, unchanged. Query relation ids sampled
    uniformly over the bank."""
    ep = generate_bank_episode(batch_size, cfg, gen, device)
    Q = cfg.K  # one query slot per key, matching ncr_task's convention
    B = batch_size
    is_chain = torch.rand(B, Q, generator=gen, device=device) < axis_b_frac
    a_idx = torch.randint(0, cfg.K, (B, Q), generator=gen, device=device)
    r1 = torch.randint(0, cfg.R, (B, Q), generator=gen, device=device)
    h_pool = torch.tensor(H_TRAIN, device=device, dtype=torch.int64)
    h1 = h_pool[torch.randint(0, len(H_TRAIN), (B, Q), generator=gen, device=device)]
    # r2 uniform over the OTHER R-1 relations (r1 != r2 enforced for the
    # scored chain queries; single-block queries carry r2=r1/h2=0 as a no-op)
    r2_off = torch.randint(1, cfg.R, (B, Q), generator=gen, device=device)
    r2 = (r1 + r2_off) % cfg.R
    h2 = h_pool[torch.randint(0, len(H_TRAIN), (B, Q), generator=gen, device=device)]
    h2 = torch.where(is_chain, h2, torch.zeros_like(h2))
    r2_eff = torch.where(is_chain, r2, r1)

    pool = ep["pool"]
    query_keys = torch.gather(pool, 1, a_idx.unsqueeze(-1).expand(B, Q, cfg.d))
    mid_idx = torch.zeros_like(a_idx)
    tgt_idx = torch.zeros_like(a_idx)
    for r in range(cfg.R):
        m1 = (r1 == r)
        mid_idx = torch.where(m1, _target_idx(ep["succ"][:, r], a_idx, h1), mid_idx)
    for r in range(cfg.R):
        m2 = (r2_eff == r)
        tgt_idx = torch.where(m2, _target_idx(ep["succ"][:, r], mid_idx, h2), tgt_idx)
    targets = torch.gather(pool, 1, tgt_idx.unsqueeze(-1).expand(B, Q, cfg.d))

    ep.update(query_keys=query_keys, r1=r1, h1=h1, r2=r2_eff, h2=h2,
             is_chain=is_chain, targets=targets, a_idx=a_idx)
    return ep


def sample_eval_batch_axis_r(cfg: BankConfig, batch_size: int, gen: torch.Generator,
                             r: int, h: int, device="cpu") -> dict:
    """Held-out single-relation eval, relation r, raw depth h (per GRID8's
    mod-K hygiene, unmodified). Every query in the batch uses relation r."""
    ep = generate_bank_episode(batch_size, cfg, gen, device)
    B, Q = batch_size, cfg.K
    a_idx = torch.randint(0, cfg.K, (B, Q), generator=gen, device=device)
    h_eff = h % cfg.K
    hops = torch.full((B, Q), h_eff, dtype=torch.int64, device=device)
    tgt_idx = _target_idx(ep["succ"][:, r], a_idx, hops)
    pool = ep["pool"]
    query_keys = torch.gather(pool, 1, a_idx.unsqueeze(-1).expand(B, Q, cfg.d))
    targets = torch.gather(pool, 1, tgt_idx.unsqueeze(-1).expand(B, Q, cfg.d))
    ep.update(query_keys=query_keys, r=r, h=h, h_eff=h_eff, targets=targets, a_idx=a_idx)
    return ep


def sample_eval_batch_axis_b(cfg: BankConfig, batch_size: int, gen: torch.Generator,
                             r1: int, h1: int, r2: int, h2: int,
                             device="cpu", exclude_fixed_points: bool = True) -> dict:
    """2-block held-out eval. m1 fix (S8.3): excludes queries whose composite
    sigma = pi_r2^h2 o pi_r1^h1 fixes the start (sigma(a)==a) -- an exact
    integer check, no tolerance. Batch is OVERSAMPLED then filtered so the
    returned batch_size is exact (never silently smaller)."""
    assert r1 != r2, "Axis B-CHAIN requires r1 != r2 (r1==r2 is an internal consistency check only)"
    keep_keys, keep_values, keep_relids, keep_qk, keep_tgt = [], [], [], [], []
    n_have = 0
    oversample = batch_size
    attempts = 0
    while n_have < batch_size:
        attempts += 1
        assert attempts <= 50, "could not fill an Axis-B eval batch after 50 oversample rounds"
        ep = generate_bank_episode(oversample, cfg, gen, device)
        B, Q = oversample, cfg.K
        a_idx = torch.randint(0, cfg.K, (B, Q), generator=gen, device=device)
        h1_eff, h2_eff = h1 % cfg.K, h2 % cfg.K
        h1t = torch.full((B, Q), h1_eff, dtype=torch.int64, device=device)
        h2t = torch.full((B, Q), h2_eff, dtype=torch.int64, device=device)
        mid_idx = _target_idx(ep["succ"][:, r1], a_idx, h1t)
        tgt_idx = _target_idx(ep["succ"][:, r2], mid_idx, h2t)
        pool = ep["pool"]
        if exclude_fixed_points:
            fixed = (tgt_idx == a_idx)                        # (B, Q), exact int compare
            mask = ~fixed
        else:
            mask = torch.ones_like(a_idx, dtype=torch.bool)
        for b in range(B):
            rows = mask[b].nonzero(as_tuple=True)[0]
            for q in rows.tolist():
                if n_have >= batch_size:
                    break
                keep_keys.append(ep["keys"][b])
                keep_values.append(ep["values"][b])
                keep_relids.append(ep["rel_ids"][b])
                keep_qk.append(pool[b, a_idx[b, q]])
                keep_tgt.append(pool[b, tgt_idx[b, q]])
                n_have += 1
            if n_have >= batch_size:
                break
        oversample = max(oversample, batch_size - n_have)
    return dict(keys=torch.stack(keep_keys), values=torch.stack(keep_values),
               rel_ids=torch.stack(keep_relids),
               query_keys=torch.stack(keep_qk).unsqueeze(1),
               targets=torch.stack(keep_tgt).unsqueeze(1),
               r1=r1, h1=h1, r2=r2, h2=h2)


# ---------------------------------------------------------------------------
# Bank-score aggregation (S8.1.6 as fixed by S8.3/S8.3a: median-of-per-seed-
# mins, NOT min-of-medians -- C1's fix, independently re-attacked and closed).
# ---------------------------------------------------------------------------

def bank_score(per_seed_per_relation: dict) -> dict:
    """per_seed_per_relation: {seed: {r: recovered_frac@0.9}}. Returns the
    median-of-per-seed-mins bank score, n_seeds_all3_hold, and (S8.3a MINOR
    (a), folded) an explicit gate requiring n_seeds_all3_hold >=
    ceil(n_seeds/2) for a HOLD verdict -- robustifies the even-seed-count
    case where two moderate values could otherwise misrepresent the
    population."""
    seeds = sorted(per_seed_per_relation)
    n = len(seeds)
    assert n >= 1, "bank_score requires at least one seed"
    per_seed_min = [min(per_seed_per_relation[s].values()) for s in seeds]
    sm = sorted(per_seed_min)
    if n % 2 == 1:
        median = sm[n // 2]
    else:
        median = (sm[n // 2 - 1] + sm[n // 2]) / 2.0
    n_all3_hold = sum(1 for v in per_seed_min if v >= 0.9)
    import math
    quorum = math.ceil(n / 2)
    band = "HOLD" if median >= 0.9 else ("DEGRADED" if median > 0.5 else "FAIL")
    hold_gated = bool(band == "HOLD" and n_all3_hold >= quorum)
    return dict(per_seed_min=dict(zip(seeds, per_seed_min)), median=median,
               band=band, n_seeds_all3_hold=n_all3_hold, quorum=quorum,
               hold_gated=hold_gated)


def _self_test():
    cfg = BankConfig()
    gen = torch.Generator().manual_seed(0)
    ep = generate_bank_episode(8, cfg, gen)
    assert ep["keys"].shape == (8, cfg.R * cfg.K, cfg.d)
    assert ep["rel_ids"].shape == (8, cfg.R * cfg.K)
    tb = sample_train_batch(cfg, 8, gen)
    assert tb["targets"].shape == (8, cfg.K, cfg.d)
    eb = sample_eval_batch_axis_r(cfg, 8, gen, r=0, h=GRID8["h_star"])
    assert eb["targets"].shape[0] == 8
    ebb = sample_eval_batch_axis_b(cfg, 4, gen, r1=0, h1=GRID8["h_star"],
                                   r2=1, h2=GRID8["h_star"])
    assert ebb["targets"].shape == (4, 1, cfg.d)
    # C1 counterexample re-check: 3 seeds, each holds 2/3 relations,
    # every per-relation median = 1.0 (old formula: FALSE HOLD/1.0); fixed
    # formula must report FAIL/0.0.
    psr = {0: {0: 1.0, 1: 1.0, 2: 0.0}, 1: {0: 1.0, 1: 0.0, 2: 1.0},
          2: {0: 0.0, 1: 1.0, 2: 1.0}}
    old_min_of_medians = min(
        sorted(psr[s][r] for s in psr)[len(psr) // 2] for r in range(3))
    assert old_min_of_medians == 1.0, "the C1 counterexample must reproduce the OLD bug"
    bs = bank_score(psr)
    assert bs["median"] == 0.0 and bs["band"] == "FAIL", bs
    assert bs["n_seeds_all3_hold"] == 0
    # negative test: relation-distinctness assert has teeth
    succ_dup = torch.stack([te._permutation_graph(4, cfg.K, gen, "cpu", torch.float32)] * 2, dim=1)
    try:
        assert_relation_distinct(succ_dup)
        raise AssertionError("distinctness check FAILED to catch a duplicated relation")
    except AssertionError as e:
        assert "IDENTICAL" in str(e), e
    print("ncr_opbank_task self-test PASSED")


if __name__ == "__main__":
    _self_test()
