"""NCR wave-1 eval-grid module -- the pinned h-grids, residue labels, and the
two-mode (claim / residue_sweep) eval pathway of NOVEL_ARCH_WATERFALL.md S3.2
(Rev 2, MA2).

Design of record: NOVEL_ARCH_WATERFALL.md S3 as amended by S3.9; harness
inherited VERBATIM from chapter2/task_e.py (the CLAUDE.md mod-K hard rule's
own fix: single Hamiltonian K-cycle `_permutation_graph` + the
`TaskEConfig.__post_init__` periodicity guard).

TWO-MODE PATHWAY (S3.2 MA2, implemented here rather than by editing
task_e.py -- chapter2 is another lane's file ownership; the SEMANTICS are the
pinned ones):

  - "claim" (default; the ONLY mode any claim-feeding path may use): every
    claim point (train-support, legacy, ladder incl. h*, cost-probe) is
    baked into a literal TaskEConfig(H_train/H_test/H_extra), so the
    INHERITED __post_init__ periodicity assert runs VERBATIM over every one
    of them at config-construction time. A claim-mode request for an h with
    h % K == 0 or h % K in the train residues crashes exactly like task_e's
    own assert (negative-tested in ncr_selftest.py).
  - "residue_sweep" (the sweep component ONLY): points 57..64 (K=8) /
    49..60 (K=12) bypass the config assert (they never enter a TaskEConfig
    hop set) and instead REQUIRE a per-point residue_label in
    {identity, train-residue, novel} in the results schema
    (`require_residue_label`, enforced + negative-tested). identity and
    train-residue points are EXCLUDED from all generalization claims and
    aggregates but INCLUDED in the reducer-detection signature (S3.2's
    disclosed confound probe).

FAR-h GROUND-TRUTH LABELS (`sample_eval_batch`): task_e's
`_iterate_permutation` costs O(h) index gathers, which is wasteful/slow at
the cost-probe depths (h up to 2^20+5). On a SINGLE K-cycle pi^h ==
pi^(h mod K) EXACTLY (integer index composition, no floating point), so
labels are computed at the reduced depth -- LABEL PATH ONLY. The batch's
`hops` tensor is then overwritten with the RAW h before any model sees it
(S3.1 depth-signal pin: every arm receives the raw integer h; no arm ever
receives h mod K). Equality of reduced-label and full-h-label batches is
proven by a self-test at moderate h (ncr_selftest.py), not assumed.

Self-contained apart from chapter2 imports (task_e). numpy-free.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
CHAPTER2 = os.path.abspath(os.path.join(_HERE, "..", "chapter2"))
if CHAPTER2 not in sys.path:
    sys.path.insert(0, CHAPTER2)

import task_e as te  # noqa: E402  (inherited harness, verbatim)

D_PIN = 16                    # S3.1: d=16, fp32 throughout
H_TRAIN = (1, 2, 3)           # S3.1 train pin (backprop through <=3 naive matmuls)
H_TEST_LEGACY = (4, 5, 6)     # Task E's own held-out set, retained for table continuity
H_LEGACY_EXTRA = 7            # the remaining legacy point (task_e default H_extra[0])

# S3.2 pinned grids. Every ladder point is == h* residue (5 mod 8 / 9 mod 12);
# residues asserted at import time below (paranoia teeth, not just prose).
GRIDS = {
    8: dict(
        ladder=(5, 13, 21, 29, 61, 125, 253, 509, 1021),
        h_star=61,
        sweep=tuple(range(57, 65)),                    # 57..64 incl. identity 64
        cost_probe=(2**10 + 5, 2**14 + 5, 2**17 + 5, 2**20 + 5),
        ladder_residue=5,
    ),
    12: dict(
        ladder=(9, 21, 45, 93, 189, 381, 765, 1533),
        h_star=57,                                     # MA3: KEPT, asymmetric confidence
        sweep=tuple(range(49, 61)),                    # 49..60 incl. identity 60
        cost_probe=(),                                 # cost probe pinned ==5 mod 8, K=8 only
        ladder_residue=9,
    ),
    # S9.7 NCR write-capacity diagnostic (matrix-thinking/NOVEL_ARCH_WATERFALL.md
    # S9.7): ADDITIVE ONLY -- K=8/K=12 above are byte-identical to their
    # pre-S9.7 values (regression-tested in ncr_wcap_selftest.py). Construction
    # rule, verified to reproduce the K=8/K=12 rows above from one closed form:
    # ladder_residue = K-3; ladder h_m = m*K-3 for m in {1,2,4,8,16,32,64,128};
    # h_star = 8*K-3 (the m=8 rung, ON-ladder -- a PROVISIONAL placeholder,
    # disclosed as uncalibrated per S9.7, not a claimed crossover); sweep = K
    # consecutive residues ending at the identity point h_star+3. cost_probe
    # deferred (), mirroring K=12's own convention.
    14: dict(
        ladder=(11, 25, 53, 109, 221, 445, 893, 1789),
        h_star=109,
        sweep=tuple(range(99, 113)),                   # 99..112 incl. identity 112
        cost_probe=(),
        ladder_residue=11,
    ),
    15: dict(
        ladder=(12, 27, 57, 117, 237, 477, 957, 1917),
        h_star=117,
        sweep=tuple(range(106, 121)),                  # 106..120 incl. identity 120
        cost_probe=(),
        ladder_residue=12,
    ),
    16: dict(
        ladder=(13, 29, 61, 125, 253, 509, 1021, 2045),
        h_star=125,
        sweep=tuple(range(113, 129)),                  # 113..128 incl. identity 128
        cost_probe=(),
        ladder_residue=13,
    ),
    # NOVEL_ARCH_WATERFALL.md S11 (early-LN K-scaling): ADDITIVE ONLY -- the
    # SAME closed form as the S9.7 keys above (14/15/16), one more K rung.
    # ladder_residue = K-3 = 21; ladder h_m = m*24-3 for m in
    # {1,2,4,8,16,32,64,128}; h_star = 8*24-3 = 189 (the m=8 rung, ON-ladder);
    # sweep = 24 consecutive residues ending at the identity point h_star+3
    # = 192. Regression-tested against a byte-identical snapshot alongside
    # GRIDS[8]/[12]/[14]/[15]/[16] (ncr_earlyln_scale_selftest.py t01).
    24: dict(
        ladder=(21, 45, 93, 189, 381, 765, 1533, 3069),
        h_star=189,
        sweep=tuple(range(169, 193)),                  # 169..192 incl. identity 192
        cost_probe=(),
        ladder_residue=21,
    ),
}

for _K, _g in GRIDS.items():
    for _h in _g["ladder"]:
        assert _h % _K == _g["ladder_residue"], (_K, _h)
    assert _g["h_star"] % _K == _g["ladder_residue"], (_K, _g["h_star"])
    for _h in _g["cost_probe"]:
        assert _h % _K == _g["ladder_residue"], (_K, _h)

TAUS = (0.9, 0.95, 0.99)      # @0.9 bar of record; @0.99 + mean_cos mandatory secondaries


def residue_label(h: int, K: int) -> str:
    """S3.2 MA2 label vocabulary: {identity, train-residue, novel}."""
    r = h % K
    if r == 0:
        return "identity"
    if r in {x % K for x in H_TRAIN}:
        return "train-residue"
    return "novel"


@dataclass(frozen=True)
class EvalPoint:
    h: int
    component: str        # train_support | legacy | ladder | h_star | residue_sweep | cost_probe
    mode: str             # claim | residue_sweep
    residue: int
    residue_label: str
    claim_eligible: bool  # feeds generalization claims/aggregates
    in_window: bool       # cost-probe behavioral values recorded but out-of-window
    timed: bool           # wall-clock measured (Axis B)


def claim_config(K: int, d: int = D_PIN) -> "te.TaskEConfig":
    """The claim-mode TaskEConfig: EVERY claim point lives in a declared hop
    set, so the inherited periodicity assert fires verbatim at construction.
    5 (K=8 ladder start) already sits in H_test; h*(K=12)=57 is added to
    H_extra explicitly (it is NOT on the K=12 ladder -- S3.2: h* computed on
    the claim path, never via the sweep).

    `d` (S9.7 write-capacity diagnostic): optional ambient dimension,
    defaults to D_PIN=16 so every existing K=8/K=12 call site is
    byte-identical. Condition A/B K=16 cells pass d=32 explicitly (the
    proportional-headroom convention, S9.2); te.TaskEConfig's own
    __post_init__ hard-asserts K<=d, unchanged."""
    g = GRIDS[K]
    extra = [H_LEGACY_EXTRA]
    for h in sorted(set(g["ladder"]) | {g["h_star"]} | set(g["cost_probe"])):
        if h not in H_TRAIN and h not in H_TEST_LEGACY and h not in extra:
            extra.append(h)
    return te.TaskEConfig(d=d, K=K, variant="permutation",
                          H_train=H_TRAIN, H_test=H_TEST_LEGACY,
                          H_extra=tuple(extra), orthogonal=True)


def assert_claim_point(h: int, K: int) -> None:
    """Mirror of the inherited task_e.py:121-132 guard, applied per point.
    train-support depths (the literal H_TRAIN) are exempt: they are
    in-distribution by definition and never claim-eligible."""
    if h in H_TRAIN:
        return
    r = h % K
    assert r != 0, (
        f"claim-mode point h={h} is a multiple of K={K}: pi^{h} is the "
        f"IDENTITY -- confounded, not held-out (task_e.py periodicity guard)")
    assert r not in {x % K for x in H_TRAIN}, (
        f"claim-mode point h={h} has h%K={r} == a TRAINING residue -- "
        f"secretly in-distribution (task_e.py periodicity guard)")


def eval_points(K: int, d: int = D_PIN) -> list:
    """The full pinned eval grid for one K, every point labeled. Claim-mode
    points are additionally validated against the verbatim-inherited config
    assert (claim_config construction) AND the per-point mirror.

    `d` (S9.7): threaded to claim_config; defaults to D_PIN, byte-identical
    for every existing K=8/K=12 call site."""
    g = GRIDS[K]
    claim_config(K, d=d)  # inherited assert runs verbatim over all claim points
    pts = []

    def add(h, component, mode, claim_eligible, in_window, timed=False):
        if mode == "claim":
            assert_claim_point(h, K)
        pts.append(EvalPoint(
            h=h, component=component, mode=mode, residue=h % K,
            residue_label=residue_label(h, K), claim_eligible=claim_eligible,
            in_window=in_window, timed=timed))

    for h in H_TRAIN:
        add(h, "train_support", "claim", False, True)
    for h in (*H_TEST_LEGACY, H_LEGACY_EXTRA):
        add(h, "legacy", "claim", False, True)
    for h in g["ladder"]:
        add(h, "ladder", "claim", True, True)
    if g["h_star"] not in g["ladder"]:
        add(g["h_star"], "h_star", "claim", True, True)
    for h in g["sweep"]:
        # sweep points never assert_claim_point; label-and-exclude instead
        add(h, "residue_sweep", "residue_sweep",
            claim_eligible=False, in_window=True)
    for h in g["cost_probe"]:
        add(h, "cost_probe", "claim", False, False, timed=True)

    # exhaustive-labeling teeth: sweep must cover every residue exactly once
    sweep_res = sorted(p.residue for p in pts if p.component == "residue_sweep")
    assert sweep_res == list(range(K)), sweep_res
    # every ladder point must be a novel residue (claim-eligibility premise)
    for p in pts:
        if p.claim_eligible:
            assert p.residue_label == "novel", p
    return pts


def require_residue_label(entry: dict) -> None:
    """Results-schema guard (S3.2 MA2): a residue_sweep result entry MUST
    carry a valid residue_label. Refuse, never default."""
    if entry.get("mode") == "residue_sweep":
        lab = entry.get("residue_label")
        assert lab in ("identity", "train-residue", "novel"), (
            f"residue_sweep entry h={entry.get('h')} missing/invalid "
            f"residue_label={lab!r} -- label-and-exclude is mandatory (S3.2 MA2)")


def sample_eval_batch(cfg: "te.TaskEConfig", batch_size: int,
                      gen: torch.Generator, h: int, device="cpu") -> dict:
    """Eval batch at raw depth h with EXACT labels computed at the reduced
    depth h % K (single-cycle identity pi^h == pi^(h mod K); label path only).
    The returned batch's `hops` tensor carries the RAW h -- the only depth
    signal any model ever sees (S3.1)."""
    h_eff = h % cfg.K if cfg.variant == "permutation" else h
    b = te.sample_batch(cfg, batch_size, gen, hop_set=(h_eff,), device=device)
    b["hops"] = torch.full_like(b["hops"], h)
    b["h_raw"] = h
    b["h_eff"] = h_eff
    return b


def _self_test():
    # S9.7: K=14/15 exercised at the default d=16 (spare-probe convention);
    # K=16/24 exercised at d=2K (K<=d would fail at the default D_PIN=16 --
    # this IS the Condition A/B proportional-headroom convention, S9.2/S11).
    d_for_K = {8: D_PIN, 12: D_PIN, 14: D_PIN, 15: D_PIN, 16: 32, 24: 48}
    for K, d in d_for_K.items():
        pts = eval_points(K, d=d)
        n_claim = sum(1 for p in pts if p.claim_eligible)
        assert n_claim == len(GRIDS[K]["ladder"]) + (0 if GRIDS[K]["h_star"] in GRIDS[K]["ladder"] else 1)
        # labels sane
        for p in pts:
            assert p.residue == p.h % K
            if p.component == "residue_sweep":
                require_residue_label(dict(mode="residue_sweep", h=p.h,
                                           residue_label=p.residue_label))
    # sweep label spot checks (K=8): 57->train-residue, 60->novel, 64->identity
    assert residue_label(57, 8) == "train-residue"
    assert residue_label(60, 8) == "novel"
    assert residue_label(64, 8) == "identity"
    assert residue_label(60, 12) == "identity"
    assert residue_label(49, 12) == "train-residue"
    # S9.7/S11 new-grid spot checks: identity point (h_star+3) at every new K,
    # and the ladder_residue point itself is 'novel' (not a train residue)
    for K in (14, 15, 16, 24):
        g = GRIDS[K]
        assert residue_label(g["h_star"] + 3, K) == "identity", K
        assert residue_label(g["ladder_residue"], K) == "novel", K
    print("ncr_task self-test PASSED "
          f"({len(eval_points(8))} K=8 points, {len(eval_points(12))} K=12 points, "
          f"{len(eval_points(14))} K=14 points, {len(eval_points(15))} K=15 points, "
          f"{len(eval_points(16, d=32))} K=16(d=32) points, "
          f"{len(eval_points(24, d=48))} K=24(d=48) points)")


if __name__ == "__main__":
    _self_test()
