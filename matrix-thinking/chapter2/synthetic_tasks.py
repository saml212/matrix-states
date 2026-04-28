"""Synthetic tasks for Chapter 2 matrix-native-from-scratch.

Each task produces (input_seq, target_token) pairs where the OPTIMAL latent
representation at the query position has a known rank K*. Chapter 2's
hypothesis: a matrix-native transformer trained on this should develop Z
with effective rank ≈ K* and should fail under rank-k truncation for k < K*.

No torch dependency. Pure python + numpy for portability and test speed.
"""
from __future__ import annotations

import numpy as np
import random
from dataclasses import dataclass
from typing import Iterator, Sequence


# ------------------------------------------------------------
# Task A: K parallel entity tracking
# ------------------------------------------------------------

@dataclass(frozen=True)
class TaskAConfig:
    """Config for the K-parallel-entity-tracking task.

    Vocab layout:
        0                       : PAD
        1 .. K                  : entity IDs (K entities)
        K+1 .. K+S              : state values (S distinct states)
        K+S+1                   : SET action (next two tokens = entity, state)
        K+S+2                   : QUERY action (next token = entity)
        K+S+3                   : SEP / query marker
        K+S+4                   : EOS

    Sequence grammar:
        (SET <entity> <state>)*  QUERY <entity> SEP
    Target: the current state of the queried entity (a token in K+1..K+S).

    Optimal latent at SEP position: a matrix whose columns i (for i in 1..K)
    encode the current state of entity i. With K independent entities, the
    lowest-rank working-memory representation has rank K.
    """

    K: int = 4                    # number of parallel entities
    S: int = 32                   # number of possible state values
    min_sets: int = 2             # at least this many SET actions per sequence
    max_sets: int = 16            # at most this many SET actions
    min_updates_per_entity: int = 1  # each entity gets at least this many SETs
    seed: int | None = None

    @property
    def vocab_size(self) -> int:
        # entities (K) + states (S) + {SET, QUERY, SEP, EOS, PAD}
        return 1 + self.K + self.S + 4

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def set_tok(self) -> int:
        return 1 + self.K + self.S

    @property
    def query_tok(self) -> int:
        return 1 + self.K + self.S + 1

    @property
    def sep_tok(self) -> int:
        return 1 + self.K + self.S + 2

    @property
    def eos_tok(self) -> int:
        return 1 + self.K + self.S + 3

    def entity_tok(self, i: int) -> int:
        assert 1 <= i <= self.K, f"entity {i} out of range [1, {self.K}]"
        return i

    def state_tok(self, s: int) -> int:
        assert 0 <= s < self.S, f"state {s} out of range [0, {self.S})"
        return 1 + self.K + s


def task_a_sample(cfg: TaskAConfig, rng: random.Random) -> tuple[list[int], int]:
    """Generate one (input_seq, target_state_tok) sample.

    The sequence has n_sets SET actions, then a QUERY + SEP. Target is the
    most recent state of the queried entity. Guaranteed: every entity gets
    at least cfg.min_updates_per_entity SET actions before the query.
    """
    # Draw number of SET actions; ensure every entity gets covered
    min_n_sets = max(cfg.min_sets, cfg.K * cfg.min_updates_per_entity)
    n_sets = rng.randint(min_n_sets, cfg.max_sets)
    # Draw which entity each SET applies to
    required_entities = list(range(1, cfg.K + 1)) * cfg.min_updates_per_entity
    extra = [rng.randint(1, cfg.K) for _ in range(n_sets - len(required_entities))]
    entities = required_entities + extra
    rng.shuffle(entities)
    # Draw states for each SET
    states = [rng.randint(0, cfg.S - 1) for _ in range(n_sets)]
    # Track current state of each entity
    current: dict[int, int] = {}
    seq: list[int] = []
    for e, s in zip(entities, states):
        seq.append(cfg.set_tok)
        seq.append(cfg.entity_tok(e))
        seq.append(cfg.state_tok(s))
        current[e] = s
    # Pick a queried entity (must have been SET at least once — satisfied by
    # min_updates_per_entity >= 1)
    q_entity = rng.randint(1, cfg.K)
    seq.append(cfg.query_tok)
    seq.append(cfg.entity_tok(q_entity))
    seq.append(cfg.sep_tok)
    target = cfg.state_tok(current[q_entity])
    return seq, target


def task_a_stream(cfg: TaskAConfig) -> Iterator[tuple[list[int], int]]:
    """Infinite stream of Task A samples. Thread-safe per caller."""
    rng = random.Random(cfg.seed)
    while True:
        yield task_a_sample(cfg, rng)


def task_a_batch(cfg: TaskAConfig, batch_size: int, rng: random.Random,
                 max_len: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched generator. Returns (inputs, targets, attention_mask).

    inputs: (B, L) int64 of token ids, padded with cfg.pad_id
    targets: (B,) int64 of target state tokens
    attention_mask: (B, L) bool of non-pad positions
    """
    samples = [task_a_sample(cfg, rng) for _ in range(batch_size)]
    seqs, targets = zip(*samples)
    L = max(len(s) for s in seqs)
    if max_len is not None:
        L = min(L, max_len)
    inputs = np.full((batch_size, L), cfg.pad_id, dtype=np.int64)
    mask = np.zeros((batch_size, L), dtype=bool)
    for i, s in enumerate(seqs):
        n = min(len(s), L)
        inputs[i, :n] = s[:n]
        mask[i, :n] = True
    return inputs, np.array(targets, dtype=np.int64), mask


# ------------------------------------------------------------
# Helper: pretty-print a sample for debugging
# ------------------------------------------------------------

def decode_task_a(seq: Sequence[int], cfg: TaskAConfig) -> str:
    """Convert token id sequence back to human-readable symbols."""
    out = []
    for t in seq:
        if t == cfg.pad_id:
            out.append("_")
        elif 1 <= t <= cfg.K:
            out.append(f"e{t}")
        elif cfg.K + 1 <= t <= cfg.K + cfg.S:
            out.append(f"s{t - cfg.K - 1}")
        elif t == cfg.set_tok:
            out.append("SET")
        elif t == cfg.query_tok:
            out.append("QUERY")
        elif t == cfg.sep_tok:
            out.append("SEP")
        elif t == cfg.eos_tok:
            out.append("EOS")
        else:
            out.append(f"?{t}")
    return " ".join(out)


# ------------------------------------------------------------
# Self-test
# ------------------------------------------------------------

def _self_test() -> None:
    cfg = TaskAConfig(K=4, S=8, min_sets=4, max_sets=10, min_updates_per_entity=1, seed=0)
    print(f"Task A — K={cfg.K}, S={cfg.S}, vocab={cfg.vocab_size}")
    rng = random.Random(42)
    for i in range(3):
        seq, target = task_a_sample(cfg, rng)
        print(f"\nSample {i}: len={len(seq)}, target={target} ({decode_task_a([target], cfg)})")
        print(f"  {decode_task_a(seq, cfg)}")

    # Verify correctness: simulate the grammar and check target equals last SET for queried entity
    rng = random.Random(123)
    n_ok = 0
    n_total = 200
    for _ in range(n_total):
        seq, target = task_a_sample(cfg, rng)
        # Walk seq, track state per entity, find query
        state_of = {}
        i = 0
        while i < len(seq):
            t = seq[i]
            if t == cfg.set_tok:
                e = seq[i + 1]
                s = seq[i + 2]
                state_of[e] = s
                i += 3
            elif t == cfg.query_tok:
                q_entity = seq[i + 1]
                assert seq[i + 2] == cfg.sep_tok, f"expected SEP, got {seq[i+2]}"
                expected = state_of[q_entity]
                assert expected == target, f"target {target} != expected {expected}"
                n_ok += 1
                break
            else:
                raise AssertionError(f"unexpected token {t} at pos {i}")
    print(f"\nSelf-test: {n_ok}/{n_total} samples consistent")
    assert n_ok == n_total

    # Batch test
    inputs, targets, mask = task_a_batch(cfg, batch_size=8, rng=random.Random(7))
    print(f"\nBatch: inputs={inputs.shape}, targets={targets.shape}, mask={mask.shape}")
    print(f"  mean seq len: {mask.sum(axis=1).mean():.1f}")
    print(f"  target range: [{targets.min()}, {targets.max()}]")


if __name__ == "__main__":
    _self_test()
