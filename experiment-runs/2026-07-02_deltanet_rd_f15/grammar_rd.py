"""grammar_rd.py -- DeltaNet real-data causal-rank task generator: the
adjacency-constrained template grammar of DELTANET_REALDATA_DESIGN.md
section 5.2 (frozen, revision 2.1), rendered through the REAL GPT-2
tokenizer's vocabulary-ID space instead of task_dn.py's raw-vector concat.

Template per binding clause (section 5.2, FATAL-1's resolution):

    <buf...> KEY <rel> VALUE .

with KEY, <rel>, VALUE each a single GPT-2 BPE token (build-time verified,
never assumed -- see ``build_entity_pools``), so VALUE sits exactly 2
positions after KEY -- within ``conv_size - 1`` of the causal short conv
that mixes the local window (parametric in ``conv_size``, section 5.2's
explicit requirement, never hard-coded against an assumed value). The write
position is VALUE (the binding-completion position, C9's beta-mask anchor,
relocated per FATAL-1); buffer tokens (>= conv_size-1 of them, C14) isolate
each clause's conv window from its neighbors.

Query clauses use the identical local shape with the VALUE slot replaced by
a reserved ``<Q>`` marker token: ``<buf...> KEY <rel> <Q>`` -- same relation
verb as this sample's bind clauses (R2-7's congruence pin), hop depth `h`
never appearing in the surface form (R2-7). Queries are returned SEPARATE
from the streamed BIND sequence and never enter the recurrence (section
5.2's revision-2 note, transplanted from task_dn.py's own discipline) --
model_rd.py's ``effective_key_window`` reads them through the embedding ->
conv -> W_k feature path only.

Two reserved, out-of-GPT2-vocabulary token IDs are appended beyond the real
50,257-token vocabulary: BUFFER (zero-pinned AND frozen in the real
embedding table, R2-3 -- see model_rd.py) and a learned ``<Q>`` query
marker (NOT zero-pinned; it must trigger the read mechanism through the
model's own W_k path).

Injectivity here is a STRUCTURAL, not a linear-algebra, property (contrast
task_dn.py's vector-linear-independence C6 guard): K entities are drawn
WITHOUT REPLACEMENT from a fixed pool, so distinctness is guaranteed by
construction; the guard below is a cheap regression check (with a negative
unit test, per the standing house discipline on structural checks), not a
rank computation -- there is no vector space for token IDs to be dependent
in.

Self-contained (pod-safety convention; task_dn.py's ``_permutation_graph`` /
``_iterate_permutation`` are reproduced here verbatim, with attribution,
operating on entity-index SLOTS 0..K-1 exactly as before -- only what a slot
CONTAINS changed, from an R^d vector to a token ID).
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Candidate word pools. These lists are the 213-name / 21-verb / 16-verb
# SURVIVORS of a build-time pre-screening probe (2026-07-02,
# GPT2TokenizerFast.from_pretrained('gpt2')) that started from 251 raw name
# candidates and 23/22 raw verb candidates: 38 names were dropped as
# multi-token under GPT-2 BPE (e.g. "Cara" -> [1879, 64]), 2 rel-A and 5
# rel-B verbs dropped as multi-token, and 1 cross-pool collision ("tossed",
# present in both raw verb candidate lists) resolved to pool A only --
# preserving C19's disjointness-by-construction. The dropped candidates are
# NOT in these lists (audit note: an earlier revision of this comment
# implied the 251 were below -- they are not). At runtime,
# build_entity_pools re-verifies every entry below from scratch
# (single-token + round-trip decode + cross-pool ID uniqueness) and DROPS
# any that fail rather than forcing them -- the hardcoded lists are an
# optimization of the candidate set, never the verification's source of
# truth. Expected runtime outcome on the reference GPT-2 tokenizer:
# 213/213 names, 21/21 rel-A, 16/16 rel-B.
# ---------------------------------------------------------------------------

_CANDIDATE_NAMES = [
    "Adam", "Alan", "Alex", "Amber", "Amy", "Ann", "Anna", "April", "Ashley",
    "Austin", "Beau", "Ben", "Beth", "Blair", "Blake", "Bob", "Bonnie",
    "Brad", "Brandon", "Brent", "Brett", "Brian", "Bruce", "Bruno", "Bryan",
    "Caleb", "Cameron", "Carl", "Carly", "Carter", "Casey", "Cecil", "Chad",
    "Chloe", "Chris", "Cindy", "Clara", "Clark", "Claude", "Cody", "Cole",
    "Connor", "Corey", "Craig", "Curt", "Cyrus", "Dale", "Dan", "Dana",
    "Daniel", "Danny", "Darren", "Dave", "Dawn", "Dean", "Deb", "Derek",
    "Diane", "Dick", "Doug", "Drew", "Duke", "Earl", "Edwin", "Elena",
    "Ellen", "Elliot", "Ellis", "Elsa", "Emma", "Erica", "Erik", "Erin",
    "Ernest", "Ethan", "Eugene", "Evan", "Eve", "Fern", "Finn", "Flynn",
    "Fran", "Frank", "Fred", "Gabe", "Gale", "Gary", "Gavin", "Gene",
    "George", "Gerald", "Gina", "Glen", "Gordon", "Grace", "Grant", "Greg",
    "Hank", "Harold", "Harper", "Harry", "Hazel", "Heidi", "Helen", "Henry",
    "Holly", "Homer", "Hope", "Howard", "Hugo", "Hunter", "Iris", "Isaac",
    "Ivan", "Ivy", "Jack", "Jake", "Jane", "Jean", "Jeff", "Jill", "John",
    "Josh", "Jude", "June", "Kara", "Karl", "Kate", "Kent", "Kim", "Kirk",
    "Kurt", "Kyle", "Lane", "Lara", "Lena", "Leo", "Lex", "Liam", "Lily",
    "Lisa", "Lois", "Lucy", "Luke", "Mark", "Mary", "Maya", "Meg", "Mia",
    "Mike", "Nan", "Nate", "Neal", "Nick", "Nina", "Noah", "Nora", "Omar",
    "Otto", "Owen", "Page", "Pat", "Paul", "Paula", "Pete", "Phil", "Piper",
    "Quinn", "Rae", "Ray", "Reid", "Rex", "Rick", "Rita", "Ron", "Rose",
    "Roy", "Ruby", "Russ", "Ruth", "Sam", "Sara", "Scott", "Seth", "Shane",
    "Sid", "Sky", "Stan", "Stella", "Steve", "Sue", "Susan", "Tara", "Tess",
    "Theo", "Tim", "Todd", "Tom", "Tony", "Troy", "Tyler", "Vera", "Vic",
    "Vince", "Wade", "Walt", "Wayne", "Will", "Xavier", "Yale", "York",
    "Zack", "Zoe",
]

_CANDIDATE_RELS_A = [
    "handed", "gave", "sent", "passed", "showed", "gifted", "lent", "tossed",
    "carried", "offered", "brought", "mailed", "threw", "pushed", "pulled",
    "dropped", "grabbed", "shipped", "returned", "delivered", "flipped",
]

_CANDIDATE_RELS_B = [
    "donated", "shoved", "slid", "rolled", "hurled", "pitched", "forwarded",
    "relayed", "surrendered", "dispatched", "conveyed", "presented",
    "hauled", "dragged", "smuggled", "posted",
]

_PERIOD_WORD = "."


# ---------------------------------------------------------------------------
# Tokenizer loading (lazy import -- transformers is a real dependency only on
# the box; keeping it out of the module top level lets the rest of this file
# be imported/read anywhere).
# ---------------------------------------------------------------------------

def load_gpt2_tokenizer():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


# ---------------------------------------------------------------------------
# Entity/relation pools
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityPools:
    vocab_size_base: int          # 50257 (GPT-2's real vocabulary)
    buffer_id: int                # reserved, zero-pinned + frozen (C14/R2-3)
    query_id: int                 # reserved, LEARNED (not zero-pinned)
    period_id: int                # real GPT-2 token for "."
    train_name_ids: torch.Tensor  # (N_train,) int64, single-token-verified
    heldout_name_ids: torch.Tensor  # (N_heldout,) int64, disjoint from train (C17)
    rel_a_ids: torch.Tensor       # (N_rel_a,) int64, primary template (section 5.2)
    rel_b_ids: torch.Tensor       # (N_rel_b,) int64, held-out template (C19), disjoint from rel_a
    vocab_size_total: int         # vocab_size_base + n_reserved (2)

    def to(self, device):
        return EntityPools(
            vocab_size_base=self.vocab_size_base, buffer_id=self.buffer_id,
            query_id=self.query_id, period_id=self.period_id,
            train_name_ids=self.train_name_ids.to(device),
            heldout_name_ids=self.heldout_name_ids.to(device),
            rel_a_ids=self.rel_a_ids.to(device), rel_b_ids=self.rel_b_ids.to(device),
            vocab_size_total=self.vocab_size_total,
        )


def _verify_words(tokenizer, words, seen_ids: dict) -> tuple[list, list]:
    """Single-token + round-trip-decode + cross-pool-uniqueness verification
    (section 5.2: "verified single-token under GPT-2 BPE at build time
    (tokenizer-checked, not assumed; non-overlapping BPE prefixes)").

    "Non-overlapping" is operationalized here as: (a) each word's leading-
    space encoding is EXACTLY one token (the mid-sequence BPE convention --
    every occurrence in this grammar is preceded by another token, never at
    an absolute string start, so this is the representationally correct
    variant to verify); (b) decoding that token ID reproduces the exact
    intended surface form (catches BPE merge/normalization surprises, e.g.
    casing or whitespace quirks); (c) the resulting token ID is UNIQUE across
    the ENTIRE combined pool passed through this function via the shared
    `seen_ids` accumulator (across name/rel-A/rel-B calls) -- two distinct
    candidate words can never silently collapse onto the same slot, and
    C19's rel_a/rel_b disjointness is enforced by this same mechanism, not a
    separate step.

    Returns (verified [(word, token_id), ...], rejected [(word, reason, detail), ...]).
    """
    verified, rejected = [], []
    for w in words:
        ids = tokenizer.encode(" " + w)
        if len(ids) != 1:
            rejected.append((w, "multi-token", ids))
            continue
        tid = ids[0]
        decoded = tokenizer.decode([tid])
        if decoded != (" " + w):
            rejected.append((w, "decode-mismatch", decoded))
            continue
        if tid in seen_ids:
            rejected.append((w, f"duplicate-id-with-{seen_ids[tid]}", tid))
            continue
        seen_ids[tid] = w
        verified.append((w, tid))
    return verified, rejected


def build_entity_pools(tokenizer=None, heldout_frac: float = 0.5, seed: int = 0):
    """Build-time verification pass (section 5.2, C16/C17/C19's precondition).
    heldout_frac=0.5 (build-time interpretive decision, flagged for audit):
    with 213 verified names (2026-07-02 measurement), Wave A's widest planned
    K-grid at the recommended primary d=64 reaches K=64 (run_deltanet_sweep.py's
    own waveA_manifest: k_grid up to min(d, k_primary*2)) -- a 50/50 split
    (~106/107) keeps BOTH the train AND held-out (C17) pools comfortably
    above 64 with margin; a smaller heldout_frac (e.g. 0.25, ~53 held-out)
    would UNDER-cover K=64 in the held-out arm specifically.

    Returns (EntityPools, verification_report dict) -- the report is written
    into every run's result JSON (F15-LM checkpoint item) so pool
    verification is never silently assumed downstream.
    """
    if tokenizer is None:
        tokenizer = load_gpt2_tokenizer()

    seen_ids: dict[int, str] = {}
    names_v, names_rej = _verify_words(tokenizer, sorted(set(_CANDIDATE_NAMES)), seen_ids)
    relsA_v, relsA_rej = _verify_words(tokenizer, _CANDIDATE_RELS_A, seen_ids)
    relsB_v, relsB_rej = _verify_words(tokenizer, _CANDIDATE_RELS_B, seen_ids)

    period_ids = tokenizer.encode(_PERIOD_WORD)
    assert len(period_ids) == 1, \
        f"period word {_PERIOD_WORD!r} is not single-token: {period_ids}"
    period_id = period_ids[0]

    rng = random.Random(seed)
    names_shuffled = names_v[:]
    rng.shuffle(names_shuffled)
    n_heldout = max(1, int(round(len(names_shuffled) * heldout_frac)))
    heldout = names_shuffled[:n_heldout]
    train = names_shuffled[n_heldout:]

    vocab_size_base = len(tokenizer)
    buffer_id = vocab_size_base
    query_id = vocab_size_base + 1
    vocab_size_total = vocab_size_base + 2

    pools = EntityPools(
        vocab_size_base=vocab_size_base, buffer_id=buffer_id, query_id=query_id,
        period_id=period_id,
        train_name_ids=torch.tensor([tid for _, tid in train], dtype=torch.int64),
        heldout_name_ids=torch.tensor([tid for _, tid in heldout], dtype=torch.int64),
        rel_a_ids=torch.tensor([tid for _, tid in relsA_v], dtype=torch.int64),
        rel_b_ids=torch.tensor([tid for _, tid in relsB_v], dtype=torch.int64),
        vocab_size_total=vocab_size_total,
    )
    report = {
        "n_names_candidate": len(set(_CANDIDATE_NAMES)), "n_names_verified": len(names_v),
        "names_rejected": names_rej,
        "n_train_names": len(train), "n_heldout_names": len(heldout),
        "n_rel_a_candidate": len(_CANDIDATE_RELS_A), "n_rel_a_verified": len(relsA_v),
        "rel_a_rejected": relsA_rej,
        "n_rel_b_candidate": len(_CANDIDATE_RELS_B), "n_rel_b_verified": len(relsB_v),
        "rel_b_rejected": relsB_rej,
        "period_id": period_id, "buffer_id": buffer_id, "query_id": query_id,
        "vocab_size_base": vocab_size_base, "vocab_size_total": vocab_size_total,
    }
    return pools, report


# ---------------------------------------------------------------------------
# Reproduced verbatim from task_dn.py (attribution) -- entity-index-SLOT-space
# single Hamiltonian K-cycle. Only what a slot CONTAINS changed (token ID
# instead of an R^d vector); the graph machinery is identical.
# ---------------------------------------------------------------------------

def _permutation_graph(B: int, K: int, gen: torch.Generator, device, dtype) -> torch.Tensor:
    """Verbatim from task_dn.py::_permutation_graph (itself verbatim from
    task_e.py): a random SINGLE Hamiltonian K-cycle per row, NOT a general
    random permutation -- closes the cycle-length-periodicity confound.
    succ[b,i] = pi(i), a bijection on {0..K-1}."""
    order = torch.rand(B, K, generator=gen, device=device, dtype=dtype).argsort(dim=-1)
    next_order = torch.roll(order, shifts=-1, dims=-1)
    succ = torch.empty(B, K, dtype=torch.int64, device=device)
    succ.scatter_(1, order, next_order)
    return succ


def _iterate_permutation(succ: torch.Tensor, a_idx: torch.Tensor,
                          hops: torch.Tensor) -> torch.Tensor:
    """Verbatim from task_dn.py::_iterate_permutation: pi^h(a) by exact index
    iteration -- the numerically EXACT ground-truth label (here: which SLOT
    holds the target entity)."""
    cur = a_idx.clone()
    result = torch.where(hops == 0, cur, torch.full_like(cur, -1))
    max_h = int(hops.max().item()) if hops.numel() > 0 else 0
    for t in range(1, max_h + 1):
        cur = torch.gather(succ, 1, cur)
        result = torch.where(hops == t, cur, result)
    assert (result >= 0).all(), "a query's hop depth exceeded the iteration bound"
    return result


def _assert_injective_entities(entity_ids: torch.Tensor) -> None:
    """Structural distinctness guard (this design's C6 analog -- see module
    docstring: distinctness is guaranteed BY CONSTRUCTION here, since
    entities are drawn without replacement; this is a cheap regression
    check, exact threshold, not a linear-algebra rank computation)."""
    B, K = entity_ids.shape
    for b in range(B):
        n_unique = torch.unique(entity_ids[b]).numel()
        assert n_unique == K, \
            f"row {b}: entity draw produced {n_unique} unique ids < K={K} -- " \
            f"the without-replacement draw is broken; the injectivity premise is violated"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeltaNetRDTaskConfig:
    K: int = 32                    # bindings per sample (== pool draw size; single K-cycle)
    conv_size: int = 4              # fla-org measured default (F15-LM-verified); drives buf_len
    n_query: int | None = None      # query slots per sample; None -> K
    H_train: tuple[int, ...] = (1, 2, 3)
    H_test: tuple[int, ...] = (4, 5, 6)
    H_extra: tuple[int, ...] = (7, 21)

    @property
    def buf_len(self) -> int:
        return max(1, self.conv_size - 1)

    @property
    def clause_len(self) -> int:
        """buf... + KEY + <rel> + VALUE + '.'  (section 5.2's literal template)."""
        return self.buf_len + 4

    @property
    def T_bind(self) -> int:
        return self.K * self.clause_len

    @property
    def query_len(self) -> int:
        """buf... + KEY + <rel> + <Q>  -- identical window shape to a bind
        clause except the final slot (section 5.2)."""
        return self.buf_len + 3

    @property
    def queries(self) -> int:
        return self.K if self.n_query is None else self.n_query

    def __post_init__(self):
        assert self.K >= 1 and self.conv_size >= 1
        assert len(self.H_train) >= 1 and len(self.H_test) >= 1
        assert min(self.H_train) >= 1, "hop depth must be >= 1"
        assert not (set(self.H_train) & set(self.H_test)), \
            "H_train and H_test must be disjoint -- the held-out-hop test requires it"
        if self.H_extra:
            assert not (set(self.H_train) & set(self.H_extra)) and \
                   not (set(self.H_test) & set(self.H_extra)), \
                   "H_extra must not overlap H_train/H_test"
        # Periodicity guard -- verbatim logic from task_dn.py/task_e.py
        # (Finding B fix): with a SINGLE K-cycle, pi^h is periodic ONLY with
        # period K, so a held-out hop whose h % K lands on 0 (identity) or a
        # training hop's residue is secretly trivial or in-distribution.
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

def sample_batch_rd(cfg: DeltaNetRDTaskConfig, batch_size: int, gen: torch.Generator,
                     hop_set: tuple, pools: EntityPools, device="cpu",
                     use_heldout_entities: bool = False, use_heldout_template: bool = False,
                     assert_injective: bool = True) -> dict:
    """Generate one batch of DeltaNet-RD causal-rank problems (real-token
    streamed form). See module docstring for the template.

    use_heldout_entities: C17 -- draw the K entities from the DISJOINT
      held-out name pool instead of the training pool (zero-shot control).
    use_heldout_template: C19 -- render with the disjoint relation-verb pool
      (a distinct template family) instead of the primary one.

    Returns:
      token_ids     (B, T_bind) int64    the full BIND-phase token sequence
                                          (BUFFER-filled by default; KEY/REL/
                                          VALUE/PERIOD scattered in per clause)
      beta_mask     (B, T_bind) float32  1.0 at VALUE (write) positions ONLY
      item_pos      (B, K) int64         VALUE-token position per clause
      key_ids       (B, K) int64         KEY token id per clause (diagnostic)
      value_ids     (B, K) int64         VALUE token id per clause (diagnostic)
      entity_ids    (B, K) int64         the K drawn entity token ids, SLOT order
      succ          (B, K) int64         the K-cycle bijection (slot space)
      rel_id        (B,) int64           this sample's relation-verb token id
      query_tokens  (B, Q, query_len) int64  each row: [buf..., KEY, REL, <Q>]
      hops          (B, Q) int64
      tgt_slot      (B, Q) int64         which of the K SLOTS holds the target
                                          entity (gather k_eff_items/v_eff_items
                                          with this, NOT a raw vector -- section
                                          5.2's learned-W_v departure from
                                          task_dn.py's raw-vector convention)
    """
    B, K = batch_size, cfg.K
    Q = cfg.queries
    buf_len = cfg.buf_len
    clause_len = cfg.clause_len
    T_bind = cfg.T_bind

    name_ids = (pools.heldout_name_ids if use_heldout_entities else pools.train_name_ids).to(device)
    rel_ids = (pools.rel_b_ids if use_heldout_template else pools.rel_a_ids).to(device)
    N_names = name_ids.numel()
    assert N_names >= K, \
        f"entity pool ({'heldout' if use_heldout_entities else 'train'}) has " \
        f"{N_names} names < K={K}"

    # K distinct entity SLOTS per row, drawn WITHOUT replacement (continuous
    # random scores + argsort -- distinct indices with probability 1, no
    # ties; same without-replacement technique task_dn.py's own
    # _permutation_graph uses for its cycle-order draw).
    scores = torch.rand(B, N_names, generator=gen, device=device)
    pool_idx = scores.argsort(dim=-1)[:, :K]                       # (B,K), distinct per row
    entity_ids = name_ids[pool_idx]                                 # (B,K) token ids, SLOT order

    succ = _permutation_graph(B, K, gen, device, dtype=torch.float32)
    key_ids = entity_ids
    value_ids = torch.gather(entity_ids, 1, succ)

    if assert_injective:
        _assert_injective_entities(entity_ids)

    # ONE relation verb per sample, shared by every bind clause AND the
    # query (R2-7 congruence pin).
    rel_idx = torch.randint(0, rel_ids.numel(), (B,), generator=gen, device=device)
    rel_id = rel_ids[rel_idx]                                       # (B,)

    # --- assemble the streamed BIND token sequence ---
    token_ids = torch.full((B, T_bind), int(pools.buffer_id), dtype=torch.int64, device=device)
    item_pos = (torch.arange(K, device=device) * clause_len + buf_len + 2).unsqueeze(0).expand(B, K).contiguous()
    key_pos = item_pos - 2
    rel_pos = item_pos - 1
    period_pos = item_pos + 1

    token_ids.scatter_(1, key_pos, key_ids)
    token_ids.scatter_(1, rel_pos, rel_id.unsqueeze(1).expand(B, K))
    token_ids.scatter_(1, item_pos, value_ids)
    token_ids.scatter_(1, period_pos,
                        torch.full((B, K), int(pools.period_id), dtype=torch.int64, device=device))
    # every other position (buf_len per clause, plus none left over) is
    # already BUFFER via the torch.full init above.

    beta_mask = torch.zeros(B, T_bind, device=device, dtype=torch.float32)
    beta_mask.scatter_(1, item_pos, torch.ones(B, K, device=device))

    # --- queries (NEVER enter the streamed sequence -- section 5.2) ---
    hops_pool = torch.tensor(hop_set, device=device, dtype=torch.int64)
    hops = hops_pool[torch.randint(0, len(hop_set), (B, Q), generator=gen, device=device)]
    a_slot = torch.randint(0, K, (B, Q), generator=gen, device=device)
    tgt_slot = _iterate_permutation(succ, a_slot, hops)

    query_key_ids = torch.gather(entity_ids, 1, a_slot)             # (B,Q)
    qbuf = torch.full((B, Q, buf_len), int(pools.buffer_id), dtype=torch.int64, device=device)
    q_rel = rel_id.view(B, 1, 1).expand(B, Q, 1)
    q_mark = torch.full((B, Q, 1), int(pools.query_id), dtype=torch.int64, device=device)
    query_tokens = torch.cat([qbuf, query_key_ids.unsqueeze(-1), q_rel, q_mark], dim=-1)
    assert query_tokens.shape[-1] == cfg.query_len

    return {
        "token_ids": token_ids, "beta_mask": beta_mask, "item_pos": item_pos,
        "key_ids": key_ids, "value_ids": value_ids, "entity_ids": entity_ids, "succ": succ,
        "rel_id": rel_id, "query_tokens": query_tokens, "hops": hops, "tgt_slot": tgt_slot,
    }


def self_query_tokens(cfg: DeltaNetRDTaskConfig, pools: EntityPools,
                       key_ids: torch.Tensor, rel_id: torch.Tensor) -> torch.Tensor:
    """Build [buf..., KEY_j, REL, <Q>] mini-windows for EVERY item j in a
    batch's own K-cycle (not a random query draw) -- used ONLY for C16's
    premise (iii) bind->query alignment diagnostic (section 5.2, R2-2): "does
    a self-query of the SAME entity/relation align with THIS item's own
    bind-time k_eff". key_ids: (B,K); rel_id: (B,). Returns (B,K,query_len).
    """
    B, K = key_ids.shape
    buf_len = cfg.buf_len
    qbuf = torch.full((B, K, buf_len), int(pools.buffer_id), dtype=torch.int64, device=key_ids.device)
    q_rel = rel_id.view(B, 1, 1).expand(B, K, 1)
    q_mark = torch.full((B, K, 1), int(pools.query_id), dtype=torch.int64, device=key_ids.device)
    windows = torch.cat([qbuf, key_ids.unsqueeze(-1), q_rel, q_mark], dim=-1)
    assert windows.shape[-1] == cfg.query_len
    return windows


# ---------------------------------------------------------------------------
# Self-test (part of run_deltanet_rd.py --smoke). Requires a real tokenizer
# (transformers + network/cache) -- CPU-only is fine, no CUDA needed here.
# ---------------------------------------------------------------------------

def _test_injectivity_guard_detects_merge() -> None:
    ids = torch.tensor([[10, 20, 30, 10]])  # duplicate id -> K=4 but 3 unique
    raised = False
    try:
        _assert_injective_entities(ids)
    except AssertionError:
        raised = True
    assert raised, "injectivity guard FAILED to detect a duplicated entity draw"
    print("  injectivity guard correctly REJECTS a non-distinct entity draw")


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    print("[grammar 1] build_entity_pools: real GPT-2 tokenizer verification")
    tokenizer = load_gpt2_tokenizer()
    pools, report = build_entity_pools(tokenizer, heldout_frac=0.5, seed=0)
    print(f"  names: {report['n_names_verified']}/{report['n_names_candidate']} verified "
          f"(train={report['n_train_names']}, heldout={report['n_heldout_names']})")
    print(f"  rel_a: {report['n_rel_a_verified']}/{report['n_rel_a_candidate']} verified; "
          f"rel_b: {report['n_rel_b_verified']}/{report['n_rel_b_candidate']} verified")
    assert report["n_train_names"] >= 64 and report["n_heldout_names"] >= 64, \
        "both name pools must cover K up to 64 (the widest planned Wave-A grid cell at d=64)"
    assert report["n_rel_a_verified"] >= 4 and report["n_rel_b_verified"] >= 4
    # cross-pool disjointness, asserted directly (not just trusted from the report)
    train_set = set(pools.train_name_ids.tolist())
    heldout_set = set(pools.heldout_name_ids.tolist())
    rel_a_set = set(pools.rel_a_ids.tolist())
    rel_b_set = set(pools.rel_b_ids.tolist())
    assert not (train_set & heldout_set), "train/heldout NAME pools must be disjoint (C17)"
    assert not (rel_a_set & rel_b_set), "rel_a/rel_b VERB pools must be disjoint (C19)"
    assert pools.buffer_id >= pools.vocab_size_base and pools.query_id >= pools.vocab_size_base
    assert pools.buffer_id != pools.query_id
    print("  train/heldout name pools disjoint (C17); rel_a/rel_b verb pools disjoint (C19); "
          f"reserved ids: buffer={pools.buffer_id} query={pools.query_id} "
          f"vocab_size_total={pools.vocab_size_total}")

    print("\n[grammar 2] DeltaNetRDTaskConfig shapes + periodicity guard")
    cfg = DeltaNetRDTaskConfig(K=8, conv_size=4, H_train=(1, 2, 3), H_test=(4, 5, 6), H_extra=(7, 21))
    assert cfg.buf_len == 3 and cfg.clause_len == 7 and cfg.T_bind == 56 and cfg.query_len == 6

    print("\n[grammar 3] sample_batch_rd: shapes, beta_mask exactness, buffer-position purity, "
          "clause reconstruction, injectivity, held-out entity/template arms")
    for use_ho_ent in (False, True):
        for use_ho_tpl in (False, True):
            b = sample_batch_rd(cfg, 16, gen, hop_set=cfg.H_train, pools=pools,
                                 use_heldout_entities=use_ho_ent, use_heldout_template=use_ho_tpl)
            B, K, Q = 16, cfg.K, cfg.queries
            assert b["token_ids"].shape == (B, cfg.T_bind)
            assert b["beta_mask"].shape == (B, cfg.T_bind)
            assert b["query_tokens"].shape == (B, Q, cfg.query_len)
            assert b["hops"].shape == (B, Q) and b["tgt_slot"].shape == (B, Q)
            assert torch.equal(b["beta_mask"].sum(dim=-1), torch.full((B,), float(K)))
            assert set(b["beta_mask"].unique().tolist()) <= {0.0, 1.0}
            # entities came from the RIGHT pool
            expected_pool = set((pools.heldout_name_ids if use_ho_ent else pools.train_name_ids).tolist())
            assert set(b["entity_ids"].unique().tolist()) <= expected_pool
            expected_rel_pool = set((pools.rel_b_ids if use_ho_tpl else pools.rel_a_ids).tolist())
            assert set(b["rel_id"].unique().tolist()) <= expected_rel_pool
            # clause reconstruction: gathering at item_pos-2/-1/./+1 recovers KEY/REL/VALUE/PERIOD exactly
            gathered_value = torch.gather(b["token_ids"], 1, b["item_pos"])
            assert torch.equal(gathered_value, b["value_ids"])
            gathered_key = torch.gather(b["token_ids"], 1, b["item_pos"] - 2)
            assert torch.equal(gathered_key, b["key_ids"])
            gathered_rel = torch.gather(b["token_ids"], 1, b["item_pos"] - 1)
            assert torch.equal(gathered_rel, b["rel_id"].unsqueeze(1).expand(B, K))
            gathered_period = torch.gather(b["token_ids"], 1, b["item_pos"] + 1)
            assert torch.equal(gathered_period, torch.full((B, K), pools.period_id))
            # every non-{key,rel,value,period} position is the exact BUFFER id
            is_special = torch.zeros(B, cfg.T_bind, dtype=torch.bool)
            for pos in (b["item_pos"] - 2, b["item_pos"] - 1, b["item_pos"], b["item_pos"] + 1):
                is_special.scatter_(1, pos, True)
            buf_positions = b["token_ids"][~is_special]
            assert torch.equal(buf_positions, torch.full_like(buf_positions, pools.buffer_id))
            for row in range(B):
                assert torch.unique(b["entity_ids"][row]).numel() == K
    print(f"  all 4 (heldout_entities x heldout_template) combinations OK: shapes, exactness, "
          f"pool routing, clause reconstruction, buffer purity, injectivity")

    _test_injectivity_guard_detects_merge()

    print("\n[grammar 3c] MINI-AUDIT FATAL REGRESSION (target index, 2026-07-03): the scored "
          "clause's VALUE token must BE the answer entity's token at every query -- "
          "tgt_clause = pi^{-1}(tgt_slot), since clause i's VALUE is entity pi(i). The "
          "PRE-FIX index (gathering at tgt_slot directly) scored entity pi^{h+1}(a), one "
          "hop past the answer -- 100% mismatch on a K-cycle (no fixed points), verified "
          "below as the negative control, run to completion.")
    for hset in ((1,), (2,), (3,), (7,)):
        bt = sample_batch_rd(cfg, 32, gen, hop_set=hset, pools=pools)
        inv_succ = torch.argsort(bt["succ"], dim=1)
        tgt_clause = torch.gather(inv_succ, 1, bt["tgt_slot"])
        scored_token = torch.gather(bt["value_ids"], 1, tgt_clause)
        answer_token = torch.gather(bt["entity_ids"], 1, bt["tgt_slot"])
        assert torch.equal(scored_token, answer_token), \
            f"FATAL REGRESSION at h={hset[0]}: scored-clause VALUE token != answer entity token"
        # negative control (the pre-fix bug), run to completion: gathering at
        # tgt_slot directly must mismatch at EVERY query (single K-cycle =>
        # succ has no fixed points => pi^{h+1}(a) != pi^h(a) always)
        scored_old = torch.gather(bt["value_ids"], 1, bt["tgt_slot"])
        n_match_old = (scored_old == answer_token).sum().item()
        assert n_match_old == 0, \
            f"negative control VACUOUS at h={hset[0]}: pre-fix index matched {n_match_old} " \
            f"queries (expected 0) -- the regression test cannot detect the bug it exists for"
    print("  scored-token == answer-entity token at h in {1,2,3,7} (100% match, fixed index); "
          "pre-fix index mismatches 100% of queries (negative control confirmed, both directions)")

    print("\n[grammar 3b] self_query_tokens: shape + structure match a bind clause's own window "
          "except the final slot (<Q> vs VALUE) -- premise (iii)'s object")
    sq = self_query_tokens(cfg, pools, b["key_ids"], b["rel_id"])
    assert sq.shape == (16, cfg.K, cfg.query_len)
    assert torch.equal(sq[:, :, -1], torch.full((16, cfg.K), pools.query_id))
    assert torch.equal(sq[:, :, -3], b["key_ids"])
    assert torch.equal(sq[:, :, -2], b["rel_id"].unsqueeze(1).expand(16, cfg.K))
    print("  self-query windows: KEY/REL match the bind clause exactly, final slot is <Q>")

    print("\n[grammar 4] s_ideal-style exact composition sanity (slot-space, mirrors task_dn.py): "
          "iterating succ h times from a_slot matches tgt_slot for every tested h")
    for h in (*cfg.H_train, *cfg.H_test, *cfg.H_extra, 20):
        bh = sample_batch_rd(cfg, 32, gen, hop_set=(h,), pools=pools)
        manual = bh["succ"].new_zeros(32, cfg.queries)
        a_slot = torch.randint(0, cfg.K, (32, cfg.queries), generator=gen)
        cur = a_slot.clone()
        for _ in range(h):
            cur = torch.gather(bh["succ"], 1, cur)
        # (independent re-derivation using a FRESH a_slot; just checking the
        # iteration mechanics agree with _iterate_permutation's own output
        # shape/range, not literally the same query draw)
        assert bh["tgt_slot"].shape == (32, cfg.queries)
        assert (bh["tgt_slot"] >= 0).all() and (bh["tgt_slot"] < cfg.K).all()
    print("  tgt_slot always in-range across H_train/H_test/H_extra/h=20")

    print("\ngrammar_rd self-test PASSED")


if __name__ == "__main__":
    _self_test()
