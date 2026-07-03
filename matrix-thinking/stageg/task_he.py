"""task_he.py -- Stage G Wave C (H_e) task-swap generator: a discrete,
byte-vocab, next-token-prediction serialization of chapter2/task_e.py's
K-relation / hop-composition grammar. See STAGE_G_DESIGN.md section 4 item 9
("Task-swap control on a composition-heavy corpus") and section 6.4 item 3
(Wave C, gated).

WHY A NEW GENERATOR, NOT A DIRECT REUSE OF task_e.py/task_dn.py: those
generators produce CONTINUOUS orthonormal-vector bindings, scored by cosine
similarity against a bespoke bilinear-readout encoder (Task D/E's own
single-Z-bottleneck harness) -- exactly the architecture STAGE_G_DESIGN.md's
own section 9 attack #9 says H_e's Wave C must NOT reuse ("Task E's H_e
evidence ... is from a single-Z-bottleneck associative-memory encoder, not
an autoregressive LM ... which is exactly why H_e is gated to Wave C and
requires a dedicated data-generator build"). This module instead serializes
the SAME K-relation / single-Hamiltonian-K-cycle / hop-composition grammar
as literal ASCII bytes so it can be trained end-to-end by MatrixThinkerG /
VectorReferenceModel's OWN forward(token_ids) interface (models.py) -- a
standard byte-vocab next-token-prediction corpus, no encoder/bottleneck of
any kind, matching section 4 item 9's exact ask: "the SAME two LM
architectures (MatrixThinker, LoopFormer, matched params, no encoder-style
single-Z bottleneck) can be trained on it directly".

GRAMMAR (build-time interpretive decision, flagged for audit -- the design
text specifies WHAT to adapt, not the exact serialization):
  entities        0..K-1, K <= 26, one ASCII lowercase letter each
                  ('a' + entity_id) -- single-BYTE-per-entity keeps every
                  eval readout a single-position argmax (see below), no
                  autoregressive decoding loop needed.
  relation line   b'R' + chr(entity_i) + chr(succ(entity_i)) + b'\\n'   (4 bytes)
  query line      b'Q' + chr(start) + 2 zero-padded decimal hop digits
                  + chr(answer) + b'\\n'                                (6 bytes)
  document        K relation lines (fresh single-Hamiltonian-K-cycle
                  permutation, PRESENTATION ORDER SHUFFLED independently of
                  the graph -- closes the trivial "position i always
                  describes entity i" shortcut) followed by Q_per_doc query
                  lines. EVERY document is FRESH (new K-cycle, new query
                  set, every batch/step) -- memorizing one global graph in
                  the WEIGHTS cannot solve this task; the graph must be
                  READ FROM THE CURRENT CONTEXT and the hop composition
                  actually carried out. This is the in-context mechanism
                  section 4 item 9 asks for ("rather than reusing Task D/E's
                  own bottlenecked-encoder harness").

SCORING ("score = composition recovery, not BPB", per the task brief):
  standard next-byte cross-entropy trains on EVERY position (the model also
  has to learn the cheap, structural R/Q line format); eval additionally
  isolates, for each query, the SINGLE next-token-prediction position whose
  target is the answer byte, and scores argmax(logits[pred_pos]) against
  the true answer entity id -- chance = 1/K. This needs NO autoregressive
  decoding loop (the minimal-version build choice, flagged for audit):
  every entity is exactly one byte, so "generate the answer" and "predict
  the one next byte at the answer position, teacher-forced" coincide by
  construction.

Held-out-hop generalization is carried IDENTICALLY to task_dn.py/task_e.py:
train documents draw every query's hop from H_train only; a SEPARATE eval
call passes a single-element hop_set (e.g. (h,) for h in H_test/H_extra) to
build a per-hop accuracy curve. The periodicity guard (single K-cycle ->
pi^h periodic with period K only) is reused verbatim from task_e.py's
TaskEConfig.__post_init__.

**h=1 COPY LEAK (audit FIX-A -- known, by construction, and handled at the
reporting layer, NOT a generator bug):** every relation line literally
contains `R<src><succ(src)>` in the SAME context window, so an h=1 query
`Q<start>01?` is answerable by PURE COPY -- find the R-line whose 2nd byte
matches `<start>`, emit its 3rd byte -- with zero composition. h=1 accuracy
is therefore a SINGLE-HOP LOOKUP (induction-head-style copy) metric, never
evidence of composition. h>=2 has no such shortcut: the intermediate
entities' bytes appear only as OTHER lines' src/dst fields, and chaining
them is exactly the composition being tested. Consequences enforced
downstream: run_stageg_he_sweep.py's aggregate() headlines
`in_dist_comp_acc_excl_h1` (h>=2 only) and reports h=1 separately as
`single_hop_lookup_acc_h1`; the held-out fields already exclude h=1
(H_test/H_extra start at 4). Keeping h=1 IN H_train is deliberate -- it
teaches the R-line -> lookup primitive the deeper hops build on
(task_e.py's own convention: "h=1 queries ARE the plain KV-binding ...
recall task", task_dn.py module docstring).

Self-contained (the discrete K-cycle graph code is an index-only adaptation
of task_e.py::_permutation_graph/_iterate_permutation, reproduced here
rather than imported -- matches this codebase's established pod-safety
convention; see matrix-thinking/deltanet/rank_utils.py's docstring for the
same reasoning and its origin, the 2026-04-29 rank_aware_v1 import-
fragility incident).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

ENTITY_BASE = ord('a')
REL, QRY, NL = ord('R'), ord('Q'), ord('\n')
REL_LINE_LEN = 4     # 'R' + src + dst + '\n'
QRY_LINE_LEN = 6     # 'Q' + start + 2 hop digits + answer + '\n'


@dataclass(frozen=True)
class HeTaskConfig:
    K: int = 12                       # entities per document (<=26, one ASCII letter each)
    Q_per_doc: int = 8                # query lines per document
    H_train: tuple[int, ...] = (1, 2, 3)
    H_test: tuple[int, ...] = (4, 5)
    H_extra: tuple[int, ...] = (7,)

    @property
    def doc_len(self) -> int:
        return self.K * REL_LINE_LEN + self.Q_per_doc * QRY_LINE_LEN

    @property
    def chance(self) -> float:
        return 1.0 / self.K

    def __post_init__(self):
        assert 1 <= self.K <= 26, f"K={self.K} must be in [1,26] (one ASCII lowercase letter/entity)"
        assert self.Q_per_doc >= 1
        assert len(self.H_train) >= 1 and len(self.H_test) >= 1
        assert min(self.H_train) >= 1, "hop depth must be >= 1"
        assert not (set(self.H_train) & set(self.H_test)), \
            "H_train and H_test must be disjoint -- the held-out-hop test requires it"
        if self.H_extra:
            assert not (set(self.H_train) & set(self.H_extra)) and \
                   not (set(self.H_test) & set(self.H_extra)), \
                   "H_extra must not overlap H_train/H_test"
        all_hops = (*self.H_train, *self.H_test, *(self.H_extra or ()))
        assert max(all_hops) <= 99, "hop depth must fit 2 zero-padded decimal digits"
        # Periodicity guard -- verbatim logic from task_e.py's
        # TaskEConfig.__post_init__ (single K-cycle -> pi^h periodic with
        # period K only; a held-out hop whose h % K collides with 0 or a
        # training residue is secretly trivial or in-distribution, not
        # genuinely held-out).
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
# Discrete K-cycle graph (index-only adaptation of task_e.py's
# _permutation_graph / _iterate_permutation -- entities ARE their own
# indices here, no continuous vector pool needed).
# ---------------------------------------------------------------------------

def _permutation_graph(B: int, K: int, gen: torch.Generator, device) -> torch.Tensor:
    """A random SINGLE Hamiltonian K-cycle per row (task_e.py's construction,
    verbatim logic): succ[b, i] = pi(i), a bijection on {0..K-1}, guaranteed
    to be exactly ONE cycle of length K (never several disjoint shorter
    cycles -- the periodicity confound task_e.py's own docstring documents
    in detail)."""
    order = torch.rand(B, K, generator=gen, device=device).argsort(dim=-1)
    next_order = torch.roll(order, shifts=-1, dims=-1)
    succ = torch.empty(B, K, dtype=torch.int64, device=device)
    succ.scatter_(1, order, next_order)
    return succ


def _iterate_permutation(succ: torch.Tensor, a_idx: torch.Tensor,
                          hops: torch.Tensor) -> torch.Tensor:
    """pi^h(a) by exact index iteration (verbatim from task_e.py/task_dn.py)."""
    cur = a_idx.clone()
    result = torch.where(hops == 0, cur, torch.full_like(cur, -1))
    max_h = int(hops.max().item()) if hops.numel() > 0 else 0
    for t in range(1, max_h + 1):
        cur = torch.gather(succ, 1, cur)
        result = torch.where(hops == t, cur, result)
    assert (result >= 0).all(), "a query's hop depth exceeded the iteration bound"
    return result


def _assert_bijection(succ: torch.Tensor, K: int) -> None:
    """Discrete analog of C6 (task_e.py's injectivity guard): a bijection on
    {0..K-1} is injective BY CONSTRUCTION (scatter_ into every index exactly
    once via a permutation of `order`) -- checked here too, not just
    asserted, so a future refactor that breaks the construction is caught
    immediately rather than silently producing wrong labels."""
    sorted_vals, _ = torch.sort(succ, dim=-1)
    arangeK = torch.arange(K, device=succ.device).unsqueeze(0).expand_as(sorted_vals)
    assert torch.equal(sorted_vals, arangeK), \
        "succ is not a bijection on {0..K-1} -- graph construction is broken"


# ---------------------------------------------------------------------------
# Batch sampler -- builds one BYTE-SERIALIZED document per row.
# ---------------------------------------------------------------------------

def sample_doc_batch(cfg: HeTaskConfig, batch_size: int, gen: torch.Generator,
                      hop_set: tuple, device="cpu", assert_bijection: bool = True) -> dict:
    """Returns:
      input_ids   (B, L) int64    byte ids 0..255, L = cfg.doc_len
      target_ids  (B, L-1) int64  input_ids[:, 1:] -- standard next-byte LM
                                   target; the training/eval loss is computed
                                   over logits[:, :-1] vs target_ids (the
                                   standard L-1 available next-byte pairs;
                                   NO wraparound placeholder).
      pred_pos    (B, Q) int64    input-sequence position p such that
                                   logits[:, p] predicts the query's answer
                                   byte (i.e. input_ids[:, p+1] == answer byte)
      answer_id   (B, Q) int64    true answer entity id (0..K-1)
      hop         (B, Q) int64    hop depth used for that query
    Q = cfg.Q_per_doc. Every row uses an INDEPENDENT fresh graph (in-context
    generalization, not weight memorization -- module docstring).
    """
    B, K, Q = batch_size, cfg.K, cfg.Q_per_doc
    succ = _permutation_graph(B, K, gen, device)
    if assert_bijection:
        _assert_bijection(succ, K)

    hops_pool = torch.tensor(hop_set, device=device, dtype=torch.int64)
    hops = hops_pool[torch.randint(0, len(hop_set), (B, Q), generator=gen, device=device)]
    a_idx = torch.randint(0, K, (B, Q), generator=gen, device=device)
    ans_idx = _iterate_permutation(succ, a_idx, hops)

    # Presentation order of the K relation lines -- shuffled INDEPENDENTLY of
    # the graph itself (closes the "position i always == entity i" shortcut).
    pres_order = torch.rand(B, K, generator=gen, device=device).argsort(dim=-1)

    L = cfg.doc_len
    input_ids = torch.empty(B, L, dtype=torch.int64)
    pred_pos = torch.empty(B, Q, dtype=torch.int64)
    succ_cpu, hops_cpu, a_idx_cpu, ans_idx_cpu = (t.cpu() for t in (succ, hops, a_idx, ans_idx))
    pres_order_cpu = pres_order.cpu()

    for b in range(B):
        buf = bytearray(L)
        pos = 0
        for i in pres_order_cpu[b].tolist():
            j = int(succ_cpu[b, i].item())
            buf[pos] = REL
            buf[pos + 1] = ENTITY_BASE + i
            buf[pos + 2] = ENTITY_BASE + j
            buf[pos + 3] = NL
            pos += REL_LINE_LEN
        for q in range(Q):
            h = int(hops_cpu[b, q].item())
            start = int(a_idx_cpu[b, q].item())
            ans = int(ans_idx_cpu[b, q].item())
            buf[pos] = QRY
            buf[pos + 1] = ENTITY_BASE + start
            hop_str = f"{h:02d}"
            buf[pos + 2] = ord(hop_str[0])
            buf[pos + 3] = ord(hop_str[1])
            buf[pos + 4] = ENTITY_BASE + ans
            buf[pos + 5] = NL
            pred_pos[b, q] = pos + 3          # logits HERE predict buf[pos+4] (the answer byte)
            pos += QRY_LINE_LEN
        assert pos == L
        input_ids[b] = torch.tensor(list(buf), dtype=torch.int64)

    input_ids = input_ids.to(device)
    pred_pos = pred_pos.to(device)
    target_ids = input_ids[:, 1:]     # (B, L-1); loss is computed over logits[:, :-1] vs this

    return {
        "input_ids": input_ids, "target_ids": target_ids,
        "pred_pos": pred_pos, "answer_id": ans_idx.to(device), "hop": hops.to(device),
    }


# ---------------------------------------------------------------------------
# Self-test (part of run_stageg_he.py --smoke)
# ---------------------------------------------------------------------------

def _test_periodicity_guard() -> None:
    """Negative test (run to completion, per the standing house rule on
    structural checks): a confounded H_test hop must be REJECTED at config
    construction time, not silently accepted."""
    raised_identity = False
    try:
        HeTaskConfig(K=8, H_train=(1, 2, 3), H_test=(8,), H_extra=())   # 8 % 8 == 0
    except AssertionError:
        raised_identity = True
    assert raised_identity, "periodicity guard FAILED to reject an identity-collapse hop (h % K == 0)"

    raised_residue = False
    try:
        HeTaskConfig(K=8, H_train=(1, 2, 3), H_test=(11,), H_extra=())  # 11 % 8 == 3, a train residue
    except AssertionError:
        raised_residue = True
    assert raised_residue, "periodicity guard FAILED to reject a train-residue-colliding hop"
    print("  periodicity guard correctly REJECTS an identity-collapse hop AND a "
          "train-residue-colliding hop")


def _test_bijection_guard_detects_break() -> None:
    """Negative test: a deliberately broken (non-bijective) succ tensor must
    be REJECTED by _assert_bijection, not silently accepted."""
    K = 6
    succ = torch.arange(K).roll(-1).unsqueeze(0)      # a valid K-cycle: 0->1->2->...->5->0
    _assert_bijection(succ, K)                          # must NOT raise
    broken = succ.clone()
    broken[0, 1] = broken[0, 0]                          # merge two sources onto the same target
    raised = False
    try:
        _assert_bijection(broken, K)
    except AssertionError:
        raised = True
    assert raised, "bijection guard FAILED to detect a broken (merged) successor map"
    print("  bijection guard accepts a genuine K-cycle and REJECTS a merged/broken one")


def _self_test() -> None:
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    cfg = HeTaskConfig(K=12, Q_per_doc=8, H_train=(1, 2, 3), H_test=(4, 5), H_extra=(7,))
    assert cfg.doc_len == 12 * 4 + 8 * 6
    assert abs(cfg.chance - 1.0 / 12) < 1e-9

    for _ in range(5):
        b = sample_doc_batch(cfg, 16, gen, hop_set=cfg.H_train)
        B, K, Q, L = 16, cfg.K, cfg.Q_per_doc, cfg.doc_len
        assert b["input_ids"].shape == (B, L)
        assert b["target_ids"].shape == (B, L - 1)
        assert b["pred_pos"].shape == (B, Q)
        assert b["answer_id"].shape == (B, Q)
        assert b["hop"].shape == (B, Q)
        assert set(torch.unique(b["hop"]).tolist()) <= set(cfg.H_train)
        assert torch.equal(b["target_ids"], b["input_ids"][:, 1:])
        # every byte is either a structural marker or a valid entity byte
        vals = torch.unique(b["input_ids"])
        allowed = {REL, QRY, NL} | {ENTITY_BASE + i for i in range(K)} | \
            {ord(c) for c in "0123456789"}
        assert set(vals.tolist()) <= allowed, f"unexpected byte value(s): {set(vals.tolist()) - allowed}"

        # STRONGEST check: parse the generated bytes back into (start, hop,
        # answer) triples and INDEPENDENTLY recompute pi^h(start) from the
        # PARSED relation lines (not from the tensors used to build the
        # string) -- an end-to-end round-trip proof the serialization
        # matches its own labels, not just a shape check.
        for row in range(B):
            raw = bytes(b["input_ids"][row].tolist())
            succ_map = {}
            for i in range(K):
                line = raw[i * REL_LINE_LEN:(i + 1) * REL_LINE_LEN]
                assert line[0] == REL and line[3] == NL
                succ_map[line[1] - ENTITY_BASE] = line[2] - ENTITY_BASE
            assert sorted(succ_map.keys()) == list(range(K)), "parsed relation lines don't cover 0..K-1"
            assert sorted(succ_map.values()) == list(range(K)), "parsed relation targets aren't a bijection"
            qbase = K * REL_LINE_LEN
            for q in range(Q):
                line = raw[qbase + q * QRY_LINE_LEN: qbase + (q + 1) * QRY_LINE_LEN]
                assert line[0] == QRY and line[5] == NL
                start = line[1] - ENTITY_BASE
                hop = int(chr(line[2]) + chr(line[3]))
                ans_parsed = line[4] - ENTITY_BASE
                cur = start
                for _ in range(hop):
                    cur = succ_map[cur]
                assert cur == ans_parsed, \
                    f"row {row} q {q}: parsed-and-recomputed pi^{hop}({start})={cur} != " \
                    f"parsed answer byte {ans_parsed}"
                assert ans_parsed == int(b["answer_id"][row, q].item()), \
                    "parsed answer byte disagrees with the tensor label"
                assert hop == int(b["hop"][row, q].item())
                # pred_pos alignment: target_ids at pred_pos must equal the answer byte.
                pp = int(b["pred_pos"][row, q].item())
                assert int(b["target_ids"][row, pp].item()) == ENTITY_BASE + ans_parsed, \
                    f"row {row} q {q}: target_ids[pred_pos] does not equal the answer byte"
    print(f"  shapes, byte-vocab range, and parse-and-recompute round-trip (K={K}, "
          f"Q={Q}, doc_len={L}) all OK across 5 fresh batches")

    _test_periodicity_guard()
    _test_bijection_guard_detects_break()

    print("task_he self-test PASSED")


if __name__ == "__main__":
    _self_test()
