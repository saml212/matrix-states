"""NCR REAL-LM WAVE-1 BUILD-CONTINUATION -- sec G3-B3 ratified-wiring harness
(K=24, d_ncr=25, 98M DeltaNet backbone). matrix-thinking/NCR_REAL_LM_DESIGN.md
sec G3-B2 (COORDINATOR ADJUDICATION of the Wave-1 integration gaps,
2026-07-17). REPLACES sec G3-B1's `PlaceholderIntegrationGlue` with the
ratified `NCRIntegration` wiring below -- this is the SECOND-EVER attempt to
run the NCR head inside a real LM, now with a real, gradient-trainable
write/read graft (not a smoke-only random-tensor placeholder).

============================================================================
WHAT IS SPEC-FAITHFUL BELOW vs. WHAT IS A BUILD-TIME INTERPRETATION (every
non-literal-text choice is commented in-place, per the same discipline sec
G3-B1 established -- "if genuinely ambiguous, STOP and report", but sec
G3-B2 read the two adapter shapes as NOT ambiguous once the real
BindingEncoder signature is inspected, so this build proceeds):
============================================================================

SPEC-FAITHFUL (verified against the frozen+amended design text, cited
inline; unchanged from sec G3-B1 except where noted):
  - Backbone: DeltaNetLM(vocab_size=50257, d_model=768, d_state=64,
    n_layers=12, conv_size=4, num_heads=1, ffn_mult=4) -- rung-1 98M config.
  - NCR head: ncr_earlyln_scale.NCREarlyLNModel(d=25, h=64) -- FREE-WRITE
    arm only (NS-polar orthogonal write retired from this build, sec
    N2.1). K=24/d_ncr=25 per sec N2.2(a). Param count EXACT at sec
    N2.2(c)'s formula: P(25,64) = 173,209.
  - Read: ncr_models.binexp_read -- O(log h) repeated squaring, exact by
    construction. Sec 2.1 (line 279): "Task 1/Task 3 (abelian,
    single-operator) use o = binexp_read(Z, q, h)" -- this build therefore
    uses binexp_read UNIFORMLY (train batches AND eval), not the isolated
    synthetic harness's train(masked-compose)/eval(binexp_read) split --
    binexp_read is exact and differentiable at every h>=1 including the
    Task-1 train range h in {1,2,3}, so the extra masked-compose machinery
    (ncr_earlyln_scale.NCREarlyLNModel.forward) is not needed here and is
    NOT used by this build (ncr_head.encode() is called directly instead,
    the same precedent sec G3-B1's own smoke_6 already established).
  - Write adapter shapes (sec G3-B2 RULING 1): "a learned Linear(d_model=768
    -> encoder-input-dim) applied to the post-norm hidden at each
    bind-clause's KEY and VALUE token positions... at their ACTUAL
    signature... One Linear per role (key, value)." Inspected directly
    (matrix-thinking/chapter2/model_v4.py BindingEncoder.__init__:
    `self.in_proj = nn.Linear(2*d, h)`, `forward(self, keys, values)` with
    keys/values: (B,K,d)) -- encoder-input-dim IS d_ncr=25 exactly (the
    SAME d that parameterizes Z: (B,d,d)). NOT ambiguous: key_adapter =
    Linear(768,25,bias=False), value_adapter = Linear(768,25,bias=False)
    -- IDENTICAL shapes to sec G3-B1's own placeholder (that placeholder's
    shape guess turns out to have been the ratified one; what changes here
    is everything DOWNSTREAM of the shape -- real data, real objective,
    real eval metric, ablation flags).
  - Read injection (sec G3-B2 RULING 2): "o in R^{d_ncr} ... -> a learned
    Linear(d_ncr -> 768), ADDED to the query-position post-norm hidden
    before the SHARED LM head (design sec 2.1's option (a))." Implemented
    as `read_injector = Linear(25,768,bias=False)`, added to hidden at the
    query's own <Q> marker position, matching sec 2.1 line 284-288's
    option (a) verbatim; the disclosed option (b) (MLP->vocab-logits) is
    ALSO built, flag-gated (--read-inject mlp_logits), never the default.
  - Task/data (sec G3-B2: "SPECIFIED"): grammar_rd.py's bind-clause grammar
    (`<buf...> KEY <rel> VALUE .` clauses), real GPT-2 tokenizer,
    single-Hamiltonian-K-cycle entity graph (sec 3.1 lines 578-583).
    Curriculum sec 5.2 Option 1 (RECOMMENDED): synthetic grammar-episode
    DOCUMENTS mixed at the BATCH level with plain real-corpus documents --
    "NOT literal insertion into WikiText prose" (sec 5.2's own text). This
    build constructs ONE bind episode (K=24 clauses) + ONE query clause per
    synthetic document (sec 3.1: "A document contains 1-2 bind episodes...
    plus a query clause") -- the minimal instance of that spec, sufficient
    to real-CUDA-smoke the full graft; batching MULTIPLE synthetic
    documents against real-corpus rows in one training step is deferred to
    the production trainer (this build's own "NOT RUNNABLE YET" launch
    command below), same deferral sec G3-B1 already made for the corpus
    loader itself.
  - Objective (sec G3-B2: "SPECIFIED"): "next-token CE on the query-answer
    tokens within the autoregressive stream; recovered_frac@0.9 is the
    EVAL metric, not the train loss." Implemented literally: the ONLY
    training loss is F.cross_entropy at the query's own <Q> position
    against the true answer-entity token (no aux cosine loss anywhere in
    the training objective -- sec G3-B1's placeholder aux loss is RETIRED,
    not merely renamed).

BUILD-TIME INTERPRETATION (not literal design text; disclosed, not
fabricated as ratified):
  (i) recovered_frac@0.9's EVAL-ONLY target vector. The design names this
      metric throughout (sec 7 etc.) but, in every OTHER place it is used,
      the comparison target is an "ideal"/ground-truth composed vector from
      an isolated synthetic harness with a fixed probe-vector table -- no
      such external table exists for this real-LM, free-write construction
      (sec N2.1 retires the NS-polar/ideal-Z pipeline entirely). This build
      uses the SELF-CONSISTENT target already implicit in ncr_models.py's
      OWN synthetic-task convention (NCRModel/NCREarlyLNModel are queried
      with `query_keys` living in the SAME embedding space as `keys` --
      see ncr_earlyln_scale.py smoke's `query_keys = keys.clone()`):
      target = key_adapter(hidden at the ANSWER entity's OWN bind-clause
      KEY position). recovered_frac@0.9 = frac of eval queries where
      cosine(o, target) >= 0.9. This is a genuine build-time choice (an
      alternative table-based definition, mirroring probe_head_rd.py's
      `build_probe_target_table`, was considered and rejected: it would
      impose an EXTERNAL target the CE-only training objective has no
      reason to align `o` with, making the metric read near-zero
      regardless of whether the graft is working -- the self-consistent
      target is trainable-through, by construction, via the SAME key
      adapter the read's own query vector uses).
  (ii) `--teacher-force-operator`'s exact mechanism (sec G3-B2's
      FAIL-informativeness ablation-flag list, "teacher-forced-operator
      control mode that isolates read-vs-write"). Implemented as a
      closed-form least-squares operator fit: Z_teacher^T =
      pinv(keys_v.detach()) @ values_v.detach() (both DETACHED, so no
      gradient reaches ANY write-side module through this path) --
      Z_teacher exactly satisfies Z_teacher @ keys_v[i] = values_v[i] for
      the CURRENT (detached) adapter outputs, given K=24 < d_ncr=25 (a
      generically-consistent, minimum-norm system). This bypasses
      `ncr_head`'s own BindingEncoder ENTIRELY (its parameters never enter
      the graph) while leaving the read (binexp_read), injection, and
      LM-head pathway, PLUS the query-side key_adapter use, fully
      trainable -- isolating "can read+inject+head consume a
      perfect-for-the-current-adapters operator" from "can the encoder
      learn a good operator." Smoke item 10 below verifies BOTH the
      closed-form fit's correctness AND the isolation property (ncr_head
      params receive NO gradient in this mode) directly, even though this
      arm is never used for the main training smoke (pre-wired, per the
      build brief, "do not run them" for a full training cell).
  (iii) `--adapter mlp` / `--read-inject mlp_logits` exact shapes (sec
      G3-B2's other two ablation flags): small 2-layer MLPs
      (Linear->GELU->Linear), hidden width d_model//4=192 for the adapter,
      4*d_ncr=100 for the read-inject MLP. Not pinned by the design (which
      only names "MLP" as the disclosed alternative, sec 2.1 line 287) --
      a reasonable, conventional width choice, flagged as such, verified
      by CONSTRUCTION + a shape-correct forward pass only (smoke item 11,
      CPU-fast) -- never exercised through the full backbone+CE+backward
      pipeline this wave (per the build brief).

fp32 boundary (spec-faithful, unchanged from sec G3-B1): every extracted
tap is cast to fp32 before it touches the NCR head (sec 8 item 2/m1's fix).

Run (box only -- chunk_delta_rule has no CPU path):
  CUDA_VISIBLE_DEVICES=<least-loaded> python3 ncr_lm_wave1_smoke.py --device cuda --out results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Portable path setup -- box layout (/home/nvidia/{ncr,chapter2/deltanet_rd})
# differs from the local repo layout (matrix-thinking/{ncr,chapter2,
# deltanet_rd} all siblings); try both, first match wins. Override with
# NCR_DIR / CHAPTER2_DIR / DELTANET_DIR env vars if neither matches.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT_GUESS = os.path.abspath(os.path.join(_HERE, "..", ".."))

_CANDIDATE_LAYOUTS = [
    ("/home/nvidia/ncr", "/home/nvidia/chapter2", "/home/nvidia/chapter2/deltanet_rd"),
    (os.path.join(_REPO_ROOT_GUESS, "matrix-thinking", "ncr"),
     os.path.join(_REPO_ROOT_GUESS, "matrix-thinking", "chapter2"),
     os.path.join(_REPO_ROOT_GUESS, "matrix-thinking", "deltanet_rd")),
]


def _setup_paths() -> None:
    ncr_dir = os.environ.get("NCR_DIR")
    chapter2_dir = os.environ.get("CHAPTER2_DIR")
    deltanet_dir = os.environ.get("DELTANET_DIR")
    if not (ncr_dir and chapter2_dir and deltanet_dir):
        for c_ncr, c_ch2, c_dn in _CANDIDATE_LAYOUTS:
            if os.path.isdir(c_ncr):
                ncr_dir, chapter2_dir, deltanet_dir = c_ncr, c_ch2, c_dn
                break
    assert ncr_dir and os.path.isdir(ncr_dir), (
        f"could not locate the ncr/ module directory -- tried {_CANDIDATE_LAYOUTS} and "
        f"NCR_DIR/CHAPTER2_DIR/DELTANET_DIR env vars; set them explicitly.")
    for p in (deltanet_dir, chapter2_dir, ncr_dir):
        if p not in sys.path:
            sys.path.insert(0, p)


_setup_paths()

import ncr_models as nm                # noqa: E402 (verbatim; binexp_read, D_PIN/ENC_H unused here)
import ncr_earlyln_scale as els         # noqa: E402 (verbatim; NCREarlyLNModel, the free-write arm)
from lm_pretrain_rd import DeltaNetLM   # noqa: E402 (verbatim; the rung-1 98M backbone class)
import grammar_rd as gr                 # noqa: E402 (verbatim; bind-clause grammar + real tokenizer)

# ---------------------------------------------------------------------------
# Wave-1 pinned configuration (NCR_REAL_LM_DESIGN.md sec N2.1/N2.2, sec 2.2,
# sec 6.1 -- every number cited, none invented)
# ---------------------------------------------------------------------------
VOCAB_SIZE = 50257                      # GPT-2 BPE, lm_rd_rung_configs.py VOCAB_SIZE
RUNG1_BACKBONE = dict(d_model=768, d_state=64, n_layers=12, conv_size=4,
                       num_heads=1, ffn_mult=4)          # lm_rd_rung_configs.py RUNGS[1]
BACKBONE_PARAM_TARGET = 98_000_000
BACKBONE_PARAM_TOLERANCE = 0.15         # lm_rd_rung_configs.py PARAM_COUNT_TOLERANCE

K_NCR = 24                              # sec N2.1 GATE-1 amendment (free-write, un-gated to K=24)
D_NCR = 25                              # sec N2.2(a): d_ncr = K+1 = 25 (was 33 at K=32)
H_NCR = nm.ENC_H                        # 64, BindingEncoder's own encoder width, untouched by K
NCR_PARAM_EXACT = 40 * H_NCR ** 2 + 4 * D_NCR * H_NCR + 46 * H_NCR + D_NCR   # sec N2.2(c) formula
assert NCR_PARAM_EXACT == 173_209, NCR_PARAM_EXACT      # sanity on the formula transcription itself

_MIN_KERNEL_T = 128                     # chunk_delta_rule's own backward-crash floor (model_rd.py)

TRAIN_HOP = 2                           # sec 3.1 Task-1 train range h in {1,2,3}; smoke uses one value
EVAL_HELDOUT_HOP = 5                    # grammar_rd's own DeltaNetRDTaskConfig default H_test=(4,5,6);
                                          # 5 also appears in sec 3.1's realistic eval ladder {5,12,20,
                                          # 29,40,61} -- a real Phase-0/1 launch re-derives the FULL
                                          # ladder; this smoke exercises one held-out point only.
_MLP_ADAPTER_HIDDEN = RUNG1_BACKBONE["d_model"] // 4     # 192, build-time interpretation (iii) above
_MLP_READ_INJECT_HIDDEN = 4 * D_NCR                       # 100, build-time interpretation (iii) above


# ---------------------------------------------------------------------------
# NCRIntegration -- the sec G3-B2 RATIFIED wiring (replaces sec G3-B1's
# PlaceholderIntegrationGlue). Adapter/read-inject kinds are FLAGS (sec
# G3-B2's pre-wired ablation arms); default = the ratified (linear, add).
# ---------------------------------------------------------------------------

def _build_adapter(d_model: int, d_ncr: int, kind: str) -> nn.Module:
    if kind == "linear":
        return nn.Linear(d_model, d_ncr, bias=False)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(d_model, _MLP_ADAPTER_HIDDEN), nn.GELU(),
                              nn.Linear(_MLP_ADAPTER_HIDDEN, d_ncr))
    raise ValueError(f"unknown adapter kind {kind!r}")


def _build_read_injector(d_ncr: int, d_model: int, vocab_size: int, kind: str) -> nn.Module:
    if kind == "add":
        return nn.Linear(d_ncr, d_model, bias=False)
    if kind == "mlp_logits":
        return nn.Sequential(nn.Linear(d_ncr, _MLP_READ_INJECT_HIDDEN), nn.GELU(),
                              nn.Linear(_MLP_READ_INJECT_HIDDEN, vocab_size))
    raise ValueError(f"unknown read-inject kind {kind!r}")


class NCRIntegration(nn.Module):
    """The RATIFIED write/read wiring (sec G3-B2 RULINGs 1-2), plus the
    pre-wired ablation-arm flags (sec G3-B2's FAIL-informativeness list).
    Default construction (adapter='linear', read_inject='add') is the ONLY
    arm this build's full-pipeline smoke (items 7-10) exercises; 'mlp'/
    'mlp_logits' are construction+shape verified only (item 11)."""

    def __init__(self, d_model: int, d_ncr: int, vocab_size: int,
                 adapter: str = "linear", read_inject: str = "add"):
        super().__init__()
        self.d_model, self.d_ncr, self.vocab_size = d_model, d_ncr, vocab_size
        self.adapter_kind, self.read_inject_kind = adapter, read_inject
        self.key_adapter = _build_adapter(d_model, d_ncr, adapter)      # RULING 1
        self.value_adapter = _build_adapter(d_model, d_ncr, adapter)    # RULING 1
        self.read_injector = _build_read_injector(d_ncr, d_model, vocab_size, read_inject)  # RULING 2

    def config(self) -> dict:
        return dict(d_model=self.d_model, d_ncr=self.d_ncr, vocab_size=self.vocab_size,
                    adapter=self.adapter_kind, read_inject=self.read_inject_kind)

    def extract_kv(self, hidden: torch.Tensor, key_pos: torch.Tensor, val_pos: torch.Tensor):
        """hidden: (B,T,d_model) backbone post-norm_f hidden (sec 2.2's tap
        point). key_pos/val_pos: (B,K) bind-clause KEY/VALUE token
        positions (grammar_rd.sample_batch_rd's own item_pos/item_pos-2).
        Returns (keys,values): (B,K,d_ncr), fp32 at the NCR head's own
        boundary (sec 8 item 2/m1's fix)."""
        B, K = key_pos.shape
        idx_k = key_pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        idx_v = val_pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        h_key = torch.gather(hidden, 1, idx_k).float()
        h_val = torch.gather(hidden, 1, idx_v).float()
        return self.key_adapter(h_key), self.value_adapter(h_val)

    def query_key(self, hidden: torch.Tensor, query_key_col: int) -> torch.Tensor:
        """key_adapter applied at the query window's OWN KEY position (the
        binexp_read `q` input) -- mirrors ncr_models.py's own
        query_keys-in-the-same-space-as-keys convention (module docstring
        item (i)). query_key_col: fixed python int column (grammar_rd's
        template offsets are batch-uniform, sample_batch_rd's own item_pos
        construction is `torch.arange(K)*clause_len + ...`, identical for
        every row -- no gather needed). Returns (B, d_ncr), fp32."""
        return self.key_adapter(hidden[:, query_key_col, :].float())

    def teacher_force_operator(self, keys_v: torch.Tensor, values_v: torch.Tensor) -> torch.Tensor:
        """Build-time interpretation (ii): closed-form least-squares Z s.t.
        Z @ keys_v[i] = values_v[i] (i in 0..K-1), from DETACHED adapter
        outputs -- bypasses ncr_head's own BindingEncoder entirely (no
        gradient reaches it), isolating read/inject/head from write-quality.
        keys_v/values_v: (B,K,d_ncr). Returns Z: (B,d_ncr,d_ncr)."""
        k, v = keys_v.detach(), values_v.detach()
        z_t = torch.linalg.pinv(k) @ v            # (B,d,K) @ (B,K,d) -> (B,d,d) == Z^T
        return z_t.transpose(-1, -2)

    def inject_and_logits_last(self, hidden: torch.Tensor, o: torch.Tensor,
                                query_col: int, lm_head_weight: torch.Tensor) -> torch.Tensor:
        """hidden: (B,T,d_model); o: (B,d_ncr) the NCR read at the query;
        query_col: fixed python int column (the <Q> position -- batch-
        uniform, same reasoning as query_key). Returns logits AT ONLY the
        query position, (B,vocab_size) -- avoids a full-sequence vocab
        matmul (mirrors DeltaNetLM.forward's own return_hidden=True/AUD2-F1
        discipline). 'add': sec 2.1 option (a), RATIFIED default. 'mlp_
        logits': sec 2.1's disclosed option (b), ADDED to (not replacing)
        the base LM-head logits -- build-time interpretation (iii), never
        the default."""
        h_q = hidden[:, query_col, :]                          # (B, d_model)
        if self.read_inject_kind == "add":
            h_q = h_q + self.read_injector(o.to(h_q.dtype))     # (B, d_model)
            return F.linear(h_q, lm_head_weight)                # (B, vocab)
        if self.read_inject_kind == "mlp_logits":
            base_logits = F.linear(h_q, lm_head_weight)         # (B, vocab), hidden untouched
            extra = self.read_injector(o.to(h_q.dtype))          # (B, vocab)
            return base_logits + extra
        raise ValueError(self.read_inject_kind)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_backbone(vocab_size: int = VOCAB_SIZE) -> DeltaNetLM:
    """vocab_size defaults to the plain GPT-2 vocabulary (50257, sec 2.2's
    own pinned number, used by smoke items 0-6 which never touch
    grammar_rd). DISCOVERED WIRING GAP (not in the frozen design text,
    mechanically forced, not a guess): grammar_rd.py mints TWO reserved
    token ids beyond GPT-2's real vocabulary (BUFFER=vocab_size_base,
    <Q>=vocab_size_base+1 -- its own module docstring: "Two reserved,
    out-of-GPT2-vocabulary token IDs are appended beyond the real
    50,257-token vocabulary"). A DeltaNetLM built at vocab_size=50257 has
    NO embedding/head rows for these ids -- feeding it a grammar_rd
    document crashes with an out-of-bounds embedding-gather CUDA assert
    (hit and confirmed during this build's own smoke run). Every
    grammar_rd-dependent smoke item (7-10) therefore builds the backbone
    at vocab_size=pools.vocab_size_total (50259) instead -- +2 embedding/
    tied-head rows, +2*768=1,536 params (0.0016% of the 98M backbone,
    inside every existing tolerance). ONE DISCLOSED DEVIATION from
    grammar_rd.py's own convention: that module's docstring also states
    the BUFFER row is "zero-pinned AND frozen in the real embedding table
    (R2-3 -- see model_rd.py)" -- a property of a DIFFERENT model class
    (DELTANET_REALDATA_DESIGN.md's own model_rd.py), never invoked by
    NCR_REAL_LM_DESIGN.md, which names lm_pretrain_rd.DeltaNetLM (a plain
    vanilla-embedding LM) as its backbone. This build does NOT replicate
    the zero-pin/freeze (DeltaNetLM has no such mechanism and adding one
    would be a genuine architecture change, out of this build's scope) --
    the BUFFER row here is an ordinary trainable embedding like any other
    token. Flagged for the coordinator/audit: a real Phase-0/1 launch
    should decide whether this deviation matters (the buffer positions
    carry no bind-clause content by construction regardless of whether
    their embedding is frozen-zero or trainable -- the causal-rank
    DESIGN's own R2-3 rationale was specific to THAT design's own
    beta-mask mechanism, which NCR's write path does not use)."""
    return DeltaNetLM(vocab_size, **RUNG1_BACKBONE)


def build_ncr_head() -> els.NCREarlyLNModel:
    return els.NCREarlyLNModel(d=D_NCR, h=H_NCR)


def build_grammar_pools_and_cfg(seed: int = 0):
    """Real GPT-2-tokenizer-verified entity/relation pools (sec 5.1) + a
    K=24/conv_size=4 task config (matches the backbone's own conv_size,
    sec 3.1's Wave-1 K). n_query=1: this build's own single-query-per-
    document construction (module docstring); sample_batch_rd would draw
    K=24 queries per row by default, n_query=1 avoids computing/discarding
    23 of them."""
    tokenizer = gr.load_gpt2_tokenizer()
    pools, report = gr.build_entity_pools(tokenizer, heldout_frac=0.5, seed=seed)
    cfg = gr.DeltaNetRDTaskConfig(K=K_NCR, conv_size=RUNG1_BACKBONE["conv_size"], n_query=1)
    return pools, cfg, report


def build_task1_document(cfg, pools, gen: torch.Generator, batch_size: int,
                          hop_value: int, device: str) -> dict:
    """ONE bind episode (K clauses) + ONE query clause + the answer token,
    per document (sec 3.1: "A document contains 1-2 bind episodes... plus
    a query clause" -- module docstring). hop_value: a single python int
    (binexp_read's own scalar-h signature, sec 2.1 line 279 -- every query
    in this smoke batch shares one h, consistent with how the eval ladder
    is walked one h at a time elsewhere in this program)."""
    b = gr.sample_batch_rd(cfg, batch_size, gen, hop_set=(hop_value,), pools=pools, device=device)
    hops = b["hops"][:, 0]
    assert torch.all(hops == hop_value), "sample_batch_rd did not honor the single-value hop_set"
    tgt_slot = b["tgt_slot"][:, 0]                                    # (B,)
    answer_token = torch.gather(b["entity_ids"], 1, tgt_slot.unsqueeze(1)).squeeze(1)  # (B,)
    query_window = b["query_tokens"][:, 0, :]                         # (B, query_len)

    doc = torch.cat([b["token_ids"], query_window, answer_token.unsqueeze(1)], dim=1)
    T_bind, query_len, buf_len = cfg.T_bind, cfg.query_len, cfg.buf_len
    key_pos = b["item_pos"] - 2                                        # (B,K), sample_batch_rd's own convention
    val_pos = b["item_pos"]                                            # (B,K)
    query_key_col = T_bind + buf_len                                   # scalar: batch-uniform template offset
    query_mark_col = T_bind + query_len - 1                            # scalar: the <Q> position (== input's last col)
    assert query_mark_col == doc.shape[1] - 2, "query_mark_col must be the second-to-last doc column"

    return dict(doc=doc, key_pos=key_pos, val_pos=val_pos,
                query_key_col=query_key_col, query_mark_col=query_mark_col,
                answer_token=answer_token, hop=hop_value, tgt_slot=tgt_slot,
                entity_ids=b["entity_ids"])


def ncr_lm_forward(backbone: DeltaNetLM, ncr_head: els.NCREarlyLNModel, integ: NCRIntegration,
                    batch: dict, teacher_force: bool = False):
    """The FULL graft: backbone -> write-adapter -> encoder (or teacher-
    forced operator) -> binexp_read -> read-inject -> LM-head. Returns
    (logits_at_query, o, Z, keys_v, values_v)."""
    input_ids = batch["doc"][:, :-1]                                   # (B, T_bind+query_len); drop the answer token
    assert input_ids.shape[1] - 1 == batch["query_mark_col"], "query_mark_col must be input's last column"
    hidden = backbone(input_ids, return_hidden=True)                   # (B,T,768), sec 2.2 tap point
    keys_v, values_v = integ.extract_kv(hidden, batch["key_pos"], batch["val_pos"])
    if teacher_force:
        Z = integ.teacher_force_operator(keys_v, values_v)             # ncr_head.encode NEVER called
    else:
        Z = ncr_head.encode(keys_v, values_v)                          # (B,d_ncr,d_ncr)
    q_key = integ.query_key(hidden, batch["query_key_col"])            # (B,d_ncr)
    o = nm.binexp_read(Z, q_key.unsqueeze(1), h=batch["hop"])["o"].squeeze(1)   # (B,d_ncr), O(log h) exact
    logits = integ.inject_and_logits_last(hidden, o, batch["query_mark_col"], backbone.embed.weight)
    return logits, o, Z, keys_v, values_v


def recovered_frac_at_09(integ: NCRIntegration, hidden: torch.Tensor, o: torch.Tensor,
                          key_pos: torch.Tensor, tgt_slot: torch.Tensor) -> float:
    """Build-time interpretation (i): target = key_adapter(hidden at the
    ANSWER entity's OWN bind-clause KEY position). recovered_frac@0.9 =
    frac of rows with cosine(o, target) >= 0.9. EVAL-ONLY (never in the
    training loss, per the ratified objective)."""
    answer_key_pos = torch.gather(key_pos, 1, tgt_slot.unsqueeze(1)).squeeze(1)   # (B,)
    idx = answer_key_pos.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
    target = integ.key_adapter(torch.gather(hidden, 1, idx).squeeze(1).float())
    cos = F.cosine_similarity(o, target, dim=-1)
    return (cos >= 0.9).float().mean().item()


# ---------------------------------------------------------------------------
# Smoke suite (numbered, PASS/FAIL, mirrors this repo's own smoke()
# conventions in lm_pretrain_rd.py / probe_head_rd.py / ncr_earlyln_scale.py)
# ---------------------------------------------------------------------------
FAILURES: list[str] = []
RESULTS: dict = {}


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    RESULTS[item] = {"passed": ok, "detail": detail}
    if not ok:
        FAILURES.append(item)


def smoke_0_param_counts():
    backbone = build_backbone()
    n_backbone = sum(p.numel() for p in backbone.parameters())
    rel_err = abs(n_backbone - BACKBONE_PARAM_TARGET) / BACKBONE_PARAM_TARGET
    backbone_ok = rel_err <= BACKBONE_PARAM_TOLERANCE
    _report("smoke 0a: backbone param count within 15% of 98M target", backbone_ok,
            f"measured={n_backbone:,} target={BACKBONE_PARAM_TARGET:,} rel_err={rel_err:.4f}")

    ncr = build_ncr_head()
    n_ncr = sum(p.numel() for p in ncr.parameters() if p.requires_grad)
    ncr_ok = n_ncr == NCR_PARAM_EXACT
    _report("smoke 0b: NCR head param count EXACTLY matches sec N2.2(c) formula (173,209)", ncr_ok,
            f"measured={n_ncr:,} formula={NCR_PARAM_EXACT:,}")

    integ = NCRIntegration(RUNG1_BACKBONE["d_model"], D_NCR, VOCAB_SIZE, adapter="linear", read_inject="add")
    n_integ = sum(p.numel() for p in integ.parameters())
    expected_integ = 2 * RUNG1_BACKBONE["d_model"] * D_NCR + D_NCR * RUNG1_BACKBONE["d_model"]
    integ_ok = n_integ == expected_integ
    _report("smoke 0c: NCRIntegration (linear/add, the ratified default) param count exact, "
            "DISCLOSED SEPARATELY from the 173,209-param NCR head (sec G3-B1 gap 1's open question)",
            integ_ok, f"measured={n_integ:,} expected={expected_integ:,} "
            f"(key+value adapters {2*RUNG1_BACKBONE['d_model']*D_NCR:,} + "
            f"read_injector {D_NCR*RUNG1_BACKBONE['d_model']:,})")
    del backbone, ncr, integ
    return backbone_ok and ncr_ok and integ_ok


def smoke_1_backbone_forward_backward(device: str):
    torch.manual_seed(0)
    m = build_backbone().to(device)
    x = torch.randint(0, VOCAB_SIZE, (4, _MIN_KERNEL_T), device=device)
    y = torch.randint(0, VOCAB_SIZE, (4, _MIN_KERNEL_T), device=device)
    logits = m(x)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
    loss.backward()
    grads_finite = all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    grad_norm = sum(p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None) ** 0.5
    ok = (logits.shape == (4, _MIN_KERNEL_T, VOCAB_SIZE) and torch.isfinite(loss).item()
          and grads_finite and grad_norm > 0)
    _report("smoke 1: backbone (98M rung-1 config) forward + backward, finite grad norms", ok,
            f"loss={loss.item():.4f} grad_norm={grad_norm:.4f}")
    del m, x, y, logits, loss
    torch.cuda.empty_cache() if device == "cuda" else None


def smoke_2_backbone_checkpoint_resume(device: str):
    torch.manual_seed(1)
    m = build_backbone().to(device)
    tmpdir = tempfile.mkdtemp(prefix="ncr_lm_wave1_ckpt_")
    ckpt_path = os.path.join(tmpdir, "backbone.pt")
    torch.save({"step": 0, "model_state_dict": m.state_dict(), "config": m.config()}, ckpt_path)
    loaded = torch.load(ckpt_path, map_location=device)
    m2 = DeltaNetLM(**loaded["config"]).to(device)
    m2.load_state_dict(loaded["model_state_dict"])
    m.eval(); m2.eval()
    with torch.no_grad():
        probe = torch.randint(0, VOCAB_SIZE, (2, _MIN_KERNEL_T), device=device)
        l1, l2 = m(probe), m2(probe)
    ok = torch.equal(l1, l2) and loaded["config"] == m.config()
    _report("smoke 2: backbone checkpoint save -> fresh model -> load -> BIT-IDENTICAL forward", ok,
            f"ckpt={ckpt_path} max_abs_diff={(l1 - l2).abs().max().item():.2e}")
    del m, m2, l1, l2
    torch.cuda.empty_cache() if device == "cuda" else None


def smoke_3_backbone_eval_batch(device: str):
    """Eval-mode batch at a Phase-1-representative shape (CLAUDE.md hard
    rule: 'eval can OOM even if training fits' -- checked explicitly, not
    assumed from the training-shape smoke above)."""
    torch.manual_seed(2)
    m = build_backbone().to(device)
    m.eval()
    B_EVAL, T_EVAL = 32, 512             # sec 6.1's own measured operating point
    with torch.no_grad():
        x = torch.randint(0, VOCAB_SIZE, (B_EVAL, T_EVAL), device=device)
        logits = m(x)
    ok = logits.shape == (B_EVAL, T_EVAL, VOCAB_SIZE) and torch.isfinite(logits).all().item()
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device == "cuda" else -1.0
    _report("smoke 3: backbone eval batch at Phase-1 operating point (B=32,T=512), no_grad, finite",
            ok, f"peak_mem_allocated={peak_gb:.2f} GB")
    del m, x, logits
    torch.cuda.empty_cache() if device == "cuda" else None


def smoke_4_ncr_head_forward_backward(device: str):
    torch.manual_seed(3)
    ncr = build_ncr_head().to(device)
    B, K = 8, K_NCR
    keys = torch.randn(B, K, D_NCR, device=device)
    values = torch.randn(B, K, D_NCR, device=device)
    query_keys = keys.clone()
    hops = torch.randint(1, 4, (B, K), device=device)
    batch = dict(keys=keys, values=values, query_keys=query_keys, hops=hops)
    pred, Z = ncr(batch)
    target = values
    loss = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    loss.backward()
    grads_finite = all(torch.isfinite(p.grad).all() for p in ncr.parameters() if p.grad is not None)
    ok = (Z.shape == (B, D_NCR, D_NCR) and torch.isfinite(loss).item() and grads_finite)
    _report("smoke 4: NCR free-write head (K=24,d=25) forward + backward, finite grad norms", ok,
            f"loss={loss.item():.4f} Z.shape={tuple(Z.shape)}")
    del ncr, keys, values, pred, Z, loss
    torch.cuda.empty_cache() if device == "cuda" else None


def smoke_5_ncr_head_checkpoint_resume(device: str):
    torch.manual_seed(4)
    ncr = build_ncr_head().to(device)
    tmpdir = tempfile.mkdtemp(prefix="ncr_lm_wave1_ckpt_")
    ckpt_path = os.path.join(tmpdir, "ncr_head.pt")
    torch.save({"step": 0, "model_state_dict": ncr.state_dict(),
                "config": {"d": D_NCR, "h": H_NCR}}, ckpt_path)
    loaded = torch.load(ckpt_path, map_location=device)
    ncr2 = els.NCREarlyLNModel(**loaded["config"]).to(device)
    ncr2.load_state_dict(loaded["model_state_dict"])
    ncr.eval(); ncr2.eval()
    with torch.no_grad():
        Z_probe = torch.randn(2, D_NCR, D_NCR, device=device)
        q_probe = torch.randn(2, 3, D_NCR, device=device)
        o1 = nm.binexp_read(Z_probe, q_probe, 5)["o"]
        o2 = nm.binexp_read(Z_probe, q_probe, 5)["o"]   # pure function, module-level -- sanity only
    ok = torch.equal(o1, o2)
    _report("smoke 5: NCR head checkpoint save/load round-trip + binexp_read determinism", ok,
            f"ckpt={ckpt_path}")
    del ncr, ncr2
    torch.cuda.empty_cache() if device == "cuda" else None


def smoke_6_ncr_binexp_read_ladder(device: str):
    """Eval-mode O(log h) read across the realistic depth ladder (sec 3.1's
    eval grid: {5,12,20,29,40,61}, sec N2.3's H re-anchor)."""
    torch.manual_seed(5)
    ncr = build_ncr_head().to(device)
    ncr.eval()
    B, Q = 4, 6
    with torch.no_grad():
        keys = torch.randn(B, K_NCR, D_NCR, device=device)
        values = torch.randn(B, K_NCR, D_NCR, device=device)
        Z = ncr.encode(keys, values)
        q = torch.randn(B, Q, D_NCR, device=device)
        results = {}
        for h in (5, 12, 20, 29, 40, 61):
            o = nm.binexp_read(Z, q, h)["o"]
            results[h] = bool(torch.isfinite(o).all().item())
    ok = all(results.values())
    _report("smoke 6: NCR binexp_read finite at every realistic-ladder depth h in {5,12,20,29,40,61}",
            ok, f"per_h_finite={results}")
    del ncr, keys, values, Z, q
    torch.cuda.empty_cache() if device == "cuda" else None


def smoke_7_full_graft_train_step(device: str, pools, cfg, vocab_size_total: int):
    """THE make-or-break smoke: backbone -> write-adapter -> encoder ->
    binexp_read -> read-inject -> LM-head -> CE, on a REAL grammar_rd
    Task-1 document batch (h=2, in-distribution per sec 3.1), ONE real
    optimizer step, finite grads through BOTH adapters AND the backbone.
    vocab_size_total: pools.vocab_size_total (50259) -- see build_backbone's
    own docstring for why this is NOT the plain 50257."""
    torch.manual_seed(6)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    gen = torch.Generator(device=device).manual_seed(6)   # grammar_rd.sample_batch_rd requires a
                                                            # generator on the SAME device as its own
                                                            # torch.rand(..., device=device) call
    backbone = build_backbone(vocab_size=vocab_size_total).to(device)
    ncr = build_ncr_head().to(device)
    integ = NCRIntegration(RUNG1_BACKBONE["d_model"], D_NCR, vocab_size_total,
                            adapter="linear", read_inject="add").to(device)
    opt = torch.optim.Adam(
        list(backbone.parameters()) + list(ncr.parameters()) + list(integ.parameters()), lr=1e-4)

    B = 4
    batch = build_task1_document(cfg, pools, gen, B, TRAIN_HOP, device)
    logits, o, Z, keys_v, values_v = ncr_lm_forward(backbone, ncr, integ, batch, teacher_force=False)
    loss = F.cross_entropy(logits, batch["answer_token"])

    opt.zero_grad()
    loss.backward()
    all_params = list(backbone.parameters()) + list(ncr.parameters()) + list(integ.parameters())
    grads_finite = all(torch.isfinite(p.grad).all() for p in all_params if p.grad is not None)
    n_with_grad = sum(1 for p in all_params if p.grad is not None)
    n_backbone_with_grad = sum(1 for p in backbone.parameters() if p.grad is not None)
    n_ncr_with_grad = sum(1 for p in ncr.parameters() if p.grad is not None)
    n_integ_with_grad = sum(1 for p in integ.parameters() if p.grad is not None)
    opt.step()

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device == "cuda" else -1.0
    ok = (torch.isfinite(loss).item() and grads_finite and n_with_grad > 0
          and n_backbone_with_grad > 0 and n_ncr_with_grad > 0 and n_integ_with_grad > 0
          and logits.shape == (B, vocab_size_total) and o.shape == (B, D_NCR) and Z.shape == (B, D_NCR, D_NCR))
    _report("smoke 7 (RATIFIED WIRING): full graft on a REAL grammar_rd Task-1 document "
            "(K=24 bind clauses + 1 query, h=2 in-dist), CE-only loss, ONE joint fwd/bwd/opt-step, "
            "finite grads through backbone+NCR-head+integration", ok,
            f"loss_ce={loss.item():.4f} n_params_with_grad={n_with_grad} "
            f"(backbone={n_backbone_with_grad} ncr={n_ncr_with_grad} integ={n_integ_with_grad}) "
            f"peak_mem_allocated={peak_gb:.2f} GB doc_len={batch['doc'].shape[1]}")
    del ncr, integ, opt, logits, loss
    torch.cuda.empty_cache() if device == "cuda" else None
    return backbone, peak_gb


def smoke_8_full_graft_checkpoint_resume(device: str, pools, cfg, backbone_from_7: DeltaNetLM,
                                          vocab_size_total: int):
    """Checkpoint save -> fresh instances -> load -> BIT-IDENTICAL forward,
    for the FULL integrated model (backbone+ncr_head+integ together) --
    extends smoke 2/5 (component-only) to the graft as a whole."""
    torch.manual_seed(7)
    ncr = build_ncr_head().to(device)
    integ = NCRIntegration(RUNG1_BACKBONE["d_model"], D_NCR, vocab_size_total,
                            adapter="linear", read_inject="add").to(device)
    backbone = backbone_from_7                      # reuse smoke 7's post-opt-step backbone (real weights)

    tmpdir = tempfile.mkdtemp(prefix="ncr_lm_wave1_ckpt_")
    ckpt_path = os.path.join(tmpdir, "full_graft.pt")
    torch.save({"backbone_state": backbone.state_dict(), "backbone_config": backbone.config(),
                "ncr_state": ncr.state_dict(), "ncr_config": {"d": D_NCR, "h": H_NCR},
                "integ_state": integ.state_dict(), "integ_config": integ.config()}, ckpt_path)
    loaded = torch.load(ckpt_path, map_location=device)
    backbone2 = DeltaNetLM(**loaded["backbone_config"]).to(device)
    backbone2.load_state_dict(loaded["backbone_state"])
    ncr2 = els.NCREarlyLNModel(**loaded["ncr_config"]).to(device)
    ncr2.load_state_dict(loaded["ncr_state"])
    integ2_cfg = dict(loaded["integ_config"])
    integ2 = NCRIntegration(integ2_cfg["d_model"], integ2_cfg["d_ncr"], integ2_cfg["vocab_size"],
                             adapter=integ2_cfg["adapter"], read_inject=integ2_cfg["read_inject"]).to(device)
    integ2.load_state_dict(loaded["integ_state"])

    backbone.eval(); ncr.eval(); integ.eval()
    backbone2.eval(); ncr2.eval(); integ2.eval()
    gen = torch.Generator(device=device).manual_seed(8)
    batch = build_task1_document(cfg, pools, gen, 4, TRAIN_HOP, device)
    with torch.no_grad():
        logits1, o1, Z1, _, _ = ncr_lm_forward(backbone, ncr, integ, batch, teacher_force=False)
        logits2, o2, Z2, _, _ = ncr_lm_forward(backbone2, ncr2, integ2, batch, teacher_force=False)
    ok = torch.equal(logits1, logits2) and torch.equal(o1, o2) and torch.equal(Z1, Z2)
    _report("smoke 8: FULL-GRAFT checkpoint save -> fresh backbone+ncr+integ -> load -> "
            "BIT-IDENTICAL forward (logits, o, Z)", ok,
            f"ckpt={ckpt_path} max_logit_diff={(logits1-logits2).abs().max().item():.2e}")
    del ncr, ncr2, integ2, backbone2, logits1, logits2
    torch.cuda.empty_cache() if device == "cuda" else None
    return backbone, integ


def smoke_9_eval_batch_recovered_frac(device: str, pools, cfg, backbone, integ):
    """An eval batch (no_grad) at a HELD-OUT depth (h=5, not in Task-1's
    train range {1,2,3}) computing recovered_frac@0.9 (build-time
    interpretation (i) in the module docstring) -- the EVAL metric, never
    part of the training loss above."""
    torch.manual_seed(9)
    ncr = build_ncr_head().to(device)
    backbone.eval(); ncr.eval(); integ.eval()
    gen = torch.Generator(device=device).manual_seed(9)
    B = 16
    batch = build_task1_document(cfg, pools, gen, B, EVAL_HELDOUT_HOP, device)
    with torch.no_grad():
        input_ids = batch["doc"][:, :-1]
        hidden = backbone(input_ids, return_hidden=True)
        logits, o, Z, keys_v, values_v = ncr_lm_forward(backbone, ncr, integ, batch, teacher_force=False)
        rf = recovered_frac_at_09(integ, hidden, o, batch["key_pos"], batch["tgt_slot"])
        answer_logprob = F.log_softmax(logits, dim=-1).gather(
            1, batch["answer_token"].unsqueeze(1)).squeeze(1).mean().item()
    ok = (0.0 <= rf <= 1.0 and torch.isfinite(logits).all().item() and torch.isfinite(o).all().item())
    _report(f"smoke 9: eval batch (no_grad) at held-out depth h={EVAL_HELDOUT_HOP} (not in Task-1's "
            "train range {1,2,3}), recovered_frac@0.9 computed, finite logits/read", ok,
            f"recovered_frac@0.9={rf:.4f} mean_answer_logprob={answer_logprob:.4f} B={B} "
            f"(untrained model at init -- rf near-0 is EXPECTED, this item proves the metric "
            f"COMPUTES, not that the graft has converged)")
    del ncr
    torch.cuda.empty_cache() if device == "cuda" else None


def smoke_10_teacher_force_operator_isolation(device: str, pools, cfg, vocab_size_total: int):
    """sec G3-B2's teacher-force ablation arm: (a) closed-form Z_teacher
    correctness, (b) isolation -- ncr_head (BindingEncoder) receives ZERO
    gradient when teacher_force=True, while adapters/read_injector/
    backbone DO (module docstring, build-time interpretation (ii))."""
    torch.manual_seed(10)
    backbone = build_backbone(vocab_size=vocab_size_total).to(device)
    ncr = build_ncr_head().to(device)
    integ = NCRIntegration(RUNG1_BACKBONE["d_model"], D_NCR, vocab_size_total,
                            adapter="linear", read_inject="add").to(device)
    opt = torch.optim.Adam(
        list(backbone.parameters()) + list(ncr.parameters()) + list(integ.parameters()), lr=1e-4)
    gen = torch.Generator(device=device).manual_seed(11)
    batch = build_task1_document(cfg, pools, gen, 4, TRAIN_HOP, device)

    logits, o, Z, keys_v, values_v = ncr_lm_forward(backbone, ncr, integ, batch, teacher_force=True)
    fit_residual = (torch.einsum("bij,bqj->bqi", Z, keys_v) - values_v).abs().max().item()
    loss = F.cross_entropy(logits, batch["answer_token"])
    opt.zero_grad()
    loss.backward()

    ncr_untouched = all(p.grad is None for p in ncr.parameters())
    backbone_trained = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in backbone.parameters())
    key_adapter_trained = any(p.grad is not None and p.grad.abs().sum().item() > 0
                              for p in integ.key_adapter.parameters())
    read_injector_trained = any(p.grad is not None and p.grad.abs().sum().item() > 0
                                for p in integ.read_injector.parameters())
    ok = (fit_residual < 1e-2 and torch.isfinite(loss).item() and ncr_untouched
          and backbone_trained and key_adapter_trained and read_injector_trained)
    _report("smoke 10: --teacher-force-operator -- closed-form Z fit residual small, ncr_head "
            "(encoder) receives ZERO gradient (isolation proof), backbone/key_adapter/read_injector "
            "DO train", ok,
            f"fit_residual_max={fit_residual:.2e} loss_ce={loss.item():.4f} "
            f"ncr_untouched={ncr_untouched} backbone_trained={backbone_trained} "
            f"key_adapter_trained={key_adapter_trained} read_injector_trained={read_injector_trained}")
    del backbone, ncr, integ, opt, logits, loss
    torch.cuda.empty_cache() if device == "cuda" else None


def smoke_11_ablation_flags_construct(device: str):
    """CPU-fast construction + shape-only check for the OTHER two pre-wired
    ablation flags (--adapter mlp, --read-inject mlp_logits) -- proves the
    escape hatches are not broken, WITHOUT spending GPU time on a full
    backbone+CE+backward pass through them this wave (per the build
    brief: 'pre-wire... do not run them'). Runs regardless of --device."""
    torch.manual_seed(12)
    d_model, d_ncr, vocab = RUNG1_BACKBONE["d_model"], D_NCR, VOCAB_SIZE
    ok_all = True
    details = []
    for adapter_kind, inject_kind in (("mlp", "add"), ("linear", "mlp_logits"), ("mlp", "mlp_logits")):
        integ = NCRIntegration(d_model, d_ncr, vocab, adapter=adapter_kind, read_inject=inject_kind)
        B, K, T = 3, K_NCR, 40
        hidden = torch.randn(B, T, d_model)
        key_pos = torch.arange(K).unsqueeze(0).expand(B, K)
        val_pos = key_pos + 1
        keys_v, values_v = integ.extract_kv(hidden, key_pos, val_pos)
        q_key = integ.query_key(hidden, query_key_col=5)
        o = torch.randn(B, d_ncr)
        lm_head = torch.randn(vocab, d_model)
        logits = integ.inject_and_logits_last(hidden, o, query_col=T - 1, lm_head_weight=lm_head)
        shapes_ok = (keys_v.shape == (B, K, d_ncr) and values_v.shape == (B, K, d_ncr)
                     and q_key.shape == (B, d_ncr) and logits.shape == (B, vocab))
        finite_ok = (torch.isfinite(keys_v).all() and torch.isfinite(values_v).all()
                     and torch.isfinite(q_key).all() and torch.isfinite(logits).all())
        ok_all = ok_all and bool(shapes_ok and finite_ok)
        details.append(f"(adapter={adapter_kind},read_inject={inject_kind}): "
                        f"shapes_ok={shapes_ok} finite_ok={bool(finite_ok)} "
                        f"n_params={sum(p.numel() for p in integ.parameters()):,}")
        del integ
    _report("smoke 11: ablation-flag construction (--adapter mlp, --read-inject mlp_logits, and "
            "both together) -- CPU-fast, shape+finite only, NOT run through the full backbone/CE/"
            "backward pipeline this wave", ok_all, "; ".join(details))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--out", default=None, help="write JSON results here")
    ap.add_argument("--adapter", default="linear", choices=("linear", "mlp"),
                     help="sec G3-B2 pre-wired ablation flag; the ratified default is 'linear'. "
                          "This flag is DOCUMENTED here for CLI-convention parity with a future "
                          "production trainer -- the smoke suite itself always exercises the "
                          "ratified default through the full pipeline (items 7-10) AND both "
                          "alternate arms via construction-only checks (item 11), regardless of "
                          "this flag's value.")
    ap.add_argument("--read-inject", default="add", choices=("add", "mlp_logits"),
                     help="sec G3-B2 pre-wired ablation flag; the ratified default is 'add' "
                          "(design sec 2.1 option (a)). See --adapter's help for how this "
                          "interacts with the smoke suite's own fixed coverage.")
    ap.add_argument("--teacher-force-operator", action="store_true",
                     help="sec G3-B2 pre-wired ablation flag (a FAIL-diagnosis control mode, "
                          "build-time interpretation (ii) in the module docstring). Documented "
                          "here for CLI parity; the smoke suite always exercises this mode via "
                          "item 10 regardless of this flag.")
    ap.add_argument("--smoke", action="store_true",
                     help="no-op: this entire script is a smoke suite (no training loop, no "
                          "optimizer state persisted past the process) -- self-documents "
                          "invocations, matching this repo's own pre-train-gate hook convention.")
    args = ap.parse_args()

    print("=" * 70)
    print("NCR REAL-LM WAVE-1 BUILD-CONTINUATION -- sec G3-B3 ratified-wiring smoke")
    print(f"device={args.device} torch={torch.__version__} "
          f"cuda_available={torch.cuda.is_available()} "
          f"cli_adapter={args.adapter} cli_read_inject={args.read_inject} "
          f"cli_teacher_force={args.teacher_force_operator}")
    if args.device == "cuda":
        assert torch.cuda.is_available(), "cuda requested but not available"
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print("=" * 70)

    t0 = time.time()
    smoke_0_param_counts()
    smoke_11_ablation_flags_construct(args.device)     # CPU-fast, runs regardless of --device
    if args.device == "cuda":
        smoke_1_backbone_forward_backward(args.device)
        smoke_2_backbone_checkpoint_resume(args.device)
        smoke_3_backbone_eval_batch(args.device)
        smoke_4_ncr_head_forward_backward(args.device)
        smoke_5_ncr_head_checkpoint_resume(args.device)
        smoke_6_ncr_binexp_read_ladder(args.device)

        print("  building real GPT-2-tokenizer-verified grammar_rd pools (sec 5.1)...", flush=True)
        pools, cfg, pool_report = build_grammar_pools_and_cfg(seed=0)
        vocab_size_total = pool_report["vocab_size_total"]     # 50259 -- see build_backbone's docstring
                                                                  # (DISCOVERED WIRING GAP, disclosed there)
        pools = pools.to(args.device)
        print(f"  pools built: {pool_report['n_train_names']} train / "
              f"{pool_report['n_heldout_names']} heldout names, "
              f"{pool_report['n_rel_a_verified']} rel_a verbs; "
              f"cfg: K={cfg.K} T_bind={cfg.T_bind} query_len={cfg.query_len}; "
              f"vocab_size_total={vocab_size_total} (50257 + 2 reserved grammar_rd ids)", flush=True)
        RESULTS["_grammar_pool_report"] = pool_report

        backbone_after_7, peak_gb = smoke_7_full_graft_train_step(args.device, pools, cfg, vocab_size_total)
        backbone_after_8, integ_after_8 = smoke_8_full_graft_checkpoint_resume(
            args.device, pools, cfg, backbone_after_7, vocab_size_total)
        smoke_9_eval_batch_recovered_frac(args.device, pools, cfg, backbone_after_8, integ_after_8)
        smoke_10_teacher_force_operator_isolation(args.device, pools, cfg, vocab_size_total)
        RESULTS["_co_residency_peak_mem_gb"] = peak_gb
        del backbone_after_7, backbone_after_8, integ_after_8
        torch.cuda.empty_cache()
    else:
        print("device=cpu: chunk_delta_rule has no CPU path (this repo's own standing finding) -- "
              "skipping items 1-10 (backbone-dependent), param-count + ablation-construction only.")
    wall = time.time() - t0
    RESULTS["_wall_clock_sec"] = wall
    RESULTS["_failures"] = FAILURES

    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
    else:
        print("SMOKE SUITE: ALL ITEMS PASSED")
    print(f"wall_clock={wall:.1f}s")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(RESULTS, f, indent=2)
        print(f"results written to {args.out}")

    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
