"""trackb_candidate3.py -- TRACKB_REDESIGN.md Rev 3 sec 3.3, Candidate 3:
the hand-specified (non-learned) k-hot write schedule, restoring
model_rd.py's own audited buffer/mask convention (`beta = beta_logit *
beta_mask`, model_rd.py:25/:958) on free-running text via a PREPROCESSING
step (a reserved write/buffer token inserted every W real tokens), not a
`DeltaNetLMMixer.forward` change.

SCOPE DECISION (this build session, stated explicitly per the task's own
cut-order framing, sec 10 item 5 -- candidate 3 is "highest scope change...
cut before candidate 1's own manifest is trimmed"): this module builds the
preprocessing adapter (token-stream insertion) and the resulting hard-mask
construction (which is architecturally trivial once the reserved positions
are known -- literally `beta = beta_logit * write_mask`, no learned
selection at all). It does NOT build candidate 3's own from-scratch
training script/model class (sec 3.3's own text: "this is architecturally
closer to model_rd.py's block than to lm_pretrain_rd.py's -- a genuinely
different model class, not a flag on the existing one" -- restoring
model_rd.py's full DeltaNetRDBlock machinery, built for K-item synthetic
episodes, to consume arbitrary free-running text end-to-end is a
substantially larger build than candidates 1/2/4's flag-on-the-existing-
mixer approach, and sec 10 ranks it accordingly). This is the honest,
registered scope boundary for this wave's build -- not silently assumed
complete.
"""
from __future__ import annotations

import torch


def insert_periodic_write_tokens(token_ids: torch.Tensor, doc_offsets: torch.Tensor, period_w: int,
                                  write_token_id: int) -> tuple:
    """sec 3.3's preprocessing step: insert a reserved WRITE token every
    `period_w` real tokens, WITHIN each document (never spanning a document
    boundary -- a reserved token inserted straddling two unrelated
    documents would carry no coherent 'write event' meaning at all, the
    same discipline this codebase's EOT-boundary handling already applies
    elsewhere). token_ids: (T,) int64, a FLAT single-sequence corpus tensor
    (load_corpus's own train/val tensor shape, sec 4.2 item 3's convention
    -- this function operates on the flat corpus, not a batched window,
    since insertion changes every downstream position's absolute index).
    doc_offsets: (n_docs,) int64 document START positions (load_corpus's
    own `{split}_doc_offsets.pt` convention).

    Returns (new_token_ids (T',), write_mask (T',) bool -- True at every
    inserted reserved-write position, new_doc_offsets (n_docs,) int64 --
    each document's start position in the NEW (post-insertion) token
    stream)."""
    assert period_w >= 1, f"period_w={period_w} must be >= 1"
    T = token_ids.numel()
    n_docs = doc_offsets.numel()
    doc_starts = doc_offsets.tolist()
    doc_ends = doc_starts[1:] + [T]

    new_tokens_parts = []
    new_write_mask_parts = []
    new_doc_offsets = []
    cursor = 0
    for start, end in zip(doc_starts, doc_ends):
        new_doc_offsets.append(cursor)
        doc_tokens = token_ids[start:end]
        doc_len = doc_tokens.numel()
        for i in range(0, doc_len, period_w):
            piece = doc_tokens[i:i + period_w]
            new_tokens_parts.append(piece)
            new_write_mask_parts.append(torch.zeros(piece.numel(), dtype=torch.bool))
            new_tokens_parts.append(torch.tensor([write_token_id], dtype=token_ids.dtype))
            new_write_mask_parts.append(torch.ones(1, dtype=torch.bool))
            cursor += piece.numel() + 1

    new_tokens = torch.cat(new_tokens_parts)
    write_mask = torch.cat(new_write_mask_parts)
    new_doc_offsets_t = torch.tensor(new_doc_offsets, dtype=torch.int64)
    assert new_tokens.numel() == write_mask.numel()
    assert int(write_mask.sum().item()) == new_tokens.numel() - T, (
        "exactly one reserved write token per period_w-sized real-token piece, per document -- "
        "the count of inserted tokens must equal (new length - original length)"
    )
    return new_tokens, write_mask, new_doc_offsets_t


def periodic_hard_beta(beta_logit: torch.Tensor, write_mask: torch.Tensor) -> torch.Tensor:
    """sec 3.3: beta architecturally pinned to ZERO everywhere except
    reserved write positions, a REAL learned value only there -- literally
    model_rd.py's own `beta = beta_logit * beta_mask` convention
    (model_rd.py:25/:958), restored here for free text via write_mask
    (this module's insert_periodic_write_tokens output) in place of
    model_rd.py's synthetic-episode `batch["beta_mask"]`. No STE, no
    gradient approximation for the mask itself -- the mask is fixed BEFORE
    training starts (a preprocessing artifact, not a learned or even a
    per-forward-pass quantity), so this is a plain, exact, always-
    differentiable elementwise multiply.

    beta_logit: (B,T,H) -- matches this codebase's own beta shape
    convention (lm_pretrain_rd.py's DeltaNetLMMixer, T is the SECOND-TO-
    LAST dim, not the last one) -- sigmoid NOT yet applied (mirrors
    model_rd.py's own `beta_logit_t * beta_mask_t` ordering -- mask applied
    to the LOGIT, sigmoid(0)=0.5 would NOT be the correct 'zero write
    mass', so the mask must gate the value that ends up meaning zero write
    mass; model_rd.py's own convention masks `beta_logit` directly, i.e.
    beta_logit=0 -> written value 0, which is what this function
    reproduces exactly -- NOT sigmoid(beta_logit)*mask).
    write_mask: (T,) bool -- reshaped here to (T,1) so it broadcasts
    against beta_logit's T (dim=-2) and H (dim=-1) axes correctly,
    regardless of how many leading batch dims beta_logit carries."""
    return beta_logit * write_mask.to(beta_logit.dtype).view(-1, 1)


def sensitivity_period_grid(chunk_size: int, k_sel: int) -> list:
    """sec 10 item 5 / attack round 2's adjudication of sec 11 item 4: if
    candidate 3 survives to Wave 2, its period W gets a CHEAP 2-3-point
    sensitivity check, not a full attack pass. The registered default
    period (sec 3.3: 'e.g. W=chunk_size/K_sel') plus one denser and one
    sparser point, all integer periods, all >=1."""
    default_w = max(1, chunk_size // k_sel)
    grid = sorted(set(w for w in (max(1, default_w // 2), default_w, default_w * 2) if w >= 1))
    return grid
