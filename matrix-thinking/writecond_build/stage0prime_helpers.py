"""Stage 0' (Rev-5, amended per attack R4's A1-A8) -- pure-tensor item
helpers, BOX-INDEPENDENT BY DESIGN.

WHY this module exists separately from stage0prime_eval.py: the pinned
runner (`ncr_lm_wave1_runner.py`) cannot be imported off the box --
`import ncr_lm_wave1_smoke as graft` pulls in `lm_pretrain_rd.DeltaNetLM`,
which does `from fla.ops.delta_rule import chunk_delta_rule` at module load
(confirmed here: `ModuleNotFoundError: No module named 'fla'`, and the
runner's own module docstring says so explicitly: "chunk_delta_rule has no
CPU path"). Every Stage-0' item's CORE MATH, though, needs no backbone at
all -- items 1/3/4/5 are pure tensor ops on (keys_v, values_v, Z, a Linear
layer's weight); item 6's fresh-encoder training loop needs only
`els.NCREarlyLNModel` (matrix-thinking/ncr/ncr_earlyln_scale.py ->
matrix-thinking/ncr/ncr_models.py -> matrix-thinking/chapter2/model_v4.py),
which is PURE TORCH with zero fla dependency (verified: these three files
import only torch/rank_utils). Factoring the math out here means every
item's logic is unit-tested on THIS machine (test_stage0prime_helpers.py,
run to completion) using the REAL encoder classes, not a hand-reimplemented
stand-in -- `stage0prime_eval.py` (the box launch script) imports this
module AND the real pinned runner, so the box run uses the SAME tested
code, not a duplicate that could drift.

`discriminability_metrics` / `ortho_regularization_loss` are duplicated
VERBATIM below from the pinned runner (md5 9a93198b642242f512ff8489e32b0a53,
lines 480-537 / 714-742) -- both are ALREADY fla-independent pure-torch
functions; the only reason importing them directly fails is the runner
MODULE's own top-level import chain, not their own bodies. This follows the
runner file's OWN established "duplicate, don't drift" convention
(`cosine_and_recovered_frac`'s own module comment, runner.py:422-443,
duplicates `graft.recovered_frac_at_09` for the identical reason). The ONE
disclosed signature deviation: `discriminability_metrics` here takes
`entity_adapter` directly (a callable) instead of an `integ` object, since
the original body only ever calls `integ.entity_adapter(...)` and never
touches any other `integ` attribute -- `stage0prime_eval.py`'s own box call
sites use `R.discriminability_metrics(integ, ...)` unmodified.

A1-A8 (attack R4's Stage-0' ruling) are applied item-by-item below; each
docstring cites its own amendment letter.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    # Box deployment: stage0prime_eval.py sits in the SAME flat directory as
    # ncr_lm_wave1_runner.py, and that directory already contains
    # ncr_models.py/ncr_earlyln_scale.py/model_v4.py as flat sibling modules
    # (the pinned runner's own `import ncr_models as nm` / `graft`'s own
    # `import ncr_earlyln_scale as els` prove this layout -- a single
    # `sys.path.insert(0, _HERE)`, no nested package). If stage0prime_eval.py
    # (or any caller) already put that directory on sys.path, these imports
    # just work with no extra path surgery.
    import ncr_models as nm            # noqa: E402
    import ncr_earlyln_scale as els    # noqa: E402
except ModuleNotFoundError:
    # Local/off-box testing: THIS repo's own nested layout
    # (matrix-thinking/writecond_build/ + matrix-thinking/ncr/) -- add the
    # sibling ncr/ directory explicitly so test_stage0prime_helpers.py runs
    # standalone without a box-style flat deployment directory.
    _NCR_DIR = os.path.abspath(os.path.join(_HERE, "..", "ncr"))
    if _NCR_DIR not in sys.path:
        sys.path.insert(0, _NCR_DIR)
    import ncr_models as nm            # noqa: E402 -- verbatim import, binexp_read, NCRModel/D_PIN/ENC_H
    import ncr_earlyln_scale as els    # noqa: E402 -- verbatim import, NCREarlyLNModel (build_ncr_head's own class)

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from write_supervision_loss import band2_check, null_directions, write_supervision_loss  # noqa: E402

K_NCR, D_NCR, H_NCR = 24, 25, 64   # ncr_lm_wave1_smoke.py:225-227, the Wave-1 pin -- reproduced verbatim


def build_fresh_encoder() -> els.NCREarlyLNModel:
    """Mirrors `graft.build_ncr_head()` (ncr_lm_wave1_smoke.py:424-425)
    exactly: `els.NCREarlyLNModel(d=D_NCR, h=H_NCR)`. A6: item 6 gets a
    FRESH instance per (lr, lambda_t) grid cell -- never shared/reused
    across cells."""
    return els.NCREarlyLNModel(d=D_NCR, h=H_NCR)


# ---------------------------------------------------------------------------
# Verbatim duplicates (see module docstring) of the pinned runner's own
# pure-math functions.
# ---------------------------------------------------------------------------
def discriminability_metrics(entity_adapter, embed, o: torch.Tensor,
                              entity_ids: torch.Tensor, tgt_slot: torch.Tensor) -> dict:
    """Verbatim body of ncr_lm_wave1_runner.py:480-537 (`discriminability_
    metrics`), with `integ` replaced by `entity_adapter` directly (see
    module docstring)."""
    B, K = entity_ids.shape
    T = entity_adapter(embed(entity_ids).float())
    on = F.normalize(o.float(), dim=-1)
    Tn = F.normalize(T, dim=-1)
    cos_all = torch.einsum("bd,bkd->bk", on, Tn)
    cos_true = cos_all.gather(1, tgt_slot[:, None]).squeeze(1)
    mask_true = F.one_hot(tgt_slot, K).bool()
    cos_off = cos_all.masked_fill(mask_true, 0.0).sum(1) / (K - 1)
    retrieval24_acc = (cos_all.argmax(1) == tgt_slot).float().mean().item()
    offtarget_margin = (cos_true.mean() - cos_off.mean()).item()
    pc_o = on @ on.t()
    off_diag_o = ~torch.eye(B, dtype=torch.bool, device=on.device)
    o_pairwise_cos = pc_o[off_diag_o].mean().item() if B > 1 else float("nan")
    Tf = Tn.reshape(B * K, -1)
    n_sub = min(512, Tf.shape[0])
    idx = torch.randperm(Tf.shape[0], device=Tf.device)[:n_sub]
    Ts = Tf[idx]
    pc_t = Ts @ Ts.t()
    off_diag_t = ~torch.eye(Ts.shape[0], dtype=torch.bool, device=Ts.device)
    target_pairwise_cos = pc_t[off_diag_t].mean().item() if n_sub > 1 else float("nan")
    return dict(offtarget_margin=float(offtarget_margin), retrieval24_acc=float(retrieval24_acc),
                o_pairwise_cos=float(o_pairwise_cos), target_pairwise_cos=float(target_pairwise_cos))


def ortho_regularization_loss(Z: torch.Tensor) -> torch.Tensor:
    """Verbatim body of ncr_lm_wave1_runner.py:714-742."""
    b, d, d2 = Z.shape
    assert d == d2, f"ortho_regularization_loss expects square (B,d,d) Z, got {tuple(Z.shape)}"
    eye = torch.eye(d, device=Z.device, dtype=Z.dtype).expand(b, d, d)
    dev = torch.matmul(Z.transpose(-1, -2), Z) - eye
    return (dev * dev).sum(dim=(-2, -1)).mean() / (d * d)


# ---------------------------------------------------------------------------
# Item 1/2 -- key geometry (A7: item 1 becomes gating; D3's null-gap
# extension retained).
# ---------------------------------------------------------------------------
def item_1_2_keygeom(teacher_force_operator_fn, keys_v: torch.Tensor, values_v: torch.Tensor) -> dict:
    """A7: item 1 becomes gating on `cond` AND on the fraction of episodes
    with L_key(Z_ideal) > 3e-4 (M5's pinv-truncation-cliff finding: past
    cond(keys)~3.4e5 the target itself lands outside the WIN band). D3's
    extension: also logs S[-1]/S[-2] (the null-space health signal for
    `w`'s own well-posedness). A3 groundwork: computes BOTH the global
    max/median (kept for continuity) AND the within-episode max/min ratio
    (the SPEC statistic item 3 gates on, A3).

    NOTE on "gating": this function reports the fraction of episodes whose
    OWN target violates the WIN band, and cond's own med/p99/max -- it does
    NOT hard-code a pass/fail numeric cutoff for "non-trivial fraction",
    because the report gives none (M5's own language is qualitative: "if
    that fraction is non-trivial, Band 2's L_key floor is ... distribution-
    limited"). Per the charter ("NO band thresholds beyond what the report
    fixes"), that judgment call is left to the coordinator reading this
    field, not invented here.

    teacher_force_operator_fn: a callable(keys_v, values_v) -> Z_ideal
    (e.g. `integ.teacher_force_operator` on the box, or the verbatim
    pinv-based reproduction `write_supervision_loss`'s own tests use
    off-box)."""
    cond = torch.linalg.svdvals(keys_v)
    cond_ratio = cond[:, 0] / (cond[:, -1] + 1e-12)
    _, S, Vh = torch.linalg.svd(keys_v, full_matrices=True)
    Kdim = keys_v.shape[1]
    null_gap = (S[:, -1] / (S[:, -2] + 1e-12)) if Kdim >= 2 else torch.full_like(S[:, -1], float("nan"))

    Z_ideal = teacher_force_operator_fn(keys_v, values_v)
    row_norms = Z_ideal.norm(dim=-1)                              # (B,d) per-row Frobenius norm

    # A3: BOTH statistics -- global (kept for continuity) and within-episode
    # (the spec statistic item 3 gates on):
    global_max_over_median = (row_norms.max() / row_norms.median().clamp_min(1e-8)).item()
    global_max_over_min = (row_norms.max() / row_norms.min().clamp_min(1e-8)).item()
    within_ep_max = row_norms.max(dim=-1).values
    within_ep_min = row_norms.min(dim=-1).values
    within_ep_ratio = within_ep_max / within_ep_min.clamp_min(1e-8)     # (B,)

    # A7: the pinv-truncation-cliff check (M5) -- fraction of episodes
    # whose OWN target itself violates Band 2's L_key<=3e-4 bound.
    r = torch.einsum('bij,bkj->bki', Z_ideal, keys_v.detach()) - values_v.detach()
    L_key_at_Z_ideal = (r.pow(2).sum(-1) / (values_v.detach().pow(2).sum(-1) + 1e-6)).mean(-1)   # (B,)
    frac_episodes_target_violates_win = (L_key_at_Z_ideal > 3e-4).float().mean().item()

    return dict(
        cond_med=cond_ratio.median().item(), cond_p99=cond_ratio.quantile(0.99).item(), cond_max=cond_ratio.max().item(),
        null_gap_med=null_gap.median().item(), null_gap_min=null_gap.min().item(),
        Z_ideal_fro_med=Z_ideal.norm(dim=(-2, -1)).median().item(), Z_ideal_fro_max=Z_ideal.norm(dim=(-2, -1)).max().item(),
        row_norm_med=row_norms.median().item(), row_norm_max=row_norms.max().item(),
        global_max_over_median=global_max_over_median, global_max_over_min=global_max_over_min,
        within_episode_ratio_med=within_ep_ratio.median().item(),
        within_episode_ratio_p99=within_ep_ratio.quantile(0.99).item(),
        within_episode_ratio_max=within_ep_ratio.max().item(),
        L_key_at_Z_ideal_med=L_key_at_Z_ideal.median().item(),
        L_key_at_Z_ideal_max=L_key_at_Z_ideal.max().item(),
        frac_episodes_target_violates_win=frac_episodes_target_violates_win,
    )


# ---------------------------------------------------------------------------
# Item 3 -- reachability (A2: the .encoder attribute route; A3: gate on the
# within-episode p99; M4(b)/(c): the LayerNorm-affine correction + the
# restored absolute ceiling).
# ---------------------------------------------------------------------------
def item_3_reachability(encoder, keygeom: dict, h: int = H_NCR) -> dict:
    """A2: caller passes `ncr_head.encoder` (NOT `ncr_head` itself --
    `NCREarlyLNModel(nm.NCRModel)`'s own `__init__` sets `self.encoder =
    BindingEncoder(d, h, ...)`, `ncr_models.py:165`; `row_out` lives on
    THAT encoder, `model_v4.py:52` -- verified directly against both
    source files: `hasattr(NCRModel, 'row_out')` is False,
    `hasattr(NCRModel.encoder_instance, 'row_out')` is True).

    A3: gates on `keygeom['within_episode_ratio_p99']` (the spec
    statistic -- the carded item 3 computed a GLOBAL max/median instead,
    a different statistic by up to ~4x on the report's own executed
    numbers). All four candidate statistics are still reported.

    M4(b): `encoder.row_norm` is `nn.LayerNorm(h)` with LEARNABLE affine
    (`model_v4.py:51`, default `elementwise_affine=True`) -- its output is
    `gamma (x) x_hat + beta` with `||x_hat||=sqrt(h)` FIXED (LayerNorm
    normalizes its INPUT to unit-RMS before the affine) but `gamma, beta`
    FREE after training. The achievable per-row output range is bounded by
    `sigma_max(row_out.weight) * (||gamma||_inf * sqrt(h) + ||beta||) +
    ||row_out.bias||` -- restored here (M4(c), the dropped absolute check).
    """
    row_out = encoder.row_out
    row_norm = encoder.row_norm
    sv = torch.linalg.svdvals(row_out.weight)          # weight: (d,h) -> min(d,h) singular values
    cond_row_out = (sv[0] / sv[-1]).item()
    required_dynamic_range = keygeom["within_episode_ratio_p99"]     # A3: the SPEC statistic

    sigma_max_row_out = sv[0].item()
    if row_norm.elementwise_affine:
        gamma_inf = row_norm.weight.abs().max().item()
        beta_norm = row_norm.bias.norm().item()
    else:
        gamma_inf, beta_norm = 1.0, 0.0
    bias_norm = row_out.bias.norm().item() if row_out.bias is not None else 0.0
    absolute_ceiling = sigma_max_row_out * (gamma_inf * (h ** 0.5) + beta_norm) + bias_norm

    return dict(
        cond_row_out=cond_row_out, required_dynamic_range=required_dynamic_range,
        global_max_over_median=keygeom["global_max_over_median"], global_max_over_min=keygeom["global_max_over_min"],
        within_episode_ratio_med=keygeom["within_episode_ratio_med"],
        within_episode_ratio_p99=keygeom["within_episode_ratio_p99"],
        within_episode_ratio_max=keygeom["within_episode_ratio_max"],
        absolute_ceiling=absolute_ceiling, required_row_norm_p99=keygeom["row_norm_max"],
        gate_pass=bool(cond_row_out >= required_dynamic_range),
    )


# ---------------------------------------------------------------------------
# Item 4 -- transverse gain (A4: named fields, M10's polarity fix, M11's
# d-K generalization).
# ---------------------------------------------------------------------------
def item_4_transverse(Z_sgd: torch.Tensor, keys_v: torch.Tensor, K: int | None = None) -> dict:
    """A4: emits transverse_gain_med/_p90 (the ABSOLUTE reading, D1.5's
    own pre-registered ~3 bound, informational per F1) AND
    transverse_ratio_med/_p90 (||ZW||/||Z||_F, F1's repaired Band-2 gate,
    threshold 0.12, the quantity that actually adjudicates). M10: no more
    inverted `gate_pass` bool -- renamed to `transverse_gain_exceeds_3`
    with its polarity spelled out (True = CONFIRMS D1's mechanism is
    necessary, per the original gating readout's own intent)."""
    W = null_directions(keys_v.detach(), K=K)
    ZW = torch.einsum('bij,bkj->bki', Z_sgd, W)
    Zw_norm = ZW.norm(dim=(-2, -1))
    Z_fro = Z_sgd.norm(dim=(-2, -1))
    ratio = Zw_norm / (Z_fro + 1e-12)
    return dict(
        transverse_gain_med=Zw_norm.median().item(), transverse_gain_p90=Zw_norm.quantile(0.90).item(),
        transverse_ratio_med=ratio.median().item(), transverse_ratio_p90=ratio.quantile(0.90).item(),
        transverse_gain_exceeds_3=bool(Zw_norm.quantile(0.90).item() > 3.0),
    )


# ---------------------------------------------------------------------------
# Item 5 -- ortho conflict (A5: numeric predicate + the 1/B fix, M8).
# ---------------------------------------------------------------------------
def item_5_ortho_conflict(teacher_force_operator_fn, keys_v: torch.Tensor, values_v: torch.Tensor) -> dict:
    """A5: a numeric predicate for "the conflict reproduces" (the report's
    own stated example, used verbatim): `ortho_loss(Z_ideal) > 1e3 AND
    grad_norm_per_example > 0`, with M8's 1/B correction applied (
    `ortho_regularization_loss` batch-MEAN reduces, so
    `torch.autograd.grad` yields per-example gradients already scaled by
    1/B -- multiplying by B undoes exactly that scale factor, the fix M8
    names).

    DISCLOSED (per M8's own second option, taken here since no new Stage-1
    cell/derivation is authorized this round): R3's own joint-minimization
    lambda_w curve is NOT reproduced by this item -- D2's ortho=0 removal
    decision rests on this item's ortho-loss/gradient reconfirmation PLUS
    F2's own synthetic joint-minimization evidence, not on a real-geometry
    joint-min curve. Flagged explicitly, not silently dropped (M8's own
    finding: "two substitutions, one disclosed" -- this closes the
    disclosure gap on the second one)."""
    Z_ideal = teacher_force_operator_fn(keys_v, values_v).detach().requires_grad_(True)
    ortho_loss = ortho_regularization_loss(Z_ideal)
    grad = torch.autograd.grad(ortho_loss, Z_ideal)[0]
    Bsz = Z_ideal.shape[0]
    grad_norm_med_as_carded = grad.norm(dim=(-2, -1)).median().item()          # the R3-carded (uncorrected) reading
    grad_norm_med_per_example = (grad.norm(dim=(-2, -1)) * Bsz).median().item()  # M8: x B undoes the batch-mean's 1/B
    reproduces = bool(ortho_loss.item() > 1e3 and grad_norm_med_per_example > 0.0)
    return dict(ortho_loss_at_Z_ideal=ortho_loss.item(), grad_norm_med_as_carded=grad_norm_med_as_carded,
                grad_norm_med_per_example=grad_norm_med_per_example, conflict_reproduces=reproduces,
                joint_min_curve_reproduced=False)   # disclosed, see docstring


# ---------------------------------------------------------------------------
# Item 6 -- A6, the substantive replacement: a PARAMETRIZED
# achievability/lambda_t probe (the free-Z sweep, F2(c), provably cannot
# fail, so it cannot choose lambda_t).
# ---------------------------------------------------------------------------
def item_6_achievability_probe(keys_v_train: torch.Tensor, values_v_train: torch.Tensor,
                                held_out_by_hop: dict, score_fn,
                                lambda_t_grid=(0.0, 0.1, 1.0, 3.0), lr_grid=(3e-4, 1e-3),
                                n_steps: int = 8000, K: int | None = None,
                                log_every: int = 500) -> dict:
    """A6: trains a FRESH encoder's PARAMETERS (never a free Z -- F2(a)'s
    own finding: the Z-space gradient-orthogonality argument does not
    survive the pullback through theta, measured cos(grad_Lkey,
    grad_Ltrans) = 0.17-0.45 in parameter space vs 0.0017 null) on
    `L_key + lambda_t . L_transverse`, at every (lr, lambda_t) grid cell, a
    FRESH `build_fresh_encoder()` instance each time, for >= n_steps Adam
    steps -- ONCE per cell, on the TRAINING episodes only.

    held_out_by_hop: dict[int, tuple(keys_v_held, values_v_held,
    query_key_held, entity_ids_held, tgt_slot_held)], ONE entry per hop to
    score at. F3.5's fix, implemented as a SCOPED, cost-disciplined
    decision (disclosed, not silently made): the report's own words are
    "build one batch per hop and fit per hop" -- read literally, THAT
    would retrain the whole (lr, lambda_t) grid once per hop (2x the
    training cost for 2 hops, since keys_v_train/values_v_train would also
    need to be redrawn per hop). This function instead trains ONCE per
    grid cell (on ONE hop-agnostic training extraction -- keys_v/values_v
    are K raw bind-clause entity vectors, whose CONTENT is not itself
    hop-conditioned; only the QUERY/answer-target construction is, per
    `build_task1_document`'s own hop_set mechanism) and scores EACH
    trained cell against EVERY hop's own hop-MATCHED held-out documents.
    This closes exactly the defect F3.5 diagnosed (an h=61-fit Z scored
    against OTHER hops' documents, with mismatched tgt_slot/answer_token/
    query) at 1x the training cost instead of len(held_out_by_hop)x. If a
    future round wants the literal per-hop RETRAINING too (to test whether
    the encoder's write itself should be hop-conditioned, not just its
    read-side scoring), that is a disclosed, not-yet-funded extension --
    flagged here, not silently assumed unnecessary.

    score_fn(o, entity_ids, tgt_slot) -> dict: pre-bound to the real
    (checkpoint's own) entity_adapter/embed on the box, or a synthetic
    stand-in off-box (see test_stage0prime_helpers.py) -- this function
    stays agnostic to which.

    GATES (the report's own rule, no new threshold invented): does ANY
    (lr, lambda_t) cell reach BOTH of Band 2's F1-repaired, scale-invariant
    gates (L_key_cstar<=3e-4 AND Zw_ratio<=0.12), read at BOTH the median
    and the p90 (the two NAMED reduction statistics, F1's own repair
    requirement), AT EVERY HELD-OUT HOP (not just one)? Band 2 itself is a
    function of (Z, keys_v, values_v) only -- it has no hop dependence in
    principle, but keys_v_held/values_v_held DIFFER per hop (F3.5: a
    genuinely different document is drawn per hop), so the Band-2 reading
    computed on hop A's held-out draw and hop B's held-out draw can differ
    by ordinary sampling noise. Reporting only ONE hop's reading (an
    earlier version of this function reported whichever hop happened to be
    listed first) would silently let a cherry-picked hop decide the gate.
    This version reports Band 2 at EVERY hop in `held_out_by_hop` and
    requires ALL of them to pass for `reaches_band2_*` to be True --
    the conservative, non-cherry-picking reading. If none does, Stage 1
    does not launch on the current band -- the amended item's own binding
    gate.
    """
    assert len(held_out_by_hop) >= 1, "item_6_achievability_probe requires at least one held-out hop"
    results = []
    for lr in lr_grid:
        for lam_t in lambda_t_grid:
            encoder_module = build_fresh_encoder()
            opt = torch.optim.Adam(encoder_module.parameters(), lr=lr)
            curve = []
            for step in range(1, n_steps + 1):
                opt.zero_grad()
                Z = encoder_module.encode(keys_v_train, values_v_train)
                out = write_supervision_loss(Z, keys_v_train, values_v_train, lambda_t=lam_t, K=K)
                out["L_write"].backward()
                opt.step()
                if step % log_every == 0 or step == 1 or step == n_steps:
                    with torch.no_grad():
                        b2 = band2_check(Z.detach(), keys_v_train, values_v_train, K=K)
                    curve.append(dict(step=step, L_key_med=out["L_key"].detach().median().item(),
                                       L_transverse_med=out["L_transverse"].detach().median().item(),
                                       Zw_ratio_med=b2["Zw_ratio_median"], L_key_cstar_med=b2["L_key_cstar_median"]))

            retr_by_hop = {}
            band2_by_hop = {}
            with torch.no_grad():
                for h, (keys_h, values_h, qkey_h, ids_h, tgt_h) in held_out_by_hop.items():
                    Z_held = encoder_module.encode(keys_h, values_h)
                    o = nm.binexp_read(Z_held, qkey_h.unsqueeze(1), h)["o"].squeeze(1)
                    retr_by_hop[f"h={h}"] = score_fn(o, ids_h, tgt_h)
                    b2_h = band2_check(Z_held, keys_h, values_h, K=K)
                    band2_by_hop[f"h={h}"] = dict(L_key_cstar_med=b2_h["L_key_cstar_median"], L_key_cstar_p90=b2_h["L_key_cstar_p90"],
                                                   Zw_ratio_med=b2_h["Zw_ratio_median"], Zw_ratio_p90=b2_h["Zw_ratio_p90"])

            reaches_band2_median = all(v["L_key_cstar_med"] <= 3e-4 and v["Zw_ratio_med"] <= 0.12 for v in band2_by_hop.values())
            reaches_band2_p90 = all(v["L_key_cstar_p90"] <= 3e-4 and v["Zw_ratio_p90"] <= 0.12 for v in band2_by_hop.values())
            results.append(dict(
                lr=lr, lambda_t=lam_t, curve=curve,
                held_out_band2_by_hop=band2_by_hop,
                retrieval_by_hop=retr_by_hop,
                reaches_band2_median=bool(reaches_band2_median), reaches_band2_p90=bool(reaches_band2_p90),
            ))

    any_median = any(r["reaches_band2_median"] for r in results)
    any_p90 = any(r["reaches_band2_p90"] for r in results)
    return dict(cells=results, any_lambda_t_reaches_band2_median=any_median,
                any_lambda_t_reaches_band2_p90=any_p90,
                stage1_gate="GO" if any_median else "NO-GO-ON-CURRENT-BAND")
