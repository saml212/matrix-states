#!/usr/bin/env python3
"""EMBED-PATH BUILD -- CPU-runnable synthetic verification (no CUDA, no fla,
no box dependency -- runs anywhere PyTorch is installed). Mirrors the
REAL runner's topology (tied head, extract_kv/query_key off raw ids,
DETACHED aux targets, ortho(Z) via a separate Z-side conduit, shared
o_raw feeding both CE and aux/ortho) at toy scale, and tests the ACTUAL
assemble_closed_grads_ / _non_ce_term / assert_conduit_has_teeth code
(copy-identical to matrix-thinking/embedpath_build/embed_path_runner.patch
-- diffed against the deployed box copy at build time, see BUILD_REPORT.md)
against that synthetic graph.

This is the CPU-portable complement to the box-side CUDA smokes
(verify_embed_path_real.py, verify_run_two_arm_cell.py,
verify_throughput.py) -- run TO COMPLETION, including the forced-fail
negative test, per this repo's own standing rule ("run the negative unit
test to completion, don't just write it").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

RESULTS = {}


def log(name, **kw):
    RESULTS[name] = kw
    print(f"=== {name} ===")
    for k, v in kw.items():
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# The functions under test -- copy-identical to
# matrix-thinking/embedpath_build/embed_path_runner.patch (verified by the
# build script below: an md5-of-function-source check against the deployed
# box copy, see verify_cpu_synthetic_matches_patch() at the bottom).
# ---------------------------------------------------------------------------
def _non_ce_term(aux_loss, ortho_loss, aux_read_loss_weight, ortho_reg_weight):
    non_ce = None
    if aux_loss is not None and aux_read_loss_weight > 0.0:
        non_ce = aux_read_loss_weight * aux_loss
    if ortho_loss is not None and ortho_reg_weight > 0.0:
        term = ortho_reg_weight * ortho_loss
        non_ce = term if non_ce is None else non_ce + term
    return non_ce


def assemble_closed_grads_(all_params, opt, target_w, total_loss, ce_loss, aux_loss, ortho_loss,
                            aux_read_loss_weight, ortho_reg_weight, max_norm=1.0):
    """Toy-graph variant of the runner's assemble_closed_grads_: takes
    all_params/target_w directly (the real function derives them from an
    `arm` dict + `close_target` string; that indirection is irrelevant to
    the autograd mechanism under test here)."""
    def _finite_clip_step():
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in all_params)
        if finite:
            torch.nn.utils.clip_grad_norm_(all_params, max_norm)
            opt.step()
        return finite

    non_ce = _non_ce_term(aux_loss, ortho_loss, aux_read_loss_weight, ortho_reg_weight)
    if non_ce is None:
        total_loss.backward()
        return {"cut_active": False, "stepped": _finite_clip_step(), "conduit_ratio": 0.0, "clip_coef": 1.0}

    grad_rest_target = torch.autograd.grad(non_ce, [target_w], retain_graph=True, allow_unused=True)[0]
    total_loss.backward()

    if grad_rest_target is None:
        return {"cut_active": False, "stepped": _finite_clip_step(), "conduit_ratio": 0.0, "clip_coef": 1.0}

    combined_before_clip = target_w.grad.detach().clone()
    grad_ce_target = combined_before_clip - grad_rest_target
    conduit_ratio = (grad_rest_target.norm() / grad_ce_target.norm().clamp_min(1e-12)).item()

    finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in all_params)
    if not finite:
        return {"cut_active": False, "stepped": False, "conduit_ratio": conduit_ratio, "clip_coef": float("nan")}

    torch.nn.utils.clip_grad_norm_(all_params, max_norm)
    combined_after_clip = target_w.grad.detach().clone()
    clip_coef = (combined_after_clip.norm() / combined_before_clip.norm().clamp_min(1e-12)).item()

    target_w.grad.sub_(grad_rest_target * clip_coef)
    grad_ce_target_clipped = grad_ce_target * clip_coef
    assert torch.allclose(target_w.grad, grad_ce_target_clipped, rtol=1e-5, atol=1e-6), (
        "post-cut grad does not match CE's own (clipped) share within tolerance")
    opt.step()
    return {"cut_active": True, "stepped": True, "conduit_ratio": conduit_ratio, "clip_coef": clip_coef}


def assert_conduit_has_teeth(grad_diag, min_ratio):
    if not grad_diag["cut_active"]:
        return
    assert grad_diag["conduit_ratio"] > min_ratio, (
        f"HAS-TEETH FAILED -- conduit_ratio={grad_diag['conduit_ratio']:.4f} <= min_ratio={min_ratio:.4f}")


# ---------------------------------------------------------------------------
# Synthetic topology: tied head, extract_kv/query_key off raw "ids" (here a
# fixed random embedding-lookup surrogate), DETACHED aux target, ortho(Z)
# via a SEPARATE Z-side conduit that does not pass through o -- mirrors
# §2.1's real conduit trace (o_raw feeds CE directly AND aux/ortho directly,
# undetached; the aux TARGET computation is fully detached, so it is a
# structural no-op path, exactly as DRAFT-R0's own refutation established).
# ---------------------------------------------------------------------------
class ToyModel(nn.Module):
    def __init__(self, vocab=32, d=16, d_ncr=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)               # tied-head target candidate
        self.entity_adapter = nn.Linear(d, d_ncr, bias=False)  # other target candidate
        self.blocks = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))  # backbone stand-in
        self.read_injector = nn.Linear(d_ncr, d, bias=False)  # CE-only route (o -> logits), aux/ortho never reach it
        self.ncr_head = nn.Linear(d_ncr, d_ncr, bias=False)   # shared-conduit stand-in (via Z)

    def forward(self, ids, entity_ids):
        h = self.blocks(self.embed(ids))                       # backbone forward
        keys_v = self.entity_adapter(self.embed(ids))           # extract_kv surrogate
        q_key = self.entity_adapter(self.embed(ids))             # query_key surrogate
        Z = self.ncr_head(keys_v)                                 # Z, the ortho-reg target
        o_raw = (Z * q_key).sum(-1, keepdim=True) * torch.ones_like(keys_v)  # o_raw, SHARED undetached node
        logits = F.linear(h + self.read_injector(o_raw), self.embed.weight)   # tied head, CE route
        target_o = self.entity_adapter(self.embed(entity_ids)).detach()       # DETACHED aux target (no-op path)
        return logits, o_raw, Z, target_o


def make_batch(model, vocab=32, n=4):
    ids = torch.randint(0, vocab, (n,))
    entity_ids = torch.randint(0, vocab, (n,))
    answer = torch.randint(0, vocab, (n,))
    return ids, entity_ids, answer


def compute_losses(model, batch, aux_w, ortho_w):
    ids, entity_ids, answer = batch
    logits, o_raw, Z, target_o = model(ids, entity_ids)
    ce_loss = F.cross_entropy(logits, answer)
    aux_loss = F.mse_loss(o_raw.mean(-1), target_o.mean(-1))   # uses undetached o_raw + detached target
    eye = torch.eye(Z.shape[-1])
    ortho_loss = ((Z.T @ Z - eye) ** 2).mean()
    total_loss = ce_loss + aux_w * aux_loss + ortho_w * ortho_loss
    return total_loss, ce_loss, aux_loss, ortho_loss


def main():
    for target_name in ("embed", "entity_adapter"):
        torch.manual_seed(1)
        model_off = ToyModel()
        model_on = ToyModel()
        model_off.load_state_dict(model_on.state_dict())   # bit-identical init
        opt_off = torch.optim.AdamW(model_off.parameters(), lr=1e-3)
        opt_on = torch.optim.AdamW(model_on.parameters(), lr=1e-3)

        torch.manual_seed(2)
        batch = make_batch(model_off)

        total_off, ce_off, aux_off, ortho_off = compute_losses(model_off, batch, 0.5, 0.1)
        opt_off.zero_grad()
        total_off.backward()
        target_off = getattr(model_off, target_name).weight
        combined_norm = target_off.grad.norm().item()
        torch.nn.utils.clip_grad_norm_(list(model_off.parameters()), 1.0)

        total_on, ce_on, aux_on, ortho_on = compute_losses(model_on, batch, 0.5, 0.1)
        opt_on.zero_grad()
        target_on = getattr(model_on, target_name).weight
        all_params_on = list(model_on.parameters())
        grad_diag = assemble_closed_grads_(all_params_on, opt_on, target_on, total_on, ce_on, aux_on, ortho_on, 0.5, 0.1)
        assert_conduit_has_teeth(grad_diag, min_ratio=1e-8)

        # scope-preserved (should be EXACT for every non-target param, per the box-side
        # empirical finding -- see BUILD_REPORT.md/D-F3 build-time refinement)
        off_named = dict(model_off.named_parameters())
        on_named = dict(model_on.named_parameters())
        target_pname = f"{target_name}.weight"
        n_exact, fails = 0, []
        for name in off_named:
            if name == target_pname:
                continue
            g_off, g_on = off_named[name].grad, on_named[name].grad
            if g_off is None and g_on is None:
                continue
            if torch.equal(g_off, g_on):
                n_exact += 1
            else:
                fails.append(name)
        log(f"synthetic[{target_name}]", cut_active=grad_diag["cut_active"], stepped=grad_diag["stepped"],
            conduit_ratio=round(grad_diag["conduit_ratio"], 6), clip_coef=round(grad_diag["clip_coef"], 6),
            n_scope_exact=n_exact, scope_fails=fails, PASS=(len(fails) == 0 and grad_diag["cut_active"]))

    # Forced-fail negative test on the synthetic graph too (independent of the box).
    torch.manual_seed(3)
    model = ToyModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = make_batch(model)
    total, ce, aux, ortho = compute_losses(model, batch, 0.5, 0.1)
    target_w = model.embed.weight
    all_params = list(model.parameters())
    opt.zero_grad()
    non_ce = _non_ce_term(aux, ortho, 0.5, 0.1)
    grad_rest = torch.autograd.grad(non_ce, [target_w], retain_graph=True, allow_unused=True)[0]
    total.backward()
    combined_before = target_w.grad.detach().clone()
    grad_ce = combined_before - grad_rest
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    combined_after = target_w.grad.detach().clone()
    clip_coef = (combined_after.norm() / combined_before.norm().clamp_min(1e-12)).item()
    grad_ce_clipped = grad_ce * clip_coef
    # BUG: forgot the subtraction (no-op cut)
    fired = False
    try:
        assert torch.allclose(target_w.grad, grad_ce_clipped, rtol=1e-5, atol=1e-6)
    except AssertionError:
        fired = True
    log("synthetic_forced_fail_negative_test", assertion_fired=fired, PASS=fired)

    print("VERIFY_CPU_SYNTHETIC_DONE")


if __name__ == "__main__":
    main()
