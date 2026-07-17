"""NCR REAL-LM WAVE-1 BUILD -- Phase-0 + Task-1 harness (K=24, d_ncr=25, 98M
DeltaNet backbone). matrix-thinking/NCR_REAL_LM_DESIGN.md sec N2 (GATE-1
AMENDMENT, PI-ratified 2026-07-17) + sec N2.8 (COORDINATOR ADJUDICATION,
"WAVE 1 (NOW): Phase-0 + Task-1 only"). This is the FIRST-EVER attempt to
run the NCR head inside a real LM (previously only the isolated synthetic
Task-E harness, NCR_ORTHO_WRITE.md).

============================================================================
WHAT IS SPEC-FAITHFUL BELOW (verified against the frozen+amended design
text, cited inline) vs. WHAT IS AN EXPLICIT, LABELED PLACEHOLDER (the
design text itself does not pin it -- see the BUILD RECORD appended to
NCR_REAL_LM_DESIGN.md, "sec G3-B1", for the full gap analysis):
============================================================================

SPEC-FAITHFUL (built exactly, reused verbatim, no invention):
  - Backbone: DeltaNetLM(vocab_size=50257, d_model=768, d_state=64,
    n_layers=12, conv_size=4, num_heads=1, ffn_mult=4) -- the rung-1 98M
    config, matrix-thinking/deltanet_rd/lm_rd_rung_configs.py RUNGS[1],
    verified-on-box 97,618,176 params (FROZEN_BIAS_LM_DESIGN.md sec 13.7's
    measured-rate anchor: 0.236 s/step at batch=32/seq=512).
  - NCR head: ncr_earlyln_scale.NCREarlyLNModel(d=25, h=64) -- the FREE-WRITE
    arm ONLY (no Newton-Schulz orthogonal projection anywhere -- NCR_
    ORTHO_WRITE.md's NS-polar pipeline is RETIRED from this build per sec
    N2.1's GATE-1 amendment). K=24/d_ncr=25 is sec N2.2(a)'s re-derivation
    ("d_ncr: 33 -> 25... 24+1=25 is the SAME formula at the amended K").
    Param count re-derived exactly at sec N2.2(c): P(25,64) = 40*64^2 +
    4*25*64 + 46*64 + 25 = 173,209 -- asserted EXACTLY below, not a
    tolerance band.
  - Read: ncr_models.binexp_read (module-level pure function, unmodified)
    -- O(log h) repeated squaring, exact by construction (sec N2.3).

EXPLICIT PLACEHOLDER (class PlaceholderIntegrationGlue below) -- NOT part
of the frozen spec, built ONLY to prove shapes/dtypes/gradients/memory
co-residency are mechanically viable, NEVER to be read as a build-time
decision the coordinator has already made:
  (1) Write-side adapter (backbone hidden dim 768 -> d_ncr=25 key/value
      vectors). NCR_REAL_LM_DESIGN.md sec 2.2 (line 388-389) pins WHERE the
      taps come from ("a side-channel module attached after the backbone's
      final layer, reading the backbone's own hidden states at bind-clause
      /query positions") but not the projection architecture/dims. This
      placeholder uses two independent nn.Linear(768, 25, bias=False) maps
      (key_adapter, value_adapter), the same shape convention as this
      repo's OWN established tap-adapter pattern (matrix-thinking/
      deltanet_rd/probe_head_rd.py build_adapter_arm) -- a precedented
      interpolation, NOT text from NCR_REAL_LM_DESIGN.md itself.
  (2) Read injection. NCR_REAL_LM_DESIGN.md sec 2.1 (line 284-288) states
      VERBATIM: "Either read injects its output o into the residual stream
      at the query position (e.g. added to the Transformer's own hidden
      state before the final LM head, or consumed by a small MLP that
      produces logits directly for the query token -- build-time decision,
      not resolved here)." This placeholder picks the FIRST disclosed
      option (additive residual, via a learned nn.Linear(25, 768,
      bias=False) up-projection) purely for smoke purposes.
  (3) Training loss / recovered_frac@0.9 scoring protocol against real
      text. Not pinned anywhere in NCR_REAL_LM_DESIGN.md. This placeholder
      borrows the H2H Demo's OWN established convention verbatim in spirit
      (matrix-thinking/deltanet_rd/probe_head_rd.py: build_probe_target_
      table / cosine_recovery_frac / probe_aux_loss / joint_loss,
      AUX_WEIGHT_DEFAULT=0.1) -- a frozen random-unit-row per-entity target
      table, cosine aux loss summed with ordinary LM cross-entropy. This is
      a REASONABLE reuse of an existing, audited pattern in this exact
      codebase, but NCR_REAL_LM_DESIGN.md never cites it for Task 1 and no
      weight/schedule is pinned -- flagged, not assumed authoritative.

fp32 boundary (spec-faithful): sec 8 item 2 / m1's fix ("the NCR head casts
backbone hidden states to fp32 at its own boundary") -- PlaceholderIntegration
Glue casts every extracted tap to fp32 before it touches the NCR head,
matching this already-ratified fix.

Run (box only -- chunk_delta_rule has no CPU path, matches this repo's own
smoke() convention throughout matrix-thinking/deltanet_rd):
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


# ---------------------------------------------------------------------------
# EXPLICIT PLACEHOLDER glue -- see module docstring. NOT spec-derived.
# ---------------------------------------------------------------------------
class PlaceholderIntegrationGlue(nn.Module):
    """ONE reasonable, precedented instantiation of the three integration
    decisions NCR_REAL_LM_DESIGN.md leaves open (module docstring items
    1-3). Exists to smoke-test mechanical viability ONLY -- gradients,
    dtypes, memory co-residency. The coordinator has NOT ratified this as
    the Phase-0/Phase-1 architecture."""

    def __init__(self, d_model: int, d_ncr: int, vocab_size: int):
        super().__init__()
        self.key_adapter = nn.Linear(d_model, d_ncr, bias=False)      # placeholder (1)
        self.value_adapter = nn.Linear(d_model, d_ncr, bias=False)    # placeholder (1)
        self.read_inject = nn.Linear(d_ncr, d_model, bias=False)      # placeholder (2)
        # placeholder (3): frozen random-unit-row per-entity target table,
        # mirrors probe_head_rd.py's build_probe_target_table exactly in spirit.
        target = torch.randn(vocab_size, d_ncr)
        target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.register_buffer("target_table", target, persistent=False)

    def extract_kv(self, hidden: torch.Tensor, key_pos: torch.Tensor, val_pos: torch.Tensor):
        """hidden: (B,T,d_model) backbone final hidden states (sec 2.2's own
        'after the backbone's final layer' tap point). key_pos/val_pos:
        (B,K) integer token positions per bind clause. Returns (keys,
        values): (B,K,d_ncr), fp32 at the NCR head's own boundary (sec 8
        item 2 / m1's fix, spec-faithful)."""
        B, K = key_pos.shape
        idx_k = key_pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        idx_v = val_pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        h_key = torch.gather(hidden, 1, idx_k).float()
        h_val = torch.gather(hidden, 1, idx_v).float()
        return self.key_adapter(h_key), self.value_adapter(h_val)

    def inject(self, hidden: torch.Tensor, o: torch.Tensor, query_pos: torch.Tensor) -> torch.Tensor:
        """Additive-residual placeholder (sec 2.1's first disclosed option).
        hidden: (B,T,d_model); o: (B,Q,d_ncr) NCR read output; query_pos:
        (B,Q) integer positions. Returns hidden with o injected at those
        positions."""
        bias = self.read_inject(o.to(hidden.dtype))                    # (B,Q,d_model)
        out = hidden.clone()
        idx = query_pos.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        out.scatter_add_(1, idx, bias)
        return out

    def aux_loss(self, pred: torch.Tensor, entity_ids: torch.Tensor) -> torch.Tensor:
        target = self.target_table[entity_ids]
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_backbone() -> DeltaNetLM:
    return DeltaNetLM(VOCAB_SIZE, **RUNG1_BACKBONE)


def build_ncr_head() -> els.NCREarlyLNModel:
    return els.NCREarlyLNModel(d=D_NCR, h=H_NCR)


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
    del backbone, ncr
    return backbone_ok and ncr_ok


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


def smoke_7_co_residency_joint_step(device: str):
    """PLACEHOLDER-glue smoke (module docstring): both modules + glue on ONE
    device, ONE optimizer, ONE joint forward-backward step. Proves memory
    co-residency and gradient flow through BOTH modules simultaneously --
    the actual make-or-break mechanical question for this wave. Uses a
    Phase-0-scale (not Phase-1-scale) shape to keep the contended-GPU
    footprint small (CLAUDE.md: 'brief GPU slot... do NOT kill production
    jobs')."""
    torch.manual_seed(6)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    backbone = build_backbone().to(device)
    ncr = build_ncr_head().to(device)
    glue = PlaceholderIntegrationGlue(RUNG1_BACKBONE["d_model"], D_NCR, VOCAB_SIZE).to(device)
    opt = torch.optim.Adam(
        list(backbone.parameters()) + list(ncr.parameters()) + list(glue.parameters()), lr=1e-4)

    B, T, K = 4, _MIN_KERNEL_T, K_NCR
    token_ids = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    lm_target = torch.randint(0, VOCAB_SIZE, (B, T), device=device)
    key_pos = torch.randint(0, T, (B, K), device=device)
    val_pos = torch.randint(0, T, (B, K), device=device)
    query_pos = torch.randint(0, T, (B, 3), device=device)
    entity_ids = torch.randint(0, VOCAB_SIZE, (B, K), device=device)      # one id per K-cycle relation

    hidden = backbone(token_ids, return_hidden=True)                     # (B,T,768), sec 2.2 tap point
    keys_v, values_v = glue.extract_kv(hidden, key_pos, val_pos)         # placeholder (1), fp32
    ncr_batch = dict(keys=keys_v, values=values_v, query_keys=keys_v,
                      hops=torch.ones(B, K, dtype=torch.long, device=device))
    ncr_pred, Z = ncr(ncr_batch)                                          # ncr_pred: (B,K,D_NCR)
    o = nm.binexp_read(Z, keys_v[:, :3, :], h=3)["o"]                     # O(log h) exact read, 3 queries
    hidden_injected = glue.inject(hidden, o, query_pos)                   # placeholder (2)
    logits = F.linear(hidden_injected, backbone.embed.weight)             # tied head, matches DeltaNetLM

    loss_ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), lm_target.reshape(-1))
    loss_aux = glue.aux_loss(ncr_pred, entity_ids)                        # (B,K,D_NCR) vs (B,K) ids
    loss = loss_ce + 0.1 * loss_aux                                       # AUX_WEIGHT_DEFAULT precedent

    opt.zero_grad()
    loss.backward()

    all_params = list(backbone.parameters()) + list(ncr.parameters()) + list(glue.parameters())
    grads_finite = all(torch.isfinite(p.grad).all() for p in all_params if p.grad is not None)
    n_with_grad = sum(1 for p in all_params if p.grad is not None)
    opt.step()

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device == "cuda" else -1.0
    ok = (torch.isfinite(loss).item() and grads_finite and n_with_grad > 0)
    _report("smoke 7 (PLACEHOLDER GLUE): backbone + NCR head co-resident, ONE joint fwd/bwd/opt-step, "
            "finite grads through BOTH modules", ok,
            f"loss_ce={loss_ce.item():.4f} loss_aux={loss_aux.item():.4f} "
            f"n_params_with_grad={n_with_grad} peak_mem_allocated={peak_gb:.2f} GB")
    del backbone, ncr, glue, opt, hidden, logits, loss
    torch.cuda.empty_cache() if device == "cuda" else None
    return peak_gb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--out", default=None, help="write JSON results here")
    ap.add_argument("--smoke", action="store_true",
                     help="no-op: this entire script is a smoke suite (no training loop, no "
                          "optimizer state persisted past the process) -- the flag exists so "
                          "invocations self-document as smoke tests, matching this repo's own "
                          "pre-train-gate hook convention (.claude/hooks/pre-train-gate.sh's "
                          "--smoke/--dry-run/--test/--quick exemption).")
    args = ap.parse_args()

    print("=" * 70)
    print("NCR REAL-LM WAVE-1 BUILD SMOKE -- Phase-0 + Task-1 harness")
    print(f"device={args.device} torch={torch.__version__} "
          f"cuda_available={torch.cuda.is_available()}")
    if args.device == "cuda":
        assert torch.cuda.is_available(), "cuda requested but not available"
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print("=" * 70)

    t0 = time.time()
    smoke_0_param_counts()
    if args.device == "cuda":
        smoke_1_backbone_forward_backward(args.device)
        smoke_2_backbone_checkpoint_resume(args.device)
        smoke_3_backbone_eval_batch(args.device)
        smoke_4_ncr_head_forward_backward(args.device)
        smoke_5_ncr_head_checkpoint_resume(args.device)
        smoke_6_ncr_binexp_read_ladder(args.device)
        peak_gb = smoke_7_co_residency_joint_step(args.device)
        RESULTS["_co_residency_peak_mem_gb"] = peak_gb
    else:
        print("device=cpu: chunk_delta_rule has no CPU path (this repo's own standing finding) -- "
              "skipping items 1-7, param-count-only run.")
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
