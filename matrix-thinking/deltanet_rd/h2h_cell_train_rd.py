"""h2h_cell_train_rd.py -- DEPLOY-STAGE WIRING for HEAD_TO_HEAD_DEMO_DESIGN.md
sec 1.19 items (3)-(6): the REAL train/eval callables the build stage
deliberately dependency-injected (h2h_calibration_wrappers_rd.py /
h2h_sweep_runner_rd.py / h2h_msweep_fanout_rd.py all take `train_fn`/
`eval_fn` params; this module supplies them and the per-cell CLI the chain
script drives).

WHAT THIS FILE OWNS (and the build-stage files deliberately do NOT):
  - build_arm_model: the three rung-1 arms at their PINNED configs
    (contender = DeltaNetLM frozen-bias per_token lambda=0.58, sec 1.2;
    ablation = AblationLM d_state-pinned, sec 1.3(a); transformer =
    TransformerLM n_layers=2 R3-F3 pin, sec 1.3(b)).
  - train_grammar_cell: tasks 1/2 (grammar_rd episodes) joint training,
    sec 1.3.1.3's exact loss (loss_CE + aux_weight * mean(1 - cos(pred,
    target)); aux_weight REVISED 0.1 -> 2.0 at calibration, see AUX_WEIGHT),
    probe head trained jointly from step 0, recovered_frac@0.9 learning
    curves logged every EVAL_EVERY steps (axis 1, sec 1.4.1) alongside
    probe_cos_mean (pre-threshold diagnosability addition), held-out-hop
    eval for task 2, aux-vs-CE tap-point gradient-norm ratio measured at
    step 500 (sec 1.3.1.3's aux_weight revision trigger -- MEASURED AND
    REPORTED here, never auto-revised; freezing is the coordinator's
    margin-freeze decision) plus an aux-swamps-CE overshoot guard.
  - train_lm_cell: task 3 (static corpora) arch-generic LM training,
    mirroring lm_pretrain_rd.train()'s own regime (fp32 + set_and_log_tf32,
    AdamW, cosine schedule, grad-clip 1.0, non-finite-grad skip, FIX-1
    fixed-seed eval windows) without that function's DeltaNetLM-only
    diagnostics.
  - Timing pilots (sec 1.7 gate 2): --pilot-pair measures REAL s/step per
    arch x task via the SAME code path as training; --msweep-pilot is
    R3-F4's checkpoints-resident 2-M-value pilot, driven through the
    AUDITED run_msweep_timing_pilot wrapper.
  - The axis-2 capped-inference eval pass (M-sweep fan-out + contender
    horizon references).
  - sec 1.7 gate 5 launch-token enforcement: every GPU mode REFUSES to run
    unless BOTH env tokens (HEADTOHEAD_PI_SIGNOFF=1,
    HEADTOHEAD_MATCH_GATE_SIGNOFF=1) are set AND the gates 6+7 box token
    files exist. --token-probe exposes this check for the chain's
    negative-test triple.

DEPLOY-STAGE PINS (flagged, not silently improvised -- each carried into
the CALIBRATION_COMPLETE report for coordinator review at margin freeze):
  DEPLOY-PIN-1: n_query_train=8 queries/episode during TRAINING (symmetric
    across all three arms; full-K queries at EVAL). The design pins the
    joint-loss formula but not the training query count; the Transformer's
    tap replicates context per query (sec 1.3.1.2's own disclosed cost), so
    full-K training queries would triple that arm's step cost asymmetrically.
  DEPLOY-PIN-2: task3 trains on openr1-mix-ext (frozen_bias_lm_sweep.py's
    own calibration_cell precedent), with BOTH corpora's val losses logged
    (lm_pretrain_rd's own OTHER_CORPUS pairing convention).
  DEPLOY-PIN-3: tasks 1/2 model vocab = pools.vocab_size_total (50259: GPT-2
    base + BUFFER + <Q>), per sec 1.3.1.1's T_val over vocab_size_total.
    The two reserved rows are ordinary learned embeddings in ALL THREE arms
    symmetrically (no per-arm asymmetry; model_rd's buffer-row zero-pinning
    is that architecture's own convention, not part of the three arms' pin).
  DEPLOY-PIN-4: axis-2 long-horizon contexts = the task's bind sequence
    right-padded with BUFFER filler tokens to the absolute horizon length
    (H2=454/H4=902/H8=1798, sec 1.4.2) -- bind->query distance therefore
    equals the horizon. The design's "filler/distractor tokens" language is
    satisfied with the task's own neutral BUFFER token; distractor-CONTENT
    filler is a coordinator-reviewable variant, not built here.
  DEPLOY-PIN-5: task2 config K=32, H_train=(1,2), H_test=(3,4) (sec 1.4's
    task-2 pin), n_query per DEPLOY-PIN-1.

Run the CPU selftest: REASONING_LINK_FORCE_CPU_STUB=1 python h2h_cell_train_rd.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from h2h_fla_stub_rd import ensure_fla_stub

_STUB_INSTALLED = ensure_fla_stub()
from lm_pretrain_rd import (DeltaNetLM, load_corpus, get_batch, get_lr,          # noqa: E402
                            corpus_fixed_seed, set_and_log_tf32,
                            DEFAULT_DATA_DIR, OTHER_CORPUS)
from ablation_mixer_rd import AblationLM                                          # noqa: E402
from transformer_baseline_rd import (TransformerLM, sink_fifo_mask,               # noqa: E402
                                     cap_length_tokens)
import probe_head_rd as ph                                                        # noqa: E402
from grammar_rd import (DeltaNetRDTaskConfig, build_entity_pools,                 # noqa: E402
                        sample_batch_rd)
from h2h_calibration_wrappers_rd import (build_full_calibration_manifest,         # noqa: E402
                                         run_msweep_timing_pilot,
                                         project_and_gate_msweep_fanout)
from h2h_sweep_runner_rd import (build_27_cell_manifest, REQUIRED_RESULT_KEYS,    # noqa: E402
                                 is_valid_result)
from h2h_msweep_fanout_rd import build_90_pass_manifest, run_msweep_fanout        # noqa: E402

# ---------------------------------------------------------------------------
# Pinned configs (sec 1.2 / 1.3 / 1.4 / 1.6)
# ---------------------------------------------------------------------------

VOCAB_BASE = 50257
CONTENDER_KW = dict(d_model=256, d_state=64, n_layers=2, conv_size=4, num_heads=1, ffn_mult=4,
                    frozen_bias_arm="per_token", frozen_bias_lambda=0.58)   # sec 1.2 pin
ABLATION_KW = dict(d_model=256, d_state=64, n_layers=2, conv_size=4, ffn_mult=4,
                   gate_extra_width=0)                                       # sec 1.3(a), d_state pinned
TRANSFORMER_KW = dict(d_model=256, n_layers=2, n_heads=4, ffn_mult=4)        # sec 1.3(b), R3-F3 pin
TAP_DIM = {"contender": 64, "ablation": 64, "transformer": 256}
VALUE_DIM = 64                                    # sec 1.3.1.1: value_dim = contender d_state at rung

FULL_STEPS = 20_000                               # sec 1.6's 20,000-step cell (0.2524 GPU-h realized rate)
WARMUP_STEPS = 100                                # M3 pin (h2h_calibration_wrappers_rd.WARMUP_STEPS)
WEIGHT_DECAY = 0.01                               # lm_pretrain_rd default, reused for every arm
GRAMMAR_BATCH = 32
GRAMMAR_EVAL_EVERY = 500                          # axis-1 learning-curve resolution (sec 1.4.1)
LM_SEQ_LEN, LM_BATCH = 512, 32                    # frozen_bias_lm_sweep.py's own rung-1 regime
LM_EVAL_EVERY, LM_EVAL_BATCHES, LM_EVAL_BATCH_SIZE = 1000, 8, 16
N_QUERY_TRAIN = 8                                 # DEPLOY-PIN-1
EVAL_EPISODE_BATCHES = 4                          # 4 x 32 = 128 fixed eval episodes per hop-set
EVAL_SEED = 20260710                              # fixed eval-episode seed, IDENTICAL across arms/seeds
                                                  # (AUDIT FIX-1 discipline applied to episodes); distinct
                                                  # from PROBE_TARGET_SEED/ANCHOR_INIT_SEED by construction
TASK3_CORPUS = "openr1-mix-ext"                   # DEPLOY-PIN-2
HORIZON_TOKENS = {"H2": 454, "H4": 902, "H8": 1798}   # sec 1.4.2 absolute-token horizons
K_SINK = 4                                        # F-NEW-2 sink+FIFO pin
GRAD_RATIO_STEP = 500                             # sec 1.3.1.3's aux_weight revision-trigger step
# sec 1.3.1.3 CALIBRATION REVISION (2026-07-09, the pre-registered trigger FIRED):
# the aux_weight=0.1 calibration run measured ce/aux backbone grad-norm ratio ~20.9x
# at step 500 (ce 0.1375 vs aux 0.0066; exceeds_10x_trigger=True on all nine task-1/2
# cells, all three arms; recovered_frac@0.9=0.0 probe-wide). The design pinned the
# TRIGGER (>10x) but left the revision TARGET unpinned; we target GRADIENT PARITY at
# the tap: 0.1 * 20.93 ~= 2.09, pinned at the round value 2.0 (rounding disclosed).
# Frozen before the 27-cell sweep. The aux_weight=0.1 run's artifacts stay archived
# (calib/archive_auxweight0.1/ on box); re-run cells carry the _auxrev2 name suffix.
AUX_WEIGHT = 2.0                                  # was ph.AUX_WEIGHT_DEFAULT (0.1)

GATE_TOKEN_FILES = ("GATE6_MATCH_GATE_PASSED.token", "GATE7_PROBE_CAPACITY_NULL_PASSED.token")


_POOLS_CACHE: dict = {}


def get_pools(device: str):
    """build_entity_pools returns (pools, report); tokenizer load is slow, so
    cache per (process, device)."""
    if device not in _POOLS_CACHE:
        pools, _report = build_entity_pools()
        _POOLS_CACHE[device] = pools.to(device)
    return _POOLS_CACHE[device]


def task_cfg(task: str, K: int | None, n_query: int | None) -> DeltaNetRDTaskConfig:
    """task1: single-hop recall (hop_set=(1,)); H_test=(2,)/H_extra=() are
    config-validity placeholders never sampled. task2: DEPLOY-PIN-5."""
    if task.startswith("task1"):
        return DeltaNetRDTaskConfig(K=K if K is not None else 32, conv_size=4, n_query=n_query,
                                    H_train=(1,), H_test=(2,), H_extra=())
    if task.startswith("task2"):
        return DeltaNetRDTaskConfig(K=32, conv_size=4, n_query=n_query,
                                    H_train=(1, 2), H_test=(3, 4), H_extra=())
    raise ValueError(f"task_cfg is for grammar tasks only, got {task!r}")


# ---------------------------------------------------------------------------
# sec 1.7 gate 5 -- launch-token enforcement (env pair + gates-6/7 box files)
# ---------------------------------------------------------------------------

def require_launch_tokens(gates_dir: str) -> None:
    """Hard-refuses (exit 3) unless BOTH env tokens are '1' AND both gate
    token files exist. The chain's negative-test triple drives this via
    --token-probe with each token independently removed."""
    missing = [v for v in ("HEADTOHEAD_PI_SIGNOFF", "HEADTOHEAD_MATCH_GATE_SIGNOFF")
               if os.environ.get(v) != "1"]
    absent = [f for f in GATE_TOKEN_FILES if not os.path.isfile(os.path.join(gates_dir, f))]
    if missing or absent:
        print(f"REFUSE (sec 1.7 gate 5): missing env tokens={missing} "
              f"missing gate token files={absent} (gates_dir={gates_dir})", file=sys.stderr)
        raise SystemExit(3)


def require_margins_frozen(margins_token: str) -> None:
    """CODE-LEVEL sweep-release gate (deploy audit MAJOR-2): the 27-cell
    sweep, the fan-out, and the horizon references are unreachable until
    the coordinator writes MARGINS_FROZEN.token -- enforced HERE, not only
    by the chain script's stage ordering. Hard-refuse: exit 4."""
    if not os.path.isfile(margins_token):
        print(f"REFUSE (margin freeze): {margins_token} missing -- the 27-cell sweep / fan-out "
              f"is NOT released until the coordinator records the margin freeze.", file=sys.stderr)
        raise SystemExit(4)


# ---------------------------------------------------------------------------
# Arms + probe rig
# ---------------------------------------------------------------------------

def build_arm_model(arch: str, vocab_size: int, seed: int, device: str):
    torch.manual_seed(seed)
    if arch == "contender":
        m = DeltaNetLM(vocab_size, **CONTENDER_KW)
    elif arch == "ablation":
        m = AblationLM(vocab_size, **ABLATION_KW)
    elif arch == "transformer":
        m = TransformerLM(vocab_size, **TRANSFORMER_KW)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    return m.to(device)


class ProbeRig:
    """T_val (frozen buffer) + shared_probe + per-arm adapter, all built via
    the AUDITED probe_head_rd factories -- this class only owns placement
    and (de)serialization."""

    def __init__(self, arch: str, vocab_size_total: int, device: str):
        self.arch = arch
        self.T_val = ph.build_probe_target_table(vocab_size_total, VALUE_DIM).to(device)
        self.T_val.requires_grad_(False)
        self.adapter = ph.build_adapter_arm(TAP_DIM[arch], VALUE_DIM).to(device)
        self.probe = ph.build_shared_probe(VALUE_DIM).to(device)

    def parameters(self):
        return list(self.adapter.parameters()) + list(self.probe.parameters())

    def pred(self, tap: torch.Tensor) -> torch.Tensor:
        return self.probe(self.adapter(tap))

    def state_dict(self):
        return {"adapter": self.adapter.state_dict(), "probe": self.probe.state_dict()}

    def load_state_dict(self, d):
        self.adapter.load_state_dict(d["adapter"])
        self.probe.load_state_dict(d["probe"])


def probe_targets(rig: ProbeRig, batch: dict) -> torch.Tensor:
    """target = T_val[token id of the entity at the query's target slot]
    (sec 1.3.1.1's target(entity_token_id) = T_val[entity_token_id])."""
    target_ids = torch.gather(batch["entity_ids"], 1, batch["tgt_slot"])   # (B,Q)
    return rig.T_val[target_ids]                                            # (B,Q,VALUE_DIM)


# ---------------------------------------------------------------------------
# Taps: audited (eval) + fused (training -- ONE context forward feeds both
# CE and the tap, verified equivalent to the audited functions at step 0)
# ---------------------------------------------------------------------------

AUDITED_TAP = {"contender": ph.contender_native_tap, "ablation": ph.ablation_native_tap,
               "transformer": ph.transformer_native_tap}


def _q_last_pathway(model, query_tokens: torch.Tensor) -> torch.Tensor:
    """Mirrors probe_head_rd.contender_native_tap's query pathway VERBATIM
    (embed -> last block norm1 -> q_proj -> q_conv1d -> last position) --
    equivalence to the audited function is asserted at step 0 of every
    grammar cell (see fused_logits_and_tap)."""
    B, Q, qlen = query_tokens.shape
    last_block = model.blocks[-1]
    a = last_block.norm1(model.embed(query_tokens.reshape(B * Q, qlen)))
    q_conv, _ = last_block.mixer.q_conv1d(last_block.mixer.q_proj(a))
    return q_conv[:, -1, :].view(B, Q, -1)


def fused_logits_and_tap(arch: str, model, token_ids: torch.Tensor, query_tokens: torch.Tensor):
    """Training-time fused pass: contender/ablation reuse ONE context forward
    for both the CE logits and the tap's final state (the audited tap
    functions run their own full context pass internally -- correct but 2x
    the forward cost when CE needs the same pass anyway). Transformer has no
    shared pass to reuse (its tap structurally requires the [ctx ++ query]
    replication, sec 1.3.1.2) so it calls the audited function directly."""
    if arch == "transformer":
        logits = model(token_ids)
        tap = ph.transformer_native_tap(model, token_ids, query_tokens, attn_mask_fn=None)
        return logits, tap
    logits, final_states = model(token_ids, return_states=True)
    q_last = _q_last_pathway(model, query_tokens)
    if arch == "contender":
        S_T_last = final_states[-1]
        assert S_T_last.shape[1] == 1, "contender tap assumes num_heads==1 (rung config)"
        tap = torch.einsum("bij,bqj->bqi", S_T_last.squeeze(1), q_last)
    else:
        tap = final_states[-1].unsqueeze(1) * q_last
    return logits, tap


def assert_fused_tap_matches_audited(arch: str, model, token_ids, query_tokens) -> float:
    """Step-0 wiring check: the fused tap must numerically match the audited
    probe_head_rd tap on the same inputs (atol/rtol 1e-3 -- a WIRING check,
    not a numerics benchmark). Under the CPU stub the contender's state is
    identically zero, so the check is trivially true there but still runs
    (shape + code-path coverage); the real check happens on box."""
    with torch.no_grad():
        _, fused = fused_logits_and_tap(arch, model, token_ids, query_tokens)
        audited = AUDITED_TAP[arch](model, token_ids, query_tokens)
        diff = (fused - audited).abs().max().item()
    assert torch.allclose(fused, audited, atol=1e-3, rtol=1e-3), (
        f"{arch}: fused training tap diverges from the audited probe_head_rd tap "
        f"(max abs diff {diff}) -- wiring bug, refusing to train")
    return diff


EVAL_CHUNK_BUDGET_BYTES = 8e9    # per-chunk transient-tensor target; H100 80GB leaves wide margin


def _transformer_episode_chunks(b: dict, T_total: int):
    """VRAM guard for the transformer tap (context replicated ONCE PER QUERY,
    B*Q sequences): the dominant transients are the tied-head LOGITS,
    (B*Q, T, 50259) fp32 -- the house-rule bottleneck -- plus attention
    scores (B*Q, n_heads, T, T). Yields episode sub-batches sized so both
    stay within EVAL_CHUNK_BUDGET_BYTES (min 1 episode; at H8 one episode's
    ~13 GB transient is the floor, well inside 80 GB)."""
    q = b["query_tokens"].shape[1]
    per_episode = (q * T_total * (VOCAB_BASE + 2) * 4
                   + q * TRANSFORMER_KW["n_heads"] * T_total * T_total * 4)
    chunk = max(1, int(EVAL_CHUNK_BUDGET_BYTES / per_episode))
    for i in range(0, b["token_ids"].shape[0], chunk):
        yield {k: (v[i:i + chunk] if isinstance(v, torch.Tensor) else v) for k, v in b.items()}


@torch.no_grad()
def eval_recovered_frac(arch: str, model, rig: ProbeRig, eval_batches: list,
                        attn_mask_fn=None) -> tuple[float, float]:
    """recovered_frac@0.9 over the FIXED eval episode set, via the AUDITED
    tap functions. attn_mask_fn (transformer only): capped b-primary read.
    Query-weighted mean; transformer batches are VRAM-chunked.

    Returns (recovered_frac@0.9, probe_cos_mean). probe_cos_mean is the
    query-weighted mean cosine between pred and target BEFORE thresholding
    (diagnosability ADDITION, 2026-07-09 aux revision: distinguishes a
    learning-but-below-bar probe, cos rising toward 0.9, from a dead one,
    cos ~= 0). recovered_frac@0.9 stays the pre-registered decision metric;
    its formula (ph.cosine_recovery_frac) is untouched."""
    model.eval()
    n_hit, n_cos, n_tot = 0.0, 0.0, 0
    for b_full in eval_batches:
        if arch == "transformer":
            T_total = b_full["token_ids"].shape[1] + b_full["query_tokens"].shape[-1]
            chunks = _transformer_episode_chunks(b_full, T_total)
        else:
            chunks = (b_full,)
        for b in chunks:
            if arch == "transformer":
                tap = ph.transformer_native_tap(model, b["token_ids"], b["query_tokens"],
                                                attn_mask_fn=attn_mask_fn)
            else:
                tap = AUDITED_TAP[arch](model, b["token_ids"], b["query_tokens"])
            pred, tgt = rig.pred(tap), probe_targets(rig, b)
            frac = ph.cosine_recovery_frac(pred, tgt).item()
            cos_mean = F.cosine_similarity(pred, tgt, dim=-1).mean().item()
            n_q = b["query_tokens"].shape[0] * b["query_tokens"].shape[1]
            n_hit += frac * n_q
            n_cos += cos_mean * n_q
            n_tot += n_q
    model.train()
    return n_hit / max(1, n_tot), n_cos / max(1, n_tot)


def make_eval_episodes(cfg_eval: DeltaNetRDTaskConfig, pools, device: str, hop_set: tuple,
                       n_batches: int | None = None) -> list:
    """Fixed eval episodes: same EVAL_SEED for every arm/seed/cell of a given
    (task, hop_set), so cross-arm deltas compare identical episodes."""
    if n_batches is None:
        n_batches = EVAL_EPISODE_BATCHES     # late-bound so the selftest's global shrink applies
    gen = torch.Generator(device=device)
    gen.manual_seed(EVAL_SEED + 1009 * cfg_eval.K + 31 * sum(hop_set))
    return [sample_batch_rd(cfg_eval, GRAMMAR_BATCH, gen, hop_set, pools, device=device)
            for _ in range(n_batches)]


# ---------------------------------------------------------------------------
# Task 1/2 -- grammar joint-training cell
# ---------------------------------------------------------------------------

def train_grammar_cell(cell: dict, device: str, ckpt_dir: str | None,
                       steps_override: int | None = None, timing_only: bool = False) -> dict:
    task, arch = cell["task"], cell["arch"]
    K = cell.get("K")
    steps = steps_override if steps_override is not None else int(round(FULL_STEPS * cell.get("budget_frac", 1.0)))
    pools = get_pools(device)
    vocab_total = pools.vocab_size_total                          # DEPLOY-PIN-3
    cfg_train = task_cfg(task, K, n_query=N_QUERY_TRAIN)
    cfg_eval = task_cfg(task, K, n_query=None)                    # full-K queries at eval

    model = build_arm_model(arch, vocab_total, cell["seed"], device)
    rig = ProbeRig(arch, vocab_total, device)
    tf32_state = set_and_log_tf32()
    opt = torch.optim.AdamW(list(model.parameters()) + rig.parameters(),
                            lr=cell["lr"], weight_decay=WEIGHT_DECAY)
    gen = torch.Generator(device=device)
    gen.manual_seed(cell["seed"])

    eval_train_hops = None if timing_only else make_eval_episodes(cfg_eval, pools, device, tuple(cfg_train.H_train))
    eval_heldout = (make_eval_episodes(cfg_eval, pools, device, tuple(cfg_train.H_test))
                    if (task.startswith("task2") and not timing_only) else None)
    capped_mask_fn = (lambda T: sink_fifo_mask(T, cap_length=cap_length_tokens(2, 2, 256),
                                               k_sink=K_SINK, device=device))

    curve, grad_ratio = [], None
    loss_first, recent_losses = None, []
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = get_lr(step, cell["lr"], WARMUP_STEPS, steps)
        batch = sample_batch_rd(cfg_train, GRAMMAR_BATCH, gen, tuple(cfg_train.H_train), pools,
                                device=device)
        if step == 1 and not timing_only:
            assert_fused_tap_matches_audited(arch, model, batch["token_ids"], batch["query_tokens"])
        logits, tap = fused_logits_and_tap(arch, model, batch["token_ids"], batch["query_tokens"])
        loss_ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                  batch["token_ids"][:, 1:].reshape(-1))
        pred, target = rig.pred(tap), probe_targets(rig, batch)
        loss = ph.joint_loss(loss_ce, pred, target, aux_weight=AUX_WEIGHT)

        if step == GRAD_RATIO_STEP and not timing_only:
            # sec 1.3.1.3 revision trigger: aux-vs-CE gradient norms where the two losses
            # actually MEET -- the shared backbone parameters (the CE loss structurally never
            # touches the tap tensor itself in this wiring, so a literal at-the-tap CE norm is
            # identically zero for every arm; the backbone-param norm is the operative
            # same-optimizer balance the trigger exists to check). Measured + REPORTED only;
            # the freeze decision is the coordinator's at margin freeze.
            params = [p for p in model.parameters() if p.requires_grad]
            g_ce = torch.autograd.grad(loss_ce, params, retain_graph=True, allow_unused=True)
            g_aux = torch.autograd.grad(AUX_WEIGHT * ph.probe_aux_loss(pred, target), params,
                                        retain_graph=True, allow_unused=True)
            n_ce = torch.sqrt(sum((g ** 2).sum() for g in g_ce if g is not None)).item()
            n_aux_l = [g for g in g_aux if g is not None]
            n_aux = torch.sqrt(sum((g ** 2).sum() for g in n_aux_l)).item() if n_aux_l else 0.0
            grad_ratio = {"step": step, "ce_grad_norm_backbone": n_ce,
                          "aux_grad_norm_backbone": n_aux,
                          "ratio_ce_over_aux": (n_ce / n_aux) if n_aux > 0 else None,
                          "exceeds_10x_trigger": (not (0.1 <= n_ce / n_aux <= 10.0))
                                                 if n_aux > 0 else True}
            # Overshoot sanity-guard (aux_weight rev 0.1 -> 2.0): fail LOUDLY if the revised
            # ratio inverts beyond 10x the OTHER way (aux swamping CE) -- the re-run must
            # self-report if 2.0 overshoots, never train through a swamped CE silently.
            assert n_aux <= 10.0 * n_ce, (
                f"sec 1.3.1.3 INVERSE trigger: aux_weight={AUX_WEIGHT} OVERSHOOT -- aux backbone "
                f"grad norm {n_aux:.6f} > 10x CE {n_ce:.6f}; calibration must re-revise aux_weight")

        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all()
                     for p in list(model.parameters()) + rig.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + rig.parameters(), 1.0)
            opt.step()
        if loss_first is None:
            loss_first = loss.item()
        recent_losses = (recent_losses + [loss.item()])[-5:]

        if not timing_only and (step % GRAMMAR_EVAL_EVERY == 0 or step == steps):
            rf_tr, cos_tr = eval_recovered_frac(arch, model, rig, eval_train_hops)
            point = {"step": step, "train_loss": loss.item(),
                     "recovered_frac_train_hops": rf_tr,
                     "probe_cos_mean_train_hops": cos_tr}
            if eval_heldout is not None:
                rf_ho, cos_ho = eval_recovered_frac(arch, model, rig, eval_heldout)
                point["recovered_frac_heldout_hops"] = rf_ho
                point["probe_cos_mean_heldout_hops"] = cos_ho
            if arch == "transformer" and task.startswith("task1"):
                # sec 1.6 item C: the Transformer's own capped-cache (b-primary) behavior at this
                # K/d, report-only at calibration (M=2, the smallest floor-eligible point).
                rf_m2, cos_m2 = eval_recovered_frac(arch, model, rig, eval_train_hops,
                                                    attn_mask_fn=capped_mask_fn)
                point["recovered_frac_capped_M2"] = rf_m2
                point["probe_cos_mean_capped_M2"] = cos_m2
            curve.append(point)
            print(f"  [{cell['name']}] step {step}/{steps} loss {loss.item():.4f} "
                  f"rf_train {point['recovered_frac_train_hops']:.4f} "
                  f"cos_train {point['probe_cos_mean_train_hops']:.4f}", flush=True)

    wall_s = time.time() - t0
    result = {"arch": arch, "task": task, "seed_idx": cell.get("seed_idx", 0),
              "seed": cell["seed"], "K": K, "role": cell.get("role"), "lr": cell["lr"],
              "step_count": steps, "wall_s": wall_s, "s_per_step": wall_s / max(1, steps),
              "n_params": sum(p.numel() for p in model.parameters()),
              "vocab_size_total": vocab_total, "n_query_train": N_QUERY_TRAIN,
              "aux_weight": AUX_WEIGHT, "tf32": tf32_state,
              "loss_first": loss_first, "loss_final_mean5": sum(recent_losses) / max(1, len(recent_losses)),
              "grad_ratio_at_tap_step500": grad_ratio, "curve": curve}
    if curve:
        result["final_recovered_frac_train_hops"] = curve[-1]["recovered_frac_train_hops"]
        result["final_probe_cos_mean_train_hops"] = curve[-1]["probe_cos_mean_train_hops"]
        result["final_metric"] = curve[-1].get("recovered_frac_heldout_hops",
                                               curve[-1]["recovered_frac_train_hops"])
        if "recovered_frac_heldout_hops" in curve[-1]:
            result["final_recovered_frac_heldout_hops"] = curve[-1]["recovered_frac_heldout_hops"]
            result["final_probe_cos_mean_heldout_hops"] = curve[-1]["probe_cos_mean_heldout_hops"]
    else:
        result["final_metric"] = float("nan")
    if ckpt_dir and not timing_only:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, cell["name"] + ".pt")
        torch.save({"arch": arch, "task": task, "K": K, "vocab_size_total": vocab_total,
                    "model": model.state_dict(), "rig": rig.state_dict(), "cell": cell},
                   ckpt_path)
        result["ckpt_path"] = ckpt_path
    return result


# ---------------------------------------------------------------------------
# Task 3 -- arch-generic LM cell (mirrors lm_pretrain_rd.train()'s regime)
# ---------------------------------------------------------------------------

def train_lm_cell(cell: dict, device: str, ckpt_dir: str | None,
                  steps_override: int | None = None, timing_only: bool = False,
                  data_dir: str = DEFAULT_DATA_DIR) -> dict:
    arch = cell["arch"]
    corpus = cell.get("corpus", TASK3_CORPUS)
    steps = steps_override if steps_override is not None else int(round(FULL_STEPS * cell.get("budget_frac", 1.0)))
    train_toks, val_toks, _meta, train_offs, val_offs = load_corpus(data_dir, corpus, device)
    other = OTHER_CORPUS[corpus]
    o_train, o_val, _om, _oto, o_voffs = load_corpus(data_dir, other, device)
    del o_train, _oto

    model = build_arm_model(arch, VOCAB_BASE, cell["seed"], device)
    tf32_state = set_and_log_tf32()
    opt = torch.optim.AdamW(model.parameters(), lr=cell["lr"], weight_decay=WEIGHT_DECAY)
    gen = torch.Generator(device=device)
    gen.manual_seed(cell["seed"])

    @torch.no_grad()
    def _val(tokens, corpus_name):
        # FIX-1 discipline: eval windows drawn from corpus_fixed_seed, identical across seeds/arms.
        eg = torch.Generator(device=device)
        eg.manual_seed(corpus_fixed_seed(corpus_name))
        model.eval()
        tot = 0.0
        for _ in range(LM_EVAL_BATCHES):
            x, y = get_batch(tokens, LM_EVAL_BATCH_SIZE, LM_SEQ_LEN, eg)
            lg = model(x)
            tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1)).item()
        model.train()
        return tot / LM_EVAL_BATCHES

    init_val = None if timing_only else _val(val_toks, corpus)
    curve, n_skipped = [], 0
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = get_lr(step, cell["lr"], WARMUP_STEPS, steps)
        x, y = get_batch(train_toks, LM_BATCH, LM_SEQ_LEN, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:
            n_skipped += 1
        if not timing_only and (step % LM_EVAL_EVERY == 0 or step == steps):
            point = {"step": step, "train_loss": loss.item(), "val_loss_own": _val(val_toks, corpus)}
            curve.append(point)
            print(f"  [{cell['name']}] step {step}/{steps} train {loss.item():.4f} "
                  f"val_own {point['val_loss_own']:.4f}", flush=True)

    wall_s = time.time() - t0
    result = {"arch": arch, "task": cell["task"], "seed_idx": cell.get("seed_idx", 0),
              "seed": cell["seed"], "role": cell.get("role"), "lr": cell["lr"], "corpus": corpus,
              "step_count": steps, "wall_s": wall_s, "s_per_step": wall_s / max(1, steps),
              "n_params": sum(p.numel() for p in model.parameters()),
              "skip_rate": n_skipped / max(1, steps), "tf32": tf32_state,
              "init_val_loss_own": init_val, "curve": curve}
    if curve:
        result["final_val_loss_own"] = curve[-1]["val_loss_own"]
        result["final_val_loss_other"] = _val(o_val, other)
        result["final_metric"] = result["final_val_loss_own"]
        del o_val, o_voffs
    else:
        result["final_metric"] = float("nan")
    if ckpt_dir and not timing_only:
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, cell["name"] + ".pt")
        torch.save({"arch": arch, "task": cell["task"], "corpus": corpus,
                    "vocab_size_total": VOCAB_BASE, "model": model.state_dict(), "cell": cell},
                   ckpt_path)
        result["ckpt_path"] = ckpt_path
    return result


def run_one_cell(cell: dict, device: str, ckpt_dir: str | None, **kw) -> dict:
    if cell["task"].startswith("task3"):
        return train_lm_cell(cell, device, ckpt_dir, **kw)
    return train_grammar_cell(cell, device, ckpt_dir, **kw)


# ---------------------------------------------------------------------------
# Cell naming / manifests (single source of truth for the chain script)
# ---------------------------------------------------------------------------

AUX_REV_SUFFIX = "_auxrev2"     # aux_weight=2.0 re-run naming (task-1/2 cells only): resume-safety
                                # re-runs the probe cells WITHOUT clobbering the failed aux_weight=0.1
                                # artifacts (archived record); task-3 cells have no probe/aux term, so
                                # their names -- and their PASSED band results -- stand unchanged.


def calibration_cells() -> list[dict]:
    cells = []
    for c in build_full_calibration_manifest():
        if c["role"] == "reused_not_relaunched":
            continue    # sec 1.6 item E: contender task3 = FROZEN_BIAS rung-1, never re-launched
        name = f"h2h_calib_{c['arch']}_{c['task']}_{c['role']}"
        if "K" in c:
            name += f"_K{c['K']}"
        if not c["task"].startswith("task3"):
            name += AUX_REV_SUFFIX
        c = {**c, "name": name, "seed_idx": 0}
        cells.append(c)
    names = [c["name"] for c in cells]
    assert len(names) == len(set(names)) == 13, f"expected 13 unique launchable calibration cells, got {names}"
    return cells


def sweep_cells() -> list[dict]:
    out = []
    for c in build_27_cell_manifest():
        c = dict(c)
        c["budget_frac"], c["role"], c["lr"] = 1.0, "sweep", 3e-4
        # M3: every arm reuses the contender's pinned LR by default; the Transformer's task-3 LR
        # is REVISED at margin freeze from the calibration LR grid (the chain blocks on
        # MARGINS_FROZEN.token before any sweep cell launches -- coordinator writes the chosen LR
        # into that token file; see apply_margin_freeze_overrides below).
        if c["task"] == "task1_sweep":
            c["K"] = 32
        out.append(c)
    return out


def apply_margin_freeze_overrides(cells: list[dict], token_path: str) -> list[dict]:
    """MARGINS_FROZEN.token may carry {'transformer_task3_lr': <float>} --
    the coordinator's LR-grid decision applied to the sweep manifest. Any
    other overrides are deliberately NOT honored (no silent re-scoping)."""
    try:
        with open(token_path) as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return cells
    lr = doc.get("transformer_task3_lr")
    if lr is not None:
        for c in cells:
            if c["arch"] == "transformer" and c["task"] == "task3_sweep":
                c["lr"] = float(lr)
    return cells


# ---------------------------------------------------------------------------
# Axis 2 -- capped-inference eval pass (M-sweep pilot + 90-pass fan-out)
# ---------------------------------------------------------------------------

def extend_context_to_horizon(token_ids: torch.Tensor, horizon_tokens: int, buffer_id: int):
    """DEPLOY-PIN-4: right-pad the bind sequence with BUFFER filler to the
    absolute horizon length."""
    B, T = token_ids.shape
    assert horizon_tokens >= T, f"horizon {horizon_tokens} < bind length {T}"
    pad = torch.full((B, horizon_tokens - T), buffer_id, dtype=token_ids.dtype,
                     device=token_ids.device)
    return torch.cat([token_ids, pad], dim=1)


def load_h2h_checkpoint(path: str, device: str):
    doc = torch.load(path, map_location=device)
    arch, vocab = doc["arch"], doc["vocab_size_total"]
    model = build_arm_model(arch, vocab, seed=0, device=device)
    model.load_state_dict(doc["model"])
    model.eval()
    rig = None
    if "rig" in doc:
        rig = ProbeRig(arch, vocab, device)
        rig.load_state_dict(doc["rig"])
    return model, rig, doc


def capped_eval_pass(model, rig: ProbeRig, arch: str, task: str, K: int | None, M: int | None,
                     horizon: str, pools, device: str) -> dict:
    """ONE axis-2 inference pass: fixed eval episodes at the task's own eval
    hop set, context extended to the horizon; transformer capped via
    sink+FIFO at cap_length(M) (M=None -> uncapped: contender reference /
    b-control)."""
    cfg_eval = task_cfg(task, K, n_query=None)
    hop_set = tuple(cfg_eval.H_test) if task.startswith("task2") else tuple(cfg_eval.H_train)
    batches = make_eval_episodes(cfg_eval, pools, device, hop_set)
    H = HORIZON_TOKENS[horizon]
    mask_fn = None
    if M is not None:
        cl = cap_length_tokens(M, TRANSFORMER_KW["n_layers"], TRANSFORMER_KW["d_model"])
        mask_fn = lambda T: sink_fifo_mask(T, cap_length=cl, k_sink=K_SINK, device=device)  # noqa: E731

    def _episode_chunks(b):
        # VRAM guard (transformer only): see _transformer_episode_chunks -- the recurrent
        # arms have no per-query replication or T^2 term and run unchunked.
        if arch != "transformer":
            yield b
            return
        yield from _transformer_episode_chunks(b, H + b["query_tokens"].shape[-1])

    n_hit, n_tot = 0.0, 0
    with torch.no_grad():
        for b_full in batches:
            for b in _episode_chunks(b_full):
                ctx = extend_context_to_horizon(b["token_ids"], H, pools.buffer_id)
                if arch == "transformer":
                    tap = ph.transformer_native_tap(model, ctx, b["query_tokens"], attn_mask_fn=mask_fn)
                else:
                    tap = AUDITED_TAP[arch](model, ctx, b["query_tokens"])
                frac = ph.cosine_recovery_frac(rig.pred(tap), probe_targets(rig, b)).item()
                n_q = b["query_tokens"].shape[0] * b["query_tokens"].shape[1]
                n_hit += frac * n_q
                n_tot += n_q
    return {"recovered_frac": n_hit / max(1, n_tot), "horizon_tokens": H,
            "cap_length": (None if M is None else
                           cap_length_tokens(M, TRANSFORMER_KW["n_layers"], TRANSFORMER_KW["d_model"])),
            "n_episodes": len(batches) * GRAMMAR_BATCH}


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------

def _atomic_dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def mode_run_cell(args) -> int:
    require_launch_tokens(args.gates_dir)
    cells = {c["name"]: c for c in (calibration_cells() if args.set == "calibration" else sweep_cells())}
    assert args.run_cell in cells, f"unknown cell {args.run_cell!r} in set {args.set}"
    cell = cells[args.run_cell]
    if args.set == "sweep":
        require_margins_frozen(args.margins_token)          # audit MAJOR-2: code-level release gate
        cell = apply_margin_freeze_overrides([cell], args.margins_token)[0]
    if is_valid_result(args.out):
        print(f"SKIP (already valid): {args.out}")
        return 0
    result = run_one_cell(cell, args.device, args.ckpt_dir)
    result = {**cell, **result}
    assert all(k in result for k in REQUIRED_RESULT_KEYS)
    _atomic_dump(args.out, result)
    print(f"CELL COMPLETE: {args.run_cell} final_metric={result['final_metric']} "
          f"wall_s={result['wall_s']:.1f}")
    return 0


def mode_pilot_pair(args) -> int:
    """sec 1.7 gate 2: ONE real short run per arch x task at the pair's own
    primary config, measured s/step -> projected s/full-cell."""
    require_launch_tokens(args.gates_dir)
    arch, task = args.pilot_pair
    base = {"arch": arch, "task": f"{task}_calib", "role": "pilot", "budget_frac": 1.0,
            "seed": 12345, "lr": 3e-4, "name": f"pilot_{arch}_{task}", "seed_idx": 0}
    if task == "task1":
        base["K"] = 32
    warmup, timed = 10, 40
    t_all = run_one_cell(dict(base), args.device, ckpt_dir=None,
                         steps_override=warmup + timed, timing_only=True)
    # First-steps include kernel autotune/compile warmup; measure a second, warm run for the rate.
    t_warm = run_one_cell(dict(base), args.device, ckpt_dir=None,
                          steps_override=timed, timing_only=True)
    s_per_step = t_warm["wall_s"] / timed
    doc = {"arch": arch, "task": task, "s_per_step_warm": s_per_step,
           "cold_wall_s": t_all["wall_s"], "projected_s_per_full_cell": s_per_step * FULL_STEPS,
           "projected_gpu_h_per_full_cell": s_per_step * FULL_STEPS / 3600.0}
    _atomic_dump(args.out, doc)
    print(f"PILOT {arch}x{task}: {s_per_step:.4f} s/step -> "
          f"{doc['projected_gpu_h_per_full_cell']:.4f} GPU-h/full-cell")
    return 0


# Pair shares (sec 1.6 arithmetic): full-cell-equivalents this pair owes across
# calibration + sweep, x 0.2524 GPU-h x the 10x bracket = the pair's gate ceiling.
PAIR_CELL_EQUIVS = {
    ("contender", "task1"): 1 + 0.25 + 3, ("ablation", "task1"): 1 + 0.25 + 3,
    ("transformer", "task1"): 1 + 0.25 + 3,
    ("contender", "task2"): 1 + 3, ("ablation", "task2"): 1 + 3, ("transformer", "task2"): 1 + 3,
    ("contender", "task3"): 3, ("ablation", "task3"): 1 + 3, ("transformer", "task3"): 0.75 + 3,
}
PLANNING_RATE_GPU_H = 0.2524
BRACKET_MULT = 10.0


def mode_gate_pilots(args) -> int:
    rows, ok_all = [], True
    for (arch, task), equivs in PAIR_CELL_EQUIVS.items():
        path = os.path.join(args.pilots_dir, f"pilot_{arch}_{task}.json")
        with open(path) as f:
            p = json.load(f)
        projected = p["projected_gpu_h_per_full_cell"] * equivs
        ceiling = equivs * PLANNING_RATE_GPU_H * BRACKET_MULT
        ok = projected <= ceiling
        ok_all = ok_all and ok
        rows.append({"arch": arch, "task": task, "cell_equivs": equivs,
                     "projected_gpu_h": projected, "share_ceiling_gpu_h": ceiling, "ok": ok})
        print(f"[pilot gate] {arch}x{task}: projected {projected:.3f} GPU-h vs "
              f"share ceiling {ceiling:.3f} -- {'OK' if ok else 'ABORT'}")
    total = sum(r["projected_gpu_h"] for r in rows)
    doc = {"rows": rows, "total_projected_train_gpu_h": total, "ok": ok_all,
           "planning_rate_gpu_h": PLANNING_RATE_GPU_H, "bracket_mult": BRACKET_MULT}
    _atomic_dump(args.out, doc)
    if not ok_all:
        print("PILOT GATE: ABORT (sec 1.7 gate 2 mechanical enforced abort)", file=sys.stderr)
        return 1
    print(f"PILOT GATE: OK (total projected training {total:.3f} GPU-h)")
    return 0


def mode_msweep_pilot(args) -> int:
    require_launch_tokens(args.gates_dir)
    device = args.device
    pools = get_pools(device)

    def _loader():
        model, rig, doc = load_h2h_checkpoint(args.ckpt, device)
        assert doc["arch"] == "transformer", "M-sweep pilot must run on a (b-primary) transformer ckpt"
        return (model, rig, doc)

    def _eval(ckpt, m, horizon):
        model, rig, doc = ckpt
        capped_eval_pass(model, rig, "transformer", doc["task"], doc.get("K"), m, horizon,
                         pools, device)

    pilot = run_msweep_timing_pilot(_eval, _loader)      # audited wrapper: residency-asserted
    gate = project_and_gate_msweep_fanout(pilot["mean_s_per_pass"],
                                          remaining_headroom_gpu_h=args.headroom_gpu_h)
    doc = {"pilot": pilot, "fanout_gate": gate, "ckpt": args.ckpt,
           "design_time_assumption_s_per_pass_REPLACED": 5.0}
    _atomic_dump(args.out, doc)
    print(f"M-SWEEP PILOT: measured {pilot['mean_s_per_pass']:.3f} s/pass "
          f"(design assumption 5.0 REPLACED); 90-pass projection "
          f"{gate['projected_gpu_h']:.4f} GPU-h ok={gate['ok']}")
    return 0 if gate["ok"] else 1


def mode_fanout_all(args) -> int:
    require_launch_tokens(args.gates_dir)
    require_margins_frozen(args.margins_token)               # audit MAJOR-2: stage-D-only mode
    device = args.device
    pools = get_pools(device)
    with open(args.ckpt_map) as f:
        ckpt_map = {tuple(k.split("|")): v for k, v in json.load(f).items()}

    def _loader(cell):
        model, rig, doc = load_h2h_checkpoint(ckpt_map[(cell["task"], str(cell["seed_idx"]))], device)
        assert doc["arch"] == "transformer", "fan-out checkpoints must be (b-primary) transformer cells"
        return (model, rig, doc)

    def _eval(ckpt, m, horizon):
        # signature pinned by run_pass_exception_isolated: eval_pass_fn(ckpt, M, horizon);
        # task/K come from the resident checkpoint's own doc.
        model, rig, doc = ckpt
        return capped_eval_pass(model, rig, "transformer", doc["task"], doc.get("K", 32),
                                m, horizon, pools, device)

    manifest = build_90_pass_manifest()
    summary = run_msweep_fanout(manifest, _eval, _loader, args.out_dir)
    _atomic_dump(os.path.join(args.out_dir, "FANOUT_SUMMARY.json"), summary)
    print(f"FANOUT: {summary}")
    return 0 if summary.get("failed", 0) == 0 else 1


def mode_horizon_ref(args) -> int:
    """Contender (uncapped, constant-state) reference passes per (task, seed,
    horizon) -- axis 2's own comparison side (18 passes)."""
    require_launch_tokens(args.gates_dir)
    require_margins_frozen(args.margins_token)               # audit MAJOR-2: stage-D-only mode
    device = args.device
    pools = get_pools(device)
    with open(args.ckpt_map) as f:
        ckpt_map = {tuple(k.split("|")): v for k, v in json.load(f).items()}
    out = {}
    for (task, seed_idx), path in sorted(ckpt_map.items()):
        model, rig, doc = load_h2h_checkpoint(path, device)
        for horizon in HORIZON_TOKENS:
            r = capped_eval_pass(model, rig, doc["arch"], task, doc.get("K", 32), None, horizon,
                                 pools, device)
            out[f"{task}|s{seed_idx}|{horizon}"] = r
            print(f"HORIZON-REF {doc['arch']} {task} s{seed_idx} {horizon}: "
                  f"rf={r['recovered_frac']:.4f}")
    _atomic_dump(args.out, out)
    return 0


def mode_calibration_report(args) -> int:
    calib_dir = os.path.join(args.res_dir, "calib")
    pilots_dir = os.path.join(args.res_dir, "pilots")
    cells = {}
    for c in calibration_cells():
        path = os.path.join(calib_dir, c["name"] + ".json")
        with open(path) as f:
            cells[c["name"]] = json.load(f)
    pilots = {}
    for fn in sorted(os.listdir(pilots_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(pilots_dir, fn)) as f:
                pilots[fn] = json.load(f)
    grad_ratios = {n: r.get("grad_ratio_at_tap_step500") for n, r in cells.items()
                   if r.get("grad_ratio_at_tap_step500") is not None}
    report = {
        "wave": "HEAD_TO_HEAD rung-1 (sec 1.19 items 3-4 complete)",
        "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "calibration_cells": cells,
        "reused_not_relaunched": {"contender/task3": "FROZEN_BIAS_LM_DESIGN.md rung-1 "
                                  "(Arm 2 per_token lam=0.58): val 2.1184/4.3426 openr1/wikitext "
                                  "(sec 12.1 verified-numbers table)"},
        "pilots": pilots,
        "aux_grad_ratio_table_sec_1_3_1_3": grad_ratios,
        "deploy_stage_pins_for_review": [
            "DEPLOY-PIN-1: n_query_train=8 (symmetric, all arms; full-K at eval)",
            "DEPLOY-PIN-2: task3 corpus=openr1-mix-ext (both corpora's val logged)",
            "DEPLOY-PIN-3: tasks 1/2 vocab=50259 (reserved rows learned, all arms)",
            "DEPLOY-PIN-4: horizon filler=BUFFER right-padding to absolute horizon",
            "DEPLOY-PIN-5: task2 K=32 H_train=(1,2) H_test=(3,4)",
            "transformer task3 sweep LR awaits margin-freeze override "
            "(transformer_task3_lr in MARGINS_FROZEN.token; default 3e-4)",
        ],
        "next_gate": "MARGINS_FROZEN.token (coordinator-only write) releases the 27-cell sweep",
    }
    _atomic_dump(args.out, report)
    print(f"CALIBRATION_COMPLETE report written: {args.out}")
    return 0


# ---------------------------------------------------------------------------
# CPU selftest (forced stub) -- logic-only; every real-kernel claim is box-side
# ---------------------------------------------------------------------------

def mode_selftest() -> int:
    import subprocess
    import tempfile
    failures = []

    def rep(item, ok, detail=""):
        print(f"[{item}] {'PASS' if ok else 'FAIL'}{(' -- ' + detail) if detail else ''}", flush=True)
        if not ok:
            failures.append(item)

    # 1. manifests: 13 launchable calibration cells, 27 sweep cells, unique names.
    cal, sw = calibration_cells(), sweep_cells()
    rep("selftest 1: manifests (13 calibration launchable, 27 sweep)",
        len(cal) == 13 and len(sw) == 27)

    # 2. task cfgs valid (task1 K=32/48, task2) -- __post_init__ guards run.
    ok2 = True
    for t, k in (("task1_calib", 32), ("task1_calib", 48), ("task2_calib", None)):
        cfg = task_cfg(t, k, n_query=N_QUERY_TRAIN)
        ok2 = ok2 and cfg.queries == N_QUERY_TRAIN
    rep("selftest 2: task cfgs construct with periodicity guards", ok2)

    # 3. launch-token refusal (negative, subprocess so SystemExit is a real exit code).
    with tempfile.TemporaryDirectory() as td:
        env = {k: v for k, v in os.environ.items()
               if k not in ("HEADTOHEAD_PI_SIGNOFF", "HEADTOHEAD_MATCH_GATE_SIGNOFF")}
        env["REASONING_LINK_FORCE_CPU_STUB"] = "1"
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--token-probe",
                            "--gates-dir", td], env=env, capture_output=True)
        rep("selftest 3: --token-probe REFUSES (exit 3) without env tokens + gate files",
            r.returncode == 3)
        # positive: env set AND token files present
        for fn in GATE_TOKEN_FILES:
            with open(os.path.join(td, fn), "w") as f:
                f.write("{}")
        env2 = dict(env, HEADTOHEAD_PI_SIGNOFF="1", HEADTOHEAD_MATCH_GATE_SIGNOFF="1")
        r2 = subprocess.run([sys.executable, os.path.abspath(__file__), "--token-probe",
                             "--gates-dir", td], env=env2, capture_output=True)
        rep("selftest 4: --token-probe passes with both env tokens + both gate files",
            r2.returncode == 0)
        # negative triple: each of the three requirements independently removed
        neg_ok = True
        for kill_env in ("HEADTOHEAD_PI_SIGNOFF", "HEADTOHEAD_MATCH_GATE_SIGNOFF"):
            e = {k: v for k, v in env2.items() if k != kill_env}
            neg_ok = neg_ok and subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--token-probe", "--gates-dir", td],
                env=e, capture_output=True).returncode == 3
        os.remove(os.path.join(td, GATE_TOKEN_FILES[0]))
        neg_ok = neg_ok and subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--token-probe", "--gates-dir", td],
            env=env2, capture_output=True).returncode == 3
        rep("selftest 5: negative-test triple (each token independently removed -> refusal)", neg_ok)

    # 6. grammar joint-training micro-run (ablation + transformer, CPU-provable arms), tiny steps.
    #    Exercises: episode gen, fused tap + step-0 audited-tap equivalence, joint loss, curve,
    #    grad-ratio probe skipped (step<500), result keys.
    for arch in ("ablation", "transformer"):
        cell = {"arch": arch, "task": "task1_calib", "role": "selftest", "budget_frac": 1.0,
                "seed": 7, "lr": 1e-3, "K": 8, "name": f"selftest_{arch}", "seed_idx": 0}
        global GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH
        saved = (GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH)
        GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH = 3, 1, 4
        try:
            r = train_grammar_cell(cell, "cpu", ckpt_dir=None, steps_override=3)
            ok = (all(k in r for k in REQUIRED_RESULT_KEYS) and len(r["curve"]) >= 1
                  and 0.0 <= r["final_recovered_frac_train_hops"] <= 1.0
                  and torch.isfinite(torch.tensor(r["loss_final_mean5"])).item())
        finally:
            GRAMMAR_EVAL_EVERY, EVAL_EPISODE_BATCHES, GRAMMAR_BATCH = saved
        rep(f"selftest 6({arch}): grammar micro-cell trains, taps match audited, keys complete", ok)

    # 7. horizon extension: length + query-distance arithmetic.
    ids = torch.full((2, 20), 5, dtype=torch.int64)
    ext = extend_context_to_horizon(ids, 33, buffer_id=9)
    rep("selftest 7: extend_context_to_horizon pads with BUFFER to exact length",
        ext.shape == (2, 33) and (ext[:, 20:] == 9).all().item())

    # 8. margin-freeze override honored for transformer/task3 ONLY.
    with tempfile.TemporaryDirectory() as td:
        tok = os.path.join(td, "MARGINS_FROZEN.token")
        with open(tok, "w") as f:
            json.dump({"transformer_task3_lr": 1e-4}, f)
        cells2 = apply_margin_freeze_overrides(sweep_cells(), tok)
        t3 = [c for c in cells2 if c["arch"] == "transformer" and c["task"] == "task3_sweep"]
        others = [c for c in cells2 if not (c["arch"] == "transformer" and c["task"] == "task3_sweep")]
        rep("selftest 8: margin-freeze LR override hits transformer/task3 only",
            all(c["lr"] == 1e-4 for c in t3) and all(c["lr"] == 3e-4 for c in others))

    print("=" * 70)
    if failures:
        print(f"SELFTEST: {len(failures)} FAILURE(S): {failures}", file=sys.stderr)
        return 1
    print("SELFTEST: ALL ITEMS PASSED")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="alias for --selftest (house convention)")
    ap.add_argument("--token-probe", action="store_true",
                    help="exit 0 iff launch tokens present (gate 5 negative-test hook)")
    ap.add_argument("--list-cells", choices=["calibration", "sweep"])
    ap.add_argument("--run-cell", type=str)
    ap.add_argument("--set", choices=["calibration", "sweep"], default="calibration")
    ap.add_argument("--pilot-pair", nargs=2, metavar=("ARCH", "TASK"))
    ap.add_argument("--gate-pilots", action="store_true")
    ap.add_argument("--pilots-dir", type=str)
    ap.add_argument("--msweep-pilot", action="store_true")
    ap.add_argument("--fanout-all", action="store_true")
    ap.add_argument("--horizon-ref", action="store_true")
    ap.add_argument("--write-calibration-report", action="store_true")
    ap.add_argument("--ckpt", type=str)
    ap.add_argument("--ckpt-map", type=str)
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument("--out", type=str)
    ap.add_argument("--out-dir", type=str)
    ap.add_argument("--res-dir", type=str)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--gates-dir", type=str, default="results/h2h_rung1/gates")
    ap.add_argument("--margins-token", type=str, default="results/h2h_rung1/MARGINS_FROZEN.token")
    ap.add_argument("--headroom-gpu-h", type=float, default=100.0)
    args = ap.parse_args()

    if args.selftest or args.smoke:
        return mode_selftest()
    if args.token_probe:
        require_launch_tokens(args.gates_dir)
        print("launch tokens present (gate 5)")
        return 0
    if args.list_cells:
        for c in (calibration_cells() if args.list_cells == "calibration" else sweep_cells()):
            print(c["name"])
        return 0
    if args.run_cell:
        return mode_run_cell(args)
    if args.pilot_pair:
        return mode_pilot_pair(args)
    if args.gate_pilots:
        return mode_gate_pilots(args)
    if args.msweep_pilot:
        return mode_msweep_pilot(args)
    if args.fanout_all:
        return mode_fanout_all(args)
    if args.horizon_ref:
        return mode_horizon_ref(args)
    if args.write_calibration_report:
        return mode_calibration_report(args)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
