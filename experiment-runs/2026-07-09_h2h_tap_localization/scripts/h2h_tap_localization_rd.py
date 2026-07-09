"""h2h_tap_localization_rd.py -- HEAD_TO_HEAD_DEMO_DESIGN.md sec 1.28/1.29's
Stage 2: TAP-PLACEMENT LOCALIZATION. Read-only on checkpoints/results, writes
ONLY under results/h2h_rung1/tap_localization/. No training.

sec 1.29 found rule 3 fires: the answer is reachable through the FULL
nonlinear forward + LM head (contender rung1 = 0.9990) but not linearly
present at the shallow tap S1@q_shallow (rf@0.9 = 0.0 under ridge, the
closed-form optimum). This script asks WHERE between those two points the
recall-enabling computation lives, so Rev 5's repaired continuous instrument
can be pointed at the right place.

Two experiments, both driven off the SAME round-3 `_auxrev2` task1-K32
checkpoints sec 1.29 used (contender/ablation only -- transformer has no
fast-weight states and its own tap OOM fix is unbuilt, sec 1.27):

  1. STATE-ZEROING LOCALIZATION: on the pinned EVAL_SEED episode set, run the
     existing `_recurrent_continuation_answer_logits` continuation three ways
     -- both layers' cached bind-phase state intact (reproduces sec 1.29's
     rung-1 number), S0 zeroed, S1 zeroed -- and read K-restricted rung-1
     accuracy each way (`_rung1_k_restricted_pred_slot`, unmodified). Whichever
     zeroing collapses accuracy carries the bindings.
  2. TAP VARIANTS, offline ridge (closed-form, GPU float64, `probe_diagnosis_
     rd.ridge_fit`/`ridge_pred`/`cos_stats` -- imported/reused unmodified, sec
     1.28's own decision-rule menu item 1 machinery):
       (i)   S1@q_shallow    -- the CURRENT tap (AUDITED_TAP, known rf@0.9=0)
       (ii)  S0@q0-pathway   -- block-0's own q_proj/q_conv1d, applied to the
                                raw query embedding EXACTLY as block 0 already
                                computes it inside any real forward pass
                                (block 0 has no upstream block to bypass, so
                                this is NOT a shortcut reconstruction the way
                                (i)'s q_shallow is for block 1)
       (iii) S1@q_true       -- block-1's ACTUAL post-block-0 query vector,
                                hooked live during the real continuation
                                forward (see `_continuation_pass`'s docstring
                                for the P=1-purity argument)
       (iv)  pre-LM-head hidden at the answer position -- the known-good end
             of the pipeline (rung 1 reads this route, sec 1.29 item 3), as a
             positive control proving the ridge harness itself has teeth.
     Fit set: fresh DIAG_FIT_SEED episodes (probe_diagnosis_rd's own seed).
     Eval set: the pinned EVAL_SEED set (same one experiment 1 uses) --
     directly comparable to sec 1.27/1.29's own numbers.

Run the CPU selftest: REASONING_LINK_FORCE_CPU_STUB=1 python h2h_tap_localization_rd.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
# probe_diagnosis_rd.py lives alongside this file ON THE BOX (deployed there for sec 1.29's
# Stage 1); locally it is only archived (git-tracked, unmodified) under the sec 1.29 experiment
# run -- this fallback makes `import probe_diagnosis_rd` resolve in BOTH places without a second
# copy of the file anywhere (CLAUDE.md: edit/duplicate nothing shared).
_ARCHIVE_SCRIPTS_DIR = os.path.normpath(os.path.join(
    _THIS_DIR, "..", "..", "experiment-runs", "2026-07-09_h2h_decisive_probe", "scripts"))
if not os.path.isfile(os.path.join(_THIS_DIR, "probe_diagnosis_rd.py")) and \
        os.path.isdir(_ARCHIVE_SCRIPTS_DIR):
    sys.path.append(_ARCHIVE_SCRIPTS_DIR)

from h2h_fla_stub_rd import ensure_fla_stub                                    # noqa: E402

_STUB_INSTALLED = ensure_fla_stub()
from h2h_cell_train_rd import (build_arm_model, load_h2h_checkpoint, get_pools,       # noqa: E402
                               task_cfg, make_eval_episodes, AUDITED_TAP, GRAMMAR_BATCH,
                               _recurrent_continuation_answer_logits,
                               _pad_query_tokens_for_continuation, _repeat_states_for_queries,
                               _rung1_k_restricted_pred_slot)
from grammar_rd import sample_batch_rd                                          # noqa: E402
from probe_diagnosis_rd import ridge_fit, ridge_pred, cos_stats, DIAG_FIT_SEED   # noqa: E402

CKPT_DIR = "/data/h2h_rung1_ckpts"
CKPT_PATHS = {
    "contender": os.path.join(CKPT_DIR, "h2h_calib_contender_task1_calib_primary_K32_auxrev2.pt"),
    "ablation": os.path.join(CKPT_DIR, "h2h_calib_ablation_task1_calib_primary_K32_auxrev2.pt"),
}
TAP_NAMES = ("i_s1_qshallow", "ii_s0_q0", "iii_s1_qtrue", "iv_prelmhead")
N_FIT_BATCHES_DEFAULT = 24     # 24 x 32 episodes x 32 queries = 24,576 fit points/tap (ridge only
                                # -- no SGD/MLP iteration cost, so this stays cheap; sec 1.28's
                                # own Stage-1 ran a heavier SGD+MLP+classifier suite in ~0.03 GPU-h)


# ---------------------------------------------------------------------------
# Per-arm query pathways + state/query combine (generalizes h2h_cell_train_rd's
# _q_last_pathway / _recurrent_tap_from_states, both hardcoded to blocks[-1]/
# final_states[-1], to an ARBITRARY block/state pair)
# ---------------------------------------------------------------------------

def _q_pathway_via_block(model, block_idx: int, query_tokens: torch.Tensor) -> torch.Tensor:
    """embed(query_tokens) -> blocks[block_idx].norm1 -> q_proj -> q_conv1d, at the <Q>
    position. For block_idx==0 this is NOT a shortcut: block 0 has no upstream block, so
    this IS exactly what block 0 computes on its own input inside any real forward pass
    (bind-phase or continuation) -- unlike `_q_last_pathway`'s use of this same recipe on
    the LAST block, which skips that block's own upstream (block 0's) processing entirely
    (the q_shallow shortcut sec 1.29 diagnosed). No _MIN_KERNEL_T floor applies here: this
    calls q_proj/q_conv1d directly, never the full mixer.forward that carries that assert."""
    B, Q, qlen = query_tokens.shape
    blk = model.blocks[block_idx]
    a = blk.norm1(model.embed(query_tokens.reshape(B * Q, qlen)))
    q_conv, _ = blk.mixer.q_conv1d(blk.mixer.q_proj(a))
    return q_conv[:, -1, :].view(B, Q, -1)


def _combine_state_query(arch: str, S: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """sec 1.3.1.2's per-arm combine formula (contender: matvec S@q; ablation: Hadamard
    S(.)q), generalized to an ARBITRARY (state, query) pair -- h2h_cell_train_rd's own
    `_recurrent_tap_from_states` is hardcoded to `final_states[-1]`."""
    if arch == "contender":
        assert S.shape[1] == 1, "contender combine assumes num_heads==1 (rung config)"
        return torch.einsum("bij,bqj->bqi", S.squeeze(1), q)
    return S.unsqueeze(1) * q


@torch.no_grad()
def _continuation_pass(arch: str, model, final_states: list, query_tokens: torch.Tensor,
                       buffer_id: int, capture_q1: bool = False):
    """Re-derives `h2h_cell_train_rd._recurrent_continuation_answer_logits` (imported,
    unmodified, called directly below for the wiring self-test) but ALSO exposes the
    pre-LM-head hidden at the answer position (tap iv) and, optionally, block-1's OWN
    q_conv1d output at that position (tap iii) via a forward hook on
    `model.blocks[-1].mixer.q_conv1d` -- ONE forward call now serves the rung-1 accuracy
    read, the positive-control tap, and the true-query tap together, instead of three.

    P=1 purity (sec 1.9 item 9's addendum, restated for this exact use): the hook only
    OBSERVES a value this SAME `model.forward(query_tokens, initial_states=final_states,
    ...)` call already computes internally -- no new read channel. The query DOES touch
    the recurrence during a continuation (updates state as it processes query_tokens,
    exactly the standing `o_t = read(S_t, q_t)` mechanism every LM step uses) but still
    never reaches BEHIND `final_states` into raw bind-phase tokens it was not already
    causally summarized through. Tap (iii) is therefore a pure function of
    (final_states, query_tokens) alone, identically to taps (i)/(ii)/(iv) and to the
    LM-head route rung 1 already reads -- the bottleneck holds for it."""
    B, Q, qlen = query_tokens.shape
    flat = query_tokens.reshape(B * Q, qlen)
    padded = _pad_query_tokens_for_continuation(flat, buffer_id)
    states_rep = _repeat_states_for_queries(final_states, Q)

    captured, handle = {}, None
    if capture_q1:
        def _hook(_module, _inp, out):
            captured["q"] = out[0]
        handle = model.blocks[-1].mixer.q_conv1d.register_forward_hook(_hook)
    try:
        hidden = model.forward(padded, initial_states=states_rep, return_hidden=True)
    finally:
        if handle is not None:
            handle.remove()

    answer_hidden = hidden[:, qlen - 1, :]                          # (B*Q,d_model)
    answer_logits = F.linear(answer_hidden, model.embed.weight).view(B, Q, -1)
    hidden_at_answer = answer_hidden.view(B, Q, -1)
    q_true = captured["q"][:, qlen - 1, :].view(B, Q, -1) if capture_q1 else None
    return answer_logits, hidden_at_answer, q_true


def _accum_loc(loc: dict, mode: str, answer_logits: torch.Tensor, b: dict) -> None:
    pred_slot = _rung1_k_restricted_pred_slot(answer_logits, b["entity_ids"])
    loc[mode][0] += (pred_slot == b["tgt_slot"]).float().sum().item()
    loc[mode][1] += b["tgt_slot"].numel()


# ---------------------------------------------------------------------------
# Extraction: one pass over a batch list -> all 4 taps (+ optionally the
# state-zeroing localization accuracy over that same batch list)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_all(arch: str, model, rig, batches: list, pools, want_loc: bool) -> dict:
    taps = {k: [] for k in TAP_NAMES}
    targets = []
    loc = {"both_intact": [0.0, 0], "s0_zeroed": [0.0, 0], "s1_zeroed": [0.0, 0]} if want_loc else None

    for b in batches:
        _, final_states = model(b["token_ids"], return_states=True)
        assert len(final_states) == 2, f"{arch}: localization assumes exactly 2 layers, got {len(final_states)}"
        answer_ids = torch.gather(b["entity_ids"], 1, b["tgt_slot"]).reshape(-1)

        logits_intact, hidden_intact, q_true = _continuation_pass(
            arch, model, final_states, b["query_tokens"], pools.buffer_id, capture_q1=True)

        if want_loc:
            _accum_loc(loc, "both_intact", logits_intact, b)
            for mode, idx in (("s0_zeroed", 0), ("s1_zeroed", 1)):
                states = list(final_states)
                states[idx] = torch.zeros_like(states[idx])
                assert bool((states[idx] == 0).all()), f"{mode}: zeroing did not actually zero the state"
                logits_z, _, _ = _continuation_pass(arch, model, states, b["query_tokens"],
                                                     pools.buffer_id, capture_q1=False)
                _accum_loc(loc, mode, logits_z, b)

        tap_i = AUDITED_TAP[arch](model, b["token_ids"], b["query_tokens"])
        q0 = _q_pathway_via_block(model, 0, b["query_tokens"])
        tap_ii = _combine_state_query(arch, final_states[0], q0)
        tap_iii = _combine_state_query(arch, final_states[-1], q_true)
        tap_iv = hidden_intact

        for name, tap in zip(TAP_NAMES, (tap_i, tap_ii, tap_iii, tap_iv)):
            Bn, Q, D = tap.shape
            taps[name].append(tap.reshape(Bn * Q, D).float())
        targets.append(rig.T_val[answer_ids].float())

    out = {"taps": {k: torch.cat(v) for k, v in taps.items()}, "targets": torch.cat(targets)}
    if want_loc:
        out["loc"] = {m: {"accuracy": v[0] / max(1, v[1]), "n": v[1]} for m, v in loc.items()}
    return out


def _ridge_table(Xf, Yf, Xe, Ye, device) -> dict:
    best = None
    for lam, W in ridge_fit(Xf, Yf).items():
        st = cos_stats(ridge_pred(W, Xe), Ye)
        if best is None or st["cos_mean"] > best[1]["cos_mean"]:
            best = (lam, st)
    perm = torch.randperm(Xf.shape[0], device=device)
    Wsh = ridge_fit(Xf, Yf[perm], lambdas=(best[0],))[best[0]]
    shuffled = cos_stats(ridge_pred(Wsh, Xe), Ye)
    return {"lambda": best[0], **best[1],
            "shuffled_control_cos_mean": shuffled["cos_mean"],
            "gap_vs_shuffled_cos_mean": best[1]["cos_mean"] - shuffled["cos_mean"]}


def run_arm(arch: str, ckpt_path: str, device: str, n_fit_batches: int) -> dict:
    t0 = time.time()
    model, rig, doc = load_h2h_checkpoint(ckpt_path, device)
    assert doc["task"].startswith("task1") and doc.get("K", 32) == 32, \
        f"{arch}: expected the round-3 task1-K32 checkpoint, got task={doc.get('task')} K={doc.get('K')}"
    pools = get_pools(device)
    cfg_eval = task_cfg("task1_calib", 32, n_query=None)
    hop_set = tuple(cfg_eval.H_train)

    eval_batches = make_eval_episodes(cfg_eval, pools, device, hop_set)      # pinned EVAL_SEED set
    gen = torch.Generator(device=device)
    gen.manual_seed(DIAG_FIT_SEED)
    fit_batches = [sample_batch_rd(cfg_eval, GRAMMAR_BATCH, gen, hop_set, pools, device=device)
                   for _ in range(n_fit_batches)]

    fit = _extract_all(arch, model, rig, fit_batches, pools, want_loc=False)
    ev = _extract_all(arch, model, rig, eval_batches, pools, want_loc=True)

    tap_table = {}
    for name in TAP_NAMES:
        Xf, Yf = fit["taps"][name].to(device), fit["targets"].to(device)
        Xe, Ye = ev["taps"][name].to(device), ev["targets"].to(device)
        tap_table[name] = _ridge_table(Xf, Yf, Xe, Ye, device)

    return {"arch": arch, "ckpt": ckpt_path, "chance_episode": 1.0 / 32,
            "localization": ev["loc"], "tap_variants": tap_table,
            "n_fit_points": fit["taps"][TAP_NAMES[0]].shape[0],
            "n_eval_points": ev["taps"][TAP_NAMES[0]].shape[0],
            "wall_s": time.time() - t0}


# ---------------------------------------------------------------------------
# CPU selftest -- plumbing only (real checkpoints/GPU are box-only)
# ---------------------------------------------------------------------------

class _FakePools:
    """buffer_id is the only field _continuation_pass/_extract_all read from a real
    `EntityPools` -- stubbed here so the selftest never needs build_entity_pools()'s
    tokenizer load (keeps the CPU selftest fast and decoupled from grammar_rd entirely)."""
    buffer_id = 0


def _synthetic_batch(vocab_size: int, B: int = 2, K: int = 8, Q: int = 3, qlen: int = 5,
                     ctx_len: int = 128) -> dict:
    """Fabricated episode dict carrying the exact fields _continuation_pass / _accum_loc /
    AUDITED_TAP read (token_ids/query_tokens/entity_ids/tgt_slot), at ctx_len==_MIN_KERNEL_T
    (128) so the CONTENDER's real DeltaNetLMMixer.forward length assert is satisfied --
    mirrors h2h_cell_train_rd.py's OWN established convention for contender-specific CPU
    selftests (`bind_tokens = torch.randint(0, vocab, (2, _MIN_KERNEL_T))`) rather than
    routing through grammar_rd, whose K=8 template is well under 128 tokens and would trip
    that exact assert for the contender arm specifically (ablation/transformer have no such
    floor, but this keeps both arches on one uniform, symmetric fixture)."""
    g = torch.Generator().manual_seed(0)
    return {
        "token_ids": torch.randint(0, vocab_size, (B, ctx_len), generator=g),
        "query_tokens": torch.randint(0, vocab_size, (B, Q, qlen), generator=g),
        "entity_ids": torch.randint(0, vocab_size, (B, K), generator=g),
        "tgt_slot": torch.randint(0, K, (B, Q), generator=g),
    }


def mode_selftest() -> int:
    failures = []

    def rep(item, ok, detail=""):
        print(f"[{item}] {'PASS' if ok else 'FAIL'}{(' -- ' + detail) if detail else ''}", flush=True)
        if not ok:
            failures.append(item)

    device = "cpu"
    pools = _FakePools()
    vocab_size = 300
    batches = [_synthetic_batch(vocab_size) for _ in range(2)]

    models = {a: build_arm_model(a, vocab_size, seed=i, device=device)
              for i, a in enumerate(("contender", "ablation"))}

    # 1. shapes: bind-phase forward yields exactly 2 layer states, both present.
    fs_by_arch = {}
    ok1 = True
    for arch, m in models.items():
        _, final_states = m(batches[0]["token_ids"], return_states=True)
        ok1 = ok1 and len(final_states) == 2 and all(s is not None for s in final_states)
        fs_by_arch[arch] = final_states
    rep("selftest 1: bind-phase forward yields 2 non-None layer states (both arms)", ok1)

    # 2. wiring equivalence: _continuation_pass(capture_q1=False) is BIT-IDENTICAL to the
    #    imported, unmodified _recurrent_continuation_answer_logits on the same inputs.
    ok2 = True
    for arch, m in models.items():
        b = batches[0]
        ref = _recurrent_continuation_answer_logits(arch, m, fs_by_arch[arch], b["query_tokens"],
                                                     pools.buffer_id)
        mine, _, _ = _continuation_pass(arch, m, fs_by_arch[arch], b["query_tokens"],
                                        pools.buffer_id, capture_q1=False)
        ok2 = ok2 and torch.allclose(ref, mine, atol=1e-6, rtol=1e-6)
    rep("selftest 2: _continuation_pass reproduces _recurrent_continuation_answer_logits exactly "
        "(both arms)", ok2)

    # 3. hook fires: capture_q1=True yields a finite, correctly-shaped q_true (both arms).
    ok3 = True
    for arch, m in models.items():
        b = batches[0]
        B, Q, _ = b["query_tokens"].shape
        _, _, q_true = _continuation_pass(arch, m, fs_by_arch[arch], b["query_tokens"],
                                          pools.buffer_id, capture_q1=True)
        ok3 = ok3 and q_true is not None and torch.isfinite(q_true).all() \
            and q_true.shape[:2] == (B, Q)
    rep("selftest 3: block-1 q_conv1d hook fires, q_true finite + correctly shaped (both arms)", ok3)

    # 4. zeroing has teeth: an all-zero state combined via EITHER formula (matvec or
    #    Hadamard) must produce an EXACTLY zero tap -- an exact, not tolerance-based, check
    #    (CLAUDE.md's own "structural checks need exact thresholds" rule). Uses a shape-matched
    #    SYNTHETIC nonzero state, not the model's own forward-computed one: under the CPU stub
    #    `_stub_chunk_delta_rule` always returns an all-zero final_state for the contender
    #    (disclosed limitation, h2h_fla_stub_rd.py's own docstring) -- real nonzero S_T is
    #    box-only for that arch, exactly like probe_head_rd.py's own smoke_3 registration. This
    #    item tests the COMBINE FORMULA's code path (real shapes, real ops), independent of
    #    whether the fixture state itself is trainable/nonzero this run.
    ok4 = True
    for arch, m in models.items():
        S_real_shape = fs_by_arch[arch][0]
        S_nonzero = torch.randn_like(S_real_shape) + 1.0     # shape-matched, guaranteed nonzero
        q0 = _q_pathway_via_block(m, 0, batches[0]["query_tokens"])
        tap_nonzero = _combine_state_query(arch, S_nonzero, q0)
        tap_zeroed = _combine_state_query(arch, torch.zeros_like(S_nonzero), q0)
        ok4 = ok4 and bool((tap_zeroed == 0).all()) and bool(tap_nonzero.abs().max() > 0)
    rep("selftest 4: zeroing a (shape-matched synthetic) nonzero state produces an EXACTLY "
        "zero combined tap, both combine formulas (matvec + Hadamard); the nonzero fixture "
        "itself produces a nonzero tap (confirms the check isn't vacuously zero either way)", ok4)

    # 5. ridge harness has teeth: fit on STRUCTURED synthetic data (Y = X @ W_true, no noise)
    #    must recover near-perfect cosine; a shuffled-target control on the SAME X must not.
    torch.manual_seed(5)
    N, Din, Dout = 400, 17, 6
    X = torch.randn(N, Din)
    W_true = torch.randn(Din, Dout)
    Y = X @ W_true
    Xf5, Yf5, Xe5, Ye5 = X[:300], Y[:300], X[300:], Y[300:]
    real = _ridge_table(Xf5, Yf5, Xe5, Ye5, "cpu")
    ok5 = (real["cos_mean"] > 0.99 and real["shuffled_control_cos_mean"] < 0.3
           and real["gap_vs_shuffled_cos_mean"] > 0.5
           and all(k in real for k in ("rf@0.5", "rf@0.7", "rf@0.9")))
    rep("selftest 5: ridge harness recovers a known-linear synthetic mapping (cos>0.99) and "
        "clears its own shuffled-tap negative control (has teeth, not merely runs)", ok5,
        f"cos_mean={real['cos_mean']:.4f} shuffled={real['shuffled_control_cos_mean']:.4f}")

    # 6. FULL integration: _extract_all (all 4 taps + localization) end-to-end on synthetic
    #    episodes/rig, both arches -- catches wiring bugs in run_arm's own call graph before
    #    spending real box time (a fake ProbeRig stand-in; only its .T_val field is read).
    class _FakeRig:
        def __init__(self, vocab_size, value_dim=6):
            self.T_val = torch.randn(vocab_size, value_dim)
    ok6 = True
    for arch, m in models.items():
        rig6 = _FakeRig(vocab_size)
        out = _extract_all(arch, m, rig6, batches, pools, want_loc=True)
        ok6 = (ok6 and set(out["taps"]) == set(TAP_NAMES)
              and all(torch.isfinite(v).all() for v in out["taps"].values())
              and out["targets"].shape[0] == out["taps"][TAP_NAMES[0]].shape[0]
              and set(out["loc"]) == {"both_intact", "s0_zeroed", "s1_zeroed"}
              and all(0.0 <= v["accuracy"] <= 1.0 for v in out["loc"].values()))
        Xf6, Yf6 = out["taps"]["iii_s1_qtrue"], out["targets"]
        tbl = _ridge_table(Xf6, Yf6, Xf6, Yf6, "cpu")
        ok6 = ok6 and all(k in tbl for k in ("cos_mean", "rf@0.9", "shuffled_control_cos_mean"))
    rep("selftest 6: full _extract_all (4 taps + state-zeroing localization) + _ridge_table "
        "integration runs end-to-end on synthetic episodes, both arches, all outputs finite "
        "and correctly shaped", ok6)

    print(f"\n{'ALL PASS' if not failures else f'FAILURES: {failures}'}", flush=True)
    return 0 if not failures else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default="results/h2h_rung1/tap_localization")
    ap.add_argument("--arms", nargs="+", default=["contender", "ablation"])
    ap.add_argument("--n-fit-batches", type=int, default=N_FIT_BATCHES_DEFAULT)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return mode_selftest()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    report = {}
    for arch in args.arms:
        r = run_arm(arch, CKPT_PATHS[arch], args.device, args.n_fit_batches)
        report[arch] = r
        path = os.path.join(args.out_dir, f"tap_localization_{arch}.json")
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        print(f"[{arch}] DONE in {r['wall_s']:.1f}s -> {path}", flush=True)
        print(json.dumps({"localization": r["localization"], "tap_variants": r["tap_variants"]},
                         indent=2), flush=True)
    wall = time.time() - t0
    report["_meta"] = {"wall_s": wall, "gpu_h": wall / 3600.0}
    with open(os.path.join(args.out_dir, "tap_localization_SUMMARY.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"TOTAL wall {wall:.1f}s (~{wall / 3600:.4f} GPU-h)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
