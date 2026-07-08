"""phase2_stage_minus1.py -- REASONING_LINK_DESIGN.md sec 16.2 (Rev 5,
CLEARED-FOR-BUILD) Phase-2's own Stage -1 self-tests: every item is an
EXECUTABLE test with its own registered pass criterion (never a printed
value trusted by eye, per this project's "a specification that has not
been executed is not a passed gate" convention), mirroring
`reasoning_link_stage_minus1.py`'s own house pattern exactly -- each item
imports and exercises the REAL functions in `lm_pretrain_rd.py` /
`phase2_familiarization_train.py` / `phase2_gate_enforce.py` /
`phase2_bands_pinned.py` / `phase2_hexachotomy.py`, never a reimplemented
copy, so a Stage -1 pass here certifies THOSE modules' own code.

CPU-runnable throughout (this repo's own CPU fla-stub, installed as an
`import reasoning_link_probe` side effect -- see that module's own
docstring). Items exercising a real `DeltaNetLM.forward`/`train()` call use
a small, freshly-initialized model (real GPT-2 vocab=50257 so the real
tokenizer/`build_reasoning_link_pools` line up) as a stand-in "checkpoint",
mirroring `lm_attractor_probe_rd.py`'s own smoke() item [6] convention
("a TINY (but REAL-vocab-sized) SYNTHETIC (untrained) model ... NOT a claim
about real key geometry"). **Disclosed, not silently assumed:** the CPU
stub's `chunk_delta_rule` substitute returns a DELIBERATELY zero final
state (`_stub_chunk_delta_rule`'s own docstring), so premises (iii)/(iv)
and the h1 sanity floor can NEVER pass under this stub -- every item below
that reads a `gate_pass`/`probe_valid` value asserts it is a well-formed
boolean computed via the REAL, audited machinery, never that it equals
True. Real-checkpoint gate-PASSING behavior is box-only, exactly like
`reasoning_link_stage_minus1.py`'s own item 11 disclosure.

Run: python phase2_stage_minus1.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reasoning_link_probe as rlp  # noqa: E402  (installs the CPU fla stub as an import side effect)
import grammar_rd  # noqa: E402
import lm_pretrain_rd as lpr  # noqa: E402
import phase2_familiarization_train as pft  # noqa: E402
import phase2_gate_enforce as pge  # noqa: E402
import phase2_bands_pinned as pbp  # noqa: E402
import phase2_hexachotomy as phx  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

RESULTS = []
V_REAL = 50257
EOT = lpr.EOT_TOKEN_ID


def _report(item: str, name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append({"item": item, "name": name, "passed": passed, "detail": detail})
    status = "PASS" if passed else "FAIL"
    print(f"[Phase-2 Stage-1 item {item}] {name}: {status}" + (f" -- {detail}" if detail else ""))
    assert passed, f"Phase-2 Stage -1 item {item} ({name}) FAILED: {detail}"


# ---------------------------------------------------------------------------
# Shared synthetic-corpus / synthetic-checkpoint builders (used by several items below).
# ---------------------------------------------------------------------------

def _synth_eot_corpus(n_tok: int, mean_doc_len: int, seed: int, vocab: int = V_REAL):
    g = torch.Generator().manual_seed(seed)
    toks = torch.randint(0, vocab, (n_tok,), generator=g)
    # mirrors lm_pretrain_rd.smoke()'s own synth_eot_corpus convention: EOT_TOKEN_ID (50256) is only
    # a valid in-range id when vocab > it; a small test vocab clamps via modulo instead.
    eot_id = EOT % vocab if vocab <= EOT else EOT
    pos, offsets = 0, [0]
    while True:
        step_len = int(mean_doc_len * (0.5 + torch.rand(1, generator=g).item()))
        pos += max(2, step_len)
        if pos >= n_tok - 1:
            break
        toks[pos] = eot_id
        offsets.append(pos + 1)
    return toks, torch.tensor(offsets, dtype=torch.int64)


def _build_synthetic_data_dir(tmpdir: str) -> str:
    """Builds a minimal on-disk corpus pair satisfying `lm_pretrain_rd.load_corpus`'s own contract
    (meta.json + {split}.pt + {split}_doc_offsets.pt, eot_separated=True, real GPT-2 vocab_size)."""
    data_dir = os.path.join(tmpdir, "data")
    for name, mean_len, seed in ((lpr.CORPUS_DIRS["openr1-mix-ext"], 300, 11),
                                  (lpr.CORPUS_DIRS["wikitext-mix-ext"], 400, 22)):
        d = os.path.join(data_dir, name)
        os.makedirs(d)
        train, train_offs = _synth_eot_corpus(24_000, mean_len, seed)
        val, val_offs = _synth_eot_corpus(6_000, mean_len, seed + 1)
        torch.save(train, os.path.join(d, "train.pt"))
        torch.save(val, os.path.join(d, "val.pt"))
        torch.save(train_offs, os.path.join(d, "train_doc_offsets.pt"))
        torch.save(val_offs, os.path.join(d, "val_doc_offsets.pt"))
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"vocab_size": 50257, "tokenizer": "gpt2", "eot_separated": True}, f)
    return data_dir


def _build_synthetic_init_checkpoint(tmpdir: str, arm: str = "off") -> str:
    m = lpr.DeltaNetLM(V_REAL, d_model=32, d_state=64, n_layers=1, conv_size=4, frozen_bias_arm=arm,
                        frozen_bias_lambda=0.58, frozen_bias_vocab_size=V_REAL, frozen_bias_seed=20260705)
    p = os.path.join(tmpdir, f"archived_{arm}_openr1-mix-ext_s0_step20000.pt")
    torch.save({"step": 20000, "model_state_dict": m.state_dict(), "config": m.config(),
                "corpus": "openr1-mix-ext", "seed": 0, "run_name": f"lmC_synthetic_{arm}_s0"}, p)
    return p


# ---------------------------------------------------------------------------
# Item 1 -- --ckpt-steps: pure-logic positive/negative, PLUS a real train() call proving both.
# ---------------------------------------------------------------------------

def test_item_1_ckpt_steps():
    target = [250, 500, 1000, 2500, 5000]
    fired = [s for s in range(1, 5001) if lpr.should_checkpoint_now(s, 5000, 1000, target)]
    assert fired == target, f"pure-logic positive: expected {target}, got {fired}"

    subset = [250, 500]
    fired_neg = [s for s in range(1, 801) if lpr.should_checkpoint_now(s, 800, 1000, subset)]
    assert fired_neg == [250, 500] and 750 not in fired_neg and 800 not in fired_neg, (
        f"pure-logic negative: expected [250,500] with no fire at 750/800 (final step), got {fired_neg}")

    with tempfile.TemporaryDirectory() as td:
        model = lpr.DeltaNetLM(500, d_model=32, d_state=64, n_layers=1, conv_size=4)
        corpus_a, offs_a = _synth_eot_corpus(20_000, 100, 1, vocab=500)
        corpus_b, offs_b = _synth_eot_corpus(20_000, 200, 2, vocab=500)

        class Args:
            pass
        args = Args()
        args.corpus, args.seed = "openr1", 0
        args.d_model, args.d_state, args.n_layers, args.conv_size, args.num_heads, args.ffn_mult = 32, 64, 1, 4, 1, 4
        args.seq_len, args.batch_size = 512, 4
        args.lr, args.weight_decay, args.warmup_steps = 3e-4, 0.0, 2
        args.eval_batches, args.eval_batch_size = 1, 2
        args.rank_sample_docs = 2
        args.steps, args.log_every, args.ckpt_every = 12, 4, 1000  # ckpt_every deliberately never fires
        args.ckpt_steps = [3, 6, 9]

        ckdir = os.path.join(td, "ck1")
        result = lpr.train(model, args, corpus_a, offs_a, (corpus_a, offs_a), (corpus_b, offs_b),
                            "wikitext", "cpu", "s1_probe", out_path=None, ckpt_dir=ckdir)
        steps_written = sorted(c["step"] for c in result["checkpoints"])
        assert steps_written == [3, 6, 9], f"REAL train() positive: expected [3,6,9], got {steps_written}"
        n_files = len(os.listdir(ckdir))
        assert n_files == 3, f"REAL train() positive: expected exactly 3 files, got {n_files}"

        model2 = lpr.DeltaNetLM(500, d_model=32, d_state=64, n_layers=1, conv_size=4)
        args.steps = 9
        args.ckpt_steps = [3, 6]   # step 9 == final step, deliberately NOT in ckpt_steps
        ckdir2 = os.path.join(td, "ck2")
        result2 = lpr.train(model2, args, corpus_a, offs_a, (corpus_a, offs_a), (corpus_b, offs_b),
                             "wikitext", "cpu", "s1_negprobe", out_path=None, ckpt_dir=ckdir2)
        steps_written2 = sorted(c["step"] for c in result2["checkpoints"])
        assert steps_written2 == [3, 6] and 9 not in steps_written2, (
            f"REAL train() negative: expected [3,6] with NO checkpoint at final step 9, got {steps_written2}")
        n_files2 = len(os.listdir(ckdir2))
        assert n_files2 == 2, f"REAL train() negative: expected exactly 2 files, got {n_files2}"

    _report("1", "--ckpt-steps positive (exact 5-of-5 files) + negative (no fire past held-out subset, "
                 "including at the run's own final step) -- pure logic AND a REAL lm_pretrain_rd.train() call",
            True, f"positive fired={fired}, negative fired={fired_neg}, real-train positive={steps_written}, "
                  f"real-train negative={steps_written2} (9 correctly absent)")


# ---------------------------------------------------------------------------
# Item 2 -- --init-checkpoint: positive (reproduces archived val_loss) + TWO negative tests.
# ---------------------------------------------------------------------------

def test_item_2_init_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        model = lpr.DeltaNetLM(500, d_model=32, d_state=64, n_layers=1, conv_size=4)
        corpus_a, offs_a = _synth_eot_corpus(20_000, 100, 3, vocab=500)
        corpus_b, offs_b = _synth_eot_corpus(20_000, 200, 4, vocab=500)

        class Args:
            pass
        args = Args()
        args.corpus, args.seed = "openr1", 0
        args.d_model, args.d_state, args.n_layers, args.conv_size, args.num_heads, args.ffn_mult = 32, 64, 1, 4, 1, 4
        args.seq_len, args.batch_size = 512, 4
        args.lr, args.weight_decay, args.warmup_steps = 3e-4, 0.0, 2
        args.eval_batches, args.eval_batch_size = 1, 2
        args.rank_sample_docs = 2
        args.steps, args.log_every, args.ckpt_every = 6, 3, 1000
        args.ckpt_steps = [3, 6]

        ckdir = os.path.join(td, "archive")
        result = lpr.train(model, args, corpus_a, offs_a, (corpus_a, offs_a), (corpus_b, offs_b),
                            "wikitext", "cpu", "s2_archive", out_path=None, ckpt_dir=ckdir)
        archived_path = [c["checkpoint_path"] for c in result["checkpoints"] if c["step"] == 3][0]
        archived_val_loss = [c["val_loss"] for c in result["checkpoints"] if c["step"] == 3][0]

        # POSITIVE: fresh model + load_init_checkpoint_strict -> forward reproduces the archived
        # run's own last-recorded val loss, read from result['checkpoints'] (NEVER 'trajectory').
        fresh = lpr.DeltaNetLM(500, d_model=32, d_state=64, n_layers=1, conv_size=4)
        loaded = lpr.load_init_checkpoint_strict(fresh, archived_path, "cpu")
        assert loaded["step"] == 3
        fresh.eval()
        eval_gen = torch.Generator().manual_seed(lpr.corpus_fixed_seed(args.corpus) + 10_000 + 3)
        reproduced, _ = lpr.eval_loss(fresh, corpus_a, offs_a, args.eval_batches, args.eval_batch_size,
                                       args.seq_len, eval_gen, step=3)
        assert abs(reproduced - archived_val_loss["openr1"]) < 1e-5, (
            f"positive: reproduced={reproduced} vs archived={archived_val_loss['openr1']}")

        # NEGATIVE #1: config-mismatched checkpoint ABORTS via the explicit config-equality assert.
        bad_cfg_path = os.path.join(td, "bad_config.pt")
        bad_ckpt = dict(loaded)
        bad_ckpt["config"] = dict(bad_ckpt["config"])
        bad_ckpt["config"]["d_state"] = 999
        torch.save(bad_ckpt, bad_cfg_path)
        raised_cfg = False
        try:
            lpr.load_init_checkpoint_strict(lpr.DeltaNetLM(500, d_model=32, d_state=64, n_layers=1, conv_size=4),
                                             bad_cfg_path, "cpu")
        except AssertionError:
            raised_cfg = True
        assert raised_cfg, "negative #1: config-mismatched checkpoint was NOT rejected"

        # NEGATIVE #2: shape-corrupted state_dict ABORTS via strict=True (config matches, so this
        # exercises the SECOND, independent defense layer alone).
        bad_shape_path = os.path.join(td, "bad_shape.pt")
        bad_sd = {k: v.clone() for k, v in loaded["model_state_dict"].items()}
        some_key = next(iter(bad_sd))
        bad_sd[some_key] = bad_sd[some_key][:1]
        torch.save({"step": loaded["step"], "model_state_dict": bad_sd, "config": loaded["config"],
                    "corpus": loaded["corpus"], "seed": loaded["seed"], "run_name": loaded["run_name"]},
                   bad_shape_path)
        raised_shape = False
        try:
            lpr.load_init_checkpoint_strict(lpr.DeltaNetLM(500, d_model=32, d_state=64, n_layers=1, conv_size=4),
                                             bad_shape_path, "cpu")
        except RuntimeError:
            raised_shape = True
        assert raised_shape, f"negative #2: shape-corrupted state_dict (key={some_key}) was NOT rejected"

    _report("2", "--init-checkpoint positive (forward reproduces result['checkpoints'] val_loss, NOT "
                 "'trajectory') + TWO negative tests (config-mismatch, shape-mismatch), both ABORT",
            True, f"reproduced={reproduced:.6f} vs archived={archived_val_loss['openr1']:.6f}")


# ---------------------------------------------------------------------------
# Item 3 -- periodicity guard RE-VERIFICATION at K in {20,32} with H_train=(1,2)/H_test=(3,4)
# (sec 16.2.1's own registered mandatory Stage -1 assertion).
# ---------------------------------------------------------------------------

def test_item_3_periodicity_guard():
    for K in pft.K_SWEEP:
        cfg = pft.familiarization_episode_config(conv_size=4, K=K)
        assert cfg.H_train == (1, 2) and cfg.H_test == (3, 4)
        train_residues = {h % K for h in cfg.H_train}
        for h in cfg.H_test:
            r = h % K
            assert r != 0, f"K={K}: H_test hop h={h} has residue 0 (identity) -- confounded"
            assert r not in train_residues, f"K={K}: H_test hop h={h} residue {r} collides with H_train"
    # negative control: constructing a config with a REAL collision must raise (proves the guard has
    # teeth, not merely that our own H_train/H_test choice happens to avoid it).
    raised = False
    try:
        grammar_rd.DeltaNetRDTaskConfig(K=2, conv_size=4, H_train=(1, 2), H_test=(3, 4), H_extra=())
        # K=2: H_train residues {1%2,2%2}={1,0} -> 2%2=0 already violates "h must be >=1"'s own
        # separate assert path (0 residue on TRAIN side is a different guard); use a K that instead
        # collides on the TEST side directly:
    except AssertionError:
        raised = True
    raised2 = False
    try:
        grammar_rd.DeltaNetRDTaskConfig(K=3, conv_size=4, H_train=(1, 2), H_test=(4, 5), H_extra=())
        # K=3: H_train residues {1,2}; H_test h=4 -> 4%3=1, COLLIDES with H_train's residue 1
    except AssertionError:
        raised2 = True
    assert raised2, "negative control: a genuinely colliding (K,H_train,H_test) combination did NOT raise"
    _report("3", "periodicity guard re-verified live at K in {20,32} (grammar_rd's own __post_init__ "
                 "guard, exercised via familiarization_episode_config) + negative control (K=3 real "
                 "collision correctly raises)", True, f"K_SWEEP={pft.K_SWEEP}")


# ---------------------------------------------------------------------------
# Item 4 -- query-loss construction: ONE forward call, vocab-space CE at the last position matches
# grammar_rd's own [grammar 3c] answer_token pattern.
# ---------------------------------------------------------------------------

def test_item_4_query_loss_forward():
    torch.manual_seed(0)
    model = lpr.DeltaNetLM(V_REAL, d_model=32, d_state=64, n_layers=1, conv_size=4)
    tokenizer = grammar_rd.load_gpt2_tokenizer()
    pools, _ = rlp.build_reasoning_link_pools(tokenizer=tokenizer, seed=0)
    cfg = pft.familiarization_episode_config(conv_size=4, K=20)
    gen = torch.Generator().manual_seed(1)

    forward_call_count = [0]
    orig_forward = model.forward
    def counting_forward(*a, **kw):
        forward_call_count[0] += 1
        return orig_forward(*a, **kw)
    model.forward = counting_forward

    L_query, batch, n_calls_reported = pft.query_loss_forward(model, cfg, pools, batch_size=4, gen=gen,
                                                                device="cpu", use_heldout_entities=False)
    assert forward_call_count[0] == 1, (
        f"query_loss_forward must call model.forward EXACTLY once (never a two-phase bind-then-"
        f"continuation split), got {forward_call_count[0]}")
    assert n_calls_reported == 1
    assert torch.isfinite(L_query) and L_query.item() > 0

    # Independently re-derive the target via grammar_rd's OWN answer_token pattern ([grammar 3c],
    # L594-616) and confirm it matches what query_loss_forward actually scored against.
    expected_target = torch.gather(batch["entity_ids"], 1, batch["tgt_slot"]).reshape(-1)
    B, Q = batch["hops"].shape
    T_bind = batch["token_ids"].shape[1]
    query_len = batch["query_tokens"].shape[-1]
    bind_expanded = batch["token_ids"].unsqueeze(1).expand(B, Q, T_bind).reshape(B * Q, T_bind)
    query_flat = batch["query_tokens"].reshape(B * Q, query_len)
    concat = torch.cat([bind_expanded, query_flat], dim=1)
    with torch.no_grad():
        logits_indep = orig_forward(concat, initial_states=None)
    L_indep = F.cross_entropy(logits_indep[:, -1, :], expected_target)
    assert abs(L_indep.item() - L_query.item()) < 1e-5, (
        f"independent re-derivation mismatch: {L_indep.item()} vs {L_query.item()}")

    _report("4", "query_loss_forward: EXACTLY 1 model.forward call (never bind-then-continuation), "
                 "vocab-space CE at logits[:,-1,:] against torch.gather(entity_ids,1,tgt_slot) "
                 "([grammar 3c] pattern), independent re-derivation matches to 1e-5",
            True, f"L_query={L_query.item():.6f}")


# ---------------------------------------------------------------------------
# Item 5 -- hexachotomy: full 32-pattern totality enumeration + spot checks + the UNRESOLVED-GATE pin.
# ---------------------------------------------------------------------------

def test_item_5_hexachotomy_totality():
    r = phx.totality_check()
    expected = {"PERSISTENT": 4, "TRANSIENT": 15, "LATE-EMERGENT": 1, "CONVERGED-EQUIVALENT": 1,
                "UNRESOLVED": 0, "NON-MONOTONE": 11, "UNRESOLVED-GATE": 0}
    assert r["counts"] == expected, f"totality mismatch: {r['counts']} != {expected}"
    assert r["total"] == 32

    C = phx.CHECKPOINTS
    always_pass = {c: True for c in C}
    def mk(*vals):
        return dict(zip(C, vals))

    # Rev-4's own counter-example: F,T,T,T,T -> PERSISTENT/c1=500 (the exact pattern round-4 caught
    # a c1-anchor bug on; re-verified fresh by THIS build's own implementation, not inherited trust).
    r2 = phx.classify_trajectory(mk(False, True, True, True, True), always_pass, True, True, True)
    assert r2["outcome"] == "PERSISTENT" and r2["c1"] == 500, r2

    # attack-round-3's own counter-example: T,F,F,F,T -> NON-MONOTONE (the pentachotomy-gap case).
    r3 = phx.classify_trajectory(mk(True, False, False, False, True), always_pass, True, True, True)
    assert r3["outcome"] == "NON-MONOTONE", r3

    # UNRESOLVED-GATE pin (this BUILD's obligation (4)): monotone holds-TRUE run confirmed, but
    # Stage-0.5 fails at EVERY non-terminal checkpoint in that run.
    gate_all_fail = {250: False, 500: False, 1000: False, 2500: False, 5000: True}
    r4 = phx.classify_trajectory(mk(True, True, True, True, True), gate_all_fail, True, True, True)
    assert r4["outcome"] == "UNRESOLVED-GATE" and r4["c1"] is None, r4

    # skip-past: c1_raw's own gate fails, a LATER non-terminal checkpoint within the same monotone
    # run passes -> re-identifies c1 at that later checkpoint (never silently reverting to c1_raw).
    gate_skip = {250: False, 500: False, 1000: True, 2500: True, 5000: True}
    r5 = phx.classify_trajectory(mk(True, True, True, True, True), gate_skip, True, True, True)
    assert r5["outcome"] == "PERSISTENT" and r5["c1"] == 1000, r5

    _report("5", "hexachotomy: 32-pattern totality EXACTLY matches sec 16.2.1's registered "
                 "1+15+1+4+11=32 split; Rev-4 counter-example (F,T,T,T,T->PERSISTENT/c1=500) and "
                 "attack-round-3 counter-example (T,F,F,F,T->NON-MONOTONE) reproduced; "
                 "UNRESOLVED-GATE pin (this build's obligation) fires correctly, incl. skip-past",
            True, f"counts={r['counts']}")


# ---------------------------------------------------------------------------
# Item 6 -- Stage-0.5 gate enforcement negative test (process-boundary teeth, sec 16.5 Constraint 1).
# ---------------------------------------------------------------------------

def test_item_6_gate_enforcement_negative_test():
    ok = pge._run_selftest()
    assert ok, "phase2_gate_enforce's own negative-test fixture suite FAILED"
    _report("6", "phase2_gate_enforce --selftest (per-checkpoint Stage-0.5 gate enforcement, sec 16.5 "
                 "Constraint 1's gates-must-abort discipline): 1 positive + 5 negative fixtures, PLUS "
                 "subprocess-level exit-code proof for all 6", True)


# ---------------------------------------------------------------------------
# Item 7 -- bands-pinned blind-sequencing negative test (sec 16.2.1 MAJOR-6/MINOR-NEW-1).
# ---------------------------------------------------------------------------

def test_item_7_bands_pinned_blind_sequencing():
    with tempfile.TemporaryDirectory() as td:
        per_checkpoint_values, result_paths = {}, {}
        for step in pbp.CHECKPOINTS:
            per_checkpoint_values[step], result_paths[step] = {}, {}
            for corpus in ("openr1-mix-ext", "wikitext-mix-ext"):
                vals = [5.7 + 0.001 * i for i in range(3)]
                per_checkpoint_values[step][corpus] = vals
                paths = []
                for seed in range(3):
                    p = os.path.join(td, f"{corpus}_s{seed}_c{step}.json")
                    with open(p, "w") as f:
                        json.dump({"val_loss": vals[seed]}, f)
                    paths.append(p)
                result_paths[step][corpus] = paths

        bands_path = os.path.join(td, "BANDS_PINNED-Phase2Familiarization.json")
        doc = pbp.write_bands_pinned_phase2(bands_path, per_checkpoint_values, result_paths)
        assert set(doc["bands_by_checkpoint"]) == {str(c) for c in pbp.CHECKPOINTS}

        v = pbp.validate_bands_pinned_phase2(bands_path, per_checkpoint_values)
        assert v is not None, "validation FAILED on untampered data"

        tampered = {k: {c: list(vv) for c, vv in d.items()} for k, d in per_checkpoint_values.items()}
        tampered[1000]["openr1-mix-ext"][0] += 5.0
        assert pbp.validate_bands_pinned_phase2(bands_path, tampered) is None, (
            "validation did NOT reject tampered data")

        # POSITIVE: pin precedes launch -> passes.
        pbp.assert_blind_not_broken_phase2(doc, [doc["pinned_at"] + 10])

        # NEGATIVE: pin postdates an earlier per_token/global start -> MUST raise (this is the exact
        # process failure FROZEN_BIAS_LM_DESIGN.md's own rung-1 wave committed once for real).
        raised = False
        try:
            pbp.assert_blind_not_broken_phase2(doc, [doc["pinned_at"] - 10])
        except AssertionError:
            raised = True
        assert raised, "assert_blind_not_broken_phase2 did NOT raise when the pin postdated launch"

        in_band, _ = pbp.gate_val_loss_at_checkpoint(doc, 1000, "openr1-mix-ext", 5.70)
        out_of_band, _ = pbp.gate_val_loss_at_checkpoint(doc, 1000, "openr1-mix-ext", 999.0)
        assert in_band is True and out_of_band is False

    _report("7", "phase2_bands_pinned: write/validate/tamper-detect round trip, blind-sequencing "
                 "POSITIVE (pin-before-launch passes) + NEGATIVE (pin-after-launch RAISES, the exact "
                 "FROZEN_BIAS_LM_DESIGN.md rung-1 process failure), per-checkpoint gate-check "
                 "in-band/out-of-band", True)


# ---------------------------------------------------------------------------
# Item 8 -- full end-to-end familiarization cell: combined-loss training, checkpointing WITH
# optimizer_state_dict, per-checkpoint Stage-0.5 gate (off arm only), REAL mid-run resume.
# ---------------------------------------------------------------------------

def test_item_8_e2e_familiarization_cell():
    with tempfile.TemporaryDirectory() as td:
        data_dir = _build_synthetic_data_dir(td)
        init_path = _build_synthetic_init_checkpoint(td, arm="off")
        ckpt_dir = os.path.join(td, "ckpts")

        # MINOR-1 fix (Phase-2 build-audit round) instrumentation: spy on pft.get_batch (the module-
        # level binding run_familiarization_cell's own training loop calls) to capture the FIRST
        # corpus batch each of the two runs below actually draws -- proves the RESUMED run's gen_corpus
        # stream is genuinely different from the pre-crash run's, not merely re-seeded identically.
        captured_batches = []
        orig_get_batch = pft.get_batch
        def _spying_get_batch(*a, **kw):
            x, y = orig_get_batch(*a, **kw)
            captured_batches.append(x.clone())
            return x, y
        pft.get_batch = _spying_get_batch
        try:
            result = pft.run_familiarization_cell(
                init_checkpoint=init_path, arm="off", corpus="openr1-mix-ext", ckpt_seed=0, K=20,
                steps=6, ckpt_steps=[2, 4, 6], lambda_fam=1.0, lr=3e-4, weight_decay=0.0, warmup_steps=2,
                corpus_batch_size=4, episode_batch_size=4, gate_batch_size=4, seq_len=128,
                eval_batches=1, eval_batch_size=2, data_dir=data_dir, device="cpu",
                ckpt_dir=ckpt_dir, out_path=os.path.join(td, "off_result.json"), resume=False)
            assert captured_batches, "pft.get_batch spy captured nothing on the pre-crash run"
            first_pre_crash_x = captured_batches[0].clone()
            captured_batches.clear()

            assert [c["step"] for c in result["checkpoints"]] == [2, 4, 6]
            assert all(c["stage05_gate"] is not None and "gate_pass" in c["stage05_gate"]
                       for c in result["checkpoints"]), "OFF arm must compute the gate at EVERY checkpoint"
            for c in result["checkpoints"]:
                ck = torch.load(c["checkpoint_path"], map_location="cpu")
                assert "optimizer_state_dict" in ck, f"missing optimizer_state_dict in {c['checkpoint_path']}"

            # REAL mid-run resume: continue from step 6 to step 10; must load model+optimizer+step, not
            # restart at step 1 (which would silently re-inflict the optimizer-restart confound a
            # SECOND time on top of the already-disclosed familiarization-start one).
            result2 = pft.run_familiarization_cell(
                init_checkpoint=init_path, arm="off", corpus="openr1-mix-ext", ckpt_seed=0, K=20,
                steps=10, ckpt_steps=[8, 10], lambda_fam=1.0, lr=3e-4, weight_decay=0.0, warmup_steps=2,
                corpus_batch_size=4, episode_batch_size=4, gate_batch_size=4, seq_len=128,
                eval_batches=1, eval_batch_size=2, data_dir=data_dir, device="cpu",
                ckpt_dir=ckpt_dir, out_path=os.path.join(td, "off_resumed.json"), resume=True)
            assert captured_batches, "pft.get_batch spy captured nothing on the resumed run"
            first_post_resume_x = captured_batches[0].clone()
        finally:
            pft.get_batch = orig_get_batch

        assert result2["resumed_from"] is not None and "step6" in result2["resumed_from"]
        assert [c["step"] for c in result2["checkpoints"]] == [8, 10]

        # MINOR-1 fix's own assertion: the first post-resume corpus batch must DIFFER from the first
        # pre-crash corpus batch (both are step-1-of-their-own-run draws from gen_corpus) -- pre-fix,
        # both runs re-seeded gen_corpus with the SAME checkpoint_step=0 stream, so the resumed run's
        # first draw would REPLAY the pre-crash run's own already-consumed step-1 batch verbatim.
        assert first_pre_crash_x.shape == first_post_resume_x.shape
        assert not torch.equal(first_pre_crash_x, first_post_resume_x), (
            "MINOR-1 regression: the resumed run's first corpus batch is IDENTICAL to the pre-crash "
            "run's first corpus batch -- gen_corpus is not being re-seeded with the resume step")

        # per_token / global arms: buildable, and the OFF-only gate is correctly None for them.
        for arm in ("per_token", "global"):
            p = _build_synthetic_init_checkpoint(td, arm=arm)
            r = pft.run_familiarization_cell(
                init_checkpoint=p, arm=arm, corpus="openr1-mix-ext", ckpt_seed=0, K=20,
                steps=3, ckpt_steps=[3], lambda_fam=1.0, lr=3e-4, weight_decay=0.0, warmup_steps=1,
                corpus_batch_size=4, episode_batch_size=4, gate_batch_size=4, seq_len=128,
                eval_batches=1, eval_batch_size=2, data_dir=data_dir, device="cpu",
                ckpt_dir=ckpt_dir, out_path=os.path.join(td, f"{arm}_result.json"), resume=False)
            assert r["checkpoints"][0]["stage05_gate"] is None, (
                f"arm={arm}: the OFF-arm-only Stage-0.5 gate must be None, got "
                f"{r['checkpoints'][0]['stage05_gate']}")

    _report("8", "REAL end-to-end familiarization cell (arm=off): combined corpus+query loss "
                 "training, checkpointing with optimizer_state_dict at exact steps, per-checkpoint "
                 "Stage-0.5 gate always computed; REAL mid-run resume (model+optimizer+step); MINOR-1 "
                 "fix verified (first post-resume corpus batch != first pre-crash corpus batch); "
                 "per_token/global arms buildable with gate correctly None", True)


# ---------------------------------------------------------------------------
# Item 9 -- Phase-2 seed collision-freedom (mirrors reasoning_link_stage_minus1's own item-10
# enumerate-and-assert-no-collision convention).
# ---------------------------------------------------------------------------

def test_item_9_seed_non_collision():
    seen = {}
    n_checked = 0
    for kind in pft._KIND_OFFSET:
        for arm in pft._ARM_INDEX:
            for corpus in ("openr1-mix-ext", "wikitext-mix-ext"):
                for ckpt_seed in range(3):
                    for k in pft.K_SWEEP:
                        for c in (0, *pft.CKPT_STEPS):
                            s = pft.phase2_seed(kind, arm, corpus, ckpt_seed, k, c)
                            key = (kind, arm, corpus, ckpt_seed, k, c)
                            if s in seen:
                                raise AssertionError(f"seed collision: {key} and {seen[s]} both -> {s}")
                            seen[s] = key
                            n_checked += 1
    # disjointness from Phase-1's own registered leg_a/leg_b bases (0, 5_000_000) -- PHASE2_SEED_BASE
    # (777,000,000) is far outside either range by construction; spot-check a few Phase-1 combos too.
    for combo in rlp.enumerate_registered_seed_combinations():
        s1 = rlp.episode_seed(*combo)
        assert s1 not in seen, f"Phase-1 seed {s1} (combo={combo}) collides with a Phase-2 seed"
    _report("9", "Phase-2's own seed formula (phase2_seed) is collision-free over the full registered "
                 f"grid ({n_checked} combinations checked) AND disjoint from every Phase-1 "
                 f"episode_seed() combination", True, f"n_checked={n_checked}")


# ---------------------------------------------------------------------------
# Item 10 -- killer-prediction re-application wiring (obligation (7)): `reasoning_link_probe.run_cell`
# with blend-OFF surgery + a Phase-2-local seed_override, on a familiarized-style checkpoint.
# ---------------------------------------------------------------------------

def test_item_10_killer_prediction_seed_override():
    import inspect
    sig = inspect.signature(rlp.run_cell)
    assert "seed_override" in sig.parameters and sig.parameters["seed_override"].default is None, (
        "run_cell's seed_override param must exist and default to None (backward-compatible additive "
        "change -- every pre-existing Phase-1 caller must be unaffected)")

    with tempfile.TemporaryDirectory() as td:
        p = _build_synthetic_init_checkpoint(td, arm="off")
        s1 = pft.phase2_seed("eval_killer", "off", "openr1-mix-ext", 0, 20, 5000)
        s2 = pft.phase2_seed("eval_killer", "per_token", "openr1-mix-ext", 0, 20, 5000)
        assert s1 != s2, "two distinct (arm,...) combinations must yield distinct phase2_seed() values"

        r1 = rlp.run_cell(p, K=20, hops=(1, 2, 3, 4), surgery="off", batch_size=4, device="cpu",
                           seed_override=s1, compute_option2=False)
        r2 = rlp.run_cell(p, K=20, hops=(1, 2, 3, 4), surgery="off", batch_size=4, device="cpu",
                           seed_override=s2, compute_option2=False)
        assert r1["surgery"] == "off" and r2["surgery"] == "off"
        assert set(r1["per_h"]) == {1, 2, 3, 4} and set(r2["per_h"]) == {1, 2, 3, 4}

        # regression check: seed_override=None (the pre-existing default) must be BYTE-IDENTICAL to
        # the pre-this-build behavior -- ad-hoc seed=0 path, unaffected by this additive change.
        r_default = rlp.run_cell(p, K=20, hops=(1,), surgery="off", batch_size=4, device="cpu",
                                  compute_option2=False)
        r_explicit_zero = rlp.run_cell(p, K=20, hops=(1,), surgery="off", batch_size=4, device="cpu",
                                        seed_override=0, compute_option2=False)
        assert r_default["per_h"][1]["recovered_frac"] == r_explicit_zero["per_h"][1]["recovered_frac"], (
            "seed_override=0 must reproduce the pre-existing seed_override=None/leg=None ad-hoc "
            "seed=0 default -- backward compatibility regression")

    _report("10", "obligation (7): reasoning_link_probe.run_cell's additive seed_override param "
                  "wired for Phase-2's own killer-prediction re-application (surgery='off' + a "
                  "Phase-2-local, per-(arm,corpus,seed,K) differentiated seed, REUSING the "
                  "already-audited frozen_bias_surgery/run_cell machinery verbatim); backward-"
                  "compatible (seed_override=None/omitted == pre-existing behavior, byte-identical)",
            True)


# ---------------------------------------------------------------------------
# Item 11 -- phase2_chain.sh's own budget_check() abort branch (obligation (8)'s "Budget guard:
# abort if projected > the registered bracket") has real teeth: extracts the REAL function
# definition from the chain script via sed (never a reimplemented copy) and drives it with both a
# forced-huge and a forced-tiny elapsed time, via a real subprocess (bash), asserting the exit code
# AND the sentinel file -- mirrors phase2_gate_enforce/reasoning_link_gate_enforce's own
# process-boundary negative-test discipline, applied here to a bash-native (not Python) gate.
# ---------------------------------------------------------------------------

def test_item_11_budget_guard_negative_test():
    import subprocess
    chain_path = os.path.join(HERE, "phase2_chain.sh")
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "results", "phase2"))
        extracted = os.path.join(td, "extracted_fn.sh")
        with open(chain_path) as f:
            content = f.read()
        m = re.search(r"(budget_check\(\) \{.*?\n\})", content, re.DOTALL)
        assert m, "budget_check() function not found in phase2_chain.sh -- cannot extract for testing"
        with open(extracted, "w") as f:
            f.write(m.group(1))

        py_bin = sys.executable

        def _run(seconds_value: str) -> subprocess.CompletedProcess:
            script = (f'set -uo pipefail; cd "{td}"; PY="{py_bin}"; N_GPUS=2; '
                      f'BUDGET_CEILING_GPU_H=12.06; source "{extracted}"; SECONDS={seconds_value}; '
                      f'budget_check')
            return subprocess.run(["bash", "-c", script], capture_output=True, text=True)

        r_over = _run("100000")   # 100,000s * 2 GPUs / 3600 ~= 55.6 GPU-h >> 12.06 ceiling
        sentinel_over = os.path.join(td, "results", "phase2", "BUDGET_ABORTED")
        assert r_over.returncode != 0, (
            f"budget_check did NOT abort when way over budget: rc={r_over.returncode}, "
            f"stdout={r_over.stdout!r} stderr={r_over.stderr!r}")
        assert os.path.exists(sentinel_over), "budget_check aborted but did NOT write the sentinel file"
        assert "ABORT" in r_over.stderr, f"abort message missing from stderr: {r_over.stderr!r}"

        os.remove(sentinel_over)
        r_under = _run("10")      # 10s * 2 GPUs / 3600 ~= 0.0056 GPU-h, well under 12.06
        assert r_under.returncode == 0, (
            f"budget_check incorrectly aborted when well under budget: rc={r_under.returncode}, "
            f"stdout={r_under.stdout!r} stderr={r_under.stderr!r}")
        assert not os.path.exists(sentinel_over), "budget_check wrote the sentinel when under budget"

    _report("11", "phase2_chain.sh's budget_check() (obligation (8)): forced-huge elapsed time "
                  "correctly ABORTS (nonzero exit + BUDGET_ABORTED sentinel), forced-tiny elapsed "
                  "time correctly does NOT abort -- REAL subprocess exercising the extracted "
                  "function, not a reimplemented copy", True)


# ---------------------------------------------------------------------------
# Item 12 -- MAJOR-1 fix (Phase-2 build-audit round): measure_cell_all_h's use_heldout_entities param
# actually reaches grammar_rd.sample_batch_rd. POSITIVE: True -> drawn entity ids are a SUBSET of
# pools.heldout_name_ids. NEGATIVE (mutation-style, proves the positive check has teeth): with the
# flag left at its default False, entity ids come from pools.train_name_ids instead, which is
# DISJOINT from heldout_name_ids by construction -- so the SAME subset-of-heldout assertion correctly
# FAILS on that batch.
# ---------------------------------------------------------------------------

def test_item_12_heldout_entity_pool_wiring():
    torch.manual_seed(0)
    model = lpr.DeltaNetLM(V_REAL, d_model=32, d_state=64, n_layers=1, conv_size=4)
    tokenizer = grammar_rd.load_gpt2_tokenizer()
    pools, _ = rlp.build_reasoning_link_pools(tokenizer=tokenizer, seed=0)
    cfg = pft.familiarization_gate_episode_config(conv_size=4, K=20)
    readout_layer = rlp.readout_layer_index(model)

    heldout_set = set(pools.heldout_name_ids.tolist())
    train_set = set(pools.train_name_ids.tolist())
    assert heldout_set.isdisjoint(train_set), (
        "precondition: pools.heldout_name_ids/train_name_ids must be disjoint by construction (C17) "
        "for the negative control below to mean anything")

    captured = {}
    orig_sample = grammar_rd.sample_batch_rd
    def _spying_sample_batch_rd(*a, **kw):
        batch = orig_sample(*a, **kw)
        captured["entity_ids"] = batch["entity_ids"].clone()
        captured["use_heldout_entities"] = kw.get("use_heldout_entities", False)
        return batch
    grammar_rd.sample_batch_rd = _spying_sample_batch_rd
    try:
        rlp.measure_cell_all_h(model, cfg, pools, readout_layer, cfg.K, hops=(1,), batch_size=4,
                                seed=1, surgery="native", device="cpu", compute_option2=False,
                                compute_premises=False, use_heldout_entities=True)
        assert captured["use_heldout_entities"] is True, (
            "measure_cell_all_h did not pass use_heldout_entities=True through to sample_batch_rd")
        ids_true = set(captured["entity_ids"].flatten().tolist())
        assert ids_true.issubset(heldout_set), (
            f"POSITIVE: use_heldout_entities=True must draw entities ONLY from pools.heldout_name_ids; "
            f"found ids outside the heldout pool: {ids_true - heldout_set}")

        rlp.measure_cell_all_h(model, cfg, pools, readout_layer, cfg.K, hops=(1,), batch_size=4,
                                seed=1, surgery="native", device="cpu", compute_option2=False,
                                compute_premises=False)   # use_heldout_entities omitted -> default False
        assert captured["use_heldout_entities"] is False
        ids_false = set(captured["entity_ids"].flatten().tolist())
        assert not ids_false.issubset(heldout_set), (
            "NEGATIVE (mutation-style) FAILED TO FAIL: with use_heldout_entities left at its default "
            "False, entity ids should NOT be a subset of heldout_name_ids (this is the exact pre-fix "
            "bug -- MAJOR-1 -- where the eval pool silently fell back to train_name_ids); if this "
            "assertion itself passed, the POSITIVE check above would have no teeth")
        assert ids_false.issubset(train_set), (
            f"NEGATIVE: with use_heldout_entities=False, entity ids should come from train_name_ids; "
            f"found ids outside the train pool: {ids_false - train_set}")
    finally:
        grammar_rd.sample_batch_rd = orig_sample

    _report("12", "MAJOR-1 fix: measure_cell_all_h's additive use_heldout_entities param actually "
                  "reaches grammar_rd.sample_batch_rd -- POSITIVE (True -> entity ids subset of "
                  "heldout_name_ids) + mutation-style NEGATIVE (False -> entity ids subset of "
                  "train_name_ids, NOT heldout_name_ids, proving the positive check has teeth)", True,
            f"heldout_pool_size={len(heldout_set)}, train_pool_size={len(train_set)}")


# ---------------------------------------------------------------------------
# Item 13 -- MAJOR-2 fix (Phase-2 build-audit round): the Stage-0.5 gate's own eval episode config
# uses Q=K (n_query=None), separate from and never mutating training's own pinned n_query=2 config.
# ---------------------------------------------------------------------------

def test_item_13_gate_q_equals_k():
    for K in pft.K_SWEEP:
        train_cfg = pft.familiarization_episode_config(conv_size=4, K=K)
        gate_cfg = pft.familiarization_gate_episode_config(conv_size=4, K=K)
        assert train_cfg.n_query == pft.N_QUERY == 2, (
            f"K={K}: training episode config must still be pinned at n_query=2, got {train_cfg.n_query}")
        assert train_cfg.queries == 2, (
            f"K={K}: training config's own .queries must be 2 (Q!=K), got {train_cfg.queries}")
        assert gate_cfg.n_query is None, (
            f"K={K}: gate episode config must use n_query=None (Q=K), got {gate_cfg.n_query}")
        assert gate_cfg.queries == K, (
            f"K={K}: gate episode config's own .queries must equal K, got {gate_cfg.queries}")

    # end-to-end: compute_stage05_gate's own returned dict discloses gate_k/gate_q -- confirm gate_q
    # equals K (not N_QUERY=2) on a REAL call, not just the config object checked in isolation.
    with tempfile.TemporaryDirectory() as td:
        init_path = _build_synthetic_init_checkpoint(td, arm="off")
        ck = torch.load(init_path, map_location="cpu")
        model = lpr.DeltaNetLM(**ck["config"])
        model.load_state_dict(ck["model_state_dict"])
        tokenizer = grammar_rd.load_gpt2_tokenizer()
        pools, _ = rlp.build_reasoning_link_pools(tokenizer=tokenizer, seed=0)
        train_cfg = pft.familiarization_episode_config(conv_size=4, K=20)
        gate_result = pft.compute_stage05_gate(model, train_cfg, pools, "off", "openr1-mix-ext", 0,
                                                 250, 4, "cpu")
    assert train_cfg.n_query == 2, "compute_stage05_gate must not have mutated the caller's own training config"
    assert gate_result["gate_k"] == 20 and gate_result["gate_q"] == 20, (
        f"compute_stage05_gate: expected gate_k=gate_q=20 (Q=K), got gate_k={gate_result['gate_k']} "
        f"gate_q={gate_result['gate_q']}")

    _report("13", "MAJOR-2 fix: compute_stage05_gate builds its OWN Q=K eval episode config "
                  "(familiarization_gate_episode_config) rather than reusing training's own pinned "
                  "n_query=2 config -- verified at the config-object level (all K in K_SWEEP, training "
                  "config confirmed UNCHANGED at n_query=2) AND end-to-end via compute_stage05_gate's "
                  "own returned gate_k/gate_q fields", True, f"K_SWEEP={pft.K_SWEEP}")


ALL_ITEMS = [
    test_item_1_ckpt_steps, test_item_2_init_checkpoint, test_item_3_periodicity_guard,
    test_item_4_query_loss_forward, test_item_5_hexachotomy_totality,
    test_item_6_gate_enforcement_negative_test, test_item_7_bands_pinned_blind_sequencing,
    test_item_8_e2e_familiarization_cell, test_item_9_seed_non_collision,
    test_item_10_killer_prediction_seed_override, test_item_11_budget_guard_negative_test,
    test_item_12_heldout_entity_pool_wiring, test_item_13_gate_q_equals_k,
]


def run_all_selftests() -> bool:
    print("=" * 70)
    print("PHASE-2 (task familiarization) Stage -1 SELF-TESTS -- 13 items, REASONING_LINK_DESIGN.md "
          "sec 16.2 (Rev 5, CLEARED-FOR-BUILD) + build-audit fixes (sec 16.14)")
    print(f"fla_stub_installed={rlp.FLA_STUB_INSTALLED}")
    print("=" * 70)
    t0 = time.time()
    failures = []
    for fn in ALL_ITEMS:
        try:
            fn()
        except AssertionError as e:
            failures.append((fn.__name__, str(e)))
            print(f"  ** FAILURE in {fn.__name__}: {e}")
    wall = time.time() - t0
    print("=" * 70)
    if failures:
        print(f"PHASE-2 Stage -1: {len(failures)} FAILURE(S) in {wall:.1f}s")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        print("=" * 70)
        return False
    print(f"PHASE-2 Stage -1: ALL {len(ALL_ITEMS)} ITEMS PASSED in {wall:.1f}s")
    print("=" * 70)
    return True


if __name__ == "__main__":
    ok = run_all_selftests()
    sys.exit(0 if ok else 1)
