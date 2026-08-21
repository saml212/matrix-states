#!/usr/bin/env python3
"""K-SCALING PATCH GENERATOR -- NCR_KSCALING_DESIGN.md DRAFT-R0 sec 5.

Produces the K-generic copies of the two PINNED files in ~/ncr_kscaling/.
The pinned originals in ~/ncr_g3b31_contrastive/ are NEVER written to; this
script reads them, verifies their md5 against the pins recorded below, and
writes patched copies elsewhere.

Every edit is an EXACT-STRING replacement whose anchor must occur EXACTLY
ONCE in the source -- a moved or reworded anchor aborts the patch instead of
silently applying somewhere else or nowhere.

Run:  python3 patch_kscaling.py --src ~/ncr_g3b31_contrastive --dst ~/ncr_kscaling
      python3 patch_kscaling.py ... --verify-only     (re-check an existing dst)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

PINNED_MD5 = {
    "ncr_lm_wave1_runner.py": "9a93198b642242f512ff8489e32b0a53",
    "ncr_lm_wave1_smoke.py": None,          # filled from the box at first run; see --record-src-md5
}

# ==========================================================================
# SMOKE-MODULE PATCHES (the graft library: constants, cfg, document builder)
# ==========================================================================
SMOKE_PATCHES = [
    # ---- S1: K/d/param formula -> derived, in ONE place -------------------
    ("S1_derived_K_d_params", """K_NCR = 24                              # sec N2.1 GATE-1 amendment (free-write, un-gated to K=24)
D_NCR = 25                              # sec N2.2(a): d_ncr = K+1 = 25 (was 33 at K=32)
H_NCR = nm.ENC_H                        # 64, BindingEncoder's own encoder width, untouched by K
NCR_PARAM_EXACT = 40 * H_NCR ** 2 + 4 * D_NCR * H_NCR + 46 * H_NCR + D_NCR   # sec N2.2(c) formula
assert NCR_PARAM_EXACT == 173_209, NCR_PARAM_EXACT      # sanity on the formula transcription itself""",
     '''# ---------------------------------------------------------------------------
# K-SCALING PATCH S1 (NCR_KSCALING_DESIGN.md sec 5.1). The pinned file set
# K_NCR=24 and D_NCR=25 as TWO INDEPENDENT hand-set constants, so changing K
# left the operator dimension behind (gate memo leg 1, PATCH-required).
# Both now come from kscaling_config, where D_NCR = K_NCR + 1 is the ONLY
# definition, and NCR_PARAM_EXACT is RE-DERIVED per K instead of asserting
# the K=24 literal 173_209.
# ---------------------------------------------------------------------------
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kscaling_config as KS             # noqa: E402

K_NCR = KS.K_NCR
D_NCR = KS.D_NCR                        # derived: K_NCR + 1, single definition in kscaling_config
H_NCR = nm.ENC_H                        # 64, BindingEncoder's own encoder width, untouched by K
NCR_PARAM_EXACT = KS.ncr_param_exact(H_NCR)
assert D_NCR == K_NCR + 1, (K_NCR, D_NCR)
assert NCR_PARAM_EXACT == 40 * H_NCR ** 2 + 4 * D_NCR * H_NCR + 46 * H_NCR + D_NCR, NCR_PARAM_EXACT
# Back-compat anchor: at K=24 the derived value must still be the pinned literal.
assert (K_NCR != 24) or (NCR_PARAM_EXACT == 173_209), NCR_PARAM_EXACT'''),

    # ---- S2: task cfg gets ladder-derived H_test/H_extra -------------------
    # DeltaNetRDTaskConfig's OWN periodicity guard runs on H_test/H_extra. Its
    # defaults are H_test=(4,5,6), H_extra=(7,21); at K=20, 21 % 20 == 1 == a
    # train residue, so constructing the config AT ALL raises AssertionError.
    # Feeding it this K's own ladder both fixes that and turns the config's
    # guard into a SECOND, independent check on the ladder we actually use.
    ("S2_cfg_ladder_hops", """    cfg = gr.DeltaNetRDTaskConfig(K=K_NCR, conv_size=RUNG1_BACKBONE["conv_size"], n_query=1)
    return pools, cfg, report""",
     '''    # K-SCALING PATCH S2 (NCR_KSCALING_DESIGN.md sec 5.2): the pinned call left
    # H_test/H_extra at their defaults (4,5,6)/(7,21). DeltaNetRDTaskConfig's own
    # __post_init__ periodicity guard runs on those, and at K=20, 21 % 20 == 1 ==
    # a TRAIN residue -- so merely CONSTRUCTING the config raised AssertionError
    # at K=20. Passing this K's own residue-verified ladder fixes that AND makes
    # grammar_rd's independent guard re-check the ladder this run evaluates on.
    cfg = gr.DeltaNetRDTaskConfig(
        K=K_NCR, conv_size=RUNG1_BACKBONE["conv_size"], n_query=1,
        H_train=tuple(KS.TRAIN_HOPS),
        H_test=tuple(KS.DEEP_LADDER[:3]),
        H_extra=tuple(KS.DEEP_LADDER[3:]) + (KS.FIXED_DIST_PROBE,))
    assert cfg.T_bind + cfg.query_len + 1 == KS.doc_len(), (cfg.T_bind, KS.doc_len())
    return pools, cfg, report'''),

    # ---- S3: T-floor left pad ---------------------------------------------
    ("S3_doc_left_pad", """    query_key_col = T_bind + buf_len                                   # scalar: batch-uniform template offset
    query_mark_col = T_bind + query_len - 1                            # scalar: the <Q> position (== input's last col)
    assert query_mark_col == doc.shape[1] - 2, "query_mark_col must be the second-to-last doc column\"""",
     '''    query_key_col = T_bind + buf_len                                   # scalar: batch-uniform template offset
    query_mark_col = T_bind + query_len - 1                            # scalar: the <Q> position (== input's last col)

    # K-SCALING PATCH S3 (NCR_KSCALING_DESIGN.md sec 5.3): T-FLOOR LEFT PAD.
    # The backbone is fed doc[:, :-1] (length 7K+6) and
    # lm_pretrain_rd.DeltaNetLMMixer.forward hard-asserts T >= 128
    # (chunk_delta_rule's backward crashes below it). MEASURED on this box
    # 2026-08-21: T_in=90 (K=12) and T_in=118 (K=16) raise AssertionError;
    # T_in in {128,146,174,230} pass. So K in {12,16} are UNRUNNABLE unpadded.
    # Prepend KS.doc_left_pad() inert BUFFER tokens -- the SAME reserved token
    # grammar_rd already uses as intra-clause filler, so no new token type
    # enters the vocabulary -- and shift all four position fields by the same
    # amount. Pad is 0 for every K >= 20, so K >= 20 (in particular the K=24
    # anchor) stays BYTE-IDENTICAL to the pinned construction.
    _pad = KS.doc_left_pad()
    if _pad:
        _pad_block = torch.full((doc.shape[0], _pad), int(pools.buffer_id),
                                 dtype=doc.dtype, device=doc.device)
        doc = torch.cat([_pad_block, doc], dim=1)
        key_pos = key_pos + _pad
        val_pos = val_pos + _pad
        query_key_col += _pad
        query_mark_col += _pad
    assert doc.shape[1] - 1 == KS.t_in(), (doc.shape, KS.t_in())
    assert doc.shape[1] - 1 >= KS.MIN_KERNEL_T, (doc.shape, KS.MIN_KERNEL_T)
    assert query_mark_col == doc.shape[1] - 2, "query_mark_col must be the second-to-last doc column"'''),
]

# ==========================================================================
# RUNNER PATCHES (ladder, provenance)
# ==========================================================================
RUNNER_PATCHES = [
    # ---- R1: per-K ladder + the strengthened assert ------------------------
    ("R1_perK_ladder", '''RUNNER_TAG = "ncr_gate3_wave1_runner_v1"          # sec G3-B6
TRAIN_HOPS = (1, 2, 3)                             # sec 3.1 Task-1 train range, verbatim
DEEP_LADDER = (5, 12, 20, 29, 40, 61)              # sec 3.1's eval ladder (NCR_ORTHO_WRITE.md sec 3), reused verbatim''',
     '''# ---------------------------------------------------------------------------
# K-SCALING PATCH R1 (NCR_KSCALING_DESIGN.md sec 5.4). RUNNER_TAG is bumped so
# a K-scaling checkpoint can never be silently resumed by, or confused with,
# the pinned K=24 wave (load_checkpoint asserts on this field).
# DEEP_LADDER is no longer a carried literal: the pinned (5,12,20,29,40,61)
# CRASHES the soundness guard at K in {12,20,28} and SILENTLY loses a rung to
# a residue collision at K in {16,24,32} (gate memo leg 1). It is replaced by
# kscaling_config's per-K residue-verified ladder.
# ---------------------------------------------------------------------------
import kscaling_config as KS                        # noqa: E402

RUNNER_TAG = "ncr_kscaling_runner_v1"
TRAIN_HOPS = tuple(KS.TRAIN_HOPS)                   # sec 3.1 Task-1 train range, verbatim
DEEP_LADDER = tuple(KS.DEEP_LADDER)                 # per-K, residue-verified (kscaling_config)
FIXED_DIST_PROBE = KS.FIXED_DIST_PROBE              # labelled fixed-effective-distance control
H_TOP = KS.H_TOP                                    # the PRIMARY readout depth for this K
PINNED_K24_LADDER = KS.REFERENCE_K24_LADDER         # kept ONLY for the smoke's negative test'''),

    ("R2_ladder_assert", '''def _assert_ladder_sound(ladder: tuple, k: int, train_hops: tuple) -> None:
    train_residues = {h % k for h in train_hops}
    for h in ladder:
        r = h % k
        assert r != 0, f"deep-ladder h={h} is IDENTITY mod K={k} (h%K=0) -- confounded, not held-out"
        assert r not in train_residues, (
            f"deep-ladder h={h} has h%K={r} colliding with a train-residue "
            f"{sorted(train_residues)} -- secretly in-distribution, not held-out")


_assert_ladder_sound(DEEP_LADDER, K_NCR, TRAIN_HOPS)''',
     '''def _assert_ladder_sound(ladder: tuple, k: int, train_hops: tuple) -> None:
    """K-SCALING PATCH R2: delegates to kscaling_config.assert_ladder_sound,
    which keeps BOTH of the pinned checks (identity residue; train-residue
    collision) and ADDS the one the pinned guard is missing -- PAIRWISE
    residue distinctness -- plus strictly-increasing depth and a matched
    squaring profile. The pinned guard passes ladders in which two rungs
    measure the SAME ground truth (it does so on the K=24 ladder of record:
    29 == 5 mod 24). The smoke's negative test proves this version REJECTS
    such a ladder."""
    KS.assert_ladder_sound(ladder, k, train_hops)


_assert_ladder_sound(DEEP_LADDER, K_NCR, TRAIN_HOPS)
assert D_NCR == K_NCR + 1, (K_NCR, D_NCR)
assert FIXED_DIST_PROBE % K_NCR == KS.FIXED_DIST_RESIDUE
assert H_TOP == DEEP_LADDER[-1] and H_TOP % K_NCR == K_NCR // 2'''),

    # ---- R3: eval the fixed-distance control alongside the ladder ----------
    ("R3_eval_fixed_dist", '''            "deep": eval_arm_at_hops(arm, pools, cfg, DEEP_LADDER, batch_size, device, seed, read_ablate, tf_this_arm),
        }
    return result''',
     '''            "deep": eval_arm_at_hops(arm, pools, cfg, DEEP_LADDER, batch_size, device, seed, read_ablate, tf_this_arm),
            # K-SCALING PATCH R3: the labelled fixed-effective-distance control
            # (residue 4 at every K, same squaring count as h_top). Kept in its
            # OWN block, never merged into "deep" -- it deliberately shares a
            # residue with ladder rung 1 and must never be counted as an
            # independent ladder probe. See kscaling_config's module docstring.
            "fixed_dist": eval_arm_at_hops(arm, pools, cfg, (FIXED_DIST_PROBE,), batch_size,
                                            device, seed, read_ablate, tf_this_arm),
        }
    return result'''),

    # ---- R4: provenance into the training results JSON --------------------
    ("R4_provenance_calib", '''        config=dict(K=K_NCR, d_ncr=D_NCR, h_ncr=H_NCR, backbone=RUNG1_BACKBONE,''',
     '''        kscaling=KS.provenance(H_NCR, RUNG1_BACKBONE["d_model"]),
        config=dict(K=K_NCR, d_ncr=D_NCR, h_ncr=H_NCR, backbone=RUNG1_BACKBONE,'''),

    ("R5_provenance_phase0", '''        config=dict(K=K_NCR, d_ncr=D_NCR, backbone=RUNG1_BACKBONE, vocab_size_total=vocab_size_total,''',
     '''        kscaling=KS.provenance(H_NCR, RUNG1_BACKBONE["d_model"]),
        config=dict(K=K_NCR, d_ncr=D_NCR, backbone=RUNG1_BACKBONE, vocab_size_total=vocab_size_total,'''),

    # ---- R6: cell-id default carries K ------------------------------------
    ("R6_cell_id", '''    cell_id = args.cell_id or f"wave1_calib_K{K_NCR}_s{args.seed}"''',
     '''    cell_id = args.cell_id or f"kscaling_K{K_NCR}_s{args.seed}"'''),

    # ---- R6b: the --k / NCR_K tripwire, on EVERY mode ----------------------
    # Placed immediately after parse_args so it also guards --mode smoke and
    # --mode phase0-timing, both of which return before the calibration block.
    ("R6b_k_tripwire", '''    args = ap.parse_args()

    print("=" * 70)''',
     '''    args = ap.parse_args()

    # K-SCALING PATCH R6b: --k is a MANDATORY, redundant restatement of the NCR_K
    # environment variable that kscaling_config actually reads. A spec whose env
    # and flag disagree (the single easiest way to launch 30 cells at the wrong K)
    # dies HERE -- before any GPU work, on every mode -- instead of producing
    # plausible, wrongly-labelled numbers.
    assert args.k == K_NCR, (
        f"--k {args.k} disagrees with NCR_K={os.environ.get('NCR_K')!r} "
        f"(kscaling_config resolved K_NCR={K_NCR}). Refusing to run: one of the two is a "
        f"typo and the results would be silently mislabelled.")
    print(f"[kscaling] K={K_NCR} d_ncr={D_NCR} chance={KS.CHANCE:.4f} ladder={DEEP_LADDER} "
          f"h_top={H_TOP} (residue {H_TOP % K_NCR} == K/2) fixed_dist_probe={FIXED_DIST_PROBE} "
          f"t_in={KS.t_in()} doc_left_pad={KS.doc_left_pad()}", flush=True)

    print("=" * 70)'''),

    # ---- R8: the deepest-gap keys were K=24-ladder LITERALS -----------------
    # build_attribution indexed deep_gap["h=61"] and retrieval24_gap_deep["h=61"].
    # h=61 is not in ANY derived ladder, so every cell at every K raised
    # KeyError('h=61') at its FIRST eval -- a launch-losing crash the module-level
    # smoke could not see, because it never calls build_attribution. Caught by
    # this build's end-to-end 3-step production run. Re-keyed to this K's own
    # h_top, with the depth named in the field rather than frozen into it.
    ("R8_deepest_gap_key_1", '''        "primary_signal_deepest_gap_h61": deep_gap["h=61"],''',
     '''        "primary_signal_deepest_gap": deep_gap[f"h={H_TOP}"],
        "primary_signal_deepest_gap_h": H_TOP,
        "primary_signal_deepest_gap_residue": H_TOP % K_NCR,'''),

    ("R8_deepest_gap_key_2", '''        "primary_signal_v2_deepest_gap_h61": retrieval24_gap_deep["h=61"],''',
     '''        "primary_signal_v2_deepest_gap": retrieval24_gap_deep[f"h={H_TOP}"],
        "primary_signal_v2_deepest_gap_h": H_TOP,
        "primary_signal_v2_deepest_gap_residue": H_TOP % K_NCR,
        "primary_signal_v2_chance": KS.CHANCE,
        "primary_signal_v2_fixed_dist_probe_h": FIXED_DIST_PROBE,'''),

    ("R9_attribution_prose", '''            "depth (h in {5,12,20,29,40,61}) -- the sec G3-B26-adopted PRIMARY signal for future "''',
     '''            f"depth (h in {list(DEEP_LADDER)}, this K's own residue-verified ladder) -- the sec "
            "G3-B26-adopted PRIMARY signal for future "'''),

    ("R7_k_flag", '''    ap.add_argument("--cell-id", default=None)''',
     '''    ap.add_argument("--k", type=int, required=True,
                     help="K-SCALING PATCH R7: mandatory restatement of NCR_K. Must equal the "
                          "env var kscaling_config read; asserted in main(). Not a second source "
                          "of truth -- a tripwire against env/flag drift across 30 specs.")
    ap.add_argument("--cell-id", default=None)'''),
]


def md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def apply_patches(text: str, patches, label: str) -> tuple[str, list]:
    log = []
    for name, old, new in patches:
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"PATCH ABORT [{label}/{name}]: anchor occurs {n} times, expected exactly 1. "
                f"The pinned file has moved underneath this patch -- re-derive the anchor, "
                f"do not loosen the match.\n--- anchor ---\n{old[:300]}")
        text = text.replace(old, new)
        log.append({"patch": name, "removed_lines": old.count("\n") + 1,
                    "added_lines": new.count("\n") + 1})
    return text, log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.expanduser("~/ncr_g3b31_contrastive"))
    ap.add_argument("--dst", default=os.path.expanduser("~/ncr_kscaling"))
    ap.add_argument("--config", default=None, help="path to kscaling_config.py to install into dst")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    src, dst = os.path.expanduser(args.src), os.path.expanduser(args.dst)
    runner_src = os.path.join(src, "ncr_lm_wave1_runner.py")
    smoke_src = os.path.join(src, "ncr_lm_wave1_smoke.py")

    got = md5(runner_src)
    want = PINNED_MD5["ncr_lm_wave1_runner.py"]
    if got != want:
        raise SystemExit(f"PIN MISMATCH: {runner_src} md5={got} != pinned {want}. "
                         f"The pinned instrument changed -- STOP and adjudicate.")

    if args.verify_only:
        rec = {f: md5(os.path.join(dst, f)) for f in sorted(os.listdir(dst)) if f.endswith(".py")}
        print(json.dumps({"dst": dst, "md5": rec}, indent=1))
        return 0

    os.makedirs(dst, exist_ok=True)
    if args.config:
        shutil.copy2(os.path.expanduser(args.config), os.path.join(dst, "kscaling_config.py"))

    out = {"src_md5": {"ncr_lm_wave1_runner.py": got,
                       "ncr_lm_wave1_smoke.py": md5(smoke_src)}, "patches": {}}

    for name, srcp, patches in (("ncr_lm_wave1_smoke.py", smoke_src, SMOKE_PATCHES),
                                 ("ncr_lm_wave1_runner.py", runner_src, RUNNER_PATCHES)):
        with open(srcp) as f:
            text = f.read()
        text, log = apply_patches(text, patches, name)
        dstp = os.path.join(dst, name)
        with open(dstp, "w") as f:
            f.write(text)
        out["patches"][name] = log
        out.setdefault("dst_md5", {})[name] = md5(dstp)

    # The pinned originals must be untouched by this process.
    assert md5(runner_src) == want, "the pinned runner was modified -- ABORT"
    out["pinned_untouched"] = True
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
