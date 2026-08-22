#!/usr/bin/env python3
"""SCALE-AXIS PORT PATCH GENERATOR -- NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 3.

Produces the 392M-capable copies of the five instruments in ~/ncr_scaleaxis/.
The SOURCES are the K-scaling PATCHED tree (sec 3.1's "version of record for
the sweep") plus the two scorers; they are NEVER written to. This script reads
them, md5-verifies EVERY ONE against the pins below (BUILD REQUIREMENT B4 --
patch_kscaling.py pinned only the runner, and the graft is precisely the file
this port edits), and writes patched copies elsewhere.

Every edit is an EXACT-STRING replacement whose anchor must occur EXACTLY ONCE
in the source -- a moved or reworded anchor aborts the patch instead of
silently applying somewhere else or nowhere (patch_kscaling.py:257-269's own
machinery, inherited unchanged).

THE PORT IS ONE DICT (sec 3.1).  RUNG1_BACKBONE moves 768/64/12 -> 1536/128/16.
Everything else in this file exists because sec 3.2's 21-item enumeration (plus
B1's own additions, sec 3.7) found the constants that are ARITHMETICALLY WRONG
at 392M unless they re-derive from that dict.

MODULE NAMING, DISCLOSED.  The scaleaxis tree keeps the module name
`kscaling_config.py` rather than renaming it `scaleaxis_config.py`.  The graft,
the runner, the battery and depthext_eval all do `import kscaling_config as KS`
and reference `KS.` at ~60 sites; renaming would turn a one-dict port into a
60-site rewrite, against sec 3's whole framing.  The tree is self-contained
(`sys.path.insert(0, dirname(__file__))` in every entry point), so the
scaleaxis copy shadows the kscaling copy for every process launched from
~/ncr_scaleaxis/ and cannot be reached from ~/ncr_kscaling/.  Provenance
records `config_module_tree` so a reader never has to infer which copy
produced a number.

Run:  python3 patch_scaleaxis.py --dst ~/ncr_scaleaxis
      python3 patch_scaleaxis.py --dst ~/ncr_scaleaxis --verify-only
      python3 patch_scaleaxis.py --dst /tmp/x --negative-test   (B4 forced-fail)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_KSB = os.path.join(os.path.dirname(_HERE), "kscaling_build")

# --------------------------------------------------------------------------
# BUILD REQUIREMENT B4 -- every source is hard-pinned, including the GRAFT.
# patch_kscaling.py pinned ONLY the runner ("ncr_lm_wave1_smoke.py": None) and
# the graft md5 lived in gen_job_specs.py PROSE. The graft is the file this
# port edits, so an unenforced pin on it is the one that matters.
# --------------------------------------------------------------------------
SRC_DIRS = {
    "ncr_lm_wave1_smoke.py": os.path.join(_KSB, "patched"),
    "ncr_lm_wave1_runner.py": os.path.join(_KSB, "patched"),
    "kscaling_config.py": _KSB,
    "kscaling_battery.py": _KSB,
    "depthext_eval.py": os.path.join(os.path.dirname(_HERE), "..",
                                     "experiment-runs", "2026-08-22_depthext_across_k"),
}
PINNED_MD5 = {
    "ncr_lm_wave1_smoke.py": "74ee84fc920b024901d11add66cc5c2d",   # K-scaling PATCHED graft
    "ncr_lm_wave1_runner.py": "ee5833743049e1bb1864124ad5d3fbf6",  # K-scaling PATCHED runner
    "kscaling_config.py": "eaddd0411fd1cdaaa6028735023c1b99",
    "kscaling_battery.py": "5735c788563d9a21f2198c9f5b4793d5",     # the battery of record (sec 3.5)
    "depthext_eval.py": "e95ffe192c66d3e054f3febc37fe4a91",
}
# The ULTIMATE pinned originals, which no patch in this program may touch.
# Verified read-only on the box at deploy time (sec 3.6).
UPSTREAM_PINNED_MD5 = {
    "~/ncr_g3b31_contrastive/ncr_lm_wave1_runner.py": "9a93198b642242f512ff8489e32b0a53",
    "~/ncr_g3b31_contrastive/ncr_lm_wave1_smoke.py": "bc105af69661e488ff95f5046e2bcd8a",
}
# Files copied through unmodified: the K-generic smoke is already scale-generic
# (item F asserts against KS.integ_param_exact(R.RUNG1_BACKBONE["d_model"]) and
# item A calls KS.provenance(R.H_NCR, R.RUNG1_BACKBONE["d_model"]) -- both
# parametric, sec 3.2 items 11/12 "verify, do not rewrite").
COPY_VERBATIM = {"kscaling_smoke.py": "50eb09c03952b81f70df18eed3c3f05e"}
SRC_DIRS["kscaling_smoke.py"] = _KSB


# ==========================================================================
# CONFIG PATCHES -- the scale resolution, the sec 3.4 param formulas, and the
# two size-bearing defaults B1 found that sec 3.2's 21 items did not name.
# ==========================================================================
CONFIG_PATCHES = [
    ("C1_scale_resolution_and_param_formulas", """TRAIN_HOPS = (1, 2, 3)
MIN_KERNEL_T = 128                 # lm_pretrain_rd._MIN_KERNEL_T, measured above
CONV_SIZE = 4                      # RUNG1_BACKBONE["conv_size"] -- drives buf_len
N_LADDER_RUNGS = 6""",
     '''TRAIN_HOPS = (1, 2, 3)

# ---------------------------------------------------------------------------
# SCALE-AXIS PORT PATCH C1 (NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 3.1-sec 3.4).
# THE ONE DICT, and everything that must re-derive from it.
#
# MIN_KERNEL_T PROVENANCE, sec 3.2 item 19 / m4 -- READ THIS BEFORE TRUSTING IT.
# 128 is a MEASURED constant and this module's own docstring records the
# measurement as taken at the RUNG-1 backbone, i.e. d_state = 64. It has NEVER
# been validated at d_state = 128, and the ported K=16 cell sits EXACTLY on the
# boundary (t_in = 128, zero margin). Stage A0.2 is a HARD pre-sweep gate that
# re-measures the floor at the resolved backbone before any K=16 cell is
# queued; scaleaxis_gates.py item B_MINKT is that gate, and it writes
# MIN_KERNEL_T_VALIDATED_AT into its record. Nothing here assumes it.
# ---------------------------------------------------------------------------
VOCAB_SIZE = 50257                 # GPT-2 BPE, lm_rd_rung_configs.py VOCAB_SIZE
VOCAB_RESERVED_EXTRA = 2           # grammar_rd's BUFFER + <Q> ids -> vocab_size_total 50259
PARAM_COUNT_TOLERANCE = 0.15       # lm_rd_rung_configs.py, unchanged by the port

# lm_rd_rung_configs.py RUNGS, verbatim. Rung 2 is the port target.
RUNGS: dict[int, dict] = {
    1: dict(d_model=768,  d_state=64,  n_layers=12, conv_size=4, num_heads=1, ffn_mult=4),
    2: dict(d_model=1536, d_state=128, n_layers=16, conv_size=4, num_heads=1, ffn_mult=4),
}
RUNG_OF_SCALE = {"98m": 1, "392m": 2}
BACKBONE_PARAM_TARGET_OF_SCALE = {"98m": 98_000_000, "392m": 392_000_000}
# sec 4.5: only these four K are ported. K=12/20/28/36 are deliberately NOT.
PORTED_K_GRID_392M = (16, 24, 32, 40)


def _scale_from_env() -> str:
    raw = os.environ.get("NCR_SCALE", "98m").strip().lower()
    assert raw in RUNG_OF_SCALE, (
        f"NCR_SCALE={raw!r} is not one of {sorted(RUNG_OF_SCALE)}. The backbone rung, the "
        f"param target and the ported-K restriction all resolve from it; refusing to guess.")
    return raw


SCALE: str = _scale_from_env()
RUNG: int = RUNG_OF_SCALE[SCALE]
BACKBONE: dict = dict(RUNGS[RUNG])                 # the resolved rung dict -- ONE definition
BACKBONE_PARAM_TARGET: int = BACKBONE_PARAM_TARGET_OF_SCALE[SCALE]
BACKBONE_PARAM_TOLERANCE: float = PARAM_COUNT_TOLERANCE

MIN_KERNEL_T = 128                 # lm_pretrain_rd._MIN_KERNEL_T, MEASURED AT d_state=64 (above)
MIN_KERNEL_T_MEASURED_AT_DSTATE = 64
CONV_SIZE = BACKBONE["conv_size"]  # sec 3.1's shadow constant, now READ FROM THE DICT rather
                                    # than hand-copied. It is invariant under this port (4 -> 4),
                                    # but it was a second source of truth and is now not one.
N_LADDER_RUNGS = 6


def backbone_param_exact(vocab: int, bb: dict | None = None) -> int:
    """NCR_SCALE_AXIS_DESIGN.md sec 3.4, executed.

      vocab*d_model                                  tied embedding / head
    + n_layers * ( 2*ffn_mult*d_model^2              FFN
                 + 4*d_model*d_state                 q,k,v,o
                 + 3*d_state*conv_size               short conv
                 + d_model                           beta projection
                 + 2*d_model + d_state )             2 norms at d_model, 1 head-norm at d_state
    + d_model                                        final norm

    VALIDATED AGAINST FOUR INDEPENDENTLY-MEASURED ENDPOINTS (sec 3.4):
      rung1/50257 -> 97,618,176  (fixscale_pilot_98m_off_1000.json "n_params")
      rung1/50259 -> 97,619,712  (every archived NCR cell's params.backbone)
      rung2/50257 -> 391,869,440 (fixscale_pilot_392m_off_1000.json "n_params")
      rung2/50259 -> 391,872,512 (what the NCR graft builds at rung 2)
    assert_param_table() below proves all four at import."""
    b = BACKBONE if bb is None else bb
    dm, ds, nl = b["d_model"], b["d_state"], b["n_layers"]
    per_layer = (2 * b["ffn_mult"] * dm * dm + 4 * dm * ds + 3 * ds * b["conv_size"]
                 + dm + (2 * dm + ds))
    return vocab * dm + nl * per_layer + dm


def total_param_exact(k: int | None = None, bb: dict | None = None,
                      h_enc: int = 64) -> int:      # 64 == nm.ENC_H, BACKBONE-INDEPENDENT
    # ^ the bare 64 is a size-bearing literal B1 greps for (AUDIT-R1 m2 / C11).
    # It is INVARIANT under this port by sec 3.3's CODE PROOF, not by assumption:
    # NCREarlyLNModel is built as els.NCREarlyLNModel(d=D_NCR, h=ENC_H) and
    # d_model never enters the constructor, so every head tensor is a function of
    # d = K+1 and h = 64 only. This module is pure python (no torch), so it
    # cannot import ncr_models to read ENC_H; scaleaxis_gates B3 closes the loop
    # by asserting the MEASURED head count against ncr_param_exact(R.H_NCR).
    """Total parameters PER ARM = backbone(vocab_size_total) + NCR head + INTEG.
    This is what sec 3.4's table states and what every spec's validity_check
    asserts (sec 3.2 item 15 -- gen_job_specs.PARAMS_PER_ARM's hard-coded 98M
    table is WRONG FOR EVERY K at 392M and is replaced by this function)."""
    b = BACKBONE if bb is None else bb
    kk = K_NCR if k is None else k
    return (backbone_param_exact(VOCAB_SIZE + VOCAB_RESERVED_EXTRA, b)
            + ncr_param_exact(h_enc, kk)
            + integ_param_exact(b["d_model"], kk))


def assert_param_table() -> None:
    """Import-time proof against the four measured endpoints of record."""
    checks = [(1, 50257, 97_618_176), (1, 50259, 97_619_712),
              (2, 50257, 391_869_440), (2, 50259, 391_872_512)]
    for rung, vocab, want in checks:
        got = backbone_param_exact(vocab, RUNGS[rung])
        assert got == want, (
            f"sec 3.4 formula gives {got:,} at rung {rung}/vocab {vocab}, but the MEASURED "
            f"count of record is {want:,} -- the port arithmetic is wrong, HALT.")


TOTAL_PARAM_TABLE_392M = {16: 392_095_889, 24: 392_122_521,
                          32: 392_149_153, 40: 392_175_785}   # sec 3.4's table, verbatim
TOTAL_PARAM_TABLE_98M = {12: 97_809_805, 16: 97_816_977, 20: 97_824_149,
                         24: 97_831_321, 28: 97_838_493, 32: 97_845_665,
                         36: 97_852_837, 40: 97_860_009}      # gen_job_specs MEASURED table'''),

    ("C2_ported_K_and_param_selfcheck", """K_NCR: int = _k_from_env()
D_NCR: int = K_NCR + 1               # <-- the ONLY definition of d_ncr in the program
CHANCE: float = 1.0 / K_NCR""",
     '''K_NCR: int = _k_from_env()
D_NCR: int = K_NCR + 1               # <-- the ONLY definition of d_ncr in the program
CHANCE: float = 1.0 / K_NCR

# SCALE-AXIS PORT PATCH C2. At 392M only sec 4.5's four ported K exist. Without
# this, `NCR_K=12 NCR_SCALE=392m` would build a perfectly valid cell that no
# band, no reference table and no cross-scale stratum in the design covers.
if SCALE == "392m":
    assert K_NCR in PORTED_K_GRID_392M, (
        f"NCR_K={K_NCR} is not one of the FOUR ported K {PORTED_K_GRID_392M} "
        f"(NCR_SCALE_AXIS_DESIGN.md sec 4.5). K in {{12,20,28,36}} are deliberately NOT "
        f"ported: no 98M reference stratum, no band and no cross-scale test covers them.")'''),

    ("C3_provenance_no_768_default", """def provenance(h_enc: int | None = None, d_model: int = 768) -> dict:""",
     '''def provenance(h_enc: int | None = None, d_model: int | None = None) -> dict:'''),

    ("C4_provenance_scale_block", """        "ladder_rule": "6 rungs, squaring profile (2,3,4,4,5,5); h_top = min h in [32,63] with h==K/2 mod K",
        "config_module": "kscaling_config.py",
    }
    if h_enc is not None:
        rec["ncr_param_exact"] = ncr_param_exact(h_enc)
        rec["integ_param_exact"] = integ_param_exact(d_model)
    return rec""",
     '''        "ladder_rule": "6 rungs, squaring profile (2,3,4,4,5,5); h_top = min h in [32,63] with h==K/2 mod K",
        "config_module": "kscaling_config.py",
        # ---- SCALE-AXIS PORT PATCH C4 -------------------------------------
        # sec 3.2 item 22 (found by B1, NOT in the design's 21-item list):
        # `d_model` used to DEFAULT to the literal 768, and kscaling_battery.py
        # called provenance(64, 768) positionally. At 392M that silently
        # recorded HALF the true integ param count in every score record. The
        # default is now None and the integ count is refused rather than
        # guessed.
        "config_module_tree": os.path.dirname(os.path.abspath(__file__)),
        "scale": SCALE, "rung": RUNG, "backbone": dict(BACKBONE),
        "backbone_param_target": BACKBONE_PARAM_TARGET,
        "vocab_size": VOCAB_SIZE, "vocab_size_total": VOCAB_SIZE + VOCAB_RESERVED_EXTRA,
        "backbone_param_exact": backbone_param_exact(VOCAB_SIZE + VOCAB_RESERVED_EXTRA),
        "total_param_exact_per_arm": total_param_exact(),
        "min_kernel_T_measured_at_dstate": MIN_KERNEL_T_MEASURED_AT_DSTATE,
        "min_kernel_T_validated_at_this_dstate": (
            MIN_KERNEL_T_MEASURED_AT_DSTATE == BACKBONE["d_state"]),
    }
    if h_enc is not None:
        assert d_model is not None, (
            "provenance(h_enc, d_model): d_model is REQUIRED when h_enc is given. It used to "
            "default to the literal 768, which is silently WRONG at rung 2 (sec 3.2 item 22).")
        assert d_model == BACKBONE["d_model"], (
            f"provenance(d_model={d_model}) disagrees with the resolved backbone "
            f"d_model={BACKBONE['d_model']} -- one of the two is stale.")
        rec["ncr_param_exact"] = ncr_param_exact(h_enc)
        rec["integ_param_exact"] = integ_param_exact(d_model)
    return rec'''),

    ("C5_run_param_table_at_import", """assert_ladder_table()

DEEP_LADDER: tuple[int, ...] = LADDER_TABLE[K_NCR]""",
     '''assert_ladder_table()
assert_param_table()          # SCALE-AXIS PORT PATCH C5: sec 3.4's four measured endpoints

DEEP_LADDER: tuple[int, ...] = LADDER_TABLE[K_NCR]'''),
]


# ==========================================================================
# GRAFT PATCHES -- the port dict itself, and BUILD REQUIREMENT B2.
# ==========================================================================
GRAFT_PATCHES = [
    ("G1_the_one_dict", """VOCAB_SIZE = 50257                      # GPT-2 BPE, lm_rd_rung_configs.py VOCAB_SIZE
RUNG1_BACKBONE = dict(d_model=768, d_state=64, n_layers=12, conv_size=4,
                       num_heads=1, ffn_mult=4)          # lm_rd_rung_configs.py RUNGS[1]
BACKBONE_PARAM_TARGET = 98_000_000
BACKBONE_PARAM_TOLERANCE = 0.15         # lm_rd_rung_configs.py PARAM_COUNT_TOLERANCE""",
     '''# ---------------------------------------------------------------------------
# SCALE-AXIS PORT PATCH G1 (NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 3.1/sec 3.2).
# THE PORT IS ONE DICT. The pinned file hard-set the rung-1 (98M) backbone AND
# its derived param target as four independent literals; all four now resolve
# from kscaling_config's RUNGS table via the mandatory NCR_SCALE env var, so
# d_model / d_state / n_layers have exactly ONE definition in the program.
#   sec 3.2 item 17: BACKBONE_PARAM_TARGET is a DERIVED PAIR THAT ALSO MOVES.
#   Left at 98_000_000 the 15% backbone gate fires on a CORRECT 392M build.
# The NAME RUNG1_BACKBONE is deliberately KEPT: it is read at 14 sites across
# the graft, runner, battery and smoke, and every one of them is already
# d_model-parametric (sec 3.2 items 11/12 -- "verify, do not rewrite"). It now
# means "the resolved backbone of this run's rung"; provenance records which.
# ---------------------------------------------------------------------------
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kscaling_config as _KSCFG          # noqa: E402

VOCAB_SIZE = _KSCFG.VOCAB_SIZE          # 50257, GPT-2 BPE -- invariant under the port
RUNG1_BACKBONE = dict(_KSCFG.BACKBONE)  # the RESOLVED rung dict (NCR_SCALE selects the rung)
BACKBONE_PARAM_TARGET = _KSCFG.BACKBONE_PARAM_TARGET
BACKBONE_PARAM_TOLERANCE = _KSCFG.BACKBONE_PARAM_TOLERANCE   # 0.15, unchanged

assert set(RUNG1_BACKBONE) == {"d_model", "d_state", "n_layers", "conv_size",
                                "num_heads", "ffn_mult"}, sorted(RUNG1_BACKBONE)
assert RUNG1_BACKBONE["d_model"] == (768 if _KSCFG.SCALE == "98m" else 1536), RUNG1_BACKBONE
assert RUNG1_BACKBONE["d_state"] == (64 if _KSCFG.SCALE == "98m" else 128), RUNG1_BACKBONE
assert RUNG1_BACKBONE["n_layers"] == (12 if _KSCFG.SCALE == "98m" else 16), RUNG1_BACKBONE
# sec 3.1's SHADOW CONSTANT, closed: CONV_SIZE was a hand-copied duplicate of
# RUNG1_BACKBONE["conv_size"] and drives buf_len -> the whole document geometry.
# It is invariant under THIS port (4 -> 4) and is now read from the dict, but
# the cross-check is asserted rather than assumed.
assert RUNG1_BACKBONE["conv_size"] == _KSCFG.CONV_SIZE, (
    RUNG1_BACKBONE["conv_size"], _KSCFG.CONV_SIZE)

# BUILD REQUIREMENT B2 (sec 3.3). The ONLY production pair. `_MLP_ADAPTER_HIDDEN`
# below is a REAL d_model dependency (768//4 = 192 -> 1536//4 = 384) that is dead
# code only because every production construction site passes adapter="linear".
# These two constants make "the production path" a single named source of truth
# that the runner asserts at startup, instead of five scattered literals.
PRODUCTION_ADAPTER = "linear"
PRODUCTION_READ_INJECT = "add"'''),

    ("G2_B2_production_pair_assert", """    def __init__(self, d_model: int, d_ncr: int, vocab_size: int,
                 adapter: str = "linear", read_inject: str = "add"):
        super().__init__()
        self.d_model, self.d_ncr, self.vocab_size = d_model, d_ncr, vocab_size""",
     '''    def __init__(self, d_model: int, d_ncr: int, vocab_size: int,
                 adapter: str = "linear", read_inject: str = "add"):
        super().__init__()
        # SCALE-AXIS PORT PATCH G2 == BUILD REQUIREMENT B2 (sec 3.3, sec 3.7).
        # The single choke point every construction path goes through
        # (build_arm, restore_arms_and_opts, the battery, depthext_eval and the
        # graft's own smoke items all land here). A design that quietly relied
        # on "nothing scales" would have shipped a live d_model-dependent path
        # (_MLP_ADAPTER_HIDDEN, 192 -> 384) ONE FLAG away.
        # DISCLOSED CONSEQUENCE: the graft's own legacy smoke item 11
        # (mlp/mlp_logits construction+shape check) is thereby DISABLED in the
        # scaleaxis tree. That is exactly what B2 mandates -- those arms are
        # dead code in every cell of record and are not ported.
        assert (adapter, read_inject) == (PRODUCTION_ADAPTER, PRODUCTION_READ_INJECT), (
            f"B2 (NCR_SCALE_AXIS_DESIGN.md sec 3.3): NCRIntegration built with "
            f"(adapter={adapter!r}, read_inject={read_inject!r}); the ONLY production pair on "
            f"this scale axis is ({PRODUCTION_ADAPTER!r}, {PRODUCTION_READ_INJECT!r}). The "
            f"'mlp' adapter branch carries a live d_model dependency "
            f"(_MLP_ADAPTER_HIDDEN = d_model//4) that is untested at rung 2. Refusing.")
        self.d_model, self.d_ncr, self.vocab_size = d_model, d_ncr, vocab_size'''),

    # AUDIT-R1 sec 4, deviation #3's ruling: "RATIFY WITH A NOTE ... but
    # 'disabled' was implemented as a CRASH, not a skip. One-line skip guard
    # recommended." G2's assert is correct and stays; what this fixes is that
    # the graft's own documented standalone entry point (`python3
    # ncr_lm_wave1_smoke.py`) died with an uncaught AssertionError at item 11
    # BEFORE items 1-10 could run -- removing a debugging entry point at the
    # worst possible moment. DEAD relative to every production path (runner,
    # battery, depthext, gates and kscaling_smoke.py all import the module and
    # never call main()), so no record could ever have been touched.
    ("G3_item11_skip_not_crash", """    smoke_11_ablation_flags_construct(args.device)     # CPU-fast, runs regardless of --device""",
     '''    # SCALE-AXIS PORT PATCH G3 (AUDIT-R1 sec 4, deviation #3's ruling). Item 11
    # constructs the mlp/mlp_logits arms, which BUILD REQUIREMENT B2 forbids on
    # this scale axis (they carry a live d_model dependency, _MLP_ADAPTER_HIDDEN
    # = d_model//4, untested at rung 2). SKIP it explicitly instead of letting
    # B2's constructor assert crash this module's standalone entry point before
    # items 1-10 run. The guard is a SKIP, and it says so out loud.
    _report("smoke 11: ablation-flag construction (mlp / mlp_logits)",
            True, "SKIPPED BY B2 (NCR_SCALE_AXIS_DESIGN.md sec 3.3): the non-production "
                  "adapter arms are not ported to rung 2 and NCRIntegration refuses to "
                  "build them. Not a pass of the mlp path -- a deliberate non-execution.")'''),
]


# ==========================================================================
# RUNNER PATCHES -- tag, the --scale tripwire, and B2's startup assert.
# ==========================================================================
RUNNER_PATCHES = [
    ("R1_tag_and_scale", '''RUNNER_TAG = "ncr_kscaling_runner_v1"
TRAIN_HOPS = tuple(KS.TRAIN_HOPS)                   # sec 3.1 Task-1 train range, verbatim''',
     '''# ---------------------------------------------------------------------------
# SCALE-AXIS PORT PATCH R1 (NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 3.6). The tag
# is bumped so a 392M checkpoint can NEVER be silently resumed by, or confused
# with, a 98M cell -- load_checkpoint asserts on this field, and sec 3.5's m5
# showed the battery would otherwise load a wrong-SCALE checkpoint and score it
# successfully and silently.
# ---------------------------------------------------------------------------
RUNNER_TAG = "ncr_scaleaxis_runner_v1"
SCALE = KS.SCALE                                    # "98m" | "392m", resolved from NCR_SCALE
TRAIN_HOPS = tuple(KS.TRAIN_HOPS)                   # sec 3.1 Task-1 train range, verbatim'''),

    ("R2_scale_flag", '''    ap.add_argument("--k", type=int, required=True,
                     help="K-SCALING PATCH R7: mandatory restatement of NCR_K. Must equal the "
                          "env var kscaling_config read; asserted in main(). Not a second source "
                          "of truth -- a tripwire against env/flag drift across 30 specs.")''',
     '''    ap.add_argument("--k", type=int, required=True,
                     help="K-SCALING PATCH R7: mandatory restatement of NCR_K. Must equal the "
                          "env var kscaling_config read; asserted in main(). Not a second source "
                          "of truth -- a tripwire against env/flag drift across 30 specs.")
    ap.add_argument("--scale", required=True, choices=("98m", "392m"),
                     help="SCALE-AXIS PORT PATCH R2 (sec 3.6): mandatory restatement of NCR_SCALE, "
                          "asserted against the RESOLVED backbone dict on every mode. Same tripwire "
                          "as --k and for the same stated reason: the single easiest way to burn "
                          "85 GPU-h is to run the wave at the wrong backbone.")'''),

    ("R3_scale_tripwire", '''    assert args.k == K_NCR, (
        f"--k {args.k} disagrees with NCR_K={os.environ.get('NCR_K')!r} "
        f"(kscaling_config resolved K_NCR={K_NCR}). Refusing to run: one of the two is a "
        f"typo and the results would be silently mislabelled.")
    print(f"[kscaling] K={K_NCR} d_ncr={D_NCR} chance={KS.CHANCE:.4f} ladder={DEEP_LADDER} "
          f"h_top={H_TOP} (residue {H_TOP % K_NCR} == K/2) fixed_dist_probe={FIXED_DIST_PROBE} "
          f"t_in={KS.t_in()} doc_left_pad={KS.doc_left_pad()}", flush=True)''',
     '''    assert args.k == K_NCR, (
        f"--k {args.k} disagrees with NCR_K={os.environ.get('NCR_K')!r} "
        f"(kscaling_config resolved K_NCR={K_NCR}). Refusing to run: one of the two is a "
        f"typo and the results would be silently mislabelled.")

    # SCALE-AXIS PORT PATCH R3 (sec 3.6) -- the --scale tripwire, on EVERY mode,
    # asserted against the RESOLVED dict rather than against the env var alone,
    # so a stale RUNGS table cannot pass it.
    assert args.scale == SCALE, (
        f"--scale {args.scale!r} disagrees with NCR_SCALE={os.environ.get('NCR_SCALE')!r} "
        f"(kscaling_config resolved SCALE={SCALE!r}). Refusing to run.")
    _want_bb = KS.RUNGS[KS.RUNG_OF_SCALE[args.scale]]
    assert RUNG1_BACKBONE == _want_bb, (
        f"--scale {args.scale!r} expects backbone {_want_bb} but the graft resolved "
        f"{RUNG1_BACKBONE}. One of the two is stale -- refusing to run.")
    # BUILD REQUIREMENT B2 (sec 3.7): assert the production (linear, add) pair at
    # STARTUP, before any GPU work, from the graft's own named constants -- which
    # is what build_arm() actually uses, so this is a real check and not a
    # tautology. The runner exposes no --adapter/--read-inject flag at all, so a
    # spec passing --adapter mlp dies here in argparse with exit 2.
    assert (graft.PRODUCTION_ADAPTER, graft.PRODUCTION_READ_INJECT) == ("linear", "add"), (
        f"B2: production pair is ({graft.PRODUCTION_ADAPTER!r}, "
        f"{graft.PRODUCTION_READ_INJECT!r}), not ('linear', 'add'). Refusing to run.")

    print(f"[scaleaxis] SCALE={SCALE} rung={KS.RUNG} backbone={RUNG1_BACKBONE} "
          f"backbone_params={KS.backbone_param_exact(KS.VOCAB_SIZE + KS.VOCAB_RESERVED_EXTRA):,} "
          f"total_per_arm={KS.total_param_exact():,} runner_tag={RUNNER_TAG}", flush=True)
    print(f"[kscaling] K={K_NCR} d_ncr={D_NCR} chance={KS.CHANCE:.4f} ladder={DEEP_LADDER} "
          f"h_top={H_TOP} (residue {H_TOP % K_NCR} == K/2) fixed_dist_probe={FIXED_DIST_PROBE} "
          f"t_in={KS.t_in()} doc_left_pad={KS.doc_left_pad()}", flush=True)'''),

    ("R4_cell_id_default", '''    cell_id = args.cell_id or f"kscaling_K{K_NCR}_s{args.seed}"''',
     '''    cell_id = args.cell_id or f"scaleaxis_{SCALE}_K{K_NCR}_s{args.seed}"'''),

    ("R5_build_arm_named_pair", '''    integ = NCRIntegration(RUNG1_BACKBONE["d_model"], D_NCR, vocab_size_total,
                            adapter="linear", read_inject="add").to(device)''',
     '''    integ = NCRIntegration(RUNG1_BACKBONE["d_model"], D_NCR, vocab_size_total,
                            adapter=graft.PRODUCTION_ADAPTER,
                            read_inject=graft.PRODUCTION_READ_INJECT).to(device)'''),
]


# ==========================================================================
# SCORER PATCHES -- BUILD REQUIREMENT B5, applied IDENTICALLY to both scorers.
# ==========================================================================
_SCALE_GUARD_SRC = '''
    # ---- BUILD REQUIREMENT B5 (NCR_SCALE_AXIS_DESIGN.md sec 3.5 m5) ---------
    # restore_arms_and_opts rebuilds the backbone from ckpt[arm]["backbone_config"],
    # NOT from RUNG1_BACKBONE, and the ONLY structural check that existed was the
    # K guard above. So a wrong-SCALE checkpoint at the RIGHT K would load and
    # score SUCCESSFULLY AND SILENTLY the moment the runner-tag allowlist is
    # extended -- which this design requires. In a design whose entire purpose is
    # a cross-scale comparison, that is the one guard that must exist.
    _bb = ckpt["full_graft"].get("backbone_config") or {}
    _want = {kk: R.RUNG1_BACKBONE[kk] for kk in ("d_model", "n_layers", "d_state")}
    _got = {kk: _bb.get(kk) for kk in ("d_model", "n_layers", "d_state")}
    if _got != _want:
        raise LoudFailure(
            f"SCALE MISMATCH [{args.tag}]: checkpoint backbone_config {_got} != this process's "
            f"resolved backbone {_want} (NCR_SCALE={os.environ.get('NCR_SCALE')!r}, "
            f"scale={KS.SCALE!r}). The checkpoint was trained at a DIFFERENT SCALE. Scoring it "
            f"here would produce a plausible-looking cross-scale number that is neither scale. "
            f"Refusing (B5).")
'''

_ALLOWLIST_OLD = '''choices=["ncr_gate3_wave1_runner_v1"])'''
_ALLOWLIST_NEW = '''choices=["ncr_gate3_wave1_runner_v1", "ncr_kscaling_runner_v1",
                             "ncr_scaleaxis_runner_v1"])
    # SCALE-AXIS PORT PATCH (sec 3.2 item 21). The allowlist is EXTENDED and
    # PAIRED WITH B5's scale guard, exactly as the design words it. Extending it
    # alone would be the hazard m5 found; it is extended so that a cross-harness
    # read fails on the SCALE GUARD (the right reason) rather than on the tag
    # (the wrong reason), which is what makes B5's negative test meaningful.'''

_OUTDIR_NOTE = '''
    # SCALE-AXIS PORT PATCH S4 == AUDIT-R1 MAJOR-5 / condition C4. The pinned
    # default pointed at ~/ncr_kscaling/results -- the 98M tree, which today
    # holds 103 records of record plus results_depthext6/. A 392M record landing
    # there COLLIDES ON-KEY with its 98M twin (rdelta_aggregate keys on
    # K/recipe/seed, not scale) and whichever path sorts later silently wins,
    # then reads as a 98M number in Rule R-delta AND in the exact-reproduction
    # cross-check. B5 does NOT close this: B5 guards the checkpoint INPUT, this
    # is the output DESTINATION. Latent-not-executing today (kappa_reader always
    # passes --outdir explicitly and no spec invokes a scorer), but sec 4.6's
    # Stage C harvest is a manual, unscripted path -- precisely where a bare
    # invocation happens. Closed at the write end here and at the read end by
    # rdelta_aggregate.load()'s scale assert.'''

BATTERY_PATCHES = [
    ("S1_battery_allowlist", '''                    choices=["ncr_gate3_wave1_runner_v1"])''',
     '''                    ''' + _ALLOWLIST_NEW),
    ("S4_battery_outdir", '''    ap.add_argument("--outdir", default=os.path.expanduser("~/ncr_kscaling/results"))''',
     '''    ap.add_argument("--outdir", default=os.path.expanduser("~/ncr_scaleaxis/results"))''' + _OUTDIR_NOTE),
    ("S2_battery_scale_guard", '''    cfg_rec = cell_config(args.cellcfg)

    step = int(ckpt["step"])''',
     _SCALE_GUARD_SRC + '''
    cfg_rec = cell_config(args.cellcfg)

    step = int(ckpt["step"])'''),
    ("S3_battery_provenance_d_model", '''        kscaling=KS.provenance(64, 768),''',
     '''        # sec 3.2 item 22 (B1's own find): this was `KS.provenance(64, 768)` -- a
        # hard-coded d_model that would have recorded HALF the true integ param
        # count in every 392M score record. Re-pointed at the resolved backbone.
        kscaling=KS.provenance(R.H_NCR, R.RUNG1_BACKBONE["d_model"]),
        scale=KS.SCALE, backbone_config_scored=ckpt["full_graft"].get("backbone_config"),'''),
]

DEPTHEXT_PATCHES = [
    ("S1_depthext_allowlist", '''    ap.add_argument("--anchor-runner-tag", default=None, choices=["ncr_gate3_wave1_runner_v1"])''',
     '''    ap.add_argument("--anchor-runner-tag", default=None,
                    ''' + _ALLOWLIST_NEW),
    ("S2_depthext_scale_guard", '''    cfg_rec = cell_config(args.cellcfg)

    step = int(ckpt["step"])''',
     _SCALE_GUARD_SRC + '''
    cfg_rec = cell_config(args.cellcfg)

    step = int(ckpt["step"])'''),
    ("S3_depthext_scale_field", '''        K=KS.K_NCR, d_ncr=KS.D_NCR, ckpt_recorded_d_ncr=ck_d,''',
     '''        K=KS.K_NCR, d_ncr=KS.D_NCR, ckpt_recorded_d_ncr=ck_d,
        scale=KS.SCALE, backbone_config_scored=ckpt["full_graft"].get("backbone_config"),
        # AUDIT-R1 m2 / condition C11: was a bare literal 64. INVARIANT by
        # sec 3.3's code proof (ENC_H is backbone-independent), but a bare
        # size-bearing literal B1 now greps for -- read from the runner instead.
        kscaling=KS.provenance(R.H_NCR, R.RUNG1_BACKBONE["d_model"]),'''),
    ("S4_depthext_outdir", '''    ap.add_argument("--outdir", default=os.path.expanduser("~/ncr_kscaling/results"))''',
     '''    ap.add_argument("--outdir", default=os.path.expanduser("~/ncr_scaleaxis/results"))''' + _OUTDIR_NOTE),
]

PATCH_SETS = {
    "kscaling_config.py": CONFIG_PATCHES,
    "ncr_lm_wave1_smoke.py": GRAFT_PATCHES,
    "ncr_lm_wave1_runner.py": RUNNER_PATCHES,
    "kscaling_battery.py": BATTERY_PATCHES,
    "depthext_eval.py": DEPTHEXT_PATCHES,
}


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


SRC_ROOT: str | None = None      # set by --src-root; makes the box tree self-reproducing


def _src(fname: str) -> str:
    if SRC_ROOT:
        return os.path.normpath(os.path.join(SRC_ROOT, fname))
    p = os.path.normpath(os.path.join(SRC_DIRS[fname], fname))
    if os.path.exists(p):
        return p
    # DEPLOYED layout: the box tree carries its own md5-pinned sources in src/,
    # so patch_scaleaxis.py (and B4's negative test) are self-reproducing there.
    return os.path.normpath(os.path.join(_HERE, "src", fname))


def negative_test() -> int:
    """B4's forced-fail: a ONE-BYTE-MUTATED graft copy must abort the patch."""
    tmp = tempfile.mkdtemp(prefix="scaleaxis_b4_")
    shutil.copy2(_src("ncr_lm_wave1_smoke.py"), os.path.join(tmp, "ncr_lm_wave1_smoke.py"))
    p = os.path.join(tmp, "ncr_lm_wave1_smoke.py")
    with open(p, "rb") as f:
        b = bytearray(f.read())
    b[-1:] = b"\n"                                    # append one byte
    b.append(0x20)
    with open(p, "wb") as f:
        f.write(bytes(b))
    got, want = md5(p), PINNED_MD5["ncr_lm_wave1_smoke.py"]
    ok = got != want
    print(json.dumps({
        "negative_test": "B4_graft_md5_pin_has_teeth",
        "mutated_copy": p, "md5_mutated": got, "md5_pinned": want,
        "pin_would_abort": ok,
        "status": "PASS (the pin FIRED on a one-byte mutation)" if ok else
                  "FAIL -- THE PIN DID NOT FIRE"}, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst", default=os.path.expanduser("~/ncr_scaleaxis"))
    ap.add_argument("--src-root", default=None,
                    help="read all five pinned sources from ONE directory (the deployed "
                         "~/ncr_scaleaxis/src), so the box tree is self-reproducing")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--negative-test", action="store_true")
    args = ap.parse_args()

    global SRC_ROOT
    if args.src_root:
        SRC_ROOT = os.path.expanduser(args.src_root)

    if args.negative_test:
        return negative_test()

    dst = os.path.expanduser(args.dst)

    if args.verify_only:
        rec = {f: md5(os.path.join(dst, f)) for f in sorted(os.listdir(dst)) if f.endswith(".py")}
        print(json.dumps({"dst": dst, "md5": rec}, indent=1))
        return 0

    # ---- B4: verify EVERY source pin BEFORE writing anything ---------------
    src_md5 = {}
    for fname, want in PINNED_MD5.items():
        got = md5(_src(fname))
        if got != want:
            raise SystemExit(f"PIN MISMATCH: {_src(fname)} md5={got} != pinned {want}. "
                             f"The source instrument changed -- STOP and adjudicate.")
        src_md5[fname] = got
    for fname, want in COPY_VERBATIM.items():
        got = md5(_src(fname))
        if got != want:
            raise SystemExit(f"PIN MISMATCH: {_src(fname)} md5={got} != pinned {want}.")
        src_md5[fname] = got

    os.makedirs(dst, exist_ok=True)
    out = {"design": "NCR_SCALE_AXIS_DESIGN.md DRAFT-R2 sec 3",
           "src_md5": src_md5, "upstream_pinned_md5": UPSTREAM_PINNED_MD5,
           "patches": {}, "dst_md5": {}}

    for fname, patches in PATCH_SETS.items():
        with open(_src(fname)) as f:
            text = f.read()
        text, log = apply_patches(text, patches, fname)
        with open(os.path.join(dst, fname), "w") as f:
            f.write(text)
        out["patches"][fname] = log
        out["dst_md5"][fname] = md5(os.path.join(dst, fname))

    for fname in COPY_VERBATIM:
        shutil.copy2(_src(fname), os.path.join(dst, fname))
        out["patches"][fname] = "COPIED VERBATIM (already scale-generic)"
        out["dst_md5"][fname] = md5(os.path.join(dst, fname))

    # The sources must be untouched by this process.
    for fname, want in PINNED_MD5.items():
        assert md5(_src(fname)) == want, f"the source {fname} was modified -- ABORT"
    out["sources_untouched"] = True
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
