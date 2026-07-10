"""INDEPENDENT AUDIT of the uncommitted stage2_instrument.py device-boundary fix
(the one-line `.to(real_raw.device)` on anchor_raw in run_query_dependence_gate).

Claims proven, all RUN TO COMPLETION:

  A. CPU NUMERICAL INVARIANCE: on CPU -- where every prior certification of
     the T/R bars and the rank-matched anchor ran -- the fixed module's gate
     report is BIT-IDENTICAL (exact float ==) to the pre-fix HEAD version's,
     for identity AND BOS-style prepare_mem, at all seven pinned probe depths.

  B. CROSS-DEVICE TEETH (the negative test the CPU-only smoke cannot run),
     run ON THE BOX with `cuda` as argv[1] -- the production surface, where
     the pre-fix error is torch's clean, catchable "Expected all tensors to
     be on the same device" RuntimeError (logged 20x in
     stage2_calib_wave.log):
       B1: the PRE-FIX module (stage2_instrument_prefix.py, md5 6b26ee70...,
           byte-identical to the box's pre-fix deploy) must RAISE that exact
           RuntimeError class on a CUDA reader, identity AND BOS prepare_mem.
       B2: the FIXED module must complete with a finite, sane report whose
           T/T_anchor track same-process CPU values at every probed depth.

  DISCLOSED CONFOUND (why MPS was NOT used): torch 2.8.0's MPS backend
  hard-aborts (MPSNDArray "buffer is not large enough" Metal assertion, exit
  -6) on nn.MultiheadAttention with a stride-0 `expand`ed query -- the
  gate's own internal q construction -- with ALL tensors already on MPS and
  zero device mixing (isolated to a minimal all-MPS repro). MPS therefore
  cannot adjudicate this fix in either direction; both module versions die
  at the REAL-path reader call before the anchor path is reached.
"""
import importlib.util
import math
import os
import sys

import torch
import torch.nn as nn

# Local (Mac) layout by default; on the box both files sit next to this script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL_FIXED = "/Users/samuellarson/Experiments/learned-representations/matrix-thinking/capability_separation/stage2_instrument.py"
FIXED_PATH = _LOCAL_FIXED if os.path.exists(_LOCAL_FIXED) else os.path.join(_HERE, "stage2_instrument.py")
PREFIX_PATH = os.path.join(_HERE, "stage2_instrument_prefix.py")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H, D_STATE, N_H = 32, 5, 2
DEPTHS_MPS = (1, 8, 64)
FIELDS = ("T", "T_anchor", "R", "R_anchor", "T_median")


def make_setup(device):
    """Deterministic reader/queries/real-states, constructed on CPU with a
    fixed seed then moved -- identical values on every device and across both
    module versions."""
    torch.manual_seed(42)
    row_queries = (torch.randn(D_STATE, H) * 0.02).to(device)
    reader = nn.MultiheadAttention(H, 4, batch_first=True, dropout=0.0).to(device)
    bos = (torch.randn(H) * 0.02).to(device)

    def real_state_fn(D):
        gen = torch.Generator().manual_seed(1000 + D)
        return torch.randn(64, H, H, generator=gen).to(device)

    def bos_prepare_mem(mem):
        B = mem.shape[0]
        return torch.cat([bos.view(1, 1, H).expand(B, 1, H), mem], dim=1)

    return row_queries, reader, real_state_fn, bos_prepare_mem


def run_gate(mod, device, depths, use_bos):
    row_queries, reader, real_state_fn, bos_prep = make_setup(device)
    prep = bos_prep if use_bos else (lambda m: m)
    return mod.run_query_dependence_gate(
        row_queries, reader, N_H, real_state_fn, prepare_mem=prep,
        depths=depths, seed=mod.PROBE_SEED, h=H,
    )


# ---------------------------------------------------------------------------
# Main audit.
# ---------------------------------------------------------------------------
si_fixed = load_module("si_fixed", FIXED_PATH)
si_prefix = load_module("si_prefix", PREFIX_PATH)
DEPTHS_FULL = si_fixed.PROBE_DEPTHS          # (1,2,4,8,16,32,64)
RUN_CUDA_TEETH = len(sys.argv) >= 2 and sys.argv[1] == "cuda"

print("=" * 88)
print("SECTION A -- CPU bit-identity: fixed vs pre-fix HEAD, identity + BOS prepare_mem,")
print(f"all {len(DEPTHS_FULL)} pinned depths, exact float equality on {FIELDS}")
print("=" * 88)
for use_bos in (False, True):
    rep_fixed = run_gate(si_fixed, "cpu", DEPTHS_FULL, use_bos)
    rep_prefix = run_gate(si_prefix, "cpu", DEPTHS_FULL, use_bos)
    assert rep_fixed["overall_pass"] == rep_prefix["overall_pass"]
    assert rep_fixed["floor_violated"] == rep_prefix["floor_violated"]
    for df, dp in zip(rep_fixed["per_depth"], rep_prefix["per_depth"]):
        assert df["D"] == dp["D"]
        for f in FIELDS:
            assert df[f] == dp[f], (
                f"CPU DRIFT (use_bos={use_bos}) at D={df['D']} field {f}: "
                f"fixed={df[f]!r} != prefix={dp[f]!r} -- the fix changed 2(e) semantics on CPU"
            )
        for f in ("anchor_floor_ok", "bar_T_ok", "bar_R_ok", "depth_pass"):
            assert df[f] == dp[f]
    print(f"  use_bos={use_bos!s:5}: {len(DEPTHS_FULL)} depths x {len(FIELDS)} fields "
          f"EXACTLY equal (float ==); pass/floor flags identical.  PASS")
print("RESULT A: the fix is a numerical NO-OP on CPU -- T/R bars and the rank-matched "
      "anchor are bit-identical where all prior certification ran.\n")

if not RUN_CUDA_TEETH:
    print("SECTION B skipped on this machine (pass `cuda` as argv[1] on the box).")
    print("NOTE: MPS was evaluated and DISQUALIFIED as a stand-in device -- see the")
    print("docstring's disclosed confound (all-MPS expanded-q MHA hard-abort, no device")
    print("mixing involved). The CUDA leg is the production surface and must run on the box.")
    print("\nAUDIT TEETH SCRIPT: SECTION A PASSED (CPU bit-identity).")
    sys.exit(0)

assert torch.cuda.is_available(), "cuda mode requested but CUDA unavailable"
dev = "cuda:0"

print("=" * 88)
print("SECTION B -- cross-device teeth on the box's CUDA (the production surface)")
print("=" * 88)

for use_bos in (False, True):
    raised = False
    try:
        run_gate(si_prefix, dev, DEPTHS_MPS, use_bos)
    except RuntimeError as e:
        raised = True
        msg = str(e).lower()
        assert ("expected all tensors to be on the same device" in msg
                or "found at least two devices" in msg), (
            f"pre-fix raised, but NOT the device-mismatch class (use_bos={use_bos}): {e}"
        )
        print(f"  B1 use_bos={use_bos!s:5}: PRE-FIX module raises the exact device-mismatch "
              f"RuntimeError the wave logged 20x -- teeth confirmed.")
        print(f"       error head: {str(e).splitlines()[0][:110]}")
    assert raised, (
        f"PRE-FIX module did NOT raise on {dev} (use_bos={use_bos}) -- teeth test is not "
        f"reproducing the box failure; audit cannot certify the fix from it"
    )

for use_bos in (False, True):
    rep_dev = run_gate(si_fixed, dev, DEPTHS_MPS, use_bos)
    rep_cpu = run_gate(si_fixed, "cpu", DEPTHS_MPS, use_bos)
    for dd, dc in zip(rep_dev["per_depth"], rep_cpu["per_depth"]):
        assert dd["D"] == dc["D"]
        for f in FIELDS:
            assert math.isfinite(dd[f]), f"non-finite {f} on {dev} at D={dd['D']}"
        rel = abs(dd["T"] - dc["T"]) / max(abs(dc["T"]), 1e-12)
        rel_a = abs(dd["T_anchor"] - dc["T_anchor"]) / max(abs(dc["T_anchor"]), 1e-12)
        assert rel < 1e-2 and rel_a < 1e-2, (
            f"CUDA-vs-CPU drift too large at D={dd['D']} (use_bos={use_bos}): "
            f"T rel={rel:.2e}, T_anchor rel={rel_a:.2e}"
        )
    print(f"  B2 use_bos={use_bos!s:5}: FIXED module completes on {dev}; all fields finite; "
          f"T/T_anchor within 1e-2 rel of CPU at every depth.  PASS")

print("RESULT B: pre-fix raises the exact box failure (device-mismatch RuntimeError) on a "
      "CUDA reader; the fix resolves it with values tracking CPU.\n")
print("AUDIT TEETH SCRIPT: ALL SECTIONS PASSED.")
