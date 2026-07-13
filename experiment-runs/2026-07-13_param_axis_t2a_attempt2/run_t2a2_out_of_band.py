#!/usr/bin/env python3
"""T2a-2 (the untrained-init NEGATIVE CONTROL, PARAM_AXIS_SCALING_DESIGN.md
sec 11.4.2) run OUT-OF-BAND, on both REQUIRED corpora.

WHY OUT-OF-BAND, stated plainly rather than buried: the pinned `--gate`
driver computes T2a-2 only AFTER its full witness loop completes -- and that
loop includes C1 (falcon-mamba-7b), whose sequential non-fused Mamba eval ran
3h49m WITHOUT completing a single corpus in the sec 12 attempt and is
expected to cost ~8h across both. T2a-2 is a REQUIRED negative control of
this gate. Serializing a required control behind a DEMOTED witness's
multi-hour cell is an artifact of the driver's loop order, not a
pre-registration requirement.

THIS SCRIPT ADDS NO SCIENCE AND CHANGES NO INSTRUMENT CODE. It imports
`run_t2a2_untrained_control` from the pinned driver (commit 95ffba8,
UNMODIFIED) and calls it with the driver's own pinned defaults -- the same
function, same 14M architecture (RUNG_14M_CONFIG), same fixed
`init_seed=T2A2_UNTRAINED_INIT_SEED=314159`, same corpus seed offset, same
n_windows/n_plants (N_ROWS_DEFAULT=2048). The computation is DETERMINISTIC,
so this is an early read of exactly the value the inline run will itself
produce hours later -- not a substitute quantity, not a re-parameterization.

The bar (sec 11.4.2, NOT MOVED): an untrained model must read
`acc_copy <= 0.02` with a KS bootstrap CI INCLUDING 0. If an untrained model
PASSES the probe, the probe is passable with no learned mechanism =>
INSTRUMENT-INVALID, HALT. A passing negative control is a FAIL of the gate.
"""
import json
import sys

sys.path.insert(0, "/home/nvidia/chapter2/deltanet_rd")

from t2a_reference_driver_v2_rd import (  # noqa: E402
    run_t2a2_untrained_control, REQUIRED_CORPORA, T2A2_UNTRAINED_INIT_SEED,
    N_ROWS_DEFAULT, N_PLANTS_DEFAULT, SEQ_LEN_DEFAULT,
)

DATA_DIR = "/data/deltanet_rd_data"
OUT = "/home/nvidia/chapter2/deltanet_rd/results/param_axis_t2a_attempt2/t2a2_out_of_band.json"

results = {
    "provenance": {
        "why_out_of_band": "inline --gate serializes T2a-2 behind C1/falcon-mamba's ~8h "
                           "sequential-Mamba cell; T2a-2 is a REQUIRED control and is "
                           "deterministic, so it is read early here via the driver's own "
                           "pinned function, unmodified.",
        "init_seed": T2A2_UNTRAINED_INIT_SEED,
        "n_windows": N_ROWS_DEFAULT,
        "n_plants": N_PLANTS_DEFAULT,
        "seq_len": SEQ_LEN_DEFAULT,
        "bar": "sec 11.4.2: acc_copy <= 0.02 AND KS bootstrap CI includes 0. "
               "A PASS of the probe by an untrained model => INSTRUMENT-INVALID, HALT.",
    },
    "cells": {},
}

for corpus in REQUIRED_CORPORA:
    print(f"=== T2a-2 untrained-init negative control: {corpus} ===", flush=True)
    r = run_t2a2_untrained_control(corpus, DATA_DIR, "cuda")
    results["cells"][corpus] = r
    print(json.dumps(r.get("t2a2_untrained_control", {"t2_void": r.get("t2_void"),
                                                       "reason": r.get("t2_void_reason")}),
                     indent=1, default=str), flush=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, default=str)

print("\n=== T2a-2 SUMMARY ===")
for corpus, r in results["cells"].items():
    ctrl = r.get("t2a2_untrained_control") or {}
    print(f"  {corpus}: void={r.get('t2_void')} "
          f"passes(control_holds)={ctrl.get('passes')} "
          f"acc_copy={ctrl.get('acc_copy')} ks={ctrl.get('ks')} ks_ci={ctrl.get('ks_ci')}")
print(f"\nwrote {OUT}")
