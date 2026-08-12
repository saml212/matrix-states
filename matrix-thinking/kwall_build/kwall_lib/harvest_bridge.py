"""Bridge to the repo's own `ncr_earlyln_scale.harvest()` — the design's
own "Enforcement point, named" instruction: this design consumes
`per_K[K]["n_converged"]` (a count) and `n_seeds` (read as `n_completed`)
ONLY, never `rate`/`gate_eligible`/`gate1_label` (3-denominator fields,
§R5 KW6.15). No `harvest()` code patch is needed or written — G2's
canonical-path contract already makes `discover_seeds_by_K`'s existing
file-glob-presence count a status-based `n_completed` count by
construction (design §4).

Importing `ncr_earlyln_scale` needs `torch`/`numpy` but NOT CUDA — this
module runs in the orchestrator's own (CPU-fine) process; only the
SUBPROCESSES it dispatches need a GPU.
"""
from __future__ import annotations

import os
import sys

_NCR_DIR_CANDIDATES = [
    os.environ.get("KWALL_NCR_DIR", ""),
    "/home/nvidia/ncr",  # the box's own real path (job-108 convention)
]


def _import_ncr_earlyln_scale():
    for cand in _NCR_DIR_CANDIDATES:
        if cand and os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
    import ncr_earlyln_scale as m  # noqa: F401 -- import for side effect (path resolution)
    return m


def per_K_resolution(outdir: str, K_values) -> dict:
    """Returns {K: (n_completed, n_converged)} for every K in `K_values`,
    read against the canonical directory `outdir` via the repo's own
    `harvest()`/`discover_seeds_by_K` — never a re-implementation."""
    m = _import_ncr_earlyln_scale()
    seeds_by_K = m.discover_seeds_by_K(outdir)
    h = m.harvest(outdir, seeds_by_K)
    out = {}
    for K in K_values:
        pk = h["per_K"].get(K) or h["per_K"].get(str(K))
        if pk is None:
            out[K] = (0, 0)
            continue
        out[K] = (pk.get("n_seeds", 0), pk.get("n_converged", 0))
    return out
