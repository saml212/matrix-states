"""Pricing/ceiling/path constants — every number here is either lifted
verbatim from the design doc's own arithmetic or derived in a comment,
per build charter item (j) ("every fixture must be producible; derive
each from the charging rules, show the arithmetic").

Source: `NCR_KWALL_CHARACTERIZATION_DESIGN.md` §4 (ORCHESTRATOR CONTRACT,
"Gate check points", "True spend, worst case", job-spec template) and §6
(POOL-ELIGIBILITY STATEMENT).
"""

# ---------------------------------------------------------------------------
# Per-attempt CLI ceilings (design §4, "--ceiling-gpuh values"). These are
# BOTH the value passed to `ncr_earlyln_scale.py --ceiling-gpuh` AND the
# value the orchestrator's ledger charges at every gate check (KW4.4 closed
# — charged == enforced, no separate per-K figure feeds the gate).
PRIMARY_CEILING_GPUH = 1.20      # >= 2*nominal for every K in {26,28,30}
                                   # (worst-case floor: K=30, 2*0.5973=1.1946)
CONDITIONAL_CEILING_GPUH = 2.32   # >= 2*nominal for every K_trig in {26,28,30}
                                   # (worst-case floor: K=30, 2*1.1561=2.3121)

# ---------------------------------------------------------------------------
# Gate thresholds (design §4, "Gate check points, exact").
HARD_GATE_CAP_GPUH = 15.00        # realized_gpu_h + ceiling(attempt) <= this
RETRY_GATE_THRESHOLD_GPUH = 12.00  # attempt-2 dispatch also requires
                                     # realized_gpu_h < this (reserves the
                                     # residual 15.00-12.00=3.00h band for
                                     # completing outstanding first attempts)
DECLARED_POOL_CEILING_GPUH = 15.50  # the ONE ceiling the pool spec carries
                                      # = 15.20 (disclosed rounding of the
                                      # 15.0157h ledger bound) + 0.30
                                      # (stated supervisor margin)

# ---------------------------------------------------------------------------
# The two per-attempt leak terms (design §4, "True spend, worst case", J6).
TAU_GPUH = 0.0157   # combined per-attempt tail: eval-overhead (0.0126) +
                      # log_every=500 training-ceiling-check granularity
                      # (0.0031). Charged on top of `ceiling(N)` for the
                      # single truly-last admitted attempt if it returns.
STARTUP_ALLOWANCE_S_GPUH = 0.0053  # process-startup term a RECONSTRUCTED
                                     # COMPLETED row's measured elapsed_h
                                     # under-counts (subprocess timer starts
                                     # after os.makedirs/resume-check/
                                     # claim_config/manual_seed/model-build,
                                     # the orchestrator's own timer starts
                                     # before subprocess.run). Conservative
                                     # (upper) end of the 0.0003-0.0053h
                                     # range derived from the frozen
                                     # KW5.7 single-attempt-tail estimate
                                     # (0.016-0.021h total) minus tau.

# T <= R_N + 12*tau + 32*s = 15.0157 + 0.1884 + 0.1696 = 15.3737 GPU-h
#   R_N <= 15.00 + tau = 15.0157   (ledger bound)
#   12  = floor(15.0157 / PRIMARY_CEILING_GPUH) -- max # of ceiling_charged
#         rows that can exist under the ledger bound (Class 1)
#   32  = 16 cells * 2 attempts -- structural cap on Class-2 (measured
#         COMPLETED, ceiling_charged=false) reconstructed rows
TRUE_SPEND_WORST_CASE_GPUH = 15.3737
assert abs((15.0157 + 12 * TAU_GPUH + 32 * STARTUP_ALLOWANCE_S_GPUH)
           - TRUE_SPEND_WORST_CASE_GPUH) < 1e-9
assert TRUE_SPEND_WORST_CASE_GPUH < DECLARED_POOL_CEILING_GPUH  # 0.1263h margin

# Reachability cap for a single primary COMPLETED (measured) row's own
# elapsed_h: ceiling + tau + s = 1.20 + 0.0157 + 0.0053 = 1.2210 (R10/R11's
# KW11.1 fixture-producibility fix — no COMPLETED row can ever legitimately
# carry a larger elapsed_h than this).
PRIMARY_COMPLETED_REACHABILITY_CAP_GPUH = round(
    PRIMARY_CEILING_GPUH + TAU_GPUH + STARTUP_ALLOWANCE_S_GPUH, 10)
assert PRIMARY_COMPLETED_REACHABILITY_CAP_GPUH == 1.2210
CONDITIONAL_COMPLETED_REACHABILITY_CAP_GPUH = round(
    CONDITIONAL_CEILING_GPUH + TAU_GPUH + STARTUP_ALLOWANCE_S_GPUH, 10)

# ---------------------------------------------------------------------------
# Grid / cell shape.
K_VALUES_PRIMARY = (26, 28, 30)
SEEDS = (0, 1, 2, 3)
D_OVERRIDE = {K: K + 1 for K in (26, 28, 30, 32)}  # 27, 29, 31 (+32 for the
                                                     # $0 K_trig==32 branch,
                                                     # cited never dispatched)
STEPS_PRIMARY = 80_000
STEPS_CONDITIONAL = 160_000
H_FIXED = 64  # never binds in this range: min(d, h+1)=d always (design §2(d))

# ---------------------------------------------------------------------------
# Absolute paths (design §4, KW5.11 — every path this design specifies is
# `${NCR_ROOT}/...`, matching job-108's own absolute-path convention).
NCR_ROOT = "/home/nvidia/ncr"
PRIMARY_OUTDIR = f"{NCR_ROOT}/results_kwall_characterization"
CONDITIONAL_OUTDIR = f"{NCR_ROOT}/results_kwall_characterization_160k"
SMOKE_OUTDIR = f"{NCR_ROOT}/results_kwall_smoke"
LEDGER_PATH = f"{PRIMARY_OUTDIR}/ORCHESTRATOR_LEDGER.json"
REPORT_PATH = f"{PRIMARY_OUTDIR}/orchestrator_report.json"
STOP_FILE_PRIMARY = f"{PRIMARY_OUTDIR}/STOP"
STOP_FILE_CONDITIONAL = f"{CONDITIONAL_OUTDIR}/STOP"

# ---------------------------------------------------------------------------
# Enums (design §4, "Unified attempts[].status x run_status enum table" —
# the single source of truth).
ATTEMPT_STATUS_VALUES = {
    "COMPLETED", "ABORTED-BUDGET", "CRASHED", "CRASHED-RECOVERED",
    "GATE-REFUSED", "STOPPED-BY-OPERATOR",
}
RUN_STATUS_ACCEPT_SET = {
    "COMPLETE", "COMPLETE-DEGRADED", "EXHAUSTED-BUDGET",
    "EXHAUSTED-BUDGET-SUSPECT-OVERCHARGE",
}
RUN_STATUS_ALL = RUN_STATUS_ACCEPT_SET | {"STOPPED-BY-OPERATOR"}
BAND_LABELS = {
    "FRONTIER-AT-K*=24", "FRONTIER-AT-K*=26", "FRONTIER-AT-K*=28",
    "FRONTIER-AT-K*=30", "GRADUAL-DECAY", "NON-MONOTONE-UNRESOLVED",
    "INCOMPLETE-AT-K",
}
TRIGGER_RESOLUTIONS = {"unanimous", "tie-break-min", "TRIGGER-UNRESOLVED"}
QUALIFIER_BANDS = {
    "CONFIRMED-WALL-AT-160K", "SLOW-CONVERGENCE-AT-160K",
    "PARTIAL-IMPROVEMENT-AT-160K",
}

EPS = 1e-6  # the tolerance used throughout validity_check's float comparisons
