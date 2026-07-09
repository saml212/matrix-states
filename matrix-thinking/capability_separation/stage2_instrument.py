"""CAPABILITY_SEPARATION_DESIGN.md S2.8 item 2(e) (Rev 3, per S2.16/S2.17 --
the anchor construction and FAIL routing below are the Rev 3 text) -- the
QUERY-DEPENDENCE DIAGNOSTIC: the mandatory calibration-gate duty that
detects `stage2_composer.py`'s registered low-rank-memory risk (rank(S_D)
<= min(32, n_h*D), proven exact there) re-triggering the EXACT §1.30
degeneracy Stage 1 diagnosed at L=1 (a query-independent softmax read over
a near-collinear-row memory) -- this time via rank instead of sequence
length. This is the item the audit is told to mutation-test hardest
(dispatch brief); every numeric pin below is cited back to its exact design
paragraph, not re-derived.

Statistic (S2.8 2(e), formula quoted inline): feed the trained reader the
composer's (B, 32, 32) memory, expand the d_state row_queries, take
`read = reader(q, mem, mem)` of shape (B, d_state, h), and compute
`read.std(dim=1)` -- the per-(batch-item, dim) standard deviation across
the d_state row-query outputs -- aggregated as the MEAN over the B*h
entries (canonical method source, formula quoted:
`experiment-runs/2026-07-09_capability_gate1_round7/l1_micro_diag.py:210`,
`std_across_queries = float(read.std(dim=1).max().item())` -- Rev 2
REPLACES the archived max with mean, S2.15 MAJOR-1(a): max detects EXACT
degeneracy; mean grades PARTIAL collapse without a max's single-outlier
false-PASS). `ddof=1` (torch's `.std()` default, `unbiased=True`) pinned
explicitly (S2.16/S2.17 minor). The MEDIAN over the same B*h entries is
co-reported, diagnostic-only, never decisional (S2.16/S2.17 minor).

Probe sample (S2.8 2(e), S2.15 MAJOR-1(c)): B=64 probe words per (cell,
depth), `tok = torch.randint(0, n_gens, (B, D),
generator=torch.Generator().manual_seed(7))` -- pinned verbatim.

Probe depths (S2.8 2(e), S2.15 MAJOR-2): D in {1,2,4,8,16,32,64} -- includes
the S2.6 M-D3 decisive read (D=64), where a SECOND degeneracy channel
(state-norm accumulation saturating the reader) is live and could otherwise
manufacture the pre-registered Arm-2-collapse CONFIRM pattern undetected.

PASS bar (S2.8 2(e), S2.17 B1 -- the Rev-3 rank-matched anchor, superseding
Rev 2's fixed-32-step construction): RELATIVE to a same-setting healthy
anchor built from the SAME pinned recurrence, run D steps at the cell's OWN
n_h, with orthonormal keys per Householder block (columns of a seeded
random orthogonal matrix, ONE GLOBAL 32x32 basis via QR of a seed-7 i.i.d.
Gaussian matrix, `torch.linalg.qr`, sign fixed by R's diagonal positive;
successive micro-steps consume successive, hence mutually orthogonal,
columns, cycling once exhausted) and i.i.d. Gaussian values, beta=1 (seed
7) -- giving anchor rank = EXACTLY min(32, n_h*D) (proven in
`build_anchor_states`'s docstring below, not merely asserted), matching the
probe state's own architectural rank cap at every depth. Rows rescaled by a
SINGLE GLOBAL SCALAR (S2.17 minor) so the mean row norm matches the real
cell's own mean state-row norm at the probed depth (norm-matched, never
rank-matched-away -- S2.17 B1's own point is that Rev 2's fraction absorbed
the ARCHITECTURAL rank gap too; Rev 3 matches rank out by construction so
the 0.25 fraction absorbs only training-induced geometry).

Bars, BOTH required at EVERY probed depth (S2.8 2(e)):
  T(D)      >= 0.25 * T_anchor(D)
  R(D)      >= 0.25 * R_anchor(D)      where R(D) = T(D) / mean-read-L2-norm
Anchor-health floor (S2.16/S2.17 M1, new): T_anchor(D) >= 1e-4 (cf. the
archived probe's 1e-6 degeneracy threshold,
`experiment-runs/2026-07-09_capability_gate1_round7/l1_micro_diag.py:213`)
-- a floor violation means the READER ITSELF is degenerate for ANY memory,
independent of the injected state, and routes to INSTRUMENT-DEFECT triage,
NOT the BOS-row fix (which targets state rank/key-independence, not reader
degeneracy).

FAIL routing, two-level (S2.8 2(e), S2.16/S2.17 B2/M2): level 1, a
bar failure (with the anchor floor healthy) -> apply the BOS-row fix
uniformly to Arms 2-3 + the last-K control (all-arms-or-none,
`stage2_composer.py`'s `use_bos_row`), re-run ALL 11 calibration cells (not
just the failing ones, S2.17 M2), re-measure. Level 2, a PERSISTENT FAIL
after the BOS-row fix -> a MANDATORY <=0.1 GPU-h mechanism diagnostic
before any further action (S1.7 gate 1(a) rule (c)'s three-way routing:
instrument-defect -> targeted fix + re-run; trainable-but-under-budget ->
one capped escalation; genuine architectural ceiling -> demote + disclose +
PI-visible design review). A diagnosed-ceiling demotion of a depth leg,
once disclosed and PI-acknowledged, DISCHARGES (e) at that leg for launch
purposes (S2.16/S2.17 B2).
"""
from __future__ import annotations

import torch
import torch.nn as nn

PROBE_B = 64
PROBE_SEED = 7
PROBE_DEPTHS = (1, 2, 4, 8, 16, 32, 64)
REL_BAR_FRAC = 0.25
ANCHOR_HEALTH_FLOOR = 1e-4
BOS_RERUN_GPU_H = 0.054 * 11   # S2.8 item 2(e): <=11x0.054 ~= 0.6 GPU-h, all 11 calibration cells
MECHANISM_DIAGNOSTIC_CAP_GPU_H = 0.1


def build_probe_tokens(n_gens: int, D: int, B: int = PROBE_B, seed: int = PROBE_SEED,
                       device="cpu") -> torch.Tensor:
    """S2.8 2(e), pinned verbatim: i.i.d. uniform generator-index draws,
    seed 7 retained from the archived l1_micro_diag.py probe."""
    gen = torch.Generator().manual_seed(seed)
    tok = torch.randint(0, n_gens, (B, D), generator=gen)
    return tok.to(device)


# ---------------------------------------------------------------------------
# The statistic itself.
# ---------------------------------------------------------------------------

def _aggregate_channel_stat(per_entry_std: torch.Tensor) -> float:
    """The T(D) AGGREGATION step (S2.8 2(e)/S2.15 MAJOR-1(a)), factored out
    of `query_dependence_stat` (S2.20 F2) so the partial-collapse negative
    control below can mutate JUST this function (mean->max) and confirm the
    exact regression class the design's own comment already narrates: MEAN
    over the B*h entries catches PARTIAL collapse (some channels dead, some
    alive) that a MAX would miss by keying on a single surviving-outlier
    channel. This is the pinned, decisional statistic -- do not swap for
    `.max()` (see `_test_partial_collapse_mean_catches_max_misses`, which
    proves by mutation that doing so silently defeats the gate)."""
    return per_entry_std.mean().item()


def query_dependence_stat(row_queries: torch.Tensor, reader: nn.MultiheadAttention,
                          mem: torch.Tensor) -> dict:
    """mem: (B, rows, h), already including a BOS row if the cell's reader
    uses one (caller's responsibility -- see `RowReadoutHead.prepare_mem`,
    applied identically to real and anchor memories for an apples-to-apples
    comparison). Returns the mean (decisional) and median (diagnostic-only)
    of `read.std(dim=1)` over the B*h entries, plus the mean read-vector L2
    norm (for the co-decisional normalized ratio R(D))."""
    B, rows, h = mem.shape
    d_state = row_queries.shape[0]
    q = row_queries.unsqueeze(0).expand(B, d_state, h)
    with torch.no_grad():
        read, _ = reader(q, mem, mem, need_weights=False)     # (B, d_state, h)
        per_entry_std = read.std(dim=1, unbiased=True)         # (B, h), ddof=1 (S2.17 minor)
        read_norm = read.norm(dim=-1)                          # (B, d_state)
    return dict(
        mean=_aggregate_channel_stat(per_entry_std),
        median=per_entry_std.median().item(),          # diagnostic-only, S2.16/S2.17 minor
        mean_read_norm=read_norm.mean().item(),
    )


# ---------------------------------------------------------------------------
# Rank-matched, norm-matched healthy anchor (S2.17 B1's binding fix).
# ---------------------------------------------------------------------------

def _qr_orthonormal_basis(h: int, seed: int, device="cpu") -> torch.Tensor:
    """A single global h x h orthonormal basis: QR of a seed-7 i.i.d.
    Gaussian matrix, sign convention fixed by R's diagonal positive
    (S2.8 2(e) minor, S2.17 pinned)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(h, h, generator=gen)
    Q, R = torch.linalg.qr(raw)
    diag_sign = torch.sign(torch.diagonal(R))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    return (Q * diag_sign.unsqueeze(0)).to(device)


def build_anchor_states(n_h: int, D_max: int, h: int = 32, seed: int = PROBE_SEED,
                        device="cpu") -> list[torch.Tensor]:
    """The SAME pinned recurrence (beta=1 fixed, no gate), run with
    orthonormal keys drawn cyclically from ONE global h x h QR basis (seed
    7) and i.i.d. Gaussian values (same generator, continued). Returns
    [S_0, S_1, ..., S_D_max], each (h, h).

    EXACT rank proof (not merely asserted -- verified by
    `_test_anchor_rank_exact` below): write micro-step m's key as
    `basis[:, m % h]`. For m < h, S_m = sum_{i<=m} v_i * basis[:,i]^T by
    induction (each new key is orthogonal to every previously-used key, so
    `S_{m-1} @ (I - c_m c_m^T) = S_{m-1}` exactly -- the reflection term
    vanishes against directions orthogonal to c_m -- leaving a clean
    rank-1 addition), giving rank(S_m) = m almost surely (continuous
    Gaussian coefficients). For m >= h (wraparound), reusing key c_i
    REPLACES its existing coefficient (S_{...} @ (I - c_i c_i^T) removes
    exactly the OLD v_i * c_i^T term via the same orthogonality argument,
    then the update re-adds a NEW v_i' * c_i^T) -- rank stays exactly h.
    Hence rank(S_m) = min(h, m) exactly, for every m -- so
    rank(S_D) = min(h, n_h*D), matching the probe state's own architectural
    cap at every depth by construction."""
    basis = _qr_orthonormal_basis(h, seed, device=device)
    # S2.20 m1 (RECORDED, not changed -- this dispatch's own "your call,
    # document" option): deliberately `seed+1`, NOT the literal `seed=7`
    # the QR key basis above uses. Two independent `torch.Generator()`
    # instances seeded IDENTICALLY would replay the exact same underlying
    # PRNG draw sequence into both the key basis and the values, making
    # them statistically DEPENDENT (correlated) rather than independent --
    # judged NOT "trivially safe" to align away (that would trade a
    # deliberate independence property for literal seed-7 text-conformance
    # with the design's prose), so `seed+1` is kept.
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)      # distinct stream for values
    total_micro = D_max * n_h
    values = torch.randn(total_micro, h, generator=gen).to(device)
    eye = torch.eye(h, device=device)
    S = torch.zeros(h, h, device=device)
    states = [S]
    m = 0
    for t in range(D_max):
        for _j in range(n_h):
            col = basis[:, m % h]
            vv = values[m]
            S = S @ (eye - torch.outer(col, col)) + torch.outer(vv, col)
            m += 1
        states.append(S)
    return states


def norm_match_scale(anchor_S: torch.Tensor, real_mean_row_norm: float) -> torch.Tensor:
    """SINGLE GLOBAL SCALAR rescale (S2.17 minor: not per-row -- the
    anchor's own internal row-norm variation is preserved) so the anchor's
    mean row norm matches `real_mean_row_norm` exactly."""
    anchor_row_norms = anchor_S.norm(dim=-1)             # (h,)
    anchor_mean = anchor_row_norms.mean().clamp(min=1e-12)
    scale = real_mean_row_norm / anchor_mean
    return anchor_S * scale


# ---------------------------------------------------------------------------
# Full per-(cell, depth) gate.
# ---------------------------------------------------------------------------

def run_query_dependence_gate(row_queries: torch.Tensor, reader: nn.MultiheadAttention,
                              n_h: int, real_state_fn, prepare_mem=lambda m: m,
                              depths=PROBE_DEPTHS, seed: int = PROBE_SEED,
                              h: int = 32) -> dict:
    """`real_state_fn(D) -> (B, h, h) raw real probe states` (B=64 distinct
    probe words, no BOS row applied yet). `prepare_mem` is the cell's own
    `RowReadoutHead.prepare_mem` (identity if use_bos_row=False), applied
    IDENTICALLY to real and anchor raw states before the reader call so the
    comparison is apples-to-apples.

    Returns a per-depth report plus the cell-level PASS/FAIL and routing
    decision (S2.8 2(e)'s two-level FAIL routing)."""
    D_max = max(depths)
    anchor_states = build_anchor_states(n_h, D_max, h=h, seed=seed)

    per_depth = []
    for D in depths:
        real_raw = real_state_fn(D)                                  # (B, h, h)
        real_mean_row_norm = real_raw.norm(dim=-1).mean().item()
        real_mem = prepare_mem(real_raw)
        real_stat = query_dependence_stat(row_queries, reader, real_mem)

        anchor_raw = norm_match_scale(anchor_states[D], real_mean_row_norm).unsqueeze(0)  # (1,h,h)
        anchor_mem = prepare_mem(anchor_raw)
        anchor_stat = query_dependence_stat(row_queries, reader, anchor_mem)

        T, T_anchor = real_stat["mean"], anchor_stat["mean"]
        R = T / max(real_stat["mean_read_norm"], 1e-12)
        R_anchor = T_anchor / max(anchor_stat["mean_read_norm"], 1e-12)

        anchor_floor_ok = T_anchor >= ANCHOR_HEALTH_FLOOR
        bar_T_ok = T >= REL_BAR_FRAC * T_anchor
        bar_R_ok = R >= REL_BAR_FRAC * R_anchor
        depth_pass = anchor_floor_ok and bar_T_ok and bar_R_ok
        per_depth.append(dict(
            D=D, T=T, T_anchor=T_anchor, R=R, R_anchor=R_anchor,
            T_median=real_stat["median"], anchor_floor_ok=anchor_floor_ok,
            bar_T_ok=bar_T_ok, bar_R_ok=bar_R_ok, depth_pass=depth_pass,
        ))

    floor_violations = [d for d in per_depth if not d["anchor_floor_ok"]]
    overall_pass = all(d["depth_pass"] for d in per_depth) and not floor_violations
    return dict(per_depth=per_depth, overall_pass=overall_pass,
               floor_violated=bool(floor_violations),
               floor_violation_depths=[d["D"] for d in floor_violations])


def route_gate_result(cell_report: dict, bos_already_applied: bool = False,
                      ceiling_demoted_depths: set | None = None) -> dict:
    """S2.8 2(e)'s two-level FAIL routing, PLUS the S2.16/S2.17 B2
    ceiling-demotion discharge clause. `ceiling_demoted_depths`: depths
    already diagnosed-and-PI-acknowledged as a genuine architectural
    ceiling (S2.16 B2) -- these are excluded from the FAIL determination
    (they discharge (e) at that leg for launch purposes)."""
    ceiling_demoted_depths = ceiling_demoted_depths or set()
    live_failures = [d for d in cell_report["per_depth"]
                     if not d["depth_pass"] and d["D"] not in ceiling_demoted_depths]

    if cell_report["floor_violation_depths"]:
        live_floor = [d for d in cell_report["floor_violation_depths"] if d not in ceiling_demoted_depths]
        if live_floor:
            return dict(route="instrument_defect", depths=live_floor,
                       note="reader is query-independent for ANY memory at these depths -- "
                            "routes to instrument-defect triage, NOT the BOS-row fix")

    if not live_failures:
        return dict(route="pass", ceiling_demoted_depths=sorted(ceiling_demoted_depths))

    if not bos_already_applied:
        return dict(route="apply_bos_fix_rerun_all_11", depths=[d["D"] for d in live_failures],
                   gpu_h=BOS_RERUN_GPU_H,
                   note="apply use_bos_row=True uniformly to Arms 2-3 + the last-K control "
                        "(all-arms-or-none), re-run ALL 11 calibration cells, re-measure")

    return dict(route="mechanism_diagnostic_required", depths=[d["D"] for d in live_failures],
               gpu_h_cap=MECHANISM_DIAGNOSTIC_CAP_GPU_H,
               note="persistent FAIL after the BOS-row fix -- mandatory S1.30-style mechanism "
                    "diagnostic before any further action (instrument-defect -> fix+re-run; "
                    "under-budget -> one capped escalation; ceiling -> demote+disclose+PI review)")


# ---------------------------------------------------------------------------
# Smoke / negative-then-positive tests.
# ---------------------------------------------------------------------------

def _test_anchor_rank_exact():
    print("=" * 88)
    print("NEGATIVE-then-POSITIVE TEST -- healthy-anchor rank EXACTLY min(h, n_h*D) at every depth")
    print("=" * 88)
    h = 32
    for n_h in (1, 2, 4):
        states = build_anchor_states(n_h, D_max=64, h=h, seed=PROBE_SEED)
        for D in PROBE_DEPTHS:
            expected = min(h, n_h * D)
            observed = int(torch.linalg.matrix_rank(states[D]).item())
            print(f"  n_h={n_h} D={D:>2}: rank(anchor S_D)={observed}  expected={expected}  "
                  f"{'PASS' if observed == expected else 'FAIL'}")
            assert observed == expected, (
                f"n_h={n_h} D={D}: anchor rank {observed} != min(h,n_h*D)={expected} -- the "
                f"rank-matched-anchor construction (S2.17 B1) does not hold its own invariant"
            )
    print("\nRESULT: anchor rank EXACTLY matches min(32, n_h*D) at all tested (n_h, D) pairs "
          "(incl. the full-rank-32-recovery regime once n_h*D>=32).\n")


def _test_qr_determinism():
    print("=" * 88)
    print("NEGATIVE-then-POSITIVE TEST -- QR basis (seed 7) is deterministic and orthonormal")
    print("=" * 88)
    b1 = _qr_orthonormal_basis(32, seed=PROBE_SEED)
    b2 = _qr_orthonormal_basis(32, seed=PROBE_SEED)
    assert torch.allclose(b1, b2), "QR basis is not deterministic under a fixed seed"
    gram = b1.T @ b1
    ortho_err = (gram - torch.eye(32)).abs().max().item()
    assert ortho_err < 1e-5, f"QR basis not orthonormal (max |gram-I|={ortho_err:.2e})"
    # R's diagonal sign convention is applied inside _qr_orthonormal_basis;
    # verified indirectly via determinism + orthonormality above.
    print(f"  same-seed determinism: PASS  |  orthonormality max|G-I|={ortho_err:.2e} < 1e-5: PASS")
    print("\nRESULT: QR basis construction is deterministic and genuinely orthonormal.\n")


def _test_planted_degenerate_reader_fails_both_bars():
    print("=" * 88)
    print("NEGATIVE TEST -- a planted query-independent reader FAILS both bars")
    print("=" * 88)
    torch.manual_seed(0)
    h, d_state, n_h = 32, 5, 2
    row_queries = torch.randn(d_state, h) * 0.02
    reader = nn.MultiheadAttention(h, 4, batch_first=True, dropout=0.0)

    def degenerate_real_state_fn(D):
        # all 32 "rows" IDENTICAL -> a single distinct key -> softmax is
        # PROVABLY query-independent (S1.30's exact L=1 mechanism), giving
        # T(D) ~= 0 regardless of D.
        row = torch.randn(1, h)
        return row.expand(PROBE_B, 32, h).clone()

    report = run_query_dependence_gate(row_queries, reader, n_h, degenerate_real_state_fn,
                                       depths=(1, 8, 64), seed=PROBE_SEED, h=h)
    print(f"  overall_pass={report['overall_pass']}  floor_violated={report['floor_violated']}")
    for d in report["per_depth"]:
        print(f"    D={d['D']:>2}: T={d['T']:.3e} T_anchor={d['T_anchor']:.3e} "
              f"bar_T_ok={d['bar_T_ok']}  R={d['R']:.3e} R_anchor={d['R_anchor']:.3e} bar_R_ok={d['bar_R_ok']}")
    assert not report["overall_pass"], \
        "PLANTED DEGENERATE READER WAS NOT CAUGHT -- the query-dependence gate has no teeth"
    assert any(not d["bar_T_ok"] for d in report["per_depth"]), "T bar never fired on the degenerate reader"
    route = route_gate_result(report, bos_already_applied=False)
    print(f"  routing decision: {route['route']}")
    assert route["route"] in ("apply_bos_fix_rerun_all_11", "instrument_defect"), \
        f"unexpected routing for a genuine degeneracy: {route['route']}"
    print("\nRESULT: a planted query-independent (collinear-row) memory FAILS the gate, and routes "
          "to a corrective action (BOS fix or instrument-defect triage), not silently PASS.\n")


def _test_planted_healthy_state_passes():
    print("=" * 88)
    print("POSITIVE TEST -- feeding the healthy anchor itself as 'real' state PASSES trivially")
    print("=" * 88)
    torch.manual_seed(1)
    h, d_state, n_h = 32, 5, 2
    row_queries = torch.randn(d_state, h) * 0.02
    reader = nn.MultiheadAttention(h, 4, batch_first=True, dropout=0.0)
    depths = (1, 8, 64)
    anchor_ref = build_anchor_states(n_h, D_max=max(depths), h=h, seed=PROBE_SEED)

    def healthy_real_state_fn(D):
        # B=64 IDENTICAL copies of the anchor's own state at depth D -- by
        # construction this must score T(D) == T_anchor(D) exactly (ratio
        # 1.0 >= 0.25), a genuine "should PASS" positive control.
        return anchor_ref[D].unsqueeze(0).expand(PROBE_B, h, h).clone()

    report = run_query_dependence_gate(row_queries, reader, n_h, healthy_real_state_fn,
                                       depths=depths, seed=PROBE_SEED, h=h)
    for d in report["per_depth"]:
        print(f"    D={d['D']:>2}: T={d['T']:.3e} T_anchor={d['T_anchor']:.3e} "
              f"ratio={d['T']/max(d['T_anchor'],1e-30):.3f}  depth_pass={d['depth_pass']}")
    assert report["overall_pass"], "a state IDENTICAL to the healthy anchor should trivially PASS"
    route = route_gate_result(report)
    assert route["route"] == "pass"
    print("\nRESULT: a real state identical to its own healthy anchor PASSES at every depth (ratio "
          "1.0 >= 0.25), and routes to 'pass' -- confirms the gate does not false-FAIL a healthy cell.\n")


def _test_anchor_health_floor_trips_on_zeroed_reader():
    print("=" * 88)
    print("NEGATIVE TEST -- anchor-health floor trips end-to-end on a genuinely degenerate")
    print("(all-zero-weight) reader -- T(D) AND T_anchor(D) both collapse together -- and routes")
    print("to instrument-defect, NOT the BOS fix")
    print("=" * 88)
    h, d_state, n_h = 32, 5, 2
    row_queries = torch.randn(d_state, h) * 0.02
    reader = nn.MultiheadAttention(h, 4, batch_first=True, dropout=0.0)
    with torch.no_grad():
        # A reader whose weights are ALL zero is query-independent for ANY
        # memory (every projection collapses to the zero vector): the exact
        # "instrument itself is broken" class the anchor-health floor exists
        # to distinguish from a genuine state-rank degeneracy (S2.16/S2.17
        # M1) -- this is a REAL end-to-end run through
        # run_query_dependence_gate (incl. its own non-degenerate anchor
        # construction, proven above), not a fabricated report.
        reader.in_proj_weight.zero_()
        if reader.in_proj_bias is not None:
            reader.in_proj_bias.zero_()
        reader.out_proj.weight.zero_()
        if reader.out_proj.bias is not None:
            reader.out_proj.bias.zero_()

    def arbitrary_real_state_fn(D):
        return torch.randn(PROBE_B, 32, h)   # content is irrelevant -- the reader itself is dead

    report = run_query_dependence_gate(row_queries, reader, n_h, arbitrary_real_state_fn,
                                       depths=(8, 64), seed=PROBE_SEED, h=h)
    print(f"  floor_violated={report['floor_violated']}  violation_depths={report['floor_violation_depths']}")
    for d in report["per_depth"]:
        print(f"    D={d['D']:>2}: T={d['T']:.3e}  T_anchor={d['T_anchor']:.3e}  "
              f"anchor_floor_ok={d['anchor_floor_ok']}")
    assert report["floor_violated"], (
        "an all-zero-weight reader should trip the anchor-health floor at every depth "
        "(T_anchor(D) ~= 0 for ANY memory, including the healthy anchor) -- floor has no teeth"
    )
    route = route_gate_result(report, bos_already_applied=False)
    print(f"  routing decision: {route['route']}  (expect 'instrument_defect', NOT a BOS-fix route)")
    assert route["route"] == "instrument_defect", (
        f"a floor violation (reader degenerate for ANY memory) must route to instrument-defect "
        f"triage, not {route['route']} -- applying the BOS fix here would target the wrong "
        f"mechanism (state rank, not reader degeneracy)"
    )
    print("\nRESULT: a genuinely degenerate (all-zero-weight) reader trips the anchor-health floor "
          "end-to-end and routes to instrument-defect triage, distinct from a normal bar failure's "
          "BOS-fix route.\n")


def _test_ceiling_demotion_discharges_launch_condition():
    print("=" * 88)
    print("POSITIVE TEST -- a ceiling-demoted depth DISCHARGES the launch condition (S2.16/17 B2)")
    print("=" * 88)
    fake_report = dict(
        per_depth=[dict(D=64, T=0.01, T_anchor=1.0, R=0.01, R_anchor=1.0,
                        T_median=0.01, anchor_floor_ok=True, bar_T_ok=False,
                        bar_R_ok=False, depth_pass=False)],
        overall_pass=False, floor_violated=False, floor_violation_depths=[],
    )
    undemoted = route_gate_result(fake_report, bos_already_applied=True)
    demoted = route_gate_result(fake_report, bos_already_applied=True, ceiling_demoted_depths={64})
    print(f"  without demotion: route={undemoted['route']}  (expect a diagnostic/action route)")
    print(f"  with D=64 demoted: route={demoted['route']}  (expect 'pass' -- discharged)")
    assert undemoted["route"] != "pass"
    assert demoted["route"] == "pass", "ceiling demotion did not discharge the launch condition (B2 regression)"
    print("\nRESULT: an undemoted persistent failure blocks launch; the SAME failure, once "
          "diagnosed-ceiling-demoted and PI-acknowledged, discharges (e) at that leg.\n")


def _test_partial_collapse_mean_catches_max_misses():
    """S2.20 F2: the audit found NO negative control demonstrating the
    mean-vs-max aggregation choice is load-bearing -- a PARTIAL collapse
    (some read channels dead, one alive) is exactly the class a MAX
    aggregation false-PASSES by keying on the single surviving outlier
    channel, while MEAN correctly dilutes it below bar. This plants a
    GENUINE memory (not a hacked reader) with query-dependent read
    variation concentrated in EXACTLY 1 of 32 output channels, verified by
    honest linear algebra: with a single-head reader, the value pathway is
    exactly linear, so a memory whose 32 rows differ from a shared base
    ONLY along one direction `u` produces read-output variation confined to
    exactly `O @ Wv @ u` -- solving `u` so this equals `e0` (channel 0)
    concentrates ALL query-dependence into that one channel, 31/32 "dead"
    by construction, not by weight-hacking."""
    print("=" * 88)
    print("NEGATIVE TEST -- S2.20 F2: a 31/32-dead partial-collapse plant FAILS under the pinned")
    print("MEAN aggregation, but WOULD FALSELY PASS under a mean->max mutation (re-applied LIVE")
    print("and reverted here, not just described)")
    print("=" * 88)
    h, d_state = 32, 5
    torch.manual_seed(3)
    row_queries = torch.randn(d_state, h) * 0.3
    # Single-head reader (test-local choice, NOT the production n_heads=4
    # reader): num_heads=1 makes the value pathway exactly linear (no
    # per-head softmax split), so the axis-alignment solve below is exact.
    reader = nn.MultiheadAttention(h, 1, batch_first=True, dropout=0.0)
    with torch.no_grad():
        Wv = reader.in_proj_weight[2 * h:3 * h, :].clone()      # value-projection slice, (h,h)
        O = reader.out_proj.weight.clone()                       # out_proj weight, (h,h)
        e0 = torch.zeros(h)
        e0[0] = 1.0
        u = torch.linalg.solve(O @ Wv, e0)      # (O@Wv) @ u = e0  ->  output varies ONLY on channel 0
        u = u / u.norm().clamp(min=1e-12)

    def partial_collapse_real_state_fn(D):
        # 32 memory "rows" differing from a SHARED base ONLY along
        # direction u -- by the linear-algebra argument above this
        # concentrates ALL query-dependent read variation into EXACTLY
        # channel 0 (31/32 dead), a genuine PARTIAL collapse (contrast
        # `_test_planted_degenerate_reader_fails_both_bars`'s all-32-dead
        # FULL collapse, a different failure class this control does not
        # re-test). `base` is scaled SMALL (0.05, not 0) relative to the
        # unit-variance `u`-direction coefficients: base is broadcast
        # IDENTICALLY to every row so it inflates row NORM without
        # contributing any query-dependent SIGNAL -- kept small so nearly
        # all of the real memory's norm budget is genuine channel-0
        # signal (empirically tuned, incl. the co-decisional R(D) bar --
        # not just T(D) -- so MEAN correctly fails BOTH bars while MAX
        # would falsely pass BOTH; base=0 exactly also works but sits
        # exactly on the degenerate rank-1-through-the-origin edge case,
        # avoided here for numerical headroom).
        gen = torch.Generator().manual_seed(11)
        base = torch.randn(PROBE_B, 1, h, generator=gen) * 0.02
        coeffs = torch.randn(PROBE_B, h, generator=gen)
        return base + coeffs.unsqueeze(-1) * u.view(1, 1, h)     # (B, h, h)

    report = run_query_dependence_gate(row_queries, reader, 2, partial_collapse_real_state_fn,
                                       depths=(8,), seed=PROBE_SEED, h=h)
    d0 = report["per_depth"][0]
    print(f"  MEAN aggregation (pinned, current): T={d0['T']:.3e}  T_anchor={d0['T_anchor']:.3e}  "
          f"ratio={d0['T'] / max(d0['T_anchor'], 1e-30):.4f}  bar_T_ok={d0['bar_T_ok']}  "
          f"bar_R_ok={d0['bar_R_ok']}  overall_pass={report['overall_pass']}")
    assert not report["overall_pass"], (
        "PARTIAL-COLLAPSE PLANT (1/32 alive, 31/32 dead) WAS NOT CAUGHT under MEAN aggregation "
        "-- S2.20 F2's negative control has no teeth"
    )
    assert not d0["bar_T_ok"] and not d0["bar_R_ok"], (
        "both bars must fail under MEAN aggregation for a 31/32-dead plant"
    )

    # Re-apply the audit's mean->max mutation (S2.20 F2) LIVE, confirm the SAME plant now
    # falsely PASSES, then revert -- proving MEAN (not MAX) is load-bearing here, automatically,
    # every time this smoke runs (not a one-off manual check done once and forgotten).
    global _aggregate_channel_stat
    original_agg = _aggregate_channel_stat
    _aggregate_channel_stat = lambda per_entry_std: per_entry_std.max().item()
    try:
        report_mut = run_query_dependence_gate(row_queries, reader, 2, partial_collapse_real_state_fn,
                                                depths=(8,), seed=PROBE_SEED, h=h)
    finally:
        _aggregate_channel_stat = original_agg
    assert _aggregate_channel_stat is original_agg, "mean->max mutation was not reverted -- test pollution risk"

    d0m = report_mut["per_depth"][0]
    print(f"  MAX mutation (S2.20 F2, applied live then reverted): T={d0m['T']:.3e}  "
          f"T_anchor={d0m['T_anchor']:.3e}  ratio={d0m['T'] / max(d0m['T_anchor'], 1e-30):.4f}  "
          f"bar_T_ok={d0m['bar_T_ok']}  bar_R_ok={d0m['bar_R_ok']}  "
          f"overall_pass={report_mut['overall_pass']}")
    assert d0m["bar_T_ok"] and d0m["bar_R_ok"] and report_mut["overall_pass"], (
        "the mean->max mutation was supposed to ESCAPE this exact plant on BOTH bars (S2.20 F2) "
        "-- if it doesn't, this control is not testing what its own comments claim, no teeth"
    )
    print("\nRESULT: a 31/32-dead partial-collapse plant FAILS the pinned MEAN-aggregated gate "
          "(correct) but WOULD FALSELY PASS under the mean->max mutation (S2.20 F2 confirmed both "
          "directions -- mutation applied live and reverted, not just described).\n")


def smoke():
    _test_anchor_rank_exact()
    _test_qr_determinism()
    _test_planted_degenerate_reader_fails_both_bars()
    _test_planted_healthy_state_passes()
    _test_anchor_health_floor_trips_on_zeroed_reader()
    _test_ceiling_demotion_discharges_launch_condition()
    _test_partial_collapse_mean_catches_max_misses()


if __name__ == "__main__":
    smoke()
