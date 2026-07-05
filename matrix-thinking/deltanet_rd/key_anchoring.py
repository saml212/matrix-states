"""key_anchoring.py -- shared, FLA-FREE (pure torch) mechanism library for
KEY_ANCHORING_DESIGN.md (Rev 5, commit ca308d6). Every function here is
CPU-testable WITHOUT importing fla/chunk_delta_rule -- deliberately split out
of model_rd.py (which imports fla at module scope for the Triton kernel) so
every piece of THIS design's own new machinery is independently verifiable in
a plain CPU-torch environment: a throwaway venv with no fla at all (the exact
methodology KEY_ANCHORING_ATTACK_R1/R2/R3.md used for every round), AND the
box's own /home/nvidia/tdenv (which does have fla, but whose GPUs are all
busy on other waves -- CPU-only verification is the point, not a fallback).

model_rd.py's DeltaNetRDBlock imports anchor_blend_gather_scatter /
frame_potential_init / raw_table_conditioning / the LAMBDA_* constants /
classify_lambda_band FROM this module rather than reimplementing them, so the
function under test here IS the function bind() actually calls at its geo3
insertion site -- never a hand-copied twin with independent drift risk (the
project's own standing "verify before claiming" / no-silent-duplication
discipline).

Newton-Schulz reuse (flagged for audit scrutiny, a deliberate build
decision): Gate 2's G2-c leg and every CPU smoke that needs an
orthogonalization pass import `newton_schulz` from geo3_simulator.py rather
than `newton_schulz_orthogonalize` from model_rd.py. geo3_simulator.py is
ALSO fla-free by its own docstring ("No fla/chunk_delta_rule dependency --
pure torch, runs anywhere"); the two implementations are mathematically
IDENTICAL (same pre-scaled cubic Newton-Schulz iteration,
X_{t+1} = 1.5 X_t - 0.5 (X_t X_t^T) X_t, X_0 = A/sqrt(K)) and have been
cross-verified to convergence-precision agreement across every attack round
on this design (KEY_ANCHORING_ATTACK_R1/R2/R3.md). Using geo3_simulator's
copy here means Gate 2 (a committed CPU test) and the smoke suite never pull
fla into their import graph just to reach a pure-torch function; the
PRODUCTION bind() path (model_rd.py) still calls its own
geo3_orthogonalize_logged (the fallback-aware, side-channel-logging version)
unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import torch
import torch.nn.functional as F

from geo3_simulator import newton_schulz as _geo3sim_newton_schulz

# ---------------------------------------------------------------------------
# sec 2.2 / sec 3.2's registered constants (Rev 4 -- R3 finding 1; Rev 5
# header). Also imported by run_deltanet_rd.py (lambda-trajectory logging +
# the startup cadence assertion) and run_deltanet_rd_exactness_sweep.py (the
# band-derivation window / manifest naming) -- ONE definition, never
# re-typed at each call site.
# ---------------------------------------------------------------------------
LAMBDA_LOG_CADENCE_STEPS = 200
LAMBDA_WINDOW_LOG_POINTS = 5

# sec 4's Gate 2 legs (the amended, three-leg CPU construction gate).
GATE2_SIGMA_RATIO_MIN = 0.1        # G2-a: sigma_64/sigma_1 >= 0.1 (raw table, train-row block)
GATE2_MAX_COS_MAX = 0.5            # G2-b: max_{i!=j} |cos(A_i,A_j)| <= 0.5
GATE2_RESID_TOL = 1e-2             # G2-c: NS admission tolerance (sec 14.10 item 2 semantics)
GATE2_N_ITER_BY_K = {16: 12, 32: 20}   # production tier per K (sec 16.3's own n_iter escalation)
GATE2_N_SUBSETS = 512

# sec 2.2's anchor-init recipe (the frozen frame-potential construction).
ANCHOR_FRAME_POTENTIAL_N_STEPS = 4000
ANCHOR_FRAME_POTENTIAL_LR = 0.05
# Registered construction seed for the anchor table's frame-potential init
# (a build-time choice this wave registers -- the design text pins the
# RECIPE exactly but not a specific numeric seed value, see KEY_ANCHORING_
# ATTACK_R3.md Target 1's own "no seed pinned for [non-regression] cases"
# note; this constant is what makes THIS build's own draw reproducible).
# Swept seeds {1..5, 42, 20260703-5} at construction time: sigma_ratio is
# 1.0000 (the Welch-floor tight-frame property) at EVERY seed (a fixed
# invariant of frame-potential minimization, sec 2.2's own disclosure);
# max|cos| varies 0.278-0.326 across seeds (no exact (107,64) ETF exists,
# so max|cos| is draw-dependent even at the frame-potential optimum) --
# 20260705 was selected because it lands closest to the design's own
# cited session number (max|cos|=0.2832; this build measures 0.28415 at
# 20260705, see the build report / gate2_construction_test.py's printed
# diagnostic for the exact achieved value).
ANCHOR_INIT_SEED = 20260705


# ---------------------------------------------------------------------------
# sec 2.2: frame-potential-minimized anchor-table construction.
# ---------------------------------------------------------------------------

def frame_potential_init(n: int, d: int, seed: int,
                          n_steps: int = ANCHOR_FRAME_POTENTIAL_N_STEPS,
                          lr: float = ANCHOR_FRAME_POTENTIAL_LR) -> torch.Tensor:
    """sec 2.2's pinned recipe: start from seeded random unit rows (n,d);
    minimize the frame potential F(X) = sum_{i!=j} (x_i.x_j)^2 by projected
    gradient descent (grad_i = 4 * sum_{j!=i} (x_i.x_j) x_j, renormalize
    every step); n_steps=4000, lr=0.05 -- deterministic, seeded, CPU,
    one-time. Frame-potential minimizers are TIGHT FRAMES (the rms
    coherence this converges to is a fixed function of (n,d) alone -- the
    Welch floor sqrt((n-d)/(d(n-1))) -- regardless of which specific tight
    frame the descent lands in; max|cos| varies run to run within a small
    band since an exact equiangular tight frame need not exist at every
    (n,d), sec 2.2's own disclosure). fp64 throughout for the iterative
    descent's numerical stability; cast to fp32 on return (the table's
    actual storage dtype, matching every other frozen-table convention in
    this codebase, e.g. embed_arms.py's own QR constructions)."""
    g = torch.Generator().manual_seed(seed)
    X = F.normalize(torch.randn(n, d, generator=g, dtype=torch.float64), dim=-1)
    eye_mask = torch.eye(n, dtype=torch.bool)
    for _ in range(n_steps):
        gram = X @ X.transpose(0, 1)                    # (n,n)
        gram = gram.masked_fill(eye_mask, 0.0)           # exclude i==j from the i!=j sum
        grad = 4.0 * (gram @ X)                          # sum_{j!=i} (x_i.x_j) x_j per row i
        X = F.normalize(X - lr * grad, dim=-1)
    return X.float()


def raw_table_conditioning(table: torch.Tensor) -> dict:
    """items 6a/6b (sec 3.1, Rev 2 attack-F1 rewrite -- REWRITTEN to the RAW
    table, never anything downstream of Newton-Schulz, which forces
    sigma_K/sigma_1 ~= 1 on ANY input and is therefore tautologically
    blind). table: (n,d), the anchor table's TRAIN-ROW block (e.g. 107x64).
    6a: sigma_min/sigma_max via direct SVD (n>d here, so this is
    sigma_d/sigma_1 -- the house salvage-ratio convention, generalized off
    -square exactly the way salvage_ratio()/model_rd.py already does for
    (K,d) key matrices). 6b: max_{i!=j} |cos(row_i, row_j)|."""
    s = torch.linalg.svdvals(table.float())
    sigma_ratio = (s[-1] / s[0].clamp(min=1e-12)).item()
    row_norm = F.normalize(table.float(), dim=-1)
    cos = row_norm @ row_norm.transpose(0, 1)
    n = table.shape[0]
    off_diag = cos[~torch.eye(n, dtype=torch.bool, device=table.device)]
    max_abs_cos = off_diag.abs().max().item()
    return {
        "sigma_ratio": sigma_ratio, "max_abs_cos": max_abs_cos,
        "sigma_ratio_pass": bool(sigma_ratio >= GATE2_SIGMA_RATIO_MIN),
        "max_abs_cos_pass": bool(max_abs_cos <= GATE2_MAX_COS_MAX),
    }


# ---------------------------------------------------------------------------
# sec 4's Gate 2 -- three-leg CPU construction gate (G2-a/6a, G2-b/6b,
# G2-c NS-conditioning). Runs BOTH at wave-launch time (init) and again at
# every admission checkpoint on the then-current trained table (sec 4's
# "re-run at every admission checkpoint" build requirement, R2-M1).
# ---------------------------------------------------------------------------

def gate2_ns_leg(table: torch.Tensor, K: int, n_iter: int,
                  n_subsets: int = GATE2_N_SUBSETS, resid_tol: float = GATE2_RESID_TOL,
                  seed: int = 0) -> dict:
    """G2-c: >= n_subsets random K-subsets (without replacement) of the
    table's rows, through the SAME (mathematically) Newton-Schulz iteration
    at the production tier for this K (sec 16.3's n_iter=12 @ K=16 /
    n_iter=20 @ K=32), 100% converging to residual <= resid_tol with ZERO
    fallbacks (sec 14.10 item 2's admission semantics, unchanged)."""
    n = table.shape[0]
    assert K <= n, f"cannot draw K={K} rows without replacement from a {n}-row table"
    g = torch.Generator().manual_seed(seed)
    max_resid = 0.0
    n_fallback = 0
    for _ in range(n_subsets):
        idx = torch.randperm(n, generator=g)[:K]
        sub = F.normalize(table[idx], dim=-1).unsqueeze(0)     # (1,K,d)
        _, resid_hist = _geo3sim_newton_schulz(sub, n_iter)
        final_resid = resid_hist[-1]
        max_resid = max(max_resid, final_resid)
        if final_resid > resid_tol:
            n_fallback += 1
    return {"K": K, "n_iter": n_iter, "n_subsets": n_subsets,
            "n_fallback": n_fallback, "max_resid": max_resid,
            "pass": n_fallback == 0}


def gate2_construction_check(table: torch.Tensor, ks=(16, 32), seed: int = 0) -> dict:
    """The full amended Gate 2 (sec 4): all three legs mandatory, ANY leg
    failing blocks the launch. table: the anchor's TRAIN-ROW block only
    (n_train=107 x d_state=64 for the registered wave)."""
    cond = raw_table_conditioning(table)
    ns_legs = {K: gate2_ns_leg(table, K, GATE2_N_ITER_BY_K[K], seed=seed + K) for K in ks}
    overall_pass = (cond["sigma_ratio_pass"] and cond["max_abs_cos_pass"]
                     and all(v["pass"] for v in ns_legs.values()))
    return {
        "g2a_sigma_ratio": cond["sigma_ratio"], "g2a_pass": cond["sigma_ratio_pass"],
        "g2b_max_abs_cos": cond["max_abs_cos"], "g2b_pass": cond["max_abs_cos_pass"],
        "g2c_ns_legs": ns_legs,
        "pass": bool(overall_pass),
    }


# ---------------------------------------------------------------------------
# sec 4's pinned REGRESSION-CASE table constructions. Reconstructed from the
# design's own prose per KEY_ANCHORING_ATTACK_R3.md's Target-1 methodology
# (its own from-scratch reproduction independently matched the design's
# pinned 6a/6b numbers to 4 decimal places using this exact recipe -- "the
# most natural reading of the design's own prose"), so these functions are
# not a fresh guess; they are a verified-reproducible transcription.
# ---------------------------------------------------------------------------

def build_collapsed_table(n: int = 107, d: int = 64, noise_sigma: float = 0.30,
                            seed: int = 42) -> torch.Tensor:
    """sec 4's moderate (noise_sigma=0.30) / severe (noise_sigma=0.05)
    pinned collapse constructions: two shared unit directions + a random
    2-way row assignment + additive Gaussian noise, renormalized, ALL drawn
    from a SINGLE torch.Generator().manual_seed(seed) -- reproduces the
    design's pinned moderate-case numbers (G2-a=0.0762, G2-b=0.5371) and
    severe-case numbers (G2-a=0.0141, G2-b=0.9333) to 4 decimal places
    (KEY_ANCHORING_ATTACK_R3.md Target 1)."""
    g = torch.Generator().manual_seed(seed)
    directions = F.normalize(torch.randn(2, d, generator=g), dim=-1)
    assign = torch.randint(0, 2, (n,), generator=g)
    base = directions[assign]
    noise = torch.randn(n, d, generator=g) * noise_sigma
    return F.normalize(base + noise, dim=-1)


def build_localized_collapse_table(healthy_table: torch.Tensor, n_collapsed: int = 10,
                                     sigma: float = 0.02, seed: int = 0) -> torch.Tensor:
    """sec 4's localized-collapse context case: n_collapsed of the healthy
    (frame-potential) init's own rows replanted onto ONE shared direction
    with small noise sigma, renormalized -- the max-statistic (6b)
    dilution-immunity demonstration: 6a passes (aggregate dilution over the
    other 97 healthy rows), 6b still fails (a max statistic cannot be
    diluted). No single seed is pinned for this case in the design text
    (KEY_ANCHORING_ATTACK_R3.md Target 1's own provenance-gap note); this
    build registers `seed` as its own reproducible choice."""
    n, d = healthy_table.shape
    assert n_collapsed <= n
    g = torch.Generator().manual_seed(seed)
    direction = F.normalize(torch.randn(d, generator=g), dim=-1)
    idx = torch.randperm(n, generator=g)[:n_collapsed]
    table = healthy_table.clone()
    noise = torch.randn(n_collapsed, d, generator=g) * sigma
    table[idx] = F.normalize(direction.unsqueeze(0) + noise, dim=-1)
    return table


# ---------------------------------------------------------------------------
# sec 2.2's Rev-4 masked gather/scatter anchor blend -- THE function bind()
# calls at its geo3 insertion site (model_rd.py imports this directly).
# ---------------------------------------------------------------------------

def anchor_blend_gather_scatter(k_eff_raw: torch.Tensor, anchor_weight: torch.Tensor,
                                  anchor_trained_mask: torch.Tensor, key_ids: torch.Tensor,
                                  lam) -> torch.Tensor:
    """sec 2.2's Rev-4 masked gather/scatter blend -- closes R3 finding 4's
    `torch.where` 0xNaN backward-poisoning path (verified real, then
    verified fixed, by an executable contrast run in
    KEY_ANCHORING_ATTACK_R3.md's round-4 verify; this is the SAME
    construction, reproduced again by this build's own smoke 4). Arithmetic
    touches TRAINED-entity rows ONLY; held-out rows are a bit-exact `clone`
    with NO graph edge into them in either direction (forward OR backward)
    -- `k_blend_raw`'s held-out rows are never multiplied, added, or
    re-normalized, so a strict `torch.equal` bypass check (sec 3.3) is
    exactly correct, not a floating-point-lucky approximation.

    k_eff_raw: (B,K,d) this episode's raw gathered (pre-orthogonalization)
      keys -- `_gather_at(k_norm_raw, item_pos)` in model_rd.py's bind().
    anchor_weight: (vocab,d) the anchor nn.Embedding's `.weight`.
    anchor_trained_mask: (vocab,) bool buffer, True at trained-entity rows
      (`pools.train_name_ids`) -- the M1/C17 bypass (sec 3.3).
    key_ids: (B,K) int64 real entity token ids (`batch["key_ids"]`).
    lam: scalar tensor or python float in [0,1], the CURRENT lambda value
      (sigmoid(raw_param) for learned mode, or the fixed-grid value).
    Returns k_blend_raw (B,K,d) -- feed DIRECTLY into
    geo3_orthogonalize_logged, an UNCHANGED call (sec 2.2)."""
    trained_here = anchor_trained_mask[key_ids]                # (B,K) bool
    t_idx = trained_here.nonzero(as_tuple=True)                 # (row_idx, col_idx)
    sub_blend = F.normalize(
        (1.0 - lam) * k_eff_raw[t_idx] + lam * anchor_weight[key_ids[t_idx]], dim=-1)
    k_blend_raw = k_eff_raw.clone()
    k_blend_raw[t_idx] = sub_blend
    return k_blend_raw


# ---------------------------------------------------------------------------
# sec 3.2's lambda-trajectory band classification + sec 3.7's engaged_frac
# band classification. Pure arithmetic on already-computed scalars -- no
# torch dependency needed, kept here so run_deltanet_rd.py / the sweep
# harness / the smoke suite share ONE classifier.
# ---------------------------------------------------------------------------

def assert_lambda_log_cadence(log_every: int) -> None:
    """sec 3.2/R3 finding 1's startup assertion: a run with learned-lambda
    trajectory logging active must use EXACTLY the registered cadence
    (LAMBDA_LOG_CADENCE_STEPS=200) -- a cadence change fires this assertion
    LOUDLY instead of silently resizing the trailing-window statistic.
    Called once at train() startup whenever anchor_active and
    anchor_lambda_mode=='learned' (run_deltanet_rd.py)."""
    assert log_every == LAMBDA_LOG_CADENCE_STEPS, (
        f"lambda-trajectory logging requires log_every == LAMBDA_LOG_CADENCE_STEPS "
        f"({LAMBDA_LOG_CADENCE_STEPS}), got log_every={log_every} -- sec 3.2/R3 finding 1: a "
        f"cadence change would silently resize the 5-logged-point trailing window this design's "
        f"lambda-band classification depends on. Fix --log-every, or (if the cadence is "
        f"deliberately changing) re-register LAMBDA_LOG_CADENCE_STEPS/LAMBDA_WINDOW_LOG_POINTS "
        f"as a documented deviation, not a silent default change.")


def _lambda_point_band(x: float) -> str:
    if 0.2 <= x <= 0.8:
        return "interior"
    if x > 0.95:
        return "pin_rediscovery"
    if x < 0.05:
        return "not_recruited"
    return "ambiguous"


def classify_lambda_band(final_value: float, trailing_mean: float, trailing_range: float) -> str:
    """sec 3.2's three-part per-seed band rule (Rev 3 -- R2 MAJOR 5;
    window pinned in LOG-POINT space at Rev 4 -- R3 finding 1): ALL THREE
    of (i) the final logged value, (ii) the trailing mean (over the last
    LAMBDA_WINDOW_LOG_POINTS logged points), in the SAME band, AND (iii)
    the trailing range (max-min over that same window) < 0.1 -- any
    failure lands in 'ambiguous' regardless of the mean (a pre-registered
    exclusion, not a post-hoc judgment call)."""
    if trailing_range >= 0.1:
        return "ambiguous"
    b_final, b_mean = _lambda_point_band(final_value), _lambda_point_band(trailing_mean)
    if b_final != b_mean:
        return "ambiguous"
    return b_final


def arm_lambda_label(per_seed_bands: list[str]) -> str:
    """Arm-level label (sec 3.2): >=2/3 seeds in a band assigns that band;
    seeds spread across non-adjacent bands (no band reaching >=2/3) ->
    'ambiguous', no headline."""
    from collections import Counter
    counts = Counter(per_seed_bands)
    n = len(per_seed_bands)
    for band, ct in counts.items():
        if band != "ambiguous" and ct / n >= 2.0 / 3.0 - 1e-9:
            return band
    return "ambiguous"


def engaged_frac_band(engaged_frac: float) -> str:
    """sec 3.7's per-entity anchor-input-alignment engagement bands:
    >=90% headline-eligible; [50%,90%) partial anchoring (Outcome A'');
    <50% not recruited regardless of aggregate drift (routes to Outcome C)."""
    if engaged_frac >= 0.9:
        return "headline_eligible"
    if engaged_frac >= 0.5:
        return "partial_anchoring"
    return "not_recruited"


def lambda_window_summary(lambda_trajectory: list[dict], window_points: int = LAMBDA_WINDOW_LOG_POINTS) -> dict:
    """Computes the sec 3.2 per-seed summary fields from a run's own logged
    lambda trajectory (a list of {"step":.., "lambda":..} dicts, one per
    logged point at LAMBDA_LOG_CADENCE_STEPS cadence). Returns final value,
    trailing-window mean/range, and the classified band -- the exact
    machine-readable fields sec 2.2's logging bullet registers."""
    assert len(lambda_trajectory) >= 1, "empty lambda trajectory"
    vals = [pt["lambda"] for pt in lambda_trajectory]
    window = vals[-window_points:]
    final_value = vals[-1]
    trailing_mean = sum(window) / len(window)
    trailing_range = max(window) - min(window)
    band = classify_lambda_band(final_value, trailing_mean, trailing_range)
    return {"final_value": final_value, "trailing_mean": trailing_mean,
            "trailing_range": trailing_range, "n_window_points": len(window), "band": band}


# ---------------------------------------------------------------------------
# sec 3.6's mechanical BANDS_PINNED blinding gate -- writer, launcher-gate
# validator, and readout-timestamp assertion. Shared by
# run_deltanet_rd_exactness_sweep.py (writer + launcher gate) and any
# readout/analysis script (timestamp assertion).
# ---------------------------------------------------------------------------

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def reference_arm_result_valid(result: dict, expected_steps: int) -> bool:
    """A reference-arm result JSON 'validates as complete' (sec 3.6 writer
    requirement 1) iff: complete==true, steps_completed==expected_steps,
    AND its final checkpoint carries the drift fields (both post-NS and
    pre-NS, sec 3.6's 'instrument symmetry' requirement)."""
    if result.get("complete") is not True:
        return False
    if result.get("steps_completed", 0) < expected_steps:
        return False
    checkpoints = result.get("checkpoints") or []
    if not checkpoints:
        return False
    final = checkpoints[-1]
    drift = final.get("drift_probe")
    if not isinstance(drift, dict):
        return False
    return ("post_ns" in drift and "mean" in drift["post_ns"]
            and "pre_ns" in drift and "mean" in drift["pre_ns"])


def derive_engaged_bands(per_k_final_drift: dict, ceiling_by_k: dict) -> dict:
    """sec 3.6's band derivation (n=3 per K, Rev 4 -- R3 finding 3):
    engaged_K = mean_ref + 2*s_ref (sample std, n=3, df=2) over the 3
    reference-arm FINAL-CHECKPOINT post-NS drift values at that K.
    Degenerate-case guard: engaged_K >= ceiling_K - 0.005 -> UNRESOLVABLE
    at that K, with a mandatory leave-one-out sensitivity report
    (disclosure-only -- no re-decision rule hangs off it, per Rev 4's own
    scoping).

    per_k_final_drift: {K: [v0, v1, v2]} (3 reference seeds' own final
    post-NS drift means). ceiling_by_k: {K: computed_ceiling} (sec 2.2:
    0.9423 @ K=32, 0.9745 @ K=16)."""
    out = {}
    for K, vals in per_k_final_drift.items():
        assert len(vals) == 3, f"K={K}: band derivation requires exactly 3 reference seeds, got {len(vals)}"
        t = torch.tensor(vals, dtype=torch.float64)
        mean_ref = t.mean().item()
        s_ref = t.std(unbiased=True).item()          # sample std, n=3, df=2
        engaged_k = mean_ref + 2.0 * s_ref
        ceiling = ceiling_by_k[K]
        unresolvable = engaged_k >= ceiling - 0.005
        loo = []
        for drop in range(3):
            keep = [vals[i] for i in range(3) if i != drop]
            kt = torch.tensor(keep, dtype=torch.float64)
            loo.append(kt.mean().item() + 2.0 * kt.std(unbiased=True).item())
        out[K] = {
            "mean_ref": mean_ref, "s_ref": s_ref, "engaged_k": engaged_k,
            "ceiling": ceiling, "unresolvable": bool(unresolvable),
            "per_seed": vals, "leave_one_out_engaged_k": loo,
        }
    return out


def write_bands_pinned(path: str, per_k_final_drift: dict, ceiling_by_k: dict,
                        reference_result_paths: dict) -> dict:
    """sec 3.6 writer requirement (a): writes BANDS_PINNED.json ONLY after
    every reference arm's result JSON validates complete (the CALLER is
    responsible for having already checked reference_arm_result_valid on
    every one of the 6 files -- this function's OWN job is to compute the
    bands and stamp the sha256 hashes + timestamp, matching the design's
    'derived bands + per-seed inputs + formula version + sha256 hashes of
    all 6 reference-arm result JSONs + timestamp' content spec).

    reference_result_paths: {K: [path0, path1, path2]} -- the 6 reference
    result JSON paths, hashed here (never re-derived silently later)."""
    bands = derive_engaged_bands(per_k_final_drift, ceiling_by_k)
    hashes = {str(K): [sha256_of_file(p) for p in paths] for K, paths in reference_result_paths.items()}
    doc = {
        "formula_version": "sec 3.6 Rev4 (n=3, engaged_K = mean_ref + 2*s_ref, df=2)",
        "bands": {str(K): v for K, v in bands.items()},
        "reference_result_paths": {str(K): paths for K, paths in reference_result_paths.items()},
        "reference_result_sha256": hashes,
        "pinned_at": time.time(),
        "pinned_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return doc


def validate_bands_pinned(path: str) -> dict | None:
    """sec 3.6 launcher-gate requirement (b): anchor cells REFUSE to launch
    unless this file exists AND re-hashing every referenced reference-arm
    JSON matches the recorded hash (a changed reference JSON post-pin is a
    pin-integrity error, never silently re-derived). Returns the parsed
    doc on success, None on any validation failure (missing file, missing
    field, or a hash mismatch) -- callers are responsible for the loud
    refusal message; this function never raises for an ordinary "not
    ready yet" state, only for a genuinely malformed file (a bug, not a
    sequencing issue)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        doc = json.load(f)
    for K, paths in doc.get("reference_result_paths", {}).items():
        recorded_hashes = doc["reference_result_sha256"][K]
        for p, expected in zip(paths, recorded_hashes):
            if not os.path.exists(p):
                return None
            if sha256_of_file(p) != expected:
                return None
    return doc


def assert_blind_not_broken(bands_doc: dict, anchor_started_ats: list) -> None:
    """sec 3.6 readout-assertion requirement (c): the pin timestamp must
    STRICTLY PRECEDE the earliest anchor-arm start time. Raises
    AssertionError (the readout aborts, the wave summary marks the blind
    broken, every affected anchor readout demotes to descriptive tier --
    sec 3.6) if violated."""
    assert anchor_started_ats, "no anchor-arm start times given -- nothing to check the blind against"
    earliest_anchor_start = min(anchor_started_ats)
    pinned_at = bands_doc["pinned_at"]
    assert pinned_at < earliest_anchor_start, (
        f"BLIND BROKEN: BANDS_PINNED.json pinned_at={pinned_at} does NOT strictly precede the "
        f"earliest anchor-arm start_at={earliest_anchor_start} -- sec 3.6's mechanical readout "
        f"assertion. Every affected anchor readout must report at descriptive tier only.")


# ---------------------------------------------------------------------------
# sec 3.6 Rev 5's --unblind-override demotion stamping -- writes into EVERY
# affected anchor-arm's OWN result JSON at assembly time (the R4-1 fix:
# never the wave summary alone), mirroring lm_pretrain_rd.py's own
# claim_tier / _assemble_result precedent (L1090, smoke-asserted L1420).
# ---------------------------------------------------------------------------

def override_stamp_payload(timestamp: float | None = None) -> dict:
    """What the launcher threads down to a spawned anchor run when
    --unblind-override is passed (sec 3.6 Rev 5). `timestamp` defaults to
    now() -- passed explicitly in tests for determinism."""
    ts = timestamp if timestamp is not None else time.time()
    return {"unblind_override": True, "unblind_override_at": ts}


def assemble_claim_tier_fields(stamp: dict | None) -> dict:
    """sec 3.6 Rev 5: fields written into EVERY anchor-arm result JSON at
    assembly time (run_deltanet_rd.py's _assemble_result), regardless of
    the override path. `claim_tier` is written ONLY on the override path
    (the demotion is the one tier verdict knowable at launch time,
    independent of anything the run later earns); `unblind_override` is
    ALWAYS present (True or False) so its absence can never be read as
    evidence of a clean blind -- presence is mandatory either way."""
    if stamp is not None and stamp.get("unblind_override"):
        return {"claim_tier": "descriptive", "unblind_override": True,
                "unblind_override_at": stamp["unblind_override_at"]}
    return {"unblind_override": False}


# ---------------------------------------------------------------------------
# sec 14.6's pinned per-entity drift measurement -- moved here (out of
# geo3_drift_diagnostic.py) so BOTH geo3_drift_diagnostic.py (reference
# arms / bare geo3) and run_deltanet_rd.py's own optional per-checkpoint
# reference-arm drift probe (sec 3.6: "PLUS the per-checkpoint drift
# diagnostic active") can import the SAME functions without creating a
# circular import (geo3_drift_diagnostic.py already does
# `from run_deltanet_rd import train as train_geo3` at module scope; having
# run_deltanet_rd.py import THAT module back would be a real cycle -- this
# fla-free shared module is the fix, not a workaround).
# ---------------------------------------------------------------------------

def sample_batch_fixed_entity(cfg, batch_size, fixed_entity_id, gen, pools, device):
    """Verbatim (moved, not rewritten) from geo3_drift_diagnostic.py's own
    function of the same name -- sec 14.6's 'fixing one entity's raw key
    and resampling its K-1 episode-mates' construction. Entity slot 0 is
    PINNED to fixed_entity_id in every row; the other K-1 slots are
    freshly, independently drawn without replacement from the train pool
    minus the fixed entity."""
    import grammar_rd as grd
    B, K = batch_size, cfg.K
    buf_len, clause_len, T_bind = cfg.buf_len, cfg.clause_len, cfg.T_bind
    other_pool = pools.train_name_ids[pools.train_name_ids != fixed_entity_id].to(device)
    assert other_pool.numel() >= K - 1, \
        f"train pool minus the fixed entity ({other_pool.numel()}) < K-1={K - 1}"
    scores = torch.rand(B, other_pool.numel(), generator=gen, device=device)
    pool_idx = scores.argsort(dim=-1)[:, :K - 1]
    other_entities = other_pool[pool_idx]
    fixed_col = torch.full((B, 1), int(fixed_entity_id), dtype=torch.int64, device=device)
    entity_ids = torch.cat([fixed_col, other_entities], dim=1)

    succ = grd._permutation_graph(B, K, gen, device, dtype=torch.float32)
    key_ids = entity_ids
    value_ids = torch.gather(entity_ids, 1, succ)

    rel_pool = pools.rel_a_ids.to(device)
    rel_idx = torch.randint(0, rel_pool.numel(), (B,), generator=gen, device=device)
    rel_id = rel_pool[rel_idx]

    token_ids = torch.full((B, T_bind), int(pools.buffer_id), dtype=torch.int64, device=device)
    item_pos = (torch.arange(K, device=device) * clause_len + buf_len + 2).unsqueeze(0).expand(B, K).contiguous()
    key_pos = item_pos - 2
    rel_pos = item_pos - 1
    period_pos = item_pos + 1
    token_ids.scatter_(1, key_pos, key_ids)
    token_ids.scatter_(1, rel_pos, rel_id.unsqueeze(1).expand(B, K))
    token_ids.scatter_(1, item_pos, value_ids)
    token_ids.scatter_(1, period_pos,
                        torch.full((B, K), int(pools.period_id), dtype=torch.int64, device=device))
    beta_mask = torch.zeros(B, T_bind, device=device, dtype=torch.float32)
    beta_mask.scatter_(1, item_pos, torch.ones(B, K, device=device))
    return {"token_ids": token_ids, "beta_mask": beta_mask, "item_pos": item_pos, "key_ids": key_ids}


def pairwise_drift_stats(out_rows: torch.Tensor) -> tuple[float, float]:
    """(mean, p10) of all pairwise cosines among one entity's rows across
    context resamples. out_rows: (n_draws, d)."""
    n = out_rows.shape[0]
    pw = F.cosine_similarity(out_rows.unsqueeze(0), out_rows.unsqueeze(1), dim=-1)
    off = pw[~torch.eye(n, dtype=torch.bool, device=out_rows.device)]
    return off.mean().item(), torch.quantile(off, 0.10).item()


@torch.no_grad()
def measure_entity_rows(model, cfg, pools, fixed_entity_id, n_resamples, gen, device,
                          pre_ns_attr: str | None = None):
    """One entity's n_resamples (post-NS) orthogonalized-key rows PLUS
    (sec 3.1's item-5 instrument symmetry) its pre-NS rows read from the
    model's own side channel, if `pre_ns_attr` names one present on the
    model (`geo3_last_k_eff_raw` for bare geo3 / candidate (c);
    `anchor_last_k_blend_raw` for candidate (d)). Returns
    (post_ns_rows, pre_ns_rows_or_None), each (n_resamples, d_state)."""
    model.eval()
    b = sample_batch_fixed_entity(cfg, n_resamples, fixed_entity_id, gen, pools, device)
    _, k_eff_items, _ = model.bind(b)
    post_ns = k_eff_items[:, 0, :].detach()
    pre_ns = None
    if pre_ns_attr is not None:
        raw = getattr(model, pre_ns_attr, None)
        if raw is not None:
            pre_ns = raw[:, 0, :].detach()
    return post_ns, pre_ns


def measure_drift(model, cfg, pools, n_entities, n_resamples, gen, device,
                    pre_ns_attr: str | None = None):
    """sec 14.6's full pinned statistic (post-NS), PLUS (when pre_ns_attr
    is given) the same pooled mean/p10 statistic on the pre-NS side
    channel -- sec 3.1's item-5 evidentiary quantity for anchor arms."""
    entity_pool = pools.train_name_ids.to(device)
    assert entity_pool.numel() >= n_entities
    perm = torch.randperm(entity_pool.numel(), generator=gen, device=device)[:n_entities]
    entities = entity_pool[perm]
    per_entity_stats = []
    all_off_diag_post, all_off_diag_pre = [], []
    for eid in entities.tolist():
        post_rows, pre_rows = measure_entity_rows(model, cfg, pools, eid, n_resamples, gen, device,
                                                    pre_ns_attr=pre_ns_attr)
        mean_c, p10_c = pairwise_drift_stats(post_rows)
        entry = {"entity_id": eid, "post_ns_mean": mean_c, "post_ns_p10": p10_c}
        n = post_rows.shape[0]
        pw = F.cosine_similarity(post_rows.unsqueeze(0), post_rows.unsqueeze(1), dim=-1)
        all_off_diag_post.append(pw[~torch.eye(n, dtype=torch.bool, device=post_rows.device)])
        if pre_rows is not None:
            mean_pre, p10_pre = pairwise_drift_stats(pre_rows)
            entry["pre_ns_mean"] = mean_pre
            entry["pre_ns_p10"] = p10_pre
            pw_pre = F.cosine_similarity(pre_rows.unsqueeze(0), pre_rows.unsqueeze(1), dim=-1)
            all_off_diag_pre.append(pw_pre[~torch.eye(n, dtype=torch.bool, device=pre_rows.device)])
        per_entity_stats.append(entry)
    pooled_post = torch.cat(all_off_diag_post)
    out = {
        "post_ns": {"mean": pooled_post.mean().item(), "p10": torch.quantile(pooled_post, 0.10).item()},
        "n_entities": n_entities, "n_resamples": n_resamples,
        "n_pooled_pairs": pooled_post.numel(), "per_entity": per_entity_stats,
    }
    if all_off_diag_pre:
        pooled_pre = torch.cat(all_off_diag_pre)
        out["pre_ns"] = {"mean": pooled_pre.mean().item(), "p10": torch.quantile(pooled_pre, 0.10).item()}
    return out


# ---------------------------------------------------------------------------
# sec 3.7's full-pool per-entity anchor-INPUT-alignment sweep + h=1
# behavioral companion. MOVED HERE (2026-07-06, keyanchor-confirm build) from
# keyanchor_drift_diagnostic.py, where they were previously the ONLY place
# either computation was wired into ANY running code (KEY_ANCHORING_DESIGN.md
# sec 9.3's documented gap: the mandatory Wave-1 training cells never called
# either function, since keyanchor_wave1_manifest() never threaded
# drift_probe=True into their specs). Moving them here lets BOTH
# keyanchor_drift_diagnostic.py's standalone probe driver AND
# run_deltanet_rd.py's own --drift-probe checkpoint block (train(), the
# actual 20,000-step admitted arms) call the SAME function -- one
# definition, never a hand-copied twin (this project's standing
# no-silent-duplication discipline, the same reasoning key_anchoring.py's own
# docstring gives for sample_batch_fixed_entity above).
#
# Both functions need model_rd.DeltaNetRDBlock / grammar_rd's batch sampler --
# imported LOCALLY inside each function body (never at this module's top
# level), so key_anchoring.py's own module-level import stays fla-free
# (model_rd.py imports fla at module scope; importing it only INSIDE these
# two functions means merely importing key_anchoring.py never requires fla --
# only CALLING these two functions does, exactly the same constraint
# sample_batch_fixed_entity's own local `import grammar_rd as grd` respects).
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_full_pool_alignment(model, cfg, pools, n_resamples, gen, device):
    """sec 3.7's per-entity anchor-INPUT-alignment sweep, full pool (ALL
    train entities, not a random n_entities sample): a_e = mean over
    >= n_resamples independent episode resamples of cos(pre-NS blended key
    of entity e, A[e]). Uses the SAME sample_batch_fixed_entity construction
    as measure_drift (sec 14.6's sampling machinery) -- eval-only, no
    training-path change. `engaged_frac` = fraction of entities with
    a_e >= 0.9 (sec 3.7's registered band driver)."""
    assert model.anchor_active, "measure_full_pool_alignment requires anchor_active=True"
    model.eval()
    entity_ids = pools.train_name_ids.tolist()
    a_e = {}
    for eid in entity_ids:
        b = sample_batch_fixed_entity(cfg, n_resamples, eid, gen, pools, device)
        _, k_eff_items, _ = model.bind(b)
        blended = model.anchor_last_k_blend_raw[:, 0, :]              # (n_resamples, d_state)
        anchor_row = model.anchor_table.weight[eid].unsqueeze(0).expand_as(blended)
        cos = F.cosine_similarity(blended, anchor_row, dim=-1)
        a_e[eid] = cos.mean().item()
    engaged_frac = sum(1 for v in a_e.values() if v >= 0.9) / len(a_e)
    return {"a_e": a_e, "engaged_frac": engaged_frac, "n_resamples": n_resamples}


@torch.no_grad()
def measure_h1_behavioral_companion(model, cfg, pools, gen, device, n_batches=4, batch_size=64):
    """sec 3.7 Rev4's non-load-bearing behavioral companion: per-entity h=1
    recovery, restricted to eval episodes CONTAINING that entity --
    bookkeeping on the existing eval (tagging each per-item cosine with its
    row's own key_ids before pooling), NO new forward passes beyond a
    standard h=1 eval sweep this diagnostic already needs to run once."""
    import grammar_rd as grd_local

    model.eval()
    per_entity_cos: dict[int, list[float]] = {}
    for _ in range(n_batches):
        b = grd_local.sample_batch_rd(cfg, batch_size, gen, hop_set=(1,), pools=pools, device=device)
        pred, targets, S_T, k_eff_items, v_eff_items = model(b, force_rank_k=None)
        cos = F.cosine_similarity(pred, targets, dim=-1)         # (B,Q)
        key_ids = b["key_ids"]                                    # (B,K) -- entities present THIS episode
        for row in range(cos.shape[0]):
            row_entities = set(key_ids[row].tolist())
            row_cos = cos[row].mean().item()                      # this row's own h=1 recovery
            for eid in row_entities:
                per_entity_cos.setdefault(eid, []).append(row_cos)
    return {eid: sum(vals) / len(vals) for eid, vals in per_entity_cos.items()}
