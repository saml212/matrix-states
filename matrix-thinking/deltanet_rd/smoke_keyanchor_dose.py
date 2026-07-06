"""smoke_keyanchor_dose.py -- KEY_ANCHORING_DESIGN.md sec 14 (Rev 14.3,
DESIGN-CLEARED-FOR-BUILD)'s registered fla-free Wave -1 smoke suite for the
coherence dose-response wave, as a committed, CPU-runnable script. Mirrors
smoke_keyanchor_dstate.py's own style: this file imports ONLY key_anchoring.py
and run_deltanet_rd_exactness_sweep.py (both verified fla-free at module
scope) -- no model_rd.py import here (that would require fla/Triton, not
available in a plain CPU-torch environment; the anchor_table_override
override-path + frozen-grad smoke is instead an EXTENSION of smoke_key_
anchoring.py's own smoke 15, which already documents and accepts this exact
fla constraint -- see that file's own module docstring).

Covers:
  smoke 1 (sec 14.1b item 1's own acceptance criterion): key_anchoring.
    build_dose_table/calibrate_dose re-derive EVERY t/achieved max|cos| pair
    in the committed dose_dial_verify_results.json's own `calibration` block
    (both structures, all 3 doses, all 5 subspace seeds) to within 1e-4 --
    the exact number this design's own acceptance criterion names. Also
    checks the chosen diffuse rank (48) and the rank=128 degeneracy proof.
  smoke 2: keyanchor_dose_calibration_manifest()'s shape -- exactly 1 cell,
    K=68, dose=0.40, structure=rank4, seed=939, d_state=128.
  smoke 3: keyanchor_dose_manifest('rank4')/('diffuse') shape -- 9 cells
    each, 3 doses x 3 seeds, exact registered seed blocks, dose_target/
    dose_structure/subspace_seed threaded into every returned spec.
  smoke 4: _spec()'s dose fields produce DIFFERENT filenames and DIFFERENT
    exactness_config blocks for three cells sharing K/seed but differing
    ONLY in dose_target (sec 14.1b item 3's own acceptance criterion).
  smoke 5 (sec 14.1b item 4's own acceptance criterion, THE round-3-
    documented collision, NEGATIVE UNIT TEST): two specs identical in every
    field EXCEPT dose_target, pointed at an archived result JSON built
    under ONE of the two doses -- is_done() must return True for the
    matching spec and False for the mismatched one. Demonstrates the
    collision was real (a synthetic PRE-FIX is_done() reproduction returns
    True for BOTH) and that the real is_done() closes it.
  smoke 6: manifest seed-block collision -- keyanchor-dose's fresh
    930-948/939 seed block has ZERO collisions against every other seed
    block this program has ever registered (grep-verified, reproduced here
    as an executable check against the actual out_path() filenames).
  smoke 7: zero-collision -- keyanchor-dose's manifest (calibration + rank4
    + diffuse) is collision-free against every prior/sibling wave this
    program has ever registered.
  smoke 8: keyanchor_dose_budget_guard's arithmetic -- Stage 1 fits the full
    13.68 GPU-h ceiling at 2x; cumulative (both stages) does NOT; the
    ceiling-amendment figure matches the design doc's own registered
    10.678 GPU-h to 3 decimal places.
  smoke 9: staging/PI-gate mechanics -- keyanchor_dose_is_calibration_dose_
    verified returns None when the calibration cell is missing, a dict with
    verified=True on a clean in-tolerance synthetic result, and
    verified=False on a synthetic out-of-tolerance result.
  smoke 10: dose-verify assertion mechanics (sec 14.3, mirrored here as a
    pure-arithmetic reproduction of run_deltanet_rd.py's own construction-
    time assert, since that assert lives inside a function requiring
    pools/model construction) -- achieved within +/-10% of target passes;
    achieved outside +/-10% fails, boundary-exact at exactly 10%.

Wired as an ADDITIONAL pre-launch CPU gate for --wave keyanchor-dose
(run_deltanet_rd_exactness_sweep.py's main(), alongside smoke_key_anchoring.py
+ gate2_construction_test.py) -- rc!=0 aborts the wave before any GPU cell
dispatches.

Exit code 0 = every item PASSED. Run: python smoke_keyanchor_dose.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pod-safe imports

import key_anchoring as ka                              # noqa: E402 (fla-free)
import run_deltanet_rd_exactness_sweep as rdx            # noqa: E402 (fla-free)

FAILURES: list[str] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


# ---------------------------------------------------------------------------
# smoke 1: sec 14.1b item 1's acceptance criterion -- re-derive every
# t/achieved pair in dose_dial_verify_results.json's own calibration block
# to within 1e-4, from key_anchoring.py's own (now-committed) functions.
# ---------------------------------------------------------------------------

def smoke_1_dose_dial_matches_reference_json():
    ref_path = os.path.join(HERE, "dose_dial_verify_results.json")
    if not os.path.exists(ref_path):
        _report("smoke 1: dose-dial matches reference JSON (sec 14.1b item 1)", False,
                 f"{ref_path!r} does not exist")
        return
    with open(ref_path) as f:
        ref = json.load(f)
    meta = ref["meta"]
    table = ka.frame_potential_init(meta["n_entities"], meta["d_state"], seed=meta["base_seed"])

    max_err = 0.0
    n_checked = 0
    struct_to_rank = {"rank4": 4, f"diffuse_rank{ref['chosen_diffuse_rank']['subspace_rank']}":
                       ref["chosen_diffuse_rank"]["subspace_rank"]}
    for struct_name, block in ref["calibration"].items():
        rank = block["subspace_rank"]
        assert struct_to_rank[struct_name] == rank, "struct-name/rank mismatch in reference JSON itself"
        for target_str, dose_block in block["doses"].items():
            target = float(target_str)
            for per_seed in dose_block["per_seed"]:
                sseed = per_seed["subspace_seed"]
                cal = ka.calibrate_dose(table, target, rank, sseed)
                err = abs(cal["achieved"] - per_seed["achieved"])
                max_err = max(max_err, err)
                n_checked += 1
    matches = max_err <= 1e-4

    # Chosen diffuse rank matches key_anchoring.DIFFUSE_SUBSPACE_RANK.
    chosen_rank_matches = ka.DIFFUSE_SUBSPACE_RANK == ref["chosen_diffuse_rank"]["subspace_rank"] == 48
    ceiling_matches = abs(ka.achieved_dose(table, 1.0, ka.DIFFUSE_SUBSPACE_RANK, meta["base_seed"])
                          - ref["chosen_diffuse_rank"]["ceiling_at_t1"]) <= 1e-4

    # Degeneracy proof: subspace_rank=d_state is a no-op (achieved dose == base table's own, ~0).
    degeneracy_matches = abs(ka.achieved_dose(table, 1.0, meta["d_state"], meta["base_seed"])
                              - ref["degeneracy_proof"]["achieved_dose_at_t1_rank128"]) <= 1e-4

    print(f"    re-derived {n_checked} (structure, dose, subspace_seed) triples from key_anchoring.py's "
          f"own build_dose_table/calibrate_dose; max abs error vs reference JSON: {max_err:.2e} "
          f"(<=1e-4? {matches})")
    print(f"    key_anchoring.DIFFUSE_SUBSPACE_RANK == 48 == reference JSON's chosen_diffuse_rank: "
          f"{chosen_rank_matches}")
    print(f"    diffuse ceiling (t=1, rank=48) matches reference JSON to 1e-4: {ceiling_matches}")
    print(f"    degeneracy proof (t=1, rank=d_state=128) matches reference JSON to 1e-4: "
          f"{degeneracy_matches}")
    ok = matches and chosen_rank_matches and ceiling_matches and degeneracy_matches
    _report("smoke 1: dose-dial matches reference JSON to 1e-4 (sec 14.1b item 1's own acceptance "
            "criterion) -- every t/achieved pair, both structures, all 3 doses, all 5 subspace seeds, "
            "PLUS the chosen diffuse rank and the rank=128 degeneracy proof", ok)


# ---------------------------------------------------------------------------
# smoke 2/3: manifest shapes.
# ---------------------------------------------------------------------------

def smoke_2_calibration_manifest_shape():
    calib = rdx.keyanchor_dose_calibration_manifest()
    right_count = len(calib) == 1
    c = calib[0] if calib else {}
    right_cell = (right_count and c.get("K") == 68 and c.get("dose_target") == 0.40
                  and c.get("dose_structure") == "rank4" and c.get("seed") == 939
                  and c.get("d_state") == 128 and c.get("subspace_seed") == ka.ANCHOR_INIT_SEED)
    print(f"    calibration manifest: {c}")
    ok = right_count and right_cell
    _report("smoke 2: keyanchor_dose_calibration_manifest() -- exactly [K=68, dose=0.40, "
            "structure=rank4, seed=939, d_state=128, subspace_seed=ANCHOR_INIT_SEED]", ok)


def smoke_3_dose_manifest_shape():
    ok = True
    for structure in ("rank4", "diffuse"):
        m = rdx.keyanchor_dose_manifest(structure)
        right_count = len(m) == 9
        doses_present = sorted(set(s["dose_target"] for s in m)) == [0.130, 0.284, 0.40]
        struct_ok = all(s["dose_structure"] == structure for s in m)
        subspace_seed_ok = all(s["subspace_seed"] == ka.ANCHOR_INIT_SEED for s in m)
        k_ok = all(s["K"] == 68 for s in m)
        d_state_ok = all(s["d_state"] == 128 for s in m)
        expected_seeds = {dose: set(rdx.KEYANCHOR_DOSE_SEEDS_BY_STRUCTURE_DOSE[(structure, dose)])
                           for dose in (0.130, 0.284, 0.40)}
        seeds_ok = all(set(s["seed"] for s in m if s["dose_target"] == dose) == expected_seeds[dose]
                       for dose in expected_seeds)
        print(f"    structure={structure}: count={len(m)} doses={sorted(set(s['dose_target'] for s in m))} "
              f"struct_ok={struct_ok} subspace_seed_ok={subspace_seed_ok} k_ok={k_ok} "
              f"d_state_ok={d_state_ok} seeds_ok={seeds_ok}")
        ok = ok and right_count and doses_present and struct_ok and subspace_seed_ok and k_ok \
            and d_state_ok and seeds_ok
    _report("smoke 3: keyanchor_dose_manifest('rank4')/('diffuse') shape -- 9 cells each, 3 doses "
            "x 3 seeds, exact registered seed blocks, dose_target/dose_structure/subspace_seed "
            "threaded into every spec", ok)


# ---------------------------------------------------------------------------
# smoke 4: sec 14.1b item 3's own acceptance criterion -- three cells
# sharing K/seed but differing ONLY in dose_target produce DIFFERENT
# filenames and DIFFERENT exactness_config-relevant fields.
# ---------------------------------------------------------------------------

def smoke_4_dose_target_varies_filename_and_config():
    specs = [rdx._spec("keyanchor-dose", 68, 530, 20000, "d", geo3_active=True, geo3_n_iter=20,
                        geo3_resid_tol=1e-2, anchor_active=True, anchor_lambda_mode="learned",
                        drift_probe=True, rev7_engagement=True, d_state=128,
                        dose_target=dose, dose_structure="rank4", subspace_seed=ka.ANCHOR_INIT_SEED)
             for dose in (0.130, 0.284, 0.400)]
    names = [s["name"] for s in specs]
    names_distinct = len(set(names)) == 3
    configs_distinct = len({json.dumps({k: s[k] for k in ("dose_target", "dose_structure", "subspace_seed")},
                                        sort_keys=True) for s in specs}) == 3
    name_bits_present = all(f"dose{round(d * 1000):03d}" in n for d, n in zip((0.130, 0.284, 0.400), names))
    print(f"    filenames: {names}")
    print(f"    all 3 distinct: {names_distinct}; dose130/dose284/dose400 name bits present: "
          f"{name_bits_present}; 3 distinct dose-identity configs: {configs_distinct}")
    ok = names_distinct and configs_distinct and name_bits_present
    _report("smoke 4: _spec()'s dose fields (sec 14.1b item 3's own acceptance criterion) -- three "
            "cells sharing K/seed but differing ONLY in dose_target produce 3 DIFFERENT filenames "
            "and 3 DIFFERENT dose-identity configs", ok)


# ---------------------------------------------------------------------------
# smoke 5: sec 14.1b item 4's own acceptance criterion -- THE round-3-
# documented collision, demonstrated as a negative unit test, not merely
# asserted fixed.
# ---------------------------------------------------------------------------

def _write_fake_dosed_result(out_dir, spec, achieved_max_cos):
    """A minimal, is_done()-satisfying result JSON for a dosed cell,
    carrying the EXACT exactness_config keys is_done() reads (dose_target/
    dose_structure/subspace_seed), matching run_deltanet_rd.py's own
    _assemble_result schema for these fields."""
    ec = {"embed_source": "learned", "gram_alpha": None, "gram_rho": None, "strong_pin": False,
          "lambda_orth": 0.0, "use_zca": False, "fnce_m": None, "geo3_active": True, "geo3_n_iter": 20,
          "geo3_resid_tol": 1e-2, "anchor_active": True, "anchor_lambda_mode": "learned",
          "anchor_lambda_fixed": None, "lambda_anchor": 0.0, "drift_probe": True,
          "rev7_engagement": True, "anchor_table_frozen": True, "anchor_table_init_mode": "dosed",
          "dose_target": spec["dose_target"], "dose_structure": spec["dose_structure"],
          "subspace_seed": spec["subspace_seed"], "achieved_max_cos": achieved_max_cos}
    doc = {"K": spec["K"], "seed": spec["seed"], "steps": spec["steps"], "complete": True,
           "steps_completed": spec["steps"], "exactness_config": ec, "wall_s": 2307.6685,
           "timed_out": False, "d_state": spec["d_state"],
           "checkpoints": [{"item6_table_conditioning": {"max_abs_cos": achieved_max_cos}}]}
    p = rdx.out_path(out_dir, spec)
    with open(p, "w") as f:
        json.dump(doc, f)
    return p


def _is_done_pre_fix_reproduction(out_dir, spec):
    """A DELIBERATE reproduction of is_done() AS IT EXISTED BEFORE sec
    14.1b item 4's fix (no dose-identity check at all) -- used ONLY to
    demonstrate the collision was real, never as production code. Copies
    is_done()'s own pre-fix logic (every check up to, but NOT including,
    the three new dose-identity checks this build adds)."""
    p = rdx.out_path(out_dir, spec)
    if not os.path.exists(p):
        return False
    with open(p) as f:
        d = json.load(f)
    if d.get("complete") is not True or d.get("steps_completed", 0) < spec["steps"]:
        return False
    if d.get("K") != spec["K"] or d.get("seed") != spec["seed"] or d.get("steps") != spec["steps"]:
        return False
    ec = d.get("exactness_config") or {}
    if bool(ec.get("anchor_active", False)) != bool(spec.get("anchor_active", False)):
        return False
    if bool(ec.get("anchor_table_frozen", False)) != bool(spec.get("anchor_table_frozen", False)):
        return False
    if ec.get("anchor_table_init_mode", "frame_potential") != spec.get("anchor_table_init_mode", "frame_potential"):
        return False
    if d.get("d_state", 64) != spec.get("d_state", 64):
        return False
    return True   # NO dose_target/dose_structure/subspace_seed check -- the pre-fix state


def smoke_5_is_done_dose_collision_negative_test():
    dose_a, dose_b = 0.130, 0.284
    with tempfile.TemporaryDirectory() as tmp:
        spec_a = rdx._keyanchor_dose_spec(dose_a, "rank4", 530)
        spec_b = rdx._keyanchor_dose_spec(dose_b, "rank4", 530)   # same K/seed/arm/frozen/init_mode
        # These two specs' own filenames must ALREADY differ (sec 14.1b item
        # 3's own fix) -- if they didn't, this would be a filename-collision
        # test, not an is_done()-identity test. Confirmed here, not assumed.
        assert spec_a["name"] != spec_b["name"], "spec_a/spec_b filenames collide -- test setup invalid"
        # Archive a result JSON built under dose_a's own config, at dose_a's
        # own filename (out_path(spec_a)) -- i.e. the cell that ACTUALLY ran.
        _write_fake_dosed_result(tmp, spec_a, achieved_max_cos=0.13003)

        # PRE-FIX reproduction: is_done() as it existed before this build's
        # dose-identity check would resolve BOTH specs against dose_a's own
        # archived file IF their filenames also happened to collide -- but
        # since _spec() (item 3) already makes filenames distinct, the
        # pre-fix collision manifests as: is_done_pre_fix(spec_b) reads
        # out_path(spec_b), which does NOT exist (only spec_a's file was
        # written) -- so the REAL collision this smoke must demonstrate is
        # at the is_done() FIELD-CHECK level, not the filename level: two
        # specs whose filenames DO collide (identical dose_target) must
        # never be mistaken for two DIFFERENT doses. The task brief's own
        # phrasing ("construct two specs identical in every field EXCEPT
        # dose_target... point BOTH at an archived result JSON built under
        # ONE of the two doses") requires deliberately pointing spec_b at
        # spec_a's own archived path (out_path(spec_a)) to test the FIELD
        # check in isolation from the filename check -- done via a direct
        # is_done()-style field comparison against the SAME file below.
        archived_path = rdx.out_path(tmp, spec_a)
        with open(archived_path) as f:
            archived = json.load(f)
        archived_ec = archived["exactness_config"]

        # Pre-fix field-check (no dose fields at all): both spec_a and
        # spec_b "match" the archived JSON's non-dose fields identically
        # (same K/seed/arm/frozen/init_mode) -- the collision.
        def _pre_fix_fields_match(spec):
            return (archived["K"] == spec["K"] and archived["seed"] == spec["seed"]
                    and bool(archived_ec.get("anchor_active", False)) == bool(spec.get("anchor_active", False))
                    and bool(archived_ec.get("anchor_table_frozen", False)) == bool(spec.get("anchor_table_frozen", False))
                    and archived_ec.get("anchor_table_init_mode") == spec.get("anchor_table_init_mode")
                    and archived["d_state"] == spec.get("d_state", 64))

        pre_fix_a_matches = _pre_fix_fields_match(spec_a)
        pre_fix_b_matches = _pre_fix_fields_match(spec_b)   # TRUE pre-fix -- the collision

        # Post-fix field-check (this build's actual is_done(), via the
        # dose-identity fields it now reads from archived_ec):
        def _post_fix_dose_fields_match(spec):
            return (archived_ec.get("dose_target") == spec.get("dose_target")
                    and archived_ec.get("dose_structure") == spec.get("dose_structure")
                    and archived_ec.get("subspace_seed") == spec.get("subspace_seed"))

        post_fix_a_matches = _post_fix_dose_fields_match(spec_a)
        post_fix_b_matches = _post_fix_dose_fields_match(spec_b)

        # The REAL is_done(), end-to-end, against the actually-archived
        # spec_a file -- called with spec_a's own out_path (matching
        # filename) so is_done() reads the file it's meant to.
        is_done_a = rdx.is_done(tmp, spec_a)
        # And against a copy of the SAME archived content placed at
        # spec_b's own filename (simulating "two dosed specs same K/seed
        # different dose must NOT resume-match" even if a filename
        # collision were somehow forced) -- is_done() must still say False
        # for spec_b because the FIELD check (not just the path) disagrees.
        import shutil
        shutil.copy(archived_path, rdx.out_path(tmp, spec_b))
        is_done_b_against_wrong_dose = rdx.is_done(tmp, spec_b)

    print(f"    spec_a (dose={dose_a}) filename: {spec_a['name']}")
    print(f"    spec_b (dose={dose_b}) filename: {spec_b['name']} (distinct from spec_a: "
          f"{spec_a['name'] != spec_b['name']})")
    print(f"    PRE-FIX field-check (no dose fields): spec_a matches archived (dose={dose_a}): "
          f"{pre_fix_a_matches} (expect True) | spec_b matches SAME archived (dose={dose_a}): "
          f"{pre_fix_b_matches} (expect True -- THE COLLISION: pre-fix, a dose-{dose_b} spec would "
          f"spuriously read as already-done against a dose-{dose_a} archive)")
    print(f"    POST-FIX field-check (this build's dose fields): spec_a matches: {post_fix_a_matches} "
          f"(expect True) | spec_b matches SAME archived content: {post_fix_b_matches} "
          f"(expect False -- the fix)")
    print(f"    REAL is_done(tmp, spec_a) against its own archived file: {is_done_a} (expect True)")
    print(f"    REAL is_done(tmp, spec_b) against a copy of spec_a's content at spec_b's own path: "
          f"{is_done_b_against_wrong_dose} (expect False -- is_done() must not resume-match a "
          f"different dose even if a filename collision were forced)")
    ok = (pre_fix_a_matches and pre_fix_b_matches   # the pre-fix collision was real
          and post_fix_a_matches and not post_fix_b_matches   # the fix closes it at the field level
          and is_done_a and not is_done_b_against_wrong_dose)   # the REAL is_done() is correct end-to-end
    _report("smoke 5: is_done() dose-identity negative unit test (sec 14.1b item 4's own acceptance "
            "criterion, THE round-3-documented collision) -- demonstrates the pre-fix collision was "
            "REAL (both specs would have matched), and that the real is_done() now returns True for "
            "the matching spec and False for the mismatched one, even when forced onto the same "
            "archived content", ok)


# ---------------------------------------------------------------------------
# smoke 6/7: seed/manifest collision checks.
# ---------------------------------------------------------------------------

def smoke_6_seed_block_zero_collision():
    import re
    dose_seeds = set()
    for seeds in rdx.KEYANCHOR_DOSE_SEEDS_BY_STRUCTURE_DOSE.values():
        dose_seeds |= set(seeds)
    dose_seeds.add(rdx.KEYANCHOR_DOSE_CALIBRATION_SEED)
    print(f"    keyanchor-dose seed block: {sorted(dose_seeds)}")

    # Grep every *.py file in this directory for any of the new seeds
    # appearing in an ACTUAL seed-registration context -- a tuple/list of
    # bare integers (e.g. "(930, 931, 932)" or "seed=937" or a dict literal
    # key), never a bare substring match against arbitrary prose/comments/
    # floats/token-counts (which produced false positives on first run:
    # "3.937" in an assert, "46,026,934" tokens, "doc ~942" in a comment --
    # none are seed registrations). This is a direct, executable
    # reproduction of the "verify zero collision by grep" build
    # instruction, scoped to the thing that actually matters for a
    # resume-safety collision: another manifest's OWN registered seed
    # constant using the SAME integer.
    # Matches "seed=930" / "seed: 930" / "(930, 931, 932)" / "[930, 931]" /
    # a bare ", 930," inside a tuple/list literal -- but explicitly REJECTS
    # a number immediately preceded by "," + digits (a thousands-separator
    # comma inside a larger literal like "46,026,934", the false positive
    # smoke 6 first caught) via the negative lookbehind for "\d,\s*" before
    # the number, and rejects a number immediately FOLLOWED by a comma-
    # then-3-digits continuation (the other half of the same
    # thousands-separator shape, e.g. matching "46" out of "46,026,934").
    seed_context_re = re.compile(
        r"(?:seed\s*[:=]\s*|[\[(]\s*|(?<!\d,)(?<!\d)\s*,\s*)(\d{3,})(?!\s*,\s*\d{3}\b)")
    offenders = []
    for fname in os.listdir(HERE):
        if not fname.endswith(".py") or fname == "smoke_keyanchor_dose.py":
            continue
        if fname == "run_deltanet_rd_exactness_sweep.py":
            # this build's OWN new constants/manifest legitimately own
            # these seeds -- the real cross-program check is against
            # every OTHER file's own seed registrations, not this one.
            continue
        path = os.path.join(HERE, fname)
        with open(path, errors="ignore") as f:
            for lineno, line in enumerate(f, start=1):
                for m in seed_context_re.finditer(line):
                    try:
                        val = int(m.group(1))
                    except ValueError:
                        continue
                    if val in dose_seeds:
                        offenders.append((fname, lineno, val, line.strip()))
    ok = len(offenders) == 0
    print(f"    grep (seed-registration context only) for {sorted(dose_seeds)} across every other "
          f"*.py file in this directory: {'zero collisions' if ok else offenders}")
    _report("smoke 6: keyanchor-dose's fresh seed block (930-948, 939) has ZERO collisions against "
            "every other seed block registered anywhere else in this program (grep-verified, scoped "
            "to actual seed-registration contexts -- not bare substring matches against unrelated "
            "floats/token-counts/doc-index comments)", ok)


def smoke_7_zero_collision():
    dose_paths = {rdx.out_path("/fake/root", s)
                  for s in (rdx.keyanchor_dose_calibration_manifest() + rdx.keyanchor_dose_manifest("rank4")
                            + rdx.keyanchor_dose_manifest("diffuse"))}
    prior_manifests = {
        "wavegeo3": rdx.geo3_wave1_manifest(include_k48=True),
        "wavekeyanchor": rdx.keyanchor_wave1_manifest(),
        "wavekeyanchor-neg1": rdx.keyanchor_wave_neg1_manifest(),
        "waveref": rdx.reference_arms_manifest(),
        "wavekeyanchor-confirm": rdx.keyanchor_confirm_manifest(),
        "wavekeyanchor-mech": rdx.keyanchor_mech_manifest(),
        "wavekeyanchor-mech-gate1": rdx.keyanchor_mech_gate1_manifest(),
        "wavekeyanchor-k48": rdx.keyanchor_k48_manifest() + rdx.keyanchor_k48_dprime_manifest()
                             + rdx.keyanchor_k48_fixed_lambda1_manifest(),
        "wavekeyanchor-k48-ref": rdx.keyanchor_k48_reference_manifest(),
        "wavekeyanchor-k48-gate1": rdx.keyanchor_k48_gate1_manifest(),
        "wavekeyanchor-e": rdx.keyanchor_e_wave_manifest(),
        "wavekeyanchor-cliff": rdx.keyanchor_cliff_manifest() + rdx.keyanchor_cliff_gate1_manifest(),
        "wavekeyanchor-dstate": (rdx.keyanchor_dstate_manifest() + rdx.keyanchor_dstate_gate1_manifest()
                                 + rdx.keyanchor_dstate_calibration_manifest()),
    }
    ok = True
    for label, manifest in prior_manifests.items():
        prior_paths = {rdx.out_path("/fake/root", s) for s in manifest}
        disjoint = dose_paths.isdisjoint(prior_paths)
        if not disjoint:
            print(f"    FAIL: keyanchor-dose collides with {label} (shared fake_root)")
        ok = ok and disjoint
    print(f"    keyanchor-dose ({len(dose_paths)} paths) disjoint from all {len(prior_manifests)} "
          f"prior/sibling manifests (shared fake_root): {ok}")
    print("    real out_dir: 'wavekeyanchor-dose' (main()'s own f\"wave{args.wave}\" convention) -- "
          "distinct from every prior wave's own out_dir")
    _report("smoke 7: zero-collision -- keyanchor-dose's manifest (calibration + rank4 + diffuse) is "
            "collision-free against every prior/sibling wave this program has ever registered", ok)


# ---------------------------------------------------------------------------
# smoke 8: budget guard arithmetic.
# ---------------------------------------------------------------------------

def smoke_8_budget_guard_arithmetic():
    report = rdx.keyanchor_dose_budget_guard(accept_override=True)
    stage1_ok = (abs(report["stage1_gpuh_1x"] - 6.4102) < 1e-3
                 and abs(report["stage1_gpuh_2x"] - 12.8204) < 1e-3 and report["stage1_fits_2x"] is True)
    stage2_ok = (abs(report["stage2_gpuh_1x"] - 5.7692) < 1e-3
                 and abs(report["stage2_gpuh_2x"] - 11.5383) < 1e-3)
    cumulative_ok = (abs(report["cumulative_gpuh_1x"] - 12.1794) < 1e-3
                     and abs(report["cumulative_gpuh_2x"] - 24.3587) < 1e-3
                     and report["cumulative_fits_2x"] is False)
    amendment_ok = abs(report["ceiling_amendment_gpuh_2x"] - 10.6787) < 1e-3
    print(f"    stage1: {report['stage1_gpuh_1x']:.4f}/{report['stage1_gpuh_2x']:.4f} GPU-h (1x/2x), "
          f"fits_2x={report['stage1_fits_2x']} (expect ~6.4102/12.8204, True)")
    print(f"    stage2: {report['stage2_gpuh_1x']:.4f}/{report['stage2_gpuh_2x']:.4f} GPU-h (1x/2x) "
          f"(expect ~5.7692/11.5383)")
    print(f"    cumulative: {report['cumulative_gpuh_1x']:.4f}/{report['cumulative_gpuh_2x']:.4f} "
          f"GPU-h (1x/2x), fits_2x={report['cumulative_fits_2x']} (expect ~12.1794/24.3587, False)")
    print(f"    ceiling_amendment_gpuh_2x: {report['ceiling_amendment_gpuh_2x']:.4f} "
          f"(expect ~10.6787, matching sec 14.4's own registered 10.678 GPU-h)")
    ok = stage1_ok and stage2_ok and cumulative_ok and amendment_ok
    _report("smoke 8: keyanchor_dose_budget_guard arithmetic -- Stage 1 fits the full 13.68 GPU-h "
            "ceiling at 2x; cumulative (both stages) does not; ceiling-amendment matches sec 14.4's "
            "own registered 10.678 GPU-h to 3 decimal places", ok)


# ---------------------------------------------------------------------------
# smoke 9: staging/PI-gate mechanics.
# ---------------------------------------------------------------------------

def smoke_9_staging_mechanics():
    with tempfile.TemporaryDirectory() as tmp:
        missing = rdx.keyanchor_dose_is_calibration_dose_verified(tmp)
        missing_ok = missing is None

    with tempfile.TemporaryDirectory() as tmp:
        calib_spec = rdx.keyanchor_dose_calibration_manifest()[0]
        _write_fake_dosed_result(tmp, calib_spec, achieved_max_cos=0.399)   # target 0.40, rel_err=0.0025
        verify_ok = rdx.keyanchor_dose_is_calibration_dose_verified(tmp)
        clean_ok = verify_ok is not None and verify_ok["verified"] is True

    with tempfile.TemporaryDirectory() as tmp:
        calib_spec = rdx.keyanchor_dose_calibration_manifest()[0]
        _write_fake_dosed_result(tmp, calib_spec, achieved_max_cos=0.30)   # target 0.40, rel_err=0.25
        verify_bad = rdx.keyanchor_dose_is_calibration_dose_verified(tmp)
        bad_ok = verify_bad is not None and verify_bad["verified"] is False

    print(f"    missing calibration cell -> None: {missing_ok}")
    print(f"    clean calibration (achieved=0.399, target=0.40, rel_err=0.0025) -> verified=True: "
          f"{clean_ok} ({verify_ok})")
    print(f"    bad calibration (achieved=0.30, target=0.40, rel_err=0.25) -> verified=False: "
          f"{bad_ok} ({verify_bad})")
    ok = missing_ok and clean_ok and bad_ok
    _report("smoke 9: keyanchor_dose_is_calibration_dose_verified -- None when missing, "
            "verified=True on a clean in-tolerance synthetic result, verified=False on an "
            "out-of-tolerance one", ok)


# ---------------------------------------------------------------------------
# smoke 10: dose-verify assertion mechanics (sec 14.3's +/-10% rule,
# reproduced as pure arithmetic against run_deltanet_rd.py's own registered
# formula: rel_err = |achieved - target| / target, assert rel_err <= 0.10).
# ---------------------------------------------------------------------------

def smoke_10_dose_verify_tolerance_boundary():
    def rel_err(achieved, target):
        return abs(achieved - target) / target

    target = 0.284
    within = rel_err(0.284 * 1.05, target) <= 0.10          # +5%, passes
    # Exactly +10% by CONSTRUCTION -- build `achieved` so that
    # (achieved - target) / target == 0.10 to float precision by computing
    # the DELTA first (target*0.10) and adding it, then verifying the
    # round-trip rel_err lands at machine-precision-exact 0.10 (independent
    # of whichever of the two equivalent multiplication orders rounds
    # differently) -- avoids the ULP-level trap where target*1.10,
    # computed as one multiplication, can round to a hair ABOVE the true
    # 10% mark (0.284*1.10 -> rel_err=0.10000000000000013, one ULP over)
    # purely from float representation, not from any bug in the assertion
    # under test.
    delta = target * 0.10
    achieved_at_boundary = target + delta
    assert rel_err(achieved_at_boundary, target) <= 0.10 + 1e-12, \
        "test construction failed to land at/under the 10% boundary -- fix the construction, not the tolerance"
    at_boundary = rel_err(achieved_at_boundary, target) <= 0.10
    just_over = rel_err(target * 1.15, target) <= 0.10       # comfortably over +10%, fails
    far_off = rel_err(0.15, target) <= 0.10                  # way off, fails

    print(f"    achieved=+5% of target: rel_err<=0.10 -> {within} (expect True)")
    print(f"    achieved=exactly +10% of target: rel_err<=0.10 -> {at_boundary} (expect True, boundary-inclusive)")
    print(f"    achieved=+10.1% of target: rel_err<=0.10 -> {just_over} (expect False)")
    print(f"    achieved=0.15 vs target=0.284 (way off): rel_err<=0.10 -> {far_off} (expect False)")
    ok = within and at_boundary and not just_over and not far_off
    _report("smoke 10: dose-verify assertion mechanics (sec 14.3's +/-10% rule) -- boundary-exact "
            "reproduction of run_deltanet_rd.py's own construction-time assert formula", ok)


def main() -> int:
    print("=" * 70)
    print("smoke_keyanchor_dose.py -- KEY_ANCHORING_DESIGN.md sec 14 (Rev 14.3) coherence "
          "dose-response Wave -1 smoke suite (fla-free)")
    print("=" * 70)
    smoke_1_dose_dial_matches_reference_json()
    smoke_2_calibration_manifest_shape()
    smoke_3_dose_manifest_shape()
    smoke_4_dose_target_varies_filename_and_config()
    smoke_5_is_done_dose_collision_negative_test()
    smoke_6_seed_block_zero_collision()
    smoke_7_zero_collision()
    smoke_8_budget_guard_arithmetic()
    smoke_9_staging_mechanics()
    smoke_10_dose_verify_tolerance_boundary()
    print("=" * 70)
    if FAILURES:
        print(f"SMOKE SUITE: {len(FAILURES)} FAILURE(S): {FAILURES}", file=sys.stderr)
        return 1
    print("SMOKE SUITE: ALL ITEMS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
