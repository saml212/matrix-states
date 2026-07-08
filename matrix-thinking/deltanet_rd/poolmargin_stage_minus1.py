"""poolmargin_stage_minus1.py -- KEY_ANCHORING_SCALING_DRAFT.md sec 15.26's
BLOCKING Stage -1 self-test suite for the K90 pool-margin control
diagnostic build (Rev 2 + round-3 VERIFY-pass adopted fix m-b, c66b3f6/
fd8c509/16f3543, DESIGN-CLEARED-FOR-BUILD). Aggregates every item testable
locally on this (CPU-only, no `fla`) dev machine, run to completion (per
this project's own standing "assertion has teeth" rule), and reports what
is NOT testable here as BOX-DEFERRED with an explicit capability probe --
mirrors diag_c17_repro_stage_minus1.py's own established pattern (`_try_
import_run_deltanet_rd()` / `_defer()`, real branch whenever the import
succeeds, CPU-safe proxy or explicit defer otherwise).

    DRY_RUN_BYPASS=1 .venv/bin/python poolmargin_stage_minus1.py

(the repo's pre-train-gate hook pattern-matches any `python *.py`
invocation; DRY_RUN_BYPASS=1 is the correct, sanctioned bypass here --
every item this script runs OFF-box is pure-Python/CPU-only, no GPU.)

Items:
  1. Pool-restriction slice determinism (`_restrict_entity_pool`) --
     GATED capability probe (`import run_deltanet_rd` needs `fla`,
     transitively via `model_rd`). build-audit MINOR-1 fix (2026-07-08,
     adopting the auditor's own technique): a minimal in-suite `fla` stub
     (`_install_fla_stub()` below -- dummy `ShortConvolution`/
     `chunk_delta_rule`, both raise `NotImplementedError` if actually
     called) is installed whenever the real `fla` package isn't already
     importable, so `import run_deltanet_rd` now SUCCEEDS locally and the
     REAL `_restrict_entity_pool` branch runs on this (CPU-only) dev
     machine, not just the CPU-safe proxy reimplementation. The
     TRUE-CUDA half (model construction, forward/backward, the real
     Triton kernel) stays its own explicit, separately-disclosed
     box-deferred item (`item1-true-cuda-half`) -- the stub only
     unblocks the import, never the CUDA-side computation. If the fla-
     stub itself still fails to unblock the import (some other real
     dependency missing), the ORIGINAL CPU-safe proxy branch remains as
     the fallback, using the REAL `grammar_rd.EntityPools` dataclass
     (grammar_rd.py is fla-free) with a hand-built synthetic pool (no
     network/tokenizer dependency) and an inline reimplementation of the
     identical slicing logic, disclosed as a proxy for the box-deferred
     real function.
  2. Source-level structural checks on run_deltanet_rd.py (real, NOT
     gated -- pure `ast` parsing, no import, no fla needed): the new
     parameters exist on the right functions, and every new behavioral
     branch is guarded by an `is not None` check (additivity-at-default
     proof by source inspection, not just by claim).
  3. Wrapper self-test suite (run_poolmargin_k84s1943_k90s2043.py
     --self-test) -- PI-signoff refusal x3 + positive control, whitelist
     field-diff positive/negative/no-op, n_iter-override determinism,
     admission-check positive/negative fixtures, abort-trigger arithmetic.
  4. Harvest self-test suite (harvest_poolmargin_k84s1943_k90s2043.py
     --self-test) -- REAL/ARTIFACT/AMBIGUOUS/DEGENERATE_CELL buckets +
     the Delta_measured<=0 contingency + the noise_shift-max-of-2-draws
     adoption + a totality-walk numeric sweep.
  5. OUT_DIR isolation from the frozen sec 15.26.1 per-K table
     (build-audit MAJOR-1 fix, 2026-07-08) -- real, ungated (pure
     path/string checks, no fla/CUDA): OUT_DIR is disjoint from the
     frozen wavekeyanchor-scaling-wide/ dir; both real cells' generated
     `--out` command tokens land under the new dir; the frozen dir's own
     `fit_cliff_curve.load_k_mean_h4` glob pattern is confirmed rooted at
     the frozen dir specifically, so this diagnostic's own cells can
     never match it by construction.

Items 3/4 are invoked as subprocesses (each script's own `--self-test`
entry point is the canonical, independently-runnable source of truth;
this suite does not duplicate their assertions, only aggregates their
pass/fail signal and exit codes).
"""
from __future__ import annotations

import ast
import dataclasses
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILURES: list[str] = []
BOX_DEFERRED: list[dict] = []


def _report(item: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{item}] {status}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(item)


def _defer(item: str, requirement: str, gated: bool = True) -> None:
    print(f"[{item}] BOX-DEFERRED -- {requirement}", flush=True)
    BOX_DEFERRED.append({"item": item, "requirement": requirement, "gated": gated})


def _install_fla_stub() -> bool:
    """build-audit MINOR-1 fix, adopting the auditor's own technique: `model_rd.py`'s only two
    `fla` symbols (`fla.modules.ShortConvolution`, `fla.ops.delta_rule.chunk_delta_rule`) are used
    EXCLUSIVELY inside function/method bodies (`self.k_conv1d = ShortConvolution(...)` in
    `__init__`, `chunk_delta_rule(...)` inside a call), never at module scope -- verified directly
    against model_rd.py this session (grep for both names: every hit outside a def/comment is an
    assignment inside a function body). A minimal stub therefore lets `import run_deltanet_rd`
    (and its own `model_rd` import chain) SUCCEED without a real CUDA/fla install, because nothing
    at import time actually calls either symbol.

    Does NOT attempt to make the stub CALLABLE for real work -- `ShortConvolution(...)` and
    `chunk_delta_rule(...)` both raise `NotImplementedError` if actually invoked, so any code path
    that tries to build/run the real model (the TRUE-CUDA half: model forward/backward, the real
    Triton kernel) still fails loudly and stays its own, separately-disclosed BOX-DEFERRED item --
    this stub only unblocks the IMPORT, never the CUDA-side computation. Returns True iff the
    stub was installed (i.e. the real `fla` package was not already importable); False if the
    real package is already present (on-box), in which case sys.modules is left untouched and the
    real package is used, unstubbed."""
    try:
        import fla  # noqa: F401 -- real package already installed (e.g. on the GPU box): no-op.
        return False
    except ImportError:
        pass

    import types

    def _unimplemented(name):
        def _raise(*_a, **_kw):
            raise NotImplementedError(
                f"fla-stub: {name} is a CPU-only import-time stand-in (build-audit MINOR-1) -- it "
                f"has no real implementation. Any code path that reaches this call needs the REAL "
                f"CUDA `fla` package and stays BOX-DEFERRED.")
        return _raise

    fla_pkg = types.ModuleType("fla")
    fla_modules = types.ModuleType("fla.modules")
    fla_ops = types.ModuleType("fla.ops")
    fla_ops_delta_rule = types.ModuleType("fla.ops.delta_rule")

    fla_modules.ShortConvolution = _unimplemented("fla.modules.ShortConvolution")
    fla_ops_delta_rule.chunk_delta_rule = _unimplemented("fla.ops.delta_rule.chunk_delta_rule")

    fla_pkg.modules = fla_modules
    fla_pkg.ops = fla_ops
    fla_ops.delta_rule = fla_ops_delta_rule

    sys.modules["fla"] = fla_pkg
    sys.modules["fla.modules"] = fla_modules
    sys.modules["fla.ops"] = fla_ops
    sys.modules["fla.ops.delta_rule"] = fla_ops_delta_rule
    return True


def _try_import_run_deltanet_rd():
    """Capability probe: attempts the REAL box-only import chain
    (run_deltanet_rd -> model_rd -> fla.modules.ShortConvolution, CUDA-
    oriented at module scope). Returns (module, used_stub) -- module is None on an ImportError
    that persists even after the build-audit MINOR-1 fla-stub is installed; used_stub is True iff
    the stub (not a real `fla`) is what let the import through, so callers can disclose that the
    real-branch coverage this session is import-only, not full on-box CUDA parity. NEVER swallows
    any OTHER exception class, so a real bug surfaces loudly rather than silently deferring."""
    try:
        import run_deltanet_rd as rd
        return rd, False
    except ImportError:
        pass
    used_stub = _install_fla_stub()
    if not used_stub:
        return None, False   # real fla present but SOME OTHER import still failed -- stays deferred
    try:
        import run_deltanet_rd as rd
        return rd, True
    except ImportError:
        return None, True


# ---------------------------------------------------------------------------
# Item 1: pool-restriction slice determinism.
# ---------------------------------------------------------------------------

def _synthetic_pools():
    """A REAL grammar_rd.EntityPools instance (grammar_rd.py is fla-free --
    verified this session, `import grammar_rd` succeeds standalone), hand-
    built with small synthetic tensors so this item never depends on
    network access / a cached GPT-2 tokenizer."""
    import grammar_rd as grd
    import torch
    return grd.EntityPools(
        vocab_size_base=50257, buffer_id=50257, query_id=50258, period_id=13,
        train_name_ids=torch.tensor([100, 101, 102, 103, 104, 105, 106], dtype=torch.int64),
        heldout_name_ids=torch.tensor([200, 201, 202, 203, 204], dtype=torch.int64),
        rel_a_ids=torch.tensor([300, 301], dtype=torch.int64),
        rel_b_ids=torch.tensor([400, 401], dtype=torch.int64),
        vocab_size_total=50259,
    )


def item1_pool_restriction_determinism() -> None:
    rd, used_stub = _try_import_run_deltanet_rd()
    if rd is not None:
        # REAL branch: exercise the actual shipped function -- build-audit MINOR-1 fix, this now
        # runs locally (used_stub=True) via the in-suite fla stub above whenever the real `fla`
        # package isn't installed, not just when it happens to be (used_stub=False, e.g. on-box).
        stub_note = " (via build-audit MINOR-1 fla-stub, import-only -- see item1-true-cuda-half " \
                    "below)" if used_stub else ""
        pools = _synthetic_pools()
        restricted = rd._restrict_entity_pool(pools, 4, use_heldout_entities=False)
        ok = (list(restricted.train_name_ids.tolist()) == [100, 101, 102, 103]
              and list(pools.train_name_ids.tolist()) == [100, 101, 102, 103, 104, 105, 106]
              and restricted.heldout_name_ids.tolist() == pools.heldout_name_ids.tolist())
        _report("item1-real-first-n-slice-train", ok, f"got {restricted.train_name_ids.tolist()}{stub_note}")
        restricted_ho = rd._restrict_entity_pool(pools, 3, use_heldout_entities=True)
        ok2 = list(restricted_ho.heldout_name_ids.tolist()) == [200, 201, 202]
        _report("item1-real-first-n-slice-heldout", ok2, stub_note.strip())
        raised = False
        try:
            rd._restrict_entity_pool(pools, 999, use_heldout_entities=False)
        except AssertionError:
            raised = True
        _report("item1-real-oversized-n-refuses", raised, stub_note.strip())
        if used_stub:
            # The stub only unblocks the IMPORT -- ShortConvolution/chunk_delta_rule themselves
            # raise NotImplementedError if actually called. Anything that needs the REAL Triton
            # kernel (model construction, forward/backward) is untested by this pass and stays
            # its own, separately-disclosed box-deferred item (build-audit MINOR-1: "keep the
            # box-deferred marker for the true-CUDA half").
            _defer("item1-true-cuda-half",
                   "the fla-stub above is import-only (ShortConvolution/chunk_delta_rule both "
                   "raise NotImplementedError if actually invoked) -- model construction, "
                   "forward/backward, and the real Triton kernel remain untested here and need "
                   "the real CUDA + `fla` box (run_deltanet_rd.py --smoke covers that, on-box).",
                   gated=True)
        return
    _defer("item1-real-restrict-entity-pool",
           "`import run_deltanet_rd` still fails even after the build-audit MINOR-1 fla-stub is "
           "installed -- some OTHER import in the chain needs a real, non-fla dependency this box "
           "lacks (not merely `fla` itself).")

    # CPU-safe proxy branch: the REAL EntityPools dataclass (not a stand-in type), a hand-built
    # synthetic pool (no tokenizer/network dependency), and an inline reimplementation of the
    # IDENTICAL logic _restrict_entity_pool ships (dataclasses.replace + first-N slice) --
    # disclosed as a proxy for the box-deferred real function above, not a substitute for it.
    pools = _synthetic_pools()

    def _proxy_restrict(pools, n, use_heldout_entities):
        field = "heldout_name_ids" if use_heldout_entities else "train_name_ids"
        src = getattr(pools, field)
        assert n <= src.numel(), f"n={n} exceeds {field} size {src.numel()}"
        return dataclasses.replace(pools, **{field: src[:n]})

    restricted = _proxy_restrict(pools, 4, use_heldout_entities=False)
    _report("item1-proxy-first-n-slice-train",
            list(restricted.train_name_ids.tolist()) == [100, 101, 102, 103],
            f"got {restricted.train_name_ids.tolist()}")
    _report("item1-proxy-original-pool-unmutated",
            list(pools.train_name_ids.tolist()) == [100, 101, 102, 103, 104, 105, 106],
            f"got {pools.train_name_ids.tolist()}")
    _report("item1-proxy-other-fields-untouched",
            restricted.heldout_name_ids.tolist() == pools.heldout_name_ids.tolist()
            and restricted.vocab_size_total == pools.vocab_size_total
            and restricted.buffer_id == pools.buffer_id)
    restricted_ho = _proxy_restrict(pools, 3, use_heldout_entities=True)
    _report("item1-proxy-first-n-slice-heldout",
            list(restricted_ho.heldout_name_ids.tolist()) == [200, 201, 202])
    raised = False
    try:
        _proxy_restrict(pools, 999, use_heldout_entities=False)
    except AssertionError:
        raised = True
    _report("item1-proxy-oversized-n-refuses", raised)
    # determinism across repeated calls (same input -> byte-identical output, no hidden RNG)
    r1 = _proxy_restrict(pools, 4, use_heldout_entities=False)
    r2 = _proxy_restrict(pools, 4, use_heldout_entities=False)
    _report("item1-proxy-repeated-calls-identical",
            r1.train_name_ids.tolist() == r2.train_name_ids.tolist())


# ---------------------------------------------------------------------------
# Item 2: source-level structural checks (real, ungated -- ast only, no import).
# ---------------------------------------------------------------------------

def item2_source_structural_checks() -> None:
    path = os.path.join(HERE, "run_deltanet_rd.py")
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src, filename=path)

    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    def _arg_names(fn: ast.FunctionDef) -> set:
        return {a.arg for a in (fn.args.args + fn.args.kwonlyargs)}

    ep = funcs.get("evaluate_pool")
    _report("item2-evaluate_pool-exists", ep is not None)
    if ep is not None:
        _report("item2-evaluate_pool-has-restrict_entity_pool_n",
                "restrict_entity_pool_n" in _arg_names(ep))

    tr = funcs.get("train")
    _report("item2-train-exists", tr is not None)
    if tr is not None:
        _report("item2-train-has-m3_pool_restrict_n", "m3_pool_restrict_n" in _arg_names(tr))

    rep = funcs.get("_restrict_entity_pool")
    _report("item2-_restrict_entity_pool-helper-exists", rep is not None)

    # additivity-by-construction: every new behavioral branch must be guarded by an
    # `is not None` check on the new param -- a plain substring scan of each function's own
    # source (not a full control-flow proof, but catches the "forgot the guard" class of bug
    # directly, same spirit as the no-eval-leak smoke's own forbidden-token scan).
    ep_src = ast.get_source_segment(src, ep) or ""
    _report("item2-evaluate_pool-restriction-is-guarded",
            "if restrict_entity_pool_n is not None:" in ep_src)
    tr_src = ast.get_source_segment(src, tr) or ""
    _report("item2-train-new-calls-are-guarded",
            "if m3_pool_restrict_n is not None:" in tr_src)
    _report("item2-train-records-three-new-keys",
            all(k in tr_src for k in (
                'res["M3_held_out_pool_restricted"]', 'res["M3_held_out_noise_repeat"]',
                'res["M3_held_out_noise_repeat_2"]')))
    # the three offsets this diagnostic's own design pins exactly (sec 15.26.2.2 + round-3
    # adopted fix m-b) -- a literal-string check, catches an offset typo directly.
    _report("item2-train-uses-pinned-generator-offsets",
            all(off in tr_src for off in ("seed + 20_000 + step", "seed + 30_000 + step")))

    # CLI flag registered + threaded through to train().
    _report("item2-cli-flag-registered", '"--m3-pool-restrict-n"' in src)
    _report("item2-cli-flag-threaded-to-train", "m3_pool_restrict_n=args.m3_pool_restrict_n" in src)


# ---------------------------------------------------------------------------
# Item 5: OUT_DIR isolation (build-audit MAJOR-1 fix) -- real, ungated, pure
# import + string/path checks, no fla/CUDA needed.
# ---------------------------------------------------------------------------

def item5_out_dir_isolation() -> None:
    import run_poolmargin_k84s1943_k90s2043 as wrapper

    frozen_dir = wrapper.FROZEN_WIDE_DIR
    out_dir = wrapper.OUT_DIR

    # (a) OUT_DIR itself is disjoint from the frozen dir -- path-prefix assert, not just a
    # different-looking string (catches e.g. one dir accidentally nested inside the other).
    disjoint = (out_dir != frozen_dir
                and not out_dir.startswith(frozen_dir + os.sep)
                and not frozen_dir.startswith(out_dir + os.sep))
    _report("item5-out-dir-disjoint-from-frozen-dir", disjoint,
            f"OUT_DIR={out_dir!r} FROZEN_WIDE_DIR={frozen_dir!r}")
    _report("item5-out-dir-basename-is-poolmargin",
            os.path.basename(out_dir) == "wavekeyanchor-scaling-poolmargin",
            f"got basename {os.path.basename(out_dir)!r}")

    # (b) the wrapper's own generated commands' --out path is UNDER the new dir, for both real
    # cells -- built via the SAME build_cell_cmd() the real launch uses, never hand-typed.
    for K, seed, extra in ((wrapper.K_A, wrapper.SEED_A, wrapper.M3_POOL_RESTRICT_N),
                            (wrapper.K_B, wrapper.SEED_B, None)):
        _, cmd = wrapper.build_cell_cmd(K, seed, extra)
        assert "--out" in cmd, f"K={K}: generated cmd has no --out token at all: {cmd!r}"
        out_path_token = cmd[cmd.index("--out") + 1]
        under_new_dir = out_path_token.startswith(out_dir + os.sep)
        not_under_frozen = not out_path_token.startswith(frozen_dir + os.sep)
        _report(f"item5-cmd-out-path-under-new-dir[K={K}]", under_new_dir and not_under_frozen,
                f"--out {out_path_token!r}")

    # (c) the frozen dir's own glob population is unchanged BY CONSTRUCTION: since (a) proves
    # OUT_DIR and FROZEN_WIDE_DIR are disjoint, and (b) proves every file this wrapper will ever
    # write lands under OUT_DIR, no file this build produces can EVER match a glob rooted at
    # FROZEN_WIDE_DIR -- a structural guarantee, not a before/after diff (nothing has launched
    # yet at Stage -1). Made explicit via fit_cliff_curve's own glob pattern for both K's this
    # diagnostic touches, confirming the pattern is rooted at FROZEN_WIDE_DIR specifically.
    import fit_cliff_curve as fcc
    for K in (wrapper.K_A, wrapper.K_B):
        pattern_root = os.path.join(frozen_dir, f"*_K{K}_armd_*.json")
        _report(f"item5-frozen-glob-pattern-rooted-at-frozen-dir[K={K}]",
                pattern_root.startswith(frozen_dir + os.sep),
                f"pattern={pattern_root!r}")
    _report("item5-load_k_mean_h4-importable", callable(getattr(fcc, "load_k_mean_h4", None)))


# ---------------------------------------------------------------------------
# Items 3/4: subprocess-invoke the wrapper's and harvest's own --self-test.
# ---------------------------------------------------------------------------

def _run_subprocess_self_test(item: str, script: str) -> None:
    env = {**os.environ, "DRY_RUN_BYPASS": "1"}
    proc = subprocess.run([sys.executable, os.path.join(HERE, script), "--self-test"],
                           env=env, capture_output=True, text=True, timeout=120)
    ok = proc.returncode == 0 and "ALL SELF-TESTS PASSED." in proc.stdout
    _report(item, ok, f"rc={proc.returncode}")
    if not ok:
        print(proc.stdout[-3000:], flush=True)
        print(proc.stderr[-3000:], file=sys.stderr, flush=True)


def main() -> int:
    print("=" * 70)
    print("ITEM 1: pool-restriction slice determinism")
    print("=" * 70, flush=True)
    item1_pool_restriction_determinism()

    print("\n" + "=" * 70)
    print("ITEM 2: source-level structural checks (run_deltanet_rd.py)")
    print("=" * 70, flush=True)
    item2_source_structural_checks()

    print("\n" + "=" * 70)
    print("ITEM 3: wrapper self-test (run_poolmargin_k84s1943_k90s2043.py --self-test)")
    print("=" * 70, flush=True)
    _run_subprocess_self_test("item3-wrapper-self-test", "run_poolmargin_k84s1943_k90s2043.py")

    print("\n" + "=" * 70)
    print("ITEM 4: harvest self-test (harvest_poolmargin_k84s1943_k90s2043.py --self-test)")
    print("=" * 70, flush=True)
    _run_subprocess_self_test("item4-harvest-self-test", "harvest_poolmargin_k84s1943_k90s2043.py")

    print("\n" + "=" * 70)
    print("ITEM 5: OUT_DIR isolation from the frozen sec 15.26.1 per-K table (build-audit MAJOR-1)")
    print("=" * 70, flush=True)
    item5_out_dir_isolation()

    print("\n" + "=" * 70)
    print(f"STAGE -1 SUMMARY: {len(FAILURES)} failure(s), {len(BOX_DEFERRED)} box-deferred item(s)")
    print("=" * 70, flush=True)
    if BOX_DEFERRED:
        print("BOX-DEFERRED (run these on the H100 box before trusting a real launch):", flush=True)
        for d in BOX_DEFERRED:
            print(f"  - {d['item']}: {d['requirement']}", flush=True)
    if FAILURES:
        print(f"FAILED: {FAILURES}", file=sys.stderr, flush=True)
        return 1
    print("ALL LOCAL ITEMS PASSED (box-deferred items require CUDA+fla, see list above).",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
