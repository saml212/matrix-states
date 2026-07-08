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
     transitively via `model_rd`). Real branch when capable; CPU-safe
     proxy branch otherwise, using the REAL `grammar_rd.EntityPools`
     dataclass (grammar_rd.py is fla-free) with a hand-built synthetic
     pool (no network/tokenizer dependency) and an inline reimplementation
     of the identical slicing logic, disclosed as a proxy for the
     box-deferred real function.
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


def _try_import_run_deltanet_rd():
    """Capability probe: attempts the REAL box-only import chain
    (run_deltanet_rd -> model_rd -> fla.modules.ShortConvolution, CUDA-
    oriented at module scope). Returns the module on success, None on
    ImportError -- NEVER swallows any OTHER exception class, so a real bug
    surfaces loudly rather than silently deferring."""
    try:
        import run_deltanet_rd as rd
        return rd
    except ImportError:
        return None


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
    rd = _try_import_run_deltanet_rd()
    if rd is not None:
        # REAL branch: exercise the actual shipped function.
        pools = _synthetic_pools()
        restricted = rd._restrict_entity_pool(pools, 4, use_heldout_entities=False)
        ok = (list(restricted.train_name_ids.tolist()) == [100, 101, 102, 103]
              and list(pools.train_name_ids.tolist()) == [100, 101, 102, 103, 104, 105, 106]
              and restricted.heldout_name_ids.tolist() == pools.heldout_name_ids.tolist())
        _report("item1-real-first-n-slice-train", ok)
        restricted_ho = rd._restrict_entity_pool(pools, 3, use_heldout_entities=True)
        ok2 = list(restricted_ho.heldout_name_ids.tolist()) == [200, 201, 202]
        _report("item1-real-first-n-slice-heldout", ok2)
        raised = False
        try:
            rd._restrict_entity_pool(pools, 999, use_heldout_entities=False)
        except AssertionError:
            raised = True
        _report("item1-real-oversized-n-refuses", raised)
        return
    _defer("item1-real-restrict-entity-pool", "needs CUDA + `fla` (transitively via model_rd) -- "
           "`import run_deltanet_rd` raises ModuleNotFoundError: No module named 'fla' in this "
           "repo's own .venv on this (CPU-only) dev machine.")

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
