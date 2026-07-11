# Stage 2 — fix + permanent regression + independent audit (2026-07-11)

Full narrative record: `matrix-thinking/REASONING_LINK_DESIGN.md` §17.2.

## The fix (one line + docstring)

`matrix-thinking/deltanet_rd/reasoning_link_probe.py::squeeze_state_head`:

```
-    return final_state[:, 0, :, :]
+    return final_state[:, 0, :, :].transpose(-1, -2)
```

Mirrors `model_rd.py::kernel_state_design_layout`'s own
`S_kv.squeeze(1).float().transpose(-1, -2)` (audit round-1 FATAL-0's fix,
which this file's independent `squeeze_state_head` never inherited).
fla returns `final_state[N,H,K,V]` (key axis first); `apply_state_power`'s
einsum (`'bij,bqj->bqi'`) requires the design `[V,K]` layout.

## Permanent regression guard

`matrix-thinking/deltanet_rd/reasoning_link_stage_minus1.py` item 20
(`test_item_20_squeeze_state_head_layout_closed_form`), registered in
`ALL_ITEMS`. Hand-computed closed form on a synthetic `(1,1,4,4)`
final_state with one nonzero entry at fla's `[key=0,value=1]` slot:
post-fix retrieval recovers the written value exactly (atol 1e-6); the
PRE-FIX untransposed layout (inlined) is proven BY EXECUTION to fail the
identical assertion at cos ~ 0 (kill-proof run to completion). CPU-only,
kernel-free, stub-independent — a standing guard.

Local verification transcript (M4 dev venv):

```
import OK, FLA_STUB_INSTALLED = True
[Stage-1 item 20] squeeze_state_head+apply_state_power closed-form layout check (REASONING_LINK sec 17 fix) -- post-fix design-convention retrieval recovers the written value exactly (atol 1e-6); the pre-fix untransposed layout is proven, by an executed mutation/kill-proof, to fail the identical assertion (cos~0, matching the real poscontrol signature): PASS
[Stage-1 item 2] hand-built S^h@q + cosine scorer reproduce pinned values to 1e-6: PASS
[Stage-1 extra] outcome-routing gates (sec 12) mechanically fire as pre-registered: PASS
```

(Run via a scratchpad harness that pre-registers a minimal
`fla.ops.gated_delta_rule` stub entry — required because of the UNRELATED,
pre-existing `_ensure_fla_stub` gap documented at §17.2's DISCOVERY note;
the full `--mode selftest` suite currently cannot import on any machine,
box included, under the chain's own documented invocation. Out of scope
here; flagged for a separately-scoped fix.)

## Independent audit (fresh Opus subagent)

VERDICT: **PASS — fix correct, minimal, correctly scoped.**

- Diff review: change is exactly the transpose + docstring; matches
  `kernel_state_design_layout`'s convention (verified from both function
  bodies, not docstrings).
- Kill-proof review: einsum index arithmetic hand-worked independently —
  post-fix e0-query recovers e1 exactly; pre-fix layout yields the exact
  zero vector (cos = 0). Teeth confirmed.
- No-behavior-change-elsewhere: `squeeze_state_head` has exactly one
  production call site (`measure_cell_all_h`), feeding `apply_state_power`
  (intended target) and `state_condition_number` — the latter proven
  transpose-invariant (svdvals(S) = svdvals(S^T), shared nonzero
  eigenvalues of S^TS/SS^T). No Stage -1 item 1-19 depends on the old
  orientation. `reasoning_link_poscontrol.py`'s "deliberately-transposed"
  arm coherently role-swaps into the pre-fix-equivalent negative control.
- Scope: git diff confirmed only the two intended files touched;
  `model_rd.py` untouched.
- Findings (non-blocking): F1 — commit the real-kernel PASS artifact
  (satisfied: post-fix adjudication + poscontrol re-run JSONs in this
  archive); F2 — fix docstring omits a harmless `.float()` dtype
  difference vs `kernel_state_design_layout` (not a layout issue).
