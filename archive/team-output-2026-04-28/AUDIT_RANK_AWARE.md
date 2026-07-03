# AUDIT_RANK_AWARE — Independent Verification of Rank-Aware Patch Fixes

Auditor: independent agent, fresh context.
Target: `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/scripts/run_matrix_codi.py`
Smoke: `/Users/samuellarson/Experiments/learned-representations/matrix-thinking/scripts/_smoke_rank_aware.py`
Original attack: `team-output-2026-04-28/ATTACK_RANK_AWARE.md`

---

## Check 1 — Attack #2 fix landed (full-SVD NaN at coincident σ): **PASS**

**Code inspection:**
- `MatrixBottleneck.forward` force-rank block (lines 323–333): uses `torch.linalg.eigh(ZZt)`, NOT `torch.linalg.svd`. Confirmed.
- `truncate_to_rank` (lines 372–405): uses `torch.linalg.eigh(ZZt)`, NOT full `torch.linalg.svd`. Confirmed.
- `compute_rank_loss` (line 434): uses `torch.linalg.svdvals` (which is gradient-safe via eigh internally — original attacker explicitly rated this safe). Confirmed.

**Smoke test:** ran `_smoke_rank_aware.py` end-to-end with `KMP_DUPLICATE_LIB_OK=TRUE`. ALL 5 tests pass, including the new TEST 5 (degenerate spectrum [3.0, 1.0, 1.0, 0.0]) — `nan_count=0, inf_count=0`. Grad max abs is large (~323K) but finite.

**Independent test (Attack A):** constructed Z with all 4 σ values = 1.0 (maximally degenerate). Result: forward NaN-free, backward NaN-free. Grad max ~3.5M but finite — eigh-mode large-but-finite behavior, will be handled by standard grad clipping during real training.

**Independent test (Attack F):** 10 SGD micro-steps with force_rank_k=2 on a TinyModel(d=8). No NaN/Inf in any parameter gradient across all 10 steps.

**Verdict: PASS.** The eigh-based projection replaces full SVD at both production sites. Fix landed correctly.

---

## Check 2 — Attack #3 fix landed (autocast wrap): **PASS**

**Code inspection (all three sites wrap with `torch.autocast(..., enabled=False)`):**
- Line 325: force-rank block in `MatrixBottleneck.forward` — wraps with `torch.autocast(device_type=Z.device.type, enabled=False)` around the float cast + eigh. ✓
- Line 397: `truncate_to_rank` — same wrap. ✓
- Line 433: `compute_rank_loss` `svdvals` call — same wrap. ✓

**Independent test (Attack B):** bf16 input tensor passed through `truncate_to_rank` while wrapped in an outer `torch.autocast(dtype=bfloat16, enabled=True)`. Result: no NaN in output, output dtype matches input bf16, backward succeeds with no NaN gradients.

**Verdict: PASS.** All three new sites are autocast-disabled. Pattern matches the existing "Fix SERIOUS-13" block at line 338.

---

## Check 3 — Fix regressions / new bugs

### 3a. Sign of entropy and nuclear losses: **PASS**
- Line 444: `loss_terms.append(-H.mean())` — minimizing -H maximizes H. ✓
- Line 449: `loss_terms.append(-nuc.mean())` — minimizing -||Z||_* maximizes ||Z||_*. ✓
- Line 1199: `L_total = L_total + _rank_lambda * L_rank` — added with positive λ. ✓
- **Independent SGD test (Attack E):** entropy increased 1.2333→1.2335; nuclear increased 6.186→6.206 after one descent step. Signs correct.

### 3b. Shape regression: **PASS**
- Verified via Attack D: input (B, d, d) → output (B, d, d) for B∈{1,3,8}, d∈{4,8,16}, k∈{2,4,8}. All preserved.

### 3c. Default-path byte identity (rank_loss=none, λ=0, force_rank=0): **PASS (functional), MINOR COSMETIC**
- `_force_rank_k` resolves to None when arg=0 (lines 1126–1128); the `force_rank_k > 0` guard at line 323 prevents any extra ops. ✓
- `_need_rank_loss` is False (line 1121); rank-loss block at lines 1195–1199 is fully guarded; `compute_rank_loss` is never called. ✓
- Cosmetic regression: `Z_list_grad = []` is allocated unconditionally during matrix-CODI training (line 629), and `Z_t` is appended unconditionally (line 644). The `L_rank = torch.zeros(1, ...)` placeholder at line 1194 is allocated every step. Neither alters the optimizer's gradient — Z_t is already alive in the autograd graph through `running_embeds → out_full`, so Z_list_grad adds only Python references, no GPU memory. The original attacker (Attack 1) already classified this as DOES NOT LAND functionally; the agent did not apply the suggested cosmetic guard. **Not a regression — it's pre-existing cosmetic noise that the maintainer chose not to address.**

### 3d. `Z_list_grad` vs `Z_list` leak when rank-loss is OFF: **PASS**
- `Z_list_grad` references Z_t which already participates in the live autograd graph via `h_next → running_embeds`. Adding a Python reference cannot extend graph lifetime beyond what already exists. Backward pass through `compute_codi_loss` is gated by `_need_rank_loss`, so `compute_rank_loss` never runs and no extra `linalg.svdvals` op enters the graph. ✓

### 3e. eigh ordering — top-k vs bottom-k: **PASS**
- `torch.linalg.eigh` returns eigenvalues in ASCENDING order. Code uses `eigvecs[..., -k:]` which slices the LAST k eigenvectors → corresponds to the LARGEST eigenvalues of Z Z^T → correct.
- **Independent test (Attack C):** Z with σ=[10, 2, 1, 0.1] → after `truncate_to_rank(Z, 2)` → output spectrum [10.0, 2.0, ~3e-7, ~3e-8]. Top-2 retained, bottom-2 zeroed. **Correct.**

---

## Check 4 — SERIOUS-13 condition (existing fix preserved): **PASS**

- Original SERIOUS-13 fix at lines 336–339 (`rank_project_k` block in `MatrixBottleneck.forward`) is intact: `with torch.autocast("cuda", enabled=False): Z_out = truncate_to_rank(Z.float(), rank_project_k).to(Z.dtype)`.
- Stylistic note: the old block hardcodes `"cuda"`; the three new blocks use `Z.device.type` which is more portable. Inconsistent style, but functionally correct — both are valid spellings.

---

## Additional findings

- **Argparse defaults** (lines 2408–2438): `--rank-loss` defaults to `"none"`, `--rank-lambda` to `0.0`, `--force-rank-during-training` to `0`. Now explicit (matches the original attacker's "preferred fix B" which was to add the keys to CONFIG too — done at lines 115–117). ✓
- **Input validation** (lines 2467–2469): `assert args.force_rank_during_training >= 0` is now in place. The original attacker's "additional note" suggested this. ✓
- **Eigh behavior at maximally degenerate spectra:** when all σ are equal (e.g. all 1.0), eigh produces an arbitrary basis in the degenerate eigenspace, but the *projection* `U_k U_k^T Z` is still well-defined and the gradient is finite (verified ~3.5M, large but non-NaN). In practice the user's grad clipping (the existing trainer applies it) will tame this.

---

## Summary table

| Check | Verdict |
|---|---|
| 1. Attack #2 fix landed (eigh replaces full SVD) | **PASS** |
| 2. Attack #3 fix landed (autocast wrap on all 3 sites) | **PASS** |
| 3a. Sign of entropy / nuclear losses | **PASS** |
| 3b. Shape preservation | **PASS** |
| 3c. Default-path byte identity | **PASS** (cosmetic noise pre-existing, acknowledged) |
| 3d. Z_list_grad memory leak when off | **PASS** |
| 3e. eigh ordering (top-k vs bottom-k) | **PASS** |
| 4. SERIOUS-13 (rank_project_k) preserved | **PASS** |

## Overall: **GO**

All critical findings from the original attack (Attack #2 NaN at coincident σ, Attack #3 autocast wrap) are correctly addressed in production code. All 5 smoke tests pass. Independent attacks (4-coincident σ, bf16 path, eigh-ordering verification, 10-step micro-train) all NaN-free.

The only residual is a microscopic cosmetic point already documented in the original attack as DOES NOT LAND: `Z_list_grad` and the `L_rank` placeholder are allocated unconditionally during matrix-CODI training. This is invariant under default flags and does not affect numeric equivalence. Safe to launch.
