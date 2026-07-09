# Box smoke results — ATTRACTOR-ROBUSTNESS 2x2 deploy (2026-07-09, box UTC)

Box: youthful-indigo-turkey (`brev-ukptqsu65`), venv `/home/nvidia/tdenv`,
torch 2.12.1+cu130, triton 3.7.1, fla 0.5.1. Repo commit synced: `f09254a`.

## ENVIRONMENT FIX REQUIRED (recorded for reproducibility)

`lm_pretrain_rd.py --smoke` item [18] initially FAILED on the gated arm's backward:
fla 0.5.1 REFUSES the Triton path for gated `chunk_bwd_dqkwg` on Hopper with
Triton >= 3.4.0 (fla issue #640: "produces incorrect results") and prescribes the
tilelang backend. The box had no CUDA toolkit (driver-only), so tilelang's
nvcc-based JIT needed pip-provisioning. Installed into `/home/nvidia/tdenv`
(torch/triton versions untouched):

- `tilelang==0.1.12` (+ deps: apache-tvm-ffi, z3-solver, ml-dtypes, cloudpickle,
  psutil, torch-c-dlpack-ext)
- `nvidia-cuda-nvcc==13.3.73` (tilelang env.py "Guess #3" pip-nvcc path; +nvidia-cuda-crt, nvidia-nvvm)
- `nvidia-cuda-cccl==13.3.3.4.1` (libcu++ headers — fixed `fatal error: nv/target` in cuda_fp16.h)

Minimal repro then passed: bf16 chunk_gated_delta_rule fwd/bwd, head_dim=64, grads
finite (tilelang kernel JIT ~9 s first compile, cached after). The NON-gated
(default) path never touched tilelang — it stays on the exact same Triton kernels
as every prior chapter (baseline cell unaffected by the env change).

## 2x2 wave smokes

| Item | Result |
|---|---|
| `lm_pretrain_rd.py --smoke` (real CUDA+fla, all 18 items) | **PASS** (RC=0) — `attrrob_smoke_lm_pretrain.log` |
| item [18a] defaults bit-identical to no-kwargs construction | PASS |
| item [18b] full 2x2 fwd/bwd/grad-finite on REAL chunk_gated_delta_rule | PASS (all 4 cells; b_alpha_proj grad finite on gated cells) |
| item [18c] b_alpha_proj constructed only when gated | PASS |
| real-kernel envelope (bf16, head_dim>=32; d_state=64) | confirmed via [18] + minimal repro |
| `run_attractor_robustness_2x2.py --smoke` (7 items incl. budget-guard teeth) | **PASS** (RC=0) — `attrrob_smoke_runner.log` |

## h2h box smokes (arriving early, from cc89a4f now on box — NOT 2x2-blocking)

| Item | Result |
|---|---|
| `h2h_cell_train_rd.py --selftest` (REASONING_LINK_FORCE_CPU_STUB=1, designed mode) | **PASS** 15/15 (incl. new blank-out companions 12-15, dial items 9-11) — `h2h_selftest_box.log` |
| bare `--selftest` without the stub env var | FAILS as expected (fla RMSNorm has no CPU path — known rule; selftest device is hardcoded "cpu", stub mode is its documented invocation) |
| `probe_head_rd.py --smoke` (stub mode, 9 items) | **PASS** (RC=0) — `probe_head_smoke_box.log` |
| `h2h_box_smoke_checklist.py --list` (manifest self-validation) | **PASS** (RC=0) — `h2h_checklist_box.log` |
| `h2h_box_smoke_driver.py` (REAL kernels, CUDA_VISIBLE_DEVICES=0) | **PASS** — "ALL BLOCKING ITEMS PASSED" — `h2h_box_driver.log` |
| — item 1: lm_pretrain smoke real-kernel subprocess | PASS |
| — item 2: tap-changes-with-q on REAL nonzero S_T (|S|=579.7) | PASS |
| — item 3 (AUD-F1): contender aux-ONLY k/v/b grads nonzero real kernel + detach-negative | PASS |
| — item 4: measured_state_bytes == 32768 fp32 | PASS |
| — smoke_3 real branch (state context-only, q_last query-dependent) | PASS |
| — smoke_8c BOX branch + negative control (real Triton) | PASS |
| — gate 6: MATCH-GATE 2-pass real kernel (param 0.0073%, FLOP 2.78%) + corruption negative test | PASS |
| — gate 7: probe-capacity null, all 3 arms recovered_frac=0.0 < 0.05 | PASS |
| Gate tokens written | `results/h2h_rung1/gates/{GATE6_MATCH_GATE_PASSED,GATE7_PROBE_CAPACITY_NULL_PASSED,BOX_SMOKE_ITEMS_1_4_PASSED}.token` |

## Corpus verification

`openr1-mix-ext` resolves (lm_pretrain_rd.CORPUS_DIRS) to
`/data/deltanet_rd_data/reasoning_mix_eot_extended` — present, meta.json
vocab_size=50257 / tokenizer=gpt2 / eot_separated=true, train.pt 2.76 GB +
val.pt + doc_offsets. Cross-corpus pair `wikitext103_mix_eot_extended` also
present and valid (50257/gpt2/true).
