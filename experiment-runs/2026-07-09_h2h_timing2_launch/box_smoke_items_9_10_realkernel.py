"""Box-only real-kernel residual check for h2h_box_smoke_checklist.py items 9-10
(continuation_blankout_inplace, continuation_blankout_fresh_instance).

Throwaway verification harness, NOT part of the audited codebase (no logic
changes to any audited file) -- mirrors h2h_box_smoke_driver.py's own established
pattern for items 1-4: sets torch's default device to CUDA immediately, then
calls the SAME audited function (_recurrent_continuation_answer_logits) with the
SAME construction selftest 12/13 already use in h2h_cell_train_rd.py's
mode_selftest, so the real Triton/fla kernel path is exercised end to end
instead of the CPU-stub / CPU-generator-pinned path that mode_selftest's own
ProbeRig/T_val construction requires elsewhere (empirically confirmed: running
the FULL --selftest suite under a blanket CUDA default device crashes inside
random_unit_rows_init's CPU-pinned torch.Generator, a design choice unrelated
to items 9-10 -- this script routes around that by not touching ProbeRig/T_val
at all, exactly the narrow scope the checklist items describe: "no new
plumbing").

This is items 9-10's own documented BOX-ONLY residual: does the REAL Triton
kernel harbor any instance-level nondeterminism invisible to state_dict()?
CPU already proves the structural claim (no path from bind_tokens back to the
continuation call); this only re-confirms it holds when S_T is a REAL nonzero
kernel-produced state, not the CPU stub's zero constant.
"""
import sys

import torch

torch.set_default_device("cuda")

sys.path.insert(0, "/home/nvidia/chapter2/deltanet_rd")

from lm_pretrain_rd import DeltaNetLM, _MIN_KERNEL_T          # noqa: E402
from ablation_mixer_rd import AblationLM                       # noqa: E402
from h2h_cell_train_rd import _recurrent_continuation_answer_logits  # noqa: E402

assert torch.cuda.is_available()
print(f"device_default={torch.tensor([1.0]).device} cuda_device={torch.cuda.get_device_name(0)}")

failures = []


def rep(item, ok, detail=""):
    print(f"[{item}] {'PASS' if ok else 'FAIL'}{(' -- ' + detail) if detail else ''}", flush=True)
    if not ok:
        failures.append(item)


def make_model(arch):
    if arch == "contender":
        return DeltaNetLM(300, d_model=32, d_state=64, n_layers=1, conv_size=4)
    return AblationLM(300, d_model=32, d_state=16, n_layers=1, conv_size=4)


# ---- item 9: continuation_blankout_inplace, real kernel ----
ok9 = True
detail9 = {}
for arch in ("contender", "ablation"):
    torch.manual_seed(51)
    vocab = 300
    m = make_model(arch)
    bind_tokens = torch.randint(0, vocab, (2, _MIN_KERNEL_T))
    query_tokens = torch.randint(0, vocab, (2, 3, 6))
    with torch.no_grad():
        _, final_states = m(bind_tokens, return_states=True)
        s_abs = final_states[-1].abs().sum().item()
        logits_before = _recurrent_continuation_answer_logits(arch, m, final_states, query_tokens,
                                                               buffer_id=vocab - 1)
        bind_tokens.copy_(torch.randint_like(bind_tokens, 0, vocab))
        logits_after = _recurrent_continuation_answer_logits(arch, m, final_states, query_tokens,
                                                              buffer_id=vocab - 1)
    identical = torch.equal(logits_before, logits_after)
    detail9[arch] = {"S_T_abs_sum": s_abs, "identical": identical, "device": str(logits_before.device)}
    ok9 = ok9 and identical and s_abs > 0.0
    print(f"    arch={arch}: S_T_abs_sum={s_abs:.4f} identical={identical} device={logits_before.device}",
          flush=True)
rep("item 9 (continuation_blankout_inplace, REAL kernel, nonzero S_T)", ok9, str(detail9))

# ---- item 10: continuation_blankout_fresh_instance, real kernel ----
ok10 = True
detail10 = {}
for arch in ("contender", "ablation"):
    torch.manual_seed(52)
    vocab = 300
    m1 = make_model(arch)
    bind_tokens = torch.randint(0, vocab, (2, _MIN_KERNEL_T))
    query_tokens = torch.randint(0, vocab, (2, 3, 6))
    with torch.no_grad():
        _, final_states = m1(bind_tokens, return_states=True)
        s_abs = final_states[-1].abs().sum().item()
        logits1 = _recurrent_continuation_answer_logits(arch, m1, final_states, query_tokens,
                                                         buffer_id=vocab - 1)
        m2 = make_model(arch)
        m2.load_state_dict(m1.state_dict())
        logits2 = _recurrent_continuation_answer_logits(arch, m2, final_states, query_tokens,
                                                         buffer_id=vocab - 1)
    identical = torch.equal(logits1, logits2)
    distinct_objects = (m1 is not m2) and (id(m1) != id(m2))
    distinct_params = all(p1.data_ptr() != p2.data_ptr()
                          for p1, p2 in zip(m1.parameters(), m2.parameters()))
    detail10[arch] = {"S_T_abs_sum": s_abs, "identical": identical,
                       "distinct_objects": distinct_objects, "distinct_params": distinct_params}
    ok10 = ok10 and identical and distinct_objects and distinct_params and s_abs > 0.0
    print(f"    arch={arch}: S_T_abs_sum={s_abs:.4f} identical={identical} "
          f"distinct_objects={distinct_objects} distinct_params={distinct_params}", flush=True)
rep("item 10 (continuation_blankout_fresh_instance, REAL kernel, nonzero S_T)", ok10, str(detail10))

print("=" * 70)
if failures:
    print(f"BOX ITEMS 9-10: {len(failures)} FAILURE(S): {failures}", file=sys.stderr)
    sys.exit(1)
print("BOX ITEMS 9-10: ALL PASSED (real kernel)")
sys.exit(0)
