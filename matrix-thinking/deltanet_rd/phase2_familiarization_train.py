"""phase2_familiarization_train.py -- REASONING_LINK_DESIGN.md sec 16.2.1
(Rev 5, CLEARED-FOR-BUILD) Phase-2 familiarization trainer: continues
training ONE archived Leg-A frozen-bias checkpoint (arm in
{off,per_token,global} x corpus x seed) on a combined objective
`L_total = L_corpus + lambda_fam * L_query`, checkpointing at the
pre-registered trajectory steps {250,500,1000,2500,5000} (steps=5000
total), and -- for the OFF arm only -- computing the per-checkpoint
Stage-0.5-familiarized gate (premises (iii)/(iv) + h1 sanity floor,
re-measured fresh at every checkpoint on THAT checkpoint's own weights).

**What this reuses verbatim (DRY; zero reimplementation of already-audited
machinery):**
  - `lm_pretrain_rd.DeltaNetLM` / `load_corpus` / `get_batch` /
    `boundary_stats` / `eval_loss` / `corpus_fixed_seed` / `get_lr` /
    `should_checkpoint_now` / `load_init_checkpoint_strict` / `_dump` /
    `set_and_log_tf32` / `window_digest`.
  - `grammar_rd.DeltaNetRDTaskConfig` / `sample_batch_rd` (marker-template
    episode construction, hop-depth exclusion-by-sampling, K-cycle
    generation -- entirely unmodified).
  - `reasoning_link_probe.build_reasoning_link_pools` (sec 4.1's
    vocab-safe token substitution: real GPT-2 period token as BUFFER, a
    verified ordinary single-token marker as <Q> -- NEVER grammar_rd.py's
    own reserved out-of-vocab ids, which would be an out-of-range
    embedding lookup against these standard-50,257-row checkpoints),
    `measure_cell_all_h` (the Stage-0.5 gate's own premises/h1-floor
    computation), `frozen_bias_surgery`, `readout_layer_index`.

**What this does NOT reuse:** `lm_pretrain_rd.train()`'s own per-step body
(that function's geo3/hard_select/rank-stat instrumentation does not apply
to familiarization, and its per-step loss is a SINGLE corpus CE term --
Phase-2's own combined two-forward-pass objective is a genuinely different
training loop, built here rather than bolted onto that already-large
function).

**Query loss -- pinned exactly as REASONING_LINK_DESIGN.md sec 16.2.1
registers it (its own 'Corrected definition' paragraph):** per step, ONE
episode batch is drawn (`hop_set=(1,2)`, `n_query=2`, marker template,
TRAINING entity pool -- `use_heldout_entities=False`), bind+query tokens
are concatenated into `concat_tokens = torch.cat([bind_expanded,
query_flat], dim=1)` (the exact `main_concat` construction sec 4.4/this
design's own already-Stage -1-verified causal-safety proof covers), and
scored via the REAL `DeltaNetLM.forward(concat_tokens, initial_states=None)`
class method (full vocab logits, never the probe's VRAM-narrowed
`forward_body` helper) at the concatenated sequence's OWN LAST position
(`logits[:, -1, :]`) against `target_ids = torch.gather(entity_ids, 1,
tgt_slot)` -- ONE forward call, never a two-phase bind-only-then-
continuation-over-query-window call (that pattern is EXPLICITLY REJECTED
by the design, sec 16.2.1: `ShortConvolution`'s own implicit zero-padding
at a cache-less call's left edge would corrupt the query window's first
`conv_size-1` positions). `L_total = L_corpus + lambda_fam * L_query`
(`lambda_fam=1.0` pinned BUILD DEFAULT), summed into ONE scalar before ONE
`.backward()` call -- two separate forward+CE passes, never four.

**Held-out eval pool (sec 16.2.1's own still-open MINOR-1 item -- MAJOR-1
fix, Phase-2 build-audit round: actually wired, not merely claimed).**
familiarization TRAINING episodes draw entities from `pools.train_name_ids`
(`use_heldout_entities=False`); every EVAL-purpose episode batch this
script draws (the Stage-0.5 gate's own premise/null-shuffle batch, via
`compute_stage05_gate`) draws from `pools.heldout_name_ids`
(`use_heldout_entities=True`) instead -- `grammar_rd.py`'s own C17
disjoint-pool mechanism, applied here for the first time to separate
TRAIN-vs-EVAL entities rather than merely zero-shot-vs-trained-pool. **Build
audit correction (this fix):** the ORIGINAL build of this module made this
claim while `reasoning_link_probe.measure_cell_all_h` had no
`use_heldout_entities` parameter at all -- `compute_stage05_gate`'s own
`use_heldout_entities=True` intent was silently dropped on the floor, and
EVERY eval-purpose episode (both the Stage-0.5 gate here AND the
killer-prediction readout in `phase2_trajectory_analysis.
killer_prediction_readout`, which reuses `reasoning_link_probe.run_cell`)
drew from `pools.train_name_ids` instead. Fixed: `measure_cell_all_h` (and
`run_cell`) now carry an ADDITIVE `use_heldout_entities` parameter, set
True at both Phase-2 eval call sites. Cycle-level disjointness is
automatic: eval draws use an eval-only seed stream, never the training
seed's own stream.

**Checkpoint saving includes `optimizer_state_dict`** ([LEARN], rung-3:
"checkpoints without optimizer state block resume forever") -- every
trajectory checkpoint this script writes carries
`model_state_dict`/`optimizer_state_dict`/`step`/`config`/`corpus`/`seed`/
`arm`/`run_name`, so a crashed cell can resume from its own last written
checkpoint (see `--resume` below) without losing optimizer momentum a
second time (stacked on TOP of the already-disclosed, ACCEPTED
optimizer-restart confound at familiarization's own START, sec 16.2.1 --
resuming mid-familiarization must not compound that with a SECOND silent
restart).

Run (Stage -1 self-tests exercise this on tiny synthetic corpora/models via
the CPU fla-stub -- see `phase2_stage_minus1.py`):
    python phase2_familiarization_train.py --init-checkpoint <path> \\
        --arm off --corpus openr1-mix-ext --seed 0 --k 32 \\
        --steps 5000 --ckpt-dir results/phase2/ckpts --out results/phase2/off_openr1_s0_k32.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reasoning_link_probe as rlp   # noqa: E402  (installs the CPU fla stub as an import side
                                      # effect when the real `fla` package is not importable -- see
                                      # that module's own docstring; MUST precede the lm_pretrain_rd
                                      # import below, which executes `from fla...import` at its own
                                      # module-import time)
import grammar_rd  # noqa: E402
from lm_pretrain_rd import (  # noqa: E402
    DeltaNetLM, FROZEN_BIAS_ARM_MODES, CORPUS_DIRS, OTHER_CORPUS, DEFAULT_DATA_DIR,
    load_corpus, get_batch, boundary_stats, eval_loss, corpus_fixed_seed, get_lr,
    should_checkpoint_now, load_init_checkpoint_strict, _dump, set_and_log_tf32, window_digest,
)

# ---------------------------------------------------------------------------
# Registered constants (sec 16.2.1).
# ---------------------------------------------------------------------------

CKPT_STEPS = (250, 500, 1000, 2500, 5000)     # sec 16.2.1's trajectory readout schedule (MAJOR-1)
STEPS_DEFAULT = 5000                           # sec 16.2.1: 5,000 is the FIXED ceiling, not an estimate
LAMBDA_FAM_DEFAULT = 1.0                       # sec 16.2.1: pinned BUILD DEFAULT (no-thumb-on-the-scale)
H_TRAIN = (1, 2)                               # sec 16.2.1 FATAL-1: familiarization trains ONLY h in {1,2}
H_TEST_HELD_OUT = (3, 4)                       # sec 16.2.1: h in {3,4} is THE H_LINK test, never trained on
N_QUERY = 2                                    # sec 16.2.1 MAJOR-R3-1b: pinned query count per episode
K_SWEEP = (20, 32)                             # sec 16.2.1 MAJOR-2: Leg A's own committed K pair -- an
                                                # EVAL/READOUT-time axis only (see K_TRAIN_DEFAULT below).

# K is NOT a training-cell axis (build-time correction, caught by this build's own cross-check
# against sec 16.2.3's own cost arithmetic: "18 cells x 1,000-5,000 steps x 0.04544 s/step" prices
# the FULL 18-cell grid with NO x2 multiplier for K anywhere in the training-cost lines -- only the
# SEPARATE "5 checkpoints x 18 cells x 2 K's x Option 1/Option 2" READOUT line carries a x2-for-K
# factor, and that line is explicitly forward-only/no-backward, i.e. eval, never training. The
# registered "18 core cells (3 arms x 2 corpora x 3 seeds)" identity tuple has NO K entry anywhere
# it is stated (sec 16.2.1's own Base-checkpoints paragraph, sec 16.2.3's own cost paragraph, the
# costs table). K therefore enters ONLY at readout time (phase2_trajectory_analysis.py's own
# killer-prediction re-application, `reasoning_link_probe.run_cell`'s own `K` parameter, which
# purely controls EVAL episode construction and is independent of what K a checkpoint trained
# under) -- training itself uses ONE FIXED K to construct its own familiarization episodes
# (K_TRAIN_DEFAULT below), and a cell's run_name/checkpoint identity carries NO _k{K} suffix.
K_TRAIN_DEFAULT = 32   # build-time choice (disclosed): matches d=64's own near-cliff committed K and
                        # this program's own Stage-0/run_stage0 K=32 default convention -- training
                        # familiarization episodes at ANY single fixed K exposes the model to the
                        # BIND/QUERY task; the K-DEPENDENT killer-prediction contrast is read out
                        # separately, post-hoc, at BOTH K=20 and K=32 on the SAME trained weights.

# Phase-2-local deterministic seed allocation -- DISJOINT BY CONSTRUCTION from
# reasoning_link_probe.episode_seed's own registered LEG_BASE (leg_a=0, leg_b=5_000_000): this base
# sits far outside either range, so no Phase-1 seed combination can ever collide with a Phase-2 one.
# This is NOT a request to extend that formula's own registered PURPOSE_BASE/LEG_BASE vocabulary
# (untouched) -- Phase-2 gets its OWN small, disjoint formula, collision-freedom verified by
# phase2_stage_minus1.py mirroring item 10's own enumerate-and-assert-no-collision convention.
PHASE2_SEED_BASE = 777_000_000
_ARM_INDEX = {"off": 0, "per_token": 1, "global": 2}
_CORPUS_INDEX = {"openr1-mix-ext": 0, "wikitext-mix-ext": 1, "openr1": 0, "wikitext": 1}
_KIND_OFFSET = {"train_corpus": 0, "train_episode": 1, "eval_val": 2, "eval_gate_null": 3,
                "eval_gate_self": 4, "eval_killer": 5,
                # sec 16.16.3's own two new kinds (Phase-2b vocab-space behavioral contrast):
                # zero new digit-width arithmetic -- kind is the outermost/last digit, no _MAX_*
                # constant depends on how many kinds exist (phase2_seed's own formula, unchanged).
                "eval_lquery_heldout": 6, "eval_lquery_ood": 7}


def wrap_phase2(payload: dict, stage: str) -> dict:
    """Phase-2-scoped mirror of `reasoning_link_probe.wrap_reasoning_link` --
    a SEPARATE, Phase-2-own wrapper rather than extending that function's
    own closed, Phase-1-scoped stage vocabulary (`"program": "REASONING-
    LINK Phase 1"`, a fixed assert-enumerated `stage` set) -- reusing it
    here would either fail its own assert or silently mislabel a Phase-2
    artifact as a Phase-1 one."""
    assert isinstance(payload, dict)
    wrapped = {"design_ref": "REASONING_LINK_DESIGN.md sec 16.2 (Rev 5, CLEARED-FOR-BUILD)",
               "program": "REASONING-LINK Phase 2 (task familiarization)", "stage": stage,
               "fla_stub_installed": rlp.FLA_STUB_INSTALLED, "timestamp": time.time()}
    wrapped.update(payload)
    return wrapped


# TRUE mixed-radix digit widths (sec 4.6-style "collision-free by construction" convention, but
# each digit's WIDTH here is set to strictly EXCEED the max value the next-inner digit's own term
# can reach, verified directly rather than eyeballed -- see phase2_stage_minus1.py item 9's own
# exhaustive-enumeration proof, which caught a real overflow bug in an earlier additive-only draft
# of this formula: checkpoint_step (up to 5,000) silently overflowed into ckpt_seed's own digit
# when both were simply ADDED with an insufficient stride).
_MAX_CHECKPOINT = 5_000     # sec 16.2.1's own trajectory ceiling; checkpoint_step in [0, 5000]
_MAX_K = 512                # generous headroom above every registered K (20, 32, up to d_state=128)
_MAX_CKPT_SEED = 12         # 10 -> 12 (sec 16.19.5 item 3(b), Phase-2b seed extension, seeds {0..11}).
                             # DISCLOSED consequence (checked, not assumed): _WIDTH_CKPT_SEED ==
                             # _MAX_CKPT_SEED, so every digit ABOVE ckpt_seed (corpus/arm/kind)
                             # changes its stride -- the returned seed value changes for EVERY
                             # corpus_idx>=1 / arm_idx>=1 / kind_idx>=1 combination, INCLUDING the 3
                             # already-archived seeds' own EVAL kinds. The design's protection is
                             # NON-INVOCATION, not bit-identity: archived seeds 0-2 are NEVER
                             # re-scored live this wave (sec 16.19.5 item 5's archived-values loader
                             # + phase2_trajectory_analysis.install_seedext_live_eval_guard's
                             # whole-harvest-runtime ckpt_seed>=3 assert); old-seed bit-identity is
                             # deliberately NOT claimed anywhere. Collision-freedom at the widened
                             # width re-proven exhaustively (phase2b_seedext_stage_minus1.py).
_MAX_CORPUS_IDX = 10        # registered grid uses 0-1; headroom via _CORPUS_INDEX's own .get default
_MAX_ARM = 3                # off/per_token/global
_WIDTH_CHECKPOINT = _MAX_CHECKPOINT + 1
_WIDTH_K = _MAX_K
_WIDTH_CKPT_SEED = _MAX_CKPT_SEED
_WIDTH_CORPUS = _MAX_CORPUS_IDX
_WIDTH_ARM = _MAX_ARM


def phase2_seed(kind: str, arm: str, corpus: str, ckpt_seed: int, k: int, checkpoint_step: int = 0) -> int:
    """TRUE mixed-radix seed: `value = checkpoint_step + k*W1 + ckpt_seed*W1*W2 + corpus_idx*W1*W2*W3
    + arm_idx*W1*W2*W3*W4 + kind_idx*W1*W2*W3*W4*W5`, each `Wn` set to strictly exceed the previous
    digit's own maximum -- collision-free BY CONSTRUCTION (a positional numeral system), not by
    hoping the additive terms never happen to coincide. Disjoint from reasoning_link_probe.
    episode_seed's own registered LEG_BASE (leg_a=0, leg_b=5_000_000) via `PHASE2_SEED_BASE`, which
    sits far outside either range."""
    assert kind in _KIND_OFFSET, f"unknown kind {kind!r}"
    assert arm in _ARM_INDEX, f"unknown arm {arm!r}"
    assert 0 <= checkpoint_step <= _MAX_CHECKPOINT, f"checkpoint_step={checkpoint_step} out of range"
    assert 0 <= k < _MAX_K, f"k={k} out of range"
    assert 0 <= ckpt_seed < _MAX_CKPT_SEED, f"ckpt_seed={ckpt_seed} out of range"
    corpus_idx = _CORPUS_INDEX.get(corpus, _MAX_CORPUS_IDX - 1)  # last slot = visible sentinel, never silent
    kind_idx, arm_idx = _KIND_OFFSET[kind], _ARM_INDEX[arm]

    value = checkpoint_step
    stride = _WIDTH_CHECKPOINT
    value += k * stride
    stride *= _WIDTH_K
    value += ckpt_seed * stride
    stride *= _WIDTH_CKPT_SEED
    value += corpus_idx * stride
    stride *= _WIDTH_CORPUS
    value += arm_idx * stride
    stride *= _WIDTH_ARM
    value += kind_idx * stride
    return PHASE2_SEED_BASE + value


# ---------------------------------------------------------------------------
# Episode config + periodicity guard re-verification (sec 16.2.1's own registered Stage -1
# requirement: "The periodicity guard must therefore be RE-VERIFIED for real ... this
# re-verification is a mandatory Stage -1 assertion"). Constructing DeltaNetRDTaskConfig with
# H_train=(1,2)/H_test=(3,4) ALREADY exercises grammar_rd.py's own __post_init__ guard live (it
# raises AssertionError on any residue collision) -- this function exists so callers never build
# the config a different, unguarded way.
# ---------------------------------------------------------------------------

def familiarization_episode_config(conv_size: int, K: int) -> grammar_rd.DeltaNetRDTaskConfig:
    return grammar_rd.DeltaNetRDTaskConfig(K=K, conv_size=conv_size, n_query=N_QUERY,
                                            H_train=H_TRAIN, H_test=H_TEST_HELD_OUT, H_extra=())


def familiarization_gate_episode_config(conv_size: int, K: int) -> grammar_rd.DeltaNetRDTaskConfig:
    """MAJOR-2 fix (Phase-2 build-audit round, sec 16.2.1 SS16.2.1): the
    Stage-0.5 gate's own EVAL episodes use Q=K (`n_query=None`), mirroring
    `reasoning_link_probe.episode_config_for_checkpoint`'s own Q=K eval
    convention -- the SAME convention the killer-prediction trajectory
    readout already uses (sec 16.2.1's own "byproduct of the scoring pass"
    citation for `run_cell`'s `compute_premises=True`) -- rather than
    reusing TRAINING's own fixed `n_query=N_QUERY=2` config
    (`familiarization_episode_config` above), whose `B*Q` (gate_batch_size
    x N_QUERY=16*2=32) is a 16x power reduction vs `Q=K=32`'s own
    `B*Q=512` that this gate, sec 16.2.1's own most stringent
    readout-soundness test, is registered to run at. As-built (pre-fix)
    `compute_stage05_gate` received the SAME `episode_cfg` object
    `run_familiarization_cell` builds for training (n_query=2) and passed
    it straight through to `measure_cell_all_h` -- this function exists so
    the gate constructs its OWN, separate, Q=K episode config instead."""
    return grammar_rd.DeltaNetRDTaskConfig(K=K, conv_size=conv_size, n_query=None,
                                            H_train=H_TRAIN, H_test=H_TEST_HELD_OUT, H_extra=())


# ---------------------------------------------------------------------------
# Query-loss forward (sec 16.2.1's "Corrected definition" paragraph, implemented verbatim).
# ---------------------------------------------------------------------------

def query_loss_forward(model: DeltaNetLM, episode_cfg: grammar_rd.DeltaNetRDTaskConfig,
                        pools: grammar_rd.EntityPools, batch_size: int, gen: torch.Generator,
                        device: str, use_heldout_entities: bool = False, step: int | None = None,
                        hop_set: tuple = H_TRAIN):
    """ONE forward call over the concatenated bind+query sequence (sec
    16.2.1: NEVER a two-phase continuation -- see this module's own
    docstring for why). Returns (L_query, batch, n_forward_calls=1) --
    `batch` is returned so a caller can also read hops/tgt_slot/entity_ids
    for diagnostics without a second draw.

    `hop_set` (sec 16.16.3's own registered parameterization -- Rev 0 had
    this HARDCODED to H_TRAIN at this call site; a real, small, disclosed
    build task, not a design assumption): additive, defaults to H_TRAIN so
    every pre-existing caller (the familiarization training loop itself,
    phase2_smoke_gpu.py, phase2_stage_minus1.py's item 4) is byte-identical,
    unaffected. Phase-2b's own eval_query_loss_heldout (phase2_trajectory_
    analysis.py) is the first caller to pass hop_set=(3,4) (the secondary,
    held-out-hop OOD readout, sec 16.16.7) alongside hop_set=(1,2) (primary).
    `sample_batch_rd`'s own `hop_set` argument governs ONLY which hops this
    ONE batch draws from -- independent of `episode_cfg.H_train`/`H_test`,
    which remain fixed at (1,2)/(3,4) for the periodicity guard regardless
    of which hop_set a given call requests (verified directly against
    grammar_rd.sample_batch_rd's own body: no cross-check against cfg.H_train/
    H_test exists there, by design -- the guard lives in DeltaNetRDTaskConfig.
    __post_init__ instead, already re-verified live by Stage -1 item 3)."""
    batch = grammar_rd.sample_batch_rd(episode_cfg, batch_size, gen, hop_set=hop_set, pools=pools,
                                        device=device, use_heldout_entities=use_heldout_entities)
    B, Q = batch["hops"].shape
    T_bind = batch["token_ids"].shape[1]
    query_len = batch["query_tokens"].shape[-1]
    bind_expanded = batch["token_ids"].unsqueeze(1).expand(B, Q, T_bind).reshape(B * Q, T_bind)
    query_flat = batch["query_tokens"].reshape(B * Q, query_len)
    concat_tokens = torch.cat([bind_expanded, query_flat], dim=1)          # main_concat, sec 4.4/16.2.1
    logits = model(concat_tokens, initial_states=None, step=step)          # REAL DeltaNetLM.forward,
                                                                             # full vocab logits, ONE call
    logits_at_q = logits[:, -1, :]                                         # the <Q>-marker position
    # grammar_rd.py's own [grammar 3c] self-test pattern (L594-616): the answer entity's OWN token.
    target_ids = torch.gather(batch["entity_ids"], 1, batch["tgt_slot"]).reshape(B * Q)
    L_query = F.cross_entropy(logits_at_q, target_ids)
    return L_query, batch, 1


# ---------------------------------------------------------------------------
# Per-checkpoint Stage-0.5-familiarized gate (sec 16.2.1 MAJOR-4/MAJOR-R3-3, OFF arm only).
# ---------------------------------------------------------------------------

def compute_stage05_gate(model: DeltaNetLM, episode_cfg: grammar_rd.DeltaNetRDTaskConfig,
                          pools: grammar_rd.EntityPools, arm: str, corpus: str, ckpt_seed: int,
                          checkpoint_step: int, batch_size: int, device: str) -> dict:
    """Re-measures premises (iii)/(iv) and the h1 sanity floor on the
    FAMILIARIZED OFF-arm checkpoint's own weights, AT this checkpoint --
    sec 16.2.1's own per-checkpoint gate. Reuses `reasoning_link_probe.
    measure_cell_all_h` verbatim (compute_premises=True, null_seed set --
    the SAME shared computation `run_stage0` already runs) rather than a
    parallel reimplementation.

    `episode_cfg` here is the CALLER's own TRAINING config (n_query=2) --
    used ONLY to read `conv_size`/`K`, never passed to `measure_cell_all_h`
    directly (MAJOR-2 fix, Phase-2 build-audit round): this gate builds its
    OWN, separate `familiarization_gate_episode_config` (Q=K, n_query=None)
    below, matching `reasoning_link_probe.episode_config_for_checkpoint`'s
    own Q=K eval convention rather than training's own fixed-Q=2 config
    (as-built, pre-fix, this function silently reused the training
    `episode_cfg` verbatim -- a 16x power reduction, `gate_batch_size *
    N_QUERY=16*2=32` vs the registered `Q=K=32`'s own `B*Q=512`, on this,
    sec 16.2.1's own most stringent readout-soundness test).

    Held-out entity pool (MAJOR-1 fix, Phase-2 build-audit round; sec
    16.2.1's own still-open eval-disjointness item): `use_heldout_entities=
    True` is now actually threaded through to `measure_cell_all_h` (as-built,
    pre-fix, this module's own docstring CLAIMED this was already resolved,
    but `measure_cell_all_h` had no such parameter at all, so every EVAL
    episode this gate drew silently came from `pools.train_name_ids`, the
    SAME pool familiarization trained on)."""
    assert arm == "off", "the Stage-0.5-familiarized gate is registered OFF-arm-only (sec 16.2.1)"
    readout_layer = rlp.readout_layer_index(model)
    gate_episode_cfg = familiarization_gate_episode_config(episode_cfg.conv_size, episode_cfg.K)
    eval_seed = phase2_seed("eval_gate_self", arm, corpus, ckpt_seed, gate_episode_cfg.K, checkpoint_step)
    null_seed = phase2_seed("eval_gate_null", arm, corpus, ckpt_seed, gate_episode_cfg.K, checkpoint_step)
    per_h, forward_counts = rlp.measure_cell_all_h(
        model, gate_episode_cfg, pools, readout_layer, gate_episode_cfg.K, hops=(1,), batch_size=batch_size,
        seed=eval_seed, surgery="native", device=device, compute_option2=False, compute_premises=True,
        null_seed=null_seed, use_heldout_entities=True)
    # native surgery is correct here (not "off"): sec 16.2.1's "What continues training, what stays
    # frozen" paragraph -- the OFF arm never had a frozen-bias table to begin with (arm='off' is
    # mechanically identical to blend-off, sec 5.2a), so surgery is a no-op for this arm specifically.
    per_h1 = per_h[1]
    gate_pass = bool(per_h1["premise_iii_pass"] and per_h1["premise_iv_pass"] and per_h1["probe_valid"])
    return wrap_phase2({
        "checkpoint_step": checkpoint_step, "arm": arm, "corpus": corpus, "ckpt_seed": ckpt_seed,
        "K": gate_episode_cfg.K, "gate_k": gate_episode_cfg.K, "gate_q": gate_episode_cfg.queries,
        "gate_pass": gate_pass, "per_h": {"1": per_h1}, "forward_counts": forward_counts,
    }, stage="stage05-gate")


# ---------------------------------------------------------------------------
# Resume support.
# ---------------------------------------------------------------------------

def find_latest_checkpoint(ckpt_dir: str, run_name: str) -> str | None:
    """Resume-safety (house convention, mech_stage2_chain.sh's own '[LEARN]
    from Stage 1: run 1 died on an unshipped dependency' + this design's
    own optimizer_state_dict requirement): if a partial run already wrote
    trajectory checkpoints, resume from the LATEST one (highest step)
    rather than restarting at step 1 and losing BOTH training progress and
    (absent optimizer_state_dict) optimizer momentum a second time."""
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None
    import glob
    import re
    matches = glob.glob(os.path.join(ckpt_dir, f"{run_name}_step*.pt"))
    if not matches:
        return None
    def _step_of(p):
        m = re.search(r"_step(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1
    return max(matches, key=_step_of)


# ---------------------------------------------------------------------------
# The training loop.
# ---------------------------------------------------------------------------

def run_familiarization_cell(init_checkpoint: str, arm: str, corpus: str, ckpt_seed: int,
                              K: int = K_TRAIN_DEFAULT, steps: int = STEPS_DEFAULT, ckpt_steps=CKPT_STEPS,
                              lambda_fam: float = LAMBDA_FAM_DEFAULT, lr: float = 3e-4,
                              weight_decay: float = 0.01, warmup_steps: int = 100,
                              corpus_batch_size: int = 32, episode_batch_size: int = 16,
                              gate_batch_size: int = 16, seq_len: int = 512,
                              eval_batches: int = 4, eval_batch_size: int = 8,
                              data_dir: str = DEFAULT_DATA_DIR, device: str = "cpu",
                              ckpt_dir: str | None = None, out_path: str | None = None,
                              resume: bool = True) -> dict:
    """Runs ONE (arm, corpus, ckpt_seed) familiarization cell -- one of the
    registered 18 core cells (3 arms x 2 corpora x 3 seeds; sec 16.2.1's
    own Base-checkpoints paragraph; K is NOT part of this identity, see
    K_TRAIN_DEFAULT's own module-level comment above) -- to `steps` (5,000
    by default), checkpointing at `ckpt_steps`. `K` here is the SINGLE,
    FIXED K used to construct THIS cell's own familiarization training
    episodes (default K_TRAIN_DEFAULT=32); the registered K in {20,32}
    READOUT sweep is applied separately, post-hoc, by
    `phase2_trajectory_analysis.py`'s own killer-prediction re-application
    on this SAME cell's saved checkpoints. Returns the assembled result
    dict (trajectory + checkpoints, same top-level-field convention
    lm_pretrain_rd.py's own `_assemble_result` uses -- 'checkpoints', never
    'trajectory', is where per-checkpoint val_loss/gate live)."""
    t0 = time.time()
    other_corpus = OTHER_CORPUS[corpus]
    train_tokens, val_same, meta_same, train_offs, val_offs_same = load_corpus(data_dir, corpus, device)
    _, val_other, meta_other, _, val_offs_other = load_corpus(data_dir, other_corpus, device)

    run_name = f"phase2fam_{arm}_{corpus}_s{ckpt_seed}"   # NO _k{K} suffix -- K is a training-episode
                                                            # hyperparameter, not part of this cell's
                                                            # own identity (see K_TRAIN_DEFAULT above)

    resumed_from = None
    start_step = 0
    if resume:
        latest = find_latest_checkpoint(ckpt_dir, run_name)
        if latest is not None:
            resumed_from = latest

    init_config = torch.load(init_checkpoint, map_location=device)["config"]
    # sec 16.16.9 Stage -1 item (c) (MINOR-3, attack-round-1 on sec 16.16): the model's ACTUAL blend
    # behavior is governed ENTIRELY by this checkpoint's own baked frozen_bias_arm config field
    # (DeltaNetLM.config(), lm_pretrain_rd.py L1174) -- the CLI --arm flag below is used only for
    # run_name/seeding/bookkeeping and could silently disagree with it if a wrong --init-checkpoint
    # path were ever passed (every phase2b_chain.sh launch already pairs the right checkpoint with
    # the right --arm by construction, per its own init_ckpt path templating -- this assertion makes
    # that pairing VERIFIED, not merely trusted).
    assert init_config["frozen_bias_arm"] == arm, (
        f"--arm={arm!r} does not match the init checkpoint's own baked frozen_bias_arm="
        f"{init_config['frozen_bias_arm']!r} ({init_checkpoint}) -- refusing to silently train with a "
        f"mismatched arm/checkpoint pairing (sec 16.16.9 MINOR-3)")
    model = DeltaNetLM(**init_config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if resumed_from is not None:
        ck = torch.load(resumed_from, map_location=device)
        model.load_state_dict(ck["model_state_dict"], strict=True)
        assert "optimizer_state_dict" in ck, (
            f"{resumed_from} has no optimizer_state_dict -- cannot resume without losing momentum "
            f"a second time (sec 16.2.1's own registered requirement); refusing to silently restart")
        opt.load_state_dict(ck["optimizer_state_dict"])
        start_step = ck["step"]
        print(f"RESUMED from {resumed_from} at step={start_step}", flush=True)
    else:
        init_ckpt = load_init_checkpoint_strict(model, init_checkpoint, device)
        print(f"loaded init checkpoint {init_checkpoint} (archived step={init_ckpt['step']}) -- "
              f"fresh Adam moments + fresh LR schedule from step 1 (sec 16.2.1's disclosed "
              f"optimizer-restart confound)", flush=True)

    conv_size = model.conv_size
    episode_cfg = familiarization_episode_config(conv_size, K)   # re-verifies the periodicity guard LIVE
    assert episode_cfg.H_train == H_TRAIN and episode_cfg.H_test == H_TEST_HELD_OUT

    tokenizer = grammar_rd.load_gpt2_tokenizer()
    pools, pool_report = rlp.build_reasoning_link_pools(tokenizer=tokenizer, seed=0)  # marker template
                                                                                        # (sec 16.8.3 fallback pin)
    assert pools.vocab_size_total == model.vocab_size, (
        f"pools.vocab_size_total={pools.vocab_size_total} != model.vocab_size={model.vocab_size} -- "
        f"sec 4.1's own 'no architecture change, no embedding-table resize' guarantee is violated")

    model.train()
    # MINOR-1 fix (Phase-2 build-audit round): seed the training RNG streams with `start_step` (0 on
    # a fresh run -- byte-identical to pre-fix behavior; the resumed step on a REAL resume) rather
    # than always `checkpoint_step=0`. As-built (pre-fix), a resumed run re-seeded these generators
    # with the SAME stream a fresh run uses, so the FIRST post-resume draw replayed the exact same
    # batch the pre-crash run already consumed at its own step 1 -- `phase2_seed`'s own mixed-radix
    # formula (each `checkpoint_step` a fully independent digit, not a continuation) makes
    # `checkpoint_step=start_step` a cheap, sufficient fix: a differently-seeded stream, disjoint by
    # construction from the `checkpoint_step=0` stream the pre-crash run already consumed from.
    gen_corpus = torch.Generator(device=device).manual_seed(phase2_seed("train_corpus", arm, corpus,
                                                                          ckpt_seed, K, start_step))
    gen_episode = torch.Generator(device=device).manual_seed(phase2_seed("train_episode", arm, corpus,
                                                                           ckpt_seed, K, start_step))

    trajectory, checkpoints = [], []
    ckpt_step_set = set(ckpt_steps)
    for step in range(start_step + 1, steps + 1):
        cur_lr = get_lr(step, lr, warmup_steps, steps)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        x, y = get_batch(train_tokens, corpus_batch_size, seq_len, gen_corpus)
        logits_corpus = model(x, step=step)
        L_corpus = F.cross_entropy(logits_corpus.reshape(-1, logits_corpus.shape[-1]), y.reshape(-1))

        L_query, ep_batch, _ = query_loss_forward(model, episode_cfg, pools, episode_batch_size,
                                                    gen_episode, device, use_heldout_entities=False,
                                                    step=step)

        L_total = L_corpus + lambda_fam * L_query
        opt.zero_grad()
        L_total.backward()
        finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if step % 50 == 0 or step == 1:
            trajectory.append({"step": step, "loss_corpus": L_corpus.item(), "loss_query": L_query.item(),
                                "loss_total": L_total.item(), "lr": cur_lr, "grad_finite": finite})

        if should_checkpoint_now(step, steps, ckpt_every=10 ** 9, ckpt_steps=ckpt_step_set):
            eval_gen_same = torch.Generator(device=device).manual_seed(
                phase2_seed("eval_val", arm, corpus, ckpt_seed, K, step))
            eval_gen_other = torch.Generator(device=device).manual_seed(
                phase2_seed("eval_val", arm, other_corpus, ckpt_seed, K, step))
            val_loss_same, eval_info_same = eval_loss(model, val_same, val_offs_same, eval_batches,
                                                        eval_batch_size, seq_len, eval_gen_same, step=step)
            val_loss_other, eval_info_other = eval_loss(model, val_other, val_offs_other, eval_batches,
                                                           eval_batch_size, seq_len, eval_gen_other, step=step)

            ckpt_path = None
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"{run_name}_step{step}.pt")
                torch.save({"step": step, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(),   # obligation (10) / [LEARN] rung-3
                            "config": model.config(), "corpus": corpus, "seed": ckpt_seed, "arm": arm,
                            "run_name": run_name}, ckpt_path)

            gate_result = None
            if arm == "off":
                gate_result = compute_stage05_gate(model, episode_cfg, pools, arm, corpus, ckpt_seed,
                                                     step, gate_batch_size, device)
                if ckpt_dir:
                    gate_path = os.path.join(ckpt_dir, f"stage05_gate_{run_name}_step{step}.json")
                    _dump(gate_path, gate_result)

            res = {
                "step": step, "val_loss": {corpus: val_loss_same, other_corpus: val_loss_other},
                "eval_windows": {corpus: eval_info_same, other_corpus: eval_info_other},
                "checkpoint_path": ckpt_path,
                "stage05_gate": ({"gate_pass": gate_result["gate_pass"],
                                   "gate_json_path": (os.path.join(ckpt_dir, f"stage05_gate_{run_name}_step{step}.json")
                                                       if ckpt_dir else None)}
                                  if gate_result is not None else None),
            }
            checkpoints.append(res)
            model.train()
            print(f"  [checkpoint step {step}] val_loss[{corpus}]={val_loss_same:.4f} "
                  f"val_loss[{other_corpus}]={val_loss_other:.4f} "
                  f"L_corpus={L_corpus.item():.4f} L_query={L_query.item():.4f}"
                  + (f" stage05_gate_pass={gate_result['gate_pass']}" if gate_result is not None else ""),
                  flush=True)

    result = {
        "run_name": run_name, "arm": arm, "corpus": corpus, "other_corpus": other_corpus,
        "seed": ckpt_seed, "K": K, "conv_size": conv_size, "n_query": N_QUERY,
        "h_train": list(H_TRAIN), "h_test_held_out": list(H_TEST_HELD_OUT),
        "lambda_fam": lambda_fam, "steps": steps, "ckpt_steps": sorted(ckpt_step_set),
        "steps_completed": steps, "resumed_from": resumed_from,
        "init_checkpoint": init_checkpoint, "pool_report_marker": pool_report["marker_word"],
        "trajectory": trajectory, "checkpoints": checkpoints, "wall_s": time.time() - t0,
        "claim_tier": "REASONING_LINK_DESIGN.md sec 16.2.1 (Rev 5) task-familiarization trainer",
    }
    _dump(out_path, result)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init-checkpoint", type=str, required=True,
                     help="archived Leg-A frozen-bias step-20,000 checkpoint's .pt path")
    ap.add_argument("--arm", choices=list(FROZEN_BIAS_ARM_MODES), required=True)
    ap.add_argument("--corpus", choices=sorted(CORPUS_DIRS), required=True)
    ap.add_argument("--seed", type=int, required=True, help="the checkpoint's own training seed (ckpt_seed)")
    ap.add_argument("--k", type=int, default=K_TRAIN_DEFAULT,
                     help="the SINGLE, FIXED K used to construct this cell's own familiarization "
                          "training episodes (default 32) -- NOT the registered K in {20,32} READOUT "
                          "sweep, which phase2_trajectory_analysis.py applies separately, post-hoc, "
                          "to this cell's saved checkpoints. This cell's identity (run_name, "
                          "checkpoint filenames) does NOT include K -- see K_TRAIN_DEFAULT's own "
                          "module-level comment for why (sec 16.2.3's own cost arithmetic prices "
                          "18 cells, not 36).")
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--ckpt-steps", type=int, nargs="+", default=list(CKPT_STEPS))
    ap.add_argument("--lambda-fam", type=float, default=LAMBDA_FAM_DEFAULT)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--corpus-batch-size", type=int, default=32)
    ap.add_argument("--episode-batch-size", type=int, default=16)
    ap.add_argument("--gate-batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt-dir", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    assert sorted(args.ckpt_steps) == list(args.ckpt_steps), "--ckpt-steps must be given sorted"
    result = run_familiarization_cell(
        args.init_checkpoint, args.arm, args.corpus, args.seed, args.k, steps=args.steps,
        ckpt_steps=args.ckpt_steps, lambda_fam=args.lambda_fam, lr=args.lr,
        weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
        corpus_batch_size=args.corpus_batch_size, episode_batch_size=args.episode_batch_size,
        gate_batch_size=args.gate_batch_size, seq_len=args.seq_len, eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size, data_dir=args.data_dir, device=args.device,
        ckpt_dir=args.ckpt_dir, out_path=args.out, resume=not args.no_resume)
    summary = {k: v for k, v in result.items() if k not in ("trajectory", "checkpoints")}
    print("\nRESULT SUMMARY:", json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
