# EMBED-PATH INTERVENTION — DESIGN (Draft R0)

**Status:** DESIGN, not yet audited or built. Novelty gate CLEARED
(`EXPERIMENT_LOG.md` 2026-08-18 #14, verdict NOVEL-TO-US, executed 0x).
This document resolves the confound that gate found and specifies the
build. Needs its own audit round before any code is written (new code,
per this repo's standing rule — EXPERIMENT_LOG #14's own closing line).

**Pinned runner:** `~/ncr_g3b31_contrastive/ncr_lm_wave1_runner.py` on
`youthful-indigo-turkey`, md5 `9a93198b642242f512ff8489e32b0a53` —
re-verified live against the pin while writing this document (matches).
All line numbers below are against this exact file, read directly off
the box, not from memory or the design archive's prose.

---

## §1 The claim

**Pre-registered claim, one sentence:** cutting the aux+ortho losses'
gradient contribution into `backbone.embed.weight` — while leaving
`entity_adapter` trainable exactly as in compB and leaving CE's own
gradient into `embed` completely untouched — will raise compB's
`retrieval24_acc` @ h=61 (pbe_repl instrument) from its measured
median 0.7246 (n=20) toward the frozen arms' 0.9844 floor (n=18).

**What a positive result WOULD establish:** that compB's
adapter-trainable deep-composition degradation is caused (not merely
correlated with) an **interaction** between entity_adapter's
trainability and embed's openness to the aux/ortho gradient via the
o-side path — i.e., embed's leak matters, but only *given* a trainable
adapter. This is a narrower and more defensible claim than "embed
openness causes collapse" standalone: the frozen arms (primary/compA)
*already have embed open* and compose at 0.9844–1.0, which is direct
evidence in our own records that embed-openness alone is NOT
sufficient to cause degradation. Any framing of this experiment that
omits that tension is not honest to the evidence we already have (see
§6.2).

**What a positive result would NOT establish:**
- That embed's leak is the *only* factor in compB's spread — the drift
  analysis's own seed-level table (`COMPB_DRIFT_ANALYSIS.md`, n=18)
  shows compB's within-arm variance is large (0.617–0.973) and
  unexplained by two other measured properties; closing embed's
  aux-path is a population-level shift test, not a claim about every
  seed's residual variance.
- Anything about §G3-B32's own TPC_fg Band-1 collapse metric (compA's
  own 0.797–0.814 target-space-collapse number) — this design tests
  the retrieval24_acc axis on the compB (trainable-adapter) arm, not
  the TPC_fg axis on the compA (frozen-adapter) arm §G3-B32 actually
  measured. Closing embed's effect on compA's own TPC_fg is a
  DIFFERENT, narrower, still-untested question (§6.4).
- Causal necessity of any single gradient unit "aux-only norm 110.13"
  quoted from R2 — that number was measured on a frozen-adapter cell;
  this design's own conduit will be measured fresh on compB's
  trainable-adapter configuration and is expected to differ (§5).

**Why this is the first interventional test, not a fourth correlational
one:** every prior round in this chain (freeze contrast, tick #5;
conditioning/drift legs, `COMPB_DRIFT_ANALYSIS.md`) held the
architecture fixed and *observed* which arms happened to differ. This
design is the first one that *changes the gradient graph itself* to
test whether the specific R2-named conduit is load-bearing. A positive
result here is evidence of a different kind than anything upstream of
it, even though the qualitative direction (embed matters) has already
been pointed at by R2's own audit-time measurement and by §G3-B32's
compA observation — so per this repo's sightedness-disclosure
convention (`COMPB_DRIFT_ANALYSIS.md` leg (a)), this is registered as
**directionally sighted, mechanistically untested** — the audit already
told us where to look; this experiment is the first one that actually
looks by intervening.

---

## §2 Mechanism + exact code delta

### 2.1 The confound resolution — and a correction to the brief's own candidate (a)

The task brief's candidate (a) reads: *"cut the AUX loss's gradient
into embed only — e.g. `entity_adapter(embed(ids).detach())` in the
aux-target computation."* Read literally against the actual pinned
code, **this is a no-op.** Both aux-target call sites already end in a
`.detach()` on the *entire* `entity_adapter(embed(...))` composite:

```python
# aux_read_supervision_loss, line 630:
target_o = integ.entity_adapter(embed(answer_token).float()).detach()
# contrastive_read_supervision_loss, line 690:
T = integ.entity_adapter(embed(entity_ids).float()).detach()
```

Inserting `.detach()` one step earlier (on `embed(ids)` before it enters
`entity_adapter`) inside an expression whose *entire output* is already
detached changes nothing — zero gradient flows from either target
computation today, regardless of where inside it you place `.detach()`.
This is not a matter of interpretation; it follows directly from what
`.detach()` does to an autograd graph. The docstring at line 599
(`aux_read_supervision_loss`) says this explicitly: *"the ONLY path
this loss can travel to reach any parameter is through `o`, i.e.
through the read itself."* So the real conduit R2 measured (110.13,
"via the o-side path") is not the target computation at all.

**The real conduit.** `o_raw` is computed once per step
(`ncr_lm_forward_ablatable`, line 360) via
`keys_v, values_v = integ.extract_kv(input_ids, ..., backbone.embed)`
and `q_key = integ.query_key(input_ids, ..., backbone.embed)` — both
of which read `backbone.embed` directly on raw token ids (sec G3-B12's
own re-basing). This SAME `o_raw` tensor is then used **live,
undetached**, in two places inside `compute_arm_losses` (line 768):

1. As `o_injected` inside `ncr_lm_forward_ablatable`
   (`o_injected = o_raw` when `read_ablate=False`), which feeds
   `logits` → `ce_loss` — the intended, load-bearing CE route.
2. Passed directly to `aux_read_supervision_loss(..., o_raw, ...)` /
   `contrastive_read_supervision_loss(..., o_raw, ...)` at lines
   833/836/840 — the aux route.

Because it is the *same tensor*, `total_loss.backward()` (line 1335,
the single combined backward call) sums CE's and aux's gradient
contributions before they ever reach `o_raw`'s upstream nodes
(`extract_kv`/`query_key`/`entity_adapter`/`embed`). There is no way
to separate "the aux share of embed.weight.grad" from "the CE share"
using a single backward pass through a shared node — by the time
gradient arrives at `embed.weight`, the two sources are already summed.

**Also newly identified (not named in R2's own text): a second,
separate conduit.** `ortho_reg_weight` (0.1 in every launched
G3-B31 cell — 0992/0993/0994's own `cmd` strings) applies
`ortho_regularization_loss(Z)` (line 743) directly to `Z =
ncr_head.encode(keys_v, values_v)`, which *also* depends on
`backbone.embed` through `extract_kv` — a route that does not go
through `o` at all, so it is not the "o-side path" R2 named, but it is
structurally the same kind of leak (aux/ortho signal reaching a frozen-
or-not-yet-tested factor through a shared upstream node) and it is
present in every cell this campaign has run with `--ortho-reg-weight
0.1`. **Any design that closes only the o-side conduit and leaves
ortho's Z-side conduit open has NOT closed "the embed factor" — it has
closed one of two known conduits into it**, and a null result under
that half-closure would be uninterpretable. §2.2 closes both.

### 2.2 The chosen resolution: split-backward gradient assembly (a refined)

Not literal freezing (rejected, §2.3) and not a duplicated
forward/write pipeline (considered and rejected as unnecessarily
expensive, §2.3) — the correct minimal-footprint fix exploits that
`compute_arm_losses` already returns `total_loss`, `ce_loss` as
separate tensors from the *same* forward graph, and that
`total_loss = ce_loss + aux_read_loss_weight*aux_loss +
ortho_reg_weight*ortho_loss` (lines 845–850) is a plain sum. So
`non_ce_term := total_loss - ce_loss` is, algebraically, *exactly*
`aux_read_loss_weight*aux_loss + ortho_reg_weight*ortho_loss` (or the
identical `ce_loss` tensor object when neither branch ran — the
existing "byte-identical when OFF" guarantee compute_arm_losses's own
docstring already promises, lines 795–815) — a valid graph node, no
new numerical error beyond one float subtraction.

Two `torch.autograd.grad()` calls over that ONE shared graph
(`retain_graph=True` on the first) give `grad_ce` and `grad_rest`
per-parameter. Assembly rule:

- `backbone.embed.weight` → `.grad = grad_ce[embed]` **only**. Aux and
  ortho's contribution (`grad_rest[embed]`) is computed (so it can be
  measured — see §5) and then explicitly discarded.
- Every other trainable parameter (`entity_adapter` included, since it
  stays trainable in this arm exactly as in compB; `ncr_head`;
  backbone's non-embed layers) → `.grad = grad_ce[p] + grad_rest[p]`,
  i.e. bit-for-bit what a normal `total_loss.backward()` would have
  produced. **Nothing about the aux loss's intended training signal to
  the write pathway (`entity_adapter`, `ncr_head`) changes.**

This is the scope discipline the naive "detach `o` before the aux
call" alternative fails: detaching `o_raw` wholesale for the aux call
would cut aux's gradient to *everything* upstream (embed **and**
entity_adapter **and** ncr_head), since the docstring-stated "only
path" is shared by all three — that reproduces something closer to a
CE-only ablation of the entire write pathway, not a surgical embed-only
cut, and would not test R2's claim at all. The split-backward keeps
aux/ortho's signal flowing to `entity_adapter`/`ncr_head` exactly as
compB already runs, and removes only the marginal contribution to
`embed.weight`.

**Scope statement (pre-empting an obvious misreading):** CE's own
gradient into `embed` already flows through *two* routes today — the
direct embedding lookup (backbone's main forward) and the tied LM head
(`out = F.linear(x, self.embed.weight)`,
`~/chapter2/deltanet_rd/lm_pretrain_rd.py:1310`) — **and also** through
the same o-side path as aux (since `o_injected` feeds `logits` too).
This design changes NONE of that. `grad_ce[embed]` already includes
CE's own use of the o-side path; that is pre-existing, architecturally
load-bearing behavior of the whole NCR read-injection mechanism, not
something any candidate under consideration should or does touch. Only
aux+ortho's *additional* contribution is removed.

### 2.3 Why (b) and (c) are worse

**(b) Freeze `embed.weight` outright.** `nn.init.normal_(self.embed.weight,
mean=0.0, std=0.02)` (`lm_pretrain_rd.py`, near line 1223) initializes
embed at random; freezing it there freezes BOTH the token-embedding
lookup table AND — because of the weight tie — the entire output
logit projection for every training step. CE would then only be able
to adapt through the transformer stack's own weights, learning to
align its final hidden state to a fixed random 768-dim output basis
(closer to a random-feature/reservoir output head than a trained LM
head). This changes primary CE training dynamics for the WHOLE model,
not just the o-side aux path — precisely the confound the archive gate
flagged. A degraded-or-unchanged retrieval result under (b) would be
inseparable from "the whole model trained worse because its output
head never adapted." Rejected — this is the option EXPERIMENT_LOG #14
already identified as invalidating.

**(c) Untie the LM head.** Gives the head its own `nn.Linear(768,
vocab_size)`, ~38.4M new parameters (`vocab_size_total × d_model`) at
this task's vocab size — a real architecture change relative to
*every other arm in this campaign* (primary, compA, compB, compD, and
the underlying `mob_g3b24`/`g3b26`/`g3b28` lineage all share a tied
head). Breaks "same everything else": a positive OR negative result
would be confounded with (i) the extra head capacity, (ii) different
optimization dynamics for a freshly-initialized head vs. one that
inherits GPT-2's tokenizer-matched embedding scale from the start, and
(iii) needing its own calibration run (this repo's own hard rule:
"a calibration run before a big sweep is mandatory") before any
comparison to the existing frozen/trainable arms would be meaningful.
Also the largest code change of the three, touching the pretrained
architecture file rather than the runner. Rejected as disproportionate
to the question being asked.

**Chosen: (a) refined (§2.2).** Zero new parameters, zero change to
the forward computation graph (both `ce_loss` and `total_loss` are
computed exactly as today — `compute_arm_losses` itself is
UNCHANGED), zero change to `build_optimizer`'s param-group structure
(embed stays in the same single param group compB already uses; only
the training loop's backward call is conditionally replaced). The only
new cost is a second `torch.autograd.grad()` pass, and because
`aux_loss`/`ortho_loss` depend only on `o_raw`/`Z` — both computed
directly from `backbone.embed` on raw ids inside `extract_kv`/
`query_key`, NOT from `hidden` — that second pass's backward graph
**does not traverse the backbone's transformer stack at all**. The
expensive part of backward (propagating through however many DeltaNet
layers `RUNG1_BACKBONE` specifies) is paid once (in `grad_ce`, as
today); the second pass only re-walks the short
`embed→entity_adapter→keys_v/values_v/q_key→Z/o→aux+ortho` subgraph.
Expected overhead is therefore well under a naive "2× backward" bound
— disclosed as an ESTIMATE requiring build-time smoke confirmation
(§3, §5), not a promise.

### 2.4 Code delta — exact functions/lines to add or change

All changes are additive or narrowly scoped; no existing function's
signature loses an argument and no existing behavior changes when the
new flag is off (mirrors `freeze_entity_adapter`'s and
`aux_read_loss_weight`'s own "OFF is byte-identical to never having
been added" convention, `compute_arm_losses` docstring lines 795–815).

**New CLI flag**, added next to `--freeze-entity-adapter`
(`action="store_true"`, line 1835):
```
--close-embed-aux-path   (default False)
```

**New helper function**, placed after `build_optimizer` (line 908,
before the `# --- Eval ---` block at line ~925) — same file section as
`freeze_entity_adapter_`:

```python
def assemble_embed_closed_grads_(arm: dict, total_loss: torch.Tensor,
                                  ce_loss: torch.Tensor) -> dict:
    """--close-embed-aux-path: replaces total_loss.backward() for full_graft
    with two torch.autograd.grad() passes over the SAME forward graph
    (retain_graph=True on the first). backbone.embed.weight receives
    grad_ce ONLY; every other trainable param (entity_adapter included,
    when not itself frozen; ncr_head; backbone's non-embed layers)
    receives grad_ce + grad_rest, bit-identical to what total_loss.backward()
    would have produced. non_ce_term = total_loss - ce_loss is EXACTLY
    aux_read_loss_weight*aux_loss + ortho_reg_weight*ortho_loss by
    compute_arm_losses's own additive construction (lines 845-850) --
    closes BOTH the o-side conduit (aux) and the Z-side conduit (ortho,
    present at ortho_reg_weight=0.1 in every launched G3-B31 cell) into
    embed, not only the one R2's own text names.
    Returns {"aux_ortho_into_embed_norm": float, "cut_active": bool} --
    the has-teeth diagnostic (see assert_embed_aux_path_closed)."""
    embed_w = arm["backbone"].embed.weight
    all_params = [p for p in (list(arm["backbone"].parameters()) +
                               list(arm["ncr"].parameters()) +
                               list(arm["integ"].parameters())) if p.requires_grad]
    if total_loss is ce_loss:          # aux AND ortho both off this step -- nothing to cut
        grad_ce = torch.autograd.grad(ce_loss, all_params, allow_unused=True)
        for p, g in zip(all_params, grad_ce):
            p.grad = g
        return {"aux_ortho_into_embed_norm": 0.0, "cut_active": False}
    non_ce_term = total_loss - ce_loss
    grad_ce = torch.autograd.grad(ce_loss, all_params, retain_graph=True, allow_unused=True)
    grad_rest = torch.autograd.grad(non_ce_term, all_params, retain_graph=False, allow_unused=True)
    embed_idx = all_params.index(embed_w)
    aux_ortho_embed_norm = (grad_rest[embed_idx].norm().item()
                             if grad_rest[embed_idx] is not None else 0.0)
    for p, gce, grest in zip(all_params, grad_ce, grad_rest):
        if p is embed_w:
            p.grad = gce
        else:
            g = gce
            if grest is not None:
                g = (g + grest) if g is not None else grest
            p.grad = g
    return {"aux_ortho_into_embed_norm": aux_ortho_embed_norm, "cut_active": True}
```

**New "has-teeth" assertion**, placed after `assert_entity_adapter_grad_none`
(line 752), same "every step, loud AssertionError" convention:

```python
def assert_embed_aux_path_closed(grad_diag: dict, step: int,
                                  min_conduit_norm: float = 1.0) -> None:
    """Run every step --close-embed-aux-path is active (full_graft only).
    (1) HAS-TEETH: the aux+ortho->embed conduit this flag closes must be
    measurably nonzero (R2's own reference measurement on the frozen-
    adapter arm was 110.13; this arm's number will differ since
    entity_adapter is trainable here -- logged, not pinned to that value)
    -- else the flag is closing an already-empty door and any null result
    downstream is uninterpretable. (2) grad_diag['cut_active'] must be True
    whenever this function is called (defensive re-assertion that the
    branch actually ran, not merely that the flag was passed)."""
    assert grad_diag["cut_active"], f"step {step}: close_embed_aux_path expected an active cut"
    assert grad_diag["aux_ortho_into_embed_norm"] > min_conduit_norm, (
        f"step {step}: HAS-TEETH CHECK FAILED -- aux+ortho->embed conduit norm "
        f"{grad_diag['aux_ortho_into_embed_norm']:.4f} is not measurably nonzero; "
        f"the flag would be closing an already-empty path")
```

**Training loop change** (replaces lines 1334–1335 inside the
per-arm loop, ~line 1315 `for arm_name, read_ablate in (...)`):

```python
opt.zero_grad()
if close_embed_aux_path and arm_name == "full_graft":
    grad_diag = assemble_embed_closed_grads_(arm, total_loss, ce_loss)
    assert_embed_aux_path_closed(grad_diag, step)
    step_embed_diag = grad_diag              # logged into rec, new field
else:
    total_loss.backward()
```

Everything downstream (the `finite = all(...)` check, `clip_grad_norm_`,
`opt.step()`, lines 1349–1358) is unchanged — it only reads `.grad`,
regardless of whether `.backward()` or the manual assembly populated it.

**`run_two_arm_cell` signature** (line 1185): add
`close_embed_aux_path: bool = False`, threaded through from `main()`'s
`run_two_arm_cell(...)` call (~line 1876) the same way
`freeze_entity_adapter` already is.

### 2.5 Checkpoint/resume bookkeeping (mirrors the seed-trap / freeze-trap pattern exactly)

`save_checkpoint` (line 1086): add `close_embed_aux_path: bool = False`
parameter, add `"close_embed_aux_path": close_embed_aux_path` to the
`ckpt = {...}` dict (next to the existing `"freeze_entity_adapter"`
key).

`run_two_arm_cell`'s resume path (immediately after the existing
freeze-mismatch assert that follows the `SEED MISMATCH` block, line
1216 region): add a third assert, same shape:

```python
ckpt_close_embed = ckpt.get("close_embed_aux_path", close_embed_aux_path)
assert ckpt_close_embed == close_embed_aux_path, (
    f"[{cell_id}] --close-embed-aux-path MISMATCH on resume: checkpoint {ckpt_path!r} was "
    f"built with close_embed_aux_path={ckpt_close_embed} but this launch passed "
    f"close_embed_aux_path={close_embed_aux_path}. Resuming would silently switch which "
    f"gradient-assembly path trains backbone.embed.weight mid-run, producing a hybrid "
    f"trajectory that is neither condition. Re-launch with the matching flag, or point "
    f"--ckpt-dir at a fresh path.")
```

**`restore_arms_and_opts` (line 1136) needs NO change.** Unlike
`freeze_entity_adapter`, this flag does not alter `build_optimizer`'s
param-group membership — `embed.weight` stays in the exact same single
param group compB's own (`freeze_entity_adapter=False`) optimizer
already uses. The flag only changes which gradient VALUE lands in
`.grad` before `opt.step()`, at training-loop level, not the
optimizer's structural shape a resume needs to match. This is a
deliberate, disclosed scoping decision, not an oversight — flagged
explicitly here so an audit can check it rather than assume it.

---

## §3 Cells, seeds, budget

**Base config — identical to compB's own launched cells** (0994 /
`mob_g3b31_compB_s{1..20}`, confirmed against `~/queue/completed/
0994_ncr_g3b31_compB_s0.json`'s own `cmd`): `--steps 20000 --batch-size
32 --eval-batch-size 64 --warmup-steps 200 --lr 3e-4 --aux-read-loss-weight
0.5 --ortho-reg-weight 0.1 --aux-loss-type contrastive+cosine
--contrastive-temperature 0.07 --ckpt-every 10000 --eval-every 1000
--ceiling-gpuh 6.0`. **No `--freeze-entity-adapter`** (entity_adapter
stays trainable, matching compB — this design tests the interaction,
not the frozen-adapter regime). Checkpoints to
`/ephemeral/embed_path_ckpts/mob_gembed_compE_s{seed}_ckpts/` per the
disk policy (EXPERIMENT_LOG #4 — root stays at 68%, never write
`results/*_ckpts`).

**Arm name: `compE`** — unclaimed in this campaign's naming
(`primary`/`compA`/`compB`/`compD` exist; `compC`/`compE` do not,
checked against both the repo's design doc and every file under
`~/ncr_g3b31_contrastive` and `~/ncr_writecond` on the box).

**Cell 1 — compE (the test arm), n=8 fresh seeds (s21–s28, continuing
past compB's own s1–s20 so no seed number collides with an existing
archived cell):** `--close-embed-aux-path` added to the base config
above. n=8 satisfies the brief's `n>=8` floor; compared against the
**already-archived** compB n=20 distribution (median 0.7246, range
[0.6172, 0.9727]) and frozen-arm n=18 distribution (min 0.9844) — both
reused at zero additional GPU cost, exactly as tick #5's freeze
contrast reused matched-n archived data.

**Bands (retrieval24_acc @ h=61, pbe_repl instrument, `ckpt_step==20000`
guard):**
- **WIN:** compE median ≥ 0.90 AND compE's minimum > compB's own
  archived maximum (0.9727) — complete separation from compB by the
  same exact-Mann-Whitney convention this campaign has used
  throughout (ties this cell's own n=8 vs compB's n=20 with a
  computable exact two-tailed p; a full 8-vs-20 separation is p =
  2/C(28,8) ≈ 1.9e-6 if realized).
- **PARTIAL:** compE median in [0.80, 0.90), or median ≥ 0.90 without
  full separation from compB's max — a real, materially-sized shift
  (≥0.10 absolute over compB's 0.7246 median) that does not clear or
  cleanly separate from the frozen floor.
- **NULL:** compE median < 0.80, or statistically indistinguishable
  from compB's archived n=20 distribution (Mann–Whitney p > 0.05, or
  median inside compB's own IQR) — the embed-interaction mechanism is
  not supported, consistent with the drift analysis's own null-heavy
  pattern on the two properties it did test.

**Cell 2 — placebo, n=4 seeds (s1–s4 in a separate `compE_placebo`
namespace):** see §4 for the choice and its own disclosed limitation.
Same base config, same `--close-embed-aux-path`-style flag but
targeting `ncr_head`'s own trainable parameters instead of
`backbone.embed.weight` (`--close-embed-aux-path` generalized to a
`--close-target=ncr_head|embed` selector, OR a second dedicated flag —
audit's call at build time; either is a one-line change to
`assemble_embed_closed_grads_`'s target-selection logic). n=4 is
explicitly underpowered for a formal separation test — it is a coarse
directional check only (does this OTHER cut also "fix" retrieval, or
does it degrade it, or is it inert?), disclosed as such, not a
pre-registered statistical claim.

**Budget.** Per-cell rate: compB's own measured 0.8–1.0 GPU-h at 20K
steps is the floor; the split-backward's disclosed overhead (§2.3 —
short subgraph only, backbone stack not re-traversed) is estimated at
+10–30%, giving **~0.9–1.3 GPU-h/cell**, to be pinned by a build-time
smoke measurement before the full n=8/n=4 launch (this repo's own
"measured build-smoke numbers, not just theoretical estimates" habit).

| cells | n | GPU-h/cell (est.) | subtotal |
|---|---|---|---|
| compE (test) | 8 | 0.9–1.3 | 7.2–10.4 |
| compE_placebo | 4 | 0.9–1.3 | 3.6–5.2 |
| **wave-1 total** | | | **10.8–15.6** |

Sits at or just over the ≤15 GPU-h target at the high end of the
overhead estimate; if the build-time smoke measures overhead above
~20%, trim the placebo to n=3 first (it is already explicitly
underpowered, so losing one more seed costs little) before trimming
the test arm below the brief's n=8 floor.

---

## §4 Controls

**Recorded baseline (already archived, zero new cost):** compB n=20
(median 0.7246, adapter trainable, embed open, `--aux-loss-type
contrastive+cosine`, no `--freeze-entity-adapter`) — the direct "same
everything else minus this one flag" comparator, and the frozen arms
(primary n=14 + compA n=6, min 0.9844) as the ceiling reference.

**Placebo — cut a matched-conduit but different-target gradient path
(ncr_head's own trainable parameters), not a perfectly causally-inert
one.** Given the architecture, there is no parameter that (a) sits on
the exact same shared o-side/Z-side conduit `embed` sits on (needed so
the placebo rules out "any partial gradient cut helps," not merely
"cutting embed specifically helps") and (b) is uncontroversially
causally irrelevant to composition. `ncr_head` is the closest
available choice — same conduit (its trainable parameters are also
reached only through `extract_kv`/`query_key`/`Z`, the identical short
subgraph `assemble_embed_closed_grads_` already isolates) — but it is
**not** obviously inert: §G3-B17's own stated purpose for the aux loss
is "teach the encoder to write a composing operator," i.e. `ncr_head`
is the aux loss's *intended* target, not a bystander. This is
disclosed rather than papered over: a placebo result showing
degradation is the *expected* direction if `ncr_head` genuinely needs
aux gradient (uninformative about embed specifically, since it just
confirms the aux loss does something), while a placebo result showing
**no change** would be the informative, surprising direction (it would
suggest ncr_head's own aux gradient is not load-bearing either, which
would reframe the whole aux-loss rationale, not just this design). The
placebo's job here is narrower than a textbook inert control: it rules
out "any comparably-sized gradient ablation improves retrieval by
regularization alone," not "closing embed specifically is the unique
fix" — that second claim rests on comparing compE directly against
compB, not against the placebo.

**What the frozen arms predict if the mechanism is correct — the
tension already in our own records.** Primary/compA (embed open,
adapter frozen) already score 0.9844–1.0000 median/floor. If "embed
openness causes degradation" were sufficient on its own, primary/compA
should ALSO show degradation, since their embed is exactly as open to
the aux/ortho conduit as compB's is (R2's own 110.13 measurement was
taken on a frozen-adapter — i.e. primary/compA-class — cell). They
don't degrade. So the mechanism this design actually tests is
necessarily an **interaction**: embed's leak plus a *trainable*
adapter, not either factor alone. A design or a write-up that claims
support for "embed openness causes collapse" without this qualifier is
misreading its own evidence — flagged here explicitly so it cannot be
reintroduced downstream.

---

## §5 Smoke / verification plan

**(1) Has-teeth, per-step (build-in, §2.4):** `assert_embed_aux_path_closed`
runs every training step `--close-embed-aux-path` is active on
`full_graft`, checking (i) the discarded `grad_rest[embed]` was
measurably nonzero before being dropped (the conduit is real, not a
vacuous cut) and (ii) the cut branch actually executed. Loud
`AssertionError`, matching this repo's own "structural checks need
exact thresholds, never silently trusted" rule (`assert_entity_adapter_
grad_none`'s own precedent, line 752).

**(2) Standalone before/after smoke sub-test (construction-time, run
once per launch config before training starts — new sub-test in
whatever smoke harness this build extends, mirroring `ncr_lm_wave1_
aux_smoke.py`'s existing sub-test (c) pattern cited at line 626):**
On one fixed batch + fixed seed:
  - **(a) Conduit-is-real, baseline (flag OFF):** run the EXISTING
    single `total_loss.backward()`. Record `embed.weight.grad.norm()`
    = `norm_combined`. Separately, on a fresh copy of the same
    forward graph, run `ce_loss.backward()` alone and record
    `embed.weight.grad.norm()` = `norm_ce_alone`. Assert
    `abs(norm_combined - norm_ce_alone) > 1.0` — i.e. aux+ortho
    demonstrably move `embed.weight.grad`'s norm, echoing R2's own
    audit-time measurement methodology (that 110.13 figure) on THIS
    build's actual code path, not assumed to still hold.
  - **(b) Cut-confirmed, exact (flag ON):** run
    `assemble_embed_closed_grads_`. Independently recompute
    `grad_ce_only = torch.autograd.grad(ce_loss, [embed.weight],
    retain_graph=True)[0]` on the same graph. Assert
    `torch.equal(embed.weight.grad, grad_ce_only)` — EXACT equality
    (this repo's "structural correctness checks need exact
    thresholds, not tolerance" rule — CLAUDE.md), not `allclose`.
  - **(c) Scope-preserved:** compare every OTHER trainable parameter's
    `.grad` between the flag-OFF run (a) and the flag-ON run (b) on
    the identical batch/seed — assert bit-identical
    (`torch.equal`) for every parameter except `embed.weight`. This
    directly operationalizes "leaving CE's path to embed untouched
    AND leaving aux's path to everything else untouched" as one
    combined equality check, not an assumption.

**(3) Standard forward/backward/grad checks (existing runner
convention, unchanged):** the pre-existing `assert_read_ablation_is_
exact_zero` and `assert_read_target_write_key_same_op` checks (called
pre-train for both arms, lines ~1288–1292) run exactly as today —
neither depends on this flag and both must still pass, proving the new
code path was not accidentally wired into the read-ablation or
same-op invariants this campaign already depends on.

**(4) Finite-gradient check:** the existing `finite = all(p.grad is
None or torch.isfinite(p.grad).all() for p in all_params)` gate (line
~1350) runs unchanged after the manual assembly — since `.grad` is a
real tensor either way, this catches a NaN/Inf introduced by the
split-backward exactly as it would catch one from a normal backward.

**(5) Checkpoint/resume smoke:** kill a compE cell mid-run, resume it,
confirm (i) the new mismatch assert fires when resumed WITHOUT the
flag (negative test — deliberately launch a mismatched resume and
confirm the loud failure, mirroring this repo's "run the negative unit
test to completion, don't just write it" rule) and (ii) a matched
resume continues training with `assert_embed_aux_path_closed` still
passing every step post-resume.

---

## §6 Honest pre-attack

**6.1 — Reproduces compA's own collapse, exactly as this design's own
opening line warns.** If compE's retrieval24_acc stays at or near
compB's 0.7246 median, the most literal reading is "closing embed
doesn't help." This would NOT fully exonerate entity_adapter's own
trainability as *a* cause — the drift analysis only ruled out two
specific properties of the trained adapter (conditioning, drift-from-
init); a null here adds "closing its aux/ortho leak into embed" as a
third ruled-out mechanism, still leaving the true cause of compB's
seed-to-seed spread genuinely open, exactly as `COMPB_DRIFT_ANALYSIS.md`
already concedes ("not established: what DOES explain the spread").

**6.2 — The interaction framing is the correct one, and is the
easiest thing for a careless read-out to get wrong.** Because
primary/compA already show embed-open+adapter-frozen composing
perfectly, any positive compE result MUST be reported as "closing
embed rescues composition GIVEN a trainable adapter" — never as
"embed openness causes collapse" standalone, which our own frozen-arm
data already falsifies. §1 and §4 state this explicitly so a future
write-up cannot silently drop the qualifier; an R1 audit should
specifically check that this design's own eventual harvest report
keeps it.

**6.3 — Ambiguity of a null result.** A null compE outcome is
consistent with BOTH "the mechanism is wrong" and "the mechanism is
right but there's an independently larger co-occurring cause of
compB's degradation that swamps the fix" (e.g. entity_adapter's own
trainable geometry degrading Z's write quality through some channel
untouched by embed at all). This design cannot distinguish those two
readings on its own — a null here should be reported as "no support,"
matching the drift analysis's own careful language, not "mechanism
disproven."

**6.4 — This design does not touch §G3-B32's own TPC_fg claim.**
§G3-B32's compA observation (TPC_fg 0.797–0.814, "confirms R2's
prediction... the EMBED factor re-opens the collapse route") is about
the FROZEN-adapter arm's target-space collapse metric, measured at a
single seed (s0) in the Aug-6 harvest — a different arm, a different
metric, and a different (and, per the later Aug-18 multi-seed
retrieval24_acc numbers, apparently not fully reconciled — s0 itself
was never rerun under the pbe_repl instrument; `run_repl_wave2.sh`'s
own seed loop starts at s1) evidence base than this design's compB/
compE retrieval24_acc test. Closing embed on top of a FROZEN adapter
(i.e. `primary`/`compA` + `--close-embed-aux-path`) to test the actual
TPC_fg claim §G3-B32 made is a natural, cheap follow-on (reuses the
exact same code delta, just launched with `--freeze-entity-adapter`
also set) but is explicitly OUT of wave-1's ≤15 GPU-h budget here —
flagged as the natural next design, not silently assumed to be covered
by this one.

**6.5 — New code, first launch, no audited precedent for this exact
pattern.** `torch.autograd.grad` with a manual `.grad` assembly
replacing `.backward()` has no prior instance in this runner's audit
history (every existing flag — `freeze_entity_adapter`, `teacher_force
_operator`, `aux_loss_type` — changes what tensors get *built*, never
how backward itself is invoked). This is new engineering, not a
config flip, and per this repo's own standing rule needs its own
audit round (checking in particular: `retain_graph` correctness across
the two `autograd.grad` calls, that `allow_unused=True` doesn't
silently mask a genuinely-unreachable parameter that SHOULD be
reachable, and that the manual `.grad` assignment interacts correctly
with `clip_grad_norm_`'s and `opt.step()`'s existing per-step logic)
before any of wave-1 launches — this document does not claim that
audit as already passed.

**6.6 — Placebo's imperfect inertness (restated from §4).** Because
`ncr_head` is plausibly load-bearing for the aux loss's own stated
purpose, a placebo result in either direction is only weakly
diagnostic on its own; it should be read alongside compE's own result,
not as an independent tie-breaker.

**6.7 — This is a directionally-sighted test, not a blind one.**
R2's own audit measurement and §G3-B32's compA observation already
point toward embed. A positive compE result confirms a
already-suspected mechanism causally for the first time; it should be
reported with that context (matching this campaign's own sightedness-
disclosure convention), not framed as a cold discovery.

---

## DRAFT-R1

**Status:** Rev-1, responding to `matrix-thinking/NCR_EMBED_PATH_ATTACK_R1.md`
(repo commit `591301a`, REV-REQUIRED, 5 FATAL / 8 MAJOR / 5 minor, every
finding demonstrated by execution). §0's frozen-correct list is NOT
re-derived here (assembled gradient VALUES, `retain_graph` necessity,
`allow_unused` safety, AdamW behaviour off a manual `.grad`, absence of
AMP, `grad_ce[embed]` covering both tied roles). Every disposition below
is labelled with its source ID (D-F1 etc.) so a future audit can check
discharge 1:1 against the attack document.

Everything in this section was produced against the live box
(`youthful-indigo-turkey`), re-reading the pinned runner fresh (md5
`9a93198b642242f512ff8489e32b0a53`, unchanged) rather than trusting
either DRAFT-R0's or the attack's line-number citations. One correction
to the attack's own premise came out of that re-read (§M6 below). The
Stage-0 pre-test (§R1.1) was designed AND EXECUTED in this pass — it is
not a paper design awaiting a future launch; its result is reported
honestly below, including where it complicates rather than simply
confirms the wave.

---

### R1.1 — STAGE 0: eval-only init-swap pre-test (D-M2b). EXECUTED. Verdict: RE-SCOPE.

**Why this is possible at all (re-confirmed by direct code read, not
re-taken on the attack's word alone).** Under P1b (`teacher_force=True`),
tracing `ncr_lm_wave1_smoke.py:347-364` (`teacher_force_operator`: `Z =
pinv(keys_v.detach()) @ values_v.detach()`, keys/values from
`extract_kv` = `entity_adapter(embed(ids))`) and
`ncr_lm_wave1_runner.py:480-521` (`discriminability_metrics`: targets `T
= entity_adapter(embed(entity_ids))`; `o = binexp_read(Z, q_key, h)["o"]`,
a parameter-free repeated-squaring read, `nm.binexp_read`, no learned
weights of its own; `retrieval24_acc = argmax_k cos(o, T_k) ==
tgt_slot`) — the scored quantity is a pure function of exactly two
trained tensors, `integ.entity_adapter.weight` and `backbone.embed.weight`.
Neither `ncr_head`, the DeltaNet stack, nor `read_injector` appear
anywhere in this computation. So a **checkpoint's own P1b h=61 score can
be recomputed after surgically replacing either or both of those two
tensors, with zero training and zero new autograd code** — a strictly
coarser but much cheaper analogue of what the wave's split-backward is
trying to achieve by degree rather than by full substitution.

**Reconstruction method (re-used, not re-derived):**
`COMPB_DRIFT_ANALYSIS.md` (leg c) already established, sanity-gated at
three seeds, that `build_arm(vocab_size_total, seed, device)` — called
directly from the pinned runner, never reimplemented — reproduces a
checkpoint's own seeded init **bit-identically** (`torch.equal`, verified
at seeds 1/12/20). `ckpt["seed"]` is stored in every checkpoint
(`save_checkpoint`, confirmed by direct read: `"seed": seed` in the
saved dict), so the reconstruction needs no side channel.

**The swap matrix — every cell actually run, no held-back cells:**

| recipient (compB, `ckpt_step==20000`) | archived P1b h=61 |
|---|---|
| s2  | 0.6172 (near compB's own min) |
| s5  | 0.7266 (≈ compB's own n=20 median 0.7246) |
| s19 | 0.8203 (high) |
| s6  | 0.9727 (compB's own n=20 max) |

× 5 swap types, applied via `.data.copy_()` under `torch.no_grad()`
between `restore_arms_and_opts` and `eval_arm_at_hops` (identical calls
`pbe_repl.py` itself uses — `build_grammar_pools_and_cfg(seed=0)`,
`load_checkpoint`, `restore_arms_and_opts`, `eval_arm_at_hops` at
`HOPS=(1,61)`, `BASE_SEED=90210`, `teacher_force=True`, exactly P1b):

- **none** — no swap. Pure parity check: must reproduce the archived
  score, or the harness itself is broken and nothing else in this
  section means anything.
- **embed_init** — `backbone.embed.weight` ← that seed's own
  `build_arm`-reconstructed init. `entity_adapter` stays at its real,
  fully-trained compB value.
- **adapter_init** — `entity_adapter.weight` ← that seed's own
  reconstructed init. `embed` stays at its real, fully-trained
  (fully aux+ortho-leaked-into) compB value.
- **both_init** — both reset.
- **embed_donor** — `backbone.embed.weight` ← `mob_g3b31_primary_s1`'s
  own **trained** (not init) embed (ckpt_step 20000, archived P1b h=61 =
  1.0) — embed that trained WITH the aux/ortho conduit open (primary is
  not frozen on embed) but WITHOUT a co-adapting trainable
  `entity_adapter` (primary freezes the adapter). The closest available
  *real* counterfactual to "what would compB's embed look like if the
  interaction hadn't been live," as opposed to a synthetic reset.
  `entity_adapter` stays at compB's own trained value. Donor/recipient
  embed shapes asserted equal before the copy (both built from the same
  `build_grammar_pools_and_cfg(seed=0)` vocab).

20 cells. Script (run via stdin over SSH, nothing written to the box —
`ssh youthful-indigo-turkey "cd ~/ncr_writecond && CUDA_VISIBLE_DEVICES=0
/home/nvidia/tdenv/bin/python3 -" < stage0_embed_swap.py`, local copy at
`/private/tmp/claude-501/.../scratchpad/stage0_embed_swap.py` this
session, not committed to the repo per the "only repo write is this
file" constraint):

```python
#!/usr/bin/env python3
"""STAGE-0 PRE-TEST (embed-path intervention, Rev-1, D-M2b): eval-only
init-swap ablation on ARCHIVED compB checkpoints. Zero training. Reuses
the audited pbe_repl instrument's own calls (build_grammar_pools_and_cfg,
load_checkpoint, restore_arms_and_opts, eval_arm_at_hops) with exactly one
inserted step between restore and eval: overwrite embed.weight and/or
entity_adapter.weight .data in-place. No new forward-pass code, no new
autograd code -- this script cannot exercise F2/F3/F5 at all, because it
never calls backward().
"""
import json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/ncr_writecond"))
import torch
import ncr_lm_wave1_runner as R

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SEED = 90210
HOPS = (1, 61)
OLD = os.path.expanduser("~/ncr_g3b31_contrastive/results")
NEW = "/ephemeral/reseed_ckpts"

RECIPIENTS = {
    "s2":  {"seed": 2,  "archived_h61": 0.6171875},
    "s5":  {"seed": 5,  "archived_h61": 0.7265625},
    "s19": {"seed": 19, "archived_h61": 0.8203125},
    "s6":  {"seed": 6,  "archived_h61": 0.97265625},
}
DONOR_NAME = "mob_g3b31_primary_s1"
SWAP_TYPES = ("none", "embed_init", "adapter_init", "both_init", "embed_donor")


def find_ckpt(tag, seed):
    name = f"mob_g3b31_{tag}_s{seed}"
    for root in (NEW, OLD):
        p = f"{root}/{name}_ckpts/{name}.ckpt.pt"
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no checkpoint for {tag} s{seed} in {NEW} or {OLD}")


def main():
    pools, cfg, pool_report = R.build_grammar_pools_and_cfg(seed=0)
    pools = pools.to(DEVICE)
    vocab = pool_report["vocab_size_total"]

    donor_path = find_ckpt("primary", 1)
    donor_ckpt = R.load_checkpoint(donor_path, DEVICE)
    assert donor_ckpt is not None, f"donor checkpoint missing at {donor_path}"
    donor_arms, _, _ = R.restore_arms_and_opts(donor_ckpt, vocab, lr=3e-4, device=DEVICE,
                                                freeze_entity_adapter=True)
    donor_embed_w = donor_arms["full_graft"]["backbone"].embed.weight.data.clone()

    results = []
    for rname, rinfo in RECIPIENTS.items():
        seed = rinfo["seed"]
        ck_path = find_ckpt("compB", seed)
        for swap in SWAP_TYPES:
            t0 = time.time()
            ckpt = R.load_checkpoint(ck_path, DEVICE)
            assert ckpt is not None, f"missing {ck_path}"
            assert ckpt["seed"] == seed, f"ckpt seed {ckpt['seed']} != expected {seed}"
            assert ckpt["step"] == 20000, f"ckpt_step guard failed: {ckpt['step']}"
            arms, _, _ = R.restore_arms_and_opts(ckpt, vocab, lr=3e-4, device=DEVICE,
                                                  freeze_entity_adapter=False)
            arm = arms["full_graft"]
            if swap in ("embed_init", "adapter_init", "both_init"):
                init_arm = R.build_arm(vocab, seed, DEVICE)   # bit-identical seeded
                                                                # reconstruction (COMPB_DRIFT_ANALYSIS.md)
            with torch.no_grad():
                if swap == "embed_init":
                    arm["backbone"].embed.weight.data.copy_(init_arm["backbone"].embed.weight.data)
                elif swap == "adapter_init":
                    arm["integ"].entity_adapter.weight.data.copy_(init_arm["integ"].entity_adapter.weight.data)
                elif swap == "both_init":
                    arm["backbone"].embed.weight.data.copy_(init_arm["backbone"].embed.weight.data)
                    arm["integ"].entity_adapter.weight.data.copy_(init_arm["integ"].entity_adapter.weight.data)
                elif swap == "embed_donor":
                    assert arm["backbone"].embed.weight.shape == donor_embed_w.shape
                    arm["backbone"].embed.weight.data.copy_(donor_embed_w)
                elif swap == "none":
                    pass
                else:
                    raise ValueError(swap)
            with torch.no_grad():
                p1b = R.eval_arm_at_hops(arm, pools, cfg, HOPS, 256, DEVICE,
                                          BASE_SEED, read_ablate=False, teacher_force=True)
            rec = dict(recipient=rname, seed=seed, swap=swap,
                       h1=p1b["h=1"]["retrieval24_acc"], h61=p1b["h=61"]["retrieval24_acc"],
                       archived_h61=rinfo["archived_h61"], elapsed_s=round(time.time() - t0, 2))
            results.append(rec)
            print(json.dumps(rec)); sys.stdout.flush()

    print("--- PARITY CHECK (swap=none vs archived) ---")
    for r in results:
        if r["swap"] == "none":
            diff = abs(r["h61"] - r["archived_h61"])
            print(f"  {r['recipient']}: eval={r['h61']:.6f} archived={r['archived_h61']:.6f} "
                  f"diff={diff:.2e} [{'OK' if diff < 1e-4 else 'MISMATCH'}]")
    print("STAGE0_SWAP_DONE")


if __name__ == "__main__":
    main()
```

**Executed result (20/20 cells, ~35s wall-clock total, one GPU, zero box
writes):**

| recipient | none (archived) | embed_init | adapter_init | both_init | embed_donor |
|---|---|---|---|---|---|
| s2  | 0.6172 | 0.9922 | 0.9961 | 0.9961 | 1.0000 |
| s5  | 0.7266 | 1.0000 | 1.0000 | 1.0000 | 0.9961 |
| s19 | 0.8203 | 0.9922 | 0.9922 | 1.0000 | 0.9883 |
| s6  | 0.9727 | 1.0000 | 1.0000 | 1.0000 | 0.9961 |
| **median** | **0.7735** | **0.9961** | **0.9980** | **1.0000** | **0.9961** |

`h=1` was 1.0 in all 20 cells (as archived) — ceiling, uninformative,
consistent with the design's own h=1 co-condition rationale (D-F1).
**Parity check: PASS, `diff = 0.00e+00` on all four `none` cells** — the
harness reproduces the archived score exactly, so the swap results are
not a construction artifact.

**Reading.** This is NOT the KILL band `D-M2b` specified ("if [embed
reset] does not recover retrieval... a non-rescue is close to
dispositive against the hypothesis") — every swap type recovers
retrieval, fully, at every recipient, regardless of where it started.
But it is also not a clean AUTHORIZE for the wave **as specified**,
because the rescue is **symmetric and saturating**: `adapter_init`
(entity_adapter reset, embed left at its real, fully-trained,
fully-aux+ortho-leaked-into compB value) recovers retrieval **just as
completely** as `embed_init` does. If the wave's registered mechanism
were right in its specific, asymmetric form — "embed's corrupted trained
value is the carrier; entity_adapter's own full compB training is not
itself the problem" — resetting adapter alone while leaving the
allegedly-corrupted embed untouched should do little. It does not do
little; it fully rescues, symmetrically with the embed-targeted swap.
The informative asymmetry the wave's causal story depends on is not
present in this evidence.

**Pre-registered partition (defined from the measurement-graph structure
above, extending D-M2b's own logic to the adapter arm the trace makes
available — stated here, then applied to the table above; not
re-fit to the numbers after the fact):**
- **KILL:** `embed_init` median ≈ `none` median (no rescue) → embed's
  trained VALUE is not implicated at all; do not build.
- **AUTHORIZE (embed-specific, licenses the wave exactly as scoped):**
  `embed_init` rescues AND `adapter_init` does NOT (stays near `none`)
  → asymmetric, embed-specific evidence.
- **RE-SCOPE (the observed reading):** BOTH `embed_init` and
  `adapter_init` independently rescue → a joint/symmetric co-adaptation
  effect, not an embed-specific one; the wave may proceed but MUST
  include a co-equal entity_adapter-targeted arm before any compE-alone
  result can be attributed to embed specifically.

**Verdict: RE-SCOPE.** Binding consequence for §3/§4 (superseding
DRAFT-R0's placebo choice and folding in D-M1/D-M2/D-M2b together, per
D-M1's own stated fallback — *"if the fourth cell is not adopted, spend
the 4 cells on the entity_adapter-target cut instead"* — now
evidence-motivated rather than merely budget-motivated):

1. **`compE_adapter` replaces `compE_placebo` entirely.** Drop the
   `ncr_head` target (D-M2 — it was already causally off the measurement
   graph; this pre-test gives an *additional*, independent reason to
   prefer `entity_adapter`: it is not just "on-path," it is empirically
   at least as strong a lever as embed itself). 4 cells, same budget
   line the placebo already had.
2. **M1's fourth 2×2 cell (frozen adapter + closed embed) is DEFERRED,
   not funded this wave** — flagged the same way §6.4 already flags the
   TPC_fg follow-on, not silently dropped. Partial justification from
   this pre-test: `adapter_init` (adapter neutralized, embed left at its
   real, fully-leaked compB value) already reaches ~1.0 in all 4 tested
   seeds, which is evidence — not proof — for M1's own one-sided ceiling
   assumption (a frozen-adapter cell with embed open is unlikely to show
   degradation). Disclosed caveat: `adapter_init` is a *snapshot
   substitution*, not a 20,000-step frozen-throughout-training run; it
   does not reproduce whatever co-adaptation history a genuinely frozen
   arm would or would not have had with embed. Revisit if `compE` /
   `compE_adapter`'s real wave results are surprising.
3. **The interaction claim (§1) is now conditioned on this evidence,
   not just a disclosed possibility** — see the restated §1 in R1.9.

**Coarseness caveat (disclosed, D-M2b's own):** this swap is a MAXIMAL
intervention (100% divergence from the trained value) versus compE's
much smaller, gradient-share-dependent divergence (only the aux+ortho
marginal contribution to embed's total received gradient is removed;
CE's larger share stays). A full-reset rescue does not guarantee
compE's finer intervention moves embed's final value far enough to
reproduce it — so this result does not by itself predict compE's
*magnitude*. What it does settle, structurally, is that the *specificity*
claim (only embed, not entity_adapter) is not supported by anything run
so far, in either direction — which is exactly why `compE_adapter` is
now mandatory rather than optional.

---

### R1.2 — D-F1: metric regime, named everywhere, with the h=1 co-condition and attrition rule

Every occurrence of "retrieval24_acc @ h=61" in this document (§1, §3,
bands, R1.1 above) means, precisely: **`P1b.result["h=61"]
["retrieval24_acc"]`, regime P1b (`teacher_force=True`, exact-write
substitution), `pbe_repl` instrument pinned at `seed=90210`, `n=256`,
`ckpt_step==20000` guard.** P0 (`teacher_force=False`) numbers are never
compared against P1b numbers or against the bands below; if a future
report needs a P0 reading it must be labelled P0 explicitly, in the same
sentence as the number, every time (this is the SECOND time this exact
defect has been caught in this campaign — R1.1's own table above states
"P1b" in its header for this reason).

**Co-condition (adopted from #7/#9, restored):** every scored cell's
own `median(P1b h=1) ≥ 0.95` or it is reported separately as an
**instrument-validity failure**, not folded into the h=61 median. (All
20 R1.1 cells and all archived compB/primary/compA cells checked so far
score `h=1 = 1.0` — the co-condition is not live yet, but it stays in
force for compE/compE_adapter.)

**Attrition rule:** verdict is read at n≥7 of 8 for compE, n≥3 of 4 for
compE_adapter; void below.

---

### R1.3 — D-F2: `non_ce` built from the returned tensors, never by subtraction; re-derived budget

`compute_arm_losses` (confirmed at line 768, unchanged) already returns
`aux_loss` and `ortho_loss` as separate tensors (`None` when their
weight is ≤0 or the arm is not `full_graft` — confirmed by direct read,
lines 833-851). Build `non_ce` from those directly:

```python
def _non_ce_term(aux_loss, ortho_loss, aux_read_loss_weight, ortho_reg_weight):
    """None iff both branches are inactive this step (defensive; compE's
    own launch config always has aux_read_loss_weight=0.5, ortho_reg_weight=0.1,
    so this is always a real tensor in practice -- guarded anyway per D-F2."""
    non_ce = None
    if aux_loss is not None and aux_read_loss_weight > 0.0:
        non_ce = aux_read_loss_weight * aux_loss
    if ortho_loss is not None and ortho_reg_weight > 0.0:
        term = ortho_reg_weight * ortho_loss
        non_ce = term if non_ce is None else non_ce + term
    return non_ce
```

This never subtracts `total_loss - ce_loss`, so it never re-walks the
backbone with a seeded-zero gradient (F2's demonstrated failure mode).
The full corrected assembly (which also folds in D-F5's cut-after-clip
and D-A2's ratio check) is given whole in R1.6, since F2/F5/A2 are one
function, not three independent patches.

**Re-derived budget (D-F2's own measured rates, unchanged by R1.1's
re-scope since cell COUNT is unchanged, only cell TARGET):**

| construction | GPU-h/cell | cells | subtotal |
|---|---|---|---|
| direct (`non_ce` from returned tensors, +2-10% overhead) | 0.82-1.10 | compE ×8 | 6.56-8.80 |
| direct | 0.82-1.10 | compE_adapter ×4 | 3.28-4.40 |
| **wave-1 total** | | 12 | **9.84-13.20 GPU-h** |

Within the ≤15 GPU-h cap, unlike DRAFT-R0's as-specified 14.4-20.0.

**`grad_rest` `None`-set assertion (D-F2's own explicit ask):** since
`allow_unused=True` is now genuinely load-bearing (backbone params
become unreachable from `non_ce`, so they come back `None`, not
zero-tensors), assert the SET of `None` entries is exactly the expected
one — see `_assert_expected_none_set` in R1.6.

---

### R1.4 — D-F3: verification plan rewritten so it runs; two-tier check; F3 extends to `embed` too

**(b) fix — `retain_graph`.** The construction in R1.6 already computes
the targeted `grad_rest[target]` call with `retain_graph=True` BEFORE
`total_loss.backward()` (which is the LAST graph traversal and does not
need to retain). A standalone smoke sub-test recomputing `grad_ce_only`
on the same graph must do the same:

```python
grad_rest_only = torch.autograd.grad(non_ce, [target_w], retain_graph=True, allow_unused=True)[0]
total_loss.backward()   # populates the FULL combined .grad for every param, incl. target_w
# grad_ce_only, derived (NOT a separate ce_loss.backward() call):
grad_ce_only = target_w.grad.detach().clone() - (grad_rest_only if grad_rest_only is not None else 0.0)
```

**(c) is rewritten as a two-tier check, and F3's own numerical
argument is extended to `embed` — a point DRAFT-R0 and the attack both
missed.** F3 established that `entity_adapter`/`ncr_head` are NOT
bit-identical between the subtraction/direct forms (float
non-associativity at the shared `o_raw`/`Z` node, ~4.5e-8 max diff,
`allclose`-true). **`embed.weight` sits at exactly the same kind of
shared node** — it is reached by CE (three routes) AND by aux/ortho (via
the same `o_raw`/`Z` path), so `grad_ce_only` derived by subtraction
above is subject to the identical non-associativity, NOT bit-identical
to a hypothetical standalone `ce_loss.backward()`. Concretely this means
**the D-A2 has-teeth identity check below must use the same tolerance
bound as `entity_adapter`, never `torch.equal`, for `embed` itself** —
a correction to DRAFT-R0's own §5(2)(b), which asked for exact equality
on `embed.weight.grad` and would have failed for the right reason for
the wrong assumed reason.

```python
EXACT_TIER = {"backbone_block_params", "read_injector"}   # aux/ortho never reach these -- torch.equal
TOL_TIER = {"embed", "entity_adapter", "ncr_head"}          # shared o_raw/Z node -- allclose bound

RTOL, ATOL = 1e-5, 1e-6   # pinned; entity_adapter's own measured max|diff| 4.470e-08 sits far inside this

def check_scope_preserved(off_grads: dict, on_grads: dict, param_tier: dict) -> None:
    for name, tier in param_tier.items():
        g_off, g_on = off_grads[name], on_grads[name]
        if g_off is None and g_on is None:
            continue
        if tier == "EXACT_TIER":
            assert torch.equal(g_off, g_on), f"{name}: expected EXACT match, got a difference"
        else:
            assert torch.allclose(g_off, g_on, rtol=RTOL, atol=ATOL), f"{name}: outside allclose bound"
            rel = (g_off - g_on).abs().max() / g_off.abs().max().clamp_min(1e-12)
            assert rel < 1e-5, f"{name}: relative diff {rel:.3e} exceeds 1e-5"
```

**Forced-fail negative test (run to completion, not merely written —
repo's own standing rule):** deliberately mis-assemble one parameter
(e.g. skip subtracting `grad_rest[target]` from `target_w.grad`, i.e.
simulate a no-op cut) and confirm `check_scope_preserved`'s companion
identity check on `target_w` itself (not the scope-preserved check,
which is about OTHER params) fires `AssertionError`. Ship this as
`test_negative_noop_cut()` in the same smoke module; the build agent
runs it once at build time and pastes the raised traceback into the
build report as proof the check has teeth, mirroring this repo's
already-adopted convention for exactly this failure class.

**Docstring fix:** delete "bit-for-bit what a normal `total_loss.backward()`
would have produced" from `assemble_closed_grads_`'s docstring (R1.6);
replace with "identical for backbone-block and read_injector params;
within a pinned `allclose` bound (rtol 1e-5, atol 1e-6) for
embed/entity_adapter/ncr_head, per float non-associativity at the
shared o_raw/Z node — see D-F3."

---

### R1.5 — D-F4: eval harness. `run_repl_wave3.sh`, written whole, against the box's CURRENT state (re-verified, not the attack's snapshot)

**Correction to the attack's own premise:** `run_repl_wave2.sh` on the
box has been edited since the attack round (file mtime `Aug 18 08:30`,
after the attack's `591301a` commit at `05:55`). The CURRENT version
already loops `compA compB compD primary` with a `case` statement
(`compB|compD) FZ="";; *) FZ="freeze";;`) — the attack's F4 quote (`for
tag in compA compB primary`, hardcoded `if [ "$tag" = "compB" ]`) is
stale. Verified directly: `md5sum` `dfba70bccd318074d95dbe698c40c77b`;
`~/ncr_writecond/rescore.log` and `repl_w8.log` show it has already
scored `compD` and re-scored two stale `compB` cells. `compE` and
`compE_adapter` are still absent from the loop (confirmed: `grep -i
compE ~/ncr_writecond/results/*.json` → no matches) — F4's core defect
(compE invisible) is real and current; only the exact quoted diff was
stale. `run_repl_wave3.sh` below is written against the box's actual
current script, not the attack's quoted snapshot:

```bash
#!/usr/bin/env bash
# run_repl_wave3.sh -- D-F4: adds compE / compE_adapter to the tag loop,
# a per-tag freeze map (both trainable-adapter targets => FZ=""), the
# /ephemeral/embed_path_ckpts root, the correct seed ranges, loud
# MISSING-CKPT, and re-score-not-skip (a stale eval record surviving a
# newer checkpoint is exactly what caused the #12/#13 stale-eval incident).
set -u
cd /home/nvidia/ncr_writecond
export CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-0}
OLD=/home/nvidia/ncr_g3b31_contrastive/results
MID=/ephemeral/reseed_ckpts
NEW=/ephemeral/embed_path_ckpts
declare -A SEEDS=( [compA]="1 24" [compB]="1 24" [compD]="1 24" [primary]="1 24" \
                    [compE]="1 8" [compE_adapter]="9 12" )
declare -A FZ=( [compA]="freeze" [compB]="" [compD]="" [primary]="freeze" \
                [compE]="" [compE_adapter]="" )
scored=0; missing=0; rescored=0
for tag in "${!SEEDS[@]}"; do
  read -r lo hi <<< "${SEEDS[$tag]}"
  PREFIX="mob_gembed_${tag}"; [ "$tag" = "compA" -o "$tag" = "compB" -o "$tag" = "compD" -o "$tag" = "primary" ] && PREFIX="mob_g3b31_${tag}"
  for s in $(seq "$lo" "$hi"); do
    NAME="${PREFIX}_s${s}"
    CK=""
    for cand in "$NEW/${NAME}_ckpts/${NAME}.ckpt.pt" "$MID/${NAME}_ckpts/${NAME}.ckpt.pt" "$OLD/${NAME}_ckpts/${NAME}.ckpt.pt"; do
      [ -f "$cand" ] && { CK="$cand"; break; }
    done
    RES="$OLD/${NAME}.json"
    OUT="/home/nvidia/ncr_writecond/results/writecond_premise_REPL_${tag}_s${s}.json"
    if [ -z "$CK" ]; then
      [ -f "$RES" ] && { echo "MISSING-CKPT ${tag}_s${s} (results JSON exists but no checkpoint in any of NEW/MID/OLD)"; missing=$((missing+1)); }
      continue
    fi
    if [ -f "$OUT" ] && [ "$CK" -ot "$OUT" ]; then
      continue                                    # ckpt not newer than the eval -- genuinely already scored
    fi
    [ -f "$OUT" ] && rescored=$((rescored+1))
    echo "=== scoring ${tag}_s${s} ($CK) ==="
    /home/nvidia/tdenv/bin/python3 pbe_repl "$CK" "${tag}_s${s}" "${FZ[$tag]}" && scored=$((scored+1)) || echo "FAILED ${tag}_s${s}"
  done
done
echo "SCORED=$scored RESCORED=$rescored MISSING=$missing"
# self-check (D-F4's own ask): FAIL LOUDLY if ANY expected arm/seed produced no output at all.
fail=0
for tag in "${!SEEDS[@]}"; do
  read -r lo hi <<< "${SEEDS[$tag]}"
  for s in $(seq "$lo" "$hi"); do
    OUT="/home/nvidia/ncr_writecond/results/writecond_premise_REPL_${tag}_s${s}.json"
    [ -f "$OUT" ] || { echo "SELF-CHECK FAIL: no output ever produced for ${tag}_s${s}"; fail=1; }
  done
done
[ "$fail" -eq 0 ] && echo "SELF-CHECK PASS: every expected arm/seed has an output"
echo REPL_WAVE3_DONE
```

**Negative test (run to completion, per D-F4's own ask):** temporarily
add a bogus entry `[compZZZ]="1 1"` with no matching checkpoint anywhere
and confirm `MISSING-CKPT compZZZ_s1` prints and the final self-check
prints `SELF-CHECK FAIL` naming it — then remove the bogus entry. This
directly exercises the loud-fail path the old script's silent `[ -f "$OUT"
] && continue` could never trigger.

Note the `[ "$CK" -ot "$OUT" ]` freshness check (bash `-ot` = "older
than") replaces bare skip-if-exists with re-score-if-the-checkpoint-is-
newer — directly closing the adjacent defect the attack named (the
stale-eval incident's own mechanism).

---

### R1.6 — D-F5 + D-F2 + D-A2, combined: cut AFTER `clip_grad_norm_`, has-teeth as a ratio, generalized to `--close-target={embed,entity_adapter}`

**Why these three collapse into one function.** D-F5 requires the clip
coefficient to be computed from the SAME global norm compB's own
backward() would produce — which means `target_w.grad` must be the FULL
combined gradient (bit-identical to `backward()`, not the split form)
at the moment `clip_grad_norm_` runs, and only THEN does the cut happen.
The simplest way to guarantee the combined side is genuinely
`backward()`'s own output (not a re-assembled approximation subject to
F3's tolerance) is to literally call `total_loss.backward()` for the
full combined gradient, and separately take ONE targeted
`torch.autograd.grad(non_ce, [target_w], retain_graph=True)` call before
it (order matters for graph retention, not for values) to get
`grad_rest[target]` alone — never re-deriving `grad_ce` via a second
`autograd.grad(ce_loss, ...)` call at all. This also means D-F2's
`allow_unused`/`None`-set assertion only needs to check ONE parameter
(the target), not every param in `all_params`, simplifying that check
too.

```python
def assemble_closed_grads_(arm: dict, total_loss: torch.Tensor, ce_loss: torch.Tensor,
                            aux_loss, ortho_loss, aux_read_loss_weight: float,
                            ortho_reg_weight: float, close_target: str,
                            max_norm: float = 1.0) -> dict:
    """--close-target={embed,entity_adapter}: cuts aux+ortho's marginal
    gradient contribution into exactly ONE named parameter, AFTER
    clip_grad_norm_ has already run over the FULL combined gradient (D-F5)
    -- so every OTHER parameter receives compB's own post-clip gradient,
    to within D-F3's pinned allclose bound, and the two arms differ in
    target_w.grad alone. Replaces total_loss.backward() entirely when
    close_target is set; the caller must NOT also call .backward().

    non_ce built from compute_arm_losses's own returned aux_loss/ortho_loss
    tensors (D-F2) -- never by total_loss - ce_loss subtraction, which
    re-walks the whole backbone with a seeded-zero gradient (F2, executed).

    Returns {"cut_active": bool, "conduit_ratio": float,
             "clip_coef": float, "target": str}."""
    target_w = {"embed": arm["backbone"].embed.weight,
                "entity_adapter": arm["integ"].entity_adapter.weight}[close_target]
    all_params = [p for p in (list(arm["backbone"].parameters()) +
                               list(arm["ncr"].parameters()) +
                               list(arm["integ"].parameters())) if p.requires_grad]

    non_ce = _non_ce_term(aux_loss, ortho_loss, aux_read_loss_weight, ortho_reg_weight)
    if non_ce is None:                      # both branches inactive this step -- nothing to cut
        total_loss.backward()
        return {"cut_active": False, "conduit_ratio": 0.0, "clip_coef": 1.0, "target": close_target}

    grad_rest_target = torch.autograd.grad(non_ce, [target_w], retain_graph=True, allow_unused=True)[0]
    total_loss.backward()                   # populates the FULL combined .grad for every param, incl. target_w

    # D-F2 None-set assertion, narrowed to the one param that matters here:
    if grad_rest_target is None:
        return {"cut_active": False, "conduit_ratio": 0.0, "clip_coef": 1.0, "target": close_target}

    combined_before_clip = target_w.grad.detach().clone()
    grad_ce_target = combined_before_clip - grad_rest_target      # derived, NOT a second ce_loss.backward() (D-F3: tolerance-tier, not exact)
    conduit_ratio = (grad_rest_target.norm() / grad_ce_target.norm().clamp_min(1e-12)).item()   # D-A2(i)

    finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in all_params)
    if not finite:
        return {"cut_active": False, "conduit_ratio": conduit_ratio, "clip_coef": float("nan"), "target": close_target}

    total_norm_before = torch.nn.utils.clip_grad_norm_(all_params, max_norm)   # scales EVERY param's .grad in place,
                                                                                 # incl. target_w's FULL combined grad --
                                                                                 # same clip coefficient compB itself gets (D-F5)
    combined_after_clip = target_w.grad.detach().clone()
    clip_coef = (combined_after_clip.norm() / combined_before_clip.norm().clamp_min(1e-12)).item()   # exact applied ratio,
                                                                                                          # not PyTorch's internal formula re-derived by hand

    target_w.grad.sub_(grad_rest_target * clip_coef)     # remove aux+ortho's (equally clipped) share -- AFTER clipping
    grad_ce_target_clipped = grad_ce_target * clip_coef
    # D-A2(ii): per-step identity check, tolerance-tier per D-F3 (embed/entity_adapter both sit on the shared o_raw/Z node)
    assert torch.allclose(target_w.grad, grad_ce_target_clipped, rtol=1e-5, atol=1e-6), (
        f"close_target={close_target}: post-cut grad does not match CE's own (clipped) share within tolerance")
    return {"cut_active": True, "conduit_ratio": conduit_ratio, "clip_coef": clip_coef, "target": close_target}
```

**Has-teeth assertion, D-A2's ratio replacing the old absolute floor:**

```python
def assert_conduit_has_teeth(grad_diag: dict, step: int, min_ratio: float) -> None:
    """D-A2(i): min_ratio is PINNED FROM A BUILD-TIME SMOKE MEASUREMENT
    (never guessed -- this repo's own 'measured, not just estimated'
    rule), e.g. the mean conduit_ratio over the first 50 real training
    steps of a full_graft cell with the flag active, times a safety
    margin (e.g. 0.5x that measured value) -- NOT the old absolute
    'min_conduit_norm=1.0' floor, which cannot detect the vacuous-pass
    mode the attack found (a large CE norm can swamp a genuinely small
    aux+ortho norm and still clear an absolute floor). Called every step
    close_target is active; skipped (not asserted True) on the rare
    'cut_active=False' step so this cannot reproduce m2's contradiction."""
    if not grad_diag["cut_active"]:
        return
    assert grad_diag["conduit_ratio"] > min_ratio, (
        f"step {step}: HAS-TEETH FAILED -- conduit_ratio={grad_diag['conduit_ratio']:.4f} "
        f"<= pinned min_ratio={min_ratio:.4f}; the flag may be closing an "
        f"already-negligible path")
```

**Training loop change** (replaces lines 1334-1335, inside the
`for arm_name, read_ablate in (...)` block at line 1317, confirmed
fresh against the box today):

```python
total_loss, ce_loss, aux_loss, ortho_loss, o_raw, aux_components = compute_arm_losses(
    arm, batch, read_ablate, tf_this_arm, aux_read_loss_weight, arm_name == "full_graft",
    ortho_reg_weight, aux_loss_type, contrastive_temperature)
opt.zero_grad()
if close_target and arm_name == "full_graft":
    grad_diag = assemble_closed_grads_(arm, total_loss, ce_loss, aux_loss, ortho_loss,
                                        aux_read_loss_weight, ortho_reg_weight, close_target)
    if grad_diag["cut_active"]:
        assert_conduit_has_teeth(grad_diag, step, min_ratio=PINNED_MIN_RATIO)   # from build smoke
    step_close_diag = grad_diag                       # logged into rec, new field
    # NOTE: clip_grad_norm_ and opt.step() are ALREADY DONE inside assemble_closed_grads_
    #       for this arm -- do NOT re-run the shared finite/clip/opt.step() block below for it.
else:
    total_loss.backward()
    # falls through to the existing shared finite/clip/opt.step() block, unchanged
```

This changes the shared block's control flow slightly (the close-target
arm now clips+steps INSIDE `assemble_closed_grads_`, not in the shared
block below it) — flagged explicitly here because it is the one place
D-F5's requirement ("cut after clip") forces a restructuring beyond a
drop-in replacement of `.backward()`.

---

### R1.7 — M1/M2/M2b: resolved together in R1.1. See there.

---

### R1.8 — D-M3: honest prior, restated as the new §1 paragraph

**Insert into §1, after "Why this is the first interventional test...":**

> **Prior, stated honestly (D-M3, informed further by R1.1's executed
> pre-test).** §G3-B32's own recorded verdict is that "the depth path
> itself (binexp = power iteration toward Z's top singular direction)
> destroys read discriminability in-LM independent of the aux loss" —
> NOT the target-space mechanism. compB's own measured TPC_fg
> (0.196-0.228) sits far below the 0.50 tripwire that would indicate the
> §G3-B26 pathology is firing in this arm at all. `COMPB_DRIFT_ANALYSIS.md`
> leg (a) measured collapse and deep composition as POSITIVELY
> associated (ρ=+0.4643, p=0.0392, n=20) — more collapse, not less, goes
> with better composition, opposite the naive direction. And R1.1's own
> swap evidence shows the rescue from perturbing either `embed` or
> `entity_adapter` alone is symmetric, not asymmetric in embed's favor.
> Taken together, this is registered as a plausible-but-contra-indicated
> mechanism, not a likely-positive one: the chain that survives all of
> this evidence is narrower than DRAFT-R0 stated it — an interaction
> between a trainable adapter's own co-adaptation and embed's openness,
> where NEITHER partner's own value is uniquely privileged as the
> carrier, which is exactly why `compE_adapter` (R1.1) is now a mandatory
> co-arm rather than a placebo.

---

### R1.9 — D-M4: bands as a strict first-match ladder (compE, D-M5's paired seeds folded in)

Archived compB paired subset (s1-s8, reused, zero new cost): `[0.6484,
0.6172, 0.9531, 0.8047, 0.7266, 0.9727, 0.7773, 0.7227]`, median
**0.75195**. (Distinct from the full n=20 archive's median 0.7246 —
both are cited explicitly wherever used, never conflated.)

Evaluated top-down, **first match wins, no ORs**:

1. **WIN:** `compE` (n=8, seeds s1-s8) `min > 0.9727` (compB's own
   archived n=20 max) **AND** median ≥ 0.90 **AND** paired Wilcoxon
   signed-rank p < 0.05 (one-sided, compE > compB, on the 8 matched
   `(compE_si, compB_si)` pairs).
2. **PARTIAL:** median ≥ **0.85195** (= compB's own paired s1-s8 median
   0.75195 + 0.10, pinned numerically here — replaces DRAFT-R0's
   unreconciled literal 0.80, which contradicted its own stated +0.10
   rationale, D-M4's own finding).
3. **NULL:** neither of the above. (Subsumes "no significant shift" and
   "shift present but below 0.85195 or p > 0.05" as descriptive
   sub-notes on the NULL read-out, not separate OR-branches that could
   independently fire.)

Unpaired cross-check against the full archived n=20 (always reported
alongside the paired read, never in place of it): exact Mann-Whitney,
n=8 vs n=20, `C(28,8) = 3,108,105`, best achievable two-tailed p =
**6.435e-07** (corrected — DRAFT-R0's `1.9e-6` was `2/C(25,8)`, the
wrong denominator, off by 3.0x). Critical `U ≥ 119/160` for p ≤ 0.05.

**compE_adapter (n=4, seeds s9-s12, paired against archived compB
s9-s12 = `[0.7383, 0.7617, 0.6797, 0.6875]`, median 0.7129):**
directional only, per the original placebo's own disclosed n=4 limit
(best exact two-tailed p at n=4 = 2/C(24,4) = 1.88e-4). Read rule (new —
D-M4 flagged the old placebo had none): **ADAPTER-LEVERAGE CONFIRMED**
if `compE_adapter` median ≥ `compE` median − 0.05 (comparable rescue,
matching R1.1's finding that adapter alone is as strong a lever as
embed alone); **EMBED-SPECIFIC** if `compE_adapter` median is more than
0.10 below `compE`'s (embed's own cut turns out to matter in a way
adapter's does not, once BOTH are actually run at wave scale rather
than via a full-reset proxy); otherwise **AMBIGUOUS**, reported as such,
not forced into either label.

---

### R1.10 — D-M5: power, seeds, corrected p (folded into R1.9's ladder above)

compE moves to seeds **s1-s8** (paired), superseding DRAFT-R0's s21-s28.
The design's own stated reason to avoid s1-s8 ("so no seed number
collides with an existing archived cell") is a naming concern only,
fully solved by `compE`'s own distinct `cell_id`/`ckpt` path
(`mob_gembed_compE_s{1..8}`, `/ephemeral/embed_path_ckpts/`) — the
SEED NUMBER may repeat; the ARCHIVE KEY may not, and does not.
`compE_adapter` uses s9-s12, paired against the same archive, without
seed overlap with `compE` itself. WIN's separation criterion is kept
(rather than lowered) because R1.9's ladder already makes PARTIAL the
practically-reachable success band (D-M5(iii)'s own ask) via the
pinned 0.85195 floor, so WIN staying rare-but-meaningful does not strand
the design's own anticipated outcome in an unreachable band.

---

### R1.11 — D-M6: runner parity — one premise corrected, two exposures fixed

**Correction (found by re-reading the box directly, not assumed):**
`ls -la ~/ncr_writecond/` shows `ncr_lm_wave1_runner.py` and
`ncr_lm_wave1_smoke.py` are **symlinks** to
`~/ncr_g3b31_contrastive/{same name}` (`Aug 14 02:06`, predating the
attack), not independent copies. `pbe_repl.py`'s own
`sys.path.insert(0, dirname(__file__))` therefore resolves to the SAME
inode the design pins and patches — patching
`~/ncr_g3b31_contrastive/ncr_lm_wave1_runner.py` automatically patches
what the scorer imports too. **This discharges the "two independent
copies" sub-concern of M6.** The other two exposures are real and
unaffected by this correction:

- **Flag-OFF parity smoke, required before wave-1 launches:** one seed
  reserved outside every existing/planned archive range (`seed=9999`,
  short-run only, ~200 steps, discarded after the check — not saved to
  any tracked results path), `close_target=None`. Assert the full loss
  trajectory (`ce_loss.item()` per step) and both arms' final
  `state_dict()`s are `torch.equal` against an unpatched run of the same
  seed/steps. Any diff means the patch changed OFF-path behavior and the
  whole campaign's prior archive is suspect.
- **`RUNNER_TAG` pinned invariant** (confirmed at line 281,
  `RUNNER_TAG = "ncr_gate3_wave1_runner_v1"`): add a code comment at its
  definition — `# PINNED. Bumping this silently un-loads EVERY archived
  compB/compA/compD/primary checkpoint via load_checkpoint's own assert
  (line ~1130). Do not change for this build.` — and a build-time
  assertion `assert RUNNER_TAG == "ncr_gate3_wave1_runner_v1"` at the top
  of `main()`.

---

### R1.12 — D-M7: bookkeeping

- `close_target` (the string, `None`/`"embed"`/`"entity_adapter"`)
  added to `rec["config"]`, `save_checkpoint`'s `ckpt = {...}` dict, and
  a resume-mismatch assert mirroring the existing seed-trap/freeze-trap
  pattern exactly (same shape as DRAFT-R0's own §2.5, just keyed on the
  string instead of a bool).
- Rounded literals corrected: `252/256 = 0.984375` (frozen floor),
  `249/256 = 0.97265625` (compB's own archived max) — used exactly, not
  `0.9844`/`0.9727`, throughout R1.9-R1.10.
- Single source of record for n: **`EXPERIMENT_LOG.md` #13** (frozen
  n=18, compB n=20) is the ONLY n citation from here forward; the
  paired subsets (s1-s8, s9-s12) drawn from that same n=20 archive are
  cited as subsets of #13, not as independent counts.

---

### R1.13 — D-M8: placement (predicted, not measured — smoke-confirmation still required)

No training was run in this pass (R1.1 is eval-only), so nothing here
is a substitute for the build-time VRAM/SM smoke this repo's own rule
requires before a full launch — this section states a PREDICTION with
its reasoning shown, to be confirmed or corrected by that smoke.

- **VRAM.** G3-B31's own measured baseline: 6.86 GB/cell. D-F2's fix
  removes the second full-model gradient tensor (8 exact-zero
  backbone-sized tensors) that DRAFT-R0's subtraction form would have
  materialized — that specific +overhead is gone. What remains:
  `retain_graph=True` still holds the FULL activation graph alive across
  the targeted `autograd.grad` call and the subsequent `backward()` (the
  flag does not let you retain only part of a graph) — a real,
  smaller-than-DRAFT-R0's-own but nonzero cost. Predicted band: **7-9
  GB**, well inside 80GB H100 headroom.
- **SM utilization / packing.** No fresh contention-pricing analysis was
  run this pass; G3-B31's own no-packing ruling (one cell per GPU, 73-80%
  SM) is carried forward unchanged rather than overridden without one —
  flagged here as inherited, not re-derived.
- **Cells-per-GPU:** 1 (per the above).
- **Wall-clock critical path:** 12 cells over 8 GPUs, one cell/GPU/slot
  → 2 sequential slots, ≈ 2× the per-cell wall-clock (~1.0-1.3h/cell) ≈
  **2-2.6 hours wall-clock**, GPU-h total unchanged at 9.84-13.20 (R1.3).
- **Required before full launch:** one real-CUDA smoke cell (not the
  reserved parity seed above — a genuine `close_target=embed` cell run
  to ~500 steps) measuring actual peak VRAM and per-step wall-clock,
  used to both confirm this band and pin `PINNED_MIN_RATIO` (R1.6) from
  a real measured `conduit_ratio`.

---

### R1.14 — minors (m1-m5), fixed

- **m1 (`list.index` footgun):** replaced by construction — R1.6's
  `assemble_closed_grads_` never builds an `all_params.index(...)` at
  all; `target_w` is selected directly from the `{"embed": ...,
  "entity_adapter": ...}` dict, no positional lookup anywhere.
- **m2 (`cut_active` contradiction):** resolved by construction —
  `assert_conduit_has_teeth` (R1.6) returns early (no assertion) when
  `cut_active` is `False`, instead of the old unconditional
  `assert grad_diag["cut_active"]` the caller used to run regardless.
- **m3 (line-number drift):** re-verified fresh against the box today
  (all citations above pulled by live `grep`/`sed`, not carried over):
  `compute_arm_losses` def 768 (unchanged, was correct); `ortho_regularization_loss(Z)`
  call site 850, additive sum 848/851 (confirmed, matches the attack's
  own correction); per-arm loop header 1317; `total_loss.backward()`
  call 1335; clip block 1354-1359 (`clip_grad_norm_` itself at 1356, one
  off from the attack's own "1357" — files under active development
  drift between reads; cite the read date next to any future line
  number, this document's is 2026-08-18); `nn.init.normal_` in
  `lm_pretrain_rd.py` at 1229; tied head `F.linear` at 1310;
  `RUNNER_TAG` def at 281; `build_arm` 861; `load_checkpoint` 1122;
  `restore_arms_and_opts` 1136; `eval_arm_at_hops` 934.
- **m4 (arbitrary absolute floor):** replaced by D-A2's ratio (R1.6),
  not a residual absolute number.
- **m5 (per-step `.item()` host sync):** `assert_conduit_has_teeth` and
  the ratio computations in R1.6 already only call `.item()` on small
  scalars once per step for the close-target arm only (not every param,
  not every step for the other arm) — logged every step is now cheap
  enough (a handful of scalar syncs, not a per-parameter loop) that the
  every-N-steps relaxation is no longer necessary; kept at every-step
  for has-teeth's own "loud, immediate" purpose.

---

### R1.15 — What Rev-1 could NOT discharge, stated plainly

- **`PINNED_MIN_RATIO` (R1.6) has no number yet.** It cannot be
  pinned from R1.1's eval-only pre-test (which never calls `backward()`
  or touches `non_ce`'s gradient at all) — it requires a real training
  step's forward+backward on live data. This is explicitly deferred to
  the build-time smoke (R1.13), not guessed here.
- **R1.1's swap evidence is coarse by construction (its own disclosed
  caveat, restated once more for emphasis):** it settles the
  SPECIFICITY question (embed vs. entity_adapter — genuinely
  undetermined, hence `compE_adapter` is now mandatory) but does not
  and cannot settle compE's expected MAGNITUDE, since compE's actual
  gradient trim is a much smaller perturbation than a full reset.
- **No fresh SM-utilization/contention-pricing pass was run this
  round** (R1.13) — G3-B31's prior no-packing ruling is carried forward,
  not re-validated at this design's own (smaller, D-F2-reduced) memory
  footprint.
- **The flag-OFF parity smoke (D-M6) has not been run** — specified
  precisely (seed 9999, ~200 steps) but not executed in this pass, since
  it requires the patched code to exist first (it is a build-time gate,
  not a pre-build one, unlike R1.1).
