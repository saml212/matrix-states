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
