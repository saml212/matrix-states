# Stage G Design — Where the Per-FLOP Signal Loss Occurs in Matrix-Token LMs

> **STATUS: CLOSED (one of STATE.md's five closed 2026-07-01→03 programs).**
> Verdict (§14.6, full mechanistic account §14.5): the matrix-vs-vector BPB
> gap has a **named, dominant mechanism** — the Kronecker-separable
> restriction forced by `RowThenColProjection`, not matrix-valued tokens
> per se. Relaxing it to a dense rank-swept bottleneck recovers ~64-70% of
> the gap at matched parameters (every seed ≥0.5 `recovered_frac`) while
> using *fewer* FLOPs than what it replaces. Most of the apparent "beats a
> vector baseline" inversion at higher rank is a capacity effect, not a
> matrix-side win (a capacity-matched vector control beats it by 0.092 BPB).
> The per-FLOP tax against a capacity-matched vector reference survives
> everywhere measured (≈16.5×, down from the original 54.7× once corrected
> for the never-actually-FLOPs-matched accounting bug this design also
> caught, §10). H_e (task-representation-mismatch, §1) is a separate,
> narrower gated hypothesis this document also defines; its Wave C gate is
> now triggered per the verdict above (confirmed-but-narrow dominant site) —
> see STATE.md "Chapter 2 — STATUS" for that thread's own status.

**Drafted 2026-07-01, before any code changes.** Status: design only — no
model/training code is written here. This document diagnoses a *different*
honest negative than Stage 0 (`matrix-thinking/chapter2/STAGE0_DESIGN.md`):
Stage 0 explained why the Task D/E associative-memory encoder (`model_v4.py`,
~171K params, h=64, single-Z-bottleneck) fails to train at `d≥32`, and found
the failure was largely a **step-budget artifact** (§12 of that document —
6/6 seeds at d=32 transition from "flat, dead" to a real climb once the
budget was extended 2.5×, something the original 8K-step sweep could never
have shown). This document targets the LM-era finding — **BPB 1.67 (matrix
d=32) vs BPB 0.87 ("FLOPs-matched" LoopFormer)**, `STATE.md` "Honest negative
results," `EXPERIMENT_LOG.md` Runs 13–15 — which has never been diagnosed at
the component level. Precedent taken seriously: `STAGE0_DESIGN.md` names its
own method as "read the forward pass line by line and derive concrete,
falsifiable mechanisms... before proposing interventions," not a
hyperparameter sweep. Stage G does the same for the LM comparison, and finds
— *before any GPU runs* — that reading the two exact scripts surfaces
confounds nobody has named yet (§2.3, §2.4) that sit logically upstream of
every candidate the brief asked to rank.

> **CHANGELOG — 2026-07-01, revision 2 (post-attack-round + H_h
> verification).** An independent adversarial review of revision 1 returned
> BUILD WITH FIXES (2 FATAL, 4 MAJOR, minors); a separately-commissioned
> independent verification of H_h landed during the revision (finding V1).
> All findings are addressed in this revision:
> **F1 (FATAL — budget arithmetic)** — revision 1's §7 priced Wave B's
> Regime-1 spot-check at ~2.5 GPU-h/cell against its own §5.4 derivation of
> 22.5 GPU-h/cell (Run 12: 168.7 min × 8 GPUs), a 9× error that made the
> "core well under 40 GPU-h" claim false (honest recompute at full scope
> ≈ 60 GPU-h). Fixed: §7 fully recomputed from the measured Regime-1 unit
> (7.5 GPU-h per 1,000 steps). Combined with V1's re-prioritization, the
> Regime-1 budget slot is reassigned to the now-required H_d continuation
> cell (option (b)+(c) of F1's menu: reduced-step Regime-1 work with
> explicit caveats, plus downgraded transfer language in §1/§10; the
> winning-intervention transfer spot-check is demoted to the stretch list).
> New core: **34.6 GPU-h ≤ 40**, headroom no longer assumed — stretch items
> fund only from measured under-run.
> **F2 (FATAL — `TiedMultiProbeHead` is dead code that doesn't tie)** — its
> `forward()` derives probes from fresh `probe_mix_u/v` Linear layers and
> never reads the stored `embed_u`/`embed_v` references, and its `out`
> projection is a full untied vocab-sized Linear; wired in as-is, H_f would
> have produced a near-null `recovered_frac` and been silently, falsely
> recorded as falsified. Fixed: §4 item 4 is now a **blocking pre-Wave-A
> build task** specifying a genuinely-tied head (two pre-specified forms)
> plus a mandatory tie-verification unit test (§12 build requirement (v)).
> **F3 (MAJOR — H_a/H_c downgrades unsupported)** — Runs 18/22 never varied
> the embedding (both arms share the outer-product embed; Run 18's flat arm
> additionally swaps in a 12.9M-param untied Linear head, an undisclosed
> second confound; Run 25's A-vs-E head comparison is depth-confounded).
> `EXPERIMENT_LOG.md`'s "definitive embedding vs operations test" label for
> Run 22 is wrong on the embedding half — it is an operations-only test.
> H_a/H_c priors restated as untested-at-this-axis and restored to equal
> Wave A screening weight (§1, §2.5, §3, §4, §6.1).
> **F4 (MAJOR — Regime 2's `d` unpinned)** — every cited Regime-2
> throughput precedent is d=16, but H_g's 10.7× width ratio and every §5.1
> number are d=32 quantities (at d=16 the width ratio is 2.7× — a different
> regime). Regime 2 pinned to **d=32**; Wave −1 calibrates throughput at
> d=32; the Run 17/19 throughput conflation fixed (§5.4, §7).
> **F5 (MAJOR — missing hypothesis)** — LoopFormer's blocks receive
> per-loop AdaLN timestep conditioning (`c = time_embedder(ti) +
> dt_embedder(dt)`); the matrix `ThinkingBlock` has no analogous signal.
> New **H_i** (iteration conditioning) with its own Wave A cell; H_g's
> depth conclusion scoped conditional on it (§1, §3, §4, §6.1).
> **F6 (MAJOR — missing hypothesis, upstream of all)** —
> `round2_matrix_script.py` uses default `nn.Embedding` init (std 1.0) for
> `embed_u/v`, so `M = u⊗v` enters the residual stream at entry-std ≈ 1 vs
> LoopFormer's explicit 0.02 init — the exact CLAUDE.md hard-rule bug class
> ("products have std=σ²"), a ~50× step-0 activation-scale mismatch in the
> ORIGINAL comparison. New **H_j** (embedding init scale): a free Wave −1
> step-0 instrumentation check plus a cheap Wave A cell (§1, §3, §4, §6.1,
> §7).
> **V1 (H_h independent verification — landed mid-revision).** Mechanism
> CONFIRMED: an independent from-scratch derivation gives Matrix **230.6M**
> vs LoopFormer **15.45M** FLOPs/token → **14.9×** (non-causal convention)
> / **11.8×** (causal-exact), far below the 28–32× throughput ratio used to
> size Run 14. It also caught a second bug in revision 1's own §5.1
> arithmetic (AdaLN/timestep conditioning counted per token; the code
> computes it per (batch, loop, block) — amortizing over L=512 gives the
> verified numbers). **Directional correction, critical:** the cited
> LoopFormer best (BPB 0.87 / PPL 10.6) occurred at **step 21,500** — not
> "~step 40K" as `EXPERIMENT_LOG.md` Run 14 states (re-verified this
> session against the raw `loopformer_96K_full.log` `*BEST*` markers: best
> at eval step 21,500; loss spike ~31K; GN=inf divergence ~52K; killed at
> step 82,600 with no DONE and 0-byte SUMMARY/results.json) — so
> LoopFormer's best consumed only **~0.48–0.61×** of Matrix's total
> training FLOPs. The comparison was never FLOPs-matched *in either
> direction*, and the miss actually **favored Matrix on compute**. H_h's
> section rewritten accordingly (§1, §2.4); H_d elevated to the single
> missing measurement, with a new required Regime-1 cell (§4 item 2, §7).
> `EXPERIMENT_LOG.md` Run 14's "~step 40K" and "~653K TFLOPs" figures are
> being corrected separately; this design cites step 21,500 and the
> verified FLOPs table (§5.1) throughout.
> **Minors** — H_g's width attribution explicitly declared NOT falsifiably
> tested in Stage G (§4 item 5); H_d gains a still-rising-at-cutoff →
> INCONCLUSIVE (not FALSIFIED) branch (§8); causal-mask 2× FLOP-convention
> footnote added (§5.1); Wave 0 cell-count arithmetic corrected; Run 17/19
> throughput conflation fixed (§5.4).

> **CHANGELOG — 2026-07-01, revision 2.1 (round-2 verify; pre-cleared
> fixes — design CLEAN TO BUILD once landed).** The round-2 verification
> confirmed all seven revision-2 fixes landed and the budget arithmetic,
> and returned 3 fresh MAJORs + 5 minors on the new text:
> **N1 (MAJOR — fallback is mainline)** — the Run 12 checkpoint is
> confirmed NOT locally recoverable (verifier: no `.pt` under
> `experiment-runs/8xh100-session1/`; the SSD's `best.pt` is a
> byte-agnostic-era model — pickle keys `byte_embed_for_expand`/`ln1`/
> `ln2`, not MatrixThinker; the retired pod volume is the only remaining
> hope). §4 item 2(a)'s fallback is now the **MAINLINE** planning
> assumption for the build phase, with a pre-registered aftermath
> paragraph: the freed 11.25 GPU-h flows down the stretch list in order,
> stretch item 2 re-anchors or drops (explicit text), and §1's
> diagnostic-pass wording notes the out-of-budget exit.
> **N2 (MAJOR — verdict scoping)** — the archived script saves only
> `raw_model.state_dict()` (no AdamW moments, no scheduler, no RNG state),
> so 0-R1 — if the checkpoint is ever recovered — is a moment-less,
> re-warmed continuation at 1.5× total budget and **cannot FALSIFY H_d**
> (Stage 0's own precedent needed 2.5–6× to date transitions). H_d's
> FALSIFY branch is scoped to arm (b) (from-scratch Regime-2 extended
> arms) ONLY; 0-R1 contributes CONFIRM or INCONCLUSIVE evidence only
> (§4 2(a), §8, §9 #11).
> **N3 (MAJOR — H_f×H_j confound)** — the tied head's form (i) bilinear
> logits at default std-1.0 init have scale ~O(30) → saturated softmax at
> step 0, and form (i) had no temperature: the H_f cell would have
> cratered for an H_j reason. Fixed: a learned scalar temperature added to
> form (i); an H_f rerun on the matched-init variant is pre-registered if
> the primary cell lands far below baseline (§4 item 4).
> **Minors** — N4: form (ii)'s concat corrected to feature-concat
> `[E_u, E_v] ∈ R^{vocab×2d}`, not row-stack. N5: the Regime-2
> standard-cell step count is pinned as a mandatory Wave −1 output, frozen
> before Wave 0 launches, so §8's `recovered_frac` is well-defined (§5.4,
> §7). N6: the data-integrity reference now points at the stats that
> survive in `round2_matrix_train.log` (train_tokens=2,188,386,413; vocab
> 50257; the sources line) — the archived `data_meta.json` is 0 bytes
> (§12(vi)). N7: 0-R1's intermediate evals trimmed to T=8 only (full
> T∈{1,2,4,8} sweep at final checkpoint only) so doubled eval frequency
> stays inside the priced 11.25 GPU-h (§7). N8: the drift canary is now
> explicitly a step-0 eval of the loaded checkpoint BEFORE any optimizer
> step (§9 #11).

---

## 1. The hypotheses, one sentence each

Ten named, falsifiable mechanisms (the brief's five, ranked, refined, and
extended by five found while reading the exact code — three at revision 1
(H_h, H_g, H_f), two added by the revision-2 attack round (H_i, H_j) — see
§2 for how each was derived):

> **H_h (FLOPs-accounting artifact — mechanism CONFIRMED pre-launch,
> direction corrected; V1).** The Run 14 "FLOPs-matched" LoopFormer step
> count (96,000) was derived from *measured wall-clock throughput ratio*
> (~28–32×), not an analytic FLOP count — the verified analytic ratio is
> **11.8–14.9×** (§5.1), with the difference plausibly H100
> kernel-launch/small-GEMM underutilization on the matrix side — **and**
> the cited LoopFormer best (BPB 0.87) occurred at step 21,500, before its
> divergence, meaning the comparison was never FLOPs-matched in either
> direction: LoopFormer's best consumed only ~0.48–0.61× of Matrix's total
> training FLOPs, so the existing headline comparison actually *favored
> Matrix on compute* (LoopFormer won ~1.9× BPB on roughly half the
> compute), and the true converged gap is **unmeasured**.

> **H_d (undertraining/budget artifact, the brief's candidate (d) — THE
> missing measurement; TEST FIRST per CLAUDE.md, elevated by V1).** The
> matrix run (3,000 steps, ~0.5 epoch of the 2.19B-token corpus, train PPL
> still steeply descending — 2,880→71.9 over the last 2,750 logged steps —
> when its LR schedule hit zero by design) was not converged within its
> budget; since LoopFormer's best-within-this-setup is already measured
> (PPL 10.6 at step 21,500, monotonically worse afterward from multi-epoch
> overfitting), Matrix's extended-budget BPB is the *single missing datum*
> needed to state the true gap — more steps would substantially close it
> (CONFIRM) or the gap persists with Matrix holding a growing compute
> advantage (a strictly *stronger* negative than the current headline).

> **H_j (embedding init scale, new — revision 2, F6; upstream of every
> other hypothesis).** `round2_matrix_script.py`'s `embed_u`/`embed_v` use
> PyTorch's default `nn.Embedding` init (std 1.0), so `M = u⊗v` enters the
> residual stream at entry-std ≈ 1 while LoopFormer explicitly inits all
> embeddings at std 0.02 — a ~50× step-0 activation-scale mismatch in the
> original comparison, and exactly the bug class CLAUDE.md's hard rule
> already documents ("outer-product embedding init: u,v std must be
> sqrt(target_std), not target_std — products have std=σ²"); if
> optimization dynamics at this scale mismatch account for part of the
> gap, every downstream architectural conclusion inherits the artifact.

> **H_g (architecture-config confound, new).** The Run 12–15 comparison held
> total *parameter count* roughly matched (~5.1–5.3M) but let two other
> axes diverge freely: effective channel width (matrix attention operates
> on flattened `d²=1,024`-dim vectors vs LoopFormer's `n_embd=96`) and
> effective depth (matrix: 8 *distinct*-weight layers × 8 iterations = 64
> layer-applications; LoopFormer: 2 *shared*-weight blocks × 8 loops = 16
> block-applications) — both inflate matrix FLOPs/token independent of any
> intrinsic matrix-vs-vector representational quality difference.

> **H_i (iteration conditioning, new — revision 2, F5).** LoopFormer's
> blocks receive explicit per-loop AdaLN timestep conditioning
> (`c = time_embedder(ti) + dt_embedder(dt)`) telling the shared weights
> which iteration they are executing, while MatrixThinker's `ThinkingBlock`
> receives no analogous signal across its 8 outer iterations — the vector
> model was given a mechanism the matrix model never had, independent of
> matrix-vs-vector representation.

> **H_f (weight tying, new).** LoopFormer ties `wte.weight = lm_head.weight`
> (Press & Wolf 2017, arXiv:1608.05859; Inan et al. 2016, arXiv:1611.01462
> — verify exact citation before external use per house convention), halving
> its embedding-table cost and gaining the well-documented small-model
> quality benefit of a copying/repetition prior; MatrixThinker's
> `MultiProbeHead` uses an **untied** `Linear(K, vocab)` output — the matrix
> side is denied a well-known, essentially-free win (and the codebase's
> nominal fix, `TiedMultiProbeHead`, is dead code that does not actually
> tie — F2, §4 item 4).

> **H_a (embedding bottleneck, the brief's candidate (a)).** Rank-1
> outer-product embeddings (`2d` DOF/token vs `d²` table entries) starve
> the representation of information the backbone needs. **Prior: UNTESTED
> at this axis (corrected in revision 2, F3).** Revision 1 downgraded H_a
> by citing Runs 18/22's T=1 results as evidence the embedding is strong —
> but both runs hold the outer-product embedding FIXED in both arms (they
> vary the downstream operations, not the embedding; Run 18's flat arm
> additionally swaps in a 12.9M-param untied Linear output head, a second
> confound), so **no experiment in this repo has ever varied the embedding
> axis**. H_a screens at full, equal weight in Wave A.

> **H_b (Kronecker-restricted attention/thinking projections, the brief's
> candidate (b), refined).** `RowThenColProjection` (`silu(A@M)@B`, shared
> `A,B ∈ R^{d×d}`) is **not** reshape-equivalent to a generic vector linear
> layer of matching flattened width. `vec(AMB) = (Bᵀ⊗A)·vec(M)` (standard
> Kronecker-product identity — Van Loan 2000, J. Comput. Appl. Math): this
> restricts every Q/K/V/O/gate projection to the ~`2d²`-parameter family of
> *separable* linear maps on the `d²`-dim flattened token, a small subset of
> the `d⁴`-parameter space of all linear maps on that space — while
> (because `rank(X⊗Y)=rank(X)·rank(Y)`) still permitting *full rank* as an
> operator when `A,B` are full rank. This is a genuine, provable structural
> restriction, distinct from H_g's width confound — but a *parameter-matched*
> generic-dense alternative is not cheaply constructible (matching
> `RowThenColProjection`'s exact `~2d²` params with a factored dense map
> forces rank ≤1; matching *expressiveness* costs `~d⁴/2d² = d²/2`× more
> params) — this ambiguity is carried into §4 and §9 rather than resolved by
> assumption.

> **H_c (output head, the brief's candidate (c)).** MultiProbeHead vs a
> tied/vector-collapse readout is the loss site. **Prior: treated as
> UNTESTED (corrected in revision 2, F3):** revision 1 downgraded H_c
> citing Run 25's "MultiProbeHead drives rank 7.55 but doesn't improve
> BPB" — but that comparison (Config A: 24-layer MultiProbe vs Config E:
> 48-layer zero-param head) varies depth and head class simultaneously at
> 293K-param byte scale, so it is suggestive, not attributable. H_c
> screens at full, equal weight in Wave A (sharing the output-head axis
> with H_f, §6.1).

> **H_e (task-representation mismatch, the brief's candidate (e)).** LM at
> this scale/data (WikiText + math CoT) rewards n-gram mixing that dense
> vector ops do per-FLOP better; on a composition-heavy task, the ranking
> inverts. **Existing evidence weighs in favor of this being real somewhere
> in the space**: Task E (`TASK_E_FINDINGS.md`, 2026-07-02, 40K-step round)
> already shows 4/5 seeds of a matrix-native architecture achieving exact
> (cosine=1.00) recovery to compositional hop-depth 21, a regime no
> vector-flat baseline in this project has matched. Untested: whether *this
> LM architecture pair*, not Task D/E's associative-memory encoder, shows
> the same inversion on a composition-heavy *sequence-prediction* task.

**Overall CONFIRM / FALSIFY / diagnostic-pass bar**, mirroring
`STAGE0_DESIGN.md` §1's structure:

- **Named dominant site (partial or full CONFIRM):** one hypothesis's
  component-swap recovers `≥50%` of the matrix-vs-vector BPB gap
  (`recovered_frac`, defined precisely in §8) while every other screened
  hypothesis recovers `≤20%`, reproducibly across `≥2` seeds.
- **Distributed tax (a different, equally decisive outcome):** no single
  hypothesis clears the `50%` bar, and/or the swap results are
  non-orthogonal (recovered fractions sum well above 100%, meaning the
  mechanisms interact and cannot be cleanly separated) — reported as such,
  not forced into a false single-cause narrative.
- **Diagnostic pass condition (independent of which of the above):** Stage G
  produces a *measured* verdict on H_h and H_d specifically. H_h's analytic
  half is already CONFIRMED pre-launch (V1: two independent derivations
  agree on 11.8–14.9× vs the 28–32× used); what remains is the H_d
  measurement — Matrix's extended-budget trajectory (§4 item 2, §7) — plus
  the corrected "true gap" statement. **Out-of-budget exit,
  pre-registered (N1):** under the now-mainline fallback (Run 12's
  checkpoint confirmed not locally recoverable), the Regime-1 half of the
  H_d measurement exits Stage G as *explicitly out-of-budget* — the
  diagnostic-pass condition is then carried by arm (b) (Regime-2
  multi-seed extended arms, which alone own H_d's FALSIFY branch per §8/N2)
  plus the corrected compute accounting, and the from-scratch extended
  Regime-1 run is named as the mandatory first follow-on rather than
  silently substituted. "The comparison was never FLOPs-matched in either
  direction; here is the measured gap under an honest accounting" is a
  decisive claim regardless of every other wave's outcome.

---

## 2. Why now, and why this shape

### 2.1 The exact comparison being diagnosed

Grounded in `experiment-runs/8xh100-session1/round2_matrix_script.py`
(`MatrixThinker`, the model actually trained for Runs 12–15 — **not**
`matrix-thinking/src/matrix_model_v2.py`, an unused, differently-parameterized
sibling; not `model_v4.py`, Task D/E's separate encoder) and
`loopformer_96K_script.py` (`LoopFormer`, ICLR 2026 arXiv:2602.11451):

| | MatrixThinker (Run 12/14) | LoopFormer (Run 13/14) |
|---|---|---|
| Params | 5,155,960 | 5,330,400 |
| Embed | `embed_u,embed_v`: rank-1 outer product, `2·vocab·d` = 3.22M, **default init (std 1.0 → entry-std ≈1, H_j)** | `wte`: dense `vocab·n_embd` = 4.82M, **tied to `lm_head`**, init std 0.02 |
| Backbone | 8 `ThinkingBlock`s (Frobenius attn + multiplicative layer), each applied at **every one of 8 iterations** = 64 layer-applications, ~180.8K params (3.5% of total), no iteration conditioning | 2 `LoopFormerBlock`s (standard causal attn + AdaLN-MLP), **shared weights**, applied 8× = 16 block-applications, ~240.8K params (4.5% of total), **per-loop AdaLN timestep conditioning (H_i)** |
| Output | `MultiProbeHead`, K=d=32, **untied** `Linear(32,vocab)` = 1.61M | `lm_head` = `wte.weight.T` (tied, 0 extra params) |
| Steps | 3,000 (0.5 epoch @ 2.19B tokens; LR schedule hit zero by design with train loss still steeply descending) | 96,000 planned; **best at step 21,500** (V1, verified against the raw log's `*BEST*` markers); loss spike ~31K; GN=inf divergence ~52K; killed at step 82,600 — no DONE, 0-byte SUMMARY/results.json |
| Measured throughput | 117–122K tok/s (8×H100 DDP) | 3.3–3.9M tok/s (8×H100 DDP) |
| Result | BPB 1.67 (T=8) | BPB 0.87 (best, **step 21,500** — `EXPERIMENT_LOG.md`'s "~step 40K" is being corrected separately) |

Structural facts fall out of reading the code that neither the brief's
candidate list nor any prior write-up names:

### 2.2 The embedding+output tables dominate BOTH models' params (>93%)

Run 12's own log states it plainly: "5,155,960 params (embed=3.3M,
think=197K, head=1.6M)" — the *backbone under test* (the thing every
"matrix vs vector operations" claim is actually about) is **3.8%** of the
param budget on the matrix side, and hand-deriving LoopFormer's equivalent
(embed 4.82M tied + pos 196.6K + 2-block backbone 240.8K + timestep
embedders 68K = 5.33M) puts its backbone at **4.5%**. Both models spend the
overwhelming majority of their matched ~5.2M params on token-identity
plumbing, not on the operations whose comparison motivates this whole
project. This single fact reframes several brief candidates: H_a
(embedding), H_j (its init), and H_f (tying) aren't marginal footnotes —
they are, by param share, most of what's actually being compared. It also
motivates §7's dual-regime manifest design (a byte-vocab harness for the
OFAT waves, so the backbone stops being drowned out by table params neither
side's "matrix vs vector" thesis is actually about).

### 2.3 The depth/width structure was never matched (H_g, new)

`MatrixThinker._one_iteration` loops over `self.layers` (8 *distinct*-weight
`ThinkingBlock`s) once per outer iteration (`for t in range(n_iterations)`);
at T=8 that is **64 total layer-applications with 8 independent weight
sets**. `LoopFormer._one_loop` loops over `self.blocks` (2 *shared*-weight
`LoopFormerBlock`s, weight-shared across all 8 loops, distinguished only by
AdaLN timestep conditioning) — **16 total block-applications with 2
independent weight sets**. The two "T=8"/"L=8" numbers in every table in
`EXPERIMENT_LOG.md` are not describing comparable objects: one is "8
sequential independently-weighted layers, run 8 times" and the other is "2
independently-weighted layers, run 8 times." Separately, matrix attention's
flattened working width is `d²=1,024`; LoopFormer's is `n_embd=96` — a 10.7×
width gap. Both facts inflate matrix FLOPs/token for reasons that have
nothing to do with "are matrix operations better or worse per FLOP than
vector operations" — they are choices about how the ~5.2M matched params
were spent, not properties of the operation type. This is the LM-comparison
analog of `STAGE0_DESIGN.md` §2.1's "H_cap ... is a real, separate,
previously-unstated design flaw" finding: reading the forward pass surfaces
a confound the existing sweep never controlled for. (The AdaLN conditioning
asymmetry noted in the same code reading is split out as its own
hypothesis, H_i — F5.)

### 2.4 The "matched FLOPs" comparison was never matched — in either direction (H_h; verified, V1)

Run 14's 96,000-step figure exists so LoopFormer's *total* FLOPs matches
MatrixThinker's — but the arithmetic behind it (`EXPERIMENT_LOG.md`, "32×
more FLOPs per step") is the *measured wall-clock throughput ratio*
(122K vs ~3.9M tok/s ≈ 32×), not a FLOP count derived from the two forward
passes. **This is now verified (V1), with two findings:**

1. **The analytic FLOPs/token ratio is 14.9× (non-causal convention) /
   11.8× (causal-exact)** — Matrix 230.6M vs LoopFormer 15.45M FLOPs/token
   (§5.1, independently derived from scratch and agreeing with this
   document's corrected derivation) — far below the 28–32× used to size
   the run. The residual (~2×) between the analytic and throughput ratios
   is consistent with H100 kernel-launch/small-GEMM underutilization on
   the matrix side (many independent `32×32`/`1024`-wide matmuls per token
   vs LoopFormer's few large, well-batched GEMMs): wall-clock was a bad
   proxy for arithmetic, exactly as hypothesized.
2. **The direction of the implied correction is the opposite of what
   revision 1 assumed.** Revision 1 reasoned: LoopFormer got more compute
   than a true match would allow, so the gap is overstated. The
   verification killed that reading by re-dating LoopFormer's best:
   BPB 0.87 / PPL 10.6 occurred at **step 21,500** (not ~40K; re-verified
   this session against `loopformer_96K_full.log` — best `*BEST*` marker
   at eval step 21,500, gradient-norm blowup from ~31K, GN=inf by 52K,
   killed at 82,600 with no final results). At step 21,500, LoopFormer had
   consumed only **0.48× (non-causal) to 0.61× (causal-exact)** of
   Matrix's total training FLOPs. **The cited comparison therefore favored
   Matrix on compute: LoopFormer won ~1.9× on BPB using roughly half the
   FLOPs.**

The fair statement, which replaces both the original headline and
revision 1's draft correction: **the comparison was never FLOPs-matched in
either direction — Matrix was undertrained against its own budget (still
steeply improving at its 3,000-step cutoff), LoopFormer's best used ~half
of Matrix's compute, and the true converged gap is unmeasured.** This
elevates H_d to the single missing measurement (LoopFormer's side of the
curve is already characterized: it peaked at 21.5K steps and degraded
monotonically afterward from multi-epoch overfitting, so its
best-within-this-setup is known) and adds one required cell to the
manifest: **Matrix's extended-budget BPB in the original regime** (§4
item 2, §7 Wave 0-R1). `EXPERIMENT_LOG.md` Run 14's "~step 40K" and
"~653K TFLOPs" figures are being corrected separately; this document cites
step 21,500 and §5.1's verified FLOPs table throughout.

### 2.5 What existing evidence does and does NOT already show (corrected this revision, F3)

Revision 1 claimed Runs 18/22 "downgrade" H_a and Run 25 "downgrades" H_c.
The attack round showed both readings over-claimed: **no run in this repo
has ever varied the embedding axis** — Runs 18 and 22 hold the
outer-product embedding fixed in both arms and vary the downstream
operations (Run 18's flat arm additionally swaps in a 12.9M-param untied
Linear output head, an undisclosed second confound; Run 22's two arms share
the MultiProbeHead), so `EXPERIMENT_LOG.md`'s "definitive embedding vs
operations test" label for Run 22 is wrong on the embedding half — it is an
operations-only test. Run 25's "MultiProbeHead ... doesn't improve BPB"
compares cells that differ in depth and head class simultaneously (Config
A: 24-layer MultiProbe vs Config E: 48-layer zero-param head). What the
existing record *does* still establish: the T=1-vs-T=8 pattern (matrix-ops
arms lead at T=1; vector-ops arms catch up and pass under iteration — Runs
18/22, both directions) survives as a fact about **the iterated regime
being where the loss concentrates**, without licensing any component-level
attribution. H_a and H_c therefore screen at full, equal Wave A weight
(§6.1), and every §3 prior is treated as a ranking heuristic only — the
attack round demonstrated the prior-setting exercise itself is fallible
(§9 attack #5).

---

## 3. Candidate mechanisms — summary table

| ID | Hypothesis | Predicts | Status / prior (before Stage G runs) |
|---|---|---|---|
| H_h | FLOPs-accounting artifact (§2.4) | Analytic FLOP ratio ≪ throughput-derived ratio; LoopFormer's best predates a true FLOPs match | **Mechanism CONFIRMED pre-launch (V1)**: 11.8–14.9× vs 28–32×; direction corrected — the miss favored Matrix on compute; Wave −1's instrumented count remains as final corroboration |
| H_d | Undertraining/budget artifact — Matrix's converged BPB is the missing datum (§1, §2.4) | Extended-budget Matrix (Regime 1 continuation + Regime 2 multi-seed) keeps improving well past its original cutoff; gap narrows materially — or persists as a stronger negative | HIGHEST — elevated by V1; matched to two recent same-project budget-artifact findings; mandated test-first by CLAUDE.md |
| H_j | Embedding init scale: default std-1.0 `u,v` → step-0 entry-std ≈1 vs LoopFormer's 0.02 (§1) | Step-0 entry-std mismatch measured ~50× (free Wave −1 check); matched-init cell (`std=sqrt(0.02)`) recovers part of the gap | HIGH — the exact CLAUDE.md hard-rule bug class, half-verifiable at zero GPU-h |
| H_g | Architecture-config confound: unmatched width (`d²` vs `n_embd`) and depth (64 vs 16 layer-applications) (§2.3) | Depth-matched control shrinks part of the gap without touching "matrix vs vector" per se; width contribution disclosed, not tested (§4 item 5) | HIGH — provable from reading the code, not inferred; conclusion conditional on H_i (F5) |
| H_i | Iteration conditioning: LoopFormer's per-loop AdaLN signal has no ThinkingBlock analog (§1) | Adding a minimal per-iteration conditioning signal to the matrix loop recovers part of the gap | MEDIUM — mechanism is LoopFormer's own named innovation; untested here |
| H_f | Weight tying / embedding-output coupling (§1) | A genuinely-tied head (§4 item 4 — NOT the existing dead-code class, F2) closes part of the gap at ~0 extra params | MEDIUM-HIGH — well-precedented mechanism, cheap once the F2 build task is done |
| H_a | Embedding bottleneck (rank-1 outer product) | Full-rank matrix embedding (`d²` DOF/token) improves the iterated regime | UNTESTED at this axis — revision 1's downgrade was based on runs that never varied the embedding (F3) |
| H_b | Kronecker-restricted attention/thinking projections | A less-restricted (rank-swept factored) projection family recovers some gap | MEDIUM — real, provable restriction, but no cheap clean control exists (ambiguity carried forward); gated |
| H_c | Output head (MultiProbeHead vs tied/collapse readout) | Output-head swap alone moves BPB substantially | UNTESTED at this axis — Run 25's head comparison is depth-confounded (F3) |
| H_e | Task-representation mismatch | A composition-heavy sequence task inverts the ranking for this exact LM architecture pair | MEDIUM — plausible by analogy to Task E, untested for *this* pair; expensive to build cleanly; gated to Wave C |

---

## 4. Candidate interventions / component swaps, ranked

Ranked by (mechanistic groundedness × cheapness), matching Stage 0's
convention. Items 1–2 are Waves −1/0 (measurement, not component swaps);
items 3–7 are the Wave A OFAT screen (five cells, §6.1); items 8–9 are
gated (Wave B/C only, conditional).

**1. (Wave −1, §7) Analytic FLOP audit + step-0 scale instrumentation —
zero GPU-h.** The analytic side of H_h is already double-derived (this
document's corrected §5.1 + V1's independent from-scratch derivation, in
agreement); the remaining Wave −1 work is (i) an *instrumented* count
(profiler-based, not by-hand — by-hand arithmetic in this program has now
needed two corrections, §9 attack #3) as final corroboration, and (ii) the
H_j free check: log step-0 residual-stream entry-std for both Regime-1
models at init (CPU forward pass, zero GPU-h) — F6 predicts ~1.0 (matrix)
vs ~0.02-scale (LoopFormer).

**2. (Wave 0) Extended-budget matrix runs — the H_d test, now the manifest's
top-priority GPU item (V1).** Two arms:
   (a) **Wave 0-R1, the conditional Regime-1 cell (revised 2.1 — the
   fallback below is now the MAINLINE, N1):** extend the *unmodified* Run
   12 MatrixThinker in its original regime (BPE vocab, 8×H100 DDP, same
   data) by **+1,500 steps as a warm-continuation from Run 12's final
   checkpoint** (11.25 GPU-h at the measured 7.5 GPU-h/1,000-step unit),
   loss logged densely, **intermediate evals at T=8 only (full T∈{1,2,4,8}
   sweep at the final checkpoint only — N7)** so eval cost stays inside
   the priced 11.25 GPU-h. Two stacked caveats, both disclosed: (i) Run
   12's cosine LR schedule decayed to zero by step 3,000, so the
   continuation re-warms a constant-then-decay schedule; (ii) **the
   archived script saves only `raw_model.state_dict()` — no AdamW
   moments, no scheduler, no RNG state (N2)** — so this is a moment-less
   re-warmed continuation at only 1.5× total budget, which *lower-bounds*
   what a properly-sized longer run could achieve. **Verdict scoping
   (N2): 0-R1 can contribute CONFIRM or INCONCLUSIVE evidence for H_d
   only — it can never FALSIFY it** (Stage 0's own precedent needed
   2.5–6× budgets to date transitions; a moment-less 1.5× continuation
   that plateaus proves nothing about a real longer schedule); H_d's
   FALSIFY branch belongs to arm (b) alone (§8).
   **Checkpoint status (N1): confirmed NOT locally recoverable** — no
   `.pt` exists under `experiment-runs/8xh100-session1/`, and the SSD's
   `best.pt` is a byte-agnostic-era model (pickle keys
   `byte_embed_for_expand`/`ln1`/`ln2`, not MatrixThinker); the retired
   `/toy_story_slam` pod volume is the only remaining hope, checked as a
   Wave −1 zero-GPU task (§12(vi)). **The build phase therefore assumes
   the fallback.** Pre-registered fallback and its aftermath (N1): a
   from-scratch extended Regime-1 run does not fit the 40 GPU-h ceiling
   alongside the OFAT program (4,500 steps = 33.75 GPU-h would consume
   it), so on fallback (mainline) Stage G (i) reports the Regime-1
   converged-gap datum as explicitly out-of-budget (§1, §10), relying on
   arm (b) for H_d's verdict (with the regime-transfer caveat, §9 attack
   #2) and naming the from-scratch extended Regime-1 run as the mandatory
   first follow-on; (ii) frees 11.25 GPU-h, which flows down the §7
   stretch list *in its existing order* — with stretch item 1 (the 0-R1
   extension) voided alongside 0-R1 itself, and **stretch item 2 (the
   Regime-1 winning-intervention spot-check) re-anchored: its comparator
   becomes Run 12's own logged step-matched eval (a cross-run comparison
   carrying §9 attack #11's full drift caveats), or, if the winning
   intervention is one whose effect cannot be read against a cross-run
   comparator within those caveats (e.g. an effect size smaller than
   plausible drift), item 2 is dropped and the freed budget continues
   down the list to Wave C / H_b.**
   (b) **Wave 0-R2 (Regime 2, byte-vocab d=32):** multi-seed extended
   (3× steps) matrix and vector arms plus the matched-budget gap-baseline
   pairs every §8 `recovered_frac` is defined against — the cheap,
   seed-replicated characterization of the same trajectory question.
   Verdict rules including the still-rising-at-cutoff → INCONCLUSIVE
   branch: §8.

**3. Embedding init-scale fix (targets H_j) — new this revision (F6).**
Re-init `embed_u`/`embed_v` (and `pos_u`/`pos_v`) at `std = sqrt(0.02) ≈
0.141` so outer-product entries enter the stream at std ≈ 0.02, matched to
LoopFormer's explicit init — per CLAUDE.md's hard rule ("u,v std must be
sqrt(target_std) — products have std=σ²"). Zero param/FLOP change, the
cheapest architecture-touching swap in the grid; ranked immediately after
the budget tests because a ~50× step-0 activation-scale mismatch changes
the optimization landscape every other component operates in.

**4. Genuinely-tied output head (targets H_f) — REWRITTEN this revision
(F2); blocking pre-Wave-A build task.** Revision 1 planned to wire in
`matrix_output_heads.py::TiedMultiProbeHead` "as-is, zero new code." The
attack round found that class is **dead code that does not tie**: its
`forward()` derives probes from fresh `probe_mix_u/v` Linear layers and
never reads the stored `embed_u`/`embed_v` references, and its `out`
projection is a full untied vocab-sized Linear — running it would have
produced a near-null `recovered_frac` and silently recorded H_f as
falsified. The build phase must instead produce a genuinely tied head, in
one of two pre-specified forms:
   (i) **Regime 2 (byte vocab) — exact bilinear tied logits, with a
   learned scalar temperature (N3):** `logit(w) = τ · (u_w^T M v_w)`
   using the embedding rows themselves (cost `O(B·L·vocab·d²)` per naive
   einsum is trivial at vocab=256), where `τ` is a single learned scalar
   (init so step-0 logit std ≈ 1). **Why the temperature is mandatory
   (N3):** at default std-1.0 embedding init, the raw bilinear form has
   scale ~O(30) (`u_w^T M v_w` sums ~d² products of std-1 terms through a
   normalized `M`) → a saturated softmax at step 0, and the H_f cell
   would crater *for an H_j reason* — an H_f×H_j confound baked into the
   swap itself. **Pre-registered follow-up: if the primary H_f cell still
   lands far below the matrix-all baseline, H_f is re-run once on the
   matched-init (H_j-fixed) variant before any verdict is recorded**, so
   a "tying hurts" reading can never be manufactured by the init-scale
   artifact.
   (ii) **Regime 1 (BPE vocab) — factored tied head:**
   `logits = (probes @ W_mix) @ [E_u, E_v]^T` where `[E_u, E_v] ∈
   R^{vocab×2d}` is the **feature-concatenation** (per-vocab-row concat of
   the two `d`-dim embeddings — not a row-stack; N4), `E_u = embed_u.weight`
   and `E_v = embed_v.weight` are the *same tensors* (not copies), and
   `W_mix ∈ R^{K×2d}` is the only new parameter — the same GEMM class as a
   width-`2d` tied `lm_head`.
**Mandatory tie-verification unit test (blocking, §12(v)):** perturb
`embed_u.weight` in-place and assert the head's logits change; assert by
tensor identity (`is`, not value equality) that the head references the
embedding tensors; assert the head owns no vocab-sized untied parameter.
Per CLAUDE.md's hard rule on negative unit tests, the test must be *run to
completion*, not merely written. Cite: Press & Wolf 2017
(arXiv:1608.05859); Inan et al. 2016 (arXiv:1611.01462) — verify exact
citations before external use.

**5. Depth-matched control (targets H_g; width attribution explicitly NOT
tested — attack-round minor).** Run MatrixThinker with `n_layers=2`
(mirroring LoopFormer's 2-shared-blocks×8-loops structure) instead of 8
distinct layers, at matched T=8 — isolates whether the "64 vs 16
layer-application, 8 vs 2 weight-set" asymmetry, not matrix-ness, drove
part of the gap. **Width is disclosed, not manipulated: Stage G contains no
width-isolated manipulation** (equalizing `d²` against `n_embd` requires
changing `d`, which changes the embedding and every downstream shape — out
of budget), so width's share of the gap is **not falsifiably tested here**;
per-cell FLOPs/token logging makes width's *arithmetic* contribution
explicit (§5.1: the attention-score term alone carries the full 10.7×
width ratio), but any *quality* attribution to width in the write-up must
be labeled hypothesis, not finding. **H_g's depth conclusion is
additionally conditional on H_i** (a depth-matched matrix model without
conditioning is still not structurally identical to LoopFormer's
conditioned loop — F5); Wave A screens H_i as its own cell so the two
verdicts are separable.

**6. Iteration conditioning (targets H_i) — new this revision (F5).** Add a
minimal AdaLN-style per-iteration conditioning signal to the unmodified
8-layer matrix baseline: a small timestep embedder (reusing LoopFormer's
own `TimestepEmbedder` pattern) producing a per-iteration scale/shift
applied at each `ThinkingBlock`'s norm. Modest params (d-independent), one
new module; screened as its own Wave A cell.

**7. Full-rank matrix embedding swap (targets H_a).** Replace
`MatrixEmbedding`'s rank-1 `u⊗v` with a direct `nn.Embedding(vocab, d*d)`
reshaped to `(d,d)` per token (`d²` DOF, no rank constraint at embedding
time — the exact "reshape equivalence" construction CLAUDE.md's hard rule
describes: "any `d²`-dim vector can be reshaped to `d×d` matrix...
structure only matters if OPERATIONS preserve it"). Same backbone, same
head — isolates whether the *embedding's own* rank-1 constraint, not
downstream operations, is a loss site. Cheap (table-size change only).
Prior corrected this revision (F3): no existing run has varied this axis
(§2.5), so H_a screens at full weight.

---

*Gated pool — not screened in Wave A, deployed conditionally:*

**8. `RowThenColProjection` → less-restricted alternative (targets H_b).**
Held out of Wave A because no cheap, clean control exists (§1's H_b entry —
matching params forces rank-1, matching expressiveness costs `~d²/2`× more
params). Deployed in Wave B only if item 5's depth-matched control
*doesn't* close the gap (ruling out the confound explanation and making the
operator-family question load-bearing) — using a rank-`r` factored
`Linear(d²,r)→Linear(r,d²)` swept over a few `r` values as a *parametric
family* bridging Kronecker-restricted and increasingly expressive, rather
than a single point comparison.

**9. Task-swap control on a composition-heavy corpus (targets H_e).** Held
to Wave C, gated on Wave A/B producing either a distributed-tax verdict or a
confirmed-but-narrow dominant site (i.e., only invoked if the "why" question
remains open after the cheaper waves). Design: adapt Task D/E's synthetic
K-relation, hop-composition generator (`task_e.py`) into a
next-token-prediction corpus — serialize entities/relations/composed-query
answers as a small byte/token vocabulary sequence, so the *same* two LM
architectures (MatrixThinker, LoopFormer, matched params, no encoder-style
single-Z bottleneck) can be trained on it directly, rather than reusing Task
D/E's own bottlenecked-encoder harness (a materially different architecture
from the one Runs 12–15 diagnosed). This is a genuine build item (a new data
generator), deferred to the build phase per this document's "design only"
scope, and flagged as the single largest engineering-risk item in the whole
manifest — same caveat Stage 0 gave its own candidate 5 (net2net curriculum).

---

## 5. FLOPs / param / memory estimate (CLAUDE.md pre-experiment checklist)

### 5.1 Verified per-token FLOP count at the Run 12–15 config (d=32, T=8, n_embd=96) — corrected this revision (V1)

Using the `2·MACs` convention, forward pass, per token. Per
`RowThenColProjection` (`silu(A@M)@B`, two `d×d @ d×d` matmuls): `4d³`.

**MatrixThinker:**
- `MatrixFrobeniusAttention`: 4 projections (`Q,K,V,O`) = `16d³` = 524,288;
  SDPA over the flattened `d²`-wide per-head vectors: `4·L·d²` = 2,097,152
  (at L=512). Per attention: 2,621,440.
- `MatrixMultiplicativeLayer`: 6 projections = `24d³` = 786,432; 2 matmuls
  for `M_mult` = `4d³` = 131,072. Per layer: 917,504.
- Per `ThinkingBlock`: 3,538,944. × 8 layers × 8 iterations = 226.5M
  (backbone).
- `MultiProbeHead` (K=32, untied): ≈ 3.3M (dominated by `2K·vocab`).
- **Total ≈ 230.6M FLOPs/token** (verified figure, V1; this document's
  own corrected derivation lands within <1% of it).

**LoopFormer** (revision 1's derivation had a bug here, caught by V1:
AdaLN conditioning and the timestep embedders were counted per *token*,
but the code computes `c` once per (batch, loop, block) — `self.adaLN(c)`
takes the `(B, n_embd)` conditioning vector, not the sequence — so their
cost amortizes over L=512 to near-zero per token):
- Per block per token: QKVO `4·n_embd²` = 73,728; attention score
  `4·L·n_embd` = 196,608; MLP `4·n_embd·intermediate` = 92,160; AdaLN +
  timestep embedders amortized ≈ 0.2K. Per block ≈ 362.6K.
- × 2 blocks × 8 loops = **5.80M (backbone — verified figure)**.
- Tied `lm_head`: `2·n_embd·vocab` ≈ 9.65M.
- **Total ≈ 15.45M FLOPs/token** (verified figure).

**Verified ratio: 230.6M / 15.45M ≈ 14.9× (non-causal convention); 11.8×
causal-exact** — vs the **28–32×** throughput ratio the existing comparison
used to size Run 14. Consequences in §2.4 (H_h — confirmed, direction
corrected). Total-training-FLOPs accounting at the actual best checkpoints
(both models: 393,216 tokens/step): Matrix, 3,000 steps ≈ 2.7×10¹⁷ forward
FLOPs; LoopFormer at its step-21,500 best ≈ 1.3×10¹⁷ — **0.48× (non-causal)
to 0.61× (causal-exact) of Matrix's total**.

Notably, `4·L·d²` (matrix attention score, 2.1M) is the single largest
per-token FLOP contributor on the matrix side, and alone carries ~10.7×
LoopFormer's score term (196.6K) — exactly the `d²/n_embd = 1,024/96`
width ratio H_g names, tracing to the unmatched channel width, not to any
Kronecker-vs-dense argument (H_b). The backbone-only ratio (226.5M/5.80M ≈
39×) vs the total ratio (14.9×) also quantifies §2.2's point: the big
embed/head GEMMs compress the total ratio; the backbones themselves differ
far more.

*(FLOP-convention footnote, attack-round minor: the `4·L·width`
attention-score terms use unmasked `L`; under causal masking the average
context is `L/2`, a 2× over-count. It applies to both models' score terms
and largely cancels in the ratio — the causal-exact 11.8× above is the
corrected-ratio figure; component shares shift slightly. Wave −1's
instrumented count uses the masked figure.)*

### 5.2 Param formulas

Matrix side, exact backbone count (verified against Run 12's own log,
§2.2): attention = `4 × 2d² = 8d²`; multiplicative ≈ `10d²` (6 projections
at `2d²` counts 12d² — minus shared-structure overlaps, plus gate matrices
`2d²`, key/val columns `4d`, biases); per `ThinkingBlock` ≈ `18–22d²`. At
`d=32`: ≈18.4–22.5K/layer, 8 layers → ~150–180K, matching the log's stated
"think=197K" to within bias/norm terms — treat the log's figure as
authoritative, this derivation as an order-of-magnitude check.

`N_loopformer(n_embd, vocab, L_blocks) = vocab·n_embd [tied embed/head] +
max_len·n_embd [pos] + L_blocks·(~2.5·n_embd² + 2·n_embd·intermediate) +
2·(freq_dim·n_embd + n_embd²) [timestep embedders]` → 5.33M at the Run 13
config, matching the log.

### 5.3 Memory

At d=32, the largest per-sample tensor is the token-matrix activation
stream: `32×32×4 bytes = 4KB/token`; at batch=768, seq=512, ~6GB before
checkpointing — Run 12 measured ~43GB/80GB with activation checkpointing
enabled, so the regime is known-safe. LoopFormer's residual stream is
smaller still. Stage G's Regime-2 cells (byte-vocab, d=32, single-GPU,
smaller batch) are far below any limit. **No cell in this manifest is
memory-bound.**

### 5.4 Wall-clock — two regimes

**Regime 1 (confirmatory — the original Runs 12–15 setting):** reuse
`round2_matrix_script.py` / `loopformer_96K_script.py` verbatim (8×H100
DDP, BPE vocab 50257, ~5M params), with the **measured** throughputs in
`EXPERIMENT_LOG.md`: 117–122K tok/s (matrix), 3.3–3.9M tok/s (LoopFormer).
**Measured unit: a 3,000-step matrix cell = 168.7 min × 8 GPUs = 22.5
GPU-h, i.e. 7.5 GPU-h per 1,000 steps** — the number revision 1's Wave B
row contradicted by 9× (F1). Regime 1 appears in the core manifest exactly
once (Wave 0-R1's required H_d continuation, §7); everything else
Regime-1-flavored is on the stretch list.

**Regime 2 (diagnostic — the OFAT screening harness, §6): byte-vocab (256,
not 50257), pinned to d=32 (F4)** — H_g's 10.7× width ratio and every §5.1
number are d=32 quantities; at d=16 the width ratio is only 2.7×, a
different regime — single-GPU, packed 4–8 cells/GPU per Stage 0's proven
discipline, following `run_ablation_matched.py`'s (Run 22's) byte-vocab
matrix-vs-flat harness pattern. Byte-vocab justification (§2.2): at BPE
vocab, embed+output tables are >93% of params on *both* sides, which would
swamp any component-swap signal; byte vocab inverts that ratio, putting the
backbone at the center of the param budget. **Throughput caveat (F4): every
existing byte-vocab precedent is d=16 and 8-GPU DDP** (Run 19: byte d=16,
260K tok/s; Run 17: BPE d=16 v2, 241K tok/s — two different configs
revision 1 conflated into one range) — no d=32 byte single-GPU throughput
has ever been measured. The provisional unit is therefore derived from Run
12's measured **d=32** figure instead (117–122K tok/s at 8-GPU DDP → ~15K
tok/s single-GPU, matrix side): **matrix-side standard Regime-2 cell = 1.0
GPU-h flat; vector-side = 0.25 GPU-h flat** (conservative: prices the
vector model at only 4× the matrix speed vs the ~27× the ÷8 extrapolation
suggests). Wave −1 calibrates both at d=32 before Waves 0/A size their
step counts, per CLAUDE.md's calibration-run rule. **Standard-cell step
count (N5):** provisionally **3,000 steps** (mirroring Run 22's byte-vocab
harness scale); Wave −1's mandatory outputs include pinning the final
standard-cell step count (and batch/seq), **frozen before Wave 0
launches** — every §8 `recovered_frac` is defined at that pinned budget,
so it cannot drift per-cell mid-sweep.

---

## 6. Instrumentation plan

### 6.1 Per-component ablation grid (OFAT, bidirectional)

Six swappable component axes; **five screen in Wave A (one cell each,
matrix-all baseline with exactly one axis flipped), all at equal weight —
revision 2 removed revision 1's downgrade of the embedding and output-head
axes (F3)**; the projection axis stays gated (§4 item 8). Shared small
byte-vocab d=32 backbone (Regime 2, §5.4):

| Axis | Matrix-native state | Alternative state | Targets |
|---|---|---|---|
| Embedding init scale | Default `nn.Embedding` init (std 1.0; entry-std ≈1) | `std=sqrt(0.02)` on `u,v` (entry-std ≈0.02, LoopFormer-matched) | H_j |
| Embedding rank | Rank-1 outer product (`u⊗v`) | Full-rank matrix embed (`d²` DOF/token, direct table) | H_a |
| Output head | `MultiProbeHead` (untied) | Genuinely-tied bilinear head (§4 item 4 — NOT the existing dead-code `TiedMultiProbeHead`, F2) | H_f, H_c |
| Depth structure | 8 distinct layers × T iterations | 2 distinct layers × T iterations (depth-matched) | H_g |
| Iteration conditioning | None | AdaLN-style per-iteration scale/shift (§4 item 6) | H_i |
| Attn/thinking projections | `RowThenColProjection` (Kronecker-separable) | Rank-swept factored dense (`r∈{1,4,16}`) — **gated per §4 item 8, not screened in Wave A** | H_b |

Each Wave A cell runs from the **matrix-all baseline** with exactly one
axis flipped (5 cells), producing 5 single-swap deltas; Wave B mirrors the
top-1/2 survivors from the **vector-all baseline** direction (flip toward
matrix) for symmetry, per the general OFAT-then-confirm discipline Stage 0
used (screen singly before combining).

### 6.2 Loss-curve trajectories (the budget-artifact check, H_d)

Every Wave 0/A/B run logs loss, PPL, and BPB at every `≤200`-step
checkpoint (Regime-1 continuation: every ≤250 steps) — dense enough to
date any late transition, reusing Stage 0's own justification: Task E and
Stage 0 both found budget artifacts hiding behind under-sampled curves.
The extended-budget Wave 0 arms (§4 item 2) are the primary H_d test;
every other wave's runs are compared against the matched-budget baselines'
checkpoints, not just final values.

### 6.3 Per-layer signal metrics — "where is the signal lost" instrumentation

Logged every `≤500` steps, on a fixed eval batch, per layer/iteration index
`l`:
1. **Effective rank of the layer's token representation** (matrix: direct
   `svdvals` on `M_l`; vector: reshape the `d²`-dim residual stream into a
   `d×d` matrix *for measurement only* — a diagnostic use of the reshape
   equivalence, not a computational one, so it does not conflict with
   CLAUDE.md's "structure only matters if operations preserve it" rule).
2. **Activation Frobenius/L2 norm by depth** — does representational
   "energy" grow, shrink, or hold steady across the 8/16/64
   layer-applications? A monotonically shrinking norm by depth is a direct,
   named signature of "signal loss," distinguishable from a healthy
   information-preserving stack. (Also directly relevant to H_j: the
   step-0 entry-scale mismatch should be visible here at init and its
   decay/persistence trackable across training.)
3. **Gradient norm per parameter group** (embed / attn-proj / thinking-MLP /
   output-head), pre- and post-clip — reused directly from Stage 0's §6.2
   MINOR-12 instrumentation, since the flat `clip_grad_norm_(1.0)` +
   large-activation-scale interaction it was designed to catch applies
   here too (and interacts with H_j).
4. **Marginal-value-per-FLOP knockout probe.** For each component type, in
   each architecture: (i) its FLOP-cost share (from §5.1's verified count);
   (ii) its quality-contribution share, measured by replacing that
   component's output with a cheap proxy (running mean over the batch, or
   identity/no-op where shape permits) at *eval time only* and recording
   the resulting loss increase `Δloss_component`. **Value density**
   `= Δloss_component / FLOP_share_component`, computed for every component
   in both architectures, directly answers "where is the per-FLOP signal
   loss" in the prompt's own framing — a low value-density component in the
   matrix model relative to its vector-model counterpart is direct, causal
   (not merely correlational) evidence for that component being a loss
   site.

---

## 7. Manifest (waves, gates, GPU-h) — fully recomputed this revision (F1, F4, V1)

**Budget units (provisional where stated, measured where stated).**
Regime-2 (byte-vocab, **d=32**, single-GPU) flat units pending Wave −1's
d=32 calibration: matrix-side standard cell = 1.0 GPU-h; vector-side = 0.25
GPU-h (derivation and conservatism: §5.4); extended cells scale linearly
with the step multiplier (3×). Regime-1 unit is **measured, not
estimated**: 22.5 GPU-h per 3,000-step matrix cell → **7.5 GPU-h per 1,000
steps** (F1).

| Wave | Purpose | Cells | GPU-h |
|---|---|---|---|
| **−1** | Zero-GPU: instrumented FLOP count corroborating the already-double-derived §5.1 numbers (H_h; fresh-context review, §12(iii)); step-0 residual-stream entry-std for both Regime-1 models at init (H_j, CPU); pod-volume checkpoint-recoverability check for 0-R1 (§12(vi) — local recovery already ruled out, N1). GPU: live timing calibration of the byte-vocab **d=32** harness (F4), 2 cells (matrix + vector baselines, ~500 steps), whose mandatory outputs include **pinning the Regime-2 standard-cell step count, frozen before Wave 0 (N5)**. | 2 | 0.3 |
| **0-R1** | **Conditional H_d cell (V1; fallback is MAINLINE, N1):** warm-continuation of Run 12's unmodified MatrixThinker, original regime, **+1,500 steps** (measured unit: 11.25 GPU-h), dense loss logging with intermediate evals at T=8 only (N7) — measures whether Matrix BPB keeps falling past its original cutoff toward/away from LoopFormer's already-characterized best (0.87 at step 21,500). CONFIRM/INCONCLUSIVE evidence only, never FALSIFY (N2). Runs ONLY if the retired-pod checkpoint is recovered; otherwise (mainline) the 11.25 GPU-h flows down the stretch list per §4 item 2(a)'s aftermath paragraph. | 0–1 | 0–11.25 |
| **0-R2** | H_d multi-seed + gap baseline (Regime 2, byte d=32): extended (3×) matrix ×2 seeds (6.0) + extended vector ×2 seeds (1.5) + matched-budget gap-baseline pair, matrix ×3 (3.0) and vector ×3 (0.75) — the pair every §8 `recovered_frac` is defined against, and the external-validity check that the byte-vocab harness reproduces the qualitative gap. | 10 | 11.25 |
| **A** | OFAT screen (§6.1): 5 single-axis swaps from the matrix-all baseline (init scale, embedding rank, tied head, depth structure, iteration conditioning) × 1 seed, budget-matched checkpoint scoring. | 5 | 5.0 |
| **B (byte)** | Confirmation: top-1/2 Wave-A survivors × 3 seeds (≤6 matrix cells, 6.0) + bidirectional vector-all-baseline mirror for the winning axis (3 vector cells, 0.75). | ≤9 | 6.75 |
| **Core total** | | **26–27** | **23.3 (mainline: fallback fires, N1) to 34.6 (checkpoint recovered) ≤ 40** |

**What happened to revision 1's Regime-1 winning-intervention spot-check
(F1's original subject):** demoted to the stretch list. V1 re-ordered the
priorities — knowing whether the gap *survives proper training at all*
(Wave 0-R1, now itself conditional per N1) is logically upstream of
knowing whether a component fix transfers; both do not fit under the
ceiling at honest prices, and §10's transfer language is downgraded
accordingly (F1 option (c) applied to the transfer claim, option (b)'s
reduced-step form retained for the stretch item, with its mainline
re-anchor-or-drop rule in §4 item 2(a)).

**Headroom is NOT assumed (F1).** 34.6 is the face-value ceiling-case core
(5.4 nominal margin); **the mainline (fallback fires, N1) core is 23.3
GPU-h**, with 0-R1's freed 11.25 GPU-h flowing down this stretch list *in
order* per §4 item 2(a)'s aftermath paragraph. The stretch list otherwise
funds **only from measured under-run** (Stage 0's §12.9 measured its flat
units 5–6× conservative from packing 4–8 cells/GPU; the same packing
applies to every Regime-2 cell here, but is not assumed), in priority
order:
1. **Wave 0-R1 extension** (+1,500 further steps, 11.25 GPU-h) if the
   continuation ran AND is still visibly descending at its cutoff —
   directly services §8's H_d INCONCLUSIVE branch. **Void on the mainline
   fallback** (no 0-R1 → nothing to extend); the flow-down skips to
   item 2.
2. **Regime-1 winning-intervention spot-check** (1,500 steps, 11.25
   GPU-h). Comparator: the Wave 0-R1 trajectory at the matched step count
   if 0-R1 ran (same-harness); **on the mainline fallback, re-anchored to
   Run 12's own logged step-matched eval — a cross-run comparison
   carrying §9 attack #11's full drift caveats — or dropped outright if
   the winning intervention's effect size is within plausible cross-run
   drift (§4 item 2(a)), with the budget continuing down this list.**
3. **Wave C (H_e, §4 item 9):** ~10 GPU-h if invoked (gate unchanged:
   only on a distributed-tax verdict or a narrow confirm that leaves the
   generalization question open).
4. **H_b's gated rank-swept family (§4 item 8):** ~5 GPU-h, deployed per
   its own gate (depth-matched control fails to close the gap).

**If core overruns** (Wave −1's d=32 calibration invalidates the Regime-2
flat units), pre-registered cut order: (i) Wave 0-R2's extended matrix
arms drop 3× → 2.5× (−1.0); (ii) Wave B confirms top-1 only (−3.0);
(iii) Wave 0-R1 drops +1,500 → +1,000 steps (−3.75; never to zero — it is
the manifest's single most load-bearing cell). Wave −1, the Wave 0-R2
matched-budget baseline pairs, and Wave A are never cut — every §8
`recovered_frac` is defined against the baselines, and Wave A is the
cheapest, highest-value wave per cell in the manifest.

---

## 8. Success criteria, defined precisely

**`recovered_frac(X)`** for a component swap `X`, relative to the
byte-vocab Wave 0-R2 matched-budget baseline pair (matrix BPB `B_M`,
vector BPB `B_V`, gap `G = B_M − B_V`):

```
recovered_frac(X) = (B_M − B_{swap X}) / G
```

- **Named dominant site**: some `X` has `recovered_frac(X) ≥ 0.5` at `≥2`
  seeds (Wave B), while every other screened `X'` has
  `recovered_frac(X') ≤ 0.2` — a clean, attributable win, matching Stage
  0's "large, qualitative jump off the floor" screening philosophy (§6.1
  there).
- **Distributed tax**: no single `X` clears `0.5`, OR the swaps'
  `recovered_frac` values are non-additive (their sum, if all combined,
  would predict `>>1.0` given single-factor measurements — a signature of
  interacting, not independent, mechanisms) — reported honestly as "the gap
  is a tax spread across N components, here is the measured breakdown," not
  forced into a false single-cause story.
- **H_d verdict (definitional, must resolve regardless of the above;
  scoping tightened in revision 2.1, N2):** CONFIRMED if extended budget
  alone closes `≥50%` of the honestly-accounted gap without any component
  swap — evidence may come from either arm (Wave 0-R1's Regime-1
  continuation if it runs, or Wave 0-R2's multi-seed byte-scale arms).
  **FALSIFY authority belongs to arm (b) (Wave 0-R2's from-scratch
  Regime-2 extended arms) ONLY:** FALSIFIED only if those from-scratch
  extended runs **plateau** (slope indistinguishable from zero across the
  final 25% of the extension) well above the vector reference, across
  seeds. **0-R1 can never falsify H_d** — a moment-less (no AdamW
  moments/scheduler/RNG, N2) re-warmed continuation at 1.5× total budget
  that plateaus proves nothing about a properly-sized longer schedule
  (Stage 0's precedent needed 2.5–6×); 0-R1 contributes CONFIRM or
  INCONCLUSIVE evidence only. **INCONCLUSIVE branch (mirrors Stage 0's
  H_undertrained handling — "cannot assess, never approached a plateau
  within budget"):** if an extended arm is still visibly descending at its
  cutoff without having closed 50%, that arm reads **INCONCLUSIVE — not
  FALSIFIED**: report the measured trajectory and its slope at cutoff,
  trigger stretch item 1 if budget allows (and if 0-R1 ran at all), and
  read every component-swap `recovered_frac` as conditional on an
  undertrained regime.
- **H_h verdict:** the analytic half is CONFIRMED pre-launch (V1 + this
  document's §5.1 agree: 11.8–14.9× vs 28–32×, far past the pre-registered
  ≥1.5× discrepancy bar); Wave −1's instrumented count either corroborates
  (expected) or reopens it. The *consequence* statement is fixed either
  way: the original comparison is restated as never-FLOPs-matched (§2.4),
  and Stage G's own accounting (per-run analytic FLOPs, params, tokens —
  §9 attack #1) replaces it.

---

## 9. Attack-yourself: ways this design could mislead

1. **FLOPs vs params vs tokens-seen — which is held fixed, and why (the
   brief's explicit ask).** Stage G's Wave A/B component swaps hold
   **step count / tokens-seen** fixed within a comparison (both arms of a
   swap see identical data), **not** FLOPs — because swapping a component
   changes per-step FLOPs by construction, and re-matching FLOPs per swap
   would require re-deriving a different step count for every cell,
   reintroducing exactly the sizing ambiguity §2.4/H_h showed the original
   comparison died of. Every run logs param count, analytic FLOPs/token
   (§5.1-style), and tokens-seen, so a reader can re-derive "matched"
   under any of the three conventions post-hoc from the raw numbers rather
   than trusting one convention baked into the manifest. **Consequence,
   stated plainly**: a `recovered_frac` measured at matched tokens-seen
   could read differently if re-computed at matched FLOPs (a swap that
   adds a cheap component looks *better* at matched-tokens than at
   matched-FLOPs). This is not fixed by this design — it is disclosed,
   with the raw-log mitigation above, exactly as the brief asked. V1 makes
   this attack concrete: the original headline changed materially under
   re-accounting (LoopFormer's "win" was on ~half the compute, §2.4).
2. **The byte-vocab harness (Regime 2) is not the regime the original
   finding was measured in.** Every Wave 0-R2/A/B result is at byte vocab,
   d=32, small scale, single-GPU. A component-swap win there could fail to
   transfer to BPE-vocab/8-GPU-DDP/~5M-param scale for reasons unrelated
   to the named mechanism (e.g. a fix that helps when embed/output tables
   are ~60% of params might do nothing when they're 93%, per §2.2's own
   numbers). Revision 2 makes this attack *more* biting, not less: the
   core manifest's only Regime-1 cell (Wave 0-R1) tests H_d, not the
   component swaps — intervention-transfer evidence now exists only if
   stretch item 2 funds (F1/V1 trade-off, accepted and stated; §10's
   claims downgraded to match).
3. **The hand-derived FLOP arithmetic has now needed TWO corrections** —
   revision 1's drafting caught an attention-score undercount (`O(L·hd)`
   vs `O(L·d²)`, a ~2.5× swing), and V1's independent verification caught
   the AdaLN per-token-vs-per-batch overcount (another ~1.5× swing on the
   LoopFormer side). Both are disclosed specifically to justify the
   process requirement (§12(iii)): no by-hand FLOP number in this program
   is load-bearing until independently re-derived or instrumented. The
   §5.1 figures are now double-derived and marked verified; Wave −1's
   instrumented count is the final check.
4. **Interaction effects across the OFAT axes.** Depth-matching (H_g),
   iteration conditioning (H_i), and the Kronecker-projection question
   (H_b) are not independent — a depth-matched model has fewer independent
   weight sets, which changes both the value of a conditioning signal and
   the effective restriction of the projection family. A win attributed to
   one in isolation might partially be latent another. Mitigated the same
   way Stage 0 handled its interaction risk (§8 attack #5 there): if Wave
   A produces zero survivors singly, a combined probe (top-2 factors
   together) is pre-registered before declaring a distributed-tax verdict,
   funded from measured under-run.
5. **The prior-setting exercise itself is demonstrably fallible.** The
   attack round showed revision 1 mislabeled the evidentiary content of
   three prior runs (F3) — assigning "downgraded" priors to H_a/H_c on
   comparisons that never varied those axes. All §3 priors are therefore
   ranking heuristics for wave ordering only; they carry zero evidentiary
   weight in the final write-up, and a surprising Wave A result on any
   axis is taken at face value.
6. **The transfer claim available at the end of Stage G is weak by
   construction.** Even if stretch item 2 funds, a single reduced-step
   Regime-1 intervention cell shows only "the fix moves the gap in the
   right direction at reduced budget" — not publication-grade transfer. A
   full-scale multi-seed Regime-1 retraining is explicitly out of scope
   (§10) and sequenced as follow-on work.
7. **A "distributed tax" verdict is unfalsifiable if defined too loosely.**
   §8's numeric bars (`≥0.5` dominant, `≤0.2` for all others,
   non-additivity as the distributed signature) are pre-registered so "we
   tried everything and nothing clearly won" cannot be read as either a
   forced dominant-site claim or a shrug — the write-up must report the
   measured `recovered_frac` for all ten hypotheses' interventions,
   dominant or not, exactly as Stage 0's §12.6 table did for its five.
8. **New-architecture components must re-pass a correctness/shape smoke
   test before trusting their numbers** — same discipline as Stage 0
   attack #7, reused verbatim: the depth-matched control (H_g), the
   conditioning module (H_i), and the tied head (H_f — especially, given
   F2's precedent of silently-null-testing dead code) all touch model
   wiring directly (§12(iv)–(v)).
9. **Task E's H_e evidence (§1, §2.5) is from a single-Z-bottleneck
   associative-memory encoder, not an autoregressive LM.** Citing it as
   "existing evidence the task-mismatch hypothesis is real somewhere" is
   sound; treating it as evidence *for this specific LM architecture pair*
   is not — which is exactly why H_e is gated to Wave C and requires a
   dedicated data-generator build (§4 item 9).
10. **GPU-h unit mismatch risk, Regime-2 side.** The 1.0/0.25 GPU-h flat
    units are extrapolated from Run 12's d=32 8-GPU throughput divided by
    8 — a linear-scaling assumption that ignores per-GPU batch effects and
    DDP overhead in an unknown direction, and no byte-vocab d=32
    single-GPU cell has ever been timed (F4). Wave −1's 2-cell d=32
    calibration exists specifically to catch this before Waves 0/A commit
    the bulk of the budget — the same role Stage 0's Wave −1 played
    (MAJOR-3 there).
11. **The Wave 0-R1 continuation (if it runs at all — fallback is
    mainline, N1) is a cross-run, cross-schedule, moment-less
    comparison.** It resumes a 3-month-old checkpoint (retired pod,
    re-staged data — §12(vi)) that contains **only
    `raw_model.state_dict()` — no AdamW moments, scheduler, or RNG state
    (N2)** — with a re-warmed schedule Run 12's from-scratch cosine
    schedule never had; its BPB trajectory is a *lower bound* on what a
    properly-sized longer schedule could reach (§4 item 2(a)), which is
    exactly why it holds no FALSIFY authority over H_d (§8), and any
    harness drift (data ordering, library versions) lands in the measured
    delta. Mitigations: bit-identical script/config where possible,
    data re-staging verified against the surviving log stats (§12(vi)),
    and **the drift canary: a step-0 eval of the loaded checkpoint BEFORE
    any optimizer step (N8)** — it must reproduce Run 12's final eval
    (BPB 1.67 / PPL 72.4 at T=8) within eval noise *before* the re-warmed
    schedule or the fresh optimizer state can move any weight. If the
    step-0 canary fails, the cell is diagnosed before being interpreted,
    not averaged over. Residual risk accepted and stated.

---

## 10. What Stage G does and does NOT show

- **Shows, already, before any GPU runs (V1):** the existing headline
  comparison was never FLOPs-matched in either direction — the sizing used
  wall-clock as a FLOPs proxy (verified analytic ratio 11.8–14.9× vs the
  28–32× used), and the cited LoopFormer best (BPB 0.87, step 21,500)
  consumed only ~0.48–0.61× of Matrix's training compute. The honest
  interim restatement of the negative result, pending Wave 0: "LoopFormer
  reached ~1.9× better BPB on roughly half the training FLOPs; Matrix's
  converged number is unmeasured."
- **Shows, if a dominant site is named:** a specific, falsifiable,
  component-level account of where the matrix-token LM's per-FLOP signal
  loss occurs, upgrading `STATE.md`'s current "the quality gap is
  algorithmic, not a speed problem" (established by the cheap-ops
  waterfall, `research/waterfall-cheap-ops-april2026.md` — which asked "can
  we make matrix ops cheaper" and correctly answered no, but never asked
  "which specific operation/config choice is responsible") into a named,
  addressable finding.
- **Shows, if distributed tax:** a decisive, budget-bounded negative — the
  gap does not trace to one fixable thing, redirecting future compute
  toward either accepting the per-FLOP cost as a genuine, structural
  property of matrix representations at this scale, or toward H_e-style
  task selection (composition-heavy data) rather than architecture tweaks.
- **Answers definitively, regardless of the above:** H_h (already
  answered — the accounting artifact is real and quantified) and H_d,
  whose verdict authority sits with arm (b)'s from-scratch Regime-2
  extended runs (the only arm with FALSIFY power — N2, §8), subject to
  the pre-registered INCONCLUSIVE branch if still descending at cutoff.
  The Regime-1 trajectory (Wave 0-R1) contributes CONFIRM/INCONCLUSIVE
  evidence only, and only if the retired-pod checkpoint is recovered —
  on the mainline fallback (N1) the Regime-1 datum is explicitly declared
  out-of-budget rather than silently substituted by the byte-scale
  result (§4 item 2(a)).
- **Does NOT show:** that any winning component fix transfers to the
  original regime — the core manifest contains **no** Regime-1
  intervention cell (V1's re-prioritization spent that budget on the H_d
  continuation); transfer evidence exists only if stretch item 2 funds,
  and even then only at "gap moves in the right direction at reduced
  budget" strength (§9 attacks #2/#6). A full-scale, multi-seed retraining
  is follow-on work only if Stage G names a dominant, fixable site.
- **Does NOT show:** whether matrix representations are *ever* competitive
  per-FLOP with vector representations on natural language at any scale —
  only whether the specific ~5M-param, BPE-vocab, T=8-iteration comparison
  this project already ran is explained. H_e's Wave C, if reached, is the
  closest this document gets to the broader question, and even that is
  scoped to one composition-heavy synthetic corpus, not natural language.

---

## 11. Sequencing — what this unlocks

1. **Stage G (this doc)** — go/no-go on whether the BPB gap has a named,
   fixable, dominant component or is a distributed tax; H_h already
   resolved (V1); H_d resolves at Wave 0 (or exits via its INCONCLUSIVE /
   out-of-budget branches, both pre-registered).
2. **Immediately, independent of any GPU run:** `EXPERIMENT_LOG.md` Run 14
   and `STATE.md`'s "matched FLOPs" language get corrected (handled
   separately per V1) — the negative result's honest restatement (§10
   bullet 1) replaces the current phrasing wherever it is cited, including
   the pebble-ai-site findings pages that quote BPB 1.67-vs-0.87 as
   FLOPs-matched.
3. **If a dominant, fixable site is found →** a targeted fix is retrained
   at full Regime-1 scale as a dedicated follow-on experiment — the actual
   publication-grade re-measurement of the gap, sequenced after Stage G
   because a fix validated only on Regime 2's diagnostic harness is not
   yet a claim about the original finding.
4. **If distributed tax →** publish the component-level breakdown as the
   honest, quantified version of the existing negative — a genuine upgrade
   in rigor (a measured breakdown plus a corrected compute accounting, not
   an unexplained gap), matching Stage 0's framing of its diagnostic-pass
   condition as a legitimate outcome.
5. **Either way →** H_e's Wave C (if reached) feeds directly into the
   existing Chapter 2.5/Task E sequencing (`STATE.md` "Path Forward")
   rather than opening a new thread — a composition-heavy LM corpus is the
   natural bridge between Task E's already-confirmed synthetic result and
   Chapter 3's real-data matrix-native-on-BPE plan.

---

## 12. Reproducibility pointers / build requirements

- This design: `matrix-thinking/STAGE_G_DESIGN.md` (revision 2; revision-1
  → revision-2 changes indexed in the header changelog against the attack
  round's finding IDs F1–F6 and the H_h verification V1).
- Diagnosed artifacts (read, not modified, until the build phase):
  `experiment-runs/8xh100-session1/round2_matrix_script.py` (`MatrixThinker`,
  the exact model trained for Runs 12/14/15), `loopformer_96K_script.py`
  (`LoopFormer`, Runs 13/14), their paired logs — `round2_matrix_train.log`
  and `loopformer_96K_full.log` are the throughput/loss-curve/best-step
  sources of truth (`*BEST*` marker at eval step 21,500 re-verified this
  session; `loopformer_96K_results.json` and the SUMMARY are 0 bytes — the
  run died without writing finals, another reason the raw log is
  authoritative).
- Reusable existing code the build phase should wire into, not rewrite:
  `matrix-thinking/scripts/run_ablation_matched.py` (Run 22's byte-vocab
  matrix-vs-flat harness — the closest precedent for Regime 2);
  `matrix-thinking/chapter2/task_e.py` (H_e's Wave C generator source, if
  reached). **Explicitly NOT reusable as-is:**
  `matrix-thinking/src/matrix_output_heads.py::TiedMultiProbeHead` (F2 —
  dead code that does not tie; see requirement (v)).
- Prior data this design is grounded in: `EXPERIMENT_LOG.md` Runs 12–15,
  18, 22, 25 (exact figures cited throughout §1–§2, with the Run 14
  best-step and FLOPs figures corrected per V1 — the log's own correction
  is tracked separately); `STATE.md` "Honest negative results" and
  "Chapter 2 — STATUS"; `TASK_E_FINDINGS.md` (2026-07-02, the H_e prior);
  `research/waterfall-cheap-ops-april2026.md` (the "speed ≠ quality"
  precedent this document extends from "is it fixable by making ops
  cheaper" to "where specifically is it lost").
- Sibling design this follows in form (not subject): `STAGE0_DESIGN.md` —
  same "read the forward pass, derive falsifiable mechanisms before
  proposing fixes" method, same wave/gate manifest structure, same
  numeric-bar success criteria, applied to a different model family and a
  different honest negative.
- **Build requirements, carried forward as blocking (mirroring Stage 0
  §6.3):**
  (i) manifest run-tags must carry a `regime` field (byte-vocab vs
  BPE-vocab) and a `variant` field (which axis was swapped), or cells
  collide under a naive resume-by-name scheme;
  (ii) every run logs analytic FLOPs/token, param count, and tokens-seen
  independently (§9 attack #1's mitigation) — not just wall-clock;
  (iii) no by-hand FLOP number is load-bearing until independently
  re-derived or instrumented (§9 attack #3 — two corrections already);
  Wave −1's instrumented count is reviewed by a fresh-context agent before
  Wave 0 launches;
  (iv) any component-swap that changes model wiring (H_g's depth-matched
  control, H_i's conditioning module, H_b's gated rank-swept family)
  passes a forward/backward/gradient-check smoke test before its numbers
  are trusted (§9 attack #8);
  (v) **F2's tied-head build task is blocking before Wave A:** implement
  §4 item 4's genuinely-tied head (the existing `TiedMultiProbeHead` is
  dead code that never reads the embeddings it stores); the
  tie-verification unit test (in-place perturbation of `embed_u.weight`
  changes logits; tensor-identity assertions; no vocab-sized untied
  parameter) must pass and — per CLAUDE.md's negative-unit-test hard
  rule — must be *run to completion*, not merely written;
  (vi) **Wave 0-R1 prerequisites, checked at Wave −1 (zero GPU) —
  fallback is the mainline assumption (N1):** local checkpoint recovery
  is already ruled out (no `.pt` under `experiment-runs/8xh100-session1/`;
  the SSD's `best.pt` is a byte-agnostic-era model, pickle keys
  `byte_embed_for_expand`/`ln1`/`ln2`); Wave −1 checks the one remaining
  possibility — Run 12's checkpoint on the retired `/toy_story_slam` pod
  volume. Data integrity for any Regime-1 cell: the BPE-tokenized corpus
  re-staged on the Brev cluster and verified against the stats that
  survive in `round2_matrix_train.log` (train_tokens=2,188,386,413; vocab
  50257; the logged sources line) — **not** the archived `data_meta.json`,
  which is 0 bytes (N6). The pre-registered outcome if either prerequisite
  fails is §4 item 2(a)'s aftermath paragraph (declare the Regime-1 datum
  out-of-budget, flow the budget down the stretch list; do not silently
  substitute).
- **Next:** Wave −1's zero-GPU tasks (instrumented FLOP corroboration,
  step-0 std check, checkpoint/data recoverability) + the F2 tied-head
  build task → fresh-context audit → Wave −1 GPU calibration on the Brev
  cluster → Waves 0/A/B per §7.

---

## 13. Results — Wave −1/0 (gap baseline) — 2026-07-02

**Status: Wave −1 COMPLETE (2/2 GPU calibration cells + the H_j Regime-1
zero-GPU check); Wave 0-R2 COMPLETE (10/10 cells, 0 failed); Wave A
LAUNCHED (5 cells, running, not analyzed here).** Raw data: archived from
`youthful-indigo-turkey:/home/nvidia/stageg/results/stageg/{wave-1,wave0}/`
to `experiment-runs/2026-07-02_stageg_waves/{wave-1,wave0}/` (per-run JSON
with dense ≤500-step checkpoint trajectories, logs, `SUMMARY.txt`/
`PROGRESS.txt`). All numbers below are pulled directly from the per-run
JSONs (`final_evals.T8.val_bpb`, `checkpoints[].evals.T8.val_bpb`,
`wall_s`, `analytic_flops_per_token`, `step0_entry_std`, `n_params`), not
hand-derived, per §12's "no by-hand number is load-bearing" build
requirement. Wave 0-R1 (the Regime-1 warm-continuation) was **not**
built — `run_stageg_sweep.py`'s own manifest docstring confirms N1's
mainline fallback fired as planned; **no Regime-1 H_d datum exists in this
archive**, disclosed as a standing caveat throughout this section.

### 13.1 Wave −1 — timing calibration + the H_j zero-GPU check

| Cell | steps | n_params | wall_s | step0_entry_std |
|---|---|---|---|---|
| matrix, d=32 | 500 | 290,328 | 1,030.4 | 1.0090 |
| vector, d=32 | 500 | 300,976 | 25.4 | 0.0678 |

Wall-clock ratio 40.6× at the calibration cell, falling inside Wave 0's
own 3-seed range (39.6–43.9×, §13.2) — the provisional standard-cell step
count (3,000, §5.4/N5) was retained, not re-priced.

**H_j Regime-1 check (design §4 item 1, F6): independently re-run this
session**, since no artifact existed on disk (`wave_neg1.py`'s
`check_step0_std` only prints; it does not persist a result file). Run
verbatim (unmodified `common.py` `MatrixThinker`-equivalent embedding
logic / `LoopFormer`, d=32/n_embd=96/vocab=50257, 5 seeds, on-box CPU,
zero GPU-h), output saved to
`experiment-runs/2026-07-02_stageg_waves/wave-1/h_j_regime1_step0_std.json`:

- Matrix (`u⊗v`, default init): entry-std = 0.9983 (seeds: 1.0007,
  0.9965, 1.0009, 0.9978, 0.9958)
- LoopFormer (`wte`, explicit std=0.02 init): entry-std = 0.0201 (seeds:
  0.0203, 0.0202, 0.0199, 0.0197, 0.0202)
- **Ratio: 49.7×** — matches F6's ~50× prediction closely.

This is a *different, exact-Regime-1* measurement from the
`step0_entry_std` field logged inside the Stage G byte-vocab (Regime 2)
harness's own JSONs (§13.1 table above, and every Wave 0 cell: matrix
≈1.004–1.009, vector ≈0.067–0.068, ratio **≈14.9×**). The two numbers
should not be conflated: Regime 2's "vector" reference is
`VectorReferenceModel`'s `wte+wpe` sum at the same explicit std=0.02
init, not LoopFormer itself, and the two harnesses differ in more than
just vocab size. What the Regime-2 numbers *do* corroborate is the
**direction and order of magnitude** — matrix-side entry-std is
consistently ~1.0 while every vector-side reference tested (LoopFormer
exactly, and the Regime-2 mirror) sits at 0.02–0.07 — not the specific
49.7× figure, which is a Regime-1-only datum.

### 13.2 Wave 0-R2 — matched-budget gap baseline (3,000 steps, 3 seeds/side)

| Family | Seed | T8 val_bpb | wall_s | n_params |
|---|---|---|---|---|
| matrix | 0 | 3.5600 | 6,168.5 | 290,328 |
| matrix | 1 | 3.5597 | 6,168.6 | 290,328 |
| matrix | 2 | 3.5458 | 6,158.0 | 290,328 |
| vector | 0 | 3.2300 | 140.4 | 300,976 |
| vector | 1 | 3.2207 | 156.0 | 300,976 |
| vector | 2 | 3.3028 | 151.2 | 300,976 |

Matrix mean T8 BPB = **3.5552**, vector mean = **3.2511**. **G (the §8
`recovered_frac` anchor) = 3.5552 − 3.2511 = 0.3040** at matched
3,000-step tokens-seen — a correction on the ≈0.28 figure this task was
scoped from; the measured mean-mean gap over all 3 seeds/side is 0.304,
not 0.28 (min-vs-max pairing gets as low as 0.243, max-vs-min as high as
0.339 — 0.304 is the defined, seed-averaged quantity).

Per-seed wall-clock ratio (matrix/vector, paired by seed): 43.93×,
39.55×, 40.73× — consistent with Wave −1's 40.6× calibration cell (§13.1).

### 13.3 Wave 0-R2 — extended arms (9,000 steps = 3×, 2 seeds/side): the gap widens, both tails flat

| Family | Seed | T8 val_bpb | wall_s |
|---|---|---|---|
| matrix | 0 | 3.4304 | 18,467.6 |
| matrix | 1 | 3.4721 | 18,506.2 |
| vector | 0 | 2.5848 | 428.8 |
| vector | 1 | 2.5959 | 429.6 |

Matrix mean = 3.4512, vector mean = 2.5904 → **G₉ = 0.8609** — the gap
**widens** from 0.304 to 0.861 under 3× extension, matching the
pre-supplied ≈0.86 figure closely. Matrix's own improvement 3,000→9,000
steps is only **Δ = 0.1039 BPB** (3.5552→3.4512); vector's is
**Δ = 0.6608 BPB** (3.2511→2.5904) — the vector reference improves the
extra budget ~6.4× more than matrix does.

**Tail-flatness check (both families, last 500 logged steps, step
8,500→9,000):** matrix s0 −0.000406, matrix s1 −0.000473, vector s0
−0.000004, vector s1 −0.000467 — every delta is an order of magnitude
inside the ±0.002 noise-floor bar. **Neither family is still visibly
descending at the 9,000-step cutoff.** The pre-registered
still-rising→INCONCLUSIVE branch (§8) does **not** trigger.

### 13.4 H_d verdict at Regime 2 (§8 rules)

Per §8: "FALSIFIED only if [the] from-scratch extended runs **plateau**
(slope indistinguishable from zero across the final 25% of the
extension) well above the vector reference, across seeds." Both criteria
are met: the tails are flat (§13.3) across both seeds, and matrix's final
BPB (3.4512) sits *further* above the vector reference (0.861) than it
did at the matched budget (0.304) — the opposite of "closing the gap."
Extended budget alone recovers only **34.2%** of G (0.1039/0.3040),
short of the 50% CONFIRM bar, and the arm that would need to still be
descending for an INCONCLUSIVE read is not (§13.3).

**H_d is FALSIFIED at Regime 2**, on arm (b)'s sole FALSIFY authority
(§8, N2): the BPB gap is not matrix-side undertraining relative to its
own budget — additional budget benefits the vector reference far more
(Δ≈0.66 vs Δ≈0.10 over the same 3× extension) and the gap grows, not
shrinks. **Standing caveat, restated per N1/N2:** this verdict covers
Regime 2 (byte vocab, d=32) only. The Regime-1 arm (Wave 0-R1, the
warm-continuation of Run 12's original BPE-vocab/8×H100-DDP config) was
voided before this wave launched — Run 12's checkpoint is confirmed not
locally recoverable (§4 item 2(a), N1) — so **no Regime-1 H_d datum
exists**, and this verdict cannot be extended to the original ~5M-param
BPE-vocab comparison without the regime-transfer caveats of §9 attack #2.

### 13.5 FLOPs / wall-clock accounting (H_h corroboration)

Analytic FLOPs/token at the Regime-2 standard cell, read directly from
every archived JSON's `analytic_flops_per_token` field (identical across
all 12 cells, as expected — same `mat_dim`/config):

- Matrix: non-causal 226,580,480; causal-exact 159,471,616
- Vector: non-causal 4,140,240; causal-exact 2,829,520
- **Ratio: 54.7× (non-causal), 56.4× (causal-exact).**

This corrects the ≈71× figure this task was scoped from — the actual
Regime-2 analytic ratio is ≈55×, distinct from (and not directly
comparable to) Regime 1's verified 11.8–14.9× (§5.1): the byte vocab
shrinks the head-GEMM term that dominates Regime 1's total on *both*
sides (§2.2), so the backbone-width term (H_g's 10.7× `d²/n_embd` ratio)
carries proportionally more of Regime 2's total — consistent with, not
contradicting, §5.1's own backbone-only-vs-total distinction.

Measured wall-clock ratio (§13.2/§13.1): 39.6–43.9× across 4 independent
cells (3 gap-baseline seeds + the Wave −1 calibration cell), i.e. still
running ~30–40% hotter than the 54.7–56.4× analytic ratio would predict
in the *opposite* direction of Regime 1's H100 kernel-launch/small-GEMM
underutilization story (there, wall-clock ran *above* the analytic
ratio, 28–32× measured vs 11.8–14.9× analytic — a ~2× gap the *same
direction* here would predict wall-clock closer to ~110–170×, not the
~40× actually measured). Flagged, not resolved: Regime 2's wall-clock
ratio sits *below* its analytic ratio, the reverse of Regime 1's pattern,
plausibly because Regime 2 is single-GPU (no DDP overhead) and much
smaller (less kernel-launch dominance per step) — a genuine cross-regime
asymmetry worth a instrumented-profiler pass if H_h's mechanism is
revisited, not something this wave was scoped to explain.

### 13.6 Compute

12 GPU cells (2 Wave −1 calibration + 10 Wave 0-R2), summed `wall_s` =
57,830.7s = **16.06 GPU-h** actual (serial-sum; single-GPU cells, so
serial-sum = GPU-h exactly) — matrix cells dominate at 15.69 GPU-h
(97.7% of the total), vector cells only 0.37 GPU-h. This corrects the
≈10.5 GPU-h figure this task was scoped from. Breaking it down against
§7's provisional prices: Wave −1's GPU calibration (0.293 GPU-h actual
vs 0.3 GPU-h priced) landed almost exactly on price; **Wave 0-R2 ran
15.77 GPU-h actual against an 11.25 GPU-h price — a ~40% overrun**,
because the provisional Regime-2 flat unit (1.0 GPU-h/vector-side 0.25
GPU-h per 3,000-step cell, §5.4) undershot the matrix side specifically:
the measured matrix-side 3,000-step cell costs ≈1.71 GPU-h (6,158–6,169s),
71% over its 1.0 GPU-h provisional price, while the vector side (≈0.04
GPU-h actual, 140–156s) came in far *under* its conservative 0.25 GPU-h
price. The zero-GPU H_j re-run (§13.1) added negligible CPU time, not
counted in GPU-h. **Budget note:** with Wave A already launched (5 cells,
priced 5.0 GPU-h) on top of this ~40%-over Wave 0-R2 actual, the
mainline core's assumed headroom (§7: 23.3 GPU-h mainline core) is
tighter than priced — worth a check-in before Wave B's `--winner` gate,
not an immediate stop.

---

## 14. Results — Wave A/B (component-swap screen + gated H_b family): the named dominant site — 2026-07-02

**Status: Wave A OFAT screen COMPLETE (5/5 axes + the pre-registered
combined probe, 6/6 cells); N3 correction rerun COMPLETE (1/1); H_b's
gated rank-swept family COMPLETE for r1/r16 (finished mid-session, see
correction note below) and r3_nl3 (3/3 seeds); r4 COMPLETE 3/3 seeds
(seeds 1–2 finalized after this section first landed — dated note in
§14.3); the Wave-B vector-family capacity control COMPLETE (3/3
seeds). All 18 cells `complete: true`.** Raw data:
archived from
`youthful-indigo-turkey:/home/nvidia/stageg/results/stageg/waveA/` (both
the `wA_`- and `wB_`-prefixed files live in this one directory on the
box) to `experiment-runs/2026-07-02_stageg_waves/waveA/` (per-run JSON,
logs, `SUMMARY.txt`/`PROGRESS.txt`/`AGGREGATE.json`). All numbers below
are pulled directly from the per-run JSONs (`final_evals.T8.val_bpb`,
`n_params`, `analytic_flops_per_token`, `wall_s`, `steps_completed`,
`complete`), not hand-derived, per §12's "no by-hand number is
load-bearing" build requirement. `recovered_frac` uses §8's fixed
anchor, taken from §13.2: `recovered_frac(X) = (3.5552 − B_X) / 0.3040`.

**Correction on an in-session stale read (flagged explicitly — a
within-cell read-timing risk, distinct from §13.4's baseline-architecture
H_d verdict).** An earlier read of this wave, taken while
`h_b_factored_r1` and `h_b_factored_r16` were still training, reported
r1's step-1000 checkpoint (bpb 3.5181, `recovered_frac` +0.122) and
r16's step-1000 checkpoint (bpb ≈2.98) as if final. Both cells kept
improving substantially past step 1000 — r1: 3.518→3.342 (step
1000→3000), r16: 2.981→2.330 — and finished (3,000/3,000,
`complete: true`) partway through drafting this section. The true final
values (§14.3) are far stronger than the stale read: r1 +0.701 (5.7× the
stale +0.122), r16 +4.029. This is a **within-cell late-transition/
read-timing risk** — a checkpoint from an in-flight run is not a
substitute for `complete: true` — distinct from §13.4's H_d verdict,
which concerns the **baseline architecture** plateauing under 3×
extended budget, not an in-flight variant's mid-run checkpoint being
misread as final. Every number below is drawn only from cells with
`complete: true` and `steps_completed == steps`; cells still training
are marked **PENDING** with their current step count and a
separately-flagged, clearly non-final checkpoint reading.

### 14.1 Wave A — OFAT screen (5 axes) + the pre-registered combined probe: the baseline is locally optimal

Five single-axis swaps from the matrix-all baseline (290,328 params,
226,580,480 FLOPs/token non-causal / 159,471,616 causal-exact — the
§13.2 gap-baseline architecture), 1 seed each, 3,000 steps:

| Hypothesis | Variant | n_params | FLOPs/tok (non-causal) | T8 val_bpb | `recovered_frac` |
|---|---|---|---|---|---|
| H_j (embed init scale) | `h_j_init_matched` | 290,328 | 226,580,480 | 3.5560 | **−0.003** |
| H_a (embed rank) | `h_a_full_rank` | 1,519,128 | 226,580,480 | 3.5562 | **−0.003** |
| H_i (iteration conditioning) | `h_i_iter_cond` | 309,416 | 226,580,480 | 3.5638 | **−0.028** |
| H_g (depth structure) | `h_g_depth_matched` | 142,470 | 56,711,168 | 3.5704 | **−0.050** |
| H_f (tied output head) | `h_f_tied_bilinear` | 280,089 | 227,020,800 | 3.6047 | **−0.163** |
| Combined probe (H_j+H_g+H_i) | `combined_hj_hg_hi` | 161,162 | 56,711,168 | 3.5671 | **−0.039** |

All six cells are negative or ≈0 — the pre-registered combined probe
(§9 attack #4, triggered because zero single-axis cell survived — see
`models.py`'s `VARIANT_AXES` comment for the exact registered rationale
for combining H_j+H_g+H_i specifically) does not rescue the screen
either. **Wave A verdict: the matrix-all baseline is locally optimal
among every training-setup/plumbing flip tested** (embedding init scale,
embedding rank, iteration conditioning, depth structure, output-head
tying, and their combination). None of these axes explain the gap.

### 14.2 N3 correction — the H_f tying penalty was an H_j artifact, not a tying penalty

Design §4 item 4 pre-registered this exact rerun: "if the primary H_f
cell still lands far below the matrix-all baseline, H_f is re-run once
on the matched-init (H_j-fixed) variant before any verdict is recorded"
(guarding against the tied-bilinear head's saturated step-0 softmax
cratering the cell *for an H_j reason*, not a tying reason).
`h_f_tied_matched_init` (same `tied_bilinear` head class,
`embed_init="matched"` added — a documented two-axis flip, deliberately
excluded from §14.1's single-axis OFAT table):

| Variant | n_params | FLOPs/tok (non-causal) | T8 val_bpb | `recovered_frac` |
|---|---|---|---|---|
| `h_f_tied_matched_init` | 280,089 | 227,020,800 | 3.5533 | **+0.006** |

This is the only cell in the whole screen (OFAT + combined + N3) whose
`val_bpb` sits at or below the baseline (3.5533 ≤ 3.5552). Since the
*only* delta between `h_f_tied_bilinear` (−0.163) and
`h_f_tied_matched_init` (+0.006) is the embedding-init flag — the tied
head implementation is identical in both cells — **the −0.163 tying
penalty was entirely an H_j×H_f init-scale artifact, not evidence
against weight tying itself.** The pre-registered rerun rule did exactly
its job: it prevented a wrong attribution ("tying hurts") from being
recorded.

### 14.3 H_b's gate fires: the factored-projection dose-response

§4 item 8 / §7 flow-down item 4's gate condition — "deployed in Wave B
only if [the] depth-matched control doesn't close the gap" — fired on
§14.1's H_g result (−0.050, does not close the gap). Per §6.1's
pre-registered design, `FactoredDenseProjection`
(`vec(M) → Linear(d²,r) → silu → Linear(r,d²) → reshape(d,d)`, replacing
`RowThenColProjection`'s Kronecker-separable `silu(A@M)@B` on every
Q/K/V/O/gate projection) was swept over `r ∈ {1,4,16}`, plus one
param-matched confirmation point built after the initial r4 result
(§14.4 explains why):

| Cell | proj_rank | n_layers | n_params | FLOPs/tok (nc) | Seeds complete | T8 val_bpb | `recovered_frac` |
|---|---|---|---|---|---|---|---|
| `h_b_factored_r1` | 1 | 8 (unchanged) | 290,328 (exact baseline match) | 145,315,840 | 1/1 | 3.3421 | **+0.701** |
| `h_b_factored_r3_nl3` | 3 | 3 (narrowed from 8) | 289,993 (99.9% of baseline) | 56,514,560 | 3/3 | 3.3167 / 3.3784 / 3.3853 (mean 3.3602) | **+0.641** (per seed: +0.784 / +0.582 / +0.559) |
| `h_b_factored_r4` | 4 | 8 (unchanged) | 781,848 (2.69× baseline) | 153,180,160 | 3/3 | 2.9395 / 2.8940 / 2.9177 (mean 2.9171) | **+2.099** (per seed: +2.025 / +2.175 / +2.097) |
| `h_b_factored_r16` | 16 | 8 (unchanged) | 2,747,928 (9.46× baseline) | 184,637,440 | 1/1 | 2.3304 | **+4.029** |

*(Updated 2026-07-02, post-write-up: `h_b_factored_r4` seeds 1–2
completed — `complete: true`, 3,000/3,000 — at 2.8940 / 2.9177,
replacing the PENDING entry this section originally carried (their
non-final step-2,000 checkpoints read 2.9641 / 2.9915). Both finals
landed at or slightly better than seed 0, tightening the cell to a
3-seed mean of 2.9171 (+2.099) and changing no verdict; §14.4's
r4-vs-capacity-control deltas are restated against the 3-seed mean.)*

Notes verified against `models.py`'s registered rationale and
`common.py`'s implementation:

- **r1 is the cleanest attribution cell in the whole program.**
  `proj_rank=1` is the *only* change from baseline — `n_layers=8`
  unchanged, `param_breakdown` identical to `h_j_init_matched`/the
  baseline architecture at every sub-count (embed 81,920 / think 197,144
  / head 11,264). Params and FLOPs both go *down* relative to Kronecker
  (145.3M vs 226.6M non-causal FLOPs/token) — this cannot be a capacity
  story.
- **r3_nl3 is param-matched (289,993 vs 290,328, 99.9%) but carries a
  disclosed depth confound**: reaching the param-matched band required
  narrowing `n_layers` 8→3 alongside `proj_rank` 4→3 (`models.py`'s own
  comment: "the measured sweep shows r=4 cannot reach the band by
  narrowing layer count alone... (nl=3, r=3) chosen for depth closer to
  the 8-layer winner plus the exact baseline match"). Since H_g's own
  isolated depth effect is −0.050, and r1 (`n_layers=8`) minus r3_nl3
  (`n_layers=3`) at nearly the same param count is
  0.701 − 0.641 = 0.060 — within noise of H_g's own −0.050 — **the depth
  confound plausibly explains nearly all of the r1-vs-r3_nl3 gap**,
  making r3_nl3's +0.641 a conservative read of the pure-projection
  effect, not an inflated one.
- **r4 and r16 are capacity-confounded** (2.69× and 9.46× baseline
  params respectively, `n_layers=8` unchanged) — addressed directly in
  §14.4.
- Every H_b cell uses fewer non-causal FLOPs/token than the Kronecker
  baseline (226.6M), including r16 (184.6M) — the entire family is
  FLOPs-cheaper than what it replaces, at every rank tested.

**§8's formal "named dominant site" bar**
(`recovered_frac(X) ≥ 0.5` at `≥2` seeds, every other axis `≤0.2`)
**is met**: `h_b_factored_r3_nl3` clears it at 3/3 seeds
(+0.784/+0.582/+0.559, all ≥0.5), while every §14.1/§14.2 axis sits at
−0.163 to +0.006. `h_b_factored_r1` — the single-seed, exact-param-match,
zero-depth-confound point — corroborates at +0.701 but does not itself
satisfy the multi-seed requirement (n=1).

### 14.4 The capacity control: vector-at-768K deflates part of the r4/r16 headline

Wave-B control (`capacity_782k`, vector family, `n_embd` 80→152,
everything else unchanged): does the *vector* reference also improve
when given r4's param budget, or does the factored-matrix model hold a
real advantage at matched capacity?

| Cell | Family | n_params | FLOPs/tok (nc) | Seeds | T8 val_bpb | `recovered_frac` |
|---|---|---|---|---|---|---|
| `capacity_782k` | vector | 768,616 | 9,270,708 | 3/3 | 2.7953 / 2.8315 / 2.8475 (mean 2.8248) | **+2.403** |
| `h_b_factored_r4` | matrix | 781,848 | 153,180,160 | 3/3 | 2.9395 / 2.8940 / 2.9177 (mean 2.9171) | +2.099 |

*(r4 row updated 2026-07-02 from seed 0 alone to the 3-seed final mean —
see §14.3's dated note; deltas below restated accordingly.)*

The vector reference **beats r4 by 0.092 BPB** at matched capacity
(3-seed means, 2.8248 vs 2.9171) — r4's apparent "inversion" (beating
the original 300,976-param Wave 0-R2 vector baseline, §13.2, by 0.33
BPB: 3.2511 vs 2.9171) is substantially a capacity effect, not a
matrix-side win: give the vector family the same param budget and it
pulls back ahead. Per-FLOP
the asymmetry is stark: the capacity-matched vector costs 9.27M
FLOPs/token (non-causal) against r4's 153.18M — **≈16.5× more expensive
per token for a worse BPB.** The per-FLOP tax identified at the Wave
0-R2 baseline (§13.5, 54.7× non-causal) shrinks at this cell (the
factored projection is itself cheaper than Kronecker) but does not
close, let alone invert.

### 14.5 Mechanistic point: the binding constraint is Kronecker-separable structure, not operator rank

`RowThenColProjection`'s form, `M ↦ silu(A@M)@B` with
`A,B ∈ ℝ^{d×d}` initialized near-identity, has ~2d² parameters (`d=32`
→ 2,048) but its *linear part* is a Kronecker product on the vectorized
`d²`-dim space (`vec(A@M@B) = (Bᵀ⊗A) vec(M)`, standard identity — Van
Loan 2000, already cited at §1/H_b): `rank(Bᵀ⊗A) = rank(A)·rank(B)`,
which can reach the full `d² = 1,024` if `A,B` are each full rank
(plausible near an identity-plus-noise init). `h_b_factored_r1`'s
`Linear(d²,1)→Linear(1,d²)` also has ~2d² parameters (the exact
param-matched point, by design) but its linear part is a composition
through a width-1 bottleneck — **rank ≤ 1 on the same `d²`-dim space, by
construction, regardless of training.** r1 (operator-rank ≤1) beats the
Kronecker baseline (operator-rank up to 1,024) by +0.701
`recovered_frac` at equal parameter count. **The binding constraint is
therefore the forced row-then-column separability of the Kronecker
form — not how much rank the replacement operator has.** This sharpens
the §14.3 headline and tempers any "low-rank is better" misreading: rank
went *down* (to the theoretical floor) while quality went *up*; what
changed is the functional family (an unstructured, if narrow, dense
bottleneck vs. a highly structured but wide-potential-rank bilinear
form) — consistent with this project's standing distinction between
rank and capacity (CLAUDE.md hard rule: "a low-rank matrix is NOT
low-capacity... rank is only the binding constraint when—"). No cheap
clean control isolates rank from structure directly here (disclosed at
§1/H_b and never resolved — `common.py`'s own `FactoredDenseProjection`
docstring lists the same asymmetry): this reading is supported by the r1
data point, not proven by a dedicated ablation.

### 14.6 Verdict — three parts, all honest

1. **Named dominant site (formally met, §8): the Kronecker-separable
   `RowThenColProjection` restriction.** Relaxing it recovers ~64.1% of
   G at matched params (3-seed `h_b_factored_r3_nl3`, every seed ≥0.5),
   or ~70.1% at the exact-param, zero-depth-confound point
   (`h_b_factored_r1`, n=1). Every other screened axis (init scale,
   embedding rank, iteration conditioning, depth structure, output-head
   tying, their combination) sits at ≤+0.006. §14.5's mechanistic
   reading: the loss is concentrated in the *forced separability* of the
   projection family, not in matrix-ness per se — the H_b winners are
   still matrix-token models (same outer-product embedding, same
   Frobenius-attention machinery, same `MultiProbeHead`).
2. **Capacity explains the rest of the r4/r16 "inversion."** r4 (782K
   params, 3-seed mean 2.9171) appears to beat the original 301K-param
   vector reference by 0.33 BPB, but a capacity-matched vector control
   (768K params, 3-seed mean 2.8248) beats r4 by 0.092 BPB (§14.4) —
   most of the apparent inversion is extra parameters, not the
   projection swap.
3. **Per-FLOP, the vector family still dominates everywhere measured.**
   Even the FLOPs-cheaper H_b family costs ≈16.5× more FLOPs/token than
   a capacity-matched vector model for a worse BPB (§14.4); the original
   Wave 0-R2 per-FLOP ratio (54.7× non-causal, §13.5) shrinks but does
   not close. **The original "signal loss per FLOP" framing survives** —
   restated with a mechanism, not just a magnitude.

**Upstream consequence.** The project's oldest negative result — "matrix
ops lose at matched compute" (STATE.md "Honest negative results", already
corrected once to "never actually FLOPs-matched," 2026-07-02) — now has
a *named mechanism*: the loss concentrates in the Kronecker-separable
structure of the attention/thinking projections specifically, not in
matrix-valued tokens as a representation. A differently-structured
matrix-token model (same outer-product embedding, same Frobenius-attention
machinery, same `MultiProbeHead`, only the internal Q/K/V/O/gate
projection family changed) recovers roughly two-thirds of the gap at
matched parameters while using *fewer* FLOPs than what it replaces —
while the per-FLOP tax against a capacity-matched vector reference
survives at ≈16.5×.

### 14.7 Compute

18 GPU cells archived (10 `wA_`-prefixed + 8 `wB_`-prefixed — note the
file-prefix wave labels do not line up cleanly with the design's §4/§7
wave semantics: the initial `h_b_factored_{r1,r4-seed0,r16}` batch was
filed under the `wA_` prefix by the launcher despite being a gated
Wave-B-per-design intervention; this section organizes results by
hypothesis, not filename — design-level "Wave A" = the 7 files in
§14.1/§14.2, "Wave B" = the H_b confirmation + capacity control in
§14.3/§14.4). Summed `wall_s` across all 18 archived JSONs (final,
updated 2026-07-02 once `h_b_factored_r4` seeds 1–2 completed; all 18
carry `complete: true`) = 83,592.5s = **23.22 GPU-h**. Breakdown: the
7-cell OFAT+combined+N3 screen = 9.54 GPU-h; the H_b family
(r1 + r3_nl3×3 + r4×3 + r16) = 13.15 GPU-h; the vector capacity control
(3 seeds) = 0.53 GPU-h.
