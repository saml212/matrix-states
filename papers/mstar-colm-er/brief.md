# Paper brief — Constant-memory recall (mstar-colm-er)

## Venue

- **Name:** The 2nd Workshop on Efficient Reasoning @ COLM 2026
- **Format:** 4-10 pages of main text, references excluded; official COLM 2026
  LaTeX kit (single column, `colm2026_conference.sty`, GitHub release tag
  `2026`); single PDF — from `papers/mstar-colm-er/venue-requirements.md`
- **Requirements source:** `papers/mstar-colm-er/venue-requirements.md`
  (live-fetched 2026-07-10; CFP https://wdlctc.github.io/efficient-reasoning-2026/,
  template sanction on the workshop's own CFP page)
- **Review style:** double-blind (anonymization grep mandatory)
- **Archival:** non-archival; accepts work under review elsewhere
- **Deadline:** July 19, 2026 AoE

## Thesis (one falsifiable sentence)

> At matched parameters and training tokens on episode-restricted K-way
> associative recall (K=32, K/d=0.5), a two-block fast-weight model holding a
> fixed 32,768-byte matrix state answers at 0.999 accuracy sustained to 8x the
> binding horizon, while a param-matched flat-vector recurrence and a
> param-matched transformer both remain at chance at every KV-cache budget
> tested, and causal state-zeroing localizes the recall to the first block's
> fast-weight state; any of these legs failing on the archived evaluation
> artifacts falsifies the claim.

## Contribution bullets

1. A three-arm matched comparison (params within 2.8%, identical tokens,
   steps, optimizer, and episode seeds) on episode-restricted K-way recall in
   which the fast-weight contender reads acc 0.9995 (mean of 3 seeds, worst
   seed 0.99902) against both baselines at chance; paired-seed CIs exclude
   the pre-registered 0.30 margin with >0.65 to spare. (C1, C2)
2. The constant-memory demonstration: the contender's accuracy is flat at
   >=0.998 for every seed as the same episode is embedded at 454, 902, and
   1798 context tokens with the state fixed at 32,768 bytes, while the
   transformer stays at chance both uncapped and at every KV-cache cap
   M x 32,768 bytes, M in {1,2,4,8,16,32} — capping does not rescue it, and
   the pre-registered degenerate-baseline clause forbids quoting any
   memory-multiplier; the verdict of record is "baseline non-competitive at
   matched params/tokens." (C3, C4)
3. A causal mechanism read: zeroing the first block's fast-weight state
   collapses recall to chance in every seed; zeroing the second block's state
   leaves it unchanged; no linear probe on either raw state recovers the
   bindings (the storage is nonlinear; the model's own downstream forward
   linearizes it). (C5, C6)
4. Honest scope: a multi-hop variant of the task is a trainability/
   seed-variance finding (3/9 contender seeds learn it, 0/9 ablation seeds;
   non-decision-grade pooled CI), not a capability claim; and at stress load
   K/d=0.75 all three arms read chance — the separation is load-bounded.
   (C7, C8)

## Per-section page budget (venue limit chosen: 7.0 main-text pages, inside the 4-10 band)

| Section | Pages | Purpose |
|---|---|---|
| 1 Introduction | 1.0 | KV-cache growth vs fixed-state memory; the claim; contributions |
| 2 Experimental setup | 1.5 | task, three arms, matching table, metric + Nichani caveat, pre-registration |
| 3 Recall separation at matched budget | 1.0 | Table 1 (per-seed acc, CIs vs 0.30 margin) |
| 4 Fixed state vs KV cache at long horizon | 1.25 | Fig 1 (money figure), capped walk, degenerate-baseline clause |
| 5 Where the recall lives | 0.75 | S0/S1 zeroing (Fig 2), nonlinear storage (tap table) |
| 6 Scope: what does not separate | 0.75 | task2 seed-variance table, K48 chance row |
| 7 Related work | 0.5 | fast weights, DeltaNet, MQAR, KV compression, Nichani et al. |
| 8 Limitations | 0.25 | scale, synthetic task, argmax decoding, single arch family |
| 9 Conclusion | 0.25 | one paragraph |
| **Total** | **7.25** | render target <= 7 text pages + tables/figures inside 10pp cap |

(Budget sums to 7.25 written; the 10-page hard cap leaves 2.75 pages of
slack for floats. Render inspection enforces the true cap.)

## Claims-to-evidence-to-figure map

Every row: pre-registered verdict record + raw artifact (path, md5) + display.
All paths relative to repo root `experiment-runs/`. `HTH` =
`matrix-thinking/HEAD_TO_HEAD_DEMO_DESIGN.md` (read-only Source).

| Claim id | Claim (with the number) | Verdict record | Raw artifact (path + md5) | Figure/table |
|---|---|---|---|---|
| C1 | Contender task1 acc_A per-seed 0.99951/1.00000/0.99902 (mean 0.9995); ablation 0.03223/0.03271/0.03687 (mean 0.0339); transformer 0.02710/0.02930/0.02856 (mean 0.0283); chance 0.03125, demonstration bar 0.09375 | HTH §1.40 (verdict of record, 2026-07-10) | `2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_{arm}_task1_sweep_s{0,1,2}_round4.json`, md5s: contender d559e498e8d24c0673727e34377519a3 / 3934ea69c60243b380f86eb3db9fdd70 / 677eb57748f06e59520ca37925ed2309; ablation 491d0eb61f1c29bdc0824ac14eba4ef0 / 2b415704c956b3b54d8786f3fdf58810 / b71b44341d44bb30b5099f7dd9d62957; transformer 988fdae364ad1cb6cd41e6c28ff9b564 / dc57f209053ae9245492724cf9024e17 / 0d715323ffd03c020ddf759a4579d065 | Table 1 |
| C2 | Paired Δ(contender−ablation) mean 0.96558, t-CI df=2 (0.95822, 0.97293); Δ(contender−transformer) mean 0.97119, CI (0.96855, 0.97383); both exclude the pre-registered 0.30 margin | HTH §1.31.1 (frozen tiers, margin pre-registered 2026-07-09) + §1.40 | same nine files as C1 (CIs recomputed from per-seed acc_A; verified 2026-07-10 by the paper agent, matches §1.40 to 5 decimals) | Table 1 |
| C3 | Contender acc_A per-seed 0.99951/0.99829/0.99902 at EVERY horizon H2=454, H4=902, H8=1798 tokens, state fixed at 32,768 bytes (>=0.998 all seeds all horizons) | HTH §1.41 (verdict of record, 2026-07-10) | `2026-07-10_h2h_mstar/fanout/contender_horizon_refs.json` md5 afd5af6b68b8947fe6c4f12a827fd916; `2026-07-10_h2h_mstar/MSTAR_VERDICT.json` md5 4f115ad55d5301122f387df504efa35c | Fig 1 |
| C4 | Uncapped transformer 0.02710/0.02930/0.02856 (below bar) ⇒ degenerate-baseline clause fires; capped M∈{1,2,4,8,16,32} all chance-level at every horizon (max single read 0.0334); per-M paired-gap CI lower bounds at H4 all >= 0.9586; verdict of record BASELINE NON-COMPETITIVE AT MATCHED PARAMS/TOKENS, never certified M*=∞, no memory-multiplier quotable | HTH §1.31.1 (degenerate-baseline clause pre-registered) + §1.41 | `2026-07-10_h2h_mstar/MSTAR_VERDICT.json` md5 4f115ad55d5301122f387df504efa35c | Fig 1 + Table 2 (appendix) |
| C5 | S0-zeroing collapses contender acc_A to 0.03394/0.00122/0.00024 (intact 0.99951/1.0/0.99902); S1-zeroing leaves 0.99951/0.99487/0.99902; 12/12 recurrent cells hard-stop clean | HTH §1.30 (localization protocol) + §1.40 (12/12 clean) | `2026-07-10_h2h_sweep_harvest/sweep_remetric/h2h_contender_task1_sweep_s{0,1,2}_round4.json` (s0_necessity_check fields), md5s as C1 | Fig 2 |
| C6 | No linear tap on raw states clears rf@0.9 (contender taps i/ii/iii: rf@0.9 = 0.0-0.003 range, all 0.0 at 0.9); pre-LM-head hidden decodes for contender only: rf@0.9 0.674, cos_mean 0.894, gap vs shuffled +0.800; ablation pre-LM-head rf@0.9 0.0 | HTH §1.30 (Tables 1-2, 2026-07-09) | `2026-07-09_h2h_tap_localization/results/tap_localization_contender.json` md5 362333c89f4223c427fe8daf54f50fce; `.../tap_localization_ablation.json` md5 ff8e352a13c2bc1e177f53f8cef47c01 | Table 3 (appendix) |
| C6b | Sweep-cohort ridge legibility at the pre-LM-head tap varies per seed (contender rf@0.9 0.686/0.771/0.951) at flat acc_A; ablation 0.0 every seed | HTH §1.40 (Leg-B diagnostic paragraph) | same contender/ablation round4 JSONs as C1 (`leg_b_ridge` fields) | prose (§5) |
| C7 | task2 (multi-hop): contender clears bar in 3/9 seeds (0.33447, 0.47949, 0.39087), ablation 0/9; pooled paired-Δ CI (−0.01999, 0.26782) marked NON-DECISION-GRADE (ablation var-ratio 6.14 > 4.0 batch-effect flag); adjudication: trainability/seed-variance confirmed, capability-boundary hypothesis rejected at this scale/budget, NOT a capability claim; seed-2 partial recall collapses to 0.010 at every extended horizon | HTH §1.42 A3 (adjudication map pinned pre-launch) + `TASK2DIAG_VERDICT.json` + §1.41 (horizon collapse) | `2026-07-10_h2h_task2diag/results/TASK2DIAG_VERDICT.json` md5 66d2291d8e65932d368d8978bfd16bdc | Table 4 |
| C8 | K48 stress (K/d=0.75, chance 0.02083, locate-only bar 0.0625): contender 0.01888, ablation 0.01953, transformer 0.02181 — all three arms at chance, no tier arithmetic | HTH §1.42 A6 (locate-only, gate-exempt, pinned pre-launch) | `TASK2DIAG_VERDICT.json` md5 66d2291d8e65932d368d8978bfd16bdc (k48 3-arm table); `2026-07-10_h2h_task2diag/results/transformer_task1_stress_K48_round4.json` md5 14e0c93f56c2a55983f929b7313eb5ac | §6 prose/table row |
| C9 | Param match: contender 14,049,408; ablation 14,048,384 (−0.007%); transformer 14,440,448 (+2.78%); all arms 20,000 steps, lr 3e-4 (task1/task2), K=32, identical episode seeds across arms per (task, seed) | HTH §1.40 (config-match verified from raws) | `2026-07-10_h2h_sweep_harvest/h2h_{contender,ablation,transformer}_task1_sweep_s0.json`, md5s 15c817f203bd7243165f953d3a6600a5 / 159687a5fbcfba334fb622c30704d553 / 427de1aaf3330f6f6d407fb12a49652a | setup table (§2) |
| C10 | State accounting: contender/ablation state constant in context length (contender 2 layers x 64x64 fp32 = 32,768 bytes; ablation 2 x 64 fp32 = 512 bytes, reported); transformer KV cache grows O(T), capped runs pin total KV bytes to M x 32,768 | HTH §1.2/§1.3 accounting table (~lines 423-465) + §1.41 (cap machinery) | derived arithmetic from pinned config (d_state=64, n_layers=2, fp32) recorded in the § record; cap wiring evidenced in `MSTAR_VERDICT.json` (cap keys) md5 4f115ad55d5301122f387df504efa35c | setup table (§2) |

No other numerical claims are planned. Any number not in this table does not
enter the draft.

## Figures to generate (single script `figures/figure-gen.py`, md5-asserted sources)

- `fig1_horizon.pdf` — MONEY FIGURE. acc_A vs context length (454/902/1798
  tokens, log-x): contender 3 seeds flat at ~1.0 with fixed 32,768-byte state;
  uncapped transformer and all capped M lines at chance; chance and
  demonstration-bar reference lines. Sources: `contender_horizon_refs.json`,
  `MSTAR_VERDICT.json`. Takeaway: recall does not decay with context and no
  KV budget rescues the baseline.
- `fig2_szero.pdf` — S0/S1 causal zeroing, grouped bars per seed (intact /
  S0-zeroed / S1-zeroed), chance line. Source: the 3 contender task1 round4
  JSONs. Takeaway: the bindings live in block 0's fast-weight state.

(Tables 1, 2, 3, 4 are typeset from the same raws via a table-block emitted by
the same script to `figures/tables_generated.tex`, so no hand-typed numbers.)

## Nearest prior work (distinguish by name)

- **Nichani, Lee & Bietti (2025), arXiv:2412.06538:** theory of associative
  memory capacity under argmax decoding; we import their caveat (a rank-1
  state supports ~d associations under argmax) as a limitation on what our
  accuracy metric can claim; our contribution is an empirical matched-budget
  separation plus a causal localization, not a capacity bound.
- **DeltaNet / parallel delta-rule (Schlag et al. 2021; Yang et al. 2024):**
  supply the contender's recurrence; neither reports a param+token-matched
  transformer/flat-vector comparison on episode-restricted recall with causal
  state-zeroing and a byte-pinned KV-cap protocol.
- **MQAR / zoology (Arora et al. 2023):** established multi-query associative
  recall as the diagnostic separating attention from recurrent models — with
  the opposite headline (attention wins MQAR; recurrent models struggle). Our
  setting differs in the queried regime: matched small budget (14M params,
  20k steps), higher load per state (K/d=0.5), and an argmax
  episode-restricted metric; we also report where our separation dies
  (K/d=0.75).
- **StreamingLLM sink+FIFO (Xiao et al. 2023):** we reuse the sink+FIFO
  eviction as the cap mechanism, but as an evaluation instrument for a
  byte-matched comparison, not as a serving method.
- **KV-cache compression (e.g. H2O, Zhang et al. 2023):** compress an
  existing competent cache; our question is whether a fixed-size fast-weight
  state can hold what a matched transformer never learns to hold at all.

## Anonymization surface (double-blind)

Tokens the grep must flag: `Larson`, `samlarson`, `saml212`, `pebble`,
`pebbleml`, `learned-representations`, `Rockie`, `idastone`, `Brev`,
`youthful-indigo-turkey`, `github.com/`, `huggingface.co/`, `acknowledg`,
`self-funded`, `funded by`. Repo paths in the submission PDF must be
genericized (`experiment-runs/...` relative paths only, no user names).

## Project-specific DO-NOT list (per styleguide, substituted for this project)

- Do NOT quote any memory-multiplier ("M x better", "M*=inf", "225x") — the
  degenerate-baseline clause fired; the only sanctioned comparative claim is
  "baseline non-competitive at matched params/tokens" with the
  matched-budget caveat. Raw byte accounting (32,768 bytes; caps at
  M x 32,768) is stated without performance-per-byte ratios.
- Do NOT call recall "exact", "continuous", or make any rank/capacity claim;
  every accuracy number carries the argmax/episode-restricted framing and the
  Nichani caveat travels with the metric definition.
- Do NOT present task2 as a capability result in either direction; it is a
  trainability/seed-variance finding, disclosed with the non-decision-grade
  pooled CI.
- Do NOT claim the separation extends beyond the tested scale (14M-class,
  20k steps), beyond K/d=0.5 (the K48 chance row is disclosed), or to
  differently-trained transformers (matched-budget caveat).
- Do NOT mention GPU-hours, cluster names, or costs.

## Dual output

- [x] Venue submission (anonymized LaTeX on the official COLM 2026 kit)
- [ ] Public write-up — not in this charter; the evidence map stays reusable.
