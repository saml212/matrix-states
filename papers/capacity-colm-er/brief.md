# Paper brief — Capacity trilogy, COLM 2026 Efficient Reasoning workshop cut

Stage 1 artifact (paper skill, repo mode). Base content: the drafted trilogy
paper at `matrix-thinking/submissions/workshop-2026/` (read-only Source),
updated with results that postdate that draft (the diffuse dose arm, the d80
refit, the d96 unlocked grid) and reframed for the venue's
memory-efficiency remit.

## Venue

- **Name:** 2nd Workshop on Efficient Reasoning @ COLM 2026
- **Format:** 4–10 pages of main text, references excluded; single PDF;
  official COLM 2026 kit (`colm2026_conference.sty`, release tag `2026`) —
  from `papers/capacity-colm-er/venue-requirements.md`, live-fetched
  2026-07-10, not a remembered number
- **Requirements source:** `papers/capacity-colm-er/venue-requirements.md`
  (evidence URLs + fetch date; no cache fallback was needed)
- **Review style:** double-blind → anonymization grep required
- **Archival:** non-archival (no official proceedings); dual-submission
  friendly — flagship-safe
- **Deadline:** July 19, 2026 AoE

## Thesis (one falsifiable sentence)

> The safe associative load of a trained delta-rule fast-weight state is not
> set by a universal capacity ratio K/d_state: the measured load frontier
> x0(d) grows super-linearly with state dimension (0.5455 at d=64 to 0.6779
> at d=80, CI-excluding the pre-registered ratio-invariance band, with no
> transition anywhere through K/d=0.9375 at d=96), and the leading
> single-variable account of that growth — anchor-table coherence — fails a
> direct frozen-dose falsification test under both concentrated and diffuse
> injection structures (19/19 cells at ceiling).

Falsifiers, by construction: a d=80 fit landing inside [0.4745, 0.6165]
would have confirmed ratio invariance (it did not); any dosed cell falling
off ceiling would have implicated coherence (none did).

## Venue angle (efficiency framing)

A fixed-size d×d fast-weight state is the constant-memory alternative to a
KV cache that grows with context. The deployment question the venue's own
topic list poses ("KV-cache compression and memory management for
long-context reasoning") is: how many bindings does a fixed state actually
hold before recall collapses, and how does that capacity move with the
state's byte size? The paper answers with a measured provisioning law: the
safe load FRACTION of the state grows with dimension, so a capacity rule
tuned at small d (e.g. "keep K/d below 0.55") systematically
under-provisions larger states. Byte honesty is mandatory: state bytes grow
as d^2 while measured capacity K* grows slower than d^2, so bindings per
byte still fall with scale; the super-linear claim is per DIMENSION
(x0 = K*/d rises), never per byte. Both sides of this arithmetic are stated
in the paper.

## Contribution bullets

1. **The capacity transition, located.** At d_state=64 the held-out recall
   frontier is a sharp transition at x0 = K/d = 0.5455 (95% bootstrap CI
   [0.5385, 0.5513], width 0.0127), a ~20x tightening over the prior
   0.25-wide bracket. (Nearest prior work reports rank/length bounds, not a
   located trained-capacity transition.)
2. **The ratio is not the law.** The identical K/d window is flat at
   ceiling at d=128 (12/12 cells h4=1.0); the frontier at d=80 sits at
   x0=0.6779 (CI [0.6683, 0.6867]), excluding the pre-registered
   ratio-invariance band entirely; d=96 shows no transition anywhere through
   K/d=0.9375. Load tolerance per dimension GROWS with d.
3. **The leading confound, killed twice.** Directly injected, frozen,
   bit-constant anchor-table coherence up to dose 0.40 — above d=64's own
   realized training band [0.373, 0.385] — leaves recall at exact ceiling in
   19/19 cells under BOTH a concentrated (rank-4) and a diffuse (rank-48)
   injection structure. (The base draft had only the rank-4 arm; the diffuse
   co-primary arm has since run and is folded in.)
4. **A disclosed measurement methodology.** Pre-registered outcome semantics
   (including degenerate-fit disclosure rules), instrument-saturation checks
   at every wave, and an admission-gate recalibration audit that unlocked the
   d=96 grid at zero new GPU spend.

## Per-section page budget (venue window 4–10pp; target 8.0pp main text)

| Section | Pages | Purpose |
|---|---|---|
| Introduction | 1.25 | constant-memory states vs growing caches; the provisioning question; contributions |
| Testbed and measurement process | 1.25 | DeltaNet state, task, exact-continuous rec@0.9 readout, load parameter K, pre-registration discipline |
| The cliff, located and dissolved | 2.00 | Act 1 (d=64 transition + fit) and Act 2 (d=128 flat), fig 1 |
| The coherence confound, exonerated | 1.25 | Act 3 dose-response, both structures, fig 2 |
| The capacity frontier grows super-linearly | 1.50 | d=80 refit vs invariance band, d=96 unlocked grid, rival accounts, byte arithmetic, fig 3 |
| Related work | 0.50 | distinguish by name |
| Limitations | 1.00 | scope, instrument-limited near-ceiling fine structure, single family |
| Conclusion | 0.25 | one paragraph |
| **Total** | **8.00** | inside the 4–10pp window |

## Claims-to-evidence-to-figure map

Rows carry all three fields: the pre-registered verdict record, the raw
artifact (path + md5), and the figure/table. Directory-level rows give the
md5 of the sorted concatenation of the named glob (the figure script
asserts per-file md5s from its own manifest).

| Claim id | Claim (with the number) | Verdict record | Raw artifact (path + md5) | Figure / table |
|---|---|---|---|---|
| C1 | d=64 curve: h4 = 1.000 (K16) / 0.6669 (K32) / 0.5676 (K34) / 0.3316 (K38) / 0.1177 (K42) / 0.0434 (K46) / 0.0215 (K48) | `KEY_ANCHORING_DESIGN.md` §12.9; archive README `experiment-runs/2026-07-06_keyanchor_cliff/` | `experiment-runs/2026-07-06_keyanchor_cliff/fit_cliff_curve_results.json` md5:c4e233fe3c7c6ec246c6e92e35134258 (curve_points; 12 underlying new-cell JSONs at `.../wavekeyanchor-cliff/*.json`, concat-md5 19ca8ed61c065710c627debb1ef3eb82) | fig1 left |
| C2 | d=64 logistic fit: x0=0.5455, 95% CI [0.5385, 0.5513] (width 0.0127), w=0.0597 CI [0.0557, 0.0642], 0/4000 degenerate bootstrap fits | `KEY_ANCHORING_DESIGN.md` §12.4 (pre-registered fit + CI bar) / §12.9 (verdict) | same fit JSON as C1 (sigmoid_fit, bootstrap_ci) | fig1 left; fig3 |
| C3 | d=128, identical K/d window (K=68/76/84/92, 3 seeds): h4=1.0 in 12/12 cells; sigmoid fit degenerate by construction (bootstrap degenerate_frac=1.0, CI null) | `KEY_ANCHORING_DESIGN.md` §13.0 (outcome semantics incl. degenerate-disclosure rule) / §13.10 (verdict: CONFIRM-SHIFTED strong form) | `experiment-runs/2026-07-06_keyanchor_dstate/fit_cliff_curve_d128_results.json` md5:0801d804c89e739da9712e9beb24b50f; 12 cell JSONs `.../wavekeyanchor-dstate/*.json` concat-md5 53bc17fb892489a6b820f654c21607e2 | fig1 right; fig3 (lower bound) |
| C4 | d=128 instrument-saturation check: hop-21 recovery spans 0.9900–1.0 across the 12 cells (not uniformly 1.0); final-checkpoint effective rank tracks K (K=84 seed-mean ≈ 83.7) | `KEY_ANCHORING_DESIGN.md` §13.10 | the 12 cell JSONs of C3 (checkpoints[-1].M3_held_out['21'].recovered_frac@0.9; ['4'].effective_rank_whole_mean) | prose (§ results) |
| C5 | Coherence framing: Welch bound forces max-abs-cos > 0 at n=107 > d=64 (construction value 0.2842, matched by the 0.284 dose arm; realized d=64 training band of final-checkpoint K-means [0.3731, 0.3845]); d=128 table exactly orthogonal at construction | `KEY_ANCHORING_DESIGN.md` §14.0b | d=64 band recomputed this session from the 12 cliff cell JSONs (C1 row, item6_table_conditioning.max_abs_cos, K-means 0.3845/0.3790/0.3789/0.3731); construction/dose value present in dose cell JSONs (C6 row) | fig2 (grey band) |
| C6 | Dose response, rank-4 (concentrated): h4=1.0 at every dose (0.130/0.284/0.40), 10/10 cells | `KEY_ANCHORING_DESIGN.md` §14.0 (pre-registered outcome 4: EXONERATE) / §14.12 (verdict) | 10 cell JSONs `experiment-runs/2026-07-06_keyanchor_dose/results/wavekeyanchor-dose/*rank4*.json` concat-md5 01508ba63b57e5a287407fc1d0535616 | fig2 |
| C7 | Dose response, diffuse (subspace rank 48): h4=1.0 at every dose, 9/9 cells — completes the co-primary EXONERATE (both structures) | `KEY_ANCHORING_DESIGN.md` §14 Rev 14.3 (co-primary registration); dose archive README Stage-2 addendum (harvest 2026-07-07) | 9 cell JSONs `.../wavekeyanchor-dose/*diffuse*.json` concat-md5 6050170519279703c179189d5e6e2a9a | fig2 |
| C8 | Frozen-dose integrity: injected coherence bit-identical across all 10 recorded checkpoints per dosed cell (e.g. dose-0.284 cell: single distinct max_abs_cos value 0.28402 across checkpoints); anchor gradient path severed pre-launch | `KEY_ANCHORING_DESIGN.md` §14.12 (frozen-constancy evidence); Rev 14.3 pre-launch gate | dose cell JSONs of C6/C7 (item6_table_conditioning.max_abs_cos per checkpoint; exactness_config.anchor_table_frozen=true) | prose (§ dose) |
| C9 | Dose-wave instrument check: hop-21 recovery 0.9987–1.0 across dosed cells (not uniformly 1.0) | `KEY_ANCHORING_DESIGN.md` §14.12 | dose cell JSONs of C6/C7 (M3_held_out['21']) | prose (§ dose) |
| C10 | d=80 refit (n=5 seeds at contested K's): x0=0.6779, 95% CI [0.6683, 0.6867], w=0.0479, 0/4000 degenerate; CI excludes the pre-registered ratio-invariance band [0.4745, 0.6165] entirely (high side) | band registered: `KEY_ANCHORING_SCALING_DRAFT.md` §15.10 (line ~621, Rev pre-run); verdict: §15.22 ("d80-escalation leg: REFUTE stands, tightened CI") | `experiment-runs/2026-07-07_keyanchor_scaling_wide/fits/fit_cliff_curve_d80_refit_results.json` md5:05dd2f9e79747f871756201165fdd548; per-seed rows in `.../fits/per_cell_verification_table.txt` md5:ae0209d9bafed7675964d8d1a555abad | fig3 |
| C11 | d=96 unlocked full grid (K=69/72/78/84/90, n=3): h4 K-means 0.9592/0.9216/0.9326/0.9581/1.0000 — no monotonic collapse anywhere through K/d=0.9375; sigmoid fit 100% bootstrap-degenerate (4000/4000), CI null; per-seed scatter real (K=72 spans 0.8426–0.9904) | `KEY_ANCHORING_SCALING_DRAFT.md` §15.25.4 (unlock) + §15.26.1 (frozen per-seed table) | `experiment-runs/2026-07-08_c17_repro/fits/fit_d96_unlocked_results.json` md5:61eaffe1744a56086af2f4115f9a9cf4; per-seed values also in `experiment-runs/2026-07-08_d96_scatter_resolution_design/fits/sim_d96_scatter_resolution_power_results.json` md5:fbd8342d266989ab9112b14997d69196 | fig3; table 1 |
| C12 | Admission-gate recalibration: 11 of 12 new d=96 cells initially flagged inadmissible; an independently audited repro instrument diagnosed a convergence-gate tolerance miscalibration — all 4,608 flagged episodes resolve by n_iter ≤ 28 (production gate 20); only the admission flag was recalibrated (n_flipped=11), never the h4 measurement; zero new GPU spend | `KEY_ANCHORING_SCALING_DRAFT.md` §15.23 (MISDIAGNOSED-ARTIFACT) + §15.25.4 (unlock record) | `experiment-runs/2026-07-08_c17_repro/fits/recalibrated_admissibility_table.json` md5:0877b840cc928dae04dff1929c2a9692 (n_flipped=11, per-cell pre/post legs); `experiment-runs/2026-07-08_c17_repro/results/keyanchor_scaling_c17repro/diag_c17_repro_analysis_K84_s1940.json` md5:6f24447c9c063e19029b1bd3dd142266 (n_episodes=4608) | prose (§ frontier + limitations) |
| C13 | Rival accounts both live at d=96: absolute-slack band [0.718, 0.739] and power-law band [0.768, 0.837] for x0(96); the degenerate fit excludes neither | bands registered: `KEY_ANCHORING_SCALING_DRAFT.md` §15.20.4; discrimination-honesty disclosure §15.26 (MAJOR-3 fix) | `experiment-runs/2026-07-08_d96_scatter_resolution_design/fits/sim_d96_scatter_resolution_power_results.json` md5:fbd8342d266989ab9112b14997d69196 (rival_bands field) | fig3 (bands) |
| C14 | Near-ceiling fine structure at d=96 is instrument-limited: a registered two-cell fresh-seed diagnostic routed to its pre-registered DEGENERATE_CELL verdict (neither cell admissible at the recalibrated gate; the failing admission leg differed from the frozen grid's), so the no-cliff claim is scoped as absence-of-monotonic-collapse, never as a settled value at K=90 | `KEY_ANCHORING_SCALING_DRAFT.md` §15.27 (harvest record, 2026-07-08) | **No repo-archived raw** (cells box-side only). CONSEQUENCE, enforced in drafting: the paper carries this ONLY as a qualitative scoping disclosure of the registered verdict; NONE of §15.27's descriptive numbers (fresh-seed h4, shift magnitudes, salvage values) appear in the paper | prose (limitations only) |
| C15 | Derived byte arithmetic (definitional, computed from C2/C10/C11 rows): a d x d fp32 state is 4d^2 bytes — 16 KB (d=64), 25 KB (d=80), 36 KB (d=96), 64 KB (d=128); located midpoint loads x0*d: ~34.9 bindings (d=64), ~54.2 (d=80), and >=90-at-ceiling observed (d=96, K=90 within its measured window); bindings per dimension rise while bindings per byte fall | derivation shown in-paper (arithmetic on C2/C10/C11 numbers; no new measurement) | C2/C10/C11 raw artifacts | table 1 |

Rule applied: every number in the draft carries an
`<!-- evidence: Cn -->` comment; a number with no row does not enter the
paper. C14's box-side numbers are excluded by construction.

## Figures to generate

Single versioned script `figures/figure-gen.py` (md5-asserting manifest of
every raw it loads; regenerates all three):

- `fig1_cliff.pdf` — two panels. Left: d=64 h4 vs K/d (7 measured loads) with
  the logistic fit and bootstrap CI band on x0; right: d=128 at the identical
  K/d window, flat at 1.0 (12/12), fit degenerate-by-construction and
  therefore drawn as points only. Caption states task, readout, and takeaway.
- `fig2_dose.pdf` — coherence dose-response at d=128/K=68: h4 vs injected
  frozen dose (0.130/0.284/0.40) for BOTH structures (rank-4 concentrated,
  rank-48 diffuse), per-seed points at ceiling; grey band = d=64's realized
  training coherence band [0.3731, 0.3845]; d=64's own cliff collapse plotted
  for reference against its matched coherence range.
- `fig3_frontier.pdf` — the located frontier x0(d): CI'd points at d=64 and
  d=80, lower-bound arrows at d=96 (>0.9375, no transition in window) and
  d=128 (>0.71875), the pre-registered ratio-invariance band, and the two
  registered rival bands for x0(96). The paper's headline figure.

## Nearest prior work (distinguish by name)

- **Schlag, Irie & Schmidhuber (2021), linear transformers as fast weight
  programmers:** supplies the delta-rule update this testbed trains; it does
  not locate a trained capacity transition or measure how the load frontier
  moves with state dimension.
- **Nichani, Lee & Bietti (2025), associative memory capacity:** capacity
  under argmax/nearest-neighbor decoding, where a rank-1 state recovers ~d
  associations; this paper's readout is exact-continuous (rec@0.9 cosine,
  no codebook), the regime where rank is load-bearing — different claim
  object, complementary result.
- **KV-cache compression line (venue topic):** measures quality retained as
  a GROWING cache is compressed; this paper measures capacity of a FIXED
  state as load grows — the constant-memory endpoint of that trade-off.
- **DeltaNet-family scaling work (DeltaProduct; gated variants):** improves
  the architecture's expressivity/hardware fit at scale; no claims here
  transfer (sub-1M synthetic testbed, disclosed), and none of those works
  measures a located load frontier.

(Exact BibTeX resolved and verified programmatically at the citation stage;
refs reused from the base draft are re-verified, not trusted.)

## Anonymization surface (double-blind)

Grep tokens (case-insensitive) for the bundle: author surname + given name,
`saml212`, `samlarson16`, `pebble`, `pebbleml`, `learned-representations`,
`idastone`, `Brev`, `youthful-indigo-turkey`, `github.com/`,
`huggingface.co/`, `acknowledg`, `self-funded`, `funded by`. Expected
matches: zero. Also banned in prose: GPU-hour/dollar cost figures, hardware
bragging, the word "audit" is NOT banned for this paper (the admission-gate
recalibration is part of the story) but funding/identity language is.

## Dual output

- [x] Venue submission (anonymized LaTeX on the official COLM 2026 kit,
  compiled locally with tectonic per the base draft's own build path)
- [ ] Public write-up — not in scope for this run (deadline-critical)
