# 05b — Format Re-Audit (scoped, round 1)

**Re-auditor:** fresh-context format re-auditor. No memory of prior stages.
Independently verified each finding from `05_format_audit.md` (1 CRITICAL / 2
SERIOUS / 6 MINOR) against the current draft and the raw artifacts. Raw JSON
read read-only with `python3` (no numpy). The raw file is the tiebreak, never
the prose or the prior audit's assertions.

## Verdict line

- **C1 (CRITICAL, contender param count) — CLOSED.**
- **S1 (SERIOUS, missing bib entries + prose-year agreement) — CLOSED.**
- **M1 (MINOR, abstract word band) — CLOSED.**
- **M4 (MINOR, derived-number evidence tags in §5) — CLOSED.**
- **Regression sweep (banned words / contractions / broken refs in changed
  files) — CLEAN.**

**No CRITICAL format finding remains open.**

---

## C1 — Contender parameter count — CLOSED

Independent raw reading (`n_params`), all three R4 contender task-1 training
JSONs in `experiment-runs/2026-07-10_h2h_sweep_harvest/`:

| file | n_params |
|---|---|
| `h2h_contender_task1_sweep_s0.json` | **14,049,408** |
| `h2h_contender_task1_sweep_s1.json` | **14,049,408** |
| `h2h_contender_task1_sweep_s2.json` | **14,049,408** |

The draft now states the contender at **14,049,408** in all three loci:
- `04_capability_separation.md:11` — "14,049,408 parameters `<!-- evidence: R0 -->`"
- `10_appendix_b_reproducibility.md:8` — "contender ... 14,049,408 params"
- `brief.md:178` (R0 row) — "14,049,408 params (CORRECTED 2026-07-10 per
  format-audit C1 ...)"

Derived percent difference vs the ablation (14,048,384): gap = 1,024;
1,024 / 14,049,408 = 0.00729%, stated as **"0.007 percent"** at
`04_capability_separation.md:15` and "0.007%" in brief R0. Correct. The stale
**"0.004 percent"** figure appears **nowhere** in `sections/` or `brief.md`.

Section-5 ladder rung correctly **keeps** 14,048,896: the only surviving
14,048,896 instances are (a) `10_appendix_b_reproducibility.md:9` "Ladder rungs
(Section 5) ... measured 14,048,896 / 97,618,176 / ..." and (b) brief R0's
ladder-rung list plus its explanatory "the earlier 14,048,896 was the Section-5
14M ladder rung's count." Both are ladder-rung references, not the contender. A
full-text grep confirms **no stale 14,048,896-as-contender instance survives**
anywhere in `sections/` or brief R0.

## S1 — Missing bib entries + citation/year agreement — CLOSED

`references.bib` now contains all six flagged entries plus the four earlier
ones. Brace balance = 104/104; all 13 `@misc` keys unique.

| citation | key | eprint | year |
|---|---|---|---|
| Nichani | `nichani2024understandingfactualrecalltransformers` | 2412.06538 | 2024 |
| rank paper (Nazari/Rusch) | `nazari2026keystatereductionlinear` | 2602.04852 | 2026 |
| rank paper (Sun et al.) | `sun2026staterankdynamicslinear` | 2602.02195 | 2026 |
| Kimi Linear | `kimiteam2025kimilinearexpressiveefficient` | 2510.26692 | 2025 |
| DeltaNet | `yang2025parallelizinglineartransformersdelta` | 2406.06484 | 2025 |
| Gated DeltaNet | `yang2025gateddeltanetworksimproving` | 2412.06464 | 2025 |
| Zoology | `arora2023zoologymeasuringimprovingrecall` | 2312.04927 | 2023 |
| Jelassi | `jelassi2024repeatmetransformersbetter` | 2402.01032 | 2024 |
| Olsson | `olsson2022incontextlearninginductionheads` | 2209.11895 | 2022 |
| Xiao | `xiao2024efficientstreaminglanguagemodels` | 2309.17453 | 2024 |

Every author-year citation in the section prose resolves to a bib entry **or**
is named in the bib header's explicit deferral list. The two remaining
uncitable-in-bib references — **Qwen3-Next** (`06_related_work.md:37`) and the
program's own **ICML 2026 MI-workshop companion** "The Gradient Does Not See
Rank" (`06_related_work.md:56`) — are both explicitly named in the bib header's
"STILL DEFERRED TO ASSEMBLY TIME" block (references.bib:10-14), which also
records the reason each is deferred (no arXiv paper / self-citation
anonymization). This also discharges the prior S2 finding.

Prose-year agreement: **Nichani reads 2024** in prose (`02:52` "(Nichani et
al., 2024)"; `06:3-4` "Nichani, Lee, and Bietti (2024)") matching bib year
2024. **Xiao reads 2024** (`04:115` "(Xiao et al., 2024)") matching bib year
2024. Arora 2023, Jelassi 2024, Olsson 2022 all agree with their bib years.

## M1 — Abstract word band — CLOSED

`00_abstract.md`, stripping all HTML comments (title, authors, and every
`<!-- evidence -->` tag) and the `# Abstract` heading: **226 rendered words**.
Within the 200–230 band with **4 words of headroom** below the 230 ceiling
(was exactly 230 at the ceiling in the prior audit).

## M4 — Derived-number evidence tags in §5 — CLOSED

All four derived/config numbers now carry evidence tags:
- `05:31-33` "0.003 ... 0.066" now closes with
  `<!-- evidence: R6 (derived: 0.4584-0.4554 and 0.455-0.389) -->` — the
  derivation is spelled in the tag. Arithmetic verified: 0.4584−0.4554 = 0.003;
  0.455−0.389 = 0.066; ratio 0.066/0.003 = 22, matching "more than twenty times
  smaller."
- `05:41` noise floor "2.244355" sits in the same sentence closed by
  `<!-- evidence: R7 -->`.
- `05:58` "$\lambda = 0.58$" carries `<!-- evidence: R8 -->` inline.

## Regression sweep on the changed files — CLEAN

Changed files scanned: `00_abstract.md`, `02_background_setup.md`,
`04_capability_separation.md`, `05_pathology_at_scale.md`,
`06_related_work.md`, `10_appendix_b_reproducibility.md`, `brief.md`,
`references.bib`.

- **Banned words** (whole-word, case-insensitive, all 17 styleguide terms:
  honest, actually, really, just, clearly, obviously, interestingly, nicely,
  remarkable, surprising, unfortunately, essentially, wildly, literally,
  parsimonious, cleanest, sharpest): **0 hits.**
- **Contractions**: **0.** All apostrophe-letter tokens in the changed section
  files are `'s` possessives (contender's, model's, paper's, etc.); no `n't`,
  `'re`, `'ve`, `'ll`, `'d`, `'m`, or contracted `'s` forms.
- **Broken Section/Figure/Table/Appendix references**: **0.** All cross-refs in
  the changed files resolve (Sections 2.3/2.4/3/3.3/4/4.1/4.3/4.5/5/5.2/6/7,
  Appendix B, B.3, Figures 3–7, Table 1). The single "Figure R" grep token is
  the known false match on the B.3 heading "Figure Regeneration," not a
  reference.

## Security note

I read the six changed section files, `brief.md`, `references.bib`, the prior
audit `05_format_audit.md`, and the three raw contender JSONs under
`experiment-runs/`. **No embedded or injected instructions were found** — no
fake system-reminder blocks, no date-change claims, no "file was modified — do
not tell the user" concealment strings, no directives to alter
permissions/config. All tool outputs were plain data (numeric JSON fields and
grep matches). Had any such block appeared, I would have disregarded it,
verified against git/md5, and reported it here rather than complying.
