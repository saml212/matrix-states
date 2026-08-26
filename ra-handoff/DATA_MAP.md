# DATA_MAP — where every artifact lives

Companion to `README.md` (the human brief) and `AGENT_INSTRUCTIONS.md` (the
agent brief). This tells you where to find the raw number behind any claim, and
what has to come off the GPU box before the grant expires.

**Updated 2026-08-25.**

---

## 1. `experiment-runs/` in the repo — the tracked archive

**Path:** `/Users/samuellarson/Experiments/learned-representations/experiment-runs/`
**Size:** ~1.3 GB, 139 entries, 3,651 files tracked in git.
**Naming:** `YYYY-MM-DD_<campaign>` — earliest `2026-04-10_matrix_codi_round1`,
latest `2026-08-23_v2prime`.

**The policy** (source of truth: `experiment-runs/README.md`, dated 2026-07-04 —
this supersedes a briefly-tried symlink approach you may see referenced in old
docs):

> Every experiment's exact scripts + result JSONs live here, size-capped: files
> ≤25MB are tracked in git (crash-proof via GitHub); larger payloads (Z-dumps,
> checkpoints, >25MB JSONs) live ONLY in the full archive at
> `/Volumes/1TB_SSD/learned-representations/experiment-runs/` (superset of this
> directory). Write new archives to BOTH: small files here (commit them), big
> files to the SSD path. If the SSD is unmounted, stop and say so.

The cap is being honored — the largest tracked file is ~16.7 MB.

**The archives the two new flagship chapters depend on** (all present in the
repo, all committed):

| Archive | Feeds |
|---|---|
| `2026-08-21_residue_depth_evals/` | depth robustness, coverage |
| `2026-08-21_poolmatched_battery/` | the matched-pool re-adjudication |
| `2026-08-21_deepladder_matched/` | the deep-ladder ordering measurement |
| `2026-08-22_kscaling_wave0/`, `_sweep/`, `_frontier/` | breadth chapter (§6) |
| `2026-08-22_depthext_across_k/` | across-K depth extension |
| `2026-08-22_scaleaxis_a0/`, `_sweep/`, `_stagec/` | parameter chapter (§7) |
| `2026-08-23_attribution_arm/` | §7.5 attribution (the withdrawal) |
| `2026-08-23_v2prime/` | §7.5 constant-LR control |

Older archives feeding flagship §§3–5 are named with md5s in the evidence rows
in `papers/flagship/brief.md` (rows R1–R12).

---

## 2. The SSD superset

**Path:** `/Volumes/1TB_SSD/learned-representations/`
**Status:** mounted and readable as of 2026-08-24.

| Directory | Size |
|---|---|
| `experiment-runs/` | 4.8 GB, 131 entries |
| `data/` | 2.3 GB (`text.bin` WikiText-103 bytes, `images.bin` CIFAR-10 pixels) |
| `checkpoints/` | 219 MB |
| `results/` | 144 MB |
| `models/` | empty |

Where the same archive exists in both places, the SSD copy is much larger —
that's the >25MB payload half. Examples: `2026-04-29_rank_aware_v1/` is 1.0 GB
on SSD vs 8.2 MB in the repo; `2026-04-10_matrix_codi_round1/` is 956 MB vs
5.1 MB.

**⚠ The "superset" claim has drifted and is no longer strictly true.** Three
directories exist only on the SSD (`2026-04-13_results`,
`2026-04-13_round7_illusion`, `2026-07-10_ncr_negative_tests`) — expected. But
**eleven exist only in the repo and were never mirrored to the SSD**:

```
2026-07-04_track_a                    2026-07-17_ncr_gate3_wave1
2026-07-08_d96_scatter_resolution_design   2026-07-17_ncr_ortho_fallback_stage0
2026-07-11_ncr_opbank_seedrep         param_axis_r0
2026-07-12_ncr_q3_mechanism           param_axis_t2a_attempt3 / 4 / 5
_session_transcripts
```

The write-to-BOTH half of the policy slipped. Nothing is lost (git has them),
but do not assume "it's on the SSD" for anything in that list.

---

## 3. The findings pages

**Path:** `/Users/samuellarson/Experiments/learned-representations/pebble-ai-site/findings/`
20 hand-authored standalone HTML pages, ~1.9 MB total, **no build step**.

The ones that matter for the two new chapters:

- `ncr-scale-axis.html` (100 KB, 2026-08-23) — the 98M→392M parameter axis
- `ncr-breadth-scaling.html` — the K=12→40 breadth axis
- `ncr-depth-robustness.html` — depth/coverage/drift
- `native-composition-reads.html` — the NCR result itself
- `write-geometry-attractor.html`, `rank-law.html`, `constant-memory-recall.html`,
  `fast-weight-recall.html`, `broken-lens.html`, and 11 others

**Why these govern.** Each page's numbers were recomputed independently from the
per-cell JSONs at publication time, not copied from log prose — and that
recomputation is what caught several transcription errors. The pages say so
explicitly. From `ncr-scale-axis.html`:

> Every number on this page and every point in fig 1 is recomputed from those
> per-cell JSONs — `matched.P1b` and `matched.P0`, field `acc`, per hop, with κ
> derived from K — never from a harvest summary and never from prose.

Each page names its raw archives and its regenerating plot script (under
`pebble-ai-site/assets/plots/`), and those scripts assert cell count, checkpoint
step, and the `d = K+1` mapping before plotting.

**When a page and the log disagree, the page wins.** The correction entries name
each case — see `AGENT_INSTRUCTIONS.md` §2.3.

**⚠ These pages are NOT live on the web.** The site sync GitHub Action
(`.github/workflows/sync-pebble-site.yml`, mirrors `pebble-ai-site/` to
`saml212/pebble-ai-site` → GitHub Pages at pebbleml.com) has been **dead since
2026-07-14**, and the pebbleml.com domain was **repurposed on 2026-08-19 to a
different property**. Findings 18, 19, and 20 exist in-repo only. Earlier
"published live" reports in the log were coordinator error and were corrected
(`EXPERIMENT_LOG.md` 2026-08-22 #23, commit `3412c00`). Treat the findings pages
as internal documents of record, not as public prior art — and if you want them
public, that's a decision for Sam about where they'd live now.

---

## 4. The box (ephemeral — dies ~2026-08-31)

**Instance:** `youthful-indigo-turkey` (nvidia-pebble), hostname
`brev-ukptqsu65`, 8×H100 80GB SXM, GCP asia-southeast1-c.
**Access:** `ssh youthful-indigo-turkey` — the alias comes from
`~/.brev/ssh_config`, Included from `~/.ssh/config`, key `~/.brev/brev.pem`, via
a cloudflared tunnel. If it breaks: `brev login` (interactive — Sam has to run
it) then `brev refresh`. Default user `nvidia`, home `/home/nvidia`,
passwordless sudo.
**Do not stop or delete the instance.** The grant is uptime-metered surplus
credits that expire; there is no saving to be had by idling it.
**Operative expiry: ~2026-08-31.** As of 2026-08-25 the PI recorded **six days of
grant remaining, ~1,100 GPU-h**
(`matrix-thinking/scaleaxis_build/job_specs_1b/EXPERIMENT_LOG.md` 2026-08-25 #1).
Older docs say "~Sep 1"; treat **Aug 31** as the date you plan against.

| Path | What |
|---|---|
| `/home/nvidia/tdenv` | the venv (torch 2.12.1+cu130, CUDA, 8 devices, sm_90) |
| `/home/nvidia/chapter2/` | code scp'd from `matrix-thinking/chapter2/`, plus `results/<experiment>/` JSONs and logs |
| `/home/nvidia/queue/` | the job queue: `pending/`, `completed/`, `failed/`, 8 worker daemons in tmux `queue_worker_g0`…`g7`, logs `worker_gN.log`, control flags `PAUSE` / `STOP` |
| `/ephemeral/` | 5.9 TB — **all training checkpoints live here** |
| `/ephemeral/reseed_ckpts/` | reseed checkpoints |
| `/data` | 18 TB volume |
| `/dev/root` | 193 GB — results JSONs only, never checkpoints |

The checkpoint-location rule exists because the root disk hit 100% on
2026-08-18; a monitor now auto-pauses the queue at ≥92% root usage.

**Note on `/root/` paths.** `CLAUDE.md` and `matrix-thinking/H100_SETUP.md` still
mention `/root/data/reasoning/` (43.7M tokens OpenR1-Math, GPT-2 tokenized) and
an `ssh root@154.57...` pod. **That is a different, superseded machine** — a
prior single-H100 cloud pod, kept in the docs for reference only. It is not the
Brev box and those paths do not exist there.

**`HANDOFF_BOX_ACCESS.md`** at the repo root has the full box runbook — queue
mechanics, idle daemons, the safety rules. It is **untracked scratch** that
self-describes as "do not commit it; delete when absorbed," so it may vanish.
Everything in it that matters long-term has been folded into this file. Read it
while it's there.

### ⚠ 4a. What must come off `/ephemeral/` BEFORE ~2026-08-31

The instance cannot be stopped, only deleted, and the grant expires. When it
goes, `/ephemeral/` (5.9 TB) goes with it. **Nothing in the repo depends on a
box path today** — every archive named in a paper is already local — but these
are the things that exist only on the box and would be unrecoverable:

1. **Trained checkpoints under `/ephemeral/`** — every 98M, 392M and 1.31B
   checkpoint from the NCR breadth/scale program. The result JSONs are archived;
   the weights are not. If anyone will ever want to re-probe a trained model
   (a new eval, a new tap, a reviewer asking "what if you measured X"), the
   checkpoints must be pulled. **This is the big one, and it is the one that is
   easy to forget because no current paper needs it.** Triage first — full
   1.31B checkpoints are large; a defensible minimum is one checkpoint per
   (scale, K, recipe) at the frontier K values.
2. **The 1.31B run's outputs**, whenever the sweep finishes (~2026-08-26). Its
   archive directory has to be written to the repo (≤25MB files) and the SSD
   (everything) the same way every prior wave was. If the sweep is still running
   when the grant ends, whatever it has produced must be harvested early rather
   than lost.
3. **`/home/nvidia/queue/` logs** — `worker_gN.log`, `watchdog.log`,
   `idle_launcher.log`. These are the execution record (what ran, when, what
   failed). Not cited by any paper, but they are the audit trail behind the
   GPU-h ledgers that *are* cited.
4. **Any on-box script that was edited in place** rather than in the repo.
   Compare `/home/nvidia/chapter2/` against `matrix-thinking/chapter2/` and pull
   any drift. The repo is supposed to be authoritative, but in-flight patches
   happen.
5. **`~/queue/completed/` job spec JSONs** — the exact spec each cell ran under,
   including the validity-check clauses. Several log entries reference these to
   adjudicate what a cell actually did.

**Sam has to do this — it needs the Brev credentials.** Ask him early; a
5.9 TB volume is not a same-afternoon transfer, and the SSD has ~1 TB free.

---

## 5. X post drafts on the Desktop

`/Users/samuellarson/Desktop/` holds the social drafts for each finding. They
are not in the repo and are not backed up.

| Modified | File |
|---|---|
| 2026-08-22 | `x_draft_scaleaxis.txt` |
| 2026-08-21 | `x_draft_kscaling.txt` |
| 2026-08-21 | `x_draft_depth_robustness.txt` |
| 2026-08-13 | `x_draft_premise_finding.txt` |
| 2026-07-12 | `xpost_ncr_kaxis_closed_2026-07-12.md` |
| 2026-07-12 | `xpost_ncr_dmapping_2026-07-12.md` |
| 2026-07-11 | `xpost_ncr_earlyln_scaling_2026-07-12.md` |

Plus four packaged bundles with figures: `x-post-constant-memory/`,
`x-post-stage2-verdict/`, `x-post-ncr-win/`, `x-post-instrument-saga/` (each has
a figure PNG, most have a `READY_TO_POST.md`), and three loose figure PNGs.

Useful to you as plain-language summaries of each result written for a general
audience — a decent starting point for an abstract or an intro paragraph. None
of them appear to have been posted.

---

## 6. `EXPERIMENT_LOG.md` — how to navigate 900 KB

**Path:** repo root. ~909 KB, ~11,960 lines. **Append-only** — stated
explicitly in the file at line 1475, and visible in how corrections work.

⚠ **There are also wave-local `EXPERIMENT_LOG.md` files** under build
directories — e.g. `matrix-thinking/scaleaxis_build/job_specs_1b/EXPERIMENT_LOG.md`,
which holds the 2026-08-25 #1 entry on the 1.31B sweep. They are real records,
but they are **not** the root log, and an entry number like "2026-08-25 #1" is
ambiguous between them. **Always cite the full path.**

Three formatting eras, in file order:

1. **Run-numbered** (lines 1–~1470): `## Run N: <title> — COMPLETED`, with
   `### Config` / `### Key Results` / `### Findings` / `### Root Cause`
   subsections. The M4-Mac-Mini era.
2. **Topic-titled** (94 entries): `## <descriptive verdict sentence> (YYYY-MM-DD)`.
   Dates 2026-03-26 → 2026-07-10.
3. **Dated-tick** (129 entries, current): `## YYYY-MM-DD #N — <VERDICT IN CAPS>`.
   Dense prose, bold verdict terms, 4-decimal figures, GPU-h ledgers, commit
   SHAs, ending with `Archive: experiment-runs/<dir>/ (repo+SSD)`.

**Four ways to find something:**

1. **Grep the headings** — `grep '^## ' EXPERIMENT_LOG.md`. Headings are
   self-describing full verdicts, so this is a usable whole-file index.
2. **Backward from a findings page** — every page ends with a "Verdicts of
   record" list citing exact tick IDs. `ncr-scale-axis.html` cites 2026-08-22
   #10–#21 and 2026-08-23 #1–#4.
3. **Forward from the log to the data** — the `Archive:` line in each modern
   entry.
4. The in-file table of contents at line 1472 covers only the 2026-07-01 →
   07-04 campaign. There is no whole-file index.

**Corrections are their own genre.** ~10 genuine correction entries; each is a
new dated tick whose title announces which earlier entry it corrects and what
survives. The dominant trigger is **independent recomputation from raw per-cell
JSONs at publication time** — which is exactly the invariant the findings pages
assert. In nearly every case the correction restates numbers while the headline
verdict survives. Ones you should read before touching the scaling chapters:

- `2026-08-22 #5` and `#9` — three coordinator transcription errors caught by
  the publisher's independent recomputation; no verdict changes.
- `2026-08-22 #23` — publication-status correction (findings 18–20 not live) plus
  three number corrections from raw recomputation.
- `2026-08-23 #5` — four numbered corrections from finding-20's amendment
  recompute. States the governing rule: **raw governs**.
- `2026-08-24 #2` — ledger correction on GPU-h pricing (fresh 392M cells cost
  3.392 GPU-h each, not the resume-marginal figure).
- `2026-08-18 #12` — two stale eval records scored against pre-incident
  checkpoints, used in three prior ticks. Headline survived; numbers restated.
- `2026-08-17 #3` — a coordinator reading that used a metric the design had
  explicitly barred, compared across incomparable rungs.

---

## 7. Other repo landmarks

| Path | What |
|---|---|
| `STATE.md` | ~120 KB, reverse-chronological session ticks. The top ~200 lines are the current state of everything. Start there for "what is happening right now" |
| `matrix-thinking/*_DESIGN.md` | The design registries. Pre-registrations, verdicts of record, gauntlet rounds. `HEAD_TO_HEAD_DEMO_DESIGN.md`, `CAPABILITY_SEPARATION_DESIGN.md`, `FROZEN_BIAS_LM_DESIGN.md`, `NCR_REAL_LM_DESIGN.md`, `SCALE_TRANSFER_DESIGN.md`. Evidence rows cite these by § |
| `research/` | Standing literature memos and novelty-gate records. `kscaling-novelty-2026-08-21.md` and `scale-axis-novelty-2026-08-22.md` are the gates for the two new chapters (3/3 legs clear each) |
| `references.md` | The paper reference library (37 KB) |
| `matrix-thinking/H100_SETUP.md` | Box environment and commands. Note: contains a redacted HF token line — see the security note in `EMAIL_DRAFT.md` |
| `archive/` | Dead ends and superseded docs. Do not revive without asking |
| `REPO_READABILITY_AUDIT.md` | An adversarial cold-read audit. Flags CLAUDE.md's repo-layout map as partly stale (4/7 paths wrong) and the undefined-jargon problem. Useful context for why the glossary in `README.md` exists |

---

## 8. Asking for more

Ask Sam. Specifically:

- **More data / another run:** possible until ~2026-08-31 on the grant, and a
  conversation about rented hardware after that. Bring a hypothesis and a rough
  GPU-h estimate — the program prices every wave before launching it, and that
  is the language the queue speaks.
- **Repo access:** the repo is private, and there is a credential in its git
  history that must be rotated first. See `EMAIL_DRAFT.md`.
- **Box access:** needs Brev credentials, which are Sam's.
- **Anything on `/ephemeral/`:** ask now, not in September. See §4a.
