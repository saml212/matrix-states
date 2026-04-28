# Control A Execution + Paper + Roadmap — Agent Brief

**Owner:** Sam Larson (samlarson16@gmail.com)
**Handoff date:** 2026-04-24
**Hard deadline:** 2026-05-08 AOE — ICML MI Workshop 2026 paper submission
**Time remaining:** ~14 days

You are picking this up cold. Read the entire brief before running anything. The scope is larger than just "run a script" — you run the experiment, update the paper, and then continue with the rest of the matrix-thinking roadmap through submission. Sam will provide H100 SSH access at handoff.

---

## 1. Mission

Three phases, in order:

1. **Execute Control A** — the pre-submission null-baseline experiment (fake-Z rank-k ablation on vanilla GPT-2 SFT for ProsQA). Iterate through smoke-test failures. Land a clean result.
2. **Incorporate into the paper** — update `PAPER_RESULTS_SUMMARY.md` with Control A, draft the §5.X control subsection the paper needs, adjust framing in §7 Discussion based on the Control A outcome.
3. **Continue the roadmap** — work through the remaining pre-submission items in `matrix-thinking/QUEUE.md` and `matrix-thinking/STATE.md` §"Path Forward", write the paper (`matrix-thinking/PAPER_WRITER_BRIEF.md`), and ship the ICML submission on time with dual output (anonymized LaTeX + website findings page).

Sam's broader goals: get hired by a frontier lab (Anthropic, DeepMind, OpenAI, etc.), win compute grants, ship workshop papers, build citation weight. Every decision you make should favor shipping a defensible paper over shipping extra experiments.

---

## 2. Required reading — do this before touching anything

In repo `/Users/samuellarson/Experiments/learned-representations/`:

### Orientation (understand the project)
- `CLAUDE.md` — workflow norms, hard rules. Every rule at the bottom applies to you.
- `STATE.md` — current project state. §"Path Forward (April 2026, post matrix-CODI results)" is your roadmap.
- `ARCHITECTURE.md` — if present, the full architecture spec.
- `EXPERIMENT_LOG.md` — chronological record of every experiment. Recent entries (Rounds 1–9, Round PC) orient you.

### The paper this work feeds
- `matrix-thinking/PAPER_WRITER_BRIEF.md` — what you're writing, target venue, format constraints, section plan, figure list, do-nots. You become the paper-writer agent in Phase 3.
- `matrix-thinking/PAPER_RESULTS_SUMMARY.md` — current tables, Control A adds to these.
- `matrix-thinking/ILLUSION_RECIPE.md` — the rank-blindness writeup for the website.
- `matrix-thinking/KILL_LIST.md` — past failure modes. Lesson 1 (linear lm_head is rank-blind to any slice of h) is directly cited in the Control A design.
- `matrix-thinking/submissions/icml-mi-workshop-2026/review/` — prior attack + rebuttal reports on the paper. Attack A16 (rank-1-solvable-task) is what Control A is designed to rebut. Know that context cold.
- `matrix-thinking/CHAPTER_2_DESIGN.md` and `matrix-thinking/BILINEAR_READOUT_PATCH_PLAN.md` — in-flight planning for work after this paper. You don't execute Chapter 2, but know it exists so you don't cut off its setup.

### The Control A cascade (all completed 2026-04-23)
- `matrix-thinking/QUEUE.md` §PRIORITY 0 — the original spec. Treat as requirements doc.
- `matrix-thinking/CONTROL_A_RESEARCH_REPORT.md` — research agent's briefing.
- `matrix-thinking/CONTROL_A_IMPLEMENTATION_NOTES.md` — current design (v2, propagating intervention). **Read this thoroughly.** The smoke-test commands, full-sweep commands, and scope-of-claim live here.
- `matrix-thinking/CONTROL_A_AUDIT_REPORT.md` — what the audit agent caught.
- `matrix-thinking/CONTROL_A_ATTACK_REPORT.md` — two fatal attacks that drove the v2 redesign.
- `matrix-thinking/CONTROL_A_FIX_RECEIPT.md` — per-item status of every audit + attack fix.
- `matrix-thinking/scripts/run_control_a.py` — the script you run. 1365 lines. Parses clean via `py_compile`. **Has never actually executed.** Expect iterations.

### Reference code
- `matrix-thinking/scripts/run_matrix_codi.py` — the parent script Control A imports from and mirrors. Especially `truncate_to_rank` (~line 336), `eval_rank_projection` (~line 1919), `ProsQADataset`, `prosqa_answer_match`, `generate_answer` (~line 1082).
- `matrix-thinking/H100_SETUP.md` — SSH, paths, launch conventions. May be slightly stale (see CONTROL_A_RESEARCH_REPORT for drift notes); verify with Sam if anything looks off.
- Existing scripts in `matrix-thinking/scripts/` — style reference. No emojis, short clear functions, minimal comments.

### Project memory
- `/Users/samuellarson/.claude/projects/-Users-samuellarson-Experiments-learned-representations/memory/MEMORY.md` — Sam's persistent memory. Note especially:
  - `feedback_commit_signing.md` — do NOT add `Co-Authored-By: Claude` trailers to commits
  - `feedback_subagent_model_policy.md` — for any sub-agents you spawn, Opus for high-priority (this whole workflow qualifies)
  - `feedback_research_recency.md` — literature checks should prioritize last 6-12 months for ML/AI

---

## 3. State at handoff — what the cascade produced, what's still unknown

**Completed cascade on 2026-04-23:** research → implement → audit → attack → bug-fix → final-eval. Every audit P0/P1 and every attack fatal/major/minor is resolved in the script (see fix receipt).

**Known gap in the cascade:** the attack agent ran on the *original* script. The bug-fixer rewrote ~1000 lines to fix two fatal flaws. **No attack pass was run on the v2 redesigned script.** If the first smoke runs fail in ways that feel like design errors (not just bugs), spawn an attack agent on the v2 script before iterating blindly.

**What's almost certainly correct:** spec compliance, the three audit P0 fixes (loader, assertions, 10-example smoke), the general shape of the propagating intervention.

**What's almost certainly going to need fixing on first smoke run** (budget 2–4 iterations):
- Import surface. Script imports 9 symbols from `run_matrix_codi.py` at module level. If any got renamed, moved inside `main()`, or changed signature, first run fails at import.
- Checkpoint format. The loader has fallbacks for raw state_dict, `{"model": sd}`, `module.` DDP prefix, `gpt2.` CodiModel-wrapper prefix. Round 4 format is unverified — Sam says checkpoints exist at `/workspace/pebble/round4_vanilla_sft/pure_sft_seed{1337,42,7}.pt` but their internal structure has not been inspected.
- Analog-position picking. `_compute_analog_positions` assumes `" The answer is:"` tokenizes with `:` as a distinct final token. If GPT-2 BPE splits it differently, the 6 analog positions are wrong. Smoke test logs `colon_pos` and `analog_positions` for the first example — verify by hand.
- Hook output shape. `model.transformer.h[6]` output can be `tuple` or `tensor` depending on HF transformers version; the hook handles one path.
- No-KV-cache decoding assumption: that greedy output without KV cache matches greedy output with KV cache on the unablated baseline. Should be true; verify empirically in smoke.

**Scientific limitation baked in** (acknowledged, not fixed): intervention is at block-6 output with 6 layers of downstream propagation; matrix-CODI's is effectively input-embedding-level with 12 layers. The paper framing must acknowledge this asymmetry — claim is "propagating rank-k intervention at a mid-sequence position in vanilla SFT," NOT "mechanistically identical null of matrix-CODI's intervention." A layer sweep `L ∈ {3, 6, 9}` would strengthen this; queue it if time allows but it's not a submission blocker.

---

## 4. Phase 1 — Execute Control A

### 4a. Pre-flight (on the H100)

Before any experiment:

1. SSH in. Verify paths from `H100_SETUP.md`. Update `H100_SETUP.md` if stale.
2. Verify checkpoints exist: `ls -la /workspace/pebble/round4_vanilla_sft/pure_sft_seed{1337,42,7}.pt`. If missing, STOP — you need to re-run Round 4 vanilla SFT first using the actual `pure_sft` semantics (no latent tokens). Refuse to proceed on a best-guess alternative; ask Sam.
3. Inspect one checkpoint's top-level structure: `python3 -c "import torch; sd = torch.load('/workspace/.../pure_sft_seed1337.pt', map_location='cpu'); print(type(sd)); print(list(sd.keys())[:5] if hasattr(sd, 'keys') else 'not dict')"`. This tells you which loader branch in `run_control_a.py:265-330` will fire.
4. Check environment: `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"`, `pip show transformers | grep Version`. Should match the versions used for Round 4 training (git log will show the installation commit if relevant).
5. Sync the repo to the H100 and install/activate the right venv. Reference `experiment-runs/_auto_sync/` for how prior rounds synced.

### 4b. Smoke test first (MANDATORY)

**Do not skip this even if it feels redundant.** The 10-example unablated accuracy assertion is the critical gate that catches silent checkpoint-load failures.

```
python3 matrix-thinking/scripts/run_control_a.py \
    --checkpoint-dir /workspace/pebble/round4_vanilla_sft \
    --ckpt-name-template pure_sft_seed{seed}.pt \
    --smoke-test-all-variants \
    --intervene-layer 6 \
    --output-dir experiment-runs/2026-04-24_control_a/smoke_all/
```

Expected output:
- All imports resolve
- Seed 1337 checkpoint loads with 124,439,808 parameters and no missing critical keys
- 10-example unablated decode produces ≥5 correct
- k=16 intervention fires, produces logits close to unablated
- k=1 intervention produces substantially different logits and low effective rank
- k="random" path runs without crashing
- All three variants (`16x16`, `16x48`, `svd_full_768`) complete smoke

**If smoke fails:** diagnose the failure, fix it in `run_control_a.py` (or `run_matrix_codi.py` if a symbol genuinely moved), re-run smoke. Repeat until clean. Commit after each successful fix with a short message. If you hit the same failure twice with different fixes, stop and think — you may be solving a surface symptom while the real bug is elsewhere. If you exceed 4 iterations without smoke passing, spawn an attack agent on the v2 script to find what you're missing.

**What NOT to do if smoke fails:**
- Don't "disable the assertion that's tripping" — assertions exist because they gate correctness. If an assertion is wrong, prove it and then remove it; don't just comment it out.
- Don't swap the checkpoint for an easier one — run on the actual Round 4 checkpoints the paper references or no-run.
- Don't reduce the `n_examples` for the full sweep just to finish faster — the decision rule has pre-registered power requirements.

### 4c. Primary full sweep (variant 16x16)

Once smoke passes cleanly:

```
python3 matrix-thinking/scripts/run_control_a.py \
    --checkpoint-dir /workspace/pebble/round4_vanilla_sft \
    --ckpt-name-template pure_sft_seed{seed}.pt \
    --seeds 1337 42 7 \
    --ks 1 2 4 8 16 \
    --variant 16x16 \
    --intervene-layer 6 \
    --n-examples 500 \
    --max-new-tokens 24 \
    --output-dir experiment-runs/2026-04-24_control_a/primary_16x16/
```

Expected wall: under 1 GPU-hour on H100 (no-KV-cache decoding is ~24× per-token; ~10.5k forward passes total).

### 4d. Interpret the primary result

The output JSON's `decision` field gives you the pre-registered verdict, but form your own interpretation:

- **`decision: flat`** — Control A itself produces a flat rank-k curve on vanilla SFT. Interpretation: either the ablation has no discriminating power on ProsQA (ProsQA is rank-1-solvable), OR the intervention site is still surface-decoding territory. Run the robustness variants (`--variant 16x48` and `--variant svd_full_768` in separate output dirs) to check if the flat result survives wider coverage. If all three flat: paper reframes as "on ProsQA, no tested architecture/objective produces rank-dependent behavior; we interpret this as the task being rank-1-solvable." If only 16x16 flat but wider variants bend: the 16x16 choice was the coverage issue and matrix-CODI's flatness is informative about the objective.
- **`decision: bending`** — Control A's curve bends. The ablation HAS discriminating power on ProsQA. Matrix-CODI's flat curves ARE informative about the CODI objective. Paper's current framing is vindicated. Strong win.
- **`decision: ambiguous`** — Run the robustness variants first. If still ambiguous, consider a layer sweep `--intervene-layer 3` and `--intervene-layer 9` to see if intervention depth matters. Report both first-token-accuracy and full-sequence-accuracy decisions separately.

**Cross-check:** the `k=None` unablated row should be within ~2pp of Round 4's 81.77% for seed 1337 (per `experiment-runs/2026-04-13_round4_vanilla_sft/pure_sft_seed1337_SUMMARY.txt`). If it's off by more, the script's decoding path diverges from the reference Round 4 eval and the result is not trustworthy regardless of the k-sweep shape.

**Cross-check:** the `k="random"` sensitivity floor should drop accuracy meaningfully (exact threshold not pre-registered, but expect sub-60% if the hidden-state region matters). If randomizing the region has negligible effect, the intervention site is uninformative regardless of the k-sweep shape — this is the "Control A is uninformative" null that you must flag loudly.

### 4e. Robustness variants (run if primary sweep shipped clean)

```
# 16x48 coverage (broader dimension coverage)
python3 matrix-thinking/scripts/run_control_a.py \
    [... same args ...] --variant 16x48 \
    --output-dir experiment-runs/2026-04-24_control_a/robust_16x48/

# svd_full_768 (full hidden state)
python3 matrix-thinking/scripts/run_control_a.py \
    [... same args ...] --variant svd_full_768 \
    --output-dir experiment-runs/2026-04-24_control_a/robust_svd_full/
```

### 4f. Log everything

Update `EXPERIMENT_LOG.md` with a new entry for Control A: date, commit SHA of the script used, results summary, decision, any caveats, pointer to the output JSON. Follow the style of recent entries.

If a lesson was learned during the run, also emit a `[LEARN]` block so `learn-capture.sh` picks it up.

---

## 5. Phase 2 — Incorporate into the paper

### 5a. Update `PAPER_RESULTS_SUMMARY.md`

Add a new section `## Table 6: Control A (null baseline on vanilla SFT)` with:
- Per-seed per-k accuracy table (full-sequence and first-token)
- Pooled Spearman r and p
- Decision per the pre-registered rule
- Cross-check against Round 4's 81.77% unablated
- Sensitivity-floor (random-h) result
- Scope-of-claim statement — see implementation notes §5

### 5b. Update the paper outline

`PAPER_WRITER_BRIEF.md` §5 "Positive Control" becomes §5 "Controls." Add §5.6 "Control A: Null Baseline on Vanilla SFT" (1 page). Structure:
1. Motivation (addresses attack A16 — ProsQA-rank-1-solvable)
2. Protocol (propagating hook-based intervention, 6 analog positions before `:`, scope-of-claim acknowledged)
3. Result (plot overlaying Control A curve with matrix-CODI curve from Round 3)
4. Interpretation — honest about what the curve does and doesn't support

### 5c. Adjust §7 Discussion based on outcome

- Flat Control A → rewrite §7 "ProsQA-might-be-rank-1-solvable" paragraph. Not a caveat anymore — a primary finding.
- Bending Control A → §7 upgrades from "acknowledged caveat" to "ablation has discriminating power; matrix-CODI's flatness is an objective-level property."
- Ambiguous → §7 retains the caveat honestly with the Control A bound.

### 5d. Figures

Add `fig7_control_a.pdf` overlaying Control A rank-k curve (primary 16x16) with the matrix-CODI flat curve from Round 3. If you ran 16x48 and svd_full_768, overlay those too in a single subplot.

---

## 6. Phase 3 — Continue the roadmap

Only after Phase 1 + 2 are done. Items in rough priority order, with time budgets:

### 6a. Settle the remaining paper tables

- **Table 3 GPT-2 large matrix-CODI.** Currently "pending (OOM×2)." Work-in-progress in `experiment-runs/2026-04-13_round9_gpt2large/`. Either resolve the OOM and run, or cut from the paper and note the gap honestly in §7. Do not fake-extrapolate. ~2–4 GPU-hours if you can get it past OOM; mark as cut in the paper if not.
- **Table 4 n_latents depth sweep.** n=16/32/64 still pending. Required for the depth-doesn't-rescue-matrix-CODI section (`PAPER_WRITER_BRIEF.md` §4.1). Sweep script is `matrix-thinking/scripts/run_matrix_codi.py --mode train_matrix`. ~6 GPU-hours total. **Priority high** — this is already in the paper outline.

### 6b. Write the paper (dual output)

Follow `PAPER_WRITER_BRIEF.md` exactly. Two outputs:
1. `pebble-ai-site/findings/matrix-codi-rank-blindness-paper.html` — website version. Match existing findings-page style. Editorial "we" not "I." Full graphs. Do NOT use the word "audit" anywhere in this output (per brief).
2. `matrix-thinking/submissions/icml-mi-workshop-2026/main.pdf` — anonymized 8-page LaTeX. Build with `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`.

Hard constraints from the brief:
- **Double-blind.** No Sam's name, no GitHub handles, no HF usernames, no pebbleml.com links in the LaTeX.
- **Anonymous code link:** `anonymous.4open.science`, not GitHub URL.
- **8 pages excluding references.** Unlimited appendix.
- **Reciprocal reviewing:** Sam must sign up for 3 reviews at submission time — flag this to him separately so he doesn't miss it.

### 6c. Submission

Upload anonymized PDF to OpenReview (`https://openreview.net/group?id=ICML.cc/2026/Workshop/Mech_Interp`) before May 8 AOE. Upload anonymized code bundle to `anonymous.4open.science`. Confirm with Sam before final submit.

### 6d. Post-submission

Land:
- Commit the final PDF + LaTeX source
- Publish the website findings page (coordinate with Sam on timing — Substack/GitHub/HF fan-out is a separate workstream, see `pebble-ai-site/PUBLIC_PRESENCE_BRIEF.md` if relevant)
- Update `STATE.md` to reflect post-submission state
- Queue the next direction per `matrix-thinking/CHAPTER_2_DESIGN.md` (your read-only awareness item — don't start Chapter 2 without Sam's go)

---

## 7. How to work

### Git hygiene
- Small, frequent commits. Descriptive messages. NO `Co-Authored-By: Claude` trailers.
- Use `/clean` skill before commits (pre-commit-gate hook requires it).
- Commit after each smoke-test fix, each experiment completion, each paper-section addition.
- Branch strategy: work on `main` unless a change is genuinely exploratory — this is a deadline sprint, not a feature-branch project.

### Smoke tests and audits
- Before any multi-GPU-hour run, smoke test at n=10 or similar.
- Before committing new code that will run on the H100, consider a quick audit pass (spawn a subagent or self-review with fresh eyes).
- Per `CLAUDE.md`: "the implementer does not review their own work" — if you write significant new code, spawn an audit agent before running.

### When to spawn subagents
- For any new experiment design question: research agent (Opus) to check literature + prior rounds.
- For any new intervention code: attack agent (Opus) on the pre-run code.
- For the paper write: paper-writer agent with `PAPER_WRITER_BRIEF.md`.
- Per Sam's subagent policy: Opus for high-priority work (this whole workflow qualifies).

### Repo conventions
- No emojis in code or docs.
- Short clear functions. Comment the non-obvious only.
- Save the exact script that ran alongside results (`shutil.copy(__file__, output_dir)`).
- One dataset per comparison — don't swap data between rounds.
- Log everything to file; produce a human-readable SUMMARY.txt at the end.

### Sam's preferences (from his memory)
- No vector collapse anywhere — all matrix, all the way.
- Byte-level is Phase 2; not this paper.
- Don't ask him questions with obvious answers during a cascade. If the spec or prior lesson gives the answer, make the call and move.
- Recent research (last 6–12 months) preferred for ML/AI literature checks.
- Continuous experimentation; don't stop; use all available compute.

---

## 8. Escalate to Sam when

- Checkpoints are missing and a re-run is needed (before you waste GPU-hours guessing).
- Smoke test fails in a way you can't diagnose after 4 iterations.
- An experimental result would require changing the paper's central claim (Control A going flat is fine and handleable; a result that invalidates §3 central mechanism argument is not).
- A submission-logistics question (reciprocal reviewing, OpenReview account, funding attribution).
- Anything that touches public presence (Substack/GitHub/HF org creation) — that work is in a separate brief (`pebble-ai-site/PUBLIC_PRESENCE_BRIEF.md`) and is not part of this mission.
- You hit a hard ethical or validity wall where shipping on deadline would require misrepresenting a result.

## 9. What NOT to do

- Do not run the full sweep before smoke passes.
- Do not modify `run_matrix_codi.py` to get `run_control_a.py` to work — fix the importer, not the parent.
- Do not cut experiments from the paper to "make the deadline" without flagging to Sam first.
- Do not fabricate or extrapolate results. Negative results are a feature.
- Do not commit secrets, local paths, or H100 SSH keys to the repo.
- Do not push force, do not amend merged commits.
- Do not submit to ICML without Sam's final approval on the PDF.

## 10. Report back

Milestone-based, not per-step:

- **M1 (pre-flight done):** Checkpoints verified on H100, env ready, first smoke iteration attempted. Brief status note.
- **M2 (smoke clean):** Smoke passed all three variants. Sweep ready.
- **M3 (primary sweep done):** Control A result in hand. Decision verdict. Cross-checks passed. JSON + SUMMARY.txt paths.
- **M4 (paper updated with Control A):** `PAPER_RESULTS_SUMMARY.md`, outline, §7 framing all reflect the new result.
- **M5 (remaining tables done or explicitly cut):** Table 3 GPT-2 large + Table 4 depth sweep.
- **M6 (paper draft v1 complete):** Both LaTeX and HTML outputs exist and render. Self-audit pass.
- **M7 (submission ready):** Anonymized, compiled, code bundle uploaded. Sam gives final approval.
- **M8 (shipped):** Submitted to OpenReview. STATE.md updated.

At each milestone: short status (under 300 words), what's done, what's next, blockers if any.

---

## 11. Files under your ownership

You may freely modify (with commits):
- `matrix-thinking/scripts/run_control_a.py` and any new scripts
- `matrix-thinking/PAPER_RESULTS_SUMMARY.md`
- `matrix-thinking/PAPER_WRITER_BRIEF.md`
- `matrix-thinking/submissions/icml-mi-workshop-2026/` (entire directory)
- `pebble-ai-site/findings/matrix-codi-rank-blindness-paper.html`
- `EXPERIMENT_LOG.md`
- `STATE.md`
- Any cascade reports in `matrix-thinking/CONTROL_A_*.md` — append or supersede as needed
- `matrix-thinking/QUEUE.md` — mark Control A done when it is

Treat as read-only without Sam's explicit ask:
- `matrix-thinking/CHAPTER_2_DESIGN.md`
- `matrix-thinking/BILINEAR_READOUT_PATCH_PLAN.md`
- `pebble-ai-site/PUBLIC_PRESENCE_BRIEF.md`
- `pebble-ai-site/Pebble_AI_Pitch_Deck.*`
- Any file in `pebble-ai-site/findings/` other than the paper HTML

---

## 12. Success criteria

You are done when:

1. Control A primary sweep (16x16) has a clean result written to `experiment-runs/2026-04-24_control_a/primary_16x16/`, decision verdict logged.
2. `PAPER_RESULTS_SUMMARY.md` has Table 6 (Control A) added.
3. `PAPER_WRITER_BRIEF.md` §5 outline is updated for Control A.
4. Paper v1 compiles to a valid anonymous 8-page PDF at `matrix-thinking/submissions/icml-mi-workshop-2026/main.pdf`.
5. Website findings page at `pebble-ai-site/findings/matrix-codi-rank-blindness-paper.html` renders cleanly.
6. Sam approves the PDF. Submission uploaded to OpenReview. Code uploaded to `anonymous.4open.science`.
7. `STATE.md` updated. `EXPERIMENT_LOG.md` current.

Then you're done. Hand back to Sam.

---

*This brief is the complete spec. If something here contradicts a later Sam directive, follow Sam.*
