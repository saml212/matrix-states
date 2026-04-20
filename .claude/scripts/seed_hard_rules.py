#!/usr/bin/env python3
# One-shot: seed workflow.db with the Hard Rules from CLAUDE.md and feedback
# from MEMORY.md. Idempotent — skips duplicates by (project, category, rule).
import sqlite3, pathlib

DB = pathlib.Path(__file__).resolve().parent.parent / "memory" / "workflow.db"
PROJECT = "learned-representations"

# (category, rule, mistake_or_None, correction_or_None)
SEEDS = [
    # Process rules
    ("process", "Verify claims before stating them", None,
     "Use web search or research agents. Never assert facts without evidence."),
    ("process", "Audit code with a separate agent before running experiments", None,
     "Implementer does not review their own work."),
    ("process", "Smoke test every model before training", None,
     "Forward pass, backward pass, gradient check."),
    ("process", "Use standard NLP benchmarks for publishable claims", None,
     "Byte BPC is for internal use only."),
    ("process", "Dead directions stay dead", None,
     "Don't revisit archived ideas unless the user asks."),
    ("process", "Save the exact script alongside experiment results", None,
     "Reproducibility requires the actual code that ran, not a description."),
    ("process", "Log everything to a file; produce a human-readable summary", None,
     "Raw logs are for debugging; summary is for the next agent."),

    # Scale rules
    ("scale", "288K params is unigram-statistics territory", None,
     "Can't draw conclusions about reasoning/generalization at this scale. Need 10M+ minimum."),
    ("scale", "Param-matched flat-vector ablation blocks ALL downstream decisions",
     "Ran matrix experiments without a vector baseline.",
     "Run the flat-vector ablation before anything else."),

    # Architecture rules
    ("matrix-arch", "Never compress matrices to vectors",
     "Compressing to vectors loses all structure.",
     "Use MultiProbeHead (bilinear probes) for output."),
    ("matrix-arch", "Reshape equivalence: d²-vector = d×d matrix bidirectionally",
     "Assumed matrix ops automatically gave structure.",
     "Structure matters only if OPERATIONS preserve it. Flatten = structure gone."),
    ("matrix-arch", "Making matrix ops cheaper does NOT fix the quality gap", None,
     "Speed ≠ quality. Cheap ops are orthogonal to model capability."),
    ("matrix-arch", "DeltaNet rank-1 updates are for recurrent models, not iterative attention", None,
     "Don't conflate recurrent and attention architectures."),
    ("matrix-arch", "'Just add layers' beats thought interleaving at 288K params", None,
     "At small scale, depth beats iteration."),

    # Training rules
    ("training", "PonderNet halting collapses at small scale", None,
     "Use fixed iterations first, adaptive later."),
    ("training", "Thought appending with causal mask is broken", None,
     "Use iterative in-place refinement instead."),
    ("training", "Outer-product embedding init: u,v std = sqrt(target_std)",
     "Used target_std for u,v — products then had std=σ² (too small).",
     "u,v std must be sqrt(target_std). Products have std=σ²."),
    ("training", "Combined speedups don't multiply when targeting overlapping pipeline stages", None,
     "Measure each optimization in isolation before combining."),
    ("training", "Sweep experiments (multiple configs, one script, sequential) save GPU downtime", None,
     "Add try/except so one crash doesn't kill remaining configs."),

    # Distributed rules
    ("distributed", "DDP eval on rank 0 only NCCL-timeouts if eval > 10 min",
     "Default NCCL timeout is 10 min. Long eval on rank 0 while other ranks wait causes a hang.",
     "Set timeout to 30 min AND cap eval batches to 50 max."),
    ("distributed", "Smoke test must include EVAL batch size, not just training", None,
     "Eval can OOM even if training fits."),
    ("distributed", "batch=96 per GPU is the safe max for mat_dim=32 on H100 80GB",
     "batch=112 fits training but OOMs during eval.",
     "Cap at 96 unless you also verify eval headroom."),
    ("distributed", "The 50K vocab logits tensor is the VRAM bottleneck", None,
     "Not model activations. Optimize vocab projections first."),

    # PyTorch rules
    ("pytorch", "nn.MultiheadAttention requires explicit attn_mask OR is_causal, not both", None,
     "Passing both throws an error in PyTorch 2.4+."),
    ("pytorch", "HF cache defaults to container disk (/root/.cache/)",
     "Container disk filled up mid-training.",
     "Symlink HF_HOME to persistent volume immediately."),

    # Data rules
    ("data", "Use the same dataset for ALL experiments in a comparison", None,
     "Don't swap data between rounds — breaks comparability."),

    # Feedback from MEMORY.md
    ("feedback", "NO vector collapse anywhere in the matrix pipeline", None,
     "Matrix all the way. Every stage must preserve 2D structure."),
    ("feedback", "User wants continuous experimentation, never stop, use all available compute", None,
     "Default posture: find the next experiment to run. Don't idle."),
    ("feedback", "Research agents on ML/AI must prioritize recent work (last 6-12 months)",
     "Cited old papers that had been superseded.",
     "Prioritize last 6-12 months for ML/AI. Neuroscience/mature fields exempt."),
    ("feedback", "Sonnet for workhorse subagents, Opus for architect/attack/high-risk", None,
     "Don't burn Opus tokens on mechanical tasks."),
    ("feedback", "Don't add Co-Authored-By Claude trailers to git commits", None,
     "User doesn't want them."),
    ("feedback", "Byte-level d=16 is the target; freeform matrix thinking; multi-byte output; report BPB", None,
     "That's the long-term vision. Current scale runs are stepping stones."),

    # Harness-specific
    ("harness", "Pin sqlite3 to /usr/bin/sqlite3, not PATH default",
     "Android commandlinetools sqlite3 at /opt/homebrew/share/... lacks FTS5.",
     "Hardcode /usr/bin/sqlite3 in hook scripts."),
    ("harness", "FTS5 triggers on regular tables require PRAGMA trusted_schema=1 for writes",
     "Got 'unsafe use of virtual table' error.",
     "Prepend PRAGMA trusted_schema=1; to every INSERT/UPDATE on the tracked table."),
]

conn = sqlite3.connect(DB)
conn.execute("PRAGMA trusted_schema=1")
cur = conn.cursor()
inserted, skipped = 0, 0
for category, rule, mistake, correction in SEEDS:
    cur.execute(
        "SELECT id FROM learnings WHERE project=? AND category=? AND rule=? LIMIT 1",
        (PROJECT, category, rule),
    )
    if cur.fetchone():
        skipped += 1
        continue
    cur.execute(
        "INSERT INTO learnings (project, category, rule, mistake, correction, source) "
        "VALUES (?,?,?,?,?,?)",
        (PROJECT, category, rule, mistake, correction, "seed"),
    )
    inserted += 1

conn.commit()
conn.close()
print(f"seeded: {inserted} inserted, {skipped} skipped (already present)")
