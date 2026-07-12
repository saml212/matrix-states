"""
Generate the first real chart for the literature-synthesis findings page
(pebble-ai-site/findings/literature-synthesis.html), replacing the
ASCII-art convergence diagram's neighboring "no real chart" gap with an
actual data-backed figure: citation count by publication year, stacked by
the article's own theme numbering (3.1-3.6 in the HTML).

DATA PROVENANCE (read this before trusting the numbers):
This is a bibliographic parse, not an experimental-raw load, so the
md5-manifest convention used for experiment JSONs does not apply --
references.md is prose written by a human/agent over months, not a
generated data artifact. The DATA list below was produced by reading
references.md (804 lines, repo root) in full on 2026-07-11 and manually
transcribing every bolded "**Title**" citation entry together with its
stated publication year and its enclosing "##"/"###" section header.
192 distinct bolded citation entries were found (one further mention --
DeltaProduct, arXiv:2502.10297 -- appears twice in the source file, once
under "## Long-Context" and again bundled into a Grazzi et al. bullet
under "## Rank / Dimensionality Measurement"; it is counted once here,
at its first/fuller appearance, and the second mention is not double
counted). Of the 192, 4 have a publication year that is genuinely
ambiguous or missing in the source text and are recorded here with
year=None and excluded from the chart (see SKIPPED below) rather than
guessed.

YEAR EXTRACTION RULE (stated so a re-parse can be checked against it):
when the source entry states an explicit venue + year (e.g. "ICLR 2025",
"NeurIPS 2025 Spotlight", "EMNLP Findings 2025"), that venue year is used,
even if an arXiv id or "submitted" month implies an earlier preprint date
(e.g. COCONUT: arXiv:2412.06769 is Dec 2024, but "Hao et al., Meta, ICLR
2025" is used -> 2025). When only a bare month+year or a bare year is
given (no venue), that value is used directly. When the source text gives
two DIFFERENT, non-venue-disambiguated years for the same entry (e.g.
"2023, updated 2025", "2019/2020", "2023/2024"), the entry is skipped as
ambiguous per the task's instruction, rather than picking one.

THEME MAPPING: references.md has 18 topical "##" sections (see the
section comments below). For a legible 6-color stacked bar, these are
grouped into the 6 meta-themes the article's own Section 03 already
argues in (3.1 through 3.6), plus a 6th bucket for the threads the
article's own Section 05 names as "investigated and excluded" (energy-
based transformers, test-time training, long-context sparse attention,
hypercomplex multi-modal fusion) -- extended here to cover every
references.md section that isn't one of the five numbered themes
(long-context, auxiliary-loss/multi-token, multi-modal/domain-agnostic,
energy-based/test-time compute, quantization). This is a real grouping
of the file's own section structure, consolidated for a 6-color palette,
not an invented taxonomy:
  reasoning  (3.5) <- "Continuous Reasoning" + "Rank / Dimensionality
                       Measurement" + "Consensus Training Methods"
  jepa       (3.1) <- "JEPA Family" + "Pure-Sensor / Language-Free
                       Representation Learning"
  structure  (3.3) <- "Structured Representations Beyond Flat Vectors"
                       + "Scaling vs Structure Debate" + "Hypercomplex
                       Multi-Modal Fusion"
  bytes      (3.2) <- "Discrete Vocabularies / Tokenization" +
                       "Byte-Level Models" + "Tokenizer Research"
  neuro      (3.4) <- "Neuroscience / Information Theory of Language" +
                       "Bio-Inspired Computation"
  excluded   (other)<- "Long-Context" + "Auxiliary Loss / Multi-Token /
                       Context Prediction" + "Multi-Modal / Domain-
                       Agnostic" + "Energy-Based / Test-Time Compute" +
                       "Quantization (Not Currently Relevant)"
"Code Repositories Referenced" (the final references.md section) lists
GitHub links to code already cited elsewhere in the file, not new
citations, and is not parsed.

Palette: Okabe-Ito. No in-figure title (caption lives in the HTML).
Output: pebble-ai-site/assets/plots/literature_synthesis_years.svg
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

BG = "#FAF5E7"
TEXT = "#1a1a1a"
MUTED = "#5a5a5a"

# Okabe-Ito
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_GREEN = "#009E73"
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_PURPLE = "#CC79A7"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["text.color"] = TEXT
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = TEXT
plt.rcParams["ytick.color"] = TEXT
plt.rcParams["axes.edgecolor"] = TEXT

OUT_DIR = "/Users/samuellarson/Experiments/learned-representations/pebble-ai-site/assets/plots"

THEME_LABEL = {
    "reasoning": "continuous reasoning (3.5)",
    "jepa": "JEPA / non-AR SSL at scale (3.1)",
    "structure": "structure vs scale (3.3)",
    "bytes": "bytes over vocab (3.2)",
    "neuro": "neuroscience of cognition (3.4)",
    "excluded": "excluded threads (Sec. 05)",
}
THEME_COLOR = {
    "reasoning": OI_ORANGE,
    "jepa": OI_SKY,
    "structure": OI_GREEN,
    "bytes": OI_BLUE,
    "neuro": OI_PURPLE,
    "excluded": OI_VERMILLION,
}
THEME_ORDER = ["reasoning", "jepa", "structure", "bytes", "neuro", "excluded"]

# (title, year_or_None, theme) -- one row per bolded "**Title**" citation
# entry in references.md, in file order. year=None rows are ambiguous/
# missing in the source and are reported+skipped, not guessed.
DATA = [
    # ## Continuous Reasoning > ### Foundational
    ("COCONUT", 2025, "reasoning"),
    ("CODI", 2025, "reasoning"),
    ("Quiet-STaR", 2024, "reasoning"),
    ("Pause Tokens", 2024, "reasoning"),
    # ### Theoretical foundation for the rank-superposition hypothesis
    ("Reasoning by Superposition", 2025, "reasoning"),
    ("CoT2", 2026, "reasoning"),
    ("Emergence of Superposition", 2025, "reasoning"),
    # ### The 2026 rebuttal
    ("The Illusion of Superposition", 2026, "reasoning"),
    # ### COCONUT successors
    ("CoLaR", 2025, "reasoning"),
    ("MarCos", 2025, "reasoning"),
    ("SIM-CoT", 2025, "reasoning"),
    ("PCCoT", 2025, "reasoning"),
    ("Inference-time Scaling for Continuous Space Reasoning", 2025, "reasoning"),
    ("A Survey on Latent Reasoning", 2025, "reasoning"),

    # ## JEPA Family
    ("LeJEPA", 2025, "jepa"),
    ("LLM-JEPA", 2025, "jepa"),
    ("LeWorldModel", 2026, "jepa"),
    ("V-JEPA 2", 2025, "jepa"),
    ("V-JEPA 2.1", 2026, "jepa"),
    ("VL-JEPA", 2025, "jepa"),
    ("ThinkJEPA", 2026, "jepa"),
    ("WavJEPA", 2025, "jepa"),
    ("Audio-JEPA", 2025, "jepa"),
    ("JEPA as a Neural Tokenizer", 2025, "jepa"),
    ("ACT-JEPA", 2025, "jepa"),

    # ## Pure-Sensor / Language-Free Representation Learning
    ("DINOv3", 2025, "jepa"),
    ("Web-SSL", 2025, "jepa"),
    ("Data or Language Supervision?", 2025, "jepa"),
    ("Object Binding in Vision Transformers", 2025, "jepa"),
    ("Massively Multimodal Foundation Models w/ Specialized MoE", 2025, "jepa"),
    ("SONAR", 2025, "jepa"),
    ("Revisiting the Platonic Rep. Hypothesis (Aristotelian View)", 2026, "jepa"),
    ("The Platonic Representation Hypothesis (original)", 2024, "jepa"),

    # ## Structured Representations Beyond Flat Vectors > Hyperbolic/Lie/Manifold
    ("HELM", 2025, "structure"),
    ("HyperET", 2025, "structure"),
    ("HypLoRA", 2025, "structure"),
    ("Hyperbolic Large Language Models (survey)", 2025, "structure"),
    ("HyperHELM", 2025, "structure"),
    ("Clebsch-Gordan Transformer", 2025, "structure"),
    # ### Geometric Algebra / Clifford
    ("CliffordNet", 2026, "structure"),
    ("Geometric Algebra Transformer (GATr)", 2023, "structure"),
    ("ViNE-GATr", 2025, "structure"),
    ("Equivariant Spherical Transformer", 2025, "structure"),
    ("Platonic Transformers", 2025, "structure"),
    # ### TPR / Tensor Product Representations
    ("Smolensky's Tensor Product Representations", 1990, "structure"),
    ("Soft TPR", 2024, "structure"),
    ("TP-Transformer (2021)", 2021, "structure"),
    ("Attention-based Iterative Decomposition for TPR", 2024, "structure"),
    # ### PHM / Quaternion / Hypercomplex
    ("PHM Layers", 2021, "structure"),
    ("Quaternion Transformer", 2019, "structure"),
    ("Compacter", 2021, "structure"),
    # ### Higher-Order Tensor
    ("Higher-Order Transformers (HOT)", 2024, "structure"),
    ("Deep Tensor Network", None, "structure"),  # SKIP: "2023, updated 2025" -- ambiguous
    ("Training Tensor Attention Efficiently", 2024, "structure"),
    ("Matrix Neural Networks (MatNet)", 2016, "structure"),

    # ## Scaling vs Structure Debate
    ("Scaling can lead to compositional generalization", 2025, "structure"),
    ("Does equivariance matter at scale?", 2025, "structure"),
    ("Deep Learning is Not So Mysterious or Different", 2025, "structure"),
    ("Compositional Gen. Requires More Than Disentangled Reps.", 2025, "structure"),
    ("Does Data Scaling Lead to Visual Compositional Generalization?", 2025, "structure"),
    ("Revisiting Compositional Generalization Capability of LLMs", 2025, "structure"),

    # ## Discrete Vocabularies / Tokenization (Modern VQ)
    ("Emu3.5", 2025, "bytes"),
    ("Emu3", 2024, "bytes"),
    ("Chameleon", 2024, "bytes"),
    ("Janus-Pro", 2025, "bytes"),
    ("TokenFlow", 2025, "bytes"),
    ("UniTok", 2025, "bytes"),
    ("QLIP", 2025, "bytes"),
    ("BSQ", 2025, "bytes"),
    ("FSQ", 2024, "bytes"),
    ("MAGVIT-v2 / LFQ", 2024, "bytes"),
    ("Open-MAGVIT2", 2024, "bytes"),
    ("SimVQ", 2025, "bytes"),
    ("RFSQ", 2025, "bytes"),
    ("TA-VQ", 2025, "bytes"),
    ("SemHiTok", 2025, "bytes"),
    ("Discrete Tokenization for Multimodal LLMs (survey)", 2025, "bytes"),
    ("UniWeTok", 2026, "bytes"),
    ("VAEVQ", 2025, "bytes"),
    ("VQRAE", 2025, "bytes"),

    # ## Byte-Level Models
    ("BLT", 2025, "bytes"),
    ("MBLM", 2025, "bytes"),
    ("MambaByte", 2024, "bytes"),
    ("EvaByte", 2025, "bytes"),
    ("Bolmo", 2025, "bytes"),
    ("bGPT", 2024, "bytes"),
    ("MEGABYTE", 2023, "bytes"),
    ("MrT5", 2025, "bytes"),
    ("ByteFlow", 2026, "bytes"),

    # ## Long-Context
    ("EverMind MSA", 2026, "excluded"),
    ("Native Sparse Attention (NSA)", 2025, "excluded"),
    ("DeepSeek Sparse Attention (DSA)", 2025, "excluded"),
    ("Infini-Attention", 2024, "excluded"),
    ("Ring Attention", 2024, "excluded"),
    ("MInference", 2024, "excluded"),
    ("StreamingLLM", 2024, "excluded"),
    ("Titans", 2025, "excluded"),
    ("TTT-Linear / TTT-MLP", 2024, "excluded"),
    ("TTT-E2E", 2025, "excluded"),
    ("Gated DeltaNet", 2025, "excluded"),
    ("DeltaProduct", 2025, "excluded"),
    ("Kimi Linear", 2025, "excluded"),

    # ## Auxiliary Loss / Multi-Token / Context Prediction
    ("ContextLM", 2026, "excluded"),
    ("Future Summary Prediction (FSP)", 2025, "excluded"),
    ("Multi-Token Prediction (Gloeckle et al.)", 2024, "excluded"),
    ("OLA-VLM", 2024, "excluded"),

    # ## Rank / Dimensionality Measurement
    ("Attention is not all you need (rank collapse)", 2021, "reasoning"),
    ("Dimensional Collapse in Transformer Attention Outputs", 2025, "reasoning"),
    ("Local Intrinsic Dimensions of Contextual LMs", 2025, "reasoning"),
    ("Higher Embedding Dim. Creates a Stronger World Model", 2025, "reasoning"),
    ("Mind the Gap: Spectral Analysis of Rank Collapse", 2024, "reasoning"),
    ("Statistical Physics of Language Model Reasoning", None, "reasoning"),  # SKIP: no year given
    ("Token Embeddings Violate the Manifold Hypothesis", 2025, "reasoning"),
    ("Measuring Intrinsic Dimension of Token Embeddings", 2025, "reasoning"),
    ("Origin of Self-Attention", 2025, "reasoning"),
    ("Minimax Rates for Pairwise Interactions in Attention", 2025, "reasoning"),
    ("Understanding Transformers for Time Series: Rank Structure", 2025, "reasoning"),
    # ### Added 2026-07-01 (Chapter 2 -- Task D/E research)
    ("Understanding Factual Recall via Associative Memories", 2025, "reasoning"),
    ("The Key to State Reduction in Linear Attention", 2026, "reasoning"),
    ("State Rank Dynamics in Linear Attention LLMs", 2026, "reasoning"),
    ("Linear Transformers Are Secretly Fast Weight Programmers", 2021, "reasoning"),
    ("TP-Transformer (Schlag et al., math problem solving)", None, "reasoning"),  # SKIP: "2019/2020"
    ("RNNs Implicitly Implement Tensor Product Representations", 2019, "reasoning"),
    ("Unlocking State-Tracking in Linear RNNs (neg. eigenvalues)", 2025, "reasoning"),
    # NOTE: "DeltaProduct" is cross-referenced again in this bullet
    # (same arXiv:2502.10297 already counted above under Long-Context)
    # and is deliberately not double-counted here.
    ("Zoology", None, "reasoning"),  # SKIP: "2023/2024"
    ("Grokked Transformers are Implicit Reasoners", 2024, "reasoning"),
    ("Faith and Fate: Limits of Transformers on Compositionality", 2023, "reasoning"),
    ("Holographic Reduced Representations", 1995, "reasoning"),
    ("Resonator Networks", 2020, "reasoning"),
    ("Tokenization Counts", 2024, "reasoning"),
    ("Efficient Numeracy via BitTokens", 2026, "reasoning"),

    # ## Multi-Modal / Domain-Agnostic
    ("Perceiver IO", 2021, "excluded"),
    ("Zonkey", 2026, "excluded"),
    ("MAGNET", 2024, "excluded"),
    ("HighMMT", 2023, "excluded"),
    ("CoCoMix", 2025, "excluded"),
    ("ThoughtBubbles", 2025, "excluded"),

    # ## Tokenizer Research
    ("VOLT", 2021, "bytes"),
    ("SuperBPE", 2025, "bytes"),
    ("ADAT", 2024, "bytes"),
    ("T-FREE", 2024, "bytes"),
    ("Scaling Laws with Vocabulary", 2024, "bytes"),
    ("Information-Theoretic Perspective on Tokenizers", 2026, "bytes"),

    # ## Energy-Based / Test-Time Compute
    ("Energy-Based Transformers", 2025, "excluded"),
    ("EBM-CoT", 2025, "excluded"),

    # ## Hypercomplex Multi-Modal Fusion
    ("Hierarchical Hypercomplex Network", 2024, "structure"),
    ("RP-KrossFuse", 2025, "structure"),
    ("Tensor Fusion Network", 2017, "structure"),

    # ## Neuroscience / Information Theory of Language
    ("Fedorenko, Piantadosi & Gibson (Nature 2024)", 2024, "neuro"),
    ("Moser et al. -- Grid Cells in Cognition", 2024, "neuro"),
    ("Chaudhuri et al. -- Intrinsic attractor manifold", 2019, "neuro"),
    ("Khona & Fiete -- Attractor and integrator networks", 2022, "neuro"),
    ("Olshausen & Field -- sparse coding", 1996, "neuro"),
    ("Lian & Burkitt -- Hippocampal Place Map", 2021, "neuro"),
    ("Veit & Nieder -- Abstract rule neurons", 2013, "neuro"),
    ("Barlow -- sensory messages", 1961, "neuro"),
    ("Rao & Ballard -- predictive coding", 1999, "neuro"),
    ("Chalk, Marre & Tkacik -- unified efficient/predictive/sparse", 2018, "neuro"),
    ("Benjamin et al. -- efficient neural codes", 2022, "neuro"),
    ("Friston -- free-energy principle", 2010, "neuro"),
    ("Parr, Pezzulo & Friston -- Active Inference", 2022, "neuro"),
    ("Millidge et al. -- neuro-mimetic predictive coding survey", 2025, "neuro"),
    ("Shannon -- entropy of printed English", 1951, "neuro"),
    ("Bentz et al. -- Entropy Rate Estimates", 2017, "neuro"),
    ("Zaslavsky et al. -- color naming", 2018, "neuro"),
    ("Levy & Jaeger -- information density", 2007, "neuro"),
    ("Ferrer-i-Cancho & Sole -- least effort", 2003, "neuro"),
    ("Kanwal et al. -- Zipf's Law of Abbreviation", 2017, "neuro"),
    ("Gustison et al. -- Menzerath's linguistic law", 2016, "neuro"),
    ("Spelke & Kinzler -- Core knowledge", 2007, "neuro"),
    ("Dehaene -- The Number Sense", 2011, "neuro"),
    ("Fodor -- The Language of Thought", 1975, "neuro"),
    ("Piantadosi -- computational origin of representation", 2021, "neuro"),
    ("Dimensionality of Cognition", 2024, "neuro"),
    ("PFC Dimensionality and Cognitive Control", 2025, "neuro"),
    ("Higher-Dimensional Representations and Memory", 2022, "neuro"),
    ("Grid Cells and Abstract Reasoning", 2024, "neuro"),
    ("High-Dimensional Brain in High-Dimensional World", 2020, "neuro"),
    ("Thousand Brains Theory", 2024, "neuro"),

    # ## Bio-Inspired Computation
    ("Dendritic Computation", 2025, "neuro"),
    ("ReSU (Rectified Spectral Units)", 2026, "neuro"),
    ("KAN (Kolmogorov-Arnold Networks)", 2025, "neuro"),
    ("SpikingLLM", 2026, "neuro"),
    ("Deep Oscillatory Neural Network", 2025, "neuro"),
    ("Predictive Coding (Nature Communications 2025)", 2025, "neuro"),

    # ## Consensus Training Methods (for thinking models)
    ("DeepSeek-R1", 2025, "reasoning"),
    ("Let's Verify Step by Step (PRM800K)", 2024, "reasoning"),
    ("PonderNet", 2021, "reasoning"),
    ("Universal Transformers", 2019, "reasoning"),
    ("LoopFormer", 2026, "reasoning"),
    ("Mixture-of-Recursions", 2025, "reasoning"),

    # ## Quantization (Not Currently Relevant)
    ("TurboQuant", 2026, "excluded"),
]

assert len(DATA) == 192, f"expected 192 parsed entries, got {len(DATA)}"

valid = [(t, y, th) for (t, y, th) in DATA if y is not None]
skipped = [(t, th) for (t, y, th) in DATA if y is None]
print(f"parsed {len(DATA)} citation entries from references.md")
print(f"  {len(valid)} with a usable year, {len(skipped)} skipped (ambiguous/missing year):")
for t, th in skipped:
    print(f"    - {t}  [{th}]")

# ---------------------------------------------------------------- bucketing
# Individual bars for 2020-2026 (where the vast majority of the bibliography
# sits); a single "pre-2020" bar aggregates the neuroscience/info-theory
# classics (Shannon 1951 through Barlow, Fodor, Smolensky, etc.) so the
# chart stays legible rather than carrying ~70 mostly-empty year bins.
years_present = sorted(set(y for _, y, _ in valid))
recent_years = [y for y in years_present if y >= 2020]
pre2020_count = sum(1 for _, y, _ in valid if y < 2020)

buckets = ["pre-2020"] + [str(y) for y in recent_years]
bucket_theme_counts = {b: Counter() for b in buckets}
for _, y, th in valid:
    b = "pre-2020" if y < 2020 else str(y)
    bucket_theme_counts[b][th] += 1

bucket_totals = {b: sum(bucket_theme_counts[b].values()) for b in buckets}
print("\nyear-bucket totals:")
for b in buckets:
    print(f"  {b}: {bucket_totals[b]}")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor=BG)
ax.set_facecolor(BG)

x = range(len(buckets))
bottom = [0] * len(buckets)
for th in THEME_ORDER:
    heights = [bucket_theme_counts[b].get(th, 0) for b in buckets]
    ax.bar(x, heights, bottom=bottom, width=0.62, color=THEME_COLOR[th],
           edgecolor=TEXT, linewidth=0.6, label=THEME_LABEL[th], zorder=3)
    bottom = [bo + h for bo, h in zip(bottom, heights)]

for xi, b in zip(x, buckets):
    ax.annotate(str(bucket_totals[b]), (xi, bucket_totals[b]),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9, color=TEXT, fontweight="bold")

xticklabels = [f"pre-2020\n(n={pre2020_count})"] + [str(y) for y in recent_years]
ax.set_xticks(list(x))
ax.set_xticklabels(xticklabels, fontsize=9)
ax.set_ylabel("citations in references.md", fontsize=10, labelpad=8)
ax.set_ylim(0, max(bucket_totals.values()) * 1.18)
ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.22, color=TEXT)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_linewidth(1.0)
    ax.spines[spine].set_color(TEXT)

legend = ax.legend(loc="upper left", frameon=True, fontsize=8, facecolor=BG,
                    edgecolor=TEXT, ncol=1, handlelength=1.2, handleheight=1.0)
legend.get_frame().set_linewidth(1.0)

plt.tight_layout()
out_path = f"{OUT_DIR}/literature_synthesis_years.svg"
plt.savefig(out_path, format="svg", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote {out_path}")
