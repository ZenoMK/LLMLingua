# Locked experiment settings -- attention scorer vs. LongLLMLingua.
#
# Every script in this directory imports from here instead of hard-coding
# these values, so a config change shows up as one diff instead of scattered
# edits across scripts. Provenance for each decision is in NOTES.md/FINDINGS.md
# and in the PR/commit history; changes to this file go through a PR like
# anything else.
#
# Settings locked by user review on 2026-08-15 (in response to NOTES.md's
# four open questions), then revised 2026-08-16: fully open-weight pipeline,
# no OpenAI/paid API anywhere -- see FINDINGS.md for that pivot's reasoning.

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Reader: the model that answers the (compressed) prompt. Local, open-weight
# -- no paid API anywhere in this pipeline. This is what achieved-vs-target
# token ratios are measured against, and (via GPU time) what "cost" means
# now that there's no per-token API bill.
# ---------------------------------------------------------------------------

READER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
READER_TEMPERATURE = 0.0  # enforced structurally via greedy decoding (do_sample=False), not passed as a generate() kwarg
READER_MAX_OUTPUT_TOKENS = 100  # matches examples/RAG.ipynb / lost-in-the-middle convention

# Decision (delegated to the harness build, 2026-08-16): Llama-3.1-8B-Instruct
# over Qwen2.5-7B-Instruct. Reasoning: it's gated on HF (meta-llama org), but
# we're already accepting that same org's license for
# LONGLLMLINGUA_COMPRESSOR_MODEL below, so it's not new friction -- one
# acceptance covers both. It's also the more current/interesting model to
# ask "does attention compression work against" than a 2023-era one, and 8B
# fits comfortably on a single mid-tier GPU in bf16.
READER_MODEL_FALLBACK = "Qwen/Qwen2.5-7B-Instruct"  # not gated (Apache-2.0) --
# switch to this ONLY if check_model_access.py or smoke_test.py show
# READER_MODEL is actually unreachable (token/license problem), and only
# after flagging that substitution explicitly, not silently.


# ---------------------------------------------------------------------------
# Compressor backbones
# ---------------------------------------------------------------------------

LONGLLMLINGUA_COMPRESSOR_MODEL = "meta-llama/Llama-2-7b-chat-hf"

# Our method's two scorer sizes. The 7b row deliberately reuses the exact
# same backbone as the LongLLMLingua row above, so any gap between "us" and
# "them" at 7B is attributable to the scoring mechanism (one-pass attention
# vs. two-pass contrastive perplexity), not to a different base model.
ATTENTION_SCORER_MODELS = {
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "7b": "meta-llama/Llama-2-7b-chat-hf",
}

# Per-model layer sweep settings (see layer_sweep.py). ~50/66/75/100%
# depth candidates on a small fixed example set, per model independently
# -- the best layer is model-specific, so no layer gets reused across the
# two sizes. Budget matches FIDELITY_CHECK's choice of "2x" (the milder of
# the two budgets, a reasonable default when the point is comparing
# layers to each other, not stress-testing a tight budget).
LAYER_SWEEP_N_EXAMPLES = 5
LAYER_SWEEP_BUDGET = "2x"

# Filled in by hand from layer_sweep.py's output once it's actually run --
# empty until then. See FINDINGS.md for the run that populates this.
ATTENTION_SCORER_LAYERS = {}


# ---------------------------------------------------------------------------
# Compression pipeline settings, per method row
# ---------------------------------------------------------------------------

# CRITICAL: reorder is OFF everywhere in this project. Our attention method
# has no document-reordering step, so LongLLMLingua must also run without
# one for an apples-to-apples comparison. "original" is llmlingua's literal
# "restore input order, don't reorder" value (see control_context_budget in
# prompt_compressor.py).
REORDER_CONTEXT = "original"

# Full LongLLMLingua: coarse context-level ranking (question-conditioned
# perplexity) + fine token-level contrastive-perplexity compression. This is
# the reproduction row -- flags match README.md's canonical LongLLMLingua
# snippet and examples/RAG.ipynb's NQ demo exactly.
LONGLLMLINGUA_KWARGS = dict(
    rank_method="longllmlingua",
    condition_compare=True,
    condition_in_question="after",
    context_budget="+100",
    reorder_context=REORDER_CONTEXT,
    use_sentence_level_filter=False,
    # dynamic_context_compression_ratio intentionally left at its 0.0
    # default (off) -- revisit only if the internal-consistency check below
    # shows something off about the compressed condition's quality.
)

# The one flag that isolates the signal comparison for this whole project:
# condition_compare toggles LongLLMLingua's two-pass contrastive-perplexity
# step, which our one-pass attention scorer is meant to replace. Everything
# else about the pipeline position (fine-grained, post-coarse-ranking,
# sentence/token selection feeding the same reader) is held constant.
CONTRASTIVE_FLAG = "condition_compare"

# Baselines -- both reachable via rank_method, no reimplementation. Run at
# the same granularity as our method (sentence-level selection, no context-
# or token-level filtering) for a fair comparison. Sentence order in the
# output follows control_sentence_budget's reconstruction, which iterates
# sentences in their original order -- so these are reorder-off by
# construction, no extra flag needed.
BM25_BASELINE_KWARGS = dict(
    rank_method="bm25",
    use_context_level_filter=False,
    use_sentence_level_filter=True,
    use_token_level_filter=False,
)

SENTBERT_BASELINE_KWARGS = dict(
    rank_method="sentbert",  # multi-qa-mpnet-base-dot-v1, per get_rank_results
    use_context_level_filter=False,
    use_sentence_level_filter=True,
    use_token_level_filter=False,
)

BASELINE_ROWS = ["bm25", "sentbert", "longllmlingua"]


# ---------------------------------------------------------------------------
# Token budgets -- measured in the READER's tokenizer (its own HF
# tokenizer now, not tiktoken -- see budgets.py). Absolute target_token
# values, matching the paper's NQ table's 2x and 4x compression conditions.
# ---------------------------------------------------------------------------

TOKEN_BUDGETS = {
    "2x": 3000,
    "4x": 2000,
}


# ---------------------------------------------------------------------------
# Benchmark: NaturalQuestions multi-doc QA (lost-in-the-middle), gold
# document at position 10 of 20 -- the hardest lost-in-the-middle case, and
# where the paper's 21.4% headline number is measured (that specific number
# is no longer our comparison target now that we're off their GPT-3.5
# reader -- see FIDELITY_CHECK below -- but position 10 is still the right
# slice to stress-test on).
# ---------------------------------------------------------------------------

# gold_at_9 is 0-indexed in the lost-in-the-middle filenames -> position 10
# of 20 in 1-indexed terms.
NQ_GOLD_POSITION_FILE = "nq-open-20_total_documents_gold_at_9.jsonl.gz"
NQ_TOTAL_EXAMPLES = 2655  # size of this specific gold-position file

METRIC = "best_subspan_em"  # see metrics.py (imported from experiments/llmlingua2/evaluation/metrics.py)


# ---------------------------------------------------------------------------
# Sample size / protocol
# ---------------------------------------------------------------------------


@dataclass
class RunProtocol:
    name: str
    n_examples: int
    seed: Optional[int]
    label: str


# Since we're not on the paper's GPT-3.5 reader, we can't cross-check
# against their published NQ numbers -- that comparison isn't meaningful
# across different readers. Replaced with an internal-consistency check on
# OUR reader: full-context (no compression) vs. LongLLMLingua-compressed
# vs. zero-shot (no documents at all). Confirms full >= compressed >
# zero-shot with sane magnitudes before trusting the harness for the real
# method sweep -- if that ordering is violated, that's a wiring bug, cheap
# to catch on ~100 examples instead of expensive to discover later.
FIDELITY_CHECK_ROWS = ["full_context", "longllmlingua", "zero_shot"]
FIDELITY_CHECK_BUDGET = "2x"  # which budget the "longllmlingua" condition targets; the milder of the two, a reasonable default for a sanity check

FIDELITY_CHECK = RunProtocol(
    name="fidelity_check",
    n_examples=100,
    seed=20260816,  # fixed subset, distinct from METHOD_SWEEP's seed
    label=(
        "Internal-consistency check on a fixed random ~100-example subset, "
        "gold at position 10, reorder off, reader=Llama-3.1-8B-Instruct: "
        "full-context vs. LongLLMLingua-compressed (2x budget) vs. "
        "zero-shot. Confirms full >= compressed > zero-shot ordering with "
        "sane magnitudes -- this replaces cross-checking against the "
        "paper's own published (GPT-3.5-reader) numbers, which isn't a "
        "valid comparison now that we're on a different reader."
    ),
)

METHOD_SWEEP = RunProtocol(
    name="method_sweep",
    n_examples=400,  # midpoint of the agreed 300-500 range -- confirm before first run
    seed=20260815,  # fixed for reproducibility; the date this protocol was locked
    label=(
        "Fixed random subset (seeded) of the full set, gold at position 10, "
        "reorder off. All method/baseline rows: attention (1.5b, 7b), "
        "LongLLMLingua, bm25, sentbert."
    ),
)
