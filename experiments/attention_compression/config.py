# Locked experiment settings -- attention scorer vs. LongLLMLingua.
#
# Every script in this directory imports from here instead of hard-coding
# these values, so a config change shows up as one diff instead of scattered
# edits across scripts. Provenance for each decision is in NOTES.md and in
# the PR/commit history; changes to this file go through a PR like anything
# else.
#
# Locked by user review on 2026-08-15, in response to NOTES.md's four open
# questions.

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Reader: the model that answers the (compressed) prompt. This is what "API
# cost" means throughout the project, and what achieved-vs-target token
# ratios are measured against.
# ---------------------------------------------------------------------------

READER_MODEL = "gpt-3.5-turbo-0613"
READER_TEMPERATURE = 0.0
READER_MAX_OUTPUT_TOKENS = 100  # matches examples/RAG.ipynb / lost-in-the-middle convention
READER_TOKENIZER_ENCODING_MODEL = "gpt-3.5-turbo-0613"  # tiktoken.encoding_for_model() argument

# gpt-3.5-turbo-0613 is an old snapshot and may be deprecated on any given
# OpenAI account. MUST run check_reader_availability.py (cheap: models.list,
# not billed) before any run that depends on this model. If it's gone: stop
# and ask before substituting a newer snapshot -- accepting drift from the
# paper's published numbers is the user's call, not an automatic fallback.
# The -16k-0613 variant is deliberately not used as a fallback here: the
# paper only reaches for it when the prompt exceeds 4k tokens, and our
# compressed NQ prompts (budgets below) are always well under that.


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


# ---------------------------------------------------------------------------
# Compression pipeline settings, per method row
# ---------------------------------------------------------------------------

# CRITICAL: reorder is OFF everywhere in this project. Our attention method
# has no document-reordering step, so LongLLMLingua must also run without
# one for an apples-to-apples comparison -- otherwise we'd be comparing our
# no-reorder numbers against their with-reorder numbers, which isn't the
# paper's own Table 1 per-position (no-reorder) comparison this project is
# anchored on. "original" is llmlingua's literal "restore input order, don't
# reorder" value (see control_context_budget in prompt_compressor.py).
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
    # default (off) -- revisit only if the Step 2 fidelity check doesn't
    # land near the paper's Table 1 position-10 number.
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
# Token budgets -- measured in the READER's tokenizer (tiktoken), not the
# scorer's. These are absolute target_token values, matching the paper's NQ
# table's 2x and 4x compression conditions.
# ---------------------------------------------------------------------------

TOKEN_BUDGETS = {
    "2x": 3000,
    "4x": 2000,
}


# ---------------------------------------------------------------------------
# Benchmark: NaturalQuestions multi-doc QA (lost-in-the-middle), gold
# document at position 10 of 20 -- the hardest lost-in-the-middle case, and
# where the paper's 21.4% headline number is measured.
# ---------------------------------------------------------------------------

# gold_at_9 is 0-indexed in the lost-in-the-middle filenames -> position 10
# of 20 in 1-indexed terms.
NQ_GOLD_POSITION_FILE = "nq-open-20_total_documents_gold_at_9.jsonl.gz"
NQ_TOTAL_EXAMPLES = 2655  # size of this specific gold-position file

METRIC = "best_subspan_em"  # see metrics.py (vendored, see that file's docstring)


# ---------------------------------------------------------------------------
# Sample size / protocol
# ---------------------------------------------------------------------------


@dataclass
class RunProtocol:
    name: str
    n_examples: int
    seed: Optional[int]
    label: str


FIDELITY_CHECK = RunProtocol(
    name="fidelity_check",
    n_examples=NQ_TOTAL_EXAMPLES,
    seed=None,
    label=(
        "Full 2,655-example set, gold at position 10, reorder off. "
        "Reproduction sanity check: LongLLMLingua only, against the "
        "paper's Table 1 position-10 column (not the with-reorder headline)."
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
