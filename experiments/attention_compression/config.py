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
# depth candidates, per model independently -- the best layer is
# model-specific, so no layer gets reused across the two sizes. Budget
# matches FIDELITY_CHECK's choice of "2x" (the milder of the two budgets,
# a reasonable default when the point is comparing layers to each other,
# not stress-testing a tight budget).
#
# Deliberately larger than the original brief's "~5 examples" suggestion:
# a discrete pick among 4 candidate layers, from only 5 examples, risks
# locking in whichever layer got lucky rather than one that's genuinely
# better -- there's no later "recheck at full scale" step for this choice
# the way the fidelity check had (n=10 there was just a smoke test for
# the harness, not the thing being decided). Plan: smoke-test the
# pipeline at n=5 (layer_sweep.py's/modal_layer_sweep.py's --limit
# override) first, then run the real decision at this value if that
# works -- see FINDINGS.md.
LAYER_SWEEP_N_EXAMPLES = 100
LAYER_SWEEP_BUDGET = "2x"
LAYER_SWEEP_SEED = 20260823  # distinct from FIDELITY_CHECK's/METHOD_SWEEP's seeds -- independent draws

# From the real n=100 sweep, RE-RUN on 2026-08-25 under the corrected
# genuine-2x budget (see FINDINGS.md) -- SUPERSEDES the original
# 2026-08-23 sweep, which ran under a budget bug (target_token=3000, a
# ~1.15x squeeze on a ~2,945-token original, barely compressing
# anything). That run found all 4 candidate layers tied within 0.01 mean
# EM and concluded "layer choice barely matters." Re-running the exact
# same sweep under a REAL ~2x budget overturned that: layers now spread
# 0.65-0.73 (1.5b) and 0.52-0.70 (7b), and 0/100 examples produce an
# identical compressed prompt across layers for either model (vs.
# 83-84% identical before). The "barely matters" conclusion was an
# artifact of the loose budget leaving almost nothing to cut -- under a
# real budget tight enough to force real selection, which layer's
# attention pattern ranks the sentences genuinely changes what survives,
# and that has real downstream effect on accuracy. Full writeup and
# per-layer numbers in FINDINGS.md's 2026-08-25 entry.
ATTENTION_SCORER_LAYERS = {
    "1.5b": 17,  # 0.73 mean EM, clear best of [13, 17, 20, 27] = [0.65, 0.73, 0.65, 0.70]
    "7b": 20,  # 0.70 mean EM, clear best of [15, 20, 23, 31] = [0.69, 0.70, 0.65, 0.52]
}

# Row name -> model size, for compress_job.py's row dispatch. Attention rows
# don't go through PromptCompressor/ROW_COMPRESSOR_MODEL like the other rows
# -- they need a real (model, tokenizer, layer) loaded via
# ATTENTION_SCORER_MODELS/ATTENTION_SCORER_LAYERS instead, so compress_job.py
# checks membership in this dict to route to that separate loading path.
ATTENTION_ROW_MODEL_SIZE = {
    "attention_1.5b": "1.5b",
    "attention_7b": "7b",
}
ATTENTION_ROWS = list(ATTENTION_ROW_MODEL_SIZE)


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

# All method_sweep rows: our method (both scorer sizes) plus every baseline
# -- see METHOD_SWEEP's label below. ATTENTION_ROWS first so a real run's
# progress output shows our own method before the baselines, not because
# order affects the results.
METHOD_SWEEP_ROWS = ATTENTION_ROWS + BASELINE_ROWS


# ---------------------------------------------------------------------------
# Compression rates -- measured in the READER's tokenizer (its own HF
# tokenizer now, not tiktoken -- see budgets.py). target_token is computed
# fresh per example as round(reader_tokens(original) * rate), NOT a fixed
# absolute number -- see budgets.compute_target_token().
#
# These rates match the paper's actual NaturalQuestions table (Table 1),
# verified directly against arxiv 2310.06839: Original Prompt = 2,946
# tokens; under the "2x constraint" LongLLMLingua achieves ~1,429 tokens
# (~0.49x, i.e. genuinely ~half); under the "4x constraint" it achieves
# ~748 tokens (~0.25x, i.e. genuinely ~quarter). Our own measured average
# original length (~2,945 tokens) matches the paper's almost exactly,
# confirming the rest of the dataset/tokenizer setup.
#
# {"2x": 3000, "4x": 2000} previously lived here -- those are Table 2's
# (LongBench) absolute token constraints, not Table 1's (NaturalQuestions)
# ratios. LongBench mixes wildly different task lengths so the paper uses
# an absolute number there instead of a ratio; NQ's prompts are all ~20
# documents so a ratio is the right and paper-faithful unit. See
# FINDINGS.md for the full misattribution writeup.
# ---------------------------------------------------------------------------

COMPRESSION_RATES = {
    "2x": 0.5,
    "4x": 0.25,
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
