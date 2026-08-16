# Compression wrappers, one per row in the comparison table. Each function
# takes an already-constructed llmlingua.PromptCompressor and returns
# compress_prompt's normal result dict unchanged. All method-specific flags
# come from config.py, so every row's settings are traceable to one place
# (see NOTES.md for what each flag does and why it's set that way).
from typing import List

from llmlingua import PromptCompressor

import config


def compress_longllmlingua(
    compressor: PromptCompressor,
    context: List[str],
    instruction: str,
    question: str,
    target_token: int,
) -> dict:
    """Full LongLLMLingua: coarse context-level ranking + fine token-level
    contrastive-perplexity compression (condition_compare=True), reorder
    OFF. This is the Step 2 reproduction row."""
    return compressor.compress_prompt(
        context,
        instruction=instruction,
        question=question,
        target_token=target_token,
        **config.LONGLLMLINGUA_KWARGS,
    )


def compress_bm25(
    compressor: PromptCompressor,
    context: List[str],
    instruction: str,
    question: str,
    target_token: int,
) -> dict:
    """Sentence-level BM25 selection -- same granularity as our attention
    method, reached via rank_method, no reimplementation."""
    return compressor.compress_prompt(
        context,
        instruction=instruction,
        question=question,
        target_token=target_token,
        **config.BM25_BASELINE_KWARGS,
    )


def compress_sentbert(
    compressor: PromptCompressor,
    context: List[str],
    instruction: str,
    question: str,
    target_token: int,
) -> dict:
    """Sentence-level embedding (multi-qa-mpnet-base-dot-v1) selection."""
    return compressor.compress_prompt(
        context,
        instruction=instruction,
        question=question,
        target_token=target_token,
        **config.SENTBERT_BASELINE_KWARGS,
    )


def compress_attention(*args, **kwargs):
    """Our method: query-to-context attention scorer, sentence-level greedy
    selection. Not implemented here -- lands in the Step 3 PR (separate
    scorer module + per-model layer sweep). Not registered in ROW_FUNCS
    yet: its real signature will need a model-size/layer selector that the
    other rows don't take, so compress_job.py will dispatch to it directly
    once it exists rather than forcing it into this shape prematurely."""
    raise NotImplementedError("Attention scorer lands in the Step 3 PR.")


ROW_FUNCS = {
    "longllmlingua": compress_longllmlingua,
    "bm25": compress_bm25,
    "sentbert": compress_sentbert,
}


def compress_full_context(context: List[str], instruction: str, question: str) -> str:
    """No compression -- the full uncompressed prompt, joined the same way
    compress_prompt joins its own output (instruction / body / question).
    Upper-bound condition for the fidelity check (config.FIDELITY_CHECK).
    Takes no compressor -- there's nothing to score or drop."""
    return "\n\n".join([instruction, "\n".join(context), question])


def compress_zero_shot(instruction: str, question: str) -> str:
    """No documents at all -- tests the reader's parametric knowledge with
    no retrieval. Lower-bound condition for the fidelity check. Takes no
    compressor."""
    return "\n\n".join([instruction, question])
