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
    scorer module + per-model layer sweep), kept as a stub so run.py's row
    dispatch table is already shaped correctly."""
    raise NotImplementedError("Attention scorer lands in the Step 3 PR.")


ROW_FUNCS = {
    "longllmlingua": compress_longllmlingua,
    "bm25": compress_bm25,
    "sentbert": compress_sentbert,
    "attention": compress_attention,
}
