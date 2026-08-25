# Compression wrappers, one per row in the comparison table. Each function
# takes an already-constructed llmlingua.PromptCompressor and returns
# compress_prompt's normal result dict unchanged. All method-specific flags
# come from config.py, so every row's settings are traceable to one place
# (see NOTES.md for what each flag does and why it's set that way).
#
# compress_attention is the one exception: it takes a (model, tokenizer,
# layer) instead of a PromptCompressor, since it's our own method, not a
# llmlingua rank_method -- see its own docstring.
from typing import List

from llmlingua import PromptCompressor

import attention_scorer
import config
from budgets import count_reader_tokens


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


def join_compressed_prompt(instruction: str, compressed_body: str, question: str) -> str:
    """Matches how compress_prompt itself joins instruction/body/question
    (prompt_compressor.py: "\\n\\n".join([instruction, compressed_prompt,
    question])) -- for rows (compress_attention here, layer_sweep.py's own
    candidate compression) that only produce a compressed body string, not
    compress_prompt's full return dict, so they still get the same final
    assembly every other row gets, for a fair comparison."""
    return "\n\n".join([instruction, compressed_body, question])


def compress_attention(
    model,
    tokenizer,
    layer: int,
    context: List[str],
    instruction: str,
    question: str,
    target_token: int,
) -> dict:
    """Our method: one-pass query-to-context attention scoring, sentence-
    level greedy selection, at a single fixed layer (config.
    ATTENTION_SCORER_LAYERS, picked by layer_sweep.py -- see FINDINGS.md).
    Needs a real model/tokenizer already loaded with
    attn_implementation="eager" (see attention_scorer.compute_token_attention
    for why). Not a llmlingua rank_method, so this doesn't take a
    PromptCompressor like the other rows -- compress_job.py dispatches to
    this directly rather than via ROW_FUNCS."""
    _, offsets, bases, scores_by_layer = attention_scorer.compute_token_attention(
        model, tokenizer, context, question, [layer]
    )
    sentences = attention_scorer.sentences_with_scores(context, offsets, bases, scores_by_layer[layer])
    compressed_body = attention_scorer.select_sentences(sentences, target_token, count_reader_tokens)
    return {"compressed_prompt": join_compressed_prompt(instruction, compressed_body, question)}


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
