# Step 3's method: query-to-context attention scoring. Single forward pass
# over [context, query], read query-token -> context-token attention at a
# chosen layer, aggregate to sentence-level scores, greedily select
# sentences by score until the reader-tokenizer budget is hit, reconstruct
# in ORIGINAL order (no reordering, matching this project's rule and
# llmlingua's own bm25/sentbert convention -- see config.py's
# REORDER_CONTEXT comment).
#
# Deliberately bare, per the project brief: no iteration, no coarse
# ranking, no dynamic compression ratio, no subsequence recovery. Sentence
# selection is greedy-by-score once; the surviving sentences go straight
# to the reader.
#
# Split into a model-dependent half (run the forward pass, needs a real
# model/tokenizer/GPU) and a model-independent half (turn raw per-token
# scores into a compressed prompt, pure Python). This isn't just tidiness
# -- it's what makes the aggregation/selection logic unit-testable without
# a GPU at all (see test_attention_scorer.py), which matters given this
# dev machine's local torch install is broken and can't run a real forward
# pass locally regardless.
#
# Which layer to read attention from, per scorer model, is a separate
# question -- see layer_sweep.py. This module takes layers as a plain
# argument; it doesn't know or care how they were chosen.
#
# Cacheability: because context comes first and query second, the
# context's KV cache is, in principle, reusable across different queries
# against the same context (real forward-pass code, not yet written here,
# would compute it once and pass it in) -- one cacheable pass, contrasted
# against LongLLMLingua's two uncacheable ones (condition_compare's `loss`
# vs `self_loss` passes process genuinely different token sequences and
# can't share a cache). This benchmark's NQ examples each have a unique
# context, so that reuse isn't exercised by this harness's data -- it's a
# structural property of the method, not something this specific
# experiment demonstrates operationally.
from typing import Callable, List, Tuple

import nltk


def _sentence_spans(doc_text: str) -> List[Tuple[str, int, int]]:
    """(sentence_text, char_start, char_end) triples within doc_text, via
    nltk.sent_tokenize -- the same tokenizer llmlingua's own
    control_sentence_budget uses for the bm25/sentbert baselines, so
    sentence boundaries are comparable across rows, not an artifact of a
    different splitter. Doesn't handle sentences nltk fails to locate by
    plain substring search (shouldn't happen -- nltk's output is a
    partition of the input -- but skips rather than crashes if it does)."""
    spans = []
    pos = 0
    for s in nltk.sent_tokenize(doc_text):
        start = doc_text.find(s, pos)
        if start == -1:
            continue
        end = start + len(s)
        spans.append((s, start, end))
        pos = end
    return spans


def sentences_with_scores(
    context: List[str],
    context_offset_mappings: List[List[Tuple[int, int]]],
    context_doc_token_base: List[int],
    token_scores: List[float],
) -> List[dict]:
    """Model-independent: aggregates already-computed per-token attention
    scores into per-sentence scores. Pure Python -- no model, no GPU.

    Args:
        context: the documents, same as compress_prompt's `context` arg.
        context_offset_mappings: per-document list of (char_start,
            char_end) for each of that document's tokens, in the scorer's
            tokenizer -- e.g. from
            tokenizer(doc, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"].
        context_doc_token_base: for each document, the index into
            `token_scores` where that document's tokens start (documents
            are tokenized independently, then concatenated for the
            forward pass -- this is where each one landed).
        token_scores: one float per context token (query-attention to
            that token, already averaged over heads/layer/query
            positions), aligned with the concatenation implied by
            `context_doc_token_base`.

    Returns one dict per sentence, across all documents, in ORIGINAL
    (document, sentence) order -- not sorted by score. Sentence score is
    the MEAN of its tokens' scores (not sum), matching how llmlingua's own
    condition_compare-free sentence-level scoring works (per-sentence
    perplexity is also a mean, via loss.mean() at granularity="sentence")
    -- sum would just reward longer sentences for having more tokens."""
    results = []
    for doc_idx, doc in enumerate(context):
        offsets = context_offset_mappings[doc_idx]
        base = context_doc_token_base[doc_idx]
        for sent_text, s_start, s_end in _sentence_spans(doc):
            sent_scores = [
                token_scores[base + i]
                for i, (tok_start, tok_end) in enumerate(offsets)
                if tok_end > tok_start and tok_start >= s_start and tok_end <= s_end
            ]
            score = sum(sent_scores) / len(sent_scores) if sent_scores else 0.0
            results.append({"doc_idx": doc_idx, "text": sent_text, "score": score})
    return results


def select_sentences(
    scored_sentences: List[dict],
    target_token: int,
    count_tokens: Callable[[str], int],
) -> str:
    """Model-independent: greedy selection + original-order reconstruction.
    Pure Python.

    Ranks sentences by score (highest first), adding them to the kept set
    until the READER-tokenizer token count of the kept text would meet or
    exceed `target_token` -- the sentence that crosses the line is
    included, so this can overshoot by at most one sentence rather than
    undershoot; same convention llmlingua's own control_sentence_budget
    uses for its bm25/sentbert rows. `count_tokens` is injected rather
    than hardcoded so callers use the reader's tokenizer (budgets.
    count_reader_tokens), matching this project's budget-matching rule --
    NOT the scorer's own tokenizer.

    Reconstructs kept sentences in their ORIGINAL document/sentence order
    (not score-rank order) -- no reordering anywhere in this project.
    Sentences within a document are joined with a single space; this
    doesn't perfectly reproduce the original's exact whitespace, which is
    fine for "bare selection, no reconstruction needed" per the project
    brief -- the point is coherent surviving sentences, not a byte-exact
    diff of the original.

    Returns the compressed context as one string (documents joined by
    "\\n", matching compress_full_context's convention) -- callers still
    need to join this with instruction/question themselves, same as every
    other row."""
    order = sorted(range(len(scored_sentences)), key=lambda i: scored_sentences[i]["score"], reverse=True)
    keep = [False] * len(scored_sentences)
    used_tokens = 0
    for i in order:
        keep[i] = True
        used_tokens += count_tokens(scored_sentences[i]["text"])
        if used_tokens >= target_token:
            break

    kept_by_doc = {}
    for i, s in enumerate(scored_sentences):
        if keep[i]:
            kept_by_doc.setdefault(s["doc_idx"], []).append(s["text"])
    return "\n".join(" ".join(kept_by_doc[doc_idx]) for doc_idx in sorted(kept_by_doc))


def compute_token_attention(model, tokenizer, context: List[str], question: str, layers: List[int]):
    """Model-dependent: the actual forward pass. Requires a real model
    loaded with attn_implementation="eager" -- optimized attention paths
    (sdpa, flash_attention_2) don't return attention weights at all, and
    output_attentions=True silently gives back None for them instead of
    erroring. Not unit-tested here (needs a GPU and real weights); see
    layer_sweep.py for where this gets exercised for real, and
    test_attention_scorer.py for how the two functions above are tested
    with synthetic inputs instead.

    Layout: [context tokens][question tokens] -- context first is what
    makes the question tokens' attention run back OVER the context at all
    (causal masking: a token can only attend to itself and earlier
    tokens). No chat template -- this reads the model's raw attention over
    content, not a template-mediated one, and we don't need it to
    generate anything.

    Takes a LIST of layers, not one, and does a single forward pass for
    all of them: output_attentions=True already returns every layer's
    attention weights in one pass, so there's no reason to re-run the
    model once per candidate layer. This matters concretely for
    layer_sweep.py, which evaluates several candidate layers per example
    -- re-running the forward pass per layer would be purely-wasted
    N-times-redundant GPU cost for identical work. Single-layer callers
    just pass a one-element list and index the result with [layer].

    Returns (context, per-document offset mappings, per-document token
    base indices, {layer: token_scores}) -- everything
    sentences_with_scores() needs, so callers typically do:
        _, offsets, bases, scores_by_layer = compute_token_attention(..., layers=[7])
        sentences = sentences_with_scores(context, offsets, bases, scores_by_layer[7])
    """
    import torch

    context_ids_list, offset_mappings, doc_token_base = [], [], []
    running_len = 0
    for doc in context:
        enc = tokenizer(doc, add_special_tokens=False, return_offsets_mapping=True)
        context_ids_list.append(enc["input_ids"])
        offset_mappings.append(enc["offset_mapping"])
        doc_token_base.append(running_len)
        running_len += len(enc["input_ids"])

    context_ids = [tid for doc_ids in context_ids_list for tid in doc_ids]
    question_ids = tokenizer("Question: " + question, add_special_tokens=False)["input_ids"]
    n_context = len(context_ids)

    input_ids = torch.tensor([context_ids + question_ids], device=model.device)
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)

    token_scores_by_layer = {}
    for layer in layers:
        layer_attn = out.attentions[layer]
        if layer_attn is None:
            raise RuntimeError(
                "model.attentions is None -- the model wasn't loaded with "
                'attn_implementation="eager" (sdpa/flash_attention_2 don\'t '
                "return attention weights)."
            )
        query_to_context = layer_attn[0, :, n_context:, :n_context]  # [heads, n_query, n_context]
        token_scores_by_layer[layer] = query_to_context.mean(dim=(0, 1)).float().cpu().tolist()  # mean over heads AND query positions

    return context, offset_mappings, doc_token_base, token_scores_by_layer
