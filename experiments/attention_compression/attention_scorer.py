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
import inspect
from typing import Callable, List, Tuple

import nltk


def _bind_kwarg_true(sig: inspect.Signature, args: tuple, kwargs: dict, kwarg_name: str):
    """Pure Python, no torch: rebuilds a call via the target callable's
    own signature so `kwarg_name` ends up True in the reconstructed
    (args, kwargs), regardless of whether the ORIGINAL caller passed it
    positionally or by keyword. Used by compute_token_attention's forward
    pre-hook to force output_attentions=True for one specific decoder
    layer without needing to know or assume transformers' internal
    calling convention for that argument -- deliberately not hardcoded to
    a positional index, which isn't part of any public API contract and
    is exactly the kind of thing that silently breaks across versions.
    Testable on its own (test_attention_scorer.py) with a plain function
    signature and fake args/kwargs -- no model, no GPU, no torch."""
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    if kwarg_name in bound.arguments:
        bound.arguments[kwarg_name] = True
    return bound.args, bound.kwargs


def _looks_like_attention_weights(ndim: int, shape: Tuple[int, ...]) -> bool:
    """Pure Python, no torch: does an (ndim, shape) pair look like
    [batch=1, heads, seq_q, seq_k] attention weights? 4D, batch size 1,
    square in the last two dims. Takes plain dim/shape values (not a
    tensor) precisely so it's testable without torch -- used to identify
    which element of a decoder layer's returned tuple is the attention
    weights by shape rather than a hardcoded index, since exact tuple
    layout varies with use_cache/output_attentions combinations across
    transformers versions."""
    return ndim == 4 and len(shape) == 4 and shape[0] == 1 and shape[-1] == shape[-2]


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
    (sdpa, flash_attention_2) never materialize a full attention matrix at
    all, so there's nothing to capture from them regardless of technique.
    Not unit-tested here (needs a GPU and real weights); see
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
    all of them -- no reason to re-run the model once per candidate layer
    for identical work. Single-layer callers just pass a one-element list
    and index the result with [layer].

    MEMORY: does NOT use the model-level output_attentions=True. That
    flag has no per-layer selectivity -- every layer computes AND RETAINS
    its full [heads, seq, seq] attention matrix simultaneously until the
    whole forward pass returns (the framework accumulates them into a
    growing tuple as it goes). For a ~3,000-token context on a 32-layer
    model, that's ~1.15GB/layer x 32 =~ 37GB just for attention, on top
    of the model weights -- confirmed the hard way with a real
    CUDA OutOfMemoryError running the 7B scorer on an A10G (24GB), mid-
    sweep, well before even reaching the last layer. Critically, this
    isn't specific to requesting many layers: even layers=[27] (one
    layer) would hit the same wall, since the model computes and retains
    every OTHER layer's attention too regardless of which ones the caller
    asked for -- this would have resurfaced identically once the chosen
    layer got wired into the real method sweep, not just here.

    Fix: leave the model-level call at output_attentions=False (its
    default -- no accumulation happens, and each layer's transient
    attention tensor is freed as soon as that layer's own forward
    returns, so peak memory from attention is O(1) layer, not O(num
    layers)), and instead use forward hooks scoped to exactly the
    requested layers: a pre-hook forces output_attentions=True for only
    that one layer's own call (found via inspect.Signature.bind rather
    than assuming whether the caller passes it positionally or by
    keyword -- transformers' internal calling convention for this isn't
    part of its public API contract and isn't something to hardcode
    around), and a paired forward hook captures that layer's own
    attention-weights tensor directly from its return value (identified
    by shape -- 4D, batch-size-1, square in the last two dims -- rather
    than a hardcoded tuple index, which also varies with use_cache/
    output_attentions combinations across versions), reduces it to a
    per-context-token score immediately, and lets the original GPU tensor
    get freed. Peak extra memory becomes O(len(layers)), not O(num_hidden_layers)
    -- comfortably fits even the 7B model on an A10G. use_cache=False on
    the call itself since this is a one-shot scoring pass, not
    generation -- no KV cache to reuse afterward.

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

    def _force_output_attentions_true(module, args, kwargs):
        return _bind_kwarg_true(inspect.signature(module.forward), args, kwargs, "output_attentions")

    token_scores_by_layer = {}

    def _make_capture_hook(layer_idx):
        def hook(module, args, output):
            items = output if isinstance(output, tuple) else (output,)
            for item in items:
                if torch.is_tensor(item) and _looks_like_attention_weights(item.dim(), tuple(item.shape)):
                    query_to_context = item[0, :, n_context:, :n_context]  # [heads, n_query, n_context]
                    token_scores_by_layer[layer_idx] = (
                        query_to_context.mean(dim=(0, 1)).float().cpu().tolist()  # mean over heads AND query positions
                    )
                    break

        return hook

    decoder_layers = model.model.layers  # LlamaModel/Qwen2Model convention -- both scorer backbones use it
    handles = []
    for layer_idx in layers:
        handles.append(decoder_layers[layer_idx].register_forward_pre_hook(_force_output_attentions_true, with_kwargs=True))
        handles.append(decoder_layers[layer_idx].register_forward_hook(_make_capture_hook(layer_idx)))

    try:
        with torch.no_grad():
            model(input_ids, output_attentions=False, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = set(layers) - set(token_scores_by_layer)
    if missing:
        raise RuntimeError(
            f"Failed to capture attention weights for layers {sorted(missing)} -- "
            'the model likely wasn\'t loaded with attn_implementation="eager" '
            "(sdpa/flash_attention_2 never materialize a full attention matrix "
            "to capture, hook or no hook)."
        )

    return context, offset_mappings, doc_token_base, token_scores_by_layer
