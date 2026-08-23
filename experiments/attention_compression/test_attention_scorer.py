# Unit tests for attention_scorer.py's model-INDEPENDENT functions
# (sentences_with_scores, select_sentences, plus the two pure helpers
# compute_token_attention's hooks rely on) -- pure Python, synthetic
# inputs, no GPU/model/tokenizer needed. compute_token_attention's actual
# forward pass isn't covered here; it needs a real model and gets
# exercised for real in layer_sweep.py instead.
#
# Run directly: python test_attention_scorer.py
import inspect
import unittest

from attention_scorer import (
    _bind_kwarg_true,
    _looks_like_attention_weights,
    select_sentences,
    sentences_with_scores,
)


def _word_offsets(text: str):
    """Tiny fake 'tokenizer' for these tests: one token per whitespace-
    separated word, offset_mapping in the same (start, end) shape real
    tokenizers return. Deliberately not a real tokenizer -- this dev
    machine's local torch install is broken and can't load one anyway,
    and the aggregation logic under test doesn't care what produced the
    offsets, only that they're correct."""
    offsets = []
    pos = 0
    for word in text.split(" "):
        start = text.index(word, pos)
        end = start + len(word)
        offsets.append((start, end))
        pos = end
    return offsets


class SentencesWithScoresTest(unittest.TestCase):
    def test_aggregates_token_scores_to_sentence_mean(self):
        doc = "The cat sat. The dog ran fast."
        offsets = _word_offsets(doc)
        # tokens: "The"(0) "cat"(1) "sat."(2) | "The"(3) "dog"(4) "ran"(5) "fast."(6)
        scores = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0]
        result = sentences_with_scores(
            context=[doc],
            context_offset_mappings=[offsets],
            context_doc_token_base=[0],
            token_scores=scores,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "The cat sat.")
        self.assertAlmostEqual(result[0]["score"], (1.0 + 2.0 + 3.0) / 3)
        self.assertEqual(result[1]["text"], "The dog ran fast.")
        self.assertAlmostEqual(result[1]["score"], (10.0 + 20.0 + 30.0 + 40.0) / 4)

    def test_uses_mean_not_sum_so_longer_sentences_arent_favored(self):
        # A short high-scoring sentence should beat a long low-scoring one
        # on mean, even though the long one has a higher token-score sum.
        doc = "Wow. This one has many low value words in it okay."
        offsets = _word_offsets(doc)
        scores = [100.0] + [1.0] * 10  # "Wow."=100, remaining 10 tokens=1 each (sum=10 < 100)
        result = sentences_with_scores([doc], [offsets], [0], scores)
        by_text = {r["text"]: r["score"] for r in result}
        self.assertEqual(by_text["Wow."], 100.0)
        self.assertAlmostEqual(by_text["This one has many low value words in it okay."], 1.0)

    def test_multiple_documents_use_correct_token_base(self):
        doc0 = "Alpha one."
        doc1 = "Beta two."
        offsets0 = _word_offsets(doc0)  # 2 tokens
        offsets1 = _word_offsets(doc1)  # 2 tokens
        scores = [5.0, 5.0, 100.0, 200.0]  # doc0's 2 tokens, then doc1's 2 tokens
        result = sentences_with_scores(
            context=[doc0, doc1],
            context_offset_mappings=[offsets0, offsets1],
            context_doc_token_base=[0, 2],
            token_scores=scores,
        )
        self.assertEqual([r["doc_idx"] for r in result], [0, 1])
        self.assertAlmostEqual(result[0]["score"], 5.0)
        self.assertAlmostEqual(result[1]["score"], 150.0)


class SelectSentencesTest(unittest.TestCase):
    def test_picks_highest_scoring_sentence_first(self):
        scored = [
            {"doc_idx": 0, "text": "low score sentence one.", "score": 1.0},
            {"doc_idx": 0, "text": "high score sentence two.", "score": 9.0},  # len 24
            {"doc_idx": 0, "text": "medium score sentence three.", "score": 5.0},
        ]
        # count_tokens = character count. Budget of 24 is hit exactly by
        # the highest-scoring sentence alone -- nothing else gets pulled in.
        result = select_sentences(scored, target_token=24, count_tokens=len)
        self.assertEqual(result, "high score sentence two.")

    def test_reconstructs_in_original_order_not_score_order(self):
        scored = [
            {"doc_idx": 0, "text": "first.", "score": 1.0},  # low score, appears first
            {"doc_idx": 0, "text": "second.", "score": 9.0},  # high score, appears second
        ]
        result = select_sentences(scored, target_token=1000, count_tokens=len)  # budget big enough for both
        self.assertEqual(result, "first. second.")  # original order preserved, not score order

    def test_separates_documents_with_newline(self):
        scored = [
            {"doc_idx": 0, "text": "doc zero.", "score": 5.0},
            {"doc_idx": 1, "text": "doc one.", "score": 5.0},
        ]
        result = select_sentences(scored, target_token=1000, count_tokens=len)
        self.assertEqual(result, "doc zero.\ndoc one.")

    def test_overshoots_rather_than_undershoots(self):
        # target=10, best sentence alone is 15 chars -- still gets included
        # (the project's convention: cross the line, don't stop short of it).
        scored = [{"doc_idx": 0, "text": "fifteen chars!!", "score": 1.0}]
        result = select_sentences(scored, target_token=10, count_tokens=len)
        self.assertEqual(result, "fifteen chars!!")


# Fake decoder-layer-shaped forward, exercising the exact ambiguity
# _bind_kwarg_true exists to handle: does the caller pass output_attentions
# positionally or by keyword? Both are legal Python and transformers'
# internal calling convention for this isn't part of its public API
# contract -- these tests don't assume either way, they check both.
def _fake_layer_forward(hidden_states, attention_mask=None, position_ids=None, output_attentions=False, use_cache=True):
    return {"output_attentions_received": output_attentions}


class BindKwargTrueTest(unittest.TestCase):
    def test_overrides_keyword_arg(self):
        sig = inspect.signature(_fake_layer_forward)
        args, kwargs = ("hidden",), {"output_attentions": False}
        new_args, new_kwargs = _bind_kwarg_true(sig, args, kwargs, "output_attentions")
        self.assertTrue(_fake_layer_forward(*new_args, **new_kwargs)["output_attentions_received"])

    def test_overrides_positional_arg(self):
        sig = inspect.signature(_fake_layer_forward)
        # output_attentions in the 4th positional slot, per _fake_layer_forward's signature
        args, kwargs = ("hidden", None, None, False), {}
        new_args, new_kwargs = _bind_kwarg_true(sig, args, kwargs, "output_attentions")
        self.assertTrue(_fake_layer_forward(*new_args, **new_kwargs)["output_attentions_received"])

    def test_overrides_when_not_passed_at_all(self):
        # output_attentions omitted entirely -- apply_defaults() should
        # still surface it (at its default, False) so it's there to override.
        sig = inspect.signature(_fake_layer_forward)
        args, kwargs = ("hidden",), {}
        new_args, new_kwargs = _bind_kwarg_true(sig, args, kwargs, "output_attentions")
        self.assertTrue(_fake_layer_forward(*new_args, **new_kwargs)["output_attentions_received"])

    def test_noop_when_signature_lacks_the_kwarg(self):
        def other_fn(x, y=1):
            return (x, y)

        sig = inspect.signature(other_fn)
        args, kwargs = (5,), {}
        new_args, new_kwargs = _bind_kwarg_true(sig, args, kwargs, "output_attentions")
        self.assertEqual(other_fn(*new_args, **new_kwargs), (5, 1))  # unchanged, no crash


class LooksLikeAttentionWeightsTest(unittest.TestCase):
    def test_matches_realistic_attention_shape(self):
        self.assertTrue(_looks_like_attention_weights(4, (1, 32, 2800, 2800)))

    def test_rejects_3d_hidden_states_shape(self):
        self.assertFalse(_looks_like_attention_weights(3, (1, 2800, 4096)))

    def test_rejects_batch_size_other_than_1(self):
        self.assertFalse(_looks_like_attention_weights(4, (2, 32, 2800, 2800)))

    def test_rejects_non_square_last_two_dims(self):
        # e.g. a KV-cache-shaped tensor: [batch, heads, seq, head_dim]
        self.assertFalse(_looks_like_attention_weights(4, (1, 32, 2800, 128)))


if __name__ == "__main__":
    unittest.main()
