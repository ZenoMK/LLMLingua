# Vendored from experiments/llmlingua2/evaluation/metrics.py (same repo) --
# just best_subspan_em and the SQuAD-style normalizer it depends on.
#
# Copied rather than imported: that module's top-level imports (`evaluate`,
# `jieba`, `fuzzywuzzy`, `rouge`) pull in heavy deps needed by its other
# metric functions (qa_f1_score, rouge_score, ...) but not by this one. The
# original uses the third-party `regex` module for the article-stripping
# regex; swapped for stdlib `re` here since the pattern (`\b(a|an|the)\b`)
# doesn't use anything `re` can't do.
import re
import string
from typing import List


def normalize_answer(s: str) -> str:
    """Normalization from the SQuAD evaluation script."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0
