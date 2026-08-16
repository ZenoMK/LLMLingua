# Token budget accounting in the READER's tokenizer -- per the locked
# settings, this is the tokenizer that determines the reader's actual
# context/generation cost and is what "achieved vs. target ratio" is
# measured against. The reader is a local HF model now (config.py), so this
# uses transformers' tokenizer, not tiktoken.
from typing import Optional

from transformers import AutoTokenizer

import config

_tokenizer = None


def get_reader_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(config.READER_MODEL)
    return _tokenizer


def count_reader_tokens(text: str) -> int:
    return len(get_reader_tokenizer().encode(text, add_special_tokens=False))


def budget_report(budget_name: Optional[str], original_tokens: int, compressed_tokens: int) -> dict:
    """achieved_ratio should land close to target_ratio; report both, don't
    silently assume compress_prompt hit the target exactly (it targets, it
    doesn't guarantee, per its own docstring). budget_name is None for the
    full_context/zero_shot pseudo-conditions, which don't target a budget
    -- target_token/target_ratio are None in that case."""
    target_token = config.TOKEN_BUDGETS[budget_name] if budget_name else None
    return {
        "budget_name": budget_name,
        "target_token": target_token,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "achieved_ratio": (
            original_tokens / compressed_tokens if compressed_tokens else float("inf")
        ),
        "target_ratio": (original_tokens / target_token if target_token else None),
    }
