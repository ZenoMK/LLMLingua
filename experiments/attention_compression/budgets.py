# Token budget accounting in the READER's tokenizer (tiktoken), not the
# scorer's -- per the locked settings, this is the tokenizer that determines
# API cost and is what "achieved vs. target ratio" is measured against.
import tiktoken

from config import READER_TOKENIZER_ENCODING_MODEL, TOKEN_BUDGETS

_encoding = None


def get_reader_encoding():
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.encoding_for_model(READER_TOKENIZER_ENCODING_MODEL)
    return _encoding


def count_reader_tokens(text: str) -> int:
    return len(get_reader_encoding().encode(text))


def budget_report(budget_name: str, original_tokens: int, compressed_tokens: int) -> dict:
    """achieved_ratio should land close to target_ratio; report both, don't
    silently assume compress_prompt hit the target exactly (it targets, it
    doesn't guarantee, per its own docstring)."""
    target_token = TOKEN_BUDGETS[budget_name]
    return {
        "budget_name": budget_name,
        "target_token": target_token,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "achieved_ratio": (
            original_tokens / compressed_tokens if compressed_tokens else float("inf")
        ),
        "target_ratio": original_tokens / target_token if target_token else float("inf"),
    }
