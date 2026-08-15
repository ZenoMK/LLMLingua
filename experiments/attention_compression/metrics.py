# Re-exports best_subspan_em from experiments/llmlingua2/evaluation/metrics.py
# (same repo) instead of duplicating its logic. That module isn't an
# installed package -- it's a sibling script directory, same convention as
# the rest of experiments/ -- so it isn't importable by dotted path; this
# adds it to sys.path the same way its own scripts assume a cwd/path setup
# to reach their neighbors.
#
# Cost of not duplicating: importing that module runs its other metric
# functions' top-level imports too (`evaluate`, `jieba`, `fuzzywuzzy`,
# `rouge`, `regex`), even though we only use best_subspan_em. Those are
# listed in requirements.txt.
import sys
from pathlib import Path

_LLMLINGUA2_EVAL_DIR = Path(__file__).resolve().parents[1] / "llmlingua2" / "evaluation"
if str(_LLMLINGUA2_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_LLMLINGUA2_EVAL_DIR))

from metrics import best_subspan_em  # noqa: E402

__all__ = ["best_subspan_em"]
