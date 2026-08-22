# Re-exports best_subspan_em from experiments/llmlingua2/evaluation/metrics.py
# (same repo) instead of duplicating its logic. Loaded via importlib by
# file path, under a private module name, rather than a sys.path +
# `import metrics` trick -- that upstream file is ALSO named metrics.py, so
# a bare `import metrics` from in here collides with THIS module's own
# name in sys.modules (Python registers a module there before its body
# finishes running, for circular-import handling) and raises
# "cannot import name 'best_subspan_em' from partially initialized module
# 'metrics'". Found this the hard way on the first real run -- would have
# been caught for free by a plain local `import metrics` test before ever
# touching Modal, in hindsight.
#
# Cost of not duplicating the logic: that module's other metric functions'
# top-level imports (`evaluate`, `jieba`, `fuzzywuzzy`, `rouge`, `regex`)
# still run when this loads, even though we only use best_subspan_em.
# Those are listed in requirements.txt.
import importlib.util
from pathlib import Path

_UPSTREAM_PATH = Path(__file__).resolve().parents[1] / "llmlingua2" / "evaluation" / "metrics.py"
_spec = importlib.util.spec_from_file_location("_llmlingua2_evaluation_metrics", _UPSTREAM_PATH)
_upstream = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_upstream)

best_subspan_em = _upstream.best_subspan_em

__all__ = ["best_subspan_em"]
