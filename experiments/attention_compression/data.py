# NaturalQuestions multi-doc QA data loading (the "lost in the middle"
# benchmark). Mirrors examples/RAG.ipynb's setup exactly, so we inherit its
# already-proven prompt format instead of reimplementing it from memory --
# fidelity to that format matters here since Step 2's whole point is
# reproducing published numbers.
#
# Clones https://github.com/nelson-liu/lost-in-the-middle (MIT-licensed, per
# its LICENSE file) the same way the notebook does, and reuses its
# `lost_in_the_middle.prompting.get_qa_prompt` for prompt construction. We do
# NOT reuse its scoring code -- we score with the vendored best_subspan_em in
# metrics.py, kept in this repo so it's independent of that external repo's
# state.
#
# Nothing in this module is called at import time. ensure_repo() makes a
# network call (git clone, public repo, no cost) -- not invoked automatically
# anywhere in this codebase.
import gzip
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import config

REPO_URL = "https://github.com/nelson-liu/lost-in-the-middle"
CACHE_DIR = Path(__file__).parent / ".cache" / "lost-in-the-middle"


def ensure_repo() -> Path:
    """Clone lost-in-the-middle if not already cached, and make its
    `lost_in_the_middle` package importable via sys.path."""
    if not CACHE_DIR.exists():
        CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(CACHE_DIR)], check=True
        )
    if str(CACHE_DIR) not in sys.path:
        sys.path.insert(0, str(CACHE_DIR))
    return CACHE_DIR


@dataclass
class NQExample:
    idx: int
    instruction: str
    question: str
    answers: List[str]
    context: List[str]  # the 20 formatted documents -- what compress_prompt's `context` arg wants


def load_position10(limit: Optional[int] = None) -> List[NQExample]:
    """Loads the gold-at-position-10 (0-indexed 9) NQ 20-doc file. Mirrors
    examples/RAG.ipynb cell 12's prompt construction exactly:
    mention_random_ordering=False, query_aware_contextualization=False."""
    repo = ensure_repo()
    from lost_in_the_middle.prompting import Document, get_qa_prompt

    path = repo / "qa_data" / "20_total_documents" / config.NQ_GOLD_POSITION_FILE
    examples = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            documents = [Document.from_dict(ctx) for ctx in row["ctxs"]]
            prompt = get_qa_prompt(
                row["question"],
                documents,
                mention_random_ordering=False,
                query_aware_contextualization=False,
            )
            parts = prompt.split("\n\n")
            instruction, question = parts[0], parts[-1]
            demonstration = "\n".join(parts[1:-1])
            examples.append(
                NQExample(
                    idx=i,
                    instruction=instruction,
                    question=question,
                    answers=row["answers"],
                    context=demonstration.split("\n"),
                )
            )
    return examples


def method_sweep_subset(examples: List[NQExample]) -> List[NQExample]:
    """Fixed random subset for method/baseline sweeps, per
    config.METHOD_SWEEP. Deterministic given the configured seed -- same
    subset every time this is called against the full 2,655-example list."""
    rng = random.Random(config.METHOD_SWEEP.seed)
    n = config.METHOD_SWEEP.n_examples
    return rng.sample(examples, min(n, len(examples)))
