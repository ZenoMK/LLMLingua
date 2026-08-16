# Compression job: loads ONE compressor/scorer backbone at a time,
# produces compressed prompts for a set of examples x rows x budgets, and
# writes them to disk as JSONL. Deliberately never loads the reader -- see
# read_job.py for the other half of the two-job split. Keeping compression
# and reading in separate processes/containers means we never hold two
# multi-GB models in GPU memory at once, and lets either half be re-run
# without re-running the other (saves GPU time across iterations).
#
# SAFETY: run_compression() loads a real model onto a GPU the moment it's
# called -- real compute cost. Nothing here executes at import time. The
# CLI entry point below refuses to execute without --i-have-approval.
import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from llmlingua import PromptCompressor

import compress
import config
import data
from budgets import budget_report, count_reader_tokens

# Which backbone PromptCompressor needs loaded for each row. bm25/sentbert
# never actually use the loaded causal LM for scoring (get_rank_results'
# bm25/sentbert branches only touch self.tokenizer, never self.model) --
# but PromptCompressor.__init__ unconditionally loads one regardless of
# rank_method, so we still have to give it something. Using the smallest
# model we already need elsewhere (the 1.5B attention scorer) minimizes
# the waste rather than loading a 7B model just to leave it idle.
ROW_COMPRESSOR_MODEL = {
    "longllmlingua": config.LONGLLMLINGUA_COMPRESSOR_MODEL,
    "bm25": config.ATTENTION_SCORER_MODELS["1.5b"],
    "sentbert": config.ATTENTION_SCORER_MODELS["1.5b"],
    # attention_1.5b / attention_7b: Step 3, not via PromptCompressor at all.
}


def _compressed_prompt_for_row(compressor, row: str, ex, target_token: Optional[int]) -> str:
    if row == "full_context":
        return compress.compress_full_context(ex.context, ex.instruction, ex.question)
    if row == "zero_shot":
        return compress.compress_zero_shot(ex.instruction, ex.question)
    fn = compress.ROW_FUNCS[row]
    return fn(compressor, ex.context, ex.instruction, ex.question, target_token)["compressed_prompt"]


def run_compression(
    rows: List[str],
    budgets_list: List[str],
    examples: List["data.NQExample"],
    origin_tokens_by_idx: Dict[int, int],
) -> List[dict]:
    """Runs every (example, budget) pair for each row in `rows`, one row's
    compressor loaded (then freed) at a time before the next row starts.
    Returns records shaped for read_job.py. Never loads a reader."""
    records = []
    for row in rows:
        model_name = ROW_COMPRESSOR_MODEL.get(row)
        compressor = PromptCompressor(model_name=model_name) if model_name else None
        try:
            budget_names = [None] if row in ("full_context", "zero_shot") else budgets_list
            for ex in examples:
                for budget_name in budget_names:
                    target_token = config.TOKEN_BUDGETS[budget_name] if budget_name else None
                    prompt = _compressed_prompt_for_row(compressor, row, ex, target_token)
                    report = budget_report(budget_name, origin_tokens_by_idx[ex.idx], count_reader_tokens(prompt))
                    records.append(
                        {
                            "idx": ex.idx,
                            "row": row,
                            "compressed_prompt": prompt,
                            "gold_answers": ex.answers,
                            "compressor_model": model_name,
                            **report,
                        }
                    )
        finally:
            if compressor is not None:
                del compressor
                gc.collect()
                torch.cuda.empty_cache()
    return records


def save_records(records: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["fidelity_check", "method_sweep"], required=True)
    parser.add_argument("--rows", nargs="+", default=None)
    parser.add_argument("--budgets", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None, help="override example count, for smoke tests")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true", help="actually load a compressor and run -- needs --i-have-approval too"
    )
    parser.add_argument("--i-have-approval", action="store_true")
    args = parser.parse_args()

    is_fidelity = args.protocol == "fidelity_check"
    protocol = config.FIDELITY_CHECK if is_fidelity else config.METHOD_SWEEP
    rows = args.rows or (config.FIDELITY_CHECK_ROWS if is_fidelity else config.BASELINE_ROWS)
    budgets_list = args.budgets or ([config.FIDELITY_CHECK_BUDGET] if is_fidelity else list(config.TOKEN_BUDGETS))

    n = args.limit if args.limit is not None else protocol.n_examples
    examples = data.load_position10(limit=n)
    if args.limit is None:
        examples = data.fidelity_check_subset(examples) if is_fidelity else data.method_sweep_subset(examples)

    print(f"protocol={protocol.name} rows={rows} budgets={budgets_list} n_examples={len(examples)}")

    if not args.execute:
        print("--execute not passed: dry run only, nothing was loaded or called.")
        return
    if not args.i_have_approval:
        raise SystemExit(
            "Refusing to execute: --execute requires --i-have-approval, "
            "passed only after a human has approved this specific run "
            "(rows/budgets/sample count/estimated cost), per the project's "
            "working agreement."
        )

    origin_tokens_by_idx = {
        ex.idx: count_reader_tokens(compress.compress_full_context(ex.context, ex.instruction, ex.question))
        for ex in examples
    }
    records = run_compression(rows, budgets_list, examples, origin_tokens_by_idx)
    save_records(records, args.out)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
