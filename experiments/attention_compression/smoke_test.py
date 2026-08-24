# 2-3 example smoke test running the full two-job pipeline end to end in
# one process: compression phase (LongLLMLingua, one budget), then reading
# phase (Llama-3.1-8B-Instruct) -- with the compressor freed before the
# reader loads, same GPU-memory discipline compress_job.py / read_job.py
# use as separate Modal containers, just sequential here for a quick check.
#
# Purpose: confirm this repo's llmlingua==0.2.2 installs and runs, and that
# the reader loads and generates, BEFORE committing to the internal-
# consistency fidelity check or the full method sweep. Run
# check_model_access.py first (even cheaper, no GPU needed).
#
# COST: loads meta-llama/Llama-2-7b-chat-hf (compressor) and then
# meta-llama/Llama-3.1-8B-Instruct (reader) -- both gated, both real GPU
# compute, presumably on Modal. No API cost anywhere now -- see FINDINGS.md
# for the GPU-hour estimate. Get a specific go-ahead before running.
#
# Usage (only after approval for this specific run):
#   python smoke_test.py --n 3 --budget 2x --i-have-approval
import argparse
import json

import compress_job
import config
import data
import read_job
from budgets import count_reader_tokens
from compress import compress_full_context


def run_smoke_test(n_examples: int = 3, budget_name: str = "2x"):
    examples = data.load_position10(limit=n_examples)
    origin_tokens_by_idx = {
        ex.idx: count_reader_tokens(compress_full_context(ex.context, ex.instruction, ex.question))
        for ex in examples
    }

    print("Compression phase: loading LongLLMLingua compressor...")
    records = compress_job.run_compression(
        rows=["longllmlingua"],
        budgets_list=[budget_name],
        examples=examples,
        origin_tokens_by_idx=origin_tokens_by_idx,
    )
    print(f"Compressed {len(records)} (example, budget) pairs. Compressor freed.")

    print("Reading phase: loading reader...")
    results = read_job.run_reading(records)

    for r in results:
        print(json.dumps(r, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--budget", default="2x", choices=list(config.COMPRESSION_RATES))
    parser.add_argument(
        "--i-have-approval",
        action="store_true",
        help="confirms a human approved this specific run (cost + scope)",
    )
    args = parser.parse_args()
    if not args.i_have_approval:
        raise SystemExit(
            "Refusing to run: this loads two gated 7-8B models and runs "
            "real generation. Pass --i-have-approval only after getting a "
            "specific go-ahead for this run, per the project's working "
            "agreement."
        )
    run_smoke_test(n_examples=args.n, budget_name=args.budget)
