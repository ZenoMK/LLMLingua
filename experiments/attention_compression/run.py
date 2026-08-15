# CLI orchestrator for the reproduction / comparison harness (Step 2/4).
#
# SAFETY: without --execute this script only loads data, builds prompts, and
# prints an estimated call count + $ cost -- it never touches the OpenAI API
# or loads a compressor model. --execute additionally requires
# --i-have-approval, and even then currently raises NotImplementedError: the
# actual execution path (loading the compressor, looping over examples,
# calling the reader) gets wired up alongside the first approved full run,
# not before -- see NOTES.md Step 2. Until then, smoke_test.py is the thing
# that actually runs end to end, on 2-3 examples, and it has its own
# approval gate.
import argparse
from pathlib import Path

import config
import data
from compress import ROW_FUNCS

# gpt-3.5-turbo-0613 pricing at time of writing: $1.50 / 1M input tokens,
# $2.00 / 1M output tokens. Re-check current pricing before trusting this
# for a real go/no-go decision -- this is a planning estimate, not a quote.
_INPUT_USD_PER_TOKEN = 1.50e-6
_OUTPUT_USD_PER_TOKEN = 2.00e-6
_ASSUMED_AVG_PROMPT_TOKENS = 1200  # refine with smoke_test.py's real numbers


def estimate_reader_cost(n_examples: int, n_rows: int, n_budgets: int) -> dict:
    n_calls = n_examples * n_rows * n_budgets
    cost = n_calls * (
        _ASSUMED_AVG_PROMPT_TOKENS * _INPUT_USD_PER_TOKEN
        + config.READER_MAX_OUTPUT_TOKENS * _OUTPUT_USD_PER_TOKEN
    )
    return {"n_calls": n_calls, "estimated_cost_usd": round(cost, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", choices=["fidelity_check", "method_sweep"], required=True
    )
    parser.add_argument("--rows", nargs="+", default=list(ROW_FUNCS))
    parser.add_argument("--budgets", nargs="+", default=list(config.TOKEN_BUDGETS))
    parser.add_argument(
        "--limit", type=int, default=None, help="override example count, for smoke tests"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually call the compressor/reader -- needs --i-have-approval too",
    )
    parser.add_argument("--i-have-approval", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    protocol = (
        config.FIDELITY_CHECK if args.protocol == "fidelity_check" else config.METHOD_SWEEP
    )
    n = args.limit if args.limit is not None else protocol.n_examples
    examples = data.load_position10(limit=n)
    if protocol is config.METHOD_SWEEP and args.limit is None:
        examples = data.method_sweep_subset(examples)

    estimate = estimate_reader_cost(len(examples), len(args.rows), len(args.budgets))
    print(f"protocol={protocol.name}")
    print(f"label={protocol.label}")
    print(f"n_examples={len(examples)} rows={args.rows} budgets={args.budgets}")
    print(
        f"estimated reader calls={estimate['n_calls']} "
        f"estimated cost=${estimate['estimated_cost_usd']}"
    )

    if not args.execute:
        print("--execute not passed: dry run only, nothing was called.")
        return

    if not args.i_have_approval:
        raise SystemExit(
            "Refusing to execute: --execute requires --i-have-approval, "
            "which should only be passed after a human has approved this "
            "specific run (model/benchmark/slice/sample count/cost), per "
            "the project's working agreement."
        )

    raise NotImplementedError(
        "Execution path (compressor loading + reader calls over the full "
        "protocol) is wired up alongside the first approved run, not "
        "before. Use smoke_test.py to validate the pipeline on 2-3 "
        "examples first."
    )


if __name__ == "__main__":
    main()
