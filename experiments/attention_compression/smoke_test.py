# 2-3 example smoke test, mirroring examples/RAG.ipynb end to end.
#
# Purpose: confirm this repo's llmlingua==0.2.2 actually installs and runs
# (PromptCompressor loads the compressor backbone, compress_prompt runs with
# the locked LongLLMLingua flags) and that the OpenAI reader is reachable --
# BEFORE committing to the full 2,655-example fidelity run. Run this first;
# see check_reader_availability.py for an even cheaper pre-check.
#
# COST: this loads meta-llama/Llama-2-7b-chat-hf (compute cost -- presumably
# on Modal, see modal_app.py) and makes n_examples real OpenAI chat-
# completion calls (small $ cost, gpt-3.5-turbo-0613 pricing). Get a
# specific go-ahead with the estimated cost before running -- don't run this
# from "the plan was approved," per the project's working agreement.
#
# Usage (only after approval for this specific run):
#   python smoke_test.py --n 3 --budget 2x --i-have-approval
import argparse
import json

from llmlingua import PromptCompressor

import budgets
import compress
import config
import data
import metrics
import reader


def run_smoke_test(n_examples: int = 3, budget_name: str = "2x"):
    examples = data.load_position10(limit=n_examples)
    target_token = config.TOKEN_BUDGETS[budget_name]

    print(
        f"Loaded {len(examples)} examples. Loading compressor "
        f"({config.LONGLLMLINGUA_COMPRESSOR_MODEL})..."
    )
    compressor = PromptCompressor(model_name=config.LONGLLMLINGUA_COMPRESSOR_MODEL)

    results = []
    for ex in examples:
        comp = compress.compress_longllmlingua(
            compressor, ex.context, ex.instruction, ex.question, target_token
        )
        report = budgets.budget_report(budget_name, comp["origin_tokens"], comp["compressed_tokens"])
        reader_resp = reader.answer_question(comp["compressed_prompt"])
        em = metrics.best_subspan_em(reader_resp.answer, ex.answers)
        row = {
            "idx": ex.idx,
            **report,
            "reader_answer": reader_resp.answer,
            "gold_answers": ex.answers,
            "best_subspan_em": em,
            "reader_prompt_tokens_billed": reader_resp.prompt_tokens,
            "reader_completion_tokens_billed": reader_resp.completion_tokens,
        }
        results.append(row)
        print(json.dumps(row, indent=2))

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--budget", default="2x", choices=list(config.TOKEN_BUDGETS))
    parser.add_argument(
        "--i-have-approval",
        action="store_true",
        help="confirms a human approved this specific run (cost + scope)",
    )
    args = parser.parse_args()
    if not args.i_have_approval:
        raise SystemExit(
            "Refusing to run: this loads a 7B model and calls the OpenAI "
            "API for real. Pass --i-have-approval only after getting a "
            "specific go-ahead for this run, per the project's working "
            "agreement."
        )
    run_smoke_test(n_examples=args.n, budget_name=args.budget)
