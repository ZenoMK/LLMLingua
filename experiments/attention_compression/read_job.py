# Reading job: loads ONLY the reader, consumes compress_job.py's saved
# records, generates answers, scores with best_subspan_em. Deliberately
# never loads a compressor -- the other half of the two-job split (see
# compress_job.py's header for why they're kept separate).
#
# SAFETY: run_reading() loads the reader model and runs real generation --
# real compute cost, and now the dominant cost in this project (no more
# per-token API charge, but the reader's GPU time is the new bottleneck --
# see the cost estimate in FINDINGS.md). Nothing here executes at import
# time. The CLI entry point below refuses to execute without
# --i-have-approval.
import argparse
import json
from pathlib import Path
from typing import List

import reader as reader_module
from metrics import best_subspan_em


def load_records(in_path: Path) -> List[dict]:
    with in_path.open() as f:
        return [json.loads(line) for line in f]


def run_reading(records: List[dict]) -> List[dict]:
    """Loads the reader once, answers every record's compressed_prompt in
    sequence, scores with best_subspan_em, frees the reader before
    returning. NOTE: unbatched -- fine for a handful of smoke-test
    examples, but batch this before it's used for the full ~4,300-call
    method sweep (see FINDINGS.md's cost estimate -- batching is the
    single biggest lever on that run's GPU-hours)."""
    model, tokenizer = reader_module.load_reader()
    try:
        results = []
        for rec in records:
            resp = reader_module.answer_question(model, tokenizer, rec["compressed_prompt"])
            em = best_subspan_em(resp.answer, rec["gold_answers"])
            results.append(
                {
                    **rec,
                    "reader_answer": resp.answer,
                    "best_subspan_em": em,
                    "reader_prompt_tokens": resp.prompt_tokens,
                    "reader_completion_tokens": resp.completion_tokens,
                }
            )
        return results
    finally:
        reader_module.unload_reader(model)


def save_results(results: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def summarize_by_row(results: List[dict]) -> dict:
    """Mean best_subspan_em per row (e.g. "full_context", "longllmlingua@2x",
    "zero_shot") plus an overall figure. An overall mean alone can't answer
    the fidelity check's actual question -- whether full_context >=
    compressed > zero_shot -- so this is what both the CLI and the Modal
    wiring report."""
    by_key = {}
    for r in results:
        key = r["row"] if r.get("budget_name") is None else f"{r['row']}@{r['budget_name']}"
        by_key.setdefault(key, []).append(r["best_subspan_em"])
    summary = {key: {"n": len(ems), "mean_em": sum(ems) / len(ems)} for key, ems in by_key.items()}
    all_ems = [r["best_subspan_em"] for r in results]
    summary["_overall"] = {"n": len(all_ems), "mean_em": sum(all_ems) / len(all_ems) if all_ems else 0.0}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="compress_job.py's output JSONL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true", help="actually load the reader and run -- needs --i-have-approval too"
    )
    parser.add_argument("--i-have-approval", action="store_true")
    args = parser.parse_args()

    records = load_records(args.in_path)
    print(f"loaded {len(records)} records from {args.in_path}")

    if not args.execute:
        print("--execute not passed: dry run only, reader not loaded, nothing was called.")
        return
    if not args.i_have_approval:
        raise SystemExit(
            "Refusing to execute: --execute requires --i-have-approval, "
            "passed only after a human has approved this specific run "
            "(record count/estimated cost), per the project's working "
            "agreement."
        )

    results = run_reading(records)
    save_results(results, args.out)
    summary = summarize_by_row(results)
    print(f"wrote {len(results)} results to {args.out}")
    for key, stats in summary.items():
        print(f"  {key}: n={stats['n']} mean_best_subspan_em={stats['mean_em']:.3f}")


if __name__ == "__main__":
    main()
