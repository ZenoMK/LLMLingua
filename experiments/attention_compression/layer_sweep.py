# Per-model layer sweep for the attention scorer (Step 3, part 2): for
# each scorer model independently, try a handful of late-layer candidates
# on a small fixed example set, and pick whichever layer gives the
# highest mean best_subspan_em once its compressed output is actually fed
# to the reader. "Best" means "leads to correct answers" -- the same
# evaluation criterion the rest of this project uses -- not an indirect
# proxy like attention entropy or spread.
#
# Two phases, same one-model-at-a-time discipline as compress_job.py /
# read_job.py: compress_candidates() loads only the scorer, produces
# every (layer, example) candidate's compressed prompt -- one forward
# pass per example (compute_token_attention batches every candidate
# layer into a single pass; there's no reason to re-run the model once
# per layer). evaluate_candidates() loads only the reader, scores every
# candidate and picks the per-model argmax layer.
#
# Kept self-contained here rather than routed through compress_job.py's
# row/protocol machinery -- this produces a ONE-TIME constant (which
# layer, per model) that gets hand-copied into
# config.ATTENTION_SCORER_LAYERS afterward, not a repeated pipeline row
# like "longllmlingua" or "bm25". Wiring the attention scorer into
# compress_job.py's row dispatch is a later PR, once this has actually
# run and there's a real layer number to wire in.
#
# SAFETY: compress_candidates()/evaluate_candidates() load a real model
# onto a GPU the moment they're called -- real compute cost. Nothing
# here executes at import time. The CLI entry point below refuses to
# execute without --i-have-approval.
# attention_scorer/config/data are safe to import unconditionally --
# none of them need transformers/llmlingua at module level (attention_scorer
# imports torch lazily inside compute_token_attention; data imports
# lost-in-the-middle lazily inside load_position10). budgets.py and
# compress.py both DO need transformers/llmlingua at module level though,
# so those two imports are kept lazy, inside compress_candidates() below,
# rather than up here -- keeping this file's top-level imports light is
# what makes candidate_layers/pick_best_layer (the pure-Python functions)
# locally testable at all without a working torch (see
# test_layer_sweep.py).
import argparse
import json
from typing import Dict, List

import attention_scorer
import config
import data


def candidate_layers(num_hidden_layers: int) -> List[int]:
    """~50%, 66%, 75%, 100% depth, as 0-indexed attention-layer indices
    (100% depth = the last layer, index num_hidden_layers - 1 -- there is
    no layer at index num_hidden_layers). Computed from the model's own
    config rather than hardcoded per model, since the two scorer sizes
    have different depths and guessing wrong would silently sweep the
    wrong layers.

    Order is preserved as given (50/66/75/100), not sorted -- irrelevant
    for correctness since every candidate gets evaluated regardless, but
    it does mean pick_best_layer's tie-breaking (Python's max() keeps the
    first-seen candidate on a tie) favors the shallower of two tied
    layers. Small models can round two fractions onto the same index;
    duplicates are dropped, keeping the first occurrence."""
    fractions = [0.50, 0.66, 0.75, 1.0]
    layers, seen = [], set()
    for f in fractions:
        idx = min(num_hidden_layers - 1, max(0, round(f * num_hidden_layers) - 1))
        if idx not in seen:
            seen.add(idx)
            layers.append(idx)
    return layers


def _build_full_prompt(instruction: str, compressed_body: str, question: str) -> str:
    """Matches how compress_prompt itself joins instruction/body/question
    (prompt_compressor.py: "\\n\\n".join([instruction, compressed_prompt,
    question])) -- attention_scorer.select_sentences only returns the
    compressed body, so this reproduces the same final assembly every
    other row gets, for a fair comparison."""
    return "\n\n".join([instruction, compressed_body, question])


def compress_candidates(model_size: str) -> List[dict]:
    """Loads config.ATTENTION_SCORER_MODELS[model_size], compresses the
    fixed layer-sweep example set at every candidate layer for that
    model's depth. Returns records ready for evaluate_candidates. Never
    loads the reader."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from budgets import budget_report, count_reader_tokens
    from compress import compress_full_context

    model_name = config.ATTENTION_SCORER_MODELS[model_size]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",  # required for output_attentions -- see attention_scorer.py
    )
    model.eval()
    try:
        layers = candidate_layers(model.config.num_hidden_layers)
        examples = data.load_position10(limit=config.LAYER_SWEEP_N_EXAMPLES)
        target_token = config.TOKEN_BUDGETS[config.LAYER_SWEEP_BUDGET]
        print(
            f"model_size={model_size} model={model_name} "
            f"num_hidden_layers={model.config.num_hidden_layers} candidate_layers={layers} "
            f"n_examples={len(examples)} budget={config.LAYER_SWEEP_BUDGET} ({target_token} tokens)"
        )

        records = []
        for ex in examples:
            origin_tokens = count_reader_tokens(compress_full_context(ex.context, ex.instruction, ex.question))
            _, offsets, bases, scores_by_layer = attention_scorer.compute_token_attention(
                model, tokenizer, ex.context, ex.question, layers
            )
            for layer in layers:
                sentences = attention_scorer.sentences_with_scores(ex.context, offsets, bases, scores_by_layer[layer])
                compressed_body = attention_scorer.select_sentences(sentences, target_token, count_reader_tokens)
                full_prompt = _build_full_prompt(ex.instruction, compressed_body, ex.question)
                report = budget_report(config.LAYER_SWEEP_BUDGET, origin_tokens, count_reader_tokens(full_prompt))
                records.append(
                    {
                        "scorer_model_size": model_size,
                        "scorer_model": model_name,
                        "num_hidden_layers": model.config.num_hidden_layers,
                        "layer": layer,
                        "idx": ex.idx,
                        "compressed_prompt": full_prompt,
                        "gold_answers": ex.answers,
                        **report,
                    }
                )
        print(f"compressed {len(records)} candidates ({len(examples)} examples x {len(layers)} layers)")
        return records
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def pick_best_layer(evaluated_records: List[dict]) -> Dict[str, dict]:
    """Pure Python: mean best_subspan_em per (scorer_model_size, layer),
    plus the argmax layer per model size. Returns:
        {model_size: {"chosen_layer": int, "by_layer": {layer: mean_em}}}
    """
    by_model_layer = {}
    for r in evaluated_records:
        by_model_layer.setdefault((r["scorer_model_size"], r["layer"]), []).append(r["best_subspan_em"])

    by_model = {}
    for (model_size, layer), ems in by_model_layer.items():
        by_model.setdefault(model_size, {})[layer] = sum(ems) / len(ems)

    return {
        model_size: {"chosen_layer": max(by_layer, key=by_layer.get), "by_layer": by_layer}
        for model_size, by_layer in by_model.items()
    }


def evaluate_candidates(records: List[dict]) -> dict:
    """Loads the reader, answers every candidate's compressed_prompt,
    scores with best_subspan_em. Never loads a scorer -- same
    one-model-at-a-time discipline read_job.py uses. Returns
    {"records": [...], "summary": pick_best_layer(...)} -- the summary is
    computed here (not returned separately for the caller to compute
    itself) so it can run inside whichever process/container actually
    has these records in memory; pick_best_layer itself is pure Python
    and doesn't care where it runs."""
    import reader as reader_module
    from metrics import best_subspan_em

    model, tokenizer = reader_module.load_reader()
    try:
        results = []
        for rec in records:
            resp = reader_module.answer_question(model, tokenizer, rec["compressed_prompt"])
            em = best_subspan_em(resp.answer, rec["gold_answers"])
            results.append({**rec, "reader_answer": resp.answer, "best_subspan_em": em})
    finally:
        reader_module.unload_reader(model)

    return {"records": results, "summary": pick_best_layer(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=list(config.ATTENTION_SCORER_MODELS), required=True)
    parser.add_argument(
        "--execute", action="store_true", help="actually load models and run -- needs --i-have-approval too"
    )
    parser.add_argument("--i-have-approval", action="store_true")
    args = parser.parse_args()

    model_name = config.ATTENTION_SCORER_MODELS[args.model_size]
    print(
        f"model_size={args.model_size} model={model_name} "
        f"n_examples={config.LAYER_SWEEP_N_EXAMPLES} budget={config.LAYER_SWEEP_BUDGET}"
    )

    if not args.execute:
        print("--execute not passed: dry run only, nothing was loaded or called.")
        return
    if not args.i_have_approval:
        raise SystemExit(
            "Refusing to execute: --execute requires --i-have-approval, "
            "passed only after a human has approved this specific run, "
            "per the project's working agreement."
        )

    records = compress_candidates(args.model_size)
    result = evaluate_candidates(records)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
