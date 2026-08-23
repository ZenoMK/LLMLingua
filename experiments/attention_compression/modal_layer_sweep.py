# Modal entrypoint for the per-model attention-scorer layer sweep (Step
# 3, part 2). For each scorer model independently (Qwen2.5-1.5B-Instruct,
# Llama-2-7B-Chat), tries ~4 late-layer candidates and picks whichever
# layer gives the highest mean best_subspan_em -- see layer_sweep.py for
# the actual logic and reasoning.
#
# Real run is config.LAYER_SWEEP_N_EXAMPLES (100) examples from a seeded
# random subset of the full file -- --limit overrides with a
# deterministic first-N for a cheap pipeline smoke test first (same
# pattern modal_fidelity_check.py uses). Run the smoke test, confirm it
# works, THEN run the real thing -- this is the decision that picks a
# permanent per-model layer, not just a harness sanity check.
#
# Two GPU calls per model (compress candidates, then evaluate them) --
# same one-model-at-a-time split as everything else in this project.
# pick_best_layer (pure Python) runs inside evaluate_candidates_remote
# rather than back on this dev machine, since this machine's local torch
# install is broken and can't import compress.py/budgets.py's heavy deps
# anyway (see layer_sweep.py's top-of-file comment) -- doesn't matter
# where pure-Python code runs, so it runs wherever the data already is.
#
# Usage (only after a specific per-run go-ahead, per the project's working
# agreement):
#   modal run modal_layer_sweep.py --limit 5 --i-have-approval             # smoke test first
#   modal run modal_layer_sweep.py --i-have-approval                       # real run, both models
#   modal run modal_layer_sweep.py --model-size 1.5b --i-have-approval     # real run, just one model
#
# Per-example evaluated records (reader_answer, best_subspan_em, the
# compressed_prompt actually sent, per layer) get written locally to
# layer_sweep_output/ -- the two remote calls already return them
# (evaluate_candidates_remote's "records" key), they just weren't being
# kept anywhere before. Matters for the n=100 real run specifically: if a
# result looks surprising, this is what lets us look at what actually got
# selected/answered without spending again to re-run.
import json
from pathlib import Path

import config
from modal_app import GPU_TYPE, add_repo_to_path, app, base_image, hf_secret


@app.function(image=base_image, gpu=GPU_TYPE, timeout=60 * 60, secrets=[hf_secret])
def compress_candidates_remote(model_size: str, limit: int = None) -> list:
    add_repo_to_path()
    import layer_sweep

    return layer_sweep.compress_candidates(model_size, limit=limit)


@app.function(image=base_image, gpu=GPU_TYPE, timeout=60 * 60, secrets=[hf_secret])
def evaluate_candidates_remote(records: list) -> dict:
    add_repo_to_path()
    import layer_sweep

    return layer_sweep.evaluate_candidates(records)


@app.local_entrypoint()
def main(model_size: str = "", limit: int = None, out: str = "", i_have_approval: bool = False):
    if not i_have_approval:
        raise SystemExit(
            "Refusing to run: this loads a scorer model then the reader "
            "on Modal GPUs, twice per model size (compress + evaluate). "
            "Pass --i-have-approval only after getting a specific "
            "go-ahead for this run, per the project's working agreement."
        )

    model_sizes = [model_size] if model_size else list(config.ATTENTION_SCORER_MODELS)
    n = limit if limit is not None else config.LAYER_SWEEP_N_EXAMPLES
    out_dir = Path(out) if out else Path(__file__).parent / "layer_sweep_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    chosen = {}
    for size in model_sizes:
        print(f"\n=== sweeping {size} ({config.ATTENTION_SCORER_MODELS[size]}), n_examples={n} ===")
        records = compress_candidates_remote.remote(size, limit)
        print(f"compressed {len(records)} candidates")
        result = evaluate_candidates_remote.remote(records)
        print(json.dumps(result["summary"], indent=2))
        chosen[size] = result["summary"][size]["chosen_layer"]

        out_path = out_dir / f"{size}_n{n}.jsonl"
        with out_path.open("w") as f:
            for rec in result["records"]:
                f.write(json.dumps(rec) + "\n")
        print(f"wrote {len(result['records'])} per-example records to {out_path}")

    print("\nChosen layers (copy into config.ATTENTION_SCORER_LAYERS by hand):")
    print(json.dumps(chosen, indent=2))
