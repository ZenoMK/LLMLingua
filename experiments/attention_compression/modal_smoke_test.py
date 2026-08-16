# Modal entrypoint for the 2-3 example smoke test (see smoke_test.py).
# Wraps smoke_test.run_smoke_test() to execute inside a Modal GPU
# container, since this dev machine has no GPU of its own. This is the
# first real Modal wiring in the project -- compress_on_gpu/read_on_gpu in
# modal_app.py are still stubs, reserved for the full two-job fidelity
# check / method sweep runs.
#
# Approved 2026-08-16 for this specific run: 3 examples, 2x budget, loads
# meta-llama/Llama-2-7b-chat-hf (compressor) then meta-llama/Llama-3.1-8B-
# Instruct (reader) on an A10G. Estimated cost: well under $1.
#
# Usage (only after a specific per-run go-ahead, per the project's working
# agreement):
#   modal run modal_smoke_test.py --n 3 --budget 2x --i-have-approval
import json
import sys

from modal_app import GPU_TYPE, app, base_image, hf_secret


@app.function(image=base_image, gpu=GPU_TYPE, timeout=60 * 30, secrets=[hf_secret])
def run_smoke_test_remote(n: int, budget: str) -> list:
    sys.path.insert(0, "/root/repo")
    sys.path.insert(0, "/root/repo/experiments/attention_compression")
    import smoke_test

    return smoke_test.run_smoke_test(n_examples=n, budget_name=budget)


@app.local_entrypoint()
def main(n: int = 3, budget: str = "2x", i_have_approval: bool = False):
    if not i_have_approval:
        raise SystemExit(
            "Refusing to run: this loads two gated 7-8B models on a Modal "
            "GPU and runs real generation. Pass --i-have-approval only "
            "after getting a specific go-ahead for this run, per the "
            "project's working agreement."
        )
    results = run_smoke_test_remote.remote(n, budget)
    for r in results:
        print(json.dumps(r, indent=2))
