# Modal entrypoint for the internal-consistency fidelity check
# (config.FIDELITY_CHECK): full-context vs. LongLLMLingua-compressed (2x
# budget) vs. zero-shot, on a fixed ~100-example subset. Confirms
# full_context >= longllmlingua@2x > zero_shot with sane magnitudes,
# before trusting this harness for the real method sweep.
#
# Unlike modal_smoke_test.py (which runs both phases in one function call,
# fine for a 3-example check), this runs the real two-job split:
# compress_on_gpu writes compressed prompts to the shared Modal Volume,
# read_on_gpu reads them back and scores them as a separate container/GPU
# allocation. Either half can be re-run independently later (e.g. if only
# the reader's decoding settings change) without redoing the other.
#
# Usage (only after a specific per-run go-ahead, per the project's working
# agreement):
#   modal run modal_fidelity_check.py --i-have-approval
#   modal run modal_fidelity_check.py --limit 10 --i-have-approval   # cheaper dry run first
import json

from modal_app import ARTIFACTS_DIR, app, compress_on_gpu, read_on_gpu


@app.local_entrypoint()
def main(limit: int = None, i_have_approval: bool = False):
    if not i_have_approval:
        raise SystemExit(
            "Refusing to run: this loads a 7B compressor then an 8B "
            "reader on Modal GPUs and runs real compute over up to 100 "
            "examples x 3 conditions (~$1-1.50 estimated). Pass "
            "--i-have-approval only after getting a specific go-ahead for "
            "this run, per the project's working agreement."
        )

    compressed_path = f"{ARTIFACTS_DIR}/fidelity_check/compressed.jsonl"
    results_path = f"{ARTIFACTS_DIR}/fidelity_check/results.jsonl"

    n_compressed = compress_on_gpu.remote(protocol="fidelity_check", out_path=compressed_path, limit=limit)
    print(f"compression phase done: {n_compressed} records written to the volume")

    summary = read_on_gpu.remote(in_path=compressed_path, out_path=results_path)
    print(json.dumps(summary, indent=2))

    full = summary.get("full_context", {}).get("mean_em")
    zero = summary.get("zero_shot", {}).get("mean_em")
    compressed_rows = {k: v for k, v in summary.items() if k not in ("full_context", "zero_shot", "_overall")}

    print("\nOrdering check (expect full_context >= compressed > zero_shot):")
    if full is None or zero is None or not compressed_rows:
        print("  Could not check -- one of the three conditions is missing from the results.")
        return
    for key, stats in compressed_rows.items():
        comp = stats["mean_em"]
        ok = full >= comp > zero
        status = "OK" if ok else "VIOLATED -- flag this, don't rerun-until-it-passes"
        print(f"  full_context={full:.3f} >= {key}={comp:.3f} > zero_shot={zero:.3f} -- {status}")
