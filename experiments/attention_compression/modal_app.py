# Modal app skeleton: two separate jobs (compress, read), mirroring
# compress_job.py / read_job.py -- never loading a compressor and the
# reader in the same container. A Modal Volume persists compressed-prompt
# records between the two jobs, and across dev iterations, so re-running
# the reader doesn't require re-running compression (and vice versa).
#
# Deployment (`modal deploy`) is not invoked anywhere in this repo
# automatically -- functions here only run when explicitly invoked (`modal
# run ...`), and that still needs a specific per-run go-ahead per the
# project's working agreement. compress_on_gpu/read_on_gpu below are still
# stubs; modal_smoke_test.py is the first real wiring, added alongside the
# 2026-08-16 approved smoke test.
import modal

app = modal.App("attention-compression-experiment")

volume = modal.Volume.from_name("attention-compression-artifacts", create_if_missing=True)
ARTIFACTS_DIR = "/artifacts"

# Created manually: `modal secret create huggingface HF_TOKEN=...` (done
# 2026-08-16, from a cached `huggingface-cli login` token -- never passed
# through this codebase or this conversation).
hf_secret = modal.Secret.from_name("huggingface")

# REPO_ROOT here is this dev machine's checkout -- add_local_dir bakes/
# mounts these into the image at build/run time, so the container runs
# whatever's on disk at whichever commit this file was invoked from. No
# separate publish step, but also no independent pin -- fine for iterating
# on a personal experiment; revisit if reproducibility across time matters
# later (e.g. pip-installing a tagged commit instead).
#
# This whole module gets re-imported inside the remote container too (to
# resolve the specific function being invoked), where add_local_python_source
# has placed this file at the shallow /root/modal_app.py rather than its
# real repo-relative depth -- so parents[2] doesn't exist there. Harmless
# in that context (the image is already built by the time this re-import
# happens; REPO_ROOT's value is only ever used below, at local build time),
# it just needs to not crash. Learned this the hard way too.
import pathlib

_this_file = pathlib.Path(__file__).resolve()
try:
    REPO_ROOT = _this_file.parents[2]
except IndexError:
    REPO_ROOT = _this_file.parent

base_image = modal.Image.debian_slim(python_version="3.10").apt_install(
    "git"  # data.py clones nelson-liu/lost-in-the-middle at runtime; not in debian_slim by default
).pip_install(
    "torch",
    # llmlingua/prompt_compressor.py manually slices past_key_values as
    # legacy tuples-of-tuples (iterative_compress_prompt's KV-cache
    # compression, prompt_compressor.py:1656-1660) -- 2023-era code that
    # predates transformers' Cache-object-based KV cache. setup.py's own
    # unbounded "transformers>=4.26.0" let pip grab 5.15.0 here, which
    # broke that unpacking ("ValueError: too many values to unpack").
    # Pinning <5.0 (tried first) wasn't enough on its own: whatever 4.45+
    # version pip resolved to already required past_key_values to be a
    # real Cache object as *input* to Llama's forward() (raised
    # "'list' object has no attribute 'get_seq_length'" instead) -- the
    # legacy-tuple-to-Cache auto-conversion shim transformers carried for a
    # while had apparently already been dropped by then. Narrowed further
    # here to right around when Llama-3.1 support first landed (~4.43),
    # betting that shim was still present that close to the cutoff -- if
    # this doesn't work either, the version-pin approach is likely a dead
    # end and prompt_compressor.py's cache handling needs an actual patch.
    "transformers>=4.43,<4.46",
    "accelerate",
    "tiktoken",
    "nltk",
    "numpy",
    "rank_bm25",
    "sentence-transformers",
    "huggingface_hub",
    "evaluate",
    "jieba",
    "fuzzywuzzy",
    "rouge",
    "regex",
    "pydantic",  # nelson-liu/lost-in-the-middle's prompting.py needs this at import time
).run_commands("python -m nltk.downloader punkt")

# add_local_dir/add_local_python_source need REPO_ROOT to be the real repo
# checkout -- true when this module is first imported locally (to actually
# build the image), false when Modal re-imports this same file inside the
# already-running remote container just to resolve the function being
# invoked (REPO_ROOT falls back to a shallow path with nothing real under
# it there). Guarding on that rather than calling these unconditionally:
# add_local_dir puts files on the container's filesystem but doesn't
# register them as importable modules -- that only matters for a plain
# top-level `import <name>` (like modal_smoke_test.py's `from modal_app
# import ...`, which runs during container bootstrap, before any
# in-function sys.path fixups have a chance to run). Learned both of these
# the hard way: the first real run crash-looped on `ModuleNotFoundError: No
# module named 'modal_app'` (add_local_dir alone wasn't enough for that
# import -- add_local_python_source is the API built for it), and the
# second crashed on IndexError from REPO_ROOT's parents[2] not existing in
# the remote re-import. This guard fixes both by making the remote
# re-import a no-op for this part instead of erroring.
if (REPO_ROOT / "llmlingua").is_dir():
    base_image = (
        base_image.add_local_dir(str(REPO_ROOT / "llmlingua"), remote_path="/root/repo/llmlingua")
        .add_local_dir(str(REPO_ROOT / "experiments"), remote_path="/root/repo/experiments")
        .add_local_python_source("modal_app")
    )

# A10G assumed sufficient for 7-8B inference in bf16 (short sequences, no
# training). Revisit once smoke_test.py gives real per-call timing --
# that's the number the GPU-hours estimate in FINDINGS.md is currently
# guessing at.
GPU_TYPE = "A10G"


def add_repo_to_path() -> None:
    """Makes this repo's llmlingua/ and the harness scripts importable
    inside a remote container -- add_local_dir puts the files on disk
    (see the base_image comment above) but doesn't put them on sys.path.
    Call this at the top of any @app.function body before importing
    anything from this repo. Shared by every entry point (smoke test,
    compress_on_gpu, read_on_gpu) so the two path strings live in one
    place instead of three copies drifting apart."""
    import sys

    sys.path.insert(0, "/root/repo")
    sys.path.insert(0, "/root/repo/experiments/attention_compression")


@app.function(
    image=base_image,
    gpu=GPU_TYPE,
    timeout=60 * 60,
    volumes={ARTIFACTS_DIR: volume},
    secrets=[hf_secret],
)
def compress_on_gpu(
    protocol: str, out_path: str, rows: list = None, budgets_list: list = None, limit: int = None
) -> int:
    """Runs compress_job.py's logic inside the container, writing records
    to the shared volume as JSONL. Never loads the reader. Returns the
    number of records written."""
    add_repo_to_path()
    from pathlib import Path

    import compress
    import compress_job
    from budgets import count_reader_tokens

    rows, budgets_list = compress_job.resolve_rows_and_budgets(protocol, rows, budgets_list)
    examples = compress_job.load_examples(protocol, limit=limit)
    print(f"protocol={protocol} rows={rows} budgets={budgets_list} n_examples={len(examples)}")

    origin_tokens_by_idx = {
        ex.idx: count_reader_tokens(compress.compress_full_context(ex.context, ex.instruction, ex.question))
        for ex in examples
    }
    records = compress_job.run_compression(rows, budgets_list, examples, origin_tokens_by_idx)
    compress_job.save_records(records, Path(out_path))
    volume.commit()
    print(f"wrote {len(records)} records to {out_path}")
    return len(records)


@app.function(
    image=base_image,
    gpu=GPU_TYPE,
    timeout=60 * 60,
    volumes={ARTIFACTS_DIR: volume},
    secrets=[hf_secret],
)
def read_on_gpu(in_path: str, out_path: str) -> dict:
    """Runs read_job.py's logic inside the container, loading only the
    reader -- never a compressor. Returns the per-row summary (see
    read_job.summarize_by_row); the full per-example results are on the
    volume at out_path.

    NOTE: unbatched, same as read_job.py itself -- fine for the fidelity
    check's 300 generations, but batch this before the ~4,300-generation
    method sweep (see read_job.py's docstring)."""
    add_repo_to_path()
    from pathlib import Path

    import read_job

    volume.reload()  # pick up whatever compress_on_gpu committed
    records = read_job.load_records(Path(in_path))
    print(f"loaded {len(records)} records from {in_path}")
    results = read_job.run_reading(records)
    read_job.save_results(results, Path(out_path))
    volume.commit()

    summary = read_job.summarize_by_row(results)
    for key, stats in summary.items():
        print(f"  {key}: n={stats['n']} mean_best_subspan_em={stats['mean_em']:.3f}")
    return summary
