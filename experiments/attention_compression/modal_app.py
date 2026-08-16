# Modal app skeleton: two separate jobs (compress, read), mirroring
# compress_job.py / read_job.py -- never loading a compressor and the
# reader in the same container. A Modal Volume persists compressed-prompt
# records between the two jobs, and across dev iterations, so re-running
# the reader doesn't require re-running compression (and vice versa).
#
# THIS IS A STUB. Nothing here is deployed or run -- `modal run` /
# `modal deploy` are not invoked anywhere in this repo automatically, and
# won't be until a specific Modal run gets its own go-ahead per the
# project's working agreement. Treat the exact API calls here (Volume,
# Secret, Image methods) as unverified until tested against a real `modal`
# install -- finalize alongside the first approved run.
import modal

app = modal.App("attention-compression-experiment")

volume = modal.Volume.from_name("attention-compression-artifacts", create_if_missing=True)
ARTIFACTS_DIR = "/artifacts"

# HF_TOKEN needs to be available in-container to pull the gated compressor/
# reader backbones -- assumes a Modal secret named "huggingface" holding it
# (`modal secret create huggingface HF_TOKEN=...`), unverified until we set
# that up for the first real run.
hf_secret = modal.Secret.from_name("huggingface")

base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "transformers>=4.26.0",
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
    )
    .run_commands("python -m nltk.downloader punkt")
    # TODO(first-run): install this repo's llmlingua/ into the image --
    # decide editable-mount vs. built-wheel-from-this-commit when we wire
    # the first real run, so the image is pinned to a specific commit of
    # this fork rather than drifting.
)

# A10G assumed sufficient for 7-8B inference in bf16 (short sequences, no
# training). Revisit once smoke_test.py gives real per-call timing --
# that's the number the GPU-hours estimate in FINDINGS.md is currently
# guessing at.
GPU_TYPE = "A10G"


@app.function(
    image=base_image,
    gpu=GPU_TYPE,
    timeout=60 * 60,
    volumes={ARTIFACTS_DIR: volume},
    secrets=[hf_secret],
)
def compress_on_gpu(protocol: str, rows: list, budgets_list: list, out_path: str, limit: int = None) -> None:
    """Runs compress_job.py's logic inside the container, writing records
    to the shared volume. Never loads the reader. Stubbed -- wiring lands
    alongside the first approved Modal run."""
    raise NotImplementedError(
        "Wire this up alongside the first approved Modal run -- see "
        "FINDINGS.md and this repo's working agreement (rule 2)."
    )


@app.function(
    image=base_image,
    gpu=GPU_TYPE,
    timeout=60 * 60,
    volumes={ARTIFACTS_DIR: volume},
    secrets=[hf_secret],
)
def read_on_gpu(in_path: str, out_path: str) -> None:
    """Runs read_job.py's logic inside the container, loading only the
    reader. Stubbed -- wiring lands alongside the first approved run, and
    should add batched generation before it's used for the full method
    sweep (see read_job.py's docstring)."""
    raise NotImplementedError(
        "Wire this up alongside the first approved Modal run -- see "
        "FINDINGS.md and this repo's working agreement (rule 2)."
    )
