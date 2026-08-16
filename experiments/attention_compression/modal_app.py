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
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

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
    .add_local_dir(str(REPO_ROOT / "llmlingua"), remote_path="/root/repo/llmlingua")
    .add_local_dir(str(REPO_ROOT / "experiments"), remote_path="/root/repo/experiments")
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
