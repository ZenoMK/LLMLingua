# Modal app skeleton for GPU-side compression work (compressor forward
# passes for LongLLMLingua and, later, the attention scorer).
#
# THIS IS A STUB. Nothing here is deployed or run -- `modal run` /
# `modal deploy` are not invoked anywhere in this repo automatically, and
# won't be until a specific Modal run gets its own go-ahead per the
# project's working agreement. The image spec below (which deps, which
# version of `modal`, how this repo's own llmlingua/ gets into the
# container -- editable install vs. built wheel vs. mounted source) is a
# best-effort sketch to be finalized and actually tested against a real
# `modal` install when we wire up the first approved run; treat the exact
# API calls here as unverified until then.
import modal

app = modal.App("longllmlingua-attention-experiment")

image = (
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
        "openai>=1.0",
    )
    .run_commands("python -m nltk.downloader punkt")
    # TODO(step2-run): install this repo's llmlingua/ into the image --
    # decide editable-mount vs. built-wheel-from-this-commit when we wire
    # the first real run, so the image is pinned to a specific commit of
    # this fork rather than drifting.
)

# A10G assumed sufficient for 7B inference in fp16/bf16 for this workload
# (short sequences, no training). Revisit if the first real run needs more
# headroom.
GPU_TYPE = "A10G"


@app.function(image=image, gpu=GPU_TYPE, timeout=60 * 30)
def compress_on_gpu(method: str, payload: dict) -> dict:
    """Runs one compression call (LongLLMLingua now, the attention scorer
    once Step 3 lands) inside the Modal container. Stubbed -- wiring lands
    alongside the first approved run, once we've confirmed the pipeline
    works via smoke_test.py."""
    raise NotImplementedError(
        "Wire this up alongside the first approved Modal run -- see "
        "NOTES.md Step 2 and this repo's working agreement (rule 2)."
    )
