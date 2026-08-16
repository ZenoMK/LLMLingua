# Attention-based compression experiment

Harness for comparing a one-pass query→context attention selector against
LongLLMLingua's two-pass contrastive-perplexity selector, at matched reader
token budgets, on NaturalQuestions multi-doc QA. Fully open-weight -- no
paid API anywhere in the pipeline, everything runs on Modal GPUs, which
also makes the whole experiment reproducible by anyone. Background and the
pivot away from an OpenAI reader are in [`../../FINDINGS.md`](../../FINDINGS.md)
(2026-08-16 entry); the original Step 1 orientation is in
[`../../NOTES.md`](../../NOTES.md).

**All settings are locked in [`config.py`](./config.py).** Every script here
imports from it rather than hard-coding model names, flags, budgets, or
sample sizes -- if a setting needs to change, change it there.

## Status

Scaffold only. Nothing in this directory has been run yet. Per this
project's working agreement, every run needs its own specific go-ahead --
scripts that load a model or run generation refuse to run without an
explicit `--i-have-approval` flag as a reminder of that.

## Pipeline shape: two decoupled jobs

1. **`compress_job.py`** -- loads one compressor/scorer backbone at a time,
   produces compressed prompts for a set of examples x rows x budgets,
   writes them to a JSONL file. Never loads the reader.
2. **`read_job.py`** -- loads only the reader (`meta-llama/Llama-3.1-8B-Instruct`),
   consumes a `compress_job.py` output file, generates answers, scores with
   `best_subspan_em`, writes results. Never loads a compressor.

Keeping these separate means we never hold two multi-GB models in one
GPU's memory at once, and either half can be re-run independently (e.g. if
you want to change the reader's decoding settings, you don't have to
recompute every compression). `modal_app.py` wraps both as separate Modal
functions sharing a Volume for the intermediate JSONL.

## Setup

```bash
pip install -e ../..                 # this fork's llmlingua, from source
pip install -r requirements.txt      # rank_bm25, sentence-transformers, modal, huggingface_hub, + metrics.py's deps
python -m nltk.downloader punkt      # needed by the bm25/sentbert sentence-level baselines
export HF_TOKEN=...                  # needed for the reader and both compressor backbones (all gated on HF)
```

`meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-3.1-8B-Instruct` both
require accepting Meta's license on Hugging Face for your account before
`HF_TOKEN` can pull them.

## Order of operations (each step needs its own go-ahead before running)

1. **`check_model_access.py`** -- confirms the reader and both compressor
   backbones (see `config.py`) are actually reachable with `HF_TOKEN`.
   Cheap (metadata only, no weights, no GPU), but still a real network call.
2. **`smoke_test.py`** -- runs 2-3 real examples through both jobs
   end-to-end (compress with LongLLMLingua, then read with the local
   reader). Confirms this fork's `llmlingua==0.2.2` installs and runs and
   that the reader loads and generates, before committing to a bigger run.
   Real GPU cost -- get that estimated and approved first.
3. **`compress_job.py --protocol fidelity_check --execute` then
   `read_job.py --execute`** -- the internal-consistency check: full-context
   vs. LongLLMLingua-compressed vs. zero-shot, on a fixed ~100-example
   subset, checking that full >= compressed > zero-shot holds with sane
   magnitudes. Replaces the earlier plan to reproduce the paper's own
   published (GPT-3.5-reader) numbers, which isn't a valid comparison now
   that we're on a different reader -- see `config.FIDELITY_CHECK`.
4. **`compress_job.py --protocol method_sweep --execute` then
   `read_job.py --execute`** -- all rows (attention 1.5B/7B, LongLLMLingua,
   bm25, sentbert) x both budgets on the fixed ~400-example subset. Depends
   on the Step 3 attention scorer, not built yet. `read_job.py` should be
   batched before this run (see its docstring) -- it's ~4,300 generations
   unbatched, the dominant cost in the project now.

## Files

| File | Purpose |
|---|---|
| `config.py` | Every locked setting -- reader, compressor backbones, pipeline flags, budgets, benchmark, sample sizes. Source of truth. |
| `data.py` | Loads the NQ gold-at-position-10 20-doc set via `nelson-liu/lost-in-the-middle`'s own prompt builder. |
| `metrics.py` | Re-exports `best_subspan_em` from `experiments/llmlingua2/evaluation/metrics.py` (sys.path trick, no logic duplicated). |
| `budgets.py` | Token counting in the reader's own HF tokenizer, achieved-vs-target ratio reporting. |
| `reader.py` | Local HF reader wrapper (load, generate, unload) -- `meta-llama/Llama-3.1-8B-Instruct`, greedy decoding. |
| `compress.py` | One wrapper per comparison row (`longllmlingua`, `bm25`, `sentbert`; `attention` stubbed for Step 3), plus the `full_context`/`zero_shot` pseudo-conditions used by the fidelity check. |
| `compress_job.py` | Compression half of the two-job pipeline. Dry-run cost/scope summary always prints; `--execute` needs `--i-have-approval`. |
| `read_job.py` | Reading half of the two-job pipeline. Same dry-run/approval gating. |
| `check_model_access.py` | Standalone approval-gated script for order-of-operations step 1. |
| `smoke_test.py` | Standalone approval-gated script for order-of-operations step 2. |
| `modal_app.py` | Modal Volume/image/two-function skeleton. Unverified stub -- see its docstring. |
