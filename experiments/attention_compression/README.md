# Attention-based compression experiment

Harness for comparing a one-pass query→context attention selector against
LongLLMLingua's two-pass contrastive-perplexity selector, at matched reader
token budgets, on NaturalQuestions multi-doc QA. Background and the full
Step 1-4 plan are in [`../../NOTES.md`](../../NOTES.md); running results go
in [`../../FINDINGS.md`](../../FINDINGS.md).

**All settings are locked in [`config.py`](./config.py).** Every script here
imports from it rather than hard-coding model names, flags, budgets, or
sample sizes -- if a setting needs to change, change it there.

## Status

Scaffold only. Nothing in this directory has been run yet. Per this
project's working agreement, every run (even free ones that touch a paid
API, like listing available models) needs its own specific go-ahead --
scripts that make network calls refuse to run without an explicit
`--i-have-approval` flag as a reminder of that.

## Setup

```bash
pip install -e ../..                 # this fork's llmlingua, from source
pip install -r requirements.txt      # openai, rank_bm25, sentence-transformers, modal
python -m nltk.downloader punkt      # needed by the bm25/sentbert sentence-level baselines
export OPENAI_API_KEY=...            # required by reader.py
```

## Order of operations (each step needs its own go-ahead before running)

1. **`check_reader_availability.py`** -- confirms `gpt-3.5-turbo-0613`
   (locked reader model, see `config.py`) is actually callable on the
   account. Free (models.list isn't billed), but still a real API call.
2. **`smoke_test.py`** -- runs 2-3 real examples end to end (compress with
   full LongLLMLingua, call the reader, score with `best_subspan_em`).
   Confirms this fork's `llmlingua==0.2.2` installs and runs, and that the
   whole pipeline is wired correctly, before committing to a full run. Has
   a real $ + compute cost -- get that estimated and approved first.
3. **`run.py --protocol fidelity_check`** -- the full 2,655-example
   reproduction (LongLLMLingua only, no reorder, gold at position 10),
   checked against the paper's Table 1 position-10 column. Not wired for
   execution yet (`--execute` currently raises `NotImplementedError`) --
   gets built out once (1) and (2) have passed.
4. **`run.py --protocol method_sweep`** -- all rows (attention 1.5B/7B,
   LongLLMLingua, bm25, sentbert) on the fixed ~400-example subset. Depends
   on the Step 3 attention scorer, not built yet.

## Files

| File | Purpose |
|---|---|
| `config.py` | Every locked setting -- reader, compressor backbones, pipeline flags, budgets, benchmark, sample sizes. Source of truth. |
| `data.py` | Loads the NQ gold-at-position-10 20-doc set via `nelson-liu/lost-in-the-middle`'s own prompt builder. |
| `metrics.py` | `best_subspan_em`, vendored from `experiments/llmlingua2/evaluation/metrics.py` (lighter-weight copy, see its docstring). |
| `budgets.py` | Token counting in the reader's tokenizer (tiktoken), achieved-vs-target ratio reporting. |
| `reader.py` | OpenAI reader wrapper + the free model-availability check. |
| `compress.py` | One wrapper per comparison row (`longllmlingua`, `bm25`, `sentbert`; `attention` stubbed for Step 3), all flags sourced from `config.py`. |
| `run.py` | CLI orchestrator: dry-run cost estimate always; `--execute` path not wired yet. |
| `check_reader_availability.py` | Standalone approval-gated script for order-of-operations step 1. |
| `smoke_test.py` | Standalone approval-gated script for order-of-operations step 2. |
| `modal_app.py` | Modal image/app skeleton for GPU compression work. Unverified stub -- see its docstring. |
