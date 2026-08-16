# FINDINGS

Chronological lab notebook for a personal blog-post experiment comparing
LongLLMLingua's two-pass contrastive-perplexity token scorer against a
one-pass query→context attention scorer, at matched reader token budgets on
NaturalQuestions multi-doc QA. Not affiliated with the upstream Microsoft
project beyond being a fork of it. Background in [`NOTES.md`](./NOTES.md);
harness code in
[`experiments/attention_compression/`](./experiments/attention_compression).

Numbers are reported honestly, not tuned to look good -- this is a personal
tinkering experiment, not a SOTA attempt. If the attention scorer scores
lower than LongLLMLingua, that's a valid, reportable result.

Every entry should be config-labeled: model, chosen layer (once relevant),
budget, reader, benchmark/slice -- so a number is traceable without having
to guess what settings produced it.

**Cumulative spend so far: $0.00** (Modal GPU-hours only -- no paid API
anywhere in the pipeline; updated after every approved run)

## Log

### 2026-08-15 -- harness scaffold, no runs yet

Built `experiments/attention_compression/`: data loading (NQ multi-doc QA,
gold at position 10, via `nelson-liu/lost-in-the-middle`), reader-tokenizer
budget accounting, an OpenAI reader wrapper, and one compression wrapper per
comparison row (`longllmlingua`, `bm25`, `sentbert`; the attention scorer
itself is a stub, lands in Step 3). All settings locked in `config.py` per
review of `NOTES.md`'s four open questions:

- Reader: `gpt-3.5-turbo-0613`, greedy (temperature 0).
- Compressor backbone: LLaMA-2-7B-Chat for both the LongLLMLingua row and
  our 7B attention row (held constant on purpose). Our 1.5B row uses
  Qwen2.5-1.5B-Instruct.
- Budgets: ~3,000 tokens (2x) and ~2,000 tokens (4x), measured in the
  reader's tokenizer.
- Benchmark: NQ multi-doc QA, gold at position 10 of 20 (hardest
  lost-in-the-middle case, where the paper's 21.4% headline is measured).
  **Reorder off on both sides** -- our method has no reordering step, so
  LongLLMLingua runs without one too, compared against the paper's Table 1
  per-position (no-reorder) column, not its with-reorder headline.
- Protocol: fidelity check on the full 2,655-example set (LongLLMLingua
  only); method/baseline sweep on a fixed random 400-example subset (seed
  `20260815`), labeled as a subset everywhere it's reported.

Nothing has executed yet. Next: confirm `gpt-3.5-turbo-0613` is still
callable on the account (free check), then a 2-3 example smoke test
end-to-end (real $ + compute cost, needs its own go-ahead), then -- pending
that going well -- the full fidelity run.

### 2026-08-16 -- pivot: fully open-weight, no paid API anywhere

No OpenAI account, and no willingness to spend on a per-token API for this
project. Replaced the reader and the whole cost model:

- **Reader: `meta-llama/Llama-3.1-8B-Instruct`** (was `gpt-3.5-turbo-0613`),
  greedy decoding, same reader for every row. Decided over
  `Qwen/Qwen2.5-7B-Instruct` (kept as `config.READER_MODEL_FALLBACK`)
  because we're already accepting Meta's HF license for the LongLLMLingua
  compressor backbone, so gating isn't new friction, and it's a more
  current model to be asking this question about than a 2023 GPT-3.5
  snapshot. Everything now runs on Modal GPUs -- no paid API in the
  pipeline at all, which also makes the experiment reproducible by anyone
  reading the eventual blog post.
- **Fidelity check redefined.** We can no longer cross-check against the
  paper's own published NQ numbers -- those were measured on their
  GPT-3.5 reader, and a number from a different reader isn't a valid
  comparison point. Replaced with an **internal-consistency check** on our
  own reader instead: full-context (no compression) vs. LongLLMLingua-
  compressed (2x budget) vs. zero-shot (no documents), on a fixed ~100-
  example subset at position 10. Expect full >= compressed > zero-shot; if
  that ordering doesn't hold with sane magnitudes, that's a wiring bug,
  and cheap to have caught on 100 examples instead of expensive to
  discover after the full method sweep. See `config.FIDELITY_CHECK`.
- **Pipeline split into two decoupled jobs.** With a compressor backbone,
  two attention-scorer backbones, and now a local reader all needing GPU
  memory, loading everything at once doesn't make sense. `compress_job.py`
  loads one compressor at a time, produces compressed prompts, saves them
  to disk, and never loads the reader; `read_job.py` loads only the
  reader, consumes those saved prompts, and never loads a compressor. This
  also means re-running the reader (e.g. after a decoding-settings change)
  doesn't require re-running compression, and vice versa -- saves GPU time
  across iterations. `run.py` (the earlier single-script orchestrator) is
  retired in favor of these two.
- Everything else from the 2026-08-15 settings is unchanged: benchmark
  (NQ multi-doc, position 10, reorder off everywhere), budgets (~3,000 /
  ~2,000 tokens, now measured in the reader's own HF tokenizer instead of
  tiktoken), the method itself (sentence-level one-pass attention, per-
  model layer sweep, two scorer sizes), and the baselines (`bm25`,
  `sentbert`, full LongLLMLingua).

**Cost model is now Modal GPU-hours, not API dollars.** Rough estimate,
still against the <$50 cap, broken out because the reader's generation
pass is now the dominant line item (it used to be an API call, now it's
GPU time):

| Phase | What | Rough GPU time (A10G) | Rough $ |
|---|---|---|---|
| Fidelity check -- compression | 100 examples x 1 budget, LongLLMLingua only (`full_context`/`zero_shot` need no compressor) | ~15-30 min | ~$0.30-0.60 |
| Fidelity check -- reading | 300 generations (100 examples x 3 conditions), unbatched | ~20-30 min | ~$0.35-0.55 |
| Method sweep -- compression | ~4,000 (example x budget x row) compressions across 5 rows -- LongLLMLingua's iterative passes dominate; bm25/sentbert/attention are each a small fraction of that | ~1-1.5 hr | ~$1.5-3 |
| Method sweep -- reading | ~4,000 generations, **unbatched** | ~5-6 hr | ~$6-8 |
| Method sweep -- reading | same, **batched** (e.g. batch=8) | ~40-60 min | ~$0.7-1 |

Unbatched total: roughly **$8-13**. Batched-reader total: roughly
**$3-6**. Either way comfortably under $50, but batching `read_job.py`
before the method-sweep run is the single biggest lever on both cost and
wall-clock time, and is flagged as a TODO in `read_job.py` rather than
built now (no need to build it before we've even confirmed the pipeline
works via the smoke test). All of the above is a planning estimate from
reasoning about the code paths, not a measurement -- the smoke test will
give real per-call timing on whatever GPU type we actually use, and that
should tighten this before the fidelity check or method sweep get
approved.

Nothing has executed yet. Next: `check_model_access.py` (confirms
`HF_TOKEN` can reach all three gated models, free), then the 2-3 example
smoke test (real GPU cost, needs its own go-ahead), then -- pending that --
the internal-consistency fidelity check.
