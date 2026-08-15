# FINDINGS

Chronological lab notebook for the attention-vs-contrastive-perplexity
prompt compression experiment (background in [`NOTES.md`](./NOTES.md);
harness code in
[`experiments/attention_compression/`](./experiments/attention_compression)).

Numbers are reported honestly, not tuned to look good -- this is a personal
blog-post experiment, not a SOTA attempt. If the attention scorer scores
lower than LongLLMLingua, that's a valid, reportable result.

Every entry should be config-labeled: model, chosen layer (once relevant),
budget, reader, benchmark/slice -- so a number is traceable without having
to guess what settings produced it.

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
