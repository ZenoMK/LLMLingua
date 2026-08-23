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

**Cumulative spend so far: ~$2-3** (Modal GPU-hours only, no paid API
anywhere in the pipeline; updated after every approved run). Precise for
the full fidelity-check run -- `modal app list` gives real start/stop
timestamps (2026-08-23 15:52:21 -> 16:24:42 = 32.35 min A10G =~ $0.59),
cheaper than the ~$0.90-1.50 estimate. Reasoned/rough for the rest
(2026-08-16 smoke-test debugging session + the 2026-08-23 `--limit 10`
validation run) -- Modal's ephemeral `modal run` apps don't keep long
listing history, so exact timestamps for those aren't recoverable after
the fact; still comfortably under the $50 project cap by a wide margin
either way.

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

### 2026-08-16 -- first real Modal run: the smoke test (succeeded, after 5 bugs)

`check_model_access.py` confirmed all three gated models reachable
(`meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-3.1-8B-Instruct`,
`Qwen/Qwen2.5-1.5B-Instruct`) once HF access was granted. Wired the actual
Modal execution (`modal_smoke_test.py`, plus finishing `modal_app.py`'s
image spec) and ran `smoke_test.py --n 3 --budget 2x` on Modal for real.
It took six attempts to get a clean run -- five distinct bugs, none of
them about the experiment's config/logic, all about getting 2023-era code
running in a fresh container in 2026. Logging all of them since the fixes
are now load-bearing:

1. **`ModuleNotFoundError: No module named 'modal_app'`** -- `add_local_dir`
   puts files on the container's filesystem but doesn't register them as
   importable modules. `modal_smoke_test.py`'s top-level `from modal_app
   import ...` runs during container bootstrap, before any in-function
   `sys.path` fixup gets a chance to run. Fix: `add_local_python_source
   ("modal_app")`, the API actually built for this.
2. **`IndexError` from `REPO_ROOT = pathlib.Path(__file__).resolve()
   .parents[2]`** -- this whole module gets re-imported inside the remote
   container too (to resolve the invoked function), where
   `add_local_python_source` places the file at a shallow `/root/
   modal_app.py` with no `parents[2]`. Harmless in that context (the image
   is already built by then) as long as it doesn't crash. Fix: guard the
   `add_local_dir`/`add_local_python_source` calls on whether `REPO_ROOT`
   looks real (`(REPO_ROOT / "llmlingua").is_dir()`), so the remote
   re-import is a no-op there instead of an error.
3. **`ImportError: cannot import name 'best_subspan_em' from partially
   initialized module 'metrics'`** -- not Modal-specific, a real bug,
   reproducible with a plain local `import metrics` (no GPU needed, would
   have been free to catch earlier). `experiments/attention_compression/
   metrics.py` and `experiments/llmlingua2/evaluation/metrics.py` share a
   basename; the `sys.path` + bare `import metrics` trick collides with
   the currently-executing module's own entry in `sys.modules`. Fix:
   rewrote to load the upstream file via `importlib.util.spec_from_file_
   location` under a private module name instead -- no `sys.path` mutation,
   no collision possible.
4. **`FileNotFoundError: 'git'`** -- `data.py` shells out to `git clone`
   for the lost-in-the-middle dataset; `debian_slim` doesn't ship `git`.
   Fix: `.apt_install("git")`.
5. **`ModuleNotFoundError: No module named 'lost_in_the_middle'`** -- that
   repo uses a `src/` layout (`src/lost_in_the_middle/`), not a
   repo-root package; `ensure_repo()` was adding the repo root to
   `sys.path` instead of `repo_root/src`. Confirmed by cloning the repo
   and looking, rather than guessing. Also needed `pydantic` added to the
   image (`prompting.py`'s only real import-time dependency, despite that
   repo's own `requirements.txt` listing a lot more for scripts we don't
   use).
6. **The real one: `transformers`/`llmlingua` KV-cache incompatibility.**
   `llmlingua/prompt_compressor.py`'s `iterative_compress_prompt` manually
   slices `past_key_values` as legacy tuples-of-tuples -- 2023-era code
   that predates `transformers`' `Cache`-object-based KV cache.
   `setup.py`'s unbounded `transformers>=4.26.0` let pip grab `5.15.0`,
   which broke the manual unpacking outright (`ValueError: too many values
   to unpack`). Pinning `<5.0` wasn't enough on its own -- whatever 4.45+
   version resolved already required `past_key_values` to be a real
   `Cache` object as *input* to Llama's forward pass
   (`AttributeError: 'list' object has no attribute 'get_seq_length'`),
   meaning the legacy-tuple-to-`Cache` auto-conversion shim was already
   gone there too. Narrowed to `transformers>=4.43,<4.46` (right around
   when Llama-3.1 architecture support first landed) and that worked --
   confirmed by a visible deprecation warning ("passing `past_key_values`
   as a tuple of tuples... deprecated and will be removed in v4.47") that
   proves the shim is present and doing its job in that window. This pin
   is now load-bearing and documented in `requirements.txt` and
   `modal_app.py` -- **do not remove or widen it** without re-verifying
   against a real run.

**Smoke test result** (`meta-llama/Llama-2-7b-chat-hf` compressor →
LongLLMLingua @ 2x budget → `meta-llama/Llama-3.1-8B-Instruct` reader →
`best_subspan_em`), 3 examples from the position-10 NQ set:

| idx | question (gist) | gold | reader answered | EM |
|---|---|---|---|---|
| 0 | first Nobel Prize in Physics | Wilhelm Conrad Röntgen | "...awarded to Wilhelm Conrad Röntgen in 1901." | 1.0 |
| 1 | next Deadpool movie release | May 18, 2018 | rambled about "Deadpool 3" development, never stated the date, hit the 100-token cap | 0.0 |
| 2 | SW wind timing across Nigeria | till September | "between April and July" | 0.0 |

Token budgets held in all three: target 3,000, compressed to 2,567-2,625
(achieved ratio 1.06-1.14x against the original ~2,780-2,950-token
prompts). 1/3 correct on 3 cherry-picked examples is not a quality signal
-- this run's only job was proving the pipeline works, and it does:
`llmlingua==0.2.2` installs and runs, both gated Llama models load, full
LongLLMLingua compression executes, the local open-weight reader
generates, scoring runs, all on Modal, zero paid APIs.

**Spend:** no exact figure available via the Modal CLI (would need the
web dashboard). Five of the six attempts failed within seconds, before
any GPU/download cost; two attempts did real work (one loaded the 7B
compressor and started compressing before crashing, the full successful
run downloaded and loaded both a 7B and an 8B model and ran compression +
generation to completion) -- low single-digit dollars at most for the
whole debugging session, consistent with the original "well under $1"
per-clean-run estimate. Worth checking the dashboard directly before
trusting this number for anything precise.

Next: the internal-consistency fidelity check (`config.FIDELITY_CHECK`) --
full-context vs. LongLLMLingua-compressed vs. zero-shot on ~100 examples --
needs its own cost estimate and go-ahead before running.

### 2026-08-23 -- fidelity check: cost estimate, one real bug, and the real two-job wiring

**Cost estimate**, grounded in the smoke test's actual cold-start numbers
(compressor load 46s+2s, reader load 47s+3s -- both one-time per job, not
per-example) plus reasoned per-example timing (compression: ~14-15
`iterative_size=200` chunks x 2 passes each for `condition_compare=True`
on the 7B compressor, ~5s/example; generation: ~2,600-token prefill + up
to 100 decoded tokens, unbatched, on the 8B reader, ~8s/example): 300
reader generations (100 examples x 3 conditions) + 100 compressions
(only the `longllmlingua` row needs one) + ~2 min cold starts -> **~50
min of A10G time, ~$0.90-1.50**.

**Found a real bug before spending anything on it.** `compress_job.py`'s
CLI loaded examples as `data.load_position10(limit=n)` (n=100) *before*
calling `fidelity_check_subset`/`method_sweep_subset` -- but those
functions `rng.sample()` whatever list they're handed, and a 100-example
list sampled down to 100 is just a shuffle of itself. The "fixed random
subset of the full 2,655-example set" every protocol docstring promises
was silently never happening; it would have quietly been "the first 100
examples in file order" instead. Caught by reading the code, not by
running it -- fixed by extracting `resolve_protocol` /
`resolve_rows_and_budgets` / `load_examples` into reusable functions
(shared by the CLI and the new Modal wiring, so the fix can't drift
between the two call sites) where `load_examples` now loads the full
2,655-example file first and only *then* subsets. Verified directly:
subset of 100 now spans idx 2-2,605 across the file, deterministic given
the seed.

**Wired the real two-job split for this run**: `compress_on_gpu` /
`read_on_gpu` in `modal_app.py` (previously `NotImplementedError` stubs)
now actually run `compress_job`/`read_job`'s logic inside separate Modal
GPU containers, handing off via the shared Volume (`commit()` after
writing, `reload()` before reading). New `modal_fidelity_check.py`
entrypoint orchestrates both calls and checks the actual ordering
(`full_context >= longllmlingua@2x > zero_shot`) the fidelity check
exists to validate -- added `read_job.summarize_by_row()` since an
overall mean EM across all three conditions mixed together can't answer
that question; needed per-row breakdown. `modal_smoke_test.py` refactored
to share the same `add_repo_to_path()` helper instead of its own inline
copy, now that it's the third place needing it.

Dry-run verified (no `--i-have-approval`): both `modal_smoke_test.py` and
`modal_fidelity_check.py` still build and register cleanly. **Not yet
run for real** -- given the smoke test needed 6 attempts before a dry run
that looked clean actually worked, a small `--limit 10` validation run is
worth doing before committing to the full ~100-example / ~$1-1.50 run.

**`--limit 10` validation run**: succeeded end to end (all 30 records
compressed and read/scored through the real two-job split with volume
handoff -- no new bugs). Ordering check technically flagged
`VIOLATED`: `full_context=0.800 >= longllmlingua@2x=0.600 >
zero_shot=0.600` -- zero_shot tied the compressed row instead of trailing
it. Pulled the raw per-example results before treating that as a problem
(`modal volume get attention-compression-artifacts
fidelity_check/results.jsonl`): `full_context` and `longllmlingua` agreed
on 8/10 examples, including agreeing on both examples they got *wrong*
even with the full uncompressed context (those are just hard questions,
not a compression failure). The two disagreements canceled out --
compression happened to drop the key sentence for one example
(Cyrus/human-rights, where zero-shot guessed right anyway) and preserved
it for another (eye evolution, where zero-shot didn't know it). All
answers read as coherent prose referencing the actual documents, nothing
garbled. Read this as n=10 noise, not a wiring bug, and proceeded to the
full run rather than iterating on a sample too small to trust.

**Full 100-example run: ordering holds.**
`full_context=0.650 >= longllmlingua@2x=0.620 > zero_shot=0.570` --
**OK**. The compression cost is small (0.65 -> 0.62, 3 points) and it
clearly beats parametric-knowledge-only (0.57), confirming both that the
compressed prompts retain the information needed to answer and that this
reader has real headroom above guessing from pretraining alone -- i.e.
retrieval content is doing real work here, and the harness measures that
correctly.

Compression stats for the `longllmlingua@2x` row (pulled from the full
results, not just the printed summary): averaged 2,945 -> 2,556 tokens
(achieved ratio 1.153x against a 2x/3,000-token target), range 2,310-2,753
tokens, **0/100 examples exceeded the token budget**.

**This validates the harness end to end.** `full_context`, `longllmlingua`,
and `zero_shot` all wire up correctly through both Modal jobs; the reader
answers coherently; scoring and budget accounting are both doing what
they claim. Next real milestone is Step 3 -- the attention scorer itself
-- since the reproduction/baseline machinery this was built to validate
now has a passing sanity check behind it.

### 2026-08-23 -- Step 3, part 1: the attention scorer (no runs)

First real piece of the method itself: `attention_scorer.py`. Per the
project brief, split from the layer sweep (separate PR, needs a real
model/GPU/approval) -- this PR is scoring + selection logic only, no
dataset wiring, no Modal run.

Design decisions made concrete in code:
- **Layout is `[context tokens][question tokens]`** -- context first is
  what lets the question tokens' attention run back *over* the context at
  all, since causal masking only lets a token attend to itself and
  earlier tokens. No chat template -- this reads the model's raw
  attention over content, not a template-mediated one; we're not asking
  it to generate anything.
- **`attn_implementation="eager"` is required**, not optional -- `sdpa`/
  `flash_attention_2` don't return attention weights at all,
  `output_attentions=True` silently gives back `None` for them instead of
  erroring. Documented loudly in `attention_scorer.py` and `layer_sweep.py`'s
  future model-loading code needs to remember it, or it'll rediscover
  this the expensive way (load a 7-8B model on Modal, then get a
  `None`-attention crash).
- **Sentence score = mean of its tokens' attention, not sum** -- matches
  how llmlingua's own sentence-level perplexity scoring works
  (`granularity="sentence"` is a `loss.mean()`), and avoids just rewarding
  longer sentences for having more tokens. Unit-tested directly
  (`test_uses_mean_not_sum_so_longer_sentences_arent_favored`).
- **Selection overshoots rather than undershoots** the token budget (adds
  the sentence that crosses the line, then stops) -- same convention
  llmlingua's own `control_sentence_budget` uses for its bm25/sentbert
  rows.
- **Reconstruction preserves original document/sentence order**, never
  rank order -- consistent with this project's no-reordering rule
  everywhere else.
- **Cacheability**: context-then-query layout means the context's KV
  cache is, in principle, reusable across different queries against the
  same context (one cacheable pass, vs. LongLLMLingua's two uncacheable
  ones). This benchmark's NQ examples each have a unique context, so nothing
  in this harness's data actually exercises that reuse -- it's a
  structural property of the method, documented honestly as such rather
  than claimed as something this specific experiment demonstrates.

**Architecture**: split model-dependent (`compute_token_attention` -- the
real forward pass, needs a GPU) from model-independent
(`sentences_with_scores`, `select_sentences` -- pure Python) code. Not
just tidiness: this dev machine's local `torch` install is broken (missing
`torch.nn`, pre-existing, unrelated to this project) and can't run a real
forward pass locally regardless, so this split is what made local testing
possible at all. 7 unit tests in `test_attention_scorer.py`, synthetic
inputs (a trivial word-level fake tokenizer, hand-computed expected
scores), no GPU needed -- all passing (`python test_attention_scorer.py`).

**Also fixed a latent bug in already-merged code**, found while checking
whether `nltk.sent_tokenize` would work in the Modal image for this:
`modal_app.py` only downloaded the `punkt` nltk resource, but `nltk>=3.9`
renamed it to `punkt_tab` (a security fix -- the old resource used
`pickle`, which allows arbitrary code execution). Reproduced locally
first (`LookupError: Resource punkt_tab not found`) before touching
Modal. Neither the bm25/sentbert baselines nor anything else in this
project has ever actually called `nltk.sent_tokenize` in a successful
run so far (the smoke test and fidelity check only exercise
`longllmlingua`/`full_context`/`zero_shot`), so this was sitting
undiscovered in merged code -- would have broken the first bm25/sentbert
row and this scorer the moment either actually ran. Now downloads both
`punkt` and `punkt_tab`, covering nltk versions on either side of the
rename.

Not yet done: the layer sweep (which layer, per scorer model, to read
attention from) and wiring the scorer into `compress_job.py`'s row
dispatch -- both need `attention_scorer.py` to exist first, which it now
does.

### 2026-08-23 -- Step 3, part 2: the layer sweep (no runs)

`layer_sweep.py` + `modal_layer_sweep.py`. For each scorer model
independently, tries ~4 late-layer candidates (~50/66/75/100% depth,
computed from the model's own `num_hidden_layers` rather than hardcoded)
on a fixed 5-example set, and picks whichever layer gives the highest
mean `best_subspan_em` once its compressed output is fed to the reader --
"best" means "leads to correct answers," the same criterion the rest of
this project uses, not an indirect proxy like attention entropy.

**Revised `compute_token_attention`'s signature** (merged in the previous
PR, but not called from anywhere else yet, so this was the moment to fix
it): it took a single `layer` and did one forward pass per call. But
`output_attentions=True` already returns *every* layer's attention
weights in one pass -- sweeping 4 candidates the old way would have been
4 fully-redundant forward passes per example for identical underlying
compute. Now takes `layers: List[int]` and returns `{layer: token_scores}`
from a single pass.

**Two phases, same one-model-at-a-time discipline as `compress_job.py`/
`read_job.py`**: `compress_candidates()` loads only the scorer, produces
every (layer, example) candidate's compressed prompt; `evaluate_candidates()`
loads only the reader, scores every candidate, and computes the per-model
argmax layer (`pick_best_layer()`) before returning -- computed wherever
the records already are (inside the Modal container) rather than shipped
back to this dev machine to compute locally, since this machine's broken
local `torch` can't import `compress.py`/`budgets.py` anyway (see below).

**Kept `candidate_layers()` and `pick_best_layer()` locally testable**,
same reasoning as the scorer itself: moved `layer_sweep.py`'s `budgets`/
`compress` imports (both transitively need `transformers`/`llmlingua`,
broken locally) from module level into `compress_candidates()`'s function
body, so importing the file itself doesn't require a working `torch`.
7 more unit tests (`test_layer_sweep.py`), all passing -- including
verifying `candidate_layers(28)` lands on the expected indices, 100%
depth is always the last valid layer (never out of range), small models
that collapse two fractions onto the same index get deduped, and
`pick_best_layer` keeps model sizes independent and breaks ties toward
the shallower layer (documented behavior, not an accident).

Dry-run verified (no `--i-have-approval`): `modal_layer_sweep.py` builds
and registers cleanly. **Not yet run for real.**

Also added `config.LAYER_SWEEP_N_EXAMPLES` / `LAYER_SWEEP_BUDGET` (5
examples at the time, the "2x" budget, matching `FIDELITY_CHECK_BUDGET`'s
reasoning) and an empty `config.ATTENTION_SCORER_LAYERS` placeholder, to
be filled in by hand once the sweep actually runs.

**Revised before running anything**: raised `LAYER_SWEEP_N_EXAMPLES` from
the brief's suggested "~5" to **100**. Picking a permanent layer is a
discrete choice among 4 candidates, decided once and then locked in for
the rest of the project -- unlike the fidelity check's n=10 (which was
just validating the harness works, resolved cleanly at n=100 with no
consequence either way), there's no later step that rechecks the layer
choice at scale. n=5 risks locking in whichever layer got lucky on those
5 questions. Agreed plan: smoke-test the pipeline at n=5 first (`--limit
5`, cheap, same role the earlier `--limit 10` fidelity-check run played),
confirm it works end to end, then run the real 100-example decision.

**Also applied the same subset-sampling fix here that `compress_job.py`
got earlier**: the layer sweep's real run now loads the FULL 2,655-example
file and takes a seeded random 100-example subset
(`data.layer_sweep_subset`, its own seed, independent of
`fidelity_check_subset`/`method_sweep_subset`) rather than just the first
100 in file order -- "first N" is still what `--limit` gives for the
cheap smoke test, where determinism matters more than representativeness,
but the real decision-making run needed the same fix. Verified locally
(no GPU needed, `data.py` doesn't need `torch`): 100-example subset spans
idx 10-2,625 across the file, deterministic given the seed, and its
overlap with the other two subsets (6% with fidelity_check's, 15% with
method_sweep's) matches what independent random draws from the same pool
should produce.

**Updated cost estimate for the real (n=100) run**: reader generations
now dominate even more than before -- 400 per model (100 examples x 4
layers) x 2 models = 800 total, exceeding the fidelity check's own 300.
Compression is comparatively cheap (~100 short forward passes per model,
no generation). Rough total: **~$1.50-3, roughly 2 hours of wall-clock
A10G time, unbatched** -- comfortably under the $50 cap, but long enough
to need running in the background rather than watched live. The n=5
smoke test first is unchanged in cost (~$0.30-0.70, a few minutes).

### 2026-08-23 -- the n=5 smoke test surfaced a real bug and a real limit

First real run of `modal_layer_sweep.py --limit 5`. Two findings, one
small and fixed immediately, one substantial:

**`ModuleNotFoundError: No module named 'config'`** -- same root cause,
third time now, as the original `modal_app` import crash-loop:
`modal_layer_sweep.py` does `import config` at module level (for its
local entrypoint's own use), but the *whole file* -- not just whichever
function is actually being invoked -- gets re-imported remotely to
resolve which decorated function to call, and `config` wasn't registered
via `add_local_python_source` the way `modal_app` was. Fixed by
registering it alongside `modal_app`, and documented the general pattern
in `modal_app.py`'s comment so it doesn't get rediscovered a fourth time
by some future `modal_*.py` entrypoint. Shipped separately as its own PR
(#10) since it's independent of everything else here.

**With that fixed, the 1.5B model ran completely cleanly** -- and the
result is itself informative: all four candidate layers tied at **exactly
0.6 mean EM** (3/5 correct, identical for every layer). A perfect 4-way
tie at n=5 is about as clean a confirmation as you could ask for that 5
examples has no power to distinguish between layers -- exactly the
concern that led to the "smoke test at 5, then decide at 100" plan
rather than trusting n=5 outright.

**The 7B model hit `CUDA OutOfMemoryError`, and it's structural.**
`output_attentions=True` is a single global flag with no per-layer
selectivity: requesting it forces the model to compute *and retain* the
full `[heads, seq, seq]` attention matrix for **all 32 layers
simultaneously** (~1.15GB each =~ 37GB) on top of the model weights
(~14GB) -- past the A10G's 24GB. Confirmed this isn't sweep-specific: even
a single requested layer would hit the same wall, since the model
computes and retains every *other* layer's attention too regardless of
what the caller asked for -- meaning this would have resurfaced
identically the moment the chosen 7B layer got wired into the real
method sweep, not just here.

Discussed two fixes (bigger GPU vs. properly scoping what gets computed);
**chose the proper fix** on the reasoning that a fixed-layer real run
would hit the same wall as the sweep, so "bigger GPU" wasn't actually
sweep-scoped -- it would mean paying the inflated rate for every 7B
attention-row example for the rest of the project, not just this one-off
calibration step.

**Fix**: `compute_token_attention` no longer uses the model-level
`output_attentions=True` at all. Instead: leave the model call at its
default (`output_attentions=False`, `use_cache=False` since this is a
one-shot scoring pass, not generation) -- with no accumulation, each
layer's transient attention tensor is freed as soon as that layer's own
forward returns, so baseline peak memory from attention is already just
O(1) layer. Then use forward hooks scoped to exactly the requested
layers: a pre-hook forces `output_attentions=True` for only that one
layer's own call, and a paired forward hook captures that layer's
attention weights directly from its return value, reduces it to a
per-context-token score immediately, and lets the GPU tensor get freed.
Peak extra memory becomes O(len(layers)) -- at most 4 -- not O(32).

Two places this could have been subtly wrong, both made defensive
instead of assumed:
- **How the pre-hook overrides `output_attentions`**: doesn't assume
  whether the caller (transformers' internal per-layer loop) passes it
  positionally or by keyword -- that calling convention isn't part of
  transformers' public API contract and isn't something worth hardcoding
  around. Uses `inspect.Signature.bind_partial` to rebuild the call
  correctly either way. Extracted as `_bind_kwarg_true`, pure Python, and
  unit-tested against both calling conventions plus the
  no-kwarg-passed-at-all case (4 tests).
- **How the capture hook finds the attention-weights tensor** in a decoder
  layer's returned tuple: not a hardcoded index (exact tuple layout
  varies with `use_cache`/`output_attentions` combinations across
  versions), but a shape match -- 4D, batch-size-1, square in the last
  two dims. Extracted as `_looks_like_attention_weights`, pure Python,
  unit-tested against the realistic shape plus three near-miss shapes
  that should NOT match (3D hidden states, batch>1, a KV-cache-shaped
  non-square tensor) (4 tests).

8 new unit tests total, all passing, no GPU needed. What's genuinely NOT
verified yet, and can't be without spending real money: whether
`model.model.layers` (the decoder stack attribute path) and the hook
mechanics actually behave as expected against a real loaded model --
that needs an actual Modal run. Re-running the same already-approved
`--limit 5` smoke test next, now covering both models this time.
