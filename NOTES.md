# Step 1 — Orientation notes

Static code-reading only. No Modal runs, no API calls, nothing executed. This
machine doesn't even have `llmlingua`'s runtime deps installed (see below), so
everything here comes from reading `llmlingua/prompt_compressor.py`,
`DOCUMENT.md`, `README.md`, the example notebooks, and the `experiments/`
eval scripts.

## Installed version

Not installed anywhere. `pip show llmlingua` finds nothing, none of the local
conda envs (`base`, `panza`, `counterfactual_icu_env`, `dreal_env`,
`expLight`, `fresh-torch`, `petase_tournament`, `pyg_fresh`,
`reproduce_alpine`, `serova2_conda`, `serova_challenge`) have it, and there's
no venv in the repo. `import llmlingua` from the system Python fails on a
missing `nltk` before it even gets to the package.

The package **as source in this repo** is `llmlingua/version.py`:

```python
_MAJOR, _MINOR, _PATCH = "0", "2", "2"
VERSION = "0.2.2"
```

So: this fork is at `0.2.2`, un-diverged from `microsoft/LLMLingua@main` (last
merge: `e0e9d99`, adding SecurityLingua). For the Modal harness we'll build a
container image that `pip install`s this repo's `llmlingua/` directly (editable
or built wheel) rather than pulling `llmlingua` from PyPI, so we're testing
exactly this code, not whatever's on PyPI.

## How `PromptCompressor` is called

```python
from llmlingua import PromptCompressor

llm_lingua = PromptCompressor(
    model_name="NousResearch/Llama-2-7b-hf",  # default; the compressor's own backbone
    device_map="cuda",
)
compressed = llm_lingua.compress_prompt(
    context,            # List[str], one string per document/chunk
    instruction=instruction,
    question=question,
    rate=0.55,           # or target_token=<int>, which overrides rate
    ...
)
```

`PromptCompressor.__init__` (`llmlingua/prompt_compressor.py:71-93`) just
loads a causal LM + tokenizer via `AutoConfig`/`AutoTokenizer`/`AutoModelForCausalLM`
(`load_model`, line 118) — this model is the **compressor**, a separate
concern from whatever reader model eventually answers the compressed prompt.
Note `trust_remote_code=True` is the default.

`compress_prompt` (`prompt_compressor.py:426-725`) is the entry point for both
LLMLingua and LongLLMLingua — it's the same function, and behavior is
controlled entirely by keyword flags (`compress_prompt_llmlingua2` is a
separate code path, only reached when `use_llmlingua2=True` at init time — not
relevant to us). Internally it's a three-stage pipeline, each stage
independently toggleable:

1. **Context-level** (`use_context_level_filter=True` by default) — drop whole
   documents to fit a coarse token budget → `control_context_budget` (line
   1173).
2. **Sentence-level** (`use_sentence_level_filter=False` by default, off in
   the canonical LongLLMLingua recipe) → `control_sentence_budget` (line
   1243).
3. **Token-level** (`use_token_level_filter=True` by default) — iterative
   perplexity-based token dropping → `iterative_compress_prompt` (line 1523).

The canonical LongLLMLingua call (from `README.md:187-202` and
`examples/RAG.ipynb`, the NaturalQuestions multi-doc demo) is:

```python
compressed_prompt = llm_lingua.compress_prompt(
    prompt_list,
    question=question,
    rate=0.55,
    condition_in_question="after_condition",
    reorder_context="sort",
    dynamic_context_compression_ratio=0.3,  # or 0.4
    condition_compare=True,
    context_budget="+100",
    rank_method="longllmlingua",
)
```

i.e. stage 1 (context) + stage 3 (token) are active, stage 2 (sentence) is
skipped. The two flags the task asked about live in different stages:

### `rank_method="longllmlingua"` — coarse, stage 1 (context-level)

Default is `rank_method="llmlingua"`. Both `"llmlingua"` and
`"longllmlingua"` route to the same scoring function,
`get_distance_longllmlingua` (`prompt_compressor.py:2032-2045`, dispatched
from `get_rank_results`, line 1818), which is **question-conditioned
perplexity per document**:

```python
context_ppl = [
    self.get_condition_ppl(
        d,
        query + " We can get the answer to this question in the given documents.",
        condition_in_question,
    )
    for d, dl in zip(corpus, context_tokens_length)
]
```

`get_condition_ppl` (line 999) runs the doc through the compressor LM
concatenated with the question, and measures the loss **only on the question
tokens** (`condition_mode="after"`, i.e. it isolates the loss on `question`
positions, conditioned on having just read `d`) — this is "how surprised is
the model by the question, having read this document," which is a proxy for
"does this document answer the question." Documents are then sorted (line
1209 in `control_context_budget`) and kept greedily until the token budget
runs out, then optionally reordered (`reorder_context="sort"` puts the
highest-ranked docs at the start/end — the "lost in the middle" mitigation).

So despite the name, `rank_method="longllmlingua"` and `"llmlingua"` are
**identical code paths** at this stage — the string only matters because
`compress_prompt` asserts a `question` must be given when it's
`"longllmlingua"` (line 564-566) and defaults `condition_in_question` to
`"after"` (line 570-571) if you didn't set it, whereas `rank_method="llmlingua"`
forces `condition_in_question="none"` (line 572-577), i.e. plain
unconditioned perplexity, no question awareness. The name really just
toggles a default. Other values of `rank_method` (`"bm25"`, `"sentbert"`,
`"bge"`, `"gzip"`, `"openai"`, `"cohere"`, `"voyageai"`, ...) swap in
completely different scorers (line 2047-2069) — this is what Step 4's BM25
and embedding baselines will use, with no reimplementation needed. Confirmed
these are also wired into `control_sentence_budget` (line 1360-1367,
the `else` branch when `rank_method != "longllmlingua"`), so a
`rank_method="bm25"` + `use_sentence_level_filter=True` +
`use_context_level_filter=False` + `use_token_level_filter=False` call gives
sentence-level BM25 selection directly — structurally the same shape as our
Step 3 method, which is convenient for a clean baseline comparison.

### `condition_compare=True` — fine-grained, stage 3 (token-level)

Default `False`. This is "Iterative Token-level **Question-aware**
Fine-Grained Compression" per `DOCUMENT.md:355`. It only affects
`iterative_compress_prompt` (line 1523).

Without it: the token-keep threshold at each iteration is computed straight
from that token's own perplexity (`self.get_estimate_threshold_base_distribution(loss, ratio, False)`,
line 1712) — i.e., "how surprising is this token given only the preceding
compressed context" (self-information; no question). Tokens with high
self-perplexity survive (they're "hard to predict," proxy for "informative").

With it: the compressor runs **two parallel forward passes** per chunk of the
iteration:
- `loss` — perplexity of each token *conditioned on the question* (context is
  `question + context`, per `condition_flag` prepending the question as a
  `prefix` at line 661-679, so every chunk is scored with the question in its
  context window).
- `self_loss` — perplexity of the *same* tokens with **no question**
  (`self_input_ids = input_ids[:, start:]`, line 1553-1554, i.e. everything
  after the question prefix).

The keep threshold is then computed on `self_loss - loss` (line 1707-1710):
tokens whose surprise **drops the most once the question is known** are kept.
That's a contrastive signal — "this token only looks predictable/dispensable
when you already know the question," i.e. question-relevant — as opposed to
the unconditioned score, which just keeps generically-surprising tokens
regardless of relevance to the question. This is the mechanism the project
brief calls "contrastive-perplexity fine-grained step."

Cost implication (directly relevant to our Step 3 efficiency claim): this is
**two full forward passes per iteration chunk**, and neither is cacheable
across different questions against the same context — `self_loss` doesn't use
the question so it's shared across questions for a fixed context, but `loss`
does, and both re-run per document set per question in practice since
`iterative_compress_prompt` is invoked fresh inside `compress_prompt` each
call. This is the "their two uncacheable passes" the brief references, versus
our single cacheable query→context attention pass (context KV can be computed
once and reused across questions, since attention *to* context tokens is read
off a forward pass where only the query's presence changes the readout, not
the context's own KV — we'll confirm this concretely with code once we build
the scorer in Step 3).

## What already exists in-repo we should reuse (not rebuild)

- **NaturalQuestions multi-doc QA dataset + exact usage pattern**:
  `examples/RAG.ipynb` (cells 6-19) is a full worked example — clones
  [`nelson-liu/lost-in-the-middle`](https://github.com/nelson-liu/lost-in-the-middle),
  loads `qa_data/20_total_documents/nq-open-20_total_documents_gold_at_9.jsonl.gz`
  (2,655 examples, 20 docs each: 1 gold + 19 distractor, gold at position 9),
  builds prompts via `lost_in_the_middle.prompting.get_qa_prompt`, then calls
  `compress_prompt` with exactly the flags above. This is the dataset the
  task means by "NaturalQuestions multi-doc QA" and it's already
  wired end-to-end in this repo's own example — strong argument for using it
  in Step 2 over a fresh LongBench subset, since we inherit a
  battle-tested prompt format and don't have to guess at the eval protocol.
- **Metric**: `lost-in-the-middle` ships its own scorer; separately this repo
  already vendors `best_subspan_em` in
  `experiments/llmlingua2/evaluation/metrics.py:194` (also `qa_f1_score`,
  etc.) — same metric family used across the LLMLingua paper line for QA
  tasks. We should use `best_subspan_em` for scoring answers, matching what's
  already in this codebase, rather than importing `lost-in-the-middle`'s
  copy.
- **LongBench harness** (`experiments/llmlingua2/evaluation/eval_longbench.py`,
  `metrics.py`, `utils.py`) — a complete, working eval loop (dataset
  loading, `query_llm` against an OpenAI/Azure endpoint, per-dataset metric
  dispatch) but wired specifically to `compress_prompt_llmlingua2`, not the
  general `compress_prompt`/LongLLMLingua path. Reusable as a structural
  template (and for `utils.py`'s `query_llm`) if we pick LongBench instead of
  NQ, but would need adapting either way.
- **No existing LongLLMLingua-specific eval script** ships in `experiments/`
  — only `llmlingua2/` and `securitylingua/` have dedicated eval dirs. We'll
  be writing the Step 2 harness from scratch, using the RAG.ipynb pattern
  above as the reference implementation of "how LongLLMLingua is supposed to
  be called."

## Proposed plan, Steps 2-4

Each bullet is its own PR per the working agreement; nothing here gets built
until this NOTES.md PR is reviewed.

**Step 2 — reproduction harness**
- New `harness/` package: `data.py` (fetch + cache the lost-in-the-middle NQ
  jsonl, build prompts), `metrics.py` (thin import/re-export of
  `best_subspan_em` from `experiments/llmlingua2/evaluation/metrics.py`),
  `compress_longllmlingua.py` (wraps the exact RAG.ipynb call above),
  `reader.py` (calls the reader model — API or Modal-hosted, open question
  below), `run.py` (orchestrator: compress → read → score → write a results
  table).
- Modal app scaffold (`modal_app.py`): one function to load the compressor
  model on a GPU container, one to call the reader, wired so the compressor
  forward passes and reader calls are logged separately (need this split for
  Step 4's cost-per-forward-pass column anyway).
- Sanity-check gate: reproduce LongLLMLingua's reported accuracy on this
  benchmark within ~1-2 points, using our own reader, in our own harness — if
  it doesn't land close, that's a flagged finding, not something to tune
  away.
- Everything here is buildable and reviewable with zero spend; only the
  *first actual Modal/API run* needs a separate go-ahead per rule 2.

**Step 3 — attention selector (split into ≥2 PRs)**
- PR A: the scorer itself — single forward pass over `[context, query]` with
  a small LM (start with Qwen2.5-1.5B-Instruct), extract query→context
  attention, mean over heads, sentence-level aggregation, greedy
  budget-constrained selection. No dataset wiring yet, just the function +
  unit-style checks on toy input.
- PR B: the layer-sweep experiment — run the scorer at ~4 depths (50/66/75/100%)
  on ~5 held-out examples per model, for both Qwen2.5-1.5B-Instruct and
  LLaMA-2-7B-Chat *separately*, pick the best layer per model empirically,
  record it. This is the first Modal run of the project and needs explicit
  go-ahead (tiny: 2 models × 4 layers × 5 examples = 40 forward passes).
- PR C (maybe folded into B): wire the chosen layers into `run.py` so the
  scorer is a selectable row alongside LongLLMLingua.
- Budget matching against the *reader's* tokenizer, achieved-vs-target ratio
  reported per example.

**Step 4 — comparison**
- Extend `run.py` to sweep all 5 rows (attention-1.5B, attention-7B,
  LongLLMLingua repro, BM25 via `rank_method="bm25"`, embedding via
  `rank_method="sentbert"`) × however many budget points we agree on, same
  reader, same metric.
- `FINDINGS.md` started in Step 2 (first entry: the sanity-check
  reproduction result), appended to chronologically through Step 4.
- README gets a running "Modal + API spend so far" line, updated after every
  approved run.
- Quality-vs-budget plot + a cost table (forward passes per compression,
  cacheable y/n) as the final deliverable artifact.

## Modal cost estimate (whole project, target: fits under $50)

I don't have a locally-installed environment to empirically measure token
counts or timing, so this is a reasoned estimate from public pricing and the
code paths above — **the actual first PR in Step 2 should re-estimate on ~5
real examples before committing to a full run**, and each run still needs its
own go-ahead regardless of this estimate.

**Assumptions to confirm with you before Step 2:**
- Sample size: ~100-150 examples per row (enough for a blog-post-credible
  curve, not a paper-grade CI).
- ~3-4 budget points for the quality-vs-budget curve.
- Reader model: the original paper mostly used `gpt-3.5-turbo`. I'd propose
  `gpt-4o-mini` as the reader (same role — cheap instruction-following model
  that determines the "API cost" the task asks us to budget-match against —
  and meaningfully cheaper than legacy gpt-3.5-turbo pricing), but this is
  your call since it's the main cost driver and changes the absolute numbers
  below by ~3x either direction. Flagging as an open question, not deciding
  it here.
- ~2,655-example full NQ set — we will *not* run the full set; ~100-150 is a
  subsample.

**Reader (API) cost** — dominant term per the task framing:
- 5 rows × 4 budget points × 150 examples = 3,000 reader calls, plus ~150
  calls for the uncompressed-prompt sanity check.
- Rough sizes: compressed prompt averages ~500-1,500 tokens depending on
  budget point (anchored on RAG.ipynb's own demo: `target_token=500` for
  "6x", `target_token=100` for "10x"), uncompressed NQ 20-doc prompt ~2,500-3,000
  tokens, answers capped at ~100 output tokens.
- At `gpt-4o-mini` pricing (~$0.15/1M in, $0.60/1M out): compressed-row calls
  ≈ 3,000 × ~$0.00015 ≈ **$0.50**; uncompressed sanity calls ≈ 150 ×
  ~$0.0006 ≈ **$0.10**. Generous rounding: **under $5 total**, even doubling
  for reruns/bugs.
- At `gpt-3.5-turbo-0125` pricing (~$0.50/1M in, $1.50/1M out), same call
  counts: still **under $10 total**.
- If instead we used full GPT-4o (not mini) — I'm not proposing this, just
  bounding — same call counts land around **$8-15**. Still fits.

**Modal GPU (compressor + scorer) cost**:
- LongLLMLingua repro compression: coarse doc-ranking pass (1 forward pass
  per doc, ~20 docs) + `iterative_compress_prompt` at `iterative_size=200`
  over a ~3,000-token prompt (~15 chunks), **doubled** by
  `condition_compare=True` (loss + self_loss) → ballpark 30-50 forward
  passes/example on the default `NousResearch/Llama-2-7b-hf` compressor.
  150 examples × ~40 passes, batched short sequences on an A10G — order of
  tens of minutes of GPU time, not hours.
- Our attention scorer: 1 forward pass/example (that's the entire pitch), ×2
  model sizes × 150 examples — a few minutes of GPU time total, plus the
  40-pass layer sweep (negligible).
- BM25/embedding baselines: CPU only, effectively free.
- At Modal's A10G rate (~$1.10/hr) or A100-40GB (~$2.50-3.70/hr) for the 7B
  rows, even a generous 2-3 hours of cumulative GPU time (including cold
  starts, model loading, dev iteration/debugging reruns) lands around
  **$3-12**.

**Total estimate: roughly $10-25** across the whole project under the
gpt-4o-mini-or-similar assumption, with real headroom under the $50 cap even
accounting for repeated/debug runs — the load-bearing unknowns are (a) which
reader model you want (biggest lever, still bounded well under $50 even at
full GPT-4o), and (b) how many examples/budget points feel right for a blog
post. Both are cheap enough that I'd rather you pick based on what curve
looks convincing, not cost.

## Open questions for review

1. Reader model: `gpt-4o-mini` (my default suggestion), `gpt-3.5-turbo` (closer
   to the original paper, ~2-3x pricier but still cheap), or a Modal-hosted
   open model (zero marginal $ but loses the "API cost" framing the task
   asks us to budget against)?
2. Benchmark: NQ multi-doc (recommended — already has a working example in
   this repo, RAG.ipynb) vs. a LongBench subset?
3. Sample size / budget-point count for Step 2-4 — 100-150 examples × 3-4
   budget points is my default; happy to go smaller for a first pass.
4. Compressor backbone for the LongLLMLingua *reproduction* row — keep the
   library default (`NousResearch/Llama-2-7b-hf`) so we're reproducing their
   actual recommended setup?
