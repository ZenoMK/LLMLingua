# Local open-weight reader -- meta-llama/Llama-3.1-8B-Instruct, greedy
# decoding (see config.py). No paid API anywhere in this pipeline: this
# replaced an earlier OpenAI-based design (see FINDINGS.md, 2026-08-16).
#
# SAFETY: load_reader() downloads/loads a multi-GB gated model -- real
# compute cost. answer_question() runs generation -- also real compute
# cost. Nothing here is called at import time, and nothing else in this
# repo calls these automatically. Every invocation needs a specific per-run
# go-ahead per the project's working agreement -- see the --i-have-approval
# gates in smoke_test.py / read_job.py.
import gc
from dataclasses import dataclass
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config


def load_reader() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Loads config.READER_MODEL for generation. Gated on HF (meta-llama
    org) -- needs HF_TOKEN with the license accepted; see
    check_model_access.py, which should be run first."""
    tokenizer = AutoTokenizer.from_pretrained(config.READER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.READER_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def unload_reader(model) -> None:
    """Frees GPU memory. Call this before loading a compressor in the same
    process (e.g. smoke_test.py runs both phases sequentially) -- keeps the
    "compress OR read, never both" memory discipline even outside of the
    two separate Modal containers compress_job.py / read_job.py are meant
    to run in for the real, full-size runs."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


@dataclass
class ReaderResponse:
    answer: str
    prompt_tokens: int
    completion_tokens: int


def answer_question(model, tokenizer, prompt: str) -> ReaderResponse:
    """One greedy generation call. Costs real GPU time -- caller is
    responsible for having gotten a per-run go-ahead, especially in a loop
    over many examples. NOTE: this is unbatched (one example per call) --
    fine for the smoke test, but the real method-sweep run (~4,300
    generations) should batch before that run gets approved; see the cost
    estimate in FINDINGS.md."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=config.READER_MAX_OUTPUT_TOKENS,
            do_sample=False,  # greedy -- config.READER_TEMPERATURE=0.0 enforced structurally
            pad_token_id=tokenizer.eos_token_id,
        )
    completion_ids = output_ids[0, input_ids.shape[1] :]
    answer = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return ReaderResponse(
        answer=answer,
        prompt_tokens=input_ids.shape[1],
        completion_tokens=completion_ids.shape[0],
    )
