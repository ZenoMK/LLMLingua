# OpenAI reader wrapper -- gpt-3.5-turbo-0613, greedy decoding (see
# config.py). This is the "downstream reader" whose API cost this whole
# project is budget-matched against.
#
# SAFETY: every function below that touches the network is a real call to
# the OpenAI API the moment it's invoked -- check_model_available() is free
# (models.list isn't billed), answer_question() costs money. Nothing in this
# module is called at import time, and nothing else in this repo calls these
# functions automatically. Every invocation needs a specific per-run
# go-ahead per the project's working agreement -- see the --i-have-approval
# gates in check_reader_availability.py and smoke_test.py.
import os
from dataclasses import dataclass

from openai import OpenAI

from config import READER_MAX_OUTPUT_TOKENS, READER_MODEL, READER_TEMPERATURE

_client = None


def client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set -- refusing to construct an OpenAI client."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def check_model_available(model: str = READER_MODEL) -> bool:
    """Free: GET /v1/models, OpenAI does not bill for listing. Run this
    before any run that depends on READER_MODEL -- old snapshots get
    deprecated without much notice. If it's not available: stop and ask
    before substituting a newer one, don't do it silently."""
    ids = {m.id for m in client().models.list().data}
    return model in ids


@dataclass
class ReaderResponse:
    answer: str
    prompt_tokens: int
    completion_tokens: int


def answer_question(prompt: str, model: str = READER_MODEL) -> ReaderResponse:
    """One chat-completion call. Costs money -- caller is responsible for
    having gotten a per-run go-ahead before calling this, especially in a
    loop over many examples."""
    response = client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=READER_TEMPERATURE,
        max_tokens=READER_MAX_OUTPUT_TOKENS,
        n=1,
    )
    choice = response.choices[0]
    usage = response.usage
    return ReaderResponse(
        answer=choice.message.content,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )
