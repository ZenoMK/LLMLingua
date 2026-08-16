# Confirm every gated Hugging Face model this project depends on (the
# reader and both compressor backbones) is actually reachable with the
# current HF_TOKEN, before spending any GPU time trying to load one.
# huggingface_hub.model_info() fetches metadata only, no weights -- cheap
# and fast. Still a real network call against an external service, so it's
# gated the same way as any other run per the project's working agreement.
#
# Replaces the earlier check_reader_availability.py (OpenAI models.list) --
# now that every model in this pipeline is open-weight and HF-gated rather
# than one paid API, "is it callable" became "is it accessible on HF" for
# all three model roles, not just the reader.
#
# Usage (only after getting a go-ahead for this specific check):
#   python check_model_access.py --i-have-approval
import argparse

from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

import config

MODELS_TO_CHECK = {
    "reader": config.READER_MODEL,
    "longllmlingua_compressor": config.LONGLLMLINGUA_COMPRESSOR_MODEL,
    **{f"attention_scorer_{size}": name for size, name in config.ATTENTION_SCORER_MODELS.items()},
}


def check_all() -> dict:
    results = {}
    for role, model_id in MODELS_TO_CHECK.items():
        try:
            model_info(model_id)
            results[role] = (model_id, True, None)
        except GatedRepoError:
            results[role] = (model_id, False, "gated -- license not accepted for this token")
        except RepositoryNotFoundError:
            results[role] = (model_id, False, "not found -- check the model id")
        except Exception as e:  # surfacing whatever HF raised, not guessing at it
            results[role] = (model_id, False, str(e))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-have-approval", action="store_true")
    args = parser.parse_args()
    if not args.i_have_approval:
        raise SystemExit(
            "Refusing to run: this calls the Hugging Face Hub API. Pass "
            "--i-have-approval after getting a go-ahead."
        )

    results = check_all()
    for role, (model_id, ok, err) in results.items():
        status = "OK" if ok else f"NOT ACCESSIBLE ({err})"
        print(f"{role} ({model_id}): {status}")
    if not all(ok for _, ok, _ in results.values()):
        print(
            "\nSTOP if anything above failed -- don't substitute a "
            "different model (e.g. config.READER_MODEL_FALLBACK) without "
            "asking first."
        )
