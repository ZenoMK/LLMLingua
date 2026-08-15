# Confirm config.READER_MODEL ("gpt-3.5-turbo-0613") is actually callable
# on this OpenAI account, before relying on it for anything. Uses the
# models.list endpoint, which OpenAI does not bill for -- but it's still a
# real network call to a paid API's account, so it's gated the same way as
# any other run per the project's working agreement: run this, don't guess.
#
# Usage (only after getting a go-ahead for this specific check):
#   python check_reader_availability.py --i-have-approval
import argparse

import config
import reader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-have-approval",
        action="store_true",
        help="confirms a human approved this specific network call",
    )
    args = parser.parse_args()
    if not args.i_have_approval:
        raise SystemExit(
            "Refusing to run: this calls the OpenAI API (models.list -- "
            "free, but real). Pass --i-have-approval after getting a "
            "go-ahead."
        )

    available = reader.check_model_available(config.READER_MODEL)
    if available:
        print(f"OK: {config.READER_MODEL} is callable on this account.")
    else:
        print(
            f"NOT AVAILABLE: {config.READER_MODEL} was not found in this "
            "account's model list. STOP here -- do not substitute a newer "
            "snapshot without asking first; that's a decision for the "
            "user, since it means accepting some drift from the paper's "
            "published numbers."
        )
