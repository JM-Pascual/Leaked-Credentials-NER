"""
CLI for running credential NER inference on a code file or inline snippet.

Usage:
    python predict.py --file path/to/file.py
    python predict.py --text "API_KEY = 'ghp_abc123...'"
    cat file.py | python predict.py
"""

import argparse
import sys

from src.inferencer import Inferencer

_RESET = "\033[0m"
_RED   = "\033[91m"
_BOLD  = "\033[1m"


def render(predictions: list[tuple[str, str]]) -> None:
    for token, label in predictions:
        if label in ("B-CRED", "I-CRED"):
            print(f"{_RED}{_BOLD}{token}{_RESET}", end=" ")
        else:
            print(token, end=" ")
    print()

    spans = [token for token, label in predictions if label in ("B-CRED", "I-CRED")]
    print()
    if spans:
        print(f"Detected {len(spans)} credential token(s):")
        for token in spans:
            print(f"  {_RED}{token}{_RESET}")
    else:
        print("No credentials detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect hardcoded credentials in code.")
    parser.add_argument("--checkpoint", default="checkpoints/cred-ner",
                        help="Path to the model checkpoint directory")
    parser.add_argument("--file", help="Path to a source code file")
    parser.add_argument("--text", help="Inline code snippet as a string")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()

    inferencer  = Inferencer(args.checkpoint)
    predictions = inferencer.predict(text)
    render(predictions)