"""
Dataset inspection tool.

  python data/inspect.py --data data/train/data.jsonl            # validate + show 5 random samples
  python data/inspect.py --data data/train/data.jsonl --n 10     # show 10 random samples
  python data/inspect.py --data data/train/data.jsonl --validate-only
"""

import json
import random
import argparse
from dataclasses import dataclass, field

# ANSI colours
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREY   = "\033[90m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

VALID_LABELS = {"O", "B-CRED", "I-CRED"}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    total:    int = 0
    valid:    int = 0
    errors:   list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def invalid(self) -> int:
        return self.total - self.valid


def _check_bio(labels: list[str], line_num: int) -> list[str]:
    """Return a list of BIO violation messages (empty = clean)."""
    issues = []
    for i, label in enumerate(labels):
        if label == "I-CRED":
            prev = labels[i - 1] if i > 0 else "O"
            if prev not in ("B-CRED", "I-CRED"):
                issues.append(f"  line {line_num}: I-CRED at position {i} without preceding B-CRED/I-CRED")
    return issues


def validate(path: str) -> tuple[list[dict], ValidationResult]:
    """Parse and validate every row. Returns (valid_samples, result)."""
    result  = ValidationResult()
    samples = []

    with open(path) as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            result.total += 1

            # JSON parseable?
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                result.errors.append(f"  line {line_num}: invalid JSON — {e}")
                continue

            # Required keys
            if "tokens" not in row or "labels" not in row:
                result.errors.append(f"  line {line_num}: missing 'tokens' or 'labels' key")
                continue

            tokens = row["tokens"]
            labels = row["labels"]

            # Both lists?
            if not isinstance(tokens, list) or not isinstance(labels, list):
                result.errors.append(f"  line {line_num}: 'tokens' and 'labels' must be lists")
                continue

            # Same length?
            if len(tokens) != len(labels):
                result.errors.append(
                    f"  line {line_num}: length mismatch — "
                    f"{len(tokens)} tokens vs {len(labels)} labels"
                )
                continue

            # Valid label values?
            bad = [l for l in labels if l not in VALID_LABELS]
            if bad:
                result.errors.append(
                    f"  line {line_num}: unknown label(s) {set(bad)}"
                )
                continue

            # No empty tokens?
            empties = [i for i, t in enumerate(tokens) if not isinstance(t, str) or not t.strip()]
            if empties:
                result.warnings.append(
                    f"  line {line_num}: empty/non-string token(s) at positions {empties}"
                )

            # BIO consistency
            bio_issues = _check_bio(labels, line_num)
            result.warnings.extend(bio_issues)

            result.valid += 1
            samples.append(row)

    return samples, result


def print_validation_report(result: ValidationResult, path: str) -> None:
    print(f"{_BOLD}Validation report — {path}{_RESET}")
    print(f"  Total rows  : {result.total}")
    print(f"  Valid       : {result.valid}")

    if result.invalid:
        print(f"  {_RED}Invalid     : {result.invalid}{_RESET}")

    positives = 0  # computed later from samples — placeholder
    if result.errors:
        print(f"\n{_RED}Errors ({len(result.errors)}):{_RESET}")
        for e in result.errors:
            print(e)
    if result.warnings:
        print(f"\n{_YELLOW}Warnings ({len(result.warnings)}):{_RESET}")
        for w in result.warnings:
            print(w)
    if not result.errors and not result.warnings:
        print(f"  {_YELLOW}No errors or warnings.{_RESET}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_sample(sample: dict, index: int) -> None:
    tokens = sample["tokens"]
    labels = sample["labels"]
    meta   = sample.get("meta", {})

    cred_type = meta.get("credential_type", meta.get("theme", "negative"))
    context   = meta.get("context", "?")
    is_neg    = meta.get("negative", False)

    header = f"Sample #{index}  [{context}]  {'NEGATIVE' if is_neg else cred_type}"
    print(f"\n{_BOLD}{header}{_RESET}")
    print("─" * min(len(header) + 2, 80))

    # Reconstruct and colour the code line by line
    lines: list[list[tuple[str, str]]] = [[]]   # list of lines, each a list of (token, label)
    for tok, lbl in zip(tokens, labels):
        lines[-1].append((tok, lbl))
        if "\n" in tok:
            lines.append([])

    for line in lines:
        for tok, lbl in line:
            if lbl == "B-CRED":
                print(f"{_RED}{tok}{_RESET}", end=" ")
            elif lbl == "I-CRED":
                print(f"{_RED}{tok}{_RESET}", end=" ")
            else:
                print(tok, end=" ")
        print()

    # Credential summary
    cred_tokens = [t for t, l in zip(tokens, labels) if l != "O"]
    if cred_tokens:
        print(f"{_GREY}  credential → {' '.join(cred_tokens)}{_RESET}")
    else:
        print(f"{_GREY}  (no credential — all O labels){_RESET}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and inspect the NER dataset.")
    parser.add_argument("--data",          type=str, default="data/train/data.jsonl")
    parser.add_argument("--n",             type=int, default=5,
                        help="Number of random samples to render (default: 5)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate, skip rendering")
    parser.add_argument("--seed",          type=int, default=None,
                        help="Random seed for sample selection")
    args = parser.parse_args()

    samples, result = validate(args.data)
    print_validation_report(result, args.data)

    if args.validate_only or not samples:
        raise SystemExit(0)

    # Dataset stats
    positives = sum(1 for s in samples if "B-CRED" in s["labels"])
    negatives = len(samples) - positives
    print(f"\n{_BOLD}Dataset stats{_RESET}")
    print(f"  Positive (has credential) : {positives}")
    print(f"  Negative (clean code)     : {negatives}")

    # Render n random samples
    if args.seed is not None:
        random.seed(args.seed)
    chosen = random.sample(samples, min(args.n, len(samples)))
    print(f"\n{_BOLD}Rendering {len(chosen)} random sample(s){_RESET}")

    for i, sample in enumerate(chosen, start=1):
        _render_sample(sample, i)