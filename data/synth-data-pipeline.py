"""
Synthetic credential NER data pipeline.

Strategy: always seeded.
  1. Generate a regex-valid credential string locally.
  2. Prompt the LLM to embed it verbatim in a realistic code snippet.
  3. Find the exact credential string in the output and apply BIO labels.
  4. Skip samples where the credential cannot be located (LLM paraphrased it).

This guarantees that every written sample has correct, trustworthy labels.

Output: JSONL — each line is {"tokens": [...], "labels": [...], "meta": {...}}
Labels: B-CRED, I-CRED, O  (BIO scheme, whitespace-tokenized)
"""

import os
import json
import random
import string
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from dotenv import load_dotenv
import requests

load_dotenv()

OPENROUTER_API_KEY = os.environ["OPEN_ROUTER_API_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "z-ai/glm-4.5-air:free",
    "poolside/laguna-xs.2:free",
    "openrouter/owl-alpha"
]

_DATA_DIR = Path(__file__).parent

def _load_json_list(filename: str, key: str) -> list[str]:
    path = _DATA_DIR / filename
    with open(path) as f:
        return json.load(f)[key]

CONTEXTS        = _load_json_list("contexts.json",        "contexts")
NEGATIVE_THEMES = _load_json_list("negative_themes.json", "themes")

SAMPLE_DELAY_S      = 10    # seconds to wait between each generated sample
RETRY_INITIAL_WAIT  = 2    # seconds before the first retry attempt
RETRY_BACKOFF       = 2     # multiplier applied on each subsequent retry

MAX_TOKENS          = 2048  # max tokens per LLM response


# ---------------------------------------------------------------------------
# Credential generators
# ---------------------------------------------------------------------------

ALNUM      = string.ascii_letters + string.digits
B64        = ALNUM + "+/"
LOWER_SLUG = string.ascii_lowercase + string.digits + "_.-~"
PRINTABLE  = "".join(chr(c) for c in range(0x21, 0x7f) if chr(c) not in "\"'`\\")
HEX        = string.hexdigits[:16]


def _rand(chars: str, n: int) -> str:
    return "".join(random.choices(chars, k=n))

def _rand_range(chars: str, lo: int, hi: int) -> str:
    return _rand(chars, random.randint(lo, hi))


def gen_aws_secret() -> str:
    return _rand(B64, 40)

def gen_azure_client_secret() -> str:
    return _rand(LOWER_SLUG, 34)

def gen_dropbox_key() -> str:
    return "sl." + _rand_range(ALNUM + "-_", 130, 140)

def gen_facebook_token() -> str:
    return "EAACEdEose0cBA" + _rand_range(ALNUM, 30, 60)

def gen_generic_secret() -> str:
    return _rand_range(PRINTABLE, 16, 48)

def gen_github_token() -> str:
    prefix = random.choice(["ghp", "gho", "ghu", "ghs", "ghr"])
    return f"{prefix}_" + _rand_range(ALNUM, 36, 52)

def gen_slack_token() -> str:
    prefix = random.choice(["xoxb", "xoxp", "xapp", "xoxa", "xoxr"])
    mid    = _rand(string.digits, random.randint(10, 13))
    suffix = _rand_range(ALNUM + "-", 20, 40)
    return f"{prefix}-{mid}-{suffix}"

def gen_stripe_key() -> str:
    kind = random.choice(["sk", "rk"])
    return f"{kind}_live_" + _rand(ALNUM, 24)

def gen_twitter_token() -> str:
    lead = str(random.randint(10, 99))
    body = _rand(string.digits, random.randint(5, 15))
    tail = _rand(ALNUM, 40)
    return f"{lead}{body}-{tail}"

def gen_google_oauth_id() -> str:
    prefix = _rand(string.digits, random.randint(6, 12))
    body   = _rand(ALNUM + "_", 32)
    return f"{prefix}-{body}.apps.googleusercontent.com"

def gen_ssh_private_key() -> str:
    body = "\n".join(_rand(B64, 64) for _ in range(8))
    return f"-----BEGIN RSA PRIVATE KEY-----\n{body}\n-----END RSA PRIVATE KEY-----"

def gen_db_connection_url() -> str:
    scheme = random.choice(["postgresql", "mysql", "mongodb", "redis"])
    user   = _rand(string.ascii_lowercase, random.randint(4, 10))
    pw     = _rand(ALNUM + "!@#$%", random.randint(12, 24))
    host   = random.choice(["db.internal", "prod-db.example.com", "10.0.0.42"])
    ports  = {"postgresql": 5432, "mysql": 3306, "mongodb": 27017, "redis": 6379}
    db     = _rand(string.ascii_lowercase, random.randint(4, 8))
    if scheme == "redis":
        return f"redis://:{pw}@{host}:{ports[scheme]}"
    return f"{scheme}://{user}:{pw}@{host}:{ports[scheme]}/{db}"

def gen_api_key_hex() -> str:
    return _rand(HEX, 32)

def gen_bearer_token() -> str:
    parts = [_rand(B64, random.randint(20, 40)) for _ in range(3)]
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Credential type registry
# ---------------------------------------------------------------------------

@dataclass
class CredentialType:
    name: str
    generator: Callable[[], str]
    description: str


CREDENTIAL_TYPES: list[CredentialType] = [
    CredentialType("AWS API Secret",          gen_aws_secret,
                   "an AWS secret access key"),
    CredentialType("Azure Client Secret",     gen_azure_client_secret,
                   "an Azure client secret"),
    CredentialType("Dropbox API Key",         gen_dropbox_key,
                   "a Dropbox API key"),
    CredentialType("Facebook Access Token",   gen_facebook_token,
                   "a Facebook Graph API access token"),
    CredentialType("Generic API Secret",      gen_generic_secret,
                   "a generic API secret token"),
    CredentialType("GitHub Token",            gen_github_token,
                   "a GitHub personal access token"),
    CredentialType("Slack Token",             gen_slack_token,
                   "a Slack bot/user OAuth token"),
    CredentialType("Stripe API Key",          gen_stripe_key,
                   "a Stripe live-mode API key"),
    CredentialType("Twitter Access Token",    gen_twitter_token,
                   "a Twitter API access token"),
    CredentialType("Google OAuth Client ID",  gen_google_oauth_id,
                   "a Google OAuth 2.0 client ID"),
    CredentialType("SSH Private Key",         gen_ssh_private_key,
                   "an RSA SSH private key (PEM format)"),
    CredentialType("Database Connection URL", gen_db_connection_url,
                   "a database connection URL with embedded credentials"),
    CredentialType("Hex API Key",             gen_api_key_hex,
                   "a 32-character hexadecimal API key"),
    CredentialType("Bearer Token",            gen_bearer_token,
                   "a JWT-style bearer token"),
]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(cred_type: CredentialType, credential: str, context: str) -> str:
    return (
        f"You are helping build a machine learning dataset for a security NER model that detects "
        f"hardcoded credentials in source code. All values are randomly generated and entirely fake — "
        f"they do not correspond to any real account or service.\n\n"
        f"Generate a short, realistic {context} snippet that contains "
        f"the following FAKE, SYNTHETIC {cred_type.description} exactly as shown below:\n\n"
        f"    {credential}\n\n"
        "Rules:\n"
        "- The value must appear VERBATIM in the code, character for character.\n"
        "- Assign it to a plausible variable name (e.g. API_KEY, password, token).\n"
        "- Surround it with realistic boilerplate that resembles code accidentally committed to a repo.\n"
        "- Output ONLY the code block, no prose or explanation."
    )


def build_negative_prompt(context: str, theme: str) -> str:
    return (
        f"You are helping build a machine learning dataset for a security NER model.\n\n"
        f"Generate a short, realistic {context} snippet that {theme}. "
        f"The code must contain NO hardcoded secrets, tokens, passwords, or API keys — "
        f"any credentials must be read from environment variables, config files, or secret managers. "
        "Output ONLY the code block, no prose or explanation."
    )


def build_freeform_prompt(context: str) -> str:
    return (
        f"Write a short, realistic {context} snippet from any ordinary software development "
        f"scenario of your choosing — a utility function, data model, CLI tool, config block, "
        f"API client, test, build script, anything plausible. "
        f"Do not include any hardcoded secrets, tokens, passwords, or API keys. "
        "Output ONLY the code block, no prose or explanation."
    )


# ---------------------------------------------------------------------------
# OpenRouter API call
# ---------------------------------------------------------------------------

def call_llm(prompt: str, model: str, max_retries: int = 6) -> Optional[str]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": prompt}],
        "max_tokens":  600,
        "temperature": 0.85,
    }
    for attempt in range(max_retries):
        resp = None
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers,
                                 json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            wait = RETRY_INITIAL_WAIT * (RETRY_BACKOFF ** attempt)
            body = ""
            if resp is not None:
                try:
                    body = f" — API response: {resp.json()}"
                except Exception:
                    body = f" — raw response: {resp.text[:200]}"
            print(f"  [warn] attempt {attempt+1} failed ({exc}){body}, retrying in {wait}s…")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Whitespace tokenization & BIO labeling
# ---------------------------------------------------------------------------

def tokenize(text: str) -> tuple[list[str], list[int]]:
    """Returns (tokens, char_offsets) where char_offsets[i] is the start of tokens[i] in text."""
    tokens:  list[str] = []
    offsets: list[int] = []
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        tokens.append(text[i:j])
        offsets.append(i)
        i = j
    return tokens, offsets


def bio_label(text: str, tokens: list[str], offsets: list[int],
              credential: str) -> Optional[list[str]]:
    """
    Locate `credential` verbatim in `text` and assign BIO labels to overlapping tokens.
    Returns None if the credential string is not found.
    """
    cred_start = text.find(credential)
    if cred_start == -1:
        return None
    cred_end = cred_start + len(credential)

    labels = ["O"] * len(tokens)
    first  = True
    for i, (tok, start) in enumerate(zip(tokens, offsets)):
        end = start + len(tok)
        if end <= cred_start or start >= cred_end:
            continue
        labels[i] = "B-CRED" if first else "I-CRED"
        first = False

    return labels


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    tokens: list[str]
    labels: list[str]
    meta:   dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps({"tokens": self.tokens, "labels": self.labels, "meta": self.meta})


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def generate_sample(cred_type: CredentialType, model: str) -> Optional[Sample]:
    credential = cred_type.generator()
    context    = random.choice(CONTEXTS)
    prompt     = build_prompt(cred_type, credential, context)
    text       = call_llm(prompt, model)

    if text is None:
        print("  [skip] API call failed after retries")
        return None

    tokens, offsets = tokenize(text)
    labels = bio_label(text, tokens, offsets, credential)

    if labels is None:
        print("  [skip] credential not found verbatim in LLM output")
        print(f"  [debug] seeded credential : {credential[:80]}")
        print(f"  [debug] LLM response head : {text[:200]}")
        return None

    return Sample(
        tokens=tokens,
        labels=labels,
        meta={
            "model":           model,
            "credential_type": cred_type.name,
            "credential":      credential,
            "context":         context,
        },
    )


def generate_negative_sample(model: str, freeform_ratio: float = 0.3) -> Optional[Sample]:
    context = random.choice(CONTEXTS)

    if random.random() < freeform_ratio:
        prompt = build_freeform_prompt(context)
        theme  = "free-form"
    else:
        theme  = random.choice(NEGATIVE_THEMES)
        prompt = build_negative_prompt(context, theme)

    text = call_llm(prompt, model)

    if text is None:
        print("  [skip] API call failed after retries")
        return None

    tokens, _ = tokenize(text)
    labels    = ["O"] * len(tokens)

    return Sample(
        tokens=tokens,
        labels=labels,
        meta={
            "model":    model,
            "context":  context,
            "negative": True,
            "theme":    theme,
        },
    )


def run(
    n_samples: int,
    output_path: str,
    negative_ratio: float = 0.0,
    freeform_ratio: float = 0.3,
    append: bool = False,
) -> None:
    """
    negative_ratio: fraction of samples that will be negative (credential-free).
      e.g. 0.25 → 1 in 4 samples is a negative example.
    freeform_ratio: fraction of negative samples generated with the open-ended prompt
      (no prescribed theme). The rest use a themed prompt from negative_themes.json.
    append: if True, opens output_path in append mode instead of overwriting.
    """
    written = skipped = 0
    mode = "a" if append else "w"

    with open(output_path, mode) as out:
        for i in range(n_samples):
            model       = MODELS[i % len(MODELS)]
            is_negative = random.random() < negative_ratio

            if is_negative:
                print(f"[{i+1}/{n_samples}] NEGATIVE | {model}")
                sample = generate_negative_sample(model, freeform_ratio)
            else:
                cred_type = random.choice(CREDENTIAL_TYPES)
                print(f"[{i+1}/{n_samples}] {cred_type.name} | {model}")
                sample = generate_sample(cred_type, model)

            if sample is None:
                skipped += 1
                continue

            out.write(sample.to_jsonl() + "\n")
            written += 1
            print(f"  → written ({len(sample.tokens)} tokens)")

            time.sleep(SAMPLE_DELAY_S)

    print(f"\nDone. {written} written, {skipped} skipped → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate seeded credential NER data.")
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of snippets to generate (default: 50)")
    parser.add_argument("--output", type=str, default="data/synth_cred_ner.jsonl",
                        help="Output JSONL path (default: data/synth_cred_ner.jsonl)")
    parser.add_argument("--negative", type=float, default=0.0,
                        help="Fraction of samples that are negative/credential-free (default: 0.0)")
    parser.add_argument("--freeform", type=float, default=0.3,
                        help="Fraction of negative samples using the open-ended prompt (default: 0.3)")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    args = parser.parse_args()

    run(args.samples, args.output, args.negative, args.freeform, args.append)