#!/usr/bin/env python3
"""
Measure how each open-weight model responds to reasoning controls, live.

Usage:
    uv run python scripts/check_reasoning.py
    uv run python scripts/check_reasoning.py --provider together
    uv run python scripts/check_reasoning.py --provider baseten --model zai-org/GLM-5.2

Fireworks, Together and Baseten run most models with reasoning on, usually at the
top tier, and bill it as output tokens. ``llm_config.yaml`` sets
``reasoning_effort: low`` on each model where that was measured to lower it. The
values are not portable: the same setting cuts reasoning on one model, does
nothing on another, and switches reasoning ON where it is opt-in. This script is
how those entries were established, and how to re-check them when a host changes
a model.

For every model registered under a host it sends one trivial classification
prompt under each setting below and prints the HTTP status, output tokens,
reasoning tokens and latency. The row matching what ``llm_config.yaml`` sends for
that model is marked.

The requests go straight to each host's chat-completions endpoint rather than
through ``get_llm_instance``, so that settings the providers cannot send yet
(``reasoning.enabled``, ``chat_template_kwargs``) can be measured too.

Unlike ``check_max_tokens.py`` this runs inference: seven small calls per model,
each capped at ``PROBE_MAX_TOKENS`` output tokens. A full sweep is a few cents.
One sample per setting is noisy, so treat a small difference as no difference.

Credentials are read from the environment or a local ``.env``:
``FIREWORKS_API_KEY``, ``TOGETHER_API_KEY``, ``BASETEN_API_KEY``. A host whose key
is absent is skipped with one line.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from majordomo_llm.providers.baseten import Baseten  # noqa: E402
from majordomo_llm.providers.fireworks import Fireworks  # noqa: E402
from majordomo_llm.providers.together import Together  # noqa: E402

load_dotenv()

CONFIG_PATH = Path(__file__).parent.parent / "src" / "majordomo_llm" / "llm_config.yaml"

#: Host -> (chat-completions base URL, API key environment variable).
HOSTS: dict[str, tuple[str, str]] = {
    "fireworks": (Fireworks.FIREWORKS_BASE_URL, "FIREWORKS_API_KEY"),
    "together": (Together.TOGETHER_BASE_URL, "TOGETHER_API_KEY"),
    "baseten": (Baseten.DEFAULT_BASE_URL, Baseten.API_KEY_ENV),
}

#: Enough room for top-tier reasoning on a one-word answer without letting a
#: runaway model spend freely.
PROBE_MAX_TOKENS = 4096

#: A single request should answer in seconds; one model took 74s at high effort.
PROBE_TIMEOUT_SECONDS = 180.0

#: Label -> extra request-body fields. The first five are what a config entry or
#: a ``get_llm_instance`` override can send today. The last two are documented by
#: the hosts but not sendable by the providers.
SETTINGS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("default (nothing sent)", {}),
    ("reasoning_effort low", {"reasoning_effort": "low"}),
    ("reasoning_effort high", {"reasoning_effort": "high"}),
    ("reasoning_effort none", {"reasoning_effort": "none"}),
    ("thinking disabled", {"thinking": {"type": "disabled"}}),
    ("reasoning enabled=false", {"reasoning": {"enabled": False}}),
    ("enable_thinking=false", {"chat_template_kwargs": {"enable_thinking": False}}),
)

MESSAGES: list[Any] = [
    {
        "role": "system",
        "content": "Classify the support ticket as one word: billing, technical, or account.",
    },
    {"role": "user", "content": "Why was I charged $49.99 when the website showed $39.99?"},
]

DIVIDER = "═" * 92


@dataclass
class Result:
    label: str
    status: int | None
    output_tokens: int | None = None
    reasoning: str = "-"
    seconds: float | None = None
    detail: str = ""

    @property
    def streaming_only(self) -> bool:
        return self.status == 400 and "only supports streaming" in self.detail


def configured_label(attrs: dict[str, Any], host: str) -> str:
    """The SETTINGS label for what llm_config.yaml makes the provider send.

    Mirrors the providers: Fireworks lets ``thinking: disabled`` win over an
    effort; on every host an effort is otherwise what is sent.
    """
    effort = attrs.get("reasoning_effort")
    thinking = attrs.get("thinking")
    if thinking == "disabled" and (host == "fireworks" or effort is None):
        return "thinking disabled"
    if effort is not None:
        return f"reasoning_effort {effort}"
    if thinking is not None:
        return f"thinking {thinking}"
    return "default (nothing sent)"


def _reasoning_cell(response: Any) -> str:
    """Reasoning tokens if the host counts them, else whether it returned any text."""
    usage = response.usage
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    tokens = getattr(details, "reasoning_tokens", None) if details else None
    if tokens is not None:
        return str(tokens)
    message = response.choices[0].message
    extra = message.model_extra or {}
    # Hosts return reasoning text under different keys, and not all count it.
    if extra.get("reasoning_content") or extra.get("reasoning"):
        return "text"
    return "-"


def probe(client: openai.OpenAI, model: str, label: str, extra: dict[str, Any]) -> Result:
    """Send one request under one setting and summarize what came back."""
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=MESSAGES,
            max_tokens=PROBE_MAX_TOKENS,
            extra_body=extra,
        )
    except openai.APIStatusError as e:
        detail = " ".join(str(e.message).split())
        return Result(label, e.status_code, seconds=time.perf_counter() - start, detail=detail)
    except openai.APIConnectionError as e:
        return Result(label, None, detail=f"connection error: {e}")

    answer = " ".join((response.choices[0].message.content or "").split())
    return Result(
        label,
        200,
        output_tokens=response.usage.completion_tokens if response.usage else None,
        reasoning=_reasoning_cell(response),
        seconds=time.perf_counter() - start,
        detail=answer,
    )


def format_row(result: Result, configured: bool) -> str:
    status = "-" if result.status is None else str(result.status)
    out = "-" if result.output_tokens is None else str(result.output_tokens)
    secs = "-" if result.seconds is None else f"{result.seconds:.1f}"
    marker = "  <- configured" if configured else ""
    return (
        f"  {result.label:<24} {status:>4} {out:>6} {result.reasoning:>9} {secs:>6}  "
        f"{result.detail[:44]}{marker}"
    )


def check_model(client: openai.OpenAI, host: str, key: str, attrs: dict[str, Any]) -> None:
    configured = configured_label(attrs, host)
    # A profile entry re-registers an upstream model under another YAML key.
    model = attrs.get("model", key)
    print(f"\n{host} / {key}")

    first = probe(client, model, *SETTINGS[0])
    if first.streaming_only:
        print("  streaming only: non-streaming calls are rejected, not probed")
        return

    print(f"  {'setting':<24} {'http':>4} {'out':>6} {'reasoning':>9} {'secs':>6}  answer / error")
    print(format_row(first, first.label == configured), flush=True)
    for label, extra in SETTINGS[1:]:
        result = probe(client, model, label, extra)
        print(format_row(result, label == configured), flush=True)

    if configured not in {label for label, _ in SETTINGS}:
        print(f"  configured: {configured} (not one of the settings probed)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--provider", action="append", choices=list(HOSTS), help="probe one host (repeatable)"
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="ID",
        help="probe only this model ID (repeatable); must be registered under the host",
    )
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text())
    hosts = args.provider or list(HOSTS)
    probed = 0

    for host in hosts:
        base_url, key_env = HOSTS[host]
        key = os.environ.get(key_env)
        if not key:
            print(f"[skip] {host}: {key_env} not set", file=sys.stderr)
            continue
        models: dict[str, dict[str, Any]] = config[host]["models"]
        wanted = [m for m in models if not args.model or m in args.model]
        if not wanted:
            continue

        client = openai.OpenAI(
            api_key=key, base_url=base_url, timeout=PROBE_TIMEOUT_SECONDS, max_retries=0
        )
        print(f"\n{DIVIDER}")
        print(f"{host}: {len(wanted)} model(s), {len(SETTINGS)} settings each")
        print(DIVIDER)
        for model in wanted:
            check_model(client, host, model, models[model])
            probed += 1

    if probed == 0:
        print(
            "\nNothing probed. Set a host's API key in the environment or a local .env, "
            "and check that --model names a registered model.",
            file=sys.stderr,
        )
        return 2

    print(
        "\nCompare each model's configured row with 'default (nothing sent)'. Where the\n"
        "configured row is not clearly lower, or another row is far lower, the entry in\n"
        "llm_config.yaml needs another look. Record what you find in the comment on that\n"
        "host's block."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
