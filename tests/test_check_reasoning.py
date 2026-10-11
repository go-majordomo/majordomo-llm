"""Tests for the pure helpers in scripts/check_reasoning.py."""

import pytest

from majordomo_llm.factory import LLM_CONFIG
from scripts.check_reasoning import HOSTS, SETTINGS, Result, configured_label, format_row


def test_every_host_is_a_registered_provider() -> None:
    assert set(HOSTS) - set(LLM_CONFIG) == set()


@pytest.mark.parametrize(
    ("attrs", "host", "expected"),
    [
        ({}, "together", "default (nothing sent)"),
        ({"reasoning_effort": "low"}, "baseten", "reasoning_effort low"),
        ({"thinking": "disabled"}, "together", "thinking disabled"),
        # Fireworks rejects the two together, and its provider lets disabled win.
        ({"reasoning_effort": "low", "thinking": "disabled"}, "fireworks", "thinking disabled"),
        ({"reasoning_effort": "low", "thinking": "disabled"}, "together", "reasoning_effort low"),
        ({"thinking": "enabled"}, "fireworks", "thinking enabled"),
    ],
)
def test_configured_label_mirrors_what_the_provider_sends(attrs, host, expected) -> None:
    assert configured_label(attrs, host) == expected


def test_every_configured_model_maps_to_a_probed_setting() -> None:
    """Otherwise the sweep cannot mark which row a model is actually using."""
    labels = {label for label, _ in SETTINGS}
    unmatched = [
        (host, model)
        for host in HOSTS
        for model, attrs in LLM_CONFIG[host]["models"].items()
        if configured_label(attrs, host) not in labels
    ]
    assert unmatched == []


def test_streaming_only_is_detected_from_the_host_error() -> None:
    rejected = Result(
        "default (nothing sent)",
        400,
        detail='This model only supports streaming. Set "stream": true.',
    )
    other = Result("default (nothing sent)", 400, detail="Input validation error")

    assert rejected.streaming_only is True
    assert other.streaming_only is False


def test_format_row_marks_the_configured_setting() -> None:
    result = Result(
        "reasoning_effort low", 200, output_tokens=7, reasoning="2", seconds=0.71, detail="billing"
    )

    row = format_row(result, configured=True)

    assert "<- configured" in row
    assert "billing" in row
    assert "<- configured" not in format_row(result, configured=False)
