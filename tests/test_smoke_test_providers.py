"""Tests for provider smoke-test capability helpers."""

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from majordomo_llm.factory import LLM_CONFIG, get_supported_providers
from scripts.smoke_test_providers import (
    EXTRA_MODELS,
    FAIL,
    GATEWAY_ROUTED_PROVIDERS,
    OK,
    PROVIDER_API_KEY_ENV,
    SKIP,
    _canonical_model,
    _pass2_models,
    _run_cell,
    _run_image,
)


def _vision_model(content: str) -> SimpleNamespace:
    response = SimpleNamespace(
        content=content,
        input_tokens=12,
        output_tokens=1,
        total_cost=0.0001,
    )
    return SimpleNamespace(
        supports_image_input=True,
        get_response=AsyncMock(return_value=response),
    )


async def test_run_image_submits_valid_png_and_records_usage() -> None:
    llm = _vision_model("BLUE")

    result = await _run_image(llm, None)

    assert result.status == OK
    assert result.input_tokens == 12
    assert result.output_tokens == 1
    assert result.total_cost == pytest.approx(0.0001)
    image_input = llm.get_response.call_args.kwargs["images"][0]
    assert image_input.media_type == "image/png"
    with Image.open(io.BytesIO(image_input.data)) as decoded:
        decoded.load()
        assert decoded.size == (32, 32)


@pytest.mark.parametrize("content", ["", "The image is red."])
async def test_run_image_rejects_empty_or_wrong_answers(content: str) -> None:
    result = await _run_image(_vision_model(content), None)

    assert result.status == FAIL
    assert "blue" in result.error


async def test_run_cell_skips_models_without_image_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = SimpleNamespace(
        supports_image_input=False,
        get_response=AsyncMock(),
    )
    monkeypatch.setattr("scripts.smoke_test_providers._build_llm", lambda *args, **kwargs: llm)

    result = await _run_cell(
        "openai",
        "text-only",
        "image",
        "capability-matrix",
        via_steward=False,
        gateway_url="http://localhost:7680",
        gateway_key=None,
        run_id="test-run",
    )

    assert result.status == SKIP
    llm.get_response.assert_not_awaited()


def test_every_direct_provider_has_an_api_key_env() -> None:
    """A provider added to the factory but not here crashed the sweep with KeyError."""
    direct = set(get_supported_providers()) - GATEWAY_ROUTED_PROVIDERS
    assert direct - set(PROVIDER_API_KEY_ENV) == set()


def test_extra_models_are_all_registered() -> None:
    for provider, models in EXTRA_MODELS.items():
        registered = LLM_CONFIG[provider]["models"]
        assert [m for m in models if m not in registered] == [], provider


def test_pass2_all_models_covers_everything_but_the_canonical() -> None:
    models = _pass2_models("anthropic", all_models=True)

    assert _canonical_model("anthropic") not in models
    assert len(models) == len(LLM_CONFIG["anthropic"]["models"]) - 1
    assert _pass2_models("anthropic", all_models=False) == EXTRA_MODELS["anthropic"]
