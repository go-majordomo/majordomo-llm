"""Tests for the Nebius provider.

The shared request/response machinery is covered once in
test_openai_compatible.py; this module asserts only Nebius-specific wiring.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers import Nebius

MODEL_ID = "moonshotai/Kimi-K3"


@pytest.fixture
def nebius_llm():
    """Create a Nebius instance with a mocked client."""
    with patch("majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"):
        return Nebius(
            model=MODEL_ID,
            input_cost=3.00,
            output_cost=15.00,
            api_key="test-key",
        )


@pytest.fixture
def text_response():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Nebius says hello!"
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 8
    response.usage.prompt_tokens_details = None
    return response


class TestNebiusWiring:
    """Tests for endpoint, provider name, and key resolution."""

    def test_provider_name(self, nebius_llm):
        assert nebius_llm.provider == "nebius"

    def test_default_base_url(self):
        with patch(
            "majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"
        ) as mock_client:
            Nebius(model=MODEL_ID, input_cost=3.00, output_cost=15.00, api_key="k")

        assert mock_client.call_args.kwargs["base_url"] == "https://api.tokenfactory.nebius.com/v1"

    def test_resolves_key_from_env(self, monkeypatch):
        monkeypatch.setenv("NEBIUS_API_KEY", "from-env")
        with patch(
            "majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"
        ) as mock_client:
            Nebius(model=MODEL_ID, input_cost=3.00, output_cost=15.00)

        assert mock_client.call_args.kwargs["api_key"] == "from-env"

    def test_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
        with pytest.raises(ConfigurationError) as exc_info:
            Nebius(model=MODEL_ID, input_cost=3.00, output_cost=15.00)

        assert "NEBIUS_API_KEY" in str(exc_info.value)

    def test_gateway_header_uses_provider_name(self):
        with patch("majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"):
            llm = Nebius(
                model=MODEL_ID,
                input_cost=3.00,
                output_cost=15.00,
                api_key="k",
                base_url="https://gateway.test/v1",
            )

        assert llm.default_headers["x-majordomo-provider"] == "nebius"


class TestNebiusGetResponse:
    """End-to-end text call against a mocked client."""

    async def test_returns_text_content(self, nebius_llm, text_response):
        nebius_llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        response = await nebius_llm.get_response("Say hello")

        assert response.content == "Nebius says hello!"
        assert response.input_tokens == 20
        assert response.output_tokens == 8
        assert response.total_cost > 0

    async def test_passes_model_id_verbatim(self, nebius_llm, text_response):
        nebius_llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await nebius_llm.get_response("Say hello")

        assert nebius_llm.client.chat.completions.create.call_args.kwargs["model"] == MODEL_ID
