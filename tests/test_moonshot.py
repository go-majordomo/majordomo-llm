"""Tests for the Moonshot provider.

The shared request/response machinery is covered once in
test_openai_compatible.py; this module asserts only Moonshot-specific wiring.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers import Moonshot

MODEL_ID = "kimi-k3"


@pytest.fixture
def moonshot_llm():
    """Create a Moonshot instance with a mocked client."""
    with patch("majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"):
        return Moonshot(
            model=MODEL_ID,
            input_cost=3.00,
            output_cost=15.00,
            api_key="test-key",
        )


@pytest.fixture
def text_response():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Moonshot says hello!"
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 8
    response.usage.prompt_tokens_details = None
    return response


class TestMoonshotWiring:
    """Tests for endpoint, provider name, and key resolution."""

    def test_provider_name(self, moonshot_llm):
        assert moonshot_llm.provider == "moonshot"

    def test_default_base_url(self):
        with patch(
            "majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"
        ) as mock_client:
            Moonshot(model=MODEL_ID, input_cost=3.00, output_cost=15.00, api_key="k")

        assert mock_client.call_args.kwargs["base_url"] == "https://api.moonshot.ai/v1"

    def test_resolves_key_from_env(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", "from-env")
        with patch(
            "majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"
        ) as mock_client:
            Moonshot(model=MODEL_ID, input_cost=3.00, output_cost=15.00)

        assert mock_client.call_args.kwargs["api_key"] == "from-env"

    def test_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        with pytest.raises(ConfigurationError) as exc_info:
            Moonshot(model=MODEL_ID, input_cost=3.00, output_cost=15.00)

        assert "MOONSHOT_API_KEY" in str(exc_info.value)

    def test_gateway_header_uses_provider_name(self):
        with patch("majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"):
            llm = Moonshot(
                model=MODEL_ID,
                input_cost=3.00,
                output_cost=15.00,
                api_key="k",
                base_url="https://gateway.test/v1",
            )

        assert llm.default_headers["x-majordomo-provider"] == "moonshot"


class TestMoonshotGetResponse:
    """End-to-end text call against a mocked client."""

    async def test_returns_text_content(self, moonshot_llm, text_response):
        moonshot_llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        response = await moonshot_llm.get_response("Say hello")

        assert response.content == "Moonshot says hello!"
        assert response.input_tokens == 20
        assert response.output_tokens == 8
        assert response.total_cost > 0

    async def test_passes_model_id_verbatim(self, moonshot_llm, text_response):
        moonshot_llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await moonshot_llm.get_response("Say hello")

        assert moonshot_llm.client.chat.completions.create.call_args.kwargs["model"] == MODEL_ID
