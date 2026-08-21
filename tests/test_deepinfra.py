"""Tests for the DeepInfra provider.

The shared request/response machinery is covered once in
test_openai_compatible.py; this module asserts only DeepInfra-specific wiring.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers import DeepInfra

MODEL_ID = "zai-org/GLM-5.2"


@pytest.fixture
def deepinfra_llm():
    """Create a DeepInfra instance with a mocked client."""
    with patch("majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"):
        return DeepInfra(
            model=MODEL_ID,
            input_cost=0.75,
            output_cost=2.40,
            api_key="test-key",
        )


@pytest.fixture
def text_response():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "DeepInfra says hello!"
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 8
    response.usage.prompt_tokens_details = None
    return response


class TestDeepInfraWiring:
    """Tests for endpoint, provider name, and key resolution."""

    def test_provider_name(self, deepinfra_llm):
        assert deepinfra_llm.provider == "deepinfra"

    def test_default_base_url(self):
        with patch(
            "majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"
        ) as mock_client:
            DeepInfra(model=MODEL_ID, input_cost=0.75, output_cost=2.40, api_key="k")

        assert mock_client.call_args.kwargs["base_url"] == "https://api.deepinfra.com/v1/openai"

    def test_resolves_key_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPINFRA_API_KEY", "from-env")
        with patch(
            "majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"
        ) as mock_client:
            DeepInfra(model=MODEL_ID, input_cost=0.75, output_cost=2.40)

        assert mock_client.call_args.kwargs["api_key"] == "from-env"

    def test_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
        with pytest.raises(ConfigurationError) as exc_info:
            DeepInfra(model=MODEL_ID, input_cost=0.75, output_cost=2.40)

        assert "DEEPINFRA_API_KEY" in str(exc_info.value)

    def test_gateway_header_uses_provider_name(self):
        with patch("majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"):
            llm = DeepInfra(
                model=MODEL_ID,
                input_cost=0.75,
                output_cost=2.40,
                api_key="k",
                base_url="https://gateway.test/v1",
            )

        assert llm.default_headers["x-majordomo-provider"] == "deepinfra"


class TestDeepInfraGetResponse:
    """End-to-end text call against a mocked client."""

    async def test_returns_text_content(self, deepinfra_llm, text_response):
        deepinfra_llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        response = await deepinfra_llm.get_response("Say hello")

        assert response.content == "DeepInfra says hello!"
        assert response.input_tokens == 20
        assert response.output_tokens == 8
        assert response.total_cost > 0

    async def test_passes_model_id_verbatim(self, deepinfra_llm, text_response):
        deepinfra_llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await deepinfra_llm.get_response("Say hello")

        assert deepinfra_llm.client.chat.completions.create.call_args.kwargs["model"] == MODEL_ID
