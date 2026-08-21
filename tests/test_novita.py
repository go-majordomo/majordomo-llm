"""Tests for the Novita provider.

The shared request/response machinery is covered once in
test_openai_compatible.py; this module asserts only Novita-specific wiring.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from majordomo_llm.exceptions import ConfigurationError
from majordomo_llm.providers import Novita

MODEL_ID = "zai-org/glm-5.2"


@pytest.fixture
def novita_llm():
    """Create a Novita instance with a mocked client."""
    with patch("majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"):
        return Novita(
            model=MODEL_ID,
            input_cost=1.40,
            output_cost=4.40,
            api_key="test-key",
        )


@pytest.fixture
def text_response():
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Novita says hello!"
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 8
    response.usage.prompt_tokens_details = None
    return response


class TestNovitaWiring:
    """Tests for endpoint, provider name, and key resolution."""

    def test_provider_name(self, novita_llm):
        assert novita_llm.provider == "novita"

    def test_default_base_url(self):
        with patch(
            "majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"
        ) as mock_client:
            Novita(model=MODEL_ID, input_cost=1.40, output_cost=4.40, api_key="k")

        assert mock_client.call_args.kwargs["base_url"] == "https://api.novita.ai/openai/v1"

    def test_resolves_key_from_env(self, monkeypatch):
        monkeypatch.setenv("NOVITA_API_KEY", "from-env")
        with patch(
            "majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"
        ) as mock_client:
            Novita(model=MODEL_ID, input_cost=1.40, output_cost=4.40)

        assert mock_client.call_args.kwargs["api_key"] == "from-env"

    def test_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("NOVITA_API_KEY", raising=False)
        with pytest.raises(ConfigurationError) as exc_info:
            Novita(model=MODEL_ID, input_cost=1.40, output_cost=4.40)

        assert "NOVITA_API_KEY" in str(exc_info.value)

    def test_gateway_header_uses_provider_name(self):
        with patch("majordomo_llm.providers._openai_compatible.openai.AsyncOpenAI"):
            llm = Novita(
                model=MODEL_ID,
                input_cost=1.40,
                output_cost=4.40,
                api_key="k",
                base_url="https://gateway.test/v1",
            )

        assert llm.default_headers["x-majordomo-provider"] == "novita"


class TestNovitaGetResponse:
    """End-to-end text call against a mocked client."""

    async def test_returns_text_content(self, novita_llm, text_response):
        novita_llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        response = await novita_llm.get_response("Say hello")

        assert response.content == "Novita says hello!"
        assert response.input_tokens == 20
        assert response.output_tokens == 8
        assert response.total_cost > 0

    async def test_passes_model_id_verbatim(self, novita_llm, text_response):
        novita_llm.client.chat.completions.create = AsyncMock(return_value=text_response)

        await novita_llm.get_response("Say hello")

        assert novita_llm.client.chat.completions.create.call_args.kwargs["model"] == MODEL_ID
