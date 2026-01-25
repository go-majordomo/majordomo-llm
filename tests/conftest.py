"""Shared test fixtures for majordomo-llm tests."""

import pytest
from unittest.mock import MagicMock


# --- Anthropic Mock Fixtures ---


@pytest.fixture
def mock_anthropic_text_response():
    """Mock Anthropic text response."""
    response = MagicMock()
    response.content = [MagicMock(type="text", text="Paris is the capital of France.")]
    response.usage.input_tokens = 25
    response.usage.output_tokens = 10
    response.usage.cache_read_input_tokens = 0
    response.stop_reason = "end_turn"
    return response


@pytest.fixture
def mock_anthropic_tool_response():
    """Mock Anthropic tool use response for structured output."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "structured_response"
    tool_block.input = {"name": "France", "capital": "Paris", "population": 67000000}

    response = MagicMock()
    response.content = [tool_block]
    response.usage.input_tokens = 50
    response.usage.output_tokens = 30
    response.usage.cache_read_input_tokens = 5
    response.stop_reason = "tool_use"
    return response


# --- OpenAI Mock Fixtures ---


@pytest.fixture
def mock_openai_text_response():
    """Mock OpenAI text response."""
    response = MagicMock()
    response.output_text = "The answer is 42."
    response.usage.input_tokens = 20
    response.usage.output_tokens = 8
    response.usage.input_tokens_details.cached_tokens = 0
    return response


@pytest.fixture
def mock_openai_structured_response():
    """Mock OpenAI structured response."""
    response = MagicMock()
    response.output_parsed = {"name": "Japan", "capital": "Tokyo", "population": 125000000}
    response.usage.input_tokens = 40
    response.usage.output_tokens = 25
    response.usage.input_tokens_details.cached_tokens = 10
    return response


# --- Gemini Mock Fixtures ---


@pytest.fixture
def mock_gemini_text_response():
    """Mock Gemini text response."""
    response = MagicMock()
    response.text = "Gemini says hello!"
    response.usage_metadata.prompt_token_count = 15
    response.usage_metadata.candidates_token_count = 5
    return response


@pytest.fixture
def mock_gemini_json_response():
    """Mock Gemini JSON response."""
    response = MagicMock()
    response.text = '{"status": "success", "value": 123}'
    response.usage_metadata.prompt_token_count = 20
    response.usage_metadata.candidates_token_count = 12
    return response
