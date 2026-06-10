"""Shared test fixtures for majordomo-llm tests."""

from unittest.mock import MagicMock

import pytest

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
    tool_block.name = "CountryInfo"
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
    response.output_text = '{"name":"Japan","capital":"Tokyo","population":125000000}'
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


# --- Streaming Mock Fixtures ---


@pytest.fixture
def mock_openai_stream_events():
    """Mock OpenAI streaming events."""
    delta_event1 = MagicMock()
    delta_event1.type = "response.output_text.delta"
    delta_event1.delta = "Hello"

    delta_event2 = MagicMock()
    delta_event2.type = "response.output_text.delta"
    delta_event2.delta = " world"

    completed_event = MagicMock()
    completed_event.type = "response.completed"
    completed_event.response.usage.input_tokens = 20
    completed_event.response.usage.output_tokens = 8
    completed_event.response.usage.input_tokens_details.cached_tokens = 0

    async def stream():
        yield delta_event1
        yield delta_event2
        yield completed_event

    return stream()


@pytest.fixture
def mock_anthropic_stream_events():
    """Mock Anthropic streaming events."""
    msg_start = MagicMock()
    msg_start.type = "message_start"
    msg_start.message.usage.input_tokens = 25
    msg_start.message.usage.cache_read_input_tokens = 0

    delta1 = MagicMock()
    delta1.type = "content_block_delta"
    delta1.delta.type = "text_delta"
    delta1.delta.text = "Hello"

    delta2 = MagicMock()
    delta2.type = "content_block_delta"
    delta2.delta.type = "text_delta"
    delta2.delta.text = " world"

    msg_delta = MagicMock()
    msg_delta.type = "message_delta"
    msg_delta.usage.output_tokens = 10

    async def stream():
        yield msg_start
        yield delta1
        yield delta2
        yield msg_delta

    return stream()


@pytest.fixture
def mock_gemini_stream_chunks():
    """Mock Gemini streaming chunks."""
    chunk1 = MagicMock()
    chunk1.text = "Hello"
    chunk1.usage_metadata = None

    chunk2 = MagicMock()
    chunk2.text = " world"
    chunk2.usage_metadata.prompt_token_count = 15
    chunk2.usage_metadata.candidates_token_count = 5

    async def stream():
        yield chunk1
        yield chunk2

    return stream()


@pytest.fixture
def mock_deepseek_stream_chunks():
    """Mock DeepSeek streaming chunks."""
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello"
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = " world"
    chunk2.usage = None

    final_chunk = MagicMock()
    final_chunk.choices = []
    final_chunk.usage.prompt_tokens = 20
    final_chunk.usage.completion_tokens = 8
    final_chunk.usage.prompt_tokens_details = None

    async def stream():
        yield chunk1
        yield chunk2
        yield final_chunk

    return stream()


@pytest.fixture
def mock_fireworks_stream_chunks():
    """Mock Fireworks streaming chunks."""
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello"
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = " world"
    chunk2.usage = None

    final_chunk = MagicMock()
    final_chunk.choices = []
    final_chunk.usage.prompt_tokens = 20
    final_chunk.usage.completion_tokens = 8
    final_chunk.usage.prompt_tokens_details = None

    async def stream():
        yield chunk1
        yield chunk2
        yield final_chunk

    return stream()


@pytest.fixture
def mock_together_stream_chunks():
    """Mock Together streaming chunks."""
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello"
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = " world"
    chunk2.usage = None

    final_chunk = MagicMock()
    final_chunk.choices = []
    final_chunk.usage.prompt_tokens = 20
    final_chunk.usage.completion_tokens = 8
    final_chunk.usage.prompt_tokens_details = None

    async def stream():
        yield chunk1
        yield chunk2
        yield final_chunk

    return stream()


@pytest.fixture
def mock_cohere_stream_events():
    """Mock Cohere streaming events."""
    event1 = MagicMock()
    event1.type = "content-delta"
    event1.delta.message.content.text = "Hello"

    event2 = MagicMock()
    event2.type = "content-delta"
    event2.delta.message.content.text = " world"

    end_event = MagicMock()
    end_event.type = "message-end"
    end_event.delta.usage.tokens.input_tokens = 20
    end_event.delta.usage.tokens.output_tokens = 8

    async def stream():
        yield event1
        yield event2
        yield end_event

    return stream()
