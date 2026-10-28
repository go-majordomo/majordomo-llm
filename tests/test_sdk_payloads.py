"""Request payloads built through each provider's real SDK client.

The provider test modules replace the SDK client with a mock, so the SDK never
sees the arguments the library passes it. That hides the failures this module
exists to catch: an SDK type that can no longer be constructed (anthropic 2.x
turned ``MessageParam`` into a ``Union`` alias, which raised ``TypeError``
before any request was sent), or an argument the SDK renamed or now rejects.

Here each provider keeps its real SDK client, wired to a ``MockTransport`` that
records the outgoing request and replies with a minimal valid response. A test
passes only if the SDK accepts every argument, serializes the request, and
parses the reply into a library response.

The transport comes from whichever HTTP library the SDK is built on. anthropic
1.0 and openai 3.0 moved from ``httpx`` to its fork ``httpx2`` and reject a
client from the other library; cohere and google-genai still use ``httpx``.

Bedrock is not covered: aiobotocore does not take an httpx-style transport, so
it stays on the mocks in ``test_bedrock.py``.
"""

import base64
import importlib
import json
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import anthropic
import cohere
import httpx
import openai
import pytest
from google import genai
from google.genai import types as genai_types

from majordomo_llm import ImageInput, LLMCascade, get_llm_instance
from majordomo_llm.providers import Anthropic

COUNTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "capital": {"type": "string"},
    },
    "required": ["name", "capital"],
}
COUNTRY = {"name": "France", "capital": "Paris"}

# Smallest valid PNG: a 1x1 transparent pixel.
PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


def _sdk_http_module(sdk: ModuleType) -> ModuleType:
    """Return the HTTP library (``httpx`` or ``httpx2``) a Stainless-built SDK uses."""
    for cls in sdk.DefaultAsyncHttpxClient.__mro__:
        if cls.__name__ == "AsyncClient":
            return importlib.import_module(cls.__module__.split(".")[0])
    raise AssertionError(f"{sdk.__name__} has no AsyncClient base class")


ANTHROPIC_HTTP = _sdk_http_module(anthropic)
OPENAI_HTTP = _sdk_http_module(openai)


@dataclass(frozen=True)
class Reply:
    """A canned HTTP reply, built in whichever HTTP library serves the request."""

    content: bytes
    headers: dict[str, str] = field(default_factory=lambda: {"content-type": "application/json"})
    status: int = 200


class RecordingTransport:
    """Replies to requests in order and keeps each request for inspection."""

    def __init__(self, *replies: Reply) -> None:
        self._replies = list(replies)
        self.requests: list[Any] = []
        self._http: ModuleType = httpx

    def __call__(self, request: Any) -> Any:
        self.requests.append(request)
        if not self._replies:
            raise AssertionError(f"Unexpected extra request to {request.url}")
        reply = self._replies.pop(0)
        return self._http.Response(reply.status, content=reply.content, headers=reply.headers)

    def client(self, http: ModuleType = httpx) -> Any:
        """Return an ``AsyncClient`` of ``http`` that sends every request here."""
        self._http = http
        return http.AsyncClient(transport=http.MockTransport(self))

    def body(self, index: int = 0) -> dict[str, Any]:
        result: dict[str, Any] = json.loads(self.requests[index].content)
        return result


def _json(payload: dict[str, Any]) -> Reply:
    return Reply(json.dumps(payload).encode())


def _sse(events: list[tuple[str, dict[str, Any]]]) -> Reply:
    body = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    return Reply(body.encode(), headers={"content-type": "text/event-stream"})


# --- Anthropic ---------------------------------------------------------------


def _anthropic_message(content: list[dict[str, Any]], stop_reason: str = "end_turn") -> Reply:
    return _json(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        }
    )


def _anthropic_text(text: str = "ok") -> Reply:
    return _anthropic_message([{"type": "text", "text": text}])


def _anthropic_tool_use(name: str, value: dict[str, Any]) -> Reply:
    return _anthropic_message(
        [{"type": "tool_use", "id": "toolu_test", "name": name, "input": value}],
        stop_reason="tool_use",
    )


def _with_anthropic_transport(llm: Anthropic, transport: RecordingTransport) -> Anthropic:
    llm.client = anthropic.AsyncAnthropic(
        api_key="test-key", http_client=transport.client(ANTHROPIC_HTTP)
    )
    return llm


def _anthropic(transport: RecordingTransport, model: str = "claude-haiku-5-5") -> Anthropic:
    llm = get_llm_instance("anthropic", model, api_key="test-key")
    assert isinstance(llm, Anthropic)
    return _with_anthropic_transport(llm, transport)


def _anthropic_direct(transport: RecordingTransport, **kwargs: Any) -> Anthropic:
    """Build an Anthropic provider with flags no configured model combines."""
    llm = Anthropic(
        model="claude-sonnet-5",
        input_cost=3.0,
        output_cost=15.0,
        api_key="test-key",
        **kwargs,
    )
    return _with_anthropic_transport(llm, transport)


def _anthropic_stream(text: str = "ok") -> Reply:
    """A complete Messages API event stream whose answer is ``text``."""
    return _sse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-test",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 10, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 2},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )


class TestAnthropicPayloads:
    async def test_text_request(self):
        transport = RecordingTransport(_anthropic_text())
        llm = _anthropic(transport)

        response = await llm.get_response("Reply with the word ok.", max_tokens=200)

        assert response.content == "ok"
        body = transport.body()
        assert body["messages"] == [{"role": "user", "content": "Reply with the word ok."}]
        assert body["max_tokens"] == 200
        assert body["system"][0]["type"] == "text"
        assert body["system"][0]["cache_control"] == {"type": "ephemeral"}

    async def test_image_request(self):
        transport = RecordingTransport(_anthropic_text("a pixel"))
        llm = _anthropic(transport)

        response = await llm.get_response(
            "Describe it.", images=(ImageInput(data=PNG_1X1, media_type="image/png"),)
        )

        assert response.content == "a pixel"
        [message] = transport.body()["messages"]
        assert message["role"] == "user"
        image_block, text_block = message["content"]
        assert image_block == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(PNG_1X1).decode("ascii"),
            },
        }
        assert text_block == {"type": "text", "text": "Describe it."}

    async def test_streaming_request(self):
        transport = RecordingTransport(_anthropic_stream())
        llm = _anthropic(transport)

        stream = await llm.get_response_stream("Reply with the word ok.")
        response = await stream.collect()

        assert response.content == "ok"
        body = transport.body()
        assert body["stream"] is True
        assert body["messages"] == [{"role": "user", "content": "Reply with the word ok."}]

    async def test_native_structured_request(self):
        transport = RecordingTransport(_anthropic_text(json.dumps(COUNTRY)))
        llm = _anthropic(transport)
        assert llm.supports_structured_outputs

        response = await llm.get_json_schema_response(
            "Tell me about France.", COUNTRY_SCHEMA, schema_name="CountryInfo"
        )

        assert json.loads(response.content) == COUNTRY
        body = transport.body()
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert "tools" not in body

    async def test_forced_tool_structured_request(self):
        transport = RecordingTransport(_anthropic_tool_use("CountryInfo", COUNTRY))
        llm = _anthropic_direct(transport, supports_structured_outputs=False)

        response = await llm.get_json_schema_response(
            "Tell me about France.", COUNTRY_SCHEMA, schema_name="CountryInfo"
        )

        assert json.loads(response.content) == COUNTRY
        body = transport.body()
        assert body["tool_choice"] == {"type": "tool", "name": "CountryInfo"}
        [tool] = body["tools"]
        assert tool["name"] == "CountryInfo"
        assert tool["input_schema"]["properties"] == COUNTRY_SCHEMA["properties"]

    async def test_web_search_structured_request_carries_prose_turn(self):
        """The two-turn path sends the SDK's own response blocks back to it."""
        transport = RecordingTransport(
            _anthropic_text("France's capital is Paris."),
            _anthropic_tool_use("CountryInfo", COUNTRY),
        )
        llm = _anthropic_direct(
            transport,
            use_web_search=True,
            supports_structured_outputs=True,
            supports_forced_tool_choice=False,
        )

        response = await llm.get_json_schema_response(
            "Tell me about France.", COUNTRY_SCHEMA, schema_name="CountryInfo"
        )

        assert json.loads(response.content) == COUNTRY
        first, final = transport.body(0), transport.body(1)
        assert first["tool_choice"] == {"type": "auto"}
        assert {tool["name"] for tool in first["tools"]} == {"CountryInfo", "web_search"}
        assert [m["role"] for m in final["messages"]] == ["user", "assistant", "user"]
        assert final["messages"][1]["content"] == [
            {"type": "text", "text": "France's capital is Paris."}
        ]
        assert final["tool_choice"] == {"type": "auto"}
        assert [tool["name"] for tool in final["tools"]] == ["CountryInfo"]


# --- OpenAI (Responses API) --------------------------------------------------


def _openai_response(text: str = "ok") -> Reply:
    return _json(
        {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-test",
            "output": [
                {
                    "type": "message",
                    "id": "msg_test",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    )


def _openai(transport: RecordingTransport) -> Any:
    llm = get_llm_instance("openai", "gpt-6.1-sol", api_key="test-key")
    llm.client = openai.AsyncOpenAI(api_key="test-key", http_client=transport.client(OPENAI_HTTP))
    return llm


class TestOpenAIPayloads:
    async def test_text_request(self):
        transport = RecordingTransport(_openai_response())
        llm = _openai(transport)

        response = await llm.get_response("Reply with the word ok.")

        assert response.content == "ok"
        assert transport.requests[0].url.path.endswith("/responses")
        assert transport.body()["model"] == "gpt-6.1-sol"

    async def test_structured_request(self):
        transport = RecordingTransport(_openai_response(json.dumps(COUNTRY)))
        llm = _openai(transport)

        response = await llm.get_json_schema_response(
            "Tell me about France.", COUNTRY_SCHEMA, schema_name="CountryInfo"
        )

        assert json.loads(response.content) == COUNTRY
        text_format = transport.body()["text"]["format"]
        assert text_format["type"] == "json_schema"
        assert text_format["name"] == "CountryInfo"


# --- OpenAI-compatible chat completions (Fireworks, Together, Baseten) -------


def _chat_completion(text: str = "ok") -> Reply:
    return _json(
        {
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
    )


CHAT_COMPLETION_MODELS = [
    ("fireworks", "accounts/fireworks/models/glm-5p3"),
    ("together", "zai-org/GLM-5.3"),
    ("baseten", "zai-org/GLM-5.3"),
]


def _chat_provider(transport: RecordingTransport, provider: str, model: str) -> Any:
    llm = get_llm_instance(provider, model, api_key="test-key")
    llm.client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://test.invalid/v1",
        http_client=transport.client(OPENAI_HTTP),
    )
    return llm


class TestChatCompletionPayloads:
    @pytest.mark.parametrize(("provider", "model"), CHAT_COMPLETION_MODELS)
    async def test_text_request(self, provider, model):
        transport = RecordingTransport(_chat_completion())
        llm = _chat_provider(transport, provider, model)

        response = await llm.get_response("Reply with the word ok.")

        assert response.content == "ok"
        body = transport.body()
        assert transport.requests[0].url.path.endswith("/chat/completions")
        assert body["model"] == model
        assert body["messages"][-1] == {"role": "user", "content": "Reply with the word ok."}

    @pytest.mark.parametrize(("provider", "model"), CHAT_COMPLETION_MODELS)
    async def test_structured_request(self, provider, model):
        transport = RecordingTransport(_chat_completion(json.dumps(COUNTRY)))
        llm = _chat_provider(transport, provider, model)

        response = await llm.get_json_schema_response(
            "Tell me about France.", COUNTRY_SCHEMA, schema_name="CountryInfo"
        )

        assert json.loads(response.content) == COUNTRY
        assert transport.body()["response_format"]["type"] in {"json_schema", "json_object"}


# --- Cohere ------------------------------------------------------------------


def _cohere_chat(text: str = "ok") -> Reply:
    return _json(
        {
            "id": "chat_test",
            "finish_reason": "COMPLETE",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
            "usage": {
                "billed_units": {"input_tokens": 10, "output_tokens": 2},
                "tokens": {"input_tokens": 10, "output_tokens": 2},
            },
        }
    )


def _cohere(transport: RecordingTransport) -> Any:
    llm = get_llm_instance("cohere", "command-a-03-2025", api_key="test-key")
    llm.client = cohere.AsyncClientV2(api_key="test-key", httpx_client=transport.client())
    return llm


class TestCoherePayloads:
    async def test_text_request(self):
        transport = RecordingTransport(_cohere_chat())
        llm = _cohere(transport)

        response = await llm.get_response("Reply with the word ok.")

        assert response.content == "ok"
        body = transport.body()
        assert transport.requests[0].url.path.endswith("/v2/chat")
        assert body["model"] == "command-a-03-2025"
        assert body["messages"][-1]["role"] == "user"

    async def test_structured_request(self):
        transport = RecordingTransport(_cohere_chat(json.dumps(COUNTRY)))
        llm = _cohere(transport)

        response = await llm.get_json_schema_response(
            "Tell me about France.", COUNTRY_SCHEMA, schema_name="CountryInfo"
        )

        assert json.loads(response.content) == COUNTRY
        assert transport.body()["response_format"]["type"] == "json_object"


# --- Gemini ------------------------------------------------------------------


def _gemini_content(text: str = "ok") -> Reply:
    return _json(
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 2,
                "totalTokenCount": 12,
            },
        }
    )


def _gemini(transport: RecordingTransport) -> Any:
    llm = get_llm_instance("gemini", "gemini-3.8-flash", api_key="test-key")
    llm.client = genai.Client(
        api_key="test-key",
        http_options=genai_types.HttpOptions(httpx_async_client=transport.client()),
    )
    return llm


class TestGeminiPayloads:
    async def test_text_request(self):
        transport = RecordingTransport(_gemini_content())
        llm = _gemini(transport)

        response = await llm.get_response("Reply with the word ok.")

        assert response.content == "ok"
        assert transport.requests[0].url.path.endswith("gemini-3.8-flash:generateContent")
        body = transport.body()
        assert body["contents"][-1]["parts"][-1] == {"text": "Reply with the word ok."}

    async def test_structured_request(self):
        transport = RecordingTransport(_gemini_content(json.dumps(COUNTRY)))
        llm = _gemini(transport)

        response = await llm.get_json_schema_response(
            "Tell me about France.", COUNTRY_SCHEMA, schema_name="CountryInfo"
        )

        assert json.loads(response.content) == COUNTRY
        assert transport.body()["generationConfig"]["responseMimeType"] == "application/json"


# --- Cascade, end to end -----------------------------------------------------


async def test_cascade_falls_through_when_a_provider_cannot_build_its_request(monkeypatch):
    """The reported failure: a request that cannot be built must not end the cascade.

    The Anthropic member raises the ``TypeError`` anthropic 2.x produced, from
    inside request construction, before anything is sent. The cascade must move
    on to OpenAI, whose real SDK client then serves the call.
    """

    def unbuildable(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("Cannot instantiate typing.Union")

    monkeypatch.setattr("majordomo_llm.providers.anthropic._anthropic_user_message", unbuildable)
    anthropic_transport = RecordingTransport()
    openai_transport = RecordingTransport(_openai_response("from openai"))

    cascade = LLMCascade(
        [("anthropic", "claude-haiku-5-5"), ("openai", "gpt-6.1-sol")], api_key="test-key"
    )
    _with_anthropic_transport(cascade.llms[0], anthropic_transport)
    cascade.llms[1].client = openai.AsyncOpenAI(
        api_key="test-key", http_client=openai_transport.client(OPENAI_HTTP)
    )

    response = await cascade.get_response("Reply with the word ok.")

    assert response.content == "from openai"
    assert anthropic_transport.requests == []
    assert len(openai_transport.requests) == 1


# --- Which provider and model answered ---------------------------------------


DIRECT_CALLS = [
    ("anthropic", "claude-haiku-5-5", lambda t: _anthropic(t), _anthropic_text),
    ("openai", "gpt-6.1-sol", _openai, _openai_response),
    ("cohere", "command-a-03-2025", _cohere, _cohere_chat),
    ("gemini", "gemini-3.8-flash", _gemini, _gemini_content),
    *[
        (provider, model, lambda t, p=provider, m=model: _chat_provider(t, p, m), _chat_completion)
        for provider, model in CHAT_COMPLETION_MODELS
    ],
]


class TestResponseProvenance:
    """Every response names the provider and model that served it."""

    @pytest.mark.parametrize(
        ("provider", "model", "build", "reply"),
        DIRECT_CALLS,
        ids=[f"{provider}/{model}" for provider, model, _, _ in DIRECT_CALLS],
    )
    async def test_direct_call(self, provider, model, build, reply):
        llm = build(RecordingTransport(reply()))

        response = await llm.get_response("Reply with the word ok.")

        assert (response.provider, response.model) == (provider, model)
        assert response.attempted == [(provider, model)]

    async def test_cascade_reports_the_primary_when_it_answers(self):
        anthropic_transport = RecordingTransport(_anthropic_text("from anthropic"))
        openai_transport = RecordingTransport()
        cascade = LLMCascade(
            [("anthropic", "claude-sonnet-5-5"), ("openai", "gpt-5.5")], api_key="test-key"
        )
        _with_anthropic_transport(cascade.llms[0], anthropic_transport)
        cascade.llms[1].client = openai.AsyncOpenAI(
            api_key="test-key", http_client=openai_transport.client(OPENAI_HTTP)
        )

        response = await cascade.get_response("hi")

        assert response.content == "from anthropic"
        assert (response.provider, response.model) == ("anthropic", "claude-sonnet-5-5")
        assert response.attempted == [("anthropic", "claude-sonnet-5-5")]
        assert openai_transport.requests == []

    async def test_cascade_reports_the_fallback_when_the_primary_fails(self):
        # A 400 is not retried by the SDK or the library, so the cascade moves on at once.
        rejected = Reply(
            json.dumps(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "no"}}
            ).encode(),
            status=400,
        )
        anthropic_transport = RecordingTransport(rejected)
        openai_transport = RecordingTransport(_openai_response("from openai"))
        cascade = LLMCascade(
            [("anthropic", "claude-sonnet-5-5"), ("openai", "gpt-5.5")], api_key="test-key"
        )
        _with_anthropic_transport(cascade.llms[0], anthropic_transport)
        cascade.llms[1].client = openai.AsyncOpenAI(
            api_key="test-key", http_client=openai_transport.client(OPENAI_HTTP)
        )

        response = await cascade.get_response("hi")

        assert response.content == "from openai"
        assert (response.provider, response.model) == ("openai", "gpt-5.5")
        assert response.attempted == [
            ("anthropic", "claude-sonnet-5-5"),
            ("openai", "gpt-5.5"),
        ]

    async def test_stream_reports_provider_after_it_finishes(self):
        llm = _anthropic(RecordingTransport(_anthropic_stream()), model="claude-sonnet-5-5")

        stream = await llm.get_response_stream("Reply with the word ok.")
        async for _ in stream:
            pass

        assert (stream.provider, stream.model) == ("anthropic", "claude-sonnet-5-5")
        assert stream.usage is not None
        assert (stream.usage.provider, stream.usage.model) == ("anthropic", "claude-sonnet-5-5")
        assert stream.usage.attempted == [("anthropic", "claude-sonnet-5-5")]
