# Proxy Routing & Custom Headers

Route LLM requests through a gateway or proxy and attach custom HTTP headers.

## Route Through a Gateway

Point any provider at a custom base URL:

```python
from majordomo_llm import get_llm_instance

llm = get_llm_instance(
    "anthropic", "claude-sonnet-4-20250514",
    base_url="https://gateway.example.com",
    default_headers={"X-Majordomo-Key": "mdm_key_here"},
)

response = await llm.get_response("Hello!")
```

The request goes to `gateway.example.com` instead of `api.anthropic.com`, with the `X-Majordomo-Key` header attached.

## Per-Request Headers

Add headers to individual calls with `extra_headers`. These are merged with `default_headers`, with per-request values winning on conflict:

```python
llm = get_llm_instance(
    "openai", "gpt-4.1",
    base_url="https://gateway.example.com",
    default_headers={
        "X-Majordomo-Key": "mdm_key_here",
        "X-Majordomo-Feature": "search",
    },
)

# This request sends all three headers
response = await llm.get_response(
    "Find recent news about AI",
    extra_headers={"X-Majordomo-Request-Id": "req_abc123"},
)
```

## Override a Default Header

Per-request headers take precedence over instance headers with the same key:

```python
llm = get_llm_instance(
    "anthropic", "claude-sonnet-4-20250514",
    base_url="https://gateway.example.com",
    default_headers={
        "X-Majordomo-Key": "mdm_key_here",
        "X-Majordomo-Feature": "search",
    },
)

# Override X-Majordomo-Feature for this one request
response = await llm.get_response(
    "Translate this to Spanish",
    extra_headers={"X-Majordomo-Feature": "translation"},
)
```

## Cascade Through a Gateway

Route all cascade providers through the same gateway:

```python
from majordomo_llm import LLMCascade

cascade = LLMCascade(
    [
        ("anthropic", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4.1"),
        ("gemini", "gemini-2.5-flash"),
    ],
    base_url="https://gateway.example.com",
    default_headers={"X-Majordomo-Key": "mdm_key_here"},
)

# All three providers route through the gateway
response = await cascade.get_response(
    "Hello!",
    extra_headers={"X-Majordomo-Request-Id": "req_abc123"},
)
```

## With Logging

`LoggingLLM` passes `extra_headers` through to the wrapped LLM:

```python
from majordomo_llm import get_llm_instance
from majordomo_llm.logging import LoggingLLM, SqliteAdapter, FileStorageAdapter

llm = get_llm_instance(
    "anthropic", "claude-sonnet-4-20250514",
    base_url="https://gateway.example.com",
    default_headers={"X-Majordomo-Key": "mdm_key_here"},
)

db = await SqliteAdapter.create("llm_logs.db")
storage = await FileStorageAdapter.create("./request_logs")
logged_llm = LoggingLLM(llm, db, storage)

# extra_headers flows through the logging wrapper to the provider
response = await logged_llm.get_response(
    "Hello!",
    extra_headers={"X-Majordomo-Request-Id": "req_abc123"},
)
```

## With Streaming

`extra_headers` works with streaming responses:

```python
stream = await llm.get_response_stream(
    "Explain quantum computing",
    extra_headers={"X-Majordomo-Request-Id": "req_stream_456"},
)
async for chunk in stream:
    print(chunk, end="", flush=True)
```

## With Structured Outputs

`extra_headers` works with structured output responses:

```python
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]

response = await llm.get_structured_json_response(
    response_model=Summary,
    user_prompt="Summarize the benefits of async programming",
    extra_headers={"X-Majordomo-Request-Id": "req_struct_789"},
)
```

## Notes

- `base_url` and `default_headers` are optional on both `get_llm_instance()` and `LLMCascade`. When omitted, requests go directly to the provider.
- `extra_headers` is optional on every API method (`get_response`, `get_response_stream`, `get_json_response`, `get_structured_json_response`). When omitted, only `default_headers` are sent.
- For DeepSeek, a custom `base_url` overrides the default `https://api.deepseek.com` endpoint.
- All providers are supported. The header merging logic is handled internally per SDK.
