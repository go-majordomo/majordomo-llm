# Cascade Failover with LLMCascade

Automatically fall back across providers when one fails.

```python
from majordomo_llm import LLMCascade

cascade = LLMCascade([
    ("anthropic", "claude-sonnet-4-20250514"),  # Primary
    ("openai", "gpt-4o"),                        # Fallback
    ("gemini", "gemini-2.5-flash"),              # Last resort
])

resp = await cascade.get_response("Hello!")
print(resp.content)
```

Streaming is also supported:

```python
stream = await cascade.get_response_stream("Hello!")
async for chunk in stream:
    print(chunk, end="")
```

Route all cascade providers through a gateway:

```python
cascade = LLMCascade(
    [
        ("anthropic", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4o"),
        ("gemini", "gemini-2.5-flash"),
    ],
    base_url="https://gateway.example.com",
    default_headers={"X-Majordomo-Key": "mdm_key_here"},
)

resp = await cascade.get_response(
    "Hello!",
    extra_headers={"X-Request-Id": "req_123"},
)
```

Notes

- Order defines priority; only `ProviderError` triggers a fallback.
- For streaming, fallback happens on stream creation errors only. Mid-stream errors propagate to the caller.
- Consider mixed providers to diversify outages and quota limits.
