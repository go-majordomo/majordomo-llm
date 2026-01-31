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

Notes
- Order defines priority; only `ProviderError` triggers a fallback.
- Consider mixed providers to diversify outages and quota limits.
