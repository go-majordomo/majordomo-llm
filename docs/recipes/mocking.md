# Offline Testing (Mocking Providers)

Avoid network calls in tests by stubbing provider methods.

```python
import pytest
from majordomo_llm import get_llm_instance
from majordomo_llm.base import LLMResponse

@pytest.mark.asyncio
async def test_mocked_response(monkeypatch):
    llm = get_llm_instance("openai", "gpt-4o")

    async def fake_get_response(*args, **kwargs):
        return LLMResponse(
            content="mocked",
            input_tokens=10,
            output_tokens=5,
            cached_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            response_time=0.01,
        )

    monkeypatch.setattr(llm, "get_response", fake_get_response)
    resp = await llm.get_response("hi")
    assert resp.content == "mocked"
```

Tips
- Patch `get_llm_instance` to return a test double for broader control.
- Use deterministic fixtures for prompts and outputs.
