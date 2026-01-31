# Structured Outputs with Pydantic

Validate responses to a Pydantic model using `get_structured_json_response`.

```python
from pydantic import BaseModel
from majordomo_llm import get_llm_instance

class Country(BaseModel):
    name: str
    capital: str
    population: int

llm = get_llm_instance("openai", "gpt-4o")
resp = await llm.get_structured_json_response(
    response_model=Country,
    user_prompt="Return info about Japan as JSON",
)
country: Country = resp.content
print(country.capital)
```

Notes
- Pydantic validates and coerces types; handle `ValidationError` for bad outputs.
- Anthropic/OpenAI may use provider-native structured output under the hood.
