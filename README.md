# majordomo-llm

[![PyPI version](https://badge.fury.io/py/majordomo-llm.svg)](https://badge.fury.io/py/majordomo-llm)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified Python interface for multiple LLM providers with automatic cost tracking, retry logic, and structured output support.

## Features

- **Unified API** - Same interface for OpenAI, Anthropic (Claude), and Google Gemini
- **Cost Tracking** - Automatic calculation of input/output token costs per request
- **Structured Outputs** - Native support for Pydantic models as response schemas
- **Automatic Retries** - Built-in exponential backoff retry logic using tenacity
- **Async First** - Fully async/await compatible for high-performance applications
- **Type Safe** - Complete type annotations and `py.typed` marker for IDE support

## Installation

```bash
pip install majordomo-llm
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add majordomo-llm
```

## Quick Start

### Basic Text Response

```python
import asyncio
from majordomo_llm import get_llm_instance

async def main():
    # Create an LLM instance
    llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")

    # Get a response
    response = await llm.get_response(
        user_prompt="What is the capital of France?",
        system_prompt="You are a helpful geography assistant.",
    )

    print(response.content)
    print(f"Tokens: {response.input_tokens} in, {response.output_tokens} out")
    print(f"Cost: ${response.total_cost:.6f}")

asyncio.run(main())
```

### JSON Response

```python
response = await llm.get_json_response(
    user_prompt="List the top 3 largest countries by area as JSON",
    system_prompt="Respond with valid JSON only.",
)

# response.content is a parsed Python dict
for country in response.content["countries"]:
    print(country["name"])
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel

class CountryInfo(BaseModel):
    name: str
    capital: str
    population: int
    area_km2: float

response = await llm.get_structured_json_response(
    response_model=CountryInfo,
    user_prompt="Give me information about Japan",
)

# response.content is a validated CountryInfo instance
country = response.content
print(f"{country.name}: {country.capital}, pop. {country.population:,}")
```

## Configuration

### Environment Variables

Set API keys for the providers you want to use:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic (Claude)
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GEMINI_API_KEY="..."
```

### Available Models

#### OpenAI
- `gpt-5`, `gpt-5-mini`, `gpt-5-nano`
- `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`

#### Anthropic
- `claude-sonnet-4-5-20250929`, `claude-opus-4-1-20250805`
- `claude-opus-4-20250514`, `claude-sonnet-4-20250514`
- `claude-3-7-sonnet-latest`, `claude-3-5-haiku-latest`

#### Gemini
- `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- `gemini-2.0-flash`, `gemini-2.0-flash-lite`

## API Reference

### Factory Functions

#### `get_llm_instance(provider: str, model: str) -> LLM`

Create an LLM instance for the specified provider and model.

```python
from majordomo_llm import get_llm_instance

llm = get_llm_instance("openai", "gpt-4o")
```

### LLM Methods

All LLM instances support these async methods:

#### `get_response(user_prompt, system_prompt=None, temperature=0.3, top_p=1.0) -> LLMResponse`

Get a plain text response.

#### `get_json_response(user_prompt, system_prompt=None, temperature=0.3, top_p=1.0) -> LLMJSONResponse`

Get a JSON response (automatically parsed).

#### `get_structured_json_response(response_model, user_prompt, system_prompt=None, temperature=0.3, top_p=1.0) -> LLMStructuredResponse`

Get a response validated against a Pydantic model.

### Response Objects

All response objects include usage metrics:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` / `dict` / `BaseModel` | The response content |
| `input_tokens` | `int` | Number of input tokens |
| `output_tokens` | `int` | Number of output tokens |
| `cached_tokens` | `int` | Number of cached tokens (if applicable) |
| `input_cost` | `float` | Cost for input tokens (USD) |
| `output_cost` | `float` | Cost for output tokens (USD) |
| `total_cost` | `float` | Total cost (USD) |
| `response_time` | `float` | Response time in seconds |

## Advanced Usage

### Direct Provider Access

You can also instantiate providers directly for more control:

```python
from majordomo_llm import Anthropic

llm = Anthropic(
    model="claude-sonnet-4-20250514",
    input_cost=3.0,    # per million tokens
    output_cost=15.0,  # per million tokens
)
```

### Web Search (Anthropic)

Enable web search for supported Claude models:

```python
from majordomo_llm.providers.anthropic import Anthropic

llm = Anthropic(
    model="claude-sonnet-4-5-20250929",
    input_cost=3.0,
    output_cost=15.0,
    use_web_search=True,
)
```

## Development

### Setup

```bash
git clone https://github.com/superset-studio/majordomo-llm.git
cd majordomo-llm
uv sync --all-extras
```

### Running Tests

```bash
uv run pytest
```

### Type Checking

```bash
uv run mypy src/majordomo_llm
```

### Linting

```bash
uv run ruff check src/majordomo_llm
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
