# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_example.py

# Run a single test
uv run pytest tests/test_example.py::test_function_name

# Type checking
uv run mypy src/majordomo_llm

# Linting
uv run ruff check src/majordomo_llm

# Fix linting issues
uv run ruff check --fix src/majordomo_llm
```

## Architecture

This library provides a unified async interface for LLM providers (OpenAI, Anthropic, Gemini, DeepSeek, Cohere) with automatic cost tracking and structured output support.

### Core Components

- **`base.py`**: Abstract `LLM` base class defining the interface (`get_response`, `get_json_response`, `get_structured_json_response`). Response dataclasses (`LLMResponse`, `LLMJSONResponse`, `LLMStructuredResponse`) inherit from `Usage` for cost/token tracking.

- **`factory.py`**: `get_llm_instance(provider, model)` factory function and `LLM_CONFIG` dict containing model configurations (costs per million tokens, feature flags like `supports_temperature_top_p`).

- **`providers/`**: Provider implementations extending `LLM`. Each provider:
  - Uses its native async client
  - Implements `get_response()` with automatic retries via tenacity
  - Overrides `_get_structured_response()` for provider-specific structured output (e.g., Anthropic uses tool calling)

### Key Patterns

- All LLM methods are async and return response objects with embedded usage metrics
- Retry logic uses exponential backoff: `@retry(wait=wait_random_exponential(min=0.2, max=1), stop=stop_after_attempt(3))`
- Costs are calculated per million tokens using `TOKENS_PER_MILLION = 1_000_000`
- Pydantic models are used for structured output schemas via `model_json_schema()`

### Environment Variables

- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GEMINI_API_KEY` - Google Gemini API key
- `DEEPSEEK_API_KEY` - DeepSeek API key
- `CO_API_KEY` - Cohere API key
