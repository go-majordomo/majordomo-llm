# Contributing to majordomo-llm

Thank you for your interest in contributing to majordomo-llm! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/superset-studio/majordomo-llm.git
   cd majordomo-llm
   ```

2. Install dependencies (including dev dependencies):
   ```bash
   uv sync --all-extras
   ```

3. Set up API keys for testing (optional):
   ```bash
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   export GEMINI_API_KEY="..."
   ```

## Code Style

We use the following tools to maintain code quality:

- **ruff** - Fast Python linter and formatter
- **mypy** - Static type checking

### Running Linters

```bash
# Lint code
uv run ruff check src/majordomo_llm

# Format code
uv run ruff format src/majordomo_llm

# Type check
uv run mypy src/majordomo_llm
```

### Code Style Guidelines

- Use type annotations for all public functions and methods
- Write docstrings in Google style for all public APIs
- Keep line length to 100 characters
- Use `str | None` instead of `Optional[str]` (PEP 604)
- Prefer f-strings over `.format()` or `%` formatting

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=majordomo_llm

# Run specific test file
uv run pytest tests/test_factory.py
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use `pytest-asyncio` for async test functions
- Mock external API calls to avoid costs and flakiness

## Secrets & Pre-commit Hooks

- Use `.env.example` as a template for local development. Copy to `.env` and fill your keys:
  ```bash
  cp .env.example .env
  ```
  Never commit `.env`.

- Enable pre-commit hooks (via uvx) for hygiene checks:
  ```bash
  uvx pre-commit install
  uvx pre-commit run --all-files
  ```
  See `.pre-commit-config.yaml` for configured hooks (private-key detection, large-file check, whitespace, etc.).

## Pull Request Process

1. **Fork the repository** and create your branch from `main`

2. **Make your changes**:
   - Write clear, concise commit messages
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**:
   ```bash
   uv run ruff check src/majordomo_llm
   uv run ruff format src/majordomo_llm
   uv run mypy src/majordomo_llm
   uv run pytest
   ```

4. **Submit a pull request**:
   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure CI checks pass

## Adding a New Provider

To add support for a new LLM provider:

1. Create a new file in `src/majordomo_llm/providers/`

2. Implement the provider class inheriting from `LLM`:
   ```python
   import os
   from majordomo_llm.base import LLM, LLMResponse, LLMJSONResponse, T
   from majordomo_llm.exceptions import ConfigurationError

   class NewProvider(LLM):
       def __init__(
           self,
           model: str,
           input_cost: float,
           output_cost: float,
           *,
           api_key: str | None = None,
           api_key_alias: str | None = None,
       ):
           # Resolve API key before calling super().__init__
           resolved_api_key = api_key or os.environ.get("NEW_PROVIDER_API_KEY")
           if not resolved_api_key:
               raise ConfigurationError(
                   "API key not found. Set NEW_PROVIDER_API_KEY or pass api_key."
               )

           super().__init__(
               provider="new_provider",
               model=model,
               input_cost=input_cost,
               output_cost=output_cost,
               api_key=resolved_api_key,        # For hashed logging
               api_key_alias=api_key_alias,     # For human-readable logging
           )
           # Initialize client with resolved_api_key

       async def get_response(self, user_prompt: str, ...) -> LLMResponse:
           # Implement text response

       async def _get_structured_response(self, response_model: Type[T], ...) -> LLMJSONResponse:
           # Implement structured response (optional override)
   ```

3. Add model configurations to `LLM_CONFIG` in `factory.py`

4. Update the factory function to handle the new provider

5. Export the provider in `__init__.py`

6. Add tests and documentation

## Reporting Issues

When reporting issues, please include:

- Python version
- majordomo-llm version
- Operating system
- Minimal code example to reproduce
- Full error traceback

## Questions?

Feel free to open an issue for questions or discussions about the project.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
