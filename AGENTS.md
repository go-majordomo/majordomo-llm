# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `src/majordomo_llm` (typed, async-first). Keep new modules under this package.
- Tests live in `tests` and mirror the package layout (e.g., `tests/test_factory.py`).
- Examples are in `examples/`. Packaging output goes to `dist/`.
- Configuration: provider defaults in `src/majordomo_llm/llm_config.yaml`. Do not hardcode secrets.

## Build, Test, and Development Commands
- Environment (recommended):
  - `uv sync --all-extras` — install dev + optional deps.
  - Alternative: `pip install -e .[dev]`.
- Run tests: `uv run pytest`
- Coverage: `uv run pytest --cov=src/majordomo_llm --cov-report=term-missing`
- Type check: `uv run mypy src/majordomo_llm`
- Lint: `uv run ruff check src/majordomo_llm`
- Format: `uv run ruff format` (use `ruff check --fix` for autofixes)
- Build artifacts: `uv build` (hatchling backend)

## Coding Style & Naming Conventions
- Python 3.12+, 4‑space indentation, max line length 100 (ruff config).
- Follow ruff rules E,F,I,UP,B,SIM; keep imports sorted.
- Type hints required; mypy runs in strict mode.
- Names: modules `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE`.
- Public API: avoid breaking changes to `majordomo_llm` exports without discussion.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio`.
- Place tests under `tests/` named `test_*.py`; async tests are supported.
- Prefer deterministic, provider‑mocked tests over live API calls.
- Include coverage for new code paths and failure cases.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope first line (e.g., "feat(factory): add provider aliasing").
- Include rationale in body and reference issues (e.g., `Closes #123`).
- PRs: clear description, screenshots or logs when relevant, test coverage, and notes on API changes.
- Update `CHANGELOG.md` for user‑visible changes.

## Security & Configuration Tips
- Configure API keys via environment variables or `.env` (see README). Never commit secrets.
- When logging, redact sensitive content; prefer test fixtures over real prompts/keys.
- Document any new env vars in `README.md` and add safe defaults.
