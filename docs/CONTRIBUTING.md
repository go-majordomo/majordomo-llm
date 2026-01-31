# Docs Authoring Checklist

Use this quick checklist when adding or updating docs.

## Prerequisites
- Install dev deps once: `uv add --dev mkdocs mkdocs-material mkdocstrings[python] pymdown-extensions`
- Preview locally: `uv run mkdocs serve`
- Build locally (optional): `uv run mkdocs build`

## Add a New Page
- Create a file under `docs/` (e.g., `docs/your-page.md`).
- Register it in `mkdocs.yml` under `nav`:
  - Example: `- Your Page: your-page.md`
- Keep titles short (<= 60 chars). One H1 per page.

## Add a New Recipe
- Place under `docs/recipes/` (e.g., `recipes/batch-processing.md`).
- Add to the Recipes section in `mkdocs.yml`.

## API Reference (mkdocstrings)
- Add directives to `docs/api/*.md`:
  - `::: majordomo_llm.module.ClassName`
- Ensure the target is importable from `src/` and avoids import-time side effects.
- Prefer Google-style docstrings for good rendering.

## Style & Content
- Write concise, task-focused sections; use lists for steps.
- Use fenced code blocks with language hints (```python, ```bash).
- Admonitions: `!!! note`, `!!! warning` for important tips.
- Link internally with relative paths; place images in `docs/assets/`.

## Publish
- Push to `main` to deploy via GitHub Pages workflow.
- For Read the Docs, ensure the project is imported once; RTD builds on push.
