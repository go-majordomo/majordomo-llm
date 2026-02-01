# Deprecation Automation

This document captures ideas for automating model deprecation handling in majordomo-llm.

## Problem Statement

LLM providers regularly deprecate and retire models. Currently, developers must:
1. Manually monitor deprecation announcements from each provider
2. Update their code to use replacement models before retirement dates
3. Risk service disruptions if they miss a deadline

This creates operational burden and reliability risk.

## Provider Deprecation Pages

| Provider | Deprecation Page |
|----------|------------------|
| OpenAI | https://platform.openai.com/docs/deprecations |
| Anthropic | https://platform.claude.com/docs/en/about-claude/model-deprecations |
| Google Gemini | https://ai.google.dev/gemini-api/docs/deprecations |

## Proposed Solution: Hybrid Approach

Combine config-based metadata, runtime handling, and automated monitoring.

### 1. Config Schema with Deprecation Metadata

Extend `llm_config.yaml` to include deprecation information:

```yaml
anthropic:
  models:
    claude-3-7-sonnet-20250219:
      input_cost: 3.00
      output_cost: 15.00
      status: deprecated  # active | deprecated | retired
      deprecated: 2025-10-28
      retirement: 2026-02-19
      replacement: claude-sonnet-4-5-20250929
      replacement_notes: "4.5 has extended thinking; adjust prompts if needed"
```

Fields:
- `status`: Current lifecycle state
- `deprecated`: Date when deprecation was announced
- `retirement`: Date when model stops working
- `replacement`: Recommended replacement model ID
- `replacement_notes`: Migration guidance (optional)

### 2. Runtime Behavior

Add deprecation handling to the LLM base class with configurable behavior:

```python
llm = get_llm_instance(
    "anthropic",
    "claude-3-7-sonnet-20250219",
    on_deprecation="warn"  # "strict" | "warn" | "auto"
)
```

| Mode | Before Retirement | After Retirement |
|------|-------------------|------------------|
| `strict` | Warn in response | Raise `DeprecatedModelError` |
| `warn` | Warn in response | Warn + auto-replace |
| `auto` | Silent | Silent auto-replace |

### 3. Response Object Changes

Extend response dataclasses to include deprecation information:

```python
@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    total_cost: float
    response_time: float

    # New deprecation fields
    requested_model: str | None = None
    actual_model: str | None = None
    deprecation_warning: str | None = None
```

Example usage:

```python
response = await llm.get_response("Hello")

if response.deprecation_warning:
    logger.warning(response.deprecation_warning)
    # "claude-3-7-sonnet-20250219 is deprecated and will be retired on 2026-02-19.
    #  Migrate to claude-sonnet-4-5-20250929."

if response.requested_model != response.actual_model:
    logger.info(f"Model replaced: {response.requested_model} -> {response.actual_model}")
```

### 4. GitHub Action for Automated PRs

Create a scheduled workflow that monitors provider deprecation pages:

```yaml
# .github/workflows/check-deprecations.yml
name: Check Model Deprecations
on:
  schedule:
    - cron: '0 9 * * 1'  # Weekly on Monday 9am UTC
  workflow_dispatch:  # Manual trigger

jobs:
  check-deprecations:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install requests beautifulsoup4 pyyaml

      - name: Check provider deprecation pages
        run: python scripts/check_deprecations.py

      - name: Create PR if changes detected
        uses: peter-evans/create-pull-request@v5
        with:
          title: "chore: update model deprecation metadata"
          body: |
            Automated deprecation check found updates.

            Please review the changes to `llm_config.yaml` and verify:
            - Deprecation dates are correct
            - Replacement models are appropriate
            - Replacement notes are helpful
          branch: auto/deprecation-updates
          labels: dependencies, automated
```

The `scripts/check_deprecations.py` script would:
1. Fetch each provider's deprecation page
2. Parse for model IDs and dates
3. Compare against current `llm_config.yaml`
4. Update config with new deprecation/replacement info
5. Exit with status indicating if changes were made

## Alternative Ideas

### Majordomo-Managed Aliases

Provide logical model names that majordomo maps to concrete models:

```python
# Developer uses logical name
llm = get_llm_instance("anthropic", "claude-sonnet-fast")
llm = get_llm_instance("anthropic", "claude-sonnet-stable")
llm = get_llm_instance("openai", "gpt-best-value")
```

Alias mappings in config:

```yaml
aliases:
  anthropic:
    claude-sonnet-fast: claude-haiku-4-5-20251001
    claude-sonnet-stable: claude-sonnet-4-20250514  # Older but well-tested
    claude-smartest: claude-opus-4-5-20251101
  openai:
    gpt-best-value: gpt-4o-mini
    gpt-reasoning: o3
```

### Webhook Notifications

Allow users to configure webhooks for deprecation events:

```python
from majordomo_llm import configure

configure(
    deprecation_webhook="https://hooks.slack.com/...",
    deprecation_webhook_events=["warning", "replacement"]
)
```

### Provider API Detection

Some providers return deprecation warnings in response headers or error messages. Could capture and surface these automatically, even without config updates.

## Trade-offs to Consider

### Automatic Replacement

**Pros:**
- Zero developer effort after setup
- No service disruption on retirement date
- Graceful degradation

**Cons:**
- Silent replacement could change behavior unexpectedly
- Replacement model might have different capabilities (context length, features)
- Cost differences could be significant

### Cost Implications

Should warn or block if replacement model has significantly different costs:

```yaml
claude-3-haiku-20240307:
  input_cost: 0.25
  output_cost: 1.25
  replacement: claude-haiku-4-5-20251001  # 4x more expensive!
  replacement_cost_increase: 4.0  # Flag for runtime warning
```

### Compatibility Scoring

Consider adding compatibility metadata:

```yaml
claude-3-7-sonnet-20250219:
  replacement: claude-sonnet-4-5-20250929
  replacement_compatibility:
    context_window: compatible  # same or larger
    features: extended  # replacement has more features
    cost: similar  # within 20%
    behavior: different  # may require prompt adjustments
```

## Implementation Phases

### Phase 1: Config Schema
- Add deprecation fields to `llm_config.yaml`
- Update existing deprecated models with metadata
- No runtime changes yet

### Phase 2: Response Warnings
- Add `deprecation_warning` to response objects
- Warn when using deprecated models
- No auto-replacement yet

### Phase 3: Auto-Replacement
- Add `on_deprecation` configuration option
- Implement replacement logic for retired models
- Add `requested_model` / `actual_model` to responses

### Phase 4: GitHub Automation
- Create `scripts/check_deprecations.py`
- Set up GitHub Action workflow
- Test with manual triggers before enabling schedule

### Phase 5: Advanced Features
- Aliases
- Webhook notifications
- Compatibility scoring

## Open Questions

1. **Default behavior**: Should `on_deprecation` default to `warn` or `strict`?

2. **Scope of replacement**: Should replacement work across providers? (e.g., if Anthropic model is retired, could fall back to OpenAI?)

3. **Testing**: How do we test replacement behavior without waiting for actual deprecations?

4. **Versioning**: Should deprecation metadata updates be considered breaking changes?

5. **Caching**: Should we cache deprecation checks to avoid hammering provider pages?
