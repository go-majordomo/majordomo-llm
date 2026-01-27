# majordomo-llm Examples

This directory contains example applications demonstrating majordomo-llm features.

## Structured Response Demo

The `structured_response_demo.py` script showcases the `get_structured_json_response()` method with various Pydantic models:

- **Sentiment Analysis** - Simple model with Enum field
- **Text Analysis** - Nested models with entity extraction
- **Code Review** - Constrained integer fields and booleans
- **Product Recommendations** - Complex nested lists with validation

### Run

```bash
uv run python examples/structured_response_demo.py
```

### Example Output

```
Demo 1: Sentiment Analysis (with Enum)
Result (SentimentAnalysis):
  Sentiment: positive
  Confidence: 95.00%
  Reasoning: The text expresses enthusiasm with words like "thrilled" and "exceeded expectations"
```

## Demo: Multi-Provider Comparison with Logging

The `demo.py` script showcases:

- Running the same prompts across multiple LLM providers (OpenAI, Anthropic, Gemini, DeepSeek, Cohere)
- Automatic request logging to SQLite with API key hash tracking
- Local file storage for request/response bodies
- Cost and performance comparison across providers

### Setup

1. Install dependencies with the logging extras:

```bash
uv sync --all-extras
```

2. Set API keys for the providers you want to test. Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=sk-...
CO_API_KEY=...
```

Or export them in your shell. You don't need all keys - the demo will skip providers without keys.

### Run

```bash
uv run python examples/demo.py
```

### Output

The demo will:

1. Run 3 prompts (code, content, customer support) against each available provider
2. Display responses, token counts, costs, and timing for each
3. Print a summary table from the logged metrics
4. Save all request/response bodies as JSON files in `examples/request_logs/`

### Files Created

After running, you'll have:

- `llm_logs.db` - SQLite database with request metrics
- `request_logs/` - Directory with JSON files for each request/response

You can query the SQLite database directly:

```bash
# Basic query
sqlite3 examples/llm_logs.db "SELECT provider, model, total_cost, response_time FROM llm_requests"

# Query with API key tracking (api_key_hash is first 16 chars of SHA256)
sqlite3 examples/llm_logs.db "SELECT provider, model, api_key_hash, api_key_alias FROM llm_requests"
```

## Prompts

The `prompts.json` file contains sample prompts across three domains:

- **code-generation**: Rust ownership explanation
- **content-generation**: Marketing tagline creation
- **customer-support**: Ticket classification

Feel free to modify or add prompts to test different scenarios.
