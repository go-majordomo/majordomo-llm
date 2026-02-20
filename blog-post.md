# Your AI Stack Needs a Control Plane

You trace every database query. You monitor every microservice. You have dashboards for API latency, error rates, and throughput.

Your LLM calls — the ones that cost a dollar each — run in total darkness.

```python
model = "claude-sonnet-4-20250514"
```

Hardcoded in 30 places across your codebase, applied to every task regardless of complexity. No attribution to features or users. No visibility into what's working and what's wasteful. No failover when a provider goes down.

Classification that returns one of five labels? Sonnet. Extracting a date from an email? Sonnet. Summarizing a paragraph? Sonnet.

It works. But without a control plane, you can't see, manage, or optimize any of it.

## The Gaps a Dashboard Won't Close

Plenty of tools will show you aggregate LLM spend. That's useful, but it's not actionable. Knowing you spent $2,400 on Anthropic last month doesn't tell you what to do differently. And it doesn't solve the operational problems that come with running AI in production:

**The invoice nobody can explain.** Your AI spend is one of the fastest-growing line items in the P&L — and nobody knows which team, feature, or experiment is driving it.

**Three teams, zero shared data.** Engineering wants GPT-4 for quality. Finance wants costs cut 40%. Product wants three new features shipped. Without data, everyone is guessing.

**Single provider, single point of failure.** One API outage, one rate limit spike, and your application stops working. No automatic failover, no policy enforcement, no recourse.

What you actually need is the ability to answer questions like:

- "For my document classification feature, does Haiku produce the same labels as Sonnet?"
- "If I switch my extraction pipeline to GPT-4o mini, do the structured outputs still validate?"
- "Which of my features are latency-sensitive enough that a faster, cheaper model would actually improve the user experience?"

To answer these questions, you need a control plane — not just metrics, but the operational infrastructure to observe, route, and optimize every LLM call.

## Introducing Majordomo: The Control Plane for Your AI Stack

Majordomo is a suite of open-source projects that make LLM operations observable and reliable. Deploy it alongside your existing stack, and get visibility and control from day one.

### majordomo-gateway

A lightweight API proxy written in Go. Deploy it once, point your LLM API calls at it, and get:

- **Automatic cost calculation** with real-time pricing (refreshed hourly from llm-prices.com, so you're never working with stale numbers)
- **Full request/response body logging** to S3 or PostgreSQL — the payloads you'll need for replay and optimization
- **Custom metadata via headers** — tag requests with feature name, user ID, session, workflow step, or anything else. Query your costs by any dimension that matters to your team
- **Multi-provider support** — OpenAI, Anthropic, Gemini, with automatic provider detection from the request path
- **Proxy keys** for multi-tenant setups — give each customer or team their own key, map it to real provider credentials (encrypted at rest with AES-256), track usage independently

The gateway is a compiled Go binary. It's fast, it uses minimal resources, and it adds negligible latency to your requests. It works with any HTTP client in any language — curl, Python requests, Node fetch, whatever you're using today.

```bash
# Before: direct to Anthropic
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -d '{"model": "claude-sonnet-4-20250514", "messages": [...]}'

# After: through the control plane (one header added)
curl http://localhost:7680/v1/messages \
  -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
  -H "X-Majordomo-Key: $MAJORDOMO_KEY" \
  -H "X-Majordomo-Feature: document-classification" \
  -d '{"model": "claude-sonnet-4-20250514", "messages": [...]}'
```

No SDK changes. No code refactoring. One header for auth, one header to tag the feature.

### majordomo-llm

A Python async client library for when you want the control plane built directly into your application code. Every response includes token counts, costs, and latency as part of the response object:

```python
from majordomo_llm import get_llm_instance

llm = get_llm_instance("anthropic", "claude-sonnet-4-20250514")
response = await llm.get_response("Classify this document: ...")

print(response.content)          # "Category: Legal"
print(response.total_cost)       # 0.000842
print(response.response_time)    # 1.23
print(response.output_tokens)    # 12
```

The library also gives you:

- **Structured outputs** that work consistently across providers. Pass a Pydantic model, get a validated instance back — whether you're using OpenAI (JSON Schema mode), Anthropic (tool calling), or Gemini (response schema). One interface, provider-specific optimizations under the hood.
- **Cascade failover** — define a priority list of (provider, model) pairs. If your primary provider errors, Majordomo automatically tries the next one. Combined with per-provider retries, this gives you multi-layer resilience without application code changes.
- **Async request logging** — fire-and-forget logging to PostgreSQL, MySQL, or SQLite, with request/response bodies stored in S3 or the filesystem. Zero impact on request latency.

## Works With Your Stack, Not Instead of It

A control plane sits underneath your application — it doesn't replace it.

If you're building with Pydantic AI, Agno, or any other framework, you don't need to change anything fundamental. The gateway is a transparent proxy — your framework talks to the gateway instead of directly to the provider. Everything else stays the same.

This is a deliberate design choice. Pydantic AI and Agno are excellent frameworks for building AI agents — tools, conversation management, structured outputs. They're not trying to be operational infrastructure, and they shouldn't have to be. Majordomo handles the control plane so your framework can focus on what it does best.

## The Replay Workflow: From Observability to Optimization

This is where the control plane pays off. Visibility is step one. Optimization is the goal.

Say you've been running your document classification feature on Claude Sonnet for three months. The gateway has logged every request and response. Now you want to know: could Haiku handle this?

Majordomo includes a replay tool that automates this workflow:

```bash
cd majordomo-gateway/replay
uv run python -m replay.main --config replay.yaml
```

Configure what to replay, what to replay it against, and optionally an LLM judge to evaluate equivalence:

```yaml
source:
  filters:
    feature: document-classification
  model: claude-sonnet-4-20250514
  days: 30
  limit: 50

target:
  provider: anthropic
  model: claude-haiku-4-20250514

judge:
  enabled: true
  provider: openai
  model: gpt-4.1-mini
```

The tool:

1. **Fetches logged requests** from the gateway database, filtered by metadata, model, and time range
2. **Replays each prompt** against your target model
3. **Compares outputs** — exact match first, then an LLM-as-judge for semantic equivalence
4. **Prints a report** — cost and latency comparisons, match rates, and the specific examples where the models diverged

If Haiku agrees with Sonnet 98% of the time on this task, you switch and save 90% on that feature's LLM costs while getting faster responses. If it doesn't, you know exactly where and why.

You're not guessing based on benchmarks. You're testing with your actual production data, your actual prompts, your actual edge cases. That's the difference between a dashboard and a control plane.

## Where Majordomo Fits

There are great tools in this space. LiteLLM, for instance, offers a comprehensive platform with support for 100+ providers, virtual keys, budgets, and rate limiting. If you want an all-in-one solution, it's a solid choice.

Majordomo takes a different approach: a modular, composable control plane.

- **Already have a framework?** The gateway works as a transparent proxy underneath Pydantic AI, Agno, or anything else that makes HTTP calls.
- **Already have an HTTP client?** No SDK required. Any language, any client.
- **Want just the library?** Use `majordomo-llm` standalone with its built-in cost tracking. No gateway needed.
- **Want just the gateway?** Deploy it as a transparent proxy. No Python required.
- **Want both?** They share a PostgreSQL schema and complement each other.

Use whichever pieces make sense. Leave the rest.

## Get Started

All projects are open source under the MIT license.

- **majordomo-gateway**: [GitHub link] | [Docs](https://majordomo-gateway.readthedocs.io)
- **majordomo-llm**: [GitHub link] | [Docs](https://majordomo-llm.readthedocs.io) | [PyPI](https://pypi.org/project/majordomo-llm/)
