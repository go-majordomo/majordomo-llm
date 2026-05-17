"""LLM-as-judge hook implementation."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from majordomo_llm.hooks.protocol import HookContext, HookOutcome

logger = logging.getLogger(__name__)


@dataclass
class LLMJudgeHook:
    """Delegates the pass/fail decision to a caller-supplied LLM call.

    The ``judge_call`` is any async callable that takes a rendered prompt
    and returns a string. The library does not construct it — consumers
    typically pass a lambda over an ``Anthropic``/``OpenAI`` client, a
    second ``LLMCascade`` (constructed without a hook pipeline to avoid
    recursion), or any other async callable.

    The judge is expected to return JSON of shape
    ``{"verdict": "pass" | "fail", "reason": str}``. Any failure path —
    timeout, JSON parse error, judge exception — is treated as pass and
    logged via ``logging.warning``. The judge must never block the
    underlying LLM call during a provider incident.
    """

    name: str
    judge_call: Callable[[str], Awaitable[str]]
    judge_prompt: str
    phase: Literal["before", "after"]
    action: Literal["block", "warn"] = "block"
    timeout_seconds: float = 10.0

    async def before_call(self, prompt: str, ctx: HookContext) -> HookOutcome:
        if self.phase != "before":
            return HookOutcome.pass_through(self.name)
        rendered = self._render(prompt=prompt, response=None)
        return await self._judge(rendered)

    async def after_call(
        self, prompt: str, response: str, ctx: HookContext
    ) -> HookOutcome:
        if self.phase != "after":
            return HookOutcome.pass_through(self.name)
        rendered = self._render(prompt=prompt, response=response)
        return await self._judge(rendered)

    def _render(self, *, prompt: str, response: str | None) -> str:
        if response is None:
            return self.judge_prompt.format(prompt=prompt)
        return self.judge_prompt.format(prompt=prompt, response=response)

    async def _judge(self, rendered: str) -> HookOutcome:
        try:
            raw = await asyncio.wait_for(
                self.judge_call(rendered), timeout=self.timeout_seconds
            )
        except TimeoutError:
            logger.warning("Judge hook %s timed out; passing through", self.name)
            return HookOutcome.pass_through(self.name, reason="judge timeout")
        except Exception:
            logger.warning(
                "Judge hook %s raised an exception; passing through", self.name,
                exc_info=True,
            )
            return HookOutcome.pass_through(self.name, reason="judge exception")

        try:
            parsed = json.loads(raw)
            verdict = parsed["verdict"]
            reason = parsed.get("reason", "")
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning(
                "Judge hook %s returned unparseable output; passing through",
                self.name,
            )
            return HookOutcome.pass_through(self.name, reason="judge parse error")

        if verdict == "pass":
            return HookOutcome.pass_through(self.name, reason=reason or None)
        if self.action == "block":
            return HookOutcome.block(self.name, reason or "judge failed")
        return HookOutcome.warn(self.name, reason or "judge failed")
