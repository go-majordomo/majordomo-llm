"""Hook pipeline that runs ``LLMHook`` instances around an arbitrary LLM call."""

import inspect
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from majordomo_llm.hooks.protocol import (
    HookBlocked,
    HookContext,
    HookOutcome,
    HookVerdict,
    LLMHook,
)

logger = logging.getLogger(__name__)


OnVerdicts = Callable[[uuid.UUID, list[HookVerdict]], Awaitable[None] | None]


class HookPipeline:
    """Runs before/after hooks around any async string-producing call.

    The pipeline is intentionally decoupled from any specific LLM type. The
    ``call`` argument to :meth:`run` can wrap an ``LLMCascade``, a raw
    provider SDK invocation, or any other coroutine that takes a prompt
    string and returns a response string.

    Args:
        hooks: Ordered list of hooks. Each is invoked in list order for
            both phases; modifications from earlier hooks are visible to
            later hooks in the same phase.
        on_verdicts: Optional callback invoked with the full verdict list
            when the pipeline run finishes (success or block). Exceptions
            raised here are caught and logged; they never propagate.
    """

    def __init__(
        self,
        hooks: list[LLMHook],
        *,
        on_verdicts: OnVerdicts | None = None,
    ) -> None:
        self._hooks = list(hooks)
        self._on_verdicts = on_verdicts

    async def run(
        self,
        prompt: str,
        call: Callable[[str], Awaitable[str]],
        *,
        caller_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Run hooks around ``call`` and return the (possibly modified) response.

        Raises:
            HookBlocked: if any hook returns ``blocked=True``.
        """
        request_id = uuid.uuid4()
        verdicts: list[HookVerdict] = []
        metadata = caller_metadata if caller_metadata is not None else {}

        current_prompt = prompt
        before_ctx = HookContext(
            request_id=request_id, phase="before", caller_metadata=metadata
        )
        for hook in self._hooks:
            outcome = await self._run_hook(
                hook.before_call(current_prompt, before_ctx),
                hook_name=hook.name,
            )
            verdicts.append(outcome.verdict)
            if outcome.modified_text is not None:
                current_prompt = outcome.modified_text
            if outcome.blocked:
                await self._emit_verdicts(request_id, verdicts)
                raise HookBlocked(verdicts)

        response = await call(current_prompt)

        after_ctx = HookContext(
            request_id=request_id, phase="after", caller_metadata=metadata
        )
        current_response = response
        for hook in self._hooks:
            outcome = await self._run_hook(
                hook.after_call(current_prompt, current_response, after_ctx),
                hook_name=hook.name,
            )
            verdicts.append(outcome.verdict)
            if outcome.modified_text is not None:
                current_response = outcome.modified_text
            if outcome.blocked:
                await self._emit_verdicts(request_id, verdicts)
                raise HookBlocked(verdicts)

        await self._emit_verdicts(request_id, verdicts)
        return current_response

    async def _run_hook(
        self,
        hook_coro: Awaitable[HookOutcome],
        *,
        hook_name: str,
    ) -> HookOutcome:
        """Invoke a hook coroutine, swallowing non-HookBlocked exceptions.

        A buggy hook must never break the underlying LLM call. Any exception
        other than HookBlocked is logged and converted to a pass verdict
        with the latency measured up to the point of failure.
        """
        start = time.monotonic()
        try:
            outcome = await hook_coro
        except HookBlocked:
            raise
        except Exception:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.exception("Hook %s raised; treating as pass", hook_name)
            return HookOutcome(
                verdict=HookVerdict(
                    hook_name=hook_name,
                    verdict="pass",
                    action_taken="pass",
                    reason="hook raised exception",
                    latency_ms=elapsed_ms,
                )
            )
        if outcome.verdict.latency_ms is None:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return HookOutcome(
                verdict=HookVerdict(
                    hook_name=outcome.verdict.hook_name,
                    verdict=outcome.verdict.verdict,
                    action_taken=outcome.verdict.action_taken,
                    reason=outcome.verdict.reason,
                    latency_ms=elapsed_ms,
                ),
                modified_text=outcome.modified_text,
                blocked=outcome.blocked,
            )
        return outcome

    async def _emit_verdicts(
        self, request_id: uuid.UUID, verdicts: list[HookVerdict]
    ) -> None:
        if self._on_verdicts is None:
            return
        try:
            result = self._on_verdicts(request_id, verdicts)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("on_verdicts callback raised; swallowing")
