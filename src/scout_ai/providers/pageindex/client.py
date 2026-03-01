"""Async LLM client routed through LiteLLM for multi-provider support.

Replaces the previous ``AsyncOpenAI``-only client with ``litellm.acompletion()``,
enabling Anthropic prompt caching via ``cache_control`` on content blocks.
Supports ``anthropic/``, ``bedrock/``, ``openai/`` model prefixes transparently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

from scout_ai.config import ScoutSettings
from scout_ai.exceptions import LLMClientError, NonRetryableError, RetryableError

log = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client using LiteLLM with optional Anthropic prompt caching."""

    def __init__(self, settings: ScoutSettings) -> None:
        self._settings = settings

    @property
    def model(self) -> str:
        return self._settings.llm_model

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Classify whether an LLM API error should be retried.

        Non-retryable: AuthenticationError, BadRequestError, NotFoundError (4xx non-429).
        Retryable (default): everything else including rate limits, timeouts, 5xx.
        """
        try:
            from litellm.exceptions import (
                AuthenticationError,
                BadRequestError,
                NotFoundError,
            )

            non_retryable = (AuthenticationError, BadRequestError, NotFoundError)
            return not isinstance(exc, non_retryable)
        except ImportError:
            # Without litellm type info, treat all errors as retryable
            return True

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        cache_system: bool = False,
        model: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> str:
        """Single completion, returns content string.

        Args:
            prompt: User message content.
            system_prompt: Optional system message. When ``cache_system`` is True,
                the system content block is tagged with ``cache_control``.
            cache_system: If True, adds ``cache_control: {"type": "ephemeral"}``
                to the system message content block for Anthropic prompt caching.
            model: Override model ID. Supports LiteLLM prefixes (e.g. ``anthropic/``).
            chat_history: Prior conversation messages.
            temperature: Override temperature.
        """
        content, _ = await self.complete_with_finish_reason(
            prompt,
            system_prompt=system_prompt,
            cache_system=cache_system,
            model=model,
            chat_history=chat_history,
            temperature=temperature,
        )
        return content

    async def complete_with_finish_reason(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        cache_system: bool = False,
        model: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> tuple[str, str]:
        """Completion returning ``(content, finish_reason)``.

        ``finish_reason`` is ``"finished"`` or ``"max_output_reached"``.
        """
        from litellm import acompletion

        effective_model = model or self._settings.llm_model
        effective_temp = temperature if temperature is not None else self._settings.llm_temperature

        messages: list[dict[str, Any]] = []

        # System message with optional cache_control
        if system_prompt:
            system_content: list[dict[str, Any]] = [{"type": "text", "text": system_prompt}]
            if cache_system:
                system_content[0]["cache_control"] = {"type": "ephemeral"}
            messages.append({"role": "system", "content": system_content})

        if chat_history:
            messages.extend(chat_history)

        messages.append({"role": "user", "content": prompt})

        max_retries = self._settings.llm_max_retries
        jitter_factor = getattr(self._settings, "retry_jitter_factor", 0.5)
        max_delay = getattr(self._settings, "retry_max_delay", 30.0)

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": effective_model,
                    "messages": messages,
                    "temperature": effective_temp,
                    "top_p": self._settings.llm_top_p,
                    "timeout": self._settings.llm_timeout,
                }
                if self._settings.llm_seed is not None:
                    kwargs["seed"] = self._settings.llm_seed
                response = await acompletion(**kwargs)
                content = response.choices[0].message.content or ""
                reason = response.choices[0].finish_reason
                mapped_reason = "max_output_reached" if reason == "length" else "finished"
                return content, mapped_reason

            except Exception as e:
                last_error = e
                retryable = self._is_retryable(e)

                if not retryable:
                    raise NonRetryableError(
                        f"Non-retryable LLM error: {e}"
                    ) from e

                base_wait = min(2 ** attempt, max_delay)
                jitter = random.uniform(0, base_wait * jitter_factor)
                wait = base_wait + jitter

                log.warning(
                    "LLM retry %d/%d: %s (retryable=%s, wait=%.1fs)",
                    attempt + 1, max_retries, e, retryable, wait,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)

        raise RetryableError(
            f"LLM API failed after {max_retries} retries: {last_error}"
        ) from last_error

    async def complete_batch(
        self,
        prompts: list[str],
        *,
        system_prompt: str | None = None,
        cache_system: bool = False,
        max_concurrent: int | None = None,
        timeout_per_task: float = 120.0,
    ) -> list[str]:
        """Run multiple completions with concurrency control.

        When ``system_prompt`` and ``cache_system`` are provided, the system
        content is cached once and reused across all batch calls — the core
        mechanism for Anthropic prompt caching cost savings.

        Returns results in order; failed tasks return empty strings.
        """
        sem = asyncio.Semaphore(max_concurrent or self._settings.retrieval_max_concurrent)

        async def _bounded(prompt: str) -> str:
            async with sem:
                try:
                    return await asyncio.wait_for(
                        self.complete(
                            prompt,
                            system_prompt=system_prompt,
                            cache_system=cache_system,
                        ),
                        timeout=timeout_per_task,
                    )
                except (asyncio.TimeoutError, LLMClientError) as e:
                    log.warning("Batch task failed: %s", e)
                    return ""

        return await asyncio.gather(*[_bounded(p) for p in prompts])

    # ── JSON extraction (static) ─────────────────────────────────────

    @staticmethod
    def extract_json(content: str) -> Any:
        """Parse JSON from LLM response, handling fences, prose, and common issues."""
        import re

        def _try_parse(s: str) -> Any | None:
            """Attempt JSON parse with common fixups."""
            s = s.strip()
            if not s:
                return None
            # Fix Python-style None → null
            s = s.replace("None", "null")
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
            # Trailing comma fix
            s = re.sub(r",\s*([}\]])", r"\1", s)
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return None

        # Strategy 1: ```json ... ``` fences
        fence_start = content.find("```json")
        if fence_start != -1:
            inner = content[fence_start + 7:]
            fence_end = inner.find("```")
            if fence_end != -1:
                result = _try_parse(inner[:fence_end])
                if result is not None:
                    return result

        # Strategy 2: ``` ... ``` generic fence
        fence_start = content.find("```")
        if fence_start != -1:
            inner = content[fence_start + 3:]
            fence_end = inner.find("```")
            if fence_end != -1:
                result = _try_parse(inner[:fence_end])
                if result is not None:
                    return result

        # Strategy 3: Full content as JSON
        result = _try_parse(content)
        if result is not None:
            return result

        # Strategy 4: Find first { ... } or [ ... ] in the response
        for open_ch, close_ch in [("{", "}"), ("[", "]")]:
            idx = content.find(open_ch)
            if idx == -1:
                continue
            depth = 0
            in_string = False
            escape = False
            for i in range(idx, len(content)):
                ch = content[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        result = _try_parse(content[idx : i + 1])
                        if result is not None:
                            return result
                        break

        log.error(
            "Failed to parse JSON from LLM response",
            extra={"response_length": len(content), "response_preview": content[:200]},
        )
        return {}

    async def close(self) -> None:
        """No-op — LiteLLM manages its own connection pooling."""
