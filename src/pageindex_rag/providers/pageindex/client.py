"""Async LLM client routed through LiteLLM for multi-provider support.

Replaces the previous ``AsyncOpenAI``-only client with ``litellm.acompletion()``,
enabling Anthropic prompt caching via ``cache_control`` on content blocks.
Supports ``anthropic/``, ``bedrock/``, ``openai/`` model prefixes transparently.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pageindex_rag.config import PageIndexSettings
from pageindex_rag.exceptions import LLMClientError

log = logging.getLogger(__name__)


class LLMClient:
    """Async LLM client using LiteLLM with optional Anthropic prompt caching."""

    def __init__(self, settings: PageIndexSettings) -> None:
        self._settings = settings

    @property
    def model(self) -> str:
        return self._settings.llm_model

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

        last_error: Exception | None = None
        for attempt in range(self._settings.llm_max_retries):
            try:
                response = await acompletion(
                    model=effective_model,
                    messages=messages,
                    temperature=effective_temp,
                    timeout=self._settings.llm_timeout,
                )
                content = response.choices[0].message.content or ""
                reason = response.choices[0].finish_reason
                mapped_reason = "max_output_reached" if reason == "length" else "finished"
                return content, mapped_reason

            except Exception as e:
                last_error = e
                wait = min(2**attempt, 30)
                log.warning(f"LLM retry {attempt + 1}/{self._settings.llm_max_retries}: {e}")
                if attempt < self._settings.llm_max_retries - 1:
                    await asyncio.sleep(wait)

        raise LLMClientError(
            f"LLM API failed after {self._settings.llm_max_retries} retries: {last_error}"
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
                    log.warning(f"Batch task failed: {e}")
                    return ""

        return await asyncio.gather(*[_bounded(p) for p in prompts])

    # ── JSON extraction (static) ─────────────────────────────────────

    @staticmethod
    def extract_json(content: str) -> Any:
        """Parse JSON from LLM response, handling ```json fences and common issues."""
        try:
            # Extract from ```json ... ``` fences
            start = content.find("```json")
            if start != -1:
                start += 7
                end = content.rfind("```")
                json_str = content[start:end].strip()
            else:
                json_str = content.strip()

            json_str = json_str.replace("None", "null")
            json_str = json_str.replace("\n", " ").replace("\r", " ")
            json_str = " ".join(json_str.split())

            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                json_str = json_str.replace(",]", "]").replace(",}", "}")
                return json.loads(json_str)
            except Exception:
                log.error("Failed to parse JSON from LLM response")
                return {}
        except Exception:
            log.error("Unexpected error extracting JSON")
            return {}

    async def close(self) -> None:
        """No-op — LiteLLM manages its own connection pooling."""
