"""Strands model provider factory.

Maps ``settings.llm.provider`` to the correct Strands model class,
so the same application code runs against Ollama locally, OpenAI,
Bedrock in production, or LiteLLM as a universal proxy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands.types.models import Model

    from scout_ai.core.config import AppSettings


def create_model(settings: AppSettings) -> Model:
    """Instantiate a Strands model from application settings.

    Lazy-imports provider-specific modules so that only the chosen
    provider's SDK needs to be installed.
    """
    provider = settings.llm.provider

    if provider == "bedrock":
        from strands.models.bedrock import BedrockModel

        return BedrockModel(
            model_id=settings.llm.model,
            region_name=settings.llm.aws_region,
            temperature=settings.llm.temperature,
        )

    if provider == "openai":
        from strands.models.openai import OpenAIModel

        client_args: dict[str, Any] = {
            "api_key": settings.llm.api_key,
        }
        if settings.llm.base_url:
            client_args["base_url"] = settings.llm.base_url

        return OpenAIModel(
            client_args=client_args,
            model_id=settings.llm.model,
            params={
                "temperature": settings.llm.temperature,
            },
        )

    if provider == "ollama":
        from strands.models.ollama import OllamaModel

        host = settings.llm.base_url
        if host.endswith("/v1"):
            host = host[:-3]

        return OllamaModel(
            host=host,
            model_id=settings.llm.model,
        )

    if provider == "anthropic":
        from strands.models.litellm import LiteLLMModel

        model_kwargs: dict[str, Any] = {}
        if settings.caching.enabled:
            model_kwargs["cache_control_injection_points"] = [
                {"location": "message", "role": "system"}
            ]
        return LiteLLMModel(
            model_id=f"anthropic/{settings.llm.model}",
            model_kwargs=model_kwargs,
        )

    if provider == "litellm":
        from strands.models.litellm import LiteLLMModel

        litellm_kwargs: dict[str, Any] = {}
        if settings.caching.enabled:
            litellm_kwargs["cache_control_injection_points"] = [
                {"location": "message", "role": "system"}
            ]
        return LiteLLMModel(
            model_id=settings.llm.model,
            model_kwargs=litellm_kwargs,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")
