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


def create_model(settings: AppSettings, model_override: str = "") -> Model:
    """Instantiate a Strands model from application settings.

    Lazy-imports provider-specific modules so that only the chosen
    provider's SDK needs to be installed.

    Args:
        settings: Application settings (drives provider selection + model params).
        model_override: If non-empty, used instead of ``settings.llm.model``.
            Typically populated from ``settings.stage_models.<stage>_model``.
    """
    provider = settings.llm.provider
    effective_model = model_override or settings.llm.model

    if provider == "bedrock":
        from strands.models.bedrock import BedrockModel

        return BedrockModel(
            model_id=effective_model,
            region_name=settings.llm.aws_region,
            temperature=settings.llm.temperature,
            top_p=settings.llm.top_p,
        )

    if provider == "openai":
        from strands.models.openai import OpenAIModel

        client_args: dict[str, Any] = {
            "api_key": settings.llm.api_key,
        }
        if settings.llm.base_url:
            client_args["base_url"] = settings.llm.base_url

        params: dict[str, Any] = {
            "temperature": settings.llm.temperature,
            "top_p": settings.llm.top_p,
        }
        if settings.llm.seed is not None:
            params["seed"] = settings.llm.seed

        return OpenAIModel(
            client_args=client_args,
            model_id=effective_model,
            params=params,
        )

    if provider == "ollama":
        from strands.models.ollama import OllamaModel

        host = settings.llm.base_url
        if host.endswith("/v1"):
            host = host[:-3]

        return OllamaModel(
            host=host,
            model_id=effective_model,
        )

    if provider == "anthropic":
        from strands.models.litellm import LiteLLMModel

        model_kwargs: dict[str, Any] = {
            "top_p": settings.llm.top_p,
        }
        if settings.caching.enabled:
            model_kwargs["cache_control_injection_points"] = [
                {"location": "message", "role": "system"}
            ]
        return LiteLLMModel(
            model_id=f"anthropic/{effective_model}",
            model_kwargs=model_kwargs,
        )

    if provider == "litellm":
        from strands.models.litellm import LiteLLMModel

        litellm_kwargs: dict[str, Any] = {
            "top_p": settings.llm.top_p,
        }
        if settings.caching.enabled:
            litellm_kwargs["cache_control_injection_points"] = [
                {"location": "message", "role": "system"}
            ]
        return LiteLLMModel(
            model_id=effective_model,
            model_kwargs=litellm_kwargs,
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")
