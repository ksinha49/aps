"""Token counting skill — wraps the existing TokenCounter as a @tool."""

from __future__ import annotations

import json

from strands import tool
from strands.types.tools import ToolContext


@tool(context=True)
def count_tokens(text: str, tool_context: ToolContext) -> str:
    """Count the number of tokens in the given text using the configured tokenizer.

    Args:
        text: The text to count tokens for.

    Returns:
        JSON string with token_count.
    """
    from scout_ai.core.config import TokenizerConfig
    from scout_ai.providers.pageindex.tokenizer import TokenCounter

    settings = tool_context.invocation_state.get("settings")
    tok_cfg = settings.tokenizer if settings else TokenizerConfig()

    tc = TokenCounter(
        method=tok_cfg.method,
        model=tok_cfg.model,
        char_to_token_ratio=tok_cfg.char_to_token_ratio,
        fallback_encoding=tok_cfg.fallback_encoding,
    )
    count = tc.count(text)
    return json.dumps({"token_count": count})
