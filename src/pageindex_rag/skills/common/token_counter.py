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
    from pageindex_rag.providers.pageindex.tokenizer import TokenCounter

    settings = tool_context.invocation_state.get("settings")
    if settings:
        method = settings.tokenizer.method
        model = settings.tokenizer.model
    else:
        method = "approximate"
        model = "gpt-4o"

    tc = TokenCounter(method=method, model=model)
    count = tc.count(text)
    return json.dumps({"token_count": count})
