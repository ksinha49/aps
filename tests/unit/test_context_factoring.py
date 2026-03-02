"""Tests for context factoring — multi-breakpoint prompt cache hierarchy."""

from __future__ import annotations

from scout_ai.context.factoring.breakpoint_strategy import compute_breakpoints
from scout_ai.context.factoring.layer_builder import ContextLayerBuilder
from scout_ai.context.models import ContextLayer


class TestComputeBreakpoints:
    """compute_breakpoints should assign cache_control to highest-value layers."""

    def test_system_always_cached(self) -> None:
        layers = [
            ContextLayer(role="system", content=[], layer_type="system"),
            ContextLayer(role="user", content=[], layer_type="query"),
        ]
        result = compute_breakpoints(layers, max_breakpoints=4)
        assert result[0].cache_breakpoint is True
        assert result[1].cache_breakpoint is False

    def test_query_never_cached(self) -> None:
        layers = [
            ContextLayer(role="system", content=[], layer_type="system"),
            ContextLayer(role="system", content=[], layer_type="document"),
            ContextLayer(role="user", content=[], layer_type="query"),
        ]
        result = compute_breakpoints(layers, max_breakpoints=4)
        query = [ly for ly in result if ly.layer_type == "query"][0]
        assert query.cache_breakpoint is False

    def test_respects_max_breakpoints(self) -> None:
        layers = [
            ContextLayer(role="system", content=[], layer_type="system"),
            ContextLayer(role="system", content=[], layer_type="tools"),
            ContextLayer(role="system", content=[], layer_type="document"),
            ContextLayer(role="user", content=[], layer_type="query"),
        ]
        result = compute_breakpoints(layers, max_breakpoints=1)
        cached = [ly for ly in result if ly.cache_breakpoint]
        assert len(cached) == 1
        assert cached[0].layer_type == "system"

    def test_priority_order(self) -> None:
        """System > tools > document in priority."""
        layers = [
            ContextLayer(role="system", content=[], layer_type="document"),
            ContextLayer(role="system", content=[], layer_type="system"),
            ContextLayer(role="system", content=[], layer_type="tools"),
            ContextLayer(role="user", content=[], layer_type="query"),
        ]
        result = compute_breakpoints(layers, max_breakpoints=2)
        cached = [ly for ly in result if ly.cache_breakpoint]
        assert {ly.layer_type for ly in cached} == {"system", "tools"}


class TestContextLayerBuilder:
    """ContextLayerBuilder should produce correct message structure."""

    def test_build_layers_basic(self) -> None:
        builder = ContextLayerBuilder(max_breakpoints=4)
        layers = builder.build_layers("System", "Document text", "What is X?")
        assert len(layers) == 3
        assert layers[0].layer_type == "system"
        assert layers[1].layer_type == "document"
        assert layers[2].layer_type == "query"

    def test_build_layers_with_tools(self) -> None:
        builder = ContextLayerBuilder()
        layers = builder.build_layers("System", "Doc", "Query", tools=[{"name": "tool1"}])
        assert len(layers) == 4
        types = [ly.layer_type for ly in layers]
        assert types == ["system", "tools", "document", "query"]

    def test_build_messages_format(self) -> None:
        builder = ContextLayerBuilder(max_breakpoints=4)
        messages = builder.build_messages("System prompt", "Document text", "My question")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_message_has_cache_control(self) -> None:
        builder = ContextLayerBuilder(max_breakpoints=4)
        messages = builder.build_messages("System prompt", "Document text", "Query")
        system_content = messages[0]["content"]
        # At least the system block should have cache_control
        has_cache = any("cache_control" in block for block in system_content)
        assert has_cache

    def test_user_message_is_plain_string(self) -> None:
        builder = ContextLayerBuilder()
        messages = builder.build_messages("System", "Doc", "What is X?")
        assert messages[1]["content"] == "What is X?"

    def test_empty_system_prompt(self) -> None:
        builder = ContextLayerBuilder()
        messages = builder.build_messages("", "Doc", "Query")
        # Should still have system for document + user for query
        assert len(messages) == 2

    def test_empty_document(self) -> None:
        builder = ContextLayerBuilder()
        messages = builder.build_messages("System", "", "Query")
        assert len(messages) == 2

    def test_max_breakpoints_respected(self) -> None:
        builder = ContextLayerBuilder(max_breakpoints=1)
        messages = builder.build_messages("System", "Doc", "Query")
        system_content = messages[0]["content"]
        cached_blocks = [b for b in system_content if "cache_control" in b]
        assert len(cached_blocks) <= 1
