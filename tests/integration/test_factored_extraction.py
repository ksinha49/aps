"""Integration test: full pipeline with multi-layer caching (context factoring)."""

from __future__ import annotations

from scout_ai.context.factoring.layer_builder import ContextLayerBuilder


class TestFactoredExtraction:
    """Verify context factoring produces correct message structures."""

    def test_layered_messages_compatible_with_llm_client(self) -> None:
        """Messages from ContextLayerBuilder should match LLMClient format."""
        builder = ContextLayerBuilder(max_breakpoints=4)
        system = "You are an extraction assistant."
        doc = "Patient has diabetes. HbA1c is 7.2%."
        query = "What is the patient's HbA1c level?"

        messages = builder.build_messages(system, doc, query)

        # Should have system + user messages
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # System should have multiple content blocks
        system_content = messages[0]["content"]
        assert isinstance(system_content, list)
        assert len(system_content) >= 1

        # User message should be plain string
        assert messages[1]["content"] == query

    def test_layered_messages_have_cache_control(self) -> None:
        """At least one system block should have cache_control."""
        builder = ContextLayerBuilder(max_breakpoints=4)
        messages = builder.build_messages("System", "Document", "Query")

        system_content = messages[0]["content"]
        cached = [b for b in system_content if "cache_control" in b]
        assert len(cached) >= 1
        assert cached[0]["cache_control"] == {"type": "ephemeral"}

    def test_single_breakpoint_mode(self) -> None:
        """With max_breakpoints=1, only one block should be cached."""
        builder = ContextLayerBuilder(max_breakpoints=1)
        messages = builder.build_messages("System", "Document", "Query")

        system_content = messages[0]["content"]
        cached = [b for b in system_content if "cache_control" in b]
        assert len(cached) == 1

    def test_document_context_in_system_message(self) -> None:
        """Document context should be in the system message for caching."""
        builder = ContextLayerBuilder()
        messages = builder.build_messages("System", "My document text", "Query")

        system_content = messages[0]["content"]
        all_text = " ".join(b["text"] for b in system_content)
        assert "My document text" in all_text
