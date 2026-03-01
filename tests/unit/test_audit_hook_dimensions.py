"""Tests for AuditHook multi-tenant dimension support."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

from scout_ai.hooks.audit_hook import AuditHook


class TestAuditHookDimensionInit:
    """Constructor and attribute storage for tenant_id / lob / domain."""

    def test_audit_hook_accepts_tenant_dimensions(self) -> None:
        """Constructor with tenant_id, lob, domain succeeds."""
        hook = AuditHook(tenant_id="t-123", lob="group", domain="aps")
        assert isinstance(hook, AuditHook)

    def test_audit_hook_default_dimensions_empty(self) -> None:
        """No args produces empty-string defaults."""
        hook = AuditHook()
        assert hook._tenant_id == ""
        assert hook._lob == ""
        assert hook._domain == ""

    def test_audit_hook_stores_dimensions(self) -> None:
        """Values are stored on the instance attributes."""
        hook = AuditHook(tenant_id="acme", lob="individual", domain="claims")
        assert hook._tenant_id == "acme"
        assert hook._lob == "individual"
        assert hook._domain == "claims"


class TestAuditHookDimensionLogging:
    """Verify dimensions appear in log output for model and tool calls."""

    def test_audit_hook_dimensions_in_model_call_log(self, caplog: logging.LogRecaptureFixture) -> None:  # type: ignore[name-defined]
        """Model-call log line includes tenant_id, lob, and domain."""
        hook = AuditHook(tenant_id="t-999", lob="group", domain="aps")

        event = MagicMock()
        event.model_id = "claude-v3"
        event.usage = {
            "inputTokens": 100,
            "outputTokens": 50,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        event.latency_ms = 42

        with caplog.at_level(logging.INFO, logger="scout_ai.hooks.audit_hook"):
            hook._on_model_call(event)

        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert "tenant_id=t-999" in msg
        assert "lob=group" in msg
        assert "domain=aps" in msg
        # Existing fields still present
        assert "model=claude-v3" in msg
        assert "prompt_tokens=100" in msg

    def test_audit_hook_dimensions_in_tool_call_log(self, caplog: logging.LogRecaptureFixture) -> None:  # type: ignore[name-defined]
        """Tool-call log line includes tenant_id, lob, and domain."""
        hook = AuditHook(tenant_id="t-888", lob="commercial", domain="claims")

        event = MagicMock()
        event.tool_name = "detect_toc"
        event.status = "success"
        event.latency_ms = 15

        with caplog.at_level(logging.INFO, logger="scout_ai.hooks.audit_hook"):
            hook._on_tool_call(event)

        assert len(caplog.records) == 1
        msg = caplog.records[0].getMessage()
        assert "tenant_id=t-888" in msg
        assert "lob=commercial" in msg
        assert "domain=claims" in msg
        # Existing fields still present
        assert "tool=detect_toc" in msg
        assert "status=success" in msg
