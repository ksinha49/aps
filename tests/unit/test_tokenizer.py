"""Unit tests for the token counter."""

import pytest

from scout_ai.exceptions import TokenizerError
from scout_ai.providers.pageindex.tokenizer import TokenCounter


class TestApproximateTokenizer:
    def test_empty_string(self):
        tc = TokenCounter(method="approximate")
        assert tc.count("") == 0

    def test_short_string(self):
        tc = TokenCounter(method="approximate")
        # "hello" is 5 chars => 5 // 4 = 1
        assert tc.count("hello") == 1

    def test_longer_string(self):
        tc = TokenCounter(method="approximate")
        text = "a" * 400
        assert tc.count(text) == 100

    def test_realistic_text(self):
        tc = TokenCounter(method="approximate")
        text = "The patient presents with chronic lower back pain radiating to the left leg."
        count = tc.count(text)
        # ~76 chars / 4 = 19
        assert 15 <= count <= 25

    def test_none_text_returns_zero(self):
        tc = TokenCounter(method="approximate")
        assert tc.count("") == 0


class TestTiktokenUnavailable:
    def test_import_error_when_missing(self, monkeypatch):
        """Verify graceful error when tiktoken is not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tiktoken":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(TokenizerError, match="tiktoken not installed"):
            TokenCounter(method="tiktoken")
