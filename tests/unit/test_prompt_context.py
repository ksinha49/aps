"""Tests for PromptContext dataclass and relaxation cascade."""

from __future__ import annotations

from scout_ai.prompts.context import PromptContext


class TestPromptContext:
    """PromptContext construction and dimension key generation."""

    def test_defaults_all_wildcards(self) -> None:
        ctx = PromptContext()
        assert ctx.lob == "*"
        assert ctx.department == "*"
        assert ctx.use_case == "*"
        assert ctx.process == "*"

    def test_custom_values(self) -> None:
        ctx = PromptContext(lob="life", department="uw", use_case="aps", process="review")
        assert ctx.lob == "life"
        assert ctx.department == "uw"
        assert ctx.use_case == "aps"
        assert ctx.process == "review"

    def test_dimension_key_format(self) -> None:
        ctx = PromptContext(lob="life", department="uw", use_case="aps", process="review")
        assert ctx.dimension_key() == "lob#life#dept#uw#uc#aps#proc#review"

    def test_dimension_key_wildcards(self) -> None:
        ctx = PromptContext()
        assert ctx.dimension_key() == "lob#*#dept#*#uc#*#proc#*"

    def test_frozen(self) -> None:
        ctx = PromptContext(lob="life")
        try:
            ctx.lob = "health"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestRelaxationCascade:
    """Relaxation cascade produces correct order of dimension keys."""

    def test_fully_specified(self) -> None:
        ctx = PromptContext(lob="life", department="uw", use_case="aps", process="review")
        keys = ctx.relaxation_cascade()
        assert keys == [
            "lob#life#dept#uw#uc#aps#proc#review",  # exact
            "lob#life#dept#uw#uc#aps#proc#*",         # relax process
            "lob#life#dept#uw#uc#*#proc#*",            # relax uc + process
            "lob#life#dept#*#uc#*#proc#*",             # relax dept + uc + process
            "lob#*#dept#*#uc#*#proc#*",                # all wildcards
        ]

    def test_already_wildcards_deduplicates(self) -> None:
        ctx = PromptContext()
        keys = ctx.relaxation_cascade()
        # All steps collapse to the same key
        assert keys == ["lob#*#dept#*#uc#*#proc#*"]

    def test_partial_specification(self) -> None:
        ctx = PromptContext(lob="life", department="*", use_case="*", process="*")
        keys = ctx.relaxation_cascade()
        assert keys == [
            "lob#life#dept#*#uc#*#proc#*",
            "lob#*#dept#*#uc#*#proc#*",
        ]

    def test_only_lob_and_process(self) -> None:
        ctx = PromptContext(lob="life", process="review")
        keys = ctx.relaxation_cascade()
        assert keys == [
            "lob#life#dept#*#uc#*#proc#review",
            "lob#life#dept#*#uc#*#proc#*",
            "lob#*#dept#*#uc#*#proc#*",
        ]
