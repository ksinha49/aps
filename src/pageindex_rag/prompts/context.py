"""Prompt resolution context for multi-dimensional prompt lookup.

A ``PromptContext`` encodes the business dimensions (LOB, department, use-case,
process) used to resolve the most specific prompt variant from DynamoDB.

The ``relaxation_cascade()`` method yields dimension keys from most-specific to
least-specific, allowing the backend to try exact matches first, then
progressively broader wildcards.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptContext:
    """Business dimensions for multi-dimensional prompt resolution.

    Each dimension defaults to ``"*"`` (wildcard = match any).
    """

    lob: str = "*"
    department: str = "*"
    use_case: str = "*"
    process: str = "*"

    def dimension_key(self) -> str:
        """Return a canonical dimension key for GSI lookups.

        Format: ``lob#{lob}#dept#{dept}#uc#{uc}#proc#{proc}``
        """
        return f"lob#{self.lob}#dept#{self.department}#uc#{self.use_case}#proc#{self.process}"

    def relaxation_cascade(self) -> list[str]:
        """Return dimension keys in order from most-specific to least-specific.

        Relaxation order (right-to-left):
        1. exact match
        2. relax process -> ``"*"``
        3. relax use_case + process -> ``"*"``
        4. relax department + use_case + process -> ``"*"``
        5. all wildcards
        """
        steps: list[PromptContext] = [
            self,
            PromptContext(lob=self.lob, department=self.department, use_case=self.use_case, process="*"),
            PromptContext(lob=self.lob, department=self.department, use_case="*", process="*"),
            PromptContext(lob=self.lob, department="*", use_case="*", process="*"),
            PromptContext(),  # all wildcards
        ]
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for ctx in steps:
            key = ctx.dimension_key()
            if key not in seen:
                seen.add(key)
                result.append(key)
        return result
