"""Tests for FileRulesBackend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scout_ai.validation.backends.file_backend import FileRulesBackend
from scout_ai.validation.models import IssueSeverity, RuleCategory


@pytest.fixture()
def yaml_rules_path(tmp_path: Path) -> Path:
    """Create a temporary YAML rules file."""
    content = """
version: 2
rules:
  - rule_id: TEST-001
    name: Test rule 1
    description: First test rule
    category: data_integrity
    target: condition
    severity: error
    params:
      pattern: "^[A-Z]\\\\d{2}$"
  - rule_id: TEST-002
    name: Test rule 2
    description: Second test rule
    category: evidence_grounding
    target: finding
    severity: warning
    enabled: false
    params: {}
"""
    path = tmp_path / "test_rules.yaml"
    path.write_text(content)
    return path


@pytest.fixture()
def json_rules_path(tmp_path: Path) -> Path:
    """Create a temporary JSON rules file."""
    data = {
        "version": 3,
        "rules": [
            {
                "rule_id": "J-001",
                "name": "JSON rule",
                "description": "A JSON rule",
                "category": "medical_business",
                "target": "lab_result",
                "severity": "warning",
                "params": {"threshold": 7.0},
            }
        ],
    }
    path = tmp_path / "test_rules.json"
    path.write_text(json.dumps(data))
    return path


class TestFileRulesBackendYAML:
    def test_load_yaml(self, yaml_rules_path: Path) -> None:
        backend = FileRulesBackend(yaml_rules_path)
        rules = backend.list_rules()
        # Only enabled rules by default
        assert len(rules) == 1
        assert rules[0].rule_id == "TEST-001"

    def test_include_disabled(self, yaml_rules_path: Path) -> None:
        backend = FileRulesBackend(yaml_rules_path)
        rules = backend.list_rules(enabled_only=False)
        assert len(rules) == 2

    def test_filter_by_category(self, yaml_rules_path: Path) -> None:
        backend = FileRulesBackend(yaml_rules_path)
        rules = backend.list_rules(category=RuleCategory.DATA_INTEGRITY, enabled_only=False)
        assert len(rules) == 1
        assert rules[0].category == RuleCategory.DATA_INTEGRITY

    def test_get_rule(self, yaml_rules_path: Path) -> None:
        backend = FileRulesBackend(yaml_rules_path)
        rule = backend.get_rule("TEST-001")
        assert rule.name == "Test rule 1"
        assert rule.severity == IssueSeverity.ERROR

    def test_get_rule_not_found(self, yaml_rules_path: Path) -> None:
        backend = FileRulesBackend(yaml_rules_path)
        with pytest.raises(KeyError):
            backend.get_rule("NONEXISTENT")

    def test_get_version(self, yaml_rules_path: Path) -> None:
        backend = FileRulesBackend(yaml_rules_path)
        assert backend.get_version() == 2


class TestFileRulesBackendJSON:
    def test_load_json(self, json_rules_path: Path) -> None:
        backend = FileRulesBackend(json_rules_path)
        rules = backend.list_rules()
        assert len(rules) == 1
        assert rules[0].rule_id == "J-001"
        assert rules[0].params["threshold"] == 7.0

    def test_version(self, json_rules_path: Path) -> None:
        backend = FileRulesBackend(json_rules_path)
        assert backend.get_version() == 3


class TestFileRulesBackendErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        backend = FileRulesBackend(tmp_path / "nonexistent.yaml")
        with pytest.raises(FileNotFoundError):
            backend.list_rules()
