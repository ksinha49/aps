"""Tests for DynamoDB prompt backend with mocked boto3 client."""

from __future__ import annotations

from typing import Any

import pytest

from pageindex_rag.prompts.backends.dynamodb_backend import DynamoDBPromptBackend
from pageindex_rag.prompts.context import PromptContext
from pageindex_rag.prompts.registry import get_prompt, reset


class FakeDynamoDBClient:
    """Minimal mock of a boto3 DynamoDB client for testing."""

    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []
        self.queries: list[dict[str, Any]] = []

    def add_item(
        self,
        pk: str,
        sk: str,
        prompt_text: str,
        dimension_key: str = "lob#*#dept#*#uc#*#proc#*",
        prompt_key: str | None = None,
    ) -> None:
        self.items.append({
            "PK": {"S": pk},
            "SK": {"S": sk},
            "prompt_text": {"S": prompt_text},
            "dimension_key": {"S": dimension_key},
            "prompt_key": {"S": prompt_key or pk},
        })

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        pk = kwargs["Key"]["PK"]["S"]
        sk = kwargs["Key"]["SK"]["S"]
        for item in self.items:
            if item["PK"]["S"] == pk and item["SK"]["S"] == sk:
                return {"Item": item}
        return {}

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.queries.append(kwargs)
        index_name = kwargs.get("IndexName")
        expr_values = kwargs.get("ExpressionAttributeValues", {})

        if index_name == "dimension-lookup":
            dk = expr_values.get(":dk", {}).get("S", "")
            pk = expr_values.get(":pk", {}).get("S", "")
            matches = [
                item for item in self.items
                if item["dimension_key"]["S"] == dk and item["prompt_key"]["S"] == pk
            ]
        else:
            pk = expr_values.get(":pk", {}).get("S", "")
            matches = [item for item in self.items if item["PK"]["S"] == pk]

        # Sort by SK descending (latest version first)
        matches.sort(key=lambda x: x["SK"]["S"], reverse=True)
        limit = kwargs.get("Limit", len(matches))
        return {"Items": matches[:limit]}


@pytest.fixture()
def fake_client() -> FakeDynamoDBClient:
    return FakeDynamoDBClient()


@pytest.fixture()
def backend(fake_client: FakeDynamoDBClient) -> DynamoDBPromptBackend:
    return DynamoDBPromptBackend(
        table_name="test-prompts",
        boto3_client=fake_client,
        cache_ttl_seconds=0.1,  # short TTL for tests
    )


class TestDynamoDBBackendBasicLookup:
    """Basic version and base-table lookup."""

    def test_get_latest_version(self, fake_client: FakeDynamoDBClient, backend: DynamoDBPromptBackend) -> None:
        fake_client.add_item("aps#indexing#TOC_DETECT_PROMPT", "v0001", "prompt v1")
        fake_client.add_item("aps#indexing#TOC_DETECT_PROMPT", "v0002", "prompt v2")
        result = backend.get("aps", "indexing", "TOC_DETECT_PROMPT")
        assert result == "prompt v2"

    def test_get_specific_version(self, fake_client: FakeDynamoDBClient, backend: DynamoDBPromptBackend) -> None:
        fake_client.add_item("aps#indexing#TOC_DETECT_PROMPT", "v0001", "prompt v1")
        fake_client.add_item("aps#indexing#TOC_DETECT_PROMPT", "v0002", "prompt v2")
        result = backend.get("aps", "indexing", "TOC_DETECT_PROMPT", version=1)
        assert result == "prompt v1"

    def test_missing_prompt_raises(self, backend: DynamoDBPromptBackend) -> None:
        with pytest.raises(KeyError, match="not found"):
            backend.get("aps", "indexing", "NONEXISTENT")


class TestDynamoDBCascadeResolution:
    """Multi-dimensional cascade resolution via GSI."""

    def test_exact_match(self, fake_client: FakeDynamoDBClient, backend: DynamoDBPromptBackend) -> None:
        fake_client.add_item(
            "aps#indexing#TOC_DETECT_PROMPT", "v0001", "exact match",
            dimension_key="lob#life#dept#uw#uc#aps#proc#review",
            prompt_key="aps#indexing#TOC_DETECT_PROMPT",
        )
        ctx = PromptContext(lob="life", department="uw", use_case="aps", process="review")
        result = backend.get("aps", "indexing", "TOC_DETECT_PROMPT", context=ctx)
        assert result == "exact match"

    def test_relaxed_process(self, fake_client: FakeDynamoDBClient, backend: DynamoDBPromptBackend) -> None:
        fake_client.add_item(
            "aps#indexing#TOC_DETECT_PROMPT", "v0001", "relaxed process",
            dimension_key="lob#life#dept#uw#uc#aps#proc#*",
            prompt_key="aps#indexing#TOC_DETECT_PROMPT",
        )
        ctx = PromptContext(lob="life", department="uw", use_case="aps", process="review")
        result = backend.get("aps", "indexing", "TOC_DETECT_PROMPT", context=ctx)
        assert result == "relaxed process"

    def test_falls_through_to_wildcard(self, fake_client: FakeDynamoDBClient, backend: DynamoDBPromptBackend) -> None:
        fake_client.add_item(
            "aps#indexing#TOC_DETECT_PROMPT", "v0001", "wildcard",
            dimension_key="lob#*#dept#*#uc#*#proc#*",
            prompt_key="aps#indexing#TOC_DETECT_PROMPT",
        )
        ctx = PromptContext(lob="life", department="uw", use_case="aps", process="review")
        result = backend.get("aps", "indexing", "TOC_DETECT_PROMPT", context=ctx)
        assert result == "wildcard"

    def test_most_specific_wins(self, fake_client: FakeDynamoDBClient, backend: DynamoDBPromptBackend) -> None:
        fake_client.add_item(
            "aps#indexing#TOC_DETECT_PROMPT", "v0001", "exact",
            dimension_key="lob#life#dept#uw#uc#aps#proc#review",
            prompt_key="aps#indexing#TOC_DETECT_PROMPT",
        )
        fake_client.add_item(
            "aps#indexing#TOC_DETECT_PROMPT", "v0001", "wildcard",
            dimension_key="lob#*#dept#*#uc#*#proc#*",
            prompt_key="aps#indexing#TOC_DETECT_PROMPT",
        )
        ctx = PromptContext(lob="life", department="uw", use_case="aps", process="review")
        result = backend.get("aps", "indexing", "TOC_DETECT_PROMPT", context=ctx)
        assert result == "exact"


class TestDynamoDBCaching:
    """Cache behavior."""

    def test_cache_hit(self, fake_client: FakeDynamoDBClient, backend: DynamoDBPromptBackend) -> None:
        fake_client.add_item(
            "aps#indexing#TOC_DETECT_PROMPT", "v0001", "cached",
            dimension_key="lob#*#dept#*#uc#*#proc#*",
            prompt_key="aps#indexing#TOC_DETECT_PROMPT",
        )
        backend.get("aps", "indexing", "TOC_DETECT_PROMPT")
        query_count = len(fake_client.queries)
        backend.get("aps", "indexing", "TOC_DETECT_PROMPT")
        # Second call should not make more queries
        assert len(fake_client.queries) == query_count

    def test_clear_cache(self, fake_client: FakeDynamoDBClient, backend: DynamoDBPromptBackend) -> None:
        fake_client.add_item(
            "aps#indexing#TOC_DETECT_PROMPT", "v0001", "v1",
            dimension_key="lob#*#dept#*#uc#*#proc#*",
            prompt_key="aps#indexing#TOC_DETECT_PROMPT",
        )
        backend.get("aps", "indexing", "TOC_DETECT_PROMPT")
        backend.clear_cache()
        query_count = len(fake_client.queries)
        backend.get("aps", "indexing", "TOC_DETECT_PROMPT")
        assert len(fake_client.queries) > query_count


class TestRegistryWithDynamoDBFallback:
    """Registry-level DynamoDB backend with file fallback."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> None:  # type: ignore[misc]
        reset()
        yield  # type: ignore[misc]
        reset()

    def test_dynamodb_miss_falls_back_to_file(self) -> None:
        fake = FakeDynamoDBClient()
        from pageindex_rag.prompts import registry
        from pageindex_rag.prompts.backends.dynamodb_backend import DynamoDBPromptBackend as DDB
        from pageindex_rag.prompts.backends.file_backend import FilePromptBackend

        ddb = DDB(table_name="test", boto3_client=fake)
        registry._primary_backend = ddb
        registry._fallback_backend = FilePromptBackend()
        registry._configured = True

        result = get_prompt("aps", "indexing", "TOC_DETECT_PROMPT")
        assert "table of contents" in result.lower()
