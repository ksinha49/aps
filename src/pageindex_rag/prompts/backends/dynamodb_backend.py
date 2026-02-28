"""DynamoDB-backed prompt storage with versioning and multi-dimensional resolution."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from pageindex_rag.prompts.context import PromptContext

logger = logging.getLogger(__name__)


class _TTLCache:
    """Thread-safe in-process cache with TTL expiration."""

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 500) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                return None
            return value

    def put(self, key: str, value: str) -> None:
        with self._lock:
            if len(self._store) >= self._max_size:
                self._evict_expired()
            self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (ts, _) in self._store.items() if now - ts > self._ttl]
        for k in expired:
            del self._store[k]


class DynamoDBPromptBackend:
    """Resolves prompts from a DynamoDB single-table with multi-dimensional lookup.

    Table schema::

        PK:  "{domain}#{category}#{name}"
        SK:  "v{version:04d}"

    GSI ``dimension-lookup``::

        PK:  dimension_key  ("lob#{lob}#dept#{dept}#uc#{uc}#proc#{proc}")
        SK:  "{domain}#{category}#{name}"

    Resolution cascade (GSI queries, most-specific first)::

        1. exact dimension match
        2. relax process → "*"
        3. relax use_case + process → "*"
        4. relax department + use_case + process → "*"
        5. all wildcards
        6. Base table PK lookup (latest version, no dimension filter)
    """

    def __init__(
        self,
        table_name: str,
        aws_region: str = "us-east-1",
        cache_ttl_seconds: float = 300.0,
        cache_max_size: int = 500,
        boto3_client: Any | None = None,
    ) -> None:
        self._table_name = table_name
        self._cache = _TTLCache(ttl_seconds=cache_ttl_seconds, max_size=cache_max_size)

        if boto3_client is not None:
            self._client = boto3_client
        else:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "boto3 is required for the DynamoDB prompt backend. "
                    "Install with: pip install pageindex-rag[dynamodb]"
                ) from exc
            self._client = boto3.client("dynamodb", region_name=aws_region)

    def get(
        self,
        domain: str,
        category: str,
        name: str,
        *,
        context: PromptContext | None = None,
        version: int | None = None,
    ) -> str:
        """Resolve a prompt from DynamoDB using the cascade algorithm."""
        pk = f"{domain}#{category}#{name}"

        # Specific version requested
        if version is not None:
            return self._get_by_version(pk, version)

        # Build cache key incorporating context
        ctx = context or PromptContext()
        cache_key = f"{pk}|{ctx.dimension_key()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Try dimension cascade via GSI
        result = self._cascade_lookup(pk, ctx)
        if result is not None:
            self._cache.put(cache_key, result)
            return result

        # Fallback: base table PK lookup (latest version)
        result = self._get_latest_from_base_table(pk)
        if result is not None:
            self._cache.put(cache_key, result)
            return result

        raise KeyError(f"Prompt {name!r} not found in DynamoDB ({domain}/{category})")

    def clear_cache(self) -> None:
        """Clear the in-process cache."""
        self._cache.clear()

    def _cascade_lookup(self, pk: str, ctx: PromptContext) -> str | None:
        """Try each relaxation level via the dimension-lookup GSI."""
        for dim_key in ctx.relaxation_cascade():
            result = self._query_gsi(dim_key, pk)
            if result is not None:
                return result
        return None

    def _query_gsi(self, dimension_key: str, prompt_key: str) -> str | None:
        """Query the dimension-lookup GSI for a specific dimension+prompt combo."""
        try:
            response = self._client.query(
                TableName=self._table_name,
                IndexName="dimension-lookup",
                KeyConditionExpression="dimension_key = :dk AND prompt_key = :pk",
                ExpressionAttributeValues={
                    ":dk": {"S": dimension_key},
                    ":pk": {"S": prompt_key},
                },
                ScanIndexForward=False,  # newest version first
                Limit=1,
            )
            items = response.get("Items", [])
            if items:
                return items[0]["prompt_text"]["S"]
        except Exception:
            logger.warning("DynamoDB GSI query failed for %s / %s", dimension_key, prompt_key, exc_info=True)
        return None

    def _get_by_version(self, pk: str, version: int) -> str:
        """Fetch a specific version from the base table."""
        sk = f"v{version:04d}"
        cache_key = f"{pk}|v{version}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            response = self._client.get_item(
                TableName=self._table_name,
                Key={"PK": {"S": pk}, "SK": {"S": sk}},
            )
            item = response.get("Item")
            if item:
                text = item["prompt_text"]["S"]
                self._cache.put(cache_key, text)
                return text
        except Exception:
            logger.warning("DynamoDB get_item failed for %s / %s", pk, sk, exc_info=True)

        raise KeyError(f"Prompt version {version} not found for {pk}")

    def _get_latest_from_base_table(self, pk: str) -> str | None:
        """Query the base table for the latest version of a prompt (no dimension filter)."""
        try:
            response = self._client.query(
                TableName=self._table_name,
                KeyConditionExpression="PK = :pk",
                ExpressionAttributeValues={":pk": {"S": pk}},
                ScanIndexForward=False,  # latest version first
                Limit=1,
            )
            items = response.get("Items", [])
            if items:
                return items[0]["prompt_text"]["S"]
        except Exception:
            logger.warning("DynamoDB base table query failed for %s", pk, exc_info=True)
        return None
