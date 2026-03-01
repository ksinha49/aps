"""DynamoDB-backed rules storage with TTL caching.

Follows the same ``_TTLCache`` pattern as ``DynamoDBPromptBackend``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from scout_ai.validation.models import (
    IssueSeverity,
    Rule,
    RuleCategory,
    RuleTarget,
)

log = logging.getLogger(__name__)


class _TTLCache:
    """Thread-safe in-process cache with TTL expiration."""

    def __init__(self, ttl_seconds: float = 600.0, max_size: int = 200) -> None:
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.monotonic() - ts > self._ttl:
                del self._store[key]
                return None
            return value

    def put(self, key: str, value: Any) -> None:
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


class DynamoDBRulesBackend:
    """Resolves rules from a DynamoDB single-table.

    Table schema::

        PK:  "RULESET#v{version:04d}"
        SK:  "{rule_id}"

    GSI ``category-index``::

        PK:  "CAT#{category}"
        SK:  "{rule_id}"
    """

    def __init__(
        self,
        table_name: str,
        aws_region: str = "us-east-1",
        cache_ttl_seconds: float = 600.0,
        cache_max_size: int = 200,
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
                    "boto3 is required for the DynamoDB rules backend. "
                    "Install with: pip install scout-ai[dynamodb]"
                ) from exc
            self._client = boto3.client("dynamodb", region_name=aws_region)

    def list_rules(
        self,
        *,
        category: RuleCategory | None = None,
        enabled_only: bool = True,
        version: int | None = None,
        lob: str = "*",
    ) -> list[Rule]:
        """Return rules from DynamoDB, optionally filtered by category."""
        cache_key = f"list|{category}|{enabled_only}|{version}|{lob}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        if category is not None:
            rules = self._query_by_category(category)
        else:
            rules = self._query_all(version)

        if enabled_only:
            rules = [r for r in rules if r.enabled]

        self._cache.put(cache_key, rules)
        return rules

    def get_rule(self, rule_id: str, *, version: int | None = None) -> Rule:
        """Get a single rule by ID."""
        cache_key = f"rule|{rule_id}|{version}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        effective_version = version or self.get_version()
        pk = f"RULESET#v{effective_version:04d}"

        try:
            response = self._client.get_item(
                TableName=self._table_name,
                Key={"PK": {"S": pk}, "SK": {"S": rule_id}},
            )
            item = response.get("Item")
            if item:
                rule = self._parse_item(item)
                self._cache.put(cache_key, rule)
                return rule
        except Exception:
            log.warning("DynamoDB get_item failed for %s / %s", pk, rule_id, exc_info=True)

        raise KeyError(f"Rule {rule_id!r} not found in DynamoDB")

    def get_version(self) -> int:
        """Return the latest ruleset version by scanning PK prefixes."""
        cache_key = "latest_version"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached  # type: ignore[return-value]

        try:
            response = self._client.query(
                TableName=self._table_name,
                KeyConditionExpression="begins_with(PK, :prefix)",
                ExpressionAttributeValues={":prefix": {"S": "RULESET#v"}},
                ScanIndexForward=False,
                Limit=1,
                ProjectionExpression="PK",
            )
            items = response.get("Items", [])
            if items:
                pk = items[0]["PK"]["S"]
                ver = int(pk.split("v")[1])
                self._cache.put(cache_key, ver)
                return ver
        except Exception:
            log.warning("Failed to determine latest version", exc_info=True)

        return 1

    def clear_cache(self) -> None:
        """Clear the in-process cache."""
        self._cache.clear()

    def _query_by_category(self, category: RuleCategory) -> list[Rule]:
        """Query the category-index GSI."""
        try:
            response = self._client.query(
                TableName=self._table_name,
                IndexName="category-index",
                KeyConditionExpression="category_key = :ck",
                ExpressionAttributeValues={":ck": {"S": f"CAT#{category.value}"}},
            )
            return [self._parse_item(item) for item in response.get("Items", [])]
        except Exception:
            log.warning("DynamoDB category query failed for %s", category.value, exc_info=True)
            return []

    def _query_all(self, version: int | None = None) -> list[Rule]:
        """Query all rules for a given version."""
        effective_version = version or self.get_version()
        pk = f"RULESET#v{effective_version:04d}"

        try:
            response = self._client.query(
                TableName=self._table_name,
                KeyConditionExpression="PK = :pk",
                ExpressionAttributeValues={":pk": {"S": pk}},
            )
            return [self._parse_item(item) for item in response.get("Items", [])]
        except Exception:
            log.warning("DynamoDB query failed for %s", pk, exc_info=True)
            return []

    @staticmethod
    def _parse_item(item: dict[str, Any]) -> Rule:
        """Parse a DynamoDB item into a Rule."""
        params_raw = item.get("params", {}).get("S", "{}")
        return Rule(
            rule_id=item["SK"]["S"],
            name=item.get("name", {}).get("S", ""),
            description=item.get("description", {}).get("S", ""),
            category=RuleCategory(item.get("category", {}).get("S", "data_integrity")),
            target=RuleTarget(item.get("target", {}).get("S", "summary")),
            severity=IssueSeverity(item.get("severity", {}).get("S", "warning")),
            enabled=item.get("enabled", {}).get("BOOL", True),
            params=json.loads(params_raw) if isinstance(params_raw, str) else {},
            version=int(item.get("version", {}).get("N", "1")),
        )
