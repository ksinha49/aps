"""S3 persistence backend — stores data in AWS S3."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import boto3

log = logging.getLogger(__name__)


class S3PersistenceBackend:
    """Stores data as objects in an S3 bucket."""

    def __init__(self, bucket: str, prefix: str = "indexes/", region: str = "us-west-2") -> None:
        try:
            import boto3 as _boto3
        except ImportError as e:
            raise ImportError(
                "boto3 is required for S3 persistence. "
                "Install with: pip install pageindex-rag[s3]"
            ) from e

        self._bucket = bucket
        self._prefix = prefix
        self._s3 = _boto3.client("s3", region_name=region)

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}.json"

    def save(self, key: str, data: str) -> None:
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._full_key(key),
            Body=data.encode("utf-8"),
            ContentType="application/json",
        )
        log.debug(f"Saved {key} to s3://{self._bucket}/{self._full_key(key)}")

    def load(self, key: str) -> str:
        try:
            response = self._s3.get_object(
                Bucket=self._bucket,
                Key=self._full_key(key),
            )
            return response["Body"].read().decode("utf-8")
        except self._s3.exceptions.NoSuchKey:
            raise KeyError(f"Not found in S3: {key}")

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(
                Bucket=self._bucket,
                Key=self._full_key(key),
            )
            return True
        except Exception:
            return False

    def delete(self, key: str) -> None:
        self._s3.delete_object(
            Bucket=self._bucket,
            Key=self._full_key(key),
        )

    def list_keys(self, prefix: str = "") -> list[str]:
        full_prefix = f"{self._prefix}{prefix}"
        response = self._s3.list_objects_v2(
            Bucket=self._bucket,
            Prefix=full_prefix,
        )
        keys = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.startswith(self._prefix):
                key = key[len(self._prefix):]
            if key.endswith(".json"):
                key = key[:-5]
            keys.append(key)
        return sorted(keys)
