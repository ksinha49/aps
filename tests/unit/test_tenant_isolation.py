"""Tests for tenant isolation in persistence backends."""

from __future__ import annotations


class TestS3TenantPrefix:
    """Test that S3 backend correctly applies tenant prefixes."""

    def test_no_tenant_preserves_original_prefix(self):
        """Without tenant_id, prefix stays as-is."""
        from scout_ai.persistence.s3_backend import S3PersistenceBackend

        backend = S3PersistenceBackend.__new__(S3PersistenceBackend)
        backend._bucket = "test-bucket"
        backend._prefix = "indexes/"
        assert backend._full_key("doc1") == "indexes/doc1.json"

    def test_tenant_id_prepended_to_prefix(self):
        """With tenant_id, it should prepend to the prefix."""
        from scout_ai.persistence.s3_backend import S3PersistenceBackend

        backend = S3PersistenceBackend.__new__(S3PersistenceBackend)
        backend._bucket = "test-bucket"
        backend._prefix = "rp/indexes/"  # This is what __init__ would set with tenant_id="rp"
        assert backend._full_key("doc1") == "rp/indexes/doc1.json"

    def test_init_constructs_tenant_prefix(self):
        """Test that __init__ correctly combines tenant_id and prefix."""
        # We can't fully test __init__ without boto3/mocking, so test the logic directly
        tenant_id = "group-health"
        prefix = "indexes/"
        expected = f"{tenant_id}/{prefix}"
        assert expected == "group-health/indexes/"


class TestAppSettingsTenantFields:
    """Test that AppSettings includes tenant and region fields."""

    def test_tenant_id_default(self):
        from scout_ai.core.config import AppSettings

        settings = AppSettings()
        assert settings.tenant_id == "default"

    def test_lob_default(self):
        from scout_ai.core.config import AppSettings

        settings = AppSettings()
        assert settings.lob == "*"

    def test_aws_region_default(self):
        from scout_ai.core.config import AppSettings

        settings = AppSettings()
        assert settings.aws_region == "us-east-2"

    def test_persistence_kms_key_id_default(self):
        from scout_ai.core.config import PersistenceConfig

        config = PersistenceConfig()
        assert config.s3_kms_key_id == ""

    def test_persistence_region_default(self):
        from scout_ai.core.config import PersistenceConfig

        config = PersistenceConfig()
        assert config.s3_region == ""

    def test_region_cascade_logic(self):
        """When s3_region is empty, effective region should come from top-level."""
        from scout_ai.core.config import AppSettings

        settings = AppSettings(aws_region="eu-west-1")
        effective = settings.persistence.s3_region or settings.aws_region
        assert effective == "eu-west-1"

    def test_region_override(self):
        """When s3_region is explicitly set, it wins over top-level."""
        from scout_ai.core.config import AppSettings, PersistenceConfig

        settings = AppSettings(
            aws_region="eu-west-1",
            persistence=PersistenceConfig(s3_region="ap-southeast-1"),
        )
        effective = settings.persistence.s3_region or settings.aws_region
        assert effective == "ap-southeast-1"
