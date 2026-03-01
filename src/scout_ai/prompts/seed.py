"""Seed DynamoDB prompt table from file-based prompt templates.

Scans all prompt template modules (aps and base domains), extracts every
uppercase string constant, and writes each as a v0001 item with all-wildcard
business dimensions.

Usage::

    python -m scout_ai.prompts.seed --table-name scout-prompts --region us-east-1
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from datetime import datetime, timezone
from types import ModuleType

import boto3

logger = logging.getLogger(__name__)

# ── Module registry ──────────────────────────────────────────────────
# Maps (domain, category) to the dotted import path of the template module.

_TEMPLATE_MODULES: dict[tuple[str, str], str] = {
    ("aps", "indexing"): "scout_ai.prompts.templates.aps.indexing",
    ("aps", "retrieval"): "scout_ai.prompts.templates.aps.retrieval",
    ("aps", "extraction"): "scout_ai.prompts.templates.aps.extraction",
    ("aps", "classification"): "scout_ai.prompts.templates.aps.classification",
    ("base", "indexing_agent"): "scout_ai.prompts.templates.base.indexing_agent",
    ("base", "retrieval_agent"): "scout_ai.prompts.templates.base.retrieval_agent",
    ("base", "extraction_agent"): "scout_ai.prompts.templates.base.extraction_agent",
}

# ── Helpers ───────────────────────────────────────────────────────────


def _extract_prompts(module: ModuleType) -> dict[str, str]:
    """Return all prompts from the module's ``_PROMPT_DATA`` dict.

    Reads directly from the ``_PROMPT_DATA`` dict to avoid triggering the
    module's ``__getattr__`` shim (which would go through the registry).
    """
    data = getattr(module, "_PROMPT_DATA", None)
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if isinstance(v, str)}
    return {}


def _build_item(
    domain: str,
    category: str,
    name: str,
    prompt_text: str,
    timestamp: str,
) -> dict[str, dict[str, str | bool]]:
    """Build a DynamoDB item dict (raw AttributeValue format)."""
    pk = f"{domain}#{category}#{name}"
    dimension_key = "lob#*#dept#*#uc#*#proc#*"
    return {
        "PK": {"S": pk},
        "SK": {"S": "v0001"},
        "prompt_text": {"S": prompt_text},
        "prompt_key": {"S": pk},
        "dimension_key": {"S": dimension_key},
        "lob": {"S": "*"},
        "department": {"S": "*"},
        "use_case": {"S": "*"},
        "process": {"S": "*"},
        "is_latest": {"BOOL": True},
        "created_at": {"S": timestamp},
        "created_by": {"S": "seed-script"},
    }


# ── Core logic ────────────────────────────────────────────────────────


def seed_prompts(table_name: str, region: str) -> int:
    """Write all file-based prompts to DynamoDB. Returns the count of items written."""
    client = boto3.client("dynamodb", region_name=region)
    timestamp = datetime.now(timezone.utc).isoformat()
    written = 0

    for (domain, category), module_path in _TEMPLATE_MODULES.items():
        logger.info("Loading module %s (domain=%s, category=%s)", module_path, domain, category)
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            logger.warning("Could not import %s -- skipping", module_path)
            continue

        prompts = _extract_prompts(module)
        if not prompts:
            logger.info("  No prompt constants found in %s", module_path)
            continue

        for name, prompt_text in prompts.items():
            item = _build_item(domain, category, name, prompt_text, timestamp)
            pk_display = f"{domain}#{category}#{name}"
            try:
                client.put_item(TableName=table_name, Item=item)
                written += 1
                logger.info("  Wrote %s", pk_display)
            except client.exceptions.ClientError as exc:
                logger.error("  Failed to write %s: %s", pk_display, exc)

    return written


# ── CLI ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m scout_ai.prompts.seed``."""
    parser = argparse.ArgumentParser(
        description="Seed DynamoDB prompt table from file-based templates.",
    )
    parser.add_argument(
        "--table-name",
        default="scout-prompts",
        help="DynamoDB table name (default: scout-prompts)",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    logger.info("Seeding table=%s in region=%s", args.table_name, args.region)
    count = seed_prompts(table_name=args.table_name, region=args.region)
    logger.info("Done. Wrote %d prompt items.", count)

    if count == 0:
        logger.warning("No prompts were written -- check module imports.")
        sys.exit(1)


if __name__ == "__main__":
    main()
