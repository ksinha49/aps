"""APS domain manifest — discovered by DomainRegistry.auto_discover()."""

from __future__ import annotations

from scout_ai.domains.aps.categories import CATEGORY_DESCRIPTIONS
from scout_ai.domains.aps.models import MedicalSectionType
from scout_ai.domains.registry import DomainConfig

domain = DomainConfig(
    name="aps",
    display_name="Attending Physician Statement",
    description="Medical record extraction for life insurance underwriting",
    section_types=[m.value for m in MedicalSectionType],
    category_descriptions={cat.value: desc for cat, desc in CATEGORY_DESCRIPTIONS.items()},
    prompts_module="scout_ai.prompts.templates.aps",
    synthesis_pipeline="scout_ai.domains.aps.synthesis.pipeline:SynthesisPipeline",
    validation_engine="scout_ai.domains.aps.validation.engine:RulesEngine",
    classifier="scout_ai.domains.aps.classifier:MedicalSectionClassifier",
    config_class="scout_ai.core.config:AppSettings",
    formatters={
        "pdf": "scout_ai.formatters.pdf_formatter:PDFFormatter",
        "json": "scout_ai.formatters.json_formatter:JSONFormatter",
    },
)
