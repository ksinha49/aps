"""Workers' Compensation domain manifest — discovered by DomainRegistry."""

from __future__ import annotations

from scout_ai.domains.registry import DomainConfig
from scout_ai.domains.workers_comp.categories import CATEGORY_DESCRIPTIONS
from scout_ai.domains.workers_comp.models import SectionType

domain = DomainConfig(
    name="workers_comp",
    display_name="Workers' Compensation",
    description="Workers' compensation claim document extraction and analysis",
    section_types=[s.value for s in SectionType],
    category_descriptions={cat.value: desc for cat, desc in CATEGORY_DESCRIPTIONS.items()},
    prompts_module="scout_ai.domains.workers_comp.prompts",
)
