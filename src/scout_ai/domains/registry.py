"""Convention-based domain registry with auto-discovery.

Each domain is a Python package under ``scout_ai/domains/`` with a
``__domain__.py`` manifest that exposes a ``domain`` attribute of type
``DomainConfig``.

Usage::

    from scout_ai.domains.registry import DomainRegistry

    registry = DomainRegistry()
    registry.auto_discover()
    aps = registry.get("aps")
    print(aps.display_name)      # "Attending Physician Statement"
    print(aps.category_descriptions)  # {str: str, ...}
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DomainConfig:
    """Manifest for a domain module.

    All dotted-path references (e.g. ``prompts_module``,
    ``synthesis_pipeline``) are lazy-loaded on first access via
    :meth:`resolve`.
    """

    name: str
    display_name: str
    description: str = ""

    # Section types recognized by this domain (e.g. medical section types)
    section_types: list[str] = field(default_factory=list)

    # Maps category string -> human-readable description
    category_descriptions: dict[str, str] = field(default_factory=dict)

    # Dotted-path references for lazy import
    prompts_module: str = ""
    synthesis_pipeline: str = ""
    validation_engine: str = ""
    classifier: str = ""
    config_class: str = ""
    formatters: dict[str, str] = field(default_factory=dict)

    def resolve(self, attr: str) -> Any:
        """Lazy-import and return the object referenced by a dotted path.

        Args:
            attr: One of the dotted-path fields (e.g. ``"synthesis_pipeline"``).

        Returns:
            The resolved Python object (class, function, module, etc.).

        Raises:
            ValueError: If the attribute is empty or not a dotted-path field.
            ImportError: If the module/object cannot be found.
        """
        dotted = getattr(self, attr, "")
        if not dotted:
            raise ValueError(f"Domain {self.name!r} has no {attr!r} configured")
        return _import_dotted_path(dotted)


class DomainRegistry:
    """Registry of discovered domain modules.

    Domains are registered either manually via :meth:`register` or
    automatically via :meth:`auto_discover`, which scans
    ``scout_ai.domains`` sub-packages for ``__domain__.py`` manifests.
    """

    def __init__(self) -> None:
        self._domains: dict[str, DomainConfig] = {}

    def register(self, config: DomainConfig) -> None:
        """Register a domain config."""
        if config.name in self._domains:
            log.warning("Domain %r already registered, overwriting", config.name)
        self._domains[config.name] = config
        log.debug("Registered domain: %s", config.name)

    def get(self, name: str) -> DomainConfig:
        """Get a domain config by name.

        Raises:
            KeyError: If the domain is not registered.
        """
        if name not in self._domains:
            raise KeyError(
                f"Domain {name!r} not found. "
                f"Available: {sorted(self._domains.keys())}"
            )
        return self._domains[name]

    def list_domains(self) -> list[DomainConfig]:
        """Return all registered domain configs, sorted by name."""
        return sorted(self._domains.values(), key=lambda d: d.name)

    def has(self, name: str) -> bool:
        """Check if a domain is registered."""
        return name in self._domains

    def auto_discover(self) -> None:
        """Scan ``scout_ai.domains`` sub-packages for ``__domain__`` manifests.

        Each sub-package is expected to have a ``__domain__.py`` module with a
        module-level ``domain`` attribute of type :class:`DomainConfig`.
        """
        import scout_ai.domains as domains_pkg

        for importer, modname, ispkg in pkgutil.iter_modules(
            domains_pkg.__path__, prefix="scout_ai.domains."
        ):
            if not ispkg:
                continue

            domain_module_name = f"{modname}.__domain__"
            try:
                mod = importlib.import_module(domain_module_name)
            except ImportError:
                log.debug("No __domain__.py in %s, skipping", modname)
                continue

            config = getattr(mod, "domain", None)
            if not isinstance(config, DomainConfig):
                log.warning(
                    "%s.__domain__.domain is not a DomainConfig, skipping",
                    modname,
                )
                continue

            self.register(config)

        log.info(
            "Auto-discovered %d domain(s): %s",
            len(self._domains),
            ", ".join(sorted(self._domains.keys())),
        )


# ── Module-level singleton ──────────────────────────────────────────

_global_registry: DomainRegistry | None = None


def get_registry() -> DomainRegistry:
    """Return the global domain registry, auto-discovering on first call."""
    global _global_registry
    if _global_registry is None:
        _global_registry = DomainRegistry()
        _global_registry.auto_discover()
    return _global_registry


# ── Internal helpers ────────────────────────────────────────────────


def _import_dotted_path(dotted: str) -> Any:
    """Import ``module.path:ClassName`` or ``module.path.attr``."""
    if ":" in dotted:
        module_path, obj_name = dotted.rsplit(":", 1)
    elif "." in dotted:
        module_path, obj_name = dotted.rsplit(".", 1)
    else:
        return importlib.import_module(dotted)

    mod = importlib.import_module(module_path)
    return getattr(mod, obj_name)
