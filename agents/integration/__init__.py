"""
agents/integration/
───────────────────
Integration Agent sub-package — modular extraction of the monolithic
integration.py for maintainability and testability.

Re-exports `IntegrationAgent` and helper utilities so existing imports continue to work:
    from agents.integration import IntegrationAgent
"""

from agents.integration.agent import (
    IntegrationAgent,
    unambiguous_import_target,
    imported_names_for_code,
    target_available_names,
    _module_path_for,
)
from agents.integration.constants import REWORK_ONLY_TYPES
from agents.integration.types import normalize_api_path
from agents.integration.resolvers import match_backend_route

__all__ = [
    "IntegrationAgent",
    "REWORK_ONLY_TYPES",
    "normalize_api_path",
    "match_backend_route",
    "unambiguous_import_target",
    "imported_names_for_code",
    "target_available_names",
    "_module_path_for",
]
