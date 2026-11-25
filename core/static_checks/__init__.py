"""
Static-check layer for the Integration Agent.

Entry point: run_all_checks(files, manifest, architecture) -> list[StaticFinding]

Pure functions over in-memory file contents — no LLM, no graph, no disk access
beyond the files passed in. Each checker module contributes a check function
with the uniform signature (files, Code Manifest, architecture) and is
registered in ALL_CHECKS; run_all_checks concatenates their findings. Database
specifics (such as the SQL dialect) come from the architecture's Persistence
Contract.
"""
from typing import Any

from core.state import StaticFinding
from core.static_checks.python_checker import check_python
from core.static_checks.js_checker import check_javascript
from core.static_checks.route_checker import check_backend_routes
from core.static_checks.sql_checker import check_sql
from core.static_checks.persistence_checker import check_persistence
from core.static_checks.html_checker import check_html_asset_links
from core.static_checks.entity_contract_checker import check_entity_contract

ALL_CHECKS = [
    check_python,
    check_javascript,
    check_backend_routes,
    check_sql,
    check_persistence,
    check_html_asset_links,
    check_entity_contract,
]


def run_all_checks(
    files: dict[str, str],
    manifest: Any,
    architecture: Any,
) -> list[StaticFinding]:
    """Run every registered checker over the generated files and return all findings."""
    findings: list[StaticFinding] = []
    for check in ALL_CHECKS:
        findings.extend(check(files, manifest, architecture))
    return findings
