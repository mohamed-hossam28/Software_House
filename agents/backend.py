"""
agents/backend.py
─────────────────
Backend Agent — a Coding agent: a name plus its profile. The coding loop
(agents/coding_loop.py) does the work; this module holds what is backend-only:
the system prompt (backend tech + the Persistence Contract and its rules) and
the backend sections of a file's request (schema overview, API contracts,
layer rules).
"""
from __future__ import annotations

from typing import Any

from agents.coding_loop import CodingAgent, CodingProfile, FileContext, filter_api_contracts_by_relevance
from agents.schemas.backend import BackendSchema
from core.entity_contract import render_entity_contract
from core.state import PersistenceContract, backend_persistence_rules, render_persistence_contract

DEFAULT_BACKEND_TECH = "FastAPI (Python 3.12)"


def _backend_tech(arch: Any) -> str:
    stack = getattr(arch, "tech_stack", None) or {}
    return stack.get("backend", DEFAULT_BACKEND_TECH)


def _build_system_prompt(backend_tech: str, contract: PersistenceContract) -> str:
    """Builds a system prompt tailored to the backend tech, carrying the persistence contract + mandate."""
    persistence = render_persistence_contract(contract) + "\n\n" + backend_persistence_rules(contract)
    return f"""\
You are an expert backend engineer specializing in {backend_tech}.
Your job is to generate the complete source code for a backend file and its manifest details.

## Rules:
1. Return a single valid JSON object containing "code", "classes", and "functions". No prose outside the JSON.
2. The "code" string must be complete, syntax-valid source code ready to write to disk. No placeholders, no TODOs.
3. You MUST define and export all classes and functions listed in the "Required Exports" list.
4. You must NOT import from any file path not listed in the Project Code Manifest.
5. All imports/requires must follow standard patterns for {backend_tech} based on the file paths in the manifest.
6. Follow the API contracts exactly (HTTP methods, paths, request/response models).
7. Write clean, idiomatic code utilizing features of {backend_tech}.
8. Include robust error handling and return appropriate error responses / HTTP status codes for {backend_tech}.
Return JSON: {{"code":"<full source>","classes":[{{"name":"<class name>","arguments":[{{"name":"<param>","type":"<type>","default":""}}],"docstring":"<desc>"}}],"functions":[{{"name":"<func name>","arguments":[{{"name":"<param>","type":"<type>","default":""}}],"return_type":"<type>","docstring":"<desc>"}}]}}

{persistence}
"""


def _system_prompt(arch: Any) -> str:
    """System prompt for this project: backend tech + persistence contract and mandate."""
    return _build_system_prompt(_backend_tech(arch), arch.persistence_contract)


def _api_contracts(ctx: FileContext) -> str:
    contracts = filter_api_contracts_by_relevance(
        getattr(ctx.arch, "api_contracts", []) or [],
        filepath=ctx.filepath,
        purpose=ctx.fm.purpose,
        task_title=ctx.task.title if ctx.task else "",
    )
    if not contracts:
        return "No relevant API contracts specified."
    lines = []
    for c in contracts:
        lock = "🔒" if getattr(c, "auth_required", False) else "🔓"
        lines.append(
            f"- {lock} {c.method} {c.path}\n"
            f"  Description: {c.description}\n"
            f"  Request Body: {c.request_body}\n"
            f"  Response Body: {c.response_body}"
        )
    return "\n".join(lines)


def _layer_rules(ctx: FileContext) -> str:
    """Architecture rules pinned by the plan: shared schemas/ module, thin routers over services/."""
    rules = []
    if any("/schemas/" in p for p in ctx.manifest.files):
        rules.append(
            "**SCHEMA RULE (single source of truth)**: All request/response models MUST be "
            "defined in the shared `schemas/` module and IMPORTED from there. NEVER redefine "
            "a request/response model class inside a router, service, or other file — that "
            "creates contract drift (e.g. `completed` vs `is_completed`)."
        )
    if any("/services/" in p for p in ctx.manifest.files):
        rules.append(
            "**SERVICE LAYER RULE**: Routers MUST be thin — parse the request, call the "
            "matching function from the `services/` module, return its result. All data "
            "access and business logic (DB sessions, queries, commits) lives in `services/`. "
            "Do not perform data access inline in a router."
        )
    return "\n".join(rules) or "(no layer-wide architecture rules for this project)"


def _sections(ctx: FileContext) -> list[str]:
    return [
        "## Database Schema Overview\n"
        + (getattr(ctx.arch, "database_schema_overview", "") or "No DB schema provided."),
        render_entity_contract(ctx.arch.entities),
        "## API Contracts\n" + _api_contracts(ctx),
        "## Layer Architecture Rules (MUST follow)\n" + _layer_rules(ctx),
    ]


BACKEND_PROFILE = CodingProfile(
    agent="backend",
    code_key="code",
    response_schema=BackendSchema,
    system_prompt=_system_prompt,
    sections=_sections,
    tech=lambda arch: (getattr(arch, "tech_stack", None) or {}).get("backend", "FastAPI"),
    default_return_type="None",
)


class BackendAgent(CodingAgent):
    """The backend Coding agent."""
    profile = BACKEND_PROFILE
