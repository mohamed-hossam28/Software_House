"""
agents/data_base.py
───────────────────
Database Agent — a Coding agent: a name plus its profile. The coding loop
(agents/coding_loop.py) does the work; this module holds what is database-only.

The system prompt is rendered from the Persistence Contract: it names the
contract's database and ORM, the model directory and the connection module's
import line, and describes only ORM model files (the ORM creates the schema).
Database files are staged by task priority before dependencies.
"""
from __future__ import annotations

from agents.coding_loop import CodingAgent, CodingProfile, FileContext
from agents.schemas.database import DatabaseSchema
from core.entity_contract import render_entity_contract
from core.state import ArchData, PersistenceContract, connection_import_line, render_persistence_contract


def _db_tech(contract: PersistenceContract) -> str:
    """The contract's database and ORM, as named in prompts."""
    return f"{contract.database} ({contract.orm})"


def _orm_constraint_block(contract: PersistenceContract) -> str:
    """Instruction forcing the contract's single ORM paradigm."""
    orm = contract.orm
    return (
        f"## ORM: {orm}\n"
        f"IMPORTANT: Use {orm} as the single data-access layer. "
        "Do NOT use Prisma, Drizzle, TypeORM, raw DDL in parallel, or any other ORM. "
        "All models must use the same paradigm as the backend models."
    )


def _build_system_prompt(contract: PersistenceContract) -> str:
    """System prompt rendered from the Persistence Contract: ORM model files only."""
    orm = contract.orm
    return f"""\
You are an expert database engineer specialising in {_db_tech(contract)}.
Generate one complete {orm} ORM model file AND a structured manifest block.
Model files live under `{contract.model_dir}` (relative to the backend Layer root).
{orm} creates the schema from these models when the init command runs, so model
files are the only database files you write.

{_orm_constraint_block(contract)}

{render_persistence_contract(contract)}

## Model Rules
1.  Model, table and field names and field types MUST match the Entity Contract EXACTLY
    (snake_case throughout, no renaming, no pluralisation changes, no casing changes).
2.  Relationships reference only models planned in the Code Manifest.
    Respect priority/depends_on ordering.
3.  Primary keys use exactly the type the Entity Contract gives them — the backend's
    schemas and route parameters use the same type.
4.  Every model has `created_at` and `updated_at` timestamp fields unless the task
    explicitly says otherwise.
5.  Never invent models or fields not present in the task or schema overview.
    Suggested additions go in the manifest `suggested_additions` field only.
6.  Make every field's optionality explicit.
7.  Model files only declare models: they never create engines, sessions or clients and
    never read the database URL. Code that needs the database imports it with
    `{connection_import_line(contract)}`.
8.  The manifest lists the exact model name and exact field names + types — the backend
    reads it to use these models.

Return JSON: {{"migration_code":"<full source of the {orm} model file>","manifest":{{"table_name":"<tbl>","columns":[{{"name":"<col>","type":"<type>","nullable":true|false,"primary_key":true|false,"foreign_key":"<table.col or null>"}}],"indexes":["<col>"],"exports":["<tbl>"],"suggested_additions":[]}},"functions":[{{"name":"<func>","arguments":[{{"name":"<p>","type":"<t>","default":""}}],"return_type":"<t>","docstring":"<desc>"}}]}}
No prose outside the JSON object.
"""


def _sections(ctx: FileContext) -> list[str]:
    task = ctx.task
    depends = ", ".join(task.depends_on) if task and task.depends_on else "(none)"
    written = [
        f"- `{path}` exports: {', '.join(m.exports) or '(none)'}"
        for path, m in ctx.manifest.files.items()
        if m.written and m.agent == "database"
    ]
    return [
        "## Model Ordering\n"
        f"- **Depends On**: {depends}\n"
        f"- **Priority**  : {task.priority if task else '(none)'}",
        _orm_constraint_block(ctx.arch.persistence_contract),
        "## Architecture — Database Schema Overview\n"
        + (getattr(ctx.arch, "database_schema_overview", "") or "No schema overview provided."),
        render_entity_contract(ctx.arch.entities),
        "## Already Written Database Files (available for FK references)\n"
        + ("\n".join(written) or "None yet."),
    ]


def _system_prompt(arch: ArchData) -> str:
    return _build_system_prompt(arch.persistence_contract)


DATABASE_PROFILE = CodingProfile(
    agent="database",
    code_key="migration_code",
    response_schema=DatabaseSchema,
    system_prompt=_system_prompt,
    sections=_sections,
    tech=lambda arch: _db_tech(arch.persistence_contract),
    default_return_type="None",
    stage_by_priority=True,
)


class DatabaseAgent(CodingAgent):
    """The database Coding agent."""
    profile = DATABASE_PROFILE
