"""
core/entity_contract.py
───────────────────────
The Entity Contract: the field-level data model the architecture agent emits
(`ArchData.entities`) and the database and backend coding agents both code
against, so a field's type (above all a primary key's) is decided once.

The field-type vocabulary is closed (`FIELD_TYPES`) and each type maps, in
code, to its SQLAlchemy column type, Pydantic annotation and route-parameter
annotation (`TYPE_MAPPINGS`). This module is the one place both live, along
with how a name is matched to an entity (`match_entity`).
"""
from __future__ import annotations

import re
from typing import Iterable, Literal, Optional, get_args

from pydantic import BaseModel, Field

#: The closed field-type vocabulary.
FieldType = Literal["int", "uuid", "str", "text", "bool", "float", "datetime", "date", "json"]
FIELD_TYPES: tuple[str, ...] = get_args(FieldType)


class TypeMapping(BaseModel):
    """How one Entity Contract field type is written in code."""
    sqlalchemy: str              # SQLAlchemy column type, e.g. "Uuid"
    pydantic: str                # Pydantic field annotation, e.g. "uuid.UUID"
    route_param: Optional[str]   # FastAPI path-parameter annotation; None: not a route param


#: Field type → its code types. Covers every member of `FIELD_TYPES`.
TYPE_MAPPINGS: dict[str, TypeMapping] = {
    "int":      TypeMapping(sqlalchemy="Integer",  pydantic="int",               route_param="int"),
    "uuid":     TypeMapping(sqlalchemy="Uuid",     pydantic="uuid.UUID",         route_param="uuid.UUID"),
    "str":      TypeMapping(sqlalchemy="String",   pydantic="str",               route_param="str"),
    "text":     TypeMapping(sqlalchemy="Text",     pydantic="str",               route_param="str"),
    "bool":     TypeMapping(sqlalchemy="Boolean",  pydantic="bool",              route_param="bool"),
    "float":    TypeMapping(sqlalchemy="Float",    pydantic="float",             route_param="float"),
    "datetime": TypeMapping(sqlalchemy="DateTime", pydantic="datetime.datetime", route_param="datetime.datetime"),
    "date":     TypeMapping(sqlalchemy="Date",     pydantic="datetime.date",     route_param="datetime.date"),
    "json":     TypeMapping(sqlalchemy="JSON",     pydantic="dict",              route_param=None),
}


class EntityField(BaseModel):
    """One field of an entity."""
    name: str
    type: FieldType
    nullable: bool = False
    primary_key: bool = False
    max_length: Optional[int] = None
    default: Optional[str] = None


class Entity(BaseModel):
    """One persisted entity: its model name, table and fields."""
    name: str
    table: str
    fields: list[EntityField] = Field(default_factory=list)


# ── entity name matching ─────────────────────────────────────────────────────
#
# How a class, export, table, resource or parameter name is matched to an
# entity: shared by the planner (field contracts) and the entity-contract check.

#: Class-name suffixes stripped when matching a name to an entity, table or API
#: resource keyword (e.g. 'TodoItem' → 'todo', 'TodoCreate' → 'todo').
ENTITY_SUFFIXES: tuple[str, ...] = (
    "create", "new", "insert", "in", "update", "put", "patch", "request",
    "payload", "form", "response", "out", "read", "view", "item", "record",
    "entity", "model", "table", "orm", "dto", "schema", "detail", "result",
)

#: Suffixes of a request schema (the body a route accepts).
REQUEST_SUFFIXES: tuple[str, ...] = (
    "create", "new", "insert", "in", "update", "put", "patch", "request", "payload", "form",
)

#: Suffixes of a response schema (the body a route returns).
RESPONSE_SUFFIXES: tuple[str, ...] = (
    "response", "out", "read", "view", "item", "detail", "result",
)


def norm_keyword(name: str) -> str:
    """'TodoItem' → 'todoitem' (lowercase, alphanumerics only)."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def entity_forms(name: str) -> set[str]:
    """
    Normalized matching forms of a name: itself, its singular, and its
    suffix-stripped stem (e.g. 'TodoItem' → {'todoitem', 'todo'},
    'users' → {'users', 'user'}). Empty forms are dropped.
    """
    kw = norm_keyword(name)
    forms = {kw}
    if kw.endswith("s"):
        forms.add(kw[:-1])
    stripped = kw
    for suf in ENTITY_SUFFIXES:
        if stripped.endswith(suf):
            stripped = stripped[: -len(suf)]
            break
    forms.add(stripped)
    if stripped.endswith("s"):
        forms.add(stripped[:-1])
    return {f for f in forms if f}


def match_entity(name: str, candidates: Iterable[str]) -> str | None:
    """
    Match a name to EXACTLY ONE candidate key (entity, table or API resource).
    Deterministic only-match rule: no match, or >1 candidate → None (never guess).
    """
    forms = entity_forms(name)
    matched = [k for k in candidates if entity_forms(k) & forms]
    return matched[0] if len(matched) == 1 else None


def is_request_schema(name: str) -> bool:
    kw = norm_keyword(name)
    return any(kw.endswith(s) for s in REQUEST_SUFFIXES)


def is_response_schema(name: str) -> bool:
    kw = norm_keyword(name)
    return any(kw.endswith(s) for s in RESPONSE_SUFFIXES)


def _render_field(field: EntityField) -> str:
    mapping = TYPE_MAPPINGS[field.type]
    sqlalchemy = mapping.sqlalchemy
    if field.type == "str" and field.max_length:
        sqlalchemy = f"{sqlalchemy}({field.max_length})"
    flags = ["primary key"] if field.primary_key else []
    flags.append("nullable" if field.nullable else "not null")
    if field.default is not None:
        flags.append(f"default {field.default}")
    code = f"SQLAlchemy `{sqlalchemy}`, Pydantic `{mapping.pydantic}`"
    if field.primary_key and mapping.route_param:
        code += f", route param `{mapping.route_param}`"
    return f"- `{field.name}`: {field.type} ({', '.join(flags)}) → {code}"


def render_entity_contract(entities: list[Entity]) -> str:
    """The Entity Contract as a prompt section both coding agents receive."""
    lines = [
        "## Entity Contract (MUST match exactly)",
        "Every ORM model, request/response schema and route parameter uses exactly these "
        "entities, fields and types. A by-id route's parameter has the primary key's type.",
    ]
    if not entities:
        lines.append("(no entities specified)")
    for entity in entities:
        lines.append(f"### {entity.name} (table `{entity.table}`)")
        lines.extend(_render_field(f) for f in entity.fields)
    return "\n".join(lines)
