"""
The Entity Contract: the architecture agent's structured, field-level data
model, shared by the database and backend coding agents so a primary key's
type (and every other field's) is decided once.
"""
import json

import pytest
from pydantic import ValidationError

from agents.architecture import ArchitectureAgent
from agents.schemas.architecture import DbSchemaOverviewSchema
from agents.backend import BackendAgent
from agents.data_base import DatabaseAgent
from core.entity_contract import FIELD_TYPES, TYPE_MAPPINGS, EntityField, render_entity_contract
from core.state import ArchData, FileManifest, ProjectState, SRSData, Task
from providers.llm.base import BaseLLMProvider, LLMResponse
from providers.llm.scripted import ScriptedLLMProvider

TODO_ENTITY = {
    "name": "Todo",
    "table": "todos",
    "fields": [
        {"name": "id", "type": "uuid", "nullable": False, "primary_key": True},
        {"name": "title", "type": "str", "nullable": False, "primary_key": False, "max_length": 200},
        {"name": "done", "type": "bool", "nullable": False, "primary_key": False, "default": "false"},
    ],
}


def _with_field_type(field_type: str) -> dict:
    entity = {**TODO_ENTITY, "fields": [dict(f) for f in TODO_ENTITY["fields"]]}
    entity["fields"][0]["type"] = field_type
    return {"database_schema_overview": "todos", "entities": [entity]}


# ── Schema validation ─────────────────────────────────────────────────────────

def test_architecture_db_reply_with_entities_validates():
    reply = DbSchemaOverviewSchema(**_with_field_type("uuid"))

    assert reply.entities[0].fields[0].type == "uuid"
    assert reply.entities[0].fields[1].max_length == 200


@pytest.mark.parametrize("field_type", ["varchar", "UUID", "integer", "string", ""])
def test_architecture_db_reply_with_an_unknown_field_type_is_rejected(field_type):
    with pytest.raises(ValidationError):
        DbSchemaOverviewSchema(**_with_field_type(field_type))


def test_every_field_type_maps_to_code_types():
    assert set(TYPE_MAPPINGS) == set(FIELD_TYPES) == {
        "int", "uuid", "str", "text", "bool", "float", "datetime", "date", "json",
    }


# ── The architecture agent emits the contract ─────────────────────────────────

class _EntityArchitectureLLM(BaseLLMProvider):
    """Answers every architecture sub-call with one body carrying the Todo entity."""

    BODY = {
        "tech_stack": {},
        "system_components": ["TodoService"],
        "security_requirements": [],
        "database_schema_overview": "todos",
        "entities": [TODO_ENTITY],
        "api_contracts": [],
    }

    @property
    def provider_name(self) -> str:
        return "mock-entity-architecture"

    def complete(self, system_prompt, user_message, model, max_tokens=8192, temperature=0.2, **kwargs):
        return LLMResponse(text=json.dumps(self.BODY), model="mock", provider=self.provider_name)


def test_architecture_agent_emits_the_entity_contract():
    state = ProjectState(project_id="arch-entities", project_name="Todo")
    state.requirements = SRSData(raw_prompt="A todo app.", functional_requirements=["Manage todos"])

    diff = ArchitectureAgent(provider=_EntityArchitectureLLM()).run(state)
    arch = ArchData(**diff["architecture"])

    assert [e.name for e in arch.entities] == ["Todo"]
    assert arch.entities[0].table == "todos"
    assert arch.entities[0].fields[0] == EntityField(name="id", type="uuid", nullable=False, primary_key=True)


def test_architecture_data_rejects_an_unknown_field_type():
    entity = {**TODO_ENTITY, "fields": [{"name": "id", "type": "varchar", "primary_key": True}]}

    with pytest.raises(ValidationError):
        ArchData(entities=[entity])


# ── Both coding agents receive the contract ──────────────────────────────────

CODING_AGENTS = {
    "database": (DatabaseAgent, {"migration_code": "VALUE = 1\n", "functions": []},
                 "output/backend/app/models/todo.py"),
    "backend": (BackendAgent, {"code": "VALUE = 1\n", "classes": [], "functions": []},
                "output/backend/app/api/routes/todo.py"),
}


@pytest.mark.parametrize("agent", sorted(CODING_AGENTS))
def test_coding_request_carries_the_entity_contract(agent, workspace):
    agent_cls, reply, path = CODING_AGENTS[agent]
    state = ProjectState(project_id=f"entities-{agent}", project_name="Todo")
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI (Python 3.12)", "database": "SQLite 3"},
        entities=[TODO_ENTITY],
    )
    state.tasks.tasks.append(Task(id=f"{agent[:2]}-001", agent=agent, title=path, description=path))
    state.code_manifest.files[path] = FileManifest(
        agent=agent, task_id=f"{agent[:2]}-001", filepath=path, purpose="Todo", exports=["TodoA", "TodoB", "TodoC"],
    )
    fake = ScriptedLLMProvider(agent, files={path: reply})

    agent_cls(provider=fake, retries=1, workspace=workspace).run(state)

    request = fake.requests[0].user_message
    assert render_entity_contract(state.architecture.entities) in request
    assert "- `id`: uuid (primary key, not null) → SQLAlchemy `Uuid`, Pydantic `uuid.UUID`, route param `uuid.UUID`" in request
    assert "- `title`: str (not null) → SQLAlchemy `String(200)`, Pydantic `str`" in request
