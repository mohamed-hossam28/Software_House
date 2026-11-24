"""The database agent's system prompt is rendered from the Persistence Contract."""

import json

import pytest

from core.state import ArchData, ProjectState, build_persistence_contract
from providers.llm.base import BaseLLMProvider, LLMResponse

#: Phrases of the old migration-file and raw-SQL rules; none may reach the prompt.
OLD_RULE_PHRASES = (
    "migration file",
    "drop table",
    "not null",
    "index on every foreign key",
    "rollback",
)


class RecordingModelProvider(BaseLLMProvider):
    """Records every system prompt and answers with a stub model file."""

    def __init__(self):
        self.system_prompts: list[str] = []

    @property
    def provider_name(self) -> str:
        return "recording-model"

    def complete(self, system_prompt, user_message, model="", max_tokens=8192,
                 temperature=0.2, stream=False, chunk_callback=None) -> LLMResponse:
        self.system_prompts.append(system_prompt)
        stub = {"migration_code": "MODEL = 1\n", "functions": []}
        if "Return a JSON ARRAY" in user_message:
            paths = [line.split("### File:", 1)[1].strip()
                     for line in user_message.splitlines() if line.startswith("### File:")]
            text = json.dumps([{**stub, "filepath": p} for p in paths])
        else:
            text = json.dumps(stub)
        return LLMResponse(text=text, input_tokens=10, output_tokens=10)


def _database_system_prompts(workspace, database_tech: str) -> list[str]:
    from agents.data_base import DatabaseAgent
    from agents.planner import PlannerAgent
    from tests.planner import MockSkeletonLLM

    state = ProjectState(project_id="db-prompt", project_name="DbPrompt")
    state.requirements.raw_prompt = "Build a todo app."
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI (Python 3.12)", "database": database_tech},
        persistence_contract=build_persistence_contract(database_tech),
    )
    PlannerAgent(provider=MockSkeletonLLM()).run(state)
    llm = RecordingModelProvider()
    DatabaseAgent(provider=llm, retries=1, workspace=workspace).run(state)
    assert llm.system_prompts, "the database agent made no LLM call"
    return llm.system_prompts


@pytest.mark.parametrize(
    ("database_tech", "orm"),
    [
        ("PostgreSQL 16", "SQLAlchemy"),
        ("MySQL 8", "SQLAlchemy"),
        ("SQLite 3", "SQLAlchemy"),
        ("MongoDB 7", "Beanie"),
    ],
)
def test_system_prompt_names_the_contracts_orm_and_connection_import(workspace, database_tech, orm):
    for prompt in _database_system_prompts(workspace, database_tech):
        assert orm in prompt
        assert "from db.session import engine, SessionLocal, get_db" in prompt
        assert "app/models/" in prompt


@pytest.mark.parametrize("database_tech", ["PostgreSQL 16", "MongoDB 7"])
def test_system_prompt_has_no_migration_file_or_raw_sql_rules(workspace, database_tech):
    for prompt in _database_system_prompts(workspace, database_tech):
        lowered = prompt.lower()
        for phrase in OLD_RULE_PHRASES:
            assert phrase not in lowered, phrase


def test_mongodb_system_prompt_does_not_ask_for_sqlalchemy(workspace):
    for prompt in _database_system_prompts(workspace, "MongoDB 7"):
        assert "SQLAlchemy" not in prompt
