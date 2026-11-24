"""The database agent's system prompt is rendered from the Persistence Contract."""

import pytest

from core.state import ArchData, ProjectState, build_persistence_contract
from tests.fixtures.scripted_llm import llm_files, scripted_coding_agent, scripted_planner

#: Phrases of the old migration-file and raw-SQL rules; none may reach the prompt.
OLD_RULE_PHRASES = (
    "migration file",
    "drop table",
    "not null",
    "index on every foreign key",
    "rollback",
)


def _database_system_prompts(workspace, database_tech: str) -> list[str]:
    from agents.data_base import DatabaseAgent
    from agents.planner import PlannerAgent

    state = ProjectState(project_id="db-prompt", project_name="DbPrompt")
    state.requirements.raw_prompt = "Build a todo app."
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI (Python 3.12)", "database": database_tech},
        persistence_contract=build_persistence_contract(database_tech),
    )
    PlannerAgent(provider=scripted_planner()).run(state)
    llm = scripted_coding_agent("database", llm_files(state.code_manifest, "database"),
                                {"migration_code": "MODEL = 1\n", "functions": []})
    DatabaseAgent(provider=llm, retries=1, workspace=workspace).run(state)
    assert llm.requests, "the database agent made no LLM call"
    return [request.system_prompt for request in llm.requests]


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
