"""Seam 1 — the persistence contract is a pure function of the database tech string."""

import os
import subprocess
import sys

import pytest

from core.state import UnknownDatabaseError, build_persistence_contract


def test_postgresql_stack_gets_alembic_migrations():
    contract = build_persistence_contract("PostgreSQL 16")

    assert contract.orm == "SQLAlchemy"
    assert contract.migrations_tool == "Alembic"
    assert contract.db_url_env_var == "DATABASE_URL"
    assert contract.connection_module == "db/session.py"
    assert contract.init_command == "alembic upgrade head"


def test_sqlite_stack_uses_create_all_and_no_migrations_tool():
    contract = build_persistence_contract("SQLite 3")

    assert contract.orm == "SQLAlchemy"
    assert contract.migrations_tool is None
    assert contract.db_url_env_var == "DATABASE_URL"
    assert contract.connection_module == "db/session.py"
    # No migrations tool: running the connection module creates the tables.
    assert contract.init_command == "python -m db.session"


@pytest.mark.parametrize("database_tech", ["Apache Cassandra", "AWS DynamoDB"])
def test_unknown_database_raises_instead_of_falling_back_to_sqlite(database_tech):
    with pytest.raises(UnknownDatabaseError, match=database_tech):
        build_persistence_contract(database_tech)


@pytest.mark.parametrize(
    ("database_tech", "orm", "migrations_tool", "init_command"),
    [
        ("MySQL 8", "SQLAlchemy", "Alembic", "alembic upgrade head"),
        ("MongoDB 7", "Beanie", None, "python -m db.session"),
        # Strings the resolver / LLM actually produce for known databases.
        ("Supabase (PostgreSQL)", "SQLAlchemy", "Alembic", "alembic upgrade head"),
        ("PostgreSQL with SQLAlchemy (Alembic)", "SQLAlchemy", "Alembic", "alembic upgrade head"),
        ("sqlite", "SQLAlchemy", None, "python -m db.session"),
    ],
)
def test_known_database_mappings(database_tech, orm, migrations_tool, init_command):
    contract = build_persistence_contract(database_tech)

    assert (contract.orm, contract.migrations_tool, contract.init_command) == (
        orm,
        migrations_tool,
        init_command,
    )
    assert contract.db_url_env_var == "DATABASE_URL"
    assert contract.connection_module == "db/session.py"


def test_contract_has_exactly_the_five_fields():
    contract = build_persistence_contract("PostgreSQL 16")

    assert set(contract.model_dump()) == {
        "orm",
        "migrations_tool",
        "db_url_env_var",
        "connection_module",
        "init_command",
    }


def test_env_example_declares_the_canonical_db_url_variable():
    from core.state import build_env_example

    env = build_env_example("PostgreSQL 16")

    assignments = [line for line in env.splitlines() if line and not line.startswith("#")]
    assert assignments == ["DATABASE_URL=postgresql://app:change-me@localhost:5432/app"]


@pytest.mark.parametrize("database_tech", ["PostgreSQL", "SQLite", "MongoDB", "MySQL"])
def test_env_example_gives_every_known_database_a_non_empty_example_url(database_tech):
    from core.state import build_env_example

    url_lines = [l for l in build_env_example(database_tech).splitlines() if l.startswith("DATABASE_URL=")]

    assert len(url_lines) == 1
    assert url_lines[0] != "DATABASE_URL="


@pytest.mark.parametrize("database_tech", ["", "Apache Cassandra"])
def test_env_example_without_a_known_database_has_no_url_variable(database_tech):
    from core.state import build_env_example

    assert "DATABASE_URL" not in build_env_example(database_tech)


# ── Connection module synthesizer (ticket #14) ────────────────────────────────

def _exec_module(source: str, monkeypatch, database_url: str) -> dict:
    monkeypatch.setenv("DATABASE_URL", database_url)
    namespace: dict = {"__name__": "db.session"}
    exec(compile(source, "db/session.py", "exec"), namespace)
    return namespace


def test_connection_module_session_dependency_yields_a_working_session(monkeypatch):
    from sqlalchemy import text

    from core.state import build_connection_module

    source = build_connection_module(build_persistence_contract("SQLite 3"), model_modules=[])
    module = _exec_module(source, monkeypatch, "sqlite://")

    sessions = module["get_db"]()
    db = next(sessions)
    assert db.execute(text("SELECT 1")).scalar() == 1
    sessions.close()  # dependency teardown runs the finally block
    assert {"engine", "SessionLocal", "get_db"} <= module.keys()


@pytest.mark.parametrize("database_tech", ["PostgreSQL 16", "MySQL 8", "SQLite 3"])
def test_connection_module_hardcodes_no_host_database_or_credentials(database_tech):
    from core.state import build_connection_module

    source = build_connection_module(build_persistence_contract(database_tech), model_modules=["models.todo"])

    assert 'os.environ["DATABASE_URL"]' in source
    for leaked in ("://", "localhost", "127.0.0.1", "@", "password", "change-me"):
        assert leaked not in source


#: Minimal SQLAlchemy model module, shared with the end-to-end suite.
TODO_MODEL_SOURCE = '''
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class TodoItem(Base):
    __tablename__ = "todo_items"
    id = Column(Integer, primary_key=True)
    title = Column(String)
'''


def _run_init_command(tmp_path, database_tech: str) -> list[str]:
    """Lay out a backend root, run `python -m db.session`, return the created tables."""
    from sqlalchemy import create_engine, inspect

    from core.state import build_connection_module

    (tmp_path / "db").mkdir()
    (tmp_path / "db" / "__init__.py").write_text("")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "__init__.py").write_text("")
    (tmp_path / "models" / "todo.py").write_text(TODO_MODEL_SOURCE)
    (tmp_path / "db" / "session.py").write_text(
        build_connection_module(build_persistence_contract(database_tech), model_modules=["models.todo"]),
        encoding="utf-8",
    )
    url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    env = {**os.environ, "DATABASE_URL": url}

    subprocess.run([sys.executable, "-m", "db.session"], cwd=tmp_path, env=env, check=True)

    return inspect(create_engine(url)).get_table_names()


def test_create_all_stack_init_command_creates_the_model_tables(tmp_path):
    contract = build_persistence_contract("SQLite 3")
    assert contract.init_command == "python -m db.session"

    assert _run_init_command(tmp_path, "SQLite 3") == ["todo_items"]


def test_migrations_stack_leaves_schema_creation_to_the_migrations_tool(tmp_path):
    # Same SQLAlchemy module shape; only the schema-creation path differs.
    assert _run_init_command(tmp_path, "PostgreSQL 16") == []


def test_mongodb_connection_module_uses_motor_not_sqlalchemy():
    import ast

    from core.state import build_connection_module

    source = build_connection_module(build_persistence_contract("MongoDB 7"), model_modules=[])
    tree = ast.parse(source)

    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    defined = {t.id for node in tree.body if isinstance(node, ast.Assign) for t in node.targets
               if isinstance(t, ast.Name)}
    defined |= {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "motor.motor_asyncio" in imported
    assert not any(m.startswith("sqlalchemy") for m in imported if m)
    assert {"engine", "SessionLocal", "get_db"} <= defined
    assert 'os.environ["DATABASE_URL"]' in source


# ── Contract rendering + persistence mandate (ticket #16) ─────────────────────

def test_rendered_contract_carries_every_value_downstream_agents_must_agree_on():
    from core.state import render_persistence_contract

    rendered = render_persistence_contract(build_persistence_contract("PostgreSQL 16"))

    for value in ("SQLAlchemy", "Alembic", "DATABASE_URL", "db/session.py",
                  "from db.session import engine, SessionLocal, get_db", "alembic upgrade head"):
        assert value in rendered


def test_rendered_contract_without_database_names_no_url_variable():
    from core.state import render_persistence_contract

    assert "DATABASE_URL" not in render_persistence_contract(None)


def _planner_prompt(tier: str, database_tech: str | None) -> str:
    from agents.planner import PlannerAgent
    from core.state import ArchData, ProjectState
    from tests.planner import MockSkeletonLLM

    state = ProjectState(project_id="tier", project_name="Tier")
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI (Python 3.12)", **({"database": database_tech} if database_tech else {})},
        persistence_contract=build_persistence_contract(database_tech) if database_tech else None,
        complexity_tier=tier,
    )
    mock = MockSkeletonLLM()
    PlannerAgent(provider=mock).run(state)
    return mock.last_user_message


@pytest.mark.parametrize("tier", ["MINIMAL", "STANDARD", "ENTERPRISE"])
def test_every_complexity_tier_carries_the_persistence_mandate_when_there_is_a_database(tier):
    from agents.architecture import _complexity_block
    from core.state import persistence_mandate

    mandate = persistence_mandate(tier)

    assert mandate in _planner_prompt(tier, "SQLite 3")
    assert mandate in _complexity_block(tier, has_database=True)


def test_stack_without_database_gets_no_persistence_mandate():
    from agents.architecture import _complexity_block
    from core.state import persistence_mandate

    mandate = persistence_mandate("STANDARD")

    assert mandate not in _planner_prompt("STANDARD", None)
    assert mandate not in _complexity_block("STANDARD", has_database=False)


def test_minimal_tier_still_requires_a_real_database():
    from core.state import persistence_mandate

    assert "small real database" in persistence_mandate("MINIMAL")
