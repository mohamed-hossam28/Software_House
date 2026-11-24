"""Seam 1 — the persistence contract is a pure function of the database tech string."""

import os
import subprocess
import sys

import pytest

from core.state import DB_ORM_MAP, PersistenceContract, UnknownDatabaseError, build_persistence_contract


@pytest.mark.parametrize(
    ("database_tech", "orm"),
    [
        ("PostgreSQL 16", "SQLAlchemy"),
        ("MySQL 8", "SQLAlchemy"),
        ("SQLite 3", "SQLAlchemy"),
        ("MongoDB 7", "Beanie"),  # unchanged by dropping Alembic
        # Strings the resolver / LLM actually produce for known databases.
        ("Supabase (PostgreSQL)", "SQLAlchemy"),
        ("PostgreSQL with SQLAlchemy (Alembic)", "SQLAlchemy"),
        ("sqlite", "SQLAlchemy"),
    ],
)
def test_every_supported_database_creates_its_schema_by_running_the_connection_module(database_tech, orm):
    contract = build_persistence_contract(database_tech)

    assert contract == PersistenceContract(
        orm=orm,
        migrations_tool=None,
        db_url_env_var="DATABASE_URL",
        connection_module="db/session.py",
        init_command="python -m db.session",
    )


@pytest.mark.parametrize("database_tech", list(DB_ORM_MAP))
def test_no_persistence_contract_names_alembic(database_tech):
    assert "alembic" not in build_persistence_contract(database_tech).model_dump_json().lower()


@pytest.mark.parametrize("database_tech", ["Apache Cassandra", "AWS DynamoDB"])
def test_unknown_database_raises_instead_of_falling_back_to_sqlite(database_tech):
    with pytest.raises(UnknownDatabaseError, match=database_tech):
        build_persistence_contract(database_tech)


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
    """Lay out a backend root, run the contract's init command, return the created tables.

    No database server runs in the test suite, so DATABASE_URL points at a
    SQLite file for every SQL database: the connection module generated from
    the PostgreSQL / MySQL contract is the same SQLAlchemy module, and the init
    command is what is under test.
    """
    import shlex

    from sqlalchemy import create_engine, inspect

    from core.state import build_connection_module

    contract = build_persistence_contract(database_tech)
    module_path = tmp_path / contract.connection_module
    module_path.parent.mkdir(parents=True)
    (module_path.parent / "__init__.py").write_text("")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "__init__.py").write_text("")
    (tmp_path / "models" / "todo.py").write_text(TODO_MODEL_SOURCE)
    module_path.write_text(
        build_connection_module(contract, model_modules=["models.todo"]),
        encoding="utf-8",
    )
    url = f"sqlite:///{(tmp_path / 'app.db').as_posix()}"
    env = {**os.environ, "DATABASE_URL": url}

    program, *args = shlex.split(contract.init_command)
    assert program == "python"
    subprocess.run([sys.executable, *args], cwd=tmp_path, env=env, check=True)

    return inspect(create_engine(url)).get_table_names()


@pytest.mark.parametrize("database_tech", ["SQLite 3", "PostgreSQL 16", "MySQL 8"])
def test_sql_database_init_command_creates_the_model_tables(tmp_path, database_tech):
    assert _run_init_command(tmp_path, database_tech) == ["todo_items"]


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

    for value in ("SQLAlchemy", "DATABASE_URL", "db/session.py",
                  "from db.session import engine, SessionLocal, get_db", "python -m db.session"):
        assert value in rendered
    assert "alembic" not in rendered.lower()


def _planner_prompt(tier: str, database_tech: str) -> str:
    from agents.planner import PlannerAgent
    from core.state import ArchData, ProjectState
    from tests.planner import MockSkeletonLLM

    state = ProjectState(project_id="tier", project_name="Tier")
    state.architecture = ArchData(
        tech_stack={"backend": "FastAPI (Python 3.12)", "database": database_tech},
        persistence_contract=build_persistence_contract(database_tech),
        complexity_tier=tier,
    )
    mock = MockSkeletonLLM()
    PlannerAgent(provider=mock).run(state)
    return mock.last_user_message


@pytest.mark.parametrize("tier", ["MINIMAL", "STANDARD", "ENTERPRISE"])
def test_every_complexity_tier_carries_the_persistence_mandate(tier):
    from agents.architecture import _complexity_block
    from core.state import persistence_mandate

    mandate = persistence_mandate(tier)

    assert mandate in _planner_prompt(tier, "SQLite 3")
    assert mandate in _complexity_block(tier)


def test_minimal_tier_still_requires_a_real_database():
    from core.state import persistence_mandate

    assert "small real database" in persistence_mandate("MINIMAL")
