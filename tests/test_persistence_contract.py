"""Seam 1 — the persistence contract is a pure function of the database tech string."""

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
