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
