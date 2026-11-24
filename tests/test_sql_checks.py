import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from core.state import (
    ArchData,
    CodeManifest,
    FileManifest,
    FindingSeverity,
    ProjectState,
    build_persistence_contract,
)
from core.static_checks import run_all_checks
from core.static_checks.sql_checker import (
    check_sql,
    extract_sql_schema,
)


def make_architecture(database: str) -> ArchData:
    """Architecture whose Persistence Contract is built for *database*."""
    return ArchData(persistence_contract=build_persistence_contract(database))


def make_manifest(contracts: dict[str, dict[str, str]]) -> CodeManifest:
    fm = FileManifest(
        agent="database",
        task_id="db-001",
        filepath="output/backend/app/models/user.py",
        purpose="synthetic",
        exports=list(contracts),
        contract=contracts,
    )
    return CodeManifest(files={fm.filepath: fm})


MIGRATION_SQL = """\
CREATE TABLE todos (
    id UUID PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    is_completed BOOLEAN DEFAULT FALSE,
    user_id UUID
);
"""


class TestDialectFromContract:
    BACKTICK_QUERY = {"output/backend/app/api/todos.py": 'text("SELECT `title` FROM todos")\n'}

    def test_mysql_contract_parses_mysql_sql(self):
        assert check_sql(self.BACKTICK_QUERY, None, make_architecture("MySQL 8")) == []

    def test_postgresql_contract_rejects_mysql_only_sql(self):
        findings = check_sql(self.BACKTICK_QUERY, None, make_architecture("PostgreSQL 16"))
        assert [f.check for f in findings] == ["sql_parse"]

    def test_mongodb_contract_has_no_sql_dialect_so_sql_checks_are_skipped(self):
        files = {"output/backend/app/api/todos.py": 'text("SELEC broken FROM todos")\n'}
        assert check_sql(files, None, make_architecture("MongoDB 7")) == []
        assert extract_sql_schema({"output/backend/schema.sql": MIGRATION_SQL}, None,
                                  make_architecture("MongoDB 7")) == {}


class TestExtractSqlSchema:
    def test_create_table_sql_defines_schema(self):
        files = {"output/backend/alembic/versions/001_init.sql": MIGRATION_SQL}
        assert extract_sql_schema(files, None, make_architecture("PostgreSQL")) == {
            "todos": {"id", "title", "is_completed", "user_id"}
        }

    def test_python_create_table_migration(self):
        files = {"output/backend/alembic/versions/001_init.py": """\
import sqlalchemy as sa

def upgrade():
    op.create_table(
        "todos",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("title", sa.String(), nullable=False),
    )
"""}
        assert extract_sql_schema(files, None, make_architecture("PostgreSQL")) == {
            "todos": {"id", "title"}
        }

    def test_python_sa_table_declaration(self):
        files = {"output/backend/app/db/schema.py": """\
import sqlalchemy as sa

todos = sa.Table(
    "todos", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("title", sa.String),
)
"""}
        assert extract_sql_schema(files, None, make_architecture("SQLite")) == {
            "todos": {"id", "title"}
        }

    def test_add_column_extends_existing_table(self):
        files = {"output/backend/alembic/versions/002_more.py": """\
def upgrade():
    op.add_column("todos", sa.Column("priority", sa.Integer(), nullable=True))
"""}
        base = {"output/backend/alembic/versions/001_init.py": """\
def upgrade():
    op.create_table(
        "todos",
        sa.Column("id", sa.Integer(), primary_key=True),
    )
"""}
        schema = extract_sql_schema({**base, **files}, None, make_architecture("PostgreSQL"))
        assert schema == {"todos": {"id", "priority"}}

    def test_manifest_contract_fallback(self):
        manifest = make_manifest({"User": {"id": "UUID PK", "email": "VARCHAR"}})
        files = {"output/backend/app/models/user.py": "x = 1\n"}
        assert extract_sql_schema(files, manifest, make_architecture("PostgreSQL")) == {
            "users": {"id", "email"}
        }

    def test_no_schema_sources_empty(self):
        assert extract_sql_schema({}, None, make_architecture("PostgreSQL")) == {}


class TestSchemaMatchingStrictness:
    def test_ddl_names_match_exactly(self):
        # DDL-derived names are exact: 'user' against DDL 'users' is a real error
        files = {
            "output/backend/alembic/versions/001_init.sql": MIGRATION_SQL,
            "output/backend/app/api/todos.py": 'text("SELECT title FROM user")\n',
        }
        findings = check_sql(files, None, make_architecture("PostgreSQL"))
        assert any(f.check == "sql_tables" for f in findings)

    def test_contract_names_tolerate_singular(self):
        # contract-derived names are approximate entities: 'user' matches 'users'
        manifest = make_manifest({"User": {"id": "UUID PK", "email": "VARCHAR"}})
        files = {"output/backend/app/api/users.py": 'text("SELECT email FROM user")\n'}
        findings = check_sql(files, manifest, make_architecture("PostgreSQL"))
        assert [f.check for f in findings if f.check == "sql_tables"] == []


class TestCheckSql:
    def test_valid_sql_parses_clean(self):
        files = {
            "output/backend/alembic/versions/001_init.sql": MIGRATION_SQL,
            "output/backend/app/api/todos.py": (
                'from sqlalchemy import text\n'
                'def list_todos():\n'
                '    return text("SELECT title, is_completed FROM todos WHERE user_id = :u")\n'
            ),
        }
        assert check_sql(files, None, make_architecture("PostgreSQL")) == []

    def test_syntax_broken_sql_is_error(self):
        files = {"output/backend/alembic/versions/001_init.sql": "CREATE TABLE todos (id\n"}
        findings = check_sql(files, None, make_architecture("PostgreSQL"))
        assert len(findings) == 1
        f = findings[0]
        assert f.severity == FindingSeverity.ERROR
        assert f.check == "sql_parse"
        assert f.file.endswith("001_init.sql")
        assert f.line >= 1

    def test_broken_sql_string_in_backend_is_error(self):
        files = {
            "output/backend/app/api/todos.py": 'text("SELEC title FROM todos")\n',
            "output/backend/alembic/versions/001_init.sql": MIGRATION_SQL,
        }
        findings = check_sql(files, None, make_architecture("PostgreSQL"))
        assert len(findings) == 1
        assert findings[0].check == "sql_parse"
        assert findings[0].severity == FindingSeverity.ERROR
        assert findings[0].file.endswith("todos.py")

    def test_statement_lines_tracked_per_statement(self):
        # MIGRATION_SQL occupies lines 1-6; two statements on line 7, broken on line 8
        code = 'SELECT 1; SELECT 2;\nSELEC title FROM todos;\nSELECT 3;'
        files = {"output/backend/alembic/versions/001_init.sql": MIGRATION_SQL + code}
        findings = check_sql(files, None, make_architecture("PostgreSQL"))
        parses = [f for f in findings if f.check == "sql_parse"]
        assert len(parses) == 1
        assert parses[0].line == 8

    def test_query_on_nonexistent_table_is_error(self):
        files = {
            "output/backend/alembic/versions/001_init.sql": MIGRATION_SQL,
            "output/backend/app/api/ghosts.py": 'text("SELECT title FROM ghosts")\n',
        }
        findings = check_sql(files, None, make_architecture("PostgreSQL"))
        tables = [f for f in findings if f.check == "sql_tables"]
        assert len(tables) == 1
        f = tables[0]
        assert f.severity == FindingSeverity.ERROR
        assert "ghosts" in f.message

    def test_query_on_missing_column_errors(self):
        files = {
            "output/backend/alembic/versions/001_init.sql": MIGRATION_SQL,
            "output/backend/app/api/todos.py": 'text("SELECT body FROM todos")\n',
        }
        findings = check_sql(files, None, make_architecture("PostgreSQL"))
        cols = [f for f in findings if f.check == "sql_columns"]
        assert len(cols) == 1
        f = cols[0]
        assert f.severity == FindingSeverity.ERROR
        assert "body" in f.message

    def test_aliased_column_resolved_to_table(self):
        files = {
            "output/backend/alembic/versions/001_init.sql": MIGRATION_SQL,
            "output/backend/app/api/todos.py": 'text("SELECT t.body FROM todos t")\n',
        }
        findings = check_sql(files, None, make_architecture("PostgreSQL"))
        assert [f.check for f in findings] == ["sql_columns"]

    def test_insert_column_list_checked(self):
        files = {
            "output/backend/alembic/versions/001_init.sql": MIGRATION_SQL,
            "output/backend/app/api/todos.py": (
                'text("INSERT INTO todos (title, body) VALUES (:t, :b)")\n'
            ),
        }
        findings = check_sql(files, None, make_architecture("PostgreSQL"))
        cols = [f for f in findings if f.check == "sql_columns"]
        assert len(cols) == 1
        assert "body" in cols[0].message

    def test_join_columns_checked_per_table(self):
        files = {
            "output/backend/alembic/versions/001_init.sql": MIGRATION_SQL,
            "output/backend/alembic/versions/002_users.sql": (
                "CREATE TABLE users (id UUID PRIMARY KEY, email VARCHAR NOT NULL);\n"
            ),
            "output/backend/app/api/todos.py": (
                'text("SELECT u.email, t.body FROM todos t JOIN users u ON u.id = t.user_id")\n'
            ),
        }
        findings = check_sql(files, None, make_architecture("PostgreSQL"))
        assert [f.check for f in findings] == ["sql_columns"]
        assert "body" in findings[0].message

    def test_cte_tables_not_flagged(self):
        files = {
            "output/backend/alembic/versions/001_init.sql": MIGRATION_SQL,
            "output/backend/app/api/todos.py": (
                'text("WITH recent AS (SELECT title FROM todos) '
                'SELECT title FROM recent")\n'
            ),
        }
        assert check_sql(files, None, make_architecture("PostgreSQL")) == []

    def test_empty_schema_skips_cross_check_but_not_parse(self):
        # no migrations and no manifest contracts → parsing still reports, no table findings
        files = {"output/backend/app/api/todos.py": 'text("SELECT title FROM todos")\n'}
        findings = check_sql(files, None, make_architecture("PostgreSQL"))
        assert findings == []

    def test_non_sql_files_ignored(self):
        files = {
            "output/static/app.js": 'fetch("/api/v1/todos").then(r => r.json());\n',
        }
        assert check_sql(files, None, make_architecture("PostgreSQL")) == []


class TestRunAllChecksIntegration:
    def test_sql_findings_surface_through_pipeline(self):
        files = {
            "output/backend/alembic/versions/001_init.sql": MIGRATION_SQL,
            "output/backend/app/api/todos.py": (
                'text("SELECT body FROM todos")\n'
                'text("SELEC broken FROM todos")\n'
            ),
        }
        findings = run_all_checks(files, None, architecture=make_architecture("PostgreSQL"))
        checks = {(f.check, f.severity) for f in findings if f.check.startswith("sql_")}
        assert ("sql_parse", FindingSeverity.ERROR) in checks
        assert ("sql_columns", FindingSeverity.ERROR) in checks

    def test_clean_sql_project_is_clean(self):
        files = {
            "output/backend/alembic/versions/001_init.sql": MIGRATION_SQL,
            "output/backend/app/api/todos.py": 'text("SELECT title FROM todos")\n',
        }
        findings = run_all_checks(files, None, architecture=make_architecture("PostgreSQL"))
        assert findings == []
