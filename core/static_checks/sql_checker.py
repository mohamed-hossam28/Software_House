"""
SQL static checks: dialect-aware parsing and schema cross-check.

Dialect — the architecture tech stack's database entry (passed through the
uniform `dialect` parameter by the Integration Agent as the raw tech string)
is mapped to a sqlglot dialect: PostgreSQL → postgres, MySQL/MariaDB →
mysql, SQLite → sqlite, SQL Server → tsql. Non-SQL or unknown database
technologies (e.g. MongoDB) skip the SQL checks entirely: parsing them as
SQL could only produce noise. The raw tech string also resolves the ORM
map's migration directory (core.state.resolve_db_tooling).

SQL sources — .sql files are parsed statement by statement (whole content,
line-tracked); the ORM map's migration directory marks migration files the
same way. Every other Python file — migration files included — contributes
its SQL strings: string literals that start with a SQL statement keyword and
the first string argument of `text(...)` calls (extracted with ast, so
findings carry the Python source line). .sql files outside the migration
directory are still parsed: a parse error should never slip through.

Schema — built from DDL before any query is checked: CREATE TABLE in parsed
SQL, `op.create_table(...)` / `op.add_column(...)` and
`sa.Table(..., sa.Column(...))` calls in Python, and SQLAlchemy model
classes declaring `__tablename__` with `Column(...)`/`mapped_column(...)`
attributes. The model classes are beyond the literal migration-file scope
but are what SQLite projects generate under `create_all()` (no DDL files);
without them SQLite schemas would be unknowable. When no DDL is found
anywhere, the manifest's deterministic database field contracts
(FileManifest.contract on database-agent files) are a fallback schema.
DDL-derived table names match queries exactly; contract-derived names are
approximate entities, so they tolerate singular/plural variants.

Findings — a statement that fails to parse is an ERROR (sql_parse); a query
referencing a table absent from the schema is an ERROR (sql_tables); a query
referencing a column absent from the table it is matched to is an ERROR
(sql_columns, repairable on the query file against the schema contract).
With no schema knowledge at all the table/column cross-check
is skipped (parsing still runs), because flagging every query would be noise.
CTE aliases are treated as known tables within their statement. An
un-prefixed column in a multi-table statement is matched against the union
of that statement's tables — flagging it only when it exists in none.

The architecture parameter is part of the uniform checker signature; the
dialect used by the checks comes from it, so it is read but not passed on.
"""
import ast
import re
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from core.state import FindingSeverity, StaticFinding, resolve_db_tooling

#: sqlglot dialect name for a database tech entry; None = not SQL-checkable.
_DIALECT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("postgres", "postgres"),
    ("postgresql", "postgres"),
    ("mysql", "mysql"),
    ("mariadb", "mysql"),
    ("sqlite", "sqlite"),
    ("mssql", "tsql"),
    ("sqlserver", "tsql"),
)

#: Python string literals starting with one of these are treated as SQL.
_SQL_STATEMENT_RE = re.compile(
    r"^\s*(select|insert|update|delete|with|create\s+(table|index)|alter\s+table|drop\s+(table|index))\b",
    re.IGNORECASE,
)

#: DDL calls that declare a table followed by Column(...) arguments.
_TABLE_CALLS = {"create_table", "Table"}

#: Model columns attribute-declared as `name = Column(String)` get their name here.
_COLUMN_CALL_ATTRS = {"Column", "mapped_column"}

_PARSE = "sql_parse"
_TABLES = "sql_tables"
_COLUMNS = "sql_columns"


def check_sql(
    files: dict[str, str],
    manifest: Any,
    architecture: Any = None,
    dialect: str = "",
) -> list[StaticFinding]:
    """Parse SQL in generated files and cross-check it against the extracted schema.

    manifest is part of the uniform checker signature and is used only as the
    schema fallback (database field contracts). `dialect` carries the raw
    tech-stack database entry (e.g. "PostgreSQL"): it selects the sqlglot
    dialect for parsing and the ORM map's migration directory; architecture
    is read only when `dialect` is empty.
    """
    database_tech = _database_tech(architecture, dialect)
    sql_dialect = resolve_sql_dialect(None, database_tech)
    if not sql_dialect:
        return []

    findings: list[StaticFinding] = []
    statements = _sql_statements(files, database_tech)
    schema, contract_tables = _build_schema(files, manifest, sql_dialect)

    parsed: list[tuple[str, int, Any]] = []
    for filepath, line, code in statements:
        try:
            parsed.append((filepath, line, sqlglot.parse_one(code, read=sql_dialect)))
        except ParseError as exc:
            findings.append(StaticFinding(
                file=filepath,
                check=_PARSE,
                severity=FindingSeverity.ERROR,
                line=line,
                message=f"SyntaxError in SQL: {str(exc).splitlines()[0]}",
            ))

    if schema:                                   # no schema knowledge → no cross-check
        for filepath, line, statement in parsed:
            findings.extend(_cross_check(filepath, line, statement, schema, contract_tables))

    return _dedupe(findings)


def resolve_sql_dialect(architecture: Any, dialect: str = "") -> str | None:
    """sqlglot dialect name for the tech stack's database entry; None when not SQL.

    The tech string may carry driver suffixes ("postgresql+asyncpg") or
    versions ("MySQL 8"), so matching is substring-based on the known
    families. Unknown or non-SQL entries (MongoDB, blank) return None.
    """
    tech = _database_tech(architecture, dialect)
    for pattern, name in _DIALECT_PATTERNS:
        if pattern in tech:
            return name
    return None


def _database_tech(architecture: Any, dialect: str = "") -> str:
    """Raw tech-stack database string: the `dialect` param, else architecture."""
    tech = str(dialect or "").lower()
    if not tech and architecture is not None:
        tech = str((getattr(architecture, "tech_stack", None) or {}).get("database", "") or "").lower()
    return tech


# ── SQL statement sources ────────────────────────────────────────────────────


def _sql_statements(files: dict[str, str], database_tech: str) -> list[tuple[str, int, str]]:
    """(file, line, SQL text) for every SQL source in *files*.

    .sql files and files under the ORM map's migration directory are SQL
    sources in full (split into statements, line-tracked); every other Python
    file contributes its SQL-looking string literals and text(...) arguments.
    """
    migration_dir = _migration_dir(database_tech)
    return_statements: list[tuple[str, int, str]] = []
    for filepath, code in sorted(files.items()):
        if filepath.endswith(".py") and code:
            return_statements.extend(_python_sql_strings(filepath, code))
            continue
        if _is_sql_source(filepath, migration_dir) and code:
            return_statements.extend(_split_sql_statements(code, filepath))
    return return_statements


def _migration_dir(database_tech: str) -> str:
    """Migration directory fragment from the ORM map, e.g. 'alembic/versions/'."""
    return str(resolve_db_tooling(database_tech or "").get("migration_dir", "") or "")


def _is_sql_source(filepath: str, migration_dir: str) -> bool:
    """.sql file, or a file living in the ORM map's migration directory."""
    if filepath.endswith(".sql"):
        return True
    return bool(migration_dir) and migration_dir in filepath


def _split_sql_statements(code: str, filepath: str) -> list[tuple[str, int, str]]:
    """(file, line, statement) pieces of a SQL script, quote/comment aware.

    Splitting before parsing isolates one broken statement so it cannot mask
    the others, and keeps the source line of each statement for findings.
    Comments are dropped (sqlglot needs none of them) so they cannot start a
    statement's buffer; a statement's line is where its first real character
    sits — two statements on one line both get that line.
    """
    statements: list[tuple[str, int, str]] = []
    buf: list[str] = []
    buf_start = 1
    line = 1
    i = 0
    n = len(code)
    while i < n:
        char = code[i]
        if char == "\n":
            line += 1
            i += 1
            continue
        if char in " \t\r":
            if buf:                              # interior whitespace only
                buf.append(char)
            i += 1
            continue
        if char in ("'", '"'):                   # string literal — scan to its close
            if not buf:
                buf_start = line
            buf.append(char)
            i += 1
            while i < n:
                if code[i] == char:
                    doubled = i + 1 < n and code[i + 1] == char
                    buf.append(char)
                    i += 1
                    if not doubled:
                        break
                    continue
                if code[i] == "\n":
                    line += 1
                buf.append(code[i])
                i += 1
            continue
        if code.startswith("--", i):             # line comment — belongs to no statement
            end = code.find("\n", i)
            i = n if end < 0 else end
            continue
        if code.startswith("/*", i):             # block comment — belongs to no statement
            end = code.find("*/", i + 2)
            end = n if end < 0 else end + 2
            line += code.count("\n", i, end)
            i = end
            continue
        if char == ";":
            text = "".join(buf).strip()
            if text:
                statements.append((filepath, buf_start, text))
            buf = []
            i += 1
            continue
        if not buf:
            buf_start = line
        buf.append(char)
        i += 1
    text = "".join(buf).strip()
    if text:
        statements.append((filepath, buf_start, text))
    return statements


def _python_sql_strings(filepath: str, code: str) -> list[tuple[str, int, str]]:
    """SQL strings in a Python file: statement-keyword literals + text(...) args."""
    strings: list[tuple[str, int, str]] = []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []                                # python_checker already reports this
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _SQL_STATEMENT_RE.match(node.value):
                strings.append((filepath, node.lineno, node.value))
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name == "text" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    strings.append((filepath, first.lineno, first.value))
    return strings


# ── Schema extraction ────────────────────────────────────────────────────────


def extract_sql_schema(files: dict[str, str], manifest: Any = None, dialect: str = "") -> dict[str, set[str]]:
    """Known tables → column names, from migration DDL (and manifest contracts).

    Every .sql file is parsed for CREATE TABLE statements; every Python file
    is AST-scanned for op.create_table / op.add_column / sa.Table calls and
    for model classes declaring __tablename__ with Column attributes. The
    manifest's database field contracts fill in tables no DDL defines.
    `dialect` is the database tech string or sqlglot dialect name.
    """
    sql_dialect = resolve_sql_dialect(None, dialect) or "postgres"
    schema, _ = _build_schema(files, manifest, sql_dialect)
    return schema


def _build_schema(
    files: dict[str, str],
    manifest: Any,
    sql_dialect: str,
) -> tuple[dict[str, set[str]], set[str]]:
    """(table → columns, contract-derived tables) from DDL and manifest contracts.

    Tables coming from DDL match queries exactly; contract-derived names are
    approximate entities and additionally tolerate singular/plural variants.
    """
    schema: dict[str, set[str]] = {}
    contract_tables: set[str] = set()

    for filepath, code in sorted(files.items()):
        if not code:
            continue
        if filepath.endswith(".sql"):
            for _, _, statement in _split_sql_statements(code, filepath):
                _absorb_ddl(statement, schema, sql_dialect)
        elif filepath.endswith(".py"):
            _absorb_python_ddl(code, schema)

    if not schema and manifest is not None:
        for name, columns in _schema_from_manifest(manifest).items():
            schema[name] = columns
            contract_tables.add(name)
    return schema, contract_tables


def _absorb_ddl(statement: str, schema: dict[str, set[str]], sql_dialect: str) -> None:
    """Merge one parsed SQL statement's CREATE TABLE into *schema*."""
    try:
        parsed = sqlglot.parse_one(statement, read=sql_dialect)
    except Exception:
        # Any sqlglot failure on a DDL line is fine to ignore here: broken
        # statements are already reported by check_sql's parse pass.
        return
    create = parsed.find(exp.Create)
    if create is None or not isinstance(create.this, exp.Schema):
        return
    table = create.this.this
    if not isinstance(table, exp.Table):
        return
    columns = schema.setdefault(table.name, set())
    for column_def in create.this.expressions:
        if isinstance(column_def, exp.ColumnDef) and column_def.name:
            columns.add(column_def.name.lower())


def _absorb_python_ddl(code: str, schema: dict[str, set[str]]) -> None:
    """Merge op.create_table / op.add_column / sa.Table / model classes into *schema*."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return                                   # python_checker already reports this
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "attr", None) or getattr(func, "id", None)
        args = [a for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        if name in _TABLE_CALLS and node.args and args:
            schema.setdefault(args[0].value, set()).update(_column_names(node))
        elif name == "add_column" and len(node.args) >= 1 and args:
            table = args[0].value
            columns = schema.setdefault(table, set())
            for column in node.args[1:]:
                if isinstance(column, ast.Call) and _column_call_name(column):
                    columns.add(_column_call_name(column))

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            table_name = _string_assign(node, "__tablename__")
            if table_name:
                columns = schema.setdefault(table_name, set())
                for stmt in ast.walk(node):
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                        func = stmt.value.func
                        attr = getattr(func, "attr", None) or getattr(func, "id", None)
                        if attr in _COLUMN_CALL_ATTRS:
                            column = _column_call_name(stmt.value)
                            if column:
                                columns.add(column)
                            else:
                                for target in stmt.targets:
                                    if isinstance(target, ast.Name):
                                        columns.add(target.id.lower())


def _column_names(call: ast.Call) -> set[str]:
    """String column names of sa.Column(...) arguments inside a DDL call."""
    columns: set[str] = set()
    for arg in call.args[1:]:                    # arg[0] is the table name
        if isinstance(arg, ast.Call):
            column = _column_call_name(arg)
            if column:
                columns.add(column)
    return columns


def _column_call_name(call: ast.Call) -> str | None:
    """Column name of a Column(...) call: its string first argument, else None."""
    if call.args:
        first = call.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value.lower()
    return None


def _string_assign(node: ast.ClassDef, attribute: str) -> str | None:
    """String value of `attribute = "..."` inside a class body."""
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == attribute:
                    value = stmt.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        return value.value
    return None


def _schema_from_manifest(manifest: Any) -> dict[str, set[str]]:
    """Manifest database field contracts as table → columns (canonical plural).

    Fallback when no DDL was found: FileManifest.contract maps an export
    (entity name, e.g. 'User') to its fields; the table is registered under
    its canonical plural form so 'users' queries match.
    """
    schema: dict[str, set[str]] = {}
    for fm in (getattr(manifest, "files", None) or {}).values():
        if getattr(fm, "agent", "") != "database":
            continue
        for export, fields in (getattr(fm, "contract", None) or {}).items():
            columns = {str(name).lower() for name in fields}
            for name in _entity_names(export):
                schema.setdefault(name, set()).update(columns)
    return schema


def _entity_names(export: str) -> tuple[str, ...]:
    """Canonical table name of an entity export: lowercase, pluralized once.

    'User' → 'users'; an already-plural export stays as written. Query-time
    matching also tolerates singular variants (see _table_columns).
    """
    base = str(export or "").strip().lower()
    if not base:
        return ()
    return (base if base.endswith("s") else f"{base}s",)


# ── Parsing and cross-check ──────────────────────────────────────────────────


def _cross_check(
    filepath: str,
    line: int,
    statement: Any,
    schema: dict[str, set[str]],
    contract_tables: set[str],
) -> list[StaticFinding]:
    """Table and column references of *statement* against *schema*.

    contract_tables are the schema entries whose names came from the
    manifest's entity contracts; only those tolerate singular/plural
    variants, so DDL-derived tables match queries exactly.
    """
    findings: list[StaticFinding] = []
    ctes = {cte.alias for cte in statement.find_all(exp.CTE) if cte.alias}
    refs = _table_refs(statement)
    tables = {name for _, name in refs}
    aliases = {alias: name for alias, name in refs if alias}

    for table in sorted(tables):
        if table not in ctes and not _table_exists(table, schema, contract_tables):
            findings.append(StaticFinding(
                file=filepath,
                check=_TABLES,
                severity=FindingSeverity.ERROR,
                line=line,
                message=f"SQL references nonexistent table '{table}' (no migration or contract defines it)",
            ))

    known = {name for name in tables if name in ctes or _table_exists(name, schema, contract_tables)}
    single_table = known - set(ctes)
    findings.extend(_insert_columns(filepath, line, statement, schema, contract_tables))
    for column in statement.find_all(exp.Column):
        name = (column.name or "").lower()
        prefix = column.table or ""
        if not name:
            continue
        if prefix:                               # alias- or table-qualified
            table = aliases.get(prefix, prefix)
            columns = _table_columns(table, schema, contract_tables)
            if columns is not None and name not in columns:
                findings.append(_missing_column(filepath, line, name, table))
        else:
            if len(single_table) == 1:
                table = next(iter(single_table))
                if name not in (_table_columns(table, schema, contract_tables) or set()):
                    findings.append(_missing_column(filepath, line, name, table))
            elif single_table and name not in _union_columns(single_table, schema, contract_tables):
                findings.append(_missing_column(filepath, line, name, ", ".join(sorted(single_table))))
    return findings


def _insert_columns(
    filepath: str,
    line: int,
    statement: Any,
    schema: dict[str, set[str]],
    contract_tables: set[str],
) -> list[StaticFinding]:
    """Column list of an INSERT statement, which is not exp.Column nodes.

    'INSERT INTO t (a, b)' stores its list as Identifiers on exp.Schema.
    """
    if not (isinstance(statement, exp.Insert) and isinstance(statement.this, exp.Schema)):
        return []
    target = statement.this.this
    table = target.name if isinstance(target, exp.Table) else ""
    columns = _table_columns(table, schema, contract_tables)
    if columns is None:
        return []                                # unknown table reported by sql_tables
    findings = []
    for identifier in statement.this.expressions:
        name = (identifier.name or "").lower()
        if name and name not in columns:
            findings.append(_missing_column(filepath, line, name, table))
    return findings


def _table_refs(statement: Any) -> list[tuple[str, str]]:
    """(alias_or_name, table name) of real table references in *statement*.

    Table nodes nested inside qualified column references (t.body) are not
    table references and are excluded.
    """
    refs: list[tuple[str, str]] = []
    for node in statement.find_all(exp.Table):
        if isinstance(node.parent, exp.Column):
            continue
        name = node.name
        if name:
            refs.append((node.alias_or_name, name))
    return refs


def _missing_column(filepath: str, line: int, name: str, table: str) -> StaticFinding:
    return StaticFinding(
        file=filepath,
        check=_COLUMNS,
        severity=FindingSeverity.ERROR,
        line=line,
        message=f"SQL references column '{name}' which no migration defines on '{table}'",
    )


def _table_exists(
    table: str,
    schema: dict[str, set[str]],
    contract_tables: set[str],
) -> bool:
    """Table known exactly, or — for contract-derived names — by name variant."""
    if table in schema:
        return True
    return any(variant in contract_tables for variant in _name_variants(table))


def _table_columns(
    table: str,
    schema: dict[str, set[str]],
    contract_tables: set[str],
) -> set[str] | None:
    """Columns of a known table, None when unknown.

    Contract-derived names tolerate singular/plural variants; DDL-derived
    names match exactly, so a real typo like 'user' against DDL 'users'
    still errors.
    """
    if table in schema:
        return schema[table]
    for variant in _name_variants(table):
        if variant in contract_tables:
            return schema[variant]
    return None


def _name_variants(table: str) -> tuple[str, ...]:
    """Singular/plural/lowercase variants of a table name for contract matching."""
    lowered = table.lower()
    variants = {lowered}
    if lowered.endswith("s"):
        variants.add(lowered[:-1])
    else:
        variants.add(f"{lowered}s")
    return tuple(sorted(variants))


def _union_columns(
    tables: set[str],
    schema: dict[str, set[str]],
    contract_tables: set[str],
) -> set[str]:
    """Every column of every known table among *tables*."""
    columns: set[str] = set()
    for table in tables:
        known = _table_columns(table, schema, contract_tables)
        if known:
            columns.update(known)
    return columns


def _dedupe(findings: list[StaticFinding]) -> list[StaticFinding]:
    """Drop repeated findings (same file, check, line, message)."""
    seen: set[tuple[str, str, int, str]] = set()
    unique: list[StaticFinding] = []
    for finding in findings:
        key = (finding.file, finding.check, finding.line, finding.message)
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return unique


__all__ = ["check_sql", "extract_sql_schema", "resolve_sql_dialect"]
