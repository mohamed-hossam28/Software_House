"""
Persistence checks — a cheap static stand-in for the future runtime/testing agent.

Run only when the stack has a database (architecture.persistence_contract set),
over the manifest-owned backend Python files (the synthesized connection module
itself is skipped). Files that do not parse are skipped: the Python checker
already reports their syntax errors.

  - in_memory_state: a module-level container (list/dict/set literal or
    comprehension, or list()/dict()/set()/defaultdict()/OrderedDict()/deque()
    call) that either starts EMPTY or is MUTATED anywhere in the file
    (.append/.update/…, `x[k] = v`, `del x[k]`, `x += …`) — i.e. data kept in
    process memory instead of the database. Name casing is irrelevant, so
    renaming a store to UPPER_CASE does not evade the check; unmutated
    non-empty containers (CORS `origins`, `tags_metadata`, `responses`) are
    configuration and pass. Dunder names (`__all__`) are exempt.
  - db_without_connection_module: the file creates its own engine, session
    factory, client or raw driver connection (create_engine, sessionmaker,
    sqlite3.connect, MongoClient, …), or — without importing the connection
    module — runs session operations (query/execute/commit/…) on a name that
    is neither a parameter of an enclosing function nor bound by
    `with <parameter>(...) as name`. Sessions injected by the caller (router
    `Depends(get_db)` → service parameter) are fine. One finding per file, at
    the first offending line.

Known limitations: containers held as class attributes, and session operations
on attributes (`self.db.query(...)`), are not inspected — Pydantic field
defaults and constructor-injected sessions look identical to stores statically.

Every finding is an ERROR so the integration repair loop regenerates the file;
the repair prompt carries the persistence contract excerpt.

The dialect parameter is part of the uniform checker signature and is unused.
"""
import ast
from typing import Any

from core.state import (
    FindingSeverity,
    PersistenceContract,
    StaticFinding,
    connection_import_line,
    connection_module_dotted,
)
from core.workspace import connection_module_path

IN_MEMORY_STATE = "in_memory_state"
DB_WITHOUT_CONNECTION_MODULE = "db_without_connection_module"
#: Check ids produced here — the repair prompt adds the persistence contract for these.
PERSISTENCE_CHECKS = frozenset({IN_MEMORY_STATE, DB_WITHOUT_CONNECTION_MODULE})

_CONTAINER_LITERALS = (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)
_CONTAINER_FACTORIES = {"list", "dict", "set", "defaultdict", "OrderedDict", "deque"}
#: Factories whose first positional argument configures, rather than fills, the container.
_CONFIGURED_FACTORIES = {"defaultdict", "deque"}
_MUTATORS = {"append", "appendleft", "extend", "insert", "add", "update", "setdefault",
             "pop", "popitem", "remove", "discard", "clear"}

#: Engine / client constructors only the connection module may call.
_ENGINE_FACTORIES = {"create_engine", "create_async_engine", "sessionmaker", "async_sessionmaker",
                     "MongoClient", "AsyncIOMotorClient"}
#: DB-API driver modules whose `.connect()` opens a raw connection.
_DRIVER_MODULES = {"sqlite3", "psycopg2", "psycopg", "pymysql", "aiosqlite", "asyncpg"}
#: Session/connection operations that read or write the database.
_SESSION_OPS = {"query", "execute", "commit", "rollback", "flush", "refresh", "scalars", "scalar"}


def check_persistence(
    files: dict[str, str],
    manifest: Any,
    architecture: Any = None,
    dialect: str = "",
) -> list[StaticFinding]:
    """Error findings for in-memory storage in backend files of a database-backed stack."""
    contract = getattr(architecture, "persistence_contract", None)
    if contract is None:
        return []

    findings: list[StaticFinding] = []
    for path in _backend_python_files(files, manifest, contract):
        try:
            tree = ast.parse(files[path])
        except SyntaxError:
            continue
        findings.extend(_in_memory_state(path, tree))
        findings.extend(_connection_bypass(path, tree, contract))
    return findings


def _backend_python_files(files: dict[str, str], manifest: Any, contract: PersistenceContract) -> list[str]:
    manifest_files = getattr(manifest, "files", {}) or {}
    module = connection_module_path(contract)
    return sorted(
        path for path in files
        if path.endswith(".py")
        and str(getattr(manifest_files.get(path), "agent", "") or "").lower() == "backend"
        and path != module
    )


# ── in_memory_state ──────────────────────────────────────────────────────────

def _in_memory_state(path: str, tree: ast.Module) -> list[StaticFinding]:
    mutated = _mutated_names(tree)
    findings = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        names = [t.id for t in targets if isinstance(t, ast.Name) and not _is_dunder(t.id)]
        if not names or not _is_container(value):
            continue
        store = next((n for n in names if n in mutated), None)
        if store is None and _is_empty_container(value):
            store = names[0]
        if store is not None:
            findings.append(StaticFinding(
                file=path,
                check=IN_MEMORY_STATE,
                severity=FindingSeverity.ERROR,
                line=node.lineno,
                message=(
                    f"module-level mutable state `{store}` keeps data in memory — "
                    "persist it to the database through the connection module instead"
                ),
            ))
    return findings


def _mutated_names(tree: ast.Module) -> set[str]:
    """Names mutated anywhere: x.append(...), x[k] = v, del x[k], x += …"""
    mutated: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _MUTATORS and isinstance(func.value, ast.Name):
                mutated.add(func.value.id)
        targets: list[ast.expr] = []
        if isinstance(node, (ast.Assign, ast.Delete)):
            targets = list(node.targets)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                mutated.add(target.value.id)
            elif isinstance(node, ast.AugAssign) and isinstance(target, ast.Name):
                mutated.add(target.id)
    return mutated


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _is_container(value: ast.expr) -> bool:
    if isinstance(value, _CONTAINER_LITERALS):
        return True
    return isinstance(value, ast.Call) and _call_name(value) in _CONTAINER_FACTORIES


def _is_empty_container(value: ast.expr) -> bool:
    if isinstance(value, (ast.List, ast.Set)):
        return not value.elts
    if isinstance(value, ast.Dict):
        return not value.keys
    if isinstance(value, ast.Call):
        max_args = 1 if _call_name(value) in _CONFIGURED_FACTORIES else 0
        return len(value.args) <= max_args and not value.keywords
    return False


# ── db_without_connection_module ─────────────────────────────────────────────

def _connection_bypass(path: str, tree: ast.Module, contract: PersistenceContract) -> list[StaticFinding]:
    """At most one finding: the first data access that bypasses the connection module."""
    imports_connection = _imports_connection_module(tree, contract)
    offenders: list[int] = []

    def visit(node: ast.AST, params: frozenset[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            params = params | _param_names(node.args)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            params = params | _sessions_from_injected_factories(node, params)
        if isinstance(node, ast.Call):
            if _is_connection_factory(node):
                offenders.append(node.lineno)
            elif not imports_connection and _is_uninjected_session_op(node, params):
                offenders.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child, params)

    visit(tree, frozenset())
    if not offenders:
        return []
    return [StaticFinding(
        file=path,
        check=DB_WITHOUT_CONNECTION_MODULE,
        severity=FindingSeverity.ERROR,
        line=min(offenders),
        message=(
            "data access bypasses the connection module — obtain sessions only via "
            f"`{connection_import_line(contract)}` (or a session parameter injected by the caller)"
        ),
    )]


def _imports_connection_module(tree: ast.Module, contract: PersistenceContract) -> bool:
    dotted = connection_module_dotted(contract)
    package, _, leaf = dotted.rpartition(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if module == dotted or module.endswith("." + dotted):
                return True
            if (module == package or module.endswith("." + package)) and any(a.name == leaf for a in node.names):
                return True
        elif isinstance(node, ast.Import):
            if any(a.name == dotted or a.name.endswith("." + dotted) for a in node.names):
                return True
    return False


def _param_names(args: ast.arguments) -> frozenset[str]:
    every = [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]
    return frozenset(a.arg for a in every if a is not None)


def _sessions_from_injected_factories(node: ast.With | ast.AsyncWith, params: frozenset[str]) -> frozenset[str]:
    """Names bound by `with <param>(...) as name` — sessions opened from an injected factory."""
    bound = set()
    for item in node.items:
        ctx, var = item.context_expr, item.optional_vars
        if (isinstance(var, ast.Name) and isinstance(ctx, ast.Call)
                and isinstance(ctx.func, ast.Name) and ctx.func.id in params):
            bound.add(var.id)
    return frozenset(bound)


def _call_name(call: ast.Call) -> str:
    func = call.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""


def _is_connection_factory(call: ast.Call) -> bool:
    """Engine / connection / client creation — only the connection module may do this."""
    name = _call_name(call)
    if name in _ENGINE_FACTORIES:
        return True
    func = call.func
    return (
        name == "connect" and isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name) and func.value.id in _DRIVER_MODULES
    )


def _is_uninjected_session_op(call: ast.Call, params: frozenset[str]) -> bool:
    """`x.query(...)`/`x.commit()`… on a name that is not an injected session."""
    func = call.func
    return (
        isinstance(func, ast.Attribute) and func.attr in _SESSION_OPS
        and isinstance(func.value, ast.Name) and func.value.id not in params
    )
