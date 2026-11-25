"""
core/state.py
─────────────
Single source of truth for the entire pipeline.
Every agent reads from and writes to a typed subset of ProjectState.
LangGraph passes this object through every node boundary and checkpoints it automatically.

Parallel-safe design:
  Each graph node returns ONLY the state keys it changed, computed by diff_state().
  Keys that parallel nodes change (Code Manifest, generated code, metrics, errors,
  current agent) each have a state-update policy pairing the delta a node sends
  with the reducer that merges it, so the fan-in is independent of finish order.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Any, Callable, Optional, Annotated, get_type_hints
from functools import cache
from datetime import datetime

from core.entity_contract import Entity


# ────────────────────────────────────────────────────────────────────────────────
# Reducer functions for parallel-safe state merging
# ────────────────────────────────────────────────────────────────────────────────

def merge_performance_metrics(
    current: Any,
    update: Any,
) -> Any:
    """
    Merge two PerformanceMetrics or dict instances.

    Additive fields (total_tokens_used, total_llm_calls) are summed.
    agent_durations dicts are merged (update overwrites per-agent key).
    """
    def _get_val(obj: Any, key: str, default: Any = 0) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    current_tokens = _get_val(current, "total_tokens_used", 0)
    update_tokens = _get_val(update, "total_tokens_used", 0)

    current_calls = _get_val(current, "total_llm_calls", 0)
    update_calls = _get_val(update, "total_llm_calls", 0)

    current_durations = _get_val(current, "agent_durations", {}) or {}
    update_durations = _get_val(update, "agent_durations", {}) or {}

    res = PerformanceMetrics(
        total_tokens_used=current_tokens + update_tokens,
        total_llm_calls=current_calls + update_calls,
        agent_durations={**current_durations, **update_durations},
    )

    if isinstance(current, dict) or isinstance(update, dict):
        return res.model_dump()
    return res


def merge_errors(
    current: list[dict[str, str]],
    update: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Append new errors from the update to the current list."""
    return current + update


def pick_latest_agent(current: str, update: str) -> str:
    """Pick the most recent agent name (last writer wins)."""
    return update if update else current


def merge_code_manifest(current: Any, update: Any) -> Any:
    """Merge CodeManifest entries across parallel node updates."""
    def _get_files(obj: Any) -> dict:
        if isinstance(obj, dict):
            return obj.get("files", {})
        if hasattr(obj, "files"):
            return {k: (v.model_dump(mode="json") if hasattr(v, "model_dump") else v) for k, v in obj.files.items()}
        return {}

    files_a = _get_files(current)
    files_b = _get_files(update)
    merged_files = {**files_a, **files_b}

    if isinstance(current, dict) or isinstance(update, dict):
        return {"files": merged_files}
    return CodeManifest(files=merged_files)


def merge_code_map(current: Any, update: Any) -> Any:
    """Merge CodeMap (database, backend, frontend dicts) across parallel node updates."""
    def _get_bucket(obj: Any, bucket: str) -> dict:
        if isinstance(obj, dict):
            return obj.get(bucket, {})
        return getattr(obj, bucket, {}) or {}

    res = CodeMap(
        database={**_get_bucket(current, "database"), **_get_bucket(update, "database")},
        backend={**_get_bucket(current, "backend"), **_get_bucket(update, "backend")},
        frontend={**_get_bucket(current, "frontend"), **_get_bucket(update, "frontend")},
    )
    if isinstance(current, dict) or isinstance(update, dict):
        return res.model_dump(mode="json")
    return res


# ────────────────────────────────────────────────────────────────────────────────
# State-update policies: one per reducer key, holding its delta rule and its merge rule
# ────────────────────────────────────────────────────────────────────────────────
#
# A node computes its changes with diff_state(before, after). For a key with a
# policy, the change sent is the policy's delta; LangGraph folds it into the
# current value with the policy's merge (the reducer annotated on ProjectState).
# Keys without a policy are replaced by the node that changed them.
#
# One owner per file: each File Manifest has exactly one owning coding agent, and
# the database, backend and frontend coding agents running in parallel only
# change their own File Manifests (and their own generated code). That is what
# makes the per-path merges below order-independent at the fan-in. The merge
# does no conflict detection: a reducer can't tell a sibling's change apart
# from the Planner's original value, so breaking the rule means last writer wins.

class StatePolicy:
    """How one reducer key's changes travel between nodes.

    delta(before, after) takes the key's JSON values from an agent's before and
    after snapshots and returns the change to send, or None when nothing changed.
    merge(current, update) is the reducer that folds a sent change into state.
    Keeping both halves together means changing one can't silently break the other.
    """

    def __init__(self, delta: Callable[[Any, Any], Any], merge: Callable[[Any, Any], Any]) -> None:
        self._delta = delta
        self._merge = merge

    def delta(self, before: Any, after: Any) -> Any:
        return self._delta(before, after)

    def merge(self, current: Any, update: Any) -> Any:
        return self._merge(current, update)


def _code_manifest_delta(before: dict, after: dict) -> Optional[dict]:
    """Only the File Manifests that differ from the before snapshot."""
    before_files = before.get("files", {})
    changed = {
        path: fm for path, fm in after.get("files", {}).items()
        if before_files.get(path) != fm
    }
    return {"files": changed} if changed else None


def _code_map_delta(before: dict, after: dict) -> Optional[dict]:
    """Only the generated files, per bucket, that differ from the before snapshot."""
    changed = {
        bucket: {
            path: code for path, code in files.items()
            if before.get(bucket, {}).get(path) != code
        }
        for bucket, files in after.items()
    }
    return changed if any(changed.values()) else None


def _metrics_delta(before: dict, after: dict) -> Optional[dict]:
    """Counter deltas plus the agent durations that changed."""
    tokens = after["total_tokens_used"] - before["total_tokens_used"]
    calls = after["total_llm_calls"] - before["total_llm_calls"]
    durations = {
        agent: seconds for agent, seconds in after["agent_durations"].items()
        if before["agent_durations"].get(agent) != seconds
    }
    if not (tokens or calls or durations):
        return None
    return PerformanceMetrics(
        total_tokens_used=tokens,
        total_llm_calls=calls,
        agent_durations=durations,
    ).model_dump(mode="json")


def _errors_delta(before: list, after: list) -> Optional[list]:
    """Only the errors appended since the before snapshot."""
    return after[len(before):] or None


def _write_refusals_delta(before: dict, after: dict) -> Optional[dict]:
    """New or changed refusals by path; a cleared refusal travels as None."""
    changed: dict[str, Optional[str]] = {
        path: reason for path, reason in after.items() if before.get(path) != reason
    }
    changed.update({path: None for path in before if path not in after})
    return changed or None


def merge_write_refusals(current: dict, update: dict) -> dict:
    """Fold refusals per path; a None value clears that path's refusal."""
    merged = {**(current or {}), **(update or {})}
    return {path: reason for path, reason in merged.items() if reason is not None}


def _current_agent_delta(before: str, after: str) -> Optional[str]:
    return after if after != before else None


#: Code Manifest: per-File-Manifest deltas, merged by per-path overwrite.
CODE_MANIFEST_POLICY  = StatePolicy(_code_manifest_delta, merge_code_manifest)
#: Generated code: per-file deltas within each bucket, merged per file.
GENERATED_CODE_POLICY = StatePolicy(_code_map_delta, merge_code_map)
#: Metrics: counter deltas, summed.
METRICS_POLICY        = StatePolicy(_metrics_delta, merge_performance_metrics)
#: Errors: new entries only, appended.
ERRORS_POLICY         = StatePolicy(_errors_delta, merge_errors)
#: Current agent: the latest wins.
CURRENT_AGENT_POLICY  = StatePolicy(_current_agent_delta, pick_latest_agent)
#: Write refusals: per-path deltas (None clears), merged per path.
WRITE_REFUSALS_POLICY = StatePolicy(_write_refusals_delta, merge_write_refusals)


@cache
def _state_policies() -> dict[str, StatePolicy]:
    """Every ProjectState key whose annotated reducer is a policy's merge."""
    policies: dict[str, StatePolicy] = {}
    hints = get_type_hints(ProjectState, include_extras=True)
    for key, hint in hints.items():
        for meta in getattr(hint, "__metadata__", ()):
            owner = getattr(meta, "__self__", None)
            if isinstance(owner, StatePolicy):
                policies[key] = owner
    return policies


def diff_state(before: "ProjectState", after: "ProjectState") -> dict[str, Any]:
    """The changes a node returns: only the state keys that differ between its
    before and after snapshots, as JSON.

    Keys with a state-update policy send that policy's delta, so LangGraph's
    reducers can merge parallel nodes' changes; every other key is sent whole.
    """
    before_dict = before.model_dump(mode="json")
    after_dict = after.model_dump(mode="json")
    policies = _state_policies()

    changes: dict[str, Any] = {}
    for key, after_value in after_dict.items():
        policy = policies.get(key)
        if policy is None:
            if before_dict[key] != after_value:
                changes[key] = after_value
            continue
        delta = policy.delta(before_dict[key], after_value)
        if delta is not None:
            changes[key] = delta
    return changes


################ Enum Values######################################################################  
class ProjectStatus(str, Enum):
    PENDING      = "pending"
    RUNNING      = "running"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED    = "completed"
    FAILED       = "failed"
  

###################### data models##############################################################

#: The supported-database table: the single source of each database's tooling.
#: The Persistence Contract is built from it, so the LLM can never override it.
#:   orm           — the one data-access layer every coding agent uses
#:   dialect       — sqlglot dialect name; None for non-SQL databases
#:   model_dir     — where ORM model files live, relative to the backend Layer root
#:   migration_dir — empty: Alembic is gone, the ORM creates the schema
#:   example_url   — placeholder DB URL for `.env.example` (never real secrets)
DB_ORM_MAP: dict[str, dict[str, Optional[str]]] = {
    "PostgreSQL": {"orm": "SQLAlchemy", "dialect": "postgres", "model_dir": "app/models/", "migration_dir": "",
                   "example_url": "postgresql://app:change-me@localhost:5432/app"},
    "SQLite":     {"orm": "SQLAlchemy", "dialect": "sqlite",   "model_dir": "app/models/", "migration_dir": "",
                   "example_url": "sqlite:///./app.db"},
    "MongoDB":    {"orm": "Beanie",     "dialect": None,       "model_dir": "app/models/", "migration_dir": "",
                   "example_url": "mongodb://localhost:27017/app"},
    "MySQL":      {"orm": "SQLAlchemy", "dialect": "mysql",    "model_dir": "app/models/", "migration_dir": "",
                   "example_url": "mysql+pymysql://app:change-me@localhost:3306/app"},
}


def _match_db_name(database_tech: str) -> Optional[str]:
    """Return the DB_ORM_MAP key whose name appears in the tech string, or None."""
    db_key = (database_tech or "").lower()
    for name in DB_ORM_MAP:
        if name.lower() in db_key:
            return name
    return None


#: The database every run falls back to: the stack resolver uses it when no
#: database is requested and substitutes it for an unsupported one.
DEFAULT_DATABASE = "PostgreSQL 16"


def is_supported_database(database_tech: str) -> bool:
    """True when the database tech string names a database in DB_ORM_MAP."""
    return _match_db_name(database_tech) is not None


class PersistenceContract(BaseModel):
    """How generated code persists data — derived from the tech stack, never by an LLM."""
    database:          str             # canonical name, a DB_ORM_MAP key
    orm:               str
    sql_dialect:       Optional[str]   # sqlglot dialect name; None for non-SQL databases
    model_dir:         str             # ORM model directory, relative to the backend Layer root
    migration_dir:     str             # always empty: no migration files are planned
    migrations_tool:   Optional[str]   # always None: the ORM creates the schema (create_all-style)
    db_url_env_var:    str             # canonical env var every artifact reads the DB URL from
    connection_module: str             # path relative to the backend root
    init_command:      str             # run once before serving


#: Canonical DB URL env var — every artifact that reads the DB URL uses this name.
DB_URL_ENV_VAR = "DATABASE_URL"

#: Backend DB connection module, relative to the backend root.
CONNECTION_MODULE = "db/session.py"

#: Fixed exports of the connection module (engine, session factory, session
#: dependency) — pinned in the manifest so backend imports never depend on
#: which parallel coding agent writes first.
CONNECTION_MODULE_EXPORTS: tuple[str, ...] = ("engine", "SessionLocal", "get_db")

class UnknownDatabaseError(ValueError):
    """The database tech string matches no entry in DB_ORM_MAP."""


def build_persistence_contract(database_tech: str) -> PersistenceContract:
    """Build the persistence contract for a database tech string.

    Raises UnknownDatabaseError for an unrecognised database instead of
    silently falling back to the SQLite-style mapping.
    """
    name = _match_db_name(database_tech)
    if name is None:
        raise UnknownDatabaseError(
            f"No persistence contract for database {database_tech!r}; "
            f"known databases: {', '.join(DB_ORM_MAP)}"
        )

    # No migrations tool for any database: running the connection module
    # creates the schema (create_all() for SQL; a no-op for MongoDB/Beanie).
    module = CONNECTION_MODULE.removesuffix(".py").replace("/", ".")
    init_command = f"python -m {module}"

    mapping = DB_ORM_MAP[name]
    return PersistenceContract(
        database=name,
        orm=mapping["orm"],
        sql_dialect=mapping["dialect"],
        model_dir=mapping["model_dir"],
        migration_dir=mapping["migration_dir"],
        migrations_tool=None,
        db_url_env_var=DB_URL_ENV_VAR,
        connection_module=CONNECTION_MODULE,
        init_command=init_command,
    )


def build_env_example(contract: PersistenceContract) -> str:
    """Deterministic `.env.example` content rendered from the persistence contract.

    Declares the contract's canonical DB URL variable with the example URL of
    the contract's database.
    """
    example_url = DB_ORM_MAP[contract.database]["example_url"]
    return (
        "# Copy to .env and fill in real values. Never commit real secrets.\n"
        f"{contract.db_url_env_var}={example_url}\n"
    )


def _build_motor_connection_module(contract: PersistenceContract) -> str:
    """Connection module for document stores (Beanie/MongoDB) — Motor client + database dependency."""
    env_var = contract.db_url_env_var
    return f'''"""
DB connection module - synthesized from the persistence contract ({contract.orm}).
Generated deterministically; do not hand-edit. Reads the URL from ${env_var}.
"""
import os

from motor.motor_asyncio import AsyncIOMotorClient

DATABASE_URL = os.environ["{env_var}"]

engine = AsyncIOMotorClient(DATABASE_URL)


def SessionLocal():
    """Return the database named in the URL."""
    return engine.get_default_database()


async def get_db():
    """Database dependency: yield the database handle."""
    yield SessionLocal()
'''


def build_connection_module(
    contract: PersistenceContract, model_modules: list[str], base_module: str | None = None
) -> str:
    """Deterministic source of the DB connection module (contract.connection_module).

    Defines CONNECTION_MODULE_EXPORTS — engine, session factory, session
    dependency — and reads the DB URL from the contract's env var only, so no
    host, database name or credentials are ever hardcoded.
    """
    env_var = contract.db_url_env_var
    if contract.orm != "SQLAlchemy":
        return _build_motor_connection_module(contract)
    # The init command (`python -m <module>`) creates every table declared by
    # the model modules with create_all().
    create_schema = f'''

def create_schema():
    """Create every table declared by the model modules (create_all())."""
    import importlib

    from sqlalchemy import MetaData

    for module_name in {model_modules!r}:
        module = importlib.import_module(module_name)
        for value in vars(module).values():
            metadata = getattr(value, "metadata", None)
            if isinstance(metadata, MetaData):
                metadata.create_all(bind=engine)


if __name__ == "__main__":
    create_schema()
'''
    # Re-export the declarative Base so callers reach it via the connection
    # module; imported last so a model module importing `engine` is not circular.
    if base_module:
        create_schema = f"\n\nfrom {base_module} import Base  # noqa: E402\n{create_schema}"
    return f'''"""
DB connection module - synthesized from the persistence contract ({contract.orm}).
Generated deterministically; do not hand-edit. Reads the URL from ${env_var}.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ["{env_var}"]

# SQLite connections are shared across FastAPI's threadpool.
_connect_args = {{"check_same_thread": False}} if DATABASE_URL.startswith("sqlite") else {{}}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """Session dependency: yield a session, always close it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
{create_schema}'''


def connection_module_dotted(contract: PersistenceContract) -> str:
    """Dotted import path of the connection module, e.g. 'db.session'."""
    return contract.connection_module.removesuffix(".py").replace("/", ".")


def connection_import_line(contract: PersistenceContract) -> str:
    """The one import statement code uses to reach the connection module."""
    return f"from {connection_module_dotted(contract)} import {', '.join(CONNECTION_MODULE_EXPORTS)}"


def render_persistence_contract(contract: PersistenceContract) -> str:
    """Prompt block giving every downstream agent identical persistence values."""
    migrations = contract.migrations_tool or "none (the ORM creates the schema)"
    return (
        "## Persistence Contract (fixed — every agent uses exactly these values)\n"
        f"- ORM: {contract.orm}\n"
        f"- Migrations tool: {migrations}\n"
        f"- DB URL env var: {contract.db_url_env_var} (read from the environment only — never hardcode)\n"
        f"- Connection module: {contract.connection_module} (relative to the backend root; "
        f"synthesized, do not write it) — import with `{connection_import_line(contract)}`\n"
        f"- Init command (run once before serving): {contract.init_command}"
    )


def backend_persistence_rules(contract: PersistenceContract) -> str:
    """Backend system-prompt rules forcing real persistence."""
    return (
        "## Persistence Mandate (MUST follow)\n"
        f"- Routers and services obtain database sessions ONLY from the connection module: "
        f"`{connection_import_line(contract)}` (`get_db` yields a session and closes it). "
        "Never create engines, sessions or connections anywhere else.\n"
        f"- All data persists to the database through {contract.orm}. NEVER use module-level "
        "lists, dicts or other mutable globals as data storage — no in-memory stores, even for "
        "simple apps."
    )


#: Persistence mandate appended to every complexity-tier calibration block —
#: a smaller tier means a smaller database, never no database.
_MINIMAL_PERSISTENCE_MANDATE = (
    "- Persistence: store all data in a small real database through the ORM — never in-memory."
)
_PERSISTENCE_MANDATE = (
    "- Persistence: all data persists to the configured database through the ORM — no in-memory storage."
)


def persistence_mandate(tier: str) -> str:
    """The persistence rule for a complexity tier (MINIMAL still means a real database)."""
    return _MINIMAL_PERSISTENCE_MANDATE if (tier or "").upper() == "MINIMAL" else _PERSISTENCE_MANDATE


#: Complexity tiers for calibrating output size to user intent (Problem 4).
COMPLEXITY_TIERS: tuple[str, ...] = ("MINIMAL", "STANDARD", "ENTERPRISE")


def classify_complexity(raw_prompt: str) -> str:
    """Keyword-scan the raw user prompt to assign a complexity tier."""
    text = (raw_prompt or "").lower()
    enterprise = ("enterprise", "production", "production-grade", "scalable", "microservices", "high-availability")
    minimal    = ("very simple", "simple", "basic", "just a", "toy", "demo", "tiny", "skeleton", "minimal")
    if any(k in text for k in enterprise):
        return "ENTERPRISE"
    if any(k in text for k in minimal):
        return "MINIMAL"
    return "STANDARD"


class SRSData(BaseModel):
    """output of req agent"""
    raw_prompt: str = ""
    functional_requirements: list[str] = Field(default_factory=list)
    non_functional_requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)   # raised if prompt is ambiguous
    is_complete: bool = False

class APIContract(BaseModel):
    """A single REST endpoint contract produced by the Architecture Agent."""
    method:        str                        # HTTP verb: GET, POST, PUT, DELETE, PATCH
    path:          str                        # URL path e.g. "/api/v1/boards"
    auth_required: bool = True
    description:   str = ""                   # what the endpoint does
    request_body:  dict[str, str] = Field(default_factory=dict)   # field → type
    response_body: dict[str, str] = Field(default_factory=dict)   # field → type

class ArchData(BaseModel):
    """Output of the Architecture Agent."""
    tech_stack: dict[str, str] = Field(default_factory=dict)        # e.g. {"backend": "FastAPI", ...}
    system_components: list[str] = Field(default_factory=list)
    api_contracts: list[APIContract] = Field(default_factory=list)
    database_schema_overview: str = ""
    #: The Entity Contract (core/entity_contract.py): structured, field-level, LLM-produced;
    #: shared by the database and backend coding agents. Separate from persistence_contract.
    entities: list[Entity] = Field(default_factory=list)
    security_requirements: list[str] = Field(default_factory=list)
    deployment_strategy: str = ""
    diagrams: list[str] = Field(default_factory=list)               # file paths
    complexity_tier: str = "STANDARD"                                # MINIMAL | STANDARD | ENTERPRISE
    # Every run has exactly one; the architecture step sets it from the resolved stack.
    persistence_contract: PersistenceContract = Field(
        default_factory=lambda: build_persistence_contract(DEFAULT_DATABASE)
    )
    assumptions: list[str] = Field(default_factory=list)            # e.g. an unsupported-database substitution

 
class Task(BaseModel):
    id: str
    agent: str                                                       # which agent owns this task
    title: str
    description: str
    depends_on: list[str] = Field(default_factory=list)             # task IDs
    status: str = "pending"                                         # set by the Planner only; Written is the done signal
    priority: int = 1
 
 
class TaskList(BaseModel):
    """Output of the Planner Agent."""
    tasks: list[Task] = Field(default_factory=list)
 
    def by_agent(self, agent: str) -> list[Task]:
        return [t for t in self.tasks if t.agent == agent]
 
    def pending(self) -> list[Task]:
        return [t for t in self.tasks if t.status == "pending"]
 
class Argument(BaseModel):
    name:    str      # e.g. "user_id"
    type:    str      # e.g. "UUID"
    default: str = "" # e.g. "None" if optional


class ClassName(BaseModel):
    name:      str             # e.g. "User"
    arguments: list[Argument] = Field(default_factory=list)  # __init__ args
    docstring: str             # what this class does


class FunctionName(BaseModel):
    name:        str             # e.g. "get_by_email"
    arguments:   list[Argument] = Field(default_factory=list)  # parameters
    return_type: str             # e.g. "User | None"
    docstring:   str             # what this function does


class FileManifest(BaseModel):
    """
    Contract entry for a single source file.

    Planner fills the top block (structure + exports contract).
    Coding agent fills the bottom block after writing the file.
    """
    # ── Planner fills — structure + strict export contract ──────────────
    agent:    str  # who owns this file
    task_id:  str  # which task produces it
    filepath: str  # exact path e.g. "backend/app/models/user.py"
    purpose:  str  # "User SQLAlchemy model" — what it should do
    exports:  list[str] = Field(default_factory=list)  # STRICT — what other files may import
    #: Import graph derived by the Planner (Pass 1): {source module path -> names imported}.
    #: The coding agents MUST use exactly these imports — never guessed paths.
    imports_from: dict[str, list[str]] = Field(default_factory=dict)
    #: Cross-agent field contract (Phase 2) — populated deterministically by the
    #: Planner for DB-model and schema files: {export name -> {field: type}}.
    #: Backend agents read this so schemas/CRUD match DB columns and API bodies
    #: exactly even while the database agent runs in parallel.
    contract: dict[str, dict[str, str]] = Field(default_factory=dict)
    #: Deterministic file content synthesized by the Planner (e.g. `.env.example`
    #: from the persistence contract). When set, the owning coding agent writes
    #: it verbatim — no LLM call.
    planned_content: Optional[str] = None

    # ── Coding agent fills after writing ────────────────────────────────
    written:   bool = False

    classes:   list[ClassName]    = Field(default_factory=list)  # name + arguments + docstring
    functions: list[FunctionName] = Field(default_factory=list)  # name + arguments + return type + docstring


class CodeManifest(BaseModel):
    """
    Registry of every file the Planner intends to be written.

    Key   → repo-relative file path  (e.g. "backend/app/models/user.py")
    Value → FileManifest describing who owns it and what it must export.

    Agents look up their own task_id here to know which file to write,
    then set `written = True` and fill `classes` and `functions` when done.
    Exports are pre-defined by the Planner and must not be changed by agents.
    """
    files: dict[str, FileManifest] = Field(default_factory=dict)

    # ── Helpers ────────────────────────────────────────────────────────
    def by_agent(self, agent: str) -> list[FileManifest]:
        """All file manifests assigned to a given agent."""
        return [m for m in self.files.values() if m.agent == agent]

    def by_task(self, task_id: str) -> Optional["FileManifest"]:
        """Return the manifest for a specific task ID, or None."""
        for m in self.files.values():
            if m.task_id == task_id:
                return m
        return None

    def mark_written(
        self,
        file_path: str,
        classes: list[ClassName] | None = None,
        functions: list[FunctionName] | None = None,
    ) -> None:
        """Called by the coding agent after it has written *file_path*.
        Exports are NOT updated here — they are set by the Planner.
        """
        if file_path in self.files:
            m = self.files[file_path]
            m.written   = True
            m.classes   = classes   or []
            m.functions = functions or []

    def unwritten(self) -> list[FileManifest]:
        """Manifests that have not yet been written — useful for progress checks."""
        return [m for m in self.files.values() if not m.written]

    def owner_of(self, file_path: str) -> str:
        """The owning agent key of *file_path* in the Code Manifest; "" when unplanned."""
        fm = self.files.get(file_path)
        return str(fm.agent or "") if fm is not None else ""


#: The generated-code buckets, one per Coding agent layer.
CODE_LAYERS: tuple[str, ...] = ("backend", "frontend", "database")

#: Owning agent key (as the Code Manifest names it, aliases included) → its code layer.
OWNER_LAYERS: dict[str, str] = {
    "backend": "backend",
    "frontend": "frontend",
    "database": "database",
    "data_base": "database",
    "db": "database",
}


def owner_layer(owner: str) -> Optional[str]:
    """The code layer an owning agent key writes to; None for an unknown owner."""
    return OWNER_LAYERS.get(str(owner or "").lower())


class CodeMap(BaseModel):
    """Accumulated code artifacts from engineering agents."""
    backend: dict[str, str] = Field(default_factory=dict)          # filepath -> code
    frontend: dict[str, str] = Field(default_factory=dict)
    database: dict[str, str] = Field(default_factory=dict)

    def all_files(self) -> dict[str, str]:
        """Every generated file across the buckets, by path."""
        files: dict[str, str] = {}
        for layer in CODE_LAYERS:
            files.update(getattr(self, layer) or {})
        return files

    def layer_of(self, filepath: str) -> Optional[str]:
        """The bucket holding *filepath*, else the layer its path names; None when neither."""
        for layer in CODE_LAYERS:
            if filepath in (getattr(self, layer) or {}):
                return layer
        norm = "/" + filepath.replace("\\", "/")
        return next((layer for layer in CODE_LAYERS if f"/{layer}/" in norm), None)

    def replace(self, filepath: str, code: str, layer: str) -> None:
        """Put *code* for *filepath* in *layer*'s bucket, removing it from every other."""
        for candidate in CODE_LAYERS:
            bucket = dict(getattr(self, candidate) or {})
            bucket.pop(filepath, None)
            if candidate == layer:
                bucket[filepath] = code
            setattr(self, candidate, bucket)

  
class FindingSeverity(str, Enum):
    """Severity of a static-check finding. Only errors trigger repair; warns log only."""
    ERROR = "error"
    WARN  = "warn"


class StaticFinding(BaseModel):
    """One result from the Integration Agent's static checks."""
    file:     str = ""                                              # workspace path e.g. output/backend/api/todo.py
    check:    str = ""                                              # check id e.g. python_syntax, python_imports, third_party, manifest_written
    severity: FindingSeverity = FindingSeverity.ERROR
    line:     int = 0                                               # 1-based source line; 0 = not applicable
    message:  str = ""


class StaticVerdict(str, Enum):
    """Outcome of the integration node's check/repair cycle.

    NOT_RUN is the default, so a run that never reached Integration can't be
    mistaken for a passing one.
    """
    NOT_RUN  = "not_run"
    PASS     = "pass"
    REPAIRED = "repaired"
    FAILED   = "failed"


class StaticChecks(BaseModel):
    """Accumulated result of the Integration Agent's static-check + repair cycle."""
    verdict:        StaticVerdict = StaticVerdict.NOT_RUN
    findings:       list[StaticFinding] = Field(default_factory=list)
    rounds:         int = 0                                          # repair rounds performed
    repaired_files: list[str] = Field(default_factory=list)
 
 
class PerformanceMetrics(BaseModel):
    total_tokens_used: int = 0
    total_llm_calls: int = 0
    agent_durations: dict[str, float] = Field(default_factory=dict)  # agent -> seconds

  
  
class HITLRequest(BaseModel):
    """
    Created by an agent when it needs human input before the pipeline can continue.
    The graph checks for this after every node and pauses if it is populated.
    """
    # Who raised the pause and why
    raised_by: str = ""                                             # agent name
    reason: str = ""                                               # short description
 
    # The actual questions the human must answer
    questions: list[str] = Field(default_factory=list)
 
    # Human fills these in — same order as questions
    answers: list[str] = Field(default_factory=list)
 
    # Lifecycle flags
    is_resolved: bool = False                                      # True once answers received



###################### project state model################################################################################
 
class ProjectState(BaseModel):
    # identity
    project_id:str=""
    project_name:str=""
    status:ProjectStatus=ProjectStatus.PENDING
    max_clarfication_rounds: int = 3
    clarfication_rounds: int = 0

    # agents outputs

    requirements:    SRSData       = Field(default_factory=SRSData)
    architecture:    ArchData      = Field(default_factory=ArchData)
    tasks:           TaskList      = Field(default_factory=TaskList)       # Planner writes; coding agents only read
    code_manifest:   Annotated[CodeManifest, CODE_MANIFEST_POLICY.merge] = Field(default_factory=CodeManifest)   # ← file-level contracts
    generated_code:  Annotated[CodeMap, GENERATED_CODE_POLICY.merge] = Field(default_factory=CodeMap)
    static_checks:   StaticChecks  = Field(default_factory=StaticChecks)
    # path -> why the safe-write guard refused its last write; cleared once written
    write_refusals:  Annotated[dict[str, str], WRITE_REFUSALS_POLICY.merge] = Field(default_factory=dict)

    ## human in the loop
    hitl: Optional[HITLRequest]=None

    # ── Supporting layers (parallel-safe via reducers) ──────────────────────
    metrics: Annotated[PerformanceMetrics, METRICS_POLICY.merge] = Field(default_factory=PerformanceMetrics)

    # ── Error tracking (parallel-safe via reducers) ─────────────────────────
    errors: Annotated[list[dict[str, str]], ERRORS_POLICY.merge] = Field(default_factory=list)     # [{agent, message, ts}]
    current_agent: Annotated[str, CURRENT_AGENT_POLICY.merge] = ""

    # ── Helpers ───────────────────────────────
    def log_error(self, agent: str, message: str) -> None:
        self.errors.append({
            "agent": agent,
            "message": message,
            "ts": datetime.utcnow().isoformat(),
        })
 
    def record_written(
        self,
        agent: str,
        filepath: str,
        code: str,
        classes: list[ClassName] | None = None,
        functions: list[FunctionName] | None = None,
    ) -> None:
        """Record a Coding agent's written file: its code in the agent's bucket, its File Manifest Written."""
        getattr(self.generated_code, agent)[filepath] = code
        self.code_manifest.mark_written(filepath, classes, functions)
        self.clear_refusal(filepath)

    def record_repaired(self, filepath: str, code: str) -> None:
        """Record a repaired file: its code in its owner's bucket (by the Code Manifest,
        else where it already is), and any write refusal for it cleared."""
        layer = owner_layer(self.code_manifest.owner_of(filepath)) or self.generated_code.layer_of(filepath)
        if layer is None:
            raise ValueError(f"Cannot determine generated-code owner for '{filepath}'")
        self.generated_code.replace(filepath, code, layer)
        self.clear_refusal(filepath)

    def clear_refusal(self, filepath: str) -> None:
        """Forget *filepath*'s write refusal once it is written."""
        self.write_refusals.pop(filepath, None)

    def record_refusal(self, filepath: str, reason: str) -> None:
        """Record why the safe-write guard refused *filepath*'s write; a later write clears it."""
        self.write_refusals[filepath] = reason or "write refused by the safe-write guard"

    def mark_agent_start(self, agent: str) -> None:
        self.current_agent = agent
        self.metrics.agent_durations.setdefault(agent, 0.0)
 
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    ####################### human in the loop methods ##############################################################
    def pause(self, raised_by: str, questions: list[str], reason: str = "") -> None:
        """
        Called by an agent that needs human input before the pipeline continues.
        Sets status to AWAITING_HUMAN and populates hitl_request.
        The graph checks hitl_request after every node — if it is set, the graph pauses.
        """
        self.status      = ProjectStatus.AWAITING_HUMAN
        self.hitl = HITLRequest(
            raised_by = raised_by,
            reason    = reason,
            questions = questions,
        )
 
    def resume(self, answers: list[str]) -> None:
        """
        Called by the API when the human submits their answers.
        Injects answers into hitl_request, marks it resolved,
        and resets status so the graph can continue from where it paused.
        """
        if self.hitl is None:
            raise ValueError("resume() called but no active HITL request exists.")
 
        self.hitl.answers     = answers
        self.hitl.is_resolved = True
        self.status                   = ProjectStatus.RUNNING
 
    @property
    def is_paused(self) -> bool:
        """Convenience check — graph edge uses this to decide whether to pause."""
        return self.status == ProjectStatus.AWAITING_HUMAN