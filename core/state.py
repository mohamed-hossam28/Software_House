"""
core/state.py
─────────────
Single source of truth for the entire pipeline.
Every agent reads from and writes to a typed subset of ProjectState.
LangGraph passes this object through every node boundary and checkpoints it automatically.

Parallel-safe design:
  Each graph node returns ONLY the state keys it changed (a "diff dict").
  Shared keys (metrics, errors) use custom reducer functions so that
  LangGraph can merge outputs from parallel nodes without conflicts.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Any, Optional, Annotated
from datetime import datetime


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


def merge_completed_task_ids(a: Any, b: Any) -> set[str]:
    """Merge set of completed task IDs across parallel node updates."""
    set_a = set(a) if isinstance(a, (set, list, tuple)) else set()
    set_b = set(b) if isinstance(b, (set, list, tuple)) else set()
    return set_a | set_b


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


def merge_tasks(current: Any, update: Any) -> Any:
    """Merge TaskList across parallel node updates."""
    if not update:
        return current
    if not current:
        return update
    return update




################ Enum Values######################################################################  
class ProjectStatus(str, Enum):
    PENDING      = "pending"
    RUNNING      = "running"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED    = "completed"
    FAILED       = "failed"
  

###################### data models##############################################################

#: Deterministic ORM / migration strategy per database technology.
#: Used to enforce a single database paradigm (Problem 5) — the LLM can never override it.
DB_ORM_MAP: dict[str, dict[str, str]] = {
    "PostgreSQL": {"orm": "SQLAlchemy", "migrations": "Alembic",                 "model_dir": "app/models/", "migration_dir": "alembic/versions/"},
    "SQLite":     {"orm": "SQLAlchemy", "migrations": "SQLAlchemy create_all()", "model_dir": "app/models/", "migration_dir": "app/db/"},
    "MongoDB":    {"orm": "Beanie",     "migrations": "None",                    "model_dir": "app/models/", "migration_dir": ""},
    "MySQL":      {"orm": "SQLAlchemy", "migrations": "Alembic",                 "model_dir": "app/models/", "migration_dir": "alembic/versions/"},
}

#: Fallback used when the database tech string matches no known provider.
_DEFAULT_ORM: dict[str, str] = {
    "orm": "SQLAlchemy", "migrations": "SQLAlchemy create_all()",
    "model_dir": "app/models/", "migration_dir": "app/db/",
}


def _match_db_name(database_tech: str) -> Optional[str]:
    """Return the DB_ORM_MAP key whose name appears in the tech string, or None."""
    db_key = (database_tech or "").lower()
    for name in DB_ORM_MAP:
        if name.lower() in db_key:
            return name
    return None


def _match_db_mapping(database_tech: str) -> Optional[dict[str, str]]:
    """Return the DB_ORM_MAP entry whose name appears in the tech string, or None."""
    name = _match_db_name(database_tech)
    return DB_ORM_MAP[name] if name else None


def resolve_db_tooling(database_tech: str) -> dict[str, str]:
    """Resolve ORM/migration strategy deterministically from a database tech string."""
    return _match_db_mapping(database_tech) or dict(_DEFAULT_ORM)


class PersistenceContract(BaseModel):
    """How generated code persists data — derived from the tech stack, never by an LLM."""
    orm:               str
    migrations_tool:   Optional[str]   # None → schema is created by the ORM (create_all-style)
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

#: Real migration tools → the command that applies migrations before serving.
#: A DB_ORM_MAP "migrations" value not listed here (e.g. create_all()) means no tool.
_MIGRATION_INIT_COMMANDS: dict[str, str] = {
    "Alembic": "alembic upgrade head",
}


class UnknownDatabaseError(ValueError):
    """The database tech string matches no entry in DB_ORM_MAP."""


def build_persistence_contract(database_tech: str) -> PersistenceContract:
    """Build the persistence contract for a database tech string.

    Raises UnknownDatabaseError for an unrecognised database instead of
    silently falling back to the SQLite-style mapping.
    """
    mapping = _match_db_mapping(database_tech)
    if mapping is None:
        raise UnknownDatabaseError(
            f"No persistence contract for database {database_tech!r}; "
            f"known databases: {', '.join(DB_ORM_MAP)}"
        )

    migrations_tool = mapping["migrations"] if mapping["migrations"] in _MIGRATION_INIT_COMMANDS else None
    if migrations_tool:
        init_command = _MIGRATION_INIT_COMMANDS[migrations_tool]
    else:
        # No migrations tool: running the connection module creates the schema.
        module = CONNECTION_MODULE.removesuffix(".py").replace("/", ".")
        init_command = f"python -m {module}"

    return PersistenceContract(
        orm=mapping["orm"],
        migrations_tool=migrations_tool,
        db_url_env_var=DB_URL_ENV_VAR,
        connection_module=CONNECTION_MODULE,
        init_command=init_command,
    )


#: Example DB URL per DB_ORM_MAP key, written into `.env.example` (placeholder
#: credentials only — never real secrets).
_EXAMPLE_DB_URLS: dict[str, str] = {
    "PostgreSQL": "postgresql://app:change-me@localhost:5432/app",
    "SQLite":     "sqlite:///./app.db",
    "MongoDB":    "mongodb://localhost:27017/app",
    "MySQL":      "mysql+pymysql://app:change-me@localhost:3306/app",
}


def build_env_example(database_tech: str) -> str:
    """Deterministic `.env.example` content built from the persistence contract.

    For a known database, declares the contract's canonical DB URL variable
    with an example value. With no database (or an unknown one, which has no
    contract), declares no DB URL variable.
    """
    lines = ["# Copy to .env and fill in real values. Never commit real secrets."]
    name = _match_db_name(database_tech)
    if name is not None:
        contract = build_persistence_contract(database_tech)
        lines.append(f"{contract.db_url_env_var}={_EXAMPLE_DB_URLS[name]}")
    return "\n".join(lines) + "\n"


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


def build_connection_module(contract: PersistenceContract, model_modules: list[str]) -> str:
    """Deterministic source of the DB connection module (contract.connection_module).

    Defines CONNECTION_MODULE_EXPORTS — engine, session factory, session
    dependency — and reads the DB URL from the contract's env var only, so no
    host, database name or credentials are ever hardcoded.
    """
    env_var = contract.db_url_env_var
    if contract.orm != "SQLAlchemy":
        return _build_motor_connection_module(contract)
    create_schema = ""
    if contract.migrations_tool is None:
        # No migrations tool: the init command (`python -m <module>`) creates
        # every table declared by the model modules.
        create_schema = f'''

def create_schema():
    """Create every table declared by the model modules (no migrations tool)."""
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


def connection_import_line(contract: PersistenceContract) -> str:
    """The one import statement code uses to reach the connection module."""
    module = contract.connection_module.removesuffix(".py").replace("/", ".")
    return f"from {module} import {', '.join(CONNECTION_MODULE_EXPORTS)}"


def render_persistence_contract(contract: Optional[PersistenceContract]) -> str:
    """Prompt block giving every downstream agent identical persistence values."""
    if contract is None:
        return "## Persistence Contract\n(no database in this stack — nothing to persist to)"
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


def backend_persistence_rules(contract: Optional[PersistenceContract]) -> str:
    """Backend system-prompt rules forcing real persistence (empty with no database)."""
    if contract is None:
        return ""
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
    security_requirements: list[str] = Field(default_factory=list)
    deployment_strategy: str = ""
    diagrams: list[str] = Field(default_factory=list)               # file paths
    complexity_tier: str = "STANDARD"                                # MINIMAL | STANDARD | ENTERPRISE
    persistence_contract: Optional[PersistenceContract] = None       # None → database unknown (error logged)

 
class Task(BaseModel):
    id: str
    agent: str                                                       # which agent owns this task
    title: str
    description: str
    depends_on: list[str] = Field(default_factory=list)             # task IDs
    status: str = "pending"                                         # pending | in_progress | done
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


class CodeMap(BaseModel):
    """Accumulated code artifacts from engineering agents."""
    backend: dict[str, str] = Field(default_factory=dict)          # filepath -> code
    frontend: dict[str, str] = Field(default_factory=dict)
    database: dict[str, str] = Field(default_factory=dict)
  
  
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
    """Outcome of the integration node's check/repair cycle."""
    PASS     = "pass"
    REPAIRED = "repaired"
    FAILED   = "failed"


class StaticChecks(BaseModel):
    """Accumulated result of the Integration Agent's static-check + repair cycle."""
    verdict:        StaticVerdict = StaticVerdict.PASS
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
    tasks:           Annotated[TaskList, merge_tasks] = Field(default_factory=TaskList)
    code_manifest:   Annotated[CodeManifest, merge_code_manifest] = Field(default_factory=CodeManifest)   # ← file-level contracts
    generated_code:  Annotated[CodeMap, merge_code_map] = Field(default_factory=CodeMap)
    static_checks:   StaticChecks  = Field(default_factory=StaticChecks)

    ## human in the loop
    hitl: Optional[HITLRequest]=None

    # ── Supporting layers (parallel-safe via reducers) ──────────────────────
    metrics: Annotated[PerformanceMetrics, merge_performance_metrics] = Field(default_factory=PerformanceMetrics)

    # ── Error tracking & completed tasks (parallel-safe via reducers) ───────
    completed_task_ids: Annotated[set[str], merge_completed_task_ids] = Field(default_factory=set)
    errors: Annotated[list[dict[str, str]], merge_errors] = Field(default_factory=list)     # [{agent, message, ts}]
    current_agent: Annotated[str, pick_latest_agent] = ""

    # ── Helpers ───────────────────────────────
    def log_error(self, agent: str, message: str) -> None:
        self.errors.append({
            "agent": agent,
            "message": message,
            "ts": datetime.utcnow().isoformat(),
        })
 
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