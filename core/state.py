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

from core.config import settings


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
    REWORK       = "rework"
 
 
class ReviewVerdict(str, Enum):
    PASS   = "pass"
    REWORK = "rework"
    FAIL   = "fail"
 

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


def resolve_db_tooling(database_tech: str) -> dict[str, str]:
    """Resolve ORM/migration strategy deterministically from a database tech string."""
    db_key = (database_tech or "").lower()
    for name, mapping in DB_ORM_MAP.items():
        if name.lower() in db_key:
            return mapping
    return dict(_DEFAULT_ORM)


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
    devops: dict[str, str] = Field(default_factory=dict)
 
 
class IntegrationStatus(BaseModel):
    merged: bool = False
    conflicts: list[str] = Field(default_factory=list)
    unified_entry_point: str = ""
    #: Truly-unresolved conflicts grouped by owning agent ("backend"/"frontend"/"database"/"devops").
    #: The graph reads this to route a REWORK loop back to the relevant coding agents.
    conflicts_by_agent: dict[str, list[str]] = Field(default_factory=dict)
    #: Truly-unresolved conflicts grouped by file path — used to inject rework
    #: instructions into the coding-agent prompts when a file is regenerated.
    rework_targets: dict[str, list[str]] = Field(default_factory=dict)
 
 
class TestResults(BaseModel):
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    coverage_pct: float = 0.0
    failures: list[dict[str, str]] = Field(default_factory=list)   # [{test, reason}]
    report_path: str = ""
 
 
class SecurityScan(BaseModel):
    scanned_at: Optional[datetime] = None
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    findings: list[dict[str, Any]] = Field(default_factory=list)
    passed: bool = False
 
 
class ReviewFeedback(BaseModel):
    verdict: ReviewVerdict = ReviewVerdict.REWORK
    score: float = 0.0                                              # 0-100
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    notes: str = ""
    rework_instructions: list[str] = Field(default_factory=list)
 
 
class PerformanceMetrics(BaseModel):
    total_tokens_used: int = 0
    total_llm_calls: int = 0
    agent_durations: dict[str, float] = Field(default_factory=dict)  # agent -> seconds

 
 
class MemoryPointers(BaseModel):
    session_checkpoint_id: str = ""
    long_term_refs: list[str] = Field(default_factory=list)
    rag_context_ids: list[str] = Field(default_factory=list)



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

    # ── Integration REWORK loop (mod 1) ───────────────────────────────
    # integration -> {coding agent} -> integration, capped at max rounds.
    # Cap sourced from config (REWORK_TRIAL env) so it's tunable without code changes.
    integration_rework_rounds: int = 0
    max_integration_rounds: int = settings.rework_trial or 2

    # agents outputs

    requirements:    SRSData       = Field(default_factory=SRSData)
    architecture:    ArchData      = Field(default_factory=ArchData)
    tasks:           Annotated[TaskList, merge_tasks] = Field(default_factory=TaskList)
    code_manifest:   Annotated[CodeManifest, merge_code_manifest] = Field(default_factory=CodeManifest)   # ← file-level contracts
    generated_code:  Annotated[CodeMap, merge_code_map] = Field(default_factory=CodeMap)
    integration:     IntegrationStatus = Field(default_factory=IntegrationStatus)
    tests:           TestResults   = Field(default_factory=TestResults)
    security:        SecurityScan  = Field(default_factory=SecurityScan)
    review:          ReviewFeedback = Field(default_factory=ReviewFeedback)

    ## human in the loop  
    hitl: Optional[HITLRequest]=None
    
    # ── Supporting layers (parallel-safe via reducers) ──────────────────────
    artifacts: dict[str, str] = Field(default_factory=dict)        # label -> filepath

    metrics: Annotated[PerformanceMetrics, merge_performance_metrics] = Field(default_factory=PerformanceMetrics)
    memory_refs: MemoryPointers = Field(default_factory=MemoryPointers)

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