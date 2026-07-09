"""
core/state.py
─────────────
Single source of truth for the entire pipeline.
Every agent reads from and writes to a typed subset of ProjectState.
LangGraph passes this object through every node boundary and checkpoints it automatically.
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import Any, Optional
from datetime import datetime




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
    data_flow: str = ""
    api_contracts: list[APIContract] = Field(default_factory=list)
    database_schema_overview: str = ""
    deployment_strategy: str = ""
    diagrams: list[str] = Field(default_factory=list)               # file paths

 
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

    # agents outputs

    requirements:    SRSData       = Field(default_factory=SRSData)
    architecture:    ArchData      = Field(default_factory=ArchData)
    tasks:           TaskList      = Field(default_factory=TaskList)
    code_manifest:   CodeManifest  = Field(default_factory=CodeManifest)   # ← file-level contracts
    generated_code:  CodeMap       = Field(default_factory=CodeMap)
    integration:     IntegrationStatus = Field(default_factory=IntegrationStatus)
    tests:           TestResults   = Field(default_factory=TestResults)
    security:        SecurityScan  = Field(default_factory=SecurityScan)
    review:          ReviewFeedback = Field(default_factory=ReviewFeedback)

    ## human in the loop  
    hitl: Optional[HITLRequest]=None
    
    # ── Supporting layers ──────────────────────
    artifacts: dict[str, str] = Field(default_factory=dict)        # label -> filepath
    metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    memory_refs: MemoryPointers = Field(default_factory=MemoryPointers)

    # ── Error tracking ─────────────────────────
    errors: list[dict[str, str]] = Field(default_factory=list)     # [{agent, message, ts}]
    current_agent: str = ""

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
        return self.model_dump()

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