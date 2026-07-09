"""
agents/planner.py
─────────────────
Planner Agent — runs after the Architecture Agent.

Input:  ProjectState.architecture  (ArchData — fully populated)
Output: ProjectState.tasks         (TaskList — granular, dependency-ordered task graph)

Execution order enforced via depends_on + priority:
  1. Database  (priority 1) — schema is the root; nothing depends on a schema that doesn't exist
  2. Backend   (priority 2) — depend on DB schema tasks
  3. Frontend  (priority 2) — PARALLEL to backend; depend on API-contract tasks only,
                              NOT on backend implementation tasks
  4. DevOps    (priority 3) — mostly independent; may reference folder-structure tasks

Granularity: one task = one file OR one cohesive function / feature.

What the Planner does NOT do:
  • Write code
  • Make architecture decisions  — those are already in state.architecture
  • Create tasks for agents outside the valid set {database, backend, frontend, devops}
  • Produce tasks with circular dependencies
"""

import logging
from typing import Any

from .base import BaseAgent
from core.state import (
    ProjectState,
    Task,
    TaskList,
    FileManifest,
    CodeManifest,
    APIContract,
)


logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────────
VALID_AGENTS: frozenset[str] = frozenset({"database", "backend", "frontend", "devops"})

# Canonical priority per agent — overrides any LLM drift
AGENT_PRIORITY: dict[str, int] = {
    "database": 1,
    "backend":  2,
    "frontend": 2,   # parallel to backend — depends on API contracts, not BE code
    "devops":   3,
}

# Task-ID prefix per agent
AGENT_PREFIX: dict[str, str] = {
    "database": "db",
    "backend":  "be",
    "frontend": "fe",
    "devops":   "do",
}


# ── System prompt ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a senior engineering project manager and technical lead with 15+ years of \
experience breaking down software architecture documents into granular, executable \
engineering tasks.

## Your job
Given a complete system architecture, produce a FLAT JSON array of engineering tasks — \
one task per file.

## Valid agents  (the ONLY values allowed in the "agent" field)
| Value      | Responsibility |
|------------|----------------|
| "database" | SQL schema files, migration files, seed / fixture data |
| "backend"  | routes, service classes, repository classes, Pydantic schemas, config |
| "frontend" | components, pages, hooks, API client module, routing, global styles |
| "devops"   | Dockerfiles, docker-compose, nginx config, CI/CD pipelines, .env templates |

## Execution order & dependency rules  (follow EXACTLY)
1. **database** tasks — priority 1 — `depends_on: []`  (they are the root)
2. **backend** tasks  — priority 2 — `depends_on` references relevant *database* task IDs
3. **frontend** tasks — priority 2 — `depends_on` references *database* schema task IDs or \
   a dedicated backend "api-contract" task ID — NEVER references backend implementation tasks
4. **devops** tasks   — priority 3 — `depends_on: []` unless they reference a specific schema task

## Forbidden patterns
- Circular dependencies  (A → B → A  is FORBIDDEN)
- Frontend tasks that depend on backend implementation tasks
- Task IDs that are not globally unique
- agents not in the valid set above
- Tasks that make architecture decisions (those are already done)
- One task that covers an entire layer  (e.g. "build the whole backend" is too coarse)

## Granularity examples
  ✓ "Create users table migration"           (database — one migration file)
  ✓ "Implement POST /api/auth/register"      (backend  — one route + handler function)
  ✓ "Build RegisterPage component"           (frontend — one .tsx file)
  ✓ "Write Dockerfile for backend service"   (devops   — one Dockerfile)
  ✗ "Build the entire backend"               (too coarse)
  ✗ "Add import statement to main.py"        (too fine)

## Task ID format
  <prefix>-<zero-padded 3-digit number>
  Prefix: database → db  |  backend → be  |  frontend → fe  |  devops → do
  Example: db-001, be-012, fe-003, do-002

## Output format
Respond with a SINGLE JSON array — no prose outside the array.
Every element must be:
{
  "id":          "<unique task ID>",
  "agent":       "<database | backend | frontend | devops>",
  "title":       "<short imperative sentence>",
  "description": "<detailed description: file path, exact function/class to build, \
contracts it must honour, acceptance criteria>",
  "depends_on":  ["<task-id>", ...],
  "priority":    <1 | 2 | 3>,
  "status":      "pending",

  "file":      "<repo-relative path to the SINGLE file this task produces, e.g. backend/app/models/user.py>",
  "exports":   ["<ClassName or function_name other agents MUST be able to import>"],
  "classes":   ["<top-level class names the file should define — planner expectation>"],
  "functions": ["<top-level function names the file should define — planner expectation>"]
}

Rules for the new fields:
- `file`    — every task MUST have a unique, non-empty file path.
- `exports` — be strict: only names that cross module boundaries belong here.
- `classes` / `functions` — suggestions; the coding agent may add helpers freely.
"""



# ── User prompt template ─────────────────────────────────────────────────────────
USER_PROMPT_TEMPLATE = """\
## Architecture Input

### Tech Stack
{tech_stack}

### System Components
{system_components}

### API Contracts
{api_contracts}

### Database Schema Overview
{database_schema_overview}

### Communication / Data Flow
{data_flow}

### Folder Structure & Infra
{deployment_strategy}

---

Generate the COMPLETE task list.
Cover ALL schema entities, ALL API endpoints, ALL pages/components implied by \
the API, and ALL devops artifacts (Dockerfile per service, compose, nginx, CI).
Output the JSON array only — no explanations:
```json
"""


# ── Formatting helpers ───────────────────────────────────────────────────────────

def _fmt_tech_stack(stack: dict[str, str]) -> str:
    return "\n".join(f"  {layer}: {tech}" for layer, tech in stack.items()) or "  (none)"


def _fmt_components(components: list[str]) -> str:
    return "\n".join(f"  - {c}" for c in components) or "  (none)"


def _fmt_api_contracts(contracts: list[APIContract]) -> str:
    if not contracts:
        return "  (none)"
    lines = []
    for c in contracts:
        lock  = "🔒" if c.auth_required else "🔓"
        lines.append(
            f"  {lock} {c.method:7} {c.path}\n"
            f"         desc         : {c.description}\n"
            f"         request_body : {c.request_body}\n"
            f"         response_body: {c.response_body}"
        )
    return "\n".join(lines)


# ── Validation helpers ───────────────────────────────────────────────────────────

def _has_cycle(tasks: list[Task]) -> bool:
    """Detect cycles in the dependency graph via iterative DFS (paint algorithm)."""
    graph: dict[str, list[str]] = {t.id: list(t.depends_on) for t in tasks}
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {t.id: WHITE for t in tasks}

    def dfs(start: str) -> bool:
        stack = [(start, False)]
        while stack:
            node, leaving = stack.pop()
            if leaving:
                color[node] = BLACK
                continue
            if color[node] == GRAY:
                return True      # back-edge → cycle
            if color[node] == BLACK:
                continue
            color[node] = GRAY
            stack.append((node, True))
            for neighbour in graph.get(node, []):
                if neighbour in color:
                    if color[neighbour] == GRAY:
                        return True
                    if color[neighbour] == WHITE:
                        stack.append((neighbour, False))
        return False

    return any(dfs(t.id) for t in tasks if color[t.id] == WHITE)


def _validate_tasks(raw_list: list[Any]) -> tuple[list[Task], list[str]]:
    """
    Parse raw dicts into Task objects with full validation.

    Returns (valid_tasks, warning_messages).
    Warnings are non-fatal — the pipeline continues with whatever tasks are valid.
    """
    warnings: list[str] = []
    tasks:    list[Task] = []
    seen_ids: set[str]   = set()

    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            warnings.append(f"task[{i}] is not a dict — skipped")
            continue

        # Required string fields
        missing = [f for f in ("id", "agent", "title", "description") if not item.get(f)]
        if missing:
            warnings.append(f"task[{i}] missing required fields {missing} — skipped")
            continue

        tid   = str(item["id"]).strip()
        agent = str(item["agent"]).strip().lower()

        if agent not in VALID_AGENTS:
            warnings.append(f"task '{tid}': unknown agent '{agent}' — skipped")
            continue

        if tid in seen_ids:
            warnings.append(f"duplicate task ID '{tid}' — skipped")
            continue

        seen_ids.add(tid)

        # Enforce canonical priority — overrides any LLM drift
        priority = AGENT_PRIORITY[agent]

        tasks.append(Task(
            id          = tid,
            agent       = agent,
            title       = str(item.get("title",       "")),
            description = str(item.get("description", "")),
            depends_on  = [str(d) for d in item.get("depends_on", [])],
            priority    = priority,
            status      = "pending",
        ))

    # ── Pass 2: validate depends_on references ───────────────────────────────────
    valid_ids = {t.id for t in tasks}
    for task in tasks:
        bad = [d for d in task.depends_on if d not in valid_ids]
        if bad:
            warnings.append(
                f"task '{task.id}': depends_on references unknown IDs {bad} — removed"
            )
            task.depends_on = [d for d in task.depends_on if d in valid_ids]

    # ── Pass 3: frontend must not depend on backend implementation ───────────────
    backend_ids = {t.id for t in tasks if t.agent == "backend"}
    for task in tasks:
        if task.agent != "frontend":
            continue
        be_deps = [d for d in task.depends_on if d in backend_ids]
        if be_deps:
            warnings.append(
                f"task '{task.id}' (frontend) depends on backend task(s) {be_deps} — removed"
            )
            task.depends_on = [d for d in task.depends_on if d not in backend_ids]

    # ── Pass 4: cycle detection ──────────────────────────────────────────────────
    if _has_cycle(tasks):
        warnings.append(
            "CYCLE DETECTED in dependency graph — clearing all depends_on to recover"
        )
        logger.error("[PlannerAgent] circular dependency detected — clearing all depends_on")
        for task in tasks:
            task.depends_on = []

    return tasks, warnings


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PlannerAgent                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class PlannerAgent(BaseAgent):
    """
    Reads ProjectState.architecture and produces a fully validated, dependency-ordered
    TaskList written into ProjectState.tasks.

    The task graph guarantees:
      - No cycles
      - Frontend tasks never depend on backend implementation
      - Priorities match the canonical execution order
      - Every task ID is unique
    """

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    # ── Prompt builder ───────────────────────────────────────────────────────────
    def _build_prompt(self, state: ProjectState) -> str:
        arch = state.architecture
        return USER_PROMPT_TEMPLATE.format(
            tech_stack               = _fmt_tech_stack(arch.tech_stack),
            system_components        = _fmt_components(arch.system_components),
            api_contracts            = _fmt_api_contracts(arch.api_contracts),
            database_schema_overview = arch.database_schema_overview or "(none provided)",
            data_flow                = arch.data_flow                 or "(none provided)",
            deployment_strategy      = arch.deployment_strategy       or "(none provided)",
        )

    # ── Response parser ──────────────────────────────────────────────────────────
    def _parse_response(self, state: ProjectState, raw: str) -> ProjectState:
        try:
            from agents.schemas.planner import TaskSchema
            raw_list: list[Any] = self.extract_json_and_validate(raw, TaskSchema, is_list=True)
        except Exception as exc:
            logger.error("[PlannerAgent] JSON parse/validation failed: %s", exc)
            state.log_error(self.name, f"JSON parse/validation error: {exc}")
            return state

        tasks, warnings = _validate_tasks(raw_list)

        # Surface all warnings as state errors for observability (non-fatal)
        for w in warnings:
            logger.warning("[PlannerAgent] %s", w)
            state.log_error(self.name, w)

        if not tasks:
            msg = "No valid tasks produced — check LLM response."
            logger.error("[PlannerAgent] %s", msg)
            state.log_error(self.name, msg)
            return state

        state.tasks = TaskList(tasks=tasks)

        # ── Build CodeManifest ───────────────────────────────────────────────────
        # Index raw dicts by task_id so we can grab the manifest fields.
        raw_by_id: dict[str, dict] = {}
        for item in raw_list:
            if isinstance(item, dict) and item.get("id"):
                raw_by_id[str(item["id"]).strip()] = item

        manifest_files: dict[str, FileManifest] = {}

        for task in tasks:
            raw_item  = raw_by_id.get(task.id, {})
            file_path = str(raw_item.get("file", "")).strip()

            if not file_path:
                logger.warning(
                    "[PlannerAgent] task '%s' has no 'file' field — skipping manifest entry",
                    task.id,
                )
                state.log_error(
                    self.name,
                    f"task '{task.id}' missing 'file' field — not added to CodeManifest",
                )
                continue

            if file_path in manifest_files:
                logger.warning(
                    "[PlannerAgent] duplicate file path '%s' (task '%s') — skipping",
                    file_path, task.id,
                )
                state.log_error(
                    self.name,
                    f"duplicate file path '{file_path}' in task '{task.id}' — skipped",
                )
                continue

            manifest_files[file_path] = FileManifest(
                agent    = task.agent,
                task_id  = task.id,
                filepath = file_path,
                purpose  = str(raw_item.get("description", ""))[:120],
                exports  = [str(e) for e in raw_item.get("exports", [])],
            )

        state.code_manifest = CodeManifest(files=manifest_files)

        # ── Summary log ─────────────────────────────────────────────────────────
        counts = {a: len(state.tasks.by_agent(a)) for a in VALID_AGENTS}
        logger.info(
            "[PlannerAgent] %d tasks generated — db:%d  be:%d  fe:%d  do:%d  |  "
            "CodeManifest: %d file entries (%d tasks had no file field)",
            len(tasks),
            counts["database"], counts["backend"],
            counts["frontend"], counts["devops"],
            len(manifest_files),
            len(tasks) - len(manifest_files),
        )
        return state



# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Quick smoke-test  (run:  python -m agents.planner)                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PlannerAgent — local smoke-test (no LLM call)")
    print("=" * 60)

    from core.state import Task, FileManifest, CodeManifest

    def _make_task(tid, agent="backend", deps=None):
        return Task(id=tid, agent=agent, title=tid, description="",
                    depends_on=deps or [], priority=2)

    # ── _has_cycle ──────────────────────────────────────────────────────────────
    assert not _has_cycle([
        _make_task("a"),
        _make_task("b", deps=["a"]),
        _make_task("c", deps=["b"]),
    ]), "FAIL: false positive cycle"
    print("[PASS] _has_cycle — linear chain: no cycle")

    t1 = _make_task("x")
    t2 = _make_task("y", deps=["x"])
    t3 = _make_task("z", deps=["y"])
    t1.depends_on = ["z"]          # x→y→z→x
    assert _has_cycle([t1, t2, t3]), "FAIL: cycle not detected"
    print("[PASS] _has_cycle — circular deps: cycle detected")

    # ── _validate_tasks ─────────────────────────────────────────────────────────
    raw = [
        {"id": "db-001", "agent": "database", "title": "Users table",
         "description": "Create users migration.", "depends_on": [], "status": "pending"},
        {"id": "be-001", "agent": "backend",  "title": "Auth routes",
         "description": "POST /auth/login.",       "depends_on": ["db-001"], "status": "pending"},
        {"id": "fe-001", "agent": "frontend", "title": "Login page",
         "description": "React login form.",       "depends_on": ["be-001"], "status": "pending"},
        # duplicate id — must be dropped
        {"id": "be-001", "agent": "backend",  "title": "Dup",
         "description": "Should be skipped.",      "depends_on": [], "status": "pending"},
        # invalid agent — must be dropped
        {"id": "xx-001", "agent": "wizard",   "title": "Bad agent",
         "description": "Should be skipped.",      "depends_on": [], "status": "pending"},
    ]
    tasks, warnings = _validate_tasks(raw)
    assert len(tasks) == 3,    f"FAIL: expected 3 tasks, got {len(tasks)}"
    assert len(warnings) >= 2, f"FAIL: expected >=2 warnings, got {len(warnings)}"
    fe = next(t for t in tasks if t.agent == "frontend")
    assert "be-001" not in fe.depends_on, "FAIL: frontend still depends on backend impl"
    print("[PASS] _validate_tasks — 3 tasks, FE->BE dep stripped, dup/bad-agent dropped")
    print("       Warnings:", [w[:70] for w in warnings])

    # ── Priority enforcement ────────────────────────────────────────────────────
    for t in tasks:
        assert t.priority == AGENT_PRIORITY[t.agent], \
            f"FAIL: {t.id} priority={t.priority}, expected {AGENT_PRIORITY[t.agent]}"
    print("[PASS] Priority enforcement — all tasks match AGENT_PRIORITY")

    # ── FileManifest / CodeManifest ─────────────────────────────────────────────
    fm = FileManifest(
        agent    = "backend",
        task_id  = "be-001",
        filepath = "backend/app/routes/auth.py",
        purpose  = "JWT auth routes",
        exports  = ["router", "login"],
    )
    assert fm.written is False
    assert fm.exports == ["router", "login"]
    assert fm.classes == [] and fm.functions == []
    print("[PASS] FileManifest — planner fields set, coding-agent fields default empty")

    cm = CodeManifest(files={fm.filepath: fm})
    cm.mark_written("backend/app/routes/auth.py")
    assert cm.files["backend/app/routes/auth.py"].written is True
    assert cm.files["backend/app/routes/auth.py"].exports == ["router", "login"], \
        "FAIL: planner exports must not be overwritten by mark_written"
    print("[PASS] CodeManifest.mark_written — written=True, planner exports preserved")

    print("\n" + "=" * 60)
    print("All PlannerAgent smoke-tests PASSED")
    print("=" * 60 + "\n")
