"""
agents/planner.py
─────────────────
Planner Agent — parallel 3-call execution, minimal prompts, streaming enabled.

SINGLE ROUND — 3 calls in parallel:
  ├── Call A: database  (needs: db_schema only)
  ├── Call B: backend   (needs: backend tech + api_contracts + table names)
  └── Call C: frontend  (needs: frontend tech + api_contracts)

VALIDATION PASS — deterministic, no LLM
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from typing import Any

from .base import BaseAgent
from core.state import (
    ProjectState, Task, TaskList, FileManifest, CodeManifest, APIContract,
)

logger = logging.getLogger(__name__)
MAX_PARALLEL_CALLS = 3

VALID_AGENTS: frozenset[str] = frozenset({"database", "backend", "frontend", "devops"})
AGENT_PRIORITY: dict[str, int] = {"database": 1, "backend": 2, "frontend": 2, "devops": 3}
AGENT_PREFIX: dict[str, str] = {"database": "db", "backend": "be", "frontend": "fe", "devops": "do"}

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Per-call system prompts                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_BASE_SYSTEM = """\
You are a senior engineering project manager. Break architecture into granular tasks.
Respond with a SINGLE JSON array — no prose outside it.
Every element:
{"id":"<ID>","agent":"<AGENT_VALUE>","title":"<short>","description":"<detailed: file path, \
class/function, contracts, acceptance criteria>","depends_on":["<id>",...],"priority":<PRIORITY>,\
"status":"pending","file":"<repo-relative path>","exports":["<name>"],"classes":["<name>"],\
"functions":["<name>"]}
Rules: file — unique non-empty path. exports — only cross-module names. \
classes/functions — suggestions, coding agent may add helpers.
"""

DATABASE_SYSTEM = _BASE_SYSTEM.replace("<AGENT_VALUE>", "database").replace("<PRIORITY>", "1") + """
## Scope: DATABASE tasks ONLY
- One task per table/migration. IDs: db-001, db-002, ...
- depends_on: [] always (root tasks). Agent: "database".
- Do NOT produce backend/frontend/devops tasks.
"""

BACKEND_SYSTEM = _BASE_SYSTEM.replace("<AGENT_VALUE>", "backend").replace("<PRIORITY>", "2") + """
## Scope: BACKEND tasks ONLY
- One task per file (routes, services, models, schemas, config).
- IDs: be-001, be-002, ... Agent: "backend".
- depends_on: reference "db-001", "db-002" etc. based on table order in schema.
- Do NOT produce database/frontend/devops tasks.
"""

FRONTEND_SYSTEM = _BASE_SYSTEM.replace("<AGENT_VALUE>", "frontend").replace("<PRIORITY>", "2") + """
## Scope: FRONTEND tasks ONLY
- One task per file (components, pages, hooks, API client, routing, styles).
- IDs: fe-001, fe-002, ... Agent: "frontend".
- depends_on: [] for most. NEVER reference be-* IDs.
- Do NOT produce database/backend/devops tasks.
"""

# ── Slim user prompt templates ───────────────────────────────────────────────────

DATABASE_USER_TEMPLATE = """\
### Database Schema Overview
{database_schema_overview}

---
Generate ALL database tasks. One per table/migration. JSON array only:
```json
"""

BACKEND_USER_TEMPLATE = """\
### Backend Technology
{backend_tech}

### API Contracts
{api_contracts}

### Database Tables (for depends_on prediction)
{table_names}

---
Generate ALL backend tasks. One per file. Reference db-* IDs in depends_on.
JSON array only:
```json
"""

FRONTEND_USER_TEMPLATE = """\
### Frontend Technology
{frontend_tech}

### API Contracts
{api_contracts}

---
Generate ALL frontend tasks. One per file. Do NOT depend on backend tasks.
JSON array only:
```json
"""


# ── Formatting helpers ───────────────────────────────────────────────────────────

def _fmt_api_contracts(contracts: list[APIContract]) -> str:
    if not contracts:
        return "  (none)"
    lines = []
    for c in contracts:
        lock = "🔒" if c.auth_required else "🔓"
        lines.append(
            f"  {lock} {c.method:7} {c.path}\n"
            f"         desc         : {c.description}\n"
            f"         request_body : {c.request_body}\n"
            f"         response_body: {c.response_body}"
        )
    return "\n".join(lines)


def _extract_table_names(schema_overview: str) -> str:
    """Extract just table names from db schema overview for backend depends_on prediction."""
    if not schema_overview:
        return "(none)"
    lines = []
    counter = 0
    for line in schema_overview.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Take just the table name (first word before parenthesis or space)
        table = line.split("(")[0].split(" ")[0].strip()
        if table:
            counter += 1
            lines.append(f"  db-{counter:03d}: {table}")
    return "\n".join(lines) or "(none)"


# ── Validation helpers ───────────────────────────────────────────────────────────

def _has_cycle(tasks: list[Task]) -> bool:
    """Detect cycles via iterative DFS (paint algorithm)."""
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
                return True
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
    """Parse raw dicts into Task objects with full validation."""
    warnings: list[str] = []
    tasks:    list[Task] = []
    seen_ids: set[str]   = set()

    for i, item in enumerate(raw_list):
        if not isinstance(item, dict):
            warnings.append(f"task[{i}] is not a dict — skipped")
            continue
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
        tasks.append(Task(
            id=tid, agent=agent,
            title=str(item.get("title", "")),
            description=str(item.get("description", "")),
            depends_on=[str(d) for d in item.get("depends_on", [])],
            priority=AGENT_PRIORITY[agent], status="pending",
        ))

    # Pass 2: validate depends_on references
    valid_ids = {t.id for t in tasks}
    for task in tasks:
        bad = [d for d in task.depends_on if d not in valid_ids]
        if bad:
            warnings.append(f"task '{task.id}': depends_on references unknown IDs {bad} — removed")
            task.depends_on = [d for d in task.depends_on if d in valid_ids]

    # Pass 3: frontend must not depend on backend
    backend_ids = {t.id for t in tasks if t.agent == "backend"}
    for task in tasks:
        if task.agent != "frontend":
            continue
        be_deps = [d for d in task.depends_on if d in backend_ids]
        if be_deps:
            warnings.append(f"task '{task.id}' (frontend) depends on backend task(s) {be_deps} — removed")
            task.depends_on = [d for d in task.depends_on if d not in backend_ids]

    # Pass 4: cycle detection
    if _has_cycle(tasks):
        warnings.append("CYCLE DETECTED — clearing all depends_on to recover")
        logger.error("[PlannerAgent] circular dependency detected — clearing all depends_on")
        for task in tasks:
            task.depends_on = []

    return tasks, warnings


def _build_manifest(tasks: list[Task], raw_list: list[Any]) -> dict[str, FileManifest]:
    """Build CodeManifest entries from validated tasks + raw LLM dicts."""
    raw_by_id: dict[str, dict] = {}
    for item in raw_list:
        if isinstance(item, dict) and item.get("id"):
            raw_by_id[str(item["id"]).strip()] = item

    manifest_files: dict[str, FileManifest] = {}
    for task in tasks:
        raw_item  = raw_by_id.get(task.id, {})
        file_path = str(raw_item.get("file", "")).strip().lstrip("/").lstrip("\\")
        if not file_path:
            logger.warning("[PlannerAgent] task '%s' has no 'file' field — skipping manifest", task.id)
            continue
        if not file_path.startswith("output/") and not file_path.startswith("output\\"):
            file_path = f"output/{file_path}"
        if file_path in manifest_files:
            logger.warning("[PlannerAgent] duplicate file path '%s' (task '%s') — skipping", file_path, task.id)
            continue
        manifest_files[file_path] = FileManifest(
            agent=task.agent, task_id=task.id, filepath=file_path,
            purpose=str(raw_item.get("description", ""))[:120],
            exports=[str(e) for e in raw_item.get("exports", [])],
        )
    return manifest_files


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  PlannerAgent                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class PlannerAgent(BaseAgent):
    """
    3 parallel LLM calls (database, backend, frontend) with minimal prompts
    and streaming enabled, then deterministic merge + validation.
    """

    @property
    def system_prompt(self) -> str:
        return _BASE_SYSTEM

    def _build_prompt(self, state: ProjectState) -> str:
        raise NotImplementedError("_build_prompt unused; run() is overridden.")

    def _parse_response(self, state: ProjectState, raw: str) -> ProjectState:
        raise NotImplementedError("_parse_response unused; run() is overridden.")

    # ── LLM helper with custom system prompt + streaming ─────────────────────
    def _call_with_system(self, system: str, user: str, agent_name: str = "") -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm_call(
                    prompt=user,
                    system_prompt=system,
                    agent_name=agent_name or self.name,
                )
                return response.text
            except Exception as exc:
                logger.warning("[PlannerAgent] LLM attempt %d/%d failed: %s",
                               attempt, self.max_retries, exc)
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff * attempt)
        raise RuntimeError("Unreachable")

    # ── Sub-call workers ─────────────────────────────────────────────────────

    def _call_database(self, db_schema: str,
                       results: dict, lock: Lock, state: ProjectState) -> None:
        """Call A: only needs database_schema_overview."""
        prompt = DATABASE_USER_TEMPLATE.format(database_schema_overview=db_schema)
        try:
            from agents.schemas.planner import TaskSchema
            raw = self._call_with_system(DATABASE_SYSTEM, prompt, "database")
            data = self.extract_json_and_validate(raw, TaskSchema, is_list=True)
            with lock:
                results["database"] = data
                state.metrics.total_llm_calls += 1
            logger.info("[PlannerAgent] ✓ database tasks complete (%d items)", len(data))
        except Exception as exc:
            logger.error("[PlannerAgent] database sub-call failed: %s", exc)
            with lock:
                state.log_error("PlannerAgent", f"database sub-call failed: {exc}")

    def _call_backend(self, backend_tech: str, api_contracts_str: str, table_names: str,
                      results: dict, lock: Lock, state: ProjectState) -> None:
        """Call B: needs backend tech key + api_contracts + table names."""
        prompt = BACKEND_USER_TEMPLATE.format(
            backend_tech=backend_tech,
            api_contracts=api_contracts_str,
            table_names=table_names,
        )
        try:
            from agents.schemas.planner import TaskSchema
            raw = self._call_with_system(BACKEND_SYSTEM, prompt, "backend")
            data = self.extract_json_and_validate(raw, TaskSchema, is_list=True)
            with lock:
                results["backend"] = data
                state.metrics.total_llm_calls += 1
            logger.info("[PlannerAgent] ✓ backend tasks complete (%d items)", len(data))
        except Exception as exc:
            logger.error("[PlannerAgent] backend sub-call failed: %s", exc)
            with lock:
                state.log_error("PlannerAgent", f"backend sub-call failed: {exc}")

    def _call_frontend(self, frontend_tech: str, api_contracts_str: str,
                       results: dict, lock: Lock, state: ProjectState) -> None:
        """Call C: needs frontend tech key + api_contracts only."""
        prompt = FRONTEND_USER_TEMPLATE.format(
            frontend_tech=frontend_tech,
            api_contracts=api_contracts_str,
        )
        try:
            from agents.schemas.planner import TaskSchema
            raw = self._call_with_system(FRONTEND_SYSTEM, prompt, "frontend")
            data = self.extract_json_and_validate(raw, TaskSchema, is_list=True)
            with lock:
                results["frontend"] = data
                state.metrics.total_llm_calls += 1
            logger.info("[PlannerAgent] ✓ frontend tasks complete (%d items)", len(data))
        except Exception as exc:
            logger.error("[PlannerAgent] frontend sub-call failed: %s", exc)
            with lock:
                state.log_error("PlannerAgent", f"frontend sub-call failed: {exc}")

    # ── Main execution ───────────────────────────────────────────────────────
    def run(self, state: dict | ProjectState) -> dict[str, Any]:  # type: ignore[override]
        ps = state if isinstance(state, ProjectState) else ProjectState(**state)
        before = ps.model_copy(deep=True)
        ps.mark_agent_start(self.name)
        t0 = time.perf_counter()
        arch = ps.architecture

        # Pre-extract only what each call needs
        db_schema       = arch.database_schema_overview or "(none provided)"
        backend_tech    = arch.tech_stack.get("backend", "FastAPI (Python 3.12)")
        frontend_tech   = arch.tech_stack.get("frontend", "React 18 + TypeScript")
        api_contracts   = _fmt_api_contracts(arch.api_contracts)
        table_names     = _extract_table_names(arch.database_schema_overview)

        results: dict[str, list[Any]] = {}
        state_lock = Lock()

        # ── SINGLE ROUND: 3 parallel calls ──────────────────────────────────
        logger.info("[PlannerAgent] ── Launching 3 parallel sub-calls (streaming) ──")
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
            futures: list[Future] = [
                pool.submit(self._call_database, db_schema, results, state_lock, ps),
                pool.submit(self._call_backend, backend_tech, api_contracts, table_names,
                            results, state_lock, ps),
                pool.submit(self._call_frontend, frontend_tech, api_contracts,
                            results, state_lock, ps),
            ]
            for f in futures:
                f.result()

        # ── Merge ────────────────────────────────────────────────────────────
        merged_raw: list[Any] = []
        for agent_key in ("database", "backend", "frontend"):
            merged_raw.extend(results.get(agent_key, []))
        logger.info("[PlannerAgent] Merged %d raw tasks from 3 sub-calls", len(merged_raw))

        # ── VALIDATION PASS ──────────────────────────────────────────────────
        tasks, warnings = _validate_tasks(merged_raw)
        for w in warnings:
            logger.warning("[PlannerAgent] %s", w)
            ps.log_error(self.name, w)

        if not tasks:
            msg = "No valid tasks produced — check LLM responses."
            logger.error("[PlannerAgent] %s", msg)
            ps.log_error(self.name, msg)
            ps.metrics.agent_durations[self.name] = round(time.perf_counter() - t0, 3)
            return self._compute_diff(before, ps)

        ps.tasks = TaskList(tasks=tasks)
        manifest_files = _build_manifest(tasks, merged_raw)
        ps.code_manifest = CodeManifest(files=manifest_files)

        duration = time.perf_counter() - t0
        ps.metrics.agent_durations[self.name] = round(duration, 3)
        counts = {a: len(ps.tasks.by_agent(a)) for a in VALID_AGENTS}
        logger.info(
            "[PlannerAgent] %d tasks in %.2fs — db:%d be:%d fe:%d do:%d | manifest:%d",
            len(tasks), duration,
            counts["database"], counts["backend"], counts["frontend"], counts["devops"],
            len(manifest_files),
        )
        return self._compute_diff(before, ps)


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
        _make_task("a"), _make_task("b", deps=["a"]), _make_task("c", deps=["b"]),
    ]), "FAIL: false positive cycle"
    print("[PASS] _has_cycle — linear chain: no cycle")

    t1, t2, t3 = _make_task("x"), _make_task("y", deps=["x"]), _make_task("z", deps=["y"])
    t1.depends_on = ["z"]
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
        {"id": "be-001", "agent": "backend",  "title": "Dup",
         "description": "Should be skipped.",      "depends_on": [], "status": "pending"},
        {"id": "xx-001", "agent": "wizard",   "title": "Bad agent",
         "description": "Should be skipped.",      "depends_on": [], "status": "pending"},
    ]
    tasks, warnings = _validate_tasks(raw)
    assert len(tasks) == 3
    assert len(warnings) >= 2
    fe = next(t for t in tasks if t.agent == "frontend")
    assert "be-001" not in fe.depends_on
    print("[PASS] _validate_tasks — 3 tasks, FE->BE dep stripped, dup/bad-agent dropped")

    for t in tasks:
        assert t.priority == AGENT_PRIORITY[t.agent]
    print("[PASS] Priority enforcement — all tasks match AGENT_PRIORITY")

    # ── _extract_table_names ────────────────────────────────────────────────────
    schema = "users (id UUID PK)\n\nworkspaces (id UUID PK)\n\ntasks (id UUID PK)"
    result = _extract_table_names(schema)
    assert "db-001: users" in result
    assert "db-002: workspaces" in result
    assert "db-003: tasks" in result
    print("[PASS] _extract_table_names — 3 tables extracted with db-NNN IDs")

    # ── _build_manifest ────────────────────────────────────────────────────────
    test_tasks = [
        Task(id="db-001", agent="database", title="T1", description="", depends_on=[], priority=1),
        Task(id="be-001", agent="backend", title="T2", description="", depends_on=["db-001"], priority=2),
    ]
    test_raw = [
        {"id": "db-001", "file": "database/migrations/001_users.sql", "exports": [], "description": "Users table"},
        {"id": "be-001", "file": "backend/app/models/user.py", "exports": ["User"], "description": "User model"},
    ]
    mf = _build_manifest(test_tasks, test_raw)
    assert len(mf) == 2
    assert mf["backend/app/models/user.py"].exports == ["User"]
    print("[PASS] _build_manifest — 2 entries with correct exports")

    # ── FileManifest / CodeManifest ─────────────────────────────────────────────
    fm = FileManifest(agent="backend", task_id="be-001", filepath="backend/app/routes/auth.py",
                      purpose="JWT auth routes", exports=["router", "login"])
    assert fm.written is False and fm.exports == ["router", "login"]
    cm = CodeManifest(files={fm.filepath: fm})
    cm.mark_written("backend/app/routes/auth.py")
    assert cm.files["backend/app/routes/auth.py"].written is True
    assert cm.files["backend/app/routes/auth.py"].exports == ["router", "login"]
    print("[PASS] FileManifest/CodeManifest — planner exports preserved after mark_written")

    print("\n" + "=" * 60)
    print("All PlannerAgent smoke-tests PASSED")
    print("=" * 60 + "\n")
