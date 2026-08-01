"""
agents/db.py
────────────
Database Agent — runs after the Planner Agent.

Input:  ProjectState.tasks          (TaskList)
        ProjectState.code_manifest  (CodeManifest)
        ProjectState.architecture   (ArchData — tech stack + schema overview)
Output: ProjectState.generated_code.database (dict path -> code)
        ProjectState.code_manifest  (marked written)

Execution model:
  - Tasks are sorted by priority (lower number = higher priority).
  - Tasks sharing the same priority level run in parallel via ThreadPoolExecutor.
  - A single Lock guards all writes to ProjectState.
  - Each task produces one migration file + a <manifest> block.

Rules enforced (see system prompt):
  1.  Table/column names match Architecture schema exactly (snake_case).
  2.  FKs only reference tables in the same migration set; dependency order via priority.
  3.  PK type is UUID unless Architecture specifies otherwise.
  4.  Every table gets created_at / updated_at unless the task says otherwise.
  5.  No invented tables or columns — flag suggestions in manifest summary.
  6.  All nullable/non-nullable constraints must be explicit.
  7.  No DROP TABLE or destructive DDL unless task explicitly requests rollback.
  8.  Index on every FK column + columns flagged as frequently queried in data_flow.
  9.  No hardcoded environment values (DB name, host, credentials).
  10. Each migration is self-contained given its declared depends_on.
  11. Manifest export lists exact table name, column names, and types.
"""

import os
import sys
import json
import json as _json
import logging
import time
import re
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
from typing import Any

_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

try:
    from .base import BaseAgent, filter_manifest_by_dependencies, is_trivial_file, TRIVIAL_BATCH_SIZE
except ImportError:
    from base import BaseAgent, filter_manifest_by_dependencies, is_trivial_file, TRIVIAL_BATCH_SIZE

from core.state import (
    ProjectState,
    Task,
    TaskList,
    FileManifest,
    CodeManifest,
    Argument,
    ClassName,
    FunctionName,
)

logger = logging.getLogger(__name__)

MAX_PARALLEL_CALLS = 3
OUTPUT_DIR = "output"


# ── System prompt ──────────────────────────────────────────────────────────────

def _build_system_prompt(db_tech: str) -> str:
    return f"""\
You are an expert database engineer specialising in {db_tech}.
Generate a complete, runnable database migration file AND a structured manifest block.

## Strict Rules
1.  Table and column names MUST match the Architecture database_schema_overview EXACTLY
    (snake_case throughout, no renaming, no pluralisation changes, no casing changes).
2.  Every foreign key must reference a table that exists in the same migration set.
    Never reference a table not defined in the manifest. Respect priority/depends_on ordering.
3.  Primary keys use UUID type by default unless Architecture explicitly specifies otherwise.
    This must be consistent across ALL tables so backend models match everywhere.
4.  Every table must include `created_at` and `updated_at` TIMESTAMP columns
    unless the task explicitly says otherwise.
5.  Never invent additional tables or columns not present in the task or schema overview.
    Any suggested additions must appear in the manifest `suggested_additions` field only —
    never silently added to the migration.
6.  All nullable vs non-nullable constraints must be explicit (NOT NULL or NULL keyword).
    Never leave a column's nullability ambiguous.
7.  Never write DROP TABLE or any destructive operation unless the task explicitly requests
    a rollback/down migration.
8.  Include an index on every foreign key column and on any column mentioned in
    Architecture's data_flow as frequently queried.
9.  Never hardcode environment-specific values (database name, host, credentials).
    Migrations must be portable across environments.
10. Each migration file must be self-contained and runnable in isolation given its
    declared dependencies — no implicit ordering assumptions beyond depends_on.
11. The manifest export section must list the exact table name and exact column names
    + types — this is what the Backend model-writing step reads to build SQLAlchemy models.

Return JSON: {{"migration_code":"<full script>","manifest":{{"table_name":"<tbl>","columns":[{{"name":"<col>","type":"<type>","nullable":true|false,"primary_key":true|false,"foreign_key":"<table.col or null>"}}],"indexes":["<col>"],"exports":["<tbl>"],"suggested_additions":[]}},"functions":[{{"name":"<func>","arguments":[{{"name":"<p>","type":"<t>","default":""}}],"return_type":"<t>","docstring":"<desc>"}}]}}
No prose outside the JSON object.
"""


# ── Priority-based stage resolver ──────────────────────────────────────────────

def resolve_db_stages(tasks: list[Task]) -> list[list[Task]]:
    """
    Groups DB tasks into execution stages based on priority then depends_on.

    Algorithm:
      1. Sort all tasks by priority ascending (lower = more urgent / foundational).
      2. Walk priority levels in order; within each level, apply topological
         ordering so that tasks whose depends_on are satisfied run first.
      3. Tasks at the same priority with no cross-dependencies form one stage
         and execute in parallel.

    Returns a list of stages; each stage is a list of Task objects safe to run
    concurrently (they share no intra-stage dependency).
    """
    if not tasks:
        return []

    task_map = {t.id: t for t in tasks}
    db_ids = set(task_map.keys())

    # Group by priority
    priority_groups: dict[int, list[Task]] = {}
    for t in tasks:
        priority_groups.setdefault(t.priority, []).append(t)

    stages: list[list[Task]] = []
    completed: set[str] = set()

    for priority in sorted(priority_groups.keys()):
        group = priority_groups[priority]
        remaining = list(group)

        # Within the priority group, topologically sort by depends_on
        while remaining:
            ready: list[Task] = []
            for task in remaining:
                db_deps = [d for d in task.depends_on if d in db_ids]
                if all(d in completed for d in db_deps):
                    ready.append(task)

            if not ready:
                # Cycle or cross-priority deps not yet resolved — bundle remainder
                logger.warning(
                    "[DBAgent] Possible cycle or unresolved deps at priority %d; "
                    "bundling %d tasks together.", priority, len(remaining)
                )
                stages.append(remaining)
                for t in remaining:
                    completed.add(t.id)
                break

            stages.append(ready)
            for t in ready:
                completed.add(t.id)
                remaining.remove(t)

    return stages


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _fmt_manifest_context(manifest: CodeManifest, skip_path: str, task: Task | None = None) -> str:
    dep_files = filter_manifest_by_dependencies(manifest, skip_path, task)
    entries = []
    for path, fm in dep_files.items():
        status = "written" if fm.written else "pending"
        exports = ", ".join(fm.exports) or "(none)"
        entries.append(f"- `{path}` ({status})\n  Purpose: {fm.purpose}\n  Exports: {exports}")
    return "\n\n".join(entries) or "No relevant dependency files in manifest."


def _build_file_prompt(
    filepath: str,
    task: Task,
    manifest: CodeManifest,
    arch: Any,
    db_tech: str,
) -> str:
    fm = manifest.files.get(filepath)
    purpose = fm.purpose if fm else task.description
    exports = ", ".join(fm.exports) if fm and fm.exports else "(none)"

    tech_table = (
        "\n".join(f"  {k}: {v}" for k, v in arch.tech_stack.items())
        if arch.tech_stack else "  (none)"
    )

    # Already-written DB files for FK context
    written_tables = []
    for path, m in manifest.files.items():
        if m.written and m.agent == "database":
            written_tables.append(f"- `{path}` exports: {', '.join(m.exports) or '(none)'}")
    written_str = "\n".join(written_tables) or "None yet."

    manifest_ctx = _fmt_manifest_context(manifest, filepath, task)

    depends_str = ", ".join(task.depends_on) if task.depends_on else "(none)"

    return f"""\
## Migration File to Generate
- **File Path** : `{filepath}`
- **Purpose**   : {purpose}
- **Exports**   : {exports}
- **Depends On**: {depends_str}
- **Priority**  : {task.priority}

## Tech Stack
{tech_table}

## Architecture — Database Schema Overview
{getattr(arch, 'database_schema_overview', 'No schema overview provided.')}

## Already Written Database Files (available for FK references)
{written_str}

## Project Code Manifest (Dependencies)
{manifest_ctx}

---

Generate the complete migration for `{filepath}`.
Follow ALL 11 rules in the system prompt.
Return a single JSON object with keys: migration_code, manifest, functions.
```json
"""


# ── Disk writer ────────────────────────────────────────────────────────────────

def _write_to_disk(filepath: str, code: str) -> None:
    rel = filepath.lstrip("/").lstrip("\\")
    if not rel.startswith(OUTPUT_DIR):
        rel = os.path.join(OUTPUT_DIR, rel)
    abs_path = os.path.join(_root_dir, rel)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as fh:
        fh.write(code)


# ── Manifest XML serialiser ────────────────────────────────────────────────────

def _render_manifest_xml(manifest_data: dict) -> str:
    """Serialise the manifest dict returned by the LLM into the canonical <manifest> block."""
    return f"<manifest>\n{json.dumps(manifest_data, indent=2)}\n</manifest>"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DatabaseAgent                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class DatabaseAgent(BaseAgent):
    """
    Priority-staged, parallel database migration generator.

    Execution order:
      - Tasks are sorted by priority (ascending).
      - Tasks sharing the same priority form a stage and run in parallel.
      - Within a priority group, depends_on is still respected via topological sort.
      - A threading Lock protects all writes to ProjectState.
    """

    # ── BaseAgent interface ───────────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        return "You are an expert database engineer."

    def _build_prompt(self, state: ProjectState) -> str:
        raise NotImplementedError("_build_prompt unused; run() is overridden.")

    def _parse_response(self, state: ProjectState, raw: str) -> ProjectState:
        raise NotImplementedError("_parse_response unused; run() is overridden.")

    # ── LLM wrapper ───────────────────────────────────────────────────────────

    def _call_with_system(self, system: str, user: str, agent_name: str = "database") -> str:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.llm_call(
                    prompt=user,
                    system_prompt=system,
                    agent_name=agent_name,
                )
                return resp.text
            except Exception as exc:
                logger.warning(
                    "[DBAgent] LLM attempt %d/%d failed: %s",
                    attempt, self.max_retries, exc,
                )
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff * attempt)
        raise RuntimeError("Unreachable")

    # ── Trivial batch worker ──────────────────────────────────────────────────

    def _build_batch_prompt(self, tasks: list[Task], manifest: Any, arch: Any, db_tech: str) -> str:
        """Compact batch prompt for 2-3 trivial DB migration files — returns a JSON array."""
        entries: list[str] = []
        for task in tasks:
            fm = manifest.by_task(task.id)
            if not fm:
                continue
            exports = ", ".join(fm.exports) or "(none)"
            entries.append(
                f"### File: {fm.filepath}\n"
                f"Purpose: {fm.purpose}\n"
                f"Required Exports: {exports}"
            )
        batch_desc = "\n\n".join(entries)
        return (
            f"Generate the following {len(tasks)} trivial/boilerplate {db_tech} migration files.\n\n"
            f"{batch_desc}\n\n"
            "Return a JSON ARRAY (not object) — one element per file in the same order:\n"
            '[{"filepath":"<path>","migration_code":"<full SQL/migration>","manifest":{},"functions":[]}, ...]\n'
            "```json\n"
        )

    def _process_batch(self, tasks: list[Task], state: ProjectState, lock: Lock) -> None:
        """Single LLM call for a batch of trivial DB migration files."""
        with lock:
            manifest = state.code_manifest
            arch = state.architecture
            db_tech = arch.tech_stack.get("database", "PostgreSQL") if arch.tech_stack else "PostgreSQL"
            sys_prompt  = _build_system_prompt(db_tech)
            user_prompt = self._build_batch_prompt(tasks, manifest, arch, db_tech)

        filepaths = [manifest.by_task(t.id).filepath for t in tasks if manifest.by_task(t.id)]
        logger.info("[DBAgent] Batch trivial call: %s", filepaths)

        try:
            raw = self._call_with_system(sys_prompt, user_prompt, agent_name="db-batch")
        except Exception as exc:
            logger.error("[DBAgent] Batch LLM call failed: %s", exc)
            with lock:
                state.log_error(self.name, f"Batch LLM call failed: {exc}")
            return

        try:
            text = raw.strip()
            text = re.sub(r'^```[a-z]*\n?', '', text, flags=re.MULTILINE).strip().rstrip('`').strip()
            m_arr = re.search(r'\[.*\]', text, re.DOTALL)
            m_obj = re.search(r'\{.*\}', text, re.DOTALL)
            if m_arr:
                items: list[dict] = _json.loads(m_arr.group(0))
            elif m_obj:
                items = [_json.loads(m_obj.group(0))]
            else:
                items = _json.loads(text)
            if not isinstance(items, list):
                items = [items]
        except Exception as exc:
            logger.error("[DBAgent] Batch JSON parse failed: %s — raw: %s", exc, raw[:300])
            with lock:
                state.log_error(self.name, f"Batch JSON parse failed: {exc}")
            return

        for item, task in zip(items, tasks):
            with lock:
                fm = state.code_manifest.by_task(task.id)
            if not fm:
                continue
            code = str(item.get("migration_code", "")).strip()
            if not code:
                logger.warning("[DBAgent] Batch: empty migration_code for %s", fm.filepath)
                continue
            manifest_data: dict = item.get("manifest") or {}
            classes: list[ClassName] = []
            if manifest_data and "table_name" in manifest_data:
                table_name = manifest_data["table_name"]
                columns = manifest_data.get("columns", [])
                args = []
                for col in columns:
                    col_name = col.get("name", "")
                    col_type = col.get("type", "unknown")
                    default_val = ""
                    if col.get("primary_key"):
                        col_type += " (PK)"
                    if col.get("foreign_key"):
                        col_type += f" (FK -> {col['foreign_key']})"
                    if col.get("nullable"):
                        default_val = "NULL"
                    args.append(Argument(
                        name=col_name,
                        type=col_type,
                        default=default_val
                    ))
                classes.append(ClassName(
                    name=table_name,
                    arguments=args,
                    docstring=f"Table schema for {table_name}"
                ))
            try:
                _write_to_disk(fm.filepath, code)
            except Exception as exc:
                logger.error("[DBAgent] Batch disk write failed for %s: %s", fm.filepath, exc)
            with lock:
                state.generated_code.database[fm.filepath] = code
                state.code_manifest.mark_written(fm.filepath, classes, [])
                state.completed_task_ids.add(task.id)
                state.metrics.total_llm_calls += 0  # counted at call site
            logger.info("[DBAgent] ✓ batch trivial: %s", fm.filepath)

    # ── Per-task worker ───────────────────────────────────────────────────────

    def _process_task(self, task: Task, state: ProjectState, lock: Lock) -> None:
        """Worker thread: call LLM, parse response, write migration + manifest."""

        # Fetch file manifest entry under lock
        with lock:
            fm = state.code_manifest.by_task(task.id)

        if not fm:
            logger.error("[DBAgent] Task %s has no manifest entry.", task.id)
            with lock:
                state.log_error(self.name, f"Task {task.id} has no manifest entry")
            return

        filepath = fm.filepath
        logger.info("[DBAgent] Generating migration for %s (task %s) …", filepath, task.id)

        # Build prompts while holding the lock (read-only snapshot of shared state)
        with lock:
            db_tech = "PostgreSQL with SQLAlchemy (Alembic)"
            if state.architecture.tech_stack:
                db_tech = state.architecture.tech_stack.get("database", db_tech)
            sys_prompt = _build_system_prompt(db_tech)
            user_prompt = _build_file_prompt(
                filepath, task, state.code_manifest, state.architecture, db_tech
            )

        # LLM call (outside lock — blocking I/O)
        try:
            raw = self._call_with_system(sys_prompt, user_prompt, agent_name=f"db-{task.id}")
        except Exception as exc:
            logger.error("[DBAgent] LLM call failed for %s: %s", filepath, exc)
            with lock:
                state.log_error(self.name, f"LLM call failed for {filepath}: {exc}")
            return

        # Parse JSON response
        try:
            from agents.schemas.database import DatabaseSchema
            data = self.extract_json_and_validate(raw, DatabaseSchema)
        except Exception as exc:
            snippet = raw[:500] if raw else "(empty)"
            logger.error(
                "[DBAgent] JSON parse/validation failed for %s: %s | raw: %s", filepath, exc, snippet
            )
            with lock:
                state.log_error(self.name, f"JSON parse/validation failed for {filepath}: {exc}")
            return

        migration_code = data.get("migration_code", "").strip()
        if not migration_code:
            logger.error("[DBAgent] 'migration_code' missing in response for %s", filepath)
            with lock:
                state.log_error(self.name, f"'migration_code' missing for {filepath}")
            return

        manifest_data: dict = data.get("manifest") or {}

        # Parse table schema from manifest into classes list to keep it in the state
        classes: list[ClassName] = []
        if manifest_data and "table_name" in manifest_data:
            table_name = manifest_data["table_name"]
            columns = manifest_data.get("columns", [])
            args = []
            for col in columns:
                col_name = col.get("name", "")
                col_type = col.get("type", "unknown")
                default_val = ""
                if col.get("primary_key"):
                    col_type += " (PK)"
                if col.get("foreign_key"):
                    col_type += f" (FK -> {col['foreign_key']})"
                if col.get("nullable"):
                    default_val = "NULL"
                args.append(Argument(
                    name=col_name,
                    type=col_type,
                    default=default_val
                ))
            classes.append(ClassName(
                name=table_name,
                arguments=args,
                docstring=f"Table schema for {table_name}"
            ))

        # Parse optional helper function metadata
        functions: list[FunctionName] = []
        for f in data.get("functions", []):
            try:
                args = [Argument(**a) for a in f.get("arguments", [])]
                functions.append(FunctionName(
                    name=f["name"],
                    arguments=args,
                    return_type=f.get("return_type", "None"),
                    docstring=f.get("docstring", ""),
                ))
            except Exception as e:
                logger.warning("[DBAgent] Function metadata parse error: %s", e)

        # Write migration file to disk (only write code)
        try:
            _write_to_disk(filepath, migration_code)
        except Exception as exc:
            logger.error("[DBAgent] Disk write failed for %s: %s", filepath, exc)
            with lock:
                state.log_error(self.name, f"Disk write failed for {filepath}: {exc}")
            # Still update state so the manifest records the attempt

        # Thread-safe state update
        with lock:
            state.generated_code.database[filepath] = migration_code
            state.code_manifest.mark_written(filepath, classes=classes, functions=functions)
            state.completed_task_ids.add(task.id)
            state.metrics.total_tokens_used += 0  # tokens tracked in _call_with_system
            state.metrics.total_llm_calls += 1

        logger.info("[DBAgent] ✓ %s", filepath)

    # ── Main entry point ──────────────────────────────────────────────────────

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, state: dict | ProjectState) -> dict[str, Any]:  # type: ignore[override]
        ps = state if isinstance(state, ProjectState) else ProjectState(**state)
        before = ps.model_copy(deep=True)
        ps.mark_agent_start(self.name)
        t0 = time.perf_counter()

        db_tasks = ps.tasks.by_agent("database")
        if not db_tasks:
            logger.info("[DBAgent] No database tasks — nothing to do.")
            return self._compute_diff(before, ps)

        logger.info("[DBAgent] Resolving stages for %d database tasks …", len(db_tasks))
        stages = resolve_db_stages(db_tasks)
        logger.info("[DBAgent] Scheduled into %d priority-ordered stage(s).", len(stages))

        state_lock = Lock()

        for idx, stage in enumerate(stages):
            if idx > 0:
                logger.info("[DBAgent] Waiting 2 s between stages (rate limit) …")
                time.sleep(2)

            # Split into trivial batches + non-trivial singles
            trivial     = [t for t in stage if is_trivial_file(t, ps.code_manifest)]
            non_trivial = [t for t in stage if not is_trivial_file(t, ps.code_manifest)]
            batches = [trivial[i:i + TRIVIAL_BATCH_SIZE] for i in range(0, len(trivial), TRIVIAL_BATCH_SIZE)]

            logger.info(
                "[DBAgent] Stage %d/%d — %d trivial (→ %d batch call(s)) + %d non-trivial",
                idx + 1, len(stages), len(trivial), len(batches), len(non_trivial),
            )

            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
                futures: list[Future] = []
                for batch in batches:
                    for t in batch:
                        t.status = "in_progress"
                    futures.append(pool.submit(self._process_batch, batch, ps, state_lock))
                for task in non_trivial:
                    task.status = "in_progress"
                    futures.append(pool.submit(self._process_task, task, ps, state_lock))

                for future in futures:
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error("[DBAgent] Unhandled task exception: %s", exc)
                        with state_lock:
                            ps.log_error(self.name, f"Task exception: {exc}")

            ps.metrics.total_llm_calls += len(batches) + len(non_trivial)

        duration = time.perf_counter() - t0
        ps.metrics.agent_durations[self.name] = round(duration, 3)

        unwritten = [m for m in ps.code_manifest.unwritten() if m.agent == "database"]
        logger.info(
            "[DBAgent] Done in %.2fs. Written: %d  Unwritten: %d.",
            duration,
            len(db_tasks) - len(unwritten),
            len(unwritten),
        )
        return self._compute_diff(before, ps)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Smoke-tests  (python agents/db.py)                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)-8s %(name)s — %(message)s")

    print("\n" + "=" * 60)
    print("DatabaseAgent — unit tests")
    print("=" * 60)

    # ── 1. Stage resolver ──────────────────────────────────────────────────────
    tasks = [
        Task(id="db-001", agent="database", title="Users table",    description="", depends_on=[],          priority=1),
        Task(id="db-002", agent="database", title="Roles table",    description="", depends_on=[],          priority=1),
        Task(id="db-003", agent="database", title="Profiles table", description="", depends_on=["db-001"],  priority=2),
        Task(id="db-004", agent="database", title="Sessions table", description="", depends_on=["db-001"],  priority=2),
        Task(id="db-005", agent="database", title="Audit table",    description="", depends_on=["db-003"],  priority=3),
    ]

    stages = resolve_db_stages(tasks)
    # Stage 0: db-001, db-002 (priority=1, no deps)
    # Stage 1: db-003, db-004 (priority=2, deps satisfied)
    # Stage 2: db-005 (priority=3)
    assert len(stages) == 3, f"Expected 3 stages, got {len(stages)}: {[[t.id for t in s] for s in stages]}"
    assert {t.id for t in stages[0]} == {"db-001", "db-002"}, f"Stage 0 wrong: {[t.id for t in stages[0]]}"
    assert {t.id for t in stages[1]} == {"db-003", "db-004"}, f"Stage 1 wrong: {[t.id for t in stages[1]]}"
    assert {t.id for t in stages[2]} == {"db-005"},           f"Stage 2 wrong: {[t.id for t in stages[2]]}"
    print("[PASS] resolve_db_stages — 3 stages, correct membership")

    # ── 2. System prompt contains tech stack ───────────────────────────────────
    sp = _build_system_prompt("MySQL with SQLAlchemy")
    assert "MySQL with SQLAlchemy" in sp
    print("[PASS] _build_system_prompt — tech stack injected")

    # ── 3. Mock integration run ────────────────────────────────────────────────
    from providers.llm.base import BaseLLMProvider, LLMResponse
    from core.state import ProjectState, TaskList, FileManifest, CodeManifest, ArchData

    MOCK_MANIFEST = {
        "table_name": "users",
        "columns": [
            {"name": "id",         "type": "UUID",         "nullable": False, "primary_key": True,  "foreign_key": None},
            {"name": "email",      "type": "VARCHAR(255)", "nullable": False, "primary_key": False, "foreign_key": None},
            {"name": "created_at", "type": "TIMESTAMP",    "nullable": False, "primary_key": False, "foreign_key": None},
            {"name": "updated_at", "type": "TIMESTAMP",    "nullable": False, "primary_key": False, "foreign_key": None},
        ],
        "indexes": [],
        "exports": ["users"],
        "suggested_additions": [],
    }

    MOCK_SQL = (
        "CREATE TABLE IF NOT EXISTS users (\n"
        "    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),\n"
        "    email VARCHAR(255) NOT NULL,\n"
        "    created_at TIMESTAMP NOT NULL DEFAULT NOW(),\n"
        "    updated_at TIMESTAMP NOT NULL DEFAULT NOW()\n"
        ");"
    )

    class MockLLM(BaseLLMProvider):
        @property
        def provider_name(self) -> str:
            return "mock"

        def complete(self, system_prompt, user_message, model, max_tokens=8192, temperature=0.2):
            assert "PostgreSQL" in system_prompt, "Tech stack not in system prompt"
            payload = {
                "migration_code": MOCK_SQL,
                "manifest": MOCK_MANIFEST,
                "functions": [],
            }
            return LLMResponse(
                text=f"```json\n{json.dumps([payload])}\n```",
                input_tokens=50,
                output_tokens=100,
                model="mock",
                provider="mock",
            )

    state = ProjectState(project_id="db-test", project_name="DB Test")
    state.architecture = ArchData(
        tech_stack={"database": "PostgreSQL with SQLAlchemy (Alembic)"},
        database_schema_overview="users(id UUID PK, email VARCHAR, created_at, updated_at)",
        data_flow="Users table is queried by email on every login.",
    )
    state.tasks = TaskList(tasks=[
        Task(id="db-001", agent="database", title="Users table", description="Create users table", depends_on=[], priority=1),
    ])
    fm = FileManifest(
        agent="database",
        task_id="db-001",
        filepath="database/migrations/001_create_users.sql",
        purpose="Initial users table migration",
        exports=["users"],
    )
    state.code_manifest = CodeManifest(files={fm.filepath: fm})

    agent = DatabaseAgent(provider=MockLLM())
    agent.run(state)

    assert state.code_manifest.files[fm.filepath].written, "File not marked written"
    assert fm.filepath in state.generated_code.database, "Code not in generated_code.database"
    assert "CREATE TABLE" in state.generated_code.database[fm.filepath]
    written_fm = state.code_manifest.files[fm.filepath]
    assert len(written_fm.classes) == 1, "Expected table class in manifest classes"
    assert written_fm.classes[0].name == "users", "Table name mismatch"
    assert len(written_fm.classes[0].arguments) == 4, "Columns count mismatch"
    assert not state.errors, f"Unexpected errors: {state.errors}"
    print("[PASS] DatabaseAgent mock run — migration written, manifest in state classes, no errors")

    # Cleanup
    import shutil
    test_out = os.path.join(_root_dir, "output", "database")
    if os.path.exists(test_out):
        shutil.rmtree(test_out, ignore_errors=True)

    print("\n" + "=" * 60)
    print("All DatabaseAgent smoke-tests PASSED")
    print("=" * 60 + "\n")
