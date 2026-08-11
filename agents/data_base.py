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
    from .base import BaseAgent, agent_refuse_corrupt, filter_manifest_by_dependencies, is_trivial_file, TRIVIAL_BATCH_SIZE, MAX_TREE_ENTRIES, MAX_MANIFEST_ENTRIES, build_project_file_tree, build_rework_instructions, build_rework_prompt, build_rework_batch_prompt
except ImportError:
    # Fallback for direct execution
    from base import BaseAgent, agent_refuse_corrupt, filter_manifest_by_dependencies, is_trivial_file, TRIVIAL_BATCH_SIZE, MAX_TREE_ENTRIES, MAX_MANIFEST_ENTRIES, build_project_file_tree, build_rework_instructions, build_rework_prompt, build_rework_batch_prompt
from agents.utils import safe_write, parse_metadata, resolve_stages

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

def _resolve_db_tech(arch: Any) -> str:
    """Resolve DB tooling string deterministically from the tech stack (no LLM override)."""
    from core.state import resolve_db_tooling
    tech = (arch.tech_stack or {}).get("database", "") if arch and arch.tech_stack else ""
    mapping = resolve_db_tooling(tech)
    base = tech or "SQLite"
    migrations = mapping.get("migrations", "None")
    return f"{base} ({mapping.get('orm', 'SQLAlchemy')} / migrations: {migrations})"


def _orm_constraint_block(arch: Any) -> str:
    """Instruction forcing a single ORM paradigm — matching the resolved DB mapping."""
    from core.state import resolve_db_tooling
    tech = (arch.tech_stack or {}).get("database", "") if arch and arch.tech_stack else ""
    mapping = resolve_db_tooling(tech)
    orm = mapping.get("orm", "SQLAlchemy")
    return (
        f"## ORM / Migration Strategy: {orm} using {mapping.get('migrations', 'None')}\n"
        f"IMPORTANT: Use {orm} as the single data-access layer. "
        "Do NOT use Prisma, Drizzle, TypeORM, raw DDL in parallel, or any other ORM. "
        "All models/migrations must use the same paradigm as the backend models."
    )


def _build_system_prompt(db_tech: str, orm_block: str = "") -> str:
    return f"""\
You are an expert database engineer specialising in {db_tech}.
Generate a complete, runnable database migration file AND a structured manifest block.

{orm_block}

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
    """Groups DB tasks into execution stages based on priority then depends_on."""
    return resolve_stages(tasks, use_priority=True)


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _fmt_manifest_context(manifest: CodeManifest, skip_path: str, task: Task | None = None) -> str:
    dep_files = filter_manifest_by_dependencies(manifest, skip_path, task)
    entries = []
    for path, fm in dep_files.items():
        status = "written" if fm.written else "pending"
        exports = ", ".join(fm.exports) or "(none)"
        entries.append(f"- `{path}` ({status})\n  Purpose: {fm.purpose}\n  Exports: {exports}")
    if len(entries) > MAX_MANIFEST_ENTRIES:
        omitted = len(entries) - MAX_MANIFEST_ENTRIES
        entries = entries[:MAX_MANIFEST_ENTRIES]
        entries.append(f"- … and {omitted} more dependency files omitted")
    return "\n\n".join(entries) or "No relevant dependency files in manifest."


def _build_file_prompt(
    filepath: str,
    task: Task,
    manifest: CodeManifest,
    arch: Any,
    db_tech: str,
    orm_block: str = "",
    rework_instructions: str = "",
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
    if len(written_tables) > MAX_MANIFEST_ENTRIES:
        omitted = len(written_tables) - MAX_MANIFEST_ENTRIES
        written_tables = written_tables[:MAX_MANIFEST_ENTRIES]
        written_tables.append(f"- … and {omitted} more written DB files omitted")
    written_str = "\n".join(written_tables) or "None yet."

    manifest_ctx = _fmt_manifest_context(manifest, filepath, task)

    depends_str = ", ".join(task.depends_on) if task.depends_on else "(none)"

    tree_str = build_project_file_tree(
        manifest, highlight_path=filepath, task=task, agent="database"
    )

    return f"""\
## Migration File to Generate
- **File Path** : `{filepath}`
- **Purpose**   : {purpose}
- **Exports**   : {exports}
- **Depends On**: {depends_str}
- **Priority**  : {task.priority}

{rework_instructions.strip()}

## Tech Stack
{tech_table}

{orm_block}

## Project File Tree (capped)
**IMPORT RULE**: Import internal modules relative to the layer root (e.g. `from db.session import ...`), NEVER with the `output.backend.` / `output.` workspace prefix.
{tree_str}

## Architecture — Database Schema Overview
{getattr(arch, 'database_schema_overview', 'No schema overview provided.')}

## Already Written Database Files (available for FK references)
{written_str}

## Project Code Manifest (Dependencies)
{manifest_ctx}

---
Generate the complete migration for `{filepath}`.
Follow ALL rules in the system prompt.
Return a single JSON object with keys: migration_code, manifest, functions.
```json
"""



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
            db_tech = _resolve_db_tech(arch)
            orm_block = _orm_constraint_block(arch)
            sys_prompt  = _build_system_prompt(db_tech, orm_block)
            user_prompt = self._build_batch_prompt(tasks, manifest, arch, db_tech)

        filepaths = [manifest.by_task(t.id).filepath for t in tasks if manifest.by_task(t.id)]
        logger.info("[DBAgent] Batch trivial call: %s", filepaths)

        try:
            raw = self._call_with_system(sys_prompt, user_prompt, agent_name="db-batch", state=state, lock=lock)
        except Exception as exc:
            logger.error("[DBAgent] Batch LLM call failed: %s", exc)
            with lock:
                state.log_error(self.name, f"Batch LLM call failed: {exc}")
            return

        try:
            items = self._parse_batch_response(raw)
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
            classes, functions = parse_metadata(item)
            try:
                written_code = safe_write(fm.filepath, code, on_corrupt=agent_refuse_corrupt(state, lock))
            except Exception as exc:
                logger.error("[DBAgent] Batch disk write failed for %s: %s", fm.filepath, exc)
                written_code = None
            if written_code is None:
                logger.warning("[DBAgent] Batch: corrupted content refused for %s", fm.filepath)
                continue
            code = written_code
            with lock:
                code_bucket = getattr(state.generated_code, self.code_bucket)
                code_bucket[fm.filepath] = code
                state.code_manifest.mark_written(fm.filepath, classes, functions)
                state.completed_task_ids.add(task.id)
            logger.info("[DBAgent] ✓ batch trivial: %s", fm.filepath)

    def _process_rework_batch(self, tasks: list[Task], state: ProjectState, lock: Lock) -> None:
        """Single LLM call for a batch of trivial rework DB files (lean targeted prompt)."""
        with lock:
            manifest = state.code_manifest
            arch = state.architecture
            db_tech = _resolve_db_tech(arch)
            orm_block = _orm_constraint_block(arch)
            sys_prompt = _build_system_prompt(db_tech, orm_block)

            entries = []
            valid_tasks = []
            for t in tasks:
                fm = manifest.by_task(t.id)
                if not fm:
                    continue
                code_bucket = getattr(state.generated_code, self.code_bucket, {}) or {}
                current_code = code_bucket.get(fm.filepath, "")
                rework = build_rework_instructions(state.integration, fm.filepath)
                if rework and current_code:
                    entries.append((fm.filepath, current_code, rework))
                    valid_tasks.append(t)
            user_prompt = build_rework_batch_prompt(entries, output_key="migration_code")

        filepaths = [fp for fp, _, _ in entries]
        logger.info("[DBAgent] Batch rework call: %s", filepaths)

        try:
            raw = self._call_with_system(sys_prompt, user_prompt, agent_name="db-rework-batch", state=state, lock=lock)
        except Exception as exc:
            logger.error("[DBAgent] Rework batch LLM call failed: %s", exc)
            with lock:
                state.log_error(self.name, f"Rework batch LLM call failed: {exc}")
            return

        try:
            items = self._parse_batch_response(raw)
        except Exception as exc:
            logger.error("[DBAgent] Rework batch JSON parse failed: %s — raw: %s", exc, raw[:300])
            with lock:
                state.log_error(self.name, f"Rework batch JSON parse failed: {exc}")
            return

        for item, task in zip(items, valid_tasks):
            with lock:
                fm = manifest.by_task(task.id)
            if not fm:
                continue
            code = str(item.get("migration_code", "")).strip()
            if not code:
                logger.warning("[DBAgent] Rework batch: empty migration_code for %s", fm.filepath)
                continue
            try:
                written_code = safe_write(fm.filepath, code, on_corrupt=agent_refuse_corrupt(state, lock))
            except Exception as exc:
                logger.error("[DBAgent] Rework batch disk write failed for %s: %s", fm.filepath, exc)
                written_code = None
            if written_code is None:
                logger.warning("[DBAgent] Rework batch: corrupted content refused for %s", fm.filepath)
                continue
            code = written_code
            with lock:
                code_bucket = getattr(state.generated_code, self.code_bucket)
                code_bucket[fm.filepath] = code
                # Rework responses return only code — preserve existing manifest metadata.
                state.completed_task_ids.add(task.id)
            logger.info("[DBAgent] ✓ rework batch: %s", fm.filepath)

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
            arch = state.architecture
            db_tech = _resolve_db_tech(arch)
            orm_block = _orm_constraint_block(arch)
            sys_prompt = _build_system_prompt(db_tech, orm_block)
            rework = build_rework_instructions(state.integration, filepath)
            code_bucket = getattr(state.generated_code, self.code_bucket, {}) or {}
            current_code = code_bucket.get(filepath, "")
            if rework and current_code:
                # Lean REWORK path: send only the conflict notes + current file content.
                user_prompt = build_rework_prompt(filepath, current_code, rework, output_key="migration_code")
            else:
                user_prompt = _build_file_prompt(
                    filepath, task, state.code_manifest, arch, db_tech, orm_block, rework_instructions=rework
                )

        # LLM call (outside lock — blocking I/O)
        try:
            raw = self._call_with_system(sys_prompt, user_prompt, agent_name=f"db-{task.id}", state=state, lock=lock)
        except Exception as exc:
            logger.error("[DBAgent] LLM call failed for %s: %s", filepath, exc)
            with lock:
                state.log_error(self.name, f"LLM call failed for {filepath}: {exc}")
            return

        # Parse JSON response
        try:
            from agents.schemas.database import DatabaseSchema
            data = self.extract_json_and_validate(raw, DatabaseSchema, state=state, lock=lock)
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

        # Rework responses return only code — preserve existing manifest metadata.
        if rework:
            classes, functions = None, None
        else:
            classes, functions = parse_metadata(data)

        # Write migration file to disk (safe-write guard refuses corrupted content)
        try:
            written_code = safe_write(filepath, migration_code, on_corrupt=agent_refuse_corrupt(state, lock))
        except Exception as exc:
            logger.error("[DBAgent] Disk write failed for %s: %s", filepath, exc)
            with lock:
                state.log_error(self.name, f"Disk write failed for {filepath}: {exc}")
            written_code = None
        if written_code is None:
            logger.error("[DBAgent] Corrupted content refused for %s — file not written", filepath)
            return
        migration_code = written_code

        # Thread-safe state update
        with lock:
            state.generated_code.database[filepath] = migration_code
            if not rework:
                state.code_manifest.mark_written(filepath, classes=classes, functions=functions)
            state.completed_task_ids.add(task.id)

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
            # Trivial rework targets are batched via the lean rework prompt;
            # non-trivial rework files get the single lean prompt in _process_task.
            def _needs_rework(t: Task) -> bool:
                fm = ps.code_manifest.by_task(t.id)
                return build_rework_instructions(ps.integration, getattr(fm, "filepath", ""))

            trivial_batch = [t for t in stage if is_trivial_file(t, ps.code_manifest) and not _needs_rework(t)]
            rework_batch  = [t for t in stage if is_trivial_file(t, ps.code_manifest) and _needs_rework(t)]
            non_trivial   = [t for t in stage if not is_trivial_file(t, ps.code_manifest)]
            batches    = [trivial_batch[i:i + TRIVIAL_BATCH_SIZE] for i in range(0, len(trivial_batch), TRIVIAL_BATCH_SIZE)]
            rework_bs  = [rework_batch[i:i + TRIVIAL_BATCH_SIZE] for i in range(0, len(rework_batch), TRIVIAL_BATCH_SIZE)]

            logger.info(
                "[DBAgent] Stage %d/%d — %d trivial (→ %d batch call(s)) + %d trivial-rework (→ %d batch call(s)) + %d non-trivial",
                idx + 1, len(stages), len(trivial_batch), len(batches), len(rework_batch), len(rework_bs), len(non_trivial),
            )

            with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CALLS) as pool:
                futures: list[Future] = []
                for batch in batches:
                    for t in batch:
                        t.status = "in_progress"
                    futures.append(pool.submit(self._process_batch, batch, ps, state_lock))
                for rb in rework_bs:
                    for t in rb:
                        t.status = "in_progress"
                    futures.append(pool.submit(self._process_rework_batch, rb, ps, state_lock))
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
